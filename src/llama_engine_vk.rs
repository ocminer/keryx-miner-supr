//! In-process llama.cpp engine, AMD/Vulkan flavor (dynamically loaded sidecar) — zero-dup:
//! llama.cpp hosts the SINGLE resident model copy on the inference GPU, the PoM walk gathers
//! straight over its VRAM tensors (wrapper exports `keryx_llama_pom_mine`/`_fetch` — byte-exact
//! per the full-model spike + the startup byte gate below), and OPoI text generation runs
//! in-process. Absent/failed .so = try the Vulkan llama-server GPU route; candle-CPU is a
//! deprecated explicit emergency fallback only, and every card keeps its own OpenCL blob.
//!
//! Consensus safety: this module only changes WHO HOSTS the model bytes on the inference card and
//! WHO GENERATES the user-facing OPoI text. The walk math is byte-identical (pom_walk_vk.comp),
//! and [`pom_byte_gate`] cross-checks the engine's gather against the host possession index at
//! every startup — any mismatch refuses zero-dup and the OpenCL blob path takes over.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Mutex, OnceLock};

use libloading::Library;

type AbiFn = unsafe extern "C" fn() -> c_int;
type LoadFn = unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void;
type Load2Fn = unsafe extern "C" fn(*const c_char, c_int, *const c_int, c_int, c_int) -> *mut c_void;
type NCtxFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type SysPromptFn = unsafe extern "C" fn(*const c_char);
type CtxInfoFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type FreeFn = unsafe extern "C" fn(*mut c_void);
type GenFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_char, c_int) -> c_int;
type ReadyFn = unsafe extern "C" fn(*mut c_void) -> bool;
type U64Fn = unsafe extern "C" fn(*mut c_void) -> u64;
type FetchFn = unsafe extern "C" fn(*mut c_void, u64, *mut u8) -> bool;
type MineFn = unsafe extern "C" fn(*mut c_void, *const u64, *const u64, *const u64, u64, u64, u32, u32, u32) -> i64;
type PciFn = unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32, *mut u32, *mut u32) -> bool;
type PickFn = unsafe extern "C" fn() -> c_int;
type PickAbiFn = unsafe extern "C" fn() -> c_int;
type DevicePciFn = unsafe extern "C" fn(c_int, *mut u32, *mut u32, *mut u32, *mut u32) -> bool;

unsafe fn install_native_log_bridge(lib: &'static Library) -> bool {
    let get = sym::<crate::native_llama_log::LogGet>(lib, "keryx_llama_log_get_v1")
        .or_else(|| sym::<crate::native_llama_log::LogGet>(lib, "llama_log_get"));
    let set = sym::<crate::native_llama_log::LogSet>(lib, "keryx_llama_log_set_v1")
        .or_else(|| sym::<crate::native_llama_log::LogSet>(lib, "llama_log_set"));
    let (Some(get), Some(set)) = (get, set) else {
        return false;
    };
    crate::native_llama_log::install(get, set)
}

const ABI: c_int = 2;
const VK_ABI: c_int = 6; // 5->6: keryx_llama_pom_mine is the PoM v4 walk (tiles, h10 era flag)

/// Walk dispatches in flight outside the engine mutex (see `pom_mine_v4`): `unload` must not free
/// the model while one is running.
static WALK_INFLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Engine {
    model: *mut c_void,
    free: FreeFn,
    generate: GenFn,
    pom_ready: ReadyFn,
    pom_n_chunks: U64Fn,
    pom_supl_bytes: U64Fn,
    pom_fetch: FetchFn,
    pom_mine: MineFn,
    pom_pci: PciFn,
    walk_dot: Option<ReadyFn>,
    walk_cm: Option<ReadyFn>,
    batch_max: Option<unsafe extern "C" fn(*mut c_void) -> u32>,
    gpu: usize,
    gguf: String,
    /// Context the model was loaded with, and whether the library has the prompt guard — see
    /// `llama_engine::unguarded_prompt_cap` (same llama.cpp b10015 batch assert).
    n_ctx: c_int,
    prompt_guard: bool,
}
// The wrapper serializes generation + walk dispatches internally (gen_lock / walk_lock).
unsafe impl Send for Engine {}

fn engine() -> &'static Mutex<Option<Engine>> {
    static E: OnceLock<Mutex<Option<Engine>>> = OnceLock::new();
    E.get_or_init(|| Mutex::new(None))
}

/// A failed model load must not leave the mining worker selected during Vulkan preflight idle.
/// Success disarms the guard; unload later releases the active in-process ownership explicitly.
struct InprocessDedicationAttempt {
    armed: bool,
}

impl Drop for InprocessDedicationAttempt {
    fn drop(&mut self) {
        if self.armed {
            crate::pom_opencl::release_inprocess_vulkan_dedication();
        }
    }
}

/// Same SIGILL gate as llama_engine::cpu_has_baked_simd — the Vulkan sidecar is built with
/// the same GGML_NATIVE=OFF flags that bake in AVX/AVX2/FMA/F16C/BMI2 with no runtime dispatch.
#[cfg(target_arch = "x86_64")]
fn cpu_has_baked_simd() -> bool {
    is_x86_feature_detected!("avx")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("bmi2")
}
#[cfg(not(target_arch = "x86_64"))]
fn cpu_has_baked_simd() -> bool {
    true
}

/// `KERYX_LLAMA_VK_SO=<path>` wins; otherwise use the platform-native Vulkan sidecar next to our
/// executable. CPUs without the baked-in SIMD set get the `-noavx` build if present, else no
/// engine (graceful fallback) rather than a SIGILL inside the AVX build.
fn so_path() -> Option<std::path::PathBuf> {
    let simd_ok = cpu_has_baked_simd();
    if let Ok(p) = std::env::var("KERYX_LLAMA_VK_SO") {
        let pb = std::path::PathBuf::from(p);
        if pb.exists() {
            if !simd_ok && !pb.to_string_lossy().contains("noavx") {
                log::warn!(
                    "llama-vk engine: this CPU lacks AVX2/FMA/F16C — if KERYX_LLAMA_VK_SO is an AVX build the process will crash with SIGILL. Honoring the explicit override anyway."
                );
            }
            return Some(pb);
        }
        log::warn!("llama-vk engine: KERYX_LLAMA_VK_SO points at a missing file — ignoring.");
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let want_noavx = !simd_ok || std::env::var("KERYX_LLAMA_FORCE_NOAVX").map_or(false, |v| v == "1");
    if want_noavx {
        #[cfg(target_os = "windows")]
        let p = dir.join("keryx-llama-vk-noavx.dll");
        #[cfg(not(target_os = "windows"))]
        let p = dir.join("libkeryx-llama-vk-noavx.so");
        if p.exists() {
            log::info!("llama-vk engine: using baseline (no-AVX) build {}.", p.display());
            return Some(p);
        }
        log::warn!(
            "llama-vk engine: this CPU lacks the AVX2/FMA/F16C/BMI2 set baked into the standard sidecar and no baseline sidecar exists at {} — NOT loading it (would SIGILL). Remaining GPU routes will be tried.",
            p.display()
        );
        return None;
    }
    #[cfg(target_os = "windows")]
    let p = dir.join("keryx-llama-vk.dll");
    #[cfg(not(target_os = "windows"))]
    let p = dir.join("libkeryx-llama-vk.so");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

unsafe fn sym<T: Copy>(lib: &'static Library, name: &str) -> Option<T> {
    let c = CString::new(name).ok()?;
    lib.get::<T>(c.as_bytes_with_nul()).ok().map(|symbol| *symbol)
}

/// Open the Vulkan sidecar once and retain it for the process lifetime. `libloading` maps this to
/// dlopen on Unix and LoadLibrary on Windows. Besides keeping every copied FFI pointer valid, the
/// process-lifetime handle protects the downstream callback retained by the native log bridge.
/// Failed loads are deliberately not cached: a sidecar copied into place later can still recover.
fn sidecar_lib(so: &std::path::Path) -> Option<&'static Library> {
    static LIB: OnceLock<Library> = OnceLock::new();
    if let Some(lib) = LIB.get() {
        return Some(lib);
    }
    let loaded = match unsafe { Library::new(so) } {
        Ok(lib) => lib,
        Err(error) => {
            log::warn!(
                "llama-vk engine: failed to load {}: {} — will retry when inference is requested.",
                so.display(),
                error
            );
            return None;
        }
    };
    if LIB.set(loaded).is_ok() {
        log::info!("llama-vk engine: loaded {} (in-process Vulkan engine).", so.display());
    }
    LIB.get()
}

// ── Discrete-GPU selection (issue #18) ──────────────────────────────────────────────────────
// The model must NOT land on an integrated GPU (UMA system RAM → the miner exits/restart-loops).
// A device index is only meaningful WITHIN the Vulkan instance that produced it: ggml enumerates
// + filters devices in its own instance, and any index from a SEPARATE enumeration (a private
// libvulkan instance, or a GGML_VK_VISIBLE_DEVICES value derived from one) can map to a different
// device on an iGPU rig — silently loading onto the iGPU or tripping a GGML_ASSERT. So the
// discrete GPU is chosen INSIDE the engine .so, against ggml's OWN device list (see
// keryx_vk_pick_discrete_device); Rust only asks the .so for the answer.

/// The discrete-GPU `main_gpu` index in GGML's own device list, via the bundled engine .so.
/// Valid for any ggml in this process tree — the in-process engine AND the bundled llama-server
/// subprocess share the same ggml build. With a published worker PCI allowlist, `None` is a
/// fail-closed result (no match or an old, untrusted picker); without one, it retains the legacy
/// meaning of no sidecar/no discrete GPU.
pub fn auto_device_allowlist_active() -> bool {
    std::env::var_os(crate::pom_opencl::LLAMA_VK_AUTO_PCI_ALLOWLIST_ENV).is_some()
}

pub fn pick_discrete_ggml_device() -> Option<i32> {
    let so = so_path()?;
    unsafe {
        let lib = sidecar_lib(&so)?;
        if !install_native_log_bridge(lib) && crate::tui_active() {
            return None;
        }
        (|| -> Option<i32> {
            let pick: PickFn = sym(lib, "keryx_vk_pick_discrete_device")?;
            if auto_device_allowlist_active() {
                // A pre-allowlist sidecar still exports the old picker but considers every Vulkan
                // device. Trust it only when the optional capability symbol proves it applies the
                // selected-worker PCI filter; otherwise auto-placement must fail closed.
                let picker_abi: PickAbiFn = sym(lib, "keryx_vk_picker_abi")?;
                if picker_abi() < 1 {
                    return None;
                }
            }
            let d = pick();
            if d >= 0 {
                return Some(d);
            }
            // The strict allowlist pick found nothing. Distinguish two causes:
            //   * PCI IS readable but no Vulkan device matched a mining GPU → a genuine wrong-card
            //     hazard on a multi-GPU/mixed-vendor rig; stay fail-closed (return None below).
            //   * The mining GPUs expose NO verifiable PCI identity anywhere (the AMD OpenCL platform
            //     reported none → the published allowlist is empty, or no Vulkan device answers
            //     VK_EXT_pci_bus_info) → the allowlist is unenforceable either way, so refusing to
            //     mine is strictly worse than v0.13.0's largest-discrete auto-pick. Fall back to it.
            // Both new probes are optional: on a pre-fallback sidecar `sym` returns None, `pci_verifiable`
            // defaults to true, and behaviour is exactly the old strict fail-closed.
            if auto_device_allowlist_active() {
                let allowlist_empty = std::env::var(crate::pom_opencl::LLAMA_VK_AUTO_PCI_ALLOWLIST_ENV)
                    .map(|v| v.trim().is_empty())
                    .unwrap_or(false);
                let pci_verifiable = sym::<PickFn>(lib, "keryx_vk_any_discrete_pci_verifiable")
                    .map(|f| f() > 0)
                    .unwrap_or(true);
                if allowlist_empty || !pci_verifiable {
                    if let Some(any) = sym::<PickFn>(lib, "keryx_vk_pick_discrete_device_unfiltered") {
                        let fallback = any();
                        if fallback >= 0 {
                            log::warn!(
                                "llama-vk engine: the selected mining GPUs expose no verifiable PCI \
                                 identity ({}), so the OpenCL worker allowlist cannot be enforced — \
                                 falling back to the largest discrete GPU (ggml device {}) for \
                                 inference, restoring pre-0.13.1 behaviour. Pin a specific card with \
                                 KERYX_LLAMA_VK_DEVICE to override.",
                                if allowlist_empty {
                                    "the AMD OpenCL platform reported none"
                                } else {
                                    "no Vulkan device answers VK_EXT_pci_bus_info"
                                },
                                fallback
                            );
                            return Some(fallback);
                        }
                    }
                }
            }
            None
        })()
    }
}

/// Map a ggml `main_gpu` index to its full PCI identity using the same filtered Vulkan device list
/// which interprets that index. This optional sidecar export is required to safely resolve an
/// explicit llama-server card against the selected OpenCL workers.
pub fn ggml_device_pci(device: i32) -> Option<(u32, u32, u32, u32)> {
    if device < 0 {
        return None;
    }
    let so = so_path()?;
    unsafe {
        let lib = sidecar_lib(&so)?;
        if !install_native_log_bridge(lib) && crate::tui_active() {
            return None;
        }
        (|| -> Option<(u32, u32, u32, u32)> {
            let picker_abi: PickAbiFn = sym(lib, "keryx_vk_picker_abi")?;
            if picker_abi() < 1 {
                return None;
            }
            let pci: DevicePciFn = sym(lib, "keryx_vk_device_pci")?;
            let (mut domain, mut bus, mut dev, mut function) = (0, 0, 0, 0);
            pci(device, &mut domain, &mut bus, &mut dev, &mut function).then_some((domain, bus, dev, function))
        })()
    }
}

/// Load the .so + the model once (idempotent, blocking — a model load takes seconds). Returns
/// whether the engine is active for `gguf`. Safe to call from multiple threads.
pub fn ensure_loaded(gguf: &str, _gpu: usize) -> bool {
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(e) = g.as_ref() {
        return e.gguf == gguf;
    }
    // Resolve the exact ggml index and its OpenCL PCI peer before opening/allocating the model.
    // When model + walk cannot coexist, preflight also removes an already-resident OpenCL blob
    // while the caller's inference drain is held. Reserving only after load is too late: Vulkan
    // would otherwise OOM on a selected non-largest (or tie-broken) target which was still mining.
    let explicit_gpu = std::env::var("KERYX_LLAMA_VK_DEVICE")
        .ok()
        .and_then(|s| s.trim().parse::<c_int>().ok())
        .filter(|gpu| *gpu >= 0);
    let main_gpu: c_int = match explicit_gpu {
        Some(gpu) => gpu,
        None if auto_device_allowlist_active() => match pick_discrete_ggml_device() {
            Some(gpu) => gpu,
            None => {
                crate::pom_opencl::release_provisional_vulkan_dedication();
                log::warn!(
                    "llama-vk engine: no trusted ggml device matches the selected OpenCL \
                     worker PCI allowlist — automatic GPU inference placement refused. \
                     Rebuild libkeryx-llama-vk.so or set KERYX_LLAMA_VK_DEVICE explicitly."
                );
                return false;
            }
        },
        None => -1,
    };
    let mut dedication_attempt = InprocessDedicationAttempt { armed: false };
    if main_gpu >= 0 {
        if !crate::pom_opencl::prepare_inprocess_vulkan_device(main_gpu) {
            return false;
        }
        dedication_attempt.armed = true;
    }
    let Some(so) = so_path() else { return false };
    unsafe {
        let Some(lib) = sidecar_lib(&so) else {
            return false;
        };
        if !install_native_log_bridge(lib) && crate::tui_active() {
            log::warn!(
                "llama-vk engine: sidecar cannot coordinate native logging with the interactive \
                 dashboard — skipping the in-process route; use --no-tui for this legacy sidecar."
            );
            return false;
        }
        let (
            Some(abi),
            Some(vk_abi),
            Some(load),
            Some(free_fn),
            Some(gen),
            Some(ready),
            Some(nch),
            Some(supl),
            Some(fetch),
            Some(mine),
            Some(pci),
        ) = (
            sym::<AbiFn>(lib, "keryx_llama_abi"),
            sym::<AbiFn>(lib, "keryx_llama_vk_abi"),
            sym::<LoadFn>(lib, "keryx_llama_load"),
            sym::<FreeFn>(lib, "keryx_llama_free"),
            sym::<GenFn>(lib, "keryx_llama_generate"),
            sym::<ReadyFn>(lib, "keryx_llama_pom_ready"),
            sym::<U64Fn>(lib, "keryx_llama_pom_n_chunks"),
            sym::<U64Fn>(lib, "keryx_llama_pom_supl_bytes"),
            sym::<FetchFn>(lib, "keryx_llama_pom_fetch"),
            sym::<MineFn>(lib, "keryx_llama_pom_mine"),
            sym::<PciFn>(lib, "keryx_llama_pom_pci"),
        )
        else {
            log::warn!("llama-vk engine: {} is missing symbols — fallbacks stay active.", so.display());
            return false;
        };
        if abi() != ABI || vk_abi() != VK_ABI {
            log::warn!(
                "llama-vk engine: {} ABI {}/{} != expected {}/{} — fallbacks stay active.",
                so.display(),
                abi(),
                vk_abi(),
                ABI,
                VK_ABI
            );
            return false;
        }
        let cg = match CString::new(gguf) {
            Ok(c) => c,
            Err(_) => return false,
        };
        // Device pin (issue #18): the model must NOT land on an integrated GPU (UMA system RAM →
        // the miner exits/restart-loops). The .so resolves `main_gpu < 0` to a discrete GPU
        // against GGML'S OWN device list — the only index space that is reliably valid, since a
        // separate-instance enumeration (or GGML_VK_VISIBLE_DEVICES computed from one) orders
        // devices differently on iGPU rigs and mislocates or asserts. The helper selects the
        // largest selected-worker card by the exact PCI allowlist, matching the max-VRAM planner on
        // heterogeneous/subset rigs. We do NOT set GGML_VK_VISIBLE_DEVICES here; the helper returns
        // a `main_gpu` in ggml's own index space. `KERYX_LLAMA_VK_DEVICE` explicitly overrides the
        // selected-worker constraint and may intentionally name an inference-only card.
        log::info!(
            "llama-vk engine: loading {gguf} (ggml GPU {}) via {} (in-process, zero-dup)…",
            if main_gpu < 0 { "auto-discrete".to_string() } else { main_gpu.to_string() },
            so.display()
        );
        let configured_ctx: Option<c_int> = std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse().ok());
        let mut n_ctx: c_int = configured_ctx.unwrap_or(4096);
        // Context ladder + walk reserve (upstream d5ccad3, same as the CUDA engine): the library
        // keeps a context only if the reserve stays free for the zero-dup walk it builds next.
        let mut model: *mut c_void = std::ptr::null_mut();
        if configured_ctx.is_none() {
            if let Some(load2) = sym::<Load2Fn>(lib, "keryx_llama_load2") {
                let (floor, cap) = crate::models::spec_for_gguf(gguf)
                    .map_or((4096, 4096), |m| (m.ctx_floor as c_int, m.ctx_cap as c_int));
                let ladder = crate::models::context_ladder(cap, floor);
                let reserve: c_int = std::env::var("KERYX_LLAMA_WALK_RESERVE_MB")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(crate::models::DEFAULT_WALK_RESERVE_MIB);
                model = load2(cg.as_ptr(), main_gpu, ladder.as_ptr(), ladder.len() as c_int, reserve);
                if !model.is_null() {
                    n_ctx = sym::<NCtxFn>(lib, "keryx_llama_n_ctx").map(|f| f(model)).filter(|&n| n > 0).unwrap_or(floor);
                }
            }
        }
        if model.is_null() {
            model = load(cg.as_ptr(), main_gpu, n_ctx);
        }
        if !model.is_null() {
            if let (Some(set_sys), Ok(sys)) = (
                sym::<SysPromptFn>(lib, "keryx_llama_set_system_prompt"),
                CString::new(crate::models::SYSTEM_PROMPT_NEXT),
            ) {
                set_sys(sys.as_ptr());
            }
            if let Some(info) = sym::<CtxInfoFn>(lib, "keryx_llama_context_info") {
                log::info!("llama-vk engine: context {}", CStr::from_ptr(info(model)).to_string_lossy());
            }
        }
        if model.is_null() {
            log::warn!("llama-vk engine: model load failed (VRAM? Vulkan ICD?) — fallbacks stay active.");
            return false;
        }
        let walk = ready(model);
        log::info!(
            "llama-vk engine: ✓ active — llama.cpp hosts the model + serves OPoI inference{}.",
            if walk { " + hosts the zero-dup PoM walk" } else { " (walk unavailable — OpenCL blob stays)" }
        );
        *g = Some(Engine {
            model,
            free: free_fn,
            generate: gen,
            pom_ready: ready,
            pom_n_chunks: nch,
            pom_supl_bytes: supl,
            pom_fetch: fetch,
            pom_mine: mine,
            pom_pci: pci,
            walk_dot: sym::<ReadyFn>(lib, "keryx_llama_pom_walk_dot"),
            walk_cm: sym::<ReadyFn>(lib, "keryx_llama_pom_walk_cm"),
            batch_max: sym::<unsafe extern "C" fn(*mut c_void) -> u32>(lib, "keryx_llama_pom_batch_max"),
            gpu: main_gpu.max(0) as usize, // -1 = auto; the actual device is read via pom_pci
            gguf: gguf.to_string(),
            n_ctx,
            prompt_guard: sym::<AbiFn>(lib, "keryx_llama_prompt_guard").map_or(false, |f| f() == 1),
        });
        dedication_attempt.armed = false;
        true
    }
}

pub fn available() -> bool {
    match engine().lock() {
        Ok(g) => g.is_some(),
        Err(_) => false,
    }
}

/// The Vulkan engine is a singleton. Never answer a request with a merely "available" but
/// different resident model.
pub fn active_for(gguf: &str) -> bool {
    match engine().lock() {
        Ok(g) => g.as_ref().map(|e| e.gguf.as_str()) == Some(gguf),
        Err(_) => false,
    }
}

/// Same guard as the CUDA engine: an unguarded library aborts the process (llama.cpp
/// `GGML_ASSERT(n_tokens_all <= n_batch)`) on a prompt longer than its batch.
fn prompt_allowed(e: &Engine, prompt: &str) -> bool {
    let cap = crate::llama_engine_prompt_cap(e.n_ctx);
    if !e.prompt_guard && prompt.len() > cap {
        log::warn!(
            "llama-vk engine: refusing a {}-byte prompt — this library has no prompt guard (cap {} bytes)",
            prompt.len(),
            cap
        );
        return false;
    }
    true
}

/// Generate up to `max_tokens` of OPoI text in-process.
pub fn generate(prompt: &str, max_tokens: usize) -> Option<String> {
    let g = engine().lock().ok()?;
    let e = g.as_ref()?;
    if !prompt_allowed(e, prompt) {
        return None;
    }
    let cp = CString::new(prompt).ok()?;
    let cap: usize = 65536;
    let mut out = vec![0u8; cap];
    let n = unsafe {
        (e.generate)(e.model, cp.as_ptr(), max_tokens as c_int, out.as_mut_ptr() as *mut c_char, cap as c_int)
    };
    if n < 0 {
        return None;
    }
    out.truncate(n as usize);
    Some(utf8_complete_prefix(out))
}

/// Generation stops at a token budget, and a multi-byte UTF-8 character can be split across the
/// last token boundary (byte-level BPE pieces). Rejecting the whole answer for that (the old
/// `String::from_utf8(..)`) turned a valid reply into "generate failed" — on AMD every pool probe
/// whose answer was cut mid-character withdrew the model. Keep the longest valid prefix instead;
/// any invalid byte inside the text (never expected) is replaced, not fatal. Mirrors the CUDA
/// engine's helper.
fn utf8_complete_prefix(buf: Vec<u8>) -> String {
    match String::from_utf8(buf) {
        Ok(s) => s,
        Err(e) => {
            let incomplete_tail = e.utf8_error().error_len().is_none();
            let valid = e.utf8_error().valid_up_to();
            log::info!(
                "llama-vk engine: answer ended {} at byte {valid} — keeping the valid prefix",
                if incomplete_tail { "inside a multi-byte character" } else { "with an invalid byte" }
            );
            let mut bytes = e.into_bytes();
            if incomplete_tail {
                bytes.truncate(valid);
            }
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }
}

pub fn generate_for(gguf: &str, prompt: &str, max_tokens: usize) -> Result<String, crate::slm::GenerateError> {
    use crate::slm::GenerateError;
    let g = engine().lock().map_err(|_| {
        log::error!("llama-vk engine: engine mutex poisoned — generation refused");
        GenerateError::Failed
    })?;
    let Some(e) = g.as_ref() else {
        log::warn!("llama-vk engine: no engine loaded — generation refused");
        return Err(GenerateError::Failed);
    };
    if e.gguf != gguf {
        log::warn!("llama-vk engine: resident model {} != requested {gguf} — generation refused", e.gguf);
        return Err(GenerateError::Failed);
    }
    if prompt.len() > crate::slm::MAX_INFERENCE_PROMPT_BYTES || !prompt_allowed(e, prompt) {
        return Err(GenerateError::PromptTooLong);
    }
    let cp = CString::new(prompt).map_err(|_| {
        log::warn!("llama-vk engine: prompt contains a NUL byte — generation refused");
        GenerateError::Failed
    })?;
    let cap: usize = 65536;
    let mut out = vec![0u8; cap];
    let n = unsafe {
        (e.generate)(e.model, cp.as_ptr(), max_tokens as c_int, out.as_mut_ptr() as *mut c_char, cap as c_int)
    };
    if n == -2 {
        log::warn!(
            "llama-vk engine: a {}-byte prompt does not fit the model context ({} tokens) — refused",
            prompt.len(),
            e.n_ctx
        );
        return Err(GenerateError::PromptTooLong);
    }
    if n < 0 {
        log::warn!("llama-vk engine: generate returned {n} for a {}-byte prompt (max_tokens {max_tokens})", prompt.len());
        return Err(GenerateError::Failed);
    }
    out.truncate(n as usize);
    Ok(utf8_complete_prefix(out))
}

/// The engine hosts a gather-ready walk (BDA available, table built).
pub fn pom_ready() -> bool {
    match engine().lock() {
        Ok(g) => g.as_ref().map_or(false, |e| unsafe { (e.pom_ready)(e.model) }),
        Err(_) => false,
    }
}

/// PCI location (domain, bus, device, function) of the engine's GPU — the OpenCL driver matches
/// this against CL_DEVICE_TOPOLOGY_AMD to find the cl_device_id whose card must NOT get its own
/// blob (its walk routes here instead).
pub fn pom_pci() -> Option<(u32, u32, u32, u32)> {
    let g = engine().lock().ok()?;
    let e = g.as_ref()?;
    let (mut d, mut b, mut dv, mut f) = (0u32, 0u32, 0u32, 0u32);
    if unsafe { (e.pom_pci)(e.model, &mut d, &mut b, &mut dv, &mut f) } {
        Some((d, b, dv, f))
    } else {
        None
    }
}

/// STARTUP BYTE GATE (consensus safety — mirrors the CUDA driver's): the pool does not
/// deep-verify every share, so a wrong gather would mine garbage silently. Checks the engine's
/// canonical chunk count equals the host possession index's, then reads evenly-spaced chunks
/// through the walk's exact gather path and byte-compares them against the index (GGUF pread).
/// Any mismatch → refuse zero-dup (caller keeps the OpenCL blob).
pub fn pom_byte_gate(index: &crate::pom::WeightIndex) -> bool {
    let g = match engine().lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    let Some(e) = g.as_ref() else { return false };
    if !unsafe { (e.pom_ready)(e.model) } {
        return false;
    }
    let n = unsafe { (e.pom_n_chunks)(e.model) };
    if n != index.n_chunks {
        log::warn!(
            "llama-vk engine: byte gate FAILED — engine N={n} != index N={} — keeping the OpenCL blob.",
            index.n_chunks
        );
        return false;
    }
    let samples = 128u64;
    for k in 0..=samples {
        let off = if k == samples { n - 1 } else { k * (n / (samples + 1)) };
        let mut got = [0u8; 32];
        if !unsafe { (e.pom_fetch)(e.model, off, got.as_mut_ptr()) } {
            log::warn!("llama-vk engine: byte gate fetch failed at chunk {off} — keeping the OpenCL blob.");
            // A failed fetch DISPATCH means the device may be hung (RDNA1 field logs: this was a
            // hard GPU hang, not a soft error). Mark it so unload() never calls vkDeviceWaitIdle/
            // free on a possibly-dead device — that blocks forever and wedges the whole rig.
            DEVICE_SUSPECT.store(true, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        if got != index.read_chunk_bytes(off) {
            log::warn!(
                "llama-vk engine: byte gate FAILED at chunk {off} — engine bytes differ from the GGUF; keeping the OpenCL blob."
            );
            return false;
        }
    }
    log::info!(
        "llama-vk engine: byte gate PASSED ({} sampled chunks match the possession index; supplement {} MiB) — zero-dup walk enabled.",
        samples + 1,
        unsafe { (e.pom_supl_bytes)(e.model) } / (1024 * 1024)
    );
    true
}

/// Grind one batch of `batch` nonces from `nonce_base` over the engine-resident weights, in
/// TDR-safe sub-dispatches (ascending, early exit on the first winning sub-batch — identical
/// semantics to `PomMiner::mine`). `p`/`t` are the pph words (era-salted by the caller) and the
/// LE target words. Returns the lowest winning nonce, or None.
/// True once the engine was unloaded to give its VRAM to the possession blob. The llama-server
/// GPU fallback consults this: it would pin to the SAME discrete card (pick_discrete_ggml_device)
/// and re-occupy the VRAM the unload just freed — recreating the small-card squeeze. With the flag
/// set (and no explicit KERYX_LLAMA_VK_DEVICE), no GPU inference route is advertised: mining wins.
pub fn evicted_for_vram() -> bool {
    EVICTED.load(std::sync::atomic::Ordering::Relaxed)
}
static EVICTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Set when a byte-gate FETCH dispatch fails — the engine's device may be hung; skip free/wait-idle.
static DEVICE_SUSPECT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Pre-flight verdict: GPU inference is UNFIT on this rig (no card can hold model + blob) —
/// set before any engine/server load so neither ever starts. Same flag the try_start guard reads.
pub fn mark_gpu_inference_unfit() {
    EVICTED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Unload the in-process engine and free all its Vulkan allocations. This is a generic model-swap
/// primitive and deliberately does not mark GPU inference unfit: normal generation failure may
/// still fall back to llama-server. The OpenCL OOM recovery path explicitly calls
/// `mark_gpu_inference_unfit` after a successful unload because only that path has chosen mining
/// VRAM over auto-placed inference. Returns whether an engine was present.
pub fn unload() -> bool {
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(e) = g.take() {
        if DEVICE_SUSPECT.load(std::sync::atomic::Ordering::Relaxed) {
            EVICTED.store(true, std::sync::atomic::Ordering::Relaxed);
            // The device may be hung (failed gate-fetch dispatch): freeing would vkDeviceWaitIdle
            // on it and block forever. Leak the engine's objects instead — the handle is dropped,
            // the flag stays, and the process keeps running (that card is lost until reboot anyway).
            log::warn!(
                "llama-vk engine: NOT freeing engine VRAM — the device is suspect (failed gate \
                 dispatch); freeing would block on a hung GPU. Objects are leaked deliberately."
            );
            return true;
        }
        // A zero-dup walk dispatch may still be running outside the mutex (it polls the inference
        // pause between sub-dispatches, so this is bounded by one sub-dispatch). Never free under it.
        let wait = std::time::Instant::now();
        while WALK_INFLIGHT.load(std::sync::atomic::Ordering::Acquire) != 0 {
            if wait.elapsed() > std::time::Duration::from_secs(60) {
                log::error!(
                    "llama-vk engine: a zero-dup walk dispatch did not finish within 60 s — NOT freeing \
                     the engine (objects leaked deliberately rather than freed under a running dispatch)."
                );
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        unsafe { (e.free)(e.model) };
        crate::pom_opencl::release_inprocess_vulkan_dedication();
        log::warn!(
            "llama-vk engine: unloaded — Vulkan model VRAM released; configured GPU inference \
             fallbacks remain eligible."
        );
        true
    } else {
        false
    }
}

/// Which kernel the engine's walk runs: "matrix cores", "packed int8 dot" or "scalar int8 dot".
pub fn pom_walk_kernel() -> &'static str {
    let Ok(g) = engine().lock() else { return "unknown" };
    let Some(e) = g.as_ref() else { return "unknown" };
    if e.walk_cm.map_or(false, |f| unsafe { f(e.model) }) {
        "matrix cores"
    } else if e.walk_dot.map_or(false, |f| unsafe { f(e.model) }) {
        "packed int8 dot"
    } else {
        "scalar int8 dot"
    }
}

struct InflightGuard;
impl Drop for InflightGuard {
    fn drop(&mut self) {
        WALK_INFLIGHT.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Grind `batch` PoM v4 nonces from `nonce_base` over the engine-resident weights, in TDR-safe
/// sub-dispatches (ascending, early exit on the first winning sub-batch — identical semantics to
/// `PomMiner::mine_v4`). `p`/`s` are the POW/SEED pph words for the era (host-salted exactly like
/// the OpenCL path), `t` the LE target words, `h10` selects the one-way H10/H14 seed. `stop()` is
/// polled between sub-dispatches (the inference pause); the grind then returns the work completed
/// so far. The engine mutex is NOT held across the dispatches, so an OPoI generation never waits
/// for a whole batch: the wrapper serializes walk/generate on the device itself.
/// Ok((lowest winning nonce, hashes_done)).
pub fn pom_mine_v4(
    p: [u64; 4],
    s: [u64; 4],
    time: u64,
    t: [u64; 4],
    nonce_base: u64,
    batch: u64,
    h10: bool,
    stop: &dyn Fn() -> bool,
) -> Result<(Option<u64>, u64), String> {
    let (model, mine, cap) = {
        let g = engine().lock().map_err(|_| "engine mutex poisoned".to_string())?;
        let e = g.as_ref().ok_or_else(|| "no engine loaded".to_string())?;
        // Counted under the lock: `unload` takes the same lock, then waits for zero.
        WALK_INFLIGHT.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let cap = e.batch_max.map_or(0, |f| unsafe { f(e.model) }) as u64;
        (e.model, e.pom_mine, cap)
    };
    let _inflight = InflightGuard;
    // Nonces per engine call: the library pipelines up to four two-phase sub-batches per call
    // (chase of the next overlapping the walk of the current), so a call covers 4 x 32768 on
    // Linux (4 x 8192 on Windows, TDR-safe); its reported capacity bounds it. The inference pause
    // is polled between calls. KERYX_VK_SUB_DISPATCH overrides.
    let default_sub: u64 = if cfg!(target_os = "windows") { 32768 } else { 131072 };
    let mut sub_max = std::env::var("KERYX_VK_SUB_DISPATCH")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n >= 8)
        .unwrap_or(default_sub);
    if cap != 0 {
        sub_max = sub_max.min(cap);
    }
    let mut done: u64 = 0;
    while done < batch {
        if done > 0 && stop() {
            break;
        }
        let sub = (batch - done).min(sub_max) as u32;
        let base = nonce_base.wrapping_add(done);
        let r = unsafe {
            (mine)(model, p.as_ptr(), s.as_ptr(), t.as_ptr(), time, base, sub, crate::pom::POM_WALK_STEPS, h10 as u32)
        };
        match r {
            -2 => return Err("walk dispatch failed".to_string()),
            -1 => {}
            off => return Ok((Some(base.wrapping_add(off as u64)), done + sub as u64)),
        }
        done += sub as u64;
    }
    Ok((None, done))
}

/// GPU byte-exactness gate for the zero-dup v4 walk (ignored: needs an AMD/Vulkan GPU, the
/// sidecar and a real GGUF). Loads the engine on `KERYX_TEST_GGUF` (copy the model to a private
/// directory first — the possession tree is cached next to it), builds the host possession index
/// from the same file, runs the startup byte gate plus a random-chunk sweep, then checks that the
/// engine's lowest-winning nonce equals the CPU reference argmin for all three seed eras (pre-H10
/// v4 fold, H10 PowHash, H14 tagged PowHash) and that a target below the minimum yields no winner.
/// Run once per kernel variant: KERYX_VK_WALK_DOT=1 (packed int8 dot) and =0 (scalar).
///   KERYX_LLAMA_VK_SO=<libkeryx-llama-vk.so> KERYX_TEST_GGUF=<model.gguf> [KERYX_LLAMA_VK_DEVICE=n]
///   cargo test --release --features pom-opencl --lib vk_zero_dup_v4 -- --ignored --nocapture
#[cfg(all(test, any(unix, windows)))]
mod zero_dup_v4_tests {
    use crate::pom::{self, WeightIndex};
    use crate::pom_v4::{
        fold64, v4_first_offset, v4_initial_state, v4_next_offset, v4_state_root, v4_transition_into, POM_V4_K,
        POM_V4_TILE_BYTES, POM_V4_TILE_CHUNKS,
    };

    /// CPU re-walk of one seed to its `final_state` (pom_v4 primitives — the consensus mirror).
    fn cpu_v4_final(index: &WeightIndex, seed: u64) -> u64 {
        let n_tiles = index.n_chunks / POM_V4_TILE_CHUNKS;
        let mut state = v4_initial_state(seed);
        let mut next = vec![0u8; state.len()];
        let mut tile = vec![0u8; POM_V4_TILE_BYTES];
        let mut off = v4_first_offset(seed, n_tiles);
        for step in 1..=POM_V4_K as u64 {
            index.read_chunks_into(off * POM_V4_TILE_CHUNKS, &mut tile);
            if step < POM_V4_K as u64 {
                off = v4_next_offset(seed, step, tile[..32].try_into().unwrap(), n_tiles);
            }
            v4_transition_into(&mut next, &state, &tile, step as u32);
            std::mem::swap(&mut state, &mut next);
        }
        fold64(&v4_state_root(&state))
    }

    fn le_le(a: &[u8; 32], b: &[u8; 32]) -> bool {
        for k in (0..4).rev() {
            let wa = u64::from_le_bytes(a[k * 8..k * 8 + 8].try_into().unwrap());
            let wb = u64::from_le_bytes(b[k * 8..k * 8 + 8].try_into().unwrap());
            if wa != wb {
                return wa < wb;
            }
        }
        true
    }

    fn words(b: &[u8; 32]) -> [u64; 4] {
        std::array::from_fn(|i| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()))
    }

    #[test]
    #[ignore]
    fn vk_zero_dup_v4_matches_cpu_reference() {
        let path = std::env::var("KERYX_TEST_GGUF").expect("set KERYX_TEST_GGUF");
        let gpu: usize = std::env::var("KERYX_LLAMA_VK_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        assert!(super::ensure_loaded(&path, gpu), "engine load failed");
        assert!(super::pom_ready(), "engine walk not ready");
        eprintln!("engine: walk kernel = {}", super::pom_walk_kernel());

        let gen = |label: &str| {
            let t0 = std::time::Instant::now();
            let r = super::generate_for(&path, "Reply with the single word pong.", 32);
            eprintln!("generate ({label}): {:?} in {:.2}s", r.as_ref().map(|s| s.trim().chars().take(40).collect::<String>()), t0.elapsed().as_secs_f64());
            assert!(r.is_ok(), "generation failed ({label})");
        };
        gen("before walk");
        let index = WeightIndex::build_from_gguf(&path, [0x5au8; 32]).expect("WeightIndex::build_from_gguf");
        eprintln!("index: N = {} chunks, {} tiles", index.n_chunks, index.n_chunks / POM_V4_TILE_CHUNKS);
        assert!(super::pom_byte_gate(&index), "byte gate failed");

        // Random-chunk sweep through the exact gather path (beyond the gate's evenly spaced samples).
        {
            let g = super::engine().lock().unwrap();
            let e = g.as_ref().unwrap();
            let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
            for _ in 0..8192 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let off = x % index.n_chunks;
                let mut got = [0u8; 32];
                assert!(unsafe { (e.pom_fetch)(e.model, off, got.as_mut_ptr()) }, "fetch dispatch failed");
                assert_eq!(got, index.read_chunk_bytes(off), "chunk {off} differs from the GGUF");
            }
        }
        eprintln!("random sweep: 8192 chunks byte-identical");

        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(23).wrapping_add(5));
        let time: u64 = 0x0123_4567_89AB_CDEF;
        let nn: u64 = std::env::var("KERYX_TEST_NONCES").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
        let p = pom::pph_words_for_era(&pph, true);
        let never = || false;
        for (h10, h14, name) in [(false, false, "pre-H10"), (true, false, "H10"), (true, true, "H14")] {
            let t0 = std::time::Instant::now();
            let mut best: Option<([u8; 32], u64)> = None;
            for nonce in 0..nn {
                let seed = if h14 {
                    pom::pom_block_seed_h14(&pph, time, nonce)
                } else if h10 {
                    pom::pom_block_seed_h10(&pph, time, nonce)
                } else {
                    pom::pom_block_seed_v4(&pph, time, nonce)
                };
                let pv = pom::pom_pow_value(cpu_v4_final(&index, seed), &pph, true);
                if best.map_or(true, |(b, _)| le_le(&pv, &b)) {
                    best = Some((pv, nonce));
                }
            }
            let (target, w_cpu) = best.unwrap();
            eprintln!("[{name}] cpu argmin nonce = {w_cpu} ({} nonces in {:.1}s)", nn, t0.elapsed().as_secs_f64());
            let s = if h14 {
                pom::pph_words(&pom::seed_h14_pph(&pph))
            } else if h10 {
                pom::pph_words(&pph)
            } else {
                pom::pph_words_v4(&pph)
            };
            let t = words(&target);
            let t0 = std::time::Instant::now();
            let (w_gpu, done) = super::pom_mine_v4(p, s, time, t, 0, nn, h10, &never).expect("engine grind");
            eprintln!("[{name}] gpu winner = {w_gpu:?} ({done} nonces counted, {:.2}s)", t0.elapsed().as_secs_f64());
            assert_eq!(w_gpu, Some(w_cpu), "[{name}] zero-dup v4 winner mismatch");
            // Below the minimum: no nonce may qualify (false-positive check).
            let mut below = target;
            let mut k = 0;
            while k < 32 {
                if below[k] > 0 {
                    below[k] -= 1;
                    break;
                }
                below[k] = 0xff;
                k += 1;
            }
            assert!(k < 32, "target underflow");
            let (w_none, _) = super::pom_mine_v4(p, s, time, words(&below), 0, nn, h10, &never).expect("engine grind");
            assert_eq!(w_none, None, "[{name}] winner below the minimum pow value");
            eprintln!("[{name}] OK");
            gen("after walk");
        }
        // A longer grind (sub-dispatch loop) followed by generation — the production interleaving.
        let s = pom::pph_words(&pom::seed_h14_pph(&pph));
        let (_, done) = super::pom_mine_v4(p, s, time, [0u64; 4], 0, 1 << 16, true, &never).expect("engine grind");
        eprintln!("full grind: {done} nonces");
        gen("after 64k grind");
    }
}
