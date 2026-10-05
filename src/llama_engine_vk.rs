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

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::{Mutex, OnceLock};

use libloading::Library;

type AbiFn = unsafe extern "C" fn() -> c_int;
type LoadFn = unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void;
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
const VK_ABI: c_int = 5; // bumped 4->5 at H5.1: keryx_llama_pom_mine gained the seed-words arg

/// Max nonces per engine dispatch (ascending sub-batches, early exit on the first winner =
/// identical lowest-nonce semantics). Linux has no desktop TDR watchdog, so 2^20 quarters the
/// submit+fence overhead and remains ~120 ms on an MI60-class card. Windows retains the OpenCL
/// driver's conservative 2^18 bound so a slow card cannot trip WDDM timeout detection.
#[cfg(not(target_os = "windows"))]
const SUB_DISPATCH_NONCES: u64 = 1 << 20;
#[cfg(target_os = "windows")]
const SUB_DISPATCH_NONCES: u64 = 1 << 18;

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
        let n_ctx: c_int = std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
        let model = load(cg.as_ptr(), main_gpu, n_ctx);
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
    String::from_utf8(out).ok()
}

pub fn generate_for(gguf: &str, prompt: &str, max_tokens: usize) -> Option<String> {
    let g = engine().lock().ok()?;
    let e = g.as_ref()?;
    if e.gguf != gguf {
        return None;
    }
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
    String::from_utf8(out).ok()
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

pub fn pom_mine(
    p: [u64; 4],
    s: [u64; 4],
    time: u64,
    t: [u64; 4],
    nonce_base: u64,
    batch: u64,
    walk_v2: bool,
) -> Option<u64> {
    let g = engine().lock().ok()?;
    let e = g.as_ref()?;
    let wv2: u32 = walk_v2 as u32; // H5 era flag -> shader push-constant (v1 fold vs mix64-chain)
    let mut done: u64 = 0;
    while done < batch {
        let sub = (batch - done).min(SUB_DISPATCH_NONCES) as u32;
        let base = nonce_base.wrapping_add(done);
        let r = unsafe {
            // p = POW words, s = SEED words (H5.1-salted at/after gate; == p pre-H5.1)
            (e.pom_mine)(e.model, p.as_ptr(), s.as_ptr(), t.as_ptr(), time, base, sub, crate::pom::POM_WALK_STEPS, wv2)
        };
        match r {
            -2 => {
                log::warn!("llama-vk engine: walk dispatch failed — no result for this batch.");
                return None;
            }
            -1 => {}
            off => return Some(base.wrapping_add(off as u64)),
        }
        done += sub as u64;
    }
    None
}
