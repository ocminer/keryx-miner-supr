//! In-process llama.cpp engine via a dlopen'd `libkeryx-llama.so` (Phase 2 candle-independence).
//!
//! When the .so sits next to the miner binary (or `KERYX_LLAMA_SO` points at it), it becomes the
//! DEFAULT engine for the primary model on the inference GPU: llama.cpp owns the single resident
//! VRAM copy, the PoM walk gathers straight over its tensor pointers (zero-dup — byte-identity
//! proven by tools/llama_zerodup_spike), and OPoI text generation runs in-process. Absent .so =
//! the legacy candle path remains only for architectures it can actually serve on a GPU.
//!
//! Consensus safety: this module only changes WHO HOSTS the model bytes and WHO GENERATES the
//! user-facing OPoI text. The walk kernel, the host possession index, proofs and `tag_fixed` are
//! untouched; `ensure_installed_inner`'s N-guard cross-checks the gather against the host index.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Arc, Mutex, OnceLock};

// Cross-platform dynamic loader: `libloading` maps to dlopen on unix and LoadLibrary on Windows, so
// the in-process llama.cpp engine works on every platform the miner ships on (previously this module
// used raw libc::dlopen and was stubbed out on Windows — the reason Windows NVIDIA rigs could not
// serve H4/H6 models and fell through to CPU). The library handle is opened ONCE into a process
// global that lives forever; the resolved fn pointers are copied out and stay valid for the run.
use libloading::Library;

type AbiFn = unsafe extern "C" fn() -> c_int;
type LoadFn = unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void;
type CountFn = unsafe extern "C" fn(*mut c_void) -> usize;
type InfoFn =
    unsafe extern "C" fn(*mut c_void, usize, *mut *const c_char, *mut *mut c_void, *mut usize, *mut c_int) -> bool;
type GenFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_char, c_int) -> c_int;
type FreeFn = unsafe extern "C" fn(*mut c_void);
// CUDA ordinal owning tensor i's bytes, or -1 (host/unified/unknown). Upstream aa29fd2 — optional
// symbol (absent on older .so builds → the ownership gate below degrades to a no-op).
type TensorDeviceFn = unsafe extern "C" fn(*mut c_void, usize) -> c_int;

/// Install a process-lifetime bridge over llama.cpp/ggml's native stderr logger. Older shipped
/// sidecars already export these stable llama API symbols, so the host fix also protects them
/// without requiring an immediate sidecar rebuild.
unsafe fn install_native_log_bridge(lib: &Library) -> bool {
    let get = lib
        .get::<crate::native_llama_log::LogGet>(b"keryx_llama_log_get_v1\0")
        .or_else(|_| lib.get::<crate::native_llama_log::LogGet>(b"llama_log_get\0"))
        .ok()
        .map(|symbol| *symbol);
    let set = lib
        .get::<crate::native_llama_log::LogSet>(b"keryx_llama_log_set_v1\0")
        .or_else(|_| lib.get::<crate::native_llama_log::LogSet>(b"llama_log_set\0"))
        .ok()
        .map(|symbol| *symbol);
    let (Some(get), Some(set)) = (get, set) else {
        return false;
    };
    crate::native_llama_log::install(get, set)
}

const ABI: c_int = 2;

struct Engine {
    model: *mut c_void,
    count: CountFn,
    info: InfoFn,
    generate: GenFn,
    free: FreeFn,
    /// Optional: absent on older `libkeryx-llama.so` builds (looked up soft, gate no-ops if None).
    tensor_device: Option<TensorDeviceFn>,
    gguf: String,
}
// The wrapper serializes generation internally; tensor info is read-only after load.
unsafe impl Send for Engine {}

/// One per-GPU engine slot. Its own `Mutex` serializes generation ON THAT CARD while letting a
/// DIFFERENT card's slot run concurrently (the whole point of the multi-GPU inference pool): the
/// caller clones the `Arc<EngineSlot>` out of `POOL` (a brief lock), releases `POOL`, then locks
/// only this slot for the duration of the generate. `Engine: Send` ⇒ `Mutex<Option<Engine>>: Sync`
/// ⇒ `Arc<EngineSlot>: Send + Sync`, so a slot can be handed to `spawn_blocking`.
struct EngineSlot {
    inner: Mutex<Option<Engine>>,
}

/// The per-GPU engine pool. `POOL[gpu]` (a slot Arc) hosts at most one llama.cpp model on CUDA
/// ordinal `gpu`. Each GPU is an independent slot so concurrent inference runs on different cards
/// and a model swap / free on one card never touches another card's resident tensors (the zero-dup
/// PoM walk gathers over those — see `ensure_loaded_on`'s safety note). Replaces the former single
/// `Mutex<Option<Engine>>` singleton; the legacy `ensure_loaded/generate/tensors/…` calls are thin
/// shims over this map so existing call sites and the walk sharing keep compiling.
fn pool() -> &'static Mutex<HashMap<usize, Arc<EngineSlot>>> {
    static P: OnceLock<Mutex<HashMap<usize, Arc<EngineSlot>>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get (or create) the slot for `gpu`. Only the map lock is held here — never a slot's `inner`
/// lock — so this never blocks on an in-flight generation.
fn slot_for(gpu: usize) -> Arc<EngineSlot> {
    let mut g = match pool().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    g.entry(gpu).or_insert_with(|| Arc::new(EngineSlot { inner: Mutex::new(None) })).clone()
}

/// Snapshot of the currently-populated GPU ordinals (a slot may exist but be empty; only ordinals
/// whose slot holds a live engine are returned).
fn loaded_gpus() -> Vec<usize> {
    let slots: Vec<(usize, Arc<EngineSlot>)> = match pool().lock() {
        Ok(g) => g.iter().map(|(k, v)| (*k, v.clone())).collect(),
        Err(_) => return Vec::new(),
    };
    slots.into_iter().filter(|(_, s)| s.inner.lock().map(|g| g.is_some()).unwrap_or(false)).map(|(k, _)| k).collect()
}

/// Actually dlopen the .so, resolve symbols, load the model for (gguf, gpu). Returns the Engine or
/// None (caller falls back). Factored out of `ensure_loaded_on` so the load happens while the
/// target slot's lock is held (per-card serialization) without duplicating the FFI plumbing.
/// `--force-model` is honored VERBATIM: no VRAM pre-check, the load is attempted regardless of
/// what the card reports (set once from main when the flag is present). The env escape
/// KERYX_LLAMA_VRAM_CHECK=0 has the same effect for diagnosis.
static FORCE_LOAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_vram_check_bypass(on: bool) {
    FORCE_LOAD.store(on, std::sync::atomic::Ordering::Relaxed);
}
fn vram_check_bypassed() -> bool {
    FORCE_LOAD.load(std::sync::atomic::Ordering::Relaxed)
}

fn load_engine(gguf: &str, gpu: usize) -> Option<Engine> {
    let lib = engine_lib()?;
    unsafe {
        let (Some(abi), Some(load), Some(count), Some(info), Some(gen), Some(free)) = (
            sym::<AbiFn>(lib, b"keryx_llama_abi\0"),
            sym::<LoadFn>(lib, b"keryx_llama_load\0"),
            sym::<CountFn>(lib, b"keryx_llama_tensor_count\0"),
            sym::<InfoFn>(lib, b"keryx_llama_tensor_info\0"),
            sym::<GenFn>(lib, b"keryx_llama_generate\0"),
            sym::<FreeFn>(lib, b"keryx_llama_free\0"),
        ) else {
            log::warn!("llama engine: shared library is missing required symbols — inference engine unavailable.");
            return None;
        };
        let got = abi();
        if got != ABI {
            log::warn!("llama engine: shared library ABI {} != expected {} — inference engine unavailable.", got, ABI);
            return None;
        }
        // Optional (upstream aa29fd2): older builds lack it — the ownership gate then no-ops.
        let tensor_device = sym::<TensorDeviceFn>(lib, b"keryx_llama_tensor_device\0");
        // --force-model: NO safety check at all — the operator's contract is "load it verbatim,
        // regardless of what the card says". The pre-check below exists to protect AUTO
        // selection from the poisonous failed-load path; a forced rig opts out of that
        // protection knowingly (set from main when --force-model is present).
        // VRAM PRE-CHECK: never ASK llama to load a model that cannot fit this card's free VRAM.
        // A failed multi-GiB load is not harmless: ggml's per-device CUDA pool teardown on the
        // failure path races the next engine load/generate on the same card, and that survivor
        // dies with a sticky illegal memory access at its first sync (reproduced: the lineup's
        // Gemma-4-12B forced onto an 8 GB card fails its 9.3 GiB weight alloc twice, then the
        // Qwen engine that DOES fit aborts in its own self-test — the field "fresh-rig staging
        // crash" / CUBLAS_STATUS_NOT_INITIALIZED restart loop). Skipping up front produces the
        // exact outcome the failed load produced (model unavailable on this card, self-test
        // fails cleanly, tier withdrawn) minus the crash trigger — and is instant instead of an
        // ~9 GiB doomed allocation. The margin covers the minimum (1024-token) context + compute
        // buffers. KERYX_LLAMA_VRAM_CHECK=0 disables (diagnostic only).
        #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
        if !vram_check_bypassed() && std::env::var("KERYX_LLAMA_VRAM_CHECK").ok().as_deref() != Some("0") {
            if let Ok(md) = std::fs::metadata(gguf) {
                // Margin guards WEIGHT-level allocation failures only (the reproduced poison
                // class: 9.3 GiB weights forced onto an 8 GB card). Context-size failures are
                // handled by llama's own benign 4096->1024 retry below, so a borderline fit
                // (12B model on a 10 GB card, ~200-400 MB slack) must be ATTEMPTED, not
                // refused — a 1200 MiB margin here broke exactly that working config.
                let need_mib = md.len() / (1024 * 1024) + 128;
                if let Some((_, free_mib, total_mib)) =
                    crate::pom_gpu::query_all_gpus_free_vram().into_iter().find(|(o, _, _)| *o as usize == gpu)
                {
                    if free_mib < need_mib {
                        log::warn!(
                            "llama engine: '{}' needs ~{} MiB on GPU {} but only {} of {} MiB are free — \
                             not attempting the load (a failed multi-GiB load can poison the card for the \
                             next engine load; this model is simply unavailable on this card).",
                            gguf,
                            need_mib,
                            gpu,
                            free_mib,
                            total_mib
                        );
                        return None;
                    }
                }
            }
        }
        let cg = CString::new(gguf).ok()?;
        log::info!("llama engine: loading {} on GPU {} (in-process, zero-dup)…", gguf, gpu);
        let configured_ctx = std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse::<c_int>().ok());
        let n_ctx: c_int = configured_ctx.unwrap_or(4096);
        let mut model = load(cg.as_ptr(), gpu as c_int, n_ctx);
        if model.is_null() && configured_ctx.is_none() && n_ctx > 1024 {
            // Context-alloc retry (upstream Keryx-Labs/keryx-miner 9e6dc8d4, adapted): a card too
            // tight for the default 4096-token context can still host the model with a smaller one.
            // Retry once at 1024 before giving up (skipped when the user pinned KERYX_LLAMA_CTX).
            log::warn!("llama engine: {}-token context did not fit on GPU {} — retrying with 1024 tokens.", n_ctx, gpu);
            model = load(cg.as_ptr(), gpu as c_int, 1024);
        }
        if model.is_null() {
            log::warn!("llama engine: model load failed on GPU {} (VRAM? arch? driver?) — OPoI inference unavailable on this card.", gpu);
            return None;
        }
        log::info!(
            "llama engine: ✓ active on GPU {} — llama.cpp hosts the model + serves OPoI inference in-process.",
            gpu
        );
        Some(Engine { model, count, info, generate: gen, free, tensor_device, gguf: gguf.to_string() })
    }
}

/// The stock `libkeryx-llama.so` bakes in AVX/AVX2/FMA/F16C/BMI2 with no runtime dispatch
/// (ggml's INS_ENB defaults under GGML_NATIVE=OFF) — on a pre-AVX CPU (mining-rig Celeron/
/// Pentium, pre-Sandy-Bridge Xeon) the first ggml CPU op dies with SIGILL before any of the
/// fallback guards below can engage. Gate on the exact feature set the build bakes in.
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

/// `KERYX_LLAMA_SO=<path>` wins; else the platform-native shared library next to our own
/// executable (`libkeryx-llama.dylib` on macOS, `libkeryx-llama.so` elsewhere). CPUs without
/// the baked-in SIMD set (and `KERYX_LLAMA_FORCE_NOAVX=1` for testing) get the `-noavx`
/// baseline build instead; absent that file the engine stays off rather than SIGILL.
fn so_path() -> Option<std::path::PathBuf> {
    let simd_ok = cpu_has_baked_simd();
    if let Ok(p) = std::env::var("KERYX_LLAMA_SO") {
        let pb = std::path::PathBuf::from(p);
        if pb.exists() {
            if !simd_ok && !pb.to_string_lossy().contains("noavx") {
                log::warn!(
                    "llama engine: this CPU lacks AVX2/FMA/F16C — if KERYX_LLAMA_SO is an AVX build the process will crash with SIGILL. Honoring the explicit override anyway."
                );
            }
            return Some(pb);
        }
        log::warn!("llama engine: KERYX_LLAMA_SO points at a missing file — ignoring.");
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let want_noavx = !simd_ok || std::env::var("KERYX_LLAMA_FORCE_NOAVX").map_or(false, |v| v == "1");
    if want_noavx {
        #[cfg(target_os = "windows")]
        let p = dir.join("keryx-llama-noavx.dll");
        #[cfg(not(target_os = "windows"))]
        let p = dir.join("libkeryx-llama-noavx.so");
        if p.exists() {
            log::info!(
                "llama engine: CPU lacks the AVX2/FMA/F16C/BMI2 set (or KERYX_LLAMA_FORCE_NOAVX) — \
                 selected the baseline no-AVX/no-BMI2 build: {}",
                p.display()
            );
            return Some(p);
        }
        log::warn!(
            "llama engine: this CPU lacks the AVX2/FMA/F16C/BMI2 set baked into libkeryx-llama.so and no {} exists in {} — NOT loading it (would SIGILL). FIX: copy libkeryx-llama-noavx.so from the release tarball into that exact directory (every release since 0.7.2 ships it next to the binary — partial/binary-only upgrades lose it), or set KERYX_LLAMA_SO=<absolute path to the -noavx .so>. No in-process GPU inference route is available meanwhile.",
            p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(), dir.display()
        );
        return None;
    }
    // macOS ships a .dylib (Mach-O). Every other unix (Linux/BSD) ships a .so (ELF). Probe the
    // native name first, and on macOS also fall back to .so — some HiveOS-adjacent tooling may
    // repackage the Linux .so alongside the macOS binary during cross-arch testing.
    #[cfg(target_os = "macos")]
    let candidates: [&str; 2] = ["libkeryx-llama.dylib", "libkeryx-llama.so"];
    #[cfg(target_os = "windows")]
    let candidates: [&str; 1] = ["keryx-llama.dll"];
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let candidates: [&str; 1] = ["libkeryx-llama.so"];
    for name in candidates {
        let p = dir.join(name);
        if p.exists() {
            log::info!(
                "llama engine: CPU has the AVX2/FMA/F16C/BMI2 set — selected the standard build: {}",
                p.display()
            );
            return Some(p);
        }
    }
    None
}

/// Open the llama.cpp shared library ONCE for the process and keep it resident forever. `libloading`
/// picks the right OS primitive (dlopen / LoadLibrary), so this works on Linux, macOS AND Windows.
/// The handle is never dropped (never dlclose/FreeLibrary'd), so fn pointers copied out via `sym`
/// stay valid for the whole run — matching the old leak-the-dlopen-handle behavior.
/// Preload the CUDA runtime libs that ship in the `lib/` folder NEXT TO `libkeryx-llama.so`, by
/// ABSOLUTE PATH into the GLOBAL symbol namespace, BEFORE the engine itself is dlopen'd.
///
/// Why this exists: the prebuilt `libkeryx-llama.so` has a DT_NEEDED on `libcudart.so.12`,
/// `libcublas.so.12`, etc. but carries NO rpath — so the dynamic loader finds those bundled libs
/// only when `LD_LIBRARY_PATH` happens to include the `lib/` dir. A user who runs the binary
/// directly (no wrapper exporting that variable) gets `libcudart.so.12: cannot open shared object
/// file`, the engine load fails, OPoI reports "no models ready", and the miner SUSPENDS ALL MINING
/// — zero shares found, zero accepted. Requiring the user to set an env var is not acceptable.
///
/// dlopen'ing each dep by absolute path with RTLD_GLOBAL publishes its soname into the process's
/// global namespace; the subsequent `Library::new(libkeryx-llama.so)` then satisfies its NEEDED
/// entries from those already-resident libs WITHOUT any filesystem search — no rpath, no env var.
/// Best-effort: a system-wide CUDA install (no bundled `lib/`) resolves the deps the normal way, so
/// a missing folder or file is skipped silently. Handles are leaked (never dlclose'd) so the
/// symbols stay live for the whole run.
#[cfg(target_os = "linux")]
fn preload_bundled_cuda_deps(so: &std::path::Path) {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
    let Some(lib_dir) = so.parent().map(|d| d.join("lib")) else {
        return;
    };
    if !lib_dir.is_dir() {
        return; // no bundled libs (e.g. a system-wide CUDA install) — nothing to preload
    }
    // Dependency order matters: load each lib AFTER everything it itself NEEDs, so its own
    // DT_NEEDED entries resolve against the already-global earlier ones. cudart is the leaf;
    // cublas needs cublasLt + cudart; curand needs cudart.
    const DEPS: [&str; 4] = ["libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcurand.so.10"];
    for name in DEPS {
        let p = lib_dir.join(name);
        if !p.exists() {
            continue;
        }
        // SAFETY: loading a trusted, self-shipped CUDA runtime lib by absolute path; none of its
        // init code runs Rust. RTLD_GLOBAL publishes its soname so the engine's NEEDED resolves here.
        match unsafe { UnixLibrary::open(Some(&p), RTLD_NOW | RTLD_GLOBAL) } {
            Ok(l) => {
                std::mem::forget(l); // keep resident for the whole process (never dlclose)
                log::debug!("llama engine: preloaded bundled {}", p.display());
            }
            Err(e) => {
                // Non-fatal: if this dep was truly required the engine load below surfaces the error.
                log::debug!("llama engine: could not preload {} ({e}) — continuing.", p.display());
            }
        }
    }
}
#[cfg(not(target_os = "linux"))]
fn preload_bundled_cuda_deps(_so: &std::path::Path) {}

fn engine_lib() -> Option<&'static Library> {
    static LIB: OnceLock<Library> = OnceLock::new();
    if let Some(lib) = LIB.get() {
        return Some(lib);
    }
    let so = so_path()?;
    // Make the bundled CUDA runtime (lib/ next to the .so) resolvable WITHOUT LD_LIBRARY_PATH.
    preload_bundled_cuda_deps(&so);
    // Do not cache a failed dlopen. Startup can probe before an automatic CUDA-runtime install has
    // completed; caching `None` made the newly installed dependency unusable until process restart.
    let loaded = match unsafe { Library::new(&so) } {
        Ok(l) => l,
        Err(e) => {
            log::warn!(
                "llama engine: failed to load {}: {} — will retry when inference is requested.",
                so.display(),
                e
            );
            return None;
        }
    };
    // Do this immediately after dlopen and before llama_backend_init/model load. If a foreign
    // sidecar lacks the stable logging API, never invoke it while an alternate screen is active;
    // the silent llama-server GPU fallback remains available instead.
    if !unsafe { install_native_log_bridge(&loaded) } && crate::tui_active() {
        log::warn!(
            "llama engine: sidecar cannot coordinate native logging with the interactive dashboard — \
             skipping the in-process route; use --no-tui for this legacy sidecar."
        );
        return None;
    }
    if LIB.set(loaded).is_ok() {
        log::info!("llama engine: loaded {} (in-process llama.cpp engine).", so.display());
    }
    LIB.get()
}

/// Resolve a nul-terminated symbol name to a fn pointer. `T` must be an `extern "C"` fn-pointer type.
/// The returned pointer is valid for the process lifetime because `lib` is the never-unloaded global
/// from `engine_lib()`.
unsafe fn sym<T: Copy>(lib: &'static Library, name: &[u8]) -> Option<T> {
    lib.get::<T>(name).ok().map(|s| *s)
}

/// Load the .so + the model for `gpu` (idempotent, blocking — a model load takes seconds) WITHOUT
/// evicting any OTHER GPU's engine. Returns whether the engine is active for `gguf` on `gpu`. Safe
/// to call from multiple threads and concurrently for different GPUs.
///
/// Per-GPU slot safety: because each GPU has its OWN slot, this can ONLY ever free-and-reload the
/// engine that lives on `gpu` itself (a same-card model swap — the caller reaches here from
/// `ensure_installed_inner` with THIS card's walk already uninstalled). A different card's resident
/// tensors are unreachable from here, so the former "different GPU must not steal the engine"
/// device-use-after-free hazard (upstream Keryx-Labs/keryx-miner@278098b) is now structurally
/// impossible rather than guarded by an early-return.
pub fn ensure_loaded_on(gguf: &str, gpu: usize) -> bool {
    let slot = slot_for(gpu);
    let mut g = match slot.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(e) = g.as_ref() {
        if e.gguf == gguf {
            return true; // already resident on this card (gpu matches by construction)
        }
        // Same-card model swap: free this card's old model before loading the new one.
        if let Some(e) = g.take() {
            unsafe { (e.free)(e.model) };
        }
    }
    match load_engine(gguf, gpu) {
        Some(e) => {
            *g = Some(e);
            true
        }
        None => false,
    }
}

/// Legacy shim: load `gguf` on `gpu`. Identical to `ensure_loaded_on` (kept so the walk-sharing
/// call sites in `pom_gpu` and the startup warmers in `main` compile unchanged).
pub fn ensure_loaded(gguf: &str, gpu: usize) -> bool {
    ensure_loaded_on(gguf, gpu)
}

/// Engine active for exactly this (gguf, gpu)?
pub fn active_for(gguf: &str, gpu: usize) -> bool {
    let slot = slot_for(gpu);
    let g = match slot.inner.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    g.as_ref().map_or(false, |e| e.gguf == gguf)
}

/// Any GPU currently hosting a llama engine?
pub fn available() -> bool {
    !loaded_gpus().is_empty()
}

/// Is a llama engine resident on this specific card?
pub fn available_on(gpu: usize) -> bool {
    let slot = slot_for(gpu);
    slot.inner.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Free the resident model and disable the engine (available() -> false). Used when llama's
/// resident layout for this model is NOT byte-compatible with the canonical possession index
/// (e.g. llama repacks Gemma's tied embeddings) — the walk must gather the canonical GGUF bytes,
/// so we free llama's VRAM and let the caller fall back to candle for BOTH walk and inference.
pub fn unload() {
    let slots: Vec<Arc<EngineSlot>> = match pool().lock() {
        Ok(g) => g.values().cloned().collect(),
        Err(p) => p.into_inner().values().cloned().collect(),
    };
    for slot in slots {
        let mut g = match slot.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(e) = g.take() {
            unsafe { (e.free)(e.model) };
        }
    }
}

/// Free the resident model and disable the engine only for the given GPU. Used both for stale-GPU
/// recovery after a transient fault on that specific device and for the per-card inference model
/// swap — a GPU that does NOT host the engine must not be able to free another card's resident
/// model. Per-GPU slots make this scope-exact by construction. (upstream Keryx-Labs/keryx-miner@278098b)
pub fn unload_for_gpu(gpu: usize) {
    let slot = slot_for(gpu);
    let mut g = match slot.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(e) = g.take() {
        unsafe { (e.free)(e.model) };
    }
}

/// Drops `gpu`'s engine handle WITHOUT calling llama's free.
///
/// Only for a device whose CUDA context is already poisoned (sticky ILLEGAL_ADDRESS). Freeing
/// through llama there runs `llama_context`'s destructor into `cudaFreeHost`/`ggml_backend_buffer_free`
/// on the dead context; ggml treats a CUDA error in teardown as unrecoverable and calls `ggml_abort`,
/// which kills the WHOLE PROCESS — on a 6-card rig that takes down five healthy cards because one
/// card faulted, then HiveOS restarts the miner, and the cycle repeats every few minutes. The engine
/// allocation is unreachable anyway until the context is destroyed, so leaking it is strictly better
/// than aborting. Returns whether a handle was actually abandoned.
pub fn abandon_for_gpu(gpu: usize) -> bool {
    let slot = slot_for(gpu);
    let mut g = match slot.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    // `Engine` has no Drop impl — the free is always explicit — so simply dropping it frees nothing.
    g.take().is_some()
}

/// Resident tensors of the engine on `gpu` in CANONICAL (name-sorted) order:
/// (name, data_ptr, nbytes, is_device). The zero-dup PoM walk on `gpu` gathers over these.
pub fn tensors_for(gpu: usize) -> Option<Vec<(String, u64, usize, bool)>> {
    let slot = slot_for(gpu);
    let g = slot.inner.lock().ok()?;
    let e = g.as_ref()?;
    let n = unsafe { (e.count)(e.model) };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut name: *const c_char = std::ptr::null();
        let mut data: *mut c_void = std::ptr::null_mut();
        let mut nbytes: usize = 0;
        let mut is_dev: c_int = 0;
        let ok = unsafe { (e.info)(e.model, i, &mut name, &mut data, &mut nbytes, &mut is_dev) };
        if !ok || name.is_null() || data.is_null() {
            return None;
        }
        let nm = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
        out.push((nm, data as u64, nbytes, is_dev != 0));
    }
    Some(out)
}

/// Legacy shim: tensors of the single/first resident engine (used only where exactly one card
/// hosts a model). Prefer `tensors_for(gpu)`.
pub fn tensors() -> Option<Vec<(String, u64, usize, bool)>> {
    let gpu = loaded_gpus().into_iter().next()?;
    tensors_for(gpu)
}

/// First resident tensor of the engine on `expected_gpu` whose bytes do NOT live on that device,
/// as (name, owning ordinal). `None` = all device tensors are on `expected_gpu` (or the .so lacks
/// `keryx_llama_tensor_device`, or no engine is loaded — the gate then does nothing).
///
/// Upstream aa29fd2: the possession walk on `expected_gpu` gathers straight over these pointers;
/// launching the kernel on a device that does not own them dereferences unmapped memory, which
/// raises a sticky CUDA_ERROR_ILLEGAL_ADDRESS and poisons that card's whole primary context —
/// inference included. Adapted to our per-GPU slot pool (upstream reads the singleton engine).
pub fn foreign_device_tensor(expected_gpu: usize) -> Option<(String, i32)> {
    let slot = slot_for(expected_gpu);
    let g = slot.inner.lock().ok()?;
    let e = g.as_ref()?;
    let tensor_device = e.tensor_device?;
    let n = unsafe { (e.count)(e.model) };
    for i in 0..n {
        let mut name: *const c_char = std::ptr::null();
        let mut data: *mut c_void = std::ptr::null_mut();
        let mut nbytes: usize = 0;
        let mut is_dev: c_int = 0;
        let ok = unsafe { (e.info)(e.model, i, &mut name, &mut data, &mut nbytes, &mut is_dev) };
        if !ok || name.is_null() || data.is_null() || is_dev == 0 {
            continue;
        }
        let owner = unsafe { tensor_device(e.model, i) };
        if owner >= 0 && owner as usize != expected_gpu {
            let nm = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
            return Some((nm, owner));
        }
    }
    None
}

/// Generate OPoI text on `gpu`'s engine. Serialized per-card (one generation per card at a time);
/// different cards run concurrently. None on any failure (caller falls back).
pub fn generate_on(gpu: usize, prompt: &str, max_tokens: usize) -> Option<String> {
    let slot = slot_for(gpu);
    let g = match slot.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let e = g.as_ref()?;
    generate_locked(e, prompt, max_tokens)
}

/// Generate only when the exact requested GGUF is still resident on this GPU. The identity check
/// and FFI call share one slot-lock epoch, closing the active_for/available/generate TOCTOU where a
/// concurrent self-test could swap models between those calls.
pub fn generate_for(gpu: usize, gguf: &str, prompt: &str, max_tokens: usize) -> Option<String> {
    let slot = slot_for(gpu);
    let g = match slot.inner.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let e = g.as_ref()?;
    if e.gguf != gguf {
        log::warn!("llama engine: refusing generation on GPU {}: requested model is not resident", gpu);
        return None;
    }
    generate_locked(e, prompt, max_tokens)
}

fn generate_locked(e: &Engine, prompt: &str, max_tokens: usize) -> Option<String> {
    if prompt.is_empty()
        || prompt.len() > crate::slm::MAX_INFERENCE_PROMPT_BYTES
        || max_tokens == 0
        || max_tokens > crate::slm::MAX_INFERENCE_TOKENS
    {
        return None;
    }
    let cp = CString::new(prompt).ok()?;
    let max_tokens = c_int::try_from(max_tokens).ok()?;
    let mut buf = vec![0u8; 64 * 1024];
    let n =
        unsafe { (e.generate)(e.model, cp.as_ptr(), max_tokens, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
    if n <= 0 {
        return None;
    }
    buf.truncate(n as usize);
    String::from_utf8(buf).ok()
}

/// Legacy shim: generate on the single/first resident engine. Prefer `generate_on(gpu, …)`.
pub fn generate(prompt: &str, max_tokens: usize) -> Option<String> {
    let gpu = loaded_gpus().into_iter().next()?;
    generate_on(gpu, prompt, max_tokens)
}
