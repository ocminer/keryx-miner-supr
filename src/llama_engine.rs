//! In-process llama.cpp engine via a dlopen'd `libkeryx-llama.so` (Phase 2 candle-independence).
//!
//! When the .so sits next to the miner binary (or `KERYX_LLAMA_SO` points at it), it becomes the
//! DEFAULT engine for the primary model on the inference GPU: llama.cpp owns the single resident
//! VRAM copy, the PoM walk gathers straight over its tensor pointers (zero-dup — byte-identity
//! proven by tools/llama_zerodup_spike), and OPoI text generation runs in-process. Absent .so =
//! the candle paths stay active as the (dormant) fallback, byte-identical to before.
//!
//! Consensus safety: this module only changes WHO HOSTS the model bytes and WHO GENERATES the
//! user-facing OPoI text. The walk kernel, the host possession index, proofs and `tag_fixed` are
//! untouched; `ensure_installed_inner`'s N-guard cross-checks the gather against the host index.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Arc, Mutex, OnceLock};

use nix::libc;

type AbiFn = unsafe extern "C" fn() -> c_int;
type LoadFn = unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void;
type CountFn = unsafe extern "C" fn(*mut c_void) -> usize;
type InfoFn = unsafe extern "C" fn(*mut c_void, usize, *mut *const c_char, *mut *mut c_void, *mut usize, *mut c_int) -> bool;
type GenFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_char, c_int) -> c_int;
type FreeFn = unsafe extern "C" fn(*mut c_void);
// CUDA ordinal owning tensor i's bytes, or -1 (host/unified/unknown). Upstream aa29fd2 — optional
// symbol (absent on older .so builds → the ownership gate below degrades to a no-op).
type TensorDeviceFn = unsafe extern "C" fn(*mut c_void, usize) -> c_int;

const ABI: c_int = 2;

struct Engine {
    model: *mut c_void,
    count: CountFn,
    info: InfoFn,
    generate: GenFn,
    free: FreeFn,
    /// Optional: absent on older `libkeryx-llama.so` builds (looked up soft, gate no-ops if None).
    tensor_device: Option<TensorDeviceFn>,
    gpu: usize,
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
    g.entry(gpu)
        .or_insert_with(|| Arc::new(EngineSlot { inner: Mutex::new(None) }))
        .clone()
}

/// Snapshot of the currently-populated GPU ordinals (a slot may exist but be empty; only ordinals
/// whose slot holds a live engine are returned).
fn loaded_gpus() -> Vec<usize> {
    let slots: Vec<(usize, Arc<EngineSlot>)> = match pool().lock() {
        Ok(g) => g.iter().map(|(k, v)| (*k, v.clone())).collect(),
        Err(_) => return Vec::new(),
    };
    slots
        .into_iter()
        .filter(|(_, s)| s.inner.lock().map(|g| g.is_some()).unwrap_or(false))
        .map(|(k, _)| k)
        .collect()
}

/// Actually dlopen the .so, resolve symbols, load the model for (gguf, gpu). Returns the Engine or
/// None (caller falls back). Factored out of `ensure_loaded_on` so the load happens while the
/// target slot's lock is held (per-card serialization) without duplicating the FFI plumbing.
fn load_engine(gguf: &str, gpu: usize) -> Option<Engine> {
    let so = so_path()?;
    let cso = CString::new(so.to_string_lossy().as_bytes()).ok()?;
    unsafe {
        let lib = libc::dlopen(cso.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if lib.is_null() {
            let err = libc::dlerror();
            let msg = if err.is_null() { "?".into() } else { CStr::from_ptr(err).to_string_lossy().into_owned() };
            log::warn!("llama engine: dlopen({}) failed: {} — candle fallback stays active.", so.display(), msg);
            return None;
        }
        let (Some(abi), Some(load), Some(count), Some(info), Some(gen), Some(free)) = (
            sym::<AbiFn>(lib, "keryx_llama_abi"),
            sym::<LoadFn>(lib, "keryx_llama_load"),
            sym::<CountFn>(lib, "keryx_llama_tensor_count"),
            sym::<InfoFn>(lib, "keryx_llama_tensor_info"),
            sym::<GenFn>(lib, "keryx_llama_generate"),
            sym::<FreeFn>(lib, "keryx_llama_free"),
        ) else {
            log::warn!("llama engine: {} is missing symbols — candle fallback stays active.", so.display());
            return None;
        };
        let got = abi();
        if got != ABI {
            log::warn!("llama engine: {} ABI {} != expected {} — candle fallback stays active.", so.display(), got, ABI);
            return None;
        }
        // Optional (upstream aa29fd2): older .so builds lack it — the ownership gate then no-ops.
        let tensor_device = sym::<TensorDeviceFn>(lib, "keryx_llama_tensor_device");
        let cg = CString::new(gguf).ok()?;
        log::info!("llama engine: loading {} on GPU {} via {} (in-process, zero-dup)…", gguf, gpu, so.display());
        let n_ctx: c_int = std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
        let model = load(cg.as_ptr(), gpu as c_int, n_ctx);
        if model.is_null() {
            log::warn!("llama engine: model load failed on GPU {} (VRAM? arch?) — candle fallback stays active.", gpu);
            return None;
        }
        log::info!("llama engine: ✓ active on GPU {} — llama.cpp hosts the model + serves OPoI inference (candle dormant).", gpu);
        Some(Engine { model, count, info, generate: gen, free, tensor_device, gpu, gguf: gguf.to_string() })
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
    let want_noavx = !simd_ok
        || std::env::var("KERYX_LLAMA_FORCE_NOAVX").map_or(false, |v| v == "1");
    if want_noavx {
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
            "llama engine: this CPU lacks the AVX2/FMA/F16C/BMI2 set baked into libkeryx-llama.so and no libkeryx-llama-noavx.so is present — NOT loading it (would SIGILL). Candle fallback stays active; ship the -noavx variant from the 0.7.2+ package to enable the llama engine on this CPU."
        );
        return None;
    }
    // macOS ships a .dylib (Mach-O). Every other unix (Linux/BSD) ships a .so (ELF). Probe the
    // native name first, and on macOS also fall back to .so — some HiveOS-adjacent tooling may
    // repackage the Linux .so alongside the macOS binary during cross-arch testing.
    #[cfg(target_os = "macos")]
    let candidates: [&str; 2] = ["libkeryx-llama.dylib", "libkeryx-llama.so"];
    #[cfg(not(target_os = "macos"))]
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

unsafe fn sym<T: Copy>(lib: *mut c_void, name: &str) -> Option<T> {
    let c = CString::new(name).ok()?;
    let p = libc::dlsym(lib, c.as_ptr());
    if p.is_null() {
        return None;
    }
    // fn-pointer types are pointer-sized; read the address as T.
    Some(std::mem::transmute_copy::<*mut c_void, T>(&p))
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
    let g = match slot.inner.lock() { Ok(g) => g, Err(_) => return false };
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
        let mut g = match slot.inner.lock() { Ok(g) => g, Err(p) => p.into_inner() };
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
    let mut g = match slot.inner.lock() { Ok(g) => g, Err(p) => p.into_inner() };
    if let Some(e) = g.take() {
        unsafe { (e.free)(e.model) };
    }
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
    let g = match slot.inner.lock() { Ok(g) => g, Err(p) => p.into_inner() };
    let e = g.as_ref()?;
    let cp = CString::new(prompt).ok()?;
    let mut buf = vec![0u8; 64 * 1024];
    let n = unsafe { (e.generate)(e.model, cp.as_ptr(), max_tokens as c_int, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
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
