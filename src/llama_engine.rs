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

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Mutex, OnceLock};

use nix::libc;

type AbiFn = unsafe extern "C" fn() -> c_int;
type LoadFn = unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void;
type CountFn = unsafe extern "C" fn(*mut c_void) -> usize;
type InfoFn = unsafe extern "C" fn(*mut c_void, usize, *mut *const c_char, *mut *mut c_void, *mut usize, *mut c_int) -> bool;
type GenFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_char, c_int) -> c_int;
type FreeFn = unsafe extern "C" fn(*mut c_void);

const ABI: c_int = 2;

struct Engine {
    model: *mut c_void,
    count: CountFn,
    info: InfoFn,
    generate: GenFn,
    free: FreeFn,
    gpu: usize,
    gguf: String,
}
// The wrapper serializes generation internally; tensor info is read-only after load.
unsafe impl Send for Engine {}

fn engine() -> &'static Mutex<Option<Engine>> {
    static E: OnceLock<Mutex<Option<Engine>>> = OnceLock::new();
    E.get_or_init(|| Mutex::new(None))
}

/// `KERYX_LLAMA_SO=<path>` wins; else the platform-native shared library next to our own
/// executable (`libkeryx-llama.dylib` on macOS, `libkeryx-llama.so` elsewhere).
fn so_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("KERYX_LLAMA_SO") {
        let pb = std::path::PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
        log::warn!("llama engine: KERYX_LLAMA_SO points at a missing file — ignoring.");
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
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

/// Load the .so + the model once (idempotent, blocking — a model load takes seconds). Returns
/// whether the engine is active for `gguf` on `gpu`. Safe to call from multiple threads.
pub fn ensure_loaded(gguf: &str, gpu: usize) -> bool {
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(e) = g.as_ref() {
        return e.gguf == gguf && e.gpu == gpu;
    }
    let Some(so) = so_path() else { return false };
    let cso = match CString::new(so.to_string_lossy().as_bytes()) {
        Ok(c) => c,
        Err(_) => return false,
    };
    unsafe {
        let lib = libc::dlopen(cso.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if lib.is_null() {
            let err = libc::dlerror();
            let msg = if err.is_null() { "?".into() } else { CStr::from_ptr(err).to_string_lossy().into_owned() };
            log::warn!("llama engine: dlopen({}) failed: {} — candle fallback stays active.", so.display(), msg);
            return false;
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
            return false;
        };
        let got = abi();
        if got != ABI {
            log::warn!("llama engine: {} ABI {} != expected {} — candle fallback stays active.", so.display(), got, ABI);
            return false;
        }
        let cg = match CString::new(gguf) {
            Ok(c) => c,
            Err(_) => return false,
        };
        log::info!("llama engine: loading {} on GPU {} via {} (in-process, zero-dup)…", gguf, gpu, so.display());
        let n_ctx: c_int = std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
        let model = load(cg.as_ptr(), gpu as c_int, n_ctx);
        if model.is_null() {
            log::warn!("llama engine: model load failed (VRAM? arch?) — candle fallback stays active.");
            return false;
        }
        *g = Some(Engine { model, count, info, generate: gen, free, gpu, gguf: gguf.to_string() });
        log::info!("llama engine: ✓ active — llama.cpp hosts the model + serves OPoI inference (candle dormant).");
        true
    }
}

/// Engine active for exactly this (gguf, gpu)?
pub fn active_for(gguf: &str, gpu: usize) -> bool {
    match engine().lock() {
        Ok(g) => g.as_ref().map_or(false, |e| e.gguf == gguf && e.gpu == gpu),
        Err(_) => false,
    }
}

pub fn available() -> bool {
    match engine().lock() {
        Ok(g) => g.is_some(),
        Err(_) => false,
    }
}

/// Free the resident model and disable the engine (available() -> false). Used when llama's
/// resident layout for this model is NOT byte-compatible with the canonical possession index
/// (e.g. llama repacks Gemma's tied embeddings) — the walk must gather the canonical GGUF bytes,
/// so we free llama's VRAM and let the caller fall back to candle for BOTH walk and inference.
pub fn unload() {
    if let Ok(mut g) = engine().lock() {
        if let Some(e) = g.take() {
            unsafe { (e.free)(e.model) };
        }
    }
}

/// Resident tensors in CANONICAL (name-sorted) order: (name, data_ptr, nbytes, is_device).
pub fn tensors() -> Option<Vec<(String, u64, usize, bool)>> {
    let g = engine().lock().ok()?;
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

/// Generate OPoI text via the in-process engine. None on any failure (caller falls back).
pub fn generate(prompt: &str, max_tokens: usize) -> Option<String> {
    let g = engine().lock().ok()?;
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
