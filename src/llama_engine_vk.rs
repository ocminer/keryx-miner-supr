//! In-process llama.cpp engine, AMD/Vulkan flavor (dlopen'd `libkeryx-llama-vk.so`) — zero-dup:
//! llama.cpp hosts the SINGLE resident model copy on the inference GPU, the PoM walk gathers
//! straight over its VRAM tensors (wrapper exports `keryx_llama_pom_mine`/`_fetch` — byte-exact
//! per the full-model spike + the startup byte gate below), and OPoI text generation runs
//! in-process. Absent/failed .so = byte-identical fallback chain to before (Vulkan llama-server
//! subprocess → candle-CPU), and every card keeps its own OpenCL blob.
//!
//! Consensus safety: this module only changes WHO HOSTS the model bytes on the inference card and
//! WHO GENERATES the user-facing OPoI text. The walk math is byte-identical (pom_walk_vk.comp),
//! and [`pom_byte_gate`] cross-checks the engine's gather against the host possession index at
//! every startup — any mismatch refuses zero-dup and the OpenCL blob path takes over.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Mutex, OnceLock};

use nix::libc;

type AbiFn = unsafe extern "C" fn() -> c_int;
type LoadFn = unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void;
type GenFn = unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_char, c_int) -> c_int;
type ReadyFn = unsafe extern "C" fn(*mut c_void) -> bool;
type U64Fn = unsafe extern "C" fn(*mut c_void) -> u64;
type FetchFn = unsafe extern "C" fn(*mut c_void, u64, *mut u8) -> bool;
type MineFn = unsafe extern "C" fn(*mut c_void, *const u64, *const u64, u64, u64, u32, u32) -> i64;
type PciFn = unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32, *mut u32, *mut u32) -> bool;

const ABI: c_int = 2;
const VK_ABI: c_int = 2;

/// Max nonces per engine dispatch — same TDR-safety reasoning as `pom_opencl::SUB_DISPATCH_NONCES`
/// (ascending sub-batches, early exit on the first winner = identical lowest-nonce semantics).
const SUB_DISPATCH_NONCES: u64 = 1 << 18;

struct Engine {
    model: *mut c_void,
    generate: GenFn,
    pom_ready: ReadyFn,
    pom_n_chunks: U64Fn,
    pom_supl_bytes: U64Fn,
    pom_fetch: FetchFn,
    pom_mine: MineFn,
    pom_pci: PciFn,
    gpu: usize,
    gguf: String,
}
// The wrapper serializes generation + walk dispatches internally (gen_lock / walk_lock).
unsafe impl Send for Engine {}

fn engine() -> &'static Mutex<Option<Engine>> {
    static E: OnceLock<Mutex<Option<Engine>>> = OnceLock::new();
    E.get_or_init(|| Mutex::new(None))
}

/// `KERYX_LLAMA_VK_SO=<path>` wins; else `libkeryx-llama-vk.so` next to our own executable.
fn so_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("KERYX_LLAMA_VK_SO") {
        let pb = std::path::PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
        log::warn!("llama-vk engine: KERYX_LLAMA_VK_SO points at a missing file — ignoring.");
    }
    let exe = std::env::current_exe().ok()?;
    let p = exe.parent()?.join("libkeryx-llama-vk.so");
    if p.exists() {
        Some(p)
    } else {
        None
    }
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
/// whether the engine is active for `gguf`. Safe to call from multiple threads.
pub fn ensure_loaded(gguf: &str, gpu: usize) -> bool {
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(e) = g.as_ref() {
        return e.gguf == gguf;
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
            log::warn!("llama-vk engine: dlopen({}) failed: {} — llama-server/candle fallbacks stay active.", so.display(), msg);
            return false;
        }
        let (Some(abi), Some(vk_abi), Some(load), Some(gen), Some(ready), Some(nch), Some(supl), Some(fetch), Some(mine), Some(pci)) = (
            sym::<AbiFn>(lib, "keryx_llama_abi"),
            sym::<AbiFn>(lib, "keryx_llama_vk_abi"),
            sym::<LoadFn>(lib, "keryx_llama_load"),
            sym::<GenFn>(lib, "keryx_llama_generate"),
            sym::<ReadyFn>(lib, "keryx_llama_pom_ready"),
            sym::<U64Fn>(lib, "keryx_llama_pom_n_chunks"),
            sym::<U64Fn>(lib, "keryx_llama_pom_supl_bytes"),
            sym::<FetchFn>(lib, "keryx_llama_pom_fetch"),
            sym::<MineFn>(lib, "keryx_llama_pom_mine"),
            sym::<PciFn>(lib, "keryx_llama_pom_pci"),
        ) else {
            log::warn!("llama-vk engine: {} is missing symbols — fallbacks stay active.", so.display());
            return false;
        };
        if abi() != ABI || vk_abi() != VK_ABI {
            log::warn!(
                "llama-vk engine: {} ABI {}/{} != expected {}/{} — fallbacks stay active.",
                so.display(), abi(), vk_abi(), ABI, VK_ABI
            );
            return false;
        }
        let cg = match CString::new(gguf) {
            Ok(c) => c,
            Err(_) => return false,
        };
        log::info!("llama-vk engine: loading {gguf} (requested GPU {gpu}) via {} (in-process, zero-dup)…", so.display());
        let n_ctx: c_int = std::env::var("KERYX_LLAMA_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
        let model = load(cg.as_ptr(), gpu as c_int, n_ctx);
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
            generate: gen,
            pom_ready: ready,
            pom_n_chunks: nch,
            pom_supl_bytes: supl,
            pom_fetch: fetch,
            pom_mine: mine,
            pom_pci: pci,
            gpu,
            gguf: gguf.to_string(),
        });
        true
    }
}

pub fn available() -> bool {
    match engine().lock() {
        Ok(g) => g.is_some(),
        Err(_) => false,
    }
}

/// Generate up to `max_tokens` of OPoI text in-process.
pub fn generate(prompt: &str, max_tokens: usize) -> Option<String> {
    let g = engine().lock().ok()?;
    let e = g.as_ref()?;
    let cp = CString::new(prompt).ok()?;
    let cap: usize = 65536;
    let mut out = vec![0u8; cap];
    let n = unsafe { (e.generate)(e.model, cp.as_ptr(), max_tokens as c_int, out.as_mut_ptr() as *mut c_char, cap as c_int) };
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
pub fn pom_mine(p: [u64; 4], time: u64, t: [u64; 4], nonce_base: u64, batch: u64) -> Option<u64> {
    let g = engine().lock().ok()?;
    let e = g.as_ref()?;
    let mut done: u64 = 0;
    while done < batch {
        let sub = (batch - done).min(SUB_DISPATCH_NONCES) as u32;
        let base = nonce_base.wrapping_add(done);
        let r = unsafe {
            (e.pom_mine)(e.model, p.as_ptr(), t.as_ptr(), time, base, sub, crate::pom::POM_WALK_STEPS)
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
