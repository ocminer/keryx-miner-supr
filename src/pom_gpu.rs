//! Proof-of-Model GPU mining — runs the `pom_mine` kernel in candle's CUDA context over the
//! resident weight blob to find a winning nonce. Foundation for the live mining loop (§6/3b).
//!
//! Loads the mining tier's GGUF raw (so we get per-tensor device pointers for the gather, like
//! `pom-q4-probe`) and builds the chunk-prefix gather index on the GPU. NOTE: this is a second
//! VRAM copy of the model (the inference engine holds its own). Fine for small tiers on the
//! testnet; the big tiers will share buffers later.
//!
//! The kernel's seed/pow folds are byte-identical to `pom::pom_block_seed`/`pom::pom_pow_value`,
//! so a nonce found here builds a `PomProof` (host) the node accepts.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::cuda_backend::cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchArgs, LaunchConfig, PushKernelArg,
};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{CudaDevice, Device};

/// The walk kernel image, embedded at build time (build.rs). Either a native-SASS FATBIN
/// (sm_75..120 + compute_75 PTX fallback — modern; runs on Blackwell with NO driver JIT) or
/// compute_XX PTX text (legacy/pascal). `POM_WALK_IMAGE_KIND` says which.
const WALK_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_mine.image"));
const WALK_IMAGE_KIND: &str = env!("POM_WALK_IMAGE_KIND"); // "fatbin" | "ptx"
/// The arch the walk image targets (build.rs bakes it in) — "sm_75..120-native" for the modern
/// fatbin, or sm_70/sm_60 for legacy/pascal PTX. Used to explain arch-mismatch load failures.
const PTX_ARCH: &str = env!("POM_PTX_ARCH");
const CHUNK_BYTES: usize = 32;

/// Build the cudarc module image for `load_module`. cudarc 0.17 can only construct `Ptx` from a PTX
/// string or a FILE path (the raw-bytes variant is crate-private), and `cuModuleLoad`/`LoadData`
/// auto-detect fatbin/cubin/ptx — so for the fatbin we materialize the embedded bytes to a temp file
/// ONCE and load it by path. This is what makes the walk run native sm_120 SASS instead of JIT'd PTX
/// (the Windows compute_75-PTX-JIT was ~5x slower than the native fatbin — "limited on power").
fn walk_image() -> candle_core::Result<candle_core::cuda_backend::cudarc::nvrtc::Ptx> {
    use candle_core::cuda_backend::cudarc::nvrtc::Ptx;
    // Bench/dev override: load an external fatbin from disk instead of the embedded image
    // (kernel-tuning A/Bs without rebuilding the Rust binary). Never set in production.
    if let Ok(p) = std::env::var("KERYX_WALK_FATBIN") {
        if !p.is_empty() {
            return Ok(Ptx::from_file(std::path::PathBuf::from(p)));
        }
    }
    if WALK_IMAGE_KIND == "fatbin" {
        static PATH: OnceLock<std::path::PathBuf> = OnceLock::new();
        let p = if let Some(p) = PATH.get() {
            p.clone()
        } else {
            // Name the temp file by image length so a new build never reuses a stale fatbin.
            let p = std::env::temp_dir().join(format!("keryx-pom-walk-{}.fatbin", WALK_IMAGE.len()));
            if !p.exists() {
                std::fs::write(&p, WALK_IMAGE)
                    .map_err(|e| candle_core::Error::Msg(format!("write walk fatbin {}: {e}", p.display())))?;
            }
            let _ = PATH.set(p.clone());
            p
        };
        Ok(Ptx::from_file(p))
    } else {
        let src = std::str::from_utf8(WALK_IMAGE)
            .map_err(|e| candle_core::Error::Msg(format!("walk PTX image is not UTF-8: {e}")))?;
        Ok(Ptx::from_src(src))
    }
}

/// The walk kernel bound to its device's stream. Mirrors the `CudaFunc` wrapper the vendored
/// candle-core used to export: stock candle 0.9 keeps that type in a private module
/// (`cuda_backend::device` is not `pub`), so since Phase 3d dropped the vendor we carry
/// (function, stream) ourselves and load the PTX through the re-exported `cudarc` driver API.
struct WalkFunc {
    func: CudaFunction,
    stream: Arc<CudaStream>,
}

impl WalkFunc {
    fn builder(&self) -> LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}

/// JIT'd walk module per CUDA context (device ordinal) — mine() must never compile on the hot
/// path. Replaces the `custom_modules` cache the vendored candle-core kept device-side.
static WALK_MODULES: OnceLock<Mutex<HashMap<usize, Arc<CudaModule>>>> = OnceLock::new();

/// Per-device cached v4 tile-offsets buffer (batch*K u32). PLAIN alloc, never zeroed — the chase
/// kernel overwrites every word before the walk reads it. The old per-batch `alloc_zeros` paid a
/// cudaMalloc + full memset + free EVERY batch (measured 6-10% of the whole solve at batch 16K).
/// Grows monotonically; ~16-64 MB/device held for the miner's lifetime (same policy as WALK_MODULES).
static V4_OFFSETS: OnceLock<Mutex<HashMap<usize, Arc<CudaSlice<u32>>>>> = OnceLock::new();

fn v4_offsets_buf(stream: &Arc<CudaStream>, len: usize) -> candle_core::Result<Arc<CudaSlice<u32>>> {
    let m = V4_OFFSETS.get_or_init(|| Mutex::new(HashMap::new()));
    let ord = stream.context().ordinal();
    let mut g = m.lock().unwrap();
    if let Some(s) = g.get(&ord) {
        if s.len() >= len {
            return Ok(s.clone());
        }
    }
    let s = Arc::new(unsafe { stream.alloc::<u32>(len) }.map_err(candle_core::Error::wrap)?);
    g.insert(ord, s.clone());
    Ok(s)
}

/// Per-device secondary stream for the overlapped chase (sub-batch pipelining in `mine_v4`).
static V4_CHASE_STREAMS: OnceLock<Mutex<HashMap<usize, Arc<CudaStream>>>> = OnceLock::new();

fn v4_chase_stream(stream: &Arc<CudaStream>) -> candle_core::Result<Arc<CudaStream>> {
    let m = V4_CHASE_STREAMS.get_or_init(|| Mutex::new(HashMap::new()));
    let ord = stream.context().ordinal();
    let mut g = m.lock().unwrap();
    if let Some(s) = g.get(&ord) {
        return Ok(s.clone());
    }
    let s = stream.context().new_stream().map_err(candle_core::Error::wrap)?;
    g.insert(ord, s.clone());
    Ok(s)
}

fn walk_load_err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!(
        "PoM walk kernel failed to load ({e}). TWO distinct causes: \
         (1) CUDA_ERROR_INVALID_PTX / 'no kernel image available' = your GPU is OLDER than this \
         build's arch ({PTX_ARCH}) — the PTX/cubin can't run on it. \
         (2) CUDA_ERROR_UNSUPPORTED_PTX_VERSION ('unsupported toolchain') = your NVIDIA DRIVER is \
         too OLD for this build's PTX toolchain — UPDATE THE DRIVER (MODERN needs 575+), it is NOT \
         a GPU-arch problem. \
         Pick the build line for your GPU: LEGACY = sm_70+ (Volta/V100, CMP 100-210, Turing+), \
         PASCAL = sm_60/61 (GTX 10-series), MODERN = sm_75+ with driver 575+. If you built from \
         source: CUDA 13.x cannot compile for Volta or Pascal — use a CUDA 12.x toolkit and set \
         POM_CUDA_ARCH=compute_70 (Volta) or compute_60 (Pascal), plus CUDA_COMPUTE_CAP to match."
    ))
}

/// Load the walk kernel, turning a bare driver error (typically CUDA_ERROR_INVALID_PTX when the
/// GPU is OLDER than the PTX target — PTX only JITs forward) into an actionable message naming
/// this build's PTX arch and the right build line for older cards.
/// Bind THIS stream's CUDA context to the calling thread. On a MULTI-GPU rig the mining worker
/// threads rotate, so a raw driver call (kernel launch, memcpy, htod upload) would otherwise run
/// against whatever context happens to be current — often another device's — dereferencing a
/// pointer that belongs to a different GPU and raising a sticky CUDA_ERROR_ILLEGAL_ADDRESS that
/// poisons the context (a silent native crash). Upstream binds at every such entry point; we must
/// too. Single-GPU never mis-binds (one context), which is why this only bit multi-GPU rigs.
fn bind_device_ctx(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
    stream.context().bind_to_thread().map_err(candle_core::Error::wrap)
}

fn load_walk_func(cuda: &CudaDevice, name: &str) -> candle_core::Result<WalkFunc> {
    let stream = cuda.cuda_stream();
    let ctx = stream.context().clone();
    let module = {
        let mut cache = WALK_MODULES.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
        match cache.get(&ctx.ordinal()) {
            Some(m) => m.clone(),
            None => {
                let m = ctx.load_module(walk_image()?).map_err(walk_load_err)?;
                cache.insert(ctx.ordinal(), m.clone());
                m
            }
        }
    };
    let func = module.load_function(name).map_err(walk_load_err)?;
    Ok(WalkFunc { func, stream })
}

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

pub struct PomGpuMiner {
    cuda: CudaDevice,
    stream: Arc<CudaStream>,
    bases_dev: CudaSlice<u64>,
    prefix_dev: CudaSlice<u64>,
    t_count: u32,
    n_total_chunks: u64,
    _tensors: Vec<QTensor>, // raw-loaded tensors kept alive so the gather pointers stay valid
    _shared: Vec<Arc<QTensor>>, // shared-with-inference tensors kept alive (zero-dup, Option C)
    _uploads: Vec<CudaSlice<u8>>, // our own device copies of llama-engine host-resident tensors
}

/// Where each canonical tensor's bytes come from for the walk gather.
enum GatherSource {
    /// llama holds this tensor on-device, byte-identical to the GGUF — walk it in place (zero-dup).
    DevicePtr(u64),
    /// llama holds it in a host (CPU) ggml buffer — upload our own device copy of the raw bytes.
    HostPtr(u64),
    /// llama repacked / duplicated / dropped it (e.g. Gemma materialises a separate `output.weight`
    /// from its tied embeddings) — upload this tensor's canonical bytes from the possession index,
    /// so the walked blob is always the canonical one R_T pins regardless of runtime repacking.
    FromIndex,
}

/// Per-tensor plan for gathering the canonical layout from llama's resident tensors.
struct GatherPlan {
    entries: Vec<(usize, GatherSource)>, // (nbytes, source), in canonical name-sorted order
    total_bytes: u64,
    index_bytes: u64, // bytes that must be re-uploaded from the possession index
    zero_dup: usize,
    host_uploads: usize,
    index_uploads: Vec<String>,
}

impl GatherPlan {
    /// Past this share of the blob, walking a raw canonical copy (`load_raw`) costs less VRAM than
    /// keeping llama resident plus the uploads — so fall back to the raw copy instead.
    fn exceeds_upload_budget(&self) -> bool {
        self.index_bytes * 4 > self.total_bytes
    }
}

/// Match llama's resident tensors against the canonical GGUF list. A canonical tensor is walked
/// from llama only on an unambiguous match — unique name AND exact byte size; a duplicated name,
/// a resized copy or a missing tensor falls back to the possession index, so runtime repacking
/// never changes what the walk reads.
fn plan_canonical_gather(
    canonical: &[(String, usize)],
    resident: &[(String, u64, usize, bool)],
) -> GatherPlan {
    let mut by_name: HashMap<&str, Vec<&(String, u64, usize, bool)>> = HashMap::with_capacity(resident.len());
    for t in resident {
        by_name.entry(t.0.as_str()).or_default().push(t);
    }
    let mut plan = GatherPlan {
        entries: Vec::with_capacity(canonical.len()),
        total_bytes: 0,
        index_bytes: 0,
        zero_dup: 0,
        host_uploads: 0,
        index_uploads: Vec::new(),
    };
    for (name, nbytes) in canonical {
        plan.total_bytes += *nbytes as u64;
        let source = match by_name.get(name.as_str()).map(Vec::as_slice) {
            Some([t]) if t.2 == *nbytes && t.3 => {
                plan.zero_dup += 1;
                GatherSource::DevicePtr(t.1)
            }
            Some([t]) if t.2 == *nbytes => {
                plan.host_uploads += 1;
                GatherSource::HostPtr(t.1)
            }
            _ => {
                plan.index_bytes += *nbytes as u64;
                plan.index_uploads.push(name.clone());
                GatherSource::FromIndex
            }
        };
        plan.entries.push((*nbytes, source));
    }
    plan
}

/// Canonical (name, nbytes) list of a GGUF in name-sorted order — the layout the possession index
/// chunks and R_T commits.
fn canonical_tensor_list(gguf: &str) -> Option<Vec<(String, usize)>> {
    let mut file = std::fs::File::open(gguf).ok()?;
    let meta = crate::gguf::GgufMeta::read(&mut file).ok()?;
    meta.sorted_names()
        .into_iter()
        .map(|name| {
            let nbytes = usize::try_from(meta.tensors[&name].nbytes).ok()?;
            Some((name, nbytes))
        })
        .collect()
}

/// True when a host possession index is available to back `FromIndex` uploads on this device.
fn has_possession_index(device_id: usize) -> bool {
    crate::pom::active_index_for(device_id as u32).is_some() || crate::pom::active_index().is_some()
}

/// None when the llama-resident layout can back the canonical walk (with per-tensor index
/// fallback for repacked tensors), Some(reason) when the raw canonical copy must be walked instead.
fn llama_gather_blocker(gguf: &str, device_id: usize) -> Option<String> {
    let resident = match crate::llama_engine::tensors_for(device_id) {
        Some(ts) => ts,
        None => return Some("llama engine tensors unavailable".into()),
    };
    let canonical = match canonical_tensor_list(gguf) {
        Some(c) => c,
        None => return Some("canonical GGUF tensor list unreadable".into()),
    };
    let plan = plan_canonical_gather(&canonical, &resident);
    if plan.exceeds_upload_budget() {
        return Some(format!(
            "llama-resident layout too foreign ({} of {} bytes would need a canonical upload)",
            plan.index_bytes, plan.total_bytes
        ));
    }
    if plan.index_bytes > 0 && !has_possession_index(device_id) {
        return Some("canonical fallback needs the host possession index".into());
    }
    None
}

impl PomGpuMiner {
    /// Load the mining model's GGUF into candle on a specific CUDA device, build the gather
    /// index, load the kernel.
    pub fn load(gguf_path: &str, device_id: usize) -> candle_core::Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();
        bind_device_ctx(&stream)?; // multi-GPU: bind this device's context before raw uploads

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order — matches pom-rt-builder / the node R_T

        let mut tensors: Vec<QTensor> = Vec::with_capacity(names.len());
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        for name in &names {
            let qt = content.tensor(&mut file, name, &device)?;
            let chunks = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
            if chunks == 0 {
                tensors.push(qt);
                continue;
            }
            bases.push(qt.device_ptr()? as usize as u64);
            prefix.push(prefix.last().unwrap() + chunks);
            tensors.push(qt);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: model produced 0 chunks".into()));
        }

        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        // Warm the module cache so mine() never compiles on the hot path.
        let _ = load_walk_func(&cuda, "pom_mine_v4")?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: tensors, _shared: Vec::new(), _uploads: Vec::new() })
    }

    /// Standalone walk source: upload the mining model's RAW GGUF bytes to `device_id` (canonical
    /// name-sorted order) and gather over our own device copies. Unlike `load` (candle QTensor
    /// parse), this only reads the GGUF header for each tensor's offset/size and uploads the
    /// bytes verbatim — so it works for ANY architecture, including the H4 llama-only archs
    /// (Qwen3.5-hybrid-SSM / GLM-4 / EXAONE-4 / Kimi-Linear-MoE) that candle has no loader for.
    /// Used on mining-only GPUs (multi-GPU rigs) and whenever llama's resident layout is not
    /// byte-compatible — the uploaded bytes ARE the canonical on-disk bytes, so no byte-gate is
    /// needed here (the N-guard in `ensure_installed_inner` still cross-checks the host index).
    pub fn load_raw(gguf_path: &str, device_id: usize) -> candle_core::Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();
        bind_device_ctx(&stream)?; // multi-GPU: bind this device's context before raw uploads

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let meta = crate::gguf::GgufMeta::read(&mut file)
            .map_err(|e| candle_core::Error::Msg(format!("PoM GPU: GGUF header parse failed: {e}")))?;
        let names = meta.sorted_names(); // canonical order — matches pom-rt-builder / the node R_T

        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        let mut uploads: Vec<CudaSlice<u8>> = Vec::new();
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut host_buf: Vec<u8> = Vec::new();
        for name in &names {
            let t = &meta.tensors[name];
            let chunks = t.nbytes / CHUNK_BYTES as u64;
            if chunks == 0 {
                continue;
            }
            host_buf.resize(t.nbytes as usize, 0);
            crate::pom::read_exact_at(&file, &mut host_buf, meta.tensor_data_offset + t.offset)
                .map_err(candle_core::Error::wrap)?;
            let dev = stream.clone_htod(host_buf.as_slice()).map_err(candle_core::Error::wrap)?;
            bases.push(dev.device_ptr(&stream).0 as u64);
            uploads.push(dev);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: model produced 0 chunks".into()));
        }
        info!("PoM raw gather: {} tensors uploaded, N={} chunks (candle-free, arch-agnostic)", bases.len(), n_total_chunks);

        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let _ = load_walk_func(&cuda, "pom_mine_v4")?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: Vec::new(), _shared: Vec::new(), _uploads: uploads })
    }

    /// Zero-dup load (Option C): build the gather over the SAME canonical name-sorted layout as
    /// `R_T`, but for each tensor reuse the inference engine's resident VRAM buffer when it holds
    /// it quantized (`shared`, the big matrices) instead of loading a second copy. Only the
    /// dequantized-in-inference tensors (token_embd, norms) are read raw here — small. `device`
    /// MUST be the same candle device the `shared` tensors live on (pointers are context-bound).
    pub fn load_shared(
        gguf_path: &str,
        device: &Device,
        shared: &std::collections::HashMap<String, Arc<QTensor>>,
    ) -> candle_core::Result<Self> {
        let cuda = match device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: shared load requires a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();
        bind_device_ctx(&stream)?; // multi-GPU: bind this device's context before raw uploads

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order — must match pom-rt-builder / the node R_T

        let mut raw: Vec<QTensor> = Vec::new();
        let mut kept_shared: Vec<Arc<QTensor>> = Vec::new();
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut shared_hits = 0usize;
        for name in &names {
            let (ptr, chunks) = if let Some(qt) = shared.get(name) {
                // Matrix already resident for inference → reuse its buffer (zero dup).
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
                let p = qt.device_ptr()? as usize as u64;
                kept_shared.push(qt.clone());
                shared_hits += 1;
                (p, c)
            } else {
                // Dequantized-in-inference (token_embd, norms): read the raw quantized bytes.
                let qt = content.tensor(&mut file, name, device)?;
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
                if c == 0 {
                    raw.push(qt);
                    continue;
                }
                let p = qt.device_ptr()? as usize as u64;
                raw.push(qt);
                (p, c)
            };
            if chunks == 0 {
                continue;
            }
            bases.push(ptr);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: shared load produced 0 chunks".into()));
        }
        info!("PoM zero-dup gather: {} shared tensors, {} raw-loaded, N={} chunks", shared_hits, raw.len(), n_total_chunks);

        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let _ = load_walk_func(&cuda, "pom_mine_v4")?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: raw, _shared: kept_shared, _uploads: Vec::new() })
    }

    /// Phase-2 canonical gather over the IN-PROCESS llama.cpp engine (candle hosts nothing): walk
    /// the canonical (name-sorted) GGUF tensor list, and for EACH canonical tensor pick its source —
    /// llama's on-device copy (zero-dup, byte-identity proven by tools/llama_zerodup_spike), llama's
    /// host (CPU) copy (small device upload), or, when llama repacked/duplicated/dropped it (e.g.
    /// Gemma materialises a separate `output.weight` from its tied embeddings), the canonical bytes
    /// from the possession index. This keeps the big matrices zero-dup while patching only the
    /// repacked tensors, so a repacking arch fits ONE resident copy (walk + inference) instead of a
    /// full second raw copy. candle's CudaDevice here is pure CUDA plumbing (context/stream/kernel).
    pub fn load_llama(gguf: &str, device_id: usize) -> candle_core::Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();
        bind_device_ctx(&stream)?; // multi-GPU: bind this device's context before raw uploads/byte-gate
        let ts = crate::llama_engine::tensors_for(device_id)
            .ok_or_else(|| candle_core::Error::Msg("PoM GPU: llama engine tensors unavailable".into()))?;
        let canonical = canonical_tensor_list(gguf)
            .ok_or_else(|| candle_core::Error::Msg("PoM GPU: canonical GGUF tensor list unreadable".into()))?;
        let plan = plan_canonical_gather(&canonical, &ts);
        if plan.exceeds_upload_budget() {
            return Err(candle_core::Error::Msg(format!(
                "PoM GPU: llama-resident layout too foreign — {} of {} bytes need a canonical upload",
                plan.index_bytes, plan.total_bytes
            )));
        }
        // Possession index backs the FromIndex uploads (repacked tensors) and the byte gate below.
        let idx_ref = crate::pom::active_index_for(device_id as u32)
            .map(|t| &t.0)
            .or_else(|| crate::pom::active_index().map(|(i, _)| i));
        if plan.index_bytes > 0 && idx_ref.is_none() {
            return Err(candle_core::Error::Msg(
                "PoM GPU: canonical fallback needs the host possession index".into(),
            ));
        }
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut uploads: Vec<CudaSlice<u8>> = Vec::new();
        for (nbytes, source) in &plan.entries {
            let chunks = (nbytes / CHUNK_BYTES) as u64;
            if chunks == 0 {
                continue;
            }
            let base = match source {
                GatherSource::DevicePtr(p) => *p,
                GatherSource::HostPtr(p) => {
                    // Host-resident in ggml (CPU buffer): the walk needs device memory — upload our
                    // own copy of the raw bytes (identical to the GGUF bytes, same as the pointer).
                    let host: &[u8] = unsafe { std::slice::from_raw_parts(*p as *const u8, *nbytes) };
                    let dev = stream.clone_htod(host).map_err(candle_core::Error::wrap)?;
                    let ptr = dev.device_ptr(&stream).0 as u64;
                    uploads.push(dev);
                    ptr
                }
                GatherSource::FromIndex => {
                    // llama repacked this tensor — upload its canonical bytes from the possession
                    // index. `start` is this tensor's canonical chunk offset (prefix accumulates in
                    // canonical order), so read_chunk_bytes(start + i) yields the right canonical chunk.
                    let idx = idx_ref.unwrap();
                    let start = *prefix.last().unwrap();
                    let mut buf = vec![0u8; chunks as usize * CHUNK_BYTES];
                    for i in 0..chunks as usize {
                        buf[i * CHUNK_BYTES..][..CHUNK_BYTES]
                            .copy_from_slice(&idx.read_chunk_bytes(start + i as u64));
                    }
                    let dev = stream.clone_htod(buf.as_slice()).map_err(candle_core::Error::wrap)?;
                    let ptr = dev.device_ptr(&stream).0 as u64;
                    uploads.push(dev);
                    ptr
                }
            };
            bases.push(base);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: llama engine produced 0 chunks".into()));
        }
        info!(
            "PoM llama canonical gather: {} tensors ({} zero-dup, {} host uploads, {} index uploads{}), N={} chunks",
            bases.len(),
            plan.zero_dup,
            plan.host_uploads,
            plan.index_uploads.len(),
            if plan.index_uploads.is_empty() {
                String::new()
            } else {
                format!(": {}", plan.index_uploads.iter().take(4).cloned().collect::<Vec<_>>().join(", "))
            },
            n_total_chunks
        );
        // BYTE GATE (consensus safety): the pool does not deep-verify every share, so a wrong
        // gather would mine garbage silently. Read back evenly-spaced chunks from the walked device
        // memory and compare them byte-for-byte against the host index (GGUF pread) — any mismatch
        // refuses to mine. (Full-model byte-identity for the zero-dup llama build was proven once by
        // tools/llama_zerodup_spike; this guards every startup and the per-tensor patch.)
        if let Some(idx) = idx_ref {
            if idx.n_chunks == n_total_chunks {
                use candle_core::cuda_backend::cudarc::driver::result as cures;
                let samples = 128u64;
                for k in 0..=samples {
                    let off = if k == samples { n_total_chunks - 1 } else { k * (n_total_chunks / (samples + 1)) };
                    let j = prefix.partition_point(|&p| p <= off) - 1;
                    let dev_addr = bases[j] + (off - prefix[j]) * CHUNK_BYTES as u64;
                    let mut got = [0u8; CHUNK_BYTES];
                    unsafe { cures::memcpy_dtoh_sync(&mut got, dev_addr).map_err(candle_core::Error::wrap)? };
                    let want = idx.read_chunk_bytes(off);
                    if got != want {
                        return Err(candle_core::Error::Msg(format!(
                            "PoM llama byte gate FAILED at chunk {off} — llama-resident bytes differ from the GGUF; refusing to mine"
                        )));
                    }
                }
                info!("PoM llama byte gate: {} sampled chunks match the host index byte-for-byte.", samples + 1);
            }
        }
        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let _ = load_walk_func(&cuda, "pom_mine_v4")?;
        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: Vec::new(), _shared: Vec::new(), _uploads: uploads })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Which v4 walk kernel this device will actually run: "tensor-core" or "classic".
    /// Resolves the full gate (capability + walk-image contents + the runtime stub probe), so it
    /// is the honest answer — not just what the card could do in principle. Logged at startup so
    /// a silent fallback is visible in the log instead of only in the hashrate.
    pub fn v4_walk_kind(&self) -> &'static str {
        let n_tiles = self.n_total_chunks / crate::pom_v4::POM_V4_TILE_CHUNKS;
        if n_tiles == 0 {
            return "classic";
        }
        let pph = [0u8; 32];
        let p = crate::pom::pph_words_for_era(&pph, true);
        let s = crate::pom::pph_words_v4(&pph);
        let k = crate::pom_v4::POM_V4_K as u32;
        if bind_device_ctx(&self.stream).is_err() {
            return "classic";
        }
        if self.v4_tc_usable(n_tiles, k, &p, &s, 0) { "tensor-core" } else { "classic" }
    }

    /// Diagnostic: launch the tensor-core walk ONCE with an always-win target, bypassing the
    /// capability/image gate, and report whether it produced a result. `false` means the loaded
    /// image's `pom_mine_v4_tc` is the empty sub-sm_80 stub (or absent) on this card — the exact
    /// condition that used to mine nothing at a hugely inflated hashrate.
    pub fn v4_tc_kernel_is_real(&self) -> bool {
        let n_tiles = self.n_total_chunks / crate::pom_v4::POM_V4_TILE_CHUNKS;
        if n_tiles == 0 || bind_device_ctx(&self.stream).is_err() {
            return false;
        }
        let pph = [0u8; 32];
        let p = crate::pom::pph_words_for_era(&pph, true);
        let s = crate::pom::pph_words_v4(&pph);
        self.v4_tc_probe(n_tiles, crate::pom_v4::POM_V4_K as u32, &p, &s, 0).unwrap_or(false)
    }

    /// v4 (relaunch) grind: one CUDA block of 32 threads per nonce over `[start, start + batch)`.
    /// Byte-exact mirror of `pom_v4::v4_walk` on the GPU. Returns the lowest winning nonce
    /// (`era_pow_fold(fold64(v4_state_root(S_K))) <= target`), or None. Dynamic shared = 2 KB
    /// (one 1 KB tile + fold scratch). The v4 SEED uses the v4 pph salt; the POW fold stays H3.
    pub fn mine_v4(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> candle_core::Result<Option<u64>> {
        let n_tiles = self.n_total_chunks / crate::pom_v4::POM_V4_TILE_CHUNKS;
        if n_tiles == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: blob too small for the v4 walk".into()));
        }
        // POW fold words = H3-salted pph ("v4 pow uses the h3 fold"); SEED words = v4-salted pph.
        let p = crate::pom::pph_words_for_era(pre_pow_hash, true);
        let s = crate::pom::pph_words_v4(pre_pow_hash);
        let t = words4(target_le);
        let k = crate::pom_v4::POM_V4_K as u32;
        bind_device_ctx(&self.stream)?; // multi-GPU: bind this device's context before the raw launch
        let winner = self.stream.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;

        if self.v4_tc_usable(n_tiles, k, &p, &s, timestamp) {
            // Tensor-core solver: resolve the whole tile-offset chain first (it depends only on
            // tile snippets, never on the walk state), then walk with a depth-3 cp.async tile
            // pipeline + 8x mma.sync.m16n8k32.s8 per step. Byte-exact vs pom_mine_v4/the host;
            // measured +35% on Blackwell. KERYX_POM_V4_TC=0 forces the classic kernel.
            //
            // The offsets buffer is CACHED per device (plain alloc, never zeroed — the chase
            // overwrites every word): the old per-batch alloc_zeros paid a cudaMalloc + a full
            // 16 MB memset + free EVERY batch (~180x/s at batch 16384), measured worth 6-10%.
            let offsets = v4_offsets_buf(&self.stream, (batch as usize) * (k as usize))?;
            // BENCH KNOB: warps/block for the tc walk MUST match the fatbin's V4_TC_WARPS. Default 4;
            // KERYX_POM_V4_TC_WARPS lets a matched test fatbin (built -DV4_TC_WARPS=N) sweep occupancy.
            let tc_warps: u64 = std::env::var("KERYX_POM_V4_TC_WARPS").ok()
                .and_then(|s| s.parse().ok()).filter(|&w: &u64| w >= 1 && w <= 16).unwrap_or(4);
            // smem/warp = state(256) + PIPE tiles. PIPE default 3 → 4 buffers × 256 words × 4 B = 4096.
            let tc_pipe: u64 = std::env::var("KERYX_POM_V4_TC_PIPE").ok()
                .and_then(|s| s.parse().ok()).filter(|&p: &u64| p >= 2 && p <= 8).unwrap_or(3);
            let tc_smem: u32 = (tc_warps * 256 * (tc_pipe + 1) * 4) as u32;
            #[allow(non_snake_case)]
            let TC_WARPS: u64 = tc_warps;
            let chase = load_walk_func(&self.cuda, "pom_mine_v4_chase")?;
            let walk = load_walk_func(&self.cuda, "pom_mine_v4_tc")?;

            // Sub-batch pipelining (KERYX_POM_V4_OVERLAP=0 to disable): chase sub-batch k+1 on a
            // second stream while sub-batch k walks — the walk stream waits per sub-batch on an
            // event. The two phases share DRAM/SMs so this only merges (not hides) the chase, but
            // it is still a measured ~+2% (bench C4, bit-exact 2048/2048). Skipped for small
            // batches where sub-batches would be tiny.
            const SUBS: u64 = 4;
            let overlap = std::env::var("KERYX_POM_V4_OVERLAP").ok().as_deref() != Some("0")
                && batch >= SUBS * 4096;
            let n_sub = if overlap { SUBS } else { 1 };
            let sub = (batch + n_sub - 1) / n_sub;
            let chase_stream = if overlap { Some(v4_chase_stream(&self.stream)?) } else { None };

            for i in 0..n_sub {
                let s_start = start + i * sub;
                let s_batch = sub.min(batch - i * sub);
                let view = offsets
                    .try_slice((i * sub) as usize * (k as usize)..((i * sub + s_batch) as usize * k as usize))
                    .ok_or_else(|| candle_core::Error::Msg("PoM v4: offsets slice out of range".into()))?;
                let chase_cfg = LaunchConfig {
                    grid_dim: (((s_batch + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                if let Some(cs) = &chase_stream {
                    let func = chase.func.clone();
                    let mut b = cs.launch_builder(&func);
                    b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&n_tiles).arg(&k)
                        .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
                        .arg(&s_start).arg(&s_batch).arg(&view);
                    unsafe { b.launch(chase_cfg).map_err(candle_core::Error::wrap)?; }
                    let ev = cs.record_event(None).map_err(candle_core::Error::wrap)?;
                    self.stream.wait(&ev).map_err(candle_core::Error::wrap)?;
                } else {
                    let mut b = chase.builder();
                    b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&n_tiles).arg(&k)
                        .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
                        .arg(&s_start).arg(&s_batch).arg(&view);
                    unsafe { b.launch(chase_cfg).map_err(candle_core::Error::wrap)?; }
                }

                let walk_cfg = LaunchConfig {
                    grid_dim: (((s_batch + TC_WARPS - 1) / TC_WARPS) as u32, 1, 1),
                    block_dim: ((TC_WARPS * 32) as u32, 1, 1),
                    shared_mem_bytes: tc_smem, // (PIPE+1) buffers × 256 words × 4B per warp
                };
                let mut b = walk.builder();
                b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&k)
                    .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3])
                    .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
                    .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
                    .arg(&s_start).arg(&s_batch).arg(&view).arg(&winner);
                unsafe { b.launch(walk_cfg).map_err(candle_core::Error::wrap)?; }
            }
        } else {
            let cfg = LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (crate::pom_v4::POM_V4_D as u32, 1, 1),
                shared_mem_bytes: 2048,
            };
            let func = load_walk_func(&self.cuda, "pom_mine_v4")?;
            let mut b = func.builder();
            b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&n_tiles).arg(&k)
                .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3])
                .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
                .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
                .arg(&start).arg(&batch).arg(&winner);
            unsafe { b.launch(cfg).map_err(candle_core::Error::wrap)?; }
        }
        self.stream.synchronize().map_err(candle_core::Error::wrap)?;
        let w = self.stream.clone_dtoh(&winner).map_err(candle_core::Error::wrap)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }

    /// Whether the tensor-core v4 solver can run on this device: needs real int8
    /// mma SASS (sm_80+ — below that the module carries only the stub) and not
    /// force-disabled via `KERYX_POM_V4_TC=0`. Logged once per device.
    fn v4_tc_available(&self) -> bool {
        if std::env::var("KERYX_POM_V4_TC").ok().as_deref() == Some("0") {
            return false;
        }
        // The DEVICE having sm_80+ is not enough: what matters is whether the WALK IMAGE we
        // actually loaded can supply sm_80+ code for `pom_mine_v4_tc`. The kernel is compiled
        // under `#if __CUDA_ARCH__ >= 800`; below that it is an EMPTY STUB. A PTX image
        // generated at compute_70/75 (the legacy/Pascal packages, POM_WALK_IMAGE=ptx, or a
        // source build without the committed fatbin) therefore carries ONLY that stub — the
        // driver JITs the stub for the Ampere card and the "walk" writes nothing at all:
        // no shares, and a hashrate inflated by however fast an empty kernel returns.
        // That combination (Ampere + legacy package — which is what a rig containing pre-sm_75
        // cards must run) is exactly the silent failure reported from the field.
        let image_kind = env!("POM_WALK_IMAGE_KIND");
        let image_arch = env!("POM_PTX_ARCH");
        let image_can_tc = image_kind == "fatbin"
            || image_arch
                .trim_start_matches("sm_")
                .split(|c: char| !c.is_ascii_digit())
                .next()
                .and_then(|d| d.parse::<u32>().ok())
                .map(|sm| sm >= 80)
                .unwrap_or(false);
        match self.stream.context().compute_capability() {
            Ok((major, _)) => {
                let ok = major >= 8 && image_can_tc;
                static LOGGED: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
                let mut seen = LOGGED.get_or_init(|| Mutex::new(std::collections::HashSet::new())).lock().unwrap();
                if seen.insert(self.stream.context().ordinal() as u32) {
                    if ok {
                        log::info!("PoM v4: tensor-core solver active (sm_{}x, chase + mma.m16n8k32 pipeline).", major);
                    } else if major >= 8 {
                        log::warn!(
                            "PoM v4: sm_{}x supports the tensor-core solver but this build's walk image \
cannot ({} {}) — it carries only the sub-sm_80 stub. Using the classic dp4a kernel (correct, ~25% slower). \
Use the MODERN package (native sm_80-120 walk) on Ampere and newer.",
                            major, image_kind, image_arch);
                    } else {
                        log::info!("PoM v4: sm_{}x has no int8 mma — using the classic dp4a kernel.", major);
                    }
                }
                ok
            }
            Err(_) => false,
        }
    }

    /// Authoritative gate for the tensor-core solver: `v4_tc_available()` (capability + image
    /// kind) AND a one-off runtime PROBE per device.
    ///
    /// The probe exists because "the image contains sm_80+ code" cannot be fully decided at
    /// compile time: the committed fatbin carries native SASS for sm_75;80;86;89;90;120 only, so
    /// a card outside that list (sm_87 Orin, sm_100/103 datacenter Blackwell) silently JITs the
    /// embedded compute_75 PTX — i.e. the EMPTY STUB again — and a stale fatbin may not export
    /// the symbol at all. Both fail the same silent way: no winners, fantasy hashrate.
    ///
    /// The probe runs the real chase+walk for ONE nonce against an all-0xFF (always-win) target:
    /// any real kernel must record that win. A stub writes nothing. On failure the device is
    /// demoted to the classic dp4a kernel for the process lifetime, loudly.
    fn v4_tc_usable(&self, n_tiles: u64, k: u32, p: &[u64; 4], s: &[u64; 4], timestamp: u64) -> bool {
        if !self.v4_tc_available() {
            return false;
        }
        static PROBED: OnceLock<Mutex<std::collections::HashMap<u32, bool>>> = OnceLock::new();
        let cache = PROBED.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let ord = self.stream.context().ordinal() as u32;
        if let Some(&ok) = cache.lock().unwrap().get(&ord) {
            return ok;
        }
        let ok = match self.v4_tc_probe(n_tiles, k, p, s, timestamp) {
            Ok(true) => {
                log::info!("PoM v4: tensor-core solver PROBE OK on GPU {} (1-nonce known-win recorded).", ord);
                true
            }
            Ok(false) => {
                log::error!(
                    "PoM v4: tensor-core solver PROBE FAILED on GPU {} — the walk kernel produced NO result \
for an always-win target, i.e. this build's image has no real sm_80+ code for this card (stub). \
Falling back to the classic dp4a kernel so the card mines CORRECTLY. If this is an Ampere or newer \
card, use the MODERN package to get the fast walk.", ord);
                false
            }
            Err(e) => {
                log::error!(
                    "PoM v4: tensor-core solver PROBE ERROR on GPU {} ({e}) — falling back to the classic \
dp4a kernel.", ord);
                false
            }
        };
        cache.lock().unwrap().insert(ord, ok);
        ok
    }

    /// One-nonce chase+walk with an always-win target. Ok(true) = a real TC kernel ran.
    fn v4_tc_probe(&self, n_tiles: u64, k: u32, p: &[u64; 4], s: &[u64; 4], timestamp: u64)
        -> candle_core::Result<bool>
    {
        const TC_WARPS: u64 = 4;
        let chase = load_walk_func(&self.cuda, "pom_mine_v4_chase")?;
        let walk = load_walk_func(&self.cuda, "pom_mine_v4_tc")?;   // missing symbol -> Err -> classic
        let offsets = v4_offsets_buf(&self.stream, k as usize)?;
        let winner = self.stream.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;
        let (start, batch) = (0u64, 1u64);
        let view = offsets
            .try_slice(0..k as usize)
            .ok_or_else(|| candle_core::Error::Msg("PoM v4 probe: offsets slice".into()))?;
        let mut b = chase.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&n_tiles).arg(&k)
            .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
            .arg(&start).arg(&batch).arg(&view);
        unsafe {
            b.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .map_err(candle_core::Error::wrap)?;
        }
        let t_max = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];   // all-0xFF target: any real walk wins
        let mut b = walk.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&k)
            .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3])
            .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
            .arg(&t_max[0]).arg(&t_max[1]).arg(&t_max[2]).arg(&t_max[3])
            .arg(&start).arg(&batch).arg(&view).arg(&winner);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: ((TC_WARPS * 32) as u32, 1, 1),
                shared_mem_bytes: (TC_WARPS as u32) * 4096,
            }).map_err(candle_core::Error::wrap)?;
        }
        self.stream.synchronize().map_err(candle_core::Error::wrap)?;
        let w = self.stream.clone_dtoh(&winner).map_err(candle_core::Error::wrap)?[0];
        Ok(w != u64::MAX)
    }

}

// Per-GPU PoM miners. Host-side WeightIndex remains shared; only the CUDA-resident worker state
// is duplicated per device. This avoids all workers contending over a single GPU0-bound miner.
fn miners() -> &'static Mutex<HashMap<u32, Arc<PomGpuMiner>>> {
    static MINERS: OnceLock<Mutex<HashMap<u32, Arc<PomGpuMiner>>>> = OnceLock::new();
    MINERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// CUDA ordinals this process's PoM walk is installed on (its `--cuda-device` set), ascending.
/// Empty until the first PoM-active job installs a miner. Used to place OPoI inference on the
/// process's OWN card(s) instead of a global "biggest" GPU — see `slm::inference_gpu_ordinal`.
pub fn walk_devices() -> Vec<u32> {
    let mut v: Vec<u32> = miners()
        .lock()
        .map(|g| g.keys().copied().collect())
        .unwrap_or_default();
    v.sort_unstable();
    v
}

// Guards the one-time shared host index build. All workers may race into PoM activation, but the
// heavy GGUF -> WeightIndex build must happen exactly once for the process.
fn index_build_lock() -> &'static Mutex<()> {
    static INDEX_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    INDEX_BUILD_LOCK.get_or_init(|| Mutex::new(()))
}

/// Install the GPU miner for a specific CUDA device.
pub fn install(device_id: u32, m: PomGpuMiner) {
    if let Ok(mut g) = miners().lock() {
        g.insert(device_id, Arc::new(m));
    }
}

/// Removes only `device_id`'s entry from a `device -> miner` map, leaving every other device's
/// entry untouched. Pulled out as a tiny generic helper (over the map's value type) purely so
/// this scoping behavior is unit-testable without a real, CUDA-backed `PomGpuMiner` — production
/// always calls it through `uninstall` against `HashMap<u32, Arc<PomGpuMiner>>`.
fn remove_device_entry<T>(map: &mut HashMap<u32, T>, device_id: u32) -> Option<T> {
    map.remove(&device_id)
}

/// Block until `item` is the only remaining handle, or the deadline passes; returns whether the
/// wait succeeded. Upstream aa29fd2 — drains an in-flight walk that cloned the miner Arc before we
/// let the caller free the tensors that walk reads.
fn wait_for_sole_owner<T>(item: &Arc<T>, timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while Arc::strong_count(item) > 1 {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    true
}

/// Drop the GPU miner for `device_id` only, releasing its hold on that device's mining-model VRAM
/// (shared Arcs + gather) so the inference engine can load another model there. Mining on that
/// device is paused during inference anyway.
///
/// Scoped to a single device on purpose: only the device colocated with inference (CUDA device 0
/// — see the `Device::new_cuda(0)` call in `slm::load_and_run_inference`) ever shares VRAM with
/// the inference engine via `load_shared`'s zero-dup path, or otherwise needs to make room for an
/// inference model swap. Other devices in a multi-GPU rig run fully standalone `PomGpuMiner`s
/// (`PomGpuMiner::load`) that never touch the inference engine's VRAM. A previous version of this
/// function called `g.clear()`, dropping every device's resident miner on every inference model
/// swap — needlessly forcing GPU1+ rigs to fully reload their GGUF from disk and rebuild the
/// gather index (`ensure_installed_inner`'s own doc comment calls this reload "Heavy") even though
/// nothing about them changed.
pub fn uninstall(device_id: u32) {
    let removed = match miners().lock() {
        Ok(mut g) => remove_device_entry(&mut g, device_id),
        Err(_) => None,
    };
    // BARRIER before the caller frees any VRAM this miner walks over (upstream aa29fd2): a mining
    // thread clones the handle out of the map and launches OUTSIDE the map lock, so removing the
    // entry does not stop an in-flight walk. Its launch synchronizes before it drops its handle, so
    // waiting for the last handle to drop is enough. Freeing under a live walk raises a sticky
    // CUDA_ERROR_ILLEGAL_ADDRESS that poisons the device's context for every user of it, inference
    // included.
    if let Some(miner) = removed {
        if !wait_for_sole_owner(&miner, std::time::Duration::from_secs(30)) {
            log::error!("PoM[gpu{}]: a walk still holds the miner after 30s — releasing anyway", device_id);
        }
    }
}

/// Whether the GPU miner is currently installed for `device_id`.
pub fn is_installed(device_id: u32) -> bool {
    miners().lock().map(|g| g.contains_key(&device_id)).unwrap_or(false)
}

/// True while the GPU miner is being (re)built — a heavy one-time model load that blocks the
/// mining worker. The PoW stall watchdog treats this like an inference pause, not a crash.
static LOADING: AtomicUsize = AtomicUsize::new(0);

/// Whether a PoM model load/rebuild is in progress (worker intentionally paused, not stalled).
pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed) > 0
}

/// Hard inference gate (ported from upstream Keryx-Labs/keryx-miner d35f85fc, adapted to our
/// per-GPU pool). Raised before an OPoI model swap frees a card's walk, lowered once the swap is
/// done. While raised, `mine()` returns immediately so the worker starts no NEW walk batch — that
/// lets `uninstall()`'s `wait_for_sole_owner` barrier see the last handle drop and free VRAM
/// cleanly, instead of timing out and freeing under a live walk (the CUDA_ERROR_ILLEGAL_ADDRESS
/// "inference while the model is loading" crash on mixed rigs).
static INFERENCE_PAUSED: AtomicBool = AtomicBool::new(false);

pub fn set_inference_paused(paused: bool) {
    INFERENCE_PAUSED.store(paused, Ordering::Release);
}

pub fn inference_paused() -> bool {
    INFERENCE_PAUSED.load(Ordering::Acquire)
}

/// Transient GPU runtime fault classifier (upstream Keryx-Labs/keryx-miner@278098b): the
/// ILLEGAL_ADDRESS class — a kernel dereferenced garbage (driver hiccup, ECC blip, a stale
/// gather pointer after an inference eviction race). The faulting CUDA context is poisoned but
/// the DEVICE is fine: dropping every resource bound to that context and rebuilding recovers it.
/// Explicitly NOT transient: OOM (retrying the same load just fails again — the existing
/// fallback/eviction logic handles it) and the box-wide poison class (CUDA_ERROR_DEINITIALIZED /
/// Xid-79 "fell off the bus" — no rebuild can help, existing behavior/reboot applies).
///
/// Error-string sources this must match (all funnel through `candle_core::Error` `Display`):
/// cudarc `DriverError` Display/Debug prints the enum name + driver text, e.g.
/// `DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")` —
/// wrapped unchanged by `candle_core::Error::wrap` in `PomGpuMiner::{mine,load*}`.
fn is_transient_gpu_runtime_fault(err: &str) -> bool {
    let s = err.to_ascii_lowercase();
    s.contains("illegal address")
        || s.contains("illegal memory")
        || s.contains("cuda_error_illegal_address")
        || s.contains("invalid device pointer")
        || s.contains("misaligned address")
}

/// Drop everything the faulted GPU's next cycle would otherwise reuse: its walk miner (gather
/// pointers into now-poisoned context memory) and — only if THIS gpu hosts it — the in-process
/// llama engine (the zero-dup walk gathers over its resident tensors; `unload_for_gpu` is a
/// scoped no-op when the engine lives on another card, so other GPUs keep mining untouched).
/// Order matters: uninstall the walk FIRST so no gather can run over tensors llama then frees.
/// The worker loop rebuilds via `ensure_installed` on its next iteration (`is_installed` = false).
fn reset_stale_gpu_state(device_id: u32) {
    uninstall(device_id);
    crate::llama_engine::unload_for_gpu(device_id as usize);
}

/// TEST-ONLY fault injection — dev knob, never set this in production. `KERYX_FAULT_INJECT_GPU=
/// <cuda ordinal>` makes the NEXT mine() call on that ordinal report a synthetic transient fault
/// (ILLEGAL_ADDRESS class) exactly ONCE per process (armed at first check, consumed by an
/// AtomicBool swap), so the per-GPU recovery path can be live-validated on a healthy rig: the
/// injected fault must drop that GPU's state and rebuild next cycle while other GPUs keep mining.
fn take_injected_fault(device_id: u32) -> bool {
    static TARGET: OnceLock<Option<u32>> = OnceLock::new();
    static ARMED: AtomicBool = AtomicBool::new(true);
    let target = *TARGET.get_or_init(|| {
        std::env::var("KERYX_FAULT_INJECT_GPU").ok().and_then(|s| s.trim().parse::<u32>().ok())
    });
    matches!(target, Some(t) if t == device_id) && ARMED.swap(false, Ordering::Relaxed)
}

/// Convenience: v4 (relaunch) grind via the installed miner for a specific device. Same transient-
/// fault teardown path as `mine`; block autotune is irrelevant (v4 is one block per nonce).
/// v4 grind-batch sizing from the GPU's SM count (ported from upstream keryx-miner PR #37,
/// GerardMensoif): the throughput plateau is broad from ~8K nonces upward (our 5090 bench: 64K/128K/
/// 256K all within noise), so the batch's real job is to keep ONE launch well inside a template
/// window at ~10 blocks/s — a fixed 64K over-runs the 100ms window on SMALL cards, wasting a whole
/// batch on a stale job. Sizing per-SM keeps launch wall-time roughly card-independent: bigger cards
/// grind more nonces in the same time, small cards fewer, both landing fresh. Env
/// KERYX_POM_V4_BATCH still overrides absolutely.
const POM_V4_NONCES_PER_SM: u64 = 384;
const POM_V4_BATCH_MIN: u64 = 8192;
const POM_V4_BATCH_FALLBACK: u64 = 32768;

fn gpu_sm_count(device_id: u32) -> Option<u64> {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};
    result::init().ok()?;
    let dev = result::device::get(device_id as i32).ok()?;
    let n = unsafe {
        result::device::get_attribute(dev, sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    }
    .ok()?;
    (n > 0).then_some(n as u64)
}

fn v4_batch_for_sm_count(sm: u64) -> u64 {
    (sm * POM_V4_NONCES_PER_SM).max(POM_V4_BATCH_MIN)
}

/// The v4 grind batch to use on `device_id` (SM-derived, floored). Logged once per device.
pub fn v4_batch_for_device(device_id: u32) -> u64 {
    let b = match gpu_sm_count(device_id) {
        Some(sm) => v4_batch_for_sm_count(sm),
        None => POM_V4_BATCH_FALLBACK,
    };
    static LOGGED: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    if let Ok(mut seen) = LOGGED.get_or_init(|| Mutex::new(std::collections::HashSet::new())).lock() {
        if seen.insert(device_id) {
            log::info!("PoM[gpu{}]: v4 grind batch = {} nonces (SM-derived).", device_id, b);
        }
    }
    b
}

pub fn mine_v4(device_id: u32, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> Option<u64> {
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    match miner.mine_v4(pre_pow_hash, timestamp, target_le, start, batch) {
        Ok(w) => w,
        Err(e) => {
            let msg = e.to_string();
            if is_transient_gpu_runtime_fault(&msg) {
                log::warn!("PoM[gpu{}]: TRANSIENT GPU FAULT during the v4 walk ({}) — rebuilding next cycle.", device_id, msg);
                reset_stale_gpu_state(device_id);
            } else {
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    log::error!("PoM[gpu{}]: v4 walk failed with a NON-transient error ({}). Logged once.", device_id, msg);
                }
            }
            None
        }
    }
}

/// Mining-tier identity for rebuilds: (model_id, gguf_path). Set once at startup.
static MINING_TIER: OnceLock<([u8; 32], String)> = OnceLock::new();

/// Record the mining tier so the miner can be rebuilt after an inference swapped the model away.
/// This is the PROCESS-WIDE DEFAULT model (single-model rigs, --light/--very-high/etc.). Per-device
/// overrides (mixed rigs / --force-model) go through `set_device_model`; `device_model()` prefers a
/// per-device entry and falls back to this default, so the single-model path is byte-identical.
pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String) {
    let _ = MINING_TIER.set((model_id, gguf_path));
}

/// Per-CUDA-device mining model (model_id, gguf_path) — populated only for mixed-rig per-card
/// best-fit / `--force-model`. Empty on single-model rigs (they use `MINING_TIER`).
fn device_models() -> &'static Mutex<HashMap<u32, ([u8; 32], String)>> {
    static DEVICE_MODELS: OnceLock<Mutex<HashMap<u32, ([u8; 32], String)>>> = OnceLock::new();
    DEVICE_MODELS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Assign a specific model to one CUDA device (per-card best-fit or `--force-model`).
pub fn set_device_model(device_id: u32, model_id: [u8; 32], gguf_path: String) {
    if let Ok(mut m) = device_models().lock() {
        m.insert(device_id, (model_id, gguf_path));
    }
}

/// Devices whose model was pinned by `--force-model`. A forced device is NEVER auto-demoted on an
/// OOM — `--force-model` means "load exactly this, no VRAM check", so we honor it even if it OOMs.
fn forced_devices() -> &'static Mutex<std::collections::HashSet<u32>> {
    static F: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

/// Mark `device_id` as `--force-model`-pinned (set from the CLI). Suppresses OOM auto-demotion.
pub fn set_device_forced(device_id: u32) {
    if let Ok(mut f) = forced_devices().lock() { f.insert(device_id); }
}

/// Whether this device's model was pinned via `--force-model`.
pub fn is_device_forced(device_id: u32) -> bool {
    forced_devices().lock().map(|f| f.contains(&device_id)).unwrap_or(false)
}

/// The model this device mines: its per-device override if set, else the process-wide default.
pub fn device_model(device_id: u32) -> Option<([u8; 32], String)> {
    if let Ok(m) = device_models().lock() {
        if let Some(v) = m.get(&device_id) {
            return Some(v.clone());
        }
    }
    MINING_TIER.get().cloned()
}

/// Every per-card model assignment (model_id, gguf_path) — the distinct tiers a mixed rig serves
/// across its cards. Used to build the declare-capabilities UNION so a smaller per-card tier that
/// is NOT in the process-wide lineup is still declared servable. Empty on a uniform single-model rig.
pub fn assigned_models() -> Vec<([u8; 32], String)> {
    device_models().lock().map(|m| m.values().cloned().collect()).unwrap_or_default()
}

/// True if this device has its OWN model (mixed rig / --force-model) vs using the shared default.
/// Decides whether it builds/uses a per-device possession index or the process-wide shared one.
fn has_device_override(device_id: u32) -> bool {
    device_models().lock().map(|m| m.contains_key(&device_id)).unwrap_or(false)
}

/// Total VRAM (MiB) of every CUDA device in **CUDA device order** (the order `Device::new_cuda(id)`
/// and the miner's worker ids use) — sourced from the CUDA driver, NOT nvidia-smi line order (which
/// can map to the wrong card). Empty vec if no CUDA driver (CPU-only / AMD). Never panics.
/// (upstream Keryx-Labs/keryx-miner@cb7f81c)
/// Per-GPU (ordinal, FREE MiB, TOTAL MiB) via the driver API. FREE is what a tier pick can
/// actually allocate: budgeting against TOTAL (the old behavior) chose tiers that cudaMalloc'd
/// straight into an OOM whenever anything else was resident on the card (another miner, a leaked
/// context, a desktop session) — which then cascaded into self-test failures and, pre-failover,
/// a whole-rig suspend. Uses each device's PRIMARY context (retain → mem_get_info → release), so
/// it neither creates spurious contexts nor disturbs live ones. Ordinals are CUDA ordinals —
/// same space as every other pom_gpu call.
pub fn query_all_gpus_free_vram() -> Vec<(u32, u64, u64)> {
    use candle_core::cuda_backend::cudarc::driver::result;
    std::panic::catch_unwind(|| {
        if result::init().is_err() {
            return Vec::new();
        }
        let count = result::device::get_count().unwrap_or(0);
        let mut out = Vec::with_capacity(count.max(0) as usize);
        for ordinal in 0..count {
            let Ok(dev) = result::device::get(ordinal) else { continue };
            // SAFETY: `dev` is a valid handle from device::get; retain/set_current/release are
            // the documented primary-ctx sequence and mem_get_info runs under that context.
            let got = unsafe {
                result::primary_ctx::retain(dev).ok().and_then(|ctx| {
                    let r = result::ctx::set_current(ctx)
                        .ok()
                        .and_then(|_| result::mem_get_info().ok());
                    let _ = result::primary_ctx::release(dev);
                    r
                })
            };
            if let Some((free_b, total_b)) = got {
                out.push((ordinal as u32, (free_b / (1024 * 1024)) as u64, (total_b / (1024 * 1024)) as u64));
            }
        }
        out
    })
    .unwrap_or_default()
}

pub fn query_all_gpus_vram() -> Vec<(u32, u64)> {
    use candle_core::cuda_backend::cudarc::driver::result;
    std::panic::catch_unwind(|| {
        if result::init().is_err() {
            return Vec::new();
        }
        let count = result::device::get_count().unwrap_or(0);
        let mut out = Vec::with_capacity(count.max(0) as usize);
        for ordinal in 0..count {
            let Ok(dev) = result::device::get(ordinal) else { continue };
            // SAFETY: `dev` is a valid handle just returned by `device::get(ordinal)`.
            if let Ok(bytes) = unsafe { result::device::total_mem(dev) } {
                out.push((ordinal as u32, (bytes / (1024 * 1024)) as u64));
            }
        }
        out
    })
    .unwrap_or_default()
}

/// Ensure the GPU miner is installed; if an inference evicted the mining model, reload it
/// (resident again) and rebuild the zero-dup gather. Heavy (model reload) but only when needed —
/// inference has priority, so mining reloads its model when it next gets the GPU. Returns true if
/// the miner is ready to mine.
/// `--low-ram`/`--save-ram`: serialize model bring-up across GPUs so peak SYSTEM RAM ≈ ONE model,
/// not N. Each card's possession-index build + llama GGUF load pulls the multi-GB weights through
/// host RAM; bringing up many cards in parallel spikes system RAM and OOMs low-RAM rigs. When set,
/// a global lock loads one card FULLY (index + walk + self-test) before the next starts. Slower
/// startup, far lower peak RAM. Off by default (parallel bring-up).
static LOW_RAM: AtomicBool = AtomicBool::new(false);
pub fn set_low_ram(on: bool) { LOW_RAM.store(on, Ordering::Relaxed); }
pub fn low_ram() -> bool { LOW_RAM.load(Ordering::Relaxed) }
fn load_lock() -> &'static Mutex<()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
}

pub fn ensure_installed(device_id: u32, daa: u64) -> bool {
    if is_installed(device_id) {
        return true;
    }
    // Flag the heavy load so the stall watchdog stays benign while the worker is blocked here.
    LOADING.fetch_add(1, Ordering::Relaxed);
    // --low-ram: one card's full model bring-up at a time (peak host RAM = one model, not N).
    let _load_guard = if low_ram() {
        Some(load_lock().lock().unwrap_or_else(|p| p.into_inner()))
    } else {
        None
    };
    let ok = ensure_installed_inner(device_id, daa);
    LOADING.fetch_sub(1, Ordering::Relaxed);
    ok
}

/// PoM tier index of the mining model at a given block DAA. Recomputed per block (not frozen at
/// index-build time) so the tier reindexing at the very-light hardfork (H2) is applied at the
/// exact boundary — e.g. Gemma 0→1 — rather than from a stale build-time value.
pub fn current_tier(device_id: u32, daa: u64) -> Option<u8> {
    let (model_id, _) = device_model(device_id)?;
    crate::models::pom_tier_index(&model_id, daa)
}

/// CUDA ordinal of a candle device (None if not CUDA) — used to check whether the inference
/// engine's resident model lives on the same GPU as the PoM miner we're about to install, before
/// sharing its tensors in place.
fn cuda_gpu_id(d: &Device) -> Option<usize> {
    match d.location() {
        candle_core::DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
        _ => None,
    }
}

fn ensure_installed_inner(device_id: u32, daa: u64) -> bool {
    // This device's model: its own (mixed rig / --force-model) or the process-wide default.
    let (model_id, gguf) = match device_model(device_id) {
        Some(x) => x,
        None => return false,
    };
    let per_device = has_device_override(device_id);
    // Single-model rigs share ONE host index (built once); per-card rigs build one index PER device.
    let index_ready = if per_device { crate::pom::has_device_index(device_id) } else { crate::pom::active_index().is_some() };
    // Build the possession index (host, heavy) the first time PoM activates — deferred from boot so
    // the pre-PoM legacy phase starts immediately and keeps host/GPU free.
    if !index_ready {
        let _guard = match index_build_lock().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let still_missing = if per_device { !crate::pom::has_device_index(device_id) } else { crate::pom::active_index().is_none() };
        if still_missing {
            // The background prefetch may still be downloading this device's model (slow IPFS
            // link / small HiveOS system disk). Building the index from a missing/partial GGUF
            // hard-fails with ENOENT ("index build failed: no such file or directory") and would
            // spam that on every job. Wait for the `.ok` completion sentinel and retry next job.
            let ready = std::path::Path::new(&gguf)
                .parent()
                .map(|d| d.join(".ok"))
                .map_or(false, |p| p.exists());
            if !ready {
                static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    info!(
                        "PoM: mining-tier model (gpu{}) not downloaded yet — deferring the possession-index \
                         build until the background prefetch finishes (slow link / small disk).", device_id
                    );
                }
                return false;
            }
            let tier = match crate::models::pom_tier_index(&model_id, daa) {
                Some(t) => t,
                None => return false,
            };
            info!("PoM: building host weight index (gpu{}) — this can take a while…", device_id);
            match crate::pom::WeightIndex::build_from_gguf(&gguf, model_id) {
                Ok(mut idx) => {
                    // Opt-in solo block-race edge (upstream 7a6e7a0): hold the FULL Merkle tree in
                    // RAM so the post-hit proof build is a pure lookup instead of a ~30-40 ms
                    // sparse recompute — at 10 BPS that latency measurably loses chain races.
                    // Costs ~2N*32 B RAM (~9.6 GB at tier-0), hence opt-in; low-RAM rigs keep the
                    // frugal sparse path. Gated by --resident-tree / KERYX_RESIDENT_TREE=1 (resolved in main).
                    if crate::pom::resident_tree_enabled() {
                        let t0 = std::time::Instant::now();
                        info!("PoM[gpu{}]: building RESIDENT tree (RAM) — proof build becomes lookup-time…", device_id);
                        idx.build_dense();
                        info!("PoM[gpu{}]: resident tree ready in {:.1}s", device_id, t0.elapsed().as_secs_f32());
                    }
                    info!("PoM[gpu{}]: host index ready — N={} chunks", device_id, idx.n_chunks);
                    if per_device { crate::pom::set_index_for(device_id, idx, tier); } else { crate::pom::set_index(idx, tier); }
                }
                Err(e) => {
                    // The build is retried on every job while the index is missing (e.g. disk too
                    // small for the tree). Rate-limit the log to ~once/5 min so the actionable reason
                    // (like the disk pre-check message) is visible without flooding the log.
                    static LAST_LOG_SECS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    if now.saturating_sub(LAST_LOG_SECS.load(Ordering::Relaxed)) >= 300 {
                        LAST_LOG_SECS.store(now, Ordering::Relaxed);
                        log::error!("PoM: possession-index build failed on gpu{}: {} (retrying each job; this message is rate-limited to ~5 min).", device_id, e);
                    }
                    return false;
                }
            }
        }
    }
    // One CUDA-resident PoM worker per GPU. This avoids all workers contending for a single
    // GPU0-bound miner object while still sharing the host-side index across the process.
    //
    // Zero-dup on the inference GPU: if the inference engine holds THIS exact model resident on
    // THIS device (split loader + `pom_force_split`), the walk shares its quantized tensors in
    // place (`load_shared`) rather than loading a second full VRAM copy — saving ~one model's
    // worth of VRAM on the serving GPU. Mining-only GPUs (no resident inference model to share)
    // fall back to a standalone copy. The N-guard below validates the gather against the host
    // index on every path, so a mismatch refuses to mine rather than producing bad proofs.
    // Phase 2 (candle-independence): if the in-process llama.cpp engine holds THIS model on THIS
    // device (or can be brought up — .so bundled/env-pointed), the walk gathers over ITS resident
    // tensors and candle hosts nothing. ensure_loaded is idempotent/cheap when already active and
    // self-disables (returns false) when no .so is present — then the candle paths below apply.
    let inference_gpu = crate::slm::inference_gpu_ordinal();
    let mut use_llama = false;
    if device_id as usize == inference_gpu {
        use_llama = crate::llama_engine::ensure_loaded(&gguf, inference_gpu);
        // Upstream 0795e92: only this GPU can serve the model — no engine here means no inference
        // anywhere, so reflect that in ai:cap instead of announcing a model we cannot answer for.
        match use_llama {
            true => crate::slm::mark_model_available(&model_id, "llama_engine_loaded"),
            false => crate::slm::mark_model_unavailable(&model_id, "llama_engine_load_failed"),
        }
    }
    // OWNERSHIP GATE (upstream aa29fd2): the walk dereferences llama's tensor pointers on THIS
    // device. If llama placed them on another card, the launch hits unmapped memory and raises a
    // sticky CUDA_ERROR_ILLEGAL_ADDRESS that poisons the primary context for every user of the
    // device, llama included — the card then loops on rebuilds until the process restarts.
    // (Scoped `unload_for_gpu` here, vs upstream's singleton `unload`, to match our per-GPU pool.)
    if use_llama {
        if let Some((name, owner)) = crate::llama_engine::foreign_device_tensor(device_id as usize) {
            log::warn!(
                "PoM[gpu{}]: llama placed '{}' on device {} — walking a raw canonical copy; inference for this model is unavailable.",
                device_id, name, owner
            );
            crate::llama_engine::unload_for_gpu(device_id as usize);
            use_llama = false;
            crate::slm::mark_model_unavailable(&model_id, "llama_wrong_device");
        }
    }
    // BYTE-COMPAT GATE: llama.cpp repacks some architectures on load (e.g. Gemma materialises a
    // separate output.weight from its tied embeddings), so its resident layout can differ from the
    // canonical GGUF the walk MUST gather and R_T pins. The canonical gather reconciles that PER
    // TENSOR (repacked/duplicated/dropped tensors are uploaded from the possession index while the
    // rest stay zero-dup); only a layout too foreign to reconcile within the upload budget (>25% of
    // the blob) falls back to a full raw canonical copy WITHOUT llama (inference then unavailable).
    // This keeps a repacking arch (Gemma) resident as one copy — walk + inference — on a 16 GB card.
    if use_llama {
        if let Some(reason) = llama_gather_blocker(&gguf, device_id as usize) {
            info!(
                "PoM[gpu{}]: {} — walking a raw canonical copy; inference for this model is unavailable.",
                device_id, reason
            );
            crate::llama_engine::unload_for_gpu(device_id as usize);
            use_llama = false;
            // This model's llama layout can't back the walk here — withdraw it from ai:cap.
            crate::slm::mark_model_unavailable(&model_id, "llama_layout_incompatible");
        }
    }
    // SERVEABILITY GATE — the network exists to serve inference, so we NEVER mine a tier we cannot
    // serve (mining it would expose the pool to inference strikes). The serving GPU hosts EVERY model
    // (it swaps on demand), so we prove THIS device's model by a one-time inference self-test THERE,
    // whichever card mines the tier — this covers mixed rigs (e.g. a 3070 mining Qwen while the big
    // card serves GLM). The result is cached per model, so only the first card per model generates; a
    // model that fails is withdrawn from ai:cap and NOT mined on any card. (`inference_gpu` above.)
    if !crate::slm::run_inference_self_test(&model_id, inference_gpu) {
        // The tier's model could not be SERVED (inference self-test failed) — usually the serving GPU
        // can't fit this tier alongside the model it already hosts (mixed rig), too little VRAM, a bad
        // file, or a GPU fault. We never mine a tier we cannot serve. DEMOTE to the next smaller
        // serveable tier — the SAME fallback the walk-OOM path uses — so a small card that auto-picked
        // an unservable tier (e.g. an 8 GB 3070 → Qwen3.5-9B whose self-test OOMs on the shared serving
        // card) falls back to a model it can both walk AND serve (Gemma-3-4B) instead of idling.
        // --force-model is honored verbatim (no demotion). Withdraw the failed model so it is not
        // re-picked; the demoted model self-tests fresh next cycle.
        crate::slm::mark_model_unavailable(&model_id, "self_test_failed");
        let vram_mb = query_all_gpus_vram().into_iter()
            .find(|(d, _)| *d == device_id).map(|(_, m)| m).unwrap_or(0);
        if is_device_forced(device_id) {
            crate::slm::set_staging_error(format!(
                "GPU {} could not SERVE the FORCED model (inference self-test failed on GPU {}) — \
                 --force-model is honored without a fallback. Choose a smaller --force-model or a serving \
                 card with more free VRAM. Not demoting (forced).",
                device_id, inference_gpu
            ));
            static WARNED_FORCED_SERVE: AtomicBool = AtomicBool::new(false);
            if !WARNED_FORCED_SERVE.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "PoM[gpu{}]: FORCED model not serveable (self-test failed on GPU {}) — NOT mining, NOT \
                     demoting (forced). Force a smaller tier or use a bigger serving card.",
                    device_id, inference_gpu
                );
            }
            return false;
        }
        match crate::slm::next_smaller_ready_spec(&model_id, vram_mb) {
            Some(smaller) => {
                log::warn!(
                    "PoM[gpu{}]: tier not serveable (inference self-test failed on GPU {}) — DEMOTING to \
                     '{}' ({} MB budget) and rebuilding next cycle. For a specific tier use --force-model.",
                    device_id, inference_gpu, smaller.name, smaller.min_vram_mb
                );
                let gguf = crate::slm::gguf_path_for(smaller).to_string_lossy().into_owned();
                set_device_model(device_id, smaller.model_id, gguf);
                crate::slm::clear_staging_error();
            }
            None => {
                crate::slm::set_staging_error(format!(
                    "GPU {} could not SERVE this model (inference self-test failed on GPU {}) and NO smaller \
                     staged tier is serveable — this card cannot mine any available tier. Use a card with more \
                     VRAM, stage a lighter model, or force a smaller tier (--very-light / --light).",
                    device_id, inference_gpu
                ));
                static WARNED_HALT_SERVE: AtomicBool = AtomicBool::new(false);
                if !WARNED_HALT_SERVE.swap(true, Ordering::Relaxed) {
                    log::warn!(
                        "PoM[gpu{}]: model not serveable (self-test failed on GPU {}) and no smaller staged \
                         tier fits — mining halted for this device until a fitting/serveable model exists.",
                        device_id, inference_gpu
                    );
                }
            }
        }
        return false;
    }
    // Non-inference GPUs (and byte-gate failures) walk a standalone raw upload of the canonical
    // GGUF bytes (`load_raw`) — candle-free, so it works for the H4 llama-only archs
    // (Qwen3.5-hybrid-SSM / GLM-4 / EXAONE-4 / Kimi-Linear-MoE) that candle cannot load at all.
    // This is what makes MULTI-GPU H4 mining work: only the single inference GPU can do the
    // llama zero-dup gather; every other card needs its own copy. catch_unwind so a driver-level
    // panic in one card's load doesn't abort the whole miner.
    let m = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if use_llama {
            info!("PoM[gpu{}]: zero-dup — walking the llama.cpp engine's resident weights (candle dormant)", device_id);
            PomGpuMiner::load_llama(&gguf, device_id as usize)
        } else {
            info!("PoM[gpu{}]: raw gather — standalone canonical GGUF upload (mining-only card)", device_id);
            PomGpuMiner::load_raw(&gguf, device_id as usize)
        }
    }))
    .unwrap_or_else(|_| Err(candle_core::Error::Msg(format!("PoM[gpu{}]: loader panicked", device_id))));
    match m {
        Ok(gm) => {
            let n = gm.n_chunks();
            // N-guard: the gather must match THIS device's host index, else blocks would be rejected.
            if let Some((idx, _)) = crate::pom::active_index_for(device_id) {
                if n != idx.n_chunks {
                    log::error!("PoM[gpu{}]: gather N={} != host index N={} — refusing to mine", device_id, n, idx.n_chunks);
                    return false;
                }
            }
            // Make the resolved walk kernel VISIBLE at startup: the old failure mode (an Ampere+
            // card silently running the stub from a PTX/legacy image) was invisible except as a
            // hashrate that was ~10x too high with zero accepted shares.
            info!("PoM[gpu{}]: v4 walk kernel = {}", device_id, gm.v4_walk_kind());
            install(device_id, gm);
            crate::slm::clear_staging_error();
            info!("PoM[gpu{}]: GPU miner ready — N={} chunks resident (matches host index)", device_id, n);
            true
        }
        Err(e) => {
            let e_msg = e.to_string();
            // upstream Keryx-Labs/keryx-miner@278098b: a transient runtime fault during the load
            // (stale llama gather pointers, byte-gate readback over a poisoned context, …) drops
            // this GPU's state — including its llama engine if hosted here — so the next cycle
            // rebuilds from scratch instead of retrying over poisoned resources forever.
            if is_transient_gpu_runtime_fault(&e_msg) {
                log::warn!(
                    "PoM[gpu{}]: transient GPU runtime fault while loading miner ({}); dropping stale miner state and forcing a rebuild on the next cycle.",
                    device_id,
                    e_msg
                );
                reset_stale_gpu_state(device_id);
                return false;
            }
            // OUT_OF_MEMORY: this tier is simply too big for THIS card. Do not loop on it forever
            // (the "device miner build failed: OUT_OF_MEMORY" spin). Withdraw it and DEMOTE this card
            // to the largest staged tier that actually fits, so it mines/serves something instead of
            // nothing; if nothing smaller is staged/fits, halt this card with clear guidance.
            // (operator: make sure a model that fits VRAM is loaded.)
            let low = e_msg.to_ascii_lowercase();
            if low.contains("out of memory") || low.contains("out_of_memory") {
                let vram_mb = query_all_gpus_vram().into_iter()
                    .find(|(d, _)| *d == device_id).map(|(_, m)| m).unwrap_or(0);
                // --force-model: honor the user's explicit choice — no VRAM check, no demotion.
                if is_device_forced(device_id) {
                    static WARNED_FORCED: AtomicBool = AtomicBool::new(false);
                    if !WARNED_FORCED.swap(true, Ordering::Relaxed) {
                        log::error!(
                            "PoM[gpu{}]: FORCED model OOMed on {} MB VRAM (device miner build: out of \
                             memory). --force-model is honored WITHOUT a VRAM check — choose a smaller \
                             --force-model or a card with more VRAM. Not demoting (forced).",
                            device_id, vram_mb
                        );
                    }
                    return false;
                }
                crate::slm::mark_model_unavailable(&model_id, "walk_build_oom");
                crate::slm::clear_self_test(&model_id);
                match crate::slm::next_smaller_ready_spec(&model_id, vram_mb) {
                    Some(smaller) => {
                        log::warn!(
                            "PoM[gpu{}]: model walk OOMed on {} MB VRAM — too big for this card. \
                             DEMOTING to '{}' ({} MB budget) and rebuilding. For a specific tier use \
                             --force-model, or use a card with more VRAM.",
                            device_id, vram_mb, smaller.name, smaller.min_vram_mb
                        );
                        let gguf = crate::slm::gguf_path_for(smaller).to_string_lossy().into_owned();
                        set_device_model(device_id, smaller.model_id, gguf);
                    }
                    None => {
                        crate::slm::set_staging_error(format!(
                            "GPU {} ({} MB VRAM) ran OUT OF MEMORY loading the model and no smaller staged tier \
                             fits — this card cannot mine any available tier. Use a card with more VRAM, or \
                             force a lighter tier (--very-light Qwen3.5-9B / --light GLM-4-9B).",
                            device_id, vram_mb
                        ));
                        static WARNED_HALT: AtomicBool = AtomicBool::new(false);
                        if !WARNED_HALT.swap(true, Ordering::Relaxed) {
                            log::error!(
                                "PoM[gpu{}]: model walk OOMed on {} MB VRAM and NO smaller staged tier \
                                 fits — this card cannot mine any available tier. Download a lighter \
                                 model (t0 Qwen3.5-9B / t1 GLM-4-9B) or use a card with more VRAM. \
                                 Mining is halted for this device until a fitting model is available.",
                                device_id, vram_mb
                            );
                        }
                    }
                }
                return false;
            }
            log::error!("PoM[gpu{}]: device miner build failed: {}", device_id, e);
            crate::slm::set_staging_error(format!(
                "GPU {} failed to LOAD the model onto the card: {}. Check the GPU/driver (nvidia-smi), VRAM, \
                 and that the model file is valid; mining on this card is suspended until the load succeeds.",
                device_id, e
            ));
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These exercise `remove_device_entry` directly with a dummy value type, rather than going
    // through `install`/`uninstall`, because `PomGpuMiner` can only be constructed via `load`/
    // `load_shared`, both of which require real CUDA hardware (`Device::new_cuda`) unavailable in
    // CI/unit-test environments. `remove_device_entry` holds the entire scoping logic that
    // `uninstall` delegates to, so this still covers the behavior that matters: only the targeted
    // device's entry is removed, every other device's entry survives untouched.

    #[test]
    fn remove_device_entry_only_clears_target_device() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(0, "gpu0-miner");
        map.insert(1, "gpu1-miner");
        map.insert(2, "gpu2-miner");

        remove_device_entry(&mut map, 0);

        assert!(!map.contains_key(&0));
        assert_eq!(map.get(&1), Some(&"gpu1-miner"));
        assert_eq!(map.get(&2), Some(&"gpu2-miner"));
        assert_eq!(map.len(), 2);
    }

    // Upstream aa29fd2: the drain barrier in `uninstall`.
    #[test]
    fn barrier_waits_for_the_last_walk_to_release_the_miner() {
        use std::sync::mpsc;
        use std::time::Duration;

        let miner = Arc::new("gpu0-miner");
        let held = Arc::clone(&miner);
        let (tx, rx) = mpsc::channel();
        let walker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(60));
            drop(held);
            let _ = tx.send(());
        });

        assert!(wait_for_sole_owner(&miner, Duration::from_secs(5)), "must wait, not give up");
        assert_eq!(Arc::strong_count(&miner), 1);
        rx.recv_timeout(Duration::from_secs(5)).unwrap();
        walker.join().unwrap();
    }

    #[test]
    fn barrier_gives_up_after_the_deadline_rather_than_hanging() {
        use std::time::Duration;

        let miner = Arc::new("gpu0-miner");
        let _stuck = Arc::clone(&miner);

        assert!(!wait_for_sole_owner(&miner, Duration::from_millis(50)));
    }

    #[test]
    fn remove_device_entry_hands_back_the_removed_miner() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(0, "gpu0-miner");

        assert_eq!(remove_device_entry(&mut map, 0), Some("gpu0-miner"));
        assert_eq!(remove_device_entry(&mut map, 0), None);
    }

    #[test]
    fn remove_device_entry_on_missing_device_is_a_no_op() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(1, "gpu1-miner");

        remove_device_entry(&mut map, 0);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"gpu1-miner"));
    }

    // upstream Keryx-Labs/keryx-miner@278098b (extended): only the ILLEGAL_ADDRESS class is
    // transient/recoverable. OOM and the box-wide poison class (DEINITIALIZED / Xid-79 bus drop)
    // must NOT match — they keep the pre-existing (non-reset) behavior.
    #[test]
    fn detects_transient_illegal_address_faults() {
        assert!(is_transient_gpu_runtime_fault("CUDA_ERROR_ILLEGAL_ADDRESS"));
        assert!(is_transient_gpu_runtime_fault("an illegal memory access was encountered"));
        // cudarc DriverError Display shape, as wrapped by candle_core::Error::wrap in mine()/load*():
        assert!(is_transient_gpu_runtime_fault(
            "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \"an illegal memory access was encountered\")"
        ));
        assert!(is_transient_gpu_runtime_fault("misaligned address"));
        assert!(is_transient_gpu_runtime_fault("invalid device pointer"));
        // The one-shot injection knob's synthetic message must classify transient too:
        assert!(is_transient_gpu_runtime_fault(
            "injected transient fault (KERYX_FAULT_INJECT_GPU): CUDA_ERROR_ILLEGAL_ADDRESS, an illegal memory access was encountered"
        ));
    }

    #[test]
    fn oom_and_bus_drop_are_not_transient() {
        assert!(!is_transient_gpu_runtime_fault("out of memory"));
        assert!(!is_transient_gpu_runtime_fault("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(!is_transient_gpu_runtime_fault(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        ));
        // Box-wide poison class (Xid-79 / fell off the bus / driver teardown): reboot-only, never reset.
        assert!(!is_transient_gpu_runtime_fault("CUDA_ERROR_DEINITIALIZED"));
        assert!(!is_transient_gpu_runtime_fault(
            "DriverError(CUDA_ERROR_DEINITIALIZED, \"driver shutting down\")"
        ));
        assert!(!is_transient_gpu_runtime_fault("CUDA_ERROR_LAUNCH_TIMEOUT"));
        assert!(!is_transient_gpu_runtime_fault("CUDA_ERROR_INVALID_PTX"));
        assert!(!is_transient_gpu_runtime_fault("PoM GPU: model produced 0 chunks"));
    }
}

#[cfg(test)]
impl PomGpuMiner {
    /// Test-only walk source: upload arbitrary chunk-aligned segments (no GGUF, no llama) so the
    /// v3 kernel can be checked against the host reference over a synthetic blob.
    pub(crate) fn load_test_segments(device_id: usize, segments: Vec<Vec<u8>>) -> candle_core::Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        let mut uploads: Vec<CudaSlice<u8>> = Vec::new();
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        for seg in &segments {
            let chunks = (seg.len() / CHUNK_BYTES) as u64;
            if chunks == 0 {
                continue;
            }
            let dev = stream.clone_htod(seg.as_slice()).map_err(candle_core::Error::wrap)?;
            bases.push(dev.device_ptr(&stream).0 as u64);
            uploads.push(dev);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let _ = load_walk_func(&cuda, "pom_mine_v4")?;
        Ok(Self {
            cuda,
            stream,
            bases_dev,
            prefix_dev,
            t_count: bases.len() as u32,
            n_total_chunks,
            _tensors: Vec::new(),
            _shared: Vec::new(),
            _uploads: uploads,
        })
    }
}

/// v4 GPU↔host byte-exact lockstep — needs a CUDA card:
/// `cargo test --release --features pom-cuda -- --ignored v4_gpu`.
/// Proves the `pom_mine_v4` kernel derives the SAME `final_state` (hence pow_value) as the host
/// `pom_v4::build_proof_v4` for a set of nonces, by bracketing the GPU's internal pow against the
/// host pow_value: the GPU must WIN at `target == host_pow` and LOSE at `target == host_pow - 1`.
#[cfg(test)]
mod v4_kernel_tests {
    use super::*;

    const PPH: [u8; 32] = [7u8; 32];
    const TS: u64 = 0x11_2233_4455;

    fn blob(n_tiles: usize) -> Vec<u8> {
        let mut b = vec![0u8; n_tiles * crate::pom_v4::POM_V4_TILE_BYTES];
        let mut h = 0xDEAD_BEEF_u64;
        for x in b.iter_mut() {
            h = crate::pom::mix64(h);
            *x = h as u8;
        }
        b
    }

    fn dec_le(mut v: [u8; 32]) -> [u8; 32] {
        for byte in v.iter_mut() {
            if *byte == 0 {
                *byte = 0xff;
            } else {
                *byte -= 1;
                break;
            }
        }
        v
    }

    /// Reports (visibly, with --nocapture) which walk kernel this GPU resolves to, and proves the
    /// resolved kernel is BIT-EXACT vs the host walk. Run per card / per package to verify that an
    /// Ampere+ card really gets the tensor-core walk (modern package) and that a stub-only image
    /// (legacy/PTX) is detected and demoted instead of silently mining nothing.
    #[test]
    #[ignore]
    fn v4_walk_kind_and_exactness() {
        let data = blob(2048);
        let index = crate::pom::index_from_ram(data.clone());
        let miner = PomGpuMiner::load_test_segments(0, vec![data]).unwrap();
        let raw = miner.v4_tc_kernel_is_real();
        let kind = miner.v4_walk_kind();
        println!("RAW TC-KERNEL PROBE: {}", if raw { "REAL (produced a result)" } else { "STUB (no output — would mine nothing)" });
        println!("RESOLVED WALK KERNEL: {kind}   (image {} {})",
                 env!("POM_WALK_IMAGE_KIND"), env!("POM_PTX_ARCH"));
        let nonce = 42u64;
        let seed = crate::pom::pom_block_seed_v4(&PPH, TS, nonce);
        let (v4, fs) = crate::pom_v4::build_proof_v4(0, seed, &index).unwrap();
        let re = crate::pom_v4::verify_proof_v4(seed, &v4, &index.r_t, index.n_chunks).unwrap();
        assert_eq!(re, fs);
        let pow = crate::pom::pom_pow_value(fs, &PPH, true);
        assert_eq!(miner.mine_v4(&PPH, TS, &pow, nonce, 1).unwrap(), Some(nonce),
                   "[{kind}] GPU did not find the nonce at the host pow target — divergence");
        assert_eq!(miner.mine_v4(&PPH, TS, &dec_le(pow), nonce, 1).unwrap(), None,
                   "[{kind}] GPU found the nonce below the host pow — divergence");
        println!("BIT-EXACT OK on the {kind} walk");
    }

    /// Throughput bench (no pool, synthetic blob): `--ignored v4_bench`. Blob size via
    /// KERYX_BENCH_TILES (default 6 GiB / 1KB tiles), duration via KERYX_BENCH_SECS (default 10).
    /// Impossible target => no winner => every batch runs to completion. Prints Mh/s.
    #[test]
    #[ignore]
    fn v4_bench() {
        let n_tiles: usize = std::env::var("KERYX_BENCH_TILES").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(6 * 1024 * 1024);
        let secs: u64 = std::env::var("KERYX_BENCH_SECS").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(10);
        let data = blob(n_tiles);
        println!("bench blob: {} tiles = {:.2} GiB", n_tiles, (n_tiles as f64) / (1024.0 * 1024.0));
        let miner = PomGpuMiner::load_test_segments(0, vec![data]).unwrap();
        let target = [0u8; 32]; // impossible (pow > 0 always)
        let batch: u64 = std::env::var("KERYX_POM_V4_BATCH").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(1 << 16);
        // warmup
        let _ = miner.mine_v4(&PPH, TS, &target, 0, batch).unwrap();
        let t0 = std::time::Instant::now();
        let mut nonces = 0u64;
        while t0.elapsed().as_secs() < secs {
            let _ = miner.mine_v4(&PPH, TS, &target, nonces, batch).unwrap();
            nonces += batch;
        }
        let el = t0.elapsed().as_secs_f64();
        println!("v4_bench: {} nonces in {:.2}s = {:.3} Mh/s (batch {})",
                 nonces, el, nonces as f64 / el / 1e6, batch);
    }

    #[test]
    #[ignore]
    fn v4_gpu_matches_host_pow() {
        let data = blob(2048);
        let index = crate::pom::index_from_ram(data.clone());
        let miner = PomGpuMiner::load_test_segments(0, vec![data]).unwrap();
        // Cover BOTH kernels: the tensor-core solver (default on sm_80+; on older cards this
        // pass silently uses the classic kernel too) and the classic dp4a kernel (forced).
        for force_classic in [false, true] {
            if force_classic {
                std::env::set_var("KERYX_POM_V4_TC", "0");
            } else {
                std::env::remove_var("KERYX_POM_V4_TC");
            }
            let tag = if force_classic { "classic" } else { "auto/tc" };
            for &nonce in &[1u64, 7, 42, 1000, 65_535, 1_000_003] {
                let seed = crate::pom::pom_block_seed_v4(&PPH, TS, nonce);
                let (v4, fs) = crate::pom_v4::build_proof_v4(0, seed, &index).unwrap();
                // Host self-consistency: the witness re-verifies to the same final_state.
                let re = crate::pom_v4::verify_proof_v4(seed, &v4, &index.r_t, index.n_chunks).unwrap();
                assert_eq!(re, fs, "host verify_proof_v4 != build_proof_v4 final_state (nonce {nonce})");
                let pow = crate::pom::pom_pow_value(fs, &PPH, true);
                // GPU wins at target == host pow_value ...
                assert_eq!(
                    miner.mine_v4(&PPH, TS, &pow, nonce, 1).unwrap(),
                    Some(nonce),
                    "[{tag}] GPU did NOT find nonce {nonce} at host pow target — GPU pow > host pow (divergence)"
                );
                // ... and LOSES one below it -> GPU pow == host pow exactly (byte-exact final_state).
                assert_eq!(
                    miner.mine_v4(&PPH, TS, &dec_le(pow), nonce, 1).unwrap(),
                    None,
                    "[{tag}] GPU found nonce {nonce} below host pow — GPU pow < host pow (divergence)"
                );
            }
        }
        std::env::remove_var("KERYX_POM_V4_TC");
    }
}
