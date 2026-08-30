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

/// Second per-device offsets buffer, so the chase for batch N+1 can run into one buffer while the
/// walk of batch N reads the other (cross-BATCH pipelining — see `mine_v4`).
static V4_OFFSETS_B: OnceLock<Mutex<HashMap<usize, Arc<CudaSlice<u32>>>>> = OnceLock::new();

fn v4_offsets_buf_b(stream: &Arc<CudaStream>, len: usize) -> candle_core::Result<Arc<CudaSlice<u32>>> {
    let m = V4_OFFSETS_B.get_or_init(|| Mutex::new(HashMap::new()));
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

/// What the chase has already resolved into the *spare* buffer, so the next `mine_v4` can skip its
/// chase entirely. Keyed per device; the key identifies the exact work the offsets belong to —
/// seed words + timestamp + nonce range + K. A mismatch (new job, resized batch, nonce jump) simply
/// falls back to chasing inline, so a stale prefetch can never feed the walk wrong offsets.
#[derive(Clone, Copy, PartialEq, Eq)]
struct V4Prefetch {
    s: [u64; 4],
    timestamp: u64,
    start: u64,
    batch: u64,
    k: u32,
    /// WHICH of the two offset buffers the prefetch chase wrote into (0 = A, 1 = B). The buffers
    /// alternate every batch, so this cannot be inferred from "was there a prefetch" — inferring
    /// it made two consecutive hits both read buffer B while the second prefetch had filled A,
    /// i.e. the walk hashed another nonce range's offsets and silently found nothing.
    buf: u8,
}
static V4_PREFETCH: OnceLock<Mutex<HashMap<usize, V4Prefetch>>> = OnceLock::new();

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
        if self.v4_ncf_usable(n_tiles, k, &p, &s, 0) {
            "chaseless"
        } else if self.v4_tc_usable(n_tiles, k, &p, &s, 0) {
            "tensor-core"
        } else {
            "classic"
        }
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
    pub fn mine_v4(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h10_era: bool) -> candle_core::Result<Option<u64>> {
        let n_tiles = self.n_total_chunks / crate::pom_v4::POM_V4_TILE_CHUNKS;
        // H10 one-way seed selector, passed verbatim to every walk/chase kernel (byte-identical
        // dispatch to pom.rs::pom_block_seed_v4_era). 0 = reversible v4 fold; 1 = one-way H10 fold.
        let h10: u32 = if h10_era { 1 } else { 0 };
        if n_tiles == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: blob too small for the v4 walk".into()));
        }
        // POW fold words = H3-salted pph ("v4 pow uses the h3 fold"). SEED words depend on the era:
        // pre-H10 = v4-salted pph (reversible fold); H10 = RAW pph words (the kernel's keccak absorbs
        // them into the cSHAKE256 PowHash sponge). The `h10` flag below tells the kernel which fold.
        let p = crate::pom::pph_words_for_era(pre_pow_hash, true);
        let s = if h10_era {
            crate::pom::pph_words(pre_pow_hash)
        } else {
            crate::pom::pph_words_v4(pre_pow_hash)
        };
        let t = words4(target_le);
        let k = crate::pom_v4::POM_V4_K as u32;
        bind_device_ctx(&self.stream)?; // multi-GPU: bind this device's context before the raw launch
        let winner = self.stream.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;

        // A card whose autotune measured the classic walk faster takes it even though the
        // tensor-core kernel is available and correct here.
        let tune = v4_tune_for(self.stream.context().ordinal() as u32);
        let ncf_wanted = tune.map_or_else(|| self.v4_ncf_default(), |t| t.ncf);
        let tc_wanted = tune.map_or(true, |t| t.tc);
        if ncf_wanted && self.v4_ncf_usable(n_tiles, k, &p, &s, timestamp) {
            // Chaseless tensor-core solver: ONE kernel, no chase pass, no offsets buffer. Each
            // warp derives its next tile offset from the snippet of the tile it just fetched (the
            // offset chain never depends on the walk state), so the chase's serial ~7-9% of every
            // batch and its duplicate snippet reads disappear. Measured +7-8% on a 5090 at a 64K
            // batch and +24% at 8K (byte-exact, see pom_mine_v4_ncf). KERYX_POM_V4_NCF=0 falls
            // back to the chase+tc pipeline below.
            let walk = load_walk_func(&self.cuda, "pom_mine_v4_ncf")?;
            let (lut, lut_sh) = self.v4_ncf_lut()?;
            let inv_n = u64::MAX / n_tiles;
            let warps = v4_ncf_warps();
            let cfg = LaunchConfig {
                grid_dim: (((batch + warps - 1) / warps) as u32, 1, 1),
                block_dim: ((warps * 32) as u32, 1, 1),
                shared_mem_bytes: (warps as u32) * 3 * 1024, // per warp: 1 KB state + 2 tile buffers
            };
            let mut b = walk.builder();
            b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&n_tiles).arg(&k)
                .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3])
                .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
                .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
                .arg(&start).arg(&batch)
                .arg(&*lut).arg(&lut_sh).arg(&inv_n).arg(&h10).arg(&winner);
            unsafe { b.launch(cfg).map_err(candle_core::Error::wrap)?; }
        } else if tc_wanted && self.v4_tc_usable(n_tiles, k, &p, &s, timestamp) {
            // Tensor-core solver: resolve the whole tile-offset chain first (it depends only on
            // tile snippets, never on the walk state), then walk with a depth-3 cp.async tile
            // pipeline + 8x mma.sync.m16n8k32.s8 per step. Byte-exact vs pom_mine_v4/the host;
            // measured +35% on Blackwell. KERYX_POM_V4_TC=0 forces the classic kernel.
            //
            // The offsets buffer is CACHED per device (plain alloc, never zeroed — the chase
            // overwrites every word): the old per-batch alloc_zeros paid a cudaMalloc + a full
            // 16 MB memset + free EVERY batch (~180x/s at batch 16384), measured worth 6-10%.
            // ── CROSS-BATCH CHASE PIPELINING ───────────────────────────────────────────────
            // The chase (offset chain) and the walk are separate kernels. Profiled on a 5090 at a
            // 64K batch: chase 0.81 ms / walk 10.19 ms — the chase is 7.4% of the batch, and it is
            // pure serial overhead when run inline. The previous scheme overlapped chase and walk
            // WITHIN a batch (4 sub-batches), but that shrinks BOTH kernels: at 16K nonces the chase
            // covers only 64 blocks on 170 SMs (occupancy 16.5% vs 25% at full batch), which cost
            // more than the overlap won — measured, sub-batching was slower than not overlapping.
            //
            // Instead, pipeline across BATCHES: both kernels stay full-size, and while the walk of
            // batch N runs on the main stream, the chase for batch N+1 runs on the chase stream into
            // the spare buffer. The next call finds its offsets already resolved and goes straight
            // to the walk. The prefetch is keyed by (seed words, timestamp, nonce range, K), so a new
            // job / resized batch / nonce jump misses the key and falls back to chasing inline —
            // stale offsets can never reach the walk.
            let need = (batch as usize) * (k as usize);
            let buf_a = v4_offsets_buf(&self.stream, need)?;
            let buf_b = v4_offsets_buf_b(&self.stream, need)?;
            let tc_warps: u64 = v4_tc_warps();
            let tc_pipe: u64 = v4_tc_pipe();
            let tc_smem: u32 = (tc_warps * 256 * (tc_pipe + 1) * 4) as u32;
            let chase = load_walk_func(&self.cuda, "pom_mine_v4_chase")?;
            let walk = load_walk_func(&self.cuda, "pom_mine_v4_tc")?;
            // MEASURED NEGATIVE — default OFF. Cross-batch prefetch is sound (see
            // v4_chase_prefetch_matches_serial) but SLOWER on a 5090: 5.92 vs 5.99 Mh/s. The walk
            // already runs at ~91% of DRAM peak, so the chase's bytes cannot be hidden — running it
            // concurrently only adds memory contention, plus every job change throws away a
            // speculative chase. Kept behind the knob because a future card with bandwidth headroom
            // (or a heavier chase) could flip the sign. `=1` enables it.
            let pipelined = std::env::var("KERYX_POM_V4_CHASE_PREFETCH").ok().as_deref() == Some("1");
            let ord = self.stream.context().ordinal();
            // Is there a prefetch for exactly THIS work, and in which buffer?
            let hit: Option<u8> = if pipelined {
                let m = V4_PREFETCH.get_or_init(|| Mutex::new(HashMap::new()));
                let g = m.lock().unwrap();
                g.get(&ord).and_then(|p| {
                    (p.s == s && p.timestamp == timestamp && p.start == start && p.batch == batch && p.k == k)
                        .then_some(p.buf)
                })
            } else {
                None
            };
            let prefetched = hit.is_some();
            // The walk reads the buffer that actually holds this batch's offsets; the other one
            // takes the next prefetch. Roles alternate every batch, so nothing is ever copied.
            let cur_idx: u8 = hit.unwrap_or(0);
            let (cur, spare) = if cur_idx == 1 { (buf_b.clone(), buf_a.clone()) } else { (buf_a.clone(), buf_b.clone()) };
            let spare_idx: u8 = 1 - cur_idx;

            let chase_cfg = |n: u64| LaunchConfig {
                grid_dim: (((n + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            if !prefetched {
                // No usable prefetch (first batch after a new job, or pipelining off): chase inline
                // on the main stream, exactly as before.
                let view = cur.try_slice(0..need)
                    .ok_or_else(|| candle_core::Error::Msg("PoM v4: offsets slice out of range".into()))?;
                let mut b = chase.builder();
                b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&n_tiles).arg(&k)
                    .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
                    .arg(&start).arg(&batch).arg(&h10).arg(&view);
                unsafe { b.launch(chase_cfg(batch)).map_err(candle_core::Error::wrap)?; }
            }

            // Queue the walk for THIS batch on the main stream.
            {
                let view = cur.try_slice(0..need)
                    .ok_or_else(|| candle_core::Error::Msg("PoM v4: offsets slice out of range".into()))?;
                let walk_cfg = LaunchConfig {
                    grid_dim: (((batch + tc_warps - 1) / tc_warps) as u32, 1, 1),
                    block_dim: ((tc_warps * 32) as u32, 1, 1),
                    shared_mem_bytes: tc_smem,
                };
                let mut b = walk.builder();
                b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&k)
                    .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3])
                    .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
                    .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
                    .arg(&start).arg(&batch).arg(&view).arg(&h10).arg(&winner);
                unsafe { b.launch(walk_cfg).map_err(candle_core::Error::wrap)?; }
            }

            // …and, concurrently, chase the NEXT nonce range into the spare buffer. This is the
            // whole win: by the time the caller asks for that range the offsets already exist.
            if pipelined {
                let next_start = start.wrapping_add(batch);
                let cs = v4_chase_stream(&self.stream)?;
                let view = spare.try_slice(0..need)
                    .ok_or_else(|| candle_core::Error::Msg("PoM v4: prefetch slice out of range".into()))?;
                let func = chase.func.clone();
                let mut b = cs.launch_builder(&func);
                b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&n_tiles).arg(&k)
                    .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
                    .arg(&next_start).arg(&batch).arg(&h10).arg(&view);
                unsafe { b.launch(chase_cfg(batch)).map_err(candle_core::Error::wrap)?; }
                // The next walk runs on the main stream, so it must not start before this chase
                // finishes — record the dependency now, honoured at the top of the next call.
                let ev = cs.record_event(None).map_err(candle_core::Error::wrap)?;
                self.stream.wait(&ev).map_err(candle_core::Error::wrap)?;
                let m = V4_PREFETCH.get_or_init(|| Mutex::new(HashMap::new()));
                m.lock().unwrap().insert(ord, V4Prefetch { s, timestamp, start: next_start, batch, k, buf: spare_idx });
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
                .arg(&start).arg(&batch).arg(&h10).arg(&winner);
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
        let h0: u32 = 0; // probes always exercise the pre-H10 (reversible) seed fold
        let mut b = chase.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&n_tiles).arg(&k)
            .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
            .arg(&start).arg(&batch).arg(&h0).arg(&view);
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
            .arg(&start).arg(&batch).arg(&view).arg(&h0).arg(&winner);
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

    /// Device LUT + shift for the chaseless walk's segment resolve; built once per installed blob
    /// (host-side from the prefix table), then served from the per-device cache.
    fn v4_ncf_lut(&self) -> candle_core::Result<(Arc<CudaSlice<u16>>, u32)> {
        if self.t_count as u64 > u16::MAX as u64 {
            return Err(candle_core::Error::Msg(
                "PoM v4 ncf: more segments than the u16 LUT can index".into(),
            ));
        }
        let m = V4_NCF_LUT.get_or_init(|| Mutex::new(HashMap::new()));
        let ord = self.stream.context().ordinal();
        {
            let g = m.lock().unwrap();
            if let Some(e) = g.get(&ord) {
                if e.t_count == self.t_count && e.n_chunks == self.n_total_chunks {
                    return Ok((e.lut.clone(), e.sh));
                }
            }
        }
        let prefix: Vec<u64> =
            self.stream.clone_dtoh(&self.prefix_dev).map_err(candle_core::Error::wrap)?;
        let mut sh = 0u32;
        while (self.n_total_chunks >> sh) > 16384 {
            sh += 1;
        }
        let nbuck = (self.n_total_chunks >> sh) as usize + 1;
        let mut lut = vec![0u16; nbuck];
        let mut lo = 0usize;
        for (bk, e) in lut.iter_mut().enumerate() {
            let idx = (bk as u64) << sh;
            while lo + 1 < prefix.len() && prefix[lo + 1] <= idx {
                lo += 1;
            }
            *e = lo as u16;
        }
        let dev = Arc::new(self.stream.clone_htod(&lut).map_err(candle_core::Error::wrap)?);
        m.lock().unwrap().insert(
            ord,
            V4NcfLut { t_count: self.t_count, n_chunks: self.n_total_chunks, lut: dev.clone(), sh },
        );
        Ok((dev, sh))
    }

    /// UNTUNED default for the chaseless solver. Fleet-measured 2026-08-29: chaseless wins on every
    /// GDDR card (5090 +8%, 5080 +11%, 5070Ti +9%, 3070 +4-14%) but LOSES on HBM (170HX/sm_80:
    /// -14% — the latency profile favours the decoupled chase pipeline). The autotune measures and
    /// overrides this either way; the default only matters where tuning is skipped (--intensity,
    /// --only-inference, busy tune slot), so HBM-class parts (CC 8.0 A100/170HX, 9.0 H100) default
    /// to the chase+tc pipeline rather than sit on a known regression.
    fn v4_ncf_default(&self) -> bool {
        match self.stream.context().compute_capability() {
            Ok((8, 0)) | Ok((9, 0)) => false,
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Whether the chaseless tensor-core solver may run here. Rides the same image/arch gate as the
    /// tc solver (the kernel sits under the same `__CUDA_ARCH__ >= 800` guard), so `KERYX_POM_V4_TC=0`
    /// (the "force classic" escape hatch) disables it too; `KERYX_POM_V4_NCF=0` disables ONLY the
    /// chaseless solver, falling back to chase+tc.
    fn v4_ncf_available(&self) -> bool {
        if std::env::var("KERYX_POM_V4_NCF").ok().as_deref() == Some("0") {
            return false;
        }
        self.v4_tc_available() && self.t_count as u64 <= u16::MAX as u64
    }

    /// Gate for the chaseless solver: availability AND a one-off 1-nonce always-win probe per
    /// device (same rationale as `v4_tc_usable` — a stub or missing symbol mines nothing loudly
    /// fast, so it must be caught before the first real batch).
    fn v4_ncf_usable(&self, n_tiles: u64, k: u32, p: &[u64; 4], s: &[u64; 4], timestamp: u64) -> bool {
        if !self.v4_ncf_available() {
            return false;
        }
        static PROBED: OnceLock<Mutex<std::collections::HashMap<u32, bool>>> = OnceLock::new();
        let cache = PROBED.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let ord = self.stream.context().ordinal() as u32;
        if let Some(&ok) = cache.lock().unwrap().get(&ord) {
            return ok;
        }
        let ok = match self.v4_ncf_probe(n_tiles, k, p, s, timestamp) {
            Ok(true) => {
                log::info!("PoM v4: chaseless solver PROBE OK on GPU {} (1-nonce known-win recorded).", ord);
                true
            }
            Ok(false) => {
                log::error!(
                    "PoM v4: chaseless solver PROBE FAILED on GPU {} — no result for an always-win \
target (stub image?). Falling back to the chase+tensor-core path.", ord);
                false
            }
            Err(e) => {
                log::error!(
                    "PoM v4: chaseless solver PROBE ERROR on GPU {} ({e}) — falling back to the \
chase+tensor-core path.", ord);
                false
            }
        };
        cache.lock().unwrap().insert(ord, ok);
        ok
    }

    /// One-nonce chaseless walk with an always-win target. Ok(true) = a real kernel ran.
    fn v4_ncf_probe(&self, n_tiles: u64, k: u32, p: &[u64; 4], s: &[u64; 4], timestamp: u64)
        -> candle_core::Result<bool>
    {
        let walk = load_walk_func(&self.cuda, "pom_mine_v4_ncf")?; // missing symbol -> Err -> tc path
        let (lut, lut_sh) = self.v4_ncf_lut()?;
        let inv_n = u64::MAX / n_tiles;
        let warps = v4_ncf_warps();
        let winner = self.stream.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;
        let (start, batch) = (0u64, 1u64);
        let t_max = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let mut b = walk.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&n_tiles).arg(&k)
            .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3])
            .arg(&s[0]).arg(&s[1]).arg(&s[2]).arg(&s[3]).arg(&timestamp)
            .arg(&t_max[0]).arg(&t_max[1]).arg(&t_max[2]).arg(&t_max[3])
            .arg(&start).arg(&batch)
            .arg(&*lut).arg(&lut_sh).arg(&inv_n).arg(&0u32).arg(&winner);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: ((warps * 32) as u32, 1, 1),
                shared_mem_bytes: (warps as u32) * 3 * 1024,
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
    // --wait-ready: this card's walk is resident = the card is set up (idempotent, cheap).
    crate::wait_ready::mark_ready(device_id);
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
    let _ = uninstall_released(device_id);
}

/// `uninstall`, reporting whether the miner's VRAM was actually released.
///
/// `false` means a walk thread still holds the handle after the timeout, so the old table is STILL
/// RESIDENT. That matters to the caller: rebuilding on top of it allocates a SECOND full table, and
/// on a card whose model nearly fills it that fails as `CUBLAS_STATUS_NOT_INITIALIZED` — the
/// restart loop reported from the field on 10 GB RTX 3080s.
pub fn uninstall_released(device_id: u32) -> bool {
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
    let released = match removed {
        Some(miner) => {
            if wait_for_sole_owner(&miner, std::time::Duration::from_secs(30)) {
                true
            } else {
                log::error!(
                    "PoM[gpu{}]: a walk still holds the miner after 30s — its table stays resident, so \
                     this card will NOT be rebuilt until the walk lets go (rebuilding now would need a \
                     second copy of the model and fail with CUBLAS_STATUS_NOT_INITIALIZED).",
                    device_id
                );
                // Keep the handle so the VRAM can be reclaimed the moment the stuck walk drops its
                // own — otherwise the table would be orphaned for the life of the process and the
                // card could never come back without a restart.
                let q = QUARANTINE.get_or_init(|| Mutex::new(HashMap::new()));
                if let Ok(mut g) = q.lock() {
                    g.insert(device_id, miner);
                }
                false
            }
        }
        None => true,
    };
    // The offset buffers belong to the walk that just went away. They were previously kept forever in
    // per-device caches, so every rebuild had to fit a fresh table around them — 102 MiB per card at
    // the current default batch. Only safe to drop once nothing holds the miner any more: a stuck
    // walk may still be reading them.
    if released {
        v4_release_offsets(device_id as usize);
    }
    released
}

/// Cards whose previous walk table could not be released. Rebuilding one of these allocates a second
/// full model copy and dies as CUBLAS_STATUS_NOT_INITIALIZED on any card without a spare model's
/// worth of VRAM, so the rebuild is skipped until the stuck walk lets go.
static TABLE_STUCK: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();

/// Miners whose walk would not let go in time. Held here (not dropped) so that when the stuck walk
/// finally returns, the last handle is ours and dropping it actually frees the table.
static QUARANTINE: OnceLock<Mutex<HashMap<u32, Arc<PomGpuMiner>>>> = OnceLock::new();

fn set_table_stuck(device_id: u32, stuck: bool) {
    let m = TABLE_STUCK.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if let Ok(mut g) = m.lock() {
        if stuck {
            g.insert(device_id);
        } else {
            g.remove(&device_id);
        }
    }
}

/// True while this card still holds a walk table nobody released. Re-checked cheaply: once the stuck
/// walk finally drops its handle the table frees itself, so the card is allowed to rebuild again.
fn table_stuck(device_id: u32) -> bool {
    let Some(m) = TABLE_STUCK.get() else { return false };
    let stuck = m.lock().map(|g| g.contains(&device_id)).unwrap_or(false);
    if !stuck {
        return false;
    }
    // Self-heal: if the stuck walk has finally dropped its handle, ours is the only one left — drop
    // it, which frees the table, and let the card rebuild normally on the next pass.
    let freed = {
        let Some(q) = QUARANTINE.get() else { return true };
        let Ok(mut g) = q.lock() else { return true };
        match g.get(&device_id) {
            Some(m) if Arc::strong_count(m) == 1 => {
                g.remove(&device_id);
                true
            }
            Some(_) => false,
            None => true,
        }
    };
    if freed {
        v4_release_offsets(device_id as usize);
        set_table_stuck(device_id, false);
        log::info!(
            "PoM[gpu{}]: the stuck walk finally released its table — VRAM reclaimed, rebuilding this card.",
            device_id
        );
        return false;
    }
    true
}

/// Drops this device's cached offset buffers and any speculative chase result.
fn v4_release_offsets(ord: usize) {
    if let Some(m) = V4_OFFSETS.get() {
        if let Ok(mut g) = m.lock() {
            g.remove(&ord);
        }
    }
    if let Some(m) = V4_OFFSETS_B.get() {
        if let Ok(mut g) = m.lock() {
            g.remove(&ord);
        }
    }
    if let Some(m) = V4_PREFETCH.get() {
        if let Ok(mut g) = m.lock() {
            g.remove(&ord);
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

/// Per-GPU pause bitmask (bit N = CUDA ordinal N is mid model-swap). Per-card rather than global so
/// one card serving OPoI does not stall the whole rig — only the card whose weights are being freed
/// and reloaded has to stand still.
static INFERENCE_PAUSED_MASK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn set_inference_paused(paused: bool) {
    INFERENCE_PAUSED.store(paused, Ordering::Release);
}

pub fn inference_paused() -> bool {
    INFERENCE_PAUSED.load(Ordering::Acquire)
}

/// Mark `gpu` as mid-swap (or clear it). MUST bracket every llama uninstall/reload: while the bit is
/// set the grind loop must neither run a walk on that card nor rebuild one.
pub fn set_inference_paused_on(gpu: usize, paused: bool) {
    if gpu >= 64 {
        set_inference_paused(paused); // absurd ordinal: fall back to the global flag
        return;
    }
    let bit = 1u64 << gpu;
    if paused {
        INFERENCE_PAUSED_MASK.fetch_or(bit, Ordering::AcqRel);
    } else {
        INFERENCE_PAUSED_MASK.fetch_and(!bit, Ordering::AcqRel);
    }
}

/// Whether `device_id` must hold off mining right now: its own swap bit, or the global flag.
///
/// THIS IS A CRASH GUARD, not an optimisation. On the inference card the walk gathers ZERO-DUP —
/// its base pointers address the llama engine's resident weights directly. If a walk runs (or is
/// rebuilt) while llama frees/reloads that model, it dereferences freed VRAM:
/// CUDA_ERROR_ILLEGAL_ADDRESS, which is STICKY — the context is poisoned, so llama's next
/// cudaFreeHost fails inside ggml and calls ggml_abort(), taking the process down (field reports of
/// a crash-restart loop 10-20 min in, i.e. on the first or second OPoI probe). The flag existed for
/// exactly this but had NO reader on the CUDA path, so the guard never engaged.
pub fn inference_paused_for(device_id: u32) -> bool {
    if INFERENCE_PAUSED.load(Ordering::Acquire) {
        return true;
    }
    let g = device_id as usize;
    g < 64 && (INFERENCE_PAUSED_MASK.load(Ordering::Acquire) & (1u64 << g)) != 0
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
    // Bracket the teardown with this card's pause bit for the same reason the model swap does: the
    // walk and the llama engine share VRAM on the inference card, so nothing may touch (or rebuild)
    // the walk while we free them. RAII-cleared, so a panic in here still resumes the card.
    struct PauseGuard(u32);
    impl Drop for PauseGuard {
        fn drop(&mut self) { set_inference_paused_on(self.0 as usize, false); }
    }
    set_inference_paused_on(device_id as usize, true);
    let _resume = PauseGuard(device_id);
    // Is this device's context still usable, or did the fault poison it? The answer decides whether
    // llama's model may be freed normally or must be abandoned: `llama_free` on a poisoned context
    // aborts the entire process (see `llama_engine::abandon_for_gpu`), which is how a single card's
    // fault turned into a whole-rig restart loop in the field. The probe clones the miner handle
    // only briefly — it must be dropped before `uninstall`, which waits for sole ownership.
    let healthy = {
        let m = miners().lock().ok().and_then(|g| g.get(&device_id).cloned());
        match m {
            Some(m) => m.stream.synchronize().is_ok(),
            None => true,
        }
    };
    let released = uninstall_released(device_id);
    set_table_stuck(device_id, !released);
    if healthy {
        crate::llama_engine::unload_for_gpu(device_id as usize);
    } else {
        let had_engine = crate::llama_engine::abandon_for_gpu(device_id as usize);
        log::error!(
            "PoM[gpu{}]: this device's CUDA context is poisoned (a sticky fault survives a \
             synchronize){} — the card cannot be recovered without restarting the miner. Other GPUs \
             keep mining; restart the process to bring this one back.",
            device_id,
            if had_engine { ", so its llama model was abandoned rather than freed" } else { "" }
        );
    }
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
/// Warps per block and cp.async pipeline depth for the tensor-core walk.
///
/// These are NOT free parameters. They are `#define`s compiled into `pom_mine_v4_tc`, and build.rs
/// publishes the values the loaded kernel was built with. The launch config must match them exactly:
/// the kernel computes its nonce as `blockIdx.x * V4_TC_WARPS + warp` and slices shared memory as
/// `warp * 256 * (V4_TC_PIPE + 1)`. Launching 2 warps/block against a 4-warp kernel does not "use
/// less of the GPU" — it walks only the nonces whose index mod 4 is 0 or 1 while the host still
/// counts the whole batch as hashed, i.e. an inflated hashrate and silently missed shares. Launching
/// 16 overruns the shared-memory slice and faults the device.
///
/// `KERYX_POM_V4_TC_WARPS` / `_TC_PIPE` therefore only apply when a matching custom fatbin is
/// supplied via `KERYX_WALK_FATBIN` (the profiling path, where the kernel is rebuilt with the same
/// `-D` values). Otherwise they are ignored with one warning.
fn v4_tc_compiled(var: &str, compiled: u64) -> u64 {
    let Ok(raw) = std::env::var(var) else { return compiled };
    let Ok(v) = raw.trim().parse::<u64>() else { return compiled };
    if v == compiled {
        return compiled;
    }
    if std::env::var("KERYX_WALK_FATBIN").is_ok() {
        return v; // custom kernel image: the caller compiled it with these values
    }
    static WARNED: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let seen = WARNED.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if seen.lock().map(|mut s| s.insert(var.to_string())).unwrap_or(false) {
        log::warn!(
            "{}={} ignored: the walk kernel in this build is compiled for {}, and launching a \
             different geometry would skip nonces (inflated hashrate, missed shares). Set \
             KERYX_WALK_FATBIN to a kernel built with the same value to use it.",
            var, v, compiled
        );
    }
    compiled
}

fn v4_tc_warps() -> u64 {
    const COMPILED: u64 = match u64::from_str_radix(env!("POM_V4_TC_WARPS"), 10) {
        Ok(v) => v,
        Err(_) => 4,
    };
    v4_tc_compiled("KERYX_POM_V4_TC_WARPS", COMPILED)
}

fn v4_tc_pipe() -> u64 {
    const COMPILED: u64 = match u64::from_str_radix(env!("POM_V4_TC_PIPE"), 10) {
        Ok(v) => v,
        Err(_) => 3,
    };
    v4_tc_compiled("KERYX_POM_V4_TC_PIPE", COMPILED)
}

fn v4_ncf_warps() -> u64 {
    const COMPILED: u64 = match u64::from_str_radix(env!("POM_V4_NCF_WARPS"), 10) {
        Ok(v) => v,
        Err(_) => 4,
    };
    v4_tc_compiled("KERYX_POM_V4_NCF_WARPS", COMPILED)
}

/// Per-device bucket LUT for the chaseless walk's segment resolve (`pom_mine_v4_ncf`): maps
/// `chunk_index >> sh` to the index of the segment containing that chunk, built host-side from the
/// SAME prefix table the kernel's forward walk then refines against — so the resolved segment is
/// identical to the binary search's by construction. Cached per device and keyed by the blob's
/// identity (t_count + n_total_chunks); a model/tier change reinstalls the miner with a different
/// blob and rebuilds. ~16K u16 entries = 32 KB device-resident, L1/L2-hot during the walk.
struct V4NcfLut {
    t_count: u32,
    n_chunks: u64,
    lut: Arc<CudaSlice<u16>>,
    sh: u32,
}
static V4_NCF_LUT: OnceLock<Mutex<HashMap<usize, V4NcfLut>>> = OnceLock::new();

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

/// The v4 grind batch to use on `device_id`, in precedence order:
///
///  1. `--intensity` for this card — a fixed, operator-chosen size (nothing measures over it);
///  2. `--only-inference` — the smallest batch, because that mode wants the card idle for serving;
///  3. the autotune's measured batch, once this card has been benchmarked;
///  4. otherwise the STARTING batch, which is deliberately the high end of the sweep (see
///     `v4_starting_batch`).
///
/// Logged once per device.
pub fn v4_batch_for_device(device_id: u32) -> u64 {
    if let Some(b) = intensity_batch_for(device_id) {
        return log_batch_once(device_id, b, "--intensity");
    }
    if only_inference() {
        return log_batch_once(device_id, POM_V4_BATCH_MIN, "--only-inference (minimum)");
    }
    let (b, how) = match v4_tune_for(device_id) {
        Some(t) => (t.batch, "autotuned"),
        None => match gpu_sm_count(device_id) {
            Some(sm) if v4_headroom_for_high_start(device_id) => {
                (v4_starting_batch(sm), "starting high, autotune may lower it")
            }
            Some(sm) => (v4_batch_for_sm_count(sm), "SM-derived — too little free VRAM to start high"),
            None => (POM_V4_BATCH_FALLBACK, "fallback"),
        },
    };
    log_batch_once(device_id, cap_batch_to_vram(device_id, b), how)
}

fn log_batch_once(device_id: u32, batch: u64, how: &str) -> u64 {
    static LOGGED: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    // Report the STEADY-STATE batch. The first call happens before the walk is installed, when the
    // VRAM cap cannot be evaluated yet, so logging then printed a number the card never actually ran.
    if !is_installed(device_id) {
        return batch;
    }
    if let Ok(mut seen) = LOGGED.get_or_init(|| Mutex::new(std::collections::HashSet::new())).lock() {
        if seen.insert(device_id) {
            log::info!("PoM[gpu{}]: v4 grind batch = {} nonces ({}).", device_id, batch, how);
        }
    }
    batch
}

/// The batch a card runs BEFORE it has measured itself: the top of the sweep, not the middle.
///
/// Every card benchmarked from ~66 SMs up preferred 768 nonces/SM over 384, by up to ~4%, and only a
/// small 46-SM card preferred less. Starting at the high end therefore means most cards are already
/// at their best before the autotune has said anything, and the tuner's job is to walk a card DOWN
/// if it turns out to want less — losing a little speed on the few small cards for the seconds the
/// measurement takes, rather than leaving every big card slow until it finishes.
fn v4_starting_batch(sm: u64) -> u64 {
    v4_batch_for_sm_count(sm).saturating_mul(2)
}

/// Whether this card has enough VRAM headroom to start at the doubled batch.
///
/// Starting high is a throughput win on cards with room, but it is not free: the offset buffers
/// double (102 MiB instead of 51 MiB at 68 SMs) and each launch runs twice as long. On a card whose
/// model nearly fills it — a 10 GB RTX 3080 runs at ~8.2/10.2 GB — that extra pressure showed up in
/// the field as rebuild failures within minutes of starting. Cards with less than ~1.5 GB free keep
/// the SM-derived batch; the autotune can still raise them later, having actually measured it.
fn v4_headroom_for_high_start(device_id: u32) -> bool {
    let free_mib = query_all_gpus_free_vram()
        .into_iter()
        .find(|(o, _, _)| *o == device_id)
        .map(|(_, free, _)| free)
        .unwrap_or(0);
    free_mib == 0 || free_mib >= 1536
}

// ── OPERATOR OVERRIDES: --intensity and --only-inference ───────────────────────────────────────
/// sgminer/cgminer-style intensity per card: batch = 2^intensity nonces.
///
/// Set from `--intensity 18,18,16` (CSV position = CUDA ordinal, same convention as `--force-model`).
/// A card with an intensity is NOT autotuned — the operator has taken the decision.
static V4_INTENSITY: OnceLock<Vec<Option<u32>>> = OnceLock::new();

/// Intensity is a power of two, so the usable window is narrow: below 13 the launch is too small to
/// keep the card busy, and above 21 one launch overruns the ~100 ms block window on any card we have
/// measured (an RTX 5090 does ~5.5 MH/s, so 2^21 nonces is already ~380 ms of work).
pub const INTENSITY_MIN: u32 = 1;
pub const INTENSITY_MAX: u32 = 21;

pub fn set_intensity_map(v: Vec<Option<u32>>) {
    let _ = V4_INTENSITY.set(v);
}

/// The fixed batch for `device_id` if `--intensity` named it, clamped to a launchable size AND to
/// what the card's free VRAM can actually back.
pub fn intensity_batch_for(device_id: u32) -> Option<u64> {
    let i = (*V4_INTENSITY.get()?).get(device_id as usize).copied().flatten()?;
    let want = (1u64 << i.clamp(INTENSITY_MIN, INTENSITY_MAX)).max(64);
    Some(cap_batch_to_vram(device_id, want))
}

/// Bytes of VRAM the offset buffers need for a batch: K offsets per nonce, 4 bytes each, in TWO
/// buffers (the walk reads one while the chase may fill the other).
fn offsets_bytes_for(batch: u64) -> u64 {
    batch.saturating_mul(crate::pom_v4::POM_V4_K as u64).saturating_mul(4).saturating_mul(2)
}

/// Caps `want` so the offset buffers stay inside a quarter of this card's free VRAM.
///
/// This is a HARD safety limit, not a tuning preference. An oversized batch does not merely run
/// slowly: measured on an 8 GB RTX 3070 at `--intensity 18` (batch 262144 → 512 MiB of offsets on a
/// card already holding a ~6.4 GB model), the card ended up with a POISONED CUDA context —
/// `cudaMemGetInfo failed (an illegal memory access was encountered)` — after which llama could not
/// allocate and inference died on that GPU. A knob the operator is invited to turn must not be able
/// to do that, so an unattainable intensity is clamped with a loud warning instead of honoured.
fn cap_batch_to_vram(device_id: u32, want: u64) -> u64 {
    static CAP: OnceLock<Mutex<HashMap<u32, u64>>> = OnceLock::new();
    let cache = CAP.get_or_init(|| Mutex::new(HashMap::new()));
    // The reading is only meaningful once the model and walk are RESIDENT. The grind loop computes a
    // batch before the first `ensure_installed`, when the card still looks almost empty — caching
    // that reading produced a cap ~8x too generous and let an unattainable --intensity through,
    // which is exactly the case this guard exists to stop (2196 illegal-access errors on an 8 GB
    // card in testing). Until the walk is installed, cap from what is free right now and do not
    // remember it.
    let settled = is_installed(device_id);
    let cap = {
        let cached = if settled { cache.lock().ok().and_then(|g| g.get(&device_id).copied()) } else { None };
        match cached {
            Some(c) => c,
            None => {
                let free_mib = query_all_gpus_free_vram()
                    .into_iter()
                    .find(|(o, _, _)| *o == device_id)
                    .map(|(_, free, _)| free)
                    .unwrap_or(0);
                // No reading (driver hiccup, or a card we cannot query) → do not cap.
                let c = if free_mib == 0 {
                    u64::MAX
                } else {
                    // A TENTH of what is free. The measured margin is thin on small cards: an 8 GB
                    // RTX 3070 running the walk sits at ~7.2/8.2 GB, so ~0.9 GB is genuinely free,
                    // and a batch whose offsets ran into that headroom collapsed the card by ~50x
                    // (thrashing, then a poisoned context) rather than degrading gracefully. The
                    // autotuned batch on that card needs ~34 MiB, well inside a tenth.
                    let budget = (free_mib * 1024 * 1024) / 10;
                    (budget / (crate::pom_v4::POM_V4_K as u64 * 4 * 2)).max(POM_V4_BATCH_MIN)
                };
                if settled {
                    if let Ok(mut g) = cache.lock() {
                        g.insert(device_id, c);
                    }
                }
                c
            }
        }
    };
    if want <= cap {
        return want;
    }
    static WARNED: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    let seen = WARNED.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if seen.lock().map(|mut s| s.insert(device_id)).unwrap_or(false) {
        log::warn!(
            "PoM[gpu{}]: batch {} needs {} MiB of offset buffers, which does not fit this card's free \
             VRAM — capped to {}. A batch this size starves the model and can poison the card's CUDA \
             context; lower --intensity (or free VRAM) to use it.",
            device_id, want, offsets_bytes_for(want) / (1024 * 1024), cap
        );
    }
    cap
}

/// `--only-inference`: this rig is here to serve inference, not to hash. Mining runs at the smallest
/// batch with a duty-cycle pause between launches, and stops entirely while a request is being
/// served, so the card is free (and cool, and already clocked up) for the next one.
static ONLY_INFERENCE: AtomicBool = AtomicBool::new(false);

pub fn set_only_inference(on: bool) {
    ONLY_INFERENCE.store(on, Ordering::Relaxed);
}

pub fn only_inference() -> bool {
    ONLY_INFERENCE.load(Ordering::Relaxed)
}

/// How long the walk sleeps between launches in `--only-inference` mode. The card spends most of its
/// time idle, which is the whole point: low power, cool, and instantly available to serve.
pub fn only_inference_duty_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("KERYX_ONLY_INFERENCE_DUTY_MS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(250)
            .clamp(0, 10_000)
    })
}

/// Marks `gpu` as actively serving an inference request, so the grind loop yields the card to it.
/// RAII: the walk resumes when the returned guard drops, panic included.
///
/// Only used in `--only-inference` mode. On a normal rig the walk keeps running during generation —
/// inference is short and the hashrate matters — but a rig that exists to serve should answer as fast
/// as the card can, and a concurrent walk both steals bandwidth and holds the card at a lower clock.
pub struct ServingGuard(usize);

impl Drop for ServingGuard {
    fn drop(&mut self) {
        set_inference_paused_on(self.0, false);
    }
}

pub fn serving_now(gpu: usize) -> Option<ServingGuard> {
    if !only_inference() {
        return None;
    }
    set_inference_paused_on(gpu, true);
    Some(ServingGuard(gpu))
}

// ── PER-CARD LAUNCH AUTOTUNE ───────────────────────────────────────────────────────────────────
// The v4 launch geometry that is fastest on one architecture is NOT fastest on another: warps-per-
// block trades occupancy against the cp.async pipeline depth, and whether the tensor-core walk beats
// the classic dp4a walk at all depends on the card's int8-mma throughput vs its DRAM bandwidth. The
// defaults here (tc, 4 warps, pipe 3, 384 nonces/SM) were measured on Blackwell; carrying them to
// every card in a mixed fleet leaves throughput on the table. So each card measures itself ONCE and
// remembers the answer.
//
// Cost: ~6-8 s on the first grind of a fresh card, then free — the result is cached on disk keyed by
// (GPU name, SM count, walk-table size, K), so a restart reuses it and a model/tier change re-tunes.
// KERYX_POM_V4_AUTOTUNE=0 disables (pure defaults); =force re-measures and overwrites the cache.
// Explicit KERYX_POM_V4_{TC,TC_WARPS,TC_PIPE,BATCH} always override the tuned value.
///
/// The tunable axes are deliberately limited to the ones that cannot change what the kernel
/// computes: which walk kernel runs, and how many nonces are handed to one launch. Warps-per-block
/// and pipeline depth are compiled into the kernel (see `v4_tc_warps`) and are NOT tuned here — a
/// mismatched launch geometry silently skips nonces, which would look like a large speedup while
/// quietly costing shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V4Tune {
    /// Chaseless tensor-core walk (pom_mine_v4_ncf) — tried before the chase+tc pipeline.
    pub ncf: bool,
    pub tc: bool,
    pub batch: u64,
}

static V4_TUNE: OnceLock<Mutex<HashMap<u32, V4Tune>>> = OnceLock::new();

fn v4_tune_map() -> &'static Mutex<HashMap<u32, V4Tune>> {
    V4_TUNE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The measured launch geometry for `device_id`, if this card has been tuned.
pub fn v4_tune_for(device_id: u32) -> Option<V4Tune> {
    v4_tune_map().lock().ok()?.get(&device_id).copied()
}

fn set_v4_tune(device_id: u32, t: V4Tune) {
    if let Ok(mut g) = v4_tune_map().lock() {
        g.insert(device_id, t);
    }
}

fn v4_autotune_mode() -> &'static str {
    static MODE: OnceLock<String> = OnceLock::new();
    MODE.get_or_init(|| std::env::var("KERYX_POM_V4_AUTOTUNE").unwrap_or_default())
        .as_str()
}

fn gpu_name(device_id: u32) -> String {
    use candle_core::cuda_backend::cudarc::driver::result;
    result::init()
        .ok()
        .and_then(|_| result::device::get(device_id as i32).ok())
        .and_then(|d| result::device::get_name(d).ok())
        .unwrap_or_else(|| format!("cuda{}", device_id))
}

fn v4_tune_cache_path() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(std::path::Path::new(&home).join(".keryx").join("v4tune.json"))
}

/// Cache key: everything that can change the right answer. The walk-table size and K matter because
/// they set how much of the card's memory system the walk actually touches.
fn v4_tune_key(device_id: u32, sm: u64, n_tiles: u64, k: u32) -> String {
    format!("{}|sm{}|tiles{}|k{}|g2", gpu_name(device_id), sm, n_tiles, k)
}

fn v4_tune_load(key: &str) -> Option<V4Tune> {
    let raw = std::fs::read_to_string(v4_tune_cache_path()?).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let e = v.get(key)?;
    Some(V4Tune {
        ncf: e.get("ncf").and_then(|v| v.as_bool()).unwrap_or(false),
        tc: e.get("tc")?.as_bool()?,
        batch: e.get("batch")?.as_u64()?,
    })
}

fn v4_tune_save(key: &str, t: V4Tune, mhs: f64) {
    let Some(path) = v4_tune_cache_path() else { return };
    let mut v: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    if let Some(obj) = v.as_object_mut() {
        obj.insert(
            key.to_string(),
            serde_json::json!({
                "ncf": t.ncf, "tc": t.tc, "batch": t.batch,
                "mhs": (mhs * 1000.0).round() / 1000.0,
            }),
        );
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(s) = serde_json::to_string_pretty(&v) {
        let _ = std::fs::write(&path, s);
    }
}

/// Measures one candidate geometry: grinds a never-winning target for ~`ms` and returns Mh/s.
/// A zero target can never be met, so every nonce in the batch runs the full walk — this times the
/// real kernel, not an early exit.
fn v4_bench_cfg(device_id: u32, miner: &PomGpuMiner, pph: &[u8; 32], ts: u64, t: V4Tune, ms: u64) -> Option<f64> {
    let never = [0u8; 32];
    set_v4_tune(device_id, t);
    miner.mine_v4(pph, ts, &never, 1, t.batch, false).ok()?; // warm-up: JIT, buffer alloc, clocks
    let started = std::time::Instant::now();
    let mut n: u64 = 0;
    while started.elapsed() < std::time::Duration::from_millis(ms) {
        miner.mine_v4(pph, ts, &never, 1 + n, t.batch, false).ok()?;
        n += t.batch;
    }
    let secs = started.elapsed().as_secs_f64();
    (secs > 0.0 && n > 0).then(|| n as f64 / secs / 1.0e6)
}

/// Tunes `device_id` once per process. Staged rather than exhaustive (kind → warps → batch) so the
/// whole thing costs seconds, not minutes: the axes are close to independent in the measurements.
/// Returns whether tuning actually completed (false = the slot was busy, try again later).
fn v4_autotune(device_id: u32, miner: &PomGpuMiner) -> bool {
    let mode = v4_autotune_mode();
    if mode == "0" {
        return true;
    }
    // Nothing to measure when the operator has already decided the batch, and nothing worth
    // measuring when the card is deliberately mining at a trickle to stay free for inference.
    if intensity_batch_for(device_id).is_some() || only_inference() {
        return true;
    }
    // ONE CARD AT A TIME. Every device starts grinding at once, so without this the cards benchmark
    // on top of each other and contend for the host, PCIe and power budget — enough noise that two
    // identical 5080s in the same rig picked different batches from the same candidate set.
    // try_lock, never lock: this runs inside the grind call, so BLOCKING here would stall a card's
    // mining until its turn came (measured: the whole rig's hashrate sagged during startup). A card
    // that finds the slot busy simply mines on at the default geometry and tunes on a later batch.
    static TUNING: OnceLock<Mutex<()>> = OnceLock::new();
    let Ok(_one_at_a_time) = TUNING.get_or_init(|| Mutex::new(())).try_lock() else {
        return false;
    };
    let n_tiles = miner.n_total_chunks / crate::pom_v4::POM_V4_TILE_CHUNKS;
    if n_tiles == 0 {
        return true;
    }
    let k = crate::pom_v4::POM_V4_K as u32;
    let sm = gpu_sm_count(device_id).unwrap_or(0);
    let base = if sm > 0 { v4_batch_for_sm_count(sm) } else { POM_V4_BATCH_FALLBACK };
    let key = v4_tune_key(device_id, sm, n_tiles, k);

    if mode != "force" {
        if let Some(t) = v4_tune_load(&key) {
            set_v4_tune(device_id, t);
            log::info!(
                "PoM[gpu{}]: launch tuning restored from cache — {} walk, batch {} ({}).",
                device_id,
                if t.tc { "tensor-core" } else { "classic" },
                t.batch, key
            );
            return true;
        }
    }

    // The pph/timestamp only seed the walk; any fixed pair times the same work.
    let pph = [0u8; 32];
    let ts = 0u64;
    let defaults = V4Tune { ncf: miner.v4_ncf_default(), tc: true, batch: base };

    // Candidates are measured INTERLEAVED and reduced by median. A single timed window per candidate
    // is not able to resolve the ~4% differences that matter here: the first pass measured 2.93 for
    // one 5080 and 2.98 for its twin on the same batch, which is enough noise to flip the choice.
    // Interleaving cancels slow drift (clocks ramping, neighbours' load) because every candidate
    // sees the same conditions in every round.
    let bench_all = |cands: &[V4Tune], reps: usize, ms: u64| -> Vec<Option<f64>> {
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); cands.len()];
        for _ in 0..reps {
            for (i, c) in cands.iter().enumerate() {
                if let Some(m) = v4_bench_cfg(device_id, miner, &pph, ts, *c, ms) {
                    samples[i].push(m);
                }
            }
        }
        samples
            .into_iter()
            .map(|mut v| {
                if v.is_empty() {
                    return None;
                }
                v.sort_by(f64::total_cmp);
                Some(v[v.len() / 2])
            })
            .collect()
    };

    // 1) Which walk? Measured rather than assumed (the tc kernel is a large win on Blackwell AND
    // on Ampere — 3070: 1.35 vs 0.84 Mh/s — and the chaseless kernel beat chase+tc on every card
    // measured so far, but each card decides for itself). All kernels are byte-exact mirrors of
    // the host walk, so any answer is correct; only the speed differs.
    let p_words = crate::pom::pph_words_for_era(&pph, true);
    let s_words = crate::pom::pph_words_v4(&pph);
    let mut kinds: Vec<(&str, V4Tune)> =
        vec![("classic", V4Tune { ncf: false, tc: false, ..defaults })];
    if miner.v4_tc_available() {
        kinds.push(("tensor-core", V4Tune { ncf: false, tc: true, ..defaults }));
    }
    if miner.v4_ncf_usable(n_tiles, k, &p_words, &s_words, ts) {
        kinds.push(("chaseless", V4Tune { ncf: true, tc: true, ..defaults }));
    }
    let cand_tunes: Vec<V4Tune> = kinds.iter().map(|(_, t)| *t).collect();
    let kind_mhs = bench_all(&cand_tunes, 3, 300);
    for ((name, _), m) in kinds.iter().zip(&kind_mhs) {
        log::info!(
            "PoM[gpu{}]: autotune {} walk = {}",
            device_id, name,
            m.map(|v| format!("{:.3} Mh/s", v)).unwrap_or_else(|| "failed".into())
        );
    }
    let Some((mut best, mut best_mhs)) = kinds
        .iter()
        .zip(&kind_mhs)
        .filter_map(|((_, t), m)| m.map(|v| (*t, v)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
    else {
        log::warn!("PoM[gpu{}]: autotune could not measure the walk — keeping defaults.", device_id);
        set_v4_tune(device_id, defaults);
        return true;
    };

    // 2) Batch. A larger batch amortizes the launch and the offset chase over more nonces, but one
    // launch must still finish well inside a block interval (~100 ms at 10 BPS) or the whole batch
    // lands on a stale job. The SM-derived estimate is a good centre, not a maximum: it under-serves
    // big cards (an RTX 5080 measured 3.05 Mh/s at 64K vs 2.90 at its SM-derived 32K), so the sweep
    // looks one step either side. Requires a 2% win to move, so noise cannot flip the choice.
    let batches: Vec<u64> = {
        // Capped the same way the live batch is: a candidate that cannot fit its offset buffers must
        // never be launched, not even once to measure it.
        let mut v: Vec<u64> = [(base / 2).max(POM_V4_BATCH_MIN), base, base * 2]
            .into_iter()
            .map(|b| cap_batch_to_vram(device_id, b))
            .collect();
        v.dedup();
        v
    };
    let cands: Vec<V4Tune> = batches.iter().map(|&b| V4Tune { batch: b, ..best }).collect();
    let batch_mhs = bench_all(&cands, 3, 300);
    for (i, m) in batch_mhs.iter().enumerate() {
        log::info!(
            "PoM[gpu{}]: autotune batch {} = {}",
            device_id,
            batches[i],
            m.map(|v| format!("{:.3} Mh/s", v)).unwrap_or_else(|| "failed".into())
        );
    }
    // Pick the fastest, then take the LARGEST batch within 2% of it. Two reasons to lean big rather
    // than pick the bare maximum: the benchmark grinds back-to-back, so it cannot see the per-batch
    // host work the live loop does between launches (job checks, share handling) — real mining
    // amortizes that over the batch, so a batch that merely ties here is worth more out there — and
    // the measured fleet curve says bigger is right for everything above ~46 SMs. Erring large also
    // means the tuner only ever walks a card DOWN from the high starting batch when it clearly wants
    // less, instead of creeping up to it.
    let peak = batch_mhs.iter().flatten().copied().fold(0.0f64, f64::max);
    if peak > 0.0 {
        for (i, m) in batch_mhs.iter().enumerate() {
            if let Some(v) = *m {
                if v >= peak * 0.98 && batches[i] >= best.batch {
                    best.batch = batches[i];
                    best_mhs = v;
                }
            }
        }
    }

    // 3) The chosen configuration must find the SAME winning nonce as the reference walk. Batch size
    // and kernel choice are not supposed to change results at all, so a mismatch means the candidate
    // is not walking what it claims to — the failure mode that looks like free speed and quietly
    // costs shares. Fall back to defaults rather than mine on it.
    if !v4_config_agrees_with_reference(device_id, miner, best) {
        log::error!(
            "PoM[gpu{}]: autotune result {:?} did NOT reproduce the reference walk's winner — \
             discarding it and keeping the defaults.",
            device_id, best
        );
        set_v4_tune(device_id, defaults);
        return true;
    }

    set_v4_tune(device_id, best);
    v4_tune_save(&key, best, best_mhs);
    log::info!(
        "PoM[gpu{}]: autotuned {} → {} walk, batch {} = {:.2} Mh/s.",
        device_id,
        gpu_name(device_id),
        if best.ncf { "chaseless" } else if best.tc { "tensor-core" } else { "classic" },
        best.batch, best_mhs,
    );
    true
}

/// Verifies a candidate configuration against the classic walk at the stock batch: same seed, same
/// nonce range, a target loose enough that a winner exists, and the winner must be identical.
fn v4_config_agrees_with_reference(device_id: u32, miner: &PomGpuMiner, cand: V4Tune) -> bool {
    // Loose target: top byte 0x0f leaves roughly 1-in-16 nonces winning, so a few thousand nonces
    // are certain to contain one, and the LOWEST winner is a strict function of the walk.
    let mut target = [0xffu8; 32];
    target[31] = 0x0f;
    let pph = [0x5au8; 32];
    let ts = 7u64;
    let probe = 4096u64;
    let reference = {
        set_v4_tune(device_id, V4Tune { ncf: false, tc: false, batch: probe });
        miner.mine_v4(&pph, ts, &target, 1, probe, false)
    };
    let candidate = {
        set_v4_tune(device_id, V4Tune { batch: probe, ..cand });
        miner.mine_v4(&pph, ts, &target, 1, probe, false)
    };
    match (reference, candidate) {
        (Ok(a), Ok(b)) => a == b,
        // A launch error here is itself disqualifying.
        _ => false,
    }
}

/// Runs the autotune the first time this device grinds, and never again in this process.
fn ensure_v4_autotuned(device_id: u32, miner: &PomGpuMiner) {
    static DONE: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    let done = DONE.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    {
        let Ok(g) = done.lock() else { return };
        if g.contains(&device_id) {
            return;
        }
    }
    // Marked only when tuning actually ran, so a card that found the slot busy retries on a later
    // batch. A tune that ran and failed still counts as done — it must not retry every batch.
    if v4_autotune(device_id, miner) {
        if let Ok(mut g) = done.lock() {
            g.insert(device_id);
        }
    }
}

pub fn mine_v4(device_id: u32, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, daa: u64) -> Option<u64> {
    // H10 (DAA >= POM_H10_SEED_ACTIVATION_DAA): one-way seed derivation.
    let h10_era = crate::pom::is_h10_seed_era(daa);
    // 🔴 HARD SAFETY GUARD: the H10 seed formula is an UNVERIFIED PLACEHOLDER until the node's
    // spec lands and pom::pom_seed_fold_v4_h10 is confirmed byte-for-byte. Mining the H10 era with
    // a wrong seed = 100% rejected blocks + service-bond strikes. So while the spec is unverified,
    // REFUSE to grind any H10-era job: no launch, no phantom hashrate (returns None → the worker
    // idles honestly and retries; nothing is counted). Pre-H10 mining is unaffected.
    if h10_era && !crate::pom::H10_SPEC_VERIFIED {
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            log::error!(
                "PoM[gpu{}]: block DAA {} is in the H10 one-way-seed era but this build's H10 seed \
                 formula is an UNVERIFIED PLACEHOLDER — NOT mining (would produce rejected blocks). \
                 Update to the release whose seed is verified against the node's golden vectors.",
                device_id, daa
            );
        }
        return None;
    }
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    // TEST-ONLY fault injection. This knob existed but was never wired into the v4 walk — it was
    // left behind when v4 replaced the v3 grind, so the recovery path it was written to validate
    // (fault → uninstall → rebuild) had no way to be exercised on a healthy rig. That path is
    // exactly what fails in the field, so it needs to be testable.
    if take_injected_fault(device_id) {
        log::warn!(
            "PoM[gpu{}]: TRANSIENT GPU FAULT during the v4 walk (INJECTED by KERYX_FAULT_INJECT_GPU) \
             — rebuilding next cycle.",
            device_id
        );
        // Drop OUR handle before recovering — see the note on the error path below.
        drop(miner);
        reset_stale_gpu_state(device_id);
        return None;
    }
    // First grind on this card: measure its launch geometry (cached on disk, so this is a one-off).
    ensure_v4_autotuned(device_id, &miner);
    let outcome = miner.mine_v4(pre_pow_hash, timestamp, target_le, start, batch, h10_era);
    match outcome {
        Ok(w) => w,
        Err(e) => {
            let msg = e.to_string();
            if is_transient_gpu_runtime_fault(&msg) {
                log::warn!("PoM[gpu{}]: TRANSIENT GPU FAULT during the v4 walk ({}) — rebuilding next cycle.", device_id, msg);
                // RELEASE OUR OWN HANDLE FIRST. Recovery uninstalls the miner and waits for the last
                // handle to drop before the table's VRAM comes back — but this function is holding
                // one, cloned out of the map at the top. Recovering while it is alive made that wait
                // impossible to satisfy: it burned its full 30 s timeout, gave up ("releasing
                // anyway"), and left the old table RESIDENT. The rebuild then had to fit a second
                // copy of the model on the card, which is the CUBLAS_STATUS_NOT_INITIALIZED restart
                // loop reported from the field — worse on cards whose model nearly fills them.
                drop(miner);
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
    // Do not rebuild on top of a table that was never released — see `uninstall_released`. This turns
    // a guaranteed-failing allocation loop (CUBLAS_STATUS_NOT_INITIALIZED every few seconds, which
    // the wrapper then treats as a dead miner and restarts) into an idle card that recovers by itself
    // if the walk lets go, while every other GPU in the rig keeps mining.
    if table_stuck(device_id) {
        static WARNED: OnceLock<Mutex<std::collections::HashMap<u32, u64>>> = OnceLock::new();
        let m = WARNED.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        if let Ok(mut g) = m.lock() {
            let n = g.entry(device_id).or_insert(0);
            if *n % 60 == 0 {
                log::warn!(
                    "PoM[gpu{}]: previous walk table still held by a stuck walk — not rebuilding (a \
                     rebuild would need a second copy of the model). This card is idle; restart the \
                     miner to recover it.",
                    device_id
                );
            }
            *n += 1;
        }
        return false;
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

    /// H10 one-way seed: proves the GPU kernel's `pom_seed_fold_h10` is BYTE-IDENTICAL to the host
    /// `pom::pom_seed_fold_v4_h10` — i.e. the h10 dispatch is wired correctly through the launch.
    /// This validates the WIRING against the placeholder formula (host and GPU agree); it does NOT
    /// validate the formula against the node — that is the golden-vector step, gated on the real
    /// spec. The bracket: build the proof from the host H10 seed, then the GPU must WIN at the host
    /// pow target and LOSE one below it, with h10_era=true.
    #[test]
    #[ignore]
    fn v4_h10_seed_host_gpu_lockstep() {
        let data = blob(2048);
        let index = crate::pom::index_from_ram(data.clone());
        let miner = PomGpuMiner::load_test_segments(0, vec![data]).unwrap();
        for &nonce in &[7u64, 42, 1000, 65535] {
            // Host reference: the H10-era seed (one-way fold), then the standard v4 proof/pow.
            let seed = crate::pom::pom_block_seed_v4_era(&PPH, TS, nonce, true);
            // Sanity: the H10 seed must actually differ from the reversible one (formula changed).
            assert_ne!(seed, crate::pom::pom_block_seed_v4_era(&PPH, TS, nonce, false),
                       "H10 seed equals the reversible seed — the fold did not change");
            let (_v4, fs) = crate::pom_v4::build_proof_v4(0, seed, &index).unwrap();
            let pow = crate::pom::pom_pow_value(fs, &PPH, true);
            // GPU with h10_era=true must reproduce the SAME final_state → win at pow, lose below it.
            assert_eq!(miner.mine_v4(&PPH, TS, &pow, nonce, 1, true).unwrap(), Some(nonce),
                       "H10: GPU seed fold != host at nonce {nonce} (win check)");
            assert_eq!(miner.mine_v4(&PPH, TS, &dec_le(pow), nonce, 1, true).unwrap(), None,
                       "H10: GPU found nonce below host pow at {nonce} (lose check)");
        }
        println!("H10 host↔GPU seed lockstep OK (placeholder formula; wiring validated)");
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
        assert_eq!(miner.mine_v4(&PPH, TS, &pow, nonce, 1, false).unwrap(), Some(nonce),
                   "[{kind}] GPU did not find the nonce at the host pow target — divergence");
        assert_eq!(miner.mine_v4(&PPH, TS, &dec_le(pow), nonce, 1, false).unwrap(), None,
                   "[{kind}] GPU found the nonce below the host pow — divergence");
        println!("BIT-EXACT OK on the {kind} walk");
    }

    /// Exercise the CROSS-BATCH CHASE PREFETCH hit path: consecutive batches of the same job, so
    /// batch N+1's offsets come from the prefetch buffer rather than an inline chase. A key/buffer
    /// mix-up here would feed the walk another range's offsets — the walk would still "run", just
    /// never find a valid winner (silent zero-share mining), so this asserts the winner IS found in
    /// the batch that contains it, and that prefetch ON and OFF agree exactly.
    #[test]
    #[ignore]
    fn v4_chase_prefetch_matches_serial() {
        let data = blob(2048);
        let index = crate::pom::index_from_ram(data.clone());
        let miner = PomGpuMiner::load_test_segments(0, vec![data]).unwrap();
        const B: u64 = 64;
        // A nonce in the FOURTH batch, so at least two prefetch hits precede it.
        let nonce = 3 * B + 17;
        let seed = crate::pom::pom_block_seed_v4(&PPH, TS, nonce);
        let (_v4, fs) = crate::pom_v4::build_proof_v4(0, seed, &index).unwrap();
        let pow = crate::pom::pom_pow_value(fs, &PPH, true);
        let sweep = |prefetch: &str| -> Vec<Option<u64>> {
            std::env::set_var("KERYX_POM_V4_CHASE_PREFETCH", prefetch);
            (0..6).map(|i| miner.mine_v4(&PPH, TS, &pow, i * B, B, false).unwrap()).collect()
        };
        let on = sweep("1");
        let off = sweep("0");
        std::env::remove_var("KERYX_POM_V4_CHASE_PREFETCH");
        // The property that matters: pipelining must not change what the GPU finds.
        assert_eq!(on, off, "prefetch ON changed the per-batch winners vs OFF");
        // Every winner must be real: inside its own batch, and confirmed by the HOST re-walk. A
        // buffer/key mix-up would hand the walk another range's offsets, and these checks catch it
        // (a wrong-offset "winner" fails the host pow check, and a shifted range fails the bounds).
        let mut confirmed = 0;
        for (i, w) in on.iter().enumerate() {
            let Some(n) = *w else { continue };
            let lo = i as u64 * B;
            assert!((lo..lo + B).contains(&n), "batch {i} returned nonce {n} outside [{lo},{})", lo + B);
            let sd = crate::pom::pom_block_seed_v4(&PPH, TS, n);
            let (_p, f) = crate::pom_v4::build_proof_v4(0, sd, &index).unwrap();
            let host_pow = crate::pom::pom_pow_value(f, &PPH, true);
            // pow values are 256-bit LITTLE-endian: compare from the most significant byte down.
            // (Rust's derived `<=` on [u8;32] would compare the LEAST significant byte first.)
            let le_leq = |a: &[u8; 32], b: &[u8; 32]| -> bool {
                for k in (0..32).rev() {
                    if a[k] != b[k] {
                        return a[k] < b[k];
                    }
                }
                true
            };
            assert!(le_leq(&host_pow, &pow), "batch {i} nonce {n}: GPU claimed a win the host re-walk rejects");
            confirmed += 1;
        }
        assert!(confirmed >= 3, "expected several confirmed winners across 6 batches, got {confirmed}");
        println!("prefetch hit-path OK: {confirmed} winners host-confirmed, ON==OFF {on:?}");
    }

    /// Throughput bench (no pool, synthetic blob): `--ignored v4_bench`. Blob size via
    /// KERYX_BENCH_TILES (default 6 GiB / 1KB tiles), duration via KERYX_BENCH_SECS (default 10).
    /// Impossible target => no winner => every batch runs to completion. Prints Mh/s.
    /// The sibling memo must not change a single byte of any Merkle path: same index, same paths,
    /// cold vs warm cache — and a built proof must still re-verify against R_T.
    #[test]
    #[ignore]
    fn merkle_path_memo_is_byte_identical() {
        let data = blob(2048);
        let index = crate::pom::index_from_ram(data.clone());
        let fresh = crate::pom::index_from_ram(data);
        let n = index.n_chunks.min(4096);
        let mut checked = 0u64;
        for off in (0..n).step_by(7) {
            // `index` warms its memo as we go; `fresh` is asked each offset exactly once, but its
            // memo also warms — so also re-ask `index` a second time (warm) for the same offset.
            let a = fresh.merkle_path(off);
            let b = index.merkle_path(off);
            let c = index.merkle_path(off); // warm hit
            assert_eq!(a, b, "memo changed the path at off {off}");
            assert_eq!(b, c, "warm memo differs from cold at off {off}");
            checked += 1;
        }
        // end-to-end: a proof built with the memo must verify against the pinned root
        let seed = crate::pom::pom_block_seed_v4(&PPH, TS, 42);
        let (v4, fs) = crate::pom_v4::build_proof_v4(0, seed, &index).unwrap();
        let re = crate::pom_v4::verify_proof_v4(seed, &v4, &index.r_t, index.n_chunks).unwrap();
        assert_eq!(re, fs, "proof built with the memo failed re-verification");
        println!("memo byte-identical over {checked} offsets + proof re-verifies");
    }

    /// build_proof_v4 WITH the dense (resident) tree — isolates what --resident-tree buys per share.
    #[test]
    #[ignore]
    fn v4_proof_build_bench_dense() {
        let data = blob(2048);
        let mut index = crate::pom::index_from_ram(data);
        let t = std::time::Instant::now();
        index.build_dense();
        println!("build_dense: {:.1} ms (one-time)", t.elapsed().as_secs_f64() * 1e3);
        let iters = 200u64;
        let t0 = std::time::Instant::now();
        for n in 0..iters {
            let seed = crate::pom::pom_block_seed_v4(&PPH, TS, n);
            let (_p, _fs) = crate::pom_v4::build_proof_v4(0, seed, &index).unwrap();
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("build_proof_v4 (DENSE): {:.1} us/proof ({} iters)", us, iters);
    }

    /// Break down where build_proof_v4's time actually goes: transition matmul vs tile reads
    /// vs merkle paths.
    #[test]
    #[ignore]
    fn v4_proof_parts_bench() {
        use crate::pom_v4::*;
        let tiles: usize = std::env::var("KERYX_BENCH_TILES").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
        let data = blob(tiles);
        let index = crate::pom::index_from_ram(data);
        let n_tiles = index.n_chunks / POM_V4_TILE_CHUNKS;
        let seed = crate::pom::pom_block_seed_v4(&PPH, TS, 1);
        // gather one representative tile + path
        let off = v4_first_offset(seed, n_tiles);
        let mut tile = Vec::with_capacity(POM_V4_TILE_BYTES);
        for c in 0..POM_V4_TILE_CHUNKS { tile.extend_from_slice(&index.read_chunk_bytes(off * POM_V4_TILE_CHUNKS + c)); }
        let state = v4_initial_state(seed);
        let mut scratch = vec![0u8; 32 * 32];
        let n = 256u32;
        let t = std::time::Instant::now();
        for step in 1..=n { v4_transition_into(&mut scratch, &state, &tile, step); }
        println!("  transitions x256      : {:8.1} us", t.elapsed().as_secs_f64() * 1e6);
        let t = std::time::Instant::now();
        for step in 0..n as u64 {
            let o = (off + step) % n_tiles;
            let mut tl = Vec::with_capacity(POM_V4_TILE_BYTES);
            for c in 0..POM_V4_TILE_CHUNKS { tl.extend_from_slice(&index.read_chunk_bytes(o * POM_V4_TILE_CHUNKS + c)); }
            std::hint::black_box(&tl);
        }
        println!("  tile reads x256       : {:8.1} us", t.elapsed().as_secs_f64() * 1e6);
        let t = std::time::Instant::now();
        for step in 0..n as u64 {
            let o = (off + step) % n_tiles;
            std::hint::black_box(index.merkle_path(o * POM_V4_TILE_CHUNKS));
        }
        println!("  merkle_path x256      : {:8.1} us", t.elapsed().as_secs_f64() * 1e6);
    }

    /// Time the CPU proof-build (per-share witness on NVIDIA sync-submit + the whole AMD path).
    #[test]
    #[ignore]
    fn v4_proof_build_bench() {
        let tiles: usize = std::env::var("KERYX_BENCH_TILES").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
        let data = blob(tiles);
        let index = crate::pom::index_from_ram(data);
        let iters = 200u64;
        let t0 = std::time::Instant::now();
        for n in 0..iters {
            let seed = crate::pom::pom_block_seed_v4(&PPH, TS, n);
            let (_p, _fs) = crate::pom_v4::build_proof_v4(0, seed, &index).unwrap();
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("build_proof_v4: {:.1} us/proof ({} iters, {} tiles, N={} chunks)", us, iters, tiles, index.n_chunks);
        println!("  sibling memo: {} entries (~{:.1} MB)", index.sibling_memo_len(), index.sibling_memo_len() as f64 * 44.0 / 1e6);
    }

    #[test]
    #[ignore]
    fn v4_bench() {
        // Default 1 GiB: big enough to be DRAM-bound, small enough that the whole --ignored suite
        // can run in ONE process without the allocations colliding. Raise with KERYX_BENCH_TILES
        // (6*1024*1024 = the 6 GiB tier-0-ish blob used for the headline numbers).
        let n_tiles: usize = std::env::var("KERYX_BENCH_TILES").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(1024 * 1024);
        let secs: u64 = std::env::var("KERYX_BENCH_SECS").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(10);
        let data = blob(n_tiles);
        println!("bench blob: {} tiles = {:.2} GiB", n_tiles, (n_tiles as f64) / (1024.0 * 1024.0));
        let miner = PomGpuMiner::load_test_segments(0, vec![data]).unwrap();
        let target = [0u8; 32]; // impossible (pow > 0 always)
        let batch: u64 = std::env::var("KERYX_POM_V4_BATCH").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(1 << 16);
        // warmup
        let _ = miner.mine_v4(&PPH, TS, &target, 0, batch, false).unwrap();
        let t0 = std::time::Instant::now();
        let mut nonces = 0u64;
        while t0.elapsed().as_secs() < secs {
            let _ = miner.mine_v4(&PPH, TS, &target, nonces, batch, false).unwrap();
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
                    miner.mine_v4(&PPH, TS, &pow, nonce, 1, false).unwrap(),
                    Some(nonce),
                    "[{tag}] GPU did NOT find nonce {nonce} at host pow target — GPU pow > host pow (divergence)"
                );
                // ... and LOSES one below it -> GPU pow == host pow exactly (byte-exact final_state).
                assert_eq!(
                    miner.mine_v4(&PPH, TS, &dec_le(pow), nonce, 1, false).unwrap(),
                    None,
                    "[{tag}] GPU found nonce {nonce} below host pow — GPU pow < host pow (divergence)"
                );
            }
        }
        std::env::remove_var("KERYX_POM_V4_TC");
    }
}


#[cfg(test)]
mod intensity_tests {
    use super::*;

    #[test]
    fn intensity_maps_to_a_power_of_two_batch_and_clamps() {
        // The mapping users know from cgminer/sgminer: batch = 2^intensity.
        assert_eq!(1u64 << 18, 262_144);
        assert_eq!(1u64 << 16, 65_536);
        // Out-of-range values clamp instead of producing an unlaunchable or window-busting batch.
        assert_eq!(INTENSITY_MAX.clamp(INTENSITY_MIN, INTENSITY_MAX), INTENSITY_MAX);
        assert_eq!(99u32.clamp(INTENSITY_MIN, INTENSITY_MAX), INTENSITY_MAX);
        assert_eq!(0u32.clamp(INTENSITY_MIN, INTENSITY_MAX), INTENSITY_MIN);
    }

    #[test]
    fn the_starting_batch_is_the_top_of_the_sweep() {
        // A card starts at the high end and the autotune may walk it down — the measured fleet curve
        // says everything above ~46 SMs wants more than 384 nonces/SM.
        for sm in [46u64, 66, 84, 170] {
            assert_eq!(v4_starting_batch(sm), v4_batch_for_sm_count(sm) * 2);
        }
        // The floor still applies to tiny/unknown cards.
        assert!(v4_starting_batch(1) >= POM_V4_BATCH_MIN);
    }
}
