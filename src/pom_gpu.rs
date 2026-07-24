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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::cuda_backend::cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchArgs, LaunchConfig, PushKernelArg,
};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{CudaDevice, Device};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine.ptx"));
/// The arch the walk PTX was compiled for (build.rs bakes POM_CUDA_ARCH in) — modern=sm_75,
/// legacy=sm_70, pascal=sm_60. Used to explain arch-mismatch load failures.
const PTX_ARCH: &str = env!("POM_PTX_ARCH");
const CHUNK_BYTES: usize = 32;

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

fn walk_load_err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!(
        "PoM walk kernel failed to load ({e}). This build's walk PTX targets {PTX_ARCH}, and PTX \
         only JIT-compiles onto {PTX_ARCH}-or-newer GPUs — on an older card the driver returns \
         CUDA_ERROR_INVALID_PTX. Use the build line that matches your GPU: LEGACY = sm_70+ \
         (Volta/V100, CMP 100-210, Turing and newer), PASCAL = sm_60/61 (GTX 10-series), \
         MODERN = sm_75+ with driver 575+. If you built from source: CUDA 13.x cannot compile \
         for Volta or Pascal — use a CUDA 12.x toolkit and set POM_CUDA_ARCH=compute_70 (Volta) \
         or compute_60 (Pascal), plus CUDA_COMPUTE_CAP to match."
    ))
}

/// Load the walk kernel, turning a bare driver error (typically CUDA_ERROR_INVALID_PTX when the
/// GPU is OLDER than the PTX target — PTX only JITs forward) into an actionable message naming
/// this build's PTX arch and the right build line for older cards.
fn load_walk_func(cuda: &CudaDevice) -> candle_core::Result<WalkFunc> {
    let stream = cuda.cuda_stream();
    let ctx = stream.context().clone();
    let module = {
        let mut cache = WALK_MODULES.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
        match cache.get(&ctx.ordinal()) {
            Some(m) => m.clone(),
            None => {
                let m = ctx.load_module(PTX.into()).map_err(walk_load_err)?;
                cache.insert(ctx.ordinal(), m.clone());
                m
            }
        }
    };
    let func = module.load_function("pom_mine").map_err(walk_load_err)?;
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
        let _ = load_walk_func(&cuda)?;

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
        let _ = load_walk_func(&cuda)?;

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
        let _ = load_walk_func(&cuda)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: raw, _shared: kept_shared, _uploads: Vec::new() })
    }

    /// Phase-2 zero-dup over the IN-PROCESS llama.cpp engine (candle hosts nothing): bases/prefix
    /// straight over the engine's resident device tensors in canonical name-sorted order (the
    /// wrapper pre-sorts; byte-identity to the on-disk GGUF proven by tools/llama_zerodup_spike).
    /// The few host-resident tensors (e.g. token_embd stays on the CPU buffer) get a small device
    /// upload of our own. candle's CudaDevice here is pure CUDA plumbing (context/stream/kernel).
    pub fn load_llama(device_id: usize) -> candle_core::Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();
        let ts = crate::llama_engine::tensors()
            .ok_or_else(|| candle_core::Error::Msg("PoM GPU: llama engine tensors unavailable".into()))?;
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut uploads: Vec<CudaSlice<u8>> = Vec::new();
        let mut n_uploaded = 0usize;
        for (_name, ptr, nbytes, is_dev) in &ts {
            let chunks = (nbytes / CHUNK_BYTES) as u64;
            if chunks == 0 {
                continue;
            }
            let base = if *is_dev {
                *ptr
            } else {
                // Host-resident in ggml (CPU buffer): the walk needs device memory — upload our
                // own copy of the raw bytes (identical to the GGUF bytes, same as the pointer).
                let host: &[u8] = unsafe { std::slice::from_raw_parts(*ptr as *const u8, *nbytes) };
                let dev = stream.clone_htod(host).map_err(candle_core::Error::wrap)?;
                use candle_core::cuda_backend::cudarc::driver::DevicePtr;
                let p = dev.device_ptr(&stream).0 as u64;
                uploads.push(dev);
                n_uploaded += 1;
                p
            };
            bases.push(base);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: llama engine produced 0 chunks".into()));
        }
        info!(
            "PoM llama zero-dup gather: {} tensors ({} host-resident uploaded), N={} chunks",
            bases.len(), n_uploaded, n_total_chunks
        );
        // BYTE GATE (consensus safety): the pool does not deep-verify every share, so a wrong
        // gather would mine garbage silently. Read back evenly-spaced chunks from the llama-owned
        // device memory and compare them byte-for-byte against the host index (GGUF pread) —
        // any mismatch refuses to mine. (Full-model byte-identity for this llama build was proven
        // once by tools/llama_zerodup_spike; this guards every startup against regressions.)
        let idx_ref = crate::pom::active_index_for(device_id as u32)
            .map(|t| &t.0)
            .or_else(|| crate::pom::active_index().map(|(i, _)| i));
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
        let _ = load_walk_func(&cuda)?;
        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: Vec::new(), _shared: Vec::new(), _uploads: uploads })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest nonce whose `pom_pow_value`
    /// is `<= target_le`, or None. `target_le` is the header's compact target as 32 LE bytes.
    /// `h3` salts the pph words host-side (`POM_H3_PPH_SALT`) — the PTX kernel is era-agnostic,
    /// it folds whatever words it receives, so there is no kernel change at the H3 gate.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h3: bool, walk_v2: bool) -> candle_core::Result<Option<u64>> {
        let p = crate::pom::pph_words_for_era(pre_pow_hash, h3);
        let t = words4(target_le);
        let k = crate::pom::POM_WALK_STEPS;
        let winner = self.stream.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;
        let grid = ((batch + 255) / 256) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let wv2: u32 = walk_v2 as u32; // H5 era flag -> kernel `unsigned int walk_v2`

        let func = load_walk_func(&self.cuda)?; // cached
        let mut b = func.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&self.n_total_chunks).arg(&k)
            .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3]).arg(&timestamp)
            .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
            .arg(&start).arg(&batch).arg(&winner).arg(&wv2);
        unsafe { b.launch(cfg).map_err(candle_core::Error::wrap)?; }
        self.stream.synchronize().map_err(candle_core::Error::wrap)?;

        let w = self.stream.clone_dtoh(&winner).map_err(candle_core::Error::wrap)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
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
fn remove_device_entry<T>(map: &mut HashMap<u32, T>, device_id: u32) {
    map.remove(&device_id);
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
    if let Ok(mut g) = miners().lock() {
        remove_device_entry(&mut g, device_id);
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

/// Convenience: search a nonce batch via the installed miner for a specific device.
pub fn mine(device_id: u32, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h3: bool, walk_v2: bool) -> Option<u64> {
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    miner.mine(pre_pow_hash, timestamp, target_le, start, batch, h3, walk_v2).ok().flatten()
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

/// The model this device mines: its per-device override if set, else the process-wide default.
pub fn device_model(device_id: u32) -> Option<([u8; 32], String)> {
    if let Ok(m) = device_models().lock() {
        if let Some(v) = m.get(&device_id) {
            return Some(v.clone());
        }
    }
    MINING_TIER.get().cloned()
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
pub fn ensure_installed(device_id: u32, daa: u64) -> bool {
    if is_installed(device_id) {
        return true;
    }
    // Flag the heavy load so the stall watchdog stays benign while the worker is blocked here.
    LOADING.fetch_add(1, Ordering::Relaxed);
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
            match crate::pom::WeightIndex::build_from_gguf(&gguf) {
                Ok(idx) => {
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
    let mut use_llama = device_id as usize == inference_gpu && crate::llama_engine::ensure_loaded(&gguf, inference_gpu);
    // BYTE-COMPAT GATE: llama.cpp repacks some architectures on load (e.g. Gemma-3 materialises a
    // separate output.weight from its tied embeddings), so its resident chunk count differs from
    // the canonical GGUF the walk MUST gather and R_T pins. When that happens the zero-dup walk is
    // impossible — free llama's VRAM and walk a raw canonical copy (`load_raw`) instead (inference
    // for this model is then unavailable). Cheaper than letting load_llama build a doomed gather
    // that the N-guard then rejects into a mine-refusing loop.
    if use_llama {
        let host_n = crate::pom::active_index_for(device_id as u32).map(|t| t.0.n_chunks)
            .or_else(|| crate::pom::active_index().map(|(i, _)| i.n_chunks));
        let llama_n = crate::llama_engine::tensors().map(|ts| {
            ts.iter().map(|(_, _, nbytes, _)| (*nbytes / CHUNK_BYTES) as u64).sum::<u64>()
        });
        if let (Some(hn), Some(ln)) = (host_n, llama_n) {
            if ln != hn {
                info!(
                    "PoM[gpu{}]: llama-resident layout N={} != canonical N={} (llama repacks this model arch) — walking a raw canonical copy; inference for this model is unavailable.", device_id, ln, hn
                );
                crate::llama_engine::unload();
                use_llama = false;
            }
        }
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
            PomGpuMiner::load_llama(device_id as usize)
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
            install(device_id, gm);
            info!("PoM[gpu{}]: GPU miner ready — N={} chunks resident (matches host index)", device_id, n);
            true
        }
        Err(e) => {
            log::error!("PoM[gpu{}]: device miner build failed: {}", device_id, e);
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

    #[test]
    fn remove_device_entry_on_missing_device_is_a_no_op() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(1, "gpu1-miner");

        remove_device_entry(&mut map, 0);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"gpu1-miner"));
    }
}
