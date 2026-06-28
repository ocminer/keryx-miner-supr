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

use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{CudaDevice, Device};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine.ptx"));
const CHUNK_BYTES: usize = 32;

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
}

impl PomGpuMiner {
    /// Load the mining model's GGUF into candle (device 0), build the gather index, load the kernel.
    pub fn load(gguf_path: &str) -> candle_core::Result<Self> {
        let device = Device::new_cuda(0)?;
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
        let _ = cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: tensors, _shared: Vec::new() })
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
        let _ = cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: raw, _shared: kept_shared })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest nonce whose `pom_pow_value`
    /// is `<= target_le`, or None. `target_le` is the header's compact target as 32 LE bytes.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> candle_core::Result<Option<u64>> {
        let p = words4(pre_pow_hash);
        let t = words4(target_le);
        let k = crate::pom::POM_WALK_STEPS;
        let winner = self.stream.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;
        let grid = ((batch + 255) / 256) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

        let func = self.cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?; // cached
        let mut b = func.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&self.n_total_chunks).arg(&k)
            .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3]).arg(&timestamp)
            .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
            .arg(&start).arg(&batch).arg(&winner);
        unsafe { b.launch(cfg).map_err(candle_core::Error::wrap)?; }
        self.stream.synchronize().map_err(candle_core::Error::wrap)?;

        let w = self.stream.clone_dtoh(&winner).map_err(candle_core::Error::wrap)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }
}

// The GPU miner instance. Option (not OnceLock) so it can be uninstalled to free VRAM when an
// inference for another model needs the GPU (inference has priority over PoW), then reinstalled
// when mining resumes.
static MINER: Mutex<Option<PomGpuMiner>> = Mutex::new(None);

/// Install the GPU miner (after loading/sharing the mining model's resident weights).
pub fn install(m: PomGpuMiner) {
    if let Ok(mut g) = MINER.lock() {
        *g = Some(m);
    }
}

/// Drop the GPU miner, releasing its hold on the mining model's VRAM (shared Arcs + gather) so
/// the inference engine can load another model. Mining is paused during inference anyway.
pub fn uninstall() {
    if let Ok(mut g) = MINER.lock() {
        *g = None;
    }
}

/// Whether the GPU miner is currently installed.
pub fn is_installed() -> bool {
    MINER.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// True while the GPU miner is being (re)built — a heavy one-time model load that blocks the
/// mining worker. The PoW stall watchdog treats this like an inference pause, not a crash.
static LOADING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a PoM model load/rebuild is in progress (worker intentionally paused, not stalled).
pub fn is_loading() -> bool {
    LOADING.load(std::sync::atomic::Ordering::Relaxed)
}

/// Convenience: search a nonce batch via the installed miner. None if not installed or no winner.
pub fn mine(pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> Option<u64> {
    let g = MINER.lock().ok()?;
    g.as_ref()?.mine(pre_pow_hash, timestamp, target_le, start, batch).ok().flatten()
}

/// Mining-tier identity for rebuilds: (model_id, gguf_path). Set once at startup.
static MINING_TIER: OnceLock<([u8; 32], String)> = OnceLock::new();

/// Record the mining tier so the miner can be rebuilt after an inference swapped the model away.
pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String) {
    let _ = MINING_TIER.set((model_id, gguf_path));
}

/// Ensure the GPU miner is installed; if an inference evicted the mining model, reload it
/// (resident again) and rebuild the zero-dup gather. Heavy (model reload) but only when needed —
/// inference has priority, so mining reloads its model when it next gets the GPU. Returns true if
/// the miner is ready to mine.
pub fn ensure_installed() -> bool {
    if is_installed() {
        return true;
    }
    // Flag the heavy load so the stall watchdog stays benign while the worker is blocked here.
    LOADING.store(true, std::sync::atomic::Ordering::Relaxed);
    let ok = ensure_installed_inner();
    LOADING.store(false, std::sync::atomic::Ordering::Relaxed);
    ok
}

fn ensure_installed_inner() -> bool {
    let (model_id, gguf) = match MINING_TIER.get() {
        Some(x) => x,
        None => return false,
    };
    // Build the possession index once (host, heavy) the first time PoM activates — deferred from
    // boot so the pre-PoM legacy phase starts immediately and keeps host/GPU free.
    if crate::pom::active_index().is_none() {
        // The background prefetch may still be downloading the mining-tier model (slow IPFS link /
        // small HiveOS system disk). Building the possession index from a missing or partial GGUF
        // hard-fails with ENOENT ("index build failed: no such file or directory") and would spam
        // that on every job. Wait for the `.ok` completion sentinel and retry next job instead.
        let ready = std::path::Path::new(gguf)
            .parent()
            .map(|d| d.join(".ok"))
            .map_or(false, |p| p.exists());
        if !ready {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                info!(
                    "PoM: mining-tier model not downloaded yet — deferring the possession-index build \
                     until the background prefetch finishes (slow link / small disk can take a while)."
                );
            }
            return false;
        }
        let tier = match crate::models::pom_tier_index(model_id) {
            Some(t) => t,
            None => return false,
        };
        info!("PoM: building possession index (first PoM activation) — this can take a while…");
        match crate::pom::WeightIndex::build_from_gguf(gguf) {
            Ok(idx) => {
                info!("PoM: weight index ready — N={} chunks", idx.n_chunks);
                crate::pom::set_index(idx, tier);
            }
            Err(e) => {
                log::error!("PoM: index build failed: {}", e);
                return false;
            }
        }
    }
    // Make the mining model resident again (evicts whatever inference loaded), then share it.
    if !crate::slm::ensure_loaded(model_id) {
        return false;
    }
    let m = if let Some((device, shared)) = crate::slm::pom_shared(model_id) {
        PomGpuMiner::load_shared(gguf, &device, &shared)
    } else {
        PomGpuMiner::load(gguf)
    };
    match m {
        Ok(gm) => {
            let n = gm.n_chunks();
            // N-guard: the gather must match the host index, else blocks would be rejected.
            if let Some((idx, _)) = crate::pom::active_index() {
                if n != idx.n_chunks {
                    log::error!("PoM: gather N={} != index N={} — refusing to mine (rejected blocks)", n, idx.n_chunks);
                    return false;
                }
            }
            install(gm);
            info!("PoM: GPU miner ready — N={} chunks resident (matches index)", n);
            true
        }
        Err(e) => {
            log::error!("PoM: rebuild failed: {}", e);
            false
        }
    }
}
