//! Proof-of-Model GPU mining on NVIDIA — cudarc driver (CUDA analog of `crate::pom_opencl`).
//!
//! Holds the tier weight blob resident in one device buffer (N*4 LE u64, canonical chunk order
//! from `pom::WeightIndex::read_chunk` — the SAME bytes the proof side uses). `mine()` launches
//! the `pom_mine` kernel over a nonce batch and returns the lowest passing nonce; the host then
//! re-verifies + builds the proof. Exposes the SAME free-function interface as `pom_opencl` so
//! `miner.rs`/`main.rs` drive AMD (OpenCL) and NVIDIA (CUDA) identically.
//!
//! Uses cudarc directly (NOT candle's CUDA backend): the supr fork's candle is CPU-only and pins
//! cudarc 0.13.9, whose API differs from upstream's candle-gather `pom_gpu`. A dedicated
//! contiguous blob + a simple indexing kernel (vs upstream's per-tensor gather) is the proven AMD
//! approach and avoids the candle/cudarc version entanglement — same chunk bytes, byte-exact folds.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use log::info;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine.ptx"));

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

pub struct PomGpuMiner {
    dev: Arc<CudaDevice>,
    func: CudaFunction,
    weights: CudaSlice<u64>,
    pub n_chunks: u64,
}

// cudarc handles are bound to the device/thread; the global Mutex serializes access (single
// mining thread), so sending the miner across threads is sound.
unsafe impl Send for PomGpuMiner {}

impl PomGpuMiner {
    /// `blob` = the tier in canonical chunk order, N*4 little-endian u64. Uploads it to GPU 0.
    pub fn new(blob: Vec<u64>, n_chunks: u64) -> Result<Self, String> {
        let dev = CudaDevice::new(0).map_err(|e| e.to_string())?;
        dev.load_ptx(Ptx::from_src(PTX_SRC), "pom_mine_mod", &["pom_mine"]).map_err(|e| e.to_string())?;
        let func = dev.get_func("pom_mine_mod", "pom_mine").ok_or("PoM: pom_mine kernel not found")?;
        let weights = dev.htod_copy(blob).map_err(|e| e.to_string())?;
        Ok(Self { dev, func, weights, n_chunks })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest nonce whose `pom_pow_value`
    /// is `<= target_le`, or None. pph/target are the 32-byte LE forms from State.
    pub fn mine(&self, pph: &[u8; 32], time: u64, target_le: &[u8; 32], start: u64, batch: u64) -> Option<u64> {
        let p = words4(pph);
        let t = words4(target_le);
        let pph_d = self.dev.htod_copy(p.to_vec()).ok()?;
        let tgt_d = self.dev.htod_copy(t.to_vec()).ok()?;
        let mut winner_d = self.dev.htod_copy(vec![u64::MAX]).ok()?;
        let k = crate::pom::POM_WALK_STEPS as u32;
        let grid = ((batch + 255) / 256) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let func = self.func.clone();
        // 9 args = the launch tuple-arity max in cudarc 0.13.
        unsafe {
            func.launch(
                cfg,
                (&self.weights, self.n_chunks, k, &pph_d, time, &tgt_d, start, batch, &mut winner_d),
            )
            .ok()?;
        }
        let w = self.dev.dtoh_sync_copy(&winner_d).ok()?;
        if w[0] == u64::MAX {
            None
        } else {
            Some(w[0])
        }
    }
}

// ============================================================================
// Module interface — mirrors `crate::pom_opencl` so miner.rs/main.rs call the
// same free functions for AMD (OpenCL) and NVIDIA (CUDA).
// ============================================================================

static MINER: Mutex<Option<PomGpuMiner>> = Mutex::new(None);

pub fn install(m: PomGpuMiner) {
    *MINER.lock().unwrap() = Some(m);
}

pub fn is_installed() -> bool {
    MINER.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Grind one batch of `batch` nonces from `nonce_base`. Returns the lowest passing nonce, or None.
pub fn mine(pph: &[u8; 32], time: u64, target_le: &[u8; 32], nonce_base: u64, batch: u64) -> Option<u64> {
    let g = MINER.lock().ok()?;
    g.as_ref()?.mine(pph, time, target_le, nonce_base, batch)
}

/// Registered mining tier (GGUF path, POM_TIERS index). Set once at startup; the first PoM-active
/// job lazily builds the index + GPU residency via ensure_installed.
static TIER: Mutex<Option<(String, u8)>> = Mutex::new(None);

/// Register the tier to mine (GGUF path on disk, POM_TIERS index). Cheap — no I/O.
pub fn set_mining_tier(gguf_path: String, tier: u8) {
    *TIER.lock().unwrap() = Some((gguf_path, tier));
}

/// Lazily build + install the registered tier on the first PoM-active iteration. Idempotent.
pub fn ensure_installed() {
    if is_installed() {
        return;
    }
    let tier = TIER.lock().unwrap().clone();
    match tier {
        Some((path, t)) => match load_tier(&path, t) {
            Ok(()) => info!("PoM: tier {t} installed (CUDA GPU-resident)."),
            Err(e) => log::warn!("PoM: load_tier failed ({path}): {e} — is the model GGUF downloaded?"),
        },
        None => log::warn!("PoM: no mining tier registered (set_mining_tier not called)."),
    }
}

/// Build the resident tier from a GGUF: `WeightIndex` (proof side, CPU/disk) + the contiguous GPU
/// blob (search side), register both. `tier` is the POM_TIERS slice index.
pub fn load_tier(gguf_path: &str, tier: u8) -> Result<(), String> {
    info!("PoM: building WeightIndex from {gguf_path} (tier {tier})…");
    let index = crate::pom::WeightIndex::build_from_gguf(gguf_path).map_err(|e| e.to_string())?;
    let n = index.n_chunks;
    info!(
        "PoM: tier {tier} index ready — {n} chunks, R_T = {} (must match the node's pinned root)",
        hex32(&index.r_t)
    );
    // Contiguous blob in canonical chunk order. read_chunk guarantees the SAME indexing the proof
    // side uses. TODO(perf): bulk-read for the big tiers (this is O(N) preads).
    let mut blob: Vec<u64> = Vec::with_capacity((n * 4) as usize);
    for off in 0..n {
        blob.extend_from_slice(&index.read_chunk(off));
    }
    crate::pom::set_index(index, tier);
    let gm = PomGpuMiner::new(blob, n)?;
    // N-guard: the GPU blob's chunk count MUST equal the host index's, else blocks get rejected.
    if gm.n_chunks() != n {
        return Err(format!("gather N={} != index N={} — refusing to mine (would be rejected)", gm.n_chunks(), n));
    }
    install(gm);
    info!("PoM: tier {tier} resident on CUDA GPU ({} MiB).", (n * 32) / (1024 * 1024));
    Ok(())
}

fn hex32(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}
