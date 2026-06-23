// PomMiner — the AMD-side OpenCL PoM mining driver (destined for the keryxopencl plugin).
// Holds the tier weight blob resident in one cl_mem buffer; mine() launches pom_mine over a
// nonce batch and returns the lowest passing nonce (the host then re-verifies + builds the proof).
// Mirrors the opencl3 0.6 patterns in plugins/opencl/src/worker.rs.

use std::ptr;
use std::sync::Arc;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_ulong, CL_BLOCKING};

pub const POM_WALK_STEPS: u32 = 256;
const POM_SRC: &str = include_str!("../plugins/opencl/resources/pom_mine.cl");

pub struct PomMiner {
    _context: Arc<Context>,
    queue: CommandQueue,
    kernel: Kernel,
    weights: Buffer<cl_ulong>,
    winner: Buffer<cl_ulong>,
    pub n_chunks: u64,
}

// OpenCL handles are plain cl_* pointers usable from any single thread; the global Mutex
// serializes all access (one mining thread), so sending the miner across threads is sound.
unsafe impl Send for PomMiner {}

impl PomMiner {
    /// `blob` = the tier in canonical chunk order, N*4 little-endian u64.
    pub fn new(device: Device, blob: &[u64], n_chunks: u64) -> Result<Self, String> {
        let context = Arc::new(Context::from_device(&device).map_err(|e| e.to_string())?);
        let queue = CommandQueue::create(&context, device.id(), 0).map_err(|e| e.to_string())?;
        let program = Program::create_and_build_from_source(&context, POM_SRC, "")?;
        let kernel = Kernel::create(&program, "pom_mine").map_err(|e| e.to_string())?;
        // worker.rs pattern: a context ref that outlives the borrow checker (Arc kept in struct).
        let cref = unsafe { Arc::as_ptr(&context).as_ref().unwrap() };
        let mut weights = Buffer::<cl_ulong>::create(cref, CL_MEM_READ_ONLY, blob.len(), ptr::null_mut())
            .map_err(|e| e.to_string())?;
        queue.enqueue_write_buffer(&mut weights, CL_BLOCKING, 0, blob, &[]).map_err(|e| e.to_string())?;
        let winner = Buffer::<cl_ulong>::create(cref, CL_MEM_READ_WRITE, 1, ptr::null_mut())
            .map_err(|e| e.to_string())?;
        Ok(Self { _context: context, queue, kernel, weights, winner, n_chunks })
    }

    /// Launch one batch of `batch` nonces from `nonce_base`. Returns the lowest nonce whose
    /// pom_pow_value <= target, or None.
    pub fn mine(&mut self, pph: [u64; 4], time: u64, target: [u64; 4], nonce_base: u64, batch: u64) -> Option<u64> {
        self.queue.enqueue_write_buffer(&mut self.winner, CL_BLOCKING, 0, &[u64::MAX], &[]).ok()?;
        ExecuteKernel::new(&self.kernel)
            .set_arg(&self.weights)
            .set_arg(&self.n_chunks)
            .set_arg(&POM_WALK_STEPS)
            .set_arg(&pph[0]).set_arg(&pph[1]).set_arg(&pph[2]).set_arg(&pph[3])
            .set_arg(&time)
            .set_arg(&target[0]).set_arg(&target[1]).set_arg(&target[2]).set_arg(&target[3])
            .set_arg(&nonce_base).set_arg(&batch)
            .set_arg(&self.winner)
            .set_global_work_size(batch as usize)
            .enqueue_nd_range(&self.queue)
            .ok()?;
        self.queue.finish().ok()?;
        let mut w = [u64::MAX];
        self.queue.enqueue_read_buffer(&self.winner, CL_BLOCKING, 0, &mut w, &[]).ok()?;
        if w[0] == u64::MAX { None } else { Some(w[0]) }
    }
}

// ============================================================================
// Module interface mirroring upstream `pom_gpu` (candle-CUDA) so miner.rs calls
// are identical: load_tier() once at startup, then mine() in the GPU loop.
// AMD-specific: weights are loaded into our own OpenCL buffer (candle is CPU here).
// ============================================================================
use std::sync::Mutex;

static POM: Mutex<Option<PomMiner>> = Mutex::new(None);

fn words(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for i in 0..4 { w[i] = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()); }
    w
}

/// Install the resident tier into the first OpenCL GPU. `blob` = N*4 LE u64 (canonical order).
pub fn install(blob: &[u64], n_chunks: u64) -> Result<(), String> {
    let dev_ids = opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
        .map_err(|e| e.to_string())?;
    let id = *dev_ids.first().ok_or("PoM: no OpenCL GPU device")?;
    let miner = PomMiner::new(opencl3::device::Device::new(id), blob, n_chunks)?;
    *POM.lock().unwrap() = Some(miner);
    Ok(())
}

pub fn is_installed() -> bool { POM.lock().unwrap().is_some() }

/// Registered mining tier (GGUF path, tier index). Set once at startup via set_mining_tier;
/// the first PoM-active job lazily builds the index + GPU residency via ensure_installed.
static TIER: Mutex<Option<(String, u8)>> = Mutex::new(None);

/// Register the tier to mine (GGUF path on disk, POM_TIERS index). Cheap — no I/O. The heavy
/// load_tier (build the Merkle tree + upload the blob) runs lazily on the first PoM-active job.
pub fn set_mining_tier(gguf_path: String, tier: u8) {
    *TIER.lock().unwrap() = Some((gguf_path, tier));
}

/// Lazily build + install the registered tier on the first PoM-active iteration. Idempotent.
/// (Unlike NVIDIA, the AMD OpenCL buffer is never evicted, so this only does work once.)
pub fn ensure_installed() {
    if is_installed() {
        return;
    }
    let tier = TIER.lock().unwrap().clone();
    match tier {
        Some((path, t)) => match load_tier(&path, t) {
            Ok(()) => log::info!("PoM: tier {t} installed (GPU-resident)."),
            Err(e) => log::warn!("PoM: load_tier failed ({path}): {e} — is the model GGUF downloaded?"),
        },
        None => log::warn!("PoM: no mining tier registered (set_mining_tier not called)."),
    }
}

/// Grind one batch of `batch` nonces from `nonce_base`. Returns the lowest nonce whose
/// pom_pow_value <= target, or None. pph/target are the 32-byte LE forms from State.
pub fn mine(pph: &[u8; 32], time: u64, target_le: &[u8; 32], nonce_base: u64, batch: u64) -> Option<u64> {
    let p = words(pph);
    let t = words(target_le);
    let mut g = POM.lock().unwrap();
    g.as_mut()?.mine(p, time, t, nonce_base, batch)
}

/// Build the resident tier from a GGUF: WeightIndex (proof side, CPU/disk) + the contiguous
/// GPU blob (search side), register both. `tier` is the POM_TIERS slice index.
/// Call once at startup when PoM mining for this tier.
pub fn load_tier(gguf_path: &str, tier: u8) -> Result<(), String> {
    log::info!("PoM: building WeightIndex from {gguf_path} (tier {tier})…");
    let index = crate::pom::WeightIndex::build_from_gguf(gguf_path).map_err(|e| e.to_string())?;
    let n = index.n_chunks;
    log::info!(
        "PoM: tier {tier} loaded — {n} chunks, computed R_T = {} (must match the node's pinned root)",
        hex32(&index.r_t)
    );
    // Build the contiguous GPU blob in canonical chunk order. read_chunk guarantees the SAME
    // indexing the proof side uses. TODO(perf): bulk-read for the big tiers (this is O(N) preads).
    let mut blob: Vec<u64> = Vec::with_capacity((n * 4) as usize);
    for off in 0..n { blob.extend_from_slice(&index.read_chunk(off)); }
    crate::pom::set_index(index, tier);
    install(&blob, n)?;
    log::info!("PoM: tier {tier} resident on GPU ({} MiB).", (n * 32) / (1024 * 1024));
    Ok(())
}

fn hex32(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for x in b { s.push_str(&format!("{x:02x}")); }
    s
}
