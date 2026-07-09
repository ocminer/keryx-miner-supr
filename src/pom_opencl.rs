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

/// Staging window for the VRAM upload: 2^22 chunks × 32 B = 128 MiB. The blob is streamed
/// GGUF → window → cl_mem, so no full host copy of the tier ever exists (the old design cached
/// the whole blob in system RAM — ~2.5 GB for Gemma, ~28 GB for the 70B tier). Cards stream
/// CONCURRENTLY (one window each), so the window is kept moderate: N cards hold N windows of
/// transient host RAM, and each pread is still multi-MB (readahead-friendly, seek cost amortized).
const UPLOAD_WINDOW_CHUNKS: u64 = 1 << 22;

/// Max nonces per kernel launch. mine() grinds its batch in sub-dispatches of this size: one
/// huge NDRange (2^22 nonces × 256 dependent reads) can run multi-second on slow cards, which
/// trips the Windows TDR watchdog (~2 s) → device lost. 2^18 keeps a dispatch to ~0.5 s even
/// at 0.5 MH/s while amortizing launch overhead (<0.5% at MI60/7600 XT rates). Sub-batches run
/// in ascending nonce order, so the first one with a winner holds the batch's lowest nonce —
/// early-returning there is result-identical and submits the share sooner.
const SUB_DISPATCH_NONCES: u64 = 1 << 18;

impl PomMiner {
    /// Build the resident tier on `device` by streaming the canonical chunks straight from the
    /// shared WeightIndex (GGUF pread) into the cl_mem buffer through a bounded staging window.
    pub fn new(device: Device, index: &crate::pom::WeightIndex, n_chunks: u64) -> Result<Self, String> {
        let context = Arc::new(Context::from_device(&device).map_err(|e| e.to_string())?);
        let queue = CommandQueue::create(&context, device.id(), 0).map_err(|e| e.to_string())?;
        let program = Program::create_and_build_from_source(&context, POM_SRC, "")?;
        let kernel = Kernel::create(&program, "pom_mine").map_err(|e| e.to_string())?;
        // worker.rs pattern: a context ref that outlives the borrow checker (Arc kept in struct).
        let cref = unsafe { Arc::as_ptr(&context).as_ref().unwrap() };
        let mut weights = Buffer::<cl_ulong>::create(cref, CL_MEM_READ_ONLY, (n_chunks * 4) as usize, ptr::null_mut())
            .map_err(|e| e.to_string())?;
        let mut staging = vec![0u64; (UPLOAD_WINDOW_CHUNKS.min(n_chunks) * 4) as usize];
        let mut done: u64 = 0;
        while done < n_chunks {
            let n = (n_chunks - done).min(UPLOAD_WINDOW_CHUNKS);
            let words = &mut staging[..(n * 4) as usize];
            // Fill the window with the raw chunk bytes, then normalize to the LE word values the
            // kernel folds (u64::from_le is a no-op on our little-endian targets) — identical to
            // chunk_to_words / u64::from_le_bytes on every chunk.
            let bytes = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 8) };
            index.read_chunks_into(done, bytes);
            for w in words.iter_mut() {
                *w = u64::from_le(*w);
            }
            queue
                .enqueue_write_buffer(&mut weights, CL_BLOCKING, (done * 32) as usize, words, &[])
                .map_err(|e| e.to_string())?;
            done += n;
        }
        let winner = Buffer::<cl_ulong>::create(cref, CL_MEM_READ_WRITE, 1, ptr::null_mut())
            .map_err(|e| e.to_string())?;
        Ok(Self { _context: context, queue, kernel, weights, winner, n_chunks })
    }

    /// Grind one batch of `batch` nonces from `nonce_base` in TDR-safe sub-dispatches. Returns
    /// the lowest nonce whose pom_pow_value <= target, or None.
    pub fn mine(&mut self, pph: [u64; 4], time: u64, target: [u64; 4], nonce_base: u64, batch: u64) -> Option<u64> {
        let mut done: u64 = 0;
        while done < batch {
            let sub = (batch - done).min(SUB_DISPATCH_NONCES);
            let base = nonce_base.wrapping_add(done);
            self.queue.enqueue_write_buffer(&mut self.winner, CL_BLOCKING, 0, &[u64::MAX], &[]).ok()?;
            ExecuteKernel::new(&self.kernel)
                .set_arg(&self.weights)
                .set_arg(&self.n_chunks)
                .set_arg(&POM_WALK_STEPS)
                .set_arg(&pph[0]).set_arg(&pph[1]).set_arg(&pph[2]).set_arg(&pph[3])
                .set_arg(&time)
                .set_arg(&target[0]).set_arg(&target[1]).set_arg(&target[2]).set_arg(&target[3])
                .set_arg(&base).set_arg(&sub)
                .set_arg(&self.winner)
                .set_global_work_size(sub as usize)
                .enqueue_nd_range(&self.queue)
                .ok()?;
            self.queue.finish().ok()?;
            let mut w = [u64::MAX];
            self.queue.enqueue_read_buffer(&self.winner, CL_BLOCKING, 0, &mut w, &[]).ok()?;
            if w[0] != u64::MAX {
                return Some(w[0]);
            }
            done += sub;
        }
        None
    }
}

// ============================================================================
// Module interface mirroring upstream `pom_gpu` (candle-CUDA) so miner.rs calls
// are identical: load_tier() once at startup, then mine() in the GPU loop.
// AMD-specific: weights are loaded into our own OpenCL buffer (candle is CPU here).
//
// MULTI-GPU: there is ONE resident PomMiner PER GPU (keyed by cl_device_id), each in its
// own Mutex, so the N per-GPU mining threads grind PoM concurrently on their own card —
// instead of all funneling onto device 0 through a single global lock (the old design made
// 3 cards perform like 1). Each thread binds itself to its card once via bind_thread_device();
// install/is_installed/mine then act on that thread's bound device. The proof-side WeightIndex
// (`crate::pom`) stays a single shared global (CPU/disk, read-only) — only the GPU search
// buffer is per-card, so there is still exactly ONE on-disk pom-tree.
// ============================================================================
use std::cell::Cell;
use std::sync::Mutex;

/// Per-GPU resident miners, keyed by cl_device_id (as usize). Each behind its own Mutex so the
/// owning thread mines without blocking the other cards. Vec (tiny N) keeps a const initializer.
/// Each card streams its blob straight from the GGUF (no host-RAM blob cache — the OS page cache
/// makes the 2nd+ card's upload cheap).
static POM_BY_DEV: Mutex<Vec<(usize, Arc<Mutex<PomMiner>>)>> = Mutex::new(Vec::new());

thread_local! {
    /// The cl_device_id this mining thread owns (set once via bind_thread_device). All PoM
    /// driver calls on this thread act on this card.
    static BOUND_DEV: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Bind the calling mining thread to its GPU (cl_device_id as usize). Called once per GPU thread
/// before the PoM loop so every card mines its own shares. AMD-only; NVIDIA's seam is untouched.
pub fn bind_thread_device(device_id: usize) {
    BOUND_DEV.with(|d| d.set(Some(device_id)));
}

fn bound_dev() -> Option<usize> {
    BOUND_DEV.with(|d| d.get())
}

/// The cl_device_id this call targets: the thread's bound card, else the first OpenCL GPU
/// (single-device fallback / tests, which never call bind_thread_device).
fn target_dev() -> Option<usize> {
    bound_dev().or_else(|| {
        opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .ok()?
            .first()
            .map(|id| *id as usize)
    })
}

fn miner_for(device_id: usize) -> Option<Arc<Mutex<PomMiner>>> {
    POM_BY_DEV.lock().unwrap().iter().find(|(d, _)| *d == device_id).map(|(_, m)| m.clone())
}

fn words(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for i in 0..4 { w[i] = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()); }
    w
}

/// Cards whose streaming upload is currently in flight. Uploads for DIFFERENT cards run
/// concurrently (each thread streams its own card); a second caller for the SAME card waits
/// here instead of double-building (only the unbound/test fallback path can race like that —
/// production mining threads each bind a distinct device).
static INSTALLING: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Make the resident tier GPU-resident on `device_id` — its own OpenCL context + buffer, filled
/// by streaming from the shared WeightIndex's GGUF. Idempotent per card. The index must exist.
fn install_resident(device_id: usize) -> Result<(), String> {
    loop {
        let mut ins = INSTALLING.lock().unwrap();
        if miner_for(device_id).is_some() {
            return Ok(());
        }
        if !ins.contains(&device_id) {
            ins.push(device_id); // claimed — we stream this card
            break;
        }
        drop(ins); // another thread is streaming this card; wait for it to finish
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    let result = (|| {
        let (index, _) = crate::pom::active_index().ok_or("PoM: no index")?;
        let n = index.n_chunks;
        let dev = opencl3::device::Device::new(device_id as opencl3::types::cl_device_id);
        let miner = PomMiner::new(dev, index, n)?;
        POM_BY_DEV.lock().unwrap().push((device_id, Arc::new(Mutex::new(miner))));
        log::info!("PoM: tier resident on GPU {device_id:#x} ({} MiB).", (n * 32) / (1024 * 1024));
        Ok(())
    })();
    INSTALLING.lock().unwrap().retain(|d| *d != device_id);
    result
}

/// True if THIS thread's bound card has the tier resident (or, with no binding, any card does).
pub fn is_installed() -> bool {
    match bound_dev() {
        Some(id) => miner_for(id).is_some(),
        None => !POM_BY_DEV.lock().unwrap().is_empty(),
    }
}

/// GPU 0 total global memory in MB via OpenCL (CL_DEVICE_GLOBAL_MEM_SIZE) — the AMD analog of
/// nvidia-smi's `memory.total`, so the model VRAM capability gate (filter_specs_by_vram) works on
/// AMD too. None if there is no OpenCL GPU.
pub fn gpu0_global_mem_mb() -> Option<u64> {
    let dev_ids = opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU).ok()?;
    let id = *dev_ids.first()?;
    let bytes = opencl3::device::Device::new(id).global_mem_size().ok()?;
    Some(bytes / (1024 * 1024))
}

/// Registered mining tier (GGUF path, tier index). Set once at startup via set_mining_tier;
/// the first PoM-active job lazily builds the index + GPU residency via ensure_installed.
static TIER: Mutex<Option<(String, u8)>> = Mutex::new(None);

/// Register the tier to mine (GGUF path on disk, POM_TIERS index). Cheap — no I/O. The heavy
/// load_tier (build the Merkle tree + upload the blob) runs lazily on the first PoM-active job.
pub fn set_mining_tier(gguf_path: String, tier: u8) {
    *TIER.lock().unwrap() = Some((gguf_path, tier));
}

/// Serializes the one-time tier build so concurrent first-time callers (the PoM loop runs one
/// mining thread PER GPU) don't both run build_from_gguf and collide on the shared on-disk Merkle
/// scratch ("failed to fill whole buffer"). The loser waits, then sees is_installed() and returns.
/// Single-GPU AMD boxes never trip this; multi-GPU rigs do. Mirrors pom_gpu's BUILD_LOCK.
static BUILD_LOCK: Mutex<()> = Mutex::new(());

/// Build the shared proof-side WeightIndex from a GGUF, once. Idempotent — returns immediately
/// if it already exists. Caller must hold BUILD_LOCK. This is the heavy, card-independent work
/// (CPU/disk); the per-card VRAM upload streams from this index in install_resident().
fn ensure_index(gguf_path: &str, tier: u8) -> Result<(), String> {
    if crate::pom::active_index().is_some() {
        return Ok(());
    }
    log::info!("PoM: building WeightIndex from {gguf_path} (tier {tier})…");
    let index = crate::pom::WeightIndex::build_from_gguf(gguf_path).map_err(|e| e.to_string())?;
    log::info!(
        "PoM: tier {tier} loaded — {} chunks, computed R_T = {} (must match the node's pinned root)",
        index.n_chunks,
        hex32(&index.r_t)
    );
    crate::pom::set_index(index, tier);
    Ok(())
}

/// Lazily build the shared tier and make it resident on THIS thread's bound GPU, on the first
/// PoM-active iteration. Safe to call concurrently from every GPU mining thread: the shared index
/// is built once (first thread under BUILD_LOCK), then each thread streams the GGUF to its OWN card,
/// so all cards end up resident and mine independently.
/// (Unlike NVIDIA, the AMD OpenCL buffer is never evicted, so each card does this work once.)
pub fn ensure_installed() {
    if is_installed() {
        return; // this thread's card is already resident
    }
    let tier = TIER.lock().unwrap().clone();
    let (path, t) = match tier {
        Some(pt) => pt,
        None => {
            log::warn!("PoM: no mining tier registered (set_mining_tier not called).");
            return;
        }
    };
    {
        // BUILD_LOCK covers ONLY the one-time shared index build. The per-card VRAM streams run
        // OUTSIDE it, concurrently across cards: holding the lock through the upload serialized
        // the streams, so a multi-card rig's hashrate ramped up one card at a time (minutes on
        // big rigs) and the pool's vardiff chasing that ramp caused low-diff reject bursts at
        // startup (issue #9). Concurrent streams cost ~one disk pass total — the page cache
        // serves the followers.
        let _build = BUILD_LOCK.lock().unwrap();
        if let Err(e) = ensure_index(&path, t) {
            log::warn!("PoM: tier build failed ({path}): {e} — is the model GGUF downloaded?");
            return;
        }
    }
    match target_dev() {
        Some(id) => match install_resident(id) {
            Ok(()) => log::info!("PoM: tier {t} installed (GPU-resident) on card {id:#x}."),
            Err(e) => log::warn!("PoM: install on card {id:#x} failed: {e}"),
        },
        None => log::warn!("PoM: no OpenCL GPU device for this thread."),
    }
}

/// Grind one batch of `batch` nonces from `nonce_base` on THIS thread's bound GPU. Returns the
/// lowest nonce whose pom_pow_value <= target, or None. pph/target are the 32-byte LE forms.
/// Per-card lock → the other GPUs' threads grind concurrently.
pub fn mine(pph: &[u8; 32], time: u64, target_le: &[u8; 32], nonce_base: u64, batch: u64, h3: bool) -> Option<u64> {
    let id = target_dev()?;
    let miner = miner_for(id)?;
    // H3: salt the pph words host-side (POM_H3_PPH_SALT) — the OpenCL kernel folds whatever words
    // it receives, so no .cl change at the gate. Byte-identical to the node's pph_words_h3.
    let p = crate::pom::pph_words_for_era(pph, h3);
    let t = words(target_le);
    let mut g = miner.lock().unwrap();
    g.mine(p, time, t, nonce_base, batch)
}

/// Build the resident tier from a GGUF (shared proof WeightIndex, streamed to VRAM) and make it
/// resident on the FIRST OpenCL GPU. The multi-GPU production path uses ensure_installed (one
/// resident copy per card); this single-device form backs the tests + any non-bound caller.
pub fn load_tier(gguf_path: &str, tier: u8) -> Result<(), String> {
    ensure_index(gguf_path, tier)?;
    let id = opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
        .map_err(|e| e.to_string())?
        .first()
        .map(|d| *d as usize)
        .ok_or("PoM: no OpenCL GPU device")?;
    install_resident(id)
}

fn hex32(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for x in b { s.push_str(&format!("{x:02x}")); }
    s
}
