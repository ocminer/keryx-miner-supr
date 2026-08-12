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
    /// H6 PoM v3 matrix-state walk kernel (one work-group per nonce). Grinds the int8 D×D×K matmul.
    kernel_v3: Kernel,
    weights: Vec<Buffer<cl_ulong>>,
    slab_shift: u32,
    winner: Buffer<cl_ulong>,
    pub n_chunks: u64,
    /// Nonces per work-item of the chosen kernel (1 = pom_mine, 2 = _ilp2, 4 = _ilp4). Autotuned.
    ilp: u64,
    /// Chosen work-group size (autotuned). Global size is rounded up to a multiple of this.
    local: usize,
}

// OpenCL handles are plain cl_* pointers usable from any single thread; the global Mutex
// serializes all access (one mining thread), so sending the miner across threads is sound.
unsafe impl Send for PomMiner {}

/// One raw kernel dispatch: `n_nonces` nonces on `kernel` with `ilp` nonces/work-item and the given
/// work-group size, blocking until complete. `None` = OpenCL error; `Some(None)` = ran clean with no
/// winner; `Some(Some(n))` = winning nonce. Free function (not a method) so mine() and the autotune
/// sweep can pass disjoint field borrows. The tid guard in every kernel makes the rounded-up global
/// size safe (padding items exit immediately).
#[allow(clippy::too_many_arguments)]
fn dispatch_raw(queue: &CommandQueue, kernel: &Kernel, weights: &[Buffer<cl_ulong>], slab_shift: u32, winner: &mut Buffer<cl_ulong>, n_chunks: u64,
                ilp: u64, local: usize, pph: [u64; 4], seed: [u64; 4], time: u64, target: [u64; 4],
                base: u64, n_nonces: u64, walk_v2: bool) -> Option<Option<u64>> {
    let wv2: u32 = walk_v2 as u32;
    queue.enqueue_write_buffer(winner, CL_BLOCKING, 0, &[u64::MAX], &[]).ok()?;
    let items = n_nonces.div_ceil(ilp);
    let global = (items.div_ceil(local as u64) * local as u64) as usize;
    // 4 slab args; absent slabs repeat slab 0 (never addressed: off>>shift bounds to real slabs).
    let sl = |i: usize| weights.get(i).unwrap_or(&weights[0]);
    ExecuteKernel::new(kernel)
        .set_arg(sl(0)).set_arg(sl(1)).set_arg(sl(2)).set_arg(sl(3))
        .set_arg(&n_chunks)
        .set_arg(&POM_WALK_STEPS)
        .set_arg(&slab_shift)
        .set_arg(&pph[0]).set_arg(&pph[1]).set_arg(&pph[2]).set_arg(&pph[3])
        .set_arg(&seed[0]).set_arg(&seed[1]).set_arg(&seed[2]).set_arg(&seed[3])
        .set_arg(&time)
        .set_arg(&target[0]).set_arg(&target[1]).set_arg(&target[2]).set_arg(&target[3])
        .set_arg(&base).set_arg(&n_nonces)
        .set_arg(winner)
        .set_arg(&wv2)
        .set_global_work_size(global)
        .set_local_work_size(local)
        .enqueue_nd_range(queue)
        .ok()?;
    queue.finish().ok()?;
    let mut w = [u64::MAX];
    queue.enqueue_read_buffer(winner, CL_BLOCKING, 0, &mut w, &[]).ok()?;
    Some(if w[0] != u64::MAX { Some(w[0]) } else { None })
}

/// H6 PoM v3 dispatch: ONE work-group per nonce, `local`=256 work-items (one per state row x).
/// The whole tier blob is a SINGLE contiguous cl_mem reinterpreted as `uint*` (chunk idx ->
/// blob32[idx*8]); a 64 KB `__local` tile is set via set_arg_local_buffer. `n_nonces` == number
/// of work-groups. Blocks until complete. Same return convention as dispatch_raw.
#[allow(clippy::too_many_arguments)]
fn dispatch_raw_v3(queue: &CommandQueue, kernel: &Kernel, blob: &Buffer<cl_ulong>, winner: &mut Buffer<cl_ulong>,
                   n_tiles: u64, k: u32, pph: [u64; 4], seed: [u64; 4], time: u64, target: [u64; 4],
                   base: u64, n_nonces: u64) -> Option<Option<u64>> {
    const V3_LOCAL: usize = 256;                 // one work-item per state row (D=256)
    const V3_TILE_BYTES: usize = 65536;          // 64 KB LDS tile
    queue.enqueue_write_buffer(winner, CL_BLOCKING, 0, &[u64::MAX], &[]).ok()?;
    let global = (n_nonces * V3_LOCAL as u64) as usize;   // n_nonces work-groups × 256 items
    ExecuteKernel::new(kernel)
        .set_arg(blob)
        .set_arg(&n_tiles)
        .set_arg(&k)
        .set_arg(&pph[0]).set_arg(&pph[1]).set_arg(&pph[2]).set_arg(&pph[3])
        .set_arg(&seed[0]).set_arg(&seed[1]).set_arg(&seed[2]).set_arg(&seed[3])
        .set_arg(&time)
        .set_arg(&target[0]).set_arg(&target[1]).set_arg(&target[2]).set_arg(&target[3])
        .set_arg(&base).set_arg(&n_nonces)
        .set_arg(winner)
        .set_arg_local_buffer(V3_TILE_BYTES)
        .set_global_work_size(global)
        .set_local_work_size(V3_LOCAL)
        .enqueue_nd_range(queue)
        .ok()?;
    queue.finish().ok()?;
    let mut w = [u64::MAX];
    queue.enqueue_read_buffer(winner, CL_BLOCKING, 0, &mut w, &[]).ok()?;
    Some(if w[0] != u64::MAX { Some(w[0]) } else { None })
}

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
        // Bake the tier's chunk count into the JIT build: the per-step `state % n` is a 64-bit
        // division with a runtime divisor (a slow ALU library routine, 256×/nonce); with -D POM_NC
        // the compiler strength-reduces it to its exact multiply-high sequence (byte-exact — the
        // compiler's own constant-division transform). Falls back to the arg if the build with the
        // define fails for any reason.
        let opts = format!("-D POM_NC={}UL", n_chunks);
        let program = match Program::create_and_build_from_source(&context, POM_SRC, &opts) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("PoM: JIT with baked chunk-count failed ({e}) — rebuilding with the runtime divisor.");
                Program::create_and_build_from_source(&context, POM_SRC, "")?
            }
        };
        let kernel = Kernel::create(&program, "pom_mine").map_err(|e| e.to_string())?;
        let kernel_ilp2 = Kernel::create(&program, "pom_mine_ilp2").map_err(|e| e.to_string())?;
        let kernel_ilp4 = Kernel::create(&program, "pom_mine_ilp4").map_err(|e| e.to_string())?;
        let kernel_v3 = Kernel::create(&program, "pom_mine_v3").map_err(|e| e.to_string())?;
        // worker.rs pattern: a context ref that outlives the borrow checker (Arc kept in struct).
        let cref = unsafe { Arc::as_ptr(&context).as_ref().unwrap() };
        // The whole tier blob lives in ONE cl_mem. AMD caps a single allocation at
        // CL_DEVICE_MAX_MEM_ALLOC_SIZE; if the blob exceeds it the driver returns a partial/broken
        // buffer and the walk reads garbage → the card hashes but never finds a valid share (the
        // classic Polaris/RX-580 post-H5 failure — the Qwen3-8B blob is ~4.8 GB). main() raises the
        // cap via GPU_SINGLE_ALLOC_PERCENT=100 before OpenCL inits; warn loudly if it's STILL too
        // small (a driver that ignores the env var) so the failure is diagnosable, not silent.
        let blob_bytes = n_chunks.saturating_mul(32);
        if let Ok(max_alloc) = device.max_mem_alloc_size() {
            // Always log the numbers — makes "hashing but no shares" reports diagnosable from the
            // log alone (is the blob within this driver's single-buffer cap or not?).
            log::info!(
                "PoM: tier blob {} MiB; device max single-buffer alloc {} MiB, global mem {} MiB.",
                blob_bytes / (1024 * 1024),
                max_alloc / (1024 * 1024),
                device.global_mem_size().map(|b| b / (1024 * 1024)).unwrap_or(0),
            );
            if blob_bytes > max_alloc {
                log::info!(
                    "PoM: tier blob ({} MiB) exceeds this GPU's reported max single-buffer allocation \
                     ({} MiB) — the blob will be SPLIT across multiple buffers (slab layout).",
                    blob_bytes / (1024 * 1024), max_alloc / (1024 * 1024),
                );
            }
        }
        // Slab layout: single buffer first (fast path, existing fleet unchanged). If ANY create or
        // write in that path fails — Polaris/ORCA rejects >~4 GiB per buffer with
        // CL_MEM_OBJECT_ALLOCATION_FAILURE even though it REPORTS an 8 GiB max-alloc (RX 580 field
        // report), so the reported limit cannot be trusted — retry with 2^SLAB_SHIFT_FALLBACK-chunk
        // (2 GiB) slabs, up to 4 (kernel arg count). KERYX_POM_CL_SLAB_SHIFT forces a slab size
        // (testing / stubborn drivers). Byte-exact either way: the kernel's pom_fetch maps canonical
        // chunk index -> (slab, intra) and the chunk bytes are identical.
        const SLAB_SHIFT_FALLBACK: u32 = 26; // 2^26 chunks * 32 B = 2 GiB per slab
        let forced_shift = std::env::var("KERYX_POM_CL_SLAB_SHIFT").ok().and_then(|v| v.parse::<u32>().ok());
        let attempts: Vec<u32> = match forced_shift {
            Some(sh) => vec![sh.clamp(20, 63)],
            None => vec![63, SLAB_SHIFT_FALLBACK],
        };
        let mut built: Option<(Vec<Buffer<cl_ulong>>, u32)> = None;
        let mut last_err = String::new();
        for shift in attempts {
            match Self::build_slabs(cref, &queue, index, n_chunks, shift) {
                Ok(v) => {
                    if shift < 63 {
                        log::info!(
                            "PoM: blob resident as {} slab(s) of {} MiB (slab_shift {shift}).",
                            v.len(), (1u64 << shift) * 32 / (1024 * 1024),
                        );
                    }
                    built = Some((v, shift));
                    break;
                }
                Err(e) => {
                    last_err = e;
                    log::warn!(
                        "PoM: blob layout with slab_shift {shift} failed ({last_err}) — {}",
                        if shift == 63 { "retrying with 2 GiB slabs (Polaris per-buffer limit workaround)." } else { "no smaller layout available." },
                    );
                }
            }
        }
        let Some((weights, slab_shift)) = built else {
            return Err(format!("blob allocation failed in every layout: {last_err}"));
        };
        let mut winner = Buffer::<cl_ulong>::create(cref, CL_MEM_READ_WRITE, 1, ptr::null_mut())
            .map_err(|e| e.to_string())?;

        // ---- per-device launch autotune (mirror of pom_gpu::autotune_block for CUDA) ----------
        // Time each (kernel-ILP × work-group size) over the resident blob with an impossible
        // target and pin the fastest. Block size + ILP affect only scheduling/occupancy, never a
        // nonce's math — byte-exact by construction; the walk is latency-bound, and how many
        // independent walks a lane carries decides how much DRAM latency it can hide (our MI50/
        // MI60 runs at ~17% of HBM2 bandwidth at ILP1 = idle miss slots). Env pins for ops:
        // KERYX_POM_CL_ILP (1|2|4) and KERYX_POM_CL_LOCAL skip the sweep.
        let pin_ilp = std::env::var("KERYX_POM_CL_ILP").ok().and_then(|s| s.parse::<u64>().ok());
        let pin_local = std::env::var("KERYX_POM_CL_LOCAL").ok().and_then(|s| s.parse::<usize>().ok());
        let variants: [(u64, &str, &Kernel); 3] =
            [(1, "pom_mine", &kernel), (2, "pom_mine_ilp2", &kernel_ilp2), (4, "pom_mine_ilp4", &kernel_ilp4)];
        let locals: &[usize] = &[64, 128, 256];
        // 2^18 per timed run: the top configs sit ~1% apart, so the sample must be big enough to
        // rank them above run-to-run noise. Full sweep ≈ 9 configs × 3 runs ≈ 0.7 s per card, once
        // per tier load.
        let tune_nonces: u64 = 1 << 18;
        let (mut best_name, mut best_ilp, mut best_local, mut best_rate) = ("pom_mine", 1u64, 256usize, 0f64);
        for (ilp, name, k) in variants {
            if pin_ilp.is_some_and(|p| p != ilp) { continue; }
            for &local in locals {
                if pin_local.is_some_and(|p| p != local) { continue; }
                // 1 warmup + 2 timed, best-of (steadier on a busy card).
                let mut rate = 0f64;
                let mut ok = true;
                for round in 0..3 {
                    let t0 = std::time::Instant::now();
                    if dispatch_raw(&queue, k, &weights, slab_shift, &mut winner, n_chunks, ilp, local,
                                    [1, 2, 3, 4], [1, 2, 3, 4], 1, [0; 4], 0, tune_nonces, true).is_none() {
                        ok = false;
                        break;
                    }
                    if round > 0 {
                        rate = rate.max(tune_nonces as f64 / t0.elapsed().as_secs_f64());
                    }
                }
                if ok && rate > best_rate {
                    best_rate = rate;
                    best_ilp = ilp;
                    best_local = local;
                    best_name = name;
                }
            }
        }
        let chosen = match best_name {
            "pom_mine_ilp2" => kernel_ilp2,
            "pom_mine_ilp4" => kernel_ilp4,
            _ => kernel,
        };
        log::info!(
            "PoM: OpenCL walk autotune -> {} (ILP x{}) @ work-group {} ({:.2} MH/s in-tune){}",
            best_name, best_ilp, best_local, best_rate / 1e6,
            if pin_ilp.is_some() || pin_local.is_some() { " [env-pinned]" } else { "" },
        );
        Ok(Self { _context: context, queue, kernel: chosen, kernel_v3, weights, slab_shift, winner, n_chunks, ilp: best_ilp, local: best_local })
    }

    /// Create + stream the blob as ceil(n_chunks / 2^shift) buffers (shift>=63 = one buffer).
    /// Fails cleanly on any create/write error so the caller can retry a smaller layout.
    fn build_slabs(cref: &Context, queue: &CommandQueue, index: &crate::pom::WeightIndex, n_chunks: u64, shift: u32) -> Result<Vec<Buffer<cl_ulong>>, String> {
        let slab_chunks: u64 = if shift >= 63 { n_chunks } else { 1u64 << shift };
        let n_slabs = n_chunks.div_ceil(slab_chunks.max(1));
        if n_slabs > 4 {
            return Err(format!("{n_slabs} slabs needed but the kernel takes at most 4 — tier too big for this layout"));
        }
        let mut slabs: Vec<Buffer<cl_ulong>> = Vec::with_capacity(n_slabs as usize);
        let mut staging = vec![0u64; (UPLOAD_WINDOW_CHUNKS.min(slab_chunks) * 4) as usize];
        for si in 0..n_slabs {
            let start = si * slab_chunks;
            let count = slab_chunks.min(n_chunks - start);
            let mut buf = Buffer::<cl_ulong>::create(cref, CL_MEM_READ_ONLY, (count * 4) as usize, ptr::null_mut())
                .map_err(|e| format!("slab {si} create ({} MiB): {e}", count * 32 / (1024 * 1024)))?;
            let mut done: u64 = 0;
            while done < count {
                let n = (count - done).min(UPLOAD_WINDOW_CHUNKS);
                let words = &mut staging[..(n * 4) as usize];
                // Raw chunk bytes -> LE word values (no-op on LE targets) — identical to
                // chunk_to_words / u64::from_le_bytes on every chunk.
                let bytes = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 8) };
                index.read_chunks_into(start + done, bytes);
                for w in words.iter_mut() {
                    *w = u64::from_le(*w);
                }
                queue
                    .enqueue_write_buffer(&mut buf, CL_BLOCKING, (done * 32) as usize, words, &[])
                    .map_err(|e| format!("slab {si} write @{} MiB: {e}", done * 32 / (1024 * 1024)))?;
                done += n;
            }
            slabs.push(buf);
        }
        Ok(slabs)
    }

    /// Grind one batch of `batch` nonces from `nonce_base` in TDR-safe sub-dispatches. Returns
    /// the lowest nonce whose pom_pow_value <= target, or None.
    pub fn mine(&mut self, pph: [u64; 4], seed: [u64; 4], time: u64, target: [u64; 4], nonce_base: u64, batch: u64, walk_v2: bool) -> Option<u64> {
        let mut done: u64 = 0;
        while done < batch {
            let sub = (batch - done).min(SUB_DISPATCH_NONCES);
            let base = nonce_base.wrapping_add(done);
            // Disjoint field borrows: kernel/queue/weights immutably, winner mutably.
            match dispatch_raw(&self.queue, &self.kernel, &self.weights, self.slab_shift, &mut self.winner, self.n_chunks,
                               self.ilp, self.local, pph, seed, time, target, base, sub, walk_v2) {
                None => return None,             // OpenCL error — abort the batch (old ok()? behavior)
                Some(Some(w)) => return Some(w), // lowest winner in this ascending sub-batch
                Some(None) => {}                 // clean, no winner — next sub-batch
            }
            done += sub;
        }
        None
    }

    /// H6 (PoM v3): grind `batch` nonces from `nonce_base` through the int8 matrix-state walk on
    /// the resident blob. One work-group per nonce; sub-dispatched to keep any single NDRange short
    /// (each v3 nonce is a 256×256×256 int8 GEMM × 256 steps — heavy). Returns the lowest winning
    /// nonce (host re-walks it byte-exact to build the witness), or None. Requires a single-slab
    /// blob (n_tiles addresses one contiguous buffer); big-VRAM RDNA cards (the H6 target) always
    /// stream one slab.
    pub fn mine_v3(&mut self, pph: [u64; 4], seed: [u64; 4], time: u64, target: [u64; 4], nonce_base: u64, batch: u64) -> Option<u64> {
        if self.weights.len() != 1 {
            log::error!("PoM v3: blob is {} slabs — the v3 kernel needs one contiguous buffer; this card can't do the H6 walk.", self.weights.len());
            return None;
        }
        let n_tiles = self.n_chunks / POM_V3_TILE_CHUNKS;
        if n_tiles == 0 {
            log::error!("PoM v3: blob too small ({} chunks < one 2048-chunk tile).", self.n_chunks);
            return None;
        }
        let k = POM_V3_K as u32;
        let mut done: u64 = 0;
        while done < batch {
            let sub = (batch - done).min(V3_SUB_DISPATCH_NONCES);
            let base = nonce_base.wrapping_add(done);
            match dispatch_raw_v3(&self.queue, &self.kernel_v3, &self.weights[0], &mut self.winner,
                                  n_tiles, k, pph, seed, time, target, base, sub) {
                None => return None,
                Some(Some(w)) => return Some(w),
                Some(None) => {}
            }
            done += sub;
        }
        None
    }
}

/// PoM v3 walk constants (mirror src/pom_v3.rs — the byte-exact host reference).
const POM_V3_TILE_CHUNKS: u64 = 2048;   // 64 KB tile / 32 B chunk
const POM_V3_K: usize = 256;            // walk steps
/// Work-groups (= nonces) per v3 NDRange. Each nonce is a heavy int8 GEMM chain; keep a dispatch
/// short so a slow card doesn't trip the Windows TDR (~2 s). 128 groups ≈ sub-second even at kh/s.
const V3_SUB_DISPATCH_NONCES: u64 = 128;

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

/// The cl_device_id whose card hosts the in-process llama engine (zero-dup): that card gets NO
/// OpenCL blob — its mine() routes over the engine's resident weights instead. Claimed once by
/// the card whose PCI location matches the engine's, after the startup byte gate passes.
static SHARED_DEV: Mutex<Option<usize>> = Mutex::new(None);
/// Card DEDICATED to OPoI inference (engine-hosted model; model + blob can't share its VRAM):
/// it deliberately mines nothing — no blob install, no walk. The other cards mine at full rate.
static DEDICATED_DEV: Mutex<Option<usize>> = Mutex::new(None);

fn is_shared_dev(device_id: usize) -> bool {
    SHARED_DEV.lock().unwrap().map_or(false, |d| d == device_id)
}

/// Set once the miner reaches the H6 (PoM v3) gate. In v3 the walk is the OpenCL `pom_mine_v3`
/// kernel over the card's OWN resident blob — the zero-dup Vulkan walk shader is the pre-H6 v2
/// walk, so a v3 card must NOT claim zero-dup (it would route to the wrong shader and never install
/// the OpenCL blob mine_v3 needs). On the 24 GB H6 target the engine (for OPoI inference) and the
/// blob (for the walk) coexist. Cleared/left false pre-fork.
static V3_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Called by the miner when it enters the PoM v3 (H6) era so the AMD driver keeps each card on its
/// OpenCL blob (mine_v3) instead of the zero-dup Vulkan v2 walk. Idempotent.
pub fn set_v3_mode(on: bool) {
    V3_MODE.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn v3_mode() -> bool {
    V3_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

/// AMD sysfs health sample (the NVML-free analog). Plain data — miner.rs maps it into the shared
/// gpu_health::GpuHealth (that module is binary-side; pom_opencl is lib-side, so no direct ref).
/// Cfg-unconditional: both the unix `amd_health` and the non-unix stub return `Option<AmdHealth>`.
#[derive(Default, Clone)]
pub struct AmdHealth {
    pub power_w: Option<f64>,
    pub temp_c: Option<u32>,
    pub fan_pct: Option<u32>,
    pub core_mhz: Option<u32>,
    pub vram_used_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
}

/// Per-GPU health for the efficiency stats on AMD (the NVML analog): power/temp/clocks/VRAM read
/// from Linux sysfs (amdgpu hwmon), matched to OpenCL device `idx` by PCI address. `idx` is the
/// plugin's "#N" — the Nth GPU device of the AMD ("Advanced Micro Devices, Inc.") OpenCL platform,
/// mirroring plugins/opencl device selection so #N here == #N in the hashrate line. No rocm-smi /
/// NVML dependency; every field is best-effort → None (so a rig without amdgpu sysfs just shows no
/// efficiency, never a wrong number). Power is what the MH/s/W figure needs.
#[cfg(unix)]
pub fn amd_health(idx: usize) -> Option<AmdHealth> {
    use opencl3::platform::get_platforms;
    // Mirror the plugin's platform pick: first AMD-vendor platform with devices.
    let plats = get_platforms().ok()?;
    let amd = plats.into_iter().find(|p| {
        p.vendor().map(|v| v == "Advanced Micro Devices, Inc.").unwrap_or(false)
            && !p.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU).unwrap_or_default().is_empty()
    })?;
    let devs = amd.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU).ok()?;
    let dev = *devs.get(idx)?;
    let (b, d, f) = device_pci(dev as usize)?;
    let want = format!("0000:{:02x}:{:02x}.{:x}", b, d, f);
    // Find the DRM card whose PCI address matches, then read its hwmon + amdgpu sysfs nodes.
    let read_u64 = |path: &std::path::Path| -> Option<u64> {
        std::fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
    };
    for entry in std::fs::read_dir("/sys/class/drm").ok()? {
        let card = entry.ok()?.path();
        if !card.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with("card") && !n.contains('-')).unwrap_or(false) {
            continue;
        }
        let devdir = card.join("device");
        let pci = std::fs::canonicalize(&devdir).ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        if pci.as_deref() != Some(want.as_str()) {
            continue;
        }
        let mut h = AmdHealth::default();
        // hwmon dir under the device (power/temp/fan)
        if let Ok(hw) = std::fs::read_dir(devdir.join("hwmon")) {
            if let Some(hwmon) = hw.filter_map(|e| e.ok()).map(|e| e.path()).next() {
                // RDNA reports power1_average; Vega20/Instinct (MI50/MI60) report power1_input instead.
                h.power_w = read_u64(&hwmon.join("power1_average"))
                    .or_else(|| read_u64(&hwmon.join("power1_input")))
                    .map(|uw| uw as f64 / 1_000_000.0);
                h.temp_c = read_u64(&hwmon.join("temp1_input")).map(|mc| (mc / 1000) as u32);
                if let (Some(cur), Some(max)) = (read_u64(&hwmon.join("pwm1")), Some(255u64)) {
                    h.fan_pct = Some(((cur * 100) / max) as u32);
                }
            }
        }
        h.vram_used_mb = read_u64(&devdir.join("mem_info_vram_used")).map(|b| b / (1024 * 1024));
        h.vram_total_mb = read_u64(&devdir.join("mem_info_vram_total")).map(|b| b / (1024 * 1024));
        // Current core clock: the "*"-marked line in pp_dpm_sclk (e.g. "3: 2482Mhz *").
        if let Ok(txt) = std::fs::read_to_string(devdir.join("pp_dpm_sclk")) {
            for line in txt.lines().filter(|l| l.contains('*')) {
                if let Some(mhz) = line.split_whitespace().find_map(|t| t.strip_suffix("Mhz").and_then(|n| n.parse::<u32>().ok())) {
                    h.core_mhz = Some(mhz);
                }
            }
        }
        return Some(h);
    }
    None
}
#[cfg(not(unix))]
pub fn amd_health(_idx: usize) -> Option<AmdHealth> { None }

/// This card's PCI (bus, device, function) via CL_DEVICE_TOPOLOGY_AMD — matched against the
/// engine's VK_EXT_pci_bus_info to identify the SAME physical card across the two APIs.
#[cfg(unix)]
fn device_pci(device_id: usize) -> Option<(u32, u32, u32)> {
    const CL_DEVICE_TOPOLOGY_AMD: opencl3::types::cl_device_info = 0x4037;
    const CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD: u32 = 1;
    let v = opencl3::device::get_device_data(device_id as opencl3::types::cl_device_id, CL_DEVICE_TOPOLOGY_AMD).ok()?;
    if v.len() < 24 || u32::from_le_bytes(v[0..4].try_into().ok()?) != CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD {
        return None;
    }
    Some((v[21] as u32, v[22] as u32, v[23] as u32))
}

/// True if `device_id` is a zero-dup-SAFE RDNA card (RDNA2+: gfx103x / gfx11 / gfx12) — the
/// Vulkan zero-dup walk matches or beats the OpenCL blob walk there (measured +1.7% on gfx1102).
/// EXCLUDED:
/// - GCN (gfx9 and older): the Vulkan walk is ~15-24% slower at equal clocks.
/// - RDNA1 (gfx101x, RX 5600/5700 series): FIELD-VERIFIED GPU HANG — the byte-gate fetch dispatch
///   (buffer-device-address gather on RADV/NAVI10) hard-hangs the GPU at the first dispatch
///   ("byte gate fetch failed at chunk 0" is the hang, not a soft failure); every later GPU call
///   on that device blocks forever, and unloading the engine (vkDeviceWaitIdle) wedges the whole
///   rig until reboot. Never dispatch the gate there — the policy check runs BEFORE the gate.
/// KERYX_ZERO_DUP=force still overrides for experiments. gfx name via CL_DEVICE_NAME.
#[cfg(unix)]
fn device_is_rdna(device_id: usize) -> bool {
    let name = opencl3::device::Device::new(device_id as opencl3::types::cl_device_id)
        .name()
        .unwrap_or_default();
    if name.contains("gfx101") {
        return false; // RDNA1: BDA fetch hangs the GPU (RX 5700 XT field logs)
    }
    name.contains("gfx10") || name.contains("gfx11") || name.contains("gfx12")
}

/// Zero-dup default policy: claim only when it never costs hashrate. `KERYX_ZERO_DUP` = `force`
/// (claim any PCI-matched card, VRAM over hashrate) / `off` (never) / unset = RDNA-only (default).
#[cfg(unix)]
fn zero_dup_allowed(device_id: usize) -> bool {
    match std::env::var("KERYX_ZERO_DUP").ok().as_deref() {
        Some("force") => true,
        Some("off") => false,
        _ => device_is_rdna(device_id),
    }
}

/// Zero-dup claim: if the in-process llama engine hosts the model on THIS card (PCI match), the
/// policy allows it, and its gather passes the byte gate against the possession index, this card
/// walks the engine's resident weights — no OpenCL blob is uploaded for it. Returns whether the
/// claim succeeded (false = the card keeps its own OpenCL blob).
/// True if this card's total VRAM comfortably holds the resident engine model PLUS the blob
/// (gguf file size ≈ model VRAM + 1 GiB margin for KV/ctx/driver). When it does NOT, keeping the
/// engine while installing the blob makes RADV silently OVERCOMMIT into GTT (the create/write
/// SUCCEED — nothing fails) and the walk crawls at PCIe latency: the autotune sweep alone can run
/// for an hour, so the card looks permanently dead at 0 hash (RX 5700 XT 8 GB field log). So the
/// engine must be unloaded BEFORE the install on such cards, not on install *failure*.
#[cfg(unix)]
fn model_plus_blob_fits(device_id: usize, blob_bytes: u64) -> bool {
    let model_bytes = TIER
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|(path, _)| std::fs::metadata(path).ok().map(|m| m.len()))
        .unwrap_or(6 << 30); // unknown → assume a big model (conservative: prefer unloading)
    let Ok(total) = opencl3::device::Device::new(device_id as opencl3::types::cl_device_id).global_mem_size() else {
        return false;
    };
    total >= blob_bytes + model_bytes + (1 << 30)
}

#[cfg(unix)]
fn try_claim_shared(device_id: usize) -> bool {
    if is_shared_dev(device_id) {
        return true;
    }
    // H6 (PoM v3): the walk is the OpenCL pom_mine_v3 kernel over this card's own blob. The zero-dup
    // Vulkan walk is the pre-H6 v2 shader, so never claim zero-dup in v3 — install the OpenCL blob
    // (the engine can still host the model on this card for OPoI inference; both fit on 24 GB).
    if v3_mode() {
        return false;
    }
    if SHARED_DEV.lock().unwrap().is_some() || !crate::llama_engine_vk::pom_ready() {
        return false;
    }
    let (Some((_, eb, ed, ef)), Some((b, d, f))) = (crate::llama_engine_vk::pom_pci(), device_pci(device_id)) else {
        return false;
    };
    if (eb, ed, ef) != (b, d, f) {
        return false; // the engine hosts the model on a different card
    }
    if !zero_dup_allowed(device_id) {
        let blob = crate::pom::active_index().map(|(i, _)| i.n_chunks.saturating_mul(32)).unwrap_or(0);
        if blob > 0 && !model_plus_blob_fits(device_id, blob) {
            // Model + blob can't share this card. DEDICATE it to OPoI inference (8x-reward path,
            // fast GPU answers) instead of unloading: no blob is installed here, the card mines
            // nothing, every other card mines at full rate. (Operator call: CPU inference is too
            // slow to be worth trading for one card's hashrate.) KERYX_ZERO_DUP=force overrides
            // on zero-dup-capable archs.
            *DEDICATED_DEV.lock().unwrap() = Some(device_id);
            log::info!(
                "PoM: card {device_id:#x} DEDICATED to OPoI inference (hosts the model; model + blob \
                 exceed its VRAM so it does not mine). Other cards mine at full rate."
            );
        } else {
            log::info!(
                "PoM: card {device_id:#x} hosts the llama engine but is not RDNA — keeping its OpenCL blob \
                 (full hashrate; set KERYX_ZERO_DUP=force to trade ~2.4 GB VRAM for it)."
            );
        }
        return false;
    }
    let Some((idx, _)) = crate::pom::active_index() else { return false };
    if !crate::llama_engine_vk::pom_byte_gate(idx) {
        // Zero-dup is OFF for this card, but the engine still holds the model on it. If model +
        // blob can't BOTH fit, unload NOW: with RADV overcommit the upcoming blob install would
        // not fail — it would silently spill to GTT and the card would crawl at ~0 hash forever.
        if !model_plus_blob_fits(device_id, idx.n_chunks.saturating_mul(32)) {
            log::warn!(
                "PoM: byte gate failed on card {device_id:#x} and model + blob exceed its VRAM —                  unloading the in-process engine so the blob gets real VRAM (OPoI inference falls                  back to CPU; mining beats inference)."
            );
            crate::llama_engine_vk::unload();
        }
        return false;
    }
    *SHARED_DEV.lock().unwrap() = Some(device_id);
    log::info!(
        "PoM: card {device_id:#x} walks the llama-engine-resident weights (zero-dup — no OpenCL blob for this card)."
    );
    true
}
#[cfg(not(unix))]
fn try_claim_shared(_device_id: usize) -> bool {
    false
}

/// True if the in-process llama engine holds its model on this OpenCL card (PCI match).
#[cfg(unix)]
fn engine_hosts_card(device_id: usize) -> bool {
    match (crate::llama_engine_vk::pom_pci(), device_pci(device_id)) {
        (Some((_, eb, ed, ef)), Some((b, d, f))) => (eb, ed, ef) == (b, d, f),
        _ => false,
    }
}
#[cfg(not(unix))]
fn engine_hosts_card(_device_id: usize) -> bool {
    false
}

/// Cards whose streaming upload is currently in flight. Uploads for DIFFERENT cards run
/// concurrently (each thread streams its own card); a second caller for the SAME card waits
/// here instead of double-building (only the unbound/test fallback path can race like that —
/// production mining threads each bind a distinct device).
static INSTALLING: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Make the resident tier GPU-resident on `device_id` — its own OpenCL context + buffer, filled
/// by streaming from the shared WeightIndex's GGUF. Idempotent per card. The index must exist.
fn install_resident(device_id: usize) -> Result<(), String> {
    let wait_start = std::time::Instant::now();
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
        // A streamer that died WITHOUT unwinding (or is wedged inside a driver call) used to
        // leave its id in INSTALLING forever — every later attempt for that card spun here
        // SILENTLY and the card idled at 0 hash with no log at all (RX 5700 XT field log:
        // one byte-gate line, then nothing ever again). Bounded wait + takeover instead.
        if wait_start.elapsed().as_secs() > 600 {
            log::warn!(
                "PoM: card {device_id:#x} waited >10 min for another install of the same card — \
                 assuming that installer is dead/wedged and taking over."
            );
            INSTALLING.lock().unwrap().retain(|d| *d != device_id);
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    // Unwind-safe claim release: if anything below panics, the Drop still clears the id so the
    // card can be retried instead of wedging every future attempt in the wait loop above.
    struct InstallClaim(usize);
    impl Drop for InstallClaim {
        fn drop(&mut self) {
            INSTALLING.lock().unwrap().retain(|d| *d != self.0);
        }
    }
    let _claim = InstallClaim(device_id);
    log::info!("PoM: initializing OpenCL context + kernel JIT on card {device_id:#x}…");
    let result = (|| {
        let (index, _) = crate::pom::active_index().ok_or("PoM: no index")?;
        let n = index.n_chunks;
        log::info!(
            "PoM: streaming tier blob ({} MiB) to card {device_id:#x} + autotune — the card shows \
             0 hash until this completes (typically 1-3 min).",
            (n * 32) / (1024 * 1024)
        );
        let dev = opencl3::device::Device::new(device_id as opencl3::types::cl_device_id);
        let miner = PomMiner::new(dev, index, n)?;
        POM_BY_DEV.lock().unwrap().push((device_id, Arc::new(Mutex::new(miner))));
        log::info!("PoM: tier resident on GPU {device_id:#x} ({} MiB).", (n * 32) / (1024 * 1024));
        Ok(())
    })();
    result
}

/// True if THIS thread's bound card has the tier resident — either its own OpenCL blob or the
/// zero-dup engine walk (or, with no binding, any card does).
pub fn is_installed() -> bool {
    // A DEDICATED inference card is deliberately not mining — treat it as resolved so the worker
    // doesn't loop ensure_installed/heartbeat on it (its 0 hash is intentional and logged once).
    if let (Some(id), Some(d)) = (bound_dev(), *DEDICATED_DEV.lock().unwrap()) {
        if id == d {
            return true;
        }
    }
    match bound_dev() {
        Some(id) => miner_for(id).is_some() || is_shared_dev(id),
        None => !POM_BY_DEV.lock().unwrap().is_empty() || SHARED_DEV.lock().unwrap().is_some(),
    }
}

/// LARGEST global memory (MB) across all OpenCL GPUs — the up-front GPU-inference feasibility
/// check: if even the biggest card can't hold model + possession blob together, loading the model
/// onto ANY mining GPU is doomed (spill/unload later), so skip GPU inference from the start.
pub fn max_gpu_global_mem_mb() -> Option<u64> {
    let dev_ids = opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU).ok()?;
    dev_ids
        .iter()
        .filter_map(|id| opencl3::device::Device::new(*id).global_mem_size().ok())
        .max()
        .map(|b| b / (1024 * 1024))
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
    let mut index = crate::pom::WeightIndex::build_from_gguf(gguf_path).map_err(|e| e.to_string())?;
    log::info!(
        "PoM: tier {tier} loaded — {} chunks, computed R_T = {} (must match the node's pinned root)",
        index.n_chunks,
        hex32(&index.r_t)
    );
    // Opt-in solo block-race edge (mirror of pom_gpu's hook, upstream 951b2e6): hold the FULL Merkle
    // tree in RAM so the post-hit proof build is a pure lookup (~0.5 ms) instead of the ~30-40 ms
    // sparse recompute — at 10 BPS that latency measurably loses chain races. Costs ~2N*32 B RAM
    // (~9.6 GB at tier-0), hence opt-in; pool miners don't need it, low-RAM rigs keep the sparse
    // path. Built ONCE on the shared index (before set_index), so all cards share it. Byte-safe:
    // build_dense self-checks its dense root against the pinned R_T.
    if crate::pom::resident_tree_enabled() {
        let t0 = std::time::Instant::now();
        log::info!("PoM: building RESIDENT Merkle tree (RAM) — proof build becomes lookup-time (~+{} MiB)…", (index.n_chunks * 64) / (1024 * 1024));
        index.build_dense();
        log::info!("PoM: resident tree ready in {:.1}s.", t0.elapsed().as_secs_f32());
    }
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
    // Attempt heartbeat: a card that repeatedly fails/loops here must SAY so — a silent 0-hash
    // card is undiagnosable from the field (the whole RX 580 / 5700 XT saga).
    {
        static ATTEMPTS: Mutex<Vec<(usize, u32)>> = Mutex::new(Vec::new());
        if let Some(id) = target_dev() {
            let mut a = ATTEMPTS.lock().unwrap();
            let e = if let Some(e) = a.iter_mut().find(|(d, _)| *d == id) { e } else { a.push((id, 0)); a.last_mut().unwrap() };
            e.1 += 1;
            if e.1 % 30 == 0 {
                log::warn!(
                    "PoM: card {id:#x} still has NO resident tier after {} attempts (installing={:?}, \
                     zero-dup card={:?}) — the install keeps failing or is stuck; send this log line \
                     when reporting.",
                    e.1,
                    INSTALLING.lock().unwrap().clone(),
                    *SHARED_DEV.lock().unwrap(),
                );
            }
        }
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
        Some(id) => {
            // Zero-dup first: when the in-process llama engine hosts the model on this very
            // card (PCI match + byte gate), walk its resident weights — skip the blob upload.
            if try_claim_shared(id) {
                return;
            }
            match install_resident(id) {
                Ok(()) => log::info!("PoM: tier {t} installed (GPU-resident) on card {id:#x}."),
                // If the failing card is the one the in-process engine holds its model on (a
                // failed/refused zero-dup claim, e.g. "byte gate fetch failed"), the resident
                // model (~6 GB) + the blob (~4.8 GB post-H5) may simply not both fit — on 8 GB
                // cards the blob alloc fails and the card would idle at 0 hash forever (the
                // field-reported "GPU 1 idle" bug). Mining beats in-process inference: unload
                // the engine to free its VRAM and retry the install once; OPoI falls back to
                // the llama-server subprocess / CPU.
                Err(e) if engine_hosts_card(id) => {
                    log::warn!(
                        "PoM: install on card {id:#x} failed ({e}) while the in-process llama engine \
                         holds the model on that card — unloading the engine to free VRAM and retrying."
                    );
                    #[cfg(unix)]
                    let unloaded = crate::llama_engine_vk::unload();
                    #[cfg(not(unix))]
                    let unloaded = false;
                    if unloaded {
                        match install_resident(id) {
                            Ok(()) => log::info!("PoM: tier {t} installed (GPU-resident) on card {id:#x} after engine unload."),
                            Err(e2) => log::warn!("PoM: install on card {id:#x} STILL failed after engine unload: {e2}"),
                        }
                    }
                }
                Err(e) => log::warn!("PoM: install on card {id:#x} failed: {e}"),
            }
        }
        None => log::warn!("PoM: no OpenCL GPU device for this thread."),
    }
}

/// Grind one batch of `batch` nonces from `nonce_base` on THIS thread's bound GPU. Returns the
/// lowest nonce whose pom_pow_value <= target, or None. pph/target are the 32-byte LE forms.
/// Per-card lock → the other GPUs' threads grind concurrently.
pub fn mine(pph: &[u8; 32], time: u64, target_le: &[u8; 32], nonce_base: u64, batch: u64, h3: bool, walk_v2: bool, h5_1: bool, h5_2: bool) -> Option<u64> {
    let id = target_dev()?;
    // H3: salt the pph words host-side (POM_H3_PPH_SALT) — both walk backends fold whatever
    // words they receive, so no kernel/shader change at the gate.
    // H5 (walk_v2): the non-foldable mix64-chain transition IS a kernel/shader change — threaded
    // into BOTH backends (the OpenCL blob kernel `pom_mine.cl` and the zero-dup Vulkan walk shader).
    // H5.1/H5.2 (h5_1/h5_2): the SEED fold swaps to that era's salted pph words
    // (`seed_pph_words_for_era` selects h5_2 > h5_1 > h3 > base) while the POW fold keeps the
    // H3-salted words — both backends take TWO word sets (pow `p` + seed `s`). The GPU kernels are
    // era-agnostic on the seed words: a new seed salt is a pure host-side change (like H3), so
    // pom_mine.cl / pom_walk_vk.comp / the vk .so are UNCHANGED — only the words we upload differ.
    let p = crate::pom::pph_words_for_era(pph, h3);
    let s = crate::pom::seed_pph_words_for_era(pph, h3, h5_1, h5_2);
    let t = words(target_le);
    // Zero-dup card: grind over the llama-engine-resident weights (byte-gate-verified).
    if is_shared_dev(id) {
        return crate::llama_engine_vk::pom_mine(p, s, time, t, nonce_base, batch, walk_v2);
    }
    let miner = miner_for(id)?;
    let mut g = miner.lock().unwrap();
    g.mine(p, s, time, t, nonce_base, batch, walk_v2)
}

/// H6 (PoM v3) grind on THIS thread's bound GPU: the int8 matrix-state walk on the resident blob.
/// Mirrors `mine()`'s host-side era word selection (POW words `p` from `pph_words_for_era`, SEED
/// words `s` from `seed_pph_words_for_era`), then dispatches the `pom_mine_v3` kernel. Returns the
/// lowest winning nonce; the caller rebuilds the witness host-side (byte-exact CPU re-walk), so a
/// kernel false-positive is silently dropped, never submitted. v3 always runs on the card's own
/// OpenCL blob (never the zero-dup Vulkan walk — that shader is the pre-H6 v2 walk).
pub fn mine_v3(pph: &[u8; 32], time: u64, target_le: &[u8; 32], nonce_base: u64, batch: u64, h3: bool, h5_1: bool, h5_2: bool) -> Option<u64> {
    let id = target_dev()?;
    let p = crate::pom::pph_words_for_era(pph, h3);
    let s = crate::pom::seed_pph_words_for_era(pph, h3, h5_1, h5_2);
    let t = words(target_le);
    let miner = miner_for(id)?;
    let mut g = miner.lock().unwrap();
    g.mine_v3(p, s, time, t, nonce_base, batch)
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

#[cfg(test)]
mod v3_byte_exact {
    //! Byte-exact GPU-vs-CPU check for the H6 (PoM v3) walk. Builds a synthetic possession blob,
    //! grinds a handful of nonces on a REAL OpenCL GPU via `pom_mine_v3`, and asserts the winner
    //! matches the CPU host reference (`pom_v3::host_walk_via_index` + `pom_pow_value`). Consensus
    //! is byte-exact, so any divergence is a kernel bug. Needs an AMD OpenCL GPU (skips if none).
    use super::*;
    use crate::pom_v3;

    // CPU reference pow-value (256-bit LE) for one nonce over `index`, era flags all false.
    fn cpu_pow(index: &crate::pom::WeightIndex, pph: &[u8; 32], time: u64, nonce: u64) -> [u8; 32] {
        let seed = crate::pom::pom_block_seed(pph, time, nonce, false, false, false);
        let (states, _snips) = pom_v3::host_walk_via_index(seed, index).unwrap();
        let d2 = 256 * 256;
        let sk = &states[256 * d2..257 * d2]; // S_K (final state)
        let root = pom_v3::v3_state_root(sk);
        let fin = pom_v3::fold64(&root);
        crate::pom::pom_pow_value(fin, pph, false)
    }

    #[test]
    fn gpu_v3_matches_cpu_reference() {
        // First AMD OpenCL GPU, or skip (CI without a GPU).
        let dev_ids = match opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU) {
            Ok(v) if !v.is_empty() => v,
            _ => { eprintln!("no OpenCL GPU — skipping v3 byte-exact test"); return; }
        };
        let dev = opencl3::device::Device::new(dev_ids[0]);
        eprintln!("v3 byte-exact on: {}", dev.name().unwrap_or_default());

        // Synthetic 4-tile blob (8192 chunks). Varied signed bytes to exercise the int8 dot.
        let n_chunks: u64 = 4 * POM_V3_TILE_CHUNKS;
        let mut data = vec![0u8; (n_chunks * 32) as usize];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i.wrapping_mul(131).wrapping_add(i / 7).wrapping_add(17)) as u8;
        }
        let index = crate::pom::index_from_ram(data);

        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(23).wrapping_add(5));
        let time: u64 = 0x0123_4567_89AB_CDEF;
        const NN: u64 = 4; // host walk is ~4.3 GMAC/nonce — keep it small (run with --release)

        // CPU: find the argmin pow over 0..NN; target = that min → only the argmin passes.
        let mut best = ([0xFFu8; 32], u64::MAX);
        for nonce in 0..NN {
            let pv = cpu_pow(&index, &pph, time, nonce);
            // LE compare (least-significant word first).
            let le_le = |a: &[u8; 32], b: &[u8; 32]| {
                for k in (0..4).rev() {
                    let (wa, wb) = (
                        u64::from_le_bytes(a[k * 8..k * 8 + 8].try_into().unwrap()),
                        u64::from_le_bytes(b[k * 8..k * 8 + 8].try_into().unwrap()),
                    );
                    if wa != wb { return wa < wb; }
                }
                true
            };
            if best.1 == u64::MAX || le_le(&pv, &best.0) { best = (pv, nonce); }
        }
        let (target, w_cpu) = best;
        eprintln!("cpu argmin nonce = {w_cpu}");

        // GPU: grind 0..NN with target = the argmin's pow. Expect it to return w_cpu.
        let mut miner = PomMiner::new(dev, &index, n_chunks).expect("PomMiner::new");
        let p = crate::pom::pph_words_for_era(&pph, false);
        let s = crate::pom::seed_pph_words_for_era(&pph, false, false, false);
        let t = words(&target);
        let got = miner.mine_v3(p, s, time, t, 0, NN);
        assert_eq!(got, Some(w_cpu), "GPU v3 winner must match the CPU reference argmin");
    }
}
