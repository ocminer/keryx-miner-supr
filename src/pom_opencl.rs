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
use opencl3::types::{cl_uint, cl_ulong, CL_BLOCKING};

pub const POM_WALK_STEPS: u32 = 256;
const POM_SRC: &str = include_str!("../plugins/opencl/resources/pom_mine.cl");

pub struct PomMiner {
    _context: Arc<Context>,
    queue: CommandQueue,
    /// PoM v4 walk kernel (v0.11.0): 256-thread groups of 8 sub-nonces × 32 lanes.
    kernel_v4: Kernel,
    /// Two-phase v4 (v0.11.5 port): phase-1 offset chase + phase-2 pipelined walk.
    kernel_chase: Kernel,
    kernel_v4_tp: Kernel,
    /// Global scratch for phase-1 offsets ([sub_dispatch][K] u32); allocated lazily on first
    /// two-phase mine_v4. None when two-phase is disabled (KERYX_POM_V4_TP=0) or not yet used.
    offsets: Option<Buffer<cl_uint>>,
    /// Two-phase enabled (default; KERYX_POM_V4_TP=0 forces the single-phase kernel).
    use_tp: bool,
    weights: Vec<Buffer<cl_ulong>>,
    /// Chunks per slab (single-slab layout: == n_chunks). Slabs are tile-aligned.
    slab_chunks: u64,
    winner: Buffer<cl_ulong>,
    pub n_chunks: u64,
}

// OpenCL handles are plain cl_* pointers usable from any single thread; the global Mutex
// serializes all access (one mining thread), so sending the miner across threads is sound.
unsafe impl Send for PomMiner {}

/// Enqueue one v4 sub-dispatch (no finish/read): 256-thread workgroups of V4_NPG(8) sub-nonces.
/// `n_nonces` nonces from `base`; groups = ceil(n_nonces/8) (tail sub-nonces walk a dummy nonce and
/// never submit — uniform barriers). The kernel CAS-mins into the shared `winner`; mine_v4 enqueues
/// the whole batch back-to-back then finishes ONCE so the GPU queue never drains mid-batch.
#[allow(clippy::too_many_arguments)]
fn enqueue_v4(queue: &CommandQueue, kernel: &Kernel, weights: &[Buffer<cl_ulong>], slab_tiles: u64,
              winner: &Buffer<cl_ulong>, n_tiles: u64, k: u32, pph: [u64; 4], seed: [u64; 4],
              time: u64, target: [u64; 4], base: u64, n_nonces: u64) -> Option<()> {
    const V4_LOCAL: usize = 256;
    const V4_NPG: u64 = 8;                       // sub-nonces per workgroup (kernel mirror)
    const V4_LDS_BYTES: usize = 8 * 512 * 4;     // 8 strips × 512 u32 = 16 KB
    // 4 slab args; absent slabs repeat slab 0 (never selected: off/slab_tiles bounds to real slabs).
    let sl = |i: usize| weights.get(i).unwrap_or(&weights[0]);
    let groups = n_nonces.div_ceil(V4_NPG);
    let global = (groups * V4_LOCAL as u64) as usize;
    ExecuteKernel::new(kernel)
        .set_arg(sl(0)).set_arg(sl(1)).set_arg(sl(2)).set_arg(sl(3))
        .set_arg(&n_tiles)
        .set_arg(&slab_tiles)
        .set_arg(&k)
        .set_arg(&pph[0]).set_arg(&pph[1]).set_arg(&pph[2]).set_arg(&pph[3])
        .set_arg(&seed[0]).set_arg(&seed[1]).set_arg(&seed[2]).set_arg(&seed[3])
        .set_arg(&time)
        .set_arg(&target[0]).set_arg(&target[1]).set_arg(&target[2]).set_arg(&target[3])
        .set_arg(&base).set_arg(&n_nonces)
        .set_arg(winner)
        .set_arg_local_buffer(V4_LDS_BYTES)
        .set_global_work_size(global)
        .set_local_work_size(V4_LOCAL)
        .enqueue_nd_range(queue)
        .ok()?;
    Some(())
}

/// Two-phase v4 sub-dispatch: phase 1 (chase) resolves all K offsets into `offsets`, then phase 2
/// (pipelined walk) reads them and prefetches tile t+1 during matmul t. Same in-order queue, so the
/// walk sees the chase's writes without an explicit barrier. `offsets` must hold ≥ n_nonces*K u32.
#[allow(clippy::too_many_arguments)]
fn enqueue_v4_tp(queue: &CommandQueue, chase: &Kernel, walk: &Kernel, weights: &[Buffer<cl_ulong>],
                 slab_tiles: u64, offsets: &Buffer<cl_uint>, winner: &Buffer<cl_ulong>, n_tiles: u64,
                 k: u32, pph: [u64; 4], seed: [u64; 4], time: u64, target: [u64; 4], base: u64,
                 n_nonces: u64) -> Option<()> {
    const V4_LOCAL: usize = 256;
    const V4_NPG: u64 = 8;
    const V4_LDS_BYTES: usize = 8 * 512 * 4;
    const CHASE_LOCAL: usize = 64;               // plain 1D latency-bound pointer chase
    let sl = |i: usize| weights.get(i).unwrap_or(&weights[0]);
    // Phase 1: one work-item per nonce, rounded up to CHASE_LOCAL.
    let chase_global = (n_nonces as usize).div_ceil(CHASE_LOCAL) * CHASE_LOCAL;
    ExecuteKernel::new(chase)
        .set_arg(sl(0)).set_arg(sl(1)).set_arg(sl(2)).set_arg(sl(3))
        .set_arg(&n_tiles)
        .set_arg(&slab_tiles)
        .set_arg(&k)
        .set_arg(&seed[0]).set_arg(&seed[1]).set_arg(&seed[2]).set_arg(&seed[3])
        .set_arg(&time)
        .set_arg(&base).set_arg(&n_nonces)
        .set_arg(offsets)
        .set_global_work_size(chase_global)
        .set_local_work_size(CHASE_LOCAL)
        .enqueue_nd_range(queue)
        .ok()?;
    // Phase 2: pipelined walk.
    let groups = n_nonces.div_ceil(V4_NPG);
    let global = (groups * V4_LOCAL as u64) as usize;
    ExecuteKernel::new(walk)
        .set_arg(sl(0)).set_arg(sl(1)).set_arg(sl(2)).set_arg(sl(3))
        .set_arg(&n_tiles)
        .set_arg(&slab_tiles)
        .set_arg(&k)
        .set_arg(&pph[0]).set_arg(&pph[1]).set_arg(&pph[2]).set_arg(&pph[3])
        .set_arg(&seed[0]).set_arg(&seed[1]).set_arg(&seed[2]).set_arg(&seed[3])
        .set_arg(&time)
        .set_arg(&target[0]).set_arg(&target[1]).set_arg(&target[2]).set_arg(&target[3])
        .set_arg(&base).set_arg(&n_nonces)
        .set_arg(offsets)
        .set_arg(winner)
        .set_arg_local_buffer(V4_LDS_BYTES)
        .set_global_work_size(global)
        .set_local_work_size(V4_LOCAL)
        .enqueue_nd_range(queue)
        .ok()?;
    Some(())
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


impl PomMiner {
    /// Build the resident tier on `device` by streaming the canonical chunks straight from the
    /// shared WeightIndex (GGUF pread) into the cl_mem buffer through a bounded staging window.
    pub fn new(device: Device, index: &crate::pom::WeightIndex, n_chunks: u64) -> Result<Self, String> {
        let context = Arc::new(Context::from_device(&device).map_err(|e| e.to_string())?);
        let queue = CommandQueue::create(&context, device.id(), 0).map_err(|e| e.to_string())?;
        // worker.rs pattern: a context ref that outlives the borrow checker (Arc kept in struct).
        let cref = unsafe { Arc::as_ptr(&context).as_ref().unwrap() };
        let dev_name = device.name().unwrap_or_default();

        // ---- blob layout FIRST (the JIT bakes the layout's divisors in below) -----------------
        // AMD caps a single allocation at CL_DEVICE_MAX_MEM_ALLOC_SIZE — and the REPORT is
        // unreliable in both directions: ORCA/RX 580 reports 8 GiB but rejects >~4 GiB, other
        // Polaris drivers report ~25% of VRAM (2012 MiB on an 8 GB card, field report) and mean
        // it. So the layout is chosen by TRYING: 1..=4 slabs, each slab TILE-ALIGNED (a multiple
        // of the 2048-chunk v3 tile so a 64 KB tile never straddles a slab) and sized
        // ceil(n/k) — NOT a power of two, so 4 slabs really can tile e.g. a 6.2 GB blob under a
        // 2012 MiB cap (4 × 1553 MiB), which the old 2-GiB-only fallback could not. Layouts whose
        // slab size exceeds the driver's reported max-alloc are skipped up front (no point
        // issuing a create the driver already told us it will refuse); the create/write result
        // decides the rest. Byte-exact in every layout: chunks keep their canonical index.
        let blob_bytes = n_chunks.saturating_mul(32);
        let reported_max = device.max_mem_alloc_size().unwrap_or(u64::MAX);
        log::info!(
            "PoM: tier blob {} MiB; device max single-buffer alloc {} MiB, global mem {} MiB.",
            blob_bytes / (1024 * 1024),
            reported_max / (1024 * 1024),
            device.global_mem_size().map(|b| b / (1024 * 1024)).unwrap_or(0),
        );
        if blob_bytes > reported_max {
            log::info!(
                "PoM: tier blob ({} MiB) exceeds this GPU's reported max single-buffer allocation \
                 ({} MiB) — the blob will be SPLIT across multiple buffers (slab layout).",
                blob_bytes / (1024 * 1024), reported_max / (1024 * 1024),
            );
        }
        const V3_TILE: u64 = POM_V4_TILE_CHUNKS; // slab alignment quantum (32 chunks = 1 KB tile)
        let align_up = |c: u64| c.div_ceil(V3_TILE) * V3_TILE;
        // KERYX_POM_CL_SLABS forces a slab count; KERYX_POM_CL_SLAB_SHIFT (legacy) forces a
        // 2^shift-chunk slab size, translated to the equivalent count. Unset = adaptive 1..=4.
        let forced_k: Option<u64> = std::env::var("KERYX_POM_CL_SLABS").ok()
            .and_then(|v| v.parse::<u64>().ok()).filter(|&k| (1..=4).contains(&k))
            .or_else(|| {
                std::env::var("KERYX_POM_CL_SLAB_SHIFT").ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .map(|sh| if sh >= 63 { 1 } else { n_chunks.div_ceil(1u64 << sh).clamp(1, 4) })
            });
        let attempts: Vec<u64> = match forced_k {
            Some(k) => vec![k],
            None => vec![1, 2, 3, 4],
        };
        let mut built: Option<(Vec<Buffer<cl_ulong>>, u64)> = None;
        let mut last_err = String::new();
        for k in attempts {
            let slab_chunks = if k == 1 { n_chunks } else { align_up(n_chunks.div_ceil(k)) };
            let slab_bytes = slab_chunks.saturating_mul(32);
            if k > 1 && slab_bytes > reported_max {
                last_err = format!("{k}-slab layout needs {} MiB/slab > reported max-alloc {} MiB",
                                   slab_bytes / (1024 * 1024), reported_max / (1024 * 1024));
                log::info!("PoM: skipping {k}-slab layout ({last_err}).");
                continue;
            }
            match Self::build_slabs(cref, &queue, index, n_chunks, slab_chunks) {
                Ok(v) => {
                    if v.len() > 1 {
                        log::info!(
                            "PoM: blob resident as {} slabs of {} MiB (tile-aligned, {} chunks/slab).",
                            v.len(), slab_bytes / (1024 * 1024), slab_chunks,
                        );
                    }
                    built = Some((v, slab_chunks));
                    break;
                }
                Err(e) => {
                    last_err = e;
                    log::warn!("PoM: {k}-slab layout failed ({last_err}) — trying a smaller slab size.");
                }
            }
        }
        let Some((weights, slab_chunks)) = built else {
            return Err(format!("blob allocation failed in every layout (1-4 slabs): {last_err}"));
        };

        // ---- JIT with the layout + tier baked in ---------------------------------------------
        // POM_NC (chunk count), POM_SLABC (chunks/slab), POM_SLABT (tiles/slab) are runtime-
        // divisor divisions in the hot paths; baking them strength-reduces each to a multiply-high
        // (byte-exact — the compiler's own constant-division transform). Falls back to runtime
        // args if the define build fails. Native int8 dot: RDNA3+/gfx11-12 use `sudot4`
        // (dot9-insts); GCN/CDNA gfx906/908/90a use `sdot4` (dot1-insts); both byte-identical to
        // the scalar unpack; everything else (Polaris/RDNA1-2/Windows Adrenalin) keeps scalar.
        // KERYX_NO_AMD_DOT4 forces scalar; failed builds retry define-less.
        let allow_dot = std::env::var("KERYX_NO_AMD_DOT4").is_err();
        let dot_def = if !allow_dot {
            None
        } else if dev_name.contains("gfx11") || dev_name.contains("gfx12") {
            Some(("sudot4 (RDNA3+ dot9-insts)", "-D USE_AMD_DOT4=1"))
        } else if dev_name.contains("gfx906") || dev_name.contains("gfx908") || dev_name.contains("gfx90a") {
            Some(("sdot4 (GCN/CDNA dot1-insts)", "-D USE_AMD_SDOT4=1"))
        } else {
            None
        };
        // v4 bakes n_tiles (POM_NT) and tiles-per-slab (POM_SLABT) — the two runtime-divisor
        // divisions in the walk's hot paths (offset % n_tiles, off / slab_tiles).
        let base = format!(
            "-D POM_NT={}UL -D POM_SLABT={}UL",
            n_chunks / V3_TILE, slab_chunks / V3_TILE,
        );
        let opts = match dot_def { Some((_, d)) => format!("{base} {d}"), None => base.clone() };
        let program = match Program::create_and_build_from_source(&context, POM_SRC, &opts) {
            Ok(p) => {
                if let Some((desc, _)) = dot_def {
                    log::info!("PoM: v3 int8 matmul using native v_dot4_i32_i8 ({desc}) on {dev_name}.");
                }
                p
            }
            Err(e) => {
                log::warn!("PoM: JIT with {opts:?} failed ({e}) — retrying without the dot4/baked-layout defines.");
                match Program::create_and_build_from_source(&context, POM_SRC, &base) {
                    Ok(p) => p,
                    Err(_) => Program::create_and_build_from_source(&context, POM_SRC, "")?,
                }
            }
        };
        let kernel_v4 = Kernel::create(&program, "pom_mine_v4").map_err(|e| e.to_string())?;
        let kernel_chase = Kernel::create(&program, "pom_mine_v4_chase").map_err(|e| e.to_string())?;
        let kernel_v4_tp = Kernel::create(&program, "pom_mine_v4_tp").map_err(|e| e.to_string())?;
        // Two-phase (offset-chase + pipelined walk) is a PORT of the CUDA v4 restructure, but it is
        // OPT-IN (KERYX_POM_V4_TP=1) and OFF by default: it is a measured LOSS with the dp4a/scalar
        // inner loop (gfx1102 -42%, gfx906 -58%, RX 7600 XT). Reason: the single-phase AMD kernel
        // already hides tile latency across nonces via workgroup occupancy, so the extra chase pass
        // is pure overhead. NVIDIA's +20% needed tensor cores to make the matmul near-free so per-
        // warp MEMORY became the bottleneck — the regime this pipeline helps. Kept (bit-exact
        // validated) as groundwork for an RDNA3 int8-WMMA inner loop, which would recreate that
        // regime; do NOT enable by default until a WMMA build measures a win.
        let use_tp = std::env::var("KERYX_POM_V4_TP").ok().as_deref() == Some("1");
        log::info!("PoM: v4 walk = {} on {dev_name}{}.",
            if use_tp { "two-phase (chase + pipelined, EXPERIMENTAL)" } else { "single-phase" },
            match dot_def { Some((d, _)) => format!(" (native int8 dot: {d})"), None => String::new() });

        let winner = Buffer::<cl_ulong>::create(cref, CL_MEM_READ_WRITE, 1, ptr::null_mut())
            .map_err(|e| e.to_string())?;
        log::info!("PoM: tier resident ({} MiB, {} slab(s)) on {dev_name} — v4 ready.",
            blob_bytes / (1024 * 1024), weights.len());
        Ok(Self { _context: context, queue, kernel_v4, kernel_chase, kernel_v4_tp,
                  offsets: None, use_tp, weights, slab_chunks, winner, n_chunks })
    }

    /// Create + stream the blob as ceil(n_chunks / slab_chunks) buffers (single slab = one buffer).
    /// Fails cleanly on any create/write error so the caller can retry a smaller layout.
    fn build_slabs(cref: &Context, queue: &CommandQueue, index: &crate::pom::WeightIndex, n_chunks: u64, slab_chunks: u64) -> Result<Vec<Buffer<cl_ulong>>, String> {
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

    /// PoM v4 (v0.11.0): grind `batch` nonces from `nonce_base` through the D=32 int8 matrix-state
    /// walk on the resident blob (32 threads/nonce, 8 nonces per 256-thread workgroup). Returns the
    /// lowest winning nonce (host re-walks it byte-exact to build the witness), or None. Multi-slab
    /// aware — slabs are tile-aligned so each step's 1 KB tile lives in one slab (v4_slab picks it).
    pub fn mine_v4(&mut self, pph: [u64; 4], seed: [u64; 4], time: u64, target: [u64; 4], nonce_base: u64, batch: u64) -> Option<u64> {
        let n_tiles = self.n_chunks / POM_V4_TILE_CHUNKS;
        if n_tiles == 0 {
            log::error!("PoM v4: blob too small ({} chunks < one 32-chunk tile).", self.n_chunks);
            return None;
        }
        let k = POM_V4_K as u32;
        let slab_tiles = self.slab_chunks / POM_V4_TILE_CHUNKS;
        // Reset the shared winner, enqueue the WHOLE batch back-to-back (each sub-dispatch stays under
        // the Windows TDR limit but there is NO finish between them, so the GPU never drains its queue
        // mid-batch — the boost clock stays pinned), then finish + read once. The kernel's atomic-min
        // makes the single winner buffer hold the batch's lowest winning nonce.
        self.queue.enqueue_write_buffer(&mut self.winner, CL_BLOCKING, 0, &[u64::MAX], &[]).ok()?;
        let sub_dispatch = v4_sub_dispatch_nonces();
        // Two-phase needs a global offsets buffer ([sub_dispatch][K] u32); allocate it once, lazily.
        if self.use_tp && self.offsets.is_none() {
            let cref = unsafe { Arc::as_ptr(&self._context).as_ref().unwrap() };
            let n = (sub_dispatch * POM_V4_K as u64).max(1);
            match Buffer::<cl_uint>::create(cref, CL_MEM_READ_WRITE, n as usize, ptr::null_mut()) {
                Ok(b) => self.offsets = Some(b),
                Err(e) => {
                    log::warn!("PoM v4: offsets buffer alloc failed ({e}) — falling back to single-phase.");
                    self.use_tp = false;
                }
            }
        }
        let mut done: u64 = 0;
        while done < batch {
            let sub = (batch - done).min(sub_dispatch);
            let base = nonce_base.wrapping_add(done);
            match (self.use_tp, self.offsets.as_ref()) {
                (true, Some(offsets)) => enqueue_v4_tp(&self.queue, &self.kernel_chase, &self.kernel_v4_tp,
                    &self.weights, slab_tiles, offsets, &self.winner, n_tiles, k, pph, seed, time, target, base, sub)?,
                _ => enqueue_v4(&self.queue, &self.kernel_v4, &self.weights, slab_tiles, &self.winner,
                    n_tiles, k, pph, seed, time, target, base, sub)?,
            }
            done += sub;
        }
        self.queue.finish().ok()?;
        let mut w = [u64::MAX];
        self.queue.enqueue_read_buffer(&self.winner, CL_BLOCKING, 0, &mut w, &[]).ok()?;
        if w[0] != u64::MAX { Some(w[0]) } else { None }
    }
}

/// PoM v4 walk constants (mirror src/pom_v4.rs — the byte-exact host reference).
const POM_V4_TILE_CHUNKS: u64 = 32;     // 1 KB tile / 32 B chunk
const POM_V4_K: usize = 256;            // walk steps
/// Nonces per v4 sub-dispatch (must be a multiple of 8 = sub-nonces/workgroup). A v4 nonce is
/// K×(32×32 int8 GEMM) — ~512x lighter than a v3 nonce — so batches are large; 8192 keeps each
/// NDRange well under the Windows TDR while filling big cards. Override via KERYX_POM_V4_SUB_DISPATCH.
fn v4_sub_dispatch_nonces() -> u64 {
    std::env::var("KERYX_POM_V4_SUB_DISPATCH").ok()
        .and_then(|s| s.trim().parse::<u64>().ok()).filter(|&n| n > 0)
        .map(|n| n.max(8) / 8 * 8)
        .unwrap_or(8192)
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
// Retained through the v4 port though currently unreferenced: zero-dup is disabled for PoM v4
// (the Vulkan walk shader only implemented v2), but this arch/VRAM policy — RDNA1 BDA hangs, RADV
// GTT overcommit — is the hard-won field knowledge a future v4 zero-dup shader would reuse.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
#[cfg(unix)]
fn model_plus_blob_fits(device_id: usize, blob_bytes: u64) -> bool {
    let model_bytes = TIER
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|(_, path, _)| std::fs::metadata(path).ok().map(|m| m.len()))
        .unwrap_or(6 << 30); // unknown → assume a big model (conservative: prefer unloading)
    let Ok(total) = opencl3::device::Device::new(device_id as opencl3::types::cl_device_id).global_mem_size() else {
        return false;
    };
    total >= blob_bytes + model_bytes + (1 << 30)
}

#[cfg(unix)]
fn try_claim_shared(_device_id: usize) -> bool {
    // PoM v4/v3 (post-H6): the walk is ALWAYS the OpenCL pom_mine_v4 kernel over the card's own
    // resident blob. The zero-dup Vulkan walk only ever implemented the pre-H6 v2 shader (there is
    // no v4 Vulkan walk), so zero-dup can never serve v4 — every card installs its own OpenCL blob.
    // The in-process llama engine still hosts the model on its card for GPU OPoI inference; the blob
    // and model coexist on a 24 GB card. (Kept as a stub so the install path's call site is unchanged.)
    false
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
static TIER: Mutex<Option<([u8; 32], String, u8)>> = Mutex::new(None);

/// Register the tier to mine (model id, GGUF path on disk, POM_TIERS index). Cheap — no I/O. The
/// heavy load_tier (build the Merkle tree + upload the blob) runs lazily on the first PoM-active job.
/// `model_id` binds the cached `pom-tree.bin` sidecar to this model (upstream e69461d).
pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String, tier: u8) {
    *TIER.lock().unwrap() = Some((model_id, gguf_path, tier));
}

/// Serializes the one-time tier build so concurrent first-time callers (the PoM loop runs one
/// mining thread PER GPU) don't both run build_from_gguf and collide on the shared on-disk Merkle
/// scratch ("failed to fill whole buffer"). The loser waits, then sees is_installed() and returns.
/// Single-GPU AMD boxes never trip this; multi-GPU rigs do. Mirrors pom_gpu's BUILD_LOCK.
static BUILD_LOCK: Mutex<()> = Mutex::new(());

/// Build the shared proof-side WeightIndex from a GGUF, once. Idempotent — returns immediately
/// if it already exists. Caller must hold BUILD_LOCK. This is the heavy, card-independent work
/// (CPU/disk); the per-card VRAM upload streams from this index in install_resident().
fn ensure_index(model_id: [u8; 32], gguf_path: &str, tier: u8) -> Result<(), String> {
    if crate::pom::active_index().is_some() {
        return Ok(());
    }
    log::info!("PoM: building WeightIndex from {gguf_path} (tier {tier})…");
    let mut index = crate::pom::WeightIndex::build_from_gguf(gguf_path, model_id).map_err(|e| e.to_string())?;
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
    let (model_id, path, t) = match tier {
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
        if let Err(e) = ensure_index(model_id, &path, t) {
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

/// PoM v4 (v0.11.0) grind on THIS thread's bound GPU: the D=32 int8 matrix-state walk on the
/// resident blob. Returns the lowest winning nonce; the caller rebuilds the witness host-side
/// (byte-exact CPU re-walk in `pow::generate_block_if_pom`), so a kernel false-positive is silently
/// dropped, never submitted. Every card runs its own OpenCL blob (no zero-dup for v4).
///
/// Word sets mirror `pom_gpu::mine_v4` / `pom_v4.rs`: the POW fold uses the H3-salted pph words
/// ("v4 pow uses the h3 fold") and the SEED fold uses the v4-salted words. Both are pure host-side
/// derivations, so the kernel is era-agnostic — only the uploaded words differ.
pub fn mine_v4(pph: &[u8; 32], time: u64, target_le: &[u8; 32], nonce_base: u64, batch: u64) -> Option<u64> {
    let id = target_dev()?;
    let p = crate::pom::pph_words_for_era(pph, true);
    let s = crate::pom::pph_words_v4(pph);
    let t = words(target_le);
    let miner = miner_for(id)?;
    let mut g = miner.lock().unwrap();
    g.mine_v4(p, s, time, t, nonce_base, batch)
}

/// Build the resident tier from a GGUF (shared proof WeightIndex, streamed to VRAM) and make it
/// resident on the FIRST OpenCL GPU. The multi-GPU production path uses ensure_installed (one
/// resident copy per card); this single-device form backs the tests + any non-bound caller.
pub fn load_tier(gguf_path: &str, tier: u8) -> Result<(), String> {
    // Non-bound/test entry: no model id at hand, so the cache is bound to the zero id and any
    // production (set_mining_tier) tree for a real model is treated as foreign and rebuilt.
    ensure_index([0u8; 32], gguf_path, tier)?;
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
mod v4_byte_exact {
    //! Byte-exact GPU-vs-CPU check for the PoM v4 (v0.11.0) walk. Builds a synthetic possession
    //! blob, grinds a handful of nonces on a REAL OpenCL GPU via `pom_mine_v4`, and asserts the
    //! winner matches the CPU host reference (`pom_v4::build_proof_v4` gives the final_state, then
    //! `pom_pow_value`). Consensus is byte-exact, so any divergence is a kernel bug. Force a slab
    //! count with KERYX_POM_CL_SLABS=2/4 to exercise the multi-slab addressing. Needs an AMD
    //! OpenCL GPU (skips if none).
    use super::*;

    // CPU reference pow-value (256-bit LE) for one nonce over `index`. v4 seed = pom_block_seed_v4;
    // pow fold uses the H3-salted words ("v4 pow uses the h3 fold").
    fn cpu_pow(index: &crate::pom::WeightIndex, pph: &[u8; 32], time: u64, nonce: u64) -> [u8; 32] {
        let seed = crate::pom::pom_block_seed_v4(pph, time, nonce);
        let (_proof, fin) = crate::pom_v4::build_proof_v4(0, seed, index).unwrap();
        crate::pom::pom_pow_value(fin, pph, true)
    }

    #[test]
    fn gpu_v4_matches_cpu_reference() {
        // First AMD OpenCL GPU, or skip (CI without a GPU).
        let dev_ids = match opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU) {
            Ok(v) if !v.is_empty() => v,
            _ => { eprintln!("no OpenCL GPU — skipping v4 byte-exact test"); return; }
        };
        let dev = opencl3::device::Device::new(dev_ids[0]);
        eprintln!("v4 byte-exact on: {}", dev.name().unwrap_or_default());

        // Synthetic 128-tile blob (4096 chunks). Multiple tiles so the offset chain visits several,
        // and 128 tiles split cleanly into tile-aligned slabs at KERYX_POM_CL_SLABS=2/4.
        let n_chunks: u64 = 128 * POM_V4_TILE_CHUNKS;
        let mut data = vec![0u8; (n_chunks * 32) as usize];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i.wrapping_mul(131).wrapping_add(i / 7).wrapping_add(17)) as u8;
        }
        let index = crate::pom::index_from_ram(data);

        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(23).wrapping_add(5));
        let time: u64 = 0x0123_4567_89AB_CDEF;
        const NN: u64 = 2048; // matches the CUDA gate: a mapping/ordering bug in either kernel or
                              // the two-phase offset chain shows up as an argmin mismatch here

        // CPU: find the argmin pow over 0..NN; target = that min → only the argmin passes.
        let mut best = ([0xFFu8; 32], u64::MAX);
        for nonce in 0..NN {
            let pv = cpu_pow(&index, &pph, time, nonce);
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

        // GPU: grind 0..NN with target = the argmin's pow. Expect it to return w_cpu. Validate BOTH
        // the single-phase kernel AND the (default-off) two-phase chase+pipeline path — regardless
        // of the KERYX_POM_V4_TP env default — so neither can silently diverge from consensus.
        let mut miner = PomMiner::new(dev, &index, n_chunks).expect("PomMiner::new");
        let p = crate::pom::pph_words_for_era(&pph, true);
        let s = crate::pom::pph_words_v4(&pph);
        let t = words(&target);
        miner.use_tp = false;
        assert_eq!(miner.mine_v4(p, s, time, t, 0, NN), Some(w_cpu), "single-phase v4 winner mismatch");
        miner.use_tp = true;
        assert_eq!(miner.mine_v4(p, s, time, t, 0, NN), Some(w_cpu), "two-phase v4 winner mismatch");
    }

    /// Throughput A/B: single-phase vs two-phase v4 on a memory-bound synthetic blob. Ignored
    /// (needs a GPU + is slow). Run: cargo test --release -p keryx-miner-supr --features pom-opencl
    /// --lib v4_two_phase_bench -- --ignored --nocapture   (KERYX_BENCH_CL_DEV picks the device).
    #[test]
    #[ignore]
    fn v4_two_phase_bench() {
        let dev_ids = match opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU) {
            Ok(v) if !v.is_empty() => v,
            _ => { eprintln!("no OpenCL GPU — skipping"); return; }
        };
        let di: usize = std::env::var("KERYX_BENCH_CL_DEV").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let dev = opencl3::device::Device::new(dev_ids[di]);
        eprintln!("v4 two-phase bench on: {}", dev.name().unwrap_or_default());

        // ~256 MB blob (262144 tiles) — far larger than any GPU last-level cache, so the cold-tile
        // chase is genuinely memory-latency-bound (where the pipeline helps).
        let n_tiles: u64 = std::env::var("KERYX_BENCH_TILES").ok().and_then(|s| s.parse().ok()).unwrap_or(262144);
        let n_chunks = n_tiles * POM_V4_TILE_CHUNKS;
        eprintln!("building {} MiB synthetic blob ({n_tiles} tiles)…", n_chunks * 32 / (1024 * 1024));
        let mut data = vec![0u8; (n_chunks * 32) as usize];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i.wrapping_mul(131).wrapping_add(i / 7).wrapping_add(17)) as u8;
        }
        let index = crate::pom::index_from_ram(data);
        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(23).wrapping_add(5));
        let p = crate::pom::pph_words_for_era(&pph, true);
        let s = crate::pom::pph_words_v4(&pph);
        let t = [0u64; 4]; // impossible target → full grind, no early winner
        let batch: u64 = std::env::var("KERYX_BENCH_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 16);

        let mut miner = PomMiner::new(dev, &index, n_chunks).expect("PomMiner::new");
        let bench = |m: &mut PomMiner, label: &str| {
            m.mine_v4(p, s, 1_700_000_000, t, 0, batch); // warmup
            let rounds = 6u64;
            let start = std::time::Instant::now();
            for r in 0..rounds { m.mine_v4(p, s, 1_700_000_000, t, r * batch, batch); }
            let secs = start.elapsed().as_secs_f64();
            let mhs = (rounds * batch) as f64 / secs / 1e6;
            eprintln!("{label}: {mhs:.3} Mh/s ({} nonces in {secs:.2}s)", rounds * batch);
            mhs
        };
        miner.use_tp = false;
        let single = bench(&mut miner, "single-phase");
        miner.use_tp = true;
        let two = bench(&mut miner, "two-phase ");
        eprintln!("two-phase speedup: {:+.1}%", (two / single - 1.0) * 100.0);
    }
}
