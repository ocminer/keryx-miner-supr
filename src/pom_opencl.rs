// PomMiner — the AMD-side OpenCL PoM mining driver (destined for the keryxopencl plugin).
// Holds the tier weight blob resident in one cl_mem buffer; mine() launches pom_mine over a
// nonce batch and returns the lowest passing nonce (the host then re-verifies + builds the proof).
// Mirrors the opencl3 0.6 patterns in plugins/opencl/src/worker.rs.

use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
    /// RDNA3+ matrix-core (WMMA) two-phase walk; None on non-gfx11/12. Paired with the chase.
    kernel_wmma: Option<Kernel>,
    /// RDNA3+ SINGLE-phase WMMA walk (no chase) — the AMD-fit combination. None on non-gfx11/12.
    kernel_wmma_sp: Option<Kernel>,
    /// gfx1100-specialized single-phase WMMA walk with one in-place state buffer.
    kernel_wmma_1state: Option<Kernel>,
    /// Selected v4 walk mode.
    mode: V4Mode,
    /// Global scratch for phase-1 offsets ([sub_dispatch][K] u32); allocated lazily on the first
    /// two-phase mine_v4. None for single-phase modes or not yet used.
    offsets: Option<Buffer<cl_uint>>,
    weights: Vec<Buffer<cl_ulong>>,
    /// Chunks per slab (single-slab layout: == n_chunks). Slabs are tile-aligned.
    slab_chunks: u64,
    winner: Buffer<cl_ulong>,
    pub n_chunks: u64,
}

// OpenCL handles are plain cl_* pointers usable from any single thread; the global Mutex
// serializes all access (one mining thread), so sending the miner across threads is sound.
unsafe impl Send for PomMiner {}

/// v4 walk kernel selection. Single-phase dp4a is the portable default; the WMMA modes are RDNA3+
/// only. SingleWmma (matrix-core matmul, NO offset chase) is the AMD-fit path — the two-phase chase
/// is dead weight on AMD because the single-phase kernel is already occupancy-latency-hidden.
#[derive(Clone, Copy, PartialEq)]
enum V4Mode {
    SingleDp4a,
    SingleWmma,
    SingleWmma1,
    TwoPhaseDp4a,
    TwoPhaseWmma,
}

fn device_arch_is(device_name: &str, arch: &str) -> bool {
    device_name.split(|c: char| !c.is_ascii_alphanumeric()).any(|token| token.eq_ignore_ascii_case(arch))
}

fn select_v4_mode(
    requested: Option<&str>,
    have_wmma_sp: bool,
    have_wmma_1state: bool,
    have_wmma_tp: bool,
    prefer_wmma_1state: bool,
) -> V4Mode {
    match requested {
        Some("wmma1") if have_wmma_1state => V4Mode::SingleWmma1,
        Some("wmma") if have_wmma_sp => V4Mode::SingleWmma,
        Some("tpwmma") if have_wmma_tp => V4Mode::TwoPhaseWmma,
        Some("tp") => V4Mode::TwoPhaseDp4a,
        Some("sp") => V4Mode::SingleDp4a,
        _ if prefer_wmma_1state && have_wmma_1state => V4Mode::SingleWmma1,
        _ if have_wmma_sp => V4Mode::SingleWmma,
        _ => V4Mode::SingleDp4a,
    }
}

/// Enqueue one v4 sub-dispatch (no finish/read): 256-thread workgroups of V4_NPG(8) sub-nonces.
/// `n_nonces` nonces from `base`; groups = ceil(n_nonces/8) (tail sub-nonces walk a dummy nonce and
/// never submit — uniform barriers). The kernel CAS-mins into the shared `winner`; mine_v4 enqueues
/// the whole batch back-to-back then finishes ONCE so the GPU queue never drains mid-batch.
#[allow(clippy::too_many_arguments)]
fn enqueue_v4(
    queue: &CommandQueue,
    kernel: &Kernel,
    weights: &[Buffer<cl_ulong>],
    slab_tiles: u64,
    winner: &Buffer<cl_ulong>,
    n_tiles: u64,
    k: u32,
    pph: [u64; 4],
    seed: [u64; 4],
    time: u64,
    target: [u64; 4],
    base: u64,
    n_nonces: u64,
    winner_base: u64,
    h10: u32,
    lds_bytes: usize,
) -> Result<(), String> {
    const V4_LOCAL: usize = 256;
    const V4_NPG: u64 = 8; // sub-nonces per workgroup (kernel mirror)
                           // 4 slab args; absent slabs repeat slab 0 (never selected: off/slab_tiles bounds to real slabs).
    let sl = |i: usize| weights.get(i).unwrap_or(&weights[0]);
    let groups = n_nonces.div_ceil(V4_NPG);
    let global = (groups * V4_LOCAL as u64) as usize;
    ExecuteKernel::new(kernel)
        .set_arg(sl(0))
        .set_arg(sl(1))
        .set_arg(sl(2))
        .set_arg(sl(3))
        .set_arg(&n_tiles)
        .set_arg(&slab_tiles)
        .set_arg(&k)
        .set_arg(&pph[0])
        .set_arg(&pph[1])
        .set_arg(&pph[2])
        .set_arg(&pph[3])
        .set_arg(&seed[0])
        .set_arg(&seed[1])
        .set_arg(&seed[2])
        .set_arg(&seed[3])
        .set_arg(&time)
        .set_arg(&target[0])
        .set_arg(&target[1])
        .set_arg(&target[2])
        .set_arg(&target[3])
        .set_arg(&base)
        .set_arg(&n_nonces)
        .set_arg(&winner_base)
        .set_arg(&h10)
        .set_arg(winner)
        .set_arg_local_buffer(lds_bytes)
        .set_global_work_size(global)
        .set_local_work_size(V4_LOCAL)
        .enqueue_nd_range(queue)
        .map_err(|e| format!("single-phase enqueue failed: {e}"))?;
    Ok(())
}

/// Two-phase v4 sub-dispatch: phase 1 (chase) resolves all K offsets into `offsets`, then phase 2
/// (pipelined walk) reads them and prefetches tile t+1 during matmul t. Same in-order queue, so the
/// walk sees the chase's writes without an explicit barrier. `offsets` must hold ≥ n_nonces*K u32.
#[allow(clippy::too_many_arguments)]
fn enqueue_v4_tp(
    queue: &CommandQueue,
    chase: &Kernel,
    walk: &Kernel,
    weights: &[Buffer<cl_ulong>],
    slab_tiles: u64,
    offsets: &Buffer<cl_uint>,
    winner: &Buffer<cl_ulong>,
    n_tiles: u64,
    k: u32,
    pph: [u64; 4],
    seed: [u64; 4],
    time: u64,
    target: [u64; 4],
    base: u64,
    n_nonces: u64,
    winner_base: u64,
    h10: u32,
    walk_lds_bytes: usize,
) -> Result<(), String> {
    const V4_LOCAL: usize = 256;
    const V4_NPG: u64 = 8;
    const CHASE_LOCAL: usize = 64; // plain 1D latency-bound pointer chase
    let sl = |i: usize| weights.get(i).unwrap_or(&weights[0]);
    // Phase 1: one work-item per nonce, rounded up to CHASE_LOCAL.
    let chase_global = (n_nonces as usize).div_ceil(CHASE_LOCAL) * CHASE_LOCAL;
    ExecuteKernel::new(chase)
        .set_arg(sl(0))
        .set_arg(sl(1))
        .set_arg(sl(2))
        .set_arg(sl(3))
        .set_arg(&n_tiles)
        .set_arg(&slab_tiles)
        .set_arg(&k)
        .set_arg(&seed[0])
        .set_arg(&seed[1])
        .set_arg(&seed[2])
        .set_arg(&seed[3])
        .set_arg(&time)
        .set_arg(&base)
        .set_arg(&n_nonces)
        .set_arg(&h10)
        .set_arg(offsets)
        .set_global_work_size(chase_global)
        .set_local_work_size(CHASE_LOCAL)
        .enqueue_nd_range(queue)
        .map_err(|e| format!("two-phase chase enqueue failed: {e}"))?;
    // Phase 2: pipelined walk.
    let groups = n_nonces.div_ceil(V4_NPG);
    let global = (groups * V4_LOCAL as u64) as usize;
    ExecuteKernel::new(walk)
        .set_arg(sl(0))
        .set_arg(sl(1))
        .set_arg(sl(2))
        .set_arg(sl(3))
        .set_arg(&n_tiles)
        .set_arg(&slab_tiles)
        .set_arg(&k)
        .set_arg(&pph[0])
        .set_arg(&pph[1])
        .set_arg(&pph[2])
        .set_arg(&pph[3])
        .set_arg(&seed[0])
        .set_arg(&seed[1])
        .set_arg(&seed[2])
        .set_arg(&seed[3])
        .set_arg(&time)
        .set_arg(&target[0])
        .set_arg(&target[1])
        .set_arg(&target[2])
        .set_arg(&target[3])
        .set_arg(&base)
        .set_arg(&n_nonces)
        .set_arg(&winner_base)
        .set_arg(&h10)
        .set_arg(offsets)
        .set_arg(winner)
        .set_arg_local_buffer(walk_lds_bytes)
        .set_global_work_size(global)
        .set_local_work_size(V4_LOCAL)
        .enqueue_nd_range(queue)
        .map_err(|e| format!("two-phase walk enqueue failed: {e}"))?;
    Ok(())
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
                blob_bytes / (1024 * 1024),
                reported_max / (1024 * 1024),
            );
        }
        const V3_TILE: u64 = POM_V4_TILE_CHUNKS; // slab alignment quantum (32 chunks = 1 KB tile)
        let align_up = |c: u64| c.div_ceil(V3_TILE) * V3_TILE;
        // KERYX_POM_CL_SLABS forces a slab count; KERYX_POM_CL_SLAB_SHIFT (legacy) forces a
        // 2^shift-chunk slab size, translated to the equivalent count. Unset = adaptive 1..=4.
        let forced_k: Option<u64> = std::env::var("KERYX_POM_CL_SLABS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&k| (1..=4).contains(&k))
            .or_else(|| {
                std::env::var("KERYX_POM_CL_SLAB_SHIFT").ok().and_then(|v| v.parse::<u32>().ok()).map(|sh| {
                    if sh >= 63 {
                        1
                    } else {
                        n_chunks.div_ceil(1u64 << sh).clamp(1, 4)
                    }
                })
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
                last_err = format!(
                    "{k}-slab layout needs {} MiB/slab > reported max-alloc {} MiB",
                    slab_bytes / (1024 * 1024),
                    reported_max / (1024 * 1024)
                );
                log::info!("PoM: skipping {k}-slab layout ({last_err}).");
                continue;
            }
            match Self::build_slabs(cref, &queue, index, n_chunks, slab_chunks) {
                Ok(v) => {
                    if v.len() > 1 {
                        log::info!(
                            "PoM: blob resident as {} slabs of {} MiB (tile-aligned, {} chunks/slab).",
                            v.len(),
                            slab_bytes / (1024 * 1024),
                            slab_chunks,
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
        // (dot9-insts); dot1-capable GCN/CDNA/RDNA targets use `sdot4`; both are byte-identical to
        // the scalar unpack. The exact allowlist avoids gfx1010, which lacks dot1-insts. If a
        // vendor compiler rejects a nominally capable target, the existing define-less retry keeps
        // correctness and availability.
        // KERYX_NO_AMD_DOT4 forces scalar; failed builds retry define-less.
        let allow_dot = std::env::var("KERYX_NO_AMD_DOT4").is_err();
        let dev_name_lc = dev_name.to_ascii_lowercase();
        let has_sdot4 = [
            "gfx906", "gfx908", "gfx90a", // Vega/CDNA 1-2
            "gfx1011", "gfx1012", // Navi 12/14 (gfx1010 is intentionally excluded)
            "gfx1030", "gfx1031", "gfx1032", "gfx1033", // RDNA2 family
            "gfx1034", "gfx1035", "gfx1036", "gfx940", "gfx941", "gfx942", // CDNA3
        ]
        .iter()
        .any(|arch| dev_name_lc.contains(arch));
        let dot_def = if !allow_dot {
            None
        } else if dev_name_lc.contains("gfx11") || dev_name_lc.contains("gfx12") {
            Some(("sudot4 (RDNA3+ dot9-insts)", "-D USE_AMD_DOT4=1"))
        } else if has_sdot4 {
            Some(("sdot4 (dot1-insts)", "-D USE_AMD_SDOT4=1"))
        } else {
            None
        };
        // v4 bakes n_tiles (POM_NT) and tiles-per-slab (POM_SLABT) — the two runtime-divisor
        // divisions in the walk's hot paths (offset % n_tiles, off / slab_tiles).
        // RDNA3/RDNA4 (gfx11/gfx12) also get -D USE_AMD_WMMA=1, which compiles the matrix-core
        // v4 walk (V_WMMA_I32_16X16X16_IU8); the kernel is #ifdef'd out elsewhere (the builtin
        // needs the wmma target feature). `have_wmma` gates kernel creation below.
        let have_wmma = dev_name_lc.contains("gfx11") || dev_name_lc.contains("gfx12");
        let base = {
            let b = format!("-D POM_NT={}UL -D POM_SLABT={}UL", n_chunks / V3_TILE, slab_chunks / V3_TILE);
            if have_wmma {
                format!("{b} -D USE_AMD_WMMA=1")
            } else {
                b
            }
        };
        let opts = match dot_def {
            Some((_, d)) => format!("{base} {d}"),
            None => base.clone(),
        };
        let program = match Program::create_and_build_from_source(&context, POM_SRC, &opts) {
            Ok(p) => {
                if let Some((desc, _)) = dot_def {
                    log::info!("PoM: v4 int8 matmul using native v_dot4_i32_i8 ({desc}) on {dev_name}.");
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
        // RDNA3+ matrix-core (WMMA) kernels: proven single-phase, gfx1100-specialized one-state
        // single-phase, and two-phase flavors. Absent on non-gfx11/12 (the builtin is #ifdef'd
        // out); create failures degrade to the proven single-phase WMMA or dp4a path.
        let (kernel_wmma_sp, kernel_wmma_1state, kernel_wmma) = if have_wmma {
            (
                Kernel::create(&program, "pom_mine_v4_wmma_sp").ok(),
                Kernel::create(&program, "pom_mine_v4_wmma_1state").ok(),
                Kernel::create(&program, "pom_mine_v4_wmma").ok(),
            )
        } else {
            (None, None, None)
        };
        // Mode: KERYX_POM_V4_MODE = sp(single-phase dp4a) | wmma(proven single-phase matrix-core)
        // | wmma1(one-state matrix-core) | tp | tpwmma. gfx1100 defaults to one-state after a
        // repeatable +4.2..4.5% A/B; gfx1102/gfx12 retain the proven WMMA default (gfx1102 measured
        // only +0.2..0.5%). Explicit `wmma` and `wmma1` remain available for diagnostics.
        let requested_mode = std::env::var("KERYX_POM_V4_MODE").ok();
        let prefer_wmma_1state = device_arch_is(&dev_name, "gfx1100");
        let mode = select_v4_mode(
            requested_mode.as_deref(),
            kernel_wmma_sp.is_some(),
            kernel_wmma_1state.is_some(),
            kernel_wmma.is_some(),
            prefer_wmma_1state,
        );
        log::info!(
            "PoM: v4 walk = {} on {dev_name}{}.",
            match mode {
                V4Mode::SingleDp4a => "single-phase dp4a",
                V4Mode::SingleWmma => "single-phase WMMA matrix-core",
                V4Mode::SingleWmma1 => "single-phase WMMA one-state (gfx1100-specialized)",
                V4Mode::TwoPhaseDp4a => "two-phase dp4a (experimental)",
                V4Mode::TwoPhaseWmma => "two-phase WMMA (experimental)",
            },
            match dot_def {
                Some((d, _)) => format!(" (native int8 dot: {d})"),
                None => String::new(),
            }
        );

        let winner =
            Buffer::<cl_ulong>::create(cref, CL_MEM_READ_WRITE, 1, ptr::null_mut()).map_err(|e| e.to_string())?;
        log::info!(
            "PoM: tier resident ({} MiB, {} slab(s)) on {dev_name} — v4 ready.",
            blob_bytes / (1024 * 1024),
            weights.len()
        );
        Ok(Self {
            _context: context,
            queue,
            kernel_v4,
            kernel_chase,
            kernel_v4_tp,
            kernel_wmma,
            kernel_wmma_sp,
            kernel_wmma_1state,
            mode,
            offsets: None,
            weights,
            slab_chunks,
            winner,
            n_chunks,
        })
    }

    /// Create + stream the blob as ceil(n_chunks / slab_chunks) buffers (single slab = one buffer).
    /// Fails cleanly on any create/write error so the caller can retry a smaller layout.
    fn build_slabs(
        cref: &Context,
        queue: &CommandQueue,
        index: &crate::pom::WeightIndex,
        n_chunks: u64,
        slab_chunks: u64,
    ) -> Result<Vec<Buffer<cl_ulong>>, String> {
        let n_slabs = n_chunks.div_ceil(slab_chunks.max(1));
        if n_slabs > 4 {
            return Err(format!(
                "{n_slabs} slabs needed but the kernel takes at most 4 — tier too big for this layout"
            ));
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
    pub fn mine_v4(
        &mut self,
        pph: [u64; 4],
        seed: [u64; 4],
        time: u64,
        target: [u64; 4],
        nonce_base: u64,
        batch: u64,
        h10: u32,
    ) -> Result<Option<u64>, String> {
        if batch == 0 {
            return Ok(None);
        }
        if batch > u32::MAX as u64 {
            return Err("batch exceeds u32 launch domain".into());
        }
        if nonce_base.checked_add(batch - 1).is_none() {
            return Err("nonce range crosses u64::MAX; split it before launch".into());
        }
        let n_tiles = self.n_chunks / POM_V4_TILE_CHUNKS;
        if n_tiles == 0 {
            return Err(format!("blob too small ({} chunks < one 32-chunk tile)", self.n_chunks));
        }
        let k = POM_V4_K as u32;
        let slab_tiles = self.slab_chunks / POM_V4_TILE_CHUNKS;
        // Reset the shared winner, enqueue the WHOLE batch back-to-back (each sub-dispatch stays under
        // the Windows TDR limit but there is NO finish between them, so the GPU never drains its queue
        // mid-batch — the boost clock stays pinned), then finish + read once. The kernel's atomic-min
        // makes the single winner buffer hold the batch's lowest winning nonce.
        self.queue
            .enqueue_write_buffer(&mut self.winner, CL_BLOCKING, 0, &[u64::MAX], &[])
            .map_err(|e| format!("winner reset failed: {e}"))?;
        let sub_dispatch = v4_sub_dispatch_nonces();
        // LDS per mode (8 sub-nonces × strip): single dp4a needs only 384 u32 (12 KB: 256-word
        // tile/leaves + the 128-word largest Merkle level), proven single WMMA 768 (24 KB),
        // experimental one-state WMMA 512 (16 KB), two-phase dp4a 512 (16 KB), and two-phase
        // WMMA 1024 (32 KB).
        const SP_DP4A_LDS: usize = 8 * 384 * 4;
        const SP_WMMA_LDS: usize = 8 * 768 * 4;
        const SP_WMMA1_LDS: usize = 8 * 512 * 4;
        const TP_DP4A_LDS: usize = 8 * 512 * 4;
        const TP_WMMA_LDS: usize = 8 * 1024 * 4;
        // Two-phase modes need a global offsets buffer ([sub_dispatch][K] u32); lazy-allocate it.
        let two_phase = matches!(self.mode, V4Mode::TwoPhaseDp4a | V4Mode::TwoPhaseWmma);
        if two_phase && self.offsets.is_none() {
            let cref = unsafe { Arc::as_ptr(&self._context).as_ref().unwrap() };
            let n = (sub_dispatch * POM_V4_K as u64).max(1);
            match Buffer::<cl_uint>::create(cref, CL_MEM_READ_WRITE, n as usize, ptr::null_mut()) {
                Ok(b) => self.offsets = Some(b),
                Err(e) => {
                    // A two-phase WMMA request still has a fully independent single-phase WMMA
                    // kernel; preserve matrix-core acceleration when only the offsets allocation
                    // failed. The dp4a mode falls back to its own single-phase counterpart.
                    self.mode = if self.mode == V4Mode::TwoPhaseWmma && self.kernel_wmma_sp.is_some() {
                        V4Mode::SingleWmma
                    } else {
                        V4Mode::SingleDp4a
                    };
                    log::warn!(
                        "PoM v4: offsets buffer alloc failed ({e}) — falling back to {}.",
                        if self.mode == V4Mode::SingleWmma { "single-phase WMMA" } else { "single-phase dp4a" }
                    );
                }
            }
        }
        let mut done: u64 = 0;
        while done < batch {
            let sub = (batch - done).min(sub_dispatch);
            let base = nonce_base.wrapping_add(done);
            // Resolve the walk kernel: WMMA modes fall back to dp4a if the kernel is missing.
            match self.mode {
                V4Mode::SingleWmma1 => enqueue_v4(
                    &self.queue,
                    self.kernel_wmma_1state.as_ref().unwrap_or(&self.kernel_v4),
                    &self.weights,
                    slab_tiles,
                    &self.winner,
                    n_tiles,
                    k,
                    pph,
                    seed,
                    time,
                    target,
                    base,
                    sub,
                    nonce_base,
                    h10,
                    if self.kernel_wmma_1state.is_some() { SP_WMMA1_LDS } else { SP_DP4A_LDS },
                )?,
                V4Mode::SingleWmma => enqueue_v4(
                    &self.queue,
                    self.kernel_wmma_sp.as_ref().unwrap_or(&self.kernel_v4),
                    &self.weights,
                    slab_tiles,
                    &self.winner,
                    n_tiles,
                    k,
                    pph,
                    seed,
                    time,
                    target,
                    base,
                    sub,
                    nonce_base,
                    h10,
                    if self.kernel_wmma_sp.is_some() { SP_WMMA_LDS } else { SP_DP4A_LDS },
                )?,
                V4Mode::TwoPhaseWmma => enqueue_v4_tp(
                    &self.queue,
                    &self.kernel_chase,
                    self.kernel_wmma.as_ref().unwrap_or(&self.kernel_v4_tp),
                    &self.weights,
                    slab_tiles,
                    self.offsets.as_ref().unwrap(),
                    &self.winner,
                    n_tiles,
                    k,
                    pph,
                    seed,
                    time,
                    target,
                    base,
                    sub,
                    nonce_base,
                    h10,
                    if self.kernel_wmma.is_some() { TP_WMMA_LDS } else { TP_DP4A_LDS },
                )?,
                V4Mode::TwoPhaseDp4a => enqueue_v4_tp(
                    &self.queue,
                    &self.kernel_chase,
                    &self.kernel_v4_tp,
                    &self.weights,
                    slab_tiles,
                    self.offsets.as_ref().unwrap(),
                    &self.winner,
                    n_tiles,
                    k,
                    pph,
                    seed,
                    time,
                    target,
                    base,
                    sub,
                    nonce_base,
                    h10,
                    TP_DP4A_LDS,
                )?,
                V4Mode::SingleDp4a => enqueue_v4(
                    &self.queue,
                    &self.kernel_v4,
                    &self.weights,
                    slab_tiles,
                    &self.winner,
                    n_tiles,
                    k,
                    pph,
                    seed,
                    time,
                    target,
                    base,
                    sub,
                    nonce_base,
                    h10,
                    SP_DP4A_LDS,
                )?,
            }
            done += sub;
        }
        self.queue.finish().map_err(|e| format!("queue finish failed: {e}"))?;
        let mut w = [u64::MAX];
        self.queue
            .enqueue_read_buffer(&self.winner, CL_BLOCKING, 0, &mut w, &[])
            .map_err(|e| format!("winner read failed: {e}"))?;
        Ok(if w[0] != u64::MAX { Some(nonce_base + w[0]) } else { None })
    }
}

/// PoM v4 walk constants (mirror src/pom_v4.rs — the byte-exact host reference).
const POM_V4_TILE_CHUNKS: u64 = 32; // 1 KB tile / 32 B chunk
const POM_V4_K: usize = 256; // walk steps
/// Nonces per v4 sub-dispatch (must be a multiple of 8 = sub-nonces/workgroup). A v4 nonce is
/// K×(32×32 int8 GEMM) — ~512x lighter than a v3 nonce — so batches are large; 8192 keeps each
/// NDRange well under the Windows TDR while filling big cards. Override via KERYX_POM_V4_SUB_DISPATCH.
fn v4_sub_dispatch_nonces() -> u64 {
    std::env::var("KERYX_POM_V4_SUB_DISPATCH")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
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

/// Exact OpenCL devices selected by the worker plugin, in worker-label order. `None` means the
/// host has not published plugin selection yet (unit tests and legacy direct library callers), in
/// which case the old all-GPU discovery remains available as a compatibility fallback. `Some([])`
/// is deliberately authoritative: OpenCL was processed but no OpenCL worker was selected, so the
/// inference/VRAM helpers must not accidentally adopt a CUDA card exposed through another OpenCL
/// platform.
static SELECTED_WORKER_DEVS: Mutex<Option<Vec<usize>>> = Mutex::new(None);

/// Internal hand-off to the bundled ggml/Vulkan picker. Values are comma-separated PCI
/// `dddd:bb:dd.f` locations derived from the exact selected OpenCL GPU workers. An explicitly present
/// but empty value means auto-selection has no eligible device and must fail closed.
pub const LLAMA_VK_AUTO_PCI_ALLOWLIST_ENV: &str = "KERYX_LLAMA_VK_AUTO_PCI_ALLOWLIST";

fn normalize_device_ids(device_ids: Vec<usize>) -> Vec<usize> {
    let mut unique = Vec::with_capacity(device_ids.len());
    for id in device_ids {
        if !unique.contains(&id) {
            unique.push(id);
        }
    }
    unique
}

fn format_pci_location((domain, bus, device, function): (u32, u32, u32, u32)) -> String {
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
}

/// Publish the `cl_device_id`s of the workers selected by the OpenCL plugin. The host calls this
/// immediately after plugin option processing, before tier sizing or inference placement. Keeping
/// the identity here closes two mixed-rig bugs: global OpenCL enumeration could size the model from
/// an unselected card (or a different vendor's platform), and could dedicate that unrelated card
/// to inference while the actual mining workers were left with an impossible memory plan.
pub fn set_selected_worker_devices(device_ids: Vec<usize>) {
    let device_ids = normalize_device_ids(device_ids);
    #[cfg(unix)]
    {
        let pci_allowlist = device_ids
            .iter()
            .copied()
            .filter(|id| device_is_gpu(*id))
            .filter_map(device_pci_full)
            .map(format_pci_location)
            .collect::<Vec<_>>()
            .join(",");
        // This runs directly after plugin option parsing, before any inference warmer starts.
        // Explicit KERYX_LLAMA_VK_DEVICE bypasses the auto-picker and therefore this allowlist.
        std::env::set_var(LLAMA_VK_AUTO_PCI_ALLOWLIST_ENV, &pci_allowlist);
        if !device_ids.is_empty() && pci_allowlist.is_empty() {
            log::warn!(
                "PoM[opencl]: selected workers expose no usable PCI identity; automatic Vulkan \
                 inference placement will fail closed (KERYX_LLAMA_VK_DEVICE remains available)."
            );
        }
    }
    log::info!("PoM[opencl]: device helpers scoped to {} selected OpenCL worker(s).", device_ids.len());
    *SELECTED_WORKER_DEVS.lock().unwrap_or_else(|p| p.into_inner()) = Some(device_ids);
}

fn configured_worker_devices() -> Option<Vec<usize>> {
    SELECTED_WORKER_DEVS.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

fn resolve_device_scope<F>(configured: Option<Vec<usize>>, discover: F) -> Vec<usize>
where
    F: FnOnce() -> Vec<usize>,
{
    configured.unwrap_or_else(discover)
}

/// Worker devices in exact plugin order. Discovery is only a pre-publication compatibility path;
/// after publication even an empty selection remains empty rather than escaping to all platforms.
fn scoped_worker_devices() -> Vec<usize> {
    resolve_device_scope(configured_worker_devices(), || {
        opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .unwrap_or_default()
            .into_iter()
            .map(|id| id as usize)
            .collect()
    })
}

fn device_is_gpu(id: usize) -> bool {
    opencl3::device::Device::new(id as opencl3::types::cl_device_id)
        .dev_type()
        .map(|ty| ty & opencl3::device::CL_DEVICE_TYPE_GPU != 0)
        .unwrap_or(false)
}

/// Inference placement and VRAM sizing only make sense for GPU workers. The plugin historically
/// enumerates `CL_DEVICE_TYPE_ALL`, so keep an explicitly selected CPU/accelerator worker bound for
/// its mining thread but exclude it from GPU inference decisions.
fn scoped_gpu_devices() -> Vec<usize> {
    scoped_worker_devices().into_iter().filter(|id| device_is_gpu(*id)).collect()
}

/// OpenCL's inference server and mining contexts can target the same physical GPU. Pause every AMD
/// walk while GPU inference runs (the server route is singleton today), and use each miner mutex as
/// the completion barrier for a command that was already enqueued when the pause began.
static INFERENCE_PAUSE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Covers the whole device-facing portion of `ensure_installed` (including any future shared-walk
/// probe), while `INSTALLING` below covers direct resident uploads. The two registries make an
/// empty `POM_BY_DEV` snapshot meaningful: no allocator/JIT/upload can be racing toward publication.
static GPU_SETUP: Mutex<Vec<usize>> = Mutex::new(Vec::new());

struct GpuSetupGuard(usize);

impl GpuSetupGuard {
    fn begin(device_id: usize) -> Option<Self> {
        let mut setup = GPU_SETUP.lock().unwrap_or_else(|p| p.into_inner());
        if INFERENCE_PAUSE_COUNT.load(Ordering::Acquire) != 0 || setup.contains(&device_id) {
            return None;
        }
        setup.push(device_id);
        Some(Self(device_id))
    }
}

impl Drop for GpuSetupGuard {
    fn drop(&mut self) {
        GPU_SETUP.lock().unwrap_or_else(|p| p.into_inner()).retain(|d| *d != self.0);
    }
}

pub struct InferenceDrainGuard;

impl Drop for InferenceDrainGuard {
    fn drop(&mut self) {
        INFERENCE_PAUSE_COUNT.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn pause_and_drain_for_inference(_gpu: usize) -> Option<InferenceDrainGuard> {
    INFERENCE_PAUSE_COUNT.fetch_add(1, Ordering::AcqRel);
    let guard = InferenceDrainGuard;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        // Taking both coordination locks after publishing the pause closes the check/claim race:
        // an older setup is visible here, while a newer one observes the pause and defers.
        let setup_busy = !GPU_SETUP.lock().unwrap_or_else(|p| p.into_inner()).is_empty();
        let upload_busy = !INSTALLING.lock().unwrap_or_else(|p| p.into_inner()).is_empty();
        if !setup_busy && !upload_busy {
            break;
        }
        if std::time::Instant::now() >= deadline {
            log::error!("PoM[opencl]: model install did not drain within 30s — refusing GPU inference");
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let residents: Vec<Arc<Mutex<PomMiner>>> =
        POM_BY_DEV.lock().map(|miners| miners.iter().map(|(_, miner)| Arc::clone(miner)).collect()).unwrap_or_default();
    for miner in residents {
        // `PomMiner::mine_v4` finishes/reads its in-order queue before releasing this mutex.
        // Taking it once therefore drains a command submitted just before the pause was raised.
        match miner.lock() {
            Ok(completed) => drop(completed),
            Err(_) => {
                log::error!("PoM[opencl]: resident miner mutex poisoned — refusing GPU inference");
                return None; // guard clears the pause count
            }
        }
    }
    Some(guard)
}

thread_local! {
    /// The cl_device_id this mining thread owns (set once via bind_thread_device). All PoM
    /// driver calls on this thread act on this card.
    static BOUND_DEV: Cell<Option<usize>> = const { Cell::new(None) };
}

/// OpenCL device ids already reported to the process-wide `--wait-ready` latch. The latch itself
/// is idempotent, but suppressing duplicate reports avoids printing the same ready line on every
/// `is_installed()` poll during the ten-second worker-registration grace period.
static READY_REPORTED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Bind the calling mining thread to its GPU (cl_device_id as usize). Called once per GPU thread
/// before the PoM loop so every card mines its own shares. AMD-only; NVIDIA's seam is untouched.
pub fn bind_thread_device(device_id: usize) {
    BOUND_DEV.with(|d| d.set(Some(device_id)));
}

fn bound_dev() -> Option<usize> {
    BOUND_DEV.with(|d| d.get())
}

fn ready_key_for_bound(bound: Option<usize>, resolved_device: usize) -> Option<u64> {
    (bound == Some(resolved_device)).then_some(resolved_device as u64)
}

/// Report readiness only for the production worker actually bound to `device_id`. Direct test and
/// legacy callers may install through the unbound fallback; they never registered a wait-ready
/// worker and therefore must not create a misleading readiness record.
fn mark_bound_worker_ready(device_id: usize) {
    let Some(key) = ready_key_for_bound(bound_dev(), device_id) else {
        return;
    };
    let first_report = {
        let mut reported = READY_REPORTED.lock().unwrap_or_else(|p| p.into_inner());
        if reported.contains(&device_id) {
            false
        } else {
            reported.push(device_id);
            true
        }
    };
    if first_report {
        crate::wait_ready::mark_ready(key);
    }
}

/// The cl_device_id this call targets: the thread's bound card, else the first selected OpenCL GPU
/// (single-device fallback / tests, which never call bind_thread_device).
fn target_dev() -> Option<usize> {
    bound_dev().or_else(|| scoped_gpu_devices().first().copied())
}

fn miner_for(device_id: usize) -> Option<Arc<Mutex<PomMiner>>> {
    POM_BY_DEV.lock().unwrap().iter().find(|(d, _)| *d == device_id).map(|(_, m)| m.clone())
}

fn words(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for i in 0..4 {
        w[i] = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// The cl_device_id whose card hosts the in-process llama engine (zero-dup): that card gets NO
/// OpenCL blob — its mine() routes over the engine's resident weights instead. Claimed once by
/// the card whose PCI location matches the engine's, after the startup byte gate passes.
static SHARED_DEV: Mutex<Option<usize>> = Mutex::new(None);
/// Card DEDICATED to OPoI inference (engine-hosted model; model + blob can't share its VRAM):
/// it deliberately mines nothing — no blob install, no walk. The other cards mine at full rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DedicationOwner {
    None,
    Provisional,
    InProcess,
    Server,
}

#[derive(Clone, Copy, Debug)]
struct DedicationState {
    device: Option<usize>,
    owner: DedicationOwner,
}

static DEDICATED_DEV: Mutex<DedicationState> =
    Mutex::new(DedicationState { device: None, owner: DedicationOwner::None });
static DEDICATION_REQUIRED: AtomicBool = AtomicBool::new(false);

fn clear_dedicated_device() {
    *DEDICATED_DEV.lock().unwrap_or_else(|p| p.into_inner()) =
        DedicationState { device: None, owner: DedicationOwner::None };
}

fn clear_dedication_owned_by(owner: DedicationOwner) {
    let mut state = DEDICATED_DEV.lock().unwrap_or_else(|p| p.into_inner());
    if state.owner == owner {
        *state = DedicationState { device: None, owner: DedicationOwner::None };
    }
}

fn dedicated_device() -> Option<usize> {
    DEDICATED_DEV.lock().unwrap_or_else(|p| p.into_inner()).device
}

fn explicit_vulkan_device() -> Option<i32> {
    std::env::var("KERYX_LLAMA_VK_DEVICE")
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .filter(|device| *device >= 0)
}

fn explicit_vulkan_device_override() -> bool {
    explicit_vulkan_device().is_some()
}

#[cfg(unix)]
#[derive(Debug, PartialEq, Eq)]
enum VulkanDeviceScope {
    SelectedWorker(usize),
    ExplicitExternal,
    RejectAutoExternal,
}

#[cfg(unix)]
fn classify_vulkan_device_scope(matched_worker: Option<usize>, explicit_override: bool) -> VulkanDeviceScope {
    match (matched_worker, explicit_override) {
        (Some(worker), _) => VulkanDeviceScope::SelectedWorker(worker),
        (None, true) => VulkanDeviceScope::ExplicitExternal,
        (None, false) => VulkanDeviceScope::RejectAutoExternal,
    }
}

/// Reserve the largest OpenCL card provisionally before capability proof opens the mining gate.
/// Vulkan preflight replaces this with the exact full-PCI target and frees any resident walk
/// before model allocation; failed selection releases the provisional reservation.
pub fn require_dedicated_inference_card() {
    DEDICATION_REQUIRED.store(true, Ordering::Release);
    let largest = scoped_gpu_devices().into_iter().max_by_key(|id| {
        opencl3::device::Device::new(*id as opencl3::types::cl_device_id).global_mem_size().unwrap_or(0)
    });
    if let Some(id) = largest {
        *DEDICATED_DEV.lock().unwrap_or_else(|p| p.into_inner()) =
            DedicationState { device: Some(id), owner: DedicationOwner::Provisional };
    }
}

/// Drop a selected worker's resident OpenCL blob before loading a non-coexistent Vulkan model on
/// that same physical GPU. The caller must already hold the global inference pause: that prevents
/// new installs and makes stale per-batch Arc clones drain quickly. Merely changing
/// `DEDICATED_DEV` is not sufficient because an already-created `cl_mem` allocation would continue
/// occupying VRAM and make the subsequent llama model load OOM.
fn evict_resident_for_dedication(device_id: usize) -> Result<(), String> {
    if INFERENCE_PAUSE_COUNT.load(Ordering::Acquire) == 0 {
        return Err("OpenCL walks are not paused; refusing to evict live mining VRAM".into());
    }
    let resident = {
        let mut miners = POM_BY_DEV.lock().unwrap_or_else(|p| p.into_inner());
        miners.iter().position(|(id, _)| *id == device_id).map(|pos| miners.swap_remove(pos).1)
    };
    let Some(mut resident) = resident else {
        return Ok(());
    };

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match Arc::try_unwrap(resident) {
            Ok(miner) => {
                // pause_and_drain_for_inference already completed its queue. Dropping the sole
                // remaining owner now releases every cl_mem before Vulkan starts allocating.
                drop(miner.into_inner().unwrap_or_else(|p| p.into_inner()));
                log::info!(
                    "PoM: released the resident OpenCL walk on card {device_id:#x} before GPU inference model load"
                );
                return Ok(());
            }
            Err(still_shared) => {
                resident = still_shared;
                if std::time::Instant::now() >= deadline {
                    let mut miners = POM_BY_DEV.lock().unwrap_or_else(|p| p.into_inner());
                    if !miners.iter().any(|(id, _)| *id == device_id) {
                        miners.push((device_id, resident));
                    }
                    return Err("timed out waiting for stale mining references to release".into());
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}

#[cfg(unix)]
fn prepare_vulkan_device(device: i32, owner: DedicationOwner, route: &str) -> bool {
    let dedication_required = DEDICATION_REQUIRED.load(Ordering::Acquire);
    let explicit_override = explicit_vulkan_device() == Some(device);
    if explicit_override && !dedication_required {
        // Coexistence was proven by the memory planner. An explicit operator pin may target an
        // inference-only card whose PCI identity the bundled (possibly older) sidecar cannot map.
        return true;
    }
    let Some(pci) = crate::llama_engine_vk::ggml_device_pci(device) else {
        if dedication_required {
            clear_dedicated_device();
        }
        log::error!(
            "PoM: cannot map {route} ggml device {device} to full PCI identity; refusing GPU \
             inference before model allocation"
        );
        return false;
    };
    let matched = scoped_gpu_devices().into_iter().find(|id| device_pci_full(*id) == Some(pci));
    match classify_vulkan_device_scope(matched, explicit_override) {
        VulkanDeviceScope::SelectedWorker(id) => {
            if dedication_required {
                *DEDICATED_DEV.lock().unwrap_or_else(|p| p.into_inner()) = DedicationState { device: Some(id), owner };
                if let Err(error) = evict_resident_for_dedication(id) {
                    clear_dedication_owned_by(owner);
                    log::error!(
                        "PoM: cannot dedicate selected OpenCL card {id:#x} to {route}: {error}; \
                         refusing model load"
                    );
                    return false;
                }
                log::warn!(
                    "PoM: selected OpenCL card {id:#x} is reserved for {route} before model load \
                     because model + walk cannot coexist"
                );
            }
            true
        }
        VulkanDeviceScope::ExplicitExternal => {
            // Explicit authority may select a non-mining inference card. Release the provisional
            // largest-worker reservation before loading so every selected worker remains active.
            if dedication_required {
                *DEDICATED_DEV.lock().unwrap_or_else(|p| p.into_inner()) = DedicationState { device: None, owner };
            }
            log::warn!(
                "PoM: explicitly pinned {route} GPU {:04x}:{:02x}:{:02x}.{:x} is outside the \
                 selected OpenCL mining workers; no mining card is reserved",
                pci.0,
                pci.1,
                pci.2,
                pci.3,
            );
            true
        }
        VulkanDeviceScope::RejectAutoExternal => {
            if dedication_required {
                clear_dedicated_device();
            }
            log::error!(
                "PoM: auto-selected {route} GPU {:04x}:{:02x}:{:02x}.{:x} is outside the exact \
                 OpenCL worker allowlist; refusing model load",
                pci.0,
                pci.1,
                pci.2,
                pci.3,
            );
            false
        }
    }
}

/// Resolve and reserve the exact in-process Vulkan target before llama.cpp allocates its model.
#[cfg(unix)]
pub fn prepare_inprocess_vulkan_device(device: i32) -> bool {
    prepare_vulkan_device(device, DedicationOwner::InProcess, "in-process Vulkan inference")
}

/// Resolve and reserve the exact llama-server Vulkan target before spawning the subprocess.
#[cfg(unix)]
pub fn prepare_vulkan_server_device(device: i32) -> bool {
    prepare_vulkan_device(device, DedicationOwner::Server, "Vulkan llama-server")
}

#[cfg(unix)]
pub fn release_inprocess_vulkan_dedication() {
    clear_dedication_owned_by(DedicationOwner::InProcess);
}

#[cfg(unix)]
pub fn release_vulkan_server_dedication() {
    clear_dedication_owned_by(DedicationOwner::Server);
}

/// A sidecar/device-selection failure before a route owns the reservation must not leave the
/// planner's provisional largest worker idle forever.
#[cfg(unix)]
pub fn release_provisional_vulkan_dedication() {
    clear_dedication_owned_by(DedicationOwner::Provisional);
}

#[cfg(unix)]
pub fn dedicate_loaded_engine_card_if_required() -> bool {
    let dedication_required = DEDICATION_REQUIRED.load(Ordering::Acquire);
    let explicit_override = explicit_vulkan_device_override();
    let Some((domain, bus, device, function)) = crate::llama_engine_vk::pom_pci() else {
        if !dedication_required && explicit_override {
            // No card needs reserving, and an explicit ggml device is operator authority. The PCI
            // identity is only mandatory when we must decide which mining worker to suppress.
            return true;
        }
        if dedication_required {
            clear_dedicated_device();
        }
        log::error!("PoM: cannot identify the loaded inference GPU; refusing to advertise a non-coexistent route");
        return false;
    };
    let matched = scoped_gpu_devices()
        .into_iter()
        .find(|id| device_pci_full(*id).map(|pci| pci == (domain, bus, device, function)).unwrap_or(false));
    if let Some(id) = matched {
        if dedication_required {
            let state = *DEDICATED_DEV.lock().unwrap_or_else(|p| p.into_inner());
            if state.owner != DedicationOwner::InProcess || state.device != Some(id) {
                log::error!(
                    "PoM: loaded inference engine landed on OpenCL card {id:#x}, but that exact \
                     worker was not reserved before model allocation; route refused"
                );
                clear_dedicated_device();
                return false;
            }
            log::warn!("PoM: OpenCL card {:#x} is dedicated to GPU inference because model + walk cannot coexist", id);
        }
        true
    } else if explicit_override {
        // The operator deliberately chose an inference-only GPU outside the OpenCL mining subset.
        // It needs no mining reservation; release the provisional largest-worker reservation and
        // accept the route. Auto-placement never gets this exception.
        if dedication_required {
            clear_dedicated_device();
        }
        log::warn!(
            "PoM: explicitly pinned Vulkan inference GPU {domain:04x}:{bus:02x}:{device:02x}.{function:x} is \
             outside the selected OpenCL mining workers; no mining card will be dedicated."
        );
        true
    } else {
        if dedication_required {
            clear_dedicated_device();
        }
        log::error!(
            "PoM: automatically loaded inference GPU {domain:04x}:{bus:02x}:{device:02x}.{function:x} is outside \
             the selected OpenCL workers; route remains withdrawn and no mining worker will be \
             suppressed"
        );
        false
    }
}

/// Revalidate a successfully serving Vulkan llama-server against the same exact selection used by
/// its mandatory pre-spawn reservation. The post-generation check prevents a route proof if device
/// identity somehow changed; it is not relied on to make VRAM available for model loading.
#[cfg(unix)]
pub fn resolve_vulkan_server_dedication(device: i32) -> bool {
    prepare_vulkan_server_device(device)
}

/// Call only after an inference server whose card could not be mapped has been stopped. No GPU
/// inference remains to protect, so keeping a provisional mining worker idle would be a leak.
#[cfg(unix)]
pub fn release_provisional_dedication_after_server_stop() {
    release_vulkan_server_dedication();
    release_provisional_vulkan_dedication();
}

#[cfg(not(unix))]
pub fn dedicate_loaded_engine_card_if_required() -> bool {
    !DEDICATION_REQUIRED.load(Ordering::Acquire)
}

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
/// plugin's "#N" and therefore indexes the exact selected worker list (including reordered/subset
/// `--opencl-device` choices), so #N here == #N in the hashrate line. No rocm-smi /
/// NVML dependency; every field is best-effort → None (so a rig without amdgpu sysfs just shows no
/// efficiency, never a wrong number). Power is what the MH/s/W figure needs.
#[cfg(unix)]
pub fn amd_health(idx: usize) -> Option<AmdHealth> {
    use opencl3::platform::get_platforms;
    let dev = match configured_worker_devices() {
        Some(devs) => *devs.get(idx)?,
        None => {
            // Legacy direct-library fallback before the host publishes plugin selection.
            let plats = get_platforms().ok()?;
            let amd = plats.into_iter().find(|p| {
                p.vendor().map(|v| v == "Advanced Micro Devices, Inc.").unwrap_or(false)
                    && !p.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU).unwrap_or_default().is_empty()
            })?;
            (*amd.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU).ok()?.get(idx)?) as usize
        }
    };
    let (domain, b, d, f) = device_pci_full(dev)?;
    let want = format!("{domain:04x}:{b:02x}:{d:02x}.{f:x}");
    // Find the DRM card whose PCI address matches, then read its hwmon + amdgpu sysfs nodes.
    let read_u64 =
        |path: &std::path::Path| -> Option<u64> { std::fs::read_to_string(path).ok()?.trim().parse::<u64>().ok() };
    for entry in std::fs::read_dir("/sys/class/drm").ok()? {
        let card = entry.ok()?.path();
        if !card
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("card") && !n.contains('-'))
            .unwrap_or(false)
        {
            continue;
        }
        let devdir = card.join("device");
        let pci =
            std::fs::canonicalize(&devdir).ok().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
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
                if let Some(mhz) =
                    line.split_whitespace().find_map(|t| t.strip_suffix("Mhz").and_then(|n| n.parse::<u32>().ok()))
                {
                    h.core_mhz = Some(mhz);
                }
            }
        }
        return Some(h);
    }
    None
}
#[cfg(not(unix))]
pub fn amd_health(_idx: usize) -> Option<AmdHealth> {
    None
}

/// This card's full PCI identity. Prefer standard `cl_khr_pci_bus_info` (includes domain), then
/// fall back to AMD's legacy topology query (which exposes only bus/device/function, domain 0).
#[cfg(unix)]
fn device_pci_full(device_id: usize) -> Option<(u32, u32, u32, u32)> {
    if let Ok(v) = opencl3::device::get_device_data(
        device_id as opencl3::types::cl_device_id,
        opencl3::device::CL_DEVICE_PCI_BUS_INFO_KHR,
    ) {
        if v.len() >= 16 {
            return Some((
                u32::from_ne_bytes(v[0..4].try_into().ok()?),
                u32::from_ne_bytes(v[4..8].try_into().ok()?),
                u32::from_ne_bytes(v[8..12].try_into().ok()?),
                u32::from_ne_bytes(v[12..16].try_into().ok()?),
            ));
        }
    }
    const CL_DEVICE_TOPOLOGY_AMD: opencl3::types::cl_device_info = 0x4037;
    const CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD: u32 = 1;
    let v = opencl3::device::get_device_data(device_id as opencl3::types::cl_device_id, CL_DEVICE_TOPOLOGY_AMD).ok()?;
    if v.len() < 24 || u32::from_le_bytes(v[0..4].try_into().ok()?) != CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD {
        return None;
    }
    Some((0, v[21] as u32, v[22] as u32, v[23] as u32))
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
    let name = opencl3::device::Device::new(device_id as opencl3::types::cl_device_id).name().unwrap_or_default();
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
    matches!(
        (crate::llama_engine_vk::pom_pci(), device_pci_full(device_id)),
        (Some(engine), Some(opencl)) if engine == opencl
    )
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
            mark_bound_worker_ready(device_id);
            return Ok(());
        }
        // This read occurs while holding the same registry lock the drain takes after raising its
        // counter, so install and generation have a total order even before a resident exists.
        if INFERENCE_PAUSE_COUNT.load(Ordering::Acquire) != 0 {
            return Err("GPU inference owns the card; resident install deferred".into());
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
        // Publication is the readiness boundary: the bound worker can now enter mine_v4 as soon
        // as the all-card latch opens. Never report before the JIT/upload and registry insertion.
        mark_bound_worker_ready(device_id);
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
    if let (Some(id), Some(d)) = (bound_dev(), dedicated_device()) {
        if id == d {
            // A deliberately inference-only worker is resolved even though it has no walk buffer.
            mark_bound_worker_ready(id);
            return true;
        }
    }
    match bound_dev() {
        Some(id) => {
            let ready = miner_for(id).is_some() || is_shared_dev(id);
            if ready {
                mark_bound_worker_ready(id);
            }
            ready
        }
        None => !POM_BY_DEV.lock().unwrap().is_empty() || SHARED_DEV.lock().unwrap().is_some(),
    }
}

/// LARGEST global memory (MB) across the selected OpenCL GPU workers — the up-front GPU-inference
/// feasibility check: if even the biggest selected card can't hold model + possession blob
/// together, loading the model onto any mining GPU is doomed (spill/unload later).
pub fn max_gpu_global_mem_mb() -> Option<u64> {
    scoped_gpu_devices()
        .into_iter()
        .filter_map(|id| opencl3::device::Device::new(id as opencl3::types::cl_device_id).global_mem_size().ok())
        .max()
        .map(|b| b / (1024 * 1024))
}

/// First selected worker GPU's total global memory in MB via OpenCL
/// (`CL_DEVICE_GLOBAL_MEM_SIZE`) — retained under the legacy `gpu0_*` name for API compatibility.
/// This is the AMD analog of nvidia-smi's `memory.total`, so tier auto-selection follows the
/// plugin's reordered/subset device list. None if there is no selected OpenCL GPU.
pub fn gpu0_global_mem_mb() -> Option<u64> {
    let id = *scoped_gpu_devices().first()?;
    let bytes = opencl3::device::Device::new(id as opencl3::types::cl_device_id).global_mem_size().ok()?;
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
    // tree in RAM so the post-hit proof build is a pure cached lookup instead of the now-optimized
    // sparse checkpoint/memo reconstruction. This still minimizes block-race latency, but costs
    // ~2N*32 B RAM (~12 GiB at the current tier-0), hence opt-in; pool miners don't need it and
    // low-RAM rigs keep the sparse path. Built ONCE on the shared index (before set_index), so all
    // cards share it. Byte-safe:
    // build_dense self-checks its dense root against the pinned R_T.
    if crate::pom::resident_tree_enabled() {
        let t0 = std::time::Instant::now();
        log::info!(
            "PoM: building RESIDENT Merkle tree (RAM) — proof build becomes lookup-time (~+{} MiB)…",
            (index.n_chunks * 64) / (1024 * 1024)
        );
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
    if INFERENCE_PAUSE_COUNT.load(Ordering::Acquire) != 0 {
        return;
    }
    if is_installed() {
        return; // this thread's card is already resident
    }
    // Attempt heartbeat: a card that repeatedly fails/loops here must SAY so — a silent 0-hash
    // card is undiagnosable from the field (the whole RX 580 / 5700 XT saga).
    {
        static ATTEMPTS: Mutex<Vec<(usize, u32)>> = Mutex::new(Vec::new());
        if let Some(id) = target_dev() {
            let mut a = ATTEMPTS.lock().unwrap();
            let e = if let Some(e) = a.iter_mut().find(|(d, _)| *d == id) {
                e
            } else {
                a.push((id, 0));
                a.last_mut().unwrap()
            };
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
            let _setup_guard = match GpuSetupGuard::begin(id) {
                Some(guard) => guard,
                None => return,
            };
            // Zero-dup first: when the in-process llama engine hosts the model on this very
            // card (PCI match + byte gate), walk its resident weights — skip the blob upload.
            if try_claim_shared(id) {
                mark_bound_worker_ready(id);
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
                        crate::llama_engine_vk::mark_gpu_inference_unfit();
                        // The route proof was earned by the engine we just destroyed. Auto Vulkan
                        // subprocess fallback is deliberately forbidden after this VRAM eviction,
                        // so retaining the cached pass would keep ai:cap/mining open with no live
                        // inference backend. Withdraw this exact model/route before the blob retry;
                        // the durable warmer may only reopen it after a fresh generation succeeds.
                        crate::slm::withdraw_model_after_amd_engine_eviction(&model_id);
                        match install_resident(id) {
                            Ok(()) => log::info!(
                                "PoM: tier {t} installed (GPU-resident) on card {id:#x} after engine unload."
                            ),
                            Err(e2) => {
                                log::warn!("PoM: install on card {id:#x} STILL failed after engine unload: {e2}")
                            }
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
pub fn mine_v4(
    pph: &[u8; 32],
    time: u64,
    target_le: &[u8; 32],
    nonce_base: u64,
    batch: u64,
    daa: u64,
) -> crate::pom::GrindResult {
    if INFERENCE_PAUSE_COUNT.load(Ordering::Acquire) != 0 {
        return Err(crate::pom::GrindError::Paused("OpenCL walk paused for GPU inference"));
    }
    let Some(id) = target_dev() else {
        return Err(crate::pom::GrindError::Paused("no OpenCL GPU bound"));
    };
    // H10 hardfork (DAA 87,360,000): at/after the gate the SEED becomes the one-way cSHAKE256
    // PowHash of the RAW pph words; below it, the reversible v4 fold over the v4-salted words.
    // Byte-identical dispatch to pom::pom_block_seed_v4_era. The POW fold (p) is unchanged (H3 words).
    let h10_era = daa >= crate::pom::h10_activation_daa();
    let p = crate::pom::pph_words_for_era(pph, true);
    let s = if h10_era { crate::pom::pph_words(pph) } else { crate::pom::pph_words_v4(pph) };
    let t = words(target_le);
    let Some(miner) = miner_for(id) else {
        return Err(crate::pom::GrindError::Paused("OpenCL walk not installed"));
    };
    let Ok(mut g) = miner.lock() else {
        log::error!("PoM[opencl]: resident miner mutex poisoned — batch aborted and not counted");
        return Err(crate::pom::GrindError::Backend("resident OpenCL miner mutex poisoned".into()));
    };
    // Close the read-flag / lock race with `pause_and_drain_for_inference`: a call that read false
    // just before the pause was raised may acquire this mutex only after the drain did.
    if INFERENCE_PAUSE_COUNT.load(Ordering::Acquire) != 0 {
        return Err(crate::pom::GrindError::Paused("OpenCL walk paused for GPU inference"));
    }
    match g.mine_v4(p, s, time, t, nonce_base, batch, if h10_era { 1 } else { 0 }) {
        Ok(winner) => Ok(crate::pom::GrindCompleted { winner, hashes_done: batch }),
        Err(e) => {
            log::warn!("PoM[opencl]: v4 batch failed ({e}) — batch aborted and not counted");
            Err(crate::pom::GrindError::Backend(e))
        }
    }
}

/// Build the resident tier from a GGUF (shared proof WeightIndex, streamed to VRAM) and make it
/// resident on the FIRST OpenCL GPU. The multi-GPU production path uses ensure_installed (one
/// resident copy per card); this single-device form backs the tests + any non-bound caller.
pub fn load_tier(gguf_path: &str, tier: u8) -> Result<(), String> {
    // Non-bound/test entry: no model id at hand, so the cache is bound to the zero id and any
    // production (set_mining_tier) tree for a real model is treated as foreign and rebuilt.
    ensure_index([0u8; 32], gguf_path, tier)?;
    let id = target_dev().ok_or("PoM: no selected OpenCL GPU device")?;
    install_resident(id)
}

fn hex32(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

#[cfg(test)]
mod v4_wmma1_tests {
    use super::{device_arch_is, ready_key_for_bound, select_v4_mode, V4Mode, POM_SRC};

    #[test]
    fn one_state_default_is_limited_to_exact_gfx1100() {
        assert!(device_arch_is("gfx1100", "gfx1100"));
        assert!(device_arch_is("AMD Radeon (gfx1100:sramecc-)", "gfx1100"));
        assert!(!device_arch_is("gfx1102", "gfx1100"));
        assert!(!device_arch_is("gfx11000", "gfx1100"));

        assert!(matches!(select_v4_mode(None, true, true, true, true), V4Mode::SingleWmma1));
        assert!(matches!(select_v4_mode(None, true, true, true, false), V4Mode::SingleWmma));
        assert!(matches!(select_v4_mode(Some("wmma1"), true, true, true, false), V4Mode::SingleWmma1));
        assert!(matches!(select_v4_mode(Some("wmma"), true, true, true, true), V4Mode::SingleWmma));
        assert!(matches!(select_v4_mode(Some("wmma1"), true, false, true, true), V4Mode::SingleWmma));
        assert!(matches!(select_v4_mode(None, false, true, false, false), V4Mode::SingleDp4a));
    }

    #[test]
    fn one_state_kernel_has_minimum_lds_and_preloads_before_stores() {
        assert!(POM_SRC.contains("#define V4W1_STRIP_U32 512"));
        let kernel = POM_SRC.split_once("void pom_mine_v4_wmma_1state(").expect("one-state WMMA kernel").1;
        assert!(kernel.contains("__local uint* state = strip;"));
        assert!(kernel.contains("__local uint* tile  = strip + 256;"));
        assert!(!kernel.contains("__local uint* sB"));
        assert!(!kernel.contains("snxt"));

        let a0 = kernel.find("const int4v a0 =").expect("first A fragment preload");
        let a1 = kernel.find("const int4v a1 =").expect("second A fragment preload");
        let jb = kernel.find("for (uint jb = 0; jb < 2; jb++)").expect("column-block loop");
        let store = kernel.find("Sc[x * 32 + j] =").expect("in-place output store");
        assert!(a0 < jb && a1 < jb && jb < store);
    }

    #[test]
    fn wait_ready_key_requires_the_exact_bound_opencl_worker() {
        let id = 0x1234_5678usize;
        assert_eq!(ready_key_for_bound(Some(id), id), Some(id as u64));
        assert_eq!(ready_key_for_bound(Some(id + 1), id), None);
        assert_eq!(ready_key_for_bound(None, id), None);
    }
}

#[cfg(all(test, unix))]
mod device_scope_tests {
    use super::{
        classify_vulkan_device_scope, format_pci_location, normalize_device_ids, resolve_device_scope,
        VulkanDeviceScope,
    };

    #[test]
    fn configured_subset_and_order_are_authoritative() {
        let selected = vec![0x30, 0x10, 0x30, 0x20];
        assert_eq!(normalize_device_ids(selected), vec![0x30, 0x10, 0x20]);
        assert_eq!(resolve_device_scope(Some(vec![0x30, 0x10]), || panic!("must not enumerate")), vec![0x30, 0x10]);
    }

    #[test]
    fn configured_empty_scope_does_not_escape_to_global_devices() {
        assert!(resolve_device_scope(Some(Vec::new()), || panic!("must not enumerate")).is_empty());
        assert_eq!(resolve_device_scope(None, || vec![7, 9]), vec![7, 9]);
    }

    #[test]
    fn pci_allowlist_identity_includes_domain() {
        assert_eq!(format_pci_location((0x1234, 0xab, 0x1c, 2)), "1234:ab:1c.2");
    }

    #[test]
    fn server_scope_preserves_explicit_and_auto_contracts() {
        assert_eq!(
            classify_vulkan_device_scope(Some(0x22), true),
            VulkanDeviceScope::SelectedWorker(0x22),
            "an explicit selected non-largest worker must be dedicated exactly"
        );
        assert_eq!(classify_vulkan_device_scope(None, true), VulkanDeviceScope::ExplicitExternal);
        assert_eq!(classify_vulkan_device_scope(None, false), VulkanDeviceScope::RejectAutoExternal);
    }
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

    fn selected_h10_eras(value: Option<&str>, both_by_default: bool) -> Result<Vec<bool>, String> {
        match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            None if both_by_default => Ok(vec![false, true]),
            None => Ok(vec![false]),
            Some("both") => Ok(vec![false, true]),
            Some("0" | "pre" | "pre-h10") => Ok(vec![false]),
            Some("1" | "h10") => Ok(vec![true]),
            Some(value) => Err(format!("invalid H10 era selector {value:?}; expected 0/pre, 1/h10, or both")),
        }
    }

    // CPU reference pow-value (256-bit LE) for one nonce over `index`. v4 seed = era-selected
    // (h10=false → reversible v4 fold; h10=true → one-way cSHAKE256 PowHash). Pow fold uses the
    // H3-salted words ("v4 pow uses the h3 fold").
    fn cpu_pow(index: &crate::pom::WeightIndex, pph: &[u8; 32], time: u64, nonce: u64, h10: bool) -> [u8; 32] {
        let seed = crate::pom::pom_block_seed_v4_era(pph, time, nonce, h10);
        let (_proof, fin) = crate::pom_v4::build_proof_v4(0, seed, index).unwrap();
        crate::pom::pom_pow_value(fin, pph, true)
    }

    #[test]
    fn gpu_v4_matches_cpu_reference() {
        // First AMD OpenCL GPU, or skip. Do not run an AMD-kernel test on a co-resident NVIDIA
        // production device merely because its ICD happens to enumerate first.
        let dev_ids = match opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU) {
            Ok(v) if !v.is_empty() => v,
            _ => {
                eprintln!("no OpenCL GPU — skipping v4 byte-exact test");
                return;
            }
        };
        let Some(dev) = dev_ids.into_iter().map(opencl3::device::Device::new).find(|dev| {
            dev.vendor()
                .map(|vendor| {
                    let vendor = vendor.to_ascii_lowercase();
                    vendor.contains("advanced micro devices") || vendor.trim() == "amd"
                })
                .unwrap_or(false)
        }) else {
            eprintln!("no AMD OpenCL GPU — skipping v4 byte-exact test");
            return;
        };
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

        let mut miner = PomMiner::new(dev, &index, n_chunks).expect("PomMiner::new");
        let modes: &[(V4Mode, &str)] = &[
            (V4Mode::SingleDp4a, "single dp4a"),
            (V4Mode::TwoPhaseDp4a, "two-phase dp4a"),
            (V4Mode::SingleWmma, "single WMMA"),
            (V4Mode::SingleWmma1, "single WMMA one-state"),
            (V4Mode::TwoPhaseWmma, "two-phase WMMA"),
        ];
        // Validate BOTH consensus eras by default: h10=false (reversible v4 fold) AND h10=true
        // (H10 one-way cSHAKE256 seed — the v0.12.0 hardfork). KERYX_TEST_H10=0/1/both selects an
        // era for isolated device runs. A keccak/absorb bug shows up as an argmin mismatch.
        let eras = selected_h10_eras(std::env::var("KERYX_TEST_H10").ok().as_deref(), true).expect("KERYX_TEST_H10");
        for &h10 in &eras {
            // CPU argmin pow over 0..NN with the era-correct seed; target = that min.
            let mut best = ([0xFFu8; 32], u64::MAX);
            for nonce in 0..NN {
                let pv = cpu_pow(&index, &pph, time, nonce, h10);
                let le_le = |a: &[u8; 32], b: &[u8; 32]| {
                    for k in (0..4).rev() {
                        let (wa, wb) = (
                            u64::from_le_bytes(a[k * 8..k * 8 + 8].try_into().unwrap()),
                            u64::from_le_bytes(b[k * 8..k * 8 + 8].try_into().unwrap()),
                        );
                        if wa != wb {
                            return wa < wb;
                        }
                    }
                    true
                };
                if best.1 == u64::MAX || le_le(&pv, &best.0) {
                    best = (pv, nonce);
                }
            }
            let (target, w_cpu) = best;
            eprintln!("[h10={h10}] cpu argmin nonce = {w_cpu}");

            let p = crate::pom::pph_words_for_era(&pph, true);
            // Seed words match the era: raw pph for H10 (keccak absorbs them), v4-salted otherwise.
            let s = if h10 { crate::pom::pph_words(&pph) } else { crate::pom::pph_words_v4(&pph) };
            let t = words(&target);
            let hf: u32 = if h10 { 1 } else { 0 };
            for &(m, name) in modes {
                if matches!(m, V4Mode::SingleWmma) && miner.kernel_wmma_sp.is_none() {
                    continue;
                }
                if matches!(m, V4Mode::SingleWmma1) && miner.kernel_wmma_1state.is_none() {
                    continue;
                }
                if matches!(m, V4Mode::TwoPhaseWmma) && miner.kernel_wmma.is_none() {
                    continue;
                }
                miner.mode = m;
                assert_eq!(
                    miner.mine_v4(p, s, time, t, 0, NN, hf).expect("OpenCL v4 grind"),
                    Some(w_cpu),
                    "{name} v4 winner mismatch (h10={h10})"
                );
                eprintln!("[h10={h10}] {name}: OK");
            }
        }
    }

    /// Throughput A/B: single-phase vs two-phase v4 on a memory-bound synthetic blob. Ignored
    /// (needs a GPU + is slow). Run: cargo test --release -p keryx-miner-supr --features pom-opencl
    /// --lib v4_two_phase_bench -- --ignored --nocapture. KERYX_BENCH_CL_DEV picks the device;
    /// KERYX_BENCH_H10=0/1/both picks the era (default pre-H10, preserving the historical bench).
    #[test]
    #[ignore]
    fn v4_two_phase_bench() {
        let dev_ids = match opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU) {
            Ok(v) if !v.is_empty() => v,
            _ => {
                eprintln!("no OpenCL GPU — skipping");
                return;
            }
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
        let t = [0u64; 4]; // impossible target → full grind, no early winner
        let batch: u64 = std::env::var("KERYX_BENCH_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 16);

        let mut miner = PomMiner::new(dev, &index, n_chunks).expect("PomMiner::new");
        let bench = |m: &mut PomMiner, label: &str, s: [u64; 4], h10: u32| {
            let _ = m.mine_v4(p, s, 1_700_000_000, t, 0, batch, h10); // warmup
            let rounds = 6u64;
            let start = std::time::Instant::now();
            for r in 0..rounds {
                let _ = m.mine_v4(p, s, 1_700_000_000, t, r * batch, batch, h10);
            }
            let secs = start.elapsed().as_secs_f64();
            let mhs = (rounds * batch) as f64 / secs / 1e6;
            eprintln!("{label}: {mhs:.3} Mh/s ({} nonces in {secs:.2}s)", rounds * batch);
            mhs
        };
        let eras = selected_h10_eras(std::env::var("KERYX_BENCH_H10").ok().as_deref(), false).expect("KERYX_BENCH_H10");
        for h10 in eras {
            let era = if h10 { "h10" } else { "pre-h10" };
            let s = if h10 { crate::pom::pph_words(&pph) } else { crate::pom::pph_words_v4(&pph) };
            let hf = u32::from(h10);
            eprintln!("benchmark era: {era}");
            miner.mode = V4Mode::SingleDp4a;
            let single = bench(&mut miner, &format!("[{era}] single-phase dp4a "), s, hf);
            miner.mode = V4Mode::TwoPhaseDp4a;
            let _ = bench(&mut miner, &format!("[{era}] two-phase dp4a    "), s, hf);
            if miner.kernel_wmma_sp.is_some() {
                miner.mode = V4Mode::SingleWmma;
                let spw = bench(&mut miner, &format!("[{era}] single-phase WMMA "), s, hf);
                eprintln!(">> [{era}] single-phase WMMA vs single-phase dp4a: {:+.1}%", (spw / single - 1.0) * 100.0);
                if miner.kernel_wmma_1state.is_some() {
                    miner.mode = V4Mode::SingleWmma1;
                    let w1 = bench(&mut miner, &format!("[{era}] one-state WMMA    "), s, hf);
                    eprintln!(
                        ">> [{era}] one-state WMMA vs proven single-phase WMMA: {:+.1}%",
                        (w1 / spw - 1.0) * 100.0
                    );
                }
            }
            if miner.kernel_wmma.is_some() {
                miner.mode = V4Mode::TwoPhaseWmma;
                let tpw = bench(&mut miner, &format!("[{era}] two-phase WMMA    "), s, hf);
                eprintln!(">> [{era}] two-phase WMMA vs single-phase dp4a: {:+.1}%", (tpw / single - 1.0) * 100.0);
            }
        }
    }

    #[test]
    fn h10_era_selector_is_explicit_and_stable() {
        assert_eq!(selected_h10_eras(None, true).unwrap(), vec![false, true]);
        assert_eq!(selected_h10_eras(None, false).unwrap(), vec![false]);
        assert_eq!(selected_h10_eras(Some("pre-h10"), true).unwrap(), vec![false]);
        assert_eq!(selected_h10_eras(Some("h10"), false).unwrap(), vec![true]);
        assert_eq!(selected_h10_eras(Some("both"), false).unwrap(), vec![false, true]);
        assert!(selected_h10_eras(Some("yes"), true).is_err());
    }
}
