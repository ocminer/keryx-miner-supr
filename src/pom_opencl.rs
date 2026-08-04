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
        Ok(Self { _context: context, queue, kernel: chosen, weights, slab_shift, winner, n_chunks, ilp: best_ilp, local: best_local })
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

fn is_shared_dev(device_id: usize) -> bool {
    SHARED_DEV.lock().unwrap().map_or(false, |d| d == device_id)
}

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

/// True if `device_id` is an RDNA (gfx10/11/12) card — the Vulkan zero-dup walk matches or beats
/// the OpenCL blob walk there (measured +1.7% on gfx1102). On GCN (gfx9 and older) the Vulkan
/// walk is ~15-24% slower at equal clocks (a RADV-vs-ROCm memory-path gap the shader can't close),
/// so those cards keep their full-speed OpenCL blob by default. gfx name via CL_DEVICE_NAME.
#[cfg(unix)]
fn device_is_rdna(device_id: usize) -> bool {
    let name = opencl3::device::Device::new(device_id as opencl3::types::cl_device_id)
        .name()
        .unwrap_or_default();
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
#[cfg(unix)]
fn try_claim_shared(device_id: usize) -> bool {
    if is_shared_dev(device_id) {
        return true;
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
        log::info!(
            "PoM: card {device_id:#x} hosts the llama engine but is not RDNA — keeping its OpenCL blob \
             (full hashrate; set KERYX_ZERO_DUP=force to trade ~2.4 GB VRAM for it)."
        );
        return false;
    }
    let Some((idx, _)) = crate::pom::active_index() else { return false };
    if !crate::llama_engine_vk::pom_byte_gate(idx) {
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

/// True if THIS thread's bound card has the tier resident — either its own OpenCL blob or the
/// zero-dup engine walk (or, with no binding, any card does).
pub fn is_installed() -> bool {
    match bound_dev() {
        Some(id) => miner_for(id).is_some() || is_shared_dev(id),
        None => !POM_BY_DEV.lock().unwrap().is_empty() || SHARED_DEV.lock().unwrap().is_some(),
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
