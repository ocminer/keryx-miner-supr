//! Proof-of-Model GPU mining — Apple Silicon (Metal) backend.
//!
//! Parity path for `pom_gpu.rs` (CUDA): exposes the **identical** public free-function surface
//! (`install`/`uninstall`/`is_installed`/`is_loading`/`mine`/`set_mining_tier`/`ensure_installed`/
//! `current_tier`/`walk_devices`) so `main.rs`/`miner.rs`/`slm.rs` stay backend-agnostic — on macOS
//! `lib.rs` aliases this module to `pom_gpu`. Nothing here compiles off macOS.
//!
//! The walk runs the `metal/pom_mine.metal` compute kernel — whose seed/pow folds are byte-identical
//! to `pom::pom_block_seed`/`pom::pom_pow_value` and the CUDA `pom_mine.cu`, so a nonce found here
//! builds a `PomProof` the node accepts. Under the hood it is a **zero-dup** walk over candle's own
//! resident quantized `MTLBuffer`s (Apple unified memory): no packed weight blob, no host copy. Two
//! small side tables are built once at load:
//!   * `prefix` — cumulative 32-byte-chunk count per tensor in canonical (name-sorted) GGUF order,
//!     length `n_tensors + 1`. Same layout the node's `R_T` root is built over, so a global chunk
//!     index addresses the same bytes here and there.
//!   * `addrs`  — `MTLBuffer.gpuAddress()` per tensor. On Apple Silicon (always Metal-3 Tier-2
//!     argument buffers) these are plain 64-bit pointers the kernel reinterprets to
//!     `device const ulong*`. `use_resource` keeps each buffer GPU-resident across the dispatch.
//!
//! The candle-owned `MTLBuffer` is reached through the vendored `QTensor::metal_storage()` accessor
//! (see vendor/candle-core/src/quantized/mod.rs). Cloning a `Buffer` only bumps the objc2 retain
//! count — no bytes copied.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_metal_kernels::metal::{
    create_command_buffer, Buffer, CommandQueue, CommandSemaphore, ComputePipeline,
    Device as MtlDevice, MTLResourceOptions,
};
use objc2_metal::{
    MTLBuffer as _, MTLDevice as _, MTLResourceOptions as ObjcMTLResourceOptions, MTLResourceUsage,
    MTLSize,
};

const METAL_SRC: &str = include_str!("../metal/pom_mine.metal");
const CHUNK_BYTES: usize = 32;
const THREADGROUP_SIZE: usize = 256;

/// Shared-storage: CPU and GPU see the same unified-memory backing, so the host writes uniforms /
/// reads the winner with no blit — the same choice candle makes for its own transient buffers.
const SHARED_STORAGE: MTLResourceOptions =
    ObjcMTLResourceOptions(ObjcMTLResourceOptions::StorageModeShared.bits());

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

// Matches PomUniforms in metal/pom_mine.metal — field order and padding are load-bearing.
#[repr(C)]
struct Uniforms {
    n_total_chunks: u64,
    k_steps: u32,
    n_tensors: u32,
    p0: u64,
    p1: u64,
    p2: u64,
    p3: u64,
    time_: u64,
    t0: u64,
    t1: u64,
    t2: u64,
    t3: u64,
    nonce_base: u64,
    n_nonces: u32,
    _pad: u32,
}

pub struct PomGpuMiner {
    device: MtlDevice,
    queue: CommandQueue,
    pipeline: ComputePipeline,
    /// Debug pipeline: writes `final_state` per tid to a device buffer (test-only diagnostic).
    #[cfg(test)]
    debug_pipeline: ComputePipeline,
    /// Prefix sums of chunk counts across tensors, length `n_tensors + 1`, in chunks.
    prefix_buf: Buffer,
    /// GPU addresses of the resident per-tensor MTLBuffers, length `n_tensors`.
    addrs_buf: Buffer,
    /// Clones of the candle-owned per-tensor MTLBuffers: they hold the retain count that keeps the
    /// unified-memory backing alive, and are handed to `use_resource` on every dispatch so the
    /// driver marks them resident even though the kernel binds only the addrs/prefix tables.
    resources: Vec<Buffer>,
    n_total_chunks: u64,
    n_tensors: u32,
}

impl PomGpuMiner {
    /// Compile the kernel + upload the prefix/addrs side tables for an already-populated tensor set.
    /// Shared by `load` (real GGUF) and the byte-exact test (synthetic blob).
    fn build(
        mdev: MtlDevice,
        prefix: &[u64],
        addrs: &[u64],
        resources: Vec<Buffer>,
        n_total_chunks: u64,
    ) -> candle_core::Result<Self> {
        let n_tensors = resources.len() as u32;
        if n_total_chunks == 0 || n_tensors == 0 {
            return Err(candle_core::Error::Msg("PoM Metal: model produced 0 chunks".into()));
        }
        let prefix_buf = mdev
            .new_buffer_with_data(prefix.as_ptr() as *const _, std::mem::size_of_val(prefix), SHARED_STORAGE)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: prefix buffer: {e}")))?;
        let addrs_buf = mdev
            .new_buffer_with_data(addrs.as_ptr() as *const _, std::mem::size_of_val(addrs), SHARED_STORAGE)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: addrs buffer: {e}")))?;

        let library = mdev
            .new_library_with_source(METAL_SRC, None)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: compile: {e}")))?;
        let func = library
            .get_function("pom_mine", None)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: get_function: {e}")))?;
        let pipeline = mdev
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: pipeline: {e}")))?;
        #[cfg(test)]
        let debug_pipeline = {
            let f = library
                .get_function("pom_walk_states", None)
                .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: get_function debug: {e}")))?;
            mdev.new_compute_pipeline_state_with_function(&f)
                .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: debug pipeline: {e}")))?
        };
        let queue = mdev
            .new_command_queue()
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: command queue: {e}")))?;

        Ok(Self {
            device: mdev,
            queue,
            pipeline,
            #[cfg(test)]
            debug_pipeline,
            prefix_buf,
            addrs_buf,
            resources,
            n_total_chunks,
            n_tensors,
        })
    }

    /// Load the mining model's GGUF onto a candle Metal device and build the bindless walk tables.
    /// Heavy — call once per (device, model).
    pub fn load(gguf_path: &str, device_id: usize) -> candle_core::Result<Self> {
        let cdev = Device::new_metal(device_id)?;
        let mdev = match &cdev {
            Device::Metal(m) => m.metal_device().clone(),
            _ => return Err(candle_core::Error::Msg("PoM Metal: not a Metal device".into())),
        };

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical name-sorted order — matches pom-rt-builder / the node R_T

        let mut resources: Vec<Buffer> = Vec::with_capacity(names.len());
        let mut addrs: Vec<u64> = Vec::with_capacity(names.len());
        let mut prefix: Vec<u64> = Vec::with_capacity(names.len() + 1);
        prefix.push(0);
        let mut cum: u64 = 0;

        for name in &names {
            let qt = content.tensor(&mut file, name, &cdev)?;
            let n_bytes = qt.storage_size_in_bytes();
            if n_bytes < CHUNK_BYTES {
                // Skip tiny tensors (biases, norms) — same behaviour as the CUDA gather (chunks==0).
                continue;
            }
            let qmet = qt
                .metal_storage()
                .ok_or_else(|| candle_core::Error::Msg("PoM Metal: QTensor has no Metal storage".into()))?;
            let buf: Buffer = qmet.buffer().clone();
            let addr = buf.as_ref().gpuAddress();
            let n_chunks = (n_bytes / CHUNK_BYTES) as u64;
            cum += n_chunks;
            prefix.push(cum);
            addrs.push(addr);
            resources.push(buf);
        }
        let n_total_chunks = cum;
        let miner = Self::build(mdev, &prefix, &addrs, resources, n_total_chunks)?;
        info!(
            "PoM Metal: {} tensors, {} chunks (~{} MiB) resident on device {} (zero-dup)",
            miner.n_tensors,
            n_total_chunks,
            (n_total_chunks as usize * CHUNK_BYTES) / (1024 * 1024),
            device_id
        );
        Ok(miner)
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest nonce whose `pom_pow_value`
    /// is `<= target_le`, or None. `batch` must fit in `u32` (POM_BATCH is < 2^32) so the winner
    /// atomic stays a 32-bit tid — byte-identical to CUDA's `atomicMin(winner, nonce)`.
    /// `h3` salts the pph words host-side (`POM_H3_PPH_SALT`) — the Metal kernel is era-agnostic,
    /// it folds whatever words it receives, so there is no shader change at the H3 gate.
    pub fn mine(
        &self,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
        h3: bool,
    ) -> candle_core::Result<Option<u64>> {
        if batch > u32::MAX as u64 {
            return Err(candle_core::Error::Msg("PoM Metal: batch exceeds u32".into()));
        }
        if batch == 0 {
            return Ok(None);
        }
        let p = crate::pom::pph_words_for_era(pre_pow_hash, h3);
        let t = words4(target_le);
        let uniforms = Uniforms {
            n_total_chunks: self.n_total_chunks,
            k_steps: crate::pom::POM_WALK_STEPS,
            n_tensors: self.n_tensors,
            p0: p[0],
            p1: p[1],
            p2: p[2],
            p3: p[3],
            time_: timestamp,
            t0: t[0],
            t1: t[1],
            t2: t[2],
            t3: t[3],
            nonce_base: start,
            n_nonces: batch as u32,
            _pad: 0,
        };

        let uniforms_buf = self
            .device
            .new_buffer_with_data(&uniforms as *const _ as *const _, std::mem::size_of::<Uniforms>(), SHARED_STORAGE)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: uniforms buffer: {e}")))?;
        let winner_init: u32 = u32::MAX;
        let winner_buf = self
            .device
            .new_buffer_with_data(&winner_init as *const _ as *const _, std::mem::size_of::<u32>(), SHARED_STORAGE)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: winner buffer: {e}")))?;

        let semaphore = Arc::new(CommandSemaphore::new());
        let cmd = create_command_buffer(&self.queue, semaphore)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: command buffer: {e}")))?;
        let enc = cmd.compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&self.prefix_buf), 0);
        enc.set_buffer(1, Some(&self.addrs_buf), 0);
        enc.set_buffer(2, Some(&uniforms_buf), 0);
        enc.set_buffer(3, Some(&winner_buf), 0);
        // Bindless: the resident per-tensor buffers are dereffed via raw gpuAddress in the kernel,
        // so nothing binds them to a slot — we must still tell the driver they'll be read.
        for buf in &self.resources {
            enc.use_resource(buf, MTLResourceUsage::Read);
        }
        let grid = MTLSize { width: batch as usize, height: 1, depth: 1 };
        let tg = MTLSize { width: THREADGROUP_SIZE, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Safe: shared-storage buffer is CPU-visible and the dispatch has completed.
        let w = unsafe { *(winner_buf.contents() as *const u32) };
        Ok(if w == u32::MAX { None } else { Some(start + w as u64) })
    }

    /// Test-only: runs the walk on the GPU for `batch` consecutive nonces starting at `start` and
    /// returns each nonce's `final_state`. Uses the SAME uniform + tensor plumbing as `mine()`,
    /// so any divergence from the CPU walk over the same weights is a kernel/indexing bug.
    #[cfg(test)]
    pub fn debug_walk_states(
        &self,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        start: u64,
        batch: u64,
        h3: bool,
    ) -> candle_core::Result<Vec<u64>> {
        assert!(batch > 0 && batch <= u32::MAX as u64);
        let p = crate::pom::pph_words_for_era(pre_pow_hash, h3);
        let uniforms = Uniforms {
            n_total_chunks: self.n_total_chunks,
            k_steps: crate::pom::POM_WALK_STEPS,
            n_tensors: self.n_tensors,
            p0: p[0],
            p1: p[1],
            p2: p[2],
            p3: p[3],
            time_: timestamp,
            t0: 0,
            t1: 0,
            t2: 0,
            t3: 0,
            nonce_base: start,
            n_nonces: batch as u32,
            _pad: 0,
        };
        let uniforms_buf = self
            .device
            .new_buffer_with_data(&uniforms as *const _ as *const _, std::mem::size_of::<Uniforms>(), SHARED_STORAGE)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal(dbg): uniforms: {e}")))?;
        let states_bytes = (batch as usize) * std::mem::size_of::<u64>();
        let states_buf = self
            .device
            .new_buffer(states_bytes, SHARED_STORAGE)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal(dbg): states buffer: {e}")))?;

        let semaphore = Arc::new(CommandSemaphore::new());
        let cmd = create_command_buffer(&self.queue, semaphore)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal(dbg): cmd: {e}")))?;
        let enc = cmd.compute_command_encoder();
        enc.set_compute_pipeline_state(&self.debug_pipeline);
        enc.set_buffer(0, Some(&self.prefix_buf), 0);
        enc.set_buffer(1, Some(&self.addrs_buf), 0);
        enc.set_buffer(2, Some(&uniforms_buf), 0);
        enc.set_buffer(3, Some(&states_buf), 0);
        for buf in &self.resources {
            enc.use_resource(buf, MTLResourceUsage::Read);
        }
        let grid = MTLSize { width: batch as usize, height: 1, depth: 1 };
        let tg = MTLSize { width: THREADGROUP_SIZE, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut out = vec![0u64; batch as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                states_buf.contents() as *const u64,
                out.as_mut_ptr(),
                batch as usize,
            );
        }
        Ok(out)
    }
}

// ─── process-global registry + free-function surface (mirrors pom_gpu.rs exactly) ────────────────

fn miners() -> &'static Mutex<HashMap<u32, Arc<PomGpuMiner>>> {
    static MINERS: OnceLock<Mutex<HashMap<u32, Arc<PomGpuMiner>>>> = OnceLock::new();
    MINERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Devices this process's PoM walk is installed on, ascending. On Apple Silicon this is the single
/// integrated GPU (ordinal 0) once a PoM job installs a miner. Mirrors `pom_gpu::walk_devices`.
pub fn walk_devices() -> Vec<u32> {
    let mut v: Vec<u32> = miners().lock().map(|g| g.keys().copied().collect()).unwrap_or_default();
    v.sort_unstable();
    v
}

fn index_build_lock() -> &'static Mutex<()> {
    static INDEX_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    INDEX_BUILD_LOCK.get_or_init(|| Mutex::new(()))
}

pub fn install(device_id: u32, m: PomGpuMiner) {
    if let Ok(mut g) = miners().lock() {
        g.insert(device_id, Arc::new(m));
    }
}

fn remove_device_entry<T>(map: &mut HashMap<u32, T>, device_id: u32) {
    map.remove(&device_id);
}

pub fn uninstall(device_id: u32) {
    if let Ok(mut g) = miners().lock() {
        remove_device_entry(&mut g, device_id);
    }
}

pub fn is_installed(device_id: u32) -> bool {
    miners().lock().map(|g| g.contains_key(&device_id)).unwrap_or(false)
}

static LOADING: AtomicUsize = AtomicUsize::new(0);

pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed) > 0
}

/// Convenience: search a nonce batch via the installed miner for a specific device.
pub fn mine(
    device_id: u32,
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    target_le: &[u8; 32],
    start: u64,
    batch: u64,
    h3: bool,
) -> Option<u64> {
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    miner.mine(pre_pow_hash, timestamp, target_le, start, batch, h3).ok().flatten()
}

/// Mining-tier identity for rebuilds: (model_id, gguf_path). Set once at startup — the PROCESS-WIDE
/// DEFAULT model (single-model rigs, --light/--very-high/etc.). Per-device overrides (mixed rigs /
/// --force-model) go through `set_device_model`; `device_model()` prefers a per-device entry and
/// falls back to this default, so the single-model path stays byte-identical to before.
static MINING_TIER: OnceLock<([u8; 32], String)> = OnceLock::new();

pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String) {
    let _ = MINING_TIER.set((model_id, gguf_path));
}

/// Per-Metal-device mining model (model_id, gguf_path) — populated only for --force-model. Apple
/// Silicon has ONE integrated GPU (ordinal 0), so in practice this map either has an entry for 0
/// (when the operator pins a model to it) or is empty (default: use `MINING_TIER`). Present for
/// API parity with the CUDA backend so `main.rs` stays backend-agnostic.
fn device_models() -> &'static Mutex<HashMap<u32, ([u8; 32], String)>> {
    static DEVICE_MODELS: OnceLock<Mutex<HashMap<u32, ([u8; 32], String)>>> = OnceLock::new();
    DEVICE_MODELS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Assign a specific model to one Metal device (per-card / `--force-model`). Byte-identical
/// signature to `pom_gpu::set_device_model` — `main.rs` calls this on macOS through the
/// module alias.
pub fn set_device_model(device_id: u32, model_id: [u8; 32], gguf_path: String) {
    if let Ok(mut m) = device_models().lock() {
        m.insert(device_id, (model_id, gguf_path));
    }
}

/// The model this device mines: its per-device override if set, else the process-wide default.
fn device_model(device_id: u32) -> Option<([u8; 32], String)> {
    if let Ok(m) = device_models().lock() {
        if let Some(v) = m.get(&device_id) {
            return Some(v.clone());
        }
    }
    MINING_TIER.get().cloned()
}

/// Total GPU-usable memory (MiB) per Metal device — Apple Silicon reports ONE integrated GPU
/// (ordinal 0). We use `MTLDevice.recommendedMaxWorkingSetSize` (the driver's own recommendation
/// for the working set the GPU can allocate under memory pressure), not `hw.memsize`: on a 24 GB
/// M2 the working set is ~18 GB, and OPoI's VRAM gate should honour that, not the whole system
/// RAM. Byte-identical return shape to the CUDA path (`Vec<(device_id, MiB)>`) so `main.rs` and
/// the OPoI capability gate stay backend-agnostic. Never panics.
pub fn query_all_gpus_vram() -> Vec<(u32, u64)> {
    use objc2_metal::MTLDevice as _;
    std::panic::catch_unwind(|| {
        let cdev = match candle_core::Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let mdev = match &cdev {
            candle_core::Device::Metal(m) => m.metal_device().clone(),
            _ => return Vec::new(),
        };
        let bytes = mdev.as_ref().recommendedMaxWorkingSetSize();
        vec![(0u32, bytes / (1024 * 1024))]
    })
    .unwrap_or_default()
}

pub fn ensure_installed(device_id: u32, daa: u64) -> bool {
    if is_installed(device_id) {
        return true;
    }
    LOADING.fetch_add(1, Ordering::Relaxed);
    let ok = ensure_installed_inner(device_id, daa);
    LOADING.fetch_sub(1, Ordering::Relaxed);
    ok
}

/// PoM tier index of THIS device's mining model at a given block DAA — reads the per-device
/// override (`set_device_model`) first, falls back to the process-wide default (`MINING_TIER`),
/// so a per-card model in a mixed rig is tagged with its own tier. Signature byte-identical to
/// the CUDA `pom_gpu::current_tier` for backend-agnostic call sites.
pub fn current_tier(device_id: u32, daa: u64) -> Option<u8> {
    let (model_id, _) = device_model(device_id)?;
    crate::models::pom_tier_index(&model_id, daa)
}

fn ensure_installed_inner(device_id: u32, daa: u64) -> bool {
    let (model_id, gguf) = match device_model(device_id) {
        Some(x) => (x.0, x.1),
        None => return false,
    };
    let model_id = &model_id;
    let gguf = &gguf;
    // Build the possession index once (host, heavy) the first time PoM activates.
    if crate::pom::active_index().is_none() {
        let _guard = match index_build_lock().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if crate::pom::active_index().is_none() {
            // The background prefetch may still be fetching the mining-tier model. Wait for the
            // `.ok` completion sentinel and retry next job rather than spamming ENOENT.
            let ready = std::path::Path::new(gguf)
                .parent()
                .map(|d| d.join(".ok"))
                .map_or(false, |p| p.exists());
            if !ready {
                static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    info!(
                        "PoM Metal: mining-tier model not downloaded yet — deferring the \
                         possession-index build until the background prefetch finishes."
                    );
                }
                return false;
            }
            let tier = match crate::models::pom_tier_index(model_id, daa) {
                Some(t) => t,
                None => return false,
            };
            info!("PoM Metal: building shared host weight index (gpu{}) — this can take a while…", device_id);
            match crate::pom::WeightIndex::build_from_gguf(gguf) {
                Ok(idx) => {
                    info!("PoM Metal: shared host index ready — N={} chunks", idx.n_chunks);
                    crate::pom::set_index(idx, tier);
                }
                Err(e) => {
                    static LAST_LOG_SECS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    if now.saturating_sub(LAST_LOG_SECS.load(Ordering::Relaxed)) >= 300 {
                        LAST_LOG_SECS.store(now, Ordering::Relaxed);
                        log::error!("PoM Metal: possession-index build failed on gpu{}: {} (retrying each job; rate-limited).", device_id, e);
                    }
                    return false;
                }
            }
        }
    }
    // One Metal-resident PoM worker per device (Apple Silicon: the single integrated GPU). Phase 1
    // is a standalone load; sharing the inference engine's resident Metal buffers (the true zero-dup
    // with OPoI) is Phase 2. The N-guard validates the gather against the host index either way.
    match PomGpuMiner::load(gguf, device_id as usize) {
        Ok(gm) => {
            let n = gm.n_chunks();
            if let Some((idx, _)) = crate::pom::active_index() {
                if n != idx.n_chunks {
                    log::error!("PoM Metal[gpu{}]: gather N={} != shared index N={} — refusing to mine", device_id, n, idx.n_chunks);
                    return false;
                }
            }
            install(device_id, gm);
            info!("PoM Metal[gpu{}]: GPU miner ready — N={} chunks resident (matches shared index)", device_id, n);
            true
        }
        Err(e) => {
            log::error!("PoM Metal[gpu{}]: device miner build failed: {}", device_id, e);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pom;

    /// Build a miner from multiple synthetic tensor buffers — exercises the same multi-buffer
    /// bindless plumbing the real GGUF path uses (`upper_bound_prefix` non-trivial, `use_resource`
    /// on N > 1 buffers, `local = off - prefix[idx]` mapping) without needing a real model.
    fn miner_from_tensor_blobs(tensor_bytes: &[&[u8]]) -> PomGpuMiner {
        let cdev = Device::new_metal(0).expect("Metal device 0");
        let mdev = match &cdev {
            Device::Metal(m) => m.metal_device().clone(),
            _ => panic!("not a Metal device"),
        };
        let mut prefix: Vec<u64> = vec![0];
        let mut addrs: Vec<u64> = Vec::new();
        let mut resources: Vec<Buffer> = Vec::new();
        let mut cum: u64 = 0;
        for bytes in tensor_bytes {
            assert!(bytes.len() % CHUNK_BYTES == 0 && !bytes.is_empty());
            let n_chunks = (bytes.len() / CHUNK_BYTES) as u64;
            let buf = mdev
                .new_buffer_with_data(bytes.as_ptr() as *const _, bytes.len(), SHARED_STORAGE)
                .expect("tensor buffer");
            addrs.push(buf.as_ref().gpuAddress());
            cum += n_chunks;
            prefix.push(cum);
            resources.push(buf);
        }
        PomGpuMiner::build(mdev, &prefix, &addrs, resources, cum).expect("build multi-tensor miner")
    }

    /// Build a miner directly over a synthetic weight blob (one tensor, `n_chunks` × 32 B) so the
    /// walk can be exercised without a GGUF or the inference engine. The kernel's `upper_bound`
    /// resolves every offset to tensor 0, `local = off`, and reads `addrs[0][off*4 .. off*4+4]` —
    /// i.e. the contiguous blob — exactly what the CPU oracle reads.
    fn miner_from_blob(bytes: &[u8]) -> PomGpuMiner {
        assert!(bytes.len() % CHUNK_BYTES == 0 && !bytes.is_empty());
        let n_chunks = (bytes.len() / CHUNK_BYTES) as u64;
        let cdev = Device::new_metal(0).expect("Metal device 0");
        let mdev = match &cdev {
            Device::Metal(m) => m.metal_device().clone(),
            _ => panic!("not a Metal device"),
        };
        let buf = mdev
            .new_buffer_with_data(bytes.as_ptr() as *const _, bytes.len(), SHARED_STORAGE)
            .expect("blob buffer");
        let addr = buf.as_ref().gpuAddress();
        let prefix = [0u64, n_chunks];
        let addrs = [addr];
        PomGpuMiner::build(mdev, &prefix, &addrs, vec![buf], n_chunks).expect("build test miner")
    }

    /// Byte-exact CPU oracle: reproduce the walk over the same blob using pom.rs primitives.
    /// Pre-H3 era (h3=false) to match the `mine(…, false)` call below.
    fn cpu_pow_value(bytes: &[u8], n_chunks: u64, pph: &[u8; 32], ts: u64, nonce: u64) -> [u8; 32] {
        let seed = pom::pom_block_seed(pph, ts, nonce, false);
        let read = |off: u64| -> [u64; pom::CHUNK_WORDS] {
            let o = (off as usize) * CHUNK_BYTES;
            let mut c = [0u8; 32];
            c.copy_from_slice(&bytes[o..o + 32]);
            pom::chunk_to_words(&c)
        };
        let final_state = pom::walk_final(seed, n_chunks, pom::POM_WALK_STEPS, read);
        pom::pom_pow_value(final_state, pph, false)
    }

    /// On-device diagnostic: compares the GPU-side bytes at the first chunk of the first N
    /// tensors (via `MTLBuffer::contents()` — Apple unified memory) against the host
    /// `WeightIndex::read_chunk(off)` at the equivalent canonical chunk index. If bytes diverge
    /// even for chunk 0 of tensor 0, the "borrow candle-metal's per-tensor buffers" strategy is
    /// wrong for this consensus use case (candle re-packed / offset / padded them), and the fix
    /// is to pack the raw GGUF quantized bytes ourselves.
    ///
    /// Requires the real Gemma-3-4B GGUF: pass its path in `KERYX_TEST_GGUF`. The host index
    /// build reuses `pom-tree.bin` in the same directory if present. Ignored (skipped) if
    /// the env var is unset so `cargo test` on a plain checkout still passes.
    #[test]
    #[ignore]
    fn metal_load_bytes_match_host_index_real_model() {
        let gguf = match std::env::var("KERYX_TEST_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip: set KERYX_TEST_GGUF to the model.gguf path");
                return;
            }
        };

        // Build the host WeightIndex (reuses pom-tree.bin next to the GGUF if present).
        eprintln!("building host WeightIndex from {gguf} (may reuse pom-tree.bin)…");
        let idx = pom::WeightIndex::build_from_gguf(&gguf).expect("host index");
        eprintln!("host index: N = {} chunks", idx.n_chunks);

        // Reproduce PomGpuMiner::load()'s tensor scan so we can inspect per-tensor buffers.
        let cdev = Device::new_metal(0).expect("Metal 0");
        let _mdev = match &cdev {
            Device::Metal(m) => m.metal_device().clone(),
            _ => panic!("not Metal"),
        };
        let mut file = std::fs::File::open(&gguf).expect("open gguf");
        let content = gguf_file::Content::read(&mut file).expect("gguf hdr");
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort();

        // Per-tensor (name, first-global-chunk-index, gpu_first_32_bytes, host_first_32_bytes,
        // buffer_length, gguf_file_offset, storage_size_in_bytes).
        let mut cum: u64 = 0;
        let mut mismatches = 0usize;
        let mut inspected = 0usize;
        let n_to_dump: usize = std::env::var("KERYX_TEST_DUMP_TENSORS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);
        for name in &names {
            let info = &content.tensor_infos[name];
            let file_off = content.tensor_data_offset + info.offset;
            let qt = content.tensor(&mut file, name, &cdev).expect("qt");
            let n_bytes = qt.storage_size_in_bytes();
            if n_bytes < CHUNK_BYTES {
                continue;
            }
            let qmet = qt.metal_storage().expect("Metal QStorage");
            let buf: Buffer = qmet.buffer().clone();
            let buf_len = buf.length();
            let n_chunks = (n_bytes / CHUNK_BYTES) as u64;
            let global_off = cum;

            // Sample GPU vs host at first, last, and a few interior chunks of the tensor.
            let gpu_ptr = buf.contents();
            assert!(!gpu_ptr.is_null(), "buffer.contents() null for {name}");
            let sample_offs: Vec<u64> = {
                let mut s = vec![0u64];
                if n_chunks > 1 {
                    s.push(n_chunks - 1);
                }
                if n_chunks > 4 {
                    s.push(n_chunks / 2);
                    s.push(1);
                }
                s
            };
            let mut gpu_first = [0u8; 32];
            let mut host_first = [0u8; 32];
            let mut tensor_ok = true;
            for &lo in &sample_offs {
                let mut gpu = [0u8; 32];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (gpu_ptr as *const u8).add((lo as usize) * 32),
                        gpu.as_mut_ptr(),
                        32,
                    );
                }
                let host_words = idx.read_chunk(global_off + lo);
                let host = pom::words_to_bytes(&host_words);
                if lo == 0 {
                    gpu_first = gpu;
                    host_first = host;
                }
                if gpu != host {
                    tensor_ok = false;
                    if inspected < n_to_dump || mismatches < 4 {
                        eprintln!(
                            "  MISMATCH tensor={name} local_off={lo} global_off={} \
                             gpu={} host={}",
                            global_off + lo,
                            hex32(&gpu),
                            hex32(&host)
                        );
                    }
                }
            }
            let match_ = tensor_ok;
            if inspected < n_to_dump {
                eprintln!(
                    "[{:>3}] {:<40} n_bytes={} buf_len={} file_off={} chunks={} global_off={}  {}",
                    inspected,
                    name,
                    n_bytes,
                    buf_len,
                    file_off,
                    n_chunks,
                    global_off,
                    if match_ { "OK" } else { "MISMATCH" }
                );
                eprintln!("      GPU  first32: {}", hex32(&gpu_first));
                eprintln!("      HOST first32: {}", hex32(&host_first));
            }
            if !match_ {
                mismatches += 1;
            }
            cum += n_chunks;
            inspected += 1;
        }
        eprintln!(
            "total tensors: {inspected}, mismatches: {mismatches}, N (metal cum)={cum} vs host N={}",
            idx.n_chunks
        );
        assert_eq!(cum, idx.n_chunks, "tensor total chunk count divergence");
        assert_eq!(mismatches, 0, "GPU chunk[0] differs from host chunk[global_off] for {mismatches} tensors");

        // ── Second diagnostic: run the CPU walk over the GPU's OWN buffers (via .contents())
        // for a specific nonce, and compare `final_state` to the host WeightIndex walk. This
        // isolates a "kernel indexing bug" (walks diverge) from a "data mapping bug" (walks agree
        // over the same underlying bytes → the Metal kernel is the culprit).
        eprintln!("\n── walk-final CPU-over-GPU-buffers vs CPU-over-WeightIndex ──");

        // Rebuild prefix + per-tensor Buffer clones (drop them at end of test).
        let mut file2 = std::fs::File::open(&gguf).expect("open gguf");
        let content2 = gguf_file::Content::read(&mut file2).expect("gguf hdr");
        let mut prefix2: Vec<u64> = vec![0];
        let mut bufs: Vec<Buffer> = Vec::new();
        let mut cum2: u64 = 0;
        for name in &names {
            let qt = content2.tensor(&mut file2, name, &cdev).expect("qt");
            let n_bytes = qt.storage_size_in_bytes();
            if n_bytes < CHUNK_BYTES {
                continue;
            }
            let qmet = qt.metal_storage().expect("Metal QStorage");
            let buf: Buffer = qmet.buffer().clone();
            let n_chunks = (n_bytes / CHUNK_BYTES) as u64;
            cum2 += n_chunks;
            prefix2.push(cum2);
            bufs.push(buf);
        }
        let n_total = cum2;

        // Reads chunk[off] via the GPU buffers, mirroring the kernel's upper_bound + local.
        let gpu_backed_read = |off: u64| -> [u64; pom::CHUNK_WORDS] {
            let i = prefix2.partition_point(|&p| p <= off) - 1;
            let local = off - prefix2[i];
            let ptr = bufs[i].contents() as *const u8;
            let mut c = [0u8; 32];
            unsafe {
                std::ptr::copy_nonoverlapping(ptr.add((local as usize) * 32), c.as_mut_ptr(), 32);
            }
            pom::chunk_to_words(&c)
        };

        // Pick a nonzero deterministic (pph, ts, nonce). Doesn't need to satisfy any target — we
        // only compare final_state.
        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(7));
        let ts: u64 = 0x1122_3344_5566_7788;
        let test_nonces: &[u64] = &[0, 1, 42, 100_000, 12345678];
        let mut host_states = Vec::new();
        for &nonce in test_nonces {
            let seed = pom::pom_block_seed(&pph, ts, nonce, false);
            let host_state = pom::walk_final(seed, n_total, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            let gpu_state = pom::walk_final(seed, n_total, pom::POM_WALK_STEPS, gpu_backed_read);
            eprintln!(
                "nonce={nonce:<10}  host_final=0x{host_state:016x}  gpubuf_final=0x{gpu_state:016x}  {}",
                if host_state == gpu_state { "OK" } else { "MISMATCH" }
            );
            assert_eq!(host_state, gpu_state, "walk over GPU buffers diverges from walk over WeightIndex for nonce {nonce}");
            host_states.push(host_state);
        }

        // ── Third diagnostic: run the ACTUAL Metal kernel via `debug_walk_states` and compare
        // GPU-reported final_state to the host walk. If this diverges the bug is IN the kernel.
        eprintln!("\n── walk-final Metal kernel vs CPU-over-WeightIndex ──");

        // Rebuild a PomGpuMiner using the same setup as `load()` so `debug_walk_states` runs the
        // full production path (prefix / addrs tables uploaded to Metal buffers, use_resource on
        // all 444 tensors during dispatch).
        let miner = PomGpuMiner::load(&gguf, 0).expect("PomGpuMiner::load");
        // Ensure the two paths see the same N so the walk trajectories can be compared at all.
        assert_eq!(miner.n_chunks(), idx.n_chunks, "N mismatch between metal miner and host index");

        // One dispatch, `batch = 8` consecutive nonces starting at each test nonce, take state[0].
        for (i, &nonce) in test_nonces.iter().enumerate() {
            let gpu_states = miner
                .debug_walk_states(&pph, ts, nonce, 8, false)
                .expect("debug_walk_states");
            let gpu_state = gpu_states[0];
            let host_state = host_states[i];
            eprintln!(
                "nonce={nonce:<10}  host_final=0x{host_state:016x}  metal_kernel_final=0x{gpu_state:016x}  {}",
                if host_state == gpu_state { "OK" } else { "MISMATCH" }
            );
        }

        // ── Fourth diagnostic: large-batch walk-state sweep — verifies the kernel is byte-exact
        // across the full batch dispatch (256+ threadgroups), catching a race or threadgroup-scope
        // bug the tiny 8-thread dispatch above cannot.
        eprintln!("\n── large-batch Metal-kernel walk states vs host walk states ──");
        let sweep_batch: u64 = 4096;
        let sweep_start: u64 = 100_000;
        let sweep_gpu = miner
            .debug_walk_states(&pph, ts, sweep_start, sweep_batch, false)
            .expect("debug_walk_states sweep");
        let mut disagreements = 0usize;
        for i in 0..sweep_batch {
            let nonce = sweep_start + i;
            let seed = pom::pom_block_seed(&pph, ts, nonce, false);
            let host = pom::walk_final(seed, n_total, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            let gpu = sweep_gpu[i as usize];
            if host != gpu {
                if disagreements < 5 {
                    eprintln!(
                        "  DIVERGE nonce={nonce} host=0x{host:016x} gpu=0x{gpu:016x}"
                    );
                }
                disagreements += 1;
            }
        }
        eprintln!("large batch: {disagreements}/{sweep_batch} disagreements");
        assert_eq!(disagreements, 0, "Metal kernel walk diverges from host walk in the large-batch dispatch");

        // ── Fifth diagnostic: exercise `mine()` end-to-end with a target derived from a known
        // nonce's CPU pow_value — the LOWEST such nonce must be the winner returned.
        eprintln!("\n── mine() winner path over the real model ──");
        let winner_start: u64 = 500_000;
        let winner_batch: u64 = 8192;
        let n_star = winner_start + winner_batch / 3;
        let seed_star = pom::pom_block_seed(&pph, ts, n_star, false);
        let final_star = pom::walk_final(seed_star, n_total, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
        let target = pom::pom_pow_value(final_star, &pph, false);
        eprintln!("target = pow_value(nonce={n_star}) = {}", hex32(&target));

        let expected = (winner_start..winner_start + winner_batch).find(|&n| {
            let s = pom::pom_block_seed(&pph, ts, n, false);
            let fs = pom::walk_final(s, n_total, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            pom::le_leq(&pom::pom_pow_value(fs, &pph, false), &target)
        });
        let got = miner
            .mine(&pph, ts, &target, winner_start, winner_batch, false)
            .expect("mine");
        eprintln!("mine() returned {got:?} — expected {expected:?}");
        assert_eq!(got, expected, "mine() winner disagrees with host reference");

        // ── Sixth diagnostic: H3 era (post-fork). The network activated H3 at DAA 43_450_000
        // (~2026-07-05). If the walk is byte-exact for h3=false but diverges for h3=true, the
        // pph-salt plumbing is wrong.
        eprintln!("\n── H3 era (h3=true) walk-state parity ──");
        let h3_sweep_batch: u64 = 1024;
        let h3_sweep_gpu = miner
            .debug_walk_states(&pph, ts, sweep_start, h3_sweep_batch, true)
            .expect("debug_walk_states h3");
        let mut h3_diverge = 0usize;
        for i in 0..h3_sweep_batch {
            let nonce = sweep_start + i;
            let seed = pom::pom_block_seed(&pph, ts, nonce, true);
            let host = pom::walk_final(seed, n_total, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            let gpu = h3_sweep_gpu[i as usize];
            if host != gpu {
                if h3_diverge < 5 {
                    eprintln!("  H3 DIVERGE nonce={nonce} host=0x{host:016x} gpu=0x{gpu:016x}");
                }
                h3_diverge += 1;
            }
        }
        eprintln!("H3 batch: {h3_diverge}/{h3_sweep_batch} disagreements");

        // Also exercise mine() winner path under h3=true.
        let h3_n_star = winner_start + winner_batch / 3;
        let h3_seed_star = pom::pom_block_seed(&pph, ts, h3_n_star, true);
        let h3_final_star = pom::walk_final(h3_seed_star, n_total, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
        let h3_target = pom::pom_pow_value(h3_final_star, &pph, true);
        let h3_expected = (winner_start..winner_start + winner_batch).find(|&n| {
            let s = pom::pom_block_seed(&pph, ts, n, true);
            let fs = pom::walk_final(s, n_total, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            pom::le_leq(&pom::pom_pow_value(fs, &pph, true), &h3_target)
        });
        let h3_got = miner
            .mine(&pph, ts, &h3_target, winner_start, winner_batch, true)
            .expect("mine h3");
        eprintln!("H3 mine() returned {h3_got:?} — expected {h3_expected:?}");
        assert_eq!(h3_diverge, 0, "H3 kernel walk diverges from host walk");
        assert_eq!(h3_got, h3_expected, "H3 mine() winner disagrees with host");
    }

    fn hex32(b: &[u8; 32]) -> String {
        let mut s = String::with_capacity(64);
        for x in b {
            s.push_str(&format!("{:02x}", x));
        }
        s
    }

    /// Multi-tensor byte-exact test: splits a deterministic pseudo-random blob into 12 unevenly
    /// sized tensors, uploads each into its own MTLBuffer (so `use_resource` handles 12 buffers,
    /// `upper_bound_prefix` performs a real search, and `local` boundary math is exercised), and
    /// checks the Metal `mine()` winner matches the CPU oracle over the concatenated bytes. This
    /// guards against the class of bugs the on-device diagnostic
    /// (`metal_load_bytes_match_host_index_real_model`) verified are absent in the real 444-tensor
    /// path, and runs on any Apple Silicon CI without the 2.4 GB GGUF.
    #[test]
    fn metal_walk_matches_host_reference_multi_tensor() {
        // 12 tensors, sizes chosen to be non-power-of-two multiples of 32 B and mutually distinct.
        let per_tensor: [usize; 12] = [
            5 * 32, 17 * 32, 41 * 32, 128 * 32, 257 * 32, 511 * 32,
            1024 * 32, 2001 * 32, 4096 * 32, 512 * 32, 91 * 32, 331 * 32,
        ];
        let total_bytes: usize = per_tensor.iter().sum();
        let mut bytes = vec![0u8; total_bytes];
        let mut s = 0xdead_beef_cafe_babeu64;
        for b in bytes.iter_mut() {
            s = pom::mix64(s);
            *b = (s & 0xff) as u8;
        }
        let mut tensors: Vec<&[u8]> = Vec::with_capacity(per_tensor.len());
        let mut off = 0usize;
        for &sz in &per_tensor {
            tensors.push(&bytes[off..off + sz]);
            off += sz;
        }
        let n_chunks = (total_bytes / CHUNK_BYTES) as u64;
        let miner = miner_from_tensor_blobs(&tensors);
        assert_eq!(miner.n_chunks(), n_chunks);

        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(53).wrapping_add(19));
        let ts: u64 = 0xcafebabe_deadbeef;
        let (start, batch) = (777u64, 8192u64);
        let n_star = start + batch / 2;
        let target = cpu_pow_value(&bytes, n_chunks, &pph, ts, n_star);
        let expected = (start..start + batch)
            .find(|&n| pom::le_leq(&cpu_pow_value(&bytes, n_chunks, &pph, ts, n), &target));
        let got = miner.mine(&pph, ts, &target, start, batch, false).expect("metal mine");
        assert_eq!(got, expected, "multi-tensor Metal winner != host reference");

        // Same batch under h3=true — proves the pph-salt plumbing is byte-exact for both eras.
        let target_h3 = {
            let seed = pom::pom_block_seed(&pph, ts, n_star, true);
            let read = |off: u64| -> [u64; pom::CHUNK_WORDS] {
                let o = (off as usize) * CHUNK_BYTES;
                let mut c = [0u8; 32];
                c.copy_from_slice(&bytes[o..o + 32]);
                pom::chunk_to_words(&c)
            };
            let final_state = pom::walk_final(seed, n_chunks, pom::POM_WALK_STEPS, read);
            pom::pom_pow_value(final_state, &pph, true)
        };
        let expected_h3 = (start..start + batch).find(|&n| {
            let seed = pom::pom_block_seed(&pph, ts, n, true);
            let read = |off: u64| -> [u64; pom::CHUNK_WORDS] {
                let o = (off as usize) * CHUNK_BYTES;
                let mut c = [0u8; 32];
                c.copy_from_slice(&bytes[o..o + 32]);
                pom::chunk_to_words(&c)
            };
            let fs = pom::walk_final(seed, n_chunks, pom::POM_WALK_STEPS, read);
            pom::le_leq(&pom::pom_pow_value(fs, &pph, true), &target_h3)
        });
        let got_h3 = miner.mine(&pph, ts, &target_h3, start, batch, true).expect("metal mine h3");
        assert_eq!(got_h3, expected_h3, "multi-tensor Metal winner (h3) != host reference");
    }

    // Requires a real Metal device — runs on Apple Silicon (CI macos runner / dev Mac), skipped
    // (compiled-out) everywhere else since the whole module is macOS-only.
    #[test]
    fn metal_walk_matches_host_reference() {
        // Deterministic pseudo-random weight blob (splitmix64 stream) — 4096 chunks = 128 KiB.
        let n_chunks = 4096u64;
        let mut bytes = vec![0u8; (n_chunks as usize) * CHUNK_BYTES];
        let mut s = 0x1234_5678_9abc_def0u64;
        for b in bytes.iter_mut() {
            s = pom::mix64(s);
            *b = (s & 0xff) as u8;
        }
        let miner = miner_from_blob(&bytes);
        assert_eq!(miner.n_chunks(), n_chunks);

        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        let ts: u64 = 0x00ff_00ff_1234u64;
        let (start, batch) = (1_000u64, 20_000u64);

        // Pick a target that at least one nonce in the batch satisfies: use the pow value of a
        // known mid-batch nonce, so the search must find the LOWEST nonce whose pv <= target.
        let n_star = start + batch / 2;
        let target = cpu_pow_value(&bytes, n_chunks, &pph, ts, n_star);

        let expected = (start..start + batch)
            .find(|&n| pom::le_leq(&cpu_pow_value(&bytes, n_chunks, &pph, ts, n), &target));
        let got = miner.mine(&pph, ts, &target, start, batch, false).expect("metal mine");
        assert_eq!(got, expected, "Metal winner != host reference");
        assert!(got.is_some(), "n_star must be a winner");
        assert!(got.unwrap() <= n_star);
    }
}
