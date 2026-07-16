//! Proof-of-Model GPU mining — Apple Silicon (Metal) backend.
//!
//! Parity path for `pom_gpu.rs` (CUDA): exposes the **identical** public free-function surface
//! (`install`/`uninstall`/`is_installed`/`is_loading`/`mine`/`set_mining_tier`/`ensure_installed`/
//! `current_tier`/`walk_devices`) so `main.rs`/`miner.rs`/`slm.rs` stay backend-agnostic — on macOS
//! `lib.rs` aliases this module to `pom_gpu`. Nothing here compiles off macOS.
//!
//! The walk runs the `metal/pom_mine.metal` compute kernel — whose seed/pow folds are byte-identical
//! to `pom::pom_block_seed`/`pom::pom_pow_value` and the CUDA `pom_mine.cu`, so a nonce found here
//! builds a `PomProof` the node accepts.
//!
//! ### Storage layout: single packed MTLBuffer (Phase 3 candle-independence for the walk)
//! Since v0.6.10.x the walk no longer borrows candle-owned per-tensor MTLBuffers via
//! `QTensor::metal_storage()`. Instead this module owns **one** contiguous MTLBuffer holding the
//! GGUF quantized bytes packed in canonical (name-sorted, skip-<32 B) order, each tensor truncated
//! to a whole-32-byte-chunk multiple (the same integer-division truncation `pom.rs::WeightIndex`
//! applies host-side, so chunk-index → same 32 bytes on both sides by construction).
//!
//! Trade-off: one ~2.4 GB copy in Apple unified memory at load time, in exchange for:
//!   * No dependency on candle's Metal Device or `QTensor::metal_storage()` (the vendored patch).
//!   * A guarantee that the walk sees the exact bytes the host possession index reads —
//!     regardless of any future changes to how candle-Metal repacks/aligns quantized blocks.
//!   * The prefix/addrs tables collapse to `prefix=[0, N]` / `addrs=[buf.gpuAddress()]`, so the
//!     kernel's `upper_bound_prefix` binary search resolves to tensor 0 in one iteration — the
//!     exact single-buffer path the `metal_walk_matches_host_reference` test has been exercising
//!     since day one.
//!
//! Consensus stays byte-identical: the walk math, `POM_H3_PPH_SALT`, chunk format and canonical
//! ordering are unchanged; only the data path is de-candled.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::quantized::gguf_file;
use candle_metal_kernels::metal::{
    create_command_buffer, Buffer, CommandQueue, CommandSemaphore, ComputePipeline,
    Device as MtlDevice, MTLResourceOptions,
};
use objc2_metal::{
    MTLBuffer as _, MTLResourceOptions as ObjcMTLResourceOptions, MTLResourceUsage, MTLSize,
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

    /// Load the mining model's GGUF into a single packed MTLBuffer and build the walk tables.
    /// Heavy — call once per (device, model). Reads raw quantized bytes straight from the GGUF
    /// (via `std::os::unix::fs::FileExt::read_exact_at`) into a Metal shared-storage buffer, no
    /// intermediate candle CPU/Metal tensor allocation. `gguf_file::Content` is still used to
    /// parse the header (metadata + `tensor_infos`) — that's a pure-CPU parser, no Metal touch.
    ///
    /// `device_id` is retained in the signature for backend-parity with `pom_gpu::load` (CUDA
    /// ordinal) but Apple Silicon exposes ONE integrated GPU (ordinal 0), so it's only
    /// pass-through for logging.
    pub fn load(gguf_path: &str, device_id: usize) -> candle_core::Result<Self> {
        use std::os::unix::fs::FileExt;

        // Metal device WITHOUT candle_core — pure candle_metal_kernels wrapper over objc2-metal.
        // Ordinal is a no-op on Apple Silicon (single integrated GPU).
        let _ = device_id;
        let mdev = MtlDevice::system_default().ok_or_else(|| {
            candle_core::Error::Msg("PoM Metal: MTLCreateSystemDefaultDevice returned nil".into())
        })?;

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical name-sorted order — matches pom-rt-builder / the node R_T

        // Pass 1: compute per-tensor file offset + packed byte count (each tensor truncated to a
        // whole 32-byte-chunk multiple, the same integer-division `pom.rs::WeightIndex` applies).
        // Tensors with byte count < 32 are skipped (biases, norms) — matches the CUDA gather.
        struct Pack {
            file_offset: u64,
            packed_bytes: usize,
        }
        let mut packs: Vec<Pack> = Vec::with_capacity(names.len());
        let mut total_bytes: usize = 0;
        for name in &names {
            let info = &content.tensor_infos[name];
            let n_elems: usize = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            if !n_elems.is_multiple_of(block_size) {
                return Err(candle_core::Error::Msg(format!(
                    "PoM Metal: tensor {name}: elements {n_elems} not divisible by block_size {block_size}"
                )));
            }
            let n_bytes = n_elems / block_size * info.ggml_dtype.type_size();
            let n_chunks = n_bytes / CHUNK_BYTES;
            if n_chunks == 0 {
                continue;
            }
            let packed_bytes = n_chunks * CHUNK_BYTES; // drops the tail < 32 B (host does the same)
            packs.push(Pack {
                file_offset: content.tensor_data_offset + info.offset,
                packed_bytes,
            });
            total_bytes = total_bytes
                .checked_add(packed_bytes)
                .ok_or_else(|| candle_core::Error::Msg("PoM Metal: total tensor byte count overflowed usize".into()))?;
        }
        if total_bytes == 0 {
            return Err(candle_core::Error::Msg("PoM Metal: model produced 0 chunks".into()));
        }
        let n_total_chunks = (total_bytes / CHUNK_BYTES) as u64;

        // Allocate ONE shared-storage MTLBuffer for the packed weights. Apple Silicon unified
        // memory: CPU writes here are visible to the GPU with no blit.
        let weights_buf = mdev
            .new_buffer(total_bytes, SHARED_STORAGE)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: alloc {total_bytes} B: {e}")))?;
        let base_ptr = weights_buf.contents();
        if base_ptr.is_null() {
            return Err(candle_core::Error::Msg("PoM Metal: buffer.contents() is null for shared-storage buffer".into()));
        }

        // Pass 2: pread raw GGUF bytes for each tensor's packed region into the Metal buffer at
        // its cumulative offset. `read_exact_at` bypasses the `File` cursor, so this is safe to
        // interleave with `gguf_file::Content::read` on the same handle.
        let mut cum: usize = 0;
        for p in &packs {
            // SAFETY: `base_ptr` is a valid, mapped writable pointer of `total_bytes` bytes (shared
            // storage on Apple Silicon exposes the same page to CPU + GPU). `cum + packed_bytes`
            // never exceeds `total_bytes` because that sum IS `total_bytes` after the loop.
            let dst = unsafe { std::slice::from_raw_parts_mut(base_ptr.add(cum), p.packed_bytes) };
            file.read_exact_at(dst, p.file_offset).map_err(candle_core::Error::wrap)?;
            cum += p.packed_bytes;
        }
        debug_assert_eq!(cum, total_bytes);

        // Single-tensor walk: prefix collapses to [0, N], addrs to [buf.gpuAddress()].
        // The kernel's `upper_bound_prefix` resolves every off < N to tensor 0, `local = off`,
        // reading `addrs[0][off*4]` — i.e. the packed buffer — exactly the path the byte-exact
        // `metal_walk_matches_host_reference` test exercises.
        let base_addr = weights_buf.as_ref().gpuAddress();
        let prefix = [0u64, n_total_chunks];
        let addrs = [base_addr];
        let resources = vec![weights_buf];
        let miner = Self::build(mdev, &prefix, &addrs, resources, n_total_chunks)?;
        info!(
            "PoM Metal: packed {} tensors ({} chunks, ~{} MiB) into 1 MTLBuffer on device {} \
             (candle-Metal independent)",
            packs.len(),
            n_total_chunks,
            (total_bytes) / (1024 * 1024),
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
        let mdev = match MtlDevice::system_default() {
            Some(d) => d,
            None => return Vec::new(),
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
        let mdev = MtlDevice::system_default().expect("Metal device 0");
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
        let mdev = MtlDevice::system_default().expect("Metal device 0");
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

    /// On-device diagnostic (real Gemma-3-4B GGUF): loads the production `PomGpuMiner::load`,
    /// which now packs raw GGUF quantized bytes into ONE Metal buffer, and asserts:
    ///
    /// 1. The packed buffer's chunks match the host `WeightIndex::read_chunk` byte-for-byte at a
    ///    broad set of sampled offsets (start, end, mid, and a stratified sweep).
    /// 2. `debug_walk_states` (real Metal kernel) → same `final_state` as the host walk over the
    ///    same `WeightIndex`, for 5 spot nonces + a 4096-nonce single-dispatch sweep.
    /// 3. The `mine()` winner path returns the correct lowest nonce for a target derived from a
    ///    real nonce's `pom_pow_value`.
    /// 4. All of the above also under `h3=true` (post-fork era) — proves the pph-salt plumbing
    ///    stays byte-exact after the switch to the packed single buffer.
    ///
    /// Gated by `KERYX_TEST_GGUF` + `#[ignore]` so `cargo test` on a plain checkout still passes;
    /// run with `KERYX_TEST_GGUF=/path/to/model.gguf cargo test --release --features pom-metal \
    /// metal_load_bytes_match_host_index_real_model -- --ignored --nocapture`.
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

        // Production loader — packs raw GGUF bytes into ONE MTLBuffer, no candle Metal touch.
        let miner = PomGpuMiner::load(&gguf, 0).expect("PomGpuMiner::load");
        eprintln!(
            "miner: N = {} chunks (packed single buffer, {} MiB)",
            miner.n_chunks(),
            (miner.n_chunks() as usize * CHUNK_BYTES) / (1024 * 1024)
        );
        assert_eq!(miner.n_chunks(), idx.n_chunks, "N mismatch between packed buffer and host index");

        // ── (1) Packed buffer vs host WeightIndex: sample chunks across the whole 2.4 GB range.
        eprintln!("\n── packed buffer vs host WeightIndex (byte-exact per-chunk) ──");
        let n = miner.n_chunks();
        // Sample offsets: 0, 1, N-1, N/2, plus a stratified sweep across the buffer at 1/16 steps.
        let mut sample_offs: Vec<u64> = vec![0, 1, n - 1, n / 2];
        for i in 1..16 {
            sample_offs.push((n * i) / 16);
        }
        sample_offs.sort_unstable();
        sample_offs.dedup();
        let base_ptr = miner.resources[0].contents() as *const u8;
        assert!(!base_ptr.is_null(), "packed buffer contents() is null");
        let mut mismatches = 0usize;
        for &off in &sample_offs {
            let mut buf = [0u8; 32];
            // SAFETY: `off < N`, `off * 32 + 32 <= N * 32 = buffer length`.
            unsafe { std::ptr::copy_nonoverlapping(base_ptr.add((off as usize) * 32), buf.as_mut_ptr(), 32) };
            let host_words = idx.read_chunk(off);
            let host = pom::words_to_bytes(&host_words);
            let ok = buf == host;
            eprintln!(
                "  off={off:>10}  packed={}  host={}  {}",
                hex32(&buf),
                hex32(&host),
                if ok { "OK" } else { "MISMATCH" }
            );
            if !ok {
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "packed buffer differs from host WeightIndex");

        // ── (2) Metal-kernel `debug_walk_states` vs host walk_final for 5 spot nonces.
        eprintln!("\n── Metal-kernel walk_final vs host walk_final (5 nonces) ──");
        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(7));
        let ts: u64 = 0x1122_3344_5566_7788;
        let test_nonces: &[u64] = &[0, 1, 42, 100_000, 12345678];
        for &nonce in test_nonces {
            let seed = pom::pom_block_seed(&pph, ts, nonce, false);
            let host_state = pom::walk_final(seed, n, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            let gpu_states = miner.debug_walk_states(&pph, ts, nonce, 8, false).expect("debug_walk_states");
            let gpu_state = gpu_states[0];
            eprintln!(
                "  nonce={nonce:<10}  host=0x{host_state:016x}  metal=0x{gpu_state:016x}  {}",
                if host_state == gpu_state { "OK" } else { "MISMATCH" }
            );
            assert_eq!(host_state, gpu_state, "walk diverges for nonce {nonce}");
        }

        // ── (3) Large-batch sweep: 4096 consecutive nonces in one dispatch, all must agree.
        eprintln!("\n── large-batch (4096 nonces) Metal-kernel walk vs host walk ──");
        let sweep_batch: u64 = 4096;
        let sweep_start: u64 = 100_000;
        let sweep_gpu = miner.debug_walk_states(&pph, ts, sweep_start, sweep_batch, false).expect("sweep");
        let mut disagreements = 0usize;
        for i in 0..sweep_batch {
            let nonce = sweep_start + i;
            let seed = pom::pom_block_seed(&pph, ts, nonce, false);
            let host = pom::walk_final(seed, n, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            if host != sweep_gpu[i as usize] {
                if disagreements < 5 {
                    eprintln!("  DIVERGE nonce={nonce} host=0x{host:016x} gpu=0x{:016x}", sweep_gpu[i as usize]);
                }
                disagreements += 1;
            }
        }
        eprintln!("large batch: {disagreements}/{sweep_batch} disagreements");
        assert_eq!(disagreements, 0, "large-batch Metal kernel walk diverges");

        // ── (4) mine() winner path: target derived from a real nonce's pow_value → the LOWEST
        //       satisfying nonce in the batch must be the winner returned.
        eprintln!("\n── mine() winner path ──");
        let winner_start: u64 = 500_000;
        let winner_batch: u64 = 8192;
        let n_star = winner_start + winner_batch / 3;
        let seed_star = pom::pom_block_seed(&pph, ts, n_star, false);
        let final_star = pom::walk_final(seed_star, n, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
        let target = pom::pom_pow_value(final_star, &pph, false);
        let expected = (winner_start..winner_start + winner_batch).find(|&nn| {
            let s = pom::pom_block_seed(&pph, ts, nn, false);
            let fs = pom::walk_final(s, n, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            pom::le_leq(&pom::pom_pow_value(fs, &pph, false), &target)
        });
        let got = miner.mine(&pph, ts, &target, winner_start, winner_batch, false).expect("mine");
        eprintln!("  mine() returned {got:?} — expected {expected:?}");
        assert_eq!(got, expected, "mine() winner disagrees with host reference");

        // ── (5) H3 era: pph-salt plumbing byte-exact under h3=true. 1024-nonce sweep + winner.
        eprintln!("\n── H3 era (h3=true) walk parity ──");
        let h3_sweep: u64 = 1024;
        let h3_gpu = miner.debug_walk_states(&pph, ts, sweep_start, h3_sweep, true).expect("h3 sweep");
        let mut h3_disagreements = 0usize;
        for i in 0..h3_sweep {
            let nonce = sweep_start + i;
            let seed = pom::pom_block_seed(&pph, ts, nonce, true);
            let host = pom::walk_final(seed, n, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            if host != h3_gpu[i as usize] {
                if h3_disagreements < 5 {
                    eprintln!("  H3 DIVERGE nonce={nonce} host=0x{host:016x} gpu=0x{:016x}", h3_gpu[i as usize]);
                }
                h3_disagreements += 1;
            }
        }
        eprintln!("H3 batch: {h3_disagreements}/{h3_sweep} disagreements");
        assert_eq!(h3_disagreements, 0, "H3 kernel walk diverges");

        let h3_seed = pom::pom_block_seed(&pph, ts, n_star, true);
        let h3_final = pom::walk_final(h3_seed, n, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
        let h3_target = pom::pom_pow_value(h3_final, &pph, true);
        let h3_expected = (winner_start..winner_start + winner_batch).find(|&nn| {
            let s = pom::pom_block_seed(&pph, ts, nn, true);
            let fs = pom::walk_final(s, n, pom::POM_WALK_STEPS, |o| idx.read_chunk(o));
            pom::le_leq(&pom::pom_pow_value(fs, &pph, true), &h3_target)
        });
        let h3_got = miner.mine(&pph, ts, &h3_target, winner_start, winner_batch, true).expect("mine h3");
        eprintln!("  H3 mine() returned {h3_got:?} — expected {h3_expected:?}");
        assert_eq!(h3_got, h3_expected, "H3 mine() winner disagrees");
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
