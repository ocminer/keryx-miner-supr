//! Proof-of-Model GPU mining — Apple Silicon (Metal) backend.
//!
//! Parity path for `pom_gpu.rs` (CUDA): exposes the **identical** public free-function surface
//! (`install`/`uninstall`/`uninstall_released`/`is_installed`/`is_loading`/`mine`/`set_mining_tier`/
//! `ensure_installed`/
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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::quantized::gguf_file;
use candle_metal_kernels::metal::{
    create_command_buffer, Buffer, CommandBuffer, CommandQueue, CommandSemaphore, ComputePipeline, Device as MtlDevice,
    MTLResourceOptions,
};
use objc2_metal::{
    MTLBuffer as _, MTLCommandBufferStatus, MTLResourceOptions as ObjcMTLResourceOptions, MTLResourceUsage, MTLSize,
};

const METAL_SRC: &str = include_str!("../metal/pom_mine.metal");
const CHUNK_BYTES: usize = 32;
/// PoM v4: one threadgroup per nonce, 32 threads (thread x owns state row x). Matches the
/// `pom_mine_v4` kernel's `threads_per_threadgroup`.
const POM_V4_THREADS: usize = 32;

/// Shared-storage: CPU and GPU see the same unified-memory backing, so the host writes uniforms /
/// reads the winner with no blit — the same choice candle makes for its own transient buffers.
const SHARED_STORAGE: MTLResourceOptions = ObjcMTLResourceOptions(ObjcMTLResourceOptions::StorageModeShared.bits());

/// A synchronous wait alone does not mean Metal executed the buffer successfully: it also returns
/// after an aborted/error completion. Never read a shared output buffer or account the batch until
/// the command status is explicitly Completed.
fn wait_for_command(cmd: &CommandBuffer, context: &str) -> candle_core::Result<()> {
    cmd.wait_until_completed();
    let status = cmd.status();
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    let detail = cmd.error().map(|e| e.into_owned()).unwrap_or_else(|| "Metal supplied no NSError detail".to_string());
    Err(candle_core::Error::Msg(format!("{context}: command buffer ended with status {status:?}: {detail}")))
}

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

// Matches PomV4Uniforms in metal/pom_mine.metal — field order and padding are load-bearing.
#[repr(C)]
struct Uniforms {
    n_tiles: u64, // n_total_chunks / POM_V4_TILE_CHUNKS(32)
    k_steps: u32, // POM_V4_K = 256
    n_tensors: u32,
    p0: u64, // POW-fold words (H3-salted pph)
    p1: u64,
    p2: u64,
    p3: u64,
    s0: u64, // SEED-fold words (v4-salted pph)
    s1: u64,
    s2: u64,
    s3: u64,
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
            .get_function("pom_mine_v4", None)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: get_function: {e}")))?;
        let pipeline = mdev
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: pipeline: {e}")))?;
        #[cfg(test)]
        let debug_pipeline = {
            let f = library
                .get_function("pom_walk_states_v4", None)
                .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: get_function debug: {e}")))?;
            mdev.new_compute_pipeline_state_with_function(&f)
                .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: debug pipeline: {e}")))?
        };
        let queue =
            mdev.new_command_queue().map_err(|e| candle_core::Error::Msg(format!("PoM Metal: command queue: {e}")))?;

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
        let mdev = MtlDevice::system_default()
            .ok_or_else(|| candle_core::Error::Msg("PoM Metal: MTLCreateSystemDefaultDevice returned nil".into()))?;

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
            packs.push(Pack { file_offset: content.tensor_data_offset + info.offset, packed_bytes });
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
            return Err(candle_core::Error::Msg(
                "PoM Metal: buffer.contents() is null for shared-storage buffer".into(),
            ));
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

    /// PoM v4 grind. Search nonces in `[start, start + batch)`; return the lowest nonce whose
    /// `pom_pow_value(fold64(v4_state_root(S_K)))` is `<= target_le`, or None. `batch` must fit in
    /// `u32` so the winner atomic stays a 32-bit tid. ONE THREADGROUP per nonce (32 threads, thread
    /// x owns state row x) — see `pom_mine_v4` in metal/pom_mine.metal. Word derivation matches
    /// `pom_gpu::mine_v4`: POW words = H3-salted pph (`pph_words_for_era(_, true)`), SEED words =
    /// v4-salted pph (`pph_words_v4`).
    pub fn mine_v4(
        &self,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> candle_core::Result<Option<u64>> {
        if batch > u32::MAX as u64 {
            return Err(candle_core::Error::Msg("PoM Metal: batch exceeds u32".into()));
        }
        if batch == 0 {
            return Ok(None);
        }
        if start.checked_add(batch - 1).is_none() {
            return Err(candle_core::Error::Msg(
                "PoM Metal: nonce range crosses u64::MAX; split it before launch".into(),
            ));
        }
        let n_tiles = self.n_total_chunks / crate::pom_v4::POM_V4_TILE_CHUNKS;
        if n_tiles == 0 {
            return Err(candle_core::Error::Msg("PoM Metal: blob too small for the v4 walk".into()));
        }
        let p = crate::pom::pph_words_for_era(pre_pow_hash, true);
        let s = crate::pom::pph_words_v4(pre_pow_hash);
        let t = words4(target_le);
        let uniforms = self.v4_uniforms(n_tiles, &p, &s, timestamp, [t[0], t[1], t[2], t[3]], start, batch as u32);

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
        for buf in &self.resources {
            enc.use_resource(buf, MTLResourceUsage::Read);
        }
        // One threadgroup (POM_V4_THREADS=32) per nonce: total threads = batch*32, tg = 32, so
        // threadgroup_position_in_grid = nonce index, thread_position_in_threadgroup = state row.
        let grid = MTLSize { width: (batch as usize) * POM_V4_THREADS, height: 1, depth: 1 };
        let tg = MTLSize { width: POM_V4_THREADS, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
        // ComputeCommandEncoder::Drop ends the encoder. Dropping explicitly here guarantees it is
        // ended before commit without sending Metal a second endEncoding message.
        drop(enc);
        cmd.commit();
        wait_for_command(&cmd, "PoM Metal")?;

        let w = unsafe { *(winner_buf.contents() as *const u32) };
        Ok(if w == u32::MAX { None } else { Some(start + w as u64) })
    }

    /// Build the v4 uniform block. Shared by mine_v4 and the debug walk (which passes target=0).
    fn v4_uniforms(
        &self,
        n_tiles: u64,
        p: &[u64; 4],
        s: &[u64; 4],
        timestamp: u64,
        t: [u64; 4],
        start: u64,
        n: u32,
    ) -> Uniforms {
        Uniforms {
            n_tiles,
            k_steps: crate::pom_v4::POM_V4_K as u32,
            n_tensors: self.n_tensors,
            p0: p[0],
            p1: p[1],
            p2: p[2],
            p3: p[3],
            s0: s[0],
            s1: s[1],
            s2: s[2],
            s3: s[3],
            time_: timestamp,
            t0: t[0],
            t1: t[1],
            t2: t[2],
            t3: t[3],
            nonce_base: start,
            n_nonces: n,
            _pad: 0,
        }
    }

    /// Test-only: runs the v4 walk on the GPU for `batch` consecutive nonces and returns each
    /// nonce's `fold64(v4_state_root(S_K))`. Same uniform + tensor plumbing as `mine_v4()`, so any
    /// divergence from `pom_v4::build_proof_v4`'s final_state is a kernel/indexing bug.
    #[cfg(test)]
    pub fn debug_walk_states_v4(
        &self,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        start: u64,
        batch: u64,
    ) -> candle_core::Result<Vec<u64>> {
        assert!(batch > 0 && batch <= u32::MAX as u64);
        let n_tiles = self.n_total_chunks / crate::pom_v4::POM_V4_TILE_CHUNKS;
        assert!(n_tiles > 0, "blob too small for the v4 walk");
        let p = crate::pom::pph_words_for_era(pre_pow_hash, true);
        let s = crate::pom::pph_words_v4(pre_pow_hash);
        let uniforms = self.v4_uniforms(n_tiles, &p, &s, timestamp, [0, 0, 0, 0], start, batch as u32);
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
        let grid = MTLSize { width: (batch as usize) * POM_V4_THREADS, height: 1, depth: 1 };
        let tg = MTLSize { width: POM_V4_THREADS, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
        drop(enc); // Drop performs the single required endEncoding before commit.
        cmd.commit();
        wait_for_command(&cmd, "PoM Metal(dbg)")?;

        let mut out = vec![0u64; batch as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(states_buf.contents() as *const u64, out.as_mut_ptr(), batch as usize);
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
    // --wait-ready: this card's walk is resident = the card is set up (idempotent, cheap).
    crate::wait_ready::mark_ready(device_id as u64);
}

fn remove_device_entry<T>(map: &mut HashMap<u32, T>, device_id: u32) -> Option<T> {
    map.remove(&device_id)
}

pub fn uninstall(device_id: u32) {
    let _ = uninstall_released(device_id);
}

fn wait_for_sole_owner<T>(item: &Arc<T>, timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while Arc::strong_count(item) > 1 {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    true
}

fn remove_entry_when_released<T>(
    resident: &mut HashMap<u32, Arc<T>>,
    device_id: u32,
    timeout: std::time::Duration,
) -> bool {
    let Some(item) = remove_device_entry(resident, device_id) else {
        return true;
    };
    if wait_for_sole_owner(&item, timeout) {
        return true;
    }
    resident.insert(device_id, item);
    false
}

/// Remove one Metal walk and report whether its model buffers were actually released.
///
/// `mine_v4` clones the resident miner before dispatching outside the registry lock. Removing the
/// map entry alone therefore does not prove that a synchronous Metal command buffer has completed.
/// Keep the registry locked while the removed Arc drains: that prevents a new walk from acquiring
/// the miner (or a replacement from being installed) until every pre-existing dispatch has dropped
/// its handle. On timeout, put the miner back so the still-live buffers remain registered and a
/// second full model cannot be installed on top of them.
#[must_use = "a false result means a Metal walk still owns model memory; do not reload/free it"]
pub fn uninstall_released(device_id: u32) -> bool {
    let mut resident = match miners().lock() {
        Ok(g) => g,
        Err(_) => {
            log::error!(
                "PoM Metal[gpu{}]: resident-miner registry is poisoned — refusing to release model buffers",
                device_id
            );
            return false;
        }
    };
    if remove_entry_when_released(&mut resident, device_id, std::time::Duration::from_secs(30)) {
        return true;
    }

    // Preserve the registry's ownership on failure. The caller must not load/free model memory,
    // and `ensure_installed` must continue to see this table instead of allocating a duplicate.
    log::error!(
        "PoM Metal[gpu{}]: a walk still holds the miner after 30s — model buffers stay resident and this card will not be rebuilt",
        device_id
    );
    false
}

pub fn is_installed(device_id: u32) -> bool {
    miners().lock().map(|g| g.contains_key(&device_id)).unwrap_or(false)
}

static LOADING: AtomicUsize = AtomicUsize::new(0);

pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed) > 0
}

/// Backend-parity with `pom_gpu::set_inference_paused` (CUDA): slm.rs pauses the walk before it
/// uninstalls a device's miner to reload the inference model, so uninstall() doesn't race a live
/// batch. On Metal the walk is a single synchronous `wait_until_completed()` dispatch (no async
/// batch in flight to protect), so this is a plain flag kept for API symmetry.
static INFERENCE_PAUSED: AtomicUsize = AtomicUsize::new(0);

pub fn set_inference_paused(paused: bool) {
    if paused {
        INFERENCE_PAUSED.fetch_add(1, Ordering::AcqRel);
    } else {
        let _ = INFERENCE_PAUSED.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| Some(n.saturating_sub(1)));
    }
}

pub fn is_inference_paused() -> bool {
    INFERENCE_PAUSED.load(Ordering::Acquire) != 0
}

// ── BACKEND PARITY with the CUDA `pom_gpu` ─────────────────────────────────────────────────────
// slm.rs and miner.rs share one code path across CUDA and Metal, so every symbol they call under
// `all(target_os = "macos", feature = "pom-metal")` must exist here too. These were added on the
// CUDA side (per-GPU inference pause, the operator batch controls) without Metal counterparts, which
// is what has been failing the macOS build since v0.11.9.

/// Per-GPU pause bits, mirroring the CUDA bitmask. A Mac has one GPU in practice, so this is the
/// global flag addressed by ordinal — but keeping the shape identical means the shared call sites
/// stay honest instead of silently pausing the wrong thing if that ever changes.
pub fn set_inference_paused_on(_gpu: usize, paused: bool) {
    set_inference_paused(paused);
}

pub fn inference_paused_for(_device_id: u32) -> bool {
    is_inference_paused()
}

/// Model staging is GPU work too. Record it before allocation begins so an inference drain cannot
/// mistake an empty resident map for an idle Metal device. The thread id permits a staging-time
/// self-test to enter this gate re-entrantly, matching the CUDA backend's contract.
fn installing_devices() -> &'static Mutex<HashMap<u32, std::thread::ThreadId>> {
    static INSTALLING: OnceLock<Mutex<HashMap<u32, std::thread::ThreadId>>> = OnceLock::new();
    INSTALLING.get_or_init(|| Mutex::new(HashMap::new()))
}

struct DeviceInstallGuard {
    device_id: u32,
    owner: std::thread::ThreadId,
}

impl DeviceInstallGuard {
    fn begin(device_id: u32) -> Option<Self> {
        let owner = std::thread::current().id();
        let mut installing = installing_devices().lock().unwrap_or_else(|p| p.into_inner());
        if inference_paused_for(device_id) || installing.contains_key(&device_id) {
            return None;
        }
        installing.insert(device_id, owner);
        Some(Self { device_id, owner })
    }
}

impl Drop for DeviceInstallGuard {
    fn drop(&mut self) {
        let mut installing = installing_devices().lock().unwrap_or_else(|p| p.into_inner());
        if installing.get(&self.device_id) == Some(&self.owner) {
            installing.remove(&self.device_id);
        }
    }
}

/// Pause the Metal walk and wait for a command buffer that already passed the pause check to finish.
/// `mine_v4` owns one cloned Arc until `wait_until_completed` returns, so map+local are the two idle
/// owners and a third owner denotes the in-flight synchronous dispatch.
pub struct InferenceDrainGuard(usize);

impl Drop for InferenceDrainGuard {
    fn drop(&mut self) {
        set_inference_paused_on(self.0, false);
    }
}

pub fn pause_and_drain_for_inference(gpu: usize) -> Option<InferenceDrainGuard> {
    set_inference_paused_on(gpu, true);
    let guard = InferenceDrainGuard(gpu);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let current = std::thread::current().id();
    loop {
        let staged_by_other_thread = installing_devices()
            .lock()
            .map(|installing| installing.get(&(gpu as u32)).map(|owner| owner != &current).unwrap_or(false))
            .unwrap_or(true);
        if !staged_by_other_thread {
            break;
        }
        if std::time::Instant::now() >= deadline {
            log::error!("PoM Metal[gpu{}]: model install did not drain within 30s — refusing GPU inference", gpu);
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let resident = miners().lock().ok().and_then(|map| map.get(&(gpu as u32)).cloned());
    if let Some(miner) = resident {
        while Arc::strong_count(&miner) > 2 {
            if std::time::Instant::now() >= deadline {
                log::error!("PoM Metal[gpu{}]: walk did not drain within 30s — refusing GPU inference", gpu);
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    Some(guard)
}

/// `--only-inference`: serve requests, barely mine. Same contract as the CUDA side.
static ONLY_INFERENCE: AtomicBool = AtomicBool::new(false);

pub fn set_only_inference(on: bool) {
    ONLY_INFERENCE.store(on, Ordering::Relaxed);
}

pub fn only_inference() -> bool {
    ONLY_INFERENCE.load(Ordering::Relaxed)
}

pub fn only_inference_duty_ms() -> u64 {
    std::env::var("KERYX_ONLY_INFERENCE_DUTY_MS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(250)
        .clamp(0, 10_000)
}

/// Yields the card to an inference request while the guard lives (`--only-inference` only).
pub struct ServingGuard(usize);

impl Drop for ServingGuard {
    fn drop(&mut self) {
        set_inference_paused_on(self.0, false);
    }
}

pub fn serving_now(gpu: usize) -> Option<ServingGuard> {
    if !only_inference() {
        return None;
    }
    set_inference_paused_on(gpu, true);
    Some(ServingGuard(gpu))
}

/// The v4 grind batch for a Metal device. There is no SM count to derive from here and no autotune
/// on this backend, so this is the plain default the Metal walk has always used; `--only-inference`
/// still drops it to the floor so the card stays free to serve.
pub fn v4_batch_for_device(_device_id: u32) -> u64 {
    if only_inference() {
        return 8192;
    }
    std::env::var("KERYX_POM_V4_BATCH")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&b| b > 0)
        .unwrap_or(1 << 16)
}

/// Convenience: search a nonce batch via the installed miner for a specific device (PoM v4).
pub fn mine_v4(
    device_id: u32,
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    target_le: &[u8; 32],
    start: u64,
    batch: u64,
    daa: u64,
) -> crate::pom::GrindResult {
    if inference_paused_for(device_id) {
        return Err(crate::pom::GrindError::Paused("Metal walk paused for GPU inference"));
    }
    // 🔴 H10 HARD SAFETY GUARD (mirrors the CUDA backend). At/after the H10 gate the walk seed is
    // the one-way cSHAKE256 PowHash (keccak-f1600 absorbing RAW pph words + timestamp + nonce),
    // which is computed INSIDE the walk kernel because the nonce varies per thread. The Metal
    // shader only implements the pre-H10 reversible v4 fold, so grinding an H10-era job here would
    // walk the wrong tiles and every share would be rejected (BadTilePath) while still burning the
    // GPU. Refuse instead: return Paused so the worker idles honestly (no phantom hashrate, nothing
    // counted) and log once with the reason. Pre-H10 / testnet jobs below the gate are unaffected.
    // Lifting this needs keccak-f1600 + the H10 seed fold ported into the .metal shader and
    // validated against pom.rs's golden vectors (see `h10_seed_properties_and_golden`).
    if crate::pom::is_h10_seed_era(daa) {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log::error!(
                "PoM[metal{}]: block DAA {} is in the H10 one-way-seed era, which the Metal walk \
                 shader does not implement — NOT mining (it would produce 100% rejected shares). \
                 Use the CUDA build for H10-era mining.",
                device_id,
                daa
            );
        }
        return Err(crate::pom::GrindError::Paused("Metal H10 seed unsupported"));
    }
    let miner = {
        let Ok(g) = miners().lock() else {
            return Err(crate::pom::GrindError::Backend("resident Metal miner mutex poisoned".into()));
        };
        let Some(miner) = g.get(&device_id) else {
            return Err(crate::pom::GrindError::Paused("Metal walk not installed"));
        };
        miner.clone()
    };
    if inference_paused_for(device_id) {
        return Err(crate::pom::GrindError::Paused("Metal walk paused for GPU inference"));
    }
    match miner.mine_v4(pre_pow_hash, timestamp, target_le, start, batch) {
        Ok(winner) => Ok(crate::pom::GrindCompleted { winner, hashes_done: batch }),
        Err(e) => {
            log::warn!("PoM[metal{}]: v4 batch failed ({e}) — batch aborted and not counted", device_id);
            Err(crate::pom::GrindError::Backend(e.to_string()))
        }
    }
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
    let _install_guard = match DeviceInstallGuard::begin(device_id) {
        Some(guard) => guard,
        None => return false,
    };
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
            let ready = std::path::Path::new(gguf).parent().map(|d| d.join(".ok")).map_or(false, |p| p.exists());
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
            match crate::pom::WeightIndex::build_from_gguf(gguf, *model_id) {
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
                        log::error!(
                            "PoM Metal: possession-index build failed on gpu{}: {} (retrying each job; rate-limited).",
                            device_id,
                            e
                        );
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
                    log::error!(
                        "PoM Metal[gpu{}]: gather N={} != shared index N={} — refusing to mine",
                        device_id,
                        n,
                        idx.n_chunks
                    );
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
    use crate::{pom, pom_v4};

    #[test]
    fn release_barrier_waits_for_an_existing_walk_owner() {
        let miner = Arc::new(());
        let in_flight = Arc::clone(&miner);
        let walk = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            drop(in_flight);
        });

        assert!(wait_for_sole_owner(&miner, std::time::Duration::from_secs(1)));
        walk.join().unwrap();
        assert_eq!(Arc::strong_count(&miner), 1);
    }

    #[test]
    fn release_barrier_times_out_while_a_walk_owner_survives() {
        let miner = Arc::new(());
        let _in_flight = Arc::clone(&miner);

        assert!(!wait_for_sole_owner(&miner, std::time::Duration::from_millis(10)));
    }

    #[test]
    fn remove_device_entry_returns_only_the_target_owner() {
        let mut resident = HashMap::from([(0, "gpu0"), (1, "gpu1")]);

        assert_eq!(remove_device_entry(&mut resident, 0), Some("gpu0"));
        assert_eq!(resident.get(&1), Some(&"gpu1"));
        assert_eq!(remove_device_entry(&mut resident, 0), None);
    }

    #[test]
    fn timed_out_release_restores_the_resident_entry() {
        let miner = Arc::new(());
        let in_flight = Arc::clone(&miner);
        let mut resident = HashMap::from([(0, miner)]);

        assert!(!remove_entry_when_released(&mut resident, 0, std::time::Duration::from_millis(10)));
        assert!(resident.contains_key(&0), "a busy miner must remain registered");

        drop(in_flight);
        assert!(remove_entry_when_released(&mut resident, 0, std::time::Duration::from_millis(10)));
        assert!(!resident.contains_key(&0));
    }

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
        let buf =
            mdev.new_buffer_with_data(bytes.as_ptr() as *const _, bytes.len(), SHARED_STORAGE).expect("blob buffer");
        let addr = buf.as_ref().gpuAddress();
        let prefix = [0u64, n_chunks];
        let addrs = [addr];
        PomGpuMiner::build(mdev, &prefix, &addrs, vec![buf], n_chunks).expect("build test miner")
    }

    /// Byte-exact CPU oracle for the v4 walk over a raw synthetic blob (no WeightIndex needed).
    /// Reproduces pom_v4::build_proof_v4's derivation using the crate's pub v4 primitives:
    /// v4_initial_state → 256×(read tile, transition, next_offset) → fold64(v4_state_root).
    fn v4_final_state(bytes: &[u8], n_chunks: u64, seed: u64) -> u64 {
        use crate::pom_v4::*;
        let n_tiles = n_chunks / POM_V4_TILE_CHUNKS;
        let mut state = v4_initial_state(seed);
        let mut off = v4_first_offset(seed, n_tiles);
        for step in 1..=POM_V4_K as u64 {
            let mut tile = vec![0u8; POM_V4_TILE_BYTES];
            for c in 0..POM_V4_TILE_CHUNKS {
                let ci = (off * POM_V4_TILE_CHUNKS + c) as usize;
                tile[c as usize * 32..c as usize * 32 + 32].copy_from_slice(&bytes[ci * 32..ci * 32 + 32]);
            }
            let snippet: [u8; 32] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
            state = v4_transition(&state, &tile, step as u32);
            if step < POM_V4_K as u64 {
                off = v4_next_offset(seed, step, &snippet, n_tiles);
            }
        }
        fold64(&v4_state_root(&state))
    }

    /// v4 pow value over a synthetic blob for a given nonce (POW words = H3-salted pph, matching
    /// mine_v4 / pom_gpu::mine_v4).
    fn v4_pow_value(bytes: &[u8], n_chunks: u64, pph: &[u8; 32], ts: u64, nonce: u64) -> [u8; 32] {
        let seed = pom::pom_block_seed_v4(pph, ts, nonce);
        pom::pom_pow_value(v4_final_state(bytes, n_chunks, seed), pph, true)
    }

    fn hex32(b: &[u8; 32]) -> String {
        let mut s = String::with_capacity(64);
        for x in b {
            s.push_str(&format!("{:02x}", x));
        }
        s
    }

    /// On-device diagnostic (real Gemma/Mistral GGUF): production `PomGpuMiner::load` packs the raw
    /// GGUF bytes into ONE Metal buffer, then we assert:
    ///   1. packed buffer chunks == host `WeightIndex::read_chunk` at sampled offsets;
    ///   2. `debug_walk_states_v4` (real Metal v4 kernel) == host `pom_v4::build_proof_v4` final_state
    ///      for spot nonces + a batch sweep;
    ///   3. `mine_v4` winner path returns the correct lowest nonce for a target from a real nonce.
    /// Gated by `KERYX_TEST_GGUF` + `#[ignore]`.
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
        eprintln!("building host WeightIndex from {gguf} …");
        let idx = pom::WeightIndex::build_from_gguf(&gguf, [0u8; 32]).expect("host index");
        let miner = PomGpuMiner::load(&gguf, 0).expect("PomGpuMiner::load");
        assert_eq!(miner.n_chunks(), idx.n_chunks, "N mismatch");
        let n = miner.n_chunks();

        // (1) packed buffer vs host WeightIndex at sampled offsets.
        eprintln!("── packed buffer vs host WeightIndex ──");
        let mut sample_offs: Vec<u64> = vec![0, 1, n - 1, n / 2];
        for i in 1..16 {
            sample_offs.push((n * i) / 16);
        }
        sample_offs.sort_unstable();
        sample_offs.dedup();
        let base_ptr = miner.resources[0].contents() as *const u8;
        let mut mismatches = 0usize;
        for &off in &sample_offs {
            let mut buf = [0u8; 32];
            unsafe { std::ptr::copy_nonoverlapping(base_ptr.add(off as usize * 32), buf.as_mut_ptr(), 32) };
            let host = pom::words_to_bytes(&idx.read_chunk(off));
            let ok = buf == host;
            eprintln!("  off={off:>10}  {}  {}", hex32(&buf), if ok { "OK" } else { "MISMATCH" });
            if !ok {
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "packed buffer differs from host WeightIndex");

        // (2) Metal v4 kernel final_state == host build_proof_v4 final_state.
        eprintln!("── Metal v4 walk vs host build_proof_v4 (5 nonces) ──");
        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(7));
        let ts: u64 = 0x1122_3344_5566_7788;
        for &nonce in &[0u64, 1, 42, 100_000, 12345678] {
            let seed = pom::pom_block_seed_v4(&pph, ts, nonce);
            let (_, host_fs) = pom_v4::build_proof_v4(1, seed, &idx).expect("host build_proof_v4");
            let gpu = miner.debug_walk_states_v4(&pph, ts, nonce, 1).expect("debug_walk_states_v4")[0];
            eprintln!(
                "  nonce={nonce:<10} host=0x{host_fs:016x} metal=0x{gpu:016x} {}",
                if host_fs == gpu { "OK" } else { "MISMATCH" }
            );
            assert_eq!(host_fs, gpu, "v4 walk diverges for nonce {nonce}");
        }

        // (2b) batch sweep.
        let (sweep_start, sweep_batch) = (500_000u64, 512u64);
        let gpu = miner.debug_walk_states_v4(&pph, ts, sweep_start, sweep_batch).expect("sweep");
        let mut diverge = 0usize;
        for i in 0..sweep_batch {
            let seed = pom::pom_block_seed_v4(&pph, ts, sweep_start + i);
            let (_, host_fs) = pom_v4::build_proof_v4(1, seed, &idx).expect("host");
            if host_fs != gpu[i as usize] {
                diverge += 1;
            }
        }
        assert_eq!(diverge, 0, "v4 batch sweep diverges ({diverge}/{sweep_batch})");

        // (3) mine_v4 winner path.
        let n_star = sweep_start + sweep_batch / 3;
        let seed_star = pom::pom_block_seed_v4(&pph, ts, n_star);
        let (_, fs_star) = pom_v4::build_proof_v4(1, seed_star, &idx).expect("host");
        let target = pom::pom_pow_value(fs_star, &pph, true);
        let expected = (sweep_start..sweep_start + sweep_batch).find(|&nn| {
            let seed = pom::pom_block_seed_v4(&pph, ts, nn);
            let (_, fs) = pom_v4::build_proof_v4(1, seed, &idx).expect("host");
            pom::le_leq(&pom::pom_pow_value(fs, &pph, true), &target)
        });
        let got = miner.mine_v4(&pph, ts, &target, sweep_start, sweep_batch).expect("mine_v4");
        eprintln!("  mine_v4 → {got:?}, expected {expected:?}");
        assert_eq!(got, expected, "mine_v4 winner disagrees with host reference");
    }

    /// Multi-tensor v4 byte-exact guard: 12 unevenly sized tensors in their own MTLBuffers, so the
    /// per-lane `v4_load_tile` prefix search + tile reads that span tensor boundaries are exercised.
    /// Checks both the raw walk final_state (debug_walk_states_v4) and the mine_v4 winner vs the CPU
    /// v4 oracle over the concatenated bytes. Runs on any Apple Silicon without a GGUF.
    #[test]
    fn metal_walk_matches_host_reference_multi_tensor() {
        let per_tensor: [usize; 12] = [
            5 * 32,
            17 * 32,
            41 * 32,
            128 * 32,
            257 * 32,
            511 * 32,
            1024 * 32,
            2001 * 32,
            4096 * 32,
            512 * 32,
            91 * 32,
            331 * 32,
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

        // Walk parity: Metal final_state == CPU v4 oracle for a batch of nonces (tiles cross the
        // 12 tensor boundaries, so v4_load_tile's per-lane prefix search is genuinely exercised).
        let (wstart, wbatch) = (321u64, 256u64);
        let gpu = miner.debug_walk_states_v4(&pph, ts, wstart, wbatch).expect("walk states");
        let mut diverge = 0usize;
        for i in 0..wbatch {
            let seed = pom::pom_block_seed_v4(&pph, ts, wstart + i);
            if v4_final_state(&bytes, n_chunks, seed) != gpu[i as usize] {
                diverge += 1;
            }
        }
        assert_eq!(diverge, 0, "multi-tensor v4 walk final_state diverges ({diverge}/{wbatch})");

        // Winner path.
        let (start, batch) = (777u64, 8192u64);
        let n_star = start + batch / 2;
        let target = v4_pow_value(&bytes, n_chunks, &pph, ts, n_star);
        let expected =
            (start..start + batch).find(|&nn| pom::le_leq(&v4_pow_value(&bytes, n_chunks, &pph, ts, nn), &target));
        let got = miner.mine_v4(&pph, ts, &target, start, batch).expect("metal mine_v4");
        assert_eq!(got, expected, "multi-tensor v4 Metal winner != host reference");
        assert!(got.is_some(), "n_star must be a winner");
    }

    /// Single-buffer v4 guard (the real model's packed-buffer path): synthetic blob, Metal walk +
    /// winner vs the CPU v4 oracle.
    #[test]
    fn metal_walk_matches_host_reference() {
        let n_chunks = 4096u64; // 128 tiles
        let mut bytes = vec![0u8; n_chunks as usize * CHUNK_BYTES];
        let mut s = 0x1234_5678_9abc_def0u64;
        for b in bytes.iter_mut() {
            s = pom::mix64(s);
            *b = (s & 0xff) as u8;
        }
        let miner = miner_from_blob(&bytes);
        assert_eq!(miner.n_chunks(), n_chunks);

        let pph: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        let ts: u64 = 0x00ff_00ff_1234u64;

        // final_state parity for spot nonces.
        for &nonce in &[0u64, 1, 7, 4242, 999_999] {
            let seed = pom::pom_block_seed_v4(&pph, ts, nonce);
            let host_fs = v4_final_state(&bytes, n_chunks, seed);
            let gpu = miner.debug_walk_states_v4(&pph, ts, nonce, 1).expect("walk")[0];
            assert_eq!(host_fs, gpu, "v4 final_state diverges for nonce {nonce}");
        }

        // Winner path — LOWEST nonce whose pow_value <= target(n_star).
        let (start, batch) = (1_000u64, 4_000u64);
        let n_star = start + batch / 2;
        let target = v4_pow_value(&bytes, n_chunks, &pph, ts, n_star);
        let expected =
            (start..start + batch).find(|&nn| pom::le_leq(&v4_pow_value(&bytes, n_chunks, &pph, ts, nn), &target));
        let got = miner.mine_v4(&pph, ts, &target, start, batch).expect("metal mine_v4");
        assert_eq!(got, expected, "v4 Metal winner != host reference");
        assert!(got.is_some() && got.unwrap() <= n_star);
    }
}
