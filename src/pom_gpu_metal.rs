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
        let queue = mdev
            .new_command_queue()
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: command queue: {e}")))?;

        Ok(Self {
            device: mdev,
            queue,
            pipeline,
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
    pub fn mine(
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
        let p = words4(pre_pow_hash);
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
) -> Option<u64> {
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    miner.mine(pre_pow_hash, timestamp, target_le, start, batch).ok().flatten()
}

/// Mining-tier identity for rebuilds: (model_id, gguf_path). Set once at startup. Global (not
/// per-device) to match the CUDA backend's surface.
static MINING_TIER: OnceLock<([u8; 32], String)> = OnceLock::new();

pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String) {
    let _ = MINING_TIER.set((model_id, gguf_path));
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

pub fn current_tier(daa: u64) -> Option<u8> {
    let (model_id, _) = MINING_TIER.get()?;
    crate::models::pom_tier_index(model_id, daa)
}

fn ensure_installed_inner(device_id: u32, daa: u64) -> bool {
    let (model_id, gguf) = match MINING_TIER.get() {
        Some(x) => x,
        None => return false,
    };
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
    fn cpu_pow_value(bytes: &[u8], n_chunks: u64, pph: &[u8; 32], ts: u64, nonce: u64) -> [u8; 32] {
        let seed = pom::pom_block_seed(pph, ts, nonce);
        let read = |off: u64| -> [u64; pom::CHUNK_WORDS] {
            let o = (off as usize) * CHUNK_BYTES;
            let mut c = [0u8; 32];
            c.copy_from_slice(&bytes[o..o + 32]);
            pom::chunk_to_words(&c)
        };
        let final_state = pom::walk_final(seed, n_chunks, pom::POM_WALK_STEPS, read);
        pom::pom_pow_value(final_state, pph)
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
        let got = miner.mine(&pph, ts, &target, start, batch).expect("metal mine");
        assert_eq!(got, expected, "Metal winner != host reference");
        assert!(got.is_some(), "n_star must be a winner");
        assert!(got.unwrap() <= n_star);
    }
}
