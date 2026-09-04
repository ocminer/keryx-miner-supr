use std::collections::HashMap;
use std::num::Wrapping;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::thread::sleep;
use std::time::Duration;

use crate::{pow, watch, Error};
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use tokio::sync::mpsc::Sender;
use tokio::task::{self, JoinHandle};
use tokio::time::MissedTickBehavior;

use crate::pow::BlockSeed;
use keryx_miner::{ManagedWorkerSpec, PluginManager};

type MinerHandler = std::thread::JoinHandle<Result<(), Error>>;

/// Process-wide OPoI pause state. Network handlers and `MinerManager`s are replaced on reconnect,
/// while their detached inference tasks can keep running. A connection-owned counter would let a
/// replacement manager resume over one of those old GPU calls.
struct InferencePauseState {
    inflight: usize,
    current_miner: Option<Weak<InferencePauseControl>>,
}

struct InferencePauseControl {
    block_channel: Arc<watch::Sender<Option<WorkerCommand>>>,
    active_flag: Arc<AtomicBool>,
    /// Last valid Stratum job for this exact `MinerManager` generation.  It is
    /// deliberately manager-local: replacing the manager on reconnect creates
    /// an empty slot, so an old connection's template can never be replayed to
    /// the replacement workers.
    resumable_stratum_job: Mutex<Option<WorkerCommand>>,
}

impl InferencePauseControl {
    fn remember_stratum_job(&self, job: WorkerCommand) {
        *self.resumable_stratum_job.lock().unwrap_or_else(|p| p.into_inner()) = Some(job);
    }

    fn forget_stratum_job(&self) {
        self.resumable_stratum_job.lock().unwrap_or_else(|p| p.into_inner()).take();
    }

    /// Resume only this manager's newest retained Stratum job.  The caller
    /// holds `inference_pause_state`, which serializes this send against both a
    /// newer notify and manager replacement.
    fn resume_stratum_job(&self) {
        self.active_flag.store(false, Ordering::SeqCst);
        let job = self.resumable_stratum_job.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if let Some(job) = job {
            if self.block_channel.send(Some(job)).is_err() {
                warn!("OPoI: could not resume the retained Stratum job because all workers exited");
            }
        }
    }
}

fn inference_pause_state() -> &'static Mutex<InferencePauseState> {
    static STATE: OnceLock<Mutex<InferencePauseState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(InferencePauseState { inflight: 0, current_miner: None }))
}

fn lock_inference_pause_state() -> std::sync::MutexGuard<'static, InferencePauseState> {
    inference_pause_state().lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// RAII token held from GPU acquisition until generation has actually ended. The final token clears
/// the flag on whichever miner is live then, even if the pool reconnected during inference.
pub struct InferencePauseGuard {
    armed: bool,
}

impl Drop for InferencePauseGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = lock_inference_pause_state();
        debug_assert!(state.inflight > 0, "OPoI inference pause counter underflow");
        state.inflight = state.inflight.saturating_sub(1);
        if state.inflight == 0 {
            keryx_miner::runtime_stats::inference_pause_ended();
            if let Some(control) = state.current_miner.as_ref().and_then(Weak::upgrade) {
                control.resume_stratum_job();
            }
        }
        self.armed = false;
    }
}

/// One slot in the bounded detached proof-builder pool. Acquisition is atomic (the old
/// load-then-increment check could oversubscribe under simultaneous GPU hits), and Drop releases
/// the slot even if proof I/O panics and the detached closure unwinds.
#[cfg(any(feature = "pom-opencl", feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
struct InflightProofPermit {
    counter: Arc<AtomicUsize>,
}

#[cfg(any(feature = "pom-opencl", feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
impl InflightProofPermit {
    fn try_acquire(counter: &Arc<AtomicUsize>, limit: usize) -> Option<Self> {
        let mut current = counter.load(Ordering::Acquire);
        loop {
            if current >= limit {
                return None;
            }
            match counter.compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(Self { counter: Arc::clone(counter) }),
                Err(observed) => current = observed,
            }
        }
    }
}

#[cfg(any(feature = "pom-opencl", feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
impl Drop for InflightProofPermit {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Stop the currently-live miner and hold the process-wide job gate until the returned token drops.
/// The synchronous stop is ordered with job dispatch through the same mutex, closing the race where
/// a notify arrives between a detached task deciding to pause and its GPU call beginning.
pub fn begin_inference_pause() -> InferencePauseGuard {
    let mut state = lock_inference_pause_state();
    let was_idle = state.inflight == 0;
    state.inflight = state.inflight.saturating_add(1);
    if was_idle {
        keryx_miner::runtime_stats::inference_pause_started();
    }
    if let Some(control) = state.current_miner.as_ref().and_then(Weak::upgrade) {
        control.active_flag.store(true, Ordering::SeqCst);
        let _ = control.block_channel.send(None);
    }
    InferencePauseGuard { armed: true }
}

pub fn inference_pause_active() -> bool {
    lock_inference_pause_state().inflight != 0
}

// How long to wait for a worker to exit after it is asked to Close before we
// assume it is frozen and force-kill it with SIGUSR1. Must comfortably exceed a
// cold GPU-kernel JIT compile: some archs (e.g. AMD gfx1102) ship no precompiled
// binary and compile the kernel from source at startup, which takes a few
// seconds. If the pool drops *during* that compile the worker is busy, not
// frozen — and force-killing a thread that is inside the GPU runtime's compiler
// raises a non-unwinding panic (signal_panic) that aborts the whole process
// before the reconnect loop can retry. The old 1s grace fired mid-compile.
// kill_switch is cleared the instant join() returns, so a healthy worker (which
// exits within ~100ms of Close) never waits anywhere near this long.
#[cfg(any(target_os = "linux", target_os = "macos"))]
const FREEZE_GRACE: Duration = Duration::from_secs(30);
#[cfg(any(target_os = "linux", target_os = "macos"))]
const FREEZE_POLL: Duration = Duration::from_millis(50);

#[cfg(any(target_os = "linux", target_os = "macos"))]
extern "C-unwind" fn signal_panic(_signal: nix::libc::c_int) {
    // MUST be `extern "C-unwind"` (upstream 13d9515): a plain `extern "C"` handler turns
    // this panic into a process-wide abort ("panic in a function that cannot unwind") — so
    // force-killing ONE genuinely frozen worker nuked the whole miner, and on HiveOS the
    // agent relaunch turned that into a shutdown crash-loop (worst during an OPoI inference
    // reload). Unwinding lets the stuck worker's join() return an Err instead.
    panic!("Forced shutdown");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn register_freeze_handler() {
    // nix's typed SigHandler only accepts `extern "C" fn`, which would reintroduce the
    // abort. Register through libc with a transmute: the C and C-unwind ABIs share the
    // same calling convention (ABI-sound), and unwind behavior follows the handler's own
    // `extern "C-unwind"` definition.
    unsafe {
        let handler: nix::libc::sighandler_t =
            std::mem::transmute(signal_panic as extern "C-unwind" fn(nix::libc::c_int));
        let _ = nix::libc::signal(nix::libc::SIGUSR1, handler);
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn trigger_freeze_handler(kill_switch: Arc<AtomicBool>, handle: &MinerHandler) -> std::thread::JoinHandle<()> {
    use std::os::unix::thread::JoinHandleExt;
    let pthread_handle = handle.as_pthread_t();
    std::thread::spawn(move || {
        // Wait up to FREEZE_GRACE for the worker to exit on its own. kill_switch
        // is cleared by Drop the moment join() returns, so poll it and bail out
        // early — a worker that finishes a slow startup compile and then honours
        // the Close command must NOT be force-killed. Only a worker still alive
        // after the full grace is treated as genuinely frozen.
        let mut waited = Duration::ZERO;
        while waited < FREEZE_GRACE {
            if !kill_switch.load(Ordering::SeqCst) {
                return; // worker exited cleanly; nothing to kill
            }
            sleep(FREEZE_POLL);
            waited += FREEZE_POLL;
        }
        if kill_switch.load(Ordering::SeqCst) {
            warn!(
                "Worker did not exit within {}s of shutdown — force-killing (assumed frozen)",
                FREEZE_GRACE.as_secs()
            );
            match nix::sys::pthread::pthread_kill(pthread_handle, nix::sys::signal::Signal::SIGUSR1) {
                Ok(()) => {
                    info!("Thread killed successfully")
                }
                Err(e) => {
                    info!("Error: {:?}", e)
                }
            }
        }
    })
}

#[cfg(any(target_os = "windows"))]
struct RawHandle(*mut std::ffi::c_void);

#[cfg(any(target_os = "windows"))]
unsafe impl Send for RawHandle {}

#[cfg(any(target_os = "windows"))]
fn register_freeze_handler() {}

#[cfg(target_os = "windows")]
fn trigger_freeze_handler(kill_switch: Arc<AtomicBool>, handle: &MinerHandler) -> std::thread::JoinHandle<()> {
    use std::os::windows::io::AsRawHandle;
    let raw_handle = RawHandle(handle.as_raw_handle());

    std::thread::spawn(move || unsafe {
        let ensure_full_move = raw_handle;
        sleep(Duration::from_millis(1000));
        if kill_switch.load(Ordering::SeqCst) {
            kernel32::TerminateThread(ensure_full_move.0, 0);
        }
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn trigger_freeze_handler(kill_switch: Arc<AtomicBool>, handle: &MinerHandler) {
    warn!("Freeze handler is not implemented. Frozen threads are ignored");
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn register_freeze_handler() {
    warn!("Freeze handler is not implemented. Frozen threads are ignored");
}

#[derive(Clone)]
enum WorkerCommand {
    Job(Box<pow::State>),
    Close,
}

/// One device's current hashrate (hashes/sec); `label` is e.g. "#0 (NVIDIA GeForce RTX 5090)".
/// `health` is the NVML sample taken in the same reporting tick (None on non-NVIDIA rigs
/// or when the CUDA↔NVML device mapping can't be verified); `efficiency_mhs_per_w` is
/// hashrate/power from that same tick — the field users tune against.
#[derive(Default, Clone)]
pub struct DeviceRate {
    pub label: String,
    pub hashrate: f64,
    pub health: Option<crate::gpu_health::GpuHealth>,
    pub efficiency_mhs_per_w: Option<f64>,
}

/// Live hashrate snapshot, refreshed by the logger every LOG_RATE. Read by the stats API.
#[derive(Default, Clone)]
pub struct MinerStats {
    pub total_hashrate: f64,
    pub devices: Vec<DeviceRate>,
    /// Sum of NVML power across devices with a verified sample (watts); None if no device
    /// reported. Basis for the rig-level efficiency figure.
    pub total_power_w: Option<f64>,
    pub total_efficiency_mhs_per_w: Option<f64>,
}

#[allow(dead_code)]
pub struct MinerManager {
    handles: Vec<MinerHandler>,
    block_channel: Arc<watch::Sender<Option<WorkerCommand>>>,
    send_channel: Sender<BlockSeed>,
    logger_handle: JoinHandle<()>,
    is_synced: bool,
    hashes_tried: Arc<AtomicU64>,
    hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    current_state_id: AtomicUsize,
    opoi_challenge_active: Arc<AtomicBool>,
    pause_control: Arc<InferencePauseControl>,
    stats: Arc<Mutex<MinerStats>>,
}

impl MinerManager {
    /// Shared live hashrate snapshot (for the stats API).
    pub fn stats(&self) -> Arc<Mutex<MinerStats>> {
        Arc::clone(&self.stats)
    }
}

impl Drop for MinerManager {
    fn drop(&mut self) {
        info!("Closing miner");
        // Stop routing new global pauses to workers that are being torn down. Detached guards remain
        // live and will operate on the replacement manager registered by `new`.
        {
            let mut state = lock_inference_pause_state();
            let is_current = state
                .current_miner
                .as_ref()
                .and_then(Weak::upgrade)
                .map_or(false, |current| Arc::ptr_eq(&current, &self.pause_control));
            if is_current {
                state.current_miner = None;
            }
        }
        self.logger_handle.abort();
        match self.block_channel.send(Some(WorkerCommand::Close)) {
            Ok(_) => {}
            Err(_) => warn!("All workers are already dead"),
        }
        while !self.handles.is_empty() {
            let handle = self.handles.pop().expect("There should be at least one");
            let kill_switch = Arc::new(AtomicBool::new(true));
            trigger_freeze_handler(kill_switch.clone(), &handle);
            match handle.join() {
                Ok(res) => match res {
                    Ok(()) => {}
                    Err(e) => error!("Error when closing Worker: {}", e),
                },
                Err(_) => error!("Worker failed to close gracefully"),
            };
            kill_switch.fetch_and(false, Ordering::SeqCst);
        }
    }
}

pub fn get_num_cpus(n_cpus: Option<u16>) -> u16 {
    n_cpus.unwrap_or_else(|| {
        num_cpus::get_physical().try_into().expect("Doesn't make sense to have more than 65,536 CPU cores")
    })
}

const LOG_RATE: Duration = Duration::from_secs(10);

impl MinerManager {
    pub fn new(send_channel: Sender<BlockSeed>, n_cpus: Option<u16>, manager: &PluginManager) -> Self {
        Self::new_with_stats(send_channel, n_cpus, manager, Arc::new(Mutex::new(MinerStats::default())))
    }

    /// Construct a mining session that publishes into a process-lifetime statistics snapshot.
    ///
    /// Pool reconnects replace `MinerManager`, but the API and TUI must keep observing the same
    /// `Arc`; otherwise their uptime/history survives while their hashrate silently freezes on the
    /// manager from the first connection.
    pub fn new_with_stats(
        send_channel: Sender<BlockSeed>,
        n_cpus: Option<u16>,
        manager: &PluginManager,
        stats: Arc<Mutex<MinerStats>>,
    ) -> Self {
        register_freeze_handler();
        let hashes_tried = Arc::new(AtomicU64::new(0));
        let hashes_by_worker = Arc::new(Mutex::new(HashMap::<String, Arc<AtomicU64>>::new()));
        let opoi_challenge_active = Arc::new(AtomicBool::new(false));
        let (send, recv) = watch::channel(None);
        let send = Arc::new(send);
        let pause_control = Arc::new(InferencePauseControl {
            block_channel: Arc::clone(&send),
            active_flag: Arc::clone(&opoi_challenge_active),
            resumable_stratum_job: Mutex::new(None),
        });
        {
            let mut state = lock_inference_pause_state();
            if state.inflight != 0 {
                opoi_challenge_active.store(true, Ordering::SeqCst);
            }
            state.current_miner = Some(Arc::downgrade(&pause_control));
        }
        let mut handles =
            Self::launch_cpu_threads(send_channel.clone(), Arc::clone(&hashes_tried), recv.clone(), n_cpus)
                .collect::<Vec<MinerHandler>>();
        if manager.has_specs() {
            handles.append(&mut Self::launch_gpu_threads(
                send_channel.clone(),
                Arc::clone(&hashes_tried),
                recv,
                manager,
                hashes_by_worker.clone(),
            ));
        }
        Self {
            handles,
            block_channel: send,
            send_channel,
            logger_handle: task::spawn(Self::log_hashrate(
                Arc::clone(&hashes_tried),
                hashes_by_worker.clone(),
                Arc::clone(&opoi_challenge_active),
                Arc::clone(&stats),
            )),
            is_synced: true,
            hashes_tried,
            current_state_id: AtomicUsize::new(0),
            hashes_by_worker,
            opoi_challenge_active,
            pause_control,
            stats,
        }
    }

    fn launch_cpu_threads(
        send_channel: Sender<BlockSeed>,
        hashes_tried: Arc<AtomicU64>,
        work_channel: watch::Receiver<Option<WorkerCommand>>,
        n_cpus: Option<u16>,
    ) -> impl Iterator<Item = MinerHandler> {
        let n_cpus = get_num_cpus(n_cpus);
        info!("launching: {} cpu miners", n_cpus);
        (0..n_cpus)
            .map(move |_| Self::launch_cpu_miner(send_channel.clone(), work_channel.clone(), Arc::clone(&hashes_tried)))
    }

    fn launch_gpu_threads(
        send_channel: Sender<BlockSeed>,
        hashes_tried: Arc<AtomicU64>,
        work_channel: watch::Receiver<Option<WorkerCommand>>,
        manager: &PluginManager,
        hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    ) -> Vec<MinerHandler> {
        let mut vec = Vec::<MinerHandler>::new();
        let specs = manager.build().unwrap();
        for spec in specs {
            let worker_hashes_tried = Arc::new(AtomicU64::new(0));
            hashes_by_worker.lock().unwrap().insert(spec.id(), worker_hashes_tried.clone());
            vec.push(Self::launch_gpu_miner(
                send_channel.clone(),
                work_channel.clone(),
                Arc::clone(&hashes_tried),
                spec,
                worker_hashes_tried,
            ));
        }
        vec
    }

    pub fn opoi_challenge_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.opoi_challenge_active)
    }

    pub async fn process_block(&mut self, block: Option<BlockSeed>) -> Result<(), Error> {
        let state = match block {
            Some(b) => {
                // Only Stratum jobs are `PartialBlock`s.  Retaining a solo/gRPC
                // `FullBlock` would change its template lifecycle: gRPC already
                // requests one fresh template after inference completes.
                let is_stratum_job = matches!(&b, BlockSeed::PartialBlock { .. });
                let id = self.current_state_id.fetch_add(1, Ordering::SeqCst);
                let command = WorkerCommand::Job(Box::new(pow::State::new(id, b)?));
                // Serialize job publication with `begin_inference_pause`. If inference already owns
                // the gate, retain the newest valid Stratum command and keep workers on `None`. If
                // inference begins immediately after this send, its begin path publishes `None`
                // while holding this same mutex, so either order ends paused before generation.
                let pause_state = lock_inference_pause_state();
                if is_stratum_job {
                    self.pause_control.remember_stratum_job(command.clone());
                } else {
                    self.pause_control.forget_stratum_job();
                }
                if pause_state.inflight != 0 {
                    self.opoi_challenge_active.store(true, Ordering::SeqCst);
                    // A valid Stratum template is resumable when the outermost
                    // guard drops.  Solo/gRPC deliberately retains its old
                    // behavior: it remains unsynced and waits for the fresh
                    // template its client requests after inference.
                    self.is_synced = is_stratum_job;
                    self.block_channel.send(None).map_err(|_e| "Failed sending block to threads")?;
                    return Ok(());
                }
                self.is_synced = true;
                let state = Some(command);
                self.block_channel.send(state).map_err(|_e| "Failed sending block to threads")?;
                drop(pause_state);
                return Ok(());
            }
            None => {
                // A real Stratum suspension (unsynced/unserveable), unlike the
                // inference pause's direct channel send, invalidates the replay
                // candidate. Serialize it with final-guard replay so whichever
                // event is newer is the command workers observe last.
                let _pause_state = lock_inference_pause_state();
                self.pause_control.forget_stratum_job();
                if !self.is_synced {
                    return Ok(());
                }
                self.is_synced = false;
                if self.opoi_challenge_active.load(Ordering::Relaxed) {
                    info!("OPoI challenge in progress — PoW template suspended, stand by");
                } else {
                    warn!("Keryxd is not synced, skipping current template");
                }
                None
            }
        };

        self.block_channel.send(state).map_err(|_e| "Failed sending block to threads")?;
        Ok(())
    }

    #[allow(unreachable_code)]
    fn launch_gpu_miner(
        send_channel: Sender<BlockSeed>,
        mut block_channel: watch::Receiver<Option<WorkerCommand>>,
        hashes_tried: Arc<AtomicU64>,
        spec: ManagedWorkerSpec,
        worker_hashes_tried: Arc<AtomicU64>,
    ) -> MinerHandler {
        std::thread::spawn(move || {
            let no_winner = spec.no_winner();
            let worker_ordinal = crate::gpu_health::ordinal_from_label(&spec.id());
            #[cfg(feature = "pom-opencl")]
            let opencl_device_id = spec.opencl_device_id();
            let mut box_ = spec.build();
            // AMD multi-GPU PoM: bind this thread to its own card so the possession tier is made
            // resident per-GPU and every card mines (and submits) its own shares — instead of all
            // GPU threads funneling onto device 0 through one global lock (3 cards ran like 1).
            #[cfg(feature = "pom-opencl")]
            if let Some(dev) = opencl_device_id {
                keryx_miner::pom_opencl::bind_thread_device(dev);
            }
            let gpu_work = box_.as_mut();
            (|| {
                info!("Spawned Thread for GPU {}", gpu_work.id());
                // --wait-ready: announce this worker so the gate knows the full card set before
                // the first card can finish staging (see wait_ready.rs).
                #[cfg(feature = "pom-opencl")]
                if let Some(d) = opencl_device_id {
                    // cl_device_id is an opaque pointer-sized handle. Preserve every bit in the
                    // readiness key so two OpenCL workers can never alias on 64-bit hosts.
                    keryx_miner::wait_ready::register_device(d as u64);
                }
                #[cfg(any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")))]
                if let Some(d) = gpu_work.id().strip_prefix('#')
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    keryx_miner::wait_ready::register_device(d as u64);
                }
                let mut nonces = vec![no_winner; 1];

                let mut state = None;
                // AMD PoM: cap on proof-builds running concurrently on detached threads (the async
                // overlap that keeps the card grinding while the CPU re-walks the winner). One winner
                // per batch means ~1-2 in flight normally; the cap only guards a pathological
                // low-difficulty burst from spawning unbounded threads (excess winners are dropped —
                // the grind found more than the CPU can prove, which is fine).
                #[cfg(any(feature = "pom-opencl", feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
                let inflight_proofs = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
                #[cfg(any(feature = "pom-opencl", feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
                const MAX_INFLIGHT_PROOFS: usize = 6;
                // PoM (post-fork): nonce cursor + per-launch batch. The kernel grinds the whole
                // batch before returning, so blocks/sec is capped at hashrate / POM_BATCH.
                //
                // Batch sizing (H100 microbench, tools/h100/bench_pom.cu — the walk is memory-latency
                // bound so a batch must supply enough waves to fill the memory pipeline AND amortize
                // the per-launch host round-trip in the driver's mine()): 2^20=122.6, 2^21=124.4,
                // 2^22=124.8, 2^24=125.2 MH/s on an H100. 2^22 captures ~all of the throughput while
                // keeping the per-launch time modest (~33 ms on an H100, ~230 ms on a 3070) so job
                // switching stays responsive. Bumped 2^20 -> 2^22.
                #[cfg(any(feature = "pom-opencl", feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
                let mut pom_cursor: Option<(usize, keryx_miner::pom::MaskedNonceCursor)> = None;
                #[cfg(any(feature = "pom-opencl", feature = "pom-cuda"))]
                // Apple Silicon: at M-class PoM rates (~0.8 MH/s on an M2) the 2^22 batch above is a
                // ~5 s blocking launch — new jobs are only picked up at batch boundaries (see the
                // get_changed() poll after the winner branch), so every winner was submitted ~5-10 s
                // after its job's creation, past the pool's job-retention window. Observed live on
                // stratum-us:4405 (2026-07-16): suprnova's stratum handler substituted its CURRENT
                // job on the lookup miss (getJob(id) || getCurrentJob(), stratum-handler.js:701 —
                // confirmed + fixed pool-side) and rejected with a false PowValueMismatch; shares
                // were accepted only when job rotation happened to be slow. 2^19 keeps the launch
                // ~0.6 s at 840 KH/s — the responsiveness class the 2^22 sizing intended — while
                // costing ~1-2% throughput vs 2^22 (2^18 measured ~8% loss on an M2; 2^19 recovers
                // most of it). The pool side also widened its job-retention window to 25 s, so a
                // one-rotation-late submit now resolves as a share (or an honest stale), never a
                // false PowValueMismatch.
                // (The Apple-specific POM_BATCH const the comment above describes was removed with
                // the v4-only cleanup — its #[cfg] attribute must not dangle onto the seam below.)
                // Driver seam: AMD = OpenCL, NVIDIA = candle-CUDA, Apple Silicon = candle-Metal. All
                // expose the same interface (is_installed / ensure_installed / mine / set_mining_tier).
                // OpenCL wins if both on; Metal is macOS-only (never combined with the others).
                #[cfg(feature = "pom-opencl")]
                use keryx_miner::pom_opencl as pom_driver;
                #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
                use keryx_miner::pom_gpu as pom_driver;
                #[cfg(all(target_os = "macos", feature = "pom-metal", not(feature = "pom-opencl"), not(feature = "pom-cuda")))]
                use keryx_miner::pom_gpu as pom_driver;

                loop {
                    nonces[0] = 0;
                    if state.is_none() {
                        state = match block_channel.wait_for_change() {
                            Ok(cmd) => match cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {return Ok(());}
                                None => None,
                            },
                            Err(e) => {
                                info!("{}: GPU thread crashed: {}", gpu_work.id(), e.to_string());
                                return Ok(());
                            }
                        };
                    }
                    // Poll commands at the top of every retry, not only after a successful PoM
                    // batch. Paused/failed backends have several early `continue` paths; without
                    // this seam a permanent error (or Metal's deliberate H10 pause) could pin a
                    // stale job and ignore WorkerCommand::Close indefinitely.
                    if state.is_some() {
                        if let Some(new_cmd) = block_channel.get_changed()? {
                            state = match new_cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => return Ok(()),
                                None => None,
                            };
                        }
                    }
                    // PoM possession mining (post-fork): grind the data-dependent walk on the GPU
                    // over the resident tier weights instead of kHeavyHash. On a winning nonce we
                    // build the proof (host) and submit; the legacy plugin path below is skipped.
                    // Engage the PoM path when the block is at/after EITHER the base PoM activation
                    // OR the H6 (PoM v3) gate. The v3 gate can be armed BELOW the base activation
                    // (e.g. testnet pom_v3=5000 while base=37.78M), so gating on base alone would keep
                    // running kHeavyHash on an H6 pool and never build a v3 proof.
                    #[cfg(any(feature = "pom-opencl", feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
                    if let Some(s0) = state.as_ref() { let _ = s0;
                        let (job_id, pph, time, target_le, daa, nonce_mask, nonce_fixed) = {
                            let s = state.as_ref().unwrap();
                            let mut pph = [0u8; 32];
                            pph.copy_from_slice(&s.pow_hash_header[0..32]);
                            (
                                s.id,
                                pph,
                                u64::from_le_bytes(s.pow_hash_header[32..40].try_into().unwrap()),
                                s.target.to_le_bytes(),
                                s.daa_score,
                                s.nonce_mask,
                                s.nonce_fixed,
                            )
                        };
                        // PoM v4 (relaunch, keryxd v1.5.1): ONE entry, no era flags, no gating.
                        // The GPU grinds the D=32 int8 matrix re-walk (pom_mine_v4); the host rebuilds
                        // the witness in pow::generate_block_if_pom (byte-exact re-walk of the resident
                        // index), so a kernel false-positive is dropped there, never submitted. v4 is
                        // one CUDA block (32 threads) per nonce — grind a bounded slice per launch
                        // (env KERYX_POM_V4_BATCH) to keep block latency low at 10 BPS. Default 64K:
                        // the tensor-core solver amortizes its per-batch chase/launch cost with batch
                        // size (measured +10% at 64K vs the old 16K on a 5070 Ti), and a 64K batch is
                        // still only ~25 ms on a mid Blackwell card — well under the 100 ms block time.
                        // NVIDIA (CUDA) + Apple Silicon (Metal): per-device v4 grind. Device id = the
                        // worker's "#N (name)" label (per-device MINERS map → no CUDA_VISIBLE_DEVICES).
                        #[cfg(any(feature = "pom-opencl", all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")))]
                        let wdid = worker_ordinal.unwrap_or(0);
                        // Batch: env override wins, else SM-derived per this card (PR #37) — keeps a
                        // launch inside the ~100ms template window at 10 BPS even on small cards.
                        #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
                        let batch = std::env::var("KERYX_POM_V4_BATCH").ok()
                            .and_then(|s| s.trim().parse::<u64>().ok()).filter(|&b| b > 0)
                            .map(|b| keryx_miner::pom_gpu::cap_v4_batch(wdid, b))
                            .unwrap_or_else(|| keryx_miner::pom_gpu::v4_batch_for_device(wdid));
                        #[cfg(all(target_os = "macos", feature = "pom-metal", not(feature = "pom-cuda")))]
                        let batch = std::env::var("KERYX_POM_V4_BATCH").ok()
                            .and_then(|s| s.trim().parse::<u64>().ok()).filter(|&b| b > 0)
                            .unwrap_or_else(|| keryx_miner::pom_gpu::v4_batch_for_device(wdid));
                        #[cfg(not(any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal"))))]
                        let batch = std::env::var("KERYX_POM_V4_BATCH").ok()
                            .and_then(|s| s.trim().parse::<u64>().ok()).filter(|&b| b > 0)
                            .unwrap_or(1 << 16);
                        // Stratum assigns each worker a nonce sub-domain as
                        // `(raw & nonce_mask) | nonce_fixed`. PoM kernels accept physical contiguous
                        // ranges, so enumerate the unique assigned values by rank and split only at
                        // a numeric gap/wrap. Normal low-bit pool masks retain full-sized batches.
                        if pom_cursor.as_ref().map(|(id, _)| *id) != Some(job_id) {
                            pom_cursor = Some((
                                job_id,
                                keryx_miner::pom::MaskedNonceCursor::new(
                                    nonce_mask,
                                    nonce_fixed,
                                    thread_rng().next_u64(),
                                ),
                            ));
                        }
                        let Some(range) = pom_cursor.as_ref().and_then(|(_, c)| c.peek(batch)) else {
                            log::warn!(
                                "PoM: Stratum nonce domain exhausted for job {} — waiting for a new template",
                                job_id
                            );
                            state = None;
                            continue;
                        };
                        #[cfg(any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")))]
                        let grind = {
                            // Metal has not ported the H10 cSHAKE seed yet. Its backend reports a
                            // paused batch, and this earlier guard avoids a pointless call/log loop
                            // while keeping the nonce cursor and accounting untouched.
                            #[cfg(all(target_os = "macos", feature = "pom-metal"))]
                            if keryx_miner::pom::is_h10_seed_era(daa) {
                                static H10_UNSUPPORTED_WARNED: std::sync::atomic::AtomicBool =
                                    std::sync::atomic::AtomicBool::new(false);
                                if !H10_UNSUPPORTED_WARNED.swap(true, Ordering::Relaxed) {
                                    log::error!(
                                        "PoM[metal{}]: H10 one-way seeds are not implemented by the \
                                         Metal shader — mining paused honestly at 0 H/s.",
                                        wdid
                                    );
                                }
                                std::thread::sleep(std::time::Duration::from_millis(1000));
                                continue;
                            }
                            // CRASH GUARD (see pom_gpu::inference_paused_for): while THIS card is
                            // swapping its llama model, neither run a walk nor rebuild one — the
                            // zero-dup walk addresses the engine's weights, so touching them mid
                            // free/reload faults with a STICKY CUDA_ERROR_ILLEGAL_ADDRESS that
                            // poisons the context and takes the process down via ggml_abort.
                            // Swaps last ~2 s; idle honestly (no hashes counted) until it clears.
                            if keryx_miner::pom_gpu::inference_paused_for(wdid) {
                                static SWAP_WAIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                                if SWAP_WAIT.fetch_add(1, Ordering::Relaxed) % 200 == 0 {
                                    log::info!("PoM[gpu{}]: holding off — llama model swap in progress on this card (walk paused, ~2 s).", wdid);
                                }
                                std::thread::sleep(std::time::Duration::from_millis(25));
                                continue;
                            }
                            if !pom_driver::is_installed(wdid) {
                                // Cooperative pre-reload check: act on a pending shutdown / newer job
                                // before a multi-second blocking model reload.
                                if let Some(new_cmd) = block_channel.get_changed()? {
                                    state = match new_cmd {
                                        Some(WorkerCommand::Job(s)) => Some(s),
                                        Some(WorkerCommand::Close) => return Ok(()),
                                        None => None,
                                    };
                                    continue;
                                }
                                pom_driver::ensure_installed(wdid, daa);
                                if !pom_driver::is_installed(wdid) {
                                    // Staging failed (model not stageable/serveable on this card yet).
                                    // Do NOT grind and above all do NOT count: pre-fix this loop fell
                                    // through to mine_v4 (silent None with no walk) while still adding
                                    // `batch` to hashes_tried — a card at 0% GPU utilization reported
                                    // full phantom Mh/s. Idle honestly and retry next pass.
                                    static PAUSED_WARN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                                    let n = PAUSED_WARN.fetch_add(1, Ordering::Relaxed);
                                    if n % 60 == 0 {
                                        log::warn!(
                                            "PoM[gpu{}]: no walk installed (staging failed/pending) — grinding PAUSED on this card, hashrate 0 (not counted). Retrying staging…", wdid);
                                    }
                                    std::thread::sleep(std::time::Duration::from_millis(1000));
                                    continue;
                                }
                            }
                            // --wait-ready: this card is set up, but the rig as a whole is not —
                            // idle honestly (no hashes counted) so staging gets the host to itself
                            // and no share/challenge traffic starts early. No-op unless the flag
                            // is on; latches permanently open once every card is ready.
                            if keryx_miner::wait_ready::holds() {
                                std::thread::sleep(std::time::Duration::from_millis(500));
                                continue;
                            }
                            pom_driver::mine_v4(wdid, &pph, time, &target_le, range.start, range.len, daa)
                        };
                        // AMD (OpenCL): the thread is already bound to its card; the deviceless API is
                        // per-GPU via thread-local binding.
                        #[cfg(feature = "pom-opencl")]
                        let grind = {
                            if !pom_driver::is_installed() {
                                if let Some(new_cmd) = block_channel.get_changed()? {
                                    state = match new_cmd {
                                        Some(WorkerCommand::Job(s)) => Some(s),
                                        Some(WorkerCommand::Close) => return Ok(()),
                                        None => None,
                                    };
                                    continue;
                                }
                                let _ = pom_driver::ensure_installed();
                                if !pom_driver::is_installed() {
                                    // The backend reports this as Paused; retry staging here without
                                    // launching or reporting fake work.
                                    static PAUSED_WARN: std::sync::atomic::AtomicU64 =
                                        std::sync::atomic::AtomicU64::new(0);
                                    if PAUSED_WARN.fetch_add(1, Ordering::Relaxed) % 60 == 0 {
                                        log::warn!(
                                            "PoM[opencl]: no walk installed after staging — grinding \
                                             PAUSED, hashrate 0 (not counted). Retrying…"
                                        );
                                    }
                                    std::thread::sleep(std::time::Duration::from_millis(1000));
                                    continue;
                                }
                            }
                            // Match the CUDA/Metal contract above: installation must be allowed to
                            // finish while the latch is closed, but no OpenCL worker may hash or
                            // submit until every registered card is ready. This is intentionally
                            // after ensure_installed(), otherwise --wait-ready would deadlock.
                            if keryx_miner::wait_ready::holds() {
                                std::thread::sleep(std::time::Duration::from_millis(500));
                                continue;
                            }
                            pom_driver::mine_v4(&pph, time, &target_le, range.start, range.len, daa)
                        };
                        // A failed/paused backend has done no accountable work. Keep the cursor on
                        // the same range so recovery cannot silently skip candidates or manufacture
                        // hashrate. OpenCL may have enqueued a partial sub-batch before an error; it
                        // deliberately reports that as failed/zero and safely retries the range.
                        let completed = match grind {
                            Ok(done)
                                if done.hashes_done > 0
                                    && done.hashes_done <= range.len
                                    && done.winner.map_or(true, |winner| {
                                        winner
                                            .checked_sub(range.start)
                                            .map_or(false, |delta| delta < done.hashes_done)
                                    }) => done,
                            Ok(done) => {
                                warn!(
                                    "PoM backend returned an invalid completion (start={}, requested={}, done={}, winner={:?}); range not counted",
                                    range.start, range.len, done.hashes_done, done.winner
                                );
                                continue;
                            }
                            Err(keryx_miner::pom::GrindError::Paused(_reason)) => {
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                continue;
                            }
                            Err(keryx_miner::pom::GrindError::Backend(_error)) => {
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                continue;
                            }
                        };
                        if let Some((_, cursor)) = pom_cursor.as_mut() {
                            cursor.commit(completed.hashes_done);
                        }
                        hashes_tried.fetch_add(completed.hashes_done, Ordering::Relaxed);
                        worker_hashes_tried.fetch_add(completed.hashes_done, Ordering::Relaxed);
                        let found = completed.winner;
                        // --only-inference duty cycle: idle the card between launches so it draws
                        // almost nothing and is instantly available to serve. No-op otherwise.
                        #[cfg(any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")))]
                        if keryx_miner::pom_gpu::only_inference() {
                            let ms = keryx_miner::pom_gpu::only_inference_duty_ms();
                            if ms > 0 {
                                std::thread::sleep(std::time::Duration::from_millis(ms));
                            }
                        }
                        if let Some(nonce) = found {
                            // NVIDIA/Apple: recompute the PoM tier per block (H2-boundary correct).
                            // POOL SHARES: overlap the host proof re-walk with grinding, like the AMD
                            // path below. Even with the optimized sparse-checkpoint tree, doing the
                            // proof re-walk inline stalls THIS card's grind on every share; the
                            // resident-tree option lowers it further at a substantial RAM cost.
                            // The template clone is cheap (Arc-backed) and the thread re-fetches the
                            // index, so the worker loops straight back into the next batch. SOLO
                            // (FullBlock) stays synchronous below: it must clear `state` so the card
                            // stops grinding an already-mined template — a detached thread cannot.
                            #[cfg(any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")))]
                            // KERYX_POM_PROOF_OVERLAP=0 forces the old synchronous submit (A/B knob
                            // and safety valve: on a CPU-starved rig the detached builds could in
                            // principle contend with the launch loop).
                            let overlap_pool_proof = std::env::var("KERYX_POM_PROOF_OVERLAP").ok().as_deref() != Some("0")
                                && state.as_ref().map_or(false, |s| s.is_pool_share());
                            #[cfg(any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")))]
                            let mut proof_spawned = false;
                            #[cfg(any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")))]
                            if overlap_pool_proof {
                                if let Some(s) = state.as_ref() {
                                    if let (Some(proof_index), Some(tier)) = (
                                        keryx_miner::pom::active_index_for(wdid),
                                        keryx_miner::pom_gpu::current_tier(wdid, s.daa_score),
                                    ) {
                                        if let Some(permit) =
                                            InflightProofPermit::try_acquire(&inflight_proofs, MAX_INFLIGHT_PROOFS)
                                        {
                                            let s_clone = s.clone();
                                            let tx = send_channel.clone();
                                            proof_spawned = true;
                                            std::thread::spawn(move || {
                                                let _permit = permit;
                                                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                                    if let Some(mut block_seed) =
                                                        s_clone.generate_block_if_pom(nonce, &proof_index.0, tier)
                                                    {
                                                        if let crate::pow::BlockSeed::PartialBlock { device_id, .. } =
                                                            &mut block_seed
                                                        {
                                                            *device_id = worker_ordinal;
                                                        }
                                                        match tx.blocking_send(block_seed.clone()) {
                                                            Ok(()) => block_seed.report_block(),
                                                            Err(e) => warn!("Could not submit PoM block — pool connection dropped ({}); reconnecting", e),
                                                        }
                                                    }
                                                }));
                                                if outcome.is_err() {
                                                    warn!(
                                                        "PoM proof builder for GPU {} panicked during proof I/O; share dropped and proof slot recovered",
                                                        wdid
                                                    );
                                                }
                                            });
                                        }
                                    }
                                }
                            }
                            #[cfg(any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")))]
                            let built = if proof_spawned { None } else { state.as_ref().and_then(|s| {
                                keryx_miner::pom::active_index_for(wdid).and_then(|index| {
                                    let tier = keryx_miner::pom_gpu::current_tier(wdid, s.daa_score)?;
                                    s.generate_block_if_pom(nonce, &index.0, tier)
                                })
                            }) };
                            // SOLO (and the overlap-declined fallback) keep the synchronous submit.
                            #[cfg(any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")))]
                            if let Some(mut block_seed) = built {
                                // Tag the share with the GPU that found it so pool accept/reject is
                                // attributed per-card (the A: / R: columns).
                                if let crate::pow::BlockSeed::PartialBlock { device_id, .. } = &mut block_seed {
                                    *device_id = worker_ordinal;
                                }
                                match send_channel.blocking_send(block_seed.clone()) {
                                    Ok(()) => block_seed.report_block(),
                                    Err(e) => warn!("Could not submit PoM block — pool connection dropped ({}); reconnecting", e),
                                }
                                if let BlockSeed::FullBlock(_) = block_seed {
                                    state = None;
                                }
                            }
                            // AMD: OVERLAP the ~300 ms host proof re-walk with GPU grinding. The card
                            // just finished a batch and would otherwise idle (dropping its boost
                            // clock) while the CPU re-walks the winner to build the witness. Instead
                            // clone the (cheap, Arc-backed) block template and build+submit the proof
                            // on a detached thread; the worker loops straight back into the next grind.
                            // The host re-walk re-checks the target, so a kernel false-positive is
                            // dropped there — never submitted.
                            #[cfg(feature = "pom-opencl")]
                            if let Some(s) = state.as_ref() {
                                if let Some((proof_index, tier)) = keryx_miner::pom::active_index() {
                                    if let Some(permit) =
                                        InflightProofPermit::try_acquire(&inflight_proofs, MAX_INFLIGHT_PROOFS)
                                    {
                                        let s_clone = s.clone();
                                        let tier = *tier;
                                        let tx = send_channel.clone();
                                        std::thread::spawn(move || {
                                            let _permit = permit;
                                            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                                if let Some(mut block_seed) =
                                                    s_clone.generate_block_if_pom(nonce, proof_index, tier)
                                                {
                                                    if let crate::pow::BlockSeed::PartialBlock { device_id, .. } =
                                                        &mut block_seed
                                                    {
                                                        *device_id = worker_ordinal;
                                                    }
                                                    match tx.blocking_send(block_seed.clone()) {
                                                        Ok(()) => block_seed.report_block(),
                                                        Err(e) => warn!("Could not submit PoM block — pool connection dropped ({}); reconnecting", e),
                                                    }
                                                }
                                            }));
                                            if outcome.is_err() {
                                                warn!(
                                                    "PoM proof builder for OpenCL GPU {} panicked during proof I/O; share dropped and proof slot recovered",
                                                    wdid
                                                );
                                            }
                                        });
                                    }
                                }
                            }
                        }
                        // Pick up a newer job before grinding the next batch. The kHeavyHash path
                        // polls this at the loop tail, but the PoM branch `continue`s before
                        // reaching it — without this it would grind a stale template forever.
                        if state.is_some() {
                            if let Some(new_cmd) = block_channel.get_changed()? {
                                state = match new_cmd {
                                    Some(WorkerCommand::Job(s)) => Some(s),
                                    Some(WorkerCommand::Close) => return Ok(()),
                                    None => None,
                                };
                            }
                        }
                        continue;
                    }
                    let state_ref = match &state {
                        Some(s) => {
                            s.load_to_gpu(gpu_work);
                            s
                        },
                        None => continue,
                    };
                    state_ref.pow_gpu(gpu_work);
                    if let Err(e) = gpu_work.sync() {
                        warn!("CUDA run ignored: {}", e);
                        continue
                    }

                    gpu_work.copy_output_to(&mut nonces)?;
                    if nonces[0] != no_winner {
                        if let Some(mut block_seed) = state_ref.generate_block_if_pow(nonces[0]) {
                            if let BlockSeed::PartialBlock { device_id, .. } = &mut block_seed {
                                *device_id = worker_ordinal;
                            }
                            match send_channel.blocking_send(block_seed.clone()) {
                                Ok(()) => block_seed.report_block(),
                                // "block_seed" is a share at pool difficulty (or a real block when it
                                // also beats network target). A send error here means the pool
                                // connection's submit channel is gone — the stratum client now detects
                                // this and reconnects (see conn_dead in client/stratum.rs), so this is
                                // transient, not a lost-forever condition.
                                Err(e) => warn!("Could not submit share — pool connection dropped ({}); reconnecting", e),
                            };
                            if let BlockSeed::FullBlock(_) = block_seed {
                                state = None;
                            }
                            nonces[0] = no_winner;
                            hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);
                            worker_hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);
                            continue;
                        } else {
                            let hash = state_ref.calculate_pow(nonces[0]);
                            warn!("Something is wrong in GPU results! Got nonce {}, with hash real {:?}  (target: {}*2^196)", nonces[0], hash.0, state_ref.target.0[3]);
                            break;
                        }
                    }

                        /*
                        info!("Output should be: {:02X?}", state_ref.calculate_pow(nonces[0]).to_le_bytes());
                        info!("We got: {:02X?} (Nonces: {:02X?})", hashes[0], nonces[0].to_le_bytes());
                        assert!(state_ref.calculate_pow(nonces[0]).to_le_bytes() == hashes[0]);
                        */
                        /*
                        info!("Output should be: {}", state_ref.calculate_pow(nonces[nonces.len()-1]).0[3]);
                        info!("We got: {} (Nonces: {})", Uint256::from_le_bytes(hashes[nonces.len()-1]).0[3], nonces[nonces.len()-1]);
                        assert!(state_ref.calculate_pow(nonces[nonces.len()-1]).0[0] == Uint256::from_le_bytes(hashes[nonces.len()-1]).0[0]);
                         */
                        /*
                        if state_ref.calculate_pow(nonces[0]).0[0] != Uint256::from_le_bytes(hashes[0]).0[0] {
                            gpu_work.sync()?;
                            let mut nonce_vec = vec![nonces[0]; 1];
                            nonce_vec.append(&mut vec![0u64; gpu_work.workload-1]);
                            gpu_work.calculate_pow_hash(&state_ref.pow_hash_header, Some(&nonce_vec));
                            gpu_work.sync()?;
                            gpu_work.calculate_matrix_mul(&mut state_ref.matrix.clone().0.as_slice().as_dbuf().unwrap());
                            gpu_work.sync()?;
                            gpu_work.calculate_heavy_hash();
                            gpu_work.sync()?;
                            let mut hashes2  = vec![[0u8; 32]; out_size];
                            let mut nonces2= vec![0u64; out_size];
                            gpu_work.copy_output_to(&mut hashes2, &mut nonces2);
                            assert!(state_ref.calculate_pow(nonces[0]).to_le_bytes() == hashes2[0]);
                            assert!(nonces2[0] == nonces[0]);
                            assert!(hashes2 == hashes);
                            assert!(false);
                        }*/

                    hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);
                    worker_hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);

                    {
                        if let Some(new_cmd) = block_channel.get_changed()? {
                            state = match new_cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {return Ok(());}
                                None => None,
                            };
                        }
                    }
                }
                Ok(())
            })()
            .map_err(|e: Error| {
                error!("{}: GPU thread crashed: {}", gpu_work.id(), e.to_string());
                e
            })
        })
    }

    #[allow(unreachable_code)]
    fn launch_cpu_miner(
        send_channel: Sender<BlockSeed>,
        mut block_channel: watch::Receiver<Option<WorkerCommand>>,
        hashes_tried: Arc<AtomicU64>,
    ) -> MinerHandler {
        let mut nonce = Wrapping(thread_rng().next_u64());
        let mut mask = Wrapping(0);
        let mut fixed = Wrapping(0);
        std::thread::spawn(move || {
            (|| {
                let mut state = None;

                loop {
                    if state.is_none() {
                        state = match block_channel.wait_for_change() {
                            Ok(cmd) => match cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {
                                    return Ok(());
                                }
                                None => None,
                            },
                            Err(e) => {
                                info!("CPU thread crashed: {}", e.to_string());
                                return Ok(());
                            }
                        };
                        if let Some(s) = &state {
                            mask = Wrapping(s.nonce_mask);
                            fixed = Wrapping(s.nonce_fixed);
                        }
                    }
                    let state_ref = match state.as_mut() {
                        Some(s) => s,
                        None => continue,
                    };
                    nonce = (nonce & mask) | fixed;

                    if let Some(block_seed) = state_ref.generate_block_if_pow(nonce.0) {
                        match send_channel.blocking_send(block_seed.clone()) {
                            Ok(()) => block_seed.report_block(),
                            Err(e) => warn!("Could not submit share — pool connection dropped ({}); reconnecting", e),
                        };
                        if let BlockSeed::FullBlock(_) = block_seed {
                            state = None;
                        }
                    }
                    nonce += Wrapping(1);
                    // TODO: Is this really necessary? can we just use Relaxed?
                    hashes_tried.fetch_add(1, Ordering::AcqRel);

                    if nonce.0 % 128 == 0 {
                        if let Some(new_cmd) = block_channel.get_changed()? {
                            state = match new_cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {
                                    return Ok(());
                                }
                                None => None,
                            };
                        }
                    }
                }
                Ok(())
            })()
            .map_err(|e: Error| {
                error!("CPU thread crashed: {}", e.to_string());
                e
            })
        })
    }

    async fn log_hashrate(
        hashes_tried: Arc<AtomicU64>,
        hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
        opoi_challenge_active: Arc<AtomicBool>,
        stats: Arc<Mutex<MinerStats>>,
    ) {
        let mut ticker = tokio::time::interval(LOG_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut last_instant = ticker.tick().await;
        // Rate-limit the per-model staging diagnostic so a stuck "preparing" box prints the WHY
        // (expected path + reason each model was rejected) roughly once a minute, not every tick.
        let mut prep_diag_ticks: u32 = 0;
        loop {
            let now = ticker.tick().await;
            let duration = (now - last_instant).as_secs_f64();
            let challenge_active = opoi_challenge_active.load(Ordering::Relaxed);
            // First-run PREPARATION: a 0 h/s reading is EXPECTED while the model is downloading
            // or the possession index / resident tree is being built (the latter can take minutes,
            // esp. with --resident-tree). Report that plainly instead of "workers stalled or
            // crashed", which alarms users who simply haven't finished the one-time model download.
            let preparing = keryx_miner::slm::mining_preparing() || {
                #[cfg(feature = "pom-cuda")]
                {
                    keryx_miner::pom_gpu::is_loading()
                }
                #[cfg(not(feature = "pom-cuda"))]
                {
                    false
                }
            };
            let stall_or_prep: &str = if preparing {
                if keryx_miner::slm::is_downloading() {
                    "still downloading the model (one-time first-run setup) — mining starts automatically when done, this is NOT a stall"
                } else if keryx_miner::slm::loaded_model_ids().is_empty() {
                    // Stuck with no model ready and nothing downloading. If a concrete failure was
                    // recorded (download / disk-full / corrupt file), surface it LOUD every cycle so it
                    // is always in the last log lines an operator sees — the reassuring "preparing…"
                    // line used to bury it. Otherwise fall back to the per-model path/reason (~once/min).
                    let staging_err = keryx_miner::slm::last_staging_error();
                    if let Some(err) = &staging_err {
                        error!("MODEL STAGING FAILED — mining is suspended: {}", err);
                    } else if prep_diag_ticks % 6 == 0 {
                        let diags = keryx_miner::slm::staging_diagnostics();
                        if diags.is_empty() {
                            warn!("model staging: no model lineup installed yet (waiting for chain tip / tier selection).");
                        } else {
                            warn!("model staging status (no model is ready yet — mining is suspended until one is):");
                            for line in diags {
                                warn!("{}", line);
                            }
                        }
                    }
                    prep_diag_ticks = prep_diag_ticks.wrapping_add(1);
                    if staging_err.is_some() {
                        "model staging FAILED — see the ERROR line above (mining stays suspended until it is fixed)"
                    } else {
                        "preparing models (staging/verifying files) — mining starts automatically when ready, this is NOT a stall"
                    }
                } else {
                    "building the possession index / resident tree — mining starts automatically when done, this is NOT a stall"
                }
            } else {
                "Workers stalled or crashed. Considered reducing workload and check that your node is synced"
            };
            let total = Self::log_single_hashrate_with(
                &hashes_tried,
                "Current hashrate is".into(),
                stall_or_prep,
                duration,
                false,
                challenge_active,
                "",
                None,
                preparing,
            );
            let mut devices = Vec::new();
            let mut power_sum = 0.0f64;
            let mut power_seen = false;
            for (device, rate) in &*hashes_by_worker.lock().unwrap() {
                // NVML sample in the same tick as the rate, so the printed/API efficiency
                // pairs a hashrate with the power draw that produced it (field request:
                // one line with temp/fan/power/clocks/vram + MH/s + MH/s/W).
                let mut health = crate::gpu_health::ordinal_from_label(device)
                    .and_then(|ord| crate::gpu_health::sample(ord, device));
                // AMD has no NVML: read power/temp/clocks/VRAM from sysfs (amdgpu hwmon), matched
                // to the plugin's "#N" device by PCI — so MH/s/W works on AMD rigs too (field
                // request). Only when the NVML path produced nothing, and only on the AMD build.
                #[cfg(all(feature = "pom-opencl", unix))]
                if health.is_none() {
                    if let Some(ord) = crate::gpu_health::ordinal_from_label(device) {
                        health =
                            keryx_miner::pom_opencl::amd_health(ord as usize).map(|a| crate::gpu_health::GpuHealth {
                                temp_c: a.temp_c,
                                fan_pct: a.fan_pct,
                                power_w: a.power_w,
                                core_mhz: a.core_mhz,
                                mem_mhz: None,
                                vram_used_mb: a.vram_used_mb,
                                vram_total_mb: a.vram_total_mb,
                            });
                    }
                }
                let mut suffix = match &health {
                    Some(h) => {
                        if let Some(w) = h.power_w {
                            power_sum += w;
                            power_seen = true;
                        }
                        let frag = h.to_log_fragment();
                        if frag.is_empty() {
                            String::new()
                        } else {
                            format!(" | {}", frag)
                        }
                    }
                    None => String::new(),
                };
                // Per-GPU accepted/rejected shares (field request): a card showing hashrate but
                // `A: 0` is producing phantom hashrate (unstable OC) and never landing a real share.
                if let Some(ord) = crate::gpu_health::ordinal_from_label(device) {
                    let (acc, rej) = crate::pow::device_share_counts(ord as u32);
                    suffix.push_str(&format!(" | A: {} / R: {}", acc, rej));
                }
                let r = Self::log_single_hashrate_with(
                    rate,
                    format!("Device {}:", device),
                    "0 hash/s (preparing)",
                    duration,
                    true,
                    challenge_active,
                    &suffix,
                    health.as_ref().and_then(|h| h.power_w),
                    preparing,
                );
                // Feed the inference router the same measured per-card rate shown to the
                // operator. Worker labels carry the backend's logical ordinal (`#N ...`) on both
                // CUDA and OpenCL. The router ignores zero samples from setup/inference pauses and
                // retains its documented VRAM proxy until a real mining interval exists.
                if let Some(ord) = crate::gpu_health::ordinal_from_label(device) {
                    keryx_miner::slm::report_card_hashrate(ord as usize, r);
                }
                let eff = crate::gpu_health::efficiency_mhs_per_w(r, health.as_ref().and_then(|h| h.power_w));
                devices.push(DeviceRate { label: device.clone(), hashrate: r, health, efficiency_mhs_per_w: eff });
            }
            let total_power_w = if power_seen { Some(power_sum) } else { None };
            let total_eff = crate::gpu_health::efficiency_mhs_per_w(total, total_power_w);
            if let (Some(w), Some(e)) = (total_power_w, total_eff) {
                info!("Rig efficiency: {:.1} W total, {:.3} MH/s/W", w, e);
            }
            // Publish a backend-neutral copy for optional frontends. This uses only values already
            // sampled for the normal ten-second log tick; no extra NVML/sysfs calls touch mining.
            let runtime_devices = devices
                .iter()
                .enumerate()
                .map(|(fallback_index, device)| {
                    let index = crate::gpu_health::ordinal_from_label(&device.label)
                        .unwrap_or_else(|| u32::try_from(fallback_index).unwrap_or(u32::MAX));
                    let (accepted, rejected) = crate::pow::device_share_counts(index);
                    let health = device.health.as_ref();
                    keryx_miner::runtime_stats::DeviceSnapshot {
                        index,
                        label: device.label.clone(),
                        backend: if cfg!(feature = "pom-opencl") {
                            "OpenCL"
                        } else if cfg!(all(target_os = "macos", feature = "pom-metal")) {
                            "Metal"
                        } else {
                            "CUDA"
                        }
                        .to_string(),
                        hashrate_hs: device.hashrate,
                        temp_c: health.and_then(|sample| sample.temp_c),
                        hotspot_c: None,
                        fan_pct: health.and_then(|sample| sample.fan_pct),
                        power_w: health.and_then(|sample| sample.power_w),
                        core_mhz: health.and_then(|sample| sample.core_mhz),
                        mem_mhz: health.and_then(|sample| sample.mem_mhz),
                        vram_used_mb: health.and_then(|sample| sample.vram_used_mb),
                        vram_total_mb: health.and_then(|sample| sample.vram_total_mb),
                        efficiency_mhs_per_w: device.efficiency_mhs_per_w,
                        accepted,
                        rejected,
                    }
                })
                .collect();
            keryx_miner::runtime_stats::publish_mining_snapshot(
                total,
                total_power_w,
                total_eff,
                preparing,
                challenge_active,
                runtime_devices,
            );
            // Publish the snapshot for the stats API (hashrates are hashes/sec).
            if let Ok(mut s) = stats.lock() {
                s.total_hashrate = total;
                s.devices = devices;
                s.total_power_w = total_power_w;
                s.total_efficiency_mhs_per_w = total_eff;
            }
            last_instant = now;
        }
    }

    /// Log one hashrate line, with an optional health suffix and power figure: prints
    /// `<prefix> <rate> <unit> | <health> | eff <x.xxx> MH/s/W` on one line. The
    /// stall-warning and OPoI-pause branches are unchanged.
    #[allow(clippy::too_many_arguments)]
    fn log_single_hashrate_with(
        counter: &Arc<AtomicU64>,
        prefix: String,
        warn_message: &str,
        duration: f64,
        keep_prefix: bool,
        challenge_active: bool,
        health_suffix: &str,
        power_w: Option<f64>,
        preparing: bool,
    ) -> f64 {
        let hashes = counter.swap(0, Ordering::AcqRel);
        let rate = (hashes as f64) / duration;
        if hashes == 0 {
            if challenge_active {
                if keep_prefix {
                    info!("{} OPoI challenge in progress — stand by", prefix);
                } else {
                    info!("OPoI challenge in progress — PoW paused, stand by");
                }
            } else if preparing {
                // Expected during first-run model download / index build — informational, not a warning.
                match keep_prefix {
                    true => info!("{} {}", prefix, warn_message),
                    false => info!("{}", warn_message),
                };
            } else {
                match keep_prefix {
                    true => warn!("{}{}", prefix, warn_message),
                    false => warn!("{}", warn_message),
                };
            }
        } else {
            let (disp, unit) = Self::hash_suffix(rate);
            match crate::gpu_health::efficiency_mhs_per_w(rate, power_w) {
                Some(eff) => info!("{} {:.2} {}{} | eff {:.3} MH/s/W", prefix, disp, unit, health_suffix, eff),
                None => info!("{} {:.2} {}{}", prefix, disp, unit, health_suffix),
            }
        }
        rate
    }

    #[inline]
    fn hash_suffix(n: f64) -> (f64, &'static str) {
        match n {
            n if n < 1_000.0 => (n, "hash/s"),
            n if n < 1_000_000.0 => (n / 1_000.0, "Khash/s"),
            n if n < 1_000_000_000.0 => (n / 1_000_000.0, "Mhash/s"),
            n if n < 1_000_000_000_000.0 => (n / 1_000_000_000.0, "Ghash/s"),
            n if n < 1_000_000_000_000_000.0 => (n / 1_000_000_000_000.0, "Thash/s"),
            _ => (n, "hash/s"),
        }
    }
}

#[cfg(test)]
mod inference_pause_tests {
    use super::*;

    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    struct ResetPauseState;

    impl Drop for ResetPauseState {
        fn drop(&mut self) {
            let mut state = lock_inference_pause_state();
            state.inflight = 0;
            state.current_miner = None;
        }
    }

    fn enter_test() -> (std::sync::MutexGuard<'static, ()>, ResetPauseState) {
        let serial = TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        {
            let mut state = lock_inference_pause_state();
            state.inflight = 0;
            state.current_miner = None;
        }
        (serial, ResetPauseState)
    }

    fn control() -> (Arc<InferencePauseControl>, watch::Receiver<Option<WorkerCommand>>, Arc<AtomicBool>) {
        let (sender, mut receiver) = watch::channel(None);
        // Consume the channel's initial value so subsequent `get_changed`
        // assertions describe an actual pause or resume publication.
        assert!(matches!(receiver.get_changed(), Ok(Some(None))));
        let active = Arc::new(AtomicBool::new(false));
        let control = Arc::new(InferencePauseControl {
            block_channel: Arc::new(sender),
            active_flag: Arc::clone(&active),
            resumable_stratum_job: Mutex::new(None),
        });
        (control, receiver, active)
    }

    fn register(control: &Arc<InferencePauseControl>) {
        let mut state = lock_inference_pause_state();
        if state.inflight != 0 {
            control.active_flag.store(true, Ordering::SeqCst);
        }
        state.current_miner = Some(Arc::downgrade(control));
    }

    fn job(id: usize) -> WorkerCommand {
        let seed = BlockSeed::PartialBlock {
            id: format!("test-{id}"),
            header_hash: [id as u64, 1, 2, 3],
            timestamp: id as u64,
            daa_score: 0,
            nonce: 0,
            target: Default::default(),
            nonce_mask: u64::MAX,
            nonce_fixed: 0,
            hash: None,
            pom_proof: Vec::new(),
            device_id: None,
        };
        WorkerCommand::Job(Box::new(pow::State::new(id, seed).expect("test job must be valid")))
    }

    fn assert_pause(receiver: &mut watch::Receiver<Option<WorkerCommand>>) {
        assert!(matches!(receiver.get_changed(), Ok(Some(None))));
    }

    fn assert_resumed_job(receiver: &mut watch::Receiver<Option<WorkerCommand>>, expected_id: usize) {
        match receiver.get_changed().expect("worker channel must remain open") {
            Some(Some(WorkerCommand::Job(state))) => assert_eq!(state.id, expected_id),
            _ => panic!("expected retained Stratum job {expected_id} to be replayed"),
        }
    }

    #[test]
    fn nested_pause_resumes_only_after_outer_guard() {
        let (_serial, _reset) = enter_test();
        let (control, mut receiver, active) = control();
        control.remember_stratum_job(job(11));
        register(&control);

        let outer = begin_inference_pause();
        assert_pause(&mut receiver);
        let inner = begin_inference_pause();
        assert_pause(&mut receiver);

        drop(inner);
        assert!(active.load(Ordering::SeqCst));
        assert!(matches!(receiver.get_changed(), Ok(None)), "inner guard must not resume mining");

        drop(outer);
        assert!(!active.load(Ordering::SeqCst));
        assert_resumed_job(&mut receiver, 11);
    }

    #[test]
    fn final_guard_resumes_current_stratum_job_without_a_new_notify() {
        let (_serial, _reset) = enter_test();
        let (control, mut receiver, active) = control();
        control.remember_stratum_job(job(21));
        register(&control);

        let pause = begin_inference_pause();
        assert_pause(&mut receiver);
        drop(pause);

        assert!(!active.load(Ordering::SeqCst));
        assert_resumed_job(&mut receiver, 21);
    }

    #[test]
    fn reconnect_resumes_only_the_new_managers_newest_job() {
        let (_serial, _reset) = enter_test();
        let (old, mut old_receiver, _) = control();
        old.remember_stratum_job(job(31));
        register(&old);

        let pause = begin_inference_pause();
        assert_pause(&mut old_receiver);

        // A reconnect installs a fresh control/generation while the detached
        // inference is still running. Its newer notify is the sole candidate;
        // the old connection's retained job is unreachable from global state.
        let (new, mut new_receiver, new_active) = control();
        new.remember_stratum_job(job(32));
        register(&new);
        assert!(new_active.load(Ordering::SeqCst));

        drop(pause);

        assert_resumed_job(&mut new_receiver, 32);
        assert!(matches!(old_receiver.get_changed(), Ok(None)), "old connection must not receive a replay");
        assert!(!new_active.load(Ordering::SeqCst));
    }

    #[cfg(any(feature = "pom-opencl", feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
    #[test]
    fn detached_proof_permit_enforces_cap_and_recovers_on_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let first = InflightProofPermit::try_acquire(&counter, 2).expect("first permit");
        let second = InflightProofPermit::try_acquire(&counter, 2).expect("second permit");
        assert_eq!(counter.load(Ordering::Acquire), 2);
        assert!(InflightProofPermit::try_acquire(&counter, 2).is_none(), "cap must be atomic");

        drop(first);
        let replacement = InflightProofPermit::try_acquire(&counter, 2).expect("drop must release a slot");
        assert_eq!(counter.load(Ordering::Acquire), 2);
        drop(second);
        drop(replacement);
        assert_eq!(counter.load(Ordering::Acquire), 0);
    }
}

#[cfg(all(test, feature = "bench"))]
mod benches {
    extern crate test;

    use self::test::{black_box, Bencher};
    use crate::pow::State;
    use crate::proto::{RpcBlock, RpcBlockHeader};
    use rand::{thread_rng, RngCore};

    #[bench]
    pub fn bench_mining(bh: &mut Bencher) {
        let mut state = State::new(
            0,
            RpcBlock {
                header: Some(RpcBlockHeader {
                    version: 1,
                    parents: vec![],
                    hash_merkle_root: "23618af45051560529440541e7dc56be27676d278b1e00324b048d410a19d764".to_string(),
                    accepted_id_merkle_root: "947d1a10378d6478b6957a0ed71866812dee33684968031b1cace4908c149d94"
                        .to_string(),
                    utxo_commitment: "ec5e8fc0bc0c637004cee262cef12e7cf6d9cd7772513dbd466176a07ab7c4f4".to_string(),
                    timestamp: 654654353,
                    bits: 0x1e7fffff,
                    nonce: 0,
                    daa_score: 654456,
                    blue_work: "d8e28a03234786".to_string(),
                    pruning_point: "be4c415d378f9113fabd3c09fcc84ddb6a00f900c87cb6a1186993ddc3014e2d".to_string(),
                    blue_score: 1164419,
                    pom_final_state: 0,
                    service_state_hash: String::new(),
                    pom_tier: 0,
                }),
                transactions: vec![],
                verbose_data: None,
            },
        )
        .unwrap();
        nonce = thread_rng().next_u64();
        bh.iter(|| {
            for _ in 0..100 {
                black_box(state.check_pow(nonce));
                nonce += 1;
            }
        });
    }
}
