use std::collections::HashMap;
use std::num::Wrapping;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

use crate::{pow, watch, Error};
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use tokio::sync::mpsc::Sender;
use tokio::task::{self, JoinHandle};
use tokio::time::MissedTickBehavior;

use crate::pow::BlockSeed;
use keryx_miner::{PluginManager, WorkerSpec};

type MinerHandler = std::thread::JoinHandle<Result<(), Error>>;

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
extern "C" fn signal_panic(_signal: nix::libc::c_int) {
    panic!("Forced shutdown");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn register_freeze_handler() {
    let handler = nix::sys::signal::SigHandler::Handler(signal_panic);
    unsafe {
        nix::sys::signal::signal(nix::sys::signal::Signal::SIGUSR1, handler).unwrap();
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
            warn!("Worker did not exit within {}s of shutdown — force-killing (assumed frozen)", FREEZE_GRACE.as_secs());
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
#[derive(Default, Clone)]
pub struct DeviceRate {
    pub label: String,
    pub hashrate: f64,
}

/// Live hashrate snapshot, refreshed by the logger every LOG_RATE. Read by the stats API.
#[derive(Default)]
pub struct MinerStats {
    pub total_hashrate: f64,
    pub devices: Vec<DeviceRate>,
}

#[allow(dead_code)]
pub struct MinerManager {
    handles: Vec<MinerHandler>,
    block_channel: watch::Sender<Option<WorkerCommand>>,
    send_channel: Sender<BlockSeed>,
    logger_handle: JoinHandle<()>,
    is_synced: bool,
    hashes_tried: Arc<AtomicU64>,
    hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    current_state_id: AtomicUsize,
    opoi_challenge_active: Arc<AtomicBool>,
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
        register_freeze_handler();
        let hashes_tried = Arc::new(AtomicU64::new(0));
        let hashes_by_worker = Arc::new(Mutex::new(HashMap::<String, Arc<AtomicU64>>::new()));
        let opoi_challenge_active = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(MinerStats::default()));
        let (send, recv) = watch::channel(None);
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
                self.is_synced = true;
                let id = self.current_state_id.fetch_add(1, Ordering::SeqCst);
                Some(WorkerCommand::Job(Box::new(pow::State::new(id, b)?)))
            }
            None => {
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
        spec: Box<dyn WorkerSpec>,
        worker_hashes_tried: Arc<AtomicU64>,
    ) -> MinerHandler {
        std::thread::spawn(move || {
            let mut box_ = spec.build();
            // AMD multi-GPU PoM: bind this thread to its own card so the possession tier is made
            // resident per-GPU and every card mines (and submits) its own shares — instead of all
            // GPU threads funneling onto device 0 through one global lock (3 cards ran like 1).
            #[cfg(feature = "pom-opencl")]
            if let Some(dev) = spec.opencl_device_id() {
                keryx_miner::pom_opencl::bind_thread_device(dev);
            }
            let gpu_work = box_.as_mut();
            (|| {
                info!("Spawned Thread for GPU {}", gpu_work.id());
                let mut nonces = vec![0u64; 1];

                let mut state = None;
                // PoM (post-fork): nonce cursor + per-launch batch. The kernel grinds the whole
                // batch before returning, so blocks/sec is capped at hashrate / POM_BATCH.
                #[cfg(any(feature = "pom-opencl", feature = "pom-cuda"))]
                let mut pom_nonce: u64 = thread_rng().next_u64();
                #[cfg(any(feature = "pom-opencl", feature = "pom-cuda"))]
                const POM_BATCH: u64 = 1 << 20;
                // Driver seam: AMD = OpenCL, NVIDIA = candle-CUDA. Both expose the same interface
                // (is_installed / ensure_installed / mine / set_mining_tier). OpenCL wins if both on.
                #[cfg(feature = "pom-opencl")]
                use keryx_miner::pom_opencl as pom_driver;
                #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
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
                    // PoM possession mining (post-fork): grind the data-dependent walk on the GPU
                    // over the resident tier weights instead of kHeavyHash. On a winning nonce we
                    // build the proof (host) and submit; the legacy plugin path below is skipped.
                    #[cfg(any(feature = "pom-opencl", feature = "pom-cuda"))]
                    if matches!(state.as_ref(), Some(s) if s.daa_score >= keryx_miner::pom::activation_daa()) {
                        let (pph, time, target_le, daa) = {
                            let s = state.as_ref().unwrap();
                            let mut pph = [0u8; 32];
                            pph.copy_from_slice(&s.pow_hash_header[0..32]);
                            (pph, u64::from_le_bytes(s.pow_hash_header[32..40].try_into().unwrap()), s.target.to_le_bytes(), s.daa_score)
                        };
                        // NVIDIA: per-device PoM. Each GPU thread builds + walks its OWN device's
                        // blob (upstream's per-device MINERS map) so no-flag multi-GPU works without
                        // CUDA_VISIBLE_DEVICES. Device id = the worker's `#N (name)` label.
                        #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
                        let found = {
                            let wdid = gpu_work.id().strip_prefix('#')
                                .and_then(|s| s.split_whitespace().next())
                                .and_then(|s| s.parse::<u32>().ok())
                                .unwrap_or(0);
                            if !pom_driver::is_installed(wdid) {
                                pom_driver::ensure_installed(wdid, daa);
                            }
                            pom_driver::mine(wdid, &pph, time, &target_le, pom_nonce, POM_BATCH)
                        };
                        // AMD: the thread is already bound to its card (bind_thread_device), so the
                        // deviceless OpenCL API is per-GPU via thread-local binding.
                        #[cfg(feature = "pom-opencl")]
                        let found = {
                            let _ = daa;
                            if !pom_driver::is_installed() {
                                let _ = pom_driver::ensure_installed();
                            }
                            pom_driver::mine(&pph, time, &target_le, pom_nonce, POM_BATCH)
                        };
                        pom_nonce = pom_nonce.wrapping_add(POM_BATCH);
                        hashes_tried.fetch_add(POM_BATCH, Ordering::AcqRel);
                        worker_hashes_tried.fetch_add(POM_BATCH, Ordering::AcqRel);
                        if let Some(nonce) = found {
                            // NVIDIA: recompute the PoM tier per block (H2-boundary correct).
                            #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
                            let built = state.as_ref().and_then(|s| {
                                keryx_miner::pom::active_index().and_then(|(idx, _)| {
                                    let tier = keryx_miner::pom_gpu::current_tier(s.daa_score)?;
                                    s.generate_block_if_pom(nonce, idx, tier)
                                })
                            });
                            #[cfg(feature = "pom-opencl")]
                            let built = state.as_ref().and_then(|s| {
                                keryx_miner::pom::active_index().and_then(|(idx, tier)| s.generate_block_if_pom(nonce, idx, *tier))
                            });
                            if let Some(block_seed) = built {
                                match send_channel.blocking_send(block_seed.clone()) {
                                    Ok(()) => block_seed.report_block(),
                                    Err(e) => warn!("Could not submit PoM block — pool connection dropped ({}); reconnecting", e),
                                }
                                if let BlockSeed::FullBlock(_) = block_seed {
                                    state = None;
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
                    if nonces[0] != 0 {
                        if let Some(block_seed) = state_ref.generate_block_if_pow(nonces[0]) {
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
                            nonces[0] = 0;
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
        loop {
            let now = ticker.tick().await;
            let duration = (now - last_instant).as_secs_f64();
            let challenge_active = opoi_challenge_active.load(Ordering::Relaxed);
            let total = Self::log_single_hashrate(
                &hashes_tried,
                "Current hashrate is".into(),
                "Workers stalled or crashed. Considered reducing workload and check that your node is synced",
                duration,
                false,
                challenge_active,
            );
            let mut devices = Vec::new();
            for (device, rate) in &*hashes_by_worker.lock().unwrap() {
                let r = Self::log_single_hashrate(rate, format!("Device {}:", device), "0 hash/s", duration, true, challenge_active);
                devices.push(DeviceRate { label: device.clone(), hashrate: r });
            }
            // Publish the snapshot for the stats API (hashrates are hashes/sec).
            if let Ok(mut s) = stats.lock() {
                s.total_hashrate = total;
                s.devices = devices;
            }
            last_instant = now;
        }
    }

    fn log_single_hashrate(
        counter: &Arc<AtomicU64>,
        prefix: String,
        warn_message: &str,
        duration: f64,
        keep_prefix: bool,
        challenge_active: bool,
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
            } else {
                match keep_prefix {
                    true => warn!("{}{}", prefix, warn_message),
                    false => warn!("{}", warn_message),
                };
            }
        } else {
            let (rate, suffix) = Self::hash_suffix(rate);
            info!("{} {:.2} {}", prefix, rate, suffix);
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
