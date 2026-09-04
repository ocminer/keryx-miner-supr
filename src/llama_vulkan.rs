//! GPU inference via a bundled llama.cpp `llama-server` subprocess (Vulkan on AMD, CUDA on NVIDIA).
//!
//! AMD: candle 0.9 has no AMD-GPU backend, so the `pom-opencl` build generates the OPoI LLM text
//! with llama.cpp (Vulkan) on the AMD GPU instead of candle on the CPU.
//! NVIDIA (`pom-cuda`): when a CUDA `llama-server` is bundled next to the miner (or pointed at via
//! `KERYX_LLAMA_SERVER`), inference prefers it over the in-process GPU engine. Without either GPU
//! engine the route remains unavailable; the deprecated CPU emergency fallback runs only when it
//! is explicitly enabled. We spawn `llama-server` once (the GGUF stays resident in VRAM) and
//! `generate()` HTTP-POSTs `/completion`.
//!
//! This is OPoI-safe: consensus verifies the fixed-point `model_fixed` commitment (computed
//! separately, bit-exact on all hardware) — the Gemma *text* is user-facing utility with no
//! determinism requirement, so a non-candle engine is fine.
//!
//! Best-effort + self-disabling: if the bundled `llama-server` is missing, Vulkan/AMD GPU is
//! unavailable, or the server doesn't come up healthy, `AVAILABLE` stays false and the caller
//! (`slm::load_and_run_inference`) withdraws the route if this GPU engine is unavailable.

use std::io::Read;
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use rand::RngCore;

static AVAILABLE: AtomicBool = AtomicBool::new(false);
static PORT: AtomicU16 = AtomicU16::new(0);
/// PID of the running llama-server (0 = none). The Child handle itself is owned by the monitor
/// thread that spawned it (see try_start) — it reaps the child and clears this on exit.
static SERVER_PID: AtomicU32 = AtomicU32::new(0);
static SERVER_GENERATION: AtomicU64 = AtomicU64::new(0);
/// Generation whose child has exited and whose monitor is still retiring shared state. PID zero
/// is published only after that cleanup, so replacement allocation has a release barrier.
static SERVER_EXITING_GENERATION: AtomicU64 = AtomicU64::new(0);
/// ggml `main_gpu` used by the active Vulkan server; -1 on CUDA/non-Vulkan or while stopped.
static SERVER_VK_DEVICE: AtomicI32 = AtomicI32::new(-1);

#[cfg(all(feature = "pom-opencl", unix))]
struct ServerDedicationAttempt {
    armed: bool,
}

#[cfg(all(feature = "pom-opencl", unix))]
impl Drop for ServerDedicationAttempt {
    fn drop(&mut self) {
        if self.armed {
            crate::pom_opencl::release_vulkan_server_dedication();
        }
    }
}

/// Serializes start/stop and holds the server identity stable for the full HTTP request. The child
/// monitor never takes this lock; it uses the generation+PID pair for compare-before-clear.
fn lifecycle() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ServerIdentity {
    gguf: std::path::PathBuf,
    model_id: Option<[u8; 32]>,
    api_key: String,
    gpu: usize,
    generation: u64,
    pid: u32,
}

fn server_identity() -> &'static Mutex<Option<ServerIdentity>> {
    static ID: OnceLock<Mutex<Option<ServerIdentity>>> = OnceLock::new();
    ID.get_or_init(|| Mutex::new(None))
}

fn normalized_model_path(path: &str) -> std::path::PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| std::path::PathBuf::from(path))
}

/// Translate a parent-process CUDA logical ordinal back to the token that selected the same
/// physical device. Overwriting an inherited `CUDA_VISIBLE_DEVICES=2,0` with logical `1` would
/// otherwise move the child to physical GPU 1 instead of the parent's logical GPU 1 (physical 0).
fn visible_device_token(visible: Option<&str>, logical_gpu: usize) -> String {
    visible
        .and_then(|list| list.split(',').map(str::trim).filter(|s| !s.is_empty()).nth(logical_gpu).map(str::to_owned))
        .unwrap_or_else(|| logical_gpu.to_string())
}

fn child_cuda_visible_device(logical_gpu: usize) -> String {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    visible_device_token(visible.as_deref(), logical_gpu)
}

fn bounded_http_timeout_secs(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.trim().parse::<u64>().ok()).unwrap_or(120).clamp(30, 120)
}

fn exact_process_owner(
    expected_generation: u64,
    expected_pid: u32,
    observed_generation: u64,
    observed_pid: u32,
) -> bool {
    expected_pid != 0 && expected_generation == observed_generation && expected_pid == observed_pid
}

fn exact_process_publishable(
    expected_generation: u64,
    expected_pid: u32,
    observed_generation: u64,
    observed_pid: u32,
    exiting_generation: u64,
) -> bool {
    exact_process_owner(expected_generation, expected_pid, observed_generation, observed_pid)
        && exiting_generation != expected_generation
}

/// Complete every old-generation cleanup action before publishing PID zero. An Acquire observer
/// of zero therefore knows that the old monitor can no longer clear any replacement state.
fn finish_monitor_cleanup<F>(pid_slot: &AtomicU32, expected_pid: u32, cleanup: F) -> bool
where
    F: FnOnce(),
{
    cleanup();
    pid_slot.compare_exchange(expected_pid, 0, Ordering::AcqRel, Ordering::Acquire).is_ok()
}

/// Run `commit` only while an exact server identity is still owned.  The identity mutex is the
/// publication linearization point shared with the child monitor: if the monitor wins, validation
/// fails and no proof is written; if the commit wins, the monitor cannot take the identity until
/// after the proof is written and will then invalidate it on exit.
fn commit_with_owned_identity<F, C>(identity_slot: &Mutex<Option<ServerIdentity>>, validate: F, commit: C) -> bool
where
    F: FnOnce(&ServerIdentity) -> bool,
    C: FnOnce(&ServerIdentity),
{
    let identity = match identity_slot.lock() {
        Ok(identity) => identity,
        Err(poisoned) => poisoned.into_inner(),
    };
    let Some(active) = identity.as_ref() else { return false };
    if !validate(active) {
        return false;
    }
    commit(active);
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplacementAction {
    None,
    WaitForMonitor,
    StopPlanned,
}

fn replacement_action(pid: u32, generation: u64, exiting_generation: u64) -> ReplacementAction {
    if pid == 0 {
        ReplacementAction::None
    } else if exiting_generation == generation {
        ReplacementAction::WaitForMonitor
    } else {
        ReplacementAction::StopPlanned
    }
}

fn authenticated_props_ready(port: u16, api_key: &str) -> bool {
    if port == 0 {
        return false;
    }
    let url = format!("http://127.0.0.1:{port}/props");
    let auth = format!("Bearer {api_key}");
    ureq::get(&url).timeout(Duration::from_secs(2)).set("Authorization", &auth).call().is_ok()
}

/// A planned replacement may preserve the old model's cached ability proof only after the exact
/// authenticated child answers immediately before the stop. This distinguishes a healthy model
/// swap from the narrow natural-exit window after `Child::wait` returned but before its monitor had
/// time to publish `SERVER_EXITING_GENERATION`.
fn current_server_responsive_locked() -> bool {
    let generation = SERVER_GENERATION.load(Ordering::Acquire);
    let pid = SERVER_PID.load(Ordering::Acquire);
    let port = PORT.load(Ordering::Acquire);
    let identity = match server_identity().lock() {
        Ok(identity) => identity.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };
    let Some(identity) = identity else { return false };
    if !AVAILABLE.load(Ordering::Acquire)
        || SERVER_EXITING_GENERATION.load(Ordering::Acquire) == generation
        || !exact_process_owner(identity.generation, identity.pid, generation, pid)
    {
        return false;
    }
    authenticated_props_ready(port, &identity.api_key)
}

fn planned_replacement_invalidates(authenticated_child_ready: bool) -> bool {
    !authenticated_child_ready
}

/// Called only while `lifecycle()` is held. A monitor already retiring a naturally exited child
/// owns route invalidation; wait for its PID-zero release barrier without stealing its identity.
/// Otherwise this is a clean planned replacement, so stop/reap but preserve cached ability proofs.
fn retire_server_for_replacement_locked() {
    let pid = SERVER_PID.load(Ordering::Acquire);
    match replacement_action(
        pid,
        SERVER_GENERATION.load(Ordering::Acquire),
        SERVER_EXITING_GENERATION.load(Ordering::Acquire),
    ) {
        ReplacementAction::None => {}
        ReplacementAction::StopPlanned => {
            let responsive = current_server_responsive_locked();
            stop_locked(
                planned_replacement_invalidates(responsive),
                if responsive { "planned_server_replacement" } else { "server_unresponsive_before_replacement" },
            );
        }
        ReplacementAction::WaitForMonitor => {
            while SERVER_PID.load(Ordering::Acquire) == pid {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

fn set_server_identity(value: Option<ServerIdentity>) {
    if let Ok(mut id) = server_identity().lock() {
        *id = value;
    }
}

/// Remove an identity only when it belongs to the monitor's exact server generation. An old child
/// may exit after a replacement has already published a newer identity; in that case it must not
/// withdraw the new route.
fn take_identity_for_generation(generation: u64) -> Option<ServerIdentity> {
    let mut identity = server_identity().lock().ok()?;
    if identity.as_ref().map(|id| id.generation) == Some(generation) {
        identity.take()
    } else {
        None
    }
}

fn invalidate_identity_route(identity: &ServerIdentity, reason: &str) {
    if let Some(model_id) = identity.model_id {
        crate::slm::invalidate_inference_route(&model_id, identity.gpu, reason);
    } else {
        // A server launched before the active model lineup was installed cannot be scoped by ID.
        // Fail safe on an actual fault/exit; planned swaps never call this helper.
        crate::slm::invalidate_inference_routes_on_gpu(identity.gpu, reason);
    }
}

/// Whether the Vulkan llama-server is up and ready to serve inference.
pub fn available() -> bool {
    let generation = SERVER_GENERATION.load(Ordering::Acquire);
    AVAILABLE.load(Ordering::Acquire)
        && SERVER_PID.load(Ordering::Acquire) != 0
        && SERVER_EXITING_GENERATION.load(Ordering::Acquire) != generation
}

/// Under one lifecycle lock, either retain an already healthy exact server or synchronously stop
/// and reap every other child before the caller allocates an in-process model. This also waits out
/// an in-flight start whose PID has not been published yet.
pub fn reuse_exact_or_stop(gguf_path: &str, gpu: usize) -> bool {
    let _lifecycle = lifecycle().lock().unwrap_or_else(|p| p.into_inner());
    if available_for(gguf_path, gpu) {
        return true;
    }
    retire_server_for_replacement_locked();
    false
}

/// The subprocess is a singleton and can serve only the one model/device it was launched with.
/// Callers must use this identity-aware gate; `available()` alone is insufficient on a mixed rig.
pub fn available_for(gguf_path: &str, gpu: usize) -> bool {
    if !available() {
        return false;
    }
    let generation = SERVER_GENERATION.load(Ordering::Acquire);
    let pid = SERVER_PID.load(Ordering::Acquire);
    server_identity()
        .lock()
        .map(|id| {
            id.as_ref().map(|active| {
                exact_process_owner(active.generation, active.pid, generation, pid)
                    && active.gguf == normalized_model_path(gguf_path)
                    && active.gpu == gpu
            }) == Some(true)
        })
        .unwrap_or(false)
}

/// Publish a successful non-empty generation as a route proof without allowing a concurrent child
/// exit to invalidate first and then be overwritten by a late success recorder. Planned swaps take
/// `lifecycle()`, while natural-exit cleanup takes the identity mutex before invalidating, so the
/// final state is ordered correctly whichever side wins.
pub fn commit_route_success(gguf_path: &str, gpu: usize, model_id: &[u8; 32]) -> bool {
    let _lifecycle = lifecycle().lock().unwrap_or_else(|p| p.into_inner());
    let wanted_path = normalized_model_path(gguf_path);
    let generation = SERVER_GENERATION.load(Ordering::Acquire);
    let pid = SERVER_PID.load(Ordering::Acquire);
    commit_with_owned_identity(
        server_identity(),
        |active| {
            AVAILABLE.load(Ordering::Acquire)
                && SERVER_EXITING_GENERATION.load(Ordering::Acquire) != generation
                && exact_process_owner(active.generation, active.pid, generation, pid)
                && active.gguf == wanted_path
                && active.gpu == gpu
                && active.model_id.as_ref() == Some(model_id)
        },
        |_| {
            crate::slm::record_serveable_on(model_id, gpu);
            crate::slm::mark_model_available(model_id, "llama_server_generation_success");
        },
    )
}

/// ggml/Vulkan `main_gpu` backing the active server. Used only to cross-map that server to an
/// exact OpenCL worker by PCI; CUDA servers leave this unset.
pub fn vulkan_server_ggml_device() -> Option<i32> {
    let device = SERVER_VK_DEVICE.load(Ordering::Acquire);
    (device >= 0).then_some(device)
}

/// Locate the `llama-server` binary: `KERYX_LLAMA_SERVER=<path>` wins (power users / testing),
/// else the bundled one next to our own executable (shipped in the AMD package; NVIDIA packages
/// bundle it from the llama.cpp-CUDA phase onward).
fn server_binary() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("KERYX_LLAMA_SERVER") {
        let pb = std::path::PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
        log::warn!("llama server: KERYX_LLAMA_SERVER points at a missing file — ignoring.");
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let bin = dir.join("llama-server");
    if bin.exists() {
        Some(bin)
    } else {
        None
    }
}

/// Whether a subprocess fallback is actually configured/bundled. Callers use this before evicting
/// a healthy mining walk to make room for a server that may not exist.
pub fn configured() -> bool {
    server_binary().is_some()
}

fn parse_vulkan_device(raw: Option<&str>) -> Option<i32> {
    raw.and_then(|s| s.trim().parse::<i32>().ok()).filter(|gpu| *gpu >= 0)
}

fn explicit_vulkan_device() -> Option<i32> {
    let raw = std::env::var("KERYX_LLAMA_VK_DEVICE").ok();
    parse_vulkan_device(raw.as_deref())
}

/// Reserve an explicit loopback port or ask the OS for an ephemeral one. The listener stays open
/// until the monitor thread is ready to spawn the child, eliminating the ordinary check/use gap;
/// a per-instance API key below provides the authoritative child identity after the handoff.
fn reserve_server_port(requested: u16) -> std::io::Result<(u16, TcpListener)> {
    let listener = TcpListener::bind(("127.0.0.1", requested))?;
    let port = listener.local_addr()?.port();
    Ok((port, listener))
}

struct SpawnAttemptCompletion(std::sync::mpsc::Sender<()>);

impl Drop for SpawnAttemptCompletion {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

fn wait_for_spawn_cleanup(done: std::sync::mpsc::Receiver<()>) -> bool {
    done.recv().is_ok()
}

/// Publish a spawned PID to its monitor or, if the monitor already abandoned that handoff, wait
/// until it has killed and reaped the child. Returning false is therefore a GPU-ownership barrier,
/// not merely an acknowledgement failure.
fn acknowledge_spawn_or_wait_cleanup(
    ack: std::sync::mpsc::SyncSender<()>,
    done: std::sync::mpsc::Receiver<()>,
) -> bool {
    if ack.send(()).is_ok() {
        true
    } else {
        let _ = wait_for_spawn_cleanup(done);
        false
    }
}

fn new_server_api_key() -> String {
    let mut entropy = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut entropy);
    format!("keryx-{}", hex::encode(entropy))
}

/// Spawn the Vulkan `llama-server` for `gguf_path` and block until it's healthy (or give up).
/// Returns true on success. Idempotent-ish: call once at startup. Any failure withdraws this route.
///
/// Device: pinned to ONE discrete GPU via ggml's `--main-gpu`, auto-selected from the exact OpenCL
/// worker PCI allowlist. Without the pin, ggml-vulkan layer-splits across EVERY visible Vulkan
/// device — iGPU included (issue #18: model layers can land in UMA system RAM) — stealing VRAM from
/// every mining card. `KERYX_LLAMA_VK_DEVICE` explicitly overrides the worker constraint with a
/// ggml `main_gpu` index and may intentionally target an inference-only card.
pub fn try_start(gguf_path: &str, port: u16) -> bool {
    let logical_gpu = std::env::var("KERYX_LLAMA_CUDA_DEVICE")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or_else(crate::slm::inference_gpu_ordinal);
    try_start_on(gguf_path, port, logical_gpu)
}

/// Start (or safely replace) the singleton server for one exact model/card. Lifecycle operations
/// are serialized, and replacement waits for the old monitor to reap its child before a new PID is
/// published, so a late old monitor cannot clear the new server's state.
pub fn try_start_on(gguf_path: &str, port: u16, logical_gpu: usize) -> bool {
    let _lifecycle = lifecycle().lock().unwrap_or_else(|p| p.into_inner());
    // The engine was unloaded to give its card's VRAM to the possession blob — a GPU llama-server
    // would pin to that SAME discrete card (pick_discrete_ggml_device) and re-occupy it, starving
    // the blob/walk again on small cards. Explicit KERYX_LLAMA_VK_DEVICE (operator pins another
    // card) still wins; otherwise no GPU inference route is advertised. The deprecated CPU
    // emergency fallback remains opt-in only. Mining VRAM takes priority.
    #[cfg(all(feature = "pom-opencl", unix))]
    if crate::llama_engine_vk::evicted_for_vram() && explicit_vulkan_device().is_none() {
        log::warn!(
            "llama server: NOT starting the GPU inference server — the in-process engine was unloaded \
             to free VRAM for the PoM blob, and the server would land on the same card. No GPU \
             inference route is advertised (set KERYX_LLAMA_VK_DEVICE=<other ggml device> to pin it elsewhere)."
        );
        return false;
    }
    if available_for(gguf_path, logical_gpu) {
        return true;
    }
    retire_server_for_replacement_locked();
    let server_bin = match server_binary() {
        Some(b) => b,
        None => {
            #[cfg(feature = "pom-opencl")]
            log::info!("llama_vulkan: no bundled llama-server next to the binary — no subprocess GPU inference route.");
            #[cfg(not(feature = "pom-opencl"))]
            log::info!(
                "llama server: no bundled llama-server (and no KERYX_LLAMA_SERVER) — this GPU inference route remains unavailable; the deprecated CPU emergency fallback runs only when explicitly enabled."
            );
            return false;
        }
    };
    if !std::path::Path::new(gguf_path).exists() {
        log::warn!("llama_vulkan: GGUF {gguf_path} not present yet — cannot start GPU inference server.");
        return false;
    }
    let (port, port_reservation) = match reserve_server_port(port) {
        Ok(reserved) => reserved,
        Err(e) => {
            log::warn!(
                "llama server: localhost port {} is already in use or unavailable ({e}) — refusing to trust another process",
                port
            );
            return false;
        }
    };
    let api_key = new_server_api_key();
    let exe_dir = server_bin.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    // The bundled llama-server's ggml/vulkan .so live next to it; libvulkan comes from the system.
    let ld = format!(
        "{}:/usr/lib/x86_64-linux-gnu{}",
        exe_dir.display(),
        std::env::var("LD_LIBRARY_PATH").map(|s| format!(":{s}")).unwrap_or_default()
    );

    let mut cmd = Command::new(&server_bin);
    #[cfg(all(feature = "pom-opencl", unix))]
    let mut vulkan_device = -1i32;
    #[cfg(all(feature = "pom-opencl", unix))]
    let mut dedication_attempt = ServerDedicationAttempt { armed: false };
    #[cfg(not(all(feature = "pom-opencl", unix)))]
    let vulkan_device = -1i32;
    cmd.args([
        "-m",
        gguf_path,
        "-ngl",
        "99", // all layers on the GPU
        "--host",
        "127.0.0.1",
        "--port",
        &port.to_string(),
        "-c",
        "4096",
        "--no-webui",
        "--api-key",
        &api_key,
        "-t",
        "2", // few CPU threads — the GPU does the work
    ])
    .env("LD_LIBRARY_PATH", ld)
    .stdout(Stdio::null())
    .stderr(Stdio::null());
    // AMD/Vulkan llama-server: pin it to one discrete GPU (see the doc comment above).
    // KERYX_LLAMA_VK_DEVICE (an explicit ggml `main_gpu` index) wins; else the engine .so picks
    // the discrete GPU against ggml's OWN device list (issue #18) — valid for this subprocess
    // because it shares the same bundled ggml. Passed as --main-gpu with single-GPU split so
    // ggml never layer-splits onto the iGPU; NOT GGML_VK_VISIBLE_DEVICES (its cross-instance
    // index mislocates or asserts on iGPU rigs). Neither = llama.cpp's own default.
    #[cfg(all(feature = "pom-opencl", unix))]
    {
        let explicit = explicit_vulkan_device();
        let dev = explicit.or_else(crate::llama_engine_vk::pick_discrete_ggml_device);
        if let Some(dev) = dev {
            // Resolve ggml's index back to the exact selected OpenCL worker and, when the memory
            // planner requires it, free that worker's resident walk before the subprocess can
            // allocate its model. Post-health mapping is too late to prevent a load-time OOM.
            if !crate::pom_opencl::prepare_vulkan_server_device(dev) {
                return false;
            }
            dedication_attempt.armed = true;
            vulkan_device = dev;
            log::info!("llama server: pinning Vulkan llama-server to ggml device {dev} (--main-gpu, split-mode none).");
            cmd.args(["--main-gpu", &dev.to_string(), "--split-mode", "none"]);
        } else if crate::llama_engine_vk::auto_device_allowlist_active() {
            crate::pom_opencl::release_provisional_vulkan_dedication();
            log::warn!(
                "llama server: no trusted ggml device matches the selected OpenCL worker PCI \
                 allowlist — refusing an unpinned/default Vulkan server. Rebuild \
                 libkeryx-llama-vk.so or set KERYX_LLAMA_VK_DEVICE explicitly."
            );
            return false;
        }
    }
    #[cfg(not(all(feature = "pom-opencl", unix)))]
    if let Ok(dev) = std::env::var("KERYX_LLAMA_VK_DEVICE") {
        cmd.env("GGML_VK_VISIBLE_DEVICES", dev);
    }

    // NVIDIA/CUDA llama-server: PIN it to the fully resolved inference GPU — llama.cpp's default
    // is to layer-split across ALL visible GPUs, which would
    // steal VRAM from every mining card. `KERYX_LLAMA_CUDA_DEVICE` overrides the ordinal.
    #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
    {
        // `logical_gpu` is already the fully resolved route (including the legacy env override).
        // Always translate through the parent's visibility map; copying raw "1" into a child of
        // CUDA_VISIBLE_DEVICES=2,0 would select physical GPU 1 instead of parent logical GPU 1.
        let dev = child_cuda_visible_device(logical_gpu);
        cmd.env("CUDA_VISIBLE_DEVICES", &dev);
        log::info!(
            "llama server: pinning CUDA llama-server to logical GPU {} (child CUDA_VISIBLE_DEVICES={}).",
            logical_gpu,
            dev
        );
    }

    // Spawn from a DEDICATED MONITOR THREAD that outlives the caller. Two reasons:
    // (1) orphan fix — on Linux the child gets PR_SET_PDEATHSIG(SIGKILL), so the kernel kills
    //     llama-server whenever the miner dies, on EVERY exit path incl. `kill -9` and panics
    //     (previously it survived the miner and squatted on VRAM). PDEATHSIG fires when the
    //     SPAWNING THREAD dies — not the process — hence the spawner must stay alive, parked in
    //     wait(), for the child's whole life.
    // (2) the wait() reaps the child (no zombie) and flips AVAILABLE off if the server crashes
    //     mid-session, so the failed GPU route is withdrawn instead of timing out per request.
    let generation = SERVER_GENERATION.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
    let (tx, rx) = std::sync::mpsc::channel::<Result<(u32, std::sync::mpsc::SyncSender<()>), String>>();
    let (spawn_done_tx, spawn_done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        // If the starter gives up before `cmd.spawn()` reports, this completion token is released
        // only after the late child (if any) has been killed and reaped. The caller waits on it
        // before releasing the lifecycle lock, so no in-process engine can allocate concurrently.
        let _spawn_attempt = SpawnAttemptCompletion(spawn_done_tx);
        // Hand the reserved endpoint to this exact child at the last possible moment. Another
        // process winning this tiny handoff race still cannot pass the authenticated /props gate.
        drop(port_reservation);
        #[cfg(target_os = "linux")]
        unsafe {
            use std::os::unix::process::CommandExt;
            cmd.pre_exec(|| {
                nix::libc::prctl(nix::libc::PR_SET_PDEATHSIG, nix::libc::SIGKILL as nix::libc::c_ulong);
                Ok(())
            });
        }
        match cmd.spawn() {
            Ok(mut ch) => {
                let pid = ch.id();
                // Do not let the monitor race ahead of publication. If the starter timed out and
                // dropped `rx`, kill+reap here instead of leaving an untracked VRAM-owning child.
                let (ack_tx, ack_rx) = std::sync::mpsc::sync_channel(0);
                if tx.send(Ok((pid, ack_tx))).is_err() || ack_rx.recv_timeout(Duration::from_secs(10)).is_err() {
                    let _ = ch.kill();
                    let _ = ch.wait();
                    return;
                }
                let status = ch.wait(); // parks here for the child's lifetime (keeps PDEATHSIG armed)
                if SERVER_GENERATION.load(Ordering::Acquire) == generation && SERVER_PID.load(Ordering::Acquire) == pid
                {
                    let _ = finish_monitor_cleanup(&SERVER_PID, pid, || {
                        // Mark exit first. A concurrent health publisher either completed before
                        // this cleanup (and is cleared below), or sees the marker and refuses the
                        // already-dead generation. PID stays non-zero until all cleanup is done.
                        SERVER_EXITING_GENERATION.store(generation, Ordering::Release);
                        SERVER_VK_DEVICE.store(-1, Ordering::Release);
                        PORT.store(0, Ordering::Release);
                        #[cfg(all(feature = "pom-opencl", unix))]
                        crate::pom_opencl::release_vulkan_server_dedication();
                        let was_up = AVAILABLE.swap(false, Ordering::AcqRel);
                        let exited_identity = take_identity_for_generation(generation);
                        if was_up {
                            // A successful self-test is an exact `(model, GPU)` runtime proof. Once
                            // the child that earned it has exited, retaining the cached proof would
                            // leave ai:cap advertised until a real request pays the full failed path.
                            if let Some(identity) = exited_identity {
                                invalidate_identity_route(&identity, "llama_server_exited");
                            }
                            log::warn!("llama server: llama-server exited ({status:?}) — GPU route withdrawn.");
                        }
                    });
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e.to_string()));
            }
        }
    });
    let spawned_pid = match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Ok((pid, ack))) => {
            SERVER_PID.store(pid, Ordering::Release);
            if !acknowledge_spawn_or_wait_cleanup(ack, spawn_done_rx) {
                // The monitor has already gone away. Never retain a PID which nobody can reap and
                // whose liveness we cannot prove. Its completion barrier guarantees the child no
                // longer owns GPU memory before PID zero is published and lifecycle is released.
                SERVER_PID.store(0, Ordering::Release);
                return false;
            }
            pid
        }
        Ok(Err(e)) => {
            log::warn!("llama server: failed to spawn llama-server ({e}) — subprocess GPU route unavailable.");
            return false;
        }
        Err(_) => {
            log::warn!("llama server: spawn did not report within 10s — subprocess GPU route unavailable.");
            // Make the monitor's tx fail immediately. Its failure branch kills and reaps any child
            // that appeared after our timeout; do not return GPU ownership until that cleanup has
            // completed. A genuinely wedged OS spawn remains fail-closed under the lifecycle lock.
            drop(rx);
            if !wait_for_spawn_cleanup(spawn_done_rx) {
                log::error!("llama server: spawn monitor ended without its cleanup completion barrier");
            }
            return false;
        }
    };

    // Poll /health until ready (model load on the GPU can take ~30-60s).
    let health = format!("http://127.0.0.1:{port}/health");
    let deadline = Instant::now() + Duration::from_secs(180);
    log::info!("llama_vulkan: starting Vulkan llama-server on port {port} (loading {gguf_path} into VRAM)…");
    while Instant::now() < deadline {
        // exit early if the child already died (the monitor thread clears SERVER_PID on exit)
        if SERVER_PID.load(Ordering::Relaxed) == 0 {
            log::warn!("llama server: llama-server exited during startup — subprocess GPU route unavailable (wrong GPU/driver, OOM?).");
            return false;
        }
        let auth = format!("Bearer {api_key}");
        let owned_props = format!("http://127.0.0.1:{port}/props");
        if ureq::get(&health).timeout(Duration::from_secs(2)).call().is_ok()
            && ureq::get(&owned_props).timeout(Duration::from_secs(2)).set("Authorization", &auth).call().is_ok()
        {
            set_server_identity(Some(ServerIdentity {
                gguf: normalized_model_path(gguf_path),
                model_id: crate::slm::model_id_for_gguf(gguf_path),
                api_key: api_key.clone(),
                gpu: logical_gpu,
                generation,
                pid: spawned_pid,
            }));
            SERVER_VK_DEVICE.store(vulkan_device, Ordering::Release);
            AVAILABLE.store(true, Ordering::Release);
            PORT.store(port, Ordering::Release);
            // A successful health response and child exit can race. The monitor may have cleared
            // this PID while AVAILABLE was still false and before an identity existed; publishing
            // unconditionally would then advertise a dead, unmonitored route. Publish first so an
            // exit after this point is handled by the monitor, then verify this exact generation
            // still owns the expected PID to cover exits before or during publication.
            if !exact_process_publishable(
                generation,
                spawned_pid,
                SERVER_GENERATION.load(Ordering::Acquire),
                SERVER_PID.load(Ordering::Acquire),
                SERVER_EXITING_GENERATION.load(Ordering::Acquire),
            ) {
                AVAILABLE.store(false, Ordering::Release);
                PORT.store(0, Ordering::Release);
                SERVER_VK_DEVICE.store(-1, Ordering::Release);
                let _ = take_identity_for_generation(generation);
                log::warn!(
                    "llama server: process exited while its healthy route was being published; route withdrawn."
                );
                return false;
            }
            #[cfg(feature = "pom-opencl")]
            log::info!("llama_vulkan: ✓ AMD GPU inference ready (Vulkan llama-server on port {port}).");
            #[cfg(not(feature = "pom-opencl"))]
            log::info!("llama server: ✓ GPU inference ready (llama.cpp llama-server on port {port}).");
            #[cfg(all(feature = "pom-opencl", unix))]
            {
                dedication_attempt.armed = false;
            }
            return true;
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    log::warn!("llama_vulkan: llama-server did not become healthy in time — GPU inference unavailable.");
    stop_locked(true, "llama_server_start_timeout");
    false
}

/// Generate up to `max_tokens` from `prompt` via the running Vulkan llama-server. Returns the
/// completion text, or None on any error. Temperature/top_p match the legacy engine path (the text
/// is user-facing — exact tokens are not consensus-relevant).
pub fn generate(prompt: &str, max_tokens: usize) -> Option<String> {
    let _lifecycle = lifecycle().lock().unwrap_or_else(|p| p.into_inner());
    generate_locked(prompt, max_tokens)
}

fn generate_locked(prompt: &str, max_tokens: usize) -> Option<String> {
    if !available()
        || crate::slm::validate_inference_request(prompt, max_tokens, crate::slm::DEFAULT_INFERENCE_DEADLINE_MS)
            .is_err()
    {
        return None;
    }
    let port = PORT.load(Ordering::Relaxed);
    let api_key = server_identity().lock().ok()?.as_ref()?.api_key.clone();
    let auth = format!("Bearer {api_key}");
    let url = format!("http://127.0.0.1:{port}/completion");
    let body = serde_json::json!({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "cache_prompt": false,
        "stream": false,
    });
    // ureq is built without its `json` feature here, so serialize/parse with serde_json directly.
    let payload = serde_json::to_string(&body).ok()?;
    // `deadline_ms` on the Stratum wire is explicitly an admission/card-queue budget, not an
    // end-to-end cancellation contract. Still impose an independent hard runtime ceiling so one
    // valid 2,048-token request cannot hold the global PoW pause for the historical 10–60 minutes.
    // On timeout stop_locked kills and reaps the child before mining is allowed to resume.
    let configured_timeout = std::env::var("KERYX_LLAMA_HTTP_TIMEOUT_SEC").ok();
    let timeout_secs = bounded_http_timeout_secs(configured_timeout.as_deref());
    let resp = match ureq::post(&url)
        .timeout(Duration::from_secs(timeout_secs))
        .set("Content-Type", "application/json")
        .set("Authorization", &auth)
        .send_string(&payload)
    {
        Ok(r) => r,
        Err(e) => {
            log::warn!("llama_vulkan: /completion request failed ({e}) — challenge cannot be served.");
            // A client-side timeout/disconnect does not prove llama-server cancelled its GPU work.
            // Kill and synchronously reap it while the caller still holds the mining drain.
            stop_locked(true, "llama_server_request_failed");
            return None;
        }
    };
    const MAX_RESPONSE_BYTES: u64 = 1024 * 1024;
    let mut raw = String::new();
    if let Err(error) = resp.into_reader().take(MAX_RESPONSE_BYTES + 1).read_to_string(&mut raw) {
        log::warn!("llama_vulkan: could not read /completion response ({error}) — retiring the server route");
        stop_locked(true, "llama_server_response_read_failed");
        return None;
    }
    if raw.len() as u64 > MAX_RESPONSE_BYTES {
        log::warn!("llama_vulkan: completion response exceeded 1 MiB — retiring the server route");
        stop_locked(true, "llama_server_response_oversized");
        return None;
    }
    let json: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(json) => json,
        Err(error) => {
            log::warn!("llama_vulkan: malformed /completion JSON ({error}) — retiring the server route");
            stop_locked(true, "llama_server_response_invalid_json");
            return None;
        }
    };
    let text = match json.get("content").and_then(serde_json::Value::as_str) {
        Some(text) => text.to_string(),
        None => {
            log::warn!("llama_vulkan: /completion response has no string content — retiring the server route");
            stop_locked(true, "llama_server_response_missing_content");
            return None;
        }
    };
    if text.is_empty() {
        log::warn!("llama_vulkan: /completion response was empty — retiring the server route");
        stop_locked(true, "llama_server_response_empty");
        None
    } else {
        Some(text)
    }
}

/// Identity-checked generation for the singleton server. A request for another model/card is not
/// silently answered by whichever primary model happened to start first.
pub fn generate_for(gguf_path: &str, gpu: usize, prompt: &str, max_tokens: usize) -> Option<String> {
    let _lifecycle = lifecycle().lock().unwrap_or_else(|p| p.into_inner());
    if !available_for(gguf_path, gpu) {
        return None;
    }
    generate_locked(prompt, max_tokens)
}

/// Kill the llama-server (best-effort). Called on failed startup; the monitor thread reaps the
/// child and clears SERVER_PID. (On Linux, normal miner death needs no call at all — PDEATHSIG
/// kills the child from the kernel side.)
pub fn stop() {
    let _lifecycle = lifecycle().lock().unwrap_or_else(|p| p.into_inner());
    stop_locked(true, "llama_server_stopped");
}

fn stop_locked(invalidate_route: bool, reason: &str) {
    AVAILABLE.store(false, Ordering::Relaxed);
    PORT.store(0, Ordering::Release);
    SERVER_VK_DEVICE.store(-1, Ordering::Release);
    let stopped_identity = take_identity_for_generation(SERVER_GENERATION.load(Ordering::Acquire));
    if invalidate_route {
        if let Some(identity) = stopped_identity {
            // A timeout, request failure, explicit stop, or unresponsive replacement target closes
            // the exact route just as a natural child exit does. A freshly authenticated planned
            // model swap passes `invalidate_route=false` and deliberately preserves ability proofs.
            // Do this before waiting for the monitor; AVAILABLE is already false, so its reaper does
            // not duplicate the withdrawal.
            invalidate_identity_route(&identity, reason);
        }
    }
    let pid = SERVER_PID.load(Ordering::Acquire);
    #[cfg(unix)]
    if pid != 0 {
        let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), nix::sys::signal::Signal::SIGKILL);
    }
    #[cfg(windows)]
    if pid != 0 {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(any(unix, windows)))]
    if pid != 0 {
        log::error!("llama server: cannot terminate child on this platform; waiting for it to exit");
    }
    // The dedicated monitor owns Child and is the only reaper. Do not return the GPU to mining
    // until it has observed process exit and cleared this exact PID.
    while pid != 0 && SERVER_PID.load(Ordering::Acquire) == pid {
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        acknowledge_spawn_or_wait_cleanup, bounded_http_timeout_secs, commit_with_owned_identity, exact_process_owner,
        exact_process_publishable, finish_monitor_cleanup, parse_vulkan_device, planned_replacement_invalidates,
        replacement_action, reserve_server_port, server_identity, take_identity_for_generation, visible_device_token,
        wait_for_spawn_cleanup, ReplacementAction, ServerIdentity, SpawnAttemptCompletion,
    };
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Barrier, Mutex};

    #[test]
    fn preserves_parent_visible_device_mapping() {
        assert_eq!(visible_device_token(Some("2,0"), 0), "2");
        assert_eq!(visible_device_token(Some("2,0"), 1), "0");
        assert_eq!(visible_device_token(Some("GPU-a, GPU-b"), 1), "GPU-b");
    }

    #[test]
    fn old_monitor_cannot_take_newer_server_identity() {
        let mut identity = server_identity().lock().unwrap_or_else(|p| p.into_inner());
        *identity = Some(ServerIdentity {
            gguf: std::path::PathBuf::from("newer.gguf"),
            model_id: Some([0x42; 32]),
            api_key: "test-key".into(),
            gpu: 7,
            generation: 42,
            pid: 4242,
        });
        drop(identity);

        assert!(take_identity_for_generation(41).is_none());
        assert_eq!(
            server_identity().lock().unwrap_or_else(|p| p.into_inner()).as_ref().map(|id| id.generation),
            Some(42)
        );
        assert_eq!(take_identity_for_generation(42).map(|id| id.gpu), Some(7));
        assert!(server_identity().lock().unwrap_or_else(|p| p.into_inner()).is_none());
    }

    #[test]
    fn falls_back_to_logical_ordinal_when_unrestricted_or_out_of_range() {
        assert_eq!(visible_device_token(None, 3), "3");
        assert_eq!(visible_device_token(Some("2,0"), 3), "3");
    }

    #[test]
    fn server_runtime_timeout_is_hard_bounded() {
        assert_eq!(bounded_http_timeout_secs(None), 120);
        assert_eq!(bounded_http_timeout_secs(Some("5")), 30);
        assert_eq!(bounded_http_timeout_secs(Some("90")), 90);
        assert_eq!(bounded_http_timeout_secs(Some("3600")), 120);
        assert_eq!(bounded_http_timeout_secs(Some("invalid")), 120);
    }

    #[test]
    fn invalid_vulkan_override_is_not_operator_authority() {
        assert_eq!(parse_vulkan_device(Some("0")), Some(0));
        assert_eq!(parse_vulkan_device(Some(" 7 ")), Some(7));
        assert_eq!(parse_vulkan_device(None), None);
        assert_eq!(parse_vulkan_device(Some("")), None);
        assert_eq!(parse_vulkan_device(Some("garbage")), None);
        assert_eq!(parse_vulkan_device(Some("-1")), None);
    }

    #[test]
    fn server_port_reservation_refuses_an_existing_listener() {
        let (port, reservation) = reserve_server_port(0).expect("ephemeral loopback reservation");
        assert_ne!(port, 0);
        assert!(reserve_server_port(port).is_err(), "an explicit occupied port must fail closed");
        drop(reservation);
        let (_same_port, replacement) = reserve_server_port(port).expect("port should be reusable after release");
        drop(replacement);
    }

    #[test]
    fn route_publication_requires_the_exact_live_process() {
        assert!(exact_process_owner(9, 1234, 9, 1234));
        assert!(!exact_process_owner(9, 1234, 9, 0), "monitor-cleared PID must fail closed");
        assert!(!exact_process_owner(9, 1234, 10, 1234), "replacement generation must not validate old route");
        assert!(!exact_process_owner(9, 0, 9, 0), "zero is never a live child PID");
        assert!(exact_process_publishable(9, 1234, 9, 1234, 8));
        assert!(
            !exact_process_publishable(9, 1234, 9, 1234, 9),
            "an exiting generation must not republish after monitor cleanup starts"
        );
    }

    #[test]
    fn monitor_pid_zero_is_a_cleanup_completion_barrier() {
        let pid = Arc::new(AtomicU32::new(4242));
        let cleanup_done = Arc::new(AtomicBool::new(false));
        let entered_cleanup = Arc::new(Barrier::new(2));
        let release_cleanup = Arc::new(Barrier::new(2));

        let monitor = {
            let pid = Arc::clone(&pid);
            let cleanup_done = Arc::clone(&cleanup_done);
            let entered_cleanup = Arc::clone(&entered_cleanup);
            let release_cleanup = Arc::clone(&release_cleanup);
            std::thread::spawn(move || {
                assert!(finish_monitor_cleanup(&pid, 4242, || {
                    entered_cleanup.wait();
                    release_cleanup.wait();
                    cleanup_done.store(true, Ordering::Release);
                }));
            })
        };

        entered_cleanup.wait();
        assert_eq!(pid.load(Ordering::Acquire), 4242, "PID cleared before cleanup completed");
        release_cleanup.wait();
        while pid.load(Ordering::Acquire) != 0 {
            std::thread::yield_now();
        }
        assert!(cleanup_done.load(Ordering::Acquire));
        monitor.join().unwrap();
    }

    #[test]
    fn replacement_waits_for_an_exiting_monitors_cleanup() {
        assert_eq!(replacement_action(0, 9, 9), ReplacementAction::None);
        assert_eq!(replacement_action(1234, 9, 8), ReplacementAction::StopPlanned);
        assert_eq!(
            replacement_action(1234, 9, 9),
            ReplacementAction::WaitForMonitor,
            "replacement must not take/discard an exiting monitor's route identity"
        );
    }

    #[test]
    fn unresponsive_child_is_faulted_even_before_monitor_marks_exit() {
        assert!(planned_replacement_invalidates(false));
        assert!(!planned_replacement_invalidates(true));
    }

    #[test]
    fn monitor_invalidation_wins_after_an_inflight_success_commit() {
        let identity = Arc::new(Mutex::new(Some(ServerIdentity {
            gguf: std::path::PathBuf::from("model.gguf"),
            model_id: Some([0x51; 32]),
            api_key: "test-key".into(),
            gpu: 0,
            generation: 7,
            pid: 700,
        })));
        let available = Arc::new(AtomicBool::new(true));
        let proof = Arc::new(AtomicBool::new(false));
        let commit_entered = Arc::new(Barrier::new(2));
        let release_commit = Arc::new(Barrier::new(2));

        let publisher = {
            let identity = Arc::clone(&identity);
            let available = Arc::clone(&available);
            let proof = Arc::clone(&proof);
            let commit_entered = Arc::clone(&commit_entered);
            let release_commit = Arc::clone(&release_commit);
            std::thread::spawn(move || {
                commit_with_owned_identity(
                    &identity,
                    |_| available.load(Ordering::Acquire),
                    |_| {
                        commit_entered.wait();
                        release_commit.wait();
                        proof.store(true, Ordering::Release);
                    },
                )
            })
        };

        commit_entered.wait();
        available.store(false, Ordering::Release);
        let monitor = {
            let identity = Arc::clone(&identity);
            let proof = Arc::clone(&proof);
            std::thread::spawn(move || {
                let _ = identity.lock().unwrap_or_else(|p| p.into_inner()).take();
                proof.store(false, Ordering::Release);
            })
        };
        release_commit.wait();

        assert!(publisher.join().unwrap(), "the success owned identity before exit cleanup");
        monitor.join().unwrap();
        assert!(!proof.load(Ordering::Acquire), "late success must not resurrect a monitor-invalidated proof");
    }

    #[test]
    fn abandoned_spawn_waits_for_cleanup_completion() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = {
            let cleaned = Arc::clone(&cleaned);
            std::thread::spawn(move || {
                let completion = SpawnAttemptCompletion(done_tx);
                cleaned.store(true, Ordering::Release);
                drop(completion);
            })
        };

        assert!(wait_for_spawn_cleanup(done_rx));
        assert!(cleaned.load(Ordering::Acquire), "cleanup must happen-before the timeout path returns");
        worker.join().unwrap();
    }

    #[test]
    fn rejected_spawn_ack_waits_for_cleanup_completion() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let (ack_tx, ack_rx) = std::sync::mpsc::sync_channel(0);
        drop(ack_rx);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = {
            let cleaned = Arc::clone(&cleaned);
            std::thread::spawn(move || {
                let completion = SpawnAttemptCompletion(done_tx);
                cleaned.store(true, Ordering::Release);
                drop(completion);
            })
        };

        assert!(!acknowledge_spawn_or_wait_cleanup(ack_tx, done_rx));
        assert!(cleaned.load(Ordering::Acquire), "failed publication ack must not return before child cleanup");
        worker.join().unwrap();
    }
}
