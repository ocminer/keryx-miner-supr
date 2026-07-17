//! GPU inference via a bundled llama.cpp `llama-server` subprocess (Vulkan on AMD, CUDA on NVIDIA).
//!
//! AMD: candle 0.9 has no AMD-GPU backend, so the `pom-opencl` build generates the OPoI LLM text
//! with llama.cpp (Vulkan) on the AMD GPU instead of candle on the CPU.
//! NVIDIA (`pom-cuda`, Phase 1 of the candle-independence plan): when a CUDA `llama-server` is
//! bundled next to the miner (or pointed at via `KERYX_LLAMA_SERVER`), inference prefers it over
//! candle — llama.cpp tracks new GGUF architectures far faster than candle, so a future
//! consensus-pinned model candle can't run stays servable. Without the binary the behavior is
//! byte-identical to before (candle GPU/CPU). We spawn `llama-server` once (the GGUF stays
//! resident in VRAM) and `generate()` HTTP-POSTs `/completion`.
//!
//! This is OPoI-safe: consensus verifies the fixed-point `model_fixed` commitment (computed
//! separately, bit-exact on all hardware) — the Gemma *text* is user-facing utility with no
//! determinism requirement, so a non-candle engine is fine.
//!
//! Best-effort + self-disabling: if the bundled `llama-server` is missing, Vulkan/AMD GPU is
//! unavailable, or the server doesn't come up healthy, `AVAILABLE` stays false and the caller
//! (`slm::load_and_run_inference`) transparently falls back to candle-CPU.

use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};
use std::time::{Duration, Instant};

static AVAILABLE: AtomicBool = AtomicBool::new(false);
static PORT: AtomicU16 = AtomicU16::new(0);
/// PID of the running llama-server (0 = none). The Child handle itself is owned by the monitor
/// thread that spawned it (see try_start) — it reaps the child and clears this on exit.
static SERVER_PID: AtomicU32 = AtomicU32::new(0);

/// Whether the Vulkan llama-server is up and ready to serve inference.
pub fn available() -> bool {
    AVAILABLE.load(Ordering::Relaxed)
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

/// Spawn the Vulkan `llama-server` for `gguf_path` and block until it's healthy (or give up).
/// Returns true on success. Idempotent-ish: call once at startup. Any failure → false (CPU fallback).
///
/// Device: pinned to ONE discrete GPU via `GGML_VK_VISIBLE_DEVICES` (first discrete AMD card by
/// default). Without the pin, ggml-vulkan layer-splits across EVERY visible Vulkan device — iGPU
/// included (issue #18: on rigs with integrated graphics, Vulkan device 0 is the Intel/AMD APU,
/// so model layers land in UMA system RAM) — stealing VRAM from every mining card. Override the
/// auto-pick with `KERYX_LLAMA_VK_DEVICE` (a `GGML_VK_VISIBLE_DEVICES` value, e.g. "1").
pub fn try_start(gguf_path: &str, port: u16) -> bool {
    if available() {
        return true;
    }
    let server_bin = match server_binary() {
        Some(b) => b,
        None => {
            #[cfg(feature = "pom-opencl")]
            log::info!("llama_vulkan: no bundled llama-server next to the binary — OPoI inference stays on CPU.");
            #[cfg(not(feature = "pom-opencl"))]
            log::info!("llama server: no bundled llama-server (and no KERYX_LLAMA_SERVER) — OPoI inference stays on candle.");
            return false;
        }
    };
    if !std::path::Path::new(gguf_path).exists() {
        log::warn!("llama_vulkan: GGUF {gguf_path} not present yet — cannot start GPU inference server.");
        return false;
    }
    let exe_dir = server_bin.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    // The bundled llama-server's ggml/vulkan .so live next to it; libvulkan comes from the system.
    let ld = format!(
        "{}:/usr/lib/x86_64-linux-gnu{}",
        exe_dir.display(),
        std::env::var("LD_LIBRARY_PATH").map(|s| format!(":{s}")).unwrap_or_default()
    );

    let mut cmd = Command::new(&server_bin);
    cmd.args([
        "-m", gguf_path,
        "-ngl", "99", // all layers on the GPU
        "--host", "127.0.0.1",
        "--port", &port.to_string(),
        "-c", "4096",
        "--no-webui",
        "-t", "2", // few CPU threads — the GPU does the work
    ])
    .env("LD_LIBRARY_PATH", ld)
    .stdout(Stdio::null())
    .stderr(Stdio::null());
    // AMD/Vulkan llama-server: pin it to one discrete GPU (see the doc comment above). Same
    // selection as the in-process engine: KERYX_LLAMA_VK_DEVICE wins, else the first discrete
    // AMD Vulkan device; neither (pure-APU box / no libvulkan) = llama.cpp's own default.
    #[cfg(all(feature = "pom-opencl", unix))]
    {
        let dev = std::env::var("KERYX_LLAMA_VK_DEVICE")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(crate::llama_engine_vk::pick_discrete_vk_device);
        if let Some(dev) = dev {
            log::info!("llama server: pinning Vulkan llama-server to device(s) {dev} (GGML_VK_VISIBLE_DEVICES).");
            cmd.env("GGML_VK_VISIBLE_DEVICES", &dev);
        }
    }
    #[cfg(not(all(feature = "pom-opencl", unix)))]
    if let Ok(dev) = std::env::var("KERYX_LLAMA_VK_DEVICE") {
        cmd.env("GGML_VK_VISIBLE_DEVICES", dev);
    }

    // NVIDIA/CUDA llama-server: PIN it to one GPU (default = the same biggest-VRAM ordinal candle
    // inference uses) — llama.cpp's default is to layer-split across ALL visible GPUs, which would
    // steal VRAM from every mining card. `KERYX_LLAMA_CUDA_DEVICE` overrides the ordinal.
    #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
    {
        let dev = std::env::var("KERYX_LLAMA_CUDA_DEVICE")
            .unwrap_or_else(|_| crate::slm::inference_gpu_ordinal().to_string());
        cmd.env("CUDA_VISIBLE_DEVICES", &dev);
        log::info!("llama server: pinning CUDA llama-server to GPU {dev} (CUDA_VISIBLE_DEVICES).");
    }

    // Spawn from a DEDICATED MONITOR THREAD that outlives the caller. Two reasons:
    // (1) orphan fix — on Linux the child gets PR_SET_PDEATHSIG(SIGKILL), so the kernel kills
    //     llama-server whenever the miner dies, on EVERY exit path incl. `kill -9` and panics
    //     (previously it survived the miner and squatted on VRAM). PDEATHSIG fires when the
    //     SPAWNING THREAD dies — not the process — hence the spawner must stay alive, parked in
    //     wait(), for the child's whole life.
    // (2) the wait() reaps the child (no zombie) and flips AVAILABLE off if the server crashes
    //     mid-session, so inference self-heals back to candle instead of timing out per request.
    let (tx, rx) = std::sync::mpsc::channel::<Result<u32, String>>();
    std::thread::spawn(move || {
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
                let _ = tx.send(Ok(ch.id()));
                let status = ch.wait(); // parks here for the child's lifetime (keeps PDEATHSIG armed)
                let was_up = AVAILABLE.swap(false, Ordering::Relaxed);
                SERVER_PID.store(0, Ordering::Relaxed);
                if was_up {
                    log::warn!("llama server: llama-server exited ({status:?}) — inference falls back to candle.");
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e.to_string()));
            }
        }
    });
    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Ok(pid)) => SERVER_PID.store(pid, Ordering::Relaxed),
        Ok(Err(e)) => {
            log::warn!("llama server: failed to spawn llama-server ({e}) — falling back to candle inference.");
            return false;
        }
        Err(_) => {
            log::warn!("llama server: spawn did not report within 10s — falling back to candle inference.");
            return false;
        }
    }

    // Poll /health until ready (model load on the GPU can take ~30-60s).
    let health = format!("http://127.0.0.1:{port}/health");
    let deadline = Instant::now() + Duration::from_secs(180);
    log::info!("llama_vulkan: starting Vulkan llama-server on port {port} (loading {gguf_path} into VRAM)…");
    while Instant::now() < deadline {
        // exit early if the child already died (the monitor thread clears SERVER_PID on exit)
        if SERVER_PID.load(Ordering::Relaxed) == 0 {
            log::warn!("llama server: llama-server exited during startup — falling back to candle inference (wrong GPU/driver, OOM?).");
            return false;
        }
        if ureq::get(&health).timeout(Duration::from_secs(2)).call().is_ok() {
            AVAILABLE.store(true, Ordering::Relaxed);
            PORT.store(port, Ordering::Relaxed);
            #[cfg(feature = "pom-opencl")]
            log::info!("llama_vulkan: ✓ AMD GPU inference ready (Vulkan llama-server on port {port}).");
            #[cfg(not(feature = "pom-opencl"))]
            log::info!("llama server: ✓ GPU inference ready (llama.cpp llama-server on port {port}).");
            return true;
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    log::warn!("llama_vulkan: llama-server did not become healthy in time — falling back to CPU inference.");
    stop();
    false
}

/// Generate up to `max_tokens` from `prompt` via the running Vulkan llama-server. Returns the
/// completion text, or None on any error (caller falls back to candle-CPU). Temperature/top_p match
/// the candle path (it's user-facing text — exact tokens are not consensus-relevant).
pub fn generate(prompt: &str, max_tokens: usize) -> Option<String> {
    if !available() {
        return None;
    }
    let port = PORT.load(Ordering::Relaxed);
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
    let resp = match ureq::post(&url)
        .timeout(Duration::from_secs(120))
        .set("Content-Type", "application/json")
        .send_string(&payload)
    {
        Ok(r) => r,
        Err(e) => {
            log::warn!("llama_vulkan: /completion request failed ({e}) — this challenge falls back to CPU.");
            return None;
        }
    };
    let raw = resp.into_string().ok()?;
    let json: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let text = json.get("content")?.as_str()?.to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Kill the llama-server (best-effort). Called on failed startup; the monitor thread reaps the
/// child and clears SERVER_PID. (On Linux, normal miner death needs no call at all — PDEATHSIG
/// kills the child from the kernel side.)
pub fn stop() {
    AVAILABLE.store(false, Ordering::Relaxed);
    let pid = SERVER_PID.swap(0, Ordering::Relaxed);
    #[cfg(unix)]
    if pid != 0 {
        let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), nix::sys::signal::Signal::SIGKILL);
    }
    #[cfg(not(unix))]
    let _ = pid;
}
