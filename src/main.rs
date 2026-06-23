#![cfg_attr(all(test, feature = "bench"), feature(test))]

#[cfg(not(feature = "static-cuda"))]
use std::env::consts::DLL_EXTENSION;
use std::env::current_exe;
use std::error::Error as StdError;
#[cfg(not(feature = "static-cuda"))]
use std::ffi::OsStr;

use clap::{App, FromArgMatches, IntoApp};
use keryx_miner::PluginManager;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::fs;
use std::sync::atomic::AtomicU16;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::cli::Opt;
use crate::client::grpc::KeryxdHandler;
use crate::client::stratum::StratumHandler;
use crate::client::Client;
use crate::miner::MinerManager;
use crate::target::Uint256;

mod api;
mod cli;
mod client;
mod escrow;
mod ipfs;
mod keryxd_messages;
mod miner;
mod pow;
mod target;
mod watch;

#[cfg(not(feature = "static-cuda"))]
const WHITELIST: [&str; 4] = ["libkeryxcuda", "libkeryxopencl", "keryxcuda", "keryxopencl"];

pub mod proto {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("protowire");
    // include!("protowire.rs"); // FIXME: https://github.com/intellij-rust/intellij-rust/issues/6579
}

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

type Hash = Uint256;

/// Attempt to install the CUDA runtime libraries candle needs, on a Debian/Ubuntu host (HiveOS).
///
/// OPoI GPU inference needs cuBLAS, cuBLASLt and cuRAND — candle creates handles for all three
/// when it opens the CUDA device. These ship with the CUDA toolkit but not with the bare NVIDIA
/// driver that mining rigs usually have. Rather than forcing miners to run apt by hand, we add
/// the NVIDIA CUDA repo and install `libcublas-12-2` (cuBLAS + cuBLASLt) and `libcurand-12-2`
/// ourselves, then register their directory with ldconfig. Runs as root on HiveOS, so no sudo.
///
/// Version 12-2 (not 12-6) is deliberate: the binary's candle kernels are compiled with the
/// CUDA 12.2 toolkit so they JIT on driver >= 535 (typical HiveOS), and the cuBLAS runtime must
/// match that minimum. Installing 12-6 here would pull a runtime needing driver >= 560.
/// Returns true on success.
#[cfg(target_os = "linux")]
fn install_cuda_libs() -> bool {
    use std::process::Command;
    // Only meaningful where apt-get exists (Debian/Ubuntu, incl. HiveOS).
    let has_apt = Command::new("sh")
        .args(["-c", "command -v apt-get"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !has_apt {
        error!("CUDA lib auto-install needs apt-get (Debian/Ubuntu) — not found on this system.");
        return false;
    }
    // The CUDA libs install into /usr/local/cuda-*/targets/x86_64-linux/lib, which is NOT in
    // the default loader search path. Installing alone is not enough: we must register that
    // directory with ldconfig so dlopen("libcublas.so.12" / "libcurand.so.10") resolves it.
    let script = r#"set -e
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb
dpkg -i cuda-keyring.deb
apt-get update -qq
apt-get install -y -qq libcublas-12-2 libcurand-12-2
CUBLAS_PATH=$(find /usr/local /usr/lib -name 'libcublas.so.12' 2>/dev/null | head -1)
if [ -z "$CUBLAS_PATH" ]; then echo "libcublas.so.12 not found after install"; exit 1; fi
echo "$(dirname "$CUBLAS_PATH")" > /etc/ld.so.conf.d/keryx-cuda.conf
ldconfig
ldconfig -p | grep -q libcublas.so.12 || { echo "libcublas still not in loader cache"; exit 1; }
ldconfig -p | grep -q libcurand.so   || { echo "libcurand still not in loader cache"; exit 1; }
rm -f cuda-keyring.deb"#;
    Command::new("bash")
        .args(["-c", script])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn adjust_console() -> Result<(), Error> {
    let console = win32console::console::WinConsole::input();
    let mut mode = console.get_mode()?;
    mode = (mode & !win32console::console::ConsoleMode::ENABLE_QUICK_EDIT_MODE)
        | win32console::console::ConsoleMode::ENABLE_EXTENDED_FLAGS;
    console.set_mode(mode)?;
    Ok(())
}

#[cfg(not(feature = "static-cuda"))]
fn filter_plugins(dirname: &str) -> Vec<String> {
    match fs::read_dir(dirname) {
        Ok(readdir) => readdir
            .map(|entry| entry.unwrap().path())
            .filter(|fname| {
                fname.is_file()
                    && fname.extension().is_some()
                    && fname.extension().and_then(OsStr::to_str).unwrap_or_default().starts_with(DLL_EXTENSION)
            })
            .filter(|fname| WHITELIST.iter().any(|lib| *lib == fname.file_stem().and_then(OsStr::to_str).unwrap()))
            .map(|path| path.to_str().unwrap().to_string())
            .collect::<Vec<String>>(),
        _ => Vec::<String>::new(),
    }
}

/// Query GPU stats via nvidia-smi and warn on power/VRAM issues for the selected model tier.
///
/// VRAM requirements (GGUF weights only, not counting CUDA workspace):
///   TinyLlama-1.1B  →  ~1.5 GB
///   DeepSeek-R1-8B  →  ~5 GB
///   DeepSeek-R1-32B → ~19 GB   (requires ≥24 GB card)
///   LLaMA-3.3-70B   → ~28 GB   (requires ≥40 GB card — does NOT fit on RTX 3090)
///
/// Power thresholds empirically derived: Xid 32 observed at ≤300W on RTX 3090 with 32B GGUF.
fn check_gpu_power_limit(needs_high: bool, needs_very_high: bool) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.limit,power.max_limit,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let (current_w, vram_mb) = match output {
        Ok(o) if o.status.success() => {
            // nvidia-smi prints one CSV line per GPU. Take only the first
            // (GPU 0, the miner's primary device) — splitting the whole
            // multi-line blob on ',' merged GPU 0's memory.total with GPU 1's
            // power.limit (e.g. "32607\n250.00"), so the VRAM field failed to
            // parse and always showed 0 on multi-GPU rigs.
            let s = String::from_utf8_lossy(&o.stdout);
            let mut parts = s.lines().next().unwrap_or("").split(',');
            let cur: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
            let _max: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
            let vram: u64 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0);
            (cur as u32, vram)
        }
        _ => return,
    };

    // VRAM check for 70B: 28 GB weights + ~2.5 GB KV cache → requires ≥32 GB card (RTX 5090+).
    // On 24 GB cards (RTX 3090 / 4090), loading will OOM and crash the miner.
    if needs_very_high && vram_mb < 30_000 {
        log::error!(
            "✗  LLaMA-3.3-70B requires ≥32 GB VRAM (RTX 5090 or equivalent) — \
             this GPU has only {} MB ({} GB).",
            vram_mb,
            vram_mb / 1024
        );
        log::error!(
            "   Use --high (DeepSeek-R1-32B, fits in 24 GB) or --light (TinyLlama only)."
        );
        // Non-fatal: let candle fail with its own OOM so the miner logs the actual error.
    }

    let model_label = if needs_very_high {
        "LLaMA-3.3-70B (--very-high)"
    } else if needs_high {
        "DeepSeek-R1-32B (--high)"
    } else {
        "DeepSeek-R1-8B (default)"
    };
    log::info!("GPU: {}W PL, {} MB VRAM — ready for {}", current_w, vram_mb, model_label);
}

async fn get_client(
    keryxd_address: String,
    mining_address: String,
    pool_password: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,
    escrow_privkey: Option<String>,
    escrow_state_file: String,
    ipfs_url: String,
) -> Result<Box<dyn Client + 'static>, Error> {
    if keryxd_address.starts_with("stratum+tcp://") {
        let (_schema, address) = keryxd_address.split_once("://").unwrap();
        Ok(StratumHandler::connect(
            address.to_string().clone(),
            mining_address.clone(),
            pool_password.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            ipfs_url.clone(),
        )
        .await?)
    } else if keryxd_address.starts_with("grpc://") {
        Ok(KeryxdHandler::connect(
            keryxd_address.clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            escrow_privkey,
            escrow_state_file,
            ipfs_url,
        )
        .await?)
    } else {
        Err("Did not recognize pool/grpc address schema".into())
    }
}

async fn client_main(
    opt: &Opt,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
    escrow_privkey: Option<String>,
) -> Result<(), Error> {
    let ipfs_url = opt.ipfs_url.clone();
    tokio::task::spawn_blocking(move || crate::ipfs::ensure_daemon(&ipfs_url)).await.ok();

    let mut client = get_client(
        opt.keryxd_address.clone(),
        opt.mining_address.clone().unwrap_or_default(),
        opt.pool_password.clone(),
        opt.mine_when_not_synced,
        block_template_ctr.clone(),
        escrow_privkey,
        opt.escrow_state_file.clone(),
        opt.ipfs_url.clone(),
    )
    .await?;

    client.register().await?;
    let mut miner_manager = MinerManager::new(client.get_block_channel(), opt.num_threads, plugin_manager);
    if let Some(bind) = opt.api_bind.clone() {
        tokio::spawn(api::serve(bind, miner_manager.stats(), env!("CARGO_PKG_VERSION").to_string()));
    }
    client.listen(&mut miner_manager).await?;
    drop(miner_manager);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    #[cfg(target_os = "windows")]
    adjust_console().unwrap_or_else(|e| {
        eprintln!("WARNING: Failed to protect console ({}). Any selection in console will freeze the miner.", e)
    });
    let mut path = current_exe().unwrap_or_default();
    path.pop(); // Getting the parent directory

    // --disable-gpu must be honoured before plugins are dlopened/registered, but
    // the CLI is parsed further down (plugins augment the arg set first). Detect
    // it from the raw args here so we can skip discovering/loading GPU workers.
    let disable_gpu = std::env::args().any(|a| a == "--disable-gpu");
    if disable_gpu {
        eprintln!("--disable-gpu: skipping GPU workers, mining on CPU only.");
    }

    // Dynamic plugin path (default): scan the binary's dir for libkeryx*.so,
    // unless --disable-gpu was passed.
    #[cfg(not(feature = "static-cuda"))]
    let plugins = if disable_gpu { Vec::new() } else { filter_plugins(path.to_str().unwrap_or(".")) };
    #[cfg(not(feature = "static-cuda"))]
    let (app, mut plugin_manager): (App, PluginManager) =
        keryx_miner::load_plugins(Opt::into_app(), &plugins)?;

    // Static-cuda single-binary build: the CUDA worker is compiled in, so
    // register it directly instead of dlopening a .so. (OpenCL is omitted.)
    // --disable-gpu skips registering it, leaving a CPU-only miner.
    #[cfg(feature = "static-cuda")]
    let plugins: Vec<String> =
        if disable_gpu { Vec::new() } else { vec!["builtin:cuda (static)".to_string()] };
    #[cfg(feature = "static-cuda")]
    let (app, mut plugin_manager): (App, PluginManager) = {
        let mut manager = PluginManager::new();
        let app = if disable_gpu {
            Opt::into_app()
        } else {
            manager.register_builtin(
                Opt::into_app(),
                Box::new(keryxcuda::CudaPlugin::new()?),
                |a| <keryxcuda::CudaOpt as clap::Args>::augment_args(a),
            )
        };
        (app, manager)
    };

    let matches = app.get_matches();

    let worker_count = plugin_manager.process_options(&matches)?;
    let mut opt: Opt = Opt::from_arg_matches(&matches)?;
    // With --disable-gpu there are no GPU workers, so default the CPU thread
    // count to the physical core count when the user didn't set --threads
    // (otherwise the miner would start with 0 workers and bail out). Checked
    // before opt.process(), which turns a missing --threads into Some(0).
    if opt.disable_gpu && opt.num_threads.is_none() {
        opt.num_threads = Some(crate::miner::get_num_cpus(None));
    }
    opt.process()?;
    // try_init: in the static-cuda build CudaPlugin::new already set a logger;
    // init() would panic on the second call.
    let _ = env_logger::builder().filter_level(opt.log_level()).parse_default_env().try_init();
    info!("=================================================================================");
    info!("                 Keryx-Miner GPU {}", env!("CARGO_PKG_VERSION"));
    info!(" Mining for: {}", opt.mining_address.as_deref().unwrap_or("(recovery mode)"));
    info!("=================================================================================");

    // Recovery mode: rebuild escrow_state.json from the Keryx public API, then exit.
    // Must run before escrow key loading to avoid creating a new random key on disk.
    // Uses escrow.key to derive the pubkey — only claimable UTXOs are returned.
    if opt.recover_escrow {
        let escrow_privkey = match escrow::load_key(&opt.escrow_key_file) {
            Ok(k) => k,
            Err(e) => {
                error!("{}", e);
                return Err(e.into());
            }
        };
        let pubkey_hex = match escrow::pubkey_hex_from_privkey(&escrow_privkey) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to derive pubkey from escrow key: {}", e);
                return Err(e.into());
            }
        };
        let url = format!("{}/api/v1/escrow/{}", opt.recover_escrow_api.trim_end_matches('/'), pubkey_hex);
        info!("Querying escrow UTXOs from {}", url);

        #[derive(serde::Deserialize)]
        struct ApiEscrowEntry {
            coinbase_txid: String,
            block_hash: String,
            confirm_daa: i64,
            amount_sompi: i64,
            output_index: i64,
        }

        let url_clone = url.clone();
        let api_entries: Vec<ApiEscrowEntry> = tokio::task::spawn_blocking(move || {
            let response = ureq::get(&url_clone)
                .call()
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            serde_json::from_reader::<_, Vec<ApiEscrowEntry>>(response.into_reader())
                .map_err(|e| format!("JSON parse error: {}", e))
        })
        .await
        .map_err(|e| format!("spawn_blocking failed: {}", e))??;

        let entries: Vec<escrow::EscrowEntry> = api_entries
            .into_iter()
            .map(|a| escrow::EscrowEntry {
                coinbase_txid: a.coinbase_txid,
                block_hash: a.block_hash,
                confirm_daa: a.confirm_daa as u64,
                amount_sompi: a.amount_sompi as u64,
                output_index: a.output_index as u32,
                claimed: false,
                slashed: false,
                orphan_slashed: false,
                orphan_retries: 0,
                orphan_retry_after_daa: None,
                is_inference: false,
            })
            .collect();

        let total_sompi: u64 = entries.iter().map(|e| e.amount_sompi).sum();
        let count = entries.len();
        let state = escrow::EscrowState { entries };
        let json = serde_json::to_string_pretty(&state)?;
        fs::write(&opt.escrow_state_file, &json)?;

        info!(
            "Recovered {} escrow entries — claimable: {:.4} KRX",
            count,
            total_sompi as f64 / 1e8
        );
        info!("State saved to '{}'.", opt.escrow_state_file);
        return Ok(());
    }

    // Resolve OPoI escrow private key (once, before the reconnect loop).
    let escrow_privkey: Option<String> = match escrow::load_or_generate_key(&opt.escrow_key_file) {
        Ok(k) => {
            info!("OPoI: escrow key loaded from '{}'.", opt.escrow_key_file);
            Some(k)
        }
        Err(e) => {
            error!("Failed to load/generate OPoI escrow key: {}", e);
            return Err(e.into());
        }
    };

    // Phase-3 OPoI: load inference models before mining starts.
    //   (no flag)    → TinyLlama + DeepSeek-R1-8B  [default]
    //   --light      → TinyLlama only
    //   --high       → TinyLlama + DeepSeek-R1-8B + DeepSeek-R1-32B
    //   --very-high  → all 4 models

    // Warn if GPU power limit is below safe threshold for the selected model tier.
    // Low PL causes CUDA FIFO instability (Xid 32) under large GEMM workloads.
    check_gpu_power_limit(opt.high || opt.very_high, opt.very_high);

    let specs: &'static [&'static keryx_miner::models::ModelSpec] = if opt.very_high {
        info!("--very-high mode: loading all 4 models (TinyLlama + DeepSeek-8B + DeepSeek-32B + LLaMA-70B).");
        &[
            &keryx_miner::models::TINYLLAMA,
            &keryx_miner::models::DEEPSEEK_R1_8B,
            &keryx_miner::models::DEEPSEEK_R1_32B,
            &keryx_miner::models::LLAMA_3_3_70B,
        ]
    } else if opt.high {
        info!("--high mode: loading TinyLlama + DeepSeek-R1-8B + DeepSeek-R1-32B.");
        &[
            &keryx_miner::models::TINYLLAMA,
            &keryx_miner::models::DEEPSEEK_R1_8B,
            &keryx_miner::models::DEEPSEEK_R1_32B,
        ]
    } else if opt.light {
        info!("--light mode: loading TinyLlama only.");
        &[&keryx_miner::models::TINYLLAMA]
    } else {
        &[&keryx_miner::models::TINYLLAMA, &keryx_miner::models::DEEPSEEK_R1_8B]
    };
    keryx_miner::slm::init_supported(specs);
    keryx_miner::slm::set_cpu_inference(opt.cpu_inference);
    if opt.cpu_inference {
        info!("--cpu-inference mode: OPoI inference runs on CPU, GPU stays dedicated to hashing.");
    }
    info!("OPoI Phase-3 active — {} model(s) supported.", specs.len());
    // Prefetch model files in the BACKGROUND, not blocking startup. The OPoI hard gate
    // (client/stratum.rs, client/grpc.rs) already keeps PoW suspended until models are
    // ready and un-suspends itself once the (one-time, ~GB) download completes — so we
    // don't need to block here. Blocking was actively harmful: the worker/plugin setup
    // below runs only AFTER prefetch returned, so a rig whose OpenCL stack can't create
    // a GPU worker would download the whole model first and ONLY THEN exit with
    // "No workers specified" (reported as a HiveOS "black screen"). Backgrounding lets
    // that worker error surface immediately, and lets the miner connect to the pool
    // while the download runs.
    info!("Prefetching model files in the background — PoW stays OPoI-gated until they're ready…");
    tokio::spawn(async move {
        match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(specs)).await {
            Ok(Ok(())) => info!("Model files ready — OPoI inference available; mining un-gates."),
            // per-share OPoI `tag_fixed` is baked in and works without LLM weights; only
            // AiRequest tasks need them, so a download failure just leaves escrow unclaimed.
            Ok(Err(e)) => warn!("Model prefetch failed — AiRequest tasks will be skipped: {}", e),
            Err(e) => warn!("Model prefetch task panicked — AiRequest tasks will be skipped: {}", e),
        }
    });

    // PoM (AMD): register the tier-0 Gemma-3-4B possession model and background-fetch its GGUF
    // (same CID the OPoI path uses). The mining loop builds the WeightIndex + GPU residency
    // lazily (pom_opencl::ensure_installed) on the first PoM-active job (DAA >= activation);
    // pre-fork it just sits registered. tier 0 fits the 16 GB AMD cards (2.48 GiB).
    #[cfg(any(feature = "pom-opencl", feature = "pom-cuda"))]
    {
        // Driver seam: AMD = OpenCL, NVIDIA = candle-CUDA (same set_mining_tier interface).
        #[cfg(feature = "pom-opencl")]
        use keryx_miner::pom_opencl as pom_driver;
        #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
        use keryx_miner::pom_gpu as pom_driver;
        let gguf = keryx_miner::slm::gguf_path_for(&keryx_miner::models::GEMMA_3_4B);
        pom_driver::set_mining_tier(gguf.to_string_lossy().into_owned(), 0);
        info!("PoM: registered tier 0 (Gemma-3-4B) at {}", gguf.display());
        if keryx_miner::pom::is_activation_overridden() {
            warn!(
                "PoM(AMD): ACTIVATION DAA OVERRIDDEN to {} via KERYX_POM_ACTIVATION_DAA — staging/testing ONLY, NOT for production!",
                keryx_miner::pom::activation_daa()
            );
        }
        let pom_specs: &'static [&'static keryx_miner::models::ModelSpec] =
            &[&keryx_miner::models::GEMMA_3_4B];
        tokio::spawn(async move {
            match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(pom_specs)).await {
                Ok(Ok(())) => info!("PoM(AMD): tier-0 Gemma GGUF ready."),
                Ok(Err(e)) => warn!("PoM(AMD): tier model download failed: {}", e),
                Err(e) => warn!("PoM(AMD): tier model download task panicked: {}", e),
            }
        });
    }

    // Verify GPU inference works before mining. OPoI challenges are mandatory, so a miner
    // that cannot run inference must fail fast with a clear message rather than spam panics.
    if opt.cpu_inference {
        info!("--cpu-inference: skipping GPU inference probe (inference runs on CPU).");
    } else {
    info!("Probing GPU inference (cuBLAS) before mining…");
    match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
        Ok(keryx_miner::slm::GpuProbe::Ok) => info!("GPU inference verified — cuBLAS loaded successfully."),
        Ok(keryx_miner::slm::GpuProbe::NoCuda) => {
            warn!("No CUDA device detected — inference will run on CPU (small models only, slow).");
        }
        Ok(keryx_miner::slm::GpuProbe::CublasMissing) => {
            warn!("CUDA GPU detected but a CUDA runtime lib is missing — installing them automatically (one-time)…");
            #[cfg(target_os = "linux")]
            {
                let installed = tokio::task::spawn_blocking(install_cuda_libs).await.unwrap_or(false);
                if !installed {
                    error!("Automatic CUDA lib install failed — install them manually then restart:");
                    error!("  apt-get install -y libcublas-12-2 libcurand-12-2");
                    return Err("CUDA runtime libs missing — cannot start OPoI mining".into());
                }
                // Re-probe in-process. The dynamic loader may still hold a stale cache, so if
                // the freshly-installed libs aren't picked up here, exit cleanly and let the
                // supervisor (HiveOS/PM2) relaunch us with a fresh loader cache.
                match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
                    Ok(keryx_miner::slm::GpuProbe::Ok) => {
                        info!("CUDA libs installed — GPU inference verified, starting mining.");
                    }
                    _ => {
                        info!("CUDA libs installed successfully — restarting miner to activate them.");
                        std::process::exit(0);
                    }
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                error!("CUDA GPU detected but a CUDA runtime lib failed to load — install the CUDA 12.6 toolkit and restart.");
                return Err("CUDA runtime libs missing — cannot start OPoI mining".into());
            }
        }
        Err(e) => {
            error!("GPU probe task panicked: {}", e);
            return Err(e.into());
        }
    }
    }
    info!("Found plugins: {:?}", plugins);
    info!("Plugins found {} workers", worker_count);
    if worker_count == 0 && opt.num_threads.unwrap_or(0) == 0 {
        error!("No workers specified");
        return Err("No workers specified".into());
    }

    let block_template_ctr = Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16));
    // Reconnect backoff. Pools blip (TCP accepts then drops, EOF, restarts).
    // Without a delay the loop busy-spins ~10x/sec, re-initialising the GPU
    // worker each time and hammering a dead pool. Start at 1s, double on each
    // rapid failure up to 30s, and reset after a session that stayed up long
    // enough to be considered healthy.
    const RECONNECT_MIN: Duration = Duration::from_secs(1);
    const RECONNECT_MAX: Duration = Duration::from_secs(30);
    const HEALTHY_SESSION: Duration = Duration::from_secs(60);
    let mut backoff = RECONNECT_MIN;
    loop {
        let started = Instant::now();
        match client_main(&opt, block_template_ctr.clone(), &plugin_manager, escrow_privkey.clone()).await {
            Ok(_) => info!("Client closed gracefully"),
            Err(e) => error!("Client closed with error {:?}", e),
        }
        // A session that lasted a while was healthy — treat the next drop as a
        // fresh blip rather than escalating the backoff.
        if started.elapsed() >= HEALTHY_SESSION {
            backoff = RECONNECT_MIN;
        }
        info!("Client closed, reconnecting in {}s", backoff.as_secs());
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(RECONNECT_MAX);
    }
}
