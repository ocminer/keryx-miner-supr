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
#[cfg(all(target_os = "macos", feature = "pom-metal"))]
mod metal_worker;
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
        "Llama-3.3-70B (very-high)"
    } else if needs_high {
        "Qwen3-32B (high)"
    } else {
        "Gemma-3-4B / Dolphin-8B (light/default)"
    };
    log::info!("GPU: {}W PL, {} MB VRAM — ready for {}", current_w, vram_mb, model_label);
}

/// GPU 0 total VRAM (MB): nvidia-smi (NVIDIA), else OpenCL CL_DEVICE_GLOBAL_MEM_SIZE (AMD, when the
/// pom-opencl driver is linked). None on CPU-only machines.
fn query_vram_mb() -> Option<u64> {
    // One process per GPU (CUDA_VISIBLE_DEVICES=<n>): query THIS process's visible GPU, not GPU 0.
    // nvidia-smi without `-i` always lists GPU 0 first regardless of CUDA_VISIBLE_DEVICES, so on a
    // mixed rig (e.g. a 5090 at slot 0 + a 3070 at slot 1) the 3070's process would wrongly read the
    // 5090's 32 GB and auto-pick a tier that OOMs. Pin nvidia-smi to the first visible device id.
    // (Miners launch with CUDA_DEVICE_ORDER=PCI_BUS_ID, so CUDA's index == nvidia-smi's index.)
    let mut cmd = std::process::Command::new("nvidia-smi");
    if let Ok(vis) = std::env::var("CUDA_VISIBLE_DEVICES") {
        // CUDA_VISIBLE_DEVICES may be a list ("2,3"); the process's GPU 0 is the FIRST entry.
        if let Some(first) = vis.split(',').next().map(str::trim).filter(|s| !s.is_empty()) {
            // Only a plain numeric index is safe to forward to `nvidia-smi -i` (UUIDs also work,
            // but ignore anything unexpected and fall back to the default GPU-0 query).
            if first.chars().all(|c| c.is_ascii_digit()) {
                cmd.args(["-i", first]);
            }
        }
    }
    if let Ok(output) = cmd
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            if let Some(mb) = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .and_then(|l| l.trim().parse::<u64>().ok())
            {
                return Some(mb);
            }
        }
    }
    // AMD: query the OpenCL GPU's global memory so the capability gate works without nvidia-smi.
    #[cfg(feature = "pom-opencl")]
    {
        return keryx_miner::pom_opencl::gpu0_global_mem_mb();
    }
    #[allow(unreachable_code)]
    None
}

/// GPU VRAM per device `(device_id, MiB)` via the CUDA/Metal `pom_gpu` driver when it is linked in,
/// else an empty list. `pom_gpu` exists ONLY under pom-cuda (non-Metal) OR macos+pom-metal — NOT in
/// the default (no-feature) build or the AMD build — so gating the query on plain `not(pom-opencl)`
/// referenced a configured-out `pom_gpu` and broke the default build. Callers already handle an empty
/// result (fall back to nvidia-smi / `select_tier_nvidia`), so returning empty here is safe.
#[cfg(not(feature = "pom-opencl"))]
fn all_gpus_vram() -> Vec<(u32, u64)> {
    #[cfg(any(
        all(feature = "pom-cuda", not(all(target_os = "macos", feature = "pom-metal"))),
        all(target_os = "macos", feature = "pom-metal")
    ))]
    { keryx_miner::pom_gpu::query_all_gpus_vram() }
    #[cfg(not(any(
        all(feature = "pom-cuda", not(all(target_os = "macos", feature = "pom-metal"))),
        all(target_os = "macos", feature = "pom-metal")
    )))]
    { Vec::new() }
}

/// OPoI capability gate (layer A): drop models GPU 0 cannot serve, so `ai:cap` never promises a
/// model the miner would fail to load. VRAM comes from nvidia-smi (NVIDIA) or OpenCL (AMD); the
/// gate is skipped only on CPU-only hosts where neither is available.
fn filter_specs_by_vram(
    specs: &'static [&'static keryx_miner::models::ModelSpec],
) -> &'static [&'static keryx_miner::models::ModelSpec] {
    // Gate against the BIGGEST card in the rig, not GPU 0: in a mixed rig the inference/primary
    // model runs on the largest-VRAM card, so a small GPU 0 must not filter it out.
    #[cfg(not(feature = "pom-opencl"))]
    let max_mb = all_gpus_vram().into_iter().map(|(_, m)| m).max();
    #[cfg(feature = "pom-opencl")]
    let max_mb: Option<u64> = None;
    let Some(gpu0_mb) = max_mb.or_else(query_vram_mb) else {
        log::warn!("Cannot query GPU VRAM (no nvidia-smi / OpenCL GPU) — skipping the model capability gate.");
        return specs;
    };
    let kept: Vec<&'static keryx_miner::models::ModelSpec> = specs
        .iter()
        .copied()
        .filter(|spec| {
            if spec.min_vram_mb <= gpu0_mb {
                true
            } else {
                log::warn!(
                    "✗  '{}' needs ≥{} MB VRAM but only {} MB on GPU 0 — not announced/downloaded.",
                    spec.name, spec.min_vram_mb, gpu0_mb
                );
                false
            }
        })
        .collect();
    if kept.len() == specs.len() {
        specs
    } else {
        Box::leak(kept.into_boxed_slice())
    }
}

/// Conservative VRAM headroom (MiB) reserved on top of a model's `min_vram_mb` budget when
/// auto-selecting a tier. `min_vram_mb` already covers weights + KV-cache + CUDA workspace; this
/// margin guards against fragmentation / driver overhead / the resident PoM possession walk so the
/// picked tier loads cleanly. Empirically an 8 GB 3070 OOMs even Gemma on the GPU, so we stay
/// conservative: 2 GB margin keeps an 8 GB card on Light (the OOM-safe choice).
#[cfg(not(feature = "pom-opencl"))]
const AUTO_TIER_HEADROOM_MB: u64 = 2_048;

/// Resolve the model tier for the NVIDIA (pom-cuda) path. `--tier <value>` takes precedence over
/// the legacy `--light/--high/--very-high` bool flags (clap enforces they are mutually exclusive).
///
/// `--tier auto` queries THIS process's visible GPU VRAM (one process per GPU) and picks the
/// LARGEST tier that fits with `AUTO_TIER_HEADROOM_MB` margin. If the largest-fitting tier's model
/// is not yet downloaded, it falls back to the largest tier that BOTH fits AND is already on disk,
/// so the miner starts immediately instead of blocking on a multi-GB IPFS download. (The picked
/// tier's model is still prefetched in the background by the normal model-fetch path.)
#[cfg(not(feature = "pom-opencl"))]
fn select_tier_nvidia(opt: &cli::Opt) -> keryx_miner::models::Tier {
    use keryx_miner::models::Tier;

    // Explicit string tier (new --tier flag) wins.
    if let Some(raw) = opt.tier.as_deref() {
        let t = raw.trim().to_ascii_lowercase();
        match t.as_str() {
            "auto" => return select_tier_auto(),
            "very-light" | "verylight" | "very_light" => {
                info!("--tier very-light: smallest tier — mines Qwen3-1.7B under PoM (post-H2).");
                return Tier::VeryLight;
            }
            "light" => {
                info!("--tier light: baseline tier — mines Gemma-3-4B under PoM.");
                return Tier::Light;
            }
            "default" => {
                info!("--tier default: mines Dolphin-8B under PoM.");
                return Tier::Default;
            }
            "high" => {
                info!("--tier high: high tier — mines Qwen3-32B under PoM.");
                return Tier::High;
            }
            "very-high" | "veryhigh" | "very_high" => {
                info!("--tier very-high: top tier — mines Llama-3.3-70B under PoM.");
                return Tier::VeryHigh;
            }
            other => {
                warn!(
                    "--tier '{}' not recognised (expected auto|light|default|high|very-high) — defaulting to auto.",
                    other
                );
                return select_tier_auto();
            }
        }
    }

    // Legacy bool flags pin an explicit tier (override auto).
    if opt.very_high {
        info!("--very-high mode: top tier — mines Llama-3.3-70B under PoM.");
        Tier::VeryHigh
    } else if opt.high {
        info!("--high mode: high tier — mines Qwen3-32B under PoM.");
        Tier::High
    } else if opt.light {
        info!("--light mode: baseline tier — mines Gemma-3-4B under PoM.");
        Tier::Light
    } else if opt.very_light {
        info!("--very-light mode: smallest tier — mines Qwen3-1.7B under PoM (post-H2).");
        Tier::VeryLight
    } else {
        // No tier flag given at all → AUTO is the default: pick the largest tier that fits this
        // GPU's VRAM (per-process) and is already on disk. Heavier model = higher PoM tier reward
        // (TIER_REWARD_BPS 82%→100% of the miner cut, active post-fork). The auto resolver is fully
        // OOM-safe (conservative VRAM margin keeps 8 GB cards on Light) and never blocks on a slow
        // IPFS fetch (it falls back to the largest fitting tier already downloaded). Pin a tier with
        // --light / --high / --very-high / --tier <name> to override.
        info!("no tier flag given — defaulting to AUTO (largest tier that fits this GPU's VRAM). Use --light/--high/--very-high/--tier to pin a tier.");
        select_tier_auto()
    }
}

/// `--tier auto` resolution: pick the largest tier that fits this GPU's VRAM, then fall back to the
/// largest tier whose model is also already downloaded (or trigger nothing — the normal background
/// prefetch downloads the picked tier's model). Logs every decision.
#[cfg(not(feature = "pom-opencl"))]
fn select_tier_auto() -> keryx_miner::models::Tier {
    use keryx_miner::models::{self, Tier};

    let Some(vram_mb) = query_vram_mb() else {
        warn!("--tier auto: cannot query GPU VRAM (no nvidia-smi) — falling back to --light (Gemma-3-4B).");
        return Tier::Light;
    };

    let (picked, need) = models::auto_select_tier(vram_mb, AUTO_TIER_HEADROOM_MB);
    info!(
        "tier auto: {} MiB VRAM -> {} ({:?}, tier {}) — budget {} MiB (weights+KV+workspace + {} MiB margin).",
        vram_mb,
        picked.pom_model_name(),
        picked,
        models::pom_tier_index(&picked.pom_spec().model_id, models::VERY_LIGHT_ACTIVATION_DAA).unwrap_or(0),
        need,
        AUTO_TIER_HEADROOM_MB,
    );

    // Availability: if the picked tier's model isn't on disk yet, fall back to the largest tier that
    // BOTH fits AND is already downloaded so mining starts now. The picked-but-missing model is still
    // background-prefetched by the normal path; on the next restart auto will select it.
    if keryx_miner::slm::spec_files_ready(picked.pom_spec()) {
        info!("tier auto: {} model is present on disk — using it.", picked.pom_model_name());
        return picked;
    }

    warn!(
        "tier auto: {} model not downloaded yet — searching for the largest fitting tier already on disk.",
        picked.pom_model_name()
    );
    for tier in Tier::DESCENDING {
        // Only consider tiers that ALSO fit this card (never downgrade VRAM-fit just to use a
        // present-but-too-big model — though by construction the present ones are smaller/equal).
        let fits = vram_mb >= tier.pom_spec().min_vram_mb.saturating_add(AUTO_TIER_HEADROOM_MB)
            || tier == Tier::Light;
        if fits && keryx_miner::slm::spec_files_ready(tier.pom_spec()) {
            warn!(
                "tier auto: falling back to {} ({:?}, tier {}) — largest fitting tier already on disk. \
                 ({} will download in the background; restart to use it.)",
                tier.pom_model_name(),
                tier,
                models::pom_tier_index(&tier.pom_spec().model_id, models::VERY_LIGHT_ACTIVATION_DAA).unwrap_or(0),
                picked.pom_model_name(),
            );
            return tier;
        }
    }

    // Nothing downloaded at all — keep the VRAM-picked tier; the background prefetch will fetch it
    // and the OPoI hard-gate keeps PoW suspended until the files are ready.
    warn!(
        "tier auto: no tier model present on disk — keeping {} ({:?}); it will download before mining starts.",
        picked.pom_model_name(),
        picked,
    );
    picked
}

/// Parse a tier name (for `--force-model` / `--tier`). None on an unrecognised token.
fn parse_tier_name(s: &str) -> Option<keryx_miner::models::Tier> {
    use keryx_miner::models::Tier;
    match s.trim().to_ascii_lowercase().as_str() {
        "very-light" | "verylight" | "very_light" => Some(Tier::VeryLight),
        "light" => Some(Tier::Light),
        "default" => Some(Tier::Default),
        "high" => Some(Tier::High),
        "very-high" | "veryhigh" | "very_high" => Some(Tier::VeryHigh),
        _ => None,
    }
}

/// The tier the user PINNED for every card (legacy flag or `--tier <name>`), or None = AUTO.
fn pinned_tier(opt: &cli::Opt) -> Option<keryx_miner::models::Tier> {
    use keryx_miner::models::Tier;
    if let Some(raw) = opt.tier.as_deref() {
        let t = raw.trim().to_ascii_lowercase();
        return if t == "auto" { None } else { parse_tier_name(&t) };
    }
    if opt.very_high { Some(Tier::VeryHigh) }
    else if opt.high { Some(Tier::High) }
    else if opt.light { Some(Tier::Light) }
    else if opt.very_light { Some(Tier::VeryLight) }
    else { None }
}

/// Resolve the mining tier for EACH CUDA device (CUDA-driver order): `--force-model <csv>` wins
/// per-card (forced, VRAM bypassed), else a pinned flag (all cards), else per-card AUTO best-fit
/// (heaviest tier that fits that card's VRAM). Fewer force-model entries than cards → the rest AUTO.
/// Falls back to a single (device 0, legacy-resolver) entry when the CUDA driver can't enumerate.
/// The bool per entry = the tier came from `--force-model` (so the VRAM capability gate is skipped).
#[cfg(not(feature = "pom-opencl"))]
fn resolve_device_tiers(opt: &cli::Opt) -> Vec<(u32, keryx_miner::models::Tier, bool)> {
    use keryx_miner::models;
    let forced: Vec<Option<models::Tier>> = opt
        .force_model
        .as_deref()
        .map(|s| s.split(',').map(|x| parse_tier_name(x)).collect())
        .unwrap_or_default();
    let mut vrams = all_gpus_vram(); // (device_id, MiB), CUDA order
    if vrams.is_empty() {
        // No enumeration (nvidia-smi/driver missing): a forced first entry still wins for GPU 0.
        return match forced.first().copied().flatten() {
            Some(t) => vec![(0, t, true)],
            None => vec![(0, select_tier_nvidia(opt), false)],
        };
    }
    vrams.sort_by_key(|(id, _)| *id);
    if let Some(raw) = opt.force_model.as_deref() {
        info!("--force-model: {} — per-card override (VRAM check bypassed; unlisted/extra cards use auto).", raw.trim());
    }
    let pinned = pinned_tier(opt);
    let mut out = Vec::with_capacity(vrams.len());
    for (dev, vram) in &vrams {
        let (tier, is_forced) = match forced.get(*dev as usize).copied().flatten() {
            Some(t) => {
                info!("GPU {}: --force-model → {} (forced; card has {} MiB).", dev, t.pom_model_name(), vram);
                (t, true)
            }
            None => {
                if forced.get(*dev as usize).is_some() {
                    warn!("GPU {}: --force-model entry unrecognised — using auto (names: very-light|light|default|high|very-high).", dev);
                }
                if let Some(t) = pinned {
                    (t, false)
                } else {
                    let (picked, need) = models::auto_select_tier(*vram, AUTO_TIER_HEADROOM_MB);
                    info!("GPU {}: auto → {} (fits {} MiB VRAM, budget {} MiB).", dev, picked.pom_model_name(), vram, need);
                    (picked, false)
                }
            }
        };
        out.push((*dev, tier, is_forced));
    }
    out
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
    pool_address: String,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
    escrow_privkey: Option<String>,
) -> Result<(), Error> {
    // IPFS/kubo setup runs in the BACKGROUND (fire-and-forget) — it's only for optional inference-
    // reward uploads and must never gate mining. Previously this was `.await`ed, so a slow kubo
    // download (e.g. first run on macOS) stalled the miner from connecting to the pool for a minute+.
    let ipfs_url = opt.ipfs_url.clone();
    tokio::task::spawn_blocking(move || crate::ipfs::ensure_daemon(&ipfs_url));

    let mut client = get_client(
        pool_address,
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

/// Tokio async worker count. The miner's async workload is tiny (one gRPC/stratum connection + a
/// few tasks and timers — the heavy work runs on `spawn_blocking` / dedicated std::threads), so we
/// cap workers instead of spawning one per logical CPU: dozens of idle executor threads on a
/// many-core rig are pure scheduler overhead. Override with `KERYX_ASYNC_WORKERS`.
/// (upstream Keryx-Labs/keryx-miner@9bb0d55)
fn tokio_worker_threads() -> usize {
    std::env::var("KERYX_ASYNC_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 8)
}

/// Optional cap for the `spawn_blocking` pool (SLM inference, IPFS upload, model prefetch). Only
/// applied when `KERYX_BLOCKING_THREADS` is set: the blocking pool spawns lazily and idles out, so
/// tokio's default costs nothing at rest and capping it low would bottleneck parallel multi-model
/// prefetch on multi-GPU rigs.
fn tokio_blocking_threads() -> Option<usize> {
    std::env::var("KERYX_BLOCKING_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.clamp(2, 64))
}

fn main() -> Result<(), Error> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.worker_threads(tokio_worker_threads()).enable_all();
    if let Some(n) = tokio_blocking_threads() {
        builder.max_blocking_threads(n);
    }
    builder.build()?.block_on(run())
}

async fn run() -> Result<(), Error> {
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
    // unless --disable-gpu was passed. Excluded on macOS+pom-metal (no .so plugins there — the
    // Metal worker is built in below).
    #[cfg(all(not(feature = "static-cuda"), not(all(target_os = "macos", feature = "pom-metal"))))]
    let plugins = if disable_gpu { Vec::new() } else { filter_plugins(path.to_str().unwrap_or(".")) };
    #[cfg(all(not(feature = "static-cuda"), not(all(target_os = "macos", feature = "pom-metal"))))]
    let (app, mut plugin_manager): (App, PluginManager) =
        keryx_miner::load_plugins(Opt::into_app(), &plugins)?;

    // macOS (Apple Silicon): register the built-in Metal PoM worker. Without this the miner finds
    // 0 workers and exits ("No workers specified"). --disable-gpu leaves a CPU-only miner.
    #[cfg(all(target_os = "macos", feature = "pom-metal"))]
    let plugins: Vec<String> =
        if disable_gpu { Vec::new() } else { vec!["builtin:metal (Apple Silicon)".to_string()] };
    #[cfg(all(target_os = "macos", feature = "pom-metal"))]
    let (app, mut plugin_manager): (App, PluginManager) = {
        let mut manager = PluginManager::new();
        let app = if disable_gpu {
            Opt::into_app()
        } else {
            manager.register_builtin(Opt::into_app(), Box::new(crate::metal_worker::MetalPlugin::new()), |a| a)
        };
        (app, manager)
    };

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

    // Clean shutdown: respond to SIGINT (Ctrl-C) and SIGTERM — the signals HiveOS / mmpOS / SMOS /
    // systemd send to STOP a miner. The GPU worker threads sit in long, uninterruptible CUDA calls
    // and can't be stopped cooperatively, so we exit the process directly and let the OS reclaim the
    // CUDA context; the orphan possession-tree sweep cleans any leftover pom-tree on the next start.
    // (Without this the miner ignored SIGINT/SIGTERM and only died on SIGKILL.)
    tokio::spawn(async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            match signal(SignalKind::terminate()) {
                Ok(mut term) => {
                    tokio::select! {
                        _ = tokio::signal::ctrl_c() => {}
                        _ = term.recv() => {}
                    }
                }
                Err(e) => {
                    log::error!("SIGTERM handler install failed ({e}) — falling back to Ctrl-C only.");
                    let _ = tokio::signal::ctrl_c().await;
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
        log::warn!("Shutdown signal received — stopping keryx-miner.");
        std::process::exit(0);
    });

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
    //   (no flag)    → AUTO: largest PoM tier that fits this GPU's VRAM (per-process)
    //   --light      → Gemma-3-4B (legacy: TinyLlama)
    //   --high       → Qwen3-32B  (legacy: + DeepSeek-R1-32B)
    //   --very-high  → Llama-3.3-70B (legacy: all 4 models)

    // PoM (OPoI v2): one flag = one tier. Each GPU mines AND serves exactly the single model it
    // proves possession of (multi-tier coverage is a network property, not per-GPU).
    //   --light → Gemma-3-4B   --high → Qwen3-32B   --very-high → Llama-3.3-70B   (no flag → AUTO)
    // AMD: ONE tier per process (the OpenCL PoM backend keeps a single resident blob — no
    // per-card model map like CUDA's set_device_model). An explicit user override (--force-model /
    // --tier / --light etc.) is honored PROCESS-WIDE; the default stays Light (Gemma-3-4B) — the
    // OOM-safe pick for low-VRAM cards, and inference (Vulkan llama-server, else candle-CPU) can
    // always serve it. A forced tier SKIPS the VRAM capability gate (same power-user contract as
    // CUDA --force-model): an undersized card will fail/OOM loading the GPU-resident walk blob;
    // llama-server falls back to candle-CPU if inference doesn't fit next to the blob.
    #[cfg(feature = "pom-opencl")]
    let (tier, tier_forced) = {
        use keryx_miner::models::Tier;
        let forced = opt.force_model.as_deref().and_then(|raw| {
            let entries: Vec<&str> = raw.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
            let first = entries.first().and_then(|s| parse_tier_name(s));
            if first.is_none() {
                warn!("AMD/OpenCL: --force-model entry unrecognised — ignoring (names: very-light|light|default|high|very-high).");
            } else if entries.iter().any(|e| parse_tier_name(e) != first) {
                warn!("AMD/OpenCL: --force-model is PROCESS-WIDE on AMD (one resident tier, no per-card map) — applying the FIRST entry to ALL cards.");
            }
            first
        });
        if let Some(t) = forced {
            info!("AMD/OpenCL: tier {} (--force-model, process-wide; VRAM check bypassed — an undersized card will fail to load the PoM blob).", t.pom_model_name());
            (t, true)
        } else if let Some(t) = pinned_tier(&opt) {
            info!("AMD/OpenCL: tier {} (pinned by flag, process-wide — the model must fit the card's VRAM).", t.pom_model_name());
            (t, false)
        } else {
            info!("AMD/OpenCL: tier light (Gemma-3-4B) — the OOM-safe default; pass --tier/--force-model to override.");
            (Tier::Light, false)
        }
    };
    // Resolve the mining tier PER CUDA DEVICE (mixed-rig best-fit / --force-model). `tier` below is
    // the PRIMARY = biggest tier across cards: it drives the inference lineup (OPoI runs on the
    // largest card), the prefetch, and the power-limit check. Per-card overrides for smaller cards
    // are registered after the primary model is set up (search: set_device_model).
    #[cfg(not(feature = "pom-opencl"))]
    let device_tiers = resolve_device_tiers(&opt);
    #[cfg(not(feature = "pom-opencl"))]
    let tier = device_tiers
        .iter()
        .map(|(_, t, _)| *t)
        .max_by_key(|t| t.pom_spec().min_vram_mb)
        .unwrap_or(keryx_miner::models::Tier::Light);
    // The primary tier came from --force-model → skip the VRAM capability gate below. (If an AUTO
    // card independently reached the same tier it fits anyway, so the gate would pass regardless.)
    #[cfg(not(feature = "pom-opencl"))]
    let tier_forced = device_tiers
        .iter()
        .any(|(_, t, forced)| *forced && *t == tier);

    // Warn if GPU power limit is below safe threshold for the RESOLVED model tier (post-auto).
    // Low PL causes CUDA FIFO instability (Xid 32) under large GEMM workloads. Driven by the
    // resolved tier (not the raw flags) so an auto-picked High/VeryHigh card gets the right warning.
    {
        use keryx_miner::models::Tier;
        check_gpu_power_limit(matches!(tier, Tier::High | Tier::VeryHigh), matches!(tier, Tier::VeryHigh));
    }
    // Stage the FINAL (post-H2) lineup ONLY. The legacy pre-OPoI-v2 lineup (TinyLlama/DeepSeek) is
    // DEAD — OPoI-v2 is permanently in the past, so `specs_for(0, ..)` would just download an
    // obsolete model nobody serves (the "why is it downloading tinyllama" bug reports). Upstream
    // dropped it for the same reason; we mirror that here.
    // The FINAL (post-H2) lineup: `specs_for` is DAA-gated, and H2 (VERY_LIGHT_ACTIVATION_DAA)
    // is a frozen frontier the network is permanently past. Using OPOI_V2_ACTIVATION_DAA (pre-H2)
    // wrongly staged `--very-light` as Gemma (pre-H2 fallback) instead of Qwen3-1.7B, and
    // `--very-high` as the 48 GB Q4 instead of the 32 GB-servable Q2_K_L. Must match the mining
    // model (`Tier::pom_spec`) and the node's `POM_TIERS_H2`.
    // --force-model contract: the forced model loads REGARDLESS of VRAM fit (power-user knob), so
    // the capability gate is skipped for it — otherwise filter_specs_by_vram silently drops the
    // forced spec and PoM never configures (the "--force-model ignored" bug class, issue #7).
    let specs_all = keryx_miner::models::specs_for(keryx_miner::models::VERY_LIGHT_ACTIVATION_DAA, tier);
    let specs_v2 = if tier_forced {
        info!("--force-model: VRAM capability gate skipped for the forced model — it will load regardless of fit (may OOM an undersized card).");
        specs_all
    } else {
        filter_specs_by_vram(specs_all)
    };
    // PoM: the highest-VRAM v2 model with a pinned R_T is the tier this GPU proves possession of.
    let pom_spec = specs_v2
        .iter()
        .copied()
        .filter(|s| keryx_miner::models::pom_tier_index(&s.model_id, keryx_miner::models::VERY_LIGHT_ACTIVATION_DAA).is_some())
        .max_by_key(|s| s.min_vram_mb);
    keryx_miner::slm::set_v2_lineup(specs_v2);
    keryx_miner::slm::init_supported(specs_v2);
    info!(
        "OPoI Phase-3 — {} uncensored model(s) staged (legacy lineup dropped, post-fork).",
        specs_v2.len(),
    );
    // Prefetch BOTH lineups in the BACKGROUND (suprnova: backgrounded so a worker/plugin error
    // surfaces immediately instead of after a multi-GB download — the HiveOS "black screen" fix).
    // The OPoI hard gate keeps PoW suspended until the files are ready and un-suspends itself.
    info!("Prefetching model files in the background — PoW stays OPoI-gated until they're ready…");
    tokio::spawn(async move {
        // The MINING-TIER model FIRST and on its own — its possession index is what mining needs.
        // The big inference tiers must NOT fill a small disk (HiveOS system SSD) before it lands.
        // NOTE: filter_specs_by_vram SKIPS the VRAM gate when the VRAM query fails (e.g. nvidia-smi
        // not on h-run.sh's PATH), so without this the legacy 32B/70B can download ahead of Gemma
        // and ENOSPC it out → "index build failed: no such file or directory" on HiveOS. Downloading
        // the mining tier first guarantees mining works even if the rest later fail on a full disk.
        if let Some(ps) = pom_spec {
            let one: &'static [&'static keryx_miner::models::ModelSpec] =
                Box::leak(vec![ps].into_boxed_slice());
            match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(one)).await {
                Ok(Ok(())) => info!("Mining-tier model '{}' ready — PoM can build its possession index.", ps.dir_name),
                Ok(Err(e)) => warn!("Mining-tier model '{}' prefetch failed: {} — PoM index build will wait + retry.", ps.dir_name, e),
                Err(e) => warn!("Mining-tier model prefetch task panicked: {}", e),
            }
        }
        // Then the rest (best-effort) for OPoI inference. A failure here (e.g. small disk) only skips
        // inference tasks — mining already has its model.
        match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(specs_v2)).await {
            Ok(Ok(())) => info!("Model files ready (uncensored lineup) — OPoI inference available."),
            Ok(Err(e)) => warn!("Model prefetch failed — AiRequest tasks will be skipped: {}", e),
            Err(e) => warn!("Model prefetch task panicked — AiRequest tasks will be skipped: {}", e),
        }
    });

    // PoM possession setup is LAZY: the index + GPU walk are built by the mining loop on the first
    // PoM-active job (DAA >= activation). Here we only record cheap config — which tier this GPU
    // mines, picked by VRAM. Driver seam: AMD = OpenCL (gguf,tier); NVIDIA = candle-CUDA (model_id,gguf)
    // with zero-dup VRAM sharing.
    #[cfg(any(feature = "pom-opencl", feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
    if let Some(spec) = pom_spec {
        let tier_idx = keryx_miner::models::pom_tier_index(&spec.model_id, keryx_miner::models::VERY_LIGHT_ACTIVATION_DAA).expect("pom_spec has a tier");
        let gpath = keryx_miner::slm::gguf_path_for(spec).to_string_lossy().into_owned();
        // PoM PASSTHROUGH live test (KERYX_POM_PASSTHROUGH): build the HOST possession index in the
        // background so kHeavyHash shares can carry a PomProof (daemon stores it pre-fork). Heavy
        // (~minutes) — backgrounded so mining starts immediately; shares before it is ready submit
        // without a proof. No GPU model needed (proof gen reads the host index).
        if keryx_miner::pom::passthrough_enabled() {
            warn!("PoM: PASSTHROUGH mode (KERYX_POM_PASSTHROUGH) — kHeavyHash shares will carry a PomProof for the pre-fork wire test.");
            let gpath_pt = gpath.clone();
            std::thread::spawn(move || {
                info!("PoM passthrough: building host possession index for proof attachment…");
                match keryx_miner::pom::WeightIndex::build_from_gguf(&gpath_pt) {
                    Ok(idx) => {
                        info!("PoM passthrough: index ready — N={} chunks; shares now carry a proof.", idx.n_chunks);
                        keryx_miner::pom::set_index(idx, tier_idx);
                    }
                    Err(e) => warn!("PoM passthrough: index build failed ({}) — shares submit without a proof.", e),
                }
            });
        }
        // AMD GPU inference: bring up the Vulkan llama.cpp server once the model GGUF is on disk
        // (background — download + VRAM load is slow). If it can't come up (no bundled llama-server,
        // no Vulkan ICD, no AMD GPU, OOM), OPoI inference transparently falls back to candle-CPU.
        #[cfg(feature = "pom-opencl")]
        {
            let gpath_llama = gpath.clone();
            std::thread::spawn(move || {
                let port: u16 = std::env::var("KERYX_LLAMA_PORT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(18099);
                let ok_marker = std::path::Path::new(&gpath_llama).parent().map(|d| d.join(".ok"));
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1800);
                loop {
                    let ready = std::path::Path::new(&gpath_llama).exists()
                        && ok_marker.as_ref().map_or(true, |m| m.exists());
                    if ready {
                        break;
                    }
                    if std::time::Instant::now() > deadline {
                        warn!("PoM(AMD): model GGUF not ready in 30 min — OPoI inference stays on candle-CPU.");
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
                keryx_miner::llama_vulkan::try_start(&gpath_llama, port);
            });
        }
        #[cfg(feature = "pom-opencl")]
        keryx_miner::pom_opencl::set_mining_tier(gpath, tier_idx);
        #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
        {
            // Force the single-device split loader so the mining tier exposes its quant tensors
            // for zero-dup VRAM sharing with inference (avoids a 2nd full copy on the big tiers).
            keryx_miner::slm::set_pom_force_split(true);
            // Global DEFAULT model = the primary (biggest) tier — the inference card and any card
            // without a per-device override use it. On a UNIFORM rig no overrides are set below, so
            // this is byte-identical to the previous single-model path.
            keryx_miner::pom_gpu::set_mining_tier(spec.model_id, gpath);
            // Per-card overrides: cards whose best-fit (or --force-model) differs from the primary
            // get their OWN model, and those distinct smaller models are prefetched too.
            let mut extra: Vec<&'static keryx_miner::models::ModelSpec> = Vec::new();
            for (dev, t, _) in &device_tiers {
                let ds = t.pom_spec();
                if ds.model_id != spec.model_id {
                    let gp = keryx_miner::slm::gguf_path_for(ds).to_string_lossy().into_owned();
                    keryx_miner::pom_gpu::set_device_model(*dev, ds.model_id, gp);
                    if !extra.iter().any(|e| e.model_id == ds.model_id) {
                        extra.push(ds);
                    }
                }
            }
            if !extra.is_empty() {
                info!(
                    "Per-card model selection ACTIVE: {} distinct smaller model(s) for lighter cards \
                     (primary = {} on the biggest card) — prefetching.",
                    extra.len(),
                    spec.dir_name,
                );
                let leaked: &'static [&'static keryx_miner::models::ModelSpec] =
                    Box::leak(extra.into_boxed_slice());
                tokio::spawn(async move {
                    let _ = tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(leaked)).await;
                });
            }
        }
        // Apple Silicon (Metal): record the mining tier (global). Phase 1 loads a standalone Metal
        // walk (no inference VRAM sharing yet), so no `set_pom_force_split` here.
        #[cfg(all(target_os = "macos", feature = "pom-metal", not(feature = "pom-opencl"), not(feature = "pom-cuda")))]
        {
            keryx_miner::pom_gpu::set_mining_tier(spec.model_id, gpath);
        }
        info!(
            "PoM: configured for tier {} ({}); possession index + GPU walk load lazily at DAA {}.",
            tier_idx, spec.dir_name, keryx_miner::pom::POM_ACTIVATION_DAA
        );
        info!(
            "H3 hardfork: this build is H3-ready. It auto-switches to the post-fork PoM convention \
             (salted folds) at DAA {} — no restart needed. Update now and leave it running.",
            keryx_miner::pom::level_activation_daa()
        );
        if keryx_miner::pom::is_level_activation_overridden() {
            warn!(
                "PoM: H3 LEVEL-ACTIVATION DAA OVERRIDDEN to {} via KERYX_POM_LEVEL_ACTIVATION_DAA — staging/testing ONLY!",
                keryx_miner::pom::level_activation_daa()
            );
        }
        if keryx_miner::pom::is_activation_overridden() {
            warn!(
                "PoM: ACTIVATION DAA OVERRIDDEN to {} via KERYX_POM_ACTIVATION_DAA — staging/testing ONLY!",
                keryx_miner::pom::activation_daa()
            );
        }
    }

    // Verify GPU inference works before mining. OPoI challenges are mandatory, so a miner
    // that cannot run inference must fail fast with a clear message rather than spam panics.
    if opt.cpu_inference {
        // Explicit operator opt-out: force OPoI inference onto the CPU for the whole session.
        keryx_miner::slm::set_cpu_inference(true);
    }
    if opt.no_shared_inference {
        // One process per GPU: keep inference on this process's own card (no global-biggest pile-up).
        info!("--no-shared-inference: OPoI inference will run on this process's own --cuda-device GPU.");
        keryx_miner::slm::set_no_shared_inference(true);
    }
    if opt.cpu_inference || keryx_miner::slm::cpu_inference_enabled() {
        info!("CPU inference mode — skipping the GPU/cuBLAS probe (OPoI inference runs on the CPU).");
    } else if cfg!(all(target_os = "macos", feature = "pom-metal")) {
        // Apple Silicon: inference targets the Metal GPU (candle-metal), not cuBLAS — the cuBLAS
        // probe is meaningless here. If candle-metal can't run this model, load_engine_with_fallback
        // degrades to CPU on its own.
        info!("Apple Silicon: OPoI inference targets the Metal GPU (auto-falls back to CPU if candle-metal can't run this model). Skipping the cuBLAS probe.");
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

    // Independent WEDGE SUPERVISOR. The stratum job-watchdog lives INSIDE listen()'s select loop, so
    // a listen() that BLOCKS on a hung `handle_message().await` (e.g. a stalled GPU worker
    // back-pressuring the job-dispatch channel) freezes it too — the reconnect loop below never
    // re-iterates and no watchdog can fire. This task is FULLY independent of the stratum loop and
    // the walk threads: it watches the accepted-share counter and, if it stops advancing for
    // KERYX_STALL_RESTART_SECS while mining is expected, exits (code 75) so the wrapper (HiveOS
    // agent / systemd Restart=always / external watchdog) relaunches a fresh process — which also
    // resets any wedged CUDA workers. Keying on the FINAL output (accepted shares) catches every
    // wedge class. Arms only AFTER the first accepted share, so model download / pre-activation /
    // brief OPoI inference pauses never trip it.
    {
        let stall_secs: u64 = std::env::var("KERYX_STALL_RESTART_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 60)
            .unwrap_or(600);
        tokio::spawn(async move {
            let mut last_acc: u64 = 0;
            let mut last_change = Instant::now();
            let mut armed = false;
            let mut tick = tokio::time::interval(Duration::from_secs(30));
            loop {
                tick.tick().await;
                let stats = match crate::client::stratum::share_stats() {
                    Some(s) => s,
                    None => continue, // client not connected yet
                };
                let acc = stats.accepted.load(std::sync::atomic::Ordering::SeqCst);
                if acc > last_acc {
                    last_acc = acc;
                    last_change = Instant::now();
                    armed = true;
                } else if armed && last_change.elapsed() >= Duration::from_secs(stall_secs) {
                    error!(
                        "WEDGE SUPERVISOR: no accepted share for {}s while mining — the miner is stuck; \
                         exiting (code 75) so the wrapper relaunches it and resets stuck workers.",
                        stall_secs
                    );
                    std::process::exit(75);
                }
            }
        });
    }

    // ── POOL FAILOVER (opt-in) ─────────────────────────────────────────────────────────────────
    // Priority list: index 0 = primary (-s), then each --backup-pool in the order given. With no
    // backup the list is length 1 and ALL of the failover machinery below is skipped → the reconnect
    // loop behaves exactly as before. `desired_pool()` (a global, default 0) is the single source of
    // truth for which pool to serve; the loop follows it, and two background tasks steer it:
    //   • failover monitor  — no job (block_template_ctr frozen) for FAILOVER_AFTER on the current
    //                          pool → step DOWN to the next pool in the list.
    //   • failback prober   — while off the primary, TCP-probe higher-priority pools and step UP to
    //                          the highest one that's reachable again (with anti-flap backoff).
    // A stalled/dead current pool also surfaces as client_main returning Err (drop / job-watchdog);
    // both the monitor's step-down and the prober's step-up just move `desired_pool`, and the
    // loop + listen()'s switch-check reconnect to it. Only the primary is failed back TO the top.
    let pools: Vec<String> = {
        let mut v = vec![opt.keryxd_address.clone()];
        v.extend(opt.backup_pool.iter().cloned());
        v
    };
    if pools.len() > 1 {
        crate::client::stratum::set_failover_enabled(true);
        let failover_after: u64 = std::env::var("KERYX_FAILOVER_AFTER_SECS")
            .ok().and_then(|s| s.parse().ok()).filter(|&n| n >= 30).unwrap_or(90);
        let grace: u64 = std::env::var("KERYX_FAILBACK_GRACE_SECS")
            .ok().and_then(|s| s.parse().ok()).filter(|&n| n >= 10).unwrap_or(60);
        let probe_secs: u64 = std::env::var("KERYX_FAILBACK_PROBE_SECS")
            .ok().and_then(|s| s.parse().ok()).filter(|&n| n >= 10).unwrap_or(30);
        info!(
            "Pool failover ENABLED — primary + {} backup(s): {:?}. Failover after {}s no-job; failback grace {}s.",
            pools.len() - 1, pools, failover_after, grace
        );

        // FAILOVER MONITOR — steps DOWN the list when the current pool delivers no jobs.
        {
            let ctr = block_template_ctr.clone();
            let n = pools.len();
            let pools_dbg = pools.clone();
            tokio::spawn(async move {
                let mut last_ctr = ctr.load(std::sync::atomic::Ordering::SeqCst);
                let mut last_job = Instant::now();
                let mut tick = tokio::time::interval(Duration::from_secs(10));
                loop {
                    tick.tick().await;
                    let c = ctr.load(std::sync::atomic::Ordering::SeqCst);
                    if c != last_ctr {
                        last_ctr = c;
                        last_job = Instant::now();
                    } else if last_job.elapsed() >= Duration::from_secs(failover_after) {
                        let cur = crate::client::stratum::desired_pool();
                        if cur + 1 < n {
                            let next = cur + 1;
                            warn!("FAILOVER: pool[{}] ({}) delivered no jobs for {}s → switching to pool[{}] ({}).",
                                cur, pools_dbg[cur], failover_after, next, pools_dbg[next]);
                            crate::client::stratum::set_desired_pool(next);
                        }
                        last_job = Instant::now(); // give the (new) pool a fresh window
                    }
                }
            });
        }

        // FAILBACK PROBER — while off the primary, TCP-probe higher-priority pools and step UP to
        // the highest reachable one. Anti-flap: if a switch-up gets bounced back quickly (pool
        // reachable but jobless), back off exponentially before re-probing that far up again.
        {
            let pools = pools.clone();
            tokio::spawn(async move {
                let max_backoff = 900u64; // cap re-probe backoff at 15 min
                let mut backoff = grace; // first probe after the grace period
                loop {
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    let cur = crate::client::stratum::desired_pool();
                    if cur == 0 {
                        backoff = probe_secs; // already on primary — nothing to fail back to
                        continue;
                    }
                    // probe higher-priority pools (0..cur) in order; switch to the first reachable.
                    let mut switched_to: Option<usize> = None;
                    for i in 0..cur {
                        if tcp_pool_reachable(&pools[i], Duration::from_secs(10)).await {
                            warn!("FAILBACK: higher-priority pool[{}] ({}) is reachable — switching back up.", i, pools[i]);
                            crate::client::stratum::set_desired_pool(i);
                            switched_to = Some(i);
                            break;
                        }
                    }
                    match switched_to {
                        Some(i) => {
                            // Verify it sticks: if the failover monitor bounces us back off pool i
                            // within ~2 min (reachable but jobless), grow the backoff; else reset.
                            tokio::time::sleep(Duration::from_secs(120)).await;
                            if crate::client::stratum::desired_pool() <= i {
                                backoff = probe_secs; // stuck on i or higher — healthy failback
                            } else {
                                backoff = (backoff * 2).min(max_backoff); // bounced → back off
                                warn!("FAILBACK: pool[{}] was reachable but not delivering jobs — backing off {}s before retrying.", i, backoff);
                            }
                        }
                        None => backoff = (backoff.max(probe_secs) * 2).min(max_backoff),
                    }
                }
            });
        }
    }

    loop {
        let started = Instant::now();
        // Serve whichever pool the failover controller currently wants (index 0 = primary when no
        // failover). Stamp ACTIVE_POOL so listen()'s switch-check matches until a task changes it.
        let target = crate::client::stratum::desired_pool().min(pools.len() - 1);
        crate::client::stratum::set_active_pool(target);
        let pool_address = pools[target].clone();
        if pools.len() > 1 {
            info!("Mining pool target: pool[{}] {}", target, pool_address);
        }
        match client_main(&opt, pool_address, block_template_ctr.clone(), &plugin_manager, escrow_privkey.clone()).await {
            Ok(_) => info!("Client closed gracefully"),
            Err(e) => error!("Client closed with error {:?}", e),
        }
        // A session that lasted a while was healthy — treat the next drop as a
        // fresh blip rather than escalating the backoff.
        if started.elapsed() >= HEALTHY_SESSION {
            backoff = RECONNECT_MIN;
        }
        // A failover/failback SWITCH (the controller picked a different pool) is NOT a failure of
        // the new target — reconnect to it immediately instead of waiting out the escalated backoff
        // that accumulated while the old pool was down. Keeps switches snappy (no long idle gap).
        if crate::client::stratum::desired_pool().min(pools.len().saturating_sub(1)) != target {
            backoff = RECONNECT_MIN;
        }
        info!("Client closed, reconnecting in {}s", backoff.as_secs());
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(RECONNECT_MAX);
    }
}

/// Best-effort TCP reachability probe of a `stratum+tcp://host:port` (or bare host:port) pool URL —
/// used by the failback prober to decide whether a higher-priority pool has recovered, WITHOUT
/// touching the active mining connection or its globals. Just a connect within `timeout`.
async fn tcp_pool_reachable(url: &str, timeout: Duration) -> bool {
    let addr = url.split_once("://").map(|(_, a)| a).unwrap_or(url);
    matches!(
        tokio::time::timeout(timeout, tokio::net::TcpStream::connect(addr)).await,
        Ok(Ok(_))
    )
}
