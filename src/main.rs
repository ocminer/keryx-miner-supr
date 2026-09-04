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
mod gpu_health;
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
    let has_apt = Command::new("sh").args(["-c", "command -v apt-get"]).status().map(|s| s.success()).unwrap_or(false);
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
    Command::new("bash").args(["-c", script]).status().map(|s| s.success()).unwrap_or(false)
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
/// Current H6 auto-select floors (weights + KV/workspace allowance): Qwen3.5-9B 7 GB,
/// GLM-4-9B 11 GB, Gemma-4-12B 15 GB, Qwen3.6-27B 22 GB, Kimi-Linear-48B 28 GB.
///
/// Power thresholds empirically derived: Xid 32 observed at ≤300W on RTX 3090 with 32B GGUF.
fn first_visible_cuda_device(visible: Option<&str>) -> Option<&str> {
    visible?.split(',').next().map(str::trim).filter(|token| !token.is_empty())
}

fn pin_nvidia_smi_to_device(command: &mut std::process::Command, visible: Option<&str>) {
    if let Some(device) = first_visible_cuda_device(visible) {
        // `Command::arg` passes the UUID/index verbatim without a shell. UUIDs are important here:
        // production pins by `GPU-...`, and silently accepting only numeric tokens queried physical
        // GPU 0 instead of the process's logical GPU 0 on heterogeneous rigs.
        command.arg("-i").arg(device);
    }
}

fn pin_nvidia_smi_to_visible_device(command: &mut std::process::Command) {
    pin_nvidia_smi_to_device(command, std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref());
}

fn check_gpu_power_limit(needs_high: bool, needs_very_high: bool) {
    let mut command = std::process::Command::new("nvidia-smi");
    pin_nvidia_smi_to_visible_device(&mut command);
    let output = command
        .args(["--query-gpu=power.limit,power.max_limit,memory.total", "--format=csv,noheader,nounits"])
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

    // VRAM check for the current top tier. On 24 GB cards the Kimi-Linear-48B
    // model cannot coexist with its KV/workspace and will OOM.
    if needs_very_high && vram_mb < 30_000 {
        log::error!(
            "✗  Kimi-Linear-48B requires about 30 GB VRAM — \
             this GPU has only {} MB ({} GB).",
            vram_mb,
            vram_mb / 1024
        );
        log::error!("   Use --high (Qwen3.6-27B) or a smaller H6 tier.");
        // Non-fatal: let candle fail with its own OOM so the miner logs the actual error.
    }

    let model_label = if needs_very_high {
        "Kimi-Linear-48B (very-high)"
    } else if needs_high {
        "Qwen3.6-27B (high)"
    } else {
        "Qwen3.5-9B / GLM-9B / Gemma-12B"
    };
    log::info!("GPU: {}W PL, {} MB VRAM — ready for {}", current_w, vram_mb, model_label);
}

/// Primary mining GPU VRAM (MB). An OpenCL build must ask the OpenCL driver first because
/// `nvidia-smi` ignores the plugin's platform/device subset and can report an unrelated NVIDIA GPU
/// on a mixed-vendor host. The OpenCL helper is already scoped to the exact selected worker list.
/// NVIDIA-only builds retain the CUDA-visible `nvidia-smi` query below.
fn query_vram_mb() -> Option<u64> {
    #[cfg(feature = "pom-opencl")]
    {
        keryx_miner::pom_opencl::gpu0_global_mem_mb()
    }
    #[cfg(not(feature = "pom-opencl"))]
    {
        // One process per GPU (CUDA_VISIBLE_DEVICES=<n>): query THIS process's visible GPU, not GPU 0.
        // nvidia-smi without `-i` always lists GPU 0 first regardless of CUDA_VISIBLE_DEVICES, so on a
        // mixed rig (e.g. a 5090 at slot 0 + a 3070 at slot 1) the 3070's process would wrongly read the
        // 5090's 32 GB and auto-pick a tier that OOMs. Pin nvidia-smi to the first visible device id.
        // (Miners launch with CUDA_DEVICE_ORDER=PCI_BUS_ID, so CUDA's index == nvidia-smi's index.)
        let mut cmd = std::process::Command::new("nvidia-smi");
        pin_nvidia_smi_to_visible_device(&mut cmd);
        if let Ok(output) = cmd.args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"]).output() {
            if output.status.success() {
                if let Some(mb) =
                    String::from_utf8_lossy(&output.stdout).lines().next().and_then(|l| l.trim().parse::<u64>().ok())
                {
                    return Some(mb);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod device_query_tests {
    use super::{first_visible_cuda_device, pin_nvidia_smi_to_device};

    #[test]
    fn visible_device_parser_accepts_uuid_and_preserves_first_mapping() {
        assert_eq!(first_visible_cuda_device(Some(" GPU-deadbeef ,2")), Some("GPU-deadbeef"));
        assert_eq!(first_visible_cuda_device(Some("3,1")), Some("3"));
        assert_eq!(first_visible_cuda_device(Some(" ,2")), None);
        assert_eq!(first_visible_cuda_device(None), None);
    }

    #[test]
    fn nvidia_smi_pin_passes_uuid_as_one_literal_argument() {
        let mut command = std::process::Command::new("nvidia-smi");
        pin_nvidia_smi_to_device(&mut command, Some("GPU-deadbeef,2"));
        let args = command.get_args().map(|arg| arg.to_string_lossy().into_owned()).collect::<Vec<_>>();
        assert_eq!(args, vec!["-i", "GPU-deadbeef"]);
    }
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
    {
        keryx_miner::pom_gpu::query_all_gpus_vram()
    }
    #[cfg(not(any(
        all(feature = "pom-cuda", not(all(target_os = "macos", feature = "pom-metal"))),
        all(target_os = "macos", feature = "pom-metal")
    )))]
    {
        Vec::new()
    }
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
                    spec.name,
                    spec.min_vram_mb,
                    gpu0_mb
                );
                false
            }
        })
        .collect();
    if kept.is_empty() {
        // NEVER stage an empty lineup: init_supported([]) leaves SUPPORTED_SPECS empty → the miner
        // reports "no model lineup installed — waiting for chain tip" and NEVER downloads a model
        // (the Windows "silent stall" bug). If the VRAM gate would drop everything the query is
        // almost certainly wrong/undersized — keep the lineup so the model downloads and mining can
        // start; a genuinely-too-big model then OOM-demotes to a fitting tier at walk build.
        log::warn!(
            "VRAM capability gate would drop ALL {} model(s) (GPU VRAM read as {} MB — likely an \
             unreliable query) — keeping the lineup so the model still downloads and mining can start.",
            specs.len(),
            gpu0_mb
        );
        return specs;
    }
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

/// Legacy call-site headroom value retained for API compatibility. The current H6 tier floors in
/// `auto_select_tier` already include the field-proven weights/KV/workspace allowance, so that
/// function intentionally does not add this value a second time. The OpenCL residency planner
/// separately decides whether an inference card must be dedicated when walk + inference do not fit.
#[cfg(feature = "pom-opencl")]
const AUTO_TIER_HEADROOM_MB: u64 = 5_120;

/// `--tier auto` for the AMD/OpenCL path: pick the largest H6 tier that fits the first selected
/// worker card's VRAM
/// (`CL_DEVICE_GLOBAL_MEM_SIZE` via `pom_opencl::gpu0_global_mem_mb`) with the AMD margin above.
/// The tier is PROCESS-WIDE on AMD (one resident model for all cards — no per-card map), so it is
/// gated on that primary worker. Falls back to VeryLight (Qwen3.5-9B) if VRAM cannot be
/// queried.
#[cfg(feature = "pom-opencl")]
fn select_tier_auto() -> keryx_miner::models::Tier {
    use keryx_miner::models::{self, Tier};
    let Some(vram_mb) = query_vram_mb() else {
        warn!(
            "AMD/OpenCL --tier auto: cannot query the selected primary worker's VRAM — falling back to --very-light (Qwen3.5-9B)."
        );
        return Tier::VeryLight;
    };
    let (picked, need) = models::auto_select_tier(vram_mb, AUTO_TIER_HEADROOM_MB);
    info!(
        "AMD/OpenCL tier auto: {} MiB VRAM (selected primary worker) -> {} ({:?}, tier {}) — budget {} MiB \
         (model weights+KV+ctx + resident PoM blob + {} MiB AMD margin). Pin a tier with \
         --light/--high/--force-model to override.",
        vram_mb,
        picked.pom_model_name(),
        picked,
        models::pom_tier_index(&picked.pom_spec().model_id, keryx_miner::pom::pom_v3_activation_daa()).unwrap_or(0),
        need,
        AUTO_TIER_HEADROOM_MB,
    );
    picked
}

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
                info!("--tier very-light: tier 0 — mines Qwen3.5-9B under PoM.");
                return Tier::VeryLight;
            }
            "light" => {
                info!("--tier light: tier 1 — mines GLM-4-9B under PoM.");
                return Tier::Light;
            }
            "default" => {
                info!("--tier default: tier 2 — mines Gemma-4-12B under PoM.");
                return Tier::Default;
            }
            "high" => {
                info!("--tier high: tier 3 — mines Qwen3.6-27B under PoM.");
                return Tier::High;
            }
            "very-high" | "veryhigh" | "very_high" => {
                info!("--tier very-high: tier 4 — mines Kimi-Linear-48B under PoM.");
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
        info!("--very-high mode: tier 4 — mines Kimi-Linear-48B under PoM.");
        Tier::VeryHigh
    } else if opt.high {
        info!("--high mode: tier 3 — mines Qwen3.6-27B under PoM.");
        Tier::High
    } else if opt.light {
        info!("--light mode: tier 1 — mines GLM-4-9B under PoM.");
        Tier::Light
    } else if opt.very_light {
        info!("--very-light mode: tier 0 — mines Qwen3.5-9B under PoM.");
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

    // Prefer the CUDA-driver VRAM (query_all_gpus_vram) and only fall back to the nvidia-smi CLI.
    // The nvidia-smi `-i 0` form fails on some setups (notably Windows), which used to make this
    // fall back to Tier::Light (GLM, 12 GB) — a model `filter_specs_by_vram` (which DOES read the
    // CUDA driver) then dropped on a 10 GB card, leaving an EMPTY lineup: "no model lineup installed
    // — no download — stall". Reading the same source here keeps the two in agreement; the fallback
    // tier is the smallest model (fits any card) so an unknown VRAM never stages an unloadable tier.
    let vram_mb = all_gpus_vram().into_iter().map(|(_, m)| m).max().or_else(query_vram_mb);
    let Some(vram_mb) = vram_mb else {
        warn!("--tier auto: cannot query GPU VRAM (CUDA driver + nvidia-smi both failed) — falling back to --very-light (Qwen3.5-9B, smallest, fits any card).");
        return Tier::VeryLight;
    };

    let (picked, need) = models::auto_select_tier(vram_mb, AUTO_TIER_HEADROOM_MB);
    info!(
        "tier auto: {} MiB VRAM -> {} ({:?}, tier {}) — budget {} MiB (weights+KV+workspace + {} MiB margin).",
        vram_mb,
        picked.pom_model_name(),
        picked,
        models::pom_tier_index(&picked.pom_spec().model_id, keryx_miner::pom::pom_v3_activation_daa()).unwrap_or(0),
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
        let fits = vram_mb >= tier.pom_spec().min_vram_mb.saturating_add(AUTO_TIER_HEADROOM_MB) || tier == Tier::Light;
        if fits && keryx_miner::slm::spec_files_ready(tier.pom_spec()) {
            warn!(
                "tier auto: falling back to {} ({:?}, tier {}) — largest fitting tier already on disk. \
                 ({} will download in the background; restart to use it.)",
                tier.pom_model_name(),
                tier,
                models::pom_tier_index(&tier.pom_spec().model_id, keryx_miner::pom::pom_v3_activation_daa())
                    .unwrap_or(0),
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
    if opt.very_high {
        Some(Tier::VeryHigh)
    } else if opt.high {
        Some(Tier::High)
    } else if opt.light {
        Some(Tier::Light)
    } else if opt.very_light {
        Some(Tier::VeryLight)
    } else {
        None
    }
}

/// Resolve the mining tier for EACH CUDA device (CUDA-driver order): `--force-model <csv>` wins
/// per-card (forced, VRAM bypassed), else a pinned flag (all cards), else per-card AUTO best-fit
/// (heaviest tier that fits that card's VRAM). Fewer force-model entries than cards → the rest AUTO.
/// Falls back to a single (device 0, legacy-resolver) entry when the CUDA driver can't enumerate.
/// The bool per entry = the tier came from `--force-model` (so the VRAM capability gate is skipped).
#[cfg(not(feature = "pom-opencl"))]
fn resolve_device_tiers(opt: &cli::Opt) -> Vec<(u32, keryx_miner::models::Tier, bool)> {
    use keryx_miner::models;
    let forced: Vec<Option<models::Tier>> =
        opt.force_model.as_deref().map(|s| s.split(',').map(|x| parse_tier_name(x)).collect()).unwrap_or_default();
    let mut vrams = all_gpus_vram(); // (device_id, TOTAL MiB), CUDA order
                                     // AUTO budgets against FREE VRAM, not total: whatever is already resident on the card
                                     // (another miner, a leaked context, a desktop) is memory the chosen tier can never
                                     // allocate — the old total-based pick sailed past the check and died in cudaMalloc,
                                     // cascading into self-test failures. Free query best-effort; total is the fallback.
    #[cfg(all(feature = "pom-cuda", not(all(target_os = "macos", feature = "pom-metal"))))]
    let free_map: std::collections::HashMap<u32, (u64, u64)> =
        keryx_miner::pom_gpu::query_all_gpus_free_vram().into_iter().map(|(d, f, t)| (d, (f, t))).collect();
    #[cfg(not(all(feature = "pom-cuda", not(all(target_os = "macos", feature = "pom-metal")))))]
    let free_map: std::collections::HashMap<u32, (u64, u64)> = std::collections::HashMap::new();
    if vrams.is_empty() {
        // No enumeration (nvidia-smi/driver missing): a forced first entry still wins for GPU 0.
        return match forced.first().copied().flatten() {
            Some(t) => vec![(0, t, true)],
            None => vec![(0, select_tier_nvidia(opt), false)],
        };
    }
    vrams.sort_by_key(|(id, _)| *id);
    if opt.force_model.is_some() {
        // Forced models load verbatim: no VRAM pre-check ("we simply load it, regardless of
        // what the card says — that is the reason we have it").
        #[cfg(any(
            all(feature = "pom-cuda", not(feature = "pom-opencl")),
            all(target_os = "macos", feature = "pom-metal")
        ))]
        keryx_miner::llama_engine::set_vram_check_bypass(true);
        info!("--force-model: VRAM pre-check disabled — forced models are loaded verbatim.");
    }
    if let Some(raw) = opt.force_model.as_deref() {
        info!(
            "--force-model: {} — per-card override (VRAM check bypassed; unlisted/extra cards use auto).",
            raw.trim()
        );
    }
    let pinned = pinned_tier(opt);
    let mut out = Vec::with_capacity(vrams.len());
    for (dev, vram) in &vrams {
        let (tier, is_forced) = match forced.get(*dev as usize).copied().flatten() {
            Some(t) => {
                // --force-model means FORCE: honor the operator's choice with NO VRAM check. If it
                // does not fit the card it will OOM — that is the user's explicit call. (Only AUTO
                // selection below is VRAM-aware.)
                info!(
                    "GPU {}: --force-model → {} (forced, no VRAM check; card has {} MiB).",
                    dev,
                    t.pom_model_name(),
                    vram
                );
                (t, true)
            }
            None => {
                if forced.get(*dev as usize).is_some() {
                    warn!("GPU {}: --force-model entry unrecognised — using auto (names: very-light|light|default|high|very-high).", dev);
                }
                if let Some(t) = pinned {
                    (t, false)
                } else {
                    let budget_mb = match free_map.get(dev) {
                        Some(&(free, _total)) if free < *vram => {
                            info!(
                                "GPU {}: {} MiB of {} MiB VRAM already in use by other processes —                                  auto-tier budgets against the {} MiB actually FREE.",
                                dev, *vram - free, vram, free
                            );
                            free
                        }
                        _ => *vram,
                    };
                    let (picked, need) = models::auto_select_tier(budget_mb, AUTO_TIER_HEADROOM_MB);
                    info!(
                        "GPU {}: auto → {} (fits {} MiB free VRAM, budget {} MiB).",
                        dev,
                        picked.pom_model_name(),
                        budget_mb,
                        need
                    );
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
    escrow_cert: Option<String>,
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
            escrow_cert,
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
    escrow_cert: Option<String>,
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
        escrow_cert,
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
    std::env::var("KERYX_ASYNC_WORKERS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(2).clamp(1, 8)
}

/// Optional cap for the `spawn_blocking` pool (SLM inference, IPFS upload, model prefetch). Only
/// applied when `KERYX_BLOCKING_THREADS` is set: the blocking pool spawns lazily and idles out, so
/// tokio's default costs nothing at rest and capping it low would bottleneck parallel multi-model
/// prefetch on multi-GPU rigs.
fn tokio_blocking_threads() -> Option<usize> {
    std::env::var("KERYX_BLOCKING_THREADS").ok().and_then(|s| s.parse::<usize>().ok()).map(|n| n.clamp(2, 64))
}

fn main() -> Result<(), Error> {
    // Force CUDA's device enumeration to PCI-bus order BEFORE any CUDA library initializes. The whole
    // codebase assumes CUDA's ordinal == nvidia-smi's ordinal (see the CUDA_VISIBLE_DEVICES handling
    // below and gpu_health.rs) — an assumption that only holds under CUDA_DEVICE_ORDER=PCI_BUS_ID.
    // Without it CUDA defaults to FASTEST_FIRST, so on a MIXED-VRAM rig the runtime/ggml layer and the
    // keryxcuda PoM plugin enumerate the cards in DIFFERENT orders: auto-tier then sizes a model for
    // one physical card but the PoM walk binds it to another, the oversized model OOMs the smaller
    // card, and that card's staging failure suspends the WHOLE rig — zero shares. The launcher used to
    // export this, but a user running the binary directly got none of it. Guarantee it in-process;
    // an explicit operator setting still wins. (No effect on single-model / homogeneous rigs.)
    if std::env::var_os("CUDA_DEVICE_ORDER").is_none() {
        std::env::set_var("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    }
    // AMD OpenCL caps a SINGLE cl_mem buffer at CL_DEVICE_MAX_MEM_ALLOC_SIZE. On Polaris (RX 580 and
    // kin) that cap is often ~25% of VRAM (or a hard ~4 GB), which is TOO SMALL for the current
    // tier-0 Qwen3.5-9B possession blob. When it exceeds the cap the driver hands back a partial/broken
    // buffer, so the card hashes at full rate but the walk reads garbage → it NEVER finds a valid
    // share and submits nothing. Raise the single-allocation cap to ~100% of VRAM BEFORE the AMD
    // OpenCL runtime initializes (it reads these env vars at platform init). Operator overrides win.
    #[cfg(feature = "pom-opencl")]
    for (k, v) in [("GPU_SINGLE_ALLOC_PERCENT", "100"), ("GPU_MAX_ALLOC_PERCENT", "100"), ("GPU_MAX_HEAP_SIZE", "100")]
    {
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, v);
        }
    }
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
    let (app, mut plugin_manager): (App, PluginManager) = keryx_miner::load_plugins(Opt::into_app(), &plugins)?;

    // macOS (Apple Silicon): register the built-in Metal PoM worker. Without this the miner finds
    // 0 workers and exits ("No workers specified"). --disable-gpu leaves a CPU-only miner.
    #[cfg(all(target_os = "macos", feature = "pom-metal"))]
    let plugins: Vec<String> = if disable_gpu { Vec::new() } else { vec!["builtin:metal (Apple Silicon)".to_string()] };
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
    let plugins: Vec<String> = if disable_gpu { Vec::new() } else { vec!["builtin:cuda (static)".to_string()] };
    #[cfg(feature = "static-cuda")]
    let (app, mut plugin_manager): (App, PluginManager) = {
        let mut manager = PluginManager::new();
        let app = if disable_gpu {
            Opt::into_app()
        } else {
            let no_winner = keryxcuda::keryx_plugin_enable_raw_nonce_v1();
            manager.register_builtin_with_output_contract(
                Opt::into_app(),
                Box::new(keryxcuda::CudaPlugin::new()?),
                no_winner,
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
    // Init the logger BEFORE opt.process(): process() logs the resolved keryxd address, and with the
    // init after it that line was swallowed — exactly the line a solo miner needs to confirm which
    // node the miner is dialing. log_level() only reads the --debug flag, already parsed here.
    // try_init: in the static-cuda build CudaPlugin::new already set a logger;
    // init() would panic on the second call.
    let _ = env_logger::builder().filter_level(opt.log_level()).parse_default_env().try_init();
    opt.process()?;

    // --delete-autotune: wipe the saved launch tuning BEFORE any GPU is touched, so this run
    // measures every card from scratch. The cache is already version-stamped and dropped
    // automatically whenever the miner version changes; this flag exists for a re-measure WITHIN
    // one version — new GPU, riser, power-limit or clock change, or a card stuck on a bad batch.
    #[cfg(feature = "pom-cuda")]
    if opt.delete_autotune {
        match keryx_miner::pom_gpu::delete_v4_tune_cache() {
            Ok(true) => info!("--delete-autotune: removed the saved autotune cache — every GPU is re-tuned this run."),
            Ok(false) => {
                info!("--delete-autotune: no saved autotune cache to remove — every GPU is tuned from scratch anyway.")
            }
            Err(e) => warn!(
                "--delete-autotune: could NOT remove the autotune cache: {e}. Continuing with the \
                 existing tuning — delete ~/.keryx/v4tune.json by hand if that is not what you want."
            ),
        }
    }

    // Resolve the resident-tree switch ONCE (read by both the CUDA and OpenCL index builders).
    // Off by default. --resident-tree or KERYX_RESIDENT_TREE=1 (back-compat) turns it on;
    // --no-resident-tree forces it off and wins over everything (escape hatch for a stray env
    // var in a flight sheet — the "I can't turn it off" complaint).
    let resident_tree = if opt.no_resident_tree {
        false
    } else {
        opt.resident_tree || std::env::var("KERYX_RESIDENT_TREE").is_ok_and(|v| v == "1")
    };
    keryx_miner::pom::set_resident_tree(resident_tree);
    log::info!(
        "PoM resident tree: {} ({})",
        if resident_tree { "ON" } else { "off" },
        if resident_tree { "lookup-time proof build, high RAM" } else { "frugal on-disk proof build" }
    );

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
    info!("                 build {}", env!("KERYX_BUILD_STAMP"));
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
            let response = ureq::get(&url_clone).call().map_err(|e| format!("HTTP request failed: {}", e))?;
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
                submit_retries: 0,
                batch_cap: 0,
                cap_set_daa: 0,
                is_inference: false,
                csv_window: escrow::csv_window_for_daa(a.confirm_daa as u64),
            })
            .collect();

        let total_sompi: u64 = entries.iter().map(|e| e.amount_sompi).sum();
        let count = entries.len();
        let state = escrow::EscrowState { entries };
        let json = serde_json::to_string_pretty(&state)?;
        fs::write(&opt.escrow_state_file, &json)?;

        info!("Recovered {} escrow entries — claimable: {:.4} KRX", count, total_sompi as f64 / 1e8);
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

    // Escrow delegation cert: binds the escrow key to the payout address. From H6 a coinbase
    // without a valid pair is an invalid block, so a bad cert fails here instead of producing
    // rejected blocks.
    // NOTE: upstream gates the hard-error on `keryx_miner::models::h6_staged()`; our tree has no
    // such helper, so the equivalent `pom_v3_activation_daa() != u64::MAX` is inlined here.
    // The escrow delegation cert is consumed ONLY by the solo (grpc) mining path — the pool
    // (stratum) path builds and signs the coinbase itself and never uses the cert. So require/
    // resolve it only when a configured pool is solo (grpc); pool mining skips it entirely.
    // (Previously this hard-errored at startup for pool miners once H6 was active.)
    let solo_mining =
        std::iter::once(&opt.keryxd_address).chain(opt.backup_pool.iter()).any(|a| !a.starts_with("stratum+tcp://"));
    let escrow_cert: Option<String> = if !solo_mining {
        info!("Pool (stratum) mining — escrow delegation cert not required (the pool signs the coinbase).");
        None
    } else {
        match (&escrow_privkey, opt.mining_address.as_deref()) {
            (Some(privkey), Some(address)) => {
                let escrow_pubkey_hex = escrow::pubkey_hex_from_privkey(privkey)?;
                let prefix = address.split(':').next().unwrap_or("keryx");
                let own_address = escrow::escrow_key_address(privkey, prefix)?;
                // Resolution order: an explicitly supplied cert wins; otherwise the miner signs its
                // own when the payout address is its escrow key's (nothing to set up); otherwise the
                // file, which is the path for a payout address whose key lives in a wallet.
                let supplied = opt.escrow_cert.as_deref().map(|c| {
                    let cert = c.trim().to_ascii_lowercase();
                    escrow::verify_escrow_cert(address, &escrow_pubkey_hex, &cert).map(|()| cert)
                });
                let resolved = match supplied {
                    Some(Ok(cert)) => {
                        info!("Escrow delegation cert taken from --escrow-cert.");
                        // Persist it so the operator does not re-pass the flag every start (signed once).
                        match escrow::save_cert(&opt.escrow_cert_file, &cert) {
                            Ok(true) => info!(
                                "Escrow delegation cert saved to '{}' — future starts need no --escrow-cert.",
                                opt.escrow_cert_file
                            ),
                            Ok(false) => {}
                            Err(e) => warn!(
                                "Could not persist escrow cert to '{}': {} (running this session anyway).",
                                opt.escrow_cert_file, e
                            ),
                        }
                        Ok(cert)
                    }
                    Some(Err(e)) => Err(e),
                    None => match escrow::self_sign_cert(privkey, address) {
                        Some(cert) => {
                            info!("Payout address is this miner's escrow key — delegation signed locally, nothing to set up.");
                            Ok(cert)
                        }
                        None => escrow::load_cert(&opt.escrow_cert_file, address, &escrow_pubkey_hex).map(|cert| {
                            info!("Escrow delegation cert loaded from '{}'.", opt.escrow_cert_file);
                            cert
                        }),
                    },
                };
                match resolved {
                    Ok(cert) => Some(cert),
                    Err(e) => {
                        if keryx_miner::pom::pom_v3_activation_daa() != u64::MAX {
                            error!("{}", e);
                            error!("Two ways to fix it, pick one:");
                            error!("  1. Mine to this miner's own address — nothing else to do: {}", own_address);
                            error!(
                                "  2. Keep your payout address and authorise this miner from the wallet holding it:"
                            );
                            error!("       keryx-cli delegate-escrow {} {}", escrow_pubkey_hex, address);
                            error!(
                                "     then pass the 128-hex output as --escrow-cert, or save it to '{}'.",
                                opt.escrow_cert_file
                            );
                            return Err(e.into());
                        }
                        warn!("No usable escrow delegation cert ({}) — it becomes mandatory at H6.", e);
                        warn!("Mining to {} would need no cert at all.", own_address);
                        None
                    }
                }
            }
            _ => None,
        }
    };

    // Resolve inference routing before ANY model warmer/probe thread is spawned. Placement is part
    // of the safety contract: a warmer that chooses first and a late --inference-cards restriction
    // can otherwise leave a model resident/proven on a forbidden card while advertising an
    // unproven allowed route.
    if opt.cpu_inference || opt.enable_cpu_inference {
        warn!(
            "CPU inference is deprecated and extremely slow; it will be attempted only because an explicit legacy flag enabled it"
        );
        keryx_miner::slm::set_cpu_inference_allowed(true);
    }
    if opt.cpu_inference {
        keryx_miner::slm::set_cpu_inference(true);
    }
    if opt.no_shared_inference {
        info!("--no-shared-inference: OPoI inference will run on this process's own --cuda-device GPU.");
        keryx_miner::slm::set_no_shared_inference(true);
    }
    if let Some(p) = keryx_miner::slm::InferencePolicy::parse(&opt.inference_policy) {
        keryx_miner::slm::set_inference_policy(p);
        info!("OPoI inference routing policy: {:?}", p);
    } else {
        warn!("--inference-policy '{}' not recognized — using default (speed).", opt.inference_policy);
    }
    if opt.no_shared_inference {
        keryx_miner::slm::set_inference_cards(vec![keryx_miner::slm::inference_gpu_ordinal()]);
    } else if let Some(list) = opt.inference_cards.clone().or_else(|| std::env::var("KERYX_INFERENCE_CARDS").ok()) {
        let tokens: Vec<&str> = list.split(',').map(str::trim).filter(|t| !t.is_empty()).collect();
        let mut cards = Vec::with_capacity(tokens.len());
        for token in tokens {
            let gpu = token.parse::<usize>().map_err(|_| {
                format!("invalid --inference-cards entry '{token}' (expected comma-separated GPU ordinals)")
            })?;
            if !cards.contains(&gpu) {
                cards.push(gpu);
            }
        }
        if cards.is_empty() {
            return Err("--inference-cards was supplied but contains no GPU ordinals".into());
        }
        #[cfg(feature = "pom-cuda")]
        {
            let visible: Vec<usize> =
                keryx_miner::pom_gpu::query_all_gpus_vram().into_iter().map(|(gpu, _)| gpu as usize).collect();
            if let Some(&bad) = cards.iter().find(|gpu| !visible.contains(gpu)) {
                return Err(format!(
                    "--inference-cards includes CUDA ordinal {bad}, but visible ordinals are {:?}",
                    visible
                )
                .into());
            }
        }
        info!("--inference-cards: OPoI inference restricted to GPU ordinals {:?}; other cards are PoW-only.", cards);
        keryx_miner::slm::set_inference_cards(cards);
    }

    // Publish low-RAM mode before any model prefetch/warmer thread starts. Setting it after those
    // threads were spawned left a startup window where every card could load concurrently despite
    // the operator explicitly requesting serialized bring-up.
    #[cfg(all(feature = "pom-cuda", not(all(target_os = "macos", feature = "pom-metal"))))]
    if opt.low_ram {
        keryx_miner::pom_gpu::set_low_ram(true);
        info!("--low-ram: loading models onto GPUs one at a time (lower peak system RAM, slower startup).");
    }

    // Phase-3 OPoI: load inference models before mining starts.
    //   (no flag)    → AUTO: largest PoM tier that fits this GPU's VRAM (per-process)
    // PoM: one flag = one H6 tier. Each GPU mines AND serves exactly the single model it
    // proves possession of (multi-tier coverage is a network property, not per-GPU).
    //   --very-light → Qwen3.5-9B; --light → GLM-9B; default → Gemma-12B;
    //   --high → Qwen3.6-27B; --very-high → Kimi-48B (no flag → AUTO).
    // AMD: ONE tier per process (the OpenCL PoM backend keeps a single resident blob — no
    // per-card model map like CUDA's set_device_model). An explicit user override (--force-model /
    // --tier / --light etc.) is honored PROCESS-WIDE; AUTO selects from the current H6 floors. A forced
    // tier SKIPS the VRAM capability gate (same power-user contract as
    // CUDA --force-model): an undersized card will fail/OOM loading the GPU-resident walk blob;
    // the model route is withdrawn if GPU inference cannot fit next to the blob.
    #[cfg(feature = "pom-opencl")]
    let (tier, tier_forced) = {
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
            info!(
                "AMD/OpenCL: tier {} (pinned by flag, process-wide — the model must fit the card's VRAM).",
                t.pom_model_name()
            );
            (t, false)
        } else {
            // No explicit tier, or `--tier auto` (pinned_tier returns None for "auto"): pick the
            // largest H6 tier that fits card 0's VRAM. Heavier tier = higher PoM reward;
            // VeryLight/Qwen3.5-9B is the floor.
            (select_tier_auto(), false)
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
    let tier_forced = device_tiers.iter().any(|(_, t, forced)| *forced && *t == tier);

    // Warn if GPU power limit is below safe threshold for the RESOLVED model tier (post-auto).
    // Low PL causes CUDA FIFO instability (Xid 32) under large GEMM workloads. Driven by the
    // resolved tier (not the raw flags) so an auto-picked High/VeryHigh card gets the right warning.
    #[cfg(not(feature = "pom-opencl"))]
    {
        use keryx_miner::models::Tier;
        check_gpu_power_limit(matches!(tier, Tier::High | Tier::VeryHigh), matches!(tier, Tier::VeryHigh));
    }
    // --force-model contract: the forced model loads REGARDLESS of VRAM fit (power-user knob), so
    // the capability gate is skipped for it — otherwise filter_specs_by_vram silently drops the
    // forced spec and PoM never configures (the "--force-model ignored" bug class, issue #7).
    // Stage the CURRENT-era lineup: resolve at the H6 gate, whose anchors (tier-0 Qwen3.5-9B) pin
    // POM_TIERS_H6 — matching `Tier::pom_spec()`. The pre-H6 lineups were retired with their models.
    let lineup_daa = keryx_miner::pom::pom_v3_activation_daa();
    let specs_all = keryx_miner::models::specs_for(lineup_daa, tier);
    let specs_v2 = if tier_forced {
        info!("--force-model: VRAM capability gate skipped for the forced model — it will load regardless of fit (may OOM an undersized card).");
        specs_all
    } else {
        filter_specs_by_vram(specs_all)
    };
    // MIXED RIG: the lineup above is the PRIMARY (biggest) tier's — but per-card auto/force can
    // assign SMALLER tiers to lighter cards (set_device_model below). Those models MUST be in
    // SUPPORTED_SPECS too: the serveability self-test resolves specs from this registry, and a
    // missing entry made a smaller card's probe fail instantly as '?' — on the old code that
    // poisoned the cache and the card never mined (found live on a 3070+5080 mixed rig). Union
    // them in (leak-once, same pattern as the filtered lineup slice).
    #[cfg(not(feature = "pom-opencl"))]
    let specs_v2 = {
        let mut extra: Vec<&'static keryx_miner::models::ModelSpec> = Vec::new();
        for (_, t, _) in &device_tiers {
            let ds = t.pom_spec();
            if !specs_v2.iter().any(|s| s.model_id == ds.model_id) && !extra.iter().any(|e| e.model_id == ds.model_id) {
                extra.push(ds);
            }
        }
        if extra.is_empty() {
            specs_v2
        } else {
            info!(
                "Mixed rig: registering {} additional per-card model(s) in the supported lineup: {}.",
                extra.len(),
                extra.iter().map(|s| s.name).collect::<Vec<_>>().join(", ")
            );
            let mut v: Vec<&'static keryx_miner::models::ModelSpec> = specs_v2.to_vec();
            v.extend(extra);
            &*Box::leak(v.into_boxed_slice())
        }
    };
    // PoM: the highest-VRAM v2 model with a pinned R_T is the tier this GPU proves possession of.
    let pom_spec = specs_v2
        .iter()
        .copied()
        .filter(|s| keryx_miner::models::pom_tier_index(&s.model_id, lineup_daa).is_some())
        .max_by_key(|s| s.min_vram_mb);
    keryx_miner::slm::set_v2_lineup(specs_v2);
    keryx_miner::slm::init_supported(specs_v2);
    info!("OPoI Phase-3 — {} uncensored model(s) staged (legacy lineup dropped, post-fork).", specs_v2.len(),);
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
            let one: &'static [&'static keryx_miner::models::ModelSpec] = Box::leak(vec![ps].into_boxed_slice());
            // RETRY until the mining-tier model is actually on disk. The previous build logged one
            // "will retry" line and then the task ENDED — nothing retried, so a failed download left
            // the rig stuck at "preparing…" forever with the real reason scrolled off. Now every
            // failure is logged LOUD (with the specific staging error) and we re-attempt with backoff,
            // so a transient gateway/disk problem self-heals AND the error stays in view until it does.
            let mut round: u32 = 0;
            loop {
                match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(one)).await {
                    Ok(Ok(())) => {
                        keryx_miner::slm::clear_staging_error();
                        info!("Mining-tier model '{}' ready — PoM can build its possession index.", ps.dir_name);
                        break;
                    }
                    Ok(Err(e)) => {
                        round += 1;
                        let detail = keryx_miner::slm::last_staging_error().unwrap_or_else(|| e.to_string());
                        error!(
                            "MINING-TIER MODEL '{}' NOT READY (attempt {}): {} — mining is SUSPENDED until this \
                             is fixed. Retrying in 60s.",
                            ps.dir_name, round, detail
                        );
                    }
                    Err(e) => {
                        round += 1;
                        error!("Mining-tier model prefetch task crashed (attempt {}): {} — retrying in 60s.", round, e);
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
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
        let tier_idx = keryx_miner::models::pom_tier_index(&spec.model_id, lineup_daa).expect("pom_spec has a tier");
        let gpath = keryx_miner::slm::gguf_path_for(spec).to_string_lossy().into_owned();
        // PoM PASSTHROUGH live test (KERYX_POM_PASSTHROUGH): build the HOST possession index in the
        // background so kHeavyHash shares can carry a PomProof (daemon stores it pre-fork). Heavy
        // (~minutes) — backgrounded so mining starts immediately; shares before it is ready submit
        // without a proof. No GPU model needed (proof gen reads the host index).
        if keryx_miner::pom::passthrough_enabled() {
            warn!("PoM: PASSTHROUGH mode (KERYX_POM_PASSTHROUGH) — kHeavyHash shares will carry a PomProof for the pre-fork wire test.");
            let gpath_pt = gpath.clone();
            let model_id_pt = spec.model_id;
            std::thread::spawn(move || {
                info!("PoM passthrough: building host possession index for proof attachment…");
                match keryx_miner::pom::WeightIndex::build_from_gguf(&gpath_pt, model_id_pt) {
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
        // no Vulkan ICD, no AMD GPU, OOM), the model is withdrawn from OPoI.
        #[cfg(feature = "pom-opencl")]
        {
            let gpath_llama = gpath.clone();
            std::thread::spawn(move || {
                let ok_marker = std::path::Path::new(&gpath_llama).parent().map(|d| d.join(".ok"));
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1800);
                loop {
                    let ready =
                        std::path::Path::new(&gpath_llama).exists() && ok_marker.as_ref().map_or(true, |m| m.exists());
                    if ready {
                        break;
                    }
                    if std::time::Instant::now() > deadline {
                        warn!("PoM(AMD): model GGUF not ready in 30 min — OPoI inference remains unavailable.");
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
                // Prefer the IN-PROCESS Vulkan engine (zero-dup: it also hosts the model the PoM
                // walk gathers over, so the inference card holds ONE resident copy). No
                // libkeryx-llama-vk.so → the llama-server subprocess; neither → unavailable.
                // Device selection lives INSIDE ensure_loaded/try_start (issue #18: pin to a
                // discrete GPU, KERYX_LLAMA_VK_DEVICE overrides) — the `0` here is only the
                // last-resort ggml index when no discrete Vulkan device is found at all.
                // UP-FRONT plan (field request): compute from VRAM math whether model + possession
                // blob (≈ gguf + gguf + 1.5 GiB KV/ctx margin) can share one card. If NOT, announce
                // the plan NOW instead of load→fail→fallback churn: the inference card will be
                // DEDICATED — it hosts the model and answers OPoI (the 8x-reward path, and fast GPU
                // answers avoid challenge timeouts), and it simply does not mine; the other cards
                // mine at full rate. The dedication itself is enforced in pom_opencl (no blob is
                // installed on the engine's card when both can't fit).
                let gguf_mb = std::fs::metadata(&gpath_llama).map(|m| m.len() / (1024 * 1024)).unwrap_or(6144);
                let need_mb = gguf_mb * 2 + 1536;
                let vram_mb = keryx_miner::pom_opencl::max_gpu_global_mem_mb().unwrap_or(0);
                if vram_mb > 0 && vram_mb < need_mb {
                    info!(
                        "PoM(AMD): model ({gguf_mb} MiB) + possession blob need ~{need_mb} MiB on one                          card; largest card has {vram_mb} MiB — the inference card will be DEDICATED to                          OPoI (it will not mine); all other cards mine at full rate."
                    );
                    keryx_miner::pom_opencl::require_dedicated_inference_card();
                }
                let inf_gpu = keryx_miner::slm::inference_gpu_for_model(&spec.model_id);
                keryx_miner::slm::warm_inference_route(spec.model_id, inf_gpu);
            });
        }
        // Apple Silicon (Metal) — llama.cpp Metal inference via the in-process engine (Phase 3b of
        // candle-independence): if `libkeryx-llama.dylib` sits next to the miner (or
        // `KERYX_LLAMA_SO` points at one), bring it up once the GGUF is on disk — slm then prefers
        // it as the required GPU route. No .dylib means the inference capability remains withdrawn
        // and the OPoI mining gate stays closed (unless the deprecated CPU override is explicit).
        // No llama-server subprocess fallback here: Metal does not ship one and llama_vulkan is
        // gated to pom-cuda/pom-opencl only.
        #[cfg(all(target_os = "macos", feature = "pom-metal"))]
        {
            let gpath_llama = gpath.clone();
            std::thread::spawn(move || {
                let ok_marker = std::path::Path::new(&gpath_llama).parent().map(|d| d.join(".ok"));
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1800);
                loop {
                    let ready =
                        std::path::Path::new(&gpath_llama).exists() && ok_marker.as_ref().map_or(true, |m| m.exists());
                    if ready {
                        break;
                    }
                    if std::time::Instant::now() > deadline {
                        warn!("PoM(Metal): model GGUF not ready in 30 min — GPU inference unavailable; OPoI mining remains gated.");
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
                let inf_gpu = keryx_miner::slm::inference_gpu_for_model(&spec.model_id);
                keryx_miner::slm::warm_inference_route(spec.model_id, inf_gpu);
            });
        }
        // CUDA route warmers are launched below, after every per-device model assignment is
        // published. Starting one here raced the mixed-rig map and could prove each tier on the
        // same largest card instead of the card that actually owns it.
        #[cfg(feature = "pom-opencl")]
        keryx_miner::pom_opencl::set_mining_tier(spec.model_id, gpath, tier_idx);
        #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
        {
            // Force the single-device split loader so the mining tier exposes its quant tensors
            // for zero-dup VRAM sharing with inference (avoids a 2nd full copy on the big tiers).
            keryx_miner::slm::set_pom_force_split(true);
            // Global DEFAULT model = the primary (biggest) tier — the inference card and any card
            // without a per-device override use it. On a UNIFORM rig no overrides are set below, so
            // this is byte-identical to the previous single-model path.
            keryx_miner::pom_gpu::set_mining_tier(spec.model_id, gpath);
            // Record which cards are --force-model-pinned so the OOM auto-demotion never overrides an
            // explicit forced choice (forced = load exactly this, no VRAM check).
            for (dev, _t, is_forced) in &device_tiers {
                if *is_forced {
                    keryx_miner::pom_gpu::set_device_forced(*dev);
                }
            }
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
            let mut route_specs = vec![spec];
            route_specs.extend(extra.iter().copied());
            if !extra.is_empty() {
                info!(
                    "Per-card model selection ACTIVE: {} distinct smaller model(s) for lighter cards \
                     (primary = {} on the biggest card) — prefetching.",
                    extra.len(),
                    spec.dir_name,
                );
                let leaked: &'static [&'static keryx_miner::models::ModelSpec] = Box::leak(extra.into_boxed_slice());
                tokio::spawn(async move {
                    let mut round = 0u32;
                    loop {
                        match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(leaked)).await {
                            Ok(Ok(())) => {
                                info!("Every per-card CUDA model is staged; all assigned tier routes can be proved.");
                                break;
                            }
                            Ok(Err(e)) => {
                                round += 1;
                                warn!("Per-card model staging failed (attempt {}): {} — retrying in 60s", round, e);
                            }
                            Err(e) => {
                                round += 1;
                                warn!(
                                    "Per-card model staging task crashed (attempt {}): {} — retrying in 60s",
                                    round, e
                                );
                            }
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    }
                });
            }

            // Prove one real serving route for every distinct tier assigned on a mixed CUDA rig,
            // including the primary. Each lightweight coordinator waits for its own completed GGUF
            // indefinitely and then stays alive so a later context reset can invalidate/re-prove the
            // exact `(model, GPU)` route. This is intentionally launched only after set_mining_tier
            // and every set_device_model call above, making model-aware placement deterministic.
            let route_models: Vec<([u8; 32], String)> = route_specs
                .into_iter()
                .map(|route_spec| {
                    (route_spec.model_id, keryx_miner::slm::gguf_path_for(route_spec).to_string_lossy().into_owned())
                })
                .collect();
            if opt.low_ram {
                // A single durable coordinator serializes the real generation probes. Successful
                // entries are cached, so later rounds are cheap health checks; an invalidated route
                // alone reloads/re-proves. Failed or not-yet-downloaded tiers cannot starve the rest.
                std::thread::spawn(move || loop {
                    for (model_id, route_path) in &route_models {
                        let ok_marker = std::path::Path::new(route_path).parent().map(|dir| dir.join(".ok"));
                        let ready = std::path::Path::new(route_path).exists()
                            && ok_marker.as_ref().map_or(true, |marker| marker.exists());
                        if !ready {
                            continue;
                        }
                        let inf_gpu = keryx_miner::slm::inference_gpu_for_model(model_id);
                        if !keryx_miner::slm::run_inference_self_test(model_id, inf_gpu) {
                            warn!(
                                "OPoI low-RAM route proof for model {:.8} on GPU {} is not ready; retrying",
                                hex::encode(model_id),
                                inf_gpu
                            );
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_secs(30));
                });
            } else {
                for (model_id, route_path) in route_models {
                    std::thread::spawn(move || {
                        let ok_marker = std::path::Path::new(&route_path).parent().map(|dir| dir.join(".ok"));
                        loop {
                            let ready = std::path::Path::new(&route_path).exists()
                                && ok_marker.as_ref().map_or(true, |marker| marker.exists());
                            if ready {
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_secs(5));
                        }
                        let inf_gpu = keryx_miner::slm::inference_gpu_for_model(&model_id);
                        keryx_miner::slm::warm_inference_route(model_id, inf_gpu);
                    });
                }
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
            tier_idx,
            spec.dir_name,
            keryx_miner::pom::POM_ACTIVATION_DAA
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

    // --wait-ready: hold mining + the OPoI declaration until every card's walk is installed
    // (see wait_ready.rs for the why — challenge starvation during multi-card bring-up on
    // low-RAM rigs). Backend-neutral state; workers register + installs mark ready.
    if opt.wait_ready {
        keryx_miner::wait_ready::enable();
        info!("--wait-ready: mining and OPoI declaration held until ALL cards are fully set up.");
    }

    // --intensity: fixed batch per card (batch = 2^intensity), CSV position = CUDA ordinal, the same
    // mapping --force-model uses. A listed card is not autotuned.
    #[cfg(all(feature = "pom-cuda", not(all(target_os = "macos", feature = "pom-metal"))))]
    if let Some(raw) = opt.intensity.as_deref() {
        let given: Vec<Option<u32>> = raw
            .split(',')
            .map(|s| {
                let s = s.trim();
                if s.is_empty() {
                    return None; // an empty slot leaves that card on autotune
                }
                s.parse::<u32>().ok()
            })
            .collect();
        // Positions follow THIS PROCESS's card list: with `--cuda-device 2,3`, the first intensity is
        // for GPU2. Without that flag the process mines every card and position == CUDA ordinal, so
        // the common case is unchanged. (`--force-model` indexes by raw ordinal; matching it here
        // would mean `--cuda-device 2 --intensity 17` silently applied to GPU0 and left GPU2 on auto,
        // which is what happened the first time this was tested.)
        let own = keryx_miner::slm::cli_cuda_devices();
        let width = own.iter().copied().max().map(|m| m + 1).unwrap_or(given.len()).max(given.len());
        let mut map: Vec<Option<u32>> = vec![None; width];
        for (pos, val) in given.iter().enumerate() {
            let dev = if own.is_empty() {
                pos
            } else {
                match own.get(pos) {
                    Some(d) => *d,
                    None => continue,
                }
            };
            if dev < map.len() {
                map[dev] = *val;
            }
        }
        let shown: Vec<String> = map
            .iter()
            .enumerate()
            .map(|(dev, i)| match i {
                Some(i) => {
                    let c = (*i).clamp(keryx_miner::pom_gpu::INTENSITY_MIN, keryx_miner::pom_gpu::INTENSITY_MAX);
                    let note = if c != *i { format!(" (clamped from {})", i) } else { String::new() };
                    format!("GPU{}={}{} → {} nonces", dev, c, note, 1u64 << c)
                }
                None => format!("GPU{}=auto", dev),
            })
            .collect();
        info!("--intensity {}: {}", raw.trim(), shown.join(", "));
        keryx_miner::pom_gpu::set_intensity_map(map);
    }

    // --only-inference: trickle-mine, and hand the card to inference the moment a request lands.
    #[cfg(all(feature = "pom-cuda", not(all(target_os = "macos", feature = "pom-metal"))))]
    if opt.only_inference {
        keryx_miner::pom_gpu::set_only_inference(true);
        info!(
            "--only-inference: mining at the minimum batch with a {} ms idle duty cycle; the walk stops \
             while a request is served so it is answered at full speed. Expect very low hashrate — that \
             is the point — and note the no-share wedge supervisor is disabled in this mode.",
            keryx_miner::pom_gpu::only_inference_duty_ms()
        );
    }

    if cfg!(feature = "pom-opencl") {
        info!("AMD/OpenCL: OPoI inference uses the GPU llama.cpp Vulkan engine; skipping the CUDA/cuBLAS probe.");
    } else if cfg!(all(target_os = "macos", feature = "pom-metal")) {
        // Apple Silicon inference is hosted by the in-process llama.cpp Metal engine.
        info!("Apple Silicon: OPoI inference targets the llama.cpp Metal GPU engine. Skipping the CUDA/cuBLAS probe.");
    } else {
        info!("Probing GPU inference (cuBLAS) before mining…");
        match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
            Ok(keryx_miner::slm::GpuProbe::Ok) => info!("GPU inference verified — cuBLAS loaded successfully."),
            Ok(keryx_miner::slm::GpuProbe::NoCuda) => {
                warn!(
                    "No CUDA device detected — H6 inference is unavailable and models will not be declared serveable."
                );
            }
            Ok(keryx_miner::slm::GpuProbe::CublasMissing) => {
                warn!(
                    "CUDA GPU detected but a CUDA runtime lib is missing — installing them automatically (one-time)…"
                );
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
        // This is the AMD (OpenCL) build and no AMD GPU compute device was found. The commonest
        // cause is running the `-amd` package on an NVIDIA (or CPU-only) box — the "amd" refers to
        // AMD Radeon GPUs, not to an x64/AMD64 CPU. Point the user at the right package.
        #[cfg(feature = "pom-opencl")]
        error!(
            "No AMD GPU found — this is the AMD build (keryx-miner-supr-amd), which mines on AMD \
             Radeon GPUs via OpenCL. If your GPUs are NVIDIA, use the NVIDIA package instead: \
             'modern' for RTX 30xx/40xx/50xx (Ampere/Ada/Blackwell) or 'legacy' for GTX 10xx/16xx \
             and RTX 20xx (Turing/Volta). The '-amd' package is NOT for NVIDIA or CPU-only systems."
        );
        #[cfg(not(feature = "pom-opencl"))]
        error!("No workers specified");
        // Hard-exit instead of returning Err: on the AMD build the in-process llama inference engine
        // (dlopen'd libkeryx-llama-vk.so) may already be loading the model on a background thread,
        // and unwinding out of main() races its teardown into a segfault (field report on an NVIDIA
        // box). exit() tears the process down cleanly without running that destructor.
        std::process::exit(1);
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
        let only_inference_mode = opt.only_inference;
        let stall_secs: u64 = std::env::var("KERYX_STALL_RESTART_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 60)
            .unwrap_or(600);
        // --only-inference trickle-mines on purpose, so long gaps between accepted shares are the
        // expected steady state, not a wedge. Left armed it would restart a perfectly healthy rig
        // every 10 minutes and interrupt the serving it exists to do.
        let stall_armed = !only_inference_mode;
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
                } else if armed && stall_armed && last_change.elapsed() >= Duration::from_secs(stall_secs) {
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
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 30)
            .unwrap_or(90);
        let grace: u64 = std::env::var("KERYX_FAILBACK_GRACE_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 10)
            .unwrap_or(60);
        let probe_secs: u64 = std::env::var("KERYX_FAILBACK_PROBE_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 10)
            .unwrap_or(30);
        info!(
            "Pool failover ENABLED — primary + {} backup(s): {:?}. Failover after {}s no-job; failback grace {}s.",
            pools.len() - 1,
            pools,
            failover_after,
            grace
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
                            warn!(
                                "FAILOVER: pool[{}] ({}) delivered no jobs for {}s → switching to pool[{}] ({}).",
                                cur, pools_dbg[cur], failover_after, next, pools_dbg[next]
                            );
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
                            warn!(
                                "FAILBACK: higher-priority pool[{}] ({}) is reachable — switching back up.",
                                i, pools[i]
                            );
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
        match client_main(
            &opt,
            pool_address,
            block_template_ctr.clone(),
            &plugin_manager,
            escrow_privkey.clone(),
            escrow_cert.clone(),
        )
        .await
        {
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
    matches!(tokio::time::timeout(timeout, tokio::net::TcpStream::connect(addr)).await, Ok(Ok(_)))
}
