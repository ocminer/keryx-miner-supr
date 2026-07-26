use clap::Parser;
use log::LevelFilter;

use crate::Error;

#[derive(Parser, Debug)]
#[clap(name = "keryx-miner", version, about = "A Keryx high performance GPU miner with OPoI inference\n\nModel tiers (default: AUTO — largest PoM tier that fits this GPU's VRAM = highest tier reward):\n  (default)    auto — largest tier that fits VRAM & is downloaded (8GB->Gemma, 24GB->Qwen3-32B, 32GB->Llama-70B)\n  --light      Gemma-3-4B — any GPU (6GB+)\n  --high       Qwen3-32B — RTX 3090 / 4090 (24GB+)\n  --very-high  Llama-3.3-70B — RTX 5090 / 48GB+\n  --tier <t>   auto | light | default | high | very-high (pins the tier)", term_width = 0)]
pub struct Opt {
    // ── OPoI / Inference ─────────────────────────────────────────────────────

    #[clap(
        long = "very-light",
        help = "Model tier: Qwen3-8B-abliterated (Q4_K_S) — 6 GB+ GPU, smallest tier (PoM tier 0, post-H5; EXAONE-4.0-1.2B in the H4→H5 window)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["light", "high", "very_high", "tier"]
    )]
    pub very_light: bool,

    #[clap(
        long = "light",
        help = "Model tier: Gemma-3-4B — any GPU (6 GB+ VRAM)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["very_light", "high", "very_high"]
    )]
    pub light: bool,

    #[clap(
        long = "high",
        help = "Model tier: Qwen3-32B — RTX 3090 / 4090 (24 GB+ VRAM) + ~60 GB free disk (model + possession tree)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["very_light", "light", "very_high"]
    )]
    pub high: bool,

    #[clap(
        long = "very-high",
        help = "Model tier: Llama-3.3-70B (Q2_K_L post-H2) — RTX 5090 / 32 GB+ VRAM + ~85 GB free disk (model + possession tree)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["very_light", "light", "high", "tier"]
    )]
    pub very_high: bool,

    #[clap(
        long = "tier",
        value_name = "TIER",
        help = "Model tier: auto | very-light | light | default | high | very-high. DEFAULT (no flag) is 'auto' — picks the LARGEST tier that fits the GPU's VRAM (per-process) AND is downloaded = highest tier reward. Pass a name to pin a tier; overrides --very-light/--light/--high/--very-high.",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["very_light", "light", "high", "very_high"]
    )]
    pub tier: Option<String>,

    #[clap(
        long = "force-model",
        value_name = "MODELS",
        help = "PER-DEVICE model override, comma-separated, mapped to CUDA device order: \
                --force-model light,very-light,high → GPU0=light, GPU1=very-light, GPU2=high. \
                FORCES the model regardless of VRAM fit (a card too small OOMs — power-user knob) and \
                bypasses the download-availability check. Fewer entries than GPUs → the remaining cards \
                use normal per-card AUTO best-fit. Overrides --tier / --light / etc. Names: \
                very-light | light | default | high | very-high. \
                AMD/OpenCL: PROCESS-WIDE (one resident model per process) — the FIRST entry applies to \
                ALL cards.",
        help_heading = "OPoI / Inference"
    )]
    pub force_model: Option<String>,

    #[clap(
        long = "cpu-inference",
        help = "Run OPoI inference on the CPU instead of the GPU — frees the GPU for hashing and avoids weak-fp16 GPUs (e.g. GTX 1060). Pairs well with --light.",
        help_heading = "OPoI / Inference"
    )]
    pub cpu_inference: bool,

    #[clap(
        long = "no-shared-inference",
        help = "Bind OPoI inference to THIS process's own --cuda-device GPU instead of the globally-biggest card. Use when running one process per GPU (separate wallets/workers): it stops every process piling inference onto one shared card. Auto-on for a single-GPU process.",
        help_heading = "OPoI / Inference"
    )]
    pub no_shared_inference: bool,

    #[clap(
        long = "ipfs-url",
        help = "IPFS Kubo API URL for uploading inference results",
        help_heading = "OPoI / Inference",
        default_value = "http://127.0.0.1:5001"
    )]
    pub ipfs_url: String,

    #[clap(
        long = "escrow-key-file",
        help = "Path to the OPoI escrow private key file (auto-generated if absent)",
        help_heading = "OPoI / Inference",
        default_value = "escrow.key"
    )]
    pub escrow_key_file: String,

    #[clap(
        long = "escrow-state-file",
        help = "Path to the escrow claim state file",
        help_heading = "OPoI / Inference",
        default_value = "escrow_state.json"
    )]
    pub escrow_state_file: String,

    #[clap(
        long = "recover-escrow",
        help = "Rebuild escrow_state.json by querying the Keryx public API. Exits after recovery.",
        help_heading = "OPoI / Inference"
    )]
    pub recover_escrow: bool,

    #[clap(
        long = "recover-escrow-api",
        help = "Base URL of the Keryx API to use for escrow recovery",
        help_heading = "OPoI / Inference",
        default_value = "https://keryx-labs.com"
    )]
    pub recover_escrow_api: String,

    // ── Mining ────────────────────────────────────────────────────────────────

    #[clap(short, long, help = "Enable debug logging level")]
    pub debug: bool,

    #[clap(short = 'a', long = "mining-address", help = "The Keryx address for the miner reward")]
    pub mining_address: Option<String>,

    #[clap(short = 's', long = "keryxd-address", default_value = "127.0.0.1", help = "The IP of the keryxd instance")]
    pub keryxd_address: String,

    #[clap(
        long = "backup-pool",
        value_name = "URL",
        help = "Backup/failover pool as a full URL (stratum+tcp://host:port). Repeatable — pass it multiple times to define a PRIORITY LIST tried in the order given. If the primary (-s) fails hard (no job for ~90s or the connection drops), the miner fails over to the next pool; in the background it keeps probing higher-priority pools and switches back as soon as one recovers. Uses the same wallet/password as -s. OFF entirely unless at least one is set."
    )]
    pub backup_pool: Vec<String>,

    #[clap(short, long, help = "Keryxd port [default: Mainnet = 22110, Testnet = 22211]")]
    port: Option<u16>,

    #[clap(
        long = "pool-password",
        default_value = "x",
        help = "Stratum pool password (sent in mining.authorize). Most keryx pools ignore it; on suprnova set e.g. 'd=16' for static difficulty 16. Do NOT use -p for this — -p is --port."
    )]
    pub pool_password: String,

    #[clap(long, help = "Use testnet instead of mainnet [default: false]")]
    testnet: bool,

    #[clap(short = 't', long = "threads", help = "Amount of CPU miner threads to launch [default: 0]")]
    pub num_threads: Option<u16>,

    #[clap(
        long = "api-bind",
        help = "Serve a JSON stats HTTP API at host:port (e.g. 127.0.0.1:4067). Endpoints: / or /stats (generic), /mmpos (mmpOS format). For HiveOS/mmpos/dashboards. Disabled if unset."
    )]
    pub api_bind: Option<String>,

    #[clap(
        long = "disable-gpu",
        help = "Disable all GPU workers and mine on CPU only, even if GPUs are present [default: false]",
        long_help = "Skip loading the GPU worker plugins entirely and mine on CPU only, even if GPUs are present in the system. \
If --threads is not given, it defaults to the number of physical CPU cores. Note: KeryxHash on CPU is orders of magnitude slower than on a GPU."
    )]
    pub disable_gpu: bool,

    #[clap(
        long = "mine-when-not-synced",
        help = "Mine even when keryxd says it is not synced",
        long_help = "Mine even when keryxd says it is not synced, only useful when passing `--allow-submit-block-when-not-synced` to keryxd  [default: false]"
    )]
    pub mine_when_not_synced: bool,

    #[clap(
        long = "model-dir",
        value_name = "DIR",
        help = "Directory to look for AND download the OPoI models (default: 'models' next to the miner binary)",
        long_help = "Directory holding the OPoI model folders (<DIR>/<Model-Name>/model.gguf). The miner LOOKS for \
models here and also SAVES downloads here — point it at a shared/network directory to reuse one model store across \
rigs (e.g. --model-dir /mnt/share/keryx-models). Created if missing. If the directory is not writable (read-only \
share), pre-staged models still load but downloads of missing models will fail. \
Default: the 'models' folder next to the miner binary."
    )]
    pub model_dir: Option<String>,

    #[clap(
        long = "hiveos",
        help = "HiveOS mode: keep models in /hive/miners/custom/models so they survive miner upgrades",
        long_help = "HiveOS mode. Defaults the model directory to /hive/miners/custom/models — OUTSIDE the \
versioned miner directory — so the 6-28 GB models survive miner upgrades (HiveOS extracts each new version \
into a fresh folder, wiping a models/ kept next to the binary). The bundled h-run.sh passes this flag \
automatically. An explicit --model-dir overrides it. If /hive/miners/custom/models cannot be created/used, \
the miner warns and falls back to the default 'models' folder next to the binary (mining continues)."
    )]
    pub hiveos: bool,
}

/// The `--hiveos` default model store — outside the versioned miner dir, survives upgrades.
const HIVEOS_MODEL_DIR: &str = "/hive/miners/custom/models";

/// Validate a model-dir path and return the canonical directory: expand a leading `~/`, create the
/// directory if missing, reject non-directories, canonicalize. Warns (does not fail) when the dir is
/// not writable — pre-staged models still load there; only downloads of missing models would fail.
fn resolve_model_dir(raw: &str) -> Result<std::path::PathBuf, Error> {
    let expanded = if let Some(rest) = raw.strip_prefix("~/") {
        std::env::var("HOME")
            .map(|h| std::path::PathBuf::from(h).join(rest))
            .map_err(|_| Error::from(format!("model dir '{}': cannot expand '~' (HOME not set)", raw)))?
    } else {
        std::path::PathBuf::from(raw)
    };
    if !expanded.exists() {
        std::fs::create_dir_all(&expanded)
            .map_err(|e| Error::from(format!("model dir '{}': cannot create directory: {}", raw, e)))?;
    }
    if !expanded.is_dir() {
        return Err(format!("model dir '{}': exists but is not a directory", raw).into());
    }
    let dir = expanded
        .canonicalize()
        .map_err(|e| Error::from(format!("model dir '{}': cannot resolve path: {}", raw, e)))?;
    let probe = dir.join(".keryx-writetest");
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
        }
        Err(e) => log::warn!(
            "model dir '{}' is not writable ({}) — pre-staged models will load, but downloads of missing models will FAIL. \
             Fix permissions if the miner needs to download.",
            dir.display(),
            e
        ),
    }
    Ok(dir)
}

impl Opt {
    pub fn process(&mut self) -> Result<(), Error> {
        if self.recover_escrow {
            return Ok(());
        }
        if self.mining_address.is_none() {
            return Err("--mining-address is required".into());
        }
        // Model-dir resolution (BEFORE any model staging/prefetch runs). Precedence:
        //   1. explicit --model-dir — hard error on an unusable path (the user asked for it);
        //   2. --hiveos → /hive/miners/custom/models (survives miner upgrades) — falls back to
        //      the default <exe>/models with a WARN if /hive is unusable (packaging default:
        //      mining must not die on a box without the HiveOS layout).
        if let Some(raw) = &self.model_dir {
            let dir = resolve_model_dir(raw)?;
            log::info!("model dir: {} (via --model-dir)", dir.display());
            keryx_miner::slm::set_model_dir(dir);
        } else if self.hiveos {
            match resolve_model_dir(HIVEOS_MODEL_DIR) {
                Ok(dir) => {
                    log::info!("model dir: {} (via --hiveos — models survive miner upgrades)", dir.display());
                    keryx_miner::slm::set_model_dir(dir);
                }
                Err(e) => log::warn!(
                    "--hiveos: {} — falling back to the 'models' folder next to the miner binary \
                     (models will NOT survive miner upgrades).",
                    e
                ),
            }
        }
        if self.keryxd_address.is_empty() {
            self.keryxd_address = "127.0.0.1".to_string();
        }

        if !self.keryxd_address.contains("://") {
            let port_str = self.port().to_string();
            let (keryxd, port) = match self.keryxd_address.contains(':') {
                true => self.keryxd_address.split_once(':').expect("We checked for `:`"),
                false => (self.keryxd_address.as_str(), port_str.as_str()),
            };
            self.keryxd_address = format!("grpc://{}:{}", keryxd, port);
        }
        // Name the mode explicitly: `-s` doubles as pool URL and node address, and "keryxd address:
        // stratum+tcp://…" read as a bug to users. This line is the first thing to check when a
        // solo miner cannot reach its node (see README "Solo mining").
        if self.keryxd_address.starts_with("stratum+tcp://") {
            log::info!("pool address: {} (stratum)", self.keryxd_address);
        } else {
            log::info!("keryxd address: {} (solo — mining direct to node)", self.keryxd_address);
        }

        if self.num_threads.is_none() {
            self.num_threads = Some(0);
        }

        Ok(())
    }

    fn port(&mut self) -> u16 {
        *self.port.get_or_insert(if self.testnet { 22211 } else { 22110 })
    }

    pub fn log_level(&self) -> LevelFilter {
        if self.debug {
            LevelFilter::Debug
        } else {
            LevelFilter::Info
        }
    }
}
