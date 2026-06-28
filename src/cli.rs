use clap::Parser;
use log::LevelFilter;

use crate::Error;

#[derive(Parser, Debug)]
#[clap(name = "keryx-miner", version, about = "A Keryx high performance GPU miner with OPoI inference\n\nModel tiers (default: TinyLlama + DeepSeek-8B — RTX 3060 12GB / 3070 / 3080):\n  --light      TinyLlama only — RTX 3060 6GB or any GPU\n  (default)    TinyLlama + DeepSeek-R1-8B — RTX 3060 12GB / 3070 / 3080\n  --high       + DeepSeek-R1-32B — RTX 3090 / 4090 (24GB+)\n  --very-high  + LLaMA-3.3-70B  — RTX 5090 (32GB+)", term_width = 0)]
pub struct Opt {
    // ── OPoI / Inference ─────────────────────────────────────────────────────

    #[clap(
        long = "light",
        help = "Model tier: TinyLlama only — any GPU (6GB+ VRAM)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["high", "very_high"]
    )]
    pub light: bool,

    #[clap(
        long = "high",
        help = "Model tier: TinyLlama + DeepSeek-R1-8B + DeepSeek-R1-32B — RTX 3090 / 4090 (24GB+)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["light", "very_high"]
    )]
    pub high: bool,

    #[clap(
        long = "very-high",
        help = "Model tier: TinyLlama + DeepSeek-R1-8B + DeepSeek-R1-32B + LLaMA-3.3-70B — RTX 5090 (32GB+)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["light", "high", "tier"]
    )]
    pub very_high: bool,

    #[clap(
        long = "tier",
        value_name = "TIER",
        help = "Model tier: auto | light | default | high | very-high. 'auto' picks the LARGEST tier that fits the GPU's VRAM (per-process: each GPU's own VRAM). Overrides --light/--high/--very-high.",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["light", "high", "very_high"]
    )]
    pub tier: Option<String>,

    #[clap(
        long = "cpu-inference",
        help = "Run OPoI inference on the CPU instead of the GPU — frees the GPU for hashing and avoids weak-fp16 GPUs (e.g. GTX 1060). Pairs well with --light.",
        help_heading = "OPoI / Inference"
    )]
    pub cpu_inference: bool,

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
}

impl Opt {
    pub fn process(&mut self) -> Result<(), Error> {
        if self.recover_escrow {
            return Ok(());
        }
        if self.mining_address.is_none() {
            return Err("--mining-address is required".into());
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
        log::info!("keryxd address: {}", self.keryxd_address);

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
