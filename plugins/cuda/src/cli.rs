use crate::Error;
use std::str::FromStr;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NonceGenEnum {
    Lean,
    Xoshiro,
}

impl FromStr for NonceGenEnum {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lean" => Ok(Self::Lean),
            "xoshiro" => Ok(Self::Xoshiro),
            _ => Err("Unknown string".into()),
        }
    }
}

#[cfg(feature = "overclock")]
#[derive(clap::Args, Debug, Default)]
pub struct OverClock {
    #[clap(long = "cuda-lock-mem-clocks", visible_alias = "mem-lock", use_delimiter = true, help = "Lock mem clocks (MHz) per GPU, comma-separated, e.g. 9500,9500 (alias: --mem-lock)")]
    pub cuda_lock_mem_clocks: Option<Vec<u32>>,
    #[clap(long = "cuda-lock-core-clocks", visible_alias = "gpu-clock", use_delimiter = true, help = "Lock core clocks (MHz) per GPU, comma-separated, e.g. 2400,2400 (alias: --gpu-clock)")]
    pub cuda_lock_core_clocks: Option<Vec<u32>>,
    #[clap(long = "cuda-power-limits", visible_alias = "power-limit", use_delimiter = true, help = "Lock power limits (W) per GPU, comma-separated, e.g. 450,450 (alias: --power-limit)")]
    pub cuda_power_limits: Option<Vec<u32>>,
    #[clap(long = "cuda-fan-speed", use_delimiter = true, help = "Lock fan speed (% 0-100) per GPU, comma-separated, e.g. 80,80. Requires NVIDIA manual-fan-control mode (nvidia-smi -i <gpu> -fcm 1, or X+coolbits) on consumer cards.")]
    pub cuda_fan_speed: Option<Vec<u32>>,
    #[clap(long = "cuda-monitor-interval", default_value = "30", help = "Seconds between periodic GPU monitor lines (temp, fan, power, clocks). 0 to disable.")]
    pub cuda_monitor_interval: u64,
}

#[derive(clap::Args, Debug)]
pub struct CudaOpt {
    #[clap(long = "cuda-device", use_delimiter = true, help = "Which CUDA GPUs to use [default: all]")]
    pub cuda_device: Option<Vec<u16>>,
    #[clap(long = "cuda-workload", help = "Ratio of nonces to GPU possible parrallel run [default: 64]")]
    pub cuda_workload: Option<Vec<f32>>,
    #[clap(
        long = "cuda-workload-absolute",
        help = "The values given by workload are not ratio, but absolute number of nonces [default: false]"
    )]
    pub cuda_workload_absolute: bool,
    #[clap(long = "cuda-disable", help = "Disable cuda workers")]
    pub cuda_disable: bool,
    #[clap(
        long = "cuda-no-blocking-sync",
        help = "Actively wait for result. Higher CPU usage, but less red blocks. Can have lower workload.",
        long_help = "Actively wait for GPU result. Increases CPU usage, but removes delays that might result in red blocks. Can have lower workload."
    )]
    pub cuda_no_blocking_sync: bool,
    #[clap(
        long = "cuda-nonce-gen",
        help = "The random method used to generate nonces. Options: (i) xoshiro - each thread in GPU will have its own random state, creating a (pseudo-)independent xoshiro sequence (ii) lean - each GPU will have a single random nonce, and each GPU thread will work on nonce + thread id.",
        default_value = "lean"
    )]
    pub cuda_nonce_gen: NonceGenEnum,

    #[cfg(feature = "overclock")]
    #[clap(flatten)]
    pub overclock: OverClock,
}
