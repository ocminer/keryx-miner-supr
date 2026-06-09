#[macro_use]
extern crate keryx_miner;

use clap::{ArgMatches, FromArgMatches};
use cust::prelude::*;
use keryx_miner::{Plugin, Worker, WorkerSpec};
use log::LevelFilter;
use std::error::Error as StdError;
#[cfg(feature = "overclock")]
use {
    log::{error, info, warn},
    nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor},
    nvml_wrapper::enums::device::GpuLockedClocksSetting,
    nvml_wrapper::Device as NvmlDevice,
    nvml_wrapper::Nvml,
    std::thread,
    std::time::Duration,
};

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

mod cli;
mod worker;

use crate::cli::{CudaOpt, NonceGenEnum};
use crate::worker::CudaGPUWorker;

const DEFAULT_WORKLOAD_SCALE: f32 = 1024.;

pub struct CudaPlugin {
    specs: Vec<CudaWorkerSpec>,
    #[cfg(feature = "overclock")]
    nvml_instance: Nvml,
    _enabled: bool,
}

impl CudaPlugin {
    fn new() -> Result<Self, Error> {
        cust::init(CudaFlags::empty())?;
        env_logger::builder().filter_level(LevelFilter::Info).parse_default_env().init();
        Ok(Self {
            specs: Vec::new(),
            _enabled: false,
            #[cfg(feature = "overclock")]
            nvml_instance: Nvml::init()?,
        })
    }
}

impl Plugin for CudaPlugin {
    fn name(&self) -> &'static str {
        "CUDA Worker"
    }

    fn enabled(&self) -> bool {
        self._enabled
    }

    fn get_worker_specs(&self) -> Vec<Box<dyn WorkerSpec>> {
        self.specs.iter().map(|spec| Box::new(*spec) as Box<dyn WorkerSpec>).collect::<Vec<Box<dyn WorkerSpec>>>()
    }

    //noinspection RsTypeCheck
    fn process_option(&mut self, matches: &ArgMatches) -> Result<usize, keryx_miner::Error> {
        let opts: CudaOpt = CudaOpt::from_arg_matches(matches)?;

        self._enabled = !opts.cuda_disable;
        if self._enabled {
            let gpus: Vec<u16> = match &opts.cuda_device {
                Some(devices) => devices.clone(),
                None => {
                    let gpu_count = Device::num_devices().unwrap() as u16;
                    (0..gpu_count).collect()
                }
            };

            // if any of cuda_lock_core_clocks / cuda_lock_mem_clocks / cuda_power_limit is valid, init nvml and try to apply
            #[cfg(feature = "overclock")]
            if opts.overclock.cuda_lock_core_clocks.is_some()
                || opts.overclock.cuda_lock_mem_clocks.is_some()
                || opts.overclock.cuda_power_limits.is_some()
            {
                for i in 0..gpus.len() {
                    let lock_mem_clock: Option<u32> = match &opts.overclock.cuda_lock_mem_clocks {
                        Some(mem_clocks) if i < mem_clocks.len() => Some(mem_clocks[i]),
                        Some(mem_clocks) if !mem_clocks.is_empty() => Some(*mem_clocks.last().unwrap()),
                        _ => None,
                    };

                    let lock_core_clock: Option<u32> = match &opts.overclock.cuda_lock_core_clocks {
                        Some(core_clocks) if i < core_clocks.len() => Some(core_clocks[i]),
                        Some(core_clocks) if !core_clocks.is_empty() => Some(*core_clocks.last().unwrap()),
                        _ => None,
                    };

                    let power_limit: Option<u32> = match &opts.overclock.cuda_power_limits {
                        Some(power_limits) if i < power_limits.len() => Some(power_limits[i]),
                        Some(power_limits) if !power_limits.is_empty() => Some(*power_limits.last().unwrap()),
                        _ => None,
                    };

                    let mut nvml_device: NvmlDevice = self.nvml_instance.device_by_index(gpus[i] as u32)?;

                    if let Some(lmc) = lock_mem_clock {
                        match nvml_device.set_mem_locked_clocks(lmc, lmc) {
                            Err(e) => error!("set mem locked clocks {:?}", e),
                            _ => info!("GPU #{} #{} lock mem clock at {} Mhz", i, &nvml_device.name()?, &lmc),
                        };
                    }

                    if let Some(lcc) = lock_core_clock {
                        match nvml_device.set_gpu_locked_clocks(GpuLockedClocksSetting::Numeric {
                            min_clock_mhz: lcc,
                            max_clock_mhz: lcc,
                        }) {
                            Err(e) => error!("set gpu locked clocks {:?}", e),
                            _ => info!("GPU #{} #{} lock core clock at {} Mhz", i, &nvml_device.name()?, &lcc),
                        };
                    };

                    if let Some(pl) = power_limit {
                        match nvml_device.set_power_management_limit(pl * 1000) {
                            Err(e) => error!("set power limit {:?}", e),
                            _ => info!("GPU #{} #{} power limit at {} W", i, &nvml_device.name()?, &pl),
                        };
                    };
                }
            }

            // Fan speed control. nvml's set_fan_speed requires the GPU to be
            // in manual fan-control mode first; on consumer cards under a
            // headless driver that means root + `nvidia-smi -i <id> -fcm 1`
            // or the X-coolbits route. We try-and-warn-on-failure so a
            // non-root operator can still pass --cuda-fan-speed and see why
            // it didn't stick.
            #[cfg(feature = "overclock")]
            if let Some(ref fans) = opts.overclock.cuda_fan_speed {
                for i in 0..gpus.len() {
                    let pct: u32 = match fans.get(i) {
                        Some(p) => *p,
                        None => *fans.last().unwrap_or(&0),
                    };
                    let pct = pct.min(100);
                    let mut nvml_device: NvmlDevice = self.nvml_instance.device_by_index(gpus[i] as u32)?;
                    let n_fans = nvml_device.num_fans().unwrap_or(1);
                    let name = nvml_device.name().unwrap_or_else(|_| "GPU".into());
                    for f in 0..n_fans {
                        match nvml_device.set_fan_speed(f, pct) {
                            Ok(()) => info!("GPU #{} #{} fan {} → {}%", i, name, f, pct),
                            Err(e) => warn!("GPU #{} #{} fan {}: set_fan_speed({}%) failed: {:?} (need manual fan-control mode + permissions)", i, name, f, pct, e),
                        }
                    }
                }
            }

            // Periodic monitor thread — logs temp / fan / power / clocks
            // every `cuda_monitor_interval` seconds. The thread is detached;
            // it dies with the process. Disabled if interval == 0.
            #[cfg(feature = "overclock")]
            if opts.overclock.cuda_monitor_interval > 0 {
                let gpus_for_monitor = gpus.clone();
                let interval = Duration::from_secs(opts.overclock.cuda_monitor_interval);
                thread::Builder::new()
                    .name("keryxcuda-monitor".into())
                    .spawn(move || {
                        // Each monitor thread builds its own NVML handle —
                        // sharing across threads via Arc<Mutex<Nvml>> would
                        // serialise everything for no benefit (NVML calls
                        // are already cheap). Init failure is logged once.
                        let nvml = match Nvml::init() {
                            Ok(n) => n,
                            Err(e) => { warn!("keryxcuda-monitor: NVML init failed: {:?} — monitor disabled", e); return; }
                        };
                        loop {
                            for (idx, &gpu_id) in gpus_for_monitor.iter().enumerate() {
                                if let Ok(dev) = nvml.device_by_index(gpu_id as u32) {
                                    let temp = dev.temperature(TemperatureSensor::Gpu).ok();
                                    let n_fans = dev.num_fans().unwrap_or(0);
                                    let fan_pct: Vec<String> = (0..n_fans)
                                        .filter_map(|f| dev.fan_speed(f).ok().map(|p| format!("{}%", p)))
                                        .collect();
                                    let power_w = dev.power_usage().ok().map(|mw| mw as f32 / 1000.0);
                                    let core_mhz = dev.clock_info(Clock::Graphics).ok();
                                    let mem_mhz = dev.clock_info(Clock::Memory).ok();
                                    let mem_used = dev.memory_info().ok();
                                    info!(
                                        "[GPU #{}] temp={}°C fan={} power={} core={} MHz mem={} MHz vram={}",
                                        idx,
                                        temp.map(|t| t.to_string()).unwrap_or_else(|| "?".into()),
                                        if fan_pct.is_empty() { "?".into() } else { fan_pct.join(",") },
                                        power_w.map(|w| format!("{:.1}W", w)).unwrap_or_else(|| "?".into()),
                                        core_mhz.map(|c| c.to_string()).unwrap_or_else(|| "?".into()),
                                        mem_mhz.map(|c| c.to_string()).unwrap_or_else(|| "?".into()),
                                        mem_used
                                            .map(|m| format!("{}/{}MB", m.used / (1024 * 1024), m.total / (1024 * 1024)))
                                            .unwrap_or_else(|| "?".into()),
                                    );
                                }
                            }
                            thread::sleep(interval);
                        }
                    })
                    .ok();
            }

            self.specs = (0..gpus.len())
                .map(|i| CudaWorkerSpec {
                    device_id: gpus[i] as u32,
                    workload: match &opts.cuda_workload {
                        Some(workload) if i < workload.len() => workload[i],
                        Some(workload) if !workload.is_empty() => *workload.last().unwrap(),
                        _ => DEFAULT_WORKLOAD_SCALE,
                    },
                    is_absolute: opts.cuda_workload_absolute,
                    blocking_sync: !opts.cuda_no_blocking_sync,
                    random: opts.cuda_nonce_gen,
                })
                .collect();
        }
        Ok(self.specs.len())
    }
}

#[derive(Copy, Clone)]
struct CudaWorkerSpec {
    device_id: u32,
    workload: f32,
    is_absolute: bool,
    blocking_sync: bool,
    random: NonceGenEnum,
}

impl WorkerSpec for CudaWorkerSpec {
    fn id(&self) -> String {
        let device = Device::get_device(self.device_id).unwrap();
        format!("#{} ({})", self.device_id, device.name().unwrap())
    }

    fn build(&self) -> Box<dyn Worker> {
        Box::new(
            CudaGPUWorker::new(self.device_id, self.workload, self.is_absolute, self.blocking_sync, self.random)
                .unwrap(),
        )
    }
}

declare_plugin!(CudaPlugin, CudaPlugin::new, CudaOpt);
