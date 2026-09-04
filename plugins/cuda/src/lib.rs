use keryx_plugin_api::declare_plugin;

use clap::{ArgMatches, FromArgMatches};
use cust::prelude::*;
use keryx_plugin_api::{Plugin, Worker, WorkerSpec};
use log::{LevelFilter, Log, Metadata, Record};
use std::error::Error as StdError;
use std::sync::Once;
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

// `CudaOpt` is re-exported (pub) so the binary's `static-cuda` path can merge
// the CUDA clap args directly (`keryxcuda::CudaOpt::augment_args`).
pub use crate::cli::CudaOpt;
use crate::cli::NonceGenEnum;
use crate::worker::CudaGPUWorker;

const DEFAULT_WORKLOAD_SCALE: f32 = 1024.;

/// A dynamically loaded plugin has a separate `log` global from the host executable. Keep its
/// normal `env_logger` installed, but silence it while the host owns the terminal for the TUI.
/// The state is a DSO-local atomic toggled through `keryx_plugin_set_tui_active_v1`; process
/// environment mutation is intentionally not used as a cross-thread synchronization primitive.
struct TuiAwareLogger {
    inner: env_logger::Logger,
}

impl Log for TuiAwareLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        !keryx_plugin_api::tui_active() && self.inner.enabled(metadata)
    }

    fn log(&self, record: &Record<'_>) {
        // Recheck here because the dashboard can start or stop between `enabled` and `log`.
        if !keryx_plugin_api::tui_active() {
            self.inner.log(record);
        }
    }

    fn flush(&self) {
        if !keryx_plugin_api::tui_active() {
            self.inner.flush();
        }
    }
}

/// Optional dynamic-plugin dashboard logger handshake. Old hosts ignore this symbol; new hosts
/// discover it with `dlsym` and pass only 0/1.
#[no_mangle]
pub extern "C" fn keryx_plugin_set_tui_active_v1(active: u8) {
    keryx_plugin_api::set_tui_active(active != 0);
}

fn init_logger() {
    // A Rust cdylib carries its own std::panicking hook slot, so the host executable's TUI panic
    // hook cannot intercept panics originating in this DSO. Preserve normal/default diagnostics in
    // classic mode, but never let that private hook write raw stderr into the alternate screen.
    static PANIC_FILTER: Once = Once::new();
    PANIC_FILTER.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !keryx_plugin_api::tui_active() {
                previous(info);
            }
        }));
    });

    let mut builder = env_logger::Builder::new();
    builder.filter_level(LevelFilter::Info).parse_default_env();
    let logger = builder.build();
    let max_level = logger.filter();

    // A statically linked plugin shares the host's `log` global, so failure here simply means the
    // host logger is already authoritative. Dynamic plugins install this logger in their own DSO.
    if log::set_boxed_logger(Box::new(TuiAwareLogger { inner: logger })).is_ok() {
        log::set_max_level(max_level);
    }
}

/// Optional dynamic-plugin output-contract handshake.
///
/// An old host does not look this symbol up, so the worker translates its raw MAX sentinel back to
/// legacy zero. A new host calls it before workers are built and can then distinguish a genuine
/// nonce zero from no winner without changing the Rust trait-object ABI.
#[no_mangle]
pub extern "C" fn keryx_plugin_enable_raw_nonce_v1() -> u64 {
    worker::enable_raw_nonce_output_v1()
}

/// Resolve a CUDA *visible logical ordinal* to the same physical GPU in NVML.
///
/// NVML indices always use the machine-global ordering, while CUDA renumbers devices after
/// `CUDA_VISIBLE_DEVICES` (and accepts UUID masks). Looking up NVML by the logical ordinal made a
/// one-card process masked to physical GPU 1 monitor and overclock physical GPU 0 instead. PCI BDF
/// is stable in both APIs and avoids relying on either enumeration order.
#[cfg(feature = "overclock")]
fn nvml_device_for_cuda(nvml: &Nvml, cuda_ordinal: u16) -> Result<NvmlDevice<'_>, Error> {
    use cust::device::DeviceAttribute;

    let cuda = Device::get_device(cuda_ordinal as u32)?;
    let domain = cuda.get_attribute(DeviceAttribute::PciDomainId)? as u32;
    let bus = cuda.get_attribute(DeviceAttribute::PciBusId)? as u32;
    let device = cuda.get_attribute(DeviceAttribute::PciDeviceId)? as u32;
    // NVML's canonical bus-id form uses an eight-digit PCI domain.
    let bus_id = format!("{domain:08x}:{bus:02x}:{device:02x}.0");
    Ok(nvml.device_by_pci_bus_id(bus_id)?)
}

pub struct CudaPlugin {
    specs: Vec<CudaWorkerSpec>,
    #[cfg(feature = "overclock")]
    nvml_instance: Nvml,
    _enabled: bool,
}

impl CudaPlugin {
    pub fn new() -> Result<Self, Error> {
        Self::new_inner(false)
    }

    // The exported plugin factory uses this constructor, while a static-cuda host calls `new`
    // directly. That distinction matters only during early TUI startup: a static plugin shares
    // the host's logger slot and must leave it free for the dashboard logger.
    fn new_dynamic() -> Result<Self, Error> {
        Self::new_inner(true)
    }

    fn new_inner(dynamic_plugin: bool) -> Result<Self, Error> {
        cust::init(CudaFlags::empty())?;
        if dynamic_plugin || !keryx_plugin_api::tui_requested() {
            init_logger();
        }
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
    fn process_option(&mut self, matches: &ArgMatches) -> Result<usize, keryx_plugin_api::Error> {
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

                    let mut nvml_device: NvmlDevice = nvml_device_for_cuda(&self.nvml_instance, gpus[i])?;

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
                    let mut nvml_device: NvmlDevice = nvml_device_for_cuda(&self.nvml_instance, gpus[i])?;
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
                            Err(e) => {
                                warn!("keryxcuda-monitor: NVML init failed: {:?} — monitor disabled", e);
                                return;
                            }
                        };
                        loop {
                            for (idx, &gpu_id) in gpus_for_monitor.iter().enumerate() {
                                if let Ok(dev) = nvml_device_for_cuda(&nvml, gpu_id) {
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
                                            .map(|m| format!(
                                                "{}/{}MB",
                                                m.used / (1024 * 1024),
                                                m.total / (1024 * 1024)
                                            ))
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

#[cfg(test)]
mod tui_panic_filter_tests {
    use super::init_logger;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FORWARDED: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn dso_panic_output_is_forwarded_only_in_classic_mode() {
        std::panic::set_hook(Box::new(|_| {
            FORWARDED.fetch_add(1, Ordering::SeqCst);
        }));
        init_logger();

        keryx_plugin_api::set_tui_active(false);
        let _ = std::panic::catch_unwind(|| panic!("classic panic probe"));
        assert_eq!(FORWARDED.load(Ordering::SeqCst), 1);

        keryx_plugin_api::set_tui_active(true);
        let _ = std::panic::catch_unwind(|| panic!("dashboard panic probe"));
        assert_eq!(FORWARDED.load(Ordering::SeqCst), 1);
        keryx_plugin_api::set_tui_active(false);
    }
}

declare_plugin!(CudaPlugin, CudaPlugin::new_dynamic, CudaOpt);
