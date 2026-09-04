use clap::{ArgMatches, FromArgMatches};
use keryx_miner::declare_plugin;
use keryx_miner::{Plugin, Worker, WorkerSpec};
use log::{info, warn, LevelFilter, Log, Metadata, Record};
use opencl3::device::{Device, CL_DEVICE_TYPE_ALL};
use opencl3::platform::{get_platforms, Platform};
use opencl3::types::cl_device_id;
use std::error::Error as StdError;
use std::sync::Once;

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

mod cli;
mod worker;

use crate::cli::{NonceGenEnum, OpenCLOpt};
use crate::worker::OpenCLGPUWorker;

// Sentinel: user did not pass --opencl-workload, so the worker resolves a
// capability-driven default ratio from the GPU arch (see worker::default_workload_scale).
const AUTO_WORKLOAD: f32 = 0.;

/// A dynamically loaded plugin has a separate `log` global from the host executable. Keep its
/// normal `env_logger` installed, but silence it while the host owns the terminal for the TUI.
/// The state is a DSO-local atomic toggled through `keryx_plugin_set_tui_active_v1`; process
/// environment mutation is intentionally not used as a cross-thread synchronization primitive.
struct TuiAwareLogger {
    inner: env_logger::Logger,
}

impl Log for TuiAwareLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        !keryx_miner::tui_active() && self.inner.enabled(metadata)
    }

    fn log(&self, record: &Record<'_>) {
        // Recheck here because the dashboard can start or stop between `enabled` and `log`.
        if !keryx_miner::tui_active() {
            self.inner.log(record);
        }
    }

    fn flush(&self) {
        if !keryx_miner::tui_active() {
            self.inner.flush();
        }
    }
}

/// Optional dynamic-plugin dashboard logger handshake. Old hosts ignore this symbol; new hosts
/// discover it with `dlsym` and pass only 0/1.
#[no_mangle]
pub extern "C" fn keryx_plugin_set_tui_active_v1(active: u8) {
    keryx_miner::set_tui_active(active != 0);
}

fn init_logger() {
    // A Rust cdylib carries its own std::panicking hook slot, so the host executable's TUI panic
    // hook cannot intercept panics originating in this DSO. Preserve normal/default diagnostics in
    // classic mode, but never let that private hook write raw stderr into the alternate screen.
    static PANIC_FILTER: Once = Once::new();
    PANIC_FILTER.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !keryx_miner::tui_active() {
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

/// Optional dynamic-plugin output-contract handshake. New hosts call this
/// before constructing workers and compare the raw result against MAX. Old
/// hosts continue to receive their historical zero no-winner sentinel.
#[no_mangle]
pub extern "C" fn keryx_plugin_enable_raw_nonce_v1() -> u64 {
    worker::enable_raw_nonce_output_v1()
}

pub struct OpenCLPlugin {
    specs: Vec<OpenCLWorkerSpec>,
    _enabled: bool,
}

impl OpenCLPlugin {
    fn new() -> Result<Self, Error> {
        init_logger();
        Ok(Self { specs: Vec::new(), _enabled: false })
    }
}

impl Plugin for OpenCLPlugin {
    fn name(&self) -> &'static str {
        "OpenCL Worker"
    }

    fn enabled(&self) -> bool {
        self._enabled
    }

    fn get_worker_specs(&self) -> Vec<Box<dyn WorkerSpec>> {
        self.specs.iter().map(|spec| Box::new(*spec) as Box<dyn WorkerSpec>).collect::<Vec<Box<dyn WorkerSpec>>>()
    }

    //noinspection RsTypeCheck
    fn process_option(&mut self, matches: &ArgMatches) -> Result<usize, keryx_miner::Error> {
        let opts: OpenCLOpt = OpenCLOpt::from_arg_matches(matches)?;

        self._enabled = opts.opencl_enable;
        let platforms = match get_platforms() {
            Ok(p) => p,
            Err(e) => {
                return Err(e.to_string().into());
            }
        };
        info!("OpenCL Found Platforms:");
        info!("=======================");
        for platform in &platforms {
            let vendor = &platform.vendor().unwrap_or_else(|_| "Unk".into());
            let name = &platform.name().unwrap_or_else(|_| "Unk".into());
            let num_devices = platform.get_devices(CL_DEVICE_TYPE_ALL).unwrap_or_default().len();
            info!("{}: {} ({} devices available)", vendor, name, num_devices);
        }
        let amd_platforms = (&platforms)
            .iter()
            .filter(|p| {
                p.vendor().unwrap_or_else(|_| "Unk".into()) == "Advanced Micro Devices, Inc."
                    && !p.get_devices(CL_DEVICE_TYPE_ALL).unwrap_or_default().is_empty()
            })
            .collect::<Vec<&Platform>>();
        let _platform: &Platform = match opts.opencl_platform {
            Some(idx) => {
                self._enabled = true;
                &platforms[idx as usize]
            }
            None if !opts.opencl_amd_disable && !amd_platforms.is_empty() => {
                self._enabled = true;
                let amd = amd_platforms[0];
                let plat_name = amd.name().unwrap_or_else(|_| "Unk".into());
                if !plat_name.contains("ROCm") && !plat_name.contains("AMD Accelerated") {
                    warn!(
                        "AMD OpenCL platform detected but does not appear to be the ROCm runtime. \
                         RDNA 3+ GPUs (RX 7000/9000) may have issues. \
                         Install rocm-opencl-runtime for best support."
                    );
                }
                amd
            }
            None => &platforms[0],
        };
        if self._enabled {
            info!(
                "Chose to mine on {}: {}.",
                &_platform.vendor().unwrap_or_else(|_| "Unk".into()),
                &_platform.name().unwrap_or_else(|_| "Unk".into())
            );

            let device_ids = _platform.get_devices(CL_DEVICE_TYPE_ALL).unwrap();
            let gpus = match opts.opencl_device {
                Some(dev) => {
                    self._enabled = true;
                    dev.iter().map(|d| device_ids[*d as usize]).collect::<Vec<cl_device_id>>()
                }
                None => device_ids,
            };

            self.specs = (0..gpus.len())
                .map(|i| OpenCLWorkerSpec {
                    _platform: *_platform,
                    index: i,
                    device_id: Device::new(gpus[i]),
                    workload: match &opts.opencl_workload {
                        Some(workload) if i < workload.len() => workload[i],
                        Some(workload) if !workload.is_empty() => *workload.last().unwrap(),
                        // AUTO: no --opencl-workload given. 0.0 is a sentinel that
                        // tells the worker to pick a capability-driven default ratio
                        // per GPU arch (the old flat 512 under-saturated big cards).
                        _ => AUTO_WORKLOAD,
                    },
                    is_absolute: opts.opencl_workload_absolute,
                    experimental_amd: opts.experimental_amd,
                    // The legacy flag remains parse-compatible, but the worker now always
                    // JIT-compiles the canonical source. Archived binaries predate the
                    // MAX/atomic-min winner contract and are intentionally never loaded.
                    use_amd_binary: opts.opencl_use_amd_binary && !opts.opencl_no_amd_binary,
                    random: opts.opencl_nonce_gen,
                })
                .collect();
        }
        Ok(self.specs.len())
    }
}

#[derive(Copy, Clone)]
struct OpenCLWorkerSpec {
    _platform: Platform,
    index: usize,
    device_id: Device,
    workload: f32,
    is_absolute: bool,
    experimental_amd: bool,
    use_amd_binary: bool,
    random: NonceGenEnum,
}

impl WorkerSpec for OpenCLWorkerSpec {
    fn id(&self) -> String {
        format!(
            "#{} {}",
            self.index,
            self.device_id
                .board_name_amd()
                .unwrap_or_else(|_| self.device_id.name().unwrap_or_else(|_| "Unknown Device".into()))
        )
    }

    fn build(&self) -> Box<dyn Worker> {
        Box::new(
            OpenCLGPUWorker::new(
                self.device_id,
                self.workload,
                self.is_absolute,
                self.experimental_amd,
                self.use_amd_binary,
                &self.random,
            )
            .unwrap(),
        )
    }

    // The raw cl_device_id for this GPU, so the AMD PoM driver can build a per-card resident
    // tier + miner (each GPU mines PoM on its own buffer/thread instead of all funneling onto
    // device 0 through one global lock).
    fn opencl_device_id(&self) -> Option<usize> {
        Some(self.device_id.id() as usize)
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

        keryx_miner::set_tui_active(false);
        let _ = std::panic::catch_unwind(|| panic!("classic panic probe"));
        assert_eq!(FORWARDED.load(Ordering::SeqCst), 1);

        keryx_miner::set_tui_active(true);
        let _ = std::panic::catch_unwind(|| panic!("dashboard panic probe"));
        assert_eq!(FORWARDED.load(Ordering::SeqCst), 1);
        keryx_miner::set_tui_active(false);
    }
}

declare_plugin!(OpenCLPlugin, OpenCLPlugin::new, OpenCLOpt);
