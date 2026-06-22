use crate::{Error, NonceGenEnum};
use cust::context::CurrentContext;
use cust::device::DeviceAttribute;
use cust::function::Function;
use cust::module::{ModuleJitOption, OptLevel};
use cust::prelude::*;
use keryx_plugin_api::xoshiro256starstar::Xoshiro256StarStar;
use keryx_plugin_api::Worker;
use log::{error, info};
use rand::{Fill, RngCore};
use std::ffi::CString;
use std::sync::{Arc, Weak};

static BPS: f32 = 1.;

static PTX_120: &str = include_str!("../resources/keryx-cuda-sm120.ptx");
static PTX_100: &str = include_str!("../resources/keryx-cuda-sm100.ptx");
static PTX_90: &str = include_str!("../resources/keryx-cuda-sm90.ptx");
static PTX_89: &str = include_str!("../resources/keryx-cuda-sm89.ptx");
static PTX_86: &str = include_str!("../resources/keryx-cuda-sm86.ptx");
// sm_80 (Ampere GA100 — A100 / CMP 170HX). HBM2 means tons of memory bandwidth
// (~1.5 TB/s on the 170HX) but only ~70 SMs with sm_80 ALU throughput per SM.
// Compute-bound for this kernel today; the HBM2 angle pays off if we ever
// stage the matrix or per-warp work through global memory streams.
static PTX_80: &str = include_str!("../resources/keryx-cuda-sm80.ptx");
static PTX_75: &str = include_str!("../resources/keryx-cuda-sm75.ptx");
static PTX_61: &str = include_str!("../resources/keryx-cuda-sm61.ptx");
// sm_30 (Kepler) and sm_20 (Fermi) dropped: CUDA 12+ no longer compiles for
// these architectures, and they predate practical GPU mining anyway.

pub struct Kernel<'kernel> {
    func: Arc<Function<'kernel>>,
    block_size: u32,
    grid_size: u32,
}

impl<'kernel> Kernel<'kernel> {
    pub fn new(module: Weak<Module>, name: &'kernel str) -> Result<Kernel<'kernel>, Error> {
        let func = Arc::new(unsafe {
            module.as_ptr().as_ref().unwrap().get_function(name).map_err(|e| {
                error!("Error loading function: {}", e);
                e
            })?
        });
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

        let device = CurrentContext::get_device()?;
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
        let grid_size = sm_count * func.max_active_blocks_per_multiprocessor(block_size.into(), 0)?;

        Ok(Self { func, block_size, grid_size })
    }

    pub fn get_workload(&self) -> u32 {
        self.block_size * self.grid_size
    }

    pub fn set_workload(&mut self, workload: u32) {
        self.grid_size = (workload + self.block_size - 1) / self.block_size
    }
}

pub struct CudaGPUWorker<'gpu> {
    // NOTE: The order is important! context must be closed last
    heavy_hash_kernel: Kernel<'gpu>,
    stream: Stream,
    start_event: Event,
    stop_event: Event,
    _module: Arc<Module>,

    rand_state: DeviceBuffer<u64>,
    final_nonce_buff: DeviceBuffer<u64>,

    device_id: u32,
    pub workload: usize,
    _context: Context,

    random: NonceGenEnum,
}

impl<'gpu> Worker for CudaGPUWorker<'gpu> {
    fn id(&self) -> String {
        let device = CurrentContext::get_device().unwrap();
        format!("#{} ({})", self.device_id, device.name().unwrap())
    }

    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]) {
        let u8matrix: Arc<[[u8; 64]; 64]> = Arc::new(matrix.map(|row| row.map(|v| v as u8)));
        let mut hash_header_gpu = self._module.get_global::<[u8; 72]>(&CString::new("hash_header").unwrap()).unwrap();
        hash_header_gpu.copy_from(hash_header).map_err(|e| e.to_string()).unwrap();

        let mut matrix_gpu = self._module.get_global::<[[u8; 64]; 64]>(&CString::new("matrix").unwrap()).unwrap();
        matrix_gpu.copy_from(&u8matrix).map_err(|e| e.to_string()).unwrap();

        let mut target_gpu = self._module.get_global::<[u64; 4]>(&CString::new("target").unwrap()).unwrap();
        target_gpu.copy_from(target).map_err(|e| e.to_string()).unwrap();
    }

    #[inline(always)]
    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64) {
        let func = &self.heavy_hash_kernel.func;
        let stream = &self.stream;
        let random: u8 = match self.random {
            NonceGenEnum::Lean => {
                self.rand_state.copy_from(&[rand::thread_rng().next_u64()]).unwrap();
                0
            }
            NonceGenEnum::Xoshiro => 1,
        };

        self.start_event.record(stream).unwrap();
        unsafe {
            launch!(
                func<<<
                    self.heavy_hash_kernel.grid_size, self.heavy_hash_kernel.block_size,
                    0, stream
                >>>(
                    nonce_mask, nonce_fixed,
                    self.workload,
                    random,
                    self.rand_state.as_device_ptr(),
                    self.final_nonce_buff.as_device_ptr()
                )
            )
            .unwrap(); // We see errors in sync
        }
        self.stop_event.record(stream).unwrap();
    }

    #[inline(always)]
    fn sync(&self) -> Result<(), Error> {
        //self.stream.synchronize()?;
        self.stop_event.synchronize()?;
        if self.stop_event.elapsed_time_f32(&self.start_event)? > 1000. / BPS {
            return Err("Cuda takes longer then block rate. Please reduce your workload.".into());
        }
        Ok(())
    }

    fn get_workload(&self) -> usize {
        self.workload
    }

    #[inline(always)]
    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error> {
        self.final_nonce_buff.copy_to(nonces)?;
        Ok(())
    }
}

impl<'gpu> CudaGPUWorker<'gpu> {
    pub fn new(
        device_id: u32,
        workload: f32,
        is_absolute: bool,
        blocking_sync: bool,
        random: NonceGenEnum,
    ) -> Result<Self, Error> {
        info!("Starting a CUDA worker");
        let sync_flag = match blocking_sync {
            true => ContextFlags::SCHED_BLOCKING_SYNC,
            false => ContextFlags::SCHED_AUTO,
        };
        let device = Device::get_device(device_id).unwrap();
        let _context = Context::new(device)?;
        _context.set_flags(sync_flag)?;

        let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
        let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
        let _module: Arc<Module>;
        info!("Device #{} compute version is {}.{}", device_id, major, minor);

        let load_ptx = |ptx, label: &str| {
            Module::from_ptx(ptx, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Failed to load {} PTX (driver too old?): {}", label, e);
                e
            })
        };

        // For sm_89 (Ada/RTX 40) and sm_100 (Blackwell/RTX 50), the PTX was compiled with
        // CUDA 13.2 (PTX ISA 9.2) which requires driver >= 570. If the driver is older, we
        // fall back to sm_86 (CUDA 12.0 / PTX 8.0, driver >= 520) which runs on all these
        // architectures via NVIDIA's backward-compatible PTX JIT.
        if major >= 12 {
            // sm_120 (RTX 50 / consumer Blackwell — GeForce RTX 5090 etc.).
            // NVIDIA splits sm_100 (datacenter Blackwell — H100/B100/GH100) and
            // sm_120 (consumer Blackwell) as separate compute architectures, so
            // the sm_100 PTX errors out with `unknown error` on a 5090 even with
            // a 580+ driver. Compiled with CUDA 12.8 nvcc against compute_120
            // (the first toolkit with native sm_120 support).
            _module = Arc::new(match load_ptx(PTX_120, "sm_120") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_120 PTX", device_id);
                    m
                }
                Err(e) => {
                    info!("GPU #{} sm_120 PTX failed; trying sm_100 then sm_86 fallback", device_id);
                    match load_ptx(PTX_100, "sm_100") {
                        Ok(m) => m,
                        Err(_) => load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?,
                    }
                }
            });
        } else if major >= 10 {
            // sm_100 (datacenter Blackwell — H100 / B100 / GH100)
            _module = Arc::new(match load_ptx(PTX_100, "sm_100") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_100 PTX", device_id);
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_86 PTX (update driver to 570+ for full Blackwell optimisation)", device_id);
                    load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?
                }
            });
        } else if major == 9 {
            // sm_90 (Hopper — H100 / H200 / GH200). Native compute_90 PTX;
            // falls back to sm_89 (JIT) then sm_86 if the driver can't load it.
            _module = Arc::new(match load_ptx(PTX_90, "sm_90") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_90 PTX", device_id);
                    m
                }
                Err(e) => {
                    info!("GPU #{} sm_90 PTX failed; trying sm_89 then sm_86 fallback", device_id);
                    match load_ptx(PTX_89, "sm_89") {
                        Ok(m) => m,
                        Err(_) => load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?,
                    }
                }
            });
        } else if major == 8 && minor >= 9 {
            // sm_89 (RTX 40 / Ada Lovelace)
            _module = Arc::new(match load_ptx(PTX_89, "sm_89") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_89 PTX", device_id);
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_86 PTX (update driver to 570+ for full Ada Lovelace optimisation)", device_id);
                    load_ptx(PTX_86, "sm_86 (fallback)").map_err(|_| e)?
                }
            });
        } else if major == 8 && minor >= 6 {
            // sm_86 (RTX 30 / Ampere)
            _module = Arc::new(load_ptx(PTX_86, "sm_86")?);
        } else if major == 8 {
            // sm_80 (datacenter Ampere — A100 / CMP 170HX). Distinct from
            // sm_86 (consumer Ampere) — A100/170HX have HBM2 + larger
            // shared memory budget but fewer SMs.
            _module = Arc::new(match load_ptx(PTX_80, "sm_80") {
                Ok(m) => {
                    info!("GPU #{} using optimised sm_80 PTX", device_id);
                    m
                }
                Err(e) => {
                    info!("GPU #{} falling back to sm_75 PTX (sm_80 PTX failed: {})", device_id, e);
                    Module::from_ptx(PTX_75, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|_| e)?
                }
            });
        } else if major > 7 || (major == 7 && minor >= 5) {
            // sm_75 (RTX 20 / Turing)
            _module = Arc::new(Module::from_ptx(PTX_75, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Error loading PTX. Make sure you have the updated driver for you devices");
                e
            })?);
        } else if major > 6 || (major == 6 && minor >= 1) {
            // sm_61 (GTX 10 / Pascal)
            _module = Arc::new(Module::from_ptx(PTX_61, &[ModuleJitOption::OptLevel(OptLevel::O4)]).map_err(|e| {
                error!("Error loading PTX. Make sure you have the updated driver for you devices");
                e
            })?);
        } else {
            return Err(format!(
                "CUDA compute {}.{} not supported. Keryx requires sm_61 (GTX 10xx) or newer.",
                major, minor
            )
            .into());
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let mut heavy_hash_kernel = Kernel::new(Arc::downgrade(&_module), "heavy_hash")?;

        let mut chosen_workload = 0u32;
        if is_absolute {
            chosen_workload = 1;
        } else {
            let cur_workload = heavy_hash_kernel.get_workload();
            if chosen_workload == 0 || chosen_workload < cur_workload {
                chosen_workload = cur_workload;
            }
        }
        chosen_workload = (chosen_workload as f32 * workload) as u32;
        info!("GPU #{} Chosen workload: {}", device_id, chosen_workload);
        heavy_hash_kernel.set_workload(chosen_workload);

        let final_nonce_buff = vec![0u64; 1].as_slice().as_dbuf()?;

        let rand_state: DeviceBuffer<u64> = match random {
            NonceGenEnum::Xoshiro => {
                info!("Using xoshiro for nonce-generation");
                let mut buffer = DeviceBuffer::<u64>::zeroed(4 * (chosen_workload as usize)).unwrap();
                info!("GPU #{} is generating initial seed. This may take some time.", device_id);
                let mut seed = [1u64; 4];
                seed.try_fill(&mut rand::thread_rng())?;
                buffer.copy_from(
                    Xoshiro256StarStar::new(&seed)
                        .iter_jump_state()
                        .take(chosen_workload as usize)
                        .flatten()
                        .collect::<Vec<u64>>()
                        .as_slice(),
                )?;
                info!("GPU #{} initialized", device_id);
                buffer
            }
            NonceGenEnum::Lean => {
                info!("Using lean nonce-generation");
                let mut buffer = DeviceBuffer::<u64>::zeroed(1).unwrap();
                let seed = rand::thread_rng().next_u64();
                buffer.copy_from(&[seed])?;
                buffer
            }
        };
        Ok(Self {
            device_id,
            _context,
            _module,
            start_event: Event::new(EventFlags::DEFAULT)?,
            // BLOCKING_SYNC makes stop_event.synchronize() sleep the thread instead of
            // busy-waiting a CPU core. cuEventSynchronize only blocks if the EVENT has this
            // flag — the context's SCHED_BLOCKING_SYNC does NOT apply to event sync. Without
            // it the miner pins one core at 100% (the reported "high CPU"); now the default
            // (blocking_sync=true) yields, and --cuda-no-blocking-sync keeps the low-latency spin.
            stop_event: Event::new(if blocking_sync { EventFlags::BLOCKING_SYNC } else { EventFlags::DEFAULT })?,
            workload: chosen_workload as usize,
            stream,
            rand_state,
            final_nonce_buff,
            heavy_hash_kernel,
            random,
        })
    }
}
