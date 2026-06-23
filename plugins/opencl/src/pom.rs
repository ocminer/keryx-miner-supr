// PomMiner — the AMD-side OpenCL PoM mining driver (destined for the keryxopencl plugin).
// Holds the tier weight blob resident in one cl_mem buffer; mine() launches pom_mine over a
// nonce batch and returns the lowest passing nonce (the host then re-verifies + builds the proof).
// Mirrors the opencl3 0.6 patterns in plugins/opencl/src/worker.rs.

use std::ptr;
use std::sync::Arc;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_ulong, CL_BLOCKING};

pub const POM_WALK_STEPS: u32 = 256;
const POM_SRC: &str = include_str!("../resources/pom_mine.cl");

pub struct PomMiner {
    _context: Arc<Context>,
    queue: CommandQueue,
    kernel: Kernel,
    weights: Buffer<cl_ulong>,
    winner: Buffer<cl_ulong>,
    pub n_chunks: u64,
}

impl PomMiner {
    /// `blob` = the tier in canonical chunk order, N*4 little-endian u64.
    pub fn new(device: Device, blob: &[u64], n_chunks: u64) -> Result<Self, String> {
        let context = Arc::new(Context::from_device(&device).map_err(|e| e.to_string())?);
        let queue = CommandQueue::create(&context, device.id(), 0).map_err(|e| e.to_string())?;
        let program = Program::create_and_build_from_source(&context, POM_SRC, "")?;
        let kernel = Kernel::create(&program, "pom_mine").map_err(|e| e.to_string())?;
        // worker.rs pattern: a context ref that outlives the borrow checker (Arc kept in struct).
        let cref = unsafe { Arc::as_ptr(&context).as_ref().unwrap() };
        let mut weights = Buffer::<cl_ulong>::create(cref, CL_MEM_READ_ONLY, blob.len(), ptr::null_mut())
            .map_err(|e| e.to_string())?;
        queue.enqueue_write_buffer(&mut weights, CL_BLOCKING, 0, blob, &[]).map_err(|e| e.to_string())?;
        let winner = Buffer::<cl_ulong>::create(cref, CL_MEM_READ_WRITE, 1, ptr::null_mut())
            .map_err(|e| e.to_string())?;
        Ok(Self { _context: context, queue, kernel, weights, winner, n_chunks })
    }

    /// Launch one batch of `batch` nonces from `nonce_base`. Returns the lowest nonce whose
    /// pom_pow_value <= target, or None.
    pub fn mine(&mut self, pph: [u64; 4], time: u64, target: [u64; 4], nonce_base: u64, batch: u64) -> Option<u64> {
        self.queue.enqueue_write_buffer(&mut self.winner, CL_BLOCKING, 0, &[u64::MAX], &[]).ok()?;
        ExecuteKernel::new(&self.kernel)
            .set_arg(&self.weights)
            .set_arg(&self.n_chunks)
            .set_arg(&POM_WALK_STEPS)
            .set_arg(&pph[0]).set_arg(&pph[1]).set_arg(&pph[2]).set_arg(&pph[3])
            .set_arg(&time)
            .set_arg(&target[0]).set_arg(&target[1]).set_arg(&target[2]).set_arg(&target[3])
            .set_arg(&nonce_base).set_arg(&batch)
            .set_arg(&self.winner)
            .set_global_work_size(batch as usize)
            .enqueue_nd_range(&self.queue)
            .ok()?;
        self.queue.finish().ok()?;
        let mut w = [u64::MAX];
        self.queue.enqueue_read_buffer(&self.winner, CL_BLOCKING, 0, &mut w, &[]).ok()?;
        if w[0] == u64::MAX { None } else { Some(w[0]) }
    }
}
