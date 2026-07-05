//! Built-in mining worker for Apple Silicon (macOS + `pom-metal`).
//!
//! On Linux the GPU mining worker threads come from the `keryxcuda` / `keryxopencl` plugins (the
//! `.so` files). macOS ships no plugin, so without this the miner finds 0 workers and exits with
//! "No workers specified" — even though the Metal PoM walk (`pom_gpu_metal`) is compiled in.
//!
//! This provides a minimal `Worker` whose only real job is to give the mining loop a device
//! identity (`#0 (Apple Silicon GPU)`), which the PoM branch parses to pick the Metal device and
//! then drives the walk via `pom_gpu` (aliased to `pom_gpu_metal`). The kHeavyHash methods are
//! no-op stubs: the PoM branch `continue`s (miner.rs) before ever reaching them, and macOS miners
//! only run post-fork (PoM), so kHeavyHash never executes here.

use clap::ArgMatches;
use keryx_miner::{Error, Plugin, Worker, WorkerSpec};

pub struct MetalPlugin;

impl MetalPlugin {
    pub fn new() -> Self {
        MetalPlugin
    }
}

impl Plugin for MetalPlugin {
    fn name(&self) -> &'static str {
        "keryxmetal"
    }
    fn enabled(&self) -> bool {
        true
    }
    fn get_worker_specs(&self) -> Vec<Box<dyn WorkerSpec>> {
        // Apple Silicon exposes a single integrated GPU (Metal device 0).
        vec![Box::new(MetalWorkerSpec { device: 0 })]
    }
    fn process_option(&mut self, _matchs: &ArgMatches) -> Result<usize, Error> {
        Ok(1)
    }
}

struct MetalWorkerSpec {
    device: u32,
}

impl WorkerSpec for MetalWorkerSpec {
    fn id(&self) -> String {
        format!("#{} (Apple Silicon GPU)", self.device)
    }
    fn build(&self) -> Box<dyn Worker> {
        Box::new(MetalWorker { device: self.device })
    }
}

struct MetalWorker {
    device: u32,
}

impl Worker for MetalWorker {
    /// The mining loop parses this as `#<N> (...)` → device N for the PoM walk. MUST start with
    /// `#<device> ` so `strip_prefix('#')` + `split_whitespace().next().parse::<u32>()` yields N.
    fn id(&self) -> String {
        format!("#{} (Apple Silicon GPU)", self.device)
    }

    // ── kHeavyHash path — NEVER reached on macOS (post-fork PoM `continue`s first). Stubs. ──
    fn load_block_constants(&mut self, _hash_header: &[u8; 72], _matrix: &[[u16; 64]; 64], _target: &[u64; 4]) {}
    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, _nonce_mask: u64, _nonce_fixed: u64) {}
    fn sync(&self) -> Result<(), Error> {
        Ok(())
    }
    fn get_workload(&self) -> usize {
        1 << 20
    }
    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error> {
        // No kHeavyHash winner (this path never runs post-fork; keep it inert if it ever does).
        if !nonces.is_empty() {
            nonces[0] = 0;
        }
        Ok(())
    }
}
