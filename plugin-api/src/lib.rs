//! Shared plugin ABI for keryx-miner-supr.
//!
//! Extracted from the `keryx_miner` lib so the binary and the worker plugins
//! can both depend on it without a Cargo cycle (see Cargo.toml). The
//! `keryx_miner` lib re-exports everything here, so `keryx_miner::Plugin` etc.
//! keep working unchanged for the rest of the tree.

use clap::ArgMatches;
use std::any::Any;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, Ordering};

pub mod xoshiro256starstar;

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

// These atomics are deliberately process/DSO-local. A statically linked worker shares the host's
// copy; each dynamic plugin changes its own copy through the optional C-ABI control hook exported
// below by the in-tree plugins. This avoids using the process environment as a runtime
// synchronization mechanism after CUDA/ROCm and monitor threads already exist.
static TUI_REQUESTED: AtomicBool = AtomicBool::new(false);
static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);

#[doc(hidden)]
pub fn set_tui_requested(requested: bool) {
    TUI_REQUESTED.store(requested, Ordering::Release);
}

#[doc(hidden)]
pub fn tui_requested() -> bool {
    TUI_REQUESTED.load(Ordering::Acquire)
}

#[doc(hidden)]
pub fn set_tui_active(active: bool) {
    TUI_ACTIVE.store(active, Ordering::Release);
}

#[doc(hidden)]
pub fn tui_active() -> bool {
    TUI_ACTIVE.load(Ordering::Acquire)
}

pub trait Plugin: Any + Send + Sync {
    fn name(&self) -> &'static str;
    fn enabled(&self) -> bool;
    fn get_worker_specs(&self) -> Vec<Box<dyn WorkerSpec>>;
    fn process_option(&mut self, matchs: &ArgMatches) -> Result<usize, Error>;
}

pub trait WorkerSpec: Any + Send + Sync {
    fn id(&self) -> String;
    fn build(&self) -> Box<dyn Worker>;
    /// OpenCL `cl_device_id` (as `usize`) of this worker's GPU, if it is an OpenCL worker.
    /// Lets the AMD PoM driver make the possession tier resident on THIS exact card so each
    /// GPU mines PoM independently (own buffer, own thread). `None` for non-OpenCL specs (CUDA).
    fn opencl_device_id(&self) -> Option<usize> {
        None
    }
}

pub trait Worker {
    fn id(&self) -> String;
    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]);
    fn calculate_hash(&mut self, nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64);
    fn sync(&self) -> Result<(), Error>;
    fn get_workload(&self) -> usize;
    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error>;
}

/// Exports the C-ABI `_plugin_create` used by the dynamic (dlopen) plugin path.
/// Unchanged from the original definition except it now lives here; `$crate`
/// resolves to `keryx_plugin_api`, whose `Plugin`/`Error` are the same types
/// the binary sees (it re-exports them).
#[macro_export]
macro_rules! declare_plugin {
    ($plugin_type:ty, $constructor:path, $args:ty) => {
        use clap::Args;
        #[no_mangle]
        pub unsafe extern "C" fn _plugin_create(
            app: *mut clap::App,
        ) -> (*mut clap::App, *mut dyn $crate::Plugin, *const $crate::Error) {
            // make sure the constructor is the correct type.
            let constructor: fn() -> Result<$plugin_type, $crate::Error> = $constructor;

            let object = match constructor() {
                Ok(obj) => obj,
                Err(e) => {
                    return (
                        app,
                        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }, // Translates to null pointer
                        Box::into_raw(Box::new(e)),
                    );
                }
            };

            let boxed: Box<dyn $crate::Plugin> = Box::new(object);

            let boxed_app = Box::new(<$args>::augment_args(unsafe { *Box::from_raw(app) }));
            (Box::into_raw(boxed_app), Box::into_raw(boxed), std::ptr::null::<$crate::Error>())
        }
    };
}
