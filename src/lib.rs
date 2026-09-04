use clap::ArgMatches;
use libloading::{Library, Symbol};

pub mod gguf;
pub mod inference;
pub mod integrity;
pub mod keccak;
pub mod models;
#[cfg(any(
    all(feature = "pom-opencl", unix),
    all(
        any(
            all(feature = "pom-cuda", not(feature = "pom-opencl")),
            all(target_os = "macos", feature = "pom-metal")
        ),
        any(unix, windows)
    )
))]
mod native_llama_log;
/// Process-wide, backend-neutral operational statistics for optional frontends.
/// Producers are best-effort and never block mining or inference work.
pub mod runtime_stats;
pub mod slm;
/// Stratum miner-telemetry (mining.hello / mining.telemetry) — display/ops only, best-effort.
pub mod telemetry;
pub mod wait_ready;
// Device-mapped quantized model forks (OPoI v2 archs) — used by slm inference and the
// PoM zero-dup gather. Device-agnostic (candle Device = CPU or CUDA), so they build
// regardless of the cuda feature.
pub mod quantized_gemma3_split;
pub mod quantized_llama_split;
pub mod quantized_qwen3_split;

// The plugin ABI (traits, Error, RNG, declare_plugin! macro) lives in the
// standalone `keryx-plugin-api` crate so the binary and the worker plugins can
// share it without a Cargo cycle. Re-export it here so `keryx_miner::Plugin`,
// `keryx_miner::Error`, `keryx_miner::xoshiro256starstar` and
// `keryx_miner::declare_plugin!` keep resolving for the rest of the tree.
pub mod pom;
#[cfg(feature = "pom-opencl")]
pub mod pom_opencl;
pub mod pom_v4;
// llama.cpp llama-server inference: AMD (Vulkan) always; NVIDIA (CUDA server) since Phase 1 of
// candle-independence — the module self-disables when no server binary is bundled/env-pointed.
#[cfg(any(feature = "pom-opencl", feature = "pom-cuda"))]
pub mod llama_vulkan;
// AMD in-process llama.cpp engine (dlopen'd libkeryx-llama-vk.so): zero-dup — llama hosts the
// model on the inference card, the PoM walk gathers over its VRAM, OPoI text runs in-process.
// Absent .so = the llama-server subprocess, then only an explicitly enabled deprecated candle-CPU
// emergency fallback, plus per-card OpenCL blobs.
#[cfg(all(feature = "pom-opencl", unix))]
pub mod llama_engine_vk;
// Windows AMD: no dlopen — stub keeps call sites identical (OpenCL blob + llama-server paths).
#[cfg(all(feature = "pom-opencl", not(unix)))]
pub mod llama_engine_vk {
    pub fn ensure_loaded(_gguf: &str, _gpu: usize) -> bool {
        false
    }
    pub fn pick_discrete_ggml_device() -> Option<i32> {
        None
    }
    pub fn available() -> bool {
        false
    }
    pub fn active_for(_gguf: &str) -> bool {
        false
    }
    pub fn generate(_prompt: &str, _max_tokens: usize) -> Option<String> {
        None
    }
    pub fn generate_for(_gguf: &str, _prompt: &str, _max_tokens: usize) -> Option<String> {
        None
    }
    pub fn pom_ready() -> bool {
        false
    }
    pub fn pom_pci() -> Option<(u32, u32, u32, u32)> {
        None
    }
    pub fn pom_byte_gate(_index: &crate::pom::WeightIndex) -> bool {
        false
    }
    pub fn pom_mine(
        _p: [u64; 4],
        _s: [u64; 4],
        _time: u64,
        _t: [u64; 4],
        _nonce_base: u64,
        _batch: u64,
        _walk_v2: bool,
    ) -> Option<u64> {
        None
    }
    pub fn unload() -> bool {
        false
    }
    pub fn evicted_for_vram() -> bool {
        false
    }
    pub fn mark_gpu_inference_unfit() {}
}
// In-process llama.cpp engine (dlopen'd libkeryx-llama.{so,dylib}): candle-independence — when
// present it hosts the model (the walk gathers over its VRAM on CUDA / zero-dup; on Metal the
// walk today keeps its own packed buffer) AND serves OPoI text. Compiled on every unix target
// In-process llama.cpp engine — Linux CUDA + macOS Metal + Windows CUDA. Loads keryx-llama.{so,dll,
// dylib} via `libloading` (dlopen / LoadLibrary), so it is now a REAL module on Windows too (it was
// previously stubbed there, which left Windows NVIDIA rigs unable to serve H4/H6 models — they only
// load via llama.cpp). The stub below only remains for exotic non-unix, non-windows targets.
#[cfg(all(
    any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")),
    any(unix, windows)
))]
pub mod llama_engine;
// Fallback stub for any target that is neither unix nor windows — keeps call sites identical.
#[cfg(all(
    any(all(feature = "pom-cuda", not(feature = "pom-opencl")), all(target_os = "macos", feature = "pom-metal")),
    not(any(unix, windows))
))]
pub mod llama_engine {
    pub fn ensure_loaded(_gguf: &str, _gpu: usize) -> bool {
        false
    }
    pub fn ensure_loaded_on(_gguf: &str, _gpu: usize) -> bool {
        false
    }
    pub fn active_for(_gguf: &str, _gpu: usize) -> bool {
        false
    }
    pub fn available() -> bool {
        false
    }
    pub fn available_on(_gpu: usize) -> bool {
        false
    }
    pub fn unload() {}
    pub fn unload_for_gpu(_gpu: usize) {}
    pub fn abandon_for_gpu(_gpu: usize) -> bool {
        false
    }
    pub fn tensors() -> Option<Vec<(String, u64, usize, bool)>> {
        None
    }
    pub fn tensors_for(_gpu: usize) -> Option<Vec<(String, u64, usize, bool)>> {
        None
    }
    pub fn foreign_device_tensor(_expected_gpu: usize) -> Option<(String, i32)> {
        None
    }
    pub fn generate(_prompt: &str, _max_tokens: usize) -> Option<String> {
        None
    }
    pub fn generate_on(_gpu: usize, _prompt: &str, _max_tokens: usize) -> Option<String> {
        None
    }
    pub fn generate_for(_gpu: usize, _gguf: &str, _prompt: &str, _max_tokens: usize) -> Option<String> {
        None
    }
}
// PoM GPU driver aliased to `pom_gpu` per platform so main.rs/miner.rs/slm.rs stay backend-agnostic:
// NVIDIA CUDA on Linux/Windows, Apple-Silicon Metal on macOS. Mutually exclusive by construction
// (the macOS Metal build never enables pom-cuda), and the `not(...)` guard makes that explicit.
#[cfg(all(feature = "pom-cuda", not(all(target_os = "macos", feature = "pom-metal"))))]
pub mod pom_gpu;
#[cfg(all(target_os = "macos", feature = "pom-metal"))]
#[path = "pom_gpu_metal.rs"]
pub mod pom_gpu;
pub use keryx_plugin_api::{
    declare_plugin, set_tui_active, set_tui_requested, tui_active, tui_requested, xoshiro256starstar, Error, Plugin,
    Worker, WorkerSpec,
};

/// Optional dynamic-plugin hook used to switch its private logger when the host enters/leaves the
/// alternate screen. `u8` keeps the C ABI explicit (`0` = classic, nonzero = dashboard).
pub type PluginTuiLogControl = unsafe extern "C" fn(u8);

#[derive(Default)]
pub struct PluginManager {
    plugins: Vec<ManagedPlugin>,
    loaded_libraries: Vec<Library>,
}

struct ManagedPlugin {
    plugin: Box<dyn Plugin>,
    no_winner: u64,
    tui_log_control: Option<PluginTuiLogControl>,
    dynamic: bool,
}

/// Host-owned metadata around an unchanged plugin `WorkerSpec` trait object. The wrapper is kept on
/// the host side deliberately: extending the Rust trait vtable would make a new miner unsafe when
/// an older `.so` remains next to it after a partial upgrade.
pub struct ManagedWorkerSpec {
    inner: Box<dyn WorkerSpec + 'static>,
    no_winner: u64,
}

impl ManagedWorkerSpec {
    pub fn id(&self) -> String {
        self.inner.id()
    }

    pub fn build(&self) -> Box<dyn Worker> {
        self.inner.build()
    }

    pub fn opencl_device_id(&self) -> Option<usize> {
        self.inner.opencl_device_id()
    }

    pub fn no_winner(&self) -> u64 {
        self.no_winner
    }
}

/**
 Plugin Manager class - allows inserting your own hashers
 Inspired by https://michael-f-bryan.github.io/rust-ffi-guide/dynamic_loading.html
*/
impl PluginManager {
    pub fn new() -> Self {
        Self { plugins: Vec::new(), loaded_libraries: Vec::new() }
    }

    pub(crate) unsafe fn load_single_plugin<'help>(
        &mut self,
        app: clap::App<'help>,
        path: &str,
    ) -> Result<clap::App<'help>, (clap::App<'help>, Error)> {
        #[allow(improper_ctypes_definitions)] // legacy Rust-trait-object plugin ABI
        type PluginCreate<'help> =
            unsafe extern "C" fn(*mut clap::App<'help>) -> (*mut clap::App<'help>, *mut dyn Plugin, *const Error);
        type EnableRawNonceV1 = unsafe extern "C" fn() -> u64;

        let lib = match Library::new(path) {
            Ok(l) => l,
            Err(e) => return Err((app, e.to_string().into())),
        };

        self.loaded_libraries.push(lib); // Save library so it persists in memory
        let lib = self.loaded_libraries.last().unwrap();

        // Optional and ABI-safe: old plugins do not expose this symbol. Seed a current plugin's
        // private atomic before its constructor installs a logger, then retain the plain function
        // pointer so the dashboard can switch it without mutating the process environment.
        let tui_log_control = lib
            .get::<PluginTuiLogControl>(b"keryx_plugin_set_tui_active_v1")
            .ok()
            .map(|control| *control);
        if let Some(control) = tui_log_control {
            control(u8::from(keryx_plugin_api::tui_requested()));
        }

        // Optional, ABI-safe capability negotiation. Absence means the historical zero sentinel,
        // so an old plugin beside this host remains safe. The new CUDA plugin only exposes raw MAX
        // after this call; without it, it translates MAX back to zero for an old host.
        let no_winner = match lib.get::<EnableRawNonceV1>(b"keryx_plugin_enable_raw_nonce_v1") {
            Ok(enable) => enable(),
            Err(_) => 0,
        };

        let constructor: Symbol<PluginCreate> = match lib.get(b"_plugin_create") {
            Ok(cons) => cons,
            Err(e) => return Err((app, e.to_string().into())),
        };

        let (app, boxed_raw, error) = constructor(Box::into_raw(Box::new(app)));
        let app = *Box::from_raw(app);

        if boxed_raw.is_null() {
            return Err((app, *Box::from_raw(error as *mut Error)));
        }
        let plugin = Box::from_raw(boxed_raw);
        self.plugins.push(ManagedPlugin { plugin, no_winner, tui_log_control, dynamic: true });

        Ok(app)
    }

    /// Register an in-process (statically linked) plugin instead of dlopening a
    /// .so. Used by the `static-cuda` build. `augment` merges the plugin's clap
    /// args into the app, mirroring what `_plugin_create` does on the dynamic
    /// path.
    pub fn register_builtin<'help>(
        &mut self,
        app: clap::App<'help>,
        plugin: Box<dyn Plugin>,
        augment: impl FnOnce(clap::App<'help>) -> clap::App<'help>,
    ) -> clap::App<'help> {
        let app = augment(app);
        self.plugins.push(ManagedPlugin { plugin, no_winner: 0, tui_log_control: None, dynamic: false });
        app
    }

    /// Register an in-process plugin whose worker output uses a negotiated no-winner sentinel.
    /// Dynamic plugins discover this through the optional symbol above; static CUDA calls the same
    /// negotiation function and passes its result here.
    pub fn register_builtin_with_output_contract<'help>(
        &mut self,
        app: clap::App<'help>,
        plugin: Box<dyn Plugin>,
        no_winner: u64,
        augment: impl FnOnce(clap::App<'help>) -> clap::App<'help>,
    ) -> clap::App<'help> {
        let app = augment(app);
        self.plugins.push(ManagedPlugin { plugin, no_winner, tui_log_control: None, dynamic: false });
        app
    }

    /// A legacy dynamic plugin can write directly to stderr and corrupt the alternate screen. The
    /// host therefore enables the dashboard only when every loaded dynamic plugin implements the
    /// optional atomic logger switch. Built-in plugins share the host logger and need no hook.
    pub fn supports_tui_logging(&self) -> bool {
        self.plugins.iter().all(|managed| !managed.dynamic || managed.tui_log_control.is_some())
    }

    /// Switch every dynamic plugin logger through its DSO-local atomic control hook.
    pub fn set_plugin_tui_logging(&self, active: bool) {
        for control in self.plugins.iter().filter_map(|managed| managed.tui_log_control) {
            // SAFETY: the pointer came from a successfully loaded library that is retained in
            // `loaded_libraries` for at least as long as this manager.
            unsafe { control(u8::from(active)) };
        }
    }

    /// Copy controls into the dashboard guard. Function pointers remain valid because the guard is
    /// declared after, and therefore dropped before, the owning PluginManager in the host.
    pub fn tui_log_controls(&self) -> Vec<PluginTuiLogControl> {
        self.plugins.iter().filter_map(|managed| managed.tui_log_control).collect()
    }

    pub fn build(&self) -> Result<Vec<ManagedWorkerSpec>, Error> {
        let mut specs = Vec::<ManagedWorkerSpec>::new();
        for managed in &self.plugins {
            if managed.plugin.enabled() {
                specs.extend(
                    managed
                        .plugin
                        .get_worker_specs()
                        .into_iter()
                        .map(|inner| ManagedWorkerSpec { inner, no_winner: managed.no_winner }),
                );
            }
        }
        Ok(specs)
    }

    /**
    Process the options for a plugin, and reports how many workers are available
    */
    pub fn process_options(&mut self, matchs: &ArgMatches) -> Result<usize, Error> {
        let mut count = 0usize;
        self.plugins.iter_mut().for_each(|managed| {
            count += match managed.plugin.process_option(matchs) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!(
                        "WARNING: Failed processing options for {} (ignore if you do not intend to use): {}",
                        managed.plugin.name(),
                        e
                    );
                    0
                }
            }
        });
        // The OpenCL PoM/inference driver lives in the host library, while CLI selection lives in
        // the worker plugin. Publish the plugin's exact post-filter `cl_device_id` list before any
        // VRAM sizing or inference placement runs. Global OpenCL enumeration is not equivalent on
        // mixed-platform rigs or with `--opencl-device` subsets/reordering.
        #[cfg(feature = "pom-opencl")]
        crate::pom_opencl::set_selected_worker_devices(
            self.plugins
                .iter()
                .filter(|managed| managed.plugin.enabled())
                .flat_map(|managed| managed.plugin.get_worker_specs())
                .filter_map(|spec| spec.opencl_device_id())
                .collect(),
        );
        Ok(count)
    }

    pub fn has_specs(&self) -> bool {
        !self.plugins.is_empty()
    }
}

pub fn load_plugins<'help>(
    app: clap::App<'help>,
    paths: &[String],
) -> Result<(clap::App<'help>, PluginManager), Error> {
    let mut factory = PluginManager::new();
    let mut app = app;
    for path in paths {
        app = unsafe {
            factory.load_single_plugin(app, path.as_str()).unwrap_or_else(|(app, e)| {
                eprintln!("WARNING: Failed loading plugin {} (ignore if you do not intend to use): {}", path, e);
                app
            })
        };
    }
    Ok((app, factory))
}
