use clap::ArgMatches;
use libloading::{Library, Symbol};

pub mod inference;
pub mod models;
pub mod slm;
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
#[cfg(feature = "pom-opencl")]
pub mod llama_vulkan;
#[cfg(feature = "pom-cuda")]
pub mod pom_gpu;
pub use keryx_plugin_api::{declare_plugin, xoshiro256starstar, Error, Plugin, Worker, WorkerSpec};

#[derive(Default)]
pub struct PluginManager {
    plugins: Vec<Box<dyn Plugin>>,
    loaded_libraries: Vec<Library>,
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
        type PluginCreate<'help> =
            unsafe fn(*const clap::App<'help>) -> (*mut clap::App<'help>, *mut dyn Plugin, *mut Error);

        let lib = match Library::new(path) {
            Ok(l) => l,
            Err(e) => return Err((app, e.to_string().into())),
        };

        self.loaded_libraries.push(lib); // Save library so it persists in memory
        let lib = self.loaded_libraries.last().unwrap();

        let constructor: Symbol<PluginCreate> = match lib.get(b"_plugin_create") {
            Ok(cons) => cons,
            Err(e) => return Err((app, e.to_string().into())),
        };

        let (app, boxed_raw, error) = constructor(Box::into_raw(Box::new(app)));
        let app = *Box::from_raw(app);

        if boxed_raw.is_null() {
            return Err((app, *Box::from_raw(error)));
        }
        let plugin = Box::from_raw(boxed_raw);
        self.plugins.push(plugin);

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
        self.plugins.push(plugin);
        app
    }

    pub fn build(&self) -> Result<Vec<Box<dyn WorkerSpec + 'static>>, Error> {
        let mut specs = Vec::<Box<dyn WorkerSpec + 'static>>::new();
        for plugin in &self.plugins {
            if plugin.enabled() {
                specs.extend(plugin.get_worker_specs());
            }
        }
        Ok(specs)
    }

    /**
    Process the options for a plugin, and reports how many workers are available
    */
    pub fn process_options(&mut self, matchs: &ArgMatches) -> Result<usize, Error> {
        let mut count = 0usize;
        self.plugins.iter_mut().for_each(|plugin| {
            count += match plugin.process_option(matchs) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!(
                        "WARNING: Failed processing options for {} (ignore if you do not intend to use): {}",
                        plugin.name(),
                        e
                    );
                    0
                }
            }
        });
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
