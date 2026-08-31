// Deterministic GPU-inference smoke test for the zero-dup split path.
// Mirrors exactly what the miner does on an OPoI challenge: force the split loader,
// load the smallest supported model onto CUDA:0, and run a real forward.
// PASS = non-empty text, no OOM.
// Run with CUDA_VISIBLE_DEVICES=<8gb idx> to validate it fits an 8 GB card.
//
// Uses the very-light (8 GB) tier model. It was pinned to the retired GEMMA_3_4B until the
// 2026-08-31 dead-fork cleanup removed that spec from the registry, which left this example
// failing to compile and broke `cargo test --workspace`.
use keryx_miner::models::QWEN3_5_9B_ABLITERATED as SMOKE_MODEL;
use keryx_miner::slm;

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    // Register the model so load_and_run_inference can resolve it (the miner does this at boot).
    static SPECS: &[&keryx_miner::models::ModelSpec] = &[&SMOKE_MODEL];
    slm::init_supported(SPECS);
    // Same flag main.rs sets for pom-cuda mining → routes the model through the split loader.
    slm::set_pom_force_split(true);
    eprintln!("[test] forcing split loader; loading + running GPU inference for the very-light model…");
    let out = slm::load_and_run_inference(&SMOKE_MODEL.model_id, "In one sentence, what is mining?", 48);
    match out {
        Some(t) if !t.trim().is_empty() => {
            eprintln!("[test] GPU INFERENCE OK ({} chars):\n{}", t.len(), t);
            std::process::exit(0);
        }
        Some(t) => {
            eprintln!("[test] EMPTY OUTPUT: {:?}", t);
            std::process::exit(2);
        }
        None => {
            eprintln!("[test] INFERENCE RETURNED None (load/run failed — see log above)");
            std::process::exit(3);
        }
    }
}
