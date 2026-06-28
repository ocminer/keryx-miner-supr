// Deterministic GPU-inference smoke test for the zero-dup Gemma3 split path.
// Mirrors exactly what the miner does on an OPoI challenge: force the split loader,
// load Gemma-3-4B onto CUDA:0, and run a real forward. PASS = non-empty text, no OOM.
// Run with CUDA_VISIBLE_DEVICES=<8gb idx> to validate it fits an 8 GB card.
use keryx_miner::models::GEMMA_3_4B;
use keryx_miner::slm;

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    // Register the model so load_and_run_inference can resolve it (the miner does this at boot).
    static SPECS: &[&keryx_miner::models::ModelSpec] = &[&GEMMA_3_4B];
    slm::init_supported(SPECS);
    // Same flag main.rs sets for pom-cuda mining → routes Gemma through the split loader.
    slm::set_pom_force_split(true);
    eprintln!("[test] forcing split loader; loading + running GPU inference for gemma-3-4b…");
    let out = slm::load_and_run_inference(&GEMMA_3_4B.model_id, "In one sentence, what is mining?", 48);
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
