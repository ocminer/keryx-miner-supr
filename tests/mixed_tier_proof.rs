//! Mixed-tier rig (one process, different models per card): every card's PoM v5 share proof must
//! be built from ITS OWN model's index and pad, and verify against what the pool/node derives for
//! that tier — `v5_pad_seed(tier, model_id_of_tier, R_T_of_tier)`.
//!
//! Regression for the 2026-11-08 pool report (GPUs 0,1 tier 3 on the shared/default index, GPU 2
//! tier 0 on a per-card index; every tier-3 share rejected with `v5 verify: BadTilePath`): building
//! the per-card tier-0 index (or any later index: era prebuild, trace opening) overwrote the
//! process-wide "mining model id", so the cards on the shared index seeded their pad with the
//! tier-0 model. GPU walk and host self-check agreed with each other, the node did not.
//!
//! CPU only (small synthetic GGUFs, no GPU). Runs in its own process because the shared index is a
//! process-wide `OnceLock`, exactly as in the miner.

use keryx_miner::pom::{self, WeightIndex};
use keryx_miner::pom_v5;

const MODEL_T3: [u8; 32] = [0xa3; 32]; // primary/default model (shared index), GPUs 0 and 1
const MODEL_T0: [u8; 32] = [0xb0; 32]; // lighter card's own model (per-device index), GPU 2
const PAD_TILES: u64 = 257; // small non-power-of-two pad: most walk steps hit the pad

/// Small standalone GGUF (two F32 tensors), content keyed by `salt`.
fn write_fixture(path: &std::path::Path, salt: u64) {
    let mut data = b"GGUF".to_vec();
    data.extend(3u32.to_le_bytes());
    data.extend(2u64.to_le_bytes());
    data.extend(0u64.to_le_bytes());
    let mut offset = 0u64;
    for (name, elements) in [("a.weight", 8200u64), ("b.weight", 8184u64)] {
        data.extend((name.len() as u64).to_le_bytes());
        data.extend(name.as_bytes());
        data.extend(1u32.to_le_bytes());
        data.extend(elements.to_le_bytes());
        data.extend(0u32.to_le_bytes());
        data.extend(offset.to_le_bytes());
        offset += elements * 4;
    }
    data.resize((data.len() + 31) / 32 * 32, 0);
    let mut x = 0x8a12_7d0b_936f_aa15u64 ^ salt;
    for _ in 0..offset {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        data.push(x as u8);
    }
    std::fs::write(path, data).unwrap();
}

fn build(dir: &std::path::Path, name: &str, salt: u64, model: [u8; 32]) -> WeightIndex {
    let d = dir.join(name);
    std::fs::create_dir_all(&d).unwrap();
    let gguf = d.join("model.gguf");
    write_fixture(&gguf, salt);
    WeightIndex::build_from_gguf(gguf.to_str().unwrap(), model).unwrap()
}

/// The share-proof path of `src/pow.rs` (`generate_block_if_pom_task`) for a nonce found by
/// `device`: its index + tier, the pad seeded via `pom::v5_pad_seed_for`, then the node's verify
/// with the pad the NODE derives for the tier's model. Returns Err on BadTilePath-class failures.
fn share_verifies_at_node(device: u32, node_model: &[u8; 32], seed: u64) -> Result<(), String> {
    let owner = pom::active_index_for(device).ok_or("no index")?;
    let (index, tier) = (&owner.0, owner.1);
    let miner_pad = pom::v5_pad_seed_for(device, tier, &index.r_t).ok_or("no mining model id")?;
    let (proof, final_state) = pom_v5::build_proof_v5(tier, seed, index, &miner_pad, PAD_TILES).map_err(|e| e.to_string())?;
    // Miner pre-submit self-check (same pad as the walk) — always passes, which is why the bug
    // was invisible on the rig.
    let self_check = pom_v5::verify_proof_v5(seed, &proof, &index.r_t, index.n_chunks, &miner_pad, PAD_TILES)
        .map_err(|e| e.to_string())?;
    assert_eq!(self_check, final_state);
    // Node/pool: pad derived from the tier's pinned model and R_T.
    let node_pad = pom_v5::v5_pad_seed(tier, node_model, &index.r_t);
    match pom_v5::verify_proof_v5(seed, &proof, &index.r_t, index.n_chunks, &node_pad, PAD_TILES) {
        Ok(s) if s == final_state => Ok(()),
        Ok(_) => Err("node final_state mismatch".into()),
        Err(e) => Err(format!("node verify: {e}")),
    }
}

#[test]
fn mixed_tier_rig_builds_each_cards_proof_from_its_own_model() {
    let dir = std::env::temp_dir().join(format!("keryx-mixed-tier-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // Startup order seen in the field: the primary (tier 3) shared index is installed first; the
    // lighter card's model is prefetched afterwards, so its per-device index is built LAST.
    let idx_t3 = build(&dir, "t3", 0x33, MODEL_T3);
    let r_t3 = idx_t3.r_t;
    pom::set_index(idx_t3, 3, MODEL_T3);
    let idx_t0 = build(&dir, "t0", 0x00, MODEL_T0);
    let r_t0 = idx_t0.r_t;
    pom::set_index_for(2, MODEL_T0, idx_t0, 0);
    // Further index builds of the tier-0 model (era prebuild / trace opening) must not matter either.
    let _opening = build(&dir, "t0-opening", 0x00, MODEL_T0);
    assert_ne!(r_t3, r_t0);

    for (device, model, tier, r_t) in [(0u32, MODEL_T3, 3u8, r_t3), (1, MODEL_T3, 3, r_t3), (2, MODEL_T0, 0, r_t0)] {
        let owner = pom::active_index_for(device).unwrap();
        assert_eq!((owner.1, owner.0.r_t), (tier, r_t), "GPU {device}: wrong index/tier");
        assert_eq!(pom::mining_model_id(device), Some(model), "GPU {device}: pad seeded from another card's model");
        for seed in [0u64, 1, 0x5eed, 0xdead_beef_0123_4567, u64::MAX] {
            share_verifies_at_node(device, &model, seed)
                .unwrap_or_else(|e| panic!("GPU {device} tier {tier} seed {seed:#x}: share rejected: {e}"));
        }
    }

    // The test discriminates: a tier-3 proof whose pad is seeded with the tier-0 model (the bug)
    // is rejected by the node's verify for at least one of these seeds.
    let owner = pom::active_index_for(0).unwrap();
    let wrong_pad = pom_v5::v5_pad_seed(3, &MODEL_T0, &owner.0.r_t);
    let node_pad = pom_v5::v5_pad_seed(3, &MODEL_T3, &owner.0.r_t);
    let rejected = (0u64..8).filter(|&seed| {
        let (proof, fs) = pom_v5::build_proof_v5(3, seed, &owner.0, &wrong_pad, PAD_TILES).unwrap();
        pom_v5::verify_proof_v5(seed, &proof, &owner.0.r_t, owner.0.n_chunks, &node_pad, PAD_TILES).map_or(true, |s| s != fs)
    });
    assert!(rejected.count() > 0, "wrong-model pad was not detected by the node verify");

    let _ = std::fs::remove_dir_all(&dir);
}
