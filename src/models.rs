/// Registry of supported inference models.
///
/// model_id = sha2-256(primary_weight_file) = CIDv0_bytes[2..34].
/// Verifiable: decode the weight CID from base58btc, skip the 2-byte multihash prefix.
///
/// Uncensored lineup (4 tiers / 4 model families):
///   --light       Gemma-3-4B-it-abliterated     (Google)  — any GPU (6 GB+)
///   (default)     Dolphin-3.0-Llama-3.1-8B       (Llama)  — RTX 3060 12GB / 3070
///   --high        Qwen3-32B-abliterated (Q4_K_M) (Qwen)   — 24 GB (3090 / 4090 / 5090)
///   --very-high   Llama-3.3-70B-abliterated      (Meta)   — 48 GB single-GPU
///
/// All GGUF weights + tokenizers are pinned on the Keryx IPFS gateway; each
/// model_id = base58-decode(weight CID)[2..34].

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Full-precision safetensors (one or more shards).
    Safetensors,
    /// GGUF quantized — LLaMA/LLaMA3 architecture.
    Gguf,
    /// GGUF quantized — Qwen2 architecture (legacy DeepSeek-R1-32B, pre-OPoI-v2 lineup).
    GgufQwen2,
    /// GGUF quantized — Qwen3 architecture (Qwen3-32B).
    GgufQwen3,
    /// GGUF quantized — Gemma 3 architecture (Gemma-3-4B, baseline tier).
    GgufGemma3,
    // ── H4 lineup formats (llama.cpp-served; candle cannot run these archs) ──
    /// GGUF quantized — EXAONE 4 architecture (H4 tier 0). llama-served.
    GgufExaone4,
    /// GGUF quantized — GLM 4 architecture (H4 tier 2). llama-served.
    GgufGlm4,
    /// GGUF quantized — Qwen3.5 hybrid-SSM architecture (H4 tier 3, Qwen3.6-27B). llama-served.
    GgufQwen35,
    /// GGUF quantized — Kimi-Linear MoE architecture (H4 tier 4). llama-served.
    GgufKimiLinear,
    /// GGUF quantized — Gemma 4 architecture (H6 tier 2). llama-served.
    GgufGemma4,
}

#[derive(Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    /// 32-byte on-chain identifier embedded in AiRequest payloads.
    pub model_id: [u8; 32],
    pub format: ModelFormat,
    pub tokenizer_cid: &'static str,
    /// Unused for GGUF (architecture embedded in file).
    pub config_cid: &'static str,
    /// Safetensors: one entry per shard. GGUF: single entry.
    pub weight_cids: &'static [&'static str],
    /// Local directory name under `<exe_dir>/models/`.
    pub dir_name: &'static str,
    /// Minimum VRAM (MB) required to actually serve this model: weights +
    /// KV cache + CUDA workspace. Used by the OPoI capability gate so `ai:cap`
    /// never announces a model the miner cannot load. 0 = never gated.
    pub min_vram_mb: u64,
}

pub const GLM_4_9B_0414: ModelSpec = ModelSpec {
    name: "glm-4-9b-0414",
    model_id: [
        0xfa, 0x2f, 0x13, 0xbe, 0x08, 0x50, 0xe2, 0x6c,
        0x5c, 0xe8, 0x6c, 0x7a, 0xc7, 0x9d, 0xa8, 0x5e,
        0x30, 0x0c, 0x1d, 0xa8, 0xb3, 0x29, 0x0f, 0x9a,
        0x18, 0xd4, 0x71, 0x05, 0xf1, 0xf2, 0x14, 0x0a,
    ],
    format: ModelFormat::GgufGlm4,
    tokenizer_cid: "",
    config_cid: "",
    weight_cids: &["QmfBGGZumBR4XGFLLPjYozvhRSt3kXjrgsV3jXciCdAeM7"],
    dir_name: "GLM-4-9B-0414",
    min_vram_mb: 12_000,
};

pub const QWEN3_6_27B: ModelSpec = ModelSpec {
    name: "qwen3.6-27b",
    model_id: [
        0xb8, 0xbd, 0xc0, 0x1f, 0xa4, 0x07, 0xea, 0xb9,
        0x43, 0xe4, 0xfe, 0xfc, 0x80, 0x74, 0x83, 0xb3,
        0x9f, 0x81, 0x42, 0x78, 0x52, 0x56, 0x04, 0x9e,
        0x1f, 0x55, 0x96, 0x98, 0xa5, 0x28, 0x47, 0x46,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    config_cid: "",
    weight_cids: &["QmamoYQGGAkBaqiWuNmwxeC9AQnt9F7sLyX57VoqbJWeUV"],
    dir_name: "Qwen3.6-27B",
    min_vram_mb: 24_000,
};

pub const KIMI_LINEAR_48B: ModelSpec = ModelSpec {
    name: "kimi-linear-48b",
    model_id: [
        0x3d, 0xc0, 0x93, 0x58, 0xad, 0x75, 0xc6, 0xef,
        0x0c, 0x9c, 0x86, 0xee, 0x4f, 0x47, 0xc4, 0xd6,
        0xac, 0xda, 0x96, 0x1f, 0xec, 0xbd, 0x0e, 0x4f,
        0x9c, 0xf5, 0x5e, 0x8f, 0x0f, 0xdf, 0xfd, 0xdb,
    ],
    format: ModelFormat::GgufKimiLinear,
    tokenizer_cid: "",
    config_cid: "",
    weight_cids: &["QmSVhtoNrL8bWJXZuEXMMWqty8qHScQMRuacuoa9ujsYqp"],
    dir_name: "Kimi-Linear-48B",
    min_vram_mb: 30_000,
};

// ── H6 lineup additions ─────────────────────────────────────────
// Active at `crate::pom::pom_v3_activation_daa()` (the H6 hardfork, matrix-walk era). Five tiers,
// mirror of the node's `POM_TIERS_H6`: tier 0 = Qwen3.5-9B (replaces BOTH Qwen3-8B and Mistral-7B),
// tier 1 = GLM-9B (slides from position 2), tier 2 = Gemma-4-12B (NEW, 16 GB cards), tiers 3-4
// unchanged. `model_id`s MUST equal the node's POM_TIERS_H6 (CIDv0[2..34] of the pinned GGUFs).

/// H6 tier-0 model — Qwen3.5-9B-abliterated Q5_K_M (huihui-ai abliteration, mradermacher GGUF).
pub const QWEN3_5_9B_ABLITERATED: ModelSpec = ModelSpec {
    name: "qwen3.5-9b-abliterated",
    model_id: [
        0xbd, 0x34, 0x56, 0x8c, 0xd8, 0x9f, 0x5f, 0x19,
        0xc6, 0xc3, 0xa6, 0xe1, 0xa6, 0x1b, 0x92, 0x9b,
        0xc8, 0x68, 0x70, 0x94, 0x09, 0xea, 0xad, 0x8e,
        0x67, 0x2d, 0x85, 0xf3, 0xc1, 0xeb, 0x57, 0x10,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    config_cid: "",
    weight_cids: &["Qmb5E3zospd78SfiRHB9iZWNz29xuwRJufieZbWzEFBuGB"],
    dir_name: "Qwen3.5-9B-abliterated",
    // ~6.5 GB Q5_K_M weights + ~1.3 GB KV/workspace → 8 GB card.
    min_vram_mb: 8_000,
};

/// H6 tier-2 model — gemma-4-12B-it-abliterated Q6_K (huihui-ai abliteration, mradermacher GGUF).
pub const GEMMA_4_12B_ABLITERATED: ModelSpec = ModelSpec {
    name: "gemma-4-12b-abliterated",
    model_id: [
        0x39, 0x99, 0x84, 0x04, 0x56, 0x00, 0xf7, 0xd5,
        0x8d, 0x1b, 0x2c, 0xf0, 0x1e, 0x6a, 0x4b, 0xf4,
        0x66, 0xfa, 0x15, 0xc7, 0xac, 0x31, 0xbd, 0x0d,
        0xd1, 0xa7, 0x1e, 0x00, 0x3b, 0x61, 0x7c, 0xc6,
    ],
    format: ModelFormat::GgufGemma4,
    tokenizer_cid: "",
    config_cid: "",
    weight_cids: &["QmSDVicqRDwitecBaPitHsAePLUEamgL4KfrBWYHVWQyx9"],
    // 15 GB, matching the Default auto-select floor (Tier::pom_tier_floor_mb). This abliterated
    // Gemma-4-12B is UNTIED, so the zero-dup llama engine hosts walk + inference in ONE resident copy
    // (~9.1 GB weights + KV + CUDA workspace ≈ 12-13 GB) — fits a 16 GB card. The capability gate
    // (main.rs) compares against FREE VRAM (~15.8 GB on a 16 GB card after display), so 15 GB (not
    // upstream's nominal 16 GB) is what actually lets a 5070 Ti/5080 announce+load Gemma. (The old
    // 20 GB was from an OOM on the pre-v0.10.6 candle path, which loaded a SECOND full copy.)
    dir_name: "Gemma-4-12B-abliterated",
    min_vram_mb: 15_000,
};

/// Whether `model_id` is one of the Proof-of-Model tier models (any era). DAA-independent —
/// used at startup to pick a mineable PoM model before any block DAA is known (the tier *index*
/// is then computed per block via `pom_tier_index`).
pub fn is_pom_model(model_id: &[u8; 32]) -> bool {
    *model_id == QWEN3_5_9B_ABLITERATED.model_id
        || *model_id == GLM_4_9B_0414.model_id
        || *model_id == GEMMA_4_12B_ABLITERATED.model_id
        || *model_id == QWEN3_6_27B.model_id
        || *model_id == KIMI_LINEAR_48B.model_id
}

/// Mirror of the node's per-block tier table (`POM_TIERS_H6`), recomputed from the block DAA.
/// A wrong index → wrong reward bracket → BadWeightPath / divergence, so this MUST match the node.
/// Below the gate this binary refuses to mine (None) — it never produces a pre-H6-era block.
///
/// Only the H6 lineup exists here: the pre-H6 tables (H4/H5, H2 5-tier, pre-H2 4-tier) and their
/// retired models were removed — `pom_v3_activation_daa()` is 0, so those arms were unreachable for
/// every possible DAA, and this binary is an H10-era miner regardless.
pub fn pom_tier_index(model_id: &[u8; 32], daa: u64) -> Option<u8> {
    if daa < crate::pom::pom_v3_activation_daa() {
        return None;
    }
    if *model_id == QWEN3_5_9B_ABLITERATED.model_id {
        Some(0)
    } else if *model_id == GLM_4_9B_0414.model_id {
        Some(1)
    } else if *model_id == GEMMA_4_12B_ABLITERATED.model_id {
        Some(2)
    } else if *model_id == QWEN3_6_27B.model_id {
        Some(3)
    } else if *model_id == KIMI_LINEAR_48B.model_id {
        Some(4)
    } else {
        None
    }
}

/// OPoI v2 hardfork activation DAA score. MUST match the node's `opoi_v2_activation`.
/// Below this score the miner runs/announces the legacy lineup; at or above it, the
/// uncensored lineup. Mainnet: 37_780_000 (2026-06-26 18:00 UTC) — same H as the node's
/// MAINNET_PARAMS.opoi_v2_activation = new(37_780_000).
pub const OPOI_V2_ACTIVATION_DAA: u64 = 37_780_000;

/// Effective OPoI v2 (lineup) activation DAA. Defaults to the consensus constant. STAGING ONLY:
/// when the PoM PoW activation is overridden (`KERYX_POM_ACTIVATION_DAA`), the lineup activation
/// moves to match it, so both the PoW switch and the v2 model swap fire together. This lets a
/// patched-low-`pom_activation` testnet exercise the FULL post-fork path (PoM-PoW + v2 weights
/// resident + proof) at low DAA. Production (no override) is byte-identical to the constant.
pub fn opoi_v2_activation_daa() -> u64 {
    if crate::pom::is_activation_overridden() {
        crate::pom::activation_daa()
    } else {
        OPOI_V2_ACTIVATION_DAA
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    VeryLight,
    Light,
    Default,
    High,
    VeryHigh,
}

impl Tier {
    /// Tiers from largest to smallest — used by `--tier auto` to pick the biggest that fits.
    pub const DESCENDING: [Tier; 5] = [Tier::VeryHigh, Tier::High, Tier::Default, Tier::Light, Tier::VeryLight];

    /// Per-tier VRAM floor (MiB) for `--tier auto` — the practical minimum to load that tier's model
    /// (weights + KV cache + CUDA workspace) via the zero-dup llama engine (ONE resident copy; every
    /// H6 model is untied). Imported VERBATIM from upstream keryx-miner v0.4.8's `pom_tier_ladder`
    /// so our auto-selection uses the same field-proven boundaries — notably Gemma-4-12B ("default")
    /// runs on a 16 GB card (5070 Ti / 5080 / 4080), where our old `min_vram+2GB` math wrongly
    /// demoted it to GLM. The floor IS the final threshold (margin already baked in — no extra
    /// headroom added, matching upstream).
    pub fn pom_tier_floor_mb(self) -> u64 {
        match self {
            Tier::VeryLight => 7_000,
            Tier::Light => 11_000,
            Tier::Default => 15_000,
            Tier::High => 22_000,
            Tier::VeryHigh => 28_000,
        }
    }

    /// Human-readable name of the model this tier mines/proves under the OPoI-v2 (PoM) lineup.
    pub fn pom_model_name(self) -> &'static str {
        self.pom_spec().name
    }

    /// The single PoM model spec this tier proves possession of — the H6 lineup, mirroring
    /// `specs_for` and the node's `POM_TIERS_H6`. Drives startup staging + `auto_select_tier`.
    pub fn pom_spec(self) -> &'static ModelSpec {
        match self {
            Tier::VeryLight => &QWEN3_5_9B_ABLITERATED,
            Tier::Light => &GLM_4_9B_0414,
            Tier::Default => &GEMMA_4_12B_ABLITERATED,
            Tier::High => &QWEN3_6_27B,
            Tier::VeryHigh => &KIMI_LINEAR_48B,
        }
    }
}

/// `--tier auto`: pick the LARGEST tier whose footprint fits the GPU's VRAM, with a conservative
/// safety margin so the chosen tier loads cleanly (weights + PoM possession walk + CUDA workspace
/// + KV cache for GPU inference). Returns the tier and its budgeted MiB requirement.
///
/// The budget is the model's `min_vram_mb` (which already accounts for weights + KV + workspace),
/// plus a `headroom_mb` margin on top. Empirically an 8 GB 3070 OOMs Gemma-3-4B on the GPU
/// (needs `--cpu-inference`), so the margin must be conservative: with the default 2 GB headroom,
/// Light (min_vram_mb=0) is the only tier that fits an 8 GB card, and Default (needs 8000) does
/// NOT — which is the correct, OOM-safe choice.
///
/// `cpu_inference`: when true, GPU inference is off, so the GPU only needs to hold the PoM walk's
/// resident weights (no inference KV/workspace), but we keep the same conservative margin.
pub fn auto_select_tier(vram_mb: u64, _headroom_mb: u64) -> (Tier, u64) {
    // Upstream-parity floors (`Tier::pom_tier_floor_mb`) ARE the threshold — margin baked in, no
    // extra headroom added (that is what wrongly pushed Gemma to 22 GB before). `_headroom_mb` is
    // kept only for call-site compatibility. Largest tier whose floor the card meets wins.
    for tier in Tier::DESCENDING {
        let floor = tier.pom_tier_floor_mb();
        if vram_mb >= floor {
            return (tier, floor);
        }
    }
    // Card below even the VeryLight floor — still floor to VeryLight (smallest model), matching
    // upstream's fallback: it may be tight but it's the only tier that could possibly load.
    (Tier::VeryLight, Tier::VeryLight.pom_tier_floor_mb())
}

/// The model set for a hardware tier — the H6 lineup (node `POM_TIERS_H6`): one model per tier,
/// tier 0 Qwen3.5-9B, 1 GLM-9B, 2 Gemma-4-12B, 3 Qwen3.6-27B, 4 Kimi-Linear-48B. Below the gate
/// there is nothing to mine, matching `pom_tier_index`.
pub fn specs_for(daa: u64, tier: Tier) -> &'static [&'static ModelSpec] {
    if daa < crate::pom::pom_v3_activation_daa() {
        return &[];
    }
    match tier {
        Tier::VeryLight => &[&QWEN3_5_9B_ABLITERATED],
        Tier::Light => &[&GLM_4_9B_0414],
        Tier::Default => &[&GEMMA_4_12B_ABLITERATED],
        Tier::High => &[&QWEN3_6_27B],
        Tier::VeryHigh => &[&KIMI_LINEAR_48B],
    }
}

/// The H6 lineup — resolves a model name/id. Retired pre-H6 models are gone: they are not mineable
/// (`pom_tier_index` returns None for them), so keeping them only invited staging a model the node
/// would reject.
pub const REGISTRY: &[&ModelSpec] = &[
    &QWEN3_5_9B_ABLITERATED,
    &GLM_4_9B_0414,
    &GEMMA_4_12B_ABLITERATED,
    &QWEN3_6_27B,
    &KIMI_LINEAR_48B,
];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The H6 per-block tier table — mirror of the node's `POM_TIERS_H6` order. `u64::MAX` sits
    /// at/after every gate on any network, so this exercises the H6 branch without touching the
    /// global testnet switch.
    #[test]
    fn h6_tier_table_mirrors_node() {
        let daa = u64::MAX;
        assert_eq!(pom_tier_index(&QWEN3_5_9B_ABLITERATED.model_id, daa), Some(0));
        assert_eq!(pom_tier_index(&GLM_4_9B_0414.model_id, daa), Some(1));
        assert_eq!(pom_tier_index(&GEMMA_4_12B_ABLITERATED.model_id, daa), Some(2));
        assert_eq!(pom_tier_index(&QWEN3_6_27B.model_id, daa), Some(3));
        assert_eq!(pom_tier_index(&KIMI_LINEAR_48B.model_id, daa), Some(4));
        // The hardware-tier -> model map agrees with the table, tier for tier.
        assert_eq!(specs_for(daa, Tier::VeryLight)[0].model_id, QWEN3_5_9B_ABLITERATED.model_id);
        assert_eq!(specs_for(daa, Tier::Light)[0].model_id, GLM_4_9B_0414.model_id);
        assert_eq!(specs_for(daa, Tier::Default)[0].model_id, GEMMA_4_12B_ABLITERATED.model_id);
        assert_eq!(specs_for(daa, Tier::High)[0].model_id, QWEN3_6_27B.model_id);
        assert_eq!(specs_for(daa, Tier::VeryHigh)[0].model_id, KIMI_LINEAR_48B.model_id);

        // Every registry model is a mineable tier, and every tier's model is in the registry.
        for spec in REGISTRY {
            assert!(is_pom_model(&spec.model_id), "{} is not a PoM model", spec.name);
            assert!(pom_tier_index(&spec.model_id, daa).is_some(), "{} has no tier", spec.name);
        }
        for tier in Tier::DESCENDING {
            let s = specs_for(daa, tier)[0];
            assert!(REGISTRY.iter().any(|r| r.model_id == s.model_id), "{} not in REGISTRY", s.name);
        }

        // An unknown model is never mineable, at any DAA.
        assert_eq!(pom_tier_index(&[0u8; 32], daa), None);
        assert!(!is_pom_model(&[0u8; 32]));
    }
}
