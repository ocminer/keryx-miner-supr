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

pub const GEMMA_3_4B: ModelSpec = ModelSpec {
    name: "gemma-3-4b",
    // CIDv0[2..34] of model.gguf — mlabonne/gemma-3-4b-it-abliterated Q4_K_M
    model_id: [
        0xad, 0x50, 0xad, 0x0b, 0xd4, 0x61, 0xd8, 0xab,
        0x44, 0xef, 0xc0, 0x21, 0x49, 0x89, 0xeb, 0x33,
        0x29, 0x16, 0x85, 0xef, 0x4a, 0xde, 0x22, 0xa0,
        0xf4, 0xf2, 0x17, 0xd0, 0x32, 0x66, 0xd8, 0x37,
    ],
    format: ModelFormat::GgufGemma3,
    tokenizer_cid: "QmTh2MsVfAvWp7grN9rvkF9NkMkCW2PhWez2WbNh81KRXD",
    config_cid: "",
    weight_cids: &["Qma1CbFzWTNhy2ReVjDG1GvM5q2Uy4VhqTbnS9c641jUQ6"],
    dir_name: "Gemma-3-4B",
    // Baseline model — never gated. ~4 GB Q4_K_M; runs on any GPU that can mine.
    min_vram_mb: 0,
};

pub const DOLPHIN_LLAMA3_8B: ModelSpec = ModelSpec {
    name: "dolphin-llama3-8b",
    // CIDv0[2..34] of model.gguf — Dolphin3.0-Llama3.1-8B Q4_K_M
    model_id: [
        0x94, 0x21, 0x06, 0x6a, 0x64, 0x00, 0xc9, 0x8b,
        0xa1, 0x37, 0x11, 0x4f, 0x7f, 0x4b, 0x7d, 0x4a,
        0x2d, 0xdf, 0x13, 0xab, 0x16, 0x3a, 0x5d, 0xe3,
        0x8c, 0x01, 0x84, 0x79, 0x3a, 0xf6, 0x31, 0x3a,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmQSe8rZQcTQ6q1xDGquv6s9wpzFT9u27U4wfGVZqwMJgJ",
    config_cid: "",
    weight_cids: &["QmYJtFpaDnVwAVSbzRo42fsb19nLpt8LHe8WVKoyxd4AkZ"],
    dir_name: "Dolphin-Llama3-8B",
    // ~4.9 GB Q4_K_M weights + ~1.6 GB KV/workspace.
    min_vram_mb: 8_000,
};

pub const QWEN3_32B: ModelSpec = ModelSpec {
    name: "qwen3-32b",
    // CIDv0[2..34] of model.gguf — Qwen3-32B-abliterated Q4_K_M (mradermacher)
    model_id: [
        0x65, 0xc6, 0xeb, 0x6f, 0xe1, 0x8b, 0x9e, 0xfd,
        0x80, 0x60, 0xab, 0x9d, 0x2d, 0x03, 0xbb, 0x9b,
        0x01, 0x05, 0x0a, 0x3b, 0x13, 0x78, 0xcb, 0xac,
        0x00, 0x0c, 0x5c, 0xc0, 0xac, 0xdc, 0x0d, 0x2a,
    ],
    format: ModelFormat::GgufQwen3,
    tokenizer_cid: "QmcuGkJvR343ry3b4jy7u5L9ior3ujas3yGAFMSyZdACb5",
    config_cid: "",
    weight_cids: &["QmVBwp5n3muQJwYNLTHSu3EnzBWviQqfh58FvHvKRfLtam"],
    dir_name: "Qwen3-32B",
    // ~19.5 GB Q4_K_M weights + ~2.5 GB KV/workspace → fits a 24 GB card (3090/4090/5090).
    min_vram_mb: 24_000,
};

pub const LLAMA_3_3_70B: ModelSpec = ModelSpec {
    name: "llama-3.3-70b",
    // CIDv0[2..34] of model.gguf — Llama-3.3-70B-Instruct-abliterated Q4_K_M (bartowski)
    model_id: [
        0x13, 0x29, 0xfb, 0xe2, 0x1b, 0x3f, 0x36, 0xf6,
        0xd0, 0x06, 0x89, 0xfc, 0xaa, 0x74, 0xf7, 0xa2,
        0x22, 0xb8, 0xcc, 0x4c, 0x08, 0xc0, 0x19, 0x1f,
        0xeb, 0x23, 0x97, 0x55, 0xa7, 0x23, 0x42, 0x1e,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmPd7WQvoQupfzpPVnVVc1Zra5SH4jKnGqNrdTHFtdQuvd",
    config_cid: "",
    weight_cids: &["QmPdTayXcEsfUwMCoMKKcLSv7Dwpp2xVBWELwrG2M7Rhzu"],
    dir_name: "Llama-3.3-70B",
    // ~42.5 GB Q4_K_M weights + ~3.5 GB KV/workspace → 48 GB card (matches the
    // --very-high 46 GB startup gate).
    min_vram_mb: 46_000,
};

/// Map a model_id to its Proof-of-Model tier index, matching the node's `POM_TIERS` order
/// (Gemma=0, Dolphin=1, Qwen3-32B=2, Llama-70B-abl=3). None for non-PoM models.
pub fn pom_tier_index(model_id: &[u8; 32]) -> Option<u8> {
    if *model_id == GEMMA_3_4B.model_id {
        Some(0)
    } else if *model_id == DOLPHIN_LLAMA3_8B.model_id {
        Some(1)
    } else if *model_id == QWEN3_32B.model_id {
        Some(2)
    } else if *model_id == LLAMA_3_3_70B.model_id {
        Some(3)
    } else {
        None
    }
}

// ── Legacy lineup (pre-OPoI-v2) ───────────────────────────────────────────────
// Served while `daa < OPOI_V2_ACTIVATION_DAA`. model_id values match the node's
// pre-v2 INFERENCE_REWARD_MINIMUMS table (8B/TinyLlama = CID-derived; 32B/70B =
// sha2-256(model.gguf) computed locally). Ported verbatim from the pre-rewrite
// registry so the transition is a true gate, not a re-derivation.

pub const TINYLLAMA: ModelSpec = ModelSpec {
    name: "tinyllama",
    // sha2-256(QmdqcmS8aMngiZWYYdeZEaW22N6XRTd9zK5ZCJG1MPmrQ3)
    model_id: [
        0xe6, 0x4a, 0xf3, 0x68, 0xec, 0x93, 0x51, 0xa5,
        0xa4, 0xc0, 0xec, 0x7a, 0xe4, 0x7d, 0x42, 0xad,
        0xa7, 0xf6, 0xb3, 0xf1, 0xa6, 0xe6, 0x0f, 0xc7,
        0x3d, 0x0e, 0xb6, 0xca, 0x29, 0x53, 0x64, 0x5c,
    ],
    format: ModelFormat::Safetensors,
    tokenizer_cid: "QmSKrRu8HRt9v2dUeVdABKDkuREa5xFhPLZdevvvBfDYmp",
    config_cid: "QmbLTR3GLjBUKw8Lj14isiwG3XZJaL61ES852vkNqNPhyd",
    weight_cids: &["QmdqcmS8aMngiZWYYdeZEaW22N6XRTd9zK5ZCJG1MPmrQ3"],
    dir_name: "TinyLlama-1.1B",
    min_vram_mb: 0,
};

pub const DEEPSEEK_R1_8B: ModelSpec = ModelSpec {
    name: "deepseek-r1-8b",
    // sha2-256(QmYK1faUGNMYZ2UKeSpUoUoFpRarZQEwfPCHbYNG2ib2mR)
    model_id: [
        0x94, 0x29, 0x67, 0x33, 0x16, 0xbc, 0x40, 0xec,
        0x06, 0x67, 0x89, 0x45, 0x34, 0x57, 0x8b, 0x41,
        0x23, 0x6f, 0xc7, 0xee, 0xa4, 0xd9, 0x31, 0xf1,
        0x48, 0x9c, 0x34, 0xc5, 0x83, 0x7f, 0x42, 0xf4,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmXVdcr2FJuHtXcBbYbBuCMic2pJTkM1LJ6WpyfvhDytHg",
    config_cid: "",
    weight_cids: &["QmYK1faUGNMYZ2UKeSpUoUoFpRarZQEwfPCHbYNG2ib2mR"],
    dir_name: "DeepSeek-R1-8B",
    min_vram_mb: 5_500,
};

pub const DEEPSEEK_R1_32B: ModelSpec = ModelSpec {
    name: "deepseek-r1-32b",
    // sha2-256(model.gguf)
    model_id: [
        0xbe, 0xd9, 0xb0, 0xf5, 0x51, 0xf5, 0xb9, 0x5b,
        0xf9, 0xda, 0x58, 0x88, 0xa4, 0x8f, 0x0f, 0x87,
        0xc3, 0x7a, 0xd6, 0xb7, 0x25, 0x19, 0xc4, 0xcb,
        0xd7, 0x75, 0xf5, 0x4a, 0xc0, 0xb9, 0xfc, 0x62,
    ],
    format: ModelFormat::GgufQwen2,
    tokenizer_cid: "Qmf3uZwnuxZUhDbhup8Q51soVMRmNxohYctG9wZemNEPHm",
    config_cid: "",
    weight_cids: &["QmSrmkEoJUPf7r9t4o79F5APycnGrRu2icaU3KKPdFVUk7"],
    dir_name: "DeepSeek-R1-32B",
    min_vram_mb: 20_000,
};

pub const LLAMA_3_3_70B_OFFICIAL: ModelSpec = ModelSpec {
    name: "llama-3.3-70b-official",
    // sha2-256(model.gguf)
    model_id: [
        0xaa, 0xd2, 0xcf, 0x33, 0x48, 0xd8, 0xc7, 0xfd,
        0xbd, 0x2c, 0x0d, 0xd5, 0x8e, 0x0d, 0x99, 0x36,
        0x84, 0x50, 0xd4, 0x3c, 0x95, 0x84, 0xae, 0xf8,
        0x1a, 0x46, 0x7d, 0xd3, 0x47, 0x56, 0x13, 0x44,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmPd7WQvoQupfzpPVnVVc1Zra5SH4jKnGqNrdTHFtdQuvd",
    config_cid: "",
    weight_cids: &["QmbRQJFZ9NuZQW9uXezANTwunnwJCKybHiCFnVQ7D4SZKb"],
    // Distinct dir from the abliterated Llama-3.3-70B so both lineups coexist on disk.
    dir_name: "Llama-3.3-70B-official",
    min_vram_mb: 30_000,
};

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
    Light,
    Default,
    High,
    VeryHigh,
}

impl Tier {
    /// Tiers from largest to smallest — used by `--tier auto` to pick the biggest that fits.
    pub const DESCENDING: [Tier; 4] = [Tier::VeryHigh, Tier::High, Tier::Default, Tier::Light];

    /// Human-readable name of the model this tier mines/proves under the OPoI-v2 (PoM) lineup.
    pub fn pom_model_name(self) -> &'static str {
        self.pom_spec().name
    }

    /// The single OPoI-v2 (PoM) model spec this tier proves possession of.
    pub fn pom_spec(self) -> &'static ModelSpec {
        match self {
            Tier::Light => &GEMMA_3_4B,
            Tier::Default => &DOLPHIN_LLAMA3_8B,
            Tier::High => &QWEN3_32B,
            Tier::VeryHigh => &LLAMA_3_3_70B,
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
pub fn auto_select_tier(vram_mb: u64, headroom_mb: u64) -> (Tier, u64) {
    for tier in Tier::DESCENDING {
        let need = tier.pom_spec().min_vram_mb.saturating_add(headroom_mb);
        if vram_mb >= need {
            return (tier, need);
        }
    }
    // Light has min_vram_mb=0; with any non-trivial card it always fits. Floor to Light.
    (Tier::Light, headroom_mb)
}

/// Cumulative model set for a hardware tier within the lineup active at `daa`.
/// DAA-gated to mirror the node's `opoi_v2_activation`: one binary runs the legacy
/// lineup before H and the uncensored lineup at/after H (hot-swapped at the crossing),
/// so miners can upgrade before the hardfork without a flag-day restart.
pub fn specs_for(daa: u64, tier: Tier) -> &'static [&'static ModelSpec] {
    if daa >= OPOI_V2_ACTIVATION_DAA {
        // PoM era: one flag = one model. Each hardware tier mines AND serves exactly the
        // single model it proves possession of — the cumulative "serve everything below my
        // tier" behaviour is dropped, because a PoM GPU is bound to its tier (serving a
        // lower tier means unloading the mined model and pausing mining). Multi-tier
        // coverage is a network property (different miners per tier), not a per-GPU one.
        match tier {
            Tier::Light => &[&GEMMA_3_4B],
            Tier::Default => &[&DOLPHIN_LLAMA3_8B],
            Tier::High => &[&QWEN3_32B],
            Tier::VeryHigh => &[&LLAMA_3_3_70B],
        }
    } else {
        match tier {
            Tier::Light => &[&TINYLLAMA],
            Tier::Default => &[&TINYLLAMA, &DEEPSEEK_R1_8B],
            Tier::High => &[&TINYLLAMA, &DEEPSEEK_R1_8B, &DEEPSEEK_R1_32B],
            Tier::VeryHigh => &[&TINYLLAMA, &DEEPSEEK_R1_8B, &DEEPSEEK_R1_32B, &LLAMA_3_3_70B_OFFICIAL],
        }
    }
}

/// Both lineups combined — resolves a model name/id regardless of era.
pub const REGISTRY: &[&ModelSpec] = &[
    &GEMMA_3_4B,
    &DOLPHIN_LLAMA3_8B,
    &QWEN3_32B,
    &LLAMA_3_3_70B,
    &TINYLLAMA,
    &DEEPSEEK_R1_8B,
    &DEEPSEEK_R1_32B,
    &LLAMA_3_3_70B_OFFICIAL,
];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}
