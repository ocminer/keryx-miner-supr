/// Registry of supported inference models.
///
/// model_id = sha2-256(primary_weight_file) = CIDv0_bytes[2..34].
/// Verifiable: decode the weight CID from base58btc, skip the 2-byte multihash prefix.
///
/// Current H6/H10 lineup (five tiers):
///   --very-light  Qwen3.5-9B-abliterated         — 7 GB auto-select floor
///   --light       GLM-4-9B-0414                  — 11 GB
///   (default)     Gemma-4-12B-abliterated        — 15 GB
///   --high        Qwen3.6-27B                    — 22 GB  (Qwen3.8-27B from the H14 gate)
///   --very-high   Kimi-Linear-48B                — 28 GB
///
/// H14 (private inference, `crate::pom::private_inference_activation_daa()`): the node's
/// `POM_TIERS_H14` swaps ONLY tier 3 to Qwen3.8-27B (new model_id, root and chunk count). A
/// tier-3 proof walked over Qwen3.6-27B past the gate fails the node's anchor check, so every
/// tier lookup below is era-aware.
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
        0xfa, 0x2f, 0x13, 0xbe, 0x08, 0x50, 0xe2, 0x6c, 0x5c, 0xe8, 0x6c, 0x7a, 0xc7, 0x9d, 0xa8, 0x5e, 0x30, 0x0c,
        0x1d, 0xa8, 0xb3, 0x29, 0x0f, 0x9a, 0x18, 0xd4, 0x71, 0x05, 0xf1, 0xf2, 0x14, 0x0a,
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
        0xb8, 0xbd, 0xc0, 0x1f, 0xa4, 0x07, 0xea, 0xb9, 0x43, 0xe4, 0xfe, 0xfc, 0x80, 0x74, 0x83, 0xb3, 0x9f, 0x81,
        0x42, 0x78, 0x52, 0x56, 0x04, 0x9e, 0x1f, 0x55, 0x96, 0x98, 0xa5, 0x28, 0x47, 0x46,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    config_cid: "",
    weight_cids: &["QmamoYQGGAkBaqiWuNmwxeC9AQnt9F7sLyX57VoqbJWeUV"],
    dir_name: "Qwen3.6-27B",
    min_vram_mb: 24_000,
};

/// H14 tier-3 model (--high), replaces Qwen3.6-27B at the private-inference gate —
/// Huihui-Qwen3.8-27B-abliterated Q4_K (arch qwen35 hybrid-SSM, same llama.cpp arch as 3.6).
/// `model_id` MUST equal the node's `QWEN3_8_27B_MODEL_ID` = CIDv0[2..34] of the pinned GGUF
/// (IPFS QmW7LDz7ZTfw9vpAR9jMhFHWriLhxh728Kihp7oTSLgvyg, 16 810 714 400 bytes).
pub const QWEN3_8_27B: ModelSpec = ModelSpec {
    name: "qwen3.8-27b",
    model_id: [
        0x73, 0x74, 0x0b, 0x44, 0x3b, 0xdc, 0x00, 0xaf, 0xda, 0x5f, 0xa3, 0x4e, 0xb9, 0x99, 0x9d, 0x3f, 0xea, 0x77,
        0xdc, 0xc3, 0xf6, 0xde, 0x23, 0x8f, 0xab, 0x70, 0x13, 0x94, 0xcd, 0xc9, 0x6f, 0xb3,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    config_cid: "",
    weight_cids: &["QmW7LDz7ZTfw9vpAR9jMhFHWriLhxh728Kihp7oTSLgvyg"],
    dir_name: "Qwen3.8-27B",
    // ~16.8 GB Q4_K weights + KV/workspace → 24 GB card (3090/4090/5090), same as Qwen3.6-27B.
    min_vram_mb: 24_000,
};

pub const KIMI_LINEAR_48B: ModelSpec = ModelSpec {
    name: "kimi-linear-48b",
    model_id: [
        0x3d, 0xc0, 0x93, 0x58, 0xad, 0x75, 0xc6, 0xef, 0x0c, 0x9c, 0x86, 0xee, 0x4f, 0x47, 0xc4, 0xd6, 0xac, 0xda,
        0x96, 0x1f, 0xec, 0xbd, 0x0e, 0x4f, 0x9c, 0xf5, 0x5e, 0x8f, 0x0f, 0xdf, 0xfd, 0xdb,
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
        0xbd, 0x34, 0x56, 0x8c, 0xd8, 0x9f, 0x5f, 0x19, 0xc6, 0xc3, 0xa6, 0xe1, 0xa6, 0x1b, 0x92, 0x9b, 0xc8, 0x68,
        0x70, 0x94, 0x09, 0xea, 0xad, 0x8e, 0x67, 0x2d, 0x85, 0xf3, 0xc1, 0xeb, 0x57, 0x10,
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
        0x39, 0x99, 0x84, 0x04, 0x56, 0x00, 0xf7, 0xd5, 0x8d, 0x1b, 0x2c, 0xf0, 0x1e, 0x6a, 0x4b, 0xf4, 0x66, 0xfa,
        0x15, 0xc7, 0xac, 0x31, 0xbd, 0x0d, 0xd1, 0xa7, 0x1e, 0x00, 0x3b, 0x61, 0x7c, 0xc6,
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
        || *model_id == QWEN3_8_27B.model_id
        || *model_id == KIMI_LINEAR_48B.model_id
}

/// Mirror of the node's per-block tier table (`POM_TIERS_H6`, `POM_TIERS_H14` from the H14
/// private-inference gate), recomputed from the block DAA.
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
        // Tier 3 until the H14 gate (node `POM_TIERS_H6`), then retired: a Qwen3.6 walk past the
        // gate would be proven against Qwen3.8's anchor and rejected, so it has NO tier there.
        (!crate::pom::is_h14_era(daa)).then_some(3)
    } else if *model_id == QWEN3_8_27B.model_id {
        // Tier 3 from the H14 gate on (node `POM_TIERS_H14`); not mineable before it.
        crate::pom::is_h14_era(daa).then_some(3)
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
    ///
    /// This is the STARTUP-STAGING model (`staging_daa()`): pre-H14 lineup until the H14 gate is
    /// known to be behind us, then Qwen3.8-27B for `High`. The per-block truth is
    /// `pom_spec_at(daa)`; the CUDA miner hot-swaps a card at the crossing
    /// (`pom_gpu::advance_era_model_if_due`).
    pub fn pom_spec(self) -> &'static ModelSpec {
        self.pom_spec_at(staging_daa())
    }

    /// The model this hardware tier mines (and serves) for a block at `daa` — mirror of the node's
    /// `pom_tiers(..)` lineup for that block. Never `None`: every tier has a model in every era
    /// this binary mines (H6 onward).
    pub fn pom_spec_at(self, daa: u64) -> &'static ModelSpec {
        match self {
            Tier::VeryLight => &QWEN3_5_9B_ABLITERATED,
            Tier::Light => &GLM_4_9B_0414,
            Tier::Default => &GEMMA_4_12B_ABLITERATED,
            Tier::High if crate::pom::is_h14_era(daa) => &QWEN3_8_27B,
            Tier::High => &QWEN3_6_27B,
            Tier::VeryHigh => &KIMI_LINEAR_48B,
        }
    }

    /// The hardware tier a PoM model belongs to, in ANY era (Qwen3.6-27B and Qwen3.8-27B are both
    /// `High`). `None` for a non-PoM model.
    pub fn for_model(model_id: &[u8; 32]) -> Option<Tier> {
        if *model_id == QWEN3_5_9B_ABLITERATED.model_id {
            Some(Tier::VeryLight)
        } else if *model_id == GLM_4_9B_0414.model_id {
            Some(Tier::Light)
        } else if *model_id == GEMMA_4_12B_ABLITERATED.model_id {
            Some(Tier::Default)
        } else if *model_id == QWEN3_6_27B.model_id || *model_id == QWEN3_8_27B.model_id {
            Some(Tier::High)
        } else if *model_id == KIMI_LINEAR_48B.model_id {
            Some(Tier::VeryHigh)
        } else {
            None
        }
    }

    /// Short dashboard/log name of the tier.
    pub fn label(self) -> &'static str {
        match self {
            Tier::VeryLight => "very-light",
            Tier::Light => "light",
            Tier::Default => "default",
            Tier::High => "high",
            Tier::VeryHigh => "very-high",
        }
    }
}

/// Estimated wall-clock time of the mainnet H14 gate (2026-10-09 14:00 UTC, the node's own
/// estimate for DAA 121 985 000 at ~10.13 DAA/s). Used ONLY to choose which era's model to stage
/// at startup, before any block DAA is known; the per-block truth is always the job's DAA, and a
/// wrong guess is corrected at the first job (CUDA hot-swap / AMD+Metal pause).
pub const H14_MAINNET_ETA_UNIX: u64 = 1_791_554_400;

/// After this long past the ETA a fresh start stages only the post-H14 model (no Qwen3.6 prefetch).
const H14_PRE_ERA_PREFETCH_GRACE_SECS: u64 = 24 * 3600;

fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// `KERYX_H14_STAGE=pre|post` forces the startup staging era (testing / operators who know better
/// than the clock). Otherwise: an overridden gate (`KERYX_POM_H14_ACTIVATION_DAA`, testing) stages
/// pre-gate and lets the live crossing swap; mainnet stages post-gate once the ETA has passed.
fn stage_post_h14_at(now: u64, forced: Option<&str>, gate_overridden: bool) -> bool {
    match forced.map(|s| s.trim().to_ascii_lowercase()) {
        Some(ref s) if s == "post" => return true,
        Some(ref s) if s == "pre" => return false,
        _ => {}
    }
    !gate_overridden && now >= H14_MAINNET_ETA_UNIX
}

/// Uncached form of `stage_post_h14` — re-reads the clock on every call, for long-running
/// heuristics (a DAA-less pool job crossing the gate while the process runs).
pub fn h14_eta_passed_now() -> bool {
    stage_post_h14_at(
        now_unix(),
        std::env::var("KERYX_H14_STAGE").ok().as_deref(),
        crate::pom::is_h14_activation_overridden(),
    )
}

/// Whether startup staging should already use the post-H14 lineup (see `stage_post_h14_at`).
pub fn stage_post_h14() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        stage_post_h14_at(
            now_unix(),
            std::env::var("KERYX_H14_STAGE").ok().as_deref(),
            crate::pom::is_h14_activation_overridden(),
        )
    })
}

/// The DAA whose lineup startup staging uses: the H14 gate once `stage_post_h14()`, else the H6
/// (`pom_v3`) gate — i.e. the pre-H14 lineup, byte-identical to 0.13.4.
pub fn staging_daa() -> u64 {
    if stage_post_h14() {
        crate::pom::private_inference_activation_daa()
    } else {
        crate::pom::pom_v3_activation_daa()
    }
}

/// Every model `tier` may have to mine across the eras this process can still meet, staging era
/// first. Before the H14 gate (and for a day after its ETA) that is BOTH tier-3 models, so the
/// crossing hot-swaps without a mid-run 16.8 GB download; afterwards only the current one.
pub fn pom_models_all_eras(tier: Tier) -> Vec<&'static ModelSpec> {
    let mut out = vec![tier.pom_spec()];
    let mut push = |s: &'static ModelSpec| {
        if !out.iter().any(|o| o.model_id == s.model_id) {
            out.push(s);
        }
    };
    if !stage_post_h14() {
        // Pre-gate staging: the post-gate model must already be on disk at the crossing.
        push(tier.pom_spec_at(crate::pom::private_inference_activation_daa()));
    } else if !crate::pom::is_h14_activation_overridden()
        && now_unix() < H14_MAINNET_ETA_UNIX.saturating_add(H14_PRE_ERA_PREFETCH_GRACE_SECS)
        && std::env::var("KERYX_H14_STAGE").is_err()
    {
        // Started just after the ETA: the gate may still be a few blocks away, keep the old one.
        push(tier.pom_spec_at(crate::pom::pom_v3_activation_daa()));
    }
    out
}

/// Pinned PoM possession anchors `(R_T, N chunks)` — mirror of the node's `POM_TIERS_H6` and
/// `POM_TIERS_H14` (consensus/core/src/config/params.rs). The miner's own tree builder recomputes
/// R_T from the GGUF; comparing it here catches a corrupt/wrong download BEFORE it mines proofs
/// the node can only reject. Verified 2026-10-05 against the cached pom-tree.meta of every
/// current model and a fresh index build of Qwen3.8-27B.
pub fn pinned_pom_anchor(model_id: &[u8; 32]) -> Option<([u8; 32], u64)> {
    const ANCHORS: &[([u8; 32], [u8; 32], u64)] = &[
        (
            QWEN3_5_9B_ABLITERATED.model_id,
            [
                0x2c, 0x49, 0x71, 0x64, 0xea, 0xf2, 0x00, 0x78, 0xad, 0xd2, 0x0e, 0x82, 0xae, 0x4e, 0x1b, 0x0f, 0xdb,
                0x27, 0xd3, 0xfd, 0xd5, 0xea, 0xef, 0xc1, 0xc4, 0x8f, 0x20, 0x41, 0x11, 0xe1, 0x4e, 0x88,
            ],
            203_469_888,
        ),
        (
            GLM_4_9B_0414.model_id,
            [
                0x1b, 0xa8, 0xb8, 0xb1, 0x34, 0x41, 0x03, 0xfa, 0xa0, 0xa7, 0x47, 0x89, 0xd9, 0x39, 0xc3, 0x3c, 0x23,
                0xba, 0x5c, 0x3c, 0x41, 0xbb, 0x1a, 0x89, 0x5a, 0xb6, 0xe8, 0xbf, 0xec, 0xb0, 0x78, 0x7d,
            ],
            258_040_832,
        ),
        (
            GEMMA_4_12B_ABLITERATED.model_id,
            [
                0x8e, 0x4d, 0x5b, 0xe3, 0xaa, 0x7c, 0x3a, 0xb9, 0x35, 0x83, 0x5f, 0xf5, 0xe1, 0x9d, 0x7a, 0x3d, 0xfa,
                0x11, 0x8a, 0xf3, 0x24, 0xd5, 0xba, 0x65, 0x16, 0x29, 0xd6, 0xed, 0x16, 0x1a, 0x1e, 0x37,
            ],
            305_318_656,
        ),
        (
            QWEN3_6_27B.model_id,
            [
                0x85, 0x23, 0xf4, 0x14, 0x8d, 0x22, 0xc7, 0x71, 0x3b, 0xfc, 0x11, 0x32, 0xb4, 0xaf, 0x3d, 0x4b, 0x97,
                0x61, 0xa2, 0x03, 0xfb, 0x33, 0xf1, 0x8e, 0xe7, 0x55, 0x67, 0xbd, 0xee, 0x51, 0x2b, 0x0a,
            ],
            516_762_688,
        ),
        (
            QWEN3_8_27B.model_id,
            [
                0x40, 0x6c, 0x19, 0x56, 0xf9, 0xf5, 0xdd, 0x13, 0x4d, 0x34, 0x61, 0xc6, 0x19, 0x11, 0x32, 0xa3, 0xb1,
                0x57, 0x2c, 0xc1, 0x6f, 0x39, 0x5a, 0x2b, 0xc2, 0xf1, 0xc6, 0x69, 0xfa, 0xe3, 0x74, 0xa1,
            ],
            524_991_232,
        ),
        (
            KIMI_LINEAR_48B.model_id,
            [
                0x95, 0x74, 0x71, 0x0f, 0xfa, 0xb6, 0x78, 0xf0, 0x68, 0xb4, 0xe6, 0x5a, 0xbe, 0x72, 0x40, 0x86, 0x2d,
                0xa1, 0x5b, 0xb1, 0x6e, 0xa8, 0x2f, 0xd1, 0x62, 0xa9, 0x35, 0x1a, 0x10, 0x51, 0x99, 0x59,
            ],
            927_994_064,
        ),
    ];
    ANCHORS.iter().find(|(id, _, _)| id == model_id).map(|(_, root, n)| (*root, *n))
}

/// Check a freshly built/reused possession index against the pinned anchor. `Err` carries an
/// operator-facing reason. ENFORCED (the caller refuses to mine) only for models introduced by
/// the H14 release (Qwen3.8-27B); for the pre-H14 lineup a mismatch is reported but tolerated,
/// keeping the pre-gate behaviour of 0.13.4 unchanged.
pub fn check_pom_anchor(model_id: &[u8; 32], r_t: &[u8; 32], n_chunks: u64) -> Result<(), String> {
    let Some((root, n)) = pinned_pom_anchor(model_id) else { return Ok(()) };
    if root == *r_t && n == n_chunks {
        return Ok(());
    }
    Err(format!(
        "possession root mismatch for model {:.16}: computed R_T={} N={}, node pins R_T={} N={} — the GGUF on          disk is corrupt or not the pinned model; delete its directory to re-download",
        hex::encode(model_id),
        hex::encode(r_t),
        n_chunks,
        hex::encode(root),
        n
    ))
}

/// Whether `check_pom_anchor` failures for this model must stop mining (see there).
pub fn pom_anchor_enforced(model_id: &[u8; 32]) -> bool {
    *model_id == QWEN3_8_27B.model_id
}

/// `--tier auto`: pick the LARGEST tier whose footprint fits the GPU's VRAM, with a conservative
/// safety margin so the chosen tier loads cleanly (weights + PoM possession walk + CUDA workspace
/// + KV cache for GPU inference). Returns the tier and its budgeted MiB requirement.
///
/// The budget is the model's `min_vram_mb` (which already accounts for weights + KV + workspace),
/// plus a `headroom_mb` margin on top. Empirically an 8 GB 3070 OOMs the larger inference tiers,
/// so the margin must be conservative: Light (min_vram_mb=0) is the safe floor on an 8 GB card,
/// while Default (needs 8000) is not selected. CPU inference is not part of normal tier sizing.
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
        Tier::High if crate::pom::is_h14_era(daa) => &[&QWEN3_8_27B],
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
    // H14 tier 3 — appended (not inserted) so the historical index of every pre-H14 entry is kept.
    &QWEN3_8_27B,
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

    /// Minimal base58btc decoder (big-endian base conversion) for the CID binding checks.
    fn b58(s: &str) -> Vec<u8> {
        const ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        let mut num: Vec<u8> = Vec::new(); // little-endian base-256
        for c in s.bytes() {
            let mut carry = ALPHABET.iter().position(|&a| a == c).expect("base58 digit") as u32;
            for b in num.iter_mut() {
                carry += *b as u32 * 58;
                *b = carry as u8;
                carry >>= 8;
            }
            while carry > 0 {
                num.push(carry as u8);
                carry >>= 8;
            }
        }
        let zeros = s.bytes().take_while(|&b| b == b'1').count();
        let mut out = vec![0u8; zeros];
        out.extend(num.iter().rev());
        out
    }

    /// The H6 per-block tier table — mirror of the node's `POM_TIERS_H6` order — just below the H14
    /// gate, i.e. the lineup 0.13.4 mines today.
    #[test]
    fn h6_tier_table_mirrors_node() {
        let daa = crate::pom::private_inference_activation_daa() - 1;
        assert_eq!(pom_tier_index(&QWEN3_5_9B_ABLITERATED.model_id, daa), Some(0));
        assert_eq!(pom_tier_index(&GLM_4_9B_0414.model_id, daa), Some(1));
        assert_eq!(pom_tier_index(&GEMMA_4_12B_ABLITERATED.model_id, daa), Some(2));
        assert_eq!(pom_tier_index(&QWEN3_6_27B.model_id, daa), Some(3));
        assert_eq!(pom_tier_index(&QWEN3_8_27B.model_id, daa), None);
        assert_eq!(pom_tier_index(&KIMI_LINEAR_48B.model_id, daa), Some(4));
        // The hardware-tier -> model map agrees with the table, tier for tier.
        assert_eq!(specs_for(daa, Tier::VeryLight)[0].model_id, QWEN3_5_9B_ABLITERATED.model_id);
        assert_eq!(specs_for(daa, Tier::Light)[0].model_id, GLM_4_9B_0414.model_id);
        assert_eq!(specs_for(daa, Tier::Default)[0].model_id, GEMMA_4_12B_ABLITERATED.model_id);
        assert_eq!(specs_for(daa, Tier::High)[0].model_id, QWEN3_6_27B.model_id);
        assert_eq!(specs_for(daa, Tier::VeryHigh)[0].model_id, KIMI_LINEAR_48B.model_id);
        for tier in Tier::DESCENDING {
            assert_eq!(specs_for(daa, tier)[0].model_id, tier.pom_spec_at(daa).model_id);
        }

        // An unknown model is never mineable, at any DAA.
        assert_eq!(pom_tier_index(&[0u8; 32], daa), None);
        assert_eq!(pom_tier_index(&[0u8; 32], u64::MAX), None);
        assert!(!is_pom_model(&[0u8; 32]));
    }

    /// The H14 lineup — mirror of the node's `POM_TIERS_H14` (h14_swaps_only_tier_3): only tier 3
    /// changes, to Qwen3.8-27B, exactly at the gate DAA.
    #[test]
    fn h14_swaps_only_tier_3_at_the_gate() {
        let gate = crate::pom::private_inference_activation_daa();
        assert_eq!(gate, crate::pom::POM_H14_ACTIVATION_DAA, "mainnet gate (no env override in tests)");
        assert_eq!(gate, 121_985_000);
        for daa in [gate, gate + 1, u64::MAX] {
            assert_eq!(pom_tier_index(&QWEN3_5_9B_ABLITERATED.model_id, daa), Some(0));
            assert_eq!(pom_tier_index(&GLM_4_9B_0414.model_id, daa), Some(1));
            assert_eq!(pom_tier_index(&GEMMA_4_12B_ABLITERATED.model_id, daa), Some(2));
            assert_eq!(pom_tier_index(&QWEN3_8_27B.model_id, daa), Some(3));
            assert_eq!(pom_tier_index(&QWEN3_6_27B.model_id, daa), None, "Qwen3.6 retired at the gate");
            assert_eq!(pom_tier_index(&KIMI_LINEAR_48B.model_id, daa), Some(4));
            assert_eq!(specs_for(daa, Tier::High)[0].model_id, QWEN3_8_27B.model_id);
            assert_eq!(Tier::High.pom_spec_at(daa).model_id, QWEN3_8_27B.model_id);
            for tier in [Tier::VeryLight, Tier::Light, Tier::Default, Tier::VeryHigh] {
                assert_eq!(tier.pom_spec_at(daa).model_id, tier.pom_spec_at(gate - 1).model_id);
            }
        }
        // Node constant QWEN3_8_27B_MODEL_ID, pinned as hex in the node's own h14 test.
        assert_eq!(hex::encode(QWEN3_8_27B.model_id), "73740b443bdc00afda5fa34eb9999d3fea77dcc3f6de238fab701394cdc96fb3");
        // model_id = CIDv0[2..34]: the pinned CID decodes to exactly this digest.
        let cid = b58(QWEN3_8_27B.weight_cids[0]);
        assert_eq!(&cid[..2], &[0x12, 0x20]);
        assert_eq!(&cid[2..34], &QWEN3_8_27B.model_id);
    }

    #[test]
    fn every_registry_model_has_a_tier_and_an_anchor() {
        let pre = crate::pom::private_inference_activation_daa() - 1;
        for spec in REGISTRY {
            assert!(is_pom_model(&spec.model_id), "{} is not a PoM model", spec.name);
            let tier = pom_tier_index(&spec.model_id, u64::MAX).or_else(|| pom_tier_index(&spec.model_id, pre));
            assert!(tier.is_some(), "{} has no tier in any era", spec.name);
            assert!(Tier::for_model(&spec.model_id).is_some(), "{} has no hardware tier", spec.name);
            assert!(pinned_pom_anchor(&spec.model_id).is_some(), "{} has no pinned anchor", spec.name);
            assert_eq!(find(spec.name).map(|s| s.model_id), Some(spec.model_id));
            // CIDv0 binding of every pinned GGUF.
            let cid = b58(spec.weight_cids[0]);
            assert_eq!(&cid[2..34], &spec.model_id, "{}", spec.name);
        }
        for tier in Tier::DESCENDING {
            for daa in [pre, pre + 1] {
                let s = tier.pom_spec_at(daa);
                assert!(REGISTRY.iter().any(|r| r.model_id == s.model_id), "{} not in REGISTRY", s.name);
                assert_eq!(Tier::for_model(&s.model_id), Some(tier));
            }
        }
        // The pre-H14 registry order is unchanged (dashboards and logs index it).
        let names: Vec<_> = REGISTRY.iter().map(|s| s.name).collect();
        assert_eq!(
            names,
            ["qwen3.5-9b-abliterated", "glm-4-9b-0414", "gemma-4-12b-abliterated", "qwen3.6-27b", "kimi-linear-48b", "qwen3.8-27b"]
        );
    }

    #[test]
    fn anchor_check_enforces_only_the_h14_model() {
        let (root, n) = pinned_pom_anchor(&QWEN3_8_27B.model_id).unwrap();
        assert_eq!(hex::encode(root), "406c1956f9f5dd134d3461c6191132a3b1572cc16f395a2bc2f1c669fae374a1");
        assert_eq!(n, 524_991_232);
        assert!(check_pom_anchor(&QWEN3_8_27B.model_id, &root, n).is_ok());
        assert!(check_pom_anchor(&QWEN3_8_27B.model_id, &root, n - 1).is_err());
        let mut bad = root;
        bad[0] ^= 1;
        assert!(check_pom_anchor(&QWEN3_8_27B.model_id, &bad, n).is_err());
        assert!(pom_anchor_enforced(&QWEN3_8_27B.model_id));
        for spec in [&QWEN3_5_9B_ABLITERATED, &GLM_4_9B_0414, &GEMMA_4_12B_ABLITERATED, &QWEN3_6_27B, &KIMI_LINEAR_48B] {
            assert!(!pom_anchor_enforced(&spec.model_id), "{}", spec.name);
        }
        // Unknown models carry no anchor and are never blocked by this check.
        assert!(check_pom_anchor(&[0u8; 32], &[0u8; 32], 0).is_ok());
    }

    /// Recompute R_T over a real GGUF with the miner's own tree builder and check it against the
    /// node's pinned anchor. (model_id is the CIDv0 digest of the UnixFS DAG root, not a plain
    /// sha256 of the file, so the content check IS this R_T.) Ignored: needs the multi-GB file.
    /// `KERYX_TEST_ANCHOR_GGUF=<model.gguf> [KERYX_TEST_ANCHOR_MODEL=qwen3.8-27b] cargo test --lib
    /// pinned_anchor_matches_a_real_gguf -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn pinned_anchor_matches_a_real_gguf() {
        let Ok(path) = std::env::var("KERYX_TEST_ANCHOR_GGUF") else {
            eprintln!("KERYX_TEST_ANCHOR_GGUF not set — skipping");
            return;
        };
        let name = std::env::var("KERYX_TEST_ANCHOR_MODEL").unwrap_or_else(|_| QWEN3_8_27B.name.to_string());
        let spec = find(&name).expect("model name");
        let t0 = std::time::Instant::now();
        let idx = crate::pom::WeightIndex::build_from_gguf(&path, spec.model_id).expect("index build");
        eprintln!(
            "{}: N={} R_T={} ({:.0}s)",
            spec.name,
            idx.n_chunks,
            hex::encode(idx.r_t),
            t0.elapsed().as_secs_f64()
        );
        check_pom_anchor(&spec.model_id, &idx.r_t, idx.n_chunks).expect("anchor");
    }

    #[test]
    fn startup_staging_era_selection() {
        let eta = H14_MAINNET_ETA_UNIX;
        // Mainnet, no override: the clock decides.
        assert!(!stage_post_h14_at(eta - 1, None, false));
        assert!(stage_post_h14_at(eta, None, false));
        // Overridden test gate: stage pre-gate and cross live.
        assert!(!stage_post_h14_at(eta + 10, None, true));
        // Explicit operator choice wins either way.
        assert!(stage_post_h14_at(eta - 1_000_000, Some("post"), false));
        assert!(!stage_post_h14_at(eta + 1_000_000, Some(" PRE "), false));
        assert!(stage_post_h14_at(eta + 1, Some("bogus"), false));
        // 2026-10-09T14:00:00Z
        assert_eq!(eta, 1_791_554_400);
    }

    /// Pre-gate staging (today) is byte-identical to 0.13.4 and prefetches Qwen3.8 for `High` only.
    #[test]
    fn pre_gate_staging_matches_0_13_4_and_prefetches_h14_model() {
        if stage_post_h14() {
            return; // running after the ETA — covered by startup_staging_era_selection
        }
        assert_eq!(staging_daa(), crate::pom::pom_v3_activation_daa());
        assert_eq!(Tier::High.pom_spec().model_id, QWEN3_6_27B.model_id);
        let high: Vec<_> = pom_models_all_eras(Tier::High).iter().map(|s| s.name).collect();
        assert_eq!(high, ["qwen3.6-27b", "qwen3.8-27b"]);
        for tier in [Tier::VeryLight, Tier::Light, Tier::Default, Tier::VeryHigh] {
            assert_eq!(pom_models_all_eras(tier).len(), 1, "{:?}", tier);
        }
    }
}
