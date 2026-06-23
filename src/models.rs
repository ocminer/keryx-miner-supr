/// Registry of supported inference models.
///
/// model_id = sha2-256(primary_weight_file) = CIDv0_bytes[2..34].
/// Verifiable: decode the weight CID from base58btc, skip the 2-byte multihash prefix.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Full-precision safetensors (one or more shards).
    Safetensors,
    /// GGUF quantized — LLaMA/LLaMA3 architecture.
    Gguf,
    /// GGUF quantized — Qwen2 architecture (DeepSeek-R1-32B distill).
    GgufQwen2,
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
}

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
};

/// PoM tier 0 — Gemma-3-4B-it-abliterated Q4_K_M. Possession model for the baseline tier
/// (node-pinned R_T 846caa40…, N=77_604_776 chunks, ~2.48 GiB). `format` = Gguf for PoM
/// possession (GGUF downloaded + WeightIndex-parsed; Gemma inference serving is a separate
/// OPoI concern). model_id = sha2-256(weight CID) = CIDv0[2..34].
pub const GEMMA_3_4B: ModelSpec = ModelSpec {
    name: "gemma-3-4b",
    model_id: [
        0xad, 0x50, 0xad, 0x0b, 0xd4, 0x61, 0xd8, 0xab,
        0x44, 0xef, 0xc0, 0x21, 0x49, 0x89, 0xeb, 0x33,
        0x29, 0x16, 0x85, 0xef, 0x4a, 0xde, 0x22, 0xa0,
        0xf4, 0xf2, 0x17, 0xd0, 0x32, 0x66, 0xd8, 0x37,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmTh2MsVfAvWp7grN9rvkF9NkMkCW2PhWez2WbNh81KRXD",
    config_cid: "",
    weight_cids: &["Qma1CbFzWTNhy2ReVjDG1GvM5q2Uy4VhqTbnS9c641jUQ6"],
    dir_name: "Gemma-3-4B",
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
};

pub const LLAMA_3_3_70B: ModelSpec = ModelSpec {
    name: "llama-3.3-70b",
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
    dir_name: "Llama-3.3-70B",
};

pub const REGISTRY: &[&ModelSpec] = &[&TINYLLAMA, &DEEPSEEK_R1_8B, &DEEPSEEK_R1_32B, &LLAMA_3_3_70B];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}
