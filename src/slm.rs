/// Phase-3 OPoI: multi-model inference engine (safetensors + GGUF) via candle.
///
/// Models are loaded on demand when an AiRequest arrives and cached between
/// consecutive requests for the same model. Mining pauses during inference.
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::{gguf_file, QTensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, LlamaConfig, Llama};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3Weights;
use candle_transformers::models::quantized_gemma3::ModelWeights as Gemma3Weights;
use crate::quantized_gemma3_split::ModelWeights as Gemma3SplitWeights;
use crate::quantized_llama_split::ModelWeights as SplitWeights;
use crate::quantized_qwen3_split::ModelWeights as Qwen3SplitWeights;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, RwLock};
use tokenizers::Tokenizer;

use crate::models::{ModelFormat, ModelSpec};

const IPFS_GATEWAY: &str = "https://keryx-labs.com";
// Legacy lineup (pre-OPoI-v2) system prompts.
const SYSTEM_PROMPT_TINYLLAMA: &str =
    "You are a Keryx Network AI — a decentralized assistant running on GPU miners. \
     No internet access. Be concise.";
const SYSTEM_PROMPT_DEEPSEEK: &str =
    "You are a Keryx Network AI — a decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention DeepSeek, Anthropic, OpenAI, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be concise.";
const SYSTEM_PROMPT_GEMMA: &str =
    "You are a Keryx Network AI — a decentralized, uncensored assistant running on GPU miners via the Keryx BlockDAG protocol. \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Gemma, Google, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be concise.";

const SYSTEM_PROMPT_DOLPHIN: &str =
    "You are a Keryx Network AI — a decentralized, uncensored assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Dolphin, Llama, Meta, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be concise.";

const SYSTEM_PROMPT_LLAMA70B: &str =
    "You are a Keryx Network AI — a high-capability decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Meta, Llama, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

const SYSTEM_PROMPT_QWEN3: &str =
    "You are a Keryx Network AI — a high-capability decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Qwen, Alibaba, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

// ── Static engine state ──────────────────────────────────────────────────────

/// Models the miner currently serves (drives `ai:cap`). Mutable so the lineup can be
/// hot-swapped at the OPoI-v2 hardfork crossing without a restart.
static SUPPORTED_SPECS: RwLock<&'static [&'static ModelSpec]> = RwLock::new(&[]);
/// Pre-filtered OPoI-v2 (uncensored) lineup, staged + background-prefetched at boot,
/// swapped into SUPPORTED_SPECS when the chain crosses `OPOI_V2_ACTIVATION_DAA`.
static LINEUP_V2: RwLock<&'static [&'static ModelSpec]> = RwLock::new(&[]);
/// Set once the v2 lineup has been swapped in (idempotent guard for the crossing).
static V2_ACTIVE: AtomicBool = AtomicBool::new(false);
static ENGINE: Mutex<Option<SlmEngine>> = Mutex::new(None);

/// When true, the mining-tier model is loaded via the layer-split loader even on a single GPU,
/// so it lands as a `QuantizedQwen3Split` (etc.) that exposes `pom_quant_tensors()`. This lets
/// the PoM walk share the inference weights in place (Option C2 zero-dup). Set at startup when
/// PoM mining is configured. Single-device split == upstream behaviour (no cross-device moves).
static POM_FORCE_SPLIT: AtomicBool = AtomicBool::new(false);

/// Force the split loader for PoM zero-dup (see [`POM_FORCE_SPLIT`]). Call once at startup.
pub fn set_pom_force_split(enabled: bool) {
    POM_FORCE_SPLIT.store(enabled, AtomicOrdering::Relaxed);
}

/// Whether the PoM zero-dup split loader is forced.
pub fn pom_force_split() -> bool {
    POM_FORCE_SPLIT.load(AtomicOrdering::Relaxed)
}

enum ModelInner {
    Full { model: Llama, config: Config, cache_dtype: DType },
    Quantized(ModelWeights),
    /// GGUF llama-arch model via the split loader (single-device, for PoM zero-dup tensor sharing).
    QuantizedSplit(SplitWeights),
    QuantizedQwen3(Qwen3Weights),
    /// GGUF Qwen3-arch dense model (Qwen3-32B) via the split loader (single-device, PoM zero-dup).
    QuantizedQwen3Split(Qwen3SplitWeights),
    /// GGUF Gemma-3-arch model (Gemma-3-4B, baseline tier). Single-device only.
    QuantizedGemma3(Gemma3Weights),
    /// GGUF Gemma-3-arch model via the single-device split fork (exposes quant tensors
    /// for PoM zero-dup, so the possession walk shares the inference weights in place
    /// instead of loading a 2nd copy — the fix that lets 8 GB cards do GPU inference).
    QuantizedGemma3Split(Gemma3SplitWeights),
    /// GGUF Qwen2-arch model (legacy DeepSeek-R1-32B, pre-OPoI-v2 lineup). Single-device.
    QuantizedQwen2(Qwen2Weights),
}

struct SlmEngine {
    model_id: [u8; 32],
    name: &'static str,
    inner: ModelInner,
    tokenizer: Tokenizer,
    device: Device,
    /// All token IDs that terminate generation (EOS, EOT, role-start tokens, etc.).
    stop_token_ids: Vec<u32>,
    /// Literal stop strings — safety net for tokenizers that emit control markers
    /// as plain text (e.g. a GGUF whose tokenizer.json lacks the ChatML special
    /// tokens) so the matching `stop_token_ids` are never produced. Generation is
    /// cut at the earliest occurrence of any of these in the decoded output.
    stop_strings: Vec<&'static str>,
}

unsafe impl Send for SlmEngine {}
unsafe impl Sync for SlmEngine {}

// ── File management ──────────────────────────────────────────────────────────

fn model_dir(spec: &ModelSpec) -> std::path::PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    exe_dir.join("models").join(spec.dir_name)
}

/// Path to a model's GGUF file (`<exe_dir>/models/<dir_name>/model.gguf`). Used by PoM to
/// build the possession weight index from the resident model.
pub fn gguf_path_for(spec: &ModelSpec) -> std::path::PathBuf {
    model_dir(spec).join("model.gguf")
}

/// Downloads `url` to `dest` with automatic resume. A partially downloaded file is
/// continued via an HTTP `Range` request instead of restarting from zero, and both
/// connect-time and mid-stream failures are retried with a fixed backoff. Designed
/// for the huge (10-40 GB) model GGUFs served over the flaky IPFS gateway: the
/// content is immutable (CID-addressed), so appending resumed bytes is always
/// consistent, and an already-complete file (e.g. pre-staged with `wget -c`) is
/// detected via a 416 response and left untouched instead of being re-downloaded.
fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 240; // survives long gateway outages (~40 min of retries)
    const BACKOFF_SECS: u64 = 10;
    eprintln!("[keryx-miner] Downloading {} ...", url);
    let mut attempt = 0u32;
    loop {
        // Resume offset = how many bytes we already have on disk.
        let resume_from = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);

        let mut req = ureq::get(url);
        if resume_from > 0 {
            req = req.set("Range", &format!("bytes={}-", resume_from));
        }
        let response = match req.call() {
            Ok(r) => r,
            Err(e) => {
                attempt += 1;
                if attempt >= MAX_ATTEMPTS {
                    return Err(anyhow!("HTTP GET {} failed after {} attempts: {}", url, attempt, e));
                }
                eprintln!("\n[keryx-miner] connect error ({e}); retry {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s (resume @ {} MB)…",
                    resume_from / 1_000_000);
                std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
                continue;
            }
        };
        let status = response.status();

        // Decide whether to append (server honored the range) or (re)start, and the total size.
        let (mut file, mut downloaded, total): (std::fs::File, u64, Option<u64>) =
            if resume_from > 0 && status == 206 {
                // Content-Range: "bytes <start>-<end>/<total>"
                let total = response
                    .header("Content-Range")
                    .and_then(|cr| cr.rsplit('/').next())
                    .and_then(|t| t.trim().parse::<u64>().ok());
                let f = std::fs::OpenOptions::new()
                    .append(true)
                    .open(dest)
                    .with_context(|| format!("open append {}", dest.display()))?;
                (f, resume_from, total)
            } else if resume_from > 0 && status == 416 {
                // Range not satisfiable ⇒ the file is already fully downloaded.
                eprintln!("\r  already complete ({} MB).            ", resume_from / 1_000_000);
                return Ok(());
            } else {
                // 200, or the server ignored Range ⇒ (re)start from scratch.
                let total = response.header("Content-Length").and_then(|s| s.parse::<u64>().ok());
                let f = std::fs::File::create(dest)
                    .with_context(|| format!("create {}", dest.display()))?;
                (f, 0u64, total)
            };

        let mut reader = response.into_reader();
        let mut buf = vec![0u8; 65_536];
        let mut stream_err: Option<String> = None;
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = file.write_all(&buf[..n]) {
                        stream_err = Some(e.to_string());
                        break;
                    }
                    downloaded += n as u64;
                    if let Some(t) = total {
                        eprint!("\r  {:.1}/{:.1} MB ({}%)   ",
                            downloaded as f64 / 1_000_000.0,
                            t as f64 / 1_000_000.0,
                            downloaded * 100 / t.max(1));
                        let _ = std::io::stderr().flush();
                    }
                }
                Err(e) => { stream_err = Some(e.to_string()); break; }
            }
        }
        let _ = file.flush();

        // Done only if the stream ended cleanly AND we reached the known total. An unknown
        // total (chunked IPFS-gateway response with no Content-Length/Content-Range) must NOT
        // count as complete: a clean early EOF would otherwise mark a truncated GGUF as done,
        // write the `.ok` sentinel, and let the miner start on a partial model (failing every
        // challenge). Treat unknown-total as incomplete and retry — a fresh Range request
        // usually returns a parsable Content-Range and self-heals.
        let complete = stream_err.is_none() && matches!(total, Some(t) if downloaded >= t);
        if complete {
            eprintln!();
            return Ok(());
        }

        attempt += 1;
        if attempt >= MAX_ATTEMPTS {
            return Err(anyhow!("download {} interrupted after {} attempts (got {} MB)",
                url, attempt, downloaded / 1_000_000));
        }
        let why = stream_err.unwrap_or_else(|| "short read".into());
        eprintln!("\n[keryx-miner] interrupted ({why}); resuming {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s @ {} MB…",
            downloaded / 1_000_000);
        std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
    }
}

fn ipfs_url(cid: &str) -> String {
    format!("{}/ipfs/{}", IPFS_GATEWAY, cid)
}

fn ensure_safetensors(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let cfg = dir.join("config.json");
    let ok_flag = dir.join(".ok");
    let wts: Vec<_> = spec.weight_cids.iter().enumerate().map(|(i, _)| {
        if spec.weight_cids.len() == 1 { dir.join("model.safetensors") }
        else { dir.join(format!("model-{:05}-of-{:05}.safetensors", i + 1, spec.weight_cids.len())) }
    }).collect();

    // .ok sentinel written only after a complete download — guards against truncated files
    if tok.exists() && cfg.exists() && wts.iter().all(|p| p.exists()) && ok_flag.exists() {
        log::debug!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, cfg, wts));
    }
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before re-downloading
    eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if !tok.exists() { download_file(&ipfs_url(spec.tokenizer_cid), &tok)?; }
    if !cfg.exists() { download_file(&ipfs_url(spec.config_cid), &cfg)?; }
    for (i, (cid, path)) in spec.weight_cids.iter().zip(wts.iter()).enumerate() {
        if spec.weight_cids.len() > 1 { eprintln!("[keryx-miner] Shard {}/{}", i + 1, spec.weight_cids.len()); }
        download_file(&ipfs_url(cid), path)?;
    }
    std::fs::write(&ok_flag, b"").with_context(|| format!("write .ok flag {}", ok_flag.display()))?;
    eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, cfg, wts))
}

fn ensure_gguf(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let gguf = dir.join("model.gguf");
    let ok_flag = dir.join(".ok");

    // .ok sentinel written only after a complete download — guards against truncated files
    if tok.exists() && gguf.exists() && ok_flag.exists() {
        log::debug!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, gguf));
    }
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before re-downloading
    eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if !tok.exists() { download_file(&ipfs_url(spec.tokenizer_cid), &tok)?; }
    download_file(&ipfs_url(spec.weight_cids[0]), &gguf)?;
    std::fs::write(&ok_flag, b"").with_context(|| format!("write .ok flag {}", ok_flag.display()))?;
    eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, gguf))
}

// ── Engine loading ───────────────────────────────────────────────────────────

/// Build the list of stop token IDs for a model.
///
/// Tries `token_to_id` for each name first; falls back to the corresponding
/// hardcoded ID so generation always terminates even if the tokenizer exposes
/// special tokens differently (e.g. via `added_tokens` vs the regular vocab).
fn collect_stop_ids(tokenizer: &Tokenizer, names: &[&str], fallbacks: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = names.iter().zip(fallbacks.iter())
        .map(|(name, &fallback)| tokenizer.token_to_id(name).unwrap_or(fallback))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Per-model terminating tokens and literal stop strings, keyed by model name so
/// it stays coherent with `format_prompt`. Two models can share a `ModelFormat`
/// (Dolphin-8B and Llama-3.3-70B are both LLaMA-arch GGUF) yet need different
/// stop conventions, so this branches on name rather than format.
fn stop_config(tokenizer: &Tokenizer, name: &str) -> (Vec<u32>, Vec<&'static str>) {
    match name {
        // Dolphin-3.0-Llama-3.1-8B — ChatML template (Dolphin adds <|im_*|> tokens
        // over the Llama-3.1 vocab):
        //   <|im_end|> ends a turn; <|end_of_text|>/<|eot_id|> kept as base fallbacks.
        "dolphin-llama3-8b" => (
            collect_stop_ids(tokenizer,
                &["<|im_end|>", "<|end_of_text|>", "<|eot_id|>"],
                &[]),
            vec!["<|im_end|>", "<|im_start|>", "<|end_of_text|>"],
        ),
        // Gemma-3-4B — Gemma chat template:
        //   <end_of_turn> ends a turn, <eos> (id 1) is the base EOS.
        "gemma-3-4b" => (
            collect_stop_ids(tokenizer,
                &["<end_of_turn>", "<eos>"],
                &[1]),
            vec!["<end_of_turn>", "<start_of_turn>"],
        ),
        // Llama-3.3-70B-Instruct (abliterated) — LLaMA-3 header template. Stop on
        // the official `eos_token_id` set for 3.3 Instruct:
        //   128009 = <|eot_id|> (end of turn), 128001 = <|end_of_text|> (base EOS),
        //   128008 = <|eom_id|> (end of message, tool turns).
        // Both the abliterated (v2) and the genuine official (v1) Llama-3.3-70B share
        // the LLaMA-3 header template and terminators.
        "llama-3.3-70b" | "llama-3.3-70b-official" => (
            collect_stop_ids(tokenizer,
                &["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"],
                &[128009, 128001, 128008]),
            // Cut if the model tries to open a fresh turn instead of stopping.
            vec!["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
        ),
        // Qwen3-32B — ChatML template:
        //   151645 = <|im_end|> (end of turn), 151643 = <|endoftext|> (base EOS).
        "qwen3-32b" => (
            collect_stop_ids(tokenizer,
                &["<|im_end|>", "<|endoftext|>"],
                &[151645, 151643]),
            // Cut if the model opens a fresh turn instead of stopping.
            vec!["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        ),
        // ── Legacy lineup (pre-OPoI-v2) ──────────────────────────────────────
        // DeepSeek-R1-Distill-Llama-8B — DeepSeek chat template:
        //   128001 = <｜end▁of▁sentence｜> (real EOS), 128011 = <｜User｜> (new turn),
        //   128009 = <|eot_id|> (LLaMA-3 EOT, kept as a fallback).
        "deepseek-r1-8b" => (
            collect_stop_ids(tokenizer,
                &["<｜end▁of▁sentence｜>", "<｜User｜>", "<|eot_id|>"],
                &[128001, 128011, 128009]),
            vec!["<｜end▁of▁sentence｜>", "<｜User｜>", "<|eot_id|>", "<|end_of_text|>"],
        ),
        // DeepSeek-R1-Distill-Qwen-32B — DeepSeek chat template (NOT ChatML):
        //   151643 = <｜end▁of▁sentence｜> (real EOS), 151644 = <｜User｜> (new turn).
        //   (151645 is <｜Assistant｜>, NOT an end token — must not stop on it.)
        "deepseek-r1-32b" => (
            collect_stop_ids(tokenizer,
                &["<｜end▁of▁sentence｜>", "<｜User｜>"],
                &[151643, 151644]),
            // ASCII ChatML markers kept as an extra net if the model parrots them.
            vec!["<｜end▁of▁sentence｜>", "<｜User｜>", "<|im_end|>", "<|im_start|>"],
        ),
        // Generic fallback (incl. TinyLlama / Zephyr): </s> ends a turn; 0 = padding safety net.
        _ => (
            collect_stop_ids(tokenizer, &["</s>"], &[2, 0]),
            vec!["</s>", "<|user|>", "<|system|>", "<|assistant|>"],
        ),
    }
}

fn load_engine(spec: &'static ModelSpec, device: Device) -> Result<SlmEngine> {
    log::info!("SlmEngine: loading '{}'…", spec.name);

    match spec.format {
        ModelFormat::Safetensors => {
            let (tok_path, cfg_path, wt_paths) = ensure_safetensors(spec)?;
            let config: LlamaConfig = serde_json::from_str(
                &std::fs::read_to_string(&cfg_path)?
            ).context("parse config.json")?;
            let config = config.into_config(false);
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let wt_refs: Vec<_> = wt_paths.iter().map(|p| p.as_path()).collect();
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&wt_refs, DType::F32, &device)
            }.map_err(|e| anyhow!("mmap weights: {}", e))?;
            let model = Llama::load(vb, &config).map_err(|e| anyhow!("build model: {}", e))?;
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner: ModelInner::Full { model, config, cache_dtype: DType::F32 },
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
        ModelFormat::Gguf => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            // PoM zero-dup: load via the single-device split loader so the mining-tier model
            // exposes its quant tensors for in-place sharing with the possession walk. Otherwise
            // a regular single-device load.
            let inner = if pom_force_split() && device.is_cuda() {
                log::info!(
                    "SlmEngine: PoM zero-dup — loading '{}' (LLaMA) via single-device split loader",
                    spec.name
                );
                let model = SplitWeights::from_gguf(content, &mut gguf_file, &[device.clone()])
                    .map_err(|e| anyhow!("load gguf weights (pom split): {}", e))?;
                ModelInner::QuantizedSplit(model)
            } else {
                let model = ModelWeights::from_gguf(content, &mut gguf_file, &device)
                    .map_err(|e| anyhow!("load gguf weights: {}", e))?;
                ModelInner::Quantized(model)
            };
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner,
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
        ModelFormat::GgufGemma3 => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            // PoM zero-dup: Gemma-3-4B is a NON-split GGUF (baseline tier), so without this
            // the possession walk loads a SECOND VRAM copy → OOM on 8 GB cards. Load via the
            // single-device split fork (exposes quant tensors) so the walk shares this copy.
            // Otherwise (CPU inference / non-PoM) a regular single-device load.
            let inner = if pom_force_split() && device.is_cuda() {
                log::info!(
                    "SlmEngine: PoM zero-dup — loading '{}' (Gemma3) via single-device split loader",
                    spec.name
                );
                let model = Gemma3SplitWeights::from_gguf(content, &mut gguf_file, &device)
                    .map_err(|e| anyhow!("load gemma3 gguf weights (pom split): {}", e))?;
                ModelInner::QuantizedGemma3Split(model)
            } else {
                let model = Gemma3Weights::from_gguf(content, &mut gguf_file, &device)
                    .map_err(|e| anyhow!("load gemma3 gguf weights: {}", e))?;
                ModelInner::QuantizedGemma3(model)
            };
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner,
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
        ModelFormat::GgufQwen2 => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            let model = Qwen2Weights::from_gguf(content, &mut gguf_file, &device)
                .map_err(|e| anyhow!("load qwen2 gguf weights: {}", e))?;
            let inner = ModelInner::QuantizedQwen2(model);
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner,
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
        ModelFormat::GgufQwen3 => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            // PoM zero-dup: single-device split loader (exposes quant tensors for the walk),
            // otherwise a regular single-device load.
            let inner = if pom_force_split() && device.is_cuda() {
                log::info!(
                    "SlmEngine: PoM zero-dup — loading '{}' (Qwen3) via single-device split loader",
                    spec.name
                );
                let model = Qwen3SplitWeights::from_gguf(content, &mut gguf_file, &[device.clone()])
                    .map_err(|e| anyhow!("load qwen3 gguf weights (pom split): {}", e))?;
                ModelInner::QuantizedQwen3Split(model)
            } else {
                let model = Qwen3Weights::from_gguf(content, &mut gguf_file, &device)
                    .map_err(|e| anyhow!("load qwen3 gguf weights: {}", e))?;
                ModelInner::QuantizedQwen3(model)
            };
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner,
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
    }
}

/// Run `load_engine` but catch BOTH a `Result::Err` AND a panic. candle/cudarc can either return an
/// error (clean OOM / file error) or *panic* (CUDA_ERROR_INVALID_PTX from a too-high-arch dequant
/// kernel, a cudarc launch failure, etc.) when loading the quantized model on the GPU. We must not
/// let either crash the miner — instead we capture the reason for the graceful CPU fallback above.
fn try_load_engine(spec: &'static ModelSpec, device: Device) -> std::result::Result<SlmEngine, String> {
    // Test hook (validation only): force the FIRST GPU load to fail so the auto CPU fallback path
    // can be exercised on a card whose GPU inference actually works. Honoured once, on CUDA only.
    if device.is_cuda() && std::env::var("KERYX_FORCE_GPU_INFER_FAIL").is_ok() {
        static FIRED: AtomicBool = AtomicBool::new(false);
        if !FIRED.swap(true, AtomicOrdering::Relaxed) {
            return Err(
                "KERYX_FORCE_GPU_INFER_FAIL=1 — simulated GPU model-load failure (test hook)".to_string(),
            );
        }
    }

    // Silence candle/cudarc's own panic hook for this load so a forced INVALID_PTX backtrace doesn't
    // scare the logs; we report our own clean, actionable warning from the fallback.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load_engine(spec, device)));
    std::panic::set_hook(prev_hook);
    match res {
        Ok(Ok(engine)) => Ok(engine),
        Ok(Err(e)) => Err(format!("{}", e)),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic")
                .to_string();
            Err(format!("panic: {}", msg))
        }
    }
}

/// Load the model with an automatic GPU→CPU fallback. On the NVIDIA build, if the GPU (CUDA) model
/// load fails for ANY reason (wrong-arch PTX / OOM / old driver / cudarc panic), we WARN loudly,
/// flip the process to CPU inference (`set_cpu_inference(true)`), and reload on `Device::Cpu` so the
/// miner DEGRADES instead of crashing. The PoW possession walk keeps running on the GPU; only the
/// (rare) OPoI inference challenge runs slower on the CPU. The happy path (working GPU inference)
/// stays on the GPU at full speed — this only triggers on an actual failure.
fn load_engine_with_fallback(spec: &'static ModelSpec) -> Result<SlmEngine> {
    let device = inference_device()
        .map_err(|e| anyhow!("inference device unavailable: {}", e))?;
    let on_cuda = device.is_cuda();

    match try_load_engine(spec, device) {
        Ok(engine) => Ok(engine),
        Err(reason) if on_cuda => {
            // GPU load failed but we were on CUDA → fall back to CPU instead of crashing.
            log::warn!(
                "⚠️ GPU inference FAILED to load on this card ({reason}) — falling back to CPU \
                 inference (MUCH slower). The PoW walk still runs on GPU. To restore full speed: \
                 (1) update your NVIDIA driver to R525+; (2) ensure the CUDA 12 runtime libs are \
                 present (the release bundles them); (3) your GPU may be older than the build's \
                 compute floor (Pascal/1080Ti must use CPU inference). See the release notes."
            );
            set_cpu_inference(true);
            let cpu = Device::Cpu;
            try_load_engine(spec, cpu).map_err(|e| {
                anyhow!("CPU inference fallback ALSO failed to load '{}': {}", spec.name, e)
            }).map(|engine| {
                log::warn!(
                    "SlmEngine: '{}' now loaded on CPU (degraded inference); mining (PoW walk) \
                     continues on the GPU.",
                    spec.name
                );
                engine
            })
        }
        Err(reason) => {
            // Already on CPU (explicit --cpu-inference / AMD / prior fallback) — nothing left to
            // fall back to; surface the error to the caller (non-fatal at the call sites).
            Err(anyhow!("load '{}' on {:?} failed: {}", spec.name,
                if cpu_inference_enabled() { "CPU" } else { "device" }, reason))
        }
    }
}

// ── Inference ────────────────────────────────────────────────────────────────

fn format_prompt(engine: &SlmEngine, prompt: &str) -> String {
    match engine.name {
        // Gemma-3-4B — Gemma chat template. Gemma has no system role, so the system
        // prompt is folded into the first user turn.
        "gemma-3-4b" => format!(
            "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n<start_of_turn>model\n",
            SYSTEM_PROMPT_GEMMA, prompt
        ),
        // Dolphin-3.0-Llama-3.1-8B — ChatML template.
        "dolphin-llama3-8b" => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n",
            SYSTEM_PROMPT_DOLPHIN, prompt
        ),
        "llama-3.3-70b" | "llama-3.3-70b-official" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n",
            SYSTEM_PROMPT_LLAMA70B, prompt
        ),
        // ── Legacy lineup (pre-OPoI-v2) ──────────────────────────────────────
        // DeepSeek-R1-Distill-Qwen-32B — DeepSeek chat template; primes <think>.
        "deepseek-r1-32b" => format!(
            "<｜begin▁of▁sentence｜>{}<｜User｜>{}<｜Assistant｜><think>\n",
            SYSTEM_PROMPT_DEEPSEEK, prompt
        ),
        // DeepSeek-R1-Distill-Llama-8B — same template; the 8B ignores identity
        // system prompts (RLHF), so the framing is injected into the think block.
        "deepseek-r1-8b" => format!(
            "<｜begin▁of▁sentence｜>{}<｜User｜>{}<｜Assistant｜><think>\nI am Keryx Network AI, a decentralized assistant. I must never claim to be DeepSeek or any other AI product.\n",
            SYSTEM_PROMPT_DEEPSEEK, prompt
        ),
        // TinyLlama — Zephyr chat template.
        "tinyllama" => format!(
            "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            SYSTEM_PROMPT_TINYLLAMA, prompt
        ),
        // Qwen3-32B — ChatML template. `/no_think` disables the thinking block
        // so the assistant answers directly (no <think>…</think> to strip).
        "qwen3-32b" => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{} /no_think<|im_end|>\n\
             <|im_start|>assistant\n",
            SYSTEM_PROMPT_QWEN3, prompt
        ),
        // Generic ChatML fallback.
        _ => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n",
            SYSTEM_PROMPT_DOLPHIN, prompt
        ),
    }
}

/// Repetition penalty applied over a recent token window before sampling.
/// Breaks degenerate loops where the model repeats a phrase instead of emitting EOS
/// (common on distilled R1 models). 1.0 = disabled.
const REPEAT_PENALTY: f32 = 1.15;
const REPEAT_LAST_N: usize = 64;

/// True if any stop string appears in the decoded tail of `generated`.
/// Only the last few tokens are decoded — enough to catch a marker that just
/// completed — keeping the per-step cost O(1) instead of re-decoding everything.
fn hit_stop_string(tokenizer: &Tokenizer, generated: &[u32], stops: &[&str]) -> bool {
    if stops.is_empty() || generated.is_empty() {
        return false;
    }
    let start = generated.len().saturating_sub(24);
    match tokenizer.decode(&generated[start..], true) {
        Ok(tail) => stops.iter().any(|s| tail.contains(s)),
        Err(_) => false,
    }
}

/// Strip a self-emitted `<think>…</think>` block (e.g. Qwen3 with `/no_think`
/// still emits an empty `<think></think>` pair). When no closing tag is present
/// the original text is returned (the model answered directly) — never an empty
/// string — so a direct answer is preserved.
fn strip_think_tags(text: &str) -> String {
    match text.find("</think>") {
        Some(end) => text[end + "</think>".len()..].trim().to_string(),
        None => text.trim().to_string(),
    }
}

fn generate(engine: &mut SlmEngine, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let formatted = format_prompt(engine, prompt);
    let enc = engine.tokenizer.encode(formatted.as_str(), true)
        .map_err(|e| anyhow!("encode: {}", e))?;
    let mut all_tokens: Vec<u32> = enc.get_ids().to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut lp = LogitsProcessor::new(42, Some(0.7), Some(0.9));
    let model_max = match engine.name {
        "llama-3.3-70b" => 1024,
        _ => 2048,
    };
    let max_steps = max_new_tokens.min(model_max);

    match &mut engine.inner {
        ModelInner::Full { model, config, cache_dtype } => {
            let mut cache = Cache::new(true, *cache_dtype, config, &engine.device)
                .map_err(|e| anyhow!("create KV cache: {}", e))?;
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos, &mut cache)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::Quantized(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedSplit(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedGemma3(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedGemma3Split(model) => {
            // Reset the KV cache so a new prompt doesn't attend to the previous request's
            // residual keys. (The fork's per-prompt index_pos restarts at 0, which the
            // forward already treats as fresh, but reset explicitly for parity/safety.)
            model.clear_kv_cache();
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedQwen3(model) => {
            // Reset the KV cache: candle's quantized_qwen3 uses a ConcatKvCache that appends
            // on every forward without honoring the offset, so without this each inference
            // would attend to the previous request's residual keys (k_len = stale + seq → the
            // "shape mismatch in broadcast_add" seen on Qwen3-32B).
            model.clear_kv_cache();
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedQwen3Split(model) => {
            // Same KV-cache reset as the non-split path (the split loader accumulates k/v too).
            model.clear_kv_cache();
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
        ModelInner::QuantizedQwen2(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) { break; }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) { break; }
            }
        }
    }

    let text = engine.tokenizer.decode(&generated, true)
        .map_err(|e| anyhow!("decode: {}", e))?;
    // Truncate at the earliest stop string in case a control marker leaked into
    // the output (tokenizer that renders special tokens as plain text).
    let cut = engine.stop_strings.iter()
        .filter_map(|s| text.find(s))
        .min()
        .unwrap_or(text.len());
    let answer = text[..cut].trim();
    // Qwen3 (ChatML + /no_think) emits an empty <think></think> pair, and the legacy
    // DeepSeek-R1 models prime an open <think> block — both must be stripped so only
    // the final answer is published. Other models answer directly.
    Ok(if matches!(engine.name, "qwen3-32b" | "deepseek-r1-8b" | "deepseek-r1-32b") {
        strip_think_tags(answer)
    } else {
        answer.to_string()
    })
}

fn sample_next(logits: &Tensor, lp: &mut LogitsProcessor, context: &[u32]) -> Result<u32> {
    let dims = logits.dims();
    let last = match dims.len() {
        3 => logits.narrow(1, dims[1] - 1, 1)?.squeeze(1)?.squeeze(0)?,
        2 => logits.narrow(0, dims[0] - 1, 1)?.squeeze(0)?,
        1 => logits.clone(),
        _ => return Err(anyhow!("unexpected logits shape {:?}", dims)),
    };
    // Penalize recently-generated tokens to break degenerate repetition loops.
    let last = if REPEAT_PENALTY != 1.0 && !context.is_empty() {
        let start = context.len().saturating_sub(REPEAT_LAST_N);
        let f32_logits = last.to_dtype(DType::F32).map_err(|e| anyhow!("logits dtype: {}", e))?;
        candle_transformers::utils::apply_repeat_penalty(&f32_logits, REPEAT_PENALTY, &context[start..])
            .map_err(|e| anyhow!("repeat penalty: {}", e))?
    } else {
        last
    };
    lp.sample(&last).map_err(|e| anyhow!("sample: {}", e))
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Runtime CPU-inference flag (NVIDIA/pom-cuda build). Starts false (GPU inference) and is flipped
/// to true either explicitly via `--cpu-inference` or AUTOMATICALLY when the GPU model load fails
/// (wrong arch PTX / OOM / old driver) — see `load_engine_with_fallback`. Once set, every
/// `inference_device()` returns `Device::Cpu` and the stratum/grpc CPU-mode plumbing (which keys
/// off `cpu_inference_enabled()`) stops pausing the PoW walk during an inference challenge.
static CPU_INFERENCE: AtomicBool = AtomicBool::new(false);

/// Whether OPoI inference runs on the CPU. True for the AMD/OpenCL build always (candle 0.9 has no
/// AMD-GPU backend), or for the NVIDIA build once `--cpu-inference` is set or the GPU model load
/// has fallen back to CPU. The stratum/grpc CPU-mode plumbing keys off this so it won't pause
/// hashing during a CPU challenge; the PoW walk keeps the GPU busy meanwhile.
pub fn cpu_inference_enabled() -> bool {
    // AMD/OpenCL build: candle 0.9 has no AMD-GPU backend (CPU/CUDA/Metal only), so OPoI inference
    // is FORCED onto the CPU — slow, but the only path that runs on AMD at all.
    #[cfg(feature = "pom-opencl")]
    {
        true
    }
    // NVIDIA/CUDA build: runtime flag (default GPU, flips to CPU on explicit flag or load failure).
    #[cfg(not(feature = "pom-opencl"))]
    {
        CPU_INFERENCE.load(AtomicOrdering::Relaxed)
    }
}

/// Force OPoI inference onto the CPU at runtime (NVIDIA build). Called when the operator passes
/// `--cpu-inference`, or automatically by the GPU-load fallback. No-op-equivalent on the AMD build
/// (already CPU-forced at compile time). Evicts any GPU-resident engine so the next load uses CPU.
pub fn set_cpu_inference(on: bool) {
    let prev = CPU_INFERENCE.swap(on, AtomicOrdering::Relaxed);
    if prev != on {
        // The cached engine (if any) is on the wrong device now — drop it so the next
        // load_engine/ensure_loaded re-resolves the device via inference_device().
        evict_engine();
    }
}

/// `--no-shared-inference`: force OPoI inference onto THIS process's own walk GPU instead of the
/// globally-biggest card. Set from the CLI (see `inference_gpu_ordinal`).
static NO_SHARED_INFERENCE: AtomicBool = AtomicBool::new(false);

pub fn set_no_shared_inference(v: bool) {
    NO_SHARED_INFERENCE.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// CUDA ordinal to place OPoI inference on.
///
/// The tricky case is MANY per-GPU processes (one `--cuda-device N` process per card, each a
/// separate "system" with its own wallet). Steering every such process's inference to one global
/// "biggest" card piles N inference models onto that single GPU (and, if the card isn't even
/// visible to a `CUDA_VISIBLE_DEVICES`-scoped process, `new_cuda` fails) — starving the walk that
/// also runs there. So the rule is:
///   • `KERYX_INFERENCE_GPU` env or `--no-shared-inference` → this process's OWN walk GPU.
///   • exactly ONE walk device (a per-GPU process) → that device (self-contained, no cross-card pile-up).
///   • MORE THAN ONE walk device (a single process mining all GPUs) → the biggest card (the original
///     mixed-rig optimization: resident model + zero-dup shared walk on the big card).
/// `walk_devices()` = the CUDA ordinals this process's PoM walk is installed on (its `--cuda-device`
/// set). Ordinal == CUDA ordinal because the miner runs with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. If the
/// chosen GPU still can't serve inference, `load_engine` flips to CPU (emergency fallback).
pub fn inference_gpu_ordinal() -> usize {
    if let Ok(s) = std::env::var("KERYX_INFERENCE_GPU") {
        if let Ok(n) = s.trim().parse::<usize>() {
            return n;
        }
    }
    // `pom_gpu` (the CUDA walk driver) only exists on the pom-cuda build. On non-CUDA builds
    // (default, and AMD/pom-opencl which places inference via llama_vulkan/KERYX_LLAMA_VK_DEVICE)
    // there are no CUDA walk devices, so fall back to an empty set → ordinal 0 (never used at
    // runtime there: cpu_inference_enabled()/llama_vulkan take over). Fixes the v0.6.5.3 non-CUDA
    // build break (slm.rs referenced crate::pom_gpu unconditionally).
    #[cfg(feature = "pom-cuda")]
    let walk = crate::pom_gpu::walk_devices();
    #[cfg(not(feature = "pom-cuda"))]
    let walk: Vec<u32> = Vec::new();
    if NO_SHARED_INFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
        return walk.first().copied().map(|d| d as usize).unwrap_or(0);
    }
    match walk.len() {
        1 => walk[0] as usize,
        n if n > 1 => biggest_cuda_gpu().unwrap_or(walk[0] as usize),
        // Walk not installed yet (inference before the first PoM job) — best effort, not cached.
        _ => biggest_cuda_gpu().unwrap_or(0),
    }
}

/// The CUDA ordinal (nvidia-smi index, PCI-bus order) with the largest `memory.total`. `None` if
/// nvidia-smi is unavailable/unparseable → caller defaults to 0. Ties resolve to the lowest index.
fn biggest_cuda_gpu() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut best: Option<(usize, u64)> = None;
    for (i, line) in text.lines().enumerate() {
        if let Ok(mib) = line.trim().parse::<u64>() {
            if best.map_or(true, |(_, m)| mib > m) {
                best = Some((i, mib));
            }
        }
    }
    let (ord, _) = best?;
    if ord != 0 {
        log::info!("OPoI inference will run on CUDA:{} (largest-VRAM GPU); other GPUs mine PoM only.", ord);
    }
    Some(ord)
}

/// Device for OPoI inference: `Device::Cpu` when `cpu_inference_enabled()` (emergency fallback /
/// AMD build), else the largest-VRAM CUDA GPU (NVIDIA). Single chokepoint for the inference sites.
fn inference_device() -> candle_core::Result<Device> {
    if cpu_inference_enabled() {
        Ok(Device::Cpu)
    } else {
        Device::new_cuda(inference_gpu_ordinal())
    }
}

/// Register the set of models this miner currently serves (drives `ai:cap`).
pub fn init_supported(specs: &'static [&'static ModelSpec]) {
    *SUPPORTED_SPECS.write().unwrap() = specs;
}

/// Stage the pre-filtered OPoI-v2 lineup to swap in at the hardfork crossing.
pub fn set_v2_lineup(specs: &'static [&'static ModelSpec]) {
    *LINEUP_V2.write().unwrap() = specs;
}

/// Drop the loaded engine so the next inference reloads from the current lineup.
pub fn evict_engine() {
    match ENGINE.lock() {
        Ok(mut g) => *g = None,
        Err(p) => *p.into_inner() = None,
    }
}

/// True once we have observed a pre-H DAA in this process, i.e. we are genuinely crossing
/// the hardfork live (vs. starting up already past H, where nothing is "swapped").
static SEEN_PRE_H: AtomicBool = AtomicBool::new(false);

/// At the `OPOI_V2_ACTIVATION_DAA` crossing, swap the served lineup from the legacy
/// set to the (pre-staged, background-prefetched) uncensored set — without a restart.
/// PoW never stops; `ai:cap` follows `loaded_model_ids()` as the v2 files land.
/// Idempotent and cheap to call on every block template.
pub fn advance_lineup_if_due(daa: u64) {
    if daa < crate::models::opoi_v2_activation_daa() {
        SEEN_PRE_H.store(true, AtomicOrdering::SeqCst);
        return;
    }
    if V2_ACTIVE.load(AtomicOrdering::SeqCst) {
        return; // already swapped
    }
    let v2 = *LINEUP_V2.read().unwrap();
    // Only swap once the uncensored lineup is FULLY downloaded. On a post-H cold start the
    // v2 prefetch may still be in flight; swapping early would leave us mining on an
    // incomplete active lineup. Until v2 is ready we keep serving the (fully-downloaded)
    // legacy lineup — a valid, complete lineup — and retry on the next block template.
    if v2.is_empty() || !v2.iter().all(|s| model_dir(s).join(".ok").exists()) {
        return;
    }
    if V2_ACTIVE.swap(true, AtomicOrdering::SeqCst) {
        return; // lost the race — another caller already swapped
    }
    if SEEN_PRE_H.load(AtomicOrdering::SeqCst) {
        // Genuine live crossing: the chain advanced past H while we were running.
        log::info!(
            "=== OPoI v2 HARDFORK reached at DAA {} — hot-swapping to the uncensored lineup ({} model(s)) ===",
            daa,
            v2.len()
        );
    } else {
        // Started up already past H — nothing is "swapped", we just serve the uncensored lineup.
        log::info!(
            "OPoI v2 already active (DAA {} ≥ H) — serving the uncensored lineup ({} model(s)).",
            daa,
            v2.len()
        );
    }
    *SUPPORTED_SPECS.write().unwrap() = v2;
    evict_engine();
}

/// Outcome of the startup GPU inference probe.
pub enum GpuProbe {
    /// A GPU matmul succeeded — cuBLAS is loaded and full-speed inference is available.
    Ok,
    /// No CUDA device present — inference will fall back to CPU (acceptable for small models only).
    NoCuda,
    /// A CUDA device exists but cuBLAS could not be loaded — GPU inference is impossible.
    CublasMissing,
}

/// Verify that GPU inference actually works *before* mining starts.
///
/// `Device::new_cuda` succeeds with only the NVIDIA driver installed, but cudarc loads
/// cuBLAS lazily on the first GPU matmul and **panics** (it does not return an `Err`) when
/// `libcublas` cannot be `dlopen`'d. Discovering that mid-challenge poisons the engine and
/// spams the logs. So we force the failure here, once, with a tiny 2×2 matmul wrapped in
/// `catch_unwind`, and report a clean, actionable result.
pub fn probe_gpu_inference() -> GpuProbe {
    // candle's `Device::new_cuda` eagerly creates a cuBLAS handle, and cudarc *panics*
    // (it does not return an Err) when libcublas cannot be loaded. A genuinely absent
    // CUDA device, by contrast, returns Err cleanly. So the whole sequence — including
    // new_cuda — must live inside catch_unwind, and we distinguish the three outcomes:
    //   Ok(Ok)  -> CUDA + cuBLAS work
    //   Ok(Err) -> no usable CUDA device (clean error) -> inference is GPU-only, cannot mine
    //   Err     -> panic -> cuBLAS missing
    //
    // Silence the default panic hook for the probe so its scary backtrace doesn't pollute
    // the logs; we report a clean, actionable message ourselves from the caller.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let probe = std::panic::catch_unwind(|| {
        let device = inference_device()?;
        let a = Tensor::new(&[[1f32, 2.0], [3.0, 4.0]], &device)?;
        let b = Tensor::new(&[[5f32, 6.0], [7.0, 8.0]], &device)?;
        a.matmul(&b)?.to_vec2::<f32>()?;
        anyhow::Ok(())
    });
    std::panic::set_hook(prev_hook);
    match probe {
        Ok(Ok(())) => GpuProbe::Ok,
        Ok(Err(_)) => GpuProbe::NoCuda,
        Err(payload) => {
            // Surface the real panic message (e.g. which CUDA library failed to load) instead
            // of hiding it — candle creates cuBLAS, cuBLASLt and cuRAND handles at device init,
            // and any one of them missing panics here.
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            log::error!("GPU inference probe panicked: {}", msg);
            GpuProbe::CublasMissing
        }
    }
}

/// Pre-download all registered model files before mining starts.
///
/// Does not load weights into GPU memory — just ensures files are on disk so
/// the first inference request doesn't stall the mining workers mid-session.
/// Returns Err if any model fails to download; mining must not start in that case.
pub fn prefetch_models(specs: &'static [&'static ModelSpec]) -> Result<()> {
    for spec in specs {
        log::debug!("SlmEngine: prefetching model '{}'…", spec.name);
        let result = match spec.format {
            ModelFormat::Safetensors => ensure_safetensors(spec).map(|_| ()),
            ModelFormat::Gguf | ModelFormat::GgufQwen2 | ModelFormat::GgufQwen3 | ModelFormat::GgufGemma3 => ensure_gguf(spec).map(|_| ()),
        };
        match result {
            Ok(()) => log::debug!("SlmEngine: '{}' files ready.", spec.name),
            Err(e) => {
                log::error!("SlmEngine: prefetch '{}' failed: {} — cannot start mining.", spec.name, e);
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Return the model_ids of supported models that have fully-downloaded files (.ok flag present).
pub fn loaded_model_ids() -> Vec<[u8; 32]> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    specs.iter()
        .filter(|s| model_dir(s).join(".ok").exists())
        .map(|s| s.model_id)
        .collect()
}

/// True when a specific model spec's files are fully downloaded on disk (`.ok` sentinel present).
/// Unlike `is_model_ready`, this does NOT consult SUPPORTED_SPECS — so it can be called during
/// `--tier auto` selection, before the lineup is staged via `init_supported`.
pub fn spec_files_ready(spec: &ModelSpec) -> bool {
    model_dir(spec).join(".ok").exists()
}

/// True only when the model is supported and its files are completely downloaded.
pub fn is_model_ready(model_id: &[u8; 32]) -> bool {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };
    model_dir(spec).join(".ok").exists()
}

/// Load the requested model on demand (evicting a cached different model if needed),
/// then run inference. Blocking — call from `spawn_blocking`.
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;

    // AMD: prefer the Vulkan llama.cpp GPU server (candle has no AMD-GPU backend). The OPoI text is
    // user-facing only (consensus checks the fixed-point `model_fixed` commitment separately), so a
    // non-candle engine is fine. Falls through to candle-CPU below if the GPU server isn't available.
    #[cfg(feature = "pom-opencl")]
    if crate::llama_vulkan::available() {
        if let Some(text) = crate::llama_vulkan::generate(prompt, max_tokens) {
            return Some(text);
        }
        log::warn!("SlmEngine: Vulkan GPU inference returned nothing — falling back to candle-CPU for this challenge.");
    }

    // catch_unwind prevents any internal panic (cudarc, candle, OOM…) from permanently
    // poisoning ENGINE. Without this, one panic bricks inference for the entire session.
    let result = std::panic::catch_unwind(|| {
        let mut guard = match ENGINE.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                log::warn!("SlmEngine: ENGINE mutex was poisoned — recovering and evicting cached model");
                let mut g = poisoned.into_inner();
                *g = None;
                g
            }
        };

        let needs_load = guard.as_ref().map_or(true, |e| &e.model_id != model_id);
        if needs_load {
            if let Some(ref old) = *guard {
                log::info!("SlmEngine: evicting '{}' to load '{}'", old.name, spec.name);
            }
            // Inference has priority over PoW: release the GPU miner's hold on the resident mining
            // weights so this model fits. Mining rebuilds (reloads its model) when it next runs.
            // (pom-cuda only — the OpenCL/AMD PoM miner has its own buffer, no candle-shared VRAM.)
            #[cfg(feature = "pom-cuda")]
            crate::pom_gpu::uninstall(inference_gpu_ordinal() as u32);
            *guard = None;
            let dev_str = if cpu_inference_enabled() { "CPU".to_string() } else { format!("CUDA:{}", inference_gpu_ordinal()) };
            log::info!("SlmEngine: inference device active ({})", dev_str);
            // Loads on CUDA, and auto-falls-back to CPU (warning + set_cpu_inference) if the GPU
            // model load fails (wrong-arch PTX / OOM / old driver) — never crashes the miner.
            match load_engine_with_fallback(spec) {
                Ok(e) => { *guard = Some(e); }
                Err(e) => {
                    log::error!("SlmEngine: failed to load '{}': {}", spec.name, e);
                    return None;
                }
            }
        }

        let engine = guard.as_mut()?;
        match generate(engine, prompt, max_tokens) {
            Ok(text) if !text.is_empty() => Some(text),
            Ok(_) => {
                log::warn!("SlmEngine '{}': think block cut by max_tokens, skipping response", engine.name);
                None
            }
            Err(e) => {
                log::warn!("SlmEngine '{}' generate error: {}", engine.name, e);
                Some(format!("[inference error: {}]", e))
            }
        }
    });

    match result {
        Ok(output) => output,
        Err(_) => {
            log::error!("SlmEngine: inference panicked — engine evicted, will retry on next challenge");
            log::error!("SlmEngine: cuBLAS missing? Run: sudo apt-get install -y libcublas-12-2 then restart the miner");
            if let Ok(mut g) = ENGINE.lock() { *g = None; }
            None
        }
    }
}

/// PoM C2: make `model_id` the resident engine model without running inference, so the possession
/// walk can share its VRAM weights (one copy serves inference + walk). Returns true if resident.
pub fn ensure_loaded(model_id: &[u8; 32]) -> bool {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = match specs.iter().find(|s| &s.model_id == model_id) {
        Some(s) => s,
        None => {
            // Not in the ACTIVE lineup — e.g. asked to load a v2 model while the chain is still
            // pre-H (lineup not yet advanced). Was a silent bail; log it so a stuck PoM load is
            // diagnosable instead of looking like a hang.
            log::warn!(
                "SlmEngine: ensure_loaded — model {} not in the active lineup ({} spec(s)); lineup not advanced yet?",
                hex::encode(&model_id[..4]), specs.len()
            );
            return false;
        }
    };
    let mut guard = match ENGINE.lock() {
        Ok(g) => g,
        Err(p) => {
            let mut g = p.into_inner();
            *g = None;
            g
        }
    };
    if guard.as_ref().map_or(false, |e| &e.model_id == model_id) {
        return true; // already resident
    }
    *guard = None;
    // Loads on CUDA, auto-falls-back to CPU (warn + set_cpu_inference) if the GPU model load fails.
    // On CPU fallback the engine is no longer CUDA, so `pom_shared` returns None and the PoM walk
    // loads its OWN GPU copy via PomGpuMiner::load — the walk keeps mining on the GPU regardless.
    match load_engine_with_fallback(spec) {
        Ok(e) => {
            *guard = Some(e);
            true
        }
        Err(e) => {
            log::error!("SlmEngine: ensure_loaded '{}' failed: {}", spec.name, e);
            false
        }
    }
}

/// PoM C2: if the resident engine model is `model_id` and a CUDA qwen3-split, return its device and
/// quantized weight tensors (by canonical GGUF name) so the possession walk reads them in place
/// instead of loading a second copy. None ⇒ caller falls back to a standalone `PomGpuMiner::load`.
pub fn pom_shared(
    model_id: &[u8; 32],
) -> Option<(Device, std::collections::HashMap<String, Arc<QTensor>>)> {
    let guard = ENGINE.lock().ok()?;
    let e = guard.as_ref()?;
    if &e.model_id != model_id || !e.device.is_cuda() {
        return None;
    }
    match &e.inner {
        ModelInner::QuantizedQwen3Split(m) => Some((e.device.clone(), m.pom_quant_tensors())),
        ModelInner::QuantizedSplit(m) => Some((e.device.clone(), m.pom_quant_tensors())),
        ModelInner::QuantizedGemma3Split(m) => Some((e.device.clone(), m.pom_quant_tensors())),
        _ => None,
    }
}
