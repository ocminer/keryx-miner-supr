/// Phase-3 OPoI: multi-model inference engine (safetensors + GGUF) via candle.
///
/// Models are loaded on demand when an AiRequest arrives and cached between
/// consecutive requests for the same model. Mining pauses during inference.
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, LlamaConfig, Llama};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;

use crate::models::{ModelFormat, ModelSpec};

const IPFS_GATEWAY: &str = "https://keryx-labs.com";
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

const SYSTEM_PROMPT_LLAMA70B: &str =
    "You are a Keryx Network AI — a high-capability decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Meta, Llama, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

// ── Static engine state ──────────────────────────────────────────────────────

static SUPPORTED_SPECS: OnceLock<&'static [&'static ModelSpec]> = OnceLock::new();
static ENGINE: Mutex<Option<SlmEngine>> = Mutex::new(None);

/// When true, OPoI inference is forced onto the CPU even if a CUDA device is present.
/// Set once at startup from the `--cpu-inference` CLI flag so weak GPUs (e.g. GTX 1060)
/// can dedicate the GPU to hashing while the CPU runs the SLM.
static FORCE_CPU_INFERENCE: AtomicBool = AtomicBool::new(false);

/// Force OPoI inference onto the CPU (see [`FORCE_CPU_INFERENCE`]). Call once at startup.
pub fn set_cpu_inference(enabled: bool) {
    FORCE_CPU_INFERENCE.store(enabled, AtomicOrdering::Relaxed);
}

/// Whether OPoI inference is forced onto the CPU.
pub fn cpu_inference_enabled() -> bool {
    FORCE_CPU_INFERENCE.load(AtomicOrdering::Relaxed)
}

enum ModelInner {
    Full { model: Llama, config: Config, cache_dtype: DType },
    Quantized(ModelWeights),
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

fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    eprintln!("[keryx-miner] Downloading {} ...", url);
    let response = ureq::get(url)
        .call()
        .map_err(|e| anyhow!("HTTP GET {}: {}", url, e))?;
    let content_length: Option<u64> = response.header("Content-Length").and_then(|s| s.parse().ok());
    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(dest)
        .with_context(|| format!("create {}", dest.display()))?;
    let mut downloaded: u64 = 0;
    let mut buf = vec![0u8; 65_536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 { break; }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        if let Some(total) = content_length {
            eprint!("\r  {:.1}/{:.1} MB ({}%)   ",
                downloaded as f64 / 1_000_000.0,
                total as f64 / 1_000_000.0,
                downloaded * 100 / total);
            let _ = std::io::stderr().flush();
        }
    }
    eprintln!();
    Ok(())
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
        log::info!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
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
        log::info!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
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
/// (DeepSeek-R1-8B and LLaMA-3.3-70B are both LLaMA-arch GGUF) yet need different
/// stop conventions, so this branches on name rather than format.
fn stop_config(tokenizer: &Tokenizer, name: &str) -> (Vec<u32>, Vec<&'static str>) {
    match name {
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
        // Genuine LLaMA-3.3-70B-Instruct — LLaMA-3 header template. The DeepSeek
        // full-width markers do NOT exist in this vocab, so stop on the real
        // LLaMA-3 terminators (the official `eos_token_id` set for 3.3 Instruct):
        //   128009 = <|eot_id|> (end of turn), 128001 = <|end_of_text|> (base EOS),
        //   128008 = <|eom_id|> (end of message, tool turns).
        "llama-3.3-70b" => (
            collect_stop_ids(tokenizer,
                &["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"],
                &[128009, 128001, 128008]),
            // Cut if the model tries to open a fresh turn instead of stopping.
            vec!["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
        ),
        // TinyLlama / Zephyr: </s> ends a turn; 0 = padding safety net.
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
            let model = ModelWeights::from_gguf(content, &mut gguf_file, &device)
                .map_err(|e| anyhow!("load gguf weights: {}", e))?;
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner: ModelInner::Quantized(model),
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
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner: ModelInner::QuantizedQwen2(model),
                tokenizer, device, stop_token_ids, stop_strings,
            })
        }
    }
}

// ── Inference ────────────────────────────────────────────────────────────────

fn format_prompt(engine: &SlmEngine, prompt: &str) -> String {
    match engine.name {
        // DeepSeek-R1-Distill-Qwen-32B — DeepSeek chat template.
        // The 32B respects the system prompt, so plain think priming is enough.
        "deepseek-r1-32b" => format!(
            "<｜begin▁of▁sentence｜>{}<｜User｜>{}<｜Assistant｜><think>\n",
            SYSTEM_PROMPT_DEEPSEEK, prompt
        ),
        // DeepSeek-R1-Distill-Llama-8B — same template but the 8B ignores system
        // prompts about identity due to RLHF. Inject an identity statement at the
        // start of the think block so the model reasons from the correct framing
        // before generating the visible answer.
        "deepseek-r1-8b" => format!(
            "<｜begin▁of▁sentence｜>{}<｜User｜>{}<｜Assistant｜><think>\nI am Keryx Network AI, a decentralized assistant. I must never claim to be DeepSeek or any other AI product.\n",
            SYSTEM_PROMPT_DEEPSEEK, prompt
        ),
        "llama-3.3-70b" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n",
            SYSTEM_PROMPT_LLAMA70B, prompt
        ),
        // Zephyr/TinyLlama chat template
        _ => format!(
            "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            SYSTEM_PROMPT_TINYLLAMA, prompt
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

/// Strip the `<think>…</think>` reasoning block emitted by DeepSeek-R1 models.
///
/// The prompt primes the assistant turn with `<think>` already open, so the
/// generated text begins inside the block: `reasoning…\n</think>\nfinal answer`.
/// Returns the trimmed final answer, or an empty string if `</think>` was never
/// emitted (max_tokens exhausted during reasoning — no final answer produced).
fn strip_think_block(text: &str) -> String {
    match text.find("</think>") {
        Some(end) => text[end + "</think>".len()..].trim().to_string(),
        None => String::new(),
    }
}

fn generate(engine: &mut SlmEngine, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let formatted = format_prompt(engine, prompt);
    // Detect whether the assistant turn was primed with <think> (R1 models).
    // Uses contains() to handle both plain priming and think-injected variants.
    let is_think_primed = formatted.contains("<｜Assistant｜><think>");
    let enc = engine.tokenizer.encode(formatted.as_str(), true)
        .map_err(|e| anyhow!("encode: {}", e))?;
    let mut all_tokens: Vec<u32> = enc.get_ids().to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut lp = LogitsProcessor::new(42, Some(0.7), Some(0.9));
    let model_max = match engine.name {
        "deepseek-r1-8b" => 1024,
        "deepseek-r1-32b" => 2048,
        "llama-3.3-70b" => 1024,
        _ => 2048,
    };
    // R1 models spend tokens on chain-of-thought before the visible answer.
    // Reserve extra budget so the think block never eats the entire quota and
    // leaves no room for </think> + the actual response.
    // R1-32B reasons more extensively than 8B — give it more overhead to avoid
    // the think block consuming all steps and producing an empty answer.
    let think_overhead = if is_think_primed {
        match engine.name {
            "deepseek-r1-32b" => 1536,
            _ => 512,
        }
    } else {
        0
    };
    let max_steps = max_new_tokens.saturating_add(think_overhead).min(model_max);

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
    // For R1 models, strip the <think>…</think> reasoning preamble so only the
    // final answer is published. Empty string = think block was cut by max_tokens
    // (no final answer produced) — the caller should not upload this to IPFS.
    Ok(if is_think_primed { strip_think_block(answer) } else { answer.to_string() })
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

/// Register the set of models this miner supports. Called once at startup.
pub fn init_supported(specs: &'static [&'static ModelSpec]) {
    let _ = SUPPORTED_SPECS.set(specs);
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
    //   Ok(Err) -> no usable CUDA device (clean error) -> CPU fallback
    //   Err     -> panic -> cuBLAS missing
    //
    // Silence the default panic hook for the probe so its scary backtrace doesn't pollute
    // the logs; we report a clean, actionable message ourselves from the caller.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let probe = std::panic::catch_unwind(|| {
        let device = Device::new_cuda(0)?;
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
        log::info!("SlmEngine: prefetching model '{}'…", spec.name);
        let result = match spec.format {
            ModelFormat::Safetensors => ensure_safetensors(spec).map(|_| ()),
            ModelFormat::Gguf | ModelFormat::GgufQwen2 => ensure_gguf(spec).map(|_| ()),
        };
        match result {
            Ok(()) => log::info!("SlmEngine: '{}' files ready.", spec.name),
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
    SUPPORTED_SPECS.get()
        .map(|specs| specs.iter()
            .filter(|s| model_dir(s).join(".ok").exists())
            .map(|s| s.model_id)
            .collect())
        .unwrap_or_default()
}

/// True only when the model is supported and its files are completely downloaded.
pub fn is_model_ready(model_id: &[u8; 32]) -> bool {
    let Some(specs) = SUPPORTED_SPECS.get() else { return false; };
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };
    model_dir(spec).join(".ok").exists()
}

/// Load the requested model on demand (evicting a cached different model if needed),
/// then run inference. Blocking — call from `spawn_blocking`.
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    let specs = SUPPORTED_SPECS.get()?;
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;

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
            *guard = None;
            let device = if cpu_inference_enabled() {
                log::info!("SlmEngine: --cpu-inference set — running on CPU device");
                Device::Cpu
            } else {
                match Device::new_cuda(0) {
                    Ok(d) => { log::info!("SlmEngine: CUDA device 0 active"); d }
                    Err(e) => { log::warn!("SlmEngine: CUDA unavailable ({e}) — CPU fallback"); Device::Cpu }
                }
            };
            match load_engine(spec, device) {
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
