use crate::quantized_gemma3_split::ModelWeights as Gemma3SplitWeights;
use crate::quantized_llama_split::ModelWeights as SplitWeights;
use crate::quantized_qwen3_split::ModelWeights as Qwen3SplitWeights;
/// Phase-3 OPoI: multi-model inference engine (safetensors + GGUF) via candle.
///
/// Models are loaded on demand when an AiRequest arrives and cached between
/// consecutive requests for the same model. Mining pauses during inference.
use anyhow::{anyhow, Context, Result};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use candle_transformers::models::quantized_gemma3::ModelWeights as Gemma3Weights;
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3Weights;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use tokenizers::Tokenizer;

use crate::models::{ModelFormat, ModelSpec};

const IPFS_GATEWAY: &str = "https://keryx-labs.com";

// The interactive dashboard owns stdout/stderr while its alternate screen is active. Preserve
// the existing download output byte-for-byte in classic mode, but never let carriage-return
// progress (or an exceptional direct diagnostic) punch through the dashboard. Structured runtime
// state and the ordinary logger remain visible inside the TUI.
fn tui_owns_terminal() -> bool {
    crate::tui_active()
}

macro_rules! classic_eprint {
    ($($arg:tt)*) => {
        if !tui_owns_terminal() {
            eprint!($($arg)*);
        }
    };
}

macro_rules! classic_eprintln {
    ($($arg:tt)*) => {
        if !tui_owns_terminal() {
            eprintln!($($arg)*);
        }
    };
}
/// Remote inference admission limits. These are enforced again at the engine boundary (not only
/// by the Stratum/gRPC parsers), because inference is also called by startup and self-test paths.
pub const MAX_INFERENCE_PROMPT_BYTES: usize = 4 * 1024;
pub const MAX_INFERENCE_TOKENS: usize = 2_048;
pub const DEFAULT_INFERENCE_DEADLINE_MS: u64 = 30_000;
pub const MAX_INFERENCE_DEADLINE_MS: u64 = 120_000;

/// Validate untrusted inference controls and return the effective routing deadline. A zero
/// deadline means the protocol default; oversized values are rejected instead of silently turning
/// into a very long queue wait.
pub fn validate_inference_request(
    prompt: &str,
    max_tokens: usize,
    deadline_ms: u64,
) -> std::result::Result<u64, &'static str> {
    if prompt.is_empty() {
        return Err("prompt is empty");
    }
    if prompt.len() > MAX_INFERENCE_PROMPT_BYTES {
        return Err("prompt exceeds 4096 bytes");
    }
    if max_tokens == 0 {
        return Err("max_tokens must be at least 1");
    }
    if max_tokens > MAX_INFERENCE_TOKENS {
        return Err("max_tokens exceeds 2048");
    }
    let effective = if deadline_ms == 0 { DEFAULT_INFERENCE_DEADLINE_MS } else { deadline_ms };
    if effective > MAX_INFERENCE_DEADLINE_MS {
        return Err("deadline_ms exceeds 120000");
    }
    Ok(effective)
}
// Legacy lineup (pre-OPoI-v2) system prompts.
const SYSTEM_PROMPT_TINYLLAMA: &str = "You are a Keryx Network AI — a decentralized assistant running on GPU miners. \
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
    Full {
        model: Llama,
        config: Config,
        cache_dtype: DType,
    },
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

/// `--model-dir` override: when set, models are looked up AND downloaded under this directory
/// instead of `<exe_dir>/models`. Set once at CLI-processing time (before any staging/prefetch).
static MODEL_DIR_OVERRIDE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// Install the `--model-dir` override (validated by the CLI). First call wins; must run before
/// model staging/prefetch so every `model_dir()` lookup and download uses the same root.
pub fn set_model_dir(dir: std::path::PathBuf) {
    let _ = MODEL_DIR_OVERRIDE.set(dir);
}

fn model_dir(spec: &ModelSpec) -> std::path::PathBuf {
    if let Some(root) = MODEL_DIR_OVERRIDE.get() {
        return root.join(spec.dir_name);
    }
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
/// The most recent model-staging failure (download / disk / corrupt file / GPU load), in
/// plain actionable English. The mining loop surfaces this LOUDLY and repeatedly while mining is
/// suspended, so an operator who only reads the last few log lines still sees WHAT went wrong (not
/// just the reassuring "preparing…" spinner). Cleared the moment a model goes ready.
static LAST_STAGING_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();
fn staging_error_slot() -> &'static Mutex<Option<String>> {
    LAST_STAGING_ERROR.get_or_init(|| Mutex::new(None))
}
/// Record a staging failure (also logged at ERROR by the caller). Shown by the miner status loop.
pub fn set_staging_error(msg: impl Into<String>) {
    if let Ok(mut g) = staging_error_slot().lock() {
        *g = Some(msg.into());
    }
    crate::runtime_stats::set_staging_error(true);
}
/// Clear the staging failure once a model is ready again.
pub fn clear_staging_error() {
    if let Ok(mut g) = staging_error_slot().lock() {
        *g = None;
    }
    crate::runtime_stats::set_staging_error(false);
}
/// The last recorded staging failure, if any (for the miner status line).
pub fn last_staging_error() -> Option<String> {
    staging_error_slot().lock().ok().and_then(|g| g.clone())
}

/// Best-effort free bytes on the filesystem holding `dir` (or its parent), via POSIX `df -kP`.
/// `None` (Windows, or `df` unavailable) → callers skip the pre-check; a real ENOSPC still surfaces
/// loudly at write time.
#[cfg(unix)]
fn free_disk_bytes(path: &std::path::Path) -> Option<u64> {
    let dir = if path.is_dir() { path } else { path.parent().unwrap_or(path) };
    let out = std::process::Command::new("df").arg("-kP").arg(dir).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let avail_kb: u64 = text.lines().last()?.split_whitespace().nth(3)?.parse().ok()?;
    Some(avail_kb.saturating_mul(1024))
}
#[cfg(not(unix))]
fn free_disk_bytes(_path: &std::path::Path) -> Option<u64> {
    None
}

fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 240; // survives long gateway outages (~40 min of retries)
    const BACKOFF_SECS: u64 = 10;
    // Mark the process as "downloading" for the whole fetch (incl. retries/resume) so the
    // hashrate reporter shows "downloading model" instead of "workers stalled or crashed".
    // RAII guard clears it on every exit path (success, error, early return).
    DOWNLOADING.store(true, AtomicOrdering::Relaxed);
    struct DlGuard;
    impl Drop for DlGuard {
        fn drop(&mut self) {
            DOWNLOADING.store(false, AtomicOrdering::Relaxed);
        }
    }
    let _dl = DlGuard;
    classic_eprintln!("[keryx-miner] Downloading {} ...", url);
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
            // A resumed request for an already-complete file gets HTTP 416 (Range Not Satisfiable),
            // which ureq surfaces as Err(Status(416, _)) — so the `status == 416` arm below is never
            // reached. Treat it here as "already fully downloaded" (e.g. a GGUF pre-fetched via
            // aria2, or a re-run over a complete file) instead of looping on it forever.
            Err(ureq::Error::Status(416, _)) if resume_from > 0 => {
                classic_eprintln!("  already complete ({} MB).", resume_from / 1_000_000);
                return Ok(());
            }
            Err(e) => {
                attempt += 1;
                if attempt >= MAX_ATTEMPTS {
                    let msg = format!(
                        "DOWNLOAD FAILED: could not fetch {} after {} attempts ({}). The IPFS gateway is \
                         unreachable or blocked — check this rig's internet/DNS/firewall, or set \
                         KERYX_IPFS_GATEWAY to a working gateway. Mining stays suspended until the model downloads.",
                        url, attempt, e
                    );
                    log::error!("[keryx-miner] {msg}");
                    set_staging_error(&msg);
                    return Err(anyhow!(msg));
                }
                classic_eprintln!("\n[keryx-miner] connect error ({e}); retry {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s (resume @ {} MB)…",
                    resume_from / 1_000_000);
                std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
                continue;
            }
        };
        let status = response.status();

        // Decide whether to append (server honored the range) or (re)start, and the total size.
        let (mut file, mut downloaded, total): (std::fs::File, u64, Option<u64>) = if resume_from > 0 && status == 206 {
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
            classic_eprintln!("\r  already complete ({} MB).            ", resume_from / 1_000_000);
            return Ok(());
        } else {
            // 200, or the server ignored Range. Never wipe a local file that already matches
            // the remote size — IPFS gateways often ignore Range and answer 200 + full
            // Content-Length, which previously truncated multi-GB GGUFs back to zero.
            let total = response.header("Content-Length").and_then(|s| s.parse::<u64>().ok());
            if resume_from > 0 {
                if let Some(t) = total {
                    if resume_from >= t {
                        classic_eprintln!("\r  already complete ({} MB).            ", resume_from / 1_000_000);
                        return Ok(());
                    }
                }
                // Partial local file + no Range support: keep the bytes and resume via a
                // fresh request without Range only when we have nothing useful; otherwise
                // refuse to truncate and retry later (gateway may regain Range support).
                if resume_from > 1_000_000 {
                    drop(response);
                    attempt += 1;
                    if attempt >= MAX_ATTEMPTS {
                        return Err(anyhow!(
                            "download {} cannot resume: server ignored Range and local partial is {} MB",
                            url,
                            resume_from / 1_000_000
                        ));
                    }
                    classic_eprintln!(
                            "\n[keryx-miner] server ignored Range (HTTP {status}); keeping local {} MB, retry {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s…",
                            resume_from / 1_000_000
                        );
                    std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
                    continue;
                }
            }
            let f = std::fs::File::create(dest).with_context(|| format!("create {}", dest.display()))?;
            (f, 0u64, total)
        };

        // DISK-SPACE PREFLIGHT: refuse to start a download we can't finish, and say exactly where +
        // how short we are, instead of streaming until ENOSPC and dying with an opaque OS error.
        if let Some(t) = total {
            let need = t.saturating_sub(downloaded);
            const MARGIN: u64 = 512_000_000; // headroom for the possession index / temp files
            if let Some(free) = free_disk_bytes(dest) {
                if free < need.saturating_add(MARGIN) {
                    let where_dir = dest.parent().unwrap_or(dest);
                    let msg = format!(
                        "NOT ENOUGH DISK SPACE for {}: need ~{} MB more (download {} MB + 512 MB headroom) \
                         but only {} MB free on the filesystem holding {}. Free up space or move --model-dir \
                         to a bigger disk, then restart the miner.",
                        dest.display(),
                        (need + MARGIN) / 1_000_000,
                        need / 1_000_000,
                        free / 1_000_000,
                        where_dir.display()
                    );
                    log::error!("[keryx-miner] {msg}");
                    classic_eprintln!("\n[keryx-miner] ERROR: {msg}\n");
                    set_staging_error(&msg);
                    return Err(anyhow!(msg));
                }
            }
        }

        let mut reader = response.into_reader();
        let mut buf = vec![0u8; 65_536];
        let mut stream_err: Option<String> = None;
        // Progress: the `\r` line is for interactive terminals only — through the HiveOS log tee
        // its fragments concatenate into one unreadable line, so rigs looked "hung" for the whole
        // multi-GB download ("models are not downloading correctly" reports). Emit a REAL log line
        // every 30 s with speed + ETA so the miner screen / agent log shows live progress, and say
        // explicitly that mining starts only after the download.
        let seg_start = std::time::Instant::now();
        let seg_base = downloaded;
        let mut last_log = std::time::Instant::now();
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
                        classic_eprint!(
                            "\r  {:.1}/{:.1} MB ({}%)   ",
                            downloaded as f64 / 1_000_000.0,
                            t as f64 / 1_000_000.0,
                            downloaded * 100 / t.max(1)
                        );
                        let _ = std::io::stderr().flush();
                        if last_log.elapsed().as_secs() >= 30 {
                            last_log = std::time::Instant::now();
                            let rate = (downloaded - seg_base) as f64 / seg_start.elapsed().as_secs_f64().max(0.001);
                            let eta_min = t.saturating_sub(downloaded) as f64 / rate.max(1.0) / 60.0;
                            log::info!(
                                "model download: {:.0}/{:.0} MB ({}%), {:.1} MB/s, ETA ~{:.0} min — mining starts \
                                 after the download + possession-index build. Do NOT restart the miner: the \
                                 download resumes, but every restart prolongs it (disable rig watchdogs until \
                                 the first accepted share).",
                                downloaded as f64 / 1e6,
                                t as f64 / 1e6,
                                downloaded * 100 / t.max(1),
                                rate / 1e6,
                                eta_min.max(1.0),
                            );
                        }
                    }
                }
                Err(e) => {
                    stream_err = Some(e.to_string());
                    break;
                }
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
            classic_eprintln!();
            return Ok(());
        }

        attempt += 1;
        if attempt >= MAX_ATTEMPTS {
            let msg = format!(
                "DOWNLOAD FAILED: {} kept interrupting after {} attempts (got {} MB). Unstable link or a \
                 flaky IPFS gateway — check this rig's internet, or set KERYX_IPFS_GATEWAY to a working \
                 gateway. The partial file is kept and will resume on restart; mining stays suspended until it completes.",
                url, attempt, downloaded / 1_000_000
            );
            log::error!("[keryx-miner] {msg}");
            set_staging_error(&msg);
            return Err(anyhow!(msg));
        }
        let why = stream_err.unwrap_or_else(|| "short read".into());
        classic_eprintln!(
            "\n[keryx-miner] interrupted ({why}); resuming {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s @ {} MB…",
            downloaded / 1_000_000
        );
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
    let wts: Vec<_> = spec
        .weight_cids
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if spec.weight_cids.len() == 1 {
                dir.join("model.safetensors")
            } else {
                dir.join(format!("model-{:05}-of-{:05}.safetensors", i + 1, spec.weight_cids.len()))
            }
        })
        .collect();

    // .ok sentinel written only after a complete download — guards against truncated files
    if tok.exists() && cfg.exists() && wts.iter().all(|p| p.exists()) && ok_flag.exists() {
        log::debug!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, cfg, wts));
    }
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before re-downloading
    classic_eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if !tok.exists() {
        download_file(&ipfs_url(spec.tokenizer_cid), &tok)?;
    }
    if !cfg.exists() {
        download_file(&ipfs_url(spec.config_cid), &cfg)?;
    }
    for (i, (cid, path)) in spec.weight_cids.iter().zip(wts.iter()).enumerate() {
        if spec.weight_cids.len() > 1 {
            classic_eprintln!("[keryx-miner] Shard {}/{}", i + 1, spec.weight_cids.len());
        }
        download_file(&ipfs_url(cid), path)?;
    }
    std::fs::write(&ok_flag, b"").with_context(|| format!("write .ok flag {}", ok_flag.display()))?;
    classic_eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, cfg, wts))
}

// (gguf_has_magic removed — adoption now uses crate::gguf::is_complete_file, which parses the
// header AND checks tensor coverage; upstream 8c1d64b.)

fn ensure_gguf(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let gguf = dir.join("model.gguf");
    let ok_flag = dir.join(".ok");

    // H4 (and any GGUF-embedded-tokenizer) models pin NO separate tokenizer.json: `tokenizer_cid`
    // is empty and llama.cpp uses the tokenizer baked into the GGUF. Only require/fetch a
    // tokenizer.json when a CID is actually pinned — otherwise ipfs_url("") = "<gateway>/ipfs/"
    // (empty CID) which 400s forever and the model never stages ("no models ready").
    let need_tok = !spec.tokenizer_cid.is_empty();

    // ADOPT a weight file that is already present instead of re-downloading it. Operators commonly
    // pre-fetch the multi-GB GGUF themselves and drop model.gguf into the dir — the miner MUST use
    // that file, never delete + re-download it. Completeness is judged by TENSOR COVERAGE
    // (crate::gguf::is_complete_file: header parses AND the on-disk size covers every tensor —
    // upstream 8c1d64b), which supersedes the old magic-only check and also catches a truncated
    // file hiding behind a stale `.ok` (the "watchdog corrupts mid-download models" class). The
    // FULL content is still verified downstream when the PoM possession index recomputes the tier
    // root R_T, so a wrong drop fails there loudly rather than mining garbage.
    // Explain the on-disk file's status up front so a hand-placed drop that gets rejected says WHY
    // (absent / not a GGUF / truncated) instead of vanishing into a silent "staging/verifying" loop.
    let gguf_reject = crate::gguf::completeness_reason(&gguf);
    let gguf_ready = gguf_reject.is_none();
    if gguf_ready && (!need_tok || tok.exists()) {
        if ok_flag.exists() {
            log::debug!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        } else {
            std::fs::write(&ok_flag, b"").ok();
            log::info!(
                "SlmEngine: adopting pre-staged model '{}' at {} (GGUF complete by tensor \
                 coverage; not re-downloading).",
                spec.name,
                dir.display()
            );
        }
        clear_staging_error();
        return Ok((tok, gguf));
    }

    // Not adoptable — name the exact expected path + the specific reason, at WARN so it surfaces in
    // a normal (non-debug) log. This is the diagnostic an operator needs when a manually-copied
    // model never becomes "ready".
    if let Some(reason) = &gguf_reject {
        log::warn!(
            "SlmEngine: model '{}' NOT ready — expected GGUF at {}: {}. \
             (Path must be exactly <model-dir>/{}/model.gguf.)",
            spec.name,
            gguf.display(),
            reason,
            spec.dir_name
        );
    } else if need_tok && !tok.exists() {
        log::warn!(
            "SlmEngine: model '{}' GGUF is complete at {} but tokenizer.json is missing — will fetch it.",
            spec.name,
            gguf.display()
        );
    }

    std::fs::create_dir_all(&dir)?;
    if ok_flag.exists() && !gguf_ready {
        log::warn!(
            "SlmEngine: '{}' at {} has a stale .ok but the GGUF is incomplete (truncated?) — repairing.",
            spec.name,
            gguf.display()
        );
    }
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before (re)downloading

    if !gguf_ready {
        classic_eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
        // download_file resumes a truncated GGUF, refuses to wipe a partial on Range-less
        // gateways (keeps bytes + backoff-retries), and no-ops a complete one (Range → 416).
        download_file(&ipfs_url(spec.weight_cids[0]), &gguf)?;
        if let Some(reason) = crate::gguf::completeness_reason(&gguf) {
            // The download reported complete but the file still doesn't parse/cover its tensors —
            // corrupt bytes, or a WRONG file a user hand-placed at this path. We can't safely repair
            // it automatically, so tell the operator EXACTLY what to do rather than loop forever.
            let msg = format!(
                "model '{}' file at {} is INVALID after download ({}). If YOU copied a file here, it is \
                 the wrong/corrupt model — DELETE {} and restart so the miner re-downloads the correct one.",
                spec.name,
                gguf.display(),
                reason,
                gguf.display()
            );
            log::error!("[keryx-miner] {msg}");
            set_staging_error(&msg);
            return Err(anyhow!(msg));
        }
    }
    if need_tok && !tok.exists() {
        download_file(&ipfs_url(spec.tokenizer_cid), &tok)?;
    }

    std::fs::write(&ok_flag, b"").with_context(|| format!("write .ok flag {}", ok_flag.display()))?;
    clear_staging_error();
    classic_eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, gguf))
}

// ── Engine loading ───────────────────────────────────────────────────────────

/// Build the list of stop token IDs for a model.
///
/// Tries `token_to_id` for each name first; falls back to the corresponding
/// hardcoded ID so generation always terminates even if the tokenizer exposes
/// special tokens differently (e.g. via `added_tokens` vs the regular vocab).
fn collect_stop_ids(tokenizer: &Tokenizer, names: &[&str], fallbacks: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = names
        .iter()
        .zip(fallbacks.iter())
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
        // Generic fallback (incl. TinyLlama / Zephyr): </s> ends a turn; 0 = padding safety net.
        _ => (collect_stop_ids(tokenizer, &["</s>"], &[2, 0]), vec!["</s>", "<|user|>", "<|system|>", "<|assistant|>"]),
    }
}

fn load_engine(spec: &'static ModelSpec, device: Device) -> Result<SlmEngine> {
    log::info!("SlmEngine: loading '{}'…", spec.name);

    match spec.format {
        ModelFormat::Safetensors => {
            let (tok_path, cfg_path, wt_paths) = ensure_safetensors(spec)?;
            let config: LlamaConfig =
                serde_json::from_str(&std::fs::read_to_string(&cfg_path)?).context("parse config.json")?;
            let config = config.into_config(false);
            let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let wt_refs: Vec<_> = wt_paths.iter().map(|p| p.as_path()).collect();
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&wt_refs, DType::F32, &device) }
                .map_err(|e| anyhow!("mmap weights: {}", e))?;
            let model = Llama::load(vb, &config).map_err(|e| anyhow!("build model: {}", e))?;
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id,
                name: spec.name,
                inner: ModelInner::Full { model, config, cache_dtype: DType::F32 },
                tokenizer,
                device,
                stop_token_ids,
                stop_strings,
            })
        }
        ModelFormat::Gguf => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file =
                std::fs::File::open(&gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file).map_err(|e| anyhow!("read gguf: {}", e))?;
            // PoM zero-dup: load via the single-device split loader so the mining-tier model
            // exposes its quant tensors for in-place sharing with the possession walk. Otherwise
            // a regular single-device load.
            let inner = if pom_force_split() && device.is_cuda() {
                log::info!("SlmEngine: PoM zero-dup — loading '{}' (LLaMA) via single-device split loader", spec.name);
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
                model_id: spec.model_id,
                name: spec.name,
                inner,
                tokenizer,
                device,
                stop_token_ids,
                stop_strings,
            })
        }
        ModelFormat::GgufGemma3 => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file =
                std::fs::File::open(&gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file).map_err(|e| anyhow!("read gguf: {}", e))?;
            // PoM zero-dup: Gemma-3-4B is a NON-split GGUF (baseline tier), so without this
            // the possession walk loads a SECOND VRAM copy → OOM on 8 GB cards. Load via the
            // single-device split fork (exposes quant tensors) so the walk shares this copy.
            // Otherwise use the regular single-device GPU loader.
            let inner = if pom_force_split() && device.is_cuda() {
                log::info!("SlmEngine: PoM zero-dup — loading '{}' (Gemma3) via single-device split loader", spec.name);
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
                model_id: spec.model_id,
                name: spec.name,
                inner,
                tokenizer,
                device,
                stop_token_ids,
                stop_strings,
            })
        }
        ModelFormat::GgufQwen2 => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file =
                std::fs::File::open(&gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file).map_err(|e| anyhow!("read gguf: {}", e))?;
            let model = Qwen2Weights::from_gguf(content, &mut gguf_file, &device)
                .map_err(|e| anyhow!("load qwen2 gguf weights: {}", e))?;
            let inner = ModelInner::QuantizedQwen2(model);
            let (stop_token_ids, stop_strings) = stop_config(&tokenizer, spec.name);
            log::info!("SlmEngine: '{}' ready (stops={:?})", spec.name, stop_token_ids);
            Ok(SlmEngine {
                model_id: spec.model_id,
                name: spec.name,
                inner,
                tokenizer,
                device,
                stop_token_ids,
                stop_strings,
            })
        }
        ModelFormat::GgufQwen3 => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file =
                std::fs::File::open(&gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file).map_err(|e| anyhow!("read gguf: {}", e))?;
            // PoM zero-dup: single-device split loader (exposes quant tensors for the walk),
            // otherwise a regular single-device load.
            let inner = if pom_force_split() && device.is_cuda() {
                log::info!("SlmEngine: PoM zero-dup — loading '{}' (Qwen3) via single-device split loader", spec.name);
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
                model_id: spec.model_id,
                name: spec.name,
                inner,
                tokenizer,
                device,
                stop_token_ids,
                stop_strings,
            })
        }
        // H4 lineup (EXAONE-4 / GLM-4 / Qwen3.6-hybrid-SSM / Kimi-Linear-MoE): candle cannot run these
        // architectures — they are served ONLY by the in-process llama.cpp engine (llama_engine.rs),
        // which is tried BEFORE this candle path. Reaching here means the .so was missing/failed to load,
        // so there is no candle fallback for these archs.
        ModelFormat::GgufExaone4
        | ModelFormat::GgufGlm4
        | ModelFormat::GgufQwen35
        | ModelFormat::GgufKimiLinear
        | ModelFormat::GgufGemma4 => {
            anyhow::bail!(
                "model '{}' ({}) is an H4 arch served only by the in-process llama.cpp engine \
                 (libkeryx-llama.so) — candle has no loader for it. Ensure the .so is present/loads.",
                spec.name,
                spec.dir_name
            )
        }
    }
}

/// Run `load_engine` but catch BOTH a `Result::Err` AND a panic. candle/cudarc can either return an
/// error (clean OOM / file error) or *panic* (CUDA_ERROR_INVALID_PTX from a too-high-arch dequant
/// kernel, a cudarc launch failure, etc.) when loading the quantized model on the GPU. We must not
/// let either crash the miner — instead we capture a structured failure and withdraw the route.
fn try_load_engine(spec: &'static ModelSpec, device: Device) -> std::result::Result<SlmEngine, String> {
    // Test hook (validation only): force the first GPU load to fail so capability withdrawal and
    // retry behavior can be exercised on a card whose GPU inference otherwise works.
    if device.is_cuda() && std::env::var("KERYX_FORCE_GPU_INFER_FAIL").is_ok() {
        static FIRED: AtomicBool = AtomicBool::new(false);
        if !FIRED.swap(true, AtomicOrdering::Relaxed) {
            return Err("KERYX_FORCE_GPU_INFER_FAIL=1 — simulated GPU model-load failure (test hook)".to_string());
        }
    }

    // Do not replace the process-global panic hook here. Model loads can overlap across cards; a
    // take/set/restore sequence races and can permanently install another thread's temporary hook.
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load_engine(spec, device)));
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

/// Last-resort legacy candle path. GPU llama.cpp routes are tried first. CPU is reached only when
/// the operator explicitly opted in (`--enable-cpu-inference`) or forced it (`--cpu-inference`).
fn load_legacy_engine_on(spec: &'static ModelSpec, gpu: usize) -> Result<SlmEngine> {
    if cpu_inference_enabled() {
        return try_load_engine(spec, Device::Cpu)
            .map_err(|reason| anyhow!("CPU inference failed for '{}': {}", spec.name, reason));
    }
    #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
    {
        let device = Device::new_cuda(gpu).map_err(|e| anyhow!("CUDA:{} inference device unavailable: {}", gpu, e))?;
        return match try_load_engine(spec, device) {
            Ok(engine) => Ok(engine),
            Err(reason) if cpu_inference_allowed() => {
                log::warn!(
                    "GPU inference failed for '{}' on CUDA:{} ({}); using explicitly enabled, deprecated CPU emergency fallback",
                    spec.name,
                    gpu,
                    reason
                );
                set_cpu_inference(true);
                try_load_engine(spec, Device::Cpu)
                    .map_err(|cpu| anyhow!("CPU inference fallback also failed for '{}': {}", spec.name, cpu))
            }
            Err(reason) => Err(anyhow!(
                "GPU inference failed for '{}' on CUDA:{} ({}); CPU fallback was not enabled",
                spec.name,
                gpu,
                reason
            )),
        };
    }
    #[cfg(not(all(feature = "pom-cuda", not(feature = "pom-opencl"))))]
    {
        let _ = gpu;
        if cpu_inference_allowed() {
            log::warn!(
                "GPU llama.cpp routes failed for '{}'; using explicitly enabled, deprecated CPU emergency fallback",
                spec.name
            );
            set_cpu_inference(true);
            try_load_engine(spec, Device::Cpu)
                .map_err(|reason| anyhow!("CPU inference failed for '{}': {}", spec.name, reason))
        } else {
            Err(anyhow!("GPU llama.cpp inference unavailable for '{}'; CPU fallback was not enabled", spec.name))
        }
    }
}

// ── Inference ────────────────────────────────────────────────────────────────

fn format_prompt(engine: &SlmEngine, prompt: &str) -> String {
    match engine.name {
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

/// Strip a self-emitted reasoning block that H6 "thinking" models (Qwen3.5, GLM-4, Gemma-4) leak
/// ahead of the answer when the GGUF's built-in chat template leaves thinking on. Each dialect has
/// its own close marker; we cut everything up to + including the LAST one present, then trim. When
/// no close marker is found the text is returned as-is (the model answered directly) — never empty,
/// so a direct answer is preserved. (Adapts upstream Keryx-Labs/keryx-miner d9e09a53 to our fork,
/// which formats via the GGUF template rather than a manual per-model prompt, so we strip the leak
/// post-hoc instead of prefilling a closed block.)
fn strip_think_tags(text: &str) -> String {
    // Ordered by how far into the output the answer begins; use the LAST occurrence of each so a
    // reasoning block that itself quotes the marker doesn't cut the real answer short.
    const CLOSERS: &[&str] = &[
        "</think>",          // Qwen3.x / GLM-4 / DeepSeek ChatML think block
        "<channel|>message", // Gemma-4 <|channel>thought … <channel|>message<answer>
        "<channel|>",        // Gemma-4 fallback (empty/closed thought channel)
        "<|/thought|>",      // GLM channel-thought variant
    ];
    if let Some(answer_start) = CLOSERS.iter().filter_map(|close| text.rfind(close).map(|pos| pos + close.len())).max()
    {
        return text[answer_start..].trim().to_string();
    }
    text.trim().to_string()
}

fn generate(engine: &mut SlmEngine, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let formatted = format_prompt(engine, prompt);
    let enc = engine.tokenizer.encode(formatted.as_str(), true).map_err(|e| anyhow!("encode: {}", e))?;
    let mut all_tokens: Vec<u32> = enc.get_ids().to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut lp = LogitsProcessor::new(42, Some(0.7), Some(0.9));
    let model_max = match engine.name {
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
                let logits = model.forward(&input, pos, &mut cache).map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) {
                    break;
                }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) {
                    break;
                }
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
                let logits = model.forward(&input, pos).map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) {
                    break;
                }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) {
                    break;
                }
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
                let logits = model.forward(&input, pos).map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) {
                    break;
                }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) {
                    break;
                }
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
                let logits = model.forward(&input, pos).map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) {
                    break;
                }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) {
                    break;
                }
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
                let logits = model.forward(&input, pos).map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) {
                    break;
                }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) {
                    break;
                }
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
                let logits = model.forward(&input, pos).map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) {
                    break;
                }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) {
                    break;
                }
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
                let logits = model.forward(&input, pos).map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) {
                    break;
                }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) {
                    break;
                }
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
                let logits = model.forward(&input, pos).map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if engine.stop_token_ids.contains(&next) {
                    break;
                }
                all_tokens.push(next);
                generated.push(next);
                if hit_stop_string(&engine.tokenizer, &generated, &engine.stop_strings) {
                    break;
                }
            }
        }
    }

    let text = engine.tokenizer.decode(&generated, true).map_err(|e| anyhow!("decode: {}", e))?;
    // Truncate at the earliest stop string in case a control marker leaked into
    // the output (tokenizer that renders special tokens as plain text).
    let cut = engine.stop_strings.iter().filter_map(|s| text.find(s)).min().unwrap_or(text.len());
    let answer = text[..cut].trim();
    // Reasoning models emit a <think> block (Qwen3-style ChatML emits an empty pair; others prime
    // an open one) which must not be published. Strip unconditionally — it is a no-op on text with
    // no tags, and it matches what the in-process llama path already does for every model. The old
    // per-name allow-list only covered retired pre-H6 models and never fired for the H6 lineup.
    Ok(strip_think_tags(answer))
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

/// Deprecated emergency fallback state. It is never enabled automatically unless the operator
/// explicitly opted in; GPU inference remains the primary and expected route.
static CPU_INFERENCE: AtomicBool = AtomicBool::new(false);

pub fn cpu_inference_enabled() -> bool {
    CPU_INFERENCE.load(AtomicOrdering::Relaxed)
}

pub fn set_cpu_inference(on: bool) {
    let previous = CPU_INFERENCE.swap(on, AtomicOrdering::Relaxed);
    if previous != on {
        evict_engine();
    }
}

static CPU_INFERENCE_ALLOWED: AtomicBool = AtomicBool::new(false);

pub fn cpu_inference_allowed() -> bool {
    CPU_INFERENCE_ALLOWED.load(AtomicOrdering::Relaxed)
}

pub fn set_cpu_inference_allowed(on: bool) {
    CPU_INFERENCE_ALLOWED.store(on, AtomicOrdering::Relaxed);
}

/// `--no-shared-inference`: force OPoI inference onto THIS process's own walk GPU instead of the
/// globally-biggest card. Set from the CLI (see `inference_gpu_ordinal`).
static NO_SHARED_INFERENCE: AtomicBool = AtomicBool::new(false);

pub fn set_no_shared_inference(v: bool) {
    NO_SHARED_INFERENCE.store(v, std::sync::atomic::Ordering::Relaxed);
}

// ── Multi-GPU OPoI inference router ───────────────────────────────────────────
//
// On a multi-GPU rig, each inference request is routed to the best-capacity FREE card that can
// serve the requested model; concurrent requests run on DIFFERENT cards (same-card requests
// serialize). A request is NEVER migrated once its card is leased ("don't criss-cross"). If every
// eligible card is busy the caller waits up to the request's deadline for one to free, then reports
// busy (not a hard reject). Single-GPU / no-CUDA rigs collapse to the one card — behavior unchanged.

/// Policy used to rank the FREE, eligible cards. Default `Speed`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InferencePolicy {
    /// Highest measured tokens/sec first (measured on the first real generation per card, cached).
    Speed,
    /// Card serving the highest max-tier (largest resident model) first.
    Reward,
    /// Highest free VRAM (nvidia-smi memory.free) first.
    Memory,
    /// Lowest current PoW hashrate first — minimizes the mining opportunity cost of the pause.
    PowMin,
}

impl InferencePolicy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "speed" => Some(Self::Speed),
            "reward" => Some(Self::Reward),
            "memory" | "mem" => Some(Self::Memory),
            "pow-min" | "powmin" | "pow" => Some(Self::PowMin),
            _ => None,
        }
    }
}

static INFERENCE_POLICY: OnceLock<InferencePolicy> = OnceLock::new();

/// Set the inference routing policy (CLI `--inference-policy`). Env `KERYX_INFERENCE_POLICY` and
/// then `Speed` are the fallbacks when this is never called.
pub fn set_inference_policy(p: InferencePolicy) {
    let _ = INFERENCE_POLICY.set(p);
}

/// The active routing policy: CLI (if set) → `KERYX_INFERENCE_POLICY` → `Speed`.
pub fn inference_policy() -> InferencePolicy {
    if let Some(p) = INFERENCE_POLICY.get() {
        return *p;
    }
    std::env::var("KERYX_INFERENCE_POLICY")
        .ok()
        .and_then(|s| InferencePolicy::parse(&s))
        .unwrap_or(InferencePolicy::Speed)
}

/// CUDA ordinals allowed to serve inference (CLI `--inference-cards` / env `KERYX_INFERENCE_CARDS`).
/// Empty ⇒ every walk card. Cards not in the set stay PoW-only.
static INFERENCE_CARDS: OnceLock<Vec<usize>> = OnceLock::new();

pub fn set_inference_cards(cards: Vec<usize>) {
    let _ = INFERENCE_CARDS.set(cards);
}

fn inference_cards_restrict() -> Vec<usize> {
    if let Some(v) = INFERENCE_CARDS.get() {
        return v.clone();
    }
    let Ok(raw) = std::env::var("KERYX_INFERENCE_CARDS") else {
        return Vec::new();
    };
    let tokens: Vec<&str> = raw.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    let parsed: Option<Vec<usize>> = tokens.iter().map(|token| token.parse::<usize>().ok()).collect();
    match parsed {
        Some(cards) if !cards.is_empty() => cards,
        _ => {
            // A malformed explicit safety restriction must never broaden to "all GPUs".
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, AtomicOrdering::Relaxed) {
                log::error!("KERYX_INFERENCE_CARDS is malformed/empty — failing closed with no usable inference card");
            }
            vec![usize::MAX]
        }
    }
}

/// Per-card inference busy flags (keyed by CUDA ordinal). A set flag ⇒ that card is mid-generation.
fn card_busy() -> &'static Mutex<std::collections::HashMap<usize, bool>> {
    static B: OnceLock<Mutex<std::collections::HashMap<usize, bool>>> = OnceLock::new();
    B.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn try_claim_card(gpu: usize) -> bool {
    let mut g = match card_busy().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let slot = g.entry(gpu).or_insert(false);
    if *slot {
        false
    } else {
        *slot = true;
        true
    }
}

fn release_card(gpu: usize) {
    let mut g = match card_busy().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    g.insert(gpu, false);
}

fn card_is_free(gpu: usize) -> bool {
    let g = match card_busy().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    !g.get(&gpu).copied().unwrap_or(false)
}

/// RAII lease of ONE inference card. While held, the card is claimed for a single generation;
/// dropping it (normal return, `?`, or a panic unwind inside `catch_unwind`) releases the card so
/// the next queued request can use it. A request is NEVER migrated to another card once leased.
pub struct InferenceLease {
    gpu: usize,
}
impl InferenceLease {
    pub fn gpu(&self) -> usize {
        self.gpu
    }
}
impl Drop for InferenceLease {
    fn drop(&mut self) {
        release_card(self.gpu);
    }
}

fn bounded_deadline_ms(deadline_ms: u64) -> u64 {
    if deadline_ms == 0 {
        DEFAULT_INFERENCE_DEADLINE_MS
    } else {
        deadline_ms.min(MAX_INFERENCE_DEADLINE_MS)
    }
}

/// Claim one exact card. Used by a self-test which must prove the same `(model, gpu)` route it
/// records, rather than being silently migrated by the policy router.
fn acquire_specific_inference_card(gpu: usize, deadline_ms: u64) -> Option<InferenceLease> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(bounded_deadline_ms(deadline_ms));
    loop {
        if try_claim_card(gpu) {
            return Some(InferenceLease { gpu });
        }
        if std::time::Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
}

/// Measured tokens/sec per card — populated on the first real generation on each card
/// (`record_card_toks`), so the `Speed` policy ranks on a MEASURED number rather than a synthetic
/// probe ("measure, don't proxy"). Until a card has a measurement it ranks by the total-VRAM proxy.
fn card_toks() -> &'static Mutex<std::collections::HashMap<usize, f64>> {
    static T: OnceLock<Mutex<std::collections::HashMap<usize, f64>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn record_card_toks(gpu: usize, toks_per_s: f64) {
    if toks_per_s.is_finite() && toks_per_s > 0.0 {
        if let Ok(mut g) = card_toks().lock() {
            g.insert(gpu, toks_per_s);
        }
    }
}

fn card_toks_get(gpu: usize) -> Option<f64> {
    card_toks().lock().ok().and_then(|g| g.get(&gpu).copied())
}

/// Per-card PoW hashrate feed for the `PowMin` policy (H/s). Zero samples during startup or an
/// inference pause do not replace the last real mining rate; otherwise the card most recently
/// paused for inference would become artificially sticky. Until a card has a positive sample, the
/// policy falls back to the total-VRAM proxy (PoM is bandwidth-bound, so a smaller card is usually
/// cheaper to pause).
static CARD_HASHRATE: OnceLock<Mutex<std::collections::HashMap<usize, f64>>> = OnceLock::new();
fn card_hashrate() -> &'static Mutex<std::collections::HashMap<usize, f64>> {
    CARD_HASHRATE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}
pub fn report_card_hashrate(gpu: usize, h_per_s: f64) {
    if h_per_s.is_finite() && h_per_s > 0.0 {
        if let Ok(mut g) = card_hashrate().lock() {
            g.insert(gpu, h_per_s);
        }
    }
}
fn card_hashrate_get(gpu: usize) -> Option<f64> {
    card_hashrate().lock().ok().and_then(|g| g.get(&gpu).copied())
}

/// `memory.total` / `memory.free` (MiB) keyed by the CUDA driver's *visible logical ordinal*.
/// nvidia-smi row numbers are physical/global and are not the same namespace under
/// `CUDA_VISIBLE_DEVICES`; using the driver keeps routing aligned with `Device::new_cuda(gpu)`.
fn gpu_mem_mib(query: &str) -> std::collections::HashMap<usize, u64> {
    #[cfg(feature = "pom-cuda")]
    {
        return match query {
            "memory.total" => {
                crate::pom_gpu::query_all_gpus_vram().into_iter().map(|(gpu, mib)| (gpu as usize, mib)).collect()
            }
            "memory.free" => crate::pom_gpu::query_all_gpus_free_vram()
                .into_iter()
                .map(|(gpu, free, _)| (gpu as usize, free))
                .collect(),
            _ => std::collections::HashMap::new(),
        };
    }
    #[cfg(not(feature = "pom-cuda"))]
    {
        let mut out = std::collections::HashMap::new();
        let Ok(o) = std::process::Command::new("nvidia-smi")
            .args([&format!("--query-gpu={}", query), "--format=csv,noheader,nounits"])
            .output()
        else {
            return out;
        };
        if !o.status.success() {
            return out;
        }
        for (i, line) in String::from_utf8_lossy(&o.stdout).lines().enumerate() {
            if let Ok(v) = line.trim().parse::<u64>() {
                out.insert(i, v);
            }
        }
        out
    }
}

/// On-disk GGUF size (bytes) of the model assigned to `gpu`, used as the `Reward` tier-magnitude
/// proxy (bigger resident model = higher tier = higher subsidy). 0 when unknown.
#[cfg(feature = "pom-cuda")]
fn card_assigned_model_bytes(gpu: usize) -> u64 {
    let Some((_mid, gguf)) = crate::pom_gpu::device_model(gpu as u32) else { return 0 };
    std::fs::metadata(&gguf).map(|m| m.len()).unwrap_or(0)
}

/// CUDA ordinals eligible to serve `model_id`, restricted by `--inference-cards`. Preference:
///   1. Walk cards whose ASSIGNED tier == `model_id` (zero-dup reuse — the model is already this
///      card's mining tier, so inference needs no extra copy beyond pausing that card's walk).
///   2. If none match by assignment (e.g. a chat request for a tier no card mines), ALL allowed
///      walk cards (the model is a declared tier; inference evicts that card's walk anyway — this
///      is the "fits free VRAM" fallback in practice).
/// Single-GPU / no-CUDA collapses to `[inference_gpu_ordinal()]` so behavior is unchanged.
fn eligible_cards(model_id: &[u8; 32]) -> Vec<usize> {
    #[cfg(feature = "pom-cuda")]
    {
        let restrict = inference_cards_restrict();
        let allowed = |g: usize| restrict.is_empty() || restrict.contains(&g);
        let cards: Vec<usize> =
            crate::pom_gpu::walk_devices().into_iter().map(|d| d as usize).filter(|g| allowed(*g)).collect();
        if cards.is_empty() {
            // Walk not installed yet (inference before the first PoM job), or every walk card
            // filtered out — fall back to the legacy single-card choice (respecting the restrict
            // set if it names a card).
            let d = inference_gpu_for_model(model_id);
            return if allowed(d) { vec![d] } else { cards };
        }
        // Once at least one route has passed the real generation self-test, never send a request to
        // a merely assumed card. A model-wide pass on a large GPU does not prove that a smaller card
        // can load the same model. Additional cards become eligible after their own successful live
        // generation/self-test records the exact pair.
        let proven: Vec<usize> = cards.iter().copied().filter(|g| model_serveable_on(model_id, *g)).collect();
        if !proven.is_empty() {
            return proven;
        }
        let matched: Vec<usize> = cards
            .iter()
            .copied()
            .filter(|g| crate::pom_gpu::device_model(*g as u32).map_or(false, |(mid, _)| &mid == model_id))
            .collect();
        if !matched.is_empty() {
            return matched;
        }
        cards
    }
    #[cfg(not(feature = "pom-cuda"))]
    {
        let _ = model_id;
        vec![inference_gpu_for_model(model_id)]
    }
}

/// Rank `cards` best-first under `policy`. Ties resolve to the lowest ordinal (stable). Called on
/// every router poll iteration, so it shells out to `nvidia-smi` ONLY when the policy (or a missing
/// measurement) actually needs it — the default `Speed` path with cached tok/s never does.
fn rank_cards(mut cards: Vec<usize>, policy: InferencePolicy) -> Vec<usize> {
    // `memory.total` proxy is needed when a Speed/PowMin card lacks a measurement, or for Reward on
    // non-CUDA. `memory.free` is needed only for the Memory policy. Query each at most once, lazily.
    let need_total = match policy {
        InferencePolicy::Memory => false,
        InferencePolicy::Speed => cards.iter().any(|g| card_toks_get(*g).is_none()),
        InferencePolicy::PowMin => cards.iter().any(|g| card_hashrate_get(*g).is_none()),
        InferencePolicy::Reward => cfg!(not(feature = "pom-cuda")),
    };
    let total = if need_total { gpu_mem_mib("memory.total") } else { std::collections::HashMap::new() };
    let free = if matches!(policy, InferencePolicy::Memory) {
        gpu_mem_mib("memory.free")
    } else {
        std::collections::HashMap::new()
    };
    // f64 sort key; higher == preferred. We negate for "lowest first" policies.
    let key = |g: usize| -> f64 {
        match policy {
            InferencePolicy::Speed => {
                card_toks_get(g).unwrap_or_else(|| total.get(&g).copied().unwrap_or(0) as f64 / 1.0e6)
            }
            InferencePolicy::Memory => free.get(&g).copied().unwrap_or(0) as f64,
            #[cfg(feature = "pom-cuda")]
            InferencePolicy::Reward => card_assigned_model_bytes(g) as f64,
            #[cfg(not(feature = "pom-cuda"))]
            InferencePolicy::Reward => total.get(&g).copied().unwrap_or(0) as f64,
            InferencePolicy::PowMin => {
                // Lowest hashrate preferred ⇒ negate. Fallback proxy: lowest total VRAM.
                let h = card_hashrate_get(g).unwrap_or_else(|| total.get(&g).copied().unwrap_or(0) as f64);
                -h
            }
        }
    };
    cards.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    cards
}

/// The card this request WOULD be routed to right now (best-ranked, free, eligible) — non-claiming.
/// `None` when every eligible card is currently busy. Replaces `inference_gpu_ordinal()` for
/// request routing; the latter stays the single-card / no-shared default and the ultimate fallback.
pub fn pick_inference_device(model_id: &[u8; 32]) -> Option<usize> {
    let free: Vec<usize> = eligible_cards(model_id).into_iter().filter(|g| card_is_free(*g)).collect();
    rank_cards(free, inference_policy()).into_iter().next()
}

/// Claim the best FREE eligible card for `model_id`, waiting up to `deadline_ms` for one to free if
/// all are busy. Returns a lease (card claimed, auto-released on drop) or `None` on deadline (caller
/// reports busy — NOT a hard reject). The request is pinned to the leased card for its whole life.
pub fn acquire_inference_card(model_id: &[u8; 32], deadline_ms: u64) -> Option<InferenceLease> {
    let policy = inference_policy();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(bounded_deadline_ms(deadline_ms));
    loop {
        let eligible = eligible_cards(model_id);
        if eligible.is_empty() {
            return None;
        }
        let free: Vec<usize> = eligible.iter().copied().filter(|g| card_is_free(*g)).collect();
        for gpu in rank_cards(free, policy) {
            if try_claim_card(gpu) {
                return Some(InferenceLease { gpu });
            }
        }
        if std::time::Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
}

/// Async router used by network handlers. Unlike `acquire_inference_card`, waiting for a busy card
/// yields the Tokio worker instead of blocking one of the miner's small async-worker pool. Card
/// ranking is resolved once per request; busy state is still checked atomically on every wake.
pub async fn acquire_inference_card_async(model_id: &[u8; 32], deadline_ms: u64) -> Option<InferenceLease> {
    let cards = rank_cards(eligible_cards(model_id), inference_policy());
    if cards.is_empty() {
        return None;
    }
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(bounded_deadline_ms(deadline_ms));
    loop {
        for &gpu in &cards {
            if try_claim_card(gpu) {
                return Some(InferenceLease { gpu });
            }
        }
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return None;
        }
        tokio::time::sleep((deadline - now).min(std::time::Duration::from_millis(25))).await;
    }
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
/// If the chosen GPU cannot serve inference, the route is withdrawn rather than moved to CPU.
/// Self-test failover is model-specific. A process-global ordinal made a successful failover for a
/// small model silently reroute every larger model to the same (often incapable) card.
fn inference_gpu_overrides() -> &'static RwLock<std::collections::HashMap<[u8; 32], usize>> {
    static OVERRIDES: OnceLock<RwLock<std::collections::HashMap<[u8; 32], usize>>> = OnceLock::new();
    OVERRIDES.get_or_init(|| RwLock::new(std::collections::HashMap::new()))
}

/// Move a model's serving route only while the destination proof is still present. Keep the proof
/// read lock through override publication: exact-route invalidation takes the same locks in this
/// order, so it either removes an older override first or waits and removes the newly published one.
/// This prevents a child exit between generation and failover bookkeeping from leaving a stale
/// model-specific override behind.
fn set_inference_override_if_proven(model_id: &[u8; 32], gpu: usize) -> bool {
    let state = match self_test_state().read() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    if state.get(&(*model_id, gpu)).copied() != Some(true) || model_is_unavailable(model_id) {
        return false;
    }
    let mut overrides = match inference_gpu_overrides().write() {
        Ok(overrides) => overrides,
        Err(poisoned) => poisoned.into_inner(),
    };
    overrides.insert(*model_id, gpu);
    true
}

fn restrict_inference_gpu(candidate: usize) -> usize {
    let restrict = inference_cards_restrict();
    if restrict.is_empty() || restrict.contains(&candidate) {
        candidate
    } else {
        // Configuration is already initialized before warmers. Prefer the first explicitly allowed
        // ordinal over ever probing/loading on a forbidden device.
        restrict[0]
    }
}

fn inference_card_allowed(gpu: usize) -> bool {
    let restrict = inference_cards_restrict();
    restrict.is_empty() || restrict.contains(&gpu)
}

pub fn inference_gpu_ordinal() -> usize {
    if let Ok(s) = std::env::var("KERYX_INFERENCE_GPU") {
        if let Ok(n) = s.trim().parse::<usize>() {
            return restrict_inference_gpu(n);
        }
    }
    // OpenCL/Vulkan owns one process-wide inference engine and records it as one logical route;
    // physical placement is tracked separately by pom_opencl's full-PCI mapping. Never let an
    // unrelated NVIDIA card discovered via nvidia-smi change this logical key on a mixed host.
    #[cfg(feature = "pom-opencl")]
    {
        return restrict_inference_gpu(0);
    }

    #[cfg(not(feature = "pom-opencl"))]
    {
        // `pom_gpu` (the CUDA walk driver) only exists on the pom-cuda build. On non-CUDA builds
        // (default, and AMD/pom-opencl which places inference via llama_vulkan/KERYX_LLAMA_VK_DEVICE)
        // there are no CUDA walk devices, so fall back to an empty set → ordinal 0 (never used at
        // runtime there: llama_vulkan takes over). Fixes the v0.6.5.3 non-CUDA
        // build break (slm.rs referenced crate::pom_gpu unconditionally).
        #[cfg(any(feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
        let walk = crate::pom_gpu::walk_devices();
        #[cfg(not(any(feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal"))))]
        let walk: Vec<u32> = Vec::new();
        if NO_SHARED_INFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
            return restrict_inference_gpu(walk.first().copied().map(|d| d as usize).unwrap_or(0));
        }
        let selected = match walk.len() {
            1 => walk[0] as usize,
            n if n > 1 => biggest_cuda_gpu_within(&walk.iter().map(|d| *d as usize).collect::<Vec<_>>())
                .unwrap_or(walk[0] as usize),
            // Walk not installed yet. This is the STARTUP case, and picking the globally-biggest card
            // here used to deadlock a process that does not own it: with `--cuda-device 0` on a rig whose
            // biggest card is 2, every staging attempt on 0 saw inference_gpu=2, the serveability gate
            // refused to mine a tier nothing could serve, so no miner ever installed — which is what
            // keeps `walk` empty. The card would sit at 0 h/s, "preparing", logging once a second forever.
            // Our own --cuda-device set is knowable without the walk: it is in this process's argv.
            _ => {
                let own = cli_cuda_devices();
                match own.len() {
                    0 => biggest_cuda_gpu_within(&[]).unwrap_or(0),
                    1 => own[0],
                    _ => biggest_cuda_gpu_within(&own).unwrap_or(own[0]),
                }
            }
        };
        restrict_inference_gpu(selected)
    }
}

/// Resolve the serving card for one model, including only that model's proven failover override.
pub fn inference_gpu_for_model(model_id: &[u8; 32]) -> usize {
    if std::env::var("KERYX_INFERENCE_GPU").is_err() {
        if let Ok(overrides) = inference_gpu_overrides().read() {
            if let Some(&gpu) = overrides.get(model_id) {
                return restrict_inference_gpu(gpu);
            }
        }

        // During mixed-rig startup the walk map is still empty, but per-device mining models are
        // already known. Prefer a card actually assigned this exact model instead of sending every
        // tier's warmer to the globally-largest GPU. Apart from wasting VRAM, the old choice could
        // prove a small tier only on the large card while leaving the small card's real route
        // untested. Explicit per-process isolation keeps its historical own-card behavior.
        #[cfg(feature = "pom-cuda")]
        if !NO_SHARED_INFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
            let mut candidates: Vec<usize> =
                crate::pom_gpu::walk_devices().into_iter().map(|gpu| gpu as usize).collect();
            if candidates.is_empty() {
                candidates = cli_cuda_devices();
            }
            if candidates.is_empty() {
                candidates = crate::pom_gpu::query_all_gpus_vram().into_iter().map(|(gpu, _)| gpu as usize).collect();
            }
            candidates.retain(|gpu| {
                inference_card_allowed(*gpu)
                    && crate::pom_gpu::device_model(*gpu as u32)
                        .map(|(assigned, _)| assigned == *model_id)
                        .unwrap_or(false)
            });
            if !candidates.is_empty() {
                candidates.sort_unstable();
                return biggest_cuda_gpu_within(&candidates).unwrap_or(candidates[0]);
            }
        }
    }
    inference_gpu_ordinal()
}

/// The CUDA ordinals this process was told to mine on, read straight from its own argv
/// (`--cuda-device 0,1` or `--cuda-device=0,1`). Empty means "all" (the flag's default) or a
/// non-CUDA invocation. Argv is used rather than plumbing the value in from the CUDA plugin because
/// that plugin cannot depend on this crate (it would be a Cargo cycle).
pub fn cli_cuda_devices() -> Vec<usize> {
    parse_cuda_devices(std::env::args())
}

fn parse_cuda_devices<I: IntoIterator<Item = String>>(args: I) -> Vec<usize> {
    let args: Vec<String> = args.into_iter().collect();
    let raw = args.iter().enumerate().find_map(|(i, a)| {
        if let Some(v) = a.strip_prefix("--cuda-device=") {
            return Some(v.to_string());
        }
        (a == "--cuda-device").then(|| args.get(i + 1).cloned()).flatten()
    });
    raw.map(|v| v.split(',').filter_map(|s| s.trim().parse::<usize>().ok()).collect()).unwrap_or_default()
}

/// The largest-VRAM CUDA ordinal, restricted to `pool` (empty = consider every visible GPU). `None`
/// if VRAM cannot be read → caller defaults to the pool's first card. Ties resolve to the lowest
/// ordinal. Logged ONCE per chosen ordinal: this is called from the staging retry loop, and logging
/// unconditionally produced a line every second for as long as staging kept failing.
fn biggest_cuda_gpu_within(pool: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, u64)> = None;
    for (i, mib) in gpu_mem_mib("memory.total") {
        if !pool.is_empty() && !pool.contains(&i) {
            continue;
        }
        if best.map_or(true, |(_, m)| mib > m) {
            best = Some((i, mib));
        }
    }
    let (ord, _) = best?;
    if ord != 0 {
        static LOGGED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<usize>>> =
            std::sync::OnceLock::new();
        let seen = LOGGED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        if seen.lock().map(|mut s| s.insert(ord)).unwrap_or(false) {
            log::info!(
                "OPoI inference will run on CUDA:{} (largest-VRAM GPU of {}); other GPUs mine PoM only.",
                ord,
                if pool.is_empty() { "this host".to_string() } else { format!("{:?}", pool) }
            );
        }
    }
    Some(ord)
}

/// Candle device used only by the CUDA readiness probe and dormant legacy models. Active H6
/// serving uses llama.cpp; CPU remains an explicit, deprecated final emergency fallback only.
fn inference_device() -> candle_core::Result<Device> {
    // Apple Silicon (Phase 3d — candle-Metal is out of the build entirely): the primary GPU
    // inference path is the in-process llama.cpp Metal engine (`libkeryx-llama.dylib`, wired
    // through `crate::llama_engine` in `load_and_run_inference`). This `candle` device is only
    // intentionally disabled; `Device::new_metal` no longer exists because candle-core is built
    // without the `metal` feature on this platform.
    #[cfg(all(target_os = "macos", feature = "pom-metal"))]
    {
        return Err(candle_core::Error::Msg("candle inference is disabled on Metal; use the llama.cpp engine".into()));
    }
    #[cfg(feature = "pom-opencl")]
    {
        return Err(candle_core::Error::Msg(
            "candle inference is disabled on OpenCL; use the Vulkan llama.cpp engine".into(),
        ));
    }
    #[cfg(all(not(feature = "pom-opencl"), not(all(target_os = "macos", feature = "pom-metal"))))]
    {
        Device::new_cuda(inference_gpu_ordinal())
    }
}

/// Register the set of models this miner currently serves (drives `ai:cap`).
pub fn init_supported(specs: &'static [&'static ModelSpec]) {
    *SUPPORTED_SPECS.write().unwrap() = specs;
    if let Some(spec) = specs.first() {
        publish_runtime_model_status(&spec.model_id);
    }
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
        log::info!("OPoI v2 already active (DAA {} ≥ H) — serving the uncensored lineup ({} model(s)).", daa, v2.len());
    }
    *SUPPORTED_SPECS.write().unwrap() = v2;
    evict_engine();
    if let Some(spec) = v2.first() {
        publish_runtime_model_status(&spec.model_id);
    }
}

/// Outcome of the startup GPU inference probe.
pub enum GpuProbe {
    /// A GPU matmul succeeded — cuBLAS is loaded and full-speed inference is available.
    Ok,
    /// No CUDA device present — H6 inference is unavailable.
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
    // Keep the process-global panic hook untouched. Probe/model loads may overlap with unrelated
    // worker panics, and swapping the global hook is not thread-safe as a scoped operation.
    let probe = std::panic::catch_unwind(|| {
        let device = inference_device()?;
        let a = Tensor::new(&[[1f32, 2.0], [3.0, 4.0]], &device)?;
        let b = Tensor::new(&[[5f32, 6.0], [7.0, 8.0]], &device)?;
        a.matmul(&b)?.to_vec2::<f32>()?;
        anyhow::Ok(())
    });
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
            ModelFormat::Gguf
            | ModelFormat::GgufQwen2
            | ModelFormat::GgufQwen3
            | ModelFormat::GgufGemma3
            | ModelFormat::GgufExaone4
            | ModelFormat::GgufGlm4
            | ModelFormat::GgufQwen35
            | ModelFormat::GgufKimiLinear
            | ModelFormat::GgufGemma4 => ensure_gguf(spec).map(|_| ()),
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

// ── OPoI serveability self-test ──────────────────────────────────────────────
// The network's whole point is serving inference. Mining a tier we CANNOT serve exposes the pool to
// inference strikes, and declaring one we cannot serve is dishonest. So a model must PROVE it can
// generate on THIS rig before we announce or mine its tier: we run one tiny real generation and gate
// declaration + mining on it. The result is cached (pass or fail) so the probe runs once per model
// bring-up, not every mining cycle. A transient GPU fault reset clears the entry so it re-probes.
const SELF_TEST_MAX_TOKENS: usize = 8;

/// A route may only recover from a withdrawal when this probe produced a fresh, non-empty
/// response.  A cached success belongs to an older engine generation and must never turn a
/// failed/empty probe back into a capability declaration.
fn current_probe_passed(out: Option<&str>) -> bool {
    out.is_some_and(|text| !text.trim().is_empty())
}

type SelfTestKey = ([u8; 32], usize);

fn self_test_state() -> &'static RwLock<std::collections::HashMap<SelfTestKey, bool>> {
    static S: OnceLock<RwLock<std::collections::HashMap<SelfTestKey, bool>>> = OnceLock::new();
    S.get_or_init(|| RwLock::new(std::collections::HashMap::new()))
}

fn self_test_failures() -> &'static RwLock<std::collections::HashMap<SelfTestKey, std::time::Instant>> {
    static FAILURES: OnceLock<RwLock<std::collections::HashMap<SelfTestKey, std::time::Instant>>> = OnceLock::new();
    FAILURES.get_or_init(|| RwLock::new(std::collections::HashMap::new()))
}

const SELF_TEST_FAILURE_RETRY: std::time::Duration = std::time::Duration::from_secs(30);

/// True once `model_id` PASSED its inference self-test on this rig and has not since been withdrawn.
/// This is the gate for both declaring the model to the pool and mining its tier.
fn any_proven_route(model_id: &[u8; 32]) -> bool {
    self_test_state()
        .read()
        .map(|m| m.iter().any(|((mid, gpu), passed)| mid == model_id && *passed && inference_card_allowed(*gpu)))
        .unwrap_or(false)
}

pub fn model_serveable(model_id: &[u8; 32]) -> bool {
    any_proven_route(model_id) && !model_is_unavailable(model_id)
}

/// Whether this exact card has proven it can load and generate the requested model.
pub fn model_serveable_on(model_id: &[u8; 32], gpu: usize) -> bool {
    if !inference_card_allowed(gpu) {
        return false;
    }
    self_test_state().read().map(|m| m.get(&(*model_id, gpu)).copied() == Some(true)).unwrap_or(false)
        && !model_is_unavailable(model_id)
}

/// Whether a self-test has already been attempted (pass or fail) for `model_id`.
pub fn self_test_attempted(model_id: &[u8; 32]) -> bool {
    self_test_state().read().map(|m| m.keys().any(|(mid, _)| mid == model_id)).unwrap_or(false)
}

/// Record that `model_id` is proven serveable (self-test passed, OR a real OPoI generation just
/// succeeded — a live answer is the strongest possible proof). Idempotent.
pub fn record_serveable(model_id: &[u8; 32]) {
    record_serveable_on(model_id, inference_gpu_for_model(model_id));
}

pub fn record_serveable_on(model_id: &[u8; 32], gpu: usize) {
    if !inference_card_allowed(gpu) {
        log::warn!("OPoI: refusing to record model {:.8} on disallowed GPU {}", hex::encode(model_id), gpu);
        return;
    }
    if let Ok(mut m) = self_test_state().write() {
        m.insert((*model_id, gpu), true);
    }
    if let Ok(mut failures) = self_test_failures().write() {
        failures.remove(&(*model_id, gpu));
    }
    publish_runtime_model_status(model_id);
}

fn record_unserveable_on(model_id: &[u8; 32], gpu: usize) {
    if let Ok(mut m) = self_test_state().write() {
        m.insert((*model_id, gpu), false);
    }
    if let Ok(mut failures) = self_test_failures().write() {
        failures.insert((*model_id, gpu), std::time::Instant::now());
    }
    publish_runtime_model_status(model_id);
}

/// The honest declare set: staged models that have PROVEN they can serve inference on this rig.
/// `declare_capabilities` announces THIS (not raw `loaded_model_ids`), so the pool never routes us a
/// request for a tier we cannot answer — and we never mine a tier we cannot serve (the walk install
/// gates on the same `model_serveable`).
///
/// Every backend uses the same proof gate. A staged file proves only bytes on disk; it says nothing
/// about whether CUDA, Vulkan, or Metal can load and generate on the selected GPU.
fn proven_serveable_model_ids() -> Vec<[u8; 32]> {
    loaded_model_ids().into_iter().filter(|m| model_serveable(m)).collect()
}

/// Exact GPU ordinals whose live proofs currently make `model_id` serveable. Keep this separate
/// from external-request accounting: startup self-tests deliberately do not increment inference
/// request counters, but the dashboard still needs to identify the route they proved.
fn proven_route_gpus(model_id: &[u8; 32]) -> Vec<usize> {
    if model_is_unavailable(model_id) {
        return Vec::new();
    }
    let mut routes = self_test_state()
        .read()
        .map(|state| {
            state
                .iter()
                .filter_map(|((candidate, gpu), passed)| {
                    (candidate == model_id && *passed && inference_card_allowed(*gpu)).then_some(*gpu)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    routes.sort_unstable();
    routes.dedup();
    routes
}

/// Keep the dashboard's displayed model and route as one coherent fact. A route loss for one
/// model can be published while another model remains serveable, so aggregate readiness alone
/// must never leave the just-failed model paired with an invented generic "GPU route" label.
fn select_runtime_status_route(
    requested_model: [u8; 32],
    requested_routes: Vec<usize>,
    proven_candidates: &[([u8; 32], Vec<usize>)],
) -> ([u8; 32], Vec<usize>) {
    if !requested_routes.is_empty() {
        return (requested_model, requested_routes);
    }
    proven_candidates
        .iter()
        .find(|(_, routes)| !routes.is_empty())
        .cloned()
        .unwrap_or((requested_model, requested_routes))
}

/// Publish only public model metadata and aggregate readiness. This never exposes prompts,
/// response text, CIDs, paths, device identities, or a peer-provided failure reason.
fn publish_runtime_model_status(model_id: &[u8; 32]) {
    if !crate::models::REGISTRY.iter().any(|spec| &spec.model_id == model_id) {
        return;
    }
    let serveable_model_ids = proven_serveable_model_ids();
    let proven_candidates = serveable_model_ids
        .iter()
        .copied()
        .filter(|candidate| crate::models::REGISTRY.iter().any(|spec| spec.model_id == *candidate))
        .map(|candidate| (candidate, proven_route_gpus(&candidate)))
        .collect::<Vec<_>>();
    let (display_model_id, display_routes) =
        select_runtime_status_route(*model_id, proven_route_gpus(model_id), &proven_candidates);
    let Some(spec) = crate::models::REGISTRY.iter().copied().find(|spec| spec.model_id == display_model_id) else {
        return;
    };
    let tier = match crate::models::REGISTRY.iter().position(|candidate| candidate.model_id == display_model_id) {
        Some(0) => "very-light",
        Some(1) => "light",
        Some(2) => "default",
        Some(3) => "high",
        Some(4) => "very-high",
        _ => "custom",
    };
    #[cfg(feature = "pom-cuda")]
    let backend = "CUDA";
    #[cfg(all(not(feature = "pom-cuda"), feature = "pom-opencl"))]
    let backend = "Vulkan";
    #[cfg(all(not(feature = "pom-cuda"), not(feature = "pom-opencl"), feature = "pom-metal"))]
    let backend = "Metal";
    #[cfg(all(not(feature = "pom-cuda"), not(feature = "pom-opencl"), not(feature = "pom-metal")))]
    let backend = "GPU";
    crate::runtime_stats::set_inference_model_status(
        spec.name,
        &display_model_id,
        tier,
        backend,
        &display_routes,
        serveable_model_ids.len(),
        last_staging_error().is_some(),
    );
}

/// Whether this process has at least one staged model with a live GPU-route proof, independent of
/// the optional `--wait-ready` declaration latch. Job handlers use this for the core
/// no-inference/no-PoW gate: while the latch is closed they must still feed templates to workers so
/// each worker can install its walk, after which the worker-side gate keeps actual hashing idle.
/// Using `serveable_model_ids()` there creates a startup deadlock because that public declaration
/// set is intentionally empty until every walk reports ready.
pub fn has_proven_serveable_model() -> bool {
    !proven_serveable_model_ids().is_empty()
}

pub fn serveable_model_ids() -> Vec<[u8; 32]> {
    // --wait-ready: declare NOTHING until every card is set up — an early declaration is what
    // invites OPoI challenges into the fragile bring-up window on low-RAM rigs. The pool sees
    // no capabilities, so it has nothing to challenge; the full set is declared the moment the
    // gate opens (declare_capabilities_if_changed re-sends on change and every 90 s anyway).
    if crate::wait_ready::holds() {
        return Vec::new();
    }
    proven_serveable_model_ids()
}

/// Forget a model's self-test result so it re-probes (used after a transient GPU-fault reset that
/// rebuilds the card's engine — the prior pass no longer necessarily holds).
pub fn clear_self_test(model_id: &[u8; 32]) {
    if let Ok(mut m) = self_test_state().write() {
        m.retain(|(mid, _), _| mid != model_id);
    }
    if let Ok(mut failures) = self_test_failures().write() {
        failures.retain(|(mid, _), _| mid != model_id);
    }
    publish_runtime_model_status(model_id);
}

/// The AMD/Vulkan engine is a process-wide singleton. If it is evicted so the OpenCL walk can own
/// VRAM, every cached card-keyed proof for that model belongs to the destroyed engine generation.
/// Withdraw model-wide; the durable warmer is the only path that may restore capability after a
/// fresh non-empty GPU generation.
pub fn withdraw_model_after_amd_engine_eviction(model_id: &[u8; 32]) {
    clear_self_test(model_id);
    mark_model_unavailable(model_id, "amd_inference_evicted_for_mining_vram");
}

/// Resolve the model identity while publishing a subprocess route. Capturing it in the server
/// identity avoids depending on a later active-lineup swap when the child eventually exits.
pub fn model_id_for_gguf(path: &str) -> Option<[u8; 32]> {
    let wanted = std::fs::canonicalize(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
    let specs = *SUPPORTED_SPECS.read().ok()?;
    if let Some(spec) = specs.iter().find(|spec| {
        let candidate = gguf_path_for(spec);
        std::fs::canonicalize(&candidate).unwrap_or(candidate) == wanted
    }) {
        return Some(spec.model_id);
    }
    #[cfg(feature = "pom-cuda")]
    for (model_id, candidate) in crate::pom_gpu::assigned_models() {
        let candidate = std::fs::canonicalize(&candidate).unwrap_or_else(|_| std::path::PathBuf::from(candidate));
        if candidate == wanted {
            return Some(model_id);
        }
    }
    None
}

/// Invalidate one exact `(model, GPU)` serving proof after that backend fails. Clean planned
/// server swaps deliberately do not call this: cached ability proofs for other warmed models on
/// the same card remain valid and must not oscillate as a singleton server changes residency.
pub fn invalidate_inference_route(model_id: &[u8; 32], gpu: usize, reason: &str) {
    let removed = self_test_state().write().map(|mut state| state.remove(&(*model_id, gpu))).unwrap_or(None).is_some();
    if let Ok(mut failures) = self_test_failures().write() {
        failures.remove(&(*model_id, gpu));
    }
    if let Ok(mut overrides) = inference_gpu_overrides().write() {
        if overrides.get(model_id).copied() == Some(gpu) {
            overrides.remove(model_id);
        }
    }
    if any_proven_route(model_id) {
        mark_model_available(model_id, "alternate_inference_route_still_proven");
    } else {
        mark_model_unavailable(model_id, reason);
    }
    if removed {
        log::warn!(
            "OPoI: invalidated model {:.8} route proof on GPU {} ({}); durable warmer will re-probe",
            hex::encode(model_id),
            gpu,
            reason
        );
    }
}

/// Invalidate every inference proof tied to one GPU after its runtime/context is torn down.
/// Proofs are `(model, GPU)` facts: retaining one after a sticky CUDA fault would let the router
/// advertise and select a route whose engine no longer exists. Other cards' proofs remain valid,
/// so a multi-GPU rig withdraws a model only when this was its final proven route.
pub fn invalidate_inference_routes_on_gpu(gpu: usize, reason: &str) {
    let mut affected = std::collections::HashSet::<[u8; 32]>::new();
    if let Ok(mut state) = self_test_state().write() {
        state.retain(|(model_id, route_gpu), _| {
            if *route_gpu == gpu {
                affected.insert(*model_id);
                false
            } else {
                true
            }
        });
    }
    if let Ok(mut failures) = self_test_failures().write() {
        failures.retain(|(model_id, route_gpu), _| {
            if *route_gpu == gpu {
                affected.insert(*model_id);
                false
            } else {
                true
            }
        });
    }
    if let Ok(mut overrides) = inference_gpu_overrides().write() {
        overrides.retain(|model_id, route_gpu| {
            if *route_gpu == gpu {
                affected.insert(*model_id);
                false
            } else {
                true
            }
        });
    }

    for model_id in &affected {
        if any_proven_route(model_id) {
            mark_model_available(model_id, "alternate_inference_route_still_proven");
        } else {
            mark_model_unavailable(model_id, reason);
        }
    }
    if !affected.is_empty() {
        log::warn!(
            "OPoI: invalidated {} model route proof(s) on GPU {} ({}); durable warmers will re-probe",
            affected.len(),
            gpu,
            reason
        );
    }
}

/// Run ONE tiny inference on `gpu` to prove `model_id` can actually serve OPoI before we declare or
/// mine its tier. Caches the outcome. PASS → `mark_model_available` (declarable + mineable); FAIL →
/// `mark_model_unavailable` (withdrawn from ai:cap; the mining gate then refuses to grind a tier we
/// cannot serve). Budgeted by nature: a warm short generation is ~1-2 s; a model that needs far
/// longer would miss the on-chain service window anyway, so slow/empty == not serveable.
pub fn run_inference_self_test(model_id: &[u8; 32], gpu: usize) -> bool {
    if !inference_card_allowed(gpu) {
        log::warn!("OPoI self-test: GPU {} is excluded by --inference-cards; probe refused", gpu);
        return false;
    }
    if let Ok(m) = self_test_state().read() {
        if let Some(&passed) = m.get(&(*model_id, gpu)) {
            if passed && !model_is_unavailable(model_id) {
                return true;
            }
            if !passed {
                let still_cooling_down = self_test_failures()
                    .read()
                    .ok()
                    .and_then(|failures| failures.get(&(*model_id, gpu)).copied())
                    .map(|at| at.elapsed() < SELF_TEST_FAILURE_RETRY)
                    .unwrap_or(false);
                if still_cooling_down {
                    return false;
                }
            }
            // A later global withdrawal invalidated the cached success. Re-run a real generation
            // so recovery is earned, rather than leaving a once-good route wedged forever.
        }
    }
    let name = {
        let specs = *SUPPORTED_SPECS.read().unwrap();
        specs.iter().find(|s| &s.model_id == model_id).map(|s| s.name).unwrap_or("?")
    };
    if name == "?" {
        // The model_id is not (yet) in SUPPORTED_SPECS — a startup ordering race: per-card staging
        // can reach the self-test before init_supported() has registered the full lineup. The old
        // path "probed" it (the spec lookup inside inference instantly returns nothing), FAILED it
        // in 0.0s, CACHED the failure and withdrew the tier permanently — a mixed rig's smaller
        // cards never mined again this process. Not-registered is NOT not-serveable: report
        // not-ready WITHOUT caching or withdrawing, so the next staging cycle re-probes properly.
        log::warn!(
            "OPoI self-test: model id not registered in the supported lineup yet (startup ordering) — \
             deferring the probe, will retry next staging cycle."
        );
        return false;
    }
    log::info!(
        "OPoI self-test: probing '{}' on GPU {} — a model must prove it can generate before we \
         declare/mine its tier (the network exists to serve inference).",
        name,
        gpu
    );
    let t0 = std::time::Instant::now();
    // Self-tests participate in the same per-card lease as live serving. Without this, parallel GPU
    // bring-up can run two model swaps on one card and clear the non-refcounted pause bit too early.
    let Some(_lease) = acquire_specific_inference_card(gpu, DEFAULT_INFERENCE_DEADLINE_MS) else {
        // Busy is not a capability verdict. Do not cache/withdraw a route merely because another
        // legitimate inference held the card during startup; the next staging cycle will retry.
        log::warn!("OPoI self-test: GPU {} remained busy for the probe deadline — deferring", gpu);
        return false;
    };
    let out = {
        // A previous waiter may have completed the same probe while we waited for the card.
        if let Ok(m) = self_test_state().read() {
            if !model_is_unavailable(model_id) {
                if let Some(&passed) = m.get(&(*model_id, gpu)) {
                    if passed {
                        return true;
                    }
                }
            }
        }
        load_and_run_inference_on(gpu, model_id, "Reply with exactly: OK", SELF_TEST_MAX_TOKENS)
    };
    drop(_lease);
    let secs = t0.elapsed().as_secs_f64();
    // Every successful backend publishes its own route proof before returning. Do not publish a
    // second time here: a subprocess can exit and have its monitor invalidate the proof between the
    // backend return and this coordinator, and a late unconditional write would resurrect a dead
    // route permanently. A non-empty response counts only while the backend-owned proof survives.
    let initial_passed = current_probe_passed(out.as_deref()) && model_serveable_on(model_id, gpu);
    let mut passed = initial_passed;
    // ── INFERENCE-HOST FAILOVER ──
    // A failed probe on the DESIGNATED host must not condemn the model (and with it the whole
    // rig: withdrawal → every card demotes/halts → zero declared models → mining suspended).
    // Before withdrawing, retry the probe on this process's OTHER walk GPUs; the first card that
    // proves it can generate becomes the inference host (recorded in INFERENCE_GPU_OVERRIDE so
    // serving follows). This also self-heals a mispicked designated ordinal (the nvidia-smi
    // PCI-order vs CUDA-ordinal mismatch under CUDA_VISIBLE_DEVICES). Skipped when the operator
    // pinned a host (KERYX_INFERENCE_GPU / --no-shared-inference: per-card processes have no
    // alternate card anyway) and on CPU-inference builds.
    if !passed
        && std::env::var("KERYX_INFERENCE_GPU").is_err()
        && !NO_SHARED_INFERENCE.load(std::sync::atomic::Ordering::Relaxed)
        && !cpu_inference_enabled()
    {
        #[cfg(any(feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
        let mut candidates: Vec<usize> = crate::pom_gpu::walk_devices().into_iter().map(|d| d as usize).collect();
        #[cfg(not(any(feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal"))))]
        let mut candidates: Vec<usize> = Vec::new();
        #[cfg(any(feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
        if candidates.is_empty() {
            candidates = cli_cuda_devices();
        }
        #[cfg(feature = "pom-cuda")]
        if candidates.is_empty() {
            candidates = crate::pom_gpu::query_all_gpus_vram().into_iter().map(|(d, _)| d as usize).collect();
        }
        candidates.retain(|&d| d != gpu && inference_card_allowed(d));
        for alt in candidates {
            log::warn!(
                "OPoI self-test: '{}' failed on GPU {} — FAILING OVER: retrying the probe on GPU {}                  (a healthy alternate host keeps the rig mining).", name, gpu, alt
            );
            let t1 = std::time::Instant::now();
            let out2 = acquire_specific_inference_card(alt, DEFAULT_INFERENCE_DEADLINE_MS).and_then(|_lease| {
                load_and_run_inference_on(alt, model_id, "Reply with exactly: OK", SELF_TEST_MAX_TOKENS)
            });
            if matches!(&out2, Some(t) if !t.trim().is_empty()) && set_inference_override_if_proven(model_id, alt) {
                log::warn!(
                    "OPoI self-test: '{}' PASSED on GPU {} in {:.1}s — inference host MOVED {} → {}                      (GPU {} failed its probe; it keeps mining PoM, GPU {} now serves).",
                    name, alt, t1.elapsed().as_secs_f64(), gpu, alt, gpu, alt
                );
                passed = true;
                break;
            }
            log::warn!("OPoI self-test: '{}' also failed on GPU {} ({:.1}s).", name, alt, t1.elapsed().as_secs_f64());
        }
    }
    if !initial_passed {
        record_unserveable_on(model_id, gpu);
    }
    if passed {
        log::info!("OPoI self-test: '{}' PASSED in {:.1}s — serveable; its tier will be declared + mined.", name, secs);
    } else if model_serveable(model_id) {
        // Another card may have proved this model while this card was waiting/probing. Keep the
        // model-level capability alive; only this exact route remains marked false.
        log::warn!(
            "OPoI self-test: '{}' FAILED on GPU {} after {:.1}s, but another GPU still has a proven route; \
             keeping the model declared and excluding GPU {} from serving it.",
            name,
            gpu,
            secs,
            gpu
        );
    } else {
        mark_model_unavailable(model_id, "inference_self_test_failed");
        log::warn!(
            "OPoI self-test: '{}' FAILED after {:.1}s (no/empty output) — NOT serveable. Withdrawing it \
             from ai:cap and NOT mining this tier (mining a tier we cannot serve would strike the pool). \
             Cause is above: GPU inference could not load/run — usually the card lacks VRAM for this \
             model, an old driver, or a missing CUDA runtime / keryx-llama engine next to the miner.",
            name,
            secs
        );
    }
    passed
}

/// Durable startup proof coordinator. Capability/job gates intentionally remain closed until a
/// real GPU generation succeeds, so there may be no mining callback available to retry a transient
/// driver/server/card-busy failure. A lightweight warmer owns that retry responsibility with a
/// capped backoff; cached success is cheap, while cached failure's own cooldown prevents load
/// thrash. The coordinator deliberately remains alive after success: a later GPU-context reset
/// invalidates the exact route proof and this same thread then earns it again without a restart.
pub fn warm_inference_route(model_id: [u8; 32], initial_gpu: usize) {
    let mut delay = std::time::Duration::from_secs(30);
    let mut target_gpu = initial_gpu;
    loop {
        // A failed probe may establish a model-specific failover override. Resolve placement every
        // round so the durable warmer follows that proven card instead of repeatedly retrying the
        // failed original and evicting/reloading both GPUs forever.
        let passed = run_inference_self_test(&model_id, target_gpu);
        if passed {
            delay = std::time::Duration::from_secs(30);
        } else {
            log::warn!(
                "OPoI startup proof for model {:.8} on GPU {} failed/deferred; retrying in {}s",
                hex::encode(model_id),
                target_gpu,
                delay.as_secs()
            );
        }
        std::thread::sleep(delay);
        if !passed {
            delay = (delay * 2).min(std::time::Duration::from_secs(300));
        }
        target_gpu = inference_gpu_for_model(&model_id);
    }
}

/// VRAM headroom (MiB) required ON TOP of a smaller tier's `min_vram_mb` when demoting a card that
/// OOMed — the walk gather + inference context need room beyond the raw weight budget.
const DEMOTE_VRAM_HEADROOM_MB: u64 = 2_000;

/// The largest supported model STRICTLY smaller (by `min_vram_mb`) than `current`, whose files are
/// staged (`.ok`) and whose budget + headroom fits `vram_mb`. Used to demote a card that OOMed on a
/// too-big tier to one that actually fits, so it mines/serves SOMETHING instead of looping on OOM.
/// `None` when nothing smaller is both ready and fits — the caller then halts that card with guidance.
pub fn next_smaller_ready_spec(current: &[u8; 32], vram_mb: u64) -> Option<&'static ModelSpec> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let cur_budget = specs.iter().find(|s| &s.model_id == current).map(|s| s.min_vram_mb)?;
    specs
        .iter()
        .copied()
        .filter(|s| s.min_vram_mb < cur_budget)
        .filter(|s| spec_files_ready(s))
        .filter(|s| s.min_vram_mb + DEMOTE_VRAM_HEADROOM_MB <= vram_mb)
        .max_by_key(|s| s.min_vram_mb)
}

/// True while a model file is actively downloading via IPFS (set around the download in
/// `ensure_gguf`). Lets the hashrate reporter say "downloading model" instead of the alarming
/// "workers stalled or crashed" during the (potentially many-minute) first-run fetch.
static DOWNLOADING: AtomicBool = AtomicBool::new(false);

/// Whether a model download is in progress right now.
pub fn is_downloading() -> bool {
    DOWNLOADING.load(AtomicOrdering::Relaxed)
}

/// True when the miner is still in first-run PREPARATION — no model is fully staged yet
/// (downloading, or waiting to). While this holds, a 0 h/s reading is EXPECTED, not a stall.
/// Backend-agnostic (reads the shared `.ok`/download state); the CUDA index/resident-tree
/// build window is reported separately via `pom_gpu::is_loading()`.
pub fn mining_preparing() -> bool {
    is_downloading() || loaded_model_ids().is_empty()
}

/// The UNION of model_ids this rig can serve = every supported-lineup model with fully-downloaded
/// files (`.ok`), PLUS every per-card assigned tier on a mixed rig (a smaller per-card model that is
/// NOT in the process-wide lineup is still servable on the card that mines it). This drives
/// declare_capabilities, so the pool routes every tier at least one card can serve — no under- or
/// over-declaration on a mixed rig.
/// Models whose files are on disk but which this miner cannot currently serve (upstream 0795e92).
fn unavailable_models() -> &'static RwLock<std::collections::HashSet<[u8; 32]>> {
    static MODELS: OnceLock<RwLock<std::collections::HashSet<[u8; 32]>>> = OnceLock::new();
    MODELS.get_or_init(|| RwLock::new(std::collections::HashSet::new()))
}

/// Withdraw a model from `ai:cap`: the files are on disk but this miner cannot serve it right now
/// (upstream 0795e92). Announcing it anyway earns assigned requests it cannot answer, hence
/// service-bond strikes. Idempotent — only the transition logs.
pub fn mark_model_unavailable(model_id: &[u8; 32], reason: &str) {
    if unavailable_models().write().unwrap().insert(*model_id) {
        log::warn!("SlmEngine: model {:.8} withdrawn from ai:cap ({})", hex::encode(model_id), reason);
    }
    publish_runtime_model_status(model_id);
}

/// Re-announce a model after it serves again (upstream 0795e92).
pub fn mark_model_available(model_id: &[u8; 32], reason: &str) {
    if unavailable_models().write().unwrap().remove(model_id) {
        log::info!("SlmEngine: model {:.8} back in ai:cap ({})", hex::encode(model_id), reason);
    }
    publish_runtime_model_status(model_id);
}

fn model_is_unavailable(model_id: &[u8; 32]) -> bool {
    unavailable_models().read().unwrap().contains(model_id)
}

pub fn loaded_model_ids() -> Vec<[u8; 32]> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let mut ids: Vec<[u8; 32]> = specs
        .iter()
        .filter(|s| spec_files_ready(s) && !model_is_unavailable(&s.model_id))
        .map(|s| s.model_id)
        .collect();
    // Per-card assignments (mixed rig / --force-model): add any assigned tier whose GGUF is staged
    // (its dir's `.ok` present) and not already listed. pom-cuda only — the map is empty elsewhere.
    #[cfg(feature = "pom-cuda")]
    for (mid, gguf) in crate::pom_gpu::assigned_models() {
        let marker_ready = std::path::Path::new(&gguf).parent().map_or(false, |d| d.join(".ok").exists());
        let ready = marker_ready && crate::gguf::is_complete_file(std::path::Path::new(&gguf));
        if ready && !ids.contains(&mid) && !model_is_unavailable(&mid) {
            ids.push(mid);
        }
    }
    ids
}

/// True when a specific model spec's files are fully downloaded on disk (`.ok` sentinel present).
/// Unlike `is_model_ready`, this does NOT consult SUPPORTED_SPECS — so it can be called during
/// `--tier auto` selection, before the lineup is staged via `init_supported`.
pub fn spec_files_ready(spec: &ModelSpec) -> bool {
    model_dir(spec).join(".ok").exists()
}

/// One human-readable status line per supported-lineup model: the exact GGUF path the miner expects
/// and whether it's ready, or the specific reason it isn't (absent / not a GGUF / truncated / no .ok
/// yet). Drives the periodic diagnostic the miner prints while stuck in "preparing models
/// (staging/verifying files)", so a manually-copied model that never becomes ready explains itself
/// instead of looping silently. Empty when the lineup hasn't been installed yet.
pub fn staging_diagnostics() -> Vec<String> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    specs
        .iter()
        .map(|s| {
            let dir = model_dir(s);
            let gguf = dir.join("model.gguf");
            let ok = dir.join(".ok").exists();
            match crate::gguf::completeness_reason(&gguf) {
                None if ok => format!("  '{}': READY ({})", s.name, gguf.display()),
                None => format!(
                    "  '{}': GGUF complete but not yet adopted at {} (miner will write .ok on next staging pass)",
                    s.name,
                    gguf.display()
                ),
                Some(reason) => format!("  '{}': NOT ready — {} [{}]", s.name, gguf.display(), reason),
            }
        })
        .collect()
}

/// True only when the model is supported, its files are completely downloaded, and it is not
/// currently withdrawn from `ai:cap` (upstream 0795e92).
pub fn is_model_ready(model_id: &[u8; 32]) -> bool {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else {
        return false;
    };
    model_dir(spec).join(".ok").exists() && !model_is_unavailable(model_id)
}

/// Load the requested model on demand (evicting a cached different model if needed), then run
/// inference. Blocking — call from `spawn_blocking`. Routes to the best FREE eligible card via the
/// policy router (single-GPU / no-CUDA collapses to the one card). Callers that need to reply "busy"
/// or pause PoW BEFORE dispatch should instead `acquire_inference_card(..)` themselves and call
/// `load_and_run_inference_on(lease.gpu(), ..)` (see the stratum handlers).
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    if let Err(reason) = validate_inference_request(prompt, max_tokens, DEFAULT_INFERENCE_DEADLINE_MS) {
        log::warn!("OPoI: rejected inference request: {}", reason);
        return None;
    }
    // Default deadline for callers that don't pass one (grpc solo path, capability challenge):
    // wait up to 30s for a free card, else give up (None → skipped, no hard reject upstream).
    let lease = acquire_inference_card(model_id, DEFAULT_INFERENCE_DEADLINE_MS)?;
    load_and_run_inference_on(lease.gpu(), model_id, prompt, max_tokens)
    // lease drops here → card released on every exit path.
}

/// As `load_and_run_inference`, but runs on the caller-leased CUDA card `gpu` (no migration). The
/// in-process llama.cpp engine branch is card-aware (its own resident model per card + per-card
/// generate); the candle/vk fallbacks remain the single-card dormant path.
pub fn load_and_run_inference_on(gpu: usize, model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    if let Err(reason) = validate_inference_request(prompt, max_tokens, DEFAULT_INFERENCE_DEADLINE_MS) {
        log::warn!("OPoI: rejected inference request for GPU {}: {}", gpu, reason);
        return None;
    }
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;

    // Race guard: never attempt inference (or a card model-swap that uninstalls the walk) for a
    // model whose GGUF is not fully on disk yet. The initial background prefetch may still be
    // downloading/loading it; serving now would uninstall+reload mid-prefetch. Drop the request —
    // the pool re-challenges once the model is ready (loaded_model_ids / ai:cap track readiness).
    if !crate::gguf::is_complete_file(&gguf_path_for(spec)) {
        log::warn!(
            "OPoI: model '{}' not fully staged yet — skipping inference (avoids a load race); will serve once ready.",
            spec.name
        );
        record_unserveable_on(model_id, gpu);
        if !any_proven_route(model_id) {
            mark_model_unavailable(model_id, "model_file_incomplete");
        }
        return None;
    }

    // GPU inference and the possession walk must never overlap on the serving card. Publishing a
    // stopped job is not itself a completion barrier: a worker may already be inside a long kernel.
    // The backend guard raises its per-card pause before draining that transient walk owner, then
    // keeps the bit raised through model load/generation. This also covers the same-model zero-dup
    // fast path, which does not uninstall the walk and was previously left unprotected.
    #[cfg(any(
        all(feature = "pom-cuda", not(feature = "pom-opencl")),
        all(target_os = "macos", feature = "pom-metal"),
    ))]
    let _walk_drain = match crate::pom_gpu::pause_and_drain_for_inference(gpu) {
        Some(guard) => guard,
        None => return None,
    };
    #[cfg(feature = "pom-opencl")]
    let _walk_drain = match crate::pom_opencl::pause_and_drain_for_inference(gpu) {
        Some(guard) => guard,
        None => return None,
    };

    // Prefer a running llama.cpp llama-server: AMD always (candle has no AMD-GPU backend; Vulkan
    // server), NVIDIA when a CUDA llama-server is bundled/env-pointed (Phase 1 of candle-
    // independence — llama.cpp tracks new GGUF archs faster than candle). The OPoI text is
    // user-facing only (consensus checks the fixed-point `model_fixed` commitment separately), so
    // a non-candle engine is fine. Falls through to the candle engine below when the server isn't
    // available. CPU is considered only afterward when the operator explicitly enabled the
    // deprecated emergency fallback.
    // Highest priority: the IN-PROCESS llama.cpp engine (Phase 2 on CUDA, Phase 3b on Apple
    // Silicon Metal). Ranks above the candle path so any host that bundles the .so/.dylib
    // gets fully candle-independent inference.
    #[cfg(any(
        all(feature = "pom-cuda", not(feature = "pom-opencl")),
        all(target_os = "macos", feature = "pom-metal"),
    ))]
    {
        if !cpu_inference_enabled() {
            let gguf = gguf_path_for(spec).to_string_lossy().into_owned();
            // A healthy exact subprocess already has the requested model resident. Reuse it in
            // the server branch below instead of evicting the mining walk and allocating a second
            // copy in-process. Conversely, stop/reap any live non-matching child before the
            // higher-priority in-process engine allocates: an unhealthy or mismatched server can
            // still own VRAM even though it is not a serveable route.
            #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
            let exact_server_ready = crate::llama_vulkan::reuse_exact_or_stop(&gguf, gpu);
            #[cfg(not(all(feature = "pom-cuda", not(feature = "pom-opencl"))))]
            let exact_server_ready = false;
            // If this card doesn't already host the requested model, free ITS walk (per-card — other
            // cards keep mining) so the inference model fits, then load it on this card. When the model
            // is ALREADY resident here (the zero-dup mining tier) this is a no-op: no uninstall, no
            // reload — byte-identical to the pre-router single-card path.
            if !exact_server_ready && !crate::llama_engine::active_for(&gguf, gpu) {
                // `_walk_drain` above already paused this exact card and waited for its current batch.
                // Now it is safe to remove a walk whose pointers may alias the old llama model.
                if !crate::pom_gpu::uninstall_released(gpu as u32) {
                    log::error!(
                        "OPoI: GPU {} walk did not drain; refusing to free/swap llama weights underneath it",
                        gpu
                    );
                    return None;
                }
                if !crate::llama_engine::ensure_loaded_on(&gguf, gpu) {
                    log::warn!(
                        "SlmEngine: in-process llama load failed on GPU {} — trying configured llama-server fallback.",
                        gpu
                    );
                }
            }
            if !exact_server_ready && crate::llama_engine::active_for(&gguf, gpu) {
                let t0 = std::time::Instant::now();
                // --only-inference: yield the card to the request. Returns None (no-op) on a normal rig,
                // where mining continues through the generation as it always has.
                if let Some(text) = crate::llama_engine::generate_for(gpu, &gguf, prompt, max_tokens) {
                    let secs = t0.elapsed().as_secs_f64();
                    if secs > 0.0 {
                        record_card_toks(gpu, text.split_whitespace().count() as f64 / secs);
                    }
                    let clean = strip_think_tags(&text);
                    if !clean.trim().is_empty() {
                        // Record proof only after post-processing. A response consisting solely of a
                        // truncated <think> block is not an answer and must never enable ai:cap.
                        record_serveable_on(model_id, gpu);
                        mark_model_available(model_id, "generation_success");
                        return Some(clean);
                    }
                    log::warn!(
                    "SlmEngine: in-process llama output on GPU {} was empty after stripping think tags — trying the next engine.",
                    gpu
                );
                }
                // REVERTED in v0.12.6 — do NOT return early here.
                //
                // v0.12.4 made this path terminal ("upstream parity") to stop a rare cascade that
                // wedged a rig. That fix caused a MUCH worse regression: on every MULTI-GPU rig the
                // cards sat at "preparing" forever with "OPoI: no models ready — mining suspended".
                // A/B on a mixed three-GPU validation rig (2x 3070 + 1x 5080, same config, healthy cards):
                //   v0.12.3 -> all three cards mine (1.31 + 1.31 + 3.26 MH/s), probes PASS on GPU 2
                //   v0.12.5 -> all three stall, self-test FAILS on GPU 1 (a 3070)
                // Failing fast here makes the self-test probe fail, which fires the inference-host
                // FAILOVER onto a small card that cannot hold the 12 GB tier; the model is then
                // withdrawn from ai:cap, loaded_model_ids() goes empty, and EVERY card stalls.
                // Falling through gives the probe the second chance it needs. The cascade this was
                // meant to fix hit once in hours; this broke every multi-GPU rig, so the trade is
                // clear. A safer cascade fix must distinguish the SELF-TEST probe from real OPoI
                // serving instead of making both terminal.
                log::warn!("SlmEngine: in-process llama generate failed on GPU {} — trying the next engine.", gpu);
            }
        }
    }

    // AMD (pom-opencl) in-process engine: the zero-dup host of the walk's model — its inference
    // costs no extra VRAM. Ranks above the llama-server subprocess.
    #[cfg(feature = "pom-opencl")]
    {
        let gguf = gguf_path_for(spec).to_string_lossy().into_owned();
        if !cpu_inference_enabled() && !crate::llama_vulkan::reuse_exact_or_stop(&gguf, gpu) {
            // The lifecycle-locked transition above synchronously stopped/reaped any old server
            // before the higher-priority engine claims dedication. A failed load therefore cannot
            // clear a still-resident subprocess model's reservation.
            if !crate::llama_engine_vk::active_for(&gguf) && crate::llama_engine_vk::available() {
                // A singleton holding another model must be stopped before loading this one. The
                // OpenCL drain above covers every walk while the engine frees/reallocates GPU memory.
                crate::llama_engine_vk::unload();
            }
            let engine_ready =
                crate::llama_engine_vk::active_for(&gguf) || crate::llama_engine_vk::ensure_loaded(&gguf, gpu);
            // The loaded engine must agree with the exact PCI reservation established before
            // allocation. Validate now, before the first generation, so a sidecar/driver mapping
            // mismatch never earns a capability proof.
            let engine_scoped = engine_ready && crate::pom_opencl::dedicate_loaded_engine_card_if_required();
            if engine_ready && !engine_scoped {
                log::error!("SlmEngine: unloading Vulkan engine whose GPU cannot be scoped safely to the selected OpenCL workers");
                crate::llama_engine_vk::unload();
            }
            if engine_scoped {
                if let Some(text) = crate::llama_engine_vk::generate_for(&gguf, prompt, max_tokens) {
                    let clean = strip_think_tags(&text);
                    if !clean.trim().is_empty() {
                        record_serveable_on(model_id, gpu);
                        mark_model_available(model_id, "llama_vk_generation_success");
                        return Some(clean);
                    }
                    log::warn!("SlmEngine: in-process llama-vk output was empty after stripping think tags.");
                }
                log::warn!("SlmEngine: in-process llama-vk generate failed — trying the next engine.");
            }
        }
    }

    #[cfg(any(feature = "pom-opencl", feature = "pom-cuda"))]
    {
        if !cpu_inference_enabled() {
            let gguf = gguf_path_for(spec).to_string_lossy().into_owned();
            if !crate::llama_vulkan::available_for(&gguf, gpu) && crate::llama_vulkan::configured() {
                // CUDA: a failed in-process engine may still own the model buffers (and the walk may
                // alias them). Tear down the drained walk before freeing those buffers and launching
                // the configured subprocess on the exact same logical card.
                #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
                {
                    if !crate::pom_gpu::uninstall_released(gpu as u32) {
                        log::error!("OPoI: GPU {} walk remained referenced; refusing llama-server model swap", gpu);
                        record_unserveable_on(model_id, gpu);
                        if !any_proven_route(model_id) {
                            mark_model_unavailable(model_id, "walk_drain_failed");
                        }
                        return None;
                    }
                    crate::llama_engine::unload_for_gpu(gpu);
                }
                #[cfg(feature = "pom-opencl")]
                if crate::llama_engine_vk::active_for(&gguf) {
                    crate::llama_engine_vk::unload();
                }
                // An unset/invalid/zero port asks the OS for a free per-process loopback port.
                // Explicit ports are reserved and refused if another miner already owns them.
                let port = std::env::var("KERYX_LLAMA_PORT").ok().and_then(|s| s.parse::<u16>().ok()).unwrap_or(0);
                let _ = crate::llama_vulkan::try_start_on(&gguf, port, gpu);
            }
            if crate::llama_vulkan::available_for(&gguf, gpu) {
                if let Some(text) = crate::llama_vulkan::generate_for(&gguf, gpu, prompt, max_tokens) {
                    let clean = strip_think_tags(&text);
                    if !clean.trim().is_empty() {
                        #[cfg(all(feature = "pom-opencl", unix))]
                        if !crate::llama_vulkan::vulkan_server_ggml_device()
                            .is_some_and(crate::pom_opencl::resolve_vulkan_server_dedication)
                        {
                            log::error!(
                                "SlmEngine: stopping Vulkan llama-server because its GPU \
                                 cannot be mapped safely to the selected OpenCL workers"
                            );
                            crate::llama_vulkan::stop();
                            crate::pom_opencl::release_provisional_dedication_after_server_stop();
                            return None;
                        }
                        if !crate::llama_vulkan::commit_route_success(&gguf, gpu, model_id) {
                            // The response itself is still valid for this request, but its child
                            // exited or was replaced before proof publication. Return the answer
                            // without reopening ai:cap; the durable warmer will earn a live proof.
                            log::warn!(
                                "SlmEngine: llama-server answered, but its exact process was no longer live at route-proof commit"
                            );
                        }
                        return Some(clean);
                    }
                    log::warn!("SlmEngine: llama-server output was empty after stripping think tags.");
                }
                log::warn!(
                "SlmEngine: llama-server inference returned nothing; trying only explicitly enabled legacy fallbacks."
            );
            }
        }
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
            #[cfg(any(feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
            if !crate::pom_gpu::uninstall_released(gpu as u32) {
                log::error!("OPoI: GPU {} walk did not drain; legacy inference load deferred", gpu);
                return None;
            }
            *guard = None;
            log::info!("SlmEngine: legacy inference device active (CUDA:{})", gpu);
            match load_legacy_engine_on(spec, gpu) {
                Ok(e) => {
                    *guard = Some(e);
                }
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
                None
            }
        }
    });

    match result {
        Ok(Some(output)) => {
            let clean = strip_think_tags(&output);
            if !clean.trim().is_empty() {
                record_serveable_on(model_id, gpu);
                mark_model_available(model_id, "legacy_generation_success");
                Some(clean)
            } else {
                record_unserveable_on(model_id, gpu);
                if !any_proven_route(model_id) {
                    mark_model_unavailable(model_id, "empty_generation");
                }
                None
            }
        }
        Ok(None) => {
            record_unserveable_on(model_id, gpu);
            if !any_proven_route(model_id) {
                mark_model_unavailable(model_id, "all_gpu_inference_routes_failed");
            }
            None
        }
        Err(_) => {
            log::error!("SlmEngine: inference panicked — engine evicted, will retry on next challenge");
            log::error!(
                "SlmEngine: cuBLAS missing? Run: sudo apt-get install -y libcublas-12-2 then restart the miner"
            );
            if let Ok(mut g) = ENGINE.lock() {
                *g = None;
            }
            record_unserveable_on(model_id, gpu);
            if !any_proven_route(model_id) {
                mark_model_unavailable(model_id, "inference_panic");
            }
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
                hex::encode(&model_id[..4]),
                specs.len()
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
    // Legacy candle sharing is GPU-only. The active H6 lineup is hosted by llama_engine instead;
    // pretending it fell back to CPU made capability and pause accounting incorrect.
    match load_legacy_engine_on(spec, inference_gpu_for_model(model_id)) {
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
pub fn pom_shared(model_id: &[u8; 32]) -> Option<(Device, std::collections::HashMap<String, Arc<QTensor>>)> {
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

#[cfg(test)]
mod cuda_device_set_tests {
    use super::parse_cuda_devices;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn reads_the_processes_own_device_set_from_argv() {
        assert_eq!(parse_cuda_devices(args(&["keryx-miner-supr", "--cuda-device", "0"])), vec![0]);
        assert_eq!(
            parse_cuda_devices(args(&["keryx-miner-supr", "--cuda-device", "1,2,3", "-s", "pool"])),
            vec![1, 2, 3]
        );
        assert_eq!(parse_cuda_devices(args(&["keryx-miner-supr", "--cuda-device=2,0"])), vec![2, 0]);
    }

    #[test]
    fn absent_or_malformed_means_all_devices() {
        // No flag at all: the miner's default is every GPU, so the set is empty (= unrestricted).
        assert!(parse_cuda_devices(args(&["keryx-miner-supr", "-s", "pool"])).is_empty());
        // Flag present but with nothing after it — must not panic or index past the end.
        assert!(parse_cuda_devices(args(&["keryx-miner-supr", "--cuda-device"])).is_empty());
        // Junk entries are dropped rather than poisoning the whole set.
        assert_eq!(parse_cuda_devices(args(&["m", "--cuda-device", "0,x,2"])), vec![0, 2]);
    }
}

#[cfg(test)]
mod withdrawal_tests {
    use super::*;

    #[test]
    fn withdrawn_models_are_hidden_until_they_recover() {
        let model_id = [0xa7u8; 32];
        assert!(!model_is_unavailable(&model_id));

        mark_model_unavailable(&model_id, "test_failure");
        assert!(model_is_unavailable(&model_id));
        assert!(!is_model_ready(&model_id));
        assert!(!loaded_model_ids().contains(&model_id));

        mark_model_available(&model_id, "test_recovery");
        assert!(!model_is_unavailable(&model_id));
    }

    #[test]
    fn withdrawal_is_idempotent_per_model() {
        let model_id = [0xb3u8; 32];
        mark_model_unavailable(&model_id, "first");
        mark_model_unavailable(&model_id, "second");
        assert!(model_is_unavailable(&model_id));

        mark_model_available(&model_id, "recovered");
        mark_model_available(&model_id, "recovered_again");
        assert!(!model_is_unavailable(&model_id));

        let other = [0xc1u8; 32];
        mark_model_unavailable(&model_id, "again");
        assert!(!model_is_unavailable(&other));
    }

    #[test]
    fn amd_engine_eviction_clears_every_cached_route_and_withdraws_model() {
        let model_id = [0xd4u8; 32];
        self_test_state().write().unwrap().insert((model_id, 0), true);
        self_test_state().write().unwrap().insert((model_id, 7), true);
        mark_model_available(&model_id, "test_setup");

        withdraw_model_after_amd_engine_eviction(&model_id);

        assert!(!self_test_state().read().unwrap().keys().any(|(mid, _)| mid == &model_id));
        assert!(model_is_unavailable(&model_id));
        mark_model_available(&model_id, "test_cleanup");
    }

    #[test]
    fn exact_route_invalidation_preserves_other_models_and_cards() {
        let model_a = [0xe1u8; 32];
        let model_b = [0xe2u8; 32];
        {
            let mut state = self_test_state().write().unwrap();
            state.insert((model_a, 0), true);
            state.insert((model_a, 1), true);
            state.insert((model_b, 0), true);
        }

        invalidate_inference_route(&model_a, 0, "test_exact_server_exit");

        let state = self_test_state().read().unwrap();
        assert!(!state.contains_key(&(model_a, 0)));
        assert_eq!(state.get(&(model_a, 1)), Some(&true));
        assert_eq!(state.get(&(model_b, 0)), Some(&true));
        drop(state);

        clear_self_test(&model_a);
        clear_self_test(&model_b);
        mark_model_available(&model_a, "test_cleanup");
        mark_model_available(&model_b, "test_cleanup");
    }
}

#[cfg(test)]
mod inference_admission_tests {
    use super::*;

    #[test]
    fn accepts_bounded_request_and_normalizes_zero_deadline() {
        assert_eq!(validate_inference_request("hello", 128, 0), Ok(DEFAULT_INFERENCE_DEADLINE_MS));
        assert_eq!(
            validate_inference_request(
                &"x".repeat(MAX_INFERENCE_PROMPT_BYTES),
                MAX_INFERENCE_TOKENS,
                MAX_INFERENCE_DEADLINE_MS,
            ),
            Ok(MAX_INFERENCE_DEADLINE_MS)
        );
    }

    #[test]
    fn rejects_empty_or_oversized_remote_controls() {
        assert_eq!(validate_inference_request("", 1, 1), Err("prompt is empty"));
        assert!(validate_inference_request(&"x".repeat(MAX_INFERENCE_PROMPT_BYTES + 1), 1, 1,).is_err());
        assert!(validate_inference_request("x", 0, 1).is_err());
        assert!(validate_inference_request("x", MAX_INFERENCE_TOKENS + 1, 1).is_err());
        assert!(validate_inference_request("x", 1, MAX_INFERENCE_DEADLINE_MS + 1).is_err());
    }

    #[test]
    fn internal_wait_deadlines_are_always_bounded() {
        assert_eq!(bounded_deadline_ms(0), DEFAULT_INFERENCE_DEADLINE_MS);
        assert_eq!(bounded_deadline_ms(u64::MAX), MAX_INFERENCE_DEADLINE_MS);
    }
}

#[cfg(test)]
mod inference_probe_tests {
    use super::{current_probe_passed, select_runtime_status_route};

    #[test]
    fn stale_cached_success_cannot_reopen_a_failed_route() {
        let stale_cached_success = true;
        let current_probe_output: Option<&str> = None;

        assert!(stale_cached_success, "regression setup must represent an old successful probe");
        assert!(!current_probe_passed(current_probe_output));
        assert!(!current_probe_passed(Some("  \n\t")));
        assert!(current_probe_passed(Some("OK")));
    }

    #[test]
    fn aggregate_readiness_selects_a_model_with_an_exact_proven_route() {
        let failed_model = [0x41; 32];
        let ready_model = [0x42; 32];
        let candidates = vec![(failed_model, Vec::new()), (ready_model, vec![1, 3])];

        assert_eq!(
            select_runtime_status_route(failed_model, Vec::new(), &candidates),
            (ready_model, vec![1, 3])
        );
    }

    #[test]
    fn requested_model_keeps_its_own_exact_route_when_still_proven() {
        let requested_model = [0x51; 32];
        let other_model = [0x52; 32];
        let candidates = vec![(other_model, vec![7])];

        assert_eq!(
            select_runtime_status_route(requested_model, vec![2], &candidates),
            (requested_model, vec![2])
        );
    }
}
