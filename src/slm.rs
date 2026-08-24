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
use std::sync::{Arc, Mutex, OnceLock, RwLock};
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
    if let Ok(mut g) = staging_error_slot().lock() { *g = Some(msg.into()); }
}
/// Clear the staging failure once a model is ready again.
pub fn clear_staging_error() {
    if let Ok(mut g) = staging_error_slot().lock() { *g = None; }
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
    if !out.status.success() { return None; }
    let text = String::from_utf8_lossy(&out.stdout);
    let avail_kb: u64 = text.lines().last()?.split_whitespace().nth(3)?.parse().ok()?;
    Some(avail_kb.saturating_mul(1024))
}
#[cfg(not(unix))]
fn free_disk_bytes(_path: &std::path::Path) -> Option<u64> { None }

fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 240; // survives long gateway outages (~40 min of retries)
    const BACKOFF_SECS: u64 = 10;
    // Mark the process as "downloading" for the whole fetch (incl. retries/resume) so the
    // hashrate reporter shows "downloading model" instead of "workers stalled or crashed".
    // RAII guard clears it on every exit path (success, error, early return).
    DOWNLOADING.store(true, AtomicOrdering::Relaxed);
    struct DlGuard;
    impl Drop for DlGuard {
        fn drop(&mut self) { DOWNLOADING.store(false, AtomicOrdering::Relaxed); }
    }
    let _dl = DlGuard;
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
            // A resumed request for an already-complete file gets HTTP 416 (Range Not Satisfiable),
            // which ureq surfaces as Err(Status(416, _)) — so the `status == 416` arm below is never
            // reached. Treat it here as "already fully downloaded" (e.g. a GGUF pre-fetched via
            // aria2, or a re-run over a complete file) instead of looping on it forever.
            Err(ureq::Error::Status(416, _)) if resume_from > 0 => {
                eprintln!("  already complete ({} MB).", resume_from / 1_000_000);
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
                // 200, or the server ignored Range. Never wipe a local file that already matches
                // the remote size — IPFS gateways often ignore Range and answer 200 + full
                // Content-Length, which previously truncated multi-GB GGUFs back to zero.
                let total = response.header("Content-Length").and_then(|s| s.parse::<u64>().ok());
                if resume_from > 0 {
                    if let Some(t) = total {
                        if resume_from >= t {
                            eprintln!("\r  already complete ({} MB).            ", resume_from / 1_000_000);
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
                        eprintln!(
                            "\n[keryx-miner] server ignored Range (HTTP {status}); keeping local {} MB, retry {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s…",
                            resume_from / 1_000_000
                        );
                        std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
                        continue;
                    }
                }
                let f = std::fs::File::create(dest)
                    .with_context(|| format!("create {}", dest.display()))?;
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
                        dest.display(), (need + MARGIN) / 1_000_000, need / 1_000_000,
                        free / 1_000_000, where_dir.display()
                    );
                    log::error!("[keryx-miner] {msg}");
                    eprintln!("\n[keryx-miner] ERROR: {msg}\n");
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
                        eprint!("\r  {:.1}/{:.1} MB ({}%)   ",
                            downloaded as f64 / 1_000_000.0,
                            t as f64 / 1_000_000.0,
                            downloaded * 100 / t.max(1));
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
                                downloaded as f64 / 1e6, t as f64 / 1e6,
                                downloaded * 100 / t.max(1), rate / 1e6, eta_min.max(1.0),
                            );
                        }
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
                spec.name, dir.display()
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
            spec.name, gguf.display(), reason, spec.dir_name
        );
    } else if need_tok && !tok.exists() {
        log::warn!(
            "SlmEngine: model '{}' GGUF is complete at {} but tokenizer.json is missing — will fetch it.",
            spec.name, gguf.display()
        );
    }

    std::fs::create_dir_all(&dir)?;
    if ok_flag.exists() && !gguf_ready {
        log::warn!(
            "SlmEngine: '{}' at {} has a stale .ok but the GGUF is incomplete (truncated?) — repairing.",
            spec.name, gguf.display()
        );
    }
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before (re)downloading

    if !gguf_ready {
        eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
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
                spec.name, gguf.display(), reason, gguf.display()
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
        // Llama-3.3-70B (abliterated / uncensored — our `--very-high`) — re-templated to ChatML.
        // The vocab is still stock LLaMA-3, so <|im_end|>/<|im_start|> are NOT atomic tokens: the
        // model writes "<|im_end|>" as plain multi-token text and never emits <|eot_id|> (128009).
        // Neither the id set nor an <|eot_id|> stop-string ever fires — only a stop-STRING on the
        // ChatML markers cuts the turn. Without it the model completes its turn, prints the marker,
        // opens `assistant`, and loops the same answer to max_tokens (a failed OPoI inference).
        // (upstream Keryx-Labs/keryx-miner@faee090)
        "llama-3.3-70b" => (
            collect_stop_ids(tokenizer, &["<|eot_id|>", "<|end_of_text|>"], &[128009, 128001]),
            vec!["<|im_end|>", "<|im_start|>", "<|eot_id|>", "<|end_of_text|>"],
        ),
        // Genuine official Llama-3.3-70B-Instruct — LLaMA-3 header template. Stop on the official
        // `eos_token_id` set: 128009 <|eot_id|>, 128001 <|end_of_text|>, 128008 <|eom_id|>.
        "llama-3.3-70b-official" => (
            collect_stop_ids(tokenizer,
                &["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"],
                &[128009, 128001, 128008]),
            vec!["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
        ),
        // Qwen3 (32B and 1.7B share the same ChatML template + 151k vocab):
        //   151645 = <|im_end|> (end of turn), 151643 = <|endoftext|> (base EOS).
        // The 1.7B must NOT fall through to the generic </s> stops — Qwen3 never emits </s>, so it
        // would run past <|im_end|> and loop into a fresh turn. (upstream Keryx-Labs/keryx-miner@a033620)
        "qwen3-32b" | "qwen3-1.7b" => (
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
                spec.name, spec.dir_name
            )
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
    // Fall back to CPU on ANY GPU failure — CUDA (NVIDIA) or Metal (Apple Silicon). A Metal device
    // may init fine but hit an unsupported quantized op mid-load; that surfaces here and degrades.
    let on_gpu = device.is_cuda() || device.is_metal();

    match try_load_engine(spec, device) {
        Ok(engine) => Ok(engine),
        Err(reason) if on_gpu && cpu_inference_allowed() => {
            // GPU load failed AND the operator opted into CPU fallback (--enable-cpu-inference /
            // --cpu-inference) → degrade to CPU instead of withdrawing.
            log::warn!(
                "⚠️ GPU inference FAILED to load on this device ({reason}) — falling back to CPU \
                 inference (MUCH slower; you enabled it). The PoW walk still runs on the GPU."
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
        Err(reason) if on_gpu => {
            // GPU load failed and CPU fallback is DISABLED (the default). Do NOT waste cycles on
            // glacial CPU inference — surface a clear, actionable error; the caller withdraws the
            // model from OPoI. The real fix is a working in-process llama.cpp engine on the GPU.
            Err(anyhow!(
                "GPU inference could not load '{}' ({}). CPU fallback is OFF by default — the model \
                 is withdrawn from OPoI rather than run on the CPU (far too slow to be useful). \
                 Fix: ensure the in-process llama.cpp engine is present and loads next to the miner \
                 — Linux: libkeryx-llama.so; Windows: keryx-llama.dll — together with the bundled \
                 CUDA runtime libs, and a driver new enough for the build (modern R575+, legacy \
                 R535+). To force CPU anyway, pass --enable-cpu-inference.",
                spec.name, reason
            ))
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
        // Qwen3 (32B + 1.7B) — ChatML template. `/no_think` disables the thinking block
        // so the assistant answers directly (only an empty <think></think> to strip).
        "qwen3-32b" | "qwen3-1.7b" => format!(
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
        "</think>",            // Qwen3.x / GLM-4 / DeepSeek ChatML think block
        "<channel|>message",   // Gemma-4 <|channel>thought … <channel|>message<answer>
        "<channel|>",          // Gemma-4 fallback (empty/closed thought channel)
        "<|/thought|>",        // GLM channel-thought variant
    ];
    for close in CLOSERS {
        if let Some(pos) = text.rfind(close) {
            return text[pos + close.len()..].trim().to_string();
        }
    }
    text.trim().to_string()
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
    Ok(if matches!(engine.name, "qwen3-32b" | "qwen3-1.7b" | "deepseek-r1-8b" | "deepseek-r1-32b") {
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

/// Whether CPU inference is ALLOWED as a fallback (default: NO). Off unless the operator passes
/// `--enable-cpu-inference` (or `--cpu-inference`, which forces CPU from the start). When off, a GPU
/// that cannot load inference withdraws the model from OPoI instead of degrading to useless,
/// glacially-slow CPU inference — see `load_engine_with_fallback`. Forced on for the AMD/OpenCL
/// build (candle 0.9 has no AMD-GPU backend, so CPU is the only path there).
static CPU_INFERENCE_ALLOWED: AtomicBool = AtomicBool::new(false);

/// True when CPU inference may be used (opt-in via `--enable-cpu-inference`/`--cpu-inference`, or
/// always on the AMD/OpenCL build). Gates the automatic GPU→CPU fallback.
pub fn cpu_inference_allowed() -> bool {
    #[cfg(feature = "pom-opencl")]
    { true }
    #[cfg(not(feature = "pom-opencl"))]
    { CPU_INFERENCE_ALLOWED.load(AtomicOrdering::Relaxed) }
}

/// Permit CPU inference (set from the CLI when `--enable-cpu-inference` or `--cpu-inference` is given).
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
    std::env::var("KERYX_INFERENCE_CARDS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse::<usize>().ok()).collect())
        .unwrap_or_default()
}

/// Per-card inference busy flags (keyed by CUDA ordinal). A set flag ⇒ that card is mid-generation.
fn card_busy() -> &'static Mutex<std::collections::HashMap<usize, bool>> {
    static B: OnceLock<Mutex<std::collections::HashMap<usize, bool>>> = OnceLock::new();
    B.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn try_claim_card(gpu: usize) -> bool {
    let mut g = match card_busy().lock() { Ok(g) => g, Err(p) => p.into_inner() };
    let slot = g.entry(gpu).or_insert(false);
    if *slot { false } else { *slot = true; true }
}

fn release_card(gpu: usize) {
    let mut g = match card_busy().lock() { Ok(g) => g, Err(p) => p.into_inner() };
    g.insert(gpu, false);
}

fn card_is_free(gpu: usize) -> bool {
    let g = match card_busy().lock() { Ok(g) => g, Err(p) => p.into_inner() };
    !g.get(&gpu).copied().unwrap_or(false)
}

/// RAII lease of ONE inference card. While held, the card is claimed for a single generation;
/// dropping it (normal return, `?`, or a panic unwind inside `catch_unwind`) releases the card so
/// the next queued request can use it. A request is NEVER migrated to another card once leased.
pub struct InferenceLease {
    gpu: usize,
}
impl InferenceLease {
    pub fn gpu(&self) -> usize { self.gpu }
}
impl Drop for InferenceLease {
    fn drop(&mut self) { release_card(self.gpu); }
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
        if let Ok(mut g) = card_toks().lock() { g.insert(gpu, toks_per_s); }
    }
}

fn card_toks_get(gpu: usize) -> Option<f64> {
    card_toks().lock().ok().and_then(|g| g.get(&gpu).copied())
}

/// Optional per-card PoW hashrate feed for the `PowMin` policy (H/s). The mining loop MAY call this;
/// when a card has no reported value the policy falls back to the total-VRAM proxy (PoM is
/// bandwidth-bound, so smaller cards ≈ lower hashrate ≈ cheapest to pause).
static CARD_HASHRATE: OnceLock<Mutex<std::collections::HashMap<usize, f64>>> = OnceLock::new();
fn card_hashrate() -> &'static Mutex<std::collections::HashMap<usize, f64>> {
    CARD_HASHRATE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}
pub fn report_card_hashrate(gpu: usize, h_per_s: f64) {
    if let Ok(mut g) = card_hashrate().lock() { g.insert(gpu, h_per_s); }
}
fn card_hashrate_get(gpu: usize) -> Option<f64> {
    card_hashrate().lock().ok().and_then(|g| g.get(&gpu).copied())
}

/// nvidia-smi `memory.total` / `memory.free` (MiB) keyed by CUDA ordinal (line order == PCI_BUS_ID
/// order, the ordinal the miner uses — see `biggest_cuda_gpu`). Empty on any failure.
fn gpu_mem_mib(query: &str) -> std::collections::HashMap<usize, u64> {
    let mut out = std::collections::HashMap::new();
    let Ok(o) = std::process::Command::new("nvidia-smi")
        .args([&format!("--query-gpu={}", query), "--format=csv,noheader,nounits"])
        .output()
    else { return out };
    if !o.status.success() { return out; }
    for (i, line) in String::from_utf8_lossy(&o.stdout).lines().enumerate() {
        if let Ok(v) = line.trim().parse::<u64>() {
            out.insert(i, v);
        }
    }
    out
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
        let cards: Vec<usize> = crate::pom_gpu::walk_devices()
            .into_iter()
            .map(|d| d as usize)
            .filter(|g| allowed(*g))
            .collect();
        if cards.is_empty() {
            // Walk not installed yet (inference before the first PoM job), or every walk card
            // filtered out — fall back to the legacy single-card choice (respecting the restrict
            // set if it names a card).
            let d = inference_gpu_ordinal();
            return if allowed(d) { vec![d] } else { cards };
        }
        let matched: Vec<usize> = cards
            .iter()
            .copied()
            .filter(|g| {
                crate::pom_gpu::device_model(*g as u32).map_or(false, |(mid, _)| &mid == model_id)
            })
            .collect();
        if !matched.is_empty() {
            return matched;
        }
        cards
    }
    #[cfg(not(feature = "pom-cuda"))]
    {
        let _ = model_id;
        vec![inference_gpu_ordinal()]
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
    let free = if matches!(policy, InferencePolicy::Memory) { gpu_mem_mib("memory.free") } else { std::collections::HashMap::new() };
    // f64 sort key; higher == preferred. We negate for "lowest first" policies.
    let key = |g: usize| -> f64 {
        match policy {
            InferencePolicy::Speed => card_toks_get(g)
                .unwrap_or_else(|| total.get(&g).copied().unwrap_or(0) as f64 / 1.0e6),
            InferencePolicy::Memory => free.get(&g).copied().unwrap_or(0) as f64,
            #[cfg(feature = "pom-cuda")]
            InferencePolicy::Reward => card_assigned_model_bytes(g) as f64,
            #[cfg(not(feature = "pom-cuda"))]
            InferencePolicy::Reward => total.get(&g).copied().unwrap_or(0) as f64,
            InferencePolicy::PowMin => {
                // Lowest hashrate preferred ⇒ negate. Fallback proxy: lowest total VRAM.
                let h = card_hashrate_get(g)
                    .unwrap_or_else(|| total.get(&g).copied().unwrap_or(0) as f64);
                -h
            }
        }
    };
    cards.sort_by(|&a, &b| {
        key(b)
            .partial_cmp(&key(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
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
    let deadline = std::time::Instant::now()
        + std::time::Duration::from_millis(if deadline_ms == 0 { 30_000 } else { deadline_ms });
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
/// Self-test failover override: when the designated inference GPU FAILS a model self-test but
/// another walk GPU PASSES it, the winner is recorded here so every later inference site follows
/// it. -1 = no override. Without this, one bad/mispicked serving card withdrew the model rig-wide
/// and suspended EVERY card ("no inference = no mining") even though a healthy host existed.
static INFERENCE_GPU_OVERRIDE: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(-1);

pub fn inference_gpu_ordinal() -> usize {
    if let Ok(s) = std::env::var("KERYX_INFERENCE_GPU") {
        if let Ok(n) = s.trim().parse::<usize>() {
            return n;
        }
    }
    let ov = INFERENCE_GPU_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if ov >= 0 {
        return ov as usize;
    }
    // `pom_gpu` (the CUDA walk driver) only exists on the pom-cuda build. On non-CUDA builds
    // (default, and AMD/pom-opencl which places inference via llama_vulkan/KERYX_LLAMA_VK_DEVICE)
    // there are no CUDA walk devices, so fall back to an empty set → ordinal 0 (never used at
    // runtime there: cpu_inference_enabled()/llama_vulkan take over). Fixes the v0.6.5.3 non-CUDA
    // build break (slm.rs referenced crate::pom_gpu unconditionally).
    #[cfg(any(feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal")))]
    let walk = crate::pom_gpu::walk_devices();
    #[cfg(not(any(feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal"))))]
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
/// AMD build), else the GPU — the Apple Metal device on macOS, otherwise the largest-VRAM CUDA GPU
/// (NVIDIA). Single chokepoint for the inference sites. If GPU init here fails,
/// `load_engine_with_fallback` degrades to CPU rather than crashing.
fn inference_device() -> candle_core::Result<Device> {
    if cpu_inference_enabled() {
        return Ok(Device::Cpu);
    }
    // Apple Silicon (Phase 3d — candle-Metal is out of the build entirely): the primary GPU
    // inference path is the in-process llama.cpp Metal engine (`libkeryx-llama.dylib`, wired
    // through `crate::llama_engine` in `load_and_run_inference`). This `candle` device is only
    // reached as the ULTIMATE fallback when the .dylib isn't present, and in that case we want
    // CPU inference — `Device::new_metal` no longer exists (candle-core is built without the
    // `metal` feature on this platform).
    #[cfg(all(target_os = "macos", feature = "pom-metal"))]
    {
        return Ok(Device::Cpu);
    }
    #[cfg(not(all(target_os = "macos", feature = "pom-metal")))]
    {
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
            ModelFormat::Gguf | ModelFormat::GgufQwen2 | ModelFormat::GgufQwen3 | ModelFormat::GgufGemma3
            | ModelFormat::GgufExaone4 | ModelFormat::GgufGlm4 | ModelFormat::GgufQwen35 | ModelFormat::GgufKimiLinear
            | ModelFormat::GgufGemma4
                => ensure_gguf(spec).map(|_| ()),
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

fn self_test_state() -> &'static RwLock<std::collections::HashMap<[u8; 32], bool>> {
    static S: OnceLock<RwLock<std::collections::HashMap<[u8; 32], bool>>> = OnceLock::new();
    S.get_or_init(|| RwLock::new(std::collections::HashMap::new()))
}

/// True once `model_id` PASSED its inference self-test on this rig and has not since been withdrawn.
/// This is the gate for both declaring the model to the pool and mining its tier.
pub fn model_serveable(model_id: &[u8; 32]) -> bool {
    let passed = self_test_state().read().map(|m| m.get(model_id).copied() == Some(true)).unwrap_or(false);
    passed && !model_is_unavailable(model_id)
}

/// Whether a self-test has already been attempted (pass or fail) for `model_id`.
pub fn self_test_attempted(model_id: &[u8; 32]) -> bool {
    self_test_state().read().map(|m| m.contains_key(model_id)).unwrap_or(false)
}

/// Record that `model_id` is proven serveable (self-test passed, OR a real OPoI generation just
/// succeeded — a live answer is the strongest possible proof). Idempotent.
pub fn record_serveable(model_id: &[u8; 32]) {
    if let Ok(mut m) = self_test_state().write() { m.insert(*model_id, true); }
}

/// The honest declare set: staged models that have PROVEN they can serve inference on this rig.
/// `declare_capabilities` announces THIS (not raw `loaded_model_ids`), so the pool never routes us a
/// request for a tier we cannot answer — and we never mine a tier we cannot serve (the walk install
/// gates on the same `model_serveable`).
///
/// The self-test gate is wired in the CUDA walk bring-up (`pom_gpu::ensure_installed_inner`). The
/// AMD/OpenCL (CPU/vk inference) and macOS/Metal builds don't run it yet, so they keep the prior
/// declaration semantics — gating them here would stop them declaring anything.
pub fn serveable_model_ids() -> Vec<[u8; 32]> {
    #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
    { loaded_model_ids().into_iter().filter(|m| model_serveable(m)).collect() }
    #[cfg(not(all(feature = "pom-cuda", not(feature = "pom-opencl"))))]
    { loaded_model_ids() }
}

/// Forget a model's self-test result so it re-probes (used after a transient GPU-fault reset that
/// rebuilds the card's engine — the prior pass no longer necessarily holds).
pub fn clear_self_test(model_id: &[u8; 32]) {
    if let Ok(mut m) = self_test_state().write() { m.remove(model_id); }
}

/// Run ONE tiny inference on `gpu` to prove `model_id` can actually serve OPoI before we declare or
/// mine its tier. Caches the outcome. PASS → `mark_model_available` (declarable + mineable); FAIL →
/// `mark_model_unavailable` (withdrawn from ai:cap; the mining gate then refuses to grind a tier we
/// cannot serve). Budgeted by nature: a warm short generation is ~1-2 s; a model that needs far
/// longer would miss the on-chain service window anyway, so slow/empty == not serveable.
pub fn run_inference_self_test(model_id: &[u8; 32], gpu: usize) -> bool {
    if let Ok(m) = self_test_state().read() {
        if let Some(&passed) = m.get(model_id) {
            return passed && !model_is_unavailable(model_id);
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
         declare/mine its tier (the network exists to serve inference).", name, gpu
    );
    let t0 = std::time::Instant::now();
    let out = load_and_run_inference_on(gpu, model_id, "Reply with exactly: OK", SELF_TEST_MAX_TOKENS);
    let secs = t0.elapsed().as_secs_f64();
    let mut passed = matches!(&out, Some(t) if !t.trim().is_empty());
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
        let candidates: Vec<usize> = crate::pom_gpu::walk_devices()
            .into_iter().map(|d| d as usize).filter(|&d| d != gpu).collect();
        #[cfg(not(any(feature = "pom-cuda", all(target_os = "macos", feature = "pom-metal"))))]
        let candidates: Vec<usize> = Vec::new();
        for alt in candidates {
            log::warn!(
                "OPoI self-test: '{}' failed on GPU {} — FAILING OVER: retrying the probe on GPU {}                  (a healthy alternate host keeps the rig mining).", name, gpu, alt
            );
            let t1 = std::time::Instant::now();
            let out2 = load_and_run_inference_on(alt, model_id, "Reply with exactly: OK", SELF_TEST_MAX_TOKENS);
            if matches!(&out2, Some(t) if !t.trim().is_empty()) {
                INFERENCE_GPU_OVERRIDE.store(alt as isize, std::sync::atomic::Ordering::Relaxed);
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
    if let Ok(mut m) = self_test_state().write() { m.insert(*model_id, passed); }
    if passed {
        mark_model_available(model_id, "inference_self_test_passed");
        log::info!("OPoI self-test: '{}' PASSED in {:.1}s — serveable; its tier will be declared + mined.", name, secs);
    } else {
        mark_model_unavailable(model_id, "inference_self_test_failed");
        log::warn!(
            "OPoI self-test: '{}' FAILED after {:.1}s (no/empty output) — NOT serveable. Withdrawing it \
             from ai:cap and NOT mining this tier (mining a tier we cannot serve would strike the pool). \
             Cause is above: GPU inference could not load/run — usually the card lacks VRAM for this \
             model, an old driver, or a missing CUDA runtime / keryx-llama engine next to the miner.",
            name, secs
        );
    }
    passed
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
    specs.iter()
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
}

/// Re-announce a model after it serves again (upstream 0795e92).
pub fn mark_model_available(model_id: &[u8; 32], reason: &str) {
    if unavailable_models().write().unwrap().remove(model_id) {
        log::info!("SlmEngine: model {:.8} back in ai:cap ({})", hex::encode(model_id), reason);
    }
}

fn model_is_unavailable(model_id: &[u8; 32]) -> bool {
    unavailable_models().read().unwrap().contains(model_id)
}

pub fn loaded_model_ids() -> Vec<[u8; 32]> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let mut ids: Vec<[u8; 32]> = specs.iter()
        .filter(|s| model_dir(s).join(".ok").exists() && !model_is_unavailable(&s.model_id))
        .map(|s| s.model_id)
        .collect();
    // Per-card assignments (mixed rig / --force-model): add any assigned tier whose GGUF is staged
    // (its dir's `.ok` present) and not already listed. pom-cuda only — the map is empty elsewhere.
    #[cfg(feature = "pom-cuda")]
    for (mid, gguf) in crate::pom_gpu::assigned_models() {
        let ready = std::path::Path::new(&gguf)
            .parent()
            .map_or(false, |d| d.join(".ok").exists());
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
    specs.iter().map(|s| {
        let dir = model_dir(s);
        let gguf = dir.join("model.gguf");
        let ok = dir.join(".ok").exists();
        match crate::gguf::completeness_reason(&gguf) {
            None if ok => format!("  '{}': READY ({})", s.name, gguf.display()),
            None => format!(
                "  '{}': GGUF complete but not yet adopted at {} (miner will write .ok on next staging pass)",
                s.name, gguf.display()
            ),
            Some(reason) => format!("  '{}': NOT ready — {} [{}]", s.name, gguf.display(), reason),
        }
    }).collect()
}

/// True only when the model is supported, its files are completely downloaded, and it is not
/// currently withdrawn from `ai:cap` (upstream 0795e92).
pub fn is_model_ready(model_id: &[u8; 32]) -> bool {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };
    model_dir(spec).join(".ok").exists() && !model_is_unavailable(model_id)
}

/// Load the requested model on demand (evicting a cached different model if needed), then run
/// inference. Blocking — call from `spawn_blocking`. Routes to the best FREE eligible card via the
/// policy router (single-GPU / no-CUDA collapses to the one card). Callers that need to reply "busy"
/// or pause PoW BEFORE dispatch should instead `acquire_inference_card(..)` themselves and call
/// `load_and_run_inference_on(lease.gpu(), ..)` (see the stratum handlers).
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    // Default deadline for callers that don't pass one (grpc solo path, capability challenge):
    // wait up to 30s for a free card, else give up (None → skipped, no hard reject upstream).
    let lease = acquire_inference_card(model_id, 30_000)?;
    load_and_run_inference_on(lease.gpu(), model_id, prompt, max_tokens)
    // lease drops here → card released on every exit path.
}

/// As `load_and_run_inference`, but runs on the caller-leased CUDA card `gpu` (no migration). The
/// in-process llama.cpp engine branch is card-aware (its own resident model per card + per-card
/// generate); the candle/vk fallbacks remain the single-card dormant path.
pub fn load_and_run_inference_on(gpu: usize, model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    let _ = gpu; // the leased card is used by the in-process llama branch (CUDA/Metal); the
                 // vk/candle fallbacks below are single-card by design and ignore it.
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;

    // Race guard: never attempt inference (or a card model-swap that uninstalls the walk) for a
    // model whose GGUF is not fully on disk yet. The initial background prefetch may still be
    // downloading/loading it; serving now would uninstall+reload mid-prefetch. Drop the request —
    // the pool re-challenges once the model is ready (loaded_model_ids / ai:cap track readiness).
    if !crate::gguf::is_complete_file(&gguf_path_for(spec)) {
        log::warn!("OPoI: model '{}' not fully staged yet — skipping inference (avoids a load race); will serve once ready.", spec.name);
        return None;
    }

    // Prefer a running llama.cpp llama-server: AMD always (candle has no AMD-GPU backend; Vulkan
    // server), NVIDIA when a CUDA llama-server is bundled/env-pointed (Phase 1 of candle-
    // independence — llama.cpp tracks new GGUF archs faster than candle). The OPoI text is
    // user-facing only (consensus checks the fixed-point `model_fixed` commitment separately), so
    // a non-candle engine is fine. Falls through to the candle engine below when the server isn't
    // available — on NVIDIA that is the exact pre-Phase-1 behavior (candle-GPU → candle-CPU).
    // Highest priority: the IN-PROCESS llama.cpp engine (Phase 2 on CUDA, Phase 3b on Apple
    // Silicon Metal). Ranks above the candle path so any host that bundles the .so/.dylib
    // gets fully candle-independent inference.
    #[cfg(any(
        all(feature = "pom-cuda", not(feature = "pom-opencl")),
        all(target_os = "macos", feature = "pom-metal"),
    ))]
    {
        let gguf = gguf_path_for(spec).to_string_lossy().into_owned();
        // If this card doesn't already host the requested model, free ITS walk (per-card — other
        // cards keep mining) so the inference model fits, then load it on this card. When the model
        // is ALREADY resident here (the zero-dup mining tier) this is a no-op: no uninstall, no
        // reload — byte-identical to the pre-router single-card path.
        if !crate::llama_engine::active_for(&gguf, gpu) {
            // Hard inference gate (upstream d35f85fc, adapted): pause + drain this process's PoM
            // walks so uninstall() frees the card's walk WITHOUT racing a live batch, then reload
            // the inference model and resume. A guard clears the pause on every exit (incl. panic).
            struct PauseGuard;
            impl Drop for PauseGuard {
                fn drop(&mut self) { crate::pom_gpu::set_inference_paused(false); }
            }
            crate::pom_gpu::set_inference_paused(true);
            let _resume = PauseGuard;
            crate::pom_gpu::uninstall(gpu as u32);
            crate::llama_engine::ensure_loaded_on(&gguf, gpu);
        }
        if crate::llama_engine::available_on(gpu) {
            let t0 = std::time::Instant::now();
            if let Some(text) = crate::llama_engine::generate_on(gpu, prompt, max_tokens) {
                let secs = t0.elapsed().as_secs_f64();
                if secs > 0.0 {
                    record_card_toks(gpu, text.split_whitespace().count() as f64 / secs);
                }
                // Upstream 0795e92: a successful generation re-announces the model in ai:cap.
                mark_model_available(model_id, "generation_success");
                record_serveable(model_id); // a live answer is proof this rig can serve the tier
                return Some(strip_think_tags(&text));
            }
            log::warn!("SlmEngine: in-process llama generate failed on GPU {} — trying the next engine.", gpu);
        } else {
            // Upstream 0795e92: this card is the only one that can serve the model and its engine
            // failed to come up — withdraw it from ai:cap so the pool stops routing us requests.
            mark_model_unavailable(model_id, "llama_load_failed");
        }
    }

    // AMD (pom-opencl) in-process engine: the zero-dup host of the walk's model — its inference
    // costs no extra VRAM. Ranks above the llama-server subprocess.
    #[cfg(feature = "pom-opencl")]
    if crate::llama_engine_vk::available() {
        if let Some(text) = crate::llama_engine_vk::generate(prompt, max_tokens) {
            return Some(strip_think_tags(&text));
        }
        log::warn!("SlmEngine: in-process llama-vk generate failed — trying the next engine.");
    }

    #[cfg(any(feature = "pom-opencl", feature = "pom-cuda"))]
    if crate::llama_vulkan::available() {
        if let Some(text) = crate::llama_vulkan::generate(prompt, max_tokens) {
            return Some(strip_think_tags(&text));
        }
        log::warn!("SlmEngine: llama-server inference returned nothing — falling back to candle for this challenge.");
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
}
