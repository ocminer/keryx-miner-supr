//! H15 traced private answers and their openings on supr's CUDA llama engine (0.15.3).
//!
//! Port of the answer-path half of upstream keryx-miner v0.5.8 `llm.rs` (`run_inference` with a
//! trace, `arm_trace`, `chain_self_check`, `open_trace`, `open_attention_phase`, `open_pieces`,
//! `open_chain`, `open_attention`, `open_recurrent`, `replay_captures`, `replay_spans`,
//! `format_prompt_parts`), re-homed on supr's per-card engine pool: every function here takes the
//! exact `(gguf, gpu)` the caller leased and drained (see `slm::try_load_and_run_traced_inference_on`
//! and `slm::try_open_trace_on`), so it never moves a model or touches another card.
//!
//! Only the in-process CUDA engine can trace (the canonical leaves and sequence leaves are
//! ggml-cuda kernels added by tools/keryx-llama/patches). Every other backend gets the stub module
//! in lib.rs and refuses private requests: an untraced private answer is invalid past
//! gate + SERVICE_LEDGER_HORIZON_DAA (`ResponseWithoutTrace`) and would only earn a strike.

use crate::models::{ModelSpec, SYSTEM_PROMPT_NEXT};
use crate::trace::{ResponseTrace, TierManifest};

/// The chat template of an audited model split around the request-bound prefix: the head the
/// prefix tokens follow, and the rest after them. `None` for a model without a prefix slot.
/// VERBATIM upstream v0.5.8: the head must tokenize to exactly the tier's committed prefix offset.
pub fn format_prompt_parts(name: &str, prompt: &str) -> Option<(String, String)> {
    Some(match name {
        "glm-4-9b-0414" => (
            "[gMASK]<sop><|user|>\nRequest id:".to_string(),
            format!("\n{}\n\n{}\n<|assistant|>\n", SYSTEM_PROMPT_NEXT, prompt),
        ),
        "qwen3.8-27b" | "qwen3.6-27b" | "qwen3.5-9b-abliterated" => (
            "<|im_start|>system\nRequest id:".to_string(),
            format!(
                "\n{}<|im_end|>\n\
                 <|im_start|>user\n{}<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n\n</think>\n\n",
                SYSTEM_PROMPT_NEXT, prompt
            ),
        ),
        "gemma-4-12b-abliterated" => (
            "<|turn>system\nRequest id:".to_string(),
            format!(
                "\n{}<turn|>\n\
                 <|turn>user\n{}<turn|>\n\
                 <|turn>model\n<|channel>thought\n<channel|>",
                SYSTEM_PROMPT_NEXT, prompt
            ),
        ),
        "kimi-linear-48b" => (
            "<|im_system|>system<|im_middle|>Request id:".to_string(),
            format!(
                "\n{}<|im_end|>\
                 <|im_user|>user<|im_middle|>{}<|im_end|>\
                 <|im_assistant|>assistant<|im_middle|>",
                SYSTEM_PROMPT_NEXT, prompt
            ),
        ),
        _ => return None,
    })
}

/// Set once a traced self-check (traced generation → TraceCommitment → openings verified by the
/// node's own verifiers) passed on this rig: the `trace` capability announced to pools.
static SELF_TEST_PASSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether this rig proved it can answer an H15 private request (see [`SELF_TEST_PASSED`]).
pub fn self_test_passed() -> bool {
    SELF_TEST_PASSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Engine ABI of the native trace engine this build expects (llama_engine::ABI).
pub const ENGINE_ABI: u32 = 4;

/// BLAKE3 over the llama.cpp patch series the engine is built with (0001–0005, in order): two rigs
/// reproduce each other's traces only with the same engine arithmetic, so a replaying rig must match it.
pub fn patch_set_hash() -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for p in [
        &include_bytes!("../tools/keryx-llama/patches/0001-canonical-trace.patch")[..],
        &include_bytes!("../tools/keryx-llama/patches/0002-sequence-leaves.patch")[..],
        &include_bytes!("../tools/keryx-llama/patches/0003-fattn-tile-f32.patch")[..],
        &include_bytes!("../tools/keryx-llama/patches/0004-rope-f64-angle.patch")[..],
        &include_bytes!("../tools/keryx-llama/patches/0005-attn-lse-capture.patch")[..],
    ] {
        h.update(&(p.len() as u64).to_le_bytes());
        h.update(p);
    }
    *h.finalize().as_bytes()
}

/// Header of a `KRXREC01` compact trace record (the pool-custody format agreed in the pool exchange):
/// `"KRXREC01" | u32 LE header_len | borsh(RecordHeader) | borsh(ResponseTrace) | blake3(header || body)`.
#[derive(Debug, Clone, PartialEq, borsh::BorshSerialize, borsh::BorshDeserialize)]
pub struct RecordHeader {
    pub engine_abi: u32,
    pub patch_set: [u8; 32],
    pub model_id: [u8; 32],
    pub tier: u8,
    pub request_hash: [u8; 32],
    pub max_tokens: u32,
    /// The card that generated the trace (a replay must run on the same GPU model).
    pub gpu_model: String,
    pub sm_count: u32,
    pub driver_version: String,
}

pub const RECORD_MAGIC: &[u8; 8] = b"KRXREC01";

/// Serializes a `KRXREC01` record.
pub fn encode_record(header: &RecordHeader, trace: &ResponseTrace) -> Vec<u8> {
    let h = borsh::to_vec(header).expect("borsh header");
    let b = borsh::to_vec(trace).expect("borsh trace");
    let mut hasher = blake3::Hasher::new();
    hasher.update(&h);
    hasher.update(&b);
    let mut out = Vec::with_capacity(8 + 4 + h.len() + b.len() + 32);
    out.extend_from_slice(RECORD_MAGIC);
    out.extend_from_slice(&(h.len() as u32).to_le_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(&b);
    out.extend_from_slice(hasher.finalize().as_bytes());
    out
}

/// Parses and integrity-checks a `KRXREC01` record.
pub fn decode_record(blob: &[u8]) -> Option<(RecordHeader, ResponseTrace)> {
    use borsh::BorshDeserialize;
    if blob.len() < 8 + 4 + 32 || &blob[..8] != RECORD_MAGIC {
        return None;
    }
    let hl = u32::from_le_bytes(blob[8..12].try_into().ok()?) as usize;
    let (hb, rest) = blob.get(12..)?.split_at_checked(hl)?;
    let (bb, digest) = rest.split_at_checked(rest.len().checked_sub(32)?)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(hb);
    hasher.update(bb);
    if hasher.finalize().as_bytes() != digest {
        return None;
    }
    Some((RecordHeader::try_from_slice(hb).ok()?, ResponseTrace::try_from_slice(bb).ok()?))
}

/// GPU model name, SM count and driver version of CUDA ordinal `gpu` (empty/0 when unknown).
pub fn gpu_identity(gpu: usize) -> (String, u32, String) {
    #[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl"), not(target_os = "macos")))]
    {
        use candle_core::cuda_backend::cudarc::driver::{result, sys};
        let sm = (|| {
            result::init().ok()?;
            let dev = result::device::get(gpu as i32).ok()?;
            let n = unsafe { result::device::get_attribute(dev, sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) }.ok()?;
            Some(n.max(0) as u32)
        })()
        .unwrap_or(0);
        // The name comes from the CUDA driver for this CUDA ordinal (it honours CUDA_VISIBLE_DEVICES;
        // an NVML index lookup reported the wrong card on a UUID-pinned process).
        let name = (|| {
            result::init().ok()?;
            let dev = result::device::get(gpu as i32).ok()?;
            result::device::get_name(dev).ok()
        })()
        .unwrap_or_default();
        let driver = nvml_wrapper::Nvml::init().ok().and_then(|n| n.sys_driver_version().ok()).unwrap_or_default();
        (name, sm, driver)
    }
    #[cfg(not(all(feature = "pom-cuda", not(feature = "pom-opencl"), not(target_os = "macos"))))]
    {
        let _ = gpu;
        (String::new(), 0, String::new())
    }
}

/// What a traced inference commits: tier, plaintext prompt length, request id and stage.
#[derive(Clone, Copy, Debug)]
pub struct Traced {
    pub tier: u8,
    pub prompt_len: u32,
    pub request_hash: [u8; 32],
    /// Stage-2 (chain) trace.
    pub chain: bool,
    /// Sequence leaves and the attention rule's manifest (with `chain`).
    pub attention: bool,
}

/// A generated answer and, when the tier is audited, the trace an opening needs.
pub struct Generated {
    pub text: String,
    pub trace: Option<ResponseTrace>,
}

/// The private body of a traced answer: the answer text with its generated token ids
/// (`answer::encode`) for a stage-2 trace, the plain text otherwise. Sealed by the caller.
pub fn answer_body(text: &str, trace: Option<&ResponseTrace>) -> Vec<u8> {
    match trace.filter(|t| t.record.stage == crate::trace::STAGE_CHAIN) {
        Some(t) => {
            let ids: Vec<u32> = t.tokens[t.record.n_prompt as usize..].iter().map(|&v| v as u32).collect();
            keryx_verifiable_inference::answer::encode(text.as_bytes(), &ids)
        }
        None => text.as_bytes().to_vec(),
    }
}

/// Cuts `text` to the longest answer the request's token budget allows, on a char boundary.
pub fn truncate_answer(mut text: String, max_tokens: usize) -> String {
    let cap = keryx_verifiable_inference::shape::max_answer_len(max_tokens.min(u32::MAX as usize) as u32).min(usize::MAX as u64)
        as usize;
    if text.len() > cap {
        let mut end = cap;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

#[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
pub use engine::*;

#[cfg(all(feature = "pom-cuda", not(feature = "pom-opencl")))]
mod engine {
    use super::*;
    use crate::llama_engine::{self, GenError, PromptPrefix};
    use std::collections::{BTreeSet, HashMap};
    use std::sync::{Arc, Mutex, OnceLock};

    /// Bytes the seeded envelope adds to an opening body.
    const SEEDED_OVERHEAD: usize = 33;

    /// Whether this build can trace private answers at all (CUDA in-process engine).
    pub fn supported() -> bool {
        true
    }

    /// Self-check verdicts by `(gguf, gpu, stage key)`: `Ok` passed, `Err` failed (sticky).
    fn checks() -> &'static Mutex<HashMap<(String, usize, u8), Result<(), String>>> {
        static C: OnceLock<Mutex<HashMap<(String, usize, u8), Result<(), String>>>> = OnceLock::new();
        C.get_or_init(Default::default)
    }

    /// Possession indices built from the GGUF for models that are not the card's mining model.
    fn built_indices() -> &'static Mutex<HashMap<[u8; 32], Arc<crate::pom::WeightIndex>>> {
        static I: OnceLock<Mutex<HashMap<[u8; 32], Arc<crate::pom::WeightIndex>>>> = OnceLock::new();
        I.get_or_init(Default::default)
    }

    /// The possession index of `spec`: the mining index of `gpu` when it is this model, else one
    /// built from the GGUF once and kept.
    enum IndexRef {
        Device(Arc<(crate::pom::WeightIndex, u8)>),
        Built(Arc<crate::pom::WeightIndex>),
    }
    impl IndexRef {
        fn get(&self) -> &crate::pom::WeightIndex {
            match self {
                IndexRef::Device(a) => &a.0,
                IndexRef::Built(a) => a,
            }
        }
    }

    fn weight_index(spec: &ModelSpec, gguf: &str, gpu: usize) -> Option<IndexRef> {
        if crate::pom::mining_model_id(gpu as u32) == Some(spec.model_id) {
            if let Some(idx) = crate::pom::active_index_for(gpu as u32) {
                return Some(IndexRef::Device(idx));
            }
        }
        if let Some(i) = built_indices().lock().ok().and_then(|m| m.get(&spec.model_id).cloned()) {
            return Some(IndexRef::Built(i));
        }
        log::info!("Trace '{}': building the possession index from the GGUF for openings (one-off)", spec.name);
        match crate::pom::build_index_preserving_mining_id(gguf, spec.model_id) {
            Ok(index) => {
                let a = Arc::new(index);
                if let Ok(mut m) = built_indices().lock() {
                    m.insert(spec.model_id, Arc::clone(&a));
                }
                Some(IndexRef::Built(a))
            }
            Err(e) => {
                log::warn!("Trace '{}': no possession index for the opening ({})", spec.name, e);
                None
            }
        }
    }

    /// Replays a response once for the `wanted` `(position, leaf)` captures, checked against its trace.
    fn replay_captures(
        gguf: &str,
        gpu: usize,
        m: &TierManifest,
        trace: &ResponseTrace,
        wanted: &[(u32, u32)],
    ) -> Result<crate::trace::Captures, String> {
        use crate::trace::{captures_match, leaf_cap, Captures};
        let rec = &trace.record;
        let slot = wanted.iter().map(|&(_, l)| leaf_cap(m, l)).max().unwrap_or(0).div_ceil(4) * 4;
        let bytes = llama_engine::replay_multi(gguf, gpu, &trace.tokens, rec.n_prompt, wanted, slot)
            .map_err(|e| format!("{e} for captures {wanted:?}"))?;
        let caps: Captures = wanted.iter().copied().zip(bytes).collect();
        if !captures_match(rec, &caps) {
            return Err("the replay does not reproduce the committed trace".to_string());
        }
        Ok(caps)
    }

    /// Replays a response for the sequence spans of `plans`.
    fn replay_spans(
        gguf: &str,
        gpu: usize,
        trace: &ResponseTrace,
        plans: &[crate::trace_seq::SpanPlan],
    ) -> Result<Vec<crate::trace::SeqCapture>, String> {
        let spans: Vec<(u32, u64, u64)> = plans.iter().map(|p| p.capture()).collect();
        llama_engine::seq_replay(gguf, gpu, &trace.tokens, trace.record.n_prompt, &spans, crate::trace_seq::max_chunks(plans))
    }

    /// The attention opening of the draw of `seed` at the phase of `path`, `prev` the partials of
    /// the previous phase, checked as the node checks it.
    #[allow(clippy::too_many_arguments)]
    fn open_attention(
        gguf: &str,
        gpu: usize,
        m: &TierManifest,
        trace: &ResponseTrace,
        seed: &[u8; 32],
        response_txid: [u8; 32],
        index: &crate::pom::WeightIndex,
        path: &[u64],
        prev: Option<&[u8]>,
    ) -> Result<Vec<u8>, String> {
        use crate::trace_seq::{attn_opening, draw_attn, seq_response_ctx};
        use keryx_verifiable_inference::attn_blocks::{encode_partials, partials_root, pick_partials};
        use keryx_verifiable_inference::chain::{attn_query, verify_attn_phase};
        let rec = &trace.record;
        let seq = trace.seq.as_ref().ok_or("no sequence leaves")?;
        let d = draw_attn(m, rec, seed).ok_or("no attention draw")?;
        let caps = replay_captures(gguf, gpu, m, trace, &d.captures(path).ok_or("no attention captures")?)?;
        let plans = d.spans(m, rec.n_pos as u64, path).ok_or("no attention spans")?;
        let spans = replay_spans(gguf, gpu, trace, &plans)?;
        let full = attn_opening(m, rec, seq, index, &d, &caps, &spans, response_txid, path, prev)
            .map_err(|e| format!("{e} (slot {}, head {}, t {}, path {path:?})", d.slot, d.head, d.t))?;
        let ctx = m.chain.ok_or("no stage-2 context")?;
        let (tier, resp) = (m.tier_ctx(index), seq_response_ctx(rec, seq));
        let parent_root = prev.and_then(|b| partials_root(b, d.dv));
        let verify = |op: &keryx_verifiable_inference::opening::AttnOpening| {
            verify_attn_phase(&tier, &ctx, &resp, seed, path, parent_root.as_ref(), op)
        };
        let op = if d.leaf(path) {
            full
        } else {
            let (q, scale) = attn_query(&tier, &ctx, &resp, seed, &full).map_err(|e| format!("attention query {e:?}"))?;
            let (keys, values) = d.rows(&full, path).ok_or("no cache rows")?;
            let p0 = d.tree.node(path).ok_or("no node")?.0;
            let partials = pick_partials(&q, scale, &keys, &values, p0, &d.tree, path, |p| {
                verify(&d.phase_fields(&full, path, encode_partials(p))).is_ok()
            })
            .ok_or_else(|| format!("no partials verify at layer {}, position {}, path {path:?}", d.layer, d.t))?;
            d.phase_fields(&full, path, partials)
        };
        verify(&op).map_err(|e| format!("attention phase {e:?} at layer {}, position {}, path {path:?}", d.layer, d.t))?;
        let body = op.encode();
        if body.len() + SEEDED_OVERHEAD > keryx_verifiable_inference::opening::MAX_BODY_LEN {
            return Err(format!("attention opening of {} B over the cap", body.len()));
        }
        Ok(body)
    }

    /// The recurrent opening of the draw of `seed`, checked as the node checks it.
    #[allow(clippy::too_many_arguments)]
    fn open_recurrent(
        gguf: &str,
        gpu: usize,
        m: &TierManifest,
        trace: &ResponseTrace,
        seed: &[u8; 32],
        response_txid: [u8; 32],
        index: &crate::pom::WeightIndex,
    ) -> Result<Vec<u8>, String> {
        use crate::trace_seq::{draw_recur, recur_opening, seq_response_ctx};
        let rec = &trace.record;
        let seq = trace.seq.as_ref().ok_or("no sequence leaves")?;
        let d = draw_recur(m, rec, seed).ok_or("no recurrent draw")?;
        let caps = replay_captures(gguf, gpu, m, trace, &d.captures())?;
        let plans = d.spans(m, rec.n_pos as u64).ok_or("no recurrent spans")?;
        let spans = replay_spans(gguf, gpu, trace, &plans)?;
        let op = recur_opening(m, rec, seq, index, &d, &caps, &spans, response_txid)?;
        let ctx = m.chain.ok_or("no stage-2 context")?;
        keryx_verifiable_inference::recur::verify_recur_point(&m.tier_ctx(index), &ctx, &seq_response_ctx(rec, seq), seed, &op)
            .map_err(|e| format!("recurrent point {e:?} at layer {}, segment {}", d.layer, d.segment))?;
        let body = op.encode();
        if body.len() + SEEDED_OVERHEAD > keryx_verifiable_inference::opening::MAX_BODY_LEN {
            return Err(format!("recurrent opening of {} B over the cap", body.len()));
        }
        Ok(body)
    }

    /// Replays a stage-2 response once for every leaf the draw of `seed` opens and builds its trace
    /// and residual openings.
    #[allow(clippy::too_many_arguments)]
    fn open_chain(
        gguf: &str,
        gpu: usize,
        m: &TierManifest,
        pctx: &keryx_verifiable_inference::prefix::PrefixCtx,
        trace: &ResponseTrace,
        seed: &[u8; 32],
        request_hash: &[u8; 32],
        response_txid: [u8; 32],
        index: &crate::pom::WeightIndex,
    ) -> Result<(Vec<u8>, Vec<u8>), String> {
        use crate::trace::{chain_opening, draw_chain};
        use keryx_verifiable_inference::opening::MAX_BODY_LEN;
        let rec = &trace.record;
        let d = draw_chain(m, pctx, rec, seed).ok_or("no stage-2 draw")?;
        let caps = replay_captures(gguf, gpu, m, trace, &d.captures(m.n_products as u64))?;
        let (mut o, mut r) = chain_opening(m, pctx, rec, &trace.tokens, seed, request_hash, response_txid, &d, &caps, index)
            .ok_or_else(|| "no stage-2 opening".to_string())?;
        o.roots = trace.seq.as_ref().map(|seq| (rec.trace_root, seq.root));
        r.roots = o.roots;
        let resid = r.encode();
        let body = o.encode();
        for (name, b) in [("stage-2", &body), ("residual", &resid)] {
            if b.len() + SEEDED_OVERHEAD > MAX_BODY_LEN {
                return Err(format!("{name} opening of {} B over the cap", b.len()));
            }
        }
        let ctx = m.chain.ok_or("no stage-2 context")?;
        crate::trace::verify_chain_opening(&m.tier_ctx(index), &ctx, pctx, rec, seed, request_hash, &body, &resid)
            .map_err(|e| format!("stage-2 opening fails its own check: {e}"))?;
        Ok((body, resid))
    }

    /// Every opening of a stage-2 response: the trace opening, the attention and recurrent ones
    /// under the attention rule when the tier has such layers, then the residual one.
    #[allow(clippy::too_many_arguments)]
    fn open_pieces(
        gguf: &str,
        gpu: usize,
        m: &TierManifest,
        pctx: &keryx_verifiable_inference::prefix::PrefixCtx,
        trace: &ResponseTrace,
        seed: &[u8; 32],
        request_hash: &[u8; 32],
        response_txid: [u8; 32],
        index: &crate::pom::WeightIndex,
    ) -> Result<Vec<Vec<u8>>, String> {
        let (body, resid) = open_chain(gguf, gpu, m, pctx, trace, seed, request_hash, response_txid, index)?;
        let mut bodies = vec![body];
        if trace.seq.is_some() {
            bodies.push(open_attention(gguf, gpu, m, trace, seed, response_txid, index, &[], None)?);
            if m.chain.is_some_and(|c| c.n_recur > 0) {
                bodies.push(open_recurrent(gguf, gpu, m, trace, seed, response_txid, index)?);
            }
        }
        bodies.push(resid);
        Ok(bodies)
    }

    /// Traces a fixed prompt under the stage-2 manifest, then opens and verifies it under two
    /// seeds as the node would. Upstream v0.5.8 `chain_self_check`.
    fn chain_self_check(spec: &ModelSpec, gguf: &str, gpu: usize, tier: u8, attention: bool) -> Result<(), String> {
        use crate::trace::{chain_ctx, prefix_tokens, verify_chain_opening, SeqTrace, TraceRecord};
        let m = if attention { TierManifest::for_attn(tier) } else { TierManifest::for_chain(tier) }.ok_or("no stage-2 manifest")?;
        let (_, pctx) = chain_ctx(tier).ok_or("no stage-2 context")?;
        let ctx = m.chain.ok_or("no stage-2 context")?;
        let (head, rest) = format_prompt_parts(spec.name, "Reply with one word.").ok_or("no prefix template")?;
        let request_hash: [u8; 32] = std::array::from_fn(|i| (i * 8 + 3) as u8);
        let tokens = prefix_tokens(&pctx, &request_hash);
        let prefix = PromptPrefix { tokens: &tokens, head_tokens: pctx.offset, rest: &rest };
        let g = llama_engine::generate_traced(gguf, gpu, &head, Some(prefix), 4, 0, 0).map_err(|e| format!("{e:?}"))?;
        let (lo, hi) = (pctx.offset as usize, pctx.offset as usize + tokens.len());
        if g.tokens.get(lo..hi) != Some(tokens.as_slice()) {
            return Err("the fed prompt does not carry the request prefix".to_string());
        }
        let record = TraceRecord::new_chain(&m, g.n_prompt, g.ubatch, g.leaves, &g.tokens).ok_or("incomplete stage-2 trace")?;
        let seq = match (attention, g.seq) {
            (true, Some(leaves)) => Some(SeqTrace::new(leaves).ok_or("empty sequence leaves")?),
            (true, None) => return Err("no sequence leaves".to_string()),
            (false, _) => None,
        };
        let trace = ResponseTrace { record, tokens: g.tokens, seq };
        let index = weight_index(spec, gguf, gpu).ok_or("no possession index")?;
        let tier_ctx = m.tier_ctx(index.get());
        for seed in [[0x11u8; 32], [0x77u8; 32]] {
            let bodies = open_pieces(gguf, gpu, &m, &pctx, &trace, &seed, &request_hash, [0u8; 32], index.get())?;
            verify_chain_opening(
                &tier_ctx,
                &ctx,
                &pctx,
                &trace.record,
                &seed,
                &request_hash,
                &bodies[0],
                bodies.last().ok_or("no residual piece")?,
            )?;
        }
        Ok(())
    }

    /// Arms the canonical trace of `tier` (stage-2 + sequence leaves under the attention rule) on
    /// the engine hosting `spec` on `gpu`, and runs the self-check once per (engine, stage). A
    /// failed self-check is sticky: that card never answers a private request of this tier (an
    /// answer it cannot open would only earn strikes). Upstream exits the process; we keep mining.
    fn arm_trace(spec: &ModelSpec, gguf: &str, gpu: usize, tier: u8, chain: bool, attention: bool) -> Result<(), String> {
        if !chain {
            // Pre-chain (stage-1) tiers are not served by 0.15.3: mainnet and testnet activate the
            // chain rule together with the trace rule, so a live request is always stage-2.
            return Err("stage-1 traces are not supported by this build".to_string());
        }
        let manifest = if attention { TierManifest::raw_attn(tier) } else { TierManifest::raw_chain(tier) }
            .ok_or_else(|| format!("tier {tier} has no trace manifest"))?;
        let key = (gguf.to_string(), gpu, if attention { crate::trace::STAGE_CHAIN | 0x80 } else { crate::trace::STAGE_CHAIN });
        if let Some(Err(e)) = checks().lock().ok().and_then(|c| c.get(&key).cloned()) {
            return Err(format!("self-check failed earlier: {e}"));
        }
        if !llama_engine::traced_with(gguf, gpu, manifest) {
            llama_engine::trace_configure(gguf, gpu, manifest).map_err(|e| format!("trace unavailable ({e})"))?;
        }
        let seq = match (attention, TierManifest::for_attn(tier)) {
            (true, Some(m)) => {
                let c = m.chain.ok_or("no stage-2 context")?;
                llama_engine::seq_configure(gguf, gpu, &m.seq_layers(), c.n_attn, c.n_recur, c.seq_heads, m.seq_interval())
            }
            (true, None) => return Err(format!("tier {tier} has no attention manifest")),
            (false, _) => llama_engine::seq_configure(gguf, gpu, &[], 0, 0, 0, 0),
        };
        seq.map_err(|e| format!("sequence leaves unavailable ({e})"))?;
        let done = checks().lock().ok().and_then(|c| c.get(&key).cloned());
        if done.is_none() {
            let t0 = std::time::Instant::now();
            let verdict = chain_self_check(spec, gguf, gpu, tier, attention);
            match &verdict {
                Ok(()) => log::info!(
                    "Trace '{}': stage-2{} self-check passed on GPU {} ({:.1}s)",
                    spec.name,
                    if attention { " + attention" } else { "" },
                    gpu,
                    t0.elapsed().as_secs_f64()
                ),
                Err(e) => log::error!(
                    "Trace '{}': GPU {} FAILS the stage-2 self-check ({}) — this card will not answer private requests of tier {}",
                    spec.name,
                    gpu,
                    e,
                    tier
                ),
            }
            if let Ok(mut c) = checks().lock() {
                c.insert(key, verdict.clone());
            }
            verdict?;
        }
        Ok(())
    }

    /// Why a traced generation produced no answer.
    #[derive(Debug)]
    pub enum TracedError {
        /// The prompt does not fit this card's context: refuse this request only.
        PromptTooLong(String),
        /// This card cannot serve the tier traced (no trace, failed self-check, bad template).
        Unsupported(String),
        /// The generation itself failed.
        Failed(String),
    }

    /// Traced generation on the leased, drained card `gpu` hosting `gguf`. Upstream v0.5.8
    /// `run_inference` with `Some(Traced)`; never answers untraced.
    pub fn generate(spec: &ModelSpec, gguf: &str, gpu: usize, prompt: &str, max_tokens: usize, t: Traced) -> Result<Generated, TracedError> {
        let Traced { tier, prompt_len, request_hash, chain, attention } = t;
        let attention = chain && attention;
        arm_trace(spec, gguf, gpu, tier, chain, attention).map_err(TracedError::Unsupported)?;
        let m = if attention { TierManifest::for_attn(tier) } else { TierManifest::for_chain(tier) }
            .ok_or_else(|| TracedError::Unsupported(format!("tier {tier} has no manifest")))?;
        let min_prompt = keryx_verifiable_inference::shape::min_prompt_positions(prompt_len);
        let per_position = keryx_verifiable_inference::shape::ANSWER_BYTES_PER_POSITION;
        let (_, pctx) = crate::trace::chain_ctx(tier).ok_or_else(|| TracedError::Unsupported("no stage-2 prefix context".into()))?;
        let (head, rest) = format_prompt_parts(spec.name, prompt)
            .ok_or_else(|| TracedError::Unsupported(format!("'{}' has no prefix template", spec.name)))?;
        let tokens = crate::trace::prefix_tokens(&pctx, &request_hash);
        let prefix = PromptPrefix { tokens: &tokens, head_tokens: pctx.offset, rest: &rest };
        let g = match llama_engine::generate_traced(gguf, gpu, &head, Some(prefix), max_tokens, min_prompt, per_position) {
            Ok(g) => g,
            Err(GenError::PromptTooLong(d)) => return Err(TracedError::PromptTooLong(d)),
            Err(GenError::PromptHead(d)) => return Err(TracedError::Unsupported(format!("prompt head: {d}"))),
            Err(e) => return Err(TracedError::Failed(format!("{e:?}"))),
        };
        let (lo, hi) = (pctx.offset as usize, pctx.offset as usize + tokens.len());
        if g.tokens.get(lo..hi) != Some(tokens.as_slice()) {
            return Err(TracedError::Failed("the fed prompt does not carry the request prefix".to_string()));
        }
        let record = crate::trace::TraceRecord::new_chain(&m, g.n_prompt, g.ubatch, g.leaves, &g.tokens)
            .ok_or_else(|| TracedError::Failed("incomplete stage-2 trace".to_string()))?;
        let seq = if attention {
            Some(g.seq.and_then(crate::trace::SeqTrace::new).ok_or_else(|| TracedError::Failed("no sequence leaves".to_string()))?)
        } else {
            None
        };
        if g.text.trim().is_empty() {
            return Err(TracedError::Failed("empty answer".to_string()));
        }
        Ok(Generated { text: truncate_answer(g.text, max_tokens), trace: Some(ResponseTrace { record, tokens: g.tokens, seq }) })
    }

    /// The opening of an audited response (bundle of its pieces), built on the card `gpu` that
    /// hosts `gguf`. Upstream v0.5.8 `open_trace` (stage-2 branch).
    #[allow(clippy::too_many_arguments)]
    pub fn open_trace(
        spec: &ModelSpec,
        gguf: &str,
        gpu: usize,
        tier: u8,
        trace: &ResponseTrace,
        seed: &[u8; 32],
        response_txid: [u8; 32],
        request_hash: &[u8; 32],
    ) -> Result<Vec<u8>, String> {
        if trace.record.stage != crate::trace::STAGE_CHAIN {
            return Err("a stage-1 trace cannot be opened by this build".to_string());
        }
        let attention = trace.seq.is_some();
        arm_trace(spec, gguf, gpu, tier, true, attention)?;
        let index = weight_index(spec, gguf, gpu).ok_or("no possession index")?;
        let m = if attention { TierManifest::for_attn(tier) } else { TierManifest::for_chain(tier) }.ok_or("no stage-2 manifest")?;
        let (_, pctx) = crate::trace::chain_ctx(tier).ok_or("no stage-2 context")?;
        let bodies = open_pieces(gguf, gpu, &m, &pctx, trace, seed, request_hash, response_txid, index.get())?;
        Ok(crate::trace_seq::bundle(&bodies))
    }

    /// The attention phase of `path` of an audited response, `prev` the partials of its previous
    /// phase. Upstream v0.5.8 `open_attention_phase`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_attention_phase(
        spec: &ModelSpec,
        gguf: &str,
        gpu: usize,
        tier: u8,
        trace: &ResponseTrace,
        seed: &[u8; 32],
        response_txid: [u8; 32],
        path: &[u64],
        prev: &[u8],
    ) -> Result<Vec<u8>, String> {
        if trace.seq.is_none() {
            return Err("no sequence leaves".to_string());
        }
        arm_trace(spec, gguf, gpu, tier, true, true)?;
        let index = weight_index(spec, gguf, gpu).ok_or("no possession index")?;
        let m = TierManifest::for_attn(tier).ok_or("no attention manifest")?;
        open_attention(gguf, gpu, &m, trace, seed, response_txid, index.get(), path, Some(prev))
    }

    /// The H15 serving self-check of `tier` on the engine hosting `spec` on `gpu` (0.15.3 mining
    /// gate: a card that cannot answer a private request traced does not mine its tier).
    pub fn self_check(spec: &ModelSpec, gguf: &str, gpu: usize, tier: u8) -> Result<(), String> {
        arm_trace(spec, gguf, gpu, tier, true, true)?;
        super::SELF_TEST_PASSED.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Cross-card reproduction diagnostic (0.15.3, the pool's "option 3" question): `gen` traces a
    /// fixed request on `gpu` and writes the trace to `path`; `replay` traces the same request here,
    /// compares roots/tokens with the file's, then builds and verifies the openings of the FILE's
    /// trace on this card for three seeds (what a different rig would have to do for an audit).
    pub fn xcard(spec: &ModelSpec, gguf: &str, gpu: usize, tier: u8, mode: &str, path: &str) -> Result<String, String> {
        use crate::trace::{chain_ctx, verify_chain_opening};
        use borsh::BorshDeserialize;
        let t0 = std::time::Instant::now();
        arm_trace(spec, gguf, gpu, tier, true, true)?;
        let mut report = format!("model {} tier {} gpu {} — armed + self-check {:.1}s\n", spec.name, tier, gpu, t0.elapsed().as_secs_f64());
        let prompt = "Explain in two sentences why the sky is blue, then name one consequence for photography.";
        let request_hash: [u8; 32] = std::array::from_fn(|i| (i * 8 + 3) as u8);
        let t = Traced { tier, prompt_len: prompt.len() as u32, request_hash, chain: true, attention: true };
        let t1 = std::time::Instant::now();
        let g = generate(spec, gguf, gpu, prompt, 96, t).map_err(|e| format!("traced generation: {e:?}"))?;
        let gen_secs = t1.elapsed().as_secs_f64();
        let mine = g.trace.ok_or("no trace")?;
        report += &format!(
            "generated: {} prompt + {} generated positions in {:.2}s, root {}, seq root {}\n",
            mine.record.n_prompt,
            mine.record.n_pos - mine.record.n_prompt,
            gen_secs,
            hex::encode(mine.record.trace_root),
            mine.seq.as_ref().map(|q| hex::encode(q.root)).unwrap_or_default()
        );
        let target = match mode {
            "gen" => {
                let bytes = borsh::to_vec(&(g.text.clone(), &mine)).map_err(|e| e.to_string())?;
                std::fs::write(path, &bytes).map_err(|e| format!("write {path}: {e}"))?;
                report += &format!("wrote {} ({} B compact record)\n", path, bytes.len());
                mine
            }
            "replay" => {
                let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
                let (text, theirs): (String, ResponseTrace) = BorshDeserialize::try_from_slice(&bytes).map_err(|e| e.to_string())?;
                report += &format!(
                    "file trace: root {} | same tokens {} | same text {} | same position leaves {} | same seq leaves {} | same trace root {}\n",
                    hex::encode(theirs.record.trace_root),
                    theirs.tokens == mine.tokens,
                    text == g.text,
                    {
                        let n = theirs.record.n_pos.min(mine.record.n_pos) as u64;
                        match (0..n).find(|&t| theirs.record.leaves_at(t) != mine.record.leaves_at(t)) {
                            None if theirs.record.n_pos == mine.record.n_pos => "yes".to_string(),
                            None => "prefix only (lengths differ)".to_string(),
                            Some(t) => format!("NO (first difference at position {t})"),
                        }
                    },
                    theirs.seq.as_ref().map(|q| &q.leaves) == mine.seq.as_ref().map(|q| &q.leaves),
                    theirs.record.trace_root == mine.record.trace_root
                );
                theirs
            }
            other => return Err(format!("unknown mode {other}")),
        };
        let index = weight_index(spec, gguf, gpu).ok_or("no possession index")?;
        let m = TierManifest::for_attn(tier).ok_or("no attention manifest")?;
        let (_, pctx) = chain_ctx(tier).ok_or("no stage-2 context")?;
        let ctx = m.chain.ok_or("no stage-2 context")?;
        let tier_ctx = m.tier_ctx(index.get());
        let mut ok = 0;
        for k in 0..3u8 {
            let seed = *blake3::hash(&[k, 0x5e, 0xed]).as_bytes();
            let t2 = std::time::Instant::now();
            match open_pieces(gguf, gpu, &m, &pctx, &target, &seed, &request_hash, [k; 32], index.get()) {
                Ok(bodies) => {
                    let v = verify_chain_opening(&tier_ctx, &ctx, &pctx, &target.record, &seed, &request_hash, &bodies[0], bodies.last().unwrap());
                    let sizes: Vec<usize> = bodies.iter().map(Vec::len).collect();
                    report += &format!(
                        "seed {k}: opening built in {:.2}s ({:.1}x generation), pieces {:?} B, node verify {}\n",
                        t2.elapsed().as_secs_f64(),
                        t2.elapsed().as_secs_f64() / gen_secs.max(0.001),
                        sizes,
                        match &v {
                            Ok(()) => "OK".to_string(),
                            Err(e) => format!("FAIL {e}"),
                        }
                    );
                    if v.is_ok() {
                        ok += 1;
                    }
                }
                Err(e) => report += &format!("seed {k}: opening FAILED after {:.2}s: {e}\n", t2.elapsed().as_secs_f64()),
            }
        }
        report += &format!("RESULT {}/3 openings verified\n", ok);
        Ok(report)
    }

    /// Engines whose trace self-check failed, so `ai:cap` and the private-request scan can skip
    /// models this rig cannot answer traced.
    pub fn self_check_failed(gguf: &str, gpu: usize) -> bool {
        checks().lock().map_or(false, |c| c.iter().any(|((g, d, _), v)| g == gguf && *d == gpu && v.is_err()))
    }

    #[allow(dead_code)]
    fn _unused(_: BTreeSet<u8>) {}
}

/// Stub for builds without the CUDA in-process engine: private answers are refused.
#[cfg(not(all(feature = "pom-cuda", not(feature = "pom-opencl"))))]
mod engine_stub {
    pub fn supported() -> bool {
        false
    }
}
#[cfg(not(all(feature = "pom-cuda", not(feature = "pom-opencl"))))]
pub use engine_stub::*;

#[cfg(test)]
mod size_tests {
    use crate::trace::{leaf_cap, TierManifest};

    /// Witness size of a traced answer, per tier (attention manifests): what the miner keeps
    /// (hashes, tokens) versus what an opening-without-replay would need (every leaf's bytes at every
    /// position, plus the full sequence leaves). Printed for the pool's custody decision.
    #[test]
    #[ignore = "report, run with --ignored --nocapture"]
    fn trace_witness_sizes_per_tier() {
        let n_gen = 2048u64;
        for tier in 0..5u8 {
            let Some(m) = TierManifest::for_attn(tier) else { continue };
            let c = m.chain.unwrap();
            let n_leaves = m.n_leaves() as u64;
            // Product leaves: exact caps (X = q8 input, Y = f32 output). Residual/expert leaves past
            // them are hidden-state vectors: 4 * n_embd bytes (n_embd = the most common input width).
            let np = m.n_products as u64;
            let prod: u64 = (0..(2 * np) as u32).map(|l| leaf_cap(&m, l) as u64).sum();
            let mut ks: std::collections::BTreeMap<u64, u32> = Default::default();
            for (e, _) in m.entries.iter().take(np as usize) {
                *ks.entry(e.k).or_default() += 1;
            }
            let n_embd = ks.into_iter().max_by_key(|(_, c)| *c).map_or(0, |(k, _)| k);
            let per_pos = prod + n_leaves.saturating_sub(2 * np) * 4 * n_embd;
            for n_prompt in [64u64, 1024] {
                let n_pos = n_prompt + n_gen;
                let mut seq_bytes = 0u64;
                let mut n_seq = 0u64;
                for [_, kind, _, heads, hd] in m.seq_layers() {
                    let (heads, hd) = (heads as u64, hd as u64);
                    if kind == 1 || kind == 2 {
                        let row = hd / 32 * 34;
                        let kv = if kind == 2 { 1 } else { 2 };
                        seq_bytes += heads * kv * n_pos * row + heads * n_pos * 4 * 8; // + lse (q heads per kv head, upper bound 8)
                        n_seq += heads * 3;
                    } else {
                        let rl = keryx_verifiable_inference::seq::record_len(if kind == 4 { keryx_verifiable_inference::manifest::RECUR_KDA } else { 0 }, hd as usize) as u64;
                        seq_bytes += heads * (n_pos * rl + keryx_verifiable_inference::seq::checkpoint_len(hd, n_pos, m.seq_interval() as u64));
                        n_seq += heads * 2;
                    }
                }
                let compact = n_pos * n_leaves * 32 + n_pos * 4 + n_pos * 4 + n_seq * 32;
                let full = n_pos * per_pos + seq_bytes;
                eprintln!(
                    "tier {tier}: n_embd {n_embd} products {np} layers {} attn {} recur {} | n_prompt {n_prompt} + {n_gen} generated | leaves/pos {n_leaves} | \
                     leaf bytes/pos {:.2} MB | compact store {:.1} MB | full witness {:.1} GB (positions {:.1} GB + sequence {:.2} GB)",
                    c.n_layers,
                    c.n_attn,
                    c.n_recur,
                    per_pos as f64 / 1e6,
                    compact as f64 / 1e6,
                    full as f64 / 1e9,
                    (n_pos * per_pos) as f64 / 1e9,
                    seq_bytes as f64 / 1e9
                );
            }
        }
    }
}

#[cfg(test)]
mod record_tests {
    use super::*;

    #[test]
    fn krxrec01_round_trips_and_detects_corruption() {
        let leaves = vec![[7u8; 32]; 2 * 5];
        // A minimal hand-built trace in the borsh layout of `ResponseTrace` (no GPU needed).
        let header = RecordHeader {
            engine_abi: ENGINE_ABI,
            patch_set: patch_set_hash(),
            model_id: [1; 32],
            tier: 0,
            request_hash: [2; 32],
            max_tokens: 64,
            gpu_model: "NVIDIA GeForce RTX 5080".into(),
            sm_count: 84,
            driver_version: "615.71.09".into(),
        };
        let raw = {
            // TraceRecord{stage, n_prompt, n_pos, ubatch, n_leaves, leaves, pos_roots, trace_root, tokens_root}
            let mut v = Vec::new();
            v.push(2u8);
            v.extend_from_slice(&1u32.to_le_bytes());
            v.extend_from_slice(&2u32.to_le_bytes());
            v.extend_from_slice(&2u32.to_le_bytes());
            v.extend_from_slice(&1u32.to_le_bytes());
            v.extend_from_slice(&1u32.to_le_bytes());
            v.extend_from_slice(&5u64.to_le_bytes());
            v.extend_from_slice(&(leaves.len() as u32).to_le_bytes());
            for l in &leaves {
                v.extend_from_slice(l);
            }
            v.extend_from_slice(&2u32.to_le_bytes());
            v.extend_from_slice(&[3u8; 64]);
            v.extend_from_slice(&[4u8; 32]);
            v.push(0);
            // tokens, seq None
            v.extend_from_slice(&2u32.to_le_bytes());
            v.extend_from_slice(&5i32.to_le_bytes());
            v.extend_from_slice(&6i32.to_le_bytes());
            v.push(0);
            v
        };
        use borsh::BorshDeserialize;
        let trace = ResponseTrace::try_from_slice(&raw).expect("hand-built trace parses");
        let blob = encode_record(&header, &trace);
        assert_eq!(&blob[..8], b"KRXREC01");
        let (h2, t2) = decode_record(&blob).expect("round trip");
        assert_eq!(h2, header);
        assert_eq!(borsh::to_vec(&t2).unwrap(), raw);
        let mut bad = blob.clone();
        bad[20] ^= 1;
        assert!(decode_record(&bad).is_none(), "a flipped byte fails the blake3 check");
        assert_ne!(patch_set_hash(), [0u8; 32]);
    }
}
