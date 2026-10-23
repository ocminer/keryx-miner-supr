//! H15 block task on supr's engine: trace the window task on a device's resident model and open
//! it for a block candidate. Cherry-picked from upstream v0.5.8 `llm.rs` (`run_block_task`,
//! `open_block_task`, `open_chain`, `arm_trace`) and re-homed on supr's per-device registries:
//! the model a device mines comes from `pom_gpu::device_model`, its possession index from
//! `pom::active_index_for`, and the engine calls go through `llama_engine` (ABI 3 trace/replay).
//!
//! Only the pool-mining path is here. The self-responder (answering `mining.ai_request` with a
//! traced response, audits, openings of served responses) is NOT ported: the pool serves inference.

use crate::trace::{self, ResponseTrace, TierManifest, TraceRecord};
use keryx_verifiable_inference::block_task::{draw_seed, task_bytes, task_prefix, task_tokens, verify_block_task, TASK_LEN};
use keryx_verifiable_inference::prefix::PrefixCtx;

/// GGUF path of the model `device` mines, when it is `model_id`.
fn resident(device: u32, model_id: &[u8; 32]) -> Option<String> {
    let (id, gguf) = crate::pom_gpu::device_model(device)?;
    if id != *model_id || gguf.is_empty() {
        log::debug!("Block task: gpu{device} does not host the requested model");
        return None;
    }
    Some(gguf)
}

/// Arms the canonical (stage-2 / chain) trace of `tier` on the engine hosting `gguf` on `device`.
fn arm_trace(gguf: &str, device: u32, tier: u8) -> bool {
    let Some(manifest) = TierManifest::raw_chain(tier) else {
        log::warn!("Block task: tier {tier} has no chain manifest");
        return false;
    };
    if !crate::llama_engine::traced_with(gguf, device as usize, manifest) {
        if let Err(e) = crate::llama_engine::trace_configure(gguf, device as usize, manifest) {
            log::warn!("Block task: gpu{device} trace unavailable ({e})");
            return false;
        }
        log::info!("Block task: canonical trace armed on gpu{device} (tier {tier})");
    }
    true
}

/// The block task trace of `tier` for the window seeded by `window_seed`, on `device`'s engine
/// hosting `model_id`. Blocking (one traced 32-token pass). `None` when the device cannot trace.
pub fn run_block_task(device: u32, model_id: &[u8; 32], tier: u8, window_seed: &[u8; 32]) -> Option<ResponseTrace> {
    let gguf = resident(device, model_id)?;
    if !arm_trace(&gguf, device, tier) {
        return None;
    }
    let m = TierManifest::for_chain(tier)?;
    let (_, pctx) = trace::chain_ctx(tier)?;
    let tokens: Vec<i32> = task_tokens(&pctx.table, window_seed).into_iter().map(|t| t as i32).collect();
    let traced = match crate::llama_engine::trace_tokens(&gguf, device as usize, &tokens) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("Block task: gpu{device} task not traced ({e})");
            return None;
        }
    };
    let record = TraceRecord::new_chain(&m, TASK_LEN, traced.ubatch, traced.leaves, &tokens);
    if record.is_none() {
        log::warn!("Block task: gpu{device} incomplete task trace");
    }
    Some(ResponseTrace { record: record?, tokens })
}

/// The task opening of block `block_hash` from the task trace of its window, checked exactly as
/// the node checks it (`verify_block_task`) before it is shipped. Blocking (one replay).
pub fn open_block_task(
    device: u32,
    model_id: &[u8; 32],
    tier: u8,
    trace: &ResponseTrace,
    window_seed: &[u8; 32],
    block_hash: &[u8; 32],
) -> Option<Vec<u8>> {
    let gguf = resident(device, model_id)?;
    if !arm_trace(&gguf, device, tier) {
        return None;
    }
    let idx = crate::pom::active_index_for(device)?;
    let index = &idx.0;
    let m = TierManifest::for_chain(tier)?;
    let (ctx, pctx) = trace::chain_ctx(tier)?;
    let body = match open_chain(&gguf, device, &m, &task_prefix(&pctx), trace, &draw_seed(block_hash), &task_bytes(window_seed), [0u8; 32], index) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("Block task: gpu{device} opening failed ({e})");
            return None;
        }
    };
    let opening = keryx_verifiable_inference::opening::Opening::decode(&body)?;
    if let Err(e) = verify_block_task(&m.tier_ctx(index), &ctx, &pctx, &trace.record.trace_root, window_seed, block_hash, &opening) {
        log::error!("Block task: gpu{device} opening fails its self-check ({e:?}) — not submitting an invalid block");
        return None;
    }
    Some(body)
}

/// Replays the drawn captures of a chain-stage trace on the GPU that produced it and builds the
/// opening body (points + prefix point + resid point) against the possession index.
#[allow(clippy::too_many_arguments)]
fn open_chain(
    gguf: &str,
    device: u32,
    m: &TierManifest,
    pctx: &PrefixCtx,
    trace: &ResponseTrace,
    seed: &[u8; 32],
    request_hash: &[u8; 32],
    response_txid: [u8; 32],
    index: &crate::pom::WeightIndex,
) -> Result<Vec<u8>, String> {
    use crate::trace::{captures_match, chain_opening_body, draw_chain, leaf_cap, Captures};
    let rec = &trace.record;
    let d = draw_chain(m, pctx, rec, seed).ok_or("no stage-2 draw")?;
    let wanted = d.captures(m.n_products as u64);
    let slot = wanted.iter().map(|&(_, l)| leaf_cap(m, l)).max().unwrap_or(0).div_ceil(4) * 4;
    let bytes = crate::llama_engine::replay_multi(gguf, device as usize, &trace.tokens, rec.n_prompt, &wanted, slot)?;
    let caps: Captures = wanted.into_iter().zip(bytes).collect();
    if !captures_match(rec, &caps) {
        return Err("the replay does not reproduce the committed trace".to_string());
    }
    chain_opening_body(m, pctx, rec, &trace.tokens, seed, request_hash, response_txid, &d, &caps, index).ok_or_else(|| "no stage-2 opening".to_string())
}
