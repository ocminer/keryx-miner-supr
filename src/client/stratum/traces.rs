//! Traces of pool-dispatched H15 answers (0.15.3), kept so the pool can have them opened when the
//! node audits the response it submitted (`mining.ai_opening`). Keyed by the request hash the pool
//! sent: the pool's response txid is not known to the miner until the pool asks for an opening.
//! In memory plus an on-disk copy (`keryx-pool-traces.traces/`, or `$KERYX_POOL_TRACE_DIR`) so a
//! restarted miner can still open; dropped after `KEEP` (the node's audit span is ~7.5 h).

use keryx_miner::trace::ResponseTrace;
use keryx_miner::trace_store::{StoredTraceOwned, TraceStore};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long a pool trace is kept: the node's 270,000-DAA audit span (~7.5 h) plus slack.
const KEEP: Duration = Duration::from_secs(9 * 3600);

#[derive(Clone)]
pub(super) struct Kept {
    pub model_id: [u8; 32],
    pub tier: u8,
    pub request_hash: [u8; 32],
    pub gpu: Option<usize>,
    pub trace: Arc<ResponseTrace>,
    /// Unix seconds of the answer.
    pub at: u64,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn store() -> &'static TraceStore {
    static S: OnceLock<TraceStore> = OnceLock::new();
    S.get_or_init(|| {
        let base = std::env::var("KERYX_POOL_TRACE_DIR").unwrap_or_else(|_| "keryx-pool-traces".to_string());
        TraceStore::new(&base)
    })
}

fn kept() -> &'static Mutex<HashMap<String, Kept>> {
    static M: OnceLock<Mutex<HashMap<String, Kept>>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = HashMap::new();
        for t in store().load_all() {
            if t.submitted_daa + KEEP.as_secs() < now() {
                store().remove(&t.response_txid);
                continue;
            }
            m.insert(
                t.response_txid.clone(),
                Kept { model_id: t.model_id, tier: t.tier, request_hash: t.request_hash, gpu: None, trace: Arc::new(t.trace), at: t.submitted_daa },
            );
        }
        if !m.is_empty() {
            log::info!("Trace: restored {} pool answer trace(s) from {}", m.len(), store().dir().display());
        }
        Mutex::new(m)
    })
}

/// Keeps the trace of a pool-dispatched answer (memory + disk) for later openings.
pub(super) fn keep(request_hash_hex: &str, request_hash: [u8; 32], model_id: [u8; 32], tier: u8, max_tokens: u32, gpu: usize, trace: ResponseTrace) {
    let trace = Arc::new(trace);
    let at = now();
    let key = request_hash_hex.to_ascii_lowercase();
    store().save_queued(StoredTraceOwned {
        response_txid: key.clone(),
        model_id,
        tier,
        request_hash,
        max_tokens,
        // Unix seconds in the DAA slot: this store is not driven by the chain.
        submitted_daa: at,
        accepted_hash: None,
        opening: None,
        trace: Arc::clone(&trace),
        attn: None,
        anchor_hash: None,
        gen_secs: None,
    });
    let mut m = kept().lock().unwrap_or_else(|p| p.into_inner());
    m.retain(|k, v| {
        let live = v.at + KEEP.as_secs() >= at;
        if !live {
            store().remove(k);
        }
        live
    });
    m.insert(key, Kept { model_id, tier, request_hash, gpu: Some(gpu), trace, at });
}

/// The kept trace of the answer to `request_hash_hex`.
pub(super) fn get(request_hash_hex: &str) -> Option<Kept> {
    kept().lock().unwrap_or_else(|p| p.into_inner()).get(&request_hash_hex.to_ascii_lowercase()).cloned()
}
