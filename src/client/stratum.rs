use futures::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_util::codec::Framed;

mod statum_codec;

use crate::client::stratum::statum_codec::{ErrorCode, MiningNotify, MiningSubmit, NewLineJsonCodecError, StratumLine};
use crate::client::stratum::statum_codec::{
    InferenceRequestParams, InferenceResultParams, MiningSubscribe, SetExtranonce, StratumCommand, StratumError,
    StratumLinePayload, StratumResult,
};
use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::PartialBlock;
use crate::{miner::MinerManager, Error, Uint256};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use log::{error, info, warn};
use num::Float;
use rand::{thread_rng, RngCore};
use statum_codec::NewLineJsonCodec;
use std::sync::OnceLock;
use tokio::sync::mpsc::{self, Sender};
use tokio::sync::Mutex;
use tokio::sync::Notify;
use tokio::task;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_stream::wrappers::ReceiverStream;

//const DIFFICULTY_1_TARGET: Uint256 = Uint256([0x00000000ffff0000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000]);
const DIFFICULTY_1_TARGET: (u64, i16) = (0xffffu64, 208); // 0xffff 2^208
const KERYX_STRATUM_DAA_CAPABILITY: &str = "keryx-stratum-v2";
const LOG_RATE: Duration = Duration::from_secs(30);
const CHALLENGE_MAX_TOKENS: usize = 128;

// ── Phase 2 OPoI — inference cache & task types ─────────────────────────────

/// AiRequest task dispatched by the bridge in a `mining.notify` 5th parameter (JSON).
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
struct AiTask {
    #[serde(default)]
    stable_id: String,
    /// H6 pool request id — echoed back verbatim in `mining.submit` so the pool can match the
    /// unsigned AiResponse to the request it dispatched. When the pool omits `stable_id` (the H6
    /// wire), this doubles as the inference-cache dedup key (see `handle_ai_task`).
    #[serde(default, rename = "reqId")]
    req_id: String,
    #[serde(alias = "model_id")]
    model_id_hex: String,
    prompt: String,
    max_tokens: usize,
    #[serde(default)]
    inference_reward: u64,
    #[serde(default)]
    request_hash: String,
    /// H6 service-bond era: the AiResponse `challenge_window_end`. The POOL owns this on the pool
    /// path (it builds and signs the coinbase), so the miner does NOT choose it here — it is parsed
    /// only for completeness/telemetry. (On the SOLO grpc path the miner still derives daa+1000.)
    #[serde(default)]
    challenge_window_end: u64,
    /// H6: wall-clock budget the pool allots for the answer. Parsed for completeness; the miner
    /// does not currently enforce a hard deadline on the (uninterruptible) GPU inference.
    #[serde(default)]
    deadline_ms: u64,
}

/// Task attached to the current mining job, cleared on each new `mining.notify`.
struct CurrentTask {
    job_id: String,
    task: AiTask,
}

/// Shared inference result cache — persists across block changes so that if the
/// same AiRequest is included in multiple consecutive job templates the miner can
/// immediately submit with a CID once inference completed for the first occurrence.
/// A completed inference result: the base58 CID for the share submit plus the raw multihash
/// bytes and token count needed to reconstruct (and sign) the 78-byte v1 AiResponse message.
#[derive(Clone)]
struct InferenceResult {
    /// base58 CIDv0 string returned by IPFS — goes on the wire as the share's CID param.
    cid_b58: String,
    /// Token/word count — the `response_length` transmitted on the unsigned submit (the pool signs
    /// the V2 AiResponse over exactly this value, so it must be sent, not re-derived pool-side).
    response_length: u32,
}

struct InferenceCacheInner {
    /// stable_id → completed inference result (CID + fields for the V2 responder signature).
    results: HashMap<String, InferenceResult>,
    /// stable_ids currently being inferred (guards against duplicate spawn_blocking calls).
    in_progress: HashSet<String>,
}

type InferenceCache = Arc<Mutex<InferenceCacheInner>>;

type BlockHandle = JoinHandle<()>;

#[derive(Default)]
pub struct ShareStats {
    pub accepted: AtomicU64,
    pub stale: AtomicU64,
    pub low_diff: AtomicU64,
    pub duplicate: AtomicU64,
    pub shares_pending: Mutex<HashMap<u32, (String, u32)>>,
}

static SHARE_STATS: OnceLock<Arc<ShareStats>> = OnceLock::new();

/// Live share counters for the stats API (None until the pool client connects).
pub fn share_stats() -> Option<Arc<ShareStats>> {
    SHARE_STATS.get().cloned()
}

// ── Multi-pool failover (opt-in; ALL of this is inert unless a --backup-pool is configured) ──
// The failover monitor / failback prober (in main) set DESIRED_POOL; the reconnect loop stamps
// ACTIVE_POOL to the index it is about to serve. `listen()` polls a slow ticker and, if failover is
// enabled AND ACTIVE != DESIRED, returns cleanly so the loop reconnects to the new target. With no
// backup, FAILOVER_ENABLED stays false and this path is never taken → behaviour is unchanged.
static FAILOVER_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static ACTIVE_POOL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static DESIRED_POOL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_failover_enabled(on: bool) {
    FAILOVER_ENABLED.store(on, Ordering::Relaxed);
}
pub fn set_active_pool(i: usize) {
    ACTIVE_POOL.store(i, Ordering::Relaxed);
}
pub fn desired_pool() -> usize {
    DESIRED_POOL.load(Ordering::Relaxed)
}
pub fn set_desired_pool(i: usize) {
    DESIRED_POOL.store(i, Ordering::Relaxed);
}
/// True when the failover controller wants a different pool than the one `listen()` is serving.
fn pool_switch_requested() -> bool {
    FAILOVER_ENABLED.load(Ordering::Relaxed)
        && ACTIVE_POOL.load(Ordering::Relaxed) != DESIRED_POOL.load(Ordering::Relaxed)
}

impl Display for ShareStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Shares: {}{}{}{}Pending: {}",
            match self.accepted.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Accepted: {} ", v),
            },
            match self.stale.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Stale: {} ", v),
            },
            match self.low_diff.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Low difficulty: {} ", v),
            },
            match self.duplicate.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Duplicate: {} ", v),
            },
            // stats-string only: don't panic if the lock is momentarily held
            // (try_lock().unwrap() raced -> TryLockError -> tokio-worker panic).
            self.shares_pending.try_lock().map(|g| g.len()).unwrap_or(0)
        )
    }
}

/// Last real DAA score seen on a job that carries one (WithTask / ShortV2). The plain `Short`
/// notify variant carries NO daa_score; without this it pinned daa to the SALT-v4 era
/// (21,932,751) — which is BELOW the PoM activation (37,780,000), so a miner fed only `Short`
/// jobs never crosses `daa >= activation_daa()` and keeps hashing kHeavyHash post-fork (empty
/// PoM proof). Remembering the last real daa lets `Short` inherit it so PoM still activates.
static LAST_DAA_SCORE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Set when the pool authoritatively tells us the PoM hardfork is active (it rejected a share with
/// "pom proof required (hardfork active)"). This only happens if the pool feeds us plain `Short`
/// notifies (no daa_score) so our activation gate never fired and we submitted proofless kHeavyHash
/// shares. Trust the pool: once set, `Short` jobs floor daa at the current frontier
/// (VERY_LIGHT_ACTIVATION_DAA) so `daa >= activation_daa()` holds AND the post-H2 tier is stamped —
/// the next job builds a real PoM proof instead of looping on rejects forever.
static POOL_FORCED_POM: AtomicBool = AtomicBool::new(false);

/// The keryx H6 pool embeds the block DAA in the wire jobId as `<rand8>_<daa>` (e.g.
/// `32fc29f1_500000` → daa 500000). The plain `Short` `mining.notify` carries NO daa_score field, so
/// this trailing suffix is the AUTHORITATIVE per-job DAA on that pool — it must reach the PoM path so
/// the fork gates (H3/H4/H5/H6) and the seed/pow salts resolve exactly as the node does for the same
/// block. Returns None when the jobId has no `_<digits>` suffix (older pools); the caller then falls
/// back to the LAST_DAA_SCORE/POOL_FORCED_POM flooring.
fn job_id_daa(job_id: &str) -> Option<u64> {
    let (_, tail) = job_id.rsplit_once('_')?;
    if tail.is_empty() || !tail.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    tail.parse::<u64>().ok()
}

#[allow(dead_code)]
pub struct StratumHandler {
    log_handler: JoinHandle<()>,

    //client: Framed<TcpStream, NewLineJsonCodec>,
    send_channel: Sender<StratumLine>,
    stream: Pin<Box<dyn Stream<Item = Result<StratumLine, NewLineJsonCodecError>>>>,
    miner_address: String,
    pool_password: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,

    target_pool: Uint256,
    target_real: Uint256,
    nonce_mask: u64,
    nonce_fixed: u64,
    extranonce: Option<String>,
    last_stratum_id: Arc<AtomicU32>,

    shares_stats: Arc<ShareStats>,
    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,
    /// Signalled when the pool connection's WRITE side dies (the socket-sink
    /// forwarder ends, or a share submit fails). The read loop (`listen`) is
    /// otherwise blind to a half-open socket and would block forever on
    /// `try_next()` while every found share fails with "channel closed" — the
    /// "0 pool hashrate until restart" bug. `listen` selects on this to bail out
    /// and let main's reconnect loop re-establish + re-subscribe.
    conn_dead: Arc<Notify>,

    /// IPFS Kubo API URL for uploading inference results (e.g. "http://127.0.0.1:5001").
    ipfs_url: String,
    /// Task dispatched by the bridge for the current mining job (None = no AiRequest in job).
    current_task_slot: Arc<Mutex<Option<CurrentTask>>>,
    /// Completed inferences: stable_id → base58 CIDv0 string (persists across block changes).
    inference_cache: InferenceCache,
    /// Count of OPoI inferences currently in flight ACROSS ALL CARDS. Per-card busy-ness is tracked
    /// by the router (`slm::acquire_inference_card`); this counter only ref-counts the GLOBAL PoW
    /// pause so PoW resumes exactly when the LAST inference finishes (concurrent inferences on
    /// different cards each pause it; the first pauses, the last resumes). Replaces the former
    /// single `challenge_in_flight` bool that rejected any second concurrent inference.
    inference_inflight: Arc<std::sync::atomic::AtomicUsize>,

    /// Miner-telemetry (mining.hello/mining.telemetry, v0.7.0). Best-effort/non-fatal: starts true;
    /// flips false for the session if the pool rejects `mining.hello`/`mining.telemetry` with error
    /// 20 (method not supported) — after which we stop sending and just keep mining.
    telemetry_on: bool,
    /// Outstanding telemetry request ids (hello + each telemetry) awaiting a pool ack/err, so an
    /// error-20 reply can be attributed to telemetry (not mistaken for a rejected share) and a
    /// success ack doesn't log a spurious "ignoring result".
    telemetry_req_ids: HashSet<u32>,
    /// Process start, for the telemetry `uptime_s` field.
    start_time: std::time::Instant,
    /// The last model-id set announced to the pool via `mining.declare_capabilities`. Declare is
    /// re-sent whenever `loaded_model_ids()` differs from this (a model finishing its load, or
    /// `--tier auto` swapping in a bigger model), so the pool ALWAYS knows what a rig serves the
    /// moment it can serve it. Empty until the first successful declare. Fixes the race where a rig
    /// authorized before its model was ready declared `models:[]` and never re-announced.
    declared_model_ids: Vec<String>,
    /// When we last SENT a `mining.declare_capabilities` (regardless of change). The declaration is a
    /// fire-and-forget notification with no pool ACK, so a single one lost to a race (pool not yet
    /// ready to associate it with the worker) or a bridge/registry reset would leave the pool showing
    /// `declared=[]` forever — the rig then mines a tier it is never asked to serve → strikes. We
    /// therefore RE-declare the current serveable set periodically, not just on change, so the pool
    /// self-heals. `None` until the first declare.
    last_declare_at: Option<std::time::Instant>,
}

/// Ref-counted PoW-pause guard. On drop it decrements the in-flight-inference counter and, when it
/// held the LAST one, clears the miner's OPoI flag so PoW resumes on the next `mining.notify`. Drop
/// runs on normal completion AND on panic unwind inside the spawned blocking task, so a panicking
/// inference can never leave PoW paused forever (nor leak the count). Construct it AFTER the
/// `fetch_add` that decided whether this was the first (pausing) inference.
struct PowPauseGuard {
    inflight: Arc<std::sync::atomic::AtomicUsize>,
    miner_flag: Arc<AtomicBool>,
}
impl Drop for PowPauseGuard {
    fn drop(&mut self) {
        if self.inflight.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.miner_flag.store(false, Ordering::SeqCst);
        }
    }
}

#[async_trait(?Send)]
impl Client for StratumHandler {
    async fn register(&mut self) -> Result<(), Error> {
        let mut id = { Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst)) };
        self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Subscribe(
                    MiningSubscribe::MiningSubscribeOptions((
                        format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
                        KERYX_STRATUM_DAA_CAPABILITY.into(),
                    )),
                )),
                jsonrpc: None,
                error: None,
            })
            .await?;
        id = Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst));

        // Always authorize with the configured mining address. (The devfund
        // address-swap cycle was removed — see docs/devfund-removed.md.)
        let pay_address = self.miner_address.clone();
        self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Authorize((
                    pay_address.clone(),
                    self.pool_password.clone(),
                ))),
                jsonrpc: None,
                error: None,
            })
            .await?;

        // Declare loaded SLM models so the bridge can challenge with the right model. If no model is
        // ready yet (still loading/streaming into VRAM at authorize time), this is a no-op here and
        // the declare is re-attempted on every `mining.notify` — so the pool learns what this rig
        // serves the instant the model is ready, not just if it happened to be ready at authorize.
        self.declare_capabilities_if_changed().await?;

        // Miner telemetry (v0.7.0): STATIC rig identity, sent ONCE. Best-effort & non-fatal — if the
        // pool doesn't support it, it replies error 20 (handled in handle_message → telemetry_on=false
        // for the session); a send failure just skips telemetry. NEVER blocks/aborts mining.
        if self.telemetry_on {
            let loaded_model = keryx_miner::pom::active_index().map(|(_index, tier)| {
                let mid = keryx_miner::slm::loaded_model_ids().first().map(hex::encode).unwrap_or_default();
                (*tier, mid)
            });
            let hello = keryx_miner::telemetry::build_hello(loaded_model);
            let id = self.last_stratum_id.fetch_add(1, Ordering::SeqCst);
            self.telemetry_req_ids.insert(id);
            if self
                .send_channel
                .send(StratumLine {
                    id: Some(id),
                    payload: StratumLinePayload::StratumCommand(StratumCommand::MiningHello((hello,))),
                    jsonrpc: None,
                    error: None,
                })
                .await
                .is_err()
            {
                self.telemetry_on = false;
            }
        }
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        info!("Waiting for stuff");
        // A half-open pool socket (peer vanished without RST/FIN) leaves the read
        // half blocked on try_next() forever while the write half is dead — every
        // found share then fails to submit and pool hashrate reads 0 until a manual
        // restart. Two backstops force a reconnect:
        //   1. conn_dead — fired the instant the write half dies (sink forwarder
        //      ends, or a submit can't be forwarded). Immediate detection.
        //   2. IDLE_TIMEOUT — no pool message at all for this long. Covers the
        //      case where writes still buffer into a dead socket (no write error)
        //      but the peer never responds. A live keryx pool sends job/notify and
        //      ACKs our frequent share submits well inside this window.
        let idle_secs: u64 = std::env::var("KERYX_POOL_IDLE_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(120);
        let idle_timeout = Duration::from_secs(idle_secs);
        //   3. JOB watchdog — the idle_timeout above resets on ANY pool message (vardiff,
        //      keepalives, share ACKs), so a connection that keeps chattering but stops delivering
        //      block templates wedges the miner (GPUs idle, zero accepted shares) WITHOUT tripping
        //      it. Observed live: a rig ran ~15 h "alive" at 0 % GPU while the pool sent no jobs and
        //      the hashrate counter kept ticking. `block_template_ctr` advances on every MiningNotify;
        //      if it stops advancing for this long, force a reconnect. Env: KERYX_POOL_JOB_TIMEOUT.
        let job_secs: u64 = std::env::var("KERYX_POOL_JOB_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(300);
        let job_ctr = self.block_template_ctr.clone();
        let mut last_ctr = job_ctr.load(Ordering::SeqCst);
        let mut last_job_at = std::time::Instant::now();
        let mut job_watch = tokio::time::interval(Duration::from_secs(30));
        job_watch.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Failover switch check (opt-in): a slow ticker that returns cleanly if the failover
        // controller has picked a different pool. No-op when no backup is configured (inert flag).
        let mut switch_watch = tokio::time::interval(Duration::from_secs(3));
        switch_watch.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Telemetry sender (v0.7.0): periodic mining.telemetry. Interval from KERYX_TELEMETRY_INTERVAL
        // (default 120s, min 30s). The first tick fires immediately and is skipped (hello already
        // carried the static block; the first metrics go out one interval in). Inert while
        // telemetry_on is false (disabled, or the pool answered mining.hello with error 20).
        let mut telemetry_watch =
            tokio::time::interval(Duration::from_secs(keryx_miner::telemetry::interval_secs()));
        telemetry_watch.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        telemetry_watch.tick().await; // consume the immediate first tick
        let conn_dead = self.conn_dead.clone();
        loop {
            tokio::select! {
                biased;
                _ = conn_dead.notified() => {
                    return Err("pool connection write side died (submit failed) — reconnecting".into());
                }
                _ = switch_watch.tick() => {
                    if pool_switch_requested() {
                        return Err("failover: switching to a different pool".into());
                    }
                }
                _ = telemetry_watch.tick() => {
                    // Best-effort periodic metrics. Runs after the tick future resolves (the
                    // stream future is dropped by then), so borrowing &mut self here is fine —
                    // same pattern as handle_message. Never returns Err (telemetry is non-fatal).
                    if self.telemetry_on {
                        self.send_telemetry(miner).await;
                    }
                }
                _ = job_watch.tick() => {
                    let now_ctr = job_ctr.load(Ordering::SeqCst);
                    if now_ctr != last_ctr {
                        last_ctr = now_ctr;
                        last_job_at = std::time::Instant::now();
                    } else if last_job_at.elapsed() >= Duration::from_secs(job_secs) {
                        return Err(format!("no new job for {}s (pool connection delivering no work) — reconnecting", job_secs).into());
                    }
                }
                res = tokio::time::timeout(idle_timeout, self.stream.try_next()) => {
                    match res {
                        Err(_elapsed) => {
                            return Err(format!("no pool message for {}s (connection stalled) — reconnecting", idle_timeout.as_secs()).into());
                        }
                        Ok(Ok(Some(msg))) => self.handle_message(msg, miner).await?,
                        // try_next() == Ok(None) is a clean end-of-stream: the pool
                        // closed the TCP connection (sent FIN with no data pending).
                        // It is NOT a malformed/empty JSON payload — the old wording
                        // ("stratum message payload is empty") alarmed operators over
                        // what is a routine pool-initiated disconnect + reconnect.
                        Ok(Ok(None)) => return Err("pool closed the connection (EOF) — reconnecting".into()),
                        Ok(Err(e)) => return Err(e.into()),
                    }
                }
            }
        }
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

impl StratumHandler {
    /// Announce the currently-ready SLM models to the pool via `mining.declare_capabilities`, but
    /// ONLY when the ready set has changed since the last announcement. Called at authorize AND on
    /// every `mining.notify` — so the moment a model finishes loading (or `--tier auto` swaps in a
    /// bigger one) the pool is told, without spamming an identical declaration each job. The pool
    /// MUST know what a rig serves as soon as it can serve it: a rig that authorized before its
    /// model was ready used to declare `models:[]` once and never correct it. Best-effort/non-fatal.
    async fn declare_capabilities_if_changed(&mut self) -> Result<(), Error> {
        // Declare only PROVEN-serveable models (passed the inference self-test / answered live), never
        // aspirationally: the pool must not route us a request for a tier we cannot actually serve.
        let model_ids: Vec<String> = keryx_miner::slm::serveable_model_ids()
            .into_iter()
            .map(hex::encode)
            .collect();
        // Only announce a non-empty set (an empty declare tells the pool nothing useful and would
        // churn while a model is still loading). Once we HAVE declared, a later drop to empty is
        // left as-is until a real model is ready again.
        if model_ids.is_empty() {
            return Ok(());
        }
        // Re-declare when the set CHANGED, OR periodically even if unchanged. The declaration has no
        // pool ACK, so a single notification lost to a startup race or a pool-side registry reset
        // would otherwise leave `declared=[]` permanently — the rig mines a tier it is never asked to
        // serve, accruing inference strikes. Re-sending every REDECLARE_EVERY makes the pool
        // self-heal without spamming a declaration on every job.
        const REDECLARE_EVERY: std::time::Duration = std::time::Duration::from_secs(90);
        let stale = self.last_declare_at.map_or(true, |t| t.elapsed() >= REDECLARE_EVERY);
        if model_ids == self.declared_model_ids && !stale {
            return Ok(()); // unchanged and freshly declared — don't re-spam every job
        }
        info!(
            "OPoI: declaring {} model(s) to pool bridge ({})",
            model_ids.len(),
            model_ids.iter().map(|m| &m[..8.min(m.len())]).collect::<Vec<_>>().join(",")
        );
        self.send_channel
            .send(StratumLine {
                id: None,
                payload: StratumLinePayload::StratumCommand(StratumCommand::MiningDeclareCapabilities(
                    model_ids.clone(),
                )),
                jsonrpc: None,
                error: None,
            })
            .await?;
        self.declared_model_ids = model_ids;
        self.last_declare_at = Some(std::time::Instant::now());
        Ok(())
    }

    pub async fn connect(
        address: String,
        miner_address: String,
        pool_password: String,
        mine_when_not_synced: bool,
        block_template_ctr: Option<Arc<AtomicU16>>,
        ipfs_url: String,
    ) -> Result<Box<Self>, Error> {
        info!("Connecting to {}", address);
        let socket = TcpStream::connect(address).await?;

        // Keep the pool connection warm. Pools (and the NAT/firewalls between us)
        // drop connections that look idle; on a vardiff'd or high-diff rig we can
        // go minutes between `mining.submit` lines, so without keepalive traffic
        // the path can be torn down and we see a clean EOF (logged historically as
        // the misleading "stratum message payload is empty") followed by a full
        // reconnect/handshake every few minutes. SO_KEEPALIVE makes the kernel
        // emit probes on an otherwise-silent socket: it refreshes NAT/conntrack
        // state and detects a genuinely half-open peer quickly. Tunables let an
        // operator widen/narrow the cadence without a rebuild.
        //   - KERYX_TCP_KEEPALIVE_SECS=0 disables it entirely.
        //   - default: first probe after 45s idle, then every 15s.
        {
            let ka_idle: u64 = std::env::var("KERYX_TCP_KEEPALIVE_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(45);
            if ka_idle > 0 {
                let ka_intvl: u64 = std::env::var("KERYX_TCP_KEEPALIVE_INTERVAL_SECS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n| n > 0)
                    .unwrap_or(15);
                let keepalive = socket2::TcpKeepalive::new()
                    .with_time(Duration::from_secs(ka_idle))
                    .with_interval(Duration::from_secs(ka_intvl));
                let sref = socket2::SockRef::from(&socket);
                if let Err(e) = sref.set_tcp_keepalive(&keepalive) {
                    warn!("could not enable TCP keepalive on the pool socket: {e}");
                }
                // Low-latency share submits: don't let Nagle hold back small frames.
                let _ = sref.set_nodelay(true);
            }
        }

        let client = Framed::new(socket, NewLineJsonCodec::new());
        let (send_channel, recv) = mpsc::channel::<StratumLine>(3);
        let (sink, stream) = client.split();
        // Connection-death signal: when the socket-sink forwarder below ends
        // (the socket write half errored), or a submit can't be forwarded, fire
        // this so `listen` stops waiting on a dead read half and reconnects.
        let conn_dead = Arc::new(Notify::new());
        {
            let cd = conn_dead.clone();
            tokio::spawn(async move {
                let _ = ReceiverStream::new(recv).map(Ok).forward(sink).await;
                // forward() only returns on a sink (socket write) error/close.
                cd.notify_one();
            });
        }

        let share_state = SHARE_STATS.get_or_init(|| Arc::new(ShareStats::default())).clone();
        let last_stratum_id = Arc::new(AtomicU32::new(0));
        let current_task_slot: Arc<Mutex<Option<CurrentTask>>> = Arc::new(Mutex::new(None));
        let inference_cache: InferenceCache = Arc::new(Mutex::new(InferenceCacheInner {
            results: HashMap::new(),
            in_progress: HashSet::new(),
        }));
        // H6 pool path: the POOL builds the coinbase and therefore signs the V2 AiResponse with its
        // OWN escrow key — the miner transmits the UNSIGNED answer (reqId + response_length) and no
        // longer holds a responder signer here. (Solo/grpc self-signing lives in grpc.rs.)
        let (block_channel, block_handle) = Self::create_block_channel(
            send_channel.clone(),
            miner_address.clone(),
            last_stratum_id.clone(),
            share_state.clone(),
            Arc::clone(&current_task_slot),
            Arc::clone(&inference_cache),
            conn_dead.clone(),
        );
        Ok(Box::new(Self {
            log_handler: task::spawn(Self::log_shares(share_state.clone())),
            stream: Box::pin(stream),
            send_channel,
            miner_address,
            pool_password,
            mine_when_not_synced,
            block_template_ctr: block_template_ctr
                .unwrap_or_else(|| Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16))),
            target_pool: Default::default(),
            target_real: Default::default(),
            nonce_mask: u64::MAX, // full nonce space until set_extranonce assigns a sub-range
            nonce_fixed: 0,
            extranonce: None,
            last_stratum_id,
            shares_stats: share_state,
            block_channel,
            block_handle,
            conn_dead,
            ipfs_url,
            current_task_slot,
            inference_cache,
            inference_inflight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            telemetry_on: keryx_miner::telemetry::enabled(),
            telemetry_req_ids: HashSet::new(),
            start_time: std::time::Instant::now(),
            declared_model_ids: Vec::new(),
            last_declare_at: None,
        }))
    }

    fn create_block_channel(
        send_channel: Sender<StratumLine>,
        miner_address: String,
        last_stratum_id: Arc<AtomicU32>,
        share_stats: Arc<ShareStats>,
        current_task_slot: Arc<Mutex<Option<CurrentTask>>>,
        inference_cache: InferenceCache,
        conn_dead: Arc<Notify>,
    ) -> (Sender<BlockSeed>, BlockHandle) {
        let (send, recv) = mpsc::channel::<BlockSeed>(1);

        let handle = tokio::spawn(async move {
            let mut recv_stream = ReceiverStream::new(recv);
            // H6 attach-once dedup: the last reqId whose unsigned AiResponse we already put on a
            // share. At 10 BPS the job rolls over faster than a PoM share is found, so we can NOT
            // gate the attach on `ct.job_id == job_id` (it would never match) — instead we attach
            // the cached answer to the FIRST share found after inference completes, then skip it on
            // every subsequent share for the same reqId so the pool gets it exactly once (it keeps
            // re-dispatching until served, and stops once served).
            let mut last_attached_req_id: Option<String> = None;
            while let Some(seed) = recv_stream.next().await {
                let (nonce, job_id, pom_proof, device_id) = match seed {
                    BlockSeed::PartialBlock { nonce, id, pom_proof, device_id, .. } => (nonce, id, pom_proof, device_id),
                    BlockSeed::FullBlock(_) => unreachable!(),
                };
                let msg_id = last_stratum_id.fetch_add(1, Ordering::SeqCst);
                // Store the finding GPU alongside the job id so the accept/reject response (matched by
                // msg_id) can be attributed per-card.
                share_stats.shares_pending.try_lock().unwrap().insert(msg_id, (job_id.clone(), device_id));
                let nonce_hex = format!("{:016x}", nonce);
                let opoi_tag = keryx_inference::tag_fixed(nonce);

                let daa_now = LAST_DAA_SCORE.load(std::sync::atomic::Ordering::Relaxed);

                // Phase 2: check the inference cache for the CURRENT task's answer, and capture the
                // reqId the pool dispatched so it can be echoed back on the submit. NOTE: we do NOT
                // require `ct.job_id == job_id` — at 10 BPS the task-carrying job has already rolled
                // over by the time a PoM share is found, so a job match would never hold and the
                // answer would never attach. The cache key stays `ct.task.stable_id` (handle_ai_task
                // copies reqId→stable_id), so we attach whenever the current task's answer is cached.
                let (result_opt, req_id): (Option<InferenceResult>, Option<String>) = {
                    let task_guard = current_task_slot.lock().await;
                    if let Some(ref ct) = *task_guard {
                        if !ct.task.stable_id.is_empty() {
                            let cache_guard = inference_cache.lock().await;
                            let res = cache_guard.results.get(&ct.task.stable_id).cloned();
                            let rid = if ct.task.req_id.is_empty() { None } else { Some(ct.task.req_id.clone()) };
                            (res, rid)
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    }
                };

                // H6 service-bond era (pool path): at/after the PoM v3 gate, when an answer (CID)
                // exists AND the pool gave us a reqId, transmit the UNSIGNED answer data. The POOL —
                // which builds the coinbase — signs the V2 AiResponse with its own escrow key over
                // the exact 78 v1 bytes, so `response_length` (the miner's token count) MUST travel
                // on the wire, not be re-derived. Below the gate, or without a CID/reqId, fall
                // through to the plain 6-slot PoM submit.
                // Attach-once: skip if we already put THIS reqId's AiResponse on an earlier share.
                let unsigned_response: Option<(String, u32)> =
                    match (&result_opt, &req_id) {
                        (Some(res), Some(rid))
                            if daa_now >= keryx_miner::pom::pom_v3_activation_daa()
                                && last_attached_req_id.as_deref() != Some(rid.as_str()) =>
                        {
                            Some((rid.clone(), res.response_length))
                        }
                        _ => None,
                    };

                let line = if !pom_proof.is_empty() {
                    // PoM (post-fork): fixed 6-slot submit — proof always at params[5], CID-or-empty
                    // at params[4]. Matches POM_STRATUM_RECIPE.md (pool relays params[5] → RpcBlock
                    // .pomProof; it does not verify). hex is lowercase per hex::encode.
                    let proof_hex = hex::encode(&pom_proof);
                    let cid = result_opt.as_ref().map(|r| r.cid_b58.clone()).unwrap_or_default();
                    if let Some((rid, response_length)) = unsigned_response {
                        // H6 (unsigned): params[6]=reqId, params[7]=response_length. The POOL signs
                        // the V2 AiResponse with its own escrow key using EXACTLY this cid +
                        // response_length (the 78 v1 bytes) and matches it to reqId.
                        // Mark this reqId attached so we don't re-emit it on every following share.
                        last_attached_req_id = Some(rid.clone());
                        info!(
                            "PoM: submitting share with proof ({} B) + unsigned AiResponse (reqId={}, len={}) for job {}",
                            pom_proof.len(), rid, response_length, job_id
                        );
                        StratumLine {
                            id: Some(msg_id),
                            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                                MiningSubmit::MiningSubmitWithUnsignedResponse((
                                    miner_address.clone(),
                                    job_id,
                                    nonce_hex,
                                    opoi_tag,
                                    cid,
                                    proof_hex,
                                    rid,
                                    response_length,
                                )),
                            )),
                            jsonrpc: None,
                            error: None,
                        }
                    } else {
                        info!(
                            "PoM: submitting share with proof ({} B, {} hex chars) for job {}",
                            pom_proof.len(),
                            proof_hex.len(),
                            job_id
                        );
                        StratumLine {
                            id: Some(msg_id),
                            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                                MiningSubmit::MiningSubmitWithPom((
                                    miner_address.clone(),
                                    job_id,
                                    nonce_hex,
                                    opoi_tag,
                                    cid,
                                    proof_hex,
                                )),
                            )),
                            jsonrpc: None,
                            error: None,
                        }
                    }
                } else if let Some(cid) = result_opt.map(|r| r.cid_b58) {
                    info!("OPoI Phase 2: submitting share with CID for job {}", job_id);
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithCID((
                                miner_address.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                                cid,
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                } else {
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithTag((
                                miner_address.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                };

                if send_channel.send(line).await.is_err() {
                    // The socket-sink forwarder is gone → the connection's write
                    // half is dead. Signal `listen` to reconnect instead of
                    // silently dropping every future share.
                    warn!("Share submit could not be forwarded — pool connection write side is dead; triggering reconnect");
                    conn_dead.notify_one();
                    break;
                }
            }
        });
        (send, handle)
    }

    /// Send one `mining.telemetry` frame (dynamic per-GPU metrics + local share counters).
    /// Best-effort/non-fatal: a stats-lock miss or a send error just skips this tick.
    async fn send_telemetry(&mut self, miner: &mut MinerManager) {
        // Snapshot the miner's own per-device hashrate (std Mutex — extract + drop the guard BEFORE
        // any await; never hold it across the send).
        let (total, per_gpu): (f64, Vec<f64>) = match miner.stats().lock() {
            Ok(s) => (s.total_hashrate, s.devices.iter().map(|d| d.hashrate).collect()),
            Err(_) => return,
        };
        let acc = self.shares_stats.accepted.load(Ordering::SeqCst);
        let stale = self.shares_stats.stale.load(Ordering::SeqCst);
        let rej = self.shares_stats.low_diff.load(Ordering::SeqCst)
            + self.shares_stats.duplicate.load(Ordering::SeqCst);
        let uptime = self.start_time.elapsed().as_secs();
        let served: Vec<String> = keryx_miner::slm::serveable_model_ids().into_iter().map(hex::encode).collect();
        let obj = keryx_miner::telemetry::build_telemetry(uptime, total, &per_gpu, acc, rej, stale, &served);

        let id = self.last_stratum_id.fetch_add(1, Ordering::SeqCst);
        // Track for error-20 attribution; bound the set in case the pool never acks.
        if self.telemetry_req_ids.len() > 64 {
            self.telemetry_req_ids.clear();
        }
        self.telemetry_req_ids.insert(id);
        let _ = self
            .send_channel
            .send(StratumLine {
                id: Some(id),
                payload: StratumLinePayload::StratumCommand(StratumCommand::MiningTelemetry((obj,))),
                jsonrpc: None,
                error: None,
            })
            .await;
    }

    async fn handle_message(&mut self, msg: StratumLine, miner: &mut MinerManager) -> Result<(), Error> {
        match msg.clone() {
            StratumLine { id, payload, error: None, .. } => {
                match payload {
                    StratumLinePayload::StratumResult { result } if id.is_some() => {
                        match result {
                            StratumResult::Plain(Some(true)) | StratumResult::Eth((true, _)) => {
                                let rid = id.expect("We checked id is not none");
                                // Telemetry ack (mining.hello/mining.telemetry) — not a share.
                                if self.telemetry_req_ids.remove(&rid) {
                                    return Ok(());
                                }
                                if let Some((_jobid, device_id)) = self
                                    .shares_stats
                                    .shares_pending
                                    .try_lock()
                                    .unwrap()
                                    .remove(&rid)
                                {
                                    self.shares_stats.accepted.fetch_add(1, Ordering::SeqCst);
                                    crate::pow::record_share_accepted(device_id);
                                    info!("Share accepted (GPU {})", device_id);
                                } else {
                                    info!("{:?} (Last: {})", msg.clone(), self.last_stratum_id.load(Ordering::SeqCst));
                                    warn!("Ignoring result for now");
                                }
                                Ok(())
                            }
                            StratumResult::Subscribe((ref _subscriptions, ref extranonce, ref nonce_size)) => {
                                self.set_extranonce(extranonce.as_str(), nonce_size)
                                /*for (name, value) in _subscriptions {
                                    match name.as_str() {
                                        "mining.set_difficulty" => {self.set_difficulty(&f32::from_str(value.as_str())?)?;},
                                        _ => {warn!("Ignored {} (={})", name, value);}
                                    }
                                }
                                Ok(())*/
                            }
                            _ => Err(format!("Inconsistent stratum message: {:?}", msg).into()),
                        }
                    }
                    StratumLinePayload::StratumCommand(command) => match command {
                        StratumCommand::SetExtranonce(SetExtranonce::SetExtranoncePlain((
                            ref extranonce,
                            ref nonce_size,
                        ))) => self.set_extranonce(extranonce.as_str(), nonce_size),
                        StratumCommand::MiningSetDifficulty((ref difficulty,)) => self.set_difficulty(difficulty),
                        // Phase 2 OPoI: bridge dispatches an AiRequest task alongside the block.
                        StratumCommand::MiningNotify(MiningNotify::MiningNotifyWithTask((
                            id,
                            header_hash,
                            timestamp,
                            daa_score,
                            task_json,
                        ))) => {
                            self.block_template_ctr
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                                .unwrap();
                            // OPoI v2 hardfork: advance the served lineup when the chain crosses H.
                            // Upstream drives this from the solo grpc job loop (grpc.rs); stratum is
                            // OUR job source, so the swap MUST be driven here or the v2 (uncensored)
                            // models never load and post-fork PoM-PoW has no weights resident.
                            keryx_miner::slm::advance_lineup_if_due(daa_score);
                            LAST_DAA_SCORE.store(daa_score, std::sync::atomic::Ordering::Relaxed);
                            // OPoI hard gate (mirrors solo grpc.rs): no models ready = no mining.
                            // Keryx core invariant — no inference, no PoW.
                            if keryx_miner::slm::loaded_model_ids().is_empty() {
                                if self.block_template_ctr.load(Ordering::SeqCst) % 200 == 0 {
                                    warn!("OPoI: no models ready — mining suspended (no inference = no mining)");
                                }
                                return miner.process_block(None).await;
                            }
                            // Models are ready and we are about to mine — make sure the pool knows
                            // which models this rig serves (re-declares only if the set changed).
                            self.declare_capabilities_if_changed().await?;
                            // The task arrives as a JSON object (locked contract); a legacy
                            // double-encoded JSON string is unwrapped so handle_ai_task's
                            // `from_str` sees the object either way.
                            let task_json = match task_json {
                                serde_json::Value::String(s) => s,
                                other => other.to_string(),
                            };
                            let inference_started =
                                self.handle_ai_task(id.clone(), task_json, miner).await;
                            if inference_started {
                                // PoW already paused inside handle_ai_task — do NOT feed a new
                                // block template or the GPU immediately resumes hashing.
                                Ok(())
                            } else {
                                miner
                                    .process_block(Some(PartialBlock {
                                        id,
                                        header_hash,
                                        timestamp,
                                        daa_score,
                                        nonce: 0,
                                        target: self.target_pool,
                                        nonce_mask: self.nonce_mask,
                                        nonce_fixed: self.nonce_fixed,
                                        hash: None,
                                        pom_proof: Vec::new(),
                                        device_id: 0,
                                    }))
                                    .await
                            }
                        }
                        StratumCommand::MiningNotify(MiningNotify::MiningNotifyShortV2((
                            id,
                            header_hash,
                            timestamp,
                            daa_score,
                        ))) => {
                            self.block_template_ctr
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                                .unwrap();
                            // OPoI v2 hardfork: advance the served lineup when the chain crosses H.
                            // Stratum is our job source (upstream drives this from solo grpc.rs), so the
                            // swap MUST happen here or post-fork PoM-PoW has no v2 weights resident.
                            keryx_miner::slm::advance_lineup_if_due(daa_score);
                            LAST_DAA_SCORE.store(daa_score, std::sync::atomic::Ordering::Relaxed);
                            // OPoI hard gate (mirrors solo grpc.rs): no models ready = no mining.
                            // Keryx core invariant — no inference, no PoW.
                            if keryx_miner::slm::loaded_model_ids().is_empty() {
                                if self.block_template_ctr.load(Ordering::SeqCst) % 200 == 0 {
                                    warn!("OPoI: no models ready — mining suspended (no inference = no mining)");
                                }
                                return miner.process_block(None).await;
                            }
                            // Models ready + about to mine — keep the pool's view of served models current.
                            self.declare_capabilities_if_changed().await?;
                            // No AiRequest in this job — clear the task slot.
                            *self.current_task_slot.lock().await = None;
                            miner
                                .process_block(Some(PartialBlock {
                                    id,
                                    header_hash,
                                    timestamp,
                                    daa_score,
                                    nonce: 0,
                                    target: self.target_pool,
                                    nonce_mask: self.nonce_mask,
                                    nonce_fixed: self.nonce_fixed,
                                    hash: None,
                                    pom_proof: Vec::new(),
                                        device_id: 0,
                                }))
                                .await
                        }
                        StratumCommand::MiningNotify(MiningNotify::MiningNotifyShort((id, header_hash, timestamp))) => {
                            self.block_template_ctr
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                                .unwrap();
                            // Authoritative per-job DAA. The keryx H6 pool has no daa_score field in
                            // the plain Short notify but embeds the DAA in the jobId as `<rand8>_<daa>`
                            // (see job_id_daa). Parse it and use it directly so the PoM/v3 gates and the
                            // seed/pow salts match the node byte-for-byte. Fall back to the historical
                            // LAST_DAA_SCORE/POOL_FORCED_POM flooring only when the jobId carries no
                            // numeric suffix (older Short-only pools).
                            let daa_score = if let Some(daa) = job_id_daa(&id) {
                                // Drive the OPoI lineup swap here (stratum is our job source), same as
                                // the ShortV2/WithTask paths, and remember it for any later suffix-less job.
                                keryx_miner::slm::advance_lineup_if_due(daa);
                                LAST_DAA_SCORE.store(daa, std::sync::atomic::Ordering::Relaxed);
                                daa
                            } else {
                                // Short stratum notify carries no daa_score; pin it to the
                                // current salt era so the host generates the right matrix.
                                // Post-relaunch the chain is on SALT v4. Inherit the last
                                // real daa (from WithTask/ShortV2) so post-fork PoM still
                                // activates on Short-only pools; floor at SALT-v4 era.
                                let base = LAST_DAA_SCORE.load(std::sync::atomic::Ordering::Relaxed)
                                    .max(crate::pow::heavy_hash::POW_SALT_V4_ACTIVATION_DAA);
                                // If the pool told us the fork is active (POOL_FORCED_POM),
                                // floor at the CURRENT frontier — activation + H2 tier AND
                                // the H3 gate (level_activation_daa, salted folds). The old
                                // floor stopped at H2: on a Short-only pool/proxy (no
                                // daa_score on the wire) the miner then mined UNSALTED
                                // pre-H3 folds forever, and every post-H3 verifier rejected
                                // ~every share as "low difficulty". H3 (like H2) is a frozen
                                // frontier the network is permanently past, so any pool
                                // forcing PoM today is necessarily post-H3.
                                if POOL_FORCED_POM.load(std::sync::atomic::Ordering::Relaxed) {
                                    // 0.7.0 is the H4 binary: floor at the H4 frontier too, so a
                                    // Short-only pool forcing PoM stamps the H4 tier and builds
                                    // proof-v2 (h4 dominates H2/H3; env-override respected).
                                    base.max(keryx_miner::models::VERY_LIGHT_ACTIVATION_DAA)
                                        .max(keryx_miner::pom::level_activation_daa())
                                        .max(keryx_miner::pom::h4_activation_daa())
                                } else {
                                    base
                                }
                            };
                            // OPoI hard gate (mirrors solo grpc.rs): no models ready = no mining.
                            // Keryx core invariant — no inference, no PoW.
                            if keryx_miner::slm::loaded_model_ids().is_empty() {
                                if self.block_template_ctr.load(Ordering::SeqCst) % 200 == 0 {
                                    warn!("OPoI: no models ready — mining suspended (no inference = no mining)");
                                }
                                return miner.process_block(None).await;
                            }
                            // Models ready + about to mine — keep the pool's view of served models current.
                            self.declare_capabilities_if_changed().await?;
                            *self.current_task_slot.lock().await = None;
                            miner
                                .process_block(Some(PartialBlock {
                                    id,
                                    header_hash,
                                    timestamp,
                                    daa_score,
                                    nonce: 0,
                                    target: self.target_pool,
                                    nonce_mask: self.nonce_mask,
                                    nonce_fixed: self.nonce_fixed,
                                    hash: None,
                                    pom_proof: Vec::new(),
                                        device_id: 0,
                                }))
                                .await
                        }
                        StratumCommand::MiningChallenge((model_id_hex, nonce_hex)) => {
                            self.handle_challenge(model_id_hex, nonce_hex, miner).await;
                            Ok(())
                        }
                        // H6 interactive chat (off-chain product path): run inference on the GPU
                        // (pausing PoW like the challenge path) and reply inline. No tx/escrow.
                        StratumCommand::MiningInferenceRequest(req) => {
                            self.handle_inference_request(req, miner).await;
                            Ok(())
                        }
                        _ => Err(format!("Unexpected stratum message: {:?}", msg).into()),
                    },
                    _ => Err(format!("Inconsistent stratum message: {:?}", msg).into()),
                }
            }
            StratumLine {
                id: Some(id),
                payload: StratumLinePayload::StratumResult { .. },
                error: Some(StratumError(code, error, _)),
                ..
            } => {
                // An errored StratumResult does NOT always correspond to a pending share: some pools
                // (observed on suprnova krx after a stale/duplicate worker session) return an error
                // result for a setup message (subscribe/authorize) whose id was never a submitted
                // share. The old `.remove(&id).unwrap()` panicked the whole miner in that case —
                // which, with failover enabled, could also crash on an unexpected reply from a backup
                // pool. Treat the job id as optional and never panic; the match on `code` below still
                // returns Err for Unauthorized/NotSubscribed so the reconnect/failover path fires.
                // Telemetry method rejected (v0.7.0): if this id was a mining.hello/mining.telemetry
                // request, an error 20 means the pool doesn't support telemetry → disable it for the
                // session and keep mining. NEVER counted as a rejected share; never fatal.
                if self.telemetry_req_ids.remove(&id) {
                    if matches!(code, ErrorCode::Unknown) && self.telemetry_on {
                        self.telemetry_on = false;
                        info!(
                            "Pool does not support miner telemetry (mining.hello/mining.telemetry): '{}' \
                             — disabled for this session, mining continues normally.",
                            error
                        );
                    }
                    return Ok(());
                }
                let pending = { self.shares_stats.shares_pending.try_lock().unwrap().remove(&id) };
                // Any error code here means this submitted share was NOT accepted — attribute the
                // rejection to the GPU that found it (the R: column).
                if let Some((_, device_id)) = &pending {
                    crate::pow::record_share_rejected(*device_id);
                }
                let jobid = pending.map(|(j, _)| j);
                match code {
                    ErrorCode::Unknown => {
                        // Match solo-mining behaviour (grpc.rs SubmitBlockResponse): a rejected
                        // share/block is logged but never fatal. Returning Err here tore down the
                        // whole connection and caused an infinite reconnect loop on every share.
                        self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                        warn!("Share rejected by pool (Job id: {:?}): {}", jobid, error);
                        Ok(())
                    }
                    ErrorCode::JobNotFound => {
                        self.shares_stats.stale.fetch_add(1, Ordering::SeqCst);
                        warn!("Stale share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::DuplicateShare => {
                        self.shares_stats.duplicate.fetch_add(1, Ordering::SeqCst);
                        warn!("Duplicate share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::LowDifficultyShare => {
                        self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                        warn!("Low difficulty share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::Unauthorized => {
                        // Distinguish a genuine auth failure (bad wallet/worker) from the pool
                        // signalling the PoM hardfork: "Unauthorized: pom proof required (hardfork
                        // active)" means the pool is post-fork and requires a proof on every share,
                        // but we were submitting proofless kHeavyHash shares — i.e. this connection
                        // only ever delivered plain `Short` notifies (no daa_score), so our gate
                        // never fired. Trust the pool as the authority on fork state: force PoM
                        // activation and STAY connected so the next job attaches a proof, instead of
                        // returning Err (which tears down the connection into a reconnect loop).
                        let low = error.to_string().to_lowercase();
                        if low.contains("pom") || low.contains("hardfork") {
                            if !POOL_FORCED_POM.swap(true, Ordering::SeqCst) {
                                warn!(
                                    "Pool requires a PoM proof (hardfork active): '{}'. This connection sent no daa_score \
                                     (Short-only notifies) so PoM never activated and we submitted proofless shares. \
                                     Forcing PoM activation for this session — the next job will build a real proof.",
                                    error
                                );
                            }
                            Ok(())
                        } else {
                            error!("Got error code {}: {}", code, error);
                            Err(error.into())
                        }
                    }
                    ErrorCode::NotSubscribed => {
                        error!("Got error code {}: {}", code, error);
                        Err(error.into())
                    }
                }
            }
            _ => Err(format!("Unhandled stratum response: {:?}", msg).into()),
        }
    }

    fn set_difficulty(&mut self, difficulty: &f32) -> Result<(), Error> {
        let mut buf = [0u64, 0u64, 0u64, 0u64];
        let (mantissa, exponent, _) = difficulty.recip().integer_decode();
        let new_mantissa = mantissa * DIFFICULTY_1_TARGET.0;
        let new_exponent = (DIFFICULTY_1_TARGET.1 + exponent) as u64;
        let start = (new_exponent / 64) as usize;
        let remainder = new_exponent % 64;

        buf[start] = new_mantissa << remainder; // bottom
        if start < 3 {
            buf[start + 1] = new_mantissa >> (64 - remainder); // top
        } else if new_mantissa.leading_zeros() < remainder as u32 {
            return Err("Target is too big".into());
        }

        self.target_pool = Uint256::new(buf);
        info!("Difficulty: {:?}, Target: 0x{}", difficulty, hex::encode(self.target_pool.to_be_bytes()));
        // Expected work per share at this target, as wall-clock at typical PoM rates. Answers the
        // recurring "the card is hashing but never finds a share" report: PoM rates are MH/s (the
        // walk is memory-hard), so at the common pool default (d=16 ≈ 2^36 hashes/share) a single
        // small card legitimately averages HOURS per share — that's the pool difficulty, not a fault.
        {
            let t = buf; // LE u64 words of the 256-bit target
            let tf = t[0] as f64 + t[1] as f64 * 2f64.powi(64) + t[2] as f64 * 2f64.powi(128) + t[3] as f64 * 2f64.powi(192);
            if tf > 0.0 {
                let hashes = 2f64.powi(256) / tf;
                let fmt = |mhs: f64| -> String {
                    let s = hashes / (mhs * 1e6);
                    if s >= 5400.0 { format!("~{:.1} h", s / 3600.0) }
                    else if s >= 90.0 { format!("~{:.0} min", s / 60.0) }
                    else { format!("~{:.0} s", s) }
                };
                info!(
                    "Share expectation at this difficulty: ≈{:.1} Ghashes/share on AVERAGE — e.g. {} at 5 MH/s \
                     (one small GPU), {} at 30 MH/s (multi-GPU rig). Long share-less stretches on a single card \
                     are NORMAL at high difficulty; on suprnova set a lower static difficulty (password d=4 or \
                     d=1) for faster share feedback.",
                    hashes / 1e9, fmt(5.0), fmt(30.0),
                );
            }
        }
        Ok(())
    }

    fn set_extranonce(&mut self, extranonce: &str, nonce_size: &u32) -> Result<(), Error> {
        self.extranonce = Some(extranonce.to_string());
        info!("Extra! {:?}", extranonce);
        self.nonce_fixed = u64::from_str_radix(extranonce, 16)? << (nonce_size * 8);
        info!("Extra Done!");
        self.nonce_mask = (1 << (nonce_size * 8)) - 1;
        Ok(())
    }

    async fn log_shares(shares_info: Arc<ShareStats>) {
        let mut ticker = tokio::time::interval(LOG_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut _last_instant = ticker.tick().await;
        loop {
            let _now = ticker.tick().await;
            info!("{}", shares_info)
        }
    }

    /// Parse the task JSON from a `MiningNotifyWithTask`, store it in `current_task_slot`,
    /// and spawn a background inference+IPFS upload if the result is not already cached.
    /// Handle a `mining.challenge` from the bridge.
    ///
    /// The bridge relays the node's periodic capability challenge: the miner must prove
    /// it has the requested model loaded and can produce inference output. The result is
    /// sent back as `mining.challenge_response` so the bridge can forward it to the node.
    async fn handle_challenge(&mut self, model_id_hex: String, nonce_hex: String, miner: &mut MinerManager) {
        let model_id_bytes = match hex::decode(&model_id_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                warn!("OPoI challenge: invalid model_id_hex '{}'", model_id_hex);
                return;
            }
        };
        let mut model_id = [0u8; 32];
        model_id.copy_from_slice(&model_id_bytes);

        if !keryx_miner::slm::is_model_ready(&model_id) {
            warn!("OPoI challenge: model {:.8} not ready — sending empty response", model_id_hex);
            self.send_channel.send(make_challenge_response_line(&model_id_hex, &nonce_hex, "")).await.ok();
            return;
        }

        // Route to the best FREE eligible card (per-card busy, not global). All eligible cards busy
        // ⇒ drop this challenge (empty response); the bridge re-challenges later. The lease pins the
        // card for the whole inference and auto-releases when the spawned task's closure ends.
        let lease = match keryx_miner::slm::acquire_inference_card(&model_id, 30_000) {
            Some(l) => l,
            None => {
                warn!("OPoI challenge: all eligible cards busy — dropping challenge for model {:.8}", model_id_hex);
                self.send_channel.send(make_challenge_response_line(&model_id_hex, &nonce_hex, "")).await.ok();
                return;
            }
        };
        let gpu = lease.gpu();

        // Pause PoW so the GPU is available for the challenge inference (ref-counted: the FIRST
        // in-flight inference pauses, the LAST resumes). In --cpu-inference mode the GPU is free.
        let cpu_inference = keryx_miner::slm::cpu_inference_enabled();
        let miner_flag = miner.opoi_challenge_flag();
        let first = self.inference_inflight.fetch_add(1, Ordering::SeqCst) == 0;
        if cpu_inference {
            info!("OPoI challenge: CPU inference — PoW continues — model={:.8} nonce={:.8}", model_id_hex, nonce_hex);
        } else if first {
            miner_flag.store(true, Ordering::SeqCst);
            miner.process_block(None).await.ok();
            info!("OPoI challenge: PoW suspended (GPU {}) — model={:.8} nonce={:.8}", gpu, model_id_hex, nonce_hex);
        } else {
            info!("OPoI challenge: concurrent on GPU {} — model={:.8} nonce={:.8}", gpu, model_id_hex, nonce_hex);
        }

        let prompt = format!("Keryx inference challenge {}: briefly describe what you are.", nonce_hex);
        let send_channel = self.send_channel.clone();
        let pause = PowPauseGuard { inflight: Arc::clone(&self.inference_inflight), miner_flag };

        tokio::task::spawn_blocking(move || {
            let _pause = pause; // drop → PoW resumes when this is the last inference (panic-safe)
            let _lease = lease; // drop → release this card for the next queued request
            let result = keryx_miner::slm::load_and_run_inference_on(gpu, &model_id, &prompt, CHALLENGE_MAX_TOKENS);
            let text = result.unwrap_or_default();
            if text.is_empty() {
                warn!("OPoI challenge: inference returned empty text for model {:.8}", model_id_hex);
            } else {
                info!("OPoI challenge: done for model {:.8} ({} chars) — PoW resumes on next notify", model_id_hex, text.len());
            }
            let line = make_challenge_response_line(&model_id_hex, &nonce_hex, &text);
            if send_channel.blocking_send(line).is_err() {
                warn!("OPoI challenge: send_channel closed, could not deliver response");
            }
        });
    }

    /// H6 interactive chat (`mining.inference_request` → `mining.inference_result`). Off-chain,
    /// low-latency product path: NO tx, NO escrow, NO consensus. Runs SLM inference on the GPU
    /// (reusing the challenge path's GPU-pause pattern) and returns the answer text INLINE. When no
    /// request is pending this is never entered, so PoW is untouched. On an unready model, empty
    /// output, or any failure it replies `{ reqId, ok:false, error }`.
    async fn handle_inference_request(&mut self, req: InferenceRequestParams, miner: &mut MinerManager) {
        let req_id = req.req_id.clone();

        // Validate the model_id (64 hex → 32 bytes) and that it is a declared/ready tier.
        let model_id_bytes = match hex::decode(&req.model_id) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                warn!("chat[{}]: invalid model_id '{}'", req_id, req.model_id);
                self.send_channel
                    .send(make_inference_result_err(&req_id, "invalid model_id"))
                    .await
                    .ok();
                return;
            }
        };
        let mut model_id = [0u8; 32];
        model_id.copy_from_slice(&model_id_bytes);
        if !keryx_miner::slm::loaded_model_ids().iter().any(|m| m == &model_id) {
            warn!("chat[{}]: model {:.8} not ready", req_id, req.model_id);
            self.send_channel
                .send(make_inference_result_err(&req_id, "model not ready"))
                .await
                .ok();
            return;
        }

        // Per-card busy guard: route to the best FREE eligible card, waiting up to the request's
        // deadline for one to free. Only reply "busy" when NO eligible card frees in time — a
        // second concurrent request on a DIFFERENT card is allowed. The lease pins the card.
        let lease = match keryx_miner::slm::acquire_inference_card(&model_id, req.deadline_ms) {
            Some(l) => l,
            None => {
                warn!("chat[{}]: all eligible cards busy through deadline — replying busy", req_id);
                self.send_channel
                    .send(make_inference_result_err(&req_id, "busy"))
                    .await
                    .ok();
                return;
            }
        };
        let gpu = lease.gpu();

        // Pause PoW (ref-counted: first inference pauses, last resumes). --cpu-inference keeps hashing.
        let cpu_inference = keryx_miner::slm::cpu_inference_enabled();
        let miner_flag = miner.opoi_challenge_flag();
        let first = self.inference_inflight.fetch_add(1, Ordering::SeqCst) == 0;
        if cpu_inference {
            info!("chat[{}]: CPU inference — PoW continues — model={:.8}", req_id, req.model_id);
        } else if first {
            miner_flag.store(true, Ordering::SeqCst);
            miner.process_block(None).await.ok();
            info!("chat[{}]: PoW suspended (GPU {}) — model={:.8}", req_id, gpu, req.model_id);
        } else {
            info!("chat[{}]: concurrent on GPU {} — model={:.8}", req_id, gpu, req.model_id);
        }

        let prompt = req.prompt.clone();
        let max_tokens = req.max_tokens;
        let send_channel = self.send_channel.clone();
        let pause = PowPauseGuard { inflight: Arc::clone(&self.inference_inflight), miner_flag };
        let model_hex = req.model_id.clone();

        tokio::task::spawn_blocking(move || {
            let _pause = pause; // PoW resumes when the last inference drops this (panic-safe)
            let _lease = lease; // releases the leased card on drop
            let started = std::time::Instant::now();
            let result = keryx_miner::slm::load_and_run_inference_on(gpu, &model_id, &prompt, max_tokens);
            let ms = started.elapsed().as_millis() as u32;
            let line = match result {
                Some(text) if !text.is_empty() => {
                    let tokens = text.split_whitespace().count() as u32;
                    info!("chat[{}]: done model={:.8} ({} tokens, {} ms) — PoW resumes on next notify", req_id, model_hex, tokens, ms);
                    make_inference_result_ok(&req_id, text, tokens, ms)
                }
                _ => {
                    warn!("chat[{}]: inference produced no output", req_id);
                    make_inference_result_err(&req_id, "inference failed")
                }
            };
            if send_channel.blocking_send(line).is_err() {
                warn!("chat[{}]: send_channel closed, could not deliver result", req_id);
            }
        });
    }

    /// Parse the task JSON from a `MiningNotifyWithTask`, store it in `current_task_slot`,
    /// Handles an AiTask dispatched by the bridge. Returns `true` if inference was launched
    /// and PoW has been paused — the caller must NOT call `process_block(Some(...))` in that
    /// case; PoW resumes automatically on the next `mining.notify` after inference completes.
    async fn handle_ai_task(&mut self, job_id: String, task_json: String, miner: &mut MinerManager) -> bool {
        let mut task: AiTask = match serde_json::from_str(&task_json) {
            Ok(t) => t,
            Err(e) => {
                warn!("OPoI: failed to parse task JSON from bridge: {}", e);
                *self.current_task_slot.lock().await = None;
                return false;
            }
        };

        // H6 wire: the pool identifies a request by `reqId` and may omit the legacy `stable_id`.
        // Use reqId as the inference-cache dedup key (and the create_block_channel lookup key) when
        // stable_id is absent, so inference still runs and the answer reaches the submit.
        if task.stable_id.is_empty() {
            task.stable_id = task.req_id.clone();
        }

        // Store task for this job so create_block_channel can look up the CID.
        *self.current_task_slot.lock().await = Some(CurrentTask { job_id, task: task.clone() });

        // Skip inference if there is no key (neither stable_id nor reqId) or it's already done/running.
        if task.stable_id.is_empty() {
            return false;
        }
        let already_handled = {
            let cache = self.inference_cache.lock().await;
            cache.results.contains_key(&task.stable_id) || cache.in_progress.contains(&task.stable_id)
        };
        if already_handled {
            return false;
        }

        // Decode model_id hex and check it is ready on disk.
        let model_id_bytes = match hex::decode(&task.model_id_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                warn!("OPoI [{}]: invalid model_id_hex '{}'", task.stable_id, task.model_id_hex);
                return false;
            }
        };
        let mut model_id = [0u8; 32];
        model_id.copy_from_slice(&model_id_bytes);

        if !keryx_miner::slm::is_model_ready(&model_id) {
            warn!("OPoI [{}]: model not ready — inference skipped", task.stable_id);
            return false;
        }

        // Per-card busy guard: route to the best FREE eligible card, waiting up to the task's
        // deadline. Skip only when NO eligible card frees in time (a concurrent AiTask on another
        // card is allowed). The lease pins the card for the whole inference + IPFS upload.
        let lease = match keryx_miner::slm::acquire_inference_card(&model_id, task.deadline_ms) {
            Some(l) => l,
            None => {
                warn!("OPoI AiTask [{}]: all eligible cards busy through deadline — skipping", task.stable_id);
                return false;
            }
        };
        let gpu = lease.gpu();

        // Pause PoW — running the PoW walk and SLM inference on the SAME card simultaneously crashes
        // the GPU. Ref-counted: first inference pauses, last resumes. --cpu-inference keeps hashing.
        let cpu_inference = keryx_miner::slm::cpu_inference_enabled();
        let miner_flag = miner.opoi_challenge_flag();
        let first = self.inference_inflight.fetch_add(1, Ordering::SeqCst) == 0;
        if !cpu_inference && first {
            miner_flag.store(true, Ordering::SeqCst);
            miner.process_block(None).await.ok();
            info!("OPoI AiTask [{}]: PoW suspended for GPU {} inference", task.stable_id, gpu);
        } else if !cpu_inference {
            info!("OPoI AiTask [{}]: concurrent inference on GPU {}", task.stable_id, gpu);
        } else {
            info!("OPoI AiTask [{}]: CPU inference — PoW continues", task.stable_id);
        }

        // Mark in-progress and spawn the blocking inference + IPFS upload.
        {
            let mut cache = self.inference_cache.lock().await;
            cache.in_progress.insert(task.stable_id.clone());
        }
        let stable_id = task.stable_id.clone();
        let prompt = task.prompt.clone();
        let max_tokens = task.max_tokens;
        let ipfs_url = self.ipfs_url.clone();
        let cache_ref = Arc::clone(&self.inference_cache);
        let pause = PowPauseGuard { inflight: Arc::clone(&self.inference_inflight), miner_flag };

        tokio::task::spawn_blocking(move || {
            let _pause = pause; // PoW resumes when the last inference drops this (panic-safe)
            let _lease = lease; // releases the leased card on drop
            run_inference_and_upload(gpu, model_id, prompt, max_tokens, ipfs_url, stable_id, cache_ref);
        });

        // GPU mode: PoW was paused, caller must not feed a new block (returns true).
        // CPU mode: PoW kept running, caller should feed a block to keep hashing (returns false).
        !cpu_inference
    }
}

impl Drop for StratumHandler {
    fn drop(&mut self) {
        self.log_handler.abort();
        self.block_handle.abort()
    }
}

// ── Phase 2 OPoI — blocking inference helpers ────────────────────────────────

/// Runs SLM inference, uploads the result to IPFS, then stores the CID in the cache.
/// Called from `spawn_blocking` — must not call async functions.
fn run_inference_and_upload(
    gpu: usize,
    model_id: [u8; 32],
    prompt: String,
    max_tokens: usize,
    ipfs_url: String,
    stable_id: String,
    cache: InferenceCache,
) {
    let result_opt = do_inference_and_upload(gpu, &model_id, &prompt, max_tokens, &ipfs_url, &stable_id);
    let mut guard = cache.blocking_lock();
    guard.in_progress.remove(&stable_id);
    if let Some(result) = result_opt {
        guard.results.insert(stable_id, result);
    }
}

/// Build a successful `mining.inference_result` line (text returned inline).
fn make_inference_result_ok(req_id: &str, text: String, tokens: u32, ms: u32) -> StratumLine {
    StratumLine {
        id: None,
        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningInferenceResult(InferenceResultParams {
            req_id: req_id.to_string(),
            ok: true,
            text: Some(text),
            tokens: Some(tokens),
            ms: Some(ms),
            error: None,
        })),
        jsonrpc: None,
        error: None,
    }
}

/// Build a failed `mining.inference_result` line.
fn make_inference_result_err(req_id: &str, error: &str) -> StratumLine {
    StratumLine {
        id: None,
        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningInferenceResult(InferenceResultParams {
            req_id: req_id.to_string(),
            ok: false,
            text: None,
            tokens: None,
            ms: None,
            error: Some(error.to_string()),
        })),
        jsonrpc: None,
        error: None,
    }
}

fn make_challenge_response_line(model_id_hex: &str, nonce_hex: &str, result: &str) -> StratumLine {
    StratumLine {
        id: None,
        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningChallengeResponse((
            model_id_hex.to_string(),
            nonce_hex.to_string(),
            result.to_string(),
        ))),
        jsonrpc: None,
        error: None,
    }
}

fn do_inference_and_upload(
    gpu: usize,
    model_id: &[u8; 32],
    prompt: &str,
    max_tokens: usize,
    ipfs_url: &str,
    stable_id: &str,
) -> Option<InferenceResult> {
    info!("OPoI [{}]: starting SLM inference (max_tokens={}, GPU {})", stable_id, max_tokens, gpu);
    let text = keryx_miner::slm::load_and_run_inference_on(gpu, model_id, prompt, max_tokens)?;
    if text.is_empty() {
        warn!("OPoI [{}]: inference returned empty text — skipping IPFS upload", stable_id);
        return None;
    }
    // Token count mirrors the solo grpc path's `result.split_whitespace().count()` — this is the
    // `response_length` the V2 responder signature covers, so the pool must commit the same value.
    let response_length = text.split_whitespace().count() as u32;
    match crate::ipfs::upload(&text, ipfs_url) {
        Ok(cid_bytes) => {
            // Convert raw 34-byte multihash to base58 CIDv0 string via AiResponsePayload helper.
            let cid_b58 = keryx_inference::AiResponsePayload::new([0u8; 32], 0, cid_bytes, 0).cid_v0();
            info!("OPoI [{}]: inference complete, IPFS CID={}", stable_id, cid_b58);
            Some(InferenceResult { cid_b58, response_length })
        }
        Err(e) => {
            warn!("OPoI [{}]: IPFS upload failed: {}", stable_id, e);
            None
        }
    }
}
