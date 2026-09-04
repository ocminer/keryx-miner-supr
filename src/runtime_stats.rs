//! Process-wide, backend-neutral runtime statistics.
//!
//! Mining clients are replaced whenever a connection is re-established, while a dashboard and
//! its counters describe the lifetime of the process. This module therefore owns one process-wide
//! hub. Hot-path values are atomics; larger, infrequently-changing snapshots use `try_read` /
//! `try_write` and simply retain the previous value on contention. A frontend should prefer
//! [`try_snapshot`] and keep its previous frame when it returns `None`.
//!
//! This is an operational surface only. It must never contain payout addresses, worker identities,
//! escrow keys/certificates/public keys, transaction IDs, prompts, CIDs, local paths, or raw peer
//! errors. Endpoints are reduced to scheme + authority before storage.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;

const UNSET: u64 = u64::MAX;
const EVENT_CAPACITY: usize = 256;
const LATENCY_WINDOW: usize = 128;
const HASHRATE_WINDOW: usize = 12;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum MiningMode {
    #[default]
    Unknown = 0,
    Pool = 1,
    Solo = 2,
}

impl MiningMode {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Pool,
            2 => Self::Solo,
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ConnectionState {
    #[default]
    Offline = 0,
    Connecting = 1,
    Connected = 2,
    Failover = 3,
}

impl ConnectionState {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Connecting,
            2 => Self::Connected,
            3 => Self::Failover,
            _ => Self::Offline,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum InferenceKind {
    #[default]
    Unknown = 0,
    PoolChallenge = 1,
    Interactive = 2,
    PoolTask = 3,
    SoloChallenge = 4,
    SoloRequest = 5,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EventKind {
    #[default]
    Info,
    Job,
    ShareAccepted,
    ShareRejected,
    BlockFound,
    BlockAccepted,
    BlockRejected,
    InferenceOk,
    InferenceFailed,
    Escrow,
    HealthWarning,
    Error,
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeEvent {
    pub uptime_ms: u64,
    pub kind: EventKind,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct DeviceSnapshot {
    pub index: u32,
    pub label: String,
    pub backend: String,
    pub hashrate_hs: f64,
    pub temp_c: Option<u32>,
    pub hotspot_c: Option<u32>,
    pub fan_pct: Option<u32>,
    pub power_w: Option<f64>,
    pub core_mhz: Option<u32>,
    pub mem_mhz: Option<u32>,
    pub vram_used_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
    pub efficiency_mhs_per_w: Option<f64>,
    pub accepted: u64,
    pub rejected: u64,
}

#[derive(Clone, Debug, Default)]
pub struct MiningSnapshot {
    /// Advances once per backend telemetry/hashrate publication, not once per UI frame.
    pub sample_sequence: u64,
    pub preparing: bool,
    pub inference_paused: bool,
    pub total_hashrate_hs: f64,
    pub average_60s_hs: Option<f64>,
    pub hashrate_history_hs: Vec<f64>,
    pub total_power_w: Option<f64>,
    pub efficiency_mhs_per_w: Option<f64>,
    pub devices: Vec<DeviceSnapshot>,
}

#[derive(Clone, Debug, Default)]
pub struct ShareSnapshot {
    pub submitted: u64,
    pub accepted: u64,
    pub stale: u64,
    pub low_diff: u64,
    pub duplicate: u64,
    pub other: u64,
    pub pending: u64,
    pub last_accepted_age_secs: Option<u64>,
}

impl ShareSnapshot {
    pub fn rejected(&self) -> u64 {
        self.stale.saturating_add(self.low_diff).saturating_add(self.duplicate).saturating_add(self.other)
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlockSnapshot {
    pub found: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub pending: u64,
    pub last_accepted_age_secs: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct InferenceSnapshot {
    /// Unique valid external inference workflows, including terminal busy/unready outcomes.
    /// Deduplicated followers and startup/self-test calls are excluded.
    pub requested: u64,
    /// Successful inference outputs which also completed any required preparation (e.g. IPFS).
    pub prepared: u64,
    /// Non-empty inference results completed all required preparation and were queued/cached.
    pub served: u64,
    /// Responses whose transport-level delivery was observed (pool share ACK where required).
    pub delivered: u64,
    pub failed: u64,
    /// Requests which could not acquire capacity by their deadline or were rejected at capacity.
    pub busy: u64,
    /// Admitted workflows currently waiting for/running on a card.
    pub active: u64,
    pub queue_depth: u64,
    pub queue_capacity: u64,
    /// Exact GPUs with a currently proven route for the displayed model. This is separate from
    /// `gpu_index`, which is the most recent externally served route and remains unset during
    /// startup self-tests by design.
    pub route_gpus: Vec<u32>,
    pub gpu_index: Option<u32>,
    pub last_latency_ms: Option<u64>,
    pub p95_latency_ms: Option<u64>,
    pub last_tokens: Option<u64>,
    pub pow_pause_total_ms: u64,
    pub model_name: String,
    pub model_id_prefix: String,
    pub tier: String,
    pub backend: String,
    pub serveable_models: u64,
    pub staging_error: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum EscrowStatus {
    #[default]
    Disabled,
    Ready,
    Validating,
    Held,
    Claiming,
    Degraded,
}

#[derive(Clone, Debug, Default)]
pub struct EscrowSnapshot {
    pub enabled: bool,
    pub status: EscrowStatus,
    pub claims_held: bool,
    pub validation_in_progress: bool,
    pub validation_pending_blocks: u64,
    pub tracked_live_outputs: u64,
    /// Gross live amount before the fixed per-claim transaction fee.
    pub pending_live_outputs: u64,
    pub pending_gross_sompi: u64,
    pub mature_outputs: u64,
    pub mature_gross_sompi: u64,
    pub quarantined_outputs: u64,
    pub in_flight_txs: u64,
    pub in_flight_outputs: u64,
    pub in_flight_gross_sompi: u64,
    pub claim_attempts: u64,
    pub claim_timeouts: u64,
    /// Retriable submit rejections, not permanent losses.
    pub claim_rejections: u64,
    pub orphan_rejections: u64,
    pub sequence_lock_rejections: u64,
    pub unknown_rejections: u64,
    pub claims_accepted: u64,
    pub accepted_outputs: u64,
    pub accepted_gross_sompi: u64,
    pub accepted_net_sompi: u64,
    pub terminal_slashed_outputs: u64,
    pub discarded_red_outputs: u64,
    pub discarded_ghost_outputs: u64,
    pub build_failures: u64,
    pub transport_failures: u64,
    pub persistence_failures: u64,
    pub last_seen_daa: Option<u64>,
    pub heartbeat_age_secs: Option<u64>,
    pub last_attempt_age_secs: Option<u64>,
    pub last_success_age_secs: Option<u64>,
    pub last_failure_age_secs: Option<u64>,
    pub last_success_outputs: u64,
    pub last_success_gross_sompi: u64,
    pub last_success_net_sompi: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ServiceBondSnapshot {
    pub available: bool,
    pub consecutive_misses: u64,
    pub last_strike_daa: Option<u64>,
    pub burned_claims: u64,
    pub burned_sompi: u64,
    pub suspended_until_daa: Option<u64>,
    pub last_heartbeat_age_secs: Option<u64>,
    pub last_failure_age_secs: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub uptime_secs: u64,
    pub mode: MiningMode,
    pub endpoint: String,
    pub connection: ConnectionState,
    pub connection_message: String,
    pub connection_latency_ms: Option<u64>,
    pub connection_generation: u64,
    pub failover_index: u64,
    pub job_sequence: u64,
    pub last_job_age_secs: Option<u64>,
    pub difficulty: Option<f64>,
    pub daa_score: Option<u64>,
    pub synced: Option<bool>,
    pub mining: MiningSnapshot,
    pub shares: ShareSnapshot,
    pub blocks: BlockSnapshot,
    pub inference: InferenceSnapshot,
    pub escrow: EscrowSnapshot,
    pub service_bond: ServiceBondSnapshot,
    pub events: Vec<RuntimeEvent>,
}

#[derive(Clone, Debug, Default)]
struct TextState {
    endpoint: String,
    connection_message: String,
    model_name: String,
    model_id_prefix: String,
    tier: String,
    inference_backend: String,
    route_gpus: Vec<u32>,
    hash_history: VecDeque<(u64, f64)>,
    latency_history: VecDeque<u64>,
}

struct RuntimeStats {
    start: Instant,
    mode: AtomicU8,
    connection: AtomicU8,
    connection_generation: AtomicU64,
    failover_index: AtomicU64,
    connected_at_ms: AtomicU64,
    connection_latency_ms: AtomicU64,
    job_sequence: AtomicU64,
    last_job_ms: AtomicU64,
    difficulty_bits: AtomicU64,
    daa_score: AtomicU64,
    synced: AtomicU8,

    mining_preparing: AtomicBool,
    mining_sample_sequence: AtomicU64,
    inference_paused: AtomicBool,
    total_hashrate_bits: AtomicU64,
    total_power_bits: AtomicU64,
    efficiency_bits: AtomicU64,

    shares_submitted: AtomicU64,
    shares_accepted: AtomicU64,
    shares_stale: AtomicU64,
    shares_low_diff: AtomicU64,
    shares_duplicate: AtomicU64,
    shares_other: AtomicU64,
    shares_pending: AtomicU64,
    last_share_accepted_ms: AtomicU64,

    blocks_found: AtomicU64,
    blocks_accepted: AtomicU64,
    blocks_rejected: AtomicU64,
    blocks_pending: AtomicU64,
    last_block_accepted_ms: AtomicU64,

    inference_requested: AtomicU64,
    inference_prepared: AtomicU64,
    inference_served: AtomicU64,
    inference_delivered: AtomicU64,
    inference_failed: AtomicU64,
    inference_busy: AtomicU64,
    inference_active: AtomicU64,
    /// Serializes queue ownership changes with connection generation changes. Queue updates are
    /// infrequent and tiny; this closes the check-then-store race where a detached task from a
    /// previous connection could otherwise write its depth after a reconnect had cleared it.
    inference_queue_update: Mutex<()>,
    inference_queue_depth: AtomicU64,
    inference_queue_capacity: AtomicU64,
    inference_last_gpu: AtomicU64,
    inference_last_latency_ms: AtomicU64,
    inference_last_tokens: AtomicU64,
    inference_pause_started_ms: AtomicU64,
    inference_pause_total_ms: AtomicU64,
    inference_serveable_models: AtomicU64,
    inference_staging_error: AtomicBool,

    escrow_enabled: AtomicBool,
    escrow_status: AtomicU8,
    escrow_held: AtomicBool,
    escrow_validating: AtomicBool,
    escrow_validation_pending: AtomicU64,
    escrow_tracked_live: AtomicU64,
    escrow_pending_outputs: AtomicU64,
    escrow_pending_gross: AtomicU64,
    escrow_mature_outputs: AtomicU64,
    escrow_mature_gross: AtomicU64,
    escrow_quarantined: AtomicU64,
    escrow_inflight_txs: AtomicU64,
    escrow_inflight_outputs: AtomicU64,
    escrow_inflight_gross: AtomicU64,
    escrow_claim_attempts: AtomicU64,
    escrow_claim_timeouts: AtomicU64,
    escrow_claim_rejections: AtomicU64,
    escrow_orphan_rejections: AtomicU64,
    escrow_sequence_rejections: AtomicU64,
    escrow_unknown_rejections: AtomicU64,
    escrow_claims_accepted: AtomicU64,
    escrow_accepted_outputs: AtomicU64,
    escrow_accepted_gross: AtomicU64,
    escrow_accepted_net: AtomicU64,
    escrow_terminal_slashed: AtomicU64,
    escrow_discarded_red: AtomicU64,
    escrow_discarded_ghost: AtomicU64,
    escrow_build_failures: AtomicU64,
    escrow_transport_failures: AtomicU64,
    escrow_persistence_failures: AtomicU64,
    escrow_persistence_degraded: AtomicBool,
    escrow_last_seen_daa: AtomicU64,
    escrow_heartbeat_ms: AtomicU64,
    escrow_last_attempt_ms: AtomicU64,
    escrow_last_success_ms: AtomicU64,
    escrow_last_failure_ms: AtomicU64,
    escrow_last_success_outputs: AtomicU64,
    escrow_last_success_gross: AtomicU64,
    escrow_last_success_net: AtomicU64,

    service_available: AtomicBool,
    service_misses: AtomicU64,
    service_last_strike_daa: AtomicU64,
    service_burned_claims: AtomicU64,
    service_burned_sompi: AtomicU64,
    service_suspended_until: AtomicU64,
    service_heartbeat_ms: AtomicU64,
    service_failure_ms: AtomicU64,

    text: RwLock<TextState>,
    devices: RwLock<Vec<DeviceSnapshot>>,
    events: Mutex<VecDeque<RuntimeEvent>>,
}

impl RuntimeStats {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            mode: AtomicU8::new(MiningMode::Unknown as u8),
            connection: AtomicU8::new(ConnectionState::Offline as u8),
            connection_generation: AtomicU64::new(0),
            failover_index: AtomicU64::new(0),
            connected_at_ms: AtomicU64::new(UNSET),
            connection_latency_ms: AtomicU64::new(UNSET),
            job_sequence: AtomicU64::new(0),
            last_job_ms: AtomicU64::new(UNSET),
            difficulty_bits: AtomicU64::new(UNSET),
            daa_score: AtomicU64::new(UNSET),
            synced: AtomicU8::new(0),
            mining_preparing: AtomicBool::new(true),
            mining_sample_sequence: AtomicU64::new(0),
            inference_paused: AtomicBool::new(false),
            total_hashrate_bits: AtomicU64::new(0f64.to_bits()),
            total_power_bits: AtomicU64::new(UNSET),
            efficiency_bits: AtomicU64::new(UNSET),
            shares_submitted: AtomicU64::new(0),
            shares_accepted: AtomicU64::new(0),
            shares_stale: AtomicU64::new(0),
            shares_low_diff: AtomicU64::new(0),
            shares_duplicate: AtomicU64::new(0),
            shares_other: AtomicU64::new(0),
            shares_pending: AtomicU64::new(0),
            last_share_accepted_ms: AtomicU64::new(UNSET),
            blocks_found: AtomicU64::new(0),
            blocks_accepted: AtomicU64::new(0),
            blocks_rejected: AtomicU64::new(0),
            blocks_pending: AtomicU64::new(0),
            last_block_accepted_ms: AtomicU64::new(UNSET),
            inference_requested: AtomicU64::new(0),
            inference_prepared: AtomicU64::new(0),
            inference_served: AtomicU64::new(0),
            inference_delivered: AtomicU64::new(0),
            inference_failed: AtomicU64::new(0),
            inference_busy: AtomicU64::new(0),
            inference_active: AtomicU64::new(0),
            inference_queue_update: Mutex::new(()),
            inference_queue_depth: AtomicU64::new(0),
            inference_queue_capacity: AtomicU64::new(0),
            inference_last_gpu: AtomicU64::new(UNSET),
            inference_last_latency_ms: AtomicU64::new(UNSET),
            inference_last_tokens: AtomicU64::new(UNSET),
            inference_pause_started_ms: AtomicU64::new(UNSET),
            inference_pause_total_ms: AtomicU64::new(0),
            inference_serveable_models: AtomicU64::new(0),
            inference_staging_error: AtomicBool::new(false),
            escrow_enabled: AtomicBool::new(false),
            escrow_status: AtomicU8::new(EscrowStatus::Disabled as u8),
            escrow_held: AtomicBool::new(false),
            escrow_validating: AtomicBool::new(false),
            escrow_validation_pending: AtomicU64::new(0),
            escrow_tracked_live: AtomicU64::new(0),
            escrow_pending_outputs: AtomicU64::new(0),
            escrow_pending_gross: AtomicU64::new(0),
            escrow_mature_outputs: AtomicU64::new(0),
            escrow_mature_gross: AtomicU64::new(0),
            escrow_quarantined: AtomicU64::new(0),
            escrow_inflight_txs: AtomicU64::new(0),
            escrow_inflight_outputs: AtomicU64::new(0),
            escrow_inflight_gross: AtomicU64::new(0),
            escrow_claim_attempts: AtomicU64::new(0),
            escrow_claim_timeouts: AtomicU64::new(0),
            escrow_claim_rejections: AtomicU64::new(0),
            escrow_orphan_rejections: AtomicU64::new(0),
            escrow_sequence_rejections: AtomicU64::new(0),
            escrow_unknown_rejections: AtomicU64::new(0),
            escrow_claims_accepted: AtomicU64::new(0),
            escrow_accepted_outputs: AtomicU64::new(0),
            escrow_accepted_gross: AtomicU64::new(0),
            escrow_accepted_net: AtomicU64::new(0),
            escrow_terminal_slashed: AtomicU64::new(0),
            escrow_discarded_red: AtomicU64::new(0),
            escrow_discarded_ghost: AtomicU64::new(0),
            escrow_build_failures: AtomicU64::new(0),
            escrow_transport_failures: AtomicU64::new(0),
            escrow_persistence_failures: AtomicU64::new(0),
            escrow_persistence_degraded: AtomicBool::new(false),
            escrow_last_seen_daa: AtomicU64::new(UNSET),
            escrow_heartbeat_ms: AtomicU64::new(UNSET),
            escrow_last_attempt_ms: AtomicU64::new(UNSET),
            escrow_last_success_ms: AtomicU64::new(UNSET),
            escrow_last_failure_ms: AtomicU64::new(UNSET),
            escrow_last_success_outputs: AtomicU64::new(0),
            escrow_last_success_gross: AtomicU64::new(0),
            escrow_last_success_net: AtomicU64::new(0),
            service_available: AtomicBool::new(false),
            service_misses: AtomicU64::new(0),
            service_last_strike_daa: AtomicU64::new(UNSET),
            service_burned_claims: AtomicU64::new(0),
            service_burned_sompi: AtomicU64::new(0),
            service_suspended_until: AtomicU64::new(UNSET),
            service_heartbeat_ms: AtomicU64::new(UNSET),
            service_failure_ms: AtomicU64::new(UNSET),
            text: RwLock::new(TextState::default()),
            devices: RwLock::new(Vec::new()),
            events: Mutex::new(VecDeque::with_capacity(EVENT_CAPACITY)),
        }
    }

    #[inline]
    fn now_ms(&self) -> u64 {
        self.start.elapsed().as_millis().min(u64::MAX as u128 - 1) as u64
    }

    fn event(&self, kind: EventKind, message: &str) {
        let Ok(mut events) = self.events.try_lock() else {
            return;
        };
        if events.len() == EVENT_CAPACITY {
            events.pop_front();
        }
        events.push_back(RuntimeEvent { uptime_ms: self.now_ms(), kind, message: sanitize_message(message) });
    }
}

fn hub() -> &'static RuntimeStats {
    static HUB: OnceLock<RuntimeStats> = OnceLock::new();
    HUB.get_or_init(RuntimeStats::new)
}

#[inline]
fn store_optional_u64(dst: &AtomicU64, value: Option<u64>) {
    dst.store(value.unwrap_or(UNSET), Ordering::Relaxed);
}

#[inline]
fn load_optional_u64(src: &AtomicU64) -> Option<u64> {
    match src.load(Ordering::Relaxed) {
        UNSET => None,
        value => Some(value),
    }
}

#[inline]
fn store_optional_f64(dst: &AtomicU64, value: Option<f64>) {
    let value = value.filter(|v| v.is_finite()).map(f64::to_bits).unwrap_or(UNSET);
    dst.store(value, Ordering::Relaxed);
}

#[inline]
fn load_optional_f64(src: &AtomicU64) -> Option<f64> {
    match src.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f64::from_bits(bits)),
    }
}

#[inline]
fn saturating_decrement(value: &AtomicU64) {
    let _ = value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(1)));
}

fn age_secs(now_ms: u64, at_ms: Option<u64>) -> Option<u64> {
    at_ms.map(|at| now_ms.saturating_sub(at) / 1_000)
}

fn sanitize_message(message: &str) -> String {
    message.chars().filter(|c| !c.is_control() || *c == ' ').take(200).collect::<String>()
}

/// Append a sanitized, bounded activity event without waiting for the frontend reader.
/// Callers must pass an operator-facing summary, never raw peer data or identifying values.
pub fn record_event(kind: EventKind, message: &str) {
    hub().event(kind, message);
}

/// Reduce an endpoint to scheme + authority and strip user-info, path, query and fragment.
pub fn sanitize_endpoint(endpoint: &str) -> String {
    let value = endpoint.trim();
    let (scheme, rest) = value.split_once("://").map_or(("", value), |(s, r)| (s, r));
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let authority = authority.rsplit_once('@').map_or(authority, |(_, host)| host);
    if scheme.is_empty() {
        authority.to_string()
    } else {
        format!("{}://{}", scheme, authority)
    }
}

/// Begin a new connection generation. Later updates carrying an older generation are ignored.
pub fn begin_connection(mode: MiningMode, endpoint: &str, failover_index: usize) -> u64 {
    let stats = hub();
    // Advance the owner and reset its connection-local queue under one lock. Without this
    // serialization, an old detached worker can pass a generation check, get descheduled, and
    // then overwrite the freshly-cleared gauges after this reconnect begins.
    let generation = {
        let _queue_update = stats.inference_queue_update.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let generation = stats.connection_generation.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        stats.inference_queue_depth.store(0, Ordering::Relaxed);
        stats.inference_queue_capacity.store(0, Ordering::Relaxed);
        generation
    };
    stats.mode.store(mode as u8, Ordering::Relaxed);
    stats.connection.store(ConnectionState::Connecting as u8, Ordering::Relaxed);
    stats.failover_index.store(failover_index as u64, Ordering::Relaxed);
    stats.connected_at_ms.store(UNSET, Ordering::Relaxed);
    stats.connection_latency_ms.store(UNSET, Ordering::Relaxed);
    stats.last_job_ms.store(UNSET, Ordering::Relaxed);
    stats.difficulty_bits.store(UNSET, Ordering::Relaxed);
    stats.daa_score.store(UNSET, Ordering::Relaxed);
    stats.synced.store(0, Ordering::Relaxed);
    stats.shares_pending.store(0, Ordering::Relaxed);
    stats.blocks_pending.store(0, Ordering::Relaxed);
    if let Ok(mut text) = stats.text.try_write() {
        text.endpoint = sanitize_endpoint(endpoint);
        text.connection_message = "connecting".to_string();
    }
    stats.event(EventKind::Info, "Connecting to mining endpoint");
    generation
}

fn current_generation(stats: &RuntimeStats, generation: u64) -> bool {
    stats.connection_generation.load(Ordering::Relaxed) == generation
}

pub fn connection_established(generation: u64, latency_ms: Option<u64>) {
    let stats = hub();
    if !current_generation(stats, generation) {
        return;
    }
    let failover = stats.failover_index.load(Ordering::Relaxed) != 0;
    stats.connection.store(
        if failover { ConnectionState::Failover as u8 } else { ConnectionState::Connected as u8 },
        Ordering::Relaxed,
    );
    stats.connected_at_ms.store(stats.now_ms(), Ordering::Relaxed);
    store_optional_u64(&stats.connection_latency_ms, latency_ms);
    if let Ok(mut text) = stats.text.try_write() {
        text.connection_message = if failover { "connected to backup" } else { "connected" }.to_string();
    }
    stats.event(
        EventKind::Info,
        if failover { "Backup mining connection established" } else { "Mining connection established" },
    );
}

pub fn connection_lost(generation: u64, safe_reason: &'static str) {
    let stats = hub();
    let _queue_update = stats.inference_queue_update.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if !current_generation(stats, generation) {
        return;
    }
    stats.inference_queue_depth.store(0, Ordering::Relaxed);
    stats.inference_queue_capacity.store(0, Ordering::Relaxed);
    stats.connection.store(ConnectionState::Offline as u8, Ordering::Relaxed);
    stats.shares_pending.store(0, Ordering::Relaxed);
    stats.blocks_pending.store(0, Ordering::Relaxed);
    if let Ok(mut text) = stats.text.try_write() {
        text.connection_message = sanitize_message(safe_reason);
    }
    stats.event(EventKind::Error, safe_reason);
}

pub fn record_job(generation: u64, daa_score: Option<u64>, synced: Option<bool>) {
    let stats = hub();
    if !current_generation(stats, generation) {
        return;
    }
    stats.job_sequence.fetch_add(1, Ordering::Relaxed);
    stats.last_job_ms.store(stats.now_ms(), Ordering::Relaxed);
    if let Some(daa) = daa_score {
        stats.daa_score.store(daa, Ordering::Relaxed);
    }
    if let Some(synced) = synced {
        stats.synced.store(if synced { 2 } else { 1 }, Ordering::Relaxed);
    }
}

pub fn set_difficulty(difficulty: f64) {
    store_optional_f64(&hub().difficulty_bits, Some(difficulty));
}

pub fn record_share_submitted() {
    let stats = hub();
    stats.shares_submitted.fetch_add(1, Ordering::Relaxed);
    stats.shares_pending.fetch_add(1, Ordering::Relaxed);
}

pub fn record_share_abandoned() {
    saturating_decrement(&hub().shares_pending);
}

fn share_event_message(action: &'static str, device_id: Option<u32>) -> String {
    device_id.map_or_else(|| action.to_owned(), |gpu| format!("{action} (GPU{gpu})"))
}

pub fn record_share_accepted(device_id: Option<u32>) {
    let stats = hub();
    saturating_decrement(&stats.shares_pending);
    stats.shares_accepted.fetch_add(1, Ordering::Relaxed);
    stats.last_share_accepted_ms.store(stats.now_ms(), Ordering::Relaxed);
    stats.event(EventKind::ShareAccepted, &share_event_message("Share accepted", device_id));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShareRejectKind {
    Stale,
    LowDifficulty,
    Duplicate,
    Other,
}

pub fn record_share_rejected(kind: ShareRejectKind, device_id: Option<u32>) {
    let stats = hub();
    saturating_decrement(&stats.shares_pending);
    match kind {
        ShareRejectKind::Stale => &stats.shares_stale,
        ShareRejectKind::LowDifficulty => &stats.shares_low_diff,
        ShareRejectKind::Duplicate => &stats.shares_duplicate,
        ShareRejectKind::Other => &stats.shares_other,
    }
    .fetch_add(1, Ordering::Relaxed);
    stats.event(EventKind::ShareRejected, &share_event_message("Share rejected", device_id));
}

/// A `FullBlock` candidate was successfully queued to the solo node connection.
pub fn record_solo_block_found() {
    let stats = hub();
    stats.blocks_found.fetch_add(1, Ordering::Relaxed);
    stats.blocks_pending.fetch_add(1, Ordering::Relaxed);
    stats.event(EventKind::BlockFound, "Solo block candidate found");
}

pub fn record_solo_block_accepted() {
    let stats = hub();
    saturating_decrement(&stats.blocks_pending);
    stats.blocks_accepted.fetch_add(1, Ordering::Relaxed);
    stats.last_block_accepted_ms.store(stats.now_ms(), Ordering::Relaxed);
    stats.event(EventKind::BlockAccepted, "Solo block accepted");
}

pub fn record_solo_block_rejected() {
    let stats = hub();
    saturating_decrement(&stats.blocks_pending);
    stats.blocks_rejected.fetch_add(1, Ordering::Relaxed);
    stats.event(EventKind::BlockRejected, "Solo block rejected");
}

pub fn publish_mining_snapshot(
    total_hashrate_hs: f64,
    total_power_w: Option<f64>,
    efficiency_mhs_per_w: Option<f64>,
    preparing: bool,
    inference_paused: bool,
    mut devices: Vec<DeviceSnapshot>,
) {
    let stats = hub();
    stats.total_hashrate_bits.store(total_hashrate_hs.max(0.0).to_bits(), Ordering::Relaxed);
    store_optional_f64(&stats.total_power_bits, total_power_w);
    store_optional_f64(&stats.efficiency_bits, efficiency_mhs_per_w);
    stats.mining_preparing.store(preparing, Ordering::Relaxed);
    stats.inference_paused.store(inference_paused, Ordering::Relaxed);
    devices.sort_by_key(|device| device.index);
    if let Ok(mut current) = stats.devices.try_write() {
        *current = devices;
        stats.mining_sample_sequence.fetch_add(1, Ordering::Relaxed);
    }
    if let Ok(mut text) = stats.text.try_write() {
        text.hash_history.push_back((stats.now_ms(), total_hashrate_hs.max(0.0)));
        while text.hash_history.len() > HASHRATE_WINDOW {
            text.hash_history.pop_front();
        }
    }
}

/// Start a unique valid external inference workflow. Self-tests must never call this.
pub fn begin_inference(kind: InferenceKind, model_id: Option<&[u8; 32]>) -> InferenceAttempt {
    let stats = hub();
    stats.inference_requested.fetch_add(1, Ordering::Relaxed);
    stats.inference_active.fetch_add(1, Ordering::Relaxed);
    if let Some(id) = model_id {
        if let Ok(mut text) = stats.text.try_write() {
            text.model_id_prefix = id[..4].iter().map(|byte| format!("{byte:02x}")).collect();
        }
    }
    InferenceAttempt { kind, started: Instant::now(), gpu: None, finished: false }
}

/// Count a valid external request rejected before an attempt object could be created.
pub fn record_inference_failed_request(kind: InferenceKind) {
    let mut attempt = begin_inference(kind, None);
    attempt.failed();
}

/// Count a valid external request rejected because all bounded capacity was occupied.
pub fn record_inference_busy_request(kind: InferenceKind) {
    let mut attempt = begin_inference(kind, None);
    attempt.busy();
}

/// Account for valid requests accepted into a protocol queue but lost before execution began.
/// They were not active attempts, so update the lifetime request/failure totals directly instead
/// of briefly perturbing the active gauge during connection teardown.
pub fn record_inference_abandoned_requests(kind: InferenceKind, count: usize) {
    if count == 0 {
        return;
    }
    let stats = hub();
    let count = count as u64;
    stats.inference_requested.fetch_add(count, Ordering::Relaxed);
    stats.inference_failed.fetch_add(count, Ordering::Relaxed);
    let message = match kind {
        InferenceKind::SoloRequest => "Queued solo inference abandoned on disconnect",
        InferenceKind::PoolTask => "Queued pool inference abandoned on disconnect",
        _ => "Queued inference abandoned on disconnect",
    };
    stats.event(EventKind::InferenceFailed, message);
}

pub struct InferenceAttempt {
    kind: InferenceKind,
    started: Instant,
    gpu: Option<u32>,
    finished: bool,
}

impl std::fmt::Debug for InferenceAttempt {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InferenceAttempt")
            .field("kind", &self.kind)
            .field("gpu", &self.gpu)
            .field("finished", &self.finished)
            .finish()
    }
}

impl InferenceAttempt {
    pub fn set_gpu(&mut self, gpu: usize) {
        self.gpu = u32::try_from(gpu).ok();
        if let Some(gpu) = self.gpu {
            hub().inference_last_gpu.store(gpu as u64, Ordering::Relaxed);
        }
    }

    pub fn served(&mut self, tokens: usize) {
        if self.finished {
            return;
        }
        let stats = hub();
        let latency = self.started.elapsed().as_millis().min(u64::MAX as u128 - 1) as u64;
        stats.inference_served.fetch_add(1, Ordering::Relaxed);
        stats.inference_last_latency_ms.store(latency, Ordering::Relaxed);
        stats.inference_last_tokens.store(tokens as u64, Ordering::Relaxed);
        if let Ok(mut text) = stats.text.try_write() {
            text.latency_history.push_back(latency);
            while text.latency_history.len() > LATENCY_WINDOW {
                text.latency_history.pop_front();
            }
        }
        self.finish();
        stats.event(EventKind::InferenceOk, "Inference completed");
    }

    pub fn failed(&mut self) {
        if self.finished {
            return;
        }
        let stats = hub();
        stats.inference_failed.fetch_add(1, Ordering::Relaxed);
        self.finish();
        stats.event(EventKind::InferenceFailed, "Inference failed");
    }

    pub fn busy(&mut self) {
        if self.finished {
            return;
        }
        let stats = hub();
        stats.inference_busy.fetch_add(1, Ordering::Relaxed);
        self.finish();
        stats.event(EventKind::InferenceFailed, "Inference capacity busy");
    }

    fn finish(&mut self) {
        if !self.finished {
            saturating_decrement(&hub().inference_active);
            self.finished = true;
        }
    }
}

impl Drop for InferenceAttempt {
    fn drop(&mut self) {
        // A panic/cancelled task is an inference failure, not a permanently active request.
        self.failed();
    }
}

pub fn record_inference_prepared() {
    hub().inference_prepared.fetch_add(1, Ordering::Relaxed);
}

pub fn record_inference_delivered() {
    hub().inference_delivered.fetch_add(1, Ordering::Relaxed);
}

/// Publish the queue owned by one mining connection.
///
/// Returns `false` when `generation` is stale or that connection is already offline. The ownership
/// check and stores are serialized with [`begin_connection`], so a detached task from an old or
/// closed connection can neither replace nor clear a newer connection's queue gauges.
pub fn set_connection_inference_queue(generation: u64, depth: usize, capacity: usize) -> bool {
    let stats = hub();
    let _queue_update = stats.inference_queue_update.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if !current_generation(stats, generation)
        || ConnectionState::from_u8(stats.connection.load(Ordering::Relaxed)) == ConnectionState::Offline
    {
        return false;
    }
    stats.inference_queue_depth.store(depth as u64, Ordering::Relaxed);
    stats.inference_queue_capacity.store(capacity as u64, Ordering::Relaxed);
    true
}

/// Clear queue gauges only if this is still the active connection generation.
pub fn clear_connection_inference_queue(generation: u64) -> bool {
    set_connection_inference_queue(generation, 0, 0)
}

pub fn inference_pause_started() {
    let stats = hub();
    stats.inference_paused.store(true, Ordering::Relaxed);
    let _ =
        stats.inference_pause_started_ms.compare_exchange(UNSET, stats.now_ms(), Ordering::Relaxed, Ordering::Relaxed);
}

pub fn inference_pause_ended() {
    let stats = hub();
    stats.inference_paused.store(false, Ordering::Relaxed);
    let started = stats.inference_pause_started_ms.swap(UNSET, Ordering::Relaxed);
    if started != UNSET {
        stats.inference_pause_total_ms.fetch_add(stats.now_ms().saturating_sub(started), Ordering::Relaxed);
    }
}

pub fn set_inference_model_status(
    name: &str,
    model_id: &[u8; 32],
    tier: &str,
    backend: &str,
    route_gpus: &[usize],
    serveable_models: usize,
    staging_error: bool,
) {
    let stats = hub();
    stats.inference_serveable_models.store(serveable_models as u64, Ordering::Relaxed);
    stats.inference_staging_error.store(staging_error, Ordering::Relaxed);
    if let Ok(mut text) = stats.text.try_write() {
        text.model_name = sanitize_message(name);
        text.model_id_prefix = model_id[..4].iter().map(|byte| format!("{byte:02x}")).collect();
        text.tier = sanitize_message(tier);
        text.inference_backend = sanitize_message(backend);
        text.route_gpus = route_gpus.iter().filter_map(|gpu| u32::try_from(*gpu).ok()).collect();
        text.route_gpus.sort_unstable();
        text.route_gpus.dedup();
    }
}

pub fn set_staging_error(present: bool) {
    hub().inference_staging_error.store(present, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EscrowGauges {
    pub held: bool,
    pub validation_in_progress: bool,
    pub validation_pending_blocks: u64,
    pub tracked_live_outputs: u64,
    pub pending_live_outputs: u64,
    pub pending_gross_sompi: u64,
    pub mature_outputs: u64,
    pub mature_gross_sompi: u64,
    pub quarantined_outputs: u64,
    pub in_flight_txs: u64,
    pub in_flight_outputs: u64,
    pub in_flight_gross_sompi: u64,
    pub last_seen_daa: u64,
}

fn refresh_escrow_status(stats: &RuntimeStats) {
    let status = if !stats.escrow_enabled.load(Ordering::Relaxed) {
        EscrowStatus::Disabled
    } else if stats.escrow_persistence_degraded.load(Ordering::Relaxed) {
        EscrowStatus::Degraded
    } else if stats.escrow_held.load(Ordering::Relaxed) {
        EscrowStatus::Held
    } else if stats.escrow_validating.load(Ordering::Relaxed) {
        EscrowStatus::Validating
    } else if stats.escrow_inflight_txs.load(Ordering::Relaxed) != 0 {
        EscrowStatus::Claiming
    } else {
        EscrowStatus::Ready
    };
    stats.escrow_status.store(status as u8, Ordering::Relaxed);
}

pub fn escrow_enabled(enabled: bool) {
    let stats = hub();
    stats.escrow_enabled.store(enabled, Ordering::Relaxed);
    if !enabled {
        stats.escrow_held.store(false, Ordering::Relaxed);
        stats.escrow_validating.store(false, Ordering::Relaxed);
        stats.escrow_validation_pending.store(0, Ordering::Relaxed);
        stats.escrow_tracked_live.store(0, Ordering::Relaxed);
        stats.escrow_pending_outputs.store(0, Ordering::Relaxed);
        stats.escrow_pending_gross.store(0, Ordering::Relaxed);
        stats.escrow_mature_outputs.store(0, Ordering::Relaxed);
        stats.escrow_mature_gross.store(0, Ordering::Relaxed);
        stats.escrow_quarantined.store(0, Ordering::Relaxed);
        stats.escrow_inflight_txs.store(0, Ordering::Relaxed);
        stats.escrow_inflight_outputs.store(0, Ordering::Relaxed);
        stats.escrow_inflight_gross.store(0, Ordering::Relaxed);
        stats.escrow_persistence_degraded.store(false, Ordering::Relaxed);
    }
    refresh_escrow_status(stats);
}

pub fn escrow_claims_held(held: bool) {
    let stats = hub();
    stats.escrow_held.store(held, Ordering::Relaxed);
    refresh_escrow_status(stats);
}

pub fn escrow_validation_progress(pending_blocks: usize) {
    let stats = hub();
    let validating = pending_blocks != 0;
    stats.escrow_validating.store(validating, Ordering::Relaxed);
    stats.escrow_validation_pending.store(pending_blocks as u64, Ordering::Relaxed);
    refresh_escrow_status(stats);
}

pub fn publish_escrow_gauges(gauges: EscrowGauges) {
    let stats = hub();
    stats.escrow_enabled.store(true, Ordering::Relaxed);
    stats.escrow_held.store(gauges.held, Ordering::Relaxed);
    stats.escrow_validating.store(gauges.validation_in_progress, Ordering::Relaxed);
    stats.escrow_validation_pending.store(gauges.validation_pending_blocks, Ordering::Relaxed);
    stats.escrow_tracked_live.store(gauges.tracked_live_outputs, Ordering::Relaxed);
    stats.escrow_pending_outputs.store(gauges.pending_live_outputs, Ordering::Relaxed);
    stats.escrow_pending_gross.store(gauges.pending_gross_sompi, Ordering::Relaxed);
    stats.escrow_mature_outputs.store(gauges.mature_outputs, Ordering::Relaxed);
    stats.escrow_mature_gross.store(gauges.mature_gross_sompi, Ordering::Relaxed);
    stats.escrow_quarantined.store(gauges.quarantined_outputs, Ordering::Relaxed);
    stats.escrow_inflight_txs.store(gauges.in_flight_txs, Ordering::Relaxed);
    stats.escrow_inflight_outputs.store(gauges.in_flight_outputs, Ordering::Relaxed);
    stats.escrow_inflight_gross.store(gauges.in_flight_gross_sompi, Ordering::Relaxed);
    if gauges.last_seen_daa != 0 {
        stats.escrow_last_seen_daa.store(gauges.last_seen_daa, Ordering::Relaxed);
    }
    refresh_escrow_status(stats);
}

pub fn escrow_heartbeat(daa_score: u64) {
    let stats = hub();
    stats.escrow_heartbeat_ms.store(stats.now_ms(), Ordering::Relaxed);
    stats.escrow_last_seen_daa.store(daa_score, Ordering::Relaxed);
}

pub fn escrow_claim_attempt(outputs: u64, gross_sompi: u64) {
    let stats = hub();
    stats.escrow_claim_attempts.fetch_add(1, Ordering::Relaxed);
    stats.escrow_last_attempt_ms.store(stats.now_ms(), Ordering::Relaxed);
    stats.escrow_inflight_txs.fetch_add(1, Ordering::Relaxed);
    stats.escrow_inflight_outputs.fetch_add(outputs, Ordering::Relaxed);
    stats.escrow_inflight_gross.fetch_add(gross_sompi, Ordering::Relaxed);
    refresh_escrow_status(stats);
    stats.event(EventKind::Escrow, "Escrow claim submitted");
}

fn escrow_remove_inflight(outputs: u64, gross_sompi: u64) {
    let stats = hub();
    saturating_decrement(&stats.escrow_inflight_txs);
    let _ = stats
        .escrow_inflight_outputs
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(outputs)));
    let _ = stats
        .escrow_inflight_gross
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(gross_sompi)));
    refresh_escrow_status(stats);
}

pub fn escrow_claim_timeout(outputs: u64, gross_sompi: u64) {
    let stats = hub();
    escrow_remove_inflight(outputs, gross_sompi);
    stats.escrow_claim_timeouts.fetch_add(1, Ordering::Relaxed);
    stats.escrow_last_failure_ms.store(stats.now_ms(), Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EscrowRejectKind {
    Orphan,
    SequenceLock,
    Unknown,
}

pub fn escrow_claim_rejected(kind: EscrowRejectKind, outputs: u64, gross_sompi: u64, terminal_slashed_outputs: u64) {
    let stats = hub();
    escrow_remove_inflight(outputs, gross_sompi);
    stats.escrow_claim_rejections.fetch_add(1, Ordering::Relaxed);
    match kind {
        EscrowRejectKind::Orphan => &stats.escrow_orphan_rejections,
        EscrowRejectKind::SequenceLock => &stats.escrow_sequence_rejections,
        EscrowRejectKind::Unknown => &stats.escrow_unknown_rejections,
    }
    .fetch_add(1, Ordering::Relaxed);
    stats.escrow_terminal_slashed.fetch_add(terminal_slashed_outputs, Ordering::Relaxed);
    stats.escrow_last_failure_ms.store(stats.now_ms(), Ordering::Relaxed);
    stats.event(EventKind::Escrow, "Escrow claim rejected for retry");
}

pub fn escrow_claim_accepted(outputs: u64, gross_sompi: u64, fee_sompi: u64) {
    let stats = hub();
    let net_sompi = gross_sompi.saturating_sub(fee_sompi);
    escrow_remove_inflight(outputs, gross_sompi);
    stats.escrow_claims_accepted.fetch_add(1, Ordering::Relaxed);
    stats.escrow_accepted_outputs.fetch_add(outputs, Ordering::Relaxed);
    stats.escrow_accepted_gross.fetch_add(gross_sompi, Ordering::Relaxed);
    stats.escrow_accepted_net.fetch_add(net_sompi, Ordering::Relaxed);
    stats.escrow_last_success_ms.store(stats.now_ms(), Ordering::Relaxed);
    stats.escrow_last_success_outputs.store(outputs, Ordering::Relaxed);
    stats.escrow_last_success_gross.store(gross_sompi, Ordering::Relaxed);
    stats.escrow_last_success_net.store(net_sompi, Ordering::Relaxed);
    stats.event(EventKind::Escrow, "Escrow claim accepted");
}

pub fn escrow_discarded_red(outputs: u64) {
    hub().escrow_discarded_red.fetch_add(outputs, Ordering::Relaxed);
}

pub fn escrow_discarded_ghost(outputs: u64) {
    hub().escrow_discarded_ghost.fetch_add(outputs, Ordering::Relaxed);
}

pub fn escrow_build_failed() {
    let stats = hub();
    stats.escrow_build_failures.fetch_add(1, Ordering::Relaxed);
    stats.escrow_last_failure_ms.store(stats.now_ms(), Ordering::Relaxed);
}

pub fn escrow_transport_failed() {
    let stats = hub();
    stats.escrow_transport_failures.fetch_add(1, Ordering::Relaxed);
    stats.escrow_last_failure_ms.store(stats.now_ms(), Ordering::Relaxed);
}

pub fn escrow_persistence_result(success: bool) {
    let stats = hub();
    if success {
        stats.escrow_persistence_degraded.store(false, Ordering::Relaxed);
        refresh_escrow_status(stats);
    } else {
        stats.escrow_persistence_degraded.store(true, Ordering::Relaxed);
        stats.escrow_persistence_failures.fetch_add(1, Ordering::Relaxed);
        stats.escrow_last_failure_ms.store(stats.now_ms(), Ordering::Relaxed);
        refresh_escrow_status(stats);
        stats.event(EventKind::Error, "Escrow state persistence failed");
    }
}

pub fn service_bond_update(
    consecutive_misses: u64,
    last_strike_daa: Option<u64>,
    burned_claims: u64,
    burned_sompi: u64,
    suspended_until_daa: Option<u64>,
) {
    let stats = hub();
    stats.service_available.store(true, Ordering::Relaxed);
    stats.service_misses.store(consecutive_misses, Ordering::Relaxed);
    store_optional_u64(&stats.service_last_strike_daa, last_strike_daa);
    stats.service_burned_claims.store(burned_claims, Ordering::Relaxed);
    stats.service_burned_sompi.store(burned_sompi, Ordering::Relaxed);
    store_optional_u64(&stats.service_suspended_until, suspended_until_daa);
    stats.service_heartbeat_ms.store(stats.now_ms(), Ordering::Relaxed);
}

pub fn service_bond_unavailable() {
    let stats = hub();
    stats.service_available.store(false, Ordering::Relaxed);
    stats.service_failure_ms.store(stats.now_ms(), Ordering::Relaxed);
}

/// Nonblocking snapshot. Keep the previous UI frame if this returns `None` during a rare writer.
pub fn try_snapshot() -> Option<Snapshot> {
    let stats = hub();
    let text = stats.text.try_read().ok()?.clone();
    let devices = stats.devices.try_read().ok()?.clone();
    let events = stats.events.try_lock().ok()?.iter().cloned().collect();
    let now_ms = stats.now_ms();
    let p95_latency_ms = if text.latency_history.is_empty() {
        None
    } else {
        let mut values: Vec<u64> = text.latency_history.iter().copied().collect();
        values.sort_unstable();
        Some(values[((values.len() - 1) * 95) / 100])
    };
    let recent_rates: Vec<f64> = text
        .hash_history
        .iter()
        .filter(|(at, _)| now_ms.saturating_sub(*at) <= 60_000)
        .map(|(_, rate)| *rate)
        .collect();
    let average_60s_hs =
        (!recent_rates.is_empty()).then(|| recent_rates.iter().sum::<f64>() / recent_rates.len() as f64);
    let synced = match stats.synced.load(Ordering::Relaxed) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    };
    let escrow_status = match stats.escrow_status.load(Ordering::Relaxed) {
        1 => EscrowStatus::Ready,
        2 => EscrowStatus::Validating,
        3 => EscrowStatus::Held,
        4 => EscrowStatus::Claiming,
        5 => EscrowStatus::Degraded,
        _ => EscrowStatus::Disabled,
    };
    let pause_total = stats.inference_pause_total_ms.load(Ordering::Relaxed).saturating_add(
        load_optional_u64(&stats.inference_pause_started_ms).map(|started| now_ms.saturating_sub(started)).unwrap_or(0),
    );
    Some(Snapshot {
        uptime_secs: now_ms / 1_000,
        mode: MiningMode::from_u8(stats.mode.load(Ordering::Relaxed)),
        endpoint: text.endpoint,
        connection: ConnectionState::from_u8(stats.connection.load(Ordering::Relaxed)),
        connection_message: text.connection_message,
        connection_latency_ms: load_optional_u64(&stats.connection_latency_ms),
        connection_generation: stats.connection_generation.load(Ordering::Relaxed),
        failover_index: stats.failover_index.load(Ordering::Relaxed),
        job_sequence: stats.job_sequence.load(Ordering::Relaxed),
        last_job_age_secs: age_secs(now_ms, load_optional_u64(&stats.last_job_ms)),
        difficulty: load_optional_f64(&stats.difficulty_bits),
        daa_score: load_optional_u64(&stats.daa_score),
        synced,
        mining: MiningSnapshot {
            sample_sequence: stats.mining_sample_sequence.load(Ordering::Relaxed),
            preparing: stats.mining_preparing.load(Ordering::Relaxed),
            inference_paused: stats.inference_paused.load(Ordering::Relaxed),
            total_hashrate_hs: f64::from_bits(stats.total_hashrate_bits.load(Ordering::Relaxed)),
            average_60s_hs,
            hashrate_history_hs: recent_rates,
            total_power_w: load_optional_f64(&stats.total_power_bits),
            efficiency_mhs_per_w: load_optional_f64(&stats.efficiency_bits),
            devices,
        },
        shares: ShareSnapshot {
            submitted: stats.shares_submitted.load(Ordering::Relaxed),
            accepted: stats.shares_accepted.load(Ordering::Relaxed),
            stale: stats.shares_stale.load(Ordering::Relaxed),
            low_diff: stats.shares_low_diff.load(Ordering::Relaxed),
            duplicate: stats.shares_duplicate.load(Ordering::Relaxed),
            other: stats.shares_other.load(Ordering::Relaxed),
            pending: stats.shares_pending.load(Ordering::Relaxed),
            last_accepted_age_secs: age_secs(now_ms, load_optional_u64(&stats.last_share_accepted_ms)),
        },
        blocks: BlockSnapshot {
            found: stats.blocks_found.load(Ordering::Relaxed),
            accepted: stats.blocks_accepted.load(Ordering::Relaxed),
            rejected: stats.blocks_rejected.load(Ordering::Relaxed),
            pending: stats.blocks_pending.load(Ordering::Relaxed),
            last_accepted_age_secs: age_secs(now_ms, load_optional_u64(&stats.last_block_accepted_ms)),
        },
        inference: InferenceSnapshot {
            requested: stats.inference_requested.load(Ordering::Relaxed),
            prepared: stats.inference_prepared.load(Ordering::Relaxed),
            served: stats.inference_served.load(Ordering::Relaxed),
            delivered: stats.inference_delivered.load(Ordering::Relaxed),
            failed: stats.inference_failed.load(Ordering::Relaxed),
            busy: stats.inference_busy.load(Ordering::Relaxed),
            active: stats.inference_active.load(Ordering::Relaxed),
            queue_depth: stats.inference_queue_depth.load(Ordering::Relaxed),
            queue_capacity: stats.inference_queue_capacity.load(Ordering::Relaxed),
            route_gpus: text.route_gpus,
            gpu_index: load_optional_u64(&stats.inference_last_gpu).and_then(|n| u32::try_from(n).ok()),
            last_latency_ms: load_optional_u64(&stats.inference_last_latency_ms),
            p95_latency_ms,
            last_tokens: load_optional_u64(&stats.inference_last_tokens),
            pow_pause_total_ms: pause_total,
            model_name: text.model_name,
            model_id_prefix: text.model_id_prefix,
            tier: text.tier,
            backend: text.inference_backend,
            serveable_models: stats.inference_serveable_models.load(Ordering::Relaxed),
            staging_error: stats.inference_staging_error.load(Ordering::Relaxed),
        },
        escrow: EscrowSnapshot {
            enabled: stats.escrow_enabled.load(Ordering::Relaxed),
            status: escrow_status,
            claims_held: stats.escrow_held.load(Ordering::Relaxed),
            validation_in_progress: stats.escrow_validating.load(Ordering::Relaxed),
            validation_pending_blocks: stats.escrow_validation_pending.load(Ordering::Relaxed),
            tracked_live_outputs: stats.escrow_tracked_live.load(Ordering::Relaxed),
            pending_live_outputs: stats.escrow_pending_outputs.load(Ordering::Relaxed),
            pending_gross_sompi: stats.escrow_pending_gross.load(Ordering::Relaxed),
            mature_outputs: stats.escrow_mature_outputs.load(Ordering::Relaxed),
            mature_gross_sompi: stats.escrow_mature_gross.load(Ordering::Relaxed),
            quarantined_outputs: stats.escrow_quarantined.load(Ordering::Relaxed),
            in_flight_txs: stats.escrow_inflight_txs.load(Ordering::Relaxed),
            in_flight_outputs: stats.escrow_inflight_outputs.load(Ordering::Relaxed),
            in_flight_gross_sompi: stats.escrow_inflight_gross.load(Ordering::Relaxed),
            claim_attempts: stats.escrow_claim_attempts.load(Ordering::Relaxed),
            claim_timeouts: stats.escrow_claim_timeouts.load(Ordering::Relaxed),
            claim_rejections: stats.escrow_claim_rejections.load(Ordering::Relaxed),
            orphan_rejections: stats.escrow_orphan_rejections.load(Ordering::Relaxed),
            sequence_lock_rejections: stats.escrow_sequence_rejections.load(Ordering::Relaxed),
            unknown_rejections: stats.escrow_unknown_rejections.load(Ordering::Relaxed),
            claims_accepted: stats.escrow_claims_accepted.load(Ordering::Relaxed),
            accepted_outputs: stats.escrow_accepted_outputs.load(Ordering::Relaxed),
            accepted_gross_sompi: stats.escrow_accepted_gross.load(Ordering::Relaxed),
            accepted_net_sompi: stats.escrow_accepted_net.load(Ordering::Relaxed),
            terminal_slashed_outputs: stats.escrow_terminal_slashed.load(Ordering::Relaxed),
            discarded_red_outputs: stats.escrow_discarded_red.load(Ordering::Relaxed),
            discarded_ghost_outputs: stats.escrow_discarded_ghost.load(Ordering::Relaxed),
            build_failures: stats.escrow_build_failures.load(Ordering::Relaxed),
            transport_failures: stats.escrow_transport_failures.load(Ordering::Relaxed),
            persistence_failures: stats.escrow_persistence_failures.load(Ordering::Relaxed),
            last_seen_daa: load_optional_u64(&stats.escrow_last_seen_daa),
            heartbeat_age_secs: age_secs(now_ms, load_optional_u64(&stats.escrow_heartbeat_ms)),
            last_attempt_age_secs: age_secs(now_ms, load_optional_u64(&stats.escrow_last_attempt_ms)),
            last_success_age_secs: age_secs(now_ms, load_optional_u64(&stats.escrow_last_success_ms)),
            last_failure_age_secs: age_secs(now_ms, load_optional_u64(&stats.escrow_last_failure_ms)),
            last_success_outputs: stats.escrow_last_success_outputs.load(Ordering::Relaxed),
            last_success_gross_sompi: stats.escrow_last_success_gross.load(Ordering::Relaxed),
            last_success_net_sompi: stats.escrow_last_success_net.load(Ordering::Relaxed),
        },
        service_bond: ServiceBondSnapshot {
            available: stats.service_available.load(Ordering::Relaxed),
            consecutive_misses: stats.service_misses.load(Ordering::Relaxed),
            last_strike_daa: load_optional_u64(&stats.service_last_strike_daa),
            burned_claims: stats.service_burned_claims.load(Ordering::Relaxed),
            burned_sompi: stats.service_burned_sompi.load(Ordering::Relaxed),
            suspended_until_daa: load_optional_u64(&stats.service_suspended_until),
            last_heartbeat_age_secs: age_secs(now_ms, load_optional_u64(&stats.service_heartbeat_ms)),
            last_failure_age_secs: age_secs(now_ms, load_optional_u64(&stats.service_failure_ms)),
        },
        events,
    })
}

/// Convenience fallback for non-animated consumers. Interactive frontends should use
/// [`try_snapshot`] so a rare lock collision keeps the previous complete frame.
pub fn snapshot() -> Snapshot {
    try_snapshot().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, MutexGuard};

    use super::{
        begin_connection, begin_inference, clear_connection_inference_queue, connection_lost, escrow_claim_accepted,
        escrow_claim_attempt, record_share_accepted, record_share_rejected, record_share_submitted, sanitize_endpoint,
        sanitize_message, set_connection_inference_queue, share_event_message, try_snapshot, ConnectionState,
        InferenceKind, MiningMode, ShareRejectKind, Snapshot,
    };

    // These tests exercise the process-wide hub and cargo runs tests in this module concurrently.
    // Serialize only those cases so one test's best-effort snapshot lock cannot make another test's
    // deliberately non-blocking event publication look as though production attribution failed.
    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    fn enter_global_test() -> MutexGuard<'static, ()> {
        TEST_SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn read_snapshot() -> Snapshot {
        for _ in 0..1_000 {
            if let Some(snapshot) = try_snapshot() {
                return snapshot;
            }
            std::thread::yield_now();
        }
        panic!("runtime snapshot remained contended");
    }

    #[test]
    fn endpoint_drops_credentials_and_paths() {
        assert_eq!(
            sanitize_endpoint("stratum+tcp://worker:secret@pool.example:5555/path?q=x"),
            "stratum+tcp://pool.example:5555"
        );
        assert_eq!(sanitize_endpoint("pool.example:5555"), "pool.example:5555");
    }

    #[test]
    fn event_messages_are_bounded_and_single_line() {
        let value = sanitize_message(&format!("hello\n{}", "x".repeat(300)));
        assert!(!value.contains('\n'));
        assert!(value.chars().count() <= 200);
    }

    #[test]
    fn share_events_use_the_actual_gpu_and_retain_a_truthful_fallback() {
        let _serial = enter_global_test();

        assert_eq!(share_event_message("Share accepted", Some(0)), "Share accepted (GPU0)");
        assert_eq!(share_event_message("Share rejected", Some(17)), "Share rejected (GPU17)");
        assert_eq!(share_event_message("Share accepted", None), "Share accepted");

        let before = read_snapshot();
        record_share_submitted();
        record_share_accepted(Some(7));
        record_share_submitted();
        record_share_rejected(ShareRejectKind::Stale, Some(3));
        let after = read_snapshot();

        assert_eq!(after.shares.submitted - before.shares.submitted, 2);
        assert_eq!(after.shares.accepted - before.shares.accepted, 1);
        assert_eq!(after.shares.stale - before.shares.stale, 1);
        assert_eq!(after.shares.pending, before.shares.pending);
        assert!(after.events.iter().any(|event| event.message == "Share accepted (GPU7)"));
        assert!(after.events.iter().any(|event| event.message == "Share rejected (GPU3)"));
    }

    #[test]
    fn inference_attempts_finish_exactly_once() {
        let _serial = enter_global_test();
        let before = read_snapshot().inference;

        let mut served = begin_inference(InferenceKind::Interactive, None);
        served.served(7);
        served.failed();
        drop(served);

        let mut busy = begin_inference(InferenceKind::PoolTask, None);
        busy.busy();
        busy.busy();
        drop(busy);

        let abandoned = begin_inference(InferenceKind::SoloRequest, None);
        drop(abandoned);

        let after = read_snapshot().inference;
        assert_eq!(after.requested - before.requested, 3);
        assert_eq!(after.served - before.served, 1);
        assert_eq!(after.busy - before.busy, 1);
        assert_eq!(after.failed - before.failed, 1);
        assert_eq!(after.active, before.active);
        assert_eq!(after.last_tokens, Some(7));
    }

    #[test]
    fn inference_queue_is_owned_by_the_current_connection_generation() {
        let _serial = enter_global_test();
        let old = begin_connection(MiningMode::Pool, "pool-old.example:5555", 0);
        assert!(set_connection_inference_queue(old, 3, 64));
        let queued = read_snapshot();
        assert_eq!(queued.inference.queue_depth, 3);
        assert_eq!(queued.inference.queue_capacity, 64);

        let current = begin_connection(MiningMode::Solo, "grpc://node.example:16110", 0);
        let reset = read_snapshot();
        assert_eq!(reset.inference.queue_depth, 0);
        assert_eq!(reset.inference.queue_capacity, 0);

        assert!(!set_connection_inference_queue(old, 63, 64));
        assert!(set_connection_inference_queue(current, 2, 256));
        assert!(!clear_connection_inference_queue(old));
        connection_lost(old, "Stale connection closed");

        let protected = read_snapshot();
        assert_eq!(protected.connection, ConnectionState::Connecting);
        assert_eq!(protected.inference.queue_depth, 2);
        assert_eq!(protected.inference.queue_capacity, 256);

        connection_lost(current, "Current connection closed");
        let cleared = read_snapshot();
        assert_eq!(cleared.connection, ConnectionState::Offline);
        assert_eq!(cleared.inference.queue_depth, 0);
        assert_eq!(cleared.inference.queue_capacity, 0);
        assert!(!set_connection_inference_queue(current, 1, 256));
        let remains_cleared = read_snapshot();
        assert_eq!(remains_cleared.inference.queue_depth, 0);
        assert_eq!(remains_cleared.inference.queue_capacity, 0);
    }

    #[test]
    fn escrow_attempt_moves_gross_amount_exactly_once() {
        let _serial = enter_global_test();
        let before = read_snapshot().escrow;
        escrow_claim_attempt(2, 100);
        let in_flight = read_snapshot().escrow;
        assert_eq!(in_flight.claim_attempts - before.claim_attempts, 1);
        assert_eq!(in_flight.in_flight_outputs - before.in_flight_outputs, 2);
        assert_eq!(in_flight.in_flight_gross_sompi - before.in_flight_gross_sompi, 100);

        escrow_claim_accepted(2, 100, 30);
        let after = read_snapshot().escrow;
        assert_eq!(after.claims_accepted - before.claims_accepted, 1);
        assert_eq!(after.accepted_outputs - before.accepted_outputs, 2);
        assert_eq!(after.accepted_gross_sompi - before.accepted_gross_sompi, 100);
        assert_eq!(after.accepted_net_sompi - before.accepted_net_sompi, 70);
        assert_eq!(after.in_flight_outputs, before.in_flight_outputs);
        assert_eq!(after.in_flight_gross_sompi, before.in_flight_gross_sompi);
    }
}
