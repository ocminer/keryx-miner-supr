use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::{FullBlock, PartialBlock};
use crate::proto::kaspad_message::Payload;
use crate::proto::rpc_client::RpcClient;
use crate::proto::{
    GetBlockRequestMessage, GetBlockTemplateRequestMessage, GetInfoRequestMessage, GetServiceStrikesRequestMessage,
    KaspadMessage, NotifyBlockAddedRequestMessage, NotifyNewBlockTemplateRequestMessage,
    NotifyVirtualSelectedParentChainChangedRequestMessage,
};
use crate::{miner::MinerManager, Error};
use keryx_miner::slm::InferenceError;

fn account_inference_result(
    result: std::result::Result<String, InferenceError>,
    attempt: &mut keryx_miner::runtime_stats::InferenceAttempt,
) -> Option<String> {
    match result {
        Ok(text) => Some(text),
        Err(InferenceError::Deferred(reason)) => {
            log::info!("OPoI: inference deferred ({}) — route unchanged", reason);
            attempt.busy();
            None
        }
        Err(InferenceError::Failed) => {
            attempt.failed();
            None
        }
    }
}

/// Max boot-time escrow-validation GetBlock requests in flight at once — each answer
/// sends the next queued one, so thousands of state entries never overwhelm the
/// HTTP/2 flow-control window or delay the mining stream.
const VALIDATION_WINDOW: usize = 64;
/// Keep queued prompts bounded even if a node repeatedly presents distinct requests faster than
/// this miner can run inference. At the protocol limit (4 KiB per prompt), this caps prompt storage
/// near one MiB while retaining enough work to absorb a burst of blocks.
const AI_REQUEST_QUEUE_CAPACITY: usize = 256;
/// Successful request identities are retained as a bounded replay filter. This is deliberately
/// larger than the live queue, so an in-flight/queued identity never needs to be evicted.
const AI_SEEN_CAPACITY: usize = 4096;
use async_trait::async_trait;
use futures_util::StreamExt;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use tokio::sync::{
    mpsc::{self, error::SendError, Sender},
    oneshot,
};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{PollSendError, PollSender};
use tonic::{transport::{Channel as TonicChannel, Endpoint}, Streaming};

static EXTRA_DATA: &str = concat!(env!("CARGO_PKG_VERSION"), "/", env!("PACKAGE_COMPILE_TIME"));
type BlockHandle = JoinHandle<Result<(), PollSendError<KaspadMessage>>>;

/// Private-inference state of a queued request (H14); `None` for a plaintext request.
#[derive(Clone, PartialEq, Eq)]
enum PrivateRequest {
    /// Opened with the escrow key; the answer is sealed under this per-responder key (key schedule
    /// 1 or 2, whichever authenticated — `keryx_inference::open_request`, node 1.6.6).
    Opened(keryx_inference::ResponseKey),
    /// Names this miner as a recipient but cannot be opened (tampered / corrupt compression):
    /// answered with a plaintext error body — a silent recipient is strikable, and a body that is
    /// not an envelope is the node-documented way to say "could not open".
    Unreadable,
}

impl std::fmt::Debug for PrivateRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrivateRequest::Opened(_) => f.write_str("Opened(<response key>)"),
            PrivateRequest::Unreadable => f.write_str("Unreadable"),
        }
    }
}

/// Body of the answer to a private request this miner could not open (upstream v0.5.6 text).
const UNREADABLE_PRIVATE_REQUEST_ANSWER: &str = "keryx: private request could not be opened by this responder";

/// Leading bytes of a private prompt compressed with raw DEFLATE (upstream v0.5.6 a9f509f).
const COMPRESSED_PROMPT_MAGIC: [u8; 4] = [0x00, b'K', b'Z', 0x01];
const MAX_DECOMPRESSED_PROMPT_LEN: u64 = 1 << 20;

/// Plaintext of an opened private prompt: inflated when it carries the compression magic.
/// `None` for a compressed prompt that is corrupt or inflates past the 1 MiB cap.
fn decode_private_prompt(bytes: &[u8]) -> Option<String> {
    use std::io::Read;
    let Some(deflated) = bytes.strip_prefix(&COMPRESSED_PROMPT_MAGIC[..]) else {
        return Some(String::from_utf8_lossy(bytes).into_owned());
    };
    let mut inflated = Vec::new();
    flate2::read::DeflateDecoder::new(deflated).take(MAX_DECOMPRESSED_PROMPT_LEN + 1).read_to_end(&mut inflated).ok()?;
    if inflated.len() as u64 > MAX_DECOMPRESSED_PROMPT_LEN {
        return None;
    }
    Some(String::from_utf8_lossy(&inflated).into_owned())
}

/// Internal, full-width identity for an AiRequest. The consensus-facing `request_hash` remains a
/// separate field and is copied unchanged into AiResponse, preserving wire compatibility.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AiRequestKey([u8; 32]);

impl AiRequestKey {
    /// Bind deduplication to both the consensus request identity and the decoded inference content.
    /// Length-prefixing the variable field makes the encoding unambiguous.
    fn new(request_hash: &[u8; 32], model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Self {
        let mut state = blake2b_simd::Params::new().hash_length(32).to_state();
        state.update(b"Keryx/AiRequestKey/v1");
        state.update(request_hash);
        state.update(model_id);
        state.update(&(prompt.len() as u64).to_le_bytes());
        state.update(prompt.as_bytes());
        state.update(&(max_tokens as u64).to_le_bytes());
        let mut key = [0u8; 32];
        key.copy_from_slice(state.finalize().as_bytes());
        Self(key)
    }
}

#[derive(Debug)]
struct QueuedAiRequest {
    key: AiRequestKey,
    request_hash: [u8; 32],
    model_id: [u8; 32],
    prompt: String,
    /// Plaintext prompt length in bytes as sealed (the trace shape bound; 0 when unknown).
    prompt_len: u32,
    max_tokens: usize,
    /// H14 private request: how the answer must be sealed.
    private: Option<PrivateRequest>,
}

/// A finished inference: the text, the trace an H15 private answer commits, the card and time.
struct Answered {
    text: String,
    trace: Option<keryx_miner::trace::ResponseTrace>,
    gpu: Option<usize>,
    gen_secs: f64,
}

/// An inference running for a queued request.
struct InferenceJob {
    key: AiRequestKey,
    request_hash: [u8; 32],
    private: Option<PrivateRequest>,
    model_id: [u8; 32],
    /// Tier the answer is traced under (H15 private requests); `None`: untraced.
    tier: Option<u8>,
    max_tokens: u32,
    rx: oneshot::Receiver<(Option<Answered>, keryx_miner::runtime_stats::InferenceAttempt)>,
}

/// FIFO with a hard capacity. On overload the oldest queued (not in-flight) request is discarded,
/// making eviction deterministic and keeping recent chain work available.
#[derive(Debug)]
struct BoundedAiRequestQueue {
    capacity: usize,
    entries: VecDeque<QueuedAiRequest>,
}

impl BoundedAiRequestQueue {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self { capacity, entries: VecDeque::with_capacity(capacity) }
    }

    fn push_back(&mut self, request: QueuedAiRequest) -> Option<QueuedAiRequest> {
        let evicted = (self.entries.len() == self.capacity).then(|| self.entries.pop_front()).flatten();
        self.entries.push_back(request);
        evicted
    }

    fn pop_front(&mut self) -> Option<QueuedAiRequest> {
        self.entries.pop_front()
    }

    fn iter(&self) -> impl Iterator<Item = &QueuedAiRequest> {
        self.entries.iter()
    }

    fn front(&self) -> Option<&QueuedAiRequest> {
        self.entries.front()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Debug, Eq, PartialEq)]
enum SeenInsert {
    Duplicate,
    Inserted { evicted: Option<AiRequestKey> },
    AllEntriesProtected,
}

/// Bounded FIFO replay filter. Eviction skips live queue/in-flight keys; among completed entries,
/// the oldest is always selected.
#[derive(Debug)]
struct BoundedAiSeen {
    capacity: usize,
    keys: HashSet<AiRequestKey>,
    order: VecDeque<AiRequestKey>,
}

impl BoundedAiSeen {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self { capacity, keys: HashSet::with_capacity(capacity), order: VecDeque::with_capacity(capacity) }
    }

    #[cfg(test)]
    fn contains(&self, key: &AiRequestKey) -> bool {
        self.keys.contains(key)
    }

    fn insert<F>(&mut self, key: AiRequestKey, is_protected: F) -> SeenInsert
    where
        F: Fn(&AiRequestKey) -> bool,
    {
        if self.keys.contains(&key) {
            return SeenInsert::Duplicate;
        }

        let evicted = if self.keys.len() == self.capacity {
            let Some(position) = self.order.iter().position(|candidate| !is_protected(candidate)) else {
                return SeenInsert::AllEntriesProtected;
            };
            let evicted = self.order.remove(position).expect("seen FIFO position must exist");
            self.keys.remove(&evicted);
            Some(evicted)
        } else {
            None
        };

        self.keys.insert(key);
        self.order.push_back(key);
        SeenInsert::Inserted { evicted }
    }

    fn remove(&mut self, key: &AiRequestKey) -> bool {
        if !self.keys.remove(key) {
            return false;
        }
        if let Some(position) = self.order.iter().position(|candidate| candidate == key) {
            self.order.remove(position);
        }
        true
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Node-issued challenge inference. A completed result is latched here until exactly one fresh
/// template request consumes it; `poll_ready` returns true only on the Running -> Ready edge.
struct ChallengeInference {
    challenge: String,
    receiver: Option<oneshot::Receiver<(
        Option<String>,
        keryx_miner::runtime_stats::InferenceAttempt,
    )>>,
    result: Option<
        Result<
            (Option<String>, keryx_miner::runtime_stats::InferenceAttempt),
            (),
        >,
    >,
}

impl ChallengeInference {
    fn running(
        challenge: String,
        receiver: oneshot::Receiver<(
            Option<String>,
            keryx_miner::runtime_stats::InferenceAttempt,
        )>,
    ) -> Self {
        Self { challenge, receiver: Some(receiver), result: None }
    }

    fn poll_ready(&mut self) -> bool {
        if self.result.is_some() {
            return false;
        }
        let Some(receiver) = self.receiver.as_mut() else {
            return false;
        };
        let result = match receiver.try_recv() {
            Ok(result) => Ok(result),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => return false,
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => Err(()),
        };
        self.receiver = None;
        self.result = Some(result);
        true
    }
}

#[allow(dead_code)]
pub struct KeryxdHandler {
    client: RpcClient<TonicChannel>,
    pub send_channel: Sender<KaspadMessage>,
    stream: Streaming<KaspadMessage>,
    miner_address: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,

    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,

    /// Bounded queue of AiRequests waiting for inference. `request_hash` is the consensus-facing
    /// identity — the AiRequest TXID past the H8 gate, the payload digest before.
    ai_request_queue: BoundedAiRequestQueue,

    /// Full content-bound keys already queued, in-flight, or recently completed.
    ai_seen_keys: BoundedAiSeen,

    /// Maps full request key → (txid, inference_reward_sompi) for pending confirmed requests.
    /// Used by poll_inference to register the escrow outpoint after a successful AiResponse.
    ai_request_txids: HashMap<AiRequestKey, (String, u64)>,

    /// In-flight SLM inference task. The stats attempt remains live through IPFS + submission,
    /// making success/failure exactly-once.
    inference_rx: Option<InferenceJob>,

    // ── 0.15.3: H15 traced responses (upstream v0.5.8 servicer state) ──
    /// AiResponses not yet accepted by the mempool, by txid: (tx, attempts, last submit).
    ai_response_inflight: HashMap<String, (crate::proto::RpcTransaction, u32, std::time::Instant)>,
    /// Traced responses by response txid, from submission to the end of their opening window.
    pending_traces: HashMap<String, PendingTrace>,
    /// Recent chain blocks in chain order, back to the oldest one a pending trace needs.
    chain_log: VecDeque<ChainEntry>,
    /// On-disk copy of the pending traces (`<escrow state>.traces/`).
    trace_store: keryx_miner::trace_store::TraceStore,
    chain_restore_queue: VecDeque<String>,
    chain_restore_pending: HashSet<String>,
    chain_restore_starts: VecDeque<String>,
    chain_restore_failures: HashMap<String, u32>,
    chain_restore_retry: Vec<(std::time::Instant, String)>,
    chain_block_retry: Vec<(std::time::Instant, String, u32)>,
    /// Traced responses waiting for their trace to reach the disk before they are sent.
    persisting: Vec<(std::sync::mpsc::Receiver<()>, String, crate::proto::RpcTransaction)>,
    /// Responses accepted before their request, sent again once it is accepted.
    response_resend: Vec<(std::time::Instant, String, u32)>,
    chain_restore_continuations: HashSet<String>,
    /// In-flight replay building an opening.
    replay_rx: Option<ReplayJob>,
    /// Opening submissions not yet accepted by a chain block, by txid.
    opening_inflight: HashMap<String, OpeningInflight>,
    /// The node answers `GetAwaitedOpenings` (keryxd >= 1.6.6).
    node_awaited_openings: bool,
    last_awaited_poll: std::time::Instant,
    awaited_polled: Vec<String>,

    /// Request identities (`request_hash`) this escrow key has already answered on-chain. Such a
    /// request is never served again: the node rejects the duplicate and the inference would be
    /// wasted (upstream v0.5.6 7ca66e6). Bounded like the replay filter.
    ai_answered_by_me: HashSet<[u8; 32]>,

    /// A template was dropped because no model was proven serveable yet. Templates are otherwise
    /// only re-requested on a node notification, so a quiet chain (solo/testnet) would never hand
    /// out the next one; the tick loop re-requests once a model is ready.
    template_deferred_for_models: bool,

    /// In-flight or completed inference for a node-issued challenge. Completion is edge-triggered
    /// so the 200 ms timer emits one, not 5-per-second, GetBlockTemplate refresh.
    challenge_inference: Option<ChallengeInference>,

    /// Last DAA score seen in a block template — used to compute challenge_window_end.
    last_known_daa: u64,

    /// IPFS Kubo API URL for uploading inference results.
    ipfs_url: String,

    /// 64-char hex Schnorr pubkey embedded in coinbase extra_data as `/escrow:<pubkey>`.
    /// The node routes 20% of the block reward to the corresponding CSV-locked escrow output.
    escrow_pubkey: Option<String>,

    /// Auto-claim module: present when an escrow private key is available.
    escrow_watcher: Option<crate::escrow::EscrowWatcher>,

    /// 128-char hex delegation cert embedded as `/esig:<cert>`, binding the escrow key above to
    /// the payout address. Mandatory from H6 — a block without it is invalid.
    escrow_cert: Option<String>,

    /// Service-ledger identity of the payout address (the node's `miner_key`), the key strikes,
    /// burns and suspensions are reported against.
    service_identity: Option<String>,

    /// Block hashes queued for boot-time escrow-state validation, drained in slices of
    /// VALIDATION_WINDOW so thousands of GetBlock requests never saturate the HTTP/2
    /// flow-control window (each consumed answer sends the next queued request).
    validation_queue: VecDeque<String>,

    /// Last reported pending-escrow figure (outputs, sompi) — the pending total is
    /// logged only when it changes, keeping the log quiet between changes.
    last_pending_escrow: Option<(u64, u64)>,

    /// Last service-bond strike poll instant. (upstream 777f2cc)
    last_strike_poll: std::time::Instant,

    /// Last rendered service-bond status — logged only on change. (upstream 777f2cc)
    strike_status: Option<String>,

    /// Generation token for process-wide display state. A dropped prior connection cannot mark a
    /// newer solo session offline.
    runtime_generation: u64,
}

#[async_trait(?Send)]
impl Client for KeryxdHandler {
    async fn register(&mut self) -> Result<(), Error> {
        // We actually register in connect
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        // Harvest in-flight inference on a timer, independently of node notifications.
        // On a sole-producer node, pausing mining for inference stops block production,
        // so the node stops sending NewBlockTemplate notifications — without this timer
        // the finished inference would never be collected and mining would deadlock.
        let mut tick = tokio::time::interval(tokio::time::Duration::from_millis(200));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            let maybe_msg = tokio::select! {
                msg = self.stream.message() => Some(msg?),
                _ = tick.tick() => None,
            };
            match maybe_msg {
                Some(Some(m)) => match m.payload {
                    Some(payload) => self.handle_message(payload, miner).await?,
                    None => warn!("keryxd message payload is empty"),
                },
                Some(None) => break, // stream closed by node
                None => {
                    // Completion edges, never mere pending state, trigger a refresh. At most one
                    // GetBlockTemplate RPC is emitted by any 200 ms tick.
                    let regular_finished = self.inference_rx.is_some() && self.poll_inference().await;
                    self.trace_tick().await?;
                    let challenge_finished = !regular_finished && self.challenge_inference_ready();
                    let models_now_ready =
                        self.template_deferred_for_models && keryx_miner::slm::has_proven_serveable_model();
                    if models_now_ready {
                        self.template_deferred_for_models = false;
                    }
                    if regular_finished || challenge_finished || models_now_ready {
                        self.client_get_block_template().await?;
                    }
                    if self.escrow_pubkey.is_some() && self.last_strike_poll.elapsed().as_secs() >= 60 {
                        self.last_strike_poll = std::time::Instant::now();
                        self.client_send(GetServiceStrikesRequestMessage {}).await?;
                    }
                }
            }
        }
        Ok(())
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

impl KeryxdHandler {
    fn challenge_inference_ready(&mut self) -> bool {
        self.challenge_inference.as_mut().map_or(false, ChallengeInference::poll_ready)
    }

    pub async fn connect<D>(
        address: D,
        miner_address: String,
        mine_when_not_synced: bool,
        block_template_ctr: Option<Arc<AtomicU16>>,
        escrow_privkey: Option<String>,
        escrow_state_file: String,
        escrow_cert: Option<String>,
        ipfs_url: String,
    ) -> Result<Box<Self>, Error>
    where
        D: std::convert::TryInto<tonic::transport::Endpoint>,
        D::Error: Into<Error>,
    {
        // Build EscrowWatcher from the resolved escrow privkey (derived or loaded from file).
        // The watcher also provides the pubkey to embed in coinbase extra_data.
        let trace_state_file = escrow_state_file.clone();
        let (escrow_pubkey, escrow_watcher) = match escrow_privkey {
            Some(ref privkey) => {
                match crate::escrow::EscrowWatcher::new(privkey, &miner_address, escrow_state_file.into()) {
                    Ok(watcher) => {
                        let pk = watcher.pubkey_hex();
                        info!("OPoI escrow active: pubkey={}", pk);
                        (Some(pk), Some(watcher))
                    }
                    Err(e) => {
                        log::error!("Failed to initialise EscrowWatcher: {} — escrow disabled", e);
                        (None, None)
                    }
                }
            }
            None => (None, None),
        };
        keryx_miner::runtime_stats::escrow_enabled(escrow_watcher.is_some());

        let service_identity = match crate::escrow::service_identity_hex(&miner_address) {
            Ok(id) => Some(id),
            Err(e) => {
                log::warn!("Cannot derive the service identity of the payout address: {}", e);
                None
            }
        };

        let endpoint: Endpoint = address.try_into().map_err(Into::into)?;
        let endpoint_label = endpoint.uri().to_string();
        let runtime_generation = keryx_miner::runtime_stats::begin_connection(
            keryx_miner::runtime_stats::MiningMode::Solo,
            &endpoint_label,
            0,
        );
        let connect_started = std::time::Instant::now();
        let mut client = match RpcClient::connect(endpoint).await {
            Ok(client) => client,
            Err(error) => {
                keryx_miner::runtime_stats::connection_lost(runtime_generation, "Solo node connection failed");
                return Err(error.into());
            }
        };
        // Outbound message channel to the node. ALL client->node messages share this:
        // mining (submit_block, GetBlockTemplate) AND OPoI traffic (per-block GetBlock,
        // escrow submit_transaction). With a capacity of 2 the OPoI traffic could fill the
        // buffer and block GetBlockTemplate, stalling template delivery → the GPU sits idle
        // between blocks. A large buffer keeps the mining requests from queuing behind OPoI.
        let (send_channel, recv) = mpsc::channel(1024);
        if let Err(error) = send_channel.send(GetInfoRequestMessage {}.into()).await {
            keryx_miner::runtime_stats::connection_lost(runtime_generation, "Solo node connection failed");
            return Err(error.into());
        }
        let stream = match client.message_stream(ReceiverStream::new(recv)).await {
            Ok(stream) => stream.into_inner(),
            Err(error) => {
                keryx_miner::runtime_stats::connection_lost(runtime_generation, "Solo node connection failed");
                return Err(error.into());
            }
        };
        keryx_miner::runtime_stats::connection_established(
            runtime_generation,
            Some(connect_started.elapsed().as_millis().min(u64::MAX as u128) as u64),
        );
        keryx_miner::runtime_stats::set_connection_inference_queue(
            runtime_generation,
            0,
            AI_REQUEST_QUEUE_CAPACITY,
        );
        let (block_channel, block_handle) = Self::create_block_channel(send_channel.clone());
        let trace_store = keryx_miner::trace_store::TraceStore::new(&trace_state_file);
        let (restored_traces, restored_openings) = Self::restore_traces(&trace_store);
        Ok(Box::new(Self {
            client,
            stream,
            send_channel,
            miner_address,
            mine_when_not_synced,
            block_template_ctr: block_template_ctr
                .unwrap_or_else(|| Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16))),
            block_channel,
            block_handle,
            ai_request_queue: BoundedAiRequestQueue::new(AI_REQUEST_QUEUE_CAPACITY),
            ai_seen_keys: BoundedAiSeen::new(AI_SEEN_CAPACITY),
            ai_request_txids: HashMap::new(),
            inference_rx: None,
            ai_response_inflight: HashMap::new(),
            pending_traces: restored_traces,
            chain_log: VecDeque::new(),
            trace_store,
            chain_restore_queue: VecDeque::new(),
            chain_restore_pending: HashSet::new(),
            chain_restore_starts: VecDeque::new(),
            chain_restore_failures: HashMap::new(),
            chain_restore_retry: Vec::new(),
            chain_block_retry: Vec::new(),
            persisting: Vec::new(),
            response_resend: Vec::new(),
            chain_restore_continuations: HashSet::new(),
            replay_rx: None,
            opening_inflight: restored_openings,
            node_awaited_openings: false,
            last_awaited_poll: std::time::Instant::now(),
            awaited_polled: Vec::new(),
            ai_answered_by_me: HashSet::new(),
            template_deferred_for_models: false,
            challenge_inference: None,
            last_known_daa: 0,
            ipfs_url,
            escrow_pubkey,
            escrow_watcher,
            escrow_cert,
            service_identity,
            validation_queue: VecDeque::new(),
            last_pending_escrow: None,
            last_strike_poll: std::time::Instant::now() - std::time::Duration::from_secs(55),
            strike_status: None,
            runtime_generation,
        }))
    }

    fn create_block_channel(send_channel: Sender<KaspadMessage>) -> (Sender<BlockSeed>, BlockHandle) {
        // KaspadMessage::submit_block(block)
        let (send, recv) = mpsc::channel::<BlockSeed>(1);
        (
            send,
            tokio::spawn(async move {
                ReceiverStream::new(recv)
                    .map(|block_seed| match block_seed {
                        FullBlock(block) => KaspadMessage::submit_block(*block),
                        PartialBlock { .. } => unreachable!("All blocks sent here should have arrived from here"),
                    })
                    .map(Ok)
                    .forward(PollSender::new(send_channel))
                    .await
            }),
        )
    }

    async fn client_send(&self, msg: impl Into<KaspadMessage>) -> Result<(), SendError<KaspadMessage>> {
        self.send_channel.send(msg.into()).await
    }

    /// Log the pending escrow total (outputs awaiting claim + KRX) whenever it changes.
    /// We have no stats module — this is the metric surface for upstream's
    /// `miner.record_escrow_pending`; quiet while the figure is unchanged.
    fn report_pending_escrow(&mut self) {
        let Some(w) = self.escrow_watcher.as_ref() else { return };
        let pending = w.pending_escrow();
        if self.last_pending_escrow != Some(pending) {
            info!("Escrow pending: {} output(s) awaiting claim, {:.8} KRX", pending.0, pending.1 as f64 / 1e8);
            self.last_pending_escrow = Some(pending);
        }
    }

    async fn client_get_block_template(&mut self) -> Result<(), SendError<KaspadMessage>> {
        // Always mine to the configured address. (Devfund address-swap cycle
        // removed — see docs/devfund-removed.md.)
        let pay_address = self.miner_address.clone();
        self.block_template_ctr.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000)).unwrap();
        // Append a per-request random nonce so that parallel blocks at the same blue_score
        // get distinct coinbase payloads → distinct tx_ids (avoids DAG coinbase collisions).
        let nonce_hex = format!("{:016x}", thread_rng().next_u64());
        // OPoI Phase 2: run the deterministic fixed-point MLP (matches node validation).
        let opoi_tag = keryx_miner::inference::compute_opoi_tag(&nonce_hex);
        // Embed escrow pubkey so the node routes 20% to the CSV-locked escrow output.
        let escrow_part = self.escrow_pubkey.as_deref().map(|pk| format!("/escrow:{}", pk)).unwrap_or_default();
        // Delegation cert binding that escrow key to the payout address. From H6 the node rejects
        // a block whose coinbase carries no valid pair.
        let esig_part = self.escrow_cert.as_deref().map(|cert| format!("/esig:{}", cert)).unwrap_or_default();
        // Announce loaded model capabilities so the node can enforce model_id matching.
        let cap_part = {
            let ids = keryx_miner::slm::serveable_model_ids();
            if ids.is_empty() {
                String::new()
            } else {
                let hex_ids: Vec<String> = ids.iter().map(|id| hex::encode(id)).collect();
                format!("/ai:cap:{}", hex_ids.join(","))
            }
        };
        let extra_data =
            format!("{}{}{}/{}/ai:v1:{}{}", EXTRA_DATA, escrow_part, esig_part, nonce_hex, opoi_tag, cap_part);
        // Harvest a latched challenge result. A still-running challenge stays installed; the
        // timer will request one fresh template on its eventual Running -> Ready transition.
        if let Some(challenge) = self.challenge_inference.as_mut() {
            challenge.poll_ready();
        }
        let mut completed_attempt = None;
        let inference_result = match self.challenge_inference.take() {
            Some(mut challenge) => match challenge.result.take() {
                Some(Ok((Some(text), runtime_attempt))) if !text.is_empty() => {
                    // challenge_str = "model_id_hex:nonce_hex"
                    let mut parts = challenge.challenge.splitn(2, ':');
                    let model_id_hex = parts.next().unwrap_or("");
                    let nonce_hex_c = parts.next().unwrap_or("");
                    info!("OPoI: sending challenge response model={:.8}", model_id_hex);
                    keryx_miner::runtime_stats::record_inference_prepared();
                    completed_attempt = Some((runtime_attempt, text.split_whitespace().count()));
                    // Response format: "model_id_hex:nonce_hex:result_text"
                    format!("{}:{}:{}", model_id_hex, nonce_hex_c, text)
                }
                Some(Ok((Some(_), _))) | Some(Ok((None, _))) | Some(Err(())) => {
                    warn!("OPoI: challenge inference failed — sending empty result, node will re-challenge");
                    String::new()
                }
                None => {
                    self.challenge_inference = Some(challenge);
                    String::new()
                }
            },
            None => String::new(),
        };
        let sent = self.client_send(GetBlockTemplateRequestMessage { pay_address, extra_data, inference_result }).await;
        if let Some((mut attempt, tokens)) = completed_attempt {
            if sent.is_ok() {
                // The response is now queued on the node transport. There is no response-specific
                // acknowledgement in this RPC, so `delivered` intentionally remains unknown.
                attempt.served(tokens);
            } else {
                attempt.failed();
            }
        }
        sent
    }

    /// Preserve the historical 16-hex-character operator label without using it as identity.
    fn request_log_id(request_hash: &[u8; 32]) -> String {
        hex::encode(&request_hash[..8])
    }

    fn ai_request_pending(&self, key: &AiRequestKey) -> bool {
        self.ai_request_queue.iter().any(|request| request.key == *key)
            || self.inference_rx.as_ref().map_or(false, |job| job.key == *key)
    }

    /// Central terminal cleanup. Metadata is removed on every outcome; retryable failures also
    /// remove the replay marker so a later block/template observation can queue the request again.
    fn finish_ai_request(&mut self, key: AiRequestKey, keep_seen: bool) -> Option<(String, u64)> {
        let escrow = self.ai_request_txids.remove(&key);
        if !keep_seen {
            self.ai_seen_keys.remove(&key);
        }
        escrow
    }

    /// Insert into the bounded replay filter and queue. Returns false for a duplicate or if every
    /// replay-cache slot is protected by live work (the latter is unreachable with production
    /// capacities, but remains a safe overload behavior).
    fn enqueue_ai_request(&mut self, request: QueuedAiRequest) -> bool {
        let key = request.key;
        let in_flight_key = self.inference_rx.as_ref().map(|job| job.key);
        let queue = &self.ai_request_queue;
        match self.ai_seen_keys.insert(key, |candidate| {
            in_flight_key.as_ref() == Some(candidate) || queue.iter().any(|queued| &queued.key == candidate)
        }) {
            SeenInsert::Duplicate => return false,
            SeenInsert::AllEntriesProtected => {
                warn!("OPoI: replay cache is full of live requests — dropping newest AiRequest");
                keryx_miner::runtime_stats::record_inference_busy_request(
                    keryx_miner::runtime_stats::InferenceKind::SoloRequest,
                );
                return false;
            }
            SeenInsert::Inserted { evicted } => {
                if let Some(old_key) = evicted {
                    // Completed replay entries normally have no escrow metadata. Remove any stale
                    // value defensively so this map remains bounded by live queue/in-flight work.
                    self.ai_request_txids.remove(&old_key);
                }
            }
        }

        if let Some(evicted) = self.ai_request_queue.push_back(request) {
            warn!(
                "OPoI: AiRequest queue full ({}); evicting oldest queued id={}",
                AI_REQUEST_QUEUE_CAPACITY,
                Self::request_log_id(&evicted.request_hash)
            );
            keryx_miner::runtime_stats::record_inference_busy_request(
                keryx_miner::runtime_stats::InferenceKind::SoloRequest,
            );
            self.finish_ai_request(evicted.key, false);
        }
        keryx_miner::runtime_stats::set_connection_inference_queue(
            self.runtime_generation,
            self.ai_request_queue.len(),
            AI_REQUEST_QUEUE_CAPACITY,
        );
        true
    }

    /// Attach escrow metadata only while the corresponding request is queued or in flight. A
    /// duplicate observation after successful completion must not recreate an orphan map entry.
    fn remember_ai_request_txid(&mut self, key: AiRequestKey, txid: String, inference_reward: u64) {
        if self.ai_request_pending(&key) {
            self.ai_request_txids.insert(key, (txid, inference_reward));
        }
    }

    /// Scans a slice of transactions for AiRequest payloads and pushes new
    /// entries into `ai_request_queue` (deduplication by a full content-bound key).
    ///
    /// Handles two formats:
    ///   - Subnetwork 0x03 + binary `AiRequestPayload` (future on-chain format)
    ///   - Any non-coinbase TX + `KRX:AI:1:` JSON prefix (web wallet format)
    fn scan_txs_for_ai_requests(&mut self, txs: &[crate::proto::RpcTransaction], block_daa: u64) {
        // Identity of a request follows the same gate as the node (H8): the transaction id past it,
        // the payload digest before. Decided from the daa of the block the request is observed in, so
        // both sides classify a request the same way across the activation. (Upstream v0.4.9 54129d80.)
        let txid_identity = block_daa >= keryx_miner::pom::reward_routing_activation_daa();
        // Never re-serve a request this escrow key already answered on-chain (upstream 7ca66e6):
        // the node rejects the duplicate, so a re-run would only burn a GPU slot.
        if let Some(mine) = self.escrow_watcher.as_ref().map(|w| w.pubkey_bytes()) {
            for request_hash in Self::own_answered_request_hashes(txs, &mine) {
                if self.ai_answered_by_me.insert(request_hash) {
                    log::debug!(
                        "OPoI: request id={} already answered by this miner on-chain",
                        Self::request_log_id(&request_hash)
                    );
                }
            }
            if self.ai_answered_by_me.len() > AI_SEEN_CAPACITY {
                self.ai_answered_by_me.clear();
                self.ai_answered_by_me.shrink_to_fit();
            }
        }
        // Hard gate: if no models are ready, refuse to accept any AiRequest.
        // Prevents miners with missing/truncated model files from ever queuing inference work.
        let ready_ids = keryx_miner::slm::serveable_model_ids();
        if ready_ids.is_empty() {
            log::warn!("OPoI: no models ready — skipping AiRequest scan (run miner with valid model files)");
            return;
        }
        log::debug!(
            "scan_ai: {} txs, subnetwork_ids: {:?}",
            txs.len(),
            txs.iter().map(|t| t.subnetwork_id.as_str()).collect::<Vec<_>>()
        );
        let escrow_secret = self.escrow_watcher.as_ref().map(|w| w.secret_bytes());
        // From the H14 gate every AiRequest is a private envelope (the node invalidates plaintext);
        // before it a private envelope is not a valid request at all.
        // Classified by the DAA of the block the request is in (like the node's era rules), so a
        // pre-gate plaintext request observed after the crossing is still answered the old way.
        let era_daa = if block_daa > 0 { block_daa } else { self.last_known_daa };
        let private_era = keryx_miner::pom::is_h14_era(era_daa);
        for tx in txs {
            // (raw, model_id, prompt, prompt_len, max_tokens, inference_reward, private state)
            let extracted: Option<(Vec<u8>, [u8; 32], String, u32, usize, u64, Option<PrivateRequest>)> =
                if tx.subnetwork_id == keryx_inference::SUBNETWORK_ID_AI_REQUEST_HEX {
                    // Binary AiRequestPayload (dedicated AI subnetwork).
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        let req = keryx_inference::AiRequestPayload::deserialize(&raw)?;
                        // Before the gate an envelope is not a private request at all: keep the
                        // 0.13.4 handling (plain bytes), so pre-gate behaviour is unchanged.
                        let (prompt, prompt_len, private) = if private_era && req.is_private() {
                            Self::open_private_request(&req, escrow_secret.as_ref())?
                        } else {
                            if private_era {
                                log::debug!("OPoI: ignoring a plaintext AiRequest past the H14 gate (invalid on-chain)");
                                return None;
                            }
                            (String::from_utf8_lossy(&req.prompt).into_owned(), req.prompt.len() as u32, None)
                        };
                        Some((raw, req.model_id, prompt, prompt_len, req.max_tokens as usize, req.inference_reward, private))
                    })
                } else if !tx.inputs.is_empty() && !private_era {
                    // KRX:AI:1: JSON format — model routed by "m" field, skipped if not loaded.
                    // Plaintext, so pre-H14 only.
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        Self::parse_krx_ai_payload(&raw).and_then(|(model_name, prompt, max_tokens)| {
                            let model_id = keryx_miner::models::find(&model_name)?.model_id;
                            let prompt_len = prompt.len() as u32;
                            Some((raw, model_id, prompt, prompt_len, max_tokens, 0u64, None))
                        })
                    })
                } else {
                    None // coinbase — skip
                };

            if let Some((raw, model_id, prompt, prompt_len, max_tokens, inference_reward, private)) = extracted {
                // An unreadable private request runs no inference (its answer is the fixed error
                // body), so the prompt limits do not apply to it.
                if private != Some(PrivateRequest::Unreadable) {
                    if let Err(reason) = keryx_miner::slm::validate_inference_request(
                        &prompt,
                        max_tokens,
                        keryx_miner::slm::DEFAULT_INFERENCE_DEADLINE_MS,
                    ) {
                        log::warn!("OPoI: rejecting invalid AiRequest: {}", reason);
                        continue;
                    }
                }
                if !ready_ids.contains(&model_id) {
                    log::debug!("OPoI: skipping AiRequest — model not supported or files not ready");
                    continue;
                }
                // The AiRequest txid, needed both for the H8 identity and for escrow-claim tracking.
                // Prefer verbose_data.transaction_id; fall back to computing it (verbose_data is not
                // populated for non-coinbase TXs in block-template/notify streams).
                let txid_hex = tx
                    .verbose_data
                    .as_ref()
                    .map(|v| v.transaction_id.clone())
                    .filter(|id| !id.is_empty())
                    .or_else(|| Self::compute_rpc_txid(tx));
                // H8 identity: TXID past the gate, payload digest before.
                let request_hash: [u8; 32] = if txid_identity {
                    match txid_hex
                        .as_deref()
                        .and_then(|h| hex::decode(h).ok())
                        .and_then(|b| <[u8; 32]>::try_from(b).ok())
                    {
                        Some(id) => id,
                        None => {
                            log::warn!("OPoI: cannot resolve the AiRequest transaction id — request skipped");
                            continue;
                        }
                    }
                } else {
                    blake2b_simd::blake2b(&raw).as_bytes()[..32].try_into().unwrap()
                };
                if self.ai_answered_by_me.contains(&request_hash) {
                    log::debug!(
                        "OPoI: skipping AiRequest id={} — already answered by this miner",
                        Self::request_log_id(&request_hash)
                    );
                    continue;
                }
                let key = AiRequestKey::new(&request_hash, &model_id, &prompt, max_tokens);
                let log_id = Self::request_log_id(&request_hash);
                let queued = self.enqueue_ai_request(QueuedAiRequest {
                    key,
                    request_hash,
                    model_id,
                    prompt,
                    prompt_len,
                    max_tokens,
                    private,
                });
                if queued {
                    info!("OPoI: queued AiRequest id={}", log_id);
                }
                // Track txid for escrow claims (the inference_reward outpoint is claimed after the
                // challenge window).
                if inference_reward > 0 {
                    if let Some(txid) = txid_hex {
                        self.remember_ai_request_txid(key, txid, inference_reward);
                    }
                }
            }
        }
    }

    /// Request identities answered in `txs` by the responder `mine` (V2 AiResponses signed with
    /// this escrow key). Mirrors upstream's own_answered_request_ids on the full request_hash.
    fn own_answered_request_hashes(txs: &[crate::proto::RpcTransaction], mine: &[u8; 32]) -> Vec<[u8; 32]> {
        txs.iter()
            .filter(|tx| tx.subnetwork_id == keryx_inference::SUBNETWORK_ID_AI_RESPONSE_HEX)
            .filter_map(|tx| hex::decode(&tx.payload).ok())
            .filter_map(|raw| keryx_inference::AiResponsePayload::deserialize(&raw))
            .filter(|resp| resp.responder.as_ref().is_some_and(|r| r.escrow_pubkey == *mine))
            .map(|resp| resp.request_hash)
            .collect()
    }

    /// Opens a private request with the escrow key (upstream v0.5.6 50d45b1 + a9f509f). `None`
    /// when the request is not for this miner (no escrow key, or not among the recipients — it
    /// joined the cohort after sealing; the audit does not expect it to answer). An addressed
    /// request that cannot be opened is still queued, to be answered with the error body.
    fn open_private_request(
        req: &keryx_inference::AiRequestPayload,
        escrow_secret: Option<&[u8; 32]>,
    ) -> Option<(String, u32, Option<PrivateRequest>)> {
        let Some(secret) = escrow_secret else {
            log::debug!("OPoI: skipping private AiRequest — no escrow key configured");
            return None;
        };
        match keryx_inference::open_request(req, secret) {
            Ok(opened) => match decode_private_prompt(&opened.prompt) {
                Some(prompt) => {
                    Some((prompt, opened.prompt.len().min(u32::MAX as usize) as u32, Some(PrivateRequest::Opened(opened.response_key.clone()))))
                }
                None => {
                    warn!("OPoI: private AiRequest opened but its compressed prompt is invalid — answering with an error body");
                    Some((String::new(), 0, Some(PrivateRequest::Unreadable)))
                }
            },
            Err(keryx_inference::PrivateError::NotARecipient) => {
                log::debug!("OPoI: skipping private AiRequest — this escrow key is not a recipient");
                None
            }
            Err(e) => {
                warn!("OPoI: private AiRequest names this miner but cannot be opened ({}) — answering with an error body", e);
                Some((String::new(), 0, Some(PrivateRequest::Unreadable)))
            }
        }
    }

    /// Compute the Kaspa transaction ID for a non-coinbase RpcTransaction.
    ///
    /// Mirrors keryx-node consensus/core/src/hashing/tx.rs `id()` with
    /// EXCLUDE_SIGNATURE_SCRIPT | EXCLUDE_MASS_COMMIT flags (standard for non-coinbase txs).
    ///
    /// Serialization: blake2b-256 keyed "TransactionID" over:
    ///   version(u16 LE) | inputs_count(u64 LE) | inputs... | outputs_count(u64 LE) | outputs...
    ///   | lock_time(u64 LE) | subnetwork_id(20B) | gas(u64 LE) | payload_len(u64 LE) | payload
    ///
    /// For each input (sig script excluded): txid(32B) | index(u32 LE) | 0u64(empty var_bytes) | seq(u64 LE)
    /// For each output: amount(u64 LE) | spk_version(u16 LE) | script_len(u64 LE) | script
    fn compute_rpc_txid(tx: &crate::proto::RpcTransaction) -> Option<String> {
        const KEY: &[u8] = b"TransactionID";
        let mut h = blake2b_simd::Params::new().hash_length(32).key(KEY).to_state();

        h.update(&(tx.version as u16).to_le_bytes());
        h.update(&(tx.inputs.len() as u64).to_le_bytes());
        for input in &tx.inputs {
            let prev = input.previous_outpoint.as_ref()?;
            let txid_bytes = hex::decode(&prev.transaction_id).ok()?;
            if txid_bytes.len() != 32 {
                return None;
            }
            h.update(&txid_bytes);
            h.update(&prev.index.to_le_bytes());
            h.update(&0u64.to_le_bytes()); // write_var_bytes(&[]) — empty sig script
            h.update(&input.sequence.to_le_bytes());
        }

        h.update(&(tx.outputs.len() as u64).to_le_bytes());
        for output in &tx.outputs {
            h.update(&output.amount.to_le_bytes());
            let spk = output.script_public_key.as_ref()?;
            h.update(&(spk.version as u16).to_le_bytes());
            let script = hex::decode(&spk.script_public_key).ok()?;
            h.update(&(script.len() as u64).to_le_bytes());
            h.update(&script);
        }

        h.update(&tx.lock_time.to_le_bytes());
        let subnet = hex::decode(&tx.subnetwork_id).ok()?;
        if subnet.len() != 20 {
            return None;
        }
        h.update(&subnet);
        h.update(&tx.gas.to_le_bytes());
        let payload = hex::decode(&tx.payload).ok()?;
        h.update(&(payload.len() as u64).to_le_bytes());
        h.update(&payload);

        Some(hex::encode(h.finalize().as_bytes()))
    }

    /// Parses a `KRX:AI:1:` JSON payload, returning `(model_name, prompt, max_tokens)`.
    fn parse_krx_ai_payload(raw: &[u8]) -> Option<(String, String, usize)> {
        const PREFIX: &[u8] = b"KRX:AI:1:";
        if raw.len() <= PREFIX.len() || !raw.starts_with(PREFIX) {
            return None;
        }
        let v: serde_json::Value = serde_json::from_slice(&raw[PREFIX.len()..]).ok()?;
        // No default model name: TinyLlama was retired with the pre-H6 lineup, and naming any
        // model here would just fail the registry lookup later. A payload without "m" is invalid.
        let model = v["m"].as_str()?.to_string();
        let prompt = v["p"].as_str()?.to_string();
        let max_tokens = usize::try_from(v["n"].as_u64().unwrap_or(128)).ok()?;
        keryx_miner::slm::validate_inference_request(
            &prompt,
            max_tokens,
            keryx_miner::slm::DEFAULT_INFERENCE_DEADLINE_MS,
        )
        .ok()?;
        Some((model, prompt, max_tokens))
    }

    /// Starts SLM inference for the next queued AiRequest, if no inference or opening replay is
    /// in flight and no audit replay is due before it (upstream v0.5.8 `urgent_replay`).
    fn try_start_inference(&mut self) -> bool {
        if self.inference_busy() {
            return false;
        }
        if !self.ai_request_queue.is_empty() && !self.pending_traces.is_empty() && self.urgent_replay(self.next_request_work()) {
            log::debug!("Trace: an audit replay is due before the next request — the request waits");
            return false;
        }
        while let Some(request) = self.ai_request_queue.pop_front() {
            keryx_miner::runtime_stats::set_connection_inference_queue(
                self.runtime_generation,
                self.ai_request_queue.len(),
                AI_REQUEST_QUEUE_CAPACITY,
            );
            let QueuedAiRequest { key, request_hash, model_id, prompt, prompt_len, max_tokens, private } = request;
            let log_id = Self::request_log_id(&request_hash);
            if self.ai_answered_by_me.contains(&request_hash) {
                info!("OPoI: request id={} already answered by this miner — skipped", log_id);
                self.finish_ai_request(key, true);
                continue;
            }
            if private == Some(PrivateRequest::Unreadable) {
                // No inference to run: the answer is the fixed error text.
                let (tx_done, rx_done) = oneshot::channel();
                let attempt = keryx_miner::runtime_stats::begin_inference(
                    keryx_miner::runtime_stats::InferenceKind::SoloRequest,
                    Some(&model_id),
                );
                let answered = Answered { text: UNREADABLE_PRIVATE_REQUEST_ANSWER.to_string(), trace: None, gpu: None, gen_secs: 0.0 };
                let _ = tx_done.send((Some(answered), attempt));
                self.inference_rx =
                    Some(InferenceJob { key, request_hash, private, model_id, tier: None, max_tokens: max_tokens as u32, rx: rx_done });
                return true;
            }
            // H15: a private answer must carry its trace commitment (node `ResponseWithoutTrace`
            // from gate + 72,000 DAA). Traced generation needs the CUDA in-process engine and an
            // audited tier; anything else refuses the request instead of sending an invalid answer.
            let traced_era = private.is_some() && self.last_known_daa >= keryx_miner::pom::verifiable_inference_activation_daa();
            let tier = if traced_era { keryx_miner::models::pom_tier_index(&model_id, self.last_known_daa) } else { None };
            if traced_era && (tier.is_none() || !keryx_miner::traced::supported()) {
                warn!(
                    "OPoI: private request id={} cannot be answered with a trace by this build/model ({}) — not served",
                    log_id,
                    if tier.is_none() { "model has no audited tier" } else { "inference engine cannot trace" }
                );
                keryx_miner::runtime_stats::record_inference_failed_request(keryx_miner::runtime_stats::InferenceKind::SoloRequest);
                self.finish_ai_request(key, true);
                continue;
            }
            // Second guard: re-check readiness at execution time (files could have been deleted).
            if !keryx_miner::slm::model_serveable(&model_id) {
                log::error!("OPoI: model became unavailable after queuing id={} — discarding request", log_id);
                keryx_miner::runtime_stats::record_inference_failed_request(
                    keryx_miner::runtime_stats::InferenceKind::SoloRequest,
                );
                self.finish_ai_request(key, false);
                continue;
            }
            let traced = tier.map(|tier| {
                let chain = keryx_miner::trace::chain_active(self.last_known_daa);
                keryx_miner::traced::Traced {
                    tier,
                    prompt_len,
                    request_hash,
                    chain,
                    attention: chain && keryx_miner::trace::attention_active(self.last_known_daa),
                }
            });
            info!("OPoI: spawning SLM inference (max_tokens={}{})", max_tokens, if traced.is_some() { ", traced" } else { "" });
            let (tx_done, rx_done) = oneshot::channel::<(Option<Answered>, keryx_miner::runtime_stats::InferenceAttempt)>();
            let task_id = log_id;
            let mut runtime_attempt = keryx_miner::runtime_stats::begin_inference(
                keryx_miner::runtime_stats::InferenceKind::SoloRequest,
                Some(&model_id),
            );
            tokio::task::spawn_blocking(move || {
                // Acquire a concrete card; CUDA's backend pauses only that GPU. Other workers
                // continue consuming current solo templates throughout inference.
                let started = std::time::Instant::now();
                let result = match keryx_miner::slm::acquire_inference_card(&model_id, keryx_miner::slm::DEFAULT_INFERENCE_DEADLINE_MS) {
                    Some(lease) => {
                        let gpu = lease.gpu();
                        runtime_attempt.set_gpu(gpu);
                        let _pause = crate::miner::begin_network_inference_pause();
                        match traced {
                            Some(t) => {
                                match keryx_miner::slm::try_load_and_run_traced_inference_on(gpu, &model_id, &prompt, max_tokens, t) {
                                    Ok(g) => Some(Answered {
                                        text: g.text,
                                        trace: g.trace,
                                        gpu: Some(gpu),
                                        gen_secs: started.elapsed().as_secs_f64(),
                                    }),
                                    Err(e) => {
                                        let _ = account_inference_result(Err(e), &mut runtime_attempt);
                                        None
                                    }
                                }
                            }
                            None => account_inference_result(
                                keryx_miner::slm::try_load_and_run_inference_on(gpu, &model_id, &prompt, max_tokens),
                                &mut runtime_attempt,
                            )
                            .map(|text| Answered { text, trace: None, gpu: Some(gpu), gen_secs: started.elapsed().as_secs_f64() }),
                        }
                    }
                    None => {
                        runtime_attempt.busy();
                        None
                    }
                };
                if result.is_none() {
                    log::warn!("OPoI: inference returned no result for id={} — AiResponse will be skipped", task_id);
                    runtime_attempt.failed();
                }
                let _ = tx_done.send((result, runtime_attempt));
            });
            self.inference_rx =
                Some(InferenceJob { key, request_hash, private, model_id, tier, max_tokens: max_tokens as u32, rx: rx_done });
            return true;
        }
        false
    }

    /// Polls the in-flight inference task. When complete, seals (private) or uploads (plaintext)
    /// the answer and submits a zero-input/zero-output AiResponse transaction.
    /// Returns `true` if inference just finished (regardless of tx success).
    async fn poll_inference(&mut self) -> bool {
        let Some(mut job) = self.inference_rx.take() else {
            return false;
        };
        let log_id = Self::request_log_id(&job.request_hash);
        let (result_opt, mut runtime_attempt) = match job.rx.try_recv() {
            Ok(result) => result,
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                self.inference_rx = Some(job);
                return false;
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                warn!("OPoI: inference task dropped for id={} — request is retryable", log_id);
                self.finish_ai_request(job.key, false);
                return true;
            }
        };
        let InferenceJob { key, request_hash, private, model_id, tier, max_tokens, .. } = job;
        let Some(answered) = result_opt else {
            // Inference returned None: model not ready or think block exhausted max_tokens.
            // Do NOT upload anything to IPFS — skip this AiResponse entirely.
            info!("OPoI: inference produced no result — AiResponse skipped");
            self.finish_ai_request(key, false);
            return true;
        };
        let Answered { text: result, trace, gpu, gen_secs } = answered;
        let trace = match (trace, tier) {
            (Some(trace), Some(tier)) => {
                speeds::note_generation(&model_id, trace.record.n_prompt, trace.record.n_pos.saturating_sub(trace.record.n_prompt), gen_secs);
                Some(TraceContext { model_id, tier, max_tokens, trace, gen_secs, gpu })
            }
            _ => None,
        };

        // The request identity was fixed when the request was queued (H8-gated: TXID or digest).
        info!("OPoI: inference complete, request_hash={}{}", hex::encode(&request_hash[..8]), if trace.is_some() { " (traced)" } else { "" });

        // H14 private request: the answer is sealed to the requester and travels INLINE in the
        // signed AiResponse; the CID field carries the sha2-256 multihash of the sealed body. The
        // plaintext is never published and no IPFS node is involved. H15: a stage-2 trace commits
        // the answer through its token ids, so the sealed plaintext is `answer::encode(text, ids)`.
        let (cid, private_body) = if let Some(p) = private {
            if self.escrow_watcher.is_none() {
                warn!("OPoI: escrow key gone — private AiResponse skipped");
                runtime_attempt.failed();
                self.finish_ai_request(key, false);
                return true;
            }
            let body = match &p {
                PrivateRequest::Opened(response_key) => {
                    if trace.is_none() && self.last_known_daa >= keryx_miner::pom::verifiable_inference_activation_daa() {
                        warn!("OPoI: private answer for id={} has no trace — AiResponse skipped (it would be invalid)", log_id);
                        runtime_attempt.failed();
                        self.finish_ai_request(key, true);
                        return true;
                    }
                    let plain = keryx_miner::traced::answer_body(&result, trace.as_ref().map(|t| &t.trace));
                    keryx_inference::seal_response(response_key, &request_hash, &plain)
                }
                PrivateRequest::Unreadable => result.clone().into_bytes(),
            };
            let cap = if keryx_miner::trace::chain_active(self.last_known_daa) {
                keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN
            } else {
                keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN_H14
            };
            if body.len() > cap {
                warn!("OPoI: sealed answer for id={} is {} bytes (> {} inline cap) — AiResponse skipped", log_id, body.len(), cap);
                runtime_attempt.failed();
                self.finish_ai_request(key, false);
                return true;
            }
            (crate::ipfs::sha256_multihash(&body), Some(body))
        } else {
            let ipfs_url = self.ipfs_url.clone();
            let result_clone = result.clone();
            let cid = match tokio::task::spawn_blocking(move || crate::ipfs::upload_with_recovery(&result_clone, &ipfs_url)).await {
                Ok(Ok(cid)) => cid,
                Ok(Err(e)) => {
                    warn!("OPoI: IPFS upload failed: {} — AiResponse tx skipped", e);
                    runtime_attempt.failed();
                    self.finish_ai_request(key, false);
                    return true;
                }
                Err(e) => {
                    warn!("OPoI: IPFS spawn_blocking failed: {} — AiResponse tx skipped", e);
                    runtime_attempt.failed();
                    self.finish_ai_request(key, false);
                    return true;
                }
            };
            (cid, None)
        };
        keryx_miner::runtime_stats::record_inference_prepared();

        let challenge_window_end = self.last_known_daa + 1000;
        let response_length = result.split_whitespace().count() as u32;
        // The trace commitment rides after the private body only.
        let trace = trace.filter(|_| private_body.is_some());
        let commitment = trace.as_ref().map(|t| t.trace.commitment());
        // H6 service-bond era: sign the response with the escrow key (payload V2) so it counts
        // as served for the tier cohort — an unsigned response no longer cancels a strike. The
        // era rule mirrors the node's: V2 is rejected before the gate, so v1 is kept below it.
        let v2 = self.last_known_daa >= keryx_miner::pom::pom_v3_activation_daa();
        let resp = match (&self.escrow_watcher, v2) {
            (Some(w), true) => Self::signed_response(w, request_hash, challenge_window_end, cid, response_length, private_body, commitment),
            (None, true) => {
                warn!("OPoI: no escrow key configured — submitting an unsigned (v1) response; it will NOT count for the service bond");
                keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length)
            }
            (_, false) => keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length),
        };
        info!(
            "OPoI: uploading response CID={}, challenge_window_end={}{}{}{}",
            resp.cid_v0(),
            challenge_window_end,
            if resp.responder.is_some() { " (signed, V2)" } else { "" },
            if resp.private_body.is_some() { " (private body inline)" } else { "" },
            if resp.trace.is_some() { " (traced)" } else { "" }
        );

        let rpc_tx = crate::proto::RpcTransaction {
            version: 0,
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
            subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_RESPONSE_HEX.to_string(),
            gas: 0,
            payload: hex::encode(resp.serialize()),
            mass: 0,
            verbose_data: None,
        };
        match Self::compute_rpc_txid(&rpc_tx) {
            Some(txid) => {
                if let Some(t) = trace.filter(|_| resp.trace.is_some()) {
                    info!(
                        "Trace: response {} commits root {} ({} prompt / {} positions, stage {}{})",
                        txid,
                        hex::encode(&t.trace.record.trace_root[..8]),
                        t.trace.record.n_prompt,
                        t.trace.record.n_pos,
                        t.trace.record.stage,
                        if t.trace.seq.is_some() { ", sequence leaves" } else { "" }
                    );
                    let pending = PendingTrace {
                        model_id: t.model_id,
                        tier: t.tier,
                        request_hash,
                        max_tokens: t.max_tokens,
                        trace: Arc::new(t.trace),
                        submitted_daa: self.last_known_daa,
                        accepted_hash: None,
                        anchor_hash: self.chain_log.back().map(|e| e.hash.clone()),
                        gen_secs: Some(t.gen_secs),
                        draw: None,
                        opening: None,
                        built: Vec::new(),
                        built_phase: None,
                        reorgs: 0,
                        node_awaits: None,
                        attn: None,
                        response_tx: Some(rpc_tx.clone()),
                        gpu: t.gpu,
                    };
                    self.pending_traces.insert(txid.clone(), pending);
                    // The response leaves only once its trace is on disk: a crash in between would
                    // leave an audit nothing can open.
                    let (done, written) = std::sync::mpsc::channel();
                    if let Some(record) = self.trace_record(&txid) {
                        self.trace_store.save_queued_notify(record, done);
                    }
                    self.persisting.push((written, txid, rpc_tx));
                } else {
                    self.ai_response_inflight.insert(txid, (rpc_tx.clone(), 1, std::time::Instant::now()));
                    if let Err(e) = self.client_send(KaspadMessage::submit_transaction(rpc_tx)).await {
                        warn!("OPoI: failed to send AiResponse tx: {}", e);
                        runtime_attempt.failed();
                        self.finish_ai_request(key, false);
                        return true;
                    }
                }
            }
            None => {
                warn!("OPoI: cannot compute the AiResponse txid — AiResponse skipped");
                runtime_attempt.failed();
                self.finish_ai_request(key, false);
                return true;
            }
        }
        runtime_attempt.served(response_length as usize);
        self.ai_answered_by_me.insert(request_hash);

        // Register inference escrow outpoint for auto-claim after the challenge window.
        if let Some((txid, inference_reward)) = self.finish_ai_request(key, true) {
            if let Some(w) = self.escrow_watcher.as_mut() {
                w.track_inference_escrow(txid, self.last_known_daa, inference_reward);
            }
        }

        true
    }

    /// V2 AiResponse signed with the escrow key. The inline private body and the trace commitment
    /// are attached BEFORE signing: the responder signature covers `v1 head || extensions` (node
    /// collateral::responder_signature_message), so neither can be swapped under a signed head.
    fn signed_response(
        w: &crate::escrow::EscrowWatcher,
        request_hash: [u8; 32],
        challenge_window_end: u64,
        cid: [u8; 34],
        response_length: u32,
        private_body: Option<Vec<u8>>,
        trace: Option<keryx_inference::TraceCommitment>,
    ) -> keryx_inference::AiResponsePayload {
        let mut unsigned = keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length);
        if let Some(body) = &private_body {
            unsigned = unsigned.with_private_body(body.clone());
        }
        if let Some(c) = trace {
            unsigned = unsigned.with_trace(c);
        }
        let responder = w.sign_responder(&unsigned.signed_bytes());
        let resp = keryx_inference::AiResponsePayload::new_v2(request_hash, challenge_window_end, cid, response_length, responder);
        let resp = match private_body {
            Some(body) => resp.with_private_body(body),
            None => resp,
        };
        match trace {
            Some(c) => resp.with_trace(c),
            None => resp,
        }
    }

    /// Logs this miner's service-bond standing when it changes: strike count, burns awaiting
    /// finality and production suspensions, matched by payout-address identity.
    fn report_service_strikes(&mut self, resp: &crate::proto::GetServiceStrikesResponseMessage) {
        let Some(me) = self.service_identity.as_deref() else { return };
        let strike = resp.strikes.iter().find(|s| s.miner.eq_ignore_ascii_case(me));
        let suspension = resp.suspended.iter().find(|s| s.miner.eq_ignore_ascii_case(me));
        let burns: Vec<_> = resp.pending_burns.iter().filter(|b| b.miner.eq_ignore_ascii_case(me)).collect();
        let consecutive_misses = strike.map_or(0, |entry| entry.consecutive_misses as u64);
        let last_strike_daa = strike.map(|entry| entry.last_strike_daa_score);
        let burned_claims: u64 = burns.iter().map(|entry| entry.burned_claims as u64).sum();
        let burned_sompi: u64 = burns.iter().map(|entry| entry.burned_sompi).sum();
        let suspended_until_daa = suspension.map(|entry| entry.until_daa_score);
        keryx_miner::runtime_stats::service_bond_update(
            consecutive_misses,
            last_strike_daa,
            burned_claims,
            burned_sompi,
            suspended_until_daa,
        );
        let status = if strike.is_none() && suspension.is_none() && burns.is_empty() {
            "clear".to_string()
        } else {
            let mut parts = Vec::new();
            if let Some(s) = strike {
                parts.push(format!("strike {} (last at daa {})", s.consecutive_misses, s.last_strike_daa_score));
            }
            if !burns.is_empty() {
                let claims: u32 = burns.iter().map(|b| b.burned_claims).sum();
                let sompi: u64 = burns.iter().map(|b| b.burned_sompi).sum();
                parts.push(format!(
                    "{} escrow claims / {:.2} KRX burning at finality",
                    claims,
                    sompi as f64 / 100_000_000.0
                ));
            }
            if let Some(s) = suspension {
                parts.push(format!("production suspended until daa {}", s.until_daa_score));
            }
            parts.join("; ")
        };
        if self.strike_status.as_deref() != Some(status.as_str()) {
            match status.as_str() {
                "clear" => info!("service-bond: no strikes against this miner"),
                s => warn!("service-bond: {}", s),
            }
            self.strike_status = Some(status);
        }
    }

    async fn handle_message(&mut self, msg: Payload, miner: &mut MinerManager) -> Result<(), Error> {
        match msg {
            // BlockAdded: scan confirmed block for AiRequests and escrow UTXOs.
            // Do NOT trigger a new block template here — NewBlockTemplate handles that.
            Payload::BlockAddedNotification(notif) => {
                if let Some(block) = notif.block {
                    if !block.transactions.is_empty() {
                        // Full block — scan directly.
                        self.scan_txs_for_ai_requests(
                            &block.transactions.clone(),
                            block.header.as_ref().map_or(0, |h| h.daa_score),
                        );
                        self.try_start_inference();
                        // Escrow: check for new escrow UTXOs and mature claims.
                        let claim_tx = self.escrow_watcher.as_mut().and_then(|w| w.handle_block(&block));
                        self.report_pending_escrow();
                        if let Some(tx) = claim_tx {
                            if let Err(error) = self.client_send(KaspadMessage::submit_transaction(tx)).await {
                                keryx_miner::runtime_stats::escrow_transport_failed();
                                return Err(Box::new(error));
                            }
                        }
                    } else {
                        // Transactions absent — fetch the full block from the node.
                        let hash = block.verbose_data.as_ref().map(|v| v.hash.clone()).unwrap_or_default();
                        if !hash.is_empty() {
                            self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
                        }
                    }
                }
            }
            Payload::NewBlockTemplateNotification(_) => self.client_get_block_template().await?,
            Payload::GetServiceStrikesResponse(resp) => match resp.error.as_ref() {
                Some(e) => {
                    keryx_miner::runtime_stats::service_bond_unavailable();
                    warn!("service-bond status unavailable: {}", e.message)
                }
                None => self.report_service_strikes(&resp),
            },
            Payload::GetBlockTemplateResponse(template) => {
                // Track DAA score for challenge_window_end computation.
                let template_daa = template.block.as_ref().and_then(|b| b.header.as_ref()).map(|h| h.daa_score);
                let template_network_difficulty = template
                    .block
                    .as_ref()
                    .and_then(|block| block.header.as_ref())
                    .and_then(|header| crate::target::network_difficulty_from_compact_target(header.bits));
                if template.block.is_some() {
                    keryx_miner::runtime_stats::record_job(
                        self.runtime_generation,
                        template_daa,
                        Some(template.is_synced),
                        template_network_difficulty,
                    );
                }
                if let Some(daa) = template_daa {
                    if daa > self.last_known_daa {
                        self.last_known_daa = daa;
                    }
                    // OPoI v2 hardfork: advance the served lineup when the chain crosses H, so the
                    // uncensored (v2) models load and post-fork PoM-PoW has weights resident. Solo
                    // path (mirrors the stratum notify handlers). Cheap + idempotent per template.
                    keryx_miner::slm::advance_lineup_if_due(daa);
                }
                // Handle node-issued inference challenge: spawn an inference task if a new
                // challenge arrived and no challenge is already in flight.
                if !template.inference_challenge.is_empty() && self.challenge_inference.is_none() {
                    let challenge = template.inference_challenge.clone();
                    let mut parts = challenge.splitn(2, ':');
                    let model_id_hex = parts.next().unwrap_or("").to_string();
                    let nonce_hex = parts.next().unwrap_or("").to_string();
                    if let Ok(model_id_bytes) = hex::decode(&model_id_hex) {
                        if model_id_bytes.len() == 32 {
                            let mut model_id = [0u8; 32];
                            model_id.copy_from_slice(&model_id_bytes);
                            if keryx_miner::slm::model_serveable(&model_id) {
                                info!(
                                    "OPoI: challenge received model={:.8} nonce={:.8} — spawning inference",
                                    model_id_hex, nonce_hex
                                );
                                let prompt =
                                    format!("Keryx inference challenge {}: briefly describe what you are.", nonce_hex);
                                let (tx_done, rx_done) = oneshot::channel::<(
                                    Option<String>,
                                    keryx_miner::runtime_stats::InferenceAttempt,
                                )>();
                                let mut runtime_attempt = keryx_miner::runtime_stats::begin_inference(
                                    keryx_miner::runtime_stats::InferenceKind::SoloChallenge,
                                    Some(&model_id),
                                );
                                tokio::task::spawn_blocking(move || {
                                    let result = match keryx_miner::slm::acquire_inference_card(
                                        &model_id,
                                        keryx_miner::slm::DEFAULT_INFERENCE_DEADLINE_MS,
                                    ) {
                                        Some(lease) => {
                                            let gpu = lease.gpu();
                                            runtime_attempt.set_gpu(gpu);
                                            let _pause = crate::miner::begin_network_inference_pause();
                                            account_inference_result(
                                                keryx_miner::slm::try_load_and_run_inference_on(
                                                    gpu, &model_id, &prompt, 64,
                                                ),
                                                &mut runtime_attempt,
                                            )
                                        }
                                        None => {
                                            runtime_attempt.busy();
                                            None
                                        }
                                    };
                                    if result.as_ref().map_or(true, String::is_empty) {
                                        runtime_attempt.failed();
                                    }
                                    let _ = tx_done.send((result, runtime_attempt));
                                });
                                self.challenge_inference = Some(ChallengeInference::running(challenge, rx_done));
                            } else {
                                warn!("OPoI: challenge for unready model={:.8} — cannot respond", model_id_hex);
                                keryx_miner::runtime_stats::record_inference_failed_request(
                                    keryx_miner::runtime_stats::InferenceKind::SoloChallenge,
                                );
                            }
                        }
                    }
                }
                // Poll in-flight inference; if done, submit AiResponse tx then get fresh template.
                if self.poll_inference().await {
                    self.client_get_block_template().await?;
                    return Ok(());
                }
                // OPoI is mandatory: refuse to mine if no models are ready.
                // Keryx core invariant — no inference, no PoW.
                if !keryx_miner::slm::has_proven_serveable_model() {
                    self.template_deferred_for_models = true;
                    if self.last_known_daa % 200 == 0 {
                        log::warn!("OPoI: no models ready — mining suspended until model files are available");
                    }
                    miner.process_block(None).await?;
                    return Ok(());
                }
                if let Some(ref block) = template.block {
                    self.scan_txs_for_ai_requests(
                        &block.transactions.clone(),
                        block.header.as_ref().map_or(0, |h| h.daa_score),
                    );
                }
                self.try_start_inference();
                // A queued task may wait for another inference to release its card while PoW keeps
                // running. Only backends requiring a global pause gate templates here; CUDA
                // continues publishing them while its exact serving GPU is drained/paused.
                if crate::miner::inference_pause_active() {
                    miner.process_block(None).await?;
                    return Ok(());
                }
                match (template.block, template.is_synced, template.error) {
                    (Some(b), true, None) => miner.process_block(Some(FullBlock(Box::new(b)))).await?,
                    (Some(b), false, None) if self.mine_when_not_synced => {
                        miner.process_block(Some(FullBlock(Box::new(b)))).await?
                    }
                    (_, false, None) => miner.process_block(None).await?,
                    (_, _, Some(e)) => {
                        return Err(format!("GetTemplate returned with an error: {:?}", e).into());
                    }
                    (None, true, None) => error!("No block and No Error!"),
                }
            }
            // GetBlock response: either a boot-time escrow-validation answer, or a full block
            // we requested (BlockAdded / VirtualChainChanged) — scanned for AiRequests and
            // escrow UTXOs.
            Payload::GetBlockResponse(msg) => {
                let mut was_validation = false;
                if let Some(e) = msg.error {
                    // A chain block missing from the trace chain log would shift every seed block
                    // after it: asked again until the node answers or a reorg removes it.
                    if let Some(c) = self.chain_log.iter().find(|c| c.daa.is_none() && e.message.contains(c.hash.as_str())) {
                        let failures = self.chain_block_retry.iter().find(|(_, h, _)| *h == c.hash).map_or(0, |(_, _, f)| *f) + 1;
                        let delay = CHAIN_RESTORE_RETRY_BASE.saturating_mul(1 << (failures - 1).min(4)).min(CHAIN_RESTORE_RETRY_MAX);
                        warn!("Trace: chain block {} unavailable — asked again in {} s", c.hash, delay.as_secs());
                        let hash = c.hash.clone();
                        self.chain_block_retry.retain(|(_, h, _)| *h != hash);
                        self.chain_block_retry.push((std::time::Instant::now() + delay, hash, failures));
                    }
                    if let Some(h) = self.chain_restore_pending.iter().find(|h| e.message.contains(h.as_str())).cloned() {
                        self.chain_restore_pending.remove(&h);
                        self.chain_restore_pump().await?;
                    }
                    // Validation answer: "cannot find header <hash>" — unknown to this
                    // node (pruned or not yet synced), the entries are kept.
                    was_validation =
                        self.escrow_watcher.as_mut().map_or(false, |w| w.on_block_validation_error(&e.message));
                    if !was_validation {
                        warn!("GetBlockResponse error: {}", e.message);
                    }
                } else if let Some(block) = msg.block {
                    let hash = block.verbose_data.as_ref().map(|v| v.hash.clone()).unwrap_or_default();
                    if let Some(daa) = block.header.as_ref().map(|h| h.daa_score) {
                        self.note_chain_block(&hash, daa);
                        if !self.pending_traces.is_empty() {
                            self.resolve_traces();
                        }
                    }
                    if self.chain_restore_pending.remove(&hash) {
                        self.chain_restore_pump().await?;
                    }
                    // Chain membership from the node's live verdict: a stored-but-reorged
                    // block must purge its entries just like a missing one.
                    let is_chain = block.verbose_data.as_ref().map_or(false, |v| v.is_chain_block);
                    was_validation =
                        self.escrow_watcher.as_mut().map_or(false, |w| w.consume_validation_ok(&hash, is_chain));
                    if !was_validation {
                        self.scan_txs_for_ai_requests(
                            &block.transactions.clone(),
                            block.header.as_ref().map_or(0, |h| h.daa_score),
                        );
                        self.try_start_inference();
                        let claim_tx = self.escrow_watcher.as_mut().and_then(|w| w.handle_block(&block));
                        self.report_pending_escrow();
                        if let Some(tx) = claim_tx {
                            if let Err(error) = self.client_send(KaspadMessage::submit_transaction(tx)).await {
                                keryx_miner::runtime_stats::escrow_transport_failed();
                                return Err(Box::new(error));
                            }
                        }
                    }
                }
                // Self-paced validation flow: every consumed answer pulls the next
                // queued request, keeping at most VALIDATION_WINDOW in flight.
                if was_validation {
                    if let Some(hash) = self.validation_queue.pop_front() {
                        self.client_send(GetBlockRequestMessage { hash, include_transactions: false }).await?;
                    }
                }
            }
            Payload::SubmitBlockResponse(res) => {
                // Feed the stats API the same way the stratum client does — a solo block IS this
                // mode's "share". Without this the API showed 0/0 while the log showed accepts.
                let stats = crate::client::stratum::share_stats_init();
                match res.error {
                    None => {
                        stats.accepted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        keryx_miner::runtime_stats::record_solo_block_accepted();
                        info!("block submitted successfully!");
                    }
                    Some(e) => {
                        stats.rejected_other.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        keryx_miner::runtime_stats::record_solo_block_rejected();
                        warn!("Failed submitting block: {:?}", e);
                    }
                }
            }
            Payload::SubmitTransactionResponse(res) => {
                // Escrow claims and OPoI submissions share this stream. Match responses to
                // in-flight claims by identity (txid, or the txid embedded in the rejection
                // text) — attributing by position slashed valid escrow entries before.
                use crate::escrow::SubmitResponseOutcome;
                let err = res.error.as_ref().map(|e| e.message.clone());
                let outcome = self.escrow_watcher.as_mut().map_or(SubmitResponseOutcome::NotOurs, |w| {
                    w.on_submit_response(&res.transaction_id, err.as_deref())
                });
                match outcome {
                    SubmitResponseOutcome::Accepted { outputs, amount_sompi } => {
                        info!("Escrow claim accepted: {} output(s), {:.8} KRX", outputs, amount_sompi as f64 / 1e8);
                        self.report_pending_escrow();
                    }
                    SubmitResponseOutcome::Handled => {}
                    SubmitResponseOutcome::NotOurs => {
                        if !self.on_trace_submit_response(&res.transaction_id, err.as_deref()) {
                            if let Some(e) = err {
                                warn!("OPoI: submit_transaction error: {}", e);
                            }
                        }
                    }
                }
            }
            Payload::GetInfoResponse(info) => {
                info!("Keryxd version: {}", info.server_version);
                self.node_awaited_openings = version_at_least(&info.server_version, AWAITED_OPENINGS_SINCE);
                // SOLO SAFETY GATE: keryxd < 1.4.4 has a consensus bug ("coin-age maturation
                // entry lost on tx re-accepted during reorg", fixed by node commit 0a7d5473)
                // that makes the node reject blocks containing coin-age spends its own
                // maturation table forgot — and escrow CLAIM transactions are exactly that
                // class. Since the batched-claim engine keeps claims flowing, mining solo
                // against a pre-1.4.4 node yields recurring "Block was not submitted: block
                // invalid" on templates carrying our own claims. Hold claims on old nodes
                // (tracking continues; claims resume automatically once the node is upgraded
                // or with KERYX_ESCROW_FORCE=1 to override).
                if let Some(w) = self.escrow_watcher.as_mut() {
                    let old_node = version_lt(&info.server_version, (1, 4, 4));
                    let forced = std::env::var("KERYX_ESCROW_FORCE").map(|v| v == "1").unwrap_or(false);
                    if old_node && !forced {
                        w.set_claims_held(true);
                        log::warn!(
                            "ESCROW: keryxd {} is older than 1.4.4 — claim submission is HELD to avoid \
                             'block invalid' rejections from the pre-1.4.4 coin-age maturation bug. \
                             UPGRADE keryxd to >= 1.4.4 to claim (escrow outputs keep accruing safely; \
                             KERYX_ESCROW_FORCE=1 overrides at your own risk).",
                            info.server_version
                        );
                    } else {
                        w.set_claims_held(false);
                    }
                }
                // Register for all notification types:
                // - NewBlockTemplate drives the mining loop
                // - BlockAdded lets us scan confirmed blocks for AiRequests
                //   that were confirmed before the miner saw them in mempool
                // - VirtualChainChanged drives escrow tracking: only chain-block coinbases
                //   materialize UTXOs, so escrow outputs are tracked from chain blocks only
                self.client_send(NotifyNewBlockTemplateRequestMessage {}).await?;
                self.client_send(NotifyBlockAddedRequestMessage {}).await?;
                // Accepted transaction ids: which chain block accepted our responses and openings
                // (the trace draw is seeded from the chain after the accepting block).
                self.client_send(NotifyVirtualSelectedParentChainChangedRequestMessage { include_accepted_transaction_ids: true })
                    .await?;
                // Boot-time escrow-state validation: check every referenced block against
                // the node so ghost entries (orphaned-chain coinbases) are purged before
                // any claim ships. Send an initial slice; each answer sends the next.
                if let Some(hashes) = self.escrow_watcher.as_mut().map(|w| w.start_state_validation()) {
                    self.validation_queue = hashes.into();
                    for _ in 0..VALIDATION_WINDOW {
                        if let Some(hash) = self.validation_queue.pop_front() {
                            self.client_send(GetBlockRequestMessage { hash, include_transactions: false }).await?;
                        }
                    }
                }
                self.client_get_block_template().await?;
            }
            Payload::NotifyNewBlockTemplateResponse(res) => match res.error {
                None => info!("Registered for new template notifications"),
                Some(e) => error!("Failed registering for new template notifications: {:?}", e),
            },
            Payload::NotifyBlockAddedResponse(res) => match res.error {
                None => info!("Registered for block added notifications (AI request scanning)"),
                Some(e) => error!("Failed registering for block added notifications: {:?}", e),
            },
            Payload::NotifyVirtualSelectedParentChainChangedResponse(res) => match res.error {
                None => {
                    info!("Registered for virtual chain notifications (escrow tracking, trace audits)");
                    self.start_chain_restore().await?;
                }
                Some(e) => error!("Failed registering for virtual chain notifications: {:?}", e),
            },
            Payload::GetAwaitedOpeningsResponse(resp) => match resp.error.as_ref() {
                Some(e) => log::debug!("Trace: awaited openings unavailable: {}", e.message),
                None => self.apply_awaited_openings(&resp),
            },
            Payload::GetVirtualSelectedParentChainFromBlockResponse(msg) => {
                let Some(start) = self.chain_restore_starts.pop_front() else { return Ok(()) };
                match msg.error {
                    Some(e) => self.on_chain_restore_failed(start, &e.message),
                    None => {
                        self.chain_restore_failures.remove(&start);
                        self.chain_restore_continuations.remove(&start);
                        if msg.added_chain_block_hashes.len() >= CHAIN_RESTORE_PAGE_MIN {
                            if let Some(last) = msg.added_chain_block_hashes.last() {
                                self.chain_restore_starts.push_front(last.clone());
                                self.chain_restore_continuations.insert(last.clone());
                            }
                        }
                        self.place_restored_chain(&start, &msg.removed_chain_block_hashes, msg.added_chain_block_hashes);
                        self.note_restored_acceptances(&msg.accepted_transaction_ids);
                    }
                }
                self.resolve_traces();
                self.request_next_chain_restore().await?;
                self.chain_restore_pump().await?;
            }
            // Virtual chain advanced: fetch every added chain block in full. Their coinbases
            // are the only ones that materialize UTXOs, so escrow tracking feeds off this
            // stream (handle_block gates tracking on is_chain_block). Removed chain blocks
            // are ignored: entries from reorged-out blocks fail their claims as orphans and
            // are cleaned up by the existing retry/slash machinery.
            Payload::VirtualSelectedParentChainChangedNotification(notif) => {
                self.on_virtual_chain_changed(&notif);
                for hash in notif.added_chain_block_hashes {
                    self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
                }
            }
            msg => info!("got unknown msg: {:?}", msg),
        }
        Ok(())
    }
}


// ── 0.15.3: H15 traced responses, audits and openings (port of upstream keryx-miner v0.5.8 grpc.rs) ──

/// Seconds between two resubmissions of an AiResponse the mempool has not accepted yet.
const AI_RESPONSE_RETRY_SECS: u64 = 3;
/// Most submissions of one AiResponse.
const AI_RESPONSE_MAX_ATTEMPTS: u32 = 100;
/// Fast opening resubmissions before slowing down.
const OPENING_FAST_RETRIES: u32 = 10;
const OPENING_SLOW_RETRY_SECS: u64 = 60;
/// A response accepted before its request is sent again this long after the request lands.
const EARLY_RESPONSE_RESEND_DELAY: std::time::Duration = std::time::Duration::from_secs(125);
const EARLY_RESPONSE_RESENDS: u32 = 10;
/// DAA a drawn but unaudited trace is kept past its seed block.
const TRACE_UNAUDITED_GRACE_DAA: u64 = 600;
/// Chain blocks kept in the log past the oldest one a pending trace needs.
const CHAIN_LOG_TAIL: usize = 512;
/// Unsigned opening bodies kept per response, by seed block (a reorg may bring a seed back).
const BUILT_OPENINGS_KEPT: usize = 4;
/// Base of the deadline margin an audit replay is started with ahead of a request.
const REPLAY_URGENT_BASE_DAA: u64 = 1_500;
/// A chain rebuild page this long or longer continues from its last block.
const CHAIN_RESTORE_PAGE_MIN: usize = 100;
/// DAA a submitted trace waits for its draw before it is dropped.
const TRACE_UNDRAWN_LIMIT_DAA: u64 = 72_000;
const CHAIN_RESTORE_INFLIGHT: usize = 8;
const CHAIN_RESTORE_RETRY_BASE: std::time::Duration = std::time::Duration::from_secs(5);
const CHAIN_RESTORE_RETRY_MAX: std::time::Duration = std::time::Duration::from_secs(60);
const REPLAY_RETRY_BASE: std::time::Duration = std::time::Duration::from_secs(30);
const REPLAY_RETRY_MAX: std::time::Duration = std::time::Duration::from_secs(600);
/// Node version that answers `GetAwaitedOpenings`.
const AWAITED_OPENINGS_SINCE: (u32, u32, u32) = (1, 6, 6);
/// Latest opening deadline after a response's acceptance on the node; matches the node.
const AUDIT_SPAN_DAA: u64 = 270_000;
/// Attention phases the node's span test reserves; matches the node.
const AUDIT_PHASES_RESERVED: u64 = 3;
/// Seconds between two `GetAwaitedOpenings` polls.
const AWAITED_POLL_SECS: u64 = 10;
/// Response txids per `GetAwaitedOpenings` poll; the node accepts up to 1 024.
const AWAITED_POLL_MAX: usize = 1_024;
/// How long an opening replay waits for its card to be free.
const REPLAY_CARD_DEADLINE_MS: u64 = 120_000;

/// Earliest retry of a replay that failed `failures` times.
fn replay_backoff(failures: u32) -> std::time::Instant {
    std::time::Instant::now() + REPLAY_RETRY_BASE.saturating_mul(1 << failures.saturating_sub(1).min(5)).min(REPLAY_RETRY_MAX)
}

fn hash32(hex_str: &str) -> Option<[u8; 32]> {
    hex::decode(hex_str).ok()?.try_into().ok()
}

/// Whether the leading `major.minor.patch` of `version` is at least `min`.
fn version_at_least(version: &str, min: (u32, u32, u32)) -> bool {
    let mut parts = version.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse::<u32>().unwrap_or(0));
    let v = (parts.next().unwrap_or(0), parts.next().unwrap_or(0), parts.next().unwrap_or(0));
    v >= min
}

/// The trace of a finished inference, on its way into the AiResponse.
struct TraceContext {
    model_id: [u8; 32],
    tier: u8,
    max_tokens: u32,
    trace: keryx_miner::trace::ResponseTrace,
    gen_secs: f64,
    gpu: Option<usize>,
}

/// A traced response, kept until its draw is settled and its opening window closes.
struct PendingTrace {
    model_id: [u8; 32],
    tier: u8,
    request_hash: [u8; 32],
    max_tokens: u32,
    trace: Arc<keryx_miner::trace::ResponseTrace>,
    submitted_daa: u64,
    accepted_hash: Option<String>,
    anchor_hash: Option<String>,
    gen_secs: Option<f64>,
    draw: Option<TraceDraw>,
    opening: Option<keryx_miner::trace_store::StoredOpening>,
    built: Vec<(String, Vec<u8>)>,
    built_phase: Option<(String, Vec<u64>, Vec<u8>)>,
    reorgs: u32,
    node_awaits: Option<bool>,
    attn: Option<AttnPhases>,
    response_tx: Option<crate::proto::RpcTransaction>,
    /// Card that generated the trace (its replays must run there); unknown once restored.
    gpu: Option<usize>,
}

/// The attention phases of an audited response, phase order.
struct AttnPhases {
    bodies: Vec<Vec<u8>>,
    txids: Vec<String>,
    payloads: Vec<String>,
    accepted: Vec<Option<String>>,
    path: Vec<u64>,
    child_blocks: Vec<Option<String>>,
    deadline_daa: u64,
    opened: bool,
    failures: u32,
    retry_at: Option<std::time::Instant>,
    dropped: HashSet<String>,
    late: HashMap<String, String>,
}

impl AttnPhases {
    fn new(body: Vec<u8>, txid: String, payload: String, deadline_daa: u64) -> Self {
        Self {
            bodies: vec![body],
            txids: vec![txid],
            payloads: vec![payload],
            accepted: vec![None],
            path: Vec::new(),
            child_blocks: Vec::new(),
            deadline_daa,
            opened: false,
            failures: 0,
            retry_at: None,
            dropped: Default::default(),
            late: Default::default(),
        }
    }

    fn restored(s: keryx_miner::trace_store::StoredAttn, txid: String) -> Option<Self> {
        let last = keryx_verifiable_inference::opening::AttnOpening::decode(s.bodies.last()?)?;
        let n = s.bodies.len();
        if last.path.len() + 1 != n {
            return None;
        }
        let mut accepted = vec![Some(String::new()); n - 1];
        accepted.push(s.accepted_hash);
        let mut txids = vec![String::new(); n - 1];
        txids.push(txid);
        let mut payloads = vec![String::new(); n - 1];
        payloads.push(s.payload);
        Some(Self {
            bodies: s.bodies,
            txids,
            payloads,
            accepted,
            child_blocks: vec![None; last.path.len()],
            path: last.path,
            deadline_daa: s.deadline_daa,
            opened: false,
            failures: 0,
            retry_at: None,
            dropped: Default::default(),
            late: Default::default(),
        })
    }

    fn stored(&self) -> keryx_miner::trace_store::StoredAttn {
        keryx_miner::trace_store::StoredAttn {
            bodies: self.bodies.clone(),
            payload: self.payloads.last().cloned().unwrap_or_default(),
            deadline_daa: self.deadline_daa,
            accepted_hash: self.accepted.last().cloned().flatten(),
        }
    }

    fn last_partials(&self) -> Vec<u8> {
        self.bodies.last().and_then(|b| keryx_verifiable_inference::opening::AttnOpening::decode(b)).map(|o| o.partials).unwrap_or_default()
    }

    fn awaiting_draw(&self) -> Option<&str> {
        if self.path.len() + 1 != self.bodies.len() || self.last_partials().is_empty() {
            return None;
        }
        self.accepted.last()?.as_deref().filter(|h| !h.is_empty())
    }

    fn next_phase_owed(&self) -> bool {
        self.path.len() == self.bodies.len()
    }

    fn ready(&self) -> bool {
        self.path.len() == self.bodies.len() && !self.opened && self.retry_at.is_none_or(|t| t <= std::time::Instant::now())
    }

    fn on_removed(&mut self, removed: &HashSet<&str>) -> Option<Vec<String>> {
        let gone = |h: &Option<String>| h.as_deref().is_some_and(|h| removed.contains(h));
        let k = self.accepted.iter().position(gone);
        let i = self.child_blocks.iter().position(gone);
        let cut = match (k, i) {
            (Some(k), Some(i)) => k.min(i),
            (Some(c), None) | (None, Some(c)) => c,
            (None, None) => return None,
        };
        let dropped = self.txids.split_off(cut + 1);
        self.dropped.extend(dropped.iter().cloned());
        self.bodies.truncate(cut + 1);
        self.payloads.truncate(cut + 1);
        self.accepted.truncate(cut + 1);
        self.path.truncate(cut);
        self.child_blocks.truncate(cut);
        if k == Some(cut) {
            self.accepted[cut] = None;
        }
        self.opened = false;
        self.failures = 0;
        Some(dropped)
    }
}

/// The draw of an accepted response once its seed block is on the chain.
struct TraceDraw {
    seed_hash: String,
    seed_daa: u64,
    accepted_daa: u64,
    seed: [u8; 32],
    audited: bool,
    deadline_daa: u64,
    opened: bool,
    replay_failures: u32,
    retry_at: Option<std::time::Instant>,
}

impl TraceDraw {
    fn replay_ready(&self) -> bool {
        self.audited && !self.opened && self.retry_at.is_none_or(|t| t <= std::time::Instant::now())
    }
}

/// A chain block in chain order; its DAA score arrives with the GetBlock answer.
struct ChainEntry {
    hash: String,
    daa: Option<u64>,
}

/// The replay of an audited response, running on the model's GPU.
struct ReplayJob {
    response_txid: String,
    seed_hash: String,
    path: Option<Vec<u64>>,
    started: std::time::Instant,
    rx: oneshot::Receiver<Option<Vec<u8>>>,
}

/// An opening the mempool has not accepted yet.
struct OpeningInflight {
    tx: crate::proto::RpcTransaction,
    response_txid: String,
    deadline_daa: u64,
    attempts: u32,
    last: std::time::Instant,
}

/// The seed block of a response accepted by chain block `accepted`.
fn seed_block_after<'a>(log: &'a VecDeque<ChainEntry>, accepted: &str) -> Option<(&'a str, u64, u64)> {
    let start = log.iter().position(|e| e.hash == accepted)?;
    let accepted_daa = log[start].daa?;
    for e in log.iter().skip(start + 1) {
        let daa = e.daa?;
        if daa >= accepted_daa + keryx_miner::trace::TRACE_SEED_DELAY_DAA {
            return Some((e.hash.as_str(), daa, accepted_daa));
        }
    }
    None
}

/// Puts `added`, the chain after `start` in chain order, right behind `start` in `log`.
fn place_chain_after(log: &mut VecDeque<ChainEntry>, start: &str, added: Vec<String>) -> Vec<String> {
    let pos = log.iter().position(|e| e.hash == start).unwrap_or(usize::MAX);
    let added_set: HashSet<&str> = added.iter().map(String::as_str).collect();
    let mut known: HashMap<String, Option<u64>> = HashMap::new();
    let mut kept: VecDeque<ChainEntry> = VecDeque::with_capacity(log.len() + added.len());
    let mut at = 0;
    for (i, e) in std::mem::take(log).into_iter().enumerate() {
        if added_set.contains(e.hash.as_str()) {
            known.insert(e.hash, e.daa);
            continue;
        }
        kept.push_back(e);
        if i == pos {
            at = kept.len();
        }
    }
    if pos == usize::MAX {
        at = kept.len();
    }
    let mut unknown = Vec::new();
    for (k, hash) in added.into_iter().enumerate() {
        let daa = known.remove(&hash).flatten();
        if daa.is_none() {
            unknown.push(hash.clone());
        }
        kept.insert(at + k, ChainEntry { hash, daa });
    }
    *log = kept;
    unknown
}

fn opening_rpc_tx(payload: String) -> crate::proto::RpcTransaction {
    crate::proto::RpcTransaction {
        version: 0,
        inputs: vec![],
        outputs: vec![],
        lock_time: 0,
        subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_OPENING_HEX.to_string(),
        gas: 0,
        payload,
        mass: 0,
        verbose_data: None,
    }
}

impl KeryxdHandler {
    /// Whether a card-occupying solo job (request inference or opening replay) is in flight.
    fn inference_busy(&self) -> bool {
        self.inference_rx.is_some() || self.replay_rx.is_some()
    }

    fn next_request_span_cost(&self) -> u64 {
        let Some(q) = self.ai_request_queue.front() else { return 0 };
        let tier = keryx_miner::models::pom_tier_index(&q.model_id, self.last_known_daa).unwrap_or(u8::MAX);
        let (n_prompt, max_tokens) = (q.prompt_len, q.max_tokens as u32);
        let work = keryx_miner::trace::OPEN_AUDIT_WORK_PASSES * keryx_miner::trace::service_work_daa(tier, n_prompt, max_tokens);
        let backlog = keryx_miner::trace::OPENING_BACKLOG_CAP_DAA;
        let phase = keryx_miner::trace::TRACE_SEED_DELAY_DAA
            + keryx_miner::trace::attn_phase_window_daa(tier, max_tokens, n_prompt, n_prompt.saturating_add(max_tokens))
            + backlog;
        work + backlog + AUDIT_PHASES_RESERVED * phase
    }

    fn next_request_work(&self) -> u64 {
        let Some(q) = self.ai_request_queue.front() else { return 0 };
        let tier = keryx_miner::models::pom_tier_index(&q.model_id, self.last_known_daa).unwrap_or(u8::MAX);
        match speeds::generation_daa(&q.model_id, q.max_tokens as u32) {
            Some(gen) => gen + keryx_miner::trace::work_daa(tier, q.prompt_len, 0, 1),
            None => keryx_miner::trace::work_daa(tier, q.prompt_len, q.max_tokens as u32, 1),
        }
    }

    /// Whether an audit replay must start before a request of `next_work` DAA.
    fn urgent_replay(&self, next_work: u64) -> bool {
        let now = self.last_known_daa;
        let mut jobs: Vec<(u64, u64, bool)> = self
            .pending_traces
            .values()
            .filter_map(|p| {
                let r = &p.trace.record;
                if p.draw.is_none() && p.accepted_hash.is_some() && keryx_miner::trace::audit_all_active(p.submitted_daa) {
                    let n_gen = r.n_pos.saturating_sub(r.n_prompt);
                    let work = speeds::replay_daa(&p.model_id, false, p.gen_secs, n_gen)
                        .unwrap_or_else(|| keryx_miner::trace::work_daa(p.tier, r.n_prompt, n_gen, 3));
                    let due = p.submitted_daa + keryx_miner::trace::TRACE_SEED_DELAY_DAA + keryx_miner::trace::opening_window_daa(p.tier, p.max_tokens);
                    return Some((due, work, false));
                }
                let d = p.draw.as_ref().filter(|d| d.audited)?;
                let n_gen = r.n_pos.saturating_sub(r.n_prompt);
                let work = |phase: bool| {
                    speeds::replay_daa(&p.model_id, phase, p.gen_secs, n_gen)
                        .unwrap_or_else(|| keryx_miner::trace::work_daa(p.tier, r.n_prompt, n_gen, if phase { 2 } else { 3 }))
                };
                if d.replay_ready() {
                    return Some((d.deadline_daa, work(false), true));
                }
                p.attn.as_ref().filter(|a| a.ready()).map(|a| (a.deadline_daa, work(true), true))
            })
            .filter(|(deadline, _, _)| *deadline > now)
            .collect();
        jobs.sort_unstable();
        let inflight: HashSet<&str> = self.opening_inflight.values().map(|o| o.response_txid.as_str()).collect();
        let chain_end = self
            .pending_traces
            .iter()
            .filter_map(|(t, p)| {
                let d = p.draw.as_ref().filter(|d| d.audited)?;
                let unfinished = !d.opened
                    || inflight.contains(t.as_str())
                    || p.attn.as_ref().is_some_and(|a| a.next_phase_owed() || a.awaiting_draw().is_some());
                unfinished.then_some(d.deadline_daa)
            })
            .max();
        if jobs.iter().any(|(_, _, ready)| *ready)
            && chain_end.is_some_and(|end| {
                end.saturating_add(self.next_request_span_cost()) > now.saturating_add(next_work).saturating_add(AUDIT_SPAN_DAA)
            })
        {
            return true;
        }
        let mut at = now.saturating_add(REPLAY_URGENT_BASE_DAA).saturating_add(next_work);
        let mut startable = false;
        jobs.into_iter().any(|(deadline, work, ready)| {
            at = at.saturating_add(work);
            startable |= ready;
            at >= deadline && startable
        })
    }

    /// Spawns the replay that builds an opening (or an attention phase) on the trace's card.
    fn spawn_replay(&mut self, txid: String, seed_hash: String, path: Option<Vec<u64>>, prev: Vec<u8>) {
        let Some(txid_bytes) = hash32(&txid) else { return };
        let Some(p) = self.pending_traces.get(&txid) else { return };
        let Some(d) = p.draw.as_ref() else { return };
        let (model_id, tier, trace, request_hash, gpu, seed) = (p.model_id, p.tier, Arc::clone(&p.trace), p.request_hash, p.gpu, d.seed);
        let (tx_done, rx_done) = oneshot::channel::<Option<Vec<u8>>>();
        let phase = path.clone().map(|pp| (pp, prev));
        let log_txid = txid.clone();
        tokio::task::spawn_blocking(move || {
            let lease = match gpu {
                Some(g) => keryx_miner::slm::acquire_inference_card_on(g, REPLAY_CARD_DEADLINE_MS),
                None => keryx_miner::slm::acquire_inference_card(&model_id, REPLAY_CARD_DEADLINE_MS),
            };
            let body = match lease {
                Some(lease) => {
                    let _pause = crate::miner::begin_network_inference_pause();
                    let job = keryx_miner::slm::OpenJob { tier, trace: &trace, seed, response_txid: txid_bytes, request_hash, phase };
                    keryx_miner::slm::try_open_trace_on(lease.gpu(), &model_id, job).ok()
                }
                None => {
                    warn!("Trace: no free card for the replay of response {}", log_txid);
                    None
                }
            };
            let _ = tx_done.send(body);
        });
        self.replay_rx = Some(ReplayJob { response_txid: txid, seed_hash, path, started: std::time::Instant::now(), rx: rx_done });
    }

    /// Starts the replay of the audited response closest to its deadline, when the cards are free.
    async fn try_start_replay(&mut self) {
        let ready: Vec<(String, String, Vec<u8>)> = self
            .pending_traces
            .iter_mut()
            .filter_map(|(txid, p)| {
                let d = p.draw.as_mut().filter(|d| d.audited && !d.opened)?;
                let body = p.built.iter().find(|(h, _)| *h == d.seed_hash)?.1.clone();
                d.opened = true;
                Some((txid.clone(), d.seed_hash.clone(), body))
            })
            .collect();
        for (txid, seed_hash, body) in ready {
            info!("Trace: response {} audited at chain block {} — opening already built", txid, seed_hash);
            self.submit_opening(&txid, &seed_hash, body).await;
        }
        if self.inference_busy() || (!self.ai_request_queue.is_empty() && !self.urgent_replay(self.next_request_work())) {
            return;
        }
        let now = self.last_known_daa;
        let node_awaited = self.node_awaited_openings;
        let released = |p: &PendingTrace, deadline: u64| node_awaited && deadline <= now && p.node_awaits != Some(true);
        let first = self.pending_traces.iter().filter_map(|(txid, p)| {
            let d = p.draw.as_ref().filter(|d| d.replay_ready())?;
            (!released(p, d.deadline_daa)).then(|| (d.deadline_daa, txid.clone(), None))
        });
        let phases = self.pending_traces.iter().filter_map(|(txid, p)| {
            p.draw.as_ref().filter(|d| d.audited)?;
            let a = p.attn.as_ref().filter(|a| a.ready())?;
            (!released(p, a.deadline_daa)).then(|| (a.deadline_daa, txid.clone(), Some(a.path.clone())))
        });
        let Some((deadline, txid, path)) = first.chain(phases).min_by_key(|(deadline, _, _)| (*deadline <= now, *deadline)) else {
            return;
        };
        if deadline <= now && !self.ai_request_queue.is_empty() {
            return;
        }
        if let Some(path) = path {
            let p = self.pending_traces.get_mut(&txid).unwrap();
            let seed_hash = p.draw.as_ref().map(|d| d.seed_hash.clone()).unwrap_or_default();
            if let Some((_, _, body)) = p.built_phase.take_if(|(s, pp, _)| *s == seed_hash && *pp == path) {
                info!("Trace: response {} — attention phase {} already built", txid, path.len());
                if let Some(a) = p.attn.as_mut() {
                    a.opened = true;
                }
                self.on_attention_phase_built(&txid, &seed_hash, path, Some(body)).await;
                return;
            }
            let Some(a) = p.attn.as_mut() else { return };
            a.opened = true;
            let prev = a.last_partials();
            info!("Trace: response {} — replaying attention phase {}", txid, path.len());
            self.spawn_replay(txid, seed_hash, Some(path), prev);
            return;
        }
        if hash32(&txid).is_none() {
            self.pending_traces.remove(&txid);
            self.trace_store.remove(&txid);
            return;
        }
        let p = self.pending_traces.get_mut(&txid).unwrap();
        let d = p.draw.as_mut().unwrap();
        d.opened = true;
        let seed_hash = d.seed_hash.clone();
        info!("Trace: response {} audited at chain block {} — replaying its opening", txid, seed_hash);
        self.spawn_replay(txid, seed_hash, None, Vec::new());
    }

    /// Polls the in-flight replay; a finished body is signed and submitted as the opening.
    async fn poll_replay(&mut self) -> bool {
        let Some(mut job) = self.replay_rx.take() else { return false };
        let body = match job.rx.try_recv() {
            Ok(body) => body,
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                self.replay_rx = Some(job);
                return false;
            }
            Err(_) => None,
        };
        if body.is_some() {
            if let Some(p) = self.pending_traces.get(&job.response_txid) {
                if let Some(base) = p.gen_secs {
                    speeds::note_replay(&p.model_id, job.path.is_some(), job.started.elapsed().as_secs_f64() / base.max(0.001));
                }
            }
        }
        if let Some(path) = job.path {
            self.on_attention_phase_built(&job.response_txid, &job.seed_hash, path, body).await;
            return true;
        }
        let Some(body) = body else {
            let retry = self
                .pending_traces
                .get_mut(&job.response_txid)
                .and_then(|p| p.draw.as_mut())
                .filter(|d| d.seed_hash == job.seed_hash)
                .map(|d| {
                    d.replay_failures += 1;
                    d.opened = false;
                    d.retry_at = Some(replay_backoff(d.replay_failures));
                });
            if retry.is_some() {
                warn!("Trace: no opening built for response {} — retrying later", job.response_txid);
            } else {
                warn!("Trace: no opening built for response {}", job.response_txid);
            }
            return true;
        };
        let Some(p) = self.pending_traces.get_mut(&job.response_txid) else { return true };
        p.built.retain(|(h, _)| *h != job.seed_hash);
        if p.built.len() >= BUILT_OPENINGS_KEPT {
            p.built.remove(0);
        }
        p.built.push((job.seed_hash.clone(), body.clone()));
        match p.draw.as_mut().filter(|d| d.seed_hash == job.seed_hash) {
            Some(d) => d.opened = true,
            None => {
                info!("Trace: seed block of response {} left the chain during the replay — opening kept for its return", job.response_txid);
                return true;
            }
        }
        self.submit_opening(&job.response_txid, &job.seed_hash, body).await;
        true
    }

    /// Signs and submits the opening `body` of response `response_txid`: one transaction per piece.
    async fn submit_opening(&mut self, response_txid: &str, seed_hash: &str, body: Vec<u8>) {
        let Some(d) = self.pending_traces.get(response_txid).and_then(|p| p.draw.as_ref()) else { return };
        if d.seed_hash != seed_hash {
            info!("Trace: opening of response {} built for a seed the node no longer awaits — dropped", response_txid);
            return;
        }
        let (deadline_daa, seed_daa) = (d.deadline_daa, d.seed_daa);
        let Some(pieces) = keryx_miner::trace_seq::unbundle(&body) else {
            warn!("Trace: malformed opening bundle for response {}", response_txid);
            return;
        };
        let mut txs = Vec::with_capacity(pieces.len());
        let mut phased = None;
        for piece in pieces {
            let Some(tx) = self.opening_tx(response_txid, seed_hash, seed_daa, &piece) else { return };
            if keryx_verifiable_inference::opening::AttnOpening::decode(&piece).is_some_and(|o| !o.partials.is_empty()) {
                phased = Some((piece.clone(), txs.len()));
            }
            txs.push((piece.len(), tx));
        }
        if let Some((body, i)) = phased {
            let tx = &txs[i].1;
            if let (Some(txid), Some(p)) = (Self::compute_rpc_txid(tx), self.pending_traces.get_mut(response_txid)) {
                p.attn = Some(AttnPhases::new(body, txid, tx.payload.clone(), deadline_daa));
            }
        }
        for (len, rpc_tx) in &txs {
            if let Some(txid) = Self::compute_rpc_txid(rpc_tx) {
                info!("Trace: submitting the opening of response {} ({} B)", response_txid, len);
                self.opening_inflight.insert(
                    txid,
                    OpeningInflight {
                        tx: rpc_tx.clone(),
                        response_txid: response_txid.to_string(),
                        deadline_daa,
                        attempts: 1,
                        last: std::time::Instant::now(),
                    },
                );
            }
        }
        if let Some(p) = self.pending_traces.get_mut(response_txid) {
            p.opening = Some(keryx_miner::trace_store::StoredOpening {
                seed_hash: seed_hash.to_string(),
                deadline_daa,
                payload: txs[0].1.payload.clone(),
                extra: txs[1..].iter().map(|(_, t)| t.payload.clone()).collect(),
            });
        }
        self.persist_trace(response_txid);
        for (i, (_, rpc_tx)) in txs.iter().enumerate() {
            if let Err(e) = self.client_send(KaspadMessage::submit_transaction(rpc_tx.clone())).await {
                warn!("Trace: failed to send the opening tx: {}", e);
                for (_, unsent) in &txs[i..] {
                    if let Some(o) = Self::compute_rpc_txid(unsent).and_then(|t| self.opening_inflight.get_mut(&t)) {
                        o.attempts = 0;
                    }
                }
                break;
            }
        }
    }

    /// The signed opening transaction of `piece`, naming its seed block once that rule is active.
    fn opening_tx(&self, response_txid: &str, seed_hash: &str, seed_daa: u64, piece: &[u8]) -> Option<crate::proto::RpcTransaction> {
        let piece = if self.last_known_daa.max(seed_daa) >= keryx_miner::pom::trace_seed_bind_activation_daa() {
            let Some(b) = hash32(seed_hash).and_then(|h| keryx_verifiable_inference::opening::seeded(piece, &h)) else {
                warn!("Trace: cannot name seed block {} in the opening of response {}", seed_hash, response_txid);
                return None;
            };
            b
        } else {
            piece.to_vec()
        };
        let payload = self.escrow_watcher.as_ref().map(|w| w.sign_opening(&piece))?;
        Some(opening_rpc_tx(hex::encode(payload)))
    }

    /// Submits the attention phase of `path` once built, if its path is still the drawn one.
    async fn on_attention_phase_built(&mut self, response_txid: &str, seed_hash: &str, path: Vec<u64>, body: Option<Vec<u8>>) {
        let Some(p) = self.pending_traces.get_mut(response_txid) else { return };
        let current = p
            .draw
            .as_ref()
            .filter(|d| d.seed_hash == seed_hash)
            .map(|d| d.seed_daa)
            .zip(p.attn.as_mut().filter(|a| a.path == path && a.bodies.len() == path.len()));
        let Some((seed_daa, a)) = current else {
            if let Some(body) = body {
                p.built_phase = Some((seed_hash.to_string(), path, body));
            }
            return;
        };
        let Some(body) = body else {
            a.failures += 1;
            a.opened = false;
            a.retry_at = Some(replay_backoff(a.failures));
            warn!("Trace: attention phase {} of response {} not built — retrying later", path.len(), response_txid);
            return;
        };
        let deadline_daa = a.deadline_daa;
        let Some(tx) = self.opening_tx(response_txid, seed_hash, seed_daa, &body) else { return };
        let Some(txid) = Self::compute_rpc_txid(&tx) else { return };
        info!("Trace: submitting attention phase {} of response {} ({} B)", path.len(), response_txid, body.len());
        let a = self.pending_traces.get_mut(response_txid).and_then(|p| p.attn.as_mut()).unwrap();
        a.bodies.push(body);
        a.txids.push(txid.clone());
        a.payloads.push(tx.payload.clone());
        a.accepted.push(a.late.remove(&txid));
        if let Some(Some(block)) = a.accepted.last() {
            info!("Trace: attention phase {} of response {} was already accepted by chain block {}", path.len(), response_txid, block);
        }
        self.opening_inflight.insert(
            txid,
            OpeningInflight { tx: tx.clone(), response_txid: response_txid.to_string(), deadline_daa, attempts: 1, last: std::time::Instant::now() },
        );
        self.persist_trace(response_txid);
        if let Err(e) = self.client_send(KaspadMessage::submit_transaction(tx)).await {
            warn!("Trace: failed to send the attention phase tx: {}", e);
        }
    }

    /// Resubmits in-flight openings the mempool has not accepted yet.
    async fn retry_pending_openings(&mut self) -> Result<(), Error> {
        if self.opening_inflight.is_empty() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let daa = self.last_known_daa;
        let mut resend: Vec<crate::proto::RpcTransaction> = Vec::new();
        self.opening_inflight.retain(|txid, o| {
            if daa > o.deadline_daa + keryx_miner::trace::OPENING_BACKLOG_CAP_DAA {
                warn!("Trace: opening {} of response {} missed its window", txid, o.response_txid);
                return false;
            }
            let wait = if o.attempts < OPENING_FAST_RETRIES { AI_RESPONSE_RETRY_SECS } else { OPENING_SLOW_RETRY_SECS };
            if now.duration_since(o.last).as_secs() < wait {
                return true;
            }
            o.attempts += 1;
            o.last = now;
            resend.push(o.tx.clone());
            true
        });
        for tx in resend {
            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
        }
        Ok(())
    }

    fn persist_trace(&self, txid: &str) {
        if let Some(record) = self.trace_record(txid) {
            self.trace_store.save_queued(record);
        }
    }

    fn trace_record(&self, txid: &str) -> Option<keryx_miner::trace_store::StoredTraceOwned> {
        let p = self.pending_traces.get(txid)?;
        Some(keryx_miner::trace_store::StoredTraceOwned {
            response_txid: txid.to_string(),
            model_id: p.model_id,
            tier: p.tier,
            request_hash: p.request_hash,
            max_tokens: p.max_tokens,
            submitted_daa: p.submitted_daa,
            accepted_hash: p.accepted_hash.clone(),
            opening: p.opening.clone(),
            trace: Arc::clone(&p.trace),
            attn: p.attn.as_ref().map(AttnPhases::stored),
            anchor_hash: p.anchor_hash.clone(),
            gen_secs: p.gen_secs,
        })
    }

    /// Loads the stored traces of a previous run and the openings they had already submitted.
    fn restore_traces(store: &keryx_miner::trace_store::TraceStore) -> (HashMap<String, PendingTrace>, HashMap<String, OpeningInflight>) {
        let mut traces = HashMap::new();
        let mut openings = HashMap::new();
        for t in store.load_all() {
            let mut resend: Vec<(&String, u64)> =
                t.opening.iter().flat_map(|o| std::iter::once(&o.payload).chain(o.extra.iter()).map(|p| (p, o.deadline_daa))).collect();
            if let Some(a) = t.attn.as_ref() {
                resend.push((&a.payload, a.deadline_daa));
            }
            let mut attn_txid = None;
            for (payload, deadline_daa) in resend {
                let rpc_tx = opening_rpc_tx(payload.clone());
                if let Some(txid) = Self::compute_rpc_txid(&rpc_tx) {
                    if t.attn.as_ref().is_some_and(|a| &a.payload == payload) {
                        attn_txid = Some(txid.clone());
                    }
                    openings.insert(
                        txid,
                        OpeningInflight {
                            tx: rpc_tx,
                            response_txid: t.response_txid.clone(),
                            deadline_daa,
                            attempts: 1,
                            last: std::time::Instant::now() - std::time::Duration::from_secs(60),
                        },
                    );
                }
            }
            let attn = t.attn.zip(attn_txid).and_then(|(a, txid)| AttnPhases::restored(a, txid));
            traces.insert(
                t.response_txid.clone(),
                PendingTrace {
                    model_id: t.model_id,
                    tier: t.tier,
                    request_hash: t.request_hash,
                    max_tokens: t.max_tokens,
                    trace: Arc::new(t.trace),
                    submitted_daa: t.submitted_daa,
                    accepted_hash: t.accepted_hash,
                    anchor_hash: t.anchor_hash,
                    gen_secs: t.gen_secs,
                    draw: None,
                    opening: t.opening,
                    built: Vec::new(),
                    built_phase: None,
                    reorgs: 0,
                    node_awaits: None,
                    attn,
                    response_tx: None,
                    gpu: None,
                },
            );
        }
        if !traces.is_empty() {
            info!("Trace: restored {} pending response(s) from {} ({} opening(s) to resend)", traces.len(), store.dir().display(), openings.len());
        }
        (traces, openings)
    }

    async fn start_chain_restore(&mut self) -> Result<(), Error> {
        let accepted: Vec<String> = self
            .pending_traces
            .values()
            .filter_map(|p| match (&p.draw, &p.attn) {
                (None, _) => p.accepted_hash.clone().or_else(|| p.anchor_hash.clone()),
                (Some(_), Some(a)) => {
                    a.accepted.iter().rev().flatten().find(|h| !h.is_empty()).cloned().or_else(|| p.accepted_hash.clone())
                }
                (Some(_), None) => None,
            })
            .collect();
        let idle = self.chain_restore_starts.is_empty();
        for hash in accepted {
            if self.chain_restore_starts.contains(&hash) {
                continue;
            }
            if !self.chain_log.iter().any(|e| e.hash == hash) {
                self.chain_log.push_back(ChainEntry { hash: hash.clone(), daa: None });
                self.chain_restore_queue.push_back(hash.clone());
            }
            self.chain_restore_starts.push_back(hash);
        }
        if idle {
            self.request_next_chain_restore().await?;
        }
        self.chain_restore_pump().await
    }

    async fn request_next_chain_restore(&mut self) -> Result<(), Error> {
        let Some(hash) = self.chain_restore_starts.front().cloned() else { return Ok(()) };
        info!("Trace: rebuilding the chain from block {} for a restored response", hash);
        self.client_send(crate::proto::GetVirtualSelectedParentChainFromBlockRequestMessage {
            start_hash: hash,
            include_accepted_transaction_ids: true,
        })
        .await?;
        Ok(())
    }

    fn place_restored_chain(&mut self, start: &str, removed: &[String], added: Vec<String>) {
        if !removed.is_empty() {
            self.on_chain_blocks_removed(removed);
        }
        let unknown = place_chain_after(&mut self.chain_log, start, added);
        self.chain_restore_queue.extend(unknown);
    }

    fn on_chain_restore_failed(&mut self, start: String, error: &str) {
        let failures = self.chain_restore_failures.entry(start.clone()).or_insert(0);
        *failures += 1;
        let delay = CHAIN_RESTORE_RETRY_BASE.saturating_mul(1 << (*failures - 1).min(4)).min(CHAIN_RESTORE_RETRY_MAX);
        warn!("Trace: chain rebuild from block {} failed: {} — retrying in {} s", start, error, delay.as_secs());
        self.chain_restore_retry.push((std::time::Instant::now() + delay, start));
    }

    /// Sends the traced responses whose trace is now on disk.
    async fn release_persisted_responses(&mut self) -> Result<(), Error> {
        let mut ready = Vec::new();
        self.persisting.retain(|(written, txid, tx)| match written.try_recv() {
            Err(std::sync::mpsc::TryRecvError::Empty) => true,
            _ => {
                ready.push((txid.clone(), tx.clone()));
                false
            }
        });
        for (txid, tx) in ready {
            info!("Trace: trace of response {} is on disk — submitting the AiResponse", txid);
            self.ai_response_inflight.insert(txid, (tx.clone(), 1, std::time::Instant::now()));
            if let Err(e) = self.client_send(KaspadMessage::submit_transaction(tx)).await {
                warn!("OPoI: failed to send AiResponse tx: {}", e);
            }
        }
        Ok(())
    }

    async fn resend_early_responses(&mut self) -> Result<(), Error> {
        let now = std::time::Instant::now();
        let traces = &self.pending_traces;
        self.response_resend
            .retain(|(_, txid, sends)| *sends < EARLY_RESPONSE_RESENDS && traces.get(txid).is_some_and(|p| p.response_tx.is_some()));
        let mut due = Vec::new();
        for (at, txid, sends) in self.response_resend.iter_mut().filter(|(at, _, _)| *at <= now) {
            *at = now + std::time::Duration::from_secs(OPENING_SLOW_RETRY_SECS);
            *sends += 1;
            if let Some(tx) = self.pending_traces.get(txid).and_then(|p| p.response_tx.clone()) {
                due.push(tx);
            }
        }
        for tx in due {
            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
        }
        Ok(())
    }

    async fn retry_chain_blocks(&mut self) -> Result<(), Error> {
        let now = std::time::Instant::now();
        let log = &self.chain_log;
        self.chain_block_retry.retain(|(_, h, _)| log.iter().any(|c| c.hash == *h && c.daa.is_none()));
        let due: Vec<String> = self.chain_block_retry.iter().filter(|(at, _, _)| *at <= now).map(|(_, h, _)| h.clone()).collect();
        for (at, h, _) in self.chain_block_retry.iter_mut() {
            if due.contains(h) {
                *at = now + CHAIN_RESTORE_RETRY_MAX;
            }
        }
        for hash in due {
            self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
        }
        Ok(())
    }

    async fn retry_chain_restores(&mut self) -> Result<(), Error> {
        let now = std::time::Instant::now();
        let (due, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.chain_restore_retry).into_iter().partition(|(at, _)| *at <= now);
        self.chain_restore_retry = later;
        let idle = self.chain_restore_starts.is_empty();
        let restoring = self.restore_needed();
        for (_, start) in due {
            let needed = (restoring && self.chain_restore_continuations.contains(&start))
                || self.pending_traces.values().any(|p| {
                    p.accepted_hash.as_ref().or(p.anchor_hash.as_ref()) == Some(&start)
                        || p.attn.as_ref().is_some_and(|a| a.accepted.iter().flatten().any(|h| *h == start))
                });
            if needed {
                self.chain_restore_starts.push_back(start);
            } else {
                self.chain_restore_failures.remove(&start);
                self.chain_restore_continuations.remove(&start);
            }
        }
        if idle && !self.chain_restore_starts.is_empty() {
            self.request_next_chain_restore().await?;
        }
        Ok(())
    }

    fn note_phase_acceptances(&mut self, accepted: &[crate::proto::RpcAcceptedTransactionIds]) -> Vec<String> {
        let mut noted = Vec::new();
        for a in accepted {
            for txid in &a.accepted_transaction_ids {
                for (response, p) in self.pending_traces.iter_mut() {
                    let Some(ph) = p.attn.as_mut() else { continue };
                    if ph.txids.last() == Some(txid) && ph.accepted.last().is_some_and(Option::is_none) {
                        *ph.accepted.last_mut().unwrap() = Some(a.accepting_block_hash.clone());
                        info!(
                            "Trace: attention phase {} of response {} accepted by chain block {}",
                            ph.bodies.len() - 1,
                            response,
                            a.accepting_block_hash
                        );
                        noted.push(response.clone());
                    } else if ph.dropped.contains(txid) {
                        ph.late.insert(txid.clone(), a.accepting_block_hash.clone());
                    }
                }
            }
        }
        noted
    }

    fn note_opening_acceptances(&mut self, accepted: &[crate::proto::RpcAcceptedTransactionIds]) {
        for a in accepted {
            for txid in &a.accepted_transaction_ids {
                if let Some(o) = self.opening_inflight.remove(txid) {
                    info!("Trace: opening {} of response {} accepted by chain block {}", txid, o.response_txid, a.accepting_block_hash);
                }
            }
        }
    }

    fn note_restored_acceptances(&mut self, accepted: &[crate::proto::RpcAcceptedTransactionIds]) {
        self.note_opening_acceptances(accepted);
        let mut noted: Vec<String> = self.note_phase_acceptances(accepted);
        for a in accepted {
            for txid in &a.accepted_transaction_ids {
                if let Some(p) = self.pending_traces.get_mut(txid).filter(|p| p.accepted_hash.is_none()) {
                    p.accepted_hash = Some(a.accepting_block_hash.clone());
                    info!("Trace: response {} accepted by chain block {}", txid, a.accepting_block_hash);
                    noted.push(txid.clone());
                }
            }
        }
        for txid in noted {
            self.persist_trace(&txid);
        }
    }

    fn restore_needed(&self) -> bool {
        self.pending_traces
            .values()
            .any(|p| (p.accepted_hash.is_some() && p.draw.is_none()) || p.attn.as_ref().is_some_and(|a| a.awaiting_draw().is_some()))
    }

    async fn chain_restore_pump(&mut self) -> Result<(), Error> {
        if !self.restore_needed() {
            self.chain_restore_queue.clear();
            return Ok(());
        }
        while self.chain_restore_pending.len() < CHAIN_RESTORE_INFLIGHT {
            let Some(hash) = self.chain_restore_queue.pop_front() else { break };
            if !self.chain_restore_pending.insert(hash.clone()) {
                continue;
            }
            self.client_send(GetBlockRequestMessage { hash, include_transactions: false }).await?;
        }
        Ok(())
    }

    fn note_chain_block(&mut self, hash: &str, daa: u64) {
        for e in self.chain_log.iter_mut().filter(|e| e.hash == hash) {
            e.daa = Some(daa);
        }
    }

    fn on_chain_blocks_removed(&mut self, removed: &[String]) {
        if removed.is_empty() {
            return;
        }
        let removed: HashSet<&str> = removed.iter().map(String::as_str).collect();
        self.chain_log.retain(|e| !removed.contains(e.hash.as_str()));
        let mut reset: Vec<String> = Vec::new();
        let mut rephased: Vec<String> = Vec::new();
        for (txid, p) in self.pending_traces.iter_mut() {
            if let Some(a) = p.attn.as_mut() {
                a.late.retain(|_, b| !removed.contains(b.as_str()));
            }
            if p.accepted_hash.as_deref().is_some_and(|h| removed.contains(h)) {
                p.accepted_hash = None;
                p.draw = None;
                reset.push(txid.clone());
            } else if p.draw.as_ref().is_some_and(|d| removed.contains(d.seed_hash.as_str())) {
                p.draw = None;
                reset.push(txid.clone());
            } else if let Some(dropped) = p.attn.as_mut().and_then(|a| a.on_removed(&removed)) {
                if let (Some(d), Some(a)) = (p.draw.as_ref(), p.attn.as_mut()) {
                    a.deadline_daa = a.deadline_daa.min(d.deadline_daa);
                }
                log::debug!(
                    "Trace: chain reorg — attention phases of response {} back to phase {}",
                    txid,
                    p.attn.as_ref().map_or(0, |a| a.bodies.len() - 1)
                );
                self.opening_inflight.retain(|t, _| !dropped.contains(t));
                rephased.push(txid.clone());
            }
        }
        for txid in rephased {
            self.persist_trace(&txid);
        }
        for txid in reset {
            log::debug!("Trace: chain reorg — the draw of response {} is reset", txid);
            self.opening_inflight.retain(|_, o| o.response_txid != txid);
            if let Some(p) = self.pending_traces.get_mut(&txid) {
                p.opening = None;
                p.attn = None;
                p.reorgs += 1;
            }
            self.persist_trace(&txid);
        }
    }

    fn rebuild_pending_after(&self, accepted: &str) -> bool {
        let Some(from) = self.chain_log.iter().position(|e| e.hash == accepted) else { return false };
        let pending = |h: &String| {
            self.chain_restore_failures.contains_key(h)
                || self.chain_restore_continuations.contains(h)
                || self.chain_restore_retry.iter().any(|(_, s)| s == h)
        };
        self.chain_log.iter().skip(from).any(|e| pending(&e.hash))
    }

    /// Draws every accepted trace whose seed block is known.
    fn resolve_traces(&mut self) {
        let mut drawn: Vec<(String, TraceDraw)> = Vec::new();
        for (txid, p) in self.pending_traces.iter() {
            if p.draw.is_some() {
                continue;
            }
            let Some(accepted) = p.accepted_hash.as_deref() else { continue };
            if self.chain_restore_starts.iter().any(|h| h == accepted) || self.rebuild_pending_after(accepted) {
                continue;
            }
            let stored = p
                .opening
                .as_ref()
                .and_then(|o| self.chain_log.iter().find(|e| e.hash == o.seed_hash))
                .and_then(|e| Some((e.hash.as_str(), e.daa?)));
            let accepted_at = self.chain_log.iter().find(|e| e.hash == accepted).and_then(|e| e.daa);
            let Some((seed_hash, daa, accepted_daa)) = (match (stored, accepted_at) {
                (Some((h, d)), Some(a)) => Some((h, d, a)),
                _ => seed_block_after(&self.chain_log, accepted),
            }) else {
                continue;
            };
            let (Some(txid_bytes), Some(seed_block)) = (hash32(txid), hash32(seed_hash)) else { continue };
            let seed = keryx_miner::trace::trace_seed(&seed_block, &txid_bytes);
            let audited = keryx_miner::trace::audit_all_active(accepted_daa) || keryx_miner::trace::audited(&seed);
            let attention =
                keryx_miner::trace::attention_active(accepted_daa) && keryx_miner::trace::TierManifest::for_attn(p.tier).is_some();
            let prompt = if attention { keryx_miner::trace::attn_first_prompt_daa(p.trace.record.n_prompt) } else { 0 };
            let deadline_daa = daa + keryx_miner::trace::opening_window_daa(p.tier, p.max_tokens) + prompt;
            drawn.push((
                txid.clone(),
                TraceDraw {
                    seed_hash: seed_hash.to_string(),
                    seed_daa: daa,
                    accepted_daa,
                    seed,
                    audited,
                    deadline_daa,
                    opened: p.opening.as_ref().is_some_and(|o| o.seed_hash == seed_hash),
                    replay_failures: 0,
                    retry_at: None,
                },
            ));
        }
        drawn.sort_by(|(ta, a), (tb, b)| (a.seed_daa, ta).cmp(&(b.seed_daa, tb)));
        let inflight: HashSet<String> = self.opening_inflight.values().map(|o| o.response_txid.clone()).collect();
        let mut open: Vec<u64> = self
            .pending_traces
            .iter()
            .filter_map(|(t, p)| {
                let d = p.draw.as_ref().filter(|d| d.audited)?;
                let unfinished = !d.opened
                    || inflight.contains(t)
                    || p.attn.as_ref().is_some_and(|a| a.next_phase_owed() || a.awaiting_draw().is_some());
                unfinished.then(|| d.deadline_daa.max(p.attn.as_ref().map_or(0, |a| a.deadline_daa)))
            })
            .collect();
        for (txid, d) in drawn.iter_mut().filter(|(_, d)| d.audited) {
            let Some(p) = self.pending_traces.get(txid) else { continue };
            let r = &p.trace.record;
            let work = keryx_miner::trace::service_work_daa(p.tier, r.n_prompt, r.n_pos.saturating_sub(r.n_prompt));
            d.deadline_daa = keryx_miner::trace::chained_deadline(d.deadline_daa, &open, d.seed_daa, work);
            open.push(d.deadline_daa);
        }
        let mut children: Vec<String> = Vec::new();
        let held: HashSet<String> = self
            .pending_traces
            .values()
            .filter_map(|p| p.attn.as_ref()?.awaiting_draw().map(|h| h.to_string()))
            .filter(|h| self.rebuild_pending_after(h))
            .collect();
        for (txid, p) in self.pending_traces.iter_mut() {
            let (Some(d), Some(a)) = (p.draw.as_ref(), p.attn.as_mut()) else { continue };
            let Some(accepted) = a.awaiting_draw() else { continue };
            if held.contains(accepted) {
                continue;
            }
            let Some((block, _, accepted_daa)) = seed_block_after(&self.chain_log, accepted) else { continue };
            let (Some(txid_bytes), Some(block_bytes)) = (hash32(txid), hash32(block)) else { continue };
            let phase = a.path.len();
            let Some(n) = keryx_miner::trace_seq::attn_children(p.tier, &p.trace.record, &d.seed, &a.path) else { continue };
            let Some(weights) = a
                .bodies
                .last()
                .and_then(|b| keryx_miner::trace_seq::attn_child_weights(p.tier, &p.trace.record, &d.seed, b))
                .filter(|w| w.len() as u64 == n)
            else {
                warn!("Trace: response {} — no draw weights for attention phase {}", txid, phase + 1);
                continue;
            };
            let child = keryx_verifiable_inference::attn_blocks::draw_weighted_child(
                &keryx_miner::trace::attn_child_seed(&block_bytes, &txid_bytes, phase as u8),
                &weights,
            );
            a.path.push(child);
            a.child_blocks.push(Some(block.to_string()));
            a.deadline_daa = d.deadline_daa.max(
                accepted_daa
                    + keryx_miner::trace::TRACE_SEED_DELAY_DAA
                    + keryx_miner::trace::attn_phase_window_daa(p.tier, p.max_tokens, p.trace.record.n_prompt, p.trace.record.n_pos),
            );
            a.opened = false;
            a.failures = 0;
            info!("Trace: response {} — attention phase {} drawn child {} of {}, due by DAA {}", txid, phase + 1, child, n, a.deadline_daa);
            children.push(txid.clone());
        }
        for txid in children {
            self.persist_trace(&txid);
        }
        for (txid, d) in drawn {
            if d.audited {
                let reorgs = self.pending_traces.get(&txid).map_or(0, |p| p.reorgs);
                if d.opened {
                    info!("Trace: response {} — audit restored, opening already submitted", txid);
                } else if reorgs == 0 {
                    info!("Trace: response {} drawn for audit, opening due by DAA {}", txid, d.deadline_daa);
                } else {
                    info!("Trace: response {} drawn for audit after {} chain reorgs, opening due by DAA {}", txid, reorgs, d.deadline_daa);
                }
            }
            if let Some(p) = self.pending_traces.get_mut(&txid) {
                p.draw = Some(d);
            }
        }
    }

    async fn poll_awaited_openings(&mut self) -> Result<(), Error> {
        let now = self.last_known_daa;
        let mut due: Vec<(u64, &String)> = self
            .pending_traces
            .iter()
            .filter(|(_, p)| {
                p.accepted_hash.is_some()
                    && p.draw.as_ref().is_none_or(|d| {
                        d.audited
                            && (now <= d.deadline_daa + keryx_miner::trace::OPENING_BACKLOG_CAP_DAA
                                || p.attn.as_ref().is_some_and(|a| a.next_phase_owed() || a.awaiting_draw().is_some()))
                    })
            })
            .map(|(t, p)| (p.draw.as_ref().map_or(0, |d| d.deadline_daa), t))
            .collect();
        due.sort_unstable();
        let response_txids: Vec<String> = due.into_iter().take(AWAITED_POLL_MAX).map(|(_, t)| t.clone()).collect();
        self.awaited_polled = response_txids.clone();
        if !response_txids.is_empty() {
            self.client_send(crate::proto::GetAwaitedOpeningsRequestMessage { response_txids }).await?;
        }
        Ok(())
    }

    /// Takes the node's seed and deadline for each opening it awaits.
    fn apply_awaited_openings(&mut self, resp: &crate::proto::GetAwaitedOpeningsResponseMessage) {
        let mut changed: Vec<String> = Vec::new();
        let mut resend: Vec<(String, keryx_miner::trace_store::StoredOpening, u64)> = Vec::new();
        let listed: HashSet<&str> = resp.openings.iter().map(|o| o.response_txid.as_str()).collect();
        for txid in &self.awaited_polled {
            if let Some(p) = self.pending_traces.get_mut(txid) {
                p.node_awaits = Some(listed.contains(txid.as_str()));
            }
        }
        for o in resp.openings.iter() {
            let idle = !self.opening_inflight.values().any(|f| f.response_txid == o.response_txid)
                && self.replay_rx.as_ref().is_none_or(|j| j.response_txid != o.response_txid);
            let (Some(txid_bytes), Some(seed)) = (hash32(&o.response_txid), hash32(&o.seed)) else { continue };
            let Some(p) = self.pending_traces.get(&o.response_txid) else { continue };
            let had_draw = p.draw.is_some();
            let same_seed = p.draw.as_ref().is_some_and(|d| d.seed == seed);
            let seeded = if same_seed {
                None
            } else {
                let from = p.accepted_hash.as_deref().and_then(|h| self.chain_log.iter().position(|e| e.hash == h)).unwrap_or(0);
                let Some(e) = self
                    .chain_log
                    .iter()
                    .skip(from)
                    .find(|e| hash32(&e.hash).is_some_and(|b| keryx_miner::trace::trace_seed(&b, &txid_bytes) == seed))
                else {
                    log::debug!("Trace: response {} — the node's seed block is not in the chain log yet", o.response_txid);
                    continue;
                };
                Some((e.hash.clone(), e.daa.unwrap_or(0)))
            };
            let p = self.pending_traces.get_mut(&o.response_txid).unwrap();
            if let Some((seed_hash, seed_daa)) = seeded {
                if had_draw {
                    info!(
                        "Trace: response {} — seed differs from the node, audit seeded by block {}, opening due by DAA {}",
                        o.response_txid, seed_hash, o.deadline_daa
                    );
                } else {
                    info!("Trace: response {} — audit seeded by the node at block {}, opening due by DAA {}", o.response_txid, seed_hash, o.deadline_daa);
                }
                let kept = p.opening.as_ref().is_some_and(|op| op.seed_hash == seed_hash);
                p.draw = Some(TraceDraw {
                    seed_hash,
                    seed_daa,
                    accepted_daa: o.accepted_daa,
                    seed,
                    audited: true,
                    deadline_daa: o.deadline_daa,
                    opened: kept,
                    replay_failures: 0,
                    retry_at: None,
                });
                if !kept {
                    p.attn = None;
                    p.opening = None;
                    self.opening_inflight.retain(|_, f| f.response_txid != o.response_txid);
                }
                changed.push(o.response_txid.clone());
                continue;
            }
            let d = p.draw.as_mut().unwrap();
            let in_phases = p.attn.as_ref().is_some_and(|a| !a.path.is_empty());
            if d.accepted_daa != o.accepted_daa {
                d.accepted_daa = o.accepted_daa;
                changed.push(o.response_txid.clone());
            }
            if !d.audited || (!in_phases && d.deadline_daa != o.deadline_daa) {
                d.audited = true;
                d.deadline_daa = o.deadline_daa;
                changed.push(o.response_txid.clone());
            }
            if d.opened && idle && !in_phases && p.attn.is_none() {
                match p.opening.as_ref().filter(|op| op.seed_hash == d.seed_hash) {
                    Some(op) => resend.push((o.response_txid.clone(), op.clone(), o.deadline_daa)),
                    None => {
                        info!("Trace: the node still awaits the opening of response {} — replaying it", o.response_txid);
                        d.opened = false;
                        changed.push(o.response_txid.clone());
                    }
                }
            }
            if let Some(a) = p.attn.as_mut().filter(|_| in_phases) {
                if !o.attention_path.is_empty() && a.path != o.attention_path {
                    warn!("Trace: response {} — attention path {:?} differs from the node's {:?}", o.response_txid, a.path, o.attention_path);
                }
                if a.deadline_daa != o.deadline_daa {
                    a.deadline_daa = o.deadline_daa;
                    changed.push(o.response_txid.clone());
                }
            }
        }
        for (response_txid, op, deadline_daa) in resend {
            info!("Trace: the node still awaits the opening of response {} — sending it again", response_txid);
            for payload in std::iter::once(&op.payload).chain(op.extra.iter()) {
                let tx = opening_rpc_tx(payload.clone());
                if let Some(txid) = Self::compute_rpc_txid(&tx) {
                    let inflight = OpeningInflight {
                        tx,
                        response_txid: response_txid.clone(),
                        deadline_daa,
                        attempts: 1,
                        last: std::time::Instant::now() - std::time::Duration::from_secs(60),
                    };
                    self.opening_inflight.insert(txid, inflight);
                }
            }
        }
        changed.dedup();
        for o in resp.openings.iter() {
            for f in self.opening_inflight.values_mut().filter(|f| f.response_txid == o.response_txid) {
                f.deadline_daa = o.deadline_daa;
            }
        }
        for txid in changed {
            self.persist_trace(&txid);
        }
    }

    /// Drops the traces past their window and the chain log they no longer need.
    fn prune_traces(&mut self) {
        let daa = self.last_known_daa;
        let before: Vec<String> = self.pending_traces.keys().cloned().collect();
        let inflight: HashSet<String> = self.opening_inflight.values().map(|o| o.response_txid.clone()).collect();
        self.pending_traces.retain(|txid, p| match p.draw.as_ref() {
            None => p.submitted_daa + TRACE_UNDRAWN_LIMIT_DAA > daa,
            Some(d) if !d.audited => d.seed_daa + TRACE_UNAUDITED_GRACE_DAA > daa,
            Some(d) => {
                let deadline = d.deadline_daa.max(p.attn.as_ref().map_or(0, |a| a.deadline_daa));
                let released = p.node_awaits == Some(false)
                    && daa > deadline
                    && !inflight.contains(txid)
                    && p.attn.as_ref().is_none_or(|a| a.awaiting_draw().is_none());
                let unfinished = !d.opened
                    || inflight.contains(txid)
                    || p.attn.as_ref().is_some_and(|a| a.next_phase_owed() || a.awaiting_draw().is_some());
                let keep = !released
                    && (daa <= deadline
                        || (unfinished && daa <= deadline + keryx_miner::trace::OPENING_BACKLOG_CAP_DAA)
                        || p.attn.as_ref().is_some_and(|a| {
                            a.awaiting_draw().is_some() && daa <= a.deadline_daa + 2 * keryx_miner::trace::TRACE_SEED_DELAY_DAA
                        }));
                if !keep && daa > d.deadline_daa && !d.opened {
                    warn!("Trace: response {} was never opened — window closed", txid);
                }
                keep
            }
        });
        for txid in before.iter().filter(|t| !self.pending_traces.contains_key(*t)) {
            self.trace_store.remove(txid);
        }
        let needed = self
            .pending_traces
            .values()
            .filter_map(|p| p.accepted_hash.as_deref())
            .filter_map(|h| self.chain_log.iter().position(|e| e.hash == h))
            .min()
            .unwrap_or(self.chain_log.len());
        let keep_from = needed.min(self.chain_log.len().saturating_sub(CHAIN_LOG_TAIL));
        self.chain_log.drain(..keep_from);
    }

    /// Resubmits in-flight AiResponses the mempool has not accepted yet (the arming gap: a
    /// response sent right after its request is refused until the request's audit arms).
    async fn retry_pending_ai_responses(&mut self) -> Result<(), Error> {
        if self.ai_response_inflight.is_empty() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let mut resend: Vec<crate::proto::RpcTransaction> = Vec::new();
        self.ai_response_inflight.retain(|txid, (tx, attempts, last)| {
            if now.duration_since(*last).as_secs() < AI_RESPONSE_RETRY_SECS {
                return true;
            }
            if *attempts >= AI_RESPONSE_MAX_ATTEMPTS {
                warn!("OPoI: giving up on AiResponse {} after {} submit attempts", txid, attempts);
                return false;
            }
            *attempts += 1;
            *last = now;
            resend.push(tx.clone());
            true
        });
        for tx in resend {
            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
        }
        Ok(())
    }

    /// The periodic trace work of the 200 ms tick.
    async fn trace_tick(&mut self) -> Result<(), Error> {
        self.poll_replay().await;
        if self.node_awaited_openings && self.last_awaited_poll.elapsed().as_secs() >= AWAITED_POLL_SECS {
            self.last_awaited_poll = std::time::Instant::now();
            self.poll_awaited_openings().await?;
        }
        self.release_persisted_responses().await?;
        self.resend_early_responses().await?;
        self.retry_pending_ai_responses().await?;
        self.prune_traces();
        self.retry_pending_openings().await?;
        self.retry_chain_restores().await?;
        self.retry_chain_blocks().await?;
        self.try_start_replay().await;
        Ok(())
    }

    /// The virtual chain moved: log its blocks, note the acceptances of our responses, openings
    /// and attention phases.
    fn on_virtual_chain_changed(&mut self, notif: &crate::proto::VirtualSelectedParentChainChangedNotificationMessage) {
        self.on_chain_blocks_removed(&notif.removed_chain_block_hashes);
        for hash in &notif.added_chain_block_hashes {
            if !self.chain_log.iter().any(|e| &e.hash == hash) {
                self.chain_log.push_back(ChainEntry { hash: hash.clone(), daa: None });
            }
        }
        let mut accepted_now: Vec<String> = Vec::new();
        for accepted in &notif.accepted_transaction_ids {
            for txid in &accepted.accepted_transaction_ids {
                let early: Vec<String> = self
                    .pending_traces
                    .iter()
                    .filter(|(_, p)| {
                        p.accepted_hash.as_ref().is_some_and(|h| *h != accepted.accepting_block_hash) && hex::encode(p.request_hash) == *txid
                    })
                    .map(|(t, _)| t.clone())
                    .collect();
                for t in early {
                    if !self.response_resend.iter().any(|(_, r, _)| *r == t) {
                        info!("Trace: response {} accepted before its request — sent again shortly", t);
                        self.response_resend.push((std::time::Instant::now() + EARLY_RESPONSE_RESEND_DELAY, t, 0));
                    }
                }
            }
            for txid in &accepted.accepted_transaction_ids {
                if let Some(p) = self.pending_traces.get_mut(txid).filter(|p| p.accepted_hash.is_none()) {
                    p.accepted_hash = Some(accepted.accepting_block_hash.clone());
                    if p.reorgs == 0 {
                        info!("Trace: response {} accepted by chain block {}", txid, accepted.accepting_block_hash);
                    } else {
                        log::debug!("Trace: response {} accepted by chain block {}", txid, accepted.accepting_block_hash);
                    }
                    accepted_now.push(txid.clone());
                }
            }
        }
        self.note_opening_acceptances(&notif.accepted_transaction_ids);
        accepted_now.extend(self.note_phase_acceptances(&notif.accepted_transaction_ids));
        for txid in accepted_now {
            self.persist_trace(&txid);
        }
    }

    /// A submit_transaction answer for one of our openings, re-sent responses or AiResponses.
    /// Returns `true` when it was ours.
    fn on_trace_submit_response(&mut self, txid_field: &str, err: Option<&str>) -> bool {
        let opening_txid = if self.opening_inflight.contains_key(txid_field) {
            Some(txid_field.to_string())
        } else {
            err.and_then(|e| self.opening_inflight.keys().find(|k| e.contains(k.as_str())).cloned())
        };
        if let Some(txid) = opening_txid {
            match err {
                None => {
                    if let Some(o) = self.opening_inflight.get_mut(&txid) {
                        o.attempts = o.attempts.max(OPENING_FAST_RETRIES);
                    }
                    info!("Trace: opening accepted by the mempool");
                }
                Some(e) if e.contains("an opening for response") => info!("Trace: another opening of this response is in the mempool — retrying"),
                Some(e) if e.contains("already") => {
                    if let Some(o) = self.opening_inflight.get_mut(&txid) {
                        o.attempts = o.attempts.max(OPENING_FAST_RETRIES);
                    }
                    log::debug!("Trace: opening already known to the node");
                }
                Some(e) if e.contains("not awaited") => {
                    let attempts = self.opening_inflight.get(&txid).map_or(1, |o| o.attempts);
                    if attempts <= 1 {
                        warn!("Trace: opening not awaited by the node yet — retrying until its window ends: {}", e);
                    } else {
                        log::debug!("Trace: opening not awaited (attempt {}): {}", attempts, e);
                    }
                }
                Some(e) => {
                    let attempts = self.opening_inflight.get(&txid).map_or(1, |o| o.attempts);
                    if attempts <= 1 {
                        warn!("Trace: opening rejected: {} — retrying until accepted or expired", e);
                    } else {
                        log::debug!("Trace: opening rejected (attempt {}): {}", attempts, e);
                    }
                }
            }
            return true;
        }
        let resent = self.response_resend.iter().position(|(_, t, _)| *t == txid_field || err.is_some_and(|e| e.contains(t.as_str())));
        if let Some(i) = resent {
            match err {
                None => {
                    info!("OPoI: response {} sent again and admitted", self.response_resend[i].1);
                    self.response_resend.remove(i);
                }
                Some(e) if e.contains("not awaited") => {
                    log::debug!("OPoI: response {} already credited", self.response_resend[i].1);
                    self.response_resend.remove(i);
                }
                Some(e) => log::debug!("OPoI: response sent again refused: {}", e),
            }
            return true;
        }
        let inflight_txid = if self.ai_response_inflight.contains_key(txid_field) {
            Some(txid_field.to_string())
        } else {
            err.and_then(|e| self.ai_response_inflight.keys().find(|k| e.contains(k.as_str())).cloned())
        };
        match (inflight_txid, err) {
            (Some(txid), None) => {
                self.ai_response_inflight.remove(&txid);
                info!("OPoI: AiResponse {} accepted by the mempool", txid);
                true
            }
            (Some(txid), Some(e)) if (e.contains(&txid) && e.contains("already")) || e.contains("same responder") => {
                self.ai_response_inflight.remove(&txid);
                info!("OPoI: AiResponse already known to the node — done");
                true
            }
            (Some(txid), Some(e)) => {
                let attempts = self.ai_response_inflight.get(&txid).map(|(_, a, _)| *a).unwrap_or(1);
                if attempts <= 1 {
                    warn!("OPoI: AiResponse rejected: {} — retrying until accepted or expired", e);
                } else {
                    log::debug!("OPoI: AiResponse rejected (attempt {}): {}", attempts, e);
                }
                true
            }
            _ => false,
        }
    }
}

/// Generation and replay speeds measured on this rig, by model, for the life of the process.
/// Upstream v0.5.8 verbatim.
mod speeds {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    const DAA_PER_SEC: f64 = 10.0;
    const EMA: f64 = 0.3;
    const MIN_TOKENS: u32 = 32;
    const PROMPT_PER_TOKEN: u32 = 16;
    const OUTLIER: f64 = 4.0;
    const FIRST_RATIO: f64 = 3.0;
    const PHASE_RATIO: f64 = 2.0;

    #[derive(Default, Clone, Copy)]
    struct Speed {
        secs_per_token: Option<f64>,
        first_ratio: Option<f64>,
        phase_ratio: Option<f64>,
    }

    static SPEEDS: Mutex<BTreeMap<[u8; 32], Speed>> = Mutex::new(BTreeMap::new());

    fn ema(old: Option<f64>, new: f64) -> Option<f64> {
        if !new.is_finite() || new <= 0.0 {
            return old;
        }
        Some(old.map_or(new, |o| o + EMA * (new - o)))
    }

    fn get(model: &[u8; 32]) -> Speed {
        SPEEDS.lock().unwrap_or_else(|p| p.into_inner()).get(model).copied().unwrap_or_default()
    }

    pub(super) fn note_generation(model: &[u8; 32], n_prompt: u32, n_gen: u32, secs: f64) {
        if n_gen < MIN_TOKENS.max(n_prompt / PROMPT_PER_TOKEN) {
            return;
        }
        let mut m = SPEEDS.lock().unwrap_or_else(|p| p.into_inner());
        let s = m.entry(*model).or_default();
        let measured = secs / n_gen as f64;
        s.secs_per_token = ema(s.secs_per_token, s.secs_per_token.map_or(measured, |o| measured.min(OUTLIER * o)));
    }

    pub(super) fn note_replay(model: &[u8; 32], phase: bool, ratio: f64) {
        let ratio = ratio.clamp(0.05, 20.0);
        let mut m = SPEEDS.lock().unwrap_or_else(|p| p.into_inner());
        let s = m.entry(*model).or_default();
        if phase {
            s.phase_ratio = ema(s.phase_ratio, ratio);
        } else {
            s.first_ratio = ema(s.first_ratio, ratio);
        }
    }

    pub(super) fn generation_daa(model: &[u8; 32], n_gen: u32) -> Option<u64> {
        get(model).secs_per_token.map(|s| (s * n_gen as f64 * DAA_PER_SEC).round() as u64)
    }

    pub(super) fn replay_daa(model: &[u8; 32], phase: bool, gen_secs: Option<f64>, n_gen: u32) -> Option<u64> {
        let s = get(model);
        let base = gen_secs.or_else(|| s.secs_per_token.map(|t| t * n_gen.max(1) as f64))?;
        let ratio = if phase { s.phase_ratio.unwrap_or(PHASE_RATIO) } else { s.first_ratio.unwrap_or(FIRST_RATIO) };
        Some((base * ratio * DAA_PER_SEC).round() as u64)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn estimates_follow_the_measured_speeds() {
            let model = [0x7Au8; 32];
            assert_eq!(generation_daa(&model, 100), None);
            assert_eq!(replay_daa(&model, false, None, 100), None);
            assert_eq!(replay_daa(&model, false, Some(10.0), 100), Some(300));
            note_generation(&model, 0, 8, 100.0);
            assert_eq!(generation_daa(&model, 100), None, "a short generation is not a speed");
            note_generation(&model, 32_000, 1_000, 200.0);
            assert_eq!(generation_daa(&model, 100), None, "nor one the prompt weighs in");
            note_generation(&model, 100, 1_000, 20.0);
            assert_eq!(generation_daa(&model, 4_096), Some(819));
            note_replay(&model, false, 1.5);
            assert_eq!(replay_daa(&model, false, Some(10.0), 0), Some(150));
            assert_eq!(replay_daa(&model, false, None, 1_000), Some(300));
            assert_eq!(replay_daa(&model, true, Some(10.0), 0), Some(200));
            note_generation(&model, 0, 1_000, 40.0);
            assert_eq!(generation_daa(&model, 1_000), Some(260));
            note_generation(&model, 0, 1_000, 1_000.0);
            assert!(generation_daa(&model, 1_000).unwrap() < 600, "an outlier moves the speed by a bounded step");
        }
    }
}

impl Drop for KeryxdHandler {
    fn drop(&mut self) {
        keryx_miner::runtime_stats::record_inference_abandoned_requests(
            keryx_miner::runtime_stats::InferenceKind::SoloRequest,
            self.ai_request_queue.len(),
        );
        keryx_miner::runtime_stats::connection_lost(self.runtime_generation, "Solo node connection closed");
        self.block_handle.abort();
        // Every queued trace write reaches the disk: the next connection restores from it.
        self.trace_store.flush();
        if !self.persisting.is_empty() {
            warn!("Trace: {} traced response(s) were not sent before the connection closed", self.persisting.len());
        }
    }
}

/// True when a keryxd `server_version` string (e.g. "1.4.3", "v1.4.3-OPoI/...") is older than
/// `min` (major, minor, patch). Unparseable versions return FALSE (fail-open: never hold claims
/// on a version we can't read — the gate is a convenience guard, not consensus).
fn version_lt(server_version: &str, min: (u64, u64, u64)) -> bool {
    let v = server_version.trim_start_matches(|c: char| !c.is_ascii_digit());
    let mut parts = v.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty());
    let (a, b, c) = (
        parts.next().and_then(|s| s.parse::<u64>().ok()),
        parts.next().and_then(|s| s.parse::<u64>().ok()),
        parts.next().and_then(|s| s.parse::<u64>().ok()),
    );
    match (a, b, c) {
        (Some(a), Some(b), Some(c)) => (a, b, c) < min,
        _ => false,
    }
}

#[cfg(test)]
mod grpc_tests {
    use super::{
        version_lt, AiRequestKey, BoundedAiRequestQueue, BoundedAiSeen, ChallengeInference, QueuedAiRequest, SeenInsert,
    };

    fn key(byte: u8) -> AiRequestKey {
        AiRequestKey([byte; 32])
    }

    fn request(byte: u8) -> QueuedAiRequest {
        QueuedAiRequest {
            key: key(byte),
            request_hash: [byte; 32],
            model_id: [0x55; 32],
            prompt: format!("prompt-{byte}"),
            max_tokens: 64,
            private: None,
        }
    }

    #[test]
    fn version_gate_parses_real_keryxd_strings() {
        assert!(version_lt("1.4.3", (1, 4, 4)));
        assert!(version_lt("v1.4.3-OPoI", (1, 4, 4)));
        assert!(version_lt("1.3.41", (1, 4, 4)));
        assert!(!version_lt("1.4.4", (1, 4, 4)));
        assert!(!version_lt("1.4.5", (1, 4, 4)));
        assert!(!version_lt("2.0.0", (1, 4, 4)));
        assert!(!version_lt("garbage", (1, 4, 4))); // fail-open
    }

    #[test]
    fn full_request_key_is_not_a_64_bit_prefix() {
        let mut request_a = [0x11; 32];
        let mut request_b = request_a;
        request_a[31] = 0xaa;
        request_b[31] = 0xbb;
        assert_eq!(&request_a[..8], &request_b[..8]);

        let model = [0x22; 32];
        let base = AiRequestKey::new(&request_a, &model, "prompt", 64);
        assert_eq!(base, AiRequestKey::new(&request_a, &model, "prompt", 64));
        assert_ne!(base, AiRequestKey::new(&request_b, &model, "prompt", 64));
        assert_ne!(base, AiRequestKey::new(&request_a, &[0x23; 32], "prompt", 64));
        assert_ne!(base, AiRequestKey::new(&request_a, &model, "prompt!", 64));
        assert_ne!(base, AiRequestKey::new(&request_a, &model, "prompt", 65));
    }

    #[test]
    fn request_queue_evicts_oldest_deterministically() {
        let mut queue = BoundedAiRequestQueue::new(2);
        assert!(queue.push_back(request(1)).is_none());
        assert!(queue.push_back(request(2)).is_none());
        let evicted = queue.push_back(request(3)).expect("full queue must evict");
        assert_eq!(evicted.key, key(1));
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.pop_front().unwrap().key, key(2));
        assert_eq!(queue.pop_front().unwrap().key, key(3));
    }

    #[test]
    fn seen_cache_is_bounded_fifo_and_never_evicts_protected_work() {
        let mut seen = BoundedAiSeen::new(2);
        assert_eq!(seen.insert(key(1), |_| false), SeenInsert::Inserted { evicted: None });
        assert_eq!(seen.insert(key(2), |_| false), SeenInsert::Inserted { evicted: None });
        assert_eq!(seen.insert(key(1), |_| false), SeenInsert::Duplicate);

        assert_eq!(
            seen.insert(key(3), |candidate| *candidate == key(1)),
            SeenInsert::Inserted { evicted: Some(key(2)) }
        );
        assert!(seen.contains(&key(1)));
        assert!(!seen.contains(&key(2)));
        assert!(seen.contains(&key(3)));
        assert_eq!(seen.len(), 2);

        assert_eq!(seen.insert(key(4), |_| true), SeenInsert::AllEntriesProtected);
        assert!(!seen.contains(&key(4)));
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn challenge_completion_emits_one_timer_edge() {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let mut challenge = ChallengeInference::running("model:nonce".to_string(), receiver);
        assert!(!challenge.poll_ready());
        let mut attempt = keryx_miner::runtime_stats::begin_inference(
            keryx_miner::runtime_stats::InferenceKind::SoloChallenge,
            None,
        );
        attempt.served(1);
        sender.send((Some("answer".to_string()), attempt)).unwrap();
        assert!(challenge.poll_ready());
        assert!(!challenge.poll_ready(), "latched completion must not request another template");
        assert_eq!(
            challenge
                .result
                .as_ref()
                .and_then(|result| result.as_ref().ok())
                .and_then(|(result, _)| result.as_deref()),
            Some("answer")
        );

        let (sender, receiver) = tokio::sync::oneshot::channel::<(
            Option<String>,
            keryx_miner::runtime_stats::InferenceAttempt,
        )>();
        let mut dropped = ChallengeInference::running("model:nonce".to_string(), receiver);
        drop(sender);
        assert!(dropped.poll_ready());
        assert!(!dropped.poll_ready(), "closed completion must also emit only one edge");
        assert!(matches!(dropped.result, Some(Err(()))));
    }
}

/// H14 private inference (ported from upstream v0.5.6 50d45b1 / 7ca66e6 / a9f509f tests).
#[cfg(test)]
mod private_inference_tests {
    use super::{decode_private_prompt, KeryxdHandler, PrivateRequest, COMPRESSED_PROMPT_MAGIC, MAX_DECOMPRESSED_PROMPT_LEN};
    use keryx_inference::{escrow_pubkey_of, open_response, seal_request, AiRequestPayload};

    /// Only the responses signed by this miner's escrow key mark a request as already answered.
    #[test]
    fn own_answered_request_hashes_keeps_only_this_responder() {
        use keryx_inference::{AiResponder, AiResponsePayload};
        let mine = [0xAAu8; 32];
        let other = [0xBBu8; 32];
        let response = |request_hash: [u8; 32], key: [u8; 32]| crate::proto::RpcTransaction {
            subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_RESPONSE_HEX.to_string(),
            payload: hex::encode(
                AiResponsePayload::new_v2(request_hash, 10, [0x12u8; 34], 3, AiResponder { escrow_pubkey: key, signature: [0u8; 64] })
                    .serialize(),
            ),
            ..Default::default()
        };
        let txs = vec![response([1u8; 32], mine), response([2u8; 32], other), response([3u8; 32], mine)];
        assert_eq!(KeryxdHandler::own_answered_request_hashes(&txs, &mine), vec![[1u8; 32], [3u8; 32]]);
        assert!(KeryxdHandler::own_answered_request_hashes(&txs, &[0xCCu8; 32]).is_empty());
    }

    /// A private request is opened for a named recipient, skipped for anyone else, and queued as
    /// unreadable when addressed to this key but tampered with.
    #[test]
    fn open_private_request_classifies_recipients() {
        let mine = [0x11u8; 32];
        let other = [0x22u8; 32];
        let recipients = [escrow_pubkey_of(&mine).unwrap()];
        let (req, _secret) = seal_request([5u8; 32], 64, 1_000, 2_000, b"sealed prompt", &recipients).unwrap();

        assert!(KeryxdHandler::open_private_request(&req, None).is_none());
        assert!(KeryxdHandler::open_private_request(&req, Some(&other)).is_none());
        let (prompt, private) = KeryxdHandler::open_private_request(&req, Some(&mine)).unwrap();
        assert_eq!(prompt, "sealed prompt");
        assert!(matches!(private, Some(PrivateRequest::Opened(_))));

        // A flipped ciphertext byte still names this key: answered, with the error body.
        let mut tampered = req.serialize();
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        let tampered = AiRequestPayload::deserialize(&tampered).unwrap();
        let (prompt, private) = KeryxdHandler::open_private_request(&tampered, Some(&mine)).unwrap();
        assert!(prompt.is_empty());
        assert_eq!(private, Some(PrivateRequest::Unreadable));
    }

    /// The sealed answer opens with the requester's root key and nobody else's, and the signed
    /// AiResponse carries it inline under a signature that covers it (node-verifiable).
    #[test]
    fn sealed_answer_round_trips_inline_under_the_responder_signature() {
        let secret_hex = "1111111111111111111111111111111111111111111111111111111111111111";
        let dir = std::env::temp_dir().join(format!("keryx-h14-grpc-test-{}", std::process::id()));
        let w = crate::escrow::EscrowWatcher::new(
            secret_hex,
            "keryx:qrxpcusyrxjxghfdumcxm2rqw4dhe3n9hyqpvgn2wfyldltf99w2xhnajuhte",
            dir,
        )
        .unwrap();
        let pubkey = w.pubkey_bytes();
        assert_eq!(escrow_pubkey_of(&w.secret_bytes()).unwrap(), pubkey);

        let (req, secret) = seal_request([5u8; 32], 64, 1_000, 2_000, b"q", &[pubkey]).unwrap();
        let (_, private) = KeryxdHandler::open_private_request(&req, Some(&w.secret_bytes())).unwrap();
        let Some(PrivateRequest::Opened(root_key)) = private else { panic!("not opened") };
        let request_hash = [9u8; 32];
        let body = keryx_inference::seal_response(&root_key, &request_hash, &pubkey, b"answer");
        assert_eq!(open_response(&secret.root_key, &request_hash, &pubkey, &body).unwrap(), b"answer");
        assert!(open_response(&[0u8; 32], &request_hash, &pubkey, &body).is_err());

        let cid = crate::ipfs::sha256_multihash(&body);
        assert_eq!(&cid[..2], &[0x12, 0x20]);
        let resp = KeryxdHandler::signed_response(&w, request_hash, 1_000, cid, 1, Some(body.clone()));
        let wire = resp.serialize();
        assert_eq!(wire.len(), keryx_inference::AI_RESPONSE_PAYLOAD_V2_LEN + keryx_inference::AI_RESPONSE_EXT_HEADER_LEN + body.len());
        let parsed = keryx_inference::AiResponsePayload::deserialize(&wire).unwrap();
        assert_eq!(parsed.private_body.as_deref(), Some(body.as_slice()));
        // Node check: schnorr over blake2b-256("KeryxServiceResponderV1" || v1 head || extension).
        let verify = |signed: &[u8]| {
            let mut h = blake2b_simd::Params::new().hash_length(32).to_state();
            h.update(b"KeryxServiceResponderV1");
            h.update(signed);
            let msg = secp256k1::Message::from_digest_slice(h.finalize().as_bytes()).unwrap();
            let r = parsed.responder.unwrap();
            let pk = secp256k1::XOnlyPublicKey::from_slice(&r.escrow_pubkey).unwrap();
            let sig = secp256k1::schnorr::Signature::from_slice(&r.signature).unwrap();
            secp256k1::SECP256K1.verify_schnorr(&sig, &msg, &pk).is_ok()
        };
        assert!(verify(&parsed.signed_bytes()));
        let mut swapped = parsed.clone();
        swapped.private_body = Some(vec![0xCD; body.len()]);
        assert!(!verify(&swapped.signed_bytes()), "body swap under a signed head must fail");
        // A body-less response keeps the historical 78-byte signed message (pre-H14 unchanged).
        let plain = KeryxdHandler::signed_response(&w, request_hash, 1_000, cid, 1, None);
        assert_eq!(plain.serialize().len(), keryx_inference::AI_RESPONSE_PAYLOAD_V2_LEN);
        assert_eq!(plain.signed_bytes().len(), keryx_inference::AI_RESPONSE_PAYLOAD_LEN);
    }

    #[test]
    fn private_prompt_inflates_when_marked() {
        use std::io::Write;
        let deflate = |data: &[u8]| {
            let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
            enc.write_all(data).unwrap();
            let mut out = COMPRESSED_PROMPT_MAGIC.to_vec();
            out.extend(enc.finish().unwrap());
            out
        };
        assert_eq!(decode_private_prompt("Bonjour, ça va ?".as_bytes()).as_deref(), Some("Bonjour, ça va ?"));
        let text = "Entrée 41 — le code est ORCHIDÉE-7319. ".repeat(500);
        let packed = deflate(text.as_bytes());
        assert!(packed.len() < text.len() / 10);
        assert_eq!(decode_private_prompt(&packed).as_deref(), Some(text.as_str()));
        let mut corrupt = COMPRESSED_PROMPT_MAGIC.to_vec();
        corrupt.extend([0xFF; 32]);
        assert_eq!(decode_private_prompt(&corrupt), None);
        let bomb = deflate(&vec![b'a'; MAX_DECOMPRESSED_PROMPT_LEN as usize + 1]);
        assert_eq!(decode_private_prompt(&bomb), None);
    }
}
