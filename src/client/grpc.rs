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
    max_tokens: usize,
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

    /// In-flight SLM inference task: (full key, request_hash, result + stats-attempt receiver).
    /// The attempt remains live through IPFS + submission, making success/failure exactly-once.
    inference_rx: Option<(
        AiRequestKey,
        [u8; 32],
        oneshot::Receiver<(Option<String>, keryx_miner::runtime_stats::InferenceAttempt)>,
    )>,

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
                    let challenge_finished = !regular_finished && self.challenge_inference_ready();
                    if regular_finished || challenge_finished {
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
            || self.inference_rx.as_ref().map_or(false, |(in_flight, _, _)| in_flight == key)
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
        let in_flight_key = self.inference_rx.as_ref().map(|(key, _, _)| *key);
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
        for tx in txs {
            // (raw, model_id, prompt, max_tokens, inference_reward)
            let extracted: Option<(Vec<u8>, [u8; 32], String, usize, u64)> =
                if tx.subnetwork_id == keryx_inference::SUBNETWORK_ID_AI_REQUEST_HEX {
                    // Binary AiRequestPayload (dedicated AI subnetwork).
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        keryx_inference::AiRequestPayload::deserialize(&raw).map(|req| {
                            let model_id = req.model_id;
                            let prompt = String::from_utf8_lossy(&req.prompt).into_owned();
                            let max_tokens = req.max_tokens as usize;
                            let inference_reward = req.inference_reward;
                            (raw, model_id, prompt, max_tokens, inference_reward)
                        })
                    })
                } else if !tx.inputs.is_empty() {
                    // KRX:AI:1: JSON format — model routed by "m" field, skipped if not loaded.
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        Self::parse_krx_ai_payload(&raw).and_then(|(model_name, prompt, max_tokens)| {
                            let model_id = keryx_miner::models::find(&model_name)?.model_id;
                            Some((raw, model_id, prompt, max_tokens, 0u64))
                        })
                    })
                } else {
                    None // coinbase — skip
                };

            if let Some((raw, model_id, prompt, max_tokens, inference_reward)) = extracted {
                if let Err(reason) = keryx_miner::slm::validate_inference_request(
                    &prompt,
                    max_tokens,
                    keryx_miner::slm::DEFAULT_INFERENCE_DEADLINE_MS,
                ) {
                    log::warn!("OPoI: rejecting invalid AiRequest: {}", reason);
                    continue;
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
                let key = AiRequestKey::new(&request_hash, &model_id, &prompt, max_tokens);
                let log_id = Self::request_log_id(&request_hash);
                let queued =
                    self.enqueue_ai_request(QueuedAiRequest { key, request_hash, model_id, prompt, max_tokens });
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

    /// Starts SLM inference for the next queued AiRequest, if no inference is
    /// already in flight and a response slot is free.
    fn try_start_inference(&mut self) -> bool {
        if self.inference_rx.is_some() {
            return false;
        }
        while let Some(request) = self.ai_request_queue.pop_front() {
            keryx_miner::runtime_stats::set_connection_inference_queue(
                self.runtime_generation,
                self.ai_request_queue.len(),
                AI_REQUEST_QUEUE_CAPACITY,
            );
            let QueuedAiRequest { key, request_hash, model_id, prompt, max_tokens } = request;
            let log_id = Self::request_log_id(&request_hash);
            // Second guard: re-check readiness at execution time (files could have been deleted).
            if !keryx_miner::slm::model_serveable(&model_id) {
                log::error!("OPoI: model became unavailable after queuing id={} — discarding request", log_id);
                keryx_miner::runtime_stats::record_inference_failed_request(
                    keryx_miner::runtime_stats::InferenceKind::SoloRequest,
                );
                self.finish_ai_request(key, false);
                continue;
            }
            info!("OPoI: spawning SLM inference (max_tokens={})", max_tokens);
            let (tx_done, rx_done) = oneshot::channel::<(
                Option<String>,
                keryx_miner::runtime_stats::InferenceAttempt,
            )>();
            let task_id = log_id;
            let mut runtime_attempt = keryx_miner::runtime_stats::begin_inference(
                keryx_miner::runtime_stats::InferenceKind::SoloRequest,
                Some(&model_id),
            );
            tokio::task::spawn_blocking(move || {
                // Acquire a concrete card first, then atomically stop whichever MinerManager is
                // current (including one created after a reconnect) before entering GPU code.
                let result = match keryx_miner::slm::acquire_inference_card(
                    &model_id,
                    keryx_miner::slm::DEFAULT_INFERENCE_DEADLINE_MS,
                ) {
                    Some(lease) => {
                        let gpu = lease.gpu();
                        runtime_attempt.set_gpu(gpu);
                        let _pause = crate::miner::begin_inference_pause();
                        keryx_miner::slm::load_and_run_inference_on(gpu, &model_id, &prompt, max_tokens)
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
            self.inference_rx = Some((key, request_hash, rx_done));
            return true;
        }
        false
    }

    /// Polls the in-flight inference task. When complete, uploads the result to
    /// IPFS and submits a zero-input/zero-output AiResponse transaction.
    /// Returns `true` if inference just finished (regardless of tx success).
    async fn poll_inference(&mut self) -> bool {
        let Some((key, request_hash, mut rx)) = self.inference_rx.take() else {
            return false;
        };
        let log_id = Self::request_log_id(&request_hash);
        let (result_opt, mut runtime_attempt) = match rx.try_recv() {
            Ok(result) => result,
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                self.inference_rx = Some((key, request_hash, rx));
                return false;
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                warn!("OPoI: inference task dropped for id={} — request is retryable", log_id);
                self.finish_ai_request(key, false);
                return true;
            }
        };
        let Some(result) = result_opt else {
            // Inference returned None: model not ready or think block exhausted max_tokens.
            // Do NOT upload anything to IPFS — skip this AiResponse entirely.
            info!("OPoI: inference produced no result — AiResponse skipped");
            self.finish_ai_request(key, false);
            return true;
        };

        // The request identity was fixed when the request was queued (H8-gated: TXID or digest).
        info!("OPoI: inference complete, request_hash={}", hex::encode(&request_hash[..8]));

        let ipfs_url = self.ipfs_url.clone();
        let result_clone = result.clone();
        let cid = match tokio::task::spawn_blocking(move || crate::ipfs::upload_with_recovery(&result_clone, &ipfs_url))
            .await
        {
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
        keryx_miner::runtime_stats::record_inference_prepared();

        let challenge_window_end = self.last_known_daa + 1000;
        let response_length = result.split_whitespace().count() as u32;
        // H6 service-bond era: sign the response with the escrow key (payload V2) so it counts
        // as served for the tier cohort — an unsigned response no longer cancels a strike. The
        // era rule mirrors the node's: V2 is rejected before the gate, so v1 is kept below it.
        let v2 = self.last_known_daa >= keryx_miner::pom::pom_v3_activation_daa();
        let resp = match (&self.escrow_watcher, v2) {
            (Some(w), true) => {
                let unsigned =
                    keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length);
                let responder = w.sign_responder(&unsigned.signed_bytes());
                keryx_inference::AiResponsePayload::new_v2(
                    request_hash,
                    challenge_window_end,
                    cid,
                    response_length,
                    responder,
                )
            }
            (None, true) => {
                warn!("OPoI: no escrow key configured — submitting an unsigned (v1) response; it will NOT count for the service bond");
                keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length)
            }
            (_, false) => {
                keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length)
            }
        };
        info!(
            "OPoI: uploading response CID={}, challenge_window_end={}{}",
            resp.cid_v0(),
            challenge_window_end,
            if resp.responder.is_some() { " (signed, V2)" } else { "" }
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
        if let Err(e) = self.client_send(KaspadMessage::submit_transaction(rpc_tx)).await {
            warn!("OPoI: failed to send AiResponse tx: {}", e);
            runtime_attempt.failed();
            self.finish_ai_request(key, false);
            return true;
        }
        runtime_attempt.served(response_length as usize);

        // Register inference escrow outpoint for auto-claim after the challenge window.
        if let Some((txid, inference_reward)) = self.finish_ai_request(key, true) {
            if let Some(w) = self.escrow_watcher.as_mut() {
                w.track_inference_escrow(txid, self.last_known_daa, inference_reward);
            }
        }

        true
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
                                            let _pause = crate::miner::begin_inference_pause();
                                            keryx_miner::slm::load_and_run_inference_on(
                                                gpu, &model_id, &prompt, 64,
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
                // running. Once it acquires a card, its process-global guard stops the walk and this
                // gate keeps every later template stopped until all GPU generation has ended.
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
                    // Validation answer: "cannot find header <hash>" — unknown to this
                    // node (pruned or not yet synced), the entries are kept.
                    was_validation =
                        self.escrow_watcher.as_mut().map_or(false, |w| w.on_block_validation_error(&e.message));
                    if !was_validation {
                        warn!("GetBlockResponse error: {}", e.message);
                    }
                } else if let Some(block) = msg.block {
                    let hash = block.verbose_data.as_ref().map(|v| v.hash.clone()).unwrap_or_default();
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
                        if let Some(e) = err {
                            warn!("OPoI: submit_transaction error: {}", e);
                        }
                    }
                }
            }
            Payload::GetInfoResponse(info) => {
                info!("Keryxd version: {}", info.server_version);
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
                self.client_send(NotifyVirtualSelectedParentChainChangedRequestMessage {}).await?;
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
                None => info!("Registered for virtual chain notifications (escrow tracking)"),
                Some(e) => error!("Failed registering for virtual chain notifications: {:?}", e),
            },
            // Virtual chain advanced: fetch every added chain block in full. Their coinbases
            // are the only ones that materialize UTXOs, so escrow tracking feeds off this
            // stream (handle_block gates tracking on is_chain_block). Removed chain blocks
            // are ignored: entries from reorged-out blocks fail their claims as orphans and
            // are cleaned up by the existing retry/slash machinery.
            Payload::VirtualSelectedParentChainChangedNotification(notif) => {
                for hash in notif.added_chain_block_hashes {
                    self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
                }
            }
            msg => info!("got unknown msg: {:?}", msg),
        }
        Ok(())
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
