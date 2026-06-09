// Automated OPoI escrow claim module.
//
// After each block, scans for coinbase outputs matching this miner's escrow script.
// When the CSV window (36 000 blocks) expires, builds a Schnorr-signed claim TX and
// broadcasts it via gRPC.  State is persisted to `escrow_state.json` so claims survive
// miner restarts.

use blake2b_simd::Params as Blake2bParams;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{fs, io};

use crate::proto::{
    RpcOutpoint, RpcScriptPublicKey, RpcTransaction, RpcTransactionInput, RpcTransactionOutput,
};

const CHALLENGE_WINDOW_BLOCKS: u64 = 36_000;
const CLAIM_FEE_SOMPI: u64 = 30_000_000;
const NATIVE_SUBNETWORK: &str = "0000000000000000000000000000000000000000";

const OP_CSV: u8 = 0xb1;
const OP_CHECKSIG: u8 = 0xac;
const SIG_HASH_ALL: u8 = 0x01;

/// Drop done (claimed / terminally-slashed) entries every N processed blocks so the
/// in-memory vector and the on-disk state stay bounded under a high block rate.
const COMPACT_EVERY_BLOCKS: u32 = 2_000;
/// Minimum wall-clock interval between state-file writes. Without this debounce the
/// watcher rewrites the entire (multi-thousand-entry) state file on every block, which
/// saturates the async client loop and starves block-template delivery → mining stalls.
const STATE_SAVE_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EscrowEntry {
    /// TXID of the source transaction: coinbase txid for block rewards,
    /// or AiRequest txid for inference escrow entries.
    pub coinbase_txid: String,
    /// Block hash of the coinbase block — used to detect red-set exclusion.
    /// Empty for inference escrow entries (non-coinbase TXs may be re-included).
    #[serde(default)]
    pub block_hash: String,
    pub confirm_daa: u64,
    pub amount_sompi: u64,
    /// Index of the escrow output in the source transaction.
    /// Coinbases: varies by mergeset size. Inference: always 1 (output[1]).
    #[serde(default = "default_output_index")]
    pub output_index: u32,
    pub claimed: bool,
    pub slashed: bool,
    #[serde(default)]
    pub orphan_slashed: bool,
    #[serde(default)]
    pub orphan_retries: u8,
    #[serde(default)]
    pub orphan_retry_after_daa: Option<u64>,
    /// True for AI inference escrow entries (from AiRequest output[1]).
    /// Red-set slashing is skipped for these since non-coinbase TXs can be re-included.
    #[serde(default)]
    pub is_inference: bool,
}

fn default_output_index() -> u32 {
    1
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub struct EscrowState {
    pub entries: Vec<EscrowEntry>,
}

/// DAA-score cooldown after an orphan rejection before retrying the claim.
const ORPHAN_RETRY_COOLDOWN_BLOCKS: u64 = 100;
/// DAA-score cooldown after a sequence-lock rejection before retrying.
/// The OP_CSV check uses the selected-chain blue score, which can lag the DAA
/// score by a few blocks in the DAG — a short cooldown avoids hammering the
/// node every block for a TX that will be rejected again immediately.
const SEQ_LOCK_RETRY_COOLDOWN_BLOCKS: u64 = 20;
/// After this many consecutive orphan rejections, give up and slash permanently.
const MAX_ORPHAN_RETRIES: u8 = 10;

pub struct EscrowWatcher {
    secp: secp256k1::Secp256k1<secp256k1::All>,
    secret_key: secp256k1::SecretKey,
    pubkey_bytes: [u8; 32],
    escrow_script: Vec<u8>,
    escrow_script_hex: String,
    payout_spk_version: u16,
    payout_spk_script: Vec<u8>,
    payout_spk_script_hex: String,
    pub state: EscrowState,
    state_path: PathBuf,
    /// Outpoint (txid, output_index) of the claim TX currently in flight.
    pub pending_claim_txid: Option<String>,
    pub pending_claim_output_index: Option<u32>,
    /// DAA score of the most recent block seen — used to set per-entry orphan cooldowns.
    last_daa_score: u64,
    /// O(1) dedup of tracked outpoints ("{txid}:{output_index}") so tracking a coinbase
    /// output no longer scans all of `state.entries`. Rebuilt from `state` on load/compaction.
    outpoint_set: HashSet<String>,
    /// block_hash -> indices into `state.entries`, for O(reds) red-set slashing instead of
    /// scanning every entry per red hash. Rebuilt on load/compaction; appended on track.
    block_index: HashMap<String, Vec<usize>>,
    /// Debounced persistence: pending unsaved changes + last write time.
    dirty: bool,
    last_save: Instant,
    /// handle_block call counter, used to trigger periodic compaction.
    blocks_since_compact: u32,
}

impl EscrowWatcher {
    pub fn new(privkey_hex: &str, mining_address: &str, state_path: PathBuf) -> Result<Self, String> {
        let privkey_bytes = hex::decode(privkey_hex)
            .map_err(|e| format!("Invalid --mining-privkey hex: {}", e))?;
        if privkey_bytes.len() != 32 {
            return Err(format!("--mining-privkey must be 32 bytes (64 hex chars), got {}", privkey_bytes.len()));
        }

        let secp = secp256k1::Secp256k1::new();
        let secret_key = secp256k1::SecretKey::from_slice(&privkey_bytes)
            .map_err(|e| format!("Invalid private key: {}", e))?;
        let keypair = secp256k1::Keypair::from_secret_key(&secp, &secret_key);
        let (xonly, _parity) = keypair.x_only_public_key();
        let pubkey_bytes: [u8; 32] = xonly.serialize();

        let escrow_script = build_escrow_script(&pubkey_bytes);
        let escrow_script_hex = hex::encode(&escrow_script);

        let (payout_spk_version, payout_spk_bytes) = decode_address(mining_address)?;
        let payout_spk_script = build_p2pk_script(&payout_spk_bytes);
        let payout_spk_script_hex = hex::encode(&payout_spk_script);

        let state = load_state(&state_path);

        info!("EscrowWatcher ready: pubkey={}", hex::encode(pubkey_bytes));

        let mut watcher = Self {
            secp,
            secret_key,
            pubkey_bytes,
            escrow_script,
            escrow_script_hex,
            payout_spk_version,
            payout_spk_script,
            payout_spk_script_hex,
            state,
            state_path,
            pending_claim_txid: None,
            pending_claim_output_index: None,
            last_daa_score: 0,
            outpoint_set: HashSet::new(),
            block_index: HashMap::new(),
            dirty: false,
            last_save: Instant::now(),
            blocks_since_compact: 0,
        };
        watcher.rebuild_indexes();
        Ok(watcher)
    }

    /// (Re)build the in-memory lookup indexes from `state.entries`.
    fn rebuild_indexes(&mut self) {
        self.outpoint_set.clear();
        self.block_index.clear();
        for (i, e) in self.state.entries.iter().enumerate() {
            self.outpoint_set.insert(format!("{}:{}", e.coinbase_txid, e.output_index));
            if !e.block_hash.is_empty() {
                self.block_index.entry(e.block_hash.clone()).or_default().push(i);
            }
        }
    }

    /// Drop done (claimed / terminally-slashed) entries and rebuild indexes.
    /// Orphan-retry entries (slashed == false) are kept.
    fn compact(&mut self) {
        let before = self.state.entries.len();
        self.state.entries.retain(|e| !e.claimed && !e.slashed);
        if self.state.entries.len() != before {
            self.rebuild_indexes();
            self.mark_dirty();
        }
    }

    #[inline]
    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Persist state at most once per `STATE_SAVE_INTERVAL` to keep disk I/O off the
    /// per-block hot path. Worst case on crash we lose a couple of seconds of tracking,
    /// which is re-derived from the chain on the next blocks.
    fn maybe_flush(&mut self) {
        if self.dirty && self.last_save.elapsed() >= STATE_SAVE_INTERVAL {
            self.save_state();
            self.dirty = false;
            self.last_save = Instant::now();
        }
    }

    /// Return the 64-char hex x-only public key of the mining key.
    pub fn pubkey_hex(&self) -> String {
        hex::encode(self.pubkey_bytes)
    }

    /// Scan a confirmed block for the miner's escrow output and check for mature claims.
    /// Returns a claim TX to submit if one is ready; `None` otherwise.
    pub fn handle_block(&mut self, block: &crate::proto::RpcBlock) -> Option<RpcTransaction> {
        let daa_score = block.header.as_ref()?.daa_score;
        self.last_daa_score = daa_score;

        // Periodic compaction so the entry vector / state file stay bounded under a flood.
        self.blocks_since_compact += 1;
        if self.blocks_since_compact >= COMPACT_EVERY_BLOCKS {
            self.blocks_since_compact = 0;
            self.compact();
        }

        let block_hash = block.verbose_data.as_ref().map(|v| v.hash.clone()).unwrap_or_default();

        // Proactively slash entries whose coinbase block just entered the red set.
        // mergeSetRedsHashes lists blocks newly classified as red by this block's GHOSTDAG —
        // their coinbase UTXOs will never exist in the virtual UTXO set.
        // O(reds) via block_index instead of scanning all entries per red hash.
        if let Some(verbose) = &block.verbose_data {
            for red_hash in &verbose.merge_set_reds_hashes {
                let Some(indices) = self.block_index.get(red_hash).cloned() else { continue };
                for i in indices {
                    if let Some(entry) = self.state.entries.get_mut(i) {
                        if !entry.claimed && !entry.slashed && !entry.is_inference && !entry.block_hash.is_empty() {
                            warn!(
                                "EscrowWatcher: block {} is red — permanently skipping coinbase={}…",
                                &red_hash[..16.min(red_hash.len())],
                                &entry.coinbase_txid[..16.min(entry.coinbase_txid.len())]
                            );
                            entry.slashed = true;
                            self.mark_dirty();
                        }
                    }
                }
            }
        }

        // Find coinbase TX (no inputs).
        let coinbase = block.transactions.iter().find(|tx| tx.inputs.is_empty())?;
        let coinbase_txid = coinbase.verbose_data.as_ref()?.transaction_id.clone();
        if coinbase_txid.is_empty() {
            return None;
        }

        // Scan all coinbase outputs — a multi-blue mergeset produces one escrow output
        // per blue block, each at a different index.  Hardcoding index 1 would miss all
        // escrow outputs beyond the first blue's pair. O(1) dedup via outpoint_set.
        for (out_idx, output) in coinbase.outputs.iter().enumerate() {
            if let Some(spk) = &output.script_public_key {
                let key = format!("{}:{}", coinbase_txid, out_idx);
                if spk.script_public_key.to_lowercase() == self.escrow_script_hex
                    && spk.version == 0
                    && !self.outpoint_set.contains(&key)
                {
                    info!(
                        "EscrowWatcher: tracked escrow coinbase={}…[{}] daa={} amount={}",
                        &coinbase_txid[..16.min(coinbase_txid.len())],
                        out_idx,
                        daa_score,
                        output.amount
                    );
                    let idx = self.state.entries.len();
                    self.state.entries.push(EscrowEntry {
                        coinbase_txid: coinbase_txid.clone(),
                        block_hash: block_hash.clone(),
                        confirm_daa: daa_score,
                        amount_sompi: output.amount,
                        output_index: out_idx as u32,
                        claimed: false,
                        slashed: false,
                        orphan_slashed: false,
                        orphan_retries: 0,
                        orphan_retry_after_daa: None,
                        is_inference: false,
                    });
                    self.outpoint_set.insert(key);
                    if !block_hash.is_empty() {
                        self.block_index.entry(block_hash.clone()).or_default().push(idx);
                    }
                    self.mark_dirty();
                }
            }
        }

        let claim = self.find_claim(daa_score);
        self.maybe_flush();
        claim
    }

    /// Scan for the next matured, eligible escrow entry and build its claim TX (if any).
    fn find_claim(&mut self, daa_score: u64) -> Option<RpcTransaction> {
        // Only one claim TX in flight at a time.
        if self.pending_claim_txid.is_some() {
            return None;
        }

        // +10 margin: OP_CSV validation uses the selected-chain blue score, which can lag the
        // virtual-state DAA score by several blocks in the BlockDAG.  A single +1 margin is
        // often not enough, causing seq-lock rejections that get retried every block.
        // Per-entry orphan/seq-lock cooldowns are checked individually so they don't block other claims.
        // Inference entries are prioritised: they carry user fees and must not be starved by the
        // large coinbase entry queue.
        let is_eligible = |e: &&EscrowEntry| {
            !e.claimed
                && !e.slashed
                && daa_score >= e.confirm_daa + CHALLENGE_WINDOW_BLOCKS + 10
                && e.orphan_retry_after_daa.map_or(true, |retry_daa| daa_score >= retry_daa)
        };
        let entry = self
            .state
            .entries
            .iter()
            .find(|e| e.is_inference && is_eligible(e))
            .or_else(|| self.state.entries.iter().find(|e| !e.is_inference && is_eligible(e)))?
            .clone();

        match self.build_claim_tx(&entry) {
            Ok(tx) => {
                debug!(
                    "EscrowWatcher: claiming escrow coinbase={}…[{}] (matured at daa {})",
                    &entry.coinbase_txid[..16.min(entry.coinbase_txid.len())],
                    entry.output_index,
                    entry.confirm_daa + CHALLENGE_WINDOW_BLOCKS
                );
                self.pending_claim_output_index = Some(entry.output_index);
                self.pending_claim_txid = Some(entry.coinbase_txid);
                Some(tx)
            }
            Err(e) => {
                warn!("EscrowWatcher: failed to build claim TX: {}", e);
                None
            }
        }
    }

    /// Called when a SubmitTransactionResponse arrives for a pending claim TX.
    pub fn on_submit_response(&mut self, error: Option<String>) {
        let txid = match self.pending_claim_txid.take() {
            Some(t) => t,
            None => return,
        };
        let out_idx = self.pending_claim_output_index.take().unwrap_or(1);

        match error {
            None => {
                info!("EscrowWatcher: claim accepted for coinbase={}…[{}]", &txid[..16.min(txid.len())], out_idx);
                if let Some(e) = self.state.entries.iter_mut().find(|e| {
                    e.coinbase_txid == txid && e.output_index == out_idx
                }) {
                    e.claimed = true;
                }
            }
            Some(err_msg) => {
                if let Some(e) = self.state.entries.iter_mut().find(|e| {
                    e.coinbase_txid == txid && e.output_index == out_idx
                }) {
                    // Retriable rejections: sequence-lock timing races and orphan/dag-reorg
                    // situations where the coinbase's block is off the selected chain.
                    // Any other rejection (double-spend, script failure) is a real slash.
                    let is_orphan = err_msg.contains("orphan");
                    let is_seq_lock = err_msg.contains("sequence lock");
                    if is_orphan {
                        e.orphan_retries += 1;
                        if e.orphan_retries >= MAX_ORPHAN_RETRIES {
                            e.orphan_slashed = false;
                            e.slashed = true;
                            debug!(
                                "EscrowWatcher: coinbase={}…[{}] slashed after {} orphan retries",
                                &txid[..16.min(txid.len())],
                                out_idx,
                                e.orphan_retries
                            );
                        } else {
                            debug!(
                                "EscrowWatcher: claim rejected for coinbase={}…[{}] (orphan, retry {}/{})",
                                &txid[..16.min(txid.len())],
                                out_idx,
                                e.orphan_retries,
                                MAX_ORPHAN_RETRIES
                            );
                            e.orphan_slashed = true;
                            // Per-entry cooldown: only this entry waits, other claims proceed.
                            e.orphan_retry_after_daa =
                                Some(self.last_daa_score + ORPHAN_RETRY_COOLDOWN_BLOCKS);
                        }
                    } else if is_seq_lock {
                        // Apply a per-entry cooldown instead of retrying every block —
                        // the OP_CSV blue-score check may lag DAA score by a few blocks.
                        e.orphan_retry_after_daa =
                            Some(self.last_daa_score + SEQ_LOCK_RETRY_COOLDOWN_BLOCKS);
                        debug!(
                            "EscrowWatcher: claim rejected for coinbase={}…[{}] (sequence lock, retry after daa {})",
                            &txid[..16.min(txid.len())],
                            out_idx,
                            self.last_daa_score + SEQ_LOCK_RETRY_COOLDOWN_BLOCKS,
                        );
                    } else {
                        warn!(
                            "EscrowWatcher: claim rejected for coinbase={}…[{}]: {}",
                            &txid[..16.min(txid.len())], out_idx, err_msg
                        );
                        e.slashed = true;
                    }
                }
            }
        }
        self.mark_dirty();
        self.maybe_flush();
    }

    fn build_claim_tx(&self, entry: &EscrowEntry) -> Result<RpcTransaction, String> {
        let amount_out = entry
            .amount_sompi
            .checked_sub(CLAIM_FEE_SOMPI)
            .filter(|&a| a > 0)
            .ok_or("escrow amount too small to cover claim fee")?;

        let coinbase_bytes: [u8; 32] = hex::decode(&entry.coinbase_txid)
            .map_err(|e| format!("bad coinbase txid: {}", e))?
            .try_into()
            .map_err(|_| "coinbase txid must be 32 bytes")?;

        let sighash = compute_sighash(
            &coinbase_bytes,
            entry.output_index,
            entry.amount_sompi,
            &self.escrow_script,
            self.payout_spk_version,
            &self.payout_spk_script,
            amount_out,
        );

        let msg = secp256k1::Message::from_digest_slice(&sighash)
            .map_err(|e| format!("sighash message error: {}", e))?;
        let keypair = secp256k1::Keypair::from_secret_key(&self.secp, &self.secret_key);
        let sig = self.secp.sign_schnorr_no_aux_rand(&msg, &keypair);

        // signature_script: OpData65 (0x41) | 64-byte sig | SIG_HASH_ALL (0x01)
        let mut sig_script = Vec::with_capacity(66);
        sig_script.push(0x41u8);
        sig_script.extend_from_slice(sig.as_ref());
        sig_script.push(SIG_HASH_ALL);

        Ok(RpcTransaction {
            version: 0,
            inputs: vec![RpcTransactionInput {
                previous_outpoint: Some(RpcOutpoint {
                    transaction_id: entry.coinbase_txid.clone(),
                    index: entry.output_index,
                }),
                signature_script: hex::encode(&sig_script),
                sequence: CHALLENGE_WINDOW_BLOCKS,
                sig_op_count: 1,
                verbose_data: None,
            }],
            outputs: vec![RpcTransactionOutput {
                amount: amount_out,
                script_public_key: Some(RpcScriptPublicKey {
                    version: self.payout_spk_version as u32,
                    script_public_key: self.payout_spk_script_hex.clone(),
                }),
                verbose_data: None,
            }],
            lock_time: 0,
            subnetwork_id: NATIVE_SUBNETWORK.to_string(),
            gas: 0,
            payload: String::new(),
            mass: 0,
            verbose_data: None,
        })
    }

    /// Record an AI inference escrow outpoint (AiRequest output[1]) to claim after the challenge window.
    /// Called by grpc.rs after a successful AiResponse submission.
    pub fn track_inference_escrow(&mut self, ai_request_txid: String, confirm_daa: u64, escrow_amount: u64) {
        if escrow_amount == 0 {
            return;
        }
        let key = format!("{}:1", ai_request_txid);
        if self.outpoint_set.contains(&key) {
            return;
        }
        info!(
            "EscrowWatcher: tracking inference escrow txid={}… daa={} amount={}",
            &ai_request_txid[..16.min(ai_request_txid.len())],
            confirm_daa,
            escrow_amount,
        );
        self.state.entries.push(EscrowEntry {
            coinbase_txid: ai_request_txid,
            block_hash: String::new(),
            confirm_daa,
            amount_sompi: escrow_amount,
            output_index: 1,
            claimed: false,
            slashed: false,
            orphan_slashed: false,
            orphan_retries: 0,
            orphan_retry_after_daa: None,
            is_inference: true,
        });
        self.outpoint_set.insert(key);
        self.mark_dirty();
        self.maybe_flush();
    }

    fn save_state(&self) {
        match serde_json::to_string_pretty(&self.state) {
            Ok(json) => {
                if let Err(e) = fs::write(&self.state_path, &json) {
                    warn!("EscrowWatcher: failed to save state: {}", e);
                }
            }
            Err(e) => warn!("EscrowWatcher: failed to serialize state: {}", e),
        }
    }
}

/// Load the OPoI escrow private key from `path`. Fails if the file does not exist.
/// Used by --recover-escrow where generating a new key would silently query with the wrong pubkey.
pub fn load_key(path: &str) -> Result<String, String> {
    let p = std::path::Path::new(path);
    if !p.exists() {
        return Err(format!(
            "Escrow key file '{}' not found. Run the miner once to generate it, or pass --escrow-key-file <path>.",
            path
        ));
    }
    let s = fs::read_to_string(p)
        .map_err(|e| format!("Failed to read escrow key file '{}': {}", path, e))?;
    let privkey = s.trim().to_string();
    if privkey.len() != 64 || !privkey.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!(
            "Escrow key file '{}' must contain exactly 64 hex chars",
            path
        ));
    }
    Ok(privkey)
}

/// Load the OPoI escrow private key from `path`, generating a new one if absent.
/// The file contains exactly 64 lowercase hex characters (32-byte Schnorr private key).
pub fn load_or_generate_key(path: &str) -> Result<String, String> {
    use rand::RngCore;
    let p = std::path::Path::new(path);
    if p.exists() {
        let s = fs::read_to_string(p)
            .map_err(|e| format!("Failed to read escrow key file '{}': {}", path, e))?;
        let privkey = s.trim().to_string();
        if privkey.len() != 64 || !privkey.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!(
                "Escrow key file '{}' must contain exactly 64 hex chars — delete it to regenerate",
                path
            ));
        }
        return Ok(privkey);
    }

    let mut privkey_bytes = [0u8; 32];
    loop {
        rand::thread_rng().fill_bytes(&mut privkey_bytes);
        if secp256k1::SecretKey::from_slice(&privkey_bytes).is_ok() {
            break;
        }
    }
    let privkey_hex = hex::encode(privkey_bytes);

    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_slice(&privkey_bytes).unwrap();
    let kp = secp256k1::Keypair::from_secret_key(&secp, &sk);
    let (xonly, _) = kp.x_only_public_key();
    let pubkey_hex = hex::encode(xonly.serialize());

    fs::write(p, &privkey_hex)
        .map_err(|e| format!("Failed to write escrow key file '{}': {}", path, e))?;

    info!("OPoI escrow keypair generated — saved to '{}'", path);
    info!("  Escrow pubkey : {}", pubkey_hex);
    info!("  Keep '{}' safe — needed to claim your OPoI escrow rewards.", path);
    Ok(privkey_hex)
}

/// Derive the x-only public key hex (64 hex chars) from a hex-encoded private key.
pub fn pubkey_hex_from_privkey(privkey_hex: &str) -> Result<String, String> {
    let privkey_bytes = hex::decode(privkey_hex)
        .map_err(|e| format!("Invalid privkey hex: {}", e))?;
    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_slice(&privkey_bytes)
        .map_err(|e| format!("Invalid private key: {}", e))?;
    let kp = secp256k1::Keypair::from_secret_key(&secp, &sk);
    let (xonly, _) = kp.x_only_public_key();
    Ok(hex::encode(xonly.serialize()))
}

fn load_state(path: &PathBuf) -> EscrowState {
    let state = match fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            warn!("EscrowWatcher: could not parse {}: {} — starting fresh", path.display(), e);
            EscrowState::default()
        }),
        Err(e) if e.kind() == io::ErrorKind::NotFound => EscrowState::default(),
        Err(e) => {
            warn!("EscrowWatcher: could not read {}: {} — starting fresh", path.display(), e);
            EscrowState::default()
        }
    };

    state
}

// ── Script builders ───────────────────────────────────────────────────────────

/// Build the CSV P2PK escrow script identical to the node's `build_escrow_script`.
///
/// `<CHALLENGE_WINDOW_BLOCKS_LE> OP_CSV OP_DATA_32 <pubkey_32> OP_CHECKSIG`
///
/// Keryx's OP_CSV pops its argument, so no OP_DROP is needed after it.
fn build_escrow_script(pubkey: &[u8; 32]) -> Vec<u8> {
    // 36 000 = 0x8CA0 — trim trailing zero bytes for minimal encoding
    let le = CHALLENGE_WINDOW_BLOCKS.to_le_bytes();
    let trimmed_len = 8 - le.iter().rev().position(|&b| b != 0).unwrap_or(8);
    let seq_bytes = &le[..trimmed_len];

    let mut script = Vec::with_capacity(3 + 1 + 33 + 1);
    script.push(seq_bytes.len() as u8); // OpData2 = 0x02
    script.extend_from_slice(seq_bytes);
    script.push(OP_CSV);
    script.push(0x20); // OpData32
    script.extend_from_slice(pubkey);
    script.push(OP_CHECKSIG);
    script
}

/// Standard Schnorr P2PK script: `OP_DATA_32 <pubkey_32> OP_CHECKSIG`
fn build_p2pk_script(pubkey: &[u8; 32]) -> Vec<u8> {
    let mut script = Vec::with_capacity(34);
    script.push(0x20); // OpData32
    script.extend_from_slice(pubkey);
    script.push(OP_CHECKSIG);
    script
}

// ── Sighash ───────────────────────────────────────────────────────────────────

/// Compute the Keryx Schnorr sighash (SIG_HASH_ALL) for a single-input escrow claim TX.
///
/// Mirrors `calc_schnorr_signature_hash` in consensus/core/src/hashing/sighash.rs.
/// All sub-hashes use Blake2b-256 keyed with `b"TransactionSigningHash"`.
fn compute_sighash(
    coinbase_txid: &[u8; 32],
    output_index: u32,
    escrow_amount: u64,
    escrow_script: &[u8],
    payout_spk_version: u16,
    payout_spk_script: &[u8],
    payout_amount: u64,
) -> [u8; 32] {
    const KEY: &[u8] = b"TransactionSigningHash";
    let new_h = || Blake2bParams::new().hash_length(32).key(KEY).to_state();

    // previous_outputs_hash: Blake2b(txid_32 | index_u32_LE)
    let prev_out_hash = {
        let mut h = new_h();
        h.update(coinbase_txid);
        h.update(&output_index.to_le_bytes());
        h.finalize()
    };

    // sequences_hash: Blake2b(sequence_u64_LE)
    let seqs_hash = {
        let mut h = new_h();
        h.update(&CHALLENGE_WINDOW_BLOCKS.to_le_bytes());
        h.finalize()
    };

    // sig_op_counts_hash: Blake2b(sig_op_count_u8)
    let sigops_hash = {
        let mut h = new_h();
        h.update(&[1u8]);
        h.finalize()
    };

    // outputs_hash: Blake2b(value_u64_LE | spk_version_u16_LE | len_u64_LE | spk_script)
    let outs_hash = {
        let mut h = new_h();
        h.update(&payout_amount.to_le_bytes());
        h.update(&payout_spk_version.to_le_bytes());
        h.update(&(payout_spk_script.len() as u64).to_le_bytes()); // write_var_bytes len
        h.update(payout_spk_script);
        h.finalize()
    };

    // payload_hash = ZERO_HASH (native subnetwork + empty payload)
    let payload_hash = [0u8; 32];

    let mut h = new_h();
    h.update(&0u16.to_le_bytes()); // tx.version
    h.update(prev_out_hash.as_bytes());
    h.update(seqs_hash.as_bytes());
    h.update(sigops_hash.as_bytes());
    // Input-specific:
    h.update(coinbase_txid); // prev_outpoint.transaction_id
    h.update(&output_index.to_le_bytes()); // prev_outpoint.index
    h.update(&0u16.to_le_bytes()); // utxo spk version = 0
    h.update(&(escrow_script.len() as u64).to_le_bytes()); // write_var_bytes len
    h.update(escrow_script); // write_var_bytes data
    h.update(&escrow_amount.to_le_bytes()); // utxo.amount
    h.update(&CHALLENGE_WINDOW_BLOCKS.to_le_bytes()); // input.sequence
    h.update(&[1u8]); // input.sig_op_count
    h.update(outs_hash.as_bytes());
    h.update(&0u64.to_le_bytes()); // tx.lock_time
    h.update(&[0u8; 20]); // tx.subnetwork_id (NATIVE = all zeros 20 bytes)
    h.update(&0u64.to_le_bytes()); // tx.gas
    h.update(&payload_hash); // payload_hash (ZERO_HASH)
    h.update(&[SIG_HASH_ALL]); // hash_type

    let out = h.finalize();
    let mut result = [0u8; 32];
    result.copy_from_slice(out.as_bytes());
    result
}

// ── Address decoding ──────────────────────────────────────────────────────────

/// Decode a bech32 Keryx address into `(spk_version, spk_32_bytes)`.
fn decode_address(addr: &str) -> Result<(u16, [u8; 32]), String> {
    let colon = addr.find(':').ok_or("Missing ':' in address")?;
    let data_with_checksum = &addr[colon + 1..];
    if data_with_checksum.len() < 9 {
        return Err("Address too short".into());
    }
    let data_str = &data_with_checksum[..data_with_checksum.len() - 8];

    const CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    let mut rev = [0xffu8; 128];
    for (i, &c) in CHARSET.iter().enumerate() {
        rev[c as usize] = i as u8;
    }

    let mut data5: Vec<u8> = Vec::with_capacity(data_str.len());
    for c in data_str.chars() {
        let idx = c as usize;
        if idx >= 128 || rev[idx] == 0xff {
            return Err(format!("Invalid bech32 character '{}'", c));
        }
        data5.push(rev[idx]);
    }

    let mut bytes: Vec<u8> = Vec::new();
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for &b in &data5 {
        buf = (buf << 5) | b as u32;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            bytes.push((buf >> bits) as u8);
        }
    }

    if bytes.len() < 33 {
        return Err(format!("Expected ≥33 decoded bytes, got {}", bytes.len()));
    }
    let version = bytes[0] as u16;
    let mut spk = [0u8; 32];
    spk.copy_from_slice(&bytes[1..33]);
    Ok((version, spk))
}
