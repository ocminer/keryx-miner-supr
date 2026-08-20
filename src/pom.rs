//! Proof-of-Model — miner-side possession proof builder (build order §6).
//!
//! Byte-exact mirror of the node's verifier (`keryx-node-hardfork consensus/core/src/pom.rs`)
//! and the canonical reference (`pom-core`). The miner runs the memory-hard walk over its
//! resident weight blob; once a winning nonce is found, `build_proof` re-walks (recording the
//! trace), commits it, and opens the `t` Fiat-Shamir-selected steps with Merkle paths to the
//! tier root `R_T` and the trace root.
//!
//! The `PomProof`/`PomOpening` structs MUST keep the exact field order/types of the node's
//! (borsh wire format), and the primitives MUST stay bit-identical (the node re-derives the
//! same challenges and recomputes the same transitions). See POM_CONSENSUS_SPEC.md.

use borsh::{BorshDeserialize, BorshSerialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

pub(crate) fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::FileExt;
        return file.read_exact_at(buf, offset);
    }
    #[cfg(target_family = "windows")]
    {
        use std::os::windows::fs::FileExt;
        let mut pos = 0usize;
        while pos < buf.len() {
            let n = file.seek_read(&mut buf[pos..], offset + pos as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read_exact_at: eof"));
            }
            pos += n;
        }
        return Ok(());
    }
}
use std::sync::OnceLock;

/// Resident-tree switch. Resolved ONCE at startup from the CLI (`--resident-tree` /
/// `--no-resident-tree`) with `KERYX_RESIDENT_TREE` as a back-compat fallback, then read by
/// both the CUDA (`pom_gpu`) and OpenCL (`pom_opencl`) index-build paths. OFF by default:
/// building the full Merkle tree in RAM (~9.6 GB at tier 0) is only worth it for solo mining.
static RESIDENT_TREE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set the resident-tree switch (call once, from main after CLI parse).
pub fn set_resident_tree(on: bool) {
    RESIDENT_TREE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the in-RAM resident Merkle tree is enabled for this process.
pub fn resident_tree_enabled() -> bool {
    RESIDENT_TREE.load(std::sync::atomic::Ordering::Relaxed)
}

pub const CHUNK_WORDS: usize = 4; // 32 B chunk

/// Walk length / opening count — MUST match the node's `POM_WALK_STEPS` / `POM_OPENINGS`.
/// K=256 — chosen compromise (~25 MH/s on a 3090, solid possession).
pub const POM_WALK_STEPS: u32 = 256;
pub const POM_OPENINGS: usize = 32;

// --- wire struct (field order == node's PomProof at keryxd v1.5.1) ---

/// PoM proof container — mirror of the node's `PomProof`. Post-relaunch the miner only ever emits a
/// **v4** witness. The legacy fields below are canonical placeholders (zeroed / empty / `None`) kept
/// ONLY so the borsh field order stays byte-identical to the node's struct; their concrete inner
/// types are irrelevant, since an empty `Vec` and a `None` `Option` serialize the same bytes for any
/// element type. The wire envelope is UNCHANGED from v3: this borsh blob is the 6th `mining.submit`
/// param (`pomProofHex`), now with `v4` populated instead of `v3`.
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProof {
    pub tier: u8,
    pub trace_root: [u8; 32],
    pub pow_value: [u8; 32],
    pub final_state: u64,
    pub initial_trace_path: Vec<[u8; 32]>,
    pub final_trace_path: Vec<[u8; 32]>,
    pub openings: Vec<[u8; 32]>,
    pub steps_v2: Option<Vec<u8>>,
    pub v3: Option<Vec<u8>>,
    /// v4 re-walk witness — the only populated proof body post-relaunch.
    pub v4: Option<crate::pom_v4::PomProofV4>,
}

impl PomProof {
    /// A canonical v4 container: every legacy field is its empty placeholder, `v4` carries the
    /// witness. `final_state` = `pom_v4::fold64(v4_state_root(S_K))`, `pow_value` = the era pow fold.
    pub fn v4(tier: u8, pow_value: [u8; 32], final_state: u64, v4: crate::pom_v4::PomProofV4) -> Self {
        Self {
            tier,
            trace_root: [0u8; 32],
            pow_value,
            final_state,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: None,
            v3: None,
            v4: Some(v4),
        }
    }

    /// Canonical borsh wire encoding (the `pomProofHex` submit param) — mirror of the node's
    /// `to_wire_bytes` for the v4 era.
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        borsh::to_vec(self).expect("PomProof borsh serialize")
    }

    /// Decode the canonical wire encoding — mirror of the node's `from_wire_bytes`.
    pub fn from_wire_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        borsh::from_slice::<PomProof>(bytes)
    }
}

// --- byte-exact primitives (mirror node) ---

#[inline]
pub fn blake(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

#[inline]
pub fn chunk_to_words(c: &[u8; 32]) -> [u64; CHUNK_WORDS] {
    let mut w = [0u64; CHUNK_WORDS];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

#[inline]
pub fn words_to_bytes(w: &[u64; CHUNK_WORDS]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, wi) in w.iter().enumerate() {
        b[i * 8..i * 8 + 8].copy_from_slice(&wi.to_le_bytes());
    }
    b
}

#[inline]
fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    blake(&buf)
}

pub fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    true
}

#[inline]
fn pph_words(pre_pow_hash: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(pre_pow_hash[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// H3 domain salt applied to the pre_pow_hash words feeding BOTH PoM folds (block seed AND pow
/// value) at/after `POM_LEVEL_ACTIVATION_DAA`. Forced-update mechanism (same spirit as the
/// kHeavyHash matrix salts): every walk trajectory and pow value changes at the gate, so pre-H3
/// binaries produce proofs the node rejects. The CUDA/OpenCL kernels are UNCHANGED — the host
/// salts the pph words before upload (the kernel folds whatever words it receives).
/// Derivation: sha256("keryx-h3-pom-pph-salt") read as 4 little-endian u64 words.
/// MUST equal the node's `POM_H3_PPH_SALT` (consensus/core/src/pom.rs @ v1.3.1).
pub const POM_H3_PPH_SALT: [u64; 4] = [0x7C99D381176D4EC4, 0xC2E28E3E28118C36, 0xD496CE1B129B76CA, 0x47CF0979FA580BCE];

/// pph words for the era selected by `h3` (raw pre-H3, XOR-salted at/after the H3 gate). Single
/// point the salt is applied — shared by the host proof builder AND every GPU backend.
#[inline]
pub fn pph_words_for_era(pre_pow_hash: &[u8; 32], h3: bool) -> [u64; 4] {
    let mut w = pph_words(pre_pow_hash);
    if h3 {
        for (wi, si) in w.iter_mut().zip(POM_H3_PPH_SALT.iter()) {
            *wi ^= si;
        }
    }
    w
}

/// H5.1 domain salt (emergency relaunch 2026-07-24) applied to the pph words feeding the WALK SEED
/// fold ONLY, at/after `h5_1_activation_daa()`. Unlike the H3 salt this does NOT touch the pow fold
/// (header-level pow + block levels stay era-stable). Every walk trajectory changes at the gate, so
/// pre-H5.1 blocks fail node body validation — the forced-update mechanism for the relaunch.
/// Derivation: sha256("keryx-h5.1-pom-pph-salt") read as 4 little-endian u64 words.
/// MUST equal the node's `POM_H5_1_PPH_SALT`.
pub const POM_H5_1_PPH_SALT: [u64; 4] = [0x0F86D1400D3F8664, 0xC296B67C7A7A6A5B, 0x5F89AD33D961FEAA, 0xAC6C9AFDFA053580];

/// H5.2 domain salt (chain anchoring, keryxd v1.4.0, 2026-07-25) applied to the pph words feeding
/// the WALK SEED fold ONLY, at/after `h5_2_activation_daa()`. Same mechanism as H5.1: rotating the
/// seed salt invalidates every pre-gate fork point (the relaunched chain's accumulated work is
/// small, so a private pre-gate branch could otherwise outweigh it). The pow fold keeps the H3
/// salt — header pow / block levels are era-stable.
/// Derivation: sha256("keryx-h5.2-pom-pph-salt") read as 4 little-endian u64 words.
/// MUST equal the node's `POM_H5_2_PPH_SALT` (keryx-node v1.4.0 consensus/core/src/pom.rs).
pub const POM_H5_2_PPH_SALT: [u64; 4] = [0x584ADE0A598D896D, 0x8783631D81BC2695, 0x2917FCF883A0B862, 0x533CCCFAC88FD614];

/// pph words feeding the SEED fold for the era selected by (`h3`, `h5_1`, `h5_2`). Each seed-salt
/// era uses RAW pph XOR its own salt (NO stacking — the v0.9.0 bug); the pow fold keeps using
/// `pph_words_for_era` (H3) in every era.
#[inline]
pub fn seed_pph_words_for_era(pre_pow_hash: &[u8; 32], h3: bool, h5_1: bool, h5_2: bool) -> [u64; 4] {
    if h5_2 {
        // H5.2 seed words = RAW pph XOR the H5.2 salt ONLY (node `pph_words_h5_2`).
        let mut w = pph_words(pre_pow_hash);
        for (wi, si) in w.iter_mut().zip(POM_H5_2_PPH_SALT.iter()) {
            *wi ^= si;
        }
        w
    } else if h5_1 {
        // H5.1 seed words = RAW pph XOR the H5.1 salt ONLY. The H3 salt is NOT stacked here —
        // node `pph_words_h5_1` = `pph_words` (raw) XOR POM_H5_1_PPH_SALT. H3 still salts the POW
        // fold (`pom_pow_value` via `pph_words_for_era`), just not the seed at/after the H5.1 gate.
        let mut w = pph_words(pre_pow_hash);
        for (wi, si) in w.iter_mut().zip(POM_H5_1_PPH_SALT.iter()) {
            *wi ^= si;
        }
        w
    } else {
        pph_words_for_era(pre_pow_hash, h3)
    }
}

#[inline]
fn pom_block_seed_from_words(p: &[u64; 4], timestamp: u64, nonce: u64) -> u64 {
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ p[0]);
    s = mix64(s ^ p[1]);
    s = mix64(s ^ p[2]);
    s = mix64(s ^ p[3]);
    s
}

/// Canonical block seed = initial walk state. mix64-fold of (nonce, time, pre_pow_hash).
/// BYTE-IDENTICAL to `pom_mine.cu::pom_seed_fold` and the node's `pom_block_seed`(`_h3`/`_h5_1`/`_h5_2`).
/// `h5_1`/`h5_2` swap the seed pph words to that era's salted set (the pow fold stays H3-salted).
pub fn pom_block_seed(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64, h3: bool, h5_1: bool, h5_2: bool) -> u64 {
    pom_block_seed_from_words(&seed_pph_words_for_era(pre_pow_hash, h3, h5_1, h5_2), timestamp, nonce)
}

/// Canonical pow value (256-bit LE) = mix64-fold of (final_state, pre_pow_hash).
/// BYTE-IDENTICAL to `pom_mine.cu::pom_pow_fold` and the node's `pom_pow_value`(`_h3`).
pub fn pom_pow_value(final_state: u64, pre_pow_hash: &[u8; 32], h3: bool) -> [u8; 32] {
    let p = pph_words_for_era(pre_pow_hash, h3);
    let o0 = mix64(final_state ^ p[0] ^ 0x9E3779B97F4A7C15);
    let o1 = mix64(o0 ^ p[1] ^ 0xC2B2AE3D27D4EB4F);
    let o2 = mix64(o1 ^ p[2] ^ 0x165667B19E3779F9);
    let o3 = mix64(o2 ^ p[3] ^ 0xD6E8FEB86659FD93);
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&o0.to_le_bytes());
    out[8..16].copy_from_slice(&o1.to_le_bytes());
    out[16..24].copy_from_slice(&o2.to_le_bytes());
    out[24..32].copy_from_slice(&o3.to_le_bytes());
    out
}

// --- v4 seed (relaunch era). The v4 WALK SEED uses its own pph salt; the v4 POW fold keeps the
// --- H3-salted pph words (`pph_words_for_era(.., true)`), i.e. "v4 pow uses the h3 fold". ---

/// v4 seed salt. MUST equal the node's `POM_V4_PPH_SALT` (consensus/core/src/pom.rs @ v1.5.1).
pub const POM_V4_PPH_SALT: [u64; 4] =
    [0x7D7BC84C8D18DE80, 0xDE48EE16AE3F1541, 0x3305F1952B30384A, 0xF78C133968D388B7];

/// v4 seed pph words = raw pph XOR the v4 salt (does NOT touch the pow fold).
#[inline]
pub fn pph_words_v4(pre_pow_hash: &[u8; 32]) -> [u64; 4] {
    let mut w = pph_words(pre_pow_hash);
    for (wi, si) in w.iter_mut().zip(POM_V4_PPH_SALT.iter()) {
        *wi ^= si;
    }
    w
}

/// v4 block seed. BYTE-IDENTICAL to the node's `pom_block_seed_v4`.
pub fn pom_block_seed_v4(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64) -> u64 {
    pom_block_seed_from_words(&pph_words_v4(pre_pow_hash), timestamp, nonce)
}

pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    assert!(!leaves.is_empty(), "merkle_root: empty leaves");
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

pub fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<[u8; 32]> {
    let mut path = Vec::new();
    let mut level = leaves.to_vec();
    let mut idx = index;
    while level.len() > 1 {
        let sib_idx = if idx & 1 == 0 { idx + 1 } else { idx - 1 };
        let sib = if sib_idx < level.len() { level[sib_idx] } else { level[idx] };
        path.push(sib);
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        idx >>= 1;
        level = next;
    }
    path
}

pub(crate) fn verify_merkle(leaf: [u8; 32], index: u64, path: &[[u8; 32]], root: &[u8; 32]) -> bool {
    let mut acc = leaf;
    let mut idx = index;
    for sib in path {
        acc = if idx & 1 == 0 { hash_pair(&acc, sib) } else { hash_pair(sib, &acc) };
        idx >>= 1;
    }
    &acc == root
}

/// Fiat-Shamir challenge step-indices — byte-layout identical to node/pom-core.

/// Source of the raw 32 B canonical chunks for `read_chunk`.
enum ChunkSource {
    /// In-RAM chunks for the synthetic test helper (`synth_index`), built without a GGUF.
    /// Test-only: production always uses `Gguf`, so it is compiled out of release builds.
    #[cfg(test)]
    Ram(Vec<u8>),
    /// Chunks read on demand from the GGUF via `pread` — NO host copy (saves ~1x model size of
    /// RAM, ~42 GB for the 70B). `table[j] = (canonical chunk index of tensor j's first chunk,
    /// absolute file byte offset of that chunk)`, ascending by chunk index; `read_chunk`
    /// binary-searches it. The GGUF's on-disk quantized bytes are byte-identical to candle's
    /// `qt.data()` used to build the leaves (`tensor` seeks to the same `tensor_data_offset + offset`).
    Gguf { file: File, table: Vec<(u64, u64)> },
}

/// Canonical weight index built once at startup from the resident model: the per-chunk
/// blake3 leaves (for Merkle paths), the recomputed tier root `R_T` (sanity-checked against
/// the consensus-pinned value), and a chunk reader. Canonical layout = name-sorted GGUF
/// tensors, `floor(len/32)` 32 B chunks — identical to `pom-rt-builder` and the node.
///
/// The Merkle tree lives on disk (pread); the raw chunks are read on demand from the GGUF
/// (`ChunkSource::Gguf`), so the index holds no full host copy of the weights.
/// Sparse Merkle tree checkpoint interval: only every K-th level is stored on disk (level 0 = leaves
/// is NEVER stored — recomputed from the GGUF on demand; the root is always stored). Cuts tree
/// storage from ~2N nodes to ~N/(2^K − 1) (~63× for K=6). Ported from upstream e1811a0 + d70678a.
const CHECKPOINT_INTERVAL: u32 = 6;

/// One checkpoint level stored on disk in the sparse Merkle tree file.
struct StoredLevel {
    level: u32,  // level index in the full tree (0 = leaves, root = total_levels - 1)
    offset: u64, // byte offset within the checkpoint file
    count: u64,  // node count at this level
}

pub struct WeightIndex {
    pub n_chunks: u64,
    pub r_t: [u8; 32],
    /// Raw 32 B chunk reader: GGUF-backed in production, RAM-backed in synthetic tests.
    chunks: ChunkSource,
    /// Sparse checkpoint file: only stored levels are persisted (pread). Unstored intermediate levels
    /// AND the leaves are recomputed on demand in `merkle_path` from the nearest checkpoint / the GGUF.
    tree_file: File,
    tree_path: PathBuf,
    /// Stored checkpoint levels (multiples of CHECKPOINT_INTERVAL + the root); level 0 never stored.
    checkpoints: Vec<StoredLevel>,
    /// Full tree depth: levels 0..total_levels-1 where total_levels-1 is the root.
    total_levels: u32,
    /// True for the SHARED, cached possession tree (`pom-tree.bin`, one per model dir, reused across
    /// every per-GPU process AND across restarts). NOT deleted on drop — other live workers `pread`
    /// the same inode and the next restart reuses it. A PRIVATE/test tree is `persistent = false`.
    persistent: bool,
    /// Optional in-RAM dense tree (all levels). When present, `merkle_path` is a pure lookup
    /// instead of the sparse recompute — removes the ~30-40 ms proof-build latency after a hit,
    /// which at 10 BPS is a real solo block-race edge (upstream 7a6e7a0). Costs ~2N*32 B of RAM
    /// (~9.6 GB at tier-0's N=150M), so it is OPT-IN via KERYX_RESIDENT_TREE=1.
    dense: Option<Vec<Vec<[u8; 32]>>>,
}

impl Drop for WeightIndex {
    fn drop(&mut self) {
        if !self.persistent {
            let _ = std::fs::remove_file(&self.tree_path);
        }
    }
}

/// On-disk sidecar next to the shared `pom-tree.bin` — the authoritative, builder-written metadata
/// a reusing process needs to reconstruct a byte-identical `WeightIndex` WITHOUT rebuilding. Its
/// presence is the build-completion sentinel. Flattened `(u64,u64)` pairs keep the borsh wire simple.
#[derive(BorshSerialize, BorshDeserialize)]
struct PomTreeMeta {
    version: u32,
    /// The model this tree was built from (upstream e69461d). On reuse we require this to equal the
    /// model the miner is currently serving, so a tree built for a DIFFERENT tier/model — e.g. a
    /// stale or poisoned `pom-tree.bin` served over a SHARED NFS mount — is rebuilt, never trusted.
    model_id: [u8; 32],
    /// GGUF length + mtime — if either differs from the live file, the cache is stale → rebuild.
    gguf_len: u64,
    gguf_mtime: i64,
    n_chunks: u64,
    r_t: [u8; 32],
    total_levels: u32,
    /// SHA-256 of the ENTIRE `pom-tree.bin` bytes (upstream e69461d). Reuse verifies the whole tree,
    /// not just the root, so a corrupted/tampered checkpoint level (correct root, wrong interior) is
    /// caught before we mine on it. Full-file hash = one sequential read of the tree at startup.
    tree_sha256: [u8; 32],
    /// Flattened `ChunkSource::Gguf` table: (first-chunk index, gguf byte offset) pairs.
    table: Vec<u64>,
    /// Flattened sparse `checkpoints`: (level, byte offset, node count) triples.
    checkpoints: Vec<u64>,
}

/// v2 = sparse checkpoint tree (was v1 = dense all-levels tree). v3 (upstream e69461d) adds
/// model-id binding + full-tree SHA-256 to the sidecar — bumping invalidates every legacy
/// unauthenticated `pom-tree.bin`/`.meta` so it is rebuilt with the authenticated sidecar.
const POM_TREE_CACHE_VERSION: u32 = 3;

/// A build lock older than this with no published tree is treated as abandoned (a crashed builder).
const POM_TREE_LOCK_STALE_SECS: u64 = 90 * 60;

fn pom_tree_mtime_secs(md: &std::fs::Metadata) -> i64 {
    md.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn pom_tree_lock_is_stale(lock_path: &std::path::Path) -> bool {
    std::fs::metadata(lock_path)
        .ok()
        .and_then(|md| md.modified().ok())
        .and_then(|t| t.elapsed().ok())
        .map(|e| e > std::time::Duration::from_secs(POM_TREE_LOCK_STALE_SECS))
        .unwrap_or(false)
}

/// Write the meta sentinel (atomically, via a temp + rename) AFTER the tree is fully on disk.
/// `model_id` binds the sidecar to the model the tree was built from (upstream e69461d); the
/// full-tree SHA-256 is computed here over the just-published `idx.tree_path`.
fn write_pom_tree_meta(
    meta_path: &std::path::Path,
    gguf_path: &str,
    model_id: [u8; 32],
    idx: &WeightIndex,
    table: &[(u64, u64)],
) -> std::io::Result<()> {
    let gm = std::fs::metadata(gguf_path)?;
    let tree_sha256 = crate::integrity::sha256_file(&idx.tree_path, |_, _| {})?;
    let mut tflat = Vec::with_capacity(table.len() * 2);
    for &(a, b) in table {
        tflat.push(a);
        tflat.push(b);
    }
    let mut cflat = Vec::with_capacity(idx.checkpoints.len() * 3);
    for cp in &idx.checkpoints {
        cflat.push(cp.level as u64);
        cflat.push(cp.offset);
        cflat.push(cp.count);
    }
    let meta = PomTreeMeta {
        version: POM_TREE_CACHE_VERSION,
        model_id,
        gguf_len: gm.len(),
        gguf_mtime: pom_tree_mtime_secs(&gm),
        n_chunks: idx.n_chunks,
        r_t: idx.r_t,
        total_levels: idx.total_levels,
        tree_sha256,
        table: tflat,
        checkpoints: cflat,
    };
    let bytes = borsh::to_vec(&meta)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let tmp = meta_path.with_extension("meta.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, meta_path)?; // atomic publish
    Ok(())
}

/// Best-effort free bytes on the filesystem holding `dir`. UNIX only, via POSIX `df -kP` (available
/// on HiveOS, SMOS, mmpOS, WSL, plain Linux). `None` → callers SKIP the disk pre-check rather than
/// block a build. Deliberately no libc/statvfs dependency, and NOT attempted on Windows (no `df`
/// there — the native Windows build returns `None` and behaves exactly as before).
#[cfg(unix)]
fn available_disk_bytes(dir: &std::path::Path) -> Option<u64> {
    let out = std::process::Command::new("df").arg("-kP").arg(dir).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // POSIX `-P` = exactly one line per filesystem; column 4 = available 1K-blocks.
    let avail_kb: u64 = text.lines().last()?.split_whitespace().nth(3)?.parse().ok()?;
    Some(avail_kb.saturating_mul(1024))
}
#[cfg(not(unix))]
fn available_disk_bytes(_dir: &std::path::Path) -> Option<u64> {
    None // Windows/other: no `df`; skip the pre-check (a real ENOSPC still surfaces at write time).
}

/// Remove `pom-tree-<pid>.bin` files in `dir` left by miner processes that are no longer running.
/// A clean exit deletes the tree via `Drop`, but a `kill -9` / crash / ENOSPC skips that, leaving
/// ~5 GB orphans that accumulate across restarts. This sweeps only DEAD-pid files (own pid + any
/// live pid are kept), so it is safe to run while other miners share the dir.
fn sweep_dead_pom_trees(dir: &std::path::Path) {
    let me = std::process::id();
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for ent in rd.flatten() {
        let name = ent.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(pid) = name
            .strip_prefix("pom-tree-")
            .and_then(|s| s.strip_suffix(".bin"))
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == me || std::path::Path::new(&format!("/proc/{pid}")).exists() {
            continue; // ours, or still running — leave it
        }
        if std::fs::remove_file(ent.path()).is_ok() {
            log::info!("PoM: swept orphan possession tree {name} (dead pid {pid})");
        }
    }
}

impl WeightIndex {
    /// Build from a GGUF on disk (CPU dtoh of each tensor). The bytes are candle's exact quantized
    /// bytes — the same the miner serves in VRAM and the builder pinned in `R_T`. The Merkle tree
    /// is streamed to a temp file next to the GGUF (disk, never tmpfs) so big tiers don't OOM.
    pub fn build_from_gguf(path: &str, model_id: [u8; 32]) -> candle_core::Result<Self> {
        let dir = std::path::Path::new(path).parent().unwrap_or_else(|| std::path::Path::new("."));
        let cache_path = dir.join("pom-tree.bin");
        let meta_path = dir.join("pom-tree.meta");
        let lock_path = dir.join("pom-tree.lock");
        // Sweep pre-0.6.5.3 per-PID orphans (~5 GB each); the shared cache below has no PID in its name.
        sweep_dead_pom_trees(dir);

        // FAST PATH: a valid shared cache already exists (built by another per-GPU process, or a
        // previous run). Reuse it read-only — ONE physical copy shared via the OS page cache across
        // all GPUs, and NO rebuild on restart. (Was: pom-tree-<PID>.bin = one identical copy + a full
        // rebuild PER process PER restart → N× host RAM, which OOM'd the 4th GPU under WSL's cap.)
        if let Some(idx) = Self::reuse_cached_tree(&cache_path, &meta_path, path, model_id) {
            log::info!("PoM: reusing cached possession tree {} — shared across GPUs, no rebuild.", cache_path.display());
            return Ok(idx);
        }

        // BUILD PATH: exactly ONE process builds; the rest wait. std has no flock, but `create_new`
        // (O_EXCL) is an atomic cross-process lock on every platform incl. WSL. Losers spin on the
        // meta sentinel; a crashed builder's lock goes stale and is stolen; a wedged build (or a
        // read-only dir) falls back to a PRIVATE per-process tree so the miner never hangs.
        use std::time::{Duration, Instant};
        let deadline = Instant::now() + Duration::from_secs(POM_TREE_LOCK_STALE_SECS);
        loop {
            if let Some(idx) = Self::reuse_cached_tree(&cache_path, &meta_path, path, model_id) {
                log::info!("PoM: cached possession tree became available — reusing (no rebuild).");
                return Ok(idx);
            }
            match OpenOptions::new().write(true).create_new(true).open(&lock_path) {
                Ok(mut lock) => {
                    let _ = writeln!(lock, "{}", std::process::id());
                    log::info!("PoM: building shared possession tree (first GPU to reach PoM) — this can take a while…");
                    let tmp = dir.join("pom-tree.bin.building");
                    let out = match Self::build_tree_to(path, tmp.clone(), true) {
                        Ok((mut idx, table)) => match std::fs::rename(&tmp, &cache_path) {
                            Ok(()) => {
                                idx.tree_path = cache_path.clone();
                                if let Err(e) = write_pom_tree_meta(&meta_path, path, model_id, &idx, &table) {
                                    log::warn!("PoM: tree built but meta sidecar write failed ({e}); other GPUs will rebuild.");
                                }
                                Ok(idx)
                            }
                            Err(e) => {
                                // Couldn't publish (e.g. Windows sharing) — keep it as a private tree.
                                log::warn!("PoM: could not publish shared tree ({e}); using a private per-process tree.");
                                idx.persistent = false;
                                Ok(idx)
                            }
                        },
                        Err(e) => Err(e),
                    };
                    let _ = std::fs::remove_file(&lock_path);
                    return out;
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if pom_tree_lock_is_stale(&lock_path) {
                        log::warn!("PoM: stealing stale possession-tree build lock (previous builder crashed).");
                        let _ = std::fs::remove_file(&lock_path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        log::warn!("PoM: shared tree not ready in time — building a private per-process tree.");
                        let tp = dir.join(format!("pom-tree-{}.bin", std::process::id()));
                        return Self::build_tree_to(path, tp, false).map(|(idx, _)| idx);
                    }
                    std::thread::sleep(Duration::from_millis(750));
                }
                Err(e) => {
                    log::warn!("PoM: cannot create tree lock ({e}) — building a private per-process tree.");
                    let tp = dir.join(format!("pom-tree-{}.bin", std::process::id()));
                    return Self::build_tree_to(path, tp, false).map(|(idx, _)| idx);
                }
            }
        }
    }

    /// Build the SPARSE checkpoint possession tree at `tree_path`. Hashes each chunk to a leaf, folds
    /// batches of 2^K leaves up K levels (`fold_levels`) and writes only the level-K node, then builds
    /// the higher checkpoints. Returns the index + a snapshot of the chunk `table` (for the meta).
    fn build_tree_to(path: &str, tree_path: PathBuf, persistent: bool) -> candle_core::Result<(Self, Vec<(u64, u64)>)> {
        let mut file = File::open(path).map_err(candle_core::Error::wrap)?;
        // Read the GGUF header with our own minimal parser (raw offsets/sizes only) rather than
        // candle's QTensor loader: the H4 llama-only archs (Qwen3.5-hybrid-SSM / GLM-4 / EXAONE-4 /
        // Kimi-Linear-MoE) have tensors candle cannot dequantize, so `content.tensor()` would fail
        // and the possession-index build would never complete. The leaf bytes we hash are the raw
        // on-disk quantized bytes either way (`read_chunk` already preads them), so this is
        // byte-identical for candle-clean models and simply also works for the new archs.
        let meta = crate::gguf::GgufMeta::read(&mut file)
            .map_err(|e| candle_core::Error::Msg(format!("PoM: GGUF header parse failed: {e}")))?;
        let names = meta.sorted_names(); // canonical order

        // DISK PRE-CHECK (best-effort). The sparse tree is only ~N/63 of the leaves (≈ gguf/32), so
        // this almost never trips now — but a truly tiny/full disk still gets a clear message instead
        // of an ENOSPC loop. Skipped on Windows / when free space can't be queried.
        let gguf_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let tree_dir = tree_path.parent().unwrap_or_else(|| std::path::Path::new("."));
        if let Some(avail) = available_disk_bytes(tree_dir) {
            let need = (gguf_len / 30).saturating_add(256 << 20); // ~sparse tree + 256 MB headroom
            if avail < need {
                return Err(candle_core::Error::Msg(format!(
                    "not enough free disk to build the PoM possession tree (needs ~{} MB free next to \
                     the model, only ~{} MB available). Free up disk.",
                    need >> 20,
                    avail >> 20,
                )));
            }
        }

        let _ = std::fs::remove_file(&tree_path); // clear a stale/partial file from a crashed run

        // Phase 0: hash chunks → leaves, fold each batch of 2^K leaves up K levels and write ONLY the
        // resulting level-K node. `fold_levels` carries the duplicate-last `hash(x,x)` every round
        // (the d70678a fix), so partial tails match the dense root. The raw chunks are NOT retained;
        // `table` records each tensor's first canonical chunk index + gguf byte offset for on-demand
        // preads (leaf recompute in `merkle_path`).
        let k = CHECKPOINT_INTERVAL;
        let batch_size = 1u64 << k;
        let mut writer = BufWriter::new(
            OpenOptions::new().read(true).write(true).create(true).truncate(true)
                .open(&tree_path).map_err(candle_core::Error::wrap)?,
        );
        let mut table: Vec<(u64, u64)> = Vec::with_capacity(names.len());
        let mut n_chunks: u64 = 0;
        let mut batch_buf: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
        // Stream each tensor's raw on-disk bytes in bounded slabs (the biggest tensors are multi-GB
        // — no full-tensor buffering needed to hash 32 B chunks). Byte-for-byte the same chunks
        // candle's qt.data() yielded, but arch-agnostic (see the GgufMeta note above).
        const SLAB_CHUNKS: u64 = 1 << 16; // 2 MiB per read
        let mut slab = vec![0u8; (SLAB_CHUNKS * 32) as usize];
        // Live progress for the possession-index build (users reported "stuck at verifying/preparing
        // forever" — this multi-minute chunk-hash pass had no output). Total chunks are known from the
        // tensor sizes up front; log %, elapsed and ETA every ~10 s so the rig shows real progress.
        let total_chunks: u64 = names.iter().map(|n| meta.tensors[n].nbytes / 32).sum();
        let idx_build_start = std::time::Instant::now();
        let mut last_idx_log = idx_build_start;
        for name in &names {
            let t = &meta.tensors[name];
            let file_off = meta.tensor_data_offset + t.offset;
            let full = t.nbytes / 32;
            if full > 0 {
                table.push((n_chunks, file_off));
            }
            let mut done: u64 = 0;
            while done < full {
                let take = SLAB_CHUNKS.min(full - done);
                let buf = &mut slab[..(take * 32) as usize];
                read_exact_at(&file, buf, file_off + done * 32).map_err(candle_core::Error::wrap)?;
                for c in 0..take as usize {
                    let chunk = &buf[c * 32..c * 32 + 32];
                    batch_buf.push(blake(chunk));
                    n_chunks += 1;
                    if batch_buf.len() == batch_size as usize {
                        writer.write_all(&fold_levels(&batch_buf, k)).map_err(candle_core::Error::wrap)?;
                        batch_buf.clear();
                    }
                }
                done += take;
                if last_idx_log.elapsed().as_secs() >= 10 {
                    last_idx_log = std::time::Instant::now();
                    let el = idx_build_start.elapsed().as_secs_f64();
                    let rate = n_chunks as f64 / el.max(0.001);
                    let pct = if total_chunks > 0 { n_chunks * 100 / total_chunks } else { 0 };
                    let eta = total_chunks.saturating_sub(n_chunks) as f64 / rate.max(1.0);
                    log::info!(
                        "PoM: building possession index — {}/{} chunks ({}%), {:.0}s elapsed, ETA ~{:.0}s — \
                         mining starts automatically when done, this is NOT a stall.",
                        n_chunks, total_chunks, pct, el, eta,
                    );
                }
            }
        }
        if !batch_buf.is_empty() {
            writer.write_all(&fold_levels(&batch_buf, k)).map_err(candle_core::Error::wrap)?;
        }
        if n_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM: model produced 0 chunks".into()));
        }
        writer.flush().map_err(candle_core::Error::wrap)?;
        drop(writer);

        // Build the higher checkpoint levels (2K, 3K, …, root) from the level-K nodes on disk.
        let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n_chunks)?;

        let table_snapshot = table.clone();
        let gguf = File::open(path).map_err(candle_core::Error::wrap)?;
        let tree_file = File::open(&tree_path).map_err(candle_core::Error::wrap)?;
        let idx = WeightIndex {
            n_chunks,
            r_t,
            chunks: ChunkSource::Gguf { file: gguf, table },
            tree_file,
            tree_path,
            checkpoints,
            total_levels,
            persistent,
            dense: None,
        };
        Ok((idx, table_snapshot))
    }

    /// Reconstruct a byte-identical `WeightIndex` from an existing shared cache + meta sidecar, or
    /// `None` if the cache is absent/stale/corrupt (→ caller rebuilds). Validated FOUR ways: cache
    /// version, GGUF length+mtime, tree-file size, and the on-disk root hash == the meta's `r_t`.
    fn reuse_cached_tree(cache_path: &std::path::Path, meta_path: &std::path::Path, gguf_path: &str, expected_model_id: [u8; 32]) -> Option<Self> {
        let bytes = std::fs::read(meta_path).ok()?;
        let meta = PomTreeMeta::try_from_slice(&bytes).ok()?;
        if meta.version != POM_TREE_CACHE_VERSION {
            return None;
        }
        // Upstream e69461d: the sidecar must name the SAME model we are serving. Over a shared NFS
        // mount a `pom-tree.bin` built for another tier (or swapped in) would otherwise be trusted
        // solely on gguf len/mtime; the model-id binding rebuilds instead.
        if meta.model_id != expected_model_id {
            log::warn!("PoM: cached tree was built for a different model — rebuilding.");
            return None;
        }
        let gm = std::fs::metadata(gguf_path).ok()?;
        if gm.len() != meta.gguf_len || pom_tree_mtime_secs(&gm) != meta.gguf_mtime {
            return None; // model changed under the cache → rebuild
        }
        if meta.table.len() % 2 != 0 || meta.checkpoints.len() % 3 != 0 || meta.checkpoints.is_empty() {
            return None;
        }
        let table: Vec<(u64, u64)> = meta.table.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        let checkpoints: Vec<StoredLevel> = meta
            .checkpoints
            .chunks_exact(3)
            .map(|c| StoredLevel { level: c[0] as u32, offset: c[1], count: c[2] })
            .collect();
        let root = checkpoints.last()?; // root checkpoint (count must be 1)
        if root.count != 1 {
            return None;
        }
        let cm = std::fs::metadata(cache_path).ok()?;
        if cm.len() < root.offset + 32 {
            return None; // truncated/incomplete tree
        }
        let tree_file = File::open(cache_path).ok()?;
        let mut root_hash = [0u8; 32];
        read_exact_at(&tree_file, &mut root_hash, root.offset).ok()?;
        if root_hash != meta.r_t {
            return None; // integrity mismatch — never mine on a corrupt tree
        }
        // Upstream e69461d: authenticate the ENTIRE tree, not just the root — a tampered interior
        // checkpoint level (valid root, wrong nodes) would otherwise produce bad Merkle paths. One
        // sequential read of the tree at startup; over NFS this also detects a torn/partial mirror.
        match crate::integrity::sha256_file(cache_path, |_, _| {}) {
            Ok(digest) if digest == meta.tree_sha256 => {}
            Ok(_) => {
                log::warn!("PoM: cached tree SHA-256 mismatch — rebuilding.");
                return None;
            }
            Err(e) => {
                log::warn!("PoM: could not hash cached tree ({e}) — rebuilding.");
                return None;
            }
        }
        let gguf = File::open(gguf_path).ok()?;
        Some(WeightIndex {
            n_chunks: meta.n_chunks,
            r_t: meta.r_t,
            chunks: ChunkSource::Gguf { file: gguf, table },
            tree_file,
            tree_path: cache_path.to_path_buf(),
            checkpoints,
            total_levels: meta.total_levels,
            persistent: true,
            dense: None,
        })
    }

    /// 32 B chunk at canonical index `off` (panics if out of range — `off < n_chunks`).
    pub fn read_chunk(&self, off: u64) -> [u64; CHUNK_WORDS] {
        chunk_to_words(&self.read_chunk_bytes(off))
    }

    /// Raw 32 B chunk bytes — used for leaf recompute in `merkle_path`.
    pub(crate) fn read_chunk_bytes(&self, off: u64) -> [u8; 32] {
        let mut arr = [0u8; 32];
        match &self.chunks {
            #[cfg(test)]
            ChunkSource::Ram(data) => {
                let base = (off as usize) * 32;
                arr.copy_from_slice(&data[base..base + 32]);
            }
            ChunkSource::Gguf { file, table } => {
                // Tensor whose canonical range contains `off`: last entry with start <= off.
                let j = table.partition_point(|&(start, _)| start <= off) - 1;
                let (start, file_off) = table[j];
                read_exact_at(file, &mut arr, file_off + (off - start) * 32).expect("PoM gguf chunk read");
            }
        }
        arr
    }

    /// Bulk chunk reader: fill `out` (length a multiple of 32) with the raw bytes of canonical
    /// chunks `[first_chunk, first_chunk + out.len()/32)`. Byte-identical to concatenating
    /// `read_chunk_bytes` over the range, but issues one pread per overlapped TENSOR instead of
    /// one per 32 B chunk — the GPU drivers stream the tier blob to VRAM through a bounded
    /// window with this (a full per-chunk loop is ~78M preads for Gemma).
    pub fn read_chunks_into(&self, first_chunk: u64, out: &mut [u8]) {
        assert!(out.len() % 32 == 0, "read_chunks_into: length must be whole chunks");
        let count = (out.len() / 32) as u64;
        assert!(first_chunk + count <= self.n_chunks, "read_chunks_into: range past n_chunks");
        match &self.chunks {
            #[cfg(test)]
            ChunkSource::Ram(data) => {
                let base = (first_chunk as usize) * 32;
                out.copy_from_slice(&data[base..base + out.len()]);
            }
            ChunkSource::Gguf { file, table } => {
                let mut off = first_chunk;
                let mut filled = 0usize;
                // Tensor whose canonical range contains `off` (same lookup as read_chunk_bytes);
                // subsequent tensors are consecutive table entries.
                let mut j = table.partition_point(|&(start, _)| start <= off) - 1;
                while filled < out.len() {
                    let (start, file_off) = table[j];
                    let tensor_end = table.get(j + 1).map_or(self.n_chunks, |&(s, _)| s);
                    let n = (tensor_end - off).min(((out.len() - filled) / 32) as u64);
                    let bytes = (n * 32) as usize;
                    read_exact_at(file, &mut out[filled..filled + bytes], file_off + (off - start) * 32)
                        .expect("PoM gguf chunk read");
                    filled += bytes;
                    off += n;
                    j += 1;
                }
            }
        }
    }

    /// Find the stored checkpoint at `level` (panics if not found).
    fn find_checkpoint(&self, level: u32) -> &StoredLevel {
        self.checkpoints.iter().find(|cp| cp.level == level).expect("PoM: checkpoint not found")
    }

    /// Number of nodes at `level` in the full tree (0-indexed, level 0 = leaves).
    fn count_at_level(&self, level: u32) -> u64 {
        let mut count = self.n_chunks;
        for _ in 0..level {
            count = count.div_ceil(2);
        }
        count
    }

    /// Hash of the subtree rooted `log2(span)` levels above `src_level`, at source index `start`,
    /// covering `span` source nodes (span is a power of two). Reads ONLY the in-range source nodes
    /// (partial subtree only at the right edge) and folds them EXACTLY `log2(span)` levels with
    /// per-level duplicate-last — matching the dense `hash(x, x)` carry of a lone inner node.
    fn compute_subtree_hash(&self, start: u64, span: u64, src_level: u32) -> [u8; 32] {
        debug_assert!(span.is_power_of_two());
        let rounds = span.trailing_zeros();
        let source_count = if src_level == 0 { self.n_chunks } else { self.find_checkpoint(src_level).count };
        if start >= source_count {
            return [0u8; 32]; // a real sibling subtree always starts in range
        }
        let end = (start + span).min(source_count);
        let nodes: Vec<[u8; 32]> = if src_level == 0 {
            (start..end).map(|i| blake(&self.read_chunk_bytes(i))).collect()
        } else {
            let cp = self.find_checkpoint(src_level);
            (start..end)
                .map(|i| {
                    let mut buf = [0u8; 32];
                    read_exact_at(&self.tree_file, &mut buf, cp.offset + i * 32).expect("PoM checkpoint read subtree");
                    buf
                })
                .collect()
        };
        fold_levels(&nodes, rounds)
    }

    /// Build the in-RAM dense tree; afterwards `merkle_path` is a pure lookup. Reads every chunk
    /// once (sequential, page-cache friendly). Ported from upstream 7a6e7a0.
    pub fn build_dense(&mut self) {
        if self.dense.is_some() {
            return;
        }
        let mut levels: Vec<Vec<[u8; 32]>> = vec![(0..self.n_chunks).map(|i| blake(&self.read_chunk_bytes(i))).collect()];
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity(cur.len().div_ceil(2));
            let mut i = 0;
            while i < cur.len() {
                let r = if i + 1 < cur.len() { cur[i + 1] } else { cur[i] };
                next.push(hash_pair(&cur[i], &r));
                i += 2;
            }
            levels.push(next);
        }
        self.dense = Some(levels);
    }

    /// Inclusion path for chunk index `off`: stored siblings read from the checkpoint file, unstored
    /// intermediate levels recomputed on the fly from the nearest checkpoint / the GGUF leaves.
    /// Byte-identical to the dense full-tree path: an out-of-range sibling is the node itself.
    pub fn merkle_path(&self, off: u64) -> Vec<[u8; 32]> {
        if let Some(dense) = &self.dense {
            let mut path = Vec::with_capacity(dense.len().saturating_sub(1));
            let mut idx = off as usize;
            for level in &dense[..dense.len() - 1] {
                let sib = idx ^ 1;
                path.push(if sib < level.len() { level[sib] } else { level[idx] });
                idx >>= 1;
            }
            return path;
        }
        let total_levels = self.total_levels;
        let mut path = Vec::with_capacity(total_levels as usize);
        let mut idx: u64 = off;
        for level in 0..total_levels {
            if level == total_levels - 1 {
                break; // root has no sibling
            }
            let sib_idx = idx ^ 1;
            let is_stored = level > 0 && (level % CHECKPOINT_INTERVAL == 0 || level == total_levels - 1);
            let node = if is_stored {
                let cp = self.find_checkpoint(level);
                let real_idx = if sib_idx < cp.count { sib_idx } else { idx };
                let mut buf = [0u8; 32];
                read_exact_at(&self.tree_file, &mut buf, cp.offset + real_idx * 32).expect("PoM checkpoint read");
                buf
            } else {
                let node_count = self.count_at_level(level);
                let real_sib_idx = if sib_idx < node_count { sib_idx } else { idx };
                let src_level = (level / CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL;
                let span = 1u64 << (level - src_level);
                self.compute_subtree_hash(real_sib_idx * span, span, src_level)
            };
            path.push(node);
            idx >>= 1;
        }
        path
    }
}

/// Compute checkpoint levels from the leaf count alone — purely arithmetic, no I/O. Returns
/// (checkpoints, total_levels). Stores only multiples of CHECKPOINT_INTERVAL + the root; level 0
/// (leaves) is never stored (recomputed from the GGUF on demand).
fn compute_checkpoint_offsets(n_chunks: u64) -> (Vec<StoredLevel>, u32) {
    let mut checkpoints = Vec::new();
    let mut count = n_chunks;
    let mut off: u64 = 0;
    let mut level: u32 = 0;
    loop {
        let is_checkpoint = (level > 0 && level % CHECKPOINT_INTERVAL == 0) || count == 1;
        if is_checkpoint {
            checkpoints.push(StoredLevel { level, offset: off, count });
        }
        if count == 1 {
            break;
        }
        if is_checkpoint {
            off += count * 32;
        }
        count = count.div_ceil(2);
        level += 1;
    }
    (checkpoints, level + 1)
}

/// Reduce `batch` by EXACTLY `rounds` canonical levels — duplicate-last each round, AND keep carrying
/// a lone node via `hash(x, x)` once the batch collapses to one node before `rounds` is reached. THE
/// d70678a FIX: `merkle_root_mini` stops at len==1, so a partial batch dropped the remaining carries
/// → wrong checkpoint node (wrong R_T) for every non-power-of-two N. A batch fold must always land
/// exactly `rounds` levels up.
#[inline]
fn fold_levels(batch: &[[u8; 32]], rounds: u32) -> [u8; 32] {
    debug_assert!(!batch.is_empty());
    let mut level = batch.to_vec();
    for _ in 0..rounds {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

/// Build higher checkpoint levels (2K, 3K, …, root) from the already-written level-K nodes in the
/// tree file. Returns (checkpoints, total_levels, R_T).
fn finalize_checkpoint_upper(
    tree_path: &std::path::Path,
    n_chunks: u64,
) -> candle_core::Result<(Vec<StoredLevel>, u32, [u8; 32])> {
    let (checkpoints, total_levels) = compute_checkpoint_offsets(n_chunks);
    let mut file_for_read = File::open(tree_path).map_err(candle_core::Error::wrap)?;
    let mut prev_offset: u64 = checkpoints[0].offset;
    let mut prev_count = checkpoints[0].count;
    let mut prev_level = checkpoints[0].level;

    let mut writer = OpenOptions::new().read(true).write(true).open(tree_path).map_err(candle_core::Error::wrap)?;
    writer.seek(SeekFrom::End(0)).map_err(candle_core::Error::wrap)?;
    let mut buf_writer = BufWriter::new(writer);

    for cp in &checkpoints[1..] {
        // Fold the previous stored level up to this checkpoint's level (K levels; the final root fold
        // may span fewer). Batch by 2^rounds and fold each batch EXACTLY `rounds` levels so a partial
        // tail carries via hash(x,x) like the dense tree; per-level counts (ceil) line up the offsets.
        let rounds = cp.level - prev_level;
        let batch_size = 1u64 << rounds;
        let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
        let mut read_idx: u64 = 0;
        while read_idx < prev_count {
            let take = batch_size.min(prev_count - read_idx);
            batch.clear();
            for i in 0..take {
                let index = read_idx + i;
                let mut node = [0u8; 32];
                read_exact_at(&file_for_read, &mut node, prev_offset + index * 32).map_err(candle_core::Error::wrap)?;
                batch.push(node);
            }
            let parent_node = fold_levels(&batch, rounds);
            buf_writer.write_all(&parent_node).map_err(candle_core::Error::wrap)?;
            read_idx += take;
        }
        buf_writer.flush().map_err(candle_core::Error::wrap)?;
        file_for_read = File::open(tree_path).map_err(candle_core::Error::wrap)?;
        prev_offset = cp.offset;
        prev_count = cp.count;
        prev_level = cp.level;
    }

    let root_cp = checkpoints.last().unwrap();
    let mut r_t = [0u8; 32];
    read_exact_at(&file_for_read, &mut r_t, root_cp.offset).map_err(candle_core::Error::wrap)?;
    Ok((checkpoints, total_levels, r_t))
}

/// Reduce a slice of leaves straight to the single canonical root (duplicate-last each level). This
/// is what the node pins in `POM_TIERS`; used ONLY as the independent dense oracle in tests. NOT safe
/// for batched sub-folds (stops at one node — the e1811a0 bug), which is why the build uses `fold_levels`.
#[cfg(test)]
#[inline]
fn merkle_root_mini(leaves: &[[u8; 32]]) -> [u8; 32] {
    debug_assert!(!leaves.is_empty());
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

/// PoM possession activation DAA score — MUST match the node's `pom_activation`.
/// `u64::MAX` = never (dormant): mining stays on legacy kHeavyHash, no proof produced.
///
/// Testnet: `5_000` = mid-chain activation, to observe the kHeavyHash→PoM transition (incl.
/// the difficulty drift: PoM ~30x slower → blocks slow at the cutover, then the DAA window
/// recovers). Mainnet will need a difficulty reset at H.
/// Mainnet: 37_780_000 (2026-06-26 18:00 UTC) — MUST equal the node's
/// MAINNET_PARAMS.pom_activation = new(37_780_000).
pub const POM_ACTIVATION_DAA: u64 = 37_780_000;

/// Effective PoM activation DAA. Defaults to the consensus `POM_ACTIVATION_DAA`. Overridable via
/// the `KERYX_POM_ACTIVATION_DAA` env var for STAGING / pre-fork live-path testing only (e.g. set
/// to 0 to force PoM on regardless of the job's daa_score). Read once. `is_activation_overridden`
/// lets startup warn loudly so an override can never be used silently in production.
pub fn activation_daa() -> u64 {
    *ACTIVATION_DAA.get_or_init(|| {
        std::env::var("KERYX_POM_ACTIVATION_DAA")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(POM_ACTIVATION_DAA)
    })
}
pub fn is_activation_overridden() -> bool {
    activation_daa() != POM_ACTIVATION_DAA
}
static ACTIVATION_DAA: OnceLock<u64> = OnceLock::new();

/// H3 (PoM block-level) hardfork activation DAA. At/after this score the PoM folds are salted
/// with `POM_H3_PPH_SALT` (forced update) and the block header commits the winning walk's
/// `final_state` (`pomFinalState`) — for our pool path the pool fills that header field from the
/// proof's `final_state`, which we already carry. A post-H3 block whose proof was built with the
/// pre-H3 (unsalted) folds verifies false → rejected. MUST equal the node's
/// MAINNET_PARAMS.pom_level_activation = new(43_450_000) and the official miner's
/// POM_LEVEL_ACTIVATION_DAA. Mainnet: 43_450_000 (~2026-07-05 18:00 UTC). Testnet: 2_000.
pub const POM_LEVEL_ACTIVATION_DAA: u64 = 43_450_000;

/// Effective H3 activation DAA. Overridable via `KERYX_POM_LEVEL_ACTIVATION_DAA` for STAGING /
/// pre-gate testing only (e.g. set to 0 to force the H3-salted folds on regardless of daa_score),
/// mirroring `activation_daa()`. Read once. Never set in production.
pub fn level_activation_daa() -> u64 {
    *LEVEL_ACTIVATION_DAA.get_or_init(|| {
        std::env::var("KERYX_POM_LEVEL_ACTIVATION_DAA")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(POM_LEVEL_ACTIVATION_DAA)
    })
}
pub fn is_level_activation_overridden() -> bool {
    level_activation_daa() != POM_LEVEL_ACTIVATION_DAA
}
static LEVEL_ACTIVATION_DAA: OnceLock<u64> = OnceLock::new();

/// H4 hardfork activation DAA — "coin-age verification" + the PoM verifier v2 (recompute-from-chunks
/// proof). At/after this score: (1) the H4 model lineup is active (`models::pom_tier_index` returns the
/// EXAONE/Mistral/GLM-4/Qwen3.6/Kimi tiers), and (2) the miner MUST build the recompute-from-chunks
/// proof v2 (`build_proof_v2`: every K chunk the walk read, each Merkle-proven under R_T) — a pre-H4
/// proof verifies false at/after the gate → rejected. MUST equal the node's H4 params + upstream
/// keryx-miner v0.3.7's `COIN_AGE_VERIFICATION_ACTIVATION_DAA`. Mainnet: 54_766_000
/// (~2026-07-18 20:31 UTC). Testnet builds: 3_000.
pub const COIN_AGE_VERIFICATION_ACTIVATION_DAA: u64 = 54_766_000;

/// H5 activation DAA score. At/after this score the possession walk switches from the frozen v1
/// XOR-fold (`transition_v1`) to the non-foldable mix64-chained `transition_v2`, both on the GPU
/// kernel (`pom_mine.cu`, `walk_v2` param) and the CPU walk/proof path — closing the pre-H5 fold
/// shortcut. MUST equal the node's `MAINNET_PARAMS.h5_activation` (= node `H5_ACTIVATION_DAA`),
/// node↔miner lockstep exactly like `COIN_AGE_VERIFICATION_ACTIVATION_DAA`. Mainnet: 59_009_037
/// (upstream keryx-miner v0.3.8-OPoI / keryx-node v1.3.4+, the H5 relaunch tip with a difficulty
/// reset). Pair with node v1.3.4+.
pub const H5_ACTIVATION_DAA: u64 = 59_009_037;

/// Effective H5 activation DAA. Overridable via `KERYX_H5_ACTIVATION_DAA` for STAGING / crossing
/// tests only (e.g. set it just above the current tip to exercise the walk_v2 + Qwen3-8B tier-0
/// crossing before mainnet reaches 59_009_037). Read once. Production (unset) = the const above.
pub fn h5_activation_daa() -> u64 {
    *H5_ACTIVATION_DAA_CELL.get_or_init(|| {
        std::env::var("KERYX_H5_ACTIVATION_DAA")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(H5_ACTIVATION_DAA)
    })
}
static H5_ACTIVATION_DAA_CELL: OnceLock<u64> = OnceLock::new();

/// H5.1 (emergency relaunch 2026-07-24) activation DAA score. At/after this score the walk seed
/// derives from the H5.1-salted pph words (`POM_H5_1_PPH_SALT`) — SEED fold only, the pow fold keeps
/// the H3 salt. Gate = virtual daa of the isolated relaunch base. MUST equal the node's
/// `MAINNET_PARAMS.h5_1_activation` / `H5_1_ACTIVATION_DAA` = 59_027_921 (keryx-node v1.3.41 /
/// keryx-miner v0.3.81). Overridable via `KERYX_H5_1_ACTIVATION_DAA` for staging/crossing tests.
pub const H5_1_ACTIVATION_DAA: u64 = 59_027_921;
pub fn h5_1_activation_daa() -> u64 {
    *H5_1_ACTIVATION_DAA_CELL.get_or_init(|| {
        std::env::var("KERYX_H5_1_ACTIVATION_DAA")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(H5_1_ACTIVATION_DAA)
    })
}
static H5_1_ACTIVATION_DAA_CELL: OnceLock<u64> = OnceLock::new();

/// H5.2 (chain anchoring, keryxd v1.4.0, 2026-07-25) activation DAA score. At/after this score the
/// walk seed derives from the H5.2-salted pph words (`POM_H5_2_PPH_SALT`) — SEED fold only, the pow
/// fold keeps the H3 salt. MUST equal the node's `H5_2_ACTIVATION_DAA` = 59_170_000 (keryx-node
/// v1.4.0 config/params.rs). Overridable via `KERYX_H5_2_ACTIVATION_DAA` for staging/crossing tests.
pub const H5_2_ACTIVATION_DAA: u64 = 59_170_000;
pub fn h5_2_activation_daa() -> u64 {
    *H5_2_ACTIVATION_DAA_CELL.get_or_init(|| {
        std::env::var("KERYX_H5_2_ACTIVATION_DAA")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(H5_2_ACTIVATION_DAA)
    })
}
static H5_2_ACTIVATION_DAA_CELL: OnceLock<u64> = OnceLock::new();

/// H6 (matrix-state walk, PoM v3) activation DAA score. At/after this score — decided per job from
/// the TEMPLATE's own `daa_score`, never wall clock or tip — the miner grinds the v3 walk on the GPU
/// and builds `PomProofV3`; a pre-H6 (trace/steps_v2) proof verifies false at/after the gate and is
/// rejected. H6 is being rolled out NOW, so this build treats H6 as ACTIVE IMMEDIATELY — the gate is
/// 0, i.e. the miner grinds the v3 walk for every job regardless of daa_score (a live miner only ever
/// works the current tip, which is at/after the fork). Overridable via `KERYX_POM_V3_ACTIVATION_DAA`
/// (e.g. a specific fork DAA, or 1000 for a testnet gate). A finite (!= u64::MAX) value is also the
/// codebase-wide "H6 armed" sentinel that enables the H6-lineup model staging (models::h6_staged etc.).
pub const POM_V3_ACTIVATION_DAA: u64 = 0;
pub fn pom_v3_activation_daa() -> u64 {
    *POM_V3_ACTIVATION_DAA_CELL.get_or_init(|| {
        std::env::var("KERYX_POM_V3_ACTIVATION_DAA")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(POM_V3_ACTIVATION_DAA)
    })
}
static POM_V3_ACTIVATION_DAA_CELL: OnceLock<u64> = OnceLock::new();

/// H8 request-identity gate. At/after this score a request is identified by the transaction id of
/// the AiRequest, not by the digest of its payload. MUST equal the node's `reward_routing_activation`:
/// a miner deriving the other identity signs responses the node cannot credit and is struck for work
/// it actually did. Mainnet default 79_251_000; override for testnet via env.
/// (Pool miners are unaffected — the pool derives the identity and relays reqId; this is the SOLO
/// keryxd path, kept in parity with upstream keryx-miner v0.4.9 commit 54129d80.)
pub fn reward_routing_activation_daa() -> u64 {
    static CELL: OnceLock<u64> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KERYX_REWARD_ROUTING_ACTIVATION_DAA")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(79_251_000)
    })
}

/// AUTO-SWITCH: is this block at/after the H6 (PoM v3 matrix-walk) gate? Decided per job from the
/// block's own `daa_score` so an already-running miner flips to the v3 walk + witness on the first
/// post-gate job. Logs ONCE on first crossing.
pub fn pom_v3_active(daa_score: u64) -> bool {
    let active = daa_score >= pom_v3_activation_daa();
    if active {
        static ANNOUNCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ANNOUNCED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log::info!(
                "H6 hardfork ACTIVE (block DAA {} >= gate {}): PoM is now the int8 matrix-state walk \
                 (v3). Building PomProofV3 witnesses.",
                daa_score,
                pom_v3_activation_daa()
            );
        }
    }
    active
}

/// Effective H4 activation DAA. Overridable via `KERYX_H4_ACTIVATION_DAA` for STAGING / pre-gate
/// testing only (e.g. set to 0 to force H4 proof v2 + lineup on regardless of daa_score). Read once.
pub fn h4_activation_daa() -> u64 {
    *H4_ACTIVATION_DAA.get_or_init(|| {
        std::env::var("KERYX_H4_ACTIVATION_DAA")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(COIN_AGE_VERIFICATION_ACTIVATION_DAA)
    })
}
static H4_ACTIVATION_DAA: OnceLock<u64> = OnceLock::new();

/// AUTO-SWITCH: is this block at/after the H3 gate? Decided per job from the block's `daa_score`,
/// so an already-running miner flips to the post-fork PoM convention (H3-salted folds) on the first
/// job past the gate — no restart, no model reload (the salt changes only the seed/pow folds, not
/// the weights/tier/R_T). Logs ONCE the moment it first crosses so the switch is visible. This is
/// what lets miners update now and just leave it running across the fork.
pub fn h3_active(daa_score: u64) -> bool {
    let active = daa_score >= level_activation_daa();
    if active {
        static ANNOUNCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ANNOUNCED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log::info!(
                "H3 hardfork ACTIVE (block DAA {} >= gate {}): auto-switched to the post-fork PoM \
                 convention (H3-salted folds). No action needed — keep mining.",
                daa_score,
                level_activation_daa()
            );
        }
    }
    active
}

/// PoM PASSTHROUGH live-test mode (`KERYX_POM_PASSTHROUGH=1`). When set, the miner keeps mining
/// kHeavyHash (the only valid PoW pre-fork) but ALSO attaches a `PomProof` to each winning share so
/// the wire envelope — stratum 6th param → pool passthrough → daemon `RpcRawBlock.body.pom_proof` —
/// can be exercised before the fork. The proof's `pom_pow_value` need NOT meet target here (the
/// nonce came from kHeavyHash search); pre-fork the daemon stores it without verifying. Read once.
/// Production default (unset) is unchanged. Requires the host possession index to be built.
pub fn passthrough_enabled() -> bool {
    *PASSTHROUGH.get_or_init(|| {
        std::env::var("KERYX_POM_PASSTHROUGH")
            .ok()
            .map(|s| matches!(s.trim(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    })
}
static PASSTHROUGH: OnceLock<bool> = OnceLock::new();

/// The resident tier weight index + tier id, installed once at startup when PoM is enabled.
static POM_INDEX: OnceLock<(WeightIndex, u8)> = OnceLock::new();

/// Install the possession index (built from the resident model) and its tier. Call once.
pub fn set_index(index: WeightIndex, tier: u8) {
    let _ = POM_INDEX.set((index, tier));
}

/// The active possession index + tier, if installed. This is the PROCESS-WIDE SHARED index used by
/// single-model rigs (every device mines the same model). Mixed-rig per-card models use the
/// per-device variants below; those fall back here when a device has no override.
pub fn active_index() -> Option<&'static (WeightIndex, u8)> {
    POM_INDEX.get()
}

/// Per-CUDA-device possession index (mixed-rig per-card models). Entries are `Box::leak`'d →
/// `&'static`, matching the OnceLock "lives forever" semantics. Empty on single-model rigs.
fn pom_indices() -> &'static std::sync::Mutex<std::collections::HashMap<u32, &'static (WeightIndex, u8)>> {
    static POM_INDICES: OnceLock<std::sync::Mutex<std::collections::HashMap<u32, &'static (WeightIndex, u8)>>> =
        OnceLock::new();
    POM_INDICES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Install a possession index for one device (per-card model). Leaked to `&'static` (models live
/// for the whole process, same as the global `POM_INDEX`).
pub fn set_index_for(device_id: u32, index: WeightIndex, tier: u8) {
    let leaked: &'static (WeightIndex, u8) = Box::leak(Box::new((index, tier)));
    if let Ok(mut m) = pom_indices().lock() {
        m.insert(device_id, leaked);
    }
}

/// This device's possession index + tier: its per-device entry if set, else the shared global one.
pub fn active_index_for(device_id: u32) -> Option<&'static (WeightIndex, u8)> {
    if let Ok(m) = pom_indices().lock() {
        if let Some(v) = m.get(&device_id) {
            return Some(*v);
        }
    }
    active_index()
}

/// Whether this device has its own per-device index installed (vs falling back to the global one).
pub fn has_device_index(device_id: u32) -> bool {
    pom_indices().lock().map(|m| m.contains_key(&device_id)).unwrap_or(false)
}

/// Test-only WeightIndex over arbitrary RAM chunks (`data` = chunk-aligned canonical bytes) — real
/// checkpoint tree + merkle paths, no GGUF. Shared by the pom.rs synth tests AND the pom_v3 mirror
/// test (which pins the node's R_T + walk vectors over a synthetic blob).
#[cfg(test)]
pub(crate) fn index_from_ram(data: Vec<u8>) -> WeightIndex {
    use std::sync::atomic::{AtomicU64, Ordering as O};
    static UNIQ: AtomicU64 = AtomicU64::new(0);
    let uid = UNIQ.fetch_add(1, O::Relaxed);
    let tree_path = std::env::temp_dir().join(format!("keryx-pom-synth-{}-{}.bin", std::process::id(), uid));
    let _ = std::fs::remove_file(&tree_path);

    let n = (data.len() / 32) as u64;
    let k = CHECKPOINT_INTERVAL;
    let batch_size = 1u64 << k; // 64 for K=6

    let mut writer = BufWriter::new(
        OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tree_path).unwrap(),
    );
    let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
    for o in 0..n as usize {
        batch.push(blake(&data[o * 32..o * 32 + 32]));
        if batch.len() == batch_size as usize {
            writer.write_all(&fold_levels(&batch, k)).unwrap();
            batch.clear();
        }
    }
    if !batch.is_empty() {
        writer.write_all(&fold_levels(&batch, k)).unwrap();
    }
    writer.flush().unwrap();
    drop(writer);

    let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n).unwrap();
    let tree_file = File::open(&tree_path).unwrap();
    WeightIndex {
        n_chunks: n,
        r_t,
        chunks: ChunkSource::Ram(data),
        tree_file,
        tree_path,
        checkpoints,
        total_levels,
        persistent: false,
        dense: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_chunk(off: u64) -> [u64; CHUNK_WORDS] {
        let mut c = [0u64; CHUNK_WORDS];
        for (j, w) in c.iter_mut().enumerate() {
            *w = mix64(off.wrapping_mul(CHUNK_WORDS as u64) + j as u64 + 1);
        }
        c
    }

    // Synthetic WeightIndex (no GGUF) — exercises the real read_chunk + O(log N) merkle_path.
    fn synth_index(n: u64) -> WeightIndex {
        use std::sync::atomic::{AtomicU64, Ordering as O};
        static UNIQ: AtomicU64 = AtomicU64::new(0);
        let uid = UNIQ.fetch_add(1, O::Relaxed);
        let tree_path = std::env::temp_dir().join(format!("keryx-pom-synth-{}-{}.bin", std::process::id(), uid));
        let _ = std::fs::remove_file(&tree_path);

        let k = CHECKPOINT_INTERVAL;
        let batch_size = 1u64 << k;
        let mut writer = BufWriter::new(
            OpenOptions::new().read(true).write(true).create(true).truncate(true)
                .open(&tree_path).unwrap(),
        );
        let mut data = Vec::new();
        let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
        for o in 0..n {
            let b = words_to_bytes(&synth_chunk(o));
            data.extend_from_slice(&b);
            batch.push(blake(&b));
            if batch.len() == batch_size as usize {
                writer.write_all(&fold_levels(&batch, k)).unwrap();
                batch.clear();
            }
        }
        if !batch.is_empty() {
            writer.write_all(&fold_levels(&batch, k)).unwrap();
        }
        writer.flush().unwrap();
        drop(writer);

        let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n).unwrap();
        let tree_file = File::open(&tree_path).unwrap();
        WeightIndex {
            n_chunks: n,
            r_t,
            chunks: ChunkSource::Ram(data),
            tree_file,
            tree_path,
            checkpoints,
            total_levels,
            persistent: false,
            dense: None,
        }
    }

    /// BYTE-EXACT GATE (consensus): the sparse checkpoint-built root MUST equal the dense canonical
    /// root (`merkle_root_mini` over all leaves = what the node pins in POM_TIERS) for every N,
    /// including the non-power-of-two sizes whose short tails used to drop the hash(x,x) carries.
    #[test]
    fn dense_merkle_path_matches_sparse() {
        for n in [64u64, 65, 100, 1000, 2000, 4096, 4968, 12345, 65536, 100000, 131072] {
            let mut idx = synth_index(n);
            let step = (n as usize / 37).max(1);
            let offs: Vec<u64> = (0..n).step_by(step).collect();
            let sparse: Vec<Vec<[u8; 32]>> = offs.iter().map(|&o| idx.merkle_path(o)).collect();
            idx.build_dense();
            for (k, &o) in offs.iter().enumerate() {
                assert_eq!(idx.merkle_path(o), sparse[k], "path mismatch n={n} off={o}");
            }
            let dense = idx.dense.as_ref().unwrap();
            assert_eq!(dense.last().unwrap()[0], idx.r_t, "dense root != r_t, n={n}");
        }
    }

    #[test]
    fn sparse_build_root_matches_dense_root() {
        for n in [64u64, 65, 100, 1000, 2000, 4096, 4968, 12345, 65536, 100000, 131072] {
            let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
            let dense = merkle_root_mini(&leaves);
            let idx = synth_index(n);
            assert_eq!(idx.r_t, dense, "sparse-built R_T != dense root for N={n}");
            let _ = std::fs::remove_file(&idx.tree_path);
        }
    }

    /// H4 v2 proof (recompute-from-chunks) builds, self-verifies, and wire round-trips — for BOTH
    /// walk eras (v1 fold in [H4,H5), v2 mix64-chain at/after H5), and the cross-era boundary holds.
    #[test]
    /// A pre-H4 proof MUST wire-encode byte-identically to the 7-field `PomProofPreH4` layout —
    /// the invariant that keeps the currently-running (pre-H4) node accepting new-miner blocks.
    #[test]
    /// GGUF-backed `read_chunk`: lay the canonical chunks across 3 "tensors" with header + inter-
    /// tensor padding (so file offset != off*32), build the per-tensor offset table, and assert
    /// `read_chunk` (pread) returns the exact canonical chunks AND that a proof verifies — same as
    /// the RAM path, with no host copy of the weights.
    #[test]
    /// H3 hardfork salt: the pph words feeding BOTH PoM folds are XOR-salted with POM_H3_PPH_SALT
    /// at/after the gate. Proves (a) the salt equals the node's sha256 derivation, (b) it changes
    /// the walk seed + pow value, and (c) proofs are era-bound — an H3 proof verifies under h3=true
    /// and is REJECTED under h3=false. That rejection IS the forced-update guarantee: a pre-H3
    /// binary's proof verifies false post-gate.
    #[test]
    /// Real-GGUF byte-identity: build the index from a downloaded model and prove that chunks
    /// read by `pread` (GGUF) verify against the model's own `R_T` (whose leaves were hashed from
    /// candle's `qt.data()`). Confirms `pread(tensor_data_offset + offset)` == `qt.data()` for real
    /// quant types. Ignored (needs the GGUF); run: `cargo test -p keryx-miner -- --ignored gguf_real`.
    #[test]
    #[ignore]
    #[test]
    fn weight_index_root_matches_standalone() {
        // The prebuilt-tree root equals the standalone merkle_root over the same leaves.
        let n = 1000u64;
        let idx = synth_index(n);
        let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        assert_eq!(idx.r_t, merkle_root(&leaves));
    }

    #[test]
    #[test]
    #[test]
    // Validates the canonical layout against the consensus-pinned R_T. Needs the Gemma GGUF.
    // Run: cargo test --lib pom -- --ignored --nocapture
    #[test]
    #[ignore = "needs Gemma-3-4B GGUF on disk"]
    // End-to-end H3 test on the REAL Gemma tier — this is exactly what generate_block_if_pom does
    // at runtime post-fork: pph from a "block header", nonce, h3=true, build_proof, then locally
    // verify_proof. If our local verify_proof PASSES, the miner is submitting proofs that satisfy
    // the same math the node uses; any pool `PowValueMismatch` rejection is on the pool/node side.
    #[test]
    #[ignore = "needs Gemma-3-4B GGUF on disk"]
    // Emit POM_SAMPLE_submit.json for the CANONICAL VECTOR that the node-built `pom-verify-test`
    // expects (pph=4d27ef7d…, ts=1_700_000_000, nonce=1366), so we can run the chain-exact
    // `verify_pom_proof` on a proof OUR build_proof produces over the real Gemma tier.
    // Run: KERYX_GEMMA_GGUF=…/Gemma-3-4B/model.gguf cargo test --lib emit_canonical_pom_sample -- --ignored --nocapture
    #[test]
    #[ignore = "needs Gemma-3-4B GGUF on disk; emits /tmp/POM_SAMPLE_submit.json"]
    /// ZERO-DUP AMD path on the REAL tier: the in-process llama engine hosts the model
    /// (libkeryx-llama-vk.so via KERYX_LLAMA_VK_SO or next to the test binary), the walk gathers
    /// over its resident VRAM tensors -> must find the SAME lowest winner the OpenCL blob walk
    /// finds (same fixed pph/target search) -> proof verifies vs the pinned R_T. Needs an AMD
    /// GPU + the .so + KERYX_GEMMA_GGUF.
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    /// H5 on-hardware correctness: the OpenCL blob kernel with walk_v2=1 must find a nonce whose v2
    /// walk the CPU `transition_v2` (via `walk_final(.., true)`) independently confirms passes the
    /// target — proving `pom_mine.cl`'s mix64-chain branch is byte-exact with `pom.rs`. The winner
    /// must ALSO differ from the v1 winner (9559 for this pph/target), i.e. the eras really diverge.
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    /// H5 on-hardware correctness for the ZERO-DUP Vulkan shader: same oracle as
    /// gpu_walk_v2_opencl_matches_cpu but through `pom_walk_vk.comp`'s walk_v2 branch. Needs the .so
    /// (KERYX_LLAMA_VK_SO) + a Vulkan GPU (KERYX_LLAMA_VK_DEVICE).
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    /// H5.1 on-hardware correctness: mining with h5_1=1 (realistic H5.1 = walk_v2=1 too) must find a
    /// nonce whose SEED derives from the H5.1-salted pph words — the CPU `pom_block_seed(.., h5_1=true)`
    /// + v2 walk independently confirms it passes target. Proves both GPU backends thread the seed
    /// words s0..s3 correctly. The H5.1 winner MUST differ from the h5_1=false v2 winner (1053), since
    /// a different seed = a different walk trajectory. OpenCL blob leg.
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    /// Engine unload frees the model + VRAM and a fresh ensure_loaded works after it — the rescue
    /// path for "byte gate failed on a small card" (unload the engine, give the VRAM to the blob).
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    fn gpu_engine_unload_reload() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        let gpu: usize = std::env::var("KERYX_LLAMA_VK_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        assert!(crate::llama_engine_vk::ensure_loaded(&path, gpu), "initial engine load failed");
        assert!(crate::llama_engine_vk::available());
        assert!(crate::llama_engine_vk::unload(), "unload must report an engine was freed");
        assert!(!crate::llama_engine_vk::available(), "engine must be gone after unload");
        assert!(!crate::llama_engine_vk::unload(), "second unload is a no-op");
        assert!(crate::llama_engine_vk::ensure_loaded(&path, gpu), "reload after unload failed");
        assert!(crate::llama_engine_vk::pom_ready(), "walk must be ready after reload");
        eprintln!("engine unload -> VRAM freed -> reload OK ✅");
    }

    /// H5.1 seed correctness for the ZERO-DUP Vulkan shader (s0..s3 push-constants). Same oracle as
    /// gpu_h5_1_seed_opencl_matches_cpu, through pom_walk_vk.comp. Needs KERYX_LLAMA_VK_SO + a Vulkan GPU.
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    /// H5.2 on-hardware correctness (OpenCL blob): mining with h5_2=1 must find a nonce whose SEED
    /// derives from the H5.2-salted pph words — the CPU `pom_block_seed(.., h5_2=true)` + v2 walk
    /// confirms it. The GPU kernel is byte-identical to the H5.1 build (a seed salt is host-side),
    /// so this proves the host salt-selection path; the winner must differ from the H5.1 winner too.
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    /// H5.2 seed correctness for the ZERO-DUP Vulkan shader. Same oracle, through pom_walk_vk.comp.
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    /// Wall-clock throughput bench for the OPENCL blob walk on a chosen card — the apples-to-
    /// apples baseline for bench_llama_vk_walk (the miner's own hashrate accounting is not a
    /// wall-clock measure). KERYX_BENCH_CL_DEV picks the OpenCL device index (default 0).
    #[test]
    #[ignore]
    #[cfg(feature = "pom-opencl")]
    fn bench_opencl_walk() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        let devs = opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_GPU).expect("cl devices");
        let di: usize = std::env::var("KERYX_BENCH_CL_DEV").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let id = devs[di] as usize;
        let name = opencl3::device::Device::new(devs[di]).name().unwrap_or_default();
        eprintln!("opencl walk bench on device {di} ({name})");
        crate::pom_opencl::bind_thread_device(id);
        crate::pom_opencl::set_mining_tier([0u8; 32], path, 1);
        crate::pom_opencl::ensure_installed();
        let pph = blake(b"bench-llama-vk");
        let target = [0u8; 32]; // impossible target -> full grind
        let batch: u64 = 1 << 21;
        let _ = crate::pom_opencl::mine_v4(&pph, 1_700_000_000, &target, 0, batch); // warmup
        let start = std::time::Instant::now();
        let rounds: u64 = 8;
        for i in 0..rounds {
            let _ = crate::pom_opencl::mine_v4(&pph, 1_700_000_000, &target, i * batch, batch);
        }
        let secs = start.elapsed().as_secs_f64();
        eprintln!(
            "opencl walk: {:.2} MH/s ({} nonces in {:.2}s)",
            (rounds * batch) as f64 / secs / 1e6,
            rounds * batch,
            secs
        );
    }

    /// Throughput bench for the zero-dup engine walk (no consensus assertion — correctness is
    /// covered by gpu_real_tier_end_to_end_llama_vk). Prints MH/s; compare against the SAME
    /// card's OpenCL blob rate. Needs the .so + KERYX_GEMMA_GGUF + a GPU.
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    fn bench_llama_vk_walk() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        let gpu: usize = std::env::var("KERYX_LLAMA_VK_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        assert!(crate::llama_engine_vk::ensure_loaded(&path, gpu), "engine load failed");
        assert!(crate::llama_engine_vk::pom_ready(), "walk not ready");
        let p = pph_words_for_era(&blake(b"bench-llama-vk"), false);
        let t = [0u64; 4]; // impossible target -> full grind, no early exit
        let batch: u64 = 1 << 21;
        // warmup
        let _ = crate::llama_engine_vk::pom_mine(p, p, 1_700_000_000, t, 0, batch, false);
        let start = std::time::Instant::now();
        let rounds: u64 = 8;
        for i in 0..rounds {
            let _ = crate::llama_engine_vk::pom_mine(p, p, 1_700_000_000, t, i * batch, batch, false);
        }
        let secs = start.elapsed().as_secs_f64();
        eprintln!(
            "llama-vk walk: {:.2} MH/s ({} nonces in {:.2}s)",
            (rounds * batch) as f64 / secs / 1e6,
            rounds * batch,
            secs
        );
    }

    /// Full AMD path on the REAL tier: load_tier (WeightIndex + GPU blob + PomMiner) -> GPU mine
    /// over the resident Gemma weights -> build proof from the resident index -> verify vs pinned R_T.
    /// Proves the GPU blob and the proof-side WeightIndex are the same canonical chunks.
    #[test]
    #[ignore]
    #[cfg(feature = "pom-opencl")]
    /// CUDA variant: run the pom_mine kernel (cudarc) over the real Gemma-3-4B tier resident in
    /// NVIDIA VRAM, build the proof from the resident WeightIndex, verify vs the pinned R_T, and
    /// assert the consensus N + R_T. Proves the CUDA search reads the SAME canonical chunks as the
    /// proof side (i.e. the NVIDIA path is byte-exact with consensus).
    /// Run: KERYX_GEMMA_GGUF=… cargo test --release --features pom-cuda gpu_real_tier_end_to_end_cuda -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(feature = "pom-cuda")]
    /// Tier-2 (Qwen3-32B) candle-CUDA consensus check: WeightIndex R_T must equal the node-pinned
    /// tier-2 root e2aa6659…, the GPU gather N must match, and a candle-CUDA-mined nonce must build
    /// a proof that verifies. Proves the bigger Qwen3 GGUF loads + gathers byte-exact (the 5090 tier).
    /// Run: KERYX_QWEN3_GGUF=… cargo test --release --features pom-cuda gpu_real_tier_qwen3_cuda -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(feature = "pom-cuda")]
    /// Emit a REAL `mining.submit` wire (params[5] = borsh PomProof hex) built over the real
    /// Gemma-3-4B tier, for the pool to replay through `_submitBlock` → keryxd `verify_pom_proof`
    /// in isolation. The proof is verified LOCALLY first, so this is a known-good vector. Writes
    /// `<KERYX_SAMPLE_OUT>_submit.json` + `_vector.txt` (default prefix /tmp/pom_sample).
    /// Run: KERYX_GEMMA_GGUF=… cargo test --features pom-opencl emit_sample_submit_wire -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(feature = "pom-opencl")]
    /// Mode B: build a proof bound to a REAL staging header's pre_pow_hash + timestamp (supplied via
    /// env), so the pool can reconstruct an RpcRawBlock and submit it to keryxd. Mines at an easy
    /// test target (network diff is infeasible here), verifies locally, and writes the
    /// `{nonce_u64_dec, pom_proof_hex_lowercase, notes}` reply JSON.
    /// Run: KERYX_GEMMA_GGUF=… KERYX_POM_B_PPH=<64hex> KERYX_POM_B_TIME=<u64> \
    ///      cargo test --release -p keryx-miner-supr --features pom-opencl emit_mode_b_proof -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(feature = "pom-opencl")]
    /// Emit a REAL H4 PoM **proof-v2** (recompute-from-chunks) over a REAL H4 model, entirely
    /// host-side (no GPU), for the pool's PoW / PoM-v2 acceptance test. Builds the WeightIndex from
    /// the GGUF (canonical R_T = the node's pinned tier root), builds the v2 proof via
    /// `build_proof_v2`, self-verifies via `verify_proof_v2`, and writes the wire hex + the full
    /// context the pool feeds into `verify_pom_proof_v2`.
    ///
    /// Run (EXAONE = tier 0, the smallest H4 model):
    ///   KERYX_H4_GGUF=$HOME/keryx-model-cache/EXAONE-4.0-1.2B/model.gguf KERYX_H4_TIER=0 \
    ///   cargo test --release -p keryx-miner-supr emit_h4_v2_proof -- --ignored --nocapture
    #[test]
    #[ignore]
    /// Validate + benchmark candle's CPU backend on the AMD OPoI inference model (Gemma-3-4B, the
    /// post-fork --light tier). Proves candle CPU can load + generate the Gemma3 quantized arch (the
    /// AMD inference path) and reports the real tok/s on this box. Needs the model staged at
    /// `<test-exe-dir>/models/Gemma-3-4B/` (symlink target/release/deps/models -> ../models).
    /// Run: cargo test --release -p keryx-miner-supr --features pom-opencl cpu_inference_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(feature = "pom-opencl")]
    fn cpu_inference_bench() {
        crate::slm::init_supported(&[&crate::models::GEMMA_3_4B]);
        let id = crate::models::GEMMA_3_4B.model_id;
        assert!(crate::slm::cpu_inference_enabled(), "pom-opencl build must force CPU inference");
        // First call loads the 2.48 GiB GGUF into RAM on CPU (one-time) + a short generation.
        let t_load = std::time::Instant::now();
        let warm = crate::slm::load_and_run_inference(&id, "Hello", 8);
        let load_s = t_load.elapsed().as_secs_f64();
        assert!(warm.is_some(), "candle CPU failed to load/run Gemma-3-4B — AMD inference path broken");
        // Second call: model resident -> the per-challenge generation rate.
        let n = 48usize;
        let t = std::time::Instant::now();
        let out = crate::slm::load_and_run_inference(&id, "The capital of France is", n);
        let s = t.elapsed().as_secs_f64();
        let text = out.expect("CPU inference returned None on the resident call");
        let sample: String = text.chars().take(120).collect();
        eprintln!("=== Gemma-3-4B CPU inference on this box ===");
        eprintln!("  load (first call, 2.48 GiB GGUF -> RAM): {:.1}s", load_s);
        eprintln!("  resident gen: {} tokens in {:.1}s => ~{:.2} tok/s", n, s, n as f64 / s);
        eprintln!("  sample: {:?}", sample);
    }

}
