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
use candle_core::quantized::gguf_file;
use candle_core::Device;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
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

pub const CHUNK_WORDS: usize = 4; // 32 B chunk
const SEED_SALT: u64 = 0x4B65727978500; // "KeryxP"

/// Walk length / opening count — MUST match the node's `POM_WALK_STEPS` / `POM_OPENINGS`.
/// K=256 — chosen compromise (~25 MH/s on a 3090, solid possession).
pub const POM_WALK_STEPS: u32 = 256;
pub const POM_OPENINGS: usize = 32;

// --- wire structs (field order == node's PomOpening/PomProof) ---

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomOpening {
    pub state_before: u64,
    pub chunk: [u8; 32],
    pub weight_path: Vec<[u8; 32]>,
    pub trace_path_before: Vec<[u8; 32]>,
    pub trace_path_after: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProof {
    pub tier: u8,
    pub trace_root: [u8; 32],
    pub pow_value: [u8; 32],
    pub final_state: u64,
    pub initial_trace_path: Vec<[u8; 32]>,
    pub final_trace_path: Vec<[u8; 32]>,
    pub openings: Vec<PomOpening>,
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
pub fn seed_state(pow_seed: u64) -> u64 {
    mix64(pow_seed ^ SEED_SALT)
}

#[inline]
pub fn transition(state: u64, chunk: &[u64; CHUNK_WORDS]) -> u64 {
    let mut h = state;
    for &w in chunk.iter() {
        h ^= w;
    }
    mix64(h)
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
fn trace_leaf(state: u64) -> [u8; 32] {
    blake(&state.to_le_bytes())
}

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
/// BYTE-IDENTICAL to `pom_mine.cu::pom_seed_fold` and the node's `pom_block_seed`(`_h3`).
pub fn pom_block_seed(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64, h3: bool) -> u64 {
    pom_block_seed_from_words(&pph_words_for_era(pre_pow_hash, h3), timestamp, nonce)
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

fn verify_merkle(leaf: [u8; 32], index: u64, path: &[[u8; 32]], root: &[u8; 32]) -> bool {
    let mut acc = leaf;
    let mut idx = index;
    for sib in path {
        acc = if idx & 1 == 0 { hash_pair(&acc, sib) } else { hash_pair(sib, &acc) };
        idx >>= 1;
    }
    &acc == root
}

/// Fiat-Shamir challenge step-indices — byte-layout identical to node/pom-core.
pub fn challenges(pre_pow_hash: &[u8; 32], nonce: u64, trace_root: &[u8; 32], pow_value: &[u8; 32], t: usize, k: u32) -> Vec<u32> {
    let mut fs = [0u8; 104];
    fs[..32].copy_from_slice(pre_pow_hash);
    fs[32..40].copy_from_slice(&nonce.to_le_bytes());
    fs[40..72].copy_from_slice(trace_root);
    fs[72..104].copy_from_slice(pow_value);
    let seed = blake(&fs);
    let mut out = Vec::with_capacity(t);
    for j in 0..t as u64 {
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(&seed);
        buf[32..].copy_from_slice(&j.to_le_bytes());
        let d = blake(&buf);
        let v = u64::from_le_bytes(d[..8].try_into().unwrap());
        out.push((v % k as u64) as u32);
    }
    out
}

/// The hot search walk: K data-dependent reads, returns only `state[K]` (no trace recording).
/// This is the per-nonce work; on GPU (slice 3b) this becomes the kernel over VRAM weights.
pub fn walk_final<F: Fn(u64) -> [u64; CHUNK_WORDS]>(seed: u64, n_chunks: u64, k: u32, read_chunk: F) -> u64 {
    let mut state = seed;
    let mut off = state % n_chunks;
    for _ in 0..k {
        state = transition(state, &read_chunk(off));
        off = state % n_chunks;
    }
    state
}

/// CPU Proof-of-Model mining (slice 3a — functional, slow). Searches nonces in
/// `nonce_start..nonce_start+max_nonces`; on the first whose `pom_pow_value <= target`,
/// re-walks to build the full `PomProof`. GPU fast-path is slice 3b. Returns the winning
/// nonce + proof, or None if the range is exhausted.
#[allow(clippy::too_many_arguments)]
pub fn mine_pom(
    index: &WeightIndex,
    tier: u8,
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    target: &[u8; 32],
    k: u32,
    t: usize,
    nonce_start: u64,
    max_nonces: u64,
    h3: bool,
) -> Option<(u64, PomProof)> {
    for nonce in nonce_start..nonce_start.saturating_add(max_nonces) {
        let seed = pom_block_seed(pre_pow_hash, timestamp, nonce, h3);
        let final_state = walk_final(seed, index.n_chunks, k, |o| index.read_chunk(o));
        if le_leq(&pom_pow_value(final_state, pre_pow_hash, h3), target) {
            let proof = build_proof(tier, pre_pow_hash, nonce, seed, index.n_chunks, k, t, |o| index.read_chunk(o), |o| index.merkle_path(o), h3);
            return Some((nonce, proof));
        }
    }
    None
}

/// PROVER. Re-walk the (already-won) nonce recording the trace, commit it, and open the
/// `t` FS-selected steps. `read_chunk(off)` reads the 32 B chunk at canonical chunk index
/// `off` from the resident weight blob; `weight_leaves` is the precomputed per-chunk leaf
/// set (`blake(chunk_bytes)`) over the canonical layout, used to produce weight Merkle paths.
#[allow(clippy::too_many_arguments)]
pub fn build_proof<F, WP>(
    tier: u8,
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    n_chunks: u64,
    k: u32,
    t: usize,
    read_chunk: F,
    weight_path: WP,
    h3: bool,
) -> PomProof
where
    F: Fn(u64) -> [u64; CHUNK_WORDS],
    WP: Fn(u64) -> Vec<[u8; 32]>,
{
    let mut trace = Vec::with_capacity(k as usize + 1);
    let mut state = seed;
    trace.push(state);
    let mut off = state % n_chunks;
    for _ in 0..k {
        state = transition(state, &read_chunk(off));
        trace.push(state);
        off = state % n_chunks;
    }
    let trace_leaves: Vec<[u8; 32]> = trace.iter().map(|&s| trace_leaf(s)).collect();
    let trace_root = merkle_root(&trace_leaves);
    let final_state = trace[k as usize];
    let pow_value = pom_pow_value(final_state, pre_pow_hash, h3);

    let chs = challenges(pre_pow_hash, nonce, &trace_root, &pow_value, t, k);
    let openings = chs
        .iter()
        .map(|&i| {
            let i = i as usize;
            let sb = trace[i];
            let off = sb % n_chunks;
            PomOpening {
                state_before: sb,
                chunk: words_to_bytes(&read_chunk(off)),
                weight_path: weight_path(off),
                trace_path_before: merkle_proof(&trace_leaves, i),
                trace_path_after: merkle_proof(&trace_leaves, i + 1),
            }
        })
        .collect();

    PomProof {
        tier,
        trace_root,
        pow_value,
        final_state,
        initial_trace_path: merkle_proof(&trace_leaves, 0),
        final_trace_path: merkle_proof(&trace_leaves, k as usize),
        openings,
    }
}

/// Self-check a built proof before submit (same logic the node runs). Cheap insurance
/// against emitting a block the node will reject.
#[allow(clippy::too_many_arguments)]
pub fn verify_proof(pre_pow_hash: &[u8; 32], nonce: u64, seed: u64, proof: &PomProof, n_chunks: u64, k: u32, t: usize, r_t: &[u8; 32], target: &[u8; 32], h3: bool) -> bool {
    if proof.openings.len() != t {
        return false;
    }
    if pom_pow_value(proof.final_state, pre_pow_hash, h3) != proof.pow_value {
        return false;
    }
    if !le_leq(&proof.pow_value, target) {
        return false;
    }
    if !verify_merkle(trace_leaf(seed), 0, &proof.initial_trace_path, &proof.trace_root) {
        return false;
    }
    if !verify_merkle(trace_leaf(proof.final_state), k as u64, &proof.final_trace_path, &proof.trace_root) {
        return false;
    }
    let chs = challenges(pre_pow_hash, nonce, &proof.trace_root, &proof.pow_value, t, k);
    for (op, &i) in proof.openings.iter().zip(chs.iter()) {
        let i = i as u64;
        if !verify_merkle(trace_leaf(op.state_before), i, &op.trace_path_before, &proof.trace_root) {
            return false;
        }
        let off = op.state_before % n_chunks;
        if !verify_merkle(blake(&op.chunk), off, &op.weight_path, r_t) {
            return false;
        }
        let state_after = transition(op.state_before, &chunk_to_words(&op.chunk));
        if !verify_merkle(trace_leaf(state_after), i + 1, &op.trace_path_after, &proof.trace_root) {
            return false;
        }
    }
    true
}

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
    /// GGUF length + mtime — if either differs from the live file, the cache is stale → rebuild.
    gguf_len: u64,
    gguf_mtime: i64,
    n_chunks: u64,
    r_t: [u8; 32],
    total_levels: u32,
    /// Flattened `ChunkSource::Gguf` table: (first-chunk index, gguf byte offset) pairs.
    table: Vec<u64>,
    /// Flattened sparse `checkpoints`: (level, byte offset, node count) triples.
    checkpoints: Vec<u64>,
}

/// v2 = sparse checkpoint tree (was v1 = dense all-levels tree). Bumping this invalidates every
/// legacy dense `pom-tree.bin` → it is rebuilt as a tiny sparse tree (and the huge old file deleted).
const POM_TREE_CACHE_VERSION: u32 = 2;

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
fn write_pom_tree_meta(
    meta_path: &std::path::Path,
    gguf_path: &str,
    idx: &WeightIndex,
    table: &[(u64, u64)],
) -> std::io::Result<()> {
    let gm = std::fs::metadata(gguf_path)?;
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
        gguf_len: gm.len(),
        gguf_mtime: pom_tree_mtime_secs(&gm),
        n_chunks: idx.n_chunks,
        r_t: idx.r_t,
        total_levels: idx.total_levels,
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
    pub fn build_from_gguf(path: &str) -> candle_core::Result<Self> {
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
        if let Some(idx) = Self::reuse_cached_tree(&cache_path, &meta_path, path) {
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
            if let Some(idx) = Self::reuse_cached_tree(&cache_path, &meta_path, path) {
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
                                if let Err(e) = write_pom_tree_meta(&meta_path, path, &idx, &table) {
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
        let device = Device::Cpu;
        let mut file = File::open(path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order

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
        for name in &names {
            let file_off = content.tensor_data_offset + content.tensor_infos[name].offset;
            let qt = content.tensor(&mut file, name, &device)?;
            let bytes = qt.data()?;
            let full = bytes.len() / 32;
            if full > 0 {
                table.push((n_chunks, file_off));
            }
            for c in 0..full {
                let chunk = &bytes[c * 32..c * 32 + 32];
                batch_buf.push(blake(chunk));
                n_chunks += 1;
                if batch_buf.len() == batch_size as usize {
                    writer.write_all(&fold_levels(&batch_buf, k)).map_err(candle_core::Error::wrap)?;
                    batch_buf.clear();
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
        };
        Ok((idx, table_snapshot))
    }

    /// Reconstruct a byte-identical `WeightIndex` from an existing shared cache + meta sidecar, or
    /// `None` if the cache is absent/stale/corrupt (→ caller rebuilds). Validated FOUR ways: cache
    /// version, GGUF length+mtime, tree-file size, and the on-disk root hash == the meta's `r_t`.
    fn reuse_cached_tree(cache_path: &std::path::Path, meta_path: &std::path::Path, gguf_path: &str) -> Option<Self> {
        let bytes = std::fs::read(meta_path).ok()?;
        let meta = PomTreeMeta::try_from_slice(&bytes).ok()?;
        if meta.version != POM_TREE_CACHE_VERSION {
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

    /// Inclusion path for chunk index `off`: stored siblings read from the checkpoint file, unstored
    /// intermediate levels recomputed on the fly from the nearest checkpoint / the GGUF leaves.
    /// Byte-identical to the dense full-tree path: an out-of-range sibling is the node itself.
    pub fn merkle_path(&self, off: u64) -> Vec<[u8; 32]> {
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
        }
    }

    /// BYTE-EXACT GATE (consensus): the sparse checkpoint-built root MUST equal the dense canonical
    /// root (`merkle_root_mini` over all leaves = what the node pins in POM_TIERS) for every N,
    /// including the non-power-of-two sizes whose short tails used to drop the hash(x,x) carries.
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

    /// GGUF-backed `read_chunk`: lay the canonical chunks across 3 "tensors" with header + inter-
    /// tensor padding (so file offset != off*32), build the per-tensor offset table, and assert
    /// `read_chunk` (pread) returns the exact canonical chunks AND that a proof verifies — same as
    /// the RAM path, with no host copy of the weights.
    #[test]
    fn gguf_chunk_source_reads_match_and_proof_verifies() {
        let n = 1000u64;
        let uid = std::process::id();
        let gguf_path = std::env::temp_dir().join(format!("keryx-pom-fakegguf-{uid}.bin"));
        let _ = std::fs::remove_file(&gguf_path);
        let mut f = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&gguf_path).unwrap();

        // 3 tensors at chunk-start boundaries, with padding so file_off is not simply off*32.
        let splits = [0u64, 400, 750, n];
        let mut table: Vec<(u64, u64)> = Vec::new();
        let mut pos: u64 = 17; // header padding
        f.seek(SeekFrom::Start(pos)).unwrap();
        for w in splits.windows(2) {
            table.push((w[0], pos));
            for o in w[0]..w[1] {
                f.write_all(&words_to_bytes(&synth_chunk(o))).unwrap();
                pos += 32;
            }
            pos += 13; // inter-tensor padding gap
            f.seek(SeekFrom::Start(pos)).unwrap();
        }
        f.flush().unwrap();
        let file = File::open(&gguf_path).unwrap();

        // Build the SPARSE tree over the canonical synth chunks, with the GGUF chunk source.
        let tree_path = std::env::temp_dir().join(format!("keryx-pom-fakegguf-tree-{uid}.bin"));
        let _ = std::fs::remove_file(&tree_path);
        let k = CHECKPOINT_INTERVAL;
        let batch_size = 1u64 << k;
        {
            let mut writer = BufWriter::new(
                OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tree_path).unwrap(),
            );
            let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
            for o in 0..n {
                batch.push(blake(&words_to_bytes(&synth_chunk(o))));
                if batch.len() == batch_size as usize {
                    writer.write_all(&fold_levels(&batch, k)).unwrap();
                    batch.clear();
                }
            }
            if !batch.is_empty() {
                writer.write_all(&fold_levels(&batch, k)).unwrap();
            }
            writer.flush().unwrap();
        }
        let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n).unwrap();
        let tree_file = File::open(&tree_path).unwrap();
        let idx = WeightIndex {
            n_chunks: n,
            r_t,
            chunks: ChunkSource::Gguf { file, table },
            tree_file,
            tree_path,
            checkpoints,
            total_levels,
            persistent: false,
        };

        // Every chunk read by pread matches the canonical chunk, across all segments + padding.
        for o in 0..n {
            assert_eq!(idx.read_chunk(o), synth_chunk(o), "chunk {o}");
        }
        // A proof built from the GGUF source verifies against R_T (target 0xff..ff = first nonce wins).
        let (k, t) = (POM_WALK_STEPS, POM_OPENINGS);
        let pph = [7u8; 32];
        let target = [0xffu8; 32];
        let (nonce, proof) = mine_pom(&idx, 2, &pph, 123, &target, k, t, 0, 1, false).expect("max target → win");
        let seed = pom_block_seed(&pph, 123, nonce, false);
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false));

        let _ = std::fs::remove_file(&gguf_path);
    }

    /// H3 hardfork salt: the pph words feeding BOTH PoM folds are XOR-salted with POM_H3_PPH_SALT
    /// at/after the gate. Proves (a) the salt equals the node's sha256 derivation, (b) it changes
    /// the walk seed + pow value, and (c) proofs are era-bound — an H3 proof verifies under h3=true
    /// and is REJECTED under h3=false. That rejection IS the forced-update guarantee: a pre-H3
    /// binary's proof verifies false post-gate.
    #[test]
    fn h3_salt_is_byte_exact_and_era_bound() {
        // (a) constant == sha256("keryx-h3-pom-pph-salt") as 4 LE u64 words — node-identical.
        assert_eq!(
            POM_H3_PPH_SALT,
            [0x7C99D381176D4EC4, 0xC2E28E3E28118C36, 0xD496CE1B129B76CA, 0x47CF0979FA580BCE]
        );
        let pph = [0x5au8; 32];
        // (b) salted words == raw XOR salt; both folds differ across the gate.
        let raw = pph_words_for_era(&pph, false);
        let salted = pph_words_for_era(&pph, true);
        for i in 0..4 {
            assert_eq!(salted[i], raw[i] ^ POM_H3_PPH_SALT[i], "word {i} salt");
        }
        assert_ne!(pom_block_seed(&pph, 42, 7, true), pom_block_seed(&pph, 42, 7, false), "H3 changes the seed");
        assert_ne!(pom_pow_value(123, &pph, true), pom_pow_value(123, &pph, false), "H3 changes the pow value");

        // (c) full round-trip on a synthetic tier. Max target → the first nonce wins.
        let (k, t) = (POM_WALK_STEPS, POM_OPENINGS);
        let idx = synth_index(4096);
        let target = [0xffu8; 32];
        let (nonce, proof) = mine_pom(&idx, 0, &pph, 123, &target, k, t, 0, 1, true).expect("h3 mine");
        let seed_h3 = pom_block_seed(&pph, 123, nonce, true);
        assert!(
            verify_proof(&pph, nonce, seed_h3, &proof, idx.n_chunks, k, t, &idx.r_t, &target, true),
            "an H3 proof must verify under the H3 folds"
        );
        // The SAME proof, checked with the pre-H3 folds → pow_value mismatch → rejected.
        let seed_pre = pom_block_seed(&pph, 123, nonce, false);
        assert!(
            !verify_proof(&pph, nonce, seed_pre, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false),
            "an H3 proof MUST be rejected under pre-H3 folds (the forced-update lever)"
        );
    }

    /// Real-GGUF byte-identity: build the index from a downloaded model and prove that chunks
    /// read by `pread` (GGUF) verify against the model's own `R_T` (whose leaves were hashed from
    /// candle's `qt.data()`). Confirms `pread(tensor_data_offset + offset)` == `qt.data()` for real
    /// quant types. Ignored (needs the GGUF); run: `cargo test -p keryx-miner -- --ignored gguf_real`.
    #[test]
    #[ignore]
    fn gguf_real_model_read_chunk_byte_identical() {
        let path = "/home/slash/KERYX-KRX/claude/Outils PoM/keryx-miner-test CPU-Llama3-70B/target/release/models/Gemma-3-4B/model.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: GGUF not found at {path}");
            return;
        }
        let idx = WeightIndex::build_from_gguf(path).expect("build index from real GGUF");
        eprintln!("real model index: N={} chunks", idx.n_chunks);
        let (k, t) = (POM_WALK_STEPS, POM_OPENINGS);
        let pph = [3u8; 32];
        let target = [0xffu8; 32]; // max → the first nonce wins, so 1 nonce suffices
        let (nonce, proof) = mine_pom(&idx, 0, &pph, 99, &target, k, t, 0, 1, false).expect("max target → win");
        let seed = pom_block_seed(&pph, 99, nonce, false);
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false),
            "GGUF-pread chunks must verify against the model's R_T (byte-identity broken otherwise)"
        );
    }

    #[test]
    fn weight_index_root_matches_standalone() {
        // The prebuilt-tree root equals the standalone merkle_root over the same leaves.
        let n = 1000u64;
        let idx = synth_index(n);
        let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        assert_eq!(idx.r_t, merkle_root(&leaves));
    }

    #[test]
    fn build_then_self_verify() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"pph");
        let nonce = 0xabc;
        let seed = pom_block_seed(&pph, 111, nonce, false);

        let proof = build_proof(2, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], false));
        // borsh wire-format round-trips (same encoding the node decodes).
        let bytes = borsh::to_vec(&proof).unwrap();
        let back: PomProof = borsh::from_slice(&bytes).unwrap();
        assert!(verify_proof(&pph, nonce, seed, &back, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], false));
        assert_eq!(back.tier, 2);
    }

    #[test]
    fn wrong_target_or_root_fails() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"pph2");
        let nonce = 7;
        let seed = pom_block_seed(&pph, 1, nonce, false);
        let proof = build_proof(0, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(!verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0u8; 32], false), "zero target must fail");
        assert!(!verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &blake(b"wrong"), &[0xff; 32], false), "wrong R_T must fail");
    }

    #[test]
    fn cpu_mine_finds_nonce_and_proof_verifies() {
        let (k, t) = (128u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"mine-pph");
        let ts = 555;
        // Target requiring pow_value MSB <= 0x10 (~6.6% of nonces) — found within a few tries.
        let mut target = [0xffu8; 32];
        target[31] = 0x10;
        let (nonce, proof) = mine_pom(&idx, 1, &pph, ts, &target, k, t, 0, 100_000, false).expect("mine a nonce");
        let seed = pom_block_seed(&pph, ts, nonce, false);
        // The proof verifies against the same target the node would use.
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false));
        assert_eq!(proof.tier, 1);
    }

    // Validates the canonical layout against the consensus-pinned R_T. Needs the Gemma GGUF.
    // Run: cargo test --lib pom -- --ignored --nocapture
    #[test]
    #[ignore = "needs Gemma-3-4B GGUF on disk"]
    fn weight_index_matches_pinned_gemma() {
        let path = std::env::var("KERYX_GEMMA_GGUF").unwrap_or_else(|_| "/home/slash/KERYX-KRX/claude/keryx-miner/target/release/models/Gemma-3-4B/model.gguf".to_string());
        let idx = WeightIndex::build_from_gguf(&path).expect("build index");
        assert_eq!(idx.n_chunks, 77_604_776, "chunk count must match pinned GEMMA_3_4B_POM_CHUNKS");
        let pinned: [u8; 32] = [
            0x84, 0x6c, 0xaa, 0x40, 0x0c, 0xf0, 0x14, 0x13, 0x21, 0x18, 0x49, 0x5d, 0x22, 0xe4, 0xbf, 0xa2,
            0x42, 0x45, 0x4e, 0xac, 0x0d, 0x83, 0x5c, 0x3f, 0x8e, 0x63, 0x47, 0xd0, 0x13, 0x9d, 0x1b, 0x7e,
        ];
        assert_eq!(idx.r_t, pinned, "miner R_T must equal node-pinned GEMMA_3_4B_POM_ROOT");

        // A real proof over the real model self-verifies against the pinned R_T.
        let pph = blake(b"gemma-pph");
        let nonce = 1234;
        let seed = pom_block_seed(&pph, 99, nonce, false);
        let proof = build_proof(0, &pph, nonce, seed, idx.n_chunks, 256, 32, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, 256, 32, &idx.r_t, &[0xff; 32], false));
    }

    // End-to-end H3 test on the REAL Gemma tier — this is exactly what generate_block_if_pom does
    // at runtime post-fork: pph from a "block header", nonce, h3=true, build_proof, then locally
    // verify_proof. If our local verify_proof PASSES, the miner is submitting proofs that satisfy
    // the same math the node uses; any pool `PowValueMismatch` rejection is on the pool/node side.
    #[test]
    #[ignore = "needs Gemma-3-4B GGUF on disk"]
    fn h3_end_to_end_real_model() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        let idx = WeightIndex::build_from_gguf(&path).expect("build index");
        assert_eq!(idx.n_chunks, 77_604_776);
        let pph = blake(b"h3-real-gemma-end-to-end");
        let ts = 1_700_000_000u64;
        for &nonce in &[1u64, 42, 12345, 999_999_999] {
            let seed = pom_block_seed(&pph, ts, nonce, true);
            let proof = build_proof(1, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, |o| idx.read_chunk(o), |o| idx.merkle_path(o), true);
            let ok = verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &[0xff; 32], true);
            eprintln!("nonce={nonce:<12}  h3=true  pow_value[0..8]={:02x?}  verify_proof={}", &proof.pow_value[0..8], if ok { "OK" } else { "REJECTED" });
            assert!(ok, "H3 proof over real Gemma tier must self-verify");
        }
    }

    // Emit POM_SAMPLE_submit.json for the CANONICAL VECTOR that the node-built `pom-verify-test`
    // expects (pph=4d27ef7d…, ts=1_700_000_000, nonce=1366), so we can run the chain-exact
    // `verify_pom_proof` on a proof OUR build_proof produces over the real Gemma tier.
    // Run: KERYX_GEMMA_GGUF=…/Gemma-3-4B/model.gguf cargo test --lib emit_canonical_pom_sample -- --ignored --nocapture
    #[test]
    #[ignore = "needs Gemma-3-4B GGUF on disk; emits /home/marcel/POM_SAMPLE_submit.json"]
    fn emit_canonical_pom_sample() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        let idx = WeightIndex::build_from_gguf(&path).expect("build index");
        assert_eq!(idx.n_chunks, 77_604_776);
        let pph: [u8; 32] = [
            0x4d, 0x27, 0xef, 0x7d, 0x41, 0xb8, 0x1e, 0xd8, 0xf8, 0xef, 0xe0, 0xca, 0x6f, 0xf2, 0xa7, 0x7a,
            0x69, 0x6e, 0xd0, 0x0e, 0xdb, 0x6d, 0x4d, 0x01, 0x5a, 0xd3, 0xab, 0xd8, 0xfd, 0xe5, 0x18, 0xa2,
        ];
        let ts: u64 = 1_700_000_000;
        let nonce: u64 = 1366;
        let seed = pom_block_seed(&pph, ts, nonce, false);
        let proof = build_proof(0, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        let bytes = borsh::to_vec(&proof).expect("borsh");
        let hexs: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        println!("proof: {} bytes, pow_value={}", bytes.len(), proof.pow_value.iter().map(|b| format!("{:02x}", b)).collect::<String>());
        let json = format!("{{\"id\":1,\"method\":\"mining.submit\",\"params\":[\"a\",\"j\",\"{:016x}\",\"tag\",\"\",\"{}\"]}}", nonce, hexs);
        std::fs::write("/home/marcel/POM_SAMPLE_submit.json", json).expect("write sample");
        println!("WROTE /home/marcel/POM_SAMPLE_submit.json");
    }

    /// ZERO-DUP AMD path on the REAL tier: the in-process llama engine hosts the model
    /// (libkeryx-llama-vk.so via KERYX_LLAMA_VK_SO or next to the test binary), the walk gathers
    /// over its resident VRAM tensors -> must find the SAME lowest winner the OpenCL blob walk
    /// finds (same fixed pph/target search) -> proof verifies vs the pinned R_T. Needs an AMD
    /// GPU + the .so + KERYX_GEMMA_GGUF.
    #[test]
    #[ignore]
    #[cfg(all(feature = "pom-opencl", unix))]
    fn gpu_real_tier_end_to_end_llama_vk() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        let idx = WeightIndex::build_from_gguf(&path).expect("build index");
        assert_eq!(idx.n_chunks, 77_604_776, "Gemma-3-4B tier N must be 77,604,776");
        let gpu: usize = std::env::var("KERYX_LLAMA_VK_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        assert!(crate::llama_engine_vk::ensure_loaded(&path, gpu), "engine load failed (.so present? VRAM?)");
        assert!(crate::llama_engine_vk::pom_ready(), "engine walk not ready (bufferDeviceAddress?)");
        assert!(crate::llama_engine_vk::pom_byte_gate(&idx), "byte gate must pass before mining");
        let pph = blake(b"gpu-real-e2e");
        let time = 1_700_000_000u64;
        let mut target = [0xffu8; 32]; // same fixed search as gpu_real_tier_end_to_end
        target[24..32].copy_from_slice(&0x0010_0000_0000_0000u64.to_le_bytes());
        let p = pph_words_for_era(&pph, false);
        let mut t = [0u64; 4];
        for (i, w) in t.iter_mut().enumerate() {
            *w = u64::from_le_bytes(target[i * 8..i * 8 + 8].try_into().unwrap());
        }
        let mut base = 0u64;
        let mut found = None;
        for _ in 0..512 {
            if let Some(n) = crate::llama_engine_vk::pom_mine(p, time, t, base, 1 << 16) {
                found = Some(n);
                break;
            }
            base = base.wrapping_add(1 << 16);
        }
        let nonce = found.expect("engine walk found no winner over the real tier");
        assert_eq!(nonce, 9559, "engine walk must find the SAME lowest winner as the OpenCL blob walk");
        let seed = pom_block_seed(&pph, time, nonce, false);
        let proof = build_proof(1, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target, false),
            "zero-dup engine proof must verify against the pinned R_T"
        );
        eprintln!(
            "ZERO-DUP engine walk mined nonce {nonce} over the REAL Gemma tier ({} chunks); proof verifies vs pinned R_T ✅",
            idx.n_chunks
        );
    }

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
        crate::pom_opencl::set_mining_tier(path, 1);
        crate::pom_opencl::ensure_installed();
        let pph = blake(b"bench-llama-vk");
        let target = [0u8; 32]; // impossible target -> full grind
        let batch: u64 = 1 << 21;
        let _ = crate::pom_opencl::mine(&pph, 1_700_000_000, &target, 0, batch, false); // warmup
        let start = std::time::Instant::now();
        let rounds: u64 = 8;
        for i in 0..rounds {
            let _ = crate::pom_opencl::mine(&pph, 1_700_000_000, &target, i * batch, batch, false);
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
        let _ = crate::llama_engine_vk::pom_mine(p, 1_700_000_000, t, 0, batch);
        let start = std::time::Instant::now();
        let rounds: u64 = 8;
        for i in 0..rounds {
            let _ = crate::llama_engine_vk::pom_mine(p, 1_700_000_000, t, i * batch, batch);
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
    fn gpu_real_tier_end_to_end() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        crate::pom_opencl::load_tier(&path, 0).expect("load_tier(real Gemma)");
        let (idx, tier) = active_index().expect("index installed by load_tier");
        let pph = blake(b"gpu-real-e2e");
        let time = 1_700_000_000u64;
        let mut target = [0xffu8; 32]; // ~1/4096 on the high word -> winner within a few batches
        target[24..32].copy_from_slice(&0x0010_0000_0000_0000u64.to_le_bytes());
        let mut base = 0u64;
        let mut found = None;
        for _ in 0..512 {
            if let Some(n) = crate::pom_opencl::mine(&pph, time, &target, base, 1 << 16, false) {
                found = Some(n);
                break;
            }
            base = base.wrapping_add(1 << 16);
        }
        let nonce = found.expect("GPU found no winner over the real tier");
        let seed = pom_block_seed(&pph, time, nonce, false);
        let proof = build_proof(*tier, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target, false),
            "real-tier GPU proof must verify against the pinned R_T"
        );
        eprintln!(
            "GPU mined nonce {nonce} over the REAL Gemma-3-4B tier ({} chunks); proof verifies vs pinned R_T 846caa40… ✅",
            idx.n_chunks
        );
    }

    /// CUDA variant: run the pom_mine kernel (cudarc) over the real Gemma-3-4B tier resident in
    /// NVIDIA VRAM, build the proof from the resident WeightIndex, verify vs the pinned R_T, and
    /// assert the consensus N + R_T. Proves the CUDA search reads the SAME canonical chunks as the
    /// proof side (i.e. the NVIDIA path is byte-exact with consensus).
    /// Run: KERYX_GEMMA_GGUF=… cargo test --release --features pom-cuda gpu_real_tier_end_to_end_cuda -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(feature = "pom-cuda")]
    fn gpu_real_tier_end_to_end_cuda() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        // Proof side: WeightIndex from the GGUF (canonical chunks + Merkle root).
        let idx = WeightIndex::build_from_gguf(&path).expect("build index");
        let pinned_rt: [u8; 32] = [
            0x84, 0x6c, 0xaa, 0x40, 0x0c, 0xf0, 0x14, 0x13, 0x21, 0x18, 0x49, 0x5d, 0x22, 0xe4,
            0xbf, 0xa2, 0x42, 0x45, 0x4e, 0xac, 0x0d, 0x83, 0x5c, 0x3f, 0x8e, 0x63, 0x47, 0xd0,
            0x13, 0x9d, 0x1b, 0x7e,
        ];
        assert_eq!(idx.n_chunks, 77_604_776, "Gemma-3-4B tier N must be 77,604,776");
        assert_eq!(idx.r_t, pinned_rt, "R_T must match the node-pinned Gemma root 846caa40…");
        // Search side: candle-CUDA gather miner (dedicated load — no inference coupling).
        let gm = crate::pom_gpu::PomGpuMiner::load(&path, 0).expect("load candle-CUDA gather");
        assert_eq!(gm.n_chunks(), idx.n_chunks, "GPU gather N must equal the proof-side index N");
        let pph = blake(b"gpu-real-e2e-cuda");
        let time = 1_700_000_000u64;
        let mut target = [0xffu8; 32]; // ~1/4096 on the high word -> winner within a few batches
        target[24..32].copy_from_slice(&0x0010_0000_0000_0000u64.to_le_bytes());
        let mut base = 0u64;
        let mut found = None;
        for _ in 0..512 {
            if let Some(n) = gm.mine(&pph, time, &target, base, 1 << 16).expect("mine") {
                found = Some(n);
                break;
            }
            base = base.wrapping_add(1 << 16);
        }
        let nonce = found.expect("CUDA GPU found no winner over the real tier");
        let seed = pom_block_seed(&pph, time, nonce, false);
        let proof = build_proof(0, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target, false),
            "real-tier CUDA GPU proof must verify against the pinned R_T"
        );
        eprintln!(
            "candle-CUDA mined nonce {nonce} over the REAL Gemma-3-4B tier ({} chunks); proof verifies vs pinned R_T 846caa40… ✅",
            idx.n_chunks
        );
    }

    /// Tier-2 (Qwen3-32B) candle-CUDA consensus check: WeightIndex R_T must equal the node-pinned
    /// tier-2 root e2aa6659…, the GPU gather N must match, and a candle-CUDA-mined nonce must build
    /// a proof that verifies. Proves the bigger Qwen3 GGUF loads + gathers byte-exact (the 5090 tier).
    /// Run: KERYX_QWEN3_GGUF=… cargo test --release --features pom-cuda gpu_real_tier_qwen3_cuda -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(feature = "pom-cuda")]
    fn gpu_real_tier_qwen3_cuda() {
        let path = std::env::var("KERYX_QWEN3_GGUF").expect("set KERYX_QWEN3_GGUF");
        let idx = WeightIndex::build_from_gguf(&path).expect("build index");
        // Node-pinned tier-2 (Qwen3-32B) invariants.
        let pinned_rt: [u8; 32] = [
            0xe2, 0xaa, 0x66, 0x59, 0xaa, 0xb4, 0x38, 0x7e, 0xb5, 0xfd, 0x79, 0x40, 0x9c, 0x0a,
            0x1a, 0x68, 0x86, 0x3a, 0x3d, 0xef, 0x3b, 0x66, 0x2c, 0xb4, 0x06, 0x16, 0x97, 0xf0,
            0xea, 0x87, 0xfa, 0x58,
        ];
        assert_eq!(idx.n_chunks, 617_380_448, "Qwen3-32B tier N must be 617,380,448");
        assert_eq!(idx.r_t, pinned_rt, "R_T must match the node-pinned Qwen3-32B root e2aa6659…");
        let gm = crate::pom_gpu::PomGpuMiner::load(&path, 0).expect("load candle-CUDA gather (Qwen3)");
        assert_eq!(gm.n_chunks(), idx.n_chunks, "GPU gather N must equal the proof-side index N");
        let pph = blake(b"gpu-real-e2e-qwen3");
        let time = 1_700_000_000u64;
        let mut target = [0xffu8; 32];
        target[24..32].copy_from_slice(&0x0010_0000_0000_0000u64.to_le_bytes());
        let mut base = 0u64;
        let mut found = None;
        for _ in 0..512 {
            if let Some(n) = gm.mine(&pph, time, &target, base, 1 << 16).expect("mine") {
                found = Some(n);
                break;
            }
            base = base.wrapping_add(1 << 16);
        }
        let nonce = found.expect("CUDA GPU found no winner over the Qwen3 tier");
        let seed = pom_block_seed(&pph, time, nonce, false);
        let proof = build_proof(2, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target, false),
            "Qwen3 tier-2 CUDA GPU proof must verify against the pinned R_T"
        );
        eprintln!(
            "candle-CUDA mined nonce {nonce} over the REAL Qwen3-32B tier ({} chunks); proof verifies vs pinned R_T e2aa6659… ✅",
            idx.n_chunks
        );
    }

    /// Emit a REAL `mining.submit` wire (params[5] = borsh PomProof hex) built over the real
    /// Gemma-3-4B tier, for the pool to replay through `_submitBlock` → keryxd `verify_pom_proof`
    /// in isolation. The proof is verified LOCALLY first, so this is a known-good vector. Writes
    /// `<KERYX_SAMPLE_OUT>_submit.json` + `_vector.txt` (default prefix /tmp/pom_sample).
    /// Run: KERYX_GEMMA_GGUF=… cargo test --features pom-opencl emit_sample_submit_wire -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(feature = "pom-opencl")]
    fn emit_sample_submit_wire() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        crate::pom_opencl::load_tier(&path, 0).expect("load_tier(real Gemma)");
        let (idx, tier) = active_index().expect("index installed by load_tier");
        // Deterministic, clearly-synthetic header inputs — NOT a real chain header. The pool feeds
        // these (pph, timestamp, nonce, tier, target) into verify_pom_proof / a synthetic header.
        let pph = blake(b"keryx-pom-sample-submit-wire-v1");
        let time = 1_700_000_000u64;
        let mut target = [0xffu8; 32]; // easy share target -> GPU finds a winner within a few batches
        target[24..32].copy_from_slice(&0x0010_0000_0000_0000u64.to_le_bytes());
        let mut base = 0u64;
        let mut found = None;
        for _ in 0..1024 {
            if let Some(n) = crate::pom_opencl::mine(&pph, time, &target, base, 1 << 16, false) {
                found = Some(n);
                break;
            }
            base = base.wrapping_add(1 << 16);
        }
        let nonce = found.expect("GPU found no winner over the real tier");
        let seed = pom_block_seed(&pph, time, nonce, false);
        let final_state = walk_final(seed, idx.n_chunks, POM_WALK_STEPS, |o| idx.read_chunk(o));
        let pow_value = pom_pow_value(final_state, &pph, false);
        assert!(le_leq(&pow_value, &target), "pow_value must satisfy the share target");
        let proof = build_proof(
            *tier, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS,
            |o| idx.read_chunk(o), |o| idx.merkle_path(o), false,
        );
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target, false),
            "sample proof MUST verify locally before handoff"
        );
        let proof_bytes = borsh::to_vec(&proof).expect("borsh");
        let proof_hex = hex::encode(&proof_bytes);
        let nonce_hex = format!("{:016x}", nonce);
        let opoi_tag = keryx_inference::tag_fixed(nonce);
        // Placeholder worker — NOT a real wallet (the live miner fills the real address).
        let worker = "keryx:SAMPLE_WORKER_PLACEHOLDER.amd-pom";
        let job_id = "sample-job-1";
        let submit = format!(
            r#"{{"id":1,"method":"mining.submit","params":["{}","{}","{}","{}","","{}"]}}"#,
            worker, job_id, nonce_hex, opoi_tag, proof_hex
        );
        let out = std::env::var("KERYX_SAMPLE_OUT").unwrap_or_else(|_| "/tmp/pom_sample".into());
        std::fs::write(format!("{out}_submit.json"), &submit).unwrap();
        let vector = format!(
            "PoM sample verification vector — tier {tier}, REAL Gemma-3-4B (verify_proof: PASS)\n\
             pre_pow_hash (32B hex): {pph}\n\
             timestamp (u64):        {time}\n\
             nonce (u64):            {nonce}   (nonceHex {nonce_hex})\n\
             tier (u8):              {tier}\n\
             target (32B LE hex):    {target}\n\
             pom_pow_value (32B hex):{powv}   (<= target ✓)\n\
             n_chunks:               {nc}\n\
             R_T tier root (hex):    {rt}\n\
             proof bytes:            {plen}   (params[5] hex chars {phlen})\n\
             submit params layout:   [worker, jobId, nonceHex, opoiTag, ipfsCID(\"\"), pomProofHex]\n\
             NOTE: worker is a placeholder; pph/time/nonce are synthetic test inputs (not a chain header).\n",
            tier = *tier, pph = hex::encode(pph), time = time, nonce = nonce, nonce_hex = nonce_hex,
            target = hex::encode(target), powv = hex::encode(pow_value), nc = idx.n_chunks,
            rt = hex::encode(idx.r_t), plen = proof_bytes.len(), phlen = proof_hex.len(),
        );
        std::fs::write(format!("{out}_vector.txt"), &vector).unwrap();
        eprintln!("{vector}");
        eprintln!("submit JSON ({} bytes) -> {out}_submit.json", submit.len());
    }

    /// Mode B: build a proof bound to a REAL staging header's pre_pow_hash + timestamp (supplied via
    /// env), so the pool can reconstruct an RpcRawBlock and submit it to keryxd. Mines at an easy
    /// test target (network diff is infeasible here), verifies locally, and writes the
    /// `{nonce_u64_dec, pom_proof_hex_lowercase, notes}` reply JSON.
    /// Run: KERYX_GEMMA_GGUF=… KERYX_POM_B_PPH=<64hex> KERYX_POM_B_TIME=<u64> \
    ///      cargo test --release -p keryx-miner-supr --features pom-opencl emit_mode_b_proof -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(feature = "pom-opencl")]
    fn emit_mode_b_proof() {
        let path = std::env::var("KERYX_GEMMA_GGUF").expect("set KERYX_GEMMA_GGUF");
        let pph_v = hex::decode(std::env::var("KERYX_POM_B_PPH").expect("set KERYX_POM_B_PPH (64 hex)").trim())
            .expect("KERYX_POM_B_PPH must be hex");
        assert_eq!(pph_v.len(), 32, "pre_pow_hash must be 32 bytes");
        let mut pph = [0u8; 32];
        pph.copy_from_slice(&pph_v);
        let time: u64 = std::env::var("KERYX_POM_B_TIME").expect("set KERYX_POM_B_TIME").trim().parse().expect("u64");

        crate::pom_opencl::load_tier(&path, 0).expect("load_tier(real Gemma)");
        let (idx, tier) = active_index().expect("index installed");
        let mut target = [0xffu8; 32]; // easy test target — finds a winner in a few batches
        target[24..32].copy_from_slice(&0x0010_0000_0000_0000u64.to_le_bytes());
        let mut base = 0u64;
        let mut found = None;
        for _ in 0..2048 {
            if let Some(n) = crate::pom_opencl::mine(&pph, time, &target, base, 1 << 16, false) {
                found = Some(n);
                break;
            }
            base = base.wrapping_add(1 << 16);
        }
        let nonce = found.expect("GPU found no winner");
        let seed = pom_block_seed(&pph, time, nonce, false);
        let final_state = walk_final(seed, idx.n_chunks, POM_WALK_STEPS, |o| idx.read_chunk(o));
        let pow_value = pom_pow_value(final_state, &pph, false);
        assert!(le_leq(&pow_value, &target), "pow_value must satisfy the easy target");
        let proof = build_proof(
            *tier, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS,
            |o| idx.read_chunk(o), |o| idx.merkle_path(o), false,
        );
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target, false),
            "Mode B proof MUST verify locally before handoff"
        );
        let proof_hex = hex::encode(borsh::to_vec(&proof).expect("borsh"));
        let notes = format!(
            "bound to real staging pre_pow_hash {pph} + timestamp {time}; tier {tier}; mined at EASY test \
             target {tgt} (NOT network bits — infeasible here); pom_pow_value {powv} (<= test target); \
             verify_proof PASS locally; R_T {rt}. Pre-fork the daemon won't call verify_pom_proof, so expect \
             InvalidPoW/LowDiff on kHeavyHash = GREEN (wire clean). For an override-verify green, pass THIS \
             easy target to verify_pom_proof (not the header bits).",
            pph = hex::encode(pph), time = time, tier = *tier, tgt = hex::encode(target),
            powv = hex::encode(pow_value), rt = hex::encode(idx.r_t),
        );
        let json = format!(
            "{{\n  \"nonce_u64_dec\": \"{}\",\n  \"pom_proof_hex_lowercase\": \"{}\",\n  \"notes\": \"{}\"\n}}\n",
            nonce, proof_hex, notes,
        );
        let out = std::env::var("KERYX_POM_B_OUT").unwrap_or_else(|_| "/tmp/pom_mode_b".into());
        std::fs::write(format!("{out}.json"), &json).unwrap();
        eprintln!("Mode B: nonce {nonce} ({} hex-char proof) -> {out}.json", proof_hex.len());
        eprintln!("{notes}");
    }

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
