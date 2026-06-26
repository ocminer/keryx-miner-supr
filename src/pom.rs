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

/// Canonical block seed = initial walk state. mix64-fold of (nonce, time, pre_pow_hash).
/// BYTE-IDENTICAL to `pom_mine.cu::pom_seed_fold` and the node's `pom_block_seed`.
pub fn pom_block_seed(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64) -> u64 {
    let p = pph_words(pre_pow_hash);
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ p[0]);
    s = mix64(s ^ p[1]);
    s = mix64(s ^ p[2]);
    s = mix64(s ^ p[3]);
    s
}

/// Canonical pow value (256-bit LE) = mix64-fold of (final_state, pre_pow_hash).
/// BYTE-IDENTICAL to `pom_mine.cu::pom_pow_fold` and the node's `pom_pow_value`.
pub fn pom_pow_value(final_state: u64, pre_pow_hash: &[u8; 32]) -> [u8; 32] {
    let p = pph_words(pre_pow_hash);
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
) -> Option<(u64, PomProof)> {
    for nonce in nonce_start..nonce_start.saturating_add(max_nonces) {
        let seed = pom_block_seed(pre_pow_hash, timestamp, nonce);
        let final_state = walk_final(seed, index.n_chunks, k, |o| index.read_chunk(o));
        if le_leq(&pom_pow_value(final_state, pre_pow_hash), target) {
            let proof = build_proof(tier, pre_pow_hash, nonce, seed, index.n_chunks, k, t, |o| index.read_chunk(o), |o| index.merkle_path(o));
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
    let pow_value = pom_pow_value(final_state, pre_pow_hash);

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
pub fn verify_proof(pre_pow_hash: &[u8; 32], nonce: u64, seed: u64, proof: &PomProof, n_chunks: u64, k: u32, t: usize, r_t: &[u8; 32], target: &[u8; 32]) -> bool {
    if proof.openings.len() != t {
        return false;
    }
    if pom_pow_value(proof.final_state, pre_pow_hash) != proof.pow_value {
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
pub struct WeightIndex {
    pub n_chunks: u64,
    pub r_t: [u8; 32],
    /// Raw 32 B chunk reader: GGUF-backed in production, RAM-backed in synthetic tests.
    chunks: ChunkSource,
    /// Disk-backed Merkle tree: all levels (level 0 = leaves … single-node root) concatenated in
    /// one file, so the 70B tree (~84 GB) need not fit in RAM. `merkle_path` reads ~log N sibling
    /// nodes via `pread`. Built once per PoM activation; deleted on drop.
    tree_file: File,
    tree_path: PathBuf,
    /// Per level: (byte offset of the level in `tree_file`, node count).
    level_offsets: Vec<(u64, u64)>,
}

impl Drop for WeightIndex {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.tree_path);
    }
}

impl WeightIndex {
    /// Build from a GGUF on disk (CPU dtoh of each tensor). The bytes are candle's exact quantized
    /// bytes — the same the miner serves in VRAM and the builder pinned in `R_T`. The Merkle tree
    /// is streamed to a temp file next to the GGUF (disk, never tmpfs) so big tiers don't OOM.
    pub fn build_from_gguf(path: &str) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let mut file = File::open(path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order

        // Tree temp file next to the GGUF (disk-backed; /tmp may be tmpfs = RAM).
        let dir = std::path::Path::new(path).parent().unwrap_or_else(|| std::path::Path::new("."));
        let tree_path = dir.join(format!("pom-tree-{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&tree_path); // clear a stale file from a crashed run
        let mut writer = BufWriter::new(
            OpenOptions::new().read(true).write(true).create(true).truncate(true)
                .open(&tree_path).map_err(candle_core::Error::wrap)?,
        );

        // Level 0: hash chunks → leaves (to disk). The raw chunks are NOT retained in RAM; instead
        // we record, per tensor, the canonical chunk index of its first chunk and that chunk's
        // absolute byte offset in the GGUF, so `read_chunk` can `pread` any chunk on demand.
        let mut table: Vec<(u64, u64)> = Vec::with_capacity(names.len());
        let mut n_chunks: u64 = 0;
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
                writer.write_all(&blake(chunk)).map_err(candle_core::Error::wrap)?;
                n_chunks += 1;
            }
        }
        if n_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM: model produced 0 chunks".into()));
        }

        // Independent read-only handle for on-demand chunk preads (the build handle is consumed).
        let gguf = File::open(path).map_err(candle_core::Error::wrap)?;
        finalize_disk_tree(writer, tree_path, n_chunks, ChunkSource::Gguf { file: gguf, table })
    }

    /// 32 B chunk at canonical index `off` (panics if out of range — `off < n_chunks`).
    pub fn read_chunk(&self, off: u64) -> [u64; CHUNK_WORDS] {
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
        chunk_to_words(&arr)
    }

    /// Inclusion path for chunk index `off`, reading each sibling from the on-disk tree (pread).
    /// Byte-identical to the in-RAM duplicate-last walk: an out-of-range sibling is the node itself.
    pub fn merkle_path(&self, off: u64) -> Vec<[u8; 32]> {
        let mut path = Vec::with_capacity(self.level_offsets.len());
        let mut idx = off;
        for &(loff, count) in &self.level_offsets[..self.level_offsets.len() - 1] {
            let sib_idx = if idx & 1 == 0 { idx + 1 } else { idx - 1 };
            let read_idx = if sib_idx < count { sib_idx } else { idx };
            let mut node = [0u8; 32];
            read_exact_at(&self.tree_file, &mut node, loff + read_idx * 32)
                .expect("PoM tree read");
            path.push(node);
            idx >>= 1;
        }
        path
    }
}

/// Reduce a tree whose level 0 (leaves) is already written to `writer`, streaming each higher level
/// to the same file (duplicate-last on odd levels), and assemble the `WeightIndex`. Shared by
/// `build_from_gguf` and tests so the disk reduction is exercised by the synthetic merkle_path tests.
fn finalize_disk_tree(
    mut writer: BufWriter<File>,
    tree_path: PathBuf,
    n_chunks: u64,
    chunks: ChunkSource,
) -> candle_core::Result<WeightIndex> {
    let mut level_offsets: Vec<(u64, u64)> = vec![(0, n_chunks)];
    loop {
        let (loff, count) = *level_offsets.last().unwrap();
        if count == 1 {
            break;
        }
        writer.flush().map_err(candle_core::Error::wrap)?;
        let next_off = loff + count * 32;
        let mut reader = BufReader::new(File::open(&tree_path).map_err(candle_core::Error::wrap)?);
        reader.seek(SeekFrom::Start(loff)).map_err(candle_core::Error::wrap)?;
        let mut next_count: u64 = 0;
        let (mut left, mut right) = ([0u8; 32], [0u8; 32]);
        let mut i: u64 = 0;
        while i < count {
            reader.read_exact(&mut left).map_err(candle_core::Error::wrap)?;
            if i + 1 < count {
                reader.read_exact(&mut right).map_err(candle_core::Error::wrap)?;
            } else {
                right = left; // duplicate-last
            }
            writer.write_all(&hash_pair(&left, &right)).map_err(candle_core::Error::wrap)?;
            next_count += 1;
            i += 2;
        }
        level_offsets.push((next_off, next_count));
    }
    writer.flush().map_err(candle_core::Error::wrap)?;
    let tree_file = writer.into_inner().map_err(|e| candle_core::Error::Msg(format!("PoM tree flush: {e}")))?;

    let (root_off, _) = *level_offsets.last().unwrap();
    let mut r_t = [0u8; 32];
    read_exact_at(&tree_file, &mut r_t, root_off).map_err(candle_core::Error::wrap)?;

    Ok(WeightIndex { n_chunks, r_t, chunks, tree_file, tree_path, level_offsets })
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

/// The active possession index + tier, if installed.
pub fn active_index() -> Option<&'static (WeightIndex, u8)> {
    POM_INDEX.get()
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
        let mut writer = BufWriter::new(
            OpenOptions::new().read(true).write(true).create(true).truncate(true)
                .open(&tree_path).unwrap(),
        );
        let mut data = Vec::new();
        for o in 0..n {
            let b = words_to_bytes(&synth_chunk(o));
            writer.write_all(&blake(&b)).unwrap();
            data.extend_from_slice(&b);
        }
        finalize_disk_tree(writer, tree_path, n, ChunkSource::Ram(data)).unwrap()
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

        // Build the tree over the canonical synth chunks, with the GGUF chunk source.
        let tree_path = std::env::temp_dir().join(format!("keryx-pom-fakegguf-tree-{uid}.bin"));
        let _ = std::fs::remove_file(&tree_path);
        let mut writer = BufWriter::new(
            OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tree_path).unwrap(),
        );
        for o in 0..n {
            writer.write_all(&blake(&words_to_bytes(&synth_chunk(o)))).unwrap();
        }
        let idx = finalize_disk_tree(writer, tree_path, n, ChunkSource::Gguf { file, table }).unwrap();

        // Every chunk read by pread matches the canonical chunk, across all segments + padding.
        for o in 0..n {
            assert_eq!(idx.read_chunk(o), synth_chunk(o), "chunk {o}");
        }
        // A proof built from the GGUF source verifies against R_T (target 0xff..ff = first nonce wins).
        let (k, t) = (POM_WALK_STEPS, POM_OPENINGS);
        let pph = [7u8; 32];
        let target = [0xffu8; 32];
        let (nonce, proof) = mine_pom(&idx, 2, &pph, 123, &target, k, t, 0, 1).expect("max target → win");
        let seed = pom_block_seed(&pph, 123, nonce);
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target));

        let _ = std::fs::remove_file(&gguf_path);
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
        let (nonce, proof) = mine_pom(&idx, 0, &pph, 99, &target, k, t, 0, 1).expect("max target → win");
        let seed = pom_block_seed(&pph, 99, nonce);
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target),
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
        let seed = pom_block_seed(&pph, 111, nonce);

        let proof = build_proof(2, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o));
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32]));
        // borsh wire-format round-trips (same encoding the node decodes).
        let bytes = borsh::to_vec(&proof).unwrap();
        let back: PomProof = borsh::from_slice(&bytes).unwrap();
        assert!(verify_proof(&pph, nonce, seed, &back, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32]));
        assert_eq!(back.tier, 2);
    }

    #[test]
    fn wrong_target_or_root_fails() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"pph2");
        let nonce = 7;
        let seed = pom_block_seed(&pph, 1, nonce);
        let proof = build_proof(0, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o));
        assert!(!verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0u8; 32]), "zero target must fail");
        assert!(!verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &blake(b"wrong"), &[0xff; 32]), "wrong R_T must fail");
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
        let (nonce, proof) = mine_pom(&idx, 1, &pph, ts, &target, k, t, 0, 100_000).expect("mine a nonce");
        let seed = pom_block_seed(&pph, ts, nonce);
        // The proof verifies against the same target the node would use.
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target));
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
        let seed = pom_block_seed(&pph, 99, nonce);
        let proof = build_proof(0, &pph, nonce, seed, idx.n_chunks, 256, 32, |o| idx.read_chunk(o), |o| idx.merkle_path(o));
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, 256, 32, &idx.r_t, &[0xff; 32]));
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
            if let Some(n) = crate::pom_opencl::mine(&pph, time, &target, base, 1 << 16) {
                found = Some(n);
                break;
            }
            base = base.wrapping_add(1 << 16);
        }
        let nonce = found.expect("GPU found no winner over the real tier");
        let seed = pom_block_seed(&pph, time, nonce);
        let proof = build_proof(*tier, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, |o| idx.read_chunk(o), |o| idx.merkle_path(o));
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target),
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
        let gm = crate::pom_gpu::PomGpuMiner::load(&path).expect("load candle-CUDA gather");
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
        let seed = pom_block_seed(&pph, time, nonce);
        let proof = build_proof(0, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, |o| idx.read_chunk(o), |o| idx.merkle_path(o));
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target),
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
        let gm = crate::pom_gpu::PomGpuMiner::load(&path).expect("load candle-CUDA gather (Qwen3)");
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
        let seed = pom_block_seed(&pph, time, nonce);
        let proof = build_proof(2, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, |o| idx.read_chunk(o), |o| idx.merkle_path(o));
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target),
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
            if let Some(n) = crate::pom_opencl::mine(&pph, time, &target, base, 1 << 16) {
                found = Some(n);
                break;
            }
            base = base.wrapping_add(1 << 16);
        }
        let nonce = found.expect("GPU found no winner over the real tier");
        let seed = pom_block_seed(&pph, time, nonce);
        let final_state = walk_final(seed, idx.n_chunks, POM_WALK_STEPS, |o| idx.read_chunk(o));
        let pow_value = pom_pow_value(final_state, &pph);
        assert!(le_leq(&pow_value, &target), "pow_value must satisfy the share target");
        let proof = build_proof(
            *tier, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS,
            |o| idx.read_chunk(o), |o| idx.merkle_path(o),
        );
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target),
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
            if let Some(n) = crate::pom_opencl::mine(&pph, time, &target, base, 1 << 16) {
                found = Some(n);
                break;
            }
            base = base.wrapping_add(1 << 16);
        }
        let nonce = found.expect("GPU found no winner");
        let seed = pom_block_seed(&pph, time, nonce);
        let final_state = walk_final(seed, idx.n_chunks, POM_WALK_STEPS, |o| idx.read_chunk(o));
        let pow_value = pom_pow_value(final_state, &pph);
        assert!(le_leq(&pow_value, &target), "pow_value must satisfy the easy target");
        let proof = build_proof(
            *tier, &pph, nonce, seed, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS,
            |o| idx.read_chunk(o), |o| idx.merkle_path(o),
        );
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, POM_WALK_STEPS, POM_OPENINGS, &idx.r_t, &target),
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
