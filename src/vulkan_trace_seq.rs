//! Vulkan sequence leaves and replay captures in the node's committed layout.
//! Adapted with changes from shmutalov/keryx-miner-rdna3's llm_engine.rs under
//! the existing Apache-2.0 project license (see LICENSE-APACHE).
//! Readback is supplied by our scoped Vulkan engine, never an implicit device 0.
use anyhow::{anyhow, Result};
use crate::trace::SeqCapture;
type SeqLayer = [u32; 5];
const TRACE_UBATCH: u32 = 8;
const GGML_Q8_0: u32 = 8;

/// One attention layer of the sequence layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqAttn {
    pub layer: u32,
    pub slot: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
}

/// One recurrent (gated delta-rule) layer of the sequence layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqRecur {
    pub layer: u32,
    pub slot: u32,
    /// Sequence layer kind: 3 = one decay per head, 4 = one per key dimension.
    pub kind: u32,
    pub n_v_heads: u32,
    pub head_dim: u32,
}

/// Sequence layer kind of a delta-rule layer with one decay per head (`RECUR_GDA`).
pub const SEQ_KIND_GDA: u32 = 3;

/// The sequence leaves a manifest under the attention rule commits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqSpec {
    pub n_attn: u32,
    pub n_recur: u32,
    /// Heads per layer in the leaf layout (`seq_heads`).
    pub heads: u32,
    pub attn: Vec<SeqAttn>,
    pub recur: Vec<SeqRecur>,
    /// Positions between recurrent state checkpoints.
    pub interval: u32,
}

impl SeqSpec {
    pub fn n_leaves(&self) -> usize {
        let h = self.heads as usize;
        2 * h * (self.n_attn + self.n_recur) as usize + h * self.n_attn as usize
    }

    /// Layers and query heads of the mass capture.
    pub(crate) fn lse_shape(&self) -> (u32, u32) {
        (self.attn.iter().map(|a| a.layer + 1).max().unwrap_or(0), self.attn.iter().map(|a| a.n_head).max().unwrap_or(0))
    }

    /// Bytes of one record of a recurrent head (`seq::record_len` of a one-decay layer).
    pub(crate) fn record_len(&self) -> usize {
        self.recur.first().map_or(0, |r| 8 * r.head_dim as usize + 8)
    }

    /// The backend's rows of the recurrent layers: (layer, kind, slot, heads, head_dim).
    pub(crate) fn recur_rows(&self) -> Vec<SeqLayer> {
        self.recur.iter().map(|r| [r.layer, r.kind, r.slot, r.n_v_heads, r.head_dim]).collect()
    }

    /// Whether this engine can commit the layout: grouped-query attention layers, and delta-rule
    /// layers with one decay per head, alike in heads and dimension (the backend's record
    /// capture), checkpointed on traced batch ends.
    pub fn check(&self) -> Result<()> {
        if self.attn.len() != self.n_attn as usize || self.recur.len() != self.n_recur as usize || self.heads == 0 {
            return Err(anyhow!("inconsistent sequence layout"));
        }
        for (i, a) in self.attn.iter().enumerate() {
            if a.slot as usize != i || a.n_head_kv == 0 || a.n_head_kv > self.heads || a.n_head % a.n_head_kv != 0 || a.head_dim % 32 != 0 {
                return Err(anyhow!("unsupported attention layer {}", a.layer));
            }
        }
        if let Some(first) = self.recur.first() {
            if self.interval == 0 || self.interval % TRACE_UBATCH != 0 {
                return Err(anyhow!("recurrent checkpoints every {} positions fall inside traced batches", self.interval));
            }
            for (i, r) in self.recur.iter().enumerate() {
                if r.slot as usize != i || r.kind != SEQ_KIND_GDA || r.n_v_heads != first.n_v_heads || r.head_dim != first.head_dim {
                    return Err(anyhow!("unsupported recurrent layer {}", r.layer));
                }
            }
            if first.n_v_heads == 0 || first.n_v_heads > self.heads || first.head_dim % 32 != 0 || first.head_dim > 128 {
                return Err(anyhow!("unsupported recurrent heads ({} of {})", first.n_v_heads, first.head_dim));
            }
        }
        Ok(())
    }
}

/// The recurrent sequence leaves of a traced evaluation, built batch by batch from the records
/// the backend takes after every gated-delta-net node (`keryx_rec.comp`): each head's records of
/// every position end to end, and its states after every whole checkpoint interval. Every leaf is
/// hashed as it grows; the leaves named at creation are also kept whole (a replay's spans).
pub(crate) struct RecurSink {
    n_recur: usize,
    heads: usize,
    seq_heads: usize,
    n_attn: usize,
    rec_len: usize,
    state_len: usize,
    interval: usize,
    rec: Vec<blake3::Hasher>,
    ckpt: Vec<blake3::Hasher>,
    keep: Vec<(u32, Vec<u8>)>,
    buf_rec: Vec<u8>,
    buf_ckpt: Vec<u8>,
    pos: usize,
    n_ckpt: usize,
}

impl RecurSink {
    /// A sink for `spec`'s recurrent leaves (`None` without any), keeping the leaves `keep` whole.
    pub(crate) fn new(spec: &SeqSpec, keep: &[u32]) -> Option<Self> {
        let first = *spec.recur.first()?;
        let (n_recur, heads, hd) = (spec.recur.len(), first.n_v_heads as usize, first.head_dim as usize);
        let rec_len = spec.record_len();
        let state_len = hd * hd * 4;
        let mut sink = Self {
            n_recur,
            heads,
            seq_heads: spec.heads as usize,
            n_attn: spec.n_attn as usize,
            rec_len,
            state_len,
            interval: spec.interval as usize,
            rec: vec![blake3::Hasher::new(); n_recur * heads],
            ckpt: vec![blake3::Hasher::new(); n_recur * heads],
            keep: Vec::new(),
            buf_rec: vec![0u8; n_recur * heads * TRACE_UBATCH as usize * rec_len],
            buf_ckpt: vec![0u8; n_recur * heads * state_len],
            pos: 0,
            n_ckpt: 0,
        };
        let mut leaves: Vec<u32> = keep.iter().copied().filter(|&l| sink.slot_head(l).is_some()).collect();
        leaves.sort_unstable();
        leaves.dedup();
        sink.keep = leaves.into_iter().map(|l| (l, Vec::new())).collect();
        Some(sink)
    }

    /// The (record?, slot * heads + head) of sequence leaf `leaf`, if it is a written recurrent leaf.
    fn slot_head(&self, leaf: u32) -> Option<(bool, usize)> {
        let base = 2 * self.seq_heads * self.n_attn;
        let i = (leaf as usize).checked_sub(base)?;
        let (sh, ck) = (i / 2, i % 2 == 1);
        let (slot, h) = (sh / self.seq_heads, sh % self.seq_heads);
        (slot < self.n_recur && h < self.heads).then_some((!ck, slot * self.heads + h))
    }

    /// Takes the records (and checkpoint) of the batch of `n_tokens` just decoded.
    pub(crate) fn after_batch_with(
        &mut self,
        n_tokens: usize,
        read: impl FnOnce(&mut [u8], &mut [u8]) -> Result<(u32, u32)>,
    ) -> Result<()> {
        if n_tokens == 0 || n_tokens > TRACE_UBATCH as usize {
            return Err(anyhow!("invalid recurrent batch size {n_tokens}"));
        }
        let (n, has_ckpt) = read(&mut self.buf_rec, &mut self.buf_ckpt)?;
        if n as usize != n_tokens {
            return Err(anyhow!("{n} recurrent records for a batch of {n_tokens}"));
        }
        self.pos += n_tokens;
        let due = self.pos % self.interval == 0;
        if (has_ckpt != 0) != due {
            return Err(anyhow!("recurrent checkpoint {} at position {}", if due { "missing" } else { "unexpected" }, self.pos));
        }
        let per = n_tokens * self.rec_len;
        for (i, hasher) in self.rec.iter_mut().enumerate() {
            hasher.update(&self.buf_rec[i * per..(i + 1) * per]);
        }
        if due {
            self.n_ckpt += 1;
            for (i, hasher) in self.ckpt.iter_mut().enumerate() {
                hasher.update(&self.buf_ckpt[i * self.state_len..(i + 1) * self.state_len]);
            }
        }
        for k in 0..self.keep.len() {
            let (is_rec, i) = self.slot_head(self.keep[k].0).expect("kept leaves are recurrent");
            let bytes = if is_rec {
                &self.buf_rec[i * per..(i + 1) * per]
            } else if due {
                &self.buf_ckpt[i * self.state_len..(i + 1) * self.state_len]
            } else {
                continue;
            };
            self.keep[k].1.extend_from_slice(bytes);
        }
        Ok(())
    }

    /// Checks the evaluation covered `n_pos` positions.
    pub(crate) fn check_end(&self, n_pos: usize) -> Result<()> {
        if self.pos != n_pos || self.n_ckpt != n_pos / self.interval {
            return Err(anyhow!("recurrent records of {} positions ({} checkpoints) for {n_pos}", self.pos, self.n_ckpt));
        }
        Ok(())
    }

    /// The hashes of the recurrent leaves into `leaves` (every sequence leaf, in order).
    pub(crate) fn hashes_into(&self, leaves: &mut [[u8; 32]]) {
        let base = 2 * self.seq_heads * self.n_attn;
        for slot in 0..self.n_recur {
            for h in 0..self.heads {
                let (i, leaf) = (slot * self.heads + h, base + 2 * (slot * self.seq_heads + h));
                leaves[leaf] = *self.rec[i].finalize().as_bytes();
                leaves[leaf + 1] = *self.ckpt[i].finalize().as_bytes();
            }
        }
    }

    /// The whole bytes of kept leaf `leaf`.
    pub(crate) fn kept(&self, leaf: u32) -> Option<&[u8]> {
        self.keep.iter().find(|(l, _)| *l == leaf).map(|(_, b)| b.as_slice())
    }
}

/// The q8_0 key and value cache rows of one layer, every position end to end.
pub(crate) struct LayerCache {
    k: Vec<u8>,
    v: Vec<u8>,
    k_row: usize,
    v_row: usize,
}

/// Parses the llama.cpp state of sequence 0 (`llama_kv_cache::state_write`): one stream, one range
/// of cells holding positions `0..n_pos` in order, unrotated q8_0 rows, value rows untransposed.
/// The cache holds the attention layers only, in layer order. With `hybrid` the recurrent memory's
/// state follows (`llama_memory_hybrid::state_write`) and is skipped. A cell of an M-RoPE model
/// carries its 2D position and token (`llama_kv_cell_ext`, 12 bytes) after its sequence count; the
/// state does not say so, and only one of the two readings holds positions `0..n_pos` in order.
pub(crate) fn parse_kv_state(b: &[u8], n_pos: usize, hybrid: bool) -> Result<Vec<LayerCache>> {
    parse_kv_state_ext(b, n_pos, hybrid, 0).or_else(|e| parse_kv_state_ext(b, n_pos, hybrid, 12).map_err(|_| e))
}

fn parse_kv_state_ext(b: &[u8], n_pos: usize, hybrid: bool, cell_ext: usize) -> Result<Vec<LayerCache>> {
    let mut o = 0usize;
    let mut take = |n: usize| -> Result<&[u8]> {
        let end = o.checked_add(n).ok_or_else(|| anyhow!("cache state offset overflow"))?;
        let s = b.get(o..end).ok_or_else(|| anyhow!("the cache state ends early"))?;
        o = end;
        Ok(s)
    };
    let u32_at = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
    // `llama_context::state_seq_get_data`: its magic and the sequence id, then the cache's state.
    let (magic, seq) = (u32_at(take(4)?), u32_at(take(4)?));
    if magic != 0xaf14_3cd8 || seq != 0 {
        return Err(anyhow!("not a state of sequence 0 (magic {magic:#x}, sequence {seq})"));
    }
    if u32_at(take(4)?) != 1 {
        return Err(anyhow!("the cache has several streams"));
    }
    let cells = u32_at(take(4)?) as usize;
    if cells != n_pos {
        return Err(anyhow!("the cache holds {cells} cells for {n_pos} positions"));
    }
    for i in 0..cells {
        let pos = u32_at(take(4)?);
        let n_seq = u32_at(take(4)?);
        take(cell_ext)?;
        let seq = if n_seq == 1 { u32_at(take(4)?) } else { u32::MAX };
        if pos as usize != i || seq != 0 {
            return Err(anyhow!("cache cell {i} holds position {pos} of {n_seq} sequences"));
        }
    }
    let v_trans = u32_at(take(4)?);
    let n_layer = u32_at(take(4)?) as usize;
    if n_layer > 4096 { return Err(anyhow!("invalid cache layer count")); }
    let (rot_k, rot_v) = (u32_at(take(4)?), u32_at(take(4)?));
    if v_trans != 0 || rot_k != 0 || rot_v != 0 {
        return Err(anyhow!("the cache is transposed or rotated (v_trans {v_trans}, rotation {rot_k}/{rot_v})"));
    }
    let mut layers: Vec<LayerCache> = Vec::with_capacity(n_layer);
    for which in 0..2 {
        for l in 0..n_layer {
            let ty = u32_at(take(4)?);
            let row = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
            if ty != GGML_Q8_0 {
                return Err(anyhow!("cache layer {l} is not q8_0 (type {ty})"));
            }
            let rows = take(row.checked_mul(cells).ok_or_else(|| anyhow!("cache row size overflow"))?)?.to_vec();
            if which == 0 {
                layers.push(LayerCache { k: rows, v: Vec::new(), k_row: row, v_row: 0 });
            } else {
                layers[l].v = rows;
                layers[l].v_row = row;
            }
        }
    }
    if o != b.len() && !hybrid {
        return Err(anyhow!("{} bytes past the cache state", b.len() - o));
    }
    Ok(layers)
}

/// The bytes of every sequence leaf of `spec`, in leaf order, from the cache rows and the masses
/// (`[layers][n_pos][heads]`, as the backend reads them back).
pub(crate) fn seq_leaf_bytes(spec: &SeqSpec, cache: &[LayerCache], lse: &[f32], n_pos: usize) -> Result<Vec<Vec<u8>>> {
    let (lse_layers, lse_heads) = spec.lse_shape();
    let (lse_layers, lse_heads) = (lse_layers as usize, lse_heads as usize);
    if lse.len() != lse_layers * n_pos * lse_heads {
        return Err(anyhow!("{} masses for {lse_layers} layers of {n_pos} positions", lse.len()));
    }
    let heads = spec.heads as usize;
    let mut out = vec![Vec::new(); spec.n_leaves()];
    // The cache holds the attention layers in layer order: the rank of a layer among them is its slot.
    let by_slot = cache.len() == spec.attn.len();
    for a in &spec.attn {
        let at = if by_slot { a.slot } else { a.layer } as usize;
        let lc = cache.get(at).ok_or_else(|| anyhow!("no cache for layer {}", a.layer))?;
        let hb = a.head_dim as usize / 32 * 34;
        let nkv = a.n_head_kv as usize;
        if lc.k_row != nkv * hb || lc.v_row != nkv * hb {
            return Err(anyhow!("cache rows of layer {} are {}/{} bytes for {nkv} heads of {hb}", a.layer, lc.k_row, lc.v_row));
        }
        let g = (a.n_head / a.n_head_kv) as usize;
        let slot = a.slot as usize;
        for h in 0..nkv {
            let kl = 2 * (slot * heads + h);
            let mut k = Vec::with_capacity(n_pos * hb);
            let mut v = Vec::with_capacity(n_pos * hb);
            let mut m = Vec::with_capacity(n_pos * g * 4);
            for t in 0..n_pos {
                k.extend_from_slice(&lc.k[t * lc.k_row + h * hb..t * lc.k_row + (h + 1) * hb]);
                v.extend_from_slice(&lc.v[t * lc.v_row + h * hb..t * lc.v_row + (h + 1) * hb]);
                for qi in 0..g {
                    let x = lse[(a.layer as usize * n_pos + t) * lse_heads + h * g + qi];
                    if !x.is_finite() {
                        return Err(anyhow!("no softmax mass for layer {} position {t} head {}", a.layer, h * g + qi));
                    }
                    m.extend_from_slice(&x.to_le_bytes());
                }
            }
            out[kl] = k;
            out[kl + 1] = v;
            out[2 * heads * (spec.n_attn + spec.n_recur) as usize + slot * heads + h] = m;
        }
    }
    Ok(out)
}

/// The capture of `[lo, hi)` of a sequence leaf: those bytes and the chaining value of every chunk.
pub fn seq_capture(leaf: &[u8], lo: u64, hi: u64) -> Result<SeqCapture> {
    use keryx_verifiable_inference::merkle::{b3_chunk_cv, B3_CHUNK};
    let bytes = leaf.get(lo as usize..hi as usize).ok_or_else(|| anyhow!("span {lo}..{hi} past a leaf of {} bytes", leaf.len()))?.to_vec();
    let cvs = if leaf.len() <= B3_CHUNK { vec![*blake3::hash(leaf).as_bytes()] } else { leaf.chunks(B3_CHUNK).enumerate().map(|(i, c)| b3_chunk_cv(c, i as u64)).collect() };
    Ok(SeqCapture { bytes, cvs })
}


#[cfg(test)]
mod tests {
    use super::*;
    fn kv_state(cells: u32, layers: &[(Vec<u8>, Vec<u8>)], row: u64) -> Vec<u8> {
        kv_state_cells(cells, layers, row, false)
    }

    /// [`kv_state`] whose cells carry the M-RoPE extension (2D position, token) when `ext`.
    fn kv_state_cells(cells: u32, layers: &[(Vec<u8>, Vec<u8>)], row: u64, ext: bool) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [0xaf14_3cd8u32, 0, 1, cells] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for i in 0..cells {
            let cell: &[u32] = if ext { &[i, 1, 0, 0, 1000 + i, 0] } else { &[i, 1, 0] };
            for v in cell {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in [0u32, layers.len() as u32, 0, 0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for which in 0..2 {
            for (k, v) in layers {
                b.extend_from_slice(&(GGML_Q8_0).to_le_bytes());
                b.extend_from_slice(&row.to_le_bytes());
                b.extend_from_slice(if which == 0 { k } else { v });
            }
        }
        b
    }

    /// Sequence leaves are the per-head cache rows of every position end to end and the masses of
    /// each head's query group, in the node's leaf order.
    #[test]
    fn sequence_leaves_follow_the_node_layout() {
        let (n_pos, hb) = (3usize, 34usize);
        let spec = SeqSpec { n_attn: 1, n_recur: 0, heads: 2, attn: vec![SeqAttn { layer: 0, slot: 0, n_head: 4, n_head_kv: 2, head_dim: 32 }], recur: vec![], interval: 0 };
        spec.check().unwrap();
        let k: Vec<u8> = (0..n_pos * 2 * hb).map(|i| i as u8).collect();
        let v: Vec<u8> = (0..n_pos * 2 * hb).map(|i| (i as u8).wrapping_mul(7)).collect();
        let state = kv_state(n_pos as u32, &[(k.clone(), v.clone())], 2 * hb as u64);
        let cache = parse_kv_state(&state, n_pos, false).unwrap();
        assert!(parse_kv_state(&state, n_pos + 1, false).is_err());
        assert!(parse_kv_state(&state[..state.len() - 1], n_pos, false).is_err());
        // A hybrid memory's recurrent state follows the cache.
        let mut hybrid = state.clone();
        hybrid.extend_from_slice(&[1, 2, 3]);
        assert!(parse_kv_state(&hybrid, n_pos, false).is_err());
        assert_eq!(parse_kv_state(&hybrid, n_pos, true).unwrap().len(), 1);
        // An M-RoPE cache's cells carry their 2D position and token.
        let mrope = kv_state_cells(n_pos as u32, &[(k.clone(), v.clone())], 2 * hb as u64, true);
        let ext = parse_kv_state(&mrope, n_pos, false).unwrap();
        assert_eq!((ext[0].k.as_slice(), ext[0].v.as_slice()), (cache[0].k.as_slice(), cache[0].v.as_slice()));
        let lse: Vec<f32> = (0..n_pos * 4).map(|i| i as f32 * 0.5).collect();
        let leaves = seq_leaf_bytes(&spec, &cache, &lse, n_pos).unwrap();
        assert_eq!(leaves.len(), spec.n_leaves());
        let layout = keryx_verifiable_inference::seq::SeqLayout { n_attn: 1, n_recur: 0, heads: 2 };
        for h in 0..2usize {
            let want_k: Vec<u8> = (0..n_pos).flat_map(|t| k[t * 2 * hb + h * hb..t * 2 * hb + (h + 1) * hb].to_vec()).collect();
            let want_v: Vec<u8> = (0..n_pos).flat_map(|t| v[t * 2 * hb + h * hb..t * 2 * hb + (h + 1) * hb].to_vec()).collect();
            assert_eq!(leaves[layout.kcache(0, h as u64).unwrap() as usize], want_k);
            assert_eq!(leaves[layout.vcache(0, h as u64).unwrap() as usize], want_v);
            let want_m: Vec<u8> = (0..n_pos).flat_map(|t| (0..2).flat_map(move |qi| (((t * 4 + h * 2 + qi) as f32) * 0.5).to_le_bytes())).collect();
            assert_eq!(leaves[layout.lse(0, h as u64).unwrap() as usize], want_m);
        }
        // An unwritten mass is an error, never a value.
        let mut holes = lse.clone();
        holes[5] = f32::NAN;
        assert!(seq_leaf_bytes(&spec, &cache, &holes, n_pos).is_err());
        // A rotated cache is refused.
        let mut rotated = state.clone();
        let rot = 16 + 12 * n_pos + 8;
        rotated[rot..rot + 4].copy_from_slice(&64u32.to_le_bytes());
        assert!(parse_kv_state(&rotated, n_pos, false).is_err());
    }

    /// Recurrent leaves land where the node's layout puts each head's records and checkpoints,
    /// sized as it decodes them; layouts the record capture cannot take are refused.
    #[test]
    fn recurrent_leaves_follow_the_node_layout() {
        use keryx_verifiable_inference::manifest::RECUR_GDA;
        use keryx_verifiable_inference::seq::{record_len, SeqLayout};
        let recur = |layer, slot| SeqRecur { layer, slot, kind: SEQ_KIND_GDA, n_v_heads: 3, head_dim: 32 };
        let spec = SeqSpec {
            n_attn: 1,
            n_recur: 2,
            heads: 4,
            attn: vec![SeqAttn { layer: 1, slot: 0, n_head: 4, n_head_kv: 2, head_dim: 32 }],
            recur: vec![recur(0, 0), recur(2, 1)],
            interval: 16,
        };
        spec.check().unwrap();
        assert_eq!(spec.record_len(), record_len(RECUR_GDA, 32));
        assert_eq!(spec.recur_rows(), vec![[0, 3, 0, 3, 32], [2, 3, 1, 3, 32]]);
        let layout = SeqLayout { n_attn: 1, n_recur: 2, heads: 4 };
        let sink = RecurSink::new(&spec, &[layout.checkpoint(1, 2).unwrap() as u32, layout.kcache(0, 0).unwrap() as u32]).unwrap();
        assert_eq!(sink.keep.len(), 1, "only recurrent leaves are kept");
        for slot in 0..2u64 {
            for h in 0..4u64 {
                let (r, c) = (layout.record(slot, h).unwrap() as u32, layout.checkpoint(slot, h).unwrap() as u32);
                let i = (slot * 3 + h) as usize;
                let want = |rec: bool| (h < 3).then_some((rec, i));
                assert_eq!((sink.slot_head(r), sink.slot_head(c)), (want(true), want(false)), "slot {slot} head {h}");
            }
        }
        assert_eq!(sink.slot_head(layout.vcache(0, 3).unwrap() as u32), None);
        assert_eq!(sink.slot_head(layout.lse(0, 0).unwrap() as u32), None);
        // Nothing recorded yet: every written recurrent leaf is the hash of nothing; the others stay.
        let mut leaves = vec![[7u8; 32]; spec.n_leaves()];
        sink.hashes_into(&mut leaves);
        let empty = *blake3::hash(b"").as_bytes();
        assert_eq!(leaves[layout.record(1, 2).unwrap() as usize], empty);
        assert_eq!(leaves[layout.checkpoint(0, 0).unwrap() as usize], empty);
        assert_eq!(leaves[layout.record(0, 3).unwrap() as usize], [7u8; 32]);
        assert!(sink.check_end(0).is_ok() && sink.check_end(1).is_err());
        // Checkpoints inside a traced batch, key-wise decays, uneven layers: refused.
        assert!(SeqSpec { interval: 12, ..spec.clone() }.check().is_err());
        assert!(SeqSpec { recur: vec![recur(0, 0), SeqRecur { kind: 4, ..recur(2, 1) }], ..spec.clone() }.check().is_err());
        assert!(SeqSpec { recur: vec![recur(0, 0), SeqRecur { n_v_heads: 2, ..recur(2, 1) }], ..spec.clone() }.check().is_err());
        assert!(SeqSpec { n_recur: 3, ..spec.clone() }.check().is_err());
    }

    /// A capture holds the asked bytes and the chaining value of every chunk of its leaf, which
    /// the opening's range proof is built from.
    #[test]
    fn sequence_captures_prove_against_the_leaf() {
        use keryx_verifiable_inference::merkle::b3_root;
        let leaf: Vec<u8> = (0..5000u32).map(|i| (i * 13) as u8).collect();
        let c = seq_capture(&leaf, 1024, 3072).unwrap();
        assert_eq!(c.bytes, leaf[1024..3072]);
        assert_eq!(c.cvs.len(), 5);
        assert_eq!(b3_root(&c.cvs), *blake3::hash(&leaf).as_bytes());
        let small = seq_capture(&leaf[..100], 0, 100).unwrap();
        assert_eq!(small.cvs, vec![*blake3::hash(&leaf[..100]).as_bytes()]);
        assert!(seq_capture(&leaf, 4096, 6000).is_err());
    }

}
