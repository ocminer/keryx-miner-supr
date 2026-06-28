//! Single-device fork of candle-transformers' `quantized_gemma3`, used by PoM
//! zero-dup. Gemma-3-4B is a NON-split GGUF (the baseline/Light tier), so the
//! stock inference path (`candle_transformers::models::quantized_gemma3`) loads
//! its own VRAM copy of every weight while the PoM possession walk loads a SECOND
//! copy — on an 8 GB card the two ~2.5 GB copies + KV-cache + PoW buffers OOM and
//! inference never completes ("OPoI inference in progress — PoW paused" forever).
//!
//! This fork is byte-for-byte the upstream gemma3 forward, with TWO additions that
//! let the possession walk SHARE the single resident inference copy (zero-dup):
//!   1. the big projection matrices are held as the PUBLIC `candle_core::quantized::
//!      QMatMul` (not upstream's private wrapper) so their raw quantized `QTensor`s
//!      can be handed to `pom_gpu::load_shared`;
//!   2. `pom_quant_tensors()` exposes those matrices keyed by canonical GGUF name.
//!
//! The dequantized-in-inference tensors (`token_embd`, the RMS norms) are NOT
//! returned — the PoM loader reads those raw separately (Option C: share the big
//! matrices, dup the small dequantized rest). Single-device == upstream behaviour
//! (no cross-device moves); the `from_gguf` signature matches upstream so it is a
//! drop-in for `Gemma3Weights::from_gguf`.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

// Gemma 3 supports a 128K context, but OPoI inference only ever runs short prompts
// (system + challenge) and caps generation at 2048 tokens, so the full 131072 RoPE
// table is pure VRAM waste. Sizing it to a generous 8192 (well above any OPoI
// sequence) shrinks the cos/sin tables ~16x — material on an 8 GB card where this
// fork must coexist with the resident PoM walk. The forward narrows by `index_pos +
// seq_len`, which never approaches this bound for OPoI.
pub const MAX_SEQ_LEN: usize = 8192;
pub const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
pub const DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR: f32 = 1.;

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul, // ffn_gate in GGUF
    feed_forward_up: QMatMul,   // ffn_up in GGUF
    feed_forward_down: QMatMul, // ffn_down in GGUF
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    // Attention components
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,

    // Specialized normalization for Q and K
    attention_q_norm: RmsNorm,
    attention_k_norm: RmsNorm,

    // Layer normalization
    attention_norm: RmsNorm,      // Applied before attention
    post_attention_norm: RmsNorm, // Applied after attention
    ffn_norm: RmsNorm,            // Applied before feedforward
    post_ffn_norm: RmsNorm,       // Applied after feedforward

    // Feed-forward network
    mlp: Mlp,

    // Attention parameters
    n_head: usize,    // Number of query heads
    n_kv_head: usize, // Number of key-value heads
    head_dim: usize,  // Dimension of each head
    q_dim: usize,     // Total dimension for queries

    sliding_window_size: Option<usize>,

    rotary_embedding: RotaryEmbedding,
    neg_inf: Tensor,

    // Cache
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i {
                            0u32
                        } else {
                            1u32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            let mask0 = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
            .to_dtype(dtype)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;

        let (q, k) = self
            .rotary_embedding
            .apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?; // concat on seq dim
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone())); // update cache

        // Repeat KV for GQA
        let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Scaled Dot-Product Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;

        self.attention_wo.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    /// True when the GGUF carries a distinct `output.weight`. Gemma usually ties
    /// the output head to `token_embd.weight` (no separate GGUF tensor), in which
    /// case `output` reuses the token_embd buffer and MUST NOT be advertised to the
    /// possession walk under the name `output.weight` (the gather, keyed by the
    /// GGUF's own tensor names, has no such entry).
    has_distinct_output: bool,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Detect architecture prefix by probing which keys exist in metadata.
        let prefix = ["gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma3");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle_core::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let _value_length = md_get("attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = md_get("rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        let _rope_freq_scaling_factor = md_get("rope.scaling.factor")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR);

        let q_dim = head_count * key_length;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Load token embeddings and output projection.
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let (output, has_distinct_output) = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => (tensor, true),
            // Tied weights: reuse token_embd if output.weight doesn't exist.
            Err(_) => (ct.tensor(reader, "token_embd.weight", device)?, false),
        };

        // Gemma 3 alternates sliding and full-attention layers, which use DIFFERENT RoPE
        // frequencies. Upstream builds a fresh MAX_SEQ_LEN cos/sin table per layer — on a
        // 34-layer 4B that is gigabytes of duplicated RoPE tensors. Only two distinct tables
        // exist (sliding vs full); build each once and clone the handle into the layers
        // (candle Tensor clone shares the same VRAM storage). Byte-identical, far smaller.
        let rope_full = RotaryEmbedding::new(key_length, rope_freq_base, device)?;
        let rope_sliding = RotaryEmbedding::new(key_length, rope_freq_base_sliding, device)?;

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?,
                rms_norm_eps,
            )?;

            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let feed_forward_gate =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let feed_forward_down =
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
            };

            // Sliding window pattern hardcoded to 6 because it's not explicitly defined.
            let is_sliding = (layer_idx + 1) % sliding_window_type > 0;
            let sliding_window_size = is_sliding.then_some(sliding_window_size);
            // Share the prebuilt RoPE table (clone = a handle to the same VRAM buffer).
            let rotary_embedding = if is_sliding {
                rope_sliding.clone()
            } else {
                rope_full.clone()
            };

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim,
                sliding_window_size,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
            })
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            has_distinct_output,
        })
    }

    /// PoM zero-dup support: the quantized weight matrices held resident in VRAM,
    /// keyed by their canonical GGUF name. These QMatMul-backed matrices keep
    /// candle's raw quantized bytes (== what `R_T` commits), so the possession walk
    /// can read them in place instead of loading a second copy. The dequantized
    /// tensors (`token_embd`, the RMS norms) are intentionally NOT returned — the
    /// PoM loader reads those raw separately. `output.weight` is only advertised
    /// when the GGUF actually carries it (Gemma ties it to `token_embd`, in which
    /// case the gather already covers it via the token_embd read).
    pub fn pom_quant_tensors(&self) -> HashMap<String, Arc<QTensor>> {
        fn inner(qmm: &QMatMul) -> Option<Arc<QTensor>> {
            match qmm {
                QMatMul::QTensor(t) => Some(t.clone()),
                _ => None,
            }
        }
        let mut m = HashMap::new();
        if self.has_distinct_output {
            if let Some(t) = inner(&self.output) {
                m.insert("output.weight".to_string(), t);
            }
        }
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("blk.{i}");
            for (name, qmm) in [
                (format!("{p}.attn_q.weight"), &l.attention_wq),
                (format!("{p}.attn_k.weight"), &l.attention_wk),
                (format!("{p}.attn_v.weight"), &l.attention_wv),
                (format!("{p}.attn_output.weight"), &l.attention_wo),
                (format!("{p}.ffn_gate.weight"), &l.mlp.feed_forward_gate),
                (format!("{p}.ffn_down.weight"), &l.mlp.feed_forward_down),
                (format!("{p}.ffn_up.weight"), &l.mlp.feed_forward_up),
            ] {
                if let Some(t) = inner(qmm) {
                    m.insert(name, t);
                }
            }
        }
        m
    }

    /// Reset every layer's KV cache. Must be called before each independent prompt
    /// so a new inference doesn't attend to the previous request's residual keys.
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache = None;
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;

        let mut layer_in = self.tok_embeddings.forward(x)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

        for layer in self.layers.iter_mut() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, x.dtype(), x.device())?)
            };

            // Attention block
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos)?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let x = (x + residual)?;

            layer_in = x;
        }

        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let output = self.output.forward(&x)?;

        Ok(output)
    }
}
