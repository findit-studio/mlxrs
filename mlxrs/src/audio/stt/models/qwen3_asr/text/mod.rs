//! Qwen3-ASR text decoder (`qwen3_asr.TextModel`) with multimodal RoPE.
//!
//! A standalone port of mlx-audio's `qwen3_asr.{TextModel, TextDecoderLayer,
//! TextAttention, TextMLP}` — the Qwen3-based decoder the Qwen3-ASR transcriber
//! and the forced aligner share. It mirrors the dense
//! [`Qwen3Model`](crate::lm::models::qwen3::Qwen3Model) (grouped-query attention
//! with per-head Q/K-norm, a SwiGLU MLP, pre-norm blocks, a final RMSNorm) but
//! is its **own** type because the released Qwen3-ASR `text_config` carries a
//! non-null MRoPE `rope_scaling` (`mrope_section`) that the dense Qwen3 config
//! rejects.
//!
//! Where the reference shares sub-components with dense Qwen3 it stays
//! structurally identical here — the MLP, the RMSNorms, and the GQA attention
//! scaffolding (per-head reshape, Q/K-norm, the fused
//! [`scaled_dot_product_attention`]). The **only** divergence is the rotary
//! application: instead of the fused base-RoPE primitive at a scalar offset, the
//! ASR decoder builds `cos`/`sin` from a 3-D `position_ids` tensor and the
//! configured `mrope_section`, then applies them as `q*cos +
//! rotate_half(q)*sin` ([`MRope`]). This follows how the Qwen2-VL /
//! Qwen3-Omni references structure multimodal-RoPE models as their own classes
//! rather than bolting MRoPE onto the standard decoder.
//!
//! For the forced aligner the positions are **text-only** (the same scalar
//! position on every MRoPE axis); the interleaved/chunked frequency selection
//! then collapses to the standard RoPE angles, so the decoder is numerically
//! identical to the dense Qwen3 decoder on those inputs while remaining correct
//! for genuine 3-D positions.

mod config;

use std::collections::HashMap;

use config::MROPE_AXES;
pub use config::{MRopeConfig, Qwen3AsrTextConfig};
use smol_str::format_smolstr;

use crate::{
  Dtype,
  array::Array,
  error::{
    Error, LayerKeyedPayload, LengthMismatchPayload, MissingKeyPayload, RankMismatchPayload,
    Result, ShapePairMismatchPayload,
  },
  lm::{
    cache::{KvCache, MaskMode, StandardKvCache},
    nn::{
      activations::swiglu,
      attention::{Mask, scaled_dot_product_attention},
      norm::RMSNorm,
    },
  },
  model_validation::reserve_or_error,
  ops::{
    indexing::{slice, take_axis},
    linalg_basic::matmul,
    shape::{broadcast_to, concatenate, expand_dims_axes, reshape, swapaxes, transpose_axes},
  },
};

// ───────────────────────────── linear helper ─────────────────────────────

/// `y = x @ weightᵀ` — a bias-free `nn.Linear` forward (every Qwen3 text
/// projection is bias-free). `weight` is the `(out, in)` `nn.Linear` layout.
fn linear(x: &Array, weight: &Array) -> Result<Array> {
  let wt = swapaxes(weight, -1, -2)?;
  matmul(x, &wt)
}

// ───────────────────────────── multimodal RoPE ─────────────────────────────

/// Multimodal rotary position embedding for the Qwen3-ASR text decoder
/// (`qwen3_asr.TextAttention.rope`, generalized to MRoPE).
///
/// Holds the precomputed inverse frequencies (`inv_freq[d] = base^(-2d/dim)` for
/// `d in [0, dim/2)`) and a per-frequency **axis selector** derived from the
/// `mrope_section`: slot `d` of the `dim/2` frequencies draws its position from
/// MRoPE axis `selector[d]` (temporal / height / width).
///
/// [`cos_sin`](Self::cos_sin) turns a `(3, B, L)` `position_ids` into the
/// `(B, L, dim)` `cos`/`sin`; [`apply`](Self::apply) rotates a
/// `(B, heads, L, dim)` query or key by `x*cos + rotate_half(x)*sin`.
#[derive(Debug)]
struct MRope {
  /// `(dim/2,)` inverse frequencies.
  inv_freq: Array,
  /// `(dim/2,)` int32 axis selector — which MRoPE axis each frequency slot
  /// reads its position from.
  selector: Array,
  /// The rotary dimension (`head_dim`).
  dim: i32,
}

impl MRope {
  /// Build the rotary from `head_dim`, the RoPE `base`, and the validated
  /// [`MRopeConfig`].
  fn new(dim: i32, base: f32, mrope: MRopeConfig) -> Result<Self> {
    let half = dim / 2;
    // inv_freq[d] = base^(-(2d)/dim) = exp(-(2d/dim) * ln(base)).
    // arange(0, dim, 2) / dim, then `base ** (-that)`.
    let idx = Array::arange::<f32>(0.0, f64::from(dim), 2.0)?;
    let scale = Array::full::<f32>(&[0i32; 0], 1.0 / f64::from(dim) as f32)?;
    let exponent = idx.multiply(&scale)?; // (dim/2,) = 2d/dim
    let neg_ln_base = Array::full::<f32>(&[0i32; 0], -(f64::from(base).ln() as f32))?;
    let inv_freq = exponent.multiply(&neg_ln_base)?.exp()?; // base^(-(2d/dim))

    let selector = Self::axis_selector(half, mrope);
    let selector = Array::from_slice::<i32>(&selector, &(half as usize,))?;
    Ok(Self {
      inv_freq,
      selector,
      dim,
    })
  }

  /// The per-frequency MRoPE axis selector over the `half = dim/2` frequency
  /// slots.
  ///
  /// - **interleaved** (Qwen3-Omni `[THTHWHTHW...TT]`): height takes slots `1,
  ///   4, 7, …` up to `section[1]*3`; width takes slots `2, 5, 8, …` up to
  ///   `section[2]*3`; temporal takes the rest.
  /// - **chunked** (Qwen2.5-VL `[TTT...HHH...WWW]`): the first `section[0]`
  ///   slots are temporal, the next `section[1]` height, the next `section[2]`
  ///   width.
  ///
  /// Mirrors `qwen3_tts.TalkerRotaryEmbedding.apply_interleaved_mrope` and
  /// `mlx_vlm.rope_utils` `_interleaved_position_selector` /
  /// `_chunked_position_selector`.
  fn axis_selector(half: i32, mrope: MRopeConfig) -> Vec<i32> {
    let half = half.max(0) as usize;
    let mut selector = vec![0i32; half];
    if mrope.interleaved {
      // axis 1 (height) at offset 1, axis 2 (width) at offset 2, step 3.
      for (axis, offset) in [(1usize, 1usize), (2usize, 2usize)] {
        let limit = (mrope.section[axis].max(0) as usize)
          .saturating_mul(3)
          .min(half);
        let mut idx = offset;
        while idx < limit {
          selector[idx] = axis as i32;
          idx += 3;
        }
      }
    } else {
      let mut offset = mrope.section[0].max(0) as usize;
      for axis in 1..MROPE_AXES {
        let len = mrope.section[axis].max(0) as usize;
        let end = offset.saturating_add(len).min(half);
        for slot in selector.iter_mut().take(end).skip(offset) {
          *slot = axis as i32;
        }
        offset = offset.saturating_add(len);
      }
    }
    selector
  }

  /// Build `(B, L, dim)` `cos`/`sin` from a `(3, B, L)` `position_ids`.
  ///
  /// For each frequency slot `d`, the position is read from the MRoPE axis
  /// `selector[d]`; `freqs[b, l, d] = position_ids[selector[d], b, l] *
  /// inv_freq[d]`; `emb = concat([freqs, freqs], -1)`; `cos`/`sin` of `emb`.
  fn cos_sin(&self, position_ids: &Array) -> Result<(Array, Array)> {
    // Select, per frequency slot, the position from its MRoPE axis:
    // take(position_ids, selector, axis=0) → (dim/2, B, L), then move the
    // frequency axis last → (B, L, dim/2).
    let selected = take_axis(position_ids, &self.selector, 0)?;
    let selected = selected.astype(Dtype::F32)?;
    let selected = transpose_axes(&selected, &[1, 2, 0])?; // (B, L, dim/2)

    // freqs = positions * inv_freq (broadcast inv_freq over (B, L, ·)).
    let freqs = selected.multiply(&self.inv_freq)?;
    let emb = concatenate(&[&freqs, &freqs], -1)?; // (B, L, dim)
    let cos = emb.cos()?;
    let sin = emb.sin()?;
    Ok((cos, sin))
  }

  /// Rotate `x` `(B, heads, L, dim)` by `x*cos + rotate_half(x)*sin`, with
  /// `cos`/`sin` `(B, L, dim)` broadcast over the head axis.
  ///
  /// `cos`/`sin` are computed in `f32` (see [`cos_sin`](Self::cos_sin)); they
  /// are cast to `x`'s dtype before the multiply so a `bf16`/`f16` query/key is
  /// not promoted to `f32` (`promote_types(bf16, f32) == f32`). This mirrors the
  /// fused `nn.RoPE` the reference applies, which computes its angles in `f32`
  /// internally yet returns the input dtype; the cast is a no-op when `x` is
  /// already `f32`.
  fn apply(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let dtype = x.dtype()?;
    // cos/sin → (B, 1, L, dim) for the head broadcast, in x's dtype.
    let cos = expand_dims_axes(cos, &[1])?.astype(dtype)?;
    let sin = expand_dims_axes(sin, &[1])?.astype(dtype)?;
    let rotated = self.rotate_half(x)?;
    let a = x.multiply(&cos)?;
    let b = rotated.multiply(&sin)?;
    a.add(&b)
  }

  /// `rotate_half(x) = concat([-x[..., dim/2:], x[..., :dim/2]], -1)` over the
  /// last (head_dim) axis — the non-traditional RoPE rotation.
  fn rotate_half(&self, x: &Array) -> Result<Array> {
    let rank = x.ndim();
    let half = self.dim / 2;
    let shape = x.shape();
    // Build full-rank start/stop/strides slicing only the last axis.
    let mut start = vec![0i32; rank];
    let mut stop: Vec<i32> = shape
      .iter()
      .map(|&d| i32::try_from(d).unwrap_or(i32::MAX))
      .collect();
    let strides = vec![1i32; rank];
    let last = rank - 1;

    // first half x[..., :half]
    let mut stop_first = stop.clone();
    stop_first[last] = half;
    let x1 = slice(x, &start, &stop_first, &strides)?;

    // second half x[..., half:]
    start[last] = half;
    stop[last] = self.dim;
    let x2 = slice(x, &start, &stop, &strides)?;

    let neg_x2 = x2.negative()?;
    concatenate(&[&neg_x2, &x1], -1)
  }
}

// ───────────────────────────── attention ─────────────────────────────

/// Grouped-query attention with per-head Q/K-norm and MRoPE
/// (`qwen3_asr.TextAttention`).
#[derive(Debug)]
struct Attention {
  n_heads: i32,
  n_kv_heads: i32,
  head_dim: i32,
  scale: f32,
  q_proj: Array,
  k_proj: Array,
  v_proj: Array,
  o_proj: Array,
  q_norm: RMSNorm,
  k_norm: RMSNorm,
}

impl Attention {
  /// `q/k/v = {q,k,v}_proj(x)`, per-head reshape, q/k RMSNorm over `head_dim`,
  /// MRoPE via the precomputed `cos`/`sin`, `cache.update`, then the fused SDPA
  /// and `o_proj`.
  fn forward(
    &self,
    x: &Array,
    rope: &MRope,
    cos: &Array,
    sin: &Array,
    mask: &MaskMode,
    cache: &mut StandardKvCache,
  ) -> Result<Array> {
    let shape = x.shape();
    let (b, l) = (shape[0] as i32, shape[1] as i32);
    let hd = self.head_dim;

    let queries = linear(x, &self.q_proj)?;
    let keys = linear(x, &self.k_proj)?;
    let values = linear(x, &self.v_proj)?;

    // Per-head reshape (B, L, n, head_dim), q/k RMSNorm over head_dim, transpose
    // to (B, n, L, head_dim).
    let queries = reshape(&queries, &[b, l, self.n_heads, hd])?;
    let queries = self.q_norm.forward(&queries)?;
    let queries = transpose_axes(&queries, &[0, 2, 1, 3])?;

    let keys = reshape(&keys, &[b, l, self.n_kv_heads, hd])?;
    let keys = self.k_norm.forward(&keys)?;
    let keys = transpose_axes(&keys, &[0, 2, 1, 3])?;

    let values = reshape(&values, &[b, l, self.n_kv_heads, hd])?;
    let values = transpose_axes(&values, &[0, 2, 1, 3])?;

    // MRoPE, then append+fetch the running K/V.
    let queries = rope.apply(&queries, cos, sin)?;
    let keys = rope.apply(&keys, cos, sin)?;
    let (keys, values) = cache.update(&keys, &values)?;

    let attn_mask = mask_mode_to_mask(mask);
    let output = scaled_dot_product_attention(&queries, &keys, &values, self.scale, attn_mask)?;
    let output = transpose_axes(&output, &[0, 2, 1, 3])?;
    let output = reshape(&output, &[b, l, self.n_heads * hd])?;
    linear(&output, &self.o_proj)
  }
}

/// Map a [`MaskMode`] to the attention [`Mask`] selector.
fn mask_mode_to_mask(mode: &MaskMode) -> Mask<'_> {
  match mode {
    MaskMode::None => Mask::None,
    MaskMode::Causal => Mask::Causal,
    MaskMode::Array(a) => Mask::Array(a),
  }
}

// ───────────────────────────── MLP ─────────────────────────────

/// Dense SwiGLU feed-forward (`qwen3_asr.TextMLP`):
/// `down_proj(silu(gate_proj(x)) * up_proj(x))` — structurally identical to the
/// dense Qwen3 MLP.
#[derive(Debug)]
struct Mlp {
  gate_proj: Array,
  up_proj: Array,
  down_proj: Array,
}

impl Mlp {
  fn forward(&self, x: &Array) -> Result<Array> {
    let gate = linear(x, &self.gate_proj)?;
    let up = linear(x, &self.up_proj)?;
    let act = swiglu(&gate, &up)?;
    linear(&act, &self.down_proj)
  }
}

// ───────────────────────────── decoder block ─────────────────────────────

/// A pre-norm Qwen3-ASR decoder block (`qwen3_asr.TextDecoderLayer`).
#[derive(Debug)]
struct TransformerBlock {
  self_attn: Attention,
  mlp: Mlp,
  input_layernorm: RMSNorm,
  post_attention_layernorm: RMSNorm,
}

impl TransformerBlock {
  fn forward(
    &self,
    x: &Array,
    rope: &MRope,
    cos: &Array,
    sin: &Array,
    mask: &MaskMode,
    cache: &mut StandardKvCache,
  ) -> Result<Array> {
    let r = self.self_attn.forward(
      &self.input_layernorm.forward(x)?,
      rope,
      cos,
      sin,
      mask,
      cache,
    )?;
    let hidden = x.add(&r)?;
    let ffn = self
      .mlp
      .forward(&self.post_attention_layernorm.forward(&hidden)?)?;
    hidden.add(&ffn)
  }
}

// ───────────────────────────── model ─────────────────────────────

/// The Qwen3-ASR head-less text decoder (`qwen3_asr.TextModel`): token
/// embedding, the per-layer MRoPE transformer blocks with a shared causal mask,
/// and the final RMSNorm. Its forward returns the normalized hidden states
/// `(B, L, hidden)` — no output projection.
///
/// This is the decoder the Qwen3 forced aligner runs under its
/// timestamp-classification head. It deliberately mirrors the dense
/// [`Qwen3Model`](crate::lm::models::qwen3::Qwen3Model) API
/// ([`embed_tokens`](Self::embed_tokens), [`forward_hidden`](Self::forward_hidden),
/// [`make_cache`](Self::make_cache)) so the aligner integration is a near
/// drop-in, differing only by the multimodal rotary embedding.
#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr-aligner")))]
#[derive(Debug)]
pub struct Qwen3AsrTextModel {
  /// `(vocab, hidden)` token-embedding table.
  embed_tokens: Array,
  layers: Vec<TransformerBlock>,
  norm: RMSNorm,
  rope: MRope,
}

impl Qwen3AsrTextModel {
  /// Embed `tokens` (`(B, L)` integer ids) to `(B, L, hidden)` via the token
  /// embedding table. No implicit eval — the returned [`Array`] is lazy.
  ///
  /// Each id must be a valid embedding row index in `[0, vocab_size)`; this thin
  /// gather mirrors the dense Qwen3 / Whisper embedding primitives and, like
  /// them, does not bound-check (MLX `take` reads the row directly, so an
  /// out-of-range id is an out-of-bounds read). The aligner that drives this
  /// decoder ([`ForcedAligner`](super::ForcedAligner)) range-validates its
  /// `input_ids` against `vocab_size` before calling this, so its callers never
  /// reach the gather with an out-of-range id; a caller invoking this primitive
  /// directly owns that same `[0, vocab_size)` contract.
  pub fn embed_tokens(&self, tokens: &Array) -> Result<Array> {
    take_axis(&self.embed_tokens, tokens, 0)
  }

  /// The number of decoder layers — also the per-layer cache cardinality
  /// [`make_cache`](Self::make_cache) builds.
  #[inline(always)]
  pub fn num_layers(&self) -> usize {
    self.layers.len()
  }

  /// Build the homogeneous per-layer cache: a [`StandardKvCache`] for every
  /// layer, in layer order.
  pub fn make_cache(&self) -> Vec<Box<dyn KvCache>> {
    (0..self.layers.len())
      .map(|_| -> Box<dyn KvCache> { Box::new(StandardKvCache::new()) })
      .collect()
  }

  /// Run the decoder over precomputed `h` (`(B, L, hidden)`) with the default
  /// **text-only** MRoPE positions (each axis = `arange(offset, offset + L)` at
  /// the first layer's cache offset), updating each layer's cache in place;
  /// returns the final-normed hidden states.
  ///
  /// The forced aligner runs a single full forward with text-only positions, so
  /// this is the path it uses. The per-layer cache must hold exactly one entry
  /// per decoder layer; a mismatched count is a recoverable
  /// [`Error::LengthMismatch`].
  pub fn forward_hidden(&self, h: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    self.forward_hidden_with_positions(h, None, cache)
  }

  /// As [`forward_hidden`](Self::forward_hidden), but with an explicit
  /// `position_ids` of shape `(3, B, L)` (the temporal / height / width MRoPE
  /// axes). `None` builds the text-only positions (every axis equal to the
  /// sequence index at the cache offset).
  pub fn forward_hidden_with_positions(
    &self,
    h: &Array,
    position_ids: Option<&Array>,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<Array> {
    if cache.len() != self.layers.len() {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "Qwen3AsrTextModel::forward: per-layer cache count vs decoder layers",
        self.layers.len(),
        cache.len(),
      )));
    }
    let shape = h.shape();
    if shape.len() != 3 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "Qwen3AsrTextModel::forward: hidden states must be rank-3 [batch, seq, hidden]",
        shape.len() as u32,
        shape,
      )));
    }
    // The token-embedding table is `(vocab, hidden)`, so its axis-1 is the width
    // every attention/MLP projection expects; reject a mismatched hidden axis
    // before the layers index it.
    let expected_hidden = self.embed_tokens.shape().get(1).copied();
    if let Some(hidden) = expected_hidden
      && shape[2] != hidden
    {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "Qwen3AsrTextModel::forward: hidden-states width vs token-embedding width",
        vec![hidden],
        vec![shape[2]],
      )));
    }
    let b = shape[0] as i32;
    let n = shape[1];

    // The shared causal mask from the first layer's cache, and the cache offset
    // for the default positions.
    let (mask, offset) = match cache.first_mut() {
      Some(c) => {
        let kv = downcast_cache(c.as_mut())?;
        let offset = kv.offset() as i32;
        (kv.make_mask(n, None, false)?, offset)
      }
      None => (MaskMode::None, 0),
    };

    // An explicit `position_ids` must be exactly `(3, batch, seq)` (the MRoPE
    // temporal/height/width axes over the batch and sequence the validated `h`
    // fixes). Reject a wrong rank or any dimension mismatch here, before
    // `cos_sin` reaches MLX `take`/`transpose`/`multiply`: a broadcastable bad
    // shape such as `(3, 1, seq)` for batch > 1 would otherwise silently reuse
    // one position row across the whole batch (wrong rotations / logits), and
    // other malformed shapes would surface as opaque MLX op errors rather than
    // the typed RankMismatch / ShapePairMismatch path. `batch`/`seq` come from
    // `h`'s already-validated rank-3 shape (no broadcasting allowed).
    if let Some(pos) = position_ids {
      let pos_shape = pos.shape();
      if pos_shape.len() != 3 {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "Qwen3AsrTextModel::forward: explicit position_ids must be rank-3 [3, batch, seq]",
          pos_shape.len() as u32,
          pos_shape,
        )));
      }
      let want = vec![MROPE_AXES, shape[0], shape[1]];
      if pos_shape != want {
        return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
          "Qwen3AsrTextModel::forward: explicit position_ids shape vs [3, batch, seq]",
          want,
          pos_shape,
        )));
      }
    }

    // Build (or borrow) the (3, B, L) position ids and the per-forward cos/sin.
    let owned_positions = match position_ids {
      Some(_) => None,
      None => Some(self.default_positions(b, n as i32, offset)?),
    };
    let positions = position_ids.unwrap_or_else(|| owned_positions.as_ref().unwrap());
    let (cos, sin) = self.rope.cos_sin(positions)?;

    let mut h = h.try_clone()?;
    for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
      let kv = downcast_cache(c.as_mut())?;
      h = layer.forward(&h, &self.rope, &cos, &sin, &mask, kv)?;
    }
    self.norm.forward(&h)
  }

  /// Embed `tokens` then run the decoder with text-only positions — the
  /// convenience composition of [`embed_tokens`](Self::embed_tokens) and
  /// [`forward_hidden`](Self::forward_hidden).
  pub fn forward_tokens(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    let h = self.embed_tokens(tokens)?;
    self.forward_hidden(&h, cache)
  }

  /// Build the text-only MRoPE `(3, B, L)` positions: every axis equals the
  /// sequence index `offset .. offset + L`, broadcast over the batch.
  fn default_positions(&self, b: i32, l: i32, offset: i32) -> Result<Array> {
    let stop = i64::from(offset) + i64::from(l);
    let pos = Array::arange::<i32>(f64::from(offset), stop as f64, 1.0)?; // (L,)
    // (L,) → (1, 1, L) → broadcast to (MROPE_AXES, B, L).
    let pos = reshape(&pos, &[1, 1, l])?;
    broadcast_to(&pos, &[MROPE_AXES as i32, b, l])
  }

  /// Build the head-less decoder from a parsed [`Qwen3AsrTextConfig`] and a flat
  /// name → [`Array`] weight map, draining the `model.*` keys it consumes
  /// (`model.embed_tokens.weight`, `model.norm.weight`, and per-layer
  /// `model.layers.{i}.…` projections / norms). A missing required weight is an
  /// [`Error::MissingKey`].
  pub fn from_weights(
    config: &Qwen3AsrTextConfig,
    weights: &mut HashMap<String, Array>,
  ) -> Result<Qwen3AsrTextModel> {
    // Validate the (public-field) config BEFORE deriving any head count or
    // projection width from it. The fields are public, so a caller can hand in a
    // structurally invalid config (e.g. `num_key_value_heads == 0`, an odd /
    // zero `head_dim`, a non-divisible GQA grouping); without this gate the
    // derived `kv_out == 0` k/v projection widths would build a model whose
    // `forward_hidden` feeds zero K/V heads into MLX SDPA, where `n_q_heads %
    // n_kv_heads` is a divide-by-zero — a public-input UB path. `validate`
    // rejects every such value with a typed error here instead.
    config.validate()?;
    let eps = config.rms_norm_eps;
    let head_dim = config.head_dim;
    let rope = MRope::new(head_dim, config.rope_theta, config.mrope()?)?;

    // Config-derived expected projection widths (the same arithmetic the forward
    // realizes). Each loaded tensor is shape-checked against these so a malformed
    // decoder checkpoint is rejected here, not as an opaque MLX error later — and
    // the accepted hidden width cannot be silently changed via the embedding
    // table's axis-1.
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let inter = config.intermediate_size;
    // q/k/v projection out-features: heads * head_dim (q) and kv_heads * head_dim
    // (k/v). `validate()` bounds `num_attention_heads * head_dim` within the i32
    // width cap; the kv product is no larger, so neither can overflow i32.
    let q_out = config.num_attention_heads.saturating_mul(head_dim);
    let kv_out = config.num_key_value_heads.saturating_mul(head_dim);

    let embed_tokens = take_shaped(
      weights,
      "model.embed_tokens.weight",
      "embed_tokens weight (vocab, hidden)",
      &[vocab, hidden],
    )?;
    let norm = RMSNorm::new(
      take_shaped(
        weights,
        "model.norm.weight",
        "final norm weight (hidden)",
        &[hidden],
      )?,
      eps,
    );

    let mut layers: Vec<TransformerBlock> = Vec::new();
    reserve_or_error(
      &mut layers,
      "Qwen3-ASR TextDecoderLayer",
      config.num_hidden_layers as usize,
    )?;
    for i in 0..config.num_hidden_layers {
      let p = format!("model.layers.{i}");
      let q = format!("{p}.self_attn");

      let self_attn = Attention {
        n_heads: config.num_attention_heads,
        n_kv_heads: config.num_key_value_heads,
        head_dim,
        scale: (head_dim as f32).powf(-0.5),
        q_proj: take_shaped(
          weights,
          &format!("{q}.q_proj.weight"),
          "q_proj weight (n_heads * head_dim, hidden)",
          &[q_out, hidden],
        )?,
        k_proj: take_shaped(
          weights,
          &format!("{q}.k_proj.weight"),
          "k_proj weight (n_kv_heads * head_dim, hidden)",
          &[kv_out, hidden],
        )?,
        v_proj: take_shaped(
          weights,
          &format!("{q}.v_proj.weight"),
          "v_proj weight (n_kv_heads * head_dim, hidden)",
          &[kv_out, hidden],
        )?,
        o_proj: take_shaped(
          weights,
          &format!("{q}.o_proj.weight"),
          "o_proj weight (hidden, n_heads * head_dim)",
          &[hidden, q_out],
        )?,
        q_norm: RMSNorm::new(
          take_shaped(
            weights,
            &format!("{q}.q_norm.weight"),
            "q_norm weight (head_dim)",
            &[head_dim],
          )?,
          eps,
        ),
        k_norm: RMSNorm::new(
          take_shaped(
            weights,
            &format!("{q}.k_norm.weight"),
            "k_norm weight (head_dim)",
            &[head_dim],
          )?,
          eps,
        ),
      };

      let mlp = Mlp {
        gate_proj: take_shaped(
          weights,
          &format!("{p}.mlp.gate_proj.weight"),
          "mlp gate_proj weight (intermediate, hidden)",
          &[inter, hidden],
        )?,
        up_proj: take_shaped(
          weights,
          &format!("{p}.mlp.up_proj.weight"),
          "mlp up_proj weight (intermediate, hidden)",
          &[inter, hidden],
        )?,
        down_proj: take_shaped(
          weights,
          &format!("{p}.mlp.down_proj.weight"),
          "mlp down_proj weight (hidden, intermediate)",
          &[hidden, inter],
        )?,
      };

      let input_layernorm = RMSNorm::new(
        take_shaped(
          weights,
          &format!("{p}.input_layernorm.weight"),
          "input_layernorm weight (hidden)",
          &[hidden],
        )?,
        eps,
      );
      let post_attention_layernorm = RMSNorm::new(
        take_shaped(
          weights,
          &format!("{p}.post_attention_layernorm.weight"),
          "post_attention_layernorm weight (hidden)",
          &[hidden],
        )?,
        eps,
      );

      layers.push(TransformerBlock {
        self_attn,
        mlp,
        input_layernorm,
        post_attention_layernorm,
      });
    }

    Ok(Qwen3AsrTextModel {
      embed_tokens,
      layers,
      norm,
      rope,
    })
  }
}

/// Downcast a per-layer `Box<dyn KvCache>` to the model's fixed
/// [`StandardKvCache`], erroring with a typed [`Error::InvariantViolation`] when
/// the cache kind does not match.
fn downcast_cache(cache: &mut dyn KvCache) -> Result<&mut StandardKvCache> {
  cache
    .as_any_mut()
    .downcast_mut::<StandardKvCache>()
    .ok_or_else(|| {
      Error::InvariantViolation(crate::error::InvariantViolationPayload::new(
        "Qwen3-ASR text per-layer cache kind",
        "every Qwen3-ASR text layer expects a StandardKvCache",
      ))
    })
}

/// Pull a required weight out of the map by `name`, erroring with the key on
/// absence.
fn take_weight(weights: &mut HashMap<String, Array>, name: &str) -> Result<Array> {
  weights.remove(name).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "Qwen3-ASR text weight map",
      format_smolstr!("{name}"),
    ))
  })
}

/// Assert a tensor's shape equals `expected` (rank + every dim) before it is
/// stored, so a checkpoint whose decoder weight disagrees with the
/// config-derived expectation is rejected here rather than running a different
/// graph (or silently changing the accepted hidden width via
/// `embed_tokens.shape()[1]`). On mismatch returns [`Error::ShapePairMismatch`]
/// wrapped in [`Error::LayerKeyed`] naming `key`. This mirrors the audio tower's
/// `expect_shape` so the decoder validates its weights the same way the audio
/// tower and the timestamp head already do.
fn expect_shape(
  tensor: &Array,
  key: &str,
  descriptor: &'static str,
  expected: &[i32],
) -> Result<()> {
  let actual = tensor.shape();
  let matches = actual.len() == expected.len()
    && actual
      .iter()
      .zip(expected.iter())
      .all(|(&a, &e)| e >= 0 && a as i64 == i64::from(e));
  if !matches {
    let expected_usize: Vec<usize> = expected.iter().map(|&e| e.max(0) as usize).collect();
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      key.to_string(),
      Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        descriptor,
        expected_usize,
        actual,
      )),
    )));
  }
  Ok(())
}

/// [`take_weight`] then assert the tensor's shape — the fused fetch-and-check
/// used for every decoder tensor stored verbatim, mirroring the audio tower's
/// `take_shaped`.
fn take_shaped(
  weights: &mut HashMap<String, Array>,
  key: &str,
  descriptor: &'static str,
  expected: &[i32],
) -> Result<Array> {
  let tensor = take_weight(weights, key)?;
  expect_shape(&tensor, key, descriptor, expected)?;
  Ok(tensor)
}

#[cfg(all(test, feature = "qwen3-asr-aligner"))]
mod tests;
