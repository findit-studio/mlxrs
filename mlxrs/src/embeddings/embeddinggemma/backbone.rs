//! EmbeddingGemma's Gemma3 text backbone.
//!
//! Ports `mlx-embeddings`'s `models/gemma3_text.py` `Gemma3Model`, which reuses
//! the `ModelArgs` / `RMSNorm` / `TransformerBlock` of `mlx-lm`'s
//! `models/gemma3_text.py`. The backbone is a Gemma3 text transformer driven as
//! a **bidirectional encoder**: the public model (see [`super`]) builds an
//! additive padding mask (`0` on real tokens, `-inf` on pad) and passes it to
//! **every** layer, so each layer runs full (non-causal) attention over the real
//! tokens — `gemma3_text.py`'s `Model.__call__` (the mlx-embeddings one) creates
//! the `extended_attention_mask` from `attention_mask` and feeds it to
//! `self.model(inputs, extended_attention_mask)`.
//!
//! ## Per-layer architecture (`gemma3_text.py` `TransformerBlock`)
//!
//! Each layer is the Gemma3 sandwich-norm block:
//!
//! ```text
//! r   = self_attn(input_layernorm(x))
//! h   = x + post_attention_layernorm(r)
//! r   = mlp(pre_feedforward_layernorm(h))
//! out = h + post_feedforward_layernorm(r)
//! ```
//!
//! - **Attention**: grouped-query (`num_attention_heads` query heads,
//!   `num_key_value_heads` kv heads), per-head `head_dim` (independent of
//!   `hidden / heads`), **query and key RMSNorm** over the head dimension,
//!   RoPE, and the fused SDPA with `scale = query_pre_attn_scalar ** -0.5`. The
//!   RoPE base alternates per layer: the **global** layers (`is_global_layer`)
//!   use `rope_theta`, the local layers use `rope_local_base_freq` (the only
//!   effect the sliding-window pattern has here — the attention is bidirectional
//!   throughout, so the window itself is not applied).
//! - **MLP**: the Gemma gated feed-forward `down(gelu_approx(gate(x)) * up(x))`
//!   (`gelu_approx` = the `tanh` GELU, `mlx.nn.gelu_approx`).
//! - **Norms**: four `RMSNorm`s per layer plus a final backbone `RMSNorm`,
//!   each with Gemma's `1.0 + weight` reparameterization (folded in at load by
//!   the private `shared::build_gemma_rms_norm`).
//!
//! The token embedding is scaled by `sqrt(hidden_size)` before the first layer
//! (`gemma3_text.py`'s `h *= hidden_size ** 0.5`).

use std::collections::HashMap;

use crate::{
  array::Array,
  error::{Error, RankMismatchPayload, Result},
  lm::nn::{
    attention::{Mask, scaled_dot_product_attention},
    norm::RMSNorm,
    rope::Rope,
  },
  model_validation::reserve_or_error,
  ops,
};

use super::{
  config::Gemma3Config,
  shared::{DtypeAlias, Embedding, Linear, build_gemma_rms_norm, dim_i32, embedding_scale_like},
};
use crate::lm::quant::PerLayerQuantization;

/// Gemma3 grouped-query attention with per-head q/k RMSNorm (`gemma3_text.py`'s
/// `Attention`).
#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
struct Attention {
  n_heads: i32,
  n_kv_heads: i32,
  head_dim: i32,
  /// SDPA scale (`query_pre_attn_scalar ** -0.5`).
  scale: f32,
  q_proj: Linear,
  k_proj: Linear,
  v_proj: Linear,
  o_proj: Linear,
  q_norm: RMSNorm,
  k_norm: RMSNorm,
  rope: Rope,
}

#[cfg(feature = "embeddinggemma")]
impl Attention {
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    config: &Gemma3Config,
    layer_idx: i32,
    eps: f32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let hidden = config.hidden_size;
    let head_dim = config.head_dim;
    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.num_key_value_heads;
    // q is `(n_heads * head_dim, hidden)`; k/v are `(n_kv_heads * head_dim,
    // hidden)`; o is `(hidden, n_heads * head_dim)`. The products are
    // overflow-checked (config caps keep them small in practice).
    let q_out = crate::model_validation::checked_mul(
      "Gemma3 Attention: n_heads * head_dim",
      "n_heads",
      n_heads,
      "head_dim",
      head_dim,
    )?;
    let kv_out = crate::model_validation::checked_mul(
      "Gemma3 Attention: n_kv_heads * head_dim",
      "n_kv_heads",
      n_kv_heads,
      "head_dim",
      head_dim,
    )?;
    // Each projection auto-detects dense-vs-quantized from its `.scales` sibling
    // (mlx-embeddings' `class_predicate` quantizes every `nn.Linear`); the dense
    // path pins the `(out, in)` weight to the config exactly as before.
    let q_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.q_proj"),
      q_out,
      hidden,
      "attn q_proj weight (n_heads*head_dim, hidden)",
      quant,
    )?;
    let k_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.k_proj"),
      kv_out,
      hidden,
      "attn k_proj weight (n_kv_heads*head_dim, hidden)",
      quant,
    )?;
    let v_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.v_proj"),
      kv_out,
      hidden,
      "attn v_proj weight (n_kv_heads*head_dim, hidden)",
      quant,
    )?;
    let o_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.o_proj"),
      hidden,
      q_out,
      "attn o_proj weight (hidden, n_heads*head_dim)",
      quant,
    )?;
    let q_norm = build_gemma_rms_norm(weights, &format!("{prefix}.q_norm"), head_dim, eps)?;
    let k_norm = build_gemma_rms_norm(weights, &format!("{prefix}.k_norm"), head_dim, eps)?;

    // RoPE base alternates: global layers use `rope_theta`, local (sliding)
    // layers use `rope_local_base_freq`. Non-traditional layout, scale 1.0,
    // matching `gemma3_text.py`'s `initialize_rope(..., traditional=False)`.
    let base = if config.is_global_layer(layer_idx) {
      config.rope_theta as f32
    } else {
      config.rope_local_base_freq as f32
    };
    let rope = Rope::new(head_dim, false, base, 1.0);

    Ok(Self {
      n_heads,
      n_kv_heads,
      head_dim,
      scale: (config.query_pre_attn_scalar as f32).powf(-0.5),
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      rope,
    })
  }

  /// `(B, L, hidden) → (B, L, hidden)` bidirectional attention with the additive
  /// padding `mask`. No KV cache (encoder), so RoPE is applied at offset 0.
  fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
    let shape = x.shape();
    let b = dim_i32(&shape, 0, "Gemma3 Attention: batch")?;
    let l = dim_i32(&shape, 1, "Gemma3 Attention: seq")?;

    let queries = self.q_proj.forward(x)?;
    let keys = self.k_proj.forward(x)?;
    let values = self.v_proj.forward(x)?;

    // Per-head reshape `(B, L, heads, head_dim)`, then transpose to
    // `(B, heads, L, head_dim)`. q/k RMSNorm over the head dimension is applied
    // **before** the transpose (the reference norms the `(B, L, heads,
    // head_dim)` layout, whose last axis is `head_dim`).
    let queries = ops::shape::reshape(&queries, &[b, l, self.n_heads, self.head_dim])?;
    let queries = self.q_norm.forward(&queries)?;
    let queries = ops::shape::transpose_axes(&queries, &[0, 2, 1, 3])?;

    let keys = ops::shape::reshape(&keys, &[b, l, self.n_kv_heads, self.head_dim])?;
    let keys = self.k_norm.forward(&keys)?;
    let keys = ops::shape::transpose_axes(&keys, &[0, 2, 1, 3])?;

    let values = ops::shape::reshape(&values, &[b, l, self.n_kv_heads, self.head_dim])?;
    let values = ops::shape::transpose_axes(&values, &[0, 2, 1, 3])?;

    // Encoder: no cache, rotate at absolute position 0.
    let queries = self.rope.apply(&queries, 0)?;
    let keys = self.rope.apply(&keys, 0)?;

    let attn =
      scaled_dot_product_attention(&queries, &keys, &values, self.scale, Mask::Array(mask))?;
    // `(B, heads, L, head_dim)` → `(B, L, heads*head_dim)`.
    let attn = ops::shape::transpose_axes(&attn, &[0, 2, 1, 3])?;
    let embed_dim = crate::model_validation::checked_mul(
      "Gemma3 Attention: n_heads * head_dim",
      "n_heads",
      self.n_heads,
      "head_dim",
      self.head_dim,
    )?;
    let attn = ops::shape::reshape(&attn, &[b, l, embed_dim])?;
    self.o_proj.forward(&attn)
  }
}

/// Gemma gated feed-forward (`gemma3_text.py`'s `MLP`):
/// `down(gelu_approx(gate(x)) * up(x))`, all three projections bias-free.
#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
struct Mlp {
  gate_proj: Linear,
  up_proj: Linear,
  down_proj: Linear,
}

#[cfg(feature = "embeddinggemma")]
impl Mlp {
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    hidden: i32,
    intermediate: i32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let gate_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.gate_proj"),
      intermediate,
      hidden,
      "MLP gate_proj weight (intermediate, hidden)",
      quant,
    )?;
    let up_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.up_proj"),
      intermediate,
      hidden,
      "MLP up_proj weight (intermediate, hidden)",
      quant,
    )?;
    let down_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.down_proj"),
      hidden,
      intermediate,
      "MLP down_proj weight (hidden, intermediate)",
      quant,
    )?;
    Ok(Self {
      gate_proj,
      up_proj,
      down_proj,
    })
  }

  /// `down(gelu_approx(gate(x)) * up(x))`.
  fn forward(&self, x: &Array) -> Result<Array> {
    let gate = self.gate_proj.forward(x)?;
    let gate = crate::lm::nn::activations::gelu_approx(&gate)?;
    let up = self.up_proj.forward(x)?;
    let h = gate.multiply(&up)?;
    self.down_proj.forward(&h)
  }
}

/// The Gemma3 residual add with f16 saturation (`gemma3_text.py`'s
/// `clip_residual`).
///
/// For any non-f16 dtype this is a plain `x + y`. For f16 — whose finite range
/// tops out at `f16::MAX` (`65504`) — a residual sum can overflow to `inf` and
/// then poison the rest of the network with `NaN`s, so the reference instead
/// adds in f32, clamps to the finite f16 range `[-f16::MAX, f16::MAX]`, and
/// casts back to f16, yielding the saturated finite value rather than `inf`.
/// Both residual additions in the block route through this.
#[cfg(feature = "embeddinggemma")]
fn clip_residual(x: &Array, y: &Array) -> Result<Array> {
  if x.dtype()? != crate::dtype::Dtype::F16 {
    return x.add(y);
  }
  // f16 finite range. `half::f16::MAX` is mlx's `finfo(float16).max` (`65504`).
  let bound = f32::from(half::f16::MAX);
  let xf = ops::misc::astype(x, crate::dtype::Dtype::F32)?;
  let yf = ops::misc::astype(y, crate::dtype::Dtype::F32)?;
  let sum = xf.add(&yf)?;
  let clipped = ops::misc::clip_with_scalar(&sum, -bound, bound)?;
  ops::misc::astype(&clipped, crate::dtype::Dtype::F16)
}

/// A Gemma3 sandwich-norm transformer block (`gemma3_text.py`'s
/// `TransformerBlock`).
#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
struct TransformerBlock {
  self_attn: Attention,
  mlp: Mlp,
  input_layernorm: RMSNorm,
  post_attention_layernorm: RMSNorm,
  pre_feedforward_layernorm: RMSNorm,
  post_feedforward_layernorm: RMSNorm,
}

#[cfg(feature = "embeddinggemma")]
impl TransformerBlock {
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    config: &Gemma3Config,
    layer_idx: i32,
    eps: f32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let prefix = format!("model.layers.{layer_idx}");
    let hidden = config.hidden_size;
    let self_attn = Attention::from_weights(
      weights,
      &format!("{prefix}.self_attn"),
      config,
      layer_idx,
      eps,
      quant,
    )?;
    let mlp = Mlp::from_weights(
      weights,
      &format!("{prefix}.mlp"),
      hidden,
      config.intermediate_size,
      quant,
    )?;
    let input_layernorm =
      build_gemma_rms_norm(weights, &format!("{prefix}.input_layernorm"), hidden, eps)?;
    let post_attention_layernorm = build_gemma_rms_norm(
      weights,
      &format!("{prefix}.post_attention_layernorm"),
      hidden,
      eps,
    )?;
    let pre_feedforward_layernorm = build_gemma_rms_norm(
      weights,
      &format!("{prefix}.pre_feedforward_layernorm"),
      hidden,
      eps,
    )?;
    let post_feedforward_layernorm = build_gemma_rms_norm(
      weights,
      &format!("{prefix}.post_feedforward_layernorm"),
      hidden,
      eps,
    )?;
    Ok(Self {
      self_attn,
      mlp,
      input_layernorm,
      post_attention_layernorm,
      pre_feedforward_layernorm,
      post_feedforward_layernorm,
    })
  }

  /// `r = attn(ln_in(x)); h = clip_residual(x, ln_post_attn(r)); r =
  /// mlp(ln_pre_ff(h)); out = clip_residual(h, ln_post_ff(r))`. Both residual
  /// adds go through [`clip_residual`] so an f16 backbone saturates to the
  /// finite f16 range instead of overflowing to `inf`/`NaN` (the reference's
  /// `clip_residual`).
  fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
    let r = self
      .self_attn
      .forward(&self.input_layernorm.forward(x)?, mask)?;
    let h = clip_residual(x, &self.post_attention_layernorm.forward(&r)?)?;
    let r = self
      .mlp
      .forward(&self.pre_feedforward_layernorm.forward(&h)?)?;
    clip_residual(&h, &self.post_feedforward_layernorm.forward(&r)?)
  }
}

/// The Gemma3 text backbone (`gemma3_text.py`'s `Gemma3Model`): token embedding
/// (scaled by `sqrt(hidden)`) → sandwich-norm transformer stack → final
/// `RMSNorm`. Driven as a bidirectional encoder by the public model.
#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
#[derive(Debug)]
pub(crate) struct Gemma3Backbone {
  embed_tokens: Embedding,
  layers: Vec<TransformerBlock>,
  norm: RMSNorm,
  hidden_size: i32,
}

#[cfg(feature = "embeddinggemma")]
impl Gemma3Backbone {
  /// Build the backbone from a validated [`Gemma3Config`] and the (sanitized)
  /// weight map, whose keys follow `mlx-lm`'s `model.*` tree
  /// (`model.embed_tokens.weight`, `model.layers.{i}.*`, `model.norm.weight`).
  ///
  /// `quant` is the parsed per-layer quantization config (from
  /// [`Gemma3Config::quantization`]); each `nn.Linear` / the token embedding
  /// loads quantized when the checkpoint carries that layer's `.scales` sibling
  /// (the mlx-embeddings `class_predicate` auto-detect), else dense.
  ///
  /// Every consumed tensor's shape is pinned to its exact config-derived
  /// dimensions (typed [`crate::Error::ShapePairMismatch`] wrapped in
  /// [`crate::Error::LayerKeyed`]); the quantized path pins the packed triple's
  /// logical shape identically.
  pub(crate) fn from_weights(
    config: &Gemma3Config,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Idempotent re-validation: a caller may build directly from an unvalidated
    // config. This bounds `num_hidden_layers` (and every dim) before the
    // per-layer reservation/loop.
    config.validate()?;
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps as f32;

    // The token embedding is an `nn.Embedding`, also quantize-aware (the
    // weight-tied projection mlx-embeddings' `class_predicate` quantizes).
    let embed_tokens = Embedding::from_weights(
      weights,
      "model.embed_tokens",
      config.vocab_size,
      hidden,
      "token-embedding table (vocab, hidden)",
      quant,
    )?;

    // `num_hidden_layers` is bounded by `MAX_CARDINALITY` in `validate`, but
    // reserve fallibly so even a within-cap heavyweight per-layer `Vec` the
    // allocator cannot satisfy is a recoverable [`Error::AllocFailure`] rather
    // than `with_capacity`'s abort (the merged LFM2 / SigLIP2 pattern).
    let mut layers: Vec<TransformerBlock> = Vec::new();
    reserve_or_error(
      &mut layers,
      "Gemma3 TransformerBlock",
      config.num_hidden_layers as usize,
    )?;
    for i in 0..config.num_hidden_layers {
      layers.push(TransformerBlock::from_weights(
        weights, config, i, eps, quant,
      )?);
    }

    let norm = build_gemma_rms_norm(weights, "model.norm", hidden, eps)?;

    Ok(Self {
      embed_tokens,
      layers,
      norm,
      hidden_size: hidden,
    })
  }

  /// Run the backbone over a `(batch, seq_len)` i32 token-id batch and the
  /// `(batch, 1, 1, seq_len)` additive padding `mask`, returning the final
  /// `(batch, seq_len, hidden)` hidden states (post final `RMSNorm`).
  ///
  /// `input_ids` is pinned to exactly rank-2 before any op (the public
  /// `embed_text` accepts an untrusted array; the embedding gather + the
  /// per-head reshape are only defined for a rank-2 batch).
  pub(crate) fn forward(&self, input_ids: &Array, mask: &Array) -> Result<Array> {
    let shape = input_ids.shape();
    if shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "embeddinggemma backbone: input_ids must be rank-2 (batch, seq_len)",
        shape.len() as u32,
        shape,
      )));
    }

    // token_embedding(ids): (B, L) → (B, L, hidden) via axis-0 gather (the
    // quantized table dequantizes the gathered rows).
    let mut h = self.embed_tokens.forward(input_ids)?;
    // Scale by sqrt(hidden_size). The reference builds this scalar in the
    // `embed_tokens` weight dtype, then casts to the hidden dtype, with no bf16
    // rounding (mlx-embeddings `gemma3_text.py`'s
    // `h *= mx.array(hidden_size ** 0.5, embed_tokens.weight.dtype).astype(h.dtype)`).
    // Here `h` is gathered straight from `embed_tokens`, so its dtype is the
    // weight (dense) / dequantized-activation (quantized) dtype and the scalar is
    // built directly in it.
    let scale = embedding_scale_like(self.hidden_size as f32, &h)?;
    h = h.multiply(&scale)?;

    for layer in &self.layers {
      h = layer.forward(&h, mask)?;
    }
    self.norm.forward(&h)
  }

  /// The token-embedding table's dtype — the dtype the additive attention mask
  /// must be cast to so the fused SDPA sees a matching-dtype mask. A cheap
  /// handle query (no eval).
  #[inline]
  pub(crate) fn embed_dtype(&self) -> Result<crate::dtype::Dtype> {
    self.embed_tokens.dtype()
  }

  /// `true` if the token embedding was loaded from a quantized checkpoint
  /// (test-only introspection for the quantized-load test).
  #[cfg(test)]
  pub(crate) fn embedding_is_quantized(&self) -> bool {
    self.embed_tokens.is_quantized()
  }

  /// `true` if every layer's attention + MLP projections were loaded quantized
  /// (test-only introspection). Empty layer stacks vacuously return `true`.
  #[cfg(test)]
  pub(crate) fn all_projections_quantized(&self) -> bool {
    self.layers.iter().all(|l| {
      l.self_attn.q_proj.is_quantized()
        && l.self_attn.k_proj.is_quantized()
        && l.self_attn.v_proj.is_quantized()
        && l.self_attn.o_proj.is_quantized()
        && l.mlp.gate_proj.is_quantized()
        && l.mlp.up_proj.is_quantized()
        && l.mlp.down_proj.is_quantized()
    })
  }
}

/// Build the `(batch, 1, 1, seq_len)` additive attention mask from a
/// `(batch, seq_len)` `{0,1}` padding mask: `0.0` where a token is attended,
/// `-inf` where it is padding — broadcastable to the SDPA `[B, N_q, T_q, T_kv]`
/// key axis.
///
/// Mirrors `gemma3_text.py` (the mlx-embeddings `Model`)
/// `get_extended_attention_mask` + the `where(mask, 0.0, -inf)` step: a rank-2
/// `attention_mask` becomes `[:, None, None, :]`, then the boolean is mapped to
/// the additive `{0, -inf}` form and cast to `dtype`. The result masks **keys**
/// only (every query attends to every real key — bidirectional), which is the
/// encoder contract.
#[cfg(feature = "embeddinggemma")]
pub(crate) fn build_additive_mask(attention_mask: &Array, dtype: DtypeAlias) -> Result<Array> {
  let shape = attention_mask.shape();
  if shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "embeddinggemma: attention_mask must be rank-2 (batch, seq_len)",
      shape.len() as u32,
      shape,
    )));
  }
  // (B, S) → (B, 1, 1, S): one broadcastable key-axis mask per batch row.
  let expanded = ops::shape::expand_dims_axes(attention_mask, &[1, 2])?;
  // `where(mask != 0, 0.0, -inf)`. The padding mask is `{0.0, 1.0}` f32 here
  // (the encode pipeline builds an f32 mask); compare against 0 to a boolean,
  // then select 0.0 / -inf.
  let zero = Array::full::<f32>(&(1,), 0.0)?;
  let keep = ops::comparison::not_equal(&expanded, &zero)?; // bool: real tokens
  let additive_zero = Array::full::<f32>(&(1,), 0.0)?;
  let neg_inf = Array::full::<f32>(&(1,), f32::NEG_INFINITY)?;
  let mask = ops::logical::select(&keep, &additive_zero, &neg_inf)?;
  // Cast to the hidden dtype so SDPA sees a matching additive mask dtype.
  ops::misc::astype(&mask, dtype)
}

#[cfg(all(test, feature = "embeddinggemma"))]
mod tests {
  use super::*;
  use crate::dtype::Dtype;

  /// Materialize `values` as an f16 `Array` (built in f32, cast to f16).
  fn f16_arr(values: &[f32]) -> Array {
    let f32_arr = Array::from_slice::<f32>(values, &(values.len(),)).unwrap();
    ops::misc::astype(&f32_arr, Dtype::F16).unwrap()
  }

  /// Cast `a` to f32, eval, and read it back (`to_vec` is dtype-strict, so an
  /// f16 array must be cast before it can be read as `f32`).
  fn to_f32(a: &Array) -> Vec<f32> {
    let mut a = ops::misc::astype(a, Dtype::F32).unwrap();
    a.eval().unwrap();
    a.to_vec::<f32>().unwrap()
  }

  #[test]
  fn clip_residual_saturates_f16_overflow_to_finite() {
    // Each operand is well within the f16 range (`f16::MAX` = 65504), but their
    // sum (120000) overflows f16. A bare f16 add would round to `inf`; the
    // reference `clip_residual` instead adds in f32 and clamps to `f16::MAX`.
    let bound = f32::from(half::f16::MAX);
    let x = f16_arr(&[60_000.0, -60_000.0, 1.0]);
    let y = f16_arr(&[60_000.0, -60_000.0, 2.0]);
    let out = clip_residual(&x, &y).expect("clip_residual");
    assert_eq!(out.dtype().unwrap(), Dtype::F16, "stays f16");
    let v = to_f32(&out);
    assert!(
      v.iter().all(|x| x.is_finite()),
      "f16 residual overflow must saturate to finite, got {v:?}"
    );
    // +overflow saturates to +f16::MAX, -overflow to -f16::MAX, the in-range
    // sum (3.0) is preserved (exactly representable in f16).
    assert_eq!(v[0], bound, "positive overflow clamps to +f16::MAX");
    assert_eq!(v[1], -bound, "negative overflow clamps to -f16::MAX");
    assert_eq!(v[2], 3.0, "in-range residual is unchanged");
  }

  #[test]
  fn clip_residual_f16_bare_add_would_overflow_to_infinity() {
    // Guard the test's own premise: a *bare* f16 add of the same operands does
    // round to infinity, so the saturation above is doing real work.
    let x = f16_arr(&[60_000.0]);
    let y = f16_arr(&[60_000.0]);
    let bare = x.add(&y).expect("bare f16 add");
    assert_eq!(bare.dtype().unwrap(), Dtype::F16);
    assert!(
      to_f32(&bare)[0].is_infinite(),
      "a bare f16 add of 60000+60000 overflows to inf"
    );
  }

  #[test]
  fn clip_residual_non_f16_is_a_plain_add() {
    // For f32 the helper must NOT clip — a sum past the f16 range stays as the
    // true f32 value (the reference returns `x + y` unchanged off the f16 path).
    let x = Array::from_slice::<f32>(&[60_000.0, 1.5], &(2usize,)).unwrap();
    let y = Array::from_slice::<f32>(&[60_000.0, 2.5], &(2usize,)).unwrap();
    let out = clip_residual(&x, &y).expect("clip_residual");
    assert_eq!(out.dtype().unwrap(), Dtype::F32, "f32 stays f32");
    let v = to_f32(&out);
    assert_eq!(v[0], 120_000.0, "f32 add is not clamped to the f16 range");
    assert_eq!(v[1], 4.0);
  }
}
