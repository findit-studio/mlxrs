//! Shared building blocks for the SigLIP2 NaFlex towers — the private
//! `nn.Linear` forward, the two-layer GELU `MLP` both towers and the
//! attention-pool head compose, and the weight fetch + shape-pinning helpers
//! (the same discipline as the merged Wav2Vec2 / LFM2 ports: every consumed
//! tensor's shape is checked before it is stored or fed to any op, with a
//! typed [`Error::ShapePairMismatch`] wrapped in [`Error::LayerKeyed`]).
//!
//! These are deliberately not the public `lm::nn` layers: mlxrs ships no
//! public `nn::Linear` (a GAP by design — each model composes the matmul),
//! and SigLIP2's `MLP` is the `gelu(approx="precise")` form
//! (`siglip.py`'s `MLP(config, approx="precise")`), i.e.
//! [`crate::lm::nn::activations::gelu_approx`].

use std::collections::HashMap;

use crate::{
  array::Array,
  error::{
    Error, LayerKeyedPayload, MissingKeyPayload, OutOfRangePayload, RankMismatchPayload, Result,
    ShapePairMismatchPayload,
  },
  lm::nn::{
    attention::{Mask, scaled_dot_product_attention},
    norm::LayerNorm,
  },
  model_validation::checked_mul,
  ops,
};

/// `y = x @ wᵀ (+ bias)` — the private `nn.Linear` forward both towers use.
///
/// `siglip.py` stores every `nn.Linear` weight as `(out, in)`, so the matmul
/// is `x @ wᵀ`. With a bias this is the fused `addmm(bias, x, wᵀ, 1, 1)`;
/// without, a plain `matmul(x, wᵀ)`. Returns a new lazy [`Array`] (no eval).
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array> {
  let wt = ops::shape::swapaxes(weight, -1, -2)?;
  match bias {
    Some(b) => ops::linalg_basic::addmm(b, x, &wt, 1.0, 1.0),
    None => ops::linalg_basic::matmul(x, &wt),
  }
}

/// The exact `approx="precise"` GELU SigLIP2's `MLP` / heads use — the `tanh`
/// approximation (`mlx.nn.GELU(approx="precise")`), forwarded to the shared
/// [`crate::lm::nn::activations::gelu_approx`].
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn gelu_precise(x: &Array) -> Result<Array> {
  crate::lm::nn::activations::gelu_approx(x)
}

/// The two-layer GELU feed-forward (`siglip.py`'s `MLP`):
/// `fc2(gelu_precise(fc1(x)))`, with biased `Linear(hidden → intermediate)`
/// then `Linear(intermediate → hidden)`.
#[cfg(feature = "siglip2-naflex")]
pub(crate) struct Mlp {
  fc1_weight: Array,
  fc1_bias: Array,
  fc2_weight: Array,
  fc2_bias: Array,
}

#[cfg(feature = "siglip2-naflex")]
impl Mlp {
  /// Build from `{prefix}.fc1.*` + `{prefix}.fc2.*`, pinning the two Linear
  /// shapes to `(intermediate, hidden)` / `(hidden, intermediate)` and the
  /// biases to `(intermediate,)` / `(hidden,)`.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    hidden: i32,
    intermediate: i32,
  ) -> Result<Self> {
    let fc1_weight = take_shaped(
      weights,
      &format!("{prefix}.fc1.weight"),
      "MLP fc1 weight (intermediate, hidden)",
      &[intermediate, hidden],
    )?;
    let fc1_bias = take_shaped(
      weights,
      &format!("{prefix}.fc1.bias"),
      "MLP fc1 bias (intermediate,)",
      &[intermediate],
    )?;
    let fc2_weight = take_shaped(
      weights,
      &format!("{prefix}.fc2.weight"),
      "MLP fc2 weight (hidden, intermediate)",
      &[hidden, intermediate],
    )?;
    let fc2_bias = take_shaped(
      weights,
      &format!("{prefix}.fc2.bias"),
      "MLP fc2 bias (hidden,)",
      &[hidden],
    )?;
    Ok(Self {
      fc1_weight,
      fc1_bias,
      fc2_weight,
      fc2_bias,
    })
  }

  /// `fc2(gelu_precise(fc1(x)))`.
  pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
    let h = linear(x, &self.fc1_weight, Some(&self.fc1_bias))?;
    let h = gelu_precise(&h)?;
    linear(&h, &self.fc2_weight, Some(&self.fc2_bias))
  }
}

/// The per-layer shape constants both towers' encoder stacks share: the
/// transformer width, the feed-forward width, the head split, and the
/// LayerNorm eps. Bundled so the layer builders take one config arg instead of
/// six positional scalars (and so a tower computes the head split once).
#[cfg(feature = "siglip2-naflex")]
#[derive(Clone, Copy)]
pub(crate) struct LayerDims {
  /// Transformer hidden / embedding dimension.
  pub hidden: i32,
  /// Feed-forward intermediate dimension.
  pub intermediate: i32,
  /// Number of attention heads.
  pub num_heads: i32,
  /// Per-head dimension (`hidden / num_heads`).
  pub head_dim: i32,
  /// SDPA scale (`head_dim**-0.5`).
  pub scale: f32,
  /// LayerNorm eps.
  pub eps: f32,
}

#[cfg(feature = "siglip2-naflex")]
impl LayerDims {
  /// Derive the per-layer dims from a tower's `(hidden, intermediate,
  /// num_heads, eps)`, computing the head split and SDPA scale once.
  /// `num_heads` must be positive and divide `hidden` (the caller validates
  /// this against the config).
  pub(crate) fn new(hidden: i32, intermediate: i32, num_heads: i32, eps: f32) -> Result<Self> {
    crate::model_validation::require_positive("siglip2: num_attention_heads", num_heads)?;
    crate::model_validation::require_divisible(
      "siglip2: hidden_size",
      hidden,
      "siglip2: num_attention_heads",
      num_heads,
    )?;
    let head_dim = hidden / num_heads;
    Ok(Self {
      hidden,
      intermediate,
      num_heads,
      head_dim,
      scale: (head_dim as f32).powf(-0.5),
      eps,
    })
  }
}

/// SigLIP2 self-attention (`siglip.py`'s `Attention`, the bias=True
/// `q/k/v/out` projection form both encoder stacks use).
///
/// `q/k/v/out` are biased `Linear(hidden, hidden)`. The query is **not**
/// pre-scaled — `scale = head_dim**-0.5` is passed to SDPA, matching the
/// reference's `mx.fast.scaled_dot_product_attention(..., scale=self.scale)`.
/// `mask` is `Mask::None` for the (full-attention) text tower and the additive
/// padded-key mask for the NaFlex vision tower.
#[cfg(feature = "siglip2-naflex")]
pub(crate) struct Attention {
  q_weight: Array,
  q_bias: Array,
  k_weight: Array,
  k_bias: Array,
  v_weight: Array,
  v_bias: Array,
  out_weight: Array,
  out_bias: Array,
  num_heads: i32,
  head_dim: i32,
  /// `head_dim**-0.5`, the SDPA scale.
  scale: f32,
}

#[cfg(feature = "siglip2-naflex")]
impl Attention {
  /// Build from `{prefix}.{q,k,v,out}_proj.{weight,bias}`, pinning each
  /// projection to `(hidden, hidden)` and each bias to `(hidden,)`.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    dims: LayerDims,
  ) -> Result<Self> {
    let hidden = dims.hidden;
    let proj = [hidden, hidden];
    Ok(Self {
      q_weight: take_shaped(
        weights,
        &format!("{prefix}.q_proj.weight"),
        "attn q_proj weight (hidden, hidden)",
        &proj,
      )?,
      q_bias: take_shaped(
        weights,
        &format!("{prefix}.q_proj.bias"),
        "attn q_proj bias (hidden,)",
        &[hidden],
      )?,
      k_weight: take_shaped(
        weights,
        &format!("{prefix}.k_proj.weight"),
        "attn k_proj weight (hidden, hidden)",
        &proj,
      )?,
      k_bias: take_shaped(
        weights,
        &format!("{prefix}.k_proj.bias"),
        "attn k_proj bias (hidden,)",
        &[hidden],
      )?,
      v_weight: take_shaped(
        weights,
        &format!("{prefix}.v_proj.weight"),
        "attn v_proj weight (hidden, hidden)",
        &proj,
      )?,
      v_bias: take_shaped(
        weights,
        &format!("{prefix}.v_proj.bias"),
        "attn v_proj bias (hidden,)",
        &[hidden],
      )?,
      out_weight: take_shaped(
        weights,
        &format!("{prefix}.out_proj.weight"),
        "attn out_proj weight (hidden, hidden)",
        &proj,
      )?,
      out_bias: take_shaped(
        weights,
        &format!("{prefix}.out_proj.bias"),
        "attn out_proj bias (hidden,)",
        &[hidden],
      )?,
      num_heads: dims.num_heads,
      head_dim: dims.head_dim,
      scale: dims.scale,
    })
  }

  /// `(B, L, C) → (B, L, C)` attention with the given key `mask`.
  pub(crate) fn forward(&self, x: &Array, mask: Mask<'_>) -> Result<Array> {
    let shape = x.shape();
    let bsz = dim_i32(&shape, 0, "siglip2 Attention: batch")?;
    let seq = dim_i32(&shape, 1, "siglip2 Attention: seq")?;

    let q = linear(x, &self.q_weight, Some(&self.q_bias))?;
    let k = linear(x, &self.k_weight, Some(&self.k_bias))?;
    let v = linear(x, &self.v_weight, Some(&self.v_bias))?;

    let q = self.split_heads(&q, bsz, seq)?;
    let k = self.split_heads(&k, bsz, seq)?;
    let v = self.split_heads(&v, bsz, seq)?;

    let attn = scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;
    let attn = ops::shape::transpose_axes(&attn, &[0, 2, 1, 3])?;
    let embed_dim = checked_mul(
      "siglip2 Attention: num_heads * head_dim",
      "num_heads",
      self.num_heads,
      "head_dim",
      self.head_dim,
    )?;
    let attn = ops::shape::reshape(&attn, &[bsz, seq, embed_dim])?;
    linear(&attn, &self.out_weight, Some(&self.out_bias))
  }

  /// `(B, L, C) → (B, n_heads, L, head_dim)`.
  fn split_heads(&self, x: &Array, bsz: i32, seq: i32) -> Result<Array> {
    let reshaped = ops::shape::reshape(x, &[bsz, seq, self.num_heads, self.head_dim])?;
    ops::shape::transpose_axes(&reshaped, &[0, 2, 1, 3])
  }
}

/// A SigLIP2 pre-norm encoder layer (`siglip.py`'s `EncoderLayer`):
/// `h = x + attn(ln1(x)); out = h + mlp(ln2(h))`. Shared verbatim by both
/// towers (`Encoder(config, approx="precise")`).
#[cfg(feature = "siglip2-naflex")]
pub(crate) struct EncoderLayer {
  layer_norm1: LayerNorm,
  attention: Attention,
  layer_norm2: LayerNorm,
  mlp: Mlp,
}

#[cfg(feature = "siglip2-naflex")]
impl EncoderLayer {
  /// Build the `i`-th layer from `{encoder_prefix}.layers.{i}.*`.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    encoder_prefix: &str,
    i: i32,
    dims: LayerDims,
  ) -> Result<Self> {
    let prefix = format!("{encoder_prefix}.layers.{i}");
    let layer_norm1 = build_layer_norm(
      weights,
      &format!("{prefix}.layer_norm1"),
      dims.hidden,
      dims.eps,
    )?;
    let attention = Attention::from_weights(weights, &format!("{prefix}.self_attn"), dims)?;
    let layer_norm2 = build_layer_norm(
      weights,
      &format!("{prefix}.layer_norm2"),
      dims.hidden,
      dims.eps,
    )?;
    let mlp = Mlp::from_weights(
      weights,
      &format!("{prefix}.mlp"),
      dims.hidden,
      dims.intermediate,
    )?;
    Ok(Self {
      layer_norm1,
      attention,
      layer_norm2,
      mlp,
    })
  }

  /// `h = x + attn(ln1(x), mask); out = h + mlp(ln2(h))`.
  pub(crate) fn forward(&self, x: &Array, mask: Mask<'_>) -> Result<Array> {
    let r = self
      .attention
      .forward(&self.layer_norm1.forward(x)?, mask)?;
    let h = x.add(&r)?;
    let r = self.mlp.forward(&self.layer_norm2.forward(&h)?)?;
    h.add(&r)
  }
}

/// Build a `LayerNorm` from `{prefix}.weight` + `{prefix}.bias`, each pinned
/// to `(hidden,)`.
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn build_layer_norm(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
  hidden: i32,
  eps: f32,
) -> Result<LayerNorm> {
  let weight = take_shaped(
    weights,
    &format!("{prefix}.weight"),
    "LayerNorm weight (hidden,)",
    &[hidden],
  )?;
  let bias = take_shaped(
    weights,
    &format!("{prefix}.bias"),
    "LayerNorm bias (hidden,)",
    &[hidden],
  )?;
  Ok(LayerNorm::new(Some(weight), Some(bias), eps))
}

/// Pull `shape[axis]` as `i32`, erroring on rank underflow or `i32::MAX`
/// overflow.
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn dim_i32(shape: &[usize], axis: usize, context: &'static str) -> Result<i32> {
  let d = *shape.get(axis).ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    ))
  })?;
  i32::try_from(d).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      context,
      "dim exceeds i32::MAX",
      smol_str::format_smolstr!("{d}"),
    ))
  })
}

/// Pull a weight by exact key from the (sanitized) checkpoint map, erroring
/// with the key if absent. Mirrors the wav2vec2 / lfm2 `take`.
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights.remove(key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "Siglip2NaflexModel::from_weights",
      key,
    ))
  })
}

/// Assert a checkpoint tensor's shape (rank + every dimension) equals the
/// `expected` shape the architecture requires, before it is stored or fed to
/// any op. Mirrors the wav2vec2 `expect_shape`: a corrupt / hostile tensor
/// that survives the config gate cannot run a *different* graph (a wrong
/// projection axis) or drive an oversized allocation. On mismatch returns an
/// [`Error::ShapePairMismatch`] (both full shapes) wrapped in an
/// [`Error::LayerKeyed`] naming the offending `key`.
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn expect_shape(
  tensor: &Array,
  key: &str,
  descriptor: &'static str,
  expected: &[i32],
) -> Result<()> {
  let actual = tensor.shape();
  // Compare in i64 so usize dims and i32 expectations widen losslessly; the
  // length check pins the rank. A negative expected dim is a builder bug and
  // never matches.
  let matches = actual.len() == expected.len()
    && actual
      .iter()
      .zip(expected.iter())
      .all(|(&a, &e)| e >= 0 && a as i64 == i64::from(e));
  if !matches {
    let expected_usize: Vec<usize> = expected.iter().map(|&e| e.max(0) as usize).collect();
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      key,
      Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        descriptor,
        expected_usize,
        actual,
      )),
    )));
  }
  Ok(())
}

/// [`take`] a weight by key, then assert its shape equals `expected` via
/// [`expect_shape`] — the fused fetch-and-shape-check the builders use for
/// every tensor stored verbatim, so a consumed tensor can never skip the gate.
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn take_shaped(
  weights: &mut HashMap<String, Array>,
  key: &str,
  descriptor: &'static str,
  expected: &[i32],
) -> Result<Array> {
  let tensor = take(weights, key)?;
  expect_shape(&tensor, key, descriptor, expected)?;
  Ok(tensor)
}
