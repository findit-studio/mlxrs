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
  dtype::Dtype,
  error::{
    Error, InvariantViolationPayload, LayerKeyedPayload, MissingKeyPayload, OutOfRangePayload,
    RankMismatchPayload, Result, ShapePairMismatchPayload,
  },
  lm::{
    nn::{
      attention::{Mask, scaled_dot_product_attention},
      norm::LayerNorm,
    },
    quant::PerLayerQuantization,
  },
  model_validation::checked_mul,
  nn::{MaybeQuantizedEmbedding, MaybeQuantizedLinear, QuantizedLinear},
  ops,
};

/// `y = x @ wᵀ (+ bias)` — the private `nn.Linear` forward both towers use.
///
/// `siglip.py` stores every `nn.Linear` weight as `(out, in)`, so the matmul
/// is `x @ wᵀ`. With a bias this is the fused `addmm(bias, x, wᵀ, 1, 1)`;
/// without, a plain `matmul(x, wᵀ)`. Returns a new lazy [`Array`] (no eval).
///
/// This is the **dense** projection used directly by the few weight-sliced
/// projections a packed quantized weight cannot represent (the attention-pool
/// head's combined-QKV `in_proj`, sliced by logical row); every other
/// projection goes through the quantize-aware [`QuantLinear`].
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array> {
  let wt = ops::shape::swapaxes(weight, -1, -2)?;
  match bias {
    Some(b) => ops::linalg_basic::addmm(b, x, &wt, 1.0, 1.0),
    None => ops::linalg_basic::matmul(x, &wt),
  }
}

/// A SigLIP2 `nn.Linear` projection `y = x @ Wᵀ (+ b)` — quantize-aware.
///
/// Mirrors `mlx.nn.Linear` (`weight` stored `(out, in)`, the forward transposes
/// it) for a dense checkpoint and `mlx.nn.QuantizedLinear` for an mlx-community
/// quantized checkpoint (mlx-embeddings' `quantize_model` /
/// `get_class_predicate`). The two cases share one [`forward`](Self::forward)
/// call site via the shared [`MaybeQuantizedLinear`], so every SigLIP2
/// projection (attention q/k/v/out, MLP fc1/fc2, the patch-embed matmul, the
/// attention-pool out_proj, and the text head) is unchanged whether the weights
/// are dense or quantized — the Whisper adoption pattern, identically.
///
/// Built by [`from_weights`](Self::from_weights), which auto-picks the quantized
/// variant per layer by the presence of a `<prefix>.scales` sibling (the
/// `class_predicate`'s `f"{p}.scales" in weights` signal).
#[cfg(feature = "siglip2-naflex")]
pub(crate) struct QuantLinear {
  inner: MaybeQuantizedLinear,
}

#[cfg(feature = "siglip2-naflex")]
impl QuantLinear {
  /// Build a projection from `<prefix>.weight` (+ the dense `<prefix>.bias` when
  /// `bias` is `true`), auto-picking the dense or quantized variant.
  ///
  /// **Dense path** (no `<prefix>.scales` sibling): pops `<prefix>.weight`
  /// `(out, in)` (+ the `<prefix>.bias` `(out,)` when `bias`), both shape-pinned
  /// to the config-derived extents via [`take_shaped`] BEFORE materialization —
  /// byte-identical to the original dense load.
  ///
  /// **Quantized path** (`<prefix>.scales` present — the sole `class_predicate`
  /// signal, requiring `quant` to resolve scheme params): the packed `uint32`
  /// weight `(out, in * bits / 32)` is structurally gated to the config-derived
  /// logical `(out, in)` by [`check_quantized_shape`]
  /// (the quantized analogue of the dense shape-pin), the dense `<prefix>.bias`
  /// is loaded with the SAME arity the dense path enforces (required + pinned
  /// `(out,)` when `bias`, dropped otherwise), and the triple is built via the
  /// shared [`QuantizedLinear::from_parts`].
  ///
  /// `quant` carries the parsed per-layer quantization config; a `<prefix>.scales`
  /// present but no resolvable scheme params (`quant == None`, an explicit
  /// `Skip`, or no global default) is a typed [`Error::InvariantViolation`],
  /// never a guessed scheme or a silent fall-through to the dense loader.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    out: i32,
    in_features: i32,
    bias: bool,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let scales_key = format!("{prefix}.scales");
    if weights.contains_key(&scales_key) {
      // Quantized path (the `.scales` sibling is the sole `class_predicate`
      // signal, as in the shared `MaybeQuantizedLinear::from_weights`): resolve
      // `(group_size, bits, mode)` for this layer. A `.scales` present with no
      // resolvable scheme (`quant == None`, or the config resolves no params)
      // is a typed error, never a silent fall-through to the dense loader.
      let Some(q) = quant.and_then(|q| q.quantization_for(prefix)) else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "siglip2 Linear carries a `.scales` sibling but the quantization config resolved no scheme parameters for this layer",
          "a quantized Linear requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      // Pin the packed weight's logical `(out, in)` (and the scales' recovery)
      // to the config BEFORE construction — the same load-time gate the dense
      // `take_shaped` enforces.
      check_quantized_shape(
        weights,
        prefix,
        "siglip2 quantized Linear weight (out, in)",
        out,
        in_features,
        q.group_size,
        q.bits,
      )?;
      // Load the dense output bias with the SAME arity as the dense branch:
      // required + pinned `(out,)` when `bias`, dropped otherwise. Passed as the
      // explicit dense bias to `from_parts` (not auto-detected), so dense and
      // quantized are arity-identical.
      let dense_bias = if bias {
        Some(take_shaped(
          weights,
          &format!("{prefix}.bias"),
          "siglip2 Linear bias (out,)",
          &[out],
        )?)
      } else {
        weights.remove(&format!("{prefix}.bias"));
        None
      };
      // Pop the packed triple by key: `uint32` weight, `.scales`, and the
      // per-group affine `.biases` (present iff `mode == "affine"`; `from_parts`
      // enforces the mode/arity contract).
      let weight = take(weights, &format!("{prefix}.weight"))?;
      let scales = take(weights, &format!("{prefix}.scales"))?;
      let quant_biases = weights.remove(&format!("{prefix}.biases"));
      let q = QuantizedLinear::from_parts(
        weight,
        scales,
        quant_biases,
        dense_bias,
        q.group_size,
        q.bits,
        q.mode.as_str(),
      )?;
      return Ok(Self {
        inner: MaybeQuantizedLinear::Quantized(q),
      });
    }

    // Dense path (unchanged): shape-pin against the config-derived `(out, in)`.
    let weight = take_shaped(
      weights,
      &format!("{prefix}.weight"),
      "siglip2 Linear weight (out, in)",
      &[out, in_features],
    )?;
    let b = if bias {
      Some(take_shaped(
        weights,
        &format!("{prefix}.bias"),
        "siglip2 Linear bias (out,)",
        &[out],
      )?)
    } else {
      None
    };
    Ok(Self {
      inner: MaybeQuantizedLinear::Dense(crate::nn::Linear::new(weight, b)),
    })
  }

  /// Wrap a pre-built **dense** `(out, in)` weight (+ optional `(out,)` bias) as
  /// a [`QuantLinear`] — for the patch-embed projection, whose dense weight is
  /// the Conv2d kernel reshaped to `(hidden, P^2 * C)` by the caller before it
  /// reaches a Linear (the quantized patch-embed weight is already packed-2D and
  /// takes the auto-detecting [`Self::from_weights`] path instead).
  pub(crate) fn dense(weight: Array, bias: Option<Array>) -> Self {
    Self {
      inner: MaybeQuantizedLinear::Dense(crate::nn::Linear::new(weight, bias)),
    }
  }

  /// `y = x @ weightᵀ (+ bias)` (dense) or `quantized_matmul(...) (+ bias)`
  /// (quantized). `x` is `(..., in)`; the result is `(..., out)`.
  pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
    self.inner.forward(x)
  }

  /// The projection's **parameter dtype** — the float precision the layer
  /// computes in, i.e. what activations should be cast to so a reduced-precision
  /// checkpoint is not silently widened by MLX type promotion
  /// (`f16 op f32 → f32`):
  ///
  /// - **dense**: the weight's own dtype;
  /// - **quantized**: the `scales` dtype (the packed weight is `uint32`;
  ///   `quantized_matmul` dequantizes through the scales' float dtype, which is
  ///   the checkpoint's compute precision — the same resolution mlx-lm /
  ///   mlx-embeddings use for a quantized layer's effective dtype).
  ///
  /// Reads only dtype metadata (no materialization / eval).
  pub(crate) fn param_dtype(&self) -> Result<Dtype> {
    match &self.inner {
      MaybeQuantizedLinear::Dense(l) => l.weight_ref().dtype(),
      MaybeQuantizedLinear::Quantized(q) => q.scales_ref().dtype(),
    }
  }
}

/// Validate a quantized layer's packed `<prefix>.weight` + `<prefix>.scales`
/// against the config-derived `(out, in_features)` BEFORE the quantized layer is
/// constructed — the quantized analogue of the dense [`expect_shape`] gate, and
/// the structural twin of Whisper's `Builder::check_quantized_shape`.
///
/// The dense path pins every consumed tensor to its exact config shape via
/// [`take_shaped`]; the quantized path must reach the same load-time gate,
/// because a corrupt quantized checkpoint could otherwise ship a packed weight
/// whose *logical* output / input dimension disagrees with the config, and the
/// first forward would then size projections from the checkpoint tensors instead
/// of the validated config. The packed `uint32` weight has a different shape than
/// the dense `(out, in)`, so the recovery mirrors mlx's quantized layout
/// (`mlx/ops.cpp:107,131,4790-4792`):
///
/// - the weight is rank-2 `uint32` `(out, in * bits / 32)`; its leading axis is
///   the logical output dim (must equal `out`) and its logical input width —
///   mlx's `w_inner_dims = w.shape(-1) * 32 / bits` — must equal `in_features`;
/// - the `scales` are rank-2 `(out, in / group_size)`; the leading axis must
///   equal `out`, and `scales.shape(-1) * group_size` must equal `in_features`.
///
/// `group_size` / `bits` are checked `> 0` before they divide (a non-positive
/// value is a malformed config and an [`Error::OutOfRange`], never a panic). The
/// per-mode value tables remain mlx-c's; this only pins the structural
/// relationship to the config the dense gate also enforces. Reads only `shape()`
/// / `dtype()` metadata (no materialization).
#[cfg(feature = "siglip2-naflex")]
fn check_quantized_shape(
  weights: &HashMap<String, Array>,
  prefix: &str,
  descriptor: &'static str,
  out: i32,
  in_features: i32,
  group_size: i32,
  bits: i32,
) -> Result<()> {
  if bits <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "siglip2 quantized layer bits",
      "must be > 0",
      smol_str::format_smolstr!("{bits}"),
    )));
  }
  if group_size <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "siglip2 quantized layer group_size",
      "must be > 0",
      smol_str::format_smolstr!("{group_size}"),
    )));
  }

  // Packed weight `(out, in * bits / 32)`, `uint32`.
  let weight_key = format!("{prefix}.weight");
  let weight = weights.get(&weight_key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "siglip2: quantized weight not found in checkpoint",
      &weight_key,
    ))
  })?;
  let w_shape = weight.shape();
  if w_shape.len() != 2 {
    let rank = w_shape.len() as u32;
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &weight_key,
      Error::RankMismatch(RankMismatchPayload::new(
        "quantized weight must be rank-2 (out, in * bits / 32)",
        rank,
        w_shape,
      )),
    )));
  }
  if weight.dtype()? != Dtype::U32 {
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &weight_key,
      Error::InvariantViolation(InvariantViolationPayload::new(
        "quantized weight dtype",
        "must be `uint32` (the packed-quantized-weight dtype)",
      )),
    )));
  }
  // Logical output dim is the leading axis; logical input width is mlx's
  // `w_inner_dims = w.shape(-1) * 32 / bits`. Compare in i64 so the recovery
  // cannot overflow on a corrupt huge packed width.
  let logical_in = (w_shape[1] as i64) * 32 / i64::from(bits);
  if w_shape[0] as i64 != i64::from(out) || logical_in != i64::from(in_features) {
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &weight_key,
      Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        descriptor,
        vec![out.max(0) as usize, in_features.max(0) as usize],
        vec![w_shape[0], logical_in.max(0) as usize],
      )),
    )));
  }

  // Scales `(out, in / group_size)`: leading axis is `out`, and the per-group
  // count recovers the same logical input width as the packed weight.
  let scales_key = format!("{prefix}.scales");
  let scales = weights.get(&scales_key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "siglip2: quantized scales not found in checkpoint",
      &scales_key,
    ))
  })?;
  let s_shape = scales.shape();
  if s_shape.len() != 2 {
    let rank = s_shape.len() as u32;
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &scales_key,
      Error::RankMismatch(RankMismatchPayload::new(
        "quantized scales must be rank-2 (out, in / group_size)",
        rank,
        s_shape,
      )),
    )));
  }
  let scales_in = (s_shape[1] as i64) * i64::from(group_size);
  if s_shape[0] as i64 != i64::from(out) || scales_in != i64::from(in_features) {
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &scales_key,
      Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "quantized scales (out, in / group_size) must match the config",
        vec![out.max(0) as usize, in_features.max(0) as usize],
        vec![s_shape[0], scales_in.max(0) as usize],
      )),
    )));
  }

  Ok(())
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
/// then `Linear(intermediate → hidden)`. Both projections are quantize-aware
/// ([`QuantLinear`]).
#[cfg(feature = "siglip2-naflex")]
pub(crate) struct Mlp {
  fc1: QuantLinear,
  fc2: QuantLinear,
}

#[cfg(feature = "siglip2-naflex")]
impl Mlp {
  /// Build from `{prefix}.fc1.*` + `{prefix}.fc2.*`, pinning the two Linear
  /// shapes to `(intermediate, hidden)` / `(hidden, intermediate)` and the
  /// biases to `(intermediate,)` / `(hidden,)`. Each projection auto-picks the
  /// dense or quantized variant by its `.scales` sibling.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    hidden: i32,
    intermediate: i32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let fc1 = QuantLinear::from_weights(
      weights,
      &format!("{prefix}.fc1"),
      intermediate,
      hidden,
      true,
      quant,
    )?;
    let fc2 = QuantLinear::from_weights(
      weights,
      &format!("{prefix}.fc2"),
      hidden,
      intermediate,
      true,
      quant,
    )?;
    Ok(Self { fc1, fc2 })
  }

  /// `fc2(gelu_precise(fc1(x)))`.
  pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
    let h = self.fc1.forward(x)?;
    let h = gelu_precise(&h)?;
    self.fc2.forward(&h)
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
  q: QuantLinear,
  k: QuantLinear,
  v: QuantLinear,
  out: QuantLinear,
  num_heads: i32,
  head_dim: i32,
  /// `head_dim**-0.5`, the SDPA scale.
  scale: f32,
}

#[cfg(feature = "siglip2-naflex")]
impl Attention {
  /// Build from `{prefix}.{q,k,v,out}_proj.{weight,bias}`, pinning each
  /// projection to `(hidden, hidden)` and each bias to `(hidden,)`. Each
  /// projection auto-picks the dense or quantized variant by its `.scales`
  /// sibling.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    dims: LayerDims,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let hidden = dims.hidden;
    let proj = |weights: &mut HashMap<String, Array>, name: &str| {
      QuantLinear::from_weights(
        weights,
        &format!("{prefix}.{name}"),
        hidden,
        hidden,
        true,
        quant,
      )
    };
    Ok(Self {
      q: proj(weights, "q_proj")?,
      k: proj(weights, "k_proj")?,
      v: proj(weights, "v_proj")?,
      out: proj(weights, "out_proj")?,
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

    let q = self.q.forward(x)?;
    let k = self.k.forward(x)?;
    let v = self.v.forward(x)?;

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
    self.out.forward(&attn)
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
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let prefix = format!("{encoder_prefix}.layers.{i}");
    let layer_norm1 = build_layer_norm(
      weights,
      &format!("{prefix}.layer_norm1"),
      dims.hidden,
      dims.eps,
    )?;
    let attention = Attention::from_weights(weights, &format!("{prefix}.self_attn"), dims, quant)?;
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
      quant,
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

/// Resolve the per-layer `(group_size, bits, mode)` scheme parameters for an
/// embedding `prefix` from the parsed quantization config, for
/// [`MaybeQuantizedEmbedding::from_weights`](crate::nn::MaybeQuantizedEmbedding::from_weights).
///
/// Returns `None` when there is no quantization config (`quant == None`) — the
/// embedding then loads dense if it has no `.scales` sibling, or errors inside
/// `from_weights` if it does (a `.scales`-without-config inconsistency). Returns
/// `Some((group_size, bits, mode))` for a config that resolves a scheme for this
/// layer. A config that is present but resolves `None` for the layer (an
/// explicit `Skip`, or no global default) maps to `None`, so a `.scales`-bearing
/// embedding under such a config surfaces the same typed `.scales`-without-params
/// error from `from_weights` the linear path returns.
///
/// `mode` is `QuantMode::as_str`'s `&'static str`, so the returned borrow
/// outlives the call.
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn resolve_quant<'a>(
  quant: Option<&'a PerLayerQuantization>,
  prefix: &str,
) -> Option<(i32, i32, &'a str)> {
  quant
    .and_then(|q| q.quantization_for(prefix))
    .map(|q| (q.group_size, q.bits, q.mode.as_str()))
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

/// Assert a quantize-aware embedding's **logical** `(num_embeddings, dim)`
/// equals the config-derived `(rows, dim)`, for BOTH the dense and quantized
/// arms — the embedding analogue of [`expect_shape`] / [`check_quantized_shape`].
///
/// A dense token embedding is pinned to `(vocab, hidden)` by [`expect_shape`] on
/// its table; the quantized arm has no such gate unless its dequantized logical
/// shape is checked too. A packed quantized table whose logical row count or
/// width disagrees with the config would otherwise mis-gather (wrong rows or an
/// out-of-range gather) at the first forward instead of failing at load. This
/// reads the logical shape via [`MaybeQuantizedEmbedding::logical_shape`] (no
/// whole-table dequantization — the quantized width is recovered from the
/// validated triple's metadata) and rejects a mismatch with the same typed
/// [`Error::ShapePairMismatch`] wrapped in [`Error::LayerKeyed`] the dense /
/// quantized-`Linear` gates use, keyed on `<prefix>.weight`.
#[cfg(feature = "siglip2-naflex")]
pub(crate) fn expect_logical_shape(
  embedding: &MaybeQuantizedEmbedding,
  key: &str,
  descriptor: &'static str,
  rows: i32,
  dim: i32,
) -> Result<()> {
  let (actual_rows, actual_dim) = embedding.logical_shape()?;
  if actual_rows != rows || actual_dim != dim {
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      key,
      Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        descriptor,
        vec![rows.max(0) as usize, dim.max(0) as usize],
        vec![actual_rows.max(0) as usize, actual_dim.max(0) as usize],
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
