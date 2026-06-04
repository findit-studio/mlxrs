//! Shared building blocks for the CLAP RoBERTa text tower — the quantize-aware
//! `nn.Linear` wrapper, the BERT/RoBERTa **post-norm** self-attention +
//! intermediate + output sub-blocks, the `LayerNorm` builder, and the weight
//! fetch + shape-pinning helpers (the same discipline as the merged
//! SigLIP2 / Wav2Vec2 / LFM2 ports: every consumed tensor's shape is checked
//! before it is stored or fed to any op, with a typed
//! [`Error::ShapePairMismatch`] wrapped in [`Error::LayerKeyed`]).
//!
//! These mirror HF `transformers`' `ClapTextModel` (a RoBERTa encoder, the
//! BERT-family layout): `RobertaSelfAttention` + `RobertaSelfOutput`
//! (post-norm), `RobertaIntermediate` (`Linear → exact GELU`), and
//! `RobertaOutput` (post-norm). The deltas from the SigLIP2 `shared` template
//! (which is pre-norm with `Mask::None`) are RoBERTa-specific: the `LayerNorm`
//! moves **after** the residual (post-norm), the FFN activation is **exact**
//! [`gelu`](crate::lm::nn::activations::gelu) (`hidden_act = "gelu"`, not the
//! `tanh` approximation), and the attention takes an **additive padding mask**
//! (RoBERTa masks pad keys, unlike SigLIP2's sticky-EOS full attention).
//!
//! Like SigLIP2's blocks these are deliberately not the public `lm::nn` layers:
//! mlxrs ships no public `nn::Linear` (each model composes the matmul), and the
//! quantize-aware projection is built per-prefix via the `.scales` sibling
//! `class_predicate` signal.

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

/// A CLAP RoBERTa `nn.Linear` projection `y = x @ Wᵀ (+ b)` — quantize-aware.
///
/// Mirrors `mlx.nn.Linear` (`weight` stored `(out, in)`, the forward transposes
/// it) for a dense checkpoint and `mlx.nn.QuantizedLinear` for a quantized one,
/// sharing one [`forward`](Self::forward) call site via [`MaybeQuantizedLinear`]
/// — so every RoBERTa projection (the attention q/k/v/out, the intermediate and
/// output dense layers, and the two projection-MLP layers) is unchanged whether
/// the weights are dense or quantized (the Whisper / SigLIP2 adoption pattern,
/// identically). This is the structural twin of
/// [`crate::embeddings::siglip2_naflex`]'s `QuantLinear`, re-declared here under
/// the `clap` feature gate (the SigLIP2 one is gated to `siglip2-naflex`).
///
/// Built by [`from_weights`](Self::from_weights), which auto-picks the quantized
/// variant per layer by the presence of a `<prefix>.scales` sibling.
#[cfg(feature = "clap")]
pub(crate) struct QuantLinear {
  inner: MaybeQuantizedLinear,
}

#[cfg(feature = "clap")]
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
  /// logical `(out, in)` by [`check_quantized_shape`], the dense `<prefix>.bias`
  /// is loaded with the SAME arity the dense path enforces (required + pinned
  /// `(out,)` when `bias`, dropped otherwise), and the triple is built via the
  /// shared [`QuantizedLinear::from_parts`].
  ///
  /// `quant` carries the parsed per-layer quantization config; a `<prefix>.scales`
  /// present but no resolvable scheme params is a typed
  /// [`Error::InvariantViolation`], never a guessed scheme or a silent
  /// fall-through to the dense loader.
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
      // signal): resolve `(group_size, bits, mode)` for this layer. A `.scales`
      // present with no resolvable scheme is a typed error, never a silent
      // fall-through to the dense loader.
      let Some(q) = quant.and_then(|q| q.quantization_for(prefix)) else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "clap RoBERTa Linear carries a `.scales` sibling but the quantization config resolved no scheme parameters for this layer",
          "a quantized Linear requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      // Pin the packed weight's logical `(out, in)` (and the scales' recovery)
      // to the config BEFORE construction — the same load-time gate the dense
      // `take_shaped` enforces.
      check_quantized_shape(
        weights,
        prefix,
        "clap RoBERTa quantized Linear weight (out, in)",
        out,
        in_features,
        q.group_size,
        q.bits,
      )?;
      // Load the dense output bias with the SAME arity as the dense branch:
      // required + pinned `(out,)` when `bias`, dropped otherwise.
      let dense_bias = if bias {
        Some(take_shaped(
          weights,
          &format!("{prefix}.bias"),
          "clap RoBERTa Linear bias (out,)",
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
      "clap RoBERTa Linear weight (out, in)",
      &[out, in_features],
    )?;
    let b = if bias {
      Some(take_shaped(
        weights,
        &format!("{prefix}.bias"),
        "clap RoBERTa Linear bias (out,)",
        &[out],
      )?)
    } else {
      None
    };
    Ok(Self {
      inner: MaybeQuantizedLinear::Dense(crate::nn::Linear::new(weight, b)),
    })
  }

  /// `y = x @ weightᵀ (+ bias)` (dense) or `quantized_matmul(...) (+ bias)`
  /// (quantized). `x` is `(..., in)`; the result is `(..., out)`.
  pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
    self.inner.forward(x)
  }

  /// `true` if this projection loaded the quantized variant (test-only
  /// introspection for the quantized-load test).
  #[cfg(test)]
  pub(crate) fn is_quantized(&self) -> bool {
    self.inner.is_quantized()
  }
}

/// Validate a quantized layer's packed `<prefix>.weight` + `<prefix>.scales`
/// against the config-derived `(out, in_features)` BEFORE the quantized layer is
/// constructed — the quantized analogue of the dense [`expect_shape`] gate, and
/// the structural twin of SigLIP2's / Whisper's `check_quantized_shape`.
///
/// The packed `uint32` weight is rank-2 `(out, in * bits / 32)`; its leading
/// axis is the logical output dim (must equal `out`) and its logical input
/// width — mlx's `w_inner_dims = w.shape(-1) * 32 / bits` — must equal
/// `in_features`. The `scales` are rank-2 `(out, in / group_size)`; the leading
/// axis must equal `out` and `scales.shape(-1) * group_size` must equal
/// `in_features`. `group_size` / `bits` are checked `> 0` before they divide.
/// Reads only `shape()` / `dtype()` metadata (no materialization).
#[cfg(feature = "clap")]
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
      "clap RoBERTa quantized layer bits",
      "must be > 0",
      smol_str::format_smolstr!("{bits}"),
    )));
  }
  if group_size <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "clap RoBERTa quantized layer group_size",
      "must be > 0",
      smol_str::format_smolstr!("{group_size}"),
    )));
  }

  // Packed weight `(out, in * bits / 32)`, `uint32`.
  let weight_key = format!("{prefix}.weight");
  let weight = weights.get(&weight_key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "clap: quantized weight not found in checkpoint",
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
      "clap: quantized scales not found in checkpoint",
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

/// The per-layer shape constants the RoBERTa encoder stack shares: the
/// transformer width, the feed-forward intermediate width, the head split, and
/// the LayerNorm eps. Bundled so the layer builders take one config arg instead
/// of five positional scalars (and so the tower computes the head split once).
#[cfg(feature = "clap")]
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

#[cfg(feature = "clap")]
impl LayerDims {
  /// Derive the per-layer dims from `(hidden, intermediate, num_heads, eps)`,
  /// computing the head split and SDPA scale once. `num_heads` must be positive
  /// and divide `hidden` (the caller validates this against the config).
  pub(crate) fn new(hidden: i32, intermediate: i32, num_heads: i32, eps: f32) -> Result<Self> {
    crate::model_validation::require_positive("clap RoBERTa: num_attention_heads", num_heads)?;
    crate::model_validation::require_divisible(
      "clap RoBERTa: hidden_size",
      hidden,
      "clap RoBERTa: num_attention_heads",
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

/// RoBERTa self-attention + the post-norm self-output
/// (HF `RobertaSelfAttention` + `RobertaSelfOutput`).
///
/// `q/k/v` are biased `Linear(hidden, hidden)`; the scaled dot-product runs over
/// the head-split projections with an **additive padding mask** (`Mask::Array`,
/// `0` on real keys / `-inf` on pad keys), so RoBERTa attends bidirectionally
/// over the real tokens only. The output dense (`output.dense`, biased
/// `Linear(hidden, hidden)`) projects the concatenated heads; the residual add
/// and `LayerNorm` (post-norm) happen in [`RobertaLayer::forward`] (mirroring
/// `RobertaSelfOutput`, whose dense lives here and whose `dropout`/residual/`LN`
/// the layer composes).
#[cfg(feature = "clap")]
pub(crate) struct Attention {
  q: QuantLinear,
  k: QuantLinear,
  v: QuantLinear,
  /// `output.dense` — the post-attention output projection.
  out: QuantLinear,
  num_heads: i32,
  head_dim: i32,
  /// `head_dim**-0.5`, the SDPA scale.
  scale: f32,
}

#[cfg(feature = "clap")]
impl Attention {
  /// Build from `{prefix}.self.{query,key,value}.{weight,bias}` and
  /// `{prefix}.output.dense.{weight,bias}`, pinning each projection to
  /// `(hidden, hidden)` and each bias to `(hidden,)`. Each projection auto-picks
  /// the dense or quantized variant by its `.scales` sibling.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    dims: LayerDims,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let hidden = dims.hidden;
    let proj = |weights: &mut HashMap<String, Array>, name: &str| {
      QuantLinear::from_weights(weights, name, hidden, hidden, true, quant)
    };
    Ok(Self {
      q: proj(weights, &format!("{prefix}.self.query"))?,
      k: proj(weights, &format!("{prefix}.self.key"))?,
      v: proj(weights, &format!("{prefix}.self.value"))?,
      out: proj(weights, &format!("{prefix}.output.dense"))?,
      num_heads: dims.num_heads,
      head_dim: dims.head_dim,
      scale: dims.scale,
    })
  }

  /// `(B, L, C) → (B, L, C)` self-attention with the additive key `mask`,
  /// returning the **output-dense** projection (before the residual + LayerNorm,
  /// which the layer applies).
  pub(crate) fn forward(&self, x: &Array, mask: Mask<'_>) -> Result<Array> {
    let shape = x.shape();
    let bsz = dim_i32(&shape, 0, "clap RoBERTa Attention: batch")?;
    let seq = dim_i32(&shape, 1, "clap RoBERTa Attention: seq")?;

    let q = self.q.forward(x)?;
    let k = self.k.forward(x)?;
    let v = self.v.forward(x)?;

    let q = self.split_heads(&q, bsz, seq)?;
    let k = self.split_heads(&k, bsz, seq)?;
    let v = self.split_heads(&v, bsz, seq)?;

    let attn = scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;
    let attn = ops::shape::transpose_axes(&attn, &[0, 2, 1, 3])?;
    let embed_dim = checked_mul(
      "clap RoBERTa Attention: num_heads * head_dim",
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

  /// `true` if every projection loaded the quantized variant (test-only).
  #[cfg(test)]
  pub(crate) fn all_quantized(&self) -> bool {
    self.q.is_quantized()
      && self.k.is_quantized()
      && self.v.is_quantized()
      && self.out.is_quantized()
  }
}

/// The RoBERTa feed-forward: the `intermediate.dense` expansion with exact GELU
/// and the `output.dense` contraction (HF `RobertaIntermediate` +
/// `RobertaOutput`'s dense).
///
/// `intermediate.dense` is a biased `Linear(hidden → intermediate)` followed by
/// **exact** [`gelu`](crate::lm::nn::activations::gelu) (`hidden_act = "gelu"`,
/// the `approx="none"` arm — NOT the `tanh` approximation SigLIP2/Gemma use).
/// `output.dense` is a biased `Linear(intermediate → hidden)`; the residual add +
/// `LayerNorm` (post-norm, mirroring `RobertaOutput`) are applied by
/// [`RobertaLayer::forward`].
#[cfg(feature = "clap")]
pub(crate) struct FeedForward {
  /// `intermediate.dense` — the `hidden → intermediate` expansion.
  intermediate: QuantLinear,
  /// `output.dense` — the `intermediate → hidden` contraction.
  output: QuantLinear,
}

#[cfg(feature = "clap")]
impl FeedForward {
  /// Build from `{layer}.intermediate.dense.*` + `{layer}.output.dense.*`,
  /// pinning the two Linear shapes to `(intermediate, hidden)` /
  /// `(hidden, intermediate)` and the biases to `(intermediate,)` / `(hidden,)`.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    layer_prefix: &str,
    hidden: i32,
    intermediate: i32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let intermediate_dense = QuantLinear::from_weights(
      weights,
      &format!("{layer_prefix}.intermediate.dense"),
      intermediate,
      hidden,
      true,
      quant,
    )?;
    let output_dense = QuantLinear::from_weights(
      weights,
      &format!("{layer_prefix}.output.dense"),
      hidden,
      intermediate,
      true,
      quant,
    )?;
    Ok(Self {
      intermediate: intermediate_dense,
      output: output_dense,
    })
  }

  /// `output.dense(gelu(intermediate.dense(x)))` — the un-residualized FFN.
  pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
    let h = self.intermediate.forward(x)?;
    let h = crate::lm::nn::activations::gelu(&h)?;
    self.output.forward(&h)
  }

  /// `true` if both dense layers loaded the quantized variant (test-only).
  #[cfg(test)]
  pub(crate) fn all_quantized(&self) -> bool {
    self.intermediate.is_quantized() && self.output.is_quantized()
  }
}

/// A RoBERTa **post-norm** transformer layer (HF `RobertaLayer`):
///
/// ```text
/// a = self_attn(x, mask)                       # RobertaSelfAttention + output.dense
/// h = attention_layer_norm(x + a)              # RobertaSelfOutput: add → LayerNorm
/// f = ffn(h)                                   # RobertaIntermediate(gelu) + output.dense
/// out = output_layer_norm(h + f)               # RobertaOutput: add → LayerNorm
/// ```
///
/// The norm is applied **after** the residual (post-norm), unlike SigLIP2's
/// pre-norm `EncoderLayer`.
#[cfg(feature = "clap")]
pub(crate) struct RobertaLayer {
  attention: Attention,
  /// `attention.output.LayerNorm` — the post-attention norm.
  attention_layer_norm: LayerNorm,
  ffn: FeedForward,
  /// `output.LayerNorm` — the post-FFN norm.
  output_layer_norm: LayerNorm,
}

#[cfg(feature = "clap")]
impl RobertaLayer {
  /// Build the `i`-th layer from `{encoder_prefix}.layer.{i}.*` (the HF
  /// `encoder.layer.{i}` tree): `attention.self.{query,key,value}`,
  /// `attention.output.dense` + `attention.output.LayerNorm`,
  /// `intermediate.dense`, and `output.dense` + `output.LayerNorm`.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    encoder_prefix: &str,
    i: i32,
    dims: LayerDims,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let layer = format!("{encoder_prefix}.layer.{i}");
    let attention = Attention::from_weights(weights, &format!("{layer}.attention"), dims, quant)?;
    let attention_layer_norm = build_layer_norm(
      weights,
      &format!("{layer}.attention.output.LayerNorm"),
      dims.hidden,
      dims.eps,
    )?;
    let ffn = FeedForward::from_weights(weights, &layer, dims.hidden, dims.intermediate, quant)?;
    let output_layer_norm = build_layer_norm(
      weights,
      &format!("{layer}.output.LayerNorm"),
      dims.hidden,
      dims.eps,
    )?;
    Ok(Self {
      attention,
      attention_layer_norm,
      ffn,
      output_layer_norm,
    })
  }

  /// `h = attn_ln(x + attn(x, mask)); out = out_ln(h + ffn(h))` — the RoBERTa
  /// post-norm block.
  pub(crate) fn forward(&self, x: &Array, mask: Mask<'_>) -> Result<Array> {
    let a = self.attention.forward(x, mask)?;
    let h = self.attention_layer_norm.forward(&x.add(&a)?)?;
    let f = self.ffn.forward(&h)?;
    self.output_layer_norm.forward(&h.add(&f)?)
  }

  /// `true` if every projection in the layer loaded quantized (test-only).
  #[cfg(test)]
  pub(crate) fn all_quantized(&self) -> bool {
    self.attention.all_quantized() && self.ffn.all_quantized()
  }
}

/// Build a `LayerNorm` from `{prefix}.weight` + `{prefix}.bias`, each pinned to
/// `(hidden,)`.
#[cfg(feature = "clap")]
pub(crate) fn build_layer_norm(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
  hidden: i32,
  eps: f32,
) -> Result<LayerNorm> {
  let weight = take_shaped(
    weights,
    &format!("{prefix}.weight"),
    "clap RoBERTa LayerNorm weight (hidden,)",
    &[hidden],
  )?;
  let bias = take_shaped(
    weights,
    &format!("{prefix}.bias"),
    "clap RoBERTa LayerNorm bias (hidden,)",
    &[hidden],
  )?;
  Ok(LayerNorm::new(Some(weight), Some(bias), eps))
}

/// Resolve the per-layer `(group_size, bits, mode)` scheme parameters for an
/// embedding `prefix` from the parsed quantization config, for
/// [`MaybeQuantizedEmbedding::from_weights`].
///
/// Returns `None` when there is no quantization config (`quant == None`) — the
/// embedding then loads dense if it has no `.scales` sibling, or errors inside
/// `from_weights` if it does. `mode` is `QuantMode::as_str`'s `&'static str`, so
/// the returned borrow outlives the call.
#[cfg(feature = "clap")]
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
#[cfg(feature = "clap")]
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

/// Pull a weight by exact key from the (sanitized) checkpoint map, erroring with
/// the key if absent.
#[cfg(feature = "clap")]
pub(crate) fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights
    .remove(key)
    .ok_or_else(|| Error::MissingKey(MissingKeyPayload::new("clap RoBERTa::from_weights", key)))
}

/// Assert a checkpoint tensor's shape (rank + every dimension) equals the
/// `expected` shape the architecture requires, before it is stored or fed to any
/// op. On mismatch returns an [`Error::ShapePairMismatch`] (both full shapes)
/// wrapped in an [`Error::LayerKeyed`] naming the offending `key`.
#[cfg(feature = "clap")]
pub(crate) fn expect_shape(
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
/// Reads the logical shape via [`MaybeQuantizedEmbedding::logical_shape`] (no
/// whole-table dequantization) and rejects a mismatch with the same typed
/// [`Error::ShapePairMismatch`] wrapped in [`Error::LayerKeyed`], keyed on
/// `<prefix>.weight`.
#[cfg(feature = "clap")]
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
/// [`expect_shape`] — the fused fetch-and-shape-check the builders use for every
/// tensor stored verbatim, so a consumed tensor can never skip the gate.
#[cfg(feature = "clap")]
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
