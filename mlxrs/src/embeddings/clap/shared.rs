//! Shared building blocks for the two CLAP towers — the quantize-aware
//! `nn.Linear` wrapper, the weight fetch + shape-pinning helpers, the RoBERTa
//! **text**-tower sub-blocks, and the HTSAT **audio**-tower Swin-Transformer
//! sub-blocks. The same discipline runs throughout (the merged SigLIP2 /
//! Wav2Vec2 / LFM2 ports): every consumed tensor's shape is checked before it
//! is stored or fed to any op, with a typed [`Error::ShapePairMismatch`]
//! wrapped in [`Error::LayerKeyed`]; every `nn.Linear` routes through
//! [`MaybeQuantizedLinear`] (dense vs quantized picked by the `.scales`
//! sibling); and a tensor built in f32 that meets an f16/bf16 activation is
//! cast back to the activation dtype before the add/softmax.
//!
//! **Text tower (RoBERTa).** Mirrors HF `transformers`' `ClapTextModel` (a
//! RoBERTa encoder, the BERT-family layout): `RobertaSelfAttention` +
//! `RobertaSelfOutput` (post-norm), `RobertaIntermediate` (`Linear → exact
//! GELU`), and `RobertaOutput` (post-norm). The deltas from the SigLIP2
//! `shared` template (which is pre-norm with `Mask::None`) are RoBERTa-specific:
//! the `LayerNorm` moves **after** the residual (post-norm), the FFN activation
//! is **exact** [`gelu`](crate::lm::nn::activations::gelu) (`hidden_act =
//! "gelu"`, not the `tanh` approximation), and the attention takes an
//! **additive padding mask** (RoBERTa masks pad keys, unlike SigLIP2's
//! sticky-EOS full attention).
//!
//! **Audio tower (HTSAT Swin).** Mirrors HF `transformers`' `ClapAudioModel`
//! Swin path (`window_partition` / `window_reverse`, `ClapAudioSelfAttention`,
//! `ClapAudioLayer`, `ClapAudioPatchMerging` in `modeling_clap.py`): the
//! window-partition reshape and its inverse, the shifted-window cyclic roll +
//! the SW-MSA attention mask, the learned **relative-position-bias** table
//! gathered by a recomputed `relative_position_index` and added pre-softmax,
//! the window self-attention, the Swin MLP, and the `2×2` patch-merging
//! downsample. Window SDPA reuses the same
//! [`scaled_dot_product_attention`] + additive [`Mask`] core as the text tower;
//! the only genuinely new code is the windowing / relative-bias / patch-merge
//! bookkeeping ([`WindowAttention`], [`SwinBlock`], [`PatchMerging`], and the
//! [`window_partition`] / [`window_reverse`] / [`relative_position_index`]
//! helpers).
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

// ════════════════════════════ ClapProjectionLayer ══════════════════════════

/// The CLAP projection head (HF `ClapProjectionLayer`):
/// `linear2(relu(linear1(x)))`, with a biased `Linear(hidden → projection_dim)`,
/// then a **ReLU**, then `Linear(projection_dim → projection_dim)` (CLAP uses
/// ReLU, not the towers' GELU — `projection_hidden_act = "relu"`).
///
/// Both towers project through this same head (`text_projection` over the
/// RoBERTa CLS feature, `audio_projection` over the HTSAT pooled feature), so it
/// lives here rather than on either tower; the `(B, hidden) → (B, projection_dim)`
/// shape is identical for both (`hidden = 768`, `projection_dim = 512`).
#[cfg(feature = "clap")]
pub(crate) struct ClapProjectionLayer {
  linear1: QuantLinear,
  linear2: QuantLinear,
}

#[cfg(feature = "clap")]
impl ClapProjectionLayer {
  /// Build from `{prefix}.linear1.*` + `{prefix}.linear2.*`, pinning `linear1` to
  /// `(projection_dim, hidden)` and `linear2` to `(projection_dim, projection_dim)`
  /// (both biased). Each Linear auto-picks the dense or quantized variant by its
  /// `.scales` sibling.
  pub(crate) fn from_weights(
    prefix: &str,
    weights: &mut HashMap<String, Array>,
    hidden: i32,
    projection_dim: i32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let linear1 = QuantLinear::from_weights(
      weights,
      &format!("{prefix}.linear1"),
      projection_dim,
      hidden,
      true,
      quant,
    )?;
    let linear2 = QuantLinear::from_weights(
      weights,
      &format!("{prefix}.linear2"),
      projection_dim,
      projection_dim,
      true,
      quant,
    )?;
    Ok(Self { linear1, linear2 })
  }

  /// `linear2(relu(linear1(x)))`.
  pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
    let h = self.linear1.forward(x)?;
    let h = relu(&h)?;
    self.linear2.forward(&h)
  }

  /// `true` if both projection layers loaded the quantized variant (test-only).
  #[cfg(test)]
  pub(crate) fn all_quantized(&self) -> bool {
    self.linear1.is_quantized() && self.linear2.is_quantized()
  }
}

/// ReLU (`max(x, 0)`), the CLAP projection activation
/// (`projection_hidden_act = "relu"`). Built with a dtype-matched rank-0 `0`
/// constant so an f16/bf16 activation is not promoted to f32.
#[cfg(feature = "clap")]
pub(crate) fn relu(x: &Array) -> Result<Array> {
  let zero = cast_like(&Array::full::<f32>(&[0i32; 0], 0.0)?, x)?;
  ops::arithmetic::maximum(x, &zero)
}

/// Cast `a` to `like`'s dtype (a no-op when they already match) — the uniform
/// stand-in for MLX weak-scalar / `astype(x.dtype)` semantics, so a tensor built
/// in f32 (a scalar floor, a gathered position/bias row) that meets an f16/bf16
/// activation is cast back rather than promoting the activation to f32.
#[cfg(feature = "clap")]
pub(crate) fn cast_like(a: &Array, like: &Array) -> Result<Array> {
  ops::misc::astype(a, like.dtype()?)
}

// ═══════════════════════ sanitize-time key + weight helpers ════════════════

/// Fallibly build the concatenation of `parts` into a freshly-reserved
/// [`String`], turning an allocator failure into a typed [`Error::AllocFailure`]
/// instead of the abort `String::push_str` / `format!` would raise on growth —
/// the SigLIP2 `fallible_concat` precedent. Each `sanitize` / split key rewrite
/// is sized by a **checkpoint-controlled** key, so a hostile checkpoint with
/// enormous keys surfaces a recoverable error instead of aborting.
#[cfg(feature = "clap")]
pub(crate) fn fallible_concat(context: &'static str, parts: &[&str]) -> Result<String> {
  let total = parts
    .iter()
    .fold(0usize, |acc, p| acc.saturating_add(p.len()));
  let mut out = String::new();
  out.try_reserve_exact(total).map_err(|e| {
    Error::AllocFailure(crate::error::AllocFailurePayload::new(
      "clap key rewrite",
      context,
      total as u64,
      e,
    ))
  })?;
  for p in parts {
    out.push_str(p);
  }
  Ok(out)
}

/// Fallibly clone `s` into an owned [`String`] (the single-part
/// [`fallible_concat`]), surfacing a typed [`Error::AllocFailure`] on allocator
/// failure rather than the abort `str::to_string` would raise.
#[cfg(feature = "clap")]
pub(crate) fn fallible_clone_str(context: &'static str, s: &str) -> Result<String> {
  fallible_concat(context, &[s])
}

/// Transpose the patch-embed Conv2d weight to mlxrs's channels-last NHWC layout
/// (the SigLIP2 `reshape_patch_weight` precedent), keyed only on the tensor's
/// shape so `sanitize` needs no config.
///
/// mlxrs [`conv2d`](crate::ops::conv::conv2d) is NHWC (weight
/// `(C_out, KH, KW, C_in)`) while HF's `Conv2d` weight is the channels-first
/// `(C_out, C_in, KH, KW)`. For the CLAP patch embed (`C_in = 1`, `KH = KW = 4`)
/// the two layouts are `(96, 1, 4, 4)` (HF NCHW) vs `(96, 4, 4, 1)` (NHWC), so the
/// channel axis is the smallest of the trailing three. This:
///
/// - rank-4 with the channel axis **last** (`shape[3] <= shape[1]` and
///   `shape[3] <= shape[2]`) is already NHWC → left unchanged (so `sanitize` is
///   idempotent and an MLX-native checkpoint passes through);
/// - rank-4 with the channel axis at **index 1** (the HF NCHW signature) →
///   transposed `[0, 2, 3, 1]` to NHWC;
/// - any other rank is a typed [`Error::RankMismatch`].
///
/// The audio tower's `PatchEmbed::from_weights` then shape-pins the result to the
/// exact `(hidden, patch, patch, in_channels)` NHWC extents, so a wrong layout
/// that slips through is caught at load, not at first forward.
#[cfg(feature = "clap")]
pub(crate) fn reshape_patch_weight(raw: &Array, key: &str) -> Result<Array> {
  let shape = raw.shape();
  if shape.len() != 4 {
    let rank = shape.len() as u32;
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      key,
      Error::RankMismatch(RankMismatchPayload::new(
        "clap patch-embed conv weight must be rank-4 (C_out, C_in, KH, KW) NCHW or (C_out, KH, KW, C_in) NHWC",
        rank,
        shape,
      )),
    )));
  }
  // The channel axis is the smallest of the trailing three (for CLAP it is `1`).
  // If it is already last (axis 3) the weight is NHWC — leave it. If it is at
  // axis 1 the weight is the HF NCHW form — transpose `[0, 2, 3, 1]`.
  let (c1, c2, c3) = (shape[1], shape[2], shape[3]);
  if c3 <= c1 && c3 <= c2 {
    // Already channels-last NHWC.
    raw.try_clone()
  } else {
    // HF channels-first NCHW `(C_out, C_in, KH, KW)` → NHWC `(C_out, KH, KW, C_in)`.
    ops::shape::transpose_axes(raw, &[0, 2, 3, 1])
  }
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

// ════════════════════════ HTSAT Swin-Transformer blocks ════════════════════
//
// The HTSAT audio encoder is a Swin-Transformer V1 over the mel "image". These
// are the genuinely new blocks (the rest of the audio tower — patch-embed
// conv2d + LayerNorm, the mean-pool head — reuses existing ops). They mirror HF
// `transformers`' `modeling_clap.py` (`src/transformers/models/clap/`): the
// free functions `window_partition` (~L95-107) / `window_reverse` (~L110-125),
// `ClapAudioSelfAttention` (~L327-433, the relative-position-bias + window
// SDPA), `ClapAudioLayer` (~L592-740, the shift roll + SW-MSA mask + the Swin
// residual block), and `ClapAudioPatchMerging` (~L743-778, the 2×2 downsample).
//
// These blocks are the audio-tower building primitives: the HTSAT tower
// ([`super::super::audio::HtsatAudioTower`]) composes `SwinBlock` stacks +
// `PatchMerging` downsamples into the four stages (`window_partition` → shift →
// window SDPA → `window_reverse` per block, with `PatchMerging` between stages),
// and the oracle tests pin each primitive.
#[cfg(feature = "clap")]
mod swin {
  use super::*;

  /// Reshape a feature map `(B, H, W, C)` into non-overlapping `window × window`
  /// windows `(num_windows · B, window², C)` — HF `window_partition`
  /// (`modeling_clap.py` ~L95-107):
  ///
  /// ```python
  /// hidden_states.view(B, H // win, win, W // win, win, C)
  ///              .permute(0, 1, 3, 2, 4, 5).contiguous()
  ///              .view(-1, win, win, C)            # then .view(-1, win*win, C)
  /// ```
  ///
  /// Here the `(-1, win, win, C)` and the subsequent `.view(-1, win*win, C)` the
  /// caller does are fused into one `(num_windows · B, window², C)` reshape. `H`
  /// and `W` must be exact multiples of `window` (the [`SwinBlock`] pads the map
  /// up to a multiple first); a non-multiple is a typed shape error from the
  /// reshape op, not a panic.
  pub(crate) fn window_partition(x: &Array, window: i32) -> Result<Array> {
    require_window(window)?;
    let shape = x.shape();
    let b = dim_i32(&shape, 0, "clap Swin window_partition: batch")?;
    let h = dim_i32(&shape, 1, "clap Swin window_partition: height")?;
    let w = dim_i32(&shape, 2, "clap Swin window_partition: width")?;
    let c = dim_i32(&shape, 3, "clap Swin window_partition: channels")?;
    let (hb, wb) = window_block_counts(h, w, window)?;
    // (B, H/win, win, W/win, win, C)
    let grid = ops::shape::reshape(x, &[b, hb, window, wb, window, c])?;
    // permute(0, 1, 3, 2, 4, 5) → (B, H/win, W/win, win, win, C)
    let grid = ops::shape::transpose_axes(&grid, &[0, 1, 3, 2, 4, 5])?;
    // → (num_windows · B, window², C)
    let num_windows_b = checked_mul(
      "clap Swin window_partition: num_windows * B",
      "B * H/win",
      checked_mul("clap Swin window_partition: B * H/win", "B", b, "H/win", hb)?,
      "W/win",
      wb,
    )?;
    let window_area = window_area(window)?;
    ops::shape::reshape(&grid, &[num_windows_b, window_area, c])
  }

  /// Invert [`window_partition`]: reshape windows `(num_windows · B, window², C)`
  /// back to the feature map `(B, H, W, C)` — HF `window_reverse`
  /// (`modeling_clap.py` ~L110-125):
  ///
  /// ```python
  /// windows.view(-1, H // win, W // win, win, win, C)
  ///        .permute(0, 1, 3, 2, 4, 5).contiguous()
  ///        .view(-1, H, W, C)
  /// ```
  ///
  /// `height` / `width` are the (padded) map dims the windows came from; the
  /// leading window axis is `(H/win) · (W/win)` per batch element.
  pub(crate) fn window_reverse(
    windows: &Array,
    window: i32,
    height: i32,
    width: i32,
  ) -> Result<Array> {
    require_window(window)?;
    let shape = windows.shape();
    let leading = dim_i32(&shape, 0, "clap Swin window_reverse: num_windows·B")?;
    let c = dim_i32(&shape, 2, "clap Swin window_reverse: channels")?;
    let (hb, wb) = window_block_counts(height, width, window)?;
    // The leading axis is num_windows · B; recover B = leading / (H/win · W/win)
    // explicitly (mlxrs `reshape` requires concrete dims — no `-1` inference).
    let num_windows = checked_mul(
      "clap Swin window_reverse: num_windows (H/win · W/win)",
      "H/win",
      hb,
      "W/win",
      wb,
    )?;
    crate::model_validation::require_divisible(
      "clap Swin window_reverse: num_windows·B",
      leading,
      "clap Swin window_reverse: num_windows",
      num_windows,
    )?;
    let batch = leading / num_windows;
    // (B, H/win, W/win, win, win, C)
    let grid = ops::shape::reshape(windows, &[batch, hb, wb, window, window, c])?;
    // permute(0, 1, 3, 2, 4, 5) → (B, H/win, win, W/win, win, C)
    let grid = ops::shape::transpose_axes(&grid, &[0, 1, 3, 2, 4, 5])?;
    // → (B, H, W, C)
    ops::shape::reshape(&grid, &[batch, height, width, c])
  }

  /// Recompute the Swin `relative_position_index` `(window², window²)` buffer
  /// deterministically from `window` — HF `ClapAudioSelfAttention.
  /// create_relative_position_index` (`modeling_clap.py`):
  ///
  /// ```python
  /// coords = stack(meshgrid([arange(win), arange(win)], indexing="ij"))  # (2,win,win)
  /// coords_flatten = flatten(coords, 1)                                   # (2, win²)
  /// relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
  /// relative_coords = relative_coords.permute(1, 2, 0)                    # (win², win², 2)
  /// relative_coords[:, :, 0] += win - 1
  /// relative_coords[:, :, 1] += win - 1
  /// relative_coords[:, :, 0] *= 2 * win - 1
  /// relative_position_index = relative_coords.sum(-1)                     # (win², win²)
  /// ```
  ///
  /// This is a **non-parameter buffer** (HF `register_buffer`); the port
  /// recomputes it here and drops it in `sanitize` rather than loading it. Each
  /// entry indexes the `((2·window − 1)²)`-row bias table. The values are bounded
  /// by `(2·window − 1)² − 1` so the gather is always in range; `window` is small
  /// (`8`), so this builds a flat `Vec<i32>` of `window⁴` entries (the same
  /// closed form the oracle test pins for `window = 2`).
  pub(crate) fn relative_position_index(window: i32) -> Result<Array> {
    require_window(window)?;
    let area = window_area(window)? as usize;
    let span = 2 * window - 1; // table side per axis; (2·window−1)² rows total.
    // Token i sits at (row = i / window, col = i % window) in the window grid.
    // For the pair (i, j): relative row = row_i − row_j, relative col = col_i −
    // col_j; shift both by (window − 1) into [0, 2·window − 2], fold the row by
    // ×(2·window − 1), and sum. (The HF arithmetic, flattened.)
    let win = window as i64;
    let span64 = i64::from(span);
    let mut index = Vec::<i32>::with_capacity(area * area);
    for i in 0..area as i64 {
      let (row_i, col_i) = (i / win, i % win);
      for j in 0..area as i64 {
        let (row_j, col_j) = (j / win, j % win);
        let rel_row = (row_i - row_j) + (win - 1);
        let rel_col = (col_i - col_j) + (win - 1);
        let value = rel_row * span64 + rel_col;
        index.push(value as i32);
      }
    }
    // Stored flat `(window² · window²,)` so the table gather is one `take`; the
    // bias builder reshapes the gathered rows back to `(window², window², heads)`.
    Array::from_slice::<i32>(&index, &(area * area,))
  }

  /// The learned Swin **relative-position-bias** — HF `ClapAudioSelfAttention`'s
  /// `relative_position_bias_table` (`modeling_clap.py` ~L327-433) gathered by the
  /// recomputed [`relative_position_index`]:
  ///
  /// ```python
  /// relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)]
  /// relative_position_bias = relative_position_bias.view(win², win², -1)
  /// relative_position_bias = relative_position_bias.permute(2, 0, 1)      # (heads, win², win²)
  /// ```
  ///
  /// Built once at construction from the `((2·window − 1)², num_heads)` table and
  /// the flat `(window⁴,)` index; the cached `(num_heads, window², window²)`
  /// result is broadcast-added (with a leading singleton batch/window axis) to the
  /// window attention logits pre-softmax.
  struct RelativePositionBias {
    /// `(num_heads, window², window²)` — added to the window SDPA logits.
    bias: Array,
  }

  impl RelativePositionBias {
    /// Gather the `(num_heads, window², window²)` bias from the
    /// `((2·window − 1)², num_heads)` `table` via the recomputed index.
    fn new(table: &Array, window: i32, num_heads: i32) -> Result<Self> {
      let area = window_area(window)?;
      let index = relative_position_index(window)?; // (window⁴,) i32
      // Gather table ROWS by the flat index (axis 0) → (window⁴, num_heads).
      // (`take` flat-indexes; `take_axis(.., 0)` selects whole rows, the
      // `table[index]` PyTorch row-gather HF does.)
      let gathered = ops::indexing::take_axis(table, &index, 0)?;
      // → (window², window², num_heads)
      let reshaped = ops::shape::reshape(&gathered, &[area, area, num_heads])?;
      // permute(2, 0, 1) → (num_heads, window², window²)
      let bias = ops::shape::transpose_axes(&reshaped, &[2, 0, 1])?;
      Ok(Self { bias })
    }

    /// The cached `(num_heads, window², window²)` bias, broadcast-ready (a leading
    /// `(1, ...)` axis is added so it sums over the `(num_windows · B)` SDPA batch
    /// axis), cast to the activation `dtype`. HF: `+ relative_position_bias.
    /// unsqueeze(0)`.
    fn additive(&self, dtype: Dtype) -> Result<Array> {
      let expanded = ops::shape::expand_dims_axes(&self.bias, &[0])?; // (1, heads, win², win²)
      ops::misc::astype(&expanded, dtype)
    }
  }

  /// The SW-MSA attention mask for a shifted Swin block — HF
  /// `ClapAudioLayer.get_attn_mask` (`modeling_clap.py` ~L640-665):
  ///
  /// ```python
  /// img_mask = zeros((1, H, W, 1))
  /// for h_slice in (0:-win, -win:-shift, -shift:None):
  ///     for w_slice in (0:-win, -win:-shift, -shift:None):
  ///         img_mask[:, h_slice, w_slice, :] = count; count += 1
  /// mask_windows = window_partition(img_mask, win).view(-1, win*win)
  /// attn_mask = mask_windows[:, None, :] − mask_windows[:, :, None]      # wrong axis order in HF;
  /// attn_mask = mask_windows.unsqueeze(1) − mask_windows.unsqueeze(2)    # actual HF
  /// attn_mask = attn_mask.masked_fill(attn_mask != 0, -100).masked_fill(attn_mask == 0, 0)
  /// ```
  ///
  /// The roll wraps three image regions together per axis (the `(0, win, shift)`
  /// boundaries); tokens that landed in the SAME source region share a label and
  /// may attend (`0`), tokens from different regions are blocked (`-100`). The
  /// result is `(num_windows, window², window²)`, broadcast (a `(1, num_windows,
  /// 1, window², window²)` reshape sums over the SDPA head axis and the
  /// per-batch-row window stack) onto the logits alongside the relative bias.
  ///
  /// Built directly from the region-label closed form (the same `img_mask` →
  /// per-window labels → `label_a == label_b ? 0 : -100` the HF code computes via
  /// `window_partition` + the subtraction + the two `masked_fill`s), so a shifted
  /// block needs no img-mask tensor round-trip; the oracle test pins it against
  /// the HF construction for a small grid.
  pub(crate) fn shifted_window_mask(
    height: i32,
    width: i32,
    window: i32,
    shift: i32,
  ) -> Result<Array> {
    require_window(window)?;
    if shift <= 0 || shift >= window {
      // Only the odd (shifted) blocks build this mask, with shift = window/2 in
      // (0, window). A non-shifted block must not call here.
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clap Swin shifted_window_mask: shift",
        "must be in (0, window) — only shifted blocks build an SW-MSA mask",
        smol_str::format_smolstr!("{shift}"),
      )));
    }
    let (hb, wb) = window_block_counts(height, width, window)?;
    let h = height as usize;
    let w = width as usize;
    let win = window as usize;
    // Region label per (row, col): the cumulative `count` HF assigns over the
    // (0:-win, -win:-shift, -shift:None) slice product. `region_id` maps a
    // coordinate to its [0, 9) HF label via the per-axis band index (0/1/2).
    let band = |pos: usize, extent: usize| -> usize {
      // HF slices: [0, extent-win) → band 0; [extent-win, extent-shift) → band 1;
      // [extent-shift, extent) → band 2.
      if pos < extent - win {
        0
      } else if pos < extent - shift as usize {
        1
      } else {
        2
      }
    };
    let mut img_label = vec![0i32; h * w];
    for (r, row) in img_label.chunks_exact_mut(w).enumerate() {
      let hband = band(r, h);
      for (c, cell) in row.iter_mut().enumerate() {
        let wband = band(c, w);
        *cell = (hband * 3 + wband) as i32; // count = h_idx*3 + w_idx (row-major)
      }
    }
    // Partition the label map into windows exactly as `window_partition` does
    // (the (B=1, H, W, 1) → (num_windows, win²) gather), then build the
    // (num_windows, win², win²) mask: 0 where two tokens share a region label,
    // -100 otherwise. We walk the window grid + intra-window positions directly.
    let num_windows = (hb as usize) * (wb as usize);
    let area = win * win;
    let mut mask = vec![0f32; num_windows * area * area];
    for wb_r in 0..hb as usize {
      for wb_c in 0..wb as usize {
        let win_idx = wb_r * (wb as usize) + wb_c;
        // Collect this window's per-token labels (row-major within the window).
        let mut labels = [0i32; 64]; // window ≤ 8 ⇒ area ≤ 64; HTSAT window = 8.
        let labels = &mut labels[..area];
        for (p, label) in labels.iter_mut().enumerate() {
          let (pr, pc) = (p / win, p % win);
          let gr = wb_r * win + pr;
          let gc = wb_c * win + pc;
          *label = img_label[gr * w + gc];
        }
        let base = win_idx * area * area;
        for a in 0..area {
          for bdx in 0..area {
            // HF: attn_mask = labels.unsqueeze(1) − labels.unsqueeze(2) ⇒
            // entry [a][b] from labels[b] − labels[a]; masked_fill !=0 → −100.
            mask[base + a * area + bdx] = if labels[a] == labels[bdx] {
              0.0
            } else {
              -100.0
            };
          }
        }
      }
    }
    let area_i = window_area(window)?;
    let mask = Array::from_slice::<f32>(&mask, &(num_windows, area_i as usize, area_i as usize))?;
    // (num_windows, win², win²) → (1, num_windows, 1, win², win²): a leading
    // batch-row axis and a head axis so it broadcast-adds onto the per-window
    // SDPA logits `(num_windows·B, heads, win², win²)` after that tensor is
    // reshaped to `(B, num_windows, heads, win², win²)`.
    let nw_i32 = checked_mul(
      "clap Swin shifted_window_mask: num_windows (H/win · W/win)",
      "H/win",
      hb,
      "W/win",
      wb,
    )?;
    ops::shape::reshape(&mask, &[1, nw_i32, 1, area_i, area_i])
  }

  /// HTSAT Swin **window self-attention with relative-position bias** — HF
  /// `ClapAudioSelfAttention` (`modeling_clap.py` ~L327-433). Standard SDPA over
  /// the `window²` tokens of each window with `qkv_bias = True` biased `Linear`s,
  /// plus the learned relative-position bias (and, for shifted blocks, the SW-MSA
  /// mask) added to the logits pre-softmax via the additive [`Mask`].
  pub(crate) struct WindowAttention {
    query: QuantLinear,
    key: QuantLinear,
    value: QuantLinear,
    /// `output.dense` — the post-attention output projection (`ClapAudioSelfOutput`).
    out: QuantLinear,
    num_heads: i32,
    head_dim: i32,
    /// `head_dim**-0.5`, the SDPA scale.
    scale: f32,
    /// The learned relative-position bias, gathered once at construction.
    relative_bias: RelativePositionBias,
  }

  impl WindowAttention {
    /// Build from `{prefix}.self.{query,key,value}` (biased, `qkv_bias = True`),
    /// `{prefix}.output.dense` (biased), and the
    /// `{prefix}.self.relative_position_bias_table`
    /// `((2·window − 1)², num_heads)` parameter. `dim` is the stage channel width;
    /// `num_heads` the stage head count.
    pub(crate) fn from_weights(
      weights: &mut HashMap<String, Array>,
      prefix: &str,
      dim: i32,
      num_heads: i32,
      window: i32,
      quant: Option<&PerLayerQuantization>,
    ) -> Result<Self> {
      crate::model_validation::require_positive("clap Swin WindowAttention: num_heads", num_heads)?;
      crate::model_validation::require_divisible(
        "clap Swin WindowAttention: dim",
        dim,
        "clap Swin WindowAttention: num_heads",
        num_heads,
      )?;
      require_window(window)?;
      let head_dim = dim / num_heads;
      let proj = |weights: &mut HashMap<String, Array>, name: &str| {
        QuantLinear::from_weights(weights, name, dim, dim, true, quant)
      };
      let query = proj(weights, &format!("{prefix}.self.query"))?;
      let key = proj(weights, &format!("{prefix}.self.key"))?;
      let value = proj(weights, &format!("{prefix}.self.value"))?;
      let out = proj(weights, &format!("{prefix}.output.dense"))?;
      // The relative-position-bias table is a dense fp32 parameter (never
      // quantized); shape-pin it to `((2·window − 1)², num_heads)`.
      let span = 2 * window - 1;
      let table_rows = checked_mul(
        "clap Swin: relative_position_bias_table rows ((2·window-1)²)",
        "2·window-1",
        span,
        "2·window-1",
        span,
      )?;
      let table = take_shaped(
        weights,
        &format!("{prefix}.self.relative_position_bias_table"),
        "clap Swin relative_position_bias_table ((2·window-1)², num_heads)",
        &[table_rows, num_heads],
      )?;
      let relative_bias = RelativePositionBias::new(&table, window, num_heads)?;
      Ok(Self {
        query,
        key,
        value,
        out,
        num_heads,
        head_dim,
        scale: (head_dim as f32).powf(-0.5),
        relative_bias,
      })
    }

    /// `(num_windows · B, window², C) → (num_windows · B, window², C)` window
    /// attention. `shift_mask` is the `(1, num_windows, 1, window², window²)`
    /// SW-MSA mask for a shifted block (`None` for a non-shifted block). The
    /// relative-position bias is always added; the shift mask is summed in when
    /// present (HF: both are added to the scores pre-softmax).
    pub(crate) fn forward(&self, x: &Array, shift_mask: Option<&Array>) -> Result<Array> {
      let shape = x.shape();
      let nw_b = dim_i32(&shape, 0, "clap Swin WindowAttention: num_windows·B")?;
      let tokens = dim_i32(&shape, 1, "clap Swin WindowAttention: window² tokens")?;

      let q = self.query.forward(x)?;
      let k = self.key.forward(x)?;
      let v = self.value.forward(x)?;

      let q = self.split_heads(&q, nw_b, tokens)?;
      let k = self.split_heads(&k, nw_b, tokens)?;
      let v = self.split_heads(&v, nw_b, tokens)?;

      // Additive mask = relative-position bias (+ SW-MSA shift mask), cast to the
      // activation dtype so an f16/bf16 checkpoint stays in its dtype.
      let bias = self.relative_bias.additive(q.dtype()?)?; // (1, heads, win², win²)
      let attn = match shift_mask {
        None => scaled_dot_product_attention(&q, &k, &v, self.scale, Mask::Array(&bias))?,
        Some(shift) => {
          // HF folds the SDPA batch as (B, num_windows) to add the per-window
          // shift mask, then folds back. The bias broadcasts over both. We sum
          // bias + shift first (broadcasting (1,heads,win²,win²) against
          // (1,num_windows,1,win²,win²) → (1,num_windows,heads,win²,win²)), fold
          // q/k/v to (B, num_windows, heads, win², win²)'s flattened batch, and
          // feed the combined mask broadcast over the SDPA batch.
          self.forward_shifted(&q, &k, &v, &bias, shift, nw_b)?
        }
      };
      let attn = ops::shape::transpose_axes(&attn, &[0, 2, 1, 3])?; // (nw·B, tokens, heads, hd)
      let embed_dim = checked_mul(
        "clap Swin WindowAttention: num_heads * head_dim",
        "num_heads",
        self.num_heads,
        "head_dim",
        self.head_dim,
      )?;
      let attn = ops::shape::reshape(&attn, &[nw_b, tokens, embed_dim])?;
      self.out.forward(&attn)
    }

    /// The shifted-block attention: combine the relative bias and the per-window
    /// SW-MSA `shift_mask`, then run SDPA with the combined additive mask
    /// broadcast over the `(num_windows · B)` window stack.
    ///
    /// `q`/`k`/`v` are `(num_windows · B, heads, window², head_dim)`. The shift
    /// mask is `(1, num_windows, 1, window², window²)`; the bias is
    /// `(1, heads, window², window²)`. Their sum is
    /// `(1, num_windows, heads, window², window²)`; we reshape it to
    /// `(num_windows, heads, window², window²)` — which broadcasts against the
    /// SDPA batch axis only when `B == 1`, and otherwise tiles per batch element.
    /// HTSAT runs the unfused single-clip path (`B == 1`), so the flat
    /// `(num_windows, ...)` mask aligns with the `(num_windows · B = num_windows)`
    /// SDPA batch directly; a `B > 1` call broadcasts the same per-window mask
    /// across every clip (HF builds one img-mask for all batch rows identically).
    fn forward_shifted(
      &self,
      q: &Array,
      k: &Array,
      v: &Array,
      bias: &Array,
      shift_mask: &Array,
      nw_b: i32,
    ) -> Result<Array> {
      let num_windows = dim_i32(&shift_mask.shape(), 1, "clap Swin shift mask: num_windows")?;
      // bias (1, heads, win², win²) + shift (1, num_windows, 1, win², win²)
      //   → (1, num_windows, heads, win², win²). The SW-MSA mask is built in f32;
      // cast it to the (already activation-dtype) bias before the add so the
      // combined mask stays in the activation dtype — an f16/bf16 SDPA rejects an
      // f32 mask (it must promote to the output dtype).
      let shift_mask = ops::misc::astype(shift_mask, bias.dtype()?)?;
      let bias5 = ops::shape::expand_dims_axes(bias, &[1])?; // (1, 1, heads, win², win²)
      let combined = bias5.add(&shift_mask)?; // (1, num_windows, heads, win², win²)
      // Drop the leading singleton → (num_windows, heads, win², win²); broadcast
      // to the SDPA batch (num_windows·B) by tiling per batch element when B > 1.
      let combined = ops::shape::squeeze_axes(&combined, &[0])?;
      let combined = self.broadcast_mask_to_batch(&combined, num_windows, nw_b)?;
      scaled_dot_product_attention(q, k, v, self.scale, Mask::Array(&combined))
    }

    /// Tile a `(num_windows, heads, win², win²)` mask up to the SDPA batch
    /// `(num_windows · B, heads, win², win²)` — a no-op view when `B == 1`
    /// (`num_windows · B == num_windows`), else the same per-window mask repeated
    /// for each of the `B` clips (HF reuses one img-mask across the batch).
    fn broadcast_mask_to_batch(&self, mask: &Array, num_windows: i32, nw_b: i32) -> Result<Array> {
      if nw_b == num_windows {
        return mask.try_clone();
      }
      let shape = mask.shape();
      let heads = dim_i32(&shape, 1, "clap Swin mask: heads")?;
      let t0 = dim_i32(&shape, 2, "clap Swin mask: win² (rows)")?;
      let t1 = dim_i32(&shape, 3, "clap Swin mask: win² (cols)")?;
      // (num_windows, heads, t, t) → (1, num_windows, heads, t, t) →
      //   broadcast (B, num_windows, ...) → flatten to (B·num_windows, ...).
      crate::model_validation::require_divisible(
        "clap Swin mask: num_windows·B",
        nw_b,
        "clap Swin mask: num_windows",
        num_windows,
      )?;
      let batch = nw_b / num_windows;
      let expanded = ops::shape::expand_dims_axes(mask, &[0])?; // (1, nw, heads, t, t)
      let tiled = ops::shape::broadcast_to(&expanded, &[batch, num_windows, heads, t0, t1])?;
      ops::shape::reshape(&tiled, &[nw_b, heads, t0, t1])
    }

    /// `(nw·B, tokens, C) → (nw·B, heads, tokens, head_dim)`.
    fn split_heads(&self, x: &Array, nw_b: i32, tokens: i32) -> Result<Array> {
      let reshaped = ops::shape::reshape(x, &[nw_b, tokens, self.num_heads, self.head_dim])?;
      ops::shape::transpose_axes(&reshaped, &[0, 2, 1, 3])
    }

    /// `true` if every projection loaded the quantized variant (test-only).
    #[cfg(test)]
    pub(crate) fn all_quantized(&self) -> bool {
      self.query.is_quantized()
        && self.key.is_quantized()
        && self.value.is_quantized()
        && self.out.is_quantized()
    }
  }

  /// The HTSAT Swin MLP — `ClapAudioIntermediate` (`Linear(dim → mlp_ratio·dim)` +
  /// **exact** GELU) + `ClapAudioOutput` (`Linear(mlp_ratio·dim → dim)`)
  /// (`modeling_clap.py` ~L551-590). `hidden_act = "gelu"` (the exact erf GELU,
  /// not the `tanh` approximation), `mlp_ratio = 4`.
  pub(crate) struct SwinMlp {
    /// `intermediate.dense` — the `dim → mlp_ratio·dim` expansion.
    intermediate: QuantLinear,
    /// `output.dense` — the `mlp_ratio·dim → dim` contraction.
    output: QuantLinear,
  }

  impl SwinMlp {
    /// Build from `{prefix}.intermediate.dense.*` + `{prefix}.output.dense.*`,
    /// pinning the two Linear shapes to `(hidden, dim)` / `(dim, hidden)` where
    /// `hidden = mlp_ratio · dim`.
    pub(crate) fn from_weights(
      weights: &mut HashMap<String, Array>,
      prefix: &str,
      dim: i32,
      hidden: i32,
      quant: Option<&PerLayerQuantization>,
    ) -> Result<Self> {
      let intermediate = QuantLinear::from_weights(
        weights,
        &format!("{prefix}.intermediate.dense"),
        hidden,
        dim,
        true,
        quant,
      )?;
      let output = QuantLinear::from_weights(
        weights,
        &format!("{prefix}.output.dense"),
        dim,
        hidden,
        true,
        quant,
      )?;
      Ok(Self {
        intermediate,
        output,
      })
    }

    /// `output.dense(gelu(intermediate.dense(x)))` — the un-residualized Swin MLP
    /// (the residual + the pre/post LayerNorms live in [`SwinBlock::forward`]).
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

  /// One HTSAT Swin-Transformer block — HF `ClapAudioLayer`
  /// (`modeling_clap.py` ~L592-740):
  ///
  /// ```text
  /// shortcut = x
  /// x = layernorm_before(x).view(B, H, W, C)
  /// x, pad = maybe_pad(x, H, W)                 # pad up to a window multiple
  /// x = roll(x, -shift) if shift else x         # cyclic shift
  /// windows = window_partition(x, win)          # (nw·B, win², C)
  /// a = window_attention(windows, attn_mask)    # SDPA + rel-bias (+ SW-MSA mask)
  /// x = window_reverse(a, win, Hp, Wp)
  /// x = roll(x, +shift) if shift else x         # reverse shift
  /// x = x[:, :H, :W, :] if padded               # depad
  /// x = shortcut + x.view(B, H·W, C)            # residual 1
  /// x = x + swin_mlp(layernorm_after(x))        # residual 2
  /// ```
  ///
  /// Even blocks use `shift = 0` (W-MSA), odd blocks `shift = window/2` (SW-MSA);
  /// the caller passes the resolved `shift`. The two `LayerNorm`s are
  /// `layernorm_before` / `layernorm_after`.
  ///
  /// The block's spatial resolution `(height, width)` is fixed at construction
  /// (each Swin stage has one resolution), so a shifted block precomputes its
  /// SW-MSA [`shifted_window_mask`] **once** here — mirroring the construction-time
  /// [`RelativePositionBias`] gather — rather than rebuilding it on every forward
  /// (the mask depends only on the padded resolution, window, and shift, all
  /// construction-fixed).
  pub(crate) struct SwinBlock {
    layernorm_before: LayerNorm,
    attention: WindowAttention,
    layernorm_after: LayerNorm,
    mlp: SwinMlp,
    window: i32,
    /// `0` for an even (W-MSA) block, `window/2` for an odd (SW-MSA) block.
    shift: i32,
    /// This block's (construction-fixed) input resolution `(height, width)`; the
    /// forward asserts its argument matches it (the cached SW-MSA mask is pinned
    /// to this resolution).
    height: i32,
    width: i32,
    /// The SW-MSA mask `(1, num_windows, 1, window², window²)`, precomputed once at
    /// construction for a shifted block (`shift > 0`); `None` for a W-MSA block.
    shift_mask: Option<Array>,
  }

  impl SwinBlock {
    /// Build the `i`-th block of a stage from `{prefix}` (the HF
    /// `...layers.{stage}.blocks.{i}` tree): `layernorm_before`, `attention.self.
    /// {query,key,value}` + `attention.self.relative_position_bias_table` +
    /// `attention.output.dense`, `layernorm_after`, `intermediate.dense`, and
    /// `output.dense`. `dim` is the stage channel width, `num_heads` the stage
    /// head count, `shift` the resolved cyclic shift (`0` or `window/2`),
    /// `hidden = mlp_ratio · dim`, `(height, width)` the stage's (fixed) input
    /// resolution — used to precompute the SW-MSA mask for a shifted block.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_weights(
      weights: &mut HashMap<String, Array>,
      prefix: &str,
      dim: i32,
      num_heads: i32,
      window: i32,
      shift: i32,
      height: i32,
      width: i32,
      hidden: i32,
      eps: f32,
      quant: Option<&PerLayerQuantization>,
    ) -> Result<Self> {
      require_window(window)?;
      if shift < 0 || shift >= window {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "clap Swin SwinBlock: shift",
          "must be in [0, window) (0 = W-MSA, window/2 = SW-MSA)",
          smol_str::format_smolstr!("{shift}"),
        )));
      }
      let layernorm_before =
        build_layer_norm(weights, &format!("{prefix}.layernorm_before"), dim, eps)?;
      let attention = WindowAttention::from_weights(
        weights,
        &format!("{prefix}.attention"),
        dim,
        num_heads,
        window,
        quant,
      )?;
      let layernorm_after =
        build_layer_norm(weights, &format!("{prefix}.layernorm_after"), dim, eps)?;
      let mlp = SwinMlp::from_weights(weights, prefix, dim, hidden, quant)?;
      // Precompute the SW-MSA mask once for a shifted block (the padded resolution,
      // window, and shift are all construction-fixed) — the `RelativePositionBias`
      // caching pattern. The forward pins its `(height, width)` argument to these.
      let shift_mask = if shift > 0 {
        let (height_pad, width_pad) = padded_resolution(height, width, window)?;
        Some(shifted_window_mask(height_pad, width_pad, window, shift)?)
      } else {
        None
      };
      Ok(Self {
        layernorm_before,
        attention,
        layernorm_after,
        mlp,
        window,
        shift,
        height,
        width,
        shift_mask,
      })
    }

    /// Run the block on `(B, L = H·W, C)` given the spatial `(height, width)` —
    /// the HF `ClapAudioLayer.forward`. Returns `(B, H·W, C)`. `(height, width)`
    /// must equal the block's construction resolution (the cached SW-MSA mask is
    /// pinned to it).
    pub(crate) fn forward(&self, x: &Array, height: i32, width: i32) -> Result<Array> {
      if height != self.height || width != self.width {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "clap Swin SwinBlock: forward (height, width)",
          "must equal the block's construction resolution (the cached SW-MSA mask is pinned to it)",
          smol_str::format_smolstr!("({height}, {width}) != ({}, {})", self.height, self.width),
        )));
      }
      let shape = x.shape();
      let b = dim_i32(&shape, 0, "clap Swin SwinBlock: batch")?;
      let c = dim_i32(&shape, 2, "clap Swin SwinBlock: channels")?;
      let shortcut = x;

      let h = self.layernorm_before.forward(x)?;
      let h = ops::shape::reshape(&h, &[b, height, width, c])?;
      // Pad up to a window multiple (HF maybe_pad; pad right/bottom only).
      let (h, pad_right, pad_bottom) = self.maybe_pad(&h, height, width)?;
      let height_pad = height + pad_bottom;
      let width_pad = width + pad_right;

      // Cyclic shift (SW-MSA): roll the map by -shift on the (H, W) axes.
      let shifted = if self.shift > 0 {
        ops::shape::roll_axes(&h, &[-self.shift, -self.shift], &[1, 2])?
      } else {
        h
      };

      // Partition → (nw·B, win², C), attend, reverse. The SW-MSA mask depends only
      // on the (construction-fixed) padded resolution, window, and shift, so it was
      // precomputed once at construction (the `RelativePositionBias` pattern, #365)
      // rather than rebuilt here each forward.
      let windows = window_partition(&shifted, self.window)?;
      let attn = self.attention.forward(&windows, self.shift_mask.as_ref())?;
      let attn = window_reverse(&attn, self.window, height_pad, width_pad)?;

      // Reverse cyclic shift.
      let attn = if self.shift > 0 {
        ops::shape::roll_axes(&attn, &[self.shift, self.shift], &[1, 2])?
      } else {
        attn
      };

      // Depad back to (B, H, W, C) if padded.
      let attn = if pad_right > 0 || pad_bottom > 0 {
        let starts = [0, 0, 0, 0];
        let stops = [b, height, width, c];
        let strides = [1, 1, 1, 1];
        ops::indexing::slice(&attn, &starts, &stops, &strides)?
      } else {
        attn
      };

      // Residual 1: shortcut + attention, with the map flattened back to (B, L, C).
      let area = checked_mul(
        "clap Swin SwinBlock: H * W",
        "height",
        height,
        "width",
        width,
      )?;
      let attn = ops::shape::reshape(&attn, &[b, area, c])?;
      let hidden = shortcut.add(&attn)?;

      // Residual 2: hidden + MLP(layernorm_after(hidden)).
      let normed = self.layernorm_after.forward(&hidden)?;
      let mlp = self.mlp.forward(&normed)?;
      hidden.add(&mlp)
    }

    /// HF `ClapAudioLayer.maybe_pad`: pad `(B, H, W, C)` on the right/bottom up to
    /// a window multiple. Returns the padded map + the `(pad_right, pad_bottom)`
    /// amounts.
    fn maybe_pad(&self, x: &Array, height: i32, width: i32) -> Result<(Array, i32, i32)> {
      let pad_right = (self.window - width % self.window) % self.window;
      let pad_bottom = (self.window - height % self.window) % self.window;
      if pad_right == 0 && pad_bottom == 0 {
        return Ok((x.try_clone()?, 0, 0));
      }
      // Pad the right of axis 2 (width) and the bottom of axis 1 (height) only —
      // HF `maybe_pad`'s `(0, 0, 0, pad_right, 0, pad_bottom)` (the trailing pad
      // pairs over the last→first axes leave batch/channel untouched).
      let pad_value = Array::full::<f32>(&[0i32; 0], 0.0)?;
      let padded = ops::shape::pad(
        x,
        &[1, 2],
        &[0, 0],
        &[pad_bottom, pad_right],
        &pad_value,
        c"constant",
      )?;
      Ok((padded, pad_right, pad_bottom))
    }

    /// `true` if every projection in the block loaded quantized (test-only).
    #[cfg(test)]
    pub(crate) fn all_quantized(&self) -> bool {
      self.attention.all_quantized() && self.mlp.all_quantized()
    }
  }

  /// HTSAT Swin **patch merging** — HF `ClapAudioPatchMerging`
  /// (`modeling_clap.py` ~L743-778). Concatenate each `2×2` neighborhood of tokens
  /// `(B, H, W, C) → (B, H/2, W/2, 4C)`, then `LayerNorm(4C)` + `Linear(4C → 2C,
  /// bias = False)`:
  ///
  /// ```python
  /// input_feature = cat([input_feature[:, r::2, c::2, :]
  ///                       for c in range(2) for r in range(2)], dim=-1)  # (B, H/2, W/2, 4C)
  /// input_feature = norm(input_feature)
  /// input_feature = reduction(input_feature)                             # (B, H/2, W/2, 2C)
  /// ```
  ///
  /// The HF comprehension order is `for c in range(2) for r in range(2)`, i.e. the
  /// concat order is `[(r0,c0), (r1,c0), (r0,c1), (r1,c1)]` — pinned exactly (a
  /// wrong neighborhood order silently corrupts the downsample).
  pub(crate) struct PatchMerging {
    norm: LayerNorm,
    /// `reduction` — the `4·dim → 2·dim`, bias-free reduction Linear.
    reduction: QuantLinear,
    dim: i32,
  }

  impl PatchMerging {
    /// Build from `{prefix}.norm` (`LayerNorm(4·dim)`) + `{prefix}.reduction`
    /// (`Linear(4·dim → 2·dim, bias = False)`). `dim` is the input (pre-merge)
    /// stage channel width.
    pub(crate) fn from_weights(
      weights: &mut HashMap<String, Array>,
      prefix: &str,
      dim: i32,
      eps: f32,
      quant: Option<&PerLayerQuantization>,
    ) -> Result<Self> {
      let four_dim = checked_mul("clap Swin PatchMerging: 4·dim", "4", 4, "dim", dim)?;
      let two_dim = checked_mul("clap Swin PatchMerging: 2·dim", "2", 2, "dim", dim)?;
      let norm = build_layer_norm(weights, &format!("{prefix}.norm"), four_dim, eps)?;
      let reduction = QuantLinear::from_weights(
        weights,
        &format!("{prefix}.reduction"),
        two_dim,
        four_dim,
        false,
        quant,
      )?;
      Ok(Self {
        norm,
        reduction,
        dim,
      })
    }

    /// `(B, L = H·W, C) → (B, (H/2)·(W/2), 2C)` patch merge, given the input
    /// `(height, width)`. `height` / `width` must be even (HTSAT's stage
    /// resolutions are; a non-even input is a typed shape error). HF operates on
    /// `(B, H, W, C)`; the tower reshapes to/from `(B, L, C)` around it.
    pub(crate) fn forward(&self, x: &Array, height: i32, width: i32) -> Result<Array> {
      let shape = x.shape();
      let b = dim_i32(&shape, 0, "clap Swin PatchMerging: batch")?;
      let c = dim_i32(&shape, 2, "clap Swin PatchMerging: channels")?;
      if height % 2 != 0 || width % 2 != 0 {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "clap Swin PatchMerging: (height, width)",
          "both must be even (the 2×2 merge halves each spatial axis)",
          smol_str::format_smolstr!("({height}, {width})"),
        )));
      }
      let map = ops::shape::reshape(x, &[b, height, width, c])?;
      // The four 2×2-strided sub-grids, in the HF comprehension order
      // `[(r0,c0), (r1,c0), (r0,c1), (r1,c1)]` (`for c in 2 for r in 2`). Each is
      // `(B, H/2, W/2, C)` via a stride-2 slice with the (row, col) phase offset.
      let sub = |row: i32, col: i32| -> Result<Array> {
        let starts = [0, row, col, 0];
        let stops = [b, height, width, c];
        let strides = [1, 2, 2, 1];
        ops::indexing::slice(&map, &starts, &stops, &strides)
      };
      let f00 = sub(0, 0)?;
      let f10 = sub(1, 0)?;
      let f01 = sub(0, 1)?;
      let f11 = sub(1, 1)?;
      // Concatenate on the channel axis → (B, H/2, W/2, 4C).
      let merged = ops::shape::concatenate(&[&f00, &f10, &f01, &f11], -1)?;
      let normed = self.norm.forward(&merged)?;
      let reduced = self.reduction.forward(&normed)?; // (B, H/2, W/2, 2C)
      // Flatten back to (B, (H/2)·(W/2), 2C).
      let h2 = height / 2;
      let w2 = width / 2;
      let new_len = checked_mul("clap Swin PatchMerging: (H/2)·(W/2)", "H/2", h2, "W/2", w2)?;
      let two_dim = checked_mul("clap Swin PatchMerging: 2·dim", "2", 2, "dim", self.dim)?;
      ops::shape::reshape(&reduced, &[b, new_len, two_dim])
    }

    /// `true` if the reduction loaded the quantized variant (test-only).
    #[cfg(test)]
    pub(crate) fn is_quantized(&self) -> bool {
      self.reduction.is_quantized()
    }
  }

  /// Reject a non-positive Swin `window` before it is used as a divisor / shape
  /// dimension.
  fn require_window(window: i32) -> Result<()> {
    crate::model_validation::require_positive("clap Swin: window_size", window)
  }

  /// `window²`, the per-window token count, with overflow checked.
  fn window_area(window: i32) -> Result<i32> {
    checked_mul(
      "clap Swin: window² (tokens per window)",
      "window",
      window,
      "window",
      window,
    )
  }

  /// The window-multiple **padded** resolution `(height_pad, width_pad)` HF
  /// `ClapAudioLayer.maybe_pad` produces (right/bottom pad only): each axis is
  /// rounded **up** to the next multiple of `window`. Kept as the single source of
  /// the pad arithmetic [`SwinBlock::maybe_pad`] applies at forward time, so the
  /// construction-time cached SW-MSA mask is built at the exact padded resolution
  /// the forward windows.
  fn padded_resolution(height: i32, width: i32, window: i32) -> Result<(i32, i32)> {
    require_window(window)?;
    crate::model_validation::require_positive("clap Swin padded_resolution: height", height)?;
    crate::model_validation::require_positive("clap Swin padded_resolution: width", width)?;
    let pad_bottom = (window - height % window) % window;
    let pad_right = (window - width % window) % window;
    let height_pad = height
      .checked_add(pad_bottom)
      .ok_or_else(|| pad_overflow("height", height, pad_bottom))?;
    let width_pad = width
      .checked_add(pad_right)
      .ok_or_else(|| pad_overflow("width", width, pad_right))?;
    Ok((height_pad, width_pad))
  }

  /// The typed overflow error `padded_resolution` raises if an axis + its pad
  /// exceeds `i32::MAX` (unreachable for the HTSAT resolutions; a soundness guard).
  fn pad_overflow(axis: &'static str, extent: i32, pad: i32) -> Error {
    Error::OutOfRange(OutOfRangePayload::new(
      "clap Swin padded_resolution: padded extent",
      "extent + pad must not overflow i32",
      smol_str::format_smolstr!("{axis}: {extent} + {pad}"),
    ))
  }

  /// The `(H/window, W/window)` window-block counts, erroring if `H` / `W` is
  /// not an exact multiple of `window` (the partition reshape requires it; the
  /// block pads the map up first).
  pub(crate) fn window_block_counts(height: i32, width: i32, window: i32) -> Result<(i32, i32)> {
    require_window(window)?;
    if height % window != 0 || width % window != 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clap Swin window_partition: (height, width)",
        "both must be exact multiples of window",
        smol_str::format_smolstr!("({height}, {width}) % {window}"),
      )));
    }
    Ok((height / window, width / window))
  }
}

// Re-exported for the audio tower ([`super::audio::HtsatAudioTower`], the
// `swin`-module consumer): it assembles `SwinBlock` stacks + `PatchMerging`
// downsamples into the four HTSAT stages.
#[cfg(feature = "clap")]
pub(crate) use swin::{PatchMerging, SwinBlock};

// The remaining window/relative-bias/mask primitives are consumed only by the
// oracle tests (the tower composes them transitively through `SwinBlock`), so
// they are re-exported under `cfg(test)` to keep the non-test build free of
// unused re-exports.
#[cfg(all(test, feature = "clap"))]
pub(crate) use swin::{
  WindowAttention, relative_position_index, shifted_window_mask, window_partition, window_reverse,
};

#[cfg(all(test, feature = "clap"))]
mod tests;
