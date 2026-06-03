//! Shared building blocks for the EmbeddingGemma backbone — the quantize-aware
//! [`Linear`] / [`Embedding`] layers, the private `nn.Linear` forward, a
//! dtype-matched scalar constant helper, and the weight fetch + shape-pinning
//! helpers (the same discipline as the merged SigLIP2 / Wav2Vec2 / LFM2 ports:
//! every consumed tensor's shape is checked before it is stored or fed to any
//! op, with a typed [`Error::ShapePairMismatch`] wrapped in [`Error::LayerKeyed`]).
//!
//! These mirror the SigLIP2 `shared` helpers, adapted to the Gemma3 backbone's
//! bias-free `nn.Linear` (every Gemma3 / Dense projection is `bias=False`).
//!
//! ## Quantize-aware layers
//!
//! [`Linear`] and [`Embedding`] each wrap the shared
//! [`crate::nn::MaybeQuantizedLinear`] / a quantized table so a model layer
//! loads either a dense or an 8-bit/4-bit quantized checkpoint through one
//! code path — the same adoption Whisper takes (see
//! [`crate::audio::stt::models::whisper`]). The builders auto-detect the
//! quantized variant from the `<prefix>.scales` sibling (mlx-embeddings'
//! `class_predicate`, which quantizes every `nn.Linear` / `nn.Embedding`), with
//! the per-layer `(group_size, bits, mode)` resolved from the parsed
//! [`crate::lm::quant::PerLayerQuantization`]. The dense path is byte-for-byte
//! the prior `(out, in)`-shape-pinned [`crate::nn::Linear`] / dense gather.

use std::collections::HashMap;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    Error, InvariantViolationPayload, LayerKeyedPayload, MissingKeyPayload, OutOfRangePayload,
    RankMismatchPayload, Result, ShapePairMismatchPayload,
  },
  lm::quant::PerLayerQuantization,
  nn::{MaybeQuantizedLinear, QuantizedLinear},
  ops,
};

/// An EmbeddingGemma linear projection `y = x @ Wᵀ` — quantize-aware, bias-free.
///
/// Mirrors `mlx.nn.Linear` (`weight` stored `(out, in)`, the forward transposes
/// it) for a dense checkpoint and `mlx.nn.QuantizedLinear` for a quantized one,
/// sharing one [`forward`](Self::forward) call site via [`MaybeQuantizedLinear`]
/// — the same adoption Whisper takes
/// ([`crate::audio::stt::models::whisper`]). Every Gemma3 projection (q/k/v/o,
/// the MLP gate/up/down, and the two Dense-head layers) is `bias=False`, so this
/// holds no dense output bias.
#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
pub(crate) struct Linear {
  inner: MaybeQuantizedLinear,
}

#[cfg(feature = "embeddinggemma")]
impl Linear {
  /// Build a projection from the (sanitized) weight map: a [`QuantizedLinear`]
  /// when `<prefix>.scales` is present AND a quantization config is threaded,
  /// else the dense `(out, in)`-shape-pinned [`crate::nn::Linear`].
  ///
  /// The `<prefix>.scales` sibling is the load-bearing "this layer is quantized"
  /// signal (mlx-embeddings' `class_predicate`, which quantizes every
  /// `nn.Linear`). On the quantized path the per-layer `(group_size, bits,
  /// mode)` is resolved from `quant` ([`PerLayerQuantization::quantization_for`]);
  /// the packed `uint32` triple's logical `(out, in)` is pinned to the
  /// config-derived extents by [`check_quantized_shape`] (the same load-time gate
  /// the dense `take_shaped` enforces, since the packed weight's shape differs
  /// from the dense `(out, in)` and cannot reach `take_shaped`), then the triple
  /// is built via [`QuantizedLinear::from_parts`]. The dense path is unchanged.
  ///
  /// A `<prefix>.scales` present but `quant.quantization_for(prefix)` resolving
  /// `None` (an explicit per-layer `Skip`, or no global default) is a
  /// config/checkpoint inconsistency surfaced as a typed
  /// [`Error::InvariantViolation`], never a guessed scheme.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] if `<prefix>.weight` (or, quantized,
  ///   `<prefix>.scales`) is absent;
  /// - [`Error::InvariantViolation`] if `<prefix>.scales` is present but no
  ///   scheme parameters resolve;
  /// - [`Error::LayerKeyed`] (shape / rank / dtype) from the dense `take_shaped`
  ///   or [`check_quantized_shape`] gate;
  /// - propagates [`QuantizedLinear::from_parts`]'s structural validation.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    out: i32,
    in_features: i32,
    descriptor: &'static str,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let scales_key = format!("{prefix}.scales");
    if quant.is_some() && weights.contains_key(&scales_key) {
      let Some(q) = quant.and_then(|q| q.quantization_for(prefix)) else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "EmbeddingGemma: Linear carries a `.scales` sibling but the quantization config resolved no scheme parameters for this layer",
          "a quantized Linear requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      // Pin the packed triple's logical `(out, in)` to the config BEFORE
      // construction (the dense `take_shaped` gate's quantized analogue).
      check_quantized_shape(
        weights,
        prefix,
        descriptor,
        out,
        in_features,
        q.group_size,
        q.bits,
      )?;
      let weight = take(weights, &format!("{prefix}.weight"))?;
      let scales = take(weights, &scales_key)?;
      // `.biases` is the per-group affine bias (present iff `mode == affine`);
      // `from_parts` enforces the mode/arity contract. No dense output bias
      // (every Gemma3 / Dense projection is `bias=False`).
      let quant_biases = weights.remove(&format!("{prefix}.biases"));
      let q = QuantizedLinear::from_parts(
        weight,
        scales,
        quant_biases,
        None,
        q.group_size,
        q.bits,
        q.mode.as_str(),
      )?;
      return Ok(Self {
        inner: MaybeQuantizedLinear::Quantized(q),
      });
    }

    // Dense path (unchanged): pin the `(out, in)` weight against the config.
    let weight = take_shaped(
      weights,
      &format!("{prefix}.weight"),
      descriptor,
      &[out, in_features],
    )?;
    Ok(Self {
      inner: MaybeQuantizedLinear::Dense(crate::nn::Linear::new(weight, None)),
    })
  }

  /// `y = x @ Wᵀ` (dense) or `quantized_matmul(...)` (quantized). `x` is
  /// `(..., in)`; the result is `(..., out)`.
  ///
  /// For a rank-2 dense weight this is bit-identical to the prior
  /// `swapaxes(-1, -2)` + `matmul` (`mlx.nn.Linear` transposes the full
  /// reverse-order axes, which for rank-2 is the `(-1, -2)` swap).
  ///
  /// # Errors
  /// Propagates the transpose / matmul / quantized-matmul op errors.
  pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
    self.inner.forward(x)
  }

  /// `true` if this projection was loaded from a quantized checkpoint.
  #[cfg(test)]
  pub(crate) fn is_quantized(&self) -> bool {
    self.inner.is_quantized()
  }
}

/// The packed quantized embedding table — the `(weight, scales, biases)` triple
/// plus the `group_size` / `bits` / `mode` scheme, mirroring
/// `mlx.nn.QuantizedEmbedding`. Built when a quantized EmbeddingGemma checkpoint
/// ships a quantized `model.embed_tokens` (mlx-embeddings' `class_predicate`
/// quantizes `nn.Embedding` alongside `nn.Linear`).
///
/// All fields are private; the triple's structural consistency is validated at
/// construction by [`Embedding::from_weights`] through the shared
/// [`crate::nn::quantized::validate_quantized_triple`].
#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
struct QuantizedEmbedding {
  weight: Array,
  scales: Array,
  biases: Option<Array>,
  group_size: i32,
  bits: i32,
  mode: String,
}

/// A token-embedding table of shape `(vocab, hidden)` — quantize-aware.
///
/// Mirrors `mlx.nn.Embedding` for a dense checkpoint and
/// `mlx.nn.QuantizedEmbedding` for a quantized one: [`forward`](Self::forward)
/// gathers rows by integer id (dequantizing the gathered rows on the quantized
/// path), exactly the Whisper [`Embedding`](crate::audio::stt::models::whisper)
/// adoption. The dense-or-quantized choice is held in a private `inner` enum so
/// the public surface does not leak the quantized table type.
#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
pub(crate) struct Embedding {
  inner: EmbeddingInner,
}

#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
enum EmbeddingInner {
  /// `(vocab, hidden)` dense embedding table.
  Dense(Array),
  /// Quantized embedding table.
  Quantized(QuantizedEmbedding),
}

#[cfg(feature = "embeddinggemma")]
impl Embedding {
  /// Build the token embedding from the (sanitized) weight map: a quantized
  /// table when `<prefix>.scales` is present AND a quantization config is
  /// threaded, else the dense `(vocab, hidden)`-shape-pinned table.
  ///
  /// Same auto-detect + load-time gate as [`Linear::from_weights`]: the
  /// `<prefix>.scales` sibling signals a quantized table, the per-layer scheme
  /// is resolved from `quant`, the packed triple's logical `(vocab, hidden)` is
  /// pinned by [`check_quantized_shape`], and the triple is validated by the
  /// shared [`crate::nn::quantized::validate_quantized_triple`].
  ///
  /// # Errors
  /// As [`Linear::from_weights`], plus the embedding-triple validation.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    n_vocab: i32,
    n_state: i32,
    descriptor: &'static str,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let scales_key = format!("{prefix}.scales");
    if quant.is_some() && weights.contains_key(&scales_key) {
      let Some(q) = quant.and_then(|q| q.quantization_for(prefix)) else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "EmbeddingGemma: embedding carries a `.scales` sibling but the quantization config resolved no scheme parameters for this layer",
          "a quantized embedding requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      check_quantized_shape(
        weights,
        prefix,
        descriptor,
        n_vocab,
        n_state,
        q.group_size,
        q.bits,
      )?;
      let weight = take(weights, &format!("{prefix}.weight"))?;
      let scales = take(weights, &scales_key)?;
      let biases = weights.remove(&format!("{prefix}.biases"));
      // The embedding has no separate dense output bias — its `biases` IS the
      // per-group affine bias — so the shared triple validator is the whole
      // contract (the linear-only dense-bias check does not apply).
      let mode = q.mode.as_str().to_string();
      crate::nn::quantized::validate_quantized_triple(
        "EmbeddingGemma Embedding",
        &weight,
        &scales,
        biases.as_ref(),
        q.group_size,
        q.bits,
        &mode,
      )?;
      return Ok(Embedding {
        inner: EmbeddingInner::Quantized(QuantizedEmbedding {
          weight,
          scales,
          biases,
          group_size: q.group_size,
          bits: q.bits,
          mode,
        }),
      });
    }

    let weight = take_shaped(
      weights,
      &format!("{prefix}.weight"),
      descriptor,
      &[n_vocab, n_state],
    )?;
    Ok(Embedding {
      inner: EmbeddingInner::Dense(weight),
    })
  }

  /// Gather embedding rows: `weight[ids]` (axis-0 gather), dequantizing the
  /// gathered rows on the quantized path (`mlx.nn.QuantizedEmbedding.__call__`).
  /// `ids` is an integer [`Array`] of shape `S`; the result is `S ++ (hidden,)`.
  ///
  /// # Errors
  /// Propagates the gather (`take_axis`) / dequantize op errors.
  pub(crate) fn forward(&self, ids: &Array) -> Result<Array> {
    match &self.inner {
      EmbeddingInner::Dense(weight) => ops::indexing::take_axis(weight, ids, 0),
      EmbeddingInner::Quantized(q) => {
        let w_rows = ops::indexing::take_axis(&q.weight, ids, 0)?;
        let s_rows = ops::indexing::take_axis(&q.scales, ids, 0)?;
        let b_rows = match &q.biases {
          Some(b) => Some(ops::indexing::take_axis(b, ids, 0)?),
          None => None,
        };
        ops::quantized::dequantize(
          &w_rows,
          &s_rows,
          b_rows.as_ref(),
          q.group_size,
          q.bits,
          &q.mode,
          None,
          None,
        )
      }
    }
  }

  /// The embedding table's dtype — the dtype the additive attention mask is cast
  /// to (so the fused SDPA sees a matching-dtype mask) and the dtype the gathered
  /// token embeddings carry. A cheap handle query (no eval). On the quantized
  /// path the gather dequantizes to the `scales` dtype (the activation dtype the
  /// checkpoint was quantized from), so the mask dtype is read from `scales`.
  pub(crate) fn dtype(&self) -> Result<Dtype> {
    match &self.inner {
      EmbeddingInner::Dense(weight) => weight.dtype(),
      EmbeddingInner::Quantized(q) => q.scales.dtype(),
    }
  }

  /// `true` if this embedding was loaded from a quantized checkpoint.
  #[cfg(test)]
  pub(crate) fn is_quantized(&self) -> bool {
    matches!(self.inner, EmbeddingInner::Quantized(_))
  }
}

/// Pin a quantized layer's packed `<prefix>.weight` + `<prefix>.scales` to the
/// config-derived `(out, in_features)` BEFORE construction — the quantized
/// analogue of the dense [`take_shaped`] gate (the Whisper
/// `Builder::check_quantized_shape`).
///
/// The dense path pins every consumed tensor to its exact config shape; the
/// quantized path must reach the same gate, because the packed `uint32` weight
/// has shape `(out, in * bits / 32)` (NOT the dense `(out, in)`) and so cannot
/// go through `take_shaped`. The recovery mirrors mlx's quantized layout
/// (`mlx/ops.cpp:107,131,4790-4792`):
///
/// - the weight is rank-2 `uint32`; its leading axis is the logical output dim
///   (must equal `out`), and its logical input width — mlx's `w.shape(-1) * 32 /
///   bits` (the contraction dim) — must equal `in_features`;
/// - the `scales` are rank-2; the leading axis must equal `out`, and
///   `scales.shape(-1) * group_size` must equal `in_features` (mlx's
///   `w.shape(-1) * 32 / bits == scales.shape(-1) * group_size`).
///
/// `group_size` / `bits` are checked `> 0` before they divide (a non-positive
/// value is a malformed config and an [`Error::OutOfRange`], never a panic). The
/// per-mode value tables remain mlx-c's. Reads only `shape()` / `dtype()`
/// metadata (no materialization), so it is bounded regardless of the declared
/// dims. On mismatch returns a typed error wrapped in an [`Error::LayerKeyed`]
/// naming the offending `<prefix>.weight` / `<prefix>.scales` key.
#[cfg(feature = "embeddinggemma")]
#[allow(clippy::too_many_arguments)]
fn check_quantized_shape(
  weights: &HashMap<String, Array>,
  prefix: &str,
  descriptor: &'static str,
  out: i32,
  in_features: i32,
  group_size: i32,
  bits: i32,
) -> Result<()> {
  // The scheme parameters divide the recovered widths; a non-positive value is a
  // malformed config (`from_parts` / the triple validator also reject it, but
  // guard here so the divisions below cannot trap).
  if bits <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "EmbeddingGemma: quantized layer bits",
      "must be > 0",
      smol_str::format_smolstr!("{bits}"),
    )));
  }
  if group_size <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "EmbeddingGemma: quantized layer group_size",
      "must be > 0",
      smol_str::format_smolstr!("{group_size}"),
    )));
  }

  // Packed weight `(out, in * bits / 32)`, `uint32`.
  let weight_key = format!("{prefix}.weight");
  let weight = weights.get(&weight_key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "EmbeddingGemma: quantized weight not found in checkpoint",
      weight_key.clone(),
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
      "EmbeddingGemma: quantized scales not found in checkpoint",
      scales_key.clone(),
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

/// A `(1,)` scalar constant carrying `value`, in the **same dtype** as `like`
/// (`like.dtype()`).
///
/// The crate's uniform stand-in for MLX *weak-scalar* / python `astype(x.dtype)`
/// semantics: a constant that meets the hidden tensor adopts the tensor's dtype
/// so a bf16 backbone is not silently promoted to f32. Gemma3 scales the token
/// embedding by `sqrt(hidden_size)` and forms its RMSNorm weight as `1.0 +
/// weight`, both through this helper. Implemented as an f32 scalar → cast; for
/// an f32 `like` this is a no-op cast (bit-identical to a direct
/// `Array::full::<f32>`).
#[cfg(feature = "embeddinggemma")]
pub(crate) fn scalar_like(value: f32, like: &Array) -> Result<Array> {
  crate::error::ensure_handler_installed();
  let s = Array::full::<f32>(&(1,), value)?;
  ops::misc::astype(&s, like.dtype()?)
}

/// The EmbeddingGemma token-embedding scale as a `(1,)` scalar in `like`'s
/// dtype: `sqrt(hidden_size)`, built in the `embed_tokens` weight dtype and then
/// cast to the hidden dtype — with **no** bf16 rounding.
///
/// Mirrors the mlx-embeddings encoder `gemma3_text.py`'s
/// `h *= mx.array(hidden_size ** 0.5, embed_tokens.weight.dtype).astype(h.dtype)`.
/// The token embeddings `h` are gathered straight from the `embed_tokens`
/// weight, so `like.dtype()` *is* the weight dtype: building the scalar in
/// `like`'s dtype reproduces both the build-in-weight-dtype and the
/// `astype(h.dtype)` (a no-op in the normal case) of the reference. The `sqrt`
/// is computed in f32 (python `** 0.5` on the integer `hidden_size`), then
/// [`scalar_like`] narrows the scalar to `like`'s dtype.
///
/// This deliberately does **not** round through bf16: that form is the
/// generative `mlx-lm` Gemma3 model (`mx.array(..., mx.bfloat16)`), not the
/// EmbeddingGemma encoder, and would alter the embeddings of an f32 or f16
/// checkpoint before the first block.
#[cfg(feature = "embeddinggemma")]
pub(crate) fn embedding_scale_like(hidden_size: f32, like: &Array) -> Result<Array> {
  scalar_like(hidden_size.sqrt(), like)
}

/// Pull `shape[axis]` as `i32`, erroring on rank underflow or `i32::MAX`
/// overflow. Mirrors the SigLIP2 `dim_i32`.
#[cfg(feature = "embeddinggemma")]
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
/// the key if absent. Mirrors the SigLIP2 / Wav2Vec2 / LFM2 `take`.
#[cfg(feature = "embeddinggemma")]
pub(crate) fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights.remove(key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "EmbeddingGemmaModel::from_weights",
      key,
    ))
  })
}

/// Assert a checkpoint tensor's shape (rank + every dimension) equals the
/// `expected` shape the architecture requires, before it is stored or fed to any
/// op. On mismatch returns an [`Error::ShapePairMismatch`] (both full shapes)
/// wrapped in an [`Error::LayerKeyed`] naming the offending `key`. Mirrors the
/// SigLIP2 `expect_shape`.
#[cfg(feature = "embeddinggemma")]
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

/// [`take`] a weight by key, then assert its shape equals `expected` via
/// [`expect_shape`] — the fused fetch-and-shape-check the builders use for every
/// tensor stored verbatim, so a consumed tensor can never skip the gate.
#[cfg(feature = "embeddinggemma")]
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

/// Build a Gemma3 [`crate::lm::nn::norm::RMSNorm`] from `{key}.weight` (pinned
/// to `(dims,)`), folding in Gemma's `1.0 + weight` reparameterization.
///
/// `gemma3_text.py`'s `RMSNorm.__call__` is `mx.fast.rms_norm(x, 1.0 + weight,
/// eps)` — the saved weight is the *delta* from unity. The fused
/// [`crate::lm::nn::norm::RMSNorm`] applies its stored weight directly, so the
/// `1.0 +` shift is materialized **once at load** (a cheap elementwise add on a
/// `(dims,)` vector) and the resulting scale is handed to the norm — exactly
/// reproducing the reference without a per-forward add.
#[cfg(feature = "embeddinggemma")]
pub(crate) fn build_gemma_rms_norm(
  weights: &mut HashMap<String, Array>,
  key: &str,
  dims: i32,
  eps: f32,
) -> Result<crate::lm::nn::norm::RMSNorm> {
  let raw = take_shaped(
    weights,
    &format!("{key}.weight"),
    "Gemma3 RMSNorm weight (dims,)",
    &[dims],
  )?;
  // `1.0 + weight`, in the weight's own dtype (no f32 promotion of a bf16
  // checkpoint).
  let one = scalar_like(1.0, &raw)?;
  let scale = one.add(&raw)?;
  Ok(crate::lm::nn::norm::RMSNorm::new(scale, eps))
}

/// `Dtype` re-export so the backbone's mask construction can name it without a
/// second `use`.
#[cfg(feature = "embeddinggemma")]
pub(crate) type DtypeAlias = Dtype;

#[cfg(all(test, feature = "embeddinggemma"))]
mod tests {
  use super::*;

  /// Read a `(1,)` scalar array back as `f32` (cast first — `to_vec` is
  /// dtype-strict, so a non-f32 scalar must be cast before it can be read).
  fn scalar_to_f32(a: &Array) -> f32 {
    let mut a = ops::misc::astype(a, Dtype::F32).unwrap();
    a.eval().unwrap();
    a.to_vec::<f32>().unwrap()[0]
  }

  /// The embedding scale must match the mlx-embeddings encoder formula
  /// `mx.array(hidden_size ** 0.5, embed_tokens.weight.dtype).astype(h.dtype)`
  /// for an f32, an f16, and a bf16 hidden dtype. In the port `h` is gathered
  /// from `embed_tokens`, so the weight dtype equals `h`'s dtype: the oracle
  /// builds `sqrt(hidden_size)` directly in the hidden dtype (an f32 scalar
  /// narrowed by the `half` crate, the same rounding MLX uses) and asserts the
  /// helper reproduces it — and, critically, that the value is **not** rounded
  /// through bf16 (the wrong, mlx-lm generative form).
  #[test]
  fn embedding_scale_matches_mlx_embeddings_formula_per_dtype() {
    // EmbeddingGemma's shipped hidden size (768) plus a couple of other sizes,
    // to confirm the rounding is per-dtype (not a single coincidental value).
    for hidden_size in [768.0_f32, 1152.0, 3584.0] {
      let sqrt = hidden_size.sqrt();
      // Build the scalar in the weight dtype, then cast to the hidden dtype.
      // Here weight dtype == hidden dtype, so each is simply `sqrt` narrowed to
      // that dtype.
      let expect_f32 = sqrt;
      let expect_f16 = half::f16::from_f32(sqrt).to_f32();
      let expect_bf16 = half::bf16::from_f32(sqrt).to_f32();

      // A `(1,)` `like` tensor in each hidden dtype.
      let one_f32 = Array::full::<f32>(&(1,), 0.0).unwrap();
      let like_f32 = ops::misc::astype(&one_f32, Dtype::F32).unwrap();
      let like_f16 = ops::misc::astype(&one_f32, Dtype::F16).unwrap();
      let like_bf16 = ops::misc::astype(&one_f32, Dtype::BF16).unwrap();

      let scale_f32 = embedding_scale_like(hidden_size, &like_f32).unwrap();
      let scale_f16 = embedding_scale_like(hidden_size, &like_f16).unwrap();
      let scale_bf16 = embedding_scale_like(hidden_size, &like_bf16).unwrap();

      assert_eq!(
        scale_f32.dtype().unwrap(),
        Dtype::F32,
        "f32 scale stays f32"
      );
      assert_eq!(
        scale_f16.dtype().unwrap(),
        Dtype::F16,
        "f16 scale stays f16"
      );
      assert_eq!(
        scale_bf16.dtype().unwrap(),
        Dtype::BF16,
        "bf16 scale stays bf16"
      );

      assert_eq!(
        scalar_to_f32(&scale_f32),
        expect_f32,
        "f32 scale must equal the full-precision f32 sqrt({hidden_size}), not a bf16-rounded value"
      );
      assert_eq!(
        scalar_to_f32(&scale_f16),
        expect_f16,
        "f16 scale must be the f32 sqrt({hidden_size}) cast to f16 (f32→f16, not bf16→f16)"
      );
      assert_eq!(
        scalar_to_f32(&scale_bf16),
        expect_bf16,
        "bf16 scale must be the f32 sqrt({hidden_size}) cast to bf16"
      );
    }
  }

  /// Pin the regression: the f32-hidden scale must be the full-precision f32
  /// `sqrt`, **never** the bf16-rounded value of the mlx-lm generative model.
  /// For `hidden_size = 768` the bf16 round is lossy against the f32 `sqrt`, so
  /// the two forms are genuinely distinguishable — guaranteeing the wrong
  /// (bf16) reference cannot regress back in.
  #[test]
  fn embedding_scale_is_not_bf16_rounded() {
    let full_f32 = 768.0_f32.sqrt();
    let bf16_rounded = half::bf16::from_f32(full_f32).to_f32();
    assert_ne!(
      bf16_rounded, full_f32,
      "bf16 rounding of sqrt(768) must drop precision vs the f32 sqrt (so the two references differ)"
    );
    let like_f32 = Array::full::<f32>(&(1,), 0.0).unwrap();
    let scale = scalar_to_f32(&embedding_scale_like(768.0, &like_f32).unwrap());
    assert_eq!(
      scale, full_f32,
      "f32-hidden scale must be the full-precision f32 sqrt"
    );
    assert_ne!(
      scale, bf16_rounded,
      "f32-hidden scale must NOT be the bf16-rounded mlx-lm value"
    );
  }
}
