//! Quantize-aware building blocks for the Qwen3 decoder: the bias-free
//! [`Linear`] projection and the token [`Embedding`] (with its weight-tied
//! [`Embedding::as_linear`] head), plus the weight fetch + shape-pinning
//! helpers every Qwen3 loader shares.
//!
//! ## Quantize-aware layers
//!
//! [`Linear`] wraps the shared [`crate::nn::MaybeQuantizedLinear`] and
//! [`Embedding`] wraps a dense table or a quantized `(weight, scales, biases)`
//! triple, so a Qwen3 projection / the token embedding loads either a dense or
//! an 8-bit/4-bit quantized checkpoint through one forward path — the same
//! adoption Whisper / EmbeddingGemma take. The builders auto-detect the
//! quantized variant from the `<prefix>.scales` sibling ALONE (mlx-lm's
//! `class_predicate`, which quantizes every `nn.Linear` / `nn.Embedding`) — the
//! same `.scales`-presence discriminator the shared
//! [`crate::nn::MaybeQuantizedLinear::from_weights`] uses — with the per-layer
//! `(group_size, bits, mode)` resolved from the parsed
//! [`crate::lm::quant::PerLayerQuantization`]. A `<prefix>.scales` present but
//! no resolvable scheme is a typed [`Error::InvariantViolation`] (a mixed /
//! malformed checkpoint), never a silent fall-through to the dense path. The
//! dense path is byte-for-byte the prior `(out, in)`-shape-pinned
//! `matmul(x, weightᵀ)` / dense gather.
//!
//! ## Shape pinning
//!
//! Every consumed dense tensor's shape is pinned to the config-derived extents
//! via [`take_shaped`] (the same discipline as the merged SigLIP2 / Wav2Vec2 /
//! LFM2 ports). The quantized path pins the packed `uint32` triple's logical
//! `(out, in)` via [`check_quantized_shape`] (the dense gate's quantized
//! analogue, since a packed weight's shape differs from the dense `(out, in)`
//! and cannot reach [`take_shaped`]), then validates the triple's full
//! contract via the shared
//! [`crate::nn::quantized::validate_quantized_triple`].

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
  ops::{self, indexing::take_axis, linalg_basic::matmul, shape::swapaxes},
};
use smol_str::format_smolstr;

/// The mlx-quantized layout's sibling-key suffixes: `<prefix>.weight` (packed
/// `uint32` matrix), `<prefix>.scales`, and the per-group affine
/// `<prefix>.biases`. A `<prefix>.scales` sibling is the load-bearing "this
/// layer is quantized" signal mlx-lm's loader keys on.
const WEIGHT_SUFFIX: &str = ".weight";
pub(crate) const SCALES_SUFFIX: &str = ".scales";
const BIASES_SUFFIX: &str = ".biases";

/// A Qwen3 linear projection `y = x @ Wᵀ` — quantize-aware, bias-free.
///
/// Mirrors `mlx.nn.Linear` (`weight` stored `(out, in)`, the forward transposes
/// it) for a dense checkpoint and `mlx.nn.QuantizedLinear` for a quantized one,
/// sharing one [`forward`](Self::forward) call site via [`MaybeQuantizedLinear`]
/// — the same adoption Whisper / EmbeddingGemma take. Every Qwen3 projection
/// (`q/k/v/o_proj`, the MLP `gate/up/down_proj`, the optional untied `lm_head`)
/// is `bias=False` (`qwen3.py:44-47,95-97,170`), so this holds no dense output
/// bias.
#[derive(Debug)]
pub(crate) struct Linear {
  inner: MaybeQuantizedLinear,
}

impl Linear {
  /// Build a projection from the (sanitized) weight map: a [`QuantizedLinear`]
  /// when `<prefix>.scales` is present, else the dense
  /// `(out, in)`-shape-pinned [`crate::nn::Linear`].
  ///
  /// The `<prefix>.scales` sibling ALONE is the load-bearing "this layer is
  /// quantized" signal (mlx-lm's `class_predicate`, which quantizes every
  /// `nn.Linear`) — the same `.scales`-presence discriminator the shared
  /// [`crate::nn::MaybeQuantizedLinear::from_weights`] uses, so a layer carrying
  /// `.scales` is NEVER reinterpreted as dense merely because the config quant
  /// block is absent. On the quantized path the per-layer `(group_size, bits,
  /// mode)` is resolved from `quant`
  /// ([`PerLayerQuantization::quantization_for`]); the packed
  /// `uint32` triple's logical `(out, in)` is pinned to the config-derived
  /// extents by [`check_quantized_shape`] (the same load-time gate the dense
  /// [`take_shaped`] enforces, since the packed weight's shape differs from the
  /// dense `(out, in)` and cannot reach [`take_shaped`]), then the triple is
  /// built via [`QuantizedLinear::from_parts`]. The dense path is unchanged.
  ///
  /// A `<prefix>.scales` present but no resolvable scheme — `quant == None` (no
  /// `quantization` block at all), an explicit per-layer `Skip`, or no global
  /// default — is a config/checkpoint inconsistency surfaced as a typed
  /// [`Error::InvariantViolation`], never a guessed scheme nor a silent dense
  /// reinterpret of the packed weight.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] if `<prefix>.weight` (or, quantized,
  ///   `<prefix>.scales`) is absent;
  /// - [`Error::InvariantViolation`] if `<prefix>.scales` is present but no
  ///   scheme parameters resolve;
  /// - [`Error::LayerKeyed`] (shape / rank / dtype) from the dense [`take_shaped`]
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
    let scales_key = format!("{prefix}{SCALES_SUFFIX}");
    if weights.contains_key(&scales_key) {
      let Some(q) = quant.and_then(|q| q.quantization_for(prefix)) else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "Qwen3: Linear carries a `.scales` sibling but the quantization config resolved no scheme parameters for this layer",
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
      let weight = take(weights, &format!("{prefix}{WEIGHT_SUFFIX}"))?;
      let scales = take(weights, &scales_key)?;
      // `.biases` is the per-group affine bias (present iff `mode == affine`);
      // `from_parts` enforces the mode/arity contract. No dense output bias
      // (every Qwen3 projection is `bias=False`).
      let quant_biases = weights.remove(&format!("{prefix}{BIASES_SUFFIX}"));
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
      &format!("{prefix}{WEIGHT_SUFFIX}"),
      descriptor,
      &[out, in_features],
    )?;
    Ok(Self {
      inner: MaybeQuantizedLinear::Dense(crate::nn::Linear::new(weight, None)),
    })
  }

  /// Construct a **dense** projection directly from a `(out, in)` `weight` (no
  /// bias) — the test constructor the attention oracle uses to build an
  /// `Attention` from in-memory matrices without a weight map.
  #[cfg(test)]
  pub(crate) fn dense(weight: Array) -> Self {
    Self {
      inner: MaybeQuantizedLinear::Dense(crate::nn::Linear::new(weight, None)),
    }
  }

  /// `y = x @ Wᵀ` (dense) or `quantized_matmul(...)` (quantized). `x` is
  /// `(..., in)`; the result is `(..., out)`.
  ///
  /// For a rank-2 dense weight this is bit-identical to the prior
  /// `swapaxes(-1, -2)` + `matmul` (`mlx.nn.Linear` transposes the full
  /// reverse-order axes, which for rank-2 is the `(-1, -2)` swap). The quantized
  /// path's [`crate::ops::quantized::quantized_matmul`] follows `x`'s dtype, so a
  /// bf16/f16 activation is not promoted.
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
/// `mlx.nn.QuantizedEmbedding`. Built when a quantized Qwen3 checkpoint ships a
/// quantized `model.embed_tokens` (mlx-lm's `class_predicate` quantizes
/// `nn.Embedding` alongside `nn.Linear`).
///
/// All fields are private; the triple's structural consistency is validated at
/// construction by [`Embedding::from_weights`] through the shared
/// [`crate::nn::quantized::validate_quantized_triple`].
#[derive(Debug)]
struct QuantizedEmbedding {
  /// Packed `(vocab, packed_hidden)` quantized table (`uint32`).
  weight: Array,
  /// Per-group `scales` `(vocab, n_groups)`.
  scales: Array,
  /// Per-group affine `biases` (`None` for scale-only `fp` modes).
  biases: Option<Array>,
  group_size: i32,
  bits: i32,
  mode: String,
}

/// The Qwen3 token-embedding table of shape `(vocab, hidden)`, with a weight-tied
/// [`Embedding::as_linear`] projection — quantize-aware.
///
/// Mirrors `mlx.nn.Embedding` for a dense checkpoint and
/// `mlx.nn.QuantizedEmbedding` for a quantized one: [`Embedding::forward`]
/// gathers rows by integer id (dequantizing the gathered rows on the quantized
/// path); [`Embedding::as_linear`] reuses the SAME table as a linear projection
/// `x @ weightᵀ` — the Qwen3 tied logit head (`qwen3.py:180`,
/// `self.model.embed_tokens.as_linear(out)`). The dense-or-quantized choice is
/// held in a private `inner` enum so the public surface does not leak the
/// quantized table type.
#[derive(Debug)]
pub(crate) struct Embedding {
  inner: EmbeddingInner,
}

#[derive(Debug)]
enum EmbeddingInner {
  /// `(vocab, hidden)` dense embedding table.
  Dense(Array),
  /// Quantized embedding table.
  Quantized(QuantizedEmbedding),
}

impl Embedding {
  /// Build the token embedding from the (sanitized) weight map: a quantized
  /// table when `<prefix>.scales` is present, else the dense
  /// `(vocab, hidden)`-shape-pinned table.
  ///
  /// Same auto-detect + load-time gate as [`Linear::from_weights`]: the
  /// `<prefix>.scales` sibling ALONE signals a quantized table (so a
  /// `.scales`-bearing table is never reinterpreted as dense when the config
  /// quant block is absent), the per-layer scheme is resolved from `quant`, the
  /// packed triple's logical `(vocab, hidden)` is pinned by
  /// [`check_quantized_shape`], and the triple is validated by the shared
  /// [`crate::nn::quantized::validate_quantized_triple`]. A `<prefix>.scales`
  /// present but no resolvable scheme (`quant == None`, an explicit per-layer
  /// `Skip`, or no global default) is a typed [`Error::InvariantViolation`],
  /// never a silent dense reinterpret. The embedding has NO separate dense
  /// output bias — its `biases` IS the per-group affine bias — so the shared
  /// triple validator is the whole contract.
  ///
  /// # Errors
  /// As [`Linear::from_weights`], plus the embedding-triple validation.
  pub(crate) fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    vocab: i32,
    hidden: i32,
    descriptor: &'static str,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let scales_key = format!("{prefix}{SCALES_SUFFIX}");
    if weights.contains_key(&scales_key) {
      let Some(q) = quant.and_then(|q| q.quantization_for(prefix)) else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "Qwen3: embedding carries a `.scales` sibling but the quantization config resolved no scheme parameters for this layer",
          "a quantized embedding requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      check_quantized_shape(
        weights,
        prefix,
        descriptor,
        vocab,
        hidden,
        q.group_size,
        q.bits,
      )?;
      let weight = take(weights, &format!("{prefix}{WEIGHT_SUFFIX}"))?;
      let scales = take(weights, &scales_key)?;
      let biases = weights.remove(&format!("{prefix}{BIASES_SUFFIX}"));
      let mode = q.mode.as_str().to_string();
      crate::nn::quantized::validate_quantized_triple(
        "Qwen3 Embedding",
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
      &format!("{prefix}{WEIGHT_SUFFIX}"),
      descriptor,
      &[vocab, hidden],
    )?;
    Ok(Embedding {
      inner: EmbeddingInner::Dense(weight),
    })
  }

  /// The number of embedding rows — the table's leading (vocab) axis, the valid
  /// token-id range `embed_tokens` bound-checks against. For a quantized table
  /// the packed `(vocab, packed_hidden)` weight's leading axis is still `vocab`,
  /// so this is the row count regardless of packing. Cheap metadata (no eval).
  pub(crate) fn row_count(&self) -> usize {
    let weight = match &self.inner {
      EmbeddingInner::Dense(weight) => weight,
      EmbeddingInner::Quantized(q) => &q.weight,
    };
    weight.shape().first().copied().unwrap_or(0)
  }

  /// Gather embedding rows: `weight[ids]` — gather along axis 0 (the vocab
  /// axis), dequantizing the gathered rows on the quantized path
  /// (`mlx.nn.QuantizedEmbedding.__call__`). `ids` is an integer [`Array`] of
  /// shape `S`; the result is `S ++ (hidden,)`. (Plain `take` would flatten the
  /// table — `take_axis(.., 0)` is the row-gather.)
  ///
  /// # Errors
  /// Propagates the gather (`take_axis`) / dequantize op errors.
  pub(crate) fn forward(&self, ids: &Array) -> Result<Array> {
    match &self.inner {
      EmbeddingInner::Dense(weight) => take_axis(weight, ids, 0),
      EmbeddingInner::Quantized(q) => {
        // `mlx.nn.QuantizedEmbedding.__call__`: gather the packed rows + the
        // per-row scales / biases by id, then dequantize the gathered rows.
        let w_rows = take_axis(&q.weight, ids, 0)?;
        let s_rows = take_axis(&q.scales, ids, 0)?;
        let b_rows = match &q.biases {
          Some(b) => Some(take_axis(b, ids, 0)?),
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

  /// Weight-tied linear projection `x @ weightᵀ` (the Qwen3 tied logit head,
  /// `qwen3.py:180`), mirroring `mlx.nn.Embedding.as_linear` (dense) and
  /// `mlx.nn.QuantizedEmbedding.as_linear`'s
  /// `quantized_matmul(x, weight, scales, biases, transpose=True, ...)`
  /// (quantized). `x` is `(..., hidden)`; the result is `(..., vocab)`. The
  /// quantized `quantized_matmul` follows `x`'s dtype (no promotion).
  ///
  /// # Errors
  /// Propagates the transpose / matmul / quantized-matmul op errors.
  pub(crate) fn as_linear(&self, x: &Array) -> Result<Array> {
    match &self.inner {
      EmbeddingInner::Dense(weight) => {
        let wt = swapaxes(weight, -1, -2)?;
        matmul(x, &wt)
      }
      EmbeddingInner::Quantized(q) => ops::quantized::quantized_matmul(
        x,
        &q.weight,
        &q.scales,
        q.biases.as_ref(),
        true,
        q.group_size,
        q.bits,
        &q.mode,
      ),
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
/// analogue of the dense [`take_shaped`] gate (the Whisper / EmbeddingGemma
/// `check_quantized_shape`).
///
/// The dense path pins every consumed tensor to its exact config shape; the
/// quantized path must reach the same gate, because the packed `uint32` weight
/// has shape `(out, in * bits / 32)` (NOT the dense `(out, in)`) and so cannot
/// go through [`take_shaped`]. The recovery mirrors mlx's quantized layout
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
      "Qwen3: quantized layer bits",
      "must be > 0",
      format_smolstr!("{bits}"),
    )));
  }
  if group_size <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Qwen3: quantized layer group_size",
      "must be > 0",
      format_smolstr!("{group_size}"),
    )));
  }

  // Packed weight `(out, in * bits / 32)`, `uint32`.
  let weight_key = format!("{prefix}{WEIGHT_SUFFIX}");
  let weight = weights.get(&weight_key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "Qwen3: quantized weight not found in checkpoint",
      format_smolstr!("{weight_key}"),
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
  let scales_key = format!("{prefix}{SCALES_SUFFIX}");
  let scales = weights.get(&scales_key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "Qwen3: quantized scales not found in checkpoint",
      format_smolstr!("{scales_key}"),
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

/// Pull a required weight out of the map by exact `key`, erroring with the key
/// on absence (mlx's `model.update(tree_unflatten(weights))` would raise).
pub(crate) fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights.remove(key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "Qwen3 weight map",
      format_smolstr!("{key}"),
    ))
  })
}

/// Assert a tensor's shape equals `expected` (rank + every dim) before it is
/// stored, so a checkpoint whose weight disagrees with the config-derived
/// expectation is rejected here rather than running a different graph (or
/// admitting an out-of-bounds embedding gather). On mismatch returns
/// [`Error::ShapePairMismatch`] wrapped in [`Error::LayerKeyed`] naming `key`,
/// mirroring how the Qwen3-ASR text loader reports a malformed decoder weight.
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

/// [`take`] a weight by key, then assert its shape via [`expect_shape`] — the
/// fused fetch-and-shape-check used for every dense tensor stored verbatim, so a
/// consumed tensor can never skip the gate. Mirrors the Qwen3-ASR text loader's
/// `take_shaped`.
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
