//! LFM2 neural-network projections: a quantize-aware [`Linear`] and a
//! quantize-aware token [`Embedding`] (the weight-tied logit head).
//!
//! Mirrors `mlx.nn.Linear` / `mlx.nn.Embedding` for a dense checkpoint and
//! `mlx.nn.QuantizedLinear` / `mlx.nn.QuantizedEmbedding` for an mlx-community
//! quantized checkpoint (e.g. `LiquidAI/LFM2.5-VL-450M-MLX-8bit`). The dense
//! and quantized cases share one [`Linear::forward`] / [`Embedding::forward`]
//! call site, so the LFM2 block code is unchanged whether the weights are
//! dense or quantized — the same adoption pattern Whisper takes
//! ([`crate::audio::stt::models::whisper`]).
//!
//! Both wrappers route through the shared
//! [`MaybeQuantizedLinear`](crate::nn::MaybeQuantizedLinear) /
//! [`validate_quantized_triple`](crate::nn::quantized::validate_quantized_triple)
//! foundation — the ONE place mlx's quantized-triple contract is mirrored — so
//! there is no per-model re-implementation of the quantization layout checks.

use std::collections::HashMap;

use crate::{
  array::Array,
  error::{Error, InvariantViolationPayload, MissingKeyPayload, Result},
  lm::quant::PerLayerQuantization,
  nn::MaybeQuantizedLinear,
  ops::{self, indexing::take_axis, linalg_basic::matmul, shape::swapaxes},
};
use smol_str::format_smolstr;

/// The `<prefix>.scales` sibling suffix the mlx-quantized layout writes next to
/// `<prefix>.weight` — the load-bearing "this layer is quantized" signal
/// mlx-lm's loader keys on (`f"{p}.scales" in weights`,
/// `mlx_lm/utils.py:349-352`). Shared by [`Linear::from_weights`] and
/// [`Embedding::from_weights`].
const SCALES_SUFFIX: &str = ".scales";
const WEIGHT_SUFFIX: &str = ".weight";
const BIASES_SUFFIX: &str = ".biases";

/// A dense or quantized linear projection: `weight` is `(out, in)`
/// (the `nn.Linear` layout) and the optional `bias` is `(out,)`.
///
/// For a dense checkpoint this is `y = x @ weight.T (+ bias)`; for a quantized
/// one it is `quantized_matmul(x, weight, scales, biases, transpose=True, …)
/// (+ bias)`. The two cases dispatch through the shared
/// [`MaybeQuantizedLinear`]. No implicit eval — every op appends to the lazy
/// graph.
#[derive(Debug)]
pub struct Linear {
  inner: MaybeQuantizedLinear,
}

impl Linear {
  /// Construct a **dense** projection from a `(out, in)` weight and optional
  /// `(out,)` bias.
  pub fn new(weight: Array, bias: Option<Array>) -> Self {
    Self {
      inner: MaybeQuantizedLinear::Dense(crate::nn::Linear::new(weight, bias)),
    }
  }

  /// Build a dense-or-quantized projection from the checkpoint weight map by
  /// the presence of the `<prefix>.scales` sibling — the weight-map analogue
  /// of mlx-lm's `f"{p}.scales" in weights` predicate (`utils.py:349-352`).
  ///
  /// Delegates to [`MaybeQuantizedLinear::from_weights`]: a `<prefix>.scales`
  /// sibling builds the quantized variant with the per-layer-resolved
  /// `(group_size, bits, mode)` from `quant`; its absence builds the dense
  /// variant. Both paths consume `<prefix>.weight` (+ the optional dense
  /// `<prefix>.bias`) from `weights` by key, and the quantized path also
  /// consumes `<prefix>.scales` / `<prefix>.biases`. `quant` is the resolved
  /// scheme for THIS layer (the caller passes
  /// [`PerLayerQuantization::quantization_for`]); a dense layer passes `None`.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] if `<prefix>.weight` (or the quantized path's
  ///   `<prefix>.scales`) is absent;
  /// - [`Error::InvariantViolation`] if `<prefix>.scales` is present but
  ///   `quant` resolved no scheme parameters;
  /// - propagates [`crate::nn::QuantizedLinear::from_parts`]'s structural
  ///   validation of the quantized triple.
  pub fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    quant: Option<(i32, i32, &str)>,
  ) -> Result<Self> {
    let inner = MaybeQuantizedLinear::from_weights(weights, prefix, quant)?;
    Ok(Self { inner })
  }

  /// Build a dense-or-quantized projection with an **explicitly-supplied** dense
  /// output `bias`, rather than auto-consuming `<prefix>.bias` from the map.
  ///
  /// Used for the short-conv `in_proj` / `out_proj`, whose dense bias presence
  /// is gated by the authoritative `conv_bias` config flag: the caller takes
  /// the `<prefix>.bias` through the
  /// [`take_if`](crate::model_validation::take_if) gate (which enforces
  /// required-when-`true` / forbidden-when-`false`), then passes the result
  /// here. The bias is applied on BOTH the dense and quantized paths — mlx's
  /// `QuantizedLinear.from_linear` preserves the source `Linear.bias` — so the
  /// dense-bias arity is identical whether the projection is dense or quantized,
  /// mirroring the Whisper `linear()` builder. The packed `<prefix>.weight`
  /// (and, on the quantized path, `<prefix>.scales` / `<prefix>.biases`) are
  /// still consumed from `weights` by key.
  ///
  /// # Errors
  /// As [`Self::from_weights`], plus
  /// [`crate::nn::QuantizedLinear::from_parts`]'s dense-bias arity check (the
  /// bias, if `Some`, must be `(out,)`).
  pub fn from_weights_with_bias(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    quant: Option<(i32, i32, &str)>,
    bias: Option<Array>,
  ) -> Result<Self> {
    let scales_key = format!("{prefix}{SCALES_SUFFIX}");
    if weights.contains_key(&scales_key) {
      let Some((group_size, bits, mode)) = quant else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "lfm2::Linear::from_weights_with_bias: checkpoint carries a `.scales` sibling for this projection but no quantization config resolved scheme parameters",
          "a quantized Linear requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      let scales = take_required(weights, prefix, SCALES_SUFFIX)?;
      let quant_biases = weights.remove(&format!("{prefix}{BIASES_SUFFIX}"));
      let q = crate::nn::QuantizedLinear::from_parts(
        weight,
        scales,
        quant_biases,
        bias,
        group_size,
        bits,
        mode,
      )?;
      Ok(Self {
        inner: MaybeQuantizedLinear::Quantized(q),
      })
    } else {
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      Ok(Self::new(weight, bias))
    }
  }

  /// `y = x @ weight.T (+ bias)` (dense) or `quantized_matmul(...) (+ bias)`
  /// (quantized). `x` is `(..., in)`; the result is `(..., out)`.
  ///
  /// # Errors
  /// Propagates the transpose / matmul / quantized-matmul / add op errors.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    self.inner.forward(x)
  }

  /// `true` if this projection was loaded from a quantized checkpoint.
  #[cfg(test)]
  pub fn is_quantized(&self) -> bool {
    self.inner.is_quantized()
  }

  /// The dense `(out, in)` weight when this is a dense projection, else `None`
  /// (a quantized projection holds a packed `uint32` weight + scales/biases, not
  /// a plain dense matrix).
  ///
  /// Used by the attention QKV builder to concatenate three dense `q/k/v`
  /// projections into one fused matmul (an exact, numerically-identical
  /// rearrangement); a quantized projection returns `None` and stays a separate
  /// quantized matmul.
  pub fn dense_weight(&self) -> Option<&Array> {
    match &self.inner {
      MaybeQuantizedLinear::Dense(l) => Some(l.weight_ref()),
      MaybeQuantizedLinear::Quantized(_) => None,
    }
  }

  /// The dense `(out,)` output bias when this is a dense projection that carries
  /// one, else `None` (a dense projection without a bias, or a quantized one).
  ///
  /// Paired with [`Self::dense_weight`] by the attention QKV builder: when the
  /// three dense `q/k/v` weights are concatenated into one fused matmul, their
  /// biases (if any) are concatenated into the fused bias the same way, so the
  /// fused split reproduces each projection's bias add bit-for-bit. LFM2's
  /// attention projections are bias-free (`bias=False`, `lfm2.py:68-70`), so on
  /// a faithful checkpoint this is always `None`; the accessor exists so a
  /// (hypothetical) biased dense projection is fused faithfully rather than
  /// having its bias silently dropped.
  pub fn dense_bias(&self) -> Option<&Array> {
    match &self.inner {
      MaybeQuantizedLinear::Dense(l) => l.bias(),
      MaybeQuantizedLinear::Quantized(_) => None,
    }
  }
}

/// The packed quantized embedding table — the `(weight, scales, biases)`
/// triple plus the `group_size` / `bits` / `mode` scheme parameters, mirroring
/// `mlx.nn.QuantizedEmbedding`
/// (`mlx/python/mlx/nn/layers/quantized.py:99-196`).
///
/// All fields are private; the triple's structural consistency is validated at
/// construction by [`Embedding::quantized`] through the shared
/// [`validate_quantized_triple`](crate::nn::quantized::validate_quantized_triple).
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

/// A token embedding table of shape `(vocab, hidden)`, with a weight-tied
/// [`Embedding::as_linear`] projection — quantize-aware.
///
/// Mirrors `mlx.nn.Embedding` for a dense checkpoint and
/// `mlx.nn.QuantizedEmbedding` for a quantized one: [`Embedding::forward`]
/// gathers rows by integer id; [`Embedding::as_linear`] reuses the SAME table
/// as a linear projection (the LFM2 weight-tied logit head — `lfm2.py:296`
/// ends with `self.model.embed_tokens.as_linear(out)`). Handling the tie
/// through ONE table keeps the gather and the head bit-identically consistent
/// under quantization, exactly as mlx's `QuantizedEmbedding` does (its
/// `__call__` dequantizes the gathered rows; its `as_linear` runs
/// `quantized_matmul` over the same packed table).
#[derive(Debug)]
pub struct Embedding {
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
  /// Construct from a `(vocab, hidden)` dense embedding table.
  pub fn new(weight: Array) -> Self {
    Self {
      inner: EmbeddingInner::Dense(weight),
    }
  }

  /// Construct a **quantized** embedding from the checkpoint's packed
  /// `(weight, scales, biases)` triple and the scheme parameters — the
  /// embedding analogue of [`crate::nn::QuantizedLinear::from_parts`].
  ///
  /// The triple is validated at LOAD time by the shared
  /// [`validate_quantized_triple`](crate::nn::quantized::validate_quantized_triple)
  /// — the ONE place mlx's construct-relevant contract is mirrored — so a
  /// malformed quantized embedding (a `biases` arity / shape / dtype that
  /// disagrees with the `mode` and `scales`, a non-`uint32` weight, or a scales
  /// trailing dim that does not recover `hidden`) is a typed error here rather
  /// than an opaque mlx-c rejection on the first [`Self::forward`] `dequantize`
  /// or [`Self::as_linear`] `quantized_matmul`. The embedding has NO separate
  /// dense output bias — its `biases` IS the per-group affine bias — so the
  /// linear-only dense-bias check from `from_parts` does not apply.
  ///
  /// Reads only `shape()` / `dtype()` metadata (no materialization / eval).
  ///
  /// # Errors
  /// - [`Error::RankMismatch`](crate::error::Error::RankMismatch) /
  ///   [`Error::ShapePairMismatch`](crate::error::Error::ShapePairMismatch) /
  ///   [`Error::InvariantViolation`] /
  ///   [`Error::UnknownEnumValue`](crate::error::Error::UnknownEnumValue) /
  ///   [`Error::OutOfRange`](crate::error::Error::OutOfRange) /
  ///   [`Error::UnsupportedDtype`](crate::error::Error::UnsupportedDtype) for a
  ///   malformed triple, exactly as documented on
  ///   [`validate_quantized_triple`](crate::nn::quantized::validate_quantized_triple).
  pub fn quantized(
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: impl Into<String>,
  ) -> Result<Self> {
    let mode = mode.into();
    crate::nn::quantized::validate_quantized_triple(
      "lfm2::Embedding::quantized",
      &weight,
      &scales,
      biases.as_ref(),
      group_size,
      bits,
      &mode,
    )?;
    Ok(Self {
      inner: EmbeddingInner::Quantized(QuantizedEmbedding {
        weight,
        scales,
        biases,
        group_size,
        bits,
        mode,
      }),
    })
  }

  /// Build a dense-or-quantized token embedding from the checkpoint weight map
  /// by the presence of the `<prefix>.scales` sibling — the weight-map
  /// analogue of mlx-lm's `f"{p}.scales" in weights` predicate, which quantizes
  /// `nn.Embedding` alongside `nn.Linear`.
  ///
  /// - **Quantized path** (`<prefix>.scales` present): pops the packed
  ///   `<prefix>.weight` (`uint32`), `<prefix>.scales`, and the optional
  ///   per-group affine `<prefix>.biases`, and builds the quantized variant
  ///   with the per-layer-resolved `(group_size, bits, mode)` from `quant`. A
  ///   present `.scales` with `quant == None` is a config/checkpoint
  ///   inconsistency surfaced as a typed [`Error::InvariantViolation`].
  /// - **Dense path** (no `<prefix>.scales`): pops `<prefix>.weight` and builds
  ///   the dense variant.
  ///
  /// Consumes the tensors from `weights` by key (the same key-remap-free
  /// consume the LFM2 loader uses).
  ///
  /// # Errors
  /// - [`Error::MissingKey`] if `<prefix>.weight` / `<prefix>.scales` is absent;
  /// - [`Error::InvariantViolation`] if `<prefix>.scales` is present but `quant`
  ///   resolved no scheme parameters;
  /// - propagates [`Self::quantized`]'s structural validation of the triple.
  pub fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    quant: Option<(i32, i32, &str)>,
  ) -> Result<Self> {
    let scales_key = format!("{prefix}{SCALES_SUFFIX}");
    if weights.contains_key(&scales_key) {
      let Some((group_size, bits, mode)) = quant else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "lfm2::Embedding::from_weights: checkpoint carries a `.scales` sibling for the token embedding but no quantization config resolved scheme parameters",
          "a quantized embedding requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      let scales = take_required(weights, prefix, SCALES_SUFFIX)?;
      let biases = weights.remove(&format!("{prefix}{BIASES_SUFFIX}"));
      Self::quantized(weight, scales, biases, group_size, bits, mode)
    } else {
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      Ok(Self::new(weight))
    }
  }

  /// Gather embedding rows: `weight[ids]` along the vocab axis (axis 0),
  /// mirroring `mlx.nn.Embedding.__call__`'s `self.weight[x]` (dense) and
  /// `mlx.nn.QuantizedEmbedding.__call__`'s
  /// `dequantize(weight[x], scales[x], biases[x], …)` (quantized). `ids` is an
  /// integer [`Array`] of shape `S`; the result is `S ++ (hidden,)`.
  ///
  /// # Errors
  /// Propagates the gather (`take_axis`) / dequantize op errors.
  pub fn forward(&self, ids: &Array) -> Result<Array> {
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

  /// Weight-tied linear projection (the LFM2 logit head, `lfm2.py:296`),
  /// mirroring `mlx.nn.Embedding.as_linear` `x @ weight.T` (dense) and
  /// `mlx.nn.QuantizedEmbedding.as_linear`'s
  /// `quantized_matmul(x, weight, scales, biases, transpose=True, …)`
  /// (quantized). `x` is `(..., hidden)`; the result is `(..., vocab)`.
  ///
  /// Uses the SAME table as [`Self::forward`], so the tie holds bit-identically
  /// in both the dense and quantized cases.
  ///
  /// # Errors
  /// Propagates the transpose / matmul / quantized-matmul op errors.
  pub fn as_linear(&self, x: &Array) -> Result<Array> {
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
  pub fn is_quantized(&self) -> bool {
    matches!(self.inner, EmbeddingInner::Quantized(_))
  }
}

/// Pop a required `<prefix><suffix>` tensor from `weights`, or return a typed
/// [`Error::MissingKey`] naming the absent key.
fn take_required(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
  suffix: &str,
) -> Result<Array> {
  let key = format!("{prefix}{suffix}");
  weights.remove(&key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "lfm2 weight map: required weight not found in checkpoint",
      format_smolstr!("{key}"),
    ))
  })
}

/// Resolve the optional per-layer quantization config from an LFM2
/// `config.json` body — the LFM2-side analogue of
/// [`crate::lm::quant::parse_quantization`] that ALSO accepts the HuggingFace
/// `quantization_config` key some post-quantize artifacts emit.
///
/// mlx-lm's loader reads the top-level `"quantization"` block; HF-format
/// quantized checkpoints (and some mlx-community artifacts) instead carry the
/// same payload under `"quantization_config"`. This tries `"quantization"`
/// first (preferring it when non-`null`) and falls back to
/// `"quantization_config"`, mirroring the audio loader's key fallback
/// ([`crate::audio::load::apply_quantization`]) without the audio-only
/// `group_size` default (swift's `Quantization` requires `group_size`, so a
/// block missing it is a recoverable error, not a silent default).
///
/// `Ok(None)` ⇒ the checkpoint is dense (neither key present, or both
/// explicitly `null`). `Ok(Some(plq))` ⇒ the global default
/// (`group_size` / `bits` / `mode`) plus any per-layer overrides.
///
/// # Errors
/// - [`Error::Parse`](crate::error::Error::Parse) if the config is not valid
///   JSON, the chosen block fails to deserialize as a
///   [`PerLayerQuantization`], or the block is missing a required key
///   (`group_size` / `bits`);
/// - [`Error::InvariantViolation`] if the chosen block is present but is not a
///   JSON object.
pub fn resolve_quantization(config_json: &str) -> Result<Option<PerLayerQuantization>> {
  use crate::error::ParsePayload;
  use serde_json::Value;

  let value: Value = serde_json::from_str(config_json).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "lfm2 resolve_quantization: config",
      "JSON",
      e,
    ))
  })?;

  // Prefer top-level `"quantization"` if non-null, else the HF
  // `"quantization_config"` artifact key if non-null, else dense (no-op).
  let block = match value.get("quantization") {
    Some(b) if !b.is_null() => b,
    _ => match value.get("quantization_config") {
      Some(b) if !b.is_null() => b,
      _ => return Ok(None),
    },
  };
  if !block.is_object() {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "lfm2 resolve_quantization: quantization block",
      "must be a JSON object",
    )));
  }
  let plq: PerLayerQuantization = serde_json::from_value(block.clone()).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "lfm2 resolve_quantization: quantization block",
      "JSON",
      e,
    ))
  })?;
  Ok(Some(plq))
}
