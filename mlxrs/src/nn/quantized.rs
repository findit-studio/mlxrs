//! Shared dense + quantized linear layers reusable by every model family
//! (`lm` / `vlm` / `audio` / `embeddings`).
//!
//! Three types, mirroring `mlx.nn`:
//!
//! - [`Linear`] — a dense `y = x @ weightᵀ (+ bias)` projection
//!   (`mlx.nn.Linear`: weight stored `(out_features, in_features)`, the
//!   forward transposes it).
//! - [`QuantizedLinear`] — the quantized equivalent
//!   (`mlx.nn.QuantizedLinear`, `mlx/python/mlx/nn/layers/quantized.py`):
//!   the packed `uint32` `weight`, per-group `scales`, optional per-group
//!   `biases`, the `group_size` / `bits` / `mode` scheme parameters, and an
//!   optional dense `bias`. Its forward is exactly mlx's `__call__`:
//!   `quantized_matmul(x, weight, scales, biases, transpose=true, group_size,
//!   bits, mode)` then `+ bias` if present.
//! - [`MaybeQuantizedLinear`] — the quantize-aware abstraction a model uses
//!   in place of a bare [`Linear`]: an enum that is either
//!   [`Linear`](MaybeQuantizedLinear::Dense) or
//!   [`QuantizedLinear`](MaybeQuantizedLinear::Quantized), with a unified
//!   [`forward`](MaybeQuantizedLinear::forward). The
//!   [`MaybeQuantizedLinear::from_weights`] helper builds the quantized
//!   variant when the checkpoint carries the sibling `<prefix>.scales` /
//!   `<prefix>.biases` tensors for the weight, and the dense variant
//!   otherwise — the weight-map analogue of mlx-audio's whisper
//!   `class_predicate` (`isinstance(m, (nn.Linear, nn.Embedding)) and
//!   f"{p}.scales" in weights`,
//!   `mlx_audio/stt/models/whisper/whisper.py:674-676`).
//!
//! ## Adopting this in a model
//!
//! A model that previously stored a bare dense `Linear` swaps it for a
//! [`MaybeQuantizedLinear`] and builds it with
//! [`MaybeQuantizedLinear::from_weights`], passing the checkpoint weight map,
//! the layer's `<prefix>`, and (for a quantized checkpoint) the resolved
//! `(group_size, bits, mode)` from the parsed
//! [`crate::lm::quant::PerLayerQuantization`]. The same `forward(&self, x)`
//! call site works for both the dense and quantized cases, so the rest of the
//! model is unchanged. This is the path Whisper takes (see
//! [`crate::audio::stt::models::whisper`]); siglip / egemma / qwen adopt it
//! identically.
//!
//! ## Library contract
//!
//! Construction is `Result`-fallible. The quantized `(weight, scales, biases)`
//! triple is validated by a single shared `validate_quantized_triple` — the
//! ONE place mlx's construct-relevant contract is mirrored, so every quantized
//! constructor (this module's [`QuantizedLinear::from_parts`] and the Whisper
//! `Embedding`'s quantized constructor) checks an identical contract with no
//! per-constructor drift: `group_size > 0`, `bits > 0`, `weight` is rank-2
//! `uint32`, `scales` rank/leading-dims match `weight` and its trailing dim
//! recovers the same logical width
//! (`weight.shape(-1) * 32 / bits == scales.shape(-1) * group_size`), the
//! `biases` arity AND its shape match the resolved `mode` (`affine` requires
//! `biases`; the `fp` modes forbid them), and the `fp`-mode scale **dtype** is
//! `uint8`. [`QuantizedLinear::from_parts`] adds the linear-only check that the
//! optional dense `bias` is exactly `(out_features,)`. The `affine` scale/bias
//! dtype rule (mlx's `issubdtype(result_type(scales, biases), floating)`) is the
//! one piece left to mlx-c at the
//! [`crate::ops::quantized::quantized_matmul`] call site: it depends on mlx's
//! full JAX-style promotion table (which mlx-c exposes no symbol to call) and is
//! enforced there as a typed error, so it is sound to defer. Per-mode absolute
//! `(group_size, bits)` value tables (`bits ∈ {2,3,4,5,6,8}` for `affine`; the
//! `mxfp*` / `nvfp4` pairings — `mlx/mlx/ops.cpp:4745-4750,4808-4823`) that mlx
//! ITSELF defers past construction are **left to mlx-c** at the same call site,
//! matching the faithful-thin-wrapper discipline the rest of the crate follows.
//! No implicit eval (only `shape()` / `dtype()` metadata is read at
//! construction); no panic on valid input; no UB.

use std::collections::HashMap;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    Error, InvariantViolationPayload, OutOfRangePayload, RankMismatchPayload, Result,
    ShapePairMismatchPayload, UnknownEnumValuePayload, UnsupportedDtypePayload,
  },
  ops::quantized,
};
use smol_str::format_smolstr;

/// The mlx quantization-mode tags (`mlx-swift`'s `QuantizationMode`,
/// `mlx/python/mlx/nn/layers/quantized.py`). The crate-wide source of these
/// is [`crate::lm::quant::QuantMode`]; this module stores the `mode` as the
/// wire-format string (mirroring [`crate::lm::nn::switch::QuantizedSwitchLinear`]),
/// so it stays usable from `embeddings`, which does not enable the `lm`
/// feature that gates `QuantMode`.
const MODE_AFFINE: &str = "affine";
const MODE_MXFP4: &str = "mxfp4";
const MODE_MXFP8: &str = "mxfp8";
const MODE_NVFP4: &str = "nvfp4";
/// The recognized mode tags, for the unknown-mode rejection diagnostic.
const KNOWN_MODES: &[&str] = &[MODE_AFFINE, MODE_MXFP4, MODE_MXFP8, MODE_NVFP4];

/// The single, complete validation of an mlx quantized `(weight, scales,
/// biases)` triple against mlx's construct-relevant contract — shared by every
/// quantized layer's constructor ([`QuantizedLinear::from_parts`] and the
/// Whisper `Embedding`'s quantized constructor) so the contract lives in ONE
/// place with no per-constructor drift.
///
/// This mirrors, in one pass, exactly the preconditions mlx enforces on the
/// triple before a [`quantized_matmul`](crate::ops::quantized::quantized_matmul)
/// / [`dequantize`](crate::ops::quantized::dequantize). Every one of these is a
/// catchable `std::invalid_argument` in mlx (surfaced through mlx-c as an error
/// code, not an assert / process-abort), so none is a soundness requirement;
/// they are mirrored here purely for **fail-fast clarity on a malformed model
/// file** — a clean typed error at load instead of an opaque mlx-c rejection on
/// the first forward. The mlx sources, in evaluation order across
/// `validate_mode_with_type` + `validate_quantized_input`:
///
/// - `bits > 0` and `group_size > 0` — `mlx/ops.cpp:5185-5194` (`dequantize`);
///   checked here FIRST, before the width recovery divides by `bits` /
///   multiplies by `group_size`, so a non-positive scheme parameter is a typed
///   error, never a division trap.
/// - `weight` rank == 2 (`weight.ndim() >= 2`, specialized to the rank-2
///   linear / embedding weight; `mlx/ops.cpp:5199-5204`).
/// - `weight.dtype() == uint32` — `mlx/ops.cpp:82-87`.
/// - `scales` rank == `weight` rank with a matching leading (batch) dim
///   (`std::equal(w.shape().begin(), w.shape().end()-2, scales.shape().begin())`
///   — `mlx/ops.cpp:97-105`).
/// - width identity `w.shape(-1) * 32 / bits == scales.shape(-1) * group_size`
///   — `mlx/ops.cpp:107-114`.
/// - `biases` (when present) shares `scales`' shape — `mlx/ops.cpp:89-95`.
/// - mode is one of `affine` / `mxfp4` / `mxfp8` / `nvfp4`
///   (`string_to_quantization_mode`, `mlx/primitives.cpp:3436-3450`).
/// - per-mode arity + scale dtype — `mlx/ops.cpp:4411-4441`
///   (`validate_mode_with_type`):
///   - `affine`: `biases` REQUIRED (a presence/arity check, mirrored here). The
///     affine scale/bias dtype VALIDITY — mlx's
///     `issubdtype(result_type(scales, biases), floating)` — is NOT mirrored at
///     construction: it is enforced by mlx-c at the first quantized op (see the
///     `MODE_AFFINE` arm below for why).
///   - `mxfp4` / `mxfp8` / `nvfp4`: `scales.dtype() == uint8` (an unambiguous
///     fixed-dtype equality, mirrored here), and `biases` FORBIDDEN.
///
/// Per-mode absolute `(group_size, bits)` value tables mlx ITSELF defers past
/// construction (`quantization_params_from_mode`) likewise stay deferred to
/// mlx-c.
///
/// Reads only `shape()` / `dtype()` metadata — no materialization / eval — so it
/// is bounded regardless of the declared dims. `context` prefixes every
/// diagnostic with the caller's name.
///
/// # Errors
/// - [`Error::OutOfRange`] if `bits` / `group_size` are not positive;
/// - [`Error::RankMismatch`] if `weight` / `scales` / `biases` have the wrong
///   rank;
/// - [`Error::InvariantViolation`] if `weight` is not `uint32`, or the per-mode
///   bias arity is violated;
/// - [`Error::ShapePairMismatch`] if the scales leading / trailing dim or the
///   `biases` shape disagree;
/// - [`Error::UnknownEnumValue`] for an unrecognized `mode`;
/// - [`Error::UnsupportedDtype`] if a `fp`-mode (`mxfp4` / `mxfp8` / `nvfp4`)
///   `scales` is not `uint8`. (The `affine` scale/bias dtype rule is deferred to
///   mlx-c at op-time, so it is NOT a construction error here.)
pub(crate) fn validate_quantized_triple(
  context: &'static str,
  weight: &Array,
  scales: &Array,
  biases: Option<&Array>,
  group_size: i32,
  bits: i32,
  mode: &str,
) -> Result<()> {
  // Basic non-zero positivity on `bits` / `group_size`. Per-mode value tables
  // are DEFERRED to mlx-c. Checked FIRST — before the width recovery below
  // divides by `bits` / multiplies by `group_size` — so a malformed
  // non-positive scheme parameter is a typed error, never a division trap.
  if bits <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "bits must be > 0 (per-mode value tables validated by mlx-c)",
      format_smolstr!("{bits}"),
    )));
  }
  if group_size <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "group_size must be > 0 (per-mode value tables validated by mlx-c)",
      format_smolstr!("{group_size}"),
    )));
  }

  // `weight` must be the rank-2 packed quantized matrix.
  let w_shape = weight.shape();
  if w_shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      w_shape.len() as u32,
      w_shape.to_vec(),
    )));
  }
  // Packed-quantization dtype signal: mlx packs the quantized weight into
  // `uint32` words and rejects any non-`uint32` quantized weight
  // (`mlx/ops.cpp:82-87`). A rank-2 dense `f32` weight with otherwise-matching
  // scales would pass every shape check and only fail deep in the FFI on the
  // first forward — reject it here instead.
  if weight.dtype()? != Dtype::U32 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      context,
      "weight must be `uint32` (the mlx-quantized-weight dtype; quantized ops reject non-`uint32` weights)",
    )));
  }
  let out_features = w_shape[0];

  // `scales` structural invariants: rank == weight rank (2); leading dim must
  // match; and the per-group count must recover the SAME logical input width as
  // the packed weight.
  let s_shape = scales.shape();
  if s_shape.len() != w_shape.len() {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      s_shape.len() as u32,
      s_shape.to_vec(),
    )));
  }
  if s_shape[0] != out_features {
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      context,
      vec![out_features],
      vec![s_shape[0]],
    )));
  }
  // Logical input width recovered from the packed weight is mlx's
  // `w_inner_dims = w.shape(-1) * 32 / bits`; from `scales` it is
  // `scales.shape(-1) * group_size`. mlx requires these to agree
  // (`mlx/ops.cpp:107-114`); a triple with a correct rank and leading dim but a
  // wrong scales trailing dim would otherwise pass every shape check here and
  // only fail deep inside the quantized op on the first forward. Compare in
  // `i64` so a corrupt huge packed width cannot overflow the recovery.
  let weight_in = (w_shape[1] as i64) * 32 / i64::from(bits);
  let scales_in = (s_shape[1] as i64) * i64::from(group_size);
  if weight_in != scales_in {
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      context,
      vec![weight_in.max(0) as usize],
      vec![scales_in.max(0) as usize],
    )));
  }

  // `biases`, when present, shares the `(out_features, n_groups)` layout with
  // `scales` (`affine_quantize` writes both with the same shape;
  // `mlx/ops.cpp:89-95`). Split the check so a divergent RANK surfaces as
  // `RankMismatch` rather than collapsing into the same-rank `ShapePairMismatch`.
  if let Some(b) = biases {
    let b_shape = b.shape();
    if b_shape.len() != s_shape.len() {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        context,
        b_shape.len() as u32,
        b_shape.to_vec(),
      )));
    }
    if b_shape != s_shape {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        context,
        s_shape.to_vec(),
        b_shape.to_vec(),
      )));
    }
  }

  // Mode parse + per-mode arity and DTYPE (`validate_mode_with_type`,
  // `mlx/ops.cpp:4411-4441`). Unknown modes are rejected so a typo never
  // reaches mlx-c with an unfamiliar tag.
  match mode {
    MODE_AFFINE => {
      // affine: biases REQUIRED. This is a presence/arity check (mlx's
      // `if (!biases) throw`), so it is mirrored here for fail-fast clarity.
      if biases.is_none() {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          context,
          "`affine` mode requires per-group biases (mlx `affine_quantize` always writes {w_q, scales, biases})",
        )));
      }
      // The affine scale/bias dtype VALIDITY — mlx's
      // `issubdtype(result_type(scales, biases), floating)` — is intentionally
      // NOT mirrored here, and is enforced by mlx-c at the first quantized op
      // (`quantized_matmul` / `dequantize`) as a typed error (sound by
      // deferral: it is a catchable `std::invalid_argument`, never an abort/UB).
      // The rule depends on mlx's full JAX-style `promote_types` promotion table
      // (e.g. `uint64` combined with a signed integer promotes to `float32`,
      // which is then accepted), and mlx-c exposes no promotion symbol to call.
      // Any boolean approximation here (e.g. "at least one operand floating") is
      // STRICTER than mlx on those promotion edges, so faithfully replicating
      // the table at construction is out of scope and version-fragile; the op
      // performs the exact check on the same `scales` / `biases`.
    }
    MODE_MXFP4 | MODE_MXFP8 | MODE_NVFP4 => {
      // fp modes: scale-only (`fp_quantize` writes {w_q, scales}, no biases),
      // and `scales.dtype()` MUST be `uint8` (the packed fp scale layout). The
      // arity check precedes the dtype check for the same reason as above.
      if biases.is_some() {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          context,
          "mxfp4 / mxfp8 / nvfp4 mode is scale-only (mlx `fp_quantize` writes {w_q, scales} with no biases); got a stale `biases`",
        )));
      }
      let s_dtype = scales.dtype()?;
      if s_dtype != Dtype::U8 {
        return Err(Error::UnsupportedDtype(UnsupportedDtypePayload::new(
          context,
          s_dtype,
          &[Dtype::U8],
        )));
      }
    }
    other => {
      return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
        context,
        other.to_string(),
        KNOWN_MODES,
      )));
    }
  }

  Ok(())
}

/// A dense linear projection `y = x @ weightᵀ (+ bias)`.
///
/// Mirrors `mlx.nn.Linear`: `weight` is stored `(out_features, in_features)`
/// and [`forward`](Self::forward) transposes it, so `y = x @ weight.T +
/// bias`. `bias` is optional (mlx's `Linear(bias=False)`).
#[derive(Debug)]
pub struct Linear {
  /// `(out_features, in_features)` weight (the `mlx.nn.Linear` layout).
  weight: Array,
  /// Optional `(out_features,)` bias.
  bias: Option<Array>,
}

impl Linear {
  /// Construct from a `(out_features, in_features)` `weight` and an optional
  /// `(out_features,)` `bias`.
  ///
  /// No shape validation here (it is a thin holder mirroring `mlx.nn.Linear`,
  /// which trusts its constructed parameters); a model that needs to pin the
  /// weight shape against a config does so at its own loader (e.g. Whisper's
  /// `Builder::take_shaped`) before calling this.
  pub fn new(weight: Array, bias: Option<Array>) -> Self {
    Self { weight, bias }
  }

  /// `y = x @ weightᵀ (+ bias)`. `x` is `(..., in_features)`; the result is
  /// `(..., out_features)`.
  ///
  /// # Errors
  /// Propagates the transpose / matmul / add op errors.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let wt = self.weight.transpose()?;
    let y = x.matmul(&wt)?;
    match &self.bias {
      Some(b) => y.add(b),
      None => Ok(y),
    }
  }

  /// The `(out_features, in_features)` weight.
  #[inline(always)]
  pub fn weight_ref(&self) -> &Array {
    &self.weight
  }

  /// The optional `(out_features,)` bias.
  #[inline(always)]
  pub fn bias(&self) -> Option<&Array> {
    self.bias.as_ref()
  }
}

/// A quantized linear projection — the quantized equivalent of [`Linear`]
/// (`mlx.nn.QuantizedLinear`, `mlx/python/mlx/nn/layers/quantized.py:198-279`).
///
/// Holds the packed `uint32` `weight` `(out_features, packed_in_features)`,
/// the per-group `scales`, the optional per-group affine `biases`, the
/// `group_size` / `bits` / `mode` quantization-scheme parameters, and an
/// optional dense output `bias` `(out_features,)`. The forward computes
/// `quantized_matmul(x, weight, scales, biases, transpose=true, group_size,
/// bits, mode)` then adds the dense `bias` (if present) — bit-for-bit with
/// mlx's `QuantizedLinear.__call__`.
///
/// All fields are private (read via the accessors) so the scheme parameters
/// — which MUST match what produced `weight` / `scales` — cannot be silently
/// mutated after the construction-time consistency checks, mirroring
/// [`crate::lm::nn::switch::QuantizedSwitchLinear`].
#[derive(Debug)]
pub struct QuantizedLinear {
  /// Packed quantized weight `(out_features, packed_in_features)` (`uint32`).
  weight: Array,
  /// Per-group `scales` (paired with `weight`).
  scales: Array,
  /// Per-group affine `biases` (the `affine`-mode addend; `None` for the
  /// scale-only `fp` modes). Distinct from the dense [`Self::bias`].
  quant_biases: Option<Array>,
  /// The optional dense output bias `(out_features,)` — the layer's
  /// `Linear.bias`, NOT the per-group quantization `biases`.
  bias: Option<Array>,
  /// Quantization group size (must match what produced `weight` / `scales`).
  group_size: i32,
  /// Quantization bit depth.
  bits: i32,
  /// Quantization mode tag (`"affine"` / `"mxfp4"` / `"mxfp8"` / `"nvfp4"`).
  mode: String,
}

impl QuantizedLinear {
  /// Construct from the checkpoint's already-quantized arrays.
  ///
  /// Mirrors the structural invariants of
  /// [`crate::lm::nn::switch::QuantizedSwitchLinear::from_parts`], adapted to
  /// the rank-2 `(out_features, packed_in_features)` linear weight (vs the
  /// rank-3 per-expert switch stack). Per-mode absolute `(group_size, bits)`
  /// value tables (`bits ∈ {2,3,4,5,6,8}` for `affine`; the `mxfp*` / `nvfp4`
  /// pairings — `mlx/mlx/ops.cpp:4745-4750,4808-4823`) are intentionally
  /// **left to mlx-c** at the
  /// [`quantized_matmul`](crate::ops::quantized::quantized_matmul) call site
  /// (mlx itself defers them past construction), matching the
  /// faithful-thin-wrapper discipline.
  ///
  /// The `(weight, scales, quant_biases, group_size, bits, mode)` triple is
  /// validated by the shared `validate_quantized_triple` (the ONE place mlx's
  /// construct-relevant contract is mirrored — rank / `uint32` weight / the
  /// scales width identity / per-mode bias arity / the `fp`-mode `uint8` scale
  /// dtype; the `affine` scale/bias dtype rule is deferred to mlx-c at op-time),
  /// so this constructor and the Whisper `Embedding`'s quantized constructor
  /// cannot drift apart. On top of the shared triple contract, this adds the
  /// linear-only dense-`bias` check:
  ///
  /// - the dense `bias`, if `Some`, is rank-1 with length exactly
  ///   `out_features` (the quantized weight's logical output dim) — a stray
  ///   `(1,)` bias that would broadcast across every channel is rejected. (This
  ///   is the layer's `Linear.bias`, distinct from the per-group quantization
  ///   `quant_biases`, and is NOT part of mlx's quantized-triple contract; the
  ///   embedding has no separate dense bias, so this check is local here.)
  ///
  /// Shape mismatches surface as typed
  /// [`Error::RankMismatch`] / [`Error::ShapePairMismatch`]; mode-arity /
  /// unknown-mode / zero-param / dtype failures as
  /// [`Error::InvariantViolation`] / [`Error::UnknownEnumValue`] /
  /// [`Error::OutOfRange`] / [`Error::UnsupportedDtype`]. Does not evaluate
  /// (lazy; only `shape()` / `dtype()` metadata is read).
  #[allow(clippy::too_many_arguments)]
  pub fn from_parts(
    weight: Array,
    scales: Array,
    quant_biases: Option<Array>,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: impl Into<String>,
  ) -> Result<Self> {
    let mode = mode.into();

    // The mlx quantized-triple contract — rank / `uint32` weight / the scales
    // leading-dim + width identity / per-mode bias arity / `fp`-mode `uint8`
    // scale dtype — in ONE shared place (the `affine` scale/bias dtype rule is
    // deferred to mlx-c at op-time; see `validate_quantized_triple`).
    validate_quantized_triple(
      "QuantizedLinear::from_parts",
      &weight,
      &scales,
      quant_biases.as_ref(),
      group_size,
      bits,
      &mode,
    )?;

    // The dense output bias, when present, is a rank-1 `(out_features,)`
    // vector. mlx broadcasts it against the `(..., out_features)` matmul
    // output; a higher-rank bias is a malformed checkpoint, and so is a rank-1
    // bias whose length is not exactly `out_features` — a stray `(1,)` bias
    // would otherwise broadcast silently across every output channel. This is
    // the layer's `Linear.bias` (distinct from the per-group `quant_biases`)
    // and is NOT part of mlx's quantized-triple contract, so it stays local to
    // the linear constructor.
    if let Some(b) = &bias {
      let out_features = weight.shape()[0];
      let b_shape = b.shape();
      if b_shape.len() != 1 {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "QuantizedLinear::from_parts: bias must be rank-1 (out_features,)",
          b_shape.len() as u32,
          b_shape.to_vec(),
        )));
      }
      if b_shape[0] != out_features {
        return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
          "QuantizedLinear::from_parts: bias length must equal out_features (the quantized weight's logical output dim)",
          vec![out_features],
          b_shape.to_vec(),
        )));
      }
    }

    Ok(Self {
      weight,
      scales,
      quant_biases,
      bias,
      group_size,
      bits,
      mode,
    })
  }

  /// `x = quantized_matmul(x, weight, scales, biases, transpose=true,
  /// group_size, bits, mode)`, then `+ bias` if a dense bias is present —
  /// `mlx.nn.QuantizedLinear.__call__`
  /// (`mlx/python/mlx/nn/layers/quantized.py:266-278`).
  ///
  /// `x` is `(..., in_features)`; the result is `(..., out_features)`.
  ///
  /// # Errors
  /// Propagates the [`quantized_matmul`](crate::ops::quantized::quantized_matmul)
  /// / add op errors (including mlx-c's `validate_quantized_input` rejections
  /// for an incompatible `group_size` / `bits` / `mode`).
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let y = quantized::quantized_matmul(
      x,
      &self.weight,
      &self.scales,
      self.quant_biases.as_ref(),
      true,
      self.group_size,
      self.bits,
      &self.mode,
    )?;
    match &self.bias {
      Some(b) => y.add(b),
      None => Ok(y),
    }
  }

  /// The packed quantized weight `(out_features, packed_in_features)`.
  #[inline(always)]
  pub fn weight_ref(&self) -> &Array {
    &self.weight
  }

  /// The per-group `scales`.
  #[inline(always)]
  pub fn scales_ref(&self) -> &Array {
    &self.scales
  }

  /// The per-group affine `biases` (`None` for scale-only `fp` modes).
  #[inline(always)]
  pub fn quant_biases(&self) -> Option<&Array> {
    self.quant_biases.as_ref()
  }

  /// The optional dense output bias `(out_features,)`.
  #[inline(always)]
  pub fn bias(&self) -> Option<&Array> {
    self.bias.as_ref()
  }

  /// The quantization group size.
  #[inline(always)]
  pub fn group_size(&self) -> i32 {
    self.group_size
  }

  /// The quantization bit depth.
  #[inline(always)]
  pub fn bits(&self) -> i32 {
    self.bits
  }

  /// The quantization mode tag.
  #[inline(always)]
  pub fn mode(&self) -> &str {
    &self.mode
  }
}

/// A quantize-aware linear layer: either a dense [`Linear`] or a
/// [`QuantizedLinear`], dispatched at the [`forward`](Self::forward) call
/// site.
///
/// This is the abstraction a model uses in place of a bare [`Linear`] so the
/// same construction + forward code path serves both dense and quantized
/// checkpoints. [`from_weights`](Self::from_weights) picks the quantized
/// variant when the checkpoint carries the sibling `<prefix>.scales` (and,
/// for `affine`, `<prefix>.biases`) tensors, and the dense variant otherwise —
/// the weight-map analogue of mlx-audio's whisper `class_predicate`
/// (`isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights`,
/// `mlx_audio/stt/models/whisper/whisper.py:674-676`).
#[derive(Debug)]
pub enum MaybeQuantizedLinear {
  /// A dense `mlx.nn.Linear`.
  Dense(Linear),
  /// A quantized `mlx.nn.QuantizedLinear`.
  Quantized(QuantizedLinear),
}

/// The `<prefix>.scales` / `<prefix>.biases` sibling suffixes the
/// mlx-quantized layout writes next to `<prefix>.weight`
/// (`mlx/python/mlx/nn/layers/quantized.py`). A `<prefix>.scales` sibling is
/// the load-bearing "this layer is quantized" signal mlx-audio / mlx-lm's
/// loaders key on.
const SCALES_SUFFIX: &str = ".scales";
const BIASES_SUFFIX: &str = ".biases";
const WEIGHT_SUFFIX: &str = ".weight";
const BIAS_SUFFIX: &str = ".bias";

impl MaybeQuantizedLinear {
  /// `true` if this is the quantized variant.
  #[inline(always)]
  pub fn is_quantized(&self) -> bool {
    matches!(self, MaybeQuantizedLinear::Quantized(_))
  }

  /// The **logical** `(out_features, in_features)` of the projection — the
  /// dense weight layout (`mlx.nn.Linear` stores `weight` as
  /// `(out_features, in_features)`), recovered identically for both arms.
  ///
  /// A model that pins its projection to a config-derived `(out, in)` uses this
  /// for both arms so the quantized arm is shape-checked at load (a packed
  /// weight whose dequantized width / row count disagrees with the config would
  /// otherwise only mis-project at the first forward) — the linear analogue of
  /// [`MaybeQuantizedEmbedding::logical_shape`].
  ///
  /// - **Dense**: the weight's own `(out_features, in_features)` (the dense
  ///   weight is the logical weight verbatim).
  /// - **Quantized**: `out_features` is the packed weight's row count (the
  ///   leading axis, validated equal to the `scales` row count at construction)
  ///   and `in_features` is `scales.shape(-1) * group_size` — the same logical
  ///   input width `validate_quantized_triple` recovers and pins against the
  ///   packed weight, so no dequantization is needed.
  ///
  /// Reads only `shape()` metadata (no materialization / eval).
  ///
  /// # Errors
  /// - [`Error::RankMismatch`] if the dense weight is not rank-2 (the quantized
  ///   arm's rank-2 layout is guaranteed by its construction-time validation);
  /// - [`Error::OutOfRange`] if a dimension exceeds `i32::MAX` (a corrupt huge
  ///   weight; the recovery is computed in `i64` so it cannot overflow first).
  pub fn logical_shape(&self) -> Result<(i32, i32)> {
    let context = "MaybeQuantizedLinear::logical_shape";
    let (out, in_features): (i64, i64) = match self {
      MaybeQuantizedLinear::Dense(l) => {
        let shape = l.weight_ref().shape();
        if shape.len() != 2 {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "MaybeQuantizedLinear::logical_shape: dense weight must be rank-2 (out_features, in_features)",
            shape.len() as u32,
            shape.to_vec(),
          )));
        }
        (shape[0] as i64, shape[1] as i64)
      }
      MaybeQuantizedLinear::Quantized(q) => {
        // The triple was validated rank-2 at construction; `out_features` is the
        // packed weight's leading axis and the logical input width is
        // `scales.shape(-1) * group_size` (mlx's per-group recovery, pinned
        // against the packed weight by `validate_quantized_triple`).
        let w_rows = q.weight.shape()[0] as i64;
        let s_shape = q.scales.shape();
        let logical_in = (s_shape[1] as i64) * i64::from(q.group_size);
        (w_rows, logical_in)
      }
    };
    let to_i32 = |v: i64| -> Result<i32> {
      i32::try_from(v).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          context,
          "logical linear dimension exceeds i32::MAX",
          format_smolstr!("{v}"),
        ))
      })
    };
    Ok((to_i32(out)?, to_i32(in_features)?))
  }

  /// Run the layer: `mlx.nn.Linear.__call__` or
  /// `mlx.nn.QuantizedLinear.__call__` depending on the variant.
  ///
  /// `x` is `(..., in_features)`; the result is `(..., out_features)`.
  ///
  /// # Errors
  /// Propagates the underlying [`Linear::forward`] / [`QuantizedLinear::forward`]
  /// op errors.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    match self {
      MaybeQuantizedLinear::Dense(l) => l.forward(x),
      MaybeQuantizedLinear::Quantized(q) => q.forward(x),
    }
  }

  /// Build a quantize-aware linear from a checkpoint weight map: a
  /// [`QuantizedLinear`] when `<prefix>.scales` is present in `weights`, else
  /// a dense [`Linear`].
  ///
  /// The weight-map analogue of mlx-audio's whisper `class_predicate`
  /// (`f"{p}.scales" in weights`,
  /// `mlx_audio/stt/models/whisper/whisper.py:674-676`): the presence of a
  /// `<prefix>.scales` sibling is the signal that this layer was
  /// pre-quantized in the checkpoint.
  ///
  /// - **Quantized path** (`<prefix>.scales` present): pops `<prefix>.weight`
  ///   (the packed `uint32` matrix), `<prefix>.scales`, the optional
  ///   `<prefix>.biases` (the per-group affine bias — present iff
  ///   `mode == "affine"`), and the optional dense `<prefix>.bias`, and builds
  ///   a [`QuantizedLinear`] with the resolved `(group_size, bits, mode)`. The
  ///   structural consistency of the triple is validated by
  ///   [`QuantizedLinear::from_parts`].
  /// - **Dense path** (no `<prefix>.scales`): pops `<prefix>.weight` and the
  ///   optional `<prefix>.bias`, and builds a [`Linear`].
  ///
  /// `quant` carries the resolved scheme parameters `(group_size, bits,
  /// mode)` for this layer — the caller resolves them from the parsed
  /// [`crate::lm::quant::PerLayerQuantization`] (e.g. via
  /// [`crate::lm::quant::PerLayerQuantization::quantization_for`]). It is only
  /// consulted on the quantized path; a dense checkpoint passes `None` (and a
  /// quantized checkpoint that nonetheless lacks the `<prefix>.scales` sibling
  /// for this layer still loads dense, matching the `class_predicate` gate).
  ///
  /// A `<prefix>.scales` present but `quant == None` is a checkpoint /
  /// config inconsistency (the weights say quantized, the config says dense)
  /// and returns a typed [`Error::InvariantViolation`] rather than guessing
  /// scheme parameters.
  ///
  /// This is a **key-remap-free consume**: it removes the consumed tensors
  /// from `weights` by key (so each is used once and the map frees as the
  /// model builds), mirroring the Whisper `Builder`'s pop-by-key discipline.
  /// It does no shape validation against a config — a model that wants to pin
  /// the dense-weight shape (e.g. Whisper) validates BEFORE calling this and
  /// passes the already-shaped tensors via the explicit constructors instead.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] if `<prefix>.weight` (or, on the quantized path,
  ///   `<prefix>.scales`) is absent;
  /// - [`Error::InvariantViolation`] if `<prefix>.scales` is present but
  ///   `quant` is `None`;
  /// - propagates [`QuantizedLinear::from_parts`]'s structural validation
  ///   errors.
  pub fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    quant: Option<(i32, i32, &str)>,
  ) -> Result<Self> {
    let scales_key = format!("{prefix}{SCALES_SUFFIX}");
    if weights.contains_key(&scales_key) {
      // Quantized path. The `<prefix>.scales` sibling is the
      // `class_predicate` signal; the config must carry resolvable scheme
      // params for it.
      let Some((group_size, bits, mode)) = quant else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "MaybeQuantizedLinear::from_weights: checkpoint carries a `.scales` sibling for this layer but no quantization config resolved scheme parameters",
          "a quantized layer requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      let scales = take_required(weights, prefix, SCALES_SUFFIX)?;
      // `.biases` is the per-group affine bias — present iff the mode is
      // `affine`. Pull it opportunistically; `from_parts` enforces the
      // mode/arity contract (affine requires it; fp modes forbid it).
      let quant_biases = weights.remove(&format!("{prefix}{BIASES_SUFFIX}"));
      // The optional dense output bias (`<prefix>.bias`, singular) — distinct
      // from the per-group `.biases`. A quantized Linear may still carry one.
      let bias = weights.remove(&format!("{prefix}{BIAS_SUFFIX}"));
      let q =
        QuantizedLinear::from_parts(weight, scales, quant_biases, bias, group_size, bits, mode)?;
      Ok(MaybeQuantizedLinear::Quantized(q))
    } else {
      // Dense path.
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      let bias = weights.remove(&format!("{prefix}{BIAS_SUFFIX}"));
      Ok(MaybeQuantizedLinear::Dense(Linear::new(weight, bias)))
    }
  }

  /// Build a quantize-aware linear with an **explicitly-supplied** dense output
  /// `bias`, rather than auto-consuming `<prefix>.bias` from the map.
  ///
  /// Identical to [`from_weights`](Self::from_weights) except for the dense
  /// output bias: where `from_weights` opportunistically pops any `<prefix>.bias`
  /// it finds, this takes the bias as an argument and applies it on BOTH the
  /// dense and the quantized path (`mlx`'s `QuantizedLinear.from_linear`
  /// preserves the source `Linear.bias`, so the dense-bias arity is identical
  /// whether the projection is dense or quantized). The packed `<prefix>.weight`
  /// (and, on the quantized path, `<prefix>.scales` / `<prefix>.biases`) are
  /// still consumed from `weights` by key; `<prefix>.bias` is **not** touched.
  ///
  /// This is the seam a caller uses to enforce a config-flag-gated bias contract
  /// (e.g. LFM2.5-VL's `projector_bias` / the LFM2 LM's `conv_bias`): the caller
  /// drains `<prefix>.bias` through the [`take_if`](crate::model_validation::take_if)
  /// gate (required-when-`true`, forbidden-when-`false`) and passes the result
  /// here, so a missing-required or stray-forbidden bias is a typed error rather
  /// than the silent auto-consume `from_weights` would do. Mirrors the LFM2 LM's
  /// `Linear::from_weights_with_bias`.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] if `<prefix>.weight` (or, on the quantized path,
  ///   `<prefix>.scales`) is absent;
  /// - [`Error::InvariantViolation`] if `<prefix>.scales` is present but `quant`
  ///   is `None`;
  /// - propagates [`QuantizedLinear::from_parts`]'s structural validation
  ///   (including the dense-bias arity check — the bias, if `Some`, must be
  ///   `(out_features,)`).
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
          "MaybeQuantizedLinear::from_weights_with_bias: checkpoint carries a `.scales` sibling for this layer but no quantization config resolved scheme parameters",
          "a quantized layer requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      let scales = take_required(weights, prefix, SCALES_SUFFIX)?;
      let quant_biases = weights.remove(&format!("{prefix}{BIASES_SUFFIX}"));
      let q =
        QuantizedLinear::from_parts(weight, scales, quant_biases, bias, group_size, bits, mode)?;
      Ok(MaybeQuantizedLinear::Quantized(q))
    } else {
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      Ok(MaybeQuantizedLinear::Dense(Linear::new(weight, bias)))
    }
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
    Error::MissingKey(crate::error::MissingKeyPayload::new(
      "MaybeQuantizedLinear::from_weights: required weight not found in checkpoint",
      key,
    ))
  })
}

/// A packed quantized embedding table — the `(weight, scales, biases)` triple
/// plus the `group_size` / `bits` / `mode` scheme parameters, mirroring
/// `mlx.nn.QuantizedEmbedding` (`mlx/python/mlx/nn/layers/quantized.py:99-196`).
///
/// Used when an mlx-community quantized checkpoint ships a quantized
/// `nn.Embedding` (the `class_predicate` quantizes `nn.Embedding` alongside
/// `nn.Linear`, both in mlx-audio's whisper `whisper.py:674-676` and in
/// mlx-embeddings' `convert.py` `get_class_predicate`).
///
/// All fields are private; the triple's structural consistency is validated at
/// construction by [`MaybeQuantizedEmbedding::from_parts`] via the shared
/// `validate_quantized_triple`, so the scheme parameters cannot drift apart
/// from the arrays they describe.
#[derive(Debug)]
pub struct QuantizedEmbedding {
  /// Packed `(num_embeddings, packed_dim)` quantized table (`uint32`).
  weight: Array,
  /// Per-group `scales` `(num_embeddings, n_groups)`.
  scales: Array,
  /// Per-group affine `biases` (`None` for scale-only `fp` modes).
  biases: Option<Array>,
  group_size: i32,
  bits: i32,
  mode: String,
}

impl QuantizedEmbedding {
  /// The packed quantized table `(num_embeddings, packed_dim)`.
  #[inline(always)]
  pub fn weight_ref(&self) -> &Array {
    &self.weight
  }

  /// The per-group `scales`.
  #[inline(always)]
  pub fn scales_ref(&self) -> &Array {
    &self.scales
  }

  /// The per-group affine `biases` (`None` for scale-only `fp` modes).
  #[inline(always)]
  pub fn biases(&self) -> Option<&Array> {
    self.biases.as_ref()
  }

  /// The quantization group size.
  #[inline(always)]
  pub fn group_size(&self) -> i32 {
    self.group_size
  }

  /// The quantization bit depth.
  #[inline(always)]
  pub fn bits(&self) -> i32 {
    self.bits
  }

  /// The quantization mode tag.
  #[inline(always)]
  pub fn mode(&self) -> &str {
    &self.mode
  }
}

/// A quantize-aware embedding table: either a dense `(num_embeddings, dim)`
/// [`Array`] or a [`QuantizedEmbedding`], dispatched at the lookup call site.
///
/// This is the embedding analogue of [`MaybeQuantizedLinear`] — the abstraction
/// a model uses in place of a bare embedding table so the same load + lookup
/// path serves both dense and quantized checkpoints. [`Self::from_weights`]
/// picks the quantized variant when the checkpoint carries the sibling
/// `<prefix>.scales` (and, for `affine`, `<prefix>.biases`) tensors, and the
/// dense variant otherwise — the weight-map analogue of the
/// `class_predicate`'s `f"{p}.scales" in weights` signal that quantizes
/// `nn.Embedding` alongside `nn.Linear`.
///
/// Two lookup forms are exposed, both mirroring `mlx.nn.Embedding` /
/// `mlx.nn.QuantizedEmbedding`:
///
/// - [`gather`](Self::gather) — `weight[ids]` along axis 0 (the row gather, the
///   embedding's `__call__`); for the quantized variant the gathered packed
///   rows / scales / biases are dequantized (`QuantizedEmbedding.__call__`).
/// - [`dense_table`](Self::dense_table) — the full `(num_embeddings, dim)` dense
///   table; for the quantized variant the whole table is dequantized. A model
///   that consumes the table by a structural op a packed table does not support
///   (a contiguous row slice, or a grid reshape + interpolation) materializes
///   the dense table once via this, then operates on it dense.
#[derive(Debug)]
pub enum MaybeQuantizedEmbedding {
  /// A dense `(num_embeddings, dim)` embedding table (`mlx.nn.Embedding`).
  Dense(Array),
  /// A quantized embedding table (`mlx.nn.QuantizedEmbedding`).
  Quantized(QuantizedEmbedding),
}

impl MaybeQuantizedEmbedding {
  /// Construct a **dense** embedding from a `(num_embeddings, dim)` table.
  #[inline(always)]
  pub fn dense(weight: Array) -> Self {
    MaybeQuantizedEmbedding::Dense(weight)
  }

  /// Construct a **quantized** embedding from the checkpoint's packed
  /// `(weight, scales, biases)` triple and the scheme parameters — mirroring
  /// `mlx.nn.QuantizedEmbedding`.
  ///
  /// The embedding analogue of [`QuantizedLinear::from_parts`]: the
  /// `(weight, scales, biases, group_size, bits, mode)` triple is validated by
  /// the shared `validate_quantized_triple` at LOAD time — the ONE place mlx's
  /// construct-relevant contract is mirrored, so this constructor and
  /// [`QuantizedLinear::from_parts`] cannot drift apart. The embedding has NO
  /// separate dense output bias — its `biases` IS the per-group affine bias — so
  /// the linear-only dense-bias check from `from_parts` does not apply here.
  ///
  /// Reads only `shape()` / `dtype()` metadata (no materialization / eval).
  ///
  /// # Errors
  /// Propagates `validate_quantized_triple`'s typed errors (rank / `uint32`
  /// weight / the scales width identity / per-mode bias arity / the `fp`-mode
  /// `uint8` scale dtype; the `affine` scale/bias dtype rule is deferred to
  /// mlx-c at op-time).
  pub fn from_parts(
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: impl Into<String>,
  ) -> Result<Self> {
    let mode = mode.into();
    validate_quantized_triple(
      "MaybeQuantizedEmbedding::from_parts",
      &weight,
      &scales,
      biases.as_ref(),
      group_size,
      bits,
      &mode,
    )?;
    Ok(MaybeQuantizedEmbedding::Quantized(QuantizedEmbedding {
      weight,
      scales,
      biases,
      group_size,
      bits,
      mode,
    }))
  }

  /// `true` if this is the quantized variant.
  #[inline(always)]
  pub fn is_quantized(&self) -> bool {
    matches!(self, MaybeQuantizedEmbedding::Quantized(_))
  }

  /// The **logical** `(num_embeddings, dim)` of the table — the dequantized
  /// shape a row gather produces (`dim` is the embedding width, NOT the packed
  /// `uint32` width of a quantized table).
  ///
  /// A model that pins its embedding table to a config-derived `(vocab, hidden)`
  /// uses this for both arms so the quantized arm is shape-checked at load (a
  /// packed weight whose dequantized width / row count disagrees with the config
  /// would otherwise only mis-gather at the first forward).
  ///
  /// - **Dense**: the table's own `(rows, dim)` (the dense table is the logical
  ///   table verbatim).
  /// - **Quantized**: `num_embeddings` is the packed weight's row count (the
  ///   leading axis, validated equal to the `scales` row count at construction)
  ///   and `dim` is `scales.shape(-1) * group_size` — the same logical width
  ///   `validate_quantized_triple` recovers and pins against the packed weight,
  ///   so no separate dequantization is needed.
  ///
  /// Reads only `shape()` metadata (no materialization / eval).
  ///
  /// # Errors
  /// - [`Error::RankMismatch`] if the dense table is not rank-2 (the quantized
  ///   arm's rank-2 layout is guaranteed by its construction-time validation);
  /// - [`Error::OutOfRange`] if a dimension exceeds `i32::MAX` (a corrupt huge
  ///   table; the recovery is computed in `i64` so it cannot overflow first).
  pub fn logical_shape(&self) -> Result<(i32, i32)> {
    let context = "MaybeQuantizedEmbedding::logical_shape";
    let (rows, dim): (i64, i64) = match self {
      MaybeQuantizedEmbedding::Dense(weight) => {
        let shape = weight.shape();
        if shape.len() != 2 {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "MaybeQuantizedEmbedding::logical_shape: dense table must be rank-2 (num_embeddings, dim)",
            shape.len() as u32,
            shape.to_vec(),
          )));
        }
        (shape[0] as i64, shape[1] as i64)
      }
      MaybeQuantizedEmbedding::Quantized(q) => {
        // The triple was validated rank-2 at construction; `num_embeddings` is the
        // packed weight's leading axis and the logical width is
        // `scales.shape(-1) * group_size` (mlx's per-group recovery,
        // `validate_quantized_triple` pins it against the packed weight).
        let w_rows = q.weight.shape()[0] as i64;
        let s_shape = q.scales.shape();
        let logical_dim = (s_shape[1] as i64) * i64::from(q.group_size);
        (w_rows, logical_dim)
      }
    };
    let to_i32 = |v: i64| -> Result<i32> {
      i32::try_from(v).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          context,
          "logical embedding dimension exceeds i32::MAX",
          format_smolstr!("{v}"),
        ))
      })
    };
    Ok((to_i32(rows)?, to_i32(dim)?))
  }

  /// Gather embedding rows: `weight[ids]` along axis 0 (the vocab axis),
  /// mirroring `mlx.nn.Embedding.__call__`'s `self.weight[x]` (dense) and
  /// `mlx.nn.QuantizedEmbedding.__call__`'s
  /// `dequantize(weight[x], scales[x], biases[x], ...)` (quantized). `ids` is an
  /// integer [`Array`] of any shape `S`; the result is `S ++ (dim,)`.
  ///
  /// # Errors
  /// Propagates the gather (`take_axis`) / dequantize op errors.
  pub fn gather(&self, ids: &Array) -> Result<Array> {
    match self {
      MaybeQuantizedEmbedding::Dense(weight) => weight.take_axis(ids, 0),
      MaybeQuantizedEmbedding::Quantized(q) => {
        // `mlx.nn.QuantizedEmbedding.__call__`: gather the packed rows + the
        // per-row scales / biases by id, then dequantize the gathered rows.
        let w_rows = q.weight.take_axis(ids, 0)?;
        let s_rows = q.scales.take_axis(ids, 0)?;
        let b_rows = match &q.biases {
          Some(b) => Some(b.take_axis(ids, 0)?),
          None => None,
        };
        quantized::dequantize(
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

  /// The full `(num_embeddings, dim)` dense table — the dense table verbatim, or
  /// the whole quantized table dequantized (`dequantize(weight, scales, biases,
  /// ...)`). A model that consumes the table by a structural op a packed table
  /// does not support (a contiguous row slice, or a grid reshape +
  /// interpolation) materializes it once through here, then operates dense.
  ///
  /// `dtype` optionally casts the dequantized result (the dense variant is
  /// returned at its stored dtype regardless — a model wanting a uniform dtype
  /// casts the dense table at load).
  ///
  /// # Errors
  /// Propagates the clone / dequantize op errors.
  pub fn dense_table(&self, dtype: Option<Dtype>) -> Result<Array> {
    match self {
      MaybeQuantizedEmbedding::Dense(weight) => weight.try_clone(),
      MaybeQuantizedEmbedding::Quantized(q) => quantized::dequantize(
        &q.weight,
        &q.scales,
        q.biases.as_ref(),
        q.group_size,
        q.bits,
        &q.mode,
        None,
        dtype,
      ),
    }
  }

  /// Build a quantize-aware embedding from a checkpoint weight map: a
  /// [`QuantizedEmbedding`] when `<prefix>.scales` is present in `weights`, else
  /// a dense table.
  ///
  /// The embedding analogue of [`MaybeQuantizedLinear::from_weights`]: the
  /// presence of a `<prefix>.scales` sibling is the `class_predicate` signal
  /// that this embedding was pre-quantized in the checkpoint.
  ///
  /// - **Quantized path** (`<prefix>.scales` present): pops `<prefix>.weight`
  ///   (the packed `uint32` table), `<prefix>.scales`, and the optional
  ///   `<prefix>.biases` (present iff `mode == "affine"`), and builds a
  ///   [`QuantizedEmbedding`] with the resolved `(group_size, bits, mode)`. The
  ///   triple is validated by [`Self::from_parts`].
  /// - **Dense path** (no `<prefix>.scales`): pops `<prefix>.weight` and builds
  ///   a dense table.
  ///
  /// `quant` carries the resolved scheme parameters `(group_size, bits, mode)`
  /// for this embedding (resolved by the caller from the parsed
  /// [`crate::lm::quant::PerLayerQuantization`]); it is only consulted on the
  /// quantized path. A `<prefix>.scales` present but `quant == None` is a
  /// checkpoint / config inconsistency and returns a typed
  /// [`Error::InvariantViolation`] rather than guessing scheme parameters.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] if `<prefix>.weight` (or, on the quantized path,
  ///   `<prefix>.scales`) is absent;
  /// - [`Error::InvariantViolation`] if `<prefix>.scales` is present but `quant`
  ///   is `None`;
  /// - propagates [`Self::from_parts`]'s structural validation errors.
  pub fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    quant: Option<(i32, i32, &str)>,
  ) -> Result<Self> {
    let scales_key = format!("{prefix}{SCALES_SUFFIX}");
    if weights.contains_key(&scales_key) {
      let Some((group_size, bits, mode)) = quant else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "MaybeQuantizedEmbedding::from_weights: checkpoint carries a `.scales` sibling for this embedding but no quantization config resolved scheme parameters",
          "a quantized embedding requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      let scales = take_required(weights, prefix, SCALES_SUFFIX)?;
      let biases = weights.remove(&format!("{prefix}{BIASES_SUFFIX}"));
      Self::from_parts(weight, scales, biases, group_size, bits, mode)
    } else {
      let weight = take_required(weights, prefix, WEIGHT_SUFFIX)?;
      Ok(MaybeQuantizedEmbedding::Dense(weight))
    }
  }
}

#[cfg(test)]
mod tests;
