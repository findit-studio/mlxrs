//! Scaled Rotary Position Embedding variants — context-extension RoPE.
//!
//! A 1:1 port of mlx-lm's scaled RoPE family
//! (`mlx_lm/models/rope_utils.py`) and the matching swift
//! `MLXLMCommon/RoPEUtils.swift` / `SuScaledRoPE.swift` layers:
//!
//! - [`Llama3Rope`] — Llama-3.1's piecewise low/high-frequency scaling
//!   (`Llama3RoPE`).
//! - [`SuScaledRope`] — the Su / **longrope** variant (Phi-3): per-dimension
//!   long-factor frequencies plus an input mscale (`SuScaledRoPE`).
//! - [`YarnRope`] — YaRN "NTK-by-parts" interpolation with a ramped blend of
//!   the extrapolation and interpolation frequencies (`YarnRoPE`).
//!
//! # The freqs path
//!
//! The base [`Rope`](super::rope::Rope) drives `mlx_fast_rope` with a scalar
//! `base` (theta) and `freqs = None`. Every variant here instead **precomputes
//! a per-dimension `freqs` array** — one inverse-frequency per feature pair,
//! shape `[dims / 2]` — and forwards it through the *freqs path*
//! ([`rope_with_freqs`](super::rope::rope_with_freqs) /
//! [`rope_dynamic_with_freqs`](super::rope::rope_dynamic_with_freqs)) with
//! `base = None`, exactly as the python/swift refs call
//! `mx.fast.rope(..., base=None, scale=1.0, offset, freqs=self._freqs)`. The
//! variants differ *only* in how that `freqs` array (and, for Su/YaRN, a scalar
//! mscale applied to `x`) is computed.
//!
//! # Why the frequencies are computed on the host
//!
//! The references build `freqs` once at construction via MLX array ops
//! (`mx.arange` / `mx.power` / `mx.where` / `mx.clip`). The result is a
//! deterministic vector of per-dimension constants that depends on nothing but
//! the scaling config — so this port computes it directly in `f64` on the host
//! from the published formula and materialises it with
//! [`Array::from_slice`](crate::array::Array::from_slice). This is numerically
//! equivalent (the same closed-form expression, evaluated at higher host
//! precision before the `f32` store), keeps the construction free of a chain of
//! lazy graph ops, and lets the parity tests assert the exact `freqs` vector
//! against hand-traced formula values. The per-step rotation itself still runs
//! on the fused `mlx_fast_rope` kernel.

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, Error, LengthMismatchPayload, NonFiniteScalarPayload,
    OutOfRangePayload, RankMismatchPayload, Result,
  },
  lm::nn::rope::{RopeOffsetRef, rope_with_freqs_offset},
};
use smol_str::format_smolstr;

use super::rope::DEFAULT_BASE;

/// Validate `dims` and return it as a `usize` half-count (`dims / 2`, the
/// `freqs` length). Mirrors the references' `precondition(dims % 2 == 0)`,
/// surfaced as a recoverable [`Error`] rather than a panic, and additionally
/// rejects a non-positive `dims` (which would yield an empty rotation).
fn freqs_half(dims: i32) -> Result<usize> {
  if dims <= 0 || dims % 2 != 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "scaled RoPE: dims",
      "must be a positive even number",
      format!("{dims}"),
    )));
  }
  Ok((dims / 2) as usize)
}

/// `base ** (arange(0, dims, 2) / dims)` evaluated in `f64` — the per-pair
/// *base* frequencies shared by every variant (mlx-lm's
/// `base ** (mx.arange(0, dims, 2) / dims)`). Element `i` corresponds to
/// feature pair `i` (`i` in `0..dims/2`); the exponent numerator is `2 * i`.
fn base_pair_freqs(base: f64, dims: i32, half: usize) -> Vec<f64> {
  let dims_f = f64::from(dims);
  (0..half)
    .map(|i| base.powf((2 * i) as f64 / dims_f))
    .collect()
}

/// Materialise a host-computed `f64` `freqs` vector as a 1-D `f32` mlx array of
/// shape `[half]` — the per-dimension inverse-frequency array `mlx_fast_rope`
/// consumes (mlx-lm stores `self._freqs` as `mx.float32`).
///
/// The host-side `freqs` are checked for **strict positive finiteness with a
/// finite reciprocal** *before* the `f32` store: every scaled-RoPE formula
/// divides and takes `ln`s of config-derived terms, so a poisoned input or an
/// arithmetic edge (e.g. a zero blend denominator) can yield a NaN/±Inf
/// element. *And* `mlx_fast_rope` itself computes `1 / freqs[i]` from this
/// array at apply time, so a `0.0` element silently becomes `+Inf` inside the
/// kernel, a *negative* element reverses the rotation direction, and a
/// positive *subnormal* (e.g. `f32::from_bits(1)` ≈ 1.4e-45) — though finite
/// and `> 0` — has a reciprocal `~7e44` that overflows `f32` to `+Inf` at
/// `1/freqs` too. The gate therefore rejects any value that is non-finite,
/// non-positive, *or* whose `1.0 / f` is not finite — the catch-all for the
/// per-dimension frequencies in one check. The check is on the narrowed `f32`
/// value: a non-finite `f64` source stays non-finite after the cast, a finite
/// `f64` that overflows the `f32` range (`±Inf` only after narrowing) is
/// caught too, and a positive `f64` underflowing to `0.0` (or to a subnormal
/// that overflows on reciprocal) in `f32` is also caught. The original `f64`
/// is reported in the message for diagnosis.
fn freqs_array(freqs: &[f64]) -> Result<Array> {
  let mut buf: Vec<f32> = Vec::with_capacity(freqs.len());
  for &v in freqs {
    let f = v as f32;
    if !f.is_finite() || f <= 0.0 || !(1.0f32 / f).is_finite() {
      // The freq must be positive, finite, AND have a finite f32 reciprocal
      // (mlx_fast_rope inverts as 1/freqs at apply time, so subnormals whose
      // reciprocal overflows f32 are rejected too). The original `f64` is
      // preserved in the OutOfRange payload's `value` for diagnosis.
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "scaled RoPE freq (check scaling config: base / factor / embeddings / betas)",
        "must be positive, finite, AND have a finite f32 reciprocal (freqs are inverted as 1/freqs by mlx_fast_rope; zero / subnormal would become +Inf at apply time)",
        format_smolstr!("{v}"),
      )));
    }
    buf.push(f);
  }
  Array::from_slice::<f32>(&buf, &(freqs.len(),))
}

/// Reject a non-finite computed scalar constant (the derived input `scale` /
/// `mscale`) before it is stored on the variant. The per-input guards already
/// reject the inputs known to poison the formula, but this is the structural
/// catch-all: no scaled-RoPE constructor may return `Ok` carrying a NaN/±Inf
/// scalar that `apply` would later multiply onto activations. `what` names the
/// constant for the error message.
fn finite_scalar(value: f64, what: &'static str) -> Result<f32> {
  let v = value as f32;
  if v.is_finite() {
    Ok(v)
  } else {
    Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      what, value,
    )))
  }
}

/// Reject a non-finite `f32` config input up front, so a NaN/±Inf never slips
/// past the positivity/`> 1` comparisons that follow (`NaN <= 0.0` and
/// `NaN > 1.0` are both `false`, so an unchecked NaN would pass every ordered
/// guard and poison the arithmetic). `what` names the field for the message.
fn require_finite_input(value: f32, what: &'static str) -> Result<()> {
  if value.is_finite() {
    Ok(())
  } else {
    Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      what,
      value as f64,
    )))
  }
}

/// Reject a non-finite OR non-positive `f32` config input up front — the
/// stronger guard for fields that feed the per-dimension `freqs` math, where
/// `0` is just as poisonous as NaN/Inf (mlx_fast_rope inverts `freqs` as
/// `1/freqs[i]` so a zero element becomes `+Inf` at apply time, and a negative
/// element flips the rotation direction). `NaN > 0.0` is `false`, so this
/// rejects NaN too. `what` names the field for the message.
fn require_positive_input(value: f32, what: &'static str) -> Result<()> {
  if value.is_finite() && value > 0.0 {
    Ok(())
  } else {
    Err(Error::OutOfRange(OutOfRangePayload::new(
      what,
      "must be a positive finite number",
      format_smolstr!("{value}"),
    )))
  }
}

/// Scale the first `dims` features of the last axis of `x` by the scalar
/// `mscale`, leaving any trailing `head_dim - dims` features untouched — the
/// `x[..., :dims] = mscale * x[..., :dims]` step Su/YaRN apply *before* the
/// rotation (mlx-lm `x[..., : self.dim] = self._scale * x[..., : self.dim]`).
///
/// `head_dim == dims` (the common case: the whole last axis is rotated) scales
/// the entire array in one broadcast multiply. When `head_dim > dims`, the
/// leading `dims` slice is scaled and concatenated back with the untouched
/// tail, matching the references' partial-features semantics.
///
/// The mscale scalar is built in `x`'s own dtype so the multiply introduces no
/// dtype promotion: MLX would otherwise upcast a `float16`/`bfloat16` `x` times
/// an `f32` scalar to `float32`. The references store the scaled value back into
/// an `x[..., :dims]` slice (`x[..., :dims] = scale * x[..., :dims]`), whose
/// dtype is the original activation dtype — so the output must keep `x`'s dtype.
/// The untouched tail in the partial-dims path is already in `x`'s dtype, so the
/// concat stays uniform.
fn scale_leading_dims(x: &Array, dims: i32, mscale: f32) -> Result<Array> {
  let ndim = x.ndim();
  if ndim == 0 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "scaled RoPE input must have rank >= 1 (at least one axis)",
      0,
      x.shape().to_vec(),
    )));
  }
  // Build the scalar in `x`'s dtype so `multiply` does not promote half-precision
  // inputs to f32 (mirrors the references' in-place-into-`x` store).
  let scalar = Array::from_slice::<f32>(&[mscale], &(1usize,))?.astype(x.dtype()?)?;
  let last = ndim - 1;
  // `shape()` is `usize`; `dims` and the split index below are `i32`, so a
  // `head_dim` past `i32::MAX` would silently wrap with `as`. Convert checked
  // and surface a recoverable error instead.
  let head_dim_usize = x.shape()[last];
  let head_dim = i32::try_from(head_dim_usize).map_err(|_| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "scaled RoPE head_dim exceeds i32::MAX",
      "i32",
      [("head_dim", head_dim_usize as u64)],
    ))
  })?;
  if head_dim == dims {
    // Whole last axis is rotated: scale x directly (scalar broadcasts).
    return x.multiply(&scalar);
  }
  if head_dim < dims {
    // `dims` is the configured rotation width and must not exceed the last
    // axis's `head_dim`.
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "scaled RoPE: dims (configured rotation width vs input last-axis head_dim)",
      "must be <= head_dim (input last-axis)",
      format_smolstr!("dims={dims}, head_dim={head_dim}"),
    )));
  }
  // head_dim > dims: scale only the leading `dims` features, keep the tail.
  let axis = last as i32;
  let parts = x.split_sections(&[dims], axis)?;
  // `split_sections` at one index yields exactly two parts.
  let head = &parts[0];
  let tail = &parts[1];
  let scaled_head = head.multiply(&scalar)?;
  scaled_head.concatenate_with(&[tail], axis)
}

/// Llama-3.1 scaled RoPE — the piecewise low/high-frequency context-extension
/// scaling. A 1:1 port of mlx-lm's `Llama3RoPE` and swift `Llama3RoPE`.
///
/// The frequencies are split by wavelength into three bands and rescaled so
/// that high-frequency (short-wavelength) components are left almost untouched
/// while low-frequency (long-wavelength) components are stretched by `factor`,
/// with a smooth interpolation in between (see the Llama-3.1 release / the
/// `apply_scaling` helper). The result is stored as the precomputed `freqs`
/// array and applied through the shared freqs path.
#[derive(Debug)]
pub struct Llama3Rope {
  dims: i32,
  traditional: bool,
  freqs: Array,
}

/// Scaling config for [`Llama3Rope`], mirroring the `rope_scaling` dict keys
/// mlx-lm reads (`factor`, `low_freq_factor`, `high_freq_factor`,
/// `original_max_position_embeddings`). Construct via [`Llama3ScalingConfig::new`]
/// for explicit values or [`Llama3ScalingConfig::with_factor`] for the
/// HF-config defaults of the optional fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Llama3ScalingConfig {
  /// Overall stretch applied to the low-frequency band (`rope_scaling["factor"]`).
  pub factor: f32,
  /// Wavelength-band lower factor (`low_freq_factor`, mlx-lm default `1.0`).
  pub low_freq_factor: f32,
  /// Wavelength-band upper factor (`high_freq_factor`, mlx-lm default `4.0`).
  pub high_freq_factor: f32,
  /// The pre-extension training context length
  /// (`original_max_position_embeddings`, mlx-lm default `8192`).
  pub original_max_position_embeddings: f32,
}

impl Llama3ScalingConfig {
  /// All four fields explicit.
  pub fn new(
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position_embeddings: f32,
  ) -> Self {
    Self {
      factor,
      low_freq_factor,
      high_freq_factor,
      original_max_position_embeddings,
    }
  }

  /// `factor` explicit with mlx-lm's defaults for the rest
  /// (`low_freq_factor = 1.0`, `high_freq_factor = 4.0`,
  /// `original_max_position_embeddings = 8192`).
  pub fn with_factor(factor: f32) -> Self {
    Self::new(factor, 1.0, 4.0, 8192.0)
  }
}

impl Llama3Rope {
  /// Construct a Llama-3 scaled RoPE, precomputing the per-dimension `freqs`
  /// from `base` and the scaling config. Mirrors `Llama3RoPE(dims, base,
  /// traditional, scaling_config)`.
  ///
  /// The frequencies follow mlx-lm exactly:
  /// ```text
  /// freqs   = base ** (arange(0, dims, 2) / dims)
  /// wavelen = 2π * freqs
  /// low_wl  = old_ctx / low_freq_factor
  /// high_wl = old_ctx / high_freq_factor
  /// freqs   = where(wavelen > low_wl, freqs * factor, freqs)
  /// medium  = (wavelen > high_wl) & (wavelen < low_wl)
  /// s       = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
  /// smooth  = freqs / ((1 - s) / factor + s)
  /// freqs   = where(medium, smooth, freqs)
  /// ```
  pub fn new(
    dims: i32,
    base: f32,
    traditional: bool,
    scaling: Llama3ScalingConfig,
  ) -> Result<Self> {
    let half = freqs_half(dims)?;
    // Input positive-finiteness: every float field feeds the freqs math.
    //   * `base`: `base ** (2i/dims)` — `0` zeroes every base freq (and a
    //     zero freq later inverts to `+Inf` in mlx_fast_rope); a negative
    //     `base` yields `NaN` for non-integer exponents.
    //   * `factor`: the long-band multiply `f * factor` AND the medium-band
    //     denominator `(1 - s) / factor + s` — `0` zeroes the long band and
    //     divides by zero in the medium band.
    //   * `low_freq_factor` / `high_freq_factor`: `low_wl = old_ctx / low_ff`
    //     and `high_wl = old_ctx / high_ff` — `0` makes the band threshold
    //     `+Inf`, and the medium denominator `high_ff - low_ff` divides by
    //     the difference. A negative factor reverses the band ordering.
    //   * `original_max_position_embeddings`: drives both wavelength
    //     thresholds — a `0` collapses them to `0`, a negative flips them.
    // A NaN/±Inf also slips past the formula's ordered comparisons
    // (`wavelen > low_wl`, band selects) and poisons every freq. The
    // `freqs_array` positive-finite gate below is the catch-all, but
    // rejecting the inputs up front yields a clearer message.
    require_positive_input(base, "base")?;
    require_positive_input(scaling.factor, "factor")?;
    require_positive_input(scaling.low_freq_factor, "low_freq_factor")?;
    require_positive_input(scaling.high_freq_factor, "high_freq_factor")?;
    require_positive_input(
      scaling.original_max_position_embeddings,
      "original_max_position_embeddings",
    )?;
    // Catch-all: rejects any freq that came out non-positive or NaN/±Inf
    // (degenerate factors, equal-band singularities, etc.) before it is stored.
    let freqs = freqs_array(&Self::compute_freqs(f64::from(base), dims, half, scaling))?;
    Ok(Self {
      dims,
      traditional,
      freqs,
    })
  }

  /// Llama-3 with `base = 10000` and the default rotation layout
  /// (`traditional = false`) — the common Llama-3.1 case.
  pub fn standard(dims: i32, scaling: Llama3ScalingConfig) -> Result<Self> {
    Self::new(dims, DEFAULT_BASE, false, scaling)
  }

  /// The host-side freqs computation (extracted so the parity tests can assert
  /// the raw vector). Evaluated in `f64`; see the formula on [`Llama3Rope::new`].
  fn compute_freqs(base: f64, dims: i32, half: usize, c: Llama3ScalingConfig) -> Vec<f64> {
    let factor = f64::from(c.factor);
    let low_ff = f64::from(c.low_freq_factor);
    let high_ff = f64::from(c.high_freq_factor);
    let old_ctx = f64::from(c.original_max_position_embeddings);
    let low_wl = old_ctx / low_ff;
    let high_wl = old_ctx / high_ff;

    base_pair_freqs(base, dims, half)
      .into_iter()
      .map(|f| {
        let wavelen = 2.0 * std::f64::consts::PI * f;
        if wavelen > low_wl {
          // Long-wavelength band: stretch by `factor`.
          f * factor
        } else if wavelen > high_wl {
          // Medium band: smooth interpolation between stretched and unscaled.
          let s = (old_ctx / wavelen - low_ff) / (high_ff - low_ff);
          f / ((1.0 - s) / factor + s)
        } else {
          // Short-wavelength band: untouched.
          f
        }
      })
      .collect()
  }

  /// Apply this Llama-3 RoPE to `x` at scalar position `offset`. Mirrors
  /// `Llama3RoPE.__call__(x, offset)` — forwards the precomputed `freqs`
  /// through the freqs path. Returns a new lazy array (no eval).
  pub fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
    self.apply_with_offset(x, RopeOffsetRef::Scalar(offset))
  }

  /// Apply this Llama-3 RoPE dispatching on a [`RopeOffsetRef`] — the
  /// per-sequence-offset counterpart of [`apply`](Llama3Rope::apply) (the
  /// swift `callAsFunction(_:offset: MLXArray)` overload).
  pub fn apply_with_offset(&self, x: &Array, offset: RopeOffsetRef<'_>) -> Result<Array> {
    rope_with_freqs_offset(x, self.dims, self.traditional, 1.0, offset, &self.freqs)
  }
}

/// Su / **longrope** scaled RoPE (Phi-3 family). A 1:1 port of mlx-lm's
/// `SuScaledRoPE` and swift `SuScaledRoPE`.
///
/// Frequencies are the per-pair bases scaled element-wise by a `long_factor`
/// vector, and the input is multiplied by a scalar mscale before rotation:
/// ```text
/// freqs = long_factor * base ** (arange(0, dims, 2) / dims)
/// f     = max_pos / orig_max
/// scale = long_mscale  or  (1.0 if f <= 1 else sqrt(1 + ln(f) / ln(orig_max)))
/// x[..., :dims] *= scale ; rope(x, freqs)
/// ```
/// Per the python source (and mlx-lm PR #707) `short_factor` / `short_mscale`
/// are unused — only the long path is materialised — so this port omits them.
#[derive(Debug)]
pub struct SuScaledRope {
  dims: i32,
  scale: f32,
  freqs: Array,
}

impl SuScaledRope {
  /// Construct a Su-scaled (longrope) RoPE. `long_factor` is the per-pair
  /// scaling vector and must have length `dims / 2` (mlx multiplies it onto the
  /// `dims/2` base frequencies). `long_mscale`, when `Some`, overrides the
  /// derived input scale.
  ///
  /// Mirrors `SuScaledRoPE(dims, base, max_position_embeddings,
  /// original_max_position_embeddings, long_factor, long_mscale)`.
  pub fn new(
    dims: i32,
    base: f32,
    max_position_embeddings: i32,
    original_max_position_embeddings: i32,
    long_factor: &[f32],
    long_mscale: Option<f32>,
  ) -> Result<Self> {
    let half = freqs_half(dims)?;
    if long_factor.len() != half {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "SuScaledRoPE long_factor length vs dims/2",
        half,
        long_factor.len(),
      )));
    }
    // Input positive-finiteness: `base` raises to `2i/dims` (a non-positive
    // base would yield a zero/NaN base-freq), and each `long_factor` entry
    // multiplies a base freq directly (a `0` entry zeroes the corresponding
    // freq — which would later invert to `+Inf` in mlx_fast_rope's `1/freqs`
    // — and a negative entry flips its rotation direction). The
    // `freqs_array` positive-finite gate below is the catch-all; this is the
    // clearer message.
    require_positive_input(base, "base")?;
    for &lf in long_factor {
      require_positive_input(lf, "long_factor entry")?;
    }
    let base_freqs = base_pair_freqs(f64::from(base), dims, half);
    let freqs: Vec<f64> = base_freqs
      .into_iter()
      .zip(long_factor)
      .map(|(f, &lf)| f64::from(lf) * f)
      .collect();
    // Catch-all: any non-finite freq is rejected before the store.
    let freqs = freqs_array(&freqs)?;

    let scale = match long_mscale {
      // An explicit override skips the derived formula entirely (the embeddings
      // values are then unused), so accept it verbatim — but it is still stored
      // and multiplied onto activations, so it must be finite.
      Some(mscale) => {
        require_finite_input(mscale, "long_mscale")?;
        mscale
      }
      None => {
        // The derived scale divides by, and takes `ln()` of,
        // `original_max_position_embeddings`, and divides `max_position_embeddings`
        // by it; a non-positive value would yield a NaN/Inf scale that silently
        // propagates. Reject it before computing the factor.
        if original_max_position_embeddings <= 0 || max_position_embeddings <= 0 {
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "SuScaledRoPE max_position_embeddings / original_max_position_embeddings (required positive to derive the input scale)",
            "must both be > 0",
            format_smolstr!(
              "max_position_embeddings={max_position_embeddings}, original_max_position_embeddings={original_max_position_embeddings}"
            ),
          )));
        }
        // factor = max_pos / orig_max; scale = 1 if factor <= 1 else default.
        let factor =
          f64::from(max_position_embeddings) / f64::from(original_max_position_embeddings);
        if factor <= 1.0 {
          1.0
        } else {
          // On the derived path with factor > 1 the scale divides by
          // `ln(orig_max)`. `orig_max == 1` makes `ln(1) == 0` → `+Inf` scale,
          // and the `> 0` guard above accepts `1`. Reject `orig_max <= 1` on
          // this path explicitly (the `finite_scalar` gate below would also
          // catch the resulting `+Inf`, but the message here is precise).
          if original_max_position_embeddings <= 1 {
            return Err(Error::OutOfRange(OutOfRangePayload::new(
              "SuScaledRoPE original_max_position_embeddings (derived-scale path, factor > 1)",
              "must be > 1 (ln(1) = 0 divides the scale to +Inf)",
              format_smolstr!("{original_max_position_embeddings}"),
            )));
          }
          // sqrt(1 + ln(factor) / ln(orig_max)); finiteness is the catch-all.
          finite_scalar(
            (1.0 + factor.ln() / f64::from(original_max_position_embeddings).ln()).sqrt(),
            "input scale",
          )?
        }
      }
    };

    Ok(Self { dims, scale, freqs })
  }

  /// The derived (or overridden) input mscale applied to the leading `dims`
  /// features before rotation.
  pub fn scale(&self) -> f32 {
    self.scale
  }

  /// Apply this Su-scaled RoPE to `x` at scalar position `offset`. Scales the
  /// leading `dims` features by [`scale`](SuScaledRope::scale), then rotates via
  /// the freqs path. Mirrors `SuScaledRoPE.__call__`.
  pub fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
    self.apply_with_offset(x, RopeOffsetRef::Scalar(offset))
  }

  /// Apply dispatching on a [`RopeOffsetRef`] — the per-sequence-offset
  /// counterpart of [`apply`](SuScaledRope::apply).
  pub fn apply_with_offset(&self, x: &Array, offset: RopeOffsetRef<'_>) -> Result<Array> {
    // Skip the leading-dims rescale when the scale is exactly 1.0 (e.g.
    // `max_position_embeddings <= original_max_position_embeddings`, or a
    // `long_mscale` of 1) — the multiply/concat is then a no-op. `self.scale`
    // is a stored field (no rounding at this point), so an exact `== 1.0` is
    // safe; mirrors the optimization YaRN already applies.
    if self.scale == 1.0 {
      return rope_with_freqs_offset(x, self.dims, false, 1.0, offset, &self.freqs);
    }
    let scaled = scale_leading_dims(x, self.dims, self.scale)?;
    rope_with_freqs_offset(&scaled, self.dims, false, 1.0, offset, &self.freqs)
  }
}

/// YaRN ("NTK-by-parts") scaled RoPE. A 1:1 port of mlx-lm's `YarnRoPE` and
/// swift `YarnRoPE`.
///
/// YaRN blends the un-extended *extrapolation* frequencies with the linearly
/// *interpolated* (`scaling_factor * base_freqs`) frequencies through a
/// per-dimension ramp derived from a wavelength-based "correction range", and
/// applies a scalar mscale to the input. See the
/// [YaRN paper](https://arxiv.org/abs/2309.00071).
#[derive(Debug)]
pub struct YarnRope {
  dims: i32,
  traditional: bool,
  mscale: f32,
  freqs: Array,
}

/// Tunables for [`YarnRope`], mirroring the `rope_scaling` keys mlx-lm reads.
/// Build with [`YarnConfig::new`] for the `scaling_factor` plus mlx-lm's
/// defaults for the rest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnConfig {
  /// Context-extension factor (`rope_scaling["factor"]`); the interpolation
  /// frequencies are `scaling_factor * base_freqs`.
  pub scaling_factor: f32,
  /// Pre-extension training length (`original_max_position_embeddings`,
  /// mlx-lm default `4096`).
  pub original_max_position_embeddings: i32,
  /// Fast-rotation correction bound (`beta_fast`, mlx-lm default `32`).
  pub beta_fast: f32,
  /// Slow-rotation correction bound (`beta_slow`, mlx-lm default `1`).
  pub beta_slow: f32,
  /// mscale numerator term (`mscale`, mlx-lm default `1`).
  pub mscale: f32,
  /// mscale denominator term (`mscale_all_dim`, mlx-lm default `0`).
  pub mscale_all_dim: f32,
}

impl YarnConfig {
  /// `scaling_factor` explicit with mlx-lm's defaults for the rest
  /// (`original_max_position_embeddings = 4096`, `beta_fast = 32`,
  /// `beta_slow = 1`, `mscale = 1`, `mscale_all_dim = 0`).
  pub fn new(scaling_factor: f32) -> Self {
    Self {
      scaling_factor,
      original_max_position_embeddings: 4096,
      beta_fast: 32.0,
      beta_slow: 1.0,
      mscale: 1.0,
      mscale_all_dim: 0.0,
    }
  }
}

impl YarnRope {
  /// Construct a YaRN RoPE, precomputing the blended `freqs` and the input
  /// mscale. Mirrors `YarnRoPE(dims, traditional, base, scaling_factor,
  /// original_max_position_embeddings, beta_fast, beta_slow, mscale,
  /// mscale_all_dim)`.
  pub fn new(dims: i32, base: f32, traditional: bool, config: YarnConfig) -> Result<Self> {
    let half = freqs_half(dims)?;

    // Input-finiteness up front: a NaN/±Inf float field would slip past the
    // ordered `<= 0` / `<= 1` guards below (`NaN <= 0.0` and `NaN > 1.0` are
    // both `false`) and poison the correction dims, the ramp, the mscale, or the
    // freqs. Reject every float input used in the arithmetic first.
    require_finite_input(base, "base")?;
    require_finite_input(config.scaling_factor, "scaling_factor")?;
    require_finite_input(config.beta_fast, "beta_fast")?;
    require_finite_input(config.beta_slow, "beta_slow")?;
    require_finite_input(config.mscale, "mscale")?;
    require_finite_input(config.mscale_all_dim, "mscale_all_dim")?;

    let base = f64::from(base);
    let dims_f = f64::from(dims);
    let scaling_factor = f64::from(config.scaling_factor);
    let orig_max = f64::from(config.original_max_position_embeddings);

    // `find_correction_dim` divides by `2 * ln(base)` and takes
    // `ln(orig_max / (num_rotations * 2π))`. Reject the inputs that make those
    // produce NaN/Inf (which would silently propagate into `low`/`high`, the
    // ramp, and the `freqs`): `base <= 1` (zero/negative `ln(base)`),
    // `original_max_position_embeddings <= 0` (NaN `ln`), and a non-positive
    // `beta_fast`/`beta_slow` (the `num_rotations` in the inner denominator —
    // zero gives Inf, negative gives a NaN `ln`).
    if base <= 1.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "YarnRoPE: base",
        "must be > 1 to derive correction dims",
        format!("{base}"),
      )));
    }
    if config.original_max_position_embeddings <= 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "YarnRoPE: original_max_position_embeddings",
        "must be positive",
        format!("{}", config.original_max_position_embeddings),
      )));
    }
    if config.beta_fast <= 0.0 || config.beta_slow <= 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "YarnRoPE beta_fast / beta_slow",
        "must both be > 0",
        format_smolstr!(
          "beta_fast={}, beta_slow={}",
          config.beta_fast,
          config.beta_slow
        ),
      )));
    }
    // `scaling_factor` scales the interpolation freqs (`freq_inter =
    // scaling_factor * base_freqs`). A non-positive value (e.g. `0`) makes
    // `freq_inter == 0`, which zeroes the blended-freq denominator on the
    // extrapolation side (`freq_inter * freq_mask + freq_extra * (1 -
    // freq_mask)` → `0` when `freq_mask == 1`) → `0/0 = NaN` freqs. Reject it
    // explicitly (the `freqs_array` gate below is the catch-all).
    if config.scaling_factor <= 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "YarnRoPE: scaling_factor",
        "must be a positive value",
        format!("{}", config.scaling_factor),
      )));
    }

    // yarn_find_correction_dim(num_rotations)
    let find_correction_dim = |num_rotations: f64| {
      (dims_f * (orig_max / (num_rotations * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln())
    };
    // yarn_find_correction_range(): [max(floor(.), 0), min(ceil(.), dims-1)]
    let low = find_correction_dim(f64::from(config.beta_fast)).floor();
    let high = find_correction_dim(f64::from(config.beta_slow)).ceil();
    let low = low.max(0.0);
    let high = high.min(dims_f - 1.0);

    // mscale = get_mscale(scaling_factor, mscale) / get_mscale(scaling_factor, mscale_all_dim)
    let get_mscale = |scale: f64, mscale: f64| {
      if scale <= 1.0 {
        1.0
      } else {
        0.1 * mscale * scale.ln() + 1.0
      }
    };
    // The denominator `get_mscale(scaling_factor, mscale_all_dim)` can be `0`
    // (e.g. a negative `mscale_all_dim` driving `0.1 * m * ln(scale) + 1` to
    // zero), giving a non-finite ratio. `finite_scalar` is the catch-all for the
    // stored mscale — no constructor returns `Ok` with a non-finite scalar.
    let mscale = finite_scalar(
      get_mscale(scaling_factor, f64::from(config.mscale))
        / get_mscale(scaling_factor, f64::from(config.mscale_all_dim)),
      "mscale",
    )?;

    // freq_extra = base_freqs ; freq_inter = scaling_factor * base_freqs
    // freq_mask = 1 - clip((arange(dims/2) - low) / (high - low), 0, 1)
    // freqs = (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1 - freq_mask))
    let extra = base_pair_freqs(base, dims, half);
    // yarn_linear_ramp_mask guards the min==max singularity by nudging max.
    let ramp_max = if (low - high).abs() < f64::EPSILON {
      high + 0.001
    } else {
      high
    };
    let freqs: Vec<f64> = extra
      .into_iter()
      .enumerate()
      .map(|(i, freq_extra)| {
        let freq_inter = scaling_factor * freq_extra;
        let linear = (i as f64 - low) / (ramp_max - low);
        let ramp = linear.clamp(0.0, 1.0);
        let freq_mask = 1.0 - ramp;
        (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1.0 - freq_mask))
      })
      .collect();
    let freqs = freqs_array(&freqs)?;

    Ok(Self {
      dims,
      traditional,
      mscale,
      freqs,
    })
  }

  /// YaRN with `base = 10000` and the default rotation layout
  /// (`traditional = false`).
  pub fn standard(dims: i32, config: YarnConfig) -> Result<Self> {
    Self::new(dims, DEFAULT_BASE, false, config)
  }

  /// The derived input mscale (`get_mscale(...) / get_mscale(...)`); applied to
  /// the leading `dims` features only when it differs from `1.0` (mlx-lm skips
  /// the multiply otherwise).
  pub fn mscale(&self) -> f32 {
    self.mscale
  }

  /// Apply this YaRN RoPE to `x` at scalar position `offset`. Mirrors
  /// `YarnRoPE.__call__`: scales the leading `dims` features by
  /// [`mscale`](YarnRope::mscale) when it is not `1.0`, then rotates via the
  /// freqs path.
  pub fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
    self.apply_with_offset(x, RopeOffsetRef::Scalar(offset))
  }

  /// Apply dispatching on a [`RopeOffsetRef`] — the per-sequence-offset
  /// counterpart of [`apply`](YarnRope::apply).
  pub fn apply_with_offset(&self, x: &Array, offset: RopeOffsetRef<'_>) -> Result<Array> {
    // mlx-lm only rescales x when mscale != 1.0; otherwise x passes through.
    if (self.mscale - 1.0).abs() < f32::EPSILON {
      rope_with_freqs_offset(x, self.dims, self.traditional, 1.0, offset, &self.freqs)
    } else {
      let scaled = scale_leading_dims(x, self.dims, self.mscale)?;
      rope_with_freqs_offset(
        &scaled,
        self.dims,
        self.traditional,
        1.0,
        offset,
        &self.freqs,
      )
    }
  }
}

#[cfg(test)]
// A handful of golden frequency / output values are written at more digits than
// f32 resolves for readability; compared with a `1e-5` tolerance (`TOL`). The
// extra digits document the reference value, not a precision claim.
#[allow(clippy::excessive_precision)]
mod tests {
  use super::*;
  use crate::{dtype::Dtype, lm::nn::rope::rope_with_freqs};

  const TOL: f32 = 1e-5;

  fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
      assert!(
        (g - w).abs() <= TOL,
        "index {i}: got {g}, want {w} (|Δ|={})",
        (g - w).abs()
      );
    }
  }

  /// `[1, 1, 2, head_dim]` input with two tokens, ascending features.
  fn input(head_dim: usize) -> Array {
    let n = 2 * head_dim;
    let data: Vec<f32> = (0..n).map(|v| v as f32).collect();
    Array::from_slice::<f32>(&data, &(1usize, 1usize, 2usize, head_dim)).unwrap()
  }

  // ───────── helper-level parity ─────────

  #[test]
  fn freqs_half_rejects_odd_and_nonpositive() {
    assert!(freqs_half(0).is_err());
    assert!(freqs_half(-2).is_err());
    assert!(freqs_half(3).is_err());
    assert_eq!(freqs_half(8).unwrap(), 4);
  }

  #[test]
  fn base_pair_freqs_matches_formula() {
    // base ** (arange(0, dims, 2) / dims) for dims=8, base=10000:
    // exponents 0, 0.25, 0.5, 0.75 -> 1, 10, 100, 1000.
    let f = base_pair_freqs(10000.0, 8, 4);
    let got: Vec<f32> = f.iter().map(|&v| v as f32).collect();
    assert_close(&got, &[1.0, 10.0, 100.0, 1000.0]);
  }

  // ───────── Llama3 ─────────

  /// Hand-traced Llama3 freqs for dims=8, base=10000, factor=8,
  /// low_freq_factor=1, high_freq_factor=4, original_max=8192.
  ///
  /// ```text
  /// base freqs = [1, 10, 100, 1000]; wavelens = 2*pi*freqs ~=
  ///   [6.2832, 62.832, 628.32, 6283.2].
  /// low_wl = 8192/1 = 8192 ; high_wl = 8192/4 = 2048.
  /// All wavelens < high_wl=2048 except the last (6283.2), which is
  /// high_wl < 6283.2 < low_wl  =>  medium band.
  ///   pairs 0,1,2: wavelen < high_wl  =>  unchanged  =>  1, 10, 100.
  ///   pair 3: medium. wavelen = 2*pi*1000 = 6283.18531.
  ///     s = (8192/6283.18531 - 1)/(4 - 1)
  ///       = (1.3037954 - 1)/3 = 0.10126513.
  ///     smooth = 1000 / ((1 - s)/8 + s)
  ///            = 1000 / (0.89873487/8 + 0.10126513)
  ///            = 1000 / (0.11234186 + 0.10126513)
  ///            = 1000 / 0.21360699 = 4681.482.
  /// ```
  #[test]
  fn llama3_freqs_hand_traced() {
    let c = Llama3ScalingConfig::new(8.0, 1.0, 4.0, 8192.0);
    let f = Llama3Rope::compute_freqs(10000.0, 8, 4, c);
    let got: Vec<f32> = f.iter().map(|&v| v as f32).collect();
    // Closed-form value 4681.482 (see derivation above); f32 store = 4681.4824.
    assert_close(&got, &[1.0, 10.0, 100.0, 4681.4824]);
  }

  #[test]
  fn llama3_high_band_low_factor_is_unscaled() {
    // With a tiny factor and a tiny low_freq_factor every wavelen falls in the
    // long band (wavelen > low_wl), so every freq is scaled by `factor`.
    // low_wl = 8192 / 0.001 = 8.192e6, high_wl = 8192/4 = 2048; all
    // wavelens (max ≈ 6283) are < 2048? No — 6283 > 2048 ⇒ medium for the last.
    // Instead use low_freq_factor large so low_wl is small and all are "long".
    let c = Llama3ScalingConfig::new(2.0, 1000.0, 4000.0, 8192.0);
    // low_wl = 8192/1000 = 8.192 ; smallest wavelen ≈ 6.283 < low_wl ⇒ first
    // pair stays in (high_wl, low_wl)? high_wl = 8192/4000 = 2.048.
    // 6.283 in (2.048, 8.192) ⇒ medium; others have larger wavelens > low_wl ⇒
    // *factor. So pair 0 medium, pairs 1..=3 scaled by factor=2.
    let f = Llama3Rope::compute_freqs(10000.0, 8, 4, c);
    // pairs 1,2,3 = base*2 = 20, 200, 2000.
    assert_close(
      &[f[1] as f32, f[2] as f32, f[3] as f32],
      &[20.0, 200.0, 2000.0],
    );
  }

  #[test]
  fn llama3_apply_matches_freqs_path() {
    // The variant's apply must equal feeding its precomputed freqs straight
    // through the freqs primitive (no input mscale for Llama3).
    let x = input(8);
    let c = Llama3ScalingConfig::with_factor(8.0);
    let r = Llama3Rope::new(8, DEFAULT_BASE, false, c).unwrap();
    let freqs = freqs_array(&Llama3Rope::compute_freqs(f64::from(DEFAULT_BASE), 8, 4, c)).unwrap();
    let mut via_apply = r.apply(&x, 3).unwrap();
    let mut via_freqs = rope_with_freqs(&x, 8, false, 1.0, 3, &freqs).unwrap();
    assert_close(
      &via_apply.to_vec::<f32>().unwrap(),
      &via_freqs.to_vec::<f32>().unwrap(),
    );
  }

  #[test]
  fn llama3_rejects_nonpositive_inputs() {
    // Every float input feeds the freqs math; a zero/negative value either
    // zeroes a freq (which later inverts to `+Inf` in mlx_fast_rope's
    // `1/freqs`) or NaNs the band-select arithmetic. The up-front positive
    // gate must reject these so no `0` or negative reaches `freqs_array`.
    // `factor = 0` ⇒ medium-band denominator `(1-s)/0 + s` is +Inf, long-band
    // multiply zeroes the freq.
    assert!(
      Llama3Rope::new(
        8,
        DEFAULT_BASE,
        false,
        Llama3ScalingConfig::new(0.0, 1.0, 4.0, 8192.0)
      )
      .is_err(),
      "factor=0 must be rejected"
    );
    // `factor < 0` ⇒ reverses rotation direction on long band.
    assert!(
      Llama3Rope::new(
        8,
        DEFAULT_BASE,
        false,
        Llama3ScalingConfig::new(-8.0, 1.0, 4.0, 8192.0)
      )
      .is_err(),
      "negative factor must be rejected"
    );
    // `base = 0` ⇒ `0 ** (2i/dims)` zeroes every base freq.
    assert!(
      Llama3Rope::new(8, 0.0, false, Llama3ScalingConfig::with_factor(8.0)).is_err(),
      "base=0 must be rejected"
    );
    // `base < 0` ⇒ `(-x).powf(non-integer)` yields NaN.
    assert!(
      Llama3Rope::new(8, -10000.0, false, Llama3ScalingConfig::with_factor(8.0)).is_err(),
      "negative base must be rejected"
    );
    // `low_freq_factor = 0` ⇒ `low_wl = old_ctx / 0 = +Inf`.
    assert!(
      Llama3Rope::new(
        8,
        DEFAULT_BASE,
        false,
        Llama3ScalingConfig::new(8.0, 0.0, 4.0, 8192.0)
      )
      .is_err(),
      "low_freq_factor=0 must be rejected"
    );
    // `high_freq_factor = 0` ⇒ `high_wl = old_ctx / 0 = +Inf`.
    assert!(
      Llama3Rope::new(
        8,
        DEFAULT_BASE,
        false,
        Llama3ScalingConfig::new(8.0, 1.0, 0.0, 8192.0)
      )
      .is_err(),
      "high_freq_factor=0 must be rejected"
    );
    // `original_max_position_embeddings = 0` ⇒ collapses both wavelength
    // thresholds to `0`.
    assert!(
      Llama3Rope::new(
        8,
        DEFAULT_BASE,
        false,
        Llama3ScalingConfig::new(8.0, 1.0, 4.0, 0.0)
      )
      .is_err(),
      "original_max_position_embeddings=0 must be rejected"
    );
    // `original_max_position_embeddings < 0` ⇒ flips the wavelength thresholds.
    assert!(
      Llama3Rope::new(
        8,
        DEFAULT_BASE,
        false,
        Llama3ScalingConfig::new(8.0, 1.0, 4.0, -8192.0)
      )
      .is_err(),
      "negative original_max_position_embeddings must be rejected"
    );
  }

  // ───────── SuScaled / longrope ─────────

  /// freqs = long_factor * base^(2i/dims). dims=8, base=10000,
  /// base freqs = [1, 10, 100, 1000], long_factor = [1, 2, 3, 4]
  /// => [1, 20, 300, 4000]. With factor = max_pos/orig_max = 32 > 1 the input
  /// mscale is non-unit; verify `apply` == (x *= scale) then freqs-rope using
  /// the hand-computed freqs vector.
  #[test]
  fn su_scaled_freqs_apply_long_factor() {
    let long_factor = [1.0f32, 2.0, 3.0, 4.0];
    let r = SuScaledRope::new(8, DEFAULT_BASE, 131072, 4096, &long_factor, None).unwrap();
    // Hand-computed freqs (not re-derived from the helper the ctor uses).
    let freqs = Array::from_slice::<f32>(&[1.0, 20.0, 300.0, 4000.0], &(4usize,)).unwrap();
    let x = input(8);
    let scalar = Array::from_slice::<f32>(&[r.scale()], &(1usize,)).unwrap();
    let scaled = x.multiply(&scalar).unwrap();
    let mut manual = rope_with_freqs(&scaled, 8, false, 1.0, 3, &freqs).unwrap();
    let mut via_apply = r.apply(&x, 3).unwrap();
    assert_close(
      &via_apply.to_vec::<f32>().unwrap(),
      &manual.to_vec::<f32>().unwrap(),
    );
  }

  /// scale = sqrt(1 + ln(factor)/ln(orig_max)), factor = max_pos/orig_max.
  /// max_pos=16384, orig_max=4096 ⇒ factor=4.
  /// scale = sqrt(1 + ln(4)/ln(4096)) = sqrt(1 + 1.386294/8.317766)
  ///       = sqrt(1 + 0.166665) = sqrt(1.166665) = 1.080123.
  #[test]
  fn su_scaled_default_scale_hand_traced() {
    let r = SuScaledRope::new(8, DEFAULT_BASE, 16384, 4096, &[1.0, 1.0, 1.0, 1.0], None).unwrap();
    assert!(
      (r.scale() - 1.080123).abs() <= TOL,
      "scale {} != 1.080123",
      r.scale()
    );
  }

  #[test]
  fn su_scaled_factor_le_one_scale_is_one() {
    // max_pos <= orig_max ⇒ factor <= 1 ⇒ scale = 1.0 (no input rescale).
    let r = SuScaledRope::new(8, DEFAULT_BASE, 4096, 4096, &[1.0, 1.0, 1.0, 1.0], None).unwrap();
    assert_eq!(r.scale(), 1.0);
  }

  #[test]
  fn su_scaled_long_mscale_override() {
    let r = SuScaledRope::new(8, DEFAULT_BASE, 131072, 4096, &[1.0; 4], Some(2.5)).unwrap();
    assert_eq!(r.scale(), 2.5);
  }

  #[test]
  fn su_scaled_rejects_wrong_long_factor_len() {
    // long_factor must be dims/2 = 4 long.
    assert!(SuScaledRope::new(8, DEFAULT_BASE, 131072, 4096, &[1.0, 2.0], None).is_err());
  }

  #[test]
  fn su_scaled_apply_equals_manual_scale_then_freqs() {
    // apply == (x[..., :dims] *= scale) then rope_with_freqs(freqs).
    let x = input(8);
    let long_factor = [1.0f32, 2.0, 3.0, 4.0];
    let r = SuScaledRope::new(8, DEFAULT_BASE, 16384, 4096, &long_factor, None).unwrap();
    let scale = r.scale();
    let freqs = {
      let base = base_pair_freqs(f64::from(DEFAULT_BASE), 8, 4);
      let v: Vec<f64> = base
        .into_iter()
        .zip(long_factor)
        .map(|(f, lf)| f64::from(lf) * f)
        .collect();
      freqs_array(&v).unwrap()
    };
    // manual: scale whole x (head_dim == dims) then freqs-rope.
    let scalar = Array::from_slice::<f32>(&[scale], &(1usize,)).unwrap();
    let scaled = x.multiply(&scalar).unwrap();
    let mut manual = rope_with_freqs(&scaled, 8, false, 1.0, 5, &freqs).unwrap();
    let mut via_apply = r.apply(&x, 5).unwrap();
    assert_close(
      &via_apply.to_vec::<f32>().unwrap(),
      &manual.to_vec::<f32>().unwrap(),
    );
  }

  #[test]
  fn su_scaled_rejects_nonpositive_embeddings_in_derived_scale() {
    // The derived (`long_mscale = None`) scale takes `ln(orig_max)` and divides
    // by it; non-positive embeddings would yield a NaN/Inf scale. The ctor must
    // reject them up front rather than store a poisoned scale.
    assert!(SuScaledRope::new(8, DEFAULT_BASE, 16384, 0, &[1.0; 4], None).is_err());
    assert!(SuScaledRope::new(8, DEFAULT_BASE, 16384, -4096, &[1.0; 4], None).is_err());
    assert!(SuScaledRope::new(8, DEFAULT_BASE, 0, 4096, &[1.0; 4], None).is_err());
    assert!(SuScaledRope::new(8, DEFAULT_BASE, -1, 4096, &[1.0; 4], None).is_err());
    // No accepted ctor here can carry a NaN scale: success implies a finite one.
  }

  #[test]
  fn su_scaled_override_skips_embeddings_validation() {
    // An explicit `long_mscale` bypasses the derived formula, so the embeddings
    // values are unused — a zero `original_max_position_embeddings` is then fine.
    let r = SuScaledRope::new(8, DEFAULT_BASE, 0, 0, &[1.0; 4], Some(1.5)).unwrap();
    assert_eq!(r.scale(), 1.5);
  }

  #[test]
  fn su_scaled_rejects_orig_max_one_on_derived_path() {
    // orig_max == 1 passes the `> 0` guard but makes `ln(orig_max) = ln(1) = 0`
    // divide the derived scale to +Inf when factor > 1 (max_pos > orig_max). The
    // ctor must reject it, not store an Inf scale.
    let r = SuScaledRope::new(8, DEFAULT_BASE, 2, 1, &[1.0; 4], None);
    assert!(r.is_err(), "orig_max=1 with factor>1 must be rejected");
    // No accepted ctor carries an Inf scale.
    if let Ok(r) = r {
      assert!(r.scale().is_finite(), "stored scale must be finite");
    }
  }

  #[test]
  fn su_scaled_rejects_nonfinite_float_inputs() {
    // NaN/±Inf floats slip past `<=`/`>` ordered comparisons; the up-front
    // finiteness guards must reject them so no non-finite constant is stored.
    assert!(
      SuScaledRope::new(8, f32::NAN, 16384, 4096, &[1.0; 4], None).is_err(),
      "NaN base"
    );
    assert!(
      SuScaledRope::new(8, f32::INFINITY, 16384, 4096, &[1.0; 4], None).is_err(),
      "Inf base"
    );
    assert!(
      SuScaledRope::new(
        8,
        DEFAULT_BASE,
        16384,
        4096,
        &[f32::NAN, 1.0, 1.0, 1.0],
        None
      )
      .is_err(),
      "NaN long_factor entry"
    );
    assert!(
      SuScaledRope::new(8, DEFAULT_BASE, 16384, 4096, &[1.0; 4], Some(f32::INFINITY)).is_err(),
      "Inf long_mscale override"
    );
  }

  #[test]
  fn su_scaled_rejects_nonpositive_long_factor_or_base() {
    // `base` and each `long_factor` entry feed the freqs math; a `0` long_factor
    // entry zeroes the corresponding base freq (which then inverts to `+Inf`
    // inside `mlx_fast_rope`'s `1/freqs`), and a negative entry flips its
    // rotation direction. `base = 0` zeroes every base freq. The up-front
    // positive gate must reject these before they reach `freqs_array`.
    assert!(
      SuScaledRope::new(8, DEFAULT_BASE, 16384, 4096, &[0.0, 1.0, 1.0, 1.0], None).is_err(),
      "zero long_factor entry must be rejected"
    );
    assert!(
      SuScaledRope::new(8, DEFAULT_BASE, 16384, 4096, &[1.0, -2.0, 1.0, 1.0], None).is_err(),
      "negative long_factor entry must be rejected"
    );
    assert!(
      SuScaledRope::new(8, 0.0, 16384, 4096, &[1.0; 4], None).is_err(),
      "base=0 must be rejected"
    );
    assert!(
      SuScaledRope::new(8, -10000.0, 16384, 4096, &[1.0; 4], None).is_err(),
      "negative base must be rejected"
    );
  }

  #[test]
  fn su_scaled_rejects_subnormal_long_factor_with_inf_reciprocal() {
    // A positive *subnormal* long_factor entry (e.g. `f32::from_bits(1)` ≈
    // 1.4e-45) passes `f > 0` and `f.is_finite()` but its reciprocal
    // `1/f ≈ 7e44` overflows `f32` to `+Inf`. Since `mlx_fast_rope` computes
    // `1/freqs[i]` at apply time, the constructor must reject before any
    // `apply()` Inf can occur.
    assert!(
      SuScaledRope::new(
        2,
        DEFAULT_BASE,
        16384,
        4096,
        &[f32::from_bits(1)],
        Some(1.0)
      )
      .is_err(),
      "subnormal long_factor entry (1/f overflows) must be rejected at construction"
    );
  }

  #[test]
  fn su_scaled_valid_inputs_yield_finite_scale_and_freqs() {
    // Positive control: a well-formed derived-scale config stores a finite scale
    // and finite freqs (the complement of the rejection tests above).
    let r = SuScaledRope::new(8, DEFAULT_BASE, 16384, 4096, &[1.0, 2.0, 3.0, 4.0], None).unwrap();
    assert!(r.scale().is_finite(), "scale must be finite");
    let mut freqs = r.freqs.try_clone().unwrap();
    for v in freqs.to_vec::<f32>().unwrap() {
      assert!(v.is_finite(), "non-finite freq {v}");
    }
  }

  #[test]
  fn su_scaled_scale_one_skip_path_matches_plain_freqs() {
    // factor = max_pos/orig_max <= 1 ⇒ scale == 1.0 ⇒ apply must skip the
    // leading-dims rescale and equal a bare freqs-rope (the YaRN-style skip).
    let long_factor = [1.0f32, 2.0, 3.0, 4.0];
    let r = SuScaledRope::new(8, DEFAULT_BASE, 4096, 4096, &long_factor, None).unwrap();
    assert_eq!(r.scale(), 1.0);

    let x = input(8);
    let freqs = {
      let base = base_pair_freqs(f64::from(DEFAULT_BASE), 8, 4);
      let v: Vec<f64> = base
        .into_iter()
        .zip(long_factor)
        .map(|(f, lf)| f64::from(lf) * f)
        .collect();
      freqs_array(&v).unwrap()
    };
    let mut via_apply = r.apply(&x, 5).unwrap();
    let mut via_freqs = rope_with_freqs(&x, 8, false, 1.0, 5, &freqs).unwrap();
    assert_close(
      &via_apply.to_vec::<f32>().unwrap(),
      &via_freqs.to_vec::<f32>().unwrap(),
    );
  }

  #[test]
  fn su_scaled_scale_one_override_skip_path() {
    // A `long_mscale = Some(1.0)` also drives scale == 1.0; the skip path must
    // still produce the correct (unscaled) rope output.
    let long_factor = [1.0f32, 2.0, 3.0, 4.0];
    let r = SuScaledRope::new(8, DEFAULT_BASE, 131072, 4096, &long_factor, Some(1.0)).unwrap();
    assert_eq!(r.scale(), 1.0);

    let x = input(8);
    let freqs = {
      let base = base_pair_freqs(f64::from(DEFAULT_BASE), 8, 4);
      let v: Vec<f64> = base
        .into_iter()
        .zip(long_factor)
        .map(|(f, lf)| f64::from(lf) * f)
        .collect();
      freqs_array(&v).unwrap()
    };
    let mut via_apply = r.apply(&x, 2).unwrap();
    let mut via_freqs = rope_with_freqs(&x, 8, false, 1.0, 2, &freqs).unwrap();
    assert_close(
      &via_apply.to_vec::<f32>().unwrap(),
      &via_freqs.to_vec::<f32>().unwrap(),
    );
  }

  // ───────── YaRN ─────────

  /// Hand-traced YaRN freqs. dims=8, base=10000, scaling_factor=4,
  /// original_max=4096, beta_fast=32, beta_slow=1.
  ///
  /// ```text
  /// base freqs (freq_extra) = [1, 10, 100, 1000];
  /// freq_inter = 4 * freq_extra = [4, 40, 400, 4000].
  ///
  /// correction_dim(b) = dims*ln(orig_max/(b*2*pi)) / (2*ln(base))
  ///   = 8*ln(4096/(b*6.283185)) / (2*9.210340).
  ///   b=32: 4096/201.06 = 20.371 ; ln=3.014113 ; 8*3.014113/18.42068
  ///         = 24.11290/18.42068 = 1.309017 -> floor -> low=1.
  ///   b=1 : 4096/6.283185 = 651.90 ; ln=6.479917 ; 8*6.479917/18.42068
  ///         = 51.83934/18.42068 = 2.814192 -> ceil -> high=3.
  /// ramp(i) = clip((i - low)/(high - low), 0, 1) = clip((i-1)/2, 0, 1):
  ///   i=0: clip(-0.5)=0 ; i=1: 0 ; i=2: 0.5 ; i=3: 1.
  /// freq_mask = 1 - ramp = [1, 1, 0.5, 0].
  /// freqs = (inter*extra) / (inter*mask + extra*(1-mask)):
  ///   i=0: (4*1)/(4*1 + 1*0)             = 4/4        = 1.
  ///   i=1: (40*10)/(40*1 + 10*0)         = 400/40     = 10.
  ///   i=2: (400*100)/(400*0.5 + 100*0.5) = 40000/250  = 160.
  ///   i=3: (4000*1000)/(4000*0 + 1000*1) = 4e6/1000   = 4000.
  /// => freqs = [1, 10, 160, 4000].
  ///
  /// mscale: scaling_factor=4 > 1, mscale=1, mscale_all_dim=0:
  ///   get_mscale(4,1) = 0.1*1*ln(4)+1 = 0.1386294+1 = 1.138629.
  ///   get_mscale(4,0) = 0.1*0*ln(4)+1 = 1.0.
  ///   mscale = 1.138629 / 1.0 = 1.138629.
  /// ```
  #[test]
  fn yarn_freqs_and_mscale_hand_traced() {
    let cfg = YarnConfig {
      scaling_factor: 4.0,
      original_max_position_embeddings: 4096,
      beta_fast: 32.0,
      beta_slow: 1.0,
      mscale: 1.0,
      mscale_all_dim: 0.0,
    };
    let r = YarnRope::new(8, DEFAULT_BASE, false, cfg).unwrap();
    assert!(
      (r.mscale() - 1.138629).abs() <= TOL,
      "mscale {} != 1.138629",
      r.mscale()
    );
    // Verify the freqs vector by applying at offset 0 with mscale forced out:
    // at offset 0 every angle is 0 so the rotation is identity regardless of
    // freqs; instead compare the freqs array against the hand-traced values by
    // reconstructing it exactly as the constructor does and checking equality.
    let freqs = yarn_reference_freqs(8, f64::from(DEFAULT_BASE), 4.0, 4096, 32.0, 1.0);
    let mut freqs_arr = freqs.try_clone().unwrap();
    assert_close(
      &freqs_arr.to_vec::<f32>().unwrap(),
      &[1.0, 10.0, 160.0, 4000.0],
    );
  }

  /// Reconstruct YaRN freqs independently of `YarnRope` (mirrors the formula)
  /// so the golden test compares two derivations of the same closed form.
  fn yarn_reference_freqs(
    dims: i32,
    base: f64,
    scaling_factor: f64,
    orig_max: i32,
    beta_fast: f64,
    beta_slow: f64,
  ) -> Array {
    let half = (dims / 2) as usize;
    let dims_f = f64::from(dims);
    let orig = f64::from(orig_max);
    let cdim =
      |n: f64| (dims_f * (orig / (n * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln());
    let low = cdim(beta_fast).floor().max(0.0);
    let high = cdim(beta_slow).ceil().min(dims_f - 1.0);
    let ramp_max = if (low - high).abs() < f64::EPSILON {
      high + 0.001
    } else {
      high
    };
    let extra = base_pair_freqs(base, dims, half);
    let v: Vec<f64> = extra
      .into_iter()
      .enumerate()
      .map(|(i, fe)| {
        let fi = scaling_factor * fe;
        let ramp = ((i as f64 - low) / (ramp_max - low)).clamp(0.0, 1.0);
        let mask = 1.0 - ramp;
        (fi * fe) / (fi * mask + fe * (1.0 - mask))
      })
      .collect();
    freqs_array(&v).unwrap()
  }

  #[test]
  fn yarn_scaling_factor_le_one_mscale_is_one() {
    // scaling_factor <= 1 ⇒ get_mscale returns 1 for both ⇒ mscale = 1, and
    // apply must skip the input rescale (x passes straight through).
    let cfg = YarnConfig::new(1.0);
    let r = YarnRope::new(8, DEFAULT_BASE, false, cfg).unwrap();
    assert_eq!(r.mscale(), 1.0);

    let x = input(8);
    let freqs = yarn_reference_freqs(8, f64::from(DEFAULT_BASE), 1.0, 4096, 32.0, 1.0);
    let mut via_apply = r.apply(&x, 4).unwrap();
    let mut via_freqs = rope_with_freqs(&x, 8, false, 1.0, 4, &freqs).unwrap();
    assert_close(
      &via_apply.to_vec::<f32>().unwrap(),
      &via_freqs.to_vec::<f32>().unwrap(),
    );
  }

  #[test]
  fn yarn_apply_includes_mscale() {
    // mscale != 1 ⇒ apply == (x[..., :dims] *= mscale) then freqs-rope.
    let cfg = YarnConfig::new(4.0);
    let r = YarnRope::new(8, DEFAULT_BASE, false, cfg).unwrap();
    let mscale = r.mscale();
    assert!((mscale - 1.0).abs() > TOL, "expected non-unit mscale");

    let x = input(8);
    let freqs = yarn_reference_freqs(8, f64::from(DEFAULT_BASE), 4.0, 4096, 32.0, 1.0);
    let scalar = Array::from_slice::<f32>(&[mscale], &(1usize,)).unwrap();
    let scaled = x.multiply(&scalar).unwrap();
    let mut manual = rope_with_freqs(&scaled, 8, false, 1.0, 6, &freqs).unwrap();
    let mut via_apply = r.apply(&x, 6).unwrap();
    assert_close(
      &via_apply.to_vec::<f32>().unwrap(),
      &manual.to_vec::<f32>().unwrap(),
    );
  }

  #[test]
  fn yarn_rejects_base_le_one() {
    // `find_correction_dim` divides by `2 * ln(base)`; base <= 1 makes that
    // zero/negative (Inf/NaN correction dims). base = 1 ⇒ ln = 0 ⇒ div-by-zero.
    let cfg = YarnConfig::new(4.0);
    assert!(YarnRope::new(8, 1.0, false, cfg).is_err());
    assert!(YarnRope::new(8, 0.0, false, cfg).is_err());
    assert!(YarnRope::new(8, -10.0, false, cfg).is_err());
  }

  #[test]
  fn yarn_rejects_nonpositive_orig_max() {
    // `ln(orig_max / ...)` is NaN/Inf for a non-positive orig_max.
    let mut cfg = YarnConfig::new(4.0);
    cfg.original_max_position_embeddings = 0;
    assert!(YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err());
    cfg.original_max_position_embeddings = -4096;
    assert!(YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err());
  }

  #[test]
  fn yarn_rejects_nonpositive_betas() {
    // betas are the `num_rotations` in the inner denominator: zero ⇒ Inf, a
    // negative ⇒ a NaN `ln`. Both poison `low`/`high` and the freqs.
    let mut cfg = YarnConfig::new(4.0);
    cfg.beta_fast = 0.0;
    assert!(YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err());
    let mut cfg = YarnConfig::new(4.0);
    cfg.beta_slow = -1.0;
    assert!(YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err());
  }

  #[test]
  fn yarn_rejects_nonpositive_scaling_factor() {
    // scaling_factor == 0 ⇒ freq_inter == 0 ⇒ the blended-freq denominator is 0
    // where freq_mask == 1 ⇒ 0/0 = NaN freqs. The ctor must reject it, not store
    // NaN freqs.
    let mut cfg = YarnConfig::new(0.0);
    assert!(
      YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err(),
      "scaling_factor=0 must be rejected"
    );
    cfg.scaling_factor = -4.0;
    assert!(
      YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err(),
      "negative scaling_factor must be rejected"
    );
  }

  #[test]
  fn yarn_rejects_nonfinite_float_inputs() {
    // NaN/±Inf floats slip past the ordered `<= 0` / `<= 1` guards; the up-front
    // finiteness checks must reject every float field so no non-finite freqs or
    // mscale is stored.
    assert!(
      YarnRope::new(8, f32::NAN, false, YarnConfig::new(4.0)).is_err(),
      "NaN base"
    );
    let mut cfg = YarnConfig::new(f32::INFINITY);
    assert!(
      YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err(),
      "Inf scaling_factor"
    );
    cfg = YarnConfig::new(4.0);
    cfg.beta_fast = f32::NAN;
    assert!(
      YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err(),
      "NaN beta_fast"
    );
    cfg = YarnConfig::new(4.0);
    cfg.mscale = f32::INFINITY;
    assert!(
      YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err(),
      "Inf mscale"
    );
    cfg = YarnConfig::new(4.0);
    cfg.mscale_all_dim = f32::NAN;
    assert!(
      YarnRope::new(8, DEFAULT_BASE, false, cfg).is_err(),
      "NaN mscale_all_dim"
    );
  }

  #[test]
  fn yarn_valid_inputs_yield_finite_freqs_and_mscale() {
    // A well-formed config must produce a finite mscale and finite freqs (the
    // positive-control complement of the rejection tests above).
    let cfg = YarnConfig::new(4.0);
    let r = YarnRope::new(8, DEFAULT_BASE, false, cfg).unwrap();
    assert!(r.mscale().is_finite());
    let mut freqs = r.freqs.try_clone().unwrap();
    for v in freqs.to_vec::<f32>().unwrap() {
      assert!(v.is_finite(), "non-finite freq {v}");
    }
  }

  // ───────── partial-dims (head_dim > dims) mscale path ─────────
  //
  // The checked `head_dim usize -> i32` conversion in `scale_leading_dims`
  // (`i32::try_from`) is not exercised by a dedicated test: triggering the
  // overflow arm requires a last-axis dimension past `i32::MAX` (> 2.1e9
  // elements), which cannot be allocated in a unit test. The `head_dim < dims`
  // and `head_dim == dims` arms below cover the in-range conversion.

  #[test]
  fn scale_leading_dims_partial_keeps_tail() {
    // head_dim=6, dims=4: first 4 features scaled by 2, last 2 untouched.
    let x = Array::from_slice::<f32>(
      &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
      ],
      &(1usize, 1usize, 2usize, 6usize),
    )
    .unwrap();
    let mut scaled = scale_leading_dims(&x, 4, 2.0).unwrap();
    assert_close(
      &scaled.to_vec::<f32>().unwrap(),
      &[
        2.0, 4.0, 6.0, 8.0, 5.0, 6.0, // token 0: [1..4]*2, [5,6] kept
        14.0, 16.0, 18.0, 20.0, 11.0, 12.0, // token 1: [7..10]*2, [11,12] kept
      ],
    );
  }

  #[test]
  fn scale_leading_dims_whole_axis() {
    // head_dim == dims: entire array scaled.
    let x =
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1usize, 1usize, 1usize, 4usize)).unwrap();
    let mut scaled = scale_leading_dims(&x, 4, 3.0).unwrap();
    assert_close(&scaled.to_vec::<f32>().unwrap(), &[3.0, 6.0, 9.0, 12.0]);
  }

  #[test]
  fn scale_leading_dims_rejects_dims_gt_head_dim() {
    let x = input(4);
    assert!(scale_leading_dims(&x, 8, 2.0).is_err());
  }

  #[test]
  fn su_scaled_partial_dims_apply() {
    // head_dim=6 > dims=4: Su applies mscale to the first 4 features only, then
    // freqs-rope over dims=4. Compare against the manual decomposition.
    let x = Array::from_slice::<f32>(
      &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
      ],
      &(1usize, 1usize, 2usize, 6usize),
    )
    .unwrap();
    let long_factor = [1.0f32, 2.0]; // dims/2 = 2
    let r = SuScaledRope::new(4, DEFAULT_BASE, 16384, 4096, &long_factor, None).unwrap();
    let scale = r.scale();
    let freqs = {
      let base = base_pair_freqs(f64::from(DEFAULT_BASE), 4, 2);
      let v: Vec<f64> = base
        .into_iter()
        .zip(long_factor)
        .map(|(f, lf)| f64::from(lf) * f)
        .collect();
      freqs_array(&v).unwrap()
    };
    let manual_scaled = scale_leading_dims(&x, 4, scale).unwrap();
    let mut manual = rope_with_freqs(&manual_scaled, 4, false, 1.0, 2, &freqs).unwrap();
    let mut via_apply = r.apply(&x, 2).unwrap();
    assert_close(
      &via_apply.to_vec::<f32>().unwrap(),
      &manual.to_vec::<f32>().unwrap(),
    );
  }

  // ───────── dtype preservation (no half-precision upcast) ─────────
  //
  // The mscale multiply must not promote a float16/bfloat16 input to float32:
  // MLX promotes `half * f32 -> f32`, so building the mscale in f32 would silently
  // upcast Q/K. The references store the scaled value back into an `x[..., :dims]`
  // slice (`x[..., :dims] = scale * x[..., :dims]`) whose dtype is the original
  // activation dtype, so the output dtype must equal the input dtype. These cover
  // both the whole-axis (`head_dim == dims`) and partial-dims (`head_dim > dims`)
  // paths; the f32 numerical-parity tests above stay as the value checks.

  /// `x` of `dtype`, shape `[1, 1, 2, head_dim]`, ascending features — the f16/
  /// bf16 counterpart of [`input`]. Built by casting the f32 input so no `half`
  /// scalar import is needed and the exact production code path is exercised.
  fn input_dtype(head_dim: usize, dtype: Dtype) -> Array {
    input(head_dim).astype(dtype).unwrap()
  }

  #[test]
  fn scale_leading_dims_preserves_half_dtype() {
    for dtype in [Dtype::F16, Dtype::BF16] {
      // head_dim == dims: whole-axis multiply.
      let whole = scale_leading_dims(&input_dtype(8, dtype), 8, 2.0).unwrap();
      assert_eq!(whole.dtype().unwrap(), dtype, "whole-axis dtype, {dtype:?}");
      // head_dim > dims: scaled head concatenated with the untouched tail.
      let partial = scale_leading_dims(&input_dtype(8, dtype), 4, 2.0).unwrap();
      assert_eq!(
        partial.dtype().unwrap(),
        dtype,
        "partial-dims dtype, {dtype:?}"
      );
    }
  }

  #[test]
  fn su_scaled_apply_preserves_half_dtype() {
    // factor = max_pos/orig_max = 4 > 1 ⇒ non-unit scale ⇒ the mscale multiply
    // runs; output dtype must match the half-precision input for both paths.
    for dtype in [Dtype::F16, Dtype::BF16] {
      // head_dim == dims = 8.
      let r = SuScaledRope::new(8, DEFAULT_BASE, 16384, 4096, &[1.0; 4], None).unwrap();
      assert!((r.scale() - 1.0).abs() > TOL, "expected non-unit scale");
      let out = r.apply(&input_dtype(8, dtype), 3).unwrap();
      assert_eq!(
        out.dtype().unwrap(),
        dtype,
        "Su head_dim==dims dtype, {dtype:?}"
      );
      // head_dim = 8 > dims = 4 (partial-dims): long_factor is dims/2 = 2.
      let r_partial = SuScaledRope::new(4, DEFAULT_BASE, 16384, 4096, &[1.0, 2.0], None).unwrap();
      let out_partial = r_partial.apply(&input_dtype(8, dtype), 3).unwrap();
      assert_eq!(
        out_partial.dtype().unwrap(),
        dtype,
        "Su head_dim>dims dtype, {dtype:?}"
      );
    }
  }

  #[test]
  fn yarn_apply_preserves_half_dtype() {
    // scaling_factor = 4 ⇒ mscale != 1 ⇒ the mscale multiply runs; output dtype
    // must match the half-precision input for both paths.
    for dtype in [Dtype::F16, Dtype::BF16] {
      let cfg = YarnConfig::new(4.0);
      let r = YarnRope::new(8, DEFAULT_BASE, false, cfg).unwrap();
      assert!((r.mscale() - 1.0).abs() > TOL, "expected non-unit mscale");
      // head_dim == dims = 8.
      let out = r.apply(&input_dtype(8, dtype), 6).unwrap();
      assert_eq!(
        out.dtype().unwrap(),
        dtype,
        "YaRN head_dim==dims dtype, {dtype:?}"
      );
      // head_dim = 8 > dims = 4 (partial-dims).
      let r_partial = YarnRope::new(4, DEFAULT_BASE, false, cfg).unwrap();
      assert!(
        (r_partial.mscale() - 1.0).abs() > TOL,
        "expected non-unit mscale"
      );
      let out_partial = r_partial.apply(&input_dtype(8, dtype), 6).unwrap();
      assert_eq!(
        out_partial.dtype().unwrap(),
        dtype,
        "YaRN head_dim>dims dtype, {dtype:?}"
      );
    }
  }
}
