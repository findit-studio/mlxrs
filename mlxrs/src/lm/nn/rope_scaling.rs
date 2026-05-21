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
  error::{Error, Result},
  lm::nn::rope::{RopeOffsetRef, rope_with_freqs_offset},
};

use super::rope::DEFAULT_BASE;

/// Validate `dims` and return it as a `usize` half-count (`dims / 2`, the
/// `freqs` length). Mirrors the references' `precondition(dims % 2 == 0)`,
/// surfaced as a recoverable [`Error`] rather than a panic, and additionally
/// rejects a non-positive `dims` (which would yield an empty rotation).
fn freqs_half(dims: i32) -> Result<usize> {
  if dims <= 0 || dims % 2 != 0 {
    return Err(Error::ShapeMismatch {
      message: format!("scaled RoPE dims must be a positive even number, got {dims}"),
    });
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
fn freqs_array(freqs: &[f64]) -> Result<Array> {
  let buf: Vec<f32> = freqs.iter().map(|&v| v as f32).collect();
  Array::from_slice::<f32>(&buf, &(freqs.len(),))
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
fn scale_leading_dims(x: &Array, dims: i32, mscale: f32) -> Result<Array> {
  let scalar = Array::from_slice::<f32>(&[mscale], &(1usize,))?;
  let ndim = x.ndim();
  if ndim == 0 {
    return Err(Error::ShapeMismatch {
      message: "scaled RoPE input must have at least one axis".to_string(),
    });
  }
  let last = ndim - 1;
  let head_dim = x.shape()[last] as i32;
  if head_dim == dims {
    // Whole last axis is rotated: scale x directly (scalar broadcasts).
    return x.multiply(&scalar);
  }
  if head_dim < dims {
    return Err(Error::ShapeMismatch {
      message: format!("scaled RoPE dims {dims} exceeds head_dim {head_dim}"),
    });
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
      return Err(Error::ShapeMismatch {
        message: format!(
          "SuScaledRoPE long_factor length {} != dims/2 {half}",
          long_factor.len()
        ),
      });
    }
    let base_freqs = base_pair_freqs(f64::from(base), dims, half);
    let freqs: Vec<f64> = base_freqs
      .into_iter()
      .zip(long_factor)
      .map(|(f, &lf)| f64::from(lf) * f)
      .collect();
    let freqs = freqs_array(&freqs)?;

    let scale = long_mscale.unwrap_or_else(|| {
      // factor = max_pos / orig_max; scale = 1 if factor <= 1 else default.
      let factor = f64::from(max_position_embeddings) / f64::from(original_max_position_embeddings);
      if factor <= 1.0 {
        1.0
      } else {
        // sqrt(1 + ln(factor) / ln(orig_max))
        (1.0 + factor.ln() / f64::from(original_max_position_embeddings).ln()).sqrt() as f32
      }
    });

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
    let scaled = scale_leading_dims(x, self.dims, self.scale)?;
    rope_with_freqs_offset(&scaled, self.dims, false, 1.0, offset, &self.freqs)
  }
}

/// YaRN ("NTK-by-parts") scaled RoPE. A 1:1 port of mlx-lm's `YarnRoPE` and
/// swift `YarnRoPE`.
///
/// YaRN blends the un-extended *extrapolation* frequencies with the linearly
/// *interpolated* (`/ scaling_factor`) frequencies through a per-dimension ramp
/// derived from a wavelength-based "correction range", and applies a scalar
/// mscale to the input. See the [YaRN paper](https://arxiv.org/abs/2309.00071).
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
    let base = f64::from(base);
    let dims_f = f64::from(dims);
    let scaling_factor = f64::from(config.scaling_factor);
    let orig_max = f64::from(config.original_max_position_embeddings);

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
    let mscale = (get_mscale(scaling_factor, f64::from(config.mscale))
      / get_mscale(scaling_factor, f64::from(config.mscale_all_dim))) as f32;

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
  use crate::lm::nn::rope::rope_with_freqs;

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

  // ───────── partial-dims (head_dim > dims) mscale path ─────────

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
}
