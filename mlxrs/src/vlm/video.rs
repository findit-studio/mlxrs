//! Model-agnostic video *preprocessing* math: frame-count sampling,
//! pixel-budget-aware target sizing, and per-frame image preparation.
//!
//! Ported from
//! [`mlx-vlm/mlx_vlm/video_generate.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/video_generate.py)
//! (`round_by_factor` / `ceil_by_factor` / `floor_by_factor`,
//! [`smart_resize`], [`smart_nframes`], and the per-frame resize+stack
//! body of `fetch_video`) and cross-checked against
//! [`mlx-swift-lm/Libraries/MLXVLM/Models/QwenVL.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/Models/QwenVL.swift)
//! `QwenVL.targetSize` (the swift mirror of `image_processing_qwen2_vl.
//! smart_resize`) and `MediaProcessing.asProcessedSequence`
//! (uniform/fps frame sampling).
//!
//! ## Scope
//! - **In scope (this module):** the portable, codec-free preprocessing
//!   *math* every Qwen-VL-class video pipeline shares —
//!     - [`smart_resize`]: compute the factor-aligned `(H, W)` that keeps
//!       the aspect ratio while landing the pixel count inside
//!       `[min_pixels, max_pixels]`.
//!     - [`smart_nframes`]: compute how many frames to sample from the
//!       fps / min-frames / max-frames / total-frames budget (with the
//!       reference's clamp + factor-rounding edges), plus
//!       [`sample_frame_indices`] for the `np.linspace(0, total-1,
//!       n).round()` index picks.
//!     - [`process_frames`]: apply the existing cross-model image
//!       [`preprocess`] to a
//!       caller-provided sequence of already-decoded frames and stack the
//!       result into the model input layout.
//! - **Out of scope — DECODING IS THE CALLER'S RESPONSIBILITY.** This
//!   module deliberately takes **already-decoded frames** (a
//!   `&[image::DynamicImage]`) and never opens a container. The python
//!   reference decodes mp4 via `cv2.VideoCapture` (`video_generate.py:
//!   207-226`) and the swift reference via `AVAssetImageGenerator`
//!   (`MediaProcessing.swift:288-330`); both pull in a platform codec.
//!   Wiring a Rust container demuxer/decoder (ffmpeg / `symphonia`-video /
//!   `re_mp4` / etc.) is a documented **follow-up** that needs a codec
//!   dependency and is intentionally not part of this portable-math port.
//!   The valuable, dependency-free part — the sampling + resize + prep
//!   arithmetic — is what lives here. Callers decode frames however they
//!   like (their own ffmpeg binding, a test fixture, pre-extracted PNGs)
//!   and hand the resulting `&[DynamicImage]` to [`process_frames`], using
//!   [`smart_nframes`] + [`sample_frame_indices`] to choose which source
//!   frames to decode in the first place.
//!
//! ## Channel layout (intentional divergence from python, parity with
//! [`crate::vlm::image`])
//! The python `fetch_video` returns planar `(T, C, H, W)` f32
//! (`video_generate.py:282`). We instead stack **channel-last**
//! `[T, H, W, 3]` because [`process_frames`] composes the existing
//! cross-model [`preprocess`], whose
//! documented output is channel-last `[H, W, 3]` (see that module's
//! `Conventions > Channel layout` block). Per the project's
//! no-per-model-arch rule, the planar `[T, C, H, W]` (or the Qwen
//! flattened-patch `[grid_t*grid_h*grid_w, ...]`) layout is a per-model
//! contract the per-model video processor owns; it is one lazy
//! `transpose_axes(&[0, 3, 1, 2])` away from the channel-last stack this
//! module emits (zero data movement until `eval`).
//!
//! ## No implicit eval
//! [`process_frames`] composes lazily and returns an un-evaluated
//! `Array`; callers `eval()` (or use a data accessor) to materialize.

use crate::{
  array::Array,
  error::{Error, Result},
  ops::shape::stack,
  vlm::image::{ImageProcessorConfig, Layout, preprocess},
};

/// `image_processing_qwen2_vl.IMAGE_FACTOR` (`video_generate.py:33`):
/// every resized side is aligned to a multiple of this. 28 = a 14×14 ViT
/// patch × the 2×2 spatial merge Qwen-VL applies.
pub const IMAGE_FACTOR: i64 = 28;
/// `MIN_PIXELS` (`video_generate.py:34`) — `4 * 28 * 28`.
pub const MIN_PIXELS: i64 = 4 * 28 * 28;
/// `MAX_PIXELS` (`video_generate.py:35`) — `16384 * 28 * 28`.
pub const MAX_PIXELS: i64 = 16384 * 28 * 28;
/// `MAX_RATIO` (`video_generate.py:36`): the absolute aspect-ratio ceiling
/// `smart_resize` rejects above.
pub const MAX_RATIO: i64 = 200;

/// `VIDEO_MIN_PIXELS` (`video_generate.py:38`) — `128 * 28 * 28`.
pub const VIDEO_MIN_PIXELS: i64 = 128 * 28 * 28;
/// `VIDEO_MAX_PIXELS` (`video_generate.py:39`) — `768 * 28 * 28`.
pub const VIDEO_MAX_PIXELS: i64 = 768 * 28 * 28;
/// `FRAME_FACTOR` (`video_generate.py:40`): sampled frame counts are
/// aligned to a multiple of this (Qwen-VL's temporal patch size).
pub const FRAME_FACTOR: i64 = 2;
/// `FPS` (`video_generate.py:41`): default frames-per-second to sample at.
pub const FPS: f64 = 2.0;
/// `FPS_MIN_FRAMES` (`video_generate.py:42`): floor on the fps-derived
/// frame count.
pub const FPS_MIN_FRAMES: i64 = 4;
/// `FPS_MAX_FRAMES` (`video_generate.py:43`): ceiling on the fps-derived
/// frame count.
pub const FPS_MAX_FRAMES: i64 = 768;

/// Largest `i64` magnitude that survives an `i64 → f64 → i64` round-trip
/// without precision loss: f64's mantissa holds 53 bits, so every integer in
/// `[-2^53, 2^53]` is exactly representable but beyond it consecutive
/// integers collapse onto the same f64. The `*_by_factor` family below does
/// `op(number / factor) * factor` in f64; for an `|number| > 2^53` the very
/// first `number as f64` is already a *different* value, so the result is
/// silently wrong (not merely overflowing). We reject such inputs up front
/// rather than emit a plausible-looking but corrupt size — every real image
/// dimension / frame count is many orders of magnitude below this.
const F64_EXACT_INT_MAX: i64 = 1 << 53;

/// `i64::MAX + 1 == 2^63` as the smallest `f64` strictly above the `i64`
/// range. `i64::MAX` itself is not f64-representable (`i64::MAX as f64`
/// rounds UP to `2^63`), so an `f64 < this` bound guarantees the subsequent
/// `as i64` cast is exact and in range.
const F64_TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

/// Multiply a (already factor-divided + rounded, integral) `quotient` float
/// by an `i64` `factor`, returning [`Error::ShapeMismatch`] instead of
/// panicking (debug) or silently wrapping (release) on overflow.
///
/// The `*_by_factor` family computes `op(number / factor) * factor`. The
/// final `quotient * factor` can blow past `i64` for a legitimately large
/// float `number` (the `_f` scale variants pass `height * beta` etc.). We
/// funnel every such product through this one checked path so NO caller can
/// panic or wrap — overflow surfaces recoverably, naming the operands.
///
/// `quotient` arrives as the post-`round`/`ceil`/`floor` `f64`, so it is
/// integral (or `±inf`/`NaN` on a degenerate input). We reject non-finite
/// quotients and any quotient outside the exactly-representable `i64` range
/// before the cast, then `checked_mul` (== an i128 product bounds-checked to
/// fit `i64`) the validated integer by `factor`.
fn factor_product(quotient: f64, factor: i64, op: &str) -> Result<i64> {
  if !quotient.is_finite() || quotient >= F64_TWO_POW_63 || quotient < i64::MIN as f64 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "{op}_by_factor: quotient ({quotient}) out of i64 range before scaling by factor ({factor})"
      ),
    });
  }
  (quotient as i64)
    .checked_mul(factor)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("{op}_by_factor: quotient ({quotient}) * factor ({factor}) overflows i64"),
    })
}

/// Guard an integer-input `*_by_factor` call before its lossy f64 division:
/// reject a `number` whose magnitude exceeds [`F64_EXACT_INT_MAX`] (the f64
/// round-trip would corrupt it) — see that constant. Normal frame-sized
/// inputs pass untouched.
fn check_factor_input(number: i64, op: &str) -> Result<()> {
  // `unsigned_abs` (not `i64::abs`) — `i64::MIN.abs()` itself overflows/panics,
  // and a hostile caller can pass `i64::MIN` (e.g. `FrameSampling::Fixed`).
  if number.unsigned_abs() > F64_EXACT_INT_MAX as u64 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "{op}_by_factor: number ({number}) exceeds the f64 exact-integer range \
         (±2^53); the float division would lose precision and corrupt the result"
      ),
    });
  }
  Ok(())
}

/// Bound the `smart_resize` `beta` (sqrt) scale path to the domain where the
/// f64 ratio is provably bit-exact with the python reference.
///
/// Python computes `beta`'s ratio as an EXACT arbitrary-precision `int / int`
/// (`(height * width) / max_pixels` for scale-down, `min_pixels / (height *
/// width)` for scale-up) and *then* takes `math.sqrt`. For huge operands that
/// `int / int -> float` is correctly-rounded straight from the true rational,
/// whereas a Rust `f64 / f64` double-rounds — each integer rounds to f64 first,
/// then the division rounds again — and can differ in the last ULP. Reproducing
/// python's single-rounding for arbitrary magnitudes would need a
/// correctly-rounded big-rational→f64 divider (risky, and out of scope per the
/// match-the-reference rule).
///
/// Instead we BOUND THE DOMAIN per branch: every integer in `[0, 2^53]` is
/// exactly f64-representable, so when BOTH operands of the *actual* division on
/// this branch are `<= 2^53` the `as f64` casts lose nothing and IEEE-754
/// division is correctly-rounded — i.e. the f64 ratio EQUALS python's
/// `int / int -> float` to the bit. Above `2^53` we reject with a recoverable
/// [`Error::ShapeMismatch`] naming the values rather than emit an
/// imprecisely-rounded size.
///
/// **Branch-specific operands** (this is why the helper is per-branch — a
/// single combined guard over-rejects):
/// - Scale-DOWN: `beta = sqrt((height * width) / max_pixels)`. The ratio
///   operands are `height * width` (numerator) and `max_pixels` (denominator);
///   `min_pixels` does NOT appear. Use [`check_beta_domain_down`].
/// - Scale-UP: `beta = sqrt(min_pixels / (height * width))`. The ratio operands
///   are `min_pixels` (numerator) and `height * width` (denominator);
///   `max_pixels` does NOT appear. Use [`check_beta_domain_up`]. A caller with
///   a small image, a huge `max_pixels` sentinel, and a normal `min_pixels`
///   (e.g. `smart_resize(1, 1, 28, 3136, i64::MAX)`) is correctly admitted to
///   the scale-up branch under this split — under a `max_pixels`-keyed guard
///   it would be over-rejected even though python returns `(56, 56)`.
///
/// The realistic video domain is unaffected: frame dims run in the thousands,
/// so `height * width` is in the millions — twenty-plus orders of magnitude
/// below `2^53` — and `max_pixels` / `min_pixels` are the
/// `16384 * 28 * 28 ≈ 1.3e7` / `4 * 28 * 28` budgets. Both guards therefore
/// always pass (and match python exactly) for real inputs; each fires only on
/// the pathological dims/budgets where its branch's f64 path could silently
/// diverge.
fn check_beta_domain_down(height: i64, width: i64, max_pixels: i64) -> Result<()> {
  // EXACT i128 product — `height`/`width` are validated positive `i64`, so
  // `height * width` can exceed `i64` (and is inexact in f64); i128 holds it
  // overflow-free and the comparison is exact.
  let area = (height as i128) * (width as i128);
  if area > F64_EXACT_INT_MAX as i128 || max_pixels > F64_EXACT_INT_MAX {
    return Err(Error::ShapeMismatch {
      message: format!(
        "smart_resize (scale-down): height*width (= {area}) or max_pixels (= {max_pixels}) \
         exceeds the exact-f64 range 2^53 ({F64_EXACT_INT_MAX}); the beta (sqrt) scale-down \
         ratio (height*width / max_pixels) is not exactly computable there and is rejected \
         rather than rounded imprecisely"
      ),
    });
  }
  Ok(())
}

/// Scale-UP companion to [`check_beta_domain_down`]: the scale-up `beta` ratio
/// is `min_pixels / (height * width)`, so the operands to bound are
/// `min_pixels` and `area`, NOT `max_pixels`. See [`check_beta_domain_down`]'s
/// doc for the precision-rationale and the over-rejection case this split
/// avoids.
fn check_beta_domain_up(height: i64, width: i64, min_pixels: i64) -> Result<()> {
  let area = (height as i128) * (width as i128);
  if area > F64_EXACT_INT_MAX as i128 || min_pixels > F64_EXACT_INT_MAX {
    return Err(Error::ShapeMismatch {
      message: format!(
        "smart_resize (scale-up): height*width (= {area}) or min_pixels (= {min_pixels}) \
         exceeds the exact-f64 range 2^53 ({F64_EXACT_INT_MAX}); the beta (sqrt) scale-up \
         ratio (min_pixels / (height*width)) is not exactly computable there and is rejected \
         rather than rounded imprecisely"
      ),
    });
  }
  Ok(())
}

/// Closest integer to `number / factor`, times `factor`.
///
/// Ports python `round_by_factor` (`video_generate.py:51-53`):
/// `round(number / factor) * factor`. Python's `round` is **round-half-to
/// -even** (banker's rounding), and this port reproduces that exactly via
/// an internal `round_half_to_even` helper so a `.5` quotient (e.g.
/// `round_by_factor(14, 28)` → `round(0.5) = 0`) matches the reference
/// rather than diverging like the swift `Int(round(...))`
/// (round-half-away-from-zero) would.
///
/// # Errors
/// [`Error::ShapeMismatch`] for an adversarial near-`i64::MAX` `number`: the
/// f64 division/`* factor` cannot represent it faithfully (precision loss
/// past ±2^53 or `i64` product overflow). The reference assumes frame-sized
/// inputs and never trips this; we surface it recoverably rather than panic
/// (debug) or silently wrap to a small valid-looking size (release).
pub fn round_by_factor(number: i64, factor: i64) -> Result<i64> {
  check_factor_input(number, "round")?;
  factor_product(
    round_half_to_even_f(number as f64 / factor as f64),
    factor,
    "round",
  )
}

/// Smallest multiple of `factor` that is `>= number`.
///
/// Ports python `ceil_by_factor` (`video_generate.py:56-58`):
/// `ceil(number / factor) * factor`.
///
/// # Errors
/// [`Error::ShapeMismatch`] when `number` cannot be faithfully scaled in f64
/// (precision loss past ±2^53 or `i64` product overflow) — see
/// [`round_by_factor`].
pub fn ceil_by_factor(number: i64, factor: i64) -> Result<i64> {
  check_factor_input(number, "ceil")?;
  factor_product((number as f64 / factor as f64).ceil(), factor, "ceil")
}

/// Largest multiple of `factor` that is `<= number`.
///
/// Ports python `floor_by_factor` (`video_generate.py:61-63`):
/// `floor(number / factor) * factor`.
///
/// # Errors
/// [`Error::ShapeMismatch`] when `number` cannot be faithfully scaled in f64
/// (precision loss past ±2^53 or `i64` product overflow) — see
/// [`round_by_factor`].
pub fn floor_by_factor(number: i64, factor: i64) -> Result<i64> {
  check_factor_input(number, "floor")?;
  factor_product((number as f64 / factor as f64).floor(), factor, "floor")
}

/// Float-input variant of [`ceil_by_factor`]: `ceil(number / factor) *
/// factor` with a real-valued `number`. Used by [`smart_resize`]'s
/// scale-up branch, where python calls `ceil_by_factor(height * beta,
/// factor)` with the FLOAT `height * beta` (`video_generate.py:92`).
/// Kept separate from the integer entry so the public `ceil_by_factor`
/// stays the faithful 1:1 of the integer call sites
/// (`smart_nframes`/`fetch_video`). Product overflow is caught by
/// [`factor_product`].
fn ceil_by_factor_f(number: f64, factor: i64) -> Result<i64> {
  factor_product((number / factor as f64).ceil(), factor, "ceil")
}

/// Float-input variant of [`floor_by_factor`]: `floor(number / factor) *
/// factor` with a real-valued `number`. Used by [`smart_resize`]'s
/// scale-down branch (python `floor_by_factor(height / beta, factor)`
/// with the FLOAT `height / beta`, `video_generate.py:88`). Product overflow
/// is caught by [`factor_product`].
fn floor_by_factor_f(number: f64, factor: i64) -> Result<i64> {
  factor_product((number / factor as f64).floor(), factor, "floor")
}

/// Round half to even (banker's rounding) — the behavior of python's
/// built-in `round()` on a single float, which `round_by_factor` relies
/// on. `round(0.5) == 0`, `round(1.5) == 2`, `round(2.5) == 2`,
/// `round(-0.5) == 0`. Rust's `f64::round` rounds half *away from zero*
/// (`0.5_f64.round() == 1.0`), so we cannot use it directly for parity.
///
/// Returns the rounded value as an `f64` (still integral) so callers that
/// must multiply by a factor can range-check before the `i64` cast — see
/// [`factor_product`]. The exactly-`.5` tie-break inspects the floor's
/// integer parity; for a non-finite or out-of-`i64`-range floor that parity
/// is meaningless, so we treat such inputs as "not a tie" and round toward
/// the floor, letting [`factor_product`]'s range-check reject them.
fn round_half_to_even_f(x: f64) -> f64 {
  let floor = x.floor();
  let diff = x - floor;
  if diff < 0.5 {
    floor
  } else if diff > 0.5 {
    floor + 1.0
  } else {
    // Exactly .5 — pick the even neighbor. Parity is only meaningful for a
    // floor that fits i64; outside that, fall through to `floor` (the value
    // is rejected downstream by `factor_product` anyway).
    if floor.is_finite() && floor >= i64::MIN as f64 && floor < F64_TWO_POW_63 {
      let floor_i = floor as i64;
      if floor_i % 2 == 0 { floor } else { floor + 1.0 }
    } else {
      floor
    }
  }
}

/// `i64`-returning banker's rounding for the frame-index picker, where the
/// input is always a bounded `linspace` point in `[0, total_frames - 1]`
/// (no factor multiply, so no overflow path). Thin wrapper over
/// [`round_half_to_even_f`].
fn round_half_to_even(x: f64) -> i64 {
  round_half_to_even_f(x) as i64
}

/// Compute a factor-aligned `(height, width)` that preserves aspect ratio
/// while keeping the total pixel count inside `[min_pixels, max_pixels]`.
///
/// Ports python `smart_resize` (`video_generate.py:66-94`); cross-checked
/// against swift `QwenVL.targetSize` (`QwenVL.swift:123-173`). The math:
///
/// 1. Reject when `max(h, w) / min(h, w) > MAX_RATIO` (integer division,
///    matching the python `max(...) / min(...)` truncation only where both
///    references agree — see below).
/// 2. `h_bar = max(factor, round_by_factor(h, factor))`,
///    `w_bar = max(factor, round_by_factor(w, factor))`.
/// 3. If `h_bar * w_bar > max_pixels`, scale **down** by
///    `beta = sqrt(h*w / max_pixels)` and `floor_by_factor`.
/// 4. Else if `h_bar * w_bar < min_pixels`, scale **up** by
///    `beta = sqrt(min_pixels / (h*w))` and `ceil_by_factor`.
///
/// # `beta` (sqrt) path exactness — bounded domain (branch-specific)
/// Python forms each `beta` ratio as an EXACT arbitrary-precision `int / int`
/// (then `math.sqrt`). This port's `beta` is **bit-exact** with the reference
/// **only while the operands of *that branch's* division are `<= 2^53`** (every
/// operand is then f64-representable and IEEE division is correctly-rounded, so
/// the f64 ratio equals python's `int / int -> float` to the bit):
/// - Scale-DOWN bound: `height * width <= 2^53` AND `max_pixels <= 2^53`
///   (operands of `(height * width) / max_pixels`).
/// - Scale-UP bound: `height * width <= 2^53` AND `min_pixels <= 2^53`
///   (operands of `min_pixels / (height * width)`).
///
/// The bounds are **per-branch on purpose**: a single combined `max_pixels`
/// guard would over-reject scale-up — e.g. `smart_resize(1, 1, 28, 3136,
/// i64::MAX)` (small image + huge `max_pixels` sentinel + normal `min_pixels`)
/// enters scale-up, where the actual ratio is `3136 / 1` (both far below
/// `2^53`), so it must succeed and return `(56, 56)` like python. Inputs above
/// the relevant bound are **rejected** with [`Error::ShapeMismatch`] rather
/// than computed imprecisely — a naive `f64 / f64` would double-round and
/// could diverge in the last bit there, and a correctly-rounded big-rational
/// divider is out of scope. Every realistic frame size (dims in the thousands,
/// area in the millions) is far below both bounds and so always matches python
/// exactly.
///
/// # Aspect-ratio check fidelity
/// The python reference computes `max(h, w) / min(h, w)` in **float**
/// (true division) and compares to `MAX_RATIO`. The swift reference uses
/// **integer** division (`max / min > 200`). We follow the python float
/// comparison (this is an mlx-vlm port) so a ratio of, say, `200.5`
/// is correctly rejected where integer division would truncate it to
/// `200` and pass.
///
/// # Errors
/// - [`Error::ShapeMismatch`] if `height <= 0` or `width <= 0` (the
///   reference assumes positive dims; we guard so the `min`/`max` and the
///   `sqrt` domain are well-defined).
/// - [`Error::ShapeMismatch`] if `factor <= 0` (division by / alignment to
///   a non-positive factor is undefined).
/// - [`Error::ShapeMismatch`] if `max_pixels <= 0` or `min_pixels < 0` or
///   `min_pixels > max_pixels` (the reference's pixel budget must be a
///   valid non-empty interval; the `sqrt` arguments must be non-negative).
/// - [`Error::ShapeMismatch`] if `max(h, w) / min(h, w) > MAX_RATIO`
///   (mirrors the python `raise ValueError`).
/// - [`Error::ShapeMismatch`] if the budget **cannot contain a positive
///   factor-aligned size** — i.e. `max_pixels < factor * factor` (the
///   smallest legal output, a `factor × factor` square, already overflows
///   the budget) or the scale-down branch floors a side to a non-positive
///   multiple of `factor`. The python reference omits this guard and
///   silently returns `(0, 0)` for an impossible budget such as
///   `smart_resize(28, 28, 28, 1, 1)` (the `min`/`max` math has no positive
///   factor-aligned solution); we surface it recoverably, naming the
///   budget, rather than emit a zero dimension. A *positive* result whose
///   area lands a single factor-step outside `[min_pixels, max_pixels]` is
///   **kept**, not rejected: the reference's `floor_by_factor` /
///   `ceil_by_factor` do not re-clamp, so re-clamping here would diverge
///   from the python output.
/// - [`Error::ShapeMismatch`] if a rescale (`beta`) is needed but the operands
///   of *that branch's* division exceed `2^53` — see the *`beta` (sqrt) path
///   exactness* note above for the per-branch bounds (scale-down: area /
///   `max_pixels`; scale-up: area / `min_pixels`). The f64 ratio cannot be
///   guaranteed bit-exact with python's `int / int` past that range, so the
///   value is rejected rather than computed imprecisely. (Only reached on the
///   scale-up / scale-down branches; a within-budget input that needs no
///   rescale never trips it.)
pub fn smart_resize(
  height: i64,
  width: i64,
  factor: i64,
  min_pixels: i64,
  max_pixels: i64,
) -> Result<(i64, i64)> {
  if height <= 0 || width <= 0 {
    return Err(Error::ShapeMismatch {
      message: format!("smart_resize: height ({height}) and width ({width}) must be positive"),
    });
  }
  if factor <= 0 {
    return Err(Error::ShapeMismatch {
      message: format!("smart_resize: factor ({factor}) must be positive"),
    });
  }
  if max_pixels <= 0 || min_pixels < 0 || min_pixels > max_pixels {
    return Err(Error::ShapeMismatch {
      message: format!(
        "smart_resize: require 0 <= min_pixels ({min_pixels}) <= max_pixels ({max_pixels}) and max_pixels > 0"
      ),
    });
  }
  // The smallest legal output is a `factor × factor` square (both sides are
  // pinned to at least `factor` by the `max(factor, ...)` guards below).
  // If even that minimal cell overflows `max_pixels`, no positive
  // factor-aligned size fits the budget; the python reference would scale
  // down to `(0, 0)`. Reject up front, naming the budget.
  //
  // EXACT i128 arithmetic — NOT f64. `factor` can be any positive `i64`, so
  // `factor * factor` can overflow `i64` (debug panic / release wrap) and is
  // *inexact* in f64 (the 53-bit mantissa collapses adjacent values: e.g.
  // `factor=3_037_000_499` and `max_pixels=factor*factor-1` both round to the
  // same f64, silently bypassing the guard and returning an over-budget size).
  // `i128` represents every `i64²` product exactly and overflow-free, so the
  // comparison and the error message are both exact for any positive `i64`.
  let min_cell = (factor as i128) * (factor as i128);
  if min_cell > max_pixels as i128 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "smart_resize: budget max_pixels ({max_pixels}) cannot contain the smallest \
         factor-aligned size ({factor}x{factor} = {min_cell})"
      ),
    });
  }

  let hi = height.max(width) as f64;
  let lo = height.min(width) as f64;
  if hi / lo > MAX_RATIO as f64 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "smart_resize: absolute aspect ratio must be < {MAX_RATIO}, got {}",
        hi / lo
      ),
    });
  }

  // Compute the factor-aligned bars BEFORE the branch decision. The
  // `*_by_factor` helpers are overflow-safe (`?`-propagating
  // [`Error::ShapeMismatch`]): an adversarial near-`i64::MAX` `height`/
  // `width` with `factor=28` would otherwise panic (debug) or wrap negative
  // (release) inside `round_by_factor`, after which `factor.max(..)` would
  // promote the wrapped value to a small valid-looking size and the late
  // `checked_mul` would never see the overflow. Surfacing it here, before
  // any `h_bar * w_bar` comparison, closes that silent-corruption path.
  let mut h_bar = factor.max(round_by_factor(height, factor)?);
  let mut w_bar = factor.max(round_by_factor(width, factor)?);

  // python branches on `h_bar * w_bar > max_pixels` / `< min_pixels` with
  // arbitrary-precision ints — i.e. EXACT integer comparisons. Mirror that
  // with `i128` (NOT f64): both bars are positive multiples of `factor` here,
  // so their product can exceed `i64` and is inexact in f64 (the 53-bit
  // mantissa collapses adjacent integers, which could pick the wrong branch
  // or skip a needed rescale on an adversarial budget). `i128` holds every
  // `i64 * i64` exactly and overflow-free.
  let bar_area = (h_bar as i128) * (w_bar as i128);
  if bar_area > max_pixels as i128 {
    // SCALE DOWN — python: beta = sqrt((height * width) / max_pixels), then
    // floor_by_factor(height / beta, factor). `beta`'s sqrt is the only float
    // math, mirroring python's `math.sqrt(...)`. Bound the f64 domain of THIS
    // branch's actual operands — `height * width` (numerator) and `max_pixels`
    // (denominator); `min_pixels` does not appear in the scale-down ratio — so
    // the division is bit-exact with the reference (see
    // `check_beta_domain_down`).
    check_beta_domain_down(height, width, max_pixels)?;
    let hw = height as f64 * width as f64;
    let beta = (hw / max_pixels as f64).sqrt();
    // python: floor_by_factor(height / beta, factor) where the argument is
    // the FLOAT `height / beta` — use the float-input helper so we don't
    // truncate before the `floor(_ / factor)` (a double-floor would diverge).
    h_bar = floor_by_factor_f(height as f64 / beta, factor)?;
    w_bar = floor_by_factor_f(width as f64 / beta, factor)?;
  } else if bar_area < min_pixels as i128 {
    // SCALE UP — python: beta = sqrt(min_pixels / (height * width)), then
    // ceil_by_factor(height * beta, factor). Bound the f64 domain of THIS
    // branch's actual operands — `min_pixels` (numerator) and `height * width`
    // (denominator); `max_pixels` does not appear in the scale-up ratio, so a
    // huge `max_pixels` sentinel must NOT over-reject here (e.g.
    // `smart_resize(1, 1, 28, 3136, i64::MAX)` is legitimate scale-up that
    // python returns `(56, 56)` for). See `check_beta_domain_up`.
    check_beta_domain_up(height, width, min_pixels)?;
    let hw = height as f64 * width as f64;
    let beta = (min_pixels as f64 / hw).sqrt();
    h_bar = ceil_by_factor_f(height as f64 * beta, factor)?;
    w_bar = ceil_by_factor_f(width as f64 * beta, factor)?;
  }

  // Validate the result is a *positive* factor-aligned size whose pixel
  // product is representable (overflow-safe `checked_mul`). The `*_by_factor`
  // helpers keep both sides multiples of `factor`, but a tight `max_pixels`
  // paired with an extreme aspect ratio can still floor one side to 0 in the
  // scale-down branch (e.g. height=537, width=4962, factor=28, max=1646 ->
  // (0, 112)), which the early `min_cell` guard does not catch. A non-positive
  // dimension is the degenerate `(0, 0)`-style result an impossible budget
  // produces, so reject it instead of returning a zero / out-of-budget dim.
  //
  // We deliberately do NOT reject a positive result whose area merely sits a
  // rounding step *outside* `[min_pixels, max_pixels]`: the reference applies
  // `floor_by_factor` / `ceil_by_factor` *without* re-clamping, so on many
  // satisfiable budgets it faithfully returns a positive size whose product is
  // marginally below `min_pixels` (scale-down floor) or above `max_pixels`
  // (scale-up ceil). Erroring on those would diverge from the python output we
  // are porting (verified: ~0.2% of valid inputs land just outside the band by
  // one factor step). Faithfulness requires accepting them.
  let representable = h_bar.checked_mul(w_bar).is_some();
  if h_bar <= 0 || w_bar <= 0 || !representable {
    return Err(Error::ShapeMismatch {
      message: format!(
        "smart_resize: no positive factor-aligned size fits budget \
         [min_pixels={min_pixels}, max_pixels={max_pixels}] for input \
         ({height}x{width}, factor={factor}); computed ({h_bar}x{w_bar})"
      ),
    });
  }
  Ok((h_bar, w_bar))
}

/// How to choose the number of frames to sample: either a fixed count or
/// an fps-driven budget. Mirrors the two mutually-exclusive branches of
/// python `smart_nframes` (`video_generate.py:161-192`), which asserts
/// `not ("fps" in ele and "nframes" in ele)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameSampling {
  /// Fixed frame count — python `ele["nframes"]` branch
  /// (`video_generate.py:173-174`). Rounded to a multiple of
  /// [`FRAME_FACTOR`].
  Fixed {
    /// Requested frame count (pre factor-rounding).
    nframes: i64,
  },
  /// fps-driven budget — python `ele.get("fps", FPS)` branch
  /// (`video_generate.py:176-187`).
  Fps {
    /// Target sampling rate (python `ele.get("fps", FPS)`). Use [`FPS`]
    /// for the reference default.
    fps: f64,
    /// Floor on the result before factor-rounding (python
    /// `ele.get("min_frames", FPS_MIN_FRAMES)`, then `ceil_by_factor`).
    /// Use [`FPS_MIN_FRAMES`] for the reference default.
    min_frames: i64,
    /// Ceiling on the result before factor-rounding (python
    /// `ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames))`, then
    /// `floor_by_factor`). `None` selects the reference default
    /// `min(FPS_MAX_FRAMES, total_frames)`.
    max_frames: Option<i64>,
  },
}

impl Default for FrameSampling {
  /// The python default path: fps-driven with [`FPS`] /
  /// [`FPS_MIN_FRAMES`] and the `max_frames` default
  /// (`min(FPS_MAX_FRAMES, total_frames)`).
  fn default() -> Self {
    Self::Fps {
      fps: FPS,
      min_frames: FPS_MIN_FRAMES,
      max_frames: None,
    }
  }
}

/// Compute how many frames to sample from a video as model input.
///
/// Ports python `smart_nframes` (`video_generate.py:161-192`) and is
/// cross-checked against the swift `estimatedFrames = round(fps *
/// duration); min(_, maxFrames); max(_, 1)` clamp in
/// `MediaProcessing._asProcessedSequence` (`MediaProcessing.swift:
/// 423-480`).
///
/// - [`FrameSampling::Fixed`]: `round_by_factor(nframes, FRAME_FACTOR)`.
/// - [`FrameSampling::Fps`]: `nframes = total_frames / video_fps * fps`,
///   then clamped to `[min_frames, max_frames]` (each factor-rounded —
///   `min` via `ceil_by_factor`, `max` via `floor_by_factor`) **and** to
///   `total_frames`, then `floor_by_factor(_, FRAME_FACTOR)`.
///
/// Both branches finally assert `FRAME_FACTOR <= nframes <= total_frames`
/// (python raises `ValueError`; we return [`Error::ShapeMismatch`]).
///
/// # Errors
/// - [`Error::ShapeMismatch`] if `total_frames <= 0` or `video_fps <= 0`
///   (the python flow assumes a real, non-empty clip; a zero `video_fps`
///   would divide-by-zero in the fps branch).
/// - [`Error::ShapeMismatch`] if the `fps` is non-positive (the budget
///   `total/video_fps*fps` would be non-positive).
/// - [`Error::ShapeMismatch`] if the resulting count falls outside
///   `[FRAME_FACTOR, total_frames]` — mirrors the python final check.
pub fn smart_nframes(sampling: FrameSampling, total_frames: i64, video_fps: f64) -> Result<i64> {
  if total_frames <= 0 {
    return Err(Error::ShapeMismatch {
      message: format!("smart_nframes: total_frames ({total_frames}) must be positive"),
    });
  }
  let nframes = match sampling {
    // Overflow-safe `?`: an adversarial near-`i64::MAX` `nframes` would
    // otherwise panic (debug) / wrap (release) inside `round_by_factor`.
    FrameSampling::Fixed { nframes } => round_by_factor(nframes, FRAME_FACTOR)?,
    FrameSampling::Fps {
      fps,
      min_frames,
      max_frames,
    } => {
      // Reject non-positive AND non-finite (NaN/inf) rates: a zero
      // `video_fps` divides-by-zero and a non-positive/NaN budget would
      // break the clamp + the final `[FRAME_FACTOR, total_frames]` check.
      if !video_fps.is_finite() || video_fps <= 0.0 {
        return Err(Error::ShapeMismatch {
          message: format!(
            "smart_nframes: video_fps ({video_fps}) must be a positive finite number"
          ),
        });
      }
      if !fps.is_finite() || fps <= 0.0 {
        return Err(Error::ShapeMismatch {
          message: format!("smart_nframes: fps ({fps}) must be a positive finite number"),
        });
      }
      // Overflow-safe `?` on every factor-rounding (the clamp bounds and the
      // final round): an adversarial `min_frames`/`max_frames`/derived count
      // near `i64::MAX` would otherwise panic/wrap in the `*_by_factor` math.
      let min_frames = ceil_by_factor(min_frames, FRAME_FACTOR)?;
      // python: max_frames default = min(FPS_MAX_FRAMES, total_frames).
      let max_frames_raw = max_frames.unwrap_or_else(|| FPS_MAX_FRAMES.min(total_frames));
      let max_frames = floor_by_factor(max_frames_raw, FRAME_FACTOR)?;
      // python: nframes = total_frames / video_fps * fps   (float)
      let raw = total_frames as f64 / video_fps * fps;
      // python: min(min(max(nframes, min_frames), max_frames), total_frames)
      // performed in float, THEN floor_by_factor. The clamp pins the result
      // into [min_frames, total_frames] before the (overflow-checked) round.
      let clamped = raw
        .max(min_frames as f64)
        .min(max_frames as f64)
        .min(total_frames as f64);
      floor_by_factor(clamped.floor() as i64, FRAME_FACTOR)?
    }
  };
  if !(FRAME_FACTOR <= nframes && nframes <= total_frames) {
    return Err(Error::ShapeMismatch {
      message: format!(
        "smart_nframes: nframes should be in [{FRAME_FACTOR}, {total_frames}], got {nframes}"
      ),
    });
  }
  Ok(nframes)
}

/// The `nframes` source-frame indices to sample, evenly spread across
/// `[0, total_frames - 1]`.
///
/// Ports python `np.linspace(0, total_frames - 1, nframes).round().
/// astype(int)` (`video_generate.py:217`). This is the index picker the
/// caller uses to decide *which* decoded frames to hand to
/// [`process_frames`]; it is codec-free (decoding stays the caller's job).
///
/// numpy computes `linspace` as `step = (stop - start) / (num - 1)` FIRST,
/// then point `i` as `start + step * i`, and finally (because the default
/// `endpoint=True`) overwrites the LAST sample with `stop` exactly. We
/// replicate that operation order bit-for-bit rather than the algebraically
/// equal `(n-1) * i / (k-1)`: the intermediate `step` changes which float
/// ties land exactly on `.5` before banker's rounding, so the orders are
/// NOT interchangeable. Example: `total_frames = 26, nframes = 23`, the
/// midpoint `i = 11` is `12.500000000000002` the numpy way (rounds to 13)
/// but exactly `12.5` the fused way (banker's-rounds to 12) — a silently
/// wrong frame. The single-sample `k == 1` case yields `[0]` (numpy's
/// `linspace` with `num=1` returns just the start). Each value is rounded
/// **half-to-even** (numpy's `ndarray.round`) and clamped into `[0,
/// total_frames - 1]` for safety against float-edge overshoot.
///
/// # Errors
/// - [`Error::ShapeMismatch`] if `total_frames <= 0` or `nframes <= 0`.
/// - [`Error::OutOfMemory`] if the `nframes`-long index buffer cannot be
///   allocated (request-scaled; surfaced recoverably rather than aborting).
pub fn sample_frame_indices(total_frames: i64, nframes: i64) -> Result<Vec<i64>> {
  if total_frames <= 0 {
    return Err(Error::ShapeMismatch {
      message: format!("sample_frame_indices: total_frames ({total_frames}) must be positive"),
    });
  }
  if nframes <= 0 {
    return Err(Error::ShapeMismatch {
      message: format!("sample_frame_indices: nframes ({nframes}) must be positive"),
    });
  }
  let n = nframes as usize;
  let mut out = crate::error::try_with_capacity::<i64>(n)?;
  let stop = (total_frames - 1) as f64;
  if nframes == 1 {
    // numpy linspace(start, stop, num=1) -> [start]
    out.push(0);
    return Ok(out);
  }
  // numpy: step = (stop - start) / (num - 1), computed BEFORE the per-point
  // multiply (start == 0 here). Reusing this single `step` for every point
  // reproduces numpy's float ties exactly (see the doc comment / the
  // `total_frames=26, nframes=23` regression).
  let step = stop / (nframes - 1) as f64;
  for i in 0..nframes {
    // endpoint=True: numpy overwrites the final sample with `stop` exactly
    // rather than trusting `step * (n-1)` to land on it.
    let v = if i == nframes - 1 {
      stop
    } else {
      step * (i as f64)
    };
    let idx = round_half_to_even(v).clamp(0, total_frames - 1);
    out.push(idx);
  }
  Ok(out)
}

/// Apply the cross-model image [`preprocess`]
/// to each already-decoded `frame` and stack into `[T, H, W, 3]`.
///
/// Ports the per-frame resize+stack body of python `fetch_video`
/// (`video_generate.py:270-282`): each frame is resized + converted to a
/// float tensor and the frames are `np.stack(..., axis=0)`'d. We reuse the
/// existing cross-model [`preprocess`] (so
/// the resize/rescale/normalize is byte-identical to single-image VLM
/// preprocessing) and [`stack`] along a new
/// leading time axis.
///
/// **Decoding is the caller's responsibility** (see the module doc): pass
/// the frames you have already decoded. To choose *which* source frames to
/// decode, use [`smart_nframes`] + [`sample_frame_indices`]; to choose the
/// resize target, use [`smart_resize`] and set `cfg.size` accordingly
/// before calling.
///
/// **Layout:** channel-last `[T, H, W, 3]` (the natural stack of the
/// channel-last `[H, W, 3]` [`preprocess`]
/// output). The python planar `[T, C, H, W]` (`video_generate.py:282`) is
/// a per-model contract reachable in one lazy `transpose_axes(&[0, 3, 1,
/// 2])`; see the module `Channel layout` block.
///
/// **`cfg.layout` constraint — only [`Layout::Hwc`] is currently
/// supported.** `process_frames` produces a channel-last `[T, H, W, 3]`
/// stack and the *video-tensor* analogues of [`Layout::Chw`] /
/// [`Layout::Bchw`] (i.e. whether a video tensor should be
/// `[T, 3, H, W]`, `[1, T, 3, H, W]`, or something else) are not yet
/// pinned to a per-model contract, so applying [`Layout`] **per-frame**
/// here would silently break the stack contract above. Passing
/// `cfg.layout != Layout::Hwc` therefore returns
/// [`Error::Backend`] rather than producing a misleading shape. Callers
/// that need a planar video tensor can post-process the returned
/// `[T, H, W, 3]` themselves (one lazy `transpose_axes(&[0, 3, 1, 2])`)
/// until a future PR defines first-class video-layout semantics.
///
/// **No implicit eval:** every frame composes lazily; the returned
/// `Array` is un-evaluated.
///
/// # Errors
/// - [`Error::ShapeMismatch`] if `frames` is empty (python `np.stack`
///   raises on an empty sequence; the swift `_asProcessedSequence` has a
///   `precondition(videoFrames.isEmpty == false)`).
/// - [`Error::Backend`] if `cfg.layout != Layout::Hwc` (see the
///   `cfg.layout` constraint above).
/// - Any error from [`preprocess`] on a
///   frame (dtype/shape/allocation) is propagated.
/// - [`Error::OutOfMemory`] if the per-frame `Array` handle vector cannot
///   be allocated (request-scaled).
pub fn process_frames(
  frames: &[::image::DynamicImage],
  cfg: &ImageProcessorConfig,
) -> Result<Array> {
  if frames.is_empty() {
    return Err(Error::ShapeMismatch {
      message: "process_frames: frames slice is empty".into(),
    });
  }
  // Reject non-Hwc per-frame layouts. Applying a planar `Layout` per
  // frame here would silently break the documented `[T, H, W, 3]` stack
  // contract (e.g. `Layout::Chw` would yield `[T, 3, H, W]` and
  // `Layout::Bchw` would yield a rank-5 `[T, 1, 3, H, W]` that matches
  // no standard video layout). Video-tensor layout semantics are not
  // yet defined; callers wanting planar output should post-process the
  // returned `[T, H, W, 3]` themselves.
  if cfg.layout != Layout::Hwc {
    return Err(Error::Backend {
      message: format!(
        "process_frames currently only supports Layout::Hwc per-frame configs; \
         got {:?}. Video-tensor layout for Chw/Bchw is not yet defined — \
         use Layout::Hwc for video inputs (and post-process the returned \
         [T, H, W, 3] if a planar shape is needed) or wait for a future PR \
         adding video layout semantics.",
        cfg.layout
      ),
    });
  }
  // Preprocess every frame to a channel-last [H, W, 3] f32 Array. The
  // owned `Array`s must outlive the `&Array` borrow slice handed to
  // `stack`, so collect them first (request-scaled → recoverable alloc).
  let mut processed = crate::error::try_with_capacity::<Array>(frames.len())?;
  for frame in frames {
    processed.push(preprocess(frame, cfg)?);
  }
  // `stack` borrows each Array; build the borrow slice (request-scaled).
  let mut refs = crate::error::try_with_capacity::<&Array>(processed.len())?;
  for a in &processed {
    refs.push(a);
  }
  stack(&refs)
}
