//! SigLIP2 NaFlex image preprocessing: aspect-preserving resize to a
//! patch budget, normalize, and patchify into the flat
//! `(num_patches, num_channels * patch_size^2)` tensor the vision tower
//! consumes, plus the `spatial_shapes (H_patch, W_patch)` and
//! `pixel_attention_mask` side outputs.
//!
//! ## Algorithm (matches upstream + the `siglip2-naflex` oracle)
//!
//! This is a port of the NaFlex preprocessing in the user's published
//! `siglip2-naflex` crate (`src/preproc/naflex.rs`), which is itself a
//! port of `transformers`'
//! `image_processing_siglip2_fast.get_image_size_for_max_num_patches`
//! and validated against PyTorch:
//!
//! 1. **Target-size selection** ([`patch_grid`]) — binary-search the
//!    largest uniform scale `s` such that, after rounding each axis up to
//!    a multiple of `patch_size`, the patch grid `H_p x W_p` fits within
//!    `max_num_patches`. This is the **authoritative** sizing formula
//!    (its `SCALE_EPS = 1e-5` termination is what matches upstream
//!    exactly on edge aspect ratios — do not loosen it).
//! 2. **Resize** — aspect-preserving resize to
//!    `(H_p * patch_size, W_p * patch_size)` via the NEON-accelerated,
//!    PIL-bit-exact [`crate::vlm::image::resize`] with
//!    [`ResizeFilter::Bilinear`] (PIL `Image.BILINEAR`, the upstream HF
//!    SigLIP2 processor's resampling filter).
//! 3. **Normalize + patchify** — map each `u8` channel through
//!    `x / 127.5 - 1.0` (the SigLIP `(x/255 - 0.5)/0.5` mean/std=0.5
//!    rescale) and flatten into `(num_patches, P^2 * C)` rows in
//!    `(row, col, channel-innermost)` order — no axis transposition —
//!    right-padded with zero rows to `max_num_patches`.
//! 4. **Side outputs** — `pixel_attention_mask` is `1` for the first
//!    `H_p * W_p` rows and `0` for the padding; `spatial_shapes` is
//!    `[H_p, W_p]`.
//!
//! ## Divergence from the `siglip2-naflex` crate's resize backend
//!
//! The user crate resizes with `image`-rs's `FilterType::Triangle`,
//! whereas this port uses `crate::vlm::image::resize`'s **PIL-bit-exact**
//! bilinear. The two are both "bilinear" but are not byte-identical
//! (`image`-rs's triangle vs PIL `Resample.c`). The PIL path is the
//! correct parity target here because the upstream HF processor (which
//! the PyTorch reference fixtures are generated from) resamples with PIL
//! `Image.BILINEAR`. The **sizing** math ([`patch_grid`]) is identical to
//! the oracle.
//!
//! ## Bounds / errors
//!
//! Every count / dimension is checked before any allocation; degenerate
//! inputs (zero or overflowing dimensions, a wrong-length RGB slice, an
//! image whose minimum grid exceeds the budget) return a typed
//! [`crate::Error`] rather than panicking.

use crate::{
  array::Array,
  embeddings::siglip2_naflex::config::MAX_PIXEL_ELEMENTS,
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, Error, InvariantViolationPayload,
    LengthMismatchPayload, OutOfRangePayload, Result,
  },
  model_validation::{Extent, elem_count, reserve_or_error},
  vlm::image::{ResizeFilter, resize},
};

/// The fixed channel count of the patchify path. The SigLIP2 NaFlex
/// preprocessing normalizes per-RGB-channel and flattens into
/// `num_channels * patch_size^2` rows; only the 3-channel (RGB) path is wired,
/// and the config validator pins `num_channels == 3`. The exported
/// [`preprocess`] re-checks this (it bypasses config validation) and derives
/// every patchify stride from the actual RGB buffer format rather than the raw
/// `num_channels` parameter, so a non-3 argument is a typed error, never an
/// out-of-bounds slice into the always-3-channel resized buffer.
const RGB_CHANNELS: u32 = 3;

/// Per-axis cap on the source image `width` / `height` (px). A single absurd
/// dimension is rejected at [`Extent`] construction, before it reaches the
/// source-clone sizing.
const MAX_SOURCE_DIM: usize = 1 << 16;

/// Total-element cap on the decoded source RGB buffer
/// (`width * height * channels` bytes, ~512 MiB). Independent of the output
/// patch budget ([`MAX_PIXEL_ELEMENTS`]): the source can dwarf the resized
/// output (a large image downscaled to a few patches), so it carries its own
/// magnitude bound so a hostile caller cannot force an unbounded source clone.
const MAX_SOURCE_PIXELS: usize = 1 << 29;

/// Tolerance for the [`patch_grid`] binary-search termination. Mirrors
/// the `siglip2-naflex` crate's `SCALE_EPS` (= the upstream
/// `transformers` value). **Do not loosen**: it is what keeps `s` clearly
/// below the boundary where `ceil()` flips, the difference between
/// emitting upstream-equivalent grids vs silently drifting by one patch
/// in the longer axis on edge inputs.
const SCALE_EPS: f64 = 1e-5;

/// Upper search bound on the uniform scale. Mirrors the oracle's
/// `scale_max = 100.0`.
const SCALE_MAX: f64 = 100.0;

/// Find the largest uniform scale `s` such that, after rounding each axis
/// up to a multiple of `patch_size`, the patch grid `H_p x W_p` fits
/// within `max_num_patches`. Returns `(H_p, W_p)` with both `>= 1`.
///
/// Direct port of the `siglip2-naflex` crate's `patch_grid` (a port of
/// upstream `get_image_size_for_max_num_patches`): binary-search
/// `s ∈ [SCALE_EPS/10, SCALE_MAX]`, terminating at `hi - lo < SCALE_EPS`.
///
/// `patch_size` and `max_num_patches` must be `> 0` (the caller validates
/// this from the config); `width` / `height` must be `> 0`. The returned
/// product is **not** guaranteed `<= max_num_patches` for adversarial
/// `u32` inputs whose entire feasible scale range falls below the
/// `SCALE_EPS/10` search floor (e.g. `width` near `u32::MAX` at
/// `height = 1`) — the caller ([`preprocess`]) re-checks the budget
/// post-hoc and rejects such inputs with [`Error::CapExceeded`].
pub fn patch_grid(height: u32, width: u32, patch_size: u32, max_num_patches: u32) -> (u32, u32) {
  let h = f64::from(height);
  let w = f64::from(width);
  let p = f64::from(patch_size);
  let m = f64::from(max_num_patches);

  // Round `scale * original` up to a multiple of `P`, then floor at `P`
  // so the smallest target is one full patch (matches upstream).
  fn scaled_pixel_size(scale: f64, original: f64, patch: f64) -> f64 {
    let scaled = scale * original;
    let scaled = (scaled / patch).ceil() * patch;
    scaled.max(patch)
  }

  let mut scale_min: f64 = SCALE_EPS / 10.0;
  let mut scale_max: f64 = SCALE_MAX;
  while (scale_max - scale_min) >= SCALE_EPS {
    let scale = 0.5 * (scale_min + scale_max);
    let target_h = scaled_pixel_size(scale, h, p);
    let target_w = scaled_pixel_size(scale, w, p);
    let num_patches = (target_h * target_w) / (p * p);
    if num_patches <= m {
      scale_min = scale;
    } else {
      scale_max = scale;
    }
  }

  let target_h = scaled_pixel_size(scale_min, h, p);
  let target_w = scaled_pixel_size(scale_min, w, p);
  let h_p = ((target_h / p) as u32).max(1);
  let w_p = ((target_w / p) as u32).max(1);
  (h_p, w_p)
}

/// The preprocessed NaFlex inputs for one image, as MLX device arrays
/// ready for the vision tower.
///
/// All three are produced lazily (built from host buffers via
/// [`Array::from_slice`]); no implicit eval. The fixed leading dimension
/// is `max_num_patches`.
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
#[derive(Debug)]
pub struct NaflexInputs {
  /// `(max_num_patches, num_channels * patch_size^2)` f32 — the flattened
  /// patch rows, normalized `x/127.5 - 1.0`, zero-padded past the active
  /// `H_p * W_p` rows.
  pub pixel_values: Array,
  /// `(max_num_patches,)` i32 — `1` for the first `H_p * W_p` rows, `0`
  /// for padding.
  pub pixel_attention_mask: Array,
  /// `(2,)` i32 — `[H_p, W_p]`, the patch grid dimensions.
  pub spatial_shapes: Array,
}

/// Reject a dimension outside `(0, cap]`.
fn require_positive_u32(context: &'static str, value: u32) -> Result<()> {
  if value == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "must be a positive dimension (> 0)",
      "0",
    )));
  }
  Ok(())
}

/// NaFlex-preprocess one interleaved RGB image into [`NaflexInputs`].
///
/// `rgb` is `width * height * 3` row-major interleaved RGB bytes (no row
/// padding). `patch_size`, `num_channels`, and `max_num_patches` come
/// from the validated [`crate::embeddings::siglip2_naflex::config::VisionConfig`]
/// (`num_channels` must be `3` — the patchify path is RGB-only).
///
/// Returns the flattened `(max_num_patches, P^2 * C)` pixel tensor plus
/// the attention mask and spatial shapes (see [`NaflexInputs`]).
///
/// ## Errors
/// - `width == 0` / `height == 0` / `patch_size == 0` /
///   `max_num_patches == 0` → [`Error::OutOfRange`].
/// - `num_channels != 3` → [`Error::InvariantViolation`] (the patchify path
///   is RGB-only; the resized buffer is always 3-channel, so a different
///   stride would slice out of bounds). The exported entry point re-checks
///   this even though the config validator pins it.
/// - `width` / `height` exceeds the per-axis cap `MAX_SOURCE_DIM`, or
///   `width * height * 3` overflows `usize` or exceeds the source-bytes cap
///   `MAX_SOURCE_PIXELS`, or `rgb.len()` disagrees with the byte count →
///   [`Error::CapExceeded`] / [`Error::ArithmeticOverflow`] /
///   [`Error::LengthMismatch`].
/// - the `pixel_values` element count `max_num_patches * (3 * patch_size^2)`
///   exceeds the `MAX_PIXEL_ELEMENTS` product cap (a hostile
///   `patch_size`/`max_num_patches` pair each within its own cap but whose
///   product is oversized) → [`Error::CapExceeded`].
/// - the selected grid `H_p * W_p` exceeds `max_num_patches` (an
///   adversarial dimension whose feasible scale range is below the search
///   floor) → [`Error::CapExceeded`].
/// - a within-cap but heavyweight `pixel_values` / mask reservation the
///   allocator cannot satisfy → [`Error::AllocFailure`].
/// - underlying [`crate::vlm::image::resize`] / [`Array::from_slice`]
///   errors propagate.
pub fn preprocess(
  rgb: &[u8],
  width: u32,
  height: u32,
  patch_size: u32,
  num_channels: u32,
  max_num_patches: u32,
) -> Result<NaflexInputs> {
  require_positive_u32("siglip2 preprocess: width", width)?;
  require_positive_u32("siglip2 preprocess: height", height)?;
  require_positive_u32("siglip2 preprocess: patch_size", patch_size)?;
  require_positive_u32("siglip2 preprocess: max_num_patches", max_num_patches)?;
  // The patchify path is RGB-only: it constructs a 3-channel `RgbImage`,
  // resizes into an always-3-channel buffer, and uses the channel count as the
  // patchify stride into that buffer. A `num_channels != 3` (the exported entry
  // bypasses the config's `num_channels == 3` pin) would slice the 3-channel
  // resized buffer with a wrong stride and read out of bounds — reject it as a
  // typed error before any sizing. Every stride below is derived from
  // `RGB_CHANNELS`, not the raw parameter, so the patchify cannot go OOB.
  if num_channels != RGB_CHANNELS {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "siglip2 preprocess: num_channels",
      "must be 3 (the patchify path is RGB-only)",
    )));
  }

  // Expected RGB byte count via the resource layer: each axis is an `Extent`
  // (per-axis capped at construction) and `elem_count` is the checked product
  // against the total source-bytes cap `MAX_SOURCE_PIXELS`. This bounds the
  // SOURCE magnitude — independent of the output patch budget — so a hostile
  // `width` / `height` cannot drive the source clone below to an unbounded
  // duplicate. The channel factor is the fixed `RGB_CHANNELS` (== `num_channels`,
  // verified above), so the slice-length contract matches the always-3-channel
  // buffer every stride below indexes.
  let channels = RGB_CHANNELS as usize;
  let expected_rgb_len = elem_count(
    "siglip2 preprocess: rgb byte count (width * height * channels)",
    &[
      Extent::new("siglip2 preprocess: width", width as usize, MAX_SOURCE_DIM)?,
      Extent::new(
        "siglip2 preprocess: height",
        height as usize,
        MAX_SOURCE_DIM,
      )?,
      Extent::new(
        "siglip2 preprocess: channels",
        channels,
        RGB_CHANNELS as usize,
      )?,
    ],
    MAX_SOURCE_PIXELS,
  )?;
  if rgb.len() != expected_rgb_len {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "siglip2 preprocess: rgb slice length vs width*height*channels",
      expected_rgb_len,
      rgb.len(),
    )));
  }

  // Per-patch flattened width = C * P^2; the `pixel_values` element count is
  // `max_num_patches * per_patch`. Both factors can be within their own config
  // cap yet their product oversized (a `patch_size` near 591 keeps `per_patch`
  // just under the width cap while `max_num_patches = 4096` passes the
  // cardinality cap → ~4.29e9 f32 / ~16 GiB). Compute both (overflow-checked so
  // a hostile config cannot wrap the allocation size) and cap the product by
  // `MAX_PIXEL_ELEMENTS` BEFORE any allocation — this also bounds the
  // intermediate resize buffer, whose `h_p*w_p*patch_size^2*3 <= total_floats`.
  // Mirrors `VisionConfig::validate` (the exported entry bypasses it).
  let p = patch_size as usize;
  let per_patch = p
    .checked_mul(p)
    .and_then(|pp| pp.checked_mul(channels))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "siglip2 preprocess: per-patch width (patch_size^2 * channels)",
        "usize",
        [
          ("patch_size", u64::from(patch_size)),
          ("channels", RGB_CHANNELS as u64),
        ],
      ))
    })?;
  let total_floats = (max_num_patches as usize)
    .checked_mul(per_patch)
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "siglip2 preprocess: pixel_values size (max_num_patches * per_patch)",
        "usize",
        [
          ("max_num_patches", u64::from(max_num_patches)),
          ("per_patch", per_patch as u64),
        ],
      ))
    })?;
  if total_floats as u64 > MAX_PIXEL_ELEMENTS as u64 {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "siglip2 preprocess: pixel_values element count (max_num_patches * patch_feature_dim)",
      "MAX_PIXEL_ELEMENTS",
      MAX_PIXEL_ELEMENTS as u64,
      total_floats as u64,
    )));
  }

  // 1. Target patch grid (the authoritative oracle formula).
  let (h_p, w_p) = patch_grid(height, width, patch_size, max_num_patches);

  // Postcondition: the binary search assumes its `scale_min = eps/10`
  // floor is feasible; for adversarial `u32` inputs (e.g. width near
  // u32::MAX at height 1) the entire feasible range can fall below that
  // floor, leaving a grid above budget. Re-check before sizing the fixed
  // pixel buffer so we never index out of bounds.
  let grid_patches = u64::from(h_p).checked_mul(u64::from(w_p)).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "siglip2 preprocess: grid patch count (H_p * W_p)",
      "u64",
      [("H_p", u64::from(h_p)), ("W_p", u64::from(w_p))],
    ))
  })?;
  if grid_patches > u64::from(max_num_patches) {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "siglip2 preprocess: selected patch grid",
      "max_num_patches",
      u64::from(max_num_patches),
      grid_patches,
    )));
  }

  // Resized pixel dimensions. `H_p, W_p <= max_num_patches` and
  // `patch_size` is small, but keep the multiply checked.
  let h_res = h_p.checked_mul(patch_size).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "siglip2 preprocess: resized height (H_p * patch_size)",
      "u32",
      [
        ("H_p", u64::from(h_p)),
        ("patch_size", u64::from(patch_size)),
      ],
    ))
  })?;
  let w_res = w_p.checked_mul(patch_size).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "siglip2 preprocess: resized width (W_p * patch_size)",
      "u32",
      [
        ("W_p", u64::from(w_p)),
        ("patch_size", u64::from(patch_size)),
      ],
    ))
  })?;

  // 2. Aspect-preserving resize via the PIL-bit-exact NEON resize.
  //    Build a borrowed-free owned RgbImage over the caller's bytes
  //    (image 0.25's `from_raw` needs an owned container), then resize.
  //    Only the RGB (3-channel) path is wired (the patchify normalize is
  //    per-RGB-channel); `num_channels` was verified `== 3` above and every
  //    stride below derives from the resulting 3-channel `to_rgb8()` buffer.
  // Fallible source clone: `image::ImageBuffer::from_raw` needs an owned
  // container, but a bare `rgb.to_vec()` aborts on a within-cap-but-heavyweight
  // duplicate. Reserve fallibly (typed `AllocFailure`) into the exact,
  // already-capped `expected_rgb_len`, then copy — no infallible duplicate.
  let mut src_buf: Vec<u8> = Vec::new();
  reserve_or_error(
    &mut src_buf,
    "siglip2 preprocess: source RGB clone",
    expected_rgb_len,
  )?;
  src_buf.extend_from_slice(rgb);
  let src_img: ::image::RgbImage = ::image::ImageBuffer::from_raw(width, height, src_buf)
    .ok_or_else(|| {
      Error::LengthMismatch(LengthMismatchPayload::new(
        "siglip2 preprocess: RgbImage::from_raw length",
        expected_rgb_len,
        rgb.len(),
      ))
    })?;
  let src_dyn = ::image::DynamicImage::ImageRgb8(src_img);
  // `vlm::resize` takes target as (height, width).
  let resized_dyn = resize(&src_dyn, (h_res, w_res), ResizeFilter::Bilinear)?;
  let resized = resized_dyn.to_rgb8();

  // 3. Normalize + patchify into the fixed (max_num_patches, P^2*C) buffer,
  //    zero-padded past the active rows. `per_patch` / `total_floats` were
  //    overflow-checked AND product-capped above. Build the buffer FALLIBLY
  //    (`reserve_or_error` → typed `AllocFailure`) then zero-fill via `resize`
  //    (which cannot reallocate — the exact capacity is already reserved),
  //    replacing the infallible `vec![0.0; total_floats]` that aborts on a
  //    within-cap but heavyweight request. The padding rows past the active
  //    `H_p*W_p` patches stay at the zero fill.
  let mut pixel_values: Vec<f32> = Vec::new();
  reserve_or_error(
    &mut pixel_values,
    "siglip2 pixel_values f32 elements",
    total_floats,
  )?;
  pixel_values.resize(total_floats, 0.0f32);

  let resized_buf = resized.as_raw();
  // Every stride is derived from the actual RGB buffer format (`channels` ==
  // `RGB_CHANNELS` == 3), never the raw `num_channels` parameter, so the
  // patchify cannot index past the always-3-channel `resized_buf`.
  let c = channels;
  let src_row_stride = (w_res as usize) * c; // bytes per resized row
  let row_bytes = p * c; // bytes per patch row (16*3 = 48)
  let h_p_us = h_p as usize;
  let w_p_us = w_p as usize;

  for py in 0..h_p_us {
    for px in 0..w_p_us {
      let patch_idx = py * w_p_us + px;
      let out_offset = patch_idx * per_patch;
      for r in 0..p {
        let src_y = py * p + r;
        let src_x = px * p;
        let src_off = src_y * src_row_stride + src_x * c;
        let dst_off = out_offset + r * row_bytes;
        // `x / 127.5 - 1.0` == `(x/255 - 0.5)/0.5` (SigLIP mean/std 0.5).
        // A tight per-row scalar loop over `row_bytes` contiguous bytes.
        // See the module-level SIMD note: this reuses the same per-row
        // shape as the NEON resize already covered; it is not a separate
        // hot loop worth its own kernel (the resize dominates).
        normalize_row(
          &resized_buf[src_off..src_off + row_bytes],
          &mut pixel_values[dst_off..dst_off + row_bytes],
        );
      }
    }
  }

  // 4. Side outputs. The mask is `max_num_patches` i32 (bounded by the
  //    cardinality cap, but reserved fallibly for the same recover-don't-abort
  //    discipline as `pixel_values`); `resize` zero-fills within the reserved
  //    capacity, then the first `n_active` slots are set to 1.
  let n_active = grid_patches as usize; // <= max_num_patches (checked above)
  let mut mask: Vec<i32> = Vec::new();
  reserve_or_error(
    &mut mask,
    "siglip2 pixel_attention_mask i32 elements",
    max_num_patches as usize,
  )?;
  mask.resize(max_num_patches as usize, 0i32);
  for slot in mask.iter_mut().take(n_active) {
    *slot = 1;
  }
  let spatial = [h_p as i32, w_p as i32];

  // Build the device arrays (lazy; no eval).
  let pixel_values =
    Array::from_slice::<f32>(&pixel_values, &(max_num_patches as usize, per_patch))?;
  let pixel_attention_mask = Array::from_slice::<i32>(&mask, &(max_num_patches as usize,))?;
  let spatial_shapes = Array::from_slice::<i32>(&spatial, &(2usize,))?;

  Ok(NaflexInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
  })
}

/// Normalize one contiguous RGB byte row into f32 via `x / 127.5 - 1.0`.
///
/// This is the SigLIP `(x/255 - 0.5) / 0.5` rescale in the FMA-friendly
/// form. `src` and `dst` have equal length (`patch_size * channels`).
#[inline]
fn normalize_row(src: &[u8], dst: &mut [f32]) {
  debug_assert_eq!(src.len(), dst.len());
  const SCALE: f32 = 1.0 / 127.5;
  for (s, d) in src.iter().zip(dst.iter_mut()) {
    *d = f32::from(*s) * SCALE - 1.0;
  }
}

#[cfg(all(test, feature = "siglip2-naflex"))]
mod tests;
