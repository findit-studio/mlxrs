//! LFM2.5-VL native-resolution image processor — faithful port of the
//! preprocessing `mlx-vlm/mlx_vlm/models/lfm2_vl/processing_lfm2_vl.py` drives
//! (which defers to the SigLIP2 *slow* image processor `Siglip2ImageProcessor`
//! for the pixel math, and supplies the `<image>`-token expansion itself).
//!
//! Two faithful pieces:
//!
//! ## Native-resolution patchify ([`preprocess_image`])
//!
//! Each image is resized to its native resolution within the patch budget while
//! preserving aspect ratio (SigLIP2 NaFlex *smart-resize*): the largest uniform
//! scale whose per-axis size, rounded up to a multiple of `patch_size`, keeps
//! the patch grid `H_p x W_p` within `max_num_patches`
//! ([`crate::embeddings::siglip2_naflex::processing::patch_grid`] is the
//! authoritative oracle formula, shared verbatim). The image is bilinear-resized
//! to `(H_p * patch_size, W_p * patch_size)` (PIL `Image.BILINEAR`, the upstream
//! SigLIP2 resample), normalized per channel with the configured
//! `image_mean` / `image_std` (SigLIP defaults `0.5` / `0.5` ⇒ the `x/127.5 -
//! 1.0` rescale), and flattened into `(num_patches, num_channels * patch_size^2)`
//! rows in `(row, col, channel-innermost)` order, right-padded with zero rows to
//! `max_num_patches`. The side outputs are the `spatial_shapes` `(H_p, W_p)` and
//! the `pixel_attention_mask` (`1` for the first `H_p * W_p` rows, `0` for the
//! padding) the vision tower consumes — the exact `NaFlex` triple
//! [`crate::vlm::models::lfm2_vl::vision::VisionModel::forward`] /
//! [`crate::vlm::models::lfm2_vl::projector::merge_input_ids_with_image_features`]
//! expect.
//!
//! This is the same patchify path as
//! [`crate::embeddings::siglip2_naflex::processing`] (which the SigLIP2 dual-
//! tower embeddings model uses), generalized over the configured `image_mean` /
//! `image_std` (the SigLIP2 NaFlex embeddings path hard-codes the `0.5` / `0.5`
//! rescale; LFM2.5-VL carries `image_mean` / `image_std` in its
//! `preprocessor_config.json`, defaulting to the same values).
//!
//! ## `<image>`-token expansion ([`expand_image_tokens`] /
//! [`num_image_tokens_from_patch_grid`])
//!
//! After the [`PixelUnshuffleBlock`](crate::vlm::models::lfm2_vl::projector)
//! downsamples the patch grid by `downsample_factor` (padding odd grid
//! dimensions up to a multiple of the factor first), each image occupies
//! `ceil(H_p / factor) * ceil(W_p / factor)` LM token positions. The processor
//! expands the single `<image>` placeholder id in the prompt token sequence into
//! exactly that many `<image>` ids per image (optionally bracketed by the
//! `image_start` / `image_end` ids when `use_image_special_tokens`), so the
//! merged token count aligns with the produced image embeddings — the
//! `_num_image_tokens_from_patch_grid` contract at
//! `processing_lfm2_vl.py:31-48`, whose `ceil` mirrors the PixelUnshuffle pad.
//!
//! ## Bounds / errors
//!
//! Every count / dimension is checked before any allocation; degenerate inputs
//! (zero / overflowing dimensions, a wrong-length RGB slice, an image whose
//! minimum grid exceeds the budget, a token sequence whose `<image>` count
//! disagrees with the image count) return a typed [`crate::Error`] rather than
//! panicking. `mlxrs` is a library, so a merely *large* (but in-bounds) input is
//! accepted — the consuming application owns input bounding.

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, Error, InvariantViolationPayload,
    LengthMismatchPayload, OutOfRangePayload, Result,
  },
  model_validation::{Extent, elem_count, require_positive, reserve_or_error},
  vlm::image::{ResizeFilter, patch_grid, resize},
};

/// The fixed channel count of the patchify path. The SigLIP2 NaFlex
/// preprocessing normalizes per-RGB-channel and flattens into
/// `num_channels * patch_size^2` rows; only the 3-channel (RGB) path is wired,
/// and the config validator pins `num_channels == 3`.
const RGB_CHANNELS: u32 = 3;

/// The channel count of the resized buffer. [`crate::vlm::image::resize`] always
/// returns an `ImageRgba8` (4-channel) image; the patchify reads the leading
/// [`RGB_CHANNELS`] (RGB) bytes of each 4-byte pixel and skips alpha.
const RGBA_CHANNELS: u32 = 4;

/// Per-axis cap on the source image `width` / `height` (px). A single absurd
/// dimension is rejected at [`Extent`] construction, before it reaches the
/// source-clone sizing.
const MAX_SOURCE_DIM: usize = 1 << 16;

/// Total-element cap on the decoded source RGB buffer
/// (`width * height * channels` bytes, ~512 MiB). Bounds the `usize` byte-count
/// arithmetic so `width * height * 3` cannot wrap (a wrapped length would be UB).
const MAX_SOURCE_PIXELS: usize = 1 << 29;

// ═══════════════════════════ Lfm2VlProcessorConfig ══════════════════════════

/// The LFM2.5-VL image-processor parameters — `processing_lfm2_vl.py`'s
/// `Siglip2ImageProcessor` knobs plus the `<image>`-expansion ids the patched
/// `Lfm2VlProcessor.__call__` consumes.
///
/// Defaults match `LiquidAI/LFM2.5-VL-450M-MLX-8bit`'s
/// `preprocessor_config.json` + `config.json`: `patch_size = 16`,
/// `num_channels = 3`, `max_num_patches = 1024`, `downsample_factor = 2`,
/// `image_mean = image_std = 0.5` (the SigLIP rescale), and the
/// `image_token_index = 396` placeholder. The special-token ids
/// (`image_token` / `image_start_token` / `image_end_token`) come from the
/// tokenizer's added-token set; the caller resolves them (the chat template
/// brackets each image with `image_start … image_token×N … image_end`) and
/// passes them here.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug, Clone, Copy)]
pub struct Lfm2VlProcessorConfig {
  patch_size: u32,
  num_channels: u32,
  max_num_patches: u32,
  downsample_factor: u32,
  image_mean: [f32; 3],
  image_std: [f32; 3],
  image_token: i32,
  image_start_token: Option<i32>,
  image_end_token: Option<i32>,
  use_image_special_tokens: bool,
}

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl Lfm2VlProcessorConfig {
  /// Build a processor config from the `(image_token, downsample_factor,
  /// patch_size, max_num_patches)` quad — the four fields the model wiring
  /// supplies from the validated [`ModelConfig`](super::config::ModelConfig) —
  /// with the SigLIP defaults for everything else (`num_channels = 3`,
  /// `image_mean = image_std = 0.5`, no special-token brackets).
  ///
  /// Use the `with_*` builders to set the per-channel mean / std (from the
  /// checkpoint's `preprocessor_config.json`) and the `image_start` /
  /// `image_end` bracket ids (from the tokenizer).
  ///
  /// # Errors
  /// [`Error::OutOfRange`] if `patch_size`, `max_num_patches`, or
  /// `downsample_factor` is `0`, or if `image_token` is negative.
  pub fn new(
    image_token: i32,
    downsample_factor: u32,
    patch_size: u32,
    max_num_patches: u32,
  ) -> Result<Self> {
    require_positive_u32("lfm2_vl processor: patch_size", patch_size)?;
    require_positive_u32("lfm2_vl processor: max_num_patches", max_num_patches)?;
    require_positive_u32("lfm2_vl processor: downsample_factor", downsample_factor)?;
    if image_token < 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "lfm2_vl processor: image_token",
        "must be a non-negative token id (>= 0)",
        smol_str::format_smolstr!("{image_token}"),
      )));
    }
    Ok(Self {
      patch_size,
      num_channels: RGB_CHANNELS,
      max_num_patches,
      downsample_factor,
      image_mean: [0.5, 0.5, 0.5],
      image_std: [0.5, 0.5, 0.5],
      image_token,
      image_start_token: None,
      image_end_token: None,
      use_image_special_tokens: false,
    })
  }

  /// Set the per-channel normalization mean (`preprocessor_config.json`
  /// `image_mean`). SigLIP default is `[0.5, 0.5, 0.5]`.
  #[must_use]
  pub fn with_image_mean(mut self, mean: [f32; 3]) -> Self {
    self.image_mean = mean;
    self
  }

  /// Set the per-channel normalization std (`preprocessor_config.json`
  /// `image_std`). SigLIP default is `[0.5, 0.5, 0.5]`.
  #[must_use]
  pub fn with_image_std(mut self, std: [f32; 3]) -> Self {
    self.image_std = std;
    self
  }

  /// Set the `<image_start>` / `<image_end>` bracket ids + enable bracketing
  /// (`use_image_special_tokens = true`). When either id is `None` the
  /// corresponding bracket is omitted even with bracketing enabled.
  #[must_use]
  pub fn with_special_tokens(mut self, start: Option<i32>, end: Option<i32>) -> Self {
    self.image_start_token = start;
    self.image_end_token = end;
    self.use_image_special_tokens = true;
    self
  }

  /// The patch side length in pixels (`16`).
  #[inline(always)]
  pub fn patch_size(&self) -> u32 {
    self.patch_size
  }

  /// The per-image patch budget (`1024`).
  #[inline(always)]
  pub fn max_num_patches(&self) -> u32 {
    self.max_num_patches
  }

  /// The pixel-unshuffle downsample factor (`2`).
  #[inline(always)]
  pub fn downsample_factor(&self) -> u32 {
    self.downsample_factor
  }

  /// The `<image>` placeholder token id (`396`).
  #[inline(always)]
  pub fn image_token(&self) -> i32 {
    self.image_token
  }
}

// ════════════════════════════ Lfm2VlImageInputs ═════════════════════════════

/// The preprocessed native-resolution inputs for one image — the `NaFlex`
/// triple the LFM2.5-VL vision tower + projector consume.
///
/// All three are produced lazily (built from host buffers via
/// [`Array::from_slice`]); no implicit eval. The fixed leading dimension is
/// `max_num_patches`.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug)]
pub struct Lfm2VlImageInputs {
  /// `(max_num_patches, num_channels * patch_size^2)` f32 — the flattened patch
  /// rows, normalized `(x/255 - mean) / std`, zero-padded past the active
  /// `H_p * W_p` rows.
  pub pixel_values: Array,
  /// `(max_num_patches,)` i32 — `1` for the first `H_p * W_p` rows, `0` for
  /// padding.
  pub pixel_attention_mask: Array,
  /// `(2,)` i32 — `[H_p, W_p]`, the patch grid dimensions.
  pub spatial_shapes: Array,
  /// The patch grid height `H_p` (host int; the same value carried lazily as
  /// `spatial_shapes[0]`). Stashed so [`grid`](Self::grid) needs no eval.
  grid_h: i32,
  /// The patch grid width `W_p` (host int; `spatial_shapes[1]`).
  grid_w: i32,
}

#[cfg(feature = "lfm2-vl")]
impl Lfm2VlImageInputs {
  /// Assemble a [`Lfm2VlImageInputs`] from its parts — the `pixel_values`
  /// `(max_num_patches, patch_feat)`, `pixel_attention_mask` `(max_num_patches,)`,
  /// `spatial_shapes` `(2,)`, and the host patch grid `(H_p, W_p)` (carried so
  /// [`grid`](Self::grid) needs no eval). Lets a caller that already holds a
  /// NaFlex triple (e.g. the cross-model
  /// [`encode_image`](crate::vlm::model::Model::encode_image) reconstructing a
  /// square fully-active grid) build the inputs without re-running
  /// [`preprocess_image`].
  pub fn from_parts(
    pixel_values: Array,
    pixel_attention_mask: Array,
    spatial_shapes: Array,
    grid_h: i32,
    grid_w: i32,
  ) -> Self {
    Self {
      pixel_values,
      pixel_attention_mask,
      spatial_shapes,
      grid_h,
      grid_w,
    }
  }

  /// The native patch grid `(H_p, W_p)` of this image (host integers; the same
  /// values carried lazily in [`spatial_shapes`](Self::spatial_shapes)). Feeds
  /// [`num_image_tokens_from_patch_grid`] / [`expand_image_tokens`].
  pub fn grid(&self) -> (i32, i32) {
    (self.grid_h, self.grid_w)
  }
}

/// Compute the number of `<image>` placeholder tokens the model expects for an
/// image whose native patch grid is `(rows, cols)`, given the pixel-unshuffle
/// `downsample_factor` — `processing_lfm2_vl.py`'s
/// `_num_image_tokens_from_patch_grid` (`:31-48`).
///
/// The [`PixelUnshuffleBlock`](super::projector::PixelUnshuffleBlock) pads each
/// odd patch-grid dimension up to the next multiple of `downsample_factor`
/// before downsampling, so the token count is
/// `ceil(rows / factor) * ceil(cols / factor)` — the text expansion must mirror
/// that padding to keep the image-token count aligned with the produced image
/// embeddings.
///
/// # Errors
/// - [`Error::OutOfRange`] if `factor < 1` or `rows < 1` / `cols < 1`;
/// - [`Error::ArithmeticOverflow`] if the `ceil(rows/f) * ceil(cols/f)` product
///   overflows `i32`.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub fn num_image_tokens_from_patch_grid(rows: i32, cols: i32, factor: i32) -> Result<i32> {
  require_positive("lfm2_vl processor: downsample_factor", factor)?;
  require_positive("lfm2_vl processor: patch grid rows", rows)?;
  require_positive("lfm2_vl processor: patch grid cols", cols)?;
  // `padded = x + (-x % factor)` — round x up to a multiple of factor. With
  // positive operands `(factor - x % factor) % factor` is the same non-negative
  // remainder complement; computed without wrapping.
  let padded_rows = round_up_to_multiple(rows, factor)?;
  let padded_cols = round_up_to_multiple(cols, factor)?;
  let down_rows = padded_rows / factor;
  let down_cols = padded_cols / factor;
  crate::model_validation::checked_mul(
    "lfm2_vl processor: image token count (ceil(rows/f) * ceil(cols/f))",
    "down_rows",
    down_rows,
    "down_cols",
    down_cols,
  )
}

/// Round `value` up to the next multiple of `factor` (both `> 0`):
/// `value + ((factor - value % factor) % factor)`, checked so a near-`i32::MAX`
/// value cannot wrap.
#[cfg(feature = "lfm2-vl")]
fn round_up_to_multiple(value: i32, factor: i32) -> Result<i32> {
  let rem = value % factor;
  if rem == 0 {
    return Ok(value);
  }
  crate::model_validation::checked_add(
    "lfm2_vl processor: round up to multiple of factor",
    "value",
    value,
    "pad",
    factor - rem,
  )
}

// ═══════════════════════════════ preprocess ═════════════════════════════════

/// Reject a dimension outside `(0, …]`.
#[cfg(feature = "lfm2-vl")]
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

/// Native-resolution preprocess one interleaved RGB image into
/// [`Lfm2VlImageInputs`].
///
/// `rgb` is `width * height * 3` row-major interleaved RGB bytes (no row
/// padding). The patch grid, resize, normalize, and patchify follow the SigLIP2
/// NaFlex slow processor the LFM2.5-VL processor defers to.
///
/// Returns the flattened `(max_num_patches, P^2 * C)` pixel tensor plus the
/// attention mask and spatial shapes (see [`Lfm2VlImageInputs`]).
///
/// ## Errors
/// - `width == 0` / `height == 0` → [`Error::OutOfRange`].
/// - `cfg.num_channels != 3` → [`Error::InvariantViolation`] (the patchify path
///   is RGB-only; the resized buffer is always 3-channel).
/// - `width` / `height` exceeds the per-axis cap `MAX_SOURCE_DIM`, or
///   `width * height * 3` overflows `usize` or exceeds the source-bytes cap
///   `MAX_SOURCE_PIXELS`, or `rgb.len()` disagrees with the byte count →
///   [`Error::CapExceeded`] / [`Error::ArithmeticOverflow`] /
///   [`Error::LengthMismatch`].
/// - the `pixel_values` element count `max_num_patches * (3 * patch_size^2)`
///   overflows `usize` → [`Error::ArithmeticOverflow`].
/// - the selected grid `H_p * W_p` exceeds `max_num_patches` (an adversarial
///   dimension whose feasible scale range is below the search floor) →
///   [`Error::CapExceeded`].
/// - a within-cap but heavyweight `pixel_values` / mask reservation the
///   allocator cannot satisfy → [`Error::AllocFailure`].
/// - underlying [`crate::vlm::image::resize`] / [`Array::from_slice`] errors
///   propagate.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub fn preprocess_image(
  rgb: &[u8],
  width: u32,
  height: u32,
  cfg: &Lfm2VlProcessorConfig,
) -> Result<Lfm2VlImageInputs> {
  require_positive_u32("lfm2_vl preprocess: width", width)?;
  require_positive_u32("lfm2_vl preprocess: height", height)?;
  let patch_size = cfg.patch_size;
  let max_num_patches = cfg.max_num_patches;
  // The patchify path is RGB-only: it constructs a 3-channel `RgbImage`, resizes
  // into an always-3-channel buffer, and uses the channel count as the patchify
  // stride into that buffer. A `num_channels != 3` would slice with a wrong
  // stride and read out of bounds — reject it before any sizing.
  if cfg.num_channels != RGB_CHANNELS {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "lfm2_vl preprocess: num_channels",
      "must be 3 (the patchify path is RGB-only)",
    )));
  }

  // Expected RGB byte count via the resource layer: each axis is an `Extent`
  // (per-axis capped at construction) and `elem_count` is the checked product
  // against the total source-bytes cap `MAX_SOURCE_PIXELS`.
  let channels = RGB_CHANNELS as usize;
  let expected_rgb_len = elem_count(
    "lfm2_vl preprocess: rgb byte count (width * height * channels)",
    &[
      Extent::new("lfm2_vl preprocess: width", width as usize, MAX_SOURCE_DIM)?,
      Extent::new(
        "lfm2_vl preprocess: height",
        height as usize,
        MAX_SOURCE_DIM,
      )?,
      Extent::new(
        "lfm2_vl preprocess: channels",
        channels,
        RGB_CHANNELS as usize,
      )?,
    ],
    MAX_SOURCE_PIXELS,
  )?;
  if rgb.len() != expected_rgb_len {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "lfm2_vl preprocess: rgb slice length vs width*height*channels",
      expected_rgb_len,
      rgb.len(),
    )));
  }

  // Per-patch flattened width = C * P^2; the `pixel_values` element count is
  // `max_num_patches * per_patch`. Both overflow-checked so a hostile config
  // cannot wrap the allocation size (a wrapped size would be UB).
  let p = patch_size as usize;
  let per_patch = p
    .checked_mul(p)
    .and_then(|pp| pp.checked_mul(channels))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl preprocess: per-patch width (patch_size^2 * channels)",
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
        "lfm2_vl preprocess: pixel_values size (max_num_patches * per_patch)",
        "usize",
        [
          ("max_num_patches", u64::from(max_num_patches)),
          ("per_patch", per_patch as u64),
        ],
      ))
    })?;

  // 1. Target patch grid (the authoritative SigLIP2 NaFlex oracle formula,
  //    shared verbatim with the embeddings NaFlex path).
  let (h_p, w_p) = patch_grid(height, width, patch_size, max_num_patches);

  // Postcondition: the binary search assumes its `scale_min = eps/10` floor is
  // feasible; for adversarial `u32` inputs the entire feasible range can fall
  // below that floor, leaving a grid above budget. Re-check before sizing the
  // fixed pixel buffer so we never index out of bounds.
  let grid_patches = u64::from(h_p).checked_mul(u64::from(w_p)).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "lfm2_vl preprocess: grid patch count (H_p * W_p)",
      "u64",
      [("H_p", u64::from(h_p)), ("W_p", u64::from(w_p))],
    ))
  })?;
  if grid_patches > u64::from(max_num_patches) {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "lfm2_vl preprocess: selected patch grid",
      "max_num_patches",
      u64::from(max_num_patches),
      grid_patches,
    )));
  }

  // Resized pixel dimensions.
  let h_res = h_p.checked_mul(patch_size).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "lfm2_vl preprocess: resized height (H_p * patch_size)",
      "u32",
      [
        ("H_p", u64::from(h_p)),
        ("patch_size", u64::from(patch_size)),
      ],
    ))
  })?;
  let w_res = w_p.checked_mul(patch_size).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "lfm2_vl preprocess: resized width (W_p * patch_size)",
      "u32",
      [
        ("W_p", u64::from(w_p)),
        ("patch_size", u64::from(patch_size)),
      ],
    ))
  })?;

  // 2. Aspect-preserving resize via the PIL-bit-exact NEON bilinear (PIL
  //    `Image.BILINEAR`, the upstream SigLIP2 resample). Build an owned RgbImage
  //    over the caller's bytes (image 0.25's `from_raw` needs an owned
  //    container), then resize. Fallible source clone: reserve into the exact,
  //    already-capped `expected_rgb_len`, then copy.
  let mut src_buf: Vec<u8> = Vec::new();
  reserve_or_error(
    &mut src_buf,
    "lfm2_vl preprocess: source RGB clone",
    expected_rgb_len,
  )?;
  src_buf.extend_from_slice(rgb);
  let src_img: ::image::RgbImage = ::image::ImageBuffer::from_raw(width, height, src_buf)
    .ok_or_else(|| {
      Error::LengthMismatch(LengthMismatchPayload::new(
        "lfm2_vl preprocess: RgbImage::from_raw length",
        expected_rgb_len,
        rgb.len(),
      ))
    })?;
  let src_dyn = ::image::DynamicImage::ImageRgb8(src_img);
  // `vlm::resize` takes target as (height, width) and returns an `ImageRgba8`.
  let resized_dyn = resize(&src_dyn, (h_res, w_res), ResizeFilter::Bilinear)?;
  let resized = resized_dyn.as_rgba8().ok_or_else(|| {
    Error::InvariantViolation(InvariantViolationPayload::new(
      "lfm2_vl preprocess: resized image format",
      "vlm::image::resize must return an RGBA8 image (the patchify reads its R/G/B bytes)",
    ))
  })?;

  // 3. Normalize + patchify into the fixed (max_num_patches, P^2*C) buffer,
  //    zero-padded past the active rows. Build the buffer FALLIBLY then
  //    zero-fill via `resize` (cannot reallocate — the exact capacity is
  //    already reserved).
  let mut pixel_values: Vec<f32> = Vec::new();
  reserve_or_error(
    &mut pixel_values,
    "lfm2_vl pixel_values f32 elements",
    total_floats,
  )?;
  pixel_values.resize(total_floats, 0.0f32);

  let resized_buf = resized.as_raw();
  let c = channels; // 3 — output channels per pixel
  let src_c = RGBA_CHANNELS as usize; // 4 — source bytes per pixel (RGBA)
  let src_row_stride = (w_res as usize) * src_c; // bytes per resized RGBA row
  let row_bytes = p * c; // RGB floats per patch row (16*3 = 48)
  let h_p_us = h_p as usize;
  let w_p_us = w_p as usize;

  // Per-channel affine `(x/255 - mean) / std`, folded to `x * scale + bias` so
  // the patchify is one multiply + add per channel — the same fused form the
  // SigLIP `x/127.5 - 1.0` (mean = std = 0.5) collapses to.
  let scale = [
    (1.0 / 255.0) / cfg.image_std[0],
    (1.0 / 255.0) / cfg.image_std[1],
    (1.0 / 255.0) / cfg.image_std[2],
  ];
  let bias = [
    -cfg.image_mean[0] / cfg.image_std[0],
    -cfg.image_mean[1] / cfg.image_std[1],
    -cfg.image_mean[2] / cfg.image_std[2],
  ];

  for py in 0..h_p_us {
    for px in 0..w_p_us {
      let patch_idx = py * w_p_us + px;
      let out_offset = patch_idx * per_patch;
      for r in 0..p {
        let src_y = py * p + r;
        let src_x = px * p;
        let src_off = src_y * src_row_stride + src_x * src_c;
        let dst_off = out_offset + r * row_bytes;
        // Per-row RGBA → RGB widen + per-channel affine over `p` pixels, reading
        // the 3 leading RGB bytes of each 4-byte RGBA source pixel and dropping
        // alpha.
        normalize_row_rgba(
          &resized_buf[src_off..src_off + p * src_c],
          &mut pixel_values[dst_off..dst_off + row_bytes],
          scale,
          bias,
        );
      }
    }
  }

  // 4. Side outputs. The mask is `max_num_patches` i32; the first `n_active`
  //    slots are set to 1.
  let n_active = grid_patches as usize; // <= max_num_patches (checked above)
  let mut mask: Vec<i32> = Vec::new();
  reserve_or_error(
    &mut mask,
    "lfm2_vl pixel_attention_mask i32 elements",
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

  Ok(Lfm2VlImageInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
    grid_h: h_p as i32,
    grid_w: w_p as i32,
  })
}

/// Normalize one contiguous RGBA byte row into a 3-channel RGB f32 row via the
/// per-channel affine `x * scale[c] + bias[c]` (= `(x/255 - mean) / std`),
/// dropping the alpha byte of each 4-byte source pixel.
///
/// `src` is `[RGBA, RGBA, …]` (`p * RGBA_CHANNELS` bytes); `dst` is the matching
/// `[RGB, RGB, …]` (`p * RGB_CHANNELS` f32).
///
/// When the three per-channel scales (and the three biases) are equal — the
/// SigLIP `mean = std = 0.5` default ⇒ `scale = 1/127.5`, `bias = -1.0` — this
/// delegates to the NEON-accelerated
/// [`crate::simd::vlm::rgba_to_rgb_affine`] dispatcher (bit-identical scalar
/// fallback off aarch64), exactly as the SigLIP2 NaFlex embeddings patchify
/// does. A genuinely per-channel mean / std (rare; the LFM2.5-VL checkpoint uses
/// the uniform default) takes the scalar per-channel path.
#[cfg(feature = "lfm2-vl")]
#[inline]
fn normalize_row_rgba(src: &[u8], dst: &mut [f32], scale: [f32; 3], bias: [f32; 3]) {
  // Uniform per-channel scale/bias ⇒ the single-scalar NEON kernel applies
  // (the common SigLIP `mean = std = 0.5` path).
  if scale[0] == scale[1] && scale[1] == scale[2] && bias[0] == bias[1] && bias[1] == bias[2] {
    crate::simd::vlm::rgba_to_rgb_affine(src, dst, scale[0], bias[0]);
    return;
  }
  // Per-channel path: read the leading 3 RGB bytes of each 4-byte RGBA pixel.
  let src_c = RGBA_CHANNELS as usize;
  let pixels = dst.len() / RGB_CHANNELS as usize;
  for i in 0..pixels {
    let s = i * src_c;
    let d = i * RGB_CHANNELS as usize;
    dst[d] = f32::from(src[s]) * scale[0] + bias[0];
    dst[d + 1] = f32::from(src[s + 1]) * scale[1] + bias[1];
    dst[d + 2] = f32::from(src[s + 2]) * scale[2] + bias[2];
  }
}

// ═══════════════════════════ <image> expansion ══════════════════════════════

/// Expand the single `<image>` placeholder id in a prompt token sequence into
/// the per-image image-token runs — `processing_lfm2_vl.py`'s
/// `_patched_call` text-expansion loop (`:378-404`), on already-tokenized ids.
///
/// `input_ids` is the prompt's token id sequence; each occurrence of
/// `cfg.image_token` marks where the `i`-th image's tokens go (in order). For
/// each image the placeholder is replaced by `cfg.image_start_token?` +
/// `num_image_tokens_from_patch_grid(rows_i, cols_i, factor)` copies of
/// `cfg.image_token` + `cfg.image_end_token?` (the start / end brackets emitted
/// only when [`Lfm2VlProcessorConfig::with_special_tokens`] enabled them). Any
/// non-placeholder id passes through unchanged.
///
/// `grids` is the per-image native patch grid `(rows, cols)` (one entry per
/// image, in prompt order) — typically each image's
/// [`Lfm2VlImageInputs::grid`].
///
/// The number of `cfg.image_token` placeholders in `input_ids` must equal
/// `grids.len()` — a mismatch is a typed [`Error::LengthMismatch`] (the
/// reference's `n_images_in_text != n_images_in_images` guard at
/// `processing_lfm2_vl.py:344-348`).
///
/// # Errors
/// - [`Error::LengthMismatch`] if the placeholder count `!= grids.len()`;
/// - [`Error::OutOfRange`] / [`Error::ArithmeticOverflow`] from
///   [`num_image_tokens_from_patch_grid`] for a degenerate / overflowing grid;
/// - [`Error::AllocFailure`] if the expanded-sequence reservation fails.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub fn expand_image_tokens(
  input_ids: &[i32],
  grids: &[(i32, i32)],
  cfg: &Lfm2VlProcessorConfig,
) -> Result<Vec<i32>> {
  let factor = i32::try_from(cfg.downsample_factor).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "lfm2_vl expand: downsample_factor",
      "must fit in i32",
      smol_str::format_smolstr!("{}", cfg.downsample_factor),
    ))
  })?;

  // First pass: validate the placeholder count matches the image count and
  // compute each image's token run length, so the expanded buffer is reserved
  // once (no reallocation in the build loop).
  let placeholder_count = input_ids
    .iter()
    .filter(|&&id| id == cfg.image_token)
    .count();
  if placeholder_count != grids.len() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "lfm2_vl expand: <image> placeholder count vs image count (the number of images in the \
         text and images should be the same)",
      grids.len(),
      placeholder_count,
    )));
  }

  // Per-image run length (excluding brackets) + total expanded length.
  let bracket_count = usize::from(cfg.use_image_special_tokens && cfg.image_start_token.is_some())
    + usize::from(cfg.use_image_special_tokens && cfg.image_end_token.is_some());
  let mut runs: Vec<usize> = Vec::new();
  reserve_or_error(
    &mut runs,
    "lfm2_vl expand: per-image run length",
    grids.len(),
  )?;
  // The expanded length is: the non-placeholder ids (kept verbatim) plus, per
  // image, its run of image tokens + brackets. Track it with a checked add so a
  // hostile grid product cannot wrap the capacity.
  let mut expanded_len = input_ids
    .len()
    .checked_sub(placeholder_count)
    .ok_or_else(|| {
      // Unreachable: placeholder_count <= input_ids.len() by construction.
      Error::InvariantViolation(InvariantViolationPayload::new(
        "lfm2_vl expand: placeholder count vs input length",
        "placeholder count cannot exceed the input length",
      ))
    })?;
  for &(rows, cols) in grids {
    let n_tokens = num_image_tokens_from_patch_grid(rows, cols, factor)? as usize;
    let run = n_tokens.checked_add(bracket_count).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl expand: per-image run (tokens + brackets)",
        "usize",
        [
          ("n_tokens", n_tokens as u64),
          ("brackets", bracket_count as u64),
        ],
      ))
    })?;
    runs.push(run);
    expanded_len = expanded_len.checked_add(run).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl expand: cumulative expanded length",
        "usize",
        [("expanded_len", expanded_len as u64), ("run", run as u64)],
      ))
    })?;
  }

  // Second pass: build the expanded id sequence.
  let mut out: Vec<i32> = Vec::new();
  reserve_or_error(&mut out, "lfm2_vl expand: expanded token id", expanded_len)?;
  let emit_start = cfg.use_image_special_tokens && cfg.image_start_token.is_some();
  let emit_end = cfg.use_image_special_tokens && cfg.image_end_token.is_some();
  let mut img_idx = 0usize;
  for &id in input_ids {
    if id == cfg.image_token {
      let n_tokens = runs[img_idx] - bracket_count;
      if emit_start {
        out.push(cfg.image_start_token.expect("emit_start ⇒ Some"));
      }
      for _ in 0..n_tokens {
        out.push(cfg.image_token);
      }
      if emit_end {
        out.push(cfg.image_end_token.expect("emit_end ⇒ Some"));
      }
      img_idx += 1;
    } else {
      out.push(id);
    }
  }
  Ok(out)
}

#[cfg(all(test, feature = "lfm2-vl"))]
mod tests;
