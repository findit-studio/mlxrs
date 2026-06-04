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
//! ## Image splitting / tiling ([`tile_image`] / [`plan_tiles`])
//!
//! mlx-vlm DELIBERATELY disables splitting (its `processing_lfm2_vl.py` forces
//! `do_image_splitting = False` and uses the SigLIP2 native-resolution path
//! above), so the tile-grid algorithm lives only in HuggingFace `transformers`
//! `Lfm2VlImageProcessor` (`image_processing_lfm2_vl.py`). [`tile_image`] is a
//! faithful port of THAT source (cited per function): it decides whether an image
//! is over the size threshold (`_is_image_too_large`), picks the tile grid that
//! best matches the aspect ratio (`find_closest_aspect_ratio` over the
//! `min_tiles..=max_tiles` candidate set), resizes + splits the image into
//! `tile_size x tile_size` tiles (`split_to_tiles`), and optionally appends a
//! within-budget smart-resized thumbnail — producing one [`Lfm2VlImageInputs`]
//! NaFlex triple per tile (+ thumbnail), each patchified at `encoder_patch_size`.
//! The per-tile grids drive [`expand_image_tokens`] (each sub-image bracketed by
//! `image_start`/`image_end` and concatenated — the mlx-vlm `_patched_call`
//! per-sub-image layout). This path is gated on
//! [`Lfm2VlProcessorConfig::do_image_splitting`]; the native-resolution
//! [`preprocess_image`] path never consults the tiling knobs.
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
  dtype::Dtype,
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, Error, InvariantViolationPayload,
    LengthMismatchPayload, OutOfRangePayload, RankMismatchPayload, Result,
  },
  model_validation::{Extent, elem_count, require_positive, reserve_or_error},
  ops,
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
  /// The image-splitting / tiling parameters (the HF `Lfm2VlImageProcessor`
  /// knobs the mlx-vlm path omits). Carried so the tiling driver
  /// ([`tile_image`]) can run when the model opts into splitting; the SigLIP2
  /// native-resolution [`preprocess_image`] path does not consult them.
  tiling: TilingConfig,
}

/// The HF `Lfm2VlImageProcessor` image-splitting parameters
/// (`image_processing_lfm2_vl.py:218-228`). Defaults match the upstream class
/// attributes. Carried on [`Lfm2VlProcessorConfig`] for the tiling driver
/// [`tile_image`]; the mlx-vlm native-resolution path leaves them inert.
#[cfg(feature = "lfm2-vl")]
#[derive(Debug, Clone, Copy)]
struct TilingConfig {
  /// `do_image_splitting` (`:220`, default `true`).
  do_image_splitting: bool,
  /// `min_tiles` (`:221`, default `2`).
  min_tiles: u32,
  /// `max_tiles` (`:222`, default `10`).
  max_tiles: u32,
  /// `use_thumbnail` (`:223`, default `true` upstream; the mlx-community
  /// `config.json` ships `false`).
  use_thumbnail: bool,
  /// `min_image_tokens` (`:224`, default `64`).
  min_image_tokens: u32,
  /// `max_image_tokens` (`:225`, default `256`).
  max_image_tokens: u32,
  /// `encoder_patch_size` (`:226`, default `16`).
  encoder_patch_size: u32,
  /// `tile_size` (`:227`, default `512`).
  tile_size: u32,
  /// `max_pixels_tolerance` (`:228`, default `2.0`).
  max_pixels_tolerance: f32,
}

#[cfg(feature = "lfm2-vl")]
impl Default for TilingConfig {
  fn default() -> Self {
    Self {
      do_image_splitting: true,
      min_tiles: 2,
      max_tiles: 10,
      use_thumbnail: true,
      min_image_tokens: 64,
      max_image_tokens: 256,
      encoder_patch_size: 16,
      tile_size: 512,
      max_pixels_tolerance: 2.0,
    }
  }
}

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl Lfm2VlProcessorConfig {
  /// Build a processor config from the `(image_token, downsample_factor,
  /// patch_size, max_num_patches)` quad — the four fields the model wiring
  /// supplies from the validated [`ModelConfig`](super::config::ModelConfig) —
  /// with the SigLIP defaults for everything else (`num_channels = 3`,
  /// `image_mean = image_std = 0.5`).
  ///
  /// `use_image_special_tokens` defaults to `true` (upstream
  /// `processing_lfm2_vl.py:331` / `config.py:87`), so an image is bracketed by
  /// the `image_start` / `image_end` ids once those are supplied. They start as
  /// `None`, so no bracket is emitted until [`with_special_tokens`] (or the
  /// individual setters) provide the ids — matching upstream, where the brackets
  /// appear only when the tokenizer carries them.
  ///
  /// [`with_special_tokens`]: Self::with_special_tokens
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
      // Upstream default (`config.py:87`, `processing_lfm2_vl.py:331`):
      // `use_image_special_tokens = True`. The bracket ids start `None`, so no
      // bracket emits until `with_special_tokens` supplies them.
      use_image_special_tokens: true,
      // The image-splitting parameters default to the HF
      // `Lfm2VlImageProcessor` class attributes; `with_tiling` overrides them
      // from the model's [`ModelConfig`](super::config::ModelConfig).
      tiling: TilingConfig::default(),
    })
  }

  /// Set the image-splitting / tiling parameters from the model's
  /// [`ModelConfig`](super::config::ModelConfig) — the HF
  /// `Lfm2VlImageProcessor` knobs (`do_image_splitting`, `min_tiles`,
  /// `max_tiles`, `use_thumbnail`, `min_image_tokens`, `max_image_tokens`,
  /// `encoder_patch_size`, `tile_size`, `max_pixels_tolerance`). Drives the
  /// [`tile_image`] splitting path; the SigLIP2 native-resolution
  /// [`preprocess_image`] path ignores them.
  ///
  /// # Errors
  /// [`Error::OutOfRange`] if any cardinality field
  /// (`min_tiles` / `max_tiles` / `min_image_tokens` / `max_image_tokens` /
  /// `encoder_patch_size` / `tile_size`) is `0`, [`Error::OutOfRange`] if
  /// `max_tiles` exceeds the tile-grid cardinality cap
  /// ([`MAX_TILES`](super::config::MAX_TILES)), [`Error::OutOfRange`] if
  /// `min_tiles > max_tiles` or `min_image_tokens > max_image_tokens`,
  /// [`Error::NonFiniteScalar`] / [`Error::OutOfRange`] if
  /// `max_pixels_tolerance` is non-finite / non-positive, and
  /// [`Error::DivisibilityConstraint`] if `tile_size` is not a whole multiple of
  /// `encoder_patch_size` (a tile must split into a whole patch grid).
  #[allow(clippy::too_many_arguments)]
  pub fn with_tiling(
    mut self,
    do_image_splitting: bool,
    min_tiles: u32,
    max_tiles: u32,
    use_thumbnail: bool,
    min_image_tokens: u32,
    max_image_tokens: u32,
    encoder_patch_size: u32,
    tile_size: u32,
    max_pixels_tolerance: f32,
  ) -> Result<Self> {
    for (name, value) in [
      ("lfm2_vl tiling: min_tiles", min_tiles),
      ("lfm2_vl tiling: max_tiles", max_tiles),
      ("lfm2_vl tiling: min_image_tokens", min_image_tokens),
      ("lfm2_vl tiling: max_image_tokens", max_image_tokens),
      ("lfm2_vl tiling: encoder_patch_size", encoder_patch_size),
      ("lfm2_vl tiling: tile_size", tile_size),
    ] {
      require_positive_u32(name, value)?;
    }
    // Bound `max_tiles` to the tile-grid cardinality cap before any tile-grid
    // work: `target_ratios` reserves / iterates `max_tiles^2` candidates, so an
    // oversized value (malformed checkpoint or caller) would otherwise drive a
    // quadratic reservation before the pixel caps apply. Mirrors
    // [`ModelConfig::validate`](super::config::ModelConfig::validate); see
    // [`MAX_TILES`](super::config::MAX_TILES) for the chosen bound.
    if max_tiles > super::config::MAX_TILES as u32 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "lfm2_vl tiling: max_tiles",
        "must not exceed the tile-grid cardinality cap (1024)",
        smol_str::format_smolstr!("{max_tiles}"),
      )));
    }
    if min_tiles > max_tiles {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "lfm2_vl tiling: min_tiles <= max_tiles",
        "the minimum tile count must not exceed the maximum",
        smol_str::format_smolstr!("min_tiles={min_tiles}, max_tiles={max_tiles}"),
      )));
    }
    if min_image_tokens > max_image_tokens {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "lfm2_vl tiling: min_image_tokens <= max_image_tokens",
        "the minimum token budget must not exceed the maximum",
        smol_str::format_smolstr!(
          "min_image_tokens={min_image_tokens}, max_image_tokens={max_image_tokens}"
        ),
      )));
    }
    crate::model_validation::require_positive_finite_f32(
      "lfm2_vl tiling: max_pixels_tolerance",
      f64::from(max_pixels_tolerance),
    )?;
    // A tile is resized to `tile_size x tile_size` then patchified at
    // `encoder_patch_size`; the split is only well-defined when `tile_size` is a
    // whole number of patches (`split_to_tiles` / `convert_image_to_patches`
    // assume `height // patch_size` is exact).
    crate::model_validation::require_divisible(
      "lfm2_vl tiling: tile_size",
      tile_size as i32,
      "encoder_patch_size",
      encoder_patch_size as i32,
    )?;
    self.tiling = TilingConfig {
      do_image_splitting,
      min_tiles,
      max_tiles,
      use_thumbnail,
      min_image_tokens,
      max_image_tokens,
      encoder_patch_size,
      tile_size,
      max_pixels_tolerance,
    };
    Ok(self)
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

  /// Set the `<image_start>` / `<image_end>` bracket ids only — the identities of
  /// the bracket tokens, not the policy that decides whether to emit them.
  ///
  /// Whether [`expand_image_tokens`] actually wraps each `<image>` run in these
  /// brackets is governed *solely* by `use_image_special_tokens`
  /// (`config.py:87`, default `true`), which the reference reads from the
  /// checkpoint and gates emission on (`processing_lfm2_vl.py:388-400`'s `if
  /// use_image_special_tokens:`). Supplying the ids does **not** turn bracketing
  /// on: a checkpoint that set `use_image_special_tokens = false` keeps emitting
  /// no brackets after this call. To enable bracketing set the flag explicitly
  /// with [`with_use_image_special_tokens`] (or build the config via the model's
  /// `processor_config`, which threads the checkpoint flag); a caller that wants
  /// brackets supplies *both* the flag and the ids.
  ///
  /// When either id is `None` the corresponding bracket is omitted even with the
  /// flag enabled.
  ///
  /// [`with_use_image_special_tokens`]: Self::with_use_image_special_tokens
  /// [`expand_image_tokens`]: crate::vlm::models::lfm2_vl::expand_image_tokens
  #[must_use]
  pub fn with_special_tokens(mut self, start: Option<i32>, end: Option<i32>) -> Self {
    self.image_start_token = start;
    self.image_end_token = end;
    self
  }

  /// Set whether images are bracketed by the `image_start` / `image_end` ids —
  /// the checkpoint's `use_image_special_tokens`
  /// (`config.py:87`, `processing_lfm2_vl.py:330-331`, default `true`). When
  /// `false`, [`expand_image_tokens`] emits **no** start / end brackets even if
  /// both ids are set, matching the reference's `if use_image_special_tokens:`
  /// gate (`processing_lfm2_vl.py:388-400`); the `<image>` placeholder still
  /// expands to its per-image token run, so the image-feature / token counts stay
  /// consistent.
  #[must_use]
  pub fn with_use_image_special_tokens(mut self, enabled: bool) -> Self {
    self.use_image_special_tokens = enabled;
    self
  }

  /// Override only the `do_image_splitting` flag, leaving the other tiling knobs
  /// at their current values. Used to pin the native-resolution primary path to
  /// `false` — mlx-vlm's `processing_lfm2_vl.py` forces
  /// `do_image_splitting = False` for the slow SigLIP2 NaFlex processor
  /// (`processing_lfm2_vl.py:196, 270-273`), so the primary
  /// [`Lfm2VlImageProcessor`](super::model::Lfm2VlImageProcessor) path never
  /// advertises tiling it does not perform. The opt-in [`tile_image`] /
  /// `split_image` path builds its config via [`with_tiling`](Self::with_tiling)
  /// and keeps the checkpoint's flag.
  #[must_use]
  pub fn with_do_image_splitting(mut self, enabled: bool) -> Self {
    self.tiling.do_image_splitting = enabled;
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

  /// Whether images are bracketed by the `image_start` / `image_end` ids when
  /// those ids are set. Defaults to `true` (upstream `config.py:87` /
  /// `processing_lfm2_vl.py:331`).
  #[inline(always)]
  pub fn use_image_special_tokens(&self) -> bool {
    self.use_image_special_tokens
  }

  /// Whether the HF tiling path ([`tile_image`]) splits an over-budget image
  /// into tiles (`do_image_splitting`, `config.py:76`). When `false`,
  /// [`tile_image`] always emits a single native-resolution sub-image.
  #[inline(always)]
  pub fn do_image_splitting(&self) -> bool {
    self.tiling.do_image_splitting
  }
}

// ════════════════════════════ Lfm2VlImageInputs ═════════════════════════════

/// The preprocessed native-resolution inputs for one image — the `NaFlex`
/// triple the LFM2.5-VL vision tower + projector consume.
///
/// All three are produced lazily (built from host buffers via
/// [`Array::from_slice`]); no implicit eval. The fixed leading dimension is
/// `max_num_patches`.
///
/// ## Single source of truth for the active grid
///
/// [`spatial_shapes`](Self::spatial_shapes) is the **only** carrier of the
/// active patch grid `(H_p, W_p)`; there is no separate host-int grid field.
/// The vision tower derives its attention mask and position resize from
/// `spatial_shapes` in
/// [`VisionModel::forward`](super::vision::VisionModel::forward), and the
/// active-row slice and PixelUnshuffle reshape in
/// [`Lfm2Vl::encode_image_inputs`](super::model::Lfm2Vl::encode_image_inputs)
/// read the same `spatial_shapes` via [`grid`](Self::grid). The two therefore
/// cannot disagree: a mask/slice disagreement is *unrepresentable* because the
/// type cannot hold two grids.
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
  /// `(2,)` i32 — `[H_p, W_p]`, the patch grid dimensions. The **single source
  /// of truth** for the active grid (see the type docs); [`grid`](Self::grid)
  /// reads `(H_p, W_p)` from here.
  pub spatial_shapes: Array,
}

#[cfg(feature = "lfm2-vl")]
impl Lfm2VlImageInputs {
  /// Assemble a [`Lfm2VlImageInputs`] from its parts — the `pixel_values`
  /// `(max_num_patches, patch_feat)`, `pixel_attention_mask` `(max_num_patches,)`,
  /// and `spatial_shapes` `(2,)` `[H_p, W_p]`. Lets a caller that already holds a
  /// NaFlex triple (e.g. the cross-model
  /// [`encode_image`](crate::vlm::model::Model::encode_image) reconstructing a
  /// square fully-active grid) build the inputs without re-running
  /// [`preprocess_image`].
  ///
  /// The active grid is read from `spatial_shapes` alone (there is no separate
  /// grid argument), so a caller cannot supply a grid that disagrees with
  /// `spatial_shapes` — the mask/slice desync class is unrepresentable.
  pub fn from_parts(
    pixel_values: Array,
    pixel_attention_mask: Array,
    spatial_shapes: Array,
  ) -> Self {
    Self {
      pixel_values,
      pixel_attention_mask,
      spatial_shapes,
    }
  }

  /// The native patch grid `(H_p, W_p)` of this image, read from the `(2,)`
  /// [`spatial_shapes`](Self::spatial_shapes) companion (the single source of
  /// truth). Feeds [`num_image_tokens_from_patch_grid`] / [`expand_image_tokens`]
  /// and the active-row slice + PixelUnshuffle reshape in
  /// [`Lfm2Vl::encode_image_inputs`](super::model::Lfm2Vl::encode_image_inputs).
  ///
  /// Materializes the tiny `(2,)` array to host integers (a 2-element read; the
  /// grid geometry is host-side). The eval is on an internal clone — `self` is
  /// not mutated — so this stays a `&self` read with no observable side effect,
  /// mirroring the vision tower's `spatial_shapes` host read.
  ///
  /// # Errors
  /// [`Error::RankMismatch`] if `spatial_shapes` is not a `(2,)` array.
  pub fn grid(&self) -> Result<(i32, i32)> {
    let shape = self.spatial_shapes.shape();
    if shape.as_slice() != [2] {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "lfm2_vl Lfm2VlImageInputs::grid: spatial_shapes must be a (2,) [H_p, W_p] array",
        shape.len() as u32,
        shape,
      )));
    }
    // Materialize the tiny `(2,)` companion to host ints. `astype` produces a
    // fresh array; the eval is on that clone, never on `self.spatial_shapes`.
    let mut s = ops::misc::astype(&self.spatial_shapes, Dtype::I32)?;
    s.eval()?;
    let v = s.to_vec::<i32>()?;
    Ok((v[0], v[1]))
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
/// only when `cfg.use_image_special_tokens` is set — `config.py:87`,
/// [`with_use_image_special_tokens`](Lfm2VlProcessorConfig::with_use_image_special_tokens)
/// — and the respective id is `Some`). Any non-placeholder id passes through
/// unchanged.
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

// ══════════════════════════ image splitting / tiling ════════════════════════
//
// Faithful port of the HuggingFace `transformers` `Lfm2VlImageProcessor` tile
// path (`image_processing_lfm2_vl.py`), which mlx-vlm deliberately OMITS — its
// `processing_lfm2_vl.py` defers to the slow `Siglip2ImageProcessor` (the
// `preprocess_image` native-resolution path above) and forces
// `do_image_splitting = False`. mlx-vlm therefore carries no tile-grid math, so
// this port's provenance is the HF source (cited per function below), not the
// mlx reference. The token layout the produced tiles drive — each tile + the
// optional thumbnail bracketed by `image_start`/`image_end` and concatenated —
// is the mlx-vlm `_patched_call` per-sub-image expansion
// (`processing_lfm2_vl.py:378-404`), reused verbatim through
// [`expand_image_tokens`] over the per-tile grids.

/// One image's tile layout decision — the output of [`plan_tiles`], the pure
/// (host-integer-only) core of the HF `resize_and_split`
/// (`image_processing_lfm2_vl.py:382-436`). Oracle-testable in isolation.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
  /// `true` iff the image is split into a `> 1`-tile grid (the image was both
  /// over the size threshold AND `do_image_splitting` is enabled). When `false`
  /// the plan is a single native-resolution sub-image at
  /// `(thumb_height, thumb_width)`.
  split: bool,
  /// Tile-grid width (columns) — `1` when not split.
  grid_width: u32,
  /// Tile-grid height (rows) — `1` when not split.
  grid_height: u32,
  /// The resized-whole-image dimensions the grid is cut from
  /// (`tile_size * grid_width`, `tile_size * grid_height`); `(0, 0)` when not
  /// split (the single sub-image uses the thumbnail dims).
  target_width: u32,
  target_height: u32,
  /// The within-budget smart-resize dimensions (HF `smart_resize`,
  /// `:331-364`) — the single sub-image's size when not split, and the
  /// thumbnail's size when split with `use_thumbnail`.
  thumb_width: u32,
  thumb_height: u32,
  /// Whether a thumbnail sub-image is appended after the tiles (only when
  /// `split` and `use_thumbnail` and the grid has `> 1` tile).
  thumbnail: bool,
}

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl TilePlan {
  /// Whether the image is split into a multi-tile grid.
  #[inline(always)]
  pub fn is_split(&self) -> bool {
    self.split
  }

  /// The tile grid `(grid_width, grid_height)` (`(1, 1)` when not split).
  #[inline(always)]
  pub fn grid(&self) -> (u32, u32) {
    (self.grid_width, self.grid_height)
  }

  /// Whether a thumbnail sub-image is appended after the tiles.
  #[inline(always)]
  pub fn has_thumbnail(&self) -> bool {
    self.thumbnail
  }

  /// The number of sub-images this plan produces: `grid_width * grid_height`
  /// tiles (`1` when not split) plus `1` for the thumbnail when present.
  /// Overflow-checked.
  pub fn sub_image_count(&self) -> Result<u32> {
    let tiles = self
      .grid_width
      .checked_mul(self.grid_height)
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "lfm2_vl tiling: tile count (grid_width * grid_height)",
          "u32",
          [
            ("grid_width", u64::from(self.grid_width)),
            ("grid_height", u64::from(self.grid_height)),
          ],
        ))
      })?;
    let extra = u32::from(self.thumbnail);
    tiles.checked_add(extra).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl tiling: sub-image count (tiles + thumbnail)",
        "u32",
        [("tiles", u64::from(tiles)), ("thumbnail", u64::from(extra))],
      ))
    })
  }
}

/// Round `number` to the closest integer multiple of `factor` — HF
/// `round_by_factor` (`image_processing_lfm2_vl.py:40-42`):
/// `round(number / factor) * factor`. `factor` must be `> 0`.
///
/// Python's built-in `round` is round-half-to-**even** (banker's rounding), so a
/// quotient landing exactly on `.5` ties to the nearest even integer (e.g.
/// `round(6.5) == 6`, `round(7.5) == 8`) — *not* round-half-away-from-zero like
/// Rust's `f64::round`. The two diverge on exact half-ties: `208 / 32 == 6.5`
/// rounds to `6` (HF) → `192`, but `6.5_f64.round() == 7.0` → `224`. When the
/// rounded dims then stay within the patch budget, that one-multiple gap
/// propagates into different tile grids / image-token counts than the reference.
/// The inputs here are integer pixel dims, so the quotient `number / factor` is
/// rational and the tie (`number % factor == factor / 2`, `factor` even) is
/// resolved exactly with integer arithmetic — no float rounding involved.
#[cfg(feature = "lfm2-vl")]
fn round_by_factor(number: u32, factor: u32) -> Result<u32> {
  require_positive_u32("lfm2_vl tiling: round_by_factor factor", factor)?;
  // Round `number / factor` to the nearest integer with ties-to-even, matching
  // Python's `round`. `q` is the floor quotient, `r` the remainder; compare
  // `2 * r` against `factor` to classify below / above / exactly-half. `2 * r`
  // is `< 2 * factor <= 2 * u32::MAX`, so widen to `u64` to avoid an overflow.
  let q = number / factor;
  let r = number % factor;
  let twice_r = u64::from(r) * 2;
  let factor_w = u64::from(factor);
  let rounded_q = if twice_r < factor_w {
    q
  } else if twice_r > factor_w {
    q + 1
  } else if q.is_multiple_of(2) {
    // Exact half-tie: round to the even neighbour (HF / Python `round`).
    q
  } else {
    q + 1
  };
  // `rounded_q * factor` is the closest multiple, bounded by `number` rounded up
  // by at most `factor`; that can exceed `u32::MAX` only for a `number` already
  // near the top of the range (the source dims are capped well below it).
  // Compute in `u64` and narrow with a checked cast.
  let scaled = u64::from(rounded_q) * factor_w;
  u32::try_from(scaled).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "lfm2_vl tiling: round_by_factor result",
      "rounded multiple must fit in u32",
      smol_str::format_smolstr!("{scaled}"),
    ))
  })
}

/// HF `find_closest_aspect_ratio` (`image_processing_lfm2_vl.py:45-82`): from
/// the candidate `(w, h)` tile grids, pick the one whose `w/h` is closest to the
/// image's `width/height`; ties break toward the grid whose target area best
/// matches the image area (`area > 0.5 * image_size^2 * w * h`).
///
/// `target_ratios` is the candidate list from [`target_ratios`] (already sorted
/// by tile count). Returns `(grid_width, grid_height)`.
#[cfg(feature = "lfm2-vl")]
fn find_closest_aspect_ratio(
  aspect_ratio: f64,
  target_ratios: &[(u32, u32)],
  width: u32,
  height: u32,
  image_size: u32,
) -> (u32, u32) {
  let mut best_ratio_diff = f64::INFINITY;
  let mut best_ratio = (1u32, 1u32);
  let area = f64::from(width) * f64::from(height);
  for &(w, h) in target_ratios {
    let target_aspect_ratio = f64::from(w) / f64::from(h);
    let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();
    if ratio_diff < best_ratio_diff {
      best_ratio_diff = ratio_diff;
      best_ratio = (w, h);
    } else if ratio_diff == best_ratio_diff {
      // Tie-break: prefer the grid whose target area better matches the image
      // area (`area > 0.5 * image_size^2 * w * h`).
      let target_area = f64::from(image_size) * f64::from(image_size) * f64::from(w) * f64::from(h);
      if area > 0.5 * target_area {
        best_ratio = (w, h);
      }
    }
  }
  best_ratio
}

/// HF `Lfm2VlImageProcessor._target_ratios` (`image_processing_lfm2_vl.py:252-261`):
/// every `(w, h)` with `1 <= w, h` and `min_tiles <= w * h <= max_tiles`,
/// de-duplicated and sorted by tile count `w * h`. `n` ranges over
/// `[min_tiles, max_tiles]` but the `w, h <= n` bound plus the product filter
/// makes the set independent of the iteration bookkeeping; this builds it
/// directly. `max_tiles` is bounded at config load
/// ([`MAX_TILES`](super::config::MAX_TILES)) so `max_tiles^2` stays a bounded
/// reservation / loop; the count is still reserved fallibly.
#[cfg(feature = "lfm2-vl")]
fn target_ratios(min_tiles: u32, max_tiles: u32) -> Result<Vec<(u32, u32)>> {
  // The candidate set is `{(w, h) : min_tiles <= w*h <= max_tiles}`, whose size
  // is bounded by `max_tiles^2`. `max_tiles` is capped at load, so this is a
  // bounded reservation; the `checked_mul` guards the arithmetic regardless, and
  // the fallible reserve avoids aborting on an infallible `with_capacity`.
  let cap = (max_tiles as usize)
    .checked_mul(max_tiles as usize)
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl tiling: target-ratio candidate bound (max_tiles^2)",
        "usize",
        [("max_tiles", u64::from(max_tiles))],
      ))
    })?;
  let mut ratios: Vec<(u32, u32)> = Vec::new();
  reserve_or_error(&mut ratios, "lfm2_vl tiling: target ratio", cap)?;
  let (min_prod, max_prod) = (u64::from(min_tiles), u64::from(max_tiles));
  for w in 1..=max_tiles {
    for h in 1..=max_tiles {
      // `w, h <= max_tiles <= u32::MAX`, so the product is exact in `u64` and
      // cannot wrap (it never narrows back to `u32`; the comparands widen too).
      let prod = u64::from(w) * u64::from(h);
      if prod >= min_prod && prod <= max_prod {
        ratios.push((w, h));
      }
    }
  }
  // Sort by tile count `w*h` (HF's `key=lambda x: x[0]*x[1]`). The `(w, h)`
  // candidates are already unique here (each pair generated once), so no dedup
  // pass is needed. HF's `sorted(set(...))` leaves the *intra-product* order
  // (ties on `w*h`) to Python's hash-based set iteration, which differs from
  // this stable insertion order — but the only consumer,
  // [`find_closest_aspect_ratio`], breaks an exact `ratio_diff` tie by the
  // area condition `area > 0.5 * tile^2 * w*h`, which selects the same grid
  // regardless of the candidate order (verified to produce identical grids to
  // HF's order across the full source-dimension range). So this deterministic
  // stable order is outcome-equivalent to HF's set order.
  // Sort key in `u64` — the product is exact and cannot wrap (`w, h <= max_tiles`).
  ratios.sort_by_key(|&(w, h)| u64::from(w) * u64::from(h));
  Ok(ratios)
}

/// HF `Lfm2VlImageProcessor._get_grid_layout` (`image_processing_lfm2_vl.py:263-281`):
/// choose the tile grid for `(height, width)` and return
/// `(grid_width, grid_height, target_width, target_height)` where
/// `target_* = tile_size * grid_*` (the resized-whole-image size the grid is cut
/// from). `min_tiles` / `max_tiles` / `tile_size` are the HF knobs.
#[cfg(feature = "lfm2-vl")]
fn get_grid_layout(
  height: u32,
  width: u32,
  min_tiles: u32,
  max_tiles: u32,
  tile_size: u32,
) -> Result<(u32, u32, u32, u32)> {
  let aspect_ratio = f64::from(width) / f64::from(height);
  let ratios = target_ratios(min_tiles, max_tiles)?;
  let (grid_width, grid_height) =
    find_closest_aspect_ratio(aspect_ratio, &ratios, width, height, tile_size);
  let target_width = tile_size.checked_mul(grid_width).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "lfm2_vl tiling: target_width (tile_size * grid_width)",
      "u32",
      [
        ("tile_size", u64::from(tile_size)),
        ("grid_width", u64::from(grid_width)),
      ],
    ))
  })?;
  let target_height = tile_size.checked_mul(grid_height).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "lfm2_vl tiling: target_height (tile_size * grid_height)",
      "u32",
      [
        ("tile_size", u64::from(tile_size)),
        ("grid_height", u64::from(grid_height)),
      ],
    ))
  })?;
  Ok((grid_width, grid_height, target_width, target_height))
}

/// HF `Lfm2VlImageProcessor.smart_resize` (`image_processing_lfm2_vl.py:331-364`):
/// rescale `(height, width)` so both dims are divisible by
/// `encoder_patch_size * downsample_factor`, the pixel count is in
/// `[min_image_tokens, max_image_tokens] * (encoder_patch_size *
/// downsample_factor)^2`-derived bounds, and the aspect ratio is preserved.
/// Returns `(new_width, new_height)` (HF returns `(w_bar, h_bar)`).
#[cfg(feature = "lfm2-vl")]
fn smart_resize(
  height: u32,
  width: u32,
  downsample_factor: u32,
  min_image_tokens: u32,
  max_image_tokens: u32,
  encoder_patch_size: u32,
) -> Result<(u32, u32)> {
  let total_factor = encoder_patch_size
    .checked_mul(downsample_factor)
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl tiling: total_factor (encoder_patch_size * downsample_factor)",
        "u32",
        [
          ("encoder_patch_size", u64::from(encoder_patch_size)),
          ("downsample_factor", u64::from(downsample_factor)),
        ],
      ))
    })?;
  require_positive_u32("lfm2_vl tiling: smart_resize total_factor", total_factor)?;
  // `min/max pixels = tokens * encoder_patch_size^2 * downsample_factor^2`
  // = `tokens * total_factor^2`. Computed in f64 (the HF arithmetic is float).
  let tf2 = f64::from(total_factor) * f64::from(total_factor);
  let min_pixels = f64::from(min_image_tokens) * tf2;
  let max_pixels = f64::from(max_image_tokens) * tf2;

  let mut h_bar = total_factor.max(round_by_factor(height, total_factor)?);
  let mut w_bar = total_factor.max(round_by_factor(width, total_factor)?);
  let hw = f64::from(height) * f64::from(width);
  let bar_pixels = f64::from(h_bar) * f64::from(w_bar);

  if bar_pixels > max_pixels {
    // `beta = sqrt(h*w / max_pixels)`; `*_bar = max(tf, floor(dim/beta/tf)*tf)`.
    let beta = (hw / max_pixels).sqrt();
    h_bar = total_factor.max(floor_to_factor(f64::from(height) / beta, total_factor)?);
    w_bar = total_factor.max(floor_to_factor(f64::from(width) / beta, total_factor)?);
  } else if bar_pixels < min_pixels {
    // `beta = sqrt(min_pixels / (h*w))`; `*_bar = ceil(dim*beta/tf)*tf`.
    let beta = (min_pixels / hw).sqrt();
    h_bar = ceil_to_factor(f64::from(height) * beta, total_factor)?;
    w_bar = ceil_to_factor(f64::from(width) * beta, total_factor)?;
  }
  Ok((w_bar, h_bar))
}

/// `floor(value / factor) * factor` (HF `math.floor(dim/beta/tf)*tf`) as `u32`,
/// with a checked narrowing cast. `value >= 0`.
#[cfg(feature = "lfm2-vl")]
fn floor_to_factor(value: f64, factor: u32) -> Result<u32> {
  let q = (value / f64::from(factor)).floor() * f64::from(factor);
  narrow_dim("lfm2_vl tiling: floor_to_factor", q)
}

/// `ceil(value / factor) * factor` (HF `math.ceil(dim*beta/tf)*tf`) as `u32`,
/// with a checked narrowing cast. `value >= 0`.
#[cfg(feature = "lfm2-vl")]
fn ceil_to_factor(value: f64, factor: u32) -> Result<u32> {
  let q = (value / f64::from(factor)).ceil() * f64::from(factor);
  narrow_dim("lfm2_vl tiling: ceil_to_factor", q)
}

/// Narrow a non-negative f64 pixel dimension to `u32`, erroring on overflow /
/// non-finiteness rather than saturating the `as` cast.
#[cfg(feature = "lfm2-vl")]
fn narrow_dim(context: &'static str, value: f64) -> Result<u32> {
  if !(value.is_finite() && (0.0..=f64::from(u32::MAX)).contains(&value)) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "resized dimension must be a finite non-negative value within u32",
      smol_str::format_smolstr!("{value}"),
    )));
  }
  Ok(value as u32)
}

/// HF `Lfm2VlImageProcessor._is_image_too_large` (`image_processing_lfm2_vl.py:366-380`):
/// `true` if `round_by_factor(height) * round_by_factor(width) >
/// max_image_tokens * encoder_patch_size^2 * downsample_factor^2 *
/// max_pixels_tolerance`, where the round floors to `encoder_patch_size` (HF
/// uses `max(encoder_patch_size, round_by_factor(dim, total_factor))`).
#[cfg(feature = "lfm2-vl")]
fn is_image_too_large(
  height: u32,
  width: u32,
  max_image_tokens: u32,
  encoder_patch_size: u32,
  downsample_factor: u32,
  max_pixels_tolerance: f32,
) -> Result<bool> {
  let total_factor = encoder_patch_size
    .checked_mul(downsample_factor)
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl tiling: too_large total_factor",
        "u32",
        [
          ("encoder_patch_size", u64::from(encoder_patch_size)),
          ("downsample_factor", u64::from(downsample_factor)),
        ],
      ))
    })?;
  let h_bar = encoder_patch_size.max(round_by_factor(height, total_factor)?);
  let w_bar = encoder_patch_size.max(round_by_factor(width, total_factor)?);
  let lhs = f64::from(h_bar) * f64::from(w_bar);
  // `max_image_tokens * encoder_patch_size^2 * downsample_factor^2 * tol`
  // = `max_image_tokens * total_factor^2 * tol`.
  let tf2 = f64::from(total_factor) * f64::from(total_factor);
  let rhs = f64::from(max_image_tokens) * tf2 * f64::from(max_pixels_tolerance);
  Ok(lhs > rhs)
}

/// Compute the tile-layout [`TilePlan`] for an image of `(height, width)` under
/// `cfg`'s tiling parameters — the pure host-integer core of the HF
/// `resize_and_split` (`image_processing_lfm2_vl.py:382-436`), separated from
/// the pixel work so the grid math is oracle-testable.
///
/// Mirrors `resize_and_split`'s `do_image_splitting = not min_tiles == max_tiles
/// == 1` shortcut and the `is_image_large and do_image_splitting` branch: a
/// large image with splitting enabled gets the multi-tile grid (+ a thumbnail
/// when `use_thumbnail` and the grid has `> 1` tile); otherwise a single
/// smart-resized sub-image.
///
/// # Errors
/// - [`Error::OutOfRange`] if `width`/`height` is `0`;
/// - propagates the grid-math / smart-resize overflow + narrowing errors.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub fn plan_tiles(height: u32, width: u32, cfg: &Lfm2VlProcessorConfig) -> Result<TilePlan> {
  require_positive_u32("lfm2_vl tiling: height", height)?;
  require_positive_u32("lfm2_vl tiling: width", width)?;
  let t = &cfg.tiling;
  // HF `resize_and_split` (`:397`): splitting also requires a non-degenerate
  // `[min_tiles, max_tiles]` band (`not min_tiles == max_tiles == 1`), AND the
  // model's `do_image_splitting` flag (HF `_preprocess:464-469` forces
  // `min_tiles = max_tiles = 1` when the flag is off — equivalent to disabling).
  let splitting_enabled = t.do_image_splitting && !(t.min_tiles == 1 && t.max_tiles == 1);
  // `downsample_factor` lives on the parent config (a single model-wide knob),
  // not the tiling block (HF carries it as a separate processor kwarg).
  let downsample_factor = cfg.downsample_factor;

  let too_large = is_image_too_large(
    height,
    width,
    t.max_image_tokens,
    t.encoder_patch_size,
    downsample_factor,
    t.max_pixels_tolerance,
  )?;

  // The within-budget smart-resize size (the single sub-image's dims, and the
  // thumbnail's dims when split).
  let (thumb_width, thumb_height) = smart_resize(
    height,
    width,
    downsample_factor,
    t.min_image_tokens,
    t.max_image_tokens,
    t.encoder_patch_size,
  )?;

  if too_large && splitting_enabled {
    let (grid_width, grid_height, target_width, target_height) =
      get_grid_layout(height, width, t.min_tiles, t.max_tiles, t.tile_size)?;
    // HF `crop_image_to_patches:317`: a thumbnail is appended only when
    // `use_thumbnail` AND the grid has more than one tile. The grid sides are
    // `>= 1`, so the product is `1` iff both are `1`; test that directly to
    // avoid forming the product.
    let multi_tile = !(grid_width == 1 && grid_height == 1);
    Ok(TilePlan {
      split: multi_tile,
      grid_width,
      grid_height,
      target_width,
      target_height,
      thumb_width,
      thumb_height,
      thumbnail: t.use_thumbnail && multi_tile,
    })
  } else {
    // Single native-resolution sub-image at the smart-resize size.
    Ok(TilePlan {
      split: false,
      grid_width: 1,
      grid_height: 1,
      target_width: 0,
      target_height: 0,
      thumb_width,
      thumb_height,
      thumbnail: false,
    })
  }
}

/// Split one interleaved RGB image into the HF `Lfm2VlImageProcessor` tile
/// sequence: a [`Lfm2VlImageInputs`] per tile (+ an optional thumbnail), each
/// the patchified NaFlex triple the vision tower consumes.
///
/// Faithful port of the HF `resize_and_split` + `crop_image_to_patches` +
/// `_preprocess` patchify (`image_processing_lfm2_vl.py:382-558`), which mlx-vlm
/// omits (it forces `do_image_splitting = False`). The plan is computed by
/// [`plan_tiles`]; this executes it:
///
/// - **split** (a large image, splitting enabled, `> 1`-tile grid): the whole
///   image is bilinear-resized to `(target_height, target_width)` and cut into a
///   `grid_height x grid_width` grid of `tile_size x tile_size` tiles (HF
///   `split_to_tiles`, `:310`). Each tile patchifies at `encoder_patch_size`
///   into a `(tile_size/P, tile_size/P)` patch grid. When `use_thumbnail`, the
///   whole image (smart-resized to the within-budget `(thumb_height,
///   thumb_width)`) is appended as a final sub-image (HF `:317-326`).
/// - **single** (small image or splitting disabled): the whole image is
///   smart-resized to `(thumb_height, thumb_width)` and patchified as ONE
///   sub-image (HF `:427-431`).
///
/// The returned sub-images are in HF batch order (tiles row-major, then the
/// thumbnail). Their per-tile grids drive [`expand_image_tokens`] (each
/// bracketed by `image_start`/`image_end` and concatenated — the mlx-vlm
/// `_patched_call:378-404` per-sub-image layout), and each is encoded by
/// [`Lfm2Vl::encode_image_inputs`](super::model::Lfm2Vl::encode_image_inputs).
///
/// `tile_size` must be a whole multiple of `encoder_patch_size` (enforced by
/// [`Lfm2VlProcessorConfig::with_tiling`]), and the produced patch grids
/// (`tile_size/P` per side, and `thumb_*/P`) must fit `max_num_patches`.
///
/// # Errors
/// - [`Error::OutOfRange`] for a zero dimension;
/// - [`Error::LengthMismatch`] if `rgb.len()` disagrees with `width*height*3`;
/// - [`Error::CapExceeded`] if a tile / thumbnail patch grid exceeds
///   `max_num_patches`;
/// - propagates the [`plan_tiles`] grid-math, the [`resize`], and the
///   [`Array::from_slice`] errors.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub fn tile_image(
  rgb: &[u8],
  width: u32,
  height: u32,
  cfg: &Lfm2VlProcessorConfig,
) -> Result<Vec<Lfm2VlImageInputs>> {
  require_positive_u32("lfm2_vl tile_image: width", width)?;
  require_positive_u32("lfm2_vl tile_image: height", height)?;
  // Validate the RGB byte count up front (the same capped-extent product the
  // native path uses) so a wrong-length slice is a typed error before any
  // resize. The per-axis caps also bound the source-clone sizing below.
  let channels = RGB_CHANNELS as usize;
  let expected_rgb_len = elem_count(
    "lfm2_vl tile_image: rgb byte count (width * height * channels)",
    &[
      Extent::new("lfm2_vl tile_image: width", width as usize, MAX_SOURCE_DIM)?,
      Extent::new(
        "lfm2_vl tile_image: height",
        height as usize,
        MAX_SOURCE_DIM,
      )?,
      Extent::new(
        "lfm2_vl tile_image: channels",
        channels,
        RGB_CHANNELS as usize,
      )?,
    ],
    MAX_SOURCE_PIXELS,
  )?;
  if rgb.len() != expected_rgb_len {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "lfm2_vl tile_image: rgb slice length vs width*height*channels",
      expected_rgb_len,
      rgb.len(),
    )));
  }

  let plan = plan_tiles(height, width, cfg)?;
  let p = cfg.tiling.encoder_patch_size;
  let mut out: Vec<Lfm2VlImageInputs> = Vec::new();
  reserve_or_error(
    &mut out,
    "lfm2_vl tile_image: sub-image",
    plan.sub_image_count()? as usize,
  )?;

  if plan.split {
    // Resize the whole image to (target_height, target_width) — the grid is cut
    // from this. `vlm::resize` takes (height, width) and returns RGBA8.
    let resized = resize_rgb_to_rgba(rgb, width, height, plan.target_height, plan.target_width)?;
    let tile_h = plan.target_height / plan.grid_height; // == tile_size (exact)
    let tile_w = plan.target_width / plan.grid_width; // == tile_size (exact)
    let grid_rows = tile_h / p; // patches per tile side
    let grid_cols = tile_w / p;
    // Each tile, row-major (HF `split_to_tiles` orders tiles row-major). The
    // tile origins `gy * tile_h` / `gx * tile_w` are bounded by
    // `target_height` / `target_width` (which `get_grid_layout` produced via a
    // `checked_mul`), but guard the products too so no path multiplies unchecked.
    for gy in 0..plan.grid_height {
      for gx in 0..plan.grid_width {
        let y0 = gy.checked_mul(tile_h).ok_or_else(|| {
          Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
            "lfm2_vl tile_image: tile origin y (grid_row * tile_h)",
            "u32",
            [("grid_row", u64::from(gy)), ("tile_h", u64::from(tile_h))],
          ))
        })?;
        let x0 = gx.checked_mul(tile_w).ok_or_else(|| {
          Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
            "lfm2_vl tile_image: tile origin x (grid_col * tile_w)",
            "u32",
            [("grid_col", u64::from(gx)), ("tile_w", u64::from(tile_w))],
          ))
        })?;
        out.push(patchify_rgba_region(
          resized.buf(),
          resized.width(),
          x0,
          y0,
          tile_w,
          tile_h,
          grid_rows,
          grid_cols,
          p,
          cfg,
        )?);
      }
    }
    if plan.thumbnail {
      // The thumbnail is the WHOLE image smart-resized to (thumb_h, thumb_w).
      let thumb = resize_rgb_to_rgba(rgb, width, height, plan.thumb_height, plan.thumb_width)?;
      let t_rows = plan.thumb_height / p;
      let t_cols = plan.thumb_width / p;
      out.push(patchify_rgba_region(
        thumb.buf(),
        thumb.width(),
        0,
        0,
        plan.thumb_width,
        plan.thumb_height,
        t_rows,
        t_cols,
        p,
        cfg,
      )?);
    }
  } else {
    // Single sub-image: smart-resize the whole image to (thumb_h, thumb_w).
    let resized = resize_rgb_to_rgba(rgb, width, height, plan.thumb_height, plan.thumb_width)?;
    let s_rows = plan.thumb_height / p;
    let s_cols = plan.thumb_width / p;
    out.push(patchify_rgba_region(
      resized.buf(),
      resized.width(),
      0,
      0,
      plan.thumb_width,
      plan.thumb_height,
      s_rows,
      s_cols,
      p,
      cfg,
    )?);
  }
  Ok(out)
}

/// An owned RGBA8 resized buffer + its dimensions — the resize output the tile
/// patchify reads regions from.
#[cfg(feature = "lfm2-vl")]
struct ResizedRgba {
  buf: Vec<u8>,
  width: u32,
}

#[cfg(feature = "lfm2-vl")]
impl ResizedRgba {
  #[inline(always)]
  fn buf(&self) -> &[u8] {
    &self.buf
  }
  #[inline(always)]
  fn width(&self) -> u32 {
    self.width
  }
}

/// Bilinear-resize an interleaved RGB image to `(target_height, target_width)`,
/// returning the owned RGBA8 buffer (the PIL-bit-exact resample the SigLIP2 path
/// uses). The caller has already validated `rgb.len()` and the source dims.
#[cfg(feature = "lfm2-vl")]
fn resize_rgb_to_rgba(
  rgb: &[u8],
  width: u32,
  height: u32,
  target_height: u32,
  target_width: u32,
) -> Result<ResizedRgba> {
  require_positive_u32("lfm2_vl tile resize: target_height", target_height)?;
  require_positive_u32("lfm2_vl tile resize: target_width", target_width)?;
  // Build an owned RgbImage over the caller's bytes (image's `from_raw` needs an
  // owned container), reserving the already-capped exact length fallibly.
  let mut src_buf: Vec<u8> = Vec::new();
  reserve_or_error(
    &mut src_buf,
    "lfm2_vl tile resize: source RGB clone",
    rgb.len(),
  )?;
  src_buf.extend_from_slice(rgb);
  let src_img: ::image::RgbImage = ::image::ImageBuffer::from_raw(width, height, src_buf)
    .ok_or_else(|| {
      Error::LengthMismatch(LengthMismatchPayload::new(
        "lfm2_vl tile resize: RgbImage::from_raw length",
        rgb.len(),
        rgb.len(),
      ))
    })?;
  let src_dyn = ::image::DynamicImage::ImageRgb8(src_img);
  let resized_dyn = resize(
    &src_dyn,
    (target_height, target_width),
    ResizeFilter::Bilinear,
  )?;
  let resized = resized_dyn.as_rgba8().ok_or_else(|| {
    Error::InvariantViolation(InvariantViolationPayload::new(
      "lfm2_vl tile resize: resized image format",
      "vlm::image::resize must return an RGBA8 image",
    ))
  })?;
  let buf = resized.as_raw();
  let mut owned: Vec<u8> = Vec::new();
  reserve_or_error(&mut owned, "lfm2_vl tile resize: RGBA clone", buf.len())?;
  owned.extend_from_slice(buf);
  Ok(ResizedRgba {
    buf: owned,
    width: target_width,
  })
}

/// Patchify a `(region_h, region_w)` rectangular region (top-left at
/// `(x0, y0)`) of an RGBA8 buffer of row width `buf_width` into a
/// [`Lfm2VlImageInputs`] at the fixed `(grid_rows, grid_cols)` patch grid,
/// `patch_size = p` — the HF `convert_image_to_patches` + `pad_along_first_dim`
/// (`image_processing_lfm2_vl.py:134-163, 527-535`) over a sub-image, reusing
/// the SigLIP per-channel affine the native path uses.
///
/// The region must be exactly `(grid_rows * p, grid_cols * p)` and its patch
/// grid must fit `max_num_patches` (the pad target). Pixels past the active
/// `grid_rows * grid_cols` rows are zero-padded; the `pixel_attention_mask` is
/// `1` for the active rows and `0` for the padding; `spatial_shapes` is
/// `(grid_rows, grid_cols)`.
#[cfg(feature = "lfm2-vl")]
#[allow(clippy::too_many_arguments)]
fn patchify_rgba_region(
  buf: &[u8],
  buf_width: u32,
  x0: u32,
  y0: u32,
  region_w: u32,
  region_h: u32,
  grid_rows: u32,
  grid_cols: u32,
  p: u32,
  cfg: &Lfm2VlProcessorConfig,
) -> Result<Lfm2VlImageInputs> {
  let max_num_patches = cfg.max_num_patches;
  // The region must tile exactly into `(grid_rows, grid_cols)` patches.
  if grid_rows.checked_mul(p) != Some(region_h) || grid_cols.checked_mul(p) != Some(region_w) {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "lfm2_vl tile patchify: region must be (grid_rows * p, grid_cols * p)",
      "the tile / thumbnail dimensions must be a whole multiple of encoder_patch_size",
    )));
  }
  // Active patch count `grid_rows * grid_cols` must fit the pad budget.
  let active = u64::from(grid_rows)
    .checked_mul(u64::from(grid_cols))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl tile patchify: active patch count (grid_rows * grid_cols)",
        "u64",
        [
          ("grid_rows", u64::from(grid_rows)),
          ("grid_cols", u64::from(grid_cols)),
        ],
      ))
    })?;
  if active > u64::from(max_num_patches) {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "lfm2_vl tile patchify: sub-image patch grid",
      "max_num_patches",
      u64::from(max_num_patches),
      active,
    )));
  }

  // Per-patch flattened width = C * p^2; the pixel buffer is
  // `max_num_patches * per_patch` (overflow-checked).
  let pu = p as usize;
  let channels = RGB_CHANNELS as usize;
  let per_patch = pu
    .checked_mul(pu)
    .and_then(|pp| pp.checked_mul(channels))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl tile patchify: per-patch width (p^2 * channels)",
        "usize",
        [("p", u64::from(p)), ("channels", RGB_CHANNELS as u64)],
      ))
    })?;
  let total_floats = (max_num_patches as usize)
    .checked_mul(per_patch)
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "lfm2_vl tile patchify: pixel_values size (max_num_patches * per_patch)",
        "usize",
        [
          ("max_num_patches", u64::from(max_num_patches)),
          ("per_patch", per_patch as u64),
        ],
      ))
    })?;

  let mut pixel_values: Vec<f32> = Vec::new();
  reserve_or_error(
    &mut pixel_values,
    "lfm2_vl tile patchify: pixel_values f32",
    total_floats,
  )?;
  pixel_values.resize(total_floats, 0.0f32);

  let src_c = RGBA_CHANNELS as usize;
  let src_row_stride = (buf_width as usize) * src_c;
  let row_bytes = pu * channels;
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
  let (gr, gc) = (grid_rows as usize, grid_cols as usize);
  let (x0u, y0u) = (x0 as usize, y0 as usize);
  for py in 0..gr {
    for px in 0..gc {
      let patch_idx = py * gc + px;
      let out_offset = patch_idx * per_patch;
      for r in 0..pu {
        let src_y = y0u + py * pu + r;
        let src_x = x0u + px * pu;
        let src_off = src_y * src_row_stride + src_x * src_c;
        let dst_off = out_offset + r * row_bytes;
        normalize_row_rgba(
          &buf[src_off..src_off + pu * src_c],
          &mut pixel_values[dst_off..dst_off + row_bytes],
          scale,
          bias,
        );
      }
    }
  }

  let n_active = active as usize;
  let mut mask: Vec<i32> = Vec::new();
  reserve_or_error(
    &mut mask,
    "lfm2_vl tile patchify: pixel_attention_mask i32",
    max_num_patches as usize,
  )?;
  mask.resize(max_num_patches as usize, 0i32);
  for slot in mask.iter_mut().take(n_active) {
    *slot = 1;
  }
  let spatial = [grid_rows as i32, grid_cols as i32];

  let pixel_values =
    Array::from_slice::<f32>(&pixel_values, &(max_num_patches as usize, per_patch))?;
  let pixel_attention_mask = Array::from_slice::<i32>(&mask, &(max_num_patches as usize,))?;
  let spatial_shapes = Array::from_slice::<i32>(&spatial, &(2usize,))?;
  Ok(Lfm2VlImageInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
  })
}

#[cfg(all(test, feature = "lfm2-vl"))]
mod tests;
