//! SigLIP2 NaFlex dual-tower configuration.
//!
//! Ports the `TextConfig` / `VisionConfig` / `ModelArgs` dataclasses of
//! [`mlx-embeddings`'s `models/siglip.py`](https://github.com/Blaizzy/mlx-embeddings/blob/main/mlx_embeddings/models/siglip.py)
//! (lines 12-64), with defaults pinned to the
//! `google/siglip2-base-patch16-naflex` checkpoint. As elsewhere in the
//! crate, parsing is forward-compatible: an unmodeled key parses cleanly
//! and an absent key falls back to its default (`#[serde(default)]`, not
//! `deny_unknown_fields`) — matching `ModelArgs.from_dict`, which filters
//! to the known signature parameters.
//!
//! Each config exposes a [`validate`](VisionConfig::validate) that pins
//! every architecture-defining field onto the shared
//! [`crate::model_validation`] toolkit **before** any tensor is
//! allocated, so a corrupt / hostile / wrong-architecture `config.json`
//! fails fast with a typed [`crate::Error`] instead of building the
//! wrong graph. Every dimension / count must be positive, every
//! derived product is overflow-checked (a wrapped size would be UB),
//! and the architecture-defining fields (`model_type`, the RGB channel
//! count, the per-head split) are pinned. `mlxrs` is a library, so a
//! merely *large* (but positive, non-overflowing) field is not rejected:
//! the consuming application owns input bounding. A pathological config
//! therefore yields a typed overflow / allocation error (via the
//! overflow-checked arithmetic and the fallible allocations downstream),
//! never a panic.

use crate::{
  error::{Error, ParsePayload, Result},
  model_validation::{pin_i32, pin_str, require_divisible, require_positive},
};

// ════════════════════════════════ TextConfig ═══════════════════════════════

/// SigLIP2 text-tower configuration. Defaults match
/// `google/siglip2-base-patch16-naflex`'s text config (a standard SigLIP2
/// text transformer + pooled projection).
///
/// Ports `siglip.py`'s `TextConfig` (lines 12-26), including its
/// `__post_init__` rule that `projection_size` defaults to `hidden_size`
/// when absent (handled by [`TextConfig::projection_size`]).
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
  /// Architecture id (`"siglip2_text_model"`).
  #[serde(default = "default_text_model_type")]
  model_type: String,
  /// Token-embedding table size / the text-tower vocabulary.
  #[serde(default = "default_text_vocab_size")]
  pub vocab_size: i32,
  /// Maximum text position-embedding length (the sticky-EOS sequence
  /// length the text tower pools the last token of).
  #[serde(default = "default_text_max_position_embeddings")]
  pub max_position_embeddings: i32,
  /// Transformer hidden / embedding dimension.
  #[serde(default = "default_hidden_size")]
  pub hidden_size: i32,
  /// Feed-forward intermediate dimension.
  #[serde(default = "default_intermediate_size")]
  pub intermediate_size: i32,
  /// Number of attention heads.
  #[serde(default = "default_num_attention_heads")]
  pub num_attention_heads: i32,
  /// Number of transformer encoder layers.
  #[serde(default = "default_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// `eps` shared by every `LayerNorm` (`1e-6`).
  #[serde(default = "default_layer_norm_eps")]
  pub layer_norm_eps: f64,
  /// Contrastive-projection output width. Absent in the checkpoint ⇒
  /// `None`, which [`TextConfig::projection_size`] resolves to
  /// `hidden_size` (mirroring `__post_init__`).
  #[serde(default)]
  projection_size: Option<i32>,
}

// ═══════════════════════════════ VisionConfig ══════════════════════════════

/// SigLIP2 NaFlex vision-tower configuration. Defaults match
/// `google/siglip2-base-patch16-naflex`'s vision config.
///
/// Ports `siglip.py`'s `VisionConfig` (lines 29-44). The NaFlex
/// parameters [`num_patches`](VisionConfig::num_patches) (the learned
/// position-embedding grid; `256` ⇒ a `16 x 16` grid that
/// [`crate::ops::interpolation::bilinear_interpolate`] resizes per image)
/// and [`max_num_patches`](VisionConfig::max_num_patches) (the per-image
/// patch budget the [`crate::embeddings::siglip2_naflex::processing`]
/// stage resizes down to) are both `Option` upstream; the accessors
/// resolve their checkpoint defaults.
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
  /// Architecture id (`"siglip2_vision_model"`).
  #[serde(default = "default_vision_model_type")]
  model_type: String,
  /// Nominal square image size. For NaFlex this is **not** the runtime
  /// resolution (that is variable, derived from `max_num_patches`); when
  /// `num_patches` is set it does not even size the position-embedding
  /// grid. Retained for parity with the upstream config.
  #[serde(default = "default_image_size")]
  pub image_size: i32,
  /// Patch side length in pixels (`16`). The Conv2d patch-embed stride /
  /// kernel, and the `3 * patch_size^2` flattened-patch width.
  #[serde(default = "default_patch_size")]
  pub patch_size: i32,
  /// Input channel count (`3`, RGB).
  #[serde(default = "default_num_channels")]
  pub num_channels: i32,
  /// Transformer hidden / embedding dimension.
  #[serde(default = "default_hidden_size")]
  pub hidden_size: i32,
  /// Feed-forward intermediate dimension.
  #[serde(default = "default_intermediate_size")]
  pub intermediate_size: i32,
  /// Number of attention heads.
  #[serde(default = "default_num_attention_heads")]
  pub num_attention_heads: i32,
  /// Number of transformer encoder layers.
  #[serde(default = "default_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// `eps` shared by every `LayerNorm` (`1e-6`).
  #[serde(default = "default_layer_norm_eps")]
  pub layer_norm_eps: f64,
  /// Whether the attention-pooling head is present (`true` for the base
  /// checkpoint; the pooled image embedding comes from it).
  #[serde(default = "default_true")]
  pub vision_use_head: bool,
  /// Learned position-embedding count. For SigLIP2 this is the size of
  /// the `position_embedding` table; `256` ⇒ a `16 x 16` grid resampled
  /// per image. Absent ⇒ `None`, resolved by
  /// [`VisionConfig::num_patches`] to `(image_size / patch_size)^2`.
  #[serde(default)]
  num_patches: Option<i32>,
  /// Per-image patch budget for the NaFlex variant (`256` for the base
  /// checkpoint). Absent ⇒ `None`, resolved by
  /// [`VisionConfig::max_num_patches`] to the same `num_patches` default.
  #[serde(default)]
  max_num_patches: Option<i32>,
}

// ═══════════════════════════════ ModelArgs ═════════════════════════════════

/// Top-level SigLIP2 NaFlex model configuration: the two tower configs
/// plus the contrastive head parameters. Ports `siglip.py`'s `ModelArgs`
/// (lines 46-64).
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Siglip2NaflexConfig {
  /// Text-tower config (`text_config`).
  pub text_config: TextConfig,
  /// Vision-tower config (`vision_config`).
  pub vision_config: VisionConfig,
  /// Top-level architecture id (`"siglip2"`).
  #[serde(default = "default_model_type")]
  model_type: String,
  /// Number of classifier labels. `0` ⇒ the contrastive dual-tower path
  /// (text + vision + `logit_scale`/`logit_bias`); `> 0` ⇒ a vision
  /// classifier head instead. The embeddings port targets the `0` path.
  #[serde(default)]
  pub num_labels: i32,
}

// ── defaults (single source of truth; the `*_constants_match_defaults`
//    test pins these against the named architecture constants) ──

#[cfg(feature = "siglip2-naflex")]
fn default_model_type() -> String {
  "siglip2".to_string()
}
#[cfg(feature = "siglip2-naflex")]
fn default_text_model_type() -> String {
  "siglip2_text_model".to_string()
}
#[cfg(feature = "siglip2-naflex")]
fn default_vision_model_type() -> String {
  "siglip2_vision_model".to_string()
}
#[cfg(feature = "siglip2-naflex")]
fn default_text_vocab_size() -> i32 {
  32000
}
#[cfg(feature = "siglip2-naflex")]
fn default_text_max_position_embeddings() -> i32 {
  64
}
#[cfg(feature = "siglip2-naflex")]
fn default_hidden_size() -> i32 {
  768
}
#[cfg(feature = "siglip2-naflex")]
fn default_intermediate_size() -> i32 {
  3072
}
#[cfg(feature = "siglip2-naflex")]
fn default_num_attention_heads() -> i32 {
  12
}
#[cfg(feature = "siglip2-naflex")]
fn default_num_hidden_layers() -> i32 {
  12
}
#[cfg(feature = "siglip2-naflex")]
fn default_layer_norm_eps() -> f64 {
  1e-6
}
#[cfg(feature = "siglip2-naflex")]
fn default_image_size() -> i32 {
  256
}
#[cfg(feature = "siglip2-naflex")]
fn default_patch_size() -> i32 {
  16
}
#[cfg(feature = "siglip2-naflex")]
fn default_num_channels() -> i32 {
  3
}
#[cfg(feature = "siglip2-naflex")]
fn default_true() -> bool {
  true
}

/// The base-patch16-naflex default for both `num_patches` and the
/// `max_num_patches` budget: a `16 x 16 = 256`-cell position grid and a
/// 256-patch per-image budget.
#[cfg(feature = "siglip2-naflex")]
pub(crate) const DEFAULT_NUM_PATCHES: i32 = 256;

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
impl TextConfig {
  /// Parse a [`TextConfig`] from an in-memory JSON string (the
  /// `text_config` sub-object of a SigLIP2 `config.json`).
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json)
      .map_err(|e| Error::Parse(ParsePayload::new("TextConfig::from_json", "config JSON", e)))
  }

  /// Architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// Contrastive-projection output width, resolving the upstream
  /// `__post_init__` default (`projection_size or hidden_size`).
  #[inline(always)]
  pub fn projection_size(&self) -> i32 {
    self.projection_size.unwrap_or(self.hidden_size)
  }

  /// Reject a structurally invalid text config with a typed error before
  /// any tensor is built.
  ///
  /// Pins `model_type` to `"siglip2_text_model"`; requires every
  /// dimension / count (`vocab_size`, `hidden_size`, `intermediate_size`,
  /// `projection_size`, `max_position_embeddings`, the layer + head
  /// counts) strictly positive; and requires `hidden_size` divisible by
  /// `num_attention_heads` (the per-head split).
  pub fn validate(&self) -> Result<()> {
    pin_str(
      "TextConfig: model_type",
      self.model_type.as_str(),
      &["siglip2_text_model"],
    )?;
    // Every dimension / count must be strictly positive. `vocab_size` sizes the
    // token-embedding table rows; `hidden_size` / `intermediate_size` /
    // `projection_size` name matmul / embedding axes;
    // `max_position_embeddings` sizes the position-embedding table and the
    // sequence length; the layer / head counts size the encoder. A
    // non-positive value would divide-by-zero or wrap a length downstream, so
    // it is rejected as [`Error::OutOfRange`]. A merely large (positive) value
    // is accepted — the consuming application owns input bounding.
    for (name, value) in [
      ("TextConfig: vocab_size", self.vocab_size),
      ("TextConfig: hidden_size", self.hidden_size),
      ("TextConfig: intermediate_size", self.intermediate_size),
      ("TextConfig: projection_size", self.projection_size()),
      (
        "TextConfig: max_position_embeddings",
        self.max_position_embeddings,
      ),
      ("TextConfig: num_attention_heads", self.num_attention_heads),
      ("TextConfig: num_hidden_layers", self.num_hidden_layers),
    ] {
      require_positive(name, value)?;
    }
    require_divisible(
      "TextConfig: hidden_size",
      self.hidden_size,
      "num_attention_heads",
      self.num_attention_heads,
    )?;
    Ok(())
  }
}

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
impl VisionConfig {
  /// Parse a [`VisionConfig`] from an in-memory JSON string (the
  /// `vision_config` sub-object of a SigLIP2 `config.json`).
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "VisionConfig::from_json",
        "config JSON",
        e,
      ))
    })
  }

  /// Architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// Learned position-embedding count, resolving the upstream default
  /// (`config.num_patches` if set, else `(image_size / patch_size)^2`).
  ///
  /// Returns an error if the fallback would divide by a non-positive
  /// `patch_size` or overflow; on a validated config (after
  /// [`validate`](Self::validate)) the happy path is exact.
  pub fn num_patches(&self) -> Result<i32> {
    if let Some(n) = self.num_patches {
      return Ok(n);
    }
    require_positive("VisionConfig: patch_size", self.patch_size)?;
    let per_side = self.image_size / self.patch_size;
    crate::model_validation::checked_mul(
      "VisionConfig: num_patches ((image_size / patch_size)^2)",
      "per_side",
      per_side,
      "per_side",
      per_side,
    )
  }

  /// Per-image patch budget, resolving the upstream default
  /// (`config.max_num_patches` if set, else the same default as
  /// [`num_patches`](Self::num_patches): `DEFAULT_NUM_PATCHES` (256)).
  #[inline]
  pub fn max_num_patches(&self) -> i32 {
    self.max_num_patches.unwrap_or(DEFAULT_NUM_PATCHES)
  }

  /// The flattened per-patch feature width: `num_channels * patch_size^2`
  /// (`3 * 16^2 = 768` for the base checkpoint) — the width of each row
  /// the NaFlex preprocessing emits and the patch-embed Linear consumes.
  ///
  /// Overflow-checked (`patch_size^2` then `* num_channels`) so a hostile
  /// `patch_size` cannot wrap; non-positive operands are rejected.
  pub fn patch_feature_dim(&self) -> Result<i32> {
    require_positive("VisionConfig: patch_size", self.patch_size)?;
    require_positive("VisionConfig: num_channels", self.num_channels)?;
    let p2 = crate::model_validation::checked_mul(
      "VisionConfig: patch_size^2",
      "patch_size",
      self.patch_size,
      "patch_size",
      self.patch_size,
    )?;
    crate::model_validation::checked_mul(
      "VisionConfig: num_channels * patch_size^2",
      "num_channels",
      self.num_channels,
      "patch_size^2",
      p2,
    )
  }

  /// Reject a structurally invalid vision config with a typed error
  /// before any tensor is built.
  ///
  /// Pins `model_type` to `"siglip2_vision_model"` and `num_channels` to
  /// `3`; requires every dimension / count (`image_size`, `patch_size`,
  /// `hidden_size`, `intermediate_size`, the layer + head counts, the
  /// resolved `num_patches` / `max_num_patches`) strictly positive;
  /// requires `hidden_size` divisible by `num_attention_heads`; and
  /// validates that the derived `patch_feature_dim`
  /// (`num_channels * patch_size^2`) arithmetic does not overflow (a
  /// wrapped width would be UB downstream).
  pub fn validate(&self) -> Result<()> {
    pin_str(
      "VisionConfig: model_type",
      self.model_type.as_str(),
      &["siglip2_vision_model"],
    )?;
    // The patch-embed + flatten path is hardcoded to RGB (3 channels);
    // a deviating count would silently mis-shape the flattened patch row.
    pin_i32("VisionConfig: num_channels", self.num_channels, 3)?;
    // Every dimension / count must be strictly positive. `image_size` feeds the
    // `num_patches` fallback `(image_size / patch_size)^2`; `patch_size` is the
    // Conv2d kernel / stride and the flattened-patch stride; `hidden_size` /
    // `intermediate_size` name matmul axes; the layer / head counts size the
    // encoder. A non-positive value would divide-by-zero or wrap a length
    // downstream. A merely large (positive) value is accepted — the consuming
    // application owns input bounding.
    for (name, value) in [
      ("VisionConfig: image_size", self.image_size),
      ("VisionConfig: patch_size", self.patch_size),
      ("VisionConfig: hidden_size", self.hidden_size),
      ("VisionConfig: intermediate_size", self.intermediate_size),
      (
        "VisionConfig: num_attention_heads",
        self.num_attention_heads,
      ),
      ("VisionConfig: num_hidden_layers", self.num_hidden_layers),
    ] {
      require_positive(name, value)?;
    }
    require_divisible(
      "VisionConfig: hidden_size",
      self.hidden_size,
      "num_attention_heads",
      self.num_attention_heads,
    )?;
    // Resolve + require positive the two patch counts (each sizes a fixed-length
    // per-image position / pixel buffer). `num_patches()` can itself error
    // (non-positive patch_size / overflow); propagate that.
    require_positive("VisionConfig: num_patches", self.num_patches()?)?;
    require_positive("VisionConfig: max_num_patches", self.max_num_patches())?;
    // Validate the flattened-patch-width arithmetic does not overflow.
    // `patch_feature_dim` (`num_channels * patch_size^2`) is the stride of
    // every NaFlex preprocessing row and the patch-embed Linear's input axis.
    // `patch_feature_dim()` is overflow-checked (a wrapped width would be UB)
    // and rejects a non-positive operand, so its result is a positive `i32`;
    // calling it here surfaces an overflow as a typed error before any tensor
    // is shaped.
    self.patch_feature_dim()?;
    Ok(())
  }
}

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
impl Siglip2NaflexConfig {
  /// Parse a [`Siglip2NaflexConfig`] from an in-memory `config.json`
  /// string. A malformed-JSON failure maps to [`Error::Parse`]; absent
  /// keys take their checkpoint defaults; unmodeled keys are ignored.
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "Siglip2NaflexConfig::from_json",
        "config JSON",
        e,
      ))
    })
  }

  /// Top-level architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// Validate both towers and the top-level fields.
  ///
  /// Pins the top-level `model_type` to `"siglip2"`, requires
  /// `num_labels >= 0`, and validates each tower config (see
  /// [`TextConfig::validate`] / [`VisionConfig::validate`]).
  pub fn validate(&self) -> Result<()> {
    pin_str(
      "Siglip2NaflexConfig: model_type",
      self.model_type.as_str(),
      &["siglip2"],
    )?;
    if self.num_labels < 0 {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "Siglip2NaflexConfig: num_labels",
        "must be non-negative (>= 0)",
        smol_str::format_smolstr!("{}", self.num_labels),
      )));
    }
    self.text_config.validate()?;
    self.vision_config.validate()?;
    Ok(())
  }
}

#[cfg(all(test, feature = "siglip2-naflex"))]
mod tests;
