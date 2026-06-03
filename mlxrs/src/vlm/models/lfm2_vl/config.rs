//! LFM2.5-VL configuration — ports `mlx-vlm/mlx_vlm/models/lfm2_vl/config.py`
//! (`TextConfig` / `VisionConfig` / `ModelConfig`).
//!
//! The text-tower config is the **existing** LFM2 LM
//! [`TextConfig`] (the VL `language.py`
//! wraps `mlx_lm.models.lfm2.Lfm2Model` verbatim, so its `text_config` IS the
//! LM `ModelArgs`); it is re-exported here rather than duplicated. This module
//! adds the SigLIP2-style [`VisionConfig`] and the top-level [`ModelConfig`]
//! (the projector / image-token / patch-merge parameters).
//!
//! As elsewhere in the crate, parsing is forward-compatible: an unmodeled key
//! parses cleanly and an absent key falls back to its reference default
//! (`#[serde(default)]`, not `deny_unknown_fields`) — matching
//! `BaseModelConfig.from_dict`. Each config exposes a `validate()` that pins
//! every architecture-defining field with the shared
//! [`crate::model_validation`] toolkit before any tensor is built, so a corrupt
//! / hostile / wrong-architecture `config.json` fails fast with a typed
//! [`crate::Error`] instead of building the wrong graph. `mlxrs` is a library,
//! so a merely *large* (but positive, non-overflowing) field is accepted — the
//! consuming application owns input bounding.

use crate::{
  error::{Error, OutOfRangePayload, ParsePayload, Result},
  model_validation::{checked_mul, require_divisible, require_positive},
};

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use crate::lm::models::lfm2::TextConfig;

// ═══════════════════════════════ VisionConfig ══════════════════════════════

/// LFM2.5-VL SigLIP2-style vision-tower configuration — `config.py`'s
/// `VisionConfig`. Defaults match the `LiquidAI/LFM2.5-VL-450M-MLX-8bit`
/// SigLIP2 vision encoder (`hidden = 768`, `layers = 12`, `heads = 12`,
/// `patch = 16`, `num_patches = 256`, `intermediate = 3072`, `eps = 1e-6`).
///
/// The patch embedding is a **`Linear`** over the processor's pre-flattened
/// `(num_patches, num_channels * patch_size^2)` patches (NOT a `Conv2d`), and
/// the `num_patches`-entry position-embedding table is a square `16 x 16` grid
/// that the [`bicubic_interpolate`](crate::ops::interpolation::bicubic_interpolate)
/// resizes per image (see [`crate::vlm::models::lfm2_vl::vision`]).
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
  /// Architecture id (`"lfm2_vl"`; the SigLIP2 vision encoder also accepts
  /// `"siglip2_vision_model"`, matching `vision.py`'s `VisionModel` guard).
  #[serde(default = "default_vision_model_type")]
  model_type: String,
  /// Transformer hidden / embedding dimension (`768`).
  #[serde(default = "default_vision_hidden_size")]
  pub hidden_size: i32,
  /// Feed-forward intermediate dimension (`3072`).
  #[serde(default = "default_vision_intermediate_size")]
  pub intermediate_size: i32,
  /// Number of transformer encoder layers (`12`).
  #[serde(default = "default_vision_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// Number of attention heads (`12`).
  #[serde(default = "default_vision_num_attention_heads")]
  pub num_attention_heads: i32,
  /// Input channel count (`3`, RGB).
  #[serde(default = "default_vision_num_channels")]
  pub num_channels: i32,
  /// Nominal square image size (`224`). For the native-resolution NaFlex path
  /// this is not the runtime resolution; retained for parity.
  #[serde(default = "default_vision_image_size")]
  pub image_size: i32,
  /// Patch side length in pixels (`16`). The flattened-patch stride
  /// `num_channels * patch_size^2` is the patch-embed Linear's input width.
  #[serde(default = "default_vision_patch_size")]
  pub patch_size: i32,
  /// Learned position-embedding count (`256` ⇒ a `16 x 16` grid resized per
  /// image by the bicubic interpolation).
  #[serde(default = "default_vision_num_patches")]
  pub num_patches: i32,
  /// `eps` shared by every `LayerNorm` (`1e-6`).
  #[serde(default = "default_vision_layer_norm_eps")]
  pub layer_norm_eps: f64,
}

#[cfg(feature = "lfm2-vl")]
fn default_vision_model_type() -> String {
  "lfm2_vl".to_string()
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_hidden_size() -> i32 {
  768
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_intermediate_size() -> i32 {
  3072
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_num_hidden_layers() -> i32 {
  12
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_num_attention_heads() -> i32 {
  12
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_num_channels() -> i32 {
  3
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_image_size() -> i32 {
  224
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_patch_size() -> i32 {
  16
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_num_patches() -> i32 {
  256
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_layer_norm_eps() -> f64 {
  1e-6
}

/// The two architecture ids `vision.py`'s `VisionModel` accepts
/// (`["lfm2_vl", "siglip2_vision_model"]`).
#[cfg(feature = "lfm2-vl")]
const VISION_MODEL_TYPES: &[&str] = &["lfm2_vl", "siglip2_vision_model"];

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl VisionConfig {
  /// Parse a [`VisionConfig`] from an in-memory JSON string (the
  /// `vision_config` sub-object of an LFM2.5-VL `config.json`).
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "lfm2_vl::VisionConfig::from_json",
        "vision config JSON",
        e,
      ))
    })
  }

  /// Architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// The flattened per-patch feature width: `num_channels * patch_size^2`
  /// (`3 * 16^2 = 768` for the base checkpoint) — the width of each row the
  /// processor emits and the patch-embed Linear consumes.
  ///
  /// Overflow-checked (`patch_size^2` then `* num_channels`) so a hostile
  /// `patch_size` cannot wrap; non-positive operands are rejected.
  pub fn patch_feature_dim(&self) -> Result<i32> {
    require_positive("lfm2_vl::VisionConfig: patch_size", self.patch_size)?;
    require_positive("lfm2_vl::VisionConfig: num_channels", self.num_channels)?;
    let p2 = checked_mul(
      "lfm2_vl::VisionConfig: patch_size^2",
      "patch_size",
      self.patch_size,
      "patch_size",
      self.patch_size,
    )?;
    checked_mul(
      "lfm2_vl::VisionConfig: num_channels * patch_size^2",
      "num_channels",
      self.num_channels,
      "patch_size^2",
      p2,
    )
  }

  /// Reject a structurally invalid vision config with a typed error before any
  /// tensor is built.
  ///
  /// Pins `model_type` to one of `vision.py`'s accepted ids
  /// (`"lfm2_vl"` / `"siglip2_vision_model"`); requires every dimension / count
  /// (`hidden_size`, `intermediate_size`, the layer + head counts,
  /// `num_channels`, `image_size`, `patch_size`, `num_patches`) strictly
  /// positive; requires `hidden_size` divisible by `num_attention_heads` (the
  /// per-head split); requires `num_patches` a perfect square (the trained
  /// position grid is `sqrt(num_patches) x sqrt(num_patches)`); and validates
  /// that the derived `patch_feature_dim` (`num_channels * patch_size^2`)
  /// arithmetic does not overflow (a wrapped width would be UB downstream).
  pub fn validate(&self) -> Result<()> {
    crate::model_validation::pin_str(
      "lfm2_vl::VisionConfig: model_type",
      self.model_type.as_str(),
      VISION_MODEL_TYPES,
    )?;
    for (name, value) in [
      ("lfm2_vl::VisionConfig: hidden_size", self.hidden_size),
      (
        "lfm2_vl::VisionConfig: intermediate_size",
        self.intermediate_size,
      ),
      (
        "lfm2_vl::VisionConfig: num_attention_heads",
        self.num_attention_heads,
      ),
      (
        "lfm2_vl::VisionConfig: num_hidden_layers",
        self.num_hidden_layers,
      ),
      ("lfm2_vl::VisionConfig: num_channels", self.num_channels),
      ("lfm2_vl::VisionConfig: image_size", self.image_size),
      ("lfm2_vl::VisionConfig: patch_size", self.patch_size),
      ("lfm2_vl::VisionConfig: num_patches", self.num_patches),
    ] {
      require_positive(name, value)?;
    }
    require_divisible(
      "lfm2_vl::VisionConfig: hidden_size",
      self.hidden_size,
      "num_attention_heads",
      self.num_attention_heads,
    )?;
    // The position-embedding table is a square grid resized per image, so
    // `num_patches` must be a perfect square (`sqrt(num_patches)` per side).
    require_perfect_square("lfm2_vl::VisionConfig: num_patches", self.num_patches)?;
    // Validate the flattened-patch-width arithmetic does not overflow.
    self.patch_feature_dim()?;
    Ok(())
  }
}

// ═══════════════════════════════ ModelConfig ═══════════════════════════════

/// Top-level LFM2.5-VL model configuration — `config.py`'s `ModelConfig`: the
/// two tower configs plus the projector / image-token / patch-merge
/// parameters. Defaults match `LiquidAI/LFM2.5-VL-450M-MLX-8bit`.
///
/// The `quantization` block (`{group_size, bits, mode}`, `bits = 8` for the
/// 8-bit checkpoint) is carried opaquely as a [`serde_json::Value`] and
/// resolved to a [`PerLayerQuantization`](crate::lm::quant::PerLayerQuantization)
/// by [`crate::lm::models::lfm2::resolve_quantization`] (which also accepts the
/// HuggingFace `quantization_config` key) at load time — the same path the LFM2
/// LM and the other quantized ports use.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
  /// Text-tower config (`text_config`) — the LFM2 LM [`TextConfig`].
  pub text_config: TextConfig,
  /// Vision-tower config (`vision_config`).
  pub vision_config: VisionConfig,
  /// Top-level architecture id (`"lfm2-vl"`).
  #[serde(default = "default_model_type")]
  model_type: String,
  /// Pixel-unshuffle downsample factor applied to the vision grid before the
  /// projector (`2` ⇒ the projector input is `hidden * factor^2 = 3072` wide).
  #[serde(default = "default_downsample_factor")]
  pub downsample_factor: i32,
  /// The `<image>` placeholder token id spliced with image features (`396`).
  #[serde(default = "default_image_token_index")]
  pub image_token_index: i32,
  /// Projector hidden width (`2560`): `Linear(hidden*factor^2 -> 2560) -> gelu
  /// -> Linear(2560 -> text hidden)`.
  #[serde(default = "default_projector_hidden_size")]
  pub projector_hidden_size: i32,
  /// Whether the projector applies a `LayerNorm` on its input (`true`).
  #[serde(default = "default_true")]
  pub projector_use_layernorm: bool,
  /// Whether the projector `Linear`s carry a bias (`true`).
  #[serde(default = "default_true")]
  pub projector_bias: bool,
  /// Which vision encoder layer's hidden state feeds the projector (`-1` ⇒ the
  /// last layer; the encoder is truncated to `vision_feature_layer + 1`).
  #[serde(default = "default_vision_feature_layer")]
  pub vision_feature_layer: i32,
  /// Maximum per-image patch budget for the native-resolution processor
  /// (`1024`).
  #[serde(default = "default_max_num_patches")]
  pub max_num_patches: i32,
  /// Image-splitting tile size in pixels (`512`).
  #[serde(default = "default_tile_size")]
  pub tile_size: i32,
  /// End-of-sequence token id (`7`).
  #[serde(default = "default_eos_token_id")]
  pub eos_token_id: i32,
  /// The raw `quantization` block (`{group_size, bits, mode}`), carried
  /// opaquely and resolved by
  /// [`crate::lm::models::lfm2::resolve_quantization`]. Absent ⇒ a dense
  /// checkpoint.
  #[serde(default)]
  quantization: Option<serde_json::Value>,
}

#[cfg(feature = "lfm2-vl")]
fn default_model_type() -> String {
  "lfm2-vl".to_string()
}
#[cfg(feature = "lfm2-vl")]
fn default_downsample_factor() -> i32 {
  2
}
#[cfg(feature = "lfm2-vl")]
fn default_image_token_index() -> i32 {
  396
}
#[cfg(feature = "lfm2-vl")]
fn default_projector_hidden_size() -> i32 {
  2560
}
#[cfg(feature = "lfm2-vl")]
fn default_true() -> bool {
  true
}
#[cfg(feature = "lfm2-vl")]
fn default_vision_feature_layer() -> i32 {
  -1
}
#[cfg(feature = "lfm2-vl")]
fn default_max_num_patches() -> i32 {
  1024
}
#[cfg(feature = "lfm2-vl")]
fn default_tile_size() -> i32 {
  512
}
#[cfg(feature = "lfm2-vl")]
fn default_eos_token_id() -> i32 {
  7
}

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl ModelConfig {
  /// Parse a [`ModelConfig`] from an in-memory `config.json` string. A
  /// malformed-JSON failure maps to [`Error::Parse`]; absent keys take their
  /// checkpoint defaults; unmodeled keys are ignored.
  ///
  /// The nested `text_config` is the LFM2 LM [`TextConfig`], whose hand-written
  /// `Deserialize` applies `__post_init__`'s RoPE-base precedence
  /// (`lfm2.py:40-42`) intrinsically — so it runs here too, during the
  /// `ModelConfig` derive's deserialization of `text_config`, and on a direct
  /// `serde_json::from_str::<ModelConfig>` alike, with no separate step.
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "lfm2_vl::ModelConfig::from_json",
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

  /// The raw `quantization` block, if present (resolved to scheme parameters by
  /// [`crate::lm::models::lfm2::resolve_quantization`] at load time).
  #[inline(always)]
  pub fn quantization(&self) -> Option<&serde_json::Value> {
    self.quantization.as_ref()
  }

  /// The resolved feature-layer count `vision_feature_layer + 1` — how many
  /// encoder layers the vision tower keeps (`vision.py`'s
  /// `encoder.layers[: feature_layer + 1]`). `-1` keeps all
  /// `num_hidden_layers`; any other value keeps `feature_layer + 1`.
  ///
  /// Returns the kept-layer count clamped to `[1, num_hidden_layers]` after
  /// resolving the Python negative-index convention; an out-of-range
  /// `vision_feature_layer` (its resolved count `< 1` or `> num_hidden_layers`)
  /// is a typed [`Error::OutOfRange`].
  pub fn vision_feature_layers_kept(&self) -> Result<i32> {
    let total = self.vision_config.num_hidden_layers;
    // `-1` ⇒ keep all layers (the common case). Otherwise the count is
    // `vision_feature_layer + 1`, which must land in `[1, total]`.
    let kept = if self.vision_feature_layer == -1 {
      total
    } else {
      self.vision_feature_layer.saturating_add(1)
    };
    if kept < 1 || kept > total {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "lfm2_vl::ModelConfig: vision_feature_layer",
        "resolved kept-layer count must be in [1, vision num_hidden_layers]",
        smol_str::format_smolstr!(
          "vision_feature_layer={}, num_hidden_layers={total}, kept={kept}",
          self.vision_feature_layer
        ),
      )));
    }
    Ok(kept)
  }

  /// Reject a structurally invalid model config with a typed error before any
  /// tensor is built.
  ///
  /// Pins the top-level `model_type` to `"lfm2-vl"`; requires the projector /
  /// patch-merge dimensions (`downsample_factor`, `projector_hidden_size`,
  /// `max_num_patches`, `tile_size`) strictly positive and `image_token_index`
  /// / `eos_token_id` non-negative; validates that `vision_feature_layer`
  /// resolves to an in-range kept-layer count; and validates both tower configs
  /// (see [`TextConfig::validate`] / [`VisionConfig::validate`]).
  pub fn validate(&self) -> Result<()> {
    crate::model_validation::pin_str(
      "lfm2_vl::ModelConfig: model_type",
      self.model_type.as_str(),
      &["lfm2-vl"],
    )?;
    for (name, value) in [
      (
        "lfm2_vl::ModelConfig: downsample_factor",
        self.downsample_factor,
      ),
      (
        "lfm2_vl::ModelConfig: projector_hidden_size",
        self.projector_hidden_size,
      ),
      (
        "lfm2_vl::ModelConfig: max_num_patches",
        self.max_num_patches,
      ),
      ("lfm2_vl::ModelConfig: tile_size", self.tile_size),
    ] {
      require_positive(name, value)?;
    }
    // Token ids index a vocabulary / placeholder set; a negative id is
    // structurally invalid (it would never match a real token).
    for (name, value) in [
      (
        "lfm2_vl::ModelConfig: image_token_index",
        self.image_token_index,
      ),
      ("lfm2_vl::ModelConfig: eos_token_id", self.eos_token_id),
    ] {
      if value < 0 {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          name,
          "must be a non-negative token id (>= 0)",
          smol_str::format_smolstr!("{value}"),
        )));
      }
    }
    self.text_config.validate()?;
    self.vision_config.validate()?;
    // Resolve + range-check the feature-layer selection against the validated
    // vision config (its `num_hidden_layers` is now known positive).
    self.vision_feature_layers_kept()?;
    Ok(())
  }
}

/// Reject a value that is not a positive perfect square. Used for
/// `num_patches` (the square trained position grid).
#[cfg(feature = "lfm2-vl")]
fn require_perfect_square(field: &'static str, value: i32) -> Result<()> {
  require_positive(field, value)?;
  let r = (value as f64).sqrt().round() as i32;
  if r.saturating_mul(r) != value {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      field,
      "must be a perfect square (the trained position grid is square)",
      smol_str::format_smolstr!("{value}"),
    )));
  }
  Ok(())
}

#[cfg(all(test, feature = "lfm2-vl"))]
mod tests;
