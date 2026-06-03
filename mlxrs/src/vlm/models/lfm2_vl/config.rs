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
//!
//! ## Image-splitting / tiling config (carried, not consumed)
//!
//! [`ModelConfig`] mirrors `config.py`'s image-splitting knobs faithfully —
//! `do_image_splitting`, `encoder_patch_size`, `max_image_tokens`,
//! `min_image_tokens`, `max_tiles`, `min_tiles`, `max_pixels_tolerance`,
//! `tile_size`, `use_thumbnail` (`config.py:76-88`). These are **carried for
//! config parity** but are **not consumed** by the processor / forward pass:
//! mlx-vlm's own `processing_lfm2_vl.py` is a compatibility shim that defers to
//! the *slow* SigLIP2 native-resolution image processor and **deliberately
//! disables splitting** (`do_image_splitting = False` —
//! `processing_lfm2_vl.py:129-132, 195-196, 270-273`, with "no tiling support,
//! just add image tokens" at `processing_lfm2_vl.py:372-373`), and mlx-vlm's
//! `lfm2_vl.py` forward consumes only the SigLIP2 NaFlex triple (`pixel_values`,
//! `spatial_shapes`, `pixel_attention_mask` — `lfm2_vl.py:115-205`), never any
//! tile / thumbnail metadata. The actual tile-grid + split implementation lives
//! in HuggingFace `transformers` (`Lfm2VlImageProcessorFast`), which the mlx-vlm
//! reference bypasses (and which is outside the mlx reference tree). The mlxrs
//! [`crate::vlm::models::lfm2_vl::processor`] therefore mirrors the same
//! native-resolution (no-split) path. Carrying the fields keeps the config a 1:1
//! mirror of mlx-vlm `ModelConfig` and leaves the tiling path wired for a future
//! port from the upstream HF fast processor should it become a faithful target.

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

/// The top-level architecture ids the LFM2.5-VL [`ModelConfig`] accepts.
/// `config.py`'s default is `"lfm2-vl"` (hyphen, `config.py:75`), but the
/// released mlx-community checkpoints (e.g. `mlx-community/LFM2.5-VL-450M-6bit` /
/// `-8bit`) ship `model_type: "lfm2_vl"` (underscore). Both are accepted so a
/// checkpoint with either spelling loads (the `VisionConfig` already accepts the
/// underscore via [`VISION_MODEL_TYPES`]).
#[cfg(feature = "lfm2-vl")]
const MODEL_TYPES: &[&str] = &["lfm2-vl", "lfm2_vl"];

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
  /// Top-level architecture id. `config.py`'s default is `"lfm2-vl"`, but the
  /// released mlx-community checkpoints ship `"lfm2_vl"` (underscore); both are
  /// accepted by [`validate`](ModelConfig::validate).
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
  /// Whether the HuggingFace fast image processor splits an over-budget image
  /// into tiles (`config.py:76`, default `true`). Carried for config parity; the
  /// mlx-vlm processor path this port mirrors deliberately runs with splitting
  /// **disabled** (the slow `Siglip2ImageProcessor` native-resolution path —
  /// `processing_lfm2_vl.py:129-132, 195-196, 270-273, 372-373`), so this flag is
  /// not consumed by [`crate::vlm::models::lfm2_vl::processor`] today. See the
  /// module-level note on the tiling deferral.
  #[serde(default = "default_true")]
  pub do_image_splitting: bool,
  /// Encoder patch size in pixels used by the tile-grid math of the HF fast image
  /// processor (`config.py:78`, default `16`). Carried for config parity (the
  /// native-resolution patch math uses the vision config's `patch_size`).
  #[serde(default = "default_encoder_patch_size")]
  pub encoder_patch_size: i32,
  /// Upper bound on the per-image `<image>`-token budget the HF fast processor
  /// targets when choosing a tile grid (`config.py:80`, default `256`).
  #[serde(default = "default_max_image_tokens")]
  pub max_image_tokens: i32,
  /// Lower bound on the per-image `<image>`-token budget the HF fast processor
  /// targets when choosing a tile grid (`config.py:84`, default `64`).
  #[serde(default = "default_min_image_tokens")]
  pub min_image_tokens: i32,
  /// Maximum number of tiles the HF fast processor may split an image into
  /// (`config.py:83`, default `10`).
  #[serde(default = "default_max_tiles")]
  pub max_tiles: i32,
  /// Minimum number of tiles the HF fast processor splits an over-budget image
  /// into (`config.py:85`, default `2`).
  #[serde(default = "default_min_tiles")]
  pub min_tiles: i32,
  /// Tolerance multiplier on the patch budget before the HF fast processor
  /// triggers a tile split (`config.py:82`, default `2.0`).
  #[serde(default = "default_max_pixels_tolerance")]
  pub max_pixels_tolerance: f32,
  /// Image-splitting tile size in pixels (`config.py:86`, default `512`).
  #[serde(default = "default_tile_size")]
  pub tile_size: i32,
  /// Whether the HF fast processor appends a downscaled thumbnail tile when
  /// splitting (`config.py:88`, default `false`). Carried for config parity; not
  /// consumed by the mlx-vlm native-resolution path this port mirrors.
  #[serde(default)]
  pub use_thumbnail: bool,
  /// Whether the prompt brackets each image with the `image_start` / `image_end`
  /// special tokens around the expanded `<image>` run (`config.py:87`, default
  /// `true`). The actual bracketing is driven by the processor's resolved token
  /// ids (see [`crate::vlm::models::lfm2_vl::processor`]); this carries the config
  /// flag for parity.
  #[serde(default = "default_true")]
  pub use_image_special_tokens: bool,
  /// The projector activation id (`config.py:91`, default `"gelu"`). Carried for
  /// config parity; the projector forward hard-codes the GELU the reference uses
  /// (`lfm2_vl.py:36`).
  #[serde(default = "default_projector_hidden_act")]
  pub projector_hidden_act: String,
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
fn default_encoder_patch_size() -> i32 {
  16
}
#[cfg(feature = "lfm2-vl")]
fn default_max_image_tokens() -> i32 {
  256
}
#[cfg(feature = "lfm2-vl")]
fn default_min_image_tokens() -> i32 {
  64
}
#[cfg(feature = "lfm2-vl")]
fn default_max_tiles() -> i32 {
  10
}
#[cfg(feature = "lfm2-vl")]
fn default_min_tiles() -> i32 {
  2
}
#[cfg(feature = "lfm2-vl")]
fn default_max_pixels_tolerance() -> f32 {
  2.0
}
#[cfg(feature = "lfm2-vl")]
fn default_projector_hidden_act() -> String {
  "gelu".to_string()
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
  /// Pins the top-level `model_type` to `"lfm2-vl"` / `"lfm2_vl"`
  /// (mlx-community checkpoints ship the underscore form); requires
  /// the projector / patch-merge dimensions (`downsample_factor`,
  /// `projector_hidden_size`, `max_num_patches`, `tile_size`) and the tile-grid
  /// cardinality fields (`encoder_patch_size`, `max_image_tokens`,
  /// `min_image_tokens`, `max_tiles`, `min_tiles`) strictly positive,
  /// `max_pixels_tolerance` positive-and-finite, the `min_* <= max_*` orderings,
  /// and `image_token_index` / `eos_token_id` non-negative; validates that
  /// `vision_feature_layer` resolves to an in-range kept-layer count; and
  /// validates both tower configs (see [`TextConfig::validate`] /
  /// [`VisionConfig::validate`]).
  pub fn validate(&self) -> Result<()> {
    crate::model_validation::pin_str(
      "lfm2_vl::ModelConfig: model_type",
      self.model_type.as_str(),
      MODEL_TYPES,
    )?;
    // The projector forward hard-codes erf GELU (`projector.rs`); pin the
    // architecture-defining activation so a checkpoint declaring a different
    // value fails loudly rather than silently running GELU.
    crate::model_validation::pin_str(
      "lfm2_vl::ModelConfig: projector_hidden_act",
      self.projector_hidden_act.as_str(),
      &["gelu"],
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
      (
        "lfm2_vl::ModelConfig: encoder_patch_size",
        self.encoder_patch_size,
      ),
      (
        "lfm2_vl::ModelConfig: max_image_tokens",
        self.max_image_tokens,
      ),
      (
        "lfm2_vl::ModelConfig: min_image_tokens",
        self.min_image_tokens,
      ),
      ("lfm2_vl::ModelConfig: max_tiles", self.max_tiles),
      ("lfm2_vl::ModelConfig: min_tiles", self.min_tiles),
    ] {
      require_positive(name, value)?;
    }
    // The tile / token budgets are inclusive `[min, max]` bands; a `min` above
    // its `max` is structurally invalid (an empty band).
    for (name, min, max) in [
      (
        "lfm2_vl::ModelConfig: min_tiles <= max_tiles",
        self.min_tiles,
        self.max_tiles,
      ),
      (
        "lfm2_vl::ModelConfig: min_image_tokens <= max_image_tokens",
        self.min_image_tokens,
        self.max_image_tokens,
      ),
    ] {
      if min > max {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          name,
          "the minimum must not exceed the maximum",
          smol_str::format_smolstr!("min={min}, max={max}"),
        )));
      }
    }
    crate::model_validation::require_positive_finite_f32(
      "lfm2_vl::ModelConfig: max_pixels_tolerance",
      self.max_pixels_tolerance as f64,
    )?;
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
