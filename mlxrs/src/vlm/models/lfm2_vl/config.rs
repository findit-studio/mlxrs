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
  model_validation::{
    checked_mul, pin_i32, require_cardinality, require_divisible, require_in_range,
    require_positive,
  },
};

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use crate::lm::models::lfm2::TextConfig;

/// Inclusive upper bound on every *width*-like vision / projector config field —
/// a matmul axis or embedding-table column (`hidden_size`, `intermediate_size`,
/// `image_size`, `patch_size`, `num_patches`, `projector_hidden_size`). A width
/// sizes a layer's parameter tensor, so a hostile value drives an oversized
/// allocation; `1 << 20` (`1048576`) bounds every width — the real SigLIP2 tower
/// widths are a few thousand and the position grid `256`, all far below — while
/// keeping a malformed width a recoverable [`Error::OutOfRange`]. Mirrors the
/// EmbeddingGemma / LFM2-LM config width-cap (`MAX_CONFIG_DIM`) discipline; the
/// LFM2 text tower carries its own (`2^24`) cap, applied by
/// [`TextConfig::validate`].
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub const MAX_CONFIG_DIM: i32 = 1 << 20;

/// Inclusive upper bound on every *cardinality*-like vision config field — a
/// count that sizes an eager per-unit `Vec` or loop (`num_hidden_layers`,
/// `num_attention_heads`, `num_channels`). A SigLIP2 tower has 12-27 layers and
/// 12-16 heads; `4096` is far above any legitimate count while still bounding
/// the per-layer build loop a hostile `num_hidden_layers` could drive. Matches
/// the EmbeddingGemma / SigLIP2 / LFM2-LM config `MAX_CARDINALITY` intent.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub const MAX_CARDINALITY: i32 = 4096;

/// Inclusive upper bound on the per-image patch budget (`max_num_patches`, and
/// the HF tile-grid token budgets `min_image_tokens` / `max_image_tokens`,
/// `encoder_patch_size`, `tile_size`, `downsample_factor`). `max_num_patches`
/// is the **leading dimension of the `pixel_values` allocation**
/// (`max_num_patches x num_channels * patch_size^2`) the native + tiled
/// patchify paths zero-fill ([`crate::vlm::models::lfm2_vl::processor`]), so an
/// unbounded value from a malformed checkpoint would drive an oversized
/// allocation per image regardless of the [`MAX_TILES`] tile-count cap.
///
/// HuggingFace `image_processing_lfm2_vl.py` imposes no hard ceiling on
/// `max_num_patches` (the SigLIP2 NaFlex default is `256`; the released
/// `LiquidAI/LFM2.5-VL` MLX checkpoints ship `1024`), so the bound is generous:
/// `1 << 16` (`65536`) is ~64x the `1024` default — it never rejects a sane
/// checkpoint — while bounding a single image's `pixel_values` leading dimension
/// to a sane size (the derived total-element cap
/// [`crate::vlm::models::lfm2_vl::processor`]'s `MAX_PIXEL_VALUES_ELEMENTS`
/// bounds the full product `max_num_patches * patch_size^2 * channels`). The
/// tile-grid token budgets share this cap since they index the same patch-budget
/// space.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub const MAX_PATCH_BUDGET: i32 = 1 << 16;

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
  /// The encoder-MLP activation (`config.py:65`, default `"gelu_pytorch_tanh"`).
  /// `vision.py:67` builds the MLP as `nn.GELU(approx="precise")`, whose
  /// `__call__` dispatches `precise` (and its PyTorch alias `tanh`) to
  /// `mlx.nn.gelu_approx` — the **tanh** approximation
  /// (`mlx/python/mlx/nn/layers/activations.py:584-585`, `gelu_approx` at `:182`).
  /// The HuggingFace activation id for that tanh GELU is `gelu_pytorch_tanh`, so
  /// the config value and the impl agree. The MLP forward
  /// ([`crate::vlm::models::lfm2_vl::vision`]) hard-codes that tanh GELU, so
  /// [`validate`](Self::validate) pins this field to `"gelu_pytorch_tanh"` — a
  /// checkpoint declaring any other activation fails loudly rather than silently
  /// running the tanh GELU under a mismatched declared activation.
  #[serde(default = "default_vision_hidden_act")]
  pub hidden_act: String,
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
#[cfg(feature = "lfm2-vl")]
fn default_vision_hidden_act() -> String {
  "gelu_pytorch_tanh".to_string()
}

/// The two architecture ids `vision.py`'s `VisionModel` accepts
/// (`["lfm2_vl", "siglip2_vision_model"]`).
#[cfg(feature = "lfm2-vl")]
const VISION_MODEL_TYPES: &[&str] = &["lfm2_vl", "siglip2_vision_model"];

/// The sole vision-MLP activation the tower implements: the tanh GELU
/// (`mlx.nn.GELU(approx="precise")` → `gelu_approx`, `vision.py:67`), whose
/// HuggingFace id is `gelu_pytorch_tanh` (`config.py:65`). The MLP forward
/// hard-codes this activation, so `validate` pins `hidden_act` to it.
#[cfg(feature = "lfm2-vl")]
const VISION_HIDDEN_ACTS: &[&str] = &["gelu_pytorch_tanh"];

/// The top-level architecture ids the LFM2.5-VL [`ModelConfig`] accepts.
/// `config.py`'s default is `"lfm2-vl"` (hyphen, `config.py:75`), but the
/// released mlx-community checkpoints (e.g. `mlx-community/LFM2.5-VL-450M-6bit` /
/// `-8bit`) ship `model_type: "lfm2_vl"` (underscore). Both are accepted so a
/// checkpoint with either spelling loads (the `VisionConfig` already accepts the
/// underscore via [`VISION_MODEL_TYPES`]).
#[cfg(feature = "lfm2-vl")]
const MODEL_TYPES: &[&str] = &["lfm2-vl", "lfm2_vl"];

/// The sole input channel count the full LFM2.5-VL model + its image processor
/// support: `3` (RGB). The processor is RGB-hard-wired —
/// [`Lfm2VlProcessorConfig::new`](crate::vlm::models::lfm2_vl::processor::Lfm2VlProcessorConfig::new)
/// sets `num_channels = RGB_CHANNELS` (`3`) and
/// [`preprocess_image`](crate::vlm::models::lfm2_vl::processor::preprocess_image)
/// rejects any processor config whose `num_channels != 3` (the patchify builds a
/// 3-channel `RgbImage`, resizes into an always-3-channel buffer, and uses the
/// channel count as the patchify stride) — and the patch-embed `Linear`'s input
/// width derives from `num_channels * patch_size^2`. So [`ModelConfig::validate`]
/// pins `vision_config.num_channels` to `3`: a checkpoint declaring a different
/// channel count is a malformed / wrong-architecture config that would otherwise
/// either run a mismatched architecture or fail late at a vision matmul shape
/// check, and is rejected at load instead.
#[cfg(feature = "lfm2-vl")]
const RGB_CHANNELS: i32 = 3;

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
  /// (`"lfm2_vl"` / `"siglip2_vision_model"`); pins `hidden_act` to the tanh
  /// GELU the MLP implements (`"gelu_pytorch_tanh"`, `vision.py:67` /
  /// `config.py:65`); bounds every width-like field (`hidden_size`,
  /// `intermediate_size`, `image_size`, `patch_size`, `num_patches`) by
  /// [`MAX_CONFIG_DIM`] and every cardinality-like field (the layer + head
  /// counts, `num_channels`) by [`MAX_CARDINALITY`] — each rejecting both a
  /// non-positive and an oversized value, so a hostile dimension cannot drive an
  /// oversized parameter / position-table / per-layer allocation; requires
  /// `hidden_size` divisible by `num_attention_heads` (the per-head split);
  /// requires `num_patches` a perfect square (the trained position grid is
  /// `sqrt(num_patches) x sqrt(num_patches)`); and validates that the derived
  /// `patch_feature_dim` (`num_channels * patch_size^2`) arithmetic does not
  /// overflow (a wrapped width would be UB downstream).
  pub fn validate(&self) -> Result<()> {
    crate::model_validation::pin_str(
      "lfm2_vl::VisionConfig: model_type",
      self.model_type.as_str(),
      VISION_MODEL_TYPES,
    )?;
    // The vision MLP forward (`vision.rs`) hard-codes the tanh GELU
    // (`nn.GELU(approx="precise")` → `gelu_approx`); pin the architecture-defining
    // activation so a checkpoint declaring a different `hidden_act` fails loudly
    // rather than silently running the tanh GELU under a mismatched declaration.
    crate::model_validation::pin_str(
      "lfm2_vl::VisionConfig: hidden_act",
      self.hidden_act.as_str(),
      VISION_HIDDEN_ACTS,
    )?;
    // Width-like fields name a matmul axis / embedding-table column (the
    // patch-embed Linear width derives from `num_channels * patch_size^2`, the
    // position table has `num_patches` rows). `require_in_range(_, 1,
    // MAX_CONFIG_DIM)` rejects both a non-positive and an oversized value as one
    // [`Error::OutOfRange`], so a hostile width cannot drive an oversized
    // parameter / position-table allocation. Mirrors the EmbeddingGemma width-cap.
    for (name, value) in [
      ("lfm2_vl::VisionConfig: hidden_size", self.hidden_size),
      (
        "lfm2_vl::VisionConfig: intermediate_size",
        self.intermediate_size,
      ),
      ("lfm2_vl::VisionConfig: image_size", self.image_size),
      ("lfm2_vl::VisionConfig: patch_size", self.patch_size),
      ("lfm2_vl::VisionConfig: num_patches", self.num_patches),
    ] {
      require_in_range(name, value, 1, MAX_CONFIG_DIM)?;
    }
    // Cardinality-like fields size an eager per-unit `Vec` / loop (the encoder
    // builds a `Vec` of `num_hidden_layers` layers; the per-head split iterates
    // `num_attention_heads`; the patchify channel loop runs `num_channels`
    // times), so each takes the much tighter [`MAX_CARDINALITY`] cap: a
    // non-positive count is [`Error::OutOfRange`], an over-cap one
    // [`Error::CapExceeded`].
    for (name, value) in [
      (
        "lfm2_vl::VisionConfig: num_attention_heads",
        self.num_attention_heads,
      ),
      (
        "lfm2_vl::VisionConfig: num_hidden_layers",
        self.num_hidden_layers,
      ),
      ("lfm2_vl::VisionConfig: num_channels", self.num_channels),
    ] {
      require_cardinality(name, i64::from(value), MAX_CARDINALITY as u64)?;
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
/// `Deserialize` is **hand-written** (via the private [`RawModelConfig`] mirror)
/// rather than derived so the top-level `eos_token_id`'s null-coalescing and its
/// fallback to `text_config.eos_token_id` are applied intrinsically on **every**
/// deserialization path — a direct `serde_json::from_str::<ModelConfig>`, the
/// `text_config`-nested path, and [`from_json`](Self::from_json) alike — exactly
/// as the LFM2 LM [`TextConfig`](crate::lm::models::lfm2::TextConfig) hand-writes
/// its `Deserialize` to apply `__post_init__` intrinsically. The released
/// `LFM2-VL` `config.json` carries a top-level `eos_token_id: null` with the real
/// value (`7`) nested under `text_config`; a derived `Deserialize` over a bare
/// `i32` would reject the present `null` (and could not see the nested value).
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug, Clone)]
pub struct ModelConfig {
  /// Text-tower config (`text_config`) — the LFM2 LM [`TextConfig`].
  pub text_config: TextConfig,
  /// Vision-tower config (`vision_config`).
  pub vision_config: VisionConfig,
  /// Top-level architecture id. `config.py`'s default is `"lfm2-vl"`, but the
  /// released mlx-community checkpoints ship `"lfm2_vl"` (underscore); both are
  /// accepted by [`validate`](ModelConfig::validate).
  model_type: String,
  /// Pixel-unshuffle downsample factor applied to the vision grid before the
  /// projector (`2` ⇒ the projector input is `hidden * factor^2 = 3072` wide).
  pub downsample_factor: i32,
  /// The `<image>` placeholder token id spliced with image features (`396`).
  pub image_token_index: i32,
  /// Projector hidden width (`2560`): `Linear(hidden*factor^2 -> 2560) -> gelu
  /// -> Linear(2560 -> text hidden)`.
  pub projector_hidden_size: i32,
  /// Whether the projector applies a `LayerNorm` on its input (`true`).
  pub projector_use_layernorm: bool,
  /// Whether the projector `Linear`s carry a bias (`true`).
  pub projector_bias: bool,
  /// Which vision encoder layer's hidden state feeds the projector (`-1` ⇒ the
  /// last layer; the encoder is truncated to `vision_feature_layer + 1`).
  pub vision_feature_layer: i32,
  /// Maximum per-image patch budget for the native-resolution processor
  /// (`1024`).
  pub max_num_patches: i32,
  /// Whether the HuggingFace fast image processor splits an over-budget image
  /// into tiles (`config.py:76`, default `true`). Carried for config parity; the
  /// mlx-vlm processor path this port mirrors deliberately runs with splitting
  /// **disabled** (the slow `Siglip2ImageProcessor` native-resolution path —
  /// `processing_lfm2_vl.py:129-132, 195-196, 270-273, 372-373`), so this flag is
  /// not consumed by [`crate::vlm::models::lfm2_vl::processor`] today. See the
  /// module-level note on the tiling deferral.
  pub do_image_splitting: bool,
  /// Encoder patch size in pixels used by the tile-grid math of the HF fast image
  /// processor (`config.py:78`, default `16`). Carried for config parity (the
  /// native-resolution patch math uses the vision config's `patch_size`).
  pub encoder_patch_size: i32,
  /// Upper bound on the per-image `<image>`-token budget the HF fast processor
  /// targets when choosing a tile grid (`config.py:80`, default `256`).
  pub max_image_tokens: i32,
  /// Lower bound on the per-image `<image>`-token budget the HF fast processor
  /// targets when choosing a tile grid (`config.py:84`, default `64`).
  pub min_image_tokens: i32,
  /// Maximum number of tiles the HF fast processor may split an image into
  /// (`config.py:83`, default `10`).
  pub max_tiles: i32,
  /// Minimum number of tiles the HF fast processor splits an over-budget image
  /// into (`config.py:85`, default `2`).
  pub min_tiles: i32,
  /// Tolerance multiplier on the patch budget before the HF fast processor
  /// triggers a tile split (`config.py:82`, default `2.0`).
  pub max_pixels_tolerance: f32,
  /// Image-splitting tile size in pixels (`config.py:86`, default `512`).
  pub tile_size: i32,
  /// Whether the HF fast processor appends a downscaled thumbnail tile when
  /// splitting (`config.py:88`, default `false`). Carried for config parity; not
  /// consumed by the mlx-vlm native-resolution path this port mirrors.
  pub use_thumbnail: bool,
  /// Whether the prompt brackets each image with the `image_start` / `image_end`
  /// special tokens around the expanded `<image>` run (`config.py:87`, default
  /// `true`). The actual bracketing is driven by the processor's resolved token
  /// ids (see [`crate::vlm::models::lfm2_vl::processor`]); this carries the config
  /// flag for parity.
  pub use_image_special_tokens: bool,
  /// The projector activation id (`config.py:91`, default `"gelu"`). Carried for
  /// config parity; the projector forward hard-codes the GELU the reference uses
  /// (`lfm2_vl.py:36`).
  pub projector_hidden_act: String,
  /// End-of-sequence token id, RESOLVED at deserialization with the precedence
  /// documented on [`eos_token_id`](ModelConfig::eos_token_id) — a present,
  /// non-null top-level `eos_token_id` wins; otherwise the nested
  /// `text_config.eos_token_id` (the canonical home, `7` in the released
  /// checkpoint); `None` only when neither is present (then the accessor falls
  /// back to [`DEFAULT_EOS_TOKEN_ID`]).
  ///
  /// `Option<i32>` (not a defaulted `i32`) because the canonical
  /// `LFM2-VL-450M` `config.json` carries a **top-level `eos_token_id: null`**
  /// (the real value lives under `text_config`) — `#[serde(default = …)]` on a
  /// bare `i32` fills an *absent* key but REJECTS a present `null`
  /// (`invalid type: null, expected i32`), which blocked loading the canonical
  /// checkpoint. The null-coalescing + nested resolution lives in this struct's
  /// hand-written [`Deserialize`](ModelConfig#impl-Deserialize), so it applies on
  /// every deserialization path. Read it through
  /// [`eos_token_id`](ModelConfig::eos_token_id).
  eos_token_id: Option<i32>,
  /// The raw `quantization` block (`{group_size, bits, mode}`), carried
  /// opaquely and resolved by
  /// [`crate::lm::models::lfm2::resolve_quantization`]. Absent ⇒ a dense
  /// checkpoint.
  quantization: Option<serde_json::Value>,
}

/// Private `#[derive(Deserialize)]` mirror of [`ModelConfig`] — one field per
/// `ModelConfig` field, carrying that field's serde defaults **verbatim**, so the
/// hand-written [`ModelConfig`] `Deserialize` parses the raw `config.json`
/// exactly as a derived impl would, then resolves the top-level-vs-nested
/// `eos_token_id` before handing back the public struct. Keeping the defaults
/// identical to [`ModelConfig`]'s former derive is load-bearing: any divergence
/// in a default would change checkpoint parsing.
///
/// Two fields differ from a plain mirror to drive the eos resolution:
///
/// - `text_config` is captured as a raw [`serde_json::Value`] (not a
///   [`TextConfig`]) so the nested `eos_token_id` — which [`TextConfig`]
///   deliberately does not model — survives the first parse; the public
///   [`TextConfig`] is then produced via `serde_json::from_value`, which still
///   runs [`TextConfig`]'s own hand-written `Deserialize` (and thus its
///   `__post_init__` RoPE override) unchanged.
/// - `eos_token_id` is `Option<i32>` so a top-level `eos_token_id: null` parses
///   to `None` (the bug: a defaulted bare `i32` rejects a present `null`); an
///   absent key is also `None` (via `#[serde(default)]`). The resolution against
///   the nested value happens in the [`ModelConfig`] `Deserialize`.
#[cfg(feature = "lfm2-vl")]
#[derive(serde::Deserialize)]
struct RawModelConfig {
  text_config: serde_json::Value,
  vision_config: VisionConfig,
  #[serde(default = "default_model_type")]
  model_type: String,
  #[serde(default = "default_downsample_factor")]
  downsample_factor: i32,
  #[serde(default = "default_image_token_index")]
  image_token_index: i32,
  #[serde(default = "default_projector_hidden_size")]
  projector_hidden_size: i32,
  #[serde(default = "default_true")]
  projector_use_layernorm: bool,
  #[serde(default = "default_true")]
  projector_bias: bool,
  #[serde(default = "default_vision_feature_layer")]
  vision_feature_layer: i32,
  #[serde(default = "default_max_num_patches")]
  max_num_patches: i32,
  #[serde(default = "default_true")]
  do_image_splitting: bool,
  #[serde(default = "default_encoder_patch_size")]
  encoder_patch_size: i32,
  #[serde(default = "default_max_image_tokens")]
  max_image_tokens: i32,
  #[serde(default = "default_min_image_tokens")]
  min_image_tokens: i32,
  #[serde(default = "default_max_tiles")]
  max_tiles: i32,
  #[serde(default = "default_min_tiles")]
  min_tiles: i32,
  #[serde(default = "default_max_pixels_tolerance")]
  max_pixels_tolerance: f32,
  #[serde(default = "default_tile_size")]
  tile_size: i32,
  #[serde(default)]
  use_thumbnail: bool,
  #[serde(default = "default_true")]
  use_image_special_tokens: bool,
  #[serde(default = "default_projector_hidden_act")]
  projector_hidden_act: String,
  #[serde(default)]
  eos_token_id: Option<i32>,
  #[serde(default)]
  quantization: Option<serde_json::Value>,
}

#[cfg(feature = "lfm2-vl")]
impl<'de> serde::Deserialize<'de> for ModelConfig {
  /// Deserialize a [`ModelConfig`] via the private [`RawModelConfig`] mirror,
  /// then resolve the `eos_token_id` with the precedence documented on
  /// [`eos_token_id`](Self::eos_token_id) — so EVERY path that materializes a
  /// `ModelConfig` (a direct `serde_json::from_str::<ModelConfig>`, the nested
  /// path, and [`from_json`](Self::from_json)) coalesces a top-level `null` and
  /// falls back to the nested `text_config.eos_token_id` identically and cannot
  /// bypass it.
  fn deserialize<D: serde::Deserializer<'de>>(
    deserializer: D,
  ) -> std::result::Result<Self, D::Error> {
    use serde::de::Error as _;
    let raw = RawModelConfig::deserialize(deserializer)?;
    // Pull the nested `text_config.eos_token_id` (the canonical home; `7` in the
    // released checkpoint) from the raw text-config object BEFORE converting it
    // to the typed `TextConfig`, which does not model token ids. A non-integer /
    // absent value yields `None` so the fallback chain continues.
    let nested_eos = raw
      .text_config
      .get("eos_token_id")
      .and_then(serde_json::Value::as_i64)
      .and_then(|v| i32::try_from(v).ok());
    // Convert the captured raw text-config to the typed `TextConfig`. This runs
    // `TextConfig`'s own hand-written `Deserialize` (and its `__post_init__` RoPE
    // override) exactly as the former nested-derive path did.
    let text_config: TextConfig =
      serde_json::from_value(raw.text_config).map_err(D::Error::custom)?;
    // Precedence: a present, non-null top-level `eos_token_id` wins; otherwise the
    // nested `text_config.eos_token_id`. `None` only when NEITHER is present —
    // then [`eos_token_id`](Self::eos_token_id) backstops with
    // [`DEFAULT_EOS_TOKEN_ID`].
    let eos_token_id = raw.eos_token_id.or(nested_eos);
    Ok(ModelConfig {
      text_config,
      vision_config: raw.vision_config,
      model_type: raw.model_type,
      downsample_factor: raw.downsample_factor,
      image_token_index: raw.image_token_index,
      projector_hidden_size: raw.projector_hidden_size,
      projector_use_layernorm: raw.projector_use_layernorm,
      projector_bias: raw.projector_bias,
      vision_feature_layer: raw.vision_feature_layer,
      max_num_patches: raw.max_num_patches,
      do_image_splitting: raw.do_image_splitting,
      encoder_patch_size: raw.encoder_patch_size,
      max_image_tokens: raw.max_image_tokens,
      min_image_tokens: raw.min_image_tokens,
      max_tiles: raw.max_tiles,
      min_tiles: raw.min_tiles,
      max_pixels_tolerance: raw.max_pixels_tolerance,
      tile_size: raw.tile_size,
      use_thumbnail: raw.use_thumbnail,
      use_image_special_tokens: raw.use_image_special_tokens,
      projector_hidden_act: raw.projector_hidden_act,
      eos_token_id,
      quantization: raw.quantization,
    })
  }
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
/// Upper bound on `max_tiles` enforced at config load
/// ([`ModelConfig::validate`] /
/// [`Lfm2VlProcessorConfig::with_tiling`](crate::vlm::models::lfm2_vl::processor::Lfm2VlProcessorConfig::with_tiling)).
///
/// HuggingFace `image_processing_lfm2_vl.py` imposes no hard cap on `max_tiles`
/// (its default is `10`), but the tile-grid candidate builder
/// (`_target_ratios`) reserves and iterates a set bounded by `max_tiles^2`. An
/// unbounded `max_tiles` from a malformed checkpoint would drive a quadratic
/// reservation / loop before the pixel caps apply, so the cardinality is bound
/// here — the same load-time discipline the other tile-grid fields follow.
///
/// `1024` is ~100x the realistic default of `10`, so it never rejects a sane
/// checkpoint, while `max_tiles^2 = 2^20` keeps the candidate reservation
/// (`(u32, u32)` pairs) at ~8 MiB worst case and every downstream
/// `tile_size * grid` product within `u32` (the grid side is `<= max_tiles`).
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub const MAX_TILES: i32 = 1024;

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

/// The reference end-of-sequence token id for `LiquidAI/LFM2-VL-450M`
/// (`text_config.eos_token_id`, `7`). Used by
/// [`ModelConfig::eos_token_id`](ModelConfig::eos_token_id) as the last-resort
/// fallback when neither the top-level nor the nested `text_config.eos_token_id`
/// is present. The runtime tokenizer EOS set is resolved separately and more
/// completely by [`crate::vlm::load::load_vlm_base_config`] (top-level →
/// `text_config`/`llm_config` → `generation_config.json` override); this constant
/// only backstops the validate-only [`ModelConfig`] field.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub const DEFAULT_EOS_TOKEN_ID: i32 = 7;

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl ModelConfig {
  /// Parse a [`ModelConfig`] from an in-memory `config.json` string. A
  /// malformed-JSON failure maps to [`Error::Parse`]; absent keys take their
  /// checkpoint defaults; unmodeled keys are ignored.
  ///
  /// The nested `text_config` is the LFM2 LM [`TextConfig`], whose hand-written
  /// `Deserialize` applies `__post_init__`'s RoPE-base precedence
  /// (`lfm2.py:40-42`) intrinsically — so it runs here too, via the
  /// [`ModelConfig`] hand-written `Deserialize` (which converts the captured raw
  /// `text_config` through `TextConfig`'s own `Deserialize`), and on a direct
  /// `serde_json::from_str::<ModelConfig>` alike, with no separate step. The same
  /// hand-written `Deserialize` resolves [`eos_token_id`](Self::eos_token_id)
  /// (top-level → nested `text_config` → [`DEFAULT_EOS_TOKEN_ID`]), so a canonical
  /// checkpoint with a top-level `eos_token_id: null` loads cleanly.
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

  /// The resolved end-of-sequence token id.
  ///
  /// Resolution precedence (applied during deserialization, see the
  /// [`ModelConfig`] `Deserialize`): a present, non-null top-level
  /// `config.json` `eos_token_id` wins; otherwise the nested
  /// `text_config.eos_token_id` (the canonical home — `7` in the released
  /// `LFM2-VL-450M` checkpoint, whose top-level value is `null`); if NEITHER is
  /// present, [`DEFAULT_EOS_TOKEN_ID`].
  ///
  /// This backs the [`validate`](Self::validate) non-negative check and config
  /// parity. The *runtime* tokenizer EOS set (what actually stops generation) is
  /// resolved independently and more completely by
  /// [`crate::vlm::load::load_vlm_base_config`] (which also honors a
  /// `generation_config.json` override and a `llm_config` alias).
  #[inline(always)]
  pub const fn eos_token_id(&self) -> i32 {
    match self.eos_token_id {
      Some(id) => id,
      None => DEFAULT_EOS_TOKEN_ID,
    }
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
  /// (mlx-community checkpoints ship the underscore form); bounds the projector
  /// width fields (`downsample_factor`, `projector_hidden_size`) by
  /// [`MAX_CONFIG_DIM`]; bounds the patch-budget fields (`max_num_patches` — the
  /// leading dimension of the `pixel_values` allocation — `tile_size`,
  /// `encoder_patch_size`, `max_image_tokens`, `min_image_tokens`) by
  /// [`MAX_PATCH_BUDGET`] so a hostile budget cannot drive an oversized per-image
  /// allocation; requires `max_tiles` / `min_tiles` positive and `max_tiles`
  /// within the [`MAX_TILES`] cardinality cap (its `max_tiles^2` candidate
  /// reservation); requires `max_pixels_tolerance` positive-and-finite, the
  /// `min_* <= max_*` orderings, and `image_token_index` / `eos_token_id`
  /// non-negative; validates that `vision_feature_layer` resolves to an in-range
  /// kept-layer count; validates both tower configs (see
  /// [`TextConfig::validate`] / [`VisionConfig::validate`]); and pins
  /// `vision_config.num_channels` to `3` (the full model + processor are
  /// RGB-only).
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
    // Width-like fields name a matmul axis (the projector `Linear`s) / a fold
    // factor (`downsample_factor` squares into the projector input width). Bound
    // each by [`MAX_CONFIG_DIM`] — a non-positive or oversized value is one
    // [`Error::OutOfRange`].
    for (name, value) in [
      (
        "lfm2_vl::ModelConfig: downsample_factor",
        self.downsample_factor,
      ),
      (
        "lfm2_vl::ModelConfig: projector_hidden_size",
        self.projector_hidden_size,
      ),
    ] {
      require_in_range(name, value, 1, MAX_CONFIG_DIM)?;
    }
    // Patch-budget fields index the per-image patch space. `max_num_patches` is
    // the **leading dimension of the `pixel_values` allocation** the patchify
    // paths zero-fill, so an unbounded value drives an oversized per-image
    // allocation regardless of the [`MAX_TILES`] tile-count cap; the HF tile-grid
    // token budgets (`tile_size`, `encoder_patch_size`, `min`/`max_image_tokens`)
    // index the same space. Bound each by [`MAX_PATCH_BUDGET`] (the derived
    // total-element cap in `processor` bounds the full `pixel_values` product).
    for (name, value) in [
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
    ] {
      require_in_range(name, value, 1, MAX_PATCH_BUDGET)?;
    }
    // The tile counts are bounded just below: `max_tiles` by the [`MAX_TILES`]
    // cardinality cap (its `max_tiles^2` candidate reservation) and `min_tiles`
    // by the `min_tiles <= max_tiles` ordering. Require both positive here first.
    for (name, value) in [
      ("lfm2_vl::ModelConfig: max_tiles", self.max_tiles),
      ("lfm2_vl::ModelConfig: min_tiles", self.min_tiles),
    ] {
      require_positive(name, value)?;
    }
    // `max_tiles` is the cardinality of the tile-grid candidate set, whose
    // builder (`processor::target_ratios`) reserves and iterates `max_tiles^2`
    // pairs; bound it so a malformed checkpoint cannot drive a quadratic
    // reservation / loop. HF imposes no hard cap, so the bound is generous.
    if self.max_tiles > MAX_TILES {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "lfm2_vl::ModelConfig: max_tiles",
        "must not exceed the tile-grid cardinality cap (1024)",
        smol_str::format_smolstr!("{}", self.max_tiles),
      )));
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
    // structurally invalid (it would never match a real token). `eos_token_id`
    // is the RESOLVED value (top-level → nested `text_config` → default), so a
    // checkpoint whose only eos lives under `text_config` is validated against
    // that real value, not a placeholder.
    for (name, value) in [
      (
        "lfm2_vl::ModelConfig: image_token_index",
        self.image_token_index,
      ),
      ("lfm2_vl::ModelConfig: eos_token_id", self.eos_token_id()),
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
    // The full LFM2.5-VL model + its image processor are RGB-only: the processor
    // hard-wires `num_channels = RGB_CHANNELS` (`processor::Lfm2VlProcessorConfig::new`)
    // and `preprocess_image` rejects any config whose `num_channels != 3` (it
    // builds a 3-channel `RgbImage` and uses the channel count as the patchify
    // stride), while the patch-embed `Linear`'s input width derives from
    // `num_channels * patch_size^2`. `VisionConfig::validate` only bounds
    // `num_channels` as a cardinality, so pin it to `3` here, in the full-model
    // path, so a non-3 (wrong-architecture / malformed) checkpoint is a typed
    // [`Error::OutOfRange`] at load rather than a mismatched architecture or a
    // late vision-matmul shape failure.
    pin_i32(
      "lfm2_vl::ModelConfig: vision_config.num_channels",
      self.vision_config.num_channels,
      RGB_CHANNELS,
    )?;
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
