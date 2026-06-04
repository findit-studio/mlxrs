//! LFM2.5-VL — the LFM2 hybrid conv+attention LM + a SigLIP2-style vision
//! tower + a pixel-unshuffle projector (`mlx-vlm/mlx_vlm/models/lfm2_vl`).
//!
//! Faithful port of `LiquidAI/LFM2.5-VL-450M-MLX-8bit`. The text tower is the
//! existing LFM2 LM ([`crate::lm::models::lfm2`], whose
//! [`TextConfig`](crate::lm::models::lfm2::TextConfig) is re-exported as this
//! module's text config); the vision tower is a native-resolution SigLIP2 ViT
//! with a **Linear** patch embedding and a per-image bicubic-resized position
//! embedding. Every `nn.Linear` is routed through the shared quantize-aware
//! [`MaybeQuantizedLinear`](crate::nn::MaybeQuantizedLinear), so the 8-bit
//! checkpoint loads through the same code path as a dense one.
//!
//! ## Module map
//!
//! - [config structs](crate::vlm::models::lfm2_vl::config) — `TextConfig` (the
//!   re-exported LFM2 LM config) / `VisionConfig` / `ModelConfig`.
//! - [vision tower](crate::vlm::models::lfm2_vl::vision) — the native-resolution
//!   SigLIP2 ViT (Linear patch embed, per-image bicubic-resized position
//!   embedding), using the pure-MLX
//!   [`bicubic_interpolate`](crate::ops::interpolation::bicubic_interpolate).
//! - [pixel-unshuffle + multimodal projector + image-feature
//!   merge](crate::vlm::models::lfm2_vl::projector).
//! - [language adapter](crate::vlm::models::lfm2_vl::language) — the thin guarded
//!   wrapper that forwards merged embeddings through the LFM2 LM.
//! - [native-resolution processor](crate::vlm::models::lfm2_vl::processor) — the
//!   SigLIP2 NaFlex smart-resize + normalize + patchify + `<image>`-token
//!   expansion the checkpoint's image processor (`Siglip2ImageProcessor`)
//!   performs.
//! - [top-level VL model + factory](crate::vlm::models::lfm2_vl::model) — the
//!   [`Lfm2Vl`](crate::vlm::models::lfm2_vl::model::Lfm2Vl)
//!   [`crate::vlm::model::Model`] implementation (vision tower → projector →
//!   mask-driven splice into the LM embeddings → LFM2 LM → logits) plus the
//!   [`constructor`](crate::vlm::models::lfm2_vl::model::constructor) /
//!   [`register`](crate::vlm::models::lfm2_vl::model::register) hooks that plug
//!   it into the VLM [`crate::vlm::load`] factory on `model_type = "lfm2-vl"`.

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod config;
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod language;
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod model;
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod processor;
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod projector;
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod vision;

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use config::{ModelConfig, TextConfig, VisionConfig};
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use language::LanguageModel;
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use model::{Lfm2Vl, constructor, register};
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use processor::{
  Lfm2VlImageInputs, Lfm2VlProcessorConfig, TilePlan, expand_image_tokens,
  num_image_tokens_from_patch_grid, plan_tiles, preprocess_image, tile_image,
};
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use projector::{
  Lfm2VlMultiModalProjector, PixelUnshuffleBlock, merge_input_ids_with_image_features,
};
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use vision::VisionModel;
