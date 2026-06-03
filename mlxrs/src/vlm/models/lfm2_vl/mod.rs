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
//! ## Phase status
//!
//! This phase ports the
//! [config structs](crate::vlm::models::lfm2_vl::config), the
//! [vision tower](crate::vlm::models::lfm2_vl::vision), and the pure-MLX bicubic
//! interpolation primitive
//! ([`crate::ops::interpolation::bicubic_interpolate`]) the position-embed
//! resize uses. The pixel-unshuffle projector, the multimodal embed-splice, the
//! native-resolution processor, and the top-level
//! [`crate::vlm::model::Model`] implementation (wiring the LM + vision +
//! projector together and registering in the VLM factory) come in a later
//! phase.

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod config;
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod vision;

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use config::{ModelConfig, TextConfig, VisionConfig};
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub use vision::VisionModel;
