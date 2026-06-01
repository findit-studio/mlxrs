//! Concrete decoder-model architectures (model-support phase).
//!
//! Each model is feature-gated to its own model feature.

/// LFM2 — the hybrid short-convolution + attention language model
/// (`mlx-lm/mlx_lm/models/lfm2.py`). The LM dependency of the LFM2.5-VL
/// vision-language model; gated on `lfm2-vl` (the single feature gating
/// both the LM and the later VL wrapper).
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod lfm2;
