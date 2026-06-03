//! Concrete vision-language-model architectures (model-support phase).
//!
//! Mirrors `mlx-vlm/mlx_vlm/models/`'s per-model split. Per the
//! model-support phase, named VLM architectures are added per-usecase here
//! (rather than bulk-ported); each is feature-gated to its own model feature.

/// LFM2.5-VL — the LFM2 hybrid conv+attention LM + a SigLIP2-style vision
/// tower + a pixel-unshuffle projector
/// (`mlx-vlm/mlx_vlm/models/lfm2_vl`). Gated on `lfm2-vl`.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub mod lfm2_vl;
