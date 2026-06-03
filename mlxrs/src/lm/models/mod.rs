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

/// Qwen3 — the dense Qwen3 text transformer
/// (`mlx-lm/mlx_lm/models/qwen3.py`). Grouped-query attention with per-head
/// Q/K RMSNorm before RoPE, a SwiGLU MLP, and an optionally-tied LM head; the
/// language backbone for the Qwen3 forced-aligner. Gated on `qwen3`.
#[cfg(feature = "qwen3")]
#[cfg_attr(docsrs, doc(cfg(feature = "qwen3")))]
pub mod qwen3;
