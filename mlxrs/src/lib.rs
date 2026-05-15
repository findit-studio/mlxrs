//! mlxrs — safe Rust bindings for [MLX](https://github.com/ml-explore/mlx) on Apple silicon.
//!
//! M1 ships `Array` + `Dtype` + `Error` + a subset of `ops.h`. See
//! `docs/superpowers/specs/` for the full design.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![cfg_attr(not(test), deny(missing_docs))]

pub use version::version;

pub mod version;

/// Language Model (LM) — text-only inference. Stub in M1; LM port lands in M3.
#[cfg(feature = "lm")]
#[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
pub mod lm;

/// Vision-Language Model (VLM) — multimodal inference. Stub in M1; VLM port lands in M4.
#[cfg(feature = "vlm")]
#[cfg_attr(docsrs, doc(cfg(feature = "vlm")))]
pub mod vlm;

/// Audio (TTS/STT/STS) — speech inference. Stub in M1; audio port lands in M5.
#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio;
