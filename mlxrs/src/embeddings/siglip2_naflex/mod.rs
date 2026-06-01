//! SigLIP2 NaFlex — a standalone dual-tower image+text **embeddings**
//! model (`google/siglip2-base-patch16-naflex`).
//!
//! Ported from
//! [`mlx-embeddings`'s `models/siglip.py`](https://github.com/Blaizzy/mlx-embeddings/blob/main/mlx_embeddings/models/siglip.py),
//! with native-resolution (NaFlex) position-embedding interpolation —
//! the piece `siglip.py` leaves as `NotImplementedError` — added here.
//! The NaFlex preprocessing + sizing is pinned against the user's
//! PyTorch-validated `siglip2-naflex` crate.
//!
//! This is a self-contained embeddings model under
//! [`crate::embeddings`]; it does **not** depend on the LFM2.5-VL vision
//! tower. The only shared low-level primitive is
//! [`crate::ops::interpolation::bicubic_interpolate`] (the per-image
//! position-embedding resize), which lives in [`crate::ops`] precisely so
//! it can be reused by an independent vision port without coupling the
//! model code.
//!
//! ## Foundation (this module)
//!
//! - [`config`] — the [`config::TextConfig`] / [`config::VisionConfig`] /
//!   [`config::Siglip2NaflexConfig`] dataclasses (serde parse +
//!   architecture-pinning validation on the shared
//!   [`crate::model_validation`] toolkit).
//! - [`processing`] — the NaFlex image preprocessing
//!   ([`processing::patch_grid`] sizing + aspect-preserving resize +
//!   normalize/patchify into the flat `(num_patches, P^2 * C)` tensor,
//!   plus `spatial_shapes` + `pixel_attention_mask`).
//!
//! The transformer towers (vision + text) and the load/factory wiring
//! are a follow-up.

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub mod config;

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub mod processing;
