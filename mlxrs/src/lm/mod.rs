//! Language Model (LM) support — text-only inference building blocks ported
//! from [mlx-lm](https://github.com/ml-explore/mlx-lm).
//!
//! M3 lands the sampling utilities ([`crate::lm::sample`]), the
//! architecture-agnostic generation foundation — the
//! [`Model`](crate::lm::model::Model) trait, the KV
//! [`cache`](crate::lm::cache), the model-load support surface
//! ([`crate::lm::load`]), and the [`generate`](crate::lm::generate) loop
//! (`generate_step` / `stream_generate` / `generate` +
//! `make_sampler` / `make_logits_processors`).

pub mod cache;
pub mod generate;
pub mod load;
pub mod model;
pub mod sample;
