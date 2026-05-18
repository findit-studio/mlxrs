//! Language Model (LM) support — text-only inference building blocks ported
//! from [mlx-lm](https://github.com/ml-explore/mlx-lm).
//!
//! M3 lands the sampling utilities ([`crate::lm::sample`]), the
//! architecture-agnostic generation foundation — the
//! [`Model`](crate::lm::model::Model) trait and the KV
//! [`cache`](crate::lm::cache) — and the model-load support surface
//! ([`crate::lm::load`]); the generation loop arrives in later M3 work.

pub mod cache;
pub mod load;
pub mod model;
pub mod sample;
