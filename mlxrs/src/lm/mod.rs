//! Language Model (LM) support — text-only inference building blocks ported
//! from [mlx-lm](https://github.com/ml-explore/mlx-lm).
//!
//! M3 lands the sampling utilities ([`crate::lm::sample`]) and the
//! architecture-agnostic generation foundation — the
//! [`Model`](crate::lm::model::Model) trait and the KV
//! [`cache`](crate::lm::cache); the loader and generation loop arrive in
//! later M3 work.

pub mod cache;
pub mod model;
pub mod sample;
