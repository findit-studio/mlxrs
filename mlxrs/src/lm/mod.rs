//! Language Model (LM) support — text-only inference building blocks ported
//! from [mlx-lm](https://github.com/ml-explore/mlx-lm).
//!
//! M3 lands the sampling utilities ([`crate::lm::sample`]), the
//! architecture-agnostic generation foundation — the
//! [`Model`](crate::lm::model::Model) trait, the KV
//! [`cache`](crate::lm::cache), the model-load support surface
//! ([`crate::lm::load`]), the local load [`factory`](crate::lm::factory)
//! (`ModelConfiguration` + `model_type` registry + `load`), and the
//! [`generate`](crate::lm::generate) loop (`generate_step` /
//! `stream_generate` / `generate` + `make_sampler` /
//! `make_logits_processors`).

pub mod cache;
pub mod factory;
pub mod generate;
pub mod load;
pub mod model;
pub mod nn;
pub mod quant;
pub mod sample;
pub mod speculative;
/// Tool-call format parsers — Python `mlx_lm.tool_parsers.*`.
///
/// Surface re-export of [`crate::tokenizer::tools`] under the canonical
/// `lm::tool_parsers` path; the parser logic and per-format documentation
/// live in the module that owns the [`crate::tokenizer::wrapper::Tokenizer`]
/// consumer. Gated on the `tokenizer-tools` capability feature, which the
/// `lm` umbrella always pulls in.
#[cfg(feature = "tokenizer-tools")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-tools")))]
pub mod tool_parsers;
