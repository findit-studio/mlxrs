//! Shared neural-network layers reusable by every model family (`lm` /
//! `vlm` / `audio` / `embeddings`), as opposed to the LM-scoped primitives in
//! [`crate::lm::nn`] (RoPE / fast-SDPA attention / norms / MoE switch layers,
//! which the LM/VLM transformer stack composes).
//!
//! This top-level module holds layers with no LM dependency, so they stay
//! reachable from `embeddings` (which does not enable the `lm` feature) as
//! well as from `lm` / `vlm` / `audio`:
//!
//! - [`crate::nn::quantized`] — the dense + quantized linear layers
//!   ([`crate::nn::Linear`] / [`crate::nn::QuantizedLinear`]) and the
//!   quantize-aware [`crate::nn::MaybeQuantizedLinear`] abstraction a model
//!   uses to load either a dense or an 8-bit/4-bit quantized checkpoint through
//!   one code path (`mlx.nn.Linear` / `mlx.nn.QuantizedLinear`), plus the
//!   embedding analogue [`crate::nn::MaybeQuantizedEmbedding`]
//!   (`mlx.nn.Embedding` / `mlx.nn.QuantizedEmbedding`).

pub mod quantized;

pub use quantized::{
  Linear, MaybeQuantizedEmbedding, MaybeQuantizedLinear, QuantizedEmbedding, QuantizedLinear,
};
