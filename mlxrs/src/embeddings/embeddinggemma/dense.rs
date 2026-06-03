//! EmbeddingGemma's `2_Dense` / `3_Dense` projection head.
//!
//! Ports the `self.dense` list of `mlx-embeddings`'s `models/gemma3_text.py`
//! `Model`: two bias-free linear layers
//! `[Linear(hidden, hidden*4), Linear(hidden*4, hidden)]` applied in sequence
//! to the **pooled** sentence vector, with **no activation between them** (the
//! SentenceTransformers `Dense` modules in EmbeddingGemma's `2_Dense` /
//! `3_Dense` folders use the identity activation). The reference applies these
//! *after* mean pooling and *before* L2-normalization:
//!
//! ```text
//! text_embeds = mean_pooling(hidden_states, attention_mask)
//! for dense in self.dense:
//!     text_embeds = dense(text_embeds)
//! text_embeds = normalize_embeddings(text_embeds)
//! ```
//!
//! so this is the projection that turns a `(batch, hidden)` pooled vector into
//! the `(batch, hidden)` projected embedding the model L2-normalizes.

use std::collections::HashMap;

use crate::{array::Array, error::Result, lm::quant::PerLayerQuantization};

use super::{config::DenseConfig, shared::Linear};

/// The two-layer bias-free projection head EmbeddingGemma applies to the pooled
/// vector: `dense1(dense0(x))`, both `nn.Linear(_, _, bias=False)`, identity
/// activation between them.
#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
#[derive(Debug)]
pub(crate) struct DenseHead {
  /// `(intermediate, hidden)` — `mlx-embeddings`'s `dense.0` (the `Linear(hidden,
  /// hidden*4)`, stored `(out, in)`). Quantize-aware: an mlx-embeddings
  /// quantized checkpoint quantizes the ST `Dense` `nn.Linear`s too (the
  /// `class_predicate` quantizes every `nn.Linear`), so this loads quantized
  /// when its `.scales` sibling is present, else dense.
  dense0: Linear,
  /// `(hidden, intermediate)` — `mlx-embeddings`'s `dense.1` (the
  /// `Linear(hidden*4, hidden)`).
  dense1: Linear,
}

#[cfg(feature = "embeddinggemma")]
impl DenseHead {
  /// Build the head from the (sanitized) weight map, whose keys are
  /// `dense.0.weight` / `dense.1.weight` (the layout [`super::sanitize`] produces
  /// from the checkpoint's `2_Dense.linear` / `3_Dense.linear`, classified by the
  /// source module-prefix number: `2_Dense → dense.0`, `3_Dense → dense.1`). Each
  /// layer is shape-pinned to its exact `(out, in)` dimensions (dense) or
  /// auto-detects a quantized triple from its `.scales` sibling, exactly like the
  /// backbone projections; `quant` carries the resolved per-layer scheme
  /// parameters.
  pub(crate) fn from_weights(
    config: &DenseConfig,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let dense0 = Linear::from_weights(
      weights,
      "dense.0",
      inter,
      hidden,
      "Dense head dense.0 weight (intermediate, hidden)",
      quant,
    )?;
    let dense1 = Linear::from_weights(
      weights,
      "dense.1",
      hidden,
      inter,
      "Dense head dense.1 weight (hidden, intermediate)",
      quant,
    )?;
    Ok(Self { dense0, dense1 })
  }

  /// `dense1(dense0(x))` — the two bias-free projections, identity activation
  /// between them. `x` is `(batch, hidden)`; the result is `(batch, hidden)`.
  pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
    let h = self.dense0.forward(x)?;
    self.dense1.forward(&h)
  }

  /// `true` if both Dense-head layers loaded from a quantized checkpoint
  /// (test-only introspection for the quantized-load test).
  #[cfg(test)]
  pub(crate) fn is_quantized(&self) -> bool {
    self.dense0.is_quantized() && self.dense1.is_quantized()
  }
}
