//! The architecture-agnostic [`Model`] seam for `mlxrs::lm` text generation,
//! mirroring mlx-lm's `model(tokens, cache)` call convention
//! ([`mlx_lm.generate`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)).
//!
//! Everything in the generation loop is generic over this one trait: a model
//! is anything that maps a token window plus its per-layer KV cache to the
//! next-token logits. Concrete architectures are **not** ported here (per the
//! project's no-model-arch rule); the single feature-gated Qwen3 test vehicle
//! arrives later in the M3 stack. The trait is the contract those impls — and
//! the deterministic `MockModel` test fixture below — satisfy.

use crate::{array::Array, error::Result, lm::cache::KvCache};

/// A causal language model: maps a token window and its per-layer KV cache to
/// next-token logits.
///
/// Mirrors mlx-lm's `nn.Module.__call__(inputs, cache)` (and mlx-swift-lm's
/// `LanguageModel.callAsFunction`): the loop only ever needs `forward`.
///
/// - `&self` — weights are immutable after load, so generation never needs
///   `&mut` on the model (matching mlx-lm, where the module is frozen for
///   inference). This also lets one model back many concurrent caches.
/// - `tokens` — an integer `[B, S]` array (the prompt chunk during prefill, a
///   single `[B, 1]` token during decode).
/// - `cache` — one [`KvCache`] **per decoder layer**, mutated in place by the
///   attention blocks (`update_and_fetch`); `make_prompt_cache` builds it.
/// - returns — `[B, S, vocab_size]` logits in the model's compute dtype (the
///   loop slices the final position and normalizes; no implicit eval here).
pub trait Model {
  /// Run a forward pass, updating `cache` in place, and return the
  /// `[B, S, vocab_size]` logits.
  ///
  /// Errors propagate as [`crate::Error`] (shape/backend); a model never
  /// panics on a recoverable mismatch.
  fn forward(&self, tokens: &Array, cache: &mut [KvCache]) -> Result<Array>;

  /// Optional embeddings entry point for multimodal models (VLM, M4): run the
  /// decoder over pre-computed input embeddings instead of token ids.
  ///
  /// Declared, not required — the default returns [`crate::Error::Backend`]
  /// so the text-only loop never depends on it while the seam exists for
  /// later milestones. Text models inherit the default; VLMs override it.
  fn forward_embeddings(&self, _embeddings: &Array, _cache: &mut [KvCache]) -> Result<Array> {
    Err(crate::error::Error::Backend {
      message: "this model does not implement `forward_embeddings` (VLM seam, M4)".into(),
    })
  }
}

/// A deterministic, dependency-free [`Model`] used across the `lm` test suite
/// (cache wiring here; the generation loop in PR-3).
///
/// `forward` ignores the input values, advances every supplied cache by the
/// token-window length so cache wiring is observable, and returns a fixed
/// `[B, S, vocab]` logits array tiled from `canned` (one logit per vocab
/// entry, broadcast across batch and sequence). It is intentionally a
/// `#[cfg(test)]` helper — exported for the crate's own tests/PR-3, not a
/// public API.
#[cfg(test)]
pub(crate) struct MockModel {
  /// Per-vocab logit values; `canned.len()` is the vocab size.
  pub canned: Vec<f32>,
  /// Number of key/value heads of the fake `[B, n_kv_heads, S, head_dim]`
  /// state pushed into each cache entry (small; deterministic).
  pub n_kv_heads: usize,
  /// Head dim of the fake KV state.
  pub head_dim: usize,
}

#[cfg(test)]
impl MockModel {
  /// A `MockModel` whose argmax is the last vocab index (so greedy decoding
  /// is trivially predictable) with a tiny `1`-head, `2`-dim fake KV state.
  pub(crate) fn new(vocab: usize) -> Self {
    let canned = (0..vocab).map(|i| i as f32).collect();
    Self {
      canned,
      n_kv_heads: 1,
      head_dim: 2,
    }
  }
}

#[cfg(test)]
impl Model for MockModel {
  fn forward(&self, tokens: &Array, cache: &mut [KvCache]) -> Result<Array> {
    // tokens is [B, S]; derive B and S the same way the loop will.
    let shape = tokens.shape();
    let (batch, seq) = match shape.as_slice() {
      [b, s] => (*b, *s),
      [s] => (1, *s),
      _ => {
        return Err(crate::error::Error::ShapeMismatch {
          message: format!("MockModel::forward expects [B, S] tokens, got {shape:?}"),
        });
      }
    };
    let vocab = self.canned.len();

    // Push a deterministic [B, n_kv_heads, S, head_dim] KV step into every
    // layer's cache so a multi-step drive shows `offset()` advancing.
    for layer in cache.iter_mut() {
      let elems = batch * self.n_kv_heads * seq * self.head_dim;
      let k = Array::from_slice::<f32>(
        &vec![1.0_f32; elems],
        &(batch, self.n_kv_heads, seq, self.head_dim),
      )?;
      let v = Array::from_slice::<f32>(
        &vec![2.0_f32; elems],
        &(batch, self.n_kv_heads, seq, self.head_dim),
      )?;
      layer.update_and_fetch(&k, &v)?;
    }

    // Logits: tile `canned` across [B, S, vocab].
    let mut data = Vec::with_capacity(batch * seq * vocab);
    for _ in 0..batch * seq {
      data.extend_from_slice(&self.canned);
    }
    Array::from_slice::<f32>(&data, &(batch, seq, vocab))
  }
}

#[cfg(test)]
mod tests {
  //! Task 1.5: the [`MockModel`] + trait/cache integration — the reusable
  //! deterministic fixture PR-3's generation-loop tests will share (both
  //! live in-crate, so this `#[cfg(test)] pub(crate)` mock is visible to
  //! them).

  use super::*;
  use crate::lm::cache::{CacheConfig, KvCache, make_prompt_cache};

  /// A `[B, S]` int token window (the loop's `forward` input shape).
  fn tokens(ids: &[i32], batch: usize, seq: usize) -> Array {
    Array::from_slice::<i32>(ids, &(batch, seq)).unwrap()
  }

  #[test]
  fn mock_model_forward_uses_cache() {
    let model = MockModel::new(5); // vocab 5, argmax == index 4
    let cfg = CacheConfig {
      num_hidden_layers: 2,
      sliding_window: None,
    };
    let mut cache = make_prompt_cache(&cfg);
    assert_eq!(cache.len(), 2);
    assert!(cache.iter().all(KvCache::is_empty));

    // Step 1: a 3-token prompt chunk -> logits [1, 3, 5]; every layer's
    // cache advances by 3.
    let mut logits = model
      .forward(&tokens(&[1, 2, 3], 1, 3), &mut cache)
      .unwrap();
    assert_eq!(logits.shape(), vec![1, 3, 5]);
    assert!(cache.iter().all(|c| c.offset() == 3));
    assert!(cache.iter().all(|c| !c.is_empty()));
    // Canned logits are 0..vocab tiled per (B,S); argmax is the last index.
    let v = logits.to_vec::<f32>().unwrap();
    assert_eq!(&v[0..5], &[0.0, 1.0, 2.0, 3.0, 4.0]);

    // Step 2: a single decode token -> [1, 1, 5]; cache advances to 4.
    let mut logits = model.forward(&tokens(&[4], 1, 1), &mut cache).unwrap();
    assert_eq!(logits.shape(), vec![1, 1, 5]);
    assert!(cache.iter().all(|c| c.offset() == 4));
    assert_eq!(
      logits.to_vec::<f32>().unwrap(),
      vec![0.0, 1.0, 2.0, 3.0, 4.0]
    );
  }

  #[test]
  fn forward_embeddings_default_is_unimplemented_seam() {
    let model = MockModel::new(3);
    let mut cache: Vec<KvCache> = Vec::new();
    let emb = Array::from_slice::<f32>(&[0.0, 1.0], &(1usize, 1, 2)).unwrap();
    // The VLM (M4) seam is declared but not implemented for text models.
    assert!(model.forward_embeddings(&emb, &mut cache).is_err());
  }

  #[test]
  fn forward_rejects_wrong_token_rank() {
    let model = MockModel::new(3);
    let mut cache: Vec<KvCache> = Vec::new();
    let bad = Array::from_slice::<f32>(&[1.0], &(1usize, 1, 1)).unwrap(); // 3-D
    assert!(model.forward(&bad, &mut cache).is_err());
  }
}
