//! The architecture-agnostic [`EmbeddingModel`] seam for `mlxrs::embeddings`,
//! mirroring the embedding forward-pass call convention of the references:
//!
//! - python `mlx-embeddings` `models/base.py::BaseModelOutput`
//!   (`last_hidden_state` / `pooler_output`) — the dataclass every encoder
//!   `Model.__call__` returns;
//! - swift `MLXEmbedders/EmbeddingModel.swift` (`EmbeddingModelOutput` with
//!   `hiddenStates` / `pooledOutput`, and the `EmbeddingModel` protocol's
//!   `callAsFunction(_:positionIds:tokenTypeIds:attentionMask:)`).
//!
//! The [`encode`](super::encode::encode) entry is generic over this one trait:
//! a model is anything that maps token ids plus an attention mask to a
//! `(batch, seq_len, hidden)` hidden-state tensor (and, optionally, a
//! pre-pooled `(batch, hidden)` vector). Concrete architectures
//! (BERT / XLM-RoBERTa / Qwen3-embed / …) are **not** ported here (project
//! no-model-arch rule); the trait is the contract those impls — and the
//! deterministic `MockEmbeddingModel` test fixture below — satisfy.

use crate::{array::Array, error::Result};

/// The output of an [`EmbeddingModel`] forward pass.
///
/// Mirrors python `mlx-embeddings` `BaseModelOutput`
/// (`last_hidden_state` + `pooler_output`) and swift `MLXEmbedders`
/// `EmbeddingModelOutput` (`hiddenStates` + `pooledOutput`), pared to the two
/// fields the pooling pipeline consumes:
///
/// - [`last_hidden_state`](Self::last_hidden_state): the per-token hidden
///   states `(batch, seq_len, hidden)` — always present (unlike the python
///   dataclass's `Optional`, the Rust seam makes the contract that every
///   encoder produces hidden states explicit; a model with nothing to pool is
///   not representable rather than a `None` the pooler would have to reject).
/// - [`pooled_output`](Self::pooled_output): an optional model-provided pooled
///   `(batch, hidden)` vector (a BERT-style `pooler_output` / CLS head). Only
///   the [`PoolingStrategy::Cls`](super::PoolingStrategy::Cls) and
///   [`PoolingStrategy::None`](super::PoolingStrategy::None) paths can consume
///   it. The current [`pool`](super::pool) dispatcher still derives CLS from
///   the hidden states directly (python `cls_pooling`), but
///   [`encode`](super::encode::encode) will prefer `pooled_output` for
///   `Cls` / `None` when present (via its post-pooling fast-path). This field
///   remains `None` for models that do not emit a dedicated pooled vector.
///
/// No implicit eval: the arrays are lazy graph nodes; materialize via
/// [`Array`] accessors.
#[derive(Debug)]
pub struct EmbeddingModelOutput {
  /// Per-token hidden states, `(batch, seq_len, hidden)`. Fed to the pooling
  /// stage. Python `last_hidden_state`, swift `hiddenStates`.
  last_hidden_state: Array,
  /// Optional model-provided pooled vector, `(batch, hidden)`. Python
  /// `pooler_output`, swift `pooledOutput`. `None` for models that do not
  /// emit a dedicated pooler head.
  pooled_output: Option<Array>,
}

impl EmbeddingModelOutput {
  /// Construct an [`EmbeddingModelOutput`] from its two components.
  pub fn new(last_hidden_state: Array, pooled_output: Option<Array>) -> Self {
    Self {
      last_hidden_state,
      pooled_output,
    }
  }

  /// Construct an output carrying only hidden states (no model-provided
  /// pooler head) — the common case for encoders pooled externally.
  pub fn from_hidden_state(last_hidden_state: Array) -> Self {
    Self::new(last_hidden_state, None)
  }

  /// The per-token hidden states, `(batch, seq_len, hidden)`.
  #[inline(always)]
  pub fn last_hidden_state(&self) -> &Array {
    &self.last_hidden_state
  }

  /// The optional model-provided pooled vector, `(batch, hidden)`.
  #[inline(always)]
  pub fn pooled_output(&self) -> Option<&Array> {
    self.pooled_output.as_ref()
  }

  /// Decompose into `(last_hidden_state, pooled_output)` by value.
  ///
  /// Allows callers to consume both arrays without cloning — the encode
  /// pipeline needs to move `pooled_output` into the post-pooling tail while
  /// retaining a reference to `last_hidden_state`'s shape for validation.
  #[inline(always)]
  pub fn into_parts(self) -> (Array, Option<Array>) {
    (self.last_hidden_state, self.pooled_output)
  }
}

/// An embedding model: maps token ids and an attention mask to per-token
/// hidden states (and, optionally, a pre-pooled vector).
///
/// Mirrors python `mlx-embeddings`'s encoder `Model.__call__(input_ids,
/// attention_mask=…)` and swift `MLXEmbedders`'s `EmbeddingModel`
/// `callAsFunction(_:…:attentionMask:)`. The [`encode`](super::encode::encode)
/// entry only ever needs [`forward`](Self::forward).
///
/// - `&self` — weights are treated as immutable after load, so encoding does
///   not require `&mut` on the model (matching the references, where the
///   module is frozen for inference). This documents immutable inference only;
///   whether a model instance can be used from concurrent encode calls depends
///   on the concrete model's thread-safety and MLX / [`Array`] constraints.
/// - `input_ids` — an `I32` `(batch, seq_len)` array of token ids, padded to
///   the batch's max length by the caller ([`encode`](super::encode::encode)
///   builds it). `I32` is MLX's default index dtype for the embedding
///   `take` / gather (matching `lm/generate.rs::token_window`), so a model's
///   lookup can index with it directly without casting.
/// - `attention_mask` — a `(batch, seq_len)` array, `1` for real tokens and
///   `0` for padding. Passed through to the pooling stage so padded positions
///   are excluded. Models that build internal additive attention biases derive
///   them from this mask.
/// - returns — an [`EmbeddingModelOutput`] whose `last_hidden_state` is
///   `(batch, seq_len, hidden)` in the model's compute dtype. No implicit eval
///   here; the pooling stage composes lazily and the caller evaluates.
pub trait EmbeddingModel {
  /// Run a forward pass and return the hidden states (and optional pooled
  /// output).
  ///
  /// Errors propagate as [`crate::Error`] (shape / backend); a model never
  /// panics on a recoverable mismatch.
  fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<EmbeddingModelOutput>;
}

/// A deterministic, dependency-free [`EmbeddingModel`] used across the
/// embeddings test suite (the [`encode`](super::encode::encode) flow tests
/// below and in `tests/`).
///
/// `forward` ignores the input *values* and returns a fixed
/// `(batch, seq_len, hidden)` hidden-state tensor whose entry at
/// `[b, s, :]` is `canned[s]` (one `hidden`-length row per sequence position,
/// identical across the batch). This makes the pooled result exactly
/// hand-computable from the mask: e.g. mean pooling over a 2-real-token row
/// averages `canned[0]` and `canned[1]`. It is intentionally a `#[cfg(test)]`
/// helper — exported for the crate's own tests, not a public API.
#[cfg(test)]
pub(crate) struct MockEmbeddingModel {
  /// Per-position hidden rows: `canned[s]` is the `(hidden,)` row emitted at
  /// sequence position `s`. All rows must share the same length (`hidden`).
  pub canned: Vec<Vec<f32>>,
  /// Optional per-batch-item rows for the model-provided `pooled_output`:
  /// `pooled[b]` is the `(hidden,)` pooler row for batch item `b`. When
  /// `Some`, `forward` emits a `(batch, hidden)` `pooled_output` (tiling /
  /// truncating these rows to the actual batch); when `None`,
  /// `pooled_output` is `None` (the common encoder case). Used to exercise
  /// the `Cls`/`None` pooled-output path in [`encode`](super::encode::encode).
  pub pooled: Option<Vec<Vec<f32>>>,
}

#[cfg(test)]
impl MockEmbeddingModel {
  /// Build a mock whose position-`s` hidden row is `canned[s]`. The longest
  /// supplied row defines `hidden`; shorter rows are zero-padded on the right
  /// so every position has a uniform width (keeps the fixture forgiving). No
  /// model-provided `pooled_output` (use [`with_pooled`](Self::with_pooled)
  /// to add one).
  pub(crate) fn new(canned: Vec<Vec<f32>>) -> Self {
    let hidden = canned.iter().map(Vec::len).max().unwrap_or(0);
    let canned = canned
      .into_iter()
      .map(|mut row| {
        row.resize(hidden, 0.0);
        row
      })
      .collect();
    Self {
      canned,
      pooled: None,
    }
  }

  /// Attach a model-provided `pooled_output`: `pooled[b]` is the `(hidden,)`
  /// pooler row for batch item `b`. `forward` then returns a `(batch, hidden)`
  /// `pooled_output` whose rows are `pooled` tiled / truncated to the request
  /// batch (rows are zero-padded on the right to a uniform width, like
  /// [`new`](Self::new)). Used to test the `Cls`/`None` pooled-output path.
  pub(crate) fn with_pooled(mut self, pooled: Vec<Vec<f32>>) -> Self {
    let hidden = pooled.iter().map(Vec::len).max().unwrap_or(0);
    let pooled = pooled
      .into_iter()
      .map(|mut row| {
        row.resize(hidden, 0.0);
        row
      })
      .collect();
    self.pooled = Some(pooled);
    self
  }
}

#[cfg(test)]
impl EmbeddingModel for MockEmbeddingModel {
  fn forward(&self, input_ids: &Array, _attention_mask: &Array) -> Result<EmbeddingModelOutput> {
    // input_ids is (batch, seq_len); tile the canned per-position rows across
    // the batch. seq_len must not exceed the number of canned positions.
    let shape = input_ids.shape();
    let (batch, seq) = match shape.as_slice() {
      [b, s] => (*b, *s),
      _ => {
        return Err(crate::error::Error::ShapeMismatch {
          message: format!(
            "MockEmbeddingModel::forward expects (batch, seq_len) ids, got {shape:?}"
          ),
        });
      }
    };
    if seq > self.canned.len() {
      return Err(crate::error::Error::ShapeMismatch {
        message: format!(
          "MockEmbeddingModel: seq_len {seq} exceeds canned positions {}",
          self.canned.len()
        ),
      });
    }
    let hidden = self.canned.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(batch * seq * hidden);
    for _ in 0..batch {
      for row in self.canned.iter().take(seq) {
        data.extend_from_slice(row);
      }
    }
    let last_hidden_state = Array::from_slice::<f32>(&data, &(batch, seq, hidden))?;

    // Optional model-provided pooled output: tile `self.pooled` rows to the
    // request batch (cycling if fewer rows than batch items were supplied)
    // and emit a `(batch, pooled_hidden)` array. `None` → no pooler head.
    let pooled_output = match &self.pooled {
      None => None,
      Some(pooled) => {
        if pooled.is_empty() {
          return Err(crate::error::Error::ShapeMismatch {
            message: "MockEmbeddingModel: pooled_output rows must be non-empty".to_string(),
          });
        }
        let pooled_hidden = pooled[0].len();
        let mut pdata = Vec::with_capacity(batch * pooled_hidden);
        for b in 0..batch {
          pdata.extend_from_slice(&pooled[b % pooled.len()]);
        }
        Some(Array::from_slice::<f32>(&pdata, &(batch, pooled_hidden))?)
      }
    };

    Ok(EmbeddingModelOutput::new(last_hidden_state, pooled_output))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn mock_forward_tiles_canned_rows_across_batch() {
    let model = MockEmbeddingModel::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    // batch 2, seq 2 -> hidden rows canned[0], canned[1] per batch item.
    let ids = Array::from_slice::<i32>(&[7, 8, 9, 10], &(2, 2)).unwrap();
    let mask = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0], &(2, 2)).unwrap();
    let out = model.forward(&ids, &mask).unwrap();
    assert_eq!(out.last_hidden_state().shape(), vec![2, 2, 2]);
    assert!(out.pooled_output().is_none());
    let (mut lhs, _) = out.into_parts();
    assert_eq!(
      lhs.to_vec::<f32>().unwrap(),
      // batch 0: [1,2],[3,4]   batch 1: [1,2],[3,4]
      vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );
  }

  #[test]
  fn mock_forward_rejects_wrong_rank() {
    let model = MockEmbeddingModel::new(vec![vec![1.0, 2.0]]);
    let bad = Array::from_slice::<i32>(&[1, 2, 3], &(3,)).unwrap(); // rank-1
    let mask = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(3,)).unwrap();
    assert!(model.forward(&bad, &mask).is_err());
  }

  #[test]
  fn mock_forward_rejects_seq_longer_than_canned() {
    let model = MockEmbeddingModel::new(vec![vec![1.0, 2.0]]); // 1 position
    let ids = Array::from_slice::<i32>(&[1, 2], &(1, 2)).unwrap(); // seq 2 > 1
    let mask = Array::from_slice::<f32>(&[1.0, 1.0], &(1, 2)).unwrap();
    assert!(model.forward(&ids, &mask).is_err());
  }
}
