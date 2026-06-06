//! The concrete Whisper inference backend the decode pipeline drives —
//! [`WhisperBackend`] and its associated [`WhisperCache`].
//!
//! Phase 1 ([`super::inference::WhisperInference`]) named the model-forward
//! surface as a trait so the decode + word-timestamp pipeline could drive any
//! backend. The pipeline was pinned to the single MLX
//! [`WhisperModel`](super::model::WhisperModel) through a `dyn
//! WhisperInference<Cache = DecoderKvCache>` trait-object alias, which can hold
//! exactly one `Cache` type and so cannot carry a second backend whose cache is
//! a different type.
//!
//! [`WhisperBackend`] replaces that alias with an enum over the concrete
//! backends, mirroring the proven `lfm` `BackendImpl` pattern (an ORT host
//! variant + an Apple-Silicon MLX variant, dispatched by `match`, with the
//! associated cache itself an enum so each variant keeps its cache in its own
//! native form). Here the always-present variant is the MLX
//! [`WhisperModel`]; on `macos` + `aarch64` a second CoreML / Neural-Engine
//! variant is compiled in and auto-selected from the checkpoint shape (a
//! `.mlmodelc` bundle). The enum implements [`WhisperInference`] by delegating
//! to the active variant, so the whole pipeline stays written against one type
//! and runs byte-identically on the MLX path.
//!
//! The variants borrow their backend (`&WhisperModel`) rather than owning it:
//! the public Whisper entry points hold the loaded model behind `&self`, and
//! the pipeline's free functions already take `&WhisperBackend`, so a borrowing
//! enum is constructed at the entry exactly where a `&WhisperModel` was coerced
//! to the `dyn` alias before — no model move or clone.

use crate::{Array, Result};

use super::{
  config::{AlignmentHeads, ModelDimensions},
  decoder::DecoderKvCache,
  inference::WhisperInference,
  model::{WhisperDecodeCache, WhisperModel},
};

/// The per-block decoder KV cache the decode primitives thread by value — the
/// [`WhisperInference::Cache`] for [`WhisperBackend`].
///
/// One variant per backend, so each backend threads its cache in its own native
/// form with no cross-backend conversion. The always-present variant is the MLX
/// backend's crate-private [`DecoderKvCache`] (the per-block on-device K/V); the
/// MLX decode path round-trips through [`WhisperCache::Mlx`]. On Apple Silicon a
/// second `CoreMl` variant carries the CoreML decoder's explicit host-side KV
/// tensors (added with the CoreML backend).
pub enum WhisperCache {
  /// The MLX decoder's per-block on-device KV cache.
  Mlx(DecoderKvCache),
}

impl WhisperCache {
  /// Borrow the MLX inner cache — the helper the MLX dispatch arm uses to
  /// unwrap the caller-threaded [`WhisperCache`] before forwarding to
  /// [`WhisperModel`].
  #[inline]
  fn as_mlx(&self) -> &DecoderKvCache {
    match self {
      Self::Mlx(c) => c,
    }
  }
}

/// The concrete Whisper inference backend — see the module docs.
///
/// Borrows the active backend so it can be built at a `&self` entry point
/// exactly where a `&WhisperModel` was coerced to the old `dyn` alias. The
/// always-present [`WhisperBackend::Mlx`] variant drives the MLX
/// [`WhisperModel`]; on `macos` + `aarch64` a `CoreMl` variant drives the
/// on-Neural-Engine CoreML backend, auto-selected from a `.mlmodelc`
/// checkpoint (added with the CoreML backend).
pub enum WhisperBackend<'a> {
  /// The MLX (Metal) backend.
  Mlx(&'a WhisperModel),
}

impl WhisperInference for WhisperBackend<'_> {
  type Cache = WhisperCache;

  #[inline]
  fn encode(&self, mel: &Array) -> Result<Array> {
    match self {
      Self::Mlx(m) => m.encode(mel),
    }
  }

  fn decode_tokens(
    &self,
    tokens: &[u32],
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)> {
    match self {
      Self::Mlx(m) => {
        let (logits, c) =
          m.decode_tokens(tokens, encoder_states, cache.map(WhisperCache::as_mlx))?;
        Ok((logits, WhisperCache::Mlx(c)))
      }
    }
  }

  fn decode_tokens_batched(
    &self,
    tokens: &[u32],
    n_group: usize,
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)> {
    match self {
      Self::Mlx(m) => {
        let (logits, c) = m.decode_tokens_batched(
          tokens,
          n_group,
          encoder_states,
          cache.map(WhisperCache::as_mlx),
        )?;
        Ok((logits, WhisperCache::Mlx(c)))
      }
    }
  }

  fn decode_token_lazy(
    &self,
    token: &Array,
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)> {
    match self {
      Self::Mlx(m) => {
        let (logits, c) =
          m.decode_token_lazy(token, encoder_states, cache.map(WhisperCache::as_mlx))?;
        Ok((logits, WhisperCache::Mlx(c)))
      }
    }
  }

  #[inline]
  fn decode_step_with_cross_qk(
    &self,
    cache: &mut WhisperDecodeCache,
    enc: &Array,
    tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)> {
    match self {
      Self::Mlx(m) => m.decode_step_with_cross_qk(cache, enc, tokens),
    }
  }

  #[inline]
  fn forward_with_cross_qk(
    &self,
    mel: &Array,
    tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)> {
    match self {
      Self::Mlx(m) => m.forward_with_cross_qk(mel, tokens),
    }
  }

  #[inline]
  fn broadcast_encoder_states(&self, enc: &Array, n_group: usize) -> Result<Array> {
    match self {
      Self::Mlx(m) => m.broadcast_encoder_states(enc, n_group),
    }
  }

  #[inline]
  fn dims(&self) -> &ModelDimensions {
    match self {
      Self::Mlx(m) => m.dims(),
    }
  }

  #[inline]
  fn alignment_heads(&self) -> &AlignmentHeads {
    match self {
      Self::Mlx(m) => m.alignment_heads(),
    }
  }

  #[inline]
  fn validate_token_ids(&self, context: &'static str, tokens: &[u32]) -> Result<()> {
    match self {
      Self::Mlx(m) => m.validate_token_ids(context, tokens),
    }
  }
}
