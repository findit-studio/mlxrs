//! The concrete Whisper inference backend the decode pipeline drives —
//! [`WhisperBackend`] and its associated [`WhisperCache`].
//!
//! Phase 1 ([`super::inference::WhisperInference`]) named the model-forward
//! surface as a trait so the decode + word-timestamp pipeline could drive any
//! backend. The pipeline was pinned to the single MLX
//! [`WhisperModel`] through a `dyn
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

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::coreml::{CoreMlKvCache, CoreMlWhisper};

/// The per-block decoder KV cache the decode primitives thread by value — the
/// [`WhisperInference::Cache`] for [`WhisperBackend`].
///
/// One variant per backend, so each backend threads its cache in its own native
/// form with no cross-backend conversion. The always-present variant is the MLX
/// backend's crate-private `DecoderKvCache` (the per-block on-device K/V); the
/// MLX decode path round-trips through [`WhisperCache::Mlx`]. On Apple Silicon a
/// second `CoreMl` variant carries the CoreML decoder's explicit host-side KV
/// tensors (added with the CoreML backend).
pub enum WhisperCache {
  /// The MLX decoder's per-block on-device KV cache.
  Mlx(DecoderKvCache),
  /// The CoreML decoder's explicit host-side KV cache — Apple Silicon only.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  CoreMl(CoreMlKvCache),
}

impl WhisperCache {
  /// Borrow the MLX inner cache, or `None` if this is another backend's
  /// variant — the helper the MLX dispatch arm uses to unwrap the
  /// caller-threaded [`WhisperCache`] before forwarding to [`WhisperModel`].
  ///
  /// The pipeline only ever pairs an MLX backend with an MLX cache (it builds
  /// the cache from the same backend it dispatches on), so the cross-variant
  /// `None` is an internal-invariant guard, not a reachable state.
  #[inline]
  fn as_mlx(&self) -> Option<&DecoderKvCache> {
    match self {
      Self::Mlx(c) => Some(c),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(_) => None,
    }
  }

  /// Borrow the CoreML inner cache, or `None` if this is another backend's
  /// variant — the [`Self::as_mlx`] analogue for the CoreML dispatch arm.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  #[inline]
  fn as_coreml(&self) -> Option<&CoreMlKvCache> {
    match self {
      Self::CoreMl(c) => Some(c),
      Self::Mlx(_) => None,
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
  /// The CoreML / Neural-Engine backend — Apple Silicon only.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  CoreMl(&'a CoreMlWhisper),
}

impl WhisperInference for WhisperBackend<'_> {
  type Cache = WhisperCache;

  #[inline]
  fn encode(&self, mel: &Array) -> Result<Array> {
    match self {
      Self::Mlx(m) => m.encode(mel),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => m.encode(mel),
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
        let (logits, c) = m.decode_tokens(tokens, encoder_states, mlx_cache_arg(cache)?)?;
        Ok((logits, WhisperCache::Mlx(c)))
      }
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => {
        let (logits, c) = m.decode_tokens(tokens, encoder_states, coreml_cache_arg(cache)?)?;
        Ok((logits, WhisperCache::CoreMl(c)))
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
        let (logits, c) =
          m.decode_tokens_batched(tokens, n_group, encoder_states, mlx_cache_arg(cache)?)?;
        Ok((logits, WhisperCache::Mlx(c)))
      }
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => {
        let (logits, c) =
          m.decode_tokens_batched(tokens, n_group, encoder_states, coreml_cache_arg(cache)?)?;
        Ok((logits, WhisperCache::CoreMl(c)))
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
        let (logits, c) = m.decode_token_lazy(token, encoder_states, mlx_cache_arg(cache)?)?;
        Ok((logits, WhisperCache::Mlx(c)))
      }
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => {
        let (logits, c) = m.decode_token_lazy(token, encoder_states, coreml_cache_arg(cache)?)?;
        Ok((logits, WhisperCache::CoreMl(c)))
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
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => m.decode_step_with_cross_qk(cache, enc, tokens),
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
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => m.forward_with_cross_qk(mel, tokens),
    }
  }

  #[inline]
  fn broadcast_encoder_states(&self, enc: &Array, n_group: usize) -> Result<Array> {
    match self {
      Self::Mlx(m) => m.broadcast_encoder_states(enc, n_group),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => m.broadcast_encoder_states(enc, n_group),
    }
  }

  #[inline]
  fn dims(&self) -> &ModelDimensions {
    match self {
      Self::Mlx(m) => m.dims(),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => m.dims(),
    }
  }

  #[inline]
  fn max_decoder_context(&self) -> usize {
    match self {
      Self::Mlx(m) => m.max_decoder_context(),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => m.max_decoder_context(),
    }
  }

  #[inline]
  fn alignment_heads(&self) -> &AlignmentHeads {
    match self {
      Self::Mlx(m) => m.alignment_heads(),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => m.alignment_heads(),
    }
  }

  #[inline]
  fn validate_token_ids(&self, context: &'static str, tokens: &[u32]) -> Result<()> {
    match self {
      Self::Mlx(m) => m.validate_token_ids(context, tokens),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      Self::CoreMl(m) => m.validate_token_ids(context, tokens),
    }
  }
}

/// Unwrap a caller-threaded [`WhisperCache`] to its MLX inner cache for an MLX
/// dispatch arm. `None` (no prior step) passes through; a `Some` carrying
/// another backend's variant is an internal-invariant violation (the pipeline
/// never mixes a backend with another backend's cache).
#[inline]
fn mlx_cache_arg(cache: Option<&WhisperCache>) -> Result<Option<&DecoderKvCache>> {
  match cache {
    None => Ok(None),
    Some(c) => c.as_mlx().map(Some).ok_or_else(cache_variant_mismatch),
  }
}

/// Unwrap a caller-threaded [`WhisperCache`] to its CoreML inner cache for a
/// CoreML dispatch arm — the [`mlx_cache_arg`] analogue.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[inline]
fn coreml_cache_arg(cache: Option<&WhisperCache>) -> Result<Option<&CoreMlKvCache>> {
  match cache {
    None => Ok(None),
    Some(c) => c.as_coreml().map(Some).ok_or_else(cache_variant_mismatch),
  }
}

/// The typed error for a backend/cache variant mismatch — an internal-invariant
/// guard the cross-variant unwrap surfaces rather than panic.
#[inline]
fn cache_variant_mismatch() -> crate::Error {
  crate::Error::InvariantViolation(crate::error::InvariantViolationPayload::new(
    "WhisperBackend",
    "decode cache variant does not match the active backend (internal invariant violation)",
  ))
}
