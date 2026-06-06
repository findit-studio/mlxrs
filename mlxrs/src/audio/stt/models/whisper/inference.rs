//! The model-forward abstraction the Whisper decode pipeline drives —
//! [`WhisperInference`].
//!
//! The decoding task ([`super::decoding::DecodingTask`]), the language-detection
//! and segment-decode free functions ([`super::decoding::detect_language`] /
//! [`super::decoding::decode`] / [`super::decoding::transcribe`]), and the
//! word-timestamp DTW ([`super::timing::find_alignment`]) all bottom out on a
//! small set of model primitives: encode a mel, run the decoder (single,
//! batched, lazy-token, and cross-`qk` variants), broadcast the encoder states
//! across best-of-N rows, and read a handful of static accessors (the
//! dimensions, the alignment heads, the vocabulary bound). This trait names
//! exactly that surface so the pipeline can drive any backend that implements
//! it, not only the MLX [`WhisperModel`].
//!
//! The MLX [`WhisperModel`] is the only implementation today; the methods are
//! thin forwards to its existing inherent methods, byte-identical in behaviour.
//!
//! ## Two method families
//!
//! The surface partitions into the two groups a non-MLX backend treats
//! differently:
//!
//! - **Inference primitives** — [`encode`](WhisperInference::encode),
//!   [`decode_tokens`](WhisperInference::decode_tokens),
//!   [`decode_tokens_batched`](WhisperInference::decode_tokens_batched),
//!   [`decode_token_lazy`](WhisperInference::decode_token_lazy),
//!   [`decode_step_with_cross_qk`](WhisperInference::decode_step_with_cross_qk),
//!   [`forward_with_cross_qk`](WhisperInference::forward_with_cross_qk), and
//!   [`broadcast_encoder_states`](WhisperInference::broadcast_encoder_states).
//!   These run the actual encoder / decoder forward; a different backend
//!   reimplements them against its own weights.
//! - **Metadata / static accessors** — [`dims`](WhisperInference::dims),
//!   [`alignment_heads`](WhisperInference::alignment_heads), and
//!   [`validate_token_ids`](WhisperInference::validate_token_ids). A different
//!   backend resolves these from its own model description.
//!
//! ## `Array` at the boundary
//!
//! Every inference method speaks in MLX [`Array`] for the logits, audio
//! features, encoder states, and cross-attention weights, and the associated
//! [`Cache`](WhisperInference::Cache) is the MLX decoder KV cache. The whole
//! decode pipeline is written against [`Array`]; a backend that computes off
//! device (e.g. host f16 tensors) converts at its own method boundaries so the
//! pipeline above it is unchanged.

use crate::{Array, Result};

use super::{
  config::{AlignmentHeads, ModelDimensions},
  decoder::DecoderKvCache,
  model::{WhisperDecodeCache, WhisperModel},
};

/// The model-forward surface the Whisper decode + word-timestamp pipeline
/// drives — see the module docs.
///
/// Implemented by the MLX [`WhisperModel`] (the only backend today) as thin
/// forwards to its inherent methods. The associated [`Cache`](Self::Cache) is
/// the per-block decoder KV cache threaded by value through the single-token /
/// batched / lazy decode primitives.
pub trait WhisperInference {
  /// The per-block decoder KV cache the decode primitives thread by value
  /// (`None` before the first step). For the MLX backend this is the decoder's
  /// crate-private `DecoderKvCache`.
  type Cache;

  /// Encode a Whisper mel `(num_frames, n_mels)` into encoder states
  /// `(1, n_audio_ctx, n_audio_state)`.
  fn encode(&self, mel: &Array) -> Result<Array>;

  /// Run the decoder over a token sequence `tokens` `(1, T)` against the
  /// encoder states with an explicit caller-owned KV cache, returning
  /// `(logits, updated_cache)` — logits `(1, T, n_vocab)` cast to `f32`.
  fn decode_tokens(
    &self,
    tokens: &[u32],
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)>;

  /// Run the decoder over `n_group` parallel candidate rows — the batched
  /// (`n_group > 1`) analogue of [`Self::decode_tokens`] for best-of-N
  /// sampling. Returns `(logits, updated_cache)` with logits `(n_group, T,
  /// n_vocab)` cast to `f32`.
  fn decode_tokens_batched(
    &self,
    tokens: &[u32],
    n_group: usize,
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)>;

  /// Warm-step decode from a token already on-device — the lazy-input analogue
  /// of [`Self::decode_tokens`]. `token` is a `(1, 1)` `u32` array kept lazy, so
  /// the pipelined decode loop never round-trips the token through a host
  /// `&[u32]`.
  fn decode_token_lazy(
    &self,
    token: &Array,
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)>;

  /// One decode step that additionally returns the per-layer cross-attention
  /// weights, threading the caller-owned [`WhisperDecodeCache`] by `&mut` (a
  /// fresh cache prefills the whole prefix, a warm cache forwards only the new
  /// tail). Returns `(logits, cross_qk)` — logits `(1, T, n_vocab)` cast to
  /// `f32`, and one `(1, n_text_head, T, n_audio_ctx)` cross-attention tensor
  /// per decoder layer.
  fn decode_step_with_cross_qk(
    &self,
    cache: &mut WhisperDecodeCache,
    enc: &Array,
    tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)>;

  /// Encode `mel` and run the decoder over the full `tokens` sequence in one
  /// cacheless forward, returning `(logits, cross_qk)` — the entry the
  /// word-timestamp DTW drives. `logits` is the full `(1, T, n_vocab)` output
  /// cast to `f32`; `cross_qk` carries one `(1, n_text_head, T, n_audio_ctx)`
  /// weight tensor per decoder layer.
  fn forward_with_cross_qk(
    &self,
    mel: &Array,
    tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)>;

  /// Broadcast `(1, n_audio_ctx, n_audio_state)` encoder states to `(n_group,
  /// n_audio_ctx, n_audio_state)` for a batched best-of-N decode. `n_group == 1`
  /// returns the states unchanged (a clone).
  fn broadcast_encoder_states(&self, enc: &Array, n_group: usize) -> Result<Array>;

  /// The model dimensions.
  fn dims(&self) -> &ModelDimensions;

  /// The maximum decoder-cache context (in tokens) this backend can hold —
  /// the ceiling the decode loop must not overrun. The MLX backend keeps a
  /// per-block KV cache that grows to the model's text context, so this is
  /// `dims().n_text_ctx()` (`448` for released Whisper, the default); a backend
  /// whose decoder cache is a fixed smaller width (e.g. the WhisperKit CoreML
  /// `TextDecoder`, capped at its `key_cache` extent) returns that cap so the
  /// shared loop stops cleanly at the cache bound instead of overrunning it.
  #[inline]
  fn max_decoder_context(&self) -> usize {
    self.dims().n_text_ctx()
  }

  /// The word-timing alignment heads — the `(layer, head)` cross-attention
  /// heads the word-timestamp DTW averages.
  fn alignment_heads(&self) -> &AlignmentHeads;

  /// Reject any token id `>= n_vocab` before it reaches the decoder
  /// token-embedding gather. `context` names the calling entry for the typed
  /// error.
  ///
  /// # Errors
  /// [`crate::Error::OutOfRange`] on the first id `>= n_vocab`.
  fn validate_token_ids(&self, context: &'static str, tokens: &[u32]) -> Result<()>;
}

/// The MLX backend — thin forwards to the inherent [`WhisperModel`] methods,
/// byte-identical in behaviour.
impl WhisperInference for WhisperModel {
  type Cache = DecoderKvCache;

  #[inline]
  fn encode(&self, mel: &Array) -> Result<Array> {
    <WhisperModel as crate::audio::stt::model::AutoregressiveStt>::encode(self, mel)
  }

  #[inline]
  fn decode_tokens(
    &self,
    tokens: &[u32],
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)> {
    self.decode_tokens(tokens, encoder_states, cache)
  }

  #[inline]
  fn decode_tokens_batched(
    &self,
    tokens: &[u32],
    n_group: usize,
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)> {
    self.decode_tokens_batched(tokens, n_group, encoder_states, cache)
  }

  #[inline]
  fn decode_token_lazy(
    &self,
    token: &Array,
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)> {
    self.decode_token_lazy(token, encoder_states, cache)
  }

  #[inline]
  fn decode_step_with_cross_qk(
    &self,
    cache: &mut WhisperDecodeCache,
    enc: &Array,
    tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)> {
    self.decode_step_with_cross_qk(cache, enc, tokens)
  }

  #[inline]
  fn forward_with_cross_qk(
    &self,
    mel: &Array,
    tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)> {
    self.forward_with_cross_qk(mel, tokens)
  }

  #[inline]
  fn broadcast_encoder_states(&self, enc: &Array, n_group: usize) -> Result<Array> {
    self.broadcast_encoder_states(enc, n_group)
  }

  #[inline]
  fn dims(&self) -> &ModelDimensions {
    self.dims()
  }

  #[inline]
  fn alignment_heads(&self) -> &AlignmentHeads {
    self.alignment_heads()
  }

  #[inline]
  fn validate_token_ids(&self, context: &'static str, tokens: &[u32]) -> Result<()> {
    self.validate_token_ids(context, tokens)
  }
}
