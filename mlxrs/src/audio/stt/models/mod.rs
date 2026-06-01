//! Concrete speech-to-text model implementations.
//!
//! Each architecture here is ported directly (weights, layer order, numerics)
//! rather than left to user code, because it does not fit the autoregressive
//! [`crate::audio::stt::model::Model`] trait (encoder + per-token
//! cross-attention `decode_step` + KV cache). They are CTC / non-AR models
//! whose inference is a single forward over the raw waveform followed by a
//! greedy per-frame collapse, so each exposes its own inherent API instead.
//!
//! - [`wav2vec2`] — Wav2Vec2 CTC (`facebook/wav2vec2-base-960h`): a
//!   convolutional feature encoder + transformer + per-frame CTC head,
//!   decoded by greedy collapse over a character vocabulary.

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub mod wav2vec2;
