//! Concrete speech-to-text model implementations.
//!
//! Each architecture here is ported directly (weights, layer order, numerics),
//! spanning both families of the STT trait architecture: CTC / non-AR models
//! whose inference is a single forward over the raw waveform followed by a
//! greedy per-frame collapse (or an encoder tower feeding a separate decoder),
//! and autoregressive encoder/decoder models:
//!
//! - `wav2vec2` — Wav2Vec2 CTC (`facebook/wav2vec2-base-960h`): a
//!   convolutional feature encoder + transformer + per-frame CTC head,
//!   decoded by greedy collapse over a character vocabulary. A non-AR model
//!   whose inference is a single forward over the raw waveform, so it exposes
//!   its own inherent CTC API.
//! - `whisper` — OpenAI Whisper: a convolutional + transformer audio encoder
//!   feeding a cross-attention text decoder, implementing the autoregressive
//!   [`crate::audio::stt::model::AutoregressiveStt`] family trait (`encode` +
//!   per-token `decode_step` + per-block KV cache) and running its own
//!   [`crate::audio::stt::model::Transcribe`] decoding task (greedy decode +
//!   logit filters + temperature fallback + the 30-second seek loop +
//!   language detection).
//! - `qwen3_asr` — the Qwen3-ASR audio encoder: a Conv2d stem (~8x mel
//!   downsample) + a transformer self-attention encoder, producing audio
//!   embeddings for the Qwen3 forced aligner. Not a standalone transcriber —
//!   it is the audio tower consumed by the (later) aligner/decoder.
//! - `sensevoice` — SenseVoice-Small (`mlx-community/SenseVoiceSmall`): a
//!   non-autoregressive CTC recognizer fronted by a Kaldi fbank + Low-Frame-Rate
//!   stacking + CMVN front-end and built on the FunASR/Paraformer SANM (a
//!   self-attention network with FSMN memory) encoder, with a CTC head and a
//!   small prompt-embedding table injecting the rich-transcription query rows
//!   (language / event / emotion / inverse-text-normalization). Inference is a
//!   single forward followed by a greedy per-frame collapse, so it implements
//!   [`crate::audio::stt::model::CtcModel`].

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub mod wav2vec2;

#[cfg(feature = "whisper")]
#[cfg_attr(docsrs, doc(cfg(feature = "whisper")))]
pub mod whisper;

#[cfg(feature = "qwen3-asr")]
#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr")))]
pub mod qwen3_asr;

#[cfg(feature = "sensevoice")]
#[cfg_attr(docsrs, doc(cfg(feature = "sensevoice")))]
pub mod sensevoice;
