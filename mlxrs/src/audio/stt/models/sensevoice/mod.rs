//! SenseVoice-Small (FunAudioLLM multilingual STT) — `mlx-community/SenseVoiceSmall`.
//!
//! A faithful port of mlx-audio's `SenseVoiceSmall`
//! ([`stt/models/sensevoice/sensevoice.py`][sv], with the swift
//! `MLXAudioSTT/Models/SenseVoice/` as the second parity reference). SenseVoice
//! is a **non-autoregressive CTC** recognizer: a single forward over the
//! fbank / LFR / CMVN features produces per-frame log-probabilities, and a
//! greedy blank-collapse yields text. There is no decoder, no KV cache, and no
//! token-by-token loop.
//!
//! ## Pipeline
//!
//! 1. **Front-end** ([`frontend`]) — Kaldi fbank (reusing
//!    [`crate::audio::features::compute_fbank_kaldi`] verbatim after a `2^15`
//!    waveform pre-scale) -> Low-Frame-Rate stacking (`7 x 80 -> 560`, stride
//!    `6`) -> global CMVN (`(feats + means) * istd`, from the model's `am.mvn`
//!    or the in-config fallback).
//! 2. **Encoder** ([`encoder`]) — the FunASR/Paraformer SANM (self-attention
//!    network with FSMN memory) tower: a width-changing first block, the
//!    constant-width main stack, a second `tp_encoders` stack, and three
//!    LayerNorms, fronted by a `sqrt(output_size)` scale + an additive
//!    sinusoidal position encoding. Every linear is quantize-aware via the
//!    shared [`crate::nn::MaybeQuantizedLinear`].
//! 3. **CTC head + rich-info decode** ([`model`]) — the `ctc_lo` projection, the
//!    prepended query-row assembly, the greedy collapse over the speech frames,
//!    and the language / emotion / event argmax heads, decoded through the
//!    [`tokenizer`] (SentencePiece / `tokens.json`). The
//!    [`crate::audio::stt::model::CtcModel`] (speech-only `logits`) +
//!    [`crate::audio::stt::model::Transcribe`] (encoder-once -> rich tags ->
//!    shared collapse) trait wiring, with the rich tags exposed through the
//!    model-local [`model::SenseVoiceResult`].
//!
//! ## Status
//!
//! This module is the feature-complete SenseVoice port: the [`config`] +
//! [`frontend`] glue + `sanitize`, the SANM/FSMN [`encoder`], the CTC head +
//! query-prefix assembly + rich-info extraction + [`tokenizer`] decode + the
//! [`model::SenseVoiceModel`] [`crate::audio::stt::model::CtcModel`] /
//! [`crate::audio::stt::model::Transcribe`] wiring, and the file-loading
//! [`loader`]: `from_weights` / `from_weights_quantized` (with the
//! `.scales`-presence quantization discriminator), the `am.mvn` / tokenizer
//! asset loading (the `post_load_hook` equivalent), the safetensors shard walk,
//! and the [`loader::load`] -> `Box<dyn Transcribe>` factory the STT registry
//! dispatches `model_type == "sensevoice"` to. Every piece is exercised by
//! shape + closed-form + synthetic-checkpoint oracle tests; the gated e2e
//! numeric-parity test against the real `mlx-community/SenseVoiceSmall`
//! checkpoint is a separate change.
//!
//! [sv]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sensevoice/sensevoice.py

pub mod config;
pub mod encoder;
pub mod frontend;
pub mod loader;
pub mod model;
pub mod tokenizer;

pub use config::{Config, EncoderConfig, FrontendConfig, MODEL_TYPE};
pub use encoder::{
  Encoder, EncoderLayerSANM, MultiHeadedAttentionSANM, PositionwiseFeedForward,
  SinusoidalPositionEncoder,
};
pub use frontend::{apply_cmvn, apply_lfr, compute_fbank, parse_am_mvn, sanitize};
pub use loader::{has_relevant_scales, load};
pub use model::{BLANK_ID, RichInfo, SenseVoiceModel, SenseVoiceResult, build_head};
pub use tokenizer::SenseVoiceTokenizer;
