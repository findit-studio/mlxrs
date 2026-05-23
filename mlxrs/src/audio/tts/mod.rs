//! Text-to-speech (TTS) — the architecture-agnostic synthesis seam.
//!
//! Ports the *shape* of mlx-audio's TTS support surface — the model-
//! agnostic [`tts/generate.py`][tts-gen] entry point, the per-model
//! `Model.generate` contract ([`tts/models/base.py`][tts-base]'s
//! `GenerationResult` envelope), and mlx-audio-swift's
//! [`MLXAudioTTS`][swift-tts] package
//! ([`SpeechGenerationModel`][swift-gen] + [`TextProcessor`][swift-tp]) —
//! as three submodules:
//!
//! - [`model`] — the [`TtsModel`](model::TtsModel) trait every concrete TTS
//!   architecture (kokoro / csm / bark / qwen3-tts / …) implements.
//! - [`generate`] — the [`tts_generate`](generate::tts_generate) Iterator
//!   that drives any [`TtsModel`](model::TtsModel) (text → assembled /
//!   streamed [`AudioChunk`](generate::AudioChunk)s), plus
//!   [`join_audio`](generate::join_audio) (concatenate every chunk into one
//!   waveform), the
//!   [`tts_generate_with_reference`](generate::tts_generate_with_reference) /
//!   [`join_audio_with_reference`](generate::join_audio_with_reference)
//!   zero-shot voice-clone entry points (threading a
//!   [`TtsReference`](generate::TtsReference)), and the config / segment /
//!   chunk types.
//! - [`TextProcessor`] (in this module) — the text-preprocessing **hook**
//!   the synthesis pipeline exposes (the *interface*, not a concrete
//!   phonemizer — G2P is model-specific). [`BasicTextProcessor`] in
//!   [`text_processor`] is a no-G2P default impl (NFC + lowercase +
//!   whitespace collapse).
//! - [`g2p`] — grapheme-to-phoneme subsystem (the [`g2p::Phonemizer`]
//!   trait, in-memory [`g2p::CMUDict`] lexicon + local-file loader,
//!   ARPAbet→IPA mapper). A separate `NeuralPhonemizer` orchestrator
//!   ships in a follow-up commit on this branch.
//!
//! This mirrors the existing [`crate::audio::stt`] STT support surface:
//! `stt` ships the [`Model`](crate::audio::stt::model::Model) trait + the
//! [`stt_generate`](crate::audio::stt::generate::stt_generate) loop, NOT
//! whisper-the-model; `tts` ships the [`TtsModel`](model::TtsModel) trait +
//! the [`tts_generate`](generate::tts_generate) loop, NOT kokoro-the-model.
//!
//! ## Out of scope — per-model architectures
//!
//! Per the project's [no per-model arch porting][noarch] rule, mlxrs ships
//! **no** concrete TTS model implementations. Every `tts/models/*`
//! architecture in mlx-audio — kokoro (ALBERT prosody encoder + iSTFT
//! decoder), csm / sesame (RVQ backbone + mimi codec), bark
//! (coarse/fine/semantic transformers), qwen3-tts, chatterbox, dia, …
//! roughly 40 model packages — is *per-model* and excluded. Those plug into
//! the [`TtsModel`](model::TtsModel) trait from user code. The shared
//! surface here is only what parameterizes *over* the model: the synthesis
//! trait, the generation/streaming driver, the config / segment / chunk
//! types, and the [`TextProcessor`] hook.
//!
//! Also excluded as per-model / out of this port's scope: the per-model
//! `convert.py` weight remappers, mlx-audio's `interpolate.py`
//! (model-specific), the `AudioPlayer` real-time playback device
//! ([`tts/audio_player.py`] — an OS-audio-device concern, not synthesis),
//! and per-run timing / memory telemetry (`real_time_factor`,
//! `peak_memory_usage`, the tokens-per-sec dicts mlx-audio's
//! `GenerationResult` also carries — instrumentation, left to the caller,
//! mirroring how [`crate::audio::stt`] yields a bare
//! [`crate::lm::generate::GenStep`]).
//!
//! [tts-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/generate.py
//! [tts-base]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/base.py
//! [`tts/audio_player.py`]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/audio_player.py
//! [swift-tts]: https://github.com/Blaizzy/mlx-audio-swift/tree/main/Sources/MLXAudioTTS
//! [swift-gen]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/Generation.swift
//! [swift-tp]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/TextProcessor.swift
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

pub mod g2p;
pub mod generate;
pub mod load;
pub mod model;
pub mod text_processor;

pub use text_processor::{BasicTextProcessor, TextProcessor};
