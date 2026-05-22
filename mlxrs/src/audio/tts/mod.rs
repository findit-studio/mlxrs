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
//!   phonemizer — G2P is model-specific).
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

pub mod generate;
pub mod model;

use crate::error::Result;

/// Text-preprocessing hook for the TTS synthesis pipeline.
///
/// A 1:1 port of mlx-audio-swift's [`TextProcessor`][swift-tp] protocol —
/// the *interface*, not a concrete phonemizer. Some TTS models (kokoro,
/// kitten-tts) require phonemized IPA input rather than raw text; a
/// [`TextProcessor`] converts natural-language text into the format the
/// target model expects.
///
/// Phonemization / G2P itself is **model-specific** and out of scope per
/// the project's [no per-model arch porting][noarch] rule — mlxrs ships the
/// hook, not a Misaki/eSpeak G2P implementation. A per-model crate
/// implements [`TextProcessor`] (e.g. a Misaki G2P adapter) and the model's
/// [`TtsModel::synthesize_segment`](model::TtsModel::synthesize_segment)
/// runs it; the [`tts_generate`](generate::tts_generate) driver itself
/// never phonemizes — it passes segment text through unchanged.
///
/// Why a separate hook trait rather than a method on
/// [`TtsModel`](model::TtsModel): mlx-audio-swift keeps `TextProcessor`
/// distinct from `SpeechGenerationModel` precisely so one G2P adapter can
/// be shared across several models, and so a caller can *inject* a custom
/// processor at load time (the swift `loadModel(..., textProcessor:)`
/// parameter). mlxrs mirrors that separation — per the
/// [mirror-reference-structure][mirror] rule, the reference's two distinct
/// protocols stay two distinct traits.
///
/// [swift-tp]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/TextProcessor.swift
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
/// [mirror]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/mirror-reference-structure.md
pub trait TextProcessor {
  /// Download or initialize any resources the processor needs before
  /// [`TextProcessor::process`] can run (a G2P lexicon, a weights file, …).
  ///
  /// Mirrors the swift `TextProcessor.prepare()`. The default impl is a
  /// no-op for processors that need no preparation (the swift protocol
  /// extension's default is likewise empty); call it once before the first
  /// `process`.
  fn prepare(&mut self) -> Result<()> {
    Ok(())
  }

  /// Convert input `text` into the format the target model expects (e.g.
  /// phonemized IPA).
  ///
  /// `language` is an optional locale code (`"en-us"`, `"en-gb"`, …) — the
  /// same `language: String?` argument the swift `process(text:language:)`
  /// takes; `None` lets the processor pick a default. Returns the processed
  /// string a per-model tokenizer then consumes.
  fn process(&self, text: &str, language: Option<&str>) -> Result<String>;
}
