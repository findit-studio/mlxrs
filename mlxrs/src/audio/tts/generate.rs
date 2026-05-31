//! End-to-end TTS synthesis: text → per-segment split → per-model
//! [`TtsModel::synthesize_segment`] → assembled / streamed audio chunks.
//!
//! Ported in *shape* from mlx-audio's model-agnostic TTS entry point
//! ([`tts/generate.py`][tts-gen]'s `generate_audio`), the per-model
//! `Model.generate` loops ([`kokoro/kokoro.py`][kokoro],
//! [`llama/llama.py`][llama] — consulted for the segment-iteration +
//! streaming-chunk shape, NOT the per-model decode algorithm, which lives in
//! per-model code per the `project_no_per_model_arch_porting`
//! rule), and mlx-audio-swift's
//! [`SpeechGenerationModel`][swift-gen] `generate` / `generateStream`.
//!
//! [`tts_generate`] composes text segmentation, the
//! [`super::model::TtsModel`] trait, and audio-chunk assembly into one
//! [`Iterator<Item = Result<AudioChunk>>`][iter] — the streaming analogue of
//! mlx-audio's `for result in model.generate(...)` loop, mirroring the
//! per-step iterator contract [`crate::audio::stt::generate::SttGenerator`]
//! exposes (so a caller familiar with the STT loop sees no new shape).
//!
//! No implicit eval: the driver never materializes the per-segment audio
//! [`Array`] — it forwards each segment's tensor straight into an
//! [`AudioChunk`], and [`join_audio`] concatenates lazily via
//! [`crate::ops::shape::concatenate`]. Materializing to `Vec<f32>` is the
//! caller's explicit `&mut` step ([`AudioChunk::samples`]).
//!
//! [tts-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/generate.py
//! [kokoro]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/kokoro/kokoro.py
//! [llama]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/llama/llama.py
//! [swift-gen]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/Generation.swift
//! [iter]: core::iter::Iterator

use derive_more::{Display, IsVariant};

use smol_str::format_smolstr;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    CapExceededPayload, DtypeMismatchPayload, EmptyInputPayload, Error, LayerKeyedPayload,
    RankMismatchPayload, Result,
  },
  ops,
};

use super::model::TtsModel;

/// Default voice id when a caller does not select one — `"af_heart"`,
/// mlx-audio `generate_audio`'s `voice` default.
pub const DEFAULT_VOICE: &str = "af_heart";

/// Default language code when a caller does not select one — `"en"`,
/// mlx-audio `generate_audio`'s `lang_code` default.
pub const DEFAULT_LANGUAGE: &str = "en";

/// Default sampling temperature — `0.7`, mlx-audio `generate_audio`'s
/// `temperature` default.
pub const DEFAULT_TEMPERATURE: f32 = 0.7;

/// Default token budget per text segment — `1200`, mlx-audio
/// `generate_audio`'s `max_tokens` default (also mlx-audio-swift's
/// `AudioGenerateParameters.maxTokens`).
pub const DEFAULT_MAX_TOKENS: usize = 1200;

/// Default streaming-segment interval in seconds — `2.0`, mlx-audio
/// `generate_audio`'s `streaming_interval` default.
pub const DEFAULT_STREAMING_INTERVAL: f32 = 2.0;

/// Maximum input-text length (UTF-8 bytes) [`tts_generate`] accepts before
/// rejecting up front — `1_048_576` (1 MiB).
///
/// A pre-allocation safety cap mirroring the STT loop's
/// [`SttGenConfig::max_audio_seconds`][stt-cap] philosophy: a crafted /
/// fuzzed multi-MB text blob would otherwise drive the per-segment split
/// (and every per-model `synthesize_segment` allocation) without bound.
/// 1 MiB of text is far longer than any realistic single TTS request
/// (~150k words). Inputs above this return a recoverable [`Error::CapExceeded`]
/// from the [`tts_generate`] constructor, before any segmentation work.
///
/// [stt-cap]: crate::audio::stt::generate::SttGenConfig::max_audio_seconds
pub const MAX_TEXT_BYTES: usize = 1024 * 1024;

/// Output audio container format — mlx-audio `generate_audio`'s
/// `audio_format` argument (the `format=` passed to its `audio_write`).
///
/// The TTS driver itself is format-agnostic (it yields raw-PCM
/// [`AudioChunk`]s); this enum is the *plumbed* caller intent a downstream
/// writer ([`crate::audio::io::save_wav`] and future encoders) consumes.
/// Mirrors mlx-audio's string `audio_format` as a typed enum (idiomatic
/// Rust — an unknown format is a compile error, not a runtime
/// `ValueError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Display, IsVariant)]
#[display("{}", self.as_str())]
pub enum AudioFormat {
  /// WAV (RIFF/PCM) — mlx-audio's default `audio_format="wav"`. The only
  /// format [`crate::audio::io::save_wav`] currently writes.
  #[default]
  Wav,
  /// FLAC — mlx-audio supports `audio_format="flac"`; mlxrs has no FLAC
  /// encoder yet (a planned `audio::io` follow-up), so this variant is the
  /// plumbed caller intent only.
  Flac,
}

impl AudioFormat {
  /// Lowercase string name — the `audio_format` value mlx-audio's
  /// `generate_audio` / `audio_write` accept (`"wav"`, `"flac"`).
  #[must_use]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Wav => "wav",
      Self::Flac => "flac",
    }
  }
}

/// How [`tts_generate`] segments the input text before synthesis.
///
/// mlx-audio's per-model `generate` loops split the prompt before the
/// per-segment synthesis loop — kokoro on a `split_pattern` regex
/// (`r"\n+"`), llama on `"\n"`. mlxrs ships a regex-free split (no
/// `regex` dependency in the audio surface): the two modes below cover the
/// mlx-audio defaults. A model wanting a bespoke segmentation pre-splits the
/// text itself and calls [`tts_generate`] once per segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Display, IsVariant)]
#[display("{}", self.as_str())]
pub enum TextSegmentation {
  /// One segment per run of newlines, blank segments dropped — the
  /// mlx-audio kokoro default (`split_pattern=r"\n+"`). This is the
  /// [`Default`].
  #[default]
  Newlines,
  /// Treat the whole input as a single segment (no splitting) — for models
  /// / callers that do their own chunking, or short single-line prompts.
  Whole,
}

impl TextSegmentation {
  /// Lowercase string name — the mlx-audio segmentation mode this variant
  /// corresponds to (`"newlines"`, `"whole"`).
  #[must_use]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Newlines => "newlines",
      Self::Whole => "whole",
    }
  }
}

/// TTS synthesis config — the typed argument bundle [`tts_generate`]
/// consumes, the mlxrs analogue of mlx-audio `generate_audio`'s keyword
/// arguments and mlx-audio-swift's `AudioGenerateParameters`.
///
/// Owns no [`Array`] / borrowed data, so it is cheap to clone and a model's
/// [`TtsModel::default_config`] can hand one back by value. The per-model
/// sampling knobs (`temperature`, `top_p`, `repetition_penalty`, …) are a
/// flat subset of mlx-audio's `generate_audio` kwargs — the TTS driver does
/// not itself run a token sampler (per-model `synthesize_segment` owns the
/// decode loop), so unlike [`crate::audio::stt::generate::SttGenConfig`]
/// (which composes the full LM [`crate::lm::generate::GenConfig`]) this is a
/// plain knob bundle the per-model code reads.
///
/// All fields are private; construct via [`TtsGenConfig::new`] (all
/// defaults) or chain [`with_voice`](TtsGenConfig::with_voice),
/// [`with_language`](TtsGenConfig::with_language), etc. setters.
#[derive(Debug, Clone, PartialEq)]
pub struct TtsGenConfig {
  /// Voice / speaker id (mlx-audio `generate_audio` `voice`; also the
  /// speaker for multi-speaker models). Default [`DEFAULT_VOICE`].
  voice: String,
  /// Language / locale code (mlx-audio `lang_code`). Default
  /// [`DEFAULT_LANGUAGE`].
  language: String,
  /// Playback speed multiplier (mlx-audio `speed`); `1.0` is unmodified.
  /// `> 1.0` faster, `< 1.0` slower. Default `1.0`.
  speed: f32,
  /// Sampling temperature for autoregressive TTS backbones (mlx-audio
  /// `temperature`). Default [`DEFAULT_TEMPERATURE`]. Ignored by
  /// non-autoregressive models (kokoro-style duration predictors).
  temperature: f32,
  /// Nucleus (top-p) cutoff for autoregressive backbones (mlx-audio
  /// `top_p`); `0.0` ⇒ unused. Default `0.0`.
  top_p: f32,
  /// Top-k cutoff for autoregressive backbones (mlx-audio `top_k`); `0` ⇒
  /// unused. Default `0`.
  top_k: i32,
  /// Repetition penalty for autoregressive backbones (mlx-audio
  /// `repetition_penalty`); `None` ⇒ unused. Default `None`.
  repetition_penalty: Option<f32>,
  /// Per-segment token budget (mlx-audio `max_tokens`). Default
  /// [`DEFAULT_MAX_TOKENS`].
  max_tokens: usize,
  /// How [`tts_generate`] splits the input text. Default
  /// [`TextSegmentation::Newlines`] (the mlx-audio kokoro default).
  segmentation: TextSegmentation,
  /// Output container format the downstream writer should use (mlx-audio
  /// `audio_format`). Default [`AudioFormat::Wav`]. The driver yields raw
  /// PCM regardless; this is plumbed caller intent.
  audio_format: AudioFormat,
  /// Streaming-segment interval in seconds — the cadence a streaming
  /// per-model decoder yields partial chunks at (mlx-audio
  /// `streaming_interval`, fed into the per-model `streaming_token_interval`
  /// computation). Default [`DEFAULT_STREAMING_INTERVAL`]. The driver
  /// forwards it to per-model code via [`TtsSegment::streaming_interval`];
  /// it does not itself chunk a segment's audio.
  streaming_interval: f32,
}

impl Default for TtsGenConfig {
  fn default() -> Self {
    Self {
      voice: DEFAULT_VOICE.to_string(),
      language: DEFAULT_LANGUAGE.to_string(),
      speed: 1.0,
      temperature: DEFAULT_TEMPERATURE,
      top_p: 0.0,
      top_k: 0,
      repetition_penalty: None,
      max_tokens: DEFAULT_MAX_TOKENS,
      segmentation: TextSegmentation::Newlines,
      audio_format: AudioFormat::Wav,
      streaming_interval: DEFAULT_STREAMING_INTERVAL,
    }
  }
}

impl TtsGenConfig {
  /// Construct a [`TtsGenConfig`] with all defaults (equivalent to
  /// `TtsGenConfig::default()`).
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }

  // ── `with_*` builders ────────────────────────────────────────────────────

  /// Set the voice / speaker id (mlx-audio `voice`). Default
  /// [`DEFAULT_VOICE`].
  #[must_use]
  pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
    self.voice = voice.into();
    self
  }

  /// Set the language / locale code (mlx-audio `lang_code`). Default
  /// [`DEFAULT_LANGUAGE`].
  #[must_use]
  pub fn with_language(mut self, language: impl Into<String>) -> Self {
    self.language = language.into();
    self
  }

  /// Set the playback speed multiplier (mlx-audio `speed`). Default `1.0`.
  #[must_use]
  pub fn with_speed(mut self, speed: f32) -> Self {
    self.speed = speed;
    self
  }

  /// Set the sampling temperature (mlx-audio `temperature`). Default
  /// [`DEFAULT_TEMPERATURE`].
  #[must_use]
  pub fn with_temperature(mut self, temperature: f32) -> Self {
    self.temperature = temperature;
    self
  }

  /// Set the nucleus (top-p) cutoff (mlx-audio `top_p`). Default `0.0`.
  #[must_use]
  pub fn with_top_p(mut self, top_p: f32) -> Self {
    self.top_p = top_p;
    self
  }

  /// Set the top-k cutoff (mlx-audio `top_k`). Default `0`.
  #[must_use]
  pub fn with_top_k(mut self, top_k: i32) -> Self {
    self.top_k = top_k;
    self
  }

  /// Set the repetition penalty (mlx-audio `repetition_penalty`). Default
  /// `None`.
  #[must_use]
  pub fn with_repetition_penalty(mut self, repetition_penalty: Option<f32>) -> Self {
    self.repetition_penalty = repetition_penalty;
    self
  }

  /// Set the per-segment token budget (mlx-audio `max_tokens`). Default
  /// [`DEFAULT_MAX_TOKENS`].
  #[must_use]
  pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
    self.max_tokens = max_tokens;
    self
  }

  /// Set the text segmentation mode. Default [`TextSegmentation::Newlines`].
  #[must_use]
  pub fn with_segmentation(mut self, segmentation: TextSegmentation) -> Self {
    self.segmentation = segmentation;
    self
  }

  /// Set the output audio format (mlx-audio `audio_format`). Default
  /// [`AudioFormat::Wav`].
  #[must_use]
  pub fn with_audio_format(mut self, audio_format: AudioFormat) -> Self {
    self.audio_format = audio_format;
    self
  }

  /// Set the streaming-segment interval in seconds (mlx-audio
  /// `streaming_interval`). Default [`DEFAULT_STREAMING_INTERVAL`].
  #[must_use]
  pub fn with_streaming_interval(mut self, streaming_interval: f32) -> Self {
    self.streaming_interval = streaming_interval;
    self
  }

  // ── `#[inline(always)]` accessors ────────────────────────────────────────

  /// Voice / speaker id.
  #[inline(always)]
  #[must_use]
  pub fn voice(&self) -> &str {
    &self.voice
  }

  /// Language / locale code.
  #[inline(always)]
  #[must_use]
  pub fn language(&self) -> &str {
    &self.language
  }

  /// Speed multiplier.
  #[inline(always)]
  #[must_use]
  pub fn speed(&self) -> f32 {
    self.speed
  }

  /// Sampling temperature.
  #[inline(always)]
  #[must_use]
  pub fn temperature(&self) -> f32 {
    self.temperature
  }

  /// Nucleus (top-p) cutoff.
  #[inline(always)]
  #[must_use]
  pub fn top_p(&self) -> f32 {
    self.top_p
  }

  /// Top-k cutoff.
  #[inline(always)]
  #[must_use]
  pub fn top_k(&self) -> i32 {
    self.top_k
  }

  /// Repetition penalty (`None` ⇒ unused).
  #[inline(always)]
  #[must_use]
  pub fn repetition_penalty(&self) -> Option<f32> {
    self.repetition_penalty
  }

  /// Per-segment token budget.
  #[inline(always)]
  #[must_use]
  pub fn max_tokens(&self) -> usize {
    self.max_tokens
  }

  /// Text segmentation mode.
  #[inline(always)]
  #[must_use]
  pub fn segmentation(&self) -> TextSegmentation {
    self.segmentation
  }

  /// Output audio container format.
  #[inline(always)]
  #[must_use]
  pub fn audio_format(&self) -> AudioFormat {
    self.audio_format
  }

  /// Streaming-segment interval in seconds.
  #[inline(always)]
  #[must_use]
  pub fn streaming_interval(&self) -> f32 {
    self.streaming_interval
  }
}

/// Zero-shot voice-clone reference for a [`tts_generate_with_reference`] /
/// [`join_audio_with_reference`] run — a reference speaker the model should
/// clone the voice from.
///
/// The mlxrs analogue of mlx-audio's `ref_audio` / `ref_text` pair. It mirrors
/// mlx-audio-swift's [`SpeechGenerationModel.generate`][swift-gen] shape, where
/// `refAudio: MLXArray?` / `refText: String?` are a **separate argument** from
/// the per-generation `generationParameters` (== [`TtsGenConfig`]) — not fields
/// of it. mlxrs keeps the same separation: the reference is a distinct,
/// borrowed argument, so [`TtsGenConfig`] stays a cheap-to-clone,
/// `PartialEq` knob bundle that owns no [`Array`].
///
/// Borrows its `&Array` / `&str` from the caller (lifetime `'a`) — the driver
/// never clones the reference audio; it threads the borrow into every
/// [`TtsSegment`]. Both fields are independently `Option` (matching swift's two
/// optional parameters): a caller can supply audio without a transcript (the
/// per-model code transcribes it, like mlx-audio's STT fallback) or neither
/// (no cloning).
///
/// ## What mlxrs does and does not do with the reference
///
/// Like the rest of the TTS driver, mlxrs is a **passthrough**: the per-model
/// `synthesize_segment` consumes the reference (encodes the speaker, conditions
/// its backbone). mlxrs does **not** decode a reference *path* here — mirroring
/// mlx-audio-swift, [`TtsReference::ref_audio`] is an already-decoded
/// **rank-1 `f32` PCM `[samples]` [`Array`]** (the caller pre-loads it with
/// [`crate::audio::io::load_audio`] + [`Array::from_slice`], resampled to the
/// model's [`TtsModel::sample_rate`] if needed). mlx-audio's Python
/// `generate_audio` accepts a *path* and pre-decodes it with `load_audio`
/// before handing the array to `model.generate`; mlxrs leaves that one I/O step
/// to the caller (the audio surface's load/decode primitives are already
/// public) and keeps the driver pure — it touches no filesystem.
///
/// [swift-gen]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/Generation.swift
#[derive(Debug, Clone, Copy, Default)]
pub struct TtsReference<'a> {
  /// Reference-speaker waveform to clone the voice from (mlx-audio
  /// `generate_audio` `ref_audio`, swift `refAudio`). A rank-1 `f32` PCM
  /// `[samples]` [`Array`] at the model's [`TtsModel::sample_rate`]; `None`
  /// when not cloning. Threaded into every [`TtsSegment::ref_audio`].
  ref_audio: Option<&'a Array>,
  /// Transcript of [`TtsReference::ref_audio`] (mlx-audio `generate_audio`
  /// `ref_text`, swift `refText`). `None` when not cloning, or when the
  /// per-model code should transcribe the reference itself. Threaded into
  /// every [`TtsSegment::ref_text`].
  ref_text: Option<&'a str>,
}

impl<'a> TtsReference<'a> {
  /// Construct a [`TtsReference`] from an optional waveform and optional
  /// transcript. `TtsReference::default()` gives both-`None` (no cloning).
  #[must_use]
  pub const fn new(ref_audio: Option<&'a Array>, ref_text: Option<&'a str>) -> Self {
    Self {
      ref_audio,
      ref_text,
    }
  }

  /// Reference-speaker waveform (`None` ⇒ no cloning).
  #[inline(always)]
  #[must_use]
  pub fn ref_audio(&self) -> Option<&'a Array> {
    self.ref_audio
  }

  /// Transcript of the reference waveform (`None` ⇒ no cloning or
  /// per-model transcription).
  #[inline(always)]
  #[must_use]
  pub fn ref_text(&self) -> Option<&'a str> {
    self.ref_text
  }
}

/// One text segment plus the resolved synthesis knobs, handed to
/// [`TtsModel::synthesize_segment`].
///
/// The mlxrs analogue of the arguments mlx-audio's per-model `Model.generate`
/// receives for one `split_pattern`-split segment. Borrows its `&str` fields
/// from the [`tts_generate`] call's text + config (lifetime `'a`) — no
/// per-segment string allocation; the per-model code reads them and feeds
/// its own tokenizer / G2P.
///
/// The optional `ref_audio` / `ref_text` voice-clone pair carries the
/// per-run [`TtsReference`] (mlx-audio `generate_audio`'s `ref_audio` /
/// `ref_text`, zero-shot voice cloning) into each segment: a caller supplies
/// it via [`tts_generate_with_reference`], the driver threads the same borrow
/// onto **every** segment, and a model that supports cloning reads them in
/// `synthesize_segment` (a model that does not ignores them). They are
/// `Option` and the driver never inspects them — purely a per-model
/// passthrough.
#[derive(Debug, Clone, Copy)]
pub struct TtsSegment<'a> {
  /// The segment's raw text (a slice of the [`tts_generate`] input). The
  /// per-model code phonemizes / tokenizes this itself — the driver passes
  /// it through unchanged (no normalization, no G2P).
  text: &'a str,
  /// Voice / speaker id (from [`TtsGenConfig::voice`]).
  voice: &'a str,
  /// Language / locale code (from [`TtsGenConfig::language`]).
  language: &'a str,
  /// Speed multiplier (from [`TtsGenConfig::speed`]).
  speed: f32,
  /// Sampling temperature (from [`TtsGenConfig::temperature`]).
  temperature: f32,
  /// Top-p cutoff (from [`TtsGenConfig::top_p`]).
  top_p: f32,
  /// Top-k cutoff (from [`TtsGenConfig::top_k`]).
  top_k: i32,
  /// Repetition penalty (from [`TtsGenConfig::repetition_penalty`]).
  repetition_penalty: Option<f32>,
  /// Per-segment token budget (from [`TtsGenConfig::max_tokens`]).
  max_tokens: usize,
  /// Streaming-segment interval in seconds (from
  /// [`TtsGenConfig::streaming_interval`]) — per-model code that streams
  /// partial chunks derives its `streaming_token_interval` from this.
  streaming_interval: f32,
  /// Zero-based index of this segment in the input (mlx-audio's
  /// `segment_idx`). Stamped onto the produced [`AudioChunk::segment_idx`].
  segment_idx: usize,
  /// Optional reference-audio waveform for zero-shot voice cloning
  /// (mlx-audio `generate_audio` `ref_audio`), from the run's
  /// [`TtsReference::ref_audio`]. A rank-1 `f32` PCM `[samples]` tensor;
  /// `None` when not cloning. Per-model passthrough — the driver never
  /// inspects it.
  ref_audio: Option<&'a Array>,
  /// Optional transcript of [`TtsSegment::ref_audio`] (mlx-audio
  /// `generate_audio` `ref_text`), from the run's [`TtsReference::ref_text`].
  /// `None` when not cloning. Per-model passthrough.
  ref_text: Option<&'a str>,
}

impl<'a> TtsSegment<'a> {
  /// Construct a [`TtsSegment`] from all its fields.
  ///
  /// Per-model code consuming a segment uses the field accessors below;
  /// the constructor is the single place the driver assembles a segment.
  #[allow(clippy::too_many_arguments)]
  #[must_use]
  pub const fn new(
    text: &'a str,
    voice: &'a str,
    language: &'a str,
    speed: f32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    repetition_penalty: Option<f32>,
    max_tokens: usize,
    streaming_interval: f32,
    segment_idx: usize,
    ref_audio: Option<&'a Array>,
    ref_text: Option<&'a str>,
  ) -> Self {
    Self {
      text,
      voice,
      language,
      speed,
      temperature,
      top_p,
      top_k,
      repetition_penalty,
      max_tokens,
      streaming_interval,
      segment_idx,
      ref_audio,
      ref_text,
    }
  }

  // ── `#[inline(always)]` accessors ────────────────────────────────────────

  /// The segment's raw text slice.
  #[inline(always)]
  #[must_use]
  pub fn text(&self) -> &'a str {
    self.text
  }

  /// Voice / speaker id.
  #[inline(always)]
  #[must_use]
  pub fn voice(&self) -> &'a str {
    self.voice
  }

  /// Language / locale code.
  #[inline(always)]
  #[must_use]
  pub fn language(&self) -> &'a str {
    self.language
  }

  /// Speed multiplier.
  #[inline(always)]
  #[must_use]
  pub fn speed(&self) -> f32 {
    self.speed
  }

  /// Sampling temperature.
  #[inline(always)]
  #[must_use]
  pub fn temperature(&self) -> f32 {
    self.temperature
  }

  /// Nucleus (top-p) cutoff.
  #[inline(always)]
  #[must_use]
  pub fn top_p(&self) -> f32 {
    self.top_p
  }

  /// Top-k cutoff.
  #[inline(always)]
  #[must_use]
  pub fn top_k(&self) -> i32 {
    self.top_k
  }

  /// Repetition penalty (`None` ⇒ unused).
  #[inline(always)]
  #[must_use]
  pub fn repetition_penalty(&self) -> Option<f32> {
    self.repetition_penalty
  }

  /// Per-segment token budget.
  #[inline(always)]
  #[must_use]
  pub fn max_tokens(&self) -> usize {
    self.max_tokens
  }

  /// Streaming-segment interval in seconds.
  #[inline(always)]
  #[must_use]
  pub fn streaming_interval(&self) -> f32 {
    self.streaming_interval
  }

  /// Zero-based segment index.
  #[inline(always)]
  #[must_use]
  pub fn segment_idx(&self) -> usize {
    self.segment_idx
  }

  /// Reference-audio waveform (`None` ⇒ no cloning).
  #[inline(always)]
  #[must_use]
  pub fn ref_audio(&self) -> Option<&'a Array> {
    self.ref_audio
  }

  /// Reference transcript (`None` ⇒ no cloning or per-model transcription).
  #[inline(always)]
  #[must_use]
  pub fn ref_text(&self) -> Option<&'a str> {
    self.ref_text
  }
}

/// One unit of synthesized audio — the streaming-chunk type
/// [`tts_generate`]'s iterator yields.
///
/// Ports the shape of mlx-audio's `GenerationResult`
/// ([`tts/models/base.py`][tts-base]) — the audio tensor plus the
/// `segment_idx` / `sample_rate` / `is_streaming_chunk` / `is_final_chunk`
/// envelope — pruned to the fields the *driver* populates. The heavy
/// per-run telemetry mlx-audio's `GenerationResult` also carries
/// (`real_time_factor`, `processing_time_seconds`, `peak_memory_usage`,
/// the `prompt` / `audio_samples` tokens-per-sec dicts) is generation
/// instrumentation, not synthesis output — left to the caller (mlxrs's
/// audio surface ships no timing/memory telemetry; mirrors how the STT loop
/// yields a bare [`crate::lm::generate::GenStep`], not mlx-audio's
/// `STTOutput` telemetry bundle).
///
/// `audio` is a **rank-1** `[samples]` `f32` PCM tensor in `[-1, 1]` —
/// kept lazy (no implicit eval); [`AudioChunk::samples`] is the caller's
/// explicit materialization step.
///
/// [tts-base]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/base.py
///
/// `Debug` mirrors [`crate::lm::generate::GenStep`] (which likewise derives
/// `Debug` while holding an [`Array`]): [`Array`]'s `Debug` impl prints only
/// `shape` + `dtype` and never evals (mlxrs's no-implicit-eval rule), so
/// debug-printing an [`AudioChunk`] does not materialize its PCM.
#[derive(Debug)]
pub struct AudioChunk {
  /// The chunk's audio: a rank-1 `[samples]` `f32` PCM tensor in `[-1, 1]`
  /// at [`AudioChunk::sample_rate`] (mlx-audio `GenerationResult.audio`).
  audio: Array,
  /// PCM sample rate in Hz (mlx-audio `GenerationResult.sample_rate`) —
  /// the producing model's [`TtsModel::sample_rate`].
  sample_rate: u32,
  /// Zero-based index of the text segment this chunk belongs to (mlx-audio
  /// `GenerationResult.segment_idx`). Multiple chunks can share a
  /// `segment_idx` when a per-model decoder streams partial audio.
  segment_idx: usize,
  /// `true` if this is a partial streaming chunk (mlx-audio
  /// `GenerationResult.is_streaming_chunk`). The driver itself yields one
  /// whole-segment chunk per segment (`false`); the flag is on the type so
  /// a per-model streaming decoder's chunks round-trip through the same
  /// [`AudioChunk`].
  is_streaming_chunk: bool,
  /// `true` if this is the final chunk of the whole synthesis run
  /// (mlx-audio `GenerationResult.is_final_chunk`). Set by [`tts_generate`]
  /// on the last chunk of the last segment.
  is_final_chunk: bool,
}

impl AudioChunk {
  /// Construct an [`AudioChunk`] from all its fields.
  ///
  /// Per-model streaming decoders that produce their own chunk envelopes use
  /// this constructor; the non-streaming [`tts_generate`] driver uses it
  /// internally for each synthesized segment.
  #[must_use]
  pub fn new(
    audio: Array,
    sample_rate: u32,
    segment_idx: usize,
    is_streaming_chunk: bool,
    is_final_chunk: bool,
  ) -> Self {
    Self {
      audio,
      sample_rate,
      segment_idx,
      is_streaming_chunk,
      is_final_chunk,
    }
  }

  /// Borrow the chunk's audio tensor without materializing it.
  ///
  /// A `&self` no-eval shape/dtype inspection accessor — callers that need
  /// to inspect the shape or dtype without a full PCM decode use this.
  /// [`AudioChunk::samples`] is the `&mut self` explicit-eval step.
  #[inline(always)]
  #[must_use]
  pub fn audio_ref(&self) -> &Array {
    &self.audio
  }

  /// PCM sample rate in Hz.
  #[inline(always)]
  #[must_use]
  pub fn sample_rate(&self) -> u32 {
    self.sample_rate
  }

  /// Zero-based segment index this chunk belongs to.
  #[inline(always)]
  #[must_use]
  pub fn segment_idx(&self) -> usize {
    self.segment_idx
  }

  /// `true` if this is a partial streaming chunk.
  #[inline(always)]
  #[must_use]
  pub fn is_streaming_chunk(&self) -> bool {
    self.is_streaming_chunk
  }

  /// `true` if this is the final chunk of the whole synthesis run.
  #[inline(always)]
  #[must_use]
  pub fn is_final_chunk(&self) -> bool {
    self.is_final_chunk
  }

  /// The chunk's audio sample count (`audio.shape()[0]`) — a `&self`,
  /// no-eval shape read.
  #[inline(always)]
  #[must_use]
  pub fn len_samples(&self) -> usize {
    // `audio` is rank-1 by the `tts_generate` post-condition (the driver
    // validates `synthesize_segment`'s output); `shape()[0]` is the sample
    // count. A defensive `0` for an unexpectedly-rank-0 tensor keeps this
    // accessor panic-free.
    self.audio.shape().first().copied().unwrap_or(0)
  }

  /// `true` if the chunk carries no audio samples.
  #[inline(always)]
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.len_samples() == 0
  }

  /// Chunk duration in seconds (`len_samples / sample_rate`).
  ///
  /// `f64` math then narrowed: the sample count can be large and
  /// `sample_rate` is `u32`; computing in `f64` avoids `f32` rounding of
  /// the division. Returns `0.0` for a zero / absent sample rate rather
  /// than a NaN/inf.
  #[inline(always)]
  #[must_use]
  pub fn duration_seconds(&self) -> f64 {
    if self.sample_rate == 0 {
      return 0.0;
    }
    self.len_samples() as f64 / f64::from(self.sample_rate)
  }

  /// Consume the chunk and return the inner audio [`Array`].
  ///
  /// Useful for callers that need to own the tensor (e.g.
  /// [`join_audio_with_reference`] collects each chunk's audio to
  /// concatenate them). No eval occurs; the array is moved out.
  #[must_use]
  pub fn into_audio(self) -> Array {
    self.audio
  }

  /// Materialize the chunk's audio into an owned `Vec<f32>` of PCM samples.
  ///
  /// This is the **explicit eval step** (`&mut self` — mlxrs's no-implicit-
  /// eval rule): every other [`AudioChunk`] accessor is `&self` and pure.
  /// A downstream writer ([`crate::audio::io::save_wav`]) calls this to get
  /// the raw buffer.
  pub fn samples(&mut self) -> Result<Vec<f32>> {
    self.audio.to_vec::<f32>()
  }
}

/// Split `text` into segments per `mode`, dropping blank segments.
///
/// The regex-free port of mlx-audio's per-model `split_pattern` split
/// ([`TextSegmentation`] documents the correspondence). Returns
/// `(start, end)` UTF-8 byte ranges into `text` rather than owned `String`s
/// — [`tts_generate`] slices `&text[start..end]` for each [`TtsSegment`],
/// so no per-segment allocation.
///
/// A "blank" segment (empty or all-whitespace) is dropped, matching
/// mlx-audio's `[p for p in prompt_text.split(...) if p.strip()]`. An input
/// that is entirely blank yields an empty `Vec` — [`tts_generate`] turns
/// that into a recoverable [`Error::EmptyInput`] (it cannot synthesize
/// silence).
fn segment_ranges(text: &str, mode: TextSegmentation) -> Vec<(usize, usize)> {
  match mode {
    TextSegmentation::Whole => {
      if text.trim().is_empty() {
        Vec::new()
      } else {
        vec![(0, text.len())]
      }
    }
    TextSegmentation::Newlines => {
      let mut out = Vec::new();
      let mut seg_start: Option<usize> = None;
      // Walk the byte string; a maximal run of non-`\n` bytes is one
      // candidate segment, mirroring a `\n+` split (consecutive newlines
      // collapse — the empty pieces between them are blank and dropped).
      for (i, ch) in text.char_indices() {
        if ch == '\n' {
          if let Some(start) = seg_start.take() {
            push_if_nonblank(&mut out, text, start, i);
          }
        } else if seg_start.is_none() {
          seg_start = Some(i);
        }
      }
      if let Some(start) = seg_start {
        push_if_nonblank(&mut out, text, start, text.len());
      }
      out
    }
  }
}

/// Push `(start, end)` onto `out` iff `text[start..end]` is not all
/// whitespace — the `if p.strip()` blank-drop, factored out so both
/// [`segment_ranges`] arms share it.
fn push_if_nonblank(out: &mut Vec<(usize, usize)>, text: &str, start: usize, end: usize) {
  if !text[start..end].trim().is_empty() {
    out.push((start, end));
  }
}

/// The [`Iterator`] returned by [`tts_generate`]: borrows the model + the
/// input text + the config, owns the per-segment range list and a cursor,
/// and yields one [`AudioChunk`] per text segment.
///
/// Lifetime `'a` ties to all three borrows (model, text, config) — the
/// same borrow pattern [`crate::audio::stt::generate::SttGenerator`] uses
/// for the model. No per-segment [`String`] is allocated: each
/// [`TtsSegment`]'s `&str` fields are slices of the borrowed text/config.
///
/// The iterator **fuses**: after it yields `Err` (a segment's
/// `synthesize_segment` failed, or the model returned a malformed tensor)
/// or finishes (all segments produced) every further `next()` is `None` —
/// never a panic, never a re-entry into the model (the same `done`-flag
/// contract the STT / LM loops guarantee).
pub struct TtsGenerator<'a, M> {
  model: &'a M,
  /// The full input text — [`TtsSegment::text`] is sliced from this.
  text: &'a str,
  /// The synthesis config — voice / language / per-segment knobs are read
  /// from here for each [`TtsSegment`].
  cfg: &'a TtsGenConfig,
  /// The zero-shot voice-clone reference (mlx-audio `ref_audio` / `ref_text`).
  /// Threaded — the same borrow — onto every segment's
  /// [`TtsSegment::ref_audio`] / [`TtsSegment::ref_text`]. Both fields are
  /// `None` for a non-cloning run.
  reference: TtsReference<'a>,
  /// `(start, end)` byte ranges of every non-blank segment, computed once
  /// in the [`tts_generate`] constructor.
  segments: Vec<(usize, usize)>,
  /// Index of the next segment to synthesize (`0..segments.len()`).
  next_segment: usize,
  /// Fused: set after a yielded `Err` or after the last segment, so the
  /// iterator never re-enters the model.
  done: bool,
}

impl<M: TtsModel> TtsGenerator<'_, M> {
  /// Number of text segments this run will synthesize (one [`AudioChunk`]
  /// per segment) — a `&self` accessor, useful for progress reporting.
  #[must_use]
  pub fn segment_count(&self) -> usize {
    self.segments.len()
  }

  /// Synthesize the segment at `idx` into an [`AudioChunk`].
  ///
  /// Mirrors one iteration of mlx-audio's per-model `generate` loop body:
  /// build the per-segment argument bundle, call the model, validate the
  /// returned audio tensor shape, wrap it with the chunk envelope.
  fn synthesize(&self, idx: usize) -> Result<AudioChunk> {
    let (start, end) = self.segments[idx];
    let segment = TtsSegment::new(
      &self.text[start..end],
      self.cfg.voice(),
      self.cfg.language(),
      self.cfg.speed(),
      self.cfg.temperature(),
      self.cfg.top_p(),
      self.cfg.top_k(),
      self.cfg.repetition_penalty(),
      self.cfg.max_tokens(),
      self.cfg.streaming_interval(),
      idx,
      // Thread the run's voice-clone reference onto every segment (the same
      // borrow each time — no per-segment clone). `None`/`None` for a
      // non-cloning run.
      self.reference.ref_audio(),
      self.reference.ref_text(),
    );

    let audio = self.model.synthesize_segment(&segment)?;

    // Validate the model's audio output is a rank-1 `[samples]` tensor —
    // the documented `synthesize_segment` post-condition. A model returning
    // anything else (a `[1, samples]` un-squeezed tensor, a rank-0 scalar)
    // is a per-model defect; surface it as a recoverable `Err` here rather
    // than letting the malformed shape silently corrupt `join_audio`'s
    // concatenate or a downstream WAV writer. Mirrors the STT loop's
    // `decode_step` `[1, V]` shape check.
    let shape = audio.shape();
    if shape.len() != 1 {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        format_smolstr!("tts_generate: segment {idx}"),
        Error::RankMismatch(RankMismatchPayload::new(
          "tts_generate: `synthesize_segment` must return a rank-1 [samples] audio tensor",
          shape.len() as u32,
          shape,
        )),
      )));
    }

    // Validate the model's audio output is `f32` PCM — the other half of the
    // documented `synthesize_segment` / [`AudioChunk`] post-condition (rank-1
    // **f32** `[samples]` in `[-1, 1]`). A model returning a rank-1 tensor of
    // some other dtype (`i32` token ids it forgot to decode, an `f16`/`f64`
    // buffer) would pass the shape check and become a "successful"
    // [`AudioChunk`] whose invariant is false — `join_audio` could then return
    // a non-`f32` tensor, and [`AudioChunk::samples`] would only fail later
    // with an opaque `DtypeMismatch`. Surface the per-model defect here, at the
    // generator boundary, naming the actual dtype (the `expected`/`got` pair).
    let dtype = audio.dtype()?;
    if dtype != Dtype::F32 {
      return Err(Error::DtypeMismatch(DtypeMismatchPayload::new(
        Dtype::F32,
        dtype,
      )));
    }

    // `is_final_chunk` ⇔ this is the last segment. The driver yields one
    // whole-segment (non-streaming) chunk per segment, so `is_streaming_chunk`
    // is always `false` here — a per-model decoder that streams partial
    // audio sets that flag on its own `AudioChunk`s.
    Ok(AudioChunk::new(
      audio,
      self.model.sample_rate(),
      idx,
      false,
      idx + 1 == self.segments.len(),
    ))
  }
}

impl<M: TtsModel> Iterator for TtsGenerator<'_, M> {
  type Item = Result<AudioChunk>;

  fn next(&mut self) -> Option<Self::Item> {
    // Fused: a prior Err or exhausting the segments ends iteration
    // permanently — no panic, no re-entry into the model.
    if self.done {
      return None;
    }
    if self.next_segment >= self.segments.len() {
      self.done = true;
      return None;
    }

    let idx = self.next_segment;
    match self.synthesize(idx) {
      Ok(chunk) => {
        self.next_segment += 1;
        // The last segment's chunk is the final one — fuse after yielding
        // it (so the `is_final_chunk == true` chunk IS produced, then
        // iteration ends, matching the STT loop's "yield-then-fuse").
        if self.next_segment >= self.segments.len() {
          self.done = true;
        }
        Some(Ok(chunk))
      }
      Err(e) => {
        // A segment error is yielded once, then the iterator ends.
        self.done = true;
        Some(Err(e))
      }
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    // Exact upper bound: at most one chunk per not-yet-produced segment
    // (fewer if a segment errors). The lower bound is 0 because any
    // segment can fail.
    let remaining = self.segments.len().saturating_sub(self.next_segment);
    (0, Some(remaining))
  }
}

/// Start an end-to-end TTS synthesis run.
///
/// Pipeline (mlx-audio `generate_audio` / per-model `Model.generate`
/// shape):
/// 1. Reject over-[`MAX_TEXT_BYTES`] input up front (pre-allocation cap).
/// 2. Split `text` into segments per [`TtsGenConfig::segmentation`]
///    (mlx-audio's `split_pattern` split; blank segments dropped). An
///    all-blank input is a recoverable [`Error::EmptyInput`] — there is
///    nothing to synthesize.
/// 3. Return a [`TtsGenerator`] iterator; each [`Iterator::next`]
///    synthesizes one segment via [`TtsModel::synthesize_segment`] and
///    yields an [`AudioChunk`].
///
/// Returns an [`Iterator`]`<Item = Result<AudioChunk>>` — the streaming
/// analogue of mlx-audio's `for result in model.generate(...)` loop. The
/// final chunk has [`AudioChunk::is_final_chunk`] set; iteration ends after
/// it. Any segment error is yielded once as `Err`, after which the iterator
/// ends (no panic, no re-entry into the model — the same fused-iterator
/// contract the STT / LM loops guarantee).
///
/// The `'a` lifetime ties the returned iterator to the `model`, `text`, and
/// `cfg` borrows — no per-segment [`String`] allocation, the
/// [`TtsSegment`]s slice the borrowed data.
///
/// Note that this driver does **not** phonemize / normalize the text:
/// text preprocessing is model-specific (a model needing IPA input runs its
/// own G2P inside `synthesize_segment`, optionally via a
/// [`TextProcessor`](super::TextProcessor) hook). It also does not itself
/// run a token sampler — the per-model `synthesize_segment` owns the decode
/// loop and reads the sampling knobs off [`TtsSegment`].
pub fn tts_generate<'a, M: TtsModel>(
  model: &'a M,
  text: &'a str,
  cfg: &'a TtsGenConfig,
) -> Result<TtsGenerator<'a, M>> {
  // No voice-clone reference — the common, non-cloning path. Forwards a
  // both-`None` `TtsReference` to the threading constructor below.
  tts_generate_with_reference(model, text, cfg, TtsReference::default())
}

/// Start an end-to-end TTS synthesis run **with a zero-shot voice-clone
/// reference**.
///
/// Identical to [`tts_generate`] but threads `reference` (mlx-audio
/// `generate_audio`'s `ref_audio` / `ref_text`, swift's `refAudio` / `refText`)
/// onto **every** produced [`TtsSegment`], so a model that supports zero-shot
/// voice cloning ([`TtsReference`] documents the contract) receives the
/// reference speaker on each segment and clones its voice. A model that does
/// not support cloning ignores the reference fields.
///
/// `reference` is a separate, borrowed argument — not part of [`TtsGenConfig`]
/// — mirroring mlx-audio-swift's
/// [`SpeechGenerationModel.generate`][swift-gen] (`refAudio` / `refText` sit
/// beside `generationParameters`, not inside it). [`tts_generate`] is exactly
/// this with `reference = TtsReference::default()` (both fields `None`).
///
/// The `'a` lifetime now also ties the returned iterator to the `reference`
/// borrows; the driver clones nothing (it copies the two `Option<&_>` onto each
/// segment).
///
/// [swift-gen]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/Generation.swift
pub fn tts_generate_with_reference<'a, M: TtsModel>(
  model: &'a M,
  text: &'a str,
  cfg: &'a TtsGenConfig,
  reference: TtsReference<'a>,
) -> Result<TtsGenerator<'a, M>> {
  // 1. Pre-allocation cap — reject a crafted multi-MB text blob BEFORE the
  //    per-segment split + per-model allocations (the TTS analogue of the
  //    STT loop's `max_audio_seconds` up-front check). `text.len()` is the
  //    UTF-8 byte length.
  if text.len() > MAX_TEXT_BYTES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "tts_generate: input text size (split the request into smaller calls)",
      "MAX_TEXT_BYTES",
      MAX_TEXT_BYTES as u64,
      text.len() as u64,
    )));
  }

  // 2. Segment. `segment_ranges` drops blank segments; an all-blank input
  //    (empty string, only whitespace / newlines) yields no segments.
  let segments = segment_ranges(text, cfg.segmentation());
  if segments.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "tts_generate: input text has no non-blank segments (provide non-empty text)",
    )));
  }

  Ok(TtsGenerator {
    model,
    text,
    cfg,
    reference,
    segments,
    next_segment: 0,
    done: false,
  })
}

/// Synthesize `text` and concatenate every produced chunk into a single
/// `[total_samples]` audio [`Array`].
///
/// The mlxrs analogue of mlx-audio `generate_audio`'s `join_audio=True`
/// path — its `write_joined_audio` does
/// `mx.concatenate(audio_chunks, axis=0)`. Drives the [`tts_generate`]
/// iterator to completion, collects each chunk's `audio`, and joins them
/// with [`crate::ops::shape::concatenate`] along axis 0.
///
/// All chunks must share a sample rate (they do — every chunk is stamped
/// from the same [`TtsModel::sample_rate`]); the joined tensor is rank-1
/// `[total_samples]` `f32` PCM. A single-segment run returns that one
/// segment's audio without an extra concatenate (mlx-audio's
/// `len(audio_chunks) > 1` guard).
///
/// Propagates the first segment error (the iterator fuses on `Err`, so no
/// work continues after a failure). Because [`tts_generate`] rejects an
/// all-blank input, this never sees an empty chunk list.
pub fn join_audio<M: TtsModel>(model: &M, text: &str, cfg: &TtsGenConfig) -> Result<Array> {
  // No voice-clone reference — forwards a both-`None` `TtsReference`.
  join_audio_with_reference(model, text, cfg, TtsReference::default())
}

/// Synthesize `text` **with a zero-shot voice-clone reference** and
/// concatenate every produced chunk into a single `[total_samples]` audio
/// [`Array`].
///
/// Identical to [`join_audio`] but threads `reference` (mlx-audio
/// `ref_audio` / `ref_text`) onto every segment — the [`join_audio`] analogue
/// of [`tts_generate_with_reference`]. [`join_audio`] is exactly this with
/// `reference = TtsReference::default()`.
///
/// Every joined chunk is guaranteed `f32` PCM: the [`tts_generate`] driver
/// rejects a non-`f32` segment output at the generator boundary
/// ([`Error::DtypeMismatch`]), so this never returns a non-`f32` tensor — the
/// error propagates here instead.
pub fn join_audio_with_reference<M: TtsModel>(
  model: &M,
  text: &str,
  cfg: &TtsGenConfig,
  reference: TtsReference<'_>,
) -> Result<Array> {
  let mut chunks: Vec<Array> = Vec::new();
  for chunk in tts_generate_with_reference(model, text, cfg, reference)? {
    chunks.push(chunk?.into_audio());
  }

  // `tts_generate` guarantees at least one segment (it errors on an
  // all-blank input), so `chunks` is non-empty here. Mirror mlx-audio's
  // `len(audio_chunks) > 1` guard: a single chunk is returned as-is, no
  // pointless one-element concatenate.
  match chunks.len() {
    1 => Ok(chunks.into_iter().next().expect("len checked == 1")),
    _ => {
      let refs: Vec<&Array> = chunks.iter().collect();
      ops::shape::concatenate(&refs, 0)
    }
  }
}

#[cfg(test)]
mod tests {
  //! Unit tests for the **model-free** surface of the TTS generate
  //! pipeline: the private text-segmentation helpers
  //! ([`segment_ranges`] / [`push_if_nonblank`] — unreachable from the
  //! integration `tests/audio_tts.rs`, which can only drive them
  //! indirectly through [`tts_generate`]), the config / segment / chunk
  //! DTO builders + accessors, and the [`AudioFormat`] /
  //! [`TextSegmentation`] enum helpers ([`as_str`](AudioFormat::as_str),
  //! `Display`, the `derive_more` `IsVariant` `is_*` predicates). The
  //! model-driven flows (iteration, fusing, dtype/shape guards) live in
  //! `tests/audio_tts.rs` and are not duplicated here.
  use super::*;

  // ───────────────── segment_ranges: hand-derived byte ranges ─────────────────

  /// Resolve `segment_ranges` to the segment *strings* so a test can assert
  /// on the sliced content directly (the driver does `&text[start..end]`).
  fn segments_of(text: &str, mode: TextSegmentation) -> Vec<&str> {
    segment_ranges(text, mode)
      .into_iter()
      .map(|(s, e)| &text[s..e])
      .collect()
  }

  /// `Newlines`: a maximal run of non-`\n` bytes is one segment; the
  /// `(start, end)` ranges index exactly the line bytes (no newline
  /// included), with no per-segment allocation.
  #[test]
  fn segment_ranges_newlines_exact_byte_ranges() {
    let text = "first\nsecond\nthird";
    let ranges = segment_ranges(text, TextSegmentation::Newlines);
    // "first" = [0,5), "second" = [6,12), "third" = [13,18).
    assert_eq!(ranges, vec![(0, 5), (6, 12), (13, 18)]);
    assert_eq!(
      segments_of(text, TextSegmentation::Newlines),
      ["first", "second", "third"]
    );
  }

  /// Consecutive newlines collapse (the `\n+` semantics) and leading /
  /// trailing / whitespace-only segments are dropped — only the two
  /// non-blank runs survive, with byte ranges into the *original* string.
  #[test]
  fn segment_ranges_newlines_collapses_and_drops_blanks() {
    // leading \n, doubled \n, a whitespace-only line "   ", trailing \n\n.
    let text = "\nalpha\n\n   \nbeta\n\n";
    let ranges = segment_ranges(text, TextSegmentation::Newlines);
    // "alpha" starts at byte 1: [1,6). After "\n\n   \n" the next non-blank
    // run "beta" is at byte 12: [12,16). The "   " run is blank → dropped.
    assert_eq!(ranges, vec![(1, 6), (12, 16)]);
    assert_eq!(
      segments_of(text, TextSegmentation::Newlines),
      ["alpha", "beta"]
    );
  }

  /// Interior whitespace inside a non-blank line is preserved verbatim —
  /// only the *blank-drop* uses `trim`; the kept range is the whole line.
  #[test]
  fn segment_ranges_newlines_preserves_interior_whitespace() {
    let text = "  hello   world  \nx";
    let segs = segments_of(text, TextSegmentation::Newlines);
    // The line is non-blank (has 'hello'), so it's kept whole — leading and
    // interior and trailing spaces all intact (no trim applied to content).
    assert_eq!(segs, ["  hello   world  ", "x"]);
  }

  /// A lone `\r` is NOT a segment separator (only `\n` splits) — a CRLF
  /// line keeps its trailing `\r` inside the segment content.
  #[test]
  fn segment_ranges_newlines_does_not_split_on_carriage_return() {
    let text = "a\r\nb";
    let segs = segments_of(text, TextSegmentation::Newlines);
    // Split only on '\n': "a\r" and "b". The '\r' stays with the first line
    // (it is non-whitespace-significant content as far as the splitter cares,
    // though trim would strip it — but trim only gates the blank check, the
    // kept range is the full "a\r").
    assert_eq!(segs, ["a\r", "b"]);
  }

  /// Multibyte UTF-8: the `(start, end)` ranges are valid char boundaries,
  /// so slicing never panics and recovers the exact multibyte segment.
  #[test]
  fn segment_ranges_newlines_multibyte_utf8_boundaries() {
    // "héllo" (é = 2 bytes) then "wörld" (ö = 2 bytes).
    let text = "héllo\nwörld";
    let ranges = segment_ranges(text, TextSegmentation::Newlines);
    // "héllo" = 6 bytes [0,6); '\n' at 6; "wörld" = 6 bytes [7,13).
    assert_eq!(ranges, vec![(0, 6), (7, 13)]);
    assert_eq!(
      segments_of(text, TextSegmentation::Newlines),
      ["héllo", "wörld"]
    );
  }

  /// A single line with no newline is exactly one segment spanning the whole
  /// string.
  #[test]
  fn segment_ranges_newlines_single_line() {
    let text = "no newline here";
    assert_eq!(
      segment_ranges(text, TextSegmentation::Newlines),
      vec![(0, text.len())]
    );
  }

  /// An all-blank input (empty, whitespace, only newlines) yields an empty
  /// `Vec` under `Newlines` — the signal [`tts_generate`] turns into a
  /// recoverable error.
  #[test]
  fn segment_ranges_newlines_all_blank_is_empty() {
    assert!(segment_ranges("", TextSegmentation::Newlines).is_empty());
    assert!(segment_ranges("\n\n\n", TextSegmentation::Newlines).is_empty());
    assert!(
      segment_ranges("   \n \t \n  ", TextSegmentation::Newlines).is_empty(),
      "whitespace-only lines all dropped"
    );
  }

  /// `Whole`: the entire input is one `(0, len)` segment — embedded
  /// newlines and surrounding whitespace are part of that single span (no
  /// split, no trim of the content).
  #[test]
  fn segment_ranges_whole_is_single_full_span() {
    let text = "  line one\nline two  ";
    let ranges = segment_ranges(text, TextSegmentation::Whole);
    assert_eq!(ranges, vec![(0, text.len())]);
    // The single segment is the verbatim input, leading/trailing spaces and
    // the embedded newline included.
    assert_eq!(
      segments_of(text, TextSegmentation::Whole),
      ["  line one\nline two  "]
    );
  }

  /// `Whole` on an all-blank input is still empty (the only place `Whole`
  /// consults `trim`: a blank whole-input has nothing to synthesize).
  #[test]
  fn segment_ranges_whole_all_blank_is_empty() {
    assert!(segment_ranges("", TextSegmentation::Whole).is_empty());
    assert!(segment_ranges("   \n\t ", TextSegmentation::Whole).is_empty());
  }

  /// `Whole` keeps a single non-blank input even if it has leading/trailing
  /// whitespace (the blank check is on `trim`, the kept range is not
  /// trimmed).
  #[test]
  fn segment_ranges_whole_keeps_padded_nonblank() {
    let text = "   hi   ";
    assert_eq!(
      segment_ranges(text, TextSegmentation::Whole),
      vec![(0, text.len())]
    );
  }

  // ───────────────── push_if_nonblank: the blank-drop predicate ─────────────────

  /// `push_if_nonblank` appends the range iff `text[start..end]` is not all
  /// whitespace.
  #[test]
  fn push_if_nonblank_keeps_nonblank_drops_blank() {
    let text = "ab   cd";
    let mut out = Vec::new();
    // [0,2) = "ab" → kept.
    push_if_nonblank(&mut out, text, 0, 2);
    // [2,5) = "   " → blank, dropped.
    push_if_nonblank(&mut out, text, 2, 5);
    // [5,7) = "cd" → kept.
    push_if_nonblank(&mut out, text, 5, 7);
    assert_eq!(out, vec![(0, 2), (5, 7)]);
  }

  /// An empty range `[i, i)` is blank (trims to "") and is dropped.
  #[test]
  fn push_if_nonblank_drops_empty_range() {
    let text = "xyz";
    let mut out = Vec::new();
    push_if_nonblank(&mut out, text, 1, 1);
    assert!(out.is_empty());
  }

  // ───────────────── AudioFormat: as_str / Display / IsVariant ─────────────────

  /// `AudioFormat::as_str` is the lowercase mlx-audio `audio_format` string.
  #[test]
  fn audio_format_as_str_and_display() {
    assert_eq!(AudioFormat::Wav.as_str(), "wav");
    assert_eq!(AudioFormat::Flac.as_str(), "flac");
    // `#[display("{}", self.as_str())]` ⇒ Display == as_str.
    assert_eq!(AudioFormat::Wav.to_string(), "wav");
    assert_eq!(format!("{}", AudioFormat::Flac), "flac");
  }

  /// `derive_more::IsVariant` generates `is_wav` / `is_flac` predicates.
  #[test]
  fn audio_format_is_variant_predicates() {
    assert!(AudioFormat::Wav.is_wav());
    assert!(!AudioFormat::Wav.is_flac());
    assert!(AudioFormat::Flac.is_flac());
    assert!(!AudioFormat::Flac.is_wav());
  }

  /// `AudioFormat::default()` is `Wav` (mlx-audio's `audio_format="wav"`).
  #[test]
  fn audio_format_default_is_wav() {
    assert_eq!(AudioFormat::default(), AudioFormat::Wav);
    assert!(AudioFormat::default().is_wav());
  }

  // ────────────── TextSegmentation: as_str / Display / IsVariant ──────────────

  /// `TextSegmentation::as_str` is the lowercase mode name; Display mirrors
  /// it.
  #[test]
  fn text_segmentation_as_str_and_display() {
    assert_eq!(TextSegmentation::Newlines.as_str(), "newlines");
    assert_eq!(TextSegmentation::Whole.as_str(), "whole");
    assert_eq!(TextSegmentation::Newlines.to_string(), "newlines");
    assert_eq!(format!("{}", TextSegmentation::Whole), "whole");
  }

  /// `IsVariant` predicates for the segmentation mode.
  #[test]
  fn text_segmentation_is_variant_predicates() {
    assert!(TextSegmentation::Newlines.is_newlines());
    assert!(!TextSegmentation::Newlines.is_whole());
    assert!(TextSegmentation::Whole.is_whole());
    assert!(!TextSegmentation::Whole.is_newlines());
  }

  /// `TextSegmentation::default()` is `Newlines` (the mlx-audio kokoro
  /// `split_pattern=r"\n+"` default).
  #[test]
  fn text_segmentation_default_is_newlines() {
    assert_eq!(TextSegmentation::default(), TextSegmentation::Newlines);
    assert!(TextSegmentation::default().is_newlines());
  }

  // ───────────────── TtsGenConfig: defaults / builders / accessors ─────────────────

  /// `TtsGenConfig::new()` equals `TtsGenConfig::default()` and carries the
  /// documented mlx-audio `generate_audio` defaults across every field.
  #[test]
  fn tts_gen_config_new_equals_default_and_carries_defaults() {
    let c = TtsGenConfig::new();
    assert_eq!(c, TtsGenConfig::default());
    assert_eq!(c.voice(), DEFAULT_VOICE);
    assert_eq!(c.language(), DEFAULT_LANGUAGE);
    assert!((c.speed() - 1.0).abs() < 1e-6);
    assert!((c.temperature() - DEFAULT_TEMPERATURE).abs() < 1e-6);
    assert!((c.top_p() - 0.0).abs() < 1e-6);
    assert_eq!(c.top_k(), 0);
    assert_eq!(c.repetition_penalty(), None);
    assert_eq!(c.max_tokens(), DEFAULT_MAX_TOKENS);
    assert_eq!(c.segmentation(), TextSegmentation::Newlines);
    assert_eq!(c.audio_format(), AudioFormat::Wav);
    assert!((c.streaming_interval() - DEFAULT_STREAMING_INTERVAL).abs() < 1e-6);
  }

  /// Every `with_*` builder sets exactly its field and the matching accessor
  /// reads it back — covering the sampling knobs (`top_p`, `top_k`,
  /// `repetition_penalty`, `max_tokens`, `audio_format`,
  /// `streaming_interval`) the integration tests do not plumb.
  #[test]
  fn tts_gen_config_builders_round_trip_all_fields() {
    let c = TtsGenConfig::new()
      .with_voice("bf_emma")
      .with_language("en-gb")
      .with_speed(1.25)
      .with_temperature(0.4)
      .with_top_p(0.9)
      .with_top_k(40)
      .with_repetition_penalty(Some(1.1))
      .with_max_tokens(256)
      .with_segmentation(TextSegmentation::Whole)
      .with_audio_format(AudioFormat::Flac)
      .with_streaming_interval(3.5);
    assert_eq!(c.voice(), "bf_emma");
    assert_eq!(c.language(), "en-gb");
    assert!((c.speed() - 1.25).abs() < 1e-6);
    assert!((c.temperature() - 0.4).abs() < 1e-6);
    assert!((c.top_p() - 0.9).abs() < 1e-6);
    assert_eq!(c.top_k(), 40);
    assert_eq!(c.repetition_penalty(), Some(1.1));
    assert_eq!(c.max_tokens(), 256);
    assert!(c.segmentation().is_whole());
    assert!(c.audio_format().is_flac());
    assert!((c.streaming_interval() - 3.5).abs() < 1e-6);
  }

  /// A `with_*` builder mutates only its own field (the others keep their
  /// defaults) — guards against a copy/paste setter writing the wrong field.
  #[test]
  fn tts_gen_config_builder_is_field_isolated() {
    let base = TtsGenConfig::default();
    let only_topk = TtsGenConfig::default().with_top_k(7);
    assert_eq!(only_topk.top_k(), 7);
    // Nothing else moved.
    assert_eq!(only_topk.voice(), base.voice());
    assert_eq!(only_topk.language(), base.language());
    assert!((only_topk.speed() - base.speed()).abs() < 1e-6);
    assert!((only_topk.temperature() - base.temperature()).abs() < 1e-6);
    assert!((only_topk.top_p() - base.top_p()).abs() < 1e-6);
    assert_eq!(only_topk.repetition_penalty(), base.repetition_penalty());
    assert_eq!(only_topk.max_tokens(), base.max_tokens());
    assert_eq!(only_topk.segmentation(), base.segmentation());
    assert_eq!(only_topk.audio_format(), base.audio_format());
    assert!((only_topk.streaming_interval() - base.streaming_interval()).abs() < 1e-6);
  }

  /// `with_repetition_penalty(None)` clears a previously-set penalty (the
  /// field is `Option<f32>`, so `None` is a meaningful reset).
  #[test]
  fn tts_gen_config_repetition_penalty_can_be_cleared() {
    let c = TtsGenConfig::new()
      .with_repetition_penalty(Some(1.3))
      .with_repetition_penalty(None);
    assert_eq!(c.repetition_penalty(), None);
  }

  /// `TtsGenConfig` is `Clone` + `PartialEq`: a clone equals its source, and
  /// changing one field makes them unequal (the type owns no `Array`, so it
  /// is cheap to clone and value-comparable — the `default_config` contract).
  #[test]
  fn tts_gen_config_clone_and_partial_eq() {
    let a = TtsGenConfig::new().with_voice("v").with_top_k(3);
    let b = a.clone();
    assert_eq!(a, b);
    let c = a.clone().with_top_k(4);
    assert_ne!(a, c, "differing top_k ⇒ unequal");
  }

  // ───────────────── TtsReference: accessors + default ─────────────────

  /// `TtsReference::new` stores both optional fields and the accessors read
  /// them back.
  #[test]
  fn tts_reference_new_accessors() {
    let wav = Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3], &[3]).unwrap();
    let r = TtsReference::new(Some(&wav), Some("caption"));
    assert!(r.ref_audio().is_some());
    assert_eq!(r.ref_text(), Some("caption"));
    // The borrowed audio is the same [3] array.
    assert_eq!(r.ref_audio().unwrap().shape(), vec![3]);
  }

  /// `TtsReference::default()` is both-`None` (a non-cloning reference) — the
  /// value [`tts_generate`] forwards on the plain (no-reference) path.
  #[test]
  fn tts_reference_default_is_both_none() {
    let r = TtsReference::default();
    assert!(r.ref_audio().is_none());
    assert!(r.ref_text().is_none());
  }

  /// The two fields are independently optional: audio-only (`Some`/`None`)
  /// and text-only (`None`/`Some`) references are both representable.
  #[test]
  fn tts_reference_fields_are_independent() {
    let wav = Array::from_slice::<f32>(&[0.0_f32], &[1]).unwrap();
    let audio_only = TtsReference::new(Some(&wav), None);
    assert!(audio_only.ref_audio().is_some() && audio_only.ref_text().is_none());
    let text_only = TtsReference::new(None, Some("t"));
    assert!(text_only.ref_audio().is_none() && text_only.ref_text() == Some("t"));
  }

  // ───────────────── TtsSegment: direct construction + accessors ─────────────────

  /// `TtsSegment::new` stores all 13 fields and every accessor reads its own
  /// back — including the sampling knobs (`top_p`, `top_k`,
  /// `repetition_penalty`, `max_tokens`, `streaming_interval`) the
  /// integration mock does not record.
  #[test]
  fn tts_segment_new_all_accessors() {
    let wav = Array::from_slice::<f32>(&[0.5_f32, -0.5], &[2]).unwrap();
    let seg = TtsSegment::new(
      "the text",
      "af_heart",
      "en",
      1.5,
      0.6,
      0.85,
      50,
      Some(1.2),
      900,
      2.5,
      4,
      Some(&wav),
      Some("ref transcript"),
    );
    assert_eq!(seg.text(), "the text");
    assert_eq!(seg.voice(), "af_heart");
    assert_eq!(seg.language(), "en");
    assert!((seg.speed() - 1.5).abs() < 1e-6);
    assert!((seg.temperature() - 0.6).abs() < 1e-6);
    assert!((seg.top_p() - 0.85).abs() < 1e-6);
    assert_eq!(seg.top_k(), 50);
    assert_eq!(seg.repetition_penalty(), Some(1.2));
    assert_eq!(seg.max_tokens(), 900);
    assert!((seg.streaming_interval() - 2.5).abs() < 1e-6);
    assert_eq!(seg.segment_idx(), 4);
    assert!(seg.ref_audio().is_some());
    assert_eq!(seg.ref_text(), Some("ref transcript"));
  }

  /// A `TtsSegment` with no reference (`None`/`None`) — the non-cloning
  /// shape the driver builds on the plain `tts_generate` path.
  #[test]
  fn tts_segment_without_reference() {
    let seg = TtsSegment::new(
      "x", "v", "en", 1.0, 0.7, 0.0, 0, None, 1200, 2.0, 0, None, None,
    );
    assert!(seg.ref_audio().is_none());
    assert!(seg.ref_text().is_none());
    assert_eq!(seg.repetition_penalty(), None);
    assert_eq!(seg.segment_idx(), 0);
  }

  // ───────────────── AudioChunk: construction / accessors / math ─────────────────

  /// `AudioChunk::new` stores its envelope and every `&self` accessor reads
  /// it back without materializing the audio (`audio_ref` is a no-eval
  /// borrow; `len_samples` is a shape read).
  #[test]
  fn audio_chunk_new_accessors_no_eval() {
    let audio = Array::from_slice::<f32>(&[0.0_f32, 0.1, 0.2, 0.3], &[4]).unwrap();
    let chunk = AudioChunk::new(audio, 16_000, 2, true, false);
    assert_eq!(chunk.sample_rate(), 16_000);
    assert_eq!(chunk.segment_idx(), 2);
    assert!(chunk.is_streaming_chunk());
    assert!(!chunk.is_final_chunk());
    assert_eq!(chunk.len_samples(), 4);
    assert!(!chunk.is_empty());
    assert_eq!(chunk.audio_ref().shape(), vec![4], "no-eval shape read");
  }

  /// `duration_seconds` is `len_samples / sample_rate` in `f64`.
  #[test]
  fn audio_chunk_duration_seconds_is_samples_over_rate() {
    let audio = Array::from_slice::<f32>(&[0.0_f32; 24_000], &[24_000]).unwrap();
    let chunk = AudioChunk::new(audio, 24_000, 0, false, true);
    assert!((chunk.duration_seconds() - 1.0).abs() < 1e-12);
  }

  /// `duration_seconds` guards a zero sample rate: it returns `0.0`, not a
  /// NaN / inf division.
  #[test]
  fn audio_chunk_duration_seconds_zero_rate_is_zero() {
    let audio = Array::from_slice::<f32>(&[0.0_f32, 0.1], &[2]).unwrap();
    let chunk = AudioChunk::new(audio, 0, 0, false, true);
    assert_eq!(chunk.duration_seconds(), 0.0);
    assert!(
      chunk.duration_seconds().is_finite(),
      "no NaN/inf for rate 0"
    );
  }

  /// A zero-length `[0]` waveform is a valid empty chunk: `is_empty()`,
  /// `len_samples() == 0`, `duration_seconds() == 0.0`.
  #[test]
  fn audio_chunk_empty_waveform() {
    let audio = Array::from_slice::<f32>(&[], &[0]).unwrap();
    let chunk = AudioChunk::new(audio, 24_000, 0, false, true);
    assert!(chunk.is_empty());
    assert_eq!(chunk.len_samples(), 0);
    assert_eq!(chunk.duration_seconds(), 0.0);
  }

  /// `into_audio` moves the inner tensor out without an eval (the shape is
  /// preserved); `samples` is the explicit `&mut` materialization step.
  #[test]
  fn audio_chunk_into_audio_and_samples() {
    let audio = Array::from_slice::<f32>(&[0.0_f32, 0.25, 0.5], &[3]).unwrap();
    let mut chunk = AudioChunk::new(audio, 24_000, 0, false, true);
    let pcm = chunk.samples().unwrap();
    assert_eq!(pcm, vec![0.0, 0.25, 0.5]);
    // `into_audio` hands back the [3] tensor.
    let moved = AudioChunk::new(
      Array::from_slice::<f32>(&[1.0_f32, 2.0], &[2]).unwrap(),
      24_000,
      0,
      false,
      true,
    )
    .into_audio();
    assert_eq!(moved.shape(), vec![2]);
  }

  // ───────────────── module constants ─────────────────

  /// The documented mlx-audio default constants have the expected values.
  #[test]
  fn default_constants_match_mlx_audio() {
    assert_eq!(DEFAULT_VOICE, "af_heart");
    assert_eq!(DEFAULT_LANGUAGE, "en");
    assert!((DEFAULT_TEMPERATURE - 0.7).abs() < 1e-6);
    assert_eq!(DEFAULT_MAX_TOKENS, 1200);
    assert!((DEFAULT_STREAMING_INTERVAL - 2.0).abs() < 1e-6);
    assert_eq!(MAX_TEXT_BYTES, 1024 * 1024);
  }
}
