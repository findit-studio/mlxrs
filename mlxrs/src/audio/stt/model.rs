//! The STT trait architecture for `mlxrs::audio::stt`.
//!
//! Three layers separate the **universal contract** every speech-to-text
//! model satisfies from the **family-specific decode procedure**, with
//! associated types carrying what varies per model (the decode cache, the
//! encoder features):
//!
//! - [`Transcribe`] — the universal "audio in, text out" contract. CLIs and
//!   pipelines depend only on this; every STT model implements it (directly
//!   or via one of the family drivers in [`super::generate`]).
//! - [`CtcModel`] — the non-autoregressive CTC family (wav2vec2 and future
//!   CTC architectures): one encoder forward producing per-frame logits, then
//!   a greedy blank-collapse. Gets [`Transcribe`] for free via the blanket
//!   `impl<M: CtcModel> Transcribe for M` in [`super::generate`].
//! - [`AutoregressiveStt`] — the encoder/decoder family (Whisper and future
//!   attention STT). Its associated [`AutoregressiveStt::Cache`] type carries
//!   the per-model, caller-held decode state, so the cache is a value owned by
//!   each [`Transcribe::transcribe`] call — no model-stored `RefCell`, no
//!   cross-utterance or concurrent sharing.
//!
//! Per the project's no-per-model-arch rule, mlxrs ships **no** concrete STT
//! model implementations: those (the conv subsampling + transformer for
//! whisper, the conformer for parakeet, etc.) live in user code on top of
//! these traits. This module is the shared contract every per-model decoder
//! conforms to.

use crate::{array::Array, audio::dsp, error::Result};

/// The universal speech-to-text contract: audio waveform in, text out.
///
/// Every STT model implements this — CTC models receive it from the blanket
/// `impl<M: CtcModel> Transcribe for M` in [`super::generate`]; autoregressive
/// models implement it directly (a simple one forwarding to
/// [`super::generate::greedy_transcribe`], a complex one — e.g. Whisper —
/// running its own decoding procedure that still reuses its
/// [`AutoregressiveStt`] hooks internally).
///
/// `audio` is a mono waveform [`Array`] (the [`crate::audio::io::load_audio`]
/// output, resampled to the model's expected rate). The model's frontend
/// ([`AutoregressiveStt::log_mel`] / [`CtcModel::logits`]) converts it to the
/// features its encoder consumes.
pub trait Transcribe {
  /// Transcribe `audio` into text under `opts`.
  fn transcribe(&self, audio: &Array, opts: &TranscribeOptions) -> Result<Transcription>;
}

/// The CTC family: non-autoregressive models that emit per-frame logits in a
/// single encoder forward, then collapse them greedily.
///
/// A CTC model supplies the three pieces the greedy-collapse driver (the
/// blanket `impl<M: CtcModel> Transcribe for M` in [`super::generate`]) needs:
/// the per-frame logits, the blank id to collapse against, and the
/// vocabulary map from collapsed ids to text. No CTC model needs to override
/// the driver, so the blanket impl is safe (no coherence conflict).
pub trait CtcModel {
  /// Per-frame logits of shape `(T', vocab)` — one row of class scores per
  /// encoder time frame — for the mono `waveform`.
  fn logits(&self, waveform: &Array) -> Result<Array>;

  /// The CTC blank class id (collapsed out of the greedy decode).
  fn blank_id(&self) -> u32;

  /// Map a collapsed id sequence to text via the model's vocabulary. Called
  /// once by the driver after blank-collapse + run-length dedup.
  fn decode_ids(&self, ids: &[u32]) -> String;
}

/// The autoregressive encoder/decoder family: an audio encoder feeding a
/// token-by-token decoder (Whisper and future attention STT).
///
/// The associated [`Self::Cache`] type is the key lever: it carries the
/// per-model decode state (Whisper's per-block self- and cross-attention KV
/// cache, an RNN-T predictor state, …) as a value the **caller** owns. Each
/// [`Transcribe::transcribe`] call mints a fresh cache via [`Self::new_cache`],
/// so decode state is never shared across utterances or threads by
/// construction — there is no model-stored cache to alias.
///
/// There is intentionally **no** blanket `impl<M: AutoregressiveStt>
/// Transcribe`: such a blanket would overlap-conflict with a model's own
/// `impl Transcribe` (e.g. Whisper's), which Rust coherence forbids without
/// specialization. Instead [`super::generate::greedy_transcribe`] is a free
/// function a model calls from inside its own [`Transcribe`] impl.
pub trait AutoregressiveStt {
  /// The per-model, caller-held decode state. Minted fresh per generation by
  /// [`Self::new_cache`] and threaded through [`Self::decode_step`] by
  /// `&mut`, so it is never shared across generations or threads.
  type Cache;

  /// The model's frontend: convert a mono waveform `audio` [`Array`] into the
  /// log-mel features its [`Self::encode`] consumes.
  ///
  /// The default delegates to [`super::generate::default_log_mel`] using this
  /// model's [`Self::mel_config`] — the standard Whisper-style log-mel front
  /// end (Slaney filterbank, the configured [`crate::audio::dsp::LogFloor`]),
  /// assuming `audio` is already at the config's
  /// [`MelConfig::sample_rate`]. A model whose frontend differs (a learned
  /// feature extractor, a non-mel front end) overrides this.
  fn log_mel(&self, audio: &Array) -> Result<Array> {
    super::generate::default_log_mel(&self.mel_config(), audio)
  }

  /// Encode the log-mel features (the [`Self::log_mel`] output) into the
  /// encoder hidden states the decoder cross-attends. Runs once per
  /// utterance; the result is reused across every [`Self::decode_step`].
  fn encode(&self, mel: &Array) -> Result<Array>;

  /// Mint a fresh, owned decode cache for one generation.
  fn new_cache(&self) -> Self::Cache;

  /// One decode step: given the running `cache`, the encoder states `enc`, and
  /// the full token sequence decoded so far, return the next-token logits of
  /// shape `(vocab,)`.
  ///
  /// `tokens` is the complete prefix (the [`Self::initial_tokens`] prompt
  /// followed by every token decoded since); the model decides how to use the
  /// `cache` to avoid recomputing the prefix. The greedy driver
  /// ([`super::generate::greedy_transcribe`]) reads `argmax` over the returned
  /// `(vocab,)` row.
  fn decode_step(&self, cache: &mut Self::Cache, enc: &Array, tokens: &[u32]) -> Result<Array>;

  /// The full prompt prefix the decode loop seeds with (Whisper's
  /// start-of-transcript + language + task + timestamp token sequence),
  /// derived from `opts`. The first decoded token follows this prefix.
  fn initial_tokens(&self, opts: &TranscribeOptions) -> Result<Vec<u32>>;

  /// The end-of-transcript token id the decode loop stops on.
  fn eot(&self) -> u32;

  /// The mel-spectrogram extraction config this model's [`Self::log_mel`]
  /// default uses.
  ///
  /// The default is the Whisper preset ([`MelConfig::whisper_default`]). A
  /// model with a different feature-extractor config (different `n_mels`,
  /// `sample_rate`, or [`crate::audio::dsp::LogFloor`]) overrides this; a
  /// model that overrides [`Self::log_mel`] wholesale need not.
  fn mel_config(&self) -> MelConfig {
    MelConfig::whisper_default()
  }
}

/// The transcription task: produce text in the source language
/// ([`Task::Transcribe`]) or translate it to English ([`Task::Translate`]).
///
/// Mirrors Whisper's `transcribe` / `translate` task tokens; CTC models
/// ignore it (they have no translation path).
#[derive(
  Debug, Clone, Copy, PartialEq, Eq, Hash, Default, derive_more::Display, derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
pub enum Task {
  /// Transcribe the audio in its source language (the default).
  #[default]
  Transcribe,
  /// Translate the audio to English.
  Translate,
}

impl Task {
  /// The task's stable string name (Whisper's task slug).
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Transcribe => "transcribe",
      Self::Translate => "translate",
    }
  }
}

/// Options controlling a [`Transcribe::transcribe`] run.
///
/// Autoregressive models map these onto their decoding parameters (Whisper
/// onto its start-of-transcript token sequence + sampler temperature); CTC
/// models ignore most of them (they have no language conditioning, no task,
/// and a fixed greedy decode).
#[derive(Debug, Clone, PartialEq)]
pub struct TranscribeOptions {
  /// The spoken language as an ISO code, or `None` to auto-detect (Whisper
  /// runs language identification when this is `None`).
  language: Option<String>,
  /// Transcribe vs translate.
  task: Task,
  /// Sampling temperature; `0.0` is deterministic greedy decoding.
  temperature: f32,
  /// Suppress timestamp tokens / segment time spans when `true`.
  no_timestamps: bool,
}

impl TranscribeOptions {
  /// A new options bundle with auto-detect language, [`Task::Transcribe`],
  /// greedy (`temperature == 0.0`) decoding, and timestamps enabled.
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      language: None,
      task: Task::Transcribe,
      temperature: 0.0,
      no_timestamps: false,
    }
  }

  /// The configured language code, or `None` to auto-detect.
  #[inline(always)]
  pub fn language(&self) -> Option<&str> {
    self.language.as_deref()
  }

  /// The transcription task.
  #[inline(always)]
  pub const fn task(&self) -> Task {
    self.task
  }

  /// The sampling temperature.
  #[inline(always)]
  pub const fn temperature(&self) -> f32 {
    self.temperature
  }

  /// Whether timestamp tokens / segment spans are suppressed.
  #[inline(always)]
  pub const fn no_timestamps(&self) -> bool {
    self.no_timestamps
  }

  /// Set the language code (the present / explicit-language state).
  #[inline(always)]
  pub fn set_language(&mut self, language: impl Into<String>) -> &mut Self {
    self.language = Some(language.into());
    self
  }

  /// Return `self` with the language code set.
  #[must_use]
  #[inline(always)]
  pub fn with_language(mut self, language: impl Into<String>) -> Self {
    self.language = Some(language.into());
    self
  }

  /// Assign the raw language wrapper (`None` ⇒ auto-detect).
  #[inline(always)]
  pub fn update_language(&mut self, language: Option<String>) -> &mut Self {
    self.language = language;
    self
  }

  /// Return `self` with the raw language wrapper assigned (`None` ⇒
  /// auto-detect).
  #[must_use]
  #[inline(always)]
  pub fn maybe_language(mut self, language: Option<String>) -> Self {
    self.language = language;
    self
  }

  /// Clear the language code (revert to auto-detect).
  #[inline(always)]
  pub fn clear_language(&mut self) -> &mut Self {
    self.language = None;
    self
  }

  /// Set the transcription task.
  #[inline(always)]
  pub const fn set_task(&mut self, task: Task) -> &mut Self {
    self.task = task;
    self
  }

  /// Return `self` with the task set.
  #[must_use]
  #[inline(always)]
  pub const fn with_task(mut self, task: Task) -> Self {
    self.task = task;
    self
  }

  /// Set the sampling temperature.
  #[inline(always)]
  pub const fn set_temperature(&mut self, temperature: f32) -> &mut Self {
    self.temperature = temperature;
    self
  }

  /// Return `self` with the sampling temperature set.
  #[must_use]
  #[inline(always)]
  pub const fn with_temperature(mut self, temperature: f32) -> Self {
    self.temperature = temperature;
    self
  }

  /// Suppress timestamp tokens / segment spans.
  #[inline(always)]
  pub const fn set_no_timestamps(&mut self) -> &mut Self {
    self.no_timestamps = true;
    self
  }

  /// Return `self` with timestamps suppressed.
  #[must_use]
  #[inline(always)]
  pub const fn with_no_timestamps(mut self) -> Self {
    self.no_timestamps = true;
    self
  }

  /// Assign the raw timestamp-suppression flag.
  #[inline(always)]
  pub const fn update_no_timestamps(&mut self, no_timestamps: bool) -> &mut Self {
    self.no_timestamps = no_timestamps;
    self
  }

  /// Return `self` with the raw timestamp-suppression flag assigned.
  #[must_use]
  #[inline(always)]
  pub const fn maybe_no_timestamps(mut self, no_timestamps: bool) -> Self {
    self.no_timestamps = no_timestamps;
    self
  }

  /// Clear timestamp suppression (re-enable timestamps).
  #[inline(always)]
  pub const fn clear_no_timestamps(&mut self) -> &mut Self {
    self.no_timestamps = false;
    self
  }
}

impl Default for TranscribeOptions {
  fn default() -> Self {
    Self::new()
  }
}

/// One contiguous span of transcribed text with its time bounds in seconds.
///
/// CTC transcriptions carry a single segment spanning the whole utterance;
/// autoregressive models with timestamp tokens emit one per timed span.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
  /// The segment's decoded text.
  text: String,
  /// Start time in seconds.
  start: f64,
  /// End time in seconds.
  end: f64,
}

impl Segment {
  /// Construct a segment from its text and `[start, end]` time bounds
  /// (seconds).
  #[inline(always)]
  pub fn new(text: impl Into<String>, start: f64, end: f64) -> Self {
    Self {
      text: text.into(),
      start,
      end,
    }
  }

  /// The segment's decoded text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// Start time in seconds.
  #[inline(always)]
  pub const fn start(&self) -> f64 {
    self.start
  }

  /// End time in seconds.
  #[inline(always)]
  pub const fn end(&self) -> f64 {
    self.end
  }
}

/// The result of a [`Transcribe::transcribe`] run: the full text plus its
/// per-segment breakdown and the detected / configured language.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcription {
  /// The full transcribed text (the concatenation of every segment's text).
  text: String,
  /// The language the audio was transcribed as, or `None` when the model does
  /// not report one (CTC models).
  language: Option<String>,
  /// The per-segment breakdown; one segment for CTC, one per timed span for
  /// timestamped autoregressive decoding.
  segments: Vec<Segment>,
}

impl Transcription {
  /// Construct a transcription from its text, language, and segments.
  #[inline(always)]
  pub fn new(text: impl Into<String>, language: Option<String>, segments: Vec<Segment>) -> Self {
    Self {
      text: text.into(),
      language,
      segments,
    }
  }

  /// The full transcribed text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// The transcribed-as language, or `None` when unreported.
  #[inline(always)]
  pub fn language(&self) -> Option<&str> {
    self.language.as_deref()
  }

  /// The per-segment breakdown.
  #[inline(always)]
  pub fn segments_slice(&self) -> &[Segment] {
    &self.segments
  }
}

/// Mel-spectrogram extraction config — the argument bundle
/// [`crate::audio::dsp::log_mel_spectrogram_with`] consumes, returned by
/// [`AutoregressiveStt::mel_config`] and consumed by
/// [`super::generate::default_log_mel`].
///
/// [`MelConfig::whisper_default`] is the Whisper preset (the only one
/// mlx-audio bundles as a "default"); per-model overrides supply custom
/// values for architectures with different mel front-ends.
///
/// `Copy` because every field is a trivially-copyable primitive.
#[derive(Debug, Clone, Copy)]
pub struct MelConfig {
  /// FFT length (mlx-audio whisper default `400`).
  n_fft: usize,
  /// STFT hop length in samples (mlx-audio whisper default `160`).
  hop_length: usize,
  /// Window length in samples; `None` ⇒ `n_fft` (mlx-audio default).
  win_length: Option<usize>,
  /// Number of mel filterbank bins (mlx-audio whisper default `80`; canary
  /// uses `128`).
  n_mels: usize,
  /// Target audio sample rate in Hz (mlx-audio whisper default `16_000`).
  /// [`super::generate::default_log_mel`] assumes the input waveform is
  /// already at this rate; a model resampling a different source rate does so
  /// (e.g. via [`super::generate::resample_waveform`]) before
  /// [`AutoregressiveStt::log_mel`].
  sample_rate: u32,
  /// Lower mel band edge (Hz; mlx-audio default `0.0`).
  f_min: f32,
  /// Upper mel band edge (Hz); `None` ⇒ `sample_rate / 2` (Nyquist), the
  /// `mel_filter_bank` default.
  f_max: Option<f32>,
  /// Numerical floor applied to mel energies before the **natural log**
  /// ([`crate::audio::dsp::LogFloor`]). Whisper frontends expect `1e-10`
  /// ([`LogFloor::Whisper`](crate::audio::dsp::LogFloor::Whisper), the
  /// default); Kaldi-style frontends expect `1e-8`
  /// ([`LogFloor::Kaldi`](crate::audio::dsp::LogFloor::Kaldi)). A model whose
  /// feature extractor uses the Kaldi floor MUST set this in its
  /// [`AutoregressiveStt::mel_config`] override — otherwise low-energy bins
  /// are shifted by `ln(1e-8) - ln(1e-10) = ln(100) ≈ 4.6` natural-log
  /// units, silently degrading transcription quality.
  log_floor: dsp::LogFloor,
}

impl MelConfig {
  /// Construct a [`MelConfig`] from all fields.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    n_fft: usize,
    hop_length: usize,
    win_length: Option<usize>,
    n_mels: usize,
    sample_rate: u32,
    f_min: f32,
    f_max: Option<f32>,
    log_floor: dsp::LogFloor,
  ) -> Self {
    Self {
      n_fft,
      hop_length,
      win_length,
      n_mels,
      sample_rate,
      f_min,
      f_max,
      log_floor,
    }
  }

  /// The Whisper preset: `n_fft=400`, `hop_length=160`, `n_mels=80`,
  /// `sample_rate=16_000`, `f_min=0`, `f_max=None` (Nyquist), Whisper
  /// log floor. Matches mlx-audio's whisper feature-extractor defaults.
  pub const fn whisper_default() -> Self {
    Self {
      n_fft: 400,
      hop_length: 160,
      win_length: None,
      n_mels: 80,
      sample_rate: 16_000,
      f_min: 0.0,
      f_max: None,
      log_floor: dsp::LogFloor::Whisper,
    }
  }

  /// FFT length.
  #[inline(always)]
  pub const fn n_fft(&self) -> usize {
    self.n_fft
  }

  /// STFT hop length in samples.
  #[inline(always)]
  pub const fn hop_length(&self) -> usize {
    self.hop_length
  }

  /// Window length in samples; `None` ⇒ `n_fft`.
  #[inline(always)]
  pub const fn win_length(&self) -> Option<usize> {
    self.win_length
  }

  /// Number of mel filterbank bins.
  #[inline(always)]
  pub const fn n_mels(&self) -> usize {
    self.n_mels
  }

  /// Target audio sample rate in Hz.
  #[inline(always)]
  pub const fn sample_rate(&self) -> u32 {
    self.sample_rate
  }

  /// Lower mel band edge in Hz.
  #[inline(always)]
  pub const fn f_min(&self) -> f32 {
    self.f_min
  }

  /// Upper mel band edge in Hz; `None` ⇒ Nyquist.
  #[inline(always)]
  pub const fn f_max(&self) -> Option<f32> {
    self.f_max
  }

  /// Numerical log floor applied before the mel's natural log.
  #[inline(always)]
  pub const fn log_floor(&self) -> dsp::LogFloor {
    self.log_floor
  }

  /// Return a copy with `n_fft` overridden.
  #[must_use]
  #[inline(always)]
  pub const fn with_n_fft(mut self, n_fft: usize) -> Self {
    self.n_fft = n_fft;
    self
  }

  /// Return a copy with `hop_length` overridden.
  #[must_use]
  #[inline(always)]
  pub const fn with_hop_length(mut self, hop_length: usize) -> Self {
    self.hop_length = hop_length;
    self
  }

  /// Return a copy with `win_length` overridden.
  #[must_use]
  #[inline(always)]
  pub const fn with_win_length(mut self, win_length: Option<usize>) -> Self {
    self.win_length = win_length;
    self
  }

  /// Return a copy with `n_mels` overridden.
  #[must_use]
  #[inline(always)]
  pub const fn with_n_mels(mut self, n_mels: usize) -> Self {
    self.n_mels = n_mels;
    self
  }

  /// Return a copy with `sample_rate` overridden.
  #[must_use]
  #[inline(always)]
  pub const fn with_sample_rate(mut self, sample_rate: u32) -> Self {
    self.sample_rate = sample_rate;
    self
  }

  /// Return a copy with `f_min` overridden.
  #[must_use]
  #[inline(always)]
  pub const fn with_f_min(mut self, f_min: f32) -> Self {
    self.f_min = f_min;
    self
  }

  /// Return a copy with `f_max` overridden.
  #[must_use]
  #[inline(always)]
  pub const fn with_f_max(mut self, f_max: Option<f32>) -> Self {
    self.f_max = f_max;
    self
  }

  /// Return a copy with `log_floor` overridden.
  #[must_use]
  #[inline(always)]
  pub const fn with_log_floor(mut self, log_floor: dsp::LogFloor) -> Self {
    self.log_floor = log_floor;
    self
  }
}
