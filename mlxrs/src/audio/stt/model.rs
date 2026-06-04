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
//!   a greedy blank-collapse. A model opts into the greedy decode by
//!   delegating to [`super::generate::greedy_ctc_transcribe`] from its own
//!   [`Transcribe`] impl (symmetric with [`super::generate::greedy_transcribe`]
//!   for [`AutoregressiveStt`]).
//! - [`AutoregressiveStt`] — the encoder/decoder family (Whisper and future
//!   attention STT). Its associated [`AutoregressiveStt::Cache`] type carries
//!   the per-model, caller-held decode state, so the cache is a value owned by
//!   each [`Transcribe::transcribe`] call — no model-stored `RefCell`, no
//!   cross-utterance or concurrent sharing.
//!
//! A separate [`ForcedAligner<Input>`] contract covers forced alignment (a
//! known transcript localized in time, rather than recognition). It is generic
//! over its transcript **input** modality — the same generic-over-input shape as
//! [`crate::embeddings::Embed`] — so the model owns how the transcript is
//! represented and tokenized while the shared contract stays object-safe and
//! model-agnostic.
//!
//! Per the project's no-per-model-arch rule, mlxrs ships **no** concrete STT
//! model implementations: those (the conv subsampling + transformer for
//! whisper, the conformer for parakeet, etc.) live in user code on top of
//! these traits. This module is the shared contract every per-model decoder
//! conforms to.

use smol_str::SmolStr;

use crate::{array::Array, audio::dsp, error::Result};

/// The universal speech-to-text contract: audio waveform in, text out.
///
/// This is the **object-safe core** — a single dyn-compatible method — so a
/// loaded model can be handed around as `Box<dyn Transcribe>` / `&dyn
/// Transcribe`. Every STT model implements it directly: a CTC model forwards
/// to [`super::generate::greedy_ctc_transcribe`], a simple autoregressive one
/// forwards to [`super::generate::greedy_transcribe`], and a complex one (e.g.
/// Whisper) runs its own decoding procedure that still reuses its
/// [`AutoregressiveStt`] hooks internally.
///
/// Ergonomic conveniences (default-options helpers) live on the
/// [`TranscribeExt`] extension trait, which is blanket-implemented for every
/// `Transcribe` (including `dyn Transcribe`) so this core stays object-safe.
///
/// `audio` is a mono waveform [`Array`] (the [`crate::audio::io::load_audio`]
/// output, resampled to the model's expected rate). The model's frontend
/// ([`AutoregressiveStt::log_mel`] / [`CtcModel::logits`]) converts it to the
/// features its encoder consumes.
pub trait Transcribe {
  /// Transcribe `audio` into text under `opts`.
  fn transcribe(&self, audio: &Array, opts: &TranscribeOptions) -> Result<Transcription>;
}

/// Ergonomic, dyn-friendly conveniences over [`Transcribe`]. Auto-implemented
/// for every `Transcribe` (including `dyn Transcribe`), so the core stays
/// object-safe while callers get generic conveniences.
pub trait TranscribeExt: Transcribe {
  /// Transcribe a waveform with default options.
  fn transcribe_audio(&self, audio: &Array) -> Result<Transcription> {
    self.transcribe(audio, &TranscribeOptions::default())
  }
}

impl<T: Transcribe + ?Sized> TranscribeExt for T {}

/// The CTC family: non-autoregressive models that emit per-frame logits in a
/// single encoder forward, then collapse them greedily.
///
/// A CTC model supplies the three pieces the greedy-collapse driver
/// ([`super::generate::greedy_ctc_transcribe`]) needs: the per-frame logits,
/// the blank id to collapse against, and the vocabulary map from collapsed ids
/// to text. A model opts into greedy transcription by delegating to
/// [`super::generate::greedy_ctc_transcribe`] from its own [`Transcribe`] impl
/// — symmetric with [`super::generate::greedy_transcribe`] for
/// [`AutoregressiveStt`], and leaving the [`Transcribe`] slot free for a model
/// that needs a custom decode.
pub trait CtcModel {
  /// Per-frame logits of shape `(T', vocab)` — one row of class scores per
  /// encoder time frame — for the mono `waveform`.
  fn logits(&self, waveform: &Array) -> Result<Array>;

  /// The CTC blank class id (collapsed out of the greedy decode).
  fn blank_id(&self) -> u32;

  /// Map a collapsed id sequence to text via the model's vocabulary. Called
  /// once by the driver after blank-collapse + run-length dedup.
  fn decode_ids(&self, ids: &[u32]) -> String;

  /// Reject a transcription this model cannot render to text, BEFORE the
  /// encoder forward — the empty-vocabulary class guard, enforced once at the
  /// shared chokepoint.
  ///
  /// [`super::generate::greedy_ctc_transcribe`] calls this at its start (ahead
  /// of the waveform validation and the forward), so EVERY route through the
  /// driver — a model's own [`Transcribe`] impl and a
  /// direct `greedy_ctc_transcribe(&model, …)` (the [`CtcModel`] path) alike —
  /// passes through this one guard. A model whose [`Self::decode_ids`] maps an
  /// id sequence to text only when it carries a non-empty vocabulary overrides
  /// this to reject the un-renderable case with a typed error, rather than the
  /// driver silently succeeding with empty text on a model loaded without its
  /// `vocab.json`.
  ///
  /// The default is `Ok(())`: a model whose `decode_ids` is total over every id
  /// sequence (a self-contained detokenizer, a test mock) needs no guard.
  fn ensure_decodable(&self) -> Result<()> {
    Ok(())
  }
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

  /// The TOTAL decoder context size in tokens — prompt prefix
  /// ([`Self::initial_tokens`]) plus every generated token. The greedy driver
  /// ([`super::generate::greedy_transcribe`]) never lets `prompt + generated`
  /// exceed this, so the model's decoder is never fed a sequence longer than
  /// its positional context.
  ///
  /// The default is Whisper's `448`-slot text-decoder context; a model with a
  /// larger or smaller decoder overrides it.
  fn max_context(&self) -> usize {
    super::generate::DEFAULT_MAX_DECODE_STEPS
  }

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
  /// Caps the number of generated tokens; `None` uses the model's
  /// [`AutoregressiveStt::max_context`]. A caller-supplied limit larger than
  /// the model's remaining context is harmlessly clamped to the context.
  max_new_tokens: Option<usize>,
  /// Compression-ratio fallback threshold: a window whose decoded text exceeds
  /// this gzip ratio is retried at a higher temperature (Whisper's
  /// `compression_ratio_threshold`, default `2.4`). `None` disables the check.
  /// Autoregressive models with a temperature-fallback schedule honor it; CTC
  /// models ignore it.
  compression_ratio_threshold: Option<f64>,
  /// Average-logprob fallback threshold: a window whose mean token log-prob is
  /// below this is retried at a higher temperature (Whisper's
  /// `logprob_threshold`, default `-1.0`). `None` disables the check.
  logprob_threshold: Option<f64>,
  /// No-speech (silence) threshold: a window whose no-speech probability
  /// exceeds this is treated as silence and skipped (Whisper's
  /// `no_speech_threshold`, default `0.6`). `None` disables the skip.
  no_speech_threshold: Option<f64>,
  /// Condition each window's decode on the previously-decoded text (Whisper's
  /// `condition_on_previous_text`, default `true`). `false` resets the decode
  /// prompt per window (more robust to repetition loops, less cross-window
  /// consistency).
  condition_on_previous_text: bool,
  /// Optional text prompting the FIRST window — a custom vocabulary or proper
  /// nouns to bias the decode (Whisper's `initial_prompt`). The text conditions
  /// the decode but is never emitted as transcript. `None` for no prompt.
  initial_prompt: Option<String>,
  /// Attach per-unit timestamps to the result (Whisper's word-timestamp
  /// cross-attention DTW, `word_timestamps`, default `false`). When set, a model
  /// that supports it populates the richer per-word timing on its model-local
  /// result; the universal [`Transcription`] still carries only segment spans.
  word_timestamps: bool,
  /// Seconds timestamps of the clips to process (Whisper's `clip_timestamps`):
  /// the list pairs up as `(start, end, start, end, …)`, each pair restricting
  /// decoding to `[start, end)`. Empty (the default) processes the whole audio.
  clip_timestamps: Vec<f64>,
}

impl TranscribeOptions {
  /// A new options bundle with auto-detect language, [`Task::Transcribe`],
  /// greedy (`temperature == 0.0`) decoding, timestamps enabled, no
  /// generated-token cap (the decode loop is bounded only by the model's
  /// [`AutoregressiveStt::max_context`]), and the standard Whisper quality-
  /// control defaults (compression-ratio `2.4`, logprob `-1.0`, no-speech `0.6`,
  /// condition-on-previous-text on, no word timestamps, no clip restriction).
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      language: None,
      task: Task::Transcribe,
      temperature: 0.0,
      no_timestamps: false,
      max_new_tokens: None,
      compression_ratio_threshold: Some(2.4),
      logprob_threshold: Some(-1.0),
      no_speech_threshold: Some(0.6),
      condition_on_previous_text: true,
      initial_prompt: None,
      word_timestamps: false,
      clip_timestamps: Vec::new(),
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

  /// The caller-supplied cap on the number of generated tokens; `None` uses
  /// the model's [`AutoregressiveStt::max_context`].
  #[inline(always)]
  pub const fn max_new_tokens(&self) -> Option<usize> {
    self.max_new_tokens
  }

  /// Set the generated-token cap. `None` uses the model's
  /// [`AutoregressiveStt::max_context`]; a value larger than the model's
  /// remaining context is harmlessly clamped to the context.
  #[inline(always)]
  pub const fn set_max_new_tokens(&mut self, n: usize) -> &mut Self {
    self.max_new_tokens = Some(n);
    self
  }

  /// Return `self` with the generated-token cap set. A value larger than the
  /// model's remaining context is harmlessly clamped to the context.
  #[must_use]
  #[inline(always)]
  pub const fn with_max_new_tokens(mut self, n: usize) -> Self {
    self.max_new_tokens = Some(n);
    self
  }

  /// Assign the raw generated-token cap wrapper (`None` ⇒ the model's
  /// [`AutoregressiveStt::max_context`]).
  #[inline(always)]
  pub const fn update_max_new_tokens(&mut self, max_new_tokens: Option<usize>) -> &mut Self {
    self.max_new_tokens = max_new_tokens;
    self
  }

  /// Return `self` with the raw generated-token cap wrapper assigned (`None` ⇒
  /// the model's [`AutoregressiveStt::max_context`]).
  #[must_use]
  #[inline(always)]
  pub const fn maybe_max_new_tokens(mut self, max_new_tokens: Option<usize>) -> Self {
    self.max_new_tokens = max_new_tokens;
    self
  }

  /// Clear the generated-token cap (revert to the model's
  /// [`AutoregressiveStt::max_context`]).
  #[inline(always)]
  pub const fn clear_max_new_tokens(&mut self) -> &mut Self {
    self.max_new_tokens = None;
    self
  }

  /// The compression-ratio fallback threshold (`None` disables).
  #[inline(always)]
  pub const fn compression_ratio_threshold(&self) -> Option<f64> {
    self.compression_ratio_threshold
  }

  /// Return `self` with the compression-ratio fallback threshold set (`None`
  /// disables the check).
  #[must_use]
  #[inline(always)]
  pub const fn with_compression_ratio_threshold(mut self, threshold: Option<f64>) -> Self {
    self.compression_ratio_threshold = threshold;
    self
  }

  /// The average-logprob fallback threshold (`None` disables).
  #[inline(always)]
  pub const fn logprob_threshold(&self) -> Option<f64> {
    self.logprob_threshold
  }

  /// Return `self` with the average-logprob fallback threshold set (`None`
  /// disables the check).
  #[must_use]
  #[inline(always)]
  pub const fn with_logprob_threshold(mut self, threshold: Option<f64>) -> Self {
    self.logprob_threshold = threshold;
    self
  }

  /// The no-speech (silence) threshold (`None` disables the skip).
  #[inline(always)]
  pub const fn no_speech_threshold(&self) -> Option<f64> {
    self.no_speech_threshold
  }

  /// Return `self` with the no-speech threshold set (`None` disables the
  /// silence skip).
  #[must_use]
  #[inline(always)]
  pub const fn with_no_speech_threshold(mut self, threshold: Option<f64>) -> Self {
    self.no_speech_threshold = threshold;
    self
  }

  /// Whether each window's decode is conditioned on the prior decoded text.
  #[inline(always)]
  pub const fn condition_on_previous_text(&self) -> bool {
    self.condition_on_previous_text
  }

  /// Return `self` with previous-text conditioning toggled.
  #[must_use]
  #[inline(always)]
  pub const fn with_condition_on_previous_text(mut self, on: bool) -> Self {
    self.condition_on_previous_text = on;
    self
  }

  /// The first-window prompt text, or `None`.
  #[inline(always)]
  pub fn initial_prompt(&self) -> Option<&str> {
    self.initial_prompt.as_deref()
  }

  /// Return `self` with the first-window prompt text set.
  #[must_use]
  #[inline(always)]
  pub fn with_initial_prompt(mut self, prompt: impl Into<String>) -> Self {
    self.initial_prompt = Some(prompt.into());
    self
  }

  /// Return `self` with the raw first-window prompt wrapper assigned (`None`
  /// for no prompt).
  #[must_use]
  #[inline(always)]
  pub fn maybe_initial_prompt(mut self, prompt: Option<String>) -> Self {
    self.initial_prompt = prompt;
    self
  }

  /// Whether per-unit (word) timestamps are requested.
  #[inline(always)]
  pub const fn word_timestamps(&self) -> bool {
    self.word_timestamps
  }

  /// Return `self` with per-unit (word) timestamps requested. A model that
  /// supports it exposes the timing on its model-local result (the universal
  /// [`Transcription`] carries only segment spans).
  #[must_use]
  #[inline(always)]
  pub const fn with_word_timestamps(mut self, on: bool) -> Self {
    self.word_timestamps = on;
    self
  }

  /// The clip-restriction seconds list (pairs of `(start, end)`); empty
  /// processes the whole audio.
  #[inline(always)]
  pub fn clip_timestamps(&self) -> &[f64] {
    &self.clip_timestamps
  }

  /// Return `self` with the clip-restriction seconds list set (pairs of
  /// `(start, end)`; an odd-length list leaves the final clip open-ended).
  #[must_use]
  #[inline(always)]
  pub fn with_clip_timestamps(mut self, clips: Vec<f64>) -> Self {
    self.clip_timestamps = clips;
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

// ───────────────────────── forced alignment ─────────────────────────

/// The forced-alignment contract: a known transcript plus its audio in, the
/// per-unit time spans out — generic over the **input** modality the model
/// accepts the transcript in.
///
/// Distinct from [`Transcribe`] — which *recognizes* unknown speech — forced
/// alignment is given the transcript and only localizes each aligned unit in
/// time (the Qwen3 forced aligner, a wav2vec2 CTC aligner, …).
///
/// Like [`crate::embeddings::Embed`] this is generic over its `Input` (rather
/// than carrying an associated input type), so a model implements it once per
/// transcript representation it accepts and `Box<dyn ForcedAligner<Input,
/// Options = T>>` / `&dyn ForcedAligner<Input, Options = T>` stay object-safe
/// for a chosen `Input` and align-time options `T`. The `Input` is
/// **model-defined**: a model that owns its tokenization accepts a raw
/// transcript (its `align` splits + tokenizes internally); a model fed
/// already-tokenized words accepts that pre-tokenized form. The shared contract
/// here fixes only `audio` in and [`ForcedAlignment`] out, with no assumption
/// about how the transcript is represented or tokenized.
///
/// Each impl declares, via the associated [`Self::Options`] type, the align-time
/// options its input needs — the minimal shared [`AlignOptions`] (a result
/// language label) for a pre-tokenized input, or a richer per-input bundle (e.g.
/// one also carrying a word-segmentation strategy) for a raw-text input.
/// Object-safety is preserved by naming the associated type at the `dyn` site
/// (`dyn ForcedAligner<Input, Options = T>`).
///
/// The per-unit spans ([`AlignedSpan`]) are the whispery `Word { text, range }`
/// equivalent: a `text` plus its `[start_time, end_time]` bounds in seconds —
/// exactly the representation an IoU comparison against another aligner
/// consumes.
pub trait ForcedAligner<Input> {
  /// The align-time options this input needs. The minimal shared
  /// [`AlignOptions`] when the input carries everything else (a pre-tokenized
  /// word sequence); a richer per-input bundle when the align step needs more
  /// (e.g. a raw-text input whose options also carry a word-segmentation
  /// strategy).
  type Options;

  /// Localize each unit of the `input` transcript in `audio` under `opts`,
  /// returning one [`AlignedSpan`] per unit in transcript order.
  ///
  /// `audio` is the model's encoder input (for the Qwen3 aligner, the
  /// precomputed log-mel features `(batch, n_mels, time)`); `input` is the
  /// model-defined transcript representation (raw text + language, or an
  /// already-tokenized word sequence); `opts` is the impl's [`Self::Options`].
  /// A model that owns its tokenization does the word splitting + subword
  /// encoding inside this method.
  fn align(&self, audio: &Array, input: Input, opts: &Self::Options) -> Result<ForcedAlignment>;
}

/// The minimal shared [`ForcedAligner::align`] options: just the result
/// `language` label.
///
/// This is the [`Options`](ForcedAligner::Options) type for an input that
/// carries everything else itself — a pre-tokenized word sequence. The
/// alignment algorithm is deterministic (an argmax over the timestamp head), so
/// unlike [`TranscribeOptions`] there are no sampling knobs; the only field is
/// the `language` label carried through to [`ForcedAlignment::language`]. It is
/// purely a result label here — how the transcript is split into aligned units
/// is the model's `align` concern. A raw-text input whose options must also
/// carry a word-segmentation strategy uses a richer per-input options type that
/// composes this one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AlignOptions {
  /// The transcript language label, carried through to the result. `None`
  /// leaves the result's language unset.
  language: Option<SmolStr>,
}

impl AlignOptions {
  /// A new options bundle with no language label.
  #[inline(always)]
  pub const fn new() -> Self {
    Self { language: None }
  }

  /// The configured language label, or `None`.
  #[inline(always)]
  pub fn language(&self) -> Option<&str> {
    self.language.as_deref()
  }

  /// Set the language label.
  #[inline(always)]
  pub fn set_language(&mut self, language: impl Into<SmolStr>) -> &mut Self {
    self.language = Some(language.into());
    self
  }

  /// Return `self` with the language label set.
  #[must_use]
  #[inline(always)]
  pub fn with_language(mut self, language: impl Into<SmolStr>) -> Self {
    self.language = Some(language.into());
    self
  }
}

/// One aligned unit span: the whispery `Word { text, range }` equivalent — a
/// `text` and its `[start_time, end_time]` bounds in **seconds**.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedSpan {
  /// The aligned unit's display text (the word / character the span localizes).
  text: SmolStr,
  /// Start time in seconds.
  start_time: f64,
  /// End time in seconds.
  end_time: f64,
}

impl AlignedSpan {
  /// Construct a span from its text and `[start_time, end_time]` bounds
  /// (seconds).
  #[inline(always)]
  pub fn new(text: impl Into<SmolStr>, start_time: f64, end_time: f64) -> Self {
    Self {
      text: text.into(),
      start_time,
      end_time,
    }
  }

  /// The aligned unit's display text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// Start time in seconds.
  #[inline(always)]
  pub const fn start_time(&self) -> f64 {
    self.start_time
  }

  /// End time in seconds.
  #[inline(always)]
  pub const fn end_time(&self) -> f64 {
    self.end_time
  }
}

/// The result of a [`ForcedAligner::align`] run: one [`AlignedSpan`] per
/// transcript word plus the language label.
#[derive(Debug, Clone, PartialEq)]
pub struct ForcedAlignment {
  /// One span per transcript word, in transcript order.
  spans: Vec<AlignedSpan>,
  /// The language label from [`AlignOptions::language`], or `None`.
  language: Option<SmolStr>,
}

impl ForcedAlignment {
  /// Construct an alignment from its spans and language label.
  #[inline(always)]
  pub fn new(spans: Vec<AlignedSpan>, language: Option<SmolStr>) -> Self {
    Self { spans, language }
  }

  /// The per-word spans, in transcript order.
  #[inline(always)]
  pub fn spans(&self) -> &[AlignedSpan] {
    &self.spans
  }

  /// The language label, or `None`.
  #[inline(always)]
  pub fn language(&self) -> Option<&str> {
    self.language.as_deref()
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
