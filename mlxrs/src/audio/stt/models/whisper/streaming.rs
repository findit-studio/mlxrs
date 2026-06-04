//! AlignAtt attention-guided streaming Whisper transcription.
//!
//! Implements the **AlignAtt** simultaneous-decoding policy (Papi, Negri,
//! Turchi 2023, *"AlignAtt: Using Attention-based Audio-Translation Alignments
//! as a Guide for Simultaneous Speech Translation"*,
//! <https://www.isca-archive.org/interspeech_2023/papi23_interspeech.pdf>) on
//! top of Whisper's existing cross-attention / alignment-head infrastructure.
//!
//! mlx-audio's Whisper does **not** ship a streaming policy (AlignAtt lives in
//! its parakeet / mega_asr models, different architectures), so there is no
//! mlx-audio Whisper reference to port. This module implements the published
//! algorithm directly, matching the Whisper-specific reference implementation
//! `backspacetg/simul_whisper`
//! (`transcriber/simul_whisper.py`, the INTERSPEECH 2024 *Simul-Whisper* code)
//! for the frame-threshold semantics and the alignment-frame computation.
//!
//! ## The AlignAtt policy
//!
//! Process audio incrementally. At each decode step, for the candidate token
//! examine the cross-attention over the encoder **audio frames** restricted to
//! the model's **alignment heads** (the same `(layer, head)` pairs the
//! word-timestamp DTW uses — averaged into a per-token distribution over audio
//! frames, via the crate-internal `timing::alignatt_frame_attention`). Find the
//! frame the token most attends to (`argmax`). If that frame is within
//! `frame_threshold`
//! (`f`) frames of the **end** of the currently-available audio
//! (`content_frames - most_attended_frame <= f`), **stop** committing tokens —
//! the token's evidence sits at the audio boundary and may change once more
//! audio arrives, so WAIT. Otherwise **commit** the token. This bounds emission
//! latency: a token is emitted only when its acoustic evidence is safely behind
//! the live audio edge.
//!
//! The reference's decision (`simul_whisper.py`):
//! ```text
//! most_attened_frame = argmax(attn_of_alignment_heads[-1, :])   # last token
//! if content_mel_len - most_attened_frame <= frame_threshold:
//!     current_tokens = current_tokens[:, :-1]   # drop the token
//!     break                                     # and wait
//! ```
//!
//! ## Usage
//!
//! [`WhisperStreaming`] maintains a growing audio buffer. Feed audio chunks
//! with [`WhisperStreaming::push_audio`] and pull newly-committed tokens with
//! [`WhisperStreaming::step`] (pass `is_last = true` on the final chunk to flush
//! the tail to eot). [`WhisperStreaming::transcribe_stream`] drives the whole
//! loop over a chunk iterator. Each [`StreamingStep`] carries the tokens the
//! policy committed on that step (never re-emitted or rolled back), their decode
//! text, and per-token timing derived from the alignment frames.
//!
//! Each step re-encodes the available audio and decodes from the committed
//! prefix (the prior committed text conditions the next chunk's decode), so
//! committed tokens grow **monotonically** — the policy only ever appends.

use crate::{
  Array, Error, Result,
  audio::stt::model::{AutoregressiveStt, Task},
  error::CapExceededPayload,
};

use super::{
  audio::{
    N_FRAMES, N_SAMPLES, N_SAMPLES_PER_TOKEN, SAMPLE_RATE, TOKENS_PER_SECOND,
    log_mel_spectrogram_whisper,
  },
  decoding::{DecodingOptions, DecodingTask, SuppressSpec},
  model::WhisperModel,
  tokenizer::{HFTokenizerWrapper, Task as WhisperTask},
};

/// The default AlignAtt frame threshold `f` (encoder frames; one frame ≈ 0.02
/// s). `25` frames ≈ 0.5 s of audio held back from the live edge — a balanced
/// latency/quality trade-off in the AlignAtt sweep (the paper varies `f` over
/// `{2,4,6,8,10,12,14}` for translation; speech recognition tolerates a larger
/// `f`). Override via [`StreamingOptions::with_frame_threshold`].
pub const DEFAULT_FRAME_THRESHOLD: usize = 25;

/// The frame threshold applied on the FINAL chunk — relaxed so the tail is not
/// withheld when no more audio will arrive (the reference's hardcoded `4` on
/// `is_last`).
pub const DEFAULT_LAST_CHUNK_FRAME_THRESHOLD: usize = 4;

/// The minimum number of new-token slots each decode step reserves past the
/// forced committed prefix — so a long committed prefix never starves the
/// continuation (the prefix is capped to `n_text_ctx/2 - MIN_SAMPLE_BUDGET`).
const MIN_SAMPLE_BUDGET: usize = 8;

/// Knobs for an AlignAtt streaming session — the latency/quality trade-off plus
/// the per-decode conditioning.
///
/// `frame_threshold` is the AlignAtt `f`: larger holds tokens back further from
/// the audio edge (lower latency risk of revision, higher emission delay),
/// smaller emits sooner (more responsive, more prone to later revision — though
/// this API never revises a committed token, a too-small `f` simply commits
/// tokens whose evidence is still near the edge).
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingOptions {
  /// The AlignAtt frame threshold `f` for non-final chunks
  /// ([`DEFAULT_FRAME_THRESHOLD`]).
  frame_threshold: usize,
  /// The frame threshold for the final chunk
  /// ([`DEFAULT_LAST_CHUNK_FRAME_THRESHOLD`]).
  last_chunk_frame_threshold: usize,
  /// The spoken language ISO code; `None` is treated as English-only here (the
  /// streaming path does not run per-chunk language detection — a multilingual
  /// caller passes the language explicitly).
  language: Option<String>,
  /// Transcribe vs translate.
  task: Task,
  /// Sampling temperature (`0.0` ⇒ greedy argmax).
  temperature: f32,
  /// Continue each chunk's decode from the in-window committed text (force it as
  /// the decode prefix so the model appends rather than re-decoding). `true`
  /// (the default) carries the running transcript across steps within a window;
  /// `false` decodes each chunk fresh from the sot sequence.
  condition_on_previous_text: bool,
  /// Cap on the in-window committed-token prefix forced into the next chunk's
  /// decode (in tokens). Bounds the forced continuation to its recent tail (the
  /// decode further caps it to leave room for new tokens within the decoder
  /// context). `0` disables the forced-prefix continuation.
  max_prompt_tokens: usize,
}

impl StreamingOptions {
  /// The default streaming options: [`DEFAULT_FRAME_THRESHOLD`] /
  /// [`DEFAULT_LAST_CHUNK_FRAME_THRESHOLD`], English-only, transcribe, greedy,
  /// conditioning on the committed text with a `224`-token prompt cap
  /// (`n_text_ctx / 2 - 1` for the released `448`-context checkpoints).
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      frame_threshold: DEFAULT_FRAME_THRESHOLD,
      last_chunk_frame_threshold: DEFAULT_LAST_CHUNK_FRAME_THRESHOLD,
      language: None,
      task: Task::Transcribe,
      temperature: 0.0,
      condition_on_previous_text: true,
      max_prompt_tokens: 223,
    }
  }

  /// The AlignAtt frame threshold `f` for non-final chunks.
  #[inline(always)]
  pub const fn frame_threshold(&self) -> usize {
    self.frame_threshold
  }

  /// The frame threshold for the final chunk.
  #[inline(always)]
  pub const fn last_chunk_frame_threshold(&self) -> usize {
    self.last_chunk_frame_threshold
  }

  /// The configured language ISO code, or `None`.
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

  /// Whether each chunk's decode is conditioned on the prior committed text.
  #[inline(always)]
  pub const fn condition_on_previous_text(&self) -> bool {
    self.condition_on_previous_text
  }

  /// The committed-prompt token cap.
  #[inline(always)]
  pub const fn max_prompt_tokens(&self) -> usize {
    self.max_prompt_tokens
  }

  /// Return `self` with the non-final-chunk frame threshold `f` set.
  #[must_use]
  #[inline(always)]
  pub const fn with_frame_threshold(mut self, f: usize) -> Self {
    self.frame_threshold = f;
    self
  }

  /// Return `self` with the final-chunk frame threshold set.
  #[must_use]
  #[inline(always)]
  pub const fn with_last_chunk_frame_threshold(mut self, f: usize) -> Self {
    self.last_chunk_frame_threshold = f;
    self
  }

  /// Return `self` with the language ISO code set.
  #[must_use]
  #[inline(always)]
  pub fn with_language(mut self, language: impl Into<String>) -> Self {
    self.language = Some(language.into());
    self
  }

  /// Return `self` with the task set.
  #[must_use]
  #[inline(always)]
  pub const fn with_task(mut self, task: Task) -> Self {
    self.task = task;
    self
  }

  /// Return `self` with the sampling temperature set.
  #[must_use]
  #[inline(always)]
  pub const fn with_temperature(mut self, temperature: f32) -> Self {
    self.temperature = temperature;
    self
  }

  /// Return `self` with previous-text conditioning toggled.
  #[must_use]
  #[inline(always)]
  pub const fn with_condition_on_previous_text(mut self, on: bool) -> Self {
    self.condition_on_previous_text = on;
    self
  }

  /// Return `self` with the committed-prompt token cap set (`0` disables prompt
  /// conditioning).
  #[must_use]
  #[inline(always)]
  pub const fn with_max_prompt_tokens(mut self, n: usize) -> Self {
    self.max_prompt_tokens = n;
    self
  }
}

impl Default for StreamingOptions {
  fn default() -> Self {
    Self::new()
  }
}

/// One committed token with its AlignAtt timing — the streaming emission unit.
///
/// `id` is the token id; `start` / `end` are the token's time bounds in
/// **seconds** (absolute, from the start of the stream), derived from the
/// token's most-attended encoder frame. A token spans one encoder frame-step
/// (`1 / TOKENS_PER_SECOND` ≈ 0.02 s); `start` is its argmax-frame time and
/// `end` the next frame's time.
#[derive(Debug, Clone, PartialEq)]
pub struct CommittedToken {
  /// The committed token id.
  id: u32,
  /// The token start time in seconds (absolute).
  start: f64,
  /// The token end time in seconds (absolute).
  end: f64,
}

impl CommittedToken {
  /// Construct a committed token from its id and `[start, end]` seconds.
  #[inline(always)]
  pub const fn new(id: u32, start: f64, end: f64) -> Self {
    Self { id, start, end }
  }

  /// The token id.
  #[inline(always)]
  pub const fn id(&self) -> u32 {
    self.id
  }

  /// The token start time in seconds (absolute).
  #[inline(always)]
  pub const fn start(&self) -> f64 {
    self.start
  }

  /// The token end time in seconds (absolute).
  #[inline(always)]
  pub const fn end(&self) -> f64 {
    self.end
  }
}

/// The result of one [`WhisperStreaming::step`] — the tokens the AlignAtt policy
/// committed on this step (an empty `tokens` means the policy is still waiting
/// for more audio), their decoded text, and whether the decode reached eot.
#[derive(Debug, Clone)]
pub struct StreamingStep {
  /// The tokens committed on this step, in decode order. Never re-emitted on a
  /// later step (committed tokens grow monotonically).
  tokens: Vec<CommittedToken>,
  /// The decoded text of [`Self::tokens`] (timestamp / special tokens dropped).
  text: String,
  /// `P(<|nospeech|>)` read off this step's decode prefill (`NaN` if the model
  /// has no no-speech token) — a caller can treat a high value as a silent
  /// window.
  no_speech_prob: f64,
  /// `true` once the decode reached eot — the utterance is finished (further
  /// `step` calls on the same buffer commit nothing).
  completed: bool,
}

impl StreamingStep {
  /// Construct a step result.
  #[inline(always)]
  pub fn new(
    tokens: Vec<CommittedToken>,
    text: impl Into<String>,
    no_speech_prob: f64,
    completed: bool,
  ) -> Self {
    Self {
      tokens,
      text: text.into(),
      no_speech_prob,
      completed,
    }
  }

  /// The tokens committed on this step.
  #[inline(always)]
  pub fn tokens(&self) -> &[CommittedToken] {
    &self.tokens
  }

  /// The decoded text of the committed tokens.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// `P(<|nospeech|>)` for this step's decode (`NaN` if the model has no
  /// no-speech token).
  #[inline(always)]
  pub const fn no_speech_prob(&self) -> f64 {
    self.no_speech_prob
  }

  /// Whether the decode reached eot on this step.
  #[inline(always)]
  pub const fn completed(&self) -> bool {
    self.completed
  }

  /// Whether this step committed no new tokens (the policy is waiting for more
  /// audio).
  #[inline(always)]
  pub fn is_empty(&self) -> bool {
    self.tokens.is_empty()
  }
}

/// An AlignAtt streaming Whisper transcriber over a single growing audio buffer.
///
/// Holds a borrowed [`WhisperModel`] and a language/task-resolved
/// [`HFTokenizerWrapper`] (built exactly as
/// [`super::decoding::transcribe`]'s — the streaming path mirrors the
/// lower-level decoding entry that takes an explicit tokenizer). The model holds
/// no decode state; this session owns the audio buffer and the committed token
/// history.
///
/// The buffer is bounded by Whisper's 30-second window. When a stream grows past
/// `N_SAMPLES` the window slides forward by dropping ONLY the already-committed
/// leading audio (the origin advances UP TO — never past — the committed-audio
/// watermark, so the overlap is not re-decoded and no UNcommitted audio is ever
/// discarded); the uncommitted tail is retained. A push that would grow the
/// uncommitted tail past one window is rejected as backpressure
/// ([`WhisperStreaming::push_audio`]) rather than silently trimmed.
///
/// Dedup is a FIRST-CLASS invariant, independent of conditioning: an absolute
/// committed-token watermark makes every [`Self::step`] drop the tokens it has
/// already emitted for the current window before appending, so a repeated decode
/// of the same buffer — or a decode with the conditioning prefix turned OFF —
/// never re-emits a committed token. The
/// already-committed tokens are ALSO carried as the next decode's conditioning
/// prefix (the Simul-Whisper `current_tokens` continuation), but that prefix is a
/// quality/continuation feature; the watermark, not the prefix, is what
/// guarantees no re-emission. The committed transcript and its absolute timings
/// therefore stay strictly monotonic across slides AND across repeated decodes,
/// whether or not the prefix is enabled. The per-token timings live in the
/// absolute stream timeline (anchored by the window origin), not reset per
/// window.
pub struct WhisperStreaming<'a> {
  model: &'a WhisperModel,
  tokenizer: HFTokenizerWrapper<'a>,
  options: StreamingOptions,
  /// The growing audio buffer (mono `f32`, `SAMPLE_RATE` Hz). On a window slide
  /// the leading committed samples are drained (up to the `committed_samples`
  /// watermark); the buffer holds the still-uncommitted tail (capped to the most
  /// recent [`N_SAMPLES`] for encoding).
  audio: Vec<f32>,
  /// Every token committed so far (sampled ids only, no sot/eot), in order —
  /// the running transcript across every window. Append-only: each [`Self::step`]
  /// appends ONLY the genuinely-new suffix of the decode (the tokens past the
  /// [`Self::window_emitted`] dedup watermark), so already-emitted tokens are
  /// never re-appended — independent of the conditioning prefix.
  committed: Vec<u32>,
  /// The dedup watermark: the count of committed tokens whose audio lies in the
  /// CURRENT window (committed since the window origin last advanced) — the
  /// reference's cumulative emitted-token count scoped to the live window
  /// (Simul-Whisper's `current_tokens` continuation; mlx-audio's
  /// `new_tokens = text_tokens[len(self._emitted_tokens):]`). [`Self::step`]
  /// slices each decode to only the suffix PAST this watermark, so a repeated
  /// decode over the same buffer — or a decode with the conditioning prefix off —
  /// cannot re-append already-committed tokens. This is the dedup INVARIANT,
  /// decoupled from the prefix (the prefix is a separate quality feature). Reset
  /// to `0` whenever a window slide advances the origin (the slide drops every
  /// committed sample, so no committed token's audio remains in the window).
  window_emitted: usize,
  /// The committed-audio watermark: the number of leading stream samples
  /// (absolute, across every window) whose audio has already been transcribed —
  /// the furthest-attended committed encoder frame mapped to samples. Monotonic
  /// non-decreasing. On a window slide the origin is advanced to at least this
  /// watermark so committed audio is dropped rather than re-decoded.
  committed_samples: usize,
  /// The number of samples consumed at the start of the current 30-second
  /// window (the window origin), so absolute token times account for windows
  /// already slid past. `0` until the buffer first exceeds [`N_SAMPLES`].
  window_origin_samples: usize,
}

impl<'a> WhisperStreaming<'a> {
  /// Build a streaming session over `model`, a language/task-resolved
  /// `tokenizer`, and `options`.
  ///
  /// `tokenizer` is built by the caller (the same
  /// [`HFTokenizerWrapper::new`] the lower-level
  /// [`super::decoding::transcribe`] takes), so a model loaded without an
  /// attached tokenizer can still be streamed. Use
  /// [`WhisperStreaming::with_model_tokenizer`] when the model carries one.
  pub fn new(
    model: &'a WhisperModel,
    tokenizer: HFTokenizerWrapper<'a>,
    options: StreamingOptions,
  ) -> Self {
    let tokenizer = match options.language() {
      Some(lang) => tokenizer.with_language(lang),
      None => tokenizer,
    };
    Self {
      model,
      tokenizer,
      options,
      audio: Vec::new(),
      committed: Vec::new(),
      window_emitted: 0,
      committed_samples: 0,
      window_origin_samples: 0,
    }
  }

  /// The streaming options.
  #[inline(always)]
  pub const fn options(&self) -> &StreamingOptions {
    &self.options
  }

  /// Every token committed so far (sampled ids, in order).
  #[inline(always)]
  pub fn committed_tokens(&self) -> &[u32] {
    &self.committed
  }

  /// The number of audio samples buffered (within the current 30-second
  /// window).
  #[inline(always)]
  pub fn buffered_samples(&self) -> usize {
    self.audio.len()
  }

  /// Append `samples` (mono `f32` at [`SAMPLE_RATE`] Hz) to the audio buffer.
  ///
  /// When the buffer grows past [`N_SAMPLES`] (30 s) the window slides forward by
  /// dropping ONLY the already-committed leading audio (the origin advances to the
  /// committed-audio watermark `committed_samples`) — the standard Whisper seek
  /// advance. The committed tokens are carried as the next decode's prefix (the
  /// Simul-Whisper continuation), so the retained-overlap text is never re-emitted
  /// and the committed transcript stays strictly monotonic. The slide NEVER drains
  /// past `committed_samples`, so no UNcommitted audio is ever silently discarded.
  /// The window origin only grows, anchoring absolute token times.
  ///
  /// # Backpressure
  /// The decode window is one 30 s segment, so the still-UNcommitted tail
  /// (everything past `committed_samples`) must fit in [`N_SAMPLES`]. The
  /// Simul-Whisper reference feeds `<= 30 s` windows and commits as it goes
  /// (`backspacetg/simul_whisper`, `transcriber/simul_whisper.py`); a single push
  /// adding more than one window of un-committable audio at once — or the policy
  /// committing nothing for over 30 s — would force the next decode to drop the
  /// OLDEST uncommitted audio (silently losing earlier speech). Rather than lose
  /// it, this rejects the push with [`Error::CapExceeded`] and leaves the buffer
  /// UNCHANGED (so the caller keeps the rejected samples); the caller must commit
  /// or flush ([`Self::step`]) before pushing more. No audio is lost on either
  /// path — the slide drops only committed audio, and an over-large uncommitted
  /// tail is surfaced as caller misuse, never trimmed.
  ///
  /// # Errors
  /// - [`Error::CapExceeded`] if appending `samples` would grow the uncommitted
  ///   tail past one 30 s window ([`N_SAMPLES`]) — see Backpressure;
  /// - [`Error::AllocFailure`] if the buffer cannot be grown to hold the new
  ///   samples.
  pub fn push_audio(&mut self, samples: &[f32]) -> Result<()> {
    // The leading committed (drainable) portion of the current buffer. The
    // watermark is monotonic and clamped to the buffered end, and a slide advances
    // the origin only up to it, so `committed_samples >= window_origin_samples`
    // and this is `<= audio.len()`; the remainder is the never-committed tail.
    let committed_in_window = self
      .committed_samples
      .saturating_sub(self.window_origin_samples)
      .min(self.audio.len());

    // Backpressure, checked BEFORE mutating the buffer so a rejected push leaves
    // the buffer unchanged and the caller keeps its samples: the uncommitted tail
    // after this push must fit one 30 s window. A drainable committed lead is
    // subtracted (it will be dropped by the slide below); what remains is the
    // audio the next decode MUST still see, and it cannot exceed N_SAMPLES without
    // forcing a silent drop of the oldest uncommitted speech.
    let prospective_len = self.audio.len().saturating_add(samples.len());
    let prospective_uncommitted = prospective_len.saturating_sub(committed_in_window);
    if prospective_uncommitted > N_SAMPLES {
      return Err(Error::CapExceeded(CapExceededPayload::new(
        "WhisperStreaming::push_audio: uncommitted audio tail",
        "N_SAMPLES",
        N_SAMPLES as u64,
        prospective_uncommitted as u64,
      )));
    }

    crate::model_validation::reserve_or_error(
      &mut self.audio,
      "WhisperStreaming: audio buffer",
      samples.len(),
    )?;
    self.audio.extend_from_slice(samples);

    // Slide once the buffer exceeds N_SAMPLES by dropping ONLY the committed
    // leading audio (origin advances to `committed_samples`). The backpressure
    // check above guarantees the uncommitted tail already fits one window, so
    // dropping the committed lead leaves the buffer within N_SAMPLES; no
    // uncommitted audio is ever discarded. The committed tokens are carried as the
    // next decode's prefix (`window_prefix`), so dropping their audio does not lose
    // the continuation. The slide drops every committed sample of this window, so
    // no committed token's audio remains: reset the dedup watermark.
    if self.audio.len() > N_SAMPLES && committed_in_window > 0 {
      self.audio.drain(..committed_in_window);
      self.window_origin_samples = self
        .window_origin_samples
        .saturating_add(committed_in_window);
      self.window_emitted = 0;
    }
    Ok(())
  }

  /// Run one AlignAtt decode over the audio buffered so far and return the
  /// tokens newly COMMITTED — the incremental emission step.
  ///
  /// `is_last` flushes the tail: on the final chunk the frame threshold relaxes
  /// to [`StreamingOptions::last_chunk_frame_threshold`] and the decode runs to
  /// eot, so the remaining audio is fully transcribed. On a non-final chunk the
  /// policy holds back tokens whose acoustic evidence is within
  /// [`StreamingOptions::frame_threshold`] frames of the live audio edge.
  ///
  /// Returns an empty [`StreamingStep`] when there is too little audio to encode
  /// (< one encoder frame) or the policy commits nothing yet. The committed
  /// tokens are appended to the running transcript and conditioned into the next
  /// step's decode.
  ///
  /// # Errors
  /// - [`crate::Error::OutOfRange`] if a dimension overflows `i32`;
  /// - propagates the front-end, encoder, decoder, filter, and sampler op
  ///   errors.
  pub fn step(&mut self, is_last: bool) -> Result<StreamingStep> {
    // Need at least one encoder frame-pair of audio to align against. An empty /
    // sub-frame buffer commits nothing (no error — the caller simply feeds more).
    if self.audio.is_empty() {
      return Ok(StreamingStep::new(
        Vec::new(),
        String::new(),
        f64::NAN,
        false,
      ));
    }

    // Log-mel of the buffered window (no trailing 30 s pad — the streaming
    // window is the live audio, not a fixed segment), then pad/trim to N_FRAMES
    // for the encoder. `content_frames` is the real (non-pad) encoder-frame
    // count = real_mel_frames / 2 (the conv stride-2 downsample).
    let audio = Array::from_slice::<f32>(&self.audio, &[self.audio.len() as i32])?;
    let mel = log_mel_spectrogram_whisper(&audio, self.model.dims().n_mels(), 0)?;
    let real_mel_frames = mel.shape()[0].min(N_FRAMES);
    let content_frames = real_mel_frames / 2;
    if content_frames == 0 {
      return Ok(StreamingStep::new(
        Vec::new(),
        String::new(),
        f64::NAN,
        false,
      ));
    }
    let mel_window = super::audio::pad_or_trim(&mel, N_FRAMES, 0)?;
    let enc = self.model.encode(&mel_window)?;

    // Build the per-chunk decode task. The in-window committed tokens are forced
    // as the decode PREFIX so the decode CONTINUES past them (the Simul-Whisper
    // `current_tokens` continuation) — the model appends new tokens rather than
    // re-decoding and re-emitting the committed text. Timestamps are off (the
    // streaming emission is token-level; timing comes from the alignment
    // frames).
    //
    // `build_initial_tokens` keeps only the prefix tail of length
    // `n_text_ctx/2 - sample_len`, so the prefix and `sample_len` must be sized
    // together: bound the prefix to leave at least `MIN_SAMPLE_BUDGET` new-token
    // slots, then set `sample_len` so the whole forced prefix survives the
    // truncation (`sample_len = n_text_ctx/2 - prefix_len`). This keeps every
    // forwarded prefix within the decoder context AND keeps the continuation
    // intact.
    let half_ctx = (self.model.dims().n_text_ctx() / 2).max(1);
    let max_prefix = half_ctx.saturating_sub(MIN_SAMPLE_BUDGET);
    let prefix = self.window_prefix(max_prefix);
    // Whether the decode is forced to CONTINUE from a committed prefix. When a
    // prefix is forced, `decode_aligned` strips it (via `sample_begin`) and
    // `aligned.tokens` is the pure continuation PAST the forced prefix — so the
    // window's already-committed tokens are absent from the decode and the dedup
    // skip is 0. When the prefix is EMPTY (conditioning off, `max_prompt_tokens`
    // is 0, or nothing committed yet), the decode restarts from the sot sequence
    // and re-decodes the whole window, so `aligned.tokens` REPEATS the window's
    // already-committed tokens, which the [`Self::window_emitted`] watermark drops
    // below. Dedup therefore holds whether or not the prefix is on.
    let prefix_forced_continuation = !prefix.is_empty();
    let sample_len = half_ctx.saturating_sub(prefix.len()).max(1);
    let decode = DecodingOptions {
      task: task_to_whisper(self.options.task),
      language: self.options.language.clone(),
      temperature: self.options.temperature,
      // Streaming decodes a single AlignAtt trajectory per chunk — no best-of /
      // beam batching.
      best_of: None,
      beam_size: None,
      length_penalty: None,
      sample_len: Some(sample_len),
      prompt: Vec::new(),
      prefix,
      suppress_tokens: SuppressSpec::NonSpeech,
      suppress_blank: true,
      without_timestamps: true,
      max_initial_timestamp: None,
    };
    let task = DecodingTask::new(self.model, &self.tokenizer, decode)?;

    // The EFFECTIVE AlignAtt threshold for this chunk: the looser
    // `last_chunk_frame_threshold` on the final chunk (so the tail is not held
    // back when no more audio will arrive), else `frame_threshold`. The decode
    // applies the SAME inequality with whichever threshold this resolves to —
    // mirroring the reference's `(4 if is_last else frame_threshold)`.
    let frame_threshold = if is_last {
      self.options.last_chunk_frame_threshold
    } else {
      self.options.frame_threshold
    };
    let aligned = task.decode_aligned(&enc, content_frames, frame_threshold)?;

    // Dedup against the running transcript via the window watermark — the
    // FIRST-CLASS invariant that keeps `step` append-only regardless of the
    // conditioning prefix. When a prefix forced the decode to continue,
    // `decode_aligned` already stripped it, so `aligned.tokens` is the genuine
    // continuation (skip 0). When the prefix was empty, the decode re-emitted the
    // window's already-committed tokens, so drop exactly those `window_emitted`
    // leading tokens. Clamping to `aligned.tokens.len()` means a decode that
    // committed FEWER tokens than already emitted (e.g. a non-final step after a
    // looser final step, or a shorter re-decode) yields an empty suffix rather
    // than slicing out of bounds — never a re-emission. Mirrors the reference's
    // `new_tokens = text_tokens[len(self._emitted_tokens):]`, decoupled from the
    // prefix.
    let skip = if prefix_forced_continuation {
      0
    } else {
      self.window_emitted.min(aligned.tokens.len())
    };
    let new_ids = &aligned.tokens[skip..];
    // `decode_aligned` keeps `frames` parallel to `tokens` (defensively truncated
    // to the same length); clamp the frame skip to the frame length in case it is
    // the shorter of the two — the per-token `new_frames.get(i)` lookup below then
    // falls back to frame 0 for any missing tail entry.
    let new_frames = &aligned.frames[skip.min(aligned.frames.len())..];

    // The encoder frame origin (in seconds) of the current window. Tokens are
    // timed by their argmax encoder frame within the window plus this offset.
    let window_offset_s = self.window_origin_samples as f64 / SAMPLE_RATE as f64;
    let mut tokens: Vec<CommittedToken> = Vec::new();
    crate::model_validation::reserve_or_error(
      &mut tokens,
      "WhisperStreaming: committed tokens",
      new_ids.len(),
    )?;
    for (i, &id) in new_ids.iter().enumerate() {
      // One encoder frame ≈ 1 / TOKENS_PER_SECOND seconds (≈ 0.02 s).
      let frame = new_frames.get(i).copied().unwrap_or(0);
      let start = window_offset_s + frame as f64 / TOKENS_PER_SECOND as f64;
      let end = window_offset_s + (frame + 1) as f64 / TOKENS_PER_SECOND as f64;
      tokens.push(CommittedToken::new(id, start, end));
    }

    // Decode the genuinely-new text and append only the new ids to the running
    // transcript. Append-only: the watermark slice above guarantees only tokens
    // past the already-committed window prefix are appended (no re-emission), and
    // `window_emitted` advances by the same count so a repeated `step` over the
    // same buffer adds nothing.
    let text = self.tokenizer.decode(new_ids, false)?;
    crate::model_validation::reserve_or_error(
      &mut self.committed,
      "WhisperStreaming: committed history",
      new_ids.len(),
    )?;
    self.committed.extend_from_slice(new_ids);
    self.window_emitted = self.window_emitted.saturating_add(new_ids.len());

    // Advance the committed-audio watermark to the furthest-attended committed
    // encoder frame (mapped to absolute samples), so the next window slide drops
    // this audio rather than re-decoding it. Uses the FULL `aligned.frames` — the
    // skipped leading tokens are committed too, so their audio counts. Monotonic
    // (`max`) and clamped to the samples buffered in this window, so the watermark
    // never runs past the audio actually seen.
    if let Some(&max_frame) = aligned.frames.iter().max() {
      let committed_end =
        self.window_origin_samples + (max_frame + 1).saturating_mul(N_SAMPLES_PER_TOKEN);
      let buffered_end = self.window_origin_samples + self.audio.len();
      self.committed_samples = self.committed_samples.max(committed_end).min(buffered_end);
    }

    Ok(StreamingStep::new(
      tokens,
      text,
      aligned.no_speech_prob,
      aligned.completed,
    ))
  }

  /// Drive the AlignAtt streaming loop over an audio-chunk iterator, returning
  /// every committed [`StreamingStep`] in order.
  ///
  /// Each item is a chunk of mono `f32` samples (at [`SAMPLE_RATE`] Hz). The
  /// final chunk is decoded with `is_last = true` (the tail is flushed to eot).
  /// An empty iterator yields a single final flush step (which commits nothing
  /// without audio).
  ///
  /// This is the convenience driver over [`Self::push_audio`] + [`Self::step`];
  /// a caller wanting per-chunk control (e.g. to interleave emission with live
  /// capture) uses those directly.
  ///
  /// # Errors
  /// Propagates [`Self::push_audio`] / [`Self::step`].
  pub fn transcribe_stream<I, C>(&mut self, chunks: I) -> Result<Vec<StreamingStep>>
  where
    I: IntoIterator<Item = C>,
    C: AsRef<[f32]>,
  {
    let mut steps: Vec<StreamingStep> = Vec::new();
    // Peek one chunk ahead so the LAST chunk is decoded with `is_last = true`.
    let mut iter = chunks.into_iter();
    let mut pending = iter.next();
    while let Some(chunk) = pending.take() {
      self.push_audio(chunk.as_ref())?;
      pending = iter.next();
      let is_last = pending.is_none();
      // On a non-final chunk, decode incrementally (commit what is safe). On the
      // final chunk, flush to eot.
      let step = self.step(is_last)?;
      if !step.is_empty() || is_last {
        steps.push(step);
      }
    }
    // No chunks at all: a single final flush (commits nothing without audio).
    if steps.is_empty() {
      steps.push(self.step(true)?);
    }
    Ok(steps)
  }

  /// The committed-token prefix forced at the start of the next chunk's decode
  /// (so the decode CONTINUES past it — the Simul-Whisper `current_tokens`
  /// continuation) — the recent tail of the running committed transcript, capped
  /// to the smaller of [`StreamingOptions::max_prompt_tokens`] and `cap` (the
  /// context-derived ceiling the caller computes to leave room for new tokens).
  /// Drawing from the running transcript (not a per-window buffer) carries the
  /// continuation across window slides, so a slid window re-decodes only the
  /// uncommitted tail behind this prefix. Empty when previous-text conditioning
  /// is off or nothing is committed yet (the decode then starts fresh from the
  /// sot sequence).
  fn window_prefix(&self, cap: usize) -> Vec<u32> {
    if !self.options.condition_on_previous_text || self.options.max_prompt_tokens == 0 {
      return Vec::new();
    }
    let keep = self.options.max_prompt_tokens.min(cap);
    let start = self.committed.len().saturating_sub(keep);
    self.committed[start..].to_vec()
  }

  /// Build a streaming session using the tokenizer ATTACHED to `model`
  /// ([`WhisperModel::with_tokenizer`]) — the convenience constructor for a
  /// model loaded with its tokenizer.
  ///
  /// # Errors
  /// [`crate::Error::InvariantViolation`] if `model` has no attached tokenizer
  /// (use [`WhisperStreaming::new`] with an explicit
  /// [`HFTokenizerWrapper`]), or the wrapper-construction error.
  pub fn with_model_tokenizer(model: &'a WhisperModel, options: StreamingOptions) -> Result<Self> {
    let wrapper = model.streaming_tokenizer(options.language(), task_to_whisper(options.task))?;
    Ok(Self::new(model, wrapper, options))
  }
}

/// Convert a universal [`Task`] into the Whisper-internal task slug.
#[inline(always)]
fn task_to_whisper(task: Task) -> WhisperTask {
  match task {
    Task::Transcribe => WhisperTask::Transcribe,
    Task::Translate => WhisperTask::Translate,
  }
}

/// Convert an absolute encoder-frame count to seconds (one frame ≈
/// `1 / TOKENS_PER_SECOND`, ≈ 0.02 s) — the AlignAtt frame→time mapping the
/// committed-token timing uses, exposed for callers mapping frames to time.
#[inline(always)]
pub fn frame_to_seconds(frame: usize) -> f64 {
  frame as f64 / TOKENS_PER_SECOND as f64
}

#[cfg(test)]
mod tests;
