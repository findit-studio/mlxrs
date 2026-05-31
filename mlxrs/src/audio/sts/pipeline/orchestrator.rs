//! [`VoiceSession`] — the default [`super::VoicePipeline`] implementor:
//! a synchronous orchestrator that composes a VAD + chunker + STT +
//! LLM + TTS + audio-out into the realtime voice loop mlx-audio's
//! `VoicePipeline._listener` / `_response_processor` /
//! `_audio_output_processor` fan-out drives.
//!
//! The session pulls mic frames from the caller's iterator, runs them
//! through the chunker, queries the VAD via [`VadFrameAdapter`],
//! tracks the silence run via [`super::TurnTakingPolicy`], finalizes
//! a turn by calling [`SttTurnAdapter::transcribe_turn`], hands the
//! text to [`LlmResponderAdapter::respond`], and streams the response
//! through [`TtsStreamAdapter::synthesize_stream`] into the supplied
//! [`crate::audio::playback::AudioOutputStream`].
//!
//! ## Why the adapter traits exist
//!
//! mlxrs's audio + lm trait surfaces already expose
//! architecture-agnostic seams ([`crate::audio::vad::VadModel`] /
//! [`crate::audio::stt::model::Model`] / [`crate::lm::model::Model`] /
//! [`crate::audio::tts::model::TtsModel`]), but their signatures speak
//! in **whole-utterance** terms (full audio → tokens) — the
//! orchestrator's per-frame loop needs a streaming-shaped view
//! ("here's the next chunk, give me a probability"). To avoid
//! altering the existing public APIs, this module
//! ships **thin per-step adapter traits** the caller wraps around
//! their concrete models. The default implementors live in user
//! code (e.g. a Silero adapter that calls `VadModel::generate` with
//! a chunked buffer); the adapters here describe the shape the
//! orchestrator needs.
//!
//! ## Synchronous design
//!
//! mlx-audio's pipeline runs four `asyncio.create_task` coroutines
//! concurrently (`_listener` + `_transcription_stepper` +
//! `_response_processor` + `_audio_output_processor`). mlxrs's
//! synchronous orchestrator runs them sequentially per mic frame:
//! VAD → chunker → policy → (on turn-end) STT → LLM → TTS → out.
//! That is correct for **batch** transcription (a finite mic
//! iterator: file replay, unit test, etc.) and a sound starting
//! point for an async wrapping in user code: an async caller can
//! call [`VoiceSession::step`] from inside an `async fn` and yield
//! to the runtime between frames without modifying the session's
//! per-frame logic.

use crate::{
  audio::{playback::AudioOutputStream, sts::pipeline::voice_pipeline::VoicePipeline},
  error::{InvariantViolationPayload, Result},
};

use super::{
  barge_in::BargeInDetector,
  chunker::{AudioChunker, PreRollBuffer},
  config::VoicePipelineConfig,
  turn_taking::TurnTakingPolicy,
};

/// Per-frame VAD adapter — the streaming shape
/// [`crate::audio::vad::VadModel`]'s whole-utterance contract
/// can't express directly.
///
/// `is_speech` returns whether `frame` (one chunker-aligned chunk
/// of mono `f32` at the configured input sample rate) contains
/// speech. The orchestrator queries this per chunk to drive the
/// silence-run counter.
///
/// The default implementor is in user code (a closure or a struct
/// wrapping a loaded silero VAD `Box<dyn VadModel>` whose
/// `generate(...).timestamps` is non-empty); the trait shape is
/// the orchestrator's only need.
pub trait VadFrameAdapter {
  /// Return whether `frame` contains speech.
  ///
  /// # Errors
  /// Implementor-defined; `is_speech` returns [`crate::error::Error`] and the
  /// orchestrator propagates the implementor's `Err` unchanged.
  fn is_speech(&mut self, frame: &[f32]) -> Result<bool>;
}

/// Per-turn STT adapter — the streaming shape
/// [`crate::audio::stt::model::Model`]'s
/// `encode_audio` + `decode_step` contract can't express directly.
///
/// `transcribe_turn` consumes the full audio of one user turn
/// (the concatenated chunker output between VAD start-of-speech
/// and policy-confirmed turn-end) and returns the recognized
/// text.
///
/// The default implementor is in user code (a struct wrapping a
/// loaded whisper / parakeet / Voxtral STT `dyn Model` that drives
/// `crate::audio::stt::stt_generate` to completion).
pub trait SttTurnAdapter {
  /// Transcribe one full turn's audio.
  ///
  /// # Errors
  /// Implementor-defined; backend / decode failures surface here.
  fn transcribe_turn(&mut self, turn_audio: &[f32]) -> Result<String>;
}

/// Per-turn LLM adapter — the realtime shape
/// [`crate::lm::model::Model`]'s `forward(tokens, cache)` token-
/// level contract can't express directly.
///
/// `respond` accepts the user's transcript and returns the
/// assistant's response. The default implementor is in user code
/// (a struct wrapping a loaded text LM that drives
/// `crate::lm::generate::generate_text` to completion with the
/// pipeline's `system_prompt` + the running conversation
/// history).
pub trait LlmResponderAdapter {
  /// Produce an assistant response for `user_text`. The
  /// implementor is responsible for plumbing the
  /// `system_prompt` + conversation history; the orchestrator
  /// only forwards the user's transcript per turn.
  ///
  /// # Errors
  /// Implementor-defined; generation / tokenization failures
  /// surface here.
  fn respond(&mut self, user_text: &str) -> Result<String>;
}

/// Streaming TTS adapter — the realtime shape
/// [`crate::audio::tts::model::TtsModel`]'s
/// `synthesize_segment(segment) -> Array` whole-segment contract
/// can't express directly.
///
/// `synthesize_stream` consumes the response text and returns a
/// boxed iterator over `Vec<f32>` chunks of mono `f32` PCM at
/// the model's `sample_rate`. The orchestrator pushes each
/// chunk to the audio sink as soon as it arrives (mlx-audio's
/// `_speak_response` / `_audio_output_processor` shape).
///
/// The default implementor is in user code (a struct wrapping a
/// loaded kokoro / csm / pocket-tts `dyn TtsModel` that drives
/// `crate::audio::tts::tts_generate` segment-by-segment).
pub trait TtsStreamAdapter {
  /// Synthesize `text` into a stream of PCM chunks. The
  /// implementor decides chunk granularity (one per text
  /// segment, one per `streaming_interval` slice, etc.).
  ///
  /// # Errors
  /// Implementor-defined; synthesis / vocoder failures surface
  /// here.
  fn synthesize_stream<'a>(
    &'a mut self,
    text: &str,
  ) -> Result<Box<dyn Iterator<Item = Result<Vec<f32>>> + 'a>>;

  /// The output sample rate (Hz). Used to populate
  /// [`super::config::VoicePipelineConfig::output_sample_rate`]
  /// when that field is `None`.
  fn sample_rate(&self) -> u32;
}

/// One event the [`VoiceSession`] emits per realtime turn —
/// surfaced for callers that want to observe / log the realtime
/// loop without intercepting the full per-frame stream (a thin
/// analogue of mlx-audio's verbose `_log_event("turn_finalized",
/// …)` flow).
///
/// The `TurnEvent` is recorded inside [`VoiceSession::run`] and
/// returned in [`VoiceSession::turn_events`] for tests + offline
/// analysis. The orchestrator does not emit events to a callback
/// channel (no `Send`able state); a caller who needs realtime
/// notifications drives the loop a frame at a time via
/// [`VoiceSession::step`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnEvent {
  /// Number of mic chunks consumed before this turn finalized.
  chunks_consumed: usize,
  /// The transcribed user text the LLM saw.
  user_text: String,
  /// The LLM response the TTS spoke.
  assistant_text: String,
  /// Whether barge-in fired during this turn (the
  /// [`BargeInDetector`] returned `true` at least once).
  barge_in_observed: bool,
}

impl TurnEvent {
  /// Construct a `TurnEvent`.
  #[must_use]
  pub fn new(
    chunks_consumed: usize,
    user_text: String,
    assistant_text: String,
    barge_in_observed: bool,
  ) -> Self {
    Self {
      chunks_consumed,
      user_text,
      assistant_text,
      barge_in_observed,
    }
  }

  /// Number of mic chunks consumed before this turn finalized.
  #[inline(always)]
  #[must_use]
  pub fn chunks_consumed(&self) -> usize {
    self.chunks_consumed
  }

  /// The transcribed user text the LLM saw.
  #[inline(always)]
  #[must_use]
  pub fn user_text(&self) -> &str {
    &self.user_text
  }

  /// The LLM response the TTS spoke.
  #[inline(always)]
  #[must_use]
  pub fn assistant_text(&self) -> &str {
    &self.assistant_text
  }

  /// Whether barge-in fired during this turn.
  #[inline(always)]
  #[must_use]
  pub fn barge_in_observed(&self) -> bool {
    self.barge_in_observed
  }
}

/// The default [`VoicePipeline`] implementor — composes every
/// trait surface into the synchronous mic-iterator-driven loop.
///
/// Generic over the user-supplied adapter / detector / policy
/// implementors so the per-frame hot path inlines away the
/// trait dispatch — same shape [`crate::lm::generate`]'s
/// generator uses for its [`crate::lm::generate::Sampler`] +
/// [`crate::lm::generate::LogitsProcessor`] traits.
///
/// `&mut` everywhere on the inner adapters because the
/// transcriber / responder / TTS streamer all carry per-turn
/// session state (mlx-audio's `VoxtralRealtimeTranscriber.session`
/// / `LocalLLMResponseEngine.conversation` / TTS streaming
/// position).
pub struct VoiceSession<V, S, L, T, C, B, P> {
  config: VoicePipelineConfig,
  vad: V,
  stt: S,
  llm: L,
  tts: T,
  chunker: C,
  barge_in: B,
  turn_policy: P,
  preroll: PreRollBuffer,
  /// Turn-event log (cleared per [`VoiceSession::run`] call); a
  /// caller can inspect after `run` returns.
  events: Vec<TurnEvent>,
  /// Per-turn state — accumulated mic audio inside one in-progress
  /// turn.
  in_progress_audio: Vec<f32>,
  /// Whether we are currently inside a speech run.
  in_speech: bool,
  /// Silence-run accumulator (ms) since the last speech frame.
  silence_ms_accum: u32,
  /// Whether the barge-in detector fired during the CURRENT in-progress
  /// turn. Reset at the start of each new turn so an idle-noise
  /// barge-in observation cannot leak into a later, unrelated turn.
  current_turn_barge_in: bool,
  /// Total chunks consumed across the lifetime of this session.
  total_chunks_consumed: usize,
}

impl<V, S, L, T, C, B, P> VoiceSession<V, S, L, T, C, B, P>
where
  V: VadFrameAdapter,
  S: SttTurnAdapter,
  L: LlmResponderAdapter,
  T: TtsStreamAdapter,
  C: AudioChunker,
  B: BargeInDetector,
  P: TurnTakingPolicy,
{
  /// Build a session wiring every trait object together. The
  /// session's [`PreRollBuffer`] capacity is derived from
  /// `config.input_sample_rate() * config.preroll_ms() / 1000`
  /// (mirror of `voice_pipeline.py:613-615`).
  ///
  /// # Errors
  /// Returns [`crate::error::Error::InvariantViolation`] when
  /// `config.input_sample_rate() == 0` — the per-chunk silence-ms
  /// accounting divides by the sample rate and a zero rate would
  /// either panic or silently produce nonsense durations.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    config: VoicePipelineConfig,
    vad: V,
    stt: S,
    llm: L,
    tts: T,
    chunker: C,
    barge_in: B,
    turn_policy: P,
  ) -> Result<Self> {
    if config.input_sample_rate() == 0 {
      return Err(crate::error::Error::InvariantViolation(
        InvariantViolationPayload::new(
          "VoiceSession::new: VoicePipelineConfig::input_sample_rate",
          "must be > 0; the orchestrator's per-chunk silence-ms accounting divides by the sample rate",
        ),
      ));
    }
    let preroll_samples =
      (config.input_sample_rate() as usize) * (config.preroll_ms() as usize) / 1_000;
    Ok(Self {
      config,
      vad,
      stt,
      llm,
      tts,
      chunker,
      barge_in,
      turn_policy,
      preroll: PreRollBuffer::new(preroll_samples),
      events: Vec::new(),
      in_progress_audio: Vec::new(),
      in_speech: false,
      silence_ms_accum: 0,
      current_turn_barge_in: false,
      total_chunks_consumed: 0,
    })
  }

  /// All [`TurnEvent`]s recorded during the most recent
  /// [`VoiceSession::run`] call.
  #[must_use]
  pub fn turn_events(&self) -> &[TurnEvent] {
    &self.events
  }

  /// Number of mic chunks consumed across this session's lifetime.
  #[must_use]
  pub fn total_chunks_consumed(&self) -> usize {
    self.total_chunks_consumed
  }

  /// Read-only access to the wrapped VAD adapter — useful for
  /// downstream tests / inspection without exposing `&mut`.
  pub fn vad(&self) -> &V {
    &self.vad
  }

  /// Read-only access to the wrapped STT adapter.
  pub fn stt(&self) -> &S {
    &self.stt
  }

  /// Read-only access to the wrapped LLM adapter.
  pub fn llm(&self) -> &L {
    &self.llm
  }

  /// Read-only access to the wrapped TTS adapter.
  pub fn tts(&self) -> &T {
    &self.tts
  }

  /// Process one mic-frame iterator step: push through the chunker,
  /// run VAD per chunk, update the turn state, and — on a
  /// turn-finalize event — drive STT + LLM + TTS into `output`.
  ///
  /// Returns the number of turns finalized in this step (typically
  /// `0` or `1`; mlx-audio's pipeline can finalize multiple if a
  /// long mic frame spans more silence than the policy threshold).
  ///
  /// `tts_playing` is the caller's signal "is the audio sink
  /// currently emitting samples"; mlxrs's
  /// [`crate::audio::playback::AudioOutputStream::is_running`]
  /// returns it for the default `AudioPlayer` sink.
  pub fn step<O: AudioOutputStream>(
    &mut self,
    frame: &[f32],
    output: &mut O,
    tts_playing: bool,
  ) -> Result<usize> {
    let mut turns_finalized = 0;
    let chunks = self.chunker.push_samples(frame)?;
    let sample_rate = self.config.input_sample_rate() as u64;

    for chunk in chunks {
      self.total_chunks_consumed += 1;
      // Per-chunk chunk_ms = chunk_samples * 1000 / sample_rate.
      // Computed PER CHUNK (not from chunks[0]) so a chunker that
      // emits variable-size frames accumulates silence-ms
      // faithfully. `sample_rate` is validated > 0 in `new()`.
      let chunk_ms = ((chunk.len() as u64) * 1_000 / sample_rate) as u32;

      // VAD first — every other branch depends on this decision,
      // and ordering matters: appending the chunk to the pre-roll
      // BEFORE the VAD-decision branch would double-feed the first
      // speech chunk (preroll already has it, then the start-of-
      // turn snapshot prepends preroll AND we append the chunk
      // separately).
      let is_speech = self.vad.is_speech(&chunk)?;

      if is_speech {
        if !self.in_speech {
          // Start of turn: reset the barge-in flag (defensive — also
          // reset in `finalize_turn` — so an "idle noise while TTS
          // playing" detection that fired in a previous turn cannot
          // leak in here), then snapshot the pre-roll into the turn
          // audio so the STT sees the leading samples the VAD ran
          // past, and append the current speech chunk ONCE.
          self.current_turn_barge_in = false;
          let preroll_snapshot = self.preroll.snapshot();
          self.in_progress_audio.extend_from_slice(&preroll_snapshot);
          self.in_progress_audio.extend_from_slice(&chunk);
          self.preroll.clear();
          self.in_speech = true;
        } else {
          // Mid-turn speech: just append.
          self.in_progress_audio.extend_from_slice(&chunk);
        }
        self.silence_ms_accum = 0;
      } else if self.in_speech {
        // Silence inside a turn: tail it on (mlx-audio carries
        // the silence frames into the STT's tail too — the
        // model uses them for endpointing) and bump the silence
        // counter.
        self.in_progress_audio.extend_from_slice(&chunk);
        self.silence_ms_accum = self.silence_ms_accum.saturating_add(chunk_ms);

        if self
          .turn_policy
          .user_finished(&self.in_progress_audio, self.silence_ms_accum)
        {
          // Finalize this turn.
          self.finalize_turn(output)?;
          turns_finalized += 1;
        }
      } else {
        // Idle non-speech: feed the pre-roll only. (No turn audio
        // accumulation, no silence counter — silence accounting is
        // a per-TURN concept.)
        self.preroll.append(&chunk);
      }

      // Barge-in: a TTS-overlap candidate is only meaningful when
      // (a) the chunk is actually speech (per the VAD) and
      // (b) we are inside a turn (the in-turn speech run is what
      //     would interrupt the TTS — idle background noise that
      //     happens to cross the energy threshold while TTS is
      //     playing is NOT a barge-in event and must not leak into
      //     a later turn's event log).
      //
      // The fence requires `is_speech && in_speech`; the start-of-
      // turn block above flips `in_speech` to true BEFORE we reach
      // this check, so the very first speech chunk that opens a
      // turn is still counted (matches the mlx-audio shape).
      if self.config.barge_in()
        && is_speech
        && self.in_speech
        && self.barge_in.detect(&chunk, tts_playing)
      {
        self.current_turn_barge_in = true;
      }
    }

    Ok(turns_finalized)
  }

  /// Force-finalize any in-progress turn — called from
  /// [`VoiceSession::run`] when the mic iterator exhausts mid-turn.
  /// A noop when no turn is in progress.
  ///
  /// Drains the chunker's residual (the buffered tail shorter than
  /// one full chunk) into the in-progress turn audio BEFORE
  /// finalizing, so the trailing partial-chunk samples reach the STT
  /// rather than being discarded by `finalize_turn`'s
  /// `chunker.reset()`.
  pub fn flush_in_progress_turn<O: AudioOutputStream>(&mut self, output: &mut O) -> Result<bool> {
    let residual = self.chunker.drain_residual();
    if self.in_speech && !residual.is_empty() {
      self.in_progress_audio.extend_from_slice(&residual);
    }
    if !self.in_speech || self.in_progress_audio.is_empty() {
      return Ok(false);
    }
    self.finalize_turn(output)?;
    Ok(true)
  }

  fn finalize_turn<O: AudioOutputStream>(&mut self, output: &mut O) -> Result<()> {
    // Pull turn audio out before calling adapters (so we can reset
    // session state cleanly even if STT errors).
    let turn_audio = std::mem::take(&mut self.in_progress_audio);
    self.in_speech = false;
    self.silence_ms_accum = 0;
    self.preroll.clear();
    self.chunker.reset();
    let barge_in_observed = std::mem::replace(&mut self.current_turn_barge_in, false);

    let user_text = self.stt.transcribe_turn(&turn_audio)?;
    let user_text_for_event = user_text.clone();
    let assistant_text = self.llm.respond(&user_text)?;

    if self.config.play_audio() {
      let stream = self.tts.synthesize_stream(&assistant_text)?;
      for chunk in stream {
        let samples = chunk?;
        let mut written = 0;
        while written < samples.len() {
          let n = output.write_samples(&samples[written..])?;
          if n == 0 {
            // Sink is fully backpressured and won't accept more
            // — surface as InvariantViolation rather than spin.
            return Err(crate::error::Error::InvariantViolation(
              InvariantViolationPayload::new(
                "VoiceSession: audio sink",
                "rejected TTS chunk (write_samples returned 0)",
              ),
            ));
          }
          written += n;
        }
      }
    }

    self.events.push(TurnEvent::new(
      self.total_chunks_consumed,
      user_text_for_event,
      assistant_text,
      barge_in_observed,
    ));
    Ok(())
  }
}

impl<V, S, L, T, C, B, P> VoicePipeline for VoiceSession<V, S, L, T, C, B, P>
where
  V: VadFrameAdapter,
  S: SttTurnAdapter,
  L: LlmResponderAdapter,
  T: TtsStreamAdapter,
  C: AudioChunker,
  B: BargeInDetector,
  P: TurnTakingPolicy,
{
  fn config(&self) -> &VoicePipelineConfig {
    &self.config
  }

  fn run<I, O>(&mut self, mic_input: I, mut output: O) -> Result<()>
  where
    I: Iterator<Item = Vec<f32>>,
    O: AudioOutputStream,
  {
    self.events.clear();
    for frame in mic_input {
      let tts_playing = output.is_running();
      let _ = self.step(&frame, &mut output, tts_playing)?;
    }
    self.flush_in_progress_turn(&mut output)?;
    output.flush()?;
    Ok(())
  }
}

#[cfg(test)]
mod tests;
