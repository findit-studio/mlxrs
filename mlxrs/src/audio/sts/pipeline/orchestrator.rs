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
//! ("here's the next chunk, give me a probability"). Per the
//! "do NOT alter existing public APIs" A8 constraint, this module
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
  error::Result,
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
  /// Implementor-defined; the orchestrator surfaces an `Err` here
  /// as [`crate::error::Error::Backend`].
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
  pub chunks_consumed: usize,
  /// The transcribed user text the LLM saw.
  pub user_text: String,
  /// The LLM response the TTS spoke.
  pub assistant_text: String,
  /// Whether barge-in fired during this turn (the
  /// [`BargeInDetector`] returned `true` at least once).
  pub barge_in_observed: bool,
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
pub struct VoiceSession<V, S, L, T, C, B, P>
where
  V: VadFrameAdapter,
  S: SttTurnAdapter,
  L: LlmResponderAdapter,
  T: TtsStreamAdapter,
  C: AudioChunker,
  B: BargeInDetector,
  P: TurnTakingPolicy,
{
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
  /// `config.input_sample_rate * config.preroll_ms / 1000`
  /// (mirror of `voice_pipeline.py:613-615`).
  ///
  /// # Errors
  /// Returns [`crate::error::Error::Backend`] when
  /// `config.input_sample_rate == 0` — the per-chunk silence-ms
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
    if config.input_sample_rate == 0 {
      return Err(crate::error::Error::Backend {
        message:
          "VoicePipelineConfig::input_sample_rate must be > 0 (got 0); the orchestrator's per-chunk \
           silence-ms accounting divides by the sample rate"
            .into(),
      });
    }
    let preroll_samples =
      (config.input_sample_rate as usize) * (config.preroll_ms as usize) / 1_000;
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
    let sample_rate = self.config.input_sample_rate as u64;

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
      if self.config.barge_in
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

    if self.config.play_audio {
      let stream = self.tts.synthesize_stream(&assistant_text)?;
      for chunk in stream {
        let samples = chunk?;
        let mut written = 0;
        while written < samples.len() {
          let n = output.write_samples(&samples[written..])?;
          if n == 0 {
            // Sink is fully backpressured and won't accept more
            // — surface as Backend error rather than spin.
            return Err(crate::error::Error::Backend {
              message: "VoiceSession: audio sink rejected TTS chunk (write_samples returned 0)"
                .into(),
            });
          }
          written += n;
        }
      }
    }

    self.events.push(TurnEvent {
      chunks_consumed: self.total_chunks_consumed,
      user_text: user_text_for_event,
      assistant_text,
      barge_in_observed,
    });
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
mod tests {
  use super::*;
  use crate::{
    audio::sts::pipeline::{
      barge_in::EnergyBargeInDetector, chunker::FixedSizeAudioChunker,
      turn_taking::SilenceTurnTakingPolicy,
    },
    error::Error,
  };
  use std::cell::RefCell;

  /// Mock VAD: speech iff chunk RMS ≥ threshold.
  struct MockVad {
    threshold: f32,
  }
  impl VadFrameAdapter for MockVad {
    fn is_speech(&mut self, frame: &[f32]) -> Result<bool> {
      let rms = EnergyBargeInDetector::rms(frame);
      Ok(rms >= self.threshold)
    }
  }

  /// Mock STT: records the audio it sees + returns canned text.
  struct MockStt {
    last_audio_len: RefCell<usize>,
    text: String,
  }
  impl SttTurnAdapter for MockStt {
    fn transcribe_turn(&mut self, turn_audio: &[f32]) -> Result<String> {
      *self.last_audio_len.borrow_mut() = turn_audio.len();
      Ok(self.text.clone())
    }
  }

  /// Mock LLM: appends the user prompt to a fixed prefix.
  struct MockLlm {
    seen: RefCell<Vec<String>>,
  }
  impl LlmResponderAdapter for MockLlm {
    fn respond(&mut self, user_text: &str) -> Result<String> {
      self.seen.borrow_mut().push(user_text.to_string());
      Ok(format!("re:{user_text}"))
    }
  }

  /// Mock TTS: emits one chunk of N samples per word, plus tracks
  /// the prompts.
  struct MockTts {
    seen: RefCell<Vec<String>>,
    samples_per_word: usize,
  }
  impl TtsStreamAdapter for MockTts {
    fn synthesize_stream<'a>(
      &'a mut self,
      text: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<Vec<f32>>> + 'a>> {
      self.seen.borrow_mut().push(text.to_string());
      let n_words = text.split_whitespace().count().max(1);
      let n = self.samples_per_word;
      Ok(Box::new((0..n_words).map(move |_| Ok(vec![0.1_f32; n]))))
    }
    fn sample_rate(&self) -> u32 {
      24_000
    }
  }

  /// Mock audio sink: records every sample it sees. `is_running`
  /// returns whether it has accepted any samples.
  struct MockSink {
    recorded: Vec<f32>,
    write_count: usize,
    flush_count: usize,
    running: bool,
  }
  impl MockSink {
    fn new() -> Self {
      Self {
        recorded: Vec::new(),
        write_count: 0,
        flush_count: 0,
        running: false,
      }
    }
  }
  impl AudioOutputStream for MockSink {
    fn write_samples(&mut self, samples: &[f32]) -> Result<usize> {
      self.recorded.extend_from_slice(samples);
      self.write_count += 1;
      self.running = true;
      Ok(samples.len())
    }
    fn flush(&mut self) -> Result<()> {
      self.flush_count += 1;
      Ok(())
    }
    fn stop(&mut self) -> Result<()> {
      self.running = false;
      Ok(())
    }
    fn is_running(&self) -> bool {
      self.running
    }
  }

  fn test_session() -> VoiceSession<
    MockVad,
    MockStt,
    MockLlm,
    MockTts,
    FixedSizeAudioChunker,
    EnergyBargeInDetector,
    SilenceTurnTakingPolicy,
  > {
    let config = VoicePipelineConfig {
      // 16 kHz, 20 ms chunks = 320 samples; 200 ms silence
      // threshold = 10 chunks worth of silence.
      input_sample_rate: 16_000,
      frame_duration_ms: 20,
      preroll_ms: 40,
      vad_end_silence_ms: 200,
      turn_max_incomplete_silence_ms: 200,
      ..VoicePipelineConfig::default()
    };
    let chunk_size = (16_000 * 20) / 1_000;
    VoiceSession::new(
      config,
      MockVad { threshold: 0.05 },
      MockStt {
        last_audio_len: RefCell::new(0),
        text: "hello world".to_string(),
      },
      MockLlm {
        seen: RefCell::new(Vec::new()),
      },
      MockTts {
        seen: RefCell::new(Vec::new()),
        samples_per_word: 100,
      },
      FixedSizeAudioChunker::new(chunk_size),
      EnergyBargeInDetector::default(),
      SilenceTurnTakingPolicy::new(200),
    )
    .expect("test session input_sample_rate is non-zero")
  }

  /// End-to-end happy path: speech → silence → finalize → STT
  /// + LLM + TTS all called in order with the expected text.
  #[test]
  fn end_to_end_drives_vad_stt_llm_tts_in_order() {
    let mut sess = test_session();
    let mut sink = MockSink::new();

    // 10 chunks of speech (sin wave above 0.05 RMS), then 12
    // chunks of silence to cross the 200 ms threshold (each
    // chunk is 20 ms).
    let chunk_size = 320;
    let speech_chunk: Vec<f32> = (0..chunk_size).map(|i| 0.3 * ((i as f32).sin())).collect();
    let silence_chunk: Vec<f32> = vec![0.0; chunk_size];

    // Pre-buffer some idle silence so pre-roll is non-empty.
    sess.step(&silence_chunk, &mut sink, false).unwrap();
    // 10 speech chunks (in one frame for simplicity).
    let mut speech_frame = Vec::new();
    for _ in 0..10 {
      speech_frame.extend_from_slice(&speech_chunk);
    }
    sess.step(&speech_frame, &mut sink, false).unwrap();

    // Now 11 silence chunks → silence_ms accumulates to 220 ms,
    // crossing the 200 ms threshold and finalizing the turn.
    let mut silence_frame = Vec::new();
    for _ in 0..11 {
      silence_frame.extend_from_slice(&silence_chunk);
    }
    let turns = sess.step(&silence_frame, &mut sink, false).unwrap();

    assert_eq!(turns, 1, "exactly one turn finalized");
    let events = sess.turn_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].user_text, "hello world");
    assert_eq!(events[0].assistant_text, "re:hello world");
    assert!(!events[0].barge_in_observed);

    // STT saw the turn audio: 10 speech chunks + (silence
    // chunks up to the finalize). At least 10 * 320 samples.
    let stt = &sess.stt;
    assert!(*stt.last_audio_len.borrow() >= 10 * chunk_size);

    // LLM saw the STT text exactly once.
    assert_eq!(sess.llm.seen.borrow().as_slice(), &["hello world"]);

    // TTS saw the LLM response exactly once.
    assert_eq!(sess.tts.seen.borrow().as_slice(), &["re:hello world"]);

    // Sink received the TTS chunks: "re:hello world" splits into 2
    // words, 100 samples each = 200 samples written.
    assert_eq!(sink.recorded.len(), 200);
  }

  /// `play_audio = false` honored: no TTS samples written to sink.
  #[test]
  fn play_audio_false_skips_tts() {
    let mut sess = test_session();
    sess.config.play_audio = false;
    let mut sink = MockSink::new();
    let chunk_size = 320;
    let speech_chunk: Vec<f32> = (0..chunk_size).map(|i| 0.3 * ((i as f32).sin())).collect();
    let silence_chunk: Vec<f32> = vec![0.0; chunk_size];
    let mut speech_frame = Vec::new();
    for _ in 0..5 {
      speech_frame.extend_from_slice(&speech_chunk);
    }
    sess.step(&speech_frame, &mut sink, false).unwrap();
    let mut silence_frame = Vec::new();
    for _ in 0..11 {
      silence_frame.extend_from_slice(&silence_chunk);
    }
    sess.step(&silence_frame, &mut sink, false).unwrap();

    assert_eq!(sess.turn_events().len(), 1);
    // No samples landed on the sink even though TTS was called'd
    // have run if play_audio were true.
    assert_eq!(sink.recorded.len(), 0);
    assert_eq!(sink.write_count, 0);
  }

  /// `run()` over a mic iterator: feed N frames, get one turn
  /// finalized + sink flushed.
  #[test]
  fn run_drives_mic_iterator_to_end() {
    let mut sess = test_session();
    let mut sink = MockSink::new();
    let chunk_size = 320;
    let speech_chunk: Vec<f32> = (0..chunk_size).map(|i| 0.3 * ((i as f32).sin())).collect();
    let silence_chunk: Vec<f32> = vec![0.0; chunk_size];
    let mic: Vec<Vec<f32>> = {
      let mut v = Vec::new();
      for _ in 0..5 {
        v.push(speech_chunk.clone());
      }
      for _ in 0..15 {
        v.push(silence_chunk.clone());
      }
      v
    };
    sess.run(mic.into_iter(), &mut sink).unwrap();

    assert_eq!(sess.turn_events().len(), 1);
    assert_eq!(sink.flush_count, 1, "sink flushed exactly once at run-end");
  }

  /// `flush_in_progress_turn` force-finalizes a turn the mic-EOF
  /// cut short.
  #[test]
  fn run_flushes_in_progress_turn_at_mic_eof() {
    let mut sess = test_session();
    let mut sink = MockSink::new();
    let chunk_size = 320;
    let speech_chunk: Vec<f32> = (0..chunk_size).map(|i| 0.3 * ((i as f32).sin())).collect();
    // Mic ends mid-turn (no trailing silence).
    sess
      .run((0..5).map(|_| speech_chunk.clone()), &mut sink)
      .unwrap();
    // Turn still finalized via flush_in_progress_turn.
    assert_eq!(sess.turn_events().len(), 1);
  }

  /// Barge-in fires when chunk energy crosses the threshold + TTS
  /// is playing. The `barge_in_observed` event field is set.
  #[test]
  fn barge_in_observed_when_user_overlaps_tts() {
    // Build a session with a pre-running sink (`is_running = true`)
    // so the very first speech chunk overlaps TTS.
    let mut sess = test_session();
    let mut sink = MockSink::new();
    sink.running = true; // pretend TTS is currently playing

    let chunk_size = 320;
    let speech_chunk: Vec<f32> = (0..chunk_size).map(|i| 0.3 * ((i as f32).sin())).collect();
    let silence_chunk: Vec<f32> = vec![0.0; chunk_size];
    // 1 speech chunk + many silence chunks → finalize turn with
    // `barge_in_observed = true`.
    sess
      .step(&speech_chunk, &mut sink, /* tts_playing= */ true)
      .unwrap();
    let mut silence_frame = Vec::new();
    for _ in 0..12 {
      silence_frame.extend_from_slice(&silence_chunk);
    }
    sess
      .step(&silence_frame, &mut sink, /* tts_playing= */ false)
      .unwrap();

    assert_eq!(sess.turn_events().len(), 1);
    assert!(sess.turn_events()[0].barge_in_observed);
  }

  /// Sink that always returns `Ok(0)` from `write_samples` →
  /// orchestrator surfaces a `Backend` error rather than spin.
  #[test]
  fn write_samples_zero_is_backend_error() {
    struct BadSink;
    impl AudioOutputStream for BadSink {
      fn write_samples(&mut self, _samples: &[f32]) -> Result<usize> {
        Ok(0)
      }
      fn flush(&mut self) -> Result<()> {
        Ok(())
      }
      fn stop(&mut self) -> Result<()> {
        Ok(())
      }
      fn is_running(&self) -> bool {
        false
      }
    }

    let mut sess = test_session();
    let mut sink = BadSink;
    let chunk_size = 320;
    let speech_chunk: Vec<f32> = (0..chunk_size).map(|i| 0.3 * ((i as f32).sin())).collect();
    let silence_chunk: Vec<f32> = vec![0.0; chunk_size];
    sess.step(&speech_chunk, &mut sink, false).unwrap();
    let mut silence_frame = Vec::new();
    for _ in 0..12 {
      silence_frame.extend_from_slice(&silence_chunk);
    }
    let err = sess.step(&silence_frame, &mut sink, false).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(message.contains("audio sink rejected"), "got: {message}")
      }
      other => panic!("expected Backend error, got: {other:?}"),
    }
  }

  /// Config accessor returns the bundled config (parity with
  /// mlx-audio's `pipeline.config: VoicePipelineConfig`).
  #[test]
  fn config_accessor_returns_bundled_config() {
    let sess = test_session();
    let cfg = sess.config();
    assert_eq!(cfg.input_sample_rate, 16_000);
    assert_eq!(cfg.frame_duration_ms, 20);
  }

  // ---------- Fix 1 (HIGH): first speech chunk must not be ----------
  // ---------- duplicated in the turn audio.                ----------
  /// Pre-roll fills with idle silence; the first speech chunk that
  /// follows must be appended to the turn audio EXACTLY ONCE — the
  /// pre-fix idle branch appended the chunk to the pre-roll BEFORE
  /// the VAD branch ran, then the start-of-turn snapshot prepended
  /// the same pre-roll back onto the turn audio AND the speech
  /// branch appended the chunk a second time. STT then saw the
  /// chunk's samples twice (preroll-copy + direct-append).
  ///
  /// We assert via a recording STT that captures the EXACT turn
  /// audio (not just its length) and verifies the speech-chunk's
  /// distinct marker value appears exactly once in the leading
  /// non-silence region.
  #[test]
  fn voice_session_first_speech_chunk_is_not_duplicated_in_turn_audio() {
    // A recording STT that captures the full turn audio buffer
    // (not just its length), so we can count how many times the
    // speech chunk's distinct marker appears.
    struct CapturingStt {
      audio: RefCell<Vec<f32>>,
    }
    impl SttTurnAdapter for CapturingStt {
      fn transcribe_turn(&mut self, turn_audio: &[f32]) -> Result<String> {
        *self.audio.borrow_mut() = turn_audio.to_vec();
        Ok("captured".to_string())
      }
    }

    let config = VoicePipelineConfig {
      input_sample_rate: 16_000,
      frame_duration_ms: 20,
      preroll_ms: 40,
      vad_end_silence_ms: 200,
      turn_max_incomplete_silence_ms: 200,
      ..VoicePipelineConfig::default()
    };
    let chunk_size = 320;
    let mut sess = VoiceSession::new(
      config,
      MockVad { threshold: 0.05 },
      CapturingStt {
        audio: RefCell::new(Vec::new()),
      },
      MockLlm {
        seen: RefCell::new(Vec::new()),
      },
      MockTts {
        seen: RefCell::new(Vec::new()),
        samples_per_word: 1,
      },
      FixedSizeAudioChunker::new(chunk_size),
      EnergyBargeInDetector::default(),
      SilenceTurnTakingPolicy::new(200),
    )
    .unwrap();

    let mut sink = MockSink::new();
    let silence_chunk: Vec<f32> = vec![0.0; chunk_size];
    // SPEECH chunk uses a distinct constant value (0.5) so we can
    // count how many samples in the captured turn audio equal it.
    // RMS = 0.5 ≥ 0.05 → VAD says speech.
    let speech_chunk: Vec<f32> = vec![0.5_f32; chunk_size];
    let silence_for_finalize: Vec<f32> = vec![0.0; chunk_size * 12];

    // Pre-feed 2 silence chunks so pre-roll fills to capacity
    // (640 samples — all 0.0 — so the only 0.5 values that should
    // appear in the captured turn audio come from the SINGLE speech
    // chunk).
    sess.step(&silence_chunk, &mut sink, false).unwrap();
    sess.step(&silence_chunk, &mut sink, false).unwrap();
    sess.step(&speech_chunk, &mut sink, false).unwrap();
    sess.step(&silence_for_finalize, &mut sink, false).unwrap();

    assert_eq!(sess.turn_events().len(), 1);
    let audio = sess.stt.audio.borrow();
    let n_marker = audio.iter().filter(|&&s| s == 0.5).count();
    assert_eq!(
      n_marker, chunk_size,
      "the speech chunk (320 samples of 0.5) must appear EXACTLY \
       ONCE in the turn audio — pre-fix the idle branch would have \
       pushed it into the pre-roll BEFORE the VAD branch ran, \
       then the start-of-turn snapshot would have copied it into \
       the turn audio AGAIN (640 marker samples total)."
    );
  }

  // ---------- Fix 2 (HIGH): EOF flush must drain the chunker -------
  // ---------- residual so the partial chunk is not dropped. --------
  /// `run()` ends mid-turn with a partial chunk in the chunker —
  /// `flush_in_progress_turn` must drain that residual into the turn
  /// audio before `finalize_turn` runs `chunker.reset()` (which would
  /// discard it).
  #[test]
  fn voice_session_run_at_mic_eof_with_partial_chunk_still_finalizes_full_audio() {
    let mut sess = test_session();
    let mut sink = MockSink::new();
    let chunk_size = 320;
    // 5 full speech chunks + 100 trailing speech samples that don't
    // make a complete 320-sample chunk.
    let speech_chunk: Vec<f32> = (0..chunk_size).map(|i| 0.3 * ((i as f32).sin())).collect();
    let partial: Vec<f32> = (0..100)
      .map(|i| 0.3 * ((i as f32 + 1000.0).sin()))
      .collect();

    let mic: Vec<Vec<f32>> = {
      let mut v = Vec::new();
      for _ in 0..5 {
        v.push(speech_chunk.clone());
      }
      v.push(partial);
      v
    };
    sess.run(mic.into_iter(), &mut sink).unwrap();

    assert_eq!(sess.turn_events().len(), 1);
    // STT must see at least the 5 full speech chunks (= 1600 samples)
    // PLUS the 100-sample residual the chunker had buffered at EOF.
    // (Pre-roll is empty because the very first sample was speech;
    // no idle frames fed it.)
    let stt_len = *sess.stt.last_audio_len.borrow();
    assert_eq!(
      stt_len,
      5 * 320 + 100,
      "STT must receive the 5 full speech chunks PLUS the 100 \
       residual samples buffered in the chunker at mic-EOF"
    );
  }

  /// `flush_in_progress_turn` called directly drains the chunker
  /// residual into the turn audio.
  #[test]
  fn voice_session_flush_in_progress_turn_drains_chunker_residual() {
    let mut sess = test_session();
    let mut sink = MockSink::new();
    let chunk_size = 320;
    let speech_chunk: Vec<f32> = (0..chunk_size).map(|i| 0.3 * ((i as f32).sin())).collect();
    let partial: Vec<f32> = (0..50).map(|i| 0.3 * ((i as f32 + 7.0).sin())).collect();

    // 3 full speech chunks + a 50-sample partial → chunker emits 3
    // chunks, retains 50.
    sess.step(&speech_chunk, &mut sink, false).unwrap();
    sess.step(&speech_chunk, &mut sink, false).unwrap();
    sess.step(&speech_chunk, &mut sink, false).unwrap();
    sess.step(&partial, &mut sink, false).unwrap();

    // Chunker residual should be 50; flush drains them.
    assert!(sess.flush_in_progress_turn(&mut sink).unwrap());
    let stt_len = *sess.stt.last_audio_len.borrow();
    assert_eq!(stt_len, 3 * 320 + 50);
  }

  // ---------- Fix 3 (MEDIUM): barge-in observations must not -------
  // ---------- leak from idle noise into a later turn.        --------
  /// Pre-fix: a non-speech "noisy idle" chunk fed while TTS was
  /// playing would set `barge_in_observed = true` — and that flag
  /// would survive into the NEXT turn (cleared only on
  /// `finalize_turn`, so the idle-noise event tagged a completely
  /// unrelated later turn). Post-fix: barge-in only fires for
  /// `is_speech && in_speech`, and is reset at start-of-turn.
  #[test]
  fn voice_session_barge_in_observed_does_not_leak_from_idle_into_later_turn() {
    let mut sess = test_session();
    let mut sink = MockSink::new();
    let chunk_size = 320;
    // The pre-fix bug used non-speech chunks below the VAD threshold
    // but ABOVE the barge-in detector's energy threshold (0.02).
    // RMS of a constant 0.04 signal = 0.04 ≥ 0.02 (barge-in
    // threshold) but < 0.05 (VAD threshold).
    let noisy_idle: Vec<f32> = vec![0.04_f32; chunk_size];
    let silence: Vec<f32> = vec![0.0; chunk_size];

    // Step 1: noisy idle while TTS playing. Pre-fix: would set
    // `barge_in_observed = true` even though no turn is in progress
    // and the VAD did not detect speech. Post-fix: ignored.
    sess
      .step(&noisy_idle, &mut sink, /* tts_playing = */ true)
      .unwrap();
    // Step 2: a stretch of true silence with TTS off.
    sess.step(&silence, &mut sink, false).unwrap();
    // Step 3: TTS stops; user starts speaking (no overlap).
    let speech_frame: Vec<f32> = (0..5 * chunk_size)
      .map(|i| 0.3 * (((i % chunk_size) as f32).sin()))
      .collect();
    sess
      .step(&speech_frame, &mut sink, /* tts_playing = */ false)
      .unwrap();
    // Step 4: a long stretch of silence (with TTS off) → finalize.
    let silence_frame: Vec<f32> = vec![0.0; 12 * chunk_size];
    sess.step(&silence_frame, &mut sink, false).unwrap();

    let events = sess.turn_events();
    assert_eq!(events.len(), 1, "exactly one turn finalized");
    assert!(
      !events[0].barge_in_observed,
      "idle-noise barge-in detection must NOT leak into a later \
       turn — only in-turn speech-while-TTS-playing counts"
    );
  }

  // ---------- Fix 4 (MEDIUM): silence accounting must be per- ------
  // ---------- chunk + sample-rate==0 must be rejected.        ------
  /// `VoiceSession::new` rejects `input_sample_rate == 0` rather
  /// than panicking at the first chunk's `chunk_ms` division.
  #[test]
  fn voice_session_new_rejects_zero_sample_rate() {
    let config = VoicePipelineConfig {
      input_sample_rate: 0,
      frame_duration_ms: 20,
      preroll_ms: 40,
      vad_end_silence_ms: 200,
      turn_max_incomplete_silence_ms: 200,
      ..VoicePipelineConfig::default()
    };
    let chunk_size = 320;
    let result = VoiceSession::new(
      config,
      MockVad { threshold: 0.05 },
      MockStt {
        last_audio_len: RefCell::new(0),
        text: "x".to_string(),
      },
      MockLlm {
        seen: RefCell::new(Vec::new()),
      },
      MockTts {
        seen: RefCell::new(Vec::new()),
        samples_per_word: 1,
      },
      FixedSizeAudioChunker::new(chunk_size),
      EnergyBargeInDetector::default(),
      SilenceTurnTakingPolicy::new(200),
    );
    match result {
      Ok(_) => panic!("expected Err, got Ok"),
      Err(Error::Backend { message }) => assert!(
        message.contains("sample_rate"),
        "expected Backend error mentioning sample_rate, got: {message}"
      ),
      Err(other) => panic!("expected Backend error, got: {other:?}"),
    }
  }

  /// `step()` computes `chunk_ms` PER CHUNK so a variable-frame
  /// chunker accumulates silence-ms faithfully (pre-fix used
  /// `chunks[0].len()` for every chunk in the batch — a chunker
  /// emitting 200-sample + 100-sample frames would attribute the
  /// 100-sample frame's silence the same 12.5 ms the 200-sample
  /// frame deserves, double-counting the second frame's silence).
  ///
  /// To make the per-chunk semantics OBSERVABLE (and so a pre-fix
  /// regression cannot pass by accident — both pre-fix and post-fix
  /// happen to cross a coarse threshold or deliver the same final
  /// audio length), this test installs a recording
  /// [`TurnTakingPolicy`] that captures every `silence_ms` value the
  /// orchestrator passes to it and asserts the FULL SEQUENCE. Under
  /// the fix the sequence is `[20, 30]`; the pre-fix code (which
  /// reused `chunks[0].len()` for every chunk) would produce
  /// `[20, 40]`.
  #[test]
  fn voice_session_silence_accounting_uses_per_chunk_duration_not_first_chunk() {
    // Two speech chunks (320 samples each, 20 ms each) to open the
    // turn, then two SILENCE chunks of DIFFERENT sizes: 320 (20 ms)
    // + 160 (10 ms). Per-chunk silence-ms must be observed as
    // `[20, 30]` (the 30 ms is the 20 ms accumulated from the first
    // silence chunk plus 10 ms from the second). The pre-fix code
    // would have produced `[20, 40]` (re-using chunks[0]'s 320
    // samples / 20 ms for the second silence chunk too).
    //
    // We use a one-shot chunker that emits all four pre-queued
    // chunks in a SINGLE push_samples call (the bug was
    // specifically that the per-call chunk_ms was computed from
    // chunks[0]). Different chunk sizes (320, 320, 320, 160) are
    // intentional — the silence accounting must reflect the real
    // 160-sample / 10 ms second silence chunk, not re-use the
    // 20 ms of chunks[0].
    let speech_chunk: Vec<f32> = (0..320).map(|i| 0.3 * ((i as f32).sin())).collect();
    let big_silence: Vec<f32> = vec![0.0; 320];
    let small_silence: Vec<f32> = vec![0.0; 160];
    struct OneShotChunker {
      chunks: Option<Vec<Vec<f32>>>,
    }
    impl AudioChunker for OneShotChunker {
      fn push_samples(&mut self, _samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        Ok(self.chunks.take().unwrap_or_default())
      }
      fn drain_residual(&mut self) -> Vec<f32> {
        Vec::new()
      }
      fn reset(&mut self) {
        self.chunks = None;
      }
    }

    /// Records every `silence_ms` value the orchestrator passes to
    /// `user_finished`, then defers to a configurable threshold. We
    /// use a high threshold (so neither chunk finalizes the turn)
    /// and assert the recorded SEQUENCE — that is the observable
    /// divergence between per-chunk (fix) and chunks[0] (pre-fix).
    ///
    /// `RefCell` because [`TurnTakingPolicy::user_finished`] takes
    /// `&self` and the test is single-threaded.
    struct RecordingTurnTaking {
      observed_silence_ms: RefCell<Vec<u32>>,
      threshold_ms: u32,
    }
    impl TurnTakingPolicy for RecordingTurnTaking {
      fn user_finished(&self, _recent_audio: &[f32], silence_ms: u32) -> bool {
        self.observed_silence_ms.borrow_mut().push(silence_ms);
        silence_ms >= self.threshold_ms
      }
    }

    let one_shot = OneShotChunker {
      chunks: Some(vec![
        speech_chunk.clone(),
        speech_chunk,
        big_silence,
        small_silence,
      ]),
    };
    let config = VoicePipelineConfig {
      input_sample_rate: 16_000,
      frame_duration_ms: 20,
      preroll_ms: 0, // disable preroll for a clean length check
      vad_end_silence_ms: 200,
      turn_max_incomplete_silence_ms: 200,
      ..VoicePipelineConfig::default()
    };
    let mut sess = VoiceSession::new(
      config,
      MockVad { threshold: 0.05 },
      MockStt {
        last_audio_len: RefCell::new(0),
        text: "x".to_string(),
      },
      MockLlm {
        seen: RefCell::new(Vec::new()),
      },
      MockTts {
        seen: RefCell::new(Vec::new()),
        samples_per_word: 1,
      },
      one_shot,
      EnergyBargeInDetector::default(),
      // Threshold = 100 ms — well above both the fix's max (30) and
      // the pre-fix's max (40), so neither finalizes the turn. The
      // sequence of observed silence_ms values is the WHOLE assertion;
      // by NOT finalizing we keep the test focused on the per-chunk
      // accumulation (no confounding finalize-side effects).
      RecordingTurnTaking {
        observed_silence_ms: RefCell::new(Vec::new()),
        threshold_ms: 100,
      },
    )
    .unwrap();

    let mut sink = MockSink::new();
    sess.step(&[], &mut sink, false).unwrap();
    // No finalize fired (threshold = 100 ms > max observed 30 ms).
    assert_eq!(
      sess.turn_events().len(),
      0,
      "high threshold (100 ms) must NOT finalize on a 30 ms silence run",
    );

    let observed = sess.turn_policy.observed_silence_ms.borrow().clone();
    // CRITICAL: assert the FULL SEQUENCE `[20, 30]`, not just the
    // final value. The pre-fix code would record `[20, 40]` (it
    // re-uses chunks[0].len() = 320 samples / 20 ms for every chunk
    // in the batch, so the 160-sample second silence chunk would be
    // double-counted as another 20 ms instead of its actual 10 ms).
    assert_eq!(
      observed,
      vec![20, 30],
      "per-chunk silence_ms must accumulate as variable-frame durations \
       [20, 30]; pre-fix (chunks[0].len()) would produce [20, 40]; got {observed:?}",
    );
  }

  /// Sanity sibling to the per-chunk-duration test: when EVERY
  /// silence chunk is the same size as chunks[0], the per-chunk and
  /// chunks[0] computations agree — both produce a [20, 40] sequence
  /// over two 320-sample silence chunks. This locks the single-size
  /// case so a future regression that "fixes" the per-chunk path the
  /// other way (e.g., always using a hard-coded frame_duration_ms)
  /// is still caught.
  #[test]
  fn voice_session_silence_accounting_records_uniform_chunk_size_correctly() {
    let speech_chunk: Vec<f32> = (0..320).map(|i| 0.3 * ((i as f32).sin())).collect();
    let silence_chunk: Vec<f32> = vec![0.0; 320];
    struct OneShotChunker {
      chunks: Option<Vec<Vec<f32>>>,
    }
    impl AudioChunker for OneShotChunker {
      fn push_samples(&mut self, _samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        Ok(self.chunks.take().unwrap_or_default())
      }
      fn drain_residual(&mut self) -> Vec<f32> {
        Vec::new()
      }
      fn reset(&mut self) {
        self.chunks = None;
      }
    }
    struct RecordingTurnTaking {
      observed_silence_ms: RefCell<Vec<u32>>,
      threshold_ms: u32,
    }
    impl TurnTakingPolicy for RecordingTurnTaking {
      fn user_finished(&self, _recent_audio: &[f32], silence_ms: u32) -> bool {
        self.observed_silence_ms.borrow_mut().push(silence_ms);
        silence_ms >= self.threshold_ms
      }
    }

    let one_shot = OneShotChunker {
      chunks: Some(vec![
        speech_chunk.clone(),
        speech_chunk,
        silence_chunk.clone(),
        silence_chunk,
      ]),
    };
    let config = VoicePipelineConfig {
      input_sample_rate: 16_000,
      frame_duration_ms: 20,
      preroll_ms: 0,
      vad_end_silence_ms: 200,
      turn_max_incomplete_silence_ms: 200,
      ..VoicePipelineConfig::default()
    };
    let mut sess = VoiceSession::new(
      config,
      MockVad { threshold: 0.05 },
      MockStt {
        last_audio_len: RefCell::new(0),
        text: "x".to_string(),
      },
      MockLlm {
        seen: RefCell::new(Vec::new()),
      },
      MockTts {
        seen: RefCell::new(Vec::new()),
        samples_per_word: 1,
      },
      one_shot,
      EnergyBargeInDetector::default(),
      RecordingTurnTaking {
        observed_silence_ms: RefCell::new(Vec::new()),
        threshold_ms: 100,
      },
    )
    .unwrap();

    let mut sink = MockSink::new();
    sess.step(&[], &mut sink, false).unwrap();
    let observed = sess.turn_policy.observed_silence_ms.borrow().clone();
    // Two 320-sample silence chunks at 16 kHz → 20 ms each, so the
    // sequence is [20, 40] under both the fix and the pre-fix; this
    // test fixes the uniform-chunk case as a sanity sibling.
    assert_eq!(observed, vec![20, 40], "uniform 320-sample silence chunks");
  }
}
