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
  /// Whether the barge-in detector fired during the current turn.
  barge_in_observed: bool,
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
  #[must_use]
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
  ) -> Self {
    let preroll_samples =
      (config.input_sample_rate as usize) * (config.preroll_ms as usize) / 1_000;
    Self {
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
      barge_in_observed: false,
      total_chunks_consumed: 0,
    }
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

    // Per-chunk chunk_ms = chunk_samples * 1000 / sample_rate.
    let chunk_ms = if !chunks.is_empty() {
      ((chunks[0].len() as u64) * 1_000 / (self.config.input_sample_rate as u64)) as u32
    } else {
      0
    };

    for chunk in chunks {
      self.total_chunks_consumed += 1;
      let is_speech = self.vad.is_speech(&chunk)?;

      // Track pre-roll while idle; once speech starts, accumulate
      // turn audio.
      if !self.in_speech {
        self.preroll.append(&chunk);
      }
      // Barge-in: while TTS is playing, an energy-positive chunk
      // counts as an overlap candidate. We don't act on it here —
      // mlx-audio's confirmation pipeline is out of scope — but
      // record it so a caller can observe it via [`TurnEvent`].
      if self.config.barge_in && self.barge_in.detect(&chunk, tts_playing) {
        self.barge_in_observed = true;
      }

      if is_speech {
        // Start a new turn: drain the pre-roll into the turn
        // audio so the STT sees the leading samples the VAD
        // ran past.
        if !self.in_speech {
          let preroll_snapshot = self.preroll.snapshot();
          self.in_progress_audio.extend_from_slice(&preroll_snapshot);
          self.preroll.clear();
          self.in_speech = true;
        }
        self.in_progress_audio.extend_from_slice(&chunk);
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
      }
    }

    Ok(turns_finalized)
  }

  /// Force-finalize any in-progress turn — called from
  /// [`VoiceSession::run`] when the mic iterator exhausts mid-turn.
  /// A noop when no turn is in progress.
  pub fn flush_in_progress_turn<O: AudioOutputStream>(&mut self, output: &mut O) -> Result<bool> {
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
    let barge_in_observed = std::mem::replace(&mut self.barge_in_observed, false);

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
}
