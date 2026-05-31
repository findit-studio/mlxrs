//! Integration tests for the voice-pipeline orchestration —
//! end-to-end mock-driven shape coverage that exercises the same
//! public surface a downstream caller composes (every type pulled in
//! via `mlxrs::audio::sts::pipeline::*` rather than the crate's
//! internal paths, so a refactor that moves a type silently breaks
//! these tests rather than passing).
//!
//! Mock-driven by design ("NO real device tests
//! required (mock-based primary)"): no concrete VAD /
//! STT / LM / TTS architecture lives in mlxrs, so loading a real
//! model in a test is impossible). The mocks here implement the
//! adapter traits exactly as a real Voxtral / Silero / NeMo / Pocket
//! TTS wrapper in user code would, so the call-sequence shape mlx-
//! audio's `_listener` / `_response_processor` /
//! `_audio_output_processor` fan-out builds is asserted end-to-end.

#![cfg(feature = "audio")]

use std::cell::RefCell;

use mlxrs::{
  audio::{
    playback::AudioOutputStream,
    sts::pipeline::{
      EnergyBargeInDetector, FixedSizeAudioChunker, LatencyProfile, LlmResponderAdapter,
      SilenceTurnTakingPolicy, SttTurnAdapter, TtsStreamAdapter, VadFrameAdapter, VoicePipeline,
      VoicePipelineConfig, VoiceSession,
    },
  },
  error::{Error, Result},
};

/// 16 kHz × 20 ms = 320-sample chunk.
const SR: u32 = 16_000;
const CHUNK_MS: u32 = 20;
const CHUNK_SIZE: usize = ((SR as usize) * (CHUNK_MS as usize)) / 1_000;

/// Energy-RMS VAD mock: a chunk is "speech" iff its RMS amplitude
/// exceeds `threshold`. Mirrors what a Silero VAD adapter would do
/// internally (the python `SileroSpeechGate` thresholds on the
/// model's probability output).
struct EnergyVad {
  threshold: f32,
}
impl VadFrameAdapter for EnergyVad {
  fn is_speech(&mut self, frame: &[f32]) -> Result<bool> {
    Ok(EnergyBargeInDetector::rms(frame) >= self.threshold)
  }
}

/// Recording STT mock: records the turn-audio length each time it
/// is called; returns the canned transcript so the downstream LLM /
/// TTS path is exercised deterministically.
struct RecordingStt {
  canned_text: String,
  audio_lengths_seen: RefCell<Vec<usize>>,
}
impl SttTurnAdapter for RecordingStt {
  fn transcribe_turn(&mut self, turn_audio: &[f32]) -> Result<String> {
    self.audio_lengths_seen.borrow_mut().push(turn_audio.len());
    Ok(self.canned_text.clone())
  }
}

/// Echoing LLM mock: returns a deterministic transformation of the
/// user transcript so the test can assert the LLM saw the STT
/// output (the mlx-audio `LocalLLMResponseEngine` shape).
struct EchoLlm {
  prompts_seen: RefCell<Vec<String>>,
}
impl LlmResponderAdapter for EchoLlm {
  fn respond(&mut self, user_text: &str) -> Result<String> {
    self.prompts_seen.borrow_mut().push(user_text.to_string());
    Ok(format!("you said: {user_text}"))
  }
}

/// Chunking TTS mock: emits the response as ≤ `chunk_samples`
/// chunks; tracks every text it was asked to synthesize. Mirrors
/// the mlx-audio `PocketTTSResponder` streaming-chunk shape.
struct ChunkingTts {
  texts_seen: RefCell<Vec<String>>,
  chunk_samples: usize,
  /// Total samples to emit per text (independent of word count).
  total_samples_per_text: usize,
}
impl TtsStreamAdapter for ChunkingTts {
  fn synthesize_stream<'a>(
    &'a mut self,
    text: &str,
  ) -> Result<Box<dyn Iterator<Item = Result<Vec<f32>>> + 'a>> {
    self.texts_seen.borrow_mut().push(text.to_string());
    let total = self.total_samples_per_text;
    let chunk = self.chunk_samples;
    let mut remaining = total;
    let mut chunks: Vec<Result<Vec<f32>>> = Vec::new();
    while remaining > 0 {
      let n = remaining.min(chunk);
      chunks.push(Ok(vec![0.25_f32; n]));
      remaining -= n;
    }
    Ok(Box::new(chunks.into_iter()))
  }
  fn sample_rate(&self) -> u32 {
    24_000
  }
}

/// Recording audio sink: keeps every sample it was asked to write;
/// flush/stop are counted; `is_running` flips to `true` after the
/// first non-empty write (a coarse approximation of the cpal
/// `AudioPlayer.isPlaying` semantics suitable for unit-style
/// assertions).
struct RecordingSink {
  recorded: Vec<f32>,
  flush_count: usize,
  stop_count: usize,
  write_count: usize,
  running: bool,
}
impl RecordingSink {
  fn new() -> Self {
    Self {
      recorded: Vec::new(),
      flush_count: 0,
      stop_count: 0,
      write_count: 0,
      running: false,
    }
  }
}
impl AudioOutputStream for RecordingSink {
  fn write_samples(&mut self, samples: &[f32]) -> Result<usize> {
    self.recorded.extend_from_slice(samples);
    self.write_count += 1;
    self.running = !samples.is_empty();
    Ok(samples.len())
  }
  fn flush(&mut self) -> Result<()> {
    self.flush_count += 1;
    self.running = false;
    Ok(())
  }
  fn stop(&mut self) -> Result<()> {
    self.stop_count += 1;
    self.running = false;
    Ok(())
  }
  fn is_running(&self) -> bool {
    self.running
  }
}

fn speech_chunk(len: usize) -> Vec<f32> {
  (0..len).map(|i| 0.3 * ((i as f32).sin())).collect()
}

fn silence_chunk(len: usize) -> Vec<f32> {
  vec![0.0; len]
}

fn build_session() -> VoiceSession<
  EnergyVad,
  RecordingStt,
  EchoLlm,
  ChunkingTts,
  FixedSizeAudioChunker,
  EnergyBargeInDetector,
  SilenceTurnTakingPolicy,
> {
  let config = VoicePipelineConfig::new()
    .with_input_sample_rate(SR)
    .with_frame_duration_ms(CHUNK_MS)
    .with_preroll_ms(40)
    .with_vad_end_silence_ms(200)
    .with_turn_max_incomplete_silence_ms(200)
    .with_latency_profile(LatencyProfile::Balanced);
  VoiceSession::new(
    config,
    EnergyVad { threshold: 0.05 },
    RecordingStt {
      canned_text: "hello there".to_string(),
      audio_lengths_seen: RefCell::new(Vec::new()),
    },
    EchoLlm {
      prompts_seen: RefCell::new(Vec::new()),
    },
    ChunkingTts {
      texts_seen: RefCell::new(Vec::new()),
      chunk_samples: 64,
      total_samples_per_text: 200,
    },
    FixedSizeAudioChunker::new(CHUNK_SIZE),
    EnergyBargeInDetector::default(),
    SilenceTurnTakingPolicy::new(200),
  )
  .expect("build_session: non-zero sample rate")
}

/// End-to-end shape check: 5 speech chunks + 12 silence chunks →
/// one finalized turn with STT called once, LLM called with STT
/// output, TTS called with LLM output, sink receives TTS samples.
#[test]
fn voice_session_drives_full_loop_in_order() {
  let mut sess = build_session();
  let mut sink = RecordingSink::new();

  // 5 speech chunks via separate steps so the per-step flow is
  // exercised (not a single fat frame).
  let speech = speech_chunk(CHUNK_SIZE);
  for _ in 0..5 {
    sess.step(&speech, &mut sink, false).expect("step ok");
  }
  // 12 silence chunks → 240 ms silence ≥ 200 ms threshold → finalize.
  let silence = silence_chunk(CHUNK_SIZE);
  for _ in 0..12 {
    sess.step(&silence, &mut sink, false).expect("step ok");
  }

  // Exactly one turn finalized.
  let events = sess.turn_events();
  assert_eq!(events.len(), 1, "expected exactly one turn finalized");
  assert_eq!(events[0].user_text(), "hello there");
  assert_eq!(events[0].assistant_text(), "you said: hello there");
  assert!(!events[0].barge_in_observed());

  // STT was called exactly once with at least the 5 speech chunks of audio.
  assert_eq!(sess.stt().audio_lengths_seen.borrow().len(), 1);
  assert!(sess.stt().audio_lengths_seen.borrow()[0] >= 5 * CHUNK_SIZE);

  // LLM saw the STT text once.
  assert_eq!(
    sess.llm().prompts_seen.borrow().as_slice(),
    &["hello there"]
  );

  // TTS saw the LLM response once.
  assert_eq!(
    sess.tts().texts_seen.borrow().as_slice(),
    &["you said: hello there"]
  );

  // Sink received exactly the TTS-mock's 200 samples per text.
  assert_eq!(sink.recorded.len(), 200);
  // Sink not flushed yet — only `run()` flushes; `step()` does not.
  assert_eq!(sink.flush_count, 0);
}

/// `run()` over a mic iterator: drives the same loop end-to-end and
/// invokes `flush()` exactly once at the end (parity with the
/// "drain barrier" the `AudioOutputStream::flush` contract
/// promises).
#[test]
fn voice_session_run_flushes_sink_at_eof() {
  let mut sess = build_session();
  let mut sink = RecordingSink::new();

  let mic_frames = (0..5)
    .map(|_| speech_chunk(CHUNK_SIZE))
    .chain((0..12).map(|_| silence_chunk(CHUNK_SIZE)));

  sess.run(mic_frames, &mut sink).expect("run ok");

  assert_eq!(sess.turn_events().len(), 1);
  assert_eq!(sink.flush_count, 1);
}

/// `run()` over an empty mic iterator: drives no turns, flushes
/// the sink exactly once.
#[test]
fn voice_session_run_with_empty_mic_only_flushes() {
  let mut sess = build_session();
  let mut sink = RecordingSink::new();
  sess
    .run(std::iter::empty::<Vec<f32>>(), &mut sink)
    .expect("run ok");
  assert!(sess.turn_events().is_empty());
  assert_eq!(sink.flush_count, 1);
  assert_eq!(sink.recorded.len(), 0);
}

/// Mic ends mid-turn (no trailing silence): `run()`'s post-loop
/// `flush_in_progress_turn` still finalizes the in-progress turn.
#[test]
fn voice_session_run_flushes_in_progress_turn_at_mic_eof() {
  let mut sess = build_session();
  let mut sink = RecordingSink::new();

  sess
    .run((0..5).map(|_| speech_chunk(CHUNK_SIZE)), &mut sink)
    .expect("run ok");

  // Even though the mic stream ended mid-speech (no trailing silence),
  // the in-progress turn is force-finalized.
  assert_eq!(sess.turn_events().len(), 1);
}

/// Two back-to-back turns: 5 speech + 12 silence + 5 speech + 12
/// silence → 2 finalized turns.
#[test]
fn voice_session_finalizes_back_to_back_turns() {
  let mut sess = build_session();
  let mut sink = RecordingSink::new();

  let speech = speech_chunk(CHUNK_SIZE);
  let silence = silence_chunk(CHUNK_SIZE);
  for _ in 0..5 {
    sess.step(&speech, &mut sink, false).unwrap();
  }
  for _ in 0..12 {
    sess.step(&silence, &mut sink, false).unwrap();
  }
  for _ in 0..5 {
    sess.step(&speech, &mut sink, false).unwrap();
  }
  for _ in 0..12 {
    sess.step(&silence, &mut sink, false).unwrap();
  }

  let events = sess.turn_events();
  assert_eq!(events.len(), 2, "two turns finalized");
  // Both turns went through every step.
  assert_eq!(sess.llm().prompts_seen.borrow().len(), 2);
  assert_eq!(sess.tts().texts_seen.borrow().len(), 2);
  // Sink saw both turns' TTS output (200 + 200 = 400 samples).
  assert_eq!(sink.recorded.len(), 400);
}

/// `play_audio = false` config is honored: STT + LLM run, but no
/// samples land on the sink.
#[test]
fn voice_session_play_audio_false_skips_tts_writes() {
  // Build a fresh session with play_audio=false (config is immutable
  // post-new — this exercises the `play_audio` knob's pipeline-side
  // effect rather than re-wiring an existing session).
  let cfg = VoicePipelineConfig::new()
    .with_input_sample_rate(SR)
    .with_frame_duration_ms(CHUNK_MS)
    .with_preroll_ms(40)
    .with_vad_end_silence_ms(200)
    .with_turn_max_incomplete_silence_ms(200)
    .with_play_audio(false);
  let mut sess2 = VoiceSession::new(
    cfg,
    EnergyVad { threshold: 0.05 },
    RecordingStt {
      canned_text: "quiet".to_string(),
      audio_lengths_seen: RefCell::new(Vec::new()),
    },
    EchoLlm {
      prompts_seen: RefCell::new(Vec::new()),
    },
    ChunkingTts {
      texts_seen: RefCell::new(Vec::new()),
      chunk_samples: 64,
      total_samples_per_text: 200,
    },
    FixedSizeAudioChunker::new(CHUNK_SIZE),
    EnergyBargeInDetector::default(),
    SilenceTurnTakingPolicy::new(200),
  )
  .expect("build_session: non-zero sample rate");
  let mut sink = RecordingSink::new();

  let speech = speech_chunk(CHUNK_SIZE);
  let silence = silence_chunk(CHUNK_SIZE);
  for _ in 0..5 {
    sess2.step(&speech, &mut sink, false).unwrap();
  }
  for _ in 0..12 {
    sess2.step(&silence, &mut sink, false).unwrap();
  }

  assert_eq!(sess2.turn_events().len(), 1);
  // LLM still ran (play_audio gates only the TTS sink-emit).
  assert_eq!(sess2.llm().prompts_seen.borrow().len(), 1);
  // TTS was NOT called when play_audio=false.
  assert_eq!(sess2.tts().texts_seen.borrow().len(), 0);
  // Sink got no samples.
  assert_eq!(sink.recorded.len(), 0);
}

/// Barge-in observation: feed a speech chunk while sink reports it
/// is running → orchestrator records `barge_in_observed = true` on
/// the resulting turn event.
#[test]
fn voice_session_records_barge_in_when_user_overlaps_tts() {
  let mut sess = build_session();
  let mut sink = RecordingSink::new();
  sink.running = true; // pretend TTS is already playing

  // 1 speech chunk + many silence chunks → finalize turn with
  // barge_in_observed = true.
  let speech = speech_chunk(CHUNK_SIZE);
  let silence = silence_chunk(CHUNK_SIZE);
  sess.step(&speech, &mut sink, true).unwrap();
  for _ in 0..12 {
    sess.step(&silence, &mut sink, false).unwrap();
  }

  let events = sess.turn_events();
  assert_eq!(events.len(), 1);
  assert!(events[0].barge_in_observed());
}

/// The `VoicePipeline` trait `config()` method is callable on the
/// concrete `VoiceSession` via static dispatch — parity with mlx-
/// audio's `pipeline.config: VoicePipelineConfig` attribute.
#[test]
fn voice_pipeline_trait_config_accessor() {
  let sess = build_session();
  let cfg: &VoicePipelineConfig = sess.config();
  assert_eq!(cfg.input_sample_rate(), SR);
  assert_eq!(cfg.frame_duration_ms(), CHUNK_MS);
  assert_eq!(cfg.latency_profile(), LatencyProfile::Balanced);
}

/// Sink that returns `Err` from `write_samples` → orchestrator
/// surfaces the error from `step()` rather than swallowing it.
#[test]
fn voice_session_propagates_sink_write_error() {
  struct ErrSink;
  impl AudioOutputStream for ErrSink {
    fn write_samples(&mut self, _samples: &[f32]) -> Result<usize> {
      Err(Error::Backend("sink-write-failure".into()))
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

  let mut sess = build_session();
  let mut sink = ErrSink;
  let speech = speech_chunk(CHUNK_SIZE);
  let silence = silence_chunk(CHUNK_SIZE);
  for _ in 0..5 {
    sess.step(&speech, &mut sink, false).unwrap();
  }
  let mut err_seen = false;
  for _ in 0..12 {
    match sess.step(&silence, &mut sink, false) {
      Ok(_) => {}
      Err(Error::Backend(message)) => {
        assert!(message.contains("sink-write-failure"), "msg: {message}");
        err_seen = true;
        break;
      }
      Err(other) => panic!("unexpected error: {other:?}"),
    }
  }
  assert!(err_seen, "expected sink-write error to surface");
}

/// `total_chunks_consumed()` increments per VAD-aligned chunk —
/// usable by callers that want to know how many chunks the
/// orchestrator has processed across the lifetime of the session
/// (mlx-audio carries the equivalent counter on
/// `pipeline._listener` internally for verbose logging).
#[test]
fn voice_session_chunks_consumed_increments_per_chunk() {
  let mut sess = build_session();
  let mut sink = RecordingSink::new();
  assert_eq!(sess.total_chunks_consumed(), 0);

  let silence = silence_chunk(CHUNK_SIZE);
  // 3 separate steps × 1 chunk each = 3 total.
  for _ in 0..3 {
    sess.step(&silence, &mut sink, false).unwrap();
  }
  assert_eq!(sess.total_chunks_consumed(), 3);

  // A 2-chunk frame in a single step adds 2 more.
  let big_frame = silence
    .iter()
    .chain(silence.iter())
    .copied()
    .collect::<Vec<_>>();
  sess.step(&big_frame, &mut sink, false).unwrap();
  assert_eq!(sess.total_chunks_consumed(), 5);
}
