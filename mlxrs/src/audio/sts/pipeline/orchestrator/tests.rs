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
  // 16 kHz, 20 ms chunks = 320 samples; 200 ms silence threshold = 10 chunks.
  let config = VoicePipelineConfig::new()
    .with_input_sample_rate(16_000)
    .with_frame_duration_ms(20)
    .with_preroll_ms(40)
    .with_vad_end_silence_ms(200)
    .with_turn_max_incomplete_silence_ms(200);
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
  assert_eq!(events[0].user_text(), "hello world");
  assert_eq!(events[0].assistant_text(), "re:hello world");
  assert!(!events[0].barge_in_observed());

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
  sess.config = sess.config.with_play_audio(false);
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
  assert!(sess.turn_events()[0].barge_in_observed());
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
    Error::InvariantViolation(p) => {
      let msg = p.to_string();
      assert!(msg.contains("audio sink"), "got: {msg}")
    }
    other => panic!("expected InvariantViolation error, got: {other:?}"),
  }
}

/// Config accessor returns the bundled config (parity with
/// mlx-audio's `pipeline.config: VoicePipelineConfig`).
#[test]
fn config_accessor_returns_bundled_config() {
  let sess = test_session();
  let cfg = sess.config();
  assert_eq!(cfg.input_sample_rate(), 16_000);
  assert_eq!(cfg.frame_duration_ms(), 20);
}

// ---------- first speech chunk must not be ----------
// ---------- duplicated in the turn audio.   ----------
/// Pre-roll fills with idle silence; the first speech chunk that
/// follows must be appended to the turn audio EXACTLY ONCE. An idle
/// branch that appended the chunk to the pre-roll BEFORE
/// the VAD branch ran would double-count it: the start-of-turn snapshot
/// would prepend the same pre-roll back onto the turn audio AND the
/// speech branch would append the chunk a second time, so STT would see
/// the chunk's samples twice (preroll-copy + direct-append).
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

  let config = VoicePipelineConfig::new()
    .with_input_sample_rate(16_000)
    .with_frame_duration_ms(20)
    .with_preroll_ms(40)
    .with_vad_end_silence_ms(200)
    .with_turn_max_incomplete_silence_ms(200);
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
       ONCE in the turn audio — an idle branch that pushed it into \
       the pre-roll BEFORE the VAD branch ran would let \
       the start-of-turn snapshot copy it into \
       the turn audio AGAIN (640 marker samples total)."
  );
}

// ---------- EOF flush must drain the chunker -------
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

// ---------- barge-in observations must not -------
// ---------- leak from idle noise into a later turn. --------
/// A non-speech "noisy idle" chunk fed while TTS is
/// playing must not set `barge_in_observed = true`: such a flag
/// would survive into the NEXT turn (cleared only on
/// `finalize_turn`), tagging a completely unrelated later turn.
/// Barge-in only fires for `is_speech && in_speech`, and is reset at
/// start-of-turn.
#[test]
fn voice_session_barge_in_observed_does_not_leak_from_idle_into_later_turn() {
  let mut sess = test_session();
  let mut sink = MockSink::new();
  let chunk_size = 320;
  // The attack uses non-speech chunks below the VAD threshold
  // but ABOVE the barge-in detector's energy threshold (0.02).
  // RMS of a constant 0.04 signal = 0.04 ≥ 0.02 (barge-in
  // threshold) but < 0.05 (VAD threshold).
  let noisy_idle: Vec<f32> = vec![0.04_f32; chunk_size];
  let silence: Vec<f32> = vec![0.0; chunk_size];

  // Step 1: noisy idle while TTS playing. This must NOT set
  // `barge_in_observed = true` — no turn is in progress
  // and the VAD did not detect speech, so it is ignored.
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
    !events[0].barge_in_observed(),
    "idle-noise barge-in detection must NOT leak into a later \
       turn — only in-turn speech-while-TTS-playing counts"
  );
}

// ---------- silence accounting must be per- ------
// ---------- chunk + sample-rate==0 must be rejected. ------
/// `VoiceSession::new` rejects `input_sample_rate == 0` rather
/// than panicking at the first chunk's `chunk_ms` division.
#[test]
fn voice_session_new_rejects_zero_sample_rate() {
  let config = VoicePipelineConfig::new()
    .with_input_sample_rate(0)
    .with_frame_duration_ms(20)
    .with_preroll_ms(40)
    .with_vad_end_silence_ms(200)
    .with_turn_max_incomplete_silence_ms(200);
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
    Err(Error::InvariantViolation(p)) => {
      let msg = p.to_string();
      assert!(
        msg.contains("sample_rate"),
        "expected InvariantViolation mentioning sample_rate, got: {msg}"
      )
    }
    Err(other) => panic!("expected InvariantViolation error, got: {other:?}"),
  }
}

/// `step()` computes `chunk_ms` PER CHUNK so a variable-frame
/// chunker accumulates silence-ms faithfully (reusing
/// `chunks[0].len()` for every chunk in the batch would be wrong — a
/// chunker emitting 200-sample + 100-sample frames would attribute the
/// 100-sample frame's silence the same 12.5 ms the 200-sample
/// frame deserves, double-counting the second frame's silence).
///
/// To make the per-chunk semantics OBSERVABLE (and so a
/// regression cannot pass by accident — both behaviors
/// happen to cross a coarse threshold or deliver the same final
/// audio length), this test installs a recording
/// [`TurnTakingPolicy`] that captures every `silence_ms` value the
/// orchestrator passes to it and asserts the FULL SEQUENCE. The
/// per-chunk computation yields `[20, 30]`; a `chunks[0].len()`-reusing
/// computation would produce `[20, 40]`.
#[test]
fn voice_session_silence_accounting_uses_per_chunk_duration_not_first_chunk() {
  // Two speech chunks (320 samples each, 20 ms each) to open the
  // turn, then two SILENCE chunks of DIFFERENT sizes: 320 (20 ms)
  // + 160 (10 ms). Per-chunk silence-ms must be observed as
  // `[20, 30]` (the 30 ms is the 20 ms accumulated from the first
  // silence chunk plus 10 ms from the second). A `chunks[0].len()`-reusing
  // computation would produce `[20, 40]` (re-using chunks[0]'s 320
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
  /// divergence between the per-chunk and chunks[0] computations.
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
  let config = VoicePipelineConfig::new()
    .with_input_sample_rate(16_000)
    .with_frame_duration_ms(20)
    .with_preroll_ms(0)
    .with_vad_end_silence_ms(200)
    .with_turn_max_incomplete_silence_ms(200);
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
    // Threshold = 100 ms — well above both the per-chunk max (30) and
    // the chunks[0]-reuse max (40), so neither finalizes the turn. The
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
  // final value. A `chunks[0].len()`-reusing computation would record
  // `[20, 40]` (re-using chunks[0].len() = 320 samples / 20 ms for every
  // chunk in the batch, so the 160-sample second silence chunk would be
  // double-counted as another 20 ms instead of its actual 10 ms).
  assert_eq!(
    observed,
    vec![20, 30],
    "per-chunk silence_ms must accumulate as variable-frame durations \
       [20, 30]; a chunks[0].len() reuse would produce [20, 40]; got {observed:?}",
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
  let config = VoicePipelineConfig::new()
    .with_input_sample_rate(16_000)
    .with_frame_duration_ms(20)
    .with_preroll_ms(0)
    .with_vad_end_silence_ms(200)
    .with_turn_max_incomplete_silence_ms(200);
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
  // sequence is [20, 40] under both the per-chunk and chunks[0]
  // computations; this test pins the uniform-chunk case as a sanity sibling.
  assert_eq!(observed, vec![20, 40], "uniform 320-sample silence chunks");
}
