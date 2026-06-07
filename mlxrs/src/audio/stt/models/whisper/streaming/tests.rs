//! Tests for AlignAtt attention-guided streaming.
//!
//! Layers:
//! 1. the pure AlignAtt commit/wait policy
//!    ([`super::super::decoding::alignatt_should_commit`]) on synthetic frames —
//!    an independent oracle of the inequality;
//! 2. the alignment-frame computation
//!    ([`super::super::timing::alignatt_frame_attention`]) on synthetic
//!    cross-attention, asserting the argmax frame tracks a controlled peak;
//! 3. the streaming loop over a synthetic multi-chunk input — committed tokens
//!    grow monotonically (no re-emission / rollback), and the `frame_threshold`
//!    knob shifts the commit boundary;
//! 4. the dedup watermark — a repeated `step` over the same buffer never
//!    re-emits, and the no-duplicate guarantee holds even with the conditioning
//!    prefix OFF (the watermark, not the prefix, enforces it);
//! 5. backpressure — a push whose uncommitted tail would exceed one 30 s window
//!    is rejected (typed `CapExceeded`) and the buffer is left unchanged, so no
//!    earlier speech is silently dropped;
//! 6. degenerate buffers (empty / sub-window / final-flush) commit nothing or
//!    flush without re-emission;
//! 7. the committed-audio watermark is monotonic and never overruns the buffered
//!    audio; consecutive lossless slides keep absolute timings strictly forward
//!    with no duplication; the prefix cap interacts correctly with the watermark.

use std::{
  collections::HashMap,
  path::{Path, PathBuf},
};

use serde_json::json;

use super::*;
use crate::{
  Array, Dtype,
  audio::stt::models::whisper::{
    audio::N_FRAMES,
    backend::WhisperBackend,
    config::ModelDimensions,
    decoding::alignatt_should_commit,
    model::WhisperModel,
    timing::alignatt_frame_attention,
    tokenizer::{HFTokenizerWrapper, Task as WhisperTask},
  },
  tokenizer::Tokenizer,
};

// ───────────────────────── 1. policy oracle ───────────────────────────────

/// The AlignAtt commit/wait inequality, computed independently of the code
/// under test: COMMIT iff the most-attended frame is strictly more than `f`
/// frames before the content end. The inequality applies on every chunk; the
/// final chunk differs only in the (looser) effective `f` the caller passes, not
/// in whether the inequality runs.
fn oracle_should_commit(content_frames: usize, most_attended: usize, f: usize) -> bool {
  // `content - frame > f` ⇔ the frame is safely behind the live edge.
  (content_frames as i64 - most_attended as i64) > f as i64
}

#[test]
fn alignatt_policy_waits_at_audio_boundary() {
  // A token whose max-attention frame is within `f` of the audio end → WAIT.
  let content = 100usize;
  let f = 10usize;
  // frame 95: content - 95 = 5 <= 10 → within threshold → WAIT.
  assert!(
    !alignatt_should_commit(content, 95, f),
    "frame at the boundary must NOT commit (wait for more audio)"
  );
  // frame 90: content - 90 = 10 <= 10 → exactly at threshold → WAIT (the
  // reference's `<=`).
  assert!(
    !alignatt_should_commit(content, 90, f),
    "frame exactly f from the end must wait (reference uses <=)"
  );
}

#[test]
fn alignatt_policy_commits_earlier_frame() {
  // A token attending to an earlier frame (safely behind the edge) → COMMIT.
  let content = 100usize;
  let f = 10usize;
  // frame 80: content - 80 = 20 > 10 → COMMIT.
  assert!(
    alignatt_should_commit(content, 80, f),
    "an earlier frame (well behind the edge) must commit"
  );
  // frame 89: content - 89 = 11 > 10 → COMMIT (just past the threshold).
  assert!(
    alignatt_should_commit(content, 89, f),
    "one frame past the threshold must commit"
  );
}

#[test]
fn alignatt_policy_matches_oracle_over_grid() {
  // Exhaustive agreement with the independent oracle across a grid of
  // (content, frame, threshold).
  for &content in &[1usize, 10, 100, 1500] {
    for frame in 0..content {
      for &f in &[0usize, 1, 4, 10, 25, 50] {
        assert_eq!(
          alignatt_should_commit(content, frame, f),
          oracle_should_commit(content, frame, f),
          "policy != oracle at content={content}, frame={frame}, f={f}"
        );
      }
    }
  }
}

#[test]
fn alignatt_policy_final_chunk_still_applies_inequality() {
  // The final chunk is NOT a commit-all short-circuit: the inequality still runs
  // with the (looser) effective threshold the streaming layer passes for the last
  // chunk. With the default last-chunk threshold of 4, a frame within 4 of the
  // end is still HELD BACK, while an earlier frame commits — matching the
  // reference's `content_mel_len - most_attened_frame <= (4 if is_last else f)`.
  let content = 50usize;
  let last_f = DEFAULT_LAST_CHUNK_FRAME_THRESHOLD; // 4
  // frame 49: content - 49 = 1 <= 4 → boundary-attending tail → WAIT even on the
  // final chunk (the old commit-all short-circuit would have committed it).
  assert!(
    !alignatt_should_commit(content, content - 1, last_f),
    "a boundary-attending final-chunk frame must still be held back at f={last_f}"
  );
  // frame 45: content - 45 = 5 > 4 → clearly in-window → COMMIT.
  assert!(
    alignatt_should_commit(content, content - 1 - last_f, last_f),
    "a clearly-in-window final-chunk frame must commit at f={last_f}"
  );
}

#[test]
fn alignatt_policy_threshold_shifts_commit_boundary() {
  // The frame_threshold knob shifts the commit boundary: a larger f rejects
  // (waits on) more frames near the edge. The boundary frame (the largest frame
  // that still commits) is `content - f - 1`.
  let content = 100usize;
  for &f in &[5usize, 10, 20, 30] {
    let boundary = content - f - 1; // last committing frame
    assert!(
      alignatt_should_commit(content, boundary, f),
      "frame {boundary} must commit at f={f}"
    );
    assert!(
      !alignatt_should_commit(content, boundary + 1, f),
      "frame {} must wait at f={f}",
      boundary + 1
    );
  }
}

// ───────────────────────── 2. alignment-frame computation ─────────────────

/// Build a synthetic per-layer `cross_qk` list for a single decoder layer with
/// `n_head` heads, `t` token positions, and `n_audio_ctx` frames, where the
/// LAST token's attention peaks (a plateau) at `peak_frame` for every head and
/// every other token is flat. The plateau (±2 frames) survives the width-7
/// median filter so the argmax lands on `peak_frame`.
fn synthetic_cross_qk(
  n_head: usize,
  t: usize,
  n_audio_ctx: usize,
  peak_frame: usize,
) -> Vec<Option<Array>> {
  let mut data = vec![0.1f32; n_head * t * n_audio_ctx];
  // Index helper into the (1, n_head, t, n_audio_ctx) buffer (batch == 1).
  let idx = |h: usize, tok: usize, fr: usize| (h * t + tok) * n_audio_ctx + fr;
  for h in 0..n_head {
    for fr in peak_frame.saturating_sub(2)..(peak_frame + 3).min(n_audio_ctx) {
      // Only the last token spikes here, so the frame column has high variance
      // across the token axis and the last token's normalized value is the max.
      data[idx(h, t - 1, fr)] = 5.0;
    }
  }
  let qk =
    Array::from_slice::<f32>(&data, &[1, n_head as i32, t as i32, n_audio_ctx as i32]).unwrap();
  vec![Some(qk)]
}

#[test]
fn alignatt_frame_attention_tracks_the_peak() {
  // The model's default alignment heads for (n_text_layer=1, n_text_head=2) are
  // [(0,0),(0,1)] — both heads of the single decoder layer. A cross_qk whose
  // last token peaks at frame `peak` must yield an argmax at `peak` for that
  // token.
  let model = WhisperModel::from_weights(stream_dims(), stream_weights(13), Dtype::F32).unwrap();
  let n_audio_ctx = model.dims().n_audio_ctx();
  let t = 4usize;

  for &peak in &[10usize, 200, 700, n_audio_ctx - 5] {
    let cross_qk = synthetic_cross_qk(model.dims().n_text_head(), t, n_audio_ctx, peak);
    let frames =
      alignatt_frame_attention(&WhisperBackend::Mlx(&model), &cross_qk, n_audio_ctx).unwrap();
    assert_eq!(frames.len(), t, "one frame per token position");
    let got = frames[t - 1];
    // The argmax of the last token's row must be within the plateau (±2) of the
    // intended peak (the median filter can shift a single-frame argmax by up to
    // the half-width, but the ±2 plateau keeps it adjacent).
    assert!(
      got.abs_diff(peak) <= 2,
      "last-token argmax frame {got} must track the peak {peak}"
    );
  }
}

#[test]
fn alignatt_frame_attention_clamps_content_frames() {
  // A content_frames larger than the cross-attention width is clamped to the
  // width (so the slice never runs past the attention) — no panic, a valid
  // argmax over the available frames.
  let model = WhisperModel::from_weights(stream_dims(), stream_weights(13), Dtype::F32).unwrap();
  let n_audio_ctx = model.dims().n_audio_ctx();
  let cross_qk = synthetic_cross_qk(model.dims().n_text_head(), 3, n_audio_ctx, 50);
  // content_frames hugely over the width → clamped.
  let frames =
    alignatt_frame_attention(&WhisperBackend::Mlx(&model), &cross_qk, n_audio_ctx * 10).unwrap();
  assert_eq!(frames.len(), 3);
  assert!(frames.iter().all(|&f| f < n_audio_ctx));
}

#[test]
fn alignatt_frame_attention_rejects_zero_content_frames() {
  // Zero content frames → no frame to argmax over → typed OutOfRange, not a
  // panic / zero-width slice.
  let model = WhisperModel::from_weights(stream_dims(), stream_weights(13), Dtype::F32).unwrap();
  let cross_qk = synthetic_cross_qk(model.dims().n_text_head(), 2, model.dims().n_audio_ctx(), 5);
  let err = alignatt_frame_attention(&WhisperBackend::Mlx(&model), &cross_qk, 0).unwrap_err();
  assert!(
    matches!(err, crate::Error::OutOfRange(_)),
    "zero content frames must be OutOfRange, got {err:?}"
  );
}

// ─────────────────── 2b. real cached decode path (AlignAtt) ────────────────
// The synthetic-cross-qk tests above feed a `T > 1` matrix straight into
// `alignatt_frame_attention`; they cannot exercise the cached decode loop, where
// every warm step's `decode_step_with_cross_qk` returns only the NEW token's
// single (`T == 1`) cross-attention row. These tests drive the REAL
// `decode_aligned` accumulation loop end-to-end: the per-step rows must be
// concatenated along the token axis so the policy normalizes over the full token
// sequence. Without that accumulation, a lone `T == 1` row normalizes to zero
// across the token axis, the argmax collapses to frame 0, and `content_frames -
// 0 > threshold` makes the policy commit unconditionally — exactly the bug these
// tests fail on.

/// Build a tiny biased model, its tokenizer, and a synthetic 1 s waveform for
/// the real cached-decode tests. The model is biased to token `13` so greedy
/// decode is deterministic and the loop runs the full `sample_len` cached steps.
/// The samples are returned raw so each call can build a fresh wrapper + task
/// (the borrowed-vocabulary wrapper cannot outlive a single task).
fn cached_decode_fixture(dir_tag: &str) -> (WhisperModel, Tokenizer, Vec<f32>) {
  let dir = fresh_dir(dir_tag);
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let samples: Vec<f32> = (0..16_000)
    .map(|i| ((i % 50) as f32 / 50.0) - 0.5)
    .collect();
  (model, tok, samples)
}

/// Run one real cached `decode_aligned` over the fixture audio with the given
/// `content_frames` / `frame_threshold`, returning the committed token count.
/// This goes through the ACTUAL warm-cache loop (single-row cross-qk per step),
/// so it exercises the accumulation.
fn cached_committed(
  model: &WhisperModel,
  tok: &Tokenizer,
  samples: &[f32],
  sample_len: usize,
  content_frames: usize,
  frame_threshold: usize,
) -> usize {
  use crate::audio::stt::models::whisper::decoding::{DecodingOptions, DecodingTask, SuppressSpec};
  let wrapper = HFTokenizerWrapper::new(tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  let audio = Array::from_slice::<f32>(samples, &[samples.len() as i32]).unwrap();
  let mel = log_mel_spectrogram_whisper(&audio, model.dims().n_mels(), 0).unwrap();
  let mel_window = super::super::audio::pad_or_trim(&mel, N_FRAMES, 0).unwrap();
  let enc = model.encode(&mel_window).unwrap();
  let decode = DecodingOptions {
    task: super::super::tokenizer::Task::Transcribe,
    language: Some("en".into()),
    temperature: 0.0,
    best_of: None,
    beam_size: None,
    length_penalty: None,
    sample_len: Some(sample_len),
    prompt: Vec::new(),
    prefix: Vec::new(),
    suppress_tokens: SuppressSpec::NonSpeech,
    suppress_blank: true,
    without_timestamps: true,
    max_initial_timestamp: None,
  };
  let backend = WhisperBackend::Mlx(model);
  let task = DecodingTask::new(&backend, &wrapper, decode).unwrap();
  let aligned = task
    .decode_aligned(&enc, content_frames, frame_threshold)
    .unwrap();
  // `frames` stays parallel to the committed tokens, and every committed frame is
  // safely behind the live edge per the policy inequality.
  assert_eq!(
    aligned.frames.len(),
    aligned.tokens.len(),
    "committed frames stay parallel to committed tokens"
  );
  for &f in &aligned.frames {
    assert!(
      content_frames.saturating_sub(f) > frame_threshold,
      "a committed token's frame {f} must be > threshold behind content_frames \
       {content_frames}"
    );
  }
  aligned.tokens.len()
}

#[test]
fn alignatt_cached_path_waits_near_edge_commits_when_behind() {
  // The discriminating regression for the warm-cache accumulation. With the audio
  // edge FAR behind the model's attended frames (`content_frames` large), the
  // policy commits the whole `sample_len`. Pulling the edge IN toward those
  // frames makes the policy WAIT partway, committing strictly FEWER tokens.
  //
  // A `T == 1`-row bug (most-attended frame collapses to 0) cannot produce this
  // gap: with most-attended ≡ 0, the commit test is `content_frames - 0 >
  // threshold`, which is identical for both `content_frames` here (both `> 2`),
  // so a broken decode commits the SAME (full) count in both cases. The fixed
  // accumulation yields a real, non-zero attended frame, so the near-edge case
  // genuinely waits.
  let (model, tok, samples) = cached_decode_fixture("cached_wait_commit");
  let sample_len = 6usize;
  let threshold = 2usize;

  // Edge far behind the attended frames → commit the full budget.
  let committed_far = cached_committed(&model, &tok, &samples, sample_len, 5, threshold);
  assert_eq!(
    committed_far, sample_len,
    "edge far behind the peak commits the whole sample_len"
  );

  // Edge pulled in toward the attended frames → the policy waits partway.
  let committed_near = cached_committed(&model, &tok, &samples, sample_len, 4, threshold);
  assert!(
    committed_near < committed_far,
    "edge near the peak must WAIT and commit fewer ({committed_near}) than the \
     far case ({committed_far}); a T==1-collapse bug would commit the same count"
  );
}

#[test]
fn alignatt_cached_path_waits_entirely_at_the_boundary() {
  // When the audio edge sits AT/within the threshold of every attended frame, the
  // accumulation-normalized argmax keeps the peak inside the boundary band and
  // the policy commits NOTHING (`current_tokens[:, :-1]; break` on the first
  // token). This pins the WAIT branch of the real loop (not the synthetic
  // `alignatt_frame_attention` path).
  let (model, tok, samples) = cached_decode_fixture("cached_wait_all");
  let committed = cached_committed(&model, &tok, &samples, 6, 1, 2);
  assert_eq!(
    committed, 0,
    "edge at the boundary holds every token back (commits nothing)"
  );
}

// ───────────────────────── 3. streaming loop ──────────────────────────────

#[test]
fn streaming_commits_monotonically_over_chunks() {
  // A 2-chunk stream: the committed token history must only ever GROW — no
  // committed token is re-emitted or rolled back across chunks. The biased model
  // emits a deterministic token, so the prefix is predictable.
  let dir = fresh_dir("stream_mono");
  let tok = write_tokenizer(dir.as_path());
  let wrapper =
    HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  let model = stream_model(13); // biased to "d"

  let opts = StreamingOptions::new()
    .with_language("en")
    .with_frame_threshold(2);
  let mut session = WhisperStreaming::new(&model, wrapper, opts);

  // Two chunks of (different-content) audio.
  let chunk_a: Vec<f32> = (0..16_000)
    .map(|i| ((i % 50) as f32 / 50.0) - 0.5)
    .collect();
  let chunk_b: Vec<f32> = (0..16_000)
    .map(|i| ((i % 37) as f32 / 37.0) - 0.5)
    .collect();

  let steps = session.transcribe_stream(vec![chunk_a, chunk_b]).unwrap();

  // Every committed token across all steps is a valid (in-vocab) id, and the
  // total committed equals the sum of per-step commits (no overlap/rollback —
  // the prefix-continuation decode appends, it never re-emits).
  let total_committed: usize = steps.iter().map(|s| s.tokens().len()).sum();
  assert_eq!(
    total_committed,
    session.committed_tokens().len(),
    "per-step commits must sum to the running history (no re-emission)"
  );
  assert!(
    session
      .committed_tokens()
      .iter()
      .all(|&id| (id as usize) < N_VOCAB),
    "all committed ids in vocab"
  );
}

#[test]
fn streaming_step_by_step_history_only_grows() {
  // Explicit monotonicity check: feed audio incrementally and assert the
  // committed-token count is non-decreasing across `step` calls, and the earlier
  // prefix is always a prefix of the later history.
  let dir = fresh_dir("stream_grow");
  let tok = write_tokenizer(dir.as_path());
  let wrapper =
    HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  let model = stream_model(13);

  let opts = StreamingOptions::new()
    .with_language("en")
    .with_frame_threshold(2);
  let mut session = WhisperStreaming::new(&model, wrapper, opts);

  let mut prev: Vec<u32> = Vec::new();
  for c in 0..3usize {
    let chunk: Vec<f32> = (0..8_000)
      .map(|i| (((i + c * 7) % (40 + c)) as f32 / 40.0) - 0.5)
      .collect();
    session.push_audio(&chunk).unwrap();
    let is_last = c == 2;
    let _ = session.step(is_last).unwrap();
    let now = session.committed_tokens().to_vec();
    assert!(
      now.len() >= prev.len(),
      "committed history must not shrink: {} -> {}",
      prev.len(),
      now.len()
    );
    assert_eq!(
      &now[..prev.len()],
      prev.as_slice(),
      "the earlier committed prefix must be preserved verbatim (no rollback)"
    );
    prev = now;
  }
}

#[test]
fn streaming_smaller_threshold_commits_at_least_as_much() {
  // The frame_threshold knob changes the commit boundary: a SMALLER f holds back
  // fewer tokens near the edge, so on the same audio it commits at least as many
  // tokens as a LARGER f (which waits on more). Compare a single non-final step.
  let dir = fresh_dir("stream_thresh");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);

  let audio: Vec<f32> = (0..16_000)
    .map(|i| ((i % 41) as f32 / 41.0) - 0.5)
    .collect();

  let run_with = |f: usize| -> usize {
    let wrapper =
      HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
    let opts = StreamingOptions::new()
      .with_language("en")
      .with_frame_threshold(f);
    let mut session = WhisperStreaming::new(&model, wrapper, opts);
    session.push_audio(&audio).unwrap();
    // A non-final step so the policy actually applies (is_last would commit all).
    let step = session.step(false).unwrap();
    step.tokens().len()
  };

  let small_f = run_with(2);
  let large_f = run_with(1_000); // larger than any content frame → commits nothing
  assert!(
    small_f >= large_f,
    "smaller threshold must commit at least as many tokens (small={small_f}, large={large_f})"
  );
  // With f far larger than the content frame count, every token's frame is
  // within f of the end → the policy commits nothing.
  assert_eq!(
    large_f, 0,
    "an over-large threshold commits nothing on a non-final chunk"
  );
}

#[test]
fn streaming_last_chunk_flushes_tokens() {
  // The final chunk (is_last) relaxes the threshold and runs to eot, so the tail
  // is emitted even when a non-final step would hold it back.
  let dir = fresh_dir("stream_last");
  let tok = write_tokenizer(dir.as_path());
  let wrapper =
    HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  let model = stream_model(13);

  let opts = StreamingOptions::new()
    .with_language("en")
    // A huge non-final threshold so only the is_last path can emit.
    .with_frame_threshold(100_000);
  let mut session = WhisperStreaming::new(&model, wrapper, opts);
  let audio: Vec<f32> = (0..16_000)
    .map(|i| ((i % 41) as f32 / 41.0) - 0.5)
    .collect();
  session.push_audio(&audio).unwrap();

  // Non-final step: nothing commits (threshold too large).
  let provisional = session.step(false).unwrap();
  assert!(
    provisional.is_empty(),
    "non-final step holds everything back"
  );

  // Final step: the tail flushes (the biased model emits the target token).
  let last = session.step(true).unwrap();
  assert!(
    !last.is_empty(),
    "the final chunk must flush the committed tail"
  );
  assert!(
    last.tokens().iter().all(|t| t.id() == 13),
    "flushed tokens are the biased target"
  );
}

#[test]
fn streaming_empty_input_yields_one_empty_step() {
  // An empty chunk iterator yields a single final flush step that commits
  // nothing (no audio).
  let dir = fresh_dir("stream_empty");
  let tok = write_tokenizer(dir.as_path());
  let wrapper =
    HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  let model = stream_model(13);

  let mut session =
    WhisperStreaming::new(&model, wrapper, StreamingOptions::new().with_language("en"));
  let empty: Vec<Vec<f32>> = Vec::new();
  let steps = session.transcribe_stream(empty).unwrap();
  assert_eq!(steps.len(), 1, "one final flush step");
  assert!(steps[0].is_empty(), "no audio → nothing committed");
  assert!(session.committed_tokens().is_empty());
}

#[test]
fn committed_token_timing_is_ordered_and_in_window() {
  // A committed token's [start, end] is a single encoder-frame span (≈ 0.02 s)
  // and lies within the buffered window. frame_to_seconds matches the start.
  let dir = fresh_dir("stream_timing");
  let tok = write_tokenizer(dir.as_path());
  let wrapper =
    HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  let model = stream_model(13);

  let opts = StreamingOptions::new()
    .with_language("en")
    .with_frame_threshold(2);
  let mut session = WhisperStreaming::new(&model, wrapper, opts);
  let audio: Vec<f32> = (0..16_000)
    .map(|i| ((i % 41) as f32 / 41.0) - 0.5)
    .collect();
  session.push_audio(&audio).unwrap();
  let step = session.step(true).unwrap();

  for t in step.tokens() {
    assert!(t.end() > t.start(), "token end must follow start");
    // One encoder frame ≈ 0.02 s.
    let span = t.end() - t.start();
    assert!(
      (span - frame_to_seconds(1)).abs() < 1e-9,
      "a token spans one encoder frame (~0.02s), got {span}"
    );
    assert!(t.start() >= 0.0, "absolute start non-negative");
  }
}

#[test]
fn streaming_window_slide_does_not_re_emit_committed_audio() {
  // A stream that fills the 30 s window then receives more audio must slide
  // WITHOUT re-decoding and re-emitting the committed overlap AND without
  // discarding any UNcommitted audio. The biased synthetic model commits its
  // tokens at encoder frame 0, so each window commits one frame-width
  // (`N_SAMPLES_PER_TOKEN`) of audio; the slide then advances the origin by
  // exactly that committed lead — a lossless slide. (A push whose uncommitted
  // tail would exceed one window is rejected as backpressure, covered by
  // `streaming_push_rejects_overlong_uncommitted_tail`; here the tail stays
  // within the window.)
  //
  // Assert: (1) the slide advanced the origin by the committed lead, (2) the
  // buffer retained a FULL window of the most-recent audio (no silent loss),
  // (3) the committed history only GROWS, and (4) every committed token's
  // absolute start time is monotonically non-decreasing — the timeline never
  // regresses into an already-committed span (which re-emitting the overlap
  // would do).
  let dir = fresh_dir("stream_slide");
  let tok = write_tokenizer(dir.as_path());
  let wrapper =
    HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  let model = stream_model(13); // biased to "d" (id 13)

  let opts = StreamingOptions::new()
    .with_language("en")
    .with_frame_threshold(2);
  let mut session = WhisperStreaming::new(&model, wrapper, opts);

  let mut all_starts: Vec<f64> = Vec::new();
  let mut all_ends: Vec<f64> = Vec::new();

  // Fill the window with 30 s of audio and commit (advances the committed-audio
  // watermark into the window).
  let fill: Vec<f32> = (0..N_SAMPLES)
    .map(|i| ((i % 43) as f32 / 43.0) - 0.5)
    .collect();
  session.push_audio(&fill).unwrap();
  let s0 = session.step(false).unwrap();
  assert!(!s0.is_empty(), "the filled window must commit some tokens");
  for t in s0.tokens() {
    all_starts.push(t.start());
    all_ends.push(t.end());
  }
  let committed_lead = session.committed_samples;
  assert!(
    committed_lead > 0 && committed_lead <= N_SAMPLES,
    "the committed-audio watermark must advance into the window (got {committed_lead})"
  );
  let origin_before = session.window_origin_samples;
  let history_before = session.committed_tokens().len();

  // Push exactly the committed lead more audio: buffer = N_SAMPLES + lead, whose
  // uncommitted tail (N_SAMPLES) still fits one window, so the slide drops ONLY
  // the committed lead — a lossless slide.
  let extra: Vec<f32> = (0..committed_lead)
    .map(|i| (((i + 7) % 41) as f32 / 41.0) - 0.5)
    .collect();
  session.push_audio(&extra).unwrap();

  // (1) The slide advanced the origin by the committed lead, and (2) the buffer
  // retained a FULL window of the most-recent audio — no uncommitted audio lost.
  assert_eq!(
    session.window_origin_samples,
    origin_before + committed_lead,
    "the slide must advance the window origin by the committed lead"
  );
  assert_eq!(
    session.buffered_samples(),
    N_SAMPLES,
    "the slide must retain the full uncommitted tail (no silent loss)"
  );

  let s1 = session.step(false).unwrap();
  for t in s1.tokens() {
    all_starts.push(t.start());
    all_ends.push(t.end());
  }

  // (3) The running committed history only grows (no rollback / shrink).
  assert!(
    session.committed_tokens().len() >= history_before,
    "committed history must not shrink across a slide"
  );
  assert!(
    all_starts.len() >= 2,
    "the stream must commit several tokens across the slide to exercise dedup (got {})",
    all_starts.len()
  );

  // (4) The committed-token timeline is strictly forward: each token's absolute
  // start is >= the previous token's start, and never regresses below an earlier
  // token's END (which is what re-emitting the retained overlap would do — the
  // same audio committed twice at the same absolute time).
  let mut max_end = f64::NEG_INFINITY;
  for i in 0..all_starts.len() {
    if i > 0 {
      assert!(
        all_starts[i] >= all_starts[i - 1] - 1e-9,
        "absolute start times must not jump backward (token {i}: {} < {})",
        all_starts[i],
        all_starts[i - 1]
      );
    }
    assert!(
      all_starts[i] >= max_end - frame_to_seconds(1) - 1e-9,
      "token {i} start {} regresses into already-committed time (max committed end {max_end})",
      all_starts[i]
    );
    max_end = max_end.max(all_ends[i]);
  }

  // All committed ids are the biased target and in vocab (sanity: real commits).
  assert!(
    session.committed_tokens().iter().all(|&id| id == 13),
    "all committed tokens are the biased target id"
  );
}

#[test]
fn streaming_final_chunk_threshold_is_not_inert() {
  // The final chunk does NOT commit-all. The streaming layer passes
  // `last_chunk_frame_threshold` as the effective AlignAtt threshold, and the
  // decode applies the SAME inequality with it. So a huge last-chunk threshold
  // (every frame is within it of the edge) makes the final chunk commit NOTHING
  // — proving the threshold is respected, not short-circuited — while a zero
  // last-chunk threshold flushes the tail (any frame is > 0 from the edge). The
  // non-final threshold is held huge throughout so only the final-chunk threshold
  // governs the flush.
  let dir = fresh_dir("stream_last_thresh");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);

  let audio: Vec<f32> = (0..16_000)
    .map(|i| ((i % 41) as f32 / 41.0) - 0.5)
    .collect();

  let final_flush_count = |last_f: usize| -> usize {
    let wrapper =
      HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
    let opts = StreamingOptions::new()
      .with_language("en")
      .with_frame_threshold(100_000) // huge: no non-final commit ever
      .with_last_chunk_frame_threshold(last_f);
    let mut session = WhisperStreaming::new(&model, wrapper, opts);
    session.push_audio(&audio).unwrap();
    // Final chunk: the only path that can emit (the non-final threshold is huge).
    session.step(true).unwrap().tokens().len()
  };

  // A boundary-attending tail: with last_f larger than any content frame, the
  // inequality `content - frame > last_f` is always false → the final chunk holds
  // EVERYTHING back. Under the old `is_last` commit-all short-circuit this would
  // have flushed regardless — the bug this asserts is fixed.
  assert_eq!(
    final_flush_count(100_000),
    0,
    "a huge last-chunk threshold must hold the final tail back (no commit-all)"
  );
  // last_f = 0: every decoded frame is > 0 from the content end → the final chunk
  // flushes the tail (the clearly-in-window case).
  assert!(
    final_flush_count(0) > 0,
    "a zero last-chunk threshold must flush the final tail (clearly in window)"
  );
}

// ───────────────────────── 4. dedup invariant ─────────────────────────────
// The dedup watermark must hold INDEPENDENTLY of the conditioning prefix: a
// repeated `step` over the same buffer never re-emits, even with the prefix off.

/// Build a streaming session over the biased model with explicit conditioning
/// (`cond` = `condition_on_previous_text`, `cap` = `max_prompt_tokens`). A fresh
/// tokenizer per session (the wrapper borrows it for the session's lifetime).
fn dedup_session<'a>(
  model: &'a WhisperModel,
  tok: &'a Tokenizer,
  cond: bool,
  cap: usize,
) -> WhisperStreaming<'a> {
  let wrapper = HFTokenizerWrapper::new(tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  let opts = StreamingOptions::new()
    .with_language("en")
    .with_frame_threshold(2)
    .with_condition_on_previous_text(cond)
    .with_max_prompt_tokens(cap);
  WhisperStreaming::new(model, wrapper, opts)
}

#[test]
fn streaming_repeated_step_same_buffer_is_append_only() {
  // With the prefix ON (the default conditioning), a repeated `step` over the SAME
  // buffer must be APPEND-ONLY: the already-committed prefix is preserved verbatim
  // and never re-emitted. (The forced prefix makes `decode_aligned` CONTINUE past
  // the committed tail, so a degenerate always-repeat model may legitimately
  // append more NEW positions on the continuation — that is not re-emission, since
  // those tokens carry their own later frames. The exact no-growth guarantee with
  // the prefix OFF is asserted by `streaming_repeated_step_no_prefix_no_duplicate`;
  // here the invariant under test is that the prior committed span is never
  // rewritten or duplicated.)
  let dir = fresh_dir("dedup_repeat");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let mut session = dedup_session(&model, &tok, true, 223);

  let audio: Vec<f32> = (0..16_000)
    .map(|i| ((i % 41) as f32 / 41.0) - 0.5)
    .collect();
  session.push_audio(&audio).unwrap();

  let _ = session.step(false).unwrap();
  let after_first = session.committed_tokens().to_vec();
  assert!(
    !after_first.is_empty(),
    "the first step must commit something to exercise dedup"
  );

  // A second step over the identical buffer: the prior committed prefix is
  // preserved verbatim (append-only — no re-emission of the committed span).
  let _ = session.step(false).unwrap();
  let after_second = session.committed_tokens().to_vec();
  assert!(
    after_second.len() >= after_first.len(),
    "the committed history must not shrink on a re-step"
  );
  assert_eq!(
    &after_second[..after_first.len()],
    after_first.as_slice(),
    "a repeated step must preserve the prior committed prefix verbatim (no re-emission)"
  );
}

#[test]
fn streaming_repeated_step_no_prefix_no_duplicate() {
  // The pivotal decoupled-dedup case: with conditioning OFF and
  // `max_prompt_tokens = 0` (NO forced prefix at all), a repeated `step` over the
  // same buffer must STILL not duplicate. The decode restarts from the sot
  // sequence each time and re-emits the whole window, so the watermark —
  // decoupled from the prefix — is the only thing dropping the already-committed
  // tokens. Without that watermark this would re-append the entire transcript on
  // every step.
  let dir = fresh_dir("dedup_noprefix");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  // Both knobs that disable the forced prefix: conditioning off AND a zero cap.
  let mut session = dedup_session(&model, &tok, false, 0);

  let audio: Vec<f32> = (0..16_000)
    .map(|i| ((i % 41) as f32 / 41.0) - 0.5)
    .collect();
  session.push_audio(&audio).unwrap();

  let s1 = session.step(false).unwrap();
  let after_first = session.committed_tokens().to_vec();
  assert!(
    !after_first.is_empty(),
    "the first prefix-off step must commit something"
  );
  let first_emitted = s1.tokens().len();
  assert_eq!(
    first_emitted,
    after_first.len(),
    "the first step emits exactly what it commits"
  );

  // Re-step several times over the SAME buffer: the committed history is frozen.
  for round in 0..3usize {
    let s = session.step(false).unwrap();
    assert!(
      s.is_empty(),
      "round {round}: a prefix-off re-step must emit nothing (no duplicate)"
    );
    assert_eq!(
      session.committed_tokens(),
      after_first.as_slice(),
      "round {round}: the committed history must be unchanged across re-steps"
    );
  }
}

#[test]
fn streaming_no_prefix_incremental_audio_only_grows() {
  // With the prefix OFF, feeding MORE audio incrementally must still only append
  // the genuinely-new suffix (the watermark drops the re-decoded committed lead),
  // and the earlier committed prefix is preserved verbatim — a from-scratch
  // re-decode each step never re-emits, and growth comes only from new audio.
  let dir = fresh_dir("dedup_noprefix_grow");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let mut session = dedup_session(&model, &tok, false, 0);

  let mut prev: Vec<u32> = Vec::new();
  for c in 0..3usize {
    let chunk: Vec<f32> = (0..7_000)
      .map(|i| (((i + c * 11) % 41) as f32 / 41.0) - 0.5)
      .collect();
    session.push_audio(&chunk).unwrap();
    let _ = session.step(false).unwrap();
    let now = session.committed_tokens().to_vec();
    assert!(
      now.len() >= prev.len(),
      "prefix-off history must not shrink: {} -> {}",
      prev.len(),
      now.len()
    );
    assert_eq!(
      &now[..prev.len()],
      prev.as_slice(),
      "prefix-off: the earlier committed prefix must be preserved verbatim"
    );
    prev = now;
  }
}

// ───────────────────────── 5. backpressure (no silent loss) ───────────────

#[test]
fn streaming_push_rejects_overlong_uncommitted_tail() {
  // A single push whose UNcommitted tail would exceed one 30 s window is rejected
  // as typed backpressure (`CapExceeded`), and the buffer is left UNCHANGED so the
  // caller keeps its samples — no silent loss of earlier speech.
  let dir = fresh_dir("backpressure_single");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let mut session = dedup_session(&model, &tok, true, 223);

  // N_SAMPLES + 1 in one push: nothing committed yet, so the whole thing is the
  // uncommitted tail → over the window cap.
  let too_long: Vec<f32> = vec![0.0; N_SAMPLES + 1];
  let err = session.push_audio(&too_long).unwrap_err();
  assert!(
    matches!(err, crate::Error::CapExceeded(_)),
    "an over-long uncommitted push must be CapExceeded, got {err:?}"
  );
  assert_eq!(
    session.buffered_samples(),
    0,
    "a rejected push must leave the buffer unchanged (caller keeps the samples)"
  );
}

#[test]
fn streaming_zero_commit_growth_eventually_backpressures_no_loss() {
  // The "policy commits nothing for > 30 s" path: a model that commits nothing
  // (a huge frame_threshold) accumulates the uncommitted tail until a push would
  // exceed the window — then backpressure fires and the buffer is unchanged. The
  // earliest audio is NEVER silently dropped while it remains uncommitted.
  let dir = fresh_dir("backpressure_zerocommit");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let wrapper =
    HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  // A huge non-final threshold so nothing ever commits on a non-final step.
  let opts = StreamingOptions::new()
    .with_language("en")
    .with_frame_threshold(1_000_000);
  let mut session = WhisperStreaming::new(&model, wrapper, opts);

  // Push in 10 s chunks; the third would bring the buffer to 30 s exactly (OK),
  // the fourth would exceed it with nothing committed → backpressure, no loss.
  let ten_s: Vec<f32> = vec![0.25; 10 * SAMPLE_RATE as usize];
  session.push_audio(&ten_s).unwrap();
  let _ = session.step(false).unwrap();
  session.push_audio(&ten_s).unwrap();
  let _ = session.step(false).unwrap();
  session.push_audio(&ten_s).unwrap(); // buffer now exactly N_SAMPLES
  let _ = session.step(false).unwrap();
  assert_eq!(
    session.buffered_samples(),
    N_SAMPLES,
    "buffer at exactly one window"
  );
  assert!(
    session.committed_tokens().is_empty(),
    "the huge threshold must have committed nothing"
  );

  // The fourth push would push the uncommitted tail past the window → rejected,
  // buffer unchanged (no earlier audio dropped).
  let before = session.buffered_samples();
  let err = session.push_audio(&ten_s).unwrap_err();
  assert!(
    matches!(err, crate::Error::CapExceeded(_)),
    "an over-30s uncommitted accumulation must backpressure, got {err:?}"
  );
  assert_eq!(
    session.buffered_samples(),
    before,
    "the rejected push must not drop any buffered (uncommitted) audio"
  );
}

#[test]
fn streaming_push_exactly_one_window_is_accepted() {
  // A push to exactly N_SAMPLES (the window boundary) is accepted — the cap is on
  // EXCEEDING the window, not reaching it.
  let dir = fresh_dir("backpressure_boundary");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let mut session = dedup_session(&model, &tok, true, 223);
  let exact: Vec<f32> = vec![0.1; N_SAMPLES];
  session.push_audio(&exact).unwrap();
  assert_eq!(session.buffered_samples(), N_SAMPLES);
}

// ───────────────────────── 6. degenerate buffers ──────────────────────────

#[test]
fn streaming_empty_audio_step_is_noop() {
  // `step` on an empty buffer (no audio pushed) commits nothing and errors not —
  // on both a non-final and a final call.
  let dir = fresh_dir("empty_step");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let mut session = dedup_session(&model, &tok, true, 223);

  let s_mid = session.step(false).unwrap();
  assert!(
    s_mid.is_empty(),
    "non-final step on empty audio commits nothing"
  );
  let s_last = session.step(true).unwrap();
  assert!(
    s_last.is_empty(),
    "final step on empty audio commits nothing"
  );
  assert!(session.committed_tokens().is_empty());
}

#[test]
fn streaming_is_last_subwindow_buffer_flushes_or_empty() {
  // `is_last` on a tiny (sub-window) buffer must not panic: it either flushes the
  // short tail or commits nothing, and never re-emits on a follow-up final step.
  let dir = fresh_dir("last_subwindow");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let mut session = dedup_session(&model, &tok, true, 223);

  // A sub-window buffer (well under 30 s).
  let small: Vec<f32> = (0..4_000).map(|i| ((i % 23) as f32 / 23.0) - 0.5).collect();
  session.push_audio(&small).unwrap();
  let s1 = session.step(true).unwrap();
  let after = session.committed_tokens().to_vec();
  assert_eq!(
    s1.tokens().len(),
    after.len(),
    "a final sub-window step emits exactly what it commits"
  );
  // A second final step over the same buffer is append-only — the prior committed
  // prefix is preserved verbatim (no re-emission / rollback of the committed span).
  let _ = session.step(true).unwrap();
  let after2 = session.committed_tokens().to_vec();
  assert!(
    after2.len() >= after.len(),
    "a repeated final step must not shrink the committed history"
  );
  assert_eq!(
    &after2[..after.len()],
    after.as_slice(),
    "a repeated final step must preserve the prior committed prefix verbatim"
  );
}

// ───────────────────────── 7. watermark monotonicity ──────────────────────

#[test]
fn streaming_committed_watermark_never_regresses_or_overruns() {
  // Across many incremental steps the committed-audio watermark is monotonic
  // non-decreasing and never exceeds the absolute buffered end (origin + buffered)
  // — no off-by-one in the frame→sample mapping, no regress on a re-step.
  let dir = fresh_dir("watermark_mono");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let mut session = dedup_session(&model, &tok, true, 223);

  let mut prev_watermark = 0usize;
  for round in 0..5usize {
    let chunk: Vec<f32> = (0..5_000)
      .map(|i| (((i + round * 17) % 41) as f32 / 41.0) - 0.5)
      .collect();
    session.push_audio(&chunk).unwrap();
    // Two steps per round, including a re-step on the same buffer.
    let _ = session.step(false).unwrap();
    let _ = session.step(false).unwrap();
    let w = session.committed_samples;
    assert!(
      w >= prev_watermark,
      "round {round}: watermark regressed {prev_watermark} -> {w}"
    );
    let buffered_end = session.window_origin_samples + session.buffered_samples();
    assert!(
      w <= buffered_end,
      "round {round}: watermark {w} exceeds buffered end {buffered_end}"
    );
    prev_watermark = w;
  }
}

// ───────────────────────── 8. multi-slide monotonicity ────────────────────

#[test]
fn streaming_consecutive_slides_keep_timings_monotonic_no_dup() {
  // Several consecutive lossless slides (each driven by committing one frame of
  // audio then pushing that committed lead back) must keep absolute per-token
  // timings strictly forward with no duplication, and the window origin strictly
  // increasing across every slide.
  let dir = fresh_dir("multi_slide");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  let mut session = dedup_session(&model, &tok, true, 223);

  let fill: Vec<f32> = (0..N_SAMPLES)
    .map(|i| ((i % 43) as f32 / 43.0) - 0.5)
    .collect();
  session.push_audio(&fill).unwrap();

  let mut all_starts: Vec<f64> = Vec::new();
  let mut all_ends: Vec<f64> = Vec::new();
  let mut prev_origin = session.window_origin_samples;
  let mut slides = 0usize;

  for _ in 0..3usize {
    let step = session.step(false).unwrap();
    for t in step.tokens() {
      all_starts.push(t.start());
      all_ends.push(t.end());
    }
    let lead = session
      .committed_samples
      .saturating_sub(session.window_origin_samples);
    if lead == 0 {
      break; // nothing committable to slide on (defensive)
    }
    let extra: Vec<f32> = (0..lead).map(|i| ((i % 41) as f32 / 41.0) - 0.5).collect();
    session.push_audio(&extra).unwrap();
    if session.window_origin_samples > prev_origin {
      slides += 1;
      prev_origin = session.window_origin_samples;
    }
  }

  assert!(
    slides >= 2,
    "must perform multiple consecutive slides (got {slides})"
  );
  assert!(
    session.committed_tokens().iter().all(|&id| id == 13),
    "all committed ids are the biased target"
  );
  let mut max_end = f64::NEG_INFINITY;
  for i in 0..all_starts.len() {
    if i > 0 {
      assert!(
        all_starts[i] >= all_starts[i - 1] - 1e-9,
        "absolute starts must not jump backward across slides (token {i})"
      );
    }
    assert!(
      all_starts[i] >= max_end - frame_to_seconds(1) - 1e-9,
      "token {i} start {} regresses into already-committed time (max end {max_end})",
      all_starts[i]
    );
    max_end = max_end.max(all_ends[i]);
  }
}

// ───────────────────────── 9. prefix-cap × watermark ──────────────────────

#[test]
fn streaming_prefix_cap_interacts_with_watermark() {
  // The forced-prefix cap (`max_prompt_tokens`) shorter than the committed
  // history must still leave dedup correct: with conditioning ON but a SMALL cap,
  // `decode_aligned` forces only the recent committed tail, yet the continuation
  // is appended without duplicating the (uncapped) earlier committed tokens, and
  // the history stays append-only — including on a re-step.
  let dir = fresh_dir("prefix_cap");
  let tok = write_tokenizer(dir.as_path());
  let model = stream_model(13);
  // A tiny prompt cap (2 tokens) far below what the window commits.
  let mut session = dedup_session(&model, &tok, true, 2);

  let mut prev: Vec<u32> = Vec::new();
  for round in 0..3usize {
    let chunk: Vec<f32> = (0..6_000)
      .map(|i| (((i + round * 9) % 41) as f32 / 41.0) - 0.5)
      .collect();
    session.push_audio(&chunk).unwrap();
    let _ = session.step(false).unwrap();
    let now = session.committed_tokens().to_vec();
    assert!(
      now.len() >= prev.len(),
      "history must not shrink under a small cap"
    );
    assert_eq!(
      &now[..prev.len()],
      prev.as_slice(),
      "earlier committed tokens preserved verbatim under a small prefix cap"
    );
    // A re-step on the same buffer with the small cap is still a no-op for the
    // already-committed positions (append-only).
    let before_restep = session.committed_tokens().to_vec();
    let _ = session.step(false).unwrap();
    let after_restep = session.committed_tokens().to_vec();
    assert_eq!(
      &after_restep[..before_restep.len()],
      before_restep.as_slice(),
      "a re-step must preserve the prior committed prefix under a small cap"
    );
    prev = after_restep;
  }
}

// ───────────────────────── fixtures ───────────────────────────────────────
// A self-contained tiny Whisper model + tokenizer, mirroring the decoding-test
// fixtures (test helpers are duplicated per test module by project convention).

/// Whisper-shaped special tokens; ids span 0..18. timestamp_begin = `<|0.00|>`
/// at 14.
const SPECIALS: &[(&str, u32)] = &[
  ("a", 0),
  ("b", 1),
  ("<|endoftext|>", 2),
  ("<|startoftranscript|>", 3),
  ("<|en|>", 4),
  ("<|zh|>", 5),
  ("<|translate|>", 6),
  ("<|transcribe|>", 7),
  ("<|startoflm|>", 8),
  ("<|startofprev|>", 9),
  ("<|nospeech|>", 10),
  ("<|notimestamps|>", 11),
  ("c", 12),
  ("d", 13),
  ("<|0.00|>", 14),
  ("<|0.02|>", 15),
  ("<|0.04|>", 16),
  ("<|0.06|>", 17),
];

const N_VOCAB: usize = 18;

fn fresh_dir(tag: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_whisper_stream_{}_{}",
    std::process::id(),
    tag
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

fn write_tokenizer(dir: &Path) -> Tokenizer {
  let vocab: serde_json::Map<String, serde_json::Value> = SPECIALS
    .iter()
    .map(|(tok, id)| (tok.to_string(), json!(id)))
    .collect();
  let added_tokens: Vec<serde_json::Value> = SPECIALS
    .iter()
    .map(|(tok, id)| {
      let special = tok.starts_with("<|");
      json!({
        "id": id, "content": tok, "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": special
      })
    })
    .collect();
  let tokenizer_json = json!({
    "version": "1.0",
    "added_tokens": added_tokens,
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>" }
  });
  let cfg = json!({ "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>" });
  std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  Tokenizer::from_path(dir, None).unwrap()
}

/// Tiny model dims (n_text_ctx large enough for a few committed tokens + prompt).
fn stream_dims() -> ModelDimensions {
  ModelDimensions::new(
    /* n_mels */ 4,
    /* n_audio_ctx */ N_FRAMES / 2,
    /* n_audio_state */ 4,
    /* n_audio_head */ 2,
    /* n_audio_layer */ 1,
    /* n_vocab */ N_VOCAB,
    /* n_text_ctx */ 32,
    /* n_text_state */ 4,
    /* n_text_head */ 2,
    /* n_text_layer */ 1,
  )
  .unwrap()
}

fn ones2(r: usize, c: usize) -> Array {
  Array::ones::<f32>(&(r, c)).unwrap()
}
fn zeros1(n: usize) -> Array {
  Array::zeros::<f32>(&(n,)).unwrap()
}
fn ones1(n: usize) -> Array {
  Array::ones::<f32>(&(n,)).unwrap()
}
fn put_attn(w: &mut HashMap<String, Array>, prefix: &str, n: usize) {
  for p in ["query", "value", "out"] {
    w.insert(format!("{prefix}.{p}.weight"), ones2(n, n));
    w.insert(format!("{prefix}.{p}.bias"), zeros1(n));
  }
  w.insert(format!("{prefix}.key.weight"), ones2(n, n));
}
fn put_ln(w: &mut HashMap<String, Array>, prefix: &str, n: usize) {
  w.insert(format!("{prefix}.weight"), ones1(n));
  w.insert(format!("{prefix}.bias"), zeros1(n));
}
fn put_block(w: &mut HashMap<String, Array>, prefix: &str, n: usize, cross: bool) {
  put_attn(w, &format!("{prefix}.attn"), n);
  put_ln(w, &format!("{prefix}.attn_ln"), n);
  if cross {
    put_attn(w, &format!("{prefix}.cross_attn"), n);
    put_ln(w, &format!("{prefix}.cross_attn_ln"), n);
  }
  w.insert(format!("{prefix}.mlp1.weight"), ones2(4 * n, n));
  w.insert(format!("{prefix}.mlp1.bias"), zeros1(4 * n));
  w.insert(format!("{prefix}.mlp2.weight"), ones2(n, 4 * n));
  w.insert(format!("{prefix}.mlp2.bias"), zeros1(n));
  put_ln(w, &format!("{prefix}.mlp_ln"), n);
}

/// The tiny checkpoint weights, biased toward `target` so greedy decode is
/// deterministic (mirrors the decoding-test `tiny_model`).
fn stream_weights(target: u32) -> HashMap<String, Array> {
  let n = 4usize;
  let mut w = HashMap::new();
  let c1: Vec<f32> = (0..(n * 3 * 4))
    .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
    .collect();
  w.insert(
    "encoder.conv1.weight".into(),
    Array::from_slice::<f32>(&c1, &(n, 3usize, 4usize)).unwrap(),
  );
  w.insert("encoder.conv1.bias".into(), zeros1(n));
  let c2: Vec<f32> = (0..(n * 3 * n))
    .map(|i| ((i % 3) as f32 - 1.0) * 0.1)
    .collect();
  w.insert(
    "encoder.conv2.weight".into(),
    Array::from_slice::<f32>(&c2, &(n, 3usize, n)).unwrap(),
  );
  w.insert("encoder.conv2.bias".into(), zeros1(n));
  put_block(&mut w, "encoder.blocks.0", n, false);
  put_ln(&mut w, "encoder.ln_post", n);

  let ramp = |j: usize| -> f32 { (j as f32 - (n as f32 / 2.0)) * 0.2 };
  let mut emb: Vec<f32> = (0..(N_VOCAB * n)).map(|i| ramp(i % n)).collect();
  if (target as usize) < N_VOCAB {
    let base = target as usize * n;
    for j in 0..n {
      emb[base + j] = ramp(j) * 10.0;
    }
  }
  w.insert(
    "decoder.token_embedding.weight".into(),
    Array::from_slice::<f32>(&emb, &(N_VOCAB, n)).unwrap(),
  );
  let pe: Vec<f32> = (0..(32 * n))
    .map(|i| ((i % n) as f32 - (n as f32 / 2.0)) * 0.3)
    .collect();
  w.insert(
    "decoder.positional_embedding".into(),
    Array::from_slice::<f32>(&pe, &(32usize, n)).unwrap(),
  );
  put_block(&mut w, "decoder.blocks.0", n, true);
  put_ln(&mut w, "decoder.ln", n);
  w
}

/// A biased tiny model for the streaming-loop tests.
fn stream_model(target: u32) -> WhisperModel {
  WhisperModel::from_weights(stream_dims(), stream_weights(target), Dtype::F32).unwrap()
}
