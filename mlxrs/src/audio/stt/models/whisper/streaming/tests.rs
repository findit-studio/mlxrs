//! Tests for AlignAtt attention-guided streaming.
//!
//! Three layers:
//! 1. the pure AlignAtt commit/wait policy
//!    ([`super::super::decoding::alignatt_should_commit`]) on synthetic frames —
//!    an independent oracle of the inequality;
//! 2. the alignment-frame computation
//!    ([`super::super::timing::alignatt_frame_attention`]) on synthetic
//!    cross-attention, asserting the argmax frame tracks a controlled peak;
//! 3. the streaming loop over a synthetic multi-chunk input — committed tokens
//!    grow monotonically (no re-emission / rollback), and the `frame_threshold`
//!    knob shifts the commit boundary.

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
    let frames = alignatt_frame_attention(&model, &cross_qk, n_audio_ctx).unwrap();
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
  let frames = alignatt_frame_attention(&model, &cross_qk, n_audio_ctx * 10).unwrap();
  assert_eq!(frames.len(), 3);
  assert!(frames.iter().all(|&f| f < n_audio_ctx));
}

#[test]
fn alignatt_frame_attention_rejects_zero_content_frames() {
  // Zero content frames → no frame to argmax over → typed OutOfRange, not a
  // panic / zero-width slice.
  let model = WhisperModel::from_weights(stream_dims(), stream_weights(13), Dtype::F32).unwrap();
  let cross_qk = synthetic_cross_qk(model.dims().n_text_head(), 2, model.dims().n_audio_ctx(), 5);
  let err = alignatt_frame_attention(&model, &cross_qk, 0).unwrap_err();
  assert!(
    matches!(err, crate::Error::OutOfRange(_)),
    "zero content frames must be OutOfRange, got {err:?}"
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
  // A stream longer than the 30 s window must slide WITHOUT re-decoding and
  // re-emitting the retained overlap. Feed > N_SAMPLES of audio in chunks; once
  // the buffer exceeds the window it slides (the committed audio is dropped past
  // the watermark, the committed tokens are carried as the decode prefix).
  // Assert: (1) a slide actually occurred, (2) the global committed history only
  // ever GROWS, and (3) every committed token's absolute start time is
  // monotonically non-decreasing across the whole stream — the timeline never
  // jumps backward into an already-committed time range (re-emitting the overlap
  // would replay the overlap's earlier absolute times).
  let dir = fresh_dir("stream_slide");
  let tok = write_tokenizer(dir.as_path());
  let wrapper =
    HFTokenizerWrapper::new(&tok, true, 2, Some("en"), WhisperTask::Transcribe).unwrap();
  let model = stream_model(13); // biased to "d" (id 13)

  let opts = StreamingOptions::new()
    .with_language("en")
    .with_frame_threshold(2);
  let mut session = WhisperStreaming::new(&model, wrapper, opts);

  // 11 s per chunk × 4 = 44 s > 30 s, so the window slides on the 3rd and 4th
  // pushes. Each chunk has distinct content so the encoder/attention differ.
  let chunk_samples = 11 * SAMPLE_RATE as usize;
  let mut total_pushed = 0usize;
  let mut all_starts: Vec<f64> = Vec::new();
  let mut all_ends: Vec<f64> = Vec::new();
  let mut prev_history = 0usize;
  let mut slid = false;

  for c in 0..4usize {
    let chunk: Vec<f32> = (0..chunk_samples)
      .map(|i| (((i + c * 13) % (43 + c)) as f32 / 43.0) - 0.5)
      .collect();
    session.push_audio(&chunk).unwrap();
    total_pushed += chunk_samples;
    // A slide caps the buffer at N_SAMPLES even though more has been pushed.
    if total_pushed > N_SAMPLES {
      assert!(
        session.buffered_samples() <= N_SAMPLES,
        "buffer must be capped to N_SAMPLES after a slide (pushed={total_pushed}, buffered={})",
        session.buffered_samples()
      );
      slid = true;
    }
    let is_last = c == 3;
    let step = session.step(is_last).unwrap();

    // The running committed history only grows (no rollback / shrink).
    let now = session.committed_tokens().len();
    assert!(
      now >= prev_history,
      "committed history must not shrink across a slide: {prev_history} -> {now}"
    );
    prev_history = now;

    for t in step.tokens() {
      all_starts.push(t.start());
      all_ends.push(t.end());
    }
  }

  assert!(
    slid,
    "the 44 s stream must have slid the 30 s window at least once"
  );
  assert!(
    all_starts.len() >= 2,
    "the stream must commit several tokens across the slide to exercise dedup (got {})",
    all_starts.len()
  );

  // The committed-token timeline is strictly forward: each token's absolute start
  // is >= the previous token's start, and never regresses below an earlier
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
    // No committed token starts before the furthest already-committed audio END
    // by more than one frame of rounding slack — i.e. the overlap is never
    // re-decoded into an earlier absolute time.
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
