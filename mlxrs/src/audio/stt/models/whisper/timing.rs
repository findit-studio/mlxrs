//! Word-level timestamps via cross-attention DTW — a faithful port of
//! mlx-audio's `timing.py` (`mlx-source/mlx-audio/mlx_audio/stt/models/whisper/timing.py`).
//!
//! After a segment is decoded, the cross-attention weights of the model's
//! **alignment heads** (`whisper.py:_alignment_heads`) are stacked, normalized
//! (softmax over audio frames, then per-token std-normalization + a median
//! filter), and averaged into a `(tokens, frames)` cost matrix. Dynamic Time
//! Warping (DTW) over `-matrix` then traces the lowest-cost monotonic alignment
//! of text tokens to audio frames, yielding a frame index per token. Tokens are
//! merged into words by the tokenizer's word-boundary logic
//! ([`HFTokenizerWrapper::split_to_word_tokens`]), and each word's start / end
//! frame is mapped to seconds via the audio token rate ([`TOKENS_PER_SECOND`]).
//!
//! The DTW dynamic program and the median filter run on the host (numpy in the
//! reference), driven by the on-device attention statistics; the on-device ops
//! (softmax / mean / std / stack) mirror the reference's `mx.*` calls exactly.

use smol_str::format_smolstr;

use crate::{
  Array, Dtype, Error, Result,
  error::OutOfRangePayload,
  model_validation::{alloc_filled, reserve_or_error},
  ops,
};

use super::{
  audio::{HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND},
  decoding::{Segment, Word},
  model::WhisperModel,
  tokenizer::HFTokenizerWrapper,
};

/// The median-filter window width applied along the frame axis when normalizing
/// the alignment weights — the reference's constant `median_filter_width = 7`
/// (`find_alignment`, `timing.py:147`). Odd and positive by construction, so the
/// median filter's middle-element selection is always well defined.
const MEDIAN_FILTER_WIDTH: usize = 7;

/// The per-word timing produced by the DTW alignment — `WordTiming`
/// (`timing.py:102-108`).
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
  /// The word text (including any leading space / merged punctuation).
  pub word: String,
  /// The token ids that decode to this word.
  pub tokens: Vec<u32>,
  /// The word start time, in seconds (relative to the segment).
  pub start: f64,
  /// The word end time, in seconds (relative to the segment).
  pub end: f64,
  /// The mean per-token probability of the word's tokens.
  pub probability: f64,
}

/// A `(tokens, frames)` host matrix with explicit dimensions — the carrier the
/// DTW / median filter operate on (the reference's numpy arrays).
struct HostMatrix {
  data: Vec<f32>,
  rows: usize,
  cols: usize,
}

impl HostMatrix {
  #[inline(always)]
  fn at(&self, r: usize, c: usize) -> f32 {
    self.data[r * self.cols + c]
  }
}

/// Apply a median filter of width `filter_width` along the **last** dimension
/// of a `(planes, cols)` host buffer — `median_filter` (`timing.py:17-49`).
///
/// The reference reflect-pads by `filter_width // 2` on each side of the last
/// axis, then takes the sliding-window median of every `filter_width` window.
/// When `cols <= pad_width` the input is returned unchanged (the reference's
/// early return). `filter_width` must be odd and positive.
///
/// Operates per plane (each row of length `cols`); the caller flattens the
/// leading dims into `planes`. The `(planes, cols)` output is sized through the
/// checked product + fallible allocation, so an overflow / out-of-memory
/// condition on a large valid alignment surfaces as a typed error.
///
/// # Errors
/// [`Error::OutOfRange`] if `planes * cols` overflows `usize`;
/// [`Error::AllocFailure`](crate::Error::AllocFailure) if the output / window
/// allocation fails.
fn median_filter(
  data: &[f32],
  planes: usize,
  cols: usize,
  filter_width: usize,
) -> Result<Vec<f32>> {
  let pad_width = filter_width / 2;
  // `if x.shape[-1] <= pad_width: return x` (`timing.py:20-21`).
  if cols <= pad_width {
    return alloc_copied("Whisper word timestamps: median-filter passthrough", data);
  }

  let area = checked_area(planes, cols, "median-filter output (planes * cols)")?;
  let mut out = alloc_filled(
    "Whisper word timestamps: median-filter output",
    0.0f32,
    area,
  )?;
  // A reusable window scratch, sorted per output position to take the median.
  let mut window = alloc_filled(
    "Whisper word timestamps: median-filter window",
    0.0f32,
    filter_width,
  )?;
  for p in 0..planes {
    let row = &data[p * cols..(p + 1) * cols];
    for i in 0..cols {
      // The window covers `[i - pad_width, i + pad_width]` over the
      // reflect-padded row; resolve each tap against the unpadded row via the
      // reflect index map (numpy `mode="reflect"`: edge sample not repeated).
      for (k, slot) in window.iter_mut().enumerate() {
        let src = reflect_index(i as isize + k as isize - pad_width as isize, cols);
        *slot = row[src];
      }
      window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
      // Odd width ⇒ the middle element is the median (`np.median`).
      out[p * cols + i] = window[pad_width];
    }
  }
  Ok(out)
}

/// Map an index into the reflect-padded last axis back to the unpadded `[0,
/// len)` range — numpy `mode="reflect"` (the edge sample is not repeated, so
/// `-1 -> 1`, `len -> len - 2`).
fn reflect_index(idx: isize, len: usize) -> usize {
  if len == 1 {
    return 0;
  }
  let period = 2 * (len as isize - 1);
  // Fold into `[0, period)`.
  let mut m = idx % period;
  if m < 0 {
    m += period;
  }
  // The second half of the period mirrors the first.
  if m >= len as isize {
    (period - m) as usize
  } else {
    m as usize
  }
}

/// Backtrace the DTW trace matrix into `(text_indices, time_indices)` —
/// `_backtrace` (`timing.py:52-73`).
///
/// `trace` is `(rows + 1, cols + 1)` of step codes (`0` diagonal, `1` up, `2`
/// left). Following the reference, the top row is forced to `2` (left) and the
/// left column to `1` (up), then the path is walked from the bottom-right to
/// the origin and reversed.
///
/// Each diagonal / up / left step decrements `i + j` by one from the start
/// value `n + m`, so the monotonic path visits at most `n + m` cells. The two
/// index vectors are therefore sized by the alignment dims; both are reserved
/// through the fallible path ([`reserve_or_error`] → typed
/// [`Error::AllocFailure`](crate::Error::AllocFailure)) against that checked
/// capacity, so a large valid alignment surfaces an out-of-memory condition as
/// a recoverable error instead of aborting on infallible `Vec` growth. No
/// magnitude cap is imposed.
///
/// # Errors
/// [`Error::OutOfRange`] if the path capacity `n + m` overflows `usize`;
/// [`Error::AllocFailure`](crate::Error::AllocFailure) if either reservation
/// fails.
fn backtrace(trace: &mut [i8], n: usize, m: usize) -> Result<(Vec<i64>, Vec<i64>)> {
  let cols = m + 1;
  let idx = |i: usize, j: usize| i * cols + j;
  // `trace[0, :] = 2; trace[:, 0] = 1` (`timing.py:55-56`).
  for j in 0..cols {
    trace[idx(0, j)] = 2;
  }
  for i in 0..=n {
    trace[idx(i, 0)] = 1;
  }

  // Every step decrements `i + j` by exactly one, so the path length is bounded
  // by the start value `n + m`; reserve both vectors to that checked capacity.
  let path_cap = n
    .checked_add(m)
    .ok_or_else(|| area_overflow("DTW path length (n + m)"))?;
  let mut text: Vec<i64> = Vec::new();
  reserve_or_error(
    &mut text,
    "Whisper word timestamps: DTW text path",
    path_cap,
  )?;
  let mut time: Vec<i64> = Vec::new();
  reserve_or_error(
    &mut time,
    "Whisper word timestamps: DTW time path",
    path_cap,
  )?;

  let mut i = n;
  let mut j = m;
  // `result.append((i - 1, j - 1))` appends a SIGNED pair: when the path slides
  // along the top (`i == 0`) or left (`j == 0`) border, the reference stores
  // `-1` (numpy int), so the indices are `i64`, mirroring numpy exactly. The
  // reservations above are exact, so neither push reallocates.
  while i > 0 || j > 0 {
    text.push(i as i64 - 1);
    time.push(j as i64 - 1);
    match trace[idx(i, j)] {
      0 => {
        i -= 1;
        j -= 1;
      }
      1 => i -= 1,
      _ => j -= 1,
    }
  }
  // `result[::-1, :].T` — reverse so the path runs origin → end.
  text.reverse();
  time.reverse();
  Ok((text, time))
}

/// Dynamic Time Warping over a `(N, M)` host cost matrix — `dtw`
/// (`timing.py:76-99`). Returns `(text_indices, time_indices)`: the two
/// equal-length (signed, numpy-int) index sequences of the lowest-cost
/// monotonic alignment.
///
/// The `(N+1, M+1)` `cost` / `trace` grids and the `<= N + M` backtrace path
/// vectors are all sized through a checked product / sum + fallible allocation,
/// so a large valid alignment surfaces an overflow / out-of-memory condition as
/// a typed error instead of aborting.
///
/// # Errors
/// [`Error::OutOfRange`] if `(N+1) * (M+1)` or the `N + M` path length overflows
/// `usize`; [`Error::AllocFailure`](crate::Error::AllocFailure) if a grid or
/// path allocation fails.
fn dtw(x: &HostMatrix) -> Result<(Vec<i64>, Vec<i64>)> {
  let n = x.rows;
  let m = x.cols;
  let rows1 = checked_inc(n, "DTW rows + 1")?;
  let cw = checked_inc(m, "DTW cols + 1")?;
  // `cost = full((N+1, M+1), inf); cost[0, 0] = 0`.
  let grid = checked_area(rows1, cw, "DTW grid ((N+1) * (M+1))")?;
  let mut cost = alloc_filled(
    "Whisper word timestamps: DTW cost grid",
    f32::INFINITY,
    grid,
  )?;
  let mut trace = alloc_filled("Whisper word timestamps: DTW trace grid", -1i8, grid)?;
  cost[0] = 0.0;

  // The reference iterates `j` (outer) then `i` (inner); preserve that order so
  // the `<` tie-breaking selects the same predecessor.
  for j in 1..=m {
    for i in 1..=n {
      let c0 = cost[(i - 1) * cw + (j - 1)];
      let c1 = cost[(i - 1) * cw + j];
      let c2 = cost[i * cw + (j - 1)];
      // `if c0 < c1 and c0 < c2: 0; elif c1 < c0 and c1 < c2: 1; else: 2`.
      let (c, t) = if c0 < c1 && c0 < c2 {
        (c0, 0i8)
      } else if c1 < c0 && c1 < c2 {
        (c1, 1i8)
      } else {
        (c2, 2i8)
      };
      cost[i * cw + j] = x.at(i - 1, j - 1) + c;
      trace[i * cw + j] = t;
    }
  }

  backtrace(&mut trace, n, m)
}

/// Compute the per-token frame alignment of `text_tokens` against `mel` and
/// merge it into per-word timings — `find_alignment` (`timing.py:111-182`).
///
/// `num_frames` is the segment's real (non-padded) mel frame count; the
/// alignment uses the first `num_frames / 2` audio-token columns of the
/// cross-attention (matching the encoder's stride-2 downsample). The median
/// filter uses the reference's fixed `median_filter_width` (`7`, odd and
/// positive), so the width can never be a degenerate `0`; `qk_scale` defaults to
/// the reference's `1.0`.
///
/// Returns an empty vector when `text_tokens` is empty, when the window is
/// sub-token (`num_frames < 2`, so `num_frames / 2` yields no audio-token
/// columns to align), or when the split yields one word (the reference's
/// eot-only guards).
///
/// # Errors
/// Propagates the model forward / on-device op / tokenizer errors;
/// [`Error::OutOfRange`] if a `text_token` is `>= eot` (a timestamp / special id
/// would index past the `[0, eot)` probability columns the per-token
/// probabilities gather over) or if a dimension overflows `i32`.
pub fn find_alignment(
  model: &WhisperModel,
  tokenizer: &HFTokenizerWrapper<'_>,
  text_tokens: &[u32],
  mel: &Array,
  num_frames: usize,
  qk_scale: f32,
) -> Result<Vec<WordTiming>> {
  if text_tokens.is_empty() {
    return Ok(Vec::new());
  }
  // A sub-token window (`num_frames < 2`, so the `num_frames / 2` audio-token
  // column count is `0`) carries no frames to align against: the reference's
  // `weights[:, :, : num_frames // 2]` would be a zero-width frame slice, and a
  // host DTW over it backtraces with `j == 0`, recording a `-1` time index that
  // would surface as a bogus negative (`-0.02s`) timestamp (or the backend
  // rejects the zero-width softmax). There are no word boundaries to place
  // inside a window shorter than one encoder frame-pair, so return no word
  // timings — the faithful degenerate-window result — BEFORE the forward pass.
  if num_frames / 2 == 0 {
    return Ok(Vec::new());
  }

  let sot_sequence = tokenizer.sot_sequence();
  let sot_len = sot_sequence.len();
  let eot = tokenizer.eot();
  // Reject any `text_token >= eot` BEFORE the forward: `text_token_probabilities`
  // slices the probability matrix to the `[0, eot)` text-vocab columns
  // (`timing.py:135`: `logits[0][...][:, : eot]`) and then gathers it by
  // `text_tokens` (`take_along_axis(.., text_tokens[:, None], axis=1)`), so a
  // timestamp / special id in `[eot, n_vocab)` — which passes the embedding
  // gather's `< n_vocab` bound — would index PAST that `eot`-wide matrix (out of
  // bounds). The text tokens aligned here are real text, `< eot`; the reference's
  // caller (`add_word_timestamps`) filters `[t for t in seg.tokens if t < eot]`
  // upstream, but this entry is public, so it enforces the precondition itself
  // with a typed error rather than relying on the caller.
  if let Some(&bad) = text_tokens.iter().find(|&&t| t >= eot) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Whisper word timestamps: alignment text token",
      "must be a text token (< eot) — timestamp / special ids index past the \
       [0, eot) probability columns",
      format_smolstr!("id={bad}, eot={eot}"),
    )));
  }
  // `tokens = [*sot_sequence, no_timestamps, *text_tokens, eot]` (`timing.py:124-131`).
  let mut tokens: Vec<u32> = Vec::new();
  reserve_or_error(
    &mut tokens,
    "Whisper word timestamps: alignment token buffer",
    sot_len + 2 + text_tokens.len(),
  )?;
  tokens.extend_from_slice(&sot_sequence);
  tokens.push(tokenizer.no_timestamps());
  tokens.extend_from_slice(text_tokens);
  tokens.push(eot);

  // `forward_with_cross_qk` takes the `&[u32]` slice directly and builds the
  // `(1, T)` `u32` decoder-input array internally (bounding `T` and the ids),
  // matching the incremental path's `&[u32]` contract.
  let (logits, cross_qk) = model.forward_with_cross_qk(mel, &tokens)?;

  // Per-token probabilities for the text positions (`timing.py:134-140`):
  // `sampled_logits = logits[0][sot_len : -2, : eot]`, softmax, gather the
  // realized `text_tokens`. The `-2` drops the no_timestamps row's prediction
  // and the eot row (the two positions after the last text token).
  let text_token_probs = text_token_probabilities(&logits, sot_len, eot, text_tokens)?;

  // Stack the alignment heads' cross-attention into `(heads, tokens, frames)`
  // (`timing.py:142-145`: `cross_qk[l][0, h]`), keep the first `num_frames / 2`
  // frames (guaranteed `> 0` by the sub-token guard above), softmax over frames,
  // std-normalize per token, median filter, then average over heads
  // (`timing.py:146-154`).
  let frames = num_frames / 2;
  let weights = stack_alignment_heads(model, &cross_qk, frames)?;
  // `weights = softmax(weights * qk_scale, axis=-1, precise=True)`.
  let scaled = if (qk_scale - 1.0).abs() > f32::EPSILON {
    ops::arithmetic::multiply(&weights, &Array::full::<f32>(&[0i32; 0], qk_scale)?)?
  } else {
    weights
  };
  let weights = ops::misc::softmax_axis(&scaled, -1, true)?;
  let weights = weights.astype(Dtype::F32)?;
  // `weights = (weights - mean) / std` — normalize each frame column across the
  // token axis (axis -2), with `std` floored so a zero-variance (degenerate)
  // column yields a finite `0` rather than a `NaN` that would poison the host
  // DTW (`normalize_alignment_weights`).
  let mut normalized = normalize_alignment_weights(&weights)?.astype(Dtype::F32)?;

  // Median-filter along the frame axis (host; the reference uses numpy), then
  // average over the head axis into the `(tokens, frames)` cost matrix. Every
  // host buffer below is sized through a checked product + fallible allocation
  // (typed `AllocFailure` / `OutOfRange`, no abort) without imposing any
  // magnitude cap on a large valid alignment.
  let (heads_n, rows, cols) = three_dims(&normalized)?;
  let planes = checked_area(heads_n, rows, "alignment planes (heads * tokens)")?;
  let host = host_copy_f32(&mut normalized)?;
  let filtered = median_filter(&host, planes, cols, MEDIAN_FILTER_WIDTH)?;
  let matrix_full = mean_over_heads(&filtered, heads_n, rows, cols)?;

  // `matrix = matrix[sot_len : -1]` — drop the sot rows and the trailing eot
  // row, leaving the text-token rows (`timing.py:155`).
  let matrix = slice_rows(&matrix_full, rows, cols, sot_len, rows.saturating_sub(1))?;
  // `text_indices, time_indices = dtw(-matrix)`: DTW minimizes cost, so the
  // negated attention (high attention → low cost) is aligned. The negated copy
  // is built through the same fallible path.
  let neg = HostMatrix {
    data: negated(&matrix.data)?,
    rows: matrix.rows,
    cols: matrix.cols,
  };
  let (text_indices, time_indices) = dtw(&neg)?;

  build_word_timings(
    tokenizer,
    text_tokens,
    eot,
    &text_token_probs,
    &text_indices,
    &time_indices,
  )
}

/// Compute per-word timings for a window's `segments` and attach them — a
/// faithful port of `add_word_timestamps` (`timing.py:219-329`).
///
/// Runs [`find_alignment`] over the window's combined text tokens, applies the
/// long-word duration hacks (sentence-boundary truncation, the per-segment
/// pause hack), merges punctuation ([`merge_punctuations`]), and distributes
/// the resulting [`WordTiming`]s back onto each segment as [`Word`]s — with the
/// frame times offset by `seek * HOP_LENGTH / SAMPLE_RATE`.
///
/// `last_speech_timestamp` is the PRIOR accepted-speech end, used only to seed
/// the per-segment pause hack; like the reference (`timing.py` returns `None`),
/// the running end is tracked internally and not handed back — the caller
/// advances the cross-window value after the hallucination-silence decision
/// (`whisper.py:1239-1241`).
///
/// `mel` is the padded `(N_FRAMES, n_mels)` window; `num_frames` the segment's
/// real frame count; `seek` the window's frame offset.
///
/// # Errors
/// Propagates [`find_alignment`].
#[allow(clippy::too_many_arguments)]
pub fn add_word_timestamps(
  model: &WhisperModel,
  tokenizer: &HFTokenizerWrapper<'_>,
  segments: &mut [Segment],
  mel: &Array,
  num_frames: usize,
  seek: usize,
  prepend_punctuations: &str,
  append_punctuations: &str,
  last_speech_timestamp: f64,
) -> Result<()> {
  if segments.is_empty() {
    return Ok(());
  }
  let eot = tokenizer.eot();

  // `text_tokens_per_segment = [[t for t in seg.tokens if t < eot] ...]`. Both
  // the per-segment lists and the flattened token stream are sized by the
  // window's decoded tokens (input-sized); reserve every buffer through the
  // fallible path (against an exact upper bound) so a large window surfaces an
  // out-of-memory condition as a typed error instead of aborting on `Vec`
  // growth. No magnitude cap is imposed.
  let mut text_tokens_per_segment: Vec<Vec<u32>> = Vec::new();
  reserve_or_error(
    &mut text_tokens_per_segment,
    "Whisper word timestamps: per-segment text tokens",
    segments.len(),
  )?;
  for s in segments.iter() {
    let mut seg_tokens: Vec<u32> = Vec::new();
    reserve_or_error(
      &mut seg_tokens,
      "Whisper word timestamps: segment text tokens",
      s.tokens.len(),
    )?;
    seg_tokens.extend(s.tokens.iter().copied().filter(|&t| t < eot));
    text_tokens_per_segment.push(seg_tokens);
  }
  let total_text_tokens: usize = text_tokens_per_segment.iter().map(Vec::len).sum();
  let mut text_tokens: Vec<u32> = Vec::new();
  reserve_or_error(
    &mut text_tokens,
    "Whisper word timestamps: window text tokens",
    total_text_tokens,
  )?;
  for v in &text_tokens_per_segment {
    text_tokens.extend_from_slice(v);
  }

  let mut alignment = find_alignment(model, tokenizer, &text_tokens, mel, num_frames, 1.0)?;

  // `word_durations = [t.end - t.start ...]`, drop zeros, median (capped 0.7).
  // Sized by the alignment word count (input-sized); reserve through the
  // fallible path against that upper bound before the filtered fill.
  let mut word_durations: Vec<f64> = Vec::new();
  reserve_or_error(
    &mut word_durations,
    "Whisper word timestamps: word durations",
    alignment.len(),
  )?;
  word_durations.extend(
    alignment
      .iter()
      .map(|t| t.end - t.start)
      .filter(|&d| d != 0.0),
  );
  let median_duration = if word_durations.is_empty() {
    0.0
  } else {
    median_f64(&mut word_durations).min(0.7)
  };
  let max_duration = median_duration * 2.0;

  // Truncate long words at sentence boundaries (`timing.py:247-257`).
  if !word_durations.is_empty() {
    const SENTENCE_END_MARKS: [&str; 6] = [".", "。", "!", "！", "?", "？"];
    for i in 1..alignment.len() {
      if alignment[i].end - alignment[i].start > max_duration {
        if SENTENCE_END_MARKS.contains(&alignment[i].word.as_str()) {
          alignment[i].end = alignment[i].start + max_duration;
        } else if SENTENCE_END_MARKS.contains(&alignment[i - 1].word.as_str()) {
          alignment[i].start = alignment[i].end - max_duration;
        }
      }
    }
  }

  merge_punctuations(&mut alignment, prepend_punctuations, append_punctuations);

  // `time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE`.
  let time_offset = seek as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
  let mut last_speech_timestamp = last_speech_timestamp;
  let mut word_index = 0usize;

  for (segment, text_seg_tokens) in segments.iter_mut().zip(&text_tokens_per_segment) {
    let mut saved_tokens = 0usize;
    // The per-segment words consume a prefix of the remaining alignment, so the
    // unconsumed tail bounds this buffer (input-sized); reserve it through the
    // fallible path against that upper bound.
    let mut words: Vec<Word> = Vec::new();
    reserve_or_error(
      &mut words,
      "Whisper word timestamps: segment words",
      alignment.len().saturating_sub(word_index),
    )?;

    // `while word_index < len(alignment) and saved_tokens < len(text_tokens)`.
    while word_index < alignment.len() && saved_tokens < text_seg_tokens.len() {
      let timing = &alignment[word_index];
      if !timing.word.is_empty() {
        words.push(Word {
          word: timing.word.clone(),
          start: round2(time_offset + timing.start),
          end: round2(time_offset + timing.end),
          probability: timing.probability,
        });
      }
      saved_tokens += timing.tokens.len();
      word_index += 1;
    }

    if !words.is_empty() {
      apply_segment_word_hacks(
        segment,
        &mut words,
        median_duration,
        max_duration,
        last_speech_timestamp,
      );
      last_speech_timestamp = segment.end;
    }
    segment.words = words;
  }

  Ok(())
}

/// The per-segment long-word truncation hacks (`timing.py:284-326`): truncate
/// the first/second word after a pause, and prefer the segment-level
/// start / end timestamp when the boundary word is too long. Mutates `words`
/// and the segment's `start` / `end`.
fn apply_segment_word_hacks(
  segment: &mut Segment,
  words: &mut [Word],
  median_duration: f64,
  max_duration: f64,
  last_speech_timestamp: f64,
) {
  // Truncate the first/second word after a pause (`timing.py:286-302`).
  let first_end = words[0].end;
  let first_start = words[0].start;
  if first_end - last_speech_timestamp > median_duration * 4.0
    && (first_end - first_start > max_duration
      || (words.len() > 1 && words[1].end - first_start > max_duration * 2.0))
  {
    if words.len() > 1 && words[1].end - words[1].start > max_duration {
      let boundary = (words[1].end / 2.0).max(words[1].end - max_duration);
      words[0].end = boundary;
      words[1].start = boundary;
    }
    words[0].start = 0.0f64.max(words[0].end - max_duration);
  }

  // Prefer the segment-level start timestamp if the first word is too long
  // (`timing.py:304-313`).
  let first_end = words[0].end;
  if segment.start < first_end && segment.start - 0.5 > words[0].start {
    words[0].start = 0.0f64.max((first_end - median_duration).min(segment.start));
  } else {
    segment.start = words[0].start;
  }

  // Prefer the segment-level end timestamp if the last word is too long
  // (`timing.py:315-324`).
  let last = words.len() - 1;
  let last_start = words[last].start;
  let last_end = words[last].end;
  if segment.end > last_start && segment.end + 0.5 < last_end {
    words[last].end = (last_start + median_duration).max(segment.end);
  } else {
    segment.end = words[last].end;
  }
}

/// `_get_end(segments)` (`whisper.py:255-259`): the last word's end if any
/// segment has words, else the last segment's `end`.
pub(crate) fn get_end(segments: &[Segment]) -> Option<f64> {
  for seg in segments.iter().rev() {
    if let Some(w) = seg.words.last() {
      return Some(w.end);
    }
  }
  segments.last().map(|s| s.end)
}

/// `word_anomaly_score(word)` (`whisper.py:1056-1066`): a heuristic anomaly
/// score from a word's probability + duration.
pub(crate) fn word_anomaly_score(word: &Word) -> f64 {
  let duration = word.end - word.start;
  let mut score = 0.0;
  if word.probability < 0.15 {
    score += 1.0;
  }
  if duration < 0.133 {
    score += (0.133 - duration) * 15.0;
  }
  if duration > 2.0 {
    score += duration - 2.0;
  }
  score
}

/// `is_segment_anomaly(segment)` (`whisper.py:1068-1076`): true when the first
/// (up to) 8 non-punctuation words of `segment` look hallucinated.
pub(crate) fn is_segment_anomaly(segment: Option<&Segment>) -> bool {
  let Some(segment) = segment else {
    return false;
  };
  if segment.words.is_empty() {
    return false;
  }
  // `words = [w for w in segment.words if w.word not in punctuation][:8]`.
  let words: Vec<&Word> = segment
    .words
    .iter()
    .filter(|w| !is_anomaly_punctuation(&w.word))
    .take(8)
    .collect();
  let score: f64 = words.iter().map(|w| word_anomaly_score(w)).sum();
  score >= 3.0 || score + 0.01 >= words.len() as f64
}

/// The first segment with words (`next_words_segment`, `whisper.py:1078-1079`).
pub(crate) fn next_words_segment(segments: &[Segment]) -> Option<&Segment> {
  segments.iter().find(|s| !s.words.is_empty())
}

/// The punctuation set the anomaly check excludes (`whisper.py:933`).
const ANOMALY_PUNCTUATION: &str =
  "\"'\u{201c}\u{00bf}([{-\"'.\u{3002},\u{ff0c}!\u{ff01}?\u{ff1f}:\u{ff1a}\u{201d})]\u{7d}\u{3001}";

/// `word in punctuation` for the anomaly filter — true when the whole word is
/// one of the punctuation characters.
fn is_anomaly_punctuation(word: &str) -> bool {
  let mut chars = word.chars();
  match (chars.next(), chars.next()) {
    (Some(c), None) => ANOMALY_PUNCTUATION.contains(c),
    _ => false,
  }
}

/// `np.median` of a non-empty host slice (sorts a copy; even-length averages
/// the two central elements — numpy semantics).
fn median_f64(xs: &mut [f64]) -> f64 {
  xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let n = xs.len();
  if n.is_multiple_of(2) {
    (xs[n / 2 - 1] + xs[n / 2]) / 2.0
  } else {
    xs[n / 2]
  }
}

/// `float(round(x, 2))` — round to 2 decimals (`timing.py:275-276`).
fn round2(x: f64) -> f64 {
  (x * 100.0).round() / 100.0
}

/// Gather the realized text-token probabilities — `timing.py:134-140`.
fn text_token_probabilities(
  logits: &Array,
  sot_len: usize,
  eot: u32,
  text_tokens: &[u32],
) -> Result<Vec<f64>> {
  let shape = logits.shape();
  // logits is `(1, T, n_vocab)`; the predicting rows are `[sot_len, T - 2)`.
  let t = shape[1];
  let vocab = shape[2];
  let start = i32::try_from(sot_len).map_err(|_| dim_overflow("sot_len"))?;
  let stop = i32::try_from(t.saturating_sub(2)).map_err(|_| dim_overflow("logits row stop"))?;
  let eot_i = i32::try_from(eot).map_err(|_| dim_overflow("eot"))?;
  let vocab_i = i32::try_from(vocab).map_err(|_| dim_overflow("n_vocab"))?;
  // `logits[0][sot_len:-2, :eot]`: drop the batch dim, slice the text rows and
  // the `[0, eot)` vocab columns.
  let sampled = ops::indexing::slice(
    logits,
    &[0, start, 0],
    &[1, stop, eot_i.min(vocab_i)],
    &[1, 1, 1],
  )?;
  // Collapse the leading singleton batch dim → `(rows, eot)`.
  let rows = stop - start;
  let sampled = sampled.reshape(&[rows, eot_i.min(vocab_i)])?;
  let probs = ops::misc::softmax_axis(&sampled, -1, true)?;

  // `take_along_axis(token_probs, text_tokens[:, None], axis=1).squeeze(1)`. The
  // gather index buffer is sized by the text-token count (input-sized), so it is
  // reserved through the fallible path before the per-token cast fill.
  let n = i32::try_from(text_tokens.len()).map_err(|_| dim_overflow("text_tokens len"))?;
  let mut idx: Vec<i32> = Vec::new();
  reserve_or_error(
    &mut idx,
    "Whisper word timestamps: probability gather index",
    text_tokens.len(),
  )?;
  for &tok in text_tokens {
    idx.push(i32::try_from(tok).map_err(|_| dim_overflow("text token id"))?);
  }
  let idx_arr = Array::from_slice::<i32>(&idx, &[n, 1])?;
  let gathered = ops::indexing::take_along_axis(&probs, &idx_arr, 1)?;
  // The `(rows, 1)` gather is copied to the host sized by the text-token count;
  // borrow it without allocating (`as_slice`) and widen to `f64` through the
  // fallible reservation, so the host copy cannot abort on infallible growth.
  let mut gathered = gathered.astype(Dtype::F32)?;
  let src = gathered.as_slice::<f32>()?;
  let mut probs64: Vec<f64> = Vec::new();
  reserve_or_error(
    &mut probs64,
    "Whisper word timestamps: text-token probabilities",
    src.len(),
  )?;
  probs64.extend(src.iter().map(|&v| f64::from(v)));
  Ok(probs64)
}

/// Stack the alignment heads' cross-attention into `(heads, tokens, frames)`,
/// keeping the first `frames` columns — `timing.py:142-146`.
fn stack_alignment_heads(
  model: &WhisperModel,
  cross_qk: &[Option<Array>],
  frames: usize,
) -> Result<Array> {
  let heads = model.alignment_heads().heads();
  if heads.is_empty() {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Whisper word timestamps: alignment heads",
      "must be non-empty",
      format_smolstr!("len=0"),
    )));
  }
  let frames_i = i32::try_from(frames).map_err(|_| dim_overflow("alignment frames"))?;
  let mut planes: Vec<Array> = Vec::new();
  reserve_or_error(
    &mut planes,
    "Whisper word timestamps: alignment head planes",
    heads.len(),
  )?;
  for &(layer, head) in heads {
    // `cross_qk[layer][0, head]` → `(T, n_audio_ctx)`; keep `[:, :frames]`.
    let qk = cross_qk
      .get(layer)
      .and_then(|q| q.as_ref())
      .ok_or_else(|| {
        Error::OutOfRange(OutOfRangePayload::new(
          "Whisper word timestamps: alignment head layer",
          "must index a decoder layer with cross-attention",
          format_smolstr!("layer={layer}"),
        ))
      })?;
    let shape = qk.shape();
    // `(1, n_text_head, T, n_audio_ctx)`.
    let t = i32::try_from(shape[2]).map_err(|_| dim_overflow("alignment T"))?;
    let head_i = i32::try_from(head).map_err(|_| dim_overflow("alignment head"))?;
    let sliced = ops::indexing::slice(
      qk,
      &[0, head_i, 0, 0],
      &[1, head_i + 1, t, frames_i],
      &[1, 1, 1, 1],
    )?;
    planes.push(sliced.reshape(&[t, frames_i])?);
  }
  // The reference list handed to `stack` is sized by the alignment-head count
  // (model-sized); reserve it through the fallible path before collecting.
  let mut refs: Vec<&Array> = Vec::new();
  reserve_or_error(
    &mut refs,
    "Whisper word timestamps: alignment head refs",
    planes.len(),
  )?;
  refs.extend(planes.iter());
  ops::shape::stack_axis(&refs, 0)
}

/// `mx.mean(weights, axis=0)` over the head axis of a `(heads, rows, cols)`
/// host buffer → `(rows, cols)`. The `(rows, cols)` accumulator is sized through
/// the checked product + fallible allocation.
///
/// # Errors
/// [`Error::OutOfRange`] if `rows * cols` overflows `usize`;
/// [`Error::AllocFailure`](crate::Error::AllocFailure) if the allocation fails.
fn mean_over_heads(data: &[f32], heads: usize, rows: usize, cols: usize) -> Result<HostMatrix> {
  let plane = checked_area(rows, cols, "mean-over-heads plane (rows * cols)")?;
  let mut out = alloc_filled("Whisper word timestamps: mean-over-heads", 0.0f32, plane)?;
  let inv = 1.0 / heads as f32;
  for h in 0..heads {
    let base = h * plane;
    for i in 0..plane {
      out[i] += data[base + i];
    }
  }
  for v in &mut out {
    *v *= inv;
  }
  Ok(HostMatrix {
    data: out,
    rows,
    cols,
  })
}

/// Slice `matrix[row_start:row_end]` of a `(rows, cols)` host matrix. The cut is
/// copied through the fallible allocation path.
///
/// # Errors
/// [`Error::AllocFailure`](crate::Error::AllocFailure) if the copy allocation
/// fails.
fn slice_rows(
  m: &HostMatrix,
  _rows: usize,
  cols: usize,
  row_start: usize,
  row_end: usize,
) -> Result<HostMatrix> {
  let start = row_start.min(row_end);
  let data = alloc_copied(
    "Whisper word timestamps: matrix row slice",
    &m.data[start * cols..row_end * cols],
  )?;
  Ok(HostMatrix {
    data,
    rows: row_end - start,
    cols,
  })
}

/// Convert the DTW token/time index path + per-token probabilities into per-word
/// timings — `timing.py:158-182`.
fn build_word_timings(
  tokenizer: &HFTokenizerWrapper<'_>,
  text_tokens: &[u32],
  eot: u32,
  text_token_probs: &[f64],
  text_indices: &[i64],
  time_indices: &[i64],
) -> Result<Vec<WordTiming>> {
  // `split_to_word_tokens(text_tokens + [eot])`.
  let mut with_eot: Vec<u32> = Vec::new();
  reserve_or_error(
    &mut with_eot,
    "Whisper word timestamps: word-split tokens",
    text_tokens.len() + 1,
  )?;
  with_eot.extend_from_slice(text_tokens);
  with_eot.push(eot);
  let (words, word_tokens) = tokenizer.split_to_word_tokens(&with_eot)?;
  // `if len(word_tokens) <= 1: return []` (`timing.py:159-165`).
  if word_tokens.len() <= 1 {
    return Ok(Vec::new());
  }

  // `word_boundaries = pad(cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))`.
  let mut word_boundaries: Vec<usize> = Vec::new();
  reserve_or_error(
    &mut word_boundaries,
    "Whisper word timestamps: word boundaries",
    word_tokens.len(),
  )?;
  word_boundaries.push(0);
  let mut acc = 0usize;
  for toks in &word_tokens[..word_tokens.len() - 1] {
    acc += toks.len();
    word_boundaries.push(acc);
  }

  // `jumps = pad(diff(text_indices), (1, 0), constant_values=1).astype(bool)`;
  // `jump_times = time_indices[jumps] / TOKENS_PER_SECOND` (`timing.py:168-169`).
  // A "jump" marks a position where the aligned text index advanced; the first
  // position is forced to a jump (`constant_values=1`). The jump_times are the
  // frame times at those advances — one per text token (the aligned tokens).
  // One entry per jump, bounded above by the DTW path length (input-sized);
  // reserve that upper bound through the fallible path before the fill.
  let mut jump_times: Vec<f64> = Vec::new();
  reserve_or_error(
    &mut jump_times,
    "Whisper word timestamps: jump times",
    text_indices.len(),
  )?;
  for k in 0..text_indices.len() {
    let is_jump = if k == 0 {
      true
    } else {
      text_indices[k] != text_indices[k - 1]
    };
    if is_jump {
      jump_times.push(time_indices[k] as f64 / TOKENS_PER_SECOND as f64);
    }
  }

  // `start_times = jump_times[word_boundaries[:-1]]`;
  // `end_times = jump_times[word_boundaries[1:]]`;
  // `word_probabilities[i] = mean(text_token_probs[bi:bj])`;
  // `zip(words, word_tokens, start_times, end_times, word_probabilities)`.
  //
  // `start_times` / `end_times` have length `word_boundaries.len() - 1` (the
  // number of real words — the trailing eot word is excluded from
  // `word_boundaries`); `words` / `word_tokens` are one longer (they include
  // eot). The reference's `zip` truncates to the shortest, so exactly
  // `word_boundaries.len() - 1` timings are produced (the eot word is dropped).
  let word_count = word_boundaries.len() - 1;
  let mut out: Vec<WordTiming> = Vec::new();
  reserve_or_error(
    &mut out,
    "Whisper word timestamps: per-word timings",
    word_count,
  )?;
  for w in 0..word_count {
    let bi = word_boundaries[w];
    let bj = word_boundaries[w + 1];
    // `jump_times` has one entry per aligned text row, so `bi` / `bj` (each
    // <= total text tokens) index it in range by construction; `get` guards a
    // degenerate alignment rather than panicking.
    let start = *jump_times.get(bi).unwrap_or(&0.0);
    let end = *jump_times
      .get(bj)
      .unwrap_or_else(|| jump_times.last().unwrap_or(&0.0));
    let prob = mean_slice(text_token_probs, bi, bj);
    out.push(WordTiming {
      word: words[w].clone(),
      tokens: word_tokens[w].clone(),
      start,
      end,
      probability: prob,
    });
  }
  Ok(out)
}

/// `np.mean(slice[i:j])` — `0.0` for an empty slice (matches the reference,
/// where a degenerate empty slice never occurs for a real word).
fn mean_slice(xs: &[f64], i: usize, j: usize) -> f64 {
  let end = j.min(xs.len());
  if end <= i {
    return 0.0;
  }
  let s: f64 = xs[i..end].iter().sum();
  s / (end - i) as f64
}

/// The three dims of a rank-3 array, as `usize`.
fn three_dims(a: &Array) -> Result<(usize, usize, usize)> {
  let s = a.shape();
  match s.as_slice() {
    [h, r, c] => Ok((*h, *r, *c)),
    _ => Err(Error::OutOfRange(OutOfRangePayload::new(
      "Whisper word timestamps: attention weights rank",
      "must be rank 3 (heads, tokens, frames)",
      format_smolstr!("ndim={}", s.len()),
    ))),
  }
}

/// The positive floor clamped onto the per-token standard deviation before the
/// `(weights - mean) / std` normalization (`timing.py:149-151`).
///
/// The reference divides by the raw std. A degenerate alignment — uniform
/// cross-attention over the token axis (an all-silence window or a tiny
/// synthetic model) — has a zero-variance frame column, so `std == 0` and the
/// raw divide yields `0/0 = NaN`, which would then poison the host DTW (whose
/// `<` comparisons all fall through on NaN to an arbitrary predecessor). The
/// floor turns that into a controlled, finite `0` for the degenerate column
/// while leaving every non-degenerate column unchanged: realistic post-softmax
/// std values are many orders of magnitude above `1e-10`, so the clamp is a
/// no-op on the normal path.
const ALIGNMENT_STD_FLOOR: f32 = 1e-10;

/// Std-normalize the alignment weights across the token axis —
/// `mean = mean(w, -2); std = var(w, -2, ddof=0).sqrt(); (w - mean) / std`
/// (`timing.py:149-151`) — with the std clamped to [`ALIGNMENT_STD_FLOOR`] so a
/// zero-variance (degenerate) column produces a finite `0` instead of `NaN`.
///
/// `weights` is the post-softmax `(heads, tokens, frames)` tensor. The result
/// is bit-identical to the reference on any column with `std >= floor` (every
/// realistic case); only the otherwise-`NaN` degenerate column is changed, to a
/// controlled finite value.
///
/// # Errors
/// Propagates the on-device reduction / arithmetic ops.
fn normalize_alignment_weights(weights: &Array) -> Result<Array> {
  let mean = ops::reduction::mean_axes(weights, &[-2], true)?;
  let std = ops::reduction::std_axes(weights, &[-2], true, 0)?;
  let std = ops::arithmetic::maximum(&std, &Array::full::<f32>(&[0i32; 0], ALIGNMENT_STD_FLOOR)?)?;
  let centered = ops::arithmetic::subtract(weights, &mean)?;
  ops::arithmetic::divide(&centered, &std)
}

/// Copy a contiguous f32 [`Array`] to a host `Vec<f32>` through the crate's
/// **fallible** reservation ([`reserve_or_error`] → typed
/// [`Error::AllocFailure`](crate::Error::AllocFailure)) so a large but valid
/// `(heads, tokens, frames)` alignment copy surfaces an out-of-memory condition
/// as a recoverable error rather than aborting the process the way the implicit
/// `Vec` growth behind a plain `to_vec` would.
///
/// No magnitude cap is imposed: a large-but-valid configuration is allowed to
/// run and only fails (typed) if the OS genuinely cannot satisfy the
/// allocation — bounding large valid input is the consumer's responsibility.
///
/// # Errors
/// [`Error::AllocFailure`](crate::Error::AllocFailure) if the host reservation
/// fails; propagates [`Array::as_slice`] (dtype / contiguity / eval) errors.
fn host_copy_f32(array: &mut Array) -> Result<Vec<f32>> {
  let src = array.as_slice::<f32>()?;
  alloc_copied("Whisper word timestamps: alignment host copy", src)
}

/// Copy a `Copy` host slice into a fresh `Vec` through the crate's **fallible**
/// reservation ([`reserve_or_error`] → typed
/// [`Error::AllocFailure`](crate::Error::AllocFailure)), so a large but valid
/// alignment buffer surfaces an out-of-memory condition as a recoverable error
/// instead of aborting the way a plain `slice.to_vec()` would. The reservation
/// is exact, so the `extend_from_slice` fill cannot reallocate.
///
/// No magnitude cap is imposed: a large-but-valid configuration runs and only
/// fails (typed) if the OS genuinely cannot satisfy the allocation — bounding
/// large valid input is the consumer's responsibility.
///
/// # Errors
/// [`Error::AllocFailure`](crate::Error::AllocFailure) if the reservation fails.
fn alloc_copied<T: Copy>(field: &'static str, src: &[T]) -> Result<Vec<T>> {
  let mut out: Vec<T> = Vec::new();
  reserve_or_error(&mut out, field, src.len())?;
  out.extend_from_slice(src);
  Ok(out)
}

/// `-src` element-wise into a fresh `Vec`, built through the fallible
/// reservation so a large valid cost matrix surfaces out-of-memory as a typed
/// error rather than aborting.
///
/// # Errors
/// [`Error::AllocFailure`](crate::Error::AllocFailure) if the reservation fails.
fn negated(src: &[f32]) -> Result<Vec<f32>> {
  let mut out: Vec<f32> = Vec::new();
  reserve_or_error(
    &mut out,
    "Whisper word timestamps: negated cost matrix",
    src.len(),
  )?;
  out.extend(src.iter().map(|&v| -v));
  Ok(out)
}

/// The checked product of two host-buffer dimensions, returning
/// [`Error::OutOfRange`] (naming `which`) if it overflows `usize`.
///
/// Used to size the DTW grids and the alignment scratch buffers (each a product
/// of two model/input dimensions, e.g. `(N+1) * (M+1)` or `heads * tokens`)
/// before a fallible allocation. It is a pure **overflow** guard — no magnitude
/// cap is imposed, so a large-but-valid alignment still allocates and only fails
/// (typed) if the product genuinely cannot be represented or the OS cannot
/// satisfy the allocation.
fn checked_area(a: usize, b: usize, which: &'static str) -> Result<usize> {
  a.checked_mul(b).ok_or_else(|| area_overflow(which))
}

/// `n + 1` checked against `usize` overflow — the `(N+1, M+1)` grid extents —
/// returning [`Error::OutOfRange`] (naming `which`) on overflow.
fn checked_inc(n: usize, which: &'static str) -> Result<usize> {
  n.checked_add(1).ok_or_else(|| area_overflow(which))
}

/// A timing dimension exceeding `i32::MAX`.
fn dim_overflow(which: &'static str) -> Error {
  Error::OutOfRange(OutOfRangePayload::new(
    "Whisper word timestamps: dimension",
    "must fit in i32",
    format_smolstr!("{which} exceeds i32::MAX"),
  ))
}

/// A host-buffer size whose checked product / increment overflowed `usize`.
fn area_overflow(which: &'static str) -> Error {
  Error::OutOfRange(OutOfRangePayload::new(
    "Whisper word timestamps: buffer size",
    "must not overflow usize",
    format_smolstr!("{which} overflows usize"),
  ))
}

/// The 16 prepend-merge punctuation characters
/// (`timing.py:226`: `"\"'“¿([{-`).
pub const PREPEND_PUNCTUATIONS: &str = "\"'\u{201c}\u{00bf}([{-";
/// The append-merge punctuation characters (`timing.py:227`).
pub const APPEND_PUNCTUATIONS: &str =
  "\"'.\u{3002},\u{ff0c}!\u{ff01}?\u{ff1f}:\u{ff1a}\u{201d})]\u{7d}\u{3001}";

/// Merge prepended / appended punctuation into adjacent words **in place** —
/// `merge_punctuations` (`timing.py:185-216`). A leading-space punctuation word
/// in `prepended` is folded into the following word; a punctuation word in
/// `appended` (not after a trailing space) is folded into the previous word.
/// The emptied slots are kept (their `word` becomes `""`), matching the
/// reference (the segment-assembly loop skips empty words but still counts
/// their tokens).
pub fn merge_punctuations(alignment: &mut [WordTiming], prepended: &str, appended: &str) {
  // Merge prepended punctuations (`timing.py:187-200`). The reference starts at
  // `i = len - 2`, `j = len - 1` and loops `while i >= 0`, so a list shorter
  // than 2 elements runs the loop zero times.
  if alignment.len() >= 2 {
    let mut j = alignment.len() - 1;
    let mut i = j - 1;
    loop {
      // `previous.word.startswith(" ") and previous.word.strip() in prepended`.
      let prev_word = alignment[i].word.clone();
      let merge = prev_word.starts_with(' ') && contains_str(prepended, prev_word.trim());
      if merge {
        let prev_tokens = std::mem::take(&mut alignment[i].tokens);
        alignment[j].word = format!("{}{}", prev_word, alignment[j].word);
        let mut merged = prev_tokens;
        merged.append(&mut alignment[j].tokens);
        alignment[j].tokens = merged;
        alignment[i].word = String::new();
      } else {
        j = i;
      }
      if i == 0 {
        break;
      }
      i -= 1;
    }
  }

  // Merge appended punctuations (`timing.py:202-216`).
  let mut i = 0usize;
  let mut j = 1usize;
  while j < alignment.len() {
    // `not previous.word.endswith(" ") and following.word in appended`.
    let follow_word = alignment[j].word.clone();
    let merge = !alignment[i].word.ends_with(' ') && contains_str(appended, &follow_word);
    if merge {
      let follow_tokens = std::mem::take(&mut alignment[j].tokens);
      alignment[i].word = format!("{}{}", alignment[i].word, follow_word);
      alignment[i].tokens.extend(follow_tokens);
      alignment[j].word = String::new();
    } else {
      i = j;
    }
    j += 1;
  }
}

/// Python `s in haystack` for the punctuation membership tests: `true` when
/// `s` is a (possibly empty) substring of `haystack`. The punctuation sets are
/// character lists, so this matches a single punctuation character — and the
/// empty string (Python `"" in haystack` is `True`).
fn contains_str(haystack: &str, s: &str) -> bool {
  s.is_empty() || haystack.contains(s)
}

#[cfg(test)]
mod tests;
