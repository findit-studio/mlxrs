use super::*;

// ───────────────────────── DTW oracle ─────────────────────────────────────

/// An independent DTW reference: a fresh, from-scratch reimplementation of the
/// `timing.py::dtw` dynamic program (inf-padded `(N+1, M+1)` cost grid, the
/// exact `c0<c1 and c0<c2 / elif c1<c0 and c1<c2 / else` predecessor selection
/// where ALL ties fall to "left", and the `_backtrace` walk with the top row
/// forced to "left" and the left column to "up"). This does not call the code
/// under test; it is the same documented algorithm computed separately, so an
/// equal result confirms the port.
fn dtw_oracle(cost: &[Vec<f32>]) -> (Vec<i64>, Vec<i64>) {
  let n = cost.len();
  let m = cost[0].len();
  let cw = m + 1;
  let mut acc = vec![f32::INFINITY; (n + 1) * cw];
  let mut trace = vec![-1i8; (n + 1) * cw];
  acc[0] = 0.0;
  for j in 1..=m {
    for i in 1..=n {
      let c0 = acc[(i - 1) * cw + (j - 1)];
      let c1 = acc[(i - 1) * cw + j];
      let c2 = acc[i * cw + (j - 1)];
      let (c, t) = if c0 < c1 && c0 < c2 {
        (c0, 0i8)
      } else if c1 < c0 && c1 < c2 {
        (c1, 1i8)
      } else {
        (c2, 2i8)
      };
      acc[i * cw + j] = cost[i - 1][j - 1] + c;
      trace[i * cw + j] = t;
    }
  }
  // `_backtrace`: force the borders, then walk from (n, m) to the origin.
  for slot in trace.iter_mut().take(cw) {
    *slot = 2;
  }
  for i in 0..=n {
    trace[i * cw] = 1;
  }
  let mut i = n;
  let mut j = m;
  let mut ti = Vec::new();
  let mut tj = Vec::new();
  while i > 0 || j > 0 {
    ti.push(i as i64 - 1);
    tj.push(j as i64 - 1);
    match trace[i * cw + j] {
      0 => {
        i -= 1;
        j -= 1;
      }
      1 => i -= 1,
      _ => j -= 1,
    }
  }
  ti.reverse();
  tj.reverse();
  (ti, tj)
}

fn host_matrix(rows: usize, cols: usize, data: Vec<f32>) -> HostMatrix {
  HostMatrix { data, rows, cols }
}

#[test]
fn dtw_diagonal_alignment_on_identity_cost() {
  // A 4x4 cost that is 0 on the diagonal and 1 elsewhere: the lowest-cost
  // monotonic path is the diagonal, aligning token i to frame i.
  let n = 4;
  let mut data = vec![1.0f32; n * n];
  for i in 0..n {
    data[i * n + i] = 0.0;
  }
  let m = host_matrix(n, n, data);
  let (text, time) = dtw(&m).expect("dtw");
  assert_eq!(text, vec![0i64, 1, 2, 3]);
  assert_eq!(time, vec![0i64, 1, 2, 3]);
}

#[test]
fn dtw_matches_independent_oracle_on_random_costs() {
  // Several non-square cost matrices: the ported DTW must produce the exact
  // same path as the independent DP oracle (same step set + tie-break).
  let cases: &[(usize, usize, &[f32])] = &[
    (
      3,
      5,
      &[
        0.2, 0.9, 0.8, 0.7, 0.6, //
        0.9, 0.1, 0.3, 0.8, 0.9, //
        0.8, 0.7, 0.2, 0.1, 0.4,
      ],
    ),
    (
      4,
      4,
      &[
        0.0, 0.5, 0.9, 0.9, //
        0.6, 0.0, 0.5, 0.9, //
        0.9, 0.6, 0.0, 0.5, //
        0.9, 0.9, 0.6, 0.0,
      ],
    ),
    (
      2,
      6,
      &[
        0.1, 0.2, 0.9, 0.9, 0.9, 0.9, //
        0.9, 0.9, 0.9, 0.3, 0.2, 0.1,
      ],
    ),
  ];
  for &(rows, cols, flat) in cases {
    let m = host_matrix(rows, cols, flat.to_vec());
    let (gt, gtime) = dtw(&m).expect("dtw");
    let grid: Vec<Vec<f32>> = (0..rows)
      .map(|r| flat[r * cols..(r + 1) * cols].to_vec())
      .collect();
    let (ot, otime) = dtw_oracle(&grid);
    assert_eq!(gt, ot, "text indices for {rows}x{cols}");
    assert_eq!(gtime, otime, "time indices for {rows}x{cols}");
    // The path is monotonic + starts at (0,0), ends at (rows-1, cols-1).
    assert_eq!((gt[0], gtime[0]), (0i64, 0i64));
    assert_eq!(
      (*gt.last().unwrap(), *gtime.last().unwrap()),
      ((rows - 1) as i64, (cols - 1) as i64)
    );
    for k in 1..gt.len() {
      assert!(gt[k] >= gt[k - 1] && gtime[k] >= gtime[k - 1]);
    }
  }
}

#[test]
fn dtw_path_length_within_reserved_capacity() {
  // The fallible backtrace reserves the path vectors to `n + m`; every diagonal
  // / up / left step decrements `i + j` by one from the start value `n + m`, so
  // the produced path must never exceed that bound (the reservation can never be
  // overrun). Check a spread of rectangular shapes.
  for &(rows, cols) in &[(1usize, 1usize), (3, 5), (5, 3), (4, 4), (2, 7)] {
    let mut data = vec![1.0f32; rows * cols];
    for d in 0..rows.min(cols) {
      data[d * cols + d] = 0.0;
    }
    let (text, time) = dtw(&host_matrix(rows, cols, data)).expect("dtw");
    assert_eq!(text.len(), time.len());
    assert!(
      text.len() <= rows + cols,
      "path length {} exceeds reserved capacity {}",
      text.len(),
      rows + cols
    );
    assert!(
      !text.is_empty(),
      "a non-empty cost grid yields a non-empty path"
    );
  }
}

// ───────────────────────── median filter ──────────────────────────────────

/// An independent median filter: reflect-pad (numpy semantics) then take each
/// window's median by sorting.
fn median_filter_oracle(row: &[f32], width: usize) -> Vec<f32> {
  let pad = width / 2;
  let n = row.len();
  if n <= pad {
    return row.to_vec();
  }
  // Build the reflect-padded row explicitly: [pad reversed | row | pad reversed].
  let mut padded = Vec::with_capacity(n + 2 * pad);
  for k in (1..=pad).rev() {
    padded.push(row[k]);
  }
  padded.extend_from_slice(row);
  for k in 1..=pad {
    padded.push(row[n - 1 - k]);
  }
  let mut out = Vec::with_capacity(n);
  for i in 0..n {
    let mut w = padded[i..i + width].to_vec();
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out.push(w[pad]);
  }
  out
}

#[test]
fn median_filter_matches_oracle() {
  let row = vec![5.0f32, 1.0, 4.0, 1.0, 9.0, 2.0, 6.0, 5.0, 3.0];
  for width in [3usize, 5, 7] {
    let got = median_filter(&row, 1, row.len(), width).expect("median filter");
    let want = median_filter_oracle(&row, width);
    assert_eq!(got, want, "width {width}");
  }
}

#[test]
fn median_filter_short_row_unchanged() {
  // `cols <= pad_width` → returned unchanged (the reference early return).
  // width 7 → pad 3; a row of length 3 is <= 3.
  let row = vec![2.0f32, 7.0, 1.0];
  let got = median_filter(&row, 1, row.len(), 7).expect("median filter");
  assert_eq!(got, row);
}

#[test]
fn median_filter_width_constant_is_odd_and_positive_and_runs() {
  // `find_alignment` now uses the fixed `MEDIAN_FILTER_WIDTH` (the reference's
  // `median_filter_width = 7`) instead of a caller-supplied width, so the
  // degenerate `filter_width == 0` panic path is structurally unreachable. Pin
  // the constant's value + odd-positive shape and confirm the filter runs on a
  // small non-empty frame axis with it.
  assert_eq!(MEDIAN_FILTER_WIDTH, 7);
  // Odd + positive is a compile-time invariant of the constant (the median's
  // middle-element selection depends on it), so check it in a `const` block.
  const { assert!(MEDIAN_FILTER_WIDTH > 0 && MEDIAN_FILTER_WIDTH % 2 == 1) };
  let row = vec![5.0f32, 1.0, 4.0, 1.0, 9.0, 2.0, 6.0, 5.0, 3.0];
  let got =
    median_filter(&row, 1, row.len(), MEDIAN_FILTER_WIDTH).expect("median filter with constant");
  assert_eq!(got, median_filter_oracle(&row, MEDIAN_FILTER_WIDTH));
}

#[test]
fn median_filter_per_plane_independent() {
  // Two planes filtered independently (the head axis is flattened into planes).
  let cols = 5;
  let p0 = vec![1.0f32, 9.0, 2.0, 8.0, 3.0];
  let p1 = vec![4.0f32, 4.0, 4.0, 1.0, 7.0];
  let mut data = p0.clone();
  data.extend_from_slice(&p1);
  let got = median_filter(&data, 2, cols, 3).expect("median filter");
  let mut want = median_filter_oracle(&p0, 3);
  want.extend(median_filter_oracle(&p1, 3));
  assert_eq!(got, want);
}

#[test]
fn reflect_index_matches_numpy_reflect() {
  // numpy reflect for len 5: indices -3..7 map to [3,2,1,0,1,2,3,4,3,2].
  let len = 5;
  let expected = [3usize, 2, 1, 0, 1, 2, 3, 4, 3, 2];
  for (k, &want) in expected.iter().enumerate() {
    let idx = k as isize - 3;
    assert_eq!(reflect_index(idx, len), want, "idx {idx}");
  }
  // Degenerate length 1 folds everything to 0.
  assert_eq!(reflect_index(-2, 1), 0);
  assert_eq!(reflect_index(3, 1), 0);
}

// ───────────────────────── frame → seconds ────────────────────────────────

#[test]
fn jump_times_frame_to_seconds_conversion() {
  // TOKENS_PER_SECOND = 50 for Whisper (16000 / (160*2)). A frame index of 50
  // maps to 1.0 s; 25 → 0.5 s.
  assert_eq!(TOKENS_PER_SECOND, 50);
  let to_seconds = |frame: usize| frame as f64 / TOKENS_PER_SECOND as f64;
  assert!((to_seconds(50) - 1.0).abs() < 1e-12);
  assert!((to_seconds(25) - 0.5).abs() < 1e-12);
  assert!((to_seconds(0) - 0.0).abs() < 1e-12);
}

#[test]
fn round2_rounds_to_two_decimals() {
  assert!((round2(1.234) - 1.23).abs() < 1e-12);
  assert!((round2(1.235) - 1.24).abs() < 1e-9);
  assert!((round2(0.005) - 0.01).abs() < 1e-9);
}

#[test]
fn median_f64_even_and_odd() {
  let mut odd = vec![3.0f64, 1.0, 2.0];
  assert_eq!(median_f64(&mut odd), 2.0);
  let mut even = vec![4.0f64, 1.0, 3.0, 2.0];
  assert_eq!(median_f64(&mut even), 2.5);
}

// ───────────────────────── punctuation merge ──────────────────────────────

fn wt(word: &str, tokens: &[u32], start: f64, end: f64) -> WordTiming {
  WordTiming {
    word: word.to_string(),
    tokens: tokens.to_vec(),
    start,
    end,
    probability: 1.0,
  }
}

#[test]
fn merge_punctuations_appends_trailing_period() {
  // "Hello" + "." → "Hello." with merged tokens; the period slot is emptied.
  let mut alignment = vec![wt("Hello", &[10], 0.0, 0.5), wt(".", &[11], 0.5, 0.6)];
  merge_punctuations(&mut alignment, PREPEND_PUNCTUATIONS, APPEND_PUNCTUATIONS);
  assert_eq!(alignment[0].word, "Hello.");
  assert_eq!(alignment[0].tokens, vec![10, 11]);
  assert_eq!(alignment[1].word, "");
  assert!(alignment[1].tokens.is_empty());
}

#[test]
fn merge_punctuations_prepends_leading_quote() {
  // " \"" (a leading-space opening quote) prepends to the following word.
  let mut alignment = vec![wt(" \"", &[20], 0.0, 0.1), wt(" world", &[21], 0.1, 0.5)];
  merge_punctuations(&mut alignment, PREPEND_PUNCTUATIONS, APPEND_PUNCTUATIONS);
  assert_eq!(alignment[0].word, "");
  assert!(alignment[0].tokens.is_empty());
  assert_eq!(alignment[1].word, " \" world");
  assert_eq!(alignment[1].tokens, vec![20, 21]);
}

#[test]
fn merge_punctuations_leaves_plain_words() {
  // No punctuation → unchanged.
  let mut alignment = vec![wt(" the", &[1], 0.0, 0.2), wt(" cat", &[2], 0.2, 0.4)];
  let before = alignment.clone();
  merge_punctuations(&mut alignment, PREPEND_PUNCTUATIONS, APPEND_PUNCTUATIONS);
  assert_eq!(alignment, before);
}

// ───────────────────────── anomaly heuristic ──────────────────────────────

fn word(w: &str, start: f64, end: f64, p: f64) -> Word {
  Word {
    word: w.to_string(),
    start,
    end,
    probability: p,
  }
}

#[test]
fn word_anomaly_score_components() {
  // Low probability (< 0.15) → +1.0.
  assert!((word_anomaly_score(&word("a", 0.0, 0.5, 0.1)) - 1.0).abs() < 1e-9);
  // Too-short duration (< 0.133): score += (0.133 - dur) * 15; here dur 0.0.
  let s = word_anomaly_score(&word("a", 1.0, 1.0, 0.9));
  assert!((s - 0.133 * 15.0).abs() < 1e-6);
  // Too-long duration (> 2.0): score += dur - 2.0.
  assert!((word_anomaly_score(&word("a", 0.0, 3.0, 0.9)) - 1.0).abs() < 1e-9);
  // A normal word scores 0.
  assert_eq!(word_anomaly_score(&word("a", 0.0, 0.5, 0.9)), 0.0);
}

#[test]
fn is_segment_anomaly_flags_hallucinated_words() {
  // A segment of low-probability words scores >= 3 → anomalous.
  let words = vec![
    word("a", 0.0, 0.5, 0.05),
    word("b", 0.5, 1.0, 0.05),
    word("c", 1.0, 1.5, 0.05),
  ];
  let seg = make_segment(words);
  assert!(is_segment_anomaly(Some(&seg)));

  // A segment of confident, normally-timed words is not anomalous.
  let good = vec![
    word("a", 0.0, 0.5, 0.9),
    word("b", 0.5, 1.0, 0.9),
    word("c", 1.0, 1.5, 0.9),
  ];
  assert!(!is_segment_anomaly(Some(&make_segment(good))));

  // None / no-words → not anomalous.
  assert!(!is_segment_anomaly(None));
  assert!(!is_segment_anomaly(Some(&make_segment(vec![]))));
}

#[test]
fn get_end_prefers_last_word_else_segment_end() {
  // Last word end wins.
  let seg = make_segment(vec![word("a", 0.0, 0.5, 0.9), word("b", 0.5, 1.2, 0.9)]);
  assert_eq!(get_end(&[seg]), Some(1.2));
  // No words → segment end.
  let mut seg2 = make_segment(vec![]);
  seg2.end = 3.4;
  assert_eq!(get_end(&[seg2]), Some(3.4));
  // Empty list → None.
  assert_eq!(get_end(&[]), None);
}

/// A minimal [`Segment`] carrying the given words (other fields are zeroed).
fn make_segment(words: Vec<Word>) -> Segment {
  Segment {
    start: 0.0,
    end: 0.0,
    text: String::new(),
    tokens: Vec::new(),
    temperature: 0.0,
    avg_logprob: 0.0,
    no_speech_prob: 0.0,
    compression_ratio: 0.0,
    words,
  }
}

// ──────────── alignment normalization: zero-variance guard ─────────────────

/// A `(heads, tokens, frames)` weights [`Array`] from row-major `f32` data.
fn weights_array(heads: usize, tokens: usize, frames: usize, data: &[f32]) -> Array {
  Array::from_slice::<f32>(data, &[heads as i32, tokens as i32, frames as i32])
    .expect("weights array")
}

/// Uniform (constant) cross-attention over the token axis is the degenerate
/// case the floor protects: every frame column has zero variance, so the raw
/// `(w - mean) / std` would be `0/0 = NaN`. The floor must yield a finite `0`
/// for the whole tensor — and feeding that through the host DTW must neither
/// panic nor produce a NaN-poisoned path.
#[test]
fn normalize_alignment_weights_uniform_is_finite_and_dtw_safe() {
  let (heads, tokens, frames) = (1usize, 4usize, 3usize);
  // All-equal weights ⇒ per-column std == 0 (degenerate).
  let data = vec![0.25f32; heads * tokens * frames];
  let mut normalized =
    normalize_alignment_weights(&weights_array(heads, tokens, frames, &data)).expect("normalize");
  let host = host_copy_f32(&mut normalized).expect("host copy");

  assert_eq!(host.len(), heads * tokens * frames);
  assert!(
    host.iter().all(|v| v.is_finite()),
    "degenerate column must normalize to a finite value, got {host:?}"
  );
  // A degenerate column floors to (w - mean) / floor == 0 / floor == 0.
  assert!(
    host.iter().all(|&v| v == 0.0),
    "expected all-zero, got {host:?}"
  );

  // The (tokens, frames) matrix built from this must drive DTW with no NaN and
  // no panic. Build the cost grid directly from the finite host buffer.
  let matrix = host_matrix(tokens, frames, host);
  let neg = HostMatrix {
    data: matrix.data.iter().map(|&v| -v).collect(),
    rows: matrix.rows,
    cols: matrix.cols,
  };
  let (text, time) = dtw(&neg).expect("dtw");
  assert_eq!(text.len(), time.len());
  assert!(!text.is_empty(), "DTW must return a non-empty path");
}

/// The normal (non-degenerate) path is numerically unchanged: with every
/// frame column having a healthy variance the floor is inactive, so the result
/// equals the reference's raw `(w - mean) / std` to within f32 tolerance.
#[test]
fn normalize_alignment_weights_nondegenerate_matches_raw_formula() {
  let (heads, tokens, frames) = (1usize, 3usize, 2usize);
  // Distinct values per (token, frame); columns have clearly non-zero variance.
  #[rustfmt::skip]
  let data: Vec<f32> = vec![
    0.10, 0.90,
    0.40, 0.20,
    0.85, 0.55,
  ];
  let mut normalized =
    normalize_alignment_weights(&weights_array(heads, tokens, frames, &data)).expect("normalize");
  let got = host_copy_f32(&mut normalized).expect("host copy");

  // Independent host reference: per-column population mean + std, then
  // (x - mean) / std (no floor — every column's std is well above 1e-10).
  let mut want = vec![0.0f32; data.len()];
  for c in 0..frames {
    let col: Vec<f32> = (0..tokens).map(|r| data[r * frames + c]).collect();
    let mean = col.iter().sum::<f32>() / tokens as f32;
    let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / tokens as f32;
    let std = var.sqrt();
    assert!(std > 1e-3, "test column must be non-degenerate");
    for r in 0..tokens {
      want[r * frames + c] = (data[r * frames + c] - mean) / std;
    }
  }

  assert_eq!(got.len(), want.len());
  for (g, w) in got.iter().zip(&want) {
    assert!(g.is_finite());
    assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
  }
}

// ──────────────── alignment host copy: fallible reservation ────────────────

/// [`host_copy_f32`] copies a small contiguous f32 array through the fallible
/// reservation path, reproducing the source values exactly (the happy path of
/// the no-abort allocation used for the full alignment tensor).
#[test]
fn host_copy_f32_reproduces_small_array() {
  let data = [-1.5f32, 0.0, 2.25, 7.0, -0.5, 3.5];
  let mut arr = Array::from_slice::<f32>(&data, &[2i32, 3i32]).expect("array");
  let host = host_copy_f32(&mut arr).expect("fallible host copy");
  assert_eq!(host.as_slice(), &data);
}
