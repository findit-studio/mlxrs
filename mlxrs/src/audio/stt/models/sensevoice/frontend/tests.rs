//! Front-end oracles for SenseVoice-Small: LFR stacking, CMVN, the `am.mvn`
//! parse, and `sanitize`.
//!
//! Every expected value is computed independently of the code under test — by
//! a hand-written reference LFR over plain `Vec`s, by closed-form CMVN
//! arithmetic, against a literal `am.mvn` fixture, or against the verbatim
//! sanitize inputs — never by invoking the implementation a second time.

use std::collections::HashMap;

use super::*;
use crate::{
  array::Array,
  audio::features::{KaldiWindow, PreemphBoundary, compute_fbank_kaldi},
  error::Error,
};

// ───────────────────────── independent LFR reference ─────────────────────────

/// A from-scratch reference LFR over plain rows, mirroring `_apply_lfr`
/// (`sensevoice.py:47-72`) but sharing no code with [`apply_lfr`]. Returns the
/// `(T_lfr, lfr_m * D)` stacked frames.
fn lfr_reference(feats: &[Vec<f32>], lfr_m: usize, lfr_n: usize) -> Vec<Vec<f32>> {
  let t = feats.len();
  let d = feats[0].len();
  let t_lfr = t.div_ceil(lfr_n);

  // Left-pad with `(lfr_m - 1) / 2` copies of the first frame.
  let left_pad = (lfr_m - 1) / 2;
  let mut padded: Vec<Vec<f32>> = Vec::new();
  for _ in 0..left_pad {
    padded.push(feats[0].clone());
  }
  padded.extend(feats.iter().cloned());
  let t_padded = padded.len();
  let last = padded[t_padded - 1].clone();

  let mut out = Vec::with_capacity(t_lfr);
  for i in 0..t_lfr {
    let start = i * lfr_n;
    let end = start + lfr_m;
    let mut frame: Vec<f32> = Vec::with_capacity(lfr_m * d);
    if end <= t_padded {
      for row in &padded[start..end] {
        frame.extend_from_slice(row);
      }
    } else {
      for row in &padded[start..t_padded] {
        frame.extend_from_slice(row);
      }
      let pad_count = lfr_m - (t_padded - start);
      for _ in 0..pad_count {
        frame.extend_from_slice(&last);
      }
    }
    out.push(frame);
  }
  out
}

/// Build a `(T, D)` test fbank where row `t`, col `c` = `t * 100 + c` (so each
/// scalar is uniquely identifiable when stacked).
fn ramp_feats(t: usize, d: usize) -> (Array, Vec<Vec<f32>>) {
  let mut flat = Vec::with_capacity(t * d);
  let mut rows = Vec::with_capacity(t);
  for ti in 0..t {
    let mut row = Vec::with_capacity(d);
    for c in 0..d {
      let v = (ti * 100 + c) as f32;
      flat.push(v);
      row.push(v);
    }
    rows.push(row);
  }
  let arr = Array::from_slice::<f32>(&flat, &[t as i32, d as i32]).unwrap();
  (arr, rows)
}

#[test]
fn lfr_headline_shape_7x80_stride6() {
  // The real SenseVoice front-end: lfr_m=7, lfr_n=6, D=80 -> 560-wide frames,
  // T -> ceil(T / 6). T = 20 -> 4 frames.
  let (arr, _rows) = ramp_feats(20, 80);
  let out = apply_lfr(&arr, 7, 6).unwrap();
  assert_eq!(
    out.shape(),
    vec![4, 560],
    "ceil(20/6)=4 frames, 7*80=560 wide"
  );
}

#[test]
fn lfr_matches_independent_reference_small() {
  // A small case with a non-trivial right-pad: T=10, D=3, lfr_m=7, lfr_n=6.
  // T_lfr = ceil(10/6) = 2; left_pad = 3.
  let (arr, rows) = ramp_feats(10, 3);
  let mut got = apply_lfr(&arr, 7, 6).unwrap();
  let want = lfr_reference(&rows, 7, 6);

  assert_eq!(got.shape(), vec![want.len(), want[0].len()]);
  let got_flat = got.to_vec::<f32>().unwrap();
  let want_flat: Vec<f32> = want.iter().flatten().copied().collect();
  assert_eq!(got_flat, want_flat);
}

#[test]
fn lfr_matches_reference_exact_multiple() {
  // T an exact multiple of lfr_n with no right-pad needed in the last window:
  // T=12, lfr_m=2, lfr_n=2 -> T_lfr=6, each frame stacks 2 rows.
  let (arr, rows) = ramp_feats(12, 4);
  let mut got = apply_lfr(&arr, 2, 2).unwrap();
  let want = lfr_reference(&rows, 2, 2);
  assert_eq!(got.shape(), vec![6, 8]);
  assert_eq!(
    got.to_vec::<f32>().unwrap(),
    want.iter().flatten().copied().collect::<Vec<_>>()
  );
}

#[test]
fn lfr_first_frame_is_left_padded_first_row() {
  // With left_pad = 3 (lfr_m=7), the very first LFR frame starts with 3 copies
  // of row 0, then rows 0,1,2,3 — i.e. the first `4 * D` values after the
  // 3-copy prefix come from rows 0..4. Check the first frame's first D values
  // equal row 0 (the left-pad copy).
  let (arr, rows) = ramp_feats(10, 3);
  let mut got = apply_lfr(&arr, 7, 6).unwrap();
  let flat = got.to_vec::<f32>().unwrap();
  // Frame 0, first D=3 values = row 0 (a left-pad copy).
  assert_eq!(&flat[0..3], rows[0].as_slice());
  // Frame 0, the 4th block (cols 9..12) = row 0 again (start=0 -> rows
  // [pad,pad,pad,0,1,2,3]; block index 3 = row 0).
  assert_eq!(&flat[9..12], rows[0].as_slice());
}

/// An empty feature matrix (zero frames — empty or sub-window audio) leaves
/// `t_padded == 0`; `apply_lfr` must reject it with a typed `EmptyInput` error
/// rather than underflow the tail-clamp index (a panic under overflow checks).
/// The loop reference likewise cannot stack zero windows, so rejecting is the
/// behavior-preserving choice.
#[test]
fn lfr_rejects_empty_input() {
  let empty = Array::from_slice::<f32>(&[], &[0, 8]).unwrap();
  assert!(matches!(apply_lfr(&empty, 7, 6), Err(Error::EmptyInput(_))));
}

#[test]
fn lfr_rejects_non_rank2() {
  let bad = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
  assert!(matches!(apply_lfr(&bad, 7, 6), Err(Error::RankMismatch(_))));
}

#[test]
fn lfr_rejects_non_positive_factors() {
  let (arr, _) = ramp_feats(4, 2);
  assert!(matches!(apply_lfr(&arr, 0, 6), Err(Error::OutOfRange(_))));
  assert!(matches!(apply_lfr(&arr, 7, 0), Err(Error::OutOfRange(_))));
}

#[test]
fn lfr_reshape_width_overflow_is_typed_not_panic() {
  // The stacked-frame width `lfr_m * D` is computed through checked arithmetic
  // (`sensevoice.py:62` reshape extent). A `lfr_m` so large that `lfr_m * D`
  // overflows `i32` must surface a typed ArithmeticOverflow, never a debug
  // overflow panic or a wrapped-small reshape. `D = 2`, `lfr_m = i32::MAX` makes
  // the product overflow; the guard fires before any tile / window work.
  let (arr, _) = ramp_feats(4, 2);
  assert!(matches!(
    apply_lfr(&arr, i32::MAX, 6),
    Err(Error::ArithmeticOverflow(_))
  ));
}

// ───────────────────────── CMVN ─────────────────────────

#[test]
fn cmvn_is_shift_then_scale() {
  // `(feats + means) * istd` (`sensevoice.py:80`), broadcast over T.
  // feats = [[1,2,3],[4,5,6]]; means = [10, 20, 30]; istd = [0.5, 0.1, 2.0].
  let feats = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
  let means = Array::from_slice::<f32>(&[10.0, 20.0, 30.0], &[3]).unwrap();
  let istd = Array::from_slice::<f32>(&[0.5, 0.1, 2.0], &[3]).unwrap();
  let mut out = apply_cmvn(&feats, &means, &istd).unwrap();
  // Row 0: (1+10)*0.5=5.5, (2+20)*0.1=2.2, (3+30)*2=66.
  // Row 1: (4+10)*0.5=7.0, (5+20)*0.1=2.5, (6+30)*2=72.
  let got = out.to_vec::<f32>().unwrap();
  let want = [5.5_f32, 2.2, 66.0, 7.0, 2.5, 72.0];
  for (g, w) in got.iter().zip(want.iter()) {
    assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
  }
}

// ───────────────────────── am.mvn parse ─────────────────────────

/// A minimal `am.mvn` fixture in the Kaldi MVN text format the reference parses
/// (`sensevoice.py:83-103`). The `<AddShift>` block carries the additive shift,
/// the `<Rescale>` block the inverse stddev; each value list follows a
/// `<LearnRateCoef> 0 [ ... ]` header.
const AM_MVN_FIXTURE: &str = r"<Nnet>
<Splice> 560 560
[ 0 ]
<AddShift> 560 560
<LearnRateCoef> 0 [ -1.5 -2.5 -3.5 ]
<Rescale> 560 560
<LearnRateCoef> 0 [ 0.5 0.25 0.125 ]
</Nnet>
";

#[test]
fn parses_am_mvn_means_and_istd() {
  let (means, istd) = parse_am_mvn(AM_MVN_FIXTURE).unwrap();
  assert_eq!(means, vec![-1.5, -2.5, -3.5]);
  assert_eq!(istd, vec![0.5, 0.25, 0.125]);
}

#[test]
fn am_mvn_missing_addshift_is_malformed() {
  let text = "<Rescale> 3 3\n<LearnRateCoef> 0 [ 1.0 1.0 1.0 ]\n";
  assert!(matches!(parse_am_mvn(text), Err(Error::MalformedData(_))));
}

#[test]
fn am_mvn_missing_rescale_is_malformed() {
  let text = "<AddShift> 3 3\n<LearnRateCoef> 0 [ -1.0 -2.0 -3.0 ]\n";
  assert!(matches!(parse_am_mvn(text), Err(Error::MalformedData(_))));
}

#[test]
fn am_mvn_non_float_token_is_malformed() {
  // A non-numeric token inside the bracket fails the float parse -> malformed.
  let text = "<AddShift> 3 3\n<LearnRateCoef> 0 [ -1.0 oops -3.0 ]\n<Rescale> 3 3\n<LearnRateCoef> 0 [ 1.0 1.0 1.0 ]\n";
  assert!(matches!(parse_am_mvn(text), Err(Error::MalformedData(_))));
}

#[test]
fn am_mvn_handles_multiline_bracket() {
  // The DOTALL flag lets the bracketed list span lines (real `am.mvn` files
  // often wrap the 560-wide vector). The `.*?` between tag and `[` is also
  // newline-spanning.
  let text = "<AddShift> 2 2\n<LearnRateCoef> 0 [ -1.0\n-2.0 ]\n<Rescale> 2 2\n<LearnRateCoef> 0 [ 3.0\n4.0 ]\n";
  let (means, istd) = parse_am_mvn(text).unwrap();
  assert_eq!(means, vec![-1.0, -2.0]);
  assert_eq!(istd, vec![3.0, 4.0]);
}

// ───────────────────────── fbank pre-emphasis boundary ─────────────────────────

#[test]
fn compute_fbank_uses_preserve_preemph_boundary() {
  // SenseVoice's `_compute_fbank` (`sensevoice.py:17-44`) forwards to
  // `mlx_audio.dsp.compute_fbank_kaldi`, which KEEPS the first sample of each
  // frame unchanged under pre-emphasis (`dsp.py:913`:
  // `first_col = strided_input[:, 0:1]`). This test pins that the front-end
  // opts into `PreemphBoundary::Preserve`: its features must equal a direct
  // `compute_fbank_kaldi(..., Preserve)` call (same fixed params + the `2^15`
  // pre-scale) and must DIFFER from the `Scale` boundary — i.e. the scaling
  // deviation would change the model input for every frame.
  let fc = FrontendConfig::default();
  // A DC-rich ramp so the first-sample boundary feeds an observably different
  // spectrum (matching the shared-helper boundary test's signal shape).
  let samples: Vec<f32> = (0..4_000).map(|i| (i as f32) / 4_000.0).collect();
  let x = Array::from_slice::<f32>(&samples, &[4_000_i32]).unwrap();

  let mut got = compute_fbank(&x, &fc).unwrap();
  let got_flat = got.to_vec::<f32>().unwrap();

  // Reproduce the front-end's exact fbank call (`win_len = 16000*25/1000 =
  // 400`, `win_inc = 160`, hamming, preemph 0.97, dither 0.0, snip_edges,
  // low 20.0, high 0.0) with the `2^15` pre-scale, once per boundary mode.
  let scale = Array::full::<f32>(&[0i32; 0], (1u32 << 15) as f32).unwrap();
  let scaled = x.multiply(&scale).unwrap();
  let make = |boundary| {
    compute_fbank_kaldi(
      &scaled,
      16_000,
      400,
      160,
      80,
      KaldiWindow::Hamming,
      0.97,
      0.0,
      true,
      20.0,
      0.0,
      None,
      boundary,
    )
    .unwrap()
  };
  let mut preserve = make(PreemphBoundary::Preserve);
  let preserve_flat = preserve.to_vec::<f32>().unwrap();
  let mut scale_feats = make(PreemphBoundary::Scale);
  let scale_flat = scale_feats.to_vec::<f32>().unwrap();

  // The front-end must use the Preserve boundary -> byte-identical to it.
  assert_eq!(got_flat.len(), preserve_flat.len());
  for (i, (g, p)) in got_flat.iter().zip(preserve_flat.iter()).enumerate() {
    assert!(
      (g - p).abs() < 1e-5,
      "compute_fbank must match compute_fbank_kaldi(Preserve) at [{i}]: {g} vs {p}"
    );
  }
  // ...and observably differ from the Scale boundary (the deviation we fixed).
  let max_diff = got_flat
    .iter()
    .zip(scale_flat.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0_f32, f32::max);
  assert!(
    max_diff > 1e-4,
    "compute_fbank (Preserve) must differ from the Scale boundary (max diff {max_diff})"
  );
}

// ───────────────────────── sanitize ─────────────────────────

#[test]
fn sanitize_strips_ctc_prefix() {
  // `ctc.ctc_lo.weight` -> `ctc_lo.weight` (`sensevoice.py:559`).
  let mut w = HashMap::new();
  w.insert(
    "ctc.ctc_lo.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0], &[1, 2]).unwrap(),
  );
  w.insert(
    "ctc.ctc_lo.bias".to_string(),
    Array::from_slice::<f32>(&[0.5], &[1]).unwrap(),
  );
  let out = sanitize(w).unwrap();
  assert!(out.contains_key("ctc_lo.weight"));
  assert!(out.contains_key("ctc_lo.bias"));
  assert!(!out.contains_key("ctc.ctc_lo.weight"));
}

#[test]
fn sanitize_transposes_rank3_fsmn_weight() {
  // A torch depthwise Conv1d weight (C_out, C_in/groups, K) = (2, 1, 3) becomes
  // MLX layout (C_out, K, C_in/groups) = (2, 3, 1) under transpose(0, 2, 1)
  // (`sensevoice.py:561-562`).
  let mut w = HashMap::new();
  // Values laid out so the axis swap is verifiable: row-major (2,1,3) =
  // [[[a,b,c]], [[d,e,f]]]. After transpose(0,2,1) -> (2,3,1) =
  // [[[a],[b],[c]], [[d],[e],[f]]] = same flat order [a,b,c,d,e,f] since the
  // middle/last dims (1 and 3) swap to (3 and 1) with a size-1 axis.
  w.insert(
    "encoder.encoders0.0.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]).unwrap(),
  );
  let mut out = sanitize(w).unwrap();
  let mut t = out
    .remove("encoder.encoders0.0.self_attn.fsmn_block.weight")
    .unwrap();
  assert_eq!(t.shape(), vec![2, 3, 1], "axis-swapped to MLX conv layout");
  // Flat data is preserved by the (1<->3) swap of size-1 / size-3 axes.
  assert_eq!(
    t.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn sanitize_leaves_non_rank3_fsmn_weight_untouched() {
  // A non-rank-3 tensor matching the name is left as-is (the reference guards
  // on `v.ndim() == 3`).
  let mut w = HashMap::new();
  w.insert(
    "x.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
  );
  let out = sanitize(w).unwrap();
  assert_eq!(out["x.fsmn_block.weight"].shape(), vec![2, 2]);
}

#[test]
fn sanitize_passes_through_unrelated_keys() {
  let mut w = HashMap::new();
  w.insert(
    "encoder.after_norm.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 1.0], &[2]).unwrap(),
  );
  let out = sanitize(w).unwrap();
  assert!(out.contains_key("encoder.after_norm.weight"));
}

// ──────────── apply_lfr vectorization A/B (gather optimization) ────────────

/// The per-frame slice / reshape / (concat-tail) / stack loop form of
/// `_apply_lfr` (`sensevoice.py:57-72`), kept verbatim as the A/B reference for
/// the single-`take` vectorized `apply_lfr`. Shares no code with the production
/// helper; the A/B asserts the two are bit-identical for every shape.
fn apply_lfr_loop_reference(feats: &Array, lfr_m: i32, lfr_n: i32) -> Array {
  let shape = feats.shape();
  let t = shape[0];
  let d = shape[1] as i32;
  let lfr_n_usize = lfr_n as usize;
  let lfr_m_usize = lfr_m as usize;
  let lfr_width = lfr_m * d;
  let t_lfr = t.div_ceil(lfr_n_usize);

  let left_pad = (lfr_m - 1) / 2;
  let first = ops::indexing::slice(feats, &[0, 0], &[1, d], &[1, 1]).unwrap();
  let padded = if left_pad > 0 {
    let head = ops::shape::tile(&first, &[left_pad, 1]).unwrap();
    ops::shape::concatenate(&[&head, feats], 0).unwrap()
  } else {
    feats.try_clone().unwrap()
  };
  let t_padded = padded.shape()[0];
  let t_padded_i32 = t_padded as i32;
  let last =
    ops::indexing::slice(&padded, &[t_padded_i32 - 1, 0], &[t_padded_i32, d], &[1, 1]).unwrap();

  let mut frames: Vec<Array> = Vec::with_capacity(t_lfr);
  for i in 0..t_lfr {
    let start = i * lfr_n_usize;
    let end = start + lfr_m_usize;
    let start_i32 = start as i32;
    let stacked = if end <= t_padded {
      let end_i32 = start_i32 + lfr_m;
      let window = ops::indexing::slice(&padded, &[start_i32, 0], &[end_i32, d], &[1, 1]).unwrap();
      ops::shape::reshape(&window, &[lfr_width]).unwrap()
    } else {
      let available =
        ops::indexing::slice(&padded, &[start_i32, 0], &[t_padded_i32, d], &[1, 1]).unwrap();
      let avail_rows = t_padded - start;
      let pad_count = (lfr_m_usize - avail_rows) as i32;
      let tail = ops::shape::tile(&last, &[pad_count, 1]).unwrap();
      let window = ops::shape::concatenate(&[&available, &tail], 0).unwrap();
      ops::shape::reshape(&window, &[lfr_width]).unwrap()
    };
    frames.push(stacked);
  }
  let refs: Vec<&Array> = frames.iter().collect();
  ops::shape::stack(&refs).unwrap()
}

/// CORRECTNESS A/B for the LFR gather: the vectorized `apply_lfr` (single
/// `take`) must be BIT-IDENTICAL to the per-frame loop reference for every shape
/// — a single off-by-one in the gather index would silently corrupt frames.
/// Covers the real config, exact-multiple (no tail pad), a heavy tail-clamp, a
/// no-left-pad case, and `lfr_m == 1` (degenerate stack width).
#[test]
fn lfr_vectorized_is_bit_identical_to_loop() {
  // (T, D, lfr_m, lfr_n): each row a distinct edge.
  let cases: &[(usize, usize, i32, i32)] = &[
    (20, 80, 7, 6), // real SenseVoice front-end (560-wide frames).
    (12, 4, 2, 2),  // exact multiple, no right-pad needed.
    (10, 3, 7, 6),  // small with left-pad + a partial tail window.
    (5, 6, 7, 6),   // T < lfr_m: the FIRST window already overruns -> tail clamp.
    (8, 4, 4, 1),   // stride 1, no left-pad rounding, many overlapping windows.
    (9, 5, 1, 3),   // lfr_m == 1: no left-pad, each frame is a single row.
    (1, 4, 7, 6),   // single input frame: everything clamps to the one row.
    (13, 7, 6, 5),  // odd-ish T with a non-trivial right-pad.
  ];
  for &(t, d, lfr_m, lfr_n) in cases {
    let (arr, _rows) = ramp_feats(t, d);
    let mut got = apply_lfr(&arr, lfr_m, lfr_n).unwrap();
    let mut want = apply_lfr_loop_reference(&arr, lfr_m, lfr_n);
    assert_eq!(
      got.shape(),
      want.shape(),
      "shape mismatch for (T={t},D={d},m={lfr_m},n={lfr_n})"
    );
    assert_eq!(
      got.to_vec::<f32>().unwrap(),
      want.to_vec::<f32>().unwrap(),
      "vectorized vs loop differ for (T={t},D={d},m={lfr_m},n={lfr_n})"
    );
  }
}

/// The vectorized `apply_lfr` must also match the from-scratch Vec oracle on the
/// hard tail-clamp case (T < lfr_m), independently confirming the gather index
/// clamp reproduces the reference's last-row right-padding.
#[test]
fn lfr_vectorized_tail_clamp_matches_vec_reference() {
  let (arr, rows) = ramp_feats(5, 6);
  let mut got = apply_lfr(&arr, 7, 6).unwrap();
  let want = lfr_reference(&rows, 7, 6);
  assert_eq!(got.shape(), vec![want.len(), want[0].len()]);
  assert_eq!(
    got.to_vec::<f32>().unwrap(),
    want.iter().flatten().copied().collect::<Vec<_>>()
  );
}

/// PERF A/B for the LFR gather: the single-`take` `apply_lfr` vs the per-frame
/// loop reference at the real front-end shape over many calls. Reports best-of-N
/// min for both; the vectorized form must be no slower.
///
/// `#[ignore]`d: timing is machine/thermal-dependent. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore = "timing micro-bench — run with --ignored --nocapture"]
fn bench_lfr_vectorized_vs_loop() {
  use std::time::Instant;
  // A long-ish utterance: ~10 s of fbank (1000 frames) at D=80, lfr 7/6.
  let (arr, _rows) = ramp_feats(1000, 80);
  crate::transforms::eval(&[&arr]).unwrap();

  for _ in 0..5 {
    let a = apply_lfr(&arr, 7, 6).unwrap();
    let b = apply_lfr_loop_reference(&arr, 7, 6);
    crate::transforms::eval(&[&a, &b]).unwrap();
  }
  let bench = |label: &str, f: &dyn Fn() -> Array| {
    let mut times = Vec::with_capacity(30);
    for _ in 0..30 {
      let t0 = Instant::now();
      let out = f();
      crate::transforms::eval(&[&out]).unwrap();
      times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
      "  {label:<14} min={:.4}ms median={:.4}ms",
      times[0],
      times[times.len() / 2]
    );
    times[0]
  };
  println!("\napply_lfr (T=1000, D=80, m=7, n=6 -> 167 frames x 560):");
  let loop_min = bench("loop", &|| apply_lfr_loop_reference(&arr, 7, 6));
  let vec_min = bench("take (vectorized)", &|| apply_lfr(&arr, 7, 6).unwrap());
  println!("  speedup (loop/take) = {:.2}x", loop_min / vec_min);
}
