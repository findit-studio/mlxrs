//! Front-end oracles for SenseVoice-Small: LFR stacking, CMVN, the `am.mvn`
//! parse, and `sanitize`.
//!
//! Every expected value is computed independently of the code under test — by
//! a hand-written reference LFR over plain `Vec`s, by closed-form CMVN
//! arithmetic, against a literal `am.mvn` fixture, or against the verbatim
//! sanitize inputs — never by invoking the implementation a second time.

use std::collections::HashMap;

use super::*;
use crate::{array::Array, error::Error};

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
