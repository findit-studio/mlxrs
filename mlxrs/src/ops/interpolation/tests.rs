//! Tests for [`bicubic_interpolate`].
//!
//! The expected values are derived from the PyTorch
//! `F.interpolate(mode="bicubic", align_corners=False)` algorithm with
//! cubic coefficient `a = -0.75`: the half-pixel source map
//! `src = (j + 0.5) * (in/out) - 0.5`, the four clamped taps
//! `floor(src) - 1 ..= floor(src) + 2`, the Keys cubic kernel, and edge
//! replication (off-edge taps fold onto the boundary). They are computed
//! in closed form (by hand / an independent f64 reference), not by
//! delegating to the code under test.

use super::*;

const EPS: f32 = 1e-5;

fn to_vec(a: &Array) -> Vec<f32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

/// Independent f64 reference for one resampling axis (the `(out, in)`
/// weight matrix). Mirrors the algorithm but is written separately so
/// the test is not comparing the code to itself.
fn ref_axis_weights(in_d: usize, out_d: usize) -> Vec<Vec<f64>> {
  const A: f64 = -0.75;
  fn k(t: f64) -> f64 {
    let x = t.abs();
    if x <= 1.0 {
      ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
      (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
      0.0
    }
  }
  let scale = in_d as f64 / out_d as f64;
  let in_last = (in_d - 1) as isize;
  let mut rows = vec![vec![0.0f64; in_d]; out_d];
  for (j, row) in rows.iter_mut().enumerate() {
    let src = (j as f64 + 0.5) * scale - 0.5;
    let base = src.floor();
    let frac = src - base;
    let base_i = base as isize;
    for off in -1isize..=2 {
      let w = k(frac - off as f64);
      let tap = (base_i + off).clamp(0, in_last) as usize;
      row[tap] += w;
    }
  }
  rows
}

/// Independent f64 reference resize of a single-channel grid.
fn ref_resize(grid: &[Vec<f64>], out_h: usize, out_w: usize) -> Vec<Vec<f64>> {
  let h_in = grid.len();
  let w_in = grid[0].len();
  let wh = ref_axis_weights(h_in, out_h);
  let ww = ref_axis_weights(w_in, out_w);
  let mut rows = vec![vec![0.0f64; w_in]; out_h];
  for (i, rrow) in rows.iter_mut().enumerate() {
    for (x, cell) in rrow.iter_mut().enumerate() {
      let mut s = 0.0;
      for kk in 0..h_in {
        s += wh[i][kk] * grid[kk][x];
      }
      *cell = s;
    }
  }
  let mut out = vec![vec![0.0f64; out_w]; out_h];
  for (i, orow) in out.iter_mut().enumerate() {
    for (j, cell) in orow.iter_mut().enumerate() {
      let mut s = 0.0;
      for x in 0..w_in {
        s += rows[i][x] * ww[j][x];
      }
      *cell = s;
    }
  }
  out
}

#[test]
fn bicubic_identity_same_size_is_bit_exact_passthrough() {
  // out == in on both axes: the fast path returns the input unchanged.
  let data: Vec<f32> = (0..(3 * 3 * 2)).map(|i| i as f32 * 0.5 - 1.0).collect();
  let grid = Array::from_slice::<f32>(&data, &(3, 3, 2)).unwrap();
  let out = bicubic_interpolate(&grid, 3, 3).unwrap();
  assert_eq!(out.shape(), vec![3, 3, 2]);
  assert_eq!(
    to_vec(&out),
    data,
    "identity resize must be exact passthrough"
  );
}

#[test]
fn bicubic_upsample_2x2_to_4x4_single_channel_matches_hand_computed() {
  // grid = [[0,1],[2,3]] (single channel). The expected 4x4 is the
  // closed-form bicubic (a=-0.75, align_corners=False) — note the
  // characteristic edge overshoot to NEGATIVE values at the corners,
  // which a=-0.75 bicubic (unlike bilinear) produces.
  let grid = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &(2, 2, 1)).unwrap();
  let out = bicubic_interpolate(&grid, 4, 4).unwrap();
  assert_eq!(out.shape(), vec![4, 4, 1]);
  let got = to_vec(&out);
  #[rustfmt::skip]
  let want: [f32; 16] = [
    -0.316406,  0.015625,  0.5625,    0.894531,
     0.347656,  0.679688,  1.226563,  1.558594,
     1.441406,  1.773438,  2.320313,  2.652344,
     2.105469,  2.4375,     2.984375,  3.316406,
  ];
  for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
    assert!((g - w).abs() < EPS, "idx {i}: got {g}, want {w}");
  }
}

#[test]
fn bicubic_downsample_width_ramp_1x4_to_1x2_matches_hand_computed() {
  // A 1x4 horizontal ramp 0,1,2,3 downsampled to 1x2. Expected from the
  // a=-0.75 kernel at the two output column centers.
  let grid = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &(1, 4, 1)).unwrap();
  let out = bicubic_interpolate(&grid, 1, 2).unwrap();
  assert_eq!(out.shape(), vec![1, 2, 1]);
  let got = to_vec(&out);
  let want = [0.40625f32, 2.59375];
  for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
    assert!((g - w).abs() < EPS, "idx {i}: got {g}, want {w}");
  }
}

#[test]
fn bicubic_constant_grid_stays_constant_partition_of_unity() {
  // The bicubic weights are a partition of unity (each output row's four
  // taps sum to 1), so a constant grid must resample to the same
  // constant everywhere — independent of the resize ratio.
  let grid = Array::from_slice::<f32>(&[5.0f32; 9], &(3, 3, 1)).unwrap();
  let out = bicubic_interpolate(&grid, 6, 7).unwrap();
  assert_eq!(out.shape(), vec![6, 7, 1]);
  for (i, v) in to_vec(&out).iter().enumerate() {
    assert!(
      (v - 5.0).abs() < EPS,
      "idx {i}: constant grid drifted to {v}"
    );
  }
}

#[test]
fn bicubic_axis_weight_rows_are_partition_of_unity() {
  // Direct check of the weight build: every output row sums to ~1.0.
  for (in_d, out_d) in [(2usize, 4usize), (3, 6), (16, 27), (16, 12)] {
    let rows = ref_axis_weights(in_d, out_d);
    for (j, r) in rows.iter().enumerate() {
      let s: f64 = r.iter().sum();
      assert!(
        (s - 1.0).abs() < 1e-9,
        "axis_w({in_d},{out_d}) row {j} sum = {s}"
      );
    }
  }
}

#[test]
fn bicubic_multichannel_is_independent_per_channel() {
  // A 2-channel grid where channel 1 = channel 0 + 10. Bicubic is linear
  // and per-channel, so the resized channel 1 must equal resized
  // channel 0 + 10 everywhere.
  let mut data = Vec::new();
  for v in [0.0f32, 1.0, 2.0, 3.0] {
    data.push(v); // channel 0
    data.push(v + 10.0); // channel 1
  }
  let grid = Array::from_slice::<f32>(&data, &(2, 2, 2)).unwrap();
  let out = bicubic_interpolate(&grid, 4, 4).unwrap();
  assert_eq!(out.shape(), vec![4, 4, 2]);
  let got = to_vec(&out);
  for px in 0..16 {
    let c0 = got[px * 2];
    let c1 = got[px * 2 + 1];
    assert!((c1 - (c0 + 10.0)).abs() < 1e-4, "px {px}: c0={c0} c1={c1}");
  }
}

#[test]
fn bicubic_matches_independent_f64_reference_on_3x3_to_5x5() {
  // Cross-check the full op against the independent f64 reference on a
  // non-trivial grid + non-square upsample.
  #[rustfmt::skip]
  let g = vec![
    vec![0.0f64, 1.0, 2.0],
    vec![3.0, 4.0, 5.0],
    vec![6.0, 7.0, 8.0],
  ];
  let flat: Vec<f32> = g.iter().flatten().map(|&v| v as f32).collect();
  let grid = Array::from_slice::<f32>(&flat, &(3, 3, 1)).unwrap();
  let out = bicubic_interpolate(&grid, 5, 5).unwrap();
  let got = to_vec(&out);
  let want = ref_resize(&g, 5, 5);
  for i in 0..5 {
    for j in 0..5 {
      let w = want[i][j] as f32;
      let gv = got[i * 5 + j];
      assert!((gv - w).abs() < 1e-4, "({i},{j}): got {gv}, want {w}");
    }
  }
}

#[test]
fn bicubic_rejects_non_rank3_grid() {
  let g2 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let err = bicubic_interpolate(&g2, 4, 4).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn bicubic_rejects_zero_and_oversize_dims() {
  let grid = Array::from_slice::<f32>(&[1.0f32; 4], &(2, 2, 1)).unwrap();
  // zero output dim
  let err = bicubic_interpolate(&grid, 0, 4).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "out_h=0: got {err}");
  // oversize output dim (> MAX_INTERP_DIM)
  let err = bicubic_interpolate(&grid, 4, MAX_INTERP_DIM + 1).unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_)),
    "out_w huge: got {err}"
  );
}

#[test]
fn bicubic_rejects_integer_grid_dtype() {
  // Build a rank-3 int32 grid; the fractional cubic weights cannot
  // resample it, so the op must reject the dtype.
  let g = Array::from_slice::<i32>(&[1, 2, 3, 4], &(2, 2, 1)).unwrap();
  let err = bicubic_interpolate(&g, 4, 4).unwrap_err();
  assert!(matches!(err, Error::UnsupportedDtype(_)), "got {err}");
}
