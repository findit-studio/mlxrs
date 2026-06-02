//! Tests for [`bilinear_interpolate`] (bilinear + antialias).
//!
//! The expected values are derived from PyTorch's
//! `F.interpolate(mode="bilinear", align_corners=False, antialias=True)`
//! algorithm (the `aa_filter` triangle path of `aten`'s
//! `UpSampleKernel.cpp`): `scale = in/out`, `center = scale*(i+0.5)`,
//! `support = scale>=1 ? scale : 1`, `invscale = scale>=1 ? 1/scale : 1`,
//! the `[0,in]`-clamped tap window `xmin..xmax`, the triangle filter
//! `f(x)=max(0,1-|x|)` at `(t - center + 0.5)*invscale`, and a per-row
//! renormalization to sum 1. They are computed in closed form (by hand /
//! an independent f64 reference), not by delegating to the code under
//! test. (Byte-exact agreement with the HF oracle weights is confirmed
//! numerically only by the gated SigLIP2 e2e parity test; these cases pin
//! the formula.)

use super::*;

const EPS: f32 = 1e-5;

fn to_vec(a: &Array) -> Vec<f32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

/// Independent f64 reference for one resampling axis (the `(out, in)`
/// weight matrix). Mirrors PyTorch's antialias linear algorithm but is
/// written separately so the test is not comparing the code to itself.
fn ref_axis_weights(in_d: usize, out_d: usize) -> Vec<Vec<f64>> {
  fn f(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
  }
  let scale = in_d as f64 / out_d as f64;
  let support = if scale >= 1.0 { scale } else { 1.0 };
  let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
  let in_i = in_d as i64;
  let mut rows = vec![vec![0.0f64; in_d]; out_d];
  for (i, row) in rows.iter_mut().enumerate() {
    let center = scale * (i as f64 + 0.5);
    let xmin = ((center - support + 0.5).floor() as i64).max(0);
    let xmax = ((center + support + 0.5).floor() as i64).min(in_i);
    let mut tot = 0.0;
    for t in xmin..xmax {
      let w = f((t as f64 - center + 0.5) * invscale);
      row[t as usize] = w;
      tot += w;
    }
    if tot != 0.0 {
      for t in xmin..xmax {
        row[t as usize] /= tot;
      }
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
fn bilinear_identity_same_size_is_bit_exact_passthrough() {
  // out == in on both axes: the fast path returns the input unchanged.
  let data: Vec<f32> = (0..(3 * 3 * 2)).map(|i| i as f32 * 0.5 - 1.0).collect();
  let grid = Array::from_slice::<f32>(&data, &(3, 3, 2)).unwrap();
  let out = bilinear_interpolate(&grid, 3, 3).unwrap();
  assert_eq!(out.shape(), vec![3, 3, 2]);
  assert_eq!(
    to_vec(&out),
    data,
    "identity resize must be exact passthrough"
  );
}

#[test]
fn bilinear_upsample_2x2_to_4x4_single_channel_matches_hand_computed() {
  // grid = [[0,1],[2,3]] (single channel). Upsampling ⇒ antialias is a
  // no-op, so this is plain bilinear align_corners=False. The 1-D axis
  // weights for 2->4 are rows [1,0],[0.75,0.25],[0.25,0.75],[0,1]; the
  // separable 4x4 below is hand-computed from those.
  let grid = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &(2, 2, 1)).unwrap();
  let out = bilinear_interpolate(&grid, 4, 4).unwrap();
  assert_eq!(out.shape(), vec![4, 4, 1]);
  let got = to_vec(&out);
  #[rustfmt::skip]
  let want: [f32; 16] = [
    0.00, 0.25, 0.75, 1.00,
    0.50, 0.75, 1.25, 1.50,
    1.50, 1.75, 2.25, 2.50,
    2.00, 2.25, 2.75, 3.00,
  ];
  for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
    assert!((g - w).abs() < EPS, "idx {i}: got {g}, want {w}");
  }
}

#[test]
fn bilinear_downsample_width_ramp_1x4_to_1x2_matches_hand_computed() {
  // A 1x4 horizontal ramp 0,1,2,3 downsampled to 1x2. scale=2 ⇒ antialias
  // active: the 4->2 axis weights are row0=[3/7,3/7,1/7,0],
  // row1=[0,1/7,3/7,3/7]. So out[0] = (0*3+1*3+2*1)/7 = 5/7 = 0.714286,
  // out[1] = (1*1+2*3+3*3)/7 = 16/7 = 2.285714. (A plain 2-tap bilinear
  // would give 0.5 and 2.5 — the antialias spreading is what differs.)
  let grid = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &(1, 4, 1)).unwrap();
  let out = bilinear_interpolate(&grid, 1, 2).unwrap();
  assert_eq!(out.shape(), vec![1, 2, 1]);
  let got = to_vec(&out);
  let want = [5.0f32 / 7.0, 16.0f32 / 7.0];
  for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
    assert!((g - w).abs() < EPS, "idx {i}: got {g}, want {w}");
  }
}

#[test]
fn bilinear_downsample_4x1_to_2x1_height_matches_hand_computed() {
  // The same antialias downsample on the HEIGHT axis (a 4x1 vertical ramp
  // -> 2x1), to pin that the row-resample matrix is built the same way.
  let grid = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &(4, 1, 1)).unwrap();
  let out = bilinear_interpolate(&grid, 2, 1).unwrap();
  assert_eq!(out.shape(), vec![2, 1, 1]);
  let got = to_vec(&out);
  let want = [5.0f32 / 7.0, 16.0f32 / 7.0];
  for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
    assert!((g - w).abs() < EPS, "idx {i}: got {g}, want {w}");
  }
}

#[test]
fn bilinear_axis_weight_rows_sum_to_one_up_and_down() {
  // Every output row's weights sum to ~1 (PyTorch renormalizes), for both
  // up- and down-sampling ratios.
  for (in_d, out_d) in [(2usize, 4usize), (3, 6), (16, 27), (16, 12), (4, 2), (8, 3)] {
    let rows = ref_axis_weights(in_d, out_d);
    for (j, r) in rows.iter().enumerate() {
      let s: f64 = r.iter().sum();
      assert!(
        (s - 1.0).abs() < 1e-12,
        "axis_w({in_d},{out_d}) row {j} sum = {s}"
      );
    }
  }
}

#[test]
fn bilinear_downsample_axis_weights_match_hand_derived() {
  // Pin the exact 4->2 antialias axis weights (the canonical downsample
  // case) against the hand-derived fractions, so a center/support drift is
  // caught here directly, not only through a resize.
  let rows = ref_axis_weights(4, 2);
  #[rustfmt::skip]
  let want = [
    [3.0/7.0, 3.0/7.0, 1.0/7.0, 0.0],
    [0.0,     1.0/7.0, 3.0/7.0, 3.0/7.0],
  ];
  for (i, r) in rows.iter().enumerate() {
    for (j, &v) in r.iter().enumerate() {
      assert!(
        (v - want[i][j]).abs() < 1e-12,
        "4->2 weight [{i}][{j}]: got {v}, want {}",
        want[i][j]
      );
    }
  }
}

#[test]
fn bilinear_constant_grid_stays_constant() {
  // The weights are a partition of unity (each output row sums to 1), so a
  // constant grid must resample to the same constant everywhere —
  // independent of the resize ratio (here a mixed up/down resize).
  let grid = Array::from_slice::<f32>(&[5.0f32; 9], &(3, 3, 1)).unwrap();
  let out = bilinear_interpolate(&grid, 6, 2).unwrap();
  assert_eq!(out.shape(), vec![6, 2, 1]);
  for (i, v) in to_vec(&out).iter().enumerate() {
    assert!(
      (v - 5.0).abs() < EPS,
      "idx {i}: constant grid drifted to {v}"
    );
  }
}

#[test]
fn bilinear_multichannel_is_independent_per_channel() {
  // A 2-channel grid where channel 1 = channel 0 + 10. Bilinear is linear
  // and per-channel, so the resized channel 1 must equal resized
  // channel 0 + 10 everywhere.
  let mut data = Vec::new();
  for v in [0.0f32, 1.0, 2.0, 3.0] {
    data.push(v); // channel 0
    data.push(v + 10.0); // channel 1
  }
  let grid = Array::from_slice::<f32>(&data, &(2, 2, 2)).unwrap();
  let out = bilinear_interpolate(&grid, 4, 4).unwrap();
  assert_eq!(out.shape(), vec![4, 4, 2]);
  let got = to_vec(&out);
  for px in 0..16 {
    let c0 = got[px * 2];
    let c1 = got[px * 2 + 1];
    assert!((c1 - (c0 + 10.0)).abs() < 1e-4, "px {px}: c0={c0} c1={c1}");
  }
}

#[test]
fn bilinear_matches_independent_f64_reference_on_3x3_to_5x5() {
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
  let out = bilinear_interpolate(&grid, 5, 5).unwrap();
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
fn bilinear_matches_independent_f64_reference_on_5x5_to_3x3_downsample() {
  // Downsample cross-check (antialias active on both axes) against the
  // independent f64 reference — the path the plain-bilinear formula would
  // get wrong.
  let g: Vec<Vec<f64>> = (0..5)
    .map(|i| (0..5).map(|j| (i * 5 + j) as f64).collect())
    .collect();
  let flat: Vec<f32> = g.iter().flatten().map(|&v| v as f32).collect();
  let grid = Array::from_slice::<f32>(&flat, &(5, 5, 1)).unwrap();
  let out = bilinear_interpolate(&grid, 3, 3).unwrap();
  let got = to_vec(&out);
  let want = ref_resize(&g, 3, 3);
  for i in 0..3 {
    for j in 0..3 {
      let w = want[i][j] as f32;
      let gv = got[i * 3 + j];
      assert!((gv - w).abs() < 1e-4, "({i},{j}): got {gv}, want {w}");
    }
  }
}

#[test]
fn bilinear_rejects_non_rank3_grid() {
  let g2 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let err = bilinear_interpolate(&g2, 4, 4).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn bilinear_rejects_zero_and_oversize_dims() {
  let grid = Array::from_slice::<f32>(&[1.0f32; 4], &(2, 2, 1)).unwrap();
  // zero output dim
  let err = bilinear_interpolate(&grid, 0, 4).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "out_h=0: got {err}");
  // oversize output dim (> MAX_INTERP_DIM)
  let err = bilinear_interpolate(&grid, 4, MAX_INTERP_DIM + 1).unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_)),
    "out_w huge: got {err}"
  );
}

#[test]
fn bilinear_rejects_over_product_weight_table() {
  // Each axis is within the per-axis `MAX_INTERP_DIM` (4096) cap, but the
  // `out * in` weight-table element count exceeds the tighter
  // `MAX_INTERP_WEIGHT_ELEMS` (4 Mi) product cap: 4096 * 2048 = 8 Mi. The
  // weight build must reject it with a typed `CapExceeded` BEFORE the
  // (infallible-`vec!`-would-be) allocation — not abort. The grid is a thin
  // `(4096, 1, 1)` column so the test stays cheap; the height-axis weight build
  // errors out before any matmul.
  let h_in = MAX_INTERP_DIM; // 4096, within the per-axis cap
  let data = vec![0.0f32; h_in]; // (4096, 1, 1)
  let grid = Array::from_slice::<f32>(&data, &(h_in, 1, 1)).unwrap();
  // out_h = 2048: 2048 * 4096 = 8 Mi > MAX_INTERP_WEIGHT_ELEMS (4 Mi).
  let err = bilinear_interpolate(&grid, MAX_INTERP_DIM / 2, 1).unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_)),
    "over-product weight table must be a typed CapExceeded, got {err}"
  );
  // Sanity: the product really does exceed the cap while each axis is in range.
  assert!((MAX_INTERP_DIM / 2) * h_in > MAX_INTERP_WEIGHT_ELEMS);
  assert!(h_in <= MAX_INTERP_DIM && MAX_INTERP_DIM / 2 <= MAX_INTERP_DIM);
}

#[test]
fn bilinear_rejects_over_product_resample_tensor() {
  // Each axis is within the per-axis `MAX_INTERP_DIM` (4096) cap AND each weight
  // table (`out * in`) is within `MAX_INTERP_WEIGHT_ELEMS` (4 Mi), yet a resample
  // TENSOR `dim * dim * C` blows past `MAX_INTERP_RESAMPLE_ELEMS` (64 Mi): with a
  // tiny `(2, 2, 64)` grid, `out_h = out_w = MAX_INTERP_DIM` makes the
  // column-resample / output product `out * out * C` ≈ 1.07e9 elements. The op
  // must reject it with a typed `CapExceeded` BEFORE any device array / matmul is
  // built — the grid stays a negligible 256 f32 so the rejection is the only
  // allocation. This is the hole the per-axis + per-table caps missed.
  let c = 64usize;
  let data = vec![0.0f32; 2 * 2 * c]; // (2, 2, 64) — tiny, 256 f32
  let grid = Array::from_slice::<f32>(&data, &(2, 2, c)).unwrap();
  let err = bilinear_interpolate(&grid, MAX_INTERP_DIM, MAX_INTERP_DIM).unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_)),
    "over-product resample tensor must be a typed CapExceeded, got {err}"
  );
  // Sanity: every AXIS is in range and every WEIGHT TABLE is within its cap, but
  // the resample tensor product exceeds the resample cap — so this is exclusively
  // the resample-product guard firing (not the per-axis or weight-table caps).
  assert!(c <= MAX_INTERP_DIM, "channel axis within the per-axis cap");
  assert!(
    MAX_INTERP_DIM * MAX_INTERP_DIM * c > MAX_INTERP_RESAMPLE_ELEMS,
    "the resample tensor product really does exceed the resample cap"
  );
}

#[test]
fn bilinear_rejects_row_resample_product_with_tiny_height() {
  // The FIRST matmul output is `(out_h, W_in * C)` — `out_h * W_in * C` elements.
  // The spec's adversarial shape is `H_in = W_in = 4096, out_h = out_w = 1024,
  // C = 4096`, where that product ≈ 1.7e10. Reproduce the row-resample blow-up
  // cheaply by keeping `H_in = 1` (so the grid is only `W_in * C` elements) while
  // `W_in * C` stays large: `(1, 4096, 16)` grid with `out_h = 4096` gives a
  // row-resample product `4096 * 4096 * 16` ≈ 268 Mi > 64 Mi, rejected before the
  // matmul — but the grid is just 64 Ki f32.
  let w_in = MAX_INTERP_DIM; // 4096
  let c = 16usize;
  let data = vec![0.0f32; w_in * c]; // (1, 4096, 16) — 64 Ki f32
  let grid = Array::from_slice::<f32>(&data, &(1, w_in, c)).unwrap();
  // out_h large, out_w = 1 (so the output / column products stay small and the
  // ROW-resample product is the one that trips).
  let err = bilinear_interpolate(&grid, MAX_INTERP_DIM, 1).unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_)),
    "over-product row-resample tensor must be a typed CapExceeded, got {err}"
  );
  // Sanity: the row-resample product (out_h * W_in * C) exceeds the cap while the
  // weight tables (out_h * H_in = 4096*1, out_w * W_in = 1*4096) are within theirs.
  assert!(MAX_INTERP_DIM * w_in * c > MAX_INTERP_RESAMPLE_ELEMS);
  assert!(MAX_INTERP_DIM <= MAX_INTERP_WEIGHT_ELEMS && w_in <= MAX_INTERP_WEIGHT_ELEMS);
}

#[test]
fn bilinear_rejects_integer_grid_dtype() {
  // Build a rank-3 int32 grid; the fractional triangle weights cannot
  // resample it, so the op must reject the dtype.
  let g = Array::from_slice::<i32>(&[1, 2, 3, 4], &(2, 2, 1)).unwrap();
  let err = bilinear_interpolate(&g, 4, 4).unwrap_err();
  assert!(matches!(err, Error::UnsupportedDtype(_)), "got {err}");
}
