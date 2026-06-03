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

// ───────────────────────────── bicubic ─────────────────────────────
//
// Reference: `mlx-vlm/mlx_vlm/models/kernels.py`'s `_bicubic_interpolate_mlx`
// with the SigLIP2 defaults `align_corners=False`, `antialias=False` (Keys'
// cubic, `a = -0.5`). The expected values are computed by an independent f64
// reimplementation of that algorithm (NOT by delegating to the code under
// test), and the canonical small grids are also pinned against hand-computed
// fractions.

/// Independent f64 Keys'-cubic kernel (`a = -0.5`), written separately from the
/// `cubic_weight` under test.
fn ref_cubic(t: f64) -> f64 {
  let a = -0.5;
  let at = t.abs();
  let at2 = at * at;
  let at3 = at2 * at;
  if at <= 1.0 {
    (a + 2.0) * at3 - (a + 3.0) * at2 + 1.0
  } else if at < 2.0 {
    a * at3 - 5.0 * a * at2 + 8.0 * a * at - 4.0 * a
  } else {
    0.0
  }
}

/// Independent f64 reference for one bicubic resampling axis: returns, per
/// output row, the `(source_index, weight)` taps (only the in-bounds,
/// renormalized ones — the zero-weight off-grid taps are dropped). Mirrors
/// `_weights_1d` (support = 2, fs = 1) without reusing the production build.
fn ref_bicubic_axis(in_d: usize, out_d: usize) -> Vec<Vec<(usize, f64)>> {
  let in_i = in_d as i64;
  let mut rows = Vec::with_capacity(out_d);
  for i in 0..out_d {
    let center = (i as f64 + 0.5) / out_d as f64 * in_d as f64 - 0.5;
    let start = (center - 2.0).floor() as i64 + 1;
    let mut taps: Vec<(i64, f64)> = Vec::new();
    let mut tot = 0.0;
    for k in 0..5i64 {
      let p = start + k;
      if p >= 0 && p < in_i {
        let w = ref_cubic(center - p as f64);
        taps.push((p, w));
        tot += w;
      }
    }
    let inv = 1.0 / (tot + 1e-8);
    rows.push(
      taps
        .into_iter()
        .map(|(p, w)| (p as usize, w * inv))
        .collect(),
    );
  }
  rows
}

/// Independent f64 bicubic resize of a single-channel `(H, W)` grid
/// (`B = C = 1`), separable height-then-width like the reference.
fn ref_bicubic_resize(grid: &[Vec<f64>], out_h: usize, out_w: usize) -> Vec<Vec<f64>> {
  let h_in = grid.len();
  let w_in = grid[0].len();
  let wy = ref_bicubic_axis(h_in, out_h);
  let wx = ref_bicubic_axis(w_in, out_w);
  // Height resample → (out_h, w_in).
  let mut tmp = vec![vec![0.0f64; w_in]; out_h];
  for (i, trow) in tmp.iter_mut().enumerate() {
    for (x, cell) in trow.iter_mut().enumerate() {
      let mut s = 0.0;
      for &(p, w) in &wy[i] {
        s += w * grid[p][x];
      }
      *cell = s;
    }
  }
  // Width resample → (out_h, out_w).
  let mut out = vec![vec![0.0f64; out_w]; out_h];
  for (i, orow) in out.iter_mut().enumerate() {
    for (j, cell) in orow.iter_mut().enumerate() {
      let mut s = 0.0;
      for &(p, w) in &wx[j] {
        s += w * tmp[i][p];
      }
      *cell = s;
    }
  }
  out
}

/// Run `bicubic_interpolate` on a single-channel `(H_in, W_in)` grid
/// (`B = C = 1`) and return the flat `out_h * out_w` row-major output.
fn bicubic_single_channel(grid: &[Vec<f64>], out_h: usize, out_w: usize) -> Vec<f32> {
  let h_in = grid.len();
  let w_in = grid[0].len();
  let flat: Vec<f32> = grid.iter().flatten().map(|&v| v as f32).collect();
  let x = Array::from_slice::<f32>(&flat, &(1, 1, h_in, w_in)).unwrap();
  let out = bicubic_interpolate(&x, out_h, out_w).unwrap();
  assert_eq!(out.shape(), vec![1, 1, out_h, out_w]);
  to_vec(&out)
}

#[test]
fn bicubic_identity_same_size_is_exact_passthrough() {
  // out == in on both axes: the source center lands exactly on each pixel, so
  // the only nonzero cubic tap is weight 1 → the resize is the input unchanged.
  #[rustfmt::skip]
  let g = vec![
    vec![0.0f64, 1.0, 2.0],
    vec![3.0, 4.0, 5.0],
    vec![6.0, 7.0, 8.0],
  ];
  let got = bicubic_single_channel(&g, 3, 3);
  let want: Vec<f32> = g.iter().flatten().map(|&v| v as f32).collect();
  for (i, (gv, w)) in got.iter().zip(want.iter()).enumerate() {
    assert!((gv - w).abs() < EPS, "idx {i}: got {gv}, want {w}");
  }
}

#[test]
fn bicubic_axis_weights_renormalize_to_one() {
  // Each output row's surviving (in-bounds) cubic taps renormalize to sum ~1.
  // `kernels.py` divides by `sum(w) + 1e-8` (an unconditional epsilon, NOT a
  // guard for a zero sum), so the renormalized sum is exactly `tot/(tot+1e-8)`
  // — a hair below 1 by `~1e-8/tot`. The tolerance accommodates that faithful
  // epsilon (`tot` is O(1) for these ratios, so the deviation is `~1e-8`).
  for (in_d, out_d) in [(4usize, 8usize), (16, 27), (16, 12), (4, 2), (3, 7), (8, 3)] {
    for (j, taps) in ref_bicubic_axis(in_d, out_d).iter().enumerate() {
      let s: f64 = taps.iter().map(|&(_, w)| w).sum();
      assert!(
        (s - 1.0).abs() < 1e-6,
        "bicubic axis({in_d},{out_d}) row {j} sum = {s}"
      );
    }
  }
}

#[test]
fn bicubic_upsample_1x2_to_1x4_matches_hand_computed() {
  // A 1x2 horizontal ramp [0, 1] upsampled to 1x4. For each output i the source
  // center is c = (i+0.5)/4*2 - 0.5 ∈ {-0.25, 0.25, 0.75, 1.25}. With only two
  // input pixels the in-bounds taps are {0, 1}; the renormalized cubic blend of
  // values {0, 1} is computed by the independent f64 reference and pinned here.
  // Note Keys' cubic (a = -0.5) deliberately OVERSHOOTS beyond the [0, 1] data
  // range at the extrapolated endpoints (the cubic-ringing characteristic) —
  // got ≈ [-0.088, 0.207, 0.793, 1.088] — which is exactly what the reference
  // produces, so the op is pinned against the reference, not a (wrong)
  // in-range / monotonic expectation.
  let g = vec![vec![0.0f64, 1.0]];
  let got = bicubic_single_channel(&g, 1, 4);
  let want = ref_bicubic_resize(&g, 1, 4);
  for j in 0..4 {
    let w = want[0][j] as f32;
    assert!(
      (got[j] - w).abs() < 1e-5,
      "col {j}: got {}, want {w}",
      got[j]
    );
  }
  // The endpoints overshoot the data range (cubic ringing) — a positive pin
  // that this is the cubic kernel, not a clamped bilinear.
  assert!(
    got[0] < 0.0,
    "left endpoint must undershoot (cubic ringing): {got:?}"
  );
  assert!(
    got[3] > 1.0,
    "right endpoint must overshoot (cubic ringing): {got:?}"
  );
}

#[test]
fn bicubic_constant_grid_stays_constant() {
  // The renormalized cubic weights are a partition of unity per output row, so a
  // constant grid must resample to the same constant everywhere — independent of
  // the (mixed up/down) resize ratio.
  let g = vec![vec![5.0f64; 4]; 4];
  let got = bicubic_single_channel(&g, 7, 2);
  for (i, v) in got.iter().enumerate() {
    assert!(
      (v - 5.0).abs() < 1e-4,
      "idx {i}: constant grid drifted to {v}"
    );
  }
}

#[test]
fn bicubic_matches_independent_f64_reference_on_4x4_to_7x7_upsample() {
  // Cross-check the full op against the independent f64 reference on a
  // non-trivial grid + non-square-ratio upsample (4x4 → 7x7).
  let g: Vec<Vec<f64>> = (0..4)
    .map(|i| (0..4).map(|j| (i * 4 + j) as f64 * 0.5 - 3.0).collect())
    .collect();
  let got = bicubic_single_channel(&g, 7, 7);
  let want = ref_bicubic_resize(&g, 7, 7);
  for i in 0..7 {
    for j in 0..7 {
      let w = want[i][j] as f32;
      let gv = got[i * 7 + j];
      assert!((gv - w).abs() < 1e-4, "({i},{j}): got {gv}, want {w}");
    }
  }
}

#[test]
fn bicubic_matches_independent_f64_reference_on_8x8_to_5x5_downsample() {
  // Downsample cross-check (5-tap support active on both axes) against the
  // independent f64 reference.
  let g: Vec<Vec<f64>> = (0..8)
    .map(|i| (0..8).map(|j| (i * 8 + j) as f64).collect())
    .collect();
  let got = bicubic_single_channel(&g, 5, 5);
  let want = ref_bicubic_resize(&g, 5, 5);
  for i in 0..5 {
    for j in 0..5 {
      let w = want[i][j] as f32;
      let gv = got[i * 5 + j];
      assert!((gv - w).abs() < 1e-4, "({i},{j}): got {gv}, want {w}");
    }
  }
}

#[test]
fn bicubic_is_independent_per_channel_and_batch() {
  // A (2, 2, 2, 2) grid: channel/batch index 1 = index 0 + 10. Bicubic is linear
  // and per-(B,C), so resized slot 1 must equal resized slot 0 + 10 everywhere.
  // Build (B=1, C=2, 2, 2) where C1 = C0 + 10.
  #[rustfmt::skip]
  let c0 = [0.0f32, 1.0, 2.0, 3.0];
  let mut data = Vec::new();
  data.extend_from_slice(&c0); // channel 0
  for v in c0 {
    data.push(v + 10.0); // channel 1
  }
  let x = Array::from_slice::<f32>(&data, &(1, 2, 2, 2)).unwrap();
  let out = bicubic_interpolate(&x, 4, 4).unwrap();
  assert_eq!(out.shape(), vec![1, 2, 4, 4]);
  let got = to_vec(&out);
  for px in 0..16 {
    let a = got[px]; // channel 0 plane
    let b = got[16 + px]; // channel 1 plane
    assert!((b - (a + 10.0)).abs() < 1e-3, "px {px}: c0={a} c1={b}");
  }
}

#[test]
fn bicubic_rejects_non_rank4_input() {
  let g3 = Array::from_slice::<f32>(&[1.0f32; 8], &(2, 2, 2)).unwrap();
  let err = bicubic_interpolate(&g3, 4, 4).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn bicubic_rejects_zero_dims() {
  let x = Array::from_slice::<f32>(&[1.0f32; 4], &(1, 1, 2, 2)).unwrap();
  let err = bicubic_interpolate(&x, 0, 4).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "out_h=0: got {err}");
}

#[test]
fn bicubic_rejects_integer_input_dtype() {
  // Fractional cubic weights cannot resample an integer grid → dtype rejected.
  let g = Array::from_slice::<i32>(&[1, 2, 3, 4], &(1, 1, 2, 2)).unwrap();
  let err = bicubic_interpolate(&g, 4, 4).unwrap_err();
  assert!(matches!(err, Error::UnsupportedDtype(_)), "got {err}");
}
