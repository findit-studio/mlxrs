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

#[test]
fn bicubic_rejects_oversize_output_dim() {
  // An over-cap `out_w` (> MAX_INTERP_DIM) is rejected by the per-axis dimension
  // guard with a typed `CapExceeded` — mirroring the bilinear oversize-dim test —
  // before any host tap table / device gather is built. The grid stays tiny so
  // the rejection is the only allocation.
  let x = Array::from_slice::<f32>(&[1.0f32; 4], &(1, 1, 2, 2)).unwrap();
  let err = bicubic_interpolate(&x, 4, MAX_INTERP_DIM + 1).unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_)),
    "out_w huge: got {err}"
  );
}

#[test]
fn bicubic_rejects_over_product_resample_tensor() {
  // Every axis is within the per-axis `MAX_INTERP_DIM` (4096) cap, yet a resample
  // TENSOR blows past `MAX_INTERP_RESAMPLE_ELEMS` (64 Mi): with a tiny `(1, 1, 2,
  // 2)` grid, `out_h = out_w = MAX_INTERP_DIM` makes the width-gather tensor
  // `B * C * out_h * out_w * TAPS = 1 * 1 * 4096 * 4096 * 5 ≈ 84 Mi` elements. The
  // op must reject it with a typed `CapExceeded` BEFORE any device array / gather
  // is built — the grid stays a negligible 4 f32. This is the hole the per-axis
  // caps alone missed (the bilinear path already caps the analogous product).
  let x = Array::from_slice::<f32>(&[1.0f32; 4], &(1, 1, 2, 2)).unwrap();
  let err = bicubic_interpolate(&x, MAX_INTERP_DIM, MAX_INTERP_DIM).unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_)),
    "over-product resample tensor must be a typed CapExceeded, got {err}"
  );
  // Sanity (compile-time): the width-gather product exceeds the resample cap, so
  // this is exclusively the resample-product guard firing, not the per-axis cap —
  // both axes are exactly at MAX_INTERP_DIM, in range.
  const {
    assert!(MAX_INTERP_DIM * MAX_INTERP_DIM * BICUBIC_TAPS > MAX_INTERP_RESAMPLE_ELEMS);
  }
}

#[test]
fn bicubic_rejects_over_product_height_gather_with_wide_input() {
  // The FIRST gather tensor is `(B, C, out_h * TAPS, W_in)` — `B*C*out_h*TAPS*W_in`
  // elements. Reproduce that blow-up cheaply by keeping the grid thin in height
  // (`H_in = 2`) but wide (`W_in = 4096`) while `out_h = 4096`: the height-gather
  // product `1*1*4096*5*4096 ≈ 84 Mi > 64 Mi` is rejected before the gather, but
  // the grid is just `2 * 4096 = 8 Ki` f32. `out_w = 1` keeps the later products
  // small so the HEIGHT-gather guard is the one that trips.
  let w_in = MAX_INTERP_DIM; // 4096
  let data = vec![0.0f32; 2 * w_in]; // (1, 1, 2, 4096) — 8 Ki f32
  let x = Array::from_slice::<f32>(&data, &(1, 1, 2, w_in)).unwrap();
  let err = bicubic_interpolate(&x, MAX_INTERP_DIM, 1).unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_)),
    "over-product height-gather tensor must be a typed CapExceeded, got {err}"
  );
  // Sanity (compile-time): the height-gather product (`out_h * TAPS * W_in` with
  // `out_h = W_in = MAX_INTERP_DIM`) exceeds the resample cap, while a tap table
  // (`out_dim * TAPS`) stays well within `MAX_INTERP_WEIGHT_ELEMS`.
  const {
    assert!(MAX_INTERP_DIM * BICUBIC_TAPS * MAX_INTERP_DIM > MAX_INTERP_RESAMPLE_ELEMS);
    assert!(MAX_INTERP_DIM * BICUBIC_TAPS <= MAX_INTERP_WEIGHT_ELEMS);
  }
}

#[test]
fn bicubic_aligned_rejects_over_product_resample_tensor() {
  // The cap guards live in the shared `bicubic_resample`, so the CLAP
  // align_corners=True entry point rejects the same adversarial `(out_h, out_w)`
  // with a typed `CapExceeded` (the path that actually feeds `reshape_mel2img`).
  let x = Array::from_slice::<f32>(&[1.0f32; 4], &(1, 1, 2, 2)).unwrap();
  let err = bicubic_interpolate_align_corners(&x, MAX_INTERP_DIM, MAX_INTERP_DIM).unwrap_err();
  assert!(
    matches!(err, Error::CapExceeded(_)),
    "aligned over-product resample tensor must be a typed CapExceeded, got {err}"
  );
}

// ─────────────────────── bicubic align_corners = True ───────────────────────
//
// The HF CLAP `reshape_mel2img` resize runs through PyTorch
// `nn.functional.interpolate(mode="bicubic", align_corners=True)`, which is a
// DIFFERENT kernel from the mlx-vlm Keys' bicubic above:
//
//   1. cubic coefficient `A = -0.75` (not Keys' `-0.5`);
//   2. source-coordinate map `c = i * (in - 1) / (out - 1)` (endpoints aligned,
//      no half-pixel shift), `c = 0` when `out == 1`;
//   3. EDGE-REPLICATE boundary with NO renormalization — an out-of-range tap
//      keeps its full cubic coefficient and reads the clamped (edge) pixel
//      (`upsample_get_value_bounded`); the four coefficients already sum to 1.
//
// The oracle below is reimplemented INDEPENDENTLY from PyTorch's own formulas
// (`aten/src/ATen/native/UpSample.h`'s `cubic_convolution1` / `cubic_convolution2`
// / `get_cubic_upsample_coefficients`, and `UpSampleKernel.cpp`'s
// `upsample_get_value_bounded`), NOT from the code under test — in particular it
// does NOT reuse `ref_cubic` (A = -0.5) and does NOT renormalize.

/// PyTorch `cubic_convolution1(x, A)` for `|x| <= 1`
/// (`aten/src/ATen/native/UpSample.h`), written independently.
fn pt_cubic_conv1(x: f64, a: f64) -> f64 {
  ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
}

/// PyTorch `cubic_convolution2(x, A)` for `1 < |x| < 2`
/// (`aten/src/ATen/native/UpSample.h`), written independently.
fn pt_cubic_conv2(x: f64, a: f64) -> f64 {
  ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
}

/// PyTorch `get_cubic_upsample_coefficients(t)` with `A = -0.75`
/// (`aten/src/ATen/native/UpSample.h`): the four taps `floor-1..floor+2`
/// weighted by the cubic of their distance to the fractional source phase `t`.
fn pt_cubic_coeffs(t: f64) -> [f64; 4] {
  const A: f64 = -0.75;
  [
    pt_cubic_conv2(t + 1.0, A),
    pt_cubic_conv1(t, A),
    pt_cubic_conv1(1.0 - t, A),
    pt_cubic_conv2((1.0 - t) + 1.0, A),
  ]
}

/// Independent f64 reference for one PyTorch `align_corners=True` bicubic axis,
/// returning, per output row, the four `(clamped_source_index, coefficient)`
/// taps — edge-replicate (clamp the index to `[0, in-1]`) and NO renormalization,
/// exactly `upsample_get_value_bounded`. The four coefficients sum to 1 by
/// construction; off-grid taps simply re-read the clamped edge pixel.
fn pt_bicubic_axis_aligned(in_d: usize, out_d: usize) -> Vec<[(usize, f64); 4]> {
  let in_i = in_d as i64;
  let scale = if out_d > 1 {
    (in_d as f64 - 1.0) / (out_d as f64 - 1.0)
  } else {
    0.0
  };
  let mut rows = Vec::with_capacity(out_d);
  for i in 0..out_d {
    let real_x = i as f64 * scale;
    let input_index = real_x.floor() as i64;
    let t = real_x - input_index as f64;
    let coeffs = pt_cubic_coeffs(t);
    let mut taps = [(0usize, 0.0f64); 4];
    for (k, slot) in taps.iter_mut().enumerate() {
      let idx = input_index - 1 + k as i64;
      let clamped = idx.max(0).min(in_i - 1) as usize; // upsample_get_value_bounded
      *slot = (clamped, coeffs[k]);
    }
    rows.push(taps);
  }
  rows
}

/// Independent f64 PyTorch `align_corners=True` bicubic resize of a
/// single-channel grid (separable height-then-width).
fn pt_bicubic_resize_aligned(grid: &[Vec<f64>], out_h: usize, out_w: usize) -> Vec<Vec<f64>> {
  let h_in = grid.len();
  let w_in = grid[0].len();
  let wy = pt_bicubic_axis_aligned(h_in, out_h);
  let wx = pt_bicubic_axis_aligned(w_in, out_w);
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

#[test]
fn bicubic_aligned_identity_same_size_is_exact_passthrough() {
  // out == in: `scale = (n-1)/(n-1) = 1`, so center i maps to source i exactly →
  // one unit tap per row → the resize is the input unchanged.
  let flat: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
  let x = Array::from_slice::<f32>(&flat, &(1, 1, 3, 3)).unwrap();
  let out = bicubic_interpolate_align_corners(&x, 3, 3).unwrap();
  let got = to_vec(&out);
  for (a, b) in got.iter().zip(flat.iter()) {
    assert!((a - b).abs() < EPS, "aligned identity differs: {a} vs {b}");
  }
}

#[test]
fn bicubic_aligned_preserves_corner_values_on_upsample() {
  // The defining property of align_corners=True: the four corner output pixels
  // equal the four corner input pixels EXACTLY (endpoints map to endpoints), so
  // upsampling a grid leaves its corners pinned (unlike align_corners=False).
  let flat: [f32; 9] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
  let x = Array::from_slice::<f32>(&flat, &(1, 1, 3, 3)).unwrap();
  let (oh, ow) = (7usize, 5usize);
  let out = bicubic_interpolate_align_corners(&x, oh, ow).unwrap();
  let v = to_vec(&out);
  let at = |r: usize, c: usize| v[r * ow + c];
  assert!(
    (at(0, 0) - 10.0).abs() < 1e-3,
    "top-left corner not pinned: {}",
    at(0, 0)
  );
  assert!(
    (at(0, ow - 1) - 30.0).abs() < 1e-3,
    "top-right corner not pinned: {}",
    at(0, ow - 1)
  );
  assert!(
    (at(oh - 1, 0) - 70.0).abs() < 1e-3,
    "bottom-left corner not pinned: {}",
    at(oh - 1, 0)
  );
  assert!(
    (at(oh - 1, ow - 1) - 90.0).abs() < 1e-3,
    "bottom-right corner not pinned: {}",
    at(oh - 1, ow - 1)
  );
}

#[test]
fn bicubic_aligned_matches_independent_pytorch_reference_on_4x4_to_7x7_upsample() {
  // Cross-check the full align_corners=True op against the INDEPENDENT PyTorch
  // (A = -0.75, edge-replicate, no-renormalize) f64 oracle on an interior-heavy
  // upsample (4x4 → 7x7). 7 = 2*(4-1)+1 makes the aligned source coordinates land
  // on a mix of exact-integer and half-integer phases, exercising several `t`.
  let mut g = vec![vec![0.0f64; 4]; 4];
  for (r, row) in g.iter_mut().enumerate() {
    for (c, cell) in row.iter_mut().enumerate() {
      *cell = ((r * 4 + c) as f64) * 0.3 - 1.1; // deterministic, signed
    }
  }
  let flat: Vec<f32> = g.iter().flatten().map(|&v| v as f32).collect();
  let x = Array::from_slice::<f32>(&flat, &(1, 1, 4, 4)).unwrap();
  let got = to_vec(&bicubic_interpolate_align_corners(&x, 7, 7).unwrap());
  let want = pt_bicubic_resize_aligned(&g, 7, 7);
  for (i, wrow) in want.iter().enumerate() {
    for (j, &w) in wrow.iter().enumerate() {
      let g = got[i * 7 + j];
      assert!((g as f64 - w).abs() < 1e-4, "aligned ({i},{j}) {g} vs {w}");
    }
  }
}

#[test]
fn bicubic_aligned_tall_input_gathers_rows_before_casting() {
  // A tall input (large `H_in`, tiny output) is exactly the case the height
  // gather bounds: only `out_h * BICUBIC_TAPS` rows are read from `x`, so the
  // resample never materializes a full-input f32 copy whose size scales with the
  // uncapped `H_in`. Verify the gather-before-cast path is (1) numerically
  // faithful to the independent PyTorch f64 oracle for a tall f32 input, and
  // (2) for an f16 input, equivalent to the reference's cast-input-to-f32-first
  // order within f16 output rounding — gathering then casting equals casting
  // then gathering, since the gather only copies elements.
  let (in_h, w, out_h) = (48usize, 3usize, 6usize);
  let mut g = vec![vec![0.0f64; w]; in_h];
  for (r, row) in g.iter_mut().enumerate() {
    for (c, cell) in row.iter_mut().enumerate() {
      // Deterministic, signed, modest magnitude so f16 stays precise.
      *cell = (((r * w + c) as f64) * 0.013 - 0.31).sin();
    }
  }
  let flat: Vec<f32> = g.iter().flatten().map(|&v| v as f32).collect();
  let x_f32 = Array::from_slice::<f32>(&flat, &(1, 1, in_h, w)).unwrap();

  // (1) tall f32 input vs the independent PyTorch oracle.
  let got_f32 = to_vec(&bicubic_interpolate_align_corners(&x_f32, out_h, w).unwrap());
  let want = pt_bicubic_resize_aligned(&g, out_h, w);
  for (i, wrow) in want.iter().enumerate() {
    for (j, &wv) in wrow.iter().enumerate() {
      let gv = got_f32[i * w + j];
      assert!(
        (gv as f64 - wv).abs() < 1e-4,
        "tall f32 aligned ({i},{j}) {gv} vs {wv}"
      );
    }
  }

  // (2) tall f16 input: gather-then-cast must match the f32 path within f16
  // output rounding, and the op must preserve the f16 input dtype on the way out.
  let x_f16 = astype(&x_f32, Dtype::F16).unwrap();
  let out_f16 = bicubic_interpolate_align_corners(&x_f16, out_h, w).unwrap();
  assert_eq!(
    out_f16.dtype().unwrap(),
    Dtype::F16,
    "f16 input must yield an f16 result"
  );
  let got_f16 = to_vec(&astype(&out_f16, Dtype::F32).unwrap());
  for (k, (&a, &b)) in got_f16.iter().zip(got_f32.iter()).enumerate() {
    assert!(
      (a - b).abs() < 5e-3,
      "tall f16 gather-then-cast diverges from the f32 path at {k}: {a} vs {b}"
    );
  }
}

#[test]
fn bicubic_aligned_1001_to_1024_row_matches_pytorch_formula_interior_and_boundary() {
  // The exact resize CLAP's `reshape_mel2img` performs on the time axis: a known
  // 1001-long input row upsampled to 1024 with align_corners=True bicubic. Drive
  // it as a `(1, 1, 1, 1001)` width resize so the op runs the real 1-D path, and
  // compare every output column against the INDEPENDENT PyTorch oracle
  // (A = -0.75, edge-replicate, no-renormalize) — including the boundary columns
  // 0 and 1023, where the off-grid tap makes edge-replicate vs the old
  // clamp-renormalize differ.
  let in_w = 1001usize;
  let out_w = 1024usize;
  // A deterministic non-linear row (so the cubic is actually exercised, not a
  // ramp the kernel reproduces trivially).
  let row: Vec<f64> = (0..in_w)
    .map(|i| (i as f64 * 0.017).sin() * 3.0 + (i as f64) * 0.002)
    .collect();
  let grid = vec![row];
  let flat: Vec<f32> = grid[0].iter().map(|&v| v as f32).collect();
  let x = Array::from_slice::<f32>(&flat, &(1, 1, 1, in_w)).unwrap();
  let got = to_vec(&bicubic_interpolate_align_corners(&x, 1, out_w).unwrap());
  let want = pt_bicubic_resize_aligned(&grid, 1, out_w);
  for j in 0..out_w {
    let w = want[0][j] as f32;
    assert!(
      (got[j] - w).abs() < 1e-3,
      "1001->1024 col {j}: got {}, want {w}",
      got[j]
    );
  }
  // Pin the two boundary columns explicitly: column 0's source coordinate is
  // exactly input 0 (its `floor-1 = -1` tap edge-replicates input[0]), and column
  // 1023's is exactly input 1000 (its `floor+2 = 1001` tap edge-replicates
  // input[1000]) — align_corners pins both endpoints to the input samples.
  assert!(
    (got[0] as f64 - grid[0][0]).abs() < 1e-4,
    "left endpoint not pinned to input[0]: {} vs {}",
    got[0],
    grid[0][0]
  );
  assert!(
    (got[out_w - 1] as f64 - grid[0][in_w - 1]).abs() < 1e-4,
    "right endpoint not pinned to input[last]: {} vs {}",
    got[out_w - 1],
    grid[0][in_w - 1]
  );
}

#[test]
fn bicubic_aligned_4_to_8_edge_columns_are_pytorch_edge_replicate() {
  // A small edge-heavy upsample (4 → 8, align_corners=True) where the boundary
  // handling is DECISIVE: at output column 0 the source coordinate is 0, so the
  // `floor-1 = -1` tap is off-grid; PyTorch edge-replicates input[0] there (and
  // does NOT renormalize), while the old clamp-renormalize convention would
  // redistribute that tap's weight. Compare the whole 1-D resize against the
  // independent PyTorch oracle, then assert the boundary columns specifically.
  let in_w = 4usize;
  let out_w = 8usize;
  let row = vec![vec![2.0f64, -1.0, 5.0, 0.5]]; // signed, non-monotone
  let flat: Vec<f32> = row[0].iter().map(|&v| v as f32).collect();
  let x = Array::from_slice::<f32>(&flat, &(1, 1, 1, in_w)).unwrap();
  let got = to_vec(&bicubic_interpolate_align_corners(&x, 1, out_w).unwrap());
  let want = pt_bicubic_resize_aligned(&row, 1, out_w);
  for j in 0..out_w {
    let w = want[0][j] as f32;
    assert!(
      (got[j] - w).abs() < 1e-4,
      "4->8 edge col {j}: got {}, want {w}",
      got[j]
    );
  }
  // Endpoints pin exactly to the input corners (align_corners property), and
  // column 0 specifically equals the PyTorch edge-replicate result computed by
  // hand from the A = -0.75 coefficients at phase t = 0 (only the center tap is
  // nonzero, so it is exactly input[0]).
  assert!((got[0] as f64 - row[0][0]).abs() < 1e-4, "col0: {}", got[0]);
  assert!(
    (got[out_w - 1] as f64 - row[0][in_w - 1]).abs() < 1e-4,
    "col7: {}",
    got[out_w - 1]
  );
}

#[test]
fn bicubic_aligned_pytorch_a_minus_0_75_differs_from_old_keys_a_minus_0_5() {
  // Regression: the CLAP align_corners=True path must use PyTorch's A = -0.75
  // edge-replicate kernel, NOT the previous Keys' A = -0.5 zero-weight +
  // renormalize kernel. Recompute what the OLD aligned builder produced (the same
  // aligned coordinate map, but `ref_cubic` = A = -0.5, off-grid taps dropped and
  // renormalized — written inline so this pins a real kernel change) and assert
  // the live op DIFFERS on a boundary-heavy upsample where both the coefficient
  // and the edge handling bite.
  fn old_keys_aligned_axis(in_d: usize, out_d: usize) -> Vec<Vec<(usize, f64)>> {
    let in_i = in_d as i64;
    let scale = if out_d > 1 {
      (in_d as f64 - 1.0) / (out_d as f64 - 1.0)
    } else {
      0.0
    };
    let mut rows = Vec::with_capacity(out_d);
    for i in 0..out_d {
      let center = i as f64 * scale;
      let start = (center - 2.0).floor() as i64 + 1;
      let mut taps: Vec<(i64, f64)> = Vec::new();
      let mut tot = 0.0;
      for k in 0..5i64 {
        let p = start + k;
        if p >= 0 && p < in_i {
          let w = ref_cubic(center - p as f64); // A = -0.5 (old)
          taps.push((p, w));
          tot += w;
        }
      }
      let inv = 1.0 / (tot + 1e-8); // renormalize (old)
      rows.push(
        taps
          .into_iter()
          .map(|(p, w)| (p as usize, w * inv))
          .collect(),
      );
    }
    rows
  }
  let row = vec![1.0f64, 5.0, 2.0, 8.0, 3.0];
  let in_w = row.len();
  let out_w = 9usize;
  let flat: Vec<f32> = row.iter().map(|&v| v as f32).collect();
  let x = Array::from_slice::<f32>(&flat, &(1, 1, 1, in_w)).unwrap();
  let got = to_vec(&bicubic_interpolate_align_corners(&x, 1, out_w).unwrap());
  // Old A=-0.5 + zero-renormalize aligned resize, computed inline.
  let old_axis = old_keys_aligned_axis(in_w, out_w);
  let old: Vec<f64> = old_axis
    .iter()
    .map(|taps| taps.iter().map(|&(p, w)| w * row[p]).sum())
    .collect();
  let differs = got
    .iter()
    .zip(old.iter())
    .any(|(g, o)| (*g as f64 - o).abs() > 1e-3);
  assert!(
    differs,
    "new A=-0.75 edge-replicate path matched the old A=-0.5 renormalize path: got={got:?} old={old:?}"
  );
  // And the live op must still MATCH the PyTorch oracle (positive pin that the
  // difference is toward PyTorch, not arbitrary).
  let want = pt_bicubic_resize_aligned(&[row], 1, out_w);
  for j in 0..out_w {
    assert!(
      (got[j] as f64 - want[0][j]).abs() < 1e-3,
      "col {j}: got {} vs pytorch {}",
      got[j],
      want[0][j]
    );
  }
}

#[test]
fn bicubic_aligned_differs_from_unaligned_on_upsample() {
  // align_corners True vs False must NOT be the same resampling (the CLAP
  // faithfulness point): upsampling a non-constant grid gives different
  // interiors under the two coordinate maps (and now also under the different
  // coefficient + edge handling).
  let x = Array::from_slice::<f32>(&[1.0f32, 5.0, 2.0, 8.0], &(1, 1, 1, 4)).unwrap();
  let aligned = to_vec(&bicubic_interpolate_align_corners(&x, 1, 9).unwrap());
  let unaligned = to_vec(&bicubic_interpolate(&x, 1, 9).unwrap());
  let differs = aligned
    .iter()
    .zip(unaligned.iter())
    .any(|(a, b)| (a - b).abs() > 1e-3);
  assert!(
    differs,
    "align_corners True and False produced identical upsamples"
  );
}

#[test]
fn bicubic_aligned_rejects_non_rank4_input() {
  let g3 = Array::from_slice::<f32>(&[1.0f32; 8], &(2, 2, 2)).unwrap();
  let err = bicubic_interpolate_align_corners(&g3, 4, 4).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn bicubic_aligned_rejects_zero_dims() {
  let x = Array::from_slice::<f32>(&[1.0f32; 4], &(1, 1, 2, 2)).unwrap();
  let err = bicubic_interpolate_align_corners(&x, 4, 0).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "out_w=0: got {err}");
}
