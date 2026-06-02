//! Tests for the SigLIP2 NaFlex preprocessing.
//!
//! The `patch_grid` expected values are the `siglip2-naflex` crate's
//! oracle table (itself validated against upstream `transformers`'
//! `get_image_size_for_max_num_patches`), so this port's sizing is pinned
//! bit-for-bit against the authoritative reference.

use super::*;

const PATCH: u32 = 16;
const CHANNELS: u32 = 3;
const M: u32 = 256;

/// `((height, width), (H_p, W_p))` — a `patch_grid` reference case.
type GridCase = ((u32, u32), (u32, u32));

fn to_vec_i32(a: &Array) -> Vec<i32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<i32>().unwrap()
}

fn to_vec_f32(a: &Array) -> Vec<f32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

#[test]
fn patch_grid_matches_oracle_reference_table() {
  // (height, width) -> (H_p, W_p), the exact `siglip2-naflex` oracle
  // table (P = 16, M = 256). The (3, 39) -> (4, 52) row is the edge-case
  // regression the oracle's 1e-5 termination fixes (a looser eps drifts
  // to (4, 53)).
  let cases: &[GridCase] = &[
    ((16, 16), (16, 16)),
    ((100, 100), (16, 16)),
    ((224, 224), (16, 16)),
    ((1080, 1920), (12, 21)),
    ((1920, 1080), (21, 12)),
    ((2160, 4096), (12, 21)),
    ((1024, 1), (256, 1)),
    ((3, 39), (4, 52)),
  ];
  for ((h, w), expected) in cases {
    let got = patch_grid(*h, *w, PATCH, M);
    assert_eq!(
      got, *expected,
      "patch_grid(h={h}, w={w}) — expected {expected:?}, got {got:?}"
    );
  }
}

#[test]
fn patch_grid_fixture_sizes_within_budget() {
  // The aspect-preserving target-size math for the fixture image sizes
  // (interpreted W x H), each verified to stay within the 256 budget.
  // (height, width) -> (H_p, W_p):
  let cases: &[GridCase] = &[
    // 900x600 (W=900, H=600)
    ((600, 900), (13, 19)),
    // 360x1280 (W=360, H=1280)
    ((1280, 360), (29, 8)),
    // 512x512
    ((512, 512), (16, 16)),
  ];
  for ((h, w), expected) in cases {
    let got = patch_grid(*h, *w, PATCH, M);
    assert_eq!(got, *expected, "patch_grid(h={h}, w={w})");
    assert!(
      u64::from(got.0) * u64::from(got.1) <= u64::from(M),
      "grid {got:?} exceeds budget {M}"
    );
  }
}

#[test]
fn patch_grid_budget_respected_on_assorted_inputs() {
  for (h, w) in [
    (1u32, 1u32),
    (1, 2048),
    (2048, 1),
    (4096, 4096),
    (640, 480),
    (3840, 2160),
    (32, 7),
    (7, 32),
  ] {
    let (h_p, w_p) = patch_grid(h, w, PATCH, M);
    assert!(
      h_p >= 1 && w_p >= 1,
      "{h}x{w} -> {h_p}x{w_p} has a zero axis"
    );
    assert!(
      u64::from(h_p) * u64::from(w_p) <= u64::from(M),
      "{h}x{w} -> {h_p}x{w_p} exceeds budget {M}"
    );
  }
}

#[test]
fn even_grid_mask_and_spatial_shapes() {
  // 512x512 -> 16x16 = 256 active patches (even grid, fills the whole
  // budget). Mask is all ones; spatial = [16, 16].
  let rgb = vec![128u8; (512 * 512 * 3) as usize];
  let out = preprocess(&rgb, 512, 512, PATCH, CHANNELS, M).unwrap();
  let ss = to_vec_i32(&out.spatial_shapes);
  assert_eq!(ss, vec![16, 16]);
  let mask = to_vec_i32(&out.pixel_attention_mask);
  assert_eq!(mask.len(), M as usize);
  assert!(mask.iter().all(|&m| m == 1), "even full grid: all ones");
}

#[test]
fn odd_grid_mask_and_spatial_shapes() {
  // A 900x600 image (W=900, H=600) -> (13, 19) = 247 active patches (odd
  // product < budget). The first 247 mask entries are 1, the rest 0.
  let (w, h) = (900u32, 600u32);
  let rgb = vec![64u8; (w * h * 3) as usize];
  let out = preprocess(&rgb, w, h, PATCH, CHANNELS, M).unwrap();
  let ss = to_vec_i32(&out.spatial_shapes);
  assert_eq!(ss, vec![13, 19]);
  let n_active = 13 * 19;
  assert_eq!(n_active, 247);
  let mask = to_vec_i32(&out.pixel_attention_mask);
  assert_eq!(mask.len(), M as usize);
  for (i, &m) in mask.iter().enumerate() {
    let want = if i < n_active { 1 } else { 0 };
    assert_eq!(m, want, "mask[{i}]");
  }
}

#[test]
fn pixel_values_shape_and_normalization_and_zero_padding() {
  // 16x16 single-patch image, every pixel R=10,G=20,B=30. After
  // normalize `x/127.5 - 1`, the first three floats are R,G,B in order
  // (channel-innermost, no transpose), and all padding rows are exactly
  // zero.
  let rgb: Vec<u8> = std::iter::repeat_n([10u8, 20, 30], 16 * 16)
    .flatten()
    .collect();
  let out = preprocess(&rgb, 16, 16, PATCH, CHANNELS, M).unwrap();
  assert_eq!(out.pixel_values.shape(), vec![M as usize, 768]);
  let pv = to_vec_f32(&out.pixel_values);

  let r = 10.0f32 / 127.5 - 1.0;
  let g = 20.0f32 / 127.5 - 1.0;
  let b = 30.0f32 / 127.5 - 1.0;
  assert!((pv[0] - r).abs() < 1e-5, "pv[0]=R got {}", pv[0]);
  assert!((pv[1] - g).abs() < 1e-5, "pv[1]=G got {}", pv[1]);
  assert!((pv[2] - b).abs() < 1e-5, "pv[2]=B got {}", pv[2]);

  // The single active patch is index 0 (1x1 grid); rows 1..256 are
  // padding and must be exactly zero.
  let ss = to_vec_i32(&out.spatial_shapes);
  let n_active = (ss[0] * ss[1]) as usize;
  for patch_i in n_active..(M as usize) {
    for j in 0..768 {
      assert_eq!(
        pv[patch_i * 768 + j],
        0.0,
        "padding patch {patch_i} idx {j}"
      );
    }
  }
}

#[test]
fn normalize_row_rgba_endpoints_and_drops_alpha() {
  // 0 -> -1, 255 -> 1 (the [-1, 1] SigLIP range). The source is RGBA: the
  // alpha byte (here 77) must be dropped, never appearing in the RGB output.
  let mut dst = [0.0f32; 3];
  normalize_row_rgba(&[0, 128, 255, 77], &mut dst);
  assert!((dst[0] - (-1.0)).abs() < 1e-6);
  assert!((dst[1] - (128.0 / 127.5 - 1.0)).abs() < 1e-6);
  assert!((dst[2] - 1.0).abs() < 1e-6, "255 -> {}", dst[2]);
}

#[test]
fn normalize_row_rgba_multi_pixel_skips_each_alpha() {
  // Two RGBA pixels: the per-pixel alpha (9, 200) is skipped and the RGB
  // triples land contiguously in the 3-channel output row.
  let mut dst = [0.0f32; 6];
  normalize_row_rgba(&[10, 20, 30, 9, 40, 50, 60, 200], &mut dst);
  for (i, &v) in [10u8, 20, 30, 40, 50, 60].iter().enumerate() {
    let want = f32::from(v) / 127.5 - 1.0;
    assert!(
      (dst[i] - want).abs() < 1e-6,
      "dst[{i}]={} want {want}",
      dst[i]
    );
  }
}

#[test]
fn pixel_values_uniform_image_after_resize_drops_alpha_correctly() {
  // A uniform 32x32 image (every pixel R=10,G=20,B=30) is RESIZED (here scaled
  // up to fill the 256-patch budget → a 16x16 patch / 256x256 px grid), so the
  // patchify reads the RGBA8 buffer the resize kernel produces (NOT the source
  // bytes). A uniform image resizes to the same uniform color, so EVERY active
  // RGB float must be exactly the normalized R/G/B — proving the RGBA→RGB
  // alpha-drop patchify (replacing the owned `to_rgb8()` copy) reads the right
  // 3 of every 4 bytes and never leaks the (255) alpha byte into a colour slot.
  let rgb: Vec<u8> = std::iter::repeat_n([10u8, 20, 30], 32 * 32)
    .flatten()
    .collect();
  let out = preprocess(&rgb, 32, 32, PATCH, CHANNELS, M).unwrap();
  let ss = to_vec_i32(&out.spatial_shapes);
  // 32x32 scales up to fill the budget → 16x16 patches (256 active, the full M).
  assert_eq!(ss, vec![16, 16], "32x32 → 16x16 patch grid (fills budget)");
  let pv = to_vec_f32(&out.pixel_values);
  let r = 10.0f32 / 127.5 - 1.0;
  let g = 20.0f32 / 127.5 - 1.0;
  let b = 30.0f32 / 127.5 - 1.0;
  let n_active = (ss[0] * ss[1]) as usize; // 256 patches
  for patch_i in 0..n_active {
    for px in 0..(PATCH * PATCH) as usize {
      let base = patch_i * 768 + px * 3;
      assert!(
        (pv[base] - r).abs() < 1e-5,
        "patch {patch_i} px {px} R = {}",
        pv[base]
      );
      assert!((pv[base + 1] - g).abs() < 1e-5, "patch {patch_i} px {px} G");
      assert!((pv[base + 2] - b).abs() < 1e-5, "patch {patch_i} px {px} B");
    }
  }
}

#[test]
fn rejects_zero_dimensions() {
  let err = preprocess(&[], 0, 480, PATCH, CHANNELS, M).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "width=0: got {err}");
  let err = preprocess(&[], 640, 0, PATCH, CHANNELS, M).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "height=0: got {err}");
}

#[test]
fn rejects_zero_patch_size_and_budget() {
  let rgb = vec![0u8; 16 * 16 * 3];
  let err = preprocess(&rgb, 16, 16, 0, CHANNELS, M).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "patch_size=0: got {err}"
  );
  let err = preprocess(&rgb, 16, 16, PATCH, CHANNELS, 0).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "max_num_patches=0: got {err}"
  );
}

#[test]
fn rejects_wrong_rgb_length() {
  let rgb = vec![0u8; 100];
  let err = preprocess(&rgb, 16, 16, PATCH, CHANNELS, M).unwrap_err();
  match err {
    Error::LengthMismatch(p) => {
      assert_eq!(p.expected(), 16 * 16 * 3);
      assert_eq!(p.actual(), 100);
    }
    _ => panic!("expected LengthMismatch, got {err}"),
  }
}

#[test]
fn rejects_non_rgb_channels() {
  // The exported `preprocess` bypasses the config's `num_channels == 3` pin.
  // A direct 4-channel call must be rejected with a typed `InvariantViolation`
  // BEFORE any sizing — it would otherwise stride the always-3-channel resized
  // buffer with 4-channel offsets and read out of bounds. A 1-channel call is
  // rejected the same way (the path is strictly RGB, not merely positive).
  let rgb4 = vec![0u8; (16 * 16 * 4) as usize];
  let err = preprocess(&rgb4, 16, 16, PATCH, 4, M).unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "num_channels=4: got {err}"
  );
  let rgb1 = vec![0u8; (16 * 16) as usize];
  let err = preprocess(&rgb1, 16, 16, PATCH, 1, M).unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "num_channels=1: got {err}"
  );
}

#[test]
fn preprocess_pixel_values_size_overflow_is_typed_error() {
  // The soundness floor stays: when the `pixel_values` element count
  // `max_num_patches * (3 * patch_size^2)` would wrap `usize`, the
  // overflow-checked arithmetic surfaces a typed `ArithmeticOverflow` BEFORE
  // any allocation (a wrapped size would be UB). At `patch_size = u32::MAX`,
  // `patch_size^2` fits `usize` on a 64-bit target but the subsequent
  // `* channels` (3) overflows. The `rgb` slice is a cheap 16x16x3 — the
  // overflow check runs after the source-length check but before
  // `patch_grid`/resize, so no large buffer is ever sized. (A large but
  // non-overflowing product is NOT rejected for magnitude — `mlxrs` is a
  // library; it instead drives a fallible reservation.)
  let rgb = vec![0u8; (16 * 16 * 3) as usize];
  let err = preprocess(&rgb, 16, 16, u32::MAX, CHANNELS, M).unwrap_err();
  assert!(
    matches!(err, Error::ArithmeticOverflow(_)),
    "expected ArithmeticOverflow, got {err}"
  );
}

#[test]
fn rejects_oversize_source_dimension() {
  // A `width` far above the per-axis source cap `MAX_SOURCE_DIM` is rejected at
  // `Extent` construction with `CapExceeded` — before the byte-count product,
  // the slice-length check, or any allocation. (The old `width*height*3`
  // usize-overflow path is now unreachable: the per-axis cap bounds each
  // dimension well below the overflow boundary.)
  let err = preprocess(&[], u32::MAX, 1, PATCH, CHANNELS, M).unwrap_err();
  match err {
    Error::CapExceeded(p) => assert_eq!(p.observed(), u32::MAX as u64, "observed width"),
    _ => panic!("expected CapExceeded, got {err}"),
  }
}

#[test]
fn rejects_oversize_source_product() {
  // Both axes are within the per-axis cap (65536 == MAX_SOURCE_DIM), but their
  // byte-count product `65536 * 65536 * 3` (~12.9G) blows past the source-bytes
  // cap `MAX_SOURCE_PIXELS`. The checked product (`elem_count`) rejects it with
  // `CapExceeded` before the slice-length check or any clone — the product is
  // the magnitude the per-axis caps do not bound.
  let err = preprocess(&[], 1 << 16, 1 << 16, PATCH, CHANNELS, M).unwrap_err();
  match err {
    Error::CapExceeded(_) => {}
    _ => panic!("expected CapExceeded, got {err}"),
  }
}
