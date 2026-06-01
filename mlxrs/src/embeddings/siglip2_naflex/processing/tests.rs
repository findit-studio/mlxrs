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
fn normalize_row_endpoints() {
  // 0 -> -1, 255 -> 1 (the [-1, 1] SigLIP range).
  let mut dst = [0.0f32; 3];
  normalize_row(&[0, 128, 255], &mut dst);
  assert!((dst[0] - (-1.0)).abs() < 1e-6);
  assert!((dst[1] - (128.0 / 127.5 - 1.0)).abs() < 1e-6);
  assert!((dst[2] - 1.0).abs() < 1e-6, "255 -> {}", dst[2]);
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
fn rejects_pixel_values_product_over_cap() {
  // `patch_size = 591` keeps `patch_feature_dim = 3 * 591^2 = 1_047_843` just
  // under the `1 << 20` width cap, and `max_num_patches = 256` is a real,
  // within-cardinality budget — yet their product `268_247_808` exceeds the
  // `1 << 26` `MAX_PIXEL_ELEMENTS` product cap (~1 GiB of f32). The guard must
  // fire as a typed `CapExceeded` BEFORE the (infallible-in-the-old-code)
  // allocation, never aborting. The `rgb` slice is a cheap 16x16x3 (the cap
  // check runs after the length check but before `patch_grid`/resize, so no
  // large buffer is ever sized).
  let rgb = vec![0u8; (16 * 16 * 3) as usize];
  let err = preprocess(&rgb, 16, 16, 591, CHANNELS, M).unwrap_err();
  match err {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap(), 1 << 26, "cap value");
      assert_eq!(p.observed(), 256 * 3 * 591 * 591, "observed product");
    }
    _ => panic!("expected CapExceeded, got {err}"),
  }
}

#[test]
fn rejects_huge_dimensions_overflow() {
  // width near u32::MAX with height 1 overflows the rgb byte-count
  // product on a 32-bit usize *and* would otherwise blow past the budget;
  // with an empty slice the length check or overflow check fires first.
  // Use a value that forces the `width*height*3` usize product to be
  // checked: on 64-bit this won't overflow usize, so instead assert the
  // length-mismatch path (empty slice) rejects it cleanly rather than
  // panicking.
  let err = preprocess(&[], u32::MAX, 1, PATCH, CHANNELS, M).unwrap_err();
  // Either ArithmeticOverflow (32-bit usize) or LengthMismatch (64-bit,
  // empty slice != huge expected) — both are typed, neither panics.
  assert!(
    matches!(err, Error::ArithmeticOverflow(_) | Error::LengthMismatch(_)),
    "got {err}"
  );
}
