//! Oracle + structural tests for the LFM2.5-VL native-resolution processor.
//!
//! Deterministic, tiny-fixture, non-gated. The smart-resize tests pin the
//! faithful native patch grid for several aspect ratios (including an odd one)
//! and the within-budget invariant; the normalize/patchify tests pin a hand-
//! computed numeric oracle on a tiny image and the zero-padding past the active
//! rows; the token-expansion tests pin the `(ceil(rows/f) * ceil(cols/f))`
//! per-image count, the multi-image packing, and the bracket / mismatch paths.

use super::*;
use crate::vlm::image::patch_grid;

/// `((height, width), (H_p, W_p))` — a `patch_grid` reference case.
type GridCase = ((u32, u32), (u32, u32));

/// Evaluate `a` to a host `Vec<f32>`.
fn to_vec_f32(a: &Array) -> Vec<f32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

/// Evaluate `a` to a host `Vec<i32>`.
fn to_vec_i32(a: &Array) -> Vec<i32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<i32>().unwrap()
}

// ───────────────────────── smart-resize (native grid) ─────────────────────────

#[test]
fn smart_resize_picks_faithful_native_grid() {
  // The shared SigLIP2 NaFlex oracle formula (P=16, the LFM2.5-VL patch size).
  // The processor maximizes the patch budget while preserving aspect ratio, so
  // a small image is upscaled to fill `max_num_patches` and a large one is
  // downscaled to fit — these are the upstream-exact grids the LFM2.5-VL slow
  // processor produces. max_num_patches = 1024 (the LFM2.5-VL budget).
  const P: u32 = 16;
  const M: u32 = 1024;
  let cases: &[GridCase] = &[
    // Small square upscaled to fill the budget (32*32 = 1024).
    ((16, 16), (32, 32)),
    ((224, 224), (32, 32)),
    // Landscape HD downscaled, aspect preserved (24*42 = 1008 <= 1024).
    ((1080, 1920), (24, 42)),
    ((1920, 1080), (42, 24)),
    // Odd aspect ratio: a very wide 3x39 image -> (9, 113) (1017 <= 1024).
    ((3, 39), (9, 113)),
    // A single-column tall image fills the column budget (256*4 = 1024).
    ((1024, 16), (256, 4)),
  ];
  for ((h, w), expected) in cases {
    let got = patch_grid(*h, *w, P, M);
    assert_eq!(
      got, *expected,
      "patch_grid(h={h}, w={w}) — expected {expected:?}, got {got:?}"
    );
    assert!(
      u64::from(got.0) * u64::from(got.1) <= u64::from(M),
      "grid {got:?} exceeds budget {M}"
    );
  }
}

#[test]
fn smart_resize_preserves_aspect_orientation() {
  // A landscape and its transpose must produce transposed grids (aspect is
  // preserved, not collapsed to square).
  const P: u32 = 16;
  const M: u32 = 1024;
  let (h1, w1) = patch_grid(600, 900, P, M);
  let (h2, w2) = patch_grid(900, 600, P, M);
  assert_eq!((h1, w1), (w2, h2), "transpose must swap the grid axes");
}

// ───────────────────────── preprocess_image (patchify) ─────────────────────────

/// A tiny processor config: patch_size = 2, num_channels = 3, downsample = 2,
/// budget = 16 patches. SigLIP mean/std 0.5 (the default).
fn tiny_cfg(max_num_patches: u32) -> Lfm2VlProcessorConfig {
  // image_token = 396, downsample_factor = 2, patch_size = 2.
  Lfm2VlProcessorConfig::new(396, 2, 2, max_num_patches).unwrap()
}

#[test]
fn preprocess_shapes_and_full_mask() {
  // An 8x8 RGB image, patch_size 2, budget 16 -> a 4x4 patch grid (16 active
  // patches, exactly filling the budget). pixel_values (16, 2*2*3=12),
  // mask (16,) all-1, spatial (2,).
  let cfg = tiny_cfg(16);
  let (w, h) = (8u32, 8u32);
  let rgb: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect();
  let out = preprocess_image(&rgb, w, h, &cfg).unwrap();

  assert_eq!(out.pixel_values.shape(), vec![16, 12]);
  assert_eq!(out.pixel_attention_mask.shape(), vec![16]);
  assert_eq!(out.spatial_shapes.shape(), vec![2]);
  // 8x8 at patch 2, budget 16 -> (4, 4) (fills the budget exactly).
  assert_eq!(out.grid().unwrap(), (4, 4));
  assert_eq!(to_vec_i32(&out.spatial_shapes), vec![4, 4]);
  // All 16 patch rows are active (no padding when the grid fills the budget).
  let mask = to_vec_i32(&out.pixel_attention_mask);
  assert!(
    mask.iter().all(|&m| m == 1),
    "all rows active when grid fills"
  );
}

#[test]
fn preprocess_partial_grid_pads_mask() {
  // A tall 4x2 image, patch_size 2, budget 4 -> grid (2, 1) (2 active < 4
  // budget), so the trailing 2 patch rows are padding (mask 0).
  let cfg = tiny_cfg(4);
  let (w, h) = (2u32, 4u32);
  let rgb: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect();
  let out = preprocess_image(&rgb, w, h, &cfg).unwrap();
  assert_eq!(out.grid().unwrap(), (2, 1));
  assert_eq!(to_vec_i32(&out.spatial_shapes), vec![2, 1]);
  let mask = to_vec_i32(&out.pixel_attention_mask);
  assert_eq!(mask, vec![1, 1, 0, 0], "2 active rows, 2 padding rows");
}

#[test]
fn preprocess_normalization_is_applied_and_padding_is_zero() {
  // A solid mid-gray (128) tall 4x2 image, patch_size 2, budget 4 -> grid
  // (2, 1): the resize 4x2 -> (4, 2) is identity (same size), so a solid input
  // stays solid. x/127.5 - 1.0 ≈ 0.00392 for every active channel; the 2
  // padding rows are exactly 0.
  let cfg = tiny_cfg(4);
  let (w, h) = (2u32, 4u32);
  let rgb = vec![128u8; (w * h * 3) as usize];
  let out = preprocess_image(&rgb, w, h, &cfg).unwrap();
  assert_eq!(out.grid().unwrap(), (2, 1));
  let pv = to_vec_f32(&out.pixel_values);
  let expected = 128.0f32 / 127.5 - 1.0; // SigLIP (x/255 - 0.5)/0.5
  // The 2 active patch rows (2 * 12 = 24 floats) carry the normalized value.
  for (i, &v) in pv.iter().take(24).enumerate() {
    assert!(
      (v - expected).abs() < 1e-5,
      "active float {i}: got {v}, want {expected}"
    );
  }
  // The remaining (4-2)*12 = 24 floats are zero padding.
  assert!(
    pv[24..].iter().all(|&v| v == 0.0),
    "padding floats must be exactly 0"
  );
}

#[test]
fn preprocess_numeric_oracle_tiny_image() {
  // A 2x2 RGB image (one patch of patch_size 2). With patch_size 2, budget 1,
  // the native grid is (1, 1) and the single patch is the resized image
  // unchanged (identity resize 2x2 -> 2x2). The flattened patch row is the
  // row-major (row, col, channel) scan, each byte through x/127.5 - 1.0.
  let cfg = tiny_cfg(1);
  let (w, h) = (2u32, 2u32);
  // Distinct per-channel values so the flatten order is observable.
  // Pixels (row-major): (0,0)=[0,10,20] (0,1)=[30,40,50]
  //                     (1,0)=[60,70,80] (1,1)=[90,100,110]
  let rgb: Vec<u8> = vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
  let out = preprocess_image(&rgb, w, h, &cfg).unwrap();
  assert_eq!(out.grid().unwrap(), (1, 1));
  let pv = to_vec_f32(&out.pixel_values);
  // The single active patch row is the 12-value flatten of the 2x2 patch in
  // (row, col, channel-innermost) order — exactly the input byte order here.
  let norm = |x: u8| x as f32 / 127.5 - 1.0;
  let want: Vec<f32> = rgb.iter().map(|&b| norm(b)).collect();
  for (i, (g, e)) in pv.iter().take(12).zip(want.iter()).enumerate() {
    assert!((g - e).abs() < 1e-5, "patch float {i}: got {g}, want {e}");
  }
}

#[test]
fn preprocess_custom_mean_std_path() {
  // A non-uniform per-channel mean/std exercises the scalar per-channel
  // normalize branch (the NEON kernel only applies for uniform scale/bias).
  let cfg = tiny_cfg(1)
    .with_image_mean([0.1, 0.2, 0.3])
    .with_image_std([0.5, 0.4, 0.25]);
  let (w, h) = (2u32, 2u32);
  let rgb: Vec<u8> = vec![255, 255, 255, 0, 0, 0, 128, 128, 128, 64, 64, 64];
  let out = preprocess_image(&rgb, w, h, &cfg).unwrap();
  let pv = to_vec_f32(&out.pixel_values);
  // Pixel (0,0) = white: ((255/255 - mean)/std) per channel.
  let want0 = [(1.0 - 0.1) / 0.5, (1.0 - 0.2) / 0.4, (1.0 - 0.3) / 0.25];
  for (i, (g, e)) in pv.iter().take(3).zip(want0.iter()).enumerate() {
    assert!((g - e).abs() < 1e-4, "white px chan {i}: got {g}, want {e}");
  }
}

#[test]
fn preprocess_rejects_wrong_rgb_length() {
  let cfg = tiny_cfg(16);
  // 4x4 needs 48 bytes; supply 47.
  let rgb = vec![0u8; 47];
  let err = preprocess_image(&rgb, 4, 4, &cfg).unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err}");
}

#[test]
fn preprocess_rejects_zero_dim() {
  let cfg = tiny_cfg(16);
  let err = preprocess_image(&[], 0, 4, &cfg).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

// ───────────────────────── token count / expansion ─────────────────────────

#[test]
fn token_count_even_and_odd_grids() {
  // Even: 4x4, factor 2 -> 2*2 = 4.
  assert_eq!(num_image_tokens_from_patch_grid(4, 4, 2).unwrap(), 4);
  // Odd rows: 5x4, factor 2 -> ceil(5/2)*ceil(4/2) = 3*2 = 6.
  assert_eq!(num_image_tokens_from_patch_grid(5, 4, 2).unwrap(), 6);
  // Odd cols: 4x7, factor 2 -> 2*ceil(7/2) = 2*4 = 8.
  assert_eq!(num_image_tokens_from_patch_grid(4, 7, 2).unwrap(), 8);
  // Both odd: 5x7, factor 2 -> 3*4 = 12.
  assert_eq!(num_image_tokens_from_patch_grid(5, 7, 2).unwrap(), 12);
  // Factor 1: identity -> rows*cols.
  assert_eq!(num_image_tokens_from_patch_grid(3, 5, 1).unwrap(), 15);
  // Factor 3: 7x7 -> ceil(7/3)*ceil(7/3) = 3*3 = 9.
  assert_eq!(num_image_tokens_from_patch_grid(7, 7, 3).unwrap(), 9);
}

#[test]
fn token_count_rejects_degenerate() {
  assert!(matches!(
    num_image_tokens_from_patch_grid(0, 4, 2).unwrap_err(),
    Error::OutOfRange(_)
  ));
  assert!(matches!(
    num_image_tokens_from_patch_grid(4, 4, 0).unwrap_err(),
    Error::OutOfRange(_)
  ));
}

#[test]
fn expand_single_image_no_brackets() {
  // input "a <image> b" with one image, grid (4, 4), factor 2 -> 4 image
  // tokens replace the single placeholder. Ids: a=1, <image>=396, b=2.
  // `use_image_special_tokens` defaults to `true` (upstream), but no bracket id
  // is supplied, so the emit gate (`&& image_start_token.is_some()`) is off and
  // no bracket appears — the run is the bare image tokens.
  let cfg = tiny_cfg(16);
  assert!(
    cfg.use_image_special_tokens(),
    "use_image_special_tokens defaults to true (upstream parity)"
  );
  let ids = [1, 396, 2];
  let grids = [(4, 4)];
  let out = expand_image_tokens(&ids, &grids, &cfg).unwrap();
  assert_eq!(out, vec![1, 396, 396, 396, 396, 2]);
}

#[test]
fn expand_single_image_with_brackets() {
  // Bracketed: <start> <image>*4 <end> around the run.
  let cfg = tiny_cfg(16).with_special_tokens(Some(100), Some(101));
  let ids = [1, 396, 2];
  let grids = [(4, 4)];
  let out = expand_image_tokens(&ids, &grids, &cfg).unwrap();
  assert_eq!(out, vec![1, 100, 396, 396, 396, 396, 101, 2]);
}

#[test]
fn expand_multi_image_packs_in_order() {
  // Two images: first grid (4,4) -> 4 tokens, second grid (5,7) -> 12 tokens.
  // Placeholders at two positions; each expands to its own count, in order.
  let cfg = tiny_cfg(16);
  let ids = [396, 9, 396];
  let grids = [(4, 4), (5, 7)];
  let out = expand_image_tokens(&ids, &grids, &cfg).unwrap();
  // 4 image tokens, then 9, then 12 image tokens.
  let mut want = vec![396; 4];
  want.push(9);
  want.extend(std::iter::repeat_n(396, 12));
  assert_eq!(out, want);
}

#[test]
fn expand_count_mismatch_is_typed_error() {
  // Two placeholders but only one grid -> mismatch (the reference's
  // n_images_in_text != n_images_in_images guard).
  let cfg = tiny_cfg(16);
  let ids = [396, 396];
  let grids = [(4, 4)];
  let err = expand_image_tokens(&ids, &grids, &cfg).unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err}");
}

#[test]
fn expand_no_images_passes_through() {
  // No placeholders, no grids -> the sequence is unchanged.
  let cfg = tiny_cfg(16);
  let ids = [1, 2, 3];
  let out = expand_image_tokens(&ids, &[], &cfg).unwrap();
  assert_eq!(out, vec![1, 2, 3]);
}

// ───────────────────────── config validation ─────────────────────────

#[test]
fn config_rejects_zero_patch_and_negative_token() {
  assert!(matches!(
    Lfm2VlProcessorConfig::new(396, 2, 0, 16).unwrap_err(),
    Error::OutOfRange(_)
  ));
  assert!(matches!(
    Lfm2VlProcessorConfig::new(-1, 2, 2, 16).unwrap_err(),
    Error::OutOfRange(_)
  ));
}
