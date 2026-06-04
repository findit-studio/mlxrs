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
  // Bracketed: <start> <image>*4 <end> around the run. The flag governs emission
  // (`tiny_cfg` defaults it to `true`, upstream parity); `with_special_tokens`
  // only supplies the ids the enabled gate uses.
  let cfg = tiny_cfg(16).with_special_tokens(Some(100), Some(101));
  assert!(
    cfg.use_image_special_tokens(),
    "the default flag drives bracket emission; with_special_tokens only sets the ids"
  );
  let ids = [1, 396, 2];
  let grids = [(4, 4)];
  let out = expand_image_tokens(&ids, &grids, &cfg).unwrap();
  assert_eq!(out, vec![1, 100, 396, 396, 396, 396, 101, 2]);
}

#[test]
fn with_special_tokens_does_not_re_enable_disabled_flag() {
  // Regression: a checkpoint that disabled image special tokens
  // (`use_image_special_tokens = false`) must keep emitting NO brackets when the
  // bracket ids are supplied afterwards. The natural caller order is
  // `processor_config()?.with_special_tokens(start, end)` — `processor_config`
  // threads the checkpoint's `false`, and supplying the ids must not flip it back
  // on (`processing_lfm2_vl.py:388-400`: the flag, not the id presence, gates the
  // brackets). Identity (ids) is decoupled from policy (the flag).
  let ids = [1, 396, 2];
  let grids = [(4, 4)];

  // Flag off (as a `false` checkpoint would set via `with_use_image_special_
  // tokens`), THEN supply the ids — the order callers reach for.
  let off_then_ids = tiny_cfg(16)
    .with_use_image_special_tokens(false)
    .with_special_tokens(Some(100), Some(101));
  assert!(
    !off_then_ids.use_image_special_tokens(),
    "with_special_tokens must NOT re-enable the disabled flag"
  );
  let out = expand_image_tokens(&ids, &grids, &off_then_ids).unwrap();
  assert_eq!(
    out,
    vec![1, 396, 396, 396, 396, 2],
    "no brackets emitted when the flag is off, even with both bracket ids set"
  );

  // The `<image>` placeholder count matches the no-special-tokens baseline
  // (`expand_single_image_no_brackets`): only the would-be brackets differ, so
  // the image-feature / token counts stay aligned with the reference.
  let baseline = expand_image_tokens(&ids, &grids, &tiny_cfg(16)).unwrap();
  let count_images = |v: &[i32]| v.iter().filter(|&&t| t == 396).count();
  assert_eq!(count_images(&out), count_images(&baseline));
  assert_eq!(
    out, baseline,
    "flag-off output equals the no-bracket-ids baseline"
  );
}

#[test]
fn expand_honors_use_image_special_tokens_flag() {
  // `use_image_special_tokens = false` suppresses the brackets even when both
  // ids are present (`processing_lfm2_vl.py:388-400`'s `if use_image_special_
  // tokens:` gate). The `<image>` run is identical to the bracketed case; only
  // the start/end ids are dropped — so the image-feature / token count is the
  // same 4 placeholders either way, no desync. The flag and the ids are
  // independent: emission tracks the flag regardless of when the ids were set.
  let ids = [1, 396, 2];
  let grids = [(4, 4)];

  // Flag on (the `tiny_cfg` default) with both ids supplied -> brackets emitted.
  let on = tiny_cfg(16).with_special_tokens(Some(100), Some(101));
  assert!(on.use_image_special_tokens());
  let out_on = expand_image_tokens(&ids, &grids, &on).unwrap();
  assert_eq!(out_on, vec![1, 100, 396, 396, 396, 396, 101, 2]);

  // Same ids, flag explicitly off -> no brackets. Setting the flag off does not
  // depend on ordering relative to `with_special_tokens` (see
  // `with_special_tokens_does_not_re_enable_disabled_flag` for the reverse
  // order); the flag alone governs the gate.
  let off = tiny_cfg(16)
    .with_special_tokens(Some(100), Some(101))
    .with_use_image_special_tokens(false);
  assert!(!off.use_image_special_tokens());
  let out_off = expand_image_tokens(&ids, &grids, &off).unwrap();
  assert_eq!(
    out_off,
    vec![1, 396, 396, 396, 396, 2],
    "no brackets when flag off"
  );

  // The number of `<image>` placeholder tokens is identical with the flag on or
  // off (only the 2 bracket ids differ) — the count stays consistent.
  let count_images = |v: &[i32]| v.iter().filter(|&&t| t == 396).count();
  assert_eq!(count_images(&out_on), count_images(&out_off));
  assert_eq!(
    out_on.len(),
    out_off.len() + 2,
    "exactly the 2 brackets differ"
  );
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

// ───────────────────────── image splitting / tiling ─────────────────────────
//
// Oracle values are computed independently from the HF
// `image_processing_lfm2_vl.py` formulas (NOT from the code under test), with
// the upstream default knobs: downsample_factor = 2, encoder_patch_size = 16,
// tile_size = 512, min_image_tokens = 64, max_image_tokens = 256,
// max_pixels_tolerance = 2.0, min_tiles = 2, max_tiles = 10.

/// A tiling config with the HF default knobs (downsample 2, patch 16, tile 512),
/// `max_num_patches = 1024` (the LFM2.5-VL budget; >= tile_size/P squared = 1024
/// AND >= 256*4). `use_thumbnail` is the argument; splitting enabled.
fn tiling_cfg(use_thumbnail: bool) -> Lfm2VlProcessorConfig {
  Lfm2VlProcessorConfig::new(396, 2, 16, 1024)
    .unwrap()
    .with_tiling(true, 2, 10, use_thumbnail, 64, 256, 16, 512, 2.0)
    .unwrap()
}

#[test]
fn with_tiling_rejects_invalid_params() {
  let base = Lfm2VlProcessorConfig::new(396, 2, 16, 1024).unwrap();
  // tile_size not divisible by encoder_patch_size (500 % 16 != 0).
  assert!(matches!(
    base
      .with_tiling(true, 2, 10, false, 64, 256, 16, 500, 2.0)
      .unwrap_err(),
    Error::DivisibilityConstraint(_)
  ));
  // min_tiles > max_tiles.
  assert!(matches!(
    base
      .with_tiling(true, 6, 4, false, 64, 256, 16, 512, 2.0)
      .unwrap_err(),
    Error::OutOfRange(_)
  ));
  // min_image_tokens > max_image_tokens.
  assert!(matches!(
    base
      .with_tiling(true, 2, 10, false, 300, 256, 16, 512, 2.0)
      .unwrap_err(),
    Error::OutOfRange(_)
  ));
  // zero tile_size.
  assert!(matches!(
    base
      .with_tiling(true, 2, 10, false, 64, 256, 16, 0, 2.0)
      .unwrap_err(),
    Error::OutOfRange(_)
  ));
}

#[test]
fn round_by_factor_ties_to_even_like_python() {
  // HF `round_by_factor` (`image_processing_lfm2_vl.py:40-42`) is
  // `round(number / factor) * factor`, and Python's `round` ties to EVEN. On an
  // exact half-tie (`number % factor == factor / 2`) the quotient rounds to its
  // nearest even neighbour — NOT away from zero. These pin the exact-tie cases
  // that round-half-away-from-zero (Rust `f64::round`) would get wrong:
  //   208/32 = 6.5 -> 6 (even)  -> 192   (away-from-zero would give 224)
  //   240/32 = 7.5 -> 8 (even)  -> 256
  //    48/32 = 1.5 -> 2 (even)  -> 64
  //    16/32 = 0.5 -> 0 (even)  -> 0
  assert_eq!(
    round_by_factor(208, 32).unwrap(),
    192,
    "6.5 ties down to 6 (even)"
  );
  assert_eq!(
    round_by_factor(240, 32).unwrap(),
    256,
    "7.5 ties up to 8 (even)"
  );
  assert_eq!(
    round_by_factor(48, 32).unwrap(),
    64,
    "1.5 ties up to 2 (even)"
  );
  assert_eq!(
    round_by_factor(16, 32).unwrap(),
    0,
    "0.5 ties down to 0 (even)"
  );
  // The reported divergence: 208@32 is 192, never the away-from-zero 224.
  assert_ne!(round_by_factor(208, 32).unwrap(), 224);
}

#[test]
fn round_by_factor_non_tie_rounds_to_nearest() {
  // Non-half quotients round to the nearest multiple (ties-to-even is moot).
  //   200/32 = 6.25  -> 6 -> 192
  //   220/32 = 6.875 -> 7 -> 224
  //   100/16 = 6.25  -> 6 -> 96
  //   110/16 = 6.875 -> 7 -> 112
  assert_eq!(round_by_factor(200, 32).unwrap(), 192);
  assert_eq!(round_by_factor(220, 32).unwrap(), 224);
  assert_eq!(round_by_factor(100, 16).unwrap(), 96);
  assert_eq!(round_by_factor(110, 16).unwrap(), 112);
  // An exact multiple is unchanged, and any factor of 1 is the identity.
  assert_eq!(round_by_factor(192, 32).unwrap(), 192);
  assert_eq!(round_by_factor(207, 1).unwrap(), 207);
}

#[test]
fn round_by_factor_rejects_zero_factor() {
  assert!(matches!(
    round_by_factor(208, 0).unwrap_err(),
    Error::OutOfRange(_)
  ));
}

#[test]
fn plan_tiles_small_image_is_single_no_split() {
  // 256x256 (h=w=256): h_bar=w_bar=256, 256*256=65536 <= max_pixels
  // (256*16^2*2^2*2.0 = 524288) so NOT too large -> single sub-image, smart
  // size 256x256, NO thumbnail.
  let cfg = tiling_cfg(true);
  let plan = plan_tiles(256, 256, &cfg).unwrap();
  assert!(!plan.is_split(), "256x256 must not split");
  assert_eq!(plan.grid(), (1, 1));
  assert!(!plan.has_thumbnail());
  assert_eq!(plan.sub_image_count().unwrap(), 1);
}

#[test]
fn plan_tiles_tiny_image_upscales_single() {
  // 100x100: below min_pixels -> smart_resize upscales to 256x256 (patch grid
  // 16x16), still a single sub-image (not too large).
  let cfg = tiling_cfg(true);
  let plan = plan_tiles(100, 100, &cfg).unwrap();
  assert!(!plan.is_split());
  assert_eq!(plan.sub_image_count().unwrap(), 1);
}

#[test]
fn plan_tiles_large_square_splits_3x3_with_thumbnail() {
  // 2048x2048 (h=w=2048): too large -> grid 3x3 (target 1536x1536), thumbnail
  // smart-resized to 512x512. 9 tiles + 1 thumbnail = 10 sub-images.
  let cfg = tiling_cfg(true);
  let plan = plan_tiles(2048, 2048, &cfg).unwrap();
  assert!(plan.is_split());
  assert_eq!(plan.grid(), (3, 3), "2048x2048 -> 3x3 grid");
  assert!(plan.has_thumbnail());
  assert_eq!(plan.sub_image_count().unwrap(), 10);
}

#[test]
fn plan_tiles_large_portrait_splits_with_oracle_grid() {
  // h=2000 w=1500 -> aspect 0.75 -> grid (gw=2, gh=3), target (1024, 1536),
  // thumbnail 416x576. 6 tiles + thumbnail = 7.
  let cfg = tiling_cfg(true);
  let plan = plan_tiles(2000, 1500, &cfg).unwrap();
  assert!(plan.is_split());
  assert_eq!(plan.grid(), (2, 3), "portrait grid (gw=2, gh=3)");
  assert_eq!(plan.sub_image_count().unwrap(), 7);
}

#[test]
fn plan_tiles_thumbnail_off_omits_thumbnail() {
  // Same large square, use_thumbnail=false -> 9 tiles, no thumbnail.
  let cfg = tiling_cfg(false);
  let plan = plan_tiles(2048, 2048, &cfg).unwrap();
  assert!(plan.is_split());
  assert_eq!(plan.grid(), (3, 3));
  assert!(!plan.has_thumbnail());
  assert_eq!(plan.sub_image_count().unwrap(), 9);
}

#[test]
fn plan_tiles_splitting_disabled_never_splits() {
  // do_image_splitting=false -> a large image stays a single (smart-resized)
  // sub-image regardless of size.
  let cfg = Lfm2VlProcessorConfig::new(396, 2, 16, 1024)
    .unwrap()
    .with_tiling(false, 2, 10, true, 64, 256, 16, 512, 2.0)
    .unwrap();
  let plan = plan_tiles(2048, 2048, &cfg).unwrap();
  assert!(!plan.is_split(), "splitting disabled -> single sub-image");
  assert_eq!(plan.sub_image_count().unwrap(), 1);
}

#[test]
fn tile_image_small_produces_single_native_subimage() {
  // A 256x256 image (not too large) -> ONE Lfm2VlImageInputs at the smart-resize
  // grid 16x16 (256/16). pixel_values (1024, 16^2*3=768), all-active mask for the
  // 256 active rows.
  let cfg = tiling_cfg(true);
  let (w, h) = (256u32, 256u32);
  let rgb: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect();
  let tiles = tile_image(&rgb, w, h, &cfg).unwrap();
  assert_eq!(tiles.len(), 1, "small image -> single sub-image");
  assert_eq!(
    tiles[0].grid().unwrap(),
    (16, 16),
    "256/16 = 16 patches/side"
  );
  assert_eq!(tiles[0].pixel_values.shape(), vec![1024, 768]);
  // 16*16 = 256 active patch rows, the rest padding.
  let mask = to_vec_i32(&tiles[0].pixel_attention_mask);
  assert_eq!(mask.iter().filter(|&&m| m == 1).count(), 256);
}

#[test]
fn tile_image_large_produces_tiles_plus_thumbnail() {
  // A 1024x1024 image. h_bar=w_bar=1024, 1024*1024=1048576 > max_pixels
  // (524288) -> too large -> splits. Aspect 1.0 -> grid 2x2 (target 1024),
  // thumbnail smart-resized to 512x512. 4 tiles + 1 thumbnail = 5 sub-images.
  // Each TILE is 512x512 -> grid 32x32 (512/16) = 1024 patches (fills the budget
  // exactly).
  let cfg = tiling_cfg(true);
  let (w, h) = (1024u32, 1024u32);
  let rgb: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 251) as u8).collect();
  let tiles = tile_image(&rgb, w, h, &cfg).unwrap();
  assert_eq!(tiles.len(), 5, "4 tiles + 1 thumbnail");
  // The 4 tiles each have a 32x32 patch grid (tile_size 512 / patch 16).
  for t in tiles.iter().take(4) {
    assert_eq!(
      t.grid().unwrap(),
      (32, 32),
      "each 512x512 tile -> 32x32 patches"
    );
    assert_eq!(t.pixel_values.shape(), vec![1024, 768]);
    // 32*32 = 1024 active rows (fills the budget exactly, no padding).
    let mask = to_vec_i32(&t.pixel_attention_mask);
    assert!(mask.iter().all(|&m| m == 1), "tile fills the budget");
  }
  // The thumbnail (last) is the whole image smart-resized to 512x512 -> 32x32.
  let thumb = tiles.last().unwrap();
  assert_eq!(thumb.grid().unwrap(), (32, 32));
}

#[test]
fn tile_image_token_count_sums_over_tiles() {
  // The whole-image token count under tiling = sum over sub-images of
  // num_image_tokens_from_patch_grid(rows, cols, factor). For the 1024x1024
  // case: 5 sub-images each 32x32 -> ceil(32/2)*ceil(32/2) = 16*16 = 256 each
  // -> 5*256 = 1280 image tokens.
  let cfg = tiling_cfg(true);
  let (w, h) = (1024u32, 1024u32);
  let rgb: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 251) as u8).collect();
  let tiles = tile_image(&rgb, w, h, &cfg).unwrap();
  let mut total = 0i32;
  let mut grids: Vec<(i32, i32)> = Vec::new();
  for t in &tiles {
    let (r, c) = t.grid().unwrap();
    total += num_image_tokens_from_patch_grid(r, c, 2).unwrap();
    grids.push((r, c));
  }
  assert_eq!(total, 1280, "5 sub-images * 256 tokens each");
  // The flattened sub-image grids drive `expand_image_tokens` 1:1 with the
  // flattened sub-images (each tile / thumbnail is one NaFlex sub-image, so the
  // prompt carries one `<image>` placeholder per sub-image — the same flat
  // alignment `get_input_embeddings` consumes). Five placeholders, five grids;
  // each expands to its own 256-token run.
  let ids: Vec<i32> = std::iter::once(1)
    .chain(std::iter::repeat_n(396, grids.len()))
    .chain(std::iter::once(2))
    .collect();
  let expanded = expand_image_tokens(&ids, &grids, &cfg).unwrap();
  // 1 + (5*256 image tokens) + 2 (the surrounding ids).
  assert_eq!(expanded.len(), 2 + 1280);
  assert_eq!(expanded[0], 1);
  assert_eq!(*expanded.last().unwrap(), 2);
}

#[test]
fn tile_image_rejects_wrong_rgb_length() {
  let cfg = tiling_cfg(true);
  // 256x256 needs 196608 bytes; supply one short.
  let rgb = vec![0u8; (256 * 256 * 3) - 1];
  let err = tile_image(&rgb, 256, 256, &cfg).unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err}");
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
