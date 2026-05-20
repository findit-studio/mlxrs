//! M4 VLM image-preprocessing primitives tests.
//!
//! Reference basis:
//! - swift `MediaProcessing.swift` (lines 81-193 — `resampleBicubic`,
//!   `resampleLanczos`, `normalize`, `asMLXArray`).
//! - python `mlx-vlm/utils.py` (`load_image`, `resize_image`,
//!   `process_image`).
//!
//! Tests are pure synthetic (no disk I/O beyond the controlled tempfile
//! round-trip in [`load_image_decodes_png_round_trip`]); the underlying
//! `image` crate handles encode/decode.

#![cfg(feature = "vlm")]

use mlxrs::{
  Array, Dtype, Error,
  vlm::image::{
    ColorOrder, ImageProcessorConfig, ResizeFilter, image_to_array, load_image, normalize_imagenet,
    patchify, preprocess, rescale, resize,
  },
};

const TOL: f32 = 1e-5;

fn close(a: f32, b: f32) -> bool {
  (a - b).abs() <= TOL
}

fn vclose(a: &[f32], b: &[f32]) -> bool {
  a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

/// Synthetic 4x4 RGB image: each pixel (x, y) gets (10*y, 10*x, 100).
fn synthetic_image(width: u32, height: u32) -> ::image::DynamicImage {
  let mut buf = ::image::RgbImage::new(width, height);
  for y in 0..height {
    for x in 0..width {
      buf.put_pixel(
        x,
        y,
        ::image::Rgb([((y * 10) % 256) as u8, ((x * 10) % 256) as u8, 100]),
      );
    }
  }
  ::image::DynamicImage::ImageRgb8(buf)
}

/// Synthetic gradient: pixel (x, y) gets R=x*8, G=y*8, B=128 (8-bit clamp).
fn gradient_image(width: u32, height: u32) -> ::image::DynamicImage {
  let mut buf = ::image::RgbImage::new(width, height);
  for y in 0..height {
    for x in 0..width {
      buf.put_pixel(
        x,
        y,
        ::image::Rgb([((x * 8) % 256) as u8, ((y * 8) % 256) as u8, 128]),
      );
    }
  }
  ::image::DynamicImage::ImageRgb8(buf)
}

// ---------- resize ----------

#[test]
fn resize_changes_shape_preserves_dtype() {
  let img = synthetic_image(8, 6);
  let out = resize(&img, (16, 32), ResizeFilter::Bicubic);
  // image::imageops::resize(image, nwidth, nheight, ...) so target.1=w, target.0=h.
  assert_eq!(out.width(), 32, "width = target.1");
  assert_eq!(out.height(), 16, "height = target.0");
}

#[test]
fn resize_filters_all_succeed() {
  let img = synthetic_image(8, 8);
  for f in [
    ResizeFilter::Nearest,
    ResizeFilter::Bilinear,
    ResizeFilter::Bicubic,
    ResizeFilter::Lanczos3,
  ] {
    let out = resize(&img, (4, 4), f);
    assert_eq!((out.width(), out.height()), (4, 4), "filter {:?}", f);
  }
}

// ---------- image_to_array ----------

#[test]
fn image_to_array_shape_dtype_range() {
  let img = synthetic_image(4, 3);
  let mut arr = image_to_array(&img, ColorOrder::Rgb).unwrap();
  assert_eq!(arr.shape(), vec![3, 4, 3], "shape [H, W, 3]");
  assert_eq!(arr.dtype().unwrap(), Dtype::F32);
  // Values must be in [0, 255] BEFORE rescale (per spec).
  let v: Vec<f32> = arr.to_vec().unwrap();
  assert!(v.iter().all(|&x| (0.0..=255.0).contains(&x)));
}

#[test]
fn image_to_array_rgb_vs_bgr_swap() {
  // 2x1 image with two distinct pixels so R/B swap is observable.
  let mut buf = ::image::RgbImage::new(2, 1);
  buf.put_pixel(0, 0, ::image::Rgb([10, 20, 30]));
  buf.put_pixel(1, 0, ::image::Rgb([40, 50, 60]));
  let img = ::image::DynamicImage::ImageRgb8(buf);

  let mut rgb = image_to_array(&img, ColorOrder::Rgb).unwrap();
  let mut bgr = image_to_array(&img, ColorOrder::Bgr).unwrap();
  let rgb_v: Vec<f32> = rgb.to_vec().unwrap();
  let bgr_v: Vec<f32> = bgr.to_vec().unwrap();
  // Channel-last [1, 2, 3]: first pixel → [10, 20, 30] RGB / [30, 20, 10] BGR.
  assert!(vclose(&rgb_v, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]));
  assert!(vclose(&bgr_v, &[30.0, 20.0, 10.0, 60.0, 50.0, 40.0]));
}

#[test]
fn image_to_array_drops_alpha_from_rgba() {
  // 1x1 RGBA pixel with non-trivial alpha; alpha must be dropped (per
  // swift `MediaProcessing.swift:187` `array[..., :3]`).
  let mut buf = ::image::RgbaImage::new(1, 1);
  buf.put_pixel(0, 0, ::image::Rgba([11, 22, 33, 44]));
  let img = ::image::DynamicImage::ImageRgba8(buf);
  let mut arr = image_to_array(&img, ColorOrder::Rgb).unwrap();
  assert_eq!(arr.shape(), vec![1, 1, 3], "alpha channel dropped");
  let v: Vec<f32> = arr.to_vec().unwrap();
  assert!(vclose(&v, &[11.0, 22.0, 33.0]));
}

// ---------- rescale ----------

#[test]
fn rescale_1_over_255_maps_uchar_to_unit_interval() {
  let img = synthetic_image(4, 4);
  let arr = image_to_array(&img, ColorOrder::Rgb).unwrap();
  let mut scaled = rescale(&arr, 1.0 / 255.0).unwrap();
  let v: Vec<f32> = scaled.to_vec().unwrap();
  // u8 [0, 255] → f32 [0, 1] is bounded by [0, 1] inclusive.
  assert!(
    v.iter().all(|&x| (0.0..=1.0).contains(&x)),
    "rescaled values out of [0, 1]: min={:?} max={:?}",
    v.iter().cloned().fold(f32::INFINITY, f32::min),
    v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
  );
}

#[test]
fn rescale_preserves_dtype() {
  let arr = Array::from_slice(&[100.0_f32, 200.0], &(2usize,)).unwrap();
  let mut scaled = rescale(&arr, 0.5).unwrap();
  assert_eq!(scaled.dtype().unwrap(), Dtype::F32);
  let v: Vec<f32> = scaled.to_vec().unwrap();
  assert!(vclose(&v, &[50.0, 100.0]));
}

#[test]
fn rescale_rejects_integer_dtypes() {
  // U8 [0, 255] with `1/255` scale would silently floor to 0 in the
  // input dtype; we surface that as a clean ShapeMismatch instead.
  let arr = Array::from_slice(&[0_u8, 128, 255], &(3usize,)).unwrap();
  let err = rescale(&arr, 1.0 / 255.0).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "want ShapeMismatch, got {err:?}",
  );
  // I32 input too — every integer dtype is rejected.
  let arr_i = Array::from_slice(&[0_i32, 1, 2], &(3usize,)).unwrap();
  let err_i = rescale(&arr_i, 0.5).unwrap_err();
  assert!(matches!(err_i, Error::ShapeMismatch { .. }));
}

// ---------- normalize_imagenet ----------

#[test]
fn normalize_imagenet_zero_mean_unit_std_for_synthetic_input() {
  // Construct an [H, W, 3] = [2, 2, 3] array where the per-channel mean
  // and std match the normalization parameters → output should be
  // ~zero-mean, unit-std after the per-channel (x - mean) / std.
  // Per-channel data:
  //   ch0: [1, 2, 3, 4], mean = 2.5, std (population) = sqrt(1.25) ≈ 1.118
  //   ch1: [10, 20, 30, 40], mean = 25.0, std ≈ 11.18
  //   ch2: [100, 100, 100, 100], mean = 100, std = 0  (test with std=1 instead to avoid /0)
  let data: [f32; 12] = [
    1.0, 10.0, 100.0, 2.0, 20.0, 100.0, 3.0, 30.0, 100.0, 4.0, 40.0, 100.0,
  ];
  let arr = Array::from_slice(&data, &(2usize, 2, 3)).unwrap();
  let mean = [2.5_f32, 25.0, 100.0];
  let std = [1.118_034_f32, 11.180_34, 1.0]; // sqrt(1.25), sqrt(125), 1
  let mut out = normalize_imagenet(&arr, &mean, &std).unwrap();
  let v: Vec<f32> = out.to_vec().unwrap();
  // Per channel: subtract mean, divide std.
  let expected: [f32; 12] = [
    (1.0 - 2.5) / 1.118_034,
    (10.0 - 25.0) / 11.180_34,
    0.0,
    (2.0 - 2.5) / 1.118_034,
    (20.0 - 25.0) / 11.180_34,
    0.0,
    (3.0 - 2.5) / 1.118_034,
    (30.0 - 25.0) / 11.180_34,
    0.0,
    (4.0 - 2.5) / 1.118_034,
    (40.0 - 25.0) / 11.180_34,
    0.0,
  ];
  assert!(vclose(&v, &expected), "got {v:?}\nexpected {expected:?}");
}

#[test]
fn normalize_imagenet_broadcasts_over_rank4_batch() {
  // [B, H, W, 3] = [2, 1, 1, 3]: two singletons, validate that the
  // (3,) mean/std broadcasts over the batch axis too.
  let arr = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(2usize, 1, 1, 3)).unwrap();
  let mean = [0.5_f32, 0.5, 0.5];
  let std = [2.0_f32, 2.0, 2.0];
  let mut out = normalize_imagenet(&arr, &mean, &std).unwrap();
  let v: Vec<f32> = out.to_vec().unwrap();
  let expected = [0.25_f32, 0.75, 1.25, 1.75, 2.25, 2.75];
  assert!(vclose(&v, &expected));
}

#[test]
fn normalize_imagenet_rejects_non_3_channel_input() {
  // [H, W, 4]: trailing dim 4 is not RGB → ShapeMismatch
  let arr = Array::from_slice(&[0.0_f32; 16], &(2usize, 2, 4)).unwrap();
  let err = normalize_imagenet(&arr, &[0.0; 3], &[1.0; 3]).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "want ShapeMismatch, got {err:?}"
  );
}

#[test]
fn normalize_imagenet_rejects_non_three_trailing_dim() {
  // Trailing dim must equal 3 (R,G,B) for the per-channel mean/std
  // broadcast to be well-defined. A rank-1 `[1]` tensor has trailing
  // dim 1 → ShapeMismatch. (Renamed from `_rejects_zero_rank` per
  // Copilot review #3272880185 — the test never built a true 0-D
  // scalar; it validates the non-3-channel-trailing-dim path.)
  let arr = Array::from_slice(&[1.0_f32], &(1usize,)).unwrap();
  let err = normalize_imagenet(&arr, &[0.0; 3], &[1.0; 3]).unwrap_err();
  assert!(matches!(err, Error::ShapeMismatch { .. }));
}

#[test]
fn normalize_imagenet_rejects_integer_dtypes() {
  // U8 [H, W, 3]: ImageNet mean/std cast to U8 would floor to 0,
  // producing garbage. Reject with ShapeMismatch so the caller is
  // forced to `astype(arr, Dtype::F32)` first.
  let arr = Array::from_slice(&[0_u8; 3], &(1usize, 1, 3)).unwrap();
  let err = normalize_imagenet(&arr, &[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225]).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "want ShapeMismatch, got {err:?}",
  );
}

// ---------- patchify ----------

#[test]
fn patchify_uniform_grid_shape() {
  // [4, 4, 3] with patch_size 2 → [4 (= 2*2), 2, 2, 3]
  let n = 4 * 4 * 3;
  let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
  let arr = Array::from_slice(&data, &(4usize, 4, 3)).unwrap();
  let out = patchify(&arr, 2).unwrap();
  assert_eq!(out.shape(), vec![4, 2, 2, 3]);
}

#[test]
fn patchify_non_divisible_dimensions_errors() {
  let arr = Array::from_slice(&[0.0_f32; 5 * 4 * 3], &(5usize, 4, 3)).unwrap();
  let err = patchify(&arr, 2).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "want ShapeMismatch, got {err:?}"
  );
}

#[test]
fn patchify_zero_patch_size_errors() {
  let arr = Array::from_slice(&[0.0_f32; 12], &(2usize, 2, 3)).unwrap();
  let err = patchify(&arr, 0).unwrap_err();
  assert!(matches!(err, Error::ShapeMismatch { .. }));
}

#[test]
fn patchify_wrong_rank_errors() {
  // [2, 3] is rank 2 → reject.
  let arr = Array::from_slice(&[0.0_f32; 6], &(2usize, 3)).unwrap();
  let err = patchify(&arr, 1).unwrap_err();
  assert!(matches!(err, Error::ShapeMismatch { .. }));
}

#[test]
fn patchify_unit_patch_size_passthrough_shape() {
  // patch_size=1 yields [H*W, 1, 1, C] — every pixel becomes its own
  // 1x1 patch.
  let arr = Array::from_slice(&[0.0_f32; 12], &(2usize, 2, 3)).unwrap();
  let out = patchify(&arr, 1).unwrap();
  assert_eq!(out.shape(), vec![4, 1, 1, 3]);
}

#[test]
fn patchify_preserves_pixel_values() {
  // Build a small image where every value is unique, patchify it, and
  // assert no value is lost or duplicated.
  let n = 4 * 4 * 3;
  let data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect(); // [1..=48]
  let arr = Array::from_slice(&data, &(4usize, 4, 3)).unwrap();
  let mut out = patchify(&arr, 2).unwrap();
  let mut sorted: Vec<f32> = out.to_vec().unwrap();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let expected: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
  assert_eq!(sorted, expected, "all pixels preserved");
}

// ---------- preprocess pipeline ----------

#[test]
fn preprocess_pipeline_imagenet_defaults() {
  // Default config: 224x224, ImageNet mean/std, 1/255 rescale.
  // We give it a 16x16 gradient and verify the full pipeline runs
  // without error and produces the expected output shape + dtype.
  let img = gradient_image(16, 16);
  let cfg = ImageProcessorConfig::default();
  let out = preprocess(&img, &cfg).unwrap();
  assert_eq!(out.shape(), vec![224, 224, 3]);
  assert_eq!(out.dtype().unwrap(), Dtype::F32);
}

#[test]
fn preprocess_no_resize_passthrough() {
  // do_resize=false: output spatial dims match the input image, not
  // cfg.size.
  let img = gradient_image(8, 6);
  let cfg = ImageProcessorConfig {
    size: (32, 32),
    do_resize: false,
    do_rescale: false,
    do_normalize: false,
    ..ImageProcessorConfig::default()
  };
  let mut out = preprocess(&img, &cfg).unwrap();
  // [H=6, W=8, 3]: cfg.size was ignored because do_resize=false.
  assert_eq!(out.shape(), vec![6, 8, 3]);
  let v: Vec<f32> = out.to_vec().unwrap();
  // Without rescale + normalize, the values must remain in [0, 255].
  assert!(v.iter().all(|&x| (0.0..=255.0).contains(&x)));
}

#[test]
fn preprocess_no_normalize_passthrough() {
  // do_normalize=false: output is just rescaled, no per-channel
  // subtract/divide; values in [0, 1].
  let img = synthetic_image(4, 4);
  let cfg = ImageProcessorConfig {
    do_resize: false,
    do_rescale: true,
    do_normalize: false,
    rescale_factor: 1.0 / 255.0,
    ..ImageProcessorConfig::default()
  };
  let mut out = preprocess(&img, &cfg).unwrap();
  let v: Vec<f32> = out.to_vec().unwrap();
  assert!(v.iter().all(|&x| (0.0..=1.0).contains(&x)));
}

#[test]
fn preprocess_no_rescale_no_normalize_keeps_raw_u8_range() {
  // do_rescale=false + do_normalize=false: output is the raw [0, 255]
  // f32 buffer, no other transform.
  let img = synthetic_image(2, 2);
  let cfg = ImageProcessorConfig {
    do_resize: false,
    do_rescale: false,
    do_normalize: false,
    ..ImageProcessorConfig::default()
  };
  let mut out = preprocess(&img, &cfg).unwrap();
  let v: Vec<f32> = out.to_vec().unwrap();
  assert!(v.iter().all(|&x| (0.0..=255.0).contains(&x)));
}

#[test]
fn preprocess_resize_applies_filter() {
  // do_resize=true, target size matches the input dims → output shape
  // matches.
  let img = synthetic_image(8, 8);
  let cfg = ImageProcessorConfig {
    size: (4, 4),
    do_resize: true,
    do_rescale: false,
    do_normalize: false,
    ..ImageProcessorConfig::default()
  };
  let out = preprocess(&img, &cfg).unwrap();
  assert_eq!(out.shape(), vec![4, 4, 3]);
}

#[test]
fn imageprocessor_config_default_is_imagenet() {
  let cfg = ImageProcessorConfig::default();
  assert_eq!(cfg.size, (224, 224));
  assert!(vclose(&cfg.mean, &[0.485, 0.456, 0.406]));
  assert!(vclose(&cfg.std, &[0.229, 0.224, 0.225]));
  assert!(close(cfg.rescale_factor, 1.0 / 255.0));
  assert!(cfg.do_resize && cfg.do_rescale && cfg.do_normalize);
  assert_eq!(cfg.resample, ResizeFilter::Bicubic);
  assert_eq!(cfg.color_order, ColorOrder::Rgb);
}

// ---------- load_image (light disk round-trip) ----------

#[test]
fn load_image_decodes_png_round_trip() {
  // Encode a small synthetic image as PNG into a tempfile, then
  // load_image it back and assert the decoded dimensions match. PNG
  // does not carry EXIF orientation, so `decoder.orientation()` returns
  // `Orientation::NoTransforms` and `apply_orientation` is a no-op —
  // this verifies the new `ImageReader` + orientation pipeline is a
  // clean drop-in for the common non-rotating case.
  let img = synthetic_image(5, 7); // 5 wide, 7 tall
  let dir = std::env::temp_dir().join(format!("mlxrs-vlm-image-test-{}", std::process::id(),));
  std::fs::create_dir_all(&dir).unwrap();
  let path = dir.join("synthetic.png");
  img
    .save_with_format(&path, ::image::ImageFormat::Png)
    .expect("encode");
  let loaded = load_image(&path).expect("decode");
  assert_eq!(loaded.width(), 5);
  assert_eq!(loaded.height(), 7);
  // Best-effort cleanup; the OS will GC /tmp eventually if this fails.
  let _ = std::fs::remove_file(&path);
  let _ = std::fs::remove_dir(&dir);
}

#[test]
fn load_image_nonexistent_path_returns_err() {
  let path = std::path::PathBuf::from(format!(
    "/tmp/mlxrs-vlm-image-does-not-exist-{}.png",
    std::process::id(),
  ));
  let err = load_image(&path).unwrap_err();
  assert!(matches!(err, Error::Backend { .. }), "got {err:?}");
}
