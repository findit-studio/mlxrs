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
    ColorOrder, ImageProcessorConfig, ResizeFilter, center_crop, image_to_array, load_image,
    normalize, normalize_imagenet, pad_to_square, patchify, preprocess, rescale, resize,
    resize_lanczos,
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
  //
  // Regression for Codex review (high): with the prior
  // `img.to_rgb8()` clone removed, this test now exercises the
  // non-`Rgb8` per-pixel `dynamic_image_rgb_pixel` projection rather
  // than the fast `as_rgb8()` path. The expected `[R, G, B]` triple
  // is unchanged because the projection is byte-equivalent to
  // `to_rgb8()` on `Rgba8` inputs (`Rgba.to_rgb()` drops alpha).
  let mut buf = ::image::RgbaImage::new(1, 1);
  buf.put_pixel(0, 0, ::image::Rgba([11, 22, 33, 44]));
  let img = ::image::DynamicImage::ImageRgba8(buf);
  let mut arr = image_to_array(&img, ColorOrder::Rgb).unwrap();
  assert_eq!(arr.shape(), vec![1, 1, 3], "alpha channel dropped");
  let v: Vec<f32> = arr.to_vec().unwrap();
  assert!(vclose(&v, &[11.0, 22.0, 33.0]));
}

#[test]
fn image_to_array_luma8_broadcasts_grey_across_rgb() {
  // Regression for Codex review (high): the non-`Rgb8` per-pixel
  // path replaces the prior infallible `img.to_rgb8()` clone.
  // Luma8 sources must broadcast the grey value across all three
  // RGB channels — identical projection to what `to_rgb8()` did
  // (image-rs's `Luma::to_rgb()` returns `(L, L, L)`).
  let mut buf = ::image::GrayImage::new(2, 2);
  buf.put_pixel(0, 0, ::image::Luma([10]));
  buf.put_pixel(1, 0, ::image::Luma([50]));
  buf.put_pixel(0, 1, ::image::Luma([100]));
  buf.put_pixel(1, 1, ::image::Luma([200]));
  let img = ::image::DynamicImage::ImageLuma8(buf);
  let mut arr = image_to_array(&img, ColorOrder::Rgb).unwrap();
  assert_eq!(arr.shape(), vec![2, 2, 3]);
  let v: Vec<f32> = arr.to_vec().unwrap();
  // Row-major (H=2, W=2, 3): pixel (x, y) at index `(y * W + x) * 3 + c`.
  // Each grey value broadcasts across (R, G, B).
  assert!(
    vclose(
      &v,
      &[
        10.0, 10.0, 10.0, // (0, 0)
        50.0, 50.0, 50.0, // (1, 0)
        100.0, 100.0, 100.0, // (0, 1)
        200.0, 200.0, 200.0, // (1, 1)
      ],
    ),
    "got {v:?}",
  );
}

#[test]
fn image_to_array_luma8_bgr_path_still_broadcasts_grey() {
  // BGR on a Luma8 source: since L → (L, L, L), the channel swap is
  // a no-op. Verifies the non-`Rgb8` branch handles `ColorOrder::Bgr`
  // correctly (the per-pixel match arms).
  let mut buf = ::image::GrayImage::new(1, 2);
  buf.put_pixel(0, 0, ::image::Luma([77]));
  buf.put_pixel(0, 1, ::image::Luma([200]));
  let img = ::image::DynamicImage::ImageLuma8(buf);
  let mut arr = image_to_array(&img, ColorOrder::Bgr).unwrap();
  assert_eq!(arr.shape(), vec![2, 1, 3]);
  let v: Vec<f32> = arr.to_vec().unwrap();
  assert!(
    vclose(&v, &[77.0, 77.0, 77.0, 200.0, 200.0, 200.0]),
    "got {v:?}",
  );
}

#[test]
fn image_to_array_rgba8_bgr_swaps_channels_drops_alpha() {
  // RGBA → BGR via the per-pixel non-`Rgb8` path. Alpha is dropped
  // and the remaining R/G/B is swapped to B/G/R. Verifies the
  // per-pixel `ColorOrder::Bgr` arm in the non-`Rgb8` branch.
  let mut buf = ::image::RgbaImage::new(2, 1);
  buf.put_pixel(0, 0, ::image::Rgba([10, 20, 30, 99]));
  buf.put_pixel(1, 0, ::image::Rgba([40, 50, 60, 88]));
  let img = ::image::DynamicImage::ImageRgba8(buf);
  let mut arr = image_to_array(&img, ColorOrder::Bgr).unwrap();
  assert_eq!(arr.shape(), vec![1, 2, 3]);
  let v: Vec<f32> = arr.to_vec().unwrap();
  // BGR (alpha dropped): (B=30, G=20, R=10) then (60, 50, 40).
  assert!(
    vclose(&v, &[30.0, 20.0, 10.0, 60.0, 50.0, 40.0]),
    "got {v:?}",
  );
}

#[test]
fn image_to_array_rgb_preserves_row_major_layout() {
  // Hand-computed 4x4 RGB image: pixel at (x, y) = ((x + 1) * 10,
  // (y + 1) * 20, x + y). Channel-last [H=4, W=4, 3] flattens
  // row-major: index = (y * W + x) * 3 + c. Verifies the
  // `chunks_exact(3)` + `extend(map(as f32))` buffer fill emits the
  // exact same byte sequence as the prior per-pixel push form.
  let (w, h) = (4u32, 4u32);
  let mut buf = ::image::RgbImage::new(w, h);
  for y in 0..h {
    for x in 0..w {
      let r = ((x + 1) * 10) as u8;
      let g = ((y + 1) * 20) as u8;
      let b = (x + y) as u8;
      buf.put_pixel(x, y, ::image::Rgb([r, g, b]));
    }
  }
  let img = ::image::DynamicImage::ImageRgb8(buf);
  let mut arr = image_to_array(&img, ColorOrder::Rgb).unwrap();
  assert_eq!(arr.shape(), vec![h as usize, w as usize, 3]);
  let v: Vec<f32> = arr.to_vec().unwrap();
  // Build the expected row-major sequence by hand.
  let mut expected = Vec::with_capacity((h * w * 3) as usize);
  for y in 0..h {
    for x in 0..w {
      expected.push(((x + 1) * 10) as f32);
      expected.push(((y + 1) * 20) as f32);
      expected.push((x + y) as f32);
    }
  }
  assert!(vclose(&v, &expected), "got {v:?}\nexpected {expected:?}");
}

#[test]
fn image_to_array_bgr_swaps_channels_correctly() {
  // Same 4x4 image as above; BGR output must have R and B columns
  // swapped at every pixel while preserving (H, W, 3) row-major
  // ordering. Verifies the `chunks_exact(3)` BGR branch produces the
  // exact byte sequence the prior `pixels()` swap form did.
  let (w, h) = (4u32, 4u32);
  let mut buf = ::image::RgbImage::new(w, h);
  for y in 0..h {
    for x in 0..w {
      let r = ((x + 1) * 10) as u8;
      let g = ((y + 1) * 20) as u8;
      let b = (x + y) as u8;
      buf.put_pixel(x, y, ::image::Rgb([r, g, b]));
    }
  }
  let img = ::image::DynamicImage::ImageRgb8(buf);
  let mut arr = image_to_array(&img, ColorOrder::Bgr).unwrap();
  assert_eq!(arr.shape(), vec![h as usize, w as usize, 3]);
  let v: Vec<f32> = arr.to_vec().unwrap();
  let mut expected = Vec::with_capacity((h * w * 3) as usize);
  for y in 0..h {
    for x in 0..w {
      // BGR: B, G, R per pixel.
      expected.push((x + y) as f32);
      expected.push(((y + 1) * 20) as f32);
      expected.push(((x + 1) * 10) as f32);
    }
  }
  assert!(vclose(&v, &expected), "got {v:?}\nexpected {expected:?}");
}

// NOTE: `image_to_array` carries a `checked_mul` overflow guard for the
// `h*w*3` product (defense-in-depth on a 32-bit `usize` target). On the
// 64-bit targets we build for, triggering that guard would require an
// `RgbImage` of dimensions whose product overflows `usize` — roughly
// `u32::MAX * u32::MAX * 3` bytes of decoded pixel data (~50 EB), which
// is unreachable through the public API. The guard exists to surface
// the wrap as a recoverable `Error::ShapeMismatch` instead of a silent
// `Vec::with_capacity` panic, and is covered by the algebraic
// `checked_mul` operator itself.

#[test]
fn image_to_array_rgb_overlong_backing_buffer_ignores_tail() {
  // `ImageBuffer::from_raw(w, h, vec)` accepts a backing Vec longer
  // than `w * h * 3`; `as_raw()` returns the full backing buffer
  // (including the tail past the logical extent). The new
  // `.get(..total)` slice must clip the iteration to exactly H*W*3
  // bytes — without it, `Vec::extend` would grow `buf` past the
  // `try_reserve_exact(total)` reservation via infallible allocation,
  // reintroducing the abort-on-OOM hazard.
  let mut overlong: Vec<u8> = vec![10, 20, 30]; // 1*1*3 logical pixel
  overlong.extend_from_slice(&[99, 99, 99, 99]); // 4-byte tail
  let buf = ::image::RgbImage::from_raw(1, 1, overlong).expect("from_raw 1x1+tail");
  let img = ::image::DynamicImage::ImageRgb8(buf);
  let mut arr = image_to_array(&img, ColorOrder::Rgb).unwrap();
  assert_eq!(arr.shape(), vec![1, 1, 3]);
  let v: Vec<f32> = arr.to_vec().unwrap();
  // Tail bytes (99s) MUST NOT appear; only the logical pixel's R=10,G=20,B=30.
  assert!(
    vclose(&v, &[10.0, 20.0, 30.0]),
    "got {v:?}, expected [10,20,30]"
  );
}

#[test]
fn image_to_array_bgr_overlong_backing_buffer_ignores_tail() {
  let mut overlong: Vec<u8> = vec![10, 20, 30];
  overlong.extend_from_slice(&[99, 99, 99, 99]);
  let buf = ::image::RgbImage::from_raw(1, 1, overlong).expect("from_raw 1x1+tail");
  let img = ::image::DynamicImage::ImageRgb8(buf);
  let mut arr = image_to_array(&img, ColorOrder::Bgr).unwrap();
  assert_eq!(arr.shape(), vec![1, 1, 3]);
  let v: Vec<f32> = arr.to_vec().unwrap();
  // BGR swap on the logical pixel: B=30, G=20, R=10. Tail bytes (99s)
  // must NOT contribute to the output.
  assert!(
    vclose(&v, &[30.0, 20.0, 10.0]),
    "got {v:?}, expected [30,20,10]"
  );
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
  // load_image it back and assert the decoded dimensions match. This
  // synthetic PNG carries no EXIF orientation metadata (image-rs 0.25
  // PNG decoders CAN expose EXIF orientation via `exif_metadata` —
  // see the `load_image` doc — but our `synthetic_image` builder
  // doesn't write one), so `decoder.orientation()` returns
  // `Orientation::NoTransforms` and `apply_orientation_fallible` is a
  // no-op here — this verifies the `ImageReader` + orientation
  // pipeline is a clean drop-in for the common non-rotating case.
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

// ---------- resize_lanczos ----------

#[test]
fn resize_lanczos_target_dimensions() {
  // 8x6 source → 16x32 target via Lanczos3. Argument order is
  // (target_h, target_w) matching the python image-processor
  // convention; output width/height must match exactly.
  let img = synthetic_image(8, 6);
  let out = resize_lanczos(&img, 16, 32);
  assert_eq!(out.width(), 32);
  assert_eq!(out.height(), 16);
}

#[test]
fn resize_lanczos_equivalent_to_resize_with_lanczos3_filter() {
  // resize_lanczos is documented as a thin wrapper around
  // resize(..., Lanczos3) — byte-for-byte output equality is the
  // strongest assertion of that contract.
  let img = synthetic_image(12, 10);
  let a = resize_lanczos(&img, 8, 16);
  let b = resize(&img, (8, 16), ResizeFilter::Lanczos3);
  assert_eq!(a.to_rgba8().into_raw(), b.to_rgba8().into_raw());
}

#[test]
fn resize_lanczos_smooth_on_constant_input_preserves_value() {
  // Lanczos3 on a constant-color input must reproduce the constant
  // (up to small floating-point error) at every output pixel — the
  // sinc kernel sums to 1, so any value `c` survives. Use a
  // mid-grey RGB pixel; check that downsample-then-upsample stays
  // tightly bounded near the source value.
  let mut buf = ::image::RgbImage::new(8, 8);
  for y in 0..8 {
    for x in 0..8 {
      buf.put_pixel(x, y, ::image::Rgb([128, 64, 200]));
    }
  }
  let img = ::image::DynamicImage::ImageRgb8(buf);
  let out = resize_lanczos(&img, 4, 4);
  assert_eq!(out.width(), 4);
  assert_eq!(out.height(), 4);
  // Every output pixel should land within 1 LSB of the source value
  // — Lanczos3 on a constant-color image is exact up to integer
  // rounding (rounding bias at the edge of the kernel may shift by
  // at most 1 byte).
  let rgba = out.to_rgba8();
  for px in rgba.pixels() {
    let [r, g, b, _] = px.0;
    assert!(r.abs_diff(128) <= 1, "R={r} expected ~128");
    assert!(g.abs_diff(64) <= 1, "G={g} expected ~64");
    assert!(b.abs_diff(200) <= 1, "B={b} expected ~200");
  }
}

// ---------- center_crop ----------

#[test]
fn center_crop_4x4_to_2x2_returns_center_pixels() {
  // Hand-traced: source = 4x4 with pixel (x, y) = (10*x + y, 0, 0).
  // The center 2x2 crop is rows y=1..3, cols x=1..3, so the cropped
  // R values are:
  //   (1, 1)=11  (2, 1)=21
  //   (1, 2)=12  (2, 2)=22
  let mut buf = ::image::RgbImage::new(4, 4);
  for y in 0..4 {
    for x in 0..4 {
      buf.put_pixel(x, y, ::image::Rgb([(10 * x + y) as u8, 0, 0]));
    }
  }
  let img = ::image::DynamicImage::ImageRgb8(buf);
  let out = center_crop(&img, 2, 2);
  assert_eq!(out.width(), 2);
  assert_eq!(out.height(), 2);
  let rgb = out.to_rgb8();
  // Row-major: (0, 0)=11, (1, 0)=21, (0, 1)=12, (1, 1)=22.
  assert_eq!(rgb.get_pixel(0, 0).0, [11, 0, 0]);
  assert_eq!(rgb.get_pixel(1, 0).0, [21, 0, 0]);
  assert_eq!(rgb.get_pixel(0, 1).0, [12, 0, 0]);
  assert_eq!(rgb.get_pixel(1, 1).0, [22, 0, 0]);
}

#[test]
fn center_crop_source_smaller_returns_source_unchanged() {
  // Swift `rectSmallerOrEqual` early-return: a 4x4 source asked for
  // an 8x8 crop returns the original image untouched. We check
  // dimensions + a sample pixel.
  let img = synthetic_image(4, 4);
  let out = center_crop(&img, 8, 8);
  assert_eq!(out.width(), 4);
  assert_eq!(out.height(), 4);
  // synthetic_image: pixel (x, y) = (10*y, 10*x, 100).
  assert_eq!(out.to_rgb8().get_pixel(2, 3).0, [30, 20, 100]);
}

#[test]
fn center_crop_one_axis_smaller_clamps_and_crops_bigger_axis() {
  // Mirrors swift `rectSmallerOrEqual` + `centerCrop`'s `min(source,
  // target)` clamp (`MediaProcessing.swift:201-210`): when only one
  // axis exceeds the target, `crop_w = min(source_w, target_w)`,
  // `crop_h = min(source_h, target_h)`, and the bigger axis is
  // center-cropped. Source `(w=4, h=8)`, target `(target_h=4,
  // target_w=6)` → `crop_w = min(4, 6) = 4`, `crop_h = min(8, 4) = 4`,
  // y-offset = `(8 - 4) / 2 = 2` → output 4x4 taken from y=2..6.
  let img = synthetic_image(4, 8);
  let out = center_crop(&img, 4, 6);
  assert_eq!(out.width(), 4);
  assert_eq!(out.height(), 4);
  // synthetic_image: pixel (x, y) = (10*y, 10*x, 100). The cropped
  // window starts at y=2, so out's (0, 0) pixel is the source's
  // (0, 2) = (20, 0, 100).
  assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [20, 0, 100]);
  assert_eq!(out.to_rgb8().get_pixel(3, 3).0, [50, 30, 100]); // src (3, 5)
}

#[test]
fn center_crop_height_only_larger_crops_height_keeps_width() {
  // Regression for Codex Finding 1 (OR-bug): a source whose width
  // exactly equals `target_w` but whose height exceeds `target_h`
  // must still crop the height. Pre-fix code returned the source
  // unchanged because `w <= target_w` short-circuited the OR.
  //
  // Source `(w=2, h=8)`, target `(target_h=4, target_w=2)` →
  // `crop_w = min(2, 2) = 2`, `crop_h = min(8, 4) = 4`,
  // y-offset = `(8 - 4) / 2 = 2` → output 2 (W) x 4 (H).
  let img = synthetic_image(2, 8);
  let out = center_crop(&img, 4, 2);
  assert_eq!(out.width(), 2);
  assert_eq!(out.height(), 4);
  // synthetic_image: pixel (x, y) = (10*y, 10*x, 100). Cropped window
  // starts at y=2 → out's (0, 0) is source (0, 2) = (20, 0, 100).
  let rgb = out.to_rgb8();
  assert_eq!(rgb.get_pixel(0, 0).0, [20, 0, 100]);
  assert_eq!(rgb.get_pixel(1, 0).0, [20, 10, 100]); // src (1, 2)
  assert_eq!(rgb.get_pixel(0, 3).0, [50, 0, 100]); // src (0, 5)
  assert_eq!(rgb.get_pixel(1, 3).0, [50, 10, 100]); // src (1, 5)
}

// ---------- pad_to_square ----------

#[test]
fn pad_to_square_4x2_with_black_fill_produces_4x4_with_pad_rows() {
  // Source = 4 wide × 2 tall, R-channel = 10*x at every y.
  // (long - short) / 2 = (4 - 2) / 2 = 1 row of fill on top, 1 on
  // bottom. Result: 4x4 with rows 0 and 3 filled, rows 1 and 2 the
  // source.
  let mut buf = ::image::RgbImage::new(4, 2);
  for y in 0..2 {
    for x in 0..4 {
      buf.put_pixel(x, y, ::image::Rgb([(10 * x) as u8, 200, 50]));
    }
  }
  let img = ::image::DynamicImage::ImageRgb8(buf);
  let out = pad_to_square(img, [0, 0, 0]).unwrap();
  assert_eq!(out.width(), 4);
  assert_eq!(out.height(), 4);
  let rgb = out.to_rgb8();
  // Top pad row.
  for x in 0..4 {
    assert_eq!(
      rgb.get_pixel(x, 0).0,
      [0, 0, 0],
      "row 0 must be fill; x={x}"
    );
  }
  // Source rows at y=1 and y=2.
  for y in 1..3 {
    for x in 0..4 {
      assert_eq!(
        rgb.get_pixel(x, y).0,
        [(10 * x) as u8, 200, 50],
        "source row y={y} x={x}"
      );
    }
  }
  // Bottom pad row.
  for x in 0..4 {
    assert_eq!(
      rgb.get_pixel(x, 3).0,
      [0, 0, 0],
      "row 3 must be fill; x={x}"
    );
  }
}

#[test]
fn pad_to_square_2x4_pads_left_and_right() {
  // Source = 2 wide × 4 tall, asymmetric R channel. Pad symmetric on
  // the x axis: 1 col fill, 2 cols source, 1 col fill.
  let mut buf = ::image::RgbImage::new(2, 4);
  for y in 0..4 {
    for x in 0..2 {
      buf.put_pixel(x, y, ::image::Rgb([(10 * x + y) as u8, 1, 2]));
    }
  }
  let img = ::image::DynamicImage::ImageRgb8(buf);
  let out = pad_to_square(img, [255, 128, 64]).unwrap();
  assert_eq!(out.width(), 4);
  assert_eq!(out.height(), 4);
  let rgb = out.to_rgb8();
  // Pad columns.
  for y in 0..4 {
    assert_eq!(rgb.get_pixel(0, y).0, [255, 128, 64]);
    assert_eq!(rgb.get_pixel(3, y).0, [255, 128, 64]);
  }
  // Source columns x=1..3 → source x=0..2 (offset by x_off=1).
  for y in 0..4 {
    for x_src in 0..2u32 {
      assert_eq!(
        rgb.get_pixel(1 + x_src, y).0,
        [(10 * x_src + y) as u8, 1, 2],
      );
    }
  }
}

#[test]
fn pad_to_square_already_square_returns_input_unchanged() {
  // No alloc / no padding when w == h; the by-value signature returns
  // the input `DynamicImage` directly (no `clone()` — see the
  // signature doc for why `Clone` would re-introduce a panic-abort
  // on near-budget inputs). Output dims and a sample pixel must
  // match the source.
  let img = synthetic_image(4, 4);
  let out = pad_to_square(img, [99, 99, 99]).unwrap();
  assert_eq!(out.width(), 4);
  assert_eq!(out.height(), 4);
  // synthetic_image pixel (2, 3) = (10*3, 10*2, 100) = (30, 20, 100).
  assert_eq!(out.to_rgb8().get_pixel(2, 3).0, [30, 20, 100]);
}

#[test]
fn pad_to_square_odd_difference_extra_row_on_bottom() {
  // Source = 3 wide × 2 tall. (long - short) = 1 → integer floor
  // puts 0 rows on top, 1 row of pad on the bottom (matching python
  // `Image.new(...).paste(img, (0, 0))` with the source at the top
  // when (width - height) // 2 == 0).
  let mut buf = ::image::RgbImage::new(3, 2);
  for y in 0..2 {
    for x in 0..3 {
      buf.put_pixel(x, y, ::image::Rgb([(x + 10 * y) as u8, 7, 8]));
    }
  }
  let img = ::image::DynamicImage::ImageRgb8(buf);
  let out = pad_to_square(img, [42, 43, 44]).unwrap();
  assert_eq!((out.width(), out.height()), (3, 3));
  let rgb = out.to_rgb8();
  // Source at rows 0 and 1, pad row at row 2.
  for y in 0..2 {
    for x in 0..3 {
      assert_eq!(
        rgb.get_pixel(x, y).0,
        [(x + 10 * y) as u8, 7, 8],
        "source y={y} x={x}",
      );
    }
  }
  for x in 0..3 {
    assert_eq!(rgb.get_pixel(x, 2).0, [42, 43, 44], "pad row x={x}");
  }
}

#[test]
fn pad_to_square_rejects_oversized_canvas() {
  // Regression for Codex Finding 2 (quadratic alloc OOM): a 100_000 x 1
  // source would drive a 100_000² × 3 ≈ 30 GiB canvas — the prior
  // infallible `RgbImage::from_pixel(size, size, ...)` would
  // vec-overflow / OOM-abort the process. The fallible signature
  // must surface this as a recoverable `Error::ShapeMismatch`,
  // bounded by `MAX_DECODED_IMAGE_BYTES` (matches `load_image`'s 512
  // MiB ceiling).
  //
  // The source itself is only 100_000 × 1 × 3 = ~300 KiB (fine to
  // allocate as the test fixture). Only the would-be `pad_to_square`
  // output trips the bound.
  let img = ::image::DynamicImage::ImageRgb8(::image::RgbImage::new(100_000, 1));
  let err = pad_to_square(img, [0, 0, 0]).expect_err("oversized canvas must be rejected");
  match err {
    Error::ShapeMismatch { message } => {
      assert!(
        message.contains("pad_to_square") && message.contains("MAX_DECODED_IMAGE_BYTES"),
        "expected ShapeMismatch mentioning the budget; got: {message}"
      );
    }
    other => panic!("expected ShapeMismatch, got {other:?}"),
  }
}

#[test]
fn pad_to_square_near_budget_nonsquare_no_second_source_copy() {
  // Regression for Codex Finding (high): the prior `img.to_rgb8()` call
  // inside the nonsquare branch materialized a *second* source-sized
  // RGB buffer infallibly. A near-budget nonsquare RGB input
  // (e.g. 13377 × 13376) would pass the `MAX_DECODED_IMAGE_BYTES`
  // canvas gate, fallibly reserve the ~512 MiB canvas, then
  // panic-abort on the second ~512 MiB `to_rgb8` clone the canvas
  // gate doesn't cover.
  //
  // We exercise the same code path (already-`Rgb8` source → row-wise
  // `copy_from_slice` fast path inside `pad_to_square`) with a much
  // smaller proxy so CI doesn't actually allocate hundreds of MiB.
  // The 4097 × 4096 = 1px-shorter-than-square shape lands the source
  // in the `w > h` branch and the canvas at
  // `4097 * 4097 * 3 ≈ 48 MiB` — well under the budget but big
  // enough to exercise both the canvas alloc and the per-row
  // `copy_from_slice` path. The check that matters: the call returns
  // either `Ok(_)` or a recoverable `Err`, NOT a panic / abort.
  let mut buf = ::image::RgbImage::new(4097, 4096);
  // Sparse marker pixels (corners + center of the source region) —
  // a full per-pixel fill would dominate the test budget and isn't
  // necessary for verifying the copy path.
  buf.put_pixel(0, 0, ::image::Rgb([1, 2, 3]));
  buf.put_pixel(4096, 0, ::image::Rgb([4, 5, 6]));
  buf.put_pixel(0, 4095, ::image::Rgb([7, 8, 9]));
  buf.put_pixel(4096, 4095, ::image::Rgb([10, 11, 12]));
  let img = ::image::DynamicImage::ImageRgb8(buf);
  // If this panics / aborts the process we'd never reach the
  // assertions — `cargo test` would surface the abort as a failed
  // test. `Ok` is the expected path under the 48 MiB budget; the
  // assertion guards the alternate-OOM case so a transient allocator
  // failure on CI does not silently no-op the regression coverage.
  let out = pad_to_square(img, [128, 128, 128])
    .expect("near-budget nonsquare path must return Ok or recoverable Err, never abort");
  // w > h → padding on the y axis; size = max(4097, 4096) = 4097.
  // y_off = (4097 - 4096) / 2 = 0 (integer floor — top edge stays
  // at 0 rows of fill, bottom edge gets the extra row).
  assert_eq!((out.width(), out.height()), (4097, 4097));
  let rgb = out.to_rgb8();
  // Source corners must land at their src coords (y_off = 0,
  // x_off = 0).
  assert_eq!(rgb.get_pixel(0, 0).0, [1, 2, 3], "src TL");
  assert_eq!(rgb.get_pixel(4096, 0).0, [4, 5, 6], "src TR");
  assert_eq!(rgb.get_pixel(0, 4095).0, [7, 8, 9], "src BL");
  assert_eq!(rgb.get_pixel(4096, 4095).0, [10, 11, 12], "src BR");
  // Bottom pad row (y = 4096) must be the uniform fill.
  assert_eq!(rgb.get_pixel(0, 4096).0, [128, 128, 128], "pad row TL");
  assert_eq!(rgb.get_pixel(4096, 4096).0, [128, 128, 128], "pad row TR");
}

#[test]
fn pad_to_square_non_rgb8_source_produces_rgb_output() {
  // Regression for Codex Finding (high): the per-pixel non-`Rgb8`
  // branch must produce correct RGB output without ever calling the
  // infallible `to_rgb8()` clone. Cover both Luma8 (grey →
  // broadcast across R/G/B) and Rgba8 (alpha → dropped, R/G/B
  // preserved) — the two non-`Rgb8` variants `image_to_array`
  // already accepts upstream.

  // --- Luma8: 3 wide × 2 tall, grey ramp by column. Pad rows on
  //     top/bottom to a 3x3 square.
  let mut luma = ::image::GrayImage::new(3, 2);
  for y in 0..2 {
    for x in 0..3 {
      luma.put_pixel(x, y, ::image::Luma([(10 * x + y) as u8]));
    }
  }
  let img_luma = ::image::DynamicImage::ImageLuma8(luma);
  let out = pad_to_square(img_luma, [200, 200, 200]).unwrap();
  assert_eq!((out.width(), out.height()), (3, 3));
  let rgb = out.to_rgb8();
  // y_off = (3 - 2) / 2 = 0 → source occupies rows 0..2, fill at
  // row 2. Each source pixel must broadcast the grey value across
  // all 3 RGB channels.
  for y in 0..2 {
    for x in 0..3 {
      let g = (10 * x + y) as u8;
      assert_eq!(
        rgb.get_pixel(x, y).0,
        [g, g, g],
        "luma broadcast y={y} x={x}",
      );
    }
  }
  for x in 0..3 {
    assert_eq!(rgb.get_pixel(x, 2).0, [200, 200, 200], "luma pad x={x}");
  }

  // --- Rgba8: 2 wide × 4 tall with an alpha gradient. The alpha
  //     must be dropped; R/G/B preserved.
  let mut rgba = ::image::RgbaImage::new(2, 4);
  for y in 0..4 {
    for x in 0..2 {
      rgba.put_pixel(
        x,
        y,
        ::image::Rgba([(10 * x + y) as u8, 50, 100, ((y * 60) % 256) as u8]),
      );
    }
  }
  let img_rgba = ::image::DynamicImage::ImageRgba8(rgba);
  let out = pad_to_square(img_rgba, [77, 88, 99]).unwrap();
  assert_eq!((out.width(), out.height()), (4, 4));
  let rgb = out.to_rgb8();
  // x_off = (4 - 2) / 2 = 1; source columns land at x=1..3, pad at
  // x=0 and x=3. Alpha must be dropped; R/G/B preserved.
  for y in 0..4 {
    assert_eq!(rgb.get_pixel(0, y).0, [77, 88, 99], "rgba pad L y={y}");
    assert_eq!(rgb.get_pixel(3, y).0, [77, 88, 99], "rgba pad R y={y}");
    for x_src in 0..2u32 {
      assert_eq!(
        rgb.get_pixel(1 + x_src, y).0,
        [(10 * x_src + y) as u8, 50, 100],
        "rgba src y={y} x_src={x_src} (alpha must be dropped)",
      );
    }
  }
}

// ---------- normalize (alias + standalone) ----------

#[test]
fn normalize_hand_computed_1x1x3() {
  // Tiny 1x1x3 array: x = [3.0, 5.0, 7.0]; mean = [1.0, 2.0, 3.0];
  // std = [2.0, 1.0, 0.5]. Expected (x - mean) / std =
  //   (3-1)/2 = 1.0,  (5-2)/1 = 3.0,  (7-3)/0.5 = 8.0.
  let arr = Array::from_slice(&[3.0_f32, 5.0, 7.0], &(1usize, 1, 3)).unwrap();
  let mean = [1.0_f32, 2.0, 3.0];
  let std = [2.0_f32, 1.0, 0.5];
  let mut out = normalize(&arr, &mean, &std).unwrap();
  let v: Vec<f32> = out.to_vec().unwrap();
  assert!(vclose(&v, &[1.0, 3.0, 8.0]), "got {v:?}");
}

#[test]
fn normalize_imagenet_is_alias_for_normalize() {
  // The deprecated `normalize_imagenet` name must produce
  // byte-identical output to the new `normalize` for the same inputs.
  let arr = Array::from_slice(&[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6], &(2usize, 1, 3)).unwrap();
  let mean = [0.485_f32, 0.456, 0.406];
  let std = [0.229_f32, 0.224, 0.225];
  let mut a = normalize(&arr, &mean, &std).unwrap();
  let mut b = normalize_imagenet(&arr, &mean, &std).unwrap();
  let va: Vec<f32> = a.to_vec().unwrap();
  let vb: Vec<f32> = b.to_vec().unwrap();
  assert!(vclose(&va, &vb));
}

#[test]
fn normalize_rejects_integer_dtypes() {
  // Integer input rejected with ShapeMismatch (mean/std cast to U8
  // would floor to zero → division undefined). Mirror the
  // `rescale_rejects_integer_dtypes` coverage for the renamed
  // function.
  let arr = Array::from_slice(&[0_u8; 3], &(1usize, 1, 3)).unwrap();
  let err = normalize(&arr, &[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225]).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "want ShapeMismatch, got {err:?}",
  );
}
