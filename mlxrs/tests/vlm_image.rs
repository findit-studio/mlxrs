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
  ops::shape::contiguous,
  vlm::image::{
    ColorOrder, ImageProcessorConfig, Layout, ResizeFilter, apply_layout, center_crop,
    image_to_array, load_image, normalize, normalize_imagenet, pad_to_square, patchify, preprocess,
    rescale, resize, resize_lanczos,
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
  let out = resize(&img, (16, 32), ResizeFilter::Bicubic).unwrap();
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
    let out = resize(&img, (4, 4), f).unwrap();
    assert_eq!((out.width(), out.height()), (4, 4), "filter {:?}", f);
  }
}

#[test]
fn resize_rejects_zero_target_dimension() {
  // Regression for Codex Finding 1 (high): `ImageProcessorConfig::size`
  // now flows from an UNTRUSTED loaded processor config. A zero target
  // dimension must be rejected as a recoverable `Error::ShapeMismatch`
  // BEFORE the source-RGBA materialization / the own resize kernel's
  // allocations, not silently produce a degenerate / panicking
  // allocation. The
  // source fixture is a normal small image; only the (untrusted)
  // target is degenerate.
  let img = synthetic_image(8, 6);
  for target in [(0, 16), (16, 0), (0, 0)] {
    let err = resize(&img, target, ResizeFilter::Bilinear)
      .expect_err("zero target dimension must be rejected");
    match err {
      Error::ShapeMismatch { message } => assert!(
        message.contains("resize") && message.contains("non-zero"),
        "expected ShapeMismatch naming the zero dim; got: {message}"
      ),
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }
}

#[test]
fn resize_rejects_oversized_target() {
  // Regression for Codex Finding 1 (high): a hostile/malformed loaded
  // config with an enormous `size` would drive the resize destination
  // allocation (`height * width * 4` bytes) to panic-abort the process,
  // taking down image + video preprocessing on the first request. The
  // target-byte guard must surface this as a recoverable
  // `Error::ShapeMismatch`,
  // bounded by `MAX_DECODED_IMAGE_BYTES` (the same 512 MiB ceiling
  // `pad_to_square` and `load_image` enforce).
  //
  // The source fixture is a tiny 8x6 image (~144 bytes); only the
  // would-be resize destination (100_000² × 4 ≈ 37 GiB) trips the cap,
  // so CI never actually allocates the oversized buffer.
  let img = synthetic_image(8, 6);
  let err = resize(&img, (100_000, 100_000), ResizeFilter::Bicubic)
    .expect_err("oversized resize target must be rejected");
  match err {
    Error::ShapeMismatch { message } => assert!(
      message.contains("resize") && message.contains("MAX_DECODED_IMAGE_BYTES"),
      "expected ShapeMismatch mentioning the budget; got: {message}"
    ),
    other => panic!("expected ShapeMismatch, got {other:?}"),
  }
}

#[test]
fn resize_rejects_overflowing_target_product() {
  // Regression for Codex Finding 1 (high): the byte-budget guard
  // computes `height * width * 4` via `u64::checked_mul`. A target whose
  // product overflows u64 (`u32::MAX` on both axes) must surface as a
  // recoverable `Error::ShapeMismatch` (overflow branch) rather than
  // wrapping to a small valid-looking byte count that bypasses the cap.
  let img = synthetic_image(8, 6);
  let err = resize(&img, (u32::MAX, u32::MAX), ResizeFilter::Bilinear)
    .expect_err("overflowing target product must be rejected");
  match err {
    Error::ShapeMismatch { message } => assert!(
      message.contains("resize") && message.contains("overflow"),
      "expected ShapeMismatch naming the overflow; got: {message}"
    ),
    other => panic!("expected ShapeMismatch, got {other:?}"),
  }
}

#[test]
fn resize_rejects_source_rgba_staging_over_cap() {
  // Regression for Codex Finding 2 (medium): `load_image`'s 512 MiB cap
  // is enforced on `decoder.total_bytes()` — the SOURCE pixel format. A
  // low-bytes-per-pixel source (Luma8, 1 B/px) can pass that cap yet
  // expand 4x when projected to the RGBA8 staging buffer `resize`
  // materializes. Before the fix, `resize` capped only the DESTINATION,
  // so an under-cap Luma8 could drive a ~2 GiB RGBA staging allocation.
  //
  // 16384x8193 Luma8 = ~128 MiB (well under the 512 MiB source cap as
  // Luma8), but `16384*8193*4 = 536_936_448 bytes > 512 MiB` as the RGBA8
  // staging — so `resize` must reject it with a recoverable
  // `ShapeMismatch` naming the RGBA-expanded byte count, BEFORE the
  // `try_reserve_exact`. The target is tiny (8x8) so the destination
  // guard does NOT fire — this isolates the SOURCE-staging guard.
  let w = 16_384u32;
  let h = 8_193u32;
  assert!(
    (w as u64 * h as u64) * 4 > 512 * 1024 * 1024,
    "fixture must exceed the 512 MiB cap as RGBA8 staging"
  );
  let luma = ::image::DynamicImage::ImageLuma8(
    ::image::ImageBuffer::from_raw(w, h, vec![0u8; (w as usize) * (h as usize)])
      .expect("Luma8 from_raw for the over-cap source fixture"),
  );
  let err = resize(&luma, (8, 8), ResizeFilter::Bilinear)
    .expect_err("Luma8 source whose RGBA8 staging exceeds the cap must be rejected");
  match err {
    Error::ShapeMismatch { message } => assert!(
      message.contains("resize")
        && message.contains("MAX_DECODED_IMAGE_BYTES")
        && message.contains("staging"),
      "expected ShapeMismatch naming the RGBA staging cap; got: {message}"
    ),
    other => panic!("expected ShapeMismatch for over-cap source staging, got {other:?}"),
  }
}

#[test]
fn resize_normal_luma8_source_still_resizes() {
  // The source-staging guard must NOT regress ordinary Luma8 inputs: a
  // small Luma8 image (whose RGBA8 staging is far under the cap) still
  // resizes successfully via the per-pixel fallible projection path.
  let mut luma = ::image::GrayImage::new(8, 6);
  for (i, p) in luma.pixels_mut().enumerate() {
    *p = ::image::Luma([(i * 5 % 256) as u8]);
  }
  let img = ::image::DynamicImage::ImageLuma8(luma);
  let out = resize(&img, (4, 4), ResizeFilter::Bilinear)
    .expect("a small Luma8 source must still resize (staging well under the cap)");
  assert_eq!(out.width(), 4, "resized width");
  assert_eq!(out.height(), 4, "resized height");
}

#[test]
fn resize_accepted_target_uses_fallible_alloc_path_never_aborts() {
  // Regression for Codex Finding (high, round 2) + the own-resize
  // omnibus follow-up: an accepted target AT OR BELOW the 512 MiB cap
  // formerly still allocated through INFALLIBLE buffers — `img.to_rgba8()`
  // (an owned RGBA clone) and, inside `fast_image_resize`, its internal
  // coefficient tables + per-row work buffers (allocated infallibly
  // inside that crate). A hostile config could pick a just-under-cap size
  // (~11585×11585 ≈ 512 MiB) and force a ~512 MiB infallible allocation →
  // process abort under memory pressure, despite the `Result` signature.
  // The cap bounded the SIZE but not the FALLIBILITY. The fix dropped
  // `fast_image_resize` for an OWN kernel (`vlm::resize`) whose every
  // buffer is `try_reserve_exact`-backed.
  //
  // We do NOT allocate 512 MiB in CI. Instead we exercise the SAME
  // fallible code path (the `try_reserve_exact`-backed source RGBA buffer
  // + the own kernel's `try_reserve_exact` coefficient tables, inter-pass
  // intermediate, and destination) at a moderate, CI-safe target that
  // genuinely allocates real buffers. The fallibility is STRUCTURAL:
  // every allocation in `resize` driven by the (now-untrusted) config
  // dims routes through `try_reserve_exact` and surfaces allocator
  // failure as `Error::OutOfMemory`. A successful return is `Ok`; an
  // allocator refusal would be a recoverable `Err(OutOfMemory)` — NEVER
  // an abort.
  let img = synthetic_image(8, 6);
  // 1024 x 768 destination = 768 * 1024 * 4 = 3 MiB — well under the cap,
  // large enough that BOTH the source-RGBA `try_reserve_exact` and the
  // destination `try_reserve_exact` + `from_vec_u8` actually run (not a
  // toy buffer the allocator never touches).
  match resize(&img, (768, 1024), ResizeFilter::Bilinear) {
    Ok(out) => {
      assert_eq!(out.width(), 1024, "width = target.1");
      assert_eq!(out.height(), 768, "height = target.0");
    }
    // Allocator could not satisfy the (bounded ≤512 MiB) reservation:
    // recoverable, not an abort. This branch documents that the
    // try_reserve path is wired — it should not fire for a 3 MiB buffer
    // on any CI host, but accepting it proves the contract is
    // "Ok-or-recoverable-Err, never abort".
    Err(Error::OutOfMemory) => {}
    Err(other) => panic!("expected Ok or recoverable OutOfMemory, got {other:?}"),
  }
}

#[test]
fn resize_non_rgba8_sources_convert_via_fallible_per_pixel_path() {
  // The source-RGBA materialization has two paths: a borrowed
  // `as_rgba8()` fast path (source already RGBA8) and a per-pixel
  // `dynamic_image_rgba_pixel` projection for every other variant
  // (Luma8 / Rgb8 / 16-bit / float). Both fill a `try_reserve_exact`
  // buffer. Verify the per-pixel path produces correct output for
  // non-RGBA8 sources, and that an already-RGBA8 source (fast path)
  // works too.

  // Rgb8 source (most common): synthetic_image yields ImageRgb8.
  let rgb8 = synthetic_image(8, 6);
  let out_rgb8 = resize(&rgb8, (4, 4), ResizeFilter::Nearest)
    .expect("Rgb8 source must resize via the per-pixel fallible path");
  assert_eq!((out_rgb8.width(), out_rgb8.height()), (4, 4));

  // Luma8 source: exercises the grayscale-broadcast `get_pixel`
  // projection. A uniform-gray image must round-trip to the same gray
  // on every output channel (R == G == B) under Nearest.
  let mut luma = ::image::GrayImage::new(8, 6);
  for p in luma.pixels_mut() {
    *p = ::image::Luma([123]);
  }
  let luma_img = ::image::DynamicImage::ImageLuma8(luma);
  let out_luma = resize(&luma_img, (4, 4), ResizeFilter::Nearest)
    .expect("Luma8 source must resize via the per-pixel fallible path");
  assert_eq!((out_luma.width(), out_luma.height()), (4, 4));
  // image_to_array drops alpha and broadcasts luma → RGB; every value
  // must be the original gray level (uniform image, Nearest filter).
  let mut arr = image_to_array(&out_luma, ColorOrder::Rgb).unwrap();
  let v: Vec<f32> = arr.to_vec().unwrap();
  assert_eq!(arr.shape(), vec![4, 4, 3], "[H, W, 3]");
  assert!(
    v.iter().all(|&x| close(x, 123.0)),
    "uniform Luma8 must broadcast to {{123, 123, 123}} after resize; got {v:?}"
  );

  // Rgba8 source: exercises the borrowed `as_rgba8()` fast path. A
  // uniform color must survive Nearest resize on all three RGB channels.
  let mut rgba = ::image::RgbaImage::new(8, 6);
  for p in rgba.pixels_mut() {
    *p = ::image::Rgba([10, 20, 30, 255]);
  }
  let rgba_img = ::image::DynamicImage::ImageRgba8(rgba);
  let out_rgba = resize(&rgba_img, (4, 4), ResizeFilter::Nearest)
    .expect("Rgba8 source must resize via the borrowed fast path");
  let mut arr2 = image_to_array(&out_rgba, ColorOrder::Rgb).unwrap();
  let v2: Vec<f32> = arr2.to_vec().unwrap();
  // Channel-last [4, 4, 3] uniform (10, 20, 30).
  for px in v2.chunks_exact(3) {
    assert!(
      close(px[0], 10.0) && close(px[1], 20.0) && close(px[2], 30.0),
      "uniform Rgba8 must survive resize as (10, 20, 30); got {px:?}"
    );
  }
}

// ---------- resize: PIL byte-exact reference values ----------
//
// The own resize kernel (`mlxrs::vlm::resize`) replaced the third-party
// `fast_image_resize` dependency and is **bit-exact with PIL
// `Image.resize`** — the reference mlx-vlm preprocessing targets (the
// swift `MediaProcessing.resampleBicubic` mirrors PIL). The expected
// arrays below were produced with **Pillow 12.2** over the IDENTICAL
// `synthetic_image` fixture (`pixel(x,y) = Rgb([(y*10)%256, (x*10)%256,
// 100])`, alpha 255 after RGBA projection), via PIL's exact fixed-point
// `precompute_coeffs` + `clip8` path:
//   - coords:   center = (out + 0.5) * scale
//   - support:  filter_support * max(scale, 1.0)   (downscale AA stretch)
//   - accum:    i32, seeded with 1<<(PRECISION_BITS-1), >> PRECISION_BITS
//               (PRECISION_BITS = 22), clamped [0,255]
// Because the scalar path reproduces PIL's integer math exactly, these
// are EQUALITY assertions (no ±1 LSB tolerance). On aarch64 the NEON
// kernel uses the same i32 math and is asserted bit-identical to scalar
// by the in-module differential test.

/// `synthetic_image` projected to a raw RGBA8 byte vector (row-major,
/// matching the resize output layout) for byte-exact comparison.
fn resize_rgba_raw(img: &::image::DynamicImage, h: u32, w: u32, f: ResizeFilter) -> Vec<u8> {
  resize(img, (h, w), f).unwrap().to_rgba8().into_raw()
}

#[test]
fn resize_pil_reference_downscale_8x6_to_4x3() {
  // 8x6 source -> 4x3 (height x width = 3 x 4). Hand-verified against
  // Pillow 12.2 (`im.resize((4, 3), Image.<FILTER>)`). Downscale, so the
  // antialiasing filter-stretch (`filterscale = scale > 1`) is exercised.
  let img = synthetic_image(8, 6);

  // Nearest: PIL maps out o -> min(floor((o+0.5)*in/out), in-1).
  assert_eq!(
    resize_rgba_raw(&img, 3, 4, ResizeFilter::Nearest),
    vec![
      10, 10, 100, 255, 10, 30, 100, 255, 10, 50, 100, 255, 10, 70, 100, 255, 30, 10, 100, 255, 30,
      30, 100, 255, 30, 50, 100, 255, 30, 70, 100, 255, 50, 10, 100, 255, 50, 30, 100, 255, 50, 50,
      100, 255, 50, 70, 100, 255,
    ],
    "Nearest 8x6->4x3 must match PIL Image.NEAREST byte-for-byte"
  );
  // Bilinear (triangle, support 1.0).
  assert_eq!(
    resize_rgba_raw(&img, 3, 4, ResizeFilter::Bilinear),
    vec![
      7, 7, 100, 255, 7, 25, 100, 255, 7, 45, 100, 255, 7, 63, 100, 255, 25, 7, 100, 255, 25, 25,
      100, 255, 25, 45, 100, 255, 25, 63, 100, 255, 43, 7, 100, 255, 43, 25, 100, 255, 43, 45, 100,
      255, 43, 63, 100, 255,
    ],
    "Bilinear 8x6->4x3 must match PIL Image.BILINEAR byte-for-byte"
  );
  // Bicubic (Keys cubic a=-0.5, support 2.0).
  assert_eq!(
    resize_rgba_raw(&img, 3, 4, ResizeFilter::Bicubic),
    vec![
      5, 5, 100, 255, 5, 25, 100, 255, 5, 45, 100, 255, 5, 65, 100, 255, 25, 5, 100, 255, 25, 25,
      100, 255, 25, 45, 100, 255, 25, 65, 100, 255, 45, 5, 100, 255, 45, 25, 100, 255, 45, 45, 100,
      255, 45, 65, 100, 255,
    ],
    "Bicubic 8x6->4x3 must match PIL Image.BICUBIC byte-for-byte"
  );
  // Lanczos (a=3, support 3.0).
  assert_eq!(
    resize_rgba_raw(&img, 3, 4, ResizeFilter::Lanczos3),
    vec![
      5, 5, 100, 255, 5, 24, 100, 255, 5, 46, 100, 255, 5, 65, 100, 255, 25, 5, 100, 255, 25, 24,
      100, 255, 25, 46, 100, 255, 25, 65, 100, 255, 45, 5, 100, 255, 45, 24, 100, 255, 45, 46, 100,
      255, 45, 65, 100, 255,
    ],
    "Lanczos3 8x6->4x3 must match PIL Image.LANCZOS byte-for-byte"
  );
}

#[test]
fn resize_pil_reference_upscale_4x4_to_8x8() {
  // 4x4 -> 8x8 upscale (filterscale clamps to 1.0 — no AA stretch).
  // Hand-verified against Pillow 12.2.
  let img = synthetic_image(4, 4);

  assert_eq!(
    resize_rgba_raw(&img, 8, 8, ResizeFilter::Bilinear),
    vec![
      0, 0, 100, 255, 0, 3, 100, 255, 0, 8, 100, 255, 0, 13, 100, 255, 0, 18, 100, 255, 0, 23, 100,
      255, 0, 28, 100, 255, 0, 30, 100, 255, 3, 0, 100, 255, 3, 3, 100, 255, 3, 8, 100, 255, 3, 13,
      100, 255, 3, 18, 100, 255, 3, 23, 100, 255, 3, 28, 100, 255, 3, 30, 100, 255, 8, 0, 100, 255,
      8, 3, 100, 255, 8, 8, 100, 255, 8, 13, 100, 255, 8, 18, 100, 255, 8, 23, 100, 255, 8, 28,
      100, 255, 8, 30, 100, 255, 13, 0, 100, 255, 13, 3, 100, 255, 13, 8, 100, 255, 13, 13, 100,
      255, 13, 18, 100, 255, 13, 23, 100, 255, 13, 28, 100, 255, 13, 30, 100, 255, 18, 0, 100, 255,
      18, 3, 100, 255, 18, 8, 100, 255, 18, 13, 100, 255, 18, 18, 100, 255, 18, 23, 100, 255, 18,
      28, 100, 255, 18, 30, 100, 255, 23, 0, 100, 255, 23, 3, 100, 255, 23, 8, 100, 255, 23, 13,
      100, 255, 23, 18, 100, 255, 23, 23, 100, 255, 23, 28, 100, 255, 23, 30, 100, 255, 28, 0, 100,
      255, 28, 3, 100, 255, 28, 8, 100, 255, 28, 13, 100, 255, 28, 18, 100, 255, 28, 23, 100, 255,
      28, 28, 100, 255, 28, 30, 100, 255, 30, 0, 100, 255, 30, 3, 100, 255, 30, 8, 100, 255, 30,
      13, 100, 255, 30, 18, 100, 255, 30, 23, 100, 255, 30, 28, 100, 255, 30, 30, 100, 255,
    ],
    "Bilinear 4x4->8x8 must match PIL Image.BILINEAR byte-for-byte"
  );
  // Bicubic spot-check a few representative pixels (the full 256-byte
  // array is verified by the differential + the bilinear full array
  // above proves the layout; bicubic upscale overshoots slightly at the
  // edges, captured here).
  let bic = resize_rgba_raw(&img, 8, 8, ResizeFilter::Bicubic);
  // (0,0) -> 0; (0,7) green channel = 31; (7,7) R=31,G=31.
  assert_eq!(&bic[0..4], &[0, 0, 100, 255], "Bicubic (0,0)");
  assert_eq!(&bic[7 * 4..7 * 4 + 4], &[0, 31, 100, 255], "Bicubic (0,7)");
  assert_eq!(
    &bic[(7 * 8 + 7) * 4..(7 * 8 + 7) * 4 + 4],
    &[31, 31, 100, 255],
    "Bicubic (7,7)"
  );
}

#[test]
fn resize_downscale_antialiasing_widens_filter_support() {
  // A hard left/right edge (left R=0, right R=200) downscaled 4->2 in
  // width. With proper antialiasing the filter support widens by the
  // scale factor (filterscale = 2.0), so the output is NOT a sharp
  // 0/200 split (that would be nearest) but a BLENDED pair — PIL gives
  // bilinear [29, 171] and bicubic [17, 183] in the R channel.
  // (Verified against Pillow 12.2.) This is the regression that proves
  // the `filterscale = max(scale, 1.0)` AA stretch is implemented.
  let mut rgba = ::image::RgbaImage::new(4, 4);
  for y in 0..4 {
    for x in 0..4 {
      let r = if x >= 2 { 200 } else { 0 };
      rgba.put_pixel(x, y, ::image::Rgba([r, 0, 0, 255]));
    }
  }
  let img = ::image::DynamicImage::ImageRgba8(rgba);

  let bil = resize(&img, (2, 2), ResizeFilter::Bilinear)
    .unwrap()
    .to_rgba8();
  let r_bil: Vec<u8> = bil.pixels().map(|p| p.0[0]).collect();
  assert_eq!(
    r_bil,
    vec![29, 171, 29, 171],
    "bilinear 4x4->2x2 AA: edge must blend to [29,171] (PIL), not a sharp [0,200] split"
  );

  let bic = resize(&img, (2, 2), ResizeFilter::Bicubic)
    .unwrap()
    .to_rgba8();
  let r_bic: Vec<u8> = bic.pixels().map(|p| p.0[0]).collect();
  assert_eq!(
    r_bic,
    vec![17, 183, 17, 183],
    "bicubic 4x4->2x2 AA: edge must blend to [17,183] (PIL)"
  );

  // Contrast: NEAREST does NOT antialias — the edge stays sharp.
  let nn = resize(&img, (2, 2), ResizeFilter::Nearest)
    .unwrap()
    .to_rgba8();
  let r_nn: Vec<u8> = nn.pixels().map(|p| p.0[0]).collect();
  assert_eq!(
    r_nn,
    vec![0, 200, 0, 200],
    "nearest 4x4->2x2: edge stays sharp"
  );
}

#[test]
fn resize_downscale_4x_to_2x_averages_with_widened_support() {
  // A 4x4 image of four solid 2x2 quadrants, downscaled 4->2. With the
  // antialiasing filter-stretch (filterscale = scale = 2.0) the support
  // window for each output pixel is wider than a single quadrant, so the
  // quadrants BLEND across their shared edges — proving the support
  // genuinely widens on downscale (a naive same-size kernel would
  // reproduce [10,20,30,40] sharply; the AA kernel does not). The exact
  // blended values are PIL's (Pillow 12.2): bilinear [14,22,28,36],
  // bicubic [13,21,29,37], lanczos [12,20,30,38].
  let mut rgba = ::image::RgbaImage::new(4, 4);
  // quadrants: TL=10, TR=20, BL=30, BR=40 (R channel; G=B=0, A=255).
  for y in 0..4 {
    for x in 0..4 {
      let r = match (x < 2, y < 2) {
        (true, true) => 10,
        (false, true) => 20,
        (true, false) => 30,
        (false, false) => 40,
      };
      rgba.put_pixel(x, y, ::image::Rgba([r, 0, 0, 255]));
    }
  }
  let img = ::image::DynamicImage::ImageRgba8(rgba);
  for (f, expected) in [
    (ResizeFilter::Bilinear, [14u8, 22, 28, 36]),
    (ResizeFilter::Bicubic, [13, 21, 29, 37]),
    (ResizeFilter::Lanczos3, [12, 20, 30, 38]),
  ] {
    let out = resize(&img, (2, 2), f).unwrap().to_rgba8();
    let r: Vec<u8> = out.pixels().map(|p| p.0[0]).collect();
    assert_eq!(
      r,
      expected.to_vec(),
      "downscale 4x4->2x2 of 2x2 quadrants must blend per PIL (filter {f:?})"
    );
  }

  // Averaging-correctness: a UNIFORM image downscales to itself exactly
  // (the kernel sums to 1.0, so a flat field is reproduced regardless of
  // how wide the support stretches).
  let mut uni = ::image::RgbaImage::new(4, 4);
  for p in uni.pixels_mut() {
    *p = ::image::Rgba([77, 0, 0, 255]);
  }
  let uni_img = ::image::DynamicImage::ImageRgba8(uni);
  for f in [
    ResizeFilter::Bilinear,
    ResizeFilter::Bicubic,
    ResizeFilter::Lanczos3,
  ] {
    let out = resize(&uni_img, (2, 2), f).unwrap().to_rgba8();
    let r: Vec<u8> = out.pixels().map(|p| p.0[0]).collect();
    assert_eq!(
      r,
      vec![77, 77, 77, 77],
      "uniform downscale must reproduce the constant ({f:?})"
    );
  }
}

#[test]
fn resize_pil_reference_width_straddle_5x1_to_2x1() {
  // 5x1 -> 2x1: a width that straddles the 4-channel vector boundary and
  // forces a per-axis-only (height = 1) convolution. R is a 0..200 ramp,
  // G the reverse, B constant 10. Verified against Pillow 12.2.
  let mut rgba = ::image::RgbaImage::new(5, 1);
  let rs = [0u8, 50, 100, 150, 200];
  let gs = [200u8, 150, 100, 50, 0];
  for x in 0..5 {
    rgba.put_pixel(
      x,
      0,
      ::image::Rgba([rs[x as usize], gs[x as usize], 10, 255]),
    );
  }
  let img = ::image::DynamicImage::ImageRgba8(rgba);
  assert_eq!(
    resize(&img, (1, 2), ResizeFilter::Bilinear)
      .unwrap()
      .to_rgba8()
      .into_raw(),
    vec![50, 150, 10, 255, 150, 50, 10, 255],
    "Bilinear 5x1->2x1 must match PIL byte-for-byte"
  );
  assert_eq!(
    resize(&img, (1, 2), ResizeFilter::Bicubic)
      .unwrap()
      .to_rgba8()
      .into_raw(),
    vec![43, 157, 10, 255, 157, 43, 10, 255],
    "Bicubic 5x1->2x1 must match PIL byte-for-byte"
  );
}

// ---------- resize: premultiplied-alpha (PIL parity) ----------
//
// PIL's `Image.resize` converts RGBA -> premultiplied `RGBa` before any
// non-NEAREST resample and back after (`Image.py`:
// `if self.mode in ["LA","RGBA"] and resample != NEAREST`). A
// straight-channel convolution is NOT byte-exact for non-opaque alpha: it
// bleeds the colour of fully-transparent pixels into their neighbours.
// The expected values below are computed from PIL's exact integer
// pipeline (`MULDIV255` premultiply -> `Resample.c` fixed-point
// convolution -> `rgba2rgbA` `CLIP8(255*c/a)` unpremultiply) and were
// cross-checked by hand-tracing the trivially separable 2x1->1x1 cases.

/// Build a `DynamicImage::ImageRgba8` from a row-major RGBA8 byte slice.
fn rgba8_image(w: u32, h: u32, bytes: &[u8]) -> ::image::DynamicImage {
  assert_eq!(bytes.len(), (w * h * 4) as usize, "rgba8_image: byte count");
  ::image::DynamicImage::ImageRgba8(
    ::image::ImageBuffer::from_raw(w, h, bytes.to_vec()).expect("rgba8_image: from_raw"),
  )
}

#[test]
fn resize_premultiplied_alpha_transparent_red_opaque_blue_downscale() {
  // The Codex example: a transparent-red pixel `(255,0,0,0)` next to an
  // opaque-blue pixel `(0,0,255,255)`. PIL premultiplies before
  // resampling, so the transparent red contributes ZERO colour — the
  // downscaled pixel is pure blue with half alpha `(0,0,255,128)`. A
  // straight-channel average would (wrongly) yield purple
  // `(128,0,128,128)`. Hand-traced + PIL-pipeline-verified.
  let transparent_red = [255u8, 0, 0, 0];
  let opaque_blue = [0u8, 0, 255, 255];

  // 2x1 -> 1x1: pure horizontal, the simplest case to hand-verify.
  let src_2x1: Vec<u8> = [transparent_red, opaque_blue].concat();
  let img_2x1 = rgba8_image(2, 1, &src_2x1);
  for f in [
    ResizeFilter::Bilinear,
    ResizeFilter::Bicubic,
    ResizeFilter::Lanczos3,
  ] {
    let out = resize(&img_2x1, (1, 1), f).unwrap().to_rgba8().into_raw();
    assert_eq!(
      out,
      vec![0, 0, 255, 128],
      "{f:?} 2x1->1x1: PIL premultiplied alpha must give pure blue \
       (0,0,255,128), not straight-channel purple (128,0,128,128)"
    );
  }

  // 2x2 -> 1x1 (both axes): columns transparent-red / opaque-blue.
  let src_2x2: Vec<u8> = [transparent_red, opaque_blue, transparent_red, opaque_blue].concat();
  let img_2x2 = rgba8_image(2, 2, &src_2x2);
  for f in [
    ResizeFilter::Bilinear,
    ResizeFilter::Bicubic,
    ResizeFilter::Lanczos3,
  ] {
    let out = resize(&img_2x2, (1, 1), f).unwrap().to_rgba8().into_raw();
    assert_eq!(
      out,
      vec![0, 0, 255, 128],
      "{f:?} 2x2->1x1: premultiplied-alpha downscale must give (0,0,255,128)"
    );
  }
}

#[test]
fn resize_premultiplied_alpha_partial_alpha_both_sides() {
  // Both pixels non-opaque: red at A=128 `(255,0,0,128)` + opaque blue
  // `(0,0,255,255)`, 2x1 -> 1x1. PIL: premultiply red -> (128,0,0,128),
  // blue -> (0,0,255,255); average -> (64,0,128,192); unpremultiply
  // `CLIP8(255*c/192)` -> (85,0,170,192). Hand-traced.
  let src: Vec<u8> = [[255u8, 0, 0, 128], [0u8, 0, 255, 255]].concat();
  let img = rgba8_image(2, 1, &src);
  for f in [
    ResizeFilter::Bilinear,
    ResizeFilter::Bicubic,
    ResizeFilter::Lanczos3,
  ] {
    let out = resize(&img, (1, 1), f).unwrap().to_rgba8().into_raw();
    assert_eq!(
      out,
      vec![85, 0, 170, 192],
      "{f:?} partial-alpha 2x1->1x1 must match PIL premultiplied result"
    );
  }
}

#[test]
fn resize_premultiplied_alpha_upscale() {
  // Upscale 2x2 -> 4x4 (columns transparent-red / opaque-blue). The
  // premultiplied path keeps the colour pure blue at every output column
  // — only the alpha ramps — instead of bleeding red into the
  // partially-transparent interpolated columns. Per-row layout is
  // identical (the source is column-constant). PIL-pipeline-verified.
  let transparent_red = [255u8, 0, 0, 0];
  let opaque_blue = [0u8, 0, 255, 255];
  let src: Vec<u8> = [transparent_red, opaque_blue, transparent_red, opaque_blue].concat();
  let img = rgba8_image(2, 2, &src);

  // Expected single row (4 RGBA pixels) per filter; all 4 output rows are
  // identical because the source is constant down each column.
  for (f, row) in [
    (
      ResizeFilter::Bilinear,
      [0u8, 0, 0, 0, 0, 0, 255, 64, 0, 0, 255, 191, 0, 0, 255, 255],
    ),
    (
      ResizeFilter::Bicubic,
      [0u8, 0, 0, 0, 0, 0, 255, 53, 0, 0, 255, 202, 0, 0, 255, 255],
    ),
    (
      ResizeFilter::Lanczos3,
      [0u8, 0, 0, 0, 0, 0, 255, 59, 0, 0, 255, 196, 0, 0, 255, 255],
    ),
  ] {
    let out = resize(&img, (4, 4), f).unwrap().to_rgba8().into_raw();
    let mut expected = Vec::with_capacity(64);
    for _ in 0..4 {
      expected.extend_from_slice(&row);
    }
    assert_eq!(
      out, expected,
      "{f:?} 2x2->4x4 premultiplied upscale: colour stays pure blue, \
       only alpha ramps (no red bleed into transparent columns)"
    );
    // Every interpolated column must have R == 0 (no transparent-red
    // colour bleed) — the core premultiplied-alpha guarantee.
    for px in out.chunks_exact(4) {
      assert_eq!(
        px[0], 0,
        "{f:?} upscale: no red bleed — R must be 0 in every output pixel"
      );
    }
  }
}

#[test]
fn resize_nearest_does_not_premultiply() {
  // PIL exempts NEAREST from premultiplication (it is a pure pixel
  // gather). The transparent-red pixel must survive a NEAREST resize with
  // its straight `(255,0,0,0)` channels intact — premultiplying it would
  // (wrongly) zero the R channel.
  let transparent_red = [255u8, 0, 0, 0];
  let opaque_blue = [0u8, 0, 255, 255];
  let src: Vec<u8> = [transparent_red, opaque_blue, transparent_red, opaque_blue].concat();
  let img = rgba8_image(2, 2, &src);

  // 2x2 -> 2x2 NEAREST is identity: every pixel gathered verbatim. The
  // transparent-red R channel stays 255 (it would be 0 if premultiplied).
  let out = resize(&img, (2, 2), ResizeFilter::Nearest)
    .unwrap()
    .to_rgba8()
    .into_raw();
  assert_eq!(
    out, src,
    "NEAREST must NOT premultiply: transparent-red keeps straight \
     channels (255,0,0,0), not the premultiplied (0,0,0,0)"
  );
}

#[test]
fn resize_opaque_alpha_unaffected_by_premultiply() {
  // Premultiply/unpremultiply are the identity for A == 255
  // (`MULDIV255(c,255) == c`; unpremultiply special-cases alpha 255), so
  // a fully-opaque image resizes bit-identically to a straight-channel
  // resize — the existing alpha=255 PIL-reference tests are unaffected.
  // Re-assert here on a non-trivial opaque RGBA gradient.
  let mut src = Vec::with_capacity(4 * 4 * 4);
  for y in 0..4u32 {
    for x in 0..4u32 {
      src.extend_from_slice(&[(x * 60) as u8, (y * 60) as u8, 100, 255]);
    }
  }
  let img = rgba8_image(4, 4, &src);
  for f in [
    ResizeFilter::Bilinear,
    ResizeFilter::Bicubic,
    ResizeFilter::Lanczos3,
  ] {
    let out = resize(&img, (2, 3), f).unwrap().to_rgba8().into_raw();
    // Alpha stays a solid 255 everywhere (premultiply identity on opaque).
    for px in out.chunks_exact(4) {
      assert_eq!(
        px[3], 255,
        "{f:?} opaque resize: alpha must remain 255 (premultiply identity)"
      );
    }
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
  let out = resize_lanczos(&img, 16, 32).unwrap();
  assert_eq!(out.width(), 32);
  assert_eq!(out.height(), 16);
}

#[test]
fn resize_lanczos_equivalent_to_resize_with_lanczos3_filter() {
  // resize_lanczos is documented as a thin wrapper around
  // resize(..., Lanczos3) — byte-for-byte output equality is the
  // strongest assertion of that contract.
  let img = synthetic_image(12, 10);
  let a = resize_lanczos(&img, 8, 16).unwrap();
  let b = resize(&img, (8, 16), ResizeFilter::Lanczos3).unwrap();
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
  let out = resize_lanczos(&img, 4, 4).unwrap();
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

// ─── P8 / VLM-1 (#120) trailing-layout post-step tests ───────────────

/// `Layout::Hwc` is the identity arm — `preprocess` and `apply_layout`
/// must produce the historical `[H, W, 3]` shape for source
/// compatibility (every pre-existing caller stays on the default).
#[test]
fn apply_layout_hwc_is_identity() {
  // Hand-built [2, 3, 3] array (H=2, W=3, C=3) — the same shape
  // [`preprocess`] emits before the layout post-step. `Hwc` must
  // return it unchanged (shape + values bit-identical).
  let n = 2 * 3 * 3;
  let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
  let arr = Array::from_slice(&src, &(2usize, 3, 3)).unwrap();
  let mut out = apply_layout(arr, Layout::Hwc).unwrap();
  assert_eq!(out.shape(), vec![2, 3, 3], "Hwc must preserve [H, W, 3]");
  let v: Vec<f32> = out.to_vec().unwrap();
  assert_eq!(v, src, "Hwc identity must preserve values byte-identical");
}

/// `Layout::Chw` permutes `[H, W, 3]` → `[3, H, W]`. Verifies the
/// shape swap AND the per-channel permutation by hand-tracing the
/// channel-last → planar reordering.
#[test]
fn apply_layout_chw_transposes_to_planar() {
  // Build a [2, 3, 3] array where pixel (y, x) carries
  // (100*y + 10*x, 100*y + 10*x + 1, 100*y + 10*x + 2) — each
  // channel uniquely traceable.
  let h = 2;
  let w = 3;
  let mut src = Vec::with_capacity(h * w * 3);
  for y in 0..h {
    for x in 0..w {
      let base = (100 * y + 10 * x) as f32;
      src.push(base);
      src.push(base + 1.0);
      src.push(base + 2.0);
    }
  }
  let arr = Array::from_slice(&src, &(h, w, 3usize)).unwrap();
  let out = apply_layout(arr, Layout::Chw).unwrap();
  // Shape: [H, W, 3] → [3, H, W].
  assert_eq!(out.shape(), vec![3, h, w], "Chw must produce [3, H, W]");
  // `transpose_axes` produces a non-contiguous strided view; materialize
  // via `contiguous` before `to_vec` (matches every other transpose
  // test in the suite).
  let mut materialized = contiguous(&out, false).unwrap();
  let v: Vec<f32> = materialized.to_vec().unwrap();
  // Channel c at (y, x) lives at planar offset c*H*W + y*W + x.
  for c in 0..3 {
    for y in 0..h {
      for x in 0..w {
        let planar_off = c * h * w + y * w + x;
        let expected = (100 * y + 10 * x) as f32 + c as f32;
        assert_eq!(
          v[planar_off], expected,
          "Chw: planar (c={c}, y={y}, x={x}) must equal channel-last source"
        );
      }
    }
  }
}

/// `Layout::Bchw` adds the leading batch axis swift's
/// `MediaProcessing.asMLXArray` produces — `[1, 3, H, W]`. Verifies
/// shape AND that the planar values match the `Chw` arm (just with a
/// leading unit dim).
#[test]
fn apply_layout_bchw_matches_swift_planar_batched() {
  // Same hand-built [2, 3, 3] as the Chw test.
  let h = 2;
  let w = 3;
  let mut src = Vec::with_capacity(h * w * 3);
  for y in 0..h {
    for x in 0..w {
      let base = (100 * y + 10 * x) as f32;
      src.push(base);
      src.push(base + 1.0);
      src.push(base + 2.0);
    }
  }
  let arr = Array::from_slice(&src, &(h, w, 3usize)).unwrap();
  let out = apply_layout(arr, Layout::Bchw).unwrap();
  // Shape: [H, W, 3] → [3, H, W] → [1, 3, H, W]; matches swift
  // `array.reshape((1, h, w, 3)).transposed(0, 3, 1, 2)`
  // (`MLXVLM/MediaProcessing.swift:190`).
  assert_eq!(
    out.shape(),
    vec![1, 3, h, w],
    "Bchw must produce [1, 3, H, W] (swift MediaProcessing.asMLXArray shape)"
  );
  // Materialize the strided transpose view before `to_vec`.
  let mut materialized = contiguous(&out, false).unwrap();
  let v: Vec<f32> = materialized.to_vec().unwrap();
  // Element layout is identical to Chw (the leading unit dim is
  // metadata-only); the per-element check from the Chw test applies
  // verbatim.
  for c in 0..3 {
    for y in 0..h {
      for x in 0..w {
        let planar_off = c * h * w + y * w + x;
        let expected = (100 * y + 10 * x) as f32 + c as f32;
        assert_eq!(v[planar_off], expected, "Bchw c={c} y={y} x={x}");
      }
    }
  }
}

/// `apply_layout` rejects non-rank-3 inputs (the post-step targets the
/// cross-model `[H, W, 3]` shape specifically — per-model processors
/// with patchified `[N, P, P, 3]` or batched `[B, H, W, 3]` compose
/// their own trailing layout).
#[test]
fn apply_layout_rejects_non_rank3_input() {
  let arr = Array::from_slice(&[0.0_f32; 12], &(2usize, 6)).unwrap();
  let err = apply_layout(arr, Layout::Chw).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "want ShapeMismatch on non-rank-3 input, got {err:?}"
  );
}

/// `apply_layout` rejects rank-3 inputs whose trailing channel dim is
/// not 3 — the layout enum is RGB-specific (the swift / python references
/// both assume a 3-channel input at this stage).
#[test]
fn apply_layout_rejects_non_3channel_trailing() {
  let arr = Array::from_slice(&[0.0_f32; 16], &(2usize, 2, 4)).unwrap();
  let err = apply_layout(arr, Layout::Chw).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "want ShapeMismatch on trailing != 3, got {err:?}"
  );
}

/// `preprocess` with default config (Layout::Hwc) keeps the historical
/// `[H, W, 3]` shape — source-compatibility guarantee for every
/// pre-existing caller of [`preprocess`].
#[test]
fn preprocess_default_layout_is_hwc_for_source_compat() {
  // Default config = Layout::Hwc; output must be [H, W, 3] for a
  // 4x4 input resized to 8x8 (the per-channel cfg.size = (224, 224)
  // default is irrelevant here — we override).
  let img = synthetic_image(4, 4);
  let cfg = ImageProcessorConfig {
    size: (8, 8),
    ..ImageProcessorConfig::default()
  };
  let out = preprocess(&img, &cfg).unwrap();
  assert_eq!(
    out.shape(),
    vec![8, 8, 3],
    "default Layout::Hwc must keep [H, W, 3] output"
  );
}

/// `preprocess` with `Layout::Bchw` produces swift's
/// `[1, 3, H, W]` — the breaking-change opt-in that VLM-1 (#120)
/// surfaces. The H*W*C product is unchanged (both layouts hold the
/// same number of f32s) so the post-step is shape-only.
#[test]
fn preprocess_layout_bchw_emits_batched_planar() {
  let img = synthetic_image(4, 4);
  let cfg = ImageProcessorConfig {
    size: (8, 8),
    layout: Layout::Bchw,
    ..ImageProcessorConfig::default()
  };
  let out = preprocess(&img, &cfg).unwrap();
  assert_eq!(
    out.shape(),
    vec![1, 3, 8, 8],
    "Layout::Bchw must emit [1, 3, H, W] (swift MediaProcessing.asMLXArray shape)"
  );
}

/// `preprocess` with `Layout::Chw` produces `[3, H, W]` (planar, no
/// batch axis) — the torchvision / timm classical-CV layout.
#[test]
fn preprocess_layout_chw_emits_planar_no_batch() {
  let img = synthetic_image(4, 4);
  let cfg = ImageProcessorConfig {
    size: (8, 8),
    layout: Layout::Chw,
    ..ImageProcessorConfig::default()
  };
  let out = preprocess(&img, &cfg).unwrap();
  assert_eq!(
    out.shape(),
    vec![3, 8, 8],
    "Layout::Chw must emit [3, H, W] (torchvision planar)"
  );
}

/// `ImageProcessorConfig::default().layout` is `Hwc` — pre-existing
/// callers see the historical channel-last output as their default.
#[test]
fn imageprocessor_config_default_layout_is_hwc() {
  let cfg = ImageProcessorConfig::default();
  assert_eq!(
    cfg.layout,
    Layout::Hwc,
    "default layout must be Hwc for source-compat"
  );
}

// ─── P8 / VLM-3 (#121) non-Rgb8 bulk-fill regression ─────────────────

/// The non-`Rgb8` branch of [`image_to_array`] used to per-pixel
/// `buf.push(f32::from(...))` three times per pixel; the bulk-fill
/// upgrade now builds a `Vec<u8>` then hands it to the same C3
/// (`rgb_widen`) / C4 (`bgr_widen`) SIMD dispatcher the `Rgb8` fast
/// path uses. This regression check verifies the non-`Rgb8` output is
/// byte-identical to the `Rgb8` fast path for the SAME pixel values —
/// proving the bulk-fill path produces the same f32 buffer as the
/// per-pixel-push shape it replaced.
#[test]
fn image_to_array_non_rgb8_bulk_fill_matches_rgb8_fast_path() {
  // Build a 3x4 Rgba8 source (the non-Rgb8 branch's hot path —
  // covers BOTH the dynamic_image_rgb_pixel alpha-drop projection
  // AND the new bulk-fill `Vec<u8>` intermediate). Pixel values are
  // identical to the Rgb8 sibling — the alpha channel must be
  // dropped by the projection.
  let h = 3;
  let w = 4;
  let mut rgba = ::image::RgbaImage::new(w, h);
  let mut rgb = ::image::RgbImage::new(w, h);
  for y in 0..h {
    for x in 0..w {
      let r = (10 * x) as u8;
      let g = (10 * y) as u8;
      let b = 200u8;
      rgba.put_pixel(x, y, ::image::Rgba([r, g, b, 128]));
      rgb.put_pixel(x, y, ::image::Rgb([r, g, b]));
    }
  }
  let img_rgba = ::image::DynamicImage::ImageRgba8(rgba);
  let img_rgb = ::image::DynamicImage::ImageRgb8(rgb);
  // RGB color order — the bulk-fill upgrade routes through `rgb_widen`.
  let mut out_rgba = image_to_array(&img_rgba, ColorOrder::Rgb).unwrap();
  let mut out_rgb = image_to_array(&img_rgb, ColorOrder::Rgb).unwrap();
  let v_rgba: Vec<f32> = out_rgba.to_vec().unwrap();
  let v_rgb: Vec<f32> = out_rgb.to_vec().unwrap();
  assert_eq!(
    v_rgba, v_rgb,
    "non-Rgb8 (Rgba8 via bulk-fill + rgb_widen) must produce the same f32 buffer as \
     the Rgb8 fast path for identical RGB pixel values (alpha dropped per \
     image_to_array's documented projection)"
  );
}

/// Same regression for `ColorOrder::Bgr` — the bulk-fill non-Rgb8
/// branch routes through `bgr_widen` and must produce the same f32
/// buffer as the Rgb8 fast path for identical RGB values.
#[test]
fn image_to_array_non_rgb8_bulk_fill_bgr_matches_rgb8_fast_path() {
  let h = 2;
  let w = 4;
  let mut rgba = ::image::RgbaImage::new(w, h);
  let mut rgb = ::image::RgbImage::new(w, h);
  for y in 0..h {
    for x in 0..w {
      let r = (10 * x + 1) as u8;
      let g = (10 * y + 3) as u8;
      let b = ((20 * x + 30 * y + 7) % 256) as u8;
      rgba.put_pixel(x, y, ::image::Rgba([r, g, b, 200]));
      rgb.put_pixel(x, y, ::image::Rgb([r, g, b]));
    }
  }
  let img_rgba = ::image::DynamicImage::ImageRgba8(rgba);
  let img_rgb = ::image::DynamicImage::ImageRgb8(rgb);
  let mut out_rgba = image_to_array(&img_rgba, ColorOrder::Bgr).unwrap();
  let mut out_rgb = image_to_array(&img_rgb, ColorOrder::Bgr).unwrap();
  let v_rgba: Vec<f32> = out_rgba.to_vec().unwrap();
  let v_rgb: Vec<f32> = out_rgb.to_vec().unwrap();
  assert_eq!(
    v_rgba, v_rgb,
    "non-Rgb8 BGR (Rgba8 via bulk-fill + bgr_widen) must match Rgb8 fast path"
  );
}

/// Non-`Rgb8` branch with `Luma8` (single-channel grayscale) source:
/// the `dynamic_image_rgb_pixel` helper broadcasts the luma to
/// (L, L, L). Verifies the bulk-fill path produces 3× the source
/// luma per pixel (no per-channel skew).
#[test]
fn image_to_array_luma8_broadcasts_to_rgb_via_bulk_fill() {
  let h: u32 = 2;
  let w: u32 = 3;
  let mut gray = ::image::GrayImage::new(w, h);
  for y in 0..h {
    for x in 0..w {
      gray.put_pixel(x, y, ::image::Luma([(10 * x + 50 * y) as u8]));
    }
  }
  let img = ::image::DynamicImage::ImageLuma8(gray);
  let mut out = image_to_array(&img, ColorOrder::Rgb).unwrap();
  assert_eq!(
    out.shape(),
    vec![h as usize, w as usize, 3],
    "Luma8 must broadcast to [H, W, 3]"
  );
  let v: Vec<f32> = out.to_vec().unwrap();
  // Per pixel (y, x): L = 10*x + 50*y; output triple must be (L, L, L).
  for y in 0..h {
    for x in 0..w {
      let l = (10 * x + 50 * y) as f32;
      let off = ((y * w + x) * 3) as usize;
      assert_eq!(
        v[off], l,
        "Luma8 R channel must equal luma at (y={y}, x={x})"
      );
      assert_eq!(v[off + 1], l, "G channel must equal luma");
      assert_eq!(v[off + 2], l, "B channel must equal luma");
    }
  }
}

// ─── P8 / VLM-2 (#125) byte-budget validation regression ─────────────

/// [`resize`] rejects an over-budget target via a typed
/// [`Error::ShapeMismatch`] BEFORE any allocation — closure regression
/// for VLM-2 (#125). Mirrors the audit-table guarantee: a hostile /
/// mis-configured `ImageProcessorConfig.size` cannot drive a
/// multi-GiB infallible alloc.
#[test]
fn resize_rejects_over_budget_target() {
  let img = synthetic_image(8, 8);
  // 64K × 64K × 4 = ~17 GiB — well over the 512 MiB cap.
  let err = resize(&img, (65_536, 65_536), ResizeFilter::Bilinear).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "want ShapeMismatch on over-budget target, got {err:?}"
  );
}

/// [`resize`] rejects a zero-dim target — the VLM-2 byte-budget guard
/// must reject zero/overflow targets before allocating.
#[test]
fn resize_rejects_zero_dim_target() {
  let img = synthetic_image(8, 8);
  let err = resize(&img, (0, 16), ResizeFilter::Bilinear).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "want ShapeMismatch on zero target, got {err:?}"
  );
}
