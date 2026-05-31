use super::*;
use ::image::{
  DynamicImage, GrayAlphaImage, GrayImage, ImageBuffer, Luma, LumaA, Rgb, RgbImage, Rgba,
  RgbaImage, metadata::Orientation,
};

/// Build a `width × height` Rgb8 image whose pixel values encode
/// `(x, y)` so any rotation/flip is checkable byte-for-byte against
/// the upstream `image::imageops` reference. Default test sizes use
/// 4x3 (non-square) so 90° rotates flip dimensions visibly.
fn xy_encoded_rgb8(width: u32, height: u32) -> DynamicImage {
  let mut buf = RgbImage::new(width, height);
  for y in 0..height {
    for x in 0..width {
      // (10*x, 10*y, 200) is unique-per-pixel for x,y < 25 and stays
      // well below 256, so rotate/flip orderings are visible at the
      // byte level.
      buf.put_pixel(x, y, Rgb([(x * 10) as u8, (y * 10) as u8, 200]));
    }
  }
  DynamicImage::ImageRgb8(buf)
}

/// Luma8 (single-channel u8) variant of [`xy_encoded_rgb8`].
/// Encodes the row-major pixel index so rotation invariants are
/// visible at the byte level — for a 4x3 image, source pixel
/// `(x, y)` carries value `y * 4 + x` (0..12).
fn xy_encoded_luma8(width: u32, height: u32) -> DynamicImage {
  let mut buf = GrayImage::new(width, height);
  for y in 0..height {
    for x in 0..width {
      buf.put_pixel(x, y, Luma([(y * width + x) as u8]));
    }
  }
  DynamicImage::ImageLuma8(buf)
}

/// LumaA8 (grayscale + alpha) variant: luma encodes `y*w + x`,
/// alpha encodes `255 - (y*w + x)` so each byte position is
/// individually traceable through the rotation.
fn xy_encoded_luma_alpha8(width: u32, height: u32) -> DynamicImage {
  let mut buf = GrayAlphaImage::new(width, height);
  for y in 0..height {
    for x in 0..width {
      let v = (y * width + x) as u8;
      buf.put_pixel(x, y, LumaA([v, 255 - v]));
    }
  }
  DynamicImage::ImageLumaA8(buf)
}

/// Rgba8 variant: each channel is independently traceable
/// (R = 10*x, G = 10*y, B = 200, A = 255 - 10*y) so the
/// 4-byte stride is visible end-to-end.
fn xy_encoded_rgba8(width: u32, height: u32) -> DynamicImage {
  let mut buf = RgbaImage::new(width, height);
  for y in 0..height {
    for x in 0..width {
      buf.put_pixel(
        x,
        y,
        Rgba([(x * 10) as u8, (y * 10) as u8, 200, 255 - (y * 10) as u8]),
      );
    }
  }
  DynamicImage::ImageRgba8(buf)
}

#[test]
fn no_transforms_passes_through_unchanged() {
  // Identity-orientation path: no rotation overhead, no clone. The
  // returned image's raw bytes must match the source exactly
  // (rules out an accidental rotate dispatch).
  let img = xy_encoded_rgb8(4, 3);
  let original_bytes = img.as_rgb8().expect("rgb8 source").as_raw().clone();
  let out = apply_orientation_fallible(img, Orientation::NoTransforms).expect("infallible path");
  assert_eq!(out.width(), 4);
  assert_eq!(out.height(), 3);
  assert_eq!(
    out.as_rgb8().expect("rgb8 output").as_raw(),
    &original_bytes
  );
}

#[test]
fn rotate180_in_place_path_matches_reference() {
  // Rotate180 goes through `rotate180_in_place` upstream (no source-
  // sized alloc). Verify the dispatcher routes it through the
  // no-allocation arm AND that the pixel transform matches
  // `image::imageops::rotate180`.
  let img = xy_encoded_rgb8(4, 3);
  let reference: ImageBuffer<Rgb<u8>, Vec<u8>> =
    ::image::imageops::rotate180(img.as_rgb8().expect("rgb8 source"));
  let out =
    apply_orientation_fallible(img, Orientation::Rotate180).expect("in-place path infallible");
  assert_eq!(out.width(), 4, "Rotate180 preserves width");
  assert_eq!(out.height(), 3, "Rotate180 preserves height");
  assert_eq!(
    out.as_rgb8().expect("rgb8 output").as_raw(),
    reference.as_raw(),
    "Rotate180 pixel bytes must match image::imageops::rotate180"
  );
}

#[test]
fn flip_horizontal_in_place_path_matches_reference() {
  // FlipHorizontal goes through `fliph_in_place` upstream — verify
  // pixel-level parity with `image::imageops::flip_horizontal`.
  let img = xy_encoded_rgb8(4, 3);
  let reference = ::image::imageops::flip_horizontal(img.as_rgb8().expect("rgb8 source"));
  let out =
    apply_orientation_fallible(img, Orientation::FlipHorizontal).expect("in-place path infallible");
  assert_eq!(
    out.as_rgb8().expect("rgb8 output").as_raw(),
    reference.as_raw(),
    "FlipHorizontal pixel bytes must match image::imageops::flip_horizontal"
  );
}

// -- Rgb8 manual rotation parity -------------------------------------

#[test]
fn rotate90_rgb8_manual_matches_reference() {
  // Rotate90 on Rgb8 now goes through the truly-fallible
  // manual `rotate_buf` path. Must:
  //   (1) succeed for a small input under `MAX_DECODED_IMAGE_BYTES`,
  //   (2) swap dimensions (4x3 → 3x4),
  //   (3) produce byte-identical pixels to `image::imageops::rotate90`.
  let img = xy_encoded_rgb8(4, 3);
  let reference = ::image::imageops::rotate90(img.as_rgb8().expect("rgb8 source"));
  let out = apply_orientation_fallible(img, Orientation::Rotate90)
    .expect("manual rotate90 should succeed for a 4x3 image well under the 512 MiB ceiling");
  assert_eq!(out.width(), 3, "Rotate90 swaps width <- height");
  assert_eq!(out.height(), 4, "Rotate90 swaps height <- width");
  assert_eq!(
    out.as_rgb8().expect("rgb8 output").as_raw(),
    reference.as_raw(),
    "manual rotate must yield byte-identical pixels to image::imageops::rotate90"
  );
}

#[test]
fn rotate270_rgb8_manual_matches_reference() {
  let img = xy_encoded_rgb8(4, 3);
  let reference = ::image::imageops::rotate270(img.as_rgb8().expect("rgb8 source"));
  let out = apply_orientation_fallible(img, Orientation::Rotate270).expect("rotate270 ok");
  assert_eq!(out.width(), 3);
  assert_eq!(out.height(), 4);
  assert_eq!(
    out.as_rgb8().expect("rgb8 output").as_raw(),
    reference.as_raw(),
  );
}

#[test]
fn rotate90_fliph_rgb8_composite_matches_reference() {
  // Rotate90FlipH = rotate90 + fliph (the composite collapses to
  // `dst.put_pixel(y, x, src.get_pixel(x, y))` in `rotate_buf`).
  // Verify the composite output is byte-identical to the
  // image-rs dispatch.
  let img = xy_encoded_rgb8(4, 3);
  let rotated = ::image::imageops::rotate90(img.as_rgb8().expect("rgb8 source"));
  let reference = ::image::imageops::flip_horizontal(&rotated);
  let out = apply_orientation_fallible(img, Orientation::Rotate90FlipH)
    .expect("Rotate90FlipH composite ok");
  assert_eq!(out.width(), reference.width());
  assert_eq!(out.height(), reference.height());
  assert_eq!(
    out.as_rgb8().expect("rgb8 output").as_raw(),
    reference.as_raw(),
  );
}

#[test]
fn rotate270_fliph_rgb8_composite_matches_reference() {
  let img = xy_encoded_rgb8(4, 3);
  let rotated = ::image::imageops::rotate270(img.as_rgb8().expect("rgb8 source"));
  let reference = ::image::imageops::flip_horizontal(&rotated);
  let out = apply_orientation_fallible(img, Orientation::Rotate270FlipH)
    .expect("Rotate270FlipH composite ok");
  assert_eq!(out.width(), 3);
  assert_eq!(out.height(), 4);
  assert_eq!(
    out.as_rgb8().expect("rgb8 output").as_raw(),
    reference.as_raw(),
  );
}

// -- Rgba8 manual rotation parity ------------------------------------

#[test]
fn rotate90_rgba8_manual_matches_reference() {
  // Rgba8 has 4 channels (R, G, B, A) — verify the per-pixel
  // 4-byte memcpy in `rotate_buf` preserves the alpha channel
  // alongside RGB.
  let img = xy_encoded_rgba8(4, 3);
  let reference = ::image::imageops::rotate90(img.as_rgba8().expect("rgba8 source"));
  let out = apply_orientation_fallible(img, Orientation::Rotate90).expect("rgba8 rotate90 ok");
  assert_eq!(out.width(), 3);
  assert_eq!(out.height(), 4);
  assert_eq!(
    out.as_rgba8().expect("rgba8 output").as_raw(),
    reference.as_raw(),
    "manual Rgba8 rotate90 must preserve all 4 channels byte-identical to image::imageops::rotate90"
  );
}

#[test]
fn rotate270_fliph_rgba8_composite_matches_reference() {
  // Composite path × alpha channel: verify the
  // collapsed `dst[h-1-y, w-1-x] = src[x, y]` index math handles
  // the 4-byte stride correctly.
  let img = xy_encoded_rgba8(4, 3);
  let rotated = ::image::imageops::rotate270(img.as_rgba8().expect("rgba8 source"));
  let reference = ::image::imageops::flip_horizontal(&rotated);
  let out =
    apply_orientation_fallible(img, Orientation::Rotate270FlipH).expect("rgba8 Rotate270FlipH ok");
  assert_eq!(out.width(), 3);
  assert_eq!(out.height(), 4);
  assert_eq!(
    out.as_rgba8().expect("rgba8 output").as_raw(),
    reference.as_raw(),
  );
}

// -- Luma8 manual rotation parity ------------------------------------

#[test]
fn rotate90_luma8_manual_matches_reference() {
  // Luma8 has 1 channel — verify the per-pixel 1-byte memcpy in
  // `rotate_buf` produces the same byte order as image-rs.
  let img = xy_encoded_luma8(4, 3);
  let reference = ::image::imageops::rotate90(img.as_luma8().expect("luma8 source"));
  let out = apply_orientation_fallible(img, Orientation::Rotate90).expect("luma8 rotate90 ok");
  assert_eq!(out.width(), 3);
  assert_eq!(out.height(), 4);
  assert_eq!(
    out.as_luma8().expect("luma8 output").as_raw(),
    reference.as_raw(),
    "manual Luma8 rotate90 must match image::imageops::rotate90 byte-identical"
  );
}

#[test]
fn rotate270_luma8_manual_matches_reference() {
  let img = xy_encoded_luma8(4, 3);
  let reference = ::image::imageops::rotate270(img.as_luma8().expect("luma8 source"));
  let out = apply_orientation_fallible(img, Orientation::Rotate270).expect("luma8 rotate270 ok");
  assert_eq!(out.width(), 3);
  assert_eq!(out.height(), 4);
  assert_eq!(
    out.as_luma8().expect("luma8 output").as_raw(),
    reference.as_raw(),
  );
}

// -- LumaA8 manual rotation parity -----------------------------------

#[test]
fn rotate90_luma_alpha8_manual_matches_reference() {
  // LumaA8 has 2 channels — verify the 2-byte stride.
  let img = xy_encoded_luma_alpha8(4, 3);
  let reference = ::image::imageops::rotate90(img.as_luma_alpha8().expect("luma_alpha8 source"));
  let out =
    apply_orientation_fallible(img, Orientation::Rotate90).expect("luma_alpha8 rotate90 ok");
  assert_eq!(out.width(), 3);
  assert_eq!(out.height(), 4);
  assert_eq!(
    out.as_luma_alpha8().expect("luma_alpha8 output").as_raw(),
    reference.as_raw(),
    "manual LumaA8 rotate90 must match image::imageops::rotate90 byte-identical"
  );
}

// -- 16-bit-PNG / float-pixel rotate parity ------------------
//
// image-rs 0.25's PNG decoder emits `Luma16`/`LumaA16`/`Rgb16`/
// `Rgba16` for `BitDepth::Sixteen` PNG inputs (`codecs/png.rs:64-71`),
// and caller-supplied `DynamicImage::ImageRgb32F`/`ImageRgba32F`
// round-trip through the same fallible rotate path. An infallible
// image-rs delegate could otherwise abort on allocator pressure for
// a 16-bit PNG with EXIF Rotate90/270; the manual rotate is generic
// over `T: Copy` so the same `try_reserve_exact` gate covers u16 and
// f32 — the tests below verify byte-identical parity with
// `image::imageops` for the new element widths.

/// Build a non-square `Rgb16` image whose subpixels encode pixel
/// position so any rotation is checkable byte-for-byte against
/// `image::imageops::rotate*`. Uses the high byte of each u16 so
/// the values survive any accidental u8-truncation regression and
/// remain unique-per-pixel for small sizes.
fn xy_encoded_rgb16(width: u32, height: u32) -> DynamicImage {
  let mut buf: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(width, height);
  for y in 0..height {
    for x in 0..width {
      // (256 * (10*x + 1), 256 * (10*y + 1), 0xBEEF) — three
      // independent u16 channels with non-zero high bytes so a
      // hypothetical truncation to u8 would zero out the channel
      // distinctly.
      buf.put_pixel(
        x,
        y,
        Rgb([
          (((x * 10) as u16) + 1) << 8,
          (((y * 10) as u16) + 1) << 8,
          0xBEEF,
        ]),
      );
    }
  }
  DynamicImage::ImageRgb16(buf)
}

/// `Luma16` (single-channel u16) test image: encodes the
/// row-major pixel index in the high byte plus a fixed low-byte
/// pattern so any rotation is byte-traceable.
fn xy_encoded_luma16(width: u32, height: u32) -> DynamicImage {
  let mut buf: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(width, height);
  for y in 0..height {
    for x in 0..width {
      let idx = (y * width + x) as u16;
      // (idx << 8) | 0x5A — high byte = row-major index, low byte
      // = constant 0x5A so a stride-1 u8-truncation would lose
      // identity.
      buf.put_pixel(x, y, Luma([(idx << 8) | 0x005A]));
    }
  }
  DynamicImage::ImageLuma16(buf)
}

/// `Rgb32F` test image: encodes pixel position in three f32
/// channels with deliberately fractional values so any 4-byte
/// stride bug surfaces as a visible bit-pattern shift.
fn xy_encoded_rgb32f(width: u32, height: u32) -> DynamicImage {
  let mut buf: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::new(width, height);
  for y in 0..height {
    for x in 0..width {
      // (x + 0.25, y + 0.5, 0.875) — distinct fractions in each
      // channel; 0.875 is exactly representable so equality
      // comparison against the rotated buffer is exact.
      buf.put_pixel(x, y, Rgb([(x as f32) + 0.25, (y as f32) + 0.5, 0.875]));
    }
  }
  DynamicImage::ImageRgb32F(buf)
}

#[test]
fn rotate90_rgb16_manual_matches_reference() {
  // Rotate90 on Rgb16 routes through the same
  // truly-fallible `rotate_buf<u16>` path the u8 variants use.
  // A non-square 4x3 image must rotate to 3x4 with byte-identical
  // pixels to `image::imageops::rotate90`. This is the regression
  // test for the 16-bit-PNG-EXIF-rotate `load_image` gap.
  let img = xy_encoded_rgb16(4, 3);
  let reference = ::image::imageops::rotate90(img.as_rgb16().expect("rgb16 source"));
  let out = apply_orientation_fallible(img, Orientation::Rotate90)
    .expect("manual Rgb16 rotate90 should succeed under MAX_DECODED_IMAGE_BYTES");
  assert_eq!(out.width(), 3, "Rotate90 swaps width <- height");
  assert_eq!(out.height(), 4, "Rotate90 swaps height <- width");
  assert_eq!(
    out.as_rgb16().expect("rgb16 output").as_raw(),
    reference.as_raw(),
    "manual Rgb16 rotate90 must yield byte-identical u16 subpixels to image::imageops::rotate90"
  );
}

#[test]
fn rotate270_luma16_manual_matches_reference() {
  // Rotate270 on Luma16 — single-channel u16 path. Verifies the
  // 1-subpixel-per-pixel stride is honored for the u16 element
  // type. Non-square 4x3 → 3x4.
  let img = xy_encoded_luma16(4, 3);
  let reference = ::image::imageops::rotate270(img.as_luma16().expect("luma16 source"));
  let out =
    apply_orientation_fallible(img, Orientation::Rotate270).expect("manual Luma16 rotate270 ok");
  assert_eq!(out.width(), 3);
  assert_eq!(out.height(), 4);
  assert_eq!(
    out.as_luma16().expect("luma16 output").as_raw(),
    reference.as_raw(),
    "manual Luma16 rotate270 must yield byte-identical u16 subpixels to image::imageops::rotate270"
  );
}

#[test]
fn rotate90_rgb32f_manual_matches_reference() {
  // Rotate90 on Rgb32F — verifies the `rotate_buf<f32>` arm.
  // Comparison is bit-exact via `Vec<f32>` equality because the
  // operation is a pure permutation (no arithmetic, no FP drift).
  let img = xy_encoded_rgb32f(4, 3);
  let reference = ::image::imageops::rotate90(img.as_rgb32f().expect("rgb32f source"));
  let out =
    apply_orientation_fallible(img, Orientation::Rotate90).expect("manual Rgb32F rotate90 ok");
  assert_eq!(out.width(), 3);
  assert_eq!(out.height(), 4);
  assert_eq!(
    out.as_rgb32f().expect("rgb32f output").as_raw(),
    reference.as_raw(),
    "manual Rgb32F rotate90 must yield bit-identical f32 subpixels to image::imageops::rotate90 \
       (permutation only — no FP arithmetic)"
  );
}

#[test]
fn synthetic_16bit_rotate_succeeds_via_orientation_entry_point() {
  // End-to-end check that mimics the 16-bit `load_image`
  // closure: build a 16-bit `DynamicImage` directly (no on-disk
  // PNG needed; a unit test does not need a real decoder to
  // exercise the post-decode rotate path) and hand it to
  // `apply_orientation_fallible` with EXIF Rotate90 / Rotate270 /
  // composite orientations. All four allocating-rotate
  // orientations must succeed via the generic `rotate_buf`
  // path rather than the infallible fallback that would abort on
  // allocator pressure. (The "real 16-bit PNG decode" path is exercised
  // implicitly through `load_image` once a 16-bit PNG with EXIF
  // orientation is supplied; we test the rotate-shaped subgoal
  // here without coupling to disk I/O.)
  for rotation in [
    Orientation::Rotate90,
    Orientation::Rotate270,
    Orientation::Rotate90FlipH,
    Orientation::Rotate270FlipH,
  ] {
    let img = xy_encoded_rgb16(4, 3);
    let out = apply_orientation_fallible(img, rotation)
      .expect("16-bit Rgb16 rotate must succeed via the fallible u16 path");
    assert_eq!(out.width(), 3, "all four rotate orientations swap w<->h");
    assert_eq!(out.height(), 4);
    assert!(
      matches!(out, DynamicImage::ImageRgb16(_)),
      "output must preserve the source DynamicImage variant (Rgb16 in, Rgb16 out)"
    );
  }
}

// -- Byte-budget gate -----------------------------------------------

#[test]
fn rotate90_accepts_small_image_within_budget() {
  // The byte-budget gate against `MAX_DECODED_IMAGE_BYTES` must
  // accept inputs that fit. Sanity check: a 1x1 Rgb8 image is
  // 3 bytes, well under the 512 MiB cap. The negative
  // (overflow → ArithmeticOverflow) path uses the identical
  // `checked_mul` chain that `pad_to_square`'s overflow tests
  // already cover; we can't construct an image at hostile
  // dimensions without OOM-ing the test process itself.
  let img = xy_encoded_rgb8(1, 1);
  let bytes_per_pixel = u64::from(img.color().bytes_per_pixel());
  let bytes = u64::from(img.width()) * u64::from(img.height()) * bytes_per_pixel;
  assert!(
    bytes <= MAX_DECODED_IMAGE_BYTES,
    "1x1 Rgb8 = {bytes} bytes must be well under MAX_DECODED_IMAGE_BYTES={MAX_DECODED_IMAGE_BYTES}"
  );
  let out = apply_orientation_fallible(img, Orientation::Rotate90).expect("1x1 ok");
  assert_eq!(out.width(), 1);
  assert_eq!(out.height(), 1);
}
