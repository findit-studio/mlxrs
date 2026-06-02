//! Scalar vs NEON differential + affine-correctness + edge coverage for
//! the fused RGBA → RGB widen + affine kernel.

use super::{rgba_to_rgb_affine, rgba_to_rgb_affine_scalar};

/// The SigLIP2 NaFlex normalize affine (`x / 127.5 - 1.0`).
const SCALE: f32 = 1.0 / 127.5;
const BIAS: f32 = -1.0;

/// Deterministic RGBA input — `(i * 7) % 256` per byte, so every byte
/// differs from its neighbours (any plane-swap / stride bug is visible)
/// and the alpha lane is non-constant (so a kernel that wrongly stored
/// alpha would diverge).
fn gen_rgba(n_pixels: usize) -> Vec<u8> {
  (0..n_pixels * 4).map(|i| ((i * 7) % 256) as u8).collect()
}

/// Scalar adapter — returns the RGB f32 output for `n_pixels`.
fn scalar(src: &[u8], scale: f32, bias: f32) -> Vec<f32> {
  let mut dst = vec![0.0f32; (src.len() / 4) * 3];
  rgba_to_rgb_affine_scalar(src, &mut dst, scale, bias);
  dst
}

/// Dispatcher adapter.
fn dispatch(src: &[u8], scale: f32, bias: f32) -> Vec<f32> {
  let mut dst = vec![0.0f32; (src.len() / 4) * 3];
  rgba_to_rgb_affine(src, &mut dst, scale, bias);
  dst
}

/// Direct NEON-arm adapter, aarch64-only; caller guards on
/// `is_neon_available()`.
#[cfg(target_arch = "aarch64")]
fn neon(src: &[u8], scale: f32, bias: f32) -> Vec<f32> {
  let mut dst = vec![0.0f32; (src.len() / 4) * 3];
  // SAFETY: the only caller checks `is_neon_available()` immediately
  // before; `dst` is sized to exactly `src_pixels * 3`.
  unsafe { super::rgba_to_rgb_affine_neon(src, &mut dst, scale, bias) };
  dst
}

/// Affine correctness against hand-computed values, single pixel.
/// The kernel computes the non-fused `(x as f32) * scale + bias` per
/// channel (multiply then add, two roundings); each output equals that
/// exact expression. The SigLIP `x / 127.5 - 1.0` maps 0 → -1.0 exactly
/// and 255 → ~+1.0 (`1/127.5` is not exactly representable in f32, so it
/// lands a few ULP off the ideal +1.0). Alpha (the 4th byte) must be
/// dropped.
#[test]
fn affine_endpoints_and_drops_alpha() {
  let out = scalar(&[0, 128, 255, 77], SCALE, BIAS);
  assert_eq!(out.len(), 3, "one RGBA pixel → one RGB triple");
  // Exact: each channel is the non-fused `x * scale + bias` of its byte.
  assert_eq!(out[0], 0.0f32 * SCALE + BIAS, "0 channel");
  assert_eq!(out[1], 128.0f32 * SCALE + BIAS, "128 channel");
  assert_eq!(out[2], 255.0f32 * SCALE + BIAS, "255 channel");
  // Intent: 0 → exactly -1.0, 255 → ~+1.0 (within a few ULP).
  assert_eq!(out[0], -1.0, "0 → -1.0 exactly");
  assert!((out[2] - 1.0).abs() < 1e-6, "255 → ~+1.0, got {}", out[2]);
}

/// Affine correctness over multiple pixels, each pixel's alpha skipped.
#[test]
fn affine_multi_pixel_skips_each_alpha() {
  // Two pixels: (10, 20, 30, a=9) and (40, 50, 60, a=200).
  let out = scalar(&[10, 20, 30, 9, 40, 50, 60, 200], SCALE, BIAS);
  let expect: Vec<f32> = [10u8, 20, 30, 40, 50, 60]
    .iter()
    .map(|&b| f32::from(b) * SCALE + BIAS)
    .collect();
  assert_eq!(out, expect, "alpha (9, 200) dropped, RGB affine applied");
}

/// Finding-1 pin: each arm is **bit-for-bit** the *original* SigLIP
/// normalize expression `f32::from(byte) * (1.0/127.5) - 1.0` (a
/// non-fused multiply-then-add, two roundings) — NOT a fused
/// multiply-add, which would shift byte 255 by ~1 ULP. Asserted with
/// `==` (exact equality) for the boundary bytes 0, 127, 128, 255, on the
/// scalar arm, the dispatcher, and (when available) the NEON arm.
#[test]
fn kernel_is_bit_identical_to_original_nonfused_normalize() {
  // The exact original per-pixel formula the kernel replaced.
  fn original(byte: u8) -> f32 {
    f32::from(byte) * (1.0f32 / 127.5f32) - 1.0f32
  }

  for &byte in &[0u8, 127, 128, 255] {
    let expected = original(byte);
    // One pixel whose R/G/B are all `byte` (alpha = 0, dropped).
    let src = [byte, byte, byte, 0u8];

    let s = scalar(&src, SCALE, BIAS);
    assert_eq!(
      s,
      vec![expected; 3],
      "scalar arm must be bit-identical to f32::from({byte}) * (1/127.5) - 1.0",
    );

    let d = dispatch(&src, SCALE, BIAS);
    assert_eq!(
      d,
      vec![expected; 3],
      "dispatcher must be bit-identical to f32::from({byte}) * (1/127.5) - 1.0",
    );

    #[cfg(target_arch = "aarch64")]
    if crate::simd::is_neon_available() {
      let v = neon(&src, SCALE, BIAS);
      assert_eq!(
        v,
        vec![expected; 3],
        "NEON arm must be bit-identical to f32::from({byte}) * (1/127.5) - 1.0",
      );
    }
  }

  // Sanity: a *fused* multiply-add genuinely differs at byte 255, so the
  // pin above is non-trivial (it would catch a regression to mul_add).
  let fused_255 = 255.0f32.mul_add(1.0f32 / 127.5f32, -1.0f32);
  let nonfused_255 = original(255);
  assert_ne!(
    fused_255, nonfused_255,
    "fused vs non-fused must differ at 255 (else the pin is vacuous)",
  );
}

/// A generic (non-normalize) affine is honoured: `scale = 2.0, bias =
/// 1.0` ⇒ `2x + 1`. Confirms the kernel is not hardwired to the SigLIP
/// constants.
#[test]
fn generic_affine_scale_bias() {
  let out = scalar(&[1, 2, 3, 255], 2.0, 1.0);
  assert_eq!(
    out,
    vec![3.0f32, 5.0, 7.0],
    "2x+1 over (1,2,3), alpha dropped"
  );
}

/// Dispatcher matches the scalar reference across the same fixtures.
#[test]
fn dispatcher_matches_scalar() {
  for &n in &[0usize, 1, 2, 15, 16, 17, 31, 64, 100] {
    let src = gen_rgba(n);
    assert_eq!(
      dispatch(&src, SCALE, BIAS),
      scalar(&src, SCALE, BIAS),
      "dispatcher vs scalar differ at n_pixels={n}"
    );
  }
}

/// NEON-vs-scalar **bit-identical** parity across a length sweep that
/// includes non-multiple-of-16 pixel counts (exercising the scalar
/// tail): the NEON f32 output must equal the scalar f32 output
/// bit-for-bit (`assert_eq!` on `Vec<f32>` is bitwise for non-NaN
/// finite values, which the affine produces from `[0,255]` inputs).
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_scalar_bit_identical() {
  if !crate::simd::is_neon_available() {
    return;
  }
  // body=0 tail=N: 0,1,15; one+ full tiles + tails: 16,17,31,32,33,47,
  // 48,49; larger: 64,100,256,1000.
  for &n in &[
    0usize, 1, 2, 7, 15, 16, 17, 31, 32, 33, 47, 48, 49, 64, 100, 256, 1000,
  ] {
    let src = gen_rgba(n);
    let s = scalar(&src, SCALE, BIAS);
    let v = neon(&src, SCALE, BIAS);
    assert_eq!(
      s.len(),
      n * 3,
      "scalar output length must be 3 per pixel (n={n})"
    );
    assert_eq!(
      v, s,
      "NEON vs scalar must be byte-identical at n_pixels={n} \
       (preserves the e2e cosine parity)"
    );
  }
}

/// NEON-vs-scalar parity under a non-normalize affine, to lock the
/// `scale`/`bias` broadcast + non-fused `vmulq_f32`+`vaddq_f32` against
/// the scalar `x * scale + bias`.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_scalar_generic_affine() {
  if !crate::simd::is_neon_available() {
    return;
  }
  for &(scale, bias) in &[
    (2.0f32, 1.0f32),
    (0.5, -0.25),
    (1.0, 0.0),
    (-1.0 / 255.0, 0.5),
  ] {
    for &n in &[1usize, 16, 17, 48, 49, 100] {
      let src = gen_rgba(n);
      assert_eq!(
        neon(&src, scale, bias),
        scalar(&src, scale, bias),
        "NEON vs scalar differ at n={n}, scale={scale}, bias={bias}"
      );
    }
  }
}

/// Edge: empty input is a no-op (no writes, no panic).
#[test]
fn empty_is_noop() {
  assert!(dispatch(&[], SCALE, BIAS).is_empty());
  assert!(scalar(&[], SCALE, BIAS).is_empty());
}

/// Edge: exactly one full NEON tile (16 pixels, body=16, tail=0).
#[test]
fn sixteen_pixels_one_full_tile() {
  let src = gen_rgba(16);
  assert_eq!(dispatch(&src, SCALE, BIAS), scalar(&src, SCALE, BIAS));
}

/// Edge: one full tile + 1 tail pixel (body=16, tail=1) — pins the
/// body-then-tail handoff (catches a `body_pixels * 3` vs
/// `body_pixels * 4` slicing bug).
#[test]
fn seventeen_pixels_tile_plus_one() {
  let src = gen_rgba(17);
  let got = dispatch(&src, SCALE, BIAS);
  assert_eq!(got, scalar(&src, SCALE, BIAS));
  assert_eq!(got.len(), 17 * 3);
}

/// Release-mode precondition guard — scalar, non-RGBA-multiple src.
#[test]
#[should_panic(expected = "rgba_to_rgb_affine_scalar: src.len() (5) must be a multiple of 4")]
fn scalar_panics_on_non_rgba_src() {
  let mut dst = [0.0f32; 3];
  rgba_to_rgb_affine_scalar(&[0u8; 5], &mut dst, SCALE, BIAS);
}

/// Release-mode precondition guard — dispatcher, pixel-count mismatch
/// (4 src pixels but dst sized for 3, i.e. way short).
#[test]
#[should_panic(
  expected = "simd::vlm::rgba_to_rgb_affine: dst.len() (9) must be exactly src pixel count (4) * 3 = Some(12)"
)]
fn dispatch_panics_on_pixel_count_mismatch() {
  let src = [0u8; 16]; // 4 RGBA pixels
  let mut dst = [0.0f32; 9]; // only 3 RGB pixels
  rgba_to_rgb_affine(&src, &mut dst, SCALE, BIAS);
}

/// Finding-2 guard — dispatcher rejects a `dst` of length `3n + 1`,
/// which the loose `dst.len() / 3 == src.len() / 4` check would accept
/// via integer truncation (1 src pixel → 3 RGB f32, but dst is 4),
/// leaving the trailing dst element stale.
#[test]
#[should_panic(
  expected = "simd::vlm::rgba_to_rgb_affine: dst.len() (4) must be exactly src pixel count (1) * 3 = Some(3)"
)]
fn dispatch_panics_on_dst_len_3n_plus_1() {
  let src = [0u8; 4]; // 1 RGBA pixel → needs exactly 3 dst f32
  let mut dst = [0.0f32; 4]; // 3*1 + 1 — loose `/3` check would pass
  rgba_to_rgb_affine(&src, &mut dst, SCALE, BIAS);
}

/// Finding-2 guard — dispatcher rejects a `dst` of length `3n + 2`
/// (same integer-truncation hole, two trailing stale elements).
#[test]
#[should_panic(
  expected = "simd::vlm::rgba_to_rgb_affine: dst.len() (5) must be exactly src pixel count (1) * 3 = Some(3)"
)]
fn dispatch_panics_on_dst_len_3n_plus_2() {
  let src = [0u8; 4]; // 1 RGBA pixel → needs exactly 3 dst f32
  let mut dst = [0.0f32; 5]; // 3*1 + 2 — loose `/3` check would pass
  rgba_to_rgb_affine(&src, &mut dst, SCALE, BIAS);
}

/// Finding-2 guard — dispatcher rejects a non-multiple-of-4 `src`
/// (checked before the exact-length assertion).
#[test]
#[should_panic(expected = "simd::vlm::rgba_to_rgb_affine: src.len() (5) must be a multiple of 4")]
fn dispatch_panics_on_non_rgba_src() {
  let mut dst = [0.0f32; 3];
  rgba_to_rgb_affine(&[0u8; 5], &mut dst, SCALE, BIAS);
}

/// Finding-2 guard — scalar arm rejects `dst.len() == 3n + 1`.
#[test]
#[should_panic(
  expected = "rgba_to_rgb_affine_scalar: dst.len() (4) must be exactly src pixel count (1) * 3 = Some(3)"
)]
fn scalar_panics_on_dst_len_3n_plus_1() {
  let mut dst = [0.0f32; 4];
  rgba_to_rgb_affine_scalar(&[0u8; 4], &mut dst, SCALE, BIAS);
}

/// Finding-2 guard — scalar arm rejects `dst.len() == 3n + 2`.
#[test]
#[should_panic(
  expected = "rgba_to_rgb_affine_scalar: dst.len() (5) must be exactly src pixel count (1) * 3 = Some(3)"
)]
fn scalar_panics_on_dst_len_3n_plus_2() {
  let mut dst = [0.0f32; 5];
  rgba_to_rgb_affine_scalar(&[0u8; 4], &mut dst, SCALE, BIAS);
}

/// Release-mode precondition guard — NEON, non-RGBA-multiple src.
#[cfg(target_arch = "aarch64")]
#[test]
#[should_panic(expected = "rgba_to_rgb_affine_neon: src.len() (5) must be a multiple of 4")]
fn neon_panics_on_non_rgba_src() {
  if !crate::simd::is_neon_available() {
    panic!(
      "rgba_to_rgb_affine_neon: src.len() (5) must be a multiple of 4 (skipped — NEON unavailable)"
    );
  }
  let mut dst = [0.0f32; 3];
  // SAFETY: NEON checked; expected-panic on the precondition violation
  // before any pointer arithmetic.
  unsafe { super::rgba_to_rgb_affine_neon(&[0u8; 5], &mut dst, SCALE, BIAS) };
}

/// Finding-2 guard — NEON arm rejects `dst.len() == 3n + 1`.
#[cfg(target_arch = "aarch64")]
#[test]
#[should_panic(
  expected = "rgba_to_rgb_affine_neon: dst.len() (4) must be exactly src pixel count (1) * 3 = Some(3)"
)]
fn neon_panics_on_dst_len_3n_plus_1() {
  if !crate::simd::is_neon_available() {
    panic!(
      "rgba_to_rgb_affine_neon: dst.len() (4) must be exactly src pixel count (1) * 3 = Some(3) \
       (skipped — NEON unavailable)"
    );
  }
  let mut dst = [0.0f32; 4]; // 3*1 + 1
  // SAFETY: NEON checked; expected-panic on the exact-length
  // precondition before any pointer arithmetic.
  unsafe { super::rgba_to_rgb_affine_neon(&[0u8; 4], &mut dst, SCALE, BIAS) };
}

/// Finding-2 guard — NEON arm rejects `dst.len() == 3n + 2`.
#[cfg(target_arch = "aarch64")]
#[test]
#[should_panic(
  expected = "rgba_to_rgb_affine_neon: dst.len() (5) must be exactly src pixel count (1) * 3 = Some(3)"
)]
fn neon_panics_on_dst_len_3n_plus_2() {
  if !crate::simd::is_neon_available() {
    panic!(
      "rgba_to_rgb_affine_neon: dst.len() (5) must be exactly src pixel count (1) * 3 = Some(3) \
       (skipped — NEON unavailable)"
    );
  }
  let mut dst = [0.0f32; 5]; // 3*1 + 2
  // SAFETY: NEON checked; expected-panic on the exact-length
  // precondition before any pointer arithmetic.
  unsafe { super::rgba_to_rgb_affine_neon(&[0u8; 4], &mut dst, SCALE, BIAS) };
}
