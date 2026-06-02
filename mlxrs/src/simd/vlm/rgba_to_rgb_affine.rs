//! Fused RGBA → RGB widen + affine: de-interleave a `&[u8]` of packed
//! RGBA pixels into a `&mut [f32]` of channel-last `[R, G, B]` triples,
//! applying a per-channel affine `x * scale + bias` and **dropping the
//! alpha** byte of each 4-byte pixel.
//!
//! This is the kernel behind the SigLIP2 NaFlex patchify-normalize: the
//! resized image is always 4-channel RGBA8, and the patchify reads each
//! pixel's leading R/G/B bytes through the SigLIP rescale
//! `x / 127.5 - 1.0` (== `x * (1/127.5) + (-1.0)`, the
//! `(x/255 - 0.5)/0.5` form) while skipping alpha. The kernel is generic
//! in `(scale, bias)` so it is reusable for any per-channel affine over
//! an RGBA source.
//!
//! # Correctness class — `Exact`
//!
//! The scalar arm and the NEON arm produce **bit-identical** output for
//! every input: both compute `(src[4i+c] as f32) * scale + bias` as a
//! **non-fused** multiply-then-add (multiply rounds, then add rounds —
//! two roundings) and write the same per-channel f32. The non-fused
//! sequence is deliberate: with `scale = 1/127.5`, `bias = -1.0` it is
//! bit-for-bit `(src[4i+c] as f32) * (1/127.5) - 1.0`, identical to the
//! original per-pixel SigLIP normalize it replaced, so there is **no**
//! ~1-ULP drift a fused multiply-add would introduce (e.g. byte 255
//! lands exactly where the original formula put it, not ~1 ULP past it).
//! The u8 → f32 widen is lossless (every u8 is exactly representable in
//! f32), and the NEON arm uses `vmulq_f32` then `vaddq_f32` — the same
//! two-rounding sequence the scalar arm performs, so there is no
//! per-lane rounding divergence. The differential test asserts
//! byte-for-byte equality across a length sweep including a
//! non-multiple-of-16 tail, and a byte-level test pins each arm to the
//! original `f32::from(byte) * (1/127.5) - 1.0` expression exactly.
//!
//! # Algorithm
//!
//! Per 16-pixel tile (64 input bytes, 48 output f32):
//! 1. `vld4q_u8` 4-way de-interleaves 64 RGBA bytes into four
//!    `uint8x16_t` planes (R, G, B, A). The A plane is loaded but never
//!    used.
//! 2. Each of the R / G / B planes is widened `u8 → u16 → u32 → f32`
//!    (four `float32x4_t` quarters per plane) and run through
//!    `vaddq_f32(vmulq_f32(plane, scale), bias)` = `plane * scale + bias`
//!    as a non-fused multiply-then-add (two roundings, matching the
//!    scalar arm's `x * scale + bias`).
//! 3. `vst3q_f32` 3-way interleaves the widened R / G / B quarters back
//!    into channel-last `[R, G, B]` output — the alpha plane is simply
//!    not stored, so it is dropped structurally (no extra shuffle).
//!
//! The `len % 16`-pixel tail (≤ 15 pixels) is delegated to the scalar
//! arm, which is bit-identical.
//!
//! # No new dependencies
//!
//! Pure `core::slice` + `core::arch::aarch64`. The dispatcher routes
//! through [`crate::simd::is_neon_available`].

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
  float32x4x3_t, vaddq_f32, vcvtq_f32_u32, vdupq_n_f32, vget_low_u8, vget_low_u16, vld4q_u8,
  vmovl_high_u8, vmovl_high_u16, vmovl_u8, vmovl_u16, vmulq_f32, vst3q_f32,
};

/// Bytes per RGBA source pixel.
const RGBA: usize = 4;
/// f32 per RGB output pixel.
const RGB: usize = 3;

/// Fused RGBA → RGB widen + affine, scalar reference.
///
/// For each pixel `i`, writes `dst[3i+c] = (src[4i+c] as f32) * scale +
/// bias` for `c ∈ {0, 1, 2}` (R, G, B); the alpha byte `src[4i+3]` is
/// dropped. Computed as a **non-fused** `x * scale + bias` (multiply
/// rounds, then add rounds) so that with `scale = 1/127.5`, `bias =
/// -1.0` it is bit-for-bit `x * (1/127.5) - 1.0` — identical to the
/// original SigLIP normalize, with no fused-multiply-add ~1-ULP drift.
/// The NEON arm uses the same non-fused `vmulq_f32` + `vaddq_f32`, so
/// the two arms agree bit-for-bit.
///
/// **Always compiled** — independent of `target_arch`. Anchors the math
/// contract, is the differential-test oracle, and is the fallback path
/// on every non-`aarch64` target.
///
/// # Preconditions
///
/// - `src.len()` must be a multiple of [`RGBA`] (4) — each input pixel
///   is 4 bytes.
/// - `dst.len()` must be **exactly** `(src.len() / 4) * 3` — one RGB
///   output triple per RGBA input pixel, with no trailing slack. A `dst`
///   of length `3n + 1` or `3n + 2` is rejected (the loose
///   `dst.len() / 3 == src.len() / 4` check would accept it via integer
///   division, then leave the trailing `dst` element(s) unwritten).
///
/// Both are asserted **unconditionally** (release-too): the function is
/// `pub`, reachable through `simd::vlm::rgba_to_rgb_affine`, and a
/// release-build length mismatch would let
/// `chunks_exact(4).zip(chunks_exact_mut(3))` truncate, silently
/// leaving trailing output pixels unwritten.
#[inline]
#[doc(hidden)]
pub fn rgba_to_rgb_affine_scalar(src: &[u8], dst: &mut [f32], scale: f32, bias: f32) {
  assert!(
    src.len().is_multiple_of(RGBA),
    "rgba_to_rgb_affine_scalar: src.len() ({}) must be a multiple of 4 (one input pixel = 4 RGBA bytes)",
    src.len(),
  );
  let src_pixels = src.len() / RGBA;
  assert_eq!(
    src_pixels.checked_mul(RGB),
    Some(dst.len()),
    "rgba_to_rgb_affine_scalar: dst.len() ({}) must be exactly src pixel count ({}) * 3 = {:?} \
     (src.len()={})",
    dst.len(),
    src_pixels,
    src_pixels.checked_mul(RGB),
    src.len(),
  );

  for (src_px, dst_px) in src.chunks_exact(RGBA).zip(dst.chunks_exact_mut(RGB)) {
    // Copy the 3 RGB channels through the affine, skip alpha (src_px[3]).
    // Non-fused `x * scale + bias` (two roundings) — bit-for-bit the
    // original `x * (1/127.5) - 1.0`, matching the NEON arm.
    dst_px[0] = f32::from(src_px[0]) * scale + bias;
    dst_px[1] = f32::from(src_px[1]) * scale + bias;
    dst_px[2] = f32::from(src_px[2]) * scale + bias;
  }
}

/// Fused RGBA → RGB widen + affine, NEON 16-pixel `vld4q_u8` +
/// `vmulq_f32` + `vaddq_f32` + `vst3q_f32` tile.
///
/// See the [module docs](self) for the algorithm. Bit-identical to
/// [`rgba_to_rgb_affine_scalar`].
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU — the caller's
///    obligation, discharged by the dispatcher
///    [`rgba_to_rgb_affine`]'s [`crate::simd::is_neon_available`] gate.
/// 2. `src.len()` must be a multiple of 4 and `dst.len()` must be
///    **exactly** `(src.len() / 4) * 3` (no trailing slack). Both are
///    asserted **unconditionally** here (release-too — a release
///    mismatch would OOB-read `src` / OOB-write `dst` in the tile body,
///    or leave trailing `dst` elements unwritten).
///
/// `vld4q_u8` / `vst3q_f32` accept unaligned addresses at full
/// throughput on aarch64 (no faulting, no perf cliff).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgba_to_rgb_affine_neon(src: &[u8], dst: &mut [f32], scale: f32, bias: f32) {
  assert!(
    src.len().is_multiple_of(RGBA),
    "rgba_to_rgb_affine_neon: src.len() ({}) must be a multiple of 4 (one input pixel = 4 RGBA bytes)",
    src.len(),
  );
  let n_pixels = src.len() / RGBA;
  assert_eq!(
    n_pixels.checked_mul(RGB),
    Some(dst.len()),
    "rgba_to_rgb_affine_neon: dst.len() ({}) must be exactly src pixel count ({}) * 3 = {:?} \
     (src.len()={})",
    dst.len(),
    n_pixels,
    n_pixels.checked_mul(RGB),
    src.len(),
  );

  let body_pixels = n_pixels - (n_pixels % 16);

  // SAFETY: the body loop reads `src.as_ptr().add(p * 4)` for
  // `p + 16 <= body_pixels`, i.e. `p * 4 + 64 <= body_pixels * 4 <=
  // src.len()` — within bounds (`vld4q_u8` reads 64 contiguous bytes).
  // It writes `dst.as_mut_ptr().add(p * 3)` for the same `p` — i.e.
  // `p * 3 + 48 <= body_pixels * 3 <= dst.len()` — within bounds
  // (`vst3q_f32` writes 12 contiguous f32 = 48 bytes per call, ×4 calls
  // = 48 f32 per tile). The `scale` / `bias` broadcasts are loop
  // invariants. NEON availability is the caller's obligation
  // (precondition #1 — discharged by the dispatcher).
  unsafe {
    let src_base = src.as_ptr();
    let dst_base = dst.as_mut_ptr();
    let scale_v = vdupq_n_f32(scale);
    let bias_v = vdupq_n_f32(bias);

    let mut p = 0usize;
    while p + 16 <= body_pixels {
      // 4-way de-interleave 64 RGBA bytes (16 pixels) into planes:
      // `planes.0 = R`, `.1 = G`, `.2 = B`, `.3 = A` (alpha unused).
      let planes = vld4q_u8(src_base.add(p * RGBA));

      // Widen one u8x16 plane to four `float32x4_t` quarters and apply
      // the non-fused `quarter * scale + bias`
      // (`vaddq_f32(vmulq_f32(quarter, scale), bias)` — multiply rounds,
      // then add rounds, matching the scalar arm bit-for-bit). The chain
      // is u8x16 -> u16x8 (low/high) -> u32x4 ×4 -> f32x4 ×4.
      macro_rules! widen_affine_quarters {
        ($plane:expr) => {{
          let lo16 = vmovl_u8(vget_low_u8($plane));
          let hi16 = vmovl_high_u8($plane);
          [
            vaddq_f32(
              vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))), scale_v),
              bias_v,
            ),
            vaddq_f32(
              vmulq_f32(vcvtq_f32_u32(vmovl_high_u16(lo16)), scale_v),
              bias_v,
            ),
            vaddq_f32(
              vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))), scale_v),
              bias_v,
            ),
            vaddq_f32(
              vmulq_f32(vcvtq_f32_u32(vmovl_high_u16(hi16)), scale_v),
              bias_v,
            ),
          ]
        }};
      }

      let r = widen_affine_quarters!(planes.0);
      let g = widen_affine_quarters!(planes.1);
      let b = widen_affine_quarters!(planes.2);

      // 3-way interleave-store the widened R/G/B quarters (alpha
      // dropped — never stored). Each `vst3q_f32` writes 4 pixels ×
      // 3 channels = 12 f32 = 48 bytes; ×4 calls = 16 pixels per tile.
      vst3q_f32(dst_base.add(p * RGB), float32x4x3_t(r[0], g[0], b[0]));
      vst3q_f32(dst_base.add(p * RGB + 12), float32x4x3_t(r[1], g[1], b[1]));
      vst3q_f32(dst_base.add(p * RGB + 24), float32x4x3_t(r[2], g[2], b[2]));
      vst3q_f32(dst_base.add(p * RGB + 36), float32x4x3_t(r[3], g[3], b[3]));

      p += 16;
    }
  }

  // Tail: `n_pixels % 16` pixels (≤ 15 = 60 input bytes + 45 f32
  // output). Delegate to the scalar arm — bit-identical.
  if body_pixels < n_pixels {
    let src_tail = body_pixels * RGBA;
    let dst_tail = body_pixels * RGB;
    rgba_to_rgb_affine_scalar(&src[src_tail..], &mut dst[dst_tail..], scale, bias);
  }
}

/// Fused RGBA → RGB widen + affine `x * scale + bias` (alpha dropped).
/// Routes to NEON on `aarch64` (when the CPU reports NEON), else to
/// [`rgba_to_rgb_affine_scalar`].
///
/// Used by the SigLIP2 NaFlex patchify-normalize
/// ([`crate::embeddings::siglip2_naflex::processing`]) with
/// `scale = 1.0 / 127.5`, `bias = -1.0`.
///
/// # Preconditions
///
/// - `src.len() % 4 == 0` — each input pixel is 4 RGBA bytes.
/// - `dst.len()` is **exactly** `(src.len() / 4) * 3` — one RGB output
///   triple per RGBA input pixel, with no trailing slack (a `dst` of
///   length `3n + 1` or `3n + 2` is rejected; the kernel writes exactly
///   `3n` f32 and must not leave a trailing element stale).
///
/// Both are asserted **unconditionally** (release-too), matching the
/// kernels' own entry-point assertions so direct callers (the bench,
/// the tests) are equally protected.
///
/// # Correctness class
///
/// `Exact` — bit-identical scalar vs NEON output (lossless u8 → f32
/// widen + a non-fused multiply-then-add on every channel).
#[inline]
#[doc(hidden)]
pub fn rgba_to_rgb_affine(src: &[u8], dst: &mut [f32], scale: f32, bias: f32) {
  assert!(
    src.len().is_multiple_of(RGBA),
    "simd::vlm::rgba_to_rgb_affine: src.len() ({}) must be a multiple of 4",
    src.len(),
  );
  let src_pixels = src.len() / RGBA;
  assert_eq!(
    src_pixels.checked_mul(RGB),
    Some(dst.len()),
    "simd::vlm::rgba_to_rgb_affine: dst.len() ({}) must be exactly src pixel count ({}) * 3 = {:?}",
    dst.len(),
    src_pixels,
    src_pixels.checked_mul(RGB),
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      // SAFETY: `is_neon_available()` confirmed NEON is on this CPU
      // (precondition #1 of `rgba_to_rgb_affine_neon`). The
      // slice-length preconditions (#2) were just asserted
      // unconditionally above.
      unsafe { rgba_to_rgb_affine_neon(src, dst, scale, bias) };
      return;
    }
  }
  rgba_to_rgb_affine_scalar(src, dst, scale, bias);
}

#[cfg(test)]
mod tests;
