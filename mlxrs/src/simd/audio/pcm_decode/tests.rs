//! Per-dtype differential tests + edge / behavioural coverage for the PCM decode.

use core::mem::MaybeUninit;

use super::{
  S16_INV_SCALE, S32_INV_SCALE, s16_to_f32_normalize, s16_to_f32_normalize_scalar,
  s32_to_f32_normalize, s32_to_f32_normalize_scalar,
};
use crate::simd::diff::{assert_eq_over_lane_sweep, lane_sweep_lengths};

// ── s16 adapters ──────────────────────────────────────────────

fn s16_scalar_init(src: &[i16]) -> Vec<f32> {
  let n = src.len();
  let mut v: Vec<f32> = Vec::with_capacity(n);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  s16_to_f32_normalize_scalar(&mut spare[..n], src);
  // SAFETY: kernel contract initializes every slot; cap was sized to n.
  unsafe { v.set_len(n) };
  v
}

fn s16_dispatch_init(src: &[i16]) -> Vec<f32> {
  let n = src.len();
  let mut v: Vec<f32> = Vec::with_capacity(n);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  s16_to_f32_normalize(&mut spare[..n], src);
  // SAFETY: kernel contract initializes every slot; cap was sized to n.
  unsafe { v.set_len(n) };
  v
}

#[cfg(target_arch = "aarch64")]
fn s16_neon_init(src: &[i16]) -> Vec<f32> {
  let n = src.len();
  let mut v: Vec<f32> = Vec::with_capacity(n);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  // SAFETY: caller guards on `is_neon_available()`; size is `n`;
  // kernel initializes every slot.
  unsafe {
    super::s16_to_f32_normalize_neon(&mut spare[..n], src);
    v.set_len(n);
  }
  v
}

/// Deterministic i16 input spanning the full signed range.
fn gen_i16(n: usize) -> Vec<i16> {
  (0..n)
    .map(|i| {
      let base = (i as i32 * 257) & 0xFFFF;
      (base as i16).wrapping_sub(i16::MIN / 4)
    })
    .collect()
}

/// `Exact` differential — s16 scalar vs dispatcher.
#[test]
fn s16_to_f32_scalar_matches_dispatcher_exact() {
  assert_eq_over_lane_sweep(8, s16_scalar_init, s16_dispatch_init, gen_i16);
}

/// NEON-vs-scalar bit-identical assertion for s16.
#[cfg(target_arch = "aarch64")]
#[test]
fn s16_to_f32_neon_matches_scalar_bit_identical() {
  if !crate::simd::is_neon_available() {
    return;
  }
  for &n in &[0usize, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 64, 1024] {
    let src = gen_i16(n);
    let scalar = s16_scalar_init(&src);
    let neon = s16_neon_init(&src);
    assert_eq!(neon, scalar, "s16 neon vs scalar differ at n={n}");
  }
}

/// Lane-sweep covers s16 boundary lengths.
#[test]
fn s16_lane_sweep_covers_tile_boundaries() {
  let sweep = lane_sweep_lengths(8);
  assert_eq!(sweep, [0, 1, 7, 8, 9, 15, 16, 24, 25]);
}

/// Edge values — pin the divisor + extreme samples. `i16::MIN /
/// 32768.0 = -1.0` exactly; `i16::MAX / 32768.0 = 0.999969...`.
#[test]
fn s16_specific_values() {
  let src = [0_i16, i16::MAX, i16::MIN, 1, -1, 16384, -16384];
  let expected = [
    0.0_f32,
    (i16::MAX as f32) * S16_INV_SCALE,
    (i16::MIN as f32) * S16_INV_SCALE,
    S16_INV_SCALE,
    -S16_INV_SCALE,
    0.5,
    -0.5,
  ];
  let out = s16_dispatch_init(&src);
  assert_eq!(out, expected);
}

/// Behavioural — dispatcher matches OLD per-push `f32::from(s) /
/// 32768.0` loop.
#[test]
fn s16_matches_old_push_loop() {
  let n = 65_536_usize;
  let src = gen_i16(n);
  let mut old: Vec<f32> = Vec::with_capacity(n);
  for &s in &src {
    old.push(f32::from(s) / 32768.0);
  }
  let new = s16_dispatch_init(&src);
  assert_eq!(new, old);
}

/// Release-mode size-mismatch panic (s16 scalar).
#[test]
#[should_panic(expected = "s16_to_f32_normalize_scalar: out.len() (5) must equal src.len() (7)")]
fn s16_scalar_panics_on_size_mismatch_in_release() {
  let src = [0_i16; 7];
  let mut v: Vec<f32> = Vec::with_capacity(5);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  s16_to_f32_normalize_scalar(&mut spare[..5], &src);
}

/// Release-mode size-mismatch panic (s16 dispatcher).
#[test]
#[should_panic(
  expected = "simd::audio::s16_to_f32_normalize: out.len() (5) must equal src.len() (7)"
)]
fn s16_dispatch_panics_on_size_mismatch_in_release() {
  let src = [0_i16; 7];
  let mut v: Vec<f32> = Vec::with_capacity(5);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  s16_to_f32_normalize(&mut spare[..5], &src);
}

/// Release-mode size-mismatch panic (s16 NEON).
#[cfg(target_arch = "aarch64")]
#[test]
#[should_panic(expected = "s16_to_f32_normalize_neon: out.len() (5) must equal src.len() (7)")]
fn s16_neon_panics_on_size_mismatch_in_release() {
  if !crate::simd::is_neon_available() {
    panic!(
      "s16_to_f32_normalize_neon: out.len() (5) must equal src.len() (7) (skipped — NEON unavailable)"
    );
  }
  let src = [0_i16; 7];
  let mut v: Vec<f32> = Vec::with_capacity(5);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  // SAFETY: NEON checked; expected-panic on intentional size mismatch.
  unsafe { super::s16_to_f32_normalize_neon(&mut spare[..5], &src) };
}

// ── s32 adapters ──────────────────────────────────────────────

fn s32_scalar_init(src: &[i32], inv_scale: f32) -> Vec<f32> {
  let n = src.len();
  let mut v: Vec<f32> = Vec::with_capacity(n);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  s32_to_f32_normalize_scalar(&mut spare[..n], src, inv_scale);
  // SAFETY: kernel contract initializes every slot; cap was sized to n.
  unsafe { v.set_len(n) };
  v
}

fn s32_dispatch_init(src: &[i32], inv_scale: f32) -> Vec<f32> {
  let n = src.len();
  let mut v: Vec<f32> = Vec::with_capacity(n);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  s32_to_f32_normalize(&mut spare[..n], src, inv_scale);
  // SAFETY: kernel contract initializes every slot; cap was sized to n.
  unsafe { v.set_len(n) };
  v
}

#[cfg(target_arch = "aarch64")]
fn s32_neon_init(src: &[i32], inv_scale: f32) -> Vec<f32> {
  let n = src.len();
  let mut v: Vec<f32> = Vec::with_capacity(n);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  // SAFETY: caller guards on `is_neon_available()`; size is `n`;
  // kernel initializes every slot.
  unsafe {
    super::s32_to_f32_normalize_neon(&mut spare[..n], src, inv_scale);
    v.set_len(n);
  }
  v
}

/// Deterministic i32 input spanning a wide range (not i32::MAX
/// fully — f32 cast at full i32 range has rounding drift even in
/// the scalar arm; the NEON `vcvtq_f32_s32` matches, but the test
/// uses bit-identical equality so the scalar reference and the
/// dispatcher both apply the same rounding).
fn gen_i32(n: usize) -> Vec<i32> {
  (0..n)
    .map(|i| {
      let mag = (i as i64 * 65537) % (1 << 23); // stay in i24 range so f32 is exact
      if i.is_multiple_of(2) {
        mag as i32
      } else {
        -(mag as i32)
      }
    })
    .collect()
}

/// `Exact` differential — s32 scalar vs dispatcher (using the 24-bit
/// divisor since the gen stays in the i24 range; both scalar and
/// NEON arms apply the same multiply, so output is bit-identical).
#[test]
fn s32_to_f32_scalar_matches_dispatcher_exact() {
  let inv_scale = 1.0_f32 / (1 << 23) as f32;
  assert_eq_over_lane_sweep(
    4,
    |src: &[i32]| s32_scalar_init(src, inv_scale),
    |src: &[i32]| s32_dispatch_init(src, inv_scale),
    gen_i32,
  );
}

/// NEON-vs-scalar bit-identical assertion for s32 with `S32_INV_SCALE`.
/// Uses full-range i32 inputs — the f32 cast may round for samples
/// outside `[-2^24, 2^24]`, but the rounding is bit-identical
/// between `vcvtq_f32_s32` and scalar `as f32` (both use the
/// current rounding mode = round-to-nearest-even).
#[cfg(target_arch = "aarch64")]
#[test]
fn s32_to_f32_neon_matches_scalar_bit_identical() {
  if !crate::simd::is_neon_available() {
    return;
  }
  let src: Vec<i32> = (0..1024)
    .map(|i| {
      // Range across `[-2^31, 2^31)`: stride a representative
      // sweep of magnitudes including beyond `2^24`.
      let raw = i as i64 * 4_194_303; // ≈ 2^22
      let bounded = (raw % (i32::MAX as i64)) as i32;
      if i % 2 == 0 { bounded } else { -bounded }
    })
    .collect();
  for &n in &[0usize, 1, 3, 4, 5, 7, 8, 9, 64, 1024] {
    let slice = &src[..n];
    let scalar = s32_scalar_init(slice, S32_INV_SCALE);
    let neon = s32_neon_init(slice, S32_INV_SCALE);
    assert_eq!(neon, scalar, "s32 neon vs scalar differ at n={n}");
  }
}

/// Lane-sweep covers s32 boundary lengths.
#[test]
fn s32_lane_sweep_covers_tile_boundaries() {
  let sweep = lane_sweep_lengths(4);
  assert_eq!(sweep, [0, 1, 3, 4, 5, 7, 8, 12, 13]);
}

/// Specific-value pin (s32 with 24-bit divisor — matches the
/// `symphonia::S24` arm of `push_samples` exactly).
#[test]
fn s32_with_24bit_divisor_specific_values() {
  let inv_scale = 1.0_f32 / 8_388_608.0; // 2^23
  let src = [0_i32, 8_388_608, -8_388_608, 4_194_304, -4_194_304];
  let expected = [0.0_f32, 1.0, -1.0, 0.5, -0.5];
  let out = s32_dispatch_init(&src, inv_scale);
  assert_eq!(out, expected);
}

/// Specific-value pin (s32 with 32-bit divisor).
#[test]
fn s32_with_32bit_divisor_specific_values() {
  let src = [0_i32, i32::MIN, 1_073_741_824, -1_073_741_824];
  let expected = [0.0_f32, -1.0, 0.5, -0.5];
  let out = s32_dispatch_init(&src, S32_INV_SCALE);
  assert_eq!(out, expected);
}

/// Empty input is a no-op (both arms).
#[test]
fn s32_empty_is_noop() {
  let out = s32_dispatch_init(&[], S32_INV_SCALE);
  assert!(out.is_empty());
  let out_scalar = s32_scalar_init(&[], S32_INV_SCALE);
  assert!(out_scalar.is_empty());
}

/// Release-mode size-mismatch panic (s32 scalar).
#[test]
#[should_panic(expected = "s32_to_f32_normalize_scalar: out.len() (5) must equal src.len() (7)")]
fn s32_scalar_panics_on_size_mismatch_in_release() {
  let src = [0_i32; 7];
  let mut v: Vec<f32> = Vec::with_capacity(5);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  s32_to_f32_normalize_scalar(&mut spare[..5], &src, 1.0);
}

/// Release-mode size-mismatch panic (s32 dispatcher).
#[test]
#[should_panic(
  expected = "simd::audio::s32_to_f32_normalize: out.len() (5) must equal src.len() (7)"
)]
fn s32_dispatch_panics_on_size_mismatch_in_release() {
  let src = [0_i32; 7];
  let mut v: Vec<f32> = Vec::with_capacity(5);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  s32_to_f32_normalize(&mut spare[..5], &src, 1.0);
}

/// Release-mode size-mismatch panic (s32 NEON).
#[cfg(target_arch = "aarch64")]
#[test]
#[should_panic(expected = "s32_to_f32_normalize_neon: out.len() (5) must equal src.len() (7)")]
fn s32_neon_panics_on_size_mismatch_in_release() {
  if !crate::simd::is_neon_available() {
    panic!(
      "s32_to_f32_normalize_neon: out.len() (5) must equal src.len() (7) (skipped — NEON unavailable)"
    );
  }
  let src = [0_i32; 7];
  let mut v: Vec<f32> = Vec::with_capacity(5);
  let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
  // SAFETY: NEON checked; expected-panic on intentional mismatch.
  unsafe { super::s32_to_f32_normalize_neon(&mut spare[..5], &src, 1.0) };
}
