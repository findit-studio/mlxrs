//! PCM sample decode → normalized f32 widen.
//!
//! Tracking: [#146](https://github.com/Findit-AI/mlxrs/issues/146).
//!
//! # The defect class
//!
//! The original `crate::audio::io::push_samples` inner loops were
//! per-sample `f32::from(s) / divisor` (or `s as f32 / divisor`)
//! `Vec::push`es — each push has a bounds check, a `len` update, and
//! a non-vectorizable iterator shape. Symphonia hands us decoded PCM
//! samples one buffer at a time as `&[T]` where `T` is the typed
//! sample width (i8 / i16 / i32; u8 / u16 / u32 offset-binary;
//! i24/u24 packed-in-i32). The decode boils down to:
//!
//! - **Signed arms**: divide by `2^(bits-1)` (= 128 / 32768 / 2^23 /
//!   2^31). The reference's `mlx-audio` exact divisors are
//!   `8 / 16 / 24 / 32` widths.
//! - **Unsigned (offset-binary) arms**: subtract midpoint
//!   `2^(bits-1)`, then divide by `2^(bits-1)`. Equivalent to first
//!   reinterpreting as `i*` then applying the signed divisor — but
//!   the wraparound semantics on u8 → i16 / u16 → i32 / u24 → i32 /
//!   u32 → i32 require a typed cast chain, not a blind reinterpret.
//!
//! # The fix — per-dtype NEON kernels for the hot widths
//!
//! This module ships NEON kernels for the **two hot signed widths**
//! that dominate real-world WAV/FLAC:
//!
//! - [`s16_to_f32_normalize`] — 8-lane (int16x8_t → two float32x4_t)
//!   widen + multiply by `1.0 / 32768.0`. Hottest path: 16-bit PCM is
//!   the default for WAV and CD-quality FLAC.
//! - [`s32_to_f32_normalize`] — 4-lane (int32x4_t → float32x4_t) widen
//!   + multiply by `1.0 / 2^31`. Used for 24-bit and 32-bit PCM
//!     (symphonia's `S24` `inner()` returns `i32` in `[-2^23, 2^23)`;
//!     we apply the appropriate divisor at the call site).
//!
//! The remaining variants (s8, offset-binary u8/u16/u24/u32) keep
//! their scalar `Vec::push` loops — they are cold (very rare 8-bit
//! PCM; offset-binary is essentially WAV-only legacy), and bundling
//! more NEON kernels for those would be dead code on the hot path.
//! The scalar reference for s16 / s32 lives in this module too so a
//! call site that needs the auditable scalar version (e.g. a
//! force-scalar build) has a clean entry point.
//!
//! # Correctness class — `Exact` (integer arms)
//!
//! The integer-to-fp widens are **lossless** for the source widths in
//! question — an i16 / i32 is exactly representable in f32's 24-bit
//! mantissa for s16, and within rounding for s32 (full i32 range
//! exceeds f32's exact-representation window of `[-2^24, 2^24]`, but
//! the rounding is identical between `vcvtq_f32_s32` and the scalar
//! `s as f32` cast — both use the current rounding mode, default
//! round-to-nearest-even).
//!
//! The multiply by `1.0 / 2^(bits-1)` is a single `f32 * f32` —
//! identical between NEON `vmulq_n_f32` and scalar `*`. The dispatcher
//! produces bit-identical output to the scalar reference for every
//! input across the s16 and s32 arms.
//!
//! Differential tests in this module use
//! [`crate::simd::diff::assert_eq_over_lane_sweep`] (Exact class).
//!
//! # `MaybeUninit<f32>` API
//!
//! Matches the widen/quantize kernels: each kernel takes `&mut [MaybeUninit<f32>]` so the
//! call site can pass `Vec::spare_capacity_mut()` directly.
//!
//! # Bench
//!
//! `mlxrs/benches/simd_pcm_decode.rs`.

use core::mem::MaybeUninit;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
  vcvtq_f32_s32, vget_low_s16, vld1q_s16, vld1q_s32, vmovl_high_s16, vmovl_s16, vmulq_n_f32,
  vst1q_f32,
};

/// f32 divisor for 16-bit signed PCM — `1.0 / 2^15 = 1.0 / 32768.0`.
/// Matches `mlx_audio.audio_io.read`'s `int16 / 32768.0` convention.
pub const S16_INV_SCALE: f32 = 1.0 / 32_768.0;

/// f32 divisor for 32-bit signed PCM — `1.0 / 2^31 = 1.0 / 2_147_483_648.0`.
/// Matches `mlx_audio.audio_io.read`'s `int32 / 2^31` convention.
pub const S32_INV_SCALE: f32 = 1.0 / 2_147_483_648.0;

// ─── 16-bit signed PCM → f32 ──────────────────────────────────────────

/// Convert `src` 16-bit signed PCM samples to f32 normalized to
/// `[-1.0, 1.0)`, writing to `out`. Scalar reference — bit-exact
/// oracle for the NEON dispatcher and fallback on non-`aarch64`.
///
/// # Preconditions
///
/// - `out.len() == src.len()` (one f32 per input i16).
///
/// Asserted **unconditionally** (release-too).
///
/// # Initialization contract
///
/// Every f32 of `out` is written via `MaybeUninit::write` before this
/// returns.
#[inline]
#[doc(hidden)]
pub fn s16_to_f32_normalize_scalar(out: &mut [MaybeUninit<f32>], src: &[i16]) {
  assert_eq!(
    out.len(),
    src.len(),
    "s16_to_f32_normalize_scalar: out.len() ({}) must equal src.len() ({}) (one f32 per i16)",
    out.len(),
    src.len(),
  );
  for (slot, &s) in out.iter_mut().zip(src.iter()) {
    slot.write(f32::from(s) * S16_INV_SCALE);
  }
}

/// Convert `src` 16-bit signed PCM samples to f32. NEON 8-lane
/// (int16x8_t → two float32x4_t) tile.
///
/// # Algorithm
///
/// Per 8-lane tile:
/// 1. Load 8 i16 via `vld1q_s16`.
/// 2. Widen to two `int32x4_t` halves via `vmovl_s16` (low) +
///    `vmovl_high_s16` (high) — sign-extending widen.
/// 3. Convert each to `float32x4_t` via `vcvtq_f32_s32` (lossless —
///    the i16 range `[-32768, 32767]` is exactly representable in
///    f32's 24-bit mantissa).
/// 4. Multiply by `1.0 / 32768.0` via `vmulq_n_f32`.
/// 5. Two `vst1q_f32` stores (4 f32 each = 8 f32 total per tile).
///
/// Tail (`src.len() % 8` samples ≤ 7) is delegated to the scalar arm.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. Caller obligation
///    (the dispatcher discharges it).
/// 2. `out.len() == src.len()` — asserted unconditionally here.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn s16_to_f32_normalize_neon(out: &mut [MaybeUninit<f32>], src: &[i16]) {
  assert_eq!(
    out.len(),
    src.len(),
    "s16_to_f32_normalize_neon: out.len() ({}) must equal src.len() ({}) (one f32 per i16)",
    out.len(),
    src.len(),
  );

  let n = src.len();
  let body_len = n - (n % 8);

  // SAFETY: the body loop reads 8 i16 via `vld1q_s16` from
  // `src.as_ptr().add(i)` for `i + 8 <= body_len <= src.len()` —
  // within bounds. It writes two `vst1q_f32` (4 f32 each = 8 f32) per
  // tile to `out.as_mut_ptr().cast::<f32>().add(i)` for the same `i`,
  // within bounds. Stores target `MaybeUninit<f32>`. NEON availability
  // is the caller's obligation (precondition #1).
  unsafe {
    let src_base = src.as_ptr();
    let dst_base = out.as_mut_ptr().cast::<f32>();
    let mut i = 0usize;
    while i + 8 <= body_len {
      let v = vld1q_s16(src_base.add(i));
      let lo32 = vmovl_s16(vget_low_s16(v));
      let hi32 = vmovl_high_s16(v);
      let lo_f = vmulq_n_f32(vcvtq_f32_s32(lo32), S16_INV_SCALE);
      let hi_f = vmulq_n_f32(vcvtq_f32_s32(hi32), S16_INV_SCALE);
      vst1q_f32(dst_base.add(i), lo_f);
      vst1q_f32(dst_base.add(i + 4), hi_f);
      i += 8;
    }
  }

  if body_len < n {
    s16_to_f32_normalize_scalar(&mut out[body_len..], &src[body_len..]);
  }
}

/// Convert `src` 16-bit signed PCM samples to f32 in `[-1.0, 1.0)`,
/// writing to `out`. Routes to NEON on `aarch64` (when NEON is
/// reported), else to [`s16_to_f32_normalize_scalar`].
///
/// # Preconditions
///
/// - `out.len() == src.len()` — asserted **unconditionally**.
///
/// # Correctness class
///
/// `Exact`.
#[inline]
#[doc(hidden)]
pub fn s16_to_f32_normalize(out: &mut [MaybeUninit<f32>], src: &[i16]) {
  assert_eq!(
    out.len(),
    src.len(),
    "simd::audio::s16_to_f32_normalize: out.len() ({}) must equal src.len() ({})",
    out.len(),
    src.len(),
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      // SAFETY: NEON gated; size precondition asserted above; kernel
      // contract initializes every slot.
      unsafe { s16_to_f32_normalize_neon(out, src) };
      return;
    }
  }
  s16_to_f32_normalize_scalar(out, src);
}

// ─── 32-bit signed PCM (and i24-as-i32) → f32 ─────────────────────────

/// Convert `src` 32-bit signed PCM samples to f32 by multiplying by
/// `inv_scale` (typically `1.0 / 2^(bits-1)` for the source's bit
/// depth — `1/2^23` for 24-bit-packed-in-i32, `1/2^31` for 32-bit).
/// Scalar reference — bit-exact oracle for the NEON dispatcher.
///
/// # Preconditions
///
/// - `out.len() == src.len()` (one f32 per input i32).
///
/// Asserted **unconditionally** (release-too).
#[inline]
#[doc(hidden)]
pub fn s32_to_f32_normalize_scalar(out: &mut [MaybeUninit<f32>], src: &[i32], inv_scale: f32) {
  assert_eq!(
    out.len(),
    src.len(),
    "s32_to_f32_normalize_scalar: out.len() ({}) must equal src.len() ({}) (one f32 per i32)",
    out.len(),
    src.len(),
  );
  for (slot, &s) in out.iter_mut().zip(src.iter()) {
    slot.write((s as f32) * inv_scale);
  }
}

/// NEON 4-lane (int32x4_t → float32x4_t) widen + multiply.
///
/// # Safety
///
/// NEON must be available; `out.len() == src.len()` (asserted).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn s32_to_f32_normalize_neon(
  out: &mut [MaybeUninit<f32>],
  src: &[i32],
  inv_scale: f32,
) {
  assert_eq!(
    out.len(),
    src.len(),
    "s32_to_f32_normalize_neon: out.len() ({}) must equal src.len() ({}) (one f32 per i32)",
    out.len(),
    src.len(),
  );

  let n = src.len();
  let body_len = n - (n % 4);

  // SAFETY: reads 4 i32 per `vld1q_s32` for `i + 4 <= body_len <= n`;
  // writes 4 f32 per `vst1q_f32` for the same `i`. Stores target
  // `MaybeUninit<f32>` backing memory. NEON availability discharged
  // by the dispatcher's gate.
  unsafe {
    let src_base = src.as_ptr();
    let dst_base = out.as_mut_ptr().cast::<f32>();
    let mut i = 0usize;
    while i + 4 <= body_len {
      let v = vld1q_s32(src_base.add(i));
      let f = vmulq_n_f32(vcvtq_f32_s32(v), inv_scale);
      vst1q_f32(dst_base.add(i), f);
      i += 4;
    }
  }

  if body_len < n {
    s32_to_f32_normalize_scalar(&mut out[body_len..], &src[body_len..], inv_scale);
  }
}

/// Convert `src` 32-bit signed PCM samples to f32 by multiplying by
/// `inv_scale`. Routes to NEON on `aarch64`, else to scalar.
///
/// # Preconditions
///
/// - `out.len() == src.len()` — asserted **unconditionally**.
#[inline]
#[doc(hidden)]
pub fn s32_to_f32_normalize(out: &mut [MaybeUninit<f32>], src: &[i32], inv_scale: f32) {
  assert_eq!(
    out.len(),
    src.len(),
    "simd::audio::s32_to_f32_normalize: out.len() ({}) must equal src.len() ({})",
    out.len(),
    src.len(),
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      // SAFETY: NEON gated; size precondition asserted; kernel
      // contract initializes every slot.
      unsafe { s32_to_f32_normalize_neon(out, src, inv_scale) };
      return;
    }
  }
  s32_to_f32_normalize_scalar(out, src, inv_scale);
}

#[cfg(test)]
mod tests;
