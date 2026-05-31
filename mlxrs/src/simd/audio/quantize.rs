//! `save_wav` f32 → i16 quantize.
//!
//! Tracking: [#152](https://github.com/Findit-AI/mlxrs/issues/152).
//!
//! # The defect class
//!
//! The original [`crate::audio::io::save_wav`] inner loop was:
//!
//! ```rust,ignore
//! for &s in samples {
//!   let clipped = s.clamp(-1.0, 1.0);
//!   let q = (clipped * I16_MUL).round() as i16; // I16_MUL = 32768.0
//!   writer.write_all(&q.to_le_bytes()).map_err(...)?;
//! }
//! ```
//!
//! Three per-sample steps — clip, scale-and-round, narrow-cast — chained
//! through a stalled BufWriter call. The narrowing cast `as i16` is
//! saturating in Rust (1.45+); the explicit clamp keeps the f32 in
//! `[-1.0, 1.0]`, so the post-`* I16_MUL` value is in `[-32768.0, 32768.0]`.
//! The `+32768.0` extreme saturates to `i16::MAX = 32767` (one LSB clip on
//! the positive boundary, identical to `torchaudio.save`'s default).
//! The round semantics are `f32::round`, i.e. round-half-away-from-zero.
//!
//! # The fix — pre-quantize into a `Vec<i16>` via NEON
//!
//! The new public surface is a **batch quantizer** (`out: &mut [MaybeUninit<i16>], src: &[f32]`)
//! that pre-quantizes all samples at once. The caller (`save_wav`) then
//! issues a single `write_all` over the resulting `&[u8]` byte view —
//! one syscall instead of `n` BufWriter pushes. The kernel triple:
//!
//! 1. **Scalar reference** — `clamp(-1.0, 1.0)` → multiply by `I16_MUL`
//!    (32768.0) → `f32::round` → `as i16` (saturating cast — `+32768.0`
//!    clamps to `+32767`). Anchors the math contract.
//! 2. **NEON kernel** — 8-lane tile: two `float32x4_t` loads → `vminq_f32` +
//!    `vmaxq_f32` clamp to `[-1.0, 1.0]` → `vmulq_n_f32` scale by
//!    `I16_MUL` (32768.0) → `vcvtaq_s32_f32` (round-to-nearest-away-from-
//!    zero, ARM `FCVTAS` instruction — bit-exact match for
//!    `f32::round() as i32`) → `vqmovn_s32` (saturating narrow to i16x4
//!    — load-bearing for the `+1.0 * 32768.0 → +32768` extreme, which
//!    saturates to `i16::MAX = 32767`) → `vst1q_s16` (8-lane store).
//!    Tail handled by the scalar arm.
//! 3. **Dispatcher** — runtime `is_neon_available()` gate; falls back to
//!    scalar on non-aarch64 or when force-scalar.
//!
//! # Correctness class — `Exact`
//!
//! The NEON instruction `vcvtaq_s32_f32` (FCVTAS) implements round-to-
//! nearest, ties away from zero — bit-exact match for `f32::round()`,
//! whose IEEE 754 contract is identical (round-half-away-from-zero).
//! The `vminq_f32`/`vmaxq_f32` clamp is bit-exact `f32::clamp`. The
//! multiplication by the constant `I16_MUL` (32768.0) is the same
//! single-rounding `f32 * f32` whether evaluated by NEON or the scalar
//! FPU. The saturating narrow `vqmovn_s32` is **load-bearing on the
//! positive extreme** (`+1.0 * 32768.0 → +32768`, which exceeds `i16::MAX
//! = 32767`); it saturates to `+32767` deterministically, matching the
//! scalar `as i16` saturating-cast. For the negative extreme `-1.0` →
//! `-32768.0` → `-32768 = i16::MIN` (in-range, no saturation).
//!
//! Differential tests in this module use
//! [`crate::simd::diff::assert_eq_over_lane_sweep`] (Exact class).
//!
//! # `MaybeUninit<i16>` API — type-encoded uninit safety
//!
//! Matches the widen/fill kernels: the kernel API takes `&mut [MaybeUninit<i16>]` so the
//! `save_wav` call site can allocate via `Vec::with_capacity(n)` +
//! `spare_capacity_mut()` and `set_len(n)` after the kernel returns.
//! No `from_raw_parts_mut` over uninit memory.
//!
//! # Bench
//!
//! The NEON kernel ships unconditionally on aarch64 because
//! auto-vectorization of the scalar arm is compiler-version-dependent,
//! so bench numbers are report-only and do not drive the ship decision.
//! The bench (`mlxrs/benches/simd_quantize.rs`) exists as a regression
//! guard against both a future scalar regression and a future NEON
//! regression.

use core::mem::MaybeUninit;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
  vcombine_s16, vcvtaq_s32_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vminq_f32, vmulq_n_f32,
  vqmovn_s32, vst1q_s16,
};

/// The f32-to-i16 quantization scale.
///
/// #130: mlxrs uses the **symmetric** convention
/// (`* 32768.0` on write + `/ 32768.0` on read — matches `torchaudio.save`'s
/// default and avoids the 1-LSB drift the asymmetric `mlx_audio.audio_io`
/// convention introduces on `read → write → read` round-trips). The
/// reference's asymmetric `write = * 32767` is byte-faithful but lossy;
/// the symmetric form is correctness-preserving (round-trip is exact within
/// `[-1.0, 1.0)` and only `+1.0` clips by one LSB on the positive extreme).
///
/// Pre-clamped samples land in `[-32768.0, 32768.0]` after the multiply.
/// `-32768.0` is exactly `i16::MIN`; `+32768.0` saturates to `i16::MAX`
/// (`32767`) via the `as i16` cast in the scalar arm (Rust 1.45+
/// saturating-cast semantics) and via `vqmovn_s32` in the NEON arm — both
/// produce the same clamped result with no UB.
const I16_MUL: f32 = 32_768.0;

/// Quantize `src` f32 samples in `[-1.0, 1.0]` to i16 in `[-32768, 32767]`,
/// writing into `out`. Scalar reference — bit-exact oracle for the NEON
/// dispatcher and the fallback on every non-`aarch64` target.
///
/// **Always compiled** — independent of `target_arch`. Anchors the
/// math contract (each input `s` becomes
/// `(s.clamp(-1.0, 1.0) * I16_MUL).round() as i16`, where `I16_MUL =
/// 32768.0` — see [`I16_MUL`] / #130 for the symmetric
/// `read = / 32768.0` / `write = * 32768.0` convention), is the
/// differential-test oracle, and is the fallback path.
///
/// # Preconditions
///
/// - `out.len() == src.len()` (one i16 per input f32).
///
/// Asserted **unconditionally** (release-too). The function is `pub`
/// and its init contract is load-bearing — a release-build size
/// mismatch would let the `for (s, q) in src.iter().zip(out.iter_mut())`
/// pair short-iterate, leaving some `MaybeUninit<i16>` slots unwritten,
/// and a caller's `Vec::set_len` would then expose uninitialized
/// memory.
///
/// # Initialization contract
///
/// Every i16 of `out` is written via `MaybeUninit::write` before this
/// returns. On return the entire slice is fully initialized; the caller
/// may treat the backing memory as `[i16]` (via `Vec::set_len`,
/// `MaybeUninit::slice_assume_init_ref`, etc.).
///
/// # Non-finite handling
///
/// `save_wav` validates all samples are finite UPFRONT before calling
/// this kernel, so NaN / infinity never reaches here. A defensive
/// `s.clamp(-1.0, 1.0)` would propagate NaN (clamp returns NaN on NaN
/// input), and `(NaN * 32768.0).round() as i16` is `0` in Rust's
/// saturating-cast semantics — so even if a non-finite slipped through
/// the kernel would emit `0` for it (no UB, no panic), matching the
/// scalar `f32 as i16` cast contract.
#[inline]
#[doc(hidden)]
pub fn f32_to_i16_quantize_scalar(out: &mut [MaybeUninit<i16>], src: &[f32]) {
  assert_eq!(
    out.len(),
    src.len(),
    "f32_to_i16_quantize_scalar: out.len() ({}) must equal src.len() ({}) (one i16 per input f32)",
    out.len(),
    src.len(),
  );
  for (s, q) in src.iter().zip(out.iter_mut()) {
    let clipped = s.clamp(-1.0, 1.0);
    q.write((clipped * I16_MUL).round() as i16);
  }
}

/// Quantize `src` f32 samples in `[-1.0, 1.0]` to i16, writing into
/// `out`. NEON 8-lane (two `float32x4_t` × narrow → `int16x8_t`) tile.
///
/// # Algorithm
///
/// Per 8-lane tile:
/// 1. Load two `float32x4_t` chunks (`vld1q_f32`).
/// 2. Clamp each to `[-1.0, 1.0]` via `vminq_f32(vmaxq_f32(v, -1.0), 1.0)`.
///    The order is chosen to propagate NaN deterministically (NaN
///    survives both `vmin`/`vmax` so the subsequent multiply produces
///    NaN, which `vcvtaq_s32_f32` converts to 0 per ARM's NaN-to-zero
///    convention — matches the scalar `(NaN * 32767.0).round() as i16`
///    behavior).
/// 3. Multiply by 32767.0 via `vmulq_n_f32`.
/// 4. Convert to i32 with round-to-nearest-away-from-zero via
///    `vcvtaq_s32_f32` (ARM `FCVTAS` — bit-exact match for
///    `f32::round() as i32`).
/// 5. Saturate-narrow the two i32x4 vectors to a single i16x8 via
///    two `vqmovn_s32` + `vcombine_s16`.
/// 6. Store via `vst1q_s16`.
///
/// Tail (`src.len() % 8` samples ≤ 7) is delegated to
/// [`f32_to_i16_quantize_scalar`].
///
/// # Initialization contract
///
/// Every i16 of `out` is written before this returns — the body loop
/// covers `out[0..body_len]` via `vst1q_s16` stores (each writes 8
/// contiguous i16), and the scalar arm covers the trailing
/// `out[body_len..]` via `MaybeUninit::write`.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. Caller obligation;
///    the dispatcher [`f32_to_i16_quantize`] discharges it.
/// 2. `out.len() == src.len()` — asserted **unconditionally** here.
///
/// `vld1q_f32`/`vst1q_s16` accept unaligned addresses at full throughput
/// on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn f32_to_i16_quantize_neon(out: &mut [MaybeUninit<i16>], src: &[f32]) {
  assert_eq!(
    out.len(),
    src.len(),
    "f32_to_i16_quantize_neon: out.len() ({}) must equal src.len() ({}) (one i16 per input f32)",
    out.len(),
    src.len(),
  );

  let n = src.len();
  let body_len = n - (n % 8);

  // SAFETY: the body loop reads two `vld1q_f32` (4 lanes each = 8 f32) per
  // tile from `src.as_ptr().add(i)` for `i + 8 <= body_len <= src.len()`
  // — within bounds. It writes `vst1q_s16` (8 lanes) per tile to
  // `out.as_mut_ptr().cast::<i16>().add(i)` for the same `i` — within
  // bounds. Stores target `MaybeUninit<i16>` backing memory, which has no
  // validity invariants beyond size + alignment and accepts any bit
  // pattern — raw-pointer writes are sound. NEON availability is the
  // caller's obligation (precondition #1).
  unsafe {
    let src_base = src.as_ptr();
    let dst_base = out.as_mut_ptr().cast::<i16>();
    let lo_bound = vdupq_n_f32(-1.0);
    let hi_bound = vdupq_n_f32(1.0);

    let mut i = 0usize;
    while i + 8 <= body_len {
      // Load two 4-lane chunks.
      let v_lo = vld1q_f32(src_base.add(i));
      let v_hi = vld1q_f32(src_base.add(i + 4));

      // Clamp to [-1.0, 1.0]. NEON's vmin/vmax both propagate NaN
      // (a NaN input produces NaN output for vminq_f32(NaN, x) per
      // ARM ARM A1.7.4 "Floating-point minimum and maximum") —
      // matches Rust's `f32::clamp` for non-NaN inputs and the
      // f32 cast-to-int contract for the NaN-pathological case.
      let v_lo = vminq_f32(vmaxq_f32(v_lo, lo_bound), hi_bound);
      let v_hi = vminq_f32(vmaxq_f32(v_hi, lo_bound), hi_bound);

      // Scale by I16_MUL = 32768.0 (symmetric `* 32768`
      // convention matching `torchaudio.save`).
      let v_lo = vmulq_n_f32(v_lo, I16_MUL);
      let v_hi = vmulq_n_f32(v_hi, I16_MUL);

      // Convert to i32 with round-to-nearest, ties away from zero
      // (FCVTAS) — bit-exact match for `f32::round() as i32`.
      let i_lo = vcvtaq_s32_f32(v_lo);
      let i_hi = vcvtaq_s32_f32(v_hi);

      // Saturating narrow to two int16x4_t halves, combine into int16x8_t.
      // Saturation is **load-bearing on the positive extreme** with the
      // 32768 scale: `+1.0 * 32768.0 = +32768` exceeds `i16::MAX = 32767`,
      // and `vqmovn_s32` saturates that to `+32767` deterministically.
      // The negative extreme `-1.0 * 32768.0 = -32768` is exactly `i16::MIN`
      // (in-range, no saturation). Bit-exact match for the scalar
      // `as i16` saturating-cast.
      let combined = vcombine_s16(vqmovn_s32(i_lo), vqmovn_s32(i_hi));

      vst1q_s16(dst_base.add(i), combined);

      i += 8;
    }
  }

  if body_len < n {
    f32_to_i16_quantize_scalar(&mut out[body_len..], &src[body_len..]);
  }
}

/// Quantize `src` f32 samples to i16, writing into `out`. Routes to
/// NEON on `aarch64` (when the CPU reports NEON), else to
/// [`f32_to_i16_quantize_scalar`].
///
/// Used by [`crate::audio::io::save_wav`] to pre-quantize the f32
/// sample buffer to i16 before a single bulk byte write — replacing
/// the original per-sample `writer.write_all(&q.to_le_bytes())` BufWriter
/// loop.
///
/// # Preconditions
///
/// - `out.len() == src.len()` — asserted **unconditionally**.
///
/// # Initialization contract
///
/// **Every i16 of `out` is written before this returns.**
///
/// # Correctness class
///
/// `Exact` — bit-exact match between scalar and NEON via NEON's
/// `vcvtaq_s32_f32` (FCVTAS, ties-away-from-zero) matching
/// `f32::round`. See module-level "Correctness class" section.
#[inline]
#[doc(hidden)]
pub fn f32_to_i16_quantize(out: &mut [MaybeUninit<i16>], src: &[f32]) {
  assert_eq!(
    out.len(),
    src.len(),
    "simd::audio::f32_to_i16_quantize: out.len() ({}) must equal src.len() ({})",
    out.len(),
    src.len(),
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      // SAFETY: `is_neon_available()` confirmed NEON is on this CPU
      // (precondition #1 of `f32_to_i16_quantize_neon`). The slice-length
      // precondition (#2) was just asserted unconditionally above. The
      // kernel writes every i16 of `out` before returning per its
      // function-level contract.
      unsafe { f32_to_i16_quantize_neon(out, src) };
      return;
    }
  }
  f32_to_i16_quantize_scalar(out, src);
}

#[cfg(test)]
mod tests {
  //! Scalar vs dispatcher + scalar vs NEON differential tests + edge
  //! coverage for the quantize.

  use core::mem::MaybeUninit;

  use super::{f32_to_i16_quantize, f32_to_i16_quantize_scalar};
  use crate::simd::diff::{assert_eq_over_lane_sweep, lane_sweep_lengths};

  /// Test adapter — call the scalar kernel on `src.len()` slots of
  /// uninit `Vec<i16>` spare capacity, return the initialized vec.
  fn quantize_scalar_init(src: &[f32]) -> Vec<i16> {
    let n = src.len();
    let mut v: Vec<i16> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<i16>] = v.spare_capacity_mut();
    f32_to_i16_quantize_scalar(&mut spare[..n], src);
    // SAFETY: kernel's function-level contract initializes every slot;
    // `n <= v.capacity()` by construction.
    unsafe { v.set_len(n) };
    v
  }

  /// Test adapter — same shape, dispatcher version.
  fn quantize_dispatch_init(src: &[f32]) -> Vec<i16> {
    let n = src.len();
    let mut v: Vec<i16> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<i16>] = v.spare_capacity_mut();
    f32_to_i16_quantize(&mut spare[..n], src);
    // SAFETY: kernel's function-level contract initializes every slot;
    // `n <= v.capacity()` by construction.
    unsafe { v.set_len(n) };
    v
  }

  /// Direct NEON-arm adapter, aarch64-only. Caller is responsible for
  /// the `is_neon_available()` guard.
  #[cfg(target_arch = "aarch64")]
  fn quantize_neon_init(src: &[f32]) -> Vec<i16> {
    let n = src.len();
    let mut v: Vec<i16> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<i16>] = v.spare_capacity_mut();
    // SAFETY: caller guards on `is_neon_available()`; size is `n`
    // exactly; kernel's contract initializes every slot.
    unsafe {
      super::f32_to_i16_quantize_neon(&mut spare[..n], src);
      v.set_len(n);
    }
    v
  }

  /// Deterministic input generator — spans `[-1.5, 1.5]` so the clamp
  /// path is exercised on both sides (an under-clamped kernel would
  /// disagree at indices with `|s| > 1.0`).
  fn gen_samples(n: usize) -> Vec<f32> {
    (0..n)
      .map(|i| {
        // mix of in-range and out-of-range, mild magnitudes, both signs
        let step = 0.137_f32;
        let v = -1.5 + (i as f32) * step;
        // wrap to keep the range bounded as `n` grows
        ((v + 1.5).rem_euclid(3.0)) - 1.5
      })
      .collect()
  }

  /// `Exact` differential — scalar vs dispatcher over the lane sweep
  /// at `lanes = 8` (matches the NEON 8-lane tile width).
  #[test]
  fn quantize_scalar_matches_dispatcher_exact() {
    assert_eq_over_lane_sweep(8, quantize_scalar_init, quantize_dispatch_init, gen_samples);
  }

  /// NEON-vs-scalar bit-identical assertion via direct kernel call.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn quantize_neon_matches_scalar_bit_identical() {
    if !crate::simd::is_neon_available() {
      return;
    }
    for &n in &[0usize, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 64, 1024] {
      let src = gen_samples(n);
      let scalar = quantize_scalar_init(&src);
      let neon = quantize_neon_init(&src);
      assert_eq!(
        neon, scalar,
        "quantize_neon vs quantize_scalar differ at n={n}"
      );
    }
  }

  /// Lane-sweep covers quantize-relevant boundary lengths.
  #[test]
  fn quantize_lane_sweep_covers_tile_boundaries() {
    let sweep = lane_sweep_lengths(8);
    assert_eq!(sweep, [0, 1, 7, 8, 9, 15, 16, 24, 25]);
  }

  /// Edge: empty input is a no-op.
  #[test]
  fn quantize_empty_is_noop() {
    assert!(quantize_dispatch_init(&[]).is_empty());
    assert!(quantize_scalar_init(&[]).is_empty());
  }

  /// Edge: pin specific values to lock contract.
  ///
  /// #130: with `I16_MUL = 32768.0` (symmetric `* 32768`
  /// convention matching `torchaudio.save`):
  ///
  /// - `0.0` → `0` (no rounding ambiguity).
  /// - `1.0` → `32767` (`1.0 * 32768.0 = 32768`, saturating-cast to
  ///   `i16::MAX = 32767` — one-LSB clip on the positive extreme).
  /// - `-1.0` → `-32768` (`-1.0 * 32768.0 = -32768 = i16::MIN`, exact).
  /// - `2.0` → `32767` (clamp at 1.0, then scale → 32768 → saturate to 32767).
  /// - `-2.0` → `-32768` (clamp at -1.0, then scale → -32768).
  /// - `0.5` → `16384` (`0.5 * 32768 = 16384`, no rounding).
  /// - `-0.5` → `-16384` (symmetric).
  #[test]
  fn quantize_specific_values() {
    let src = [0.0_f32, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
    let expected = [0_i16, 32767, -32768, 32767, -32768, 16384, -16384];
    let out = quantize_dispatch_init(&src);
    assert_eq!(out, expected);
    let out_scalar = quantize_scalar_init(&src);
    assert_eq!(out_scalar, expected);
  }

  /// Behavioural test — the dispatcher must produce byte-identical
  /// output to the per-sample `(s.clamp(-1.0, 1.0) * 32768.0).round() as
  /// i16` loop over a representative sample population. #130: scale is
  /// `32768.0`, matching the symmetric write convention.
  #[test]
  fn quantize_matches_reference_loop() {
    let n = 65_536_usize;
    let src: Vec<f32> = (0..n)
      .map(|i| {
        // A spread of values: mix in-range and at-boundary samples,
        // alternate signs.
        let mag = 0.001 * (i % 2048) as f32;
        if i.is_multiple_of(2) { mag } else { -mag }
      })
      .collect();

    // Reference path — inline copy of the current idiom (32768.0 scale).
    let mut reference: Vec<i16> = Vec::with_capacity(n);
    for &s in &src {
      let clipped = s.clamp(-1.0, 1.0);
      reference.push((clipped * 32_768.0).round() as i16);
    }

    let new = quantize_dispatch_init(&src);
    assert_eq!(new, reference);
  }

  /// #130 regression — `read` divides by `I16_DIV =
  /// 32768.0` and `write` multiplies by `I16_MUL = 32768.0`, so the
  /// symmetric `f32 → i16 → f32` round-trip is exact (within rounding)
  /// for in-range samples. This pins the symmetry so a future asymmetric
  /// regression (e.g. someone reverting `I16_MUL` to 32767) immediately
  /// flunks here without needing a full filesystem round-trip through
  /// `save_wav` / `load_audio`.
  ///
  /// We check that for every multiple of `1/32768` in `[-1.0, 1.0)`:
  ///   `(f * 32768).round() / 32768.0  ==  f`  (bit-exact)
  ///
  /// 65535 grid points — chosen because each one corresponds to one i16
  /// codepoint and is exactly representable in f32.
  #[test]
  fn quantize_read_write_round_trip_is_symmetric() {
    const I16_DIV: f32 = 32_768.0;
    for k in -32_768_i32..32_768_i32 {
      let f = k as f32 / I16_DIV;
      let quantized: Vec<i16> = quantize_dispatch_init(&[f]);
      let q = quantized[0];
      // For in-range samples the quantized value is exactly k (no
      // rounding); the +1.0 boundary is excluded (k == 32768 would
      // saturate but `k as f32 / 32768.0` for `k in [-32768, 32768)`
      // never produces `+1.0`).
      assert_eq!(
        q, k as i16,
        "f={f} (k={k}) must quantize to exactly {k} (got {q})"
      );
      // Read-side reconstruction: q / 32768.0 must equal the input f
      // bit-exactly.
      let reconstructed = q as f32 / I16_DIV;
      assert_eq!(
        reconstructed.to_bits(),
        f.to_bits(),
        "round-trip drift at k={k}: f={f}, reconstructed={reconstructed}"
      );
    }
  }

  /// Release-mode precondition guard for the scalar kernel.
  #[test]
  #[should_panic(expected = "f32_to_i16_quantize_scalar: out.len() (5) must equal src.len() (7)")]
  fn quantize_scalar_panics_on_size_mismatch_in_release() {
    let src = [0.0_f32; 7];
    let mut v: Vec<i16> = Vec::with_capacity(5);
    let spare: &mut [MaybeUninit<i16>] = v.spare_capacity_mut();
    f32_to_i16_quantize_scalar(&mut spare[..5], &src);
  }

  /// Release-mode precondition guard for the dispatcher.
  #[test]
  #[should_panic(
    expected = "simd::audio::f32_to_i16_quantize: out.len() (5) must equal src.len() (7)"
  )]
  fn quantize_dispatch_panics_on_size_mismatch_in_release() {
    let src = [0.0_f32; 7];
    let mut v: Vec<i16> = Vec::with_capacity(5);
    let spare: &mut [MaybeUninit<i16>] = v.spare_capacity_mut();
    f32_to_i16_quantize(&mut spare[..5], &src);
  }

  /// Release-mode precondition guard for the NEON kernel.
  #[cfg(target_arch = "aarch64")]
  #[test]
  #[should_panic(expected = "f32_to_i16_quantize_neon: out.len() (5) must equal src.len() (7)")]
  fn quantize_neon_panics_on_size_mismatch_in_release() {
    if !crate::simd::is_neon_available() {
      panic!(
        "f32_to_i16_quantize_neon: out.len() (5) must equal src.len() (7) (skipped — NEON unavailable)"
      );
    }
    let src = [0.0_f32; 7];
    let mut v: Vec<i16> = Vec::with_capacity(5);
    let spare: &mut [MaybeUninit<i16>] = v.spare_capacity_mut();
    // SAFETY: NEON checked; expected-panic test on the intentional size
    // mismatch (precondition #2 violation) before any pointer
    // arithmetic.
    unsafe { super::f32_to_i16_quantize_neon(&mut spare[..5], &src) };
  }
}
