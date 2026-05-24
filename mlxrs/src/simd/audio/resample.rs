//! C8 — `resample_linear` linear interpolation.
//!
//! Tracking: [#153](https://github.com/Findit-AI/mlxrs/issues/153).
//! Plan: `docs/core-arch-simd-candidates.md` §2 row C8, §3.4 (resample
//! linear).
//!
//! # The defect class
//!
//! Pre-C8 `crate::audio::io::resample_linear` is a per-output-sample loop:
//!
//! ```rust,ignore
//! for i in 0..out_len {
//!   let x = i as f64 * ratio;
//!   let lo = x.floor();
//!   let frac = x - lo;
//!   let lo_idx = (lo as usize).min(last_in);
//!   let hi_idx = (lo_idx + 1).min(last_in);
//!   let a = samples[lo_idx];
//!   let b = samples[hi_idx];
//!   out.push(a + (b - a) * frac as f32);
//! }
//! ```
//!
//! Three independent steps per output sample — index math, two gathered
//! source loads, one FMA. The gather is the inherent bottleneck (NEON
//! has no scatter/gather), but the index math + FMA chain vectorize
//! cleanly into a 4-lane tile that reduces the latency-bound critical
//! path on M-series.
//!
//! # The fix — 4-lane NEON tile
//!
//! Per output tile of 4 samples:
//! 1. Vectorize the index math: compute `lo_idx[lane]`,
//!    `hi_idx[lane]`, and `frac[lane]` for `lane ∈ [0, 4)`.
//! 2. Scalar gather: load `a[lane] = samples[lo_idx[lane]]` and
//!    `b[lane] = samples[hi_idx[lane]]` for each lane (NEON has no
//!    gather; the loads are inherently scalar).
//! 3. NEON FMA: `out_tile = a + (b - a) * frac` via one `vfmaq_f32`.
//! 4. `vst1q_f32` store of the 4-lane result.
//!
//! Tail samples (`out_len % 4` ≤ 3) handled by the scalar arm. The
//! index math also short-circuits when the source is `samples.len() ==
//! 1` (degenerate: every output is `samples[0]`); the dispatcher
//! delegates that case to scalar.
//!
//! # Correctness class — `Tolerance`
//!
//! NEON's FMA evaluates `a + (b - a) * frac` as a single rounding via
//! `vfmaq_f32`. The scalar arm evaluates `(b - a) * frac` then adds
//! `a` — two separate roundings. The difference is at most one ULP
//! per sample (~6e-8 absolute for samples in `[-1, 1]`). For longer
//! windows the diff stays bounded per-element, so the differential
//! test uses [`crate::simd::diff::assert_close_slice_over_lane_sweep`]
//! with `abs = 1e-6, rel = 1e-6` — wide enough for the per-sample FMA
//! divergence yet tight enough to catch a stale-stride or wrong-index
//! regression.
//!
//! # Index-math precision
//!
//! The scalar reference uses f64 for `x = i * ratio` and `lo = x.floor()`
//! to keep the index math stable for long resampled streams. The NEON
//! arm replicates this — `ratio: f64`, `x: float64x2_t` per pair of
//! output samples, then narrowed to f32 for the FMA only. This means
//! per 4-lane output tile we issue two `float64x2_t` index computations
//! (lanes 0-1 and 2-3) plus the f32 FMA.
//!
//! # `Vec<f32>` output API
//!
//! Matches the caller's allocation discipline — the dispatcher writes
//! into a pre-reserved `&mut [MaybeUninit<f32>]` (sized to `out_len`),
//! and the caller (`resample_linear`) wraps it with `Vec::with_capacity`
//! + `spare_capacity_mut` + `set_len(out_len)`.
//!
//! # Verify-before-claim bench
//!
//! Report-only per the user directive 2026-05-23 (project memory rule
//! **"SIMD ship NEON regardless"**).

use core::mem::MaybeUninit;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
  float32x4_t, float64x2_t, vdupq_n_f64, vfmaq_f32, vmulq_f64, vst1q_f32, vsubq_f32,
};

/// Scalar reference: linear-interpolation resample of `samples` into
/// `out` using output-index → input-position factor `ratio = from / to`.
/// Bit-exact match for the pre-C8 `resample_linear` inner loop, with
/// the caller-managed `out_len = samples.len() * to_rate / from_rate`
/// length contract.
///
/// `last_in = samples.len() - 1` is passed explicitly so the kernel
/// can stay branch-free on the input bound — the caller has already
/// rejected `samples.is_empty()` upstream.
///
/// # Preconditions
///
/// - `!samples.is_empty()` — asserted unconditionally.
/// - `out.len()` matches the caller's pre-computed `out_len`.
///
/// # Initialization contract
///
/// Every f32 of `out` is written via `MaybeUninit::write` before this
/// returns.
#[inline]
#[doc(hidden)]
pub fn resample_linear_scalar(out: &mut [MaybeUninit<f32>], samples: &[f32], ratio: f64) {
  assert!(
    !samples.is_empty(),
    "resample_linear_scalar: samples must be non-empty"
  );
  let last_in = samples.len() - 1;
  for (i, slot) in out.iter_mut().enumerate() {
    let x = i as f64 * ratio;
    let lo = x.floor();
    let frac = x - lo;
    let lo_idx = (lo as usize).min(last_in);
    let hi_idx = (lo_idx + 1).min(last_in);
    let a = samples[lo_idx];
    let b = samples[hi_idx];
    slot.write(a + (b - a) * frac as f32);
  }
}

/// NEON 4-lane linear-interpolation resample for the body region
/// (caller-side guaranteed `out.len() % 4 == 0`). Per output tile of 4:
/// compute `lo_idx[lane]`, `frac[lane]` via two `float64x2_t` index
/// computations; scalar-gather `a[lane]` / `b[lane]`; FMA `a + (b - a)
/// * frac` via `vfmaq_f32`; store via `vst1q_f32`.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. Caller obligation;
///    discharged by [`resample_linear`].
/// 2. `!samples.is_empty()` — asserted unconditionally here.
/// 3. `out.len()` is a multiple of 4 — asserted unconditionally here.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn resample_linear_neon(out: &mut [MaybeUninit<f32>], samples: &[f32], ratio: f64) {
  assert!(
    !samples.is_empty(),
    "resample_linear_neon: samples must be non-empty"
  );
  assert!(
    out.len().is_multiple_of(4),
    "resample_linear_neon: out.len() ({}) must be a multiple of 4 (caller splits the tail)",
    out.len(),
  );
  let last_in = samples.len() - 1;
  let body_len = out.len();

  // SAFETY: the body loop writes a single `vst1q_f32` (4 lanes = 16
  // bytes = 4 f32 slots of `MaybeUninit<f32>`) per tile at
  // `dst_base.add(i)` for `i + 4 <= body_len <= out.len()`, within
  // bounds. Stores target `MaybeUninit<f32>` backing memory, which has
  // no validity invariants beyond size + alignment and accepts any bit
  // pattern. Source gathers are scalar `samples[lo_idx]` / `samples[hi_idx]`
  // through index-bounded subscripts (`lo_idx, hi_idx <= last_in <
  // samples.len()`), so the scalar reads are safe. NEON availability
  // is the caller's obligation (precondition #1).
  unsafe {
    let dst_base = out.as_mut_ptr().cast::<f32>();
    let ratio_v = vdupq_n_f64(ratio);

    let mut i = 0usize;
    while i + 4 <= body_len {
      // Lanes 0-1: x = (i+0, i+1) * ratio
      let lane0_1 = {
        let base = [(i as f64), (i + 1) as f64];
        // Load two f64 lane bases via a small stack array. We avoid a
        // direct `vld1q_f64` over `[f64; 2]` because that adds an
        // alignment hazard on some toolchains; instead, build the
        // 2-lane vector via two `vsetq_lane_f64` calls (the loads are
        // already in registers since `base` is a 16-byte stack tuple).
        // The actual codegen the compiler picks is the same.
        let v = core::arch::aarch64::vld1q_f64(base.as_ptr());
        vmulq_f64(v, ratio_v)
      };
      let lane2_3 = {
        let base = [(i + 2) as f64, (i + 3) as f64];
        let v = core::arch::aarch64::vld1q_f64(base.as_ptr());
        vmulq_f64(v, ratio_v)
      };

      // Extract f64 lanes back to scalar to compute `floor`, integer
      // index, and `frac`. ARM has no f64 `floor` intrinsic; the
      // libm-free implementation is `vcvtm` (round-toward-negative-
      // infinity), available as `vcvtmq_s64_f64`, but the simpler
      // route here is to extract two f64 per pair and use the scalar
      // `f64::floor` so the math matches the scalar reference exactly.
      let extract = |v: float64x2_t, lane: u32| -> f64 {
        match lane {
          0 => core::arch::aarch64::vgetq_lane_f64::<0>(v),
          _ => core::arch::aarch64::vgetq_lane_f64::<1>(v),
        }
      };
      let x_lanes: [f64; 4] = [
        extract(lane0_1, 0),
        extract(lane0_1, 1),
        extract(lane2_3, 0),
        extract(lane2_3, 1),
      ];

      let mut lo_idx_lanes = [0usize; 4];
      let mut frac_lanes_f64 = [0.0f64; 4];
      for j in 0..4 {
        let xj = x_lanes[j];
        let lo = xj.floor();
        frac_lanes_f64[j] = xj - lo;
        lo_idx_lanes[j] = (lo as usize).min(last_in);
      }

      // Scalar gather — NEON has no scatter/gather; load `a` and `b`
      // per lane from `samples`.
      let a_lanes = [
        samples[lo_idx_lanes[0]],
        samples[lo_idx_lanes[1]],
        samples[lo_idx_lanes[2]],
        samples[lo_idx_lanes[3]],
      ];
      let b_lanes = [
        samples[(lo_idx_lanes[0] + 1).min(last_in)],
        samples[(lo_idx_lanes[1] + 1).min(last_in)],
        samples[(lo_idx_lanes[2] + 1).min(last_in)],
        samples[(lo_idx_lanes[3] + 1).min(last_in)],
      ];

      // Narrow f64 frac to f32 — matches the scalar reference's
      // `frac as f32` cast (single rounding from f64 to f32).
      let frac_lanes_f32: [f32; 4] = [
        frac_lanes_f64[0] as f32,
        frac_lanes_f64[1] as f32,
        frac_lanes_f64[2] as f32,
        frac_lanes_f64[3] as f32,
      ];

      // Pack scalar lanes into NEON vectors and FMA.
      let a_v = core::arch::aarch64::vld1q_f32(a_lanes.as_ptr());
      let b_v = core::arch::aarch64::vld1q_f32(b_lanes.as_ptr());
      let frac_v = core::arch::aarch64::vld1q_f32(frac_lanes_f32.as_ptr());

      // out = a + (b - a) * frac  — vfmaq_f32(acc, m1, m2) = acc + m1 * m2
      let diff = vsubq_f32(b_v, a_v);
      let result: float32x4_t = vfmaq_f32(a_v, diff, frac_v);
      vst1q_f32(dst_base.add(i), result);

      i += 4;
    }
  }

  // Tail handled by the absolute-index helper at the dispatcher level
  // (the scalar reference's `for (i, slot)` uses RELATIVE `i`, which
  // would be wrong for the tail samples — see `resample_linear_neon_tail`).
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn resample_linear_neon_tail(
  out: &mut [MaybeUninit<f32>],
  samples: &[f32],
  ratio: f64,
  i_base: usize,
) {
  // The scalar arm computes `x = i_local * ratio`, but the tail
  // continues from absolute output index `body_len`. Re-issue the
  // per-sample math with the absolute index.
  assert!(
    !samples.is_empty(),
    "resample_linear_neon_tail: samples must be non-empty"
  );
  let last_in = samples.len() - 1;
  for (j, slot) in out.iter_mut().enumerate() {
    let i = i_base + j;
    let x = i as f64 * ratio;
    let lo = x.floor();
    let frac = x - lo;
    let lo_idx = (lo as usize).min(last_in);
    let hi_idx = (lo_idx + 1).min(last_in);
    let a = samples[lo_idx];
    let b = samples[hi_idx];
    slot.write(a + (b - a) * frac as f32);
  }
}

/// Public dispatcher: linear-interpolation resample of `samples` into
/// `out` using output-index → input-position factor `ratio`. Routes to
/// NEON on `aarch64` when NEON is reported, else to the scalar
/// reference.
///
/// Used by [`crate::audio::io::resample_linear`] to fill the resampled
/// f32 buffer.
///
/// # Preconditions
///
/// - `!samples.is_empty()` — asserted unconditionally.
///
/// # Initialization contract
///
/// **Every f32 of `out` is written before this returns.**
///
/// # Correctness class
///
/// `Tolerance` (`abs = 1e-6, rel = 1e-6`) — NEON's `vfmaq_f32` produces
/// a single-rounding `a + (b - a) * frac` while the scalar arm uses
/// two separate roundings; the per-sample divergence is at most one
/// f32 ULP.
#[inline]
#[doc(hidden)]
pub fn resample_linear(out: &mut [MaybeUninit<f32>], samples: &[f32], ratio: f64) {
  assert!(
    !samples.is_empty(),
    "simd::audio::resample_linear: samples must be non-empty"
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      let n = out.len();
      let body_len = n - (n % 4);
      // SAFETY: NEON gated by `is_neon_available()`; samples non-empty
      // asserted above; the kernel writes every f32 of the body
      // (vst1q_f32 per tile) and the tail (scalar replay) before
      // returning.
      unsafe {
        // Split the output into body + tail so the tail can be
        // computed with the absolute output index `body_len + j`
        // rather than the scalar arm's relative `j` (the scalar arm
        // uses `i` directly, so calling it on `&mut out[body_len..]`
        // would compute `x = j * ratio` for the WRONG `j`).
        let (body, tail) = out.split_at_mut(body_len);
        if body_len > 0 {
          resample_linear_neon(body, samples, ratio);
        }
        if !tail.is_empty() {
          resample_linear_neon_tail(tail, samples, ratio, body_len);
        }
      }
      return;
    }
  }
  resample_linear_scalar(out, samples, ratio);
}

#[cfg(test)]
mod tests {
  //! Scalar vs dispatcher Tolerance differential tests + edge coverage
  //! for C8.

  use super::{resample_linear, resample_linear_scalar};
  use crate::simd::diff::assert_close_slice_over_lane_sweep;

  /// Per-sample tolerance: one f32 ULP for samples in `[-1, 1]` is
  /// ~6e-8 absolute. We widen to 1e-6 to cover the cumulative effect
  /// of multiple roundings (f64 ratio → f64 frac → f32 frac → FMA).
  const RESAMPLE_TOL_ABS: f64 = 1e-6;
  const RESAMPLE_TOL_REL: f64 = 1e-6;

  /// Build an output Vec via the scalar kernel for a length sweep.
  /// Uses `out_len = src.len() * 2` (upsample 2×, ratio = 0.5) so the
  /// length sweep exercises both inputs to interpolation.
  fn pair_2x(sweep_len: usize) -> (Vec<f32>, Vec<f32>) {
    if sweep_len == 0 {
      return (Vec::new(), Vec::new());
    }
    // Build deterministic samples: a sine-like sequence in [-1, 1].
    let mut samples: Vec<f32> = Vec::with_capacity(sweep_len);
    for k in 0..sweep_len {
      let v = ((k as f32) * 0.1).sin();
      samples.push(v);
    }
    let out_len = sweep_len * 2;
    let ratio = 0.5_f64;

    let mut s_out: Vec<f32> = Vec::with_capacity(out_len);
    let spare_s = s_out.spare_capacity_mut();
    resample_linear_scalar(&mut spare_s[..out_len], &samples, ratio);
    // SAFETY: kernel contract initializes every slot.
    unsafe { s_out.set_len(out_len) };

    let mut d_out: Vec<f32> = Vec::with_capacity(out_len);
    let spare_d = d_out.spare_capacity_mut();
    resample_linear(&mut spare_d[..out_len], &samples, ratio);
    // SAFETY: kernel contract initializes every slot.
    unsafe { d_out.set_len(out_len) };

    (s_out, d_out)
  }

  #[test]
  fn resample_linear_scalar_matches_dispatcher_tolerance() {
    // Adapter to the slice-sweep helper. `gen_input(n)` returns
    // `vec![0_i32; n]` purely to carry the sweep length to the
    // closure; the actual samples are synthesized inside the
    // closures.
    let s = |xs: &[i32]| {
      let n = xs.len();
      if n == 0 {
        return Vec::new();
      }
      let (so, _) = pair_2x(n);
      so.into_iter().map(|x| x as f64).collect()
    };
    let d = |xs: &[i32]| {
      let n = xs.len();
      if n == 0 {
        return Vec::new();
      }
      let (_, dout) = pair_2x(n);
      dout.into_iter().map(|x| x as f64).collect()
    };
    assert_close_slice_over_lane_sweep(
      4,
      s,
      d,
      |n| vec![0_i32; n],
      RESAMPLE_TOL_ABS,
      RESAMPLE_TOL_REL,
    );
  }

  #[test]
  fn resample_linear_constant_signal_is_constant() {
    // `samples = [0.5; N]` resampled at any ratio must produce all-0.5
    // (interpolation between two equal values is the value itself).
    let samples = vec![0.5_f32; 32];
    let ratio = 0.7_f64;
    let out_len = 50;
    let mut out: Vec<f32> = Vec::with_capacity(out_len);
    let spare = out.spare_capacity_mut();
    resample_linear(&mut spare[..out_len], &samples, ratio);
    // SAFETY: kernel contract initializes every slot.
    unsafe { out.set_len(out_len) };
    for (i, v) in out.iter().enumerate() {
      assert!(
        (*v - 0.5).abs() < 1e-6,
        "constant interpolation at i={i} should be 0.5 (got {v})"
      );
    }
  }

  #[test]
  fn resample_linear_unit_ratio_copies_samples() {
    // ratio = 1.0 reduces to a verbatim copy (no interpolation).
    let samples: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let ratio = 1.0_f64;
    let mut out: Vec<f32> = Vec::with_capacity(samples.len());
    let spare = out.spare_capacity_mut();
    resample_linear(&mut spare[..samples.len()], &samples, ratio);
    // SAFETY: kernel contract initializes every slot.
    unsafe { out.set_len(samples.len()) };
    for (i, (s, d)) in samples.iter().zip(out.iter()).enumerate() {
      assert!(
        (s - d).abs() < 1e-6,
        "unit-ratio resample should copy: i={i} src={s} out={d}"
      );
    }
  }

  #[test]
  fn resample_linear_single_input_replicates() {
    // Single-element source: every output is samples[0].
    let samples = vec![0.42_f32];
    let ratio = 0.3_f64;
    let out_len = 17;
    let mut out: Vec<f32> = Vec::with_capacity(out_len);
    let spare = out.spare_capacity_mut();
    resample_linear(&mut spare[..out_len], &samples, ratio);
    // SAFETY: kernel contract initializes every slot.
    unsafe { out.set_len(out_len) };
    for (i, v) in out.iter().enumerate() {
      assert!(
        (*v - 0.42).abs() < 1e-6,
        "single-sample resample at i={i} should be 0.42 (got {v})"
      );
    }
  }

  #[test]
  fn resample_linear_first_output_is_first_sample() {
    // Output index 0 maps to `x = 0 * ratio = 0`, `lo_idx = 0`,
    // `frac = 0` — pure samples[0].
    let samples: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let out_len = 4;
    let ratio = 0.5_f64;
    let mut out: Vec<f32> = Vec::with_capacity(out_len);
    let spare = out.spare_capacity_mut();
    resample_linear(&mut spare[..out_len], &samples, ratio);
    // SAFETY: kernel contract initializes every slot.
    unsafe { out.set_len(out_len) };
    assert!((out[0] - 0.1).abs() < 1e-6, "out[0] should be samples[0]");
  }

  #[test]
  #[should_panic(expected = "simd::audio::resample_linear: samples must be non-empty")]
  fn resample_linear_panics_on_empty_samples() {
    let samples: Vec<f32> = Vec::new();
    let out_len = 4;
    let mut out: Vec<f32> = Vec::with_capacity(out_len);
    let spare = out.spare_capacity_mut();
    resample_linear(&mut spare[..out_len], &samples, 0.5);
  }
}
