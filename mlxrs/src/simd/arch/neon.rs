// Architecture and the f64 dot/reduction kernel adapted from the `dia`
// project (github — MIT/Apache-2.0), src/ops/.

//! aarch64 NEON f64 dot product and `sum_of_squares`.
//!
//! 2-lane FMA over `float64x2_t`. Two parallel accumulators hide FMA
//! latency on cores where dependent FMAs serialize — the common case
//! on Apple silicon and Cortex-A series.
//!
//! Each `pub(crate) unsafe fn` is annotated `#[target_feature(enable =
//! "neon")]` and assumes the caller has verified NEON availability via
//! [`crate::simd::neon_available`]. NEON is part of the AArch64
//! baseline so this is essentially always-on, but the explicit gate
//! keeps the dispatcher pattern symmetric and makes the scalar
//! fallback a real, tested branch.
//!
//! The reduction tree (two 2-lane `float64x2_t` accumulators →
//! `vaddq_f64` → `vaddvq_f64`, then a scalar `f64::mul_add` tail)
//! mirrors [`crate::simd::scalar`]'s 4-accumulator tree exactly, so
//! the scalar and NEON outputs are **bit-identical** for every input.

use core::arch::aarch64::{float64x2_t, vaddq_f64, vaddvq_f64, vdupq_n_f64, vfmaq_f64, vld1q_f64};

/// `Σ a[i] * b[i]`. NEON 2-lane f64.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. This is the
///    caller's obligation — the public dispatcher [`crate::simd::dot`]
///    discharges it via [`crate::simd::neon_available`].
/// 2. `a.len() == b.len()`. The dispatcher asserts this
///    *unconditionally* (a release mismatch would OOB-read `b`); it is
///    debug-asserted here.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot(a: &[f64], b: &[f64]) -> f64 {
  debug_assert_eq!(a.len(), b.len(), "neon::dot: length mismatch");
  let n = a.len();

  // SAFETY: every `vld1q_f64` reads 2 contiguous f64; the pointer adds
  // are bounded by the loop conditions (`i + 4 <= n` / `i + 2 <= n`)
  // and the caller-promised `a.len() == b.len()`. The scalar-tail
  // `get_unchecked` indices satisfy `i < n`.
  unsafe {
    let mut acc0: float64x2_t = vdupq_n_f64(0.0);
    let mut acc1: float64x2_t = vdupq_n_f64(0.0);
    let mut i = 0usize;
    // 4-wide unroll (2 NEON regs × 2 lanes) to hide FMA latency.
    while i + 4 <= n {
      let a0 = vld1q_f64(a.as_ptr().add(i));
      let b0 = vld1q_f64(b.as_ptr().add(i));
      let a1 = vld1q_f64(a.as_ptr().add(i + 2));
      let b1 = vld1q_f64(b.as_ptr().add(i + 2));
      acc0 = vfmaq_f64(acc0, a0, b0);
      acc1 = vfmaq_f64(acc1, a1, b1);
      i += 4;
    }
    // 2-wide tail.
    if i + 2 <= n {
      let a0 = vld1q_f64(a.as_ptr().add(i));
      let b0 = vld1q_f64(b.as_ptr().add(i));
      acc0 = vfmaq_f64(acc0, a0, b0);
      i += 2;
    }
    let acc = vaddq_f64(acc0, acc1);
    let mut sum = vaddvq_f64(acc);
    // Scalar tail must FMA each element directly into `sum` via
    // `f64::mul_add` — one rounding, matching `simd::scalar::dot`'s
    // final loop. Routing through `scalar::dot` for the tail would
    // compute its own sum (one rounding) and then `sum += that` (a
    // second rounding), drifting by ½ ulp on lengths ≡ 1 or 3 mod 4
    // and breaking the bit-identical contract.
    while i < n {
      sum = f64::mul_add(*a.get_unchecked(i), *b.get_unchecked(i), sum);
      i += 1;
    }
    sum
  }
}

/// `Σ v[i]²`. NEON 2-lane f64.
///
/// The `b ≡ a` specialization of [`dot`]: `vfmaq_f64(acc, v, v)`
/// squares and accumulates a 2-lane f64 vector in one instruction.
/// The reduction tree is identical to [`dot`] and to
/// [`crate::simd::scalar::sum_of_squares`].
///
/// # Safety
///
/// NEON must be available on the executing CPU — the caller's
/// obligation, discharged by the public dispatcher
/// [`crate::simd::sum_of_squares`] via
/// [`crate::simd::neon_available`]. (There is no slice-length
/// precondition: the kernel reads only `v` itself.)
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn sum_of_squares(v: &[f64]) -> f64 {
  let n = v.len();

  // SAFETY: every `vld1q_f64` reads 2 contiguous f64; the pointer adds
  // are bounded by the loop conditions (`i + 4 <= n` / `i + 2 <= n`).
  // The scalar-tail `get_unchecked` index satisfies `i < n`.
  unsafe {
    let mut acc0: float64x2_t = vdupq_n_f64(0.0);
    let mut acc1: float64x2_t = vdupq_n_f64(0.0);
    let mut i = 0usize;
    // 4-wide unroll (2 NEON regs × 2 lanes) to hide FMA latency.
    while i + 4 <= n {
      let v0 = vld1q_f64(v.as_ptr().add(i));
      let v1 = vld1q_f64(v.as_ptr().add(i + 2));
      acc0 = vfmaq_f64(acc0, v0, v0);
      acc1 = vfmaq_f64(acc1, v1, v1);
      i += 4;
    }
    // 2-wide tail.
    if i + 2 <= n {
      let v0 = vld1q_f64(v.as_ptr().add(i));
      acc0 = vfmaq_f64(acc0, v0, v0);
      i += 2;
    }
    let acc = vaddq_f64(acc0, acc1);
    let mut sum = vaddvq_f64(acc);
    // Scalar tail: one `f64::mul_add` rounding per element, matching
    // `simd::scalar::sum_of_squares`'s final loop (see `dot` above for
    // why recursing into the scalar kernel would double-round).
    while i < n {
      let x = *v.get_unchecked(i);
      sum = f64::mul_add(x, x, sum);
      i += 1;
    }
    sum
  }
}
