// Architecture and the f64 dot/reduction kernel adapted from the `dia`
// project (github — MIT/Apache-2.0), src/ops/.

//! Scalar f64 dot product and `sum_of_squares` reduction.
//!
//! Implementation matches the NEON kernel's reduction tree exactly so
//! the scalar and NEON paths are **bit-identical** on `aarch64`:
//!
//! - Per-element FMA via `f64::mul_add` (one IEEE 754 rounding, same
//!   as `vfmaq_f64`).
//! - Four partial accumulators over the modulo-4 residue classes,
//!   mirroring NEON's two 2-lane registers (`acc0[0]`, `acc0[1]`,
//!   `acc1[0]`, `acc1[1]`).
//! - Final reduction tree `((s00 + s10) + (s01 + s11))`, identical to
//!   NEON's `vaddq_f64 + vaddvq_f64` sequence.
//!
//! Result is bit-identical to the `crate::simd::arch::neon` kernels
//! (`dot` / `sum_of_squares`) for every input on `aarch64`. (There is
//! no x86 backend in mlxrs today; the dispatcher falls through to this
//! scalar reference on `x86_64`, and on `aarch64` under `--cfg
//! mlxrs_force_scalar`.)

/// Inner product of two equal-length f64 slices: `Σ a[i] * b[i]`.
///
/// # Panics
///
/// If `a.len() != b.len()`. Enforced **unconditionally** (in release
/// too), matching the public dispatcher [`crate::simd::dot`]: without
/// it, `dot(&short, &long)` would silently return the dot over the
/// shorter length, and `dot(&long, &short)` would panic via indexing
/// — an asymmetric, fail-quiet contract. The unconditional assert
/// makes this public scalar API reject mismatched input symmetrically.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
  assert_eq!(
    a.len(),
    b.len(),
    "scalar::dot: a.len() ({}) must equal b.len() ({})",
    a.len(),
    b.len()
  );
  let n = a.len();
  let mut s00 = 0.0_f64; // accumulates positions ≡ 0 mod 4
  let mut s01 = 0.0_f64; // ≡ 1 mod 4
  let mut s10 = 0.0_f64; // ≡ 2 mod 4
  let mut s11 = 0.0_f64; // ≡ 3 mod 4
  let mut i = 0usize;
  while i + 4 <= n {
    s00 = f64::mul_add(a[i], b[i], s00);
    s01 = f64::mul_add(a[i + 1], b[i + 1], s01);
    s10 = f64::mul_add(a[i + 2], b[i + 2], s10);
    s11 = f64::mul_add(a[i + 3], b[i + 3], s11);
    i += 4;
  }
  // 2-wide tail: NEON also FMAs into acc0 only.
  if i + 2 <= n {
    s00 = f64::mul_add(a[i], b[i], s00);
    s01 = f64::mul_add(a[i + 1], b[i + 1], s01);
    i += 2;
  }
  // Reduction tree matches NEON's `vaddq_f64(acc0, acc1)` then
  // `vaddvq_f64(acc) = acc[0] + acc[1]`.
  let mut sum = (s00 + s10) + (s01 + s11);
  // Final scalar tail for odd lengths.
  while i < n {
    sum = f64::mul_add(a[i], b[i], sum);
    i += 1;
  }
  sum
}

/// Sum of squares of an f64 slice: `Σ v[i]²`.
///
/// This is the `b ≡ a` specialization of [`dot`] — squaring and
/// accumulating each element. The reduction tree is identical to
/// [`dot`] (and to the `crate::simd::arch::neon::sum_of_squares` NEON
/// kernel), so on `aarch64` the scalar and NEON outputs are
/// bit-identical.
///
/// Used by `crate::audio`'s `integrated_loudness` for the per-block
/// K-weighted mean-square reduction.
#[inline]
pub fn sum_of_squares(v: &[f64]) -> f64 {
  let n = v.len();
  let mut s00 = 0.0_f64; // accumulates positions ≡ 0 mod 4
  let mut s01 = 0.0_f64; // ≡ 1 mod 4
  let mut s10 = 0.0_f64; // ≡ 2 mod 4
  let mut s11 = 0.0_f64; // ≡ 3 mod 4
  let mut i = 0usize;
  while i + 4 <= n {
    s00 = f64::mul_add(v[i], v[i], s00);
    s01 = f64::mul_add(v[i + 1], v[i + 1], s01);
    s10 = f64::mul_add(v[i + 2], v[i + 2], s10);
    s11 = f64::mul_add(v[i + 3], v[i + 3], s11);
    i += 4;
  }
  // 2-wide tail: NEON also FMAs into acc0 only.
  if i + 2 <= n {
    s00 = f64::mul_add(v[i], v[i], s00);
    s01 = f64::mul_add(v[i + 1], v[i + 1], s01);
    i += 2;
  }
  // Reduction tree matches NEON's `vaddq_f64(acc0, acc1)` then
  // `vaddvq_f64(acc) = acc[0] + acc[1]`.
  let mut sum = (s00 + s10) + (s01 + s11);
  // Final scalar tail for odd lengths.
  while i < n {
    sum = f64::mul_add(v[i], v[i], sum);
    i += 1;
  }
  sum
}
