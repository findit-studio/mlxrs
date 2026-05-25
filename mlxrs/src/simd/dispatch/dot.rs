// Architecture and the f64 dot/reduction kernel adapted from the `dia`
// project (github — MIT/Apache-2.0), src/ops/.

//! Dot product and `sum_of_squares` dispatchers.
//!
//! Each public fn routes to the best available SIMD backend on this
//! `target_arch` after runtime CPU-feature detection, falling back to
//! [`crate::simd::scalar`] when no SIMD backend applies (non-`aarch64`
//! targets or `--cfg mlxrs_force_scalar`).

use crate::simd::scalar;
#[cfg(target_arch = "aarch64")]
use crate::simd::{arch, neon_available};

/// Inner product of two equal-length f64 slices: `Σ a[i] * b[i]`.
///
/// Routes to NEON on `aarch64` (when the CPU reports NEON), else to
/// [`crate::simd::scalar::dot`]. Callers needing byte-identical scalar
/// output across every build configuration call
/// [`crate::simd::scalar::dot`] directly.
///
/// # Panics
///
/// If `a.len() != b.len()`. This is enforced **unconditionally** — the
/// NEON kernel reads raw pointers bounded only by `a.len()` and would
/// otherwise load past the end of `b` in release builds, where its
/// `debug_assert!` is a no-op.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
  assert_eq!(
    a.len(),
    b.len(),
    "simd::dot: a.len() ({}) must equal b.len() ({})",
    a.len(),
    b.len()
  );
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: `neon_available()` confirmed NEON is on this CPU.
      // `a.len() == b.len()` was asserted unconditionally above —
      // the kernel's debug-asserted length precondition holds.
      return unsafe { arch::neon::dot(a, b) };
    }
  }
  scalar::dot(a, b)
}

/// Sum of squares of an f64 slice: `Σ v[i]²`.
///
/// The `b ≡ a` specialization of [`dot`]. Routes to NEON on `aarch64`
/// (when the CPU reports NEON), else to
/// [`crate::simd::scalar::sum_of_squares`].
///
/// On `aarch64` the NEON and scalar paths produce **bit-identical**
/// results — both use a `f64::mul_add` per-element FMA and the same
/// 4-accumulator reduction tree. There is no slice-length
/// precondition (a sum over a single slice cannot be mismatched), so
/// this dispatcher cannot panic.
#[inline]
pub fn sum_of_squares(v: &[f64]) -> f64 {
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: `neon_available()` confirmed NEON is on this CPU. The
      // kernel has no slice-length precondition — it reads only `v`.
      return unsafe { arch::neon::sum_of_squares(v) };
    }
  }
  scalar::sum_of_squares(v)
}
