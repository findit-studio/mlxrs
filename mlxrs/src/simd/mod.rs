// Architecture and the f64 dot/reduction kernel adapted from the `dia`
// project (github — MIT/Apache-2.0), src/ops/.

//! Hand-written `core::arch` SIMD kernels for the host-CPU numeric
//! loops mlxrs runs *itself* (not through MLX FFI).
//!
//! The overwhelming majority of mlxrs tensor math is delegated to MLX
//! and runs on MLX's own SIMD/Metal kernels — those are out of scope
//! here. This module covers the small set of Rust-side `&[f64]` /
//! `&[f32]` loops that run on the CPU regardless (audio DSP,
//! preprocessing), where a `core::arch` kernel is a genuine win.
//!
//! ## Layered architecture
//!
//! Mirrors the `dia` project's `src/ops/` four-layer shape:
//!
//! - [`scalar`](crate::simd::scalar) — bit-exact scalar reference
//!   kernels. **Always compiled**, independent of `target_arch`. The
//!   math contract is anchored here; it is also the differential-test
//!   oracle and the fallback path.
//! - `arch` — architecture-specific SIMD backends, gated behind
//!   `#[cfg(target_arch = "aarch64")]` (so not linkable from these
//!   always-rendered docs). `arch::neon` holds
//!   `#[target_feature(enable = "neon")] unsafe fn` kernels.
//! - `dispatch` — runtime-detection routers. Each public dispatcher
//!   asserts its slice-length preconditions **unconditionally**, then
//!   picks NEON (if available) or the scalar fallback.
//! - this module (`simd`) — module doc + the public dispatcher
//!   re-exports + the [`neon_available`](crate::simd::neon_available)
//!   detector.
//!
//! ## Public surface
//!
//! - [`dot`](crate::simd::dot) — `Σ a[i] * b[i]`, f64.
//! - [`sum_of_squares`](crate::simd::sum_of_squares) — `Σ v[i]²`,
//!   f64. Used by [`crate::audio`]'s `integrated_loudness` for the
//!   per-block K-weighted mean-square.
//!
//! ## Always on — no cargo feature
//!
//! SIMD is **unconditional**: there is no `simd` cargo feature. Whether
//! the NEON backend runs is gated purely on `#[cfg(target_arch =
//! "aarch64")]` plus runtime CPU detection
//! ([`neon_available`](crate::simd::neon_available)); on every other
//! target the dispatchers route to [`scalar`](crate::simd::scalar)
//! automatically. The [`scalar`](crate::simd::scalar) and `dispatch`
//! layers therefore compile on **all** targets — only the `arch`
//! module is `aarch64`-gated (the
//! [`neon_available`](crate::simd::neon_available) detector is a
//! `const false` stub elsewhere).
//! This matches the `dia` reference (no simd feature). A pure-scalar
//! build for bisecting a numeric regression — even on a NEON-capable
//! host — is available via the `--cfg mlxrs_force_scalar` build escape
//! (see [`neon_available`](crate::simd::neon_available)).
//!
//! ## Cross-path determinism
//!
//! On `aarch64`, [`scalar`](crate::simd::scalar) and the
//! `arch::neon` kernels produce **bit-identical** results. This is
//! deliberate:
//!
//! 1. both use `f64::mul_add` for each per-element FMA — one IEEE 754
//!    rounding, identical to `vfmaq_f64`;
//! 2. [`scalar`](crate::simd::scalar)'s reduction tree mirrors NEON's
//!    — 4 partial sums over modulo-4 indices, combined
//!    `((s00 + s10) + (s01 + s11))`.
//!
//! Verified by the `differential_tests` module below
//! (`assert_eq!` on `f64::to_bits()`).

#[cfg(target_arch = "aarch64")]
pub mod arch;
mod dispatch;
pub mod scalar;

pub use dispatch::{dot, sum_of_squares};

// ─── runtime CPU-feature detection ───────────────────────────────────
//
// `--cfg mlxrs_force_scalar` overrides detection so the scalar path
// can be exercised even on a NEON-capable host — set it via
// `RUSTFLAGS="--cfg mlxrs_force_scalar"`.

/// Whether the NEON SIMD backend is usable on the executing CPU.
///
/// `true` when NEON is reported by the CPU **and** `--cfg
/// mlxrs_force_scalar` is not set. NEON is part of the AArch64
/// baseline, so on a normal `aarch64` host this is effectively always
/// `true`; the explicit check keeps the scalar fallback a real,
/// reachable branch and honours the force-scalar escape.
///
/// On every non-`aarch64` target there is no NEON backend to gate, so
/// this is a `const false` stub: every dispatcher then routes to
/// [`scalar`]. The stub keeps the symbol present on all targets so
/// intra-doc links resolve in a non-`aarch64` rustdoc build.
#[cfg(target_arch = "aarch64")]
pub fn neon_available() -> bool {
  if cfg!(mlxrs_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

/// Whether the NEON SIMD backend is usable on the executing CPU.
///
/// `true` when NEON is reported by the CPU **and** `--cfg
/// mlxrs_force_scalar` is not set. NEON is part of the AArch64
/// baseline, so on a normal `aarch64` host this is effectively always
/// `true`; the explicit check keeps the scalar fallback a real,
/// reachable branch and honours the force-scalar escape.
///
/// On every non-`aarch64` target there is no NEON backend to gate, so
/// this is a `const false` stub: every dispatcher then routes to
/// [`scalar`]. The stub keeps the symbol present on all targets so
/// intra-doc links resolve in a non-`aarch64` rustdoc build.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_available() -> bool {
  false
}

#[cfg(test)]
mod differential_tests {
  //! Scalar vs NEON differential tests.
  //!
  //! Contract: on `aarch64` (the deployment target), [`super::scalar`]
  //! and [`super::arch::neon`] produce **bit-identical** results for
  //! every primitive. Achieved by (1) `f64::mul_add` per-element FMA
  //! on both, and (2) a scalar reduction tree that mirrors NEON's
  //! (4 partial sums over modulo-4 indices, then `((s00 + s10) +
  //! (s01 + s11))`).
  //!
  //! On non-`aarch64` targets the dispatcher routes to scalar, so the
  //! differential check is a scalar-vs-scalar identity — still a
  //! useful regression guard, and trivially bit-equal.

  /// `sum_of_squares` scalar vs the SIMD dispatcher, over lengths that
  /// straddle the 2-lane / 4-wide-unroll boundaries (the scalar-tail
  /// paths). On `aarch64` the matched reduction tree makes this
  /// bit-identical — asserted on `to_bits()`.
  #[test]
  fn sum_of_squares_scalar_matches_simd() {
    // Lengths straddle the 4-wide unroll (mod 4 ∈ {0,1,2,3}) and the
    // 2-wide tail; 0 exercises the empty case.
    for n in [0usize, 1, 3, 7, 8, 9, 17, 33, 129, 1024] {
      // A spread of magnitudes / signs — fully deterministic (no rng
      // dep): a sign-alternating, slowly-growing sequence.
      let v: Vec<f64> = (0..n)
        .map(|i| {
          let mag = 0.5 + (i as f64) * 0.013_f64;
          if i % 2 == 0 { mag } else { -mag }
        })
        .collect();
      let s = super::scalar::sum_of_squares(&v);
      let d = super::sum_of_squares(&v);
      assert_eq!(
        s.to_bits(),
        d.to_bits(),
        "sum_of_squares n={n}: scalar/SIMD not bit-identical (s={s}, d={d})"
      );
    }
  }

  /// `dot` scalar vs the SIMD dispatcher, same length sweep. The
  /// `sum_of_squares` kernel is the `b ≡ a` specialization of `dot`;
  /// covering `dot` directly locks the general kernel's reduction
  /// tree too.
  #[test]
  fn dot_scalar_matches_simd() {
    for n in [0usize, 1, 3, 7, 8, 9, 17, 33, 129, 1024] {
      let a: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64) * 0.013_f64).collect();
      let b: Vec<f64> = (0..n)
        .map(|i| {
          let mag = 0.25 + (i as f64) * 0.007_f64;
          if i % 3 == 0 { -mag } else { mag }
        })
        .collect();
      let s = super::scalar::dot(&a, &b);
      let d = super::dot(&a, &b);
      assert_eq!(
        s.to_bits(),
        d.to_bits(),
        "dot n={n}: scalar/SIMD not bit-identical (s={s}, d={d})"
      );
    }
  }

  /// A self-dot (`Σ a[i]²` via `dot(a, a)`) must equal
  /// `sum_of_squares(a)` bit-for-bit — the specialization claim. Both
  /// run the same reduction tree; this pins that they agree.
  #[test]
  fn sum_of_squares_equals_self_dot() {
    for n in [0usize, 1, 7, 8, 33, 129] {
      let a: Vec<f64> = (0..n)
        .map(|i| {
          let mag = 0.5 + (i as f64) * 0.011_f64;
          if i % 2 == 0 { mag } else { -mag }
        })
        .collect();
      let via_dot = super::dot(&a, &a);
      let via_ss = super::sum_of_squares(&a);
      assert_eq!(
        via_dot.to_bits(),
        via_ss.to_bits(),
        "n={n}: sum_of_squares != dot(a, a) (dot={via_dot}, ss={via_ss})"
      );
    }
  }

  /// Large-magnitude / catastrophic-cancellation-style band test.
  /// `dia` does the analogous check on `dot`. `sum_of_squares` cannot
  /// cancel (every term `v²` is ≥ 0), but a `1e16` element dwarfs the
  /// unit-scale ones — reduction order still matters. The matched
  /// scalar/NEON trees keep them bit-identical; the absolute value is
  /// pinned so a future kernel rewrite that changes the order
  /// surfaces here.
  #[test]
  fn sum_of_squares_large_magnitude_band() {
    let v: Vec<f64> = vec![1e16, 1.0, 1.0, 1.0, 1e16, 1.0, 1.0];
    let s = super::scalar::sum_of_squares(&v);
    let d = super::sum_of_squares(&v);
    assert_eq!(
      s.to_bits(),
      d.to_bits(),
      "large-magnitude sum_of_squares: scalar/SIMD not bit-identical (s={s}, d={d})"
    );
    // Two 1e16 terms dominate: `2 * (1e16)² = 2e32`. The five unit
    // terms (`+5`) are far below the f64 ulp at 2e32 and are lost —
    // this is expected, not a bug; it just pins the reduction's
    // large-magnitude behaviour.
    assert!(
      (s - 2e32).abs() < 1e17,
      "sum_of_squares large-magnitude result drifted from ~2e32: {s}"
    );
  }

  /// Mismatched `dot` lengths must `panic!` — not OOB-read. The
  /// dispatcher asserts `a.len() == b.len()` **unconditionally**
  /// before routing to the unsafe NEON kernel; this test would
  /// silently read past `b` if that guard were `debug_assert!`-only.
  #[test]
  #[should_panic(expected = "simd::dot")]
  fn dot_dispatch_panics_on_length_mismatch() {
    let a = vec![1.0_f64; 8];
    let b = vec![1.0_f64; 4];
    let _ = super::dot(&a, &b);
  }

  /// `scalar::dot` (the public scalar API, not the dispatcher) must
  /// `panic!` on mismatched lengths with the **shorter slice first**.
  /// Pre-fix this branch used a `debug_assert!`, so in release it
  /// silently returned the dot over `a.len()` (the shorter length) —
  /// a wrong result, no panic. The unconditional `assert_eq!` makes
  /// it fail loud, matching the dispatcher.
  #[test]
  #[should_panic(expected = "scalar::dot")]
  fn scalar_dot_panics_on_length_mismatch_shorter_first() {
    let a = vec![1.0_f64; 4];
    let b = vec![1.0_f64; 8];
    let _ = super::scalar::dot(&a, &b);
  }

  /// Symmetric case: `scalar::dot` with the **longer slice first**.
  /// This direction always panicked (out-of-bounds indexing of `b`),
  /// but it now panics via the explicit length `assert_eq!` instead —
  /// the contract is symmetric, fail-loud in both orders.
  #[test]
  #[should_panic(expected = "scalar::dot")]
  fn scalar_dot_panics_on_length_mismatch_longer_first() {
    let a = vec![1.0_f64; 8];
    let b = vec![1.0_f64; 4];
    let _ = super::scalar::dot(&a, &b);
  }
}
