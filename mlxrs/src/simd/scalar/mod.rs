// Architecture and the f64 dot/reduction kernel adapted from the `dia`
// project (github — MIT/Apache-2.0), src/ops/.

//! Scalar reference implementations of the [`crate::simd`] primitives.
//!
//! **Always compiled** — independent of `target_arch` (there is no
//! `simd` cargo feature). The scalar path is:
//!
//! 1. the *algorithmic* contract — same math, same input handling;
//! 2. the differential-test oracle;
//! 3. the fallback path on non-`aarch64` targets, or when `--cfg
//!    mlxrs_force_scalar` is set.
//!
//! The reduction kernels here are written to mirror the NEON kernels'
//! reduction tree (4 partial accumulators over modulo-4 residues, then
//! `((s00 + s10) + (s01 + s11))`) and use `f64::mul_add` for each
//! per-element FMA. The consequence is that on `aarch64` the scalar
//! and NEON outputs are **bit-identical** — see the
//! `crate::simd::arch::neon` kernels and the differential tests in
//! [`crate::simd`].

mod dot;

pub use dot::{dot, sum_of_squares};
