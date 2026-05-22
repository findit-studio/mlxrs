// Architecture and the f64 dot/reduction kernel adapted from the `dia`
// project (github — MIT/Apache-2.0), src/ops/.

//! Architecture-specific SIMD backends for the [`crate::simd`]
//! primitives.
//!
//! This whole module is compiled only under `#[cfg(target_arch =
//! "aarch64")]` — the single backend that exists today is NEON.
//! Backends supply f64 outputs **bit-identical** to
//! [`crate::simd::scalar`] — the correctness contract is anchored by
//! the scalar reference and verified by the differential tests in
//! [`crate::simd`].
//!
//! Coverage:
//! - NEON (`aarch64`): `dot`, `sum_of_squares` (f64×2 lanes, FMA).
//!
//! There is no x86_64 backend today — `x86_64` builds route through
//! [`crate::simd::scalar`]. mlxrs ships `aarch64-darwin` as the default
//! target; an AVX2 tier can be added here later behind the same
//! dispatcher shape.

pub(crate) mod neon;
