// Architecture and the f64 dot/reduction kernel adapted from the `dia`
// project (github — MIT/Apache-2.0), src/ops/.

//! Public dispatchers for the [`crate::simd`] primitives.
//!
//! Each dispatcher selects the best-available SIMD backend at runtime
//! (via `is_aarch64_feature_detected!` against the `aarch64`-gated
//! `crate::simd::arch` module), falling back to
//! [`crate::simd::scalar`] when no SIMD backend applies. Callers
//! needing scalar output explicitly call [`crate::simd::scalar`].

mod dot;

pub use dot::{dot, sum_of_squares};
