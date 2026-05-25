//! Embedding normalization.
//!
//! Ported from `mlx-embeddings` `models/base.py::normalize_embeddings`
//! (`x / maximum(linalg.norm(x, ord=p, axis, keepdims), eps)`) and
//! `MLXEmbedders` `MLXArray+Helper.l2Normalized`.
//!
//! The general [`normalize`] is genuinely parameterized — the order `p`
//! is forwarded to `mlx.linalg.norm` (via [`crate::ops::linalg_full::norm`],
//! which wraps `mlx_linalg_norm`), so `p != 2` (L1, L∞, …) is a real
//! `mlx_linalg_norm` reduction, not a hand-rolled `sum(|x|^p)^(1/p)`.

use crate::{
  array::Array,
  error::Result,
  ops::{
    arithmetic::{divide, maximum},
    linalg_full::norm,
  },
};

use super::scalar_like;

/// python `mlx-embeddings` default normalization eps (`base.py`,
/// `normalize_embeddings(..., eps=1e-9)`). This is the project default:
/// `mlx-embeddings` is the primary embeddings reference.
pub const DEFAULT_NORMALIZE_EPS: f32 = 1e-9;

/// swift `MLXEmbedders` `l2Normalized` eps (`MLXArray+Helper.swift`,
/// `eps: Float = 1e-12`). Exposed for callers that want exact swift
/// parity; the crate default ([`DEFAULT_NORMALIZE_EPS`]) follows python.
pub const SWIFT_L2_EPS: f32 = 1e-12;

/// Parameterized vector normalization: `x / max(||x||_p, eps)`.
///
/// Mirrors `mlx-embeddings` `normalize_embeddings(embeddings, p, axis,
/// keepdims, eps)`. The norm is computed by `mlx.linalg.norm` (real
/// `ord=p` reduction, not a hand-rolled p-norm), then clamped from below
/// by `eps` (clamp-then-divide, matching both references — more stable
/// than `norm + eps`).
///
/// - `p`: norm order forwarded to `mlx_linalg_norm` (`2.0` = L2,
///   `1.0` = L1, `f64::INFINITY` = L∞, …).
/// - `axis`: reduction axis (python default `-1`).
/// - `keepdims`: keep the reduced axis (python default `true`); must be
///   `true` for the divide to broadcast back over `x` unless `axis` is
///   the last dim and `x` is 1-D.
/// - `eps`: divide-by-zero floor. Pass [`DEFAULT_NORMALIZE_EPS`] for the
///   python default or [`SWIFT_L2_EPS`] for swift parity.
pub fn normalize(x: &Array, p: f64, axis: i32, keepdims: bool, eps: f32) -> Result<Array> {
  let n = norm(x, p, &[axis], keepdims)?;
  // python `mx.maximum(linalg.norm(x, ...), eps)`: `eps` is a Python
  // scalar → an MLX *weak* scalar that adopts the norm/array dtype, NOT a
  // forced-f32 `Array`. Build the floor in `n.dtype()` (= `x.dtype()`,
  // `norm` is dtype-preserving) so a f16/bf16 `x` is not promoted to f32;
  // for f32 `x` this is a no-op cast → bit-identical to the prior
  // `Array::full::<f32>`. The `divide` then stays in the input dtype.
  //
  // C4/C5 (Copilot review 4307622782, #3256688284 / #3256688269): for a
  // f16 (or bf16) input the default `eps = 1e-9` ([`DEFAULT_NORMALIZE_-
  // EPS`]) is BELOW the half subnormal floor, so the dtype-cast weak
  // scalar rounds to `0.0` and an all-zero f16 vector normalizes to
  // `0/0 = NaN`. This is NOT a defect: it is the EXACT python
  // `mlx-embeddings` / MLX weak-scalar per-dtype behavior (`mx.maximum`'s
  // python-scalar eps adopts the array dtype and underflows there too),
  // verified against the reference and explicitly asserted by
  // `tests/embeddings.rs::normalize_zero_vector_f16_bf16_eps_floor_in_-
  // dtype`. Flooring in f32 here would DIVERGE from the python reference
  // (violates the standing "match the reference" rule), so it is
  // deliberately preserved. This is *intentionally distinct* from the
  // scalar [`super::cosine_similarity`] convenience, which has NO python
  // reference and therefore guards its final divide in f32 to guarantee
  // finite `0.0` for all dtypes (C3). The two are intentionally
  // different (reference-faithful vs. mlxrs-only convenience), not
  // contradictory.
  let floor = scalar_like(eps, &n)?;
  let n = maximum(&n, &floor)?;
  divide(x, &n)
}

/// L2-normalize along the last axis: `x / max(||x||_2, eps)`.
///
/// Convenience for `normalize(x, 2.0, -1, true, eps)`. `eps` defaults to
/// the python [`DEFAULT_NORMALIZE_EPS`] (`1e-9`) — note swift's
/// `l2Normalized` uses `1e-12` ([`SWIFT_L2_EPS`]); pass that explicitly
/// for byte-exact swift parity. Mirrors `mlx-embeddings`
/// `base.normalize_embeddings` (`p=2`, `axis=-1`, `keepdims=True`).
pub fn l2_normalize_eps(embeddings: &Array, eps: f32) -> Result<Array> {
  normalize(embeddings, 2.0, -1, true, eps)
}

/// L2-normalize along the last axis with the python default eps
/// (`1e-9`). Back-compat shim for the pre-existing public API; identical
/// to `l2_normalize_eps(embeddings, DEFAULT_NORMALIZE_EPS)`.
pub fn l2_normalize(embeddings: &Array) -> Result<Array> {
  l2_normalize_eps(embeddings, DEFAULT_NORMALIZE_EPS)
}
