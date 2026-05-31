//! Cosine similarity helpers.
//!
//! The matrix form mirrors `mlx-embeddings` usage
//! (`mx.matmul(embeddings, embeddings.T)` over L2-normalized rows).

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Error, LengthMismatchPayload, RankMismatchPayload, Result},
  ops::{
    arithmetic::{abs, divide, multiply, sqrt, square},
    comparison::equal,
    linalg_basic::matmul,
    logical::{logical_or, select},
    misc::astype,
    reduction::{max, sum},
    shape::transpose_axes,
  },
};

use super::{normalize::l2_normalize, scalar_like};

/// Validate the scalar [`cosine_similarity`] rank/length contract *before*
/// any arithmetic, so a wrong-rank caller gets a recoverable
/// [`Error::RankMismatch`] and an unequal-length caller gets
/// [`Error::LengthMismatch`] instead of a *silently broadcast*
/// (mathematically invalid) score. Mirrors the `pooling.rs`
/// `validate_token_embeddings_*` panic→`Err` precondition style.
///
/// Without this, MLX broadcasting lets e.g. `a=(3,)` against `b=(1,)`
/// produce a "cosine" `> 1` (the dot broadcasts `b`, but `||b||_2` is the
/// 1-element norm) — silent retrieval-ranking corruption on a dim/config
/// mismatch. Requires both `a` and `b` rank-1 with equal length. No
/// behavior change for valid equal-length 1-D inputs.
fn validate_cosine_similarity_vectors(a: &Array, b: &Array) -> Result<()> {
  let a_shape = a.shape();
  let b_shape = b.shape();
  if a_shape.len() != 1 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "cosine_similarity: `a` must be a rank-1 vector",
      a_shape.len() as u32,
      a_shape,
    )));
  }
  if b_shape.len() != 1 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "cosine_similarity: `b` must be a rank-1 vector",
      b_shape.len() as u32,
      b_shape,
    )));
  }
  if a_shape[0] != b_shape[0] {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "cosine_similarity: `a` and `b` lengths",
      a_shape[0],
      b_shape[0],
    )));
  }
  Ok(())
}

/// Cosine similarity of two 1-D vectors:
/// `dot(a, b) / (||a||_2 * ||b||_2)`, computed with a **numerically-stable
/// max-abs-scaled** formulation.
///
/// Identical vectors give `≈ 1.0`, orthogonal vectors `≈ 0.0`. Both inputs
/// must be 1-D with the same length; a non-rank-1 input or an unequal
/// length (which MLX would otherwise *silently broadcast* into an invalid
/// score) returns [`Err(Error::LengthMismatch)`](Error::LengthMismatch)
/// instead. A valid length-0 input (the rank/length validator treats
/// `(0,)` vs `(0,)` as equal-length rank-1) short-circuits to a finite
/// `0.0` (the empty-vector contract; no reduction over an empty array).
///
/// Accepts any float dtype (`f32`/`f16`/`bf16`); inputs are widened to
/// `f32` (lossless: f16/bf16→f32, a no-op for f32) and the result is
/// returned as `f32`.
///
/// ## Why max-abs scaling
/// A naïve `sqrt(sum(square(x)))` norm **underflows to `0`** for genuinely
/// tiny nonzero vectors (`square(1e-23) = 1e-46 → 0` in f32; the half
/// dtypes underflow far sooner) and **overflows to `+Inf`** for huge ones
/// (`square(f32::MAX) = +Inf`), so any zero/result predicate *derived from
/// that norm* misclassifies tiny nonzero vectors as zero or leaks
/// `0*Inf = NaN`. This fn instead first scales each vector by its
/// **max-abs (Chebyshev / ∞-norm)** `s = max(|x|)`:
/// - `s` is computed with `abs` + a full `max`-reduce only — **no
///   `square`**, so it is *exact* and free of underflow/overflow:
///   `s == 0.0` **iff** every element is *exactly* `0`.
/// - The **exact zero predicate** is therefore `max(|a|) == 0 ∨
///   max(|b|) == 0` (`logical_or(equal(s_a,0), equal(s_b,0))`), evaluated
///   on the max-abs scalars directly. It is `NaN`-free and **cannot** be
///   triggered by a nonzero vector no matter how tiny — that is the whole
///   point versus the prior L2-norm-derived predicate (which underflowed
///   `square` and misclassified tiny vectors, or produced `0*Inf = NaN`).
/// - After dividing by a div-by-zero-safe scale (`1.0` substituted where
///   the zero predicate holds, so the materialized branch never divides by
///   `0`), every scaled element has `|x̂_i| ≤ 1` with the max-magnitude
///   element exactly `1.0`, so `sum(square(x̂)) ∈ [1, n]` and
///   `‖x̂‖₂ ∈ [1, sqrt(n)]`: **no underflow to `0`, no overflow to
///   `+Inf`**. The dot/norms are thus computed entirely on `O(1)`-magnitude
///   data and `‖â‖₂·‖b̂‖₂` is bounded well away from both `0` and `+Inf`
///   for any realistic dimension.
/// - Cosine is **scale-invariant**, so dividing each vector by its own
///   positive scale leaves the cosine *exactly* unchanged: the result is
///   the exact scale-invariant cosine in `[-1, 1]` for **every** finite
///   vector pair from `≈1e-23` to `f32::MAX` (and f16/bf16), with
///   underflow / overflow / `0*Inf` all *structurally impossible*.
///
/// The conventional finite `0.0` is returned **only** for a genuine
/// all-zero vector (max-abs `== 0`, exact) or a length-0 input — never for
/// a nonzero vector, however tiny or huge. This is an **mlxrs-only
/// convenience** with **no python/swift/mlx-c reference** (parity-audited),
/// so it deliberately uses this stable formulation: there is no
/// bit-identity-to-reference constraint, only a correct, robust, terminal
/// cosine. This is *intentionally distinct* from the python-faithful
/// dtype-aware weak-scalar eps used by
/// [`normalize`](crate::embeddings::normalize())/[`l2_normalize`]/
/// [`cosine_similarity_matrix`] (which mirror python `mx.maximum(norm,
/// eps)` — reference-faithful, a `f16` zero vector → `NaN` there on
/// purpose; their unconditional clamp is **not** replicated here).
pub fn cosine_similarity(a: &Array, b: &Array) -> Result<f32> {
  validate_cosine_similarity_vectors(a, b)?;
  // Length-0 short-circuit: the rank/length validator treats `(0,)` vs
  // `(0,)` as equal-length rank-1 and passes it through. Reducing an empty
  // array is ill-defined here; the established empty-vector contract is a
  // finite `0.0`. Return it directly, before any reduction. (Lengths are
  // validated equal above, so checking `a` suffices.)
  if a.shape()[0] == 0 {
    return Ok(0.0);
  }
  // Widen to f32 (consistent with the established F32-final approach):
  // f16/bf16→f32 is lossless, a no-op for f32. All subsequent arithmetic
  // and the zero predicate run at f32.
  let a_f32 = astype(a, Dtype::F32)?;
  let b_f32 = astype(b, Dtype::F32)?;
  // Max-abs (Chebyshev / ∞-norm) scale of each vector: `abs` then a full
  // `max`-reduce to a scalar. NO `square` is involved, so this is EXACT
  // and immune to underflow/overflow — `s == 0.0` iff every element is
  // *exactly* `0` (a genuine all-zero vector), and it is never `NaN`.
  let s_a = max(&abs(&a_f32)?, false)?;
  let s_b = max(&abs(&b_f32)?, false)?;
  // Exact zero predicate (no float fuzz): a genuine all-zero vector ⇔ its
  // max-abs is *exactly* `0`. Evaluated on the max-abs scalars directly —
  // it can never be `NaN` and cannot be triggered by a nonzero vector
  // regardless of how tiny (the whole point vs. the prior norm-derived
  // predicate, which underflowed `square` and misclassified tiny vectors).
  let is_zero = logical_or(
    &equal(&s_a, &scalar_like(0.0, &s_a)?)?,
    &equal(&s_b, &scalar_like(0.0, &s_b)?)?,
  )?;
  // Force the scale to `1.0` wherever the zero predicate holds, so the
  // materialized divide NEVER evaluates `x / 0` (both `select` branches
  // are computed). A nonzero scale is left exactly as-is.
  let safe_sa = select(&is_zero, &scalar_like(1.0, &s_a)?, &s_a)?;
  let safe_sb = select(&is_zero, &scalar_like(1.0, &s_b)?, &s_b)?;
  // Scale each vector by its own max-abs. Now every `|x̂_i| ≤ 1` with the
  // max-magnitude element exactly `1.0`, so `sum(square(x̂)) ∈ [1, n]` and
  // `‖x̂‖₂ ∈ [1, sqrt(n)]`: NO underflow-to-`0`, NO overflow-to-`+Inf`,
  // for any realistic dimension. Cosine is scale-invariant, so this leaves
  // the cosine *exactly* unchanged.
  let ah = divide(&a_f32, &safe_sa)?;
  let bh = divide(&b_f32, &safe_sb)?;
  // dot/norms entirely on O(1)-magnitude scaled data: `na,nb ∈ [1,
  // sqrt(n)]`, so `na*nb` is bounded well away from `0` and `+Inf`.
  let dot = sum(&multiply(&ah, &bh)?, false)?;
  let na = sqrt(&sum(&square(&ah)?, false)?)?;
  let nb = sqrt(&sum(&square(&bh)?, false)?)?;
  let ratio = divide(&dot, &multiply(&na, &nb)?)?;
  // Finite `0.0` ONLY for a genuine all-zero (or length-0, handled above)
  // vector — exact, never a tiny/huge nonzero vector.
  let mut sim_f32 = select(&is_zero, &scalar_like(0.0, &ratio)?, &ratio)?;
  sim_f32.item::<f32>()
}

/// Pairwise cosine similarity matrix for a `(n, d)` batch of row vectors.
///
/// Rows are L2-normalized, then `normalized @ normalized.T` yields the
/// `(n, n)` similarity matrix (diagonal `≈ 1.0`). Mirrors the
/// `mx.matmul(embeddings, embeddings.T)` pattern in `mlx-embeddings`.
///
/// The L2-normalize step
/// uses [`l2_normalize`]'s python-faithful dtype-aware weak-scalar eps, so
/// an all-zero **f16**/**bf16** row normalizes to `NaN` (the default
/// `1e-9` eps underflows below the half subnormal floor — see the
/// [`normalize`](crate::embeddings::normalize()) body comment). This is
/// **intentional python
/// `mlx-embeddings` / MLX per-dtype parity** (reference-faithful,
/// explicitly asserted by
/// `tests/embeddings.rs::normalize_zero_vector_f16_bf16_eps_floor_in_-
/// dtype`), NOT a defect: flooring in f32 here would diverge from the
/// python reference. This is *deliberately distinct* from the scalar
/// [`cosine_similarity`] (an mlxrs-only convenience with no python
/// reference, which DOES guarantee finite `0.0` for f16/bf16 zero vectors
/// via an f32-guarded final divide). The two are intentionally
/// different by design (matrix path = reference parity; scalar =
/// finite-0.0 convenience), not contradictory.
pub fn cosine_similarity_matrix(embeddings: &Array) -> Result<Array> {
  let normalized = l2_normalize(embeddings)?;
  let transposed = transpose_axes(&normalized, &[1, 0])?;
  matmul(&normalized, &transposed)
}
