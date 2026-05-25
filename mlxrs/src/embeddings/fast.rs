//! Embedding-scoped `mlx_fast` wrappers.
//!
//! `mlxrs` has no general `ops::fast` module yet (the fast-ops port is
//! out of M3 scope), so the two fused-norm primitives that the
//! [`pool`](super::pool) dispatcher applies to the *pooled* sentence
//! vector (post-pooling, before matryoshka truncation / L2-normalize —
//! swift `Pooling`'s `applyLayerNorm` step) are surfaced here, bounded
//! to embedding use. These are *not* the model's internal token-level
//! normalization (per-architecture, out of scope):
//!
//! - [`layer_norm`] → `mlx_fast_layer_norm` (backs the dispatcher's
//!   `apply_layer_norm` flag — swift `MLXFast.layerNorm`, eps `1e-5`).
//! - [`rms_norm`] → `mlx_fast_rms_norm` (an RMSNorm post-pool variant —
//!   some embedding backbones, e.g. gemma/llama-bidirec, normalize the
//!   pooled vector with RMSNorm rather than LayerNorm).
//!
//! These are deliberately *not* a general `mlx_fast` port. Only the two
//! norm fns are wrapped; `rope`, the metal/cuda custom-kernel surface,
//! `scaled_dot_product_attention`, etc. are intentionally skipped — they
//! are not embedding-pooling support surface.

use crate::{
  array::Array,
  error::{Result, check},
  stream::default_stream,
};

/// Optional affine weight/bias forwarded to a fused-norm call.
///
/// mlx-c's `mlx_fast_layer_norm` / `mlx_fast_rms_norm` accept the
/// `weight`/`bias` handles as "may be null"; a fresh empty `mlx_array`
/// (`mlx_array_new()`) *is* the null handle per the mlx-c convention, so
/// `None` maps to that and the kernel runs the un-affine path.
#[inline]
fn null_array() -> Array {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle
  // (NULL ctx) per the mlx-c convention. Wrapped in the RAII newtype so
  // it is freed on drop; passing it as a "may be null" affine arg is the
  // documented way to request the no-weight/no-bias path.
  Array(unsafe { mlxrs_sys::mlx_array_new() })
}

/// Fused Layer Normalization over the last axis: `mlx_fast_layer_norm`.
///
/// `(x - mean) / sqrt(var + eps)`, optionally affine-scaled by `weight`
/// and shifted by `bias` (both `None` ⇒ the plain normalize path, which
/// is what the pooling dispatcher's `apply_layer_norm` uses). Mirrors
/// swift `MLXEmbedders` `Pooling.callAsFunction(applyLayerNorm:)`'s
/// `MLXFast.layerNorm(pooled, eps: 1e-5)` — hence the `1e-5` default at
/// the call site.
///
/// - `x`: any float array; normalization is over the last dim.
/// - `weight` / `bias`: optional `(hidden,)` affine params.
/// - `eps`: variance floor (swift uses `1e-5`).
pub fn layer_norm(
  x: &Array,
  weight: Option<&Array>,
  bias: Option<&Array>,
  eps: f32,
) -> Result<Array> {
  let null_w = null_array();
  let null_b = null_array();
  let w = weight.unwrap_or(&null_w);
  let b = bias.unwrap_or(&null_b);
  // SAFETY: `mlx_array_new()` yields a fresh empty out handle (NULL ctx);
  // wrapped in the RAII newtype FIRST so an early return / panic frees it.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles, live for
  // the call and not retained past it; `w`/`b` may be the empty
  // (null-equivalent) handle, which mlx-c explicitly accepts for the
  // optional affine params; the out-param was freshly allocated above and
  // is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fast_layer_norm(&mut out.0, x.0, w.0, b.0, eps, default_stream())
  })?;
  Ok(out)
}

/// Fused Root-Mean-Square Normalization over the last axis:
/// `mlx_fast_rms_norm`.
///
/// `x / sqrt(mean(x^2) + eps)`, optionally affine-scaled by `weight`
/// (`None` ⇒ the plain RMSNorm path used by the dispatcher's
/// `apply_rms_norm`). RMSNorm has no `bias`. Provided because several
/// embedding backbones (gemma, llama-bidirec) RMS-normalize rather than
/// LayerNorm-normalize before pooling.
///
/// - `x`: any float array; normalization is over the last dim.
/// - `weight`: optional `(hidden,)` affine scale.
/// - `eps`: variance floor.
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
  let null_w = null_array();
  let w = weight.unwrap_or(&null_w);
  // SAFETY: `mlx_array_new()` yields a fresh empty out handle (NULL ctx);
  // wrapped in the RAII newtype FIRST so an early return / panic frees it.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles, live for
  // the call and not retained past it; `w` may be the empty
  // (null-equivalent) handle, which mlx-c explicitly accepts for the
  // optional affine param; the out-param was freshly allocated above and
  // is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_fast_rms_norm(&mut out.0, x.0, w.0, eps, default_stream()) })?;
  Ok(out)
}
