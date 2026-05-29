//! Comparison ops: element-wise comparisons + boolean queries.
//!
//! Element-wise binary comparisons (`equal`, `not_equal`, `less`, `less_equal`,
//! `greater`, `greater_equal`) follow the canonical binary-op archetype and
//! produce `Bool` arrays. Boolean queries (`allclose`, `isclose`) take
//! tolerance scalars; per-element predicates (`isfinite`, `isinf`, `isnan`)
//! are unary.

use crate::{
  array::Array,
  error::{Result, check},
  stream::default_stream,
};

/// Element-wise equality: `out[i] = a[i] == b[i]` (with broadcasting).
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.equal.html).
pub fn equal(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_equal(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise inequality: `out[i] = a[i] != b[i]` (with broadcasting).
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.not_equal.html).
pub fn not_equal(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_not_equal(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise strict less-than: `out[i] = a[i] < b[i]` (with broadcasting).
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.less.html).
pub fn less(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_less(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise less-than-or-equal: `out[i] = a[i] <= b[i]` (with broadcasting).
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.less_equal.html).
pub fn less_equal(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_less_equal(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise strict greater-than: `out[i] = a[i] > b[i]` (with broadcasting).
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.greater.html).
pub fn greater(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_greater(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise greater-than-or-equal: `out[i] = a[i] >= b[i]` (with broadcasting).
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.greater_equal.html).
pub fn greater_equal(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_greater_equal(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Returns a scalar Bool array (a 0-d / 1-element `Array`) whose value is
/// `true` iff every element pair satisfies `|a - b| <= atol + rtol * |b|`.
/// If `equal_nan` is true, NaN positions are treated as equal.
///
/// Call `.item::<bool>()` on the returned array to extract a Rust `bool`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.allclose.html).
pub fn allclose(a: &Array, b: &Array, rtol: f64, atol: f64, equal_nan: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_allclose(
      &mut out.0,
      a.0,
      b.0,
      rtol,
      atol,
      equal_nan,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Element-wise variant of `allclose`: returns a Bool array of the same
/// (broadcast) shape, `true` where `|a - b| <= atol + rtol * |b|`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.isclose.html).
pub fn isclose(a: &Array, b: &Array, rtol: f64, atol: f64, equal_nan: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_isclose(
      &mut out.0,
      a.0,
      b.0,
      rtol,
      atol,
      equal_nan,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Element-wise finite check: `true` where `a[i]` is neither inf nor NaN.
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.isfinite.html).
pub fn isfinite(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_isfinite(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise inf check: `true` where `a[i]` is +inf or -inf.
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.isinf.html).
pub fn isinf(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_isinf(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise NaN check: `true` where `a[i]` is NaN.
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.isnan.html).
pub fn isnan(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_isnan(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise negative-infinity check: `true` where `a[i]` is -inf.
/// Output dtype is Bool. Integer inputs (which cannot hold inf) yield an
/// all-`false` mask, matching mlx (`isneginf` returns `full(shape, false)`
/// for non-inexact dtypes).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.isneginf.html).
pub fn isneginf(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_isneginf(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise positive-infinity check: `true` where `a[i]` is +inf.
/// Output dtype is Bool. Integer inputs yield an all-`false` mask (see
/// [`isneginf`]).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.isposinf.html).
pub fn isposinf(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_isposinf(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}
