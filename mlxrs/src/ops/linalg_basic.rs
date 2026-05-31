//! Basic linalg ops: addmm (trinary+scalar template), matmul, inner, outer,
//! plus the gathered/batched matmul `gather_mm` (the mixture-of-experts primitive).
//!
//! The matrix-structure ops (#259) live alongside these:
//! `tensordot` (int-axis + axes-lists forms), `diagonal`, `trace`, `tril`, `triu`.

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, Error, LengthMismatchPayload, OutOfRangePayload,
    Result, check,
  },
  ffi::opt_array,
  shape::dim_ptr,
  stream::default_stream,
};

/// Reject an axis-list length that would overflow the signed `int i` loop the
/// core `tensordot` runs over `axes_a.size()` (mlx `ops.cpp` ~5398). The count
/// is a direct FFI argument, so per the #259 issue-266 decision (option A) it is
/// capped binding-side. Mirrors `ops/shape.rs`'s `check_count` (the shared
/// extraction is tracked in the #259 duplication cleanup); pulled into a helper
/// so the cap is unit-testable at the boundary without allocating a multi-GB
/// slice.
fn check_axis_count(context: &'static str, len: usize) -> Result<()> {
  const CAP: usize = i32::MAX as usize;
  if len > CAP {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      context, "i32::MAX", CAP as u64, len as u64,
    )));
  }
  Ok(())
}

/// Reject `offset == i32::MIN` for `diagonal`/`trace`, whose core computes
/// `std::max(-offset, 0)` (mlx `ops.cpp` ~5973): negating `i32::MIN` is
/// signed-overflow UB, and `offset` is the wrapper's own direct scalar argument
/// (reachable on any normal 2-D-or-higher input), so it is rejected binding-side
/// (#259 issue-266 decision A). Every other offset only feeds bounds-checked
/// slice math C-side.
fn guard_offset(context: &'static str, offset: i32) -> Result<()> {
  if offset == i32::MIN {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "must not be i32::MIN (its negation is UB in MLX)",
      "i32::MIN",
    )));
  }
  Ok(())
}

/// Reject `k` values that would drive the `arange(-k, m - k)` the core `tri`
/// builds for `tril`/`triu` (mlx `ops.cpp` ~372, where `m` is the last matrix
/// dimension of `x`) into signed-`int` overflow UB: the start `-k` overflows at
/// `i32::MIN`, and the stop `m - k` overflows `i32` for sufficiently negative
/// `k`. `k` is the wrapper's own direct scalar argument (reachable on any normal
/// 2-D-or-higher input), so per the #259 issue-266 decision (option A) it is guarded
/// binding-side. MLX validates `ndim >= 2` before this arithmetic, so when `x`
/// is < 2-D the guard is skipped and MLX emits the dimension error; the residual
/// shape-product overflows inside `arange`/`broadcast` remain mlx-core-internal
/// (upstream ml-explore/mlx#3601).
fn guard_tri_k(context: &'static str, x: &Array, k: i32) -> Result<()> {
  if x.ndim() < 2 {
    return Ok(());
  }
  let shape = x.shape();
  // mlx `ShapeElem` is `int`, so a materialized dimension is <= i32::MAX; widen
  // to i64 for the endpoint math and reject any non-i32-representable result.
  let m = shape[shape.len() - 1] as i64;
  let start = -(k as i64); // core `tri` start = -k
  let stop = m - k as i64; // core `tri` stop  = m - k
  let representable = |v: i64| (i32::MIN as i64..=i32::MAX as i64).contains(&v);
  if !representable(start) || !representable(stop) {
    return Err(Error::ArithmeticOverflow(ArithmeticOverflowPayload::new(
      context, "i32",
    )));
  }
  Ok(())
}

/// `alpha * (a @ b) + beta * c` — fused matmul + scaled add.
///
/// CANONICAL TRINARY+SCALAR TEMPLATE — pattern: 3 array inputs + 2 primitive
/// scalar inputs.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.addmm.html).
pub fn addmm(c: &Array, a: &Array, b: &Array, alpha: f32, beta: f32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_addmm(&mut out.0, c.0, a.0, b.0, alpha, beta, default_stream()) })?;
  Ok(out)
}

/// Matrix multiplication: `a @ b`. Generalizes to batched matmul (last two
/// dims of each input are the matmul dims; leading dims broadcast).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.matmul.html).
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_matmul(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Ordinary inner product of two 1-D arrays. For higher-rank inputs, mlx
/// contracts over the last axis of each (matching numpy `inner`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.inner.html).
pub fn inner(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_inner(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Outer product of two 1-D arrays. Higher-rank inputs are flattened first
/// (matching numpy `outer`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.outer.html).
pub fn outer(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_outer(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Tensor contraction over the last `axis` dimensions of `a` and the first
/// `axis` dimensions of `b` (the integer-axis form).
///
/// Mirrors `mlx.core.tensordot(a, b, axes=axis)` with an integer `axes`
/// argument (python `python/src/ops.cpp`, the `int` branch of the `axes`
/// variant) / `mlx_tensordot_axis`. `axis = 2` is the python/numpy default.
/// For the explicit per-operand axis-list form, see [`tensordot_axes`].
///
/// # Soundness
/// `axis` is a single scalar forwarded straight to the C++ op, which
/// bounds-checks it (`axis < 0` and `axis > min(a.ndim(), b.ndim())` both throw
/// `std::invalid_argument`, surfaced here as [`Error::Backend`]) before any
/// arithmetic on it. There is no direct-argument overflow path, so no guard is
/// added (per the #266 decision A scalar-arg note).
///
/// # Errors
/// Returns an error if `axis` is out of range or the contracted shapes do not
/// match (surfaced from the underlying MLX call).
pub fn tensordot(a: &Array, b: &Array, axis: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; `axis` is a plain scalar; the backend rc is
  // surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_tensordot_axis(&mut out.0, a.0, b.0, axis, default_stream()) })?;
  Ok(out)
}

/// Tensor contraction over explicit, per-operand contraction axes (the
/// axis-list form).
///
/// Mirrors `mlx.core.tensordot(a, b, axes=[axes_a, axes_b])` (python
/// `python/src/ops.cpp`, the `list[list[int]]` branch, which requires exactly
/// two lists) / `mlx_tensordot`. The two axis lists must have equal length;
/// element `axes_a[i]` of `a` is contracted against `axes_b[i]` of `b`.
///
/// # Soundness
/// We pre-check `axes_a.len() == axes_b.len()` and surface a typed
/// [`Error::LengthMismatch`] (the C++ op throws the same mismatch, but a Rust
/// pre-check yields a precise typed error rather than an opaque backend string).
///
/// The axis-list length is itself a direct FFI count argument, and the core
/// `tensordot` loop iterates it with a signed `int i`
/// (`for (int i = 0; i < axes_a.size(); i++)`, mlx `ops.cpp` ~5398), so a slice
/// longer than [`i32::MAX`] drives that index into signed-overflow UB on
/// otherwise-valid inputs. Per the #266 decision (option A) this direct-argument
/// count is capped binding-side via `check_axis_count` (capping one list
/// suffices, the two lengths being already equal), returning a typed
/// [`Error::CapExceeded`].
///
/// The axis *values* (negative / out-of-range entries reaching the C++
/// `x.shape(axes_a.at(i))` indexing and the `cdims1[n + ndim]` normalization),
/// and any overflow inside the contraction-size product `csize *= x.shape(...)`
/// (`ops.cpp` ~5397-5407), are mlx-core-internal and reachable only via
/// malformed axes / degenerate shapes; per the #266 decision (option A) those
/// transitive paths are tracked upstream (ml-explore/mlx#3601) and are
/// intentionally NOT guarded here.
///
/// # Errors
/// Returns [`Error::LengthMismatch`] if the two axis lists differ in length,
/// [`Error::CapExceeded`] if a list is longer than [`i32::MAX`], or an MLX error
/// if the axes/shapes are otherwise invalid.
pub fn tensordot_axes(a: &Array, b: &Array, axes_a: &[i32], axes_b: &[i32]) -> Result<Array> {
  if axes_a.len() != axes_b.len() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "tensordot_axes: axes_a.len() vs axes_b.len()",
      axes_a.len(),
      axes_b.len(),
    )));
  }
  // DIRECT-ARG SOUNDNESS (#259 issue-266 decision A): cap the axis-list count
  // before it reaches the core signed-`int` loop (axes_a.len() == axes_b.len()
  // holds here, so capping one caps both).
  check_axis_count("tensordot_axes: axis-list length", axes_a.len())?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; `dim_ptr` routes empty `axes_a`/`axes_b`
  // through static sentinels so each `(ptr, len)` pair is never singular; the
  // backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_tensordot(
      &mut out.0,
      a.0,
      b.0,
      dim_ptr(axes_a),
      axes_a.len(),
      dim_ptr(axes_b),
      axes_b.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Extract diagonals along the plane spanned by `axis1` and `axis2`.
///
/// Mirrors `mlx.core.diagonal(a, offset, axis1, axis2)` / `mlx_diagonal`.
/// `offset` shifts the diagonal (positive = above the main diagonal). The
/// python defaults are `offset = 0`, `axis1 = 0`, `axis2 = 1`. `axis1`/`axis2`
/// may be negative (counted from the end).
///
/// # Soundness
/// `axis1`/`axis2` are normalized C-side as `axis + ndim` (only when `axis < 0`,
/// so the add cannot overflow) and bounds-checked before use. `offset`, however,
/// reaches `std::max(-offset, 0)` in the core (mlx `ops.cpp` ~5973): negating
/// `i32::MIN` is signed-overflow UB, and `offset` is this wrapper's own direct
/// scalar argument (reachable on any normal 2-D-or-higher input), so it is rejected via
/// `guard_offset` per the #266 decision (option A). The remaining slice math
/// is bounds-checked C-side.
///
/// # Errors
/// Returns [`Error::OutOfRange`] if `offset == i32::MIN`, or an MLX error if
/// `axis1`/`axis2` are out of range or equal.
pub fn diagonal(a: &Array, offset: i32, axis1: i32, axis2: i32) -> Result<Array> {
  guard_offset("diagonal: offset", offset)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; `offset`/`axis1`/`axis2` are plain scalars;
  // the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_diagonal(&mut out.0, a.0, offset, axis1, axis2, default_stream())
  })?;
  Ok(out)
}

/// Sum along the diagonals of an array.
///
/// Mirrors `mlx.core.trace(a, offset, axis1, axis2, dtype)` / `mlx_trace`.
/// Equivalent to summing [`diagonal`] along its last axis. The python defaults
/// are `offset = 0`, `axis1 = 0`, `axis2 = 1`, `dtype = None`; when `dtype` is
/// `None` the output dtype is inferred from the input array (matching the
/// python binding's `!dtype.has_value()` branch, which calls the C++
/// `trace(a, offset, axis1, axis2)` overload that defaults the accumulation
/// type to the input's).
///
/// # Soundness
/// `trace` delegates to `diagonal` C-side, so the same `std::max(-offset, 0)`
/// applies (mlx `ops.cpp` ~6050 -> ~5973): `offset == i32::MIN` is rejected via
/// `guard_offset` per the #266 decision (option A). `axis1`/`axis2` are
/// bounds-checked C-side and `dtype` is a plain enum value.
///
/// # Errors
/// Returns [`Error::OutOfRange`] if `offset == i32::MIN`, an MLX error if
/// `axis1`/`axis2` are invalid, or a dtype error if `self.dtype()` fails when
/// `dtype` is `None`.
pub fn trace(
  a: &Array,
  offset: i32,
  axis1: i32,
  axis2: i32,
  dtype: Option<Dtype>,
) -> Result<Array> {
  guard_offset("trace: offset", offset)?;
  // `dtype = None` mirrors python's input-dtype inference. `Array::dtype()` is
  // a pure metadata read, so resolving it here reproduces the C++
  // `trace(a, offset, axis1, axis2)` overload's "accumulate in the input type"
  // behavior without a separate binding.
  let dtype = match dtype {
    Some(d) => d,
    None => a.dtype()?,
  };
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; `offset`/`axis1`/`axis2` are plain scalars and
  // `dtype` is a plain enum; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_trace(
      &mut out.0,
      a.0,
      offset,
      axis1,
      axis2,
      mlxrs_sys::mlx_dtype::from(dtype),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Lower triangle of an array: zeros every entry strictly above the `k`-th
/// diagonal.
///
/// Mirrors `mlx.core.tril(x, k)` / `mlx_tril`. `k = 0` (the python default)
/// keeps the main diagonal and below; `k > 0` keeps additional super-diagonals;
/// `k < 0` drops sub-diagonals. The input must be at least 2-D.
///
/// # Soundness
/// `tril` passes `k` to the internal `tri`, whose `arange(-k, m - k, ...)`
/// (`ops.cpp` ~372, with `m == x.shape(-1)`) negates and subtracts `k`. Although
/// the `arange` runs inside a transitively-called op, the overflowing operand is
/// `k` itself — this wrapper's own direct scalar argument, reachable on any
/// normal 2-D-or-higher input — so per the #266 decision (option A) it is guarded
/// binding-side via `guard_tri_k` (rejecting a `-k` or `m - k` that escapes
/// `i32`). The residual per-shape overflows inside `arange`/`broadcast` stay
/// mlx-core-internal (upstream ml-explore/mlx#3601).
///
/// # Errors
/// Returns [`Error::ArithmeticOverflow`] if `-k` or `m - k` overflows `i32`, or
/// an MLX error if `x` is less than 2-D.
pub fn tril(x: &Array, k: i32) -> Result<Array> {
  guard_tri_k("tril: k", x, k)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; `k` is a plain scalar; the backend rc is
  // surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_tril(&mut out.0, x.0, k, default_stream()) })?;
  Ok(out)
}

/// Upper triangle of an array: zeros every entry strictly below the `k`-th
/// diagonal.
///
/// Mirrors `mlx.core.triu(x, k)` / `mlx_triu`. `k = 0` (the python default)
/// keeps the main diagonal and above; `k > 0` drops super-diagonals; `k < 0`
/// keeps additional sub-diagonals. The input must be at least 2-D.
///
/// # Soundness
/// `triu` passes `k - 1` to the internal `tri` (`ops.cpp` ~388), so the core
/// computes `arange(-(k - 1), m - (k - 1), ...)`. The `k - 1` subtraction and
/// the inner `-(k - 1)` / `m - (k - 1)` are all on `k`, this wrapper's direct
/// scalar argument (reachable on any normal 2-D-or-higher input), so per the #266
/// decision (option A) they are guarded binding-side: `k - 1` via
/// [`i32::checked_sub`] and the `tri` endpoints via `guard_tri_k`. The
/// residual per-shape overflows stay mlx-core-internal
/// (upstream ml-explore/mlx#3601).
///
/// # Errors
/// Returns [`Error::ArithmeticOverflow`] if `k - 1`, `-(k - 1)`, or
/// `m - (k - 1)` overflows `i32`, or an MLX error if `x` is less than 2-D.
pub fn triu(x: &Array, k: i32) -> Result<Array> {
  // `triu` feeds `k - 1` into the core `tri`; guard that subtraction (UB at
  // i32::MIN) before reusing the shared `arange(-k', m - k')` endpoint guard.
  let kk = k.checked_sub(1).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::new("triu: k - 1", "i32"))
  })?;
  guard_tri_k("triu: k", x, kk)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; `k` is a plain scalar; the backend rc is
  // surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_triu(&mut out.0, x.0, k, default_stream()) })?;
  Ok(out)
}

/// Batched/gathered matmul: like [`matmul`] but selects per-batch rows of `a` /
/// `b` via optional `lhs_indices` / `rhs_indices` flat batch indices. The
/// indices contain flat indices along the **batch** dimensions of each input
/// (all but the last two dims); the last two dims of each input are still the
/// matmul dims and contract normally.
///
/// This is the dense primitive behind mixture-of-experts (MoE) `SwitchLinear`:
/// `a` is `[N, 1, K]` per-token input, `b` is `[E, K, M]` per-expert weights,
/// and `rhs_indices` is the `[N]` per-token expert assignment — the result is
/// `[N, 1, M]`, with token `i` matmul'd against expert `rhs_indices[i]`.
///
/// `sorted_indices` promises `rhs_indices` is sorted, enabling a faster
/// kernel (mlx-lm's `SwitchGLU` sets this on the `_gather_sort` path).
///
/// Mirrors python `mx.gather_mm` and swift `gatherMM`. See
/// [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html).
pub fn gather_mm(
  a: &Array,
  b: &Array,
  lhs_indices: Option<&Array>,
  rhs_indices: Option<&Array>,
  sorted_indices: bool,
) -> Result<Array> {
  let (lhs_h, _lhs_guard) = opt_array(lhs_indices);
  let (rhs_h, _rhs_guard) = opt_array(rhs_indices);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it) — `lhs_h` / `rhs_h` are either the borrowed
  // optional handles or NULL-ctx placeholders kept alive by their guards, which
  // `mlx_gather_mm` accepts for the optional `lhs_indices` / `rhs_indices`
  // (verified in vendor mlx/c/ops.cpp:1445-1469: each `ctx ? optional : nullopt`);
  // the out-param was freshly allocated above and is written by this call;
  // the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_gather_mm(
      &mut out.0,
      a.0,
      b.0,
      lhs_h,
      rhs_h,
      sorted_indices,
      default_stream(),
    )
  })?;
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::{check_axis_count, diagonal, tensordot, tensordot_axes, trace, tril, triu};
  use crate::{array::Array, dtype::Dtype, error::Error};

  // [[1,2,3],[4,5,6],[7,8,9]] — the shared 3x3 fixture for the structure ops.
  fn mat3() -> Array {
    Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap()
  }

  // ---- direct-argument soundness guards (#259 issue-266 decision A) ----

  // The real overflow path needs an axis list with > i32::MAX entries (~8GB+),
  // impractical to allocate, so exercise the cap at its boundary with a
  // synthetic `len` — mirrors `ops/shape.rs`'s `check_count_boundary`.
  #[test]
  fn check_axis_count_boundary() {
    assert!(check_axis_count("t", 0).is_ok());
    assert!(check_axis_count("t", i32::MAX as usize).is_ok());
    let over = i32::MAX as usize + 1;
    match check_axis_count("ctx", over) {
      Err(Error::CapExceeded(p)) => {
        assert_eq!(p.context(), "ctx");
        assert_eq!(p.cap(), i32::MAX as u64);
        assert_eq!(p.observed(), over as u64);
      }
      other => panic!("expected Err(CapExceeded) one past the cap, got {other:?}"),
    }
  }

  // core `diagonal` computes `std::max(-offset, 0)`; negating i32::MIN is UB, so
  // the wrapper rejects it as a typed OutOfRange on a normal 2-D input (no FFI).
  #[test]
  fn diagonal_offset_i32_min_is_typed_error() {
    match diagonal(&mat3(), i32::MIN, 0, 1) {
      Err(Error::OutOfRange(p)) => assert_eq!(p.context(), "diagonal: offset"),
      other => panic!("expected OutOfRange for i32::MIN offset, got {other:?}"),
    }
    // A normal offset is unaffected (the guard does not over-reject).
    assert!(diagonal(&mat3(), 1, 0, 1).is_ok());
    assert!(diagonal(&mat3(), -1, 0, 1).is_ok());
  }

  // `trace` delegates to `diagonal`, so the same i32::MIN offset rejection holds.
  #[test]
  fn trace_offset_i32_min_is_typed_error() {
    match trace(&mat3(), i32::MIN, 0, 1, None) {
      Err(Error::OutOfRange(p)) => assert_eq!(p.context(), "trace: offset"),
      other => panic!("expected OutOfRange for i32::MIN offset, got {other:?}"),
    }
    assert!(trace(&mat3(), 0, 0, 1, None).is_ok());
  }

  // core `tril` -> `tri` builds `arange(-k, m - k)`: `-k` is UB at i32::MIN and
  // `m - k` overflows i32 for very-negative k. Both are rejected as a typed
  // ArithmeticOverflow on a normal 2-D input.
  #[test]
  fn tril_k_overflow_is_typed_error() {
    match tril(&mat3(), i32::MIN) {
      Err(Error::ArithmeticOverflow(p)) => assert_eq!(p.context(), "tril: k"),
      other => panic!("expected ArithmeticOverflow for i32::MIN k, got {other:?}"),
    }
    // k = i32::MIN + 1: `-k` is representable but `m - k` (3 - (i32::MIN+1))
    // overflows i32 -> still rejected (exercises the stop-endpoint guard).
    assert!(matches!(
      tril(&mat3(), i32::MIN + 1),
      Err(Error::ArithmeticOverflow(_))
    ));
    assert!(tril(&mat3(), 0).is_ok());
    assert!(tril(&mat3(), -1).is_ok());
  }

  // `triu` computes `k - 1` first (UB at i32::MIN), then the same `tri` endpoints.
  #[test]
  fn triu_k_overflow_is_typed_error() {
    match triu(&mat3(), i32::MIN) {
      Err(Error::ArithmeticOverflow(p)) => assert_eq!(p.context(), "triu: k - 1"),
      other => panic!("expected ArithmeticOverflow for i32::MIN k, got {other:?}"),
    }
    // k = i32::MIN + 1: `k - 1` == i32::MIN, whose negation in `tri` overflows ->
    // caught by the shared endpoint guard.
    assert!(matches!(
      triu(&mat3(), i32::MIN + 1),
      Err(Error::ArithmeticOverflow(_))
    ));
    assert!(triu(&mat3(), 0).is_ok());
    assert!(triu(&mat3(), 1).is_ok());
  }

  #[test]
  fn tensordot_int_full_contraction() {
    // axis=2 contracts both axes of two 2x2 matrices: sum of the elementwise
    // product = 1*1 + 2*2 + 3*3 + 4*4 = 30.
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let mut c = tensordot(&a, &b, 2).unwrap();
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![30.0]);
  }

  #[test]
  fn tensordot_int_zero_axes_is_outer() {
    // axis=0 contracts nothing -> outer product, shape (2,)+(2,) = (2,2):
    // outer([1,2],[3,4]) = [[3,4],[6,8]].
    let a = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let b = Array::from_slice(&[3.0f32, 4.0], &[2]).unwrap();
    let mut c = tensordot(&a, &b, 0).unwrap();
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![3.0, 4.0, 6.0, 8.0]);
  }

  #[test]
  fn tensordot_int_one_axis_is_matmul() {
    // For 2-D operands, axis=1 contracts a's last with b's first -> matmul.
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let mut c = tensordot(&a, &b, 1).unwrap();
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
  }

  #[test]
  fn tensordot_int_negative_axis_errors() {
    // The C++ int form rejects axis < 0 (ops.cpp ~5371) -> typed Err, no panic.
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(tensordot(&a, &b, -1).is_err());
  }

  #[test]
  fn tensordot_axes_matmul_equivalent() {
    // Contract a's axis 1 with b's axis 0 -> standard matmul.
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let mut c = tensordot_axes(&a, &b, &[1], &[0]).unwrap();
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
  }

  #[test]
  fn tensordot_axes_full_contraction() {
    // Contract both axes pairwise: 1*1+2*2+3*3+4*4 = 30.
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let mut c = tensordot_axes(&a, &b, &[0, 1], &[0, 1]).unwrap();
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![30.0]);
  }

  #[test]
  fn tensordot_axes_negative_axis_matches_matmul() {
    // a's axis -1 (== 1) contracted with b's axis 0 -> matmul, exercising the
    // C-side negative-axis normalization on the axes-lists path.
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let mut c = tensordot_axes(&a, &b, &[-1], &[0]).unwrap();
    assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
  }

  #[test]
  fn tensordot_axes_length_mismatch_is_typed_error() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    match tensordot_axes(&a, &b, &[0, 1], &[0]).unwrap_err() {
      Error::LengthMismatch(p) => {
        assert_eq!(p.expected(), 2);
        assert_eq!(p.actual(), 1);
      }
      other => panic!("expected LengthMismatch, got {other:?}"),
    }
  }

  #[test]
  fn diagonal_main() {
    // main diagonal of the 3x3 fixture -> [1,5,9].
    let mut d = diagonal(&mat3(), 0, 0, 1).unwrap();
    assert_eq!(d.to_vec::<f32>().unwrap(), vec![1.0, 5.0, 9.0]);
  }

  #[test]
  fn diagonal_positive_offset() {
    // offset=1 -> super-diagonal [2,6].
    let mut d = diagonal(&mat3(), 1, 0, 1).unwrap();
    assert_eq!(d.to_vec::<f32>().unwrap(), vec![2.0, 6.0]);
  }

  #[test]
  fn diagonal_negative_offset() {
    // offset=-1 -> sub-diagonal [4,8].
    let mut d = diagonal(&mat3(), -1, 0, 1).unwrap();
    assert_eq!(d.to_vec::<f32>().unwrap(), vec![4.0, 8.0]);
  }

  #[test]
  fn diagonal_negative_axes() {
    // axis1=-2, axis2=-1 on a 2-D array are the same as 0,1 -> [1,5,9].
    let mut d = diagonal(&mat3(), 0, -2, -1).unwrap();
    assert_eq!(d.to_vec::<f32>().unwrap(), vec![1.0, 5.0, 9.0]);
  }

  #[test]
  fn trace_main() {
    // trace of the 3x3 fixture = 1+5+9 = 15.
    let mut t = trace(&mat3(), 0, 0, 1, None).unwrap();
    assert_eq!(t.to_vec::<f32>().unwrap(), vec![15.0]);
  }

  #[test]
  fn trace_positive_offset() {
    // offset=1 -> sum of super-diagonal [2,6] = 8.
    let mut t = trace(&mat3(), 1, 0, 1, None).unwrap();
    assert_eq!(t.to_vec::<f32>().unwrap(), vec![8.0]);
  }

  #[test]
  fn trace_negative_offset() {
    // offset=-1 -> sum of sub-diagonal [4,8] = 12.
    let mut t = trace(&mat3(), -1, 0, 1, None).unwrap();
    assert_eq!(t.to_vec::<f32>().unwrap(), vec![12.0]);
  }

  #[test]
  fn trace_explicit_dtype_promotes() {
    // Integer input traced into Float32: 1+4 = 5.0, and the OUTPUT dtype is the
    // requested Float32 (not the input I32) — proving `dtype` is forwarded.
    let a = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]).unwrap();
    let mut t = trace(&a, 0, 0, 1, Some(Dtype::F32)).unwrap();
    assert_eq!(t.dtype().unwrap(), Dtype::F32);
    assert_eq!(t.to_vec::<f32>().unwrap(), vec![5.0]);
  }

  #[test]
  fn trace_default_dtype_is_input_dtype() {
    // dtype=None infers the input dtype: an I32 input yields an I32 trace.
    let a = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]).unwrap();
    let mut t = trace(&a, 0, 0, 1, None).unwrap();
    assert_eq!(t.dtype().unwrap(), Dtype::I32);
    assert_eq!(t.to_vec::<i32>().unwrap(), vec![5]);
  }

  #[test]
  fn tril_k_zero() {
    // Lower triangle incl. main diagonal: zeros strictly above.
    let mut l = tril(&mat3(), 0).unwrap();
    assert_eq!(
      l.to_vec::<f32>().unwrap(),
      vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
    );
  }

  #[test]
  fn tril_k_positive() {
    // k=1 also keeps the first super-diagonal.
    let mut l = tril(&mat3(), 1).unwrap();
    assert_eq!(
      l.to_vec::<f32>().unwrap(),
      vec![1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
  }

  #[test]
  fn tril_k_negative() {
    // k=-1 drops the main diagonal too.
    let mut l = tril(&mat3(), -1).unwrap();
    assert_eq!(
      l.to_vec::<f32>().unwrap(),
      vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]
    );
  }

  #[test]
  fn triu_k_zero() {
    // Upper triangle incl. main diagonal: zeros strictly below.
    let mut u = triu(&mat3(), 0).unwrap();
    assert_eq!(
      u.to_vec::<f32>().unwrap(),
      vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
    );
  }

  #[test]
  fn triu_k_positive() {
    // k=1 drops the main diagonal, keeps strictly-upper.
    let mut u = triu(&mat3(), 1).unwrap();
    assert_eq!(
      u.to_vec::<f32>().unwrap(),
      vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
    );
  }

  #[test]
  fn triu_k_negative() {
    // k=-1 also keeps the first sub-diagonal.
    let mut u = triu(&mat3(), -1).unwrap();
    assert_eq!(
      u.to_vec::<f32>().unwrap(),
      vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]
    );
  }

  #[test]
  fn tril_requires_2d() {
    // 1-D input: the C++ op rejects ndim < 2 -> typed Err, no panic.
    let v = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    assert!(tril(&v, 0).is_err());
  }
}
