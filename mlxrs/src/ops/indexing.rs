//! Indexing ops: slice (start/stop/strides), plus
//! take / take_axis / take_along_axis / gather.

use std::ffi::c_int;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    CapExceededPayload, EmptyInputPayload, Error, InvariantViolationPayload, LengthMismatchPayload,
    MultiLengthMismatchPayload, OutOfRangePayload, Result, check,
  },
  ffi::VectorArrayGuard,
  shape::dim_ptr,
  stream::default_stream,
};

/// Slice `a` with NumPy-style `start`/`stop`/`strides` per dimension.
///
/// CANONICAL INDEXING TEMPLATE — pattern: 3 parallel slices passed as
/// (ptr, len) triples to mlx-c.
///
/// All three slices must be the same length and equal to `a.ndim()`. For a
/// 0-D scalar input that means three empty slices, which is the correct
/// no-op slice — empty inputs are routed through `dim_ptr`'s static sentinel
/// (avoiding a dangling pointer), not rejected.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.slice.html).
pub fn slice(a: &Array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
  if start.len() != stop.len() || start.len() != strides.len() {
    return Err(Error::MultiLengthMismatch(MultiLengthMismatchPayload::new(
      "slice: start/stop/strides",
      vec![
        ("start", start.len()),
        ("stop", stop.len()),
        ("strides", strides.len()),
      ],
    )));
  }
  if start.len() != a.ndim() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "slice: start/stop/strides length",
      a.ndim(),
      start.len(),
    )));
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_slice(
      &mut out.0,
      a.0,
      dim_ptr(start),
      start.len(),
      dim_ptr(stop),
      stop.len(),
      dim_ptr(strides),
      strides.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Take elements of `a` at the flat positions in `indices` (treating `a` as
/// a 1-D array and returning the flat-take). Output dtype matches `a`'s.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take.html).
pub fn take(a: &Array, indices: &Array) -> Result<Array> {
  // Reject Bool indices before the FFI call (see `reject_bool_index`).
  reject_bool_index("take: index dtype", indices)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_take(&mut out.0, a.0, indices.0, default_stream()) })?;
  Ok(out)
}

/// Take elements of `a` at `indices` along `axis`. Output dtype matches `a`'s.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take.html).
pub fn take_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
  // Reject Bool indices before the FFI call (see `reject_bool_index`).
  reject_bool_index("take_axis: index dtype", indices)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_take_axis(&mut out.0, a.0, indices.0, axis as c_int, default_stream())
  })?;
  Ok(out)
}

/// Take elements of `a` along `axis` using a per-position `indices` array
/// (broadcasts `indices` against the non-`axis` dims of `a`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html).
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
  // Reject Bool indices before the FFI call (see `reject_bool_index`):
  // `take_along_axis` builds its `GatherAxis` primitive with no op-build dtype
  // check, so a Bool index would otherwise be carried into lazy eval.
  reject_bool_index("take_along_axis: index dtype", indices)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_take_along_axis(&mut out.0, a.0, indices.0, axis as c_int, default_stream())
  })?;
  Ok(out)
}

/// Scatter `values` into `a` at `indices` along `axis` (inverse of
/// [`take_along_axis`]). `indices` and `values` broadcast against the
/// non-`axis` dims of `a`; returns a new array (the source is unchanged).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.put_along_axis.html).
pub fn put_along_axis(a: &Array, indices: &Array, values: &Array, axis: i32) -> Result<Array> {
  // Reject Bool indices before the FFI call (see `reject_bool_index`):
  // `put_along_axis` builds its `ScatterAxis` primitive with no op-build dtype
  // check, so a Bool index would otherwise be carried into lazy eval.
  reject_bool_index("put_along_axis: index dtype", indices)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_put_along_axis(
      &mut out.0,
      a.0,
      indices.0,
      values.0,
      axis as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Add `values` into `a` at `indices` along `axis`, **accumulating** on
/// duplicate indices (the additive counterpart of [`put_along_axis`], which
/// instead overwrites). `indices` and `values` broadcast against the
/// non-`axis` dims of `a`; returns a new array (the source is unchanged).
///
/// mlx's `a.at[..., idx].add(v)` / `mx.scatter_add_axis` — the primitive
/// behind mlx-swift `FrequencyPenaltyContext`'s `zeros(vocab).at[tokens]
/// .add(ones)` histogram and mlx-lm's `logits.at[:, idx].add(values)`
/// logit-bias, where repeated token ids must each contribute.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter_add_axis.html).
pub fn scatter_add_axis(a: &Array, indices: &Array, values: &Array, axis: i32) -> Result<Array> {
  // Reject Bool indices before the FFI call (see `reject_bool_index`):
  // `scatter_add_axis` builds its `ScatterAxis` primitive with no op-build dtype
  // check, so a Bool index would otherwise be carried into lazy eval.
  reject_bool_index("scatter_add_axis: index dtype", indices)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  // `mlx_scatter_add_axis` is byte-identical in ownership to the already-
  // wrapped `mlx_put_along_axis` (verified in vendor mlx/c/ops.cpp): a
  // try/catch `mlx_array_set_(*res, scatter_add_axis(get_(a), get_(indices),
  // get_(values), axis, get_(s)))` — `mlx_array_get_` only borrows (throws,
  // caught → rc, if NULL), `mlx_array_set_` writes into the pre-allocated
  // out (or allocs on NULL ctx); no input handle is retained or freed.
  check(unsafe {
    mlxrs_sys::mlx_scatter_add_axis(
      &mut out.0,
      a.0,
      indices.0,
      values.0,
      axis as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Gather slices of `a` indexed by `indices` along `axes`, with a per-axis
/// `slice_sizes`. The number of `indices` arrays must match `axes.len()`;
/// `slice_sizes.len()` must equal `a.ndim()` (one entry per dimension of `a`).
///
/// Mirrors `concatenate`'s variadic-input + handler-installed pattern. Empty
/// `indices` is rejected because `mlx_gather` requires at least one index
/// array (one per gather axis).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather.html).
pub fn gather(a: &Array, indices: &[&Array], axes: &[i32], slice_sizes: &[i32]) -> Result<Array> {
  if indices.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "gather: indices slice",
    )));
  }
  if indices.len() != axes.len() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "gather: indices.len() vs axes.len()",
      axes.len(),
      indices.len(),
    )));
  }
  // slice_sizes is a shape extent (one per dim of `a`); it must be non-negative
  // and have rank == a.ndim(). Without these guards, negative or wrong-rank
  // values cross into mlx::core::Shape construction.
  if slice_sizes.len() != a.ndim() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "gather: slice_sizes.len() vs a.ndim()",
      a.ndim(),
      slice_sizes.len(),
    )));
  }
  crate::shape::validate_dims(slice_sizes)?;
  // Reject Bool indices before the FFI call (see `reject_bool_index`). Core
  // `gather` throws `std::invalid_argument` for them (mlx-c catches it), but the
  // binding guards it here for a uniform typed error across the indexing family.
  for idx in indices {
    reject_bool_index("gather: index dtype", idx)?;
  }
  crate::error::ensure_handler_installed();
  let raw: Vec<mlxrs_sys::mlx_array> = indices.iter().map(|a| a.0).collect();
  // SAFETY: `raw` is a contiguous, live `Vec<mlx_array>` (`mlx_array` is `Copy`);
  // `(ptr, len)` is a valid pair; mlx-c copies the handles into its own
  // `std::vector` and does not retain the Rust pointer. The RAII guard
  // frees the returned vector (NULL ctx is a defined no-op).
  let vec = unsafe { mlxrs_sys::mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
  let _vec_guard = VectorArrayGuard(vec);
  if vec.ctx.is_null() {
    return Err(
      crate::error::LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::Backend(
          "mlx_vector_array_new_data returned NULL".into(),
        )),
    );
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_gather(
      &mut out.0,
      a.0,
      vec,
      dim_ptr(axes),
      axes.len(),
      dim_ptr(slice_sizes),
      slice_sizes.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}

// ---------------------------------------------------------------------------
// Multi-axis scatter family (inverse of `gather`).
//
// `updates` has its leading `indices.ndim()` dims correspond to the scattered
// locations, and its trailing dims broadcast against `a`'s shape (the values
// written at each location). The number of `indices` arrays must equal
// `axes.len()` (one index array per scattered axis), the same precondition as
// `gather`. Unlike `gather`, an empty `indices`/`axes` pair is permitted by
// core scatter, so it is NOT rejected here. There is no count cap: core
// scatter constrains `indices.size() <= a.ndim()` itself and iterates the
// axes via `ndim`-bounded loops, so no user-unbounded `int` arithmetic is
// reachable from a direct argument.
// ---------------------------------------------------------------------------

/// Reject a Bool-dtype index for the gather/scatter indexing family. The
/// motivating hazard is `mlx::core::scatter`, which rejects bool indices with a
/// bare `throw("[scatter] Boolean indices not supported.")` of a string literal
/// — NOT a `std::exception` — which the mlx-c wrapper's `catch (std::exception&)`
/// does not catch, so a bool index would unwind uncaught across the `extern "C"`
/// boundary and terminate the process instead of returning an error. Guard it
/// binding-side with a typed error.
///
/// The audit of the merged indexing wrappers against vendored `mlx/ops.cpp`
/// found two behaviours at op-construction time:
///   - `gather` (and therefore `take` / `take_axis`, which delegate to it)
///     throws `std::invalid_argument` for bool indices, which mlx-c DOES catch
///     and surface via `check()`.
///   - `take_along_axis` / `put_along_axis` / `scatter_add_axis` build their
///     `GatherAxis` / `ScatterAxis` primitive with NO op-build dtype check at
///     all, so a Bool index is not rejected at the binding boundary and is
///     instead carried into lazy eval (where bool indices are semantically
///     invalid and unguarded).
///
/// Bool indices are meaningless for every integer-gather/scatter op, so this
/// guard is applied uniformly to all of them: the scatter paths (the
/// const-char*-thrower) for soundness, and the gather/take family for a
/// consistent typed `Err` instead of a deeper/less-precise failure. The mlx-c
/// `catch (std::exception&)` narrowness — and the const-char* throw itself — are
/// upstream concerns. `dtype()` is a cheap metadata read — no eval — unlike
/// index-value bounds checking. `context` names the calling op in the error.
fn reject_bool_index(context: &'static str, idx: &Array) -> Result<()> {
  if idx.dtype()? == Dtype::Bool {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      context,
      "indices must not be Bool (Bool indices are not supported by mlx indexing ops)",
    )));
  }
  Ok(())
}

/// Builds the borrowed `mlx_vector_array` from `indices` and dispatches the
/// given multi-axis scatter binding. Mirrors `gather`'s vector-array plumbing.
fn scatter_multi(
  context: &'static str,
  a: &Array,
  indices: &[&Array],
  updates: &Array,
  axes: &[i32],
  ffi: unsafe extern "C" fn(
    *mut mlxrs_sys::mlx_array,
    mlxrs_sys::mlx_array,
    mlxrs_sys::mlx_vector_array,
    mlxrs_sys::mlx_array,
    *const c_int,
    usize,
    mlxrs_sys::mlx_stream,
  ) -> c_int,
) -> Result<Array> {
  if indices.len() != axes.len() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      context,
      axes.len(),
      indices.len(),
    )));
  }
  // Cheap arity bound BEFORE the wrapper allocates the index vector: core
  // scatter rejects `indices.size() > a.ndim()` (mlx ops.cpp ~3578), but only
  // after mlx-c has built its own axes vector. Rejecting it here first avoids
  // the wrapper-side `Vec<mlx_array>` + `mlx_vector_array` allocation on a
  // malformed oversized index/axes pair.
  if indices.len() > a.ndim() {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "scatter: number of index arrays",
      "a.ndim()",
      a.ndim() as u64,
      indices.len() as u64,
    )));
  }
  // Reject Bool indices before the FFI call: mlx core throws an uncaught
  // (non-std::exception) C++ exception for them — see `reject_bool_index`.
  for idx in indices {
    reject_bool_index("scatter: index dtype", idx)?;
  }
  crate::error::ensure_handler_installed();
  let raw: Vec<mlxrs_sys::mlx_array> = indices.iter().map(|a| a.0).collect();
  // SAFETY: `raw` is a contiguous, live `Vec<mlx_array>` (`mlx_array` is `Copy`);
  // `(ptr, len)` is a valid pair; mlx-c copies the handles into its own
  // `std::vector` and does not retain the Rust pointer. The RAII guard
  // frees the returned vector (NULL ctx is a defined no-op).
  let vec = unsafe { mlxrs_sys::mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
  let _vec_guard = VectorArrayGuard(vec);
  if vec.ctx.is_null() {
    return Err(
      crate::error::LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::Backend(
          "mlx_vector_array_new_data returned NULL".into(),
        )),
    );
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the
  // call, not retained by mlx past it). Verified in vendor mlx/c/ops.cpp: each
  // `mlx_scatter*` is a try/catch `mlx_array_set_(*res, scatter*(get_(a),
  // vector_array_get_(indices), get_(updates), {axes…}, get_(s)))` — every
  // `*_get_` only borrows and the input handles are never retained or freed;
  // the out-param was freshly allocated above and is written by `set_`. The
  // backend rc is surfaced via `check()`.
  check(unsafe {
    ffi(
      &mut out.0,
      a.0,
      vec,
      updates.0,
      dim_ptr(axes),
      axes.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Scatter `updates` into `a` at `indices` along `axes`, **overwriting** the
/// targeted locations (inverse of [`gather`]). The number of `indices` arrays
/// must equal `axes.len()`; `updates`' leading dims index the locations and
/// its trailing dims broadcast against `a`. Returns a new array (the source is
/// unchanged).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter.html).
pub fn scatter(a: &Array, indices: &[&Array], updates: &Array, axes: &[i32]) -> Result<Array> {
  scatter_multi(
    "scatter: indices.len() vs axes.len()",
    a,
    indices,
    updates,
    axes,
    mlxrs_sys::mlx_scatter,
  )
}

/// Scatter-**add** `updates` into `a` at `indices` along `axes`, accumulating
/// on duplicate locations. Shapes follow [`scatter`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter_add.html).
pub fn scatter_add(a: &Array, indices: &[&Array], updates: &Array, axes: &[i32]) -> Result<Array> {
  scatter_multi(
    "scatter_add: indices.len() vs axes.len()",
    a,
    indices,
    updates,
    axes,
    mlxrs_sys::mlx_scatter_add,
  )
}

/// Scatter-**max** `updates` into `a` at `indices` along `axes`, keeping the
/// element-wise maximum on duplicate locations. Shapes follow [`scatter`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter_max.html).
pub fn scatter_max(a: &Array, indices: &[&Array], updates: &Array, axes: &[i32]) -> Result<Array> {
  scatter_multi(
    "scatter_max: indices.len() vs axes.len()",
    a,
    indices,
    updates,
    axes,
    mlxrs_sys::mlx_scatter_max,
  )
}

/// Scatter-**min** `updates` into `a` at `indices` along `axes`, keeping the
/// element-wise minimum on duplicate locations. Shapes follow [`scatter`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter_min.html).
pub fn scatter_min(a: &Array, indices: &[&Array], updates: &Array, axes: &[i32]) -> Result<Array> {
  scatter_multi(
    "scatter_min: indices.len() vs axes.len()",
    a,
    indices,
    updates,
    axes,
    mlxrs_sys::mlx_scatter_min,
  )
}

/// Scatter-**prod** `updates` into `a` at `indices` along `axes`, multiplying
/// on duplicate locations. Shapes follow [`scatter`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter_prod.html).
pub fn scatter_prod(a: &Array, indices: &[&Array], updates: &Array, axes: &[i32]) -> Result<Array> {
  scatter_multi(
    "scatter_prod: indices.len() vs axes.len()",
    a,
    indices,
    updates,
    axes,
    mlxrs_sys::mlx_scatter_prod,
  )
}

// ---------------------------------------------------------------------------
// Single-axis scatter family — `indices`/`updates` as single arrays and one
// `int` axis (the convenience overloads of the multi-axis forms above, mirror
// of the already-wrapped `scatter_add_axis`). `axis` is a scalar normalized
// and bounds-checked C-side, so — like `scatter_add_axis` / `take_axis` — no
// Rust-side guard is added. Distinct from `scatter_add_axis` (which is
// `mlx::core::scatter_add_axis`, a take_along_axis-style op); these wrap
// `mlx::core::scatter*(a, {indices}, updates, {axis})`.
// ---------------------------------------------------------------------------

/// Dispatches a single-axis scatter binding. Mirrors [`scatter_add_axis`].
fn scatter_single(
  a: &Array,
  indices: &Array,
  updates: &Array,
  axis: i32,
  ffi: unsafe extern "C" fn(
    *mut mlxrs_sys::mlx_array,
    mlxrs_sys::mlx_array,
    mlxrs_sys::mlx_array,
    mlxrs_sys::mlx_array,
    c_int,
    mlxrs_sys::mlx_stream,
  ) -> c_int,
) -> Result<Array> {
  // Reject Bool indices before the FFI call (see `reject_bool_index`): mlx core
  // throws an uncaught (non-std::exception) C++ exception for them.
  reject_bool_index("scatter: index dtype", indices)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the
  // call, not retained by mlx past it). Verified in vendor mlx/c/ops.cpp: each
  // `mlx_scatter*_single` is a try/catch `mlx_array_set_(*res, scatter*(get_(a),
  // get_(indices), get_(updates), axis, get_(s)))` — every `*_get_` only
  // borrows and the input handles are never retained or freed; the out-param
  // was freshly allocated above and is written by `set_`. The backend rc is
  // surfaced via `check()`.
  check(unsafe {
    ffi(
      &mut out.0,
      a.0,
      indices.0,
      updates.0,
      axis as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Scatter `updates` into `a` at `indices` along a single `axis`,
/// **overwriting** the targeted locations. The single-axis convenience form of
/// [`scatter`]: `indices`' leading dims index the locations and `updates`'
/// trailing dims broadcast against `a`. Returns a new array.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter.html).
pub fn scatter_axis(a: &Array, indices: &Array, updates: &Array, axis: i32) -> Result<Array> {
  scatter_single(a, indices, updates, axis, mlxrs_sys::mlx_scatter_single)
}

/// Scatter-**add** `updates` into `a` at `indices` along a single `axis`. The
/// single-axis convenience form of [`scatter_add`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter_add.html).
pub fn scatter_add_single(a: &Array, indices: &Array, updates: &Array, axis: i32) -> Result<Array> {
  scatter_single(a, indices, updates, axis, mlxrs_sys::mlx_scatter_add_single)
}

/// Scatter-**max** `updates` into `a` at `indices` along a single `axis`. The
/// single-axis convenience form of [`scatter_max`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter_max.html).
pub fn scatter_max_single(a: &Array, indices: &Array, updates: &Array, axis: i32) -> Result<Array> {
  scatter_single(a, indices, updates, axis, mlxrs_sys::mlx_scatter_max_single)
}

/// Scatter-**min** `updates` into `a` at `indices` along a single `axis`. The
/// single-axis convenience form of [`scatter_min`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter_min.html).
pub fn scatter_min_single(a: &Array, indices: &Array, updates: &Array, axis: i32) -> Result<Array> {
  scatter_single(a, indices, updates, axis, mlxrs_sys::mlx_scatter_min_single)
}

/// Scatter-**prod** `updates` into `a` at `indices` along a single `axis`. The
/// single-axis convenience form of [`scatter_prod`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.scatter_prod.html).
pub fn scatter_prod_single(
  a: &Array,
  indices: &Array,
  updates: &Array,
  axis: i32,
) -> Result<Array> {
  scatter_single(
    a,
    indices,
    updates,
    axis,
    mlxrs_sys::mlx_scatter_prod_single,
  )
}

// ---------------------------------------------------------------------------
// slice-update family (inverse of `slice`) — replace / reduce a strided
// sub-region of `src` with `update`. Mirrors `slice` EXACTLY: start/stop/
// strides are 3 parallel (ptr, len) triples that must be the same length and
// equal to `src.ndim()`; empty slices route through `dim_ptr`'s static
// sentinel; `validate_dims` is NOT applied because strides may be negative.
// ---------------------------------------------------------------------------

/// Dispatches a strided slice-update binding. Mirrors [`slice()`]'s length
/// guards + (ptr, len) plumbing.
#[allow(clippy::too_many_arguments)]
fn slice_update_impl(
  context_multi: &'static str,
  context_len: &'static str,
  src: &Array,
  update: &Array,
  start: &[i32],
  stop: &[i32],
  strides: &[i32],
  ffi: unsafe extern "C" fn(
    *mut mlxrs_sys::mlx_array,
    mlxrs_sys::mlx_array,
    mlxrs_sys::mlx_array,
    *const c_int,
    usize,
    *const c_int,
    usize,
    *const c_int,
    usize,
    mlxrs_sys::mlx_stream,
  ) -> c_int,
) -> Result<Array> {
  if start.len() != stop.len() || start.len() != strides.len() {
    return Err(Error::MultiLengthMismatch(MultiLengthMismatchPayload::new(
      context_multi,
      vec![
        ("start", start.len()),
        ("stop", stop.len()),
        ("strides", strides.len()),
      ],
    )));
  }
  if start.len() != src.ndim() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      context_len,
      src.ndim(),
      start.len(),
    )));
  }
  // DIRECT-ARG SOUNDNESS (#266 decision A): mlx `normalize_slice` (ops.cpp ~646)
  // does int32 arithmetic on each direct `stride`. Its full overflow surface is
  // exactly THREE independent paths, all guarded per axis below:
  //   1. `stride == 0`: the output-extent formula divides by `stride`.
  //   2. `stride == i32::MIN`: the negative branch computes `-stride`, which is
  //      signed-overflow UB for INT_MIN regardless of axis size. This is NOT
  //      subsumed by check 3: for a zero-length axis, `0 + abs(i32::MIN) - 1`
  //      equals i32::MAX exactly (not greater), so it would slip past the
  //      magnitude bound — it must be rejected explicitly.
  //   3. large `abs(stride)`: the output extent is `(span + stride - 1) / stride`
  //      with `span` clamped to `0..=axis_size`, so the worst-case numerator
  //      magnitude is `axis_size + abs(stride) - 1`; bound it in i64.
  // start/stop cannot add a fourth path: they are clamped to `0..=axis_size`
  // before the subtraction, and the `s + n` negative-index normalization only
  // fires for `s < 0` (bounded). Other negative strides (reverse-stepping
  // slices) stay valid. `shape()` is a cheap metadata read, no eval.
  let shape = src.shape();
  for (axis, &stride) in strides.iter().enumerate() {
    if stride == 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "slice-update: stride",
        "must be non-zero (mlx normalize_slice divides by it)",
        "0",
      )));
    }
    if stride == i32::MIN {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "slice-update: stride",
        "must not be i32::MIN (its negation is UB in mlx normalize_slice, any axis size)",
        "i32::MIN",
      )));
    }
    // i64 so the bound check cannot itself wrap; `stride != i32::MIN` here so
    // `abs(stride)` is representable.
    if shape[axis] as i64 + (stride as i64).abs() - 1 > i32::MAX as i64 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "slice-update: stride",
        "axis_size + abs(stride) - 1 must fit in i32 (mlx normalize_slice int32 overflow)",
        "out-of-range magnitude",
      )));
    }
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the
  // call, not retained by mlx past it). Verified in vendor mlx/c/ops.cpp: each
  // `mlx_slice_update*` is a try/catch `mlx_array_set_(*res, slice_update*(
  // get_(src), get_(update), Shape(start..), Shape(stop..), Shape(strides..),
  // get_(s)))` — every `*_get_` only borrows and the input handles are never
  // retained or freed; empties route through `dim_ptr`'s sentinel; the
  // out-param was freshly allocated above and is written by `set_`. The
  // backend rc is surfaced via `check()`.
  check(unsafe {
    ffi(
      &mut out.0,
      src.0,
      update.0,
      dim_ptr(start),
      start.len(),
      dim_ptr(stop),
      stop.len(),
      dim_ptr(strides),
      strides.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Update `src` by **overwriting** the strided sub-region selected by
/// `start`/`stop`/`strides` with `update` (inverse of [`slice()`]). All three
/// index slices must be the same length and equal to `src.ndim()`; `update`
/// must broadcast to the selected region. Returns a new array (the source is
/// unchanged).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.slice_update.html).
pub fn slice_update(
  src: &Array,
  update: &Array,
  start: &[i32],
  stop: &[i32],
  strides: &[i32],
) -> Result<Array> {
  slice_update_impl(
    "slice_update: start/stop/strides",
    "slice_update: start/stop/strides length",
    src,
    update,
    start,
    stop,
    strides,
    mlxrs_sys::mlx_slice_update,
  )
}

/// Update `src` by **adding** `update` into the strided sub-region selected by
/// `start`/`stop`/`strides`. Shapes follow [`slice_update`].
pub fn slice_update_add(
  src: &Array,
  update: &Array,
  start: &[i32],
  stop: &[i32],
  strides: &[i32],
) -> Result<Array> {
  slice_update_impl(
    "slice_update_add: start/stop/strides",
    "slice_update_add: start/stop/strides length",
    src,
    update,
    start,
    stop,
    strides,
    mlxrs_sys::mlx_slice_update_add,
  )
}

/// Update `src` by taking the element-wise **maximum** of `update` and the
/// strided sub-region selected by `start`/`stop`/`strides`. Shapes follow
/// [`slice_update`].
pub fn slice_update_max(
  src: &Array,
  update: &Array,
  start: &[i32],
  stop: &[i32],
  strides: &[i32],
) -> Result<Array> {
  slice_update_impl(
    "slice_update_max: start/stop/strides",
    "slice_update_max: start/stop/strides length",
    src,
    update,
    start,
    stop,
    strides,
    mlxrs_sys::mlx_slice_update_max,
  )
}

/// Update `src` by taking the element-wise **minimum** of `update` and the
/// strided sub-region selected by `start`/`stop`/`strides`. Shapes follow
/// [`slice_update`].
pub fn slice_update_min(
  src: &Array,
  update: &Array,
  start: &[i32],
  stop: &[i32],
  strides: &[i32],
) -> Result<Array> {
  slice_update_impl(
    "slice_update_min: start/stop/strides",
    "slice_update_min: start/stop/strides length",
    src,
    update,
    start,
    stop,
    strides,
    mlxrs_sys::mlx_slice_update_min,
  )
}

/// Update `src` by **multiplying** the strided sub-region selected by
/// `start`/`stop`/`strides` by `update`. Shapes follow [`slice_update`].
pub fn slice_update_prod(
  src: &Array,
  update: &Array,
  start: &[i32],
  stop: &[i32],
  strides: &[i32],
) -> Result<Array> {
  slice_update_impl(
    "slice_update_prod: start/stop/strides",
    "slice_update_prod: start/stop/strides length",
    src,
    update,
    start,
    stop,
    strides,
    mlxrs_sys::mlx_slice_update_prod,
  )
}

/// Update `src` by **overwriting** a sub-region whose per-`axes` offsets are
/// taken from the `start` array (a runtime, data-dependent counterpart of
/// [`slice_update`]; the region's extent matches `update`'s shape). `start` is
/// an integer array validated C-side; `axes` selects which dimensions `start`
/// addresses. Returns a new array.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.slice_update.html).
pub fn slice_update_dynamic(
  src: &Array,
  update: &Array,
  start: &Array,
  axes: &[i32],
) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the
  // call, not retained by mlx past it). Verified in vendor mlx/c/ops.cpp:
  // `mlx_slice_update_dynamic` is a try/catch `mlx_array_set_(*res,
  // slice_update(get_(src), get_(update), get_(start), {axes…}, get_(s)))` —
  // every `*_get_` only borrows and the input handles are never retained or
  // freed; empty `axes` routes through `dim_ptr`'s sentinel; the out-param was
  // freshly allocated above and is written by `set_`. The backend rc is
  // surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_slice_update_dynamic(
      &mut out.0,
      src.0,
      update.0,
      start.0,
      dim_ptr(axes),
      axes.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}
