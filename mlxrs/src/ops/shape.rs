//! Shape ops: reshape (Phase 3.5 archetype #3 — IntoShape pattern) and
//! concatenate (Phase 3.5 archetype #4 — variadic input), plus the Phase 4
//! Branch B fan-out: transpose/expand_dims/squeeze/broadcast_to/stack/split/
//! flatten/swapaxes/pad.

use std::ffi::c_int;

use crate::{
  array::Array,
  error::{Error, LengthMismatchPayload, MultiLengthMismatchPayload, Result, check},
  ffi::VectorArrayGuard,
  shape::{IntoShape, dim_ptr, stride_ptr, validate_dims},
  stream::default_stream,
};

/// Reshape `a` to a new shape. Errors on incompatible total element count
/// (the C++ side validates).
///
/// CANONICAL SHAPE ARCHETYPE — the `IntoShape::with_shape` callback pattern
/// used by every shape-taking op. Every reshape/expand_dims/squeeze/etc.
/// follows this exact shape.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reshape.html).
pub fn reshape(a: &Array, shape: &impl IntoShape) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_reshape(&mut out.0, a.0, dim_ptr(s), s.len(), default_stream())
    })?;
    Ok(out)
  })
}

/// Concatenate `arrays` along `axis`.
///
/// CANONICAL VARIADIC-INPUT TEMPLATE — pattern: build an `mlx_vector_array`
/// on the C side from a Rust slice, RAII-wrap for cleanup. Every fn taking
/// `Vec<Array>` (stack, meshgrid, broadcast_arrays) follows this shape.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.concatenate.html).
pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
  // Concatenating zero arrays has no defined result shape — reject before
  // FFI rather than constructing an empty vector_array (which would also
  // hand mlx-c a Rust dangling pointer for `Vec::as_ptr()` on an empty Vec).
  if arrays.is_empty() {
    return Err(Error::ShapeMismatch(
      "concatenate: arrays slice is empty".into(),
    ));
  }
  // Install the error handler before the first fallible FFI call. Without
  // this, mlx_vector_array_new_data could fail and trigger mlx-c's default
  // printf+exit handler before default_stream() (the usual install site)
  // is reached. Codex PR #5 finding 3.
  crate::error::ensure_handler_installed();

  // Build a contiguous Vec<mlx_array> (mlx_array is Copy) and pass to
  // mlx_vector_array_new_data. RAII-free the vector_array via guard.
  let raw: Vec<mlxrs_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
  // SAFETY: `raw` is a contiguous, live `Vec<mlx_array>` (`mlx_array` is `Copy`);
  // `(ptr, len)` is a valid pair; mlx-c copies the handles into its own
  // `std::vector` and does not retain the Rust pointer. The RAII guard
  // frees the returned vector (NULL ctx is a defined no-op).
  let vec = unsafe { mlxrs_sys::mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
  let _vec_guard = VectorArrayGuard(vec);

  // Drain the captured backend message immediately if vector construction
  // failed — passing a NULL vec into mlx_concatenate_axis would discard the
  // original error and surface a less useful "null vector" failure instead.
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
  check(unsafe { mlxrs_sys::mlx_concatenate_axis(&mut out.0, vec, axis, default_stream()) })?;
  Ok(out)
}

/// Transpose with full reverse permutation (i.e. swap the order of all axes).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.transpose.html).
pub fn transpose(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_transpose(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Transpose with a custom axis permutation. `axes` may be empty for a 0-D
/// scalar input; in that case the call routes through `dim_ptr`'s static
/// sentinel rather than handing mlx-c a Rust dangling pointer.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.transpose.html).
pub fn transpose_axes(a: &Array, axes: &[i32]) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_transpose_axes(&mut out.0, a.0, dim_ptr(axes), axes.len(), default_stream())
  })?;
  Ok(out)
}

/// Insert size-1 dimensions at each of the given `axes`. Empty `axes` is a
/// short-circuit identity (`try_clone`) — same rationale as `sum_axes`,
/// keeping the FFI call out of the dangling-pointer path.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.expand_dims.html).
pub fn expand_dims_axes(a: &Array, axes: &[i32]) -> Result<Array> {
  if axes.is_empty() {
    return a.try_clone();
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_expand_dims_axes(&mut out.0, a.0, dim_ptr(axes), axes.len(), default_stream())
  })?;
  Ok(out)
}

/// Remove the size-1 dimensions named by `axes`. Empty `axes` short-circuits
/// to `try_clone` (numpy/mlx semantics: squeezing no axes is identity).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.squeeze.html).
pub fn squeeze_axes(a: &Array, axes: &[i32]) -> Result<Array> {
  if axes.is_empty() {
    return a.try_clone();
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_squeeze_axes(&mut out.0, a.0, dim_ptr(axes), axes.len(), default_stream())
  })?;
  Ok(out)
}

/// Broadcast `a` to `shape` (NumPy broadcasting rules). The output is a
/// strided view; use `Array::contiguous()` (M2) to materialize a copy.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.broadcast_to.html).
pub fn broadcast_to(a: &Array, shape: &impl IntoShape) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_broadcast_to(&mut out.0, a.0, dim_ptr(s), s.len(), default_stream())
    })?;
    Ok(out)
  })
}

/// Stack `arrays` along a new axis 0 (use `stack_axis` for a different axis).
/// Mirrors `concatenate` in error/handler discipline.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stack.html).
pub fn stack(arrays: &[&Array]) -> Result<Array> {
  if arrays.is_empty() {
    return Err(Error::ShapeMismatch("stack: arrays slice is empty".into()));
  }
  crate::error::ensure_handler_installed();
  let raw: Vec<mlxrs_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
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
  check(unsafe { mlxrs_sys::mlx_stack(&mut out.0, vec, default_stream()) })?;
  Ok(out)
}

/// Stack `arrays` along a new `axis`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stack.html).
pub fn stack_axis(arrays: &[&Array], axis: i32) -> Result<Array> {
  if arrays.is_empty() {
    return Err(Error::ShapeMismatch(
      "stack_axis: arrays slice is empty".into(),
    ));
  }
  crate::error::ensure_handler_installed();
  let raw: Vec<mlxrs_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
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
  check(unsafe { mlxrs_sys::mlx_stack_axis(&mut out.0, vec, axis as c_int, default_stream()) })?;
  Ok(out)
}

/// Split `a` along `axis` at each of the given `indices` (NumPy `split`
/// section semantics: `indices = [3, 5]` of a length-10 axis yields three
/// parts of lengths `[3, 2, 5]`). Empty `indices` returns a single-element
/// vector — `[a]` — matching mlx-python.
///
/// Returns the parts as a `Vec<Array>` whose length is `indices.len() + 1`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.split.html).
pub fn split_sections(a: &Array, indices: &[i32], axis: i32) -> Result<Vec<Array>> {
  crate::error::ensure_handler_installed();
  // Pre-create an empty vector_array so the FFI has a non-null ctx to write
  // into. mlx_split_sections wraps `mlx_vector_array_set_` (see
  // vendor/mlx-c/mlx/c/private/vector.h), which on a non-null ctx assigns
  // INTO the existing `std::vector` rather than replacing the handle —
  // `vec_out.ctx` is therefore stable across the FFI call and the guard
  // captured before it correctly frees the populated vector on drop. This
  // ordering also covers the early-return case: if `check` returns Err, the
  // guard already owns the (possibly partial) vector and frees it.
  // SAFETY: `mlx_vector_array_new()` returns a fresh empty out-param handle (NULL
  // ctx) per the mlx-c convention; the RAII guard captures it before the
  // populating call so a partial/early-return vector is still freed.
  let mut vec_out = unsafe { mlxrs_sys::mlx_vector_array_new() };
  let _vec_guard = VectorArrayGuard(vec_out);
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_split_sections(
      &mut vec_out,
      a.0,
      dim_ptr(indices),
      indices.len(),
      axis as c_int,
      default_stream(),
    )
  })?;
  // SAFETY: pure read of a valid populated `mlx_vector_array`; mlx-c does not
  // mutate or retain it and returns a plain length.
  let n = unsafe { mlxrs_sys::mlx_vector_array_size(vec_out) };
  let mut parts = Vec::with_capacity(n);
  for i in 0..n {
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut part = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_vector_array_get(&mut part.0, vec_out, i) })?;
    parts.push(part);
  }
  Ok(parts)
}

/// Flatten `a` into a 1-D array along the contiguous dim range
/// `[start_axis, end_axis]` (inclusive on both ends, NumPy convention).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.flatten.html).
pub fn flatten(a: &Array, start_axis: i32, end_axis: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_flatten(
      &mut out.0,
      a.0,
      start_axis as c_int,
      end_axis as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Swap two axes of `a`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.swapaxes.html).
pub fn swapaxes(a: &Array, axis1: i32, axis2: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_swapaxes(
      &mut out.0,
      a.0,
      axis1 as c_int,
      axis2 as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Pad `a` with `pad_value` along each of the given `axes` by `low` (before)
/// and `high` (after) entries respectively. `mode` is the C-side mode string
/// (currently `"constant"` is the only mlx-supported mode).
///
/// `axes`/`low`/`high` must all have the same length. The empty-slice case
/// (zero-axis pad against a 0-D scalar) is routed through `dim_ptr`'s static
/// sentinel rather than the dangling pointer Rust returns from
/// `<&[T]>::as_ptr` for empty slices.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.pad.html).
pub fn pad(
  a: &Array,
  axes: &[i32],
  low: &[i32],
  high: &[i32],
  pad_value: &Array,
  mode: &std::ffi::CStr,
) -> Result<Array> {
  if axes.len() != low.len() || axes.len() != high.len() {
    return Err(Error::MultiLengthMismatch(MultiLengthMismatchPayload::new(
      "pad: axes/low/high",
      vec![
        ("axes", axes.len()),
        ("low", low.len()),
        ("high", high.len()),
      ],
    )));
  }
  // `low`/`high` are shape extents (counts of padding entries), not axis
  // indices, so negatives are invalid and must be rejected before they reach
  // mlx::core::Shape construction (Codex PR #7-target finding). `axes` itself
  // is an axis-index list — negative axes follow numpy semantics and are
  // intentionally NOT validated here.
  crate::shape::validate_dims(low)?;
  crate::shape::validate_dims(high)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_pad(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      dim_ptr(low),
      low.len(),
      dim_ptr(high),
      high.len(),
      pad_value.0,
      mode.as_ptr(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Force `a` to be row-contiguous, copying its buffer if necessary.
///
/// Mirrors `mlx.core.contiguous(a, allow_col_major=False)` (python,
/// `python/src/ops.cpp:5463`) and `MLX.contiguous(_:allowColMajor:stream:)`
/// (swift, `Source/MLX/Ops.swift:3279`). When `a` is already row-contiguous
/// (or already col-major + `allow_col_major == true`), mlx-c returns the
/// input unchanged with a refcount bump; otherwise it materializes a fresh
/// contiguous copy of the data. The returned array always satisfies
/// `as_strided`'s element-bounds contract for `(shape, strides, offset)`
/// computed from its declared `shape()`.
///
/// `allow_col_major == false` is the natural default for callers (us)
/// that hand the result to `as_strided` / raw-pointer slice extraction,
/// since those paths assume row-major layout. Pass `true` only when the
/// downstream op (e.g. a GEMM that accepts col-major operands) handles
/// either order.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.contiguous.html).
pub fn contiguous(a: &Array, allow_col_major: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_contiguous(&mut out.0, a.0, allow_col_major, default_stream()) })?;
  Ok(out)
}

/// Strided view: reinterpret `a`'s buffer with custom `shape`, `strides`,
/// and `offset` (in *elements*, not bytes). Mirrors `mx.as_strided` (python)
/// and `MLX.asStrided(_:_:strides:offset:stream:)` (swift,
/// `Source/MLX/Ops.swift`).
///
/// # Safety
///
/// This is `unsafe` because MLX (and the python/swift bindings it backs)
/// documents `as_strided` as fundamentally unchecked: "It is the user's
/// responsibility to ensure that the resulting array does not point to
/// invalid memory." The wrapper does not (and cannot, without duplicating
/// MLX's internal bounds reasoning) verify that the reachable element range
/// `offset + Σ (shape[i]−1) · strides[i]` (over both signs) lies inside
/// `a`'s flattened storage. A view that escapes that range will later
/// cause invalid native reads when the array is evaluated.
///
/// The caller MUST ensure:
/// - `shape.len() == strides.len()` (also enforced and surfaced as a
///   recoverable [`Error::ShapeMismatch`] before any FFI call — checking
///   it here is a redundant convenience, not a substitute for the caller's
///   own bounds-correctness reasoning).
/// - every entry of `shape` is non-negative.
/// - the reachable element range stays inside the flattened input.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.as_strided.html).
pub unsafe fn as_strided(
  a: &Array,
  shape: &impl IntoShape,
  strides: &[i64],
  offset: usize,
) -> Result<Array> {
  shape.with_shape(|s| {
    if s.len() != strides.len() {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "as_strided: shape length vs strides length",
        s.len(),
        strides.len(),
      )));
    }
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; `dim_ptr`/`stride_ptr` route empty slices
    // through static sentinels so the `(ptr, n)` pair is never singular; the
    // backend rc is surfaced via `check()`. Element-range / offset bounds are
    // the caller's responsibility per this fn's `# Safety` contract.
    check(unsafe {
      mlxrs_sys::mlx_as_strided(
        &mut out.0,
        a.0,
        dim_ptr(s),
        s.len(),
        stride_ptr(strides),
        strides.len(),
        offset,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}
