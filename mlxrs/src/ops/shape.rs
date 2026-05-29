//! Shape ops: reshape (Phase 3.5 archetype #3 — IntoShape pattern) and
//! concatenate (Phase 3.5 archetype #4 — variadic input), plus the Phase 4
//! Branch B fan-out: transpose/expand_dims/squeeze/broadcast_to/stack/split/
//! flatten/swapaxes/pad.

use std::ffi::c_int;

use smol_str::format_smolstr;

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, EmptyInputPayload, Error, LengthMismatchPayload,
    MultiLengthMismatchPayload, OutOfRangePayload, Result, check,
  },
  ffi::VectorArrayGuard,
  shape::{IntoShape, dim_ptr, stride_ptr, validate_dims},
  stream::default_stream,
};

/// Reject a collection length that would overflow an MLX C++ `int` loop
/// counter. Several MLX ops iterate a `size_t` collection (`axes.size()`,
/// the tiled output rank) with a signed `int i` (`for (int i = 0; i < ...; )`,
/// e.g. `ops.cpp` ~1343 / ~6334), so a safe caller passing a slice with more
/// than [`i32::MAX`] entries — every entry individually valid, so no earlier
/// error fires — drives that `int i++` into signed-overflow UB.
///
/// This guards the *count* (the FFI `…_num` argument), which is orthogonal to
/// the value-overflow guards ([`checked_total_shift`], the `tile` product
/// bound): even all-zero / all-valid entries are unsafe past the cap. The cap
/// is the largest count the C++ `int` index can hold, so any reachable in-range
/// count passes and only the genuinely-unrepresentable ones are rejected with a
/// typed [`Error::CapExceeded`].
///
/// Pulled into a helper so the guard is unit-testable at the boundary without
/// allocating a multi-GB slice (constructing one to trip it at runtime is
/// impractical): tests pass a synthetic `len` to exercise the comparison.
fn check_count(context: &'static str, cap_name: &'static str, len: usize) -> Result<()> {
  const CAP: usize = i32::MAX as usize;
  if len > CAP {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      context, cap_name, CAP as u64, len as u64,
    )));
  }
  Ok(())
}

/// Bound the LARGEST intermediate-shape rank `mlx::core::tile` materialises, so
/// no safe `tile(a, reps)` can drive a C++ `int` rank index into signed-overflow
/// UB. Returns `Ok(())` iff that rank is representable, else a typed
/// [`Error::CapExceeded`].
///
/// **Why the final-rank cap is insufficient.** `tile` (`ops.cpp` ~1340-1351)
/// does NOT reshape/broadcast at the final output rank directly. For each
/// aligned axis whose rep `!= 1` it pushes an EXTRA leading dim into two
/// intermediate shapes before the final reshape:
/// ```text
/// for (int i = 0; i < shape.size(); i++) {        // aligned rank = max(reps.len, ndim)
///   if (reps[i] != 1) { expand_shape.push_back(1); broad_shape.push_back(reps[i]); }
///   expand_shape.push_back(shape[i]);  broad_shape.push_back(shape[i]);
///   final_shape.push_back(reps[i] * shape[i]);
/// }
/// x = reshape(arr, expand_shape);   // Reshape::output_shape: `for (int i …; i < shape.size())`
/// x = broadcast_to(x, broad_shape); // broadcast_shapes: `int ndim = s.size()`  ← narrowing
/// return reshape(x, final_shape);
/// ```
/// So `expand_shape`/`broad_shape` have rank `aligned_rank + count(aligned reps
/// != 1)` — STRICTLY LARGER than `final_shape`'s `aligned_rank` whenever any rep
/// `!= 1`. `broadcast_shapes` (`utils.cpp` ~141) then narrows that rank with
/// `int ndim1 = s1.size()` (a `size_t`→`int` assignment) and `Reshape::
/// output_shape` (`primitives.cpp` ~3902) iterates it with `int i`; both are UB
/// once the intermediate rank exceeds [`i32::MAX`]. A final-rank-only cap lets a
/// crafted `reps` (e.g. `len == i32::MAX/2 + 1`, every rep `== 2`) pass — its
/// final rank `≈ i32::MAX/2` is fine, but its intermediate rank `≈ i32::MAX`
/// is not — so the intermediate rank is the quantity that must be capped.
///
/// **The maximum rank.** Aligned rank is `max(reps.len(), a.ndim())`. The padded
/// `1`s (whichever operand is shorter) are by construction `== 1`, so they never
/// add an extra dim; only the explicit entries of `reps` with value `!= 1` do.
/// Hence the max intermediate rank is `aligned_rank + count(r in reps : r != 1)`.
/// Capping it covers the final rank too (`final_rank == aligned_rank <=` it).
///
/// All arithmetic is `usize` and saturating/checked: the inputs are slice lengths
/// (`<= usize::MAX`), and `aligned_rank + extra <= 2 * reps.len()` cannot itself
/// wrap on a 64-bit target, but `saturating_add` is used so the bound is correct
/// even in the degenerate (impossible-to-allocate) case rather than relying on
/// the platform width. Pulled into a helper so the cap is unit-testable with a
/// synthetic `reps` length/content without allocating a multi-GB array.
fn check_tile_intermediate_rank(ndim: usize, reps: &[i32]) -> Result<()> {
  let aligned_rank = reps.len().max(ndim);
  // One EXTRA intermediate dim per explicit rep `!= 1` (padded 1s never count).
  let extra = reps.iter().filter(|&&r| r != 1).count();
  let max_intermediate_rank = aligned_rank.saturating_add(extra);
  check_count(
    "tile: max intermediate rank aligned_rank + count(reps != 1)",
    "i32::MAX",
    max_intermediate_rank,
  )
}

/// Sum a multi-element `shift` slice with `i32::MIN`-aware overflow checking,
/// mirroring the MLX C++ `roll(const array&, const Shape&, ...)` overloads
/// which fold `shift` into a single `int total_shift` (`total_shift += s;`
/// over an unchecked `int` accumulator — `ops.cpp` lines ~6369-6374 / ~6390-6399).
///
/// We reproduce that exact summing semantics here, but safely: the fold uses
/// [`i32::checked_add`] so an overflowing sum surfaces as a typed
/// [`Error::ArithmeticOverflow`] instead of driving C++ signed-overflow UB.
/// An empty slice folds to `0` (a no-op roll), matching the C++ loop over an
/// empty `Shape`.
///
/// The returned total is additionally rejected when it equals [`i32::MIN`]:
/// MLX computes the per-axis offset as `(sh < 0) ? (-sh) % size : ...`
/// (`ops.cpp` line ~6351), and negating `i32::MIN` is itself signed-overflow
/// UB. `i32::MIN` is the *only* value whose negation overflows, and because
/// MLX takes the shift magnitude modulo the axis size, rejecting it loses no
/// reachable behavior (every representable roll distance is expressible by a
/// non-`MIN` shift).
fn checked_total_shift(context: &'static str, shift: &[i32]) -> Result<i32> {
  let mut total: i32 = 0;
  for &s in shift {
    total = total
      .checked_add(s)
      .ok_or_else(|| Error::ArithmeticOverflow(ArithmeticOverflowPayload::new(context, "i32")))?;
  }
  if total == i32::MIN {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "shift sum must not be i32::MIN (its negation is UB in MLX)",
      format_smolstr!("{total}"),
    )));
  }
  Ok(total)
}

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
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "concatenate: arrays slice",
    )));
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
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "stack: arrays slice",
    )));
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
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "stack_axis: arrays slice",
    )));
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

/// Move the axis at `source` to `destination`, keeping the relative order of
/// the other axes. Mirrors `mx.moveaxis(a, source, destination)` (python,
/// `python/src/ops.cpp`) and `MLX.movedAxis` (swift). Negative axes follow the
/// usual numpy convention (resolved C-side).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.moveaxis.html).
pub fn moveaxis(a: &Array, source: i32, destination: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `a.0` is a valid borrowed handle (live for the call, not retained
  // by mlx past it); the out-param was freshly allocated above and is written
  // by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_moveaxis(
      &mut out.0,
      a.0,
      source as c_int,
      destination as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Roll the flattened array by `shift` places, then restore the original
/// shape. Elements rolled past one end re-enter at the other. Positive shifts
/// roll right, negative roll left.
///
/// `shift` is a slice to mirror the C-side `mlx_roll(shift, shift_num)`
/// signature; when more than one shift is given mlx sums them into a single
/// `int`. The empty-slice case folds to a `0` (no-op) shift.
///
/// **Bounded soundness guard.** The MLX C++ `roll(a, Shape shift)` overload
/// sums `shift` into an unchecked `int total_shift` (`ops.cpp` ~6369-6374),
/// then offsets via `(-sh) % size` (~6351). To prevent the safe Rust API from
/// driving the C++ signed-overflow UB in either the sum or the `i32::MIN`
/// negation, the slice is folded here with a `checked_add` + `i32::MIN`
/// rejection, and the single checked total is forwarded to mlx-c — preserving
/// the C++ summing semantics exactly. An overflowing sum yields
/// [`Error::ArithmeticOverflow`]; an `i32::MIN` total yields [`Error::OutOfRange`].
/// A shift sum of exactly `i32::MIN` is thus rejected as a degenerate magnitude
/// (slightly stricter than mlx-core, which would no-op such a shift only on a
/// size-0 axis) — but callers never need `i32::MIN`, since the shift is taken
/// modulo the axis size and every representable roll distance is expressible by
/// a non-`MIN` shift.
///
/// This is the no-axis form of `mx.roll(a, shift)` (python,
/// `python/src/ops.cpp`); see [`roll_axis`] / [`roll_axes`] for the
/// axis-targeted forms.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.roll.html).
pub fn roll(a: &Array, shift: &[i32]) -> Result<Array> {
  // Fold the multi-shift slice ourselves (matching MLX's own summing) so the
  // unchecked C++ `int` accumulation + `i32::MIN` negation cannot be reached.
  let total = checked_total_shift("roll: shift sum", shift)?;
  let total = [total];
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `a.0` is a valid borrowed handle (live for the call, not retained
  // by mlx past it); the out-param was freshly allocated above and is written
  // by this call; `total` is a live length-1 slice so the `(ptr, len)` pair is
  // valid; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_roll(
      &mut out.0,
      a.0,
      dim_ptr(&total),
      total.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Roll along a single `axis` by `shift` places. When more than one shift is
/// given mlx applies their sum to that axis. Negative axes follow the numpy
/// convention (resolved C-side).
///
/// **Bounded soundness guard.** The MLX C++ `roll(a, Shape shift, int axis)`
/// overload folds `shift` into an unchecked `int total_shift` (`ops.cpp`
/// ~6390-6399) and then negates via `(-sh) % size` (~6351). As in [`roll`],
/// the slice is summed here with a `checked_add` + `i32::MIN` rejection and the
/// single checked total is forwarded, so neither the C++ `int` sum nor the `i32::MIN`
/// negation can overflow ([`Error::ArithmeticOverflow`] / [`Error::OutOfRange`]
/// respectively). An empty `shift` folds to a `0` (no-op) shift. As in [`roll`],
/// a shift sum of exactly `i32::MIN` is rejected as a degenerate magnitude
/// (slightly stricter than mlx-core, which would no-op it only on a size-0 axis)
/// — callers never need `i32::MIN` since the shift is taken modulo the axis size.
///
/// Mirrors the single-axis form of `mx.roll(a, shift, axis)` (python).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.roll.html).
pub fn roll_axis(a: &Array, shift: &[i32], axis: i32) -> Result<Array> {
  // Fold the multi-shift slice ourselves (matching MLX's own summing) so the
  // unchecked C++ `int` accumulation + `i32::MIN` negation cannot be reached.
  let total = checked_total_shift("roll_axis: shift sum", shift)?;
  let total = [total];
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `a.0` is a valid borrowed handle (live for the call, not retained
  // by mlx past it); the out-param was freshly allocated above and is written
  // by this call; `total` is a live length-1 slice so the `(ptr, len)` pair is
  // valid; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_roll_axis(
      &mut out.0,
      a.0,
      dim_ptr(&total),
      total.len(),
      axis as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Roll along each of the given `axes` by the matching `shift`, paired
/// positionally. The wrapper enforces **exactly one shift per axis**
/// (`shift.len() == axes.len()`) before the FFI call: it does NOT broadcast a
/// scalar shift the way the python `mx.roll` binding does (that broadcast is a
/// binding-side convenience applied before dispatch), and unlike the underlying
/// MLX C++ — which rejects only `shift.size() < axes.size()` and *silently
/// ignores* extra shifts when `shift.size() > axes.size()` (`ops.cpp`
/// ~6326-6356) — both a too-few and a too-many count are rejected here as a
/// typed [`Error::LengthMismatch`]. Callers wanting scalar-broadcast semantics
/// should repeat the shift, or use [`roll`] / [`roll_axis`].
///
/// **Bounded soundness guard.** MLX computes each per-axis offset as
/// `(sh < 0) ? (-sh) % size : ...` (`ops.cpp` ~6351); negating `i32::MIN` is
/// signed-overflow UB. Each `shift[i]` is therefore checked for `i32::MIN` and
/// rejected with a typed [`Error::OutOfRange`] before dispatch. (Unlike
/// [`roll`] / [`roll_axis`], the per-axis shifts are NOT summed — each applies
/// to its own axis — so there is no sum-overflow path here, only the negation.)
/// As in [`roll`], a shift of exactly `i32::MIN` is rejected as a degenerate
/// magnitude — slightly stricter than mlx-core, which only no-ops it on a
/// size-0 axis — but callers never need it since the shift is taken modulo the
/// axis size.
///
/// Additionally, the MLX C++ overload iterates the axis list with a signed
/// `int i` against `axes.size()` (`ops.cpp` ~6334), so `axes.len()` (and, by
/// the 1:1 pairing, `shift.len()`) exceeding [`i32::MAX`] would overflow that
/// loop counter — that count is guarded with a typed [`Error::CapExceeded`]
/// before dispatch (the C++ `shift.size() < axes.size()` check does *not* bound
/// the count). See the private `check_count` helper.
///
/// Both `shift` and `axes` route empty slices through `dim_ptr`'s static
/// sentinel. Mirrors the tuple-axis form of `mx.roll(a, shift, axis)`
/// (python).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.roll.html).
pub fn roll_axes(a: &Array, shift: &[i32], axes: &[i32]) -> Result<Array> {
  // Enforce 1:1 shift:axis pairing in Rust. MLX only guards the too-few case
  // and silently drops extra shifts, so we reject BOTH directions here for a
  // faithful, predictable contract.
  if shift.len() != axes.len() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "roll_axes: shift.len() vs axes.len()",
      axes.len(),
      shift.len(),
    )));
  }
  // Bound the slice lengths before they become a C++ `Shape`/`vector<int>`
  // iterated by a signed `int i` (~6334). The 1:1 check above means bounding
  // one bounds the other, but guard both explicitly so the soundness invariant
  // is local to each FFI `…_num` argument.
  check_count("roll_axes: axes.len()", "i32::MAX", axes.len())?;
  check_count("roll_axes: shift.len()", "i32::MAX", shift.len())?;
  // Reject any per-axis i32::MIN shift: MLX negates a negative shift via
  // `(-sh)`, and negating i32::MIN is signed-overflow UB.
  if let Some((i, &s)) = shift.iter().enumerate().find(|&(_, &s)| s == i32::MIN) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "roll_axes: shift",
      "no shift may be i32::MIN (its negation is UB in MLX)",
      format_smolstr!("shift[{i}]={s}"),
    )));
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `a.0` is a valid borrowed handle (live for the call, not retained
  // by mlx past it); the out-param was freshly allocated above and is written
  // by this call; `dim_ptr` routes empty `shift`/`axes` through static
  // sentinels so each `(ptr, len)` pair is never singular; the backend rc is
  // surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_roll_axes(
      &mut out.0,
      a.0,
      dim_ptr(shift),
      shift.len(),
      dim_ptr(axes),
      axes.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Construct a new array by tiling `a` by `reps` repetitions per dimension.
/// `reps` is matched against `a`'s shape from the trailing dimension (numpy
/// `tile` semantics: a shorter `reps` is left-padded with 1s, a longer one
/// prepends new leading dims). Mirrors `mx.tile(a, reps)` (python,
/// `python/src/ops.cpp`).
///
/// The `reps` empty-slice case routes through `dim_ptr`'s static sentinel
/// rather than handing mlx-c a Rust dangling pointer.
///
/// **Bounded soundness guard.** MLX builds each output dim as
/// `reps[i] * shape[i]` in unchecked `int` arithmetic (`ops.cpp` line ~1350,
/// where both `reps` and `Shape` hold `int32_t`), so a large `reps[i]` can
/// overflow `int` → C++ signed-overflow UB; a negative `reps[i]` would also
/// flow into the broadcast shape. Before dispatch the wrapper rejects negative
/// reps ([`Error::OutOfRange`]) and recomputes every aligned output dim in
/// `i64` (numpy `tile` alignment: `reps` right-aligned to `a.shape()`, the
/// shorter of the two left-padded with 1s), failing with
/// [`Error::ArithmeticOverflow`] if any product would exceed [`i32::MAX`].
/// `reps[i] == 0` is left as-is — it is valid and yields a size-0 output dim.
///
/// **Intermediate-rank cap (supersedes the final-rank cap).** `tile` does not
/// reshape/broadcast at the final output rank: for every aligned axis whose rep
/// `!= 1` it materialises an EXTRA dim in two intermediate shapes (`expand_shape`
/// / `broad_shape`, `ops.cpp` ~1340-1351) before the final reshape, so those
/// intermediates have rank `max(reps.len(), a.ndim()) + count(reps != 1)` —
/// strictly larger than the final rank `max(reps.len(), a.ndim())`. That larger
/// rank flows into `broadcast_shapes` (`utils.cpp` ~141: `int ndim = s.size()`,
/// a `size_t`→`int` narrowing) and `Reshape::output_shape` (`primitives.cpp`
/// ~3902: `for (int i …)`), both UB once it exceeds [`i32::MAX`]. The wrapper
/// therefore caps the **maximum intermediate rank** with a typed
/// [`Error::CapExceeded`] (see `check_tile_intermediate_rank`); because that
/// rank `>=` the final rank, this single cap bounds every `int`-indexed rank in
/// the op. `a.ndim()` is always small, so in practice this bounds `reps.len()`
/// (and the count of non-unit reps within it).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.tile.html).
pub fn tile(a: &Array, reps: &[i32]) -> Result<Array> {
  // Bound the LARGEST rank MLX materialises during tile BEFORE touching
  // `a.shape()`: `a.ndim()` is the cheap metadata read, so an oversized `reps`
  // is rejected without the `shape()` Vec allocation. This is the intermediate
  // (expand/broadcast) rank `max(reps.len(), a.ndim()) + count(reps != 1)`, NOT
  // just the final rank — the intermediates carry one extra dim per non-unit rep
  // (ops.cpp ~1340-1351) and feed `broadcast_shapes`/`Reshape::output_shape`,
  // which index rank with a C++ `int`. Capping the intermediate rank (>= final
  // rank) covers every `int`-indexed rank in the op in one check.
  check_tile_intermediate_rank(a.ndim(), reps)?;
  // Reject negatives, then bound every aligned output dim (reps[i]*shape[i]) in
  // i64 so the unchecked C++ int multiply (ops.cpp ~1350) cannot overflow.
  // numpy alignment: right-align `reps` against `a.shape()`, treating the
  // shorter operand's leading (missing) dims as an implicit 1. We walk both
  // from the trailing end so the right-alignment falls out for free.
  let shape = a.shape();
  let mut reps_rev = reps.iter().rev();
  let mut shape_rev = shape.iter().rev();
  loop {
    let next_rep = reps_rev.next();
    let next_dim = shape_rev.next();
    if next_rep.is_none() && next_dim.is_none() {
      break; // exhausted both: every aligned dim has been bounded
    }
    // A missing trailing entry on either side is an implicit 1.
    let rep: i64 = match next_rep {
      Some(&r) => {
        if r < 0 {
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "tile: reps",
            "every rep must be non-negative",
            format_smolstr!("{r}"),
          )));
        }
        i64::from(r)
      }
      None => 1,
    };
    // `shape` entries originate from a live mlx array (each in 0..=i32::MAX).
    let dim: i64 = next_dim.map_or(1, |&d| d as i64);
    let out_dim = rep * dim; // both in 0..=i32::MAX, so the product fits in i64
    if out_dim > i64::from(i32::MAX) {
      return Err(Error::ArithmeticOverflow(
        ArithmeticOverflowPayload::with_operands(
          "tile: reps[i] * shape[i] output dim",
          "i32",
          [("rep", rep as u64), ("dim", dim as u64)],
        ),
      ));
    }
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `a.0` is a valid borrowed handle (live for the call, not retained
  // by mlx past it); the out-param was freshly allocated above and is written
  // by this call; `dim_ptr` routes an empty `reps` through a static sentinel
  // so the `(ptr, len)` pair is never singular; the backend rc is surfaced
  // via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_tile(&mut out.0, a.0, dim_ptr(reps), reps.len(), default_stream())
  })?;
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;

  // Boundary test for the `int`-loop count guard (#259 Codex MEDIUM). The
  // real overflow path needs a slice with > i32::MAX entries, which is
  // impractical to allocate (~8GB+), so we exercise the guard at its boundary
  // by feeding `check_count` a synthetic `len` — no allocation required. This
  // is the source of truth for the per-op `…_num` guards (roll_axes' shift/axes
  // lengths, tile's output rank), which are thin forwards to this helper.
  #[test]
  fn check_count_boundary() {
    // At and below the cap: accepted (every representable in-range count must
    // pass so no legitimate caller is rejected).
    assert!(check_count("t", "i32::MAX", 0).is_ok());
    assert!(check_count("t", "i32::MAX", i32::MAX as usize).is_ok());
    // One past the cap: rejected as a typed CapExceeded carrying the cap and
    // the observed count (never UB, never a panic).
    let over = i32::MAX as usize + 1;
    match check_count("ctx", "i32::MAX", over) {
      Err(Error::CapExceeded(p)) => {
        assert_eq!(p.context(), "ctx");
        assert_eq!(p.cap_name(), "i32::MAX");
        assert_eq!(p.cap(), i32::MAX as u64);
        assert_eq!(p.observed(), over as u64);
      }
      other => panic!("expected Err(CapExceeded) one past the cap, got {other:?}"),
    }
  }

  // Boundary test for tile's INTERMEDIATE-rank guard (#259 Codex HIGH). The
  // real overflow needs a `reps` slice with ~i32::MAX entries (impractical to
  // allocate), so the cap logic is exercised synthetically: a small `reps`
  // whose non-unit count is known, paired with a crafted `ndim`, drives
  // `aligned_rank + count(reps != 1)` across the boundary without any array.
  // This proves the EXTRA intermediate dim per non-unit rep is what tips the
  // sum over the cap — the exact channel the final-rank cap missed.
  #[test]
  fn tile_intermediate_rank_boundary() {
    let cap = i32::MAX as usize;

    // Final rank at the cap, but ONE non-unit rep adds an extra intermediate
    // dim -> intermediate rank cap+1 -> rejected. (A final-rank-only cap would
    // have ACCEPTED this: aligned_rank == cap passes, but the broadcast/expand
    // intermediate is cap+1.) This is the regression the HIGH finding targets.
    match check_tile_intermediate_rank(cap, &[2]) {
      Err(Error::CapExceeded(p)) => {
        assert_eq!(p.cap(), cap as u64);
        assert_eq!(p.observed(), cap as u64 + 1); // aligned cap + 1 extra dim
      }
      other => panic!("expected CapExceeded for cap+non_unit_rep, got {other:?}"),
    }

    // Same aligned rank, but the rep is 1 (a no-op axis): NO extra dim, so the
    // intermediate rank == aligned rank == cap -> accepted. Confirms padded/unit
    // reps never inflate the rank.
    assert!(check_tile_intermediate_rank(cap, &[1]).is_ok());

    // The count is over EXPLICIT non-unit reps: three non-unit reps add three
    // extra dims. ndim small so aligned_rank == reps.len() == 5; intermediate
    // rank == 5 + 3 == 8. Well under the cap -> accepted (sanity on the formula).
    assert!(check_tile_intermediate_rank(1, &[2, 1, 3, 1, 4]).is_ok());

    // Drive the count itself over the edge: aligned_rank == cap-2 (via ndim),
    // plus 3 non-unit reps == cap+1 -> rejected. Proves it is the *summed* count
    // (not a single dim) that is bounded.
    match check_tile_intermediate_rank(cap - 2, &[2, 3, 4]) {
      Err(Error::CapExceeded(p)) => assert_eq!(p.observed(), cap as u64 + 1),
      other => panic!("expected CapExceeded for cap-2 + 3 non-unit reps, got {other:?}"),
    }

    // Exactly at the cap with non-unit reps accounted for: aligned_rank ==
    // cap-1 + 1 non-unit rep == cap -> accepted (the boundary is inclusive).
    assert!(check_tile_intermediate_rank(cap - 1, &[2]).is_ok());

    // Empty reps / unit reps with small ndim: trivially fine.
    assert!(check_tile_intermediate_rank(0, &[]).is_ok());
    assert!(check_tile_intermediate_rank(3, &[]).is_ok());
  }
}
