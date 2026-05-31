//! Misc ops: argmax/argmin (optional-axis archetype, U32 output),
//! cumulative reductions, sort/argsort/topk/partition, clip, *_like
//! constructors, and astype.

use std::ffi::c_int;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Error, FfiNullHandlePayload, Result, check},
  stream::default_stream,
};

/// RAII guard for a temporary scalar `mlx_array` (e.g. clip bounds, full_like
/// fill value). Local twin of `array::construction::ScalarGuard`; duplicated
/// here intentionally so this module stays self-contained — promotion to a
/// shared helper waits for a 3rd consumer.
struct ScalarGuard(mlxrs_sys::mlx_array);
impl Drop for ScalarGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. Runs during `Drop` /
    // thread teardown: must not touch TLS, call `check()`, panic, or unwind
    // across `extern "C"`; the rc is discarded silently per the crate's
    // Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_array_free(self.0);
    }
  }
}

/// Checked f32 scalar constructor. `mlx_array_new_float32` is a fallible
/// sentinel-handle FFI: it returns NULL `ctx` on allocation failure (and may
/// invoke the error handler on the way out). This helper:
///   1. Installs the safe error handler before the call so a stripped or
///      disabled `#[ctor]` cannot let the default `printf+exit` fire.
///   2. Wraps the raw handle in `ScalarGuard` immediately on return — RAII
///      coverage for the rare panic path between the FFI call and the null
///      check. `ScalarGuard::drop` calls `mlx_array_free`, which is a defined
///      no-op on a NULL `ctx` (it dispatches to `delete (T*)nullptr`), so
///      wrapping a null handle is safe.
///   3. Checks `ctx.is_null` and drains `error::LAST` into `Err` — the guard
///      then drops harmlessly.
///
/// See `concatenate` in `ops::shape` for the same pattern on `mlx_vector_array`.
fn checked_scalar_f32(value: f32) -> Result<ScalarGuard> {
  crate::error::ensure_handler_installed();
  // SAFETY: fallible sentinel-handle ctor: the error handler is installed before
  // the call (no default `printf+exit`), the raw handle is wrapped in its
  // RAII guard before the NULL-ctx check (free is a defined no-op on a
  // NULL ctx), and the inputs are valid for the duration of the call.
  let raw = unsafe { mlxrs_sys::mlx_array_new_float32(value) };
  // Wrap first for RAII; see step 2 above.
  let guard = ScalarGuard(raw);
  if raw.ctx.is_null() {
    return Err(
      crate::error::LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::FfiNullHandle(FfiNullHandlePayload::new(
          "mlx_array_new_float32",
        ))),
    );
  }
  Ok(guard)
}

/// Index of the maximum value, optionally along `axis`. Output dtype is U32.
///
/// CANONICAL OPTIONAL-AXIS TEMPLATE — pattern: dispatch between `mlx_argmax`
/// (full reduction) and `mlx_argmax_axis` (per-axis) based on `Option<i32>`.
/// Every fn with optional-axis semantics (argmin, all_axis, any_axis) follows
/// this shape.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argmax.html).
pub fn argmax(a: &Array, axis: Option<i32>, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    match axis {
      Some(ax) => {
        mlxrs_sys::mlx_argmax_axis(&mut out.0, a.0, ax as c_int, keepdims, default_stream())
      }
      None => mlxrs_sys::mlx_argmax(&mut out.0, a.0, keepdims, default_stream()),
    }
  })?;
  Ok(out)
}

/// Index of the minimum value, optionally along `axis`. Output dtype is U32
/// (same as argmax — mlx returns unsigned indices for both).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argmin.html).
pub fn argmin(a: &Array, axis: Option<i32>, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    match axis {
      Some(ax) => {
        mlxrs_sys::mlx_argmin_axis(&mut out.0, a.0, ax as c_int, keepdims, default_stream())
      }
      None => mlxrs_sys::mlx_argmin(&mut out.0, a.0, keepdims, default_stream()),
    }
  })?;
  Ok(out)
}

/// Cumulative sum along `axis`. `reverse` flips the scan direction;
/// `inclusive` includes the current index in the running total.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cumsum.html).
pub fn cumsum(a: &Array, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_cumsum(
      &mut out.0,
      a.0,
      axis as c_int,
      reverse,
      inclusive,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Cumulative product along `axis`. See [`cumsum`] for `reverse`/`inclusive`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cumprod.html).
pub fn cumprod(a: &Array, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_cumprod(
      &mut out.0,
      a.0,
      axis as c_int,
      reverse,
      inclusive,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Cumulative maximum along `axis`. See [`cumsum`] for `reverse`/`inclusive`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cummax.html).
pub fn cummax(a: &Array, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_cummax(
      &mut out.0,
      a.0,
      axis as c_int,
      reverse,
      inclusive,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Cumulative minimum along `axis`. See [`cumsum`] for `reverse`/`inclusive`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cummin.html).
pub fn cummin(a: &Array, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_cummin(
      &mut out.0,
      a.0,
      axis as c_int,
      reverse,
      inclusive,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Sort the flattened array in ascending order.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sort.html).
pub fn sort(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_sort(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Sort along `axis` in ascending order.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sort.html).
pub fn sort_axis(a: &Array, axis: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_sort_axis(&mut out.0, a.0, axis as c_int, default_stream()) })?;
  Ok(out)
}

/// Indices that would sort the flattened array. Output dtype is U32.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argsort.html).
pub fn argsort(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_argsort(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Indices that would sort along `axis`. Output dtype is U32.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argsort.html).
pub fn argsort_axis(a: &Array, axis: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_argsort_axis(&mut out.0, a.0, axis as c_int, default_stream()) })?;
  Ok(out)
}

/// Top-`k` elements of the flattened array. Returned values are unsorted
/// among themselves (matching mlx Python semantics).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.topk.html).
pub fn topk(a: &Array, k: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_topk(&mut out.0, a.0, k as c_int, default_stream()) })?;
  Ok(out)
}

/// Top-`k` elements along `axis`. Returned values are unsorted among themselves.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.topk.html).
pub fn topk_axis(a: &Array, k: i32, axis: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_topk_axis(&mut out.0, a.0, k as c_int, axis as c_int, default_stream())
  })?;
  Ok(out)
}

/// Partition the flattened array around index `kth`: elements at positions
/// `< kth` are ≤ the `kth`-positioned element; positions `> kth` are ≥. Order
/// within each side is unspecified.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.partition.html).
pub fn partition(a: &Array, kth: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_partition(&mut out.0, a.0, kth as c_int, default_stream()) })?;
  Ok(out)
}

/// Partition along `axis` around index `kth`. See [`partition`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.partition.html).
pub fn partition_axis(a: &Array, kth: i32, axis: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_partition_axis(
      &mut out.0,
      a.0,
      kth as c_int,
      axis as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Indices that would partition the flattened array around `kth`. Output
/// dtype is U32. See [`partition`] for the partition semantics.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argpartition.html).
pub fn argpartition(a: &Array, kth: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_argpartition(&mut out.0, a.0, kth as c_int, default_stream()) })?;
  Ok(out)
}

/// Indices that would partition along `axis` around `kth`. Output dtype is
/// U32. See [`partition`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argpartition.html).
pub fn argpartition_axis(a: &Array, kth: i32, axis: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_argpartition_axis(
      &mut out.0,
      a.0,
      kth as c_int,
      axis as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Softmax along `axis`. `precise` uses the higher-precision accumulation
/// path (matches mlx-python's `precise` kwarg).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.softmax.html).
pub fn softmax_axis(a: &Array, axis: i32, precise: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_softmax_axis(&mut out.0, a.0, axis as c_int, precise, default_stream())
  })?;
  Ok(out)
}

/// Clamp every element of `a` into `[a_min, a_max]`. Bounds are themselves
/// `mlx_array`s (broadcast against `a`); see [`clip_with_scalar`] for the
/// scalar-bounds ergonomic form.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.clip.html).
pub fn clip(a: &Array, a_min: &Array, a_max: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_clip(&mut out.0, a.0, a_min.0, a_max.0, default_stream()) })?;
  Ok(out)
}

/// Clamp every element of `a` into `[min, max]` using f32 scalar bounds.
/// Wraps each scalar in a temporary `mlx_array_new_float32` handle for the
/// duration of the call.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.clip.html).
pub fn clip_with_scalar(a: &Array, min: f32, max: f32) -> Result<Array> {
  let lo = checked_scalar_f32(min)?;
  let hi = checked_scalar_f32(max)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_clip(&mut out.0, a.0, lo.0, hi.0, default_stream()) })?;
  Ok(out)
}

/// Array of ones with the same shape and dtype as `a`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.ones_like.html).
pub fn ones_like(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_ones_like(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Array of zeros with the same shape and dtype as `a`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.zeros_like.html).
pub fn zeros_like(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_zeros_like(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Array filled with `value` (cast to f32 internally), with the same shape
/// and dtype as `a`. The output dtype is `a`'s dtype; the scalar is cast
/// during the FFI call.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.full_like.html).
pub fn full_like(a: &Array, value: f32) -> Result<Array> {
  let dtype = mlxrs_sys::mlx_dtype::from(a.dtype()?);
  let scalar = checked_scalar_f32(value)?;
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_full_like(&mut out.0, a.0, scalar.0, dtype, default_stream()) })?;
  Ok(out)
}

/// Cast `a` to `dtype`. Returns a new array; the source is unchanged.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.astype.html).
pub fn astype(a: &Array, dtype: Dtype) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_astype(
      &mut out.0,
      a.0,
      mlxrs_sys::mlx_dtype::from(dtype),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Bit-preserving dtype reinterpretation. Mirrors `mx.view`
/// (`mlx/ops.cpp` — `array view(const array& a, const Dtype& dtype, ...)`)
/// and the python `mx.core.view` binding. Unlike [`astype`] (which performs
/// a value-preserving numeric cast), `view` keeps the underlying bit-pattern
/// intact: when source and target dtypes are the same width, the shape is
/// preserved and a signed/unsigned reinterpret round-trips losslessly. For
/// different-width dtypes the last axis is rescaled to keep total byte count
/// (mlx requires `last_axis_bytes` to be a multiple of the target dtype's
/// element size; not exercised by mlxrs callers today but documented for
/// parity).
///
/// Use this (not [`astype`]) when you need an i32 → u32 (or u32 → i32)
/// reinterpret that preserves the sign bit as a high data bit — the AWQ
/// `qweight` / `qzeros` shift-and-mask pipeline depends on negative i32
/// values keeping their high bit through the cast.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.view.html).
pub fn view(a: &Array, dtype: Dtype) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_view(
      &mut out.0,
      a.0,
      mlxrs_sys::mlx_dtype::from(dtype),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Stop-gradient: forward identity that severs the backward pass. The returned
/// array has the same shape, dtype, and values as `a`, but is a leaf in the
/// computation graph — gradients do not flow through it (mlx inserts a
/// `StopGradient` primitive whose VJP is zero). Use to freeze a sub-expression
/// (e.g. a target/detached activation) during differentiation.
///
/// DIRECT-ARG SOUNDNESS (issue #266): no bounded guard needed — `stop_gradient`
/// is unary with no scalar args and performs no arithmetic; it forwards `a`'s
/// shape/dtype verbatim into the new node.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stop_gradient.html).
pub fn stop_gradient(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_stop_gradient(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}
