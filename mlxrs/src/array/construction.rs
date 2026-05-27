//! Array constructors — generic over `T: Element` where applicable.

use smol_str::format_smolstr;

use crate::{
  array::Array,
  dtype::{Dtype, Element},
  error::{
    ArithmeticOverflowPayload, Error, LengthMismatchPayload, OutOfRangePayload, Result, check,
    check_handle,
  },
  shape::{IntoShape, dim_ptr, validate_dims},
  stream::default_stream,
};

/// RAII guard for a temporary `mlx_array` (e.g. the scalar val passed to mlx_full).
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

/// Substitutes a real-`T` static for an empty data slice's dangling pointer,
/// keeping zero-element FFI calls UB-free. mlx-c reinterprets the `void*` as
/// `*const T` based on dtype before constructing `mlx::core::array`, so the
/// pointer must be associated with a real `T` allocation — a `[u8]` cast to
/// `*const T` is not enough (Codex PR #5 round-2 finding).
#[inline]
fn data_ptr<T>(data: &[T]) -> *const T
where
  T: Element,
{
  if data.is_empty() {
    T::sentinel_ptr()
  } else {
    data.as_ptr()
  }
}

impl Array {
  /// Creates an array filled with ones. Dtype is determined by the type parameter.
  ///
  /// ```no_run
  /// # fn run() -> mlxrs::Result<()> {
  /// let a = mlxrs::Array::ones::<f32>(&(3, 3))?;
  /// # Ok(()) }
  /// ```
  pub fn ones<T>(shape: &impl IntoShape) -> Result<Self>
  where
    T: Element,
  {
    // CRITICAL: must be the first call in this function. The very first FFI
    // call below (`mlx_array_new()`) is wrapped in mlx-c's standard
    // `try/catch -> mlx_error(e.what())` boilerplate; without an installed
    // handler, mlx-c's default handler `printf + exit(-1)` would fire on a
    // throw and terminate the process before `check()` could observe the
    // failure. See issue #215 (stripped-ctor regression history). Although
    // `default_stream()` (below) also installs the handler, it runs AFTER
    // the raw FFI ctor here.
    //
    // TEST COVERAGE: smoke-only (see `stripped_ctor_constructors`); the
    // install-at-call-site requirement is enforced by code review, NOT by
    // an executable regression test (a normal-input smoke does not throw
    // in `mlx_array_new()`, and the AST/syn-based structural alternative
    // is forbidden by issue #215).
    crate::error::ensure_handler_installed();
    shape.with_shape(|s| {
      validate_dims(s)?;
      // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
      // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
      // early return / panic frees it, then populated by the following call.
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
      // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
      // not retained by mlx past it); the out-param was freshly allocated above
      // and is written by this call; the backend rc is surfaced via `check()`.
      check(unsafe {
        mlxrs_sys::mlx_ones(
          &mut out.0,
          dim_ptr(s),
          s.len(),
          mlxrs_sys::mlx_dtype::from(T::DTYPE),
          default_stream(),
        )
      })?;
      Ok(out)
    })
  }

  /// Creates an array filled with zeros.
  pub fn zeros<T>(shape: &impl IntoShape) -> Result<Self>
  where
    T: Element,
  {
    // CRITICAL: must be the first call in this function. See `Array::ones`
    // for the full rationale — the raw `mlx_array_new()` below is in mlx-c's
    // `try/catch -> mlx_error` wrapper and runs BEFORE `default_stream()`'s
    // handler install.
    //
    // TEST COVERAGE: smoke-only (see `stripped_ctor_constructors`); the
    // install-at-call-site requirement is enforced by code review, NOT by
    // an executable regression test.
    crate::error::ensure_handler_installed();
    shape.with_shape(|s| {
      validate_dims(s)?;
      // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
      // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
      // early return / panic frees it, then populated by the following call.
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
      // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
      // not retained by mlx past it); the out-param was freshly allocated above
      // and is written by this call; the backend rc is surfaced via `check()`.
      check(unsafe {
        mlxrs_sys::mlx_zeros(
          &mut out.0,
          dim_ptr(s),
          s.len(),
          mlxrs_sys::mlx_dtype::from(T::DTYPE),
          default_stream(),
        )
      })?;
      Ok(out)
    })
  }

  /// Creates an array filled with `value` (cast to f32 internally).
  ///
  /// `mlx_full` takes a scalar `mlx_array` for `vals`; this helper builds
  /// the scalar via `mlx_array_new_float32(value)` and frees it on return.
  pub fn full<T>(shape: &impl IntoShape, value: f32) -> Result<Self>
  where
    T: Element,
  {
    // CRITICAL: must be the first call in this function. `Array::full` is the
    // worst-case constructor for the stripped-ctor exit(-1) regression: the
    // raw `mlx_array_new_float32(value)` below heap-allocates a fresh
    // `mlx::core::array(val)` and is in mlx-c's standard
    // `try/catch -> mlx_error(e.what())` wrapper, so a `std::bad_alloc`
    // (or any other backend throw) would invoke mlx-c's default
    // `printf + exit(-1)` handler and terminate the process before
    // `check_handle` could observe the NULL handle. The later
    // `mlx_full(..., default_stream())` would install the handler, but it
    // is too late for failures in the scalar constructor itself. See issue
    // #215 (stripped-ctor regression history).
    //
    // TEST COVERAGE: smoke-only (see `stripped_ctor_constructors`); the
    // install-at-call-site requirement is enforced by code review, NOT by
    // an executable regression test (a normal-input smoke does not throw
    // — exercising `std::bad_alloc` requires an allocator-shim test build
    // out of scope here, and the AST/syn-based structural alternative is
    // forbidden by issue #215).
    crate::error::ensure_handler_installed();
    shape.with_shape(|s| {
      validate_dims(s)?;
      // SAFETY: fallible sentinel-handle ctor: the error handler is installed before
      // the call (no default `printf+exit`), the raw handle is wrapped in its
      // RAII guard before the NULL-ctx check (free is a defined no-op on a
      // NULL ctx), and the inputs are valid for the duration of the call.
      let scalar = ScalarGuard(unsafe { mlxrs_sys::mlx_array_new_float32(value) });
      // Explicit NULL-ctx check on the scalar handle BEFORE passing it to
      // `mlx_full`. `mlx_array_new_float32` reports failure via the
      // sentinel-handle pattern (NULL `ctx`, plus a message stashed in the
      // TLS slot by our handler). Without this check, a scalar-constructor
      // OOM would silently flow into `mlx_full(scalar=NULL_ctx, ...)` and
      // surface as a generic backend error instead of the original
      // bad-alloc message — and on a stripped-ctor build the original
      // exit(-1) would already have fired in the line above (which is why
      // the `ensure_handler_installed()` call at the top of this function
      // is the load-bearing fix; this NULL check is a secondary
      // correctness improvement that propagates the original allocator
      // error rather than a downstream cascade).
      if scalar.0.ctx.is_null() {
        return Err(crate::error::take_last().unwrap_or(Error::Backend(
          "mlx_array_new_float32 returned null handle".into(),
        )));
      }
      // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
      // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
      // early return / panic frees it, then populated by the following call.
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
      // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
      // not retained by mlx past it); the out-param was freshly allocated above
      // and is written by this call; the backend rc is surfaced via `check()`.
      check(unsafe {
        mlxrs_sys::mlx_full(
          &mut out.0,
          dim_ptr(s),
          s.len(),
          scalar.0,
          mlxrs_sys::mlx_dtype::from(T::DTYPE),
          default_stream(),
        )
      })?;
      Ok(out)
    })
  }

  /// Creates an `n×n` identity matrix.
  pub fn eye<T>(n: usize) -> Result<Self>
  where
    T: Element,
  {
    // CRITICAL: must be the first call in this function. See `Array::ones`
    // for the full rationale — the raw `mlx_array_new()` below is in mlx-c's
    // `try/catch -> mlx_error` wrapper and runs BEFORE `default_stream()`'s
    // handler install.
    //
    // TEST COVERAGE: smoke-only (see `stripped_ctor_constructors`); the
    // install-at-call-site requirement is enforced by code review, NOT by
    // an executable regression test.
    crate::error::ensure_handler_installed();
    let n_i32 = i32::try_from(n).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "Array::eye: n",
        "must fit in i32",
        format_smolstr!("{n}"),
      ))
    })?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_eye(
        &mut out.0,
        n_i32,
        n_i32,
        0,
        mlxrs_sys::mlx_dtype::from(T::DTYPE),
        default_stream(),
      )
    })?;
    Ok(out)
  }

  /// Creates a 1-D f32 array of evenly-spaced values in `[start, stop)`.
  pub fn arange(start: f32, stop: f32, step: f32) -> Result<Self> {
    // CRITICAL: must be the first call in this function. See `Array::ones`
    // for the full rationale — the raw `mlx_array_new()` below is in mlx-c's
    // `try/catch -> mlx_error` wrapper and runs BEFORE `default_stream()`'s
    // handler install.
    //
    // TEST COVERAGE: smoke-only (see `stripped_ctor_constructors`); the
    // install-at-call-site requirement is enforced by code review, NOT by
    // an executable regression test.
    crate::error::ensure_handler_installed();
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_arange(
        &mut out.0,
        f64::from(start),
        f64::from(stop),
        f64::from(step),
        mlxrs_sys::mlx_dtype::from(Dtype::F32),
        default_stream(),
      )
    })?;
    Ok(out)
  }

  /// Creates a 1-D f32 array of `num` evenly-spaced values in `[start, stop]`.
  pub fn linspace(start: f32, stop: f32, num: usize) -> Result<Self> {
    // CRITICAL: must be the first call in this function. See `Array::ones`
    // for the full rationale — the raw `mlx_array_new()` below is in mlx-c's
    // `try/catch -> mlx_error` wrapper and runs BEFORE `default_stream()`'s
    // handler install.
    //
    // TEST COVERAGE: smoke-only (see `stripped_ctor_constructors`); the
    // install-at-call-site requirement is enforced by code review, NOT by
    // an executable regression test.
    crate::error::ensure_handler_installed();
    let n_i32 = i32::try_from(num).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "Array::linspace: num",
        "must fit in i32",
        format_smolstr!("{num}"),
      ))
    })?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_linspace(
        &mut out.0,
        f64::from(start),
        f64::from(stop),
        n_i32,
        mlxrs_sys::mlx_dtype::from(Dtype::F32),
        default_stream(),
      )
    })?;
    Ok(out)
  }

  /// Creates an array from a contiguous `&[T]` buffer plus shape. Buffer is COPIED.
  pub fn from_slice<T>(data: &[T], shape: &impl IntoShape) -> Result<Self>
  where
    T: Element,
  {
    // CRITICAL: must be the first call in this function. `from_slice` is the
    // only `Array` constructor that does NOT go through `default_stream()`
    // (so there is no later installer to rescue a stripped `#[ctor]`); its
    // only FFI call, `mlx_array_new_data`, is in mlx-c's standard
    // `try/catch -> mlx_error` wrapper and can throw on copy/alloc failure
    // (see `mlxrs-sys/vendor/mlx-c/mlx/c/array.cpp`). Without an installed
    // handler, mlx-c's default `printf + exit(-1)` would terminate the
    // process before `check_handle` could observe the NULL handle. See
    // issue #215.
    //
    // TEST COVERAGE: smoke-only (see `stripped_ctor_constructors`); the
    // install-at-call-site requirement is enforced by code review, NOT by
    // an executable regression test (a normal-input smoke does not throw
    // — exercising `std::bad_alloc` requires an allocator-shim test build
    // out of scope here, and the AST/syn-based structural alternative is
    // forbidden by issue #215).
    crate::error::ensure_handler_installed();
    shape.with_shape(|s| {
      // FFI safety boundary: validate the slice we're about to hand to
      // mlx_array_new_data. validate_dims rules out negative dims (so the
      // `as usize` cast below is well-defined); checked_mul rules out
      // release-build wrapping on the shape product, which could otherwise
      // match data.len() and pass the equality guard with an undersized
      // buffer.
      validate_dims(s)?;
      // Carry the accumulated product, the offending dim, and its index in the
      // overflow payload so a `Display` rendering identifies which dim multiply
      // tripped the cap (vs. just naming the operation).
      let total: usize = s.iter().enumerate().try_fold(1usize, |acc, (idx, &d)| {
        let d_usize = d as usize;
        acc.checked_mul(d_usize).ok_or_else(|| {
          Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
            "Array::from_slice: shape product",
            "usize",
            [
              ("acc", acc as u64),
              ("dim", d_usize as u64),
              ("dim_index", idx as u64),
            ],
          ))
        })
      })?;
      if total != data.len() {
        return Err(Error::LengthMismatch(LengthMismatchPayload::new(
          "Array::from_slice: shape product vs data.len()",
          total,
          data.len(),
        )));
      }
      let dim_i32 = i32::try_from(s.len()).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "Array::from_slice: ndim",
          "must fit in i32",
          format_smolstr!("{}", s.len()),
        ))
      })?;
      // SAFETY: fallible sentinel-handle ctor: the error handler is installed first;
      // the (data, dims, ndim) triple was validated above (shape product ==
      // data.len(), non-negative dims, real `T` allocation via `data_ptr`'s
      // typed sentinel for the empty case); mlx-c copies the buffer in.
      let arr = unsafe {
        mlxrs_sys::mlx_array_new_data(
          data_ptr(data).cast::<std::ffi::c_void>(),
          dim_ptr(s),
          dim_i32,
          mlxrs_sys::mlx_dtype::from(T::DTYPE),
        )
      };
      check_handle(arr)
    })
  }
}
