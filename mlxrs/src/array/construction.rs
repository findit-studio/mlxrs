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

  /// Creates an array of `shape` filled with `value` (dtype `T`).
  ///
  /// `value` is a `T`, so the fill is **exact** and an out-of-range value is a
  /// *compile* error (you cannot write `300u8`). The scalar handed to
  /// `mlx_full` is a 0-d `T` array built via [`Array::from_slice`], so mlx
  /// never casts, rounds, or wraps the value. (`T = f64` only runs on a CPU
  /// stream — Metal has no native f64, as for any f64 op.)
  pub fn full<T>(shape: &impl IntoShape, value: T) -> Result<Self>
  where
    T: Element,
  {
    // The fill scalar is an exact 0-d `T` array. `from_slice` is itself
    // stripped-ctor-safe (#215), validates, and installs the error handler, so
    // the `mlx_array_new` / `mlx_full` calls below are guarded regardless of
    // order. `mlx_full` broadcasts the scalar to `shape`; the scalar is already
    // dtype `T`, so no value cast can occur.
    let scalar = Self::from_slice(&[value], &[0i32; 0])?;
    crate::error::ensure_handler_installed();
    shape.with_shape(|s| {
      validate_dims(s)?;
      // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL
      // ctx); it is wrapped in the RAII newtype FIRST so an early return / panic
      // frees it, then populated by the following call.
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
      // SAFETY: `scalar.0` is a live, valid handle (the `from_slice` array,
      // borrowed for the call, not retained by mlx past it); the out-param was
      // freshly allocated above and is written by this call; the backend rc is
      // surfaced via `check()`.
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

  /// Creates an `n × m` matrix with ones on the `k`-th diagonal, zeros elsewhere.
  ///
  /// Mirrors mlx-python `eye(n, m=None, k=0)`: `n` rows, `m` columns
  /// (defaults to `n`, giving a square matrix), and `k` selects the diagonal
  /// — `0` is the main diagonal, `> 0` shifts above it, `< 0` below it.
  pub fn eye<T>(n: usize, m: Option<usize>, k: i32) -> Result<Self>
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
    let m = m.unwrap_or(n);
    let n_i32 = i32::try_from(n).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "Array::eye: n",
        "must fit in i32",
        format_smolstr!("{n}"),
      ))
    })?;
    let m_i32 = i32::try_from(m).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "Array::eye: m",
        "must fit in i32",
        format_smolstr!("{m}"),
      ))
    })?;
    // mlx's `eye` evaluates `-k` (`-k >= n` and `std::max(0, -k)` — see the
    // vendored `mlx/ops.cpp`), so `k == i32::MIN` overflows in C++ (UB) from a
    // safe call. Reject it pre-FFI. (#259 / Codex; upstream: mlx eye negates k.)
    if k == i32::MIN {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Array::eye: k",
        "must be greater than i32::MIN (mlx evaluates -k, which overflows there)",
        format_smolstr!("{k}"),
      )));
    }
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    // `k` is passed through directly: a negative `k` is the valid lower diagonal.
    check(unsafe {
      mlxrs_sys::mlx_eye(
        &mut out.0,
        n_i32,
        m_i32,
        k,
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
