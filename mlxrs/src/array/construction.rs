//! Array constructors — generic over `T: Element` where applicable.

use smol_str::format_smolstr;

use crate::{
  array::Array,
  dtype::{Dtype, Element},
  error::{
    ArithmeticOverflowPayload, Error, LengthMismatchPayload, NonFiniteScalarPayload,
    OutOfRangePayload, Result, UnsupportedDtypePayload, check, check_handle,
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

  /// Creates a 1-D array of evenly-spaced values in `[start, stop)`.
  ///
  /// The output dtype is `T`; `start`, `stop`, and `step` are taken as `f64`
  /// (any `Into<f64>` — `f32`/`i32`/… all pass), matching mlx-python's FFI, so
  /// an integer range stays exact up to 2^53 rather than rounding through the
  /// `f32` 2^24 window. A wrong-direction or zero-length range yields an empty
  /// array; an infinite `step` in the correct direction yields `[start]` (mlx's
  /// special case).
  ///
  /// # Soundness
  /// This mirrors the case ladder of vendored mlx `arange` (`ops.cpp` +
  /// `backend/cpu/arange.h`) and rejects, up front, every input that would reach
  /// one of mlx's unchecked C++ `static_cast`s from a safe call:
  /// - `Bool` is unsupported (mlx throws for any range, including empty);
  /// - a NaN or infinite `start`/`stop` has no well-defined length;
  /// - a zero `step` yields a non-finite length mlx `static_cast`s to `int` (UB);
  /// - the length `ceil((stop - start) / step)` is only guarded `> INT_MAX`, so a
  ///   non-finite or huge-negative length would be an out-of-range cast — that is
  ///   rejected, or returned empty for a wrong-direction range;
  /// - mlx narrows the seeds `start` and `start + step` from `double` into `T` —
  ///   a `static_cast` that is UB outside `T`'s range for every dtype narrower
  ///   than f64 (a float above its finite max, or an integer whose truncated
  ///   value escapes `[MIN, MAX]`). For an integer `T`, mlx then accumulates
  ///   `first + i * delta` in the promoted int; i32/i64 can overflow that
  ///   arithmetic (UB), so the `delta` and post-last value are additionally
  ///   range-checked via an exact `i128` model of the recurrence.
  pub fn arange<T>(
    start: impl Into<f64>,
    stop: impl Into<f64>,
    step: impl Into<f64>,
  ) -> Result<Self>
  where
    T: Element,
  {
    let start: f64 = start.into();
    let stop: f64 = stop.into();
    let step: f64 = step.into();
    // mlx rejects bool arange for EVERY range (vendored `ops.cpp`); match it here
    // so the empty fast path below cannot mask it.
    if T::DTYPE == Dtype::Bool {
      return Err(Error::UnsupportedDtype(UnsupportedDtypePayload::new(
        "Array::arange",
        Dtype::Bool,
        ARANGE_SUPPORTED_DTYPES,
      )));
    }
    // mlx throws "Cannot compute length" for a NaN/infinite `start`/`stop` (an
    // infinite `step` is the special case below, not an error).
    if start.is_nan() || stop.is_nan() || step.is_nan() {
      let v = if start.is_nan() {
        start
      } else if stop.is_nan() {
        stop
      } else {
        step
      };
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "Array::arange: start/stop/step must not be NaN",
        v,
      )));
    }
    if start.is_infinite() || stop.is_infinite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "Array::arange: start/stop must be finite",
        if start.is_infinite() { start } else { stop },
      )));
    }
    if step == 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Array::arange: step",
        "must be non-zero",
        format_smolstr!("{step}"),
      )));
    }
    // mlx: an infinite `step` in the correct direction emits the single value
    // `start` (`array({start}, dtype)`, one `double` -> `T` cast); a
    // wrong-direction infinite step is an empty range.
    if step.is_infinite() {
      let correct_dir = (step > 0.0 && start < stop) || (step < 0.0 && start > stop);
      if !correct_dir {
        return Self::from_slice::<T>(&[], &[0i32]);
      }
      if !representable_in(start, T::DTYPE) {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "Array::arange: start",
          "must be representable in the output dtype",
          format_smolstr!("{start}"),
        )));
      }
      // fall through to `mlx_arange`, which returns `[start]`.
    } else {
      // Finite step: model mlx's length `static_cast`.
      let real_size = ((stop - start) / step).ceil();
      // `start`/`stop` finite + `step` finite-nonzero ⇒ `real_size` is finite;
      // the `is_finite` guard is defense-in-depth.
      if !real_size.is_finite() || real_size > f64::from(i32::MAX) {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "Array::arange: range length",
          "must be finite and not exceed i32::MAX",
          format_smolstr!("{real_size}"),
        )));
      }
      if real_size <= 0.0 {
        // Wrong-direction / zero-length → empty. Built directly (NOT via the
        // FFI): mlx would still `static_cast` `start` into `T` for a zero-length
        // arange, which would be UB for an out-of-range `start`.
        return Self::from_slice::<T>(&[], &[0i32]);
      }
      // mlx narrows the two seeds `start` and `start + step` from `double` into
      // `T` (a `static_cast`, UB outside the destination range) for EVERY dtype
      // narrower than f64 — integer (truncating) and float alike.
      if !representable_in(start, T::DTYPE) || !representable_in(start + step, T::DTYPE) {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "Array::arange: endpoint",
          "must be representable in the output dtype",
          format_smolstr!("start={start}, start+step={}", start + step),
        )));
      }
      // For an integer `T`, mlx then forms `delta = next - first` and accumulates
      // `first + i * delta` — the subtraction and every step run in the PROMOTED
      // C++ int. For i8/i16 the promoted operands stay tiny and narrow back to
      // `T` each step (defined wrapping), and unsigned wraps; only i32 (promotes
      // to `int`) and i64 can overflow that arithmetic (UB), so model those two
      // exactly in `i128` and require BOTH the `delta` and the post-last value to
      // fit the promoted int. The seeds are in `T` range (checked above), so the
      // `trunc` casts are exact and bound the whole monotonic sequence.
      if matches!(T::DTYPE, Dtype::I32 | Dtype::I64) {
        let first = start.trunc() as i128;
        let next = (start + step).trunc() as i128;
        let delta = next - first;
        let post_last = first + (real_size as i128) * delta;
        let (plo, phi) = if T::DTYPE == Dtype::I64 {
          (i128::from(i64::MIN), i128::from(i64::MAX))
        } else {
          (i128::from(i32::MIN), i128::from(i32::MAX))
        };
        if !(plo..=phi).contains(&delta) || !(plo..=phi).contains(&post_last) {
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "Array::arange: range",
            "overflows the signed integer accumulation",
            format_smolstr!("start={start}, step={step}, len={real_size}"),
          )));
        }
      }
    }
    // CRITICAL: `ensure_handler_installed` must precede the raw `mlx_array_new()`
    // below — see `Array::ones`. The pure-Rust guards above touch no mlx state.
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
        start,
        stop,
        step,
        mlxrs_sys::mlx_dtype::from(T::DTYPE),
        default_stream(),
      )
    })?;
    Ok(out)
  }

  /// Creates a 1-D array of `num` evenly-spaced values in `[start, stop]`.
  ///
  /// The output dtype is `T`; `start`/`stop` are taken as `f64` (any
  /// `Into<f64>`). `num == 1` yields `[start]` (mlx's special case), `num == 0`
  /// an empty array.
  ///
  /// # Soundness
  /// mlx `astype`s each ramp sample into `T` — a `static_cast<T>` (vendored
  /// `ops.cpp` / `backend/cpu/copy.cpp`) that is C++ UB for a value outside `T`'s
  /// range — and first builds the ramp in an inner dtype that narrows the `f64`
  /// bounds. `num == 0` is returned empty without touching the FFI (mlx would
  /// still narrow the endpoints in the formula). For `num == 1`
  /// (`astype(array({start}), dtype)`, where `array({start})` is **float32** via
  /// `TypeToDtype<double>`) and every non-f64 `num >= 2` ramp, each endpoint is
  /// narrowed `f64 -> f32 -> T`; this rejects (with [`Error::OutOfRange`]) an
  /// endpoint out of range at either cast. For the narrowing `f32 -> T` astypes
  /// (f16/bf16/integer), an interior `num >= 2` sample can round a few ULP past
  /// the endpoints, so the whole ramp is bounded with a conservative margin
  /// rather than just the endpoints.
  pub fn linspace<T>(start: impl Into<f64>, stop: impl Into<f64>, num: usize) -> Result<Self>
  where
    T: Element,
  {
    let start: f64 = start.into();
    let stop: f64 = stop.into();
    // `num == 0` is empty, but mlx still constructs `array(start, f32)` /
    // `array(stop, f32)` in the ramp formula (a `double -> float` narrowing, UB
    // outside f32 range) before producing the empty result — so return the empty
    // array directly and never reach that cast.
    if num == 0 {
      return Self::from_slice::<T>(&[], &[0i32]);
    }
    let n_i32 = i32::try_from(num).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "Array::linspace: num",
        "must fit in i32",
        format_smolstr!("{num}"),
      ))
    })?;
    // Range-check the source value mlx actually `static_cast`s into `T`. The ramp
    // is built in an inner dtype then `astype`d to `T`. The inner dtype is f32 for
    // every non-f64 output AND for `num == 1` regardless of output — `num == 1` is
    // `astype(array({start}), dtype)` and `array({start})` is float32
    // (`TypeToDtype<double>` is float32, vendored `dtype.cpp`); only `num >= 2`
    // with an f64 output keeps the f64 inner.
    if num == 1 {
      // Single value: `start` narrowed f64 -> f32 -> `T`.
      if !representable_in(start, Dtype::F32)
        || !representable_in(f64::from(start as f32), T::DTYPE)
      {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "Array::linspace: start",
          "must be representable in the output dtype",
          format_smolstr!("{start}"),
        )));
      }
    } else if T::DTYPE != Dtype::F64 {
      // num >= 2, non-f64: the endpoints are narrowed f64 -> f32 exactly
      // (`start` at t=0, `stop` at t=1).
      if !representable_in(start, Dtype::F32) || !representable_in(stop, Dtype::F32) {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "Array::linspace: endpoint",
          "must be representable in f32 (the ramp inner dtype)",
          format_smolstr!("start={start}, stop={stop}"),
        )));
      }
      // The f32 -> `T` astype narrows for f16/bf16/integer outputs (NOT for
      // f32/bool/complex64). Under the `(1 - t) * a + t * b` f32 ramp, an interior
      // sample can round up to ~`mag * 2^-22` past the endpoints (for `t >= 0.5`,
      // `1 - t` is exact by Sterbenz, so the coefficients sum to 1; for `t < 0.5`
      // they deviate by <= `2^-24`, plus the product/sum roundings). Bound the
      // FULL ramp — not just the endpoints — with a conservative `mag * 2^-18`
      // margin (16x that worst case) before the astype range check.
      if !matches!(
        T::DTYPE,
        Dtype::F32 | Dtype::F64 | Dtype::Bool | Dtype::Complex64
      ) {
        let a = f64::from(start as f32);
        let b = f64::from(stop as f32);
        let margin = a.abs().max(b.abs()) * f64::from(f32::EPSILON) * 32.0;
        if !representable_in(a.max(b) + margin, T::DTYPE)
          || !representable_in(a.min(b) - margin, T::DTYPE)
        {
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "Array::linspace: range",
            "the f32 ramp leaves the integer/half output dtype range",
            format_smolstr!("start={start}, stop={stop}"),
          )));
        }
      }
    }
    // CRITICAL: `ensure_handler_installed` must precede the raw `mlx_array_new()`
    // below — see `Array::ones`. The pure-Rust guards above touch no mlx state.
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
      mlxrs_sys::mlx_linspace(
        &mut out.0,
        start,
        stop,
        n_i32,
        mlxrs_sys::mlx_dtype::from(T::DTYPE),
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

/// Dtypes mlx `arange` accepts — every dtype except `Bool`, which mlx rejects
/// outright. Used only for the [`Error::UnsupportedDtype`] payload's
/// "supported" list.
const ARANGE_SUPPORTED_DTYPES: &[Dtype] = &[
  Dtype::U8,
  Dtype::U16,
  Dtype::U32,
  Dtype::U64,
  Dtype::I8,
  Dtype::I16,
  Dtype::I32,
  Dtype::I64,
  Dtype::F16,
  Dtype::BF16,
  Dtype::F32,
  Dtype::F64,
  Dtype::Complex64,
];

/// Half-open `f64` bounds `[lo, hi)` of the values whose `static_cast` into
/// `dtype`'s underlying C++ integer type is in range (and therefore not UB);
/// `None` for `bool` / float / complex dtypes, which take no such cast in
/// `arange`/`linspace`.
///
/// The upper bound is exclusive at `MAX + 1` so a fractional value like `255.9`
/// (which truncates to an in-range `255`) still passes for `u8`. For `i64`/`u64`
/// the exact `MAX` is not representable in `f64` and rounds up to `2^63`/`2^64`;
/// using that rounded power of two as the *exclusive* upper bound stays sound —
/// it rejects exactly the out-of-range cast while still admitting every value
/// `f64` can actually represent below it.
fn integer_cast_bounds(dtype: Dtype) -> Option<(f64, f64)> {
  Some(match dtype {
    Dtype::U8 => (0.0, f64::from(u8::MAX) + 1.0),
    Dtype::U16 => (0.0, f64::from(u16::MAX) + 1.0),
    Dtype::U32 => (0.0, f64::from(u32::MAX) + 1.0),
    Dtype::U64 => (0.0, u64::MAX as f64 + 1.0),
    Dtype::I8 => (f64::from(i8::MIN), f64::from(i8::MAX) + 1.0),
    Dtype::I16 => (f64::from(i16::MIN), f64::from(i16::MAX) + 1.0),
    Dtype::I32 => (f64::from(i32::MIN), f64::from(i32::MAX) + 1.0),
    Dtype::I64 => (i64::MIN as f64, i64::MAX as f64 + 1.0),
    _ => return None,
  })
}

/// Whether `static_cast`ing the `f64` `v` into `dtype`'s C++ type is in range
/// (and therefore not UB). This models the cast mlx performs when narrowing a
/// `double` arange/linspace bound into the output type:
/// - `f64` is the bound width, so no narrowing — always representable;
/// - `Bool` takes any value (`0` vs non-`0`), so its astype never UBs;
/// - float dtypes UB outside their finite range (a `double` above `FLT_MAX`
///   narrowed to `float` is out of range);
/// - integer dtypes truncate toward zero first, so they are in range iff the
///   truncated value fits `[MIN, MAX]`.
fn representable_in(v: f64, dtype: Dtype) -> bool {
  match dtype {
    Dtype::F64 | Dtype::Bool => true,
    Dtype::F32 | Dtype::Complex64 => v.abs() <= f64::from(f32::MAX),
    Dtype::F16 => v.abs() <= f64::from(half::f16::MAX),
    Dtype::BF16 => v.abs() <= f64::from(half::bf16::MAX),
    _ => integer_cast_bounds(dtype).is_some_and(|(lo, hi)| (lo..hi).contains(&v.trunc())),
  }
}
