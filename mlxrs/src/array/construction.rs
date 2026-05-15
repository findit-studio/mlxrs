//! Array constructors — generic over `T: Element` where applicable.

use crate::{
  array::Array,
  dtype::{Dtype, Element},
  error::{Error, Result, check, check_handle},
  shape::{IntoShape, dim_ptr, validate_dims},
  stream::default_stream,
};

/// RAII guard for a temporary `mlx_array` (e.g. the scalar val passed to mlx_full).
struct ScalarGuard(mlxrs_sys::mlx_array);
impl Drop for ScalarGuard {
  fn drop(&mut self) {
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
fn data_ptr<T: Element>(data: &[T]) -> *const T {
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
  pub fn ones<T: Element>(shape: &impl IntoShape) -> Result<Self> {
    shape.with_shape(|s| {
      validate_dims(s)?;
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
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
  pub fn zeros<T: Element>(shape: &impl IntoShape) -> Result<Self> {
    shape.with_shape(|s| {
      validate_dims(s)?;
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
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
  pub fn full<T: Element>(shape: &impl IntoShape, value: f32) -> Result<Self> {
    shape.with_shape(|s| {
      validate_dims(s)?;
      let scalar = ScalarGuard(unsafe { mlxrs_sys::mlx_array_new_float32(value) });
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
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
  pub fn eye<T: Element>(n: usize) -> Result<Self> {
    let n_i32 = i32::try_from(n).map_err(|_| Error::ShapeMismatch {
      message: format!("eye dim {n} exceeds i32::MAX"),
    })?;
    let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
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
    let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
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
    let n_i32 = i32::try_from(num).map_err(|_| Error::ShapeMismatch {
      message: format!("linspace num {num} exceeds i32::MAX"),
    })?;
    let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
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
  pub fn from_slice<T: Element>(data: &[T], shape: &impl IntoShape) -> Result<Self> {
    // The only Array constructor that doesn't go through default_stream;
    // install the error handler explicitly so the safety guarantee matches.
    crate::error::ensure_handler_installed();
    shape.with_shape(|s| {
      // FFI safety boundary: validate the slice we're about to hand to
      // mlx_array_new_data. validate_dims rules out negative dims (so the
      // `as usize` cast below is well-defined); checked_mul rules out
      // release-build wrapping on the shape product, which could otherwise
      // match data.len() and pass the equality guard with an undersized
      // buffer.
      validate_dims(s)?;
      let total: usize = s
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or_else(|| Error::ShapeMismatch {
          message: format!("from_slice: shape product overflows usize for shape {s:?}"),
        })?;
      if total != data.len() {
        return Err(Error::ShapeMismatch {
          message: format!(
            "from_slice: shape product {total} != data.len() {}",
            data.len()
          ),
        });
      }
      let dim_i32 = i32::try_from(s.len()).map_err(|_| Error::ShapeMismatch {
        message: format!("ndim {} exceeds i32::MAX", s.len()),
      })?;
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
