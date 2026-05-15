//! Array constructors — generic over `T: Element` where applicable.

use crate::{
  array::Array,
  dtype::{Dtype, Element},
  error::{Error, Result, check, check_handle},
  shape::IntoShape,
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
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
      check(unsafe {
        mlxrs_sys::mlx_ones(
          &mut out.0,
          s.as_ptr(),
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
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
      check(unsafe {
        mlxrs_sys::mlx_zeros(
          &mut out.0,
          s.as_ptr(),
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
      let scalar = ScalarGuard(unsafe { mlxrs_sys::mlx_array_new_float32(value) });
      let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
      check(unsafe {
        mlxrs_sys::mlx_full(
          &mut out.0,
          s.as_ptr(),
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
    shape.with_shape(|s| {
      // `IntoShape` already rejects negative dims for the i32 paths, so the
      // `as usize` cast below is well-defined. We still need `checked_mul`
      // because release builds wrap on overflow, and three or more large
      // dims can wrap to a small value that matches `data.len()` — handing
      // mlx-c a buffer smaller than the shape declares.
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
          data.as_ptr().cast::<std::ffi::c_void>(),
          s.as_ptr(),
          dim_i32,
          mlxrs_sys::mlx_dtype::from(T::DTYPE),
        )
      };
      check_handle(arr)
    })
  }
}
