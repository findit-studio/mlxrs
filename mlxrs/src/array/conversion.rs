//! Array introspection: shape, dtype, scalar/buffer extraction.

use std::ffi::CStr;

use crate::{
  array::Array,
  dtype::{Dtype, Element},
  error::{Error, Result},
};

impl Array {
  /// Number of dimensions.
  pub fn ndim(&self) -> usize {
    // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
    // it, and the call returns a plain scalar (no out-param, no rc).
    unsafe { mlxrs_sys::mlx_array_ndim(self.0) }
  }

  /// Total number of elements.
  pub fn size(&self) -> usize {
    // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
    // it, and the call returns a plain scalar (no out-param, no rc).
    unsafe { mlxrs_sys::mlx_array_size(self.0) }
  }

  /// Element type.
  pub fn dtype(&self) -> Result<Dtype> {
    // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
    // it, and the call returns a plain scalar (no out-param, no rc).
    Dtype::try_from(unsafe { mlxrs_sys::mlx_array_dtype(self.0) })
  }

  /// Shape as a `Vec<usize>`.
  pub fn shape(&self) -> Vec<usize> {
    let n = self.ndim();
    (0..n)
      // SAFETY: pure read of a valid borrowed handle for `0 <= i < ndim`; mlx-c does
      // not mutate or retain the handle and returns a plain scalar.
      .map(|i| unsafe { mlxrs_sys::mlx_array_dim(self.0, i as std::ffi::c_int) as usize })
      .collect()
  }

  /// Scalar extraction. Implicitly evaluates the array (mlx requires this for data access).
  pub fn item<T: Element>(&mut self) -> Result<T> {
    let actual = self.dtype()?;
    if actual != T::DTYPE {
      return Err(Error::DtypeMismatch {
        expected: T::DTYPE,
        got: actual,
      });
    }
    self.eval()?;
    // SAFETY: `self.0` was evaluated (`self.eval()` above) and its dtype verified
    // `== T::DTYPE` above, satisfying `Element::item`'s # Safety contract.
    unsafe { T::item(self.0) }
  }

  /// Materialize the underlying buffer as `Vec<T>`. Forces eval. Errors with
  /// `Error::NonContiguous` if the array is strided/broadcast: `mlx_array_size`
  /// (logical element count) can exceed the contiguous storage reachable from
  /// the data pointer for views, so reading `size` elements would read past
  /// the allocation. M2 will add `.contiguous()` to materialize strided views.
  pub fn to_vec<T: Element>(&mut self) -> Result<Vec<T>> {
    let actual = self.dtype()?;
    if actual != T::DTYPE {
      return Err(Error::DtypeMismatch {
        expected: T::DTYPE,
        got: actual,
      });
    }
    self.eval()?;
    if !is_row_contiguous(self.0) {
      return Err(Error::NonContiguous);
    }
    // SAFETY: array materialized by the prior `eval()`, dtype verified `== T::DTYPE`
    // and row-contiguity checked above; the NULL/zero-length case is guarded
    // before this call, so `(ptr, len)` is a valid non-null slice.
    unsafe {
      let (ptr, len) = T::data(self.0);
      // Zero-element arrays (shape `[0]`, `[2,0]`, ...) yield NULL from mlx;
      // `from_raw_parts(NULL, 0)` is UB per Rust's slice contract, so return
      // an empty Vec without touching the pointer.
      if len == 0 {
        return Ok(Vec::new());
      }
      assert!(!ptr.is_null(), "mlx data pointer NULL after eval");
      Ok(std::slice::from_raw_parts(ptr, len).to_vec())
    }
  }

  /// Borrow the underlying buffer as `&[T]`. Forces eval. Errors with
  /// `Error::NonContiguous` if the array is strided (post-transpose, etc.).
  pub fn as_slice<T: Element>(&mut self) -> Result<&[T]> {
    let actual = self.dtype()?;
    if actual != T::DTYPE {
      return Err(Error::DtypeMismatch {
        expected: T::DTYPE,
        got: actual,
      });
    }
    self.eval()?;
    if !is_row_contiguous(self.0) {
      return Err(Error::NonContiguous);
    }
    // SAFETY: array materialized by the prior `eval()`, dtype verified `== T::DTYPE`
    // and row-contiguity checked above; the NULL/zero-length case is guarded
    // before this call, so `(ptr, len)` is a valid non-null slice.
    unsafe {
      let (ptr, len) = T::data(self.0);
      // Same zero-element guard as `to_vec`: NULL data ptr is legitimate
      // when `len == 0`, and `from_raw_parts(NULL, 0)` is still UB.
      if len == 0 {
        return Ok(&[]);
      }
      assert!(!ptr.is_null(), "mlx data pointer NULL after eval");
      Ok(std::slice::from_raw_parts(ptr, len))
    }
  }
}

/// Compute row-major contiguity from shape + strides. mlx-c does not expose
/// `mlx_array_is_contiguous` directly, so we replicate the standard check:
/// for each dim from innermost to outermost, the stride must equal the running
/// product of trailing dims. Dims of size 1 are skipped (any stride is fine).
fn is_row_contiguous(arr: mlxrs_sys::mlx_array) -> bool {
  // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
  // it, and the call returns a plain scalar (no out-param, no rc).
  let ndim = unsafe { mlxrs_sys::mlx_array_ndim(arr) };
  if ndim == 0 {
    return true;
  }
  // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
  // it, and the call returns a plain scalar (no out-param, no rc).
  let shape_ptr = unsafe { mlxrs_sys::mlx_array_shape(arr) };
  // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
  // it, and the call returns a plain scalar (no out-param, no rc).
  let strides_ptr = unsafe { mlxrs_sys::mlx_array_strides(arr) };
  if shape_ptr.is_null() || strides_ptr.is_null() {
    return false;
  }
  // SAFETY: `arr` is a valid borrowed handle and `ndim > 0` was checked above; the
  // shape/strides pointers were NULL-checked, and mlx-c guarantees each
  // spans `ndim` elements, so the `(ptr, ndim)` slice is in bounds.
  let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim) };
  // SAFETY: `arr` is a valid borrowed handle and `ndim > 0` was checked above; the
  // shape/strides pointers were NULL-checked, and mlx-c guarantees each
  // spans `ndim` elements, so the `(ptr, ndim)` slice is in bounds.
  let strides = unsafe { std::slice::from_raw_parts(strides_ptr, ndim) };
  let mut expected: usize = 1;
  for i in (0..ndim).rev() {
    let dim = shape[i] as usize;
    if dim == 1 {
      continue;
    }
    if strides[i] != expected {
      return false;
    }
    expected = expected.saturating_mul(dim);
  }
  true
}

impl std::fmt::Debug for Array {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let shape = self.shape();
    let dtype = self.dtype().ok();
    write!(f, "Array(shape={shape:?}, dtype={dtype:?})")
  }
}

/// RAII guard for a temporary `mlx_string` handle (e.g. the Display buffer).
struct StringGuard(mlxrs_sys::mlx_string);
impl Drop for StringGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. Runs during `Drop` /
    // thread teardown: must not touch TLS, call `check()`, panic, or unwind
    // across `extern "C"`; the rc is discarded silently per the crate's
    // Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_string_free(self.0);
    }
  }
}

impl std::fmt::Display for Array {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    crate::error::ensure_handler_installed();
    // mlx_array_tostring → upstream `operator<<(ostream, array)` calls
    // `a.eval()` before printing, so Display re-enters eval. It must honor
    // the cleared-thread poison guard like Array::eval does, otherwise
    // formatting a lazy array on a recycled-cleared worker silently
    // degrades to `Array(<tostring failed>)` instead of failing fast.
    // (Debug only reads shape/dtype metadata — no eval — so it is not
    // guarded; panicking in Debug during a debugger session is hostile.)
    crate::stream::assert_streams_not_cleared();
    // SAFETY: `mlx_string_new()` returns a fresh empty out-param `mlx_string`
    // (NULL ctx) per the mlx-c convention; populated by the following call
    // and freed via the local guard / explicit `mlx_string_free`.
    let mut s = StringGuard(unsafe { mlxrs_sys::mlx_string_new() });
    // SAFETY: `self.0` is a valid borrowed handle; `s` is a fresh `mlx_string`
    // out-param freed via the local guard/explicit free; mlx-c writes the
    // formatted string into it and the rc is surfaced (checked below).
    let rc = unsafe { mlxrs_sys::mlx_array_tostring(&mut s.0, self.0) };
    if rc != 0 {
      return write!(f, "Array(<tostring failed: rc={rc}>)");
    }
    // SAFETY: `s` is a live `mlx_string` (freed only after this borrow); mlx-c
    // returns its internal NUL-terminated buffer, valid until the string is
    // freed. The returned pointer is NULL-checked before use.
    let cstr = unsafe { CStr::from_ptr(mlxrs_sys::mlx_string_data(s.0)) };
    write!(f, "{}", cstr.to_string_lossy())
  }
}
