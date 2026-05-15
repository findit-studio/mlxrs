//! `IntoShape` trait — zero-allocation shape conversion via slice callback.
//!
//! Returns `Result` so we propagate `usize > i32::MAX` as `Error::ShapeMismatch`
//! instead of silently saturating.
//!
//! `IntoShape` is **sealed**: downstream crates cannot implement it. The
//! single-source-of-truth validation lives at the FFI boundary (callers use
//! [`validate_dims`] inside their `with_shape` callback before passing the
//! slice to mlx-c). Sealing prevents a downstream impl from supplying a
//! callback slice that bypasses the call-site check.

use std::ffi::c_int;

use crate::error::{Error, Result};

mod sealed {
  pub trait Sealed {}
  impl Sealed for &[i32] {}
  impl<const N: usize> Sealed for [i32; N] {}
  impl Sealed for &[usize] {}
  impl Sealed for (usize,) {}
  impl Sealed for (usize, usize) {}
  impl Sealed for (usize, usize, usize) {}
  impl Sealed for (usize, usize, usize, usize) {}
}

/// Types that can supply an mlx-c-shaped `&[c_int]` to a callback without
/// heap allocation (for ranks ≤ 8). Sealed; not implementable downstream.
pub trait IntoShape: sealed::Sealed {
  /// Invoke `f` with the shape as `&[c_int]`. The caller MUST run
  /// [`validate_dims`] on the slice before any unsafe FFI use, even though
  /// the built-in impls validate eagerly — sealing is defense in depth, not
  /// a substitute for the boundary check.
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R>;
}

/// FFI-boundary safety check. Reject negative dimensions before any unsafe
/// mlx-c call. A negative `i32` silently sign-extends when cast to `usize`
/// (`-1i32 as usize == usize::MAX`), so an unchecked dim crossing into
/// mlx-c can produce malformed allocation/copy behavior or invalid array
/// metadata. Every safe-layer entry point that consumes a shape slice MUST
/// call this on the slice produced by `IntoShape::with_shape` before
/// passing to FFI.
pub fn validate_dims(s: &[c_int]) -> Result<()> {
  for (i, &d) in s.iter().enumerate() {
    if d < 0 {
      return Err(Error::ShapeMismatch {
        message: format!("dim[{i}] = {d} is negative; shapes must be non-negative"),
      });
    }
  }
  Ok(())
}

/// Static sentinel for FFI calls that would otherwise pass a Rust dangling
/// pointer (the well-defined `NonNull::dangling()` returned by `<&[i32]>::as_ptr`
/// on an empty slice). The C++ side calls `std::vector<int>(p, p + n)`; while
/// `p + 0` is well-defined for any pointer in C++17+, constructing a vector
/// from a "singular" iterator (one not associated with any allocation) is
/// strictly UB per `[iterator.requirements.general]`. Routing empty-len cases
/// through a real static i32 keeps the iterator non-singular without changing
/// observed behavior — the value is never read because `n == 0`.
static EMPTY_DIM_SENTINEL: c_int = 0;

/// Returns `s.as_ptr()` for non-empty slices; otherwise a non-singular pointer
/// into [`EMPTY_DIM_SENTINEL`]. Use at every FFI call site that passes a
/// `(*const c_int, len)` pair to mlx-c.
#[inline]
pub(crate) fn dim_ptr(s: &[c_int]) -> *const c_int {
  if s.is_empty() {
    &EMPTY_DIM_SENTINEL as *const c_int
  } else {
    s.as_ptr()
  }
}

impl IntoShape for &[i32] {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    f(self)
  }
}

impl<const N: usize> IntoShape for [i32; N] {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    f(&self[..])
  }
}

fn convert_dim(d: usize) -> Result<c_int> {
  i32::try_from(d).map_err(|_| Error::ShapeMismatch {
    message: format!("dim {d} exceeds i32::MAX ({})", i32::MAX),
  })
}

impl IntoShape for &[usize] {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    if self.len() <= 8 {
      let mut buf = [0i32; 8];
      for (i, &d) in self.iter().enumerate() {
        buf[i] = convert_dim(d)?;
      }
      f(&buf[..self.len()])
    } else {
      let v: Vec<c_int> = self
        .iter()
        .map(|&d| convert_dim(d))
        .collect::<Result<_>>()?;
      f(&v)
    }
  }
}

impl IntoShape for (usize,) {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    let s = [convert_dim(self.0)?];
    f(&s)
  }
}

impl IntoShape for (usize, usize) {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    let s = [convert_dim(self.0)?, convert_dim(self.1)?];
    f(&s)
  }
}

impl IntoShape for (usize, usize, usize) {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    let s = [
      convert_dim(self.0)?,
      convert_dim(self.1)?,
      convert_dim(self.2)?,
    ];
    f(&s)
  }
}

impl IntoShape for (usize, usize, usize, usize) {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    let s = [
      convert_dim(self.0)?,
      convert_dim(self.1)?,
      convert_dim(self.2)?,
      convert_dim(self.3)?,
    ];
    f(&s)
  }
}
