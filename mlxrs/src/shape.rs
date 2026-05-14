//! `IntoShape` trait — zero-allocation shape conversion via slice callback.
//!
//! Returns `Result` so we propagate `usize > i32::MAX` as `Error::ShapeMismatch`
//! instead of silently saturating.

use std::ffi::c_int;

use crate::error::{Error, Result};

/// Types that can supply an mlx-c-shaped `&[c_int]` to a callback without
/// heap allocation (for ranks ≤ 8).
pub trait IntoShape {
  /// Invoke `f` with the shape as `&[c_int]`.
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R>;
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
