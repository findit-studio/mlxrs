//! Method-form indexing bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// Slice with NumPy-style start/stop/strides. See [`crate::ops::indexing::slice`].
  pub fn slice(&self, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
    crate::ops::indexing::slice(self, start, stop, strides)
  }
}
