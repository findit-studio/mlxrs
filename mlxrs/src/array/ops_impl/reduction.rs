//! Method-form reduction bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// Sum elements along the given axes. See [`crate::ops::reduction::sum_axes`].
  pub fn sum_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::reduction::sum_axes(self, axes, keepdims)
  }

  /// Sum all elements. See [`crate::ops::reduction::sum`].
  pub fn sum(&self, keepdims: bool) -> Result<Array> {
    crate::ops::reduction::sum(self, keepdims)
  }
}
