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

  /// Mean along the given axes. See [`crate::ops::reduction::mean_axes`].
  pub fn mean_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::reduction::mean_axes(self, axes, keepdims)
  }

  /// Mean of all elements. See [`crate::ops::reduction::mean`].
  pub fn mean(&self, keepdims: bool) -> Result<Array> {
    crate::ops::reduction::mean(self, keepdims)
  }

  /// Maximum along the given axes. See [`crate::ops::reduction::max_axes`].
  pub fn max_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::reduction::max_axes(self, axes, keepdims)
  }

  /// Maximum of all elements. See [`crate::ops::reduction::max`].
  pub fn max(&self, keepdims: bool) -> Result<Array> {
    crate::ops::reduction::max(self, keepdims)
  }

  /// Minimum along the given axes. See [`crate::ops::reduction::min_axes`].
  pub fn min_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::reduction::min_axes(self, axes, keepdims)
  }

  /// Minimum of all elements. See [`crate::ops::reduction::min`].
  pub fn min(&self, keepdims: bool) -> Result<Array> {
    crate::ops::reduction::min(self, keepdims)
  }

  /// Product along the given axes. See [`crate::ops::reduction::prod_axes`].
  pub fn prod_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::reduction::prod_axes(self, axes, keepdims)
  }

  /// Product of all elements. See [`crate::ops::reduction::prod`].
  pub fn prod(&self, keepdims: bool) -> Result<Array> {
    crate::ops::reduction::prod(self, keepdims)
  }
}
