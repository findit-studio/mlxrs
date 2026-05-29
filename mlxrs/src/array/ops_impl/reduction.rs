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

  /// Variance along the given axes. See [`crate::ops::reduction::var_axes`].
  pub fn var_axes(&self, axes: &[i32], keepdims: bool, ddof: i32) -> Result<Array> {
    crate::ops::reduction::var_axes(self, axes, keepdims, ddof)
  }

  /// Variance of all elements. See [`crate::ops::reduction::var`].
  pub fn var(&self, keepdims: bool, ddof: i32) -> Result<Array> {
    crate::ops::reduction::var(self, keepdims, ddof)
  }

  /// Standard deviation along the given axes. See [`crate::ops::reduction::std_axes`].
  pub fn std_axes(&self, axes: &[i32], keepdims: bool, ddof: i32) -> Result<Array> {
    crate::ops::reduction::std_axes(self, axes, keepdims, ddof)
  }

  /// Standard deviation of all elements. See [`crate::ops::reduction::std`].
  pub fn std(&self, keepdims: bool, ddof: i32) -> Result<Array> {
    crate::ops::reduction::std(self, keepdims, ddof)
  }

  /// Logical AND along the given axes. See [`crate::ops::reduction::all_axes`].
  pub fn all_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::reduction::all_axes(self, axes, keepdims)
  }

  /// Logical AND of all elements. See [`crate::ops::reduction::all`].
  pub fn all(&self, keepdims: bool) -> Result<Array> {
    crate::ops::reduction::all(self, keepdims)
  }

  /// Logical OR along the given axes. See [`crate::ops::reduction::any_axes`].
  pub fn any_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::reduction::any_axes(self, axes, keepdims)
  }

  /// Logical OR of all elements. See [`crate::ops::reduction::any`].
  pub fn any(&self, keepdims: bool) -> Result<Array> {
    crate::ops::reduction::any(self, keepdims)
  }

  /// `log(sum(exp(a)))` along the given axes. See [`crate::ops::reduction::logsumexp_axes`].
  pub fn logsumexp_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::reduction::logsumexp_axes(self, axes, keepdims)
  }

  /// `log(sum(exp(a)))` of all elements. See [`crate::ops::reduction::logsumexp`].
  pub fn logsumexp(&self, keepdims: bool) -> Result<Array> {
    crate::ops::reduction::logsumexp(self, keepdims)
  }

  /// Median along the given axes. See [`crate::ops::reduction::median_axes`].
  pub fn median_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::reduction::median_axes(self, axes, keepdims)
  }

  /// Median of all elements. See [`crate::ops::reduction::median`].
  pub fn median(&self, keepdims: bool) -> Result<Array> {
    crate::ops::reduction::median(self, keepdims)
  }
}
