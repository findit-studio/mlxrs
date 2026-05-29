//! Method-form comparison bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// Element-wise equality. See [`crate::ops::comparison::equal`].
  pub fn equal(&self, rhs: &Array) -> Result<Array> {
    crate::ops::comparison::equal(self, rhs)
  }

  /// Element-wise inequality. See [`crate::ops::comparison::not_equal`].
  pub fn not_equal(&self, rhs: &Array) -> Result<Array> {
    crate::ops::comparison::not_equal(self, rhs)
  }

  /// Element-wise strict less-than. See [`crate::ops::comparison::less`].
  pub fn less(&self, rhs: &Array) -> Result<Array> {
    crate::ops::comparison::less(self, rhs)
  }

  /// Element-wise less-than-or-equal. See [`crate::ops::comparison::less_equal`].
  pub fn less_equal(&self, rhs: &Array) -> Result<Array> {
    crate::ops::comparison::less_equal(self, rhs)
  }

  /// Element-wise strict greater-than. See [`crate::ops::comparison::greater`].
  pub fn greater(&self, rhs: &Array) -> Result<Array> {
    crate::ops::comparison::greater(self, rhs)
  }

  /// Element-wise greater-than-or-equal. See [`crate::ops::comparison::greater_equal`].
  pub fn greater_equal(&self, rhs: &Array) -> Result<Array> {
    crate::ops::comparison::greater_equal(self, rhs)
  }

  /// Scalar-Bool "all-close" check. See [`crate::ops::comparison::allclose`].
  pub fn allclose(&self, rhs: &Array, rtol: f64, atol: f64, equal_nan: bool) -> Result<Array> {
    crate::ops::comparison::allclose(self, rhs, rtol, atol, equal_nan)
  }

  /// Element-wise close-to check. See [`crate::ops::comparison::isclose`].
  pub fn isclose(&self, rhs: &Array, rtol: f64, atol: f64, equal_nan: bool) -> Result<Array> {
    crate::ops::comparison::isclose(self, rhs, rtol, atol, equal_nan)
  }

  /// Element-wise finite check. See [`crate::ops::comparison::isfinite`].
  pub fn isfinite(&self) -> Result<Array> {
    crate::ops::comparison::isfinite(self)
  }

  /// Element-wise inf check. See [`crate::ops::comparison::isinf`].
  pub fn isinf(&self) -> Result<Array> {
    crate::ops::comparison::isinf(self)
  }

  /// Element-wise NaN check. See [`crate::ops::comparison::isnan`].
  pub fn isnan(&self) -> Result<Array> {
    crate::ops::comparison::isnan(self)
  }

  /// Element-wise negative-infinity check. See [`crate::ops::comparison::isneginf`].
  pub fn isneginf(&self) -> Result<Array> {
    crate::ops::comparison::isneginf(self)
  }

  /// Element-wise positive-infinity check. See [`crate::ops::comparison::isposinf`].
  pub fn isposinf(&self) -> Result<Array> {
    crate::ops::comparison::isposinf(self)
  }
}
