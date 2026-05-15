//! Method-form arithmetic bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// Element-wise addition. See [`crate::ops::arithmetic::add`].
  pub fn add(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::add(self, rhs)
  }

  /// Element-wise subtraction. See [`crate::ops::arithmetic::subtract`].
  pub fn subtract(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::subtract(self, rhs)
  }

  /// Element-wise multiplication. See [`crate::ops::arithmetic::multiply`].
  pub fn multiply(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::multiply(self, rhs)
  }

  /// Element-wise division. See [`crate::ops::arithmetic::divide`].
  pub fn divide(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::divide(self, rhs)
  }

  /// Element-wise maximum. See [`crate::ops::arithmetic::maximum`].
  pub fn maximum(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::maximum(self, rhs)
  }

  /// Element-wise minimum. See [`crate::ops::arithmetic::minimum`].
  pub fn minimum(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::minimum(self, rhs)
  }

  /// Element-wise power. See [`crate::ops::arithmetic::power`].
  pub fn power(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::power(self, rhs)
  }

  /// Element-wise unary negation. See [`crate::ops::arithmetic::negative`].
  pub fn negative(&self) -> Result<Array> {
    crate::ops::arithmetic::negative(self)
  }

  /// Element-wise absolute value. See [`crate::ops::arithmetic::abs`].
  pub fn abs(&self) -> Result<Array> {
    crate::ops::arithmetic::abs(self)
  }

  /// Element-wise square root. See [`crate::ops::arithmetic::sqrt`].
  pub fn sqrt(&self) -> Result<Array> {
    crate::ops::arithmetic::sqrt(self)
  }

  /// Element-wise square. See [`crate::ops::arithmetic::square`].
  pub fn square(&self) -> Result<Array> {
    crate::ops::arithmetic::square(self)
  }

  /// Element-wise natural exponential. See [`crate::ops::arithmetic::exp`].
  pub fn exp(&self) -> Result<Array> {
    crate::ops::arithmetic::exp(self)
  }

  /// Element-wise natural logarithm. See [`crate::ops::arithmetic::log`].
  pub fn log(&self) -> Result<Array> {
    crate::ops::arithmetic::log(self)
  }

  /// Element-wise sine. See [`crate::ops::arithmetic::sin`].
  pub fn sin(&self) -> Result<Array> {
    crate::ops::arithmetic::sin(self)
  }

  /// Element-wise cosine. See [`crate::ops::arithmetic::cos`].
  pub fn cos(&self) -> Result<Array> {
    crate::ops::arithmetic::cos(self)
  }

  /// Element-wise tangent. See [`crate::ops::arithmetic::tan`].
  pub fn tan(&self) -> Result<Array> {
    crate::ops::arithmetic::tan(self)
  }

  /// Element-wise hyperbolic tangent. See [`crate::ops::arithmetic::tanh`].
  pub fn tanh(&self) -> Result<Array> {
    crate::ops::arithmetic::tanh(self)
  }
}
