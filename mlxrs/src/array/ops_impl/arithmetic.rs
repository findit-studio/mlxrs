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

  // ───────── M2a unary long-tail ─────────

  /// Element-wise base-10 logarithm. See [`crate::ops::arithmetic::log10`].
  pub fn log10(&self) -> Result<Array> {
    crate::ops::arithmetic::log10(self)
  }

  /// Element-wise base-2 logarithm. See [`crate::ops::arithmetic::log2`].
  pub fn log2(&self) -> Result<Array> {
    crate::ops::arithmetic::log2(self)
  }

  /// Element-wise `log(1 + a)`. See [`crate::ops::arithmetic::log1p`].
  pub fn log1p(&self) -> Result<Array> {
    crate::ops::arithmetic::log1p(self)
  }

  /// Element-wise `exp(a) - 1`. See [`crate::ops::arithmetic::expm1`].
  pub fn expm1(&self) -> Result<Array> {
    crate::ops::arithmetic::expm1(self)
  }

  /// Element-wise error function. See [`crate::ops::arithmetic::erf`].
  pub fn erf(&self) -> Result<Array> {
    crate::ops::arithmetic::erf(self)
  }

  /// Element-wise inverse error function. See [`crate::ops::arithmetic::erfinv`].
  pub fn erfinv(&self) -> Result<Array> {
    crate::ops::arithmetic::erfinv(self)
  }

  /// Element-wise logistic sigmoid. See [`crate::ops::arithmetic::sigmoid`].
  pub fn sigmoid(&self) -> Result<Array> {
    crate::ops::arithmetic::sigmoid(self)
  }

  /// Element-wise ceiling. See [`crate::ops::arithmetic::ceil`].
  pub fn ceil(&self) -> Result<Array> {
    crate::ops::arithmetic::ceil(self)
  }

  /// Element-wise floor. See [`crate::ops::arithmetic::floor`].
  pub fn floor(&self) -> Result<Array> {
    crate::ops::arithmetic::floor(self)
  }

  /// Element-wise round to `decimals` decimal places. See [`crate::ops::arithmetic::round`].
  pub fn round(&self, decimals: i32) -> Result<Array> {
    crate::ops::arithmetic::round(self, decimals)
  }

  /// Element-wise sign. See [`crate::ops::arithmetic::sign`].
  pub fn sign(&self) -> Result<Array> {
    crate::ops::arithmetic::sign(self)
  }

  /// Element-wise reciprocal `1 / a`. See [`crate::ops::arithmetic::reciprocal`].
  pub fn reciprocal(&self) -> Result<Array> {
    crate::ops::arithmetic::reciprocal(self)
  }

  /// Element-wise reciprocal square root. See [`crate::ops::arithmetic::rsqrt`].
  pub fn rsqrt(&self) -> Result<Array> {
    crate::ops::arithmetic::rsqrt(self)
  }

  /// Element-wise complex conjugate. See [`crate::ops::arithmetic::conjugate`].
  pub fn conjugate(&self) -> Result<Array> {
    crate::ops::arithmetic::conjugate(self)
  }

  /// Real part of a complex array. See [`crate::ops::arithmetic::real`].
  pub fn real(&self) -> Result<Array> {
    crate::ops::arithmetic::real(self)
  }

  /// Imaginary part of a complex array. See [`crate::ops::arithmetic::imag`].
  pub fn imag(&self) -> Result<Array> {
    crate::ops::arithmetic::imag(self)
  }

  /// Element-wise radians-to-degrees. See [`crate::ops::arithmetic::degrees`].
  pub fn degrees(&self) -> Result<Array> {
    crate::ops::arithmetic::degrees(self)
  }

  /// Element-wise degrees-to-radians. See [`crate::ops::arithmetic::radians`].
  pub fn radians(&self) -> Result<Array> {
    crate::ops::arithmetic::radians(self)
  }

  /// Element-wise hyperbolic sine. See [`crate::ops::arithmetic::sinh`].
  pub fn sinh(&self) -> Result<Array> {
    crate::ops::arithmetic::sinh(self)
  }

  /// Element-wise hyperbolic cosine. See [`crate::ops::arithmetic::cosh`].
  pub fn cosh(&self) -> Result<Array> {
    crate::ops::arithmetic::cosh(self)
  }

  /// Element-wise arc-sine. See [`crate::ops::arithmetic::arcsin`].
  pub fn arcsin(&self) -> Result<Array> {
    crate::ops::arithmetic::arcsin(self)
  }

  /// Element-wise arc-cosine. See [`crate::ops::arithmetic::arccos`].
  pub fn arccos(&self) -> Result<Array> {
    crate::ops::arithmetic::arccos(self)
  }

  /// Element-wise arc-tangent. See [`crate::ops::arithmetic::arctan`].
  pub fn arctan(&self) -> Result<Array> {
    crate::ops::arithmetic::arctan(self)
  }

  /// Element-wise inverse hyperbolic sine. See [`crate::ops::arithmetic::arcsinh`].
  pub fn arcsinh(&self) -> Result<Array> {
    crate::ops::arithmetic::arcsinh(self)
  }

  /// Element-wise inverse hyperbolic cosine. See [`crate::ops::arithmetic::arccosh`].
  pub fn arccosh(&self) -> Result<Array> {
    crate::ops::arithmetic::arccosh(self)
  }

  /// Element-wise inverse hyperbolic tangent. See [`crate::ops::arithmetic::arctanh`].
  pub fn arctanh(&self) -> Result<Array> {
    crate::ops::arithmetic::arctanh(self)
  }

  /// Replace `NaN`/`±inf` element-wise. See [`crate::ops::arithmetic::nan_to_num`].
  pub fn nan_to_num(&self, nan: f32, posinf: Option<f32>, neginf: Option<f32>) -> Result<Array> {
    crate::ops::arithmetic::nan_to_num(self, nan, posinf, neginf)
  }

  /// Element-wise bitwise NOT. See [`crate::ops::arithmetic::bitwise_invert`].
  pub fn bitwise_invert(&self) -> Result<Array> {
    crate::ops::arithmetic::bitwise_invert(self)
  }

  // ───────── M2a binary long-tail ─────────

  /// Element-wise two-argument arc-tangent. See [`crate::ops::arithmetic::arctan2`].
  pub fn arctan2(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::arctan2(self, rhs)
  }

  /// Element-wise floor-division. See [`crate::ops::arithmetic::floor_divide`].
  pub fn floor_divide(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::floor_divide(self, rhs)
  }

  /// Element-wise remainder (Python `%`). See [`crate::ops::arithmetic::remainder`].
  pub fn remainder(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::remainder(self, rhs)
  }

  /// Element-wise simultaneous quotient + remainder. See [`crate::ops::arithmetic::divmod`].
  pub fn divmod(&self, rhs: &Array) -> Result<(Array, Array)> {
    crate::ops::arithmetic::divmod(self, rhs)
  }

  /// Element-wise bitwise AND. See [`crate::ops::arithmetic::bitwise_and`].
  pub fn bitwise_and(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::bitwise_and(self, rhs)
  }

  /// Element-wise bitwise OR. See [`crate::ops::arithmetic::bitwise_or`].
  pub fn bitwise_or(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::bitwise_or(self, rhs)
  }

  /// Element-wise bitwise XOR. See [`crate::ops::arithmetic::bitwise_xor`].
  pub fn bitwise_xor(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::bitwise_xor(self, rhs)
  }

  /// Element-wise left-shift. See [`crate::ops::arithmetic::left_shift`].
  pub fn left_shift(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::left_shift(self, rhs)
  }

  /// Element-wise right-shift. See [`crate::ops::arithmetic::right_shift`].
  pub fn right_shift(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::right_shift(self, rhs)
  }
}
