//! Operator overloads for `&Array <op> &Array` and `-&Array`. Gated behind
//! the `unstable-ops-overload` feature; OFF by default.
//!
//! # Experimental — does NOT follow SemVer; may be removed in any minor release.
//!
//! Cargo features unionize across the dependency graph, so a library author who
//! enables this transitively forces it on every downstream consumer. This crate
//! deliberately keeps the safe, fallible `a.add(&b)?` API in core; the operator
//! forms here panic on shape mismatch / dtype error and exist only as a
//! prototyping convenience. Library authors must NEVER enable this feature
//! transitively. End-user binaries may opt in.

use crate::array::Array;

/// `&a + &b` — panics on shape mismatch or dtype error. Use `a.add(&b)?` for `Result`.
#[cfg_attr(docsrs, doc(cfg(feature = "unstable-ops-overload")))]
impl<'b> std::ops::Add<&'b Array> for &Array {
  type Output = Array;
  fn add(self, rhs: &'b Array) -> Array {
    crate::ops::arithmetic::add(self, rhs).expect("Array + Array: shape/dtype error")
  }
}

/// `&a - &b` — panics on shape mismatch or dtype error. Use `a.subtract(&b)?` for `Result`.
#[cfg_attr(docsrs, doc(cfg(feature = "unstable-ops-overload")))]
impl<'b> std::ops::Sub<&'b Array> for &Array {
  type Output = Array;
  fn sub(self, rhs: &'b Array) -> Array {
    crate::ops::arithmetic::subtract(self, rhs).expect("Array - Array: shape/dtype error")
  }
}

/// `&a * &b` — panics on shape mismatch or dtype error. Use `a.multiply(&b)?` for `Result`.
#[cfg_attr(docsrs, doc(cfg(feature = "unstable-ops-overload")))]
impl<'b> std::ops::Mul<&'b Array> for &Array {
  type Output = Array;
  fn mul(self, rhs: &'b Array) -> Array {
    crate::ops::arithmetic::multiply(self, rhs).expect("Array * Array: shape/dtype error")
  }
}

/// `&a / &b` — panics on shape mismatch or dtype error. Use `a.divide(&b)?` for `Result`.
#[cfg_attr(docsrs, doc(cfg(feature = "unstable-ops-overload")))]
impl<'b> std::ops::Div<&'b Array> for &Array {
  type Output = Array;
  fn div(self, rhs: &'b Array) -> Array {
    crate::ops::arithmetic::divide(self, rhs).expect("Array / Array: shape/dtype error")
  }
}

/// `-&a` — element-wise negation. Panics on dtype error. Use `a.negative()?` for `Result`.
#[cfg_attr(docsrs, doc(cfg(feature = "unstable-ops-overload")))]
impl std::ops::Neg for &Array {
  type Output = Array;
  fn neg(self) -> Array {
    crate::ops::arithmetic::negative(self).expect("-Array: dtype error")
  }
}
