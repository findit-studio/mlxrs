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
//!
//! All four arity combinations are provided for each binary op:
//! `&a op &b`, `a op &b`, `&a op b`, `a op b`. Likewise `Neg` is implemented
//! for both `&Array` and `Array`. Owning variants simply borrow internally and
//! drop the consumed operand(s) at the end of the expression.

#[cfg(feature = "unstable-ops-overload")]
#[cfg_attr(docsrs, doc(cfg(feature = "unstable-ops-overload")))]
const _: () = {
  use crate::array::Array;

  // ───────── Add ─────────

  /// `&a + &b` — panics on shape mismatch or dtype error. Use `a.add(&b)?` for `Result`.
  impl<'b> std::ops::Add<&'b Array> for &Array {
    type Output = Array;
    fn add(self, rhs: &'b Array) -> Array {
      crate::ops::arithmetic::add(self, rhs).expect("Array + Array: shape/dtype error")
    }
  }

  /// `a + &b` — consumes `self`, borrows `rhs`. Panics on shape mismatch or dtype error.
  impl<'b> std::ops::Add<&'b Array> for Array {
    type Output = Array;
    fn add(self, rhs: &'b Array) -> Array {
      crate::ops::arithmetic::add(&self, rhs).expect("Array + Array: shape/dtype error")
    }
  }

  /// `&a + b` — borrows `self`, consumes `rhs`. Panics on shape mismatch or dtype error.
  impl std::ops::Add<Array> for &Array {
    type Output = Array;
    fn add(self, rhs: Array) -> Array {
      crate::ops::arithmetic::add(self, &rhs).expect("Array + Array: shape/dtype error")
    }
  }

  /// `a + b` — consumes both operands. Panics on shape mismatch or dtype error.
  impl std::ops::Add<Array> for Array {
    type Output = Array;
    fn add(self, rhs: Array) -> Array {
      crate::ops::arithmetic::add(&self, &rhs).expect("Array + Array: shape/dtype error")
    }
  }

  // ───────── Sub ─────────

  /// `&a - &b` — panics on shape mismatch or dtype error. Use `a.subtract(&b)?` for `Result`.
  impl<'b> std::ops::Sub<&'b Array> for &Array {
    type Output = Array;
    fn sub(self, rhs: &'b Array) -> Array {
      crate::ops::arithmetic::subtract(self, rhs).expect("Array - Array: shape/dtype error")
    }
  }

  /// `a - &b` — consumes `self`, borrows `rhs`. Panics on shape mismatch or dtype error.
  impl<'b> std::ops::Sub<&'b Array> for Array {
    type Output = Array;
    fn sub(self, rhs: &'b Array) -> Array {
      crate::ops::arithmetic::subtract(&self, rhs).expect("Array - Array: shape/dtype error")
    }
  }

  /// `&a - b` — borrows `self`, consumes `rhs`. Panics on shape mismatch or dtype error.
  impl std::ops::Sub<Array> for &Array {
    type Output = Array;
    fn sub(self, rhs: Array) -> Array {
      crate::ops::arithmetic::subtract(self, &rhs).expect("Array - Array: shape/dtype error")
    }
  }

  /// `a - b` — consumes both operands. Panics on shape mismatch or dtype error.
  impl std::ops::Sub<Array> for Array {
    type Output = Array;
    fn sub(self, rhs: Array) -> Array {
      crate::ops::arithmetic::subtract(&self, &rhs).expect("Array - Array: shape/dtype error")
    }
  }

  // ───────── Mul ─────────

  /// `&a * &b` — panics on shape mismatch or dtype error. Use `a.multiply(&b)?` for `Result`.
  impl<'b> std::ops::Mul<&'b Array> for &Array {
    type Output = Array;
    fn mul(self, rhs: &'b Array) -> Array {
      crate::ops::arithmetic::multiply(self, rhs).expect("Array * Array: shape/dtype error")
    }
  }

  /// `a * &b` — consumes `self`, borrows `rhs`. Panics on shape mismatch or dtype error.
  impl<'b> std::ops::Mul<&'b Array> for Array {
    type Output = Array;
    fn mul(self, rhs: &'b Array) -> Array {
      crate::ops::arithmetic::multiply(&self, rhs).expect("Array * Array: shape/dtype error")
    }
  }

  /// `&a * b` — borrows `self`, consumes `rhs`. Panics on shape mismatch or dtype error.
  impl std::ops::Mul<Array> for &Array {
    type Output = Array;
    fn mul(self, rhs: Array) -> Array {
      crate::ops::arithmetic::multiply(self, &rhs).expect("Array * Array: shape/dtype error")
    }
  }

  /// `a * b` — consumes both operands. Panics on shape mismatch or dtype error.
  impl std::ops::Mul<Array> for Array {
    type Output = Array;
    fn mul(self, rhs: Array) -> Array {
      crate::ops::arithmetic::multiply(&self, &rhs).expect("Array * Array: shape/dtype error")
    }
  }

  // ───────── Div ─────────

  /// `&a / &b` — panics on shape mismatch or dtype error. Use `a.divide(&b)?` for `Result`.
  impl<'b> std::ops::Div<&'b Array> for &Array {
    type Output = Array;
    fn div(self, rhs: &'b Array) -> Array {
      crate::ops::arithmetic::divide(self, rhs).expect("Array / Array: shape/dtype error")
    }
  }

  /// `a / &b` — consumes `self`, borrows `rhs`. Panics on shape mismatch or dtype error.
  impl<'b> std::ops::Div<&'b Array> for Array {
    type Output = Array;
    fn div(self, rhs: &'b Array) -> Array {
      crate::ops::arithmetic::divide(&self, rhs).expect("Array / Array: shape/dtype error")
    }
  }

  /// `&a / b` — borrows `self`, consumes `rhs`. Panics on shape mismatch or dtype error.
  impl std::ops::Div<Array> for &Array {
    type Output = Array;
    fn div(self, rhs: Array) -> Array {
      crate::ops::arithmetic::divide(self, &rhs).expect("Array / Array: shape/dtype error")
    }
  }

  /// `a / b` — consumes both operands. Panics on shape mismatch or dtype error.
  impl std::ops::Div<Array> for Array {
    type Output = Array;
    fn div(self, rhs: Array) -> Array {
      crate::ops::arithmetic::divide(&self, &rhs).expect("Array / Array: shape/dtype error")
    }
  }

  // ───────── Neg ─────────

  /// `-&a` — element-wise negation. Panics on dtype error. Use `a.negative()?` for `Result`.
  impl std::ops::Neg for &Array {
    type Output = Array;
    fn neg(self) -> Array {
      crate::ops::arithmetic::negative(self).expect("-Array: dtype error")
    }
  }

  /// `-a` — element-wise negation, consumes `self`. Panics on dtype error.
  impl std::ops::Neg for Array {
    type Output = Array;
    fn neg(self) -> Array {
      crate::ops::arithmetic::negative(&self).expect("-Array: dtype error")
    }
  }
};
