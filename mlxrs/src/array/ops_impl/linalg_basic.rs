//! Method-form linalg bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// `alpha * (self @ rhs) + beta * c`. See [`crate::ops::linalg_basic::addmm`].
  ///
  /// Argument order matches the free function: `c` (the additive term) leads,
  /// then `rhs`, then the scalar coefficients. `self` is the left matmul
  /// operand. Both `c` and `rhs` are `&Array`, so positional intent matters —
  /// keeping `c` in the first slot as in the free fn avoids the swap footgun.
  pub fn addmm(&self, c: &Array, rhs: &Array, alpha: f32, beta: f32) -> Result<Array> {
    crate::ops::linalg_basic::addmm(c, self, rhs, alpha, beta)
  }
}
