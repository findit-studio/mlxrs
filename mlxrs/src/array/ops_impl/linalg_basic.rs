//! Method-form linalg bridges.

use crate::{array::Array, dtype::Dtype, error::Result};

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

  /// `self @ rhs`. See [`crate::ops::linalg_basic::matmul`].
  pub fn matmul(&self, rhs: &Array) -> Result<Array> {
    crate::ops::linalg_basic::matmul(self, rhs)
  }

  /// Inner product `self · rhs`. See [`crate::ops::linalg_basic::inner`].
  pub fn inner(&self, rhs: &Array) -> Result<Array> {
    crate::ops::linalg_basic::inner(self, rhs)
  }

  /// Outer product `self ⊗ rhs`. See [`crate::ops::linalg_basic::outer`].
  pub fn outer(&self, rhs: &Array) -> Result<Array> {
    crate::ops::linalg_basic::outer(self, rhs)
  }

  /// Tensor contraction (integer-axis form). See
  /// [`crate::ops::linalg_basic::tensordot`].
  pub fn tensordot(&self, rhs: &Array, axis: i32) -> Result<Array> {
    crate::ops::linalg_basic::tensordot(self, rhs, axis)
  }

  /// Tensor contraction (per-operand axis-list form). See
  /// [`crate::ops::linalg_basic::tensordot_axes`].
  pub fn tensordot_axes(&self, rhs: &Array, axes_self: &[i32], axes_rhs: &[i32]) -> Result<Array> {
    crate::ops::linalg_basic::tensordot_axes(self, rhs, axes_self, axes_rhs)
  }

  /// Extract diagonals. See [`crate::ops::linalg_basic::diagonal`].
  pub fn diagonal(&self, offset: i32, axis1: i32, axis2: i32) -> Result<Array> {
    crate::ops::linalg_basic::diagonal(self, offset, axis1, axis2)
  }

  /// Sum along the diagonals. See [`crate::ops::linalg_basic::trace`].
  pub fn trace(&self, offset: i32, axis1: i32, axis2: i32, dtype: Option<Dtype>) -> Result<Array> {
    crate::ops::linalg_basic::trace(self, offset, axis1, axis2, dtype)
  }

  /// Lower triangle. See [`crate::ops::linalg_basic::tril`].
  pub fn tril(&self, k: i32) -> Result<Array> {
    crate::ops::linalg_basic::tril(self, k)
  }

  /// Upper triangle. See [`crate::ops::linalg_basic::triu`].
  pub fn triu(&self, k: i32) -> Result<Array> {
    crate::ops::linalg_basic::triu(self, k)
  }
}
