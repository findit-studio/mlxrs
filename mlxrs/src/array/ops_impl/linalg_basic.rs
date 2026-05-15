//! Method-form linalg bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// `alpha * (self @ rhs) + beta * c`. See [`crate::ops::linalg_basic::addmm`].
  pub fn addmm(&self, rhs: &Array, c: &Array, alpha: f32, beta: f32) -> Result<Array> {
    crate::ops::linalg_basic::addmm(c, self, rhs, alpha, beta)
  }
}
