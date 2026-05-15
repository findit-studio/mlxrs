//! Method-form arithmetic bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// Element-wise addition. See [`crate::ops::arithmetic::add`].
  pub fn add(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::add(self, rhs)
  }
}
