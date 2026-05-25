//! Method-form logical bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// Element-wise logical AND. See [`crate::ops::logical::logical_and`].
  pub fn logical_and(&self, rhs: &Array) -> Result<Array> {
    crate::ops::logical::logical_and(self, rhs)
  }

  /// Element-wise logical OR. See [`crate::ops::logical::logical_or`].
  pub fn logical_or(&self, rhs: &Array) -> Result<Array> {
    crate::ops::logical::logical_or(self, rhs)
  }

  /// Element-wise logical NOT. See [`crate::ops::logical::logical_not`].
  pub fn logical_not(&self) -> Result<Array> {
    crate::ops::logical::logical_not(self)
  }

  /// Element-wise selection (`mlx.core.where`). The receiver is the *condition*
  /// array; `x` is selected where it is true, `y` otherwise.
  /// See [`crate::ops::logical::select`].
  pub fn select(&self, x: &Array, y: &Array) -> Result<Array> {
    crate::ops::logical::select(self, x, y)
  }
}
