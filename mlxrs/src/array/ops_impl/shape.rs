//! Method-form shape bridges.

use crate::{array::Array, error::Result, shape::IntoShape};

impl Array {
  /// Reshape this array to the new `shape`. See [`crate::ops::shape::reshape`].
  pub fn reshape(&self, shape: &impl IntoShape) -> Result<Array> {
    crate::ops::shape::reshape(self, shape)
  }

  /// Concatenate with other arrays along `axis`. See [`crate::ops::shape::concatenate`].
  pub fn concatenate_with(&self, others: &[&Array], axis: i32) -> Result<Array> {
    let mut all: Vec<&Array> = Vec::with_capacity(others.len() + 1);
    all.push(self);
    all.extend_from_slice(others);
    crate::ops::shape::concatenate(&all, axis)
  }
}
