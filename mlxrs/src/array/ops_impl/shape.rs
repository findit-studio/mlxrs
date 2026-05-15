//! Method-form shape bridges.

use smallvec::SmallVec;

use crate::{array::Array, error::Result, shape::IntoShape};

impl Array {
  /// Reshape this array to the new `shape`. See [`crate::ops::shape::reshape`].
  pub fn reshape(&self, shape: &impl IntoShape) -> Result<Array> {
    crate::ops::shape::reshape(self, shape)
  }

  /// Concatenate with other arrays along `axis`. See [`crate::ops::shape::concatenate`].
  ///
  /// Uses an inline `SmallVec` for the prepended array list so the common
  /// 2-to-4-input cases avoid a heap allocation on this hot path; spills to
  /// the heap only past 4 total inputs.
  pub fn concatenate_with(&self, others: &[&Array], axis: i32) -> Result<Array> {
    let mut all: SmallVec<[&Array; 4]> = SmallVec::with_capacity(others.len() + 1);
    all.push(self);
    all.extend_from_slice(others);
    crate::ops::shape::concatenate(&all, axis)
  }
}
