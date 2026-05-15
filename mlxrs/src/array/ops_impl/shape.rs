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

  /// Full reverse-order transpose. See [`crate::ops::shape::transpose`].
  pub fn transpose(&self) -> Result<Array> {
    crate::ops::shape::transpose(self)
  }

  /// Custom-permutation transpose. See [`crate::ops::shape::transpose_axes`].
  pub fn transpose_axes(&self, axes: &[i32]) -> Result<Array> {
    crate::ops::shape::transpose_axes(self, axes)
  }

  /// Insert size-1 dims at each `axes`. See [`crate::ops::shape::expand_dims_axes`].
  pub fn expand_dims_axes(&self, axes: &[i32]) -> Result<Array> {
    crate::ops::shape::expand_dims_axes(self, axes)
  }

  /// Drop size-1 dims at each `axes`. See [`crate::ops::shape::squeeze_axes`].
  pub fn squeeze_axes(&self, axes: &[i32]) -> Result<Array> {
    crate::ops::shape::squeeze_axes(self, axes)
  }

  /// Broadcast to `shape`. See [`crate::ops::shape::broadcast_to`].
  pub fn broadcast_to(&self, shape: &impl IntoShape) -> Result<Array> {
    crate::ops::shape::broadcast_to(self, shape)
  }

  /// Stack with other arrays along a new `axis`. See [`crate::ops::shape::stack_axis`].
  ///
  /// Uses an inline `SmallVec` for the prepended array list so the common
  /// 2-to-4-input cases avoid a heap allocation on this hot path; spills to
  /// the heap only past 4 total inputs. Mirrors `concatenate_with`.
  pub fn stack_with(&self, others: &[&Array], axis: i32) -> Result<Array> {
    let mut all: SmallVec<[&Array; 4]> = SmallVec::with_capacity(others.len() + 1);
    all.push(self);
    all.extend_from_slice(others);
    crate::ops::shape::stack_axis(&all, axis)
  }

  /// Split along `axis` at each of the given `indices`.
  /// See [`crate::ops::shape::split_sections`].
  pub fn split_sections(&self, indices: &[i32], axis: i32) -> Result<Vec<Array>> {
    crate::ops::shape::split_sections(self, indices, axis)
  }

  /// Flatten dims `[start_axis, end_axis]`. See [`crate::ops::shape::flatten`].
  pub fn flatten(&self, start_axis: i32, end_axis: i32) -> Result<Array> {
    crate::ops::shape::flatten(self, start_axis, end_axis)
  }

  /// Swap two axes. See [`crate::ops::shape::swapaxes`].
  pub fn swapaxes(&self, axis1: i32, axis2: i32) -> Result<Array> {
    crate::ops::shape::swapaxes(self, axis1, axis2)
  }

  /// Pad along the given `axes` with `low` / `high` widths and `pad_value`.
  /// See [`crate::ops::shape::pad`].
  pub fn pad(
    &self,
    axes: &[i32],
    low: &[i32],
    high: &[i32],
    pad_value: &Array,
    mode: &std::ffi::CStr,
  ) -> Result<Array> {
    crate::ops::shape::pad(self, axes, low, high, pad_value, mode)
  }
}
