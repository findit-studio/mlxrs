//! Method-form indexing bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// Slice with NumPy-style start/stop/strides. See [`crate::ops::indexing::slice`].
  pub fn slice(&self, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
    crate::ops::indexing::slice(self, start, stop, strides)
  }

  /// Take elements at flat positions in `indices`. See [`crate::ops::indexing::take`].
  pub fn take(&self, indices: &Array) -> Result<Array> {
    crate::ops::indexing::take(self, indices)
  }

  /// Take elements at `indices` along `axis`. See [`crate::ops::indexing::take_axis`].
  pub fn take_axis(&self, indices: &Array, axis: i32) -> Result<Array> {
    crate::ops::indexing::take_axis(self, indices, axis)
  }

  /// Take per-position elements along `axis`. See [`crate::ops::indexing::take_along_axis`].
  pub fn take_along_axis(&self, indices: &Array, axis: i32) -> Result<Array> {
    crate::ops::indexing::take_along_axis(self, indices, axis)
  }

  /// Scatter per-position `values` along `axis`. See [`crate::ops::indexing::put_along_axis`].
  pub fn put_along_axis(&self, indices: &Array, values: &Array, axis: i32) -> Result<Array> {
    crate::ops::indexing::put_along_axis(self, indices, values, axis)
  }

  /// Gather slices indexed by `indices` along `axes`. See [`crate::ops::indexing::gather`].
  pub fn gather(&self, indices: &[&Array], axes: &[i32], slice_sizes: &[i32]) -> Result<Array> {
    crate::ops::indexing::gather(self, indices, axes, slice_sizes)
  }

  /// Overwrite a strided sub-region with `update`. See [`crate::ops::indexing::slice_update`].
  pub fn slice_update(
    &self,
    update: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
  ) -> Result<Array> {
    crate::ops::indexing::slice_update(self, update, start, stop, strides)
  }

  /// Add `update` into a strided sub-region. See [`crate::ops::indexing::slice_update_add`].
  pub fn slice_update_add(
    &self,
    update: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
  ) -> Result<Array> {
    crate::ops::indexing::slice_update_add(self, update, start, stop, strides)
  }

  /// Element-wise max into a strided sub-region. See [`crate::ops::indexing::slice_update_max`].
  pub fn slice_update_max(
    &self,
    update: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
  ) -> Result<Array> {
    crate::ops::indexing::slice_update_max(self, update, start, stop, strides)
  }

  /// Element-wise min into a strided sub-region. See [`crate::ops::indexing::slice_update_min`].
  pub fn slice_update_min(
    &self,
    update: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
  ) -> Result<Array> {
    crate::ops::indexing::slice_update_min(self, update, start, stop, strides)
  }

  /// Multiply a strided sub-region by `update`. See [`crate::ops::indexing::slice_update_prod`].
  pub fn slice_update_prod(
    &self,
    update: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
  ) -> Result<Array> {
    crate::ops::indexing::slice_update_prod(self, update, start, stop, strides)
  }

  /// Overwrite a dynamically-offset sub-region. See [`crate::ops::indexing::slice_update_dynamic`].
  pub fn slice_update_dynamic(&self, update: &Array, start: &Array, axes: &[i32]) -> Result<Array> {
    crate::ops::indexing::slice_update_dynamic(self, update, start, axes)
  }
}
