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

  /// Gather slices indexed by `indices` along `axes`. See [`crate::ops::indexing::gather`].
  pub fn gather(&self, indices: &[&Array], axes: &[i32], slice_sizes: &[i32]) -> Result<Array> {
    crate::ops::indexing::gather(self, indices, axes, slice_sizes)
  }
}
