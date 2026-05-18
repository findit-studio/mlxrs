//! Method-form misc bridges.

use crate::{array::Array, dtype::Dtype, error::Result};

impl Array {
  /// Index of the maximum value. See [`crate::ops::misc::argmax`].
  pub fn argmax(&self, axis: Option<i32>, keepdims: bool) -> Result<Array> {
    crate::ops::misc::argmax(self, axis, keepdims)
  }

  /// Index of the minimum value. See [`crate::ops::misc::argmin`].
  pub fn argmin(&self, axis: Option<i32>, keepdims: bool) -> Result<Array> {
    crate::ops::misc::argmin(self, axis, keepdims)
  }

  /// Cumulative sum along `axis`. See [`crate::ops::misc::cumsum`].
  pub fn cumsum(&self, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
    crate::ops::misc::cumsum(self, axis, reverse, inclusive)
  }

  /// Cumulative product along `axis`. See [`crate::ops::misc::cumprod`].
  pub fn cumprod(&self, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
    crate::ops::misc::cumprod(self, axis, reverse, inclusive)
  }

  /// Cumulative maximum along `axis`. See [`crate::ops::misc::cummax`].
  pub fn cummax(&self, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
    crate::ops::misc::cummax(self, axis, reverse, inclusive)
  }

  /// Cumulative minimum along `axis`. See [`crate::ops::misc::cummin`].
  pub fn cummin(&self, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
    crate::ops::misc::cummin(self, axis, reverse, inclusive)
  }

  /// Sort the flattened array. See [`crate::ops::misc::sort`].
  pub fn sort(&self) -> Result<Array> {
    crate::ops::misc::sort(self)
  }

  /// Sort along `axis`. See [`crate::ops::misc::sort_axis`].
  pub fn sort_axis(&self, axis: i32) -> Result<Array> {
    crate::ops::misc::sort_axis(self, axis)
  }

  /// Indices that would sort the flattened array. See [`crate::ops::misc::argsort`].
  pub fn argsort(&self) -> Result<Array> {
    crate::ops::misc::argsort(self)
  }

  /// Indices that would sort along `axis`. See [`crate::ops::misc::argsort_axis`].
  pub fn argsort_axis(&self, axis: i32) -> Result<Array> {
    crate::ops::misc::argsort_axis(self, axis)
  }

  /// Top-`k` elements of the flattened array. See [`crate::ops::misc::topk`].
  pub fn topk(&self, k: i32) -> Result<Array> {
    crate::ops::misc::topk(self, k)
  }

  /// Top-`k` elements along `axis`. See [`crate::ops::misc::topk_axis`].
  pub fn topk_axis(&self, k: i32, axis: i32) -> Result<Array> {
    crate::ops::misc::topk_axis(self, k, axis)
  }

  /// Partition around index `kth`. See [`crate::ops::misc::partition`].
  pub fn partition(&self, kth: i32) -> Result<Array> {
    crate::ops::misc::partition(self, kth)
  }

  /// Partition along `axis` around index `kth`. See [`crate::ops::misc::partition_axis`].
  pub fn partition_axis(&self, kth: i32, axis: i32) -> Result<Array> {
    crate::ops::misc::partition_axis(self, kth, axis)
  }

  /// Indices that would partition around `kth`. See [`crate::ops::misc::argpartition`].
  pub fn argpartition(&self, kth: i32) -> Result<Array> {
    crate::ops::misc::argpartition(self, kth)
  }

  /// Indices that would partition along `axis` around `kth`. See [`crate::ops::misc::argpartition_axis`].
  pub fn argpartition_axis(&self, kth: i32, axis: i32) -> Result<Array> {
    crate::ops::misc::argpartition_axis(self, kth, axis)
  }

  /// Softmax along `axis` (`precise` = higher-precision accumulation). See [`crate::ops::misc::softmax_axis`].
  pub fn softmax_axis(&self, axis: i32, precise: bool) -> Result<Array> {
    crate::ops::misc::softmax_axis(self, axis, precise)
  }

  /// Clamp into `[a_min, a_max]` (array bounds). See [`crate::ops::misc::clip`].
  pub fn clip(&self, a_min: &Array, a_max: &Array) -> Result<Array> {
    crate::ops::misc::clip(self, a_min, a_max)
  }

  /// Clamp into `[min, max]` (scalar bounds). See [`crate::ops::misc::clip_with_scalar`].
  pub fn clip_with_scalar(&self, min: f32, max: f32) -> Result<Array> {
    crate::ops::misc::clip_with_scalar(self, min, max)
  }

  /// Array of ones with the same shape/dtype. See [`crate::ops::misc::ones_like`].
  pub fn ones_like(&self) -> Result<Array> {
    crate::ops::misc::ones_like(self)
  }

  /// Array of zeros with the same shape/dtype. See [`crate::ops::misc::zeros_like`].
  pub fn zeros_like(&self) -> Result<Array> {
    crate::ops::misc::zeros_like(self)
  }

  /// Array filled with `value`, same shape/dtype. See [`crate::ops::misc::full_like`].
  pub fn full_like(&self, value: f32) -> Result<Array> {
    crate::ops::misc::full_like(self, value)
  }

  /// Cast to `dtype`. See [`crate::ops::misc::astype`].
  pub fn astype(&self, dtype: Dtype) -> Result<Array> {
    crate::ops::misc::astype(self, dtype)
  }
}
