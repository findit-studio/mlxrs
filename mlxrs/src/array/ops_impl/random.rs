//! Method-form random bridges.
//!
//! Most random ops are best constructed via the free functions because they
//! take a `key: &Array` and a `shape` — `Array::ones`-style constructors fit
//! that shape better than method dispatch on the key. The method bridges here
//! cover the cases where the receiver is the natural input array (the data
//! being permuted, or the key being split).

use crate::{array::Array, error::Result};

impl Array {
  /// Split this PRNG key into two independent subkeys.
  /// See [`crate::ops::random::split`].
  pub fn split_key(&self) -> Result<(Array, Array)> {
    crate::ops::random::split(self)
  }

  /// Split this PRNG key into `num` independent subkeys (returned as a
  /// `[num, 2]` array). See [`crate::ops::random::split_num`].
  pub fn split_key_num(&self, num: i32) -> Result<Array> {
    crate::ops::random::split_num(self, num)
  }

  /// Random permutation of this array along `axis`.
  /// See [`crate::ops::random::permutation`].
  pub fn permutation(&self, axis: i32, key: &Array) -> Result<Array> {
    crate::ops::random::permutation(self, axis, key)
  }
}
