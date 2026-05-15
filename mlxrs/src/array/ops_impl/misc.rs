//! Method-form misc bridges.

use crate::{array::Array, error::Result};

impl Array {
  /// Index of the maximum value. See [`crate::ops::misc::argmax`].
  pub fn argmax(&self, axis: Option<i32>, keepdims: bool) -> Result<Array> {
    crate::ops::misc::argmax(self, axis, keepdims)
  }
}
