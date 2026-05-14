//! Method-form bridges: `a.add(&b)`, `a.reshape(...)`, etc.
//!
//! Phase 3 ships add + reshape only. Phase 3.5/4 add the rest.

use crate::{
  array::Array,
  error::{Result, check},
  shape::IntoShape,
  stream::default_stream,
};

impl Array {
  /// Element-wise addition. See [`crate::ops::arithmetic::add`].
  pub fn add(&self, rhs: &Array) -> Result<Array> {
    crate::ops::arithmetic::add(self, rhs)
  }

  /// Reshape this array to the new `shape`. Errors on incompatible shape
  /// (the C++ side validates total-element-count equality).
  pub fn reshape(&self, shape: &impl IntoShape) -> Result<Array> {
    shape.with_shape(|s| {
      let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
      check(unsafe {
        mlxrs_sys::mlx_reshape(&mut out.0, self.0, s.as_ptr(), s.len(), default_stream())
      })?;
      Ok(out)
    })
  }
}
