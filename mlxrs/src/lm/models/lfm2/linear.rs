//! A `nn.Linear` expressed as `matmul(x, weight.T)[+ bias]`.
//!
//! `mlxrs` has no `Linear` struct — the established pattern (see
//! `embeddings/colvision.rs`) is to forward through [`matmul`] with the
//! weight transposed. `nn.Linear(in, out)` stores its `weight` as
//! `(out, in)`, so `y = x @ weight.T (+ bias)` reproduces it exactly.

use crate::{
  array::Array,
  error::Result,
  ops::{linalg_basic::matmul, shape::swapaxes},
};

/// A dense linear projection: `weight` is `(out_features, in_features)`
/// (the `nn.Linear` layout) and `bias` is the optional `(out_features,)`
/// shift.
///
/// Holds the loaded tensors; [`forward`](Linear::forward) computes
/// `x @ weight.T (+ bias)`. No implicit eval — every op appends to the lazy
/// graph.
#[derive(Debug)]
pub struct Linear {
  weight: Array,
  bias: Option<Array>,
}

impl Linear {
  /// Construct from a `(out, in)` weight and optional `(out,)` bias.
  pub fn new(weight: Array, bias: Option<Array>) -> Self {
    Self { weight, bias }
  }

  /// `x @ weight.T (+ bias)`. `weight` is `(out, in)`, so the last two axes
  /// are swapped to `(in, out)` before the matmul; `x`'s trailing axis must
  /// be `in_features`. Returns a new lazy [`Array`].
  pub fn forward(&self, x: &Array) -> Result<Array> {
    // `weight.T` over the last two axes — `weight` is rank-2 `(out, in)`, so
    // `swapaxes(-1, -2)` yields `(in, out)` and `matmul` contracts `x`'s
    // trailing `in` against it, batching `x`'s leading dims.
    let wt = swapaxes(&self.weight, -1, -2)?;
    let y = matmul(x, &wt)?;
    match &self.bias {
      Some(b) => y.add(b),
      None => Ok(y),
    }
  }
}
