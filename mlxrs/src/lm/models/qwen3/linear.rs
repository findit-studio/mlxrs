//! A `nn.Linear` expressed as `matmul(x, weight.T)[+ bias]`.
//!
//! `mlxrs` has no shared `Linear` struct — the established pattern (see
//! `lfm2/linear.rs`, `embeddings/colvision.rs`) is to forward through
//! [`matmul`] with the weight transposed. `nn.Linear(in, out)` stores its
//! `weight` as `(out, in)`, so `y = x @ weight.T (+ bias)` reproduces it
//! exactly. Every Qwen3 projection (`q/k/v/o_proj`, `gate/up/down_proj`, the
//! optional `lm_head`) is bias-free, but the optional `bias` is kept so this
//! mirrors the general `nn.Linear`.

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
    let wt = swapaxes(&self.weight, -1, -2)?;
    let y = matmul(x, &wt)?;
    match &self.bias {
      Some(b) => y.add(b),
      None => Ok(y),
    }
  }
}
