//! [`clip_grad_norm`] — global-norm gradient clipping.
//!
//! Mirrors Python `mlx.optimizers.clip_grad_norm`
//! (`mlx/python/mlx/optimizers/optimizers.py:951..=976`).
//!
//! ```text
//! norm_squared = Σ g.square().sum()  for each gradient g
//! total_norm = sqrt(norm_squared)
//! normalizer = min(max_norm / (total_norm + 1e-6), 1.0)
//! clipped_grads = { k: g·normalizer for (k, g) in grads }
//! ```
//!
//! Returns `(clipped_grads, total_norm)` — the original gradient norm is
//! returned alongside the clipped gradients for diagnostic logging.

use crate::{
  Array, Result,
  lm::load::Weights,
  ops::{arithmetic, reduction::sum},
};

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// Clip the global norm of all gradients in `grads` so it does not exceed
/// `max_norm`.
///
/// Returns the pre-clip global norm as a scalar [`Array`]. The
/// `grads` map is mutated in-place: each entry is replaced with the
/// rescaled gradient (Python returns a new dict; the Rust port mutates
/// in-place since the caller usually owns the gradient map outright).
pub fn clip_grad_norm(grads: &mut Weights, max_norm: f32) -> Result<Array> {
  // Total norm² = Σ (g²).sum()
  let mut norm_squared: Option<Array> = None;
  for grad in grads.values() {
    let sq = arithmetic::square(grad)?;
    let s = sum(&sq, false)?;
    norm_squared = Some(match norm_squared.take() {
      Some(acc) => arithmetic::add(&acc, &s)?,
      None => s,
    });
  }
  let total_norm = match norm_squared {
    Some(ns) => arithmetic::sqrt(&ns)?,
    None => scalar(0.0)?,
  };
  // normalizer = min(max_norm / (total_norm + 1e-6), 1.0)
  let max_s = scalar(max_norm)?;
  let eps = scalar(1e-6)?;
  let denom = arithmetic::add(&total_norm, &eps)?;
  let ratio = arithmetic::divide(&max_s, &denom)?;
  let one = scalar(1.0)?;
  let normalizer = arithmetic::minimum(&ratio, &one)?;
  // Rescale each gradient in-place.
  let keys: Vec<String> = grads.keys().cloned().collect();
  for key in keys {
    let g = grads.remove(&key).expect("key from .keys() must exist");
    let clipped = arithmetic::multiply(&g, &normalizer)?;
    grads.insert(key, clipped);
  }
  Ok(total_norm)
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::collections::HashMap;

  #[test]
  fn clip_grad_norm_no_clip_when_below_threshold() -> Result<()> {
    // grads = {"w1": [3, 4]} → norm² = 9+16 = 25 → norm = 5
    // max_norm = 10 → normalizer = min(10/(5+1e-6), 1.0) = 1.0
    let mut grads: Weights = HashMap::new();
    grads.insert("w1".into(), Array::from_slice::<f32>(&[3.0, 4.0], &[2])?);
    let mut norm = clip_grad_norm(&mut grads, 10.0)?;
    assert!((norm.item::<f32>()? - 5.0).abs() < 1e-4);
    // Gradient should be unchanged.
    let mut got = grads["w1"].try_clone()?;
    let v: Vec<f32> = got.to_vec()?;
    assert!((v[0] - 3.0).abs() < 1e-5 && (v[1] - 4.0).abs() < 1e-5);
    Ok(())
  }

  #[test]
  fn clip_grad_norm_rescales_when_above_threshold() -> Result<()> {
    // grads = {"w1": [3, 4]} → norm = 5; max_norm = 2.5 → normalizer ≈ 0.5
    let mut grads: Weights = HashMap::new();
    grads.insert("w1".into(), Array::from_slice::<f32>(&[3.0, 4.0], &[2])?);
    let mut norm = clip_grad_norm(&mut grads, 2.5)?;
    assert!((norm.item::<f32>()? - 5.0).abs() < 1e-4);
    let mut got = grads["w1"].try_clone()?;
    let v: Vec<f32> = got.to_vec()?;
    // Rescaled: 3·0.5 ≈ 1.5, 4·0.5 = 2.0
    assert!((v[0] - 1.5).abs() < 1e-3);
    assert!((v[1] - 2.0).abs() < 1e-3);
    Ok(())
  }

  #[test]
  fn clip_grad_norm_handles_multiple_entries() -> Result<()> {
    // grads = {"w1": [2, 3], "w2": [1]} → norm² = 4+9+1 = 14 → norm = sqrt(14) ≈ 3.7417
    let mut grads: Weights = HashMap::new();
    grads.insert("w1".into(), Array::from_slice::<f32>(&[2.0, 3.0], &[2])?);
    grads.insert("w2".into(), Array::from_slice::<f32>(&[1.0], &[1])?);
    let mut norm = clip_grad_norm(&mut grads, 100.0)?;
    let got = norm.item::<f32>()?;
    assert!((got - 14.0_f32.sqrt()).abs() < 1e-4, "got {got}");
    Ok(())
  }

  #[test]
  fn clip_grad_norm_empty_map_returns_zero_norm() -> Result<()> {
    let mut grads: Weights = HashMap::new();
    let mut norm = clip_grad_norm(&mut grads, 1.0)?;
    assert!((norm.item::<f32>()?).abs() < 1e-6);
    Ok(())
  }
}
