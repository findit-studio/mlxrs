//! [`RMSprop`] — running-average-of-squared-gradients normalization
//! (Tieleman & Hinton, 2012).
//!
//! Mirrors Python `mlx.optimizers.RMSprop`
//! (`mlx/python/mlx/optimizers/optimizers.py:297..=350`).
//!
//! Update formula:
//!
//! ```text
//! v = α·v + (1-α)·g²
//! w_new = w - lr·g / (sqrt(v) + eps)
//! ```
//!
//! Per-parameter state: a single `v` Array (Python `state["v"]`).

use std::collections::HashMap;

use crate::{
  Array, Result,
  error::Error,
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like, zeros_like_map},
  },
  ops::arithmetic,
};

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// RMSprop optimizer.
pub struct RMSprop {
  /// Learning rate `λ`.
  pub learning_rate: LearningRate,
  /// Smoothing constant `α`. Default Python: `0.99`.
  pub alpha: f32,
  /// Numerical-stability epsilon. Default Python: `1e-8`.
  pub eps: f32,
  step_count: usize,
  current_lr: f32,
  state: HashMap<String, Array>,
}

impl RMSprop {
  /// Construct an [`RMSprop`] optimizer.
  pub fn new(learning_rate: impl Into<LearningRate>, alpha: f32, eps: f32) -> Result<Self> {
    if alpha < 0.0 {
      return Err(Error::Backend {
        message: format!("RMSprop: alpha must be >= 0, got {alpha}"),
      });
    }
    if eps < 0.0 {
      return Err(Error::Backend {
        message: format!("RMSprop: epsilon must be >= 0, got {eps}"),
      });
    }
    let lr = learning_rate.into();
    let current_lr = lr.current(0);
    Ok(Self {
      learning_rate: lr,
      alpha,
      eps,
      step_count: 0,
      current_lr,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor (`alpha=0.99`, `eps=1e-8`).
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, 0.99, 1e-8)
  }
}

impl Optimizer for RMSprop {
  fn init(&mut self, params: &Weights) -> Result<()> {
    self.state = zeros_like_map(params)?;
    Ok(())
  }

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    if self.state.is_empty() {
      self.init(gradients)?;
    }
    // Resolve scheduled LR at PRE-increment step, then increment
    // (matches Python `optimizers.py:102..=106`).
    self.current_lr = self.learning_rate.current(self.step_count);
    self.step_count += 1;
    let alpha_s = scalar(self.alpha)?;
    let one_minus_alpha = scalar(1.0 - self.alpha)?;
    let eps_s = scalar(self.eps)?;
    let lr_s = scalar(self.current_lr)?;
    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      let prev_v = match self.state.get(key) {
        Some(v) => v.try_clone()?,
        None => zeros_like(param)?,
      };
      // v = α·v + (1-α)·g²
      let g_sq = arithmetic::square(grad)?;
      let v_scaled = arithmetic::multiply(&alpha_s, &prev_v)?;
      let g_sq_scaled = arithmetic::multiply(&one_minus_alpha, &g_sq)?;
      let v_new = arithmetic::add(&v_scaled, &g_sq_scaled)?;
      // w_new = w - lr·g / (sqrt(v) + eps)
      let lr_g = arithmetic::multiply(&lr_s, grad)?;
      let sqrt_v = arithmetic::sqrt(&v_new)?;
      let denom = arithmetic::add(&sqrt_v, &eps_s)?;
      let step_term = arithmetic::divide(&lr_g, &denom)?;
      let new_w = arithmetic::subtract(param, &step_term)?;
      params.insert(key.clone(), new_w);
      self.state.insert(key.clone(), v_new);
    }
    Ok(())
  }

  fn step(&self) -> usize {
    self.step_count
  }

  fn learning_rate(&self) -> f32 {
    self.current_lr
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn read_scalar(a: &Array) -> Result<f32> {
    let mut clone = a.try_clone()?;
    clone.item::<f32>()
  }

  #[test]
  fn rmsprop_single_step_matches_python_ref() -> Result<()> {
    // Python first step: v=(1-α)·g², w_new = w - lr·g / (sqrt(v) + eps).
    // w=1.0, g=0.5, lr=0.001, α=0.99, eps=1e-8
    //   v = 0.01·0.25 = 0.0025
    //   sqrt(v) = 0.05
    //   step = 0.001·0.5 / (0.05 + 1e-8) ≈ 0.01
    //   w_new ≈ 0.99
    let mut rms = RMSprop::default_with_lr(0.001)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.5)?);
    rms.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.99).abs() < 1e-4, "got {got}");
    Ok(())
  }

  #[test]
  fn rmsprop_rejects_negative_alpha() {
    assert!(RMSprop::new(0.001, -0.1, 1e-8).is_err());
  }

  #[test]
  fn rmsprop_rejects_negative_eps() {
    assert!(RMSprop::new(0.001, 0.99, -1e-8).is_err());
  }
}
