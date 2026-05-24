//! [`Lion`] — sign-of-momentum optimizer (Chen, 2023).
//!
//! Mirrors Python `mlx.optimizers.Lion`
//! (`mlx/python/mlx/optimizers/optimizers.py:650..=705`).
//!
//! Update formula:
//!
//! ```text
//! c = β₁·m + (1-β₁)·g
//! m = β₂·m + (1-β₂)·g
//! if weight_decay > 0: w = (1 - lr·weight_decay)·w
//! w_new = w - lr·sign(c)
//! ```
//!
//! Per-parameter state: a single `m` Array.

use std::collections::HashMap;

use crate::{
  Array, Result,
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like, zeros_like_map},
  },
  ops::arithmetic,
};

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// Lion optimizer.
pub struct Lion {
  /// Learning rate `η`.
  pub learning_rate: LearningRate,
  /// Running-average coefficients `(β₁, β₂)`. Default Python:
  /// `(0.9, 0.99)`.
  pub betas: (f32, f32),
  /// Weight decay `λ`. Default Python: `0.0`.
  pub weight_decay: f32,
  step_count: usize,
  current_lr: f32,
  state: HashMap<String, Array>,
}

impl Lion {
  /// Construct a [`Lion`] optimizer.
  pub fn new(
    learning_rate: impl Into<LearningRate>,
    betas: (f32, f32),
    weight_decay: f32,
  ) -> Result<Self> {
    let lr = learning_rate.into();
    let current_lr = lr.current(0);
    Ok(Self {
      learning_rate: lr,
      betas,
      weight_decay,
      step_count: 0,
      current_lr,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor (`betas=(0.9, 0.99)`, `wd=0.0`).
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, (0.9, 0.99), 0.0)
  }
}

impl Optimizer for Lion {
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
    let (b1, b2) = self.betas;
    let b1_s = scalar(b1)?;
    let b2_s = scalar(b2)?;
    let one_minus_b1 = scalar(1.0 - b1)?;
    let one_minus_b2 = scalar(1.0 - b2)?;
    let lr_s = scalar(self.current_lr)?;
    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      let prev_m = match self.state.get(key) {
        Some(m) => m.try_clone()?,
        None => zeros_like(param)?,
      };
      // c = β₁·m + (1-β₁)·g
      let m_b1 = arithmetic::multiply(&b1_s, &prev_m)?;
      let g_b1 = arithmetic::multiply(&one_minus_b1, grad)?;
      let c = arithmetic::add(&m_b1, &g_b1)?;
      // m = β₂·m + (1-β₂)·g
      let m_b2 = arithmetic::multiply(&b2_s, &prev_m)?;
      let g_b2 = arithmetic::multiply(&one_minus_b2, grad)?;
      let m_new = arithmetic::add(&m_b2, &g_b2)?;
      // weight decay: w = (1 - lr·wd)·w
      let param_after_decay = if self.weight_decay > 0.0 {
        let decay = scalar(1.0 - self.current_lr * self.weight_decay)?;
        arithmetic::multiply(param, &decay)?
      } else {
        param.try_clone()?
      };
      // w_new = w - lr·sign(c)
      let sign_c = arithmetic::sign(&c)?;
      let step_term = arithmetic::multiply(&lr_s, &sign_c)?;
      let new_w = arithmetic::subtract(&param_after_decay, &step_term)?;
      params.insert(key.clone(), new_w);
      self.state.insert(key.clone(), m_new);
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
  fn lion_single_step_matches_python_ref() -> Result<()> {
    // Python first step: m=0;
    //   c = (1-β₁)·g; m = (1-β₂)·g
    //   no weight decay branch (wd=0)
    //   w_new = w - lr·sign(c) = w - lr·sign(g)
    // w=1.0, g=0.5, lr=0.001 → w_new ≈ 0.999
    let mut lion = Lion::default_with_lr(0.001)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.5)?);
    lion.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.999).abs() < 1e-5, "got {got}");
    Ok(())
  }

  #[test]
  fn lion_with_weight_decay_applies_before_sign_step() -> Result<()> {
    // Python: w = (1 - lr·wd)·w then w_new = w - lr·sign(c).
    // w=1.0, lr=0.01, wd=0.1 → w_decay = 1.0·(1 - 0.001) = 0.999
    // c = 0.1·g = 0.05 → sign(c) = 1 → step = lr·1 = 0.01
    // w_new = 0.999 - 0.01 = 0.989
    let mut lion = Lion::new(0.01, (0.9, 0.99), 0.1)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.5)?);
    lion.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.989).abs() < 1e-5, "got {got}");
    Ok(())
  }
}
