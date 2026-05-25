//! [`AdaDelta`] — adaptive-learning-rate method (Zeiler, 2012).
//!
//! Mirrors Python `mlx.optimizers.AdaDelta`
//! (`mlx/python/mlx/optimizers/optimizers.py:403..=467`).
//!
//! Update formula:
//!
//! ```text
//! v = ρ·v + (1-ρ)·g²
//! Δw = sqrt(u + eps) / sqrt(v + eps) · g
//! u = ρ·u + (1-ρ)·Δw²
//! w_new = w - lr·Δw
//! ```
//!
//! Per-parameter state: `(v, u)` tuple.

use std::collections::HashMap;

use crate::{
  Array, Result,
  error::Error,
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like},
  },
  ops::arithmetic,
};

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// AdaDelta optimizer.
pub struct AdaDelta {
  /// Learning rate `λ`.
  pub learning_rate: LearningRate,
  /// Running-average coefficient `ρ`. Default Python: `0.9`.
  pub rho: f32,
  /// Numerical-stability epsilon. Default Python: `1e-6`.
  pub eps: f32,
  step_count: usize,
  current_lr: f32,
  state: HashMap<String, (Array, Array)>,
}

impl AdaDelta {
  /// Construct an [`AdaDelta`] optimizer.
  pub fn new(learning_rate: impl Into<LearningRate>, rho: f32, eps: f32) -> Result<Self> {
    if rho < 0.0 {
      return Err(Error::Backend {
        message: format!("AdaDelta: rho must be >= 0, got {rho}"),
      });
    }
    if eps < 0.0 {
      return Err(Error::Backend {
        message: format!("AdaDelta: epsilon must be >= 0, got {eps}"),
      });
    }
    let lr = learning_rate.into();
    let current_lr = lr.current(0);
    Ok(Self {
      learning_rate: lr,
      rho,
      eps,
      step_count: 0,
      current_lr,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor (`rho=0.9`, `eps=1e-6`).
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, 0.9, 1e-6)
  }
}

impl Optimizer for AdaDelta {
  fn init(&mut self, params: &Weights) -> Result<()> {
    let mut out = HashMap::with_capacity(params.len());
    for (key, value) in params {
      out.insert(key.clone(), (zeros_like(value)?, zeros_like(value)?));
    }
    self.state = out;
    Ok(())
  }

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    if self.state.is_empty() {
      self.init(gradients)?;
    }
    // Resolve scheduled LR at the PRE-increment step, then increment
    // (matches Python `optimizers.py:102..=106`).
    self.current_lr = self.learning_rate.current(self.step_count);
    self.step_count += 1;
    let rho_s = scalar(self.rho)?;
    let one_minus_rho = scalar(1.0 - self.rho)?;
    let eps_s = scalar(self.eps)?;
    let lr_s = scalar(self.current_lr)?;
    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      let (prev_v, prev_u) = match self.state.get(key) {
        Some((v, u)) => (v.try_clone()?, u.try_clone()?),
        None => (zeros_like(param)?, zeros_like(param)?),
      };
      // v = ρ·v + (1-ρ)·g²
      let g_sq = arithmetic::square(grad)?;
      let v_scaled = arithmetic::multiply(&rho_s, &prev_v)?;
      let g_sq_scaled = arithmetic::multiply(&one_minus_rho, &g_sq)?;
      let v_new = arithmetic::add(&v_scaled, &g_sq_scaled)?;
      // Δw = sqrt(u + eps) / sqrt(v + eps) · g
      let u_plus_eps = arithmetic::add(&prev_u, &eps_s)?;
      let v_plus_eps = arithmetic::add(&v_new, &eps_s)?;
      let sqrt_u = arithmetic::sqrt(&u_plus_eps)?;
      let sqrt_v = arithmetic::sqrt(&v_plus_eps)?;
      let ratio = arithmetic::divide(&sqrt_u, &sqrt_v)?;
      let dw = arithmetic::multiply(&ratio, grad)?;
      // u = ρ·u + (1-ρ)·Δw²
      let dw_sq = arithmetic::square(&dw)?;
      let u_scaled = arithmetic::multiply(&rho_s, &prev_u)?;
      let dw_sq_scaled = arithmetic::multiply(&one_minus_rho, &dw_sq)?;
      let u_new = arithmetic::add(&u_scaled, &dw_sq_scaled)?;
      // w_new = w - lr·Δw
      let step_term = arithmetic::multiply(&lr_s, &dw)?;
      let new_w = arithmetic::subtract(param, &step_term)?;
      params.insert(key.clone(), new_w);
      self.state.insert(key.clone(), (v_new, u_new));
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
  fn adadelta_single_step_matches_python_ref() -> Result<()> {
    // Python first step (u=v=0):
    //   v = (1-ρ)·g²
    //   Δw = sqrt(0+eps) / sqrt(v+eps) · g
    //   u = (1-ρ)·Δw²
    //   w_new = w - lr·Δw
    // w=1.0, g=0.5, lr=1.0, ρ=0.9, eps=1e-6
    //   v = 0.1·0.25 = 0.025
    //   Δw = sqrt(1e-6) / sqrt(0.025+1e-6) · 0.5
    //      = 0.001 / 0.158114... · 0.5
    //      ≈ 0.003163
    //   w_new ≈ 1.0 - 1.0·0.003163 = 0.996837
    let mut adadelta = AdaDelta::default_with_lr(1.0)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.5)?);
    adadelta.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.996_837).abs() < 1e-4, "got {got}");
    Ok(())
  }

  #[test]
  fn adadelta_rejects_negative_rho() {
    assert!(AdaDelta::new(1.0, -0.1, 1e-6).is_err());
  }

  #[test]
  fn adadelta_rejects_negative_eps() {
    assert!(AdaDelta::new(1.0, 0.9, -1e-6).is_err());
  }
}
