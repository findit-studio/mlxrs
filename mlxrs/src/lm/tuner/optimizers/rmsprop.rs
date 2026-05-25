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

/// Validate `alpha` is finite and in `[0.0, 1.0)`.
fn validate_alpha(alpha: f32) -> Result<()> {
  if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
    return Err(Error::Backend {
      message: format!("RMSprop: alpha must be finite and in [0.0, 1.0), got {alpha}"),
    });
  }
  Ok(())
}

/// Validate `eps` is finite and `>= 0.0`.
fn validate_eps(eps: f32) -> Result<()> {
  if !eps.is_finite() || eps < 0.0 {
    return Err(Error::Backend {
      message: format!("RMSprop: eps must be finite and >= 0.0, got {eps}"),
    });
  }
  Ok(())
}

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// RMSprop optimizer.
pub struct RMSprop {
  /// Learning rate `λ`.
  learning_rate: LearningRate,
  /// Smoothing constant `α`. Default Python: `0.99`.
  alpha: f32,
  /// Numerical-stability epsilon. Default Python: `1e-8`.
  eps: f32,
  step_count: usize,
  current_lr: f32,
  /// Skip-if-fresh stamp — `Some(N)` means `current_lr` is valid for step N.
  lr_resolved_for_step: Option<usize>,
  state: HashMap<String, Array>,
}

impl RMSprop {
  /// Construct an [`RMSprop`] optimizer.
  pub fn new(learning_rate: impl Into<LearningRate>, alpha: f32, eps: f32) -> Result<Self> {
    validate_alpha(alpha)?;
    validate_eps(eps)?;
    let lr = learning_rate.into();
    let current_lr = lr.try_current(0)?;
    Ok(Self {
      learning_rate: lr,
      alpha,
      eps,
      step_count: 0,
      current_lr,
      lr_resolved_for_step: None,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor (`alpha=0.99`, `eps=1e-8`).
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, 0.99, 1e-8)
  }

  /// The learning rate (or schedule).
  #[inline(always)]
  pub fn learning_rate_ref(&self) -> &LearningRate {
    &self.learning_rate
  }

  /// Smoothing constant `α`.
  #[inline(always)]
  pub fn alpha(&self) -> f32 {
    self.alpha
  }

  /// Numerical-stability epsilon.
  #[inline(always)]
  pub fn eps(&self) -> f32 {
    self.eps
  }

  /// Set the learning rate. Returns `Ok(self)` on success or `Err` if the
  /// resolved value at the current step is non-finite.
  pub fn with_learning_rate(mut self, learning_rate: impl Into<LearningRate>) -> Result<Self> {
    let lr = learning_rate.into();
    let current_lr = lr.try_current(self.step_count)?;
    self.learning_rate = lr;
    self.current_lr = current_lr;
    self.lr_resolved_for_step = Some(self.step_count);
    Ok(self)
  }

  /// Set alpha. Returns `Ok(self)` on success or `Err` if `alpha` is not
  /// finite or outside `[0.0, 1.0)`.
  pub fn with_alpha(mut self, alpha: f32) -> Result<Self> {
    validate_alpha(alpha)?;
    self.alpha = alpha;
    Ok(self)
  }

  /// Set epsilon. Returns `Ok(self)` on success or `Err` if `eps` is not
  /// finite or `< 0.0`.
  pub fn with_eps(mut self, eps: f32) -> Result<Self> {
    validate_eps(eps)?;
    self.eps = eps;
    Ok(self)
  }
}

impl Optimizer for RMSprop {
  fn init(&mut self, params: &Weights) -> Result<()> {
    self.state = zeros_like_map(params)?;
    Ok(())
  }

  fn preflight(&mut self) -> Result<()> {
    if self.lr_resolved_for_step == Some(self.step_count) {
      return Ok(()); // cache hit: schedule already consulted at this step
    }
    self.current_lr = self.learning_rate.try_current(self.step_count)?;
    self.lr_resolved_for_step = Some(self.step_count);
    Ok(())
  }

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    if self.state.is_empty() {
      self.init(gradients)?;
    }
    // Resolve scheduled LR via skip-if-fresh cache (no-op if MultiOptimizer
    // already preflighted this step). Matches Python `optimizers.py:102..=106`.
    self.preflight()?;
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
  fn rmsprop_new_rejects_alpha_above_one() {
    // alpha >= 1.0 makes the EMA non-contractive (denominator → 0).
    assert!(RMSprop::new(0.001, 1.0, 1e-8).is_err());
    assert!(RMSprop::new(0.001, 1.5, 1e-8).is_err());
  }

  #[test]
  fn rmsprop_new_rejects_nan_alpha() {
    assert!(RMSprop::new(0.001, f32::NAN, 1e-8).is_err());
  }

  #[test]
  fn rmsprop_rejects_negative_eps() {
    assert!(RMSprop::new(0.001, 0.99, -1e-8).is_err());
  }

  #[test]
  fn rmsprop_new_rejects_nan_eps() {
    assert!(RMSprop::new(0.001, 0.99, f32::NAN).is_err());
  }

  #[test]
  fn rmsprop_builder_with_alpha_rejects_negative() {
    let res = RMSprop::default_with_lr(0.001).and_then(|r| r.with_alpha(-1.0));
    assert!(res.is_err());
  }

  #[test]
  fn rmsprop_with_alpha_rejects_above_one() {
    let res = RMSprop::default_with_lr(0.001).and_then(|r| r.with_alpha(1.0));
    assert!(res.is_err());
  }

  #[test]
  fn rmsprop_with_alpha_rejects_nan() {
    let res = RMSprop::default_with_lr(0.001).and_then(|r| r.with_alpha(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn rmsprop_builder_with_eps_rejects_negative() {
    let res = RMSprop::default_with_lr(0.001).and_then(|r| r.with_eps(-1.0));
    assert!(res.is_err());
  }

  #[test]
  fn rmsprop_with_eps_rejects_nan() {
    let res = RMSprop::default_with_lr(0.001).and_then(|r| r.with_eps(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn rmsprop_with_learning_rate_rejects_fixed_nan() {
    let res = RMSprop::default_with_lr(0.001)
      .and_then(|r| r.with_learning_rate(LearningRate::Fixed(f32::NAN)));
    assert!(res.is_err(), "with_learning_rate must reject Fixed(NaN)");
  }
}
