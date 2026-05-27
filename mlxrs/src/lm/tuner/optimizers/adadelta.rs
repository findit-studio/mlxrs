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

use smol_str::format_smolstr;

use crate::{
  Array, Result,
  error::{Error, NonFiniteScalarPayload, OutOfRangePayload},
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like},
  },
  ops::arithmetic,
};

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

fn validate_rho(rho: f32) -> Result<()> {
  if !rho.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "AdaDelta: rho",
      rho as f64,
    )));
  }
  if !(0.0..1.0).contains(&rho) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "AdaDelta: rho",
      "must be in [0.0, 1.0)",
      format_smolstr!("{rho}"),
    )));
  }
  Ok(())
}

fn validate_eps(eps: f32) -> Result<()> {
  if !eps.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "AdaDelta: eps",
      eps as f64,
    )));
  }
  if eps < 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "AdaDelta: eps",
      "must be >= 0.0",
      format_smolstr!("{eps}"),
    )));
  }
  Ok(())
}

/// AdaDelta optimizer.
pub struct AdaDelta {
  /// Learning rate `λ`.
  learning_rate: LearningRate,
  /// Running-average coefficient `ρ`. Default Python: `0.9`.
  rho: f32,
  /// Numerical-stability epsilon. Default Python: `1e-6`.
  eps: f32,
  step_count: usize,
  current_lr: f32,
  /// Skip-if-fresh stamp — `Some(N)` means `current_lr` is valid for step N.
  lr_resolved_for_step: Option<usize>,
  state: HashMap<String, (Array, Array)>,
}

impl AdaDelta {
  /// Construct an [`AdaDelta`] optimizer.
  pub fn new(learning_rate: impl Into<LearningRate>, rho: f32, eps: f32) -> Result<Self> {
    validate_rho(rho)?;
    validate_eps(eps)?;
    let lr = learning_rate.into();
    let current_lr = lr.try_current(0)?;
    Ok(Self {
      learning_rate: lr,
      rho,
      eps,
      step_count: 0,
      current_lr,
      lr_resolved_for_step: None,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor (`rho=0.9`, `eps=1e-6`).
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, 0.9, 1e-6)
  }

  /// The learning rate (or schedule).
  #[inline(always)]
  pub fn learning_rate_ref(&self) -> &LearningRate {
    &self.learning_rate
  }

  /// Running-average coefficient `ρ`.
  #[inline(always)]
  pub fn rho(&self) -> f32 {
    self.rho
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

  /// Set rho. Returns `Ok(self)` on success or `Err` if `rho` is not finite
  /// or is outside `[0.0, 1.0)`.
  pub fn with_rho(mut self, rho: f32) -> Result<Self> {
    validate_rho(rho)?;
    self.rho = rho;
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

impl Optimizer for AdaDelta {
  fn init(&mut self, params: &Weights) -> Result<()> {
    let mut out = HashMap::with_capacity(params.len());
    for (key, value) in params {
      out.insert(key.clone(), (zeros_like(value)?, zeros_like(value)?));
    }
    self.state = out;
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
  fn adadelta_rejects_rho_at_one() {
    // rho == 1.0 → running average never decays, equivalent to no update
    assert!(AdaDelta::new(1.0, 1.0, 1e-6).is_err());
  }

  #[test]
  fn adadelta_rejects_non_finite_rho() {
    assert!(AdaDelta::new(1.0, f32::NAN, 1e-6).is_err());
    assert!(AdaDelta::new(1.0, f32::INFINITY, 1e-6).is_err());
  }

  #[test]
  fn adadelta_rejects_negative_eps() {
    assert!(AdaDelta::new(1.0, 0.9, -1e-6).is_err());
  }

  #[test]
  fn adadelta_rejects_non_finite_eps() {
    assert!(AdaDelta::new(1.0, 0.9, f32::NAN).is_err());
  }

  #[test]
  fn adadelta_builder_with_rho_rejects_negative() {
    let res = AdaDelta::default_with_lr(1.0).and_then(|a| a.with_rho(-0.5));
    assert!(res.is_err());
  }

  #[test]
  fn adadelta_builder_with_rho_rejects_at_one() {
    let res = AdaDelta::default_with_lr(1.0).and_then(|a| a.with_rho(1.0));
    assert!(res.is_err());
  }

  #[test]
  fn adadelta_builder_with_rho_rejects_non_finite() {
    let res = AdaDelta::default_with_lr(1.0).and_then(|a| a.with_rho(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn adadelta_builder_with_eps_rejects_negative() {
    let res = AdaDelta::default_with_lr(1.0).and_then(|a| a.with_eps(-1e-6));
    assert!(res.is_err());
  }

  #[test]
  fn adadelta_builder_with_eps_rejects_non_finite() {
    let res = AdaDelta::default_with_lr(1.0).and_then(|a| a.with_eps(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn adadelta_with_learning_rate_rejects_fixed_nan() {
    let res = AdaDelta::default_with_lr(1.0)
      .and_then(|a| a.with_learning_rate(LearningRate::Fixed(f32::NAN)));
    assert!(res.is_err(), "with_learning_rate must reject Fixed(NaN)");
  }
}
