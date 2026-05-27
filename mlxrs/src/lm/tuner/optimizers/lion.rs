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

use smol_str::format_smolstr;

use crate::{
  Array, Result,
  error::{Error, NonFiniteScalarPayload, OutOfRangePayload},
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like, zeros_like_map},
  },
  ops::arithmetic,
};

/// Validate that `betas` are both finite and in `[0.0, 1.0)`.
fn validate_betas(betas: (f32, f32)) -> Result<()> {
  let (b1, b2) = betas;
  if !b1.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Lion: betas.0",
      b1 as f64,
    )));
  }
  if !b2.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Lion: betas.1",
      b2 as f64,
    )));
  }
  if !(0.0..1.0).contains(&b1) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Lion: betas.0",
      "must be in [0.0, 1.0)",
      format_smolstr!("{b1}"),
    )));
  }
  if !(0.0..1.0).contains(&b2) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Lion: betas.1",
      "must be in [0.0, 1.0)",
      format_smolstr!("{b2}"),
    )));
  }
  Ok(())
}

/// Validate that `weight_decay` is finite and `>= 0.0`.
fn validate_weight_decay(weight_decay: f32) -> Result<()> {
  if !weight_decay.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Lion: weight_decay",
      weight_decay as f64,
    )));
  }
  if weight_decay < 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Lion: weight_decay",
      "must be >= 0.0",
      format_smolstr!("{weight_decay}"),
    )));
  }
  Ok(())
}

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// Lion optimizer.
pub struct Lion {
  /// Learning rate `η`.
  learning_rate: LearningRate,
  /// Running-average coefficients `(β₁, β₂)`. Default Python:
  /// `(0.9, 0.99)`.
  betas: (f32, f32),
  /// Weight decay `λ`. Default Python: `0.0`.
  weight_decay: f32,
  step_count: usize,
  current_lr: f32,
  /// Skip-if-fresh stamp — `Some(N)` means `current_lr` is valid for step N.
  lr_resolved_for_step: Option<usize>,
  state: HashMap<String, Array>,
}

impl Lion {
  /// Construct a [`Lion`] optimizer.
  pub fn new(
    learning_rate: impl Into<LearningRate>,
    betas: (f32, f32),
    weight_decay: f32,
  ) -> Result<Self> {
    validate_betas(betas)?;
    validate_weight_decay(weight_decay)?;
    let lr = learning_rate.into();
    let current_lr = lr.try_current(0)?;
    Ok(Self {
      learning_rate: lr,
      betas,
      weight_decay,
      step_count: 0,
      current_lr,
      lr_resolved_for_step: None,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor (`betas=(0.9, 0.99)`, `wd=0.0`).
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, (0.9, 0.99), 0.0)
  }

  /// The learning rate (or schedule).
  #[inline(always)]
  pub fn learning_rate_ref(&self) -> &LearningRate {
    &self.learning_rate
  }

  /// Running-average coefficients `(β₁, β₂)`.
  #[inline(always)]
  pub fn betas(&self) -> (f32, f32) {
    self.betas
  }

  /// Weight decay coefficient.
  #[inline(always)]
  pub fn weight_decay(&self) -> f32 {
    self.weight_decay
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

  /// Set betas. Returns `Ok(self)` on success or `Err` if either beta is not
  /// finite or is outside `[0.0, 1.0)`.
  pub fn with_betas(mut self, betas: (f32, f32)) -> Result<Self> {
    validate_betas(betas)?;
    self.betas = betas;
    Ok(self)
  }

  /// Set weight decay. Returns `Ok(self)` on success or `Err` if
  /// `weight_decay` is not finite or `< 0.0`.
  pub fn with_weight_decay(mut self, weight_decay: f32) -> Result<Self> {
    validate_weight_decay(weight_decay)?;
    self.weight_decay = weight_decay;
    Ok(self)
  }
}

impl Optimizer for Lion {
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

  // ── Validation rejection tests ────────────────────────────────────────────

  #[test]
  fn lion_new_rejects_betas_above_one() {
    assert!(Lion::new(0.001, (0.9, 1.1), 0.0).is_err());
    assert!(Lion::new(0.001, (1.0, 0.99), 0.0).is_err());
  }

  #[test]
  fn lion_new_rejects_non_finite_betas() {
    assert!(Lion::new(0.001, (f32::NAN, 0.99), 0.0).is_err());
    assert!(Lion::new(0.001, (0.9, f32::INFINITY), 0.0).is_err());
  }

  #[test]
  fn lion_new_rejects_negative_weight_decay() {
    assert!(Lion::new(0.001, (0.9, 0.99), -0.1).is_err());
  }

  #[test]
  fn lion_new_rejects_non_finite_weight_decay() {
    assert!(Lion::new(0.001, (0.9, 0.99), f32::NAN).is_err());
  }

  #[test]
  fn lion_with_betas_rejects_above_one() {
    let res = Lion::default_with_lr(0.001).and_then(|l| l.with_betas((0.9, 1.1)));
    assert!(res.is_err());
  }

  #[test]
  fn lion_with_betas_rejects_non_finite() {
    let res = Lion::default_with_lr(0.001).and_then(|l| l.with_betas((f32::NAN, 0.99)));
    assert!(res.is_err());
  }

  #[test]
  fn lion_with_weight_decay_rejects_negative() {
    let res = Lion::default_with_lr(0.001).and_then(|l| l.with_weight_decay(-0.1));
    assert!(res.is_err());
  }

  #[test]
  fn lion_with_weight_decay_rejects_non_finite() {
    let res = Lion::default_with_lr(0.001).and_then(|l| l.with_weight_decay(f32::INFINITY));
    assert!(res.is_err());
  }

  #[test]
  fn lion_with_learning_rate_rejects_fixed_nan() {
    let res = Lion::default_with_lr(0.001)
      .and_then(|l| l.with_learning_rate(LearningRate::Fixed(f32::NAN)));
    assert!(res.is_err(), "with_learning_rate must reject Fixed(NaN)");
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
