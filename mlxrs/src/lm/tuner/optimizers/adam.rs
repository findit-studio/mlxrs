//! [`Adam`] / [`AdamW`] / [`Adamax`] — adaptive moments family.
//!
//! Mirrors Python `mlx.optimizers.Adam` / `AdamW` / `Adamax`
//! (`mlx/python/mlx/optimizers/optimizers.py:470..=647`) and the Swift
//! `Adam` / `AdamW` / `Adamax` classes in
//! `mlx-swift/Source/MLXOptimizers/Optimizers.swift`.
//!
//! Update formulas:
//!
//! Adam (`optimizers.py:512..=535`):
//!
//! ```text
//! m = β₁·m + (1-β₁)·g
//! v = β₂·v + (1-β₂)·g²
//! if bias_correction:
//!   c₁ = lr / (1 - β₁ᵗ); c₂ = rsqrt(1 - β₂ᵗ)
//!   w_new = w - (c₁·m) / (sqrt(v)·c₂ + eps)
//! else:
//!   w_new = w - lr·m / (sqrt(v) + eps)
//! ```
//!
//! AdamW (`optimizers.py:580..=588`): decoupled weight decay applied to the
//! parameter BEFORE Adam's step:
//!
//! ```text
//! w_decoupled = w·(1 - lr·weight_decay)
//! w_new = Adam_step(g, w_decoupled, state)
//! ```
//!
//! Adamax (`optimizers.py:632..=647`): replaces `v`'s running mean with an
//! ∞-norm (per-element max):
//!
//! ```text
//! m = β₁·m + (1-β₁)·g
//! v = max(β₂·v, |g|)
//! w_new = w - lr·m / (v + eps)
//! ```
//!
//! Per-parameter state: `(m, v)` tuple keyed by parameter name.

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

/// Validate that `betas` are both finite and in `[0.0, 1.0)`.
fn validate_betas(optimizer: &str, betas: (f32, f32)) -> Result<()> {
  let (b1, b2) = betas;
  if !b1.is_finite() || !b2.is_finite() || !(0.0..1.0).contains(&b1) || !(0.0..1.0).contains(&b2) {
    return Err(Error::Backend {
      message: format!("{optimizer}: betas must be finite and in [0.0, 1.0), got ({b1}, {b2})"),
    });
  }
  Ok(())
}

/// Validate that `eps` is finite and `>= 0.0`.
fn validate_eps(optimizer: &str, eps: f32) -> Result<()> {
  if !eps.is_finite() || eps < 0.0 {
    return Err(Error::Backend {
      message: format!("{optimizer}: epsilon must be finite and >= 0.0, got {eps}"),
    });
  }
  Ok(())
}

/// Validate that `weight_decay` is finite and `>= 0.0`.
fn validate_weight_decay(optimizer: &str, weight_decay: f32) -> Result<()> {
  if !weight_decay.is_finite() || weight_decay < 0.0 {
    return Err(Error::Backend {
      message: format!("{optimizer}: weight_decay must be finite and >= 0.0, got {weight_decay}"),
    });
  }
  Ok(())
}

/// `(m, v)` state pair shared by [`Adam`], [`AdamW`], and [`Adamax`].
type Moments = HashMap<String, (Array, Array)>;

fn fresh_moments(params: &Weights) -> Result<Moments> {
  let mut out = HashMap::with_capacity(params.len());
  for (key, value) in params {
    out.insert(key.clone(), (zeros_like(value)?, zeros_like(value)?));
  }
  Ok(out)
}

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// Adam optimizer.
///
/// Mirrors Python `mlx.optimizers.Adam`
/// (`mlx/python/mlx/optimizers/optimizers.py:470..=535`).
pub struct Adam {
  /// Learning rate `λ`.
  learning_rate: LearningRate,
  /// Running-average coefficients `(β₁, β₂)`. Default Python:
  /// `(0.9, 0.999)`.
  betas: (f32, f32),
  /// Numerical-stability epsilon. Default Python: `1e-8`.
  eps: f32,
  /// Bias-correction flag. Default Python: `False`.
  bias_correction: bool,
  step_count: usize,
  current_lr: f32,
  /// Step number at which `current_lr` was last resolved — skip-if-fresh stamp.
  /// `Some(0)` after construction (new() already resolves at step 0).
  lr_resolved_for_step: Option<usize>,
  /// Per-parameter `(m, v)` moments.
  pub(crate) state: Moments,
}

impl Adam {
  /// Construct an [`Adam`] optimizer.
  pub fn new(
    learning_rate: impl Into<LearningRate>,
    betas: (f32, f32),
    eps: f32,
    bias_correction: bool,
  ) -> Result<Self> {
    validate_betas("Adam", betas)?;
    validate_eps("Adam", eps)?;
    let lr = learning_rate.into();
    let current_lr = lr.try_current(0)?;
    Ok(Self {
      learning_rate: lr,
      betas,
      eps,
      bias_correction,
      step_count: 0,
      current_lr,
      lr_resolved_for_step: None,
      state: HashMap::new(),
    })
  }

  /// Convenience constructor with the Python defaults
  /// (`betas=(0.9, 0.999)`, `eps=1e-8`, `bias_correction=False`).
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, (0.9, 0.999), 1e-8, false)
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

  /// Numerical-stability epsilon.
  #[inline(always)]
  pub fn eps(&self) -> f32 {
    self.eps
  }

  /// Whether bias correction is enabled.
  #[inline(always)]
  pub fn bias_correction(&self) -> bool {
    self.bias_correction
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
    validate_betas("Adam", betas)?;
    self.betas = betas;
    Ok(self)
  }

  /// Set epsilon. Returns `Ok(self)` on success or `Err` if `eps` is not
  /// finite or `< 0.0`.
  pub fn with_eps(mut self, eps: f32) -> Result<Self> {
    validate_eps("Adam", eps)?;
    self.eps = eps;
    Ok(self)
  }

  /// Set bias_correction. Returns `self` for chaining.
  #[must_use]
  pub fn with_bias_correction(mut self, bias_correction: bool) -> Self {
    self.bias_correction = bias_correction;
    self
  }

  /// Adam's per-parameter update — shared by [`Adam`] and (via wrapping)
  /// [`AdamW`]. Returns the new weight; the optimizer caller is responsible
  /// for inserting it into `params` (because [`AdamW`] needs to mutate the
  /// parameter BEFORE calling this).
  fn adam_step(&mut self, key: &str, grad: &Array, param: &Array) -> Result<Array> {
    let (b1, b2) = self.betas;
    let lr = self.current_lr;

    let (prev_m, prev_v) = match self.state.get(key) {
      Some((m, v)) => (m.try_clone()?, v.try_clone()?),
      None => (zeros_like(param)?, zeros_like(param)?),
    };

    let b1_s = scalar(b1)?;
    let b2_s = scalar(b2)?;
    let one_minus_b1 = scalar(1.0 - b1)?;
    let one_minus_b2 = scalar(1.0 - b2)?;
    let eps_s = scalar(self.eps)?;

    // m = β₁·m + (1-β₁)·g
    let m_scaled = arithmetic::multiply(&b1_s, &prev_m)?;
    let g_scaled = arithmetic::multiply(&one_minus_b1, grad)?;
    let m_new = arithmetic::add(&m_scaled, &g_scaled)?;

    // v = β₂·v + (1-β₂)·g²
    let g_sq = arithmetic::square(grad)?;
    let v_scaled = arithmetic::multiply(&b2_s, &prev_v)?;
    let g_sq_scaled = arithmetic::multiply(&one_minus_b2, &g_sq)?;
    let v_new = arithmetic::add(&v_scaled, &g_sq_scaled)?;

    let step_term = if self.bias_correction {
      // c₁ = lr / (1 - β₁ᵗ); c₂ = rsqrt(1 - β₂ᵗ)
      let t = self.step_count as f32;
      let c1 = lr / (1.0 - b1.powf(t));
      let c2 = (1.0_f32 - b2.powf(t)).powf(-0.5);
      let c1_s = scalar(c1)?;
      let c2_s = scalar(c2)?;
      let num = arithmetic::multiply(&c1_s, &m_new)?;
      let sqrt_v = arithmetic::sqrt(&v_new)?;
      let denom = arithmetic::multiply(&sqrt_v, &c2_s)?;
      let denom = arithmetic::add(&denom, &eps_s)?;
      arithmetic::divide(&num, &denom)?
    } else {
      // w_new = w - lr·m / (sqrt(v) + eps)
      let lr_s = scalar(lr)?;
      let lr_m = arithmetic::multiply(&lr_s, &m_new)?;
      let sqrt_v = arithmetic::sqrt(&v_new)?;
      let denom = arithmetic::add(&sqrt_v, &eps_s)?;
      arithmetic::divide(&lr_m, &denom)?
    };

    let new_w = arithmetic::subtract(param, &step_term)?;
    self.state.insert(key.into(), (m_new, v_new));
    Ok(new_w)
  }
}

impl Optimizer for Adam {
  fn init(&mut self, params: &Weights) -> Result<()> {
    self.state = fresh_moments(params)?;
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
    // Adam's bias-correction term uses the POST-increment step (Python `step =
    // self.step` at `optimizers.py:519` runs AFTER the increment), preserved
    // here because `adam_step` reads `self.step_count` after the increment.
    self.preflight()?;
    self.step_count += 1;
    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      let param_clone = param.try_clone()?;
      let new_w = self.adam_step(key, grad, &param_clone)?;
      params.insert(key.clone(), new_w);
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

/// AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
///
/// Mirrors Python `mlx.optimizers.AdamW`
/// (`mlx/python/mlx/optimizers/optimizers.py:538..=588`).
pub struct AdamW {
  inner: Adam,
  /// Weight decay `λ`. Default Python: `0.01`.
  weight_decay: f32,
}

impl AdamW {
  /// Construct an [`AdamW`] optimizer.
  pub fn new(
    learning_rate: impl Into<LearningRate>,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    bias_correction: bool,
  ) -> Result<Self> {
    validate_weight_decay("AdamW", weight_decay)?;
    Ok(Self {
      inner: Adam::new(learning_rate, betas, eps, bias_correction)?,
      weight_decay,
    })
  }

  /// Convenience constructor with the Python defaults
  /// (`betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`,
  /// `bias_correction=False`).
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, (0.9, 0.999), 1e-8, 0.01, false)
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
    let current_lr = lr.try_current(self.inner.step_count)?;
    self.inner.learning_rate = lr;
    self.inner.current_lr = current_lr;
    self.inner.lr_resolved_for_step = Some(self.inner.step_count);
    Ok(self)
  }

  /// Set weight decay. Returns `Ok(self)` on success or `Err` if
  /// `weight_decay` is not finite or `< 0.0`.
  pub fn with_weight_decay(mut self, weight_decay: f32) -> Result<Self> {
    validate_weight_decay("AdamW", weight_decay)?;
    self.weight_decay = weight_decay;
    Ok(self)
  }
}

impl Optimizer for AdamW {
  fn init(&mut self, params: &Weights) -> Result<()> {
    self.inner.init(params)
  }

  fn preflight(&mut self) -> Result<()> {
    self.inner.preflight()
  }

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    if self.inner.state.is_empty() {
      self.inner.init(gradients)?;
    }
    // Resolve scheduled LR via skip-if-fresh cache (no-op if MultiOptimizer
    // already preflighted this step). Matches Python `optimizers.py:102..=106`.
    self.inner.preflight()?;
    self.inner.step_count += 1;
    let lr = self.inner.current_lr;
    let decay_factor = scalar(1.0 - lr * self.weight_decay)?;
    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      // Decoupled weight decay: w_decoupled = w·(1 - lr·wd).
      let w_decoupled = arithmetic::multiply(param, &decay_factor)?;
      let new_w = self.inner.adam_step(key, grad, &w_decoupled)?;
      params.insert(key.clone(), new_w);
    }
    Ok(())
  }

  fn step(&self) -> usize {
    self.inner.step_count
  }

  fn learning_rate(&self) -> f32 {
    self.inner.current_lr
  }
}

/// Adamax: Adam variant using the `∞`-norm denominator.
///
/// Mirrors Python `mlx.optimizers.Adamax`
/// (`mlx/python/mlx/optimizers/optimizers.py:591..=647`).
pub struct Adamax {
  /// Learning rate `λ`.
  learning_rate: LearningRate,
  /// Running-average coefficients `(β₁, β₂)`. Default Python:
  /// `(0.9, 0.999)`.
  betas: (f32, f32),
  /// Numerical-stability epsilon. Default Python: `1e-8`.
  eps: f32,
  step_count: usize,
  current_lr: f32,
  /// Skip-if-fresh stamp — `Some(N)` means `current_lr` is valid for step N.
  lr_resolved_for_step: Option<usize>,
  state: Moments,
}

impl Adamax {
  /// Construct an [`Adamax`] optimizer.
  pub fn new(learning_rate: impl Into<LearningRate>, betas: (f32, f32), eps: f32) -> Result<Self> {
    validate_betas("Adamax", betas)?;
    validate_eps("Adamax", eps)?;
    let lr = learning_rate.into();
    let current_lr = lr.try_current(0)?;
    Ok(Self {
      learning_rate: lr,
      betas,
      eps,
      step_count: 0,
      current_lr,
      lr_resolved_for_step: None,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor.
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, (0.9, 0.999), 1e-8)
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

  /// Set betas. Returns `Ok(self)` on success or `Err` if either beta is not
  /// finite or is outside `[0.0, 1.0)`.
  pub fn with_betas(mut self, betas: (f32, f32)) -> Result<Self> {
    validate_betas("Adamax", betas)?;
    self.betas = betas;
    Ok(self)
  }

  /// Set epsilon. Returns `Ok(self)` on success or `Err` if `eps` is not
  /// finite or `< 0.0`.
  pub fn with_eps(mut self, eps: f32) -> Result<Self> {
    validate_eps("Adamax", eps)?;
    self.eps = eps;
    Ok(self)
  }
}

impl Optimizer for Adamax {
  fn init(&mut self, params: &Weights) -> Result<()> {
    self.state = fresh_moments(params)?;
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
    let eps_s = scalar(self.eps)?;
    let lr_s = scalar(self.current_lr)?;
    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      let (prev_m, prev_v) = match self.state.get(key) {
        Some((m, v)) => (m.try_clone()?, v.try_clone()?),
        None => (zeros_like(param)?, zeros_like(param)?),
      };
      // m = β₁·m + (1-β₁)·g
      let m_scaled = arithmetic::multiply(&b1_s, &prev_m)?;
      let g_scaled = arithmetic::multiply(&one_minus_b1, grad)?;
      let m_new = arithmetic::add(&m_scaled, &g_scaled)?;
      // v = max(β₂·v, |g|)
      let v_scaled = arithmetic::multiply(&b2_s, &prev_v)?;
      let abs_g = arithmetic::abs(grad)?;
      let v_new = arithmetic::maximum(&v_scaled, &abs_g)?;
      // w_new = w - lr·m / (v + eps)
      let lr_m = arithmetic::multiply(&lr_s, &m_new)?;
      let denom = arithmetic::add(&v_new, &eps_s)?;
      let step_term = arithmetic::divide(&lr_m, &denom)?;
      let new_w = arithmetic::subtract(param, &step_term)?;
      params.insert(key.clone(), new_w);
      self.state.insert(key.clone(), (m_new, v_new));
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

  fn p_g(p: f32, g: f32) -> Result<(Weights, Weights)> {
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(p)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(g)?);
    Ok((params, grads))
  }

  #[test]
  fn adam_single_step_no_bias_correction_matches_python_ref() -> Result<()> {
    // Python (no bias correction): m=(1-β₁)g, v=(1-β₂)g²;
    // w_new = w - lr·m / (sqrt(v) + eps).
    // w=1.0, g=0.5, lr=0.001, β=(0.9,0.999), eps=1e-8
    //   m = 0.1 * 0.5 = 0.05
    //   v = 0.001 * 0.25 = 0.00025
    //   step = 0.001 * 0.05 / (sqrt(0.00025) + 1e-8)
    //        = 0.00005 / 0.01581138...
    //        ≈ 0.0031623
    //   w_new ≈ 1.0 - 0.0031623 = 0.9968377
    let mut adam = Adam::default_with_lr(0.001)?;
    let (mut params, grads) = p_g(1.0, 0.5)?;
    adam.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.996_837_7).abs() < 1e-5, "got {got}");
    Ok(())
  }

  #[test]
  fn adam_bias_correction_step1_matches_python_ref() -> Result<()> {
    // Python with bias correction at t=1:
    //   m=(1-β₁)g, v=(1-β₂)g²
    //   c₁ = lr / (1 - β₁); c₂ = rsqrt(1 - β₂)
    //   numerator = c₁·m
    //   denominator = sqrt(v)·c₂ + eps
    //   step = numerator / denominator
    // At t=1 this reduces to:
    //   c₁·m = (lr / 0.1) * 0.1*g = lr * g
    //   sqrt(v)*c₂ = |g| * sqrt(1-β₂) * rsqrt(1-β₂) = |g|
    //   step ≈ lr*g / (|g| + eps) = lr * sign(g) (for g != 0)
    // w=1.0, g=0.5, lr=0.001 → step ≈ 0.001 → w_new ≈ 0.999
    let mut adam = Adam::new(0.001, (0.9, 0.999), 1e-8, true)?;
    let (mut params, grads) = p_g(1.0, 0.5)?;
    adam.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.999).abs() < 1e-4, "got {got}");
    Ok(())
  }

  #[test]
  fn adamw_decoupled_weight_decay_applies_before_step() -> Result<()> {
    // Python AdamW first step: w_decoupled = w·(1 - lr·wd) then Adam step.
    // w=1.0, lr=0.001, wd=0.01 → w_decoupled = 1.0·(1 - 1e-5) = 0.99999
    // Then Adam step with g=0.5 ≈ 0.0031623 → w_new ≈ 0.99999 - 0.0031623
    //                                              ≈ 0.99683
    let mut adamw = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01, false)?;
    let (mut params, grads) = p_g(1.0, 0.5)?;
    adamw.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.996_827_7).abs() < 1e-5, "got {got}");
    Ok(())
  }

  #[test]
  fn adamax_single_step_matches_python_ref() -> Result<()> {
    // Python Adamax first step:
    //   m = (1-β₁)·g = 0.1·0.5 = 0.05
    //   v = max(β₂·0, |g|) = 0.5
    //   step = lr·m / (v + eps) = 0.001·0.05 / 0.5 = 1e-4
    //   w_new = 1.0 - 1e-4 = 0.9999
    let mut adamax = Adamax::default_with_lr(0.001)?;
    let (mut params, grads) = p_g(1.0, 0.5)?;
    adamax.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.9999).abs() < 1e-5, "got {got}");
    Ok(())
  }

  #[test]
  fn adam_two_consecutive_steps_advance_state() -> Result<()> {
    let mut adam = Adam::default_with_lr(0.001)?;
    let (mut params, grads) = p_g(1.0, 0.5)?;
    adam.apply_gradients(&grads, &mut params)?;
    let after_one = read_scalar(&params["w"])?;
    adam.apply_gradients(&grads, &mut params)?;
    let after_two = read_scalar(&params["w"])?;
    assert!(after_two < after_one, "weight should keep decreasing");
    assert_eq!(adam.step(), 2);
    Ok(())
  }

  #[test]
  fn adamax_builder_with_eps_rejects_negative() {
    let res = Adamax::default_with_lr(0.001).and_then(|a| a.with_eps(-1e-8));
    assert!(res.is_err());
  }

  // ── Adam betas validation ────────────────────────────────────────────────

  #[test]
  fn adam_new_rejects_betas_above_one() {
    // b2 >= 1.0 → sqrt(negative) at bias-correction → NaN weights
    assert!(Adam::new(0.001, (0.9, 1.1), 1e-8, false).is_err());
    assert!(Adam::new(0.001, (1.0, 0.999), 1e-8, false).is_err());
  }

  #[test]
  fn adam_new_rejects_betas_negative() {
    assert!(Adam::new(0.001, (-0.1, 0.999), 1e-8, false).is_err());
    assert!(Adam::new(0.001, (0.9, -0.1), 1e-8, false).is_err());
  }

  #[test]
  fn adam_new_rejects_non_finite_betas() {
    assert!(Adam::new(0.001, (f32::NAN, 0.999), 1e-8, false).is_err());
    assert!(Adam::new(0.001, (0.9, f32::INFINITY), 1e-8, false).is_err());
  }

  #[test]
  fn adam_with_betas_rejects_above_one() {
    let res = Adam::default_with_lr(0.001).and_then(|a| a.with_betas((0.9, 1.1)));
    assert!(res.is_err());
  }

  #[test]
  fn adam_with_betas_rejects_non_finite() {
    let res = Adam::default_with_lr(0.001).and_then(|a| a.with_betas((f32::NAN, 0.999)));
    assert!(res.is_err());
  }

  #[test]
  fn adam_with_eps_rejects_negative() {
    let res = Adam::default_with_lr(0.001).and_then(|a| a.with_eps(-1e-8));
    assert!(res.is_err());
  }

  #[test]
  fn adam_with_eps_rejects_non_finite() {
    let res = Adam::default_with_lr(0.001).and_then(|a| a.with_eps(f32::NAN));
    assert!(res.is_err());
  }

  // ── AdamW weight_decay validation ────────────────────────────────────────

  #[test]
  fn adamw_new_rejects_negative_weight_decay() {
    assert!(AdamW::new(0.001, (0.9, 0.999), 1e-8, -0.01, false).is_err());
  }

  #[test]
  fn adamw_new_rejects_non_finite_weight_decay() {
    assert!(AdamW::new(0.001, (0.9, 0.999), 1e-8, f32::NAN, false).is_err());
  }

  #[test]
  fn adamw_with_weight_decay_rejects_negative() {
    let res = AdamW::default_with_lr(0.001).and_then(|a| a.with_weight_decay(-0.1));
    assert!(res.is_err());
  }

  #[test]
  fn adamw_with_weight_decay_rejects_non_finite() {
    let res = AdamW::default_with_lr(0.001).and_then(|a| a.with_weight_decay(f32::INFINITY));
    assert!(res.is_err());
  }

  // ── Adamax betas validation ───────────────────────────────────────────────

  #[test]
  fn adamax_new_rejects_betas_above_one() {
    assert!(Adamax::new(0.001, (0.9, 1.1), 1e-8).is_err());
    assert!(Adamax::new(0.001, (1.0, 0.999), 1e-8).is_err());
  }

  #[test]
  fn adamax_new_rejects_non_finite_betas() {
    assert!(Adamax::new(0.001, (f32::NAN, 0.999), 1e-8).is_err());
    assert!(Adamax::new(0.001, (0.9, f32::INFINITY), 1e-8).is_err());
  }

  #[test]
  fn adamax_with_betas_rejects_above_one() {
    let res = Adamax::default_with_lr(0.001).and_then(|a| a.with_betas((0.9, 1.1)));
    assert!(res.is_err());
  }

  #[test]
  fn adamax_with_betas_rejects_non_finite() {
    let res = Adamax::default_with_lr(0.001).and_then(|a| a.with_betas((f32::NAN, 0.999)));
    assert!(res.is_err());
  }

  #[test]
  fn adamax_with_eps_rejects_non_finite() {
    let res = Adamax::default_with_lr(0.001).and_then(|a| a.with_eps(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn adam_with_learning_rate_rejects_fixed_nan() {
    let res = Adam::default_with_lr(0.001)
      .and_then(|a| a.with_learning_rate(LearningRate::Fixed(f32::NAN)));
    assert!(
      res.is_err(),
      "Adam::with_learning_rate must reject Fixed(NaN)"
    );
  }

  #[test]
  fn adamw_with_learning_rate_rejects_fixed_nan() {
    let res = AdamW::default_with_lr(0.001)
      .and_then(|a| a.with_learning_rate(LearningRate::Fixed(f32::NAN)));
    assert!(
      res.is_err(),
      "AdamW::with_learning_rate must reject Fixed(NaN)"
    );
  }

  #[test]
  fn adamax_with_learning_rate_rejects_fixed_nan() {
    let res = Adamax::default_with_lr(0.001)
      .and_then(|a| a.with_learning_rate(LearningRate::Fixed(f32::NAN)));
    assert!(
      res.is_err(),
      "Adamax::with_learning_rate must reject Fixed(NaN)"
    );
  }
}
