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

/// Validate that `betas` are both finite and in `[0.0, 1.0)`. `context` is the
/// `<optimizer>: betas.<n>` site label and MUST be `'static` so the typed
/// payload retains a static call-site (no `format!` allocation per error).
fn validate_betas(
  context_b1: &'static str,
  context_b2: &'static str,
  betas: (f32, f32),
) -> Result<()> {
  let (b1, b2) = betas;
  if !b1.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      context_b1, b1 as f64,
    )));
  }
  if !b2.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      context_b2, b2 as f64,
    )));
  }
  if !(0.0..1.0).contains(&b1) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context_b1,
      "must be in [0.0, 1.0)",
      format_smolstr!("{b1}"),
    )));
  }
  if !(0.0..1.0).contains(&b2) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context_b2,
      "must be in [0.0, 1.0)",
      format_smolstr!("{b2}"),
    )));
  }
  Ok(())
}

/// Validate that `eps` is finite and `>= 0.0`.
fn validate_eps(context: &'static str, eps: f32) -> Result<()> {
  if !eps.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      context, eps as f64,
    )));
  }
  if eps < 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "must be >= 0.0",
      format_smolstr!("{eps}"),
    )));
  }
  Ok(())
}

/// Validate that `weight_decay` is finite and `>= 0.0`.
fn validate_weight_decay(context: &'static str, weight_decay: f32) -> Result<()> {
  if !weight_decay.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      context,
      weight_decay as f64,
    )));
  }
  if weight_decay < 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "must be >= 0.0",
      format_smolstr!("{weight_decay}"),
    )));
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
    validate_betas("Adam: betas.0", "Adam: betas.1", betas)?;
    validate_eps("Adam: eps", eps)?;
    let lr = learning_rate.into();
    let current_lr = lr.try_current(0)?;
    Ok(Self {
      learning_rate: lr,
      betas,
      eps,
      bias_correction,
      step_count: 0,
      current_lr,
      // Stamp the cache for step 0: the constructor's `try_current(0)` above
      // already consumed one schedule slot. Leaving `None` would force the
      // first `preflight()` at step 0 to re-resolve, double-calling stateful
      // schedules.
      lr_resolved_for_step: Some(0),
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
    validate_betas("Adam: betas.0", "Adam: betas.1", betas)?;
    self.betas = betas;
    Ok(self)
  }

  /// Set epsilon. Returns `Ok(self)` on success or `Err` if `eps` is not
  /// finite or `< 0.0`.
  pub fn with_eps(mut self, eps: f32) -> Result<Self> {
    validate_eps("Adam: eps", eps)?;
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
    validate_weight_decay("AdamW: weight_decay", weight_decay)?;
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
    validate_weight_decay("AdamW: weight_decay", weight_decay)?;
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
    validate_betas("Adamax: betas.0", "Adamax: betas.1", betas)?;
    validate_eps("Adamax: eps", eps)?;
    let lr = learning_rate.into();
    let current_lr = lr.try_current(0)?;
    Ok(Self {
      learning_rate: lr,
      betas,
      eps,
      step_count: 0,
      current_lr,
      // Stamp the cache for step 0: the constructor's `try_current(0)` above
      // already consumed one schedule slot. Leaving `None` would force the
      // first `preflight()` at step 0 to re-resolve, double-calling stateful
      // schedules.
      lr_resolved_for_step: Some(0),
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
    validate_betas("Adamax: betas.0", "Adamax: betas.1", betas)?;
    self.betas = betas;
    Ok(self)
  }

  /// Set epsilon. Returns `Ok(self)` on success or `Err` if `eps` is not
  /// finite or `< 0.0`.
  pub fn with_eps(mut self, eps: f32) -> Result<Self> {
    validate_eps("Adamax: eps", eps)?;
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
mod tests;
