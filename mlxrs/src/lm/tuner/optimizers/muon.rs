//! [`Muon`] — MomentUm Orthogonalized by Newton-schulz (Keller Jordan, 2024).
//!
//! Mirrors Python `mlx.optimizers.Muon`
//! (`mlx/python/mlx/optimizers/optimizers.py:851..=948`).
//!
//! Update formula:
//!
//! ```text
//! if weight_decay != 0: g = g + weight_decay·w
//! v = momentum·v + (1-momentum)·g
//! if nesterov: update = g·(1-momentum) + v·momentum
//! else:        update = v
//! lr_ = lr.astype(g.dtype)
//! if update.ndim >= 2:
//!   if update.ndim > 2: reshape to (M, prod(rest))
//!   update = newton_schulz5(update, steps=ns_steps)
//!   if reshaped: reshape back
//!   lr_ *= max(1, M/N)^0.5
//! w_new = w - lr_·update
//! ```
//!
//! Newton-Schulz iteration (5-step orthogonalization) — see
//! `_zeropower_via_newtonschulz5` in the python ref. Each step uses two
//! `addmm` calls with the polynomial coefficients `(a, b, c) =
//! (3.4445, -4.7750, 2.0315)`.
//!
//! Per-parameter state: a single velocity `v` Array.

use std::collections::HashMap;

use smol_str::format_smolstr;

use crate::{
  Array, Result,
  error::{Error, NonFiniteScalarPayload, OutOfRangePayload, RankMismatchPayload},
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like, zeros_like_map},
  },
  ops::{arithmetic, linalg_basic::addmm, linalg_full::norm_l2, shape::reshape},
};

/// Validate that `momentum` is finite. Non-finite momentum propagates NaN/Inf
/// into the velocity accumulator at the first `apply_gradients` call.
fn validate_momentum_finite(momentum: f32) -> Result<()> {
  if !momentum.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Muon: momentum",
      momentum as f64,
    )));
  }
  Ok(())
}

/// Validate that `weight_decay` is finite and `>= 0.0`.
fn validate_weight_decay(weight_decay: f32) -> Result<()> {
  if !weight_decay.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Muon: weight_decay",
      weight_decay as f64,
    )));
  }
  if weight_decay < 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Muon: weight_decay",
      "must be >= 0.0",
      format_smolstr!("{weight_decay}"),
    )));
  }
  Ok(())
}

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// Muon optimizer.
pub struct Muon {
  /// Learning rate `λ`.
  learning_rate: LearningRate,
  /// Momentum strength. Default Python: `0.95`.
  momentum: f32,
  /// Weight decay (L2 penalty). Default Python: `0.01`.
  weight_decay: f32,
  /// Enable Nesterov momentum. Default Python: `true`.
  nesterov: bool,
  /// Newton-Schulz iteration steps. Default Python: `5`.
  ns_steps: usize,
  step_count: usize,
  current_lr: f32,
  /// Skip-if-fresh stamp — `Some(N)` means `current_lr` is valid for step N.
  lr_resolved_for_step: Option<usize>,
  state: HashMap<String, Array>,
}

impl Muon {
  /// Construct a [`Muon`] optimizer.
  pub fn new(
    learning_rate: impl Into<LearningRate>,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    ns_steps: usize,
  ) -> Result<Self> {
    validate_momentum_finite(momentum)?;
    validate_weight_decay(weight_decay)?;
    let lr = learning_rate.into();
    let current_lr = lr.try_current(0)?;
    Ok(Self {
      learning_rate: lr,
      momentum,
      weight_decay,
      nesterov,
      ns_steps,
      step_count: 0,
      current_lr,
      lr_resolved_for_step: None,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor.
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, 0.95, 0.01, true, 5)
  }

  /// The learning rate (or schedule).
  #[inline(always)]
  pub fn learning_rate_ref(&self) -> &LearningRate {
    &self.learning_rate
  }

  /// Momentum strength.
  #[inline(always)]
  pub fn momentum(&self) -> f32 {
    self.momentum
  }

  /// Weight decay coefficient.
  #[inline(always)]
  pub fn weight_decay(&self) -> f32 {
    self.weight_decay
  }

  /// Whether Nesterov momentum is enabled.
  #[inline(always)]
  pub fn nesterov(&self) -> bool {
    self.nesterov
  }

  /// Newton-Schulz iteration step count.
  #[inline(always)]
  pub fn ns_steps(&self) -> usize {
    self.ns_steps
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

  /// Set momentum. Returns `Ok(self)` on success or `Err` if `momentum` is
  /// not finite (NaN/Inf would corrupt velocity at the first
  /// `apply_gradients` call).
  pub fn with_momentum(mut self, momentum: f32) -> Result<Self> {
    validate_momentum_finite(momentum)?;
    self.momentum = momentum;
    Ok(self)
  }

  /// Set weight decay. Returns `Ok(self)` on success or `Err` if
  /// `weight_decay` is not finite or `< 0.0`.
  pub fn with_weight_decay(mut self, weight_decay: f32) -> Result<Self> {
    validate_weight_decay(weight_decay)?;
    self.weight_decay = weight_decay;
    Ok(self)
  }

  /// Set nesterov flag. Returns `self` for chaining.
  #[must_use]
  pub fn with_nesterov(mut self, nesterov: bool) -> Self {
    self.nesterov = nesterov;
    self
  }

  /// Set Newton-Schulz iteration steps. Returns `self` for chaining.
  #[must_use]
  pub fn with_ns_steps(mut self, ns_steps: usize) -> Self {
    self.ns_steps = ns_steps;
    self
  }

  /// Newton-Schulz 5 iteration on a 2D matrix. Mirrors the Python
  /// `_zeropower_via_newtonschulz5` (`optimizers.py:896..=915`).
  fn newton_schulz5(&self, x: &Array, steps: usize) -> Result<Array> {
    let shape = x.shape();
    if shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "Muon.newton_schulz5: expected 2D input",
        shape.len() as u32,
        shape.to_vec(),
      )));
    }
    let (a, b, c) = (3.4445_f32, -4.7750_f32, 2.0315_f32);
    let transpose_needed = shape[shape.len() - 2] > shape[shape.len() - 1];
    let mut x = if transpose_needed {
      x.transpose()?
    } else {
      x.try_clone()?
    };
    // x = x / (norm(x, keepdims=True) + 1e-7)
    let n = norm_l2(&x, &[], true)?;
    let denom = arithmetic::add(&n, &scalar(1e-7)?)?;
    x = arithmetic::divide(&x, &denom)?;
    for _ in 0..steps {
      // A = x @ x.T
      let xt = x.transpose()?;
      let a_mat = crate::ops::linalg_basic::matmul(&x, &xt)?;
      // B = addmm(b*A, A, A, beta=1.0, alpha=c)
      let b_a = arithmetic::multiply(&scalar(b)?, &a_mat)?;
      let big_b = addmm(&b_a, &a_mat, &a_mat, c, 1.0)?;
      // x = addmm(a*x, B, x, beta=1.0, alpha=1.0)
      let a_x = arithmetic::multiply(&scalar(a)?, &x)?;
      x = addmm(&a_x, &big_b, &x, 1.0, 1.0)?;
    }
    if transpose_needed {
      x = x.transpose()?;
    }
    Ok(x)
  }
}

impl Optimizer for Muon {
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
    let mu_s = scalar(self.momentum)?;
    let one_minus_mu = scalar(1.0 - self.momentum)?;
    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      // weight decay: g = g + wd·w
      let g_eff = if self.weight_decay != 0.0 {
        let wd_s = scalar(self.weight_decay)?;
        let wd_term = arithmetic::multiply(&wd_s, param)?;
        arithmetic::add(grad, &wd_term)?
      } else {
        grad.try_clone()?
      };
      // v = momentum·v + (1-momentum)·g
      let prev_v = match self.state.get(key) {
        Some(v) => v.try_clone()?,
        None => zeros_like(param)?,
      };
      let v_scaled = arithmetic::multiply(&mu_s, &prev_v)?;
      let g_scaled = arithmetic::multiply(&one_minus_mu, &g_eff)?;
      let v_new = arithmetic::add(&v_scaled, &g_scaled)?;
      // update = nesterov ? g·(1-μ) + v·μ : v
      let mut update = if self.nesterov {
        let g_term = arithmetic::multiply(&g_eff, &one_minus_mu)?;
        let v_term = arithmetic::multiply(&v_new, &mu_s)?;
        arithmetic::add(&g_term, &v_term)?
      } else {
        v_new.try_clone()?
      };
      let mut lr_eff = self.current_lr;
      let original_shape = update.shape();
      if update.ndim() >= 2 {
        let reshape_needed = update.ndim() > 2;
        if reshape_needed {
          // (M, prod(rest))
          let m_dim = original_shape[0];
          let n_dim: usize = original_shape[1..].iter().product();
          update = reshape(&update, &(m_dim, n_dim))?;
        }
        update = self.newton_schulz5(&update, self.ns_steps)?;
        if reshape_needed {
          update = reshape(&update, &original_shape.as_slice())?;
        }
        let updated_shape = update.shape();
        let m_d = updated_shape[updated_shape.len() - 2] as f32;
        let n_d = updated_shape[updated_shape.len() - 1] as f32;
        lr_eff *= (1.0_f32.max(m_d / n_d)).sqrt();
      }
      let lr_s = scalar(lr_eff)?;
      let step_term = arithmetic::multiply(&lr_s, &update)?;
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

  // ── Validation rejection tests ────────────────────────────────────────────

  #[test]
  fn muon_new_rejects_non_finite_momentum() {
    assert!(Muon::new(0.01, f32::NAN, 0.01, true, 5).is_err());
    assert!(Muon::new(0.01, f32::INFINITY, 0.01, true, 5).is_err());
  }

  #[test]
  fn muon_new_rejects_negative_weight_decay() {
    assert!(Muon::new(0.01, 0.95, -0.1, true, 5).is_err());
  }

  #[test]
  fn muon_new_rejects_non_finite_weight_decay() {
    assert!(Muon::new(0.01, 0.95, f32::NAN, true, 5).is_err());
  }

  #[test]
  fn muon_with_momentum_rejects_non_finite() {
    let res = Muon::default_with_lr(0.01).and_then(|m| m.with_momentum(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn muon_with_weight_decay_rejects_negative() {
    let res = Muon::default_with_lr(0.01).and_then(|m| m.with_weight_decay(-0.1));
    assert!(res.is_err());
  }

  #[test]
  fn muon_with_weight_decay_rejects_non_finite() {
    let res = Muon::default_with_lr(0.01).and_then(|m| m.with_weight_decay(f32::INFINITY));
    assert!(res.is_err());
  }

  #[test]
  fn muon_with_learning_rate_rejects_fixed_nan() {
    let res =
      Muon::default_with_lr(0.01).and_then(|m| m.with_learning_rate(LearningRate::Fixed(f32::NAN)));
    assert!(res.is_err(), "with_learning_rate must reject Fixed(NaN)");
  }

  #[test]
  fn muon_1d_param_runs_without_newton_schulz() -> Result<()> {
    // 1D params skip the Newton-Schulz branch (ndim < 2) and reduce to
    // plain momentum SGD (with weight decay).
    let mut muon = Muon::default_with_lr(0.01)?;
    let mut params: Weights = HashMap::new();
    params.insert(
      "w".into(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3])?,
    );
    let mut grads: Weights = HashMap::new();
    grads.insert(
      "w".into(),
      Array::from_slice::<f32>(&[0.1, 0.2, 0.3], &[3])?,
    );
    muon.apply_gradients(&grads, &mut params)?;
    let mut got = params["w"].try_clone()?;
    let v: Vec<f32> = got.to_vec()?;
    // First step with default momentum=0.95, nesterov=true, weight_decay=0.01:
    //   g_eff[i] = g[i] + 0.01·w[i]
    //   v = 0.05·g_eff
    //   update = g_eff·0.05 + v·0.95 = 0.05·g_eff + 0.95·(0.05·g_eff)
    //          = 0.05·g_eff·(1 + 0.95) = 0.0975·g_eff
    //   w_new = w - lr·update = w - 0.01·0.0975·g_eff
    //         = w - 0.000975·g_eff
    // For w[0]=1.0, g_eff[0] = 0.1 + 0.01 = 0.11
    //   w_new[0] = 1.0 - 0.000975·0.11 = 0.99989275
    assert!(
      (v[0] - 0.999_892_8).abs() < 1e-5,
      "expected ~0.9998928, got {}",
      v[0]
    );
    Ok(())
  }

  #[test]
  fn muon_2d_param_invokes_newton_schulz_branch() -> Result<()> {
    let mut muon = Muon::new(0.01, 0.0, 0.0, false, 5)?;
    let mut params: Weights = HashMap::new();
    params.insert(
      "w".into(),
      Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2))?,
    );
    let mut grads: Weights = HashMap::new();
    grads.insert(
      "w".into(),
      Array::from_slice::<f32>(&[0.5, 0.0, 0.0, 0.5], &(2, 2))?,
    );
    muon.apply_gradients(&grads, &mut params)?;
    let mut got = params["w"].try_clone()?;
    let v: Vec<f32> = got.to_vec()?;
    // Just verify the update moved (no exact match — Newton-Schulz output
    // depends on the polynomial iteration; the test confirms the branch
    // ran and produced a finite, distinct value).
    assert!(v[0].is_finite() && v[3].is_finite());
    assert!((v[0] - 1.0).abs() > 1e-6 || (v[3] - 1.0).abs() > 1e-6);
    Ok(())
  }
}
