//! [`SGD`] — stochastic gradient descent with optional (Nesterov) momentum,
//! weight decay, and dampening.
//!
//! Mirrors Python `mlx.optimizers.SGD`
//! (`mlx/python/mlx/optimizers/optimizers.py:230..=294`) and Swift `SGD`
//! (`mlx-swift/Source/MLXOptimizers/Optimizers.swift`).
//!
//! Update formula (`optimizers.py:231..=294`):
//!
//! ```text
//! if weight_decay != 0:  g = g + weight_decay * w
//! if momentum <= 0:      w_new = w - lr * g                 (vanilla SGD)
//! else:                  v = momentum * v
//!                        v += (1 - dampening) * g     if dampening > 0
//!                        v += g                       otherwise
//!                        update = g + momentum * v    if nesterov
//!                                  v                  otherwise
//!                        w_new = w - lr * update
//! ```
//!
//! Per-parameter state: a single velocity `Array` keyed by parameter name.

use std::collections::HashMap;

use smol_str::format_smolstr;

use crate::{
  Array, Result,
  error::{Error, InvariantViolationPayload, NonFiniteScalarPayload, OutOfRangePayload},
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like, zeros_like_map},
  },
  ops::arithmetic,
};

/// Stochastic gradient descent.
///
/// Mirrors Python `mlx.optimizers.SGD`
/// (`mlx/python/mlx/optimizers/optimizers.py:230..=294`).
pub struct SGD {
  /// Learning rate `λ` (or a step-driven schedule producing the same).
  learning_rate: LearningRate,
  /// Momentum coefficient `µ`. Default Python: `0.0` (vanilla SGD).
  momentum: f32,
  /// Weight decay (L2 penalty). Default Python: `0.0`.
  weight_decay: f32,
  /// Dampening `τ`. Default Python: `0.0`.
  dampening: f32,
  /// Enable Nesterov momentum. Default Python: `false`. Requires
  /// `momentum > 0` and `dampening == 0` (checked at construction).
  nesterov: bool,
  /// 1-based step counter, incremented at the top of every
  /// [`SGD::apply_gradients`] call (matches Python).
  step_count: usize,
  /// Last resolved learning rate after schedule eval (for
  /// [`Optimizer::learning_rate`]).
  current_lr: f32,
  /// Step number at which `current_lr` was last resolved via
  /// [`Optimizer::preflight`] — used as the skip-if-fresh stamp so a
  /// schedule is consulted at most once per step regardless of caller
  /// pattern (standalone or [`super::multi::MultiOptimizer`]).
  ///
  /// `None` until the first preflight; `Some(N)` means `current_lr` is
  /// the schedule's value at step `N` and any preflight at the same `N`
  /// is a no-op (cache hit).
  lr_resolved_for_step: Option<usize>,
  /// Per-parameter velocity state `v` (Python `state["v"]`).
  state: HashMap<String, Array>,
}

impl SGD {
  /// Construct an [`SGD`] optimizer. Mirrors Python `SGD.__init__`
  /// (`optimizers.py:248..=266`).
  ///
  /// Errors with [`Error::Backend`] if `nesterov && (momentum <= 0 ||
  /// dampening != 0)` — same precondition as the Python `ValueError`.
  pub fn new(
    learning_rate: impl Into<LearningRate>,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: bool,
  ) -> Result<Self> {
    Self::validate_momentum_finite(momentum)?;
    Self::validate_weight_decay(weight_decay)?;
    Self::validate_dampening(dampening)?;
    Self::validate_nesterov(momentum, dampening, nesterov)?;
    let lr = learning_rate.into();
    let current_lr = lr.try_current(0)?;
    Ok(Self {
      learning_rate: lr,
      momentum,
      weight_decay,
      dampening,
      nesterov,
      step_count: 0,
      current_lr,
      lr_resolved_for_step: None,
      state: HashMap::new(),
    })
  }

  /// Construct a vanilla SGD (no momentum / decay / dampening / Nesterov).
  /// Convenience wrapper over [`SGD::new`].
  pub fn vanilla(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, 0.0, 0.0, 0.0, false)
  }

  /// Validate the Nesterov invariant: `nesterov` requires `momentum > 0` and
  /// `dampening == 0`. Called by both `new` and the `with_*` builders that
  /// affect these fields.
  ///
  /// Explicit `!is_finite() || momentum <= 0.0` (not a negated `momentum > 0.0`,
  /// which trips clippy's `neg_cmp_op_on_partial_ord`) so that `f32::NAN`
  /// AND non-positive momentum both trip the guard — under IEEE-754 every
  /// comparison with NaN is false, so a bare `<= 0.0` would silently accept
  /// NaN and propagate it into velocity/weights at the first `apply_gradients`.
  fn validate_nesterov(momentum: f32, dampening: f32, nesterov: bool) -> Result<()> {
    if nesterov && (!momentum.is_finite() || momentum <= 0.0 || dampening != 0.0) {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "SGD: Nesterov momentum",
        "requires momentum > 0 (finite) and dampening == 0",
      )));
    }
    Ok(())
  }

  /// Reject non-finite momentum unconditionally (independent of the Nesterov
  /// branch): even with `nesterov=false`, `momentum=f32::NAN` or `Inf` would
  /// propagate into the velocity Array at the first `apply_gradients` call.
  /// Called by both `new` and `with_momentum`.
  fn validate_momentum_finite(momentum: f32) -> Result<()> {
    if !momentum.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "SGD: momentum",
        momentum as f64,
      )));
    }
    Ok(())
  }

  /// Validate `weight_decay` is finite and `>= 0.0`.
  fn validate_weight_decay(weight_decay: f32) -> Result<()> {
    if !weight_decay.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "SGD: weight_decay",
        weight_decay as f64,
      )));
    }
    if weight_decay < 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "SGD: weight_decay",
        "must be >= 0.0",
        format_smolstr!("{weight_decay}"),
      )));
    }
    Ok(())
  }

  /// Validate `dampening` is finite and `>= 0.0`.
  fn validate_dampening(dampening: f32) -> Result<()> {
    if !dampening.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "SGD: dampening",
        dampening as f64,
      )));
    }
    if dampening < 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "SGD: dampening",
        "must be >= 0.0",
        format_smolstr!("{dampening}"),
      )));
    }
    Ok(())
  }

  /// The learning rate (or schedule) of this optimizer.
  #[inline(always)]
  pub fn learning_rate_ref(&self) -> &LearningRate {
    &self.learning_rate
  }

  /// Momentum coefficient `µ`.
  #[inline(always)]
  pub fn momentum(&self) -> f32 {
    self.momentum
  }

  /// Weight decay (L2 penalty).
  #[inline(always)]
  pub fn weight_decay(&self) -> f32 {
    self.weight_decay
  }

  /// Dampening `τ`.
  #[inline(always)]
  pub fn dampening(&self) -> f32 {
    self.dampening
  }

  /// Whether Nesterov momentum is enabled.
  #[inline(always)]
  pub fn nesterov(&self) -> bool {
    self.nesterov
  }

  /// Set the learning rate. Returns `Ok(self)` on success or `Err` if the
  /// resolved value at the current step is non-finite (NaN/Inf would scale
  /// updates into garbage).
  pub fn with_learning_rate(mut self, learning_rate: impl Into<LearningRate>) -> Result<Self> {
    let lr = learning_rate.into();
    let current_lr = lr.try_current(self.step_count)?;
    self.learning_rate = lr;
    self.current_lr = current_lr;
    self.lr_resolved_for_step = Some(self.step_count);
    Ok(self)
  }

  /// Set momentum `µ`. Returns `Ok(self)` on success or `Err` if the
  /// momentum is non-finite (NaN/Inf would corrupt velocity at the first
  /// `apply_gradients` regardless of the Nesterov branch) or if the
  /// resulting state violates the Nesterov invariant.
  pub fn with_momentum(mut self, momentum: f32) -> Result<Self> {
    Self::validate_momentum_finite(momentum)?;
    Self::validate_nesterov(momentum, self.dampening, self.nesterov)?;
    self.momentum = momentum;
    Ok(self)
  }

  /// Set weight decay. Returns `Ok(self)` on success or `Err` if
  /// `weight_decay` is not finite or `< 0.0`.
  pub fn with_weight_decay(mut self, weight_decay: f32) -> Result<Self> {
    Self::validate_weight_decay(weight_decay)?;
    self.weight_decay = weight_decay;
    Ok(self)
  }

  /// Set dampening `τ`. Returns `Ok(self)` on success or `Err` if `dampening`
  /// is not finite, `< 0.0`, or the resulting state violates the Nesterov
  /// invariant.
  pub fn with_dampening(mut self, dampening: f32) -> Result<Self> {
    Self::validate_dampening(dampening)?;
    Self::validate_nesterov(self.momentum, dampening, self.nesterov)?;
    self.dampening = dampening;
    Ok(self)
  }

  /// Set Nesterov flag. Returns `Ok(self)` on success or `Err` if the
  /// resulting state violates the Nesterov invariant.
  pub fn with_nesterov(mut self, nesterov: bool) -> Result<Self> {
    Self::validate_nesterov(self.momentum, self.dampening, nesterov)?;
    self.nesterov = nesterov;
    Ok(self)
  }
}

impl Optimizer for SGD {
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
    // Resolve scheduled LR at the PRE-increment step via skip-if-fresh cache.
    // preflight() is a no-op if MultiOptimizer already preflighted this step.
    // (matches Python `mlx.optimizers.Optimizer.apply_gradients` which
    // updates `state[scheduled_param] = scheduler(self.step)` BEFORE
    // `self.state["step"] = self.step + 1` — `optimizers.py:102..=106`).
    self.preflight()?;
    self.step_count += 1;
    let lr = self.current_lr;

    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      // ── Optional weight decay: g = g + weight_decay * w ──
      let effective_grad = if self.weight_decay != 0.0 {
        let wd = Array::full::<f32>(&[0i32; 0], self.weight_decay)?;
        let decay_term = arithmetic::multiply(&wd, param)?;
        arithmetic::add(grad, &decay_term)?
      } else {
        grad.try_clone()?
      };

      // ── Vanilla SGD branch (no momentum) ──
      if self.momentum <= 0.0 {
        let lr_scalar = Array::full::<f32>(&[0i32; 0], lr)?;
        let step = arithmetic::multiply(&lr_scalar, &effective_grad)?;
        let new_w = arithmetic::subtract(param, &step)?;
        params.insert(key.clone(), new_w);
        continue;
      }

      // ── Momentum / Nesterov branch ──
      let prev_v = match self.state.get(key) {
        Some(v) => v.try_clone()?,
        None => zeros_like(param)?,
      };
      let mu_scalar = Array::full::<f32>(&[0i32; 0], self.momentum)?;
      let v_scaled = arithmetic::multiply(&mu_scalar, &prev_v)?;
      let v_new = if self.dampening > 0.0 {
        let one_minus_damp = Array::full::<f32>(&[0i32; 0], 1.0 - self.dampening)?;
        let g_damped = arithmetic::multiply(&one_minus_damp, &effective_grad)?;
        arithmetic::add(&v_scaled, &g_damped)?
      } else {
        arithmetic::add(&v_scaled, &effective_grad)?
      };

      let update = if self.nesterov {
        // update = g + momentum * v
        let mu_v = arithmetic::multiply(&mu_scalar, &v_new)?;
        arithmetic::add(&effective_grad, &mu_v)?
      } else {
        v_new.try_clone()?
      };

      let lr_scalar = Array::full::<f32>(&[0i32; 0], lr)?;
      let step = arithmetic::multiply(&lr_scalar, &update)?;
      let new_w = arithmetic::subtract(param, &step)?;
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

  fn scalar(v: f32) -> Result<Array> {
    Array::full::<f32>(&[0i32; 0], v)
  }

  fn read_scalar(a: &Array) -> Result<f32> {
    let mut clone = a.try_clone()?;
    clone.item::<f32>()
  }

  #[test]
  fn vanilla_sgd_single_step_matches_python_ref() -> Result<()> {
    // Python: w_new = w - lr * g; w=1.0, g=0.5, lr=0.1 → w_new = 0.95.
    let mut sgd = SGD::vanilla(0.1)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.5)?);
    sgd.apply_gradients(&grads, &mut params)?;
    let got = read_scalar(&params["w"])?;
    assert!((got - 0.95).abs() < 1e-6, "expected 0.95, got {got}");
    assert_eq!(sgd.step(), 1);
    assert!((sgd.learning_rate() - 0.1).abs() < 1e-6);
    Ok(())
  }

  #[test]
  fn sgd_with_momentum_single_step_matches_python_ref() -> Result<()> {
    // Python (first step, v=0): v = 0 + g; w_new = w - lr * v.
    // w=1.0, g=0.5, lr=0.1, momentum=0.9 → v=0.5, w_new=0.95.
    let mut sgd = SGD::new(0.1, 0.9, 0.0, 0.0, false)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.5)?);
    sgd.apply_gradients(&grads, &mut params)?;
    assert!((read_scalar(&params["w"])? - 0.95).abs() < 1e-6);
    let v = read_scalar(&sgd.state["w"])?;
    assert!((v - 0.5).abs() < 1e-6, "expected v=0.5, got {v}");
    Ok(())
  }

  #[test]
  fn sgd_with_weight_decay_matches_python_ref() -> Result<()> {
    // Python: g_eff = g + wd*w; w_new = w - lr * g_eff (vanilla; momentum=0).
    // w=2.0, g=1.0, lr=0.1, wd=0.5 → g_eff=2.0, w_new=1.8.
    let mut sgd = SGD::new(0.1, 0.0, 0.5, 0.0, false)?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(2.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(1.0)?);
    sgd.apply_gradients(&grads, &mut params)?;
    assert!((read_scalar(&params["w"])? - 1.8).abs() < 1e-6);
    Ok(())
  }

  #[test]
  fn sgd_nesterov_precondition_rejects_zero_momentum() {
    // `SGD` does not implement `Debug`, so `expect_err` is unavailable; match
    // the `Result` directly to surface the typed payload.
    match SGD::new(0.1, 0.0, 0.0, 0.0, true) {
      Err(Error::InvariantViolation(payload)) => {
        assert_eq!(payload.context(), "SGD: Nesterov momentum");
        assert_eq!(
          payload.requirement(),
          "requires momentum > 0 (finite) and dampening == 0"
        );
      }
      Err(other) => panic!("expected InvariantViolation, got: {other:?}"),
      Ok(_) => panic!("nesterov with momentum=0 must be rejected"),
    }
  }

  #[test]
  fn sgd_builder_with_nesterov_rejects_zero_momentum() {
    // vanilla has momentum=0; enabling nesterov should fail
    let res = SGD::vanilla(0.1).and_then(|s| s.with_nesterov(true));
    assert!(res.is_err());
  }

  #[test]
  fn sgd_builder_with_momentum_zero_rejects_nesterov() {
    // Start with valid nesterov=true (momentum > 0), then set momentum to 0
    let res = SGD::new(0.1, 0.9, 0.0, 0.0, true).and_then(|s| s.with_momentum(0.0));
    assert!(res.is_err());
  }

  #[test]
  fn sgd_builder_with_dampening_rejects_nesterov() {
    // Start with valid nesterov=true, then add dampening
    let res = SGD::new(0.1, 0.9, 0.0, 0.0, true).and_then(|s| s.with_dampening(0.1));
    assert!(res.is_err());
  }

  #[test]
  fn sgd_schedule_advances_lr_each_step() -> Result<()> {
    // Schedule: lr(step) = 0.1 / max(step, 1). The Python
    // `apply_gradients` resolves scheduled parameters at the PRE-increment
    // step (`optimizers.py:102..=106`), so the first call sees step 0
    // (lr = 0.1 / max(0,1) = 0.1), the second call sees step 1 (also
    // 0.1), and the third call sees step 2 (lr = 0.05).
    let sched: Box<dyn Fn(usize) -> f32> = Box::new(|step| 0.1 / (step as f32).max(1.0));
    let mut sgd = SGD::vanilla(LearningRate::Schedule(sched))?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(1.0)?);
    sgd.apply_gradients(&grads, &mut params)?;
    assert!((sgd.learning_rate() - 0.1).abs() < 1e-6);
    sgd.apply_gradients(&grads, &mut params)?;
    // step==1 → lr(1) = 0.1
    assert!((sgd.learning_rate() - 0.1).abs() < 1e-6);
    sgd.apply_gradients(&grads, &mut params)?;
    // step==2 → lr(2) = 0.05
    assert!((sgd.learning_rate() - 0.05).abs() < 1e-6);
    Ok(())
  }

  #[test]
  fn optimizer_lr_schedule_resolves_at_pre_increment_step() -> Result<()> {
    // Identity-of-step schedule: returns the step value as the LR. Verifies
    // each apply_gradients resolves at the PRE-increment counter (Python
    // semantics: scheduler is called with `self.step` BEFORE the
    // increment, `optimizers.py:102..=106`).
    let sched: Box<dyn Fn(usize) -> f32> = Box::new(|step| step as f32);
    let mut sgd = SGD::vanilla(LearningRate::Schedule(sched))?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.0)?); // zero grads keep params unchanged
    sgd.apply_gradients(&grads, &mut params)?;
    assert!(
      (sgd.learning_rate() - 0.0).abs() < 1e-6,
      "first call must see step 0, got {}",
      sgd.learning_rate()
    );
    sgd.apply_gradients(&grads, &mut params)?;
    assert!(
      (sgd.learning_rate() - 1.0).abs() < 1e-6,
      "second call must see step 1, got {}",
      sgd.learning_rate()
    );
    sgd.apply_gradients(&grads, &mut params)?;
    assert!(
      (sgd.learning_rate() - 2.0).abs() < 1e-6,
      "third call must see step 2, got {}",
      sgd.learning_rate()
    );
    // step counter must still post-increment to N after N calls.
    assert_eq!(sgd.step(), 3);
    Ok(())
  }

  #[test]
  fn sgd_new_rejects_nan_weight_decay() {
    assert!(SGD::new(0.1, 0.0, f32::NAN, 0.0, false).is_err());
  }

  #[test]
  fn sgd_new_rejects_inf_weight_decay() {
    assert!(SGD::new(0.1, 0.0, f32::INFINITY, 0.0, false).is_err());
  }

  #[test]
  fn sgd_new_rejects_negative_weight_decay() {
    assert!(SGD::new(0.1, 0.0, -0.1, 0.0, false).is_err());
  }

  #[test]
  fn sgd_with_weight_decay_rejects_nan() {
    let res = SGD::vanilla(0.1).and_then(|s| s.with_weight_decay(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn sgd_with_weight_decay_rejects_inf() {
    let res = SGD::vanilla(0.1).and_then(|s| s.with_weight_decay(f32::INFINITY));
    assert!(res.is_err());
  }

  #[test]
  fn sgd_with_weight_decay_rejects_negative() {
    let res = SGD::vanilla(0.1).and_then(|s| s.with_weight_decay(-0.1));
    assert!(res.is_err());
  }

  #[test]
  fn sgd_new_rejects_nan_dampening() {
    assert!(SGD::new(0.1, 0.0, 0.0, f32::NAN, false).is_err());
  }

  #[test]
  fn sgd_new_rejects_inf_dampening() {
    assert!(SGD::new(0.1, 0.0, 0.0, f32::INFINITY, false).is_err());
  }

  #[test]
  fn sgd_new_rejects_negative_dampening() {
    assert!(SGD::new(0.1, 0.0, 0.0, -0.1, false).is_err());
  }

  #[test]
  fn sgd_with_dampening_rejects_nan() {
    let res = SGD::vanilla(0.1).and_then(|s| s.with_dampening(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn sgd_with_dampening_rejects_inf() {
    let res = SGD::vanilla(0.1).and_then(|s| s.with_dampening(f32::INFINITY));
    assert!(res.is_err());
  }

  #[test]
  fn sgd_with_dampening_rejects_negative() {
    let res = SGD::vanilla(0.1).and_then(|s| s.with_dampening(-0.1));
    assert!(res.is_err());
  }

  #[test]
  fn sgd_validate_nesterov_rejects_nan_momentum() {
    // `validate_nesterov` must reject `f32::NAN` momentum under the
    // Nesterov branch — `momentum <= 0` is false for NaN under IEEE-754,
    // so `!is_finite() || momentum <= 0.0` is the correct guard.
    assert!(SGD::new(0.1, f32::NAN, 0.0, 0.0, true).is_err());
    // The corresponding builder path must also reject.
    let with_path = SGD::new(0.1, 0.9, 0.0, 0.0, false)
      .unwrap()
      .with_momentum(f32::NAN);
    assert!(with_path.is_err());
  }

  #[test]
  fn sgd_rejects_nan_momentum_even_without_nesterov() {
    // Non-finite momentum must be rejected unconditionally — without this
    // guard `SGD::new(..., NaN, ..., false)` would succeed and the first
    // `apply_gradients` would propagate NaN into velocity (Nesterov-path
    // -agnostic).
    assert!(SGD::new(0.1, f32::NAN, 0.0, 0.0, false).is_err());
    assert!(SGD::new(0.1, f32::INFINITY, 0.0, 0.0, false).is_err());
    // Builder path: vanilla SGD has momentum=0; toggling to NaN must reject.
    let with_path = SGD::vanilla(0.1).unwrap().with_momentum(f32::NAN);
    assert!(with_path.is_err());
  }

  #[test]
  fn sgd_with_learning_rate_rejects_fixed_nan() {
    let res = SGD::vanilla(0.1).and_then(|s| s.with_learning_rate(LearningRate::Fixed(f32::NAN)));
    assert!(res.is_err(), "with_learning_rate must reject Fixed(NaN)");
  }

  #[test]
  fn sgd_apply_gradients_rejects_schedule_that_goes_nan() -> Result<()> {
    // Step 0 → 0.1 (finite, new() accepts). Step 1 → NaN (must error
    // before any param mutation).
    let sched: Box<dyn Fn(usize) -> f32> = Box::new(|step| if step == 0 { 0.1 } else { f32::NAN });
    let mut sgd = SGD::vanilla(LearningRate::Schedule(sched))?;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("w".into(), scalar(0.5)?);
    // First call (step 0, lr=0.1) must succeed.
    sgd.apply_gradients(&grads, &mut params)?;
    // Second call (step 1, lr=NaN) must error before mutating params.
    let w_before = {
      let mut c = params["w"].try_clone()?;
      c.item::<f32>()?
    };
    let result = sgd.apply_gradients(&grads, &mut params);
    assert!(
      result.is_err(),
      "apply_gradients must reject schedule NaN at step 1"
    );
    // Params must not have been mutated.
    let w_after = {
      let mut c = params["w"].try_clone()?;
      c.item::<f32>()?
    };
    assert!(
      (w_before - w_after).abs() < 1e-9,
      "params must not be mutated when LR goes NaN: before={w_before}, after={w_after}"
    );
    Ok(())
  }
}
