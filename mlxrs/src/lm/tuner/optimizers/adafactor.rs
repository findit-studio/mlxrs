//! [`Adafactor`] — sublinear-memory adaptive moments (Shazeer & Stern,
//! 2018, <https://arxiv.org/abs/1804.04235>).
//!
//! Mirrors Python `mlx.optimizers.Adafactor`
//! (`mlx/python/mlx/optimizers/optimizers.py:708..=848`).
//!
//! For 2D+ tensors, Adafactor factors the running squared-gradient state
//! into per-row and per-column running averages (`exp_avg_sq_row`,
//! `exp_avg_sq_col`) instead of a full per-element `v` — that's the key
//! memory win (factored: `O(M+N)` state for an `M×N` tensor, vs the
//! `O(M·N)` Adam pays). For 1D tensors and scalars, falls back to a
//! standard `exp_avg_sq` running mean.
//!
//! ## Scope cut: scalar (0D) parameters
//!
//! Python's `mx.mean(update, axis=-1)` over an empty axis tuple errors;
//! the upstream Adafactor implicitly assumes parameters are 1D+. We
//! mirror this — a 0D parameter is treated as 1D (the upstream `factored
//! = parameter.ndim >= 2` branch falls through to the non-factored
//! `exp_avg_sq` path, which `mx.zeros_like` over a 0D array would also
//! handle natively in Python).

use std::collections::HashMap;

use smol_str::format_smolstr;

use crate::{
  Array, Result,
  error::{Error, InvariantViolationPayload, NonFiniteScalarPayload, OutOfRangePayload},
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like},
  },
  ops::{
    arithmetic,
    reduction::{mean, mean_axes},
    shape::expand_dims_axes,
  },
};

/// Validate `clip_threshold` is finite and `> 0.0` (used as a divisor).
fn validate_clip_threshold(clip_threshold: f32) -> Result<()> {
  if !clip_threshold.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Adafactor: clip_threshold",
      clip_threshold as f64,
    )));
  }
  if clip_threshold <= 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Adafactor: clip_threshold",
      "must be > 0.0 (used as a divisor)",
      format_smolstr!("{clip_threshold}"),
    )));
  }
  Ok(())
}

/// Validate `decay_rate` is finite AND `<= 0.0` — the Python contract is that
/// `β₂(step) = 1 - step^decay_rate` must stay in `[0, 1)` for every step.
/// With `decay_rate > 0` the exponent grows with step, so `β₂` goes negative
/// after the first iteration, and the squared-gradient running average
/// becomes negative; the later `rsqrt` then produces NaN weights. The
/// Python default is `-0.8`; values `<= 0` keep `β₂` monotonically
/// approaching 1 from below.
fn validate_decay_rate(decay_rate: f32) -> Result<()> {
  if !decay_rate.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Adafactor: decay_rate",
      decay_rate as f64,
    )));
  }
  if decay_rate > 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Adafactor: decay_rate",
      "must be <= 0.0 (so that 1 - step^decay_rate stays in [0, 1))",
      format_smolstr!("{decay_rate}"),
    )));
  }
  Ok(())
}

/// Validate `eps = (ε₁, ε₂)` — both must be finite and `>= 0.0`.
fn validate_eps(eps: (f32, f32)) -> Result<()> {
  let (e1, e2) = eps;
  if !e1.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Adafactor: eps.0",
      e1 as f64,
    )));
  }
  if !e2.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Adafactor: eps.1",
      e2 as f64,
    )));
  }
  if e1 < 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Adafactor: eps.0",
      "must be >= 0.0",
      format_smolstr!("{e1}"),
    )));
  }
  if e2 < 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Adafactor: eps.1",
      "must be >= 0.0",
      format_smolstr!("{e2}"),
    )));
  }
  Ok(())
}

/// Validate `beta_1` is `None` or finite and in `[0.0, 1.0)`.
fn validate_beta_1(beta_1: Option<f32>) -> Result<()> {
  if let Some(b) = beta_1 {
    if !b.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "Adafactor: beta_1",
        b as f64,
      )));
    }
    if !(0.0..1.0).contains(&b) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Adafactor: beta_1",
        "must be None or in [0.0, 1.0)",
        format_smolstr!("{b}"),
      )));
    }
  }
  Ok(())
}

/// Validate `weight_decay` is finite and `>= 0.0`.
fn validate_weight_decay(weight_decay: f32) -> Result<()> {
  if !weight_decay.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "Adafactor: weight_decay",
      weight_decay as f64,
    )));
  }
  if weight_decay < 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Adafactor: weight_decay",
      "must be >= 0.0",
      format_smolstr!("{weight_decay}"),
    )));
  }
  Ok(())
}

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// Payload for the [`AdafactorState::Factored`] variant (2D+ tensors).
struct FactoredState {
  row: Array,
  col: Array,
  exp_avg: Option<Array>,
}

impl FactoredState {
  fn new(row: Array, col: Array, exp_avg: Option<Array>) -> Self {
    Self { row, col, exp_avg }
  }
}

/// Payload for the [`AdafactorState::NonFactored`] variant (1D / 0D tensors).
struct NonFactoredState {
  exp_avg_sq: Array,
  exp_avg: Option<Array>,
}

impl NonFactoredState {
  fn new(exp_avg_sq: Array, exp_avg: Option<Array>) -> Self {
    Self {
      exp_avg_sq,
      exp_avg,
    }
  }
}

/// Per-parameter state for Adafactor.
///
/// - [`AdafactorState::Factored`] for 2D+ tensors (row/col factored running
///   averages, with optional `exp_avg` first-moment when `beta_1.is_some()`).
/// - [`AdafactorState::NonFactored`] for 1D / 0D tensors (full `exp_avg_sq`).
enum AdafactorState {
  /// 2D+ tensors — factored running squared-gradient state.
  Factored(FactoredState),
  /// 1D / 0D tensors — full per-element running squared-gradient state.
  NonFactored(NonFactoredState),
}

/// Adafactor optimizer.
pub struct Adafactor {
  /// Learning rate `λ` (only consulted when `relative_step == false`).
  /// `None` is equivalent to Python's `learning_rate=None`.
  learning_rate: Option<LearningRate>,
  /// `(ε₁, ε₂)`. `ε₁` is added to the squared gradient for numerical
  /// stability; `ε₂` clamps the parameter scale.
  /// Default Python: `(1e-30, 1e-3)`.
  eps: (f32, f32),
  /// Clips the unscaled update at this RMS-norm. Default Python: `1.0`.
  clip_threshold: f32,
  /// Coefficient for the running average of the squared gradient. The
  /// effective `β₂` at step `t` is `1 - t^decay_rate`. Default Python:
  /// `-0.8`.
  decay_rate: f32,
  /// First-moment coefficient. `None` disables the first-moment branch.
  /// Default Python: `None`.
  beta_1: Option<f32>,
  /// Weight decay coefficient. Default Python: `0.0`.
  weight_decay: f32,
  /// If true, scale the learning rate by `max(eps₂, RMS(w))`.
  /// Default Python: `true`.
  scale_parameter: bool,
  /// If true, ignore `learning_rate` and compute a relative step size.
  /// Default Python: `true`.
  relative_step: bool,
  /// If true (with `relative_step`), compute the relative step from the
  /// current step (warmup). Default Python: `false`.
  warmup_init: bool,
  step_count: usize,
  current_lr: f32,
  /// Skip-if-fresh stamp — `Some(N)` means `current_lr` is valid for step N.
  lr_resolved_for_step: Option<usize>,
  state: HashMap<String, AdafactorState>,
}

impl Adafactor {
  /// Construct an [`Adafactor`] optimizer.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    learning_rate: Option<LearningRate>,
    eps: (f32, f32),
    clip_threshold: f32,
    decay_rate: f32,
    beta_1: Option<f32>,
    weight_decay: f32,
    scale_parameter: bool,
    relative_step: bool,
    warmup_init: bool,
  ) -> Result<Self> {
    validate_eps(eps)?;
    validate_clip_threshold(clip_threshold)?;
    validate_decay_rate(decay_rate)?;
    validate_beta_1(beta_1)?;
    validate_weight_decay(weight_decay)?;
    let current_lr = match learning_rate.as_ref() {
      Some(lr) => lr.try_current(0)?,
      None => 0.0,
    };
    Ok(Self {
      learning_rate,
      eps,
      clip_threshold,
      decay_rate,
      beta_1,
      weight_decay,
      scale_parameter,
      relative_step,
      warmup_init,
      step_count: 0,
      current_lr,
      lr_resolved_for_step: None,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor.
  pub fn default_python() -> Result<Self> {
    Self::new(None, (1e-30, 1e-3), 1.0, -0.8, None, 0.0, true, true, false)
  }

  /// The learning rate (or schedule), if set.
  #[inline(always)]
  pub fn learning_rate_ref(&self) -> Option<&LearningRate> {
    self.learning_rate.as_ref()
  }

  /// `(ε₁, ε₂)` tuple.
  #[inline(always)]
  pub fn eps(&self) -> (f32, f32) {
    self.eps
  }

  /// Clip threshold.
  #[inline(always)]
  pub fn clip_threshold(&self) -> f32 {
    self.clip_threshold
  }

  /// Decay rate for the squared-gradient running average.
  #[inline(always)]
  pub fn decay_rate(&self) -> f32 {
    self.decay_rate
  }

  /// First-moment coefficient, if set.
  #[inline(always)]
  pub fn beta_1(&self) -> Option<f32> {
    self.beta_1
  }

  /// Weight decay coefficient.
  #[inline(always)]
  pub fn weight_decay(&self) -> f32 {
    self.weight_decay
  }

  /// Whether the learning rate is scaled by the parameter RMS.
  #[inline(always)]
  pub fn scale_parameter(&self) -> bool {
    self.scale_parameter
  }

  /// Whether relative-step mode is enabled.
  #[inline(always)]
  pub fn relative_step(&self) -> bool {
    self.relative_step
  }

  /// Whether warmup-init mode is enabled.
  #[inline(always)]
  pub fn warmup_init(&self) -> bool {
    self.warmup_init
  }

  /// Set the learning rate. Returns `Ok(self)` on success or `Err` if the
  /// resolved value at the current step is non-finite.
  pub fn with_learning_rate(mut self, learning_rate: Option<LearningRate>) -> Result<Self> {
    let current_lr = match learning_rate.as_ref() {
      Some(lr) => lr.try_current(self.step_count)?,
      None => 0.0,
    };
    self.learning_rate = learning_rate;
    self.current_lr = current_lr;
    self.lr_resolved_for_step = Some(self.step_count);
    Ok(self)
  }

  /// Set eps `(ε₁, ε₂)`. Returns `Ok(self)` on success or `Err` if either
  /// component is not finite or `< 0.0`.
  pub fn with_eps(mut self, eps: (f32, f32)) -> Result<Self> {
    validate_eps(eps)?;
    self.eps = eps;
    Ok(self)
  }

  /// Set clip threshold. Returns `Ok(self)` on success or `Err` if
  /// `clip_threshold` is not finite or `<= 0.0` (it is used as a divisor).
  pub fn with_clip_threshold(mut self, clip_threshold: f32) -> Result<Self> {
    validate_clip_threshold(clip_threshold)?;
    self.clip_threshold = clip_threshold;
    Ok(self)
  }

  /// Set decay rate. Returns `Ok(self)` on success or `Err` if `decay_rate`
  /// is not finite (NaN/Inf in `step.powf(decay_rate)` produces a NaN β₂).
  pub fn with_decay_rate(mut self, decay_rate: f32) -> Result<Self> {
    validate_decay_rate(decay_rate)?;
    self.decay_rate = decay_rate;
    Ok(self)
  }

  /// Set beta_1. Returns `Result<Self>` because `beta_1` controls whether
  /// per-parameter first-moment state (`exp_avg`) is allocated by the
  /// private `init_state_for` helper; toggling it AFTER any parameters
  /// have been initialized would silently desynchronize existing-vs-new
  /// parameters (existing parameters keep their original `exp_avg`
  /// `None`/`Some` shape; new ones get the new shape). Rejected post-init
  /// to preserve state-shape consistency; construct a fresh Adafactor
  /// instead if you need to change `beta_1` mid-training.
  ///
  /// **Note:** this consuming form is intended for pre-init builder
  /// chaining. After parameters have been initialized via
  /// `apply_gradients`, use the non-consuming [`Self::try_set_beta_1`]
  /// instead so a failed validation does NOT drop the populated optimizer
  /// state.
  pub fn with_beta_1(mut self, beta_1: Option<f32>) -> Result<Self> {
    if !self.state.is_empty() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Adafactor::with_beta_1",
        "cannot toggle beta_1 after parameter state is initialized (would desynchronize existing \
         vs new parameters' exp_avg shape); construct a fresh Adafactor or use try_set_beta_1 \
         (which preserves state on error)",
      )));
    }
    validate_beta_1(beta_1)?;
    self.beta_1 = beta_1;
    Ok(self)
  }

  /// Non-consuming `beta_1` setter — preserves the optimizer's populated
  /// per-parameter state on validation error (the consuming [`Self::with_beta_1`]
  /// would drop it). Same post-init rejection rule: returns `Err` when
  /// `state` is non-empty, leaving `self` unchanged on either branch.
  pub fn try_set_beta_1(&mut self, beta_1: Option<f32>) -> Result<()> {
    if !self.state.is_empty() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Adafactor::try_set_beta_1",
        "cannot toggle beta_1 after parameter state is initialized (would desynchronize existing \
         vs new parameters' exp_avg shape); construct a fresh Adafactor instead",
      )));
    }
    validate_beta_1(beta_1)?;
    self.beta_1 = beta_1;
    Ok(())
  }

  /// Set weight decay. Returns `Ok(self)` on success or `Err` if
  /// `weight_decay` is not finite or `< 0.0`.
  pub fn with_weight_decay(mut self, weight_decay: f32) -> Result<Self> {
    validate_weight_decay(weight_decay)?;
    self.weight_decay = weight_decay;
    Ok(self)
  }

  /// Set scale_parameter. Returns `self` for chaining.
  #[must_use]
  pub fn with_scale_parameter(mut self, scale_parameter: bool) -> Self {
    self.scale_parameter = scale_parameter;
    self
  }

  /// Set relative_step. Returns `self` for chaining.
  #[must_use]
  pub fn with_relative_step(mut self, relative_step: bool) -> Self {
    self.relative_step = relative_step;
    self
  }

  /// Set warmup_init. Returns `self` for chaining.
  #[must_use]
  pub fn with_warmup_init(mut self, warmup_init: bool) -> Self {
    self.warmup_init = warmup_init;
    self
  }

  fn init_state_for(&self, param: &Array) -> Result<AdafactorState> {
    let shape = param.shape();
    let exp_avg = if self.beta_1.is_some() {
      Some(zeros_like(param)?)
    } else {
      None
    };
    if param.ndim() >= 2 {
      // row shape = shape[..-1]; col shape = shape[..-2] + shape[-1:]
      let row_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
      let mut col_shape: Vec<usize> = shape[..shape.len() - 2].to_vec();
      col_shape.push(shape[shape.len() - 1]);
      // Mirror the parameter's dtype by allocating a same-shape sub-tensor
      // then casting to the row/col target shape via reshape. Cheaper:
      // build a 1.0-filled tensor of the right shape (Array::full uses
      // f32 internally, then cast to param dtype) — sufficient for state
      // init where we only need zero values.
      let dtype = param.dtype()?;
      let row = Array::full::<f32>(&row_shape.as_slice(), 0.0)?.astype(dtype)?;
      let col = Array::full::<f32>(&col_shape.as_slice(), 0.0)?.astype(dtype)?;
      Ok(AdafactorState::Factored(FactoredState::new(
        row, col, exp_avg,
      )))
    } else {
      Ok(AdafactorState::NonFactored(NonFactoredState::new(
        zeros_like(param)?,
        exp_avg,
      )))
    }
  }

  fn compute_rms(&self, a: &Array) -> Result<Array> {
    let sq = arithmetic::square(a)?;
    let m = mean(&sq, false)?;
    arithmetic::sqrt(&m)
  }

  fn compute_learning_rate(&self, parameter_rms: &Array) -> Result<Array> {
    let step = self.step_count as f32;
    let relative_step = if self.relative_step {
      let min_step = if self.warmup_init { 1e-6 * step } else { 1e-2 };
      let rsqrt_step = step.sqrt().recip();
      min_step.min(rsqrt_step)
    } else {
      self.current_lr
    };
    let rel_s = scalar(relative_step)?;
    if self.scale_parameter {
      let eps2_s = scalar(self.eps.1)?;
      let param_scale = arithmetic::maximum(&eps2_s, parameter_rms)?;
      arithmetic::multiply(&param_scale, &rel_s)
    } else {
      Ok(rel_s)
    }
  }
}

impl Optimizer for Adafactor {
  fn init(&mut self, params: &Weights) -> Result<()> {
    let mut out = HashMap::with_capacity(params.len());
    for (key, value) in params {
      out.insert(key.clone(), self.init_state_for(value)?);
    }
    self.state = out;
    Ok(())
  }

  fn preflight(&mut self) -> Result<()> {
    if self.lr_resolved_for_step == Some(self.step_count) {
      return Ok(()); // cache hit: schedule already consulted at this step
    }
    self.current_lr = match self.learning_rate.as_ref() {
      Some(lr) => lr.try_current(self.step_count)?,
      None => 0.0,
    };
    self.lr_resolved_for_step = Some(self.step_count);
    Ok(())
  }

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    if self.state.is_empty() {
      self.init(gradients)?;
    }
    // Resolve scheduled LR via skip-if-fresh cache (no-op if MultiOptimizer
    // already preflighted this step). Adafactor's internal `step_f` (used for
    // `beta_2 = 1 - step^decay_rate` and for the `relative_step` rsqrt
    // branch) is read AFTER the increment to match Python `step = self.step`
    // at `optimizers.py:808` which runs AFTER the base `apply_gradients` has
    // already incremented `step`.
    self.preflight()?;
    self.step_count += 1;
    let step_f = self.step_count as f32;
    // β₂ at this step: 1 - step^decay_rate. We compute as a scalar tensor.
    let beta_2_val = 1.0 - step_f.powf(self.decay_rate);
    let beta_2_s = scalar(beta_2_val)?;
    let one_minus_beta_2 = scalar(1.0 - beta_2_val)?;
    let eps0 = scalar(self.eps.0)?;
    let one = scalar(1.0)?;
    let clip = scalar(self.clip_threshold)?;
    for (key, grad) in gradients {
      let Some(param) = params.get(key) else {
        continue;
      };
      let parameter_rms = self.compute_rms(param)?;
      let learning_rate = self.compute_learning_rate(&parameter_rms)?;
      // update = g² + ε₁
      let g_sq = arithmetic::square(grad)?;
      let update = arithmetic::add(&g_sq, &eps0)?;
      let st = self
        .state
        .remove(key)
        .unwrap_or(self.init_state_for(param)?);
      let (new_state, mut update_arr) = match st {
        AdafactorState::Factored(fs) => {
          let row = fs.row;
          let col = fs.col;
          let exp_avg = fs.exp_avg;
          let ndim = grad.ndim();
          let row_axis = (ndim - 1) as i32;
          let col_axis = (ndim - 2) as i32;
          // exp_avg_sq_row = β₂·row + (1-β₂)·mean(update, axis=-1)
          let upd_row_mean = mean_axes(&update, &[row_axis], false)?;
          let row_scaled = arithmetic::multiply(&beta_2_s, &row)?;
          let row_term = arithmetic::multiply(&one_minus_beta_2, &upd_row_mean)?;
          let row_new = arithmetic::add(&row_scaled, &row_term)?;
          // exp_avg_sq_col = β₂·col + (1-β₂)·mean(update, axis=-2)
          let upd_col_mean = mean_axes(&update, &[col_axis], false)?;
          let col_scaled = arithmetic::multiply(&beta_2_s, &col)?;
          let col_term = arithmetic::multiply(&one_minus_beta_2, &upd_col_mean)?;
          let col_new = arithmetic::add(&col_scaled, &col_term)?;
          // approximate exp_moving_avg via row/col — Python uses axis=-1 on
          // the row tensor (which has ndim-1 dims), not on the gradient.
          let row_inner_axis = (row_new.ndim() as i32) - 1;
          let row_mean = mean_axes(&row_new, &[row_inner_axis], true)?;
          let row_norm = arithmetic::divide(&row_new, &row_mean)?;
          let r_factor = arithmetic::rsqrt(&row_norm)?;
          let c_factor = arithmetic::rsqrt(&col_new)?;
          // r_factor: shape[..-1] → expand to shape[..-1] + (1,)
          // c_factor: shape[..-2] + shape[-1:] → expand to
          //   shape[..-2] + (1,) + shape[-1:]
          // Then matmul gives the outer product back to shape[..-1] + shape[-1:].
          let r_expanded = expand_dims_axes(&r_factor, &[-1])?;
          // c_factor needs an axis inserted at position ndim-2 (so the new
          // axis is the second-to-last). Python: mx.expand_dims(c_factor,
          // axis=0) on a 1D tensor (for 2D inputs) → (1, N). Generalized:
          // axis = ndim - 2.
          let c_expand_at = (ndim as i32) - 2;
          let c_expanded = expand_dims_axes(&c_factor, &[c_expand_at])?;
          let approx = crate::ops::linalg_basic::matmul(&r_expanded, &c_expanded)?;
          let update_calc = arithmetic::multiply(&approx, grad)?;
          (
            AdafactorState::Factored(FactoredState::new(row_new, col_new, exp_avg)),
            update_calc,
          )
        }
        AdafactorState::NonFactored(nfs) => {
          let exp_avg_sq = nfs.exp_avg_sq;
          let exp_avg = nfs.exp_avg;
          // exp_avg_sq = β₂·old + (1-β₂)·update
          let old_scaled = arithmetic::multiply(&beta_2_s, &exp_avg_sq)?;
          let upd_scaled = arithmetic::multiply(&one_minus_beta_2, &update)?;
          let new_eas = arithmetic::add(&old_scaled, &upd_scaled)?;
          // update = rsqrt(new_eas) · g
          let rs = arithmetic::rsqrt(&new_eas)?;
          let update_calc = arithmetic::multiply(&rs, grad)?;
          (
            AdafactorState::NonFactored(NonFactoredState::new(new_eas, exp_avg)),
            update_calc,
          )
        }
      };
      // clip: update = update / max(1, RMS(update) / clip_threshold)
      let upd_rms = self.compute_rms(&update_arr)?;
      let rms_over_clip = arithmetic::divide(&upd_rms, &clip)?;
      let denom = arithmetic::maximum(&one, &rms_over_clip)?;
      update_arr = arithmetic::divide(&update_arr, &denom)?;
      // update = lr · update
      update_arr = arithmetic::multiply(&learning_rate, &update_arr)?;
      // β₁ first moment
      let final_state = match new_state {
        AdafactorState::Factored(fs) if self.beta_1.is_some() && fs.exp_avg.is_some() => {
          let row = fs.row;
          let col = fs.col;
          let prev_ea = fs.exp_avg.unwrap();
          let b1 = self.beta_1.unwrap();
          let b1_s = scalar(b1)?;
          let one_minus_b1 = scalar(1.0 - b1)?;
          let prev_scaled = arithmetic::multiply(&b1_s, &prev_ea)?;
          let upd_scaled = arithmetic::multiply(&one_minus_b1, &update_arr)?;
          let new_ea = arithmetic::add(&prev_scaled, &upd_scaled)?;
          update_arr = new_ea.try_clone()?;
          AdafactorState::Factored(FactoredState::new(row, col, Some(new_ea)))
        }
        AdafactorState::NonFactored(nfs) if self.beta_1.is_some() && nfs.exp_avg.is_some() => {
          let exp_avg_sq = nfs.exp_avg_sq;
          let prev_ea = nfs.exp_avg.unwrap();
          let b1 = self.beta_1.unwrap();
          let b1_s = scalar(b1)?;
          let one_minus_b1 = scalar(1.0 - b1)?;
          let prev_scaled = arithmetic::multiply(&b1_s, &prev_ea)?;
          let upd_scaled = arithmetic::multiply(&one_minus_b1, &update_arr)?;
          let new_ea = arithmetic::add(&prev_scaled, &upd_scaled)?;
          update_arr = new_ea.try_clone()?;
          AdafactorState::NonFactored(NonFactoredState::new(exp_avg_sq, Some(new_ea)))
        }
        other => other,
      };
      // weight decay: w += w·(-wd·lr) === w·(1 - wd·lr) but Python adds
      //   parameter += parameter * (-weight_decay * learning_rate)
      // then w_new = parameter - update
      let param_after_decay = if self.weight_decay != 0.0 {
        let neg_wd_lr_s = arithmetic::multiply(&scalar(-self.weight_decay)?, &learning_rate)?;
        let extra = arithmetic::multiply(param, &neg_wd_lr_s)?;
        arithmetic::add(param, &extra)?
      } else {
        param.try_clone()?
      };
      let new_w = arithmetic::subtract(&param_after_decay, &update_arr)?;
      params.insert(key.clone(), new_w);
      self.state.insert(key.clone(), final_state);
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
  fn adafactor_new_rejects_negative_eps() {
    // eps.0 < 0
    assert!(
      Adafactor::new(
        None,
        (-1e-30, 1e-3),
        1.0,
        -0.8,
        None,
        0.0,
        true,
        true,
        false
      )
      .is_err()
    );
    // eps.1 < 0
    assert!(
      Adafactor::new(
        None,
        (1e-30, -1e-3),
        1.0,
        -0.8,
        None,
        0.0,
        true,
        true,
        false
      )
      .is_err()
    );
  }

  #[test]
  fn adafactor_new_rejects_non_finite_eps() {
    assert!(
      Adafactor::new(
        None,
        (f32::NAN, 1e-3),
        1.0,
        -0.8,
        None,
        0.0,
        true,
        true,
        false
      )
      .is_err()
    );
    assert!(
      Adafactor::new(
        None,
        (1e-30, f32::INFINITY),
        1.0,
        -0.8,
        None,
        0.0,
        true,
        true,
        false
      )
      .is_err()
    );
  }

  #[test]
  fn adafactor_new_rejects_non_positive_clip_threshold() {
    assert!(Adafactor::new(None, (1e-30, 1e-3), 0.0, -0.8, None, 0.0, true, true, false).is_err());
    assert!(
      Adafactor::new(
        None,
        (1e-30, 1e-3),
        -1.0,
        -0.8,
        None,
        0.0,
        true,
        true,
        false
      )
      .is_err()
    );
  }

  #[test]
  fn adafactor_new_rejects_non_finite_clip_threshold() {
    assert!(
      Adafactor::new(
        None,
        (1e-30, 1e-3),
        f32::NAN,
        -0.8,
        None,
        0.0,
        true,
        true,
        false
      )
      .is_err()
    );
  }

  #[test]
  fn adafactor_new_rejects_non_finite_decay_rate() {
    assert!(
      Adafactor::new(
        None,
        (1e-30, 1e-3),
        1.0,
        f32::NAN,
        None,
        0.0,
        true,
        true,
        false
      )
      .is_err()
    );
    assert!(
      Adafactor::new(
        None,
        (1e-30, 1e-3),
        1.0,
        f32::INFINITY,
        None,
        0.0,
        true,
        true,
        false
      )
      .is_err()
    );
  }

  #[test]
  fn adafactor_new_rejects_negative_weight_decay() {
    assert!(
      Adafactor::new(
        None,
        (1e-30, 1e-3),
        1.0,
        -0.8,
        None,
        -0.1,
        true,
        true,
        false
      )
      .is_err()
    );
  }

  #[test]
  fn adafactor_new_rejects_non_finite_weight_decay() {
    assert!(
      Adafactor::new(
        None,
        (1e-30, 1e-3),
        1.0,
        -0.8,
        None,
        f32::NAN,
        true,
        true,
        false
      )
      .is_err()
    );
  }

  #[test]
  fn adafactor_with_eps_rejects_negative() {
    let res = Adafactor::default_python().and_then(|a| a.with_eps((-1e-30, 1e-3)));
    assert!(res.is_err());
  }

  #[test]
  fn adafactor_with_eps_rejects_non_finite() {
    let res = Adafactor::default_python().and_then(|a| a.with_eps((f32::NAN, 1e-3)));
    assert!(res.is_err());
  }

  #[test]
  fn adafactor_with_clip_threshold_rejects_non_positive() {
    let res = Adafactor::default_python().and_then(|a| a.with_clip_threshold(0.0));
    assert!(res.is_err());
    let res2 = Adafactor::default_python().and_then(|a| a.with_clip_threshold(-1.0));
    assert!(res2.is_err());
  }

  #[test]
  fn adafactor_with_clip_threshold_rejects_non_finite() {
    let res = Adafactor::default_python().and_then(|a| a.with_clip_threshold(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn adafactor_with_decay_rate_rejects_non_finite() {
    let res = Adafactor::default_python().and_then(|a| a.with_decay_rate(f32::NAN));
    assert!(res.is_err());
    let res2 = Adafactor::default_python().and_then(|a| a.with_decay_rate(f32::INFINITY));
    assert!(res2.is_err());
  }

  #[test]
  fn adafactor_with_weight_decay_rejects_negative() {
    let res = Adafactor::default_python().and_then(|a| a.with_weight_decay(-0.1));
    assert!(res.is_err());
  }

  #[test]
  fn adafactor_with_weight_decay_rejects_non_finite() {
    let res = Adafactor::default_python().and_then(|a| a.with_weight_decay(f32::NAN));
    assert!(res.is_err());
  }

  #[test]
  fn adafactor_1d_param_runs_one_step_without_error() -> Result<()> {
    // 1D tensors take the NonFactored branch. Verify the step completes
    // and produces a different weight (no Python ref number — Adafactor's
    // relative-step + RMS clip + lr scaling chain is not easily
    // reduced to a closed-form scalar in two lines).
    let mut adafactor = Adafactor::default_python()?;
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
    adafactor.apply_gradients(&grads, &mut params)?;
    let mut got = params["w"].try_clone()?;
    let v: Vec<f32> = got.to_vec()?;
    // Step must move (Adafactor's relative step is non-zero by default).
    assert!(
      (v[0] - 1.0).abs() > 1e-8,
      "expected w[0] to move, got {}",
      v[0]
    );
    Ok(())
  }

  #[test]
  fn adafactor_2d_param_runs_one_step_without_error() -> Result<()> {
    // 2D tensors take the Factored branch.
    let mut adafactor = Adafactor::default_python()?;
    let mut params: Weights = HashMap::new();
    params.insert(
      "w".into(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2))?,
    );
    let mut grads: Weights = HashMap::new();
    grads.insert(
      "w".into(),
      Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &(2, 2))?,
    );
    adafactor.apply_gradients(&grads, &mut params)?;
    let mut got = params["w"].try_clone()?;
    let _: Vec<f32> = got.to_vec()?;
    Ok(())
  }

  #[test]
  fn adafactor_new_rejects_nan_beta_1() {
    assert!(
      Adafactor::new(
        None,
        (1e-30, 1e-3),
        1.0,
        -0.8,
        Some(f32::NAN),
        0.0,
        true,
        true,
        false
      )
      .is_err()
    );
  }

  #[test]
  fn adafactor_with_beta_1_rejects_nan_some() {
    let res = Adafactor::default_python().and_then(|a| a.with_beta_1(Some(f32::NAN)));
    assert!(res.is_err());
  }

  #[test]
  fn adafactor_with_beta_1_rejects_above_one_some() {
    // beta_1 >= 1.0 makes the EMA non-contractive.
    let res = Adafactor::default_python().and_then(|a| a.with_beta_1(Some(1.0)));
    assert!(res.is_err());
    let res2 = Adafactor::default_python().and_then(|a| a.with_beta_1(Some(1.5)));
    assert!(res2.is_err());
  }

  #[test]
  fn adafactor_with_beta_1_accepts_none() -> Result<()> {
    // None is always valid (disables first-moment branch).
    let _a = Adafactor::default_python()?.with_beta_1(None)?;
    Ok(())
  }

  #[test]
  fn adafactor_try_set_beta_1_rejects_nan_pre_init() {
    let mut adafactor = Adafactor::default_python().unwrap();
    // Pre-init: validation should still fire on NaN.
    assert!(adafactor.try_set_beta_1(Some(f32::NAN)).is_err());
  }

  #[test]
  fn adafactor_with_beta_1_rejects_post_init() -> Result<()> {
    // `with_beta_1` controls per-parameter exp_avg state shape; toggling
    // it after any parameter has been initialized would silently produce
    // existing-vs-new parameter shape mismatch. Must error post-init.
    let adafactor = Adafactor::default_python()?;
    let mut params = HashMap::from([(
      "w".to_string(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2))?,
    )]);
    let grads = HashMap::from([(
      "w".to_string(),
      Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &(2, 2))?,
    )]);
    let mut adafactor = adafactor;
    adafactor.apply_gradients(&grads, &mut params)?;
    // state is now non-empty — toggling beta_1 must error.
    assert!(adafactor.with_beta_1(Some(0.9)).is_err());
    Ok(())
  }

  #[test]
  fn adafactor_with_learning_rate_rejects_fixed_nan() {
    let res = Adafactor::default_python()
      .and_then(|a| a.with_learning_rate(Some(LearningRate::Fixed(f32::NAN))));
    assert!(
      res.is_err(),
      "with_learning_rate must reject Some(Fixed(NaN))"
    );
  }

  #[test]
  fn adafactor_try_set_beta_1_preserves_state_on_error() -> Result<()> {
    // Non-consuming setter must reject post-init AND leave the optimizer
    // usable (state preserved). Caller still has the populated `adafactor`
    // after the failed setter call.
    let mut adafactor = Adafactor::default_python()?;
    let original_beta_1 = adafactor.beta_1;
    let mut params = HashMap::from([(
      "w".to_string(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2))?,
    )]);
    let grads = HashMap::from([(
      "w".to_string(),
      Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &(2, 2))?,
    )]);
    adafactor.apply_gradients(&grads, &mut params)?;
    // try_set_beta_1 errors post-init…
    assert!(adafactor.try_set_beta_1(Some(0.9)).is_err());
    // …and beta_1 + state are untouched, so training can continue.
    assert_eq!(adafactor.beta_1, original_beta_1);
    assert!(!adafactor.state.is_empty(), "state preserved on error");
    // Prove the optimizer still works after the rejected setter call.
    adafactor.apply_gradients(&grads, &mut params)?;
    Ok(())
  }
}
