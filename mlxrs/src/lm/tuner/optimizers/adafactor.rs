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

use crate::{
  Array, Result,
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

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// Per-parameter state for Adafactor.
///
/// - `Factored { row, col, exp_avg }` for 2D+ tensors (with optional
///   `exp_avg` first-moment when `beta_1.is_some()`).
/// - `NonFactored { exp_avg_sq, exp_avg }` for 1D / 0D tensors.
enum AdafactorState {
  Factored {
    row: Array,
    col: Array,
    exp_avg: Option<Array>,
  },
  NonFactored {
    exp_avg_sq: Array,
    exp_avg: Option<Array>,
  },
}

/// Adafactor optimizer.
pub struct Adafactor {
  /// Learning rate `λ` (only consulted when `relative_step == false`).
  /// `None` is equivalent to Python's `learning_rate=None`.
  pub learning_rate: Option<LearningRate>,
  /// `(ε₁, ε₂)`. `ε₁` is added to the squared gradient for numerical
  /// stability; `ε₂` clamps the parameter scale.
  /// Default Python: `(1e-30, 1e-3)`.
  pub eps: (f32, f32),
  /// Clips the unscaled update at this RMS-norm. Default Python: `1.0`.
  pub clip_threshold: f32,
  /// Coefficient for the running average of the squared gradient. The
  /// effective `β₂` at step `t` is `1 - t^decay_rate`. Default Python:
  /// `-0.8`.
  pub decay_rate: f32,
  /// First-moment coefficient. `None` disables the first-moment branch.
  /// Default Python: `None`.
  pub beta_1: Option<f32>,
  /// Weight decay coefficient. Default Python: `0.0`.
  pub weight_decay: f32,
  /// If true, scale the learning rate by `max(eps₂, RMS(w))`.
  /// Default Python: `true`.
  pub scale_parameter: bool,
  /// If true, ignore `learning_rate` and compute a relative step size.
  /// Default Python: `true`.
  pub relative_step: bool,
  /// If true (with `relative_step`), compute the relative step from the
  /// current step (warmup). Default Python: `false`.
  pub warmup_init: bool,
  step_count: usize,
  current_lr: f32,
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
    let current_lr = learning_rate
      .as_ref()
      .map(|lr| lr.current(0))
      .unwrap_or(0.0);
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
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor.
  pub fn default_python() -> Result<Self> {
    Self::new(None, (1e-30, 1e-3), 1.0, -0.8, None, 0.0, true, true, false)
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
      Ok(AdafactorState::Factored { row, col, exp_avg })
    } else {
      Ok(AdafactorState::NonFactored {
        exp_avg_sq: zeros_like(param)?,
        exp_avg,
      })
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

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    if self.state.is_empty() {
      self.init(gradients)?;
    }
    // Resolve scheduled LR at PRE-increment step, then increment
    // (matches Python `optimizers.py:102..=106`). Adafactor's internal
    // `step_f` (used for `beta_2 = 1 - step^decay_rate` and for the
    // `relative_step` rsqrt branch) is read AFTER the increment to match
    // Python `step = self.step` at `optimizers.py:808` which runs AFTER
    // the base `apply_gradients` has already incremented `step`.
    if let Some(lr) = &self.learning_rate {
      self.current_lr = lr.current(self.step_count);
    }
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
        AdafactorState::Factored { row, col, exp_avg } => {
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
            AdafactorState::Factored {
              row: row_new,
              col: col_new,
              exp_avg,
            },
            update_calc,
          )
        }
        AdafactorState::NonFactored {
          exp_avg_sq,
          exp_avg,
        } => {
          // exp_avg_sq = β₂·old + (1-β₂)·update
          let old_scaled = arithmetic::multiply(&beta_2_s, &exp_avg_sq)?;
          let upd_scaled = arithmetic::multiply(&one_minus_beta_2, &update)?;
          let new_eas = arithmetic::add(&old_scaled, &upd_scaled)?;
          // update = rsqrt(new_eas) · g
          let rs = arithmetic::rsqrt(&new_eas)?;
          let update_calc = arithmetic::multiply(&rs, grad)?;
          (
            AdafactorState::NonFactored {
              exp_avg_sq: new_eas,
              exp_avg,
            },
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
        AdafactorState::Factored {
          row,
          col,
          exp_avg: Some(prev_ea),
        } if self.beta_1.is_some() => {
          let b1 = self.beta_1.unwrap();
          let b1_s = scalar(b1)?;
          let one_minus_b1 = scalar(1.0 - b1)?;
          let prev_scaled = arithmetic::multiply(&b1_s, &prev_ea)?;
          let upd_scaled = arithmetic::multiply(&one_minus_b1, &update_arr)?;
          let new_ea = arithmetic::add(&prev_scaled, &upd_scaled)?;
          update_arr = new_ea.try_clone()?;
          AdafactorState::Factored {
            row,
            col,
            exp_avg: Some(new_ea),
          }
        }
        AdafactorState::NonFactored {
          exp_avg_sq,
          exp_avg: Some(prev_ea),
        } if self.beta_1.is_some() => {
          let b1 = self.beta_1.unwrap();
          let b1_s = scalar(b1)?;
          let one_minus_b1 = scalar(1.0 - b1)?;
          let prev_scaled = arithmetic::multiply(&b1_s, &prev_ea)?;
          let upd_scaled = arithmetic::multiply(&one_minus_b1, &update_arr)?;
          let new_ea = arithmetic::add(&prev_scaled, &upd_scaled)?;
          update_arr = new_ea.try_clone()?;
          AdafactorState::NonFactored {
            exp_avg_sq,
            exp_avg: Some(new_ea),
          }
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
}
