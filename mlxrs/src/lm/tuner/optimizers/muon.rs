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

use crate::{
  Array, Result,
  lm::{
    load::Weights,
    tuner::optimizers::base::{LearningRate, Optimizer, zeros_like, zeros_like_map},
  },
  ops::{arithmetic, linalg_basic::addmm, linalg_full::norm_l2, shape::reshape},
};

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

/// Muon optimizer.
pub struct Muon {
  /// Learning rate `λ`.
  pub learning_rate: LearningRate,
  /// Momentum strength. Default Python: `0.95`.
  pub momentum: f32,
  /// Weight decay (L2 penalty). Default Python: `0.01`.
  pub weight_decay: f32,
  /// Enable Nesterov momentum. Default Python: `true`.
  pub nesterov: bool,
  /// Newton-Schulz iteration steps. Default Python: `5`.
  pub ns_steps: usize,
  step_count: usize,
  current_lr: f32,
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
    let lr = learning_rate.into();
    let current_lr = lr.current(0);
    Ok(Self {
      learning_rate: lr,
      momentum,
      weight_decay,
      nesterov,
      ns_steps,
      step_count: 0,
      current_lr,
      state: HashMap::new(),
    })
  }

  /// Python-default-args constructor.
  pub fn default_with_lr(learning_rate: impl Into<LearningRate>) -> Result<Self> {
    Self::new(learning_rate, 0.95, 0.01, true, 5)
  }

  /// Newton-Schulz 5 iteration on a 2D matrix. Mirrors the Python
  /// `_zeropower_via_newtonschulz5` (`optimizers.py:896..=915`).
  fn newton_schulz5(&self, x: &Array, steps: usize) -> Result<Array> {
    let shape = x.shape();
    if shape.len() != 2 {
      return Err(crate::error::Error::ShapeMismatch {
        message: format!("Muon.newton_schulz5: expected 2D input, got shape {shape:?}"),
      });
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

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    if self.state.is_empty() {
      self.init(gradients)?;
    }
    // Resolve scheduled LR at PRE-increment step, then increment
    // (matches Python `optimizers.py:102..=106`).
    self.current_lr = self.learning_rate.current(self.step_count);
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
