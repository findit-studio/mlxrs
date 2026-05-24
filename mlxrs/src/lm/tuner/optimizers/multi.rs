//! [`MultiOptimizer`] — route different parameter groups to different
//! optimizer instances via a list of name-predicate filters.
//!
//! Mirrors Python `mlx.optimizers.MultiOptimizer`
//! (`mlx/python/mlx/optimizers/optimizers.py:157..=227`).
//!
//! Construction: `MultiOptimizer::new(optimizers, filters)` where
//! `optimizers.len() == filters.len() + 1` (the last optimizer is the
//! fallback — any parameter name not matched by an earlier filter is
//! routed there). Each filter is a `Fn(&str, &Array) -> bool` predicate;
//! it sees the parameter's flat name plus a reference to the gradient
//! array for that parameter.
//!
//! Per-parameter routing collapses naturally to flat-`HashMap` filtering
//! since [`crate::lm::load::Weights`] is already flat. The Python
//! `_split_dictionary` (which round-trips through
//! `tree_flatten`/`tree_unflatten`) becomes a simple `for (key, grad) in
//! gradients` walk that pushes the entry into the first matching
//! sub-optimizer's gradient slice.

use std::collections::HashMap;

use crate::{
  Array, Result,
  error::Error,
  lm::{load::Weights, tuner::optimizers::base::Optimizer},
};

/// Name predicate type: `(flat_name, gradient_ref) -> bool`.
pub type FilterFn = Box<dyn Fn(&str, &Array) -> bool>;

/// Predicate-routed composite optimizer.
pub struct MultiOptimizer {
  /// Sub-optimizers, in priority order. The last one is the implicit
  /// fallback (no predicate; receives any parameter not claimed earlier).
  optimizers: Vec<Box<dyn Optimizer>>,
  /// Filters paired with each optimizer EXCEPT the last (the fallback
  /// has no filter — Python: `filters + [lambda *a, **k: True]`).
  filters: Vec<FilterFn>,
  step_count: usize,
  current_lr: f32,
}

impl MultiOptimizer {
  /// Construct a [`MultiOptimizer`]. `filters.len()` MUST equal
  /// `optimizers.len() - 1` (the last optimizer is the fallback).
  pub fn new(optimizers: Vec<Box<dyn Optimizer>>, filters: Vec<FilterFn>) -> Result<Self> {
    if optimizers.is_empty() {
      return Err(Error::Backend {
        message: "MultiOptimizer: at least one optimizer is required".into(),
      });
    }
    if filters.len() != optimizers.len() - 1 {
      return Err(Error::Backend {
        message: format!(
          "MultiOptimizer: expected {} filters (optimizers - 1), got {}",
          optimizers.len() - 1,
          filters.len(),
        ),
      });
    }
    let current_lr = optimizers[0].learning_rate();
    Ok(Self {
      optimizers,
      filters,
      step_count: 0,
      current_lr,
    })
  }

  /// Split a [`Weights`] map into per-optimizer slices, routing each entry
  /// to the FIRST optimizer whose filter accepts it (or the fallback last
  /// optimizer if none do). Mirrors Python `_split_dictionary`
  /// (`optimizers.py:184..=196`).
  fn split_dictionary(&self, weights: &Weights) -> Result<Vec<Weights>> {
    let mut parts: Vec<Weights> = (0..self.optimizers.len()).map(|_| HashMap::new()).collect();
    for (key, value) in weights {
      let mut placed = false;
      for (i, filter) in self.filters.iter().enumerate() {
        if filter(key, value) {
          parts[i].insert(key.clone(), value.try_clone()?);
          placed = true;
          break;
        }
      }
      if !placed {
        // Fallback (last optimizer).
        let last = parts.len() - 1;
        parts[last].insert(key.clone(), value.try_clone()?);
      }
    }
    Ok(parts)
  }
}

impl Optimizer for MultiOptimizer {
  fn init(&mut self, params: &Weights) -> Result<()> {
    let split = self.split_dictionary(params)?;
    for (opt, p) in self.optimizers.iter_mut().zip(split) {
      opt.init(&p)?;
    }
    Ok(())
  }

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    let grad_split = self.split_dictionary(gradients)?;
    // For each sub-optimizer slice, build a matching param slice (only
    // the keys present in this slice) and call its apply_gradients.
    for (opt, gs) in self.optimizers.iter_mut().zip(grad_split.iter()) {
      let mut ps: Weights = HashMap::with_capacity(gs.len());
      for key in gs.keys() {
        if let Some(v) = params.get(key) {
          ps.insert(key.clone(), v.try_clone()?);
        }
      }
      opt.apply_gradients(gs, &mut ps)?;
      // Merge the updated params back.
      for (k, v) in ps {
        params.insert(k, v);
      }
    }
    self.step_count += 1;
    self.current_lr = self.optimizers[0].learning_rate();
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
  use crate::lm::tuner::optimizers::sgd::SGD;

  fn scalar(v: f32) -> Result<Array> {
    Array::full::<f32>(&[0i32; 0], v)
  }

  fn read_scalar(a: &Array) -> Result<f32> {
    let mut clone = a.try_clone()?;
    clone.item::<f32>()
  }

  #[test]
  fn multi_routes_by_filter_to_distinct_sgd_lrs() -> Result<()> {
    // Two SGDs at different LRs. Names starting with "bias." → 1e-3,
    // anything else → 1e-1 (fallback).
    let bias_sgd: Box<dyn Optimizer> = Box::new(SGD::vanilla(1e-3)?);
    let weight_sgd: Box<dyn Optimizer> = Box::new(SGD::vanilla(1e-1)?);
    let bias_filter: FilterFn = Box::new(|name, _| name.starts_with("bias."));
    let mut multi = MultiOptimizer::new(vec![bias_sgd, weight_sgd], vec![bias_filter])?;
    let mut params: Weights = HashMap::new();
    params.insert("bias.0".into(), scalar(1.0)?);
    params.insert("layer.weight".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("bias.0".into(), scalar(1.0)?);
    grads.insert("layer.weight".into(), scalar(1.0)?);
    multi.apply_gradients(&grads, &mut params)?;
    // bias.0: 1 - 1e-3 ≈ 0.999
    // layer.weight: 1 - 1e-1 = 0.9
    assert!((read_scalar(&params["bias.0"])? - 0.999).abs() < 1e-6);
    assert!((read_scalar(&params["layer.weight"])? - 0.9).abs() < 1e-6);
    Ok(())
  }

  #[test]
  fn multi_rejects_wrong_filter_count() {
    let res = MultiOptimizer::new(
      vec![
        Box::new(SGD::vanilla(0.1).unwrap()),
        Box::new(SGD::vanilla(0.1).unwrap()),
      ],
      vec![], // need exactly 1 filter for 2 optimizers
    );
    assert!(res.is_err());
  }
}
