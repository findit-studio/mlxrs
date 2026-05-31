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
  error::{EmptyInputPayload, Error, LengthMismatchPayload},
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
      return Err(Error::EmptyInput(EmptyInputPayload::new(
        "MultiOptimizer: optimizers",
      )));
    }
    if filters.len() != optimizers.len() - 1 {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "MultiOptimizer: filters (must equal optimizers - 1)",
        optimizers.len() - 1,
        filters.len(),
      )));
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

  /// Recursive preflight: validates EVERY descendant optimizer (including
  /// nested `MultiOptimizer` children) at the current step before any
  /// param mutation (#244): without this override, an outer
  /// `MultiOptimizer` would call the trait-default no-op for a nested
  /// `MultiOptimizer` child, then commit earlier siblings before the
  /// nested child's own internal preflight (inside its `apply_gradients`)
  /// could reject. The override walks the tree so the atomicity gate
  /// reaches every leaf.
  fn preflight(&mut self) -> Result<()> {
    for optimizer in &mut self.optimizers {
      optimizer.preflight()?;
    }
    Ok(())
  }

  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()> {
    // LR-ATOMICITY GUARANTEE (#244): recursive preflight on every
    // descendant BEFORE mutating any params or child state. Scope is
    // **LR validation only**: if any descendant's schedule resolves to a
    // non-finite value at the current step, the whole `apply_gradients`
    // returns `Err` with no observable side effect — a retry after
    // fixing the bad LR remains idempotent. The preflight also resolves
    // and CACHES the LR with a step stamp so a stateful schedule closure
    // is called AT MOST ONCE per step.
    //
    // **NOT atomic for non-LR apply failures** (explicit scope cut):
    // if a later child's `apply_gradients` fails for a non-LR reason
    // (shape/dtype mismatch, FFI error, etc.) AFTER earlier children
    // have already committed their param + state updates, the call
    // returns `Err` but earlier children remain advanced. Wrapping the
    // call in a full two-phase commit (stage every child's update into
    // a side buffer, swap on all-success) would close this but pays a
    // per-step state-clone cost; we defer that until a real workload
    // demands it. Callers that need apply-failure atomicity should
    // snapshot params + optimizer state before the call.
    //
    // Routes through `self.preflight()` so nested `MultiOptimizer`
    // children get their own recursive preflight.
    self.preflight()?;
    let grad_split = self.split_dictionary(gradients)?;
    // Apply each child in order. The preflight gate above ensures none of
    // these will fail on a runtime LR validation; per-step optimizer math
    // failures (e.g. shape mismatch) still bubble Err but those aren't
    // atomicity-violating in the same way.
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

  #[test]
  fn multi_optimizer_atomicity_on_mid_run_nan_schedule() -> Result<()> {
    // Regression test for issue #244: MultiOptimizer must preflight ALL child
    // optimizers BEFORE mutating any params. If child-1's schedule goes NaN
    // at step 1, child-0's step_count must NOT advance on the second call.
    //
    // child-0: fixed LR=0.1, handles key "x" (via filter)
    // child-1 (fallback): schedule returning 0.1 at step 0 and NaN at step>=1,
    //   handles key "y"
    use crate::lm::tuner::optimizers::base::LearningRate;

    let child0: Box<dyn Optimizer> = Box::new(SGD::vanilla(0.1)?);
    let bad_schedule =
      LearningRate::Schedule(Box::new(|step| if step == 0 { 0.1_f32 } else { f32::NAN }));
    let child1: Box<dyn Optimizer> = Box::new(SGD::vanilla(bad_schedule)?);
    // child0 handles "x"; child1 (fallback) handles "y"
    let x_filter: FilterFn = Box::new(|name, _| name == "x");
    let mut multi = MultiOptimizer::new(vec![child0, child1], vec![x_filter])?;

    let mut params: Weights = HashMap::new();
    params.insert("x".into(), scalar(1.0)?);
    params.insert("y".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("x".into(), scalar(1.0)?);
    grads.insert("y".into(), scalar(1.0)?);

    // First apply_gradients (step 0 for both children): must succeed.
    multi.apply_gradients(&grads, &mut params)?;
    assert_eq!(
      multi.step(),
      1,
      "multi step_count must be 1 after first apply"
    );

    // Second apply_gradients: child1's LR is NaN at step 1 — preflight must
    // reject BEFORE child0's apply runs. child0's step_count must stay at 1.
    let err = multi.apply_gradients(&grads, &mut params);
    assert!(
      err.is_err(),
      "second apply_gradients must err when child1 schedule goes NaN"
    );
    // The MultiOptimizer's own step_count should also still be 1 (no commit).
    assert_eq!(
      multi.step(),
      1,
      "MultiOptimizer step_count must not advance when preflight rejects"
    );
    // params must be unchanged from after the first (successful) step.
    // x: 1.0 - 0.1*1.0 = 0.9 (from first step); must not have changed.
    let x_val = read_scalar(&params["x"])?;
    assert!(
      (x_val - 0.9).abs() < 1e-6,
      "x param must not be mutated by the rejected second apply (got {x_val})"
    );
    Ok(())
  }

  #[test]
  fn multi_optimizer_atomicity_holds_for_stateful_schedule() -> Result<()> {
    // A stateful schedule with interior mutability: returns 0.1 on the FIRST
    // call at each step, then f32::NAN on every subsequent call at that same
    // step. Without the resolve-once-cache, MultiOptimizer's preflight would
    // see 0.1 (finite, passes the gate) but each child's apply_gradients would
    // then call the schedule again and see NaN — breaking atomicity by
    // half-committing child-0 before child-1 errors. The skip-if-fresh stamp
    // in preflight guarantees the schedule is called ONCE per step and the
    // cached value is used for the commit.
    use crate::lm::tuner::optimizers::base::LearningRate;
    use std::{cell::Cell, rc::Rc};

    let call_count = Rc::new(Cell::new(0u32));
    let bad_schedule = LearningRate::Schedule(Box::new(move |_step| {
      let n = call_count.get();
      call_count.set(n + 1);
      // First call at any step: finite. Second+ call: NaN.
      if n.is_multiple_of(2) {
        0.1_f32
      } else {
        f32::NAN
      }
    }));

    let child0: Box<dyn Optimizer> = Box::new(SGD::vanilla(0.1)?);
    let child1: Box<dyn Optimizer> = Box::new(SGD::vanilla(bad_schedule)?);
    // child0 handles "x" (via filter); child1 (fallback) handles "y".
    let x_filter: FilterFn = Box::new(|name, _| name == "x");
    let mut multi = MultiOptimizer::new(vec![child0, child1], vec![x_filter])?;

    let mut params: Weights = HashMap::new();
    params.insert("x".into(), scalar(1.0)?);
    params.insert("y".into(), scalar(1.0)?);
    let mut grads: Weights = HashMap::new();
    grads.insert("x".into(), scalar(1.0)?);
    grads.insert("y".into(), scalar(1.0)?);

    // Step 0: preflight calls the schedule once (call #0 → 0.1, finite).
    // apply_gradients reads from the cache — does NOT call the schedule again.
    // Without the cache, child1's apply_gradients would make call #1 → NaN,
    // and we'd see an error despite preflight passing.
    multi.apply_gradients(&grads, &mut params)?;
    assert_eq!(multi.step(), 1, "step must advance after successful apply");

    // Verify both params updated (schedule returned 0.1 for both children).
    // x: child0 (fixed 0.1): 1.0 - 0.1·1.0 = 0.9
    // y: child1 (cached 0.1): 1.0 - 0.1·1.0 = 0.9
    let x_val = read_scalar(&params["x"])?;
    let y_val = read_scalar(&params["y"])?;
    assert!((x_val - 0.9).abs() < 1e-6, "x should be 0.9, got {x_val}");
    assert!((y_val - 0.9).abs() < 1e-6, "y should be 0.9, got {y_val}");

    Ok(())
  }
}
