//! [`Optimizer`] trait + [`LearningRate`] schedule wrapper.
//!
//! Ports the base interface of Python `mlx.optimizers.Optimizer`
//! (`mlx/python/mlx/optimizers/optimizers.py:10..=155`) +
//! the Swift `Optimizer` protocol +
//! `OptimizerBase` /
//! `OptimizerBaseArrayState`
//! (`mlx-swift/Source/MLXOptimizers/Optimizers.swift:1..=100`).
//!
//! Python keeps optimizer state in a nested `dict` walked in lock-step with
//! the parameter tree via `tree_map`. mlxrs flattens this to a `HashMap<String,
//! Array>` (or `HashMap<String, (Array, Array)>` for two-moment families) —
//! see the [module-level deviation note](super#trait-shape-deviation-from-python).
//!
//! ## Learning-rate schedules
//!
//! Each optimizer takes a [`LearningRate`] at construction time. This is
//! either a `LearningRate::Fixed(f32)` (Python `float`) or a
//! `LearningRate::Schedule(Box<dyn Fn(usize) -> f32>)` (Python
//! `Callable[[step], float]`) — mirroring the Python `Union[float,
//! Callable]` pattern. The optimizer queries the schedule on every
//! [`Optimizer::apply_gradients`] call via [`LearningRate::current`],
//! passing the optimizer's step counter.

use crate::{Array, Result, lm::load::Weights};

/// Learning-rate value or step-driven schedule.
///
/// Mirrors Python's `Union[float, Callable[[mx.array], mx.array]]`
/// argument shape on every optimizer's `learning_rate` parameter
/// (`optimizers.py:230..=254`, `297..=325`, etc.).
pub enum LearningRate {
  /// Fixed scalar learning rate (Python `float`).
  Fixed(f32),
  /// Step-driven schedule (Python `Callable[[step], float]`). The boxed
  /// closure is called with the optimizer's step counter each time
  /// [`Optimizer::apply_gradients`] is invoked, BEFORE the step is
  /// incremented (so the first call sees step 0, the second call sees
  /// step 1, matching Python's `optimizers.py:102..=106`).
  Schedule(Box<dyn Fn(usize) -> f32>),
}

impl LearningRate {
  /// Resolve the learning rate at `step` (0-based at the first
  /// `apply_gradients` call, matching Python's scheduled-parameter
  /// resolution at `optimizers.py:102..=103` which runs BEFORE the step
  /// counter is incremented at `optimizers.py:106`).
  pub fn current(&self, step: usize) -> f32 {
    match self {
      LearningRate::Fixed(v) => *v,
      LearningRate::Schedule(f) => f(step),
    }
  }
}

impl std::fmt::Debug for LearningRate {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      LearningRate::Fixed(v) => f.debug_tuple("Fixed").field(v).finish(),
      LearningRate::Schedule(_) => f.debug_tuple("Schedule").field(&"<closure>").finish(),
    }
  }
}

impl From<f32> for LearningRate {
  fn from(value: f32) -> Self {
    LearningRate::Fixed(value)
  }
}

/// Common interface for all gradient-descent optimizers.
///
/// Mirrors Python `mlx.optimizers.Optimizer`
/// (`mlx/python/mlx/optimizers/optimizers.py:10..=155`) +
/// the Swift `Optimizer` protocol
/// (`mlx-swift/Source/MLXOptimizers/Optimizers.swift:12..=16`).
///
/// ## Lifecycle
///
/// 1. Construct (`Type::new(...)` per optimizer).
/// 2. Optional: call [`Optimizer::init`] with the parameter tree to pre-
///    populate state (Python `optimizer.init(params)`). If skipped, the
///    first [`Optimizer::apply_gradients`] call auto-inits.
/// 3. Each training step: build `gradients` (e.g. via
///    [`crate::transforms::value_and_grad`]), call
///    [`Optimizer::apply_gradients`] with `gradients` + `params`. The
///    optimizer mutates `params` in-place with the updated weights and
///    advances its internal step counter.
pub trait Optimizer {
  /// Pre-allocate per-parameter optimizer state for every entry in
  /// `params`. Mirrors Python `Optimizer.init(parameters)`
  /// (`optimizers.py:31..=73`). Safe to call multiple times — re-init wipes
  /// existing state.
  ///
  /// Idiom: most callers SKIP this and let
  /// [`Optimizer::apply_gradients`] lazy-init on first call (matching the
  /// Python `if not self._initialized: self.init(gradients)` guard at
  /// `optimizers.py:98..=99`).
  fn init(&mut self, params: &Weights) -> Result<()>;

  /// Apply `gradients` to `params` in-place. Mirrors Python
  /// `Optimizer.apply_gradients(gradients, parameters)`
  /// (`optimizers.py:85..=109`).
  ///
  /// - Lazy-inits per-parameter state on first call (matching Python's
  ///   `if not self._initialized: self.init(gradients)` guard).
  /// - Resolves the learning-rate schedule (if any) at the PRE-increment
  ///   step (matching Python's `state[scheduled_param] = scheduler(self.step)`
  ///   at `optimizers.py:102..=103`), then increments the internal step
  ///   counter (matching Python's `self.state["step"] = self.step + 1` at
  ///   `optimizers.py:106`). Optimizers whose per-param formula uses the
  ///   POST-increment step (e.g. Adam bias correction) read `step_count`
  ///   AFTER the increment, matching Python's `step = self.step` reads in
  ///   `apply_single`.
  /// - For each parameter present in `gradients`: looks up the matching
  ///   entry in `params`, computes the updated weight, and writes it back
  ///   into `params`. Parameters NOT in `gradients` are left untouched
  ///   (Python: "gradients can be a subset of parameters").
  fn apply_gradients(&mut self, gradients: &Weights, params: &mut Weights) -> Result<()>;

  /// Current optimizer step (1-based; 0 before the first
  /// [`Optimizer::apply_gradients`] call). Mirrors Python `Optimizer.step`
  /// at `optimizers.py:131..=133`.
  fn step(&self) -> usize;

  /// Effective learning rate at the most recent step (after any schedule
  /// has been applied). Mirrors Python `Optimizer.learning_rate` at
  /// `optimizers.py:135..=141`.
  fn learning_rate(&self) -> f32;
}

/// Helper: build a `HashMap<String, Array>` of zero-filled state tensors,
/// one per param entry, with the same shape and dtype as each parameter.
///
/// Mirrors the Python `init_single` recipes that all do
/// `state["v"] = mx.zeros_like(parameter)`. Centralized so each optimizer's
/// `init` stays a one-liner.
pub(crate) fn zeros_like_map(params: &Weights) -> Result<std::collections::HashMap<String, Array>> {
  let mut out = std::collections::HashMap::with_capacity(params.len());
  for (key, value) in params {
    out.insert(key.clone(), zeros_like(value)?);
  }
  Ok(out)
}

/// Build a fresh zero-filled `Array` with the same shape and dtype as
/// `template`. Re-export of [`crate::ops::misc::zeros_like`].
pub(crate) fn zeros_like(template: &Array) -> Result<Array> {
  crate::ops::misc::zeros_like(template)
}
