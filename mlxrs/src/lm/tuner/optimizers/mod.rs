//! Optimizers ported from [`mlx/python/mlx/optimizers/optimizers.py`] +
//! [`mlx-swift/Source/MLXOptimizers/Optimizers.swift`].
//!
//! Ten gradient-descent optimizer families, each implementing the common
//! [`Optimizer`] trait:
//!
//! - [`SGD`] — stochastic gradient descent + (Nesterov) momentum + weight
//!   decay + dampening.
//! - [`RMSprop`] — running-average-of-squared-gradients normalization.
//! - [`Adagrad`] — cumulative-squared-gradients normalization.
//! - [`AdaDelta`] — `(u/v)` running ratio with no global learning rate.
//! - [`Adam`] / [`AdamW`] / [`Adamax`] — bias-corrected adaptive moments
//!   family, with [`AdamW`] adding decoupled weight decay and [`Adamax`]
//!   using the `∞`-norm denominator.
//! - [`Lion`] — sign-of-momentum update (smaller compute / memory than Adam).
//! - [`Adafactor`] — sublinear-memory adaptive moments (row+col running
//!   averages instead of full per-element `v`).
//! - [`Muon`] — momentum + Newton-Schulz orthogonalization on 2D+ updates.
//!
//! Plus [`MultiOptimizer`] for routing different parameter groups to
//! different optimizer instances, and the [`schedulers`] sub-module for
//! step-driven learning-rate schedules ([`schedulers::cosine_decay`],
//! [`schedulers::exponential_decay`], [`schedulers::step_decay`],
//! [`schedulers::linear_schedule`], [`schedulers::join_schedules`]).
//!
//! ## Trait shape (deviation from Python)
//!
//! Python keeps state in a nested `dict` keyed by the parameter tree path
//! (`tree_map(apply_single, gradients, parameters, state)`). The Rust port
//! flattens this to a `HashMap<String, ...>` — each optimizer owns its own
//! per-parameter state keyed by the parameter's *flat name* (the same flat
//! string keys [`crate::lm::load::Weights`] uses, e.g.
//! `"model.layers.0.self_attn.q_proj.weight"`). Reasons:
//!
//! - mlxrs's [`crate::lm::load::Weights`] is already a flat `HashMap<String,
//!   Array>` (mirroring the safetensors / GGUF on-disk format), and the
//!   training loop hands the optimizer a [`Weights`]-shaped tree of
//!   gradients + parameters. The flat shape is the natural Rust idiom.
//! - The Python `tree_map` walks the per-parameter `state` dict in lock-step
//!   with the parameter tree; a flat `HashMap` keyed by the same flat path
//!   is the structural equivalent, just spelled differently.
//! - This follows the Rust-idiomatic API shape: ndarray-flavored
//!   ergonomics over verbatim Python/Swift mirroring.
//!
//! ## Scope cuts
//!
//! - **Distributed training** (`mx.distributed.AllReduce` / `Group.barrier`)
//!   is out of scope for v1; single-process training only. Can be added
//!   later via the already-bound but unwrapped `mlxrs_sys::mlx_distributed_*`
//!   symbols.
//! - **MultiOptimizer** ships the trait + a minimal predicate-routing impl;
//!   the full per-parameter-tree Python complexity (`tree_merge`,
//!   `_split_dictionary` with `tree_flatten`/`tree_unflatten` round-trip)
//!   collapses naturally to flat-map filtering.
//! - **TensorBoard / W&B integrations** are out of scope; callers add their
//!   own progress callback (see [`super::trainer::TrainingCallback`]).
//!
//! [`mlx/python/mlx/optimizers/optimizers.py`]: https://github.com/ml-explore/mlx/blob/main/python/mlx/optimizers/optimizers.py
//! [`mlx-swift/Source/MLXOptimizers/Optimizers.swift`]: https://github.com/ml-explore/mlx-swift/blob/main/Source/MLXOptimizers/Optimizers.swift
//! [`Weights`]: crate::lm::load::Weights

pub mod adadelta;
pub mod adafactor;
pub mod adagrad;
pub mod adam;
pub mod base;
pub mod clip;
pub mod lion;
pub mod multi;
pub mod muon;
pub mod rmsprop;
pub mod schedulers;
pub mod sgd;

pub use adadelta::AdaDelta;
pub use adafactor::Adafactor;
pub use adagrad::Adagrad;
pub use adam::{Adam, AdamW, Adamax};
pub use base::{LearningRate, Optimizer};
pub use clip::clip_grad_norm;
pub use lion::Lion;
pub use multi::MultiOptimizer;
pub use muon::Muon;
pub use rmsprop::RMSprop;
pub use schedulers::{
  Schedule, cosine_decay, exponential_decay, join_schedules, linear_schedule, step_decay,
};
pub use sgd::SGD;
