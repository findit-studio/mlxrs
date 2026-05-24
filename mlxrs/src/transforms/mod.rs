//! Function transforms: autograd (`value_and_grad`, `grad`, `vjp`, `jvp`),
//! custom-VJP overrides, gradient checkpointing, and bulk eval / async-eval.
//!
//! Mirrors `mlx-swift`'s `MLX.Transforms` (`Transforms.swift`,
//! `Transforms+Eval.swift`, `Transforms+Grad.swift`, `Transforms+Internal.swift`)
//! and `mlx.core.{value_and_grad,grad,vjp,jvp,custom_function,custom_vjp,
//! checkpoint,eval,async_eval}` on the Python side.
//!
//! ## API surface (autograd + eval chunk)
//!
//! - [`crate::transforms::closure::Closure`] — RAII wrapper over
//!   `mlx_closure` (foundation; landed in the previous chunk).
//! - [`crate::transforms::autograd::value_and_grad`] /
//!   [`crate::transforms::autograd::grad`] — return a Rust closure that, when
//!   invoked on a slice of [`crate::Array`], runs the forward pass and
//!   computes gradients with respect to a chosen subset of inputs.
//! - [`crate::transforms::autograd::vjp`] /
//!   [`crate::transforms::autograd::jvp`] — one-shot vector-Jacobian and
//!   Jacobian-vector products over a user function evaluated at `primals`.
//! - [`crate::transforms::eval::eval`] /
//!   [`crate::transforms::eval::async_eval`] — synchronously / asynchronously
//!   materialize the lazy graph rooted at a batch of arrays.
//!
//! Custom-VJP and checkpoint land in the subsequent chunks.

pub mod autograd;
pub mod closure;
pub mod eval;

pub use autograd::{grad, jvp, value_and_grad, vjp};
pub use closure::Closure;
pub use eval::{async_eval, eval};
