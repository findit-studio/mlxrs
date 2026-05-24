//! Function transforms: autograd (`value_and_grad`, `grad`, `vjp`, `jvp`),
//! custom-VJP overrides, gradient checkpointing, and bulk eval / async-eval.
//!
//! Mirrors `mlx-swift`'s `MLX.Transforms` (`Transforms.swift`,
//! `Transforms+Eval.swift`, `Transforms+Grad.swift`, `Transforms+Internal.swift`)
//! and `mlx.core.{value_and_grad,grad,vjp,jvp,custom_function,custom_vjp,
//! checkpoint,eval,async_eval}` on the Python side.
//!
//! ## API surface
//!
//! - [`crate::transforms::closure::Closure`] — RAII wrapper over
//!   `mlx_closure` that owns the captured Rust callable for the FFI's
//!   lifetime. Used internally by the autograd builders; exposed in case a
//!   caller needs to build a closure directly.
//! - [`crate::transforms::autograd::value_and_grad`] /
//!   [`crate::transforms::autograd::grad`] — return a Rust closure that, when
//!   invoked on a slice of [`crate::Array`], runs the forward pass and
//!   computes gradients with respect to a chosen subset of inputs. The
//!   returned closure is `Fn`-callable repeatedly with different inputs.
//! - [`crate::transforms::autograd::vjp`] /
//!   [`crate::transforms::autograd::jvp`] — one-shot vector-Jacobian and
//!   Jacobian-vector products over a user function evaluated at `primals`.
//! - [`crate::transforms::custom::custom_vjp`] /
//!   [`crate::transforms::custom::custom_function`] — wrap a forward function
//!   with a user-defined backward (cotangent) function, overriding the
//!   autograd-derived VJP.
//! - [`crate::transforms::checkpoint::checkpoint`] — wrap a function so its
//!   activations are recomputed (rather than stored) during the backward
//!   pass, trading compute for memory.
//! - [`crate::transforms::eval::eval`] / [`crate::transforms::eval::async_eval`]
//!   — synchronously / asynchronously materialize the lazy graph rooted at a
//!   batch of arrays.
//!
//! ## Threading
//!
//! Like the rest of mlxrs, `Closure` and the returned `impl Fn` callables are
//! `!Send` + `!Sync` (they own [`crate::Array`] handles transitively through
//! the trampoline's closure, and mlx's evaluator is single-threaded — see
//! `crate::array::Array` for the rationale). The Rust callable passed in
//! (`F: Fn(&[Array]) -> Result<Vec<Array>>`) is still required `+ 'static` so
//! it can outlive the construction scope and be invoked from mlx-c.

pub mod autograd;
pub mod checkpoint;
pub mod closure;
pub mod custom;
pub mod eval;

pub use autograd::{grad, jvp, value_and_grad, vjp};
pub use checkpoint::checkpoint;
pub use closure::Closure;
pub use custom::{custom_function, custom_vjp};
pub use eval::{async_eval, eval};
