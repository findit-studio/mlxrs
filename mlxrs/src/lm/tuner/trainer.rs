//! Training-loop orchestration ported from mlx-lm
//! [`tuner/trainer.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py).
//!
//! ## v1 status — mechanics-only [`train`]
//!
//! The public [`train`] loop wires up the optimizer step, callbacks, eval,
//! and save hooks end-to-end, but does NOT yet compute REAL per-parameter
//! gradients — it dispatches `optimizer.apply_gradients(zeros_like(params),
//! params)` because mlxrs has no `nn::Module` trait yet to bind
//! `params → loss` for [`crate::transforms::value_and_grad`]. The full
//! autograd path arrives once the [`crate::lm::model::Model`] trait grows
//! parameter binding (tracked separately on the M3 roadmap).
//!
//! Callers must explicitly opt in via
//! [`TrainingArgs::acknowledge_no_real_gradients`] = `true` before invoking
//! [`train`] in v1. The flag exists so a future production caller cannot
//! accidentally run multi-hour training jobs against the stub thinking
//! they're getting actual parameter updates — the `Err` returned otherwise
//! points the caller at this v1 limitation. Mechanics-only validation
//! (callbacks fire, optimizer state advances, save hook runs at the right
//! cadence) is the use case this v1 enables; real model fine-tuning is not.
//!
//! ## Periodic-event cadence — OPTIMIZER STEPS (deviation from Python)
//!
//! [`TrainingArgs::iters`] counts MICROBATCH iterations (matching Python
//! `mlx-lm/tuner/trainer.py`), but [`TrainingArgs::steps_per_report`],
//! [`TrainingArgs::steps_per_eval`], and [`TrainingArgs::steps_per_save`]
//! count OPTIMIZER STEPS — they fire only after a complete
//! [`TrainingArgs::grad_accumulation_steps`] window. Python counts these
//! in microbatches, which makes a caller's report frequency silently
//! depend on `grad_accumulation_steps`. mlxrs decouples them so a caller
//! bumping `grad_accumulation_steps` doesn't accidentally inflate their
//! report / eval / save frequency. Total optimizer steps the loop will
//! execute is `iters / grad_accumulation_steps` (floored — any final
//! partial window is dropped).
//!
//! ## Surface
//!
//! - [`TrainingArgs`] — config (Python `@dataclass class TrainingArgs`,
//!   `trainer.py:41..=83`).
//! - [`default_loss`] — token-level masked cross-entropy (Python
//!   `default_loss`, `trainer.py:86..=99`).
//! - [`grad_checkpoint`] — wrap a layer-forward in `mlxrs::transforms::checkpoint`
//!   (Python `grad_checkpoint`, `trainer.py:25..=38`). The Python version
//!   monkey-patches `type(layer).__call__`; the Rust version returns a
//!   wrapped closure (Rust idiom — composition over mutation).
//! - [`iterate_batches`] — length-sorted, padded, optionally-shuffled batch
//!   iterator (Python `iterate_batches`, `trainer.py:102..=173`).
//! - [`evaluate`] — eval-loss helper (Python `evaluate`, `trainer.py:176..=215`).
//! - [`train`] — the main training loop (Python `train`,
//!   `trainer.py:218..=387`). v1 mechanics-only; see the status note above.
//! - [`TrainingCallback`] — progress-reporting hook trait (Python
//!   `TrainingCallback`, `mlx_lm/tuner/callbacks.py`).
//!
//! ## Scope cuts (deviations from Python — see issue #163)
//!
//! - **Distributed training** (`mx.distributed.AllReduce`, `Group.barrier`,
//!   `average_gradients`, `mx.distributed.all_sum`) — out of scope for v1;
//!   single-process training only. Callers running multi-node have to add
//!   per-step `all_sum`/`average_gradients` themselves via the not-yet-
//!   wrapped `mlxrs_sys::mlx_distributed_*` FFI.
//! - **Adapter checkpoint save / final save** (`mx.save_safetensors`) —
//!   delegated to the caller via the [`TrainingCallback::on_save`] hook
//!   (NOT auto-saved by [`train`]). Rust idiom: don't write to disk inside
//!   library code unless the caller explicitly asks; the [`crate::io`]
//!   module exposes the safetensors / GGUF load+save primitives the
//!   caller composes.
//! - **`mx.metal.is_available()` / `mx.set_wired_limit(...)` /
//!   `mx.get_cache_memory()` / `mx.clear_cache()` / `mx.get_peak_memory()`**
//!   — call sites are no-ops in v1 (mlxrs's memory module covers the same
//!   surface but it is not auto-tuned inside `train` — caller does it).
//! - **`mx.compile` + `partial(state=..., inputs=..., outputs=...)`** —
//!   the Python `@mx.compile`'d `step(...)` closure is NOT replicated; the
//!   Rust loop computes value+grad → optimizer step per iteration straight
//!   from `crate::transforms::value_and_grad`. The compile-graph
//!   optimization is opt-in and out of scope for the v1 training surface.
//! - **`mx.random.state` thread-state** — out of scope; callers seed their
//!   own RNG and pass it through `iterate_batches`'s `seed`.

use std::{collections::HashMap, marker::PhantomData, time::Instant};

use smol_str::format_smolstr;

use crate::{
  Array, Dtype, Result,
  error::{
    EmptyInputPayload, Error, InvariantViolationPayload, LengthMismatchPayload, MissingKeyPayload,
    OutOfRangePayload, RankMismatchPayload, ShapePairMismatchPayload,
  },
  lm::{
    cache::KvCache,
    load::Weights,
    model::Model,
    perplexity,
    tuner::{
      datasets::{Dataset, Example},
      optimizers::Optimizer,
    },
  },
  ops::{arithmetic, comparison, logical, reduction},
  transforms,
};

// ─────────────────────────── TrainingArgs ───────────────────────────

/// Training-loop configuration. Mirrors Python `tuner.TrainingArgs`
/// (`trainer.py:41..=83`).
#[derive(Debug, Clone)]
pub struct TrainingArgs {
  /// Minibatch size (Python `batch_size`, default `4`).
  batch_size: usize,
  /// Total training iterations (Python `iters`, default `100`).
  iters: usize,
  /// Number of validation batches per eval (Python `val_batches`, default
  /// `25`). `None` uses the entire validation set (Python `-1`).
  val_batches: Option<usize>,
  /// OPTIMIZER steps between training-loss reports (Python
  /// `steps_per_report`, default `10`). NOTE: counts OPTIMIZER steps (not
  /// microbatches like the Python ref) — see the v1 status note in the
  /// module-level doc-comment for the rationale.
  steps_per_report: usize,
  /// OPTIMIZER steps between validations (Python `steps_per_eval`,
  /// default `200`). Counts OPTIMIZER steps (see [`Self::steps_per_report`]
  /// for the deviation from the Python ref).
  steps_per_eval: usize,
  /// OPTIMIZER steps between checkpoint saves (Python `steps_per_save`,
  /// default `100`). Counts OPTIMIZER steps (see [`Self::steps_per_report`]
  /// for the deviation from the Python ref).
  steps_per_save: usize,
  /// Maximum per-example sequence length after padding/truncation (Python
  /// `max_seq_length`, default `2048`).
  max_seq_length: usize,
  /// Save/load path for the trained adapter weights (Python `adapter_file`,
  /// default `adapters.safetensors`).
  adapter_file: String,
  /// Enable gradient checkpointing on the first decoder layer (Python
  /// `grad_checkpoint`, default `false`). Caller wraps the layer via
  /// [`grad_checkpoint`] before training; this flag is informational
  /// (training loop does not auto-wrap).
  grad_checkpoint: bool,
  /// Number of micro-batches accumulated before an optimizer step (Python
  /// `grad_accumulation_steps`, default `1`). The training loop
  /// accumulates the SUM of per-microbatch gradients across one window,
  /// divides by this count (the MEAN), then dispatches to
  /// [`crate::lm::tuner::optimizers::Optimizer::apply_gradients`]. The
  /// final partial window at the end of [`Self::iters`] is DROPPED — no
  /// optimizer call fires for it — so the total optimizer step count is
  /// `iters / grad_accumulation_steps` (floored).
  grad_accumulation_steps: usize,
  /// Cache-clear threshold in bytes (Python `clear_cache_threshold`,
  /// default `0` = disabled). v1 is a no-op (memory management out of
  /// scope), kept for API parity.
  clear_cache_threshold: usize,
  /// Caller-side acknowledgment that [`train`]'s v1 path runs the
  /// optimizer / callback / save mechanics but does NOT compute real
  /// per-parameter gradients (the `nn::Module` trait that binds
  /// `params → loss` for [`crate::transforms::value_and_grad`] is not yet
  /// ported). Default is `false` so a future production caller cannot
  /// accidentally run a long training job thinking the model is being
  /// updated. When `false`, [`train`] returns
  /// [`Error::InvariantViolation`] pointing at this
  /// field; set to `true` to opt into the mechanics-only training path.
  ///
  /// **No Python parity:** this field is mlxrs-specific (Python's
  /// `mx.value_and_grad` works against any callable that closes over
  /// `mx.array` parameters, so the Python trainer has nothing analogous
  /// to fence off).
  acknowledge_no_real_gradients: bool,
}

impl TrainingArgs {
  /// Construct a [`TrainingArgs`] with the Python-default values.
  ///
  /// Equivalent to [`TrainingArgs::default()`]. Use `.with_*` builder
  /// methods to override individual fields.
  pub fn new() -> Self {
    Self {
      batch_size: 4,
      iters: 100,
      val_batches: Some(25),
      steps_per_report: 10,
      steps_per_eval: 200,
      steps_per_save: 100,
      max_seq_length: 2048,
      adapter_file: "adapters.safetensors".into(),
      grad_checkpoint: false,
      grad_accumulation_steps: 1,
      clear_cache_threshold: 0,
      // Caller MUST flip this to `true` to opt into the v1 mechanics-only
      // `train()` (see the field doc-comment + module-level v1 status note).
      acknowledge_no_real_gradients: false,
    }
  }

  /// Minibatch size.
  #[inline(always)]
  pub fn batch_size(&self) -> usize {
    self.batch_size
  }

  /// Total training iterations.
  #[inline(always)]
  pub fn iters(&self) -> usize {
    self.iters
  }

  /// Number of validation batches per eval (`None` = entire val set).
  #[inline(always)]
  pub fn val_batches(&self) -> Option<usize> {
    self.val_batches
  }

  /// OPTIMIZER steps between training-loss reports.
  #[inline(always)]
  pub fn steps_per_report(&self) -> usize {
    self.steps_per_report
  }

  /// OPTIMIZER steps between validations.
  #[inline(always)]
  pub fn steps_per_eval(&self) -> usize {
    self.steps_per_eval
  }

  /// OPTIMIZER steps between checkpoint saves.
  #[inline(always)]
  pub fn steps_per_save(&self) -> usize {
    self.steps_per_save
  }

  /// Maximum per-example sequence length after padding/truncation.
  #[inline(always)]
  pub fn max_seq_length(&self) -> usize {
    self.max_seq_length
  }

  /// Save/load path for the trained adapter weights.
  #[inline(always)]
  pub fn adapter_file(&self) -> &str {
    &self.adapter_file
  }

  /// Whether gradient checkpointing is enabled (informational flag).
  #[inline(always)]
  pub fn grad_checkpoint(&self) -> bool {
    self.grad_checkpoint
  }

  /// Number of micro-batches accumulated before an optimizer step.
  #[inline(always)]
  pub fn grad_accumulation_steps(&self) -> usize {
    self.grad_accumulation_steps
  }

  /// Cache-clear threshold in bytes (`0` = disabled).
  #[inline(always)]
  pub fn clear_cache_threshold(&self) -> usize {
    self.clear_cache_threshold
  }

  /// Whether the caller has acknowledged the v1 no-real-gradients limitation.
  #[inline(always)]
  pub fn acknowledge_no_real_gradients(&self) -> bool {
    self.acknowledge_no_real_gradients
  }

  /// Set `batch_size`. Returns `self` for chaining.
  #[must_use]
  pub fn with_batch_size(mut self, batch_size: usize) -> Self {
    self.batch_size = batch_size;
    self
  }

  /// Set `iters`. Returns `self` for chaining.
  #[must_use]
  pub fn with_iters(mut self, iters: usize) -> Self {
    self.iters = iters;
    self
  }

  /// Set `val_batches`. Returns `self` for chaining.
  #[must_use]
  pub fn with_val_batches(mut self, val_batches: Option<usize>) -> Self {
    self.val_batches = val_batches;
    self
  }

  /// Set `steps_per_report`. Returns `self` for chaining.
  #[must_use]
  pub fn with_steps_per_report(mut self, steps_per_report: usize) -> Self {
    self.steps_per_report = steps_per_report;
    self
  }

  /// Set `steps_per_eval`. Returns `self` for chaining.
  #[must_use]
  pub fn with_steps_per_eval(mut self, steps_per_eval: usize) -> Self {
    self.steps_per_eval = steps_per_eval;
    self
  }

  /// Set `steps_per_save`. Returns `self` for chaining.
  #[must_use]
  pub fn with_steps_per_save(mut self, steps_per_save: usize) -> Self {
    self.steps_per_save = steps_per_save;
    self
  }

  /// Set `max_seq_length`. Returns `self` for chaining.
  #[must_use]
  pub fn with_max_seq_length(mut self, max_seq_length: usize) -> Self {
    self.max_seq_length = max_seq_length;
    self
  }

  /// Set `adapter_file`. Returns `self` for chaining.
  #[must_use]
  pub fn with_adapter_file(mut self, adapter_file: impl Into<String>) -> Self {
    self.adapter_file = adapter_file.into();
    self
  }

  /// Set `grad_checkpoint`. Returns `self` for chaining.
  #[must_use]
  pub fn with_grad_checkpoint(mut self, grad_checkpoint: bool) -> Self {
    self.grad_checkpoint = grad_checkpoint;
    self
  }

  /// Set `grad_accumulation_steps`. Returns `self` for chaining.
  #[must_use]
  pub fn with_grad_accumulation_steps(mut self, grad_accumulation_steps: usize) -> Self {
    self.grad_accumulation_steps = grad_accumulation_steps;
    self
  }

  /// Set `clear_cache_threshold`. Returns `self` for chaining.
  #[must_use]
  pub fn with_clear_cache_threshold(mut self, clear_cache_threshold: usize) -> Self {
    self.clear_cache_threshold = clear_cache_threshold;
    self
  }

  /// Set `acknowledge_no_real_gradients`. Returns `self` for chaining.
  #[must_use]
  pub fn with_acknowledge_no_real_gradients(mut self, acknowledge_no_real_gradients: bool) -> Self {
    self.acknowledge_no_real_gradients = acknowledge_no_real_gradients;
    self
  }
}

impl Default for TrainingArgs {
  fn default() -> Self {
    Self::new()
  }
}

// ─────────────────────────── default_loss ───────────────────────────

/// Token-level masked cross-entropy loss for next-token prediction.
///
/// Mirrors Python `default_loss` (`trainer.py:86..=99`), with an exclusive
/// upper bound on the mask (`steps < length` instead of Python's
/// `steps <= length`) to drop the first padded token from the supervised
/// targets. Matches the masking pattern used by mlx-lm's own DWQ trainer
/// (`mlx_lm/quant/dwq.py:115` — `mx.arange(1, 1 + targets.shape[1]) <
/// lengths[:, 1:]`).
///
/// ```text
/// inputs  = batch[:, :-1]
/// targets = batch[:, 1:]
/// logits  = model(inputs)
/// steps   = arange(1, T+1)
/// mask    = (steps >= lengths[:, 0:1]) & (steps < lengths[:, 1:])
/// ce      = cross_entropy(logits, targets) * mask
/// ntoks   = mask.sum()
/// loss    = ce.astype(float32).sum() / ntoks
/// ```
///
/// `batch` is an `[B, S]` integer-token tensor; `lengths` is `[B, 2]`
/// where each row is `(offset, length)`:
/// - tokens at positions `[0, offset)` are the prompt prefix (excluded from
///   the loss);
/// - tokens at positions `[offset, length)` are the completion (included).
///
/// The shifted target at position `length - 1` corresponds to the FIRST
/// padded slot in the unshifted batch (`batch[:, length]` after pad), so
/// the exclusive upper bound excludes it from the supervised loss — the
/// training signal never asks the model to predict the pad token 0 from
/// the last real completion token.
///
/// Returns `(loss_scalar, ntoks_scalar)` — both 0D `Array`s in f32.
///
/// `model.forward` is called WITHOUT a KV cache (training does a fresh
/// forward per step, unlike inference). A future grad-accumulation
/// micro-batching pass through this fn would re-evaluate the same logits
/// — caller controls invocation count.
pub fn default_loss<M>(model: &M, batch: &Array, lengths: &Array) -> Result<(Array, Array)>
where
  M: Model,
{
  let shape = batch.shape();
  let (_b, s) = match shape.as_slice() {
    [b, s] => (*b, *s),
    other => {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "default_loss: batch must be rank-2 [B, S]",
        other.len() as u32,
        other.to_vec(),
      )));
    }
  };
  if s < 2 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "default_loss: batch S",
      "must be >= 2 for next-token prediction",
      format_smolstr!("{s}"),
    )));
  }
  let lengths_shape = lengths.shape();
  let expected_lengths_shape = [shape[0], 2_usize];
  if lengths_shape.as_slice() != expected_lengths_shape {
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      "default_loss: lengths must be [B, 2] = (offset, length)",
      expected_lengths_shape.to_vec(),
      lengths_shape.to_vec(),
    )));
  }
  // inputs = batch[:, :-1], targets = batch[:, 1:]
  let b_dim = shape[0] as i32;
  let s_dim = s as i32;
  let inputs = crate::ops::indexing::slice(batch, &[0, 0], &[b_dim, s_dim - 1], &[1, 1])?;
  let targets = crate::ops::indexing::slice(batch, &[0, 1], &[b_dim, s_dim], &[1, 1])?;
  // Forward — empty cache slice (training does a fresh forward per step).
  let mut cache: Vec<Box<dyn KvCache>> = Vec::new();
  let logits = model.forward(&inputs, &mut cache)?;
  // steps = arange(1, targets.shape[1] + 1) → [1..T]
  let t_dim = targets.shape()[1] as f32;
  let steps = Array::arange::<f32>(1.0, t_dim + 1.0, 1.0)?;
  // mask = (steps >= lengths[:, 0:1]) & (steps < lengths[:, 1:])
  // lengths[:, 0:1] is [B, 1]; lengths[:, 1:] is [B, 1].
  // Exclusive upper bound (`<`) drops the supervised target at
  // `step == length`, which corresponds to the FIRST padded slot in the
  // un-shifted batch. See the function's doc-comment for the off-by-one
  // analysis + mlx-lm DWQ reference.
  let offset = crate::ops::indexing::slice(lengths, &[0, 0], &[b_dim, 1], &[1, 1])?;
  let length = crate::ops::indexing::slice(lengths, &[0, 1], &[b_dim, 2], &[1, 1])?;
  // arange returns f32; cast steps to the same dtype as offset (int)
  // before comparison. Python does the comparison implicitly across
  // f32-int via mlx broadcasting → both promoted to f32.
  let offset_f = offset.astype(Dtype::F32)?;
  let length_f = length.astype(Dtype::F32)?;
  let ge = comparison::greater_equal(&steps, &offset_f)?;
  let lt = comparison::less(&steps, &length_f)?;
  let mask = logical::logical_and(&ge, &lt)?;
  // Cross-entropy (reduction="none") → [B, T]
  let ce = perplexity::cross_entropy_none(&logits, &targets)?;
  // ce * mask
  let mask_f = mask.astype(Dtype::F32)?;
  let ce_masked = arithmetic::multiply(&ce, &mask_f)?;
  // ntoks = mask.sum() (int)
  let mut ntoks = reduction::sum(&mask_f, false)?;
  // Reject zero-supervised-token batches BEFORE the divide rather than
  // producing NaN/Inf downstream (train accumulates `loss * ntoks` and
  // evaluate divides by `total_tokens`; both would silently poison
  // metrics if any batch contained only prompt-only / fully-truncated
  // rows under the exclusive `<` upper bound). The check forces an eval
  // on `ntoks` one division earlier than the caller's `.item::<f32>()`
  // would; the trade-off is an explicit, actionable error vs a silent
  // numerical fault.
  let ntoks_count = ntoks.item::<f32>()?;
  if ntoks_count == 0.0 {
    // The supervised-token set is empty after the length mask: every
    // example in the batch is too short (prompt-only or fully truncated)
    // or has length <= 1. Reject before the divide rather than emitting
    // NaN/Inf downstream; the caller should filter such examples upstream.
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "default_loss: supervised tokens after the length mask (batch produced 0 supervised tokens)",
    )));
  }
  // loss = ce.astype(f32).sum() / ntoks
  let ce_sum = reduction::sum(&ce_masked.astype(Dtype::F32)?, false)?;
  let loss = arithmetic::divide(&ce_sum, &ntoks)?;
  Ok((loss, ntoks))
}

// ─────────────────────────── grad_checkpoint ───────────────────────────

/// Wrap a forward function `f` so its activations are recomputed on the
/// backward pass instead of being stored.
///
/// Mirrors Python `grad_checkpoint` (`trainer.py:25..=38`), with the
/// key difference that Python monkey-patches `type(layer).__call__` (so the
/// wrap is global per layer type) while Rust returns a wrapped closure
/// (composition over mutation — caller substitutes the wrapped fn into
/// the model's forward chain).
///
/// Thin re-export of [`crate::transforms::checkpoint::checkpoint`].
pub fn grad_checkpoint<F>(f: F) -> Result<impl Fn(&[Array]) -> Result<Vec<Array>>>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
  transforms::checkpoint::checkpoint(f)
}

// ─────────────────────────── TrainingCallback ───────────────────────────

/// Hook trait for training-loop progress reporting.
///
/// Mirrors Python `TrainingCallback` (`mlx_lm/tuner/callbacks.py`); each
/// method has a default no-op impl so callers override only what they
/// need.
pub trait TrainingCallback {
  /// Invoked at the end of every [`TrainingArgs::steps_per_report`]
  /// iteration with a summary of the most recent training window.
  fn on_train_loss_report(&mut self, _info: &TrainInfo) {}

  /// Invoked at the end of every [`TrainingArgs::steps_per_eval`]
  /// iteration (and before iteration 1) with a summary of the most recent
  /// validation pass.
  fn on_val_loss_report(&mut self, _info: &ValInfo) {}

  /// Invoked at the end of every [`TrainingArgs::steps_per_save`]
  /// iteration with the current iteration count + the configured
  /// [`TrainingArgs::adapter_file`] path. Default no-op so callers opt
  /// into saving.
  fn on_save(&mut self, _it: usize, _adapter_file: &str) -> Result<()> {
    Ok(())
  }
}

/// Per-window training summary handed to [`TrainingCallback::on_train_loss_report`].
#[derive(Debug, Clone)]
pub struct TrainInfo {
  /// 1-based iteration index at which this report fired.
  iteration: usize,
  /// Mean training loss over the most recent report window.
  train_loss: f32,
  /// Optimizer's resolved learning rate at this iteration.
  learning_rate: f32,
  /// Iterations / second over the most recent report window.
  iterations_per_second: f32,
  /// Tokens / second over the most recent report window.
  tokens_per_second: f32,
  /// Cumulative trained tokens so far.
  trained_tokens: usize,
}

impl TrainInfo {
  /// Construct a [`TrainInfo`].
  pub fn new(
    iteration: usize,
    train_loss: f32,
    learning_rate: f32,
    iterations_per_second: f32,
    tokens_per_second: f32,
    trained_tokens: usize,
  ) -> Self {
    Self {
      iteration,
      train_loss,
      learning_rate,
      iterations_per_second,
      tokens_per_second,
      trained_tokens,
    }
  }

  /// 1-based iteration index at which this report fired.
  #[inline(always)]
  pub fn iteration(&self) -> usize {
    self.iteration
  }

  /// Mean training loss over the most recent report window.
  #[inline(always)]
  pub fn train_loss(&self) -> f32 {
    self.train_loss
  }

  /// Optimizer's resolved learning rate at this iteration.
  #[inline(always)]
  pub fn learning_rate(&self) -> f32 {
    self.learning_rate
  }

  /// Iterations / second over the most recent report window.
  #[inline(always)]
  pub fn iterations_per_second(&self) -> f32 {
    self.iterations_per_second
  }

  /// Tokens / second over the most recent report window.
  #[inline(always)]
  pub fn tokens_per_second(&self) -> f32 {
    self.tokens_per_second
  }

  /// Cumulative trained tokens so far.
  #[inline(always)]
  pub fn trained_tokens(&self) -> usize {
    self.trained_tokens
  }
}

/// Per-eval validation summary handed to [`TrainingCallback::on_val_loss_report`].
#[derive(Debug, Clone)]
pub struct ValInfo {
  /// 1-based iteration index at which this eval fired (note Python uses
  /// `it - 1` for pre-first-step eval; this port mirrors that).
  iteration: usize,
  /// Mean validation loss across `num_batches` eval batches.
  val_loss: f32,
  /// Wall-clock seconds the eval took.
  val_time: f32,
}

impl ValInfo {
  /// Construct a [`ValInfo`].
  pub fn new(iteration: usize, val_loss: f32, val_time: f32) -> Self {
    Self {
      iteration,
      val_loss,
      val_time,
    }
  }

  /// 1-based iteration index at which this eval fired.
  #[inline(always)]
  pub fn iteration(&self) -> usize {
    self.iteration
  }

  /// Mean validation loss across eval batches.
  #[inline(always)]
  pub fn val_loss(&self) -> f32 {
    self.val_loss
  }

  /// Wall-clock seconds the eval took.
  #[inline(always)]
  pub fn val_time(&self) -> f32 {
    self.val_time
  }
}

/// No-op [`TrainingCallback`] used as the default when the caller doesn't
/// provide one.
pub struct NoopCallback;

impl TrainingCallback for NoopCallback {}

// ─────────────────────────── iterate_batches ───────────────────────────

/// One yielded batch from [`iterate_batches`]:
///
/// - `tokens` — the `[B, max_len_in_batch]` int32 token tensor (padded
///   with `0` past each row's true length, truncated to
///   [`TrainingArgs::max_seq_length`] before padding).
/// - `lengths` — the `[B, 2]` `(offset, length)` per-row metadata used by
///   [`default_loss`] to build the per-token loss mask.
pub struct Batch {
  /// `[B, S]` int32 token tensor.
  tokens: Array,
  /// `[B, 2]` `(offset, length)` per-row metadata.
  lengths: Array,
  // PhantomData<'_>-equivalent: keep `Batch` consistent with future fields
  // (e.g. an associated key for distributed sharding) without breaking the
  // ABI.
  _marker: PhantomData<()>,
}

impl Batch {
  /// Construct a [`Batch`] from a token tensor and a lengths tensor.
  pub fn new(tokens: Array, lengths: Array) -> Self {
    Self {
      tokens,
      lengths,
      _marker: PhantomData,
    }
  }

  /// The `[B, S]` int32 token tensor.
  #[inline(always)]
  pub fn tokens_ref(&self) -> &Array {
    &self.tokens
  }

  /// The `[B, 2]` `(offset, length)` per-row metadata tensor.
  #[inline(always)]
  pub fn lengths_ref(&self) -> &Array {
    &self.lengths
  }
}

/// Iterate over `dataset` in length-sorted, padded batches matching the
/// Python `iterate_batches` (`trainer.py:102..=173`) — sans distributed
/// sharding.
///
/// - Sorts examples by length (Python `sorted(range(len(dataset)), key=...)`).
/// - Forms `batch_size`-sized groups in length-sorted order (Python
///   `batch_idx = [idx[i:i+batch_size] for i in range(0, len-bs+1, bs)]`).
/// - Each yielded batch pads every example to `1 + 32·ceil((max_len_in_batch
///   + 31) / 32)` (Python `pad_to = 32` heuristic at `trainer.py:157..=159`),
///   clamped to `max_seq_length`.
/// - If `shuffle_seed` is `Some(seed)`, batch groups are shuffled with a
///   deterministic RNG seeded by `seed` (Python `np.random.seed` +
///   `np.random.permutation`).
///
/// Returns an iterator that yields `Result<Batch>` (errors mid-iter
/// short-circuit to the caller).
///
/// `loop_forever` flag mirrors Python `loop: bool` — when true, the
/// iterator restarts after exhausting all batch groups (used by [`train`]
/// for the main loop; eval passes `false` to terminate after one pass).
pub fn iterate_batches<'a, D: Dataset + 'a>(
  dataset: &'a D,
  batch_size: usize,
  max_seq_length: usize,
  loop_forever: bool,
  shuffle_seed: Option<u64>,
) -> Result<impl Iterator<Item = Result<Batch>> + 'a> {
  if dataset.len() < batch_size {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "iterate_batches: dataset size",
      "must be >= batch_size",
      format_smolstr!("{} (batch_size={batch_size})", dataset.len()),
    )));
  }
  // Length-sort indices.
  let mut idx: Vec<usize> = (0..dataset.len()).collect();
  let lens: Vec<usize> = (0..dataset.len())
    .map(|i| dataset.process(i).map(|(toks, _)| toks.len()))
    .collect::<Result<_>>()?;
  idx.sort_by_key(|&i| lens[i]);
  // Group into batch_size chunks (drop the ragged tail, Python: range
  // `0 .. len-bs+1` step bs).
  let num_batches = dataset.len() / batch_size;
  let mut batch_idx: Vec<Vec<usize>> = Vec::with_capacity(num_batches);
  for i in 0..num_batches {
    batch_idx.push(idx[i * batch_size..(i + 1) * batch_size].to_vec());
  }
  Ok(BatchIter {
    dataset,
    batch_idx,
    max_seq_length,
    cursor: 0,
    order: Vec::new(),
    loop_forever,
    shuffle_seed,
    rng_state: shuffle_seed,
    first_pass: true,
  })
}

struct BatchIter<'a, D: Dataset> {
  dataset: &'a D,
  batch_idx: Vec<Vec<usize>>,
  max_seq_length: usize,
  cursor: usize,
  order: Vec<usize>,
  loop_forever: bool,
  shuffle_seed: Option<u64>,
  rng_state: Option<u64>,
  first_pass: bool,
}

impl<D: Dataset> Iterator for BatchIter<'_, D> {
  type Item = Result<Batch>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.cursor >= self.order.len() {
      // End of one pass.
      if !self.first_pass && !self.loop_forever {
        return None;
      }
      self.first_pass = false;
      // Refresh the iteration order. With shuffle: deterministic Fisher-
      // Yates seeded by `rng_state` (advanced per restart so each pass
      // shuffles differently). Without: in-order.
      self.order = (0..self.batch_idx.len()).collect();
      if self.shuffle_seed.is_some()
        && let Some(seed) = self.rng_state
      {
        fisher_yates_shuffle(&mut self.order, seed);
        // Advance the seed for the next loop pass so successive
        // re-shuffles are distinct (and not the same permutation).
        self.rng_state = Some(seed.wrapping_add(1));
      }
      self.cursor = 0;
      if self.order.is_empty() {
        return None;
      }
    }
    let batch_slot = self.order[self.cursor];
    self.cursor += 1;
    Some(build_batch(
      self.dataset,
      &self.batch_idx[batch_slot],
      self.max_seq_length,
    ))
  }
}

fn build_batch<D: Dataset>(dataset: &D, indices: &[usize], max_seq_length: usize) -> Result<Batch> {
  let mut examples: Vec<Example> = Vec::with_capacity(indices.len());
  for &i in indices {
    examples.push(dataset.process(i)?);
  }
  let lengths: Vec<usize> = examples.iter().map(|(toks, _)| toks.len()).collect();
  // Pad to one plus nearest multiple of pad_to (32) or max_seq_length.
  let pad_to = 32usize;
  let max_in_batch = *lengths.iter().max().unwrap_or(&0);
  let mut max_len_in_batch = 1 + pad_to * max_in_batch.div_ceil(pad_to);
  if max_len_in_batch > max_seq_length {
    max_len_in_batch = max_seq_length;
  }
  let batch_size = examples.len();
  let mut buf = vec![0i32; batch_size * max_len_in_batch];
  let mut len_buf = vec![0i32; batch_size * 2];
  for (j, (toks, offset)) in examples.iter().enumerate() {
    let truncated = toks.len().min(max_seq_length).min(max_len_in_batch);
    for (k, &t) in toks[..truncated].iter().enumerate() {
      buf[j * max_len_in_batch + k] = t as i32;
    }
    len_buf[j * 2] = (*offset).min(truncated) as i32;
    len_buf[j * 2 + 1] = truncated as i32;
  }
  let tokens = Array::from_slice::<i32>(&buf, &(batch_size, max_len_in_batch))?;
  let lengths_arr = Array::from_slice::<i32>(&len_buf, &(batch_size, 2usize))?;
  Ok(Batch::new(tokens, lengths_arr))
}

/// Deterministic Fisher-Yates shuffle. Uses a SplitMix64 RNG so the same
/// `seed` produces the same permutation across runs/platforms (mirrors
/// Python's `np.random.seed(seed); np.random.permutation(...)` determinism
/// without pulling in `rand`).
fn fisher_yates_shuffle<T>(slice: &mut [T], seed: u64) {
  let mut state = seed;
  for i in (1..slice.len()).rev() {
    // SplitMix64 step.
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let j = (z as usize) % (i + 1);
    slice.swap(i, j);
  }
}

// ─────────────────────────── evaluate ───────────────────────────

/// Evaluate `model` on `dataset` for at most `num_batches` batches,
/// returning the token-weighted mean cross-entropy loss.
///
/// Mirrors Python `evaluate` (`trainer.py:176..=215`), without
/// distributed `all_sum`. Each batch's loss + token count is accumulated;
/// the final loss is `total_loss / total_tokens`. `num_batches` of `None`
/// uses the whole eval set (matching Python's `num_batches == -1` sentinel).
pub fn evaluate<M: Model, D: Dataset, F>(
  model: &M,
  dataset: &D,
  batch_size: usize,
  num_batches: Option<usize>,
  max_seq_length: usize,
  mut loss_fn: F,
) -> Result<f32>
where
  F: FnMut(&M, &Array, &Array) -> Result<(Array, Array)>,
{
  let mut total_loss = 0.0_f32;
  let mut total_tokens = 0.0_f32;
  // Eval iterator: NO shuffle, NO loop. One pass over the (length-sorted)
  // batches.
  let iter = iterate_batches(dataset, batch_size, max_seq_length, false, None)?;
  let cap = num_batches.unwrap_or(usize::MAX);
  for (i, batch) in iter.enumerate() {
    if i >= cap {
      break;
    }
    let batch = batch?;
    let (mut loss, mut ntoks) = loss_fn(model, batch.tokens_ref(), batch.lengths_ref())?;
    let loss_f = loss.item::<f32>()?;
    let ntoks_f = ntoks.item::<f32>()?;
    // Token-weighted accumulation: total += per_token_loss · ntoks
    total_loss += loss_f * ntoks_f;
    total_tokens += ntoks_f;
  }
  if total_tokens == 0.0 {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "evaluate: eval set (produced no batches with tokens)",
    )));
  }
  Ok(total_loss / total_tokens)
}

// ─────────────────────────── train ───────────────────────────

/// Run the training loop on `model` + `optimizer` over `train_dataset`,
/// optionally evaluating on `val_dataset` every
/// [`TrainingArgs::steps_per_eval`] OPTIMIZER STEPS.
///
/// Mirrors Python `train` (`trainer.py:218..=387`), with the scope cuts
/// documented in the
/// [scope-cuts module-level note](self#scope-cuts-deviations-from-python),
/// the v1 mechanics-only / no-real-gradients gate documented in the
/// [v1 status module-level note](self#v1-status--mechanics-only-train),
/// and the optimizer-step periodic cadence documented in the
/// [periodic-event cadence note](self#periodic-event-cadence--optimizer-steps-deviation-from-python).
///
/// Per microbatch the loop computes `(loss, grads)` and accumulates
/// `grads` into a running sum across [`TrainingArgs::grad_accumulation_steps`]
/// microbatches; once the window is complete it divides the accumulator
/// by `grad_accumulation_steps` and dispatches the MEAN to
/// [`Optimizer::apply_gradients`]. Any final partial window at the end
/// of [`TrainingArgs::iters`] is dropped. (`grads` is currently
/// `zeros_like(params)` per the v1 mechanics-only note above.)
///
/// ## Parameter handoff
///
/// `params` is a mutable [`Weights`] map (the same flat-key shape mlxrs
/// uses everywhere). The caller owns the parameter map and the optimizer
/// mutates it in place each step. The model is read-only — it consumes the
/// parameters indirectly (e.g. baked into its captured state at load time).
/// This deviates from Python's `model.update(params)` per-step pattern
/// because mlxrs has no `nn.Module` runtime parameter system yet (a future
/// follow-up will introduce a `Module` trait + `update()` hook).
///
/// ## Loss closure
///
/// `loss_fn` takes `(model, tokens, lengths)` and returns
/// `(loss_scalar, ntoks_scalar)`. The defaults are [`default_loss`]; pass
/// a custom closure for specialized losses (label smoothing, KD, etc.).
#[allow(clippy::too_many_arguments)]
pub fn train<M, D, O, L, C>(
  model: &M,
  optimizer: &mut O,
  params: &mut Weights,
  train_dataset: &D,
  val_dataset: Option<&D>,
  args: &TrainingArgs,
  loss_fn: L,
  callback: &mut C,
) -> Result<()>
where
  M: Model,
  D: Dataset,
  O: Optimizer + ?Sized,
  L: Fn(&M, &Array, &Array) -> Result<(Array, Array)>,
  C: TrainingCallback,
{
  if !args.acknowledge_no_real_gradients() {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "train: TrainingArgs::acknowledge_no_real_gradients",
      "must be set to `true` to run the v1 mechanics-only training path",
    )));
  }
  if args.iters() == 0 {
    return Ok(());
  }
  // Validate every interval field used as a modulo divisor. A `0`
  // interval would underflow `it % 0` (panic) the first time the loop
  // tested the periodic-report / eval / save predicate, so reject up
  // front with a clear error instead of letting it crash at iteration 1.
  if args.grad_accumulation_steps() == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "train: grad_accumulation_steps",
      "must be >= 1",
    )));
  }
  if args.steps_per_report() == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "train: steps_per_report",
      "must be >= 1",
    )));
  }
  if args.steps_per_eval() == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "train: steps_per_eval",
      "must be >= 1",
    )));
  }
  if args.steps_per_save() == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "train: steps_per_save",
      "must be >= 1",
    )));
  }
  // Total OPTIMIZER steps the loop will execute. Microbatch count is
  // `args.iters()`; one optimizer step per `args.grad_accumulation_steps()`
  // microbatches; any final partial window is DROPPED (no optimizer step
  // for it). The floored division is therefore the right count.
  let total_optim_steps = args.iters() / args.grad_accumulation_steps();
  // Periodic-report window accumulators. `window_steps` is OPTIMIZER
  // STEPS in the current report window, `window_secs` is the cumulative
  // wall-clock time across all microbatches that fed those steps,
  // `window_microbatches` is the per-microbatch count used to denominate
  // the mean train loss (mirrors mlx-lm's per-microbatch loss semantic —
  // dividing by `window_steps` instead would inflate the reported loss
  // by `grad_accumulation_steps×` for every callback / log line / early-
  // stop monitor).
  let mut window_loss = 0.0_f32;
  let mut window_tokens = 0.0_f32;
  let mut window_steps = 0usize;
  let mut window_microbatches = 0usize;
  let mut window_secs = 0.0_f32;
  let mut trained_tokens = 0usize;
  // Gradient-accumulation state. `accumulated_grads` collects the SUM of
  // per-microbatch gradients across one optimizer window, then is divided
  // by `args.grad_accumulation_steps()` (the MEAN) before being dispatched
  // to the optimizer.
  let mut accumulated_grads: Option<Weights> = None;
  let mut accum_count: usize = 0;
  // OPTIMIZER step counter (NOT microbatch counter). Periodic events —
  // train-loss reports, val-loss evals, save hooks — fire on this
  // counter, so the per-event cadence is independent of
  // `grad_accumulation_steps`. Deviation from `mlx-lm/tuner/trainer.py`
  // which counts microbatches (see the v1 status note in the module-level
  // doc-comment); chosen so a caller bumping `grad_accumulation_steps`
  // doesn't accidentally inflate their report / eval / save frequency.
  let mut optim_step: usize = 0;
  // Per-microbatch timing accumulator for the current optimizer window.
  // Folded into `window_secs` and reset every time the optimizer fires.
  let mut window_micro_secs = 0.0_f32;
  let mut iter = iterate_batches(
    train_dataset,
    args.batch_size(),
    args.max_seq_length(),
    true,
    None,
  )?;
  // Pre-loop val — emit BEFORE the first optimizer step (Python
  // trainer.py:286..=317 does this implicitly by checking `it == 1`
  // before its first step body). `iteration: 0` matches the
  // microbatch-based semantics, which fire at `iteration: it - 1 = 0`.
  if let Some(val) = val_dataset
    && total_optim_steps >= 1
  {
    run_val(model, val, args, 0, callback, &loss_fn)?;
  }
  for _microbatch_it in 1..=args.iters() {
    let micro_start = Instant::now();
    let batch = iter.next().ok_or_else(|| {
      Error::InvariantViolation(InvariantViolationPayload::new(
        "train: batch iterator",
        "must never be exhausted (loop=true should never end)",
      ))
    })??;
    // Compute loss + (placeholder) gradients. NOTE: this is the v1
    // mechanics-only path — production code threads `value_and_grad`
    // over a future `nn::Module` trait that binds `params -> loss`. v1
    // ships a no-grad pass-through (`build_zero_grads`) gated by
    // [`TrainingArgs::acknowledge_no_real_gradients`] so the optimizer /
    // callback / save mechanics can be tested end-to-end.
    let (loss_scalar, ntoks_scalar) = (loss_fn)(model, batch.tokens_ref(), batch.lengths_ref())?;
    let mut loss_val = loss_scalar.try_clone()?;
    let mut ntoks_val = ntoks_scalar.try_clone()?;
    let loss_f = loss_val.item::<f32>()?;
    let ntoks_f = ntoks_val.item::<f32>()?;
    let grads: Weights = build_zero_grads(params)?;
    // Accumulate (sum) into the current window.
    accumulated_grads = Some(match accumulated_grads {
      None => grads,
      Some(acc) => add_weights(&acc, &grads)?,
    });
    accum_count += 1;
    window_loss += loss_f;
    window_tokens += ntoks_f;
    window_microbatches += 1;
    trained_tokens += ntoks_f as usize;
    window_micro_secs += micro_start.elapsed().as_secs_f32();
    // Optimizer step fires only when the accumulation window is full.
    // Partial windows at the end of `iters` are DROPPED (no
    // apply_gradients call for them); see the contract documented on
    // [`TrainingArgs::grad_accumulation_steps`] + the v1 status note.
    if accum_count < args.grad_accumulation_steps() {
      continue;
    }
    let avg = divide_weights(
      accumulated_grads
        .as_ref()
        .expect("accumulated_grads must be Some after at least one accum"),
      args.grad_accumulation_steps() as f32,
    )?;
    optimizer.apply_gradients(&avg, params)?;
    optim_step += 1;
    accumulated_grads = None;
    accum_count = 0;
    window_steps += 1;
    window_secs += window_micro_secs;
    window_micro_secs = 0.0;
    let is_last_optim_step = optim_step == total_optim_steps;
    // Periodic train-loss report (cadence in OPTIMIZER STEPS).
    if optim_step.is_multiple_of(args.steps_per_report()) || is_last_optim_step {
      // Mean train loss is denominated by COMPLETED MICROBATCHES, not by
      // optimizer-step count: `window_loss` aggregates one summand per
      // microbatch (line ~767), so dividing by `window_steps` (=
      // window_microbatches / grad_accumulation_steps) inflates the
      // reported loss by `grad_accumulation_steps×`. See trainer module
      // doc note + the regression test
      // `grad_accumulation_steps_4_reports_constant_loss_at_2_not_8`.
      let mean_loss = if window_microbatches > 0 {
        window_loss / (window_microbatches as f32)
      } else {
        0.0
      };
      let it_sec = if window_secs > 0.0 {
        (window_steps as f32) / window_secs
      } else {
        0.0
      };
      let tok_sec = if window_secs > 0.0 {
        window_tokens / window_secs
      } else {
        0.0
      };
      callback.on_train_loss_report(&TrainInfo::new(
        optim_step,
        mean_loss,
        optimizer.learning_rate(),
        it_sec,
        tok_sec,
        trained_tokens,
      ));
      window_loss = 0.0;
      window_tokens = 0.0;
      window_steps = 0;
      window_microbatches = 0;
      window_secs = 0.0;
    }
    // Periodic mid-training eval (cadence in OPTIMIZER STEPS). Fires
    // both on the regular cadence and at the final optimizer step (so
    // the caller always sees an end-of-training validation).
    if let Some(val) = val_dataset
      && (optim_step.is_multiple_of(args.steps_per_eval()) || is_last_optim_step)
    {
      run_val(model, val, args, optim_step, callback, &loss_fn)?;
    }
    // Periodic save hook (cadence in OPTIMIZER STEPS).
    if optim_step.is_multiple_of(args.steps_per_save()) {
      callback.on_save(optim_step, args.adapter_file())?;
    }
  }
  // Final save hook (Python: writes adapters.safetensors at the end).
  // Iteration label is the LAST optimizer step (0 if there were no
  // optimizer steps, e.g. iters < grad_accumulation_steps).
  callback.on_save(optim_step, args.adapter_file())?;
  Ok(())
}

/// Run one validation pass and dispatch `on_val_loss_report` with the
/// matching [`ValInfo`]. Centralized so the train loop's pre-loop
/// (iteration 0) and per-step (iteration = `optim_step`) val call sites
/// share one body.
fn run_val<M, D, L, C>(
  model: &M,
  val: &D,
  args: &TrainingArgs,
  iteration: usize,
  callback: &mut C,
  loss_fn: &L,
) -> Result<()>
where
  M: Model,
  D: Dataset,
  L: Fn(&M, &Array, &Array) -> Result<(Array, Array)>,
  C: TrainingCallback,
{
  let val_start = Instant::now();
  let val_loss = evaluate(
    model,
    val,
    args.batch_size(),
    args.val_batches(),
    args.max_seq_length(),
    |m, b, l| (loss_fn)(m, b, l),
  )?;
  let val_time = val_start.elapsed().as_secs_f32();
  callback.on_val_loss_report(&ValInfo::new(iteration, val_loss, val_time));
  Ok(())
}

/// Build a `Weights` with `zeros_like` of each entry's `Array`. Used by
/// the v1 [`train`] loop's pass-through gradient path — production
/// integration replaces this with `value_and_grad(loss_closure)` over a
/// future `Module` trait that maps `params → loss`.
fn build_zero_grads(params: &Weights) -> Result<Weights> {
  let mut grads: Weights = HashMap::with_capacity(params.len());
  for (key, value) in params {
    grads.insert(key.clone(), crate::ops::misc::zeros_like(value)?);
  }
  Ok(grads)
}

/// Element-wise sum of two parameter-keyed gradient maps. `a` and `b`
/// must share the same key set (the trainer always builds them from the
/// same `params`, so this is an internal invariant — any missing key is
/// reported via [`Error::MissingKey`] rather than silently dropped).
fn add_weights(a: &Weights, b: &Weights) -> Result<Weights> {
  if a.len() != b.len() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "trainer::add_weights: lhs vs rhs key counts",
      a.len(),
      b.len(),
    )));
  }
  let mut out: Weights = HashMap::with_capacity(a.len());
  for (key, lhs) in a {
    let Some(rhs) = b.get(key) else {
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "trainer::add_weights: key missing from rhs",
        key.as_str(),
      )));
    };
    out.insert(key.clone(), arithmetic::add(lhs, rhs)?);
  }
  Ok(out)
}

/// Scalar-divide every entry in a parameter-keyed gradient map by
/// `divisor`. Used by the [`train`] gradient-accumulation path to
/// average the per-microbatch summed gradients before dispatching to
/// the optimizer.
fn divide_weights(w: &Weights, divisor: f32) -> Result<Weights> {
  let divisor_scalar = Array::full::<f32>(&[0i32; 0], divisor)?;
  let mut out: Weights = HashMap::with_capacity(w.len());
  for (key, value) in w {
    out.insert(key.clone(), arithmetic::divide(value, &divisor_scalar)?);
  }
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::lm::tuner::optimizers::sgd::SGD;

  // Tiny in-memory dataset for trainer tests.
  struct FakeDataset {
    samples: Vec<Example>,
  }
  impl FakeDataset {
    fn new(n: usize, len: usize) -> Self {
      let samples = (0..n)
        .map(|i| ((0..len).map(|k| ((i + k) as u32) % 32).collect(), 0_usize))
        .collect();
      Self { samples }
    }
  }
  impl Dataset for FakeDataset {
    fn len(&self) -> usize {
      self.samples.len()
    }
    fn get(&self, _idx: usize) -> Result<&serde_json::Value> {
      Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "FakeDataset::get",
        "is not used by the trainer iterator",
      )))
    }
    fn process(&self, idx: usize) -> Result<Example> {
      Ok((self.samples[idx].0.clone(), self.samples[idx].1))
    }
  }

  // Tiny model: returns vocab=8 uniform logits, ignores cache.
  struct FakeModel;
  impl Model for FakeModel {
    fn forward(&self, tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
      let shape = tokens.shape();
      let (b, s) = (shape[0], shape[1]);
      // Uniform logits over vocab=8.
      let vocab = 8;
      let n = b * s * vocab;
      let buf = vec![0.1_f32; n];
      Array::from_slice::<f32>(&buf, &(b, s, vocab))
    }
  }

  #[test]
  fn training_args_default_matches_python() {
    let a = TrainingArgs::default();
    assert_eq!(a.batch_size(), 4);
    assert_eq!(a.iters(), 100);
    assert_eq!(a.val_batches(), Some(25));
    assert_eq!(a.steps_per_report(), 10);
    assert_eq!(a.steps_per_eval(), 200);
    assert_eq!(a.steps_per_save(), 100);
    assert_eq!(a.max_seq_length(), 2048);
    assert!(!a.grad_checkpoint());
    assert_eq!(a.grad_accumulation_steps(), 1);
    // mlxrs-specific: defaults to false so a fresh `TrainingArgs` cannot
    // accidentally run the v1 mechanics-only stub.
    assert!(!a.acknowledge_no_real_gradients());
  }

  #[test]
  fn default_loss_matches_masked_cross_entropy() -> Result<()> {
    // FakeModel returns uniform vocab=8 logits regardless of input. We
    // construct a small [B=1, S=3] batch with lengths=(1,3): mask is at
    // positions {step : step >= 1 && step < 3} = {1, 2} of the [S-1=2]-
    // element target → 2 tokens contribute. The exclusive upper bound
    // (`<`) doesn't change this case because the target range stops at
    // T=2 (steps never reach step==length=3); see
    // `default_loss_excludes_padded_target_at_length_boundary` for the
    // regression that exercises the boundary.
    let model = FakeModel;
    // batch [B=1, S=3]: tokens [1, 2, 3]
    let batch = Array::from_slice::<i32>(&[1, 2, 3], &(1, 3))?;
    // lengths [B=1, 2]: (offset=1, length=3)
    let lengths = Array::from_slice::<i32>(&[1, 3], &(1, 2))?;
    let (mut loss, mut ntoks) = default_loss(&model, &batch, &lengths)?;
    let loss_v = loss.item::<f32>()?;
    let ntoks_v = ntoks.item::<f32>()?;
    // Uniform logits → cross-entropy per token = log(8) ≈ 2.0794
    assert!((loss_v - 8.0_f32.ln()).abs() < 1e-4, "got loss {loss_v}");
    // mask at positions {1,2} of the 2-element target → ntoks=2.
    assert!((ntoks_v - 2.0).abs() < 1e-6, "got ntoks {ntoks_v}");
    Ok(())
  }

  #[test]
  fn default_loss_excludes_padded_target_at_length_boundary() -> Result<()> {
    // Construct a [B=1, S=4] batch with lengths=(0, 2). Valid tokens are
    // at positions [0, 2): batch[0], batch[1]. Positions batch[2..4) are
    // padding (zeros). After shifting, targets has [S-1=3] positions:
    //   target[0] = batch[1] (valid)
    //   target[1] = batch[2] (PAD — boundary)
    //   target[2] = batch[3] (PAD)
    // The mask `steps >= offset && steps < length` with offset=0 and
    // length=2 keeps steps ∈ {1} (since arange runs over [1, 4)):
    //   step 1: 1 >= 0 AND 1 < 2 → ✓ (target[0] = batch[1] = valid)
    //   step 2: 2 >= 0 AND 2 < 2 → ✗ (would be target[1] = batch[2] = PAD)
    //   step 3: 3 >= 0 AND 3 < 2 → ✗
    // An inclusive `<=` upper bound would INCLUDE step 2 (the boundary
    // pad), counting batch[2] = 0 as a supervised target and skewing
    // training toward predicting the pad token. ntoks must be 1, not 2.
    let model = FakeModel;
    let batch = Array::from_slice::<i32>(&[1, 2, 0, 0], &(1, 4))?;
    let lengths = Array::from_slice::<i32>(&[0, 2], &(1, 2))?;
    let (mut loss, mut ntoks) = default_loss(&model, &batch, &lengths)?;
    let loss_v = loss.item::<f32>()?;
    let ntoks_v = ntoks.item::<f32>()?;
    assert!(
      (ntoks_v - 1.0).abs() < 1e-6,
      "expected ntoks=1 (boundary pad excluded by `<` upper bound), got {ntoks_v}",
    );
    // Single supervised token, uniform logits over vocab=8 → loss = log(8).
    assert!(
      (loss_v - 8.0_f32.ln()).abs() < 1e-4,
      "expected loss=log(8) for single supervised token, got {loss_v}",
    );
    Ok(())
  }

  #[test]
  fn iterate_batches_emits_expected_shape_for_known_dataset_size() -> Result<()> {
    let dataset = FakeDataset::new(8, 4); // 8 examples × len 4
    let iter = iterate_batches(&dataset, 4, 64, false, None)?;
    let mut count = 0;
    for b in iter {
      let b = b?;
      assert_eq!(b.tokens_ref().shape()[0], 4);
      assert_eq!(b.lengths_ref().shape(), &[4, 2]);
      count += 1;
    }
    assert_eq!(count, 2, "8/4=2 batches expected");
    Ok(())
  }

  #[test]
  fn iterate_batches_rejects_too_small_dataset() {
    let dataset = FakeDataset::new(2, 4);
    let res = iterate_batches(&dataset, 4, 64, false, None);
    assert!(res.is_err());
  }

  #[test]
  fn iterate_batches_loop_forever_yields_more_batches_than_dataset_size() -> Result<()> {
    let dataset = FakeDataset::new(4, 4); // 1 batch per pass
    let mut iter = iterate_batches(&dataset, 4, 64, true, Some(0xCAFE))?;
    // Take 5 batches (way more than 1 per pass) — must not exhaust.
    for _ in 0..5 {
      assert!(iter.next().is_some());
    }
    Ok(())
  }

  #[test]
  fn evaluate_returns_correct_loss_for_known_eval_set() -> Result<()> {
    let dataset = FakeDataset::new(4, 6); // 1 batch of 4 examples
    let model = FakeModel;
    let loss = evaluate(&model, &dataset, 4, Some(1), 64, |m, b, l| {
      default_loss(m, b, l)
    })?;
    // Uniform logits over vocab=8: cross-entropy per token = log(8) ≈ 2.0794.
    assert!((loss - 8.0_f32.ln()).abs() < 1e-4, "got {loss}");
    Ok(())
  }

  struct CountingCallback {
    train_reports: usize,
    val_reports: usize,
    saves: usize,
  }
  impl TrainingCallback for CountingCallback {
    fn on_train_loss_report(&mut self, _info: &TrainInfo) {
      self.train_reports += 1;
    }
    fn on_val_loss_report(&mut self, _info: &ValInfo) {
      self.val_reports += 1;
    }
    fn on_save(&mut self, _it: usize, _adapter_file: &str) -> Result<()> {
      self.saves += 1;
      Ok(())
    }
  }

  #[test]
  fn train_completes_n_iters_with_progress_callback() -> Result<()> {
    let dataset = FakeDataset::new(4, 6); // 1 batch per pass
    let model = FakeModel;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
    let mut sgd = SGD::vanilla(0.01)?;
    let mut cb = CountingCallback {
      train_reports: 0,
      val_reports: 0,
      saves: 0,
    };
    let args = TrainingArgs::new()
      .with_iters(6)
      .with_steps_per_report(2)
      .with_steps_per_eval(4)
      .with_steps_per_save(3)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1))
      .with_acknowledge_no_real_gradients(true);
    train(
      &model,
      &mut sgd,
      &mut params,
      &dataset,
      Some(&dataset),
      &args,
      default_loss,
      &mut cb,
    )?;
    // 6 iters @ steps_per_report=2 → 3 windows (it=2,4,6).
    assert_eq!(cb.train_reports, 3);
    // val: it=1, it=4 (multiple of 4), it=6 (final). = 3 vals
    assert_eq!(cb.val_reports, 3);
    // save: it=3, it=6 plus final → 3 saves
    assert_eq!(cb.saves, 3);
    Ok(())
  }

  #[test]
  fn grad_checkpoint_wraps_layer_without_changing_output() -> Result<()> {
    // x → x² wrapped in checkpoint produces the same forward value.
    let plain = |xs: &[Array]| Ok(vec![crate::ops::arithmetic::square(&xs[0])?]);
    let wrapped = grad_checkpoint(plain)?;
    let x = Array::full::<f32>(&[0i32; 0], 3.0)?;
    let mut out = wrapped(&[x])?;
    assert_eq!(out[0].item::<f32>()?, 9.0);
    Ok(())
  }

  // ─────────── acknowledge_no_real_gradients gate ───────────

  #[test]
  fn train_rejects_when_acknowledge_no_real_gradients_is_false() -> Result<()> {
    let dataset = FakeDataset::new(4, 6);
    let model = FakeModel;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
    let mut sgd = SGD::vanilla(0.01)?;
    let mut cb = NoopCallback;
    // Default args leaves `acknowledge_no_real_gradients` = false.
    let args = TrainingArgs::new()
      .with_iters(1)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1));
    assert!(!args.acknowledge_no_real_gradients());
    let res = train(
      &model,
      &mut sgd,
      &mut params,
      &dataset,
      None,
      &args,
      default_loss,
      &mut cb,
    );
    match res {
      Err(Error::InvariantViolation(payload)) => {
        assert_eq!(
          payload.context(),
          "train: TrainingArgs::acknowledge_no_real_gradients"
        );
        assert_eq!(
          payload.requirement(),
          "must be set to `true` to run the v1 mechanics-only training path"
        );
      }
      other => panic!("expected Err(InvariantViolation), got {other:?}"),
    }
    Ok(())
  }

  #[test]
  fn train_runs_when_acknowledge_no_real_gradients_is_true() -> Result<()> {
    let dataset = FakeDataset::new(4, 6);
    let model = FakeModel;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
    let mut sgd = SGD::vanilla(0.01)?;
    let mut cb = NoopCallback;
    let args = TrainingArgs::new()
      .with_iters(1)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1))
      .with_acknowledge_no_real_gradients(true);
    let res = train(
      &model,
      &mut sgd,
      &mut params,
      &dataset,
      None,
      &args,
      default_loss,
      &mut cb,
    );
    assert!(
      res.is_ok(),
      "train should run when opt-in is set; got {res:?}"
    );
    Ok(())
  }

  // ─────────── zero-interval rejection ───────────

  fn args_for_zero_interval_tests() -> TrainingArgs {
    TrainingArgs::new()
      .with_iters(1)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1))
      .with_acknowledge_no_real_gradients(true)
  }

  fn run_train_with_args(args: &TrainingArgs) -> crate::Result<()> {
    let dataset = FakeDataset::new(4, 6);
    let model = FakeModel;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
    let mut sgd = SGD::vanilla(0.01)?;
    let mut cb = NoopCallback;
    train(
      &model,
      &mut sgd,
      &mut params,
      &dataset,
      None,
      args,
      default_loss,
      &mut cb,
    )
  }

  #[test]
  fn train_rejects_zero_steps_per_report() {
    let args = args_for_zero_interval_tests().with_steps_per_report(0);
    let res = run_train_with_args(&args);
    match res {
      Err(Error::InvariantViolation(payload)) => {
        assert_eq!(payload.context(), "train: steps_per_report");
        assert_eq!(payload.requirement(), "must be >= 1");
      }
      other => panic!("expected Err(InvariantViolation) for steps_per_report=0; got {other:?}"),
    }
  }

  #[test]
  fn train_rejects_zero_steps_per_eval() {
    let args = args_for_zero_interval_tests().with_steps_per_eval(0);
    let res = run_train_with_args(&args);
    match res {
      Err(Error::InvariantViolation(payload)) => {
        assert_eq!(payload.context(), "train: steps_per_eval");
        assert_eq!(payload.requirement(), "must be >= 1");
      }
      other => panic!("expected Err(InvariantViolation) for steps_per_eval=0; got {other:?}"),
    }
  }

  #[test]
  fn train_rejects_zero_steps_per_save() {
    let args = args_for_zero_interval_tests().with_steps_per_save(0);
    let res = run_train_with_args(&args);
    match res {
      Err(Error::InvariantViolation(payload)) => {
        assert_eq!(payload.context(), "train: steps_per_save");
        assert_eq!(payload.requirement(), "must be >= 1");
      }
      other => panic!("expected Err(InvariantViolation) for steps_per_save=0; got {other:?}"),
    }
  }

  #[test]
  fn train_rejects_zero_grad_accumulation_steps() {
    let args = args_for_zero_interval_tests().with_grad_accumulation_steps(0);
    let res = run_train_with_args(&args);
    match res {
      Err(Error::InvariantViolation(payload)) => {
        assert_eq!(payload.context(), "train: grad_accumulation_steps");
        assert_eq!(payload.requirement(), "must be >= 1");
      }
      other => {
        panic!("expected Err(InvariantViolation) for grad_accumulation_steps=0; got {other:?}")
      }
    }
  }

  // ─────────── grad accumulation respects window cadence ───────────

  /// Counting optimizer wrapper: counts `apply_gradients` invocations
  /// without modifying params. Used to assert the train loop fires the
  /// optimizer at the OPTIMIZER STEP cadence (one call per accumulation
  /// window completion) rather than per microbatch.
  struct CountingOptimizer {
    apply_calls: usize,
    step_count: usize,
    lr: f32,
  }
  impl crate::lm::tuner::optimizers::Optimizer for CountingOptimizer {
    fn init(&mut self, _params: &Weights) -> Result<()> {
      Ok(())
    }
    fn apply_gradients(&mut self, _gradients: &Weights, _params: &mut Weights) -> Result<()> {
      self.apply_calls += 1;
      self.step_count += 1;
      Ok(())
    }
    fn step(&self) -> usize {
      self.step_count
    }
    fn learning_rate(&self) -> f32 {
      self.lr
    }
  }

  fn build_train_fixture() -> Result<(
    FakeModel,
    FakeDataset,
    Weights,
    NoopCallback,
    CountingOptimizer,
  )> {
    let dataset = FakeDataset::new(4, 6);
    let model = FakeModel;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
    let cb = NoopCallback;
    let opt = CountingOptimizer {
      apply_calls: 0,
      step_count: 0,
      lr: 0.0,
    };
    Ok((model, dataset, params, cb, opt))
  }

  #[test]
  fn grad_accumulation_steps_2_calls_optimizer_every_other_iter() -> Result<()> {
    // iters=10, grad_accumulation_steps=2 → 5 optimizer calls.
    let (model, dataset, mut params, mut cb, mut opt) = build_train_fixture()?;
    let args = TrainingArgs::new()
      .with_iters(10)
      .with_grad_accumulation_steps(2)
      // steps_per_* large enough to avoid firing during this test (we
      // only care about optimizer call count here).
      .with_steps_per_report(100)
      .with_steps_per_eval(100)
      .with_steps_per_save(100)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1))
      .with_acknowledge_no_real_gradients(true);
    train(
      &model,
      &mut opt,
      &mut params,
      &dataset,
      None,
      &args,
      default_loss,
      &mut cb,
    )?;
    assert_eq!(
      opt.apply_calls, 5,
      "iters=10 + grad_accumulation_steps=2 must produce 5 optimizer steps; got {}",
      opt.apply_calls,
    );
    Ok(())
  }

  #[test]
  fn grad_accumulation_steps_partial_window_at_end_drops() -> Result<()> {
    // iters=11, grad_accumulation_steps=4 → only 11/4 = 2 complete
    // windows (microbatches 1..=4 → step 1, 5..=8 → step 2). The final
    // 3 microbatches (9, 10, 11) form a partial window which is DROPPED
    // (no third optimizer call).
    let (model, dataset, mut params, mut cb, mut opt) = build_train_fixture()?;
    let args = TrainingArgs::new()
      .with_iters(11)
      .with_grad_accumulation_steps(4)
      .with_steps_per_report(100)
      .with_steps_per_eval(100)
      .with_steps_per_save(100)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1))
      .with_acknowledge_no_real_gradients(true);
    train(
      &model,
      &mut opt,
      &mut params,
      &dataset,
      None,
      &args,
      default_loss,
      &mut cb,
    )?;
    assert_eq!(
      opt.apply_calls, 2,
      "iters=11 + grad_accumulation_steps=4 must drop the final partial \
       window of 3 microbatches; expected 2 optimizer calls, got {}",
      opt.apply_calls,
    );
    Ok(())
  }

  #[test]
  fn grad_accumulation_steps_1_is_identity_to_microbatch_count() -> Result<()> {
    // The grad_accumulation_steps=1 case must NOT regress — every
    // microbatch is its own optimizer step.
    let (model, dataset, mut params, mut cb, mut opt) = build_train_fixture()?;
    let args = TrainingArgs::new()
      .with_iters(7)
      .with_grad_accumulation_steps(1)
      .with_steps_per_report(100)
      .with_steps_per_eval(100)
      .with_steps_per_save(100)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1))
      .with_acknowledge_no_real_gradients(true);
    train(
      &model,
      &mut opt,
      &mut params,
      &dataset,
      None,
      &args,
      default_loss,
      &mut cb,
    )?;
    assert_eq!(opt.apply_calls, 7);
    Ok(())
  }

  // ─────────── report-loss denominator parity with mlx-lm ───────────

  /// Recording callback: captures `train_loss` from every
  /// `on_train_loss_report` call. Used to prove the report denominator
  /// matches the per-microbatch loss (mlx-lm parity) instead of the
  /// optimizer-step count (which would inflate every reported loss by
  /// `grad_accumulation_steps×`).
  struct LossRecordingCallback {
    losses: Vec<f32>,
  }
  impl TrainingCallback for LossRecordingCallback {
    fn on_train_loss_report(&mut self, info: &TrainInfo) {
      self.losses.push(info.train_loss());
    }
  }

  #[test]
  fn grad_accumulation_steps_4_reports_constant_loss_at_2_not_8() -> Result<()> {
    // Regression guard against loss inflation: when each microbatch's loss
    // is constant 2.0 and `grad_accumulation_steps = 4`, summing one term
    // per microbatch into `window_loss` and then dividing by `window_steps`
    // (which only increments per completed accumulation window) would
    // report an 8.0 loss — every callback / log line / early-stop monitor
    // would see the per-microbatch loss multiplied by 4×.
    //
    // The denominator is the completed-microbatch count, so every report
    // fires at the true per-microbatch loss 2.0.
    let dataset = FakeDataset::new(4, 6);
    let model = FakeModel;
    let mut params: Weights = HashMap::new();
    params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
    let mut opt = CountingOptimizer {
      apply_calls: 0,
      step_count: 0,
      lr: 0.0,
    };
    let mut cb = LossRecordingCallback { losses: Vec::new() };
    // Constant-loss mock: returns (loss=2.0, ntoks=1.0) for every
    // microbatch, regardless of inputs. Drives `window_loss` to grow by
    // exactly +2.0 per microbatch so the inflation factor is unambiguous.
    let const_loss_fn =
      |_m: &FakeModel, _batch: &Array, _lengths: &Array| -> Result<(Array, Array)> {
        let loss = Array::full::<f32>(&[0i32; 0], 2.0)?;
        let ntoks = Array::full::<f32>(&[0i32; 0], 1.0)?;
        Ok((loss, ntoks))
      };
    let args = TrainingArgs::new()
      .with_iters(12)
      .with_grad_accumulation_steps(4)
      // 12 microbatches / 4 = 3 optimizer steps. steps_per_report=1 fires
      // a report on EVERY optimizer step, so we get 3 callbacks
      // (windows of 4 microbatches each, all with constant per-microbatch
      // loss = 2.0).
      .with_steps_per_report(1)
      .with_steps_per_eval(100)
      .with_steps_per_save(100)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1))
      .with_acknowledge_no_real_gradients(true);
    train(
      &model,
      &mut opt,
      &mut params,
      &dataset,
      None,
      &args,
      const_loss_fn,
      &mut cb,
    )?;
    assert_eq!(
      cb.losses.len(),
      3,
      "iters=12 + grad_accumulation_steps=4 + steps_per_report=1 must fire 3 train-loss reports; got {}",
      cb.losses.len(),
    );
    for (i, &loss) in cb.losses.iter().enumerate() {
      assert!(
        (loss - 2.0).abs() < 1e-6,
        "report #{i} train_loss = {loss}, expected 2.0 (per-microbatch loss); dividing \
         `window_loss / window_steps` (4×constant-2.0 by 1 optimizer-step) would wrongly \
         report 8.0",
      );
    }
    Ok(())
  }

  // ─────────── default_loss rejects zero-supervised-token batches ───────────

  #[test]
  fn default_loss_rejects_zero_token_batch_after_mask() -> Result<()> {
    // Construct a [B=2, S=2] batch where BOTH rows have lengths=(0, 1).
    // - Shifted targets has T=S-1=1 position; arange runs over [1, 2).
    // - Mask is `steps >= 0 && steps < 1` over step ∈ {1}: never true.
    // With the exclusive `<` upper bound, mask.sum() == 0 → without the
    // zero-token guard, `ce_sum / ntoks` would produce NaN/Inf and poison
    // every downstream accumulator (`train`'s `window_loss`, `evaluate`'s
    // `total_loss`) silently. The guard returns an explicit `Backend`
    // error before the divide so the caller filters the offending rows.
    let model = FakeModel;
    // Two rows, two tokens each; padding is fine since the mask zeros
    // every position out anyway.
    let batch = Array::from_slice::<i32>(&[0, 0, 0, 0], &(2, 2))?;
    // Both rows: offset=0, length=1.
    let lengths = Array::from_slice::<i32>(&[0, 1, 0, 1], &(2, 2))?;
    let err = default_loss(&model, &batch, &lengths)
      .expect_err("expected default_loss to reject zero-token batch");
    match err {
      Error::EmptyInput(p) => {
        assert!(
          p.context().contains("0 supervised tokens"),
          "expected context to mention '0 supervised tokens', got: {}",
          p.context(),
        );
      }
      other => panic!("expected Error::EmptyInput, got: {other:?}"),
    }
    Ok(())
  }

  #[test]
  fn default_loss_rejects_lengths_with_extra_batch_row() -> Result<()> {
    // batch is [B=2, S=2] but lengths is [B+1=3, 2] — a rank-only guard
    // would accept this and silently slice only the first 2 rows,
    // building masks from mismatched metadata. The full-shape guard must
    // reject up-front.
    let model = FakeModel;
    let batch = Array::from_slice::<i32>(&[1, 2, 3, 4], &(2, 2))?;
    let lengths = Array::from_slice::<i32>(&[0, 2, 0, 2, 0, 2], &(3, 2))?;
    let err = default_loss(&model, &batch, &lengths)
      .expect_err("expected ShapePairMismatch for extra length row");
    match err {
      Error::ShapePairMismatch(p) => {
        assert_eq!(p.expected(), &[2_usize, 2_usize][..]);
        assert_eq!(p.actual(), &[3_usize, 2_usize][..]);
      }
      other => panic!("expected Error::ShapePairMismatch, got: {other:?}"),
    }
    Ok(())
  }

  #[test]
  fn default_loss_rejects_lengths_with_missing_batch_row() -> Result<()> {
    // batch is [B=2, S=2] but lengths is [B-1=1, 2] — a rank-only guard
    // would accept this too (rank=2 and dim[1]=2 both held), then the
    // per-row slice would either OOB or silently truncate metadata. The
    // full-shape guard must reject up-front.
    let model = FakeModel;
    let batch = Array::from_slice::<i32>(&[1, 2, 3, 4], &(2, 2))?;
    let lengths = Array::from_slice::<i32>(&[0, 2], &(1, 2))?;
    let err = default_loss(&model, &batch, &lengths)
      .expect_err("expected ShapePairMismatch for missing length row");
    match err {
      Error::ShapePairMismatch(p) => {
        assert_eq!(p.expected(), &[2_usize, 2_usize][..]);
        assert_eq!(p.actual(), &[1_usize, 2_usize][..]);
      }
      other => panic!("expected Error::ShapePairMismatch, got: {other:?}"),
    }
    Ok(())
  }
}
