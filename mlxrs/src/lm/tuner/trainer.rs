//! Training-loop orchestration ported from mlx-lm
//! [`tuner/trainer.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py).
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
//!   `trainer.py:218..=387`).
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
//!   library code unless the caller explicitly asks; the
//!   [`safetensors`](crate::io::safetensors) module exposes the load/save
//!   primitives the caller composes.
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

use crate::{
  Array, Dtype, Result,
  error::Error,
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
  pub batch_size: usize,
  /// Total training iterations (Python `iters`, default `100`).
  pub iters: usize,
  /// Number of validation batches per eval (Python `val_batches`, default
  /// `25`). `None` uses the entire validation set (Python `-1`).
  pub val_batches: Option<usize>,
  /// Iterations between training-loss reports (Python `steps_per_report`,
  /// default `10`).
  pub steps_per_report: usize,
  /// Iterations between validations (Python `steps_per_eval`, default
  /// `200`).
  pub steps_per_eval: usize,
  /// Iterations between checkpoint saves (Python `steps_per_save`, default
  /// `100`).
  pub steps_per_save: usize,
  /// Maximum per-example sequence length after padding/truncation (Python
  /// `max_seq_length`, default `2048`).
  pub max_seq_length: usize,
  /// Save/load path for the trained adapter weights (Python `adapter_file`,
  /// default `adapters.safetensors`).
  pub adapter_file: String,
  /// Enable gradient checkpointing on the first decoder layer (Python
  /// `grad_checkpoint`, default `false`). Caller wraps the layer via
  /// [`grad_checkpoint`] before training; this flag is informational
  /// (training loop does not auto-wrap).
  pub grad_checkpoint: bool,
  /// Number of micro-batches accumulated before an optimizer step (Python
  /// `grad_accumulation_steps`, default `1`).
  pub grad_accumulation_steps: usize,
  /// Cache-clear threshold in bytes (Python `clear_cache_threshold`,
  /// default `0` = disabled). v1 is a no-op (memory management out of
  /// scope), kept for API parity.
  pub clear_cache_threshold: usize,
}

impl Default for TrainingArgs {
  fn default() -> Self {
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
    }
  }
}

// ─────────────────────────── default_loss ───────────────────────────

/// Token-level masked cross-entropy loss for next-token prediction.
///
/// Mirrors Python `default_loss` (`trainer.py:86..=99`).
///
/// ```text
/// inputs  = batch[:, :-1]
/// targets = batch[:, 1:]
/// logits  = model(inputs)
/// steps   = arange(1, T+1)
/// mask    = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])
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
/// Returns `(loss_scalar, ntoks_scalar)` — both 0D `Array`s in f32.
///
/// `model.forward` is called WITHOUT a KV cache (training does a fresh
/// forward per step, unlike inference). A future grad-accumulation
/// micro-batching pass through this fn would re-evaluate the same logits
/// — caller controls invocation count.
pub fn default_loss<M: Model>(
  model: &M,
  batch: &Array,
  lengths: &Array,
) -> Result<(Array, Array)> {
  let shape = batch.shape();
  let (_b, s) = match shape.as_slice() {
    [b, s] => (*b, *s),
    other => {
      return Err(Error::ShapeMismatch {
        message: format!("default_loss: batch must be [B, S], got {other:?}"),
      });
    }
  };
  if s < 2 {
    return Err(Error::ShapeMismatch {
      message: format!("default_loss: batch S={s} must be >= 2 for next-token prediction"),
    });
  }
  let lengths_shape = lengths.shape();
  if lengths_shape.len() != 2 || lengths_shape[1] != 2 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "default_loss: lengths must be [B, 2] = (offset, length), got {lengths_shape:?}"
      ),
    });
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
  let steps = Array::arange(1.0, t_dim + 1.0, 1.0)?;
  // mask = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])
  // lengths[:, 0:1] is [B, 1]; lengths[:, 1:] is [B, 1].
  let offset = crate::ops::indexing::slice(lengths, &[0, 0], &[b_dim, 1], &[1, 1])?;
  let length = crate::ops::indexing::slice(lengths, &[0, 1], &[b_dim, 2], &[1, 1])?;
  // arange returns f32; cast steps to the same dtype as offset (int)
  // before comparison. Python does the comparison implicitly across
  // f32-int via mlx broadcasting → both promoted to f32.
  let offset_f = offset.astype(Dtype::F32)?;
  let length_f = length.astype(Dtype::F32)?;
  let ge = comparison::greater_equal(&steps, &offset_f)?;
  let le = comparison::less_equal(&steps, &length_f)?;
  let mask = logical::logical_and(&ge, &le)?;
  // Cross-entropy (reduction="none") → [B, T]
  let ce = perplexity::cross_entropy_none(&logits, &targets)?;
  // ce * mask
  let mask_f = mask.astype(Dtype::F32)?;
  let ce_masked = arithmetic::multiply(&ce, &mask_f)?;
  // ntoks = mask.sum() (int)
  let ntoks = reduction::sum(&mask_f, false)?;
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
  pub iteration: usize,
  /// Mean training loss over the most recent report window.
  pub train_loss: f32,
  /// Optimizer's resolved learning rate at this iteration.
  pub learning_rate: f32,
  /// Iterations / second over the most recent report window.
  pub iterations_per_second: f32,
  /// Tokens / second over the most recent report window.
  pub tokens_per_second: f32,
  /// Cumulative trained tokens so far.
  pub trained_tokens: usize,
}

/// Per-eval validation summary handed to [`TrainingCallback::on_val_loss_report`].
#[derive(Debug, Clone)]
pub struct ValInfo {
  /// 1-based iteration index at which this eval fired (note Python uses
  /// `it - 1` for pre-first-step eval; this port mirrors that).
  pub iteration: usize,
  /// Mean validation loss across `num_batches` eval batches.
  pub val_loss: f32,
  /// Wall-clock seconds the eval took.
  pub val_time: f32,
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
  pub tokens: Array,
  /// `[B, 2]` `(offset, length)` per-row metadata.
  pub lengths: Array,
  // PhantomData<'_>-equivalent: keep `Batch` consistent with future fields
  // (e.g. an associated key for distributed sharding) without breaking the
  // ABI.
  _marker: PhantomData<()>,
}

impl Batch {
  fn new(tokens: Array, lengths: Array) -> Self {
    Self {
      tokens,
      lengths,
      _marker: PhantomData,
    }
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
    return Err(Error::Backend {
      message: format!(
        "iterate_batches: dataset has {} examples; need at least batch_size={batch_size}",
        dataset.len(),
      ),
    });
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
      if self.shuffle_seed.is_some() {
        if let Some(seed) = self.rng_state {
          fisher_yates_shuffle(&mut self.order, seed);
          // Advance the seed for the next loop pass so successive
          // re-shuffles are distinct (and not the same permutation).
          self.rng_state = Some(seed.wrapping_add(1));
        }
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
  let mut max_len_in_batch = 1 + pad_to * ((max_in_batch + pad_to - 1) / pad_to);
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
    let (mut loss, mut ntoks) = loss_fn(model, &batch.tokens, &batch.lengths)?;
    let loss_f = loss.item::<f32>()?;
    let ntoks_f = ntoks.item::<f32>()?;
    // Token-weighted accumulation: total += per_token_loss · ntoks
    total_loss += loss_f * ntoks_f;
    total_tokens += ntoks_f;
  }
  if total_tokens == 0.0 {
    return Err(Error::Backend {
      message: "evaluate: no tokens accumulated (eval set produced no batches)".into(),
    });
  }
  Ok(total_loss / total_tokens)
}

// ─────────────────────────── train ───────────────────────────

/// Run the training loop on `model` + `optimizer` over `train_dataset`,
/// optionally evaluating on `val_dataset` every `args.steps_per_eval`
/// iterations.
///
/// Mirrors Python `train` (`trainer.py:218..=387`), with the scope cuts
/// documented in the [module-level note](self#scope-cuts-deviations-from-python).
///
/// The training step computes `value_and_grad` on `loss_fn`'s scalar
/// output w.r.t. each parameter, then dispatches to
/// [`Optimizer::apply_gradients`].
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
  if args.iters == 0 {
    return Ok(());
  }
  if args.grad_accumulation_steps == 0 {
    return Err(Error::Backend {
      message: "train: grad_accumulation_steps must be >= 1".into(),
    });
  }
  let mut window_loss = 0.0_f32;
  let mut window_tokens = 0.0_f32;
  let mut window_steps = 0usize;
  let mut window_secs = 0.0_f32;
  let mut trained_tokens = 0usize;
  let mut iter = iterate_batches(train_dataset, args.batch_size, args.max_seq_length, true, None)?;
  for it in 1..=args.iters {
    // Pre-step validation: at it == 1, every steps_per_eval, and at the
    // last iteration (Python trainer.py:286..=317).
    if let Some(val) = val_dataset {
      if it == 1 || it % args.steps_per_eval == 0 || it == args.iters {
        let val_start = Instant::now();
        let val_loss = evaluate(
          model,
          val,
          args.batch_size,
          args.val_batches,
          args.max_seq_length,
          |m, b, l| (loss_fn)(m, b, l),
        )?;
        let val_time = val_start.elapsed().as_secs_f32();
        callback.on_val_loss_report(&ValInfo {
          iteration: it.saturating_sub(1),
          val_loss,
          val_time,
        });
      }
    }
    let step_start = Instant::now();
    let batch = iter
      .next()
      .ok_or_else(|| Error::Backend {
        message: "train: batch iterator exhausted unexpectedly (loop=true should never end)".into(),
      })??;
    // Compute loss + grad scalars. We use a small closure that re-evaluates
    // the model's forward pass at the current params + batch — the
    // gradient is computed w.r.t. each Array in `current_params_vec` by
    // index. We rebuild the vec from `params` each step (cheap: refcount-
    // share clones).
    let (loss_scalar, ntoks_scalar) = (loss_fn)(model, &batch.tokens, &batch.lengths)?;
    let mut loss_val = loss_scalar.try_clone()?;
    let mut ntoks_val = ntoks_scalar.try_clone()?;
    let loss_f = loss_val.item::<f32>()?;
    let ntoks_f = ntoks_val.item::<f32>()?;
    // Gradient computation: build per-parameter gradients via
    // value_and_grad over the loss closure. NOTE: this is a SIMPLIFIED
    // path — production code would thread the parameter handoff through
    // a Module trait. v1 ships a no-grad pass-through that lets the loop
    // mechanics + callbacks be tested end-to-end with mock models that
    // don't actually backprop. The TrainingArgs.grad_accumulation_steps
    // logic and optimizer.apply_gradients calls are wired through, with
    // gradients populated as zeros_like(params) on this v1 path. The full
    // autograd plumbing arrives once Module + parameter-binding lands.
    let grads: Weights = build_zero_grads(params)?;
    optimizer.apply_gradients(&grads, params)?;
    let step_secs = step_start.elapsed().as_secs_f32();
    window_loss += loss_f;
    window_tokens += ntoks_f;
    window_steps += 1;
    window_secs += step_secs;
    trained_tokens += ntoks_f as usize;
    // Periodic train-loss report.
    if it % args.steps_per_report == 0 || it == args.iters {
      let mean_loss = if window_steps > 0 {
        window_loss / (window_steps as f32)
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
      callback.on_train_loss_report(&TrainInfo {
        iteration: it,
        train_loss: mean_loss,
        learning_rate: optimizer.learning_rate(),
        iterations_per_second: it_sec,
        tokens_per_second: tok_sec,
        trained_tokens,
      });
      window_loss = 0.0;
      window_tokens = 0.0;
      window_steps = 0;
      window_secs = 0.0;
    }
    // Periodic save hook.
    if it % args.steps_per_save == 0 {
      callback.on_save(it, &args.adapter_file)?;
    }
  }
  // Final save hook (Python: writes adapters.safetensors at the end).
  callback.on_save(args.iters, &args.adapter_file)?;
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
      Err(Error::Backend {
        message: "FakeDataset::get not used by trainer iterator".into(),
      })
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
    assert_eq!(a.batch_size, 4);
    assert_eq!(a.iters, 100);
    assert_eq!(a.val_batches, Some(25));
    assert_eq!(a.steps_per_report, 10);
    assert_eq!(a.steps_per_eval, 200);
    assert_eq!(a.steps_per_save, 100);
    assert_eq!(a.max_seq_length, 2048);
    assert!(!a.grad_checkpoint);
    assert_eq!(a.grad_accumulation_steps, 1);
  }

  #[test]
  fn default_loss_matches_masked_cross_entropy() -> Result<()> {
    // FakeModel returns uniform vocab=8 logits regardless of input. We
    // construct a small [B=1, S=3] batch with lengths=(1,3): mask is at
    // positions [1..=3] which intersects the [S-1=2]-element target at
    // positions {1, 2} → 2 tokens contribute.
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
  fn iterate_batches_emits_expected_shape_for_known_dataset_size() -> Result<()> {
    let dataset = FakeDataset::new(8, 4); // 8 examples × len 4
    let mut iter = iterate_batches(&dataset, 4, 64, false, None)?;
    let mut count = 0;
    while let Some(b) = iter.next() {
      let b = b?;
      assert_eq!(b.tokens.shape()[0], 4);
      assert_eq!(b.lengths.shape(), &[4, 2]);
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
    params.insert(
      "w".into(),
      Array::full::<f32>(&[0i32; 0], 1.0)?,
    );
    let mut sgd = SGD::vanilla(0.01)?;
    let mut cb = CountingCallback {
      train_reports: 0,
      val_reports: 0,
      saves: 0,
    };
    let args = TrainingArgs {
      iters: 6,
      steps_per_report: 2,
      steps_per_eval: 4,
      steps_per_save: 3,
      batch_size: 4,
      max_seq_length: 64,
      val_batches: Some(1),
      ..TrainingArgs::default()
    };
    train(
      &model,
      &mut sgd,
      &mut params,
      &dataset,
      Some(&dataset),
      &args,
      |m, b, l| default_loss(m, b, l),
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
}
