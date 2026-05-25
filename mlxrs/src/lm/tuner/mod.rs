//! Fine-tuning support surface ported from
//! [`mlx_lm.tuner`](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tuner).
//!
//! M3 ships the **data side** of the tuner subtree ([`datasets`]), the
//! **training loop + optimizers** ([`trainer`] + [`optimizers`], the port
//! of `mlx_lm/tuner/trainer.py` + `mlx_lm/tuner/optimizers/*.py` now that
//! the autograd FFI (#204) is wired), and the **distillation losses**
//! ([`losses`] — `kl_div_loss` / `js_div_loss`, the port of
//! `mlx_lm/tuner/losses.py` now that custom Metal kernels (#205) are
//! wired). The dataset types are pre-tokenization (token ids + loss mask
//! only) and therefore have no [`crate::array::Array`] dependency, so they
//! are happily `Send` and pose no `!Send`-handle issues for a future
//! multi-worker data loader; the loss + optimizer types DO own
//! [`crate::array::Array`] handles so they inherit the same
//! `!Send + !Sync` constraint as the rest of mlxrs.
//!
//! **Inference-time** LoRA / DoRA *adapter loading* (the runtime surface
//! that applies a pre-trained adapter to a base model's weight map) is in
//! [`crate::lm::lora`] — that is intentionally a sibling of `tuner`, since
//! it is consumed at inference time, not at training time.
//!
//! # Scope (what the `tuner` subtree IS)
//!
//! - [`datasets`] — local jsonl-backed dataset types
//!   ([`datasets::TextDataset`], [`datasets::ChatDataset`],
//!   [`datasets::CompletionsDataset`], [`datasets::ConcatenatedDataset`],
//!   [`datasets::CacheDataset`]) and the [`datasets::load_dataset`] entry
//!   point that auto-detects the right shape from the jsonl content.
//! - [`trainer`] — the training loop driver (mechanics-only / v1) with
//!   gradient accumulation, LR schedule wiring, and pluggable callbacks.
//!   Callers must opt in via
//!   [`trainer::TrainingArgs::acknowledge_no_real_gradients`] until the
//!   future `Module` trait enables real `value_and_grad`.
//! - [`optimizers`] — 10 optimizers ported from `mlx_lm/tuner/optimizers/`
//!   (SGD, RMSprop, Adagrad, AdaDelta, Adam, AdamW, Adamax, Lion,
//!   Adafactor, Muon) + `MultiOptimizer` + `clip_grad_norm`.
//! - [`losses`] — distillation training losses
//!   ([`losses::kl_div_loss`], [`losses::js_div_loss`]) with hand-written
//!   Metal kernels for the forward AND backward passes and a custom VJP
//!   that wires those kernels into autograd.
//!
//! # Scope boundary (what the `tuner` subtree is NOT)
//!
//! - The HuggingFace Hub dataset loaders (`load_hf_dataset` /
//!   `load_custom_hf_dataset` in `mlx_lm/tuner/datasets.py`) — excluded
//!   per the project's local-only policy (see [`crate::lm::lora`] for the
//!   same fence on adapter loading).
//! - The training-side LoRA *initialization* (random `lora_a` / zero
//!   `lora_b` initializers, `print_trainable_parameters`, the optimizer
//!   wiring) — those live in the same physical Python file as the
//!   inference adapter (`mlx_lm/tuner/lora.py`) but are out-of-scope for
//!   the M3 inference surface and will land alongside a future PR that
//!   builds on [`trainer`].
//!
//! [`mlx_lm/tuner/trainer.py`]: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py

pub mod datasets;
pub mod losses;
pub mod optimizers;
pub mod trainer;
