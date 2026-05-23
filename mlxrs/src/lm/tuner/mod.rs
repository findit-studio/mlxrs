//! Fine-tuning support surface ported from
//! [`mlx_lm.tuner`](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tuner).
//!
//! M3 ships only the **data side** of the tuner subtree —
//! [`datasets`] — since the actual training loop (loss / optimizer /
//! backward pass) is blocked on autograd (the A4 milestone). The dataset
//! types are pre-tokenization (token ids + loss mask only) and therefore
//! have no [`crate::array::Array`] dependency, so they are happily `Send`
//! and pose no `!Send`-handle issues for a future multi-worker data loader.
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
//!
//! # Scope boundary (what the `tuner` subtree is NOT)
//!
//! - The training loop ([`mlx_lm/tuner/trainer.py`]) — needs autograd
//!   (the A4 milestone). When A4 lands, the loss / optimizer / gradient
//!   wiring will sit alongside [`datasets`] in this module.
//! - The HuggingFace Hub dataset loaders (`load_hf_dataset` /
//!   `load_custom_hf_dataset` in `mlx_lm/tuner/datasets.py`) — excluded
//!   per the project's local-only policy (see [`crate::lm::lora`] for the
//!   same fence on adapter loading).
//! - The training-side LoRA *initialization* (random `lora_a` / zero
//!   `lora_b` initializers, `print_trainable_parameters`, the optimizer
//!   wiring) — those live in the same physical Python file as the
//!   inference adapter (`mlx_lm/tuner/lora.py`) but are out-of-scope for
//!   the M3 inference surface and will land alongside the future
//!   training loop.
//!
//! [`mlx_lm/tuner/trainer.py`]: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py

pub mod datasets;
