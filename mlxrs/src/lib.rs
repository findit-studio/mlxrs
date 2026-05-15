//! mlxrs — safe Rust bindings for [MLX](https://github.com/ml-explore/mlx) on Apple silicon.
//!
//! M1 ships `Array` + `Dtype` + `Error` + a subset of `ops.h`. See
//! `docs/superpowers/specs/` for the full design.
//!
//! ## Caveats
//! - `Array` is **`!Send` and `!Sync`** in M1 — single-thread use only. Cross-thread
//!   sharing requires care: the underlying C++ `array_desc` is shared by `Clone`
//!   and mutates non-atomic state internally. M2 will provide a `SharedArray`
//!   newtype (`Arc<Mutex<Array>>`-style) with a documented cross-thread contract.
//! - **Async Metal kernel failures bypass `Result<T, Error>` and abort the process.**
//!   The rc/sentinel chain only catches synchronous errors. Recovery via
//!   `set_terminate` shim is M2 work.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![cfg_attr(not(test), deny(missing_docs))]

pub use array::Array;
pub use dtype::{Dtype, Element};
pub use error::{Error, Result};
pub use shape::IntoShape;
pub use version::version;

pub mod array;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod shape;
pub mod version;

/// Language Model (LM) — text-only inference. Stub in M1; port lands in M3
/// (loader, tokenizer, sampling, generation loop). Per-model architectures
/// (Llama/Qwen/Mistral/etc.) are added per-usecase, not bulk-ported from
/// mlx-lm/models/.
#[cfg(feature = "lm")]
#[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
pub mod lm;

/// Vision-Language Model (VLM) — multimodal inference. Stub in M1; port lands
/// in M4 (image processors, chat-template shims, loader). Per-model
/// architectures (Qwen-VL/LLaVA/etc.) are added per-usecase, not bulk-ported
/// from mlx-vlm/models/.
#[cfg(feature = "vlm")]
#[cfg_attr(docsrs, doc(cfg(feature = "vlm")))]
pub mod vlm;

/// Audio (TTS/STT/STS) — speech inference. Stub in M1; port lands in M5
/// (audio I/O, pipeline scaffolding). Per-model architectures
/// (Whisper/Sesame/etc.) are added per-usecase, not bulk-ported from
/// mlx-audio/models/.
#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio;

/// Embedding utilities — high-level loading, tokenizer integration, pooling,
/// and similarity. Stub in M1; port lands in M3 alongside tokenizers +
/// model-loading. Per-model architectures (BERT/XLM-RoBERTa/Qwen3-embed/etc.)
/// are added per-usecase, not bulk-ported from mlx-embeddings/models/.
#[cfg(feature = "embeddings")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddings")))]
pub mod embeddings;

// ───── internal modules below ─────
pub(crate) mod stream; // INTERNAL: M2 lifts to public Stream
