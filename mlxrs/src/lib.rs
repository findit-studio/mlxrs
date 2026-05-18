//! mlxrs — safe Rust bindings for [MLX](https://github.com/ml-explore/mlx) on Apple silicon.
//!
//! M1 ships `Array` + `Dtype` + `Error` + a subset of `ops.h`. See
//! `docs/superpowers/specs/` for the full design.
//!
//! ## Caveats
//! - `Array` is **`!Send` and `!Sync`** — single-thread use only, like MLX's
//!   own C++/Python/Swift APIs (which deliberately do not share arrays across
//!   threads). The underlying C++ `array_desc` is refcount-shared by `Clone`
//!   and mutates non-atomic state internally, and mlx's `eval` is itself not
//!   concurrency-safe. There is **no shared-array wrapper**: to use array
//!   data on another thread, extract owned data via [`Array::to_vec`] /
//!   [`Array::item`] (which yield `Send` values) and move that.
//! - **Async Metal kernel failures bypass `Result<T, Error>` and abort the
//!   process.** The rc/sentinel chain only catches synchronous errors. A
//!   `set_terminate`-style recovery shim is **not implementable** (mlx-c
//!   exposes no hook) and is deferred to M3+ (diagnostics-only is planned).
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![cfg_attr(not(test), deny(missing_docs))]

pub use array::Array;
pub use device::{Device, DeviceKind};
pub use dtype::{Dtype, Element};
pub use error::{Error, Result};
pub use shape::IntoShape;
pub use stream::Stream;
pub use version::version;

pub mod array;
pub mod device;
pub mod diagnostics;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod shape;
pub mod stream;
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

/// Embedding utilities — pooling strategies (+ unified dispatcher),
/// parameterized normalization, fused post-pool LayerNorm/RMSNorm
/// (applied to the pooled sentence vector, matching swift `Pooling`'s
/// `pool → optional layer/rms-norm → optional matryoshka truncation →
/// optional L2-normalize` pipeline; *not* token-level pre-pool
/// normalization, which is part of the model architecture and out of
/// scope), `sentence-transformers` pooling-config parsing, and similarity.
/// Ported (M3) from `mlx-embeddings` (`models/pooling.py`,
/// `models/base.py`, `utils.py`) and swift `MLXEmbedders`
/// (`Pooling.swift`, `MLXArray+Helper.swift`). Per-model architectures
/// (BERT/XLM-RoBERTa/Qwen3-embed/etc.), loaders, tokenizer integration,
/// model-id registries, and `generate`/`load` are out of scope
/// (no-model-arch rule).
#[cfg(feature = "embeddings")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddings")))]
pub mod embeddings;

/// Tokenizer support — HF `tokenizers` integration, streaming detokenizers,
/// chat-template rendering, and tool-call parsing. Port lands in M3, ported
/// from `mlx-lm`'s `tokenizer_utils.py` + `chat_templates/` + `tool_parsers/`
/// and cross-referenced against `mlx-swift-lm`'s `MLXLMCommon` tokenizer /
/// tool abstractions. Model-specific tokenizer registration (the Python
/// `NewlineTokenizer`) is per-model architecture and intentionally out of
/// scope. Enabled transitively by `lm`, `vlm`, and `embeddings`.
#[cfg(feature = "tokenizer")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer")))]
pub mod tokenizer;

/// Operator overloads (`&a + &b`, `&a - &b`, `&a * &b`, `&a / &b`, `-&a`).
/// Gated; OFF by default. Panics on shape/dtype error — see module docs.
#[cfg(feature = "unstable-ops-overload")]
#[cfg_attr(docsrs, doc(cfg(feature = "unstable-ops-overload")))]
pub mod ops_traits;
