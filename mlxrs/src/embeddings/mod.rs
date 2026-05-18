//! Embedding utilities — pooling strategies, normalization, similarity,
//! and `sentence-transformers` pooling-config parsing.
//!
//! Pure `Array`-op helpers ported from
//! [`mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings)
//! (`models/pooling.py`, `models/base.py`, `utils.py`) and
//! [`MLXEmbedders`](https://github.com/ml-explore/mlx-swift-examples)
//! (`Pooling.swift`, `MLXArray+Helper.swift`). These operate on the
//! hidden states produced by an embedding model; the model itself,
//! per-architecture loaders, tokenizer integration, model-id registries,
//! ColVision, and `generate`/`load` are added per-usecase and are out of
//! scope here (no-model-arch rule).
//!
//! ## Conventions
//! - `token_embeddings`: `(batch, seq_len, hidden)` float array.
//! - `attention_mask`: `(batch, seq_len)` array, `1` for real tokens, `0`
//!   for padding.
//! - Pooling returns `(batch, hidden)` (except
//!   [`PoolingStrategy::None`](crate::embeddings::PoolingStrategy::None),
//!   which passes the `(batch, seq, hidden)` hidden states through).
//! - No implicit eval: functions compose lazily; call [`crate::Array`]
//!   accessors to materialize.
//!
//! ## Surface
//! - Pooling: [`mean_pooling`](crate::embeddings::mean_pooling),
//!   [`cls_pooling`](crate::embeddings::cls_pooling),
//!   [`max_pooling`](crate::embeddings::max_pooling),
//!   [`last_token_pooling`](crate::embeddings::last_token_pooling),
//!   [`first_token_pooling`](crate::embeddings::first_token_pooling),
//!   plus the unified
//!   [`PoolingStrategy`](crate::embeddings::PoolingStrategy) enum +
//!   [`pool`](crate::embeddings::pool) dispatcher (mirrors python
//!   `pool_by_config` + swift `Pooling.callAsFunction`).
//! - Normalization: parameterized
//!   [`normalize`](crate::embeddings::normalize()) (real
//!   `mlx_linalg_norm` `ord=p`),
//!   [`l2_normalize`](crate::embeddings::l2_normalize) /
//!   [`l2_normalize_eps`](crate::embeddings::l2_normalize_eps)
//!   convenience, eps constants
//!   [`DEFAULT_NORMALIZE_EPS`](crate::embeddings::DEFAULT_NORMALIZE_EPS)
//!   (python `1e-9`) and
//!   [`SWIFT_L2_EPS`](crate::embeddings::SWIFT_L2_EPS) (`1e-12`).
//! - Fused post-pool norms (mlx-c-surfaced), applied by the
//!   [`pool`](crate::embeddings::pool) dispatcher to the *already-pooled*
//!   sentence vector (after the pooling reduction, before matryoshka
//!   truncation / L2-normalize — matching swift `Pooling`'s
//!   `applyLayerNorm` on the pooled output, *not* the model's internal
//!   token-level normalization, which is per-architecture and out of
//!   scope): [`layer_norm`](crate::embeddings::layer_norm)
//!   (`mlx_fast_layer_norm`),
//!   [`rms_norm`](crate::embeddings::rms_norm) (`mlx_fast_rms_norm`).
//! - ST-config parsing:
//!   [`pooling_from_st_config_str`](crate::embeddings::pooling_from_st_config_str) /
//!   [`pooling_from_st_config_bytes`](crate::embeddings::pooling_from_st_config_bytes) /
//!   [`pooling_from_st_config_path`](crate::embeddings::pooling_from_st_config_path)
//!   → [`StPoolingConfig`](crate::embeddings::StPoolingConfig).
//! - Similarity:
//!   [`cosine_similarity`](crate::embeddings::cosine_similarity),
//!   [`cosine_similarity_matrix`](crate::embeddings::cosine_similarity_matrix).

pub mod config;
pub mod fast;
pub mod normalize;
pub mod pooling;
pub mod similarity;

use crate::{array::Array, error::Result, ops::misc::astype};

/// Build a `(1,)` scalar constant carrying `value`, in the **same dtype**
/// as `like` (`like.dtype()`).
///
/// This is the crate's uniform stand-in for MLX *weak-scalar* / python
/// `astype(x.dtype)` semantics: every constant or `-inf`/`eps`/`0` floor
/// that meets the embedding tensor must adopt the embedding's dtype so a
/// f16/bf16 input is **not** silently promoted to f32 (Codex round-4
/// systemic dtype-fidelity finding). `mlx-embeddings` does exactly this
/// (`mask.astype(token_embeddings.dtype)`, and python scalars `-inf` /
/// `eps` / `1e-9` are MLX weak scalars that adopt the array dtype).
///
/// Implemented as f32 scalar → [`astype`] to `like.dtype()`. For a f32
/// `like` this is a dtype-preserving no-op cast, so f32 results are
/// **bit-identical** to a direct `Array::full::<f32>` (regression-safe).
fn scalar_like(value: f32, like: &Array) -> Result<Array> {
  // C2 (Copilot review 4307622782, #3256688235): `Array::full` calls the
  // fallible `mlx_array_new_float32` ctor BEFORE its `mlx_full` call (whose
  // `default_stream()` arg is what runs `ensure_handler_installed()`), so
  // with the eager `#[ctor]` stripped that first ctor could reach mlx-c
  // with no error handler installed → mlx-c's default `printf + exit(-1)`
  // instead of a recoverable `Err`. Defense-in-depth per the #13/#24
  // crate-wide contract (same as `ops/linalg_full.rs` resolving its CPU
  // stream first): install the handler at the entry point, before any
  // fallible scalar construction. Bounded to `scalar_like` (the #20-scope
  // call site); `Array::full`'s internal ordering is out of scope here.
  crate::error::ensure_handler_installed();
  let s = Array::full::<f32>(&(1,), value)?;
  astype(&s, like.dtype()?)
}

pub use config::{
  StPoolingConfig, pooling_from_st_config_bytes, pooling_from_st_config_path,
  pooling_from_st_config_str,
};
pub use fast::{layer_norm, rms_norm};
pub use normalize::{
  DEFAULT_NORMALIZE_EPS, SWIFT_L2_EPS, l2_normalize, l2_normalize_eps, normalize,
};
pub use pooling::{
  LAYER_NORM_EPS, PoolingStrategy, RMS_NORM_EPS, cls_pooling, first_token_pooling,
  last_token_pooling, max_pooling, mean_pooling, pool, truncate_last_dim,
};
pub use similarity::{cosine_similarity, cosine_similarity_matrix};
