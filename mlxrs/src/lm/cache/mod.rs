//! Key/value caches for incremental decoding, ported from
//! [`mlx_lm.models.cache`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
//! (`KVCache` / `ConcatenateKVCache` / `RotatingKVCache`) and cross-checked
//! against mlx-swift-lm's `MLXLMCommon` `KVCache` protocol.
//!
//! A cache holds the per-layer attention keys/values seen so far so each
//! decode step only re-computes the new token. One [`KvCache`] exists per
//! decoder layer; [`make_prompt_cache`] builds the vector the
//! [`Model`](crate::lm::model::Model) mutates in place.
//!
//! The surface mirrors the reference structure faithfully: a [`KvCache`]
//! trait (mlx-lm `_BaseCache` / swift `KVCache` protocol) with the
//! refinement traits [`QuantizedKvCache`] (swift `QuantizedKVCacheProtocol`)
//! and [`BatchPositionedKvCache`] (swift `BatchPositionedKVCache`), plus the
//! [`RopeOffset`] (swift `RoPEOffset`) and [`MaskMode`] (swift
//! `ScaledDotProductAttentionMaskMode`) enums. The concrete caches beyond
//! [`StandardKvCache`] / [`RotatingKvCache`] (Chunked / Quantized / Arrays /
//! CacheList / Batch / prompt) land in later PRs; [`from_state`] is left
//! extensible for them.
//!
//! [`StandardKvCache`] is the `KVCache` port: mlx-lm's `step`-sized
//! over-allocated buffer is a pure allocation optimization with **no**
//! effect on what that cache returns, so — `mlxrs::Array` being functional
//! (no in-place buffer slicing) — it reproduces the observable semantics
//! directly via `concatenate`/`slice` (exactly mlx-lm's `ConcatenateKVCache`,
//! the documented twin with identical observable behavior minus the step
//! buffer).
//!
//! [`RotatingKvCache`] is **not** so simplifiable: mlx-lm's
//! `RotatingKVCache` overwrites slots *in place* at a ring cursor, so the
//! returned buffer is in **physical ring order**, which an attention mask
//! built the mlx-lm way depends on. It is therefore a literal 1:1 port of
//! `RotatingKVCache` — including its `_idx` cursor, the distinct
//! `_update_in_place` (S==1) / `_update_concat` (S>1) paths, and the step
//! buffer (emulated with placeholder rows whose values are provably
//! overwritten or sliced off before any observer, so the physical order and
//! every grow/trim/rotate branch match mlx-lm byte-for-byte).
//!
//! No implicit eval: every op is a pure [`crate::ops`] composition
//! returning a `Result`.

use crate::{
  array::Array,
  error::{Error, Result},
};

mod chunked;
mod mask;
mod rotating;
mod standard;
mod util;

pub use chunked::*;
pub use mask::{create_attention_mask, create_causal_mask};
pub use rotating::RotatingKvCache;
pub use standard::StandardKvCache;

/// mlx-lm's default `RotatingKVCache.keep` for sliding-window models
/// (`make_prompt_cache(...) -> RotatingKVCache(max_size=..., keep=4)`).
pub const ROTATING_DEFAULT_KEEP: i32 = 4;

/// Offset to use when applying rotary position embeddings — mlx-swift-lm's
/// `RoPEOffset` (a scalar position for the common case, or a per-sequence
/// `[B]` array for batched caches).
pub enum RopeOffset {
  /// A single scalar RoPE position (the offset all sequences share).
  Scalar(usize),
  /// Per-sequence RoPE offsets with shape `[B]` (batched caches).
  Batch(Array),
}

/// Attention-mask mode returned by [`KvCache::make_mask`] — mlx-swift-lm's
/// `ScaledDotProductAttentionMaskMode`, equivalently mlx-lm's
/// `None | "causal" | array` triad.
pub enum MaskMode {
  /// No mask (mlx-lm `None`; e.g. single-token decode).
  None,
  /// The implicit causal mask (mlx-lm's `"causal"` sentinel — no array
  /// materialized).
  Causal,
  /// An explicit boolean/additive mask array.
  Array(Array),
}

/// A quantized key or value triple `(weight, scales, biases)` — the
/// `biases` is optional (mlx-swift-lm `(MLXArray, MLXArray, MLXArray?)`).
pub type QTriple = (Array, Array, Option<Array>);

/// The per-layer KV cache contract — mlx-lm's `_BaseCache` /
/// mlx-swift-lm's `KVCache` protocol.
///
/// Concrete caches ([`StandardKvCache`], [`RotatingKvCache`], and the
/// later-PR Chunked/Quantized/… types) implement this; the generation loop
/// and `make_prompt_cache` work uniformly over `Box<dyn KvCache>`.
pub trait KvCache {
  /// The current cached offset (mlx-lm `cache.offset` — the raw position
  /// the attention mask / RoPE use).
  fn offset(&self) -> usize;

  /// Offset to apply with rotary position embeddings — swift
  /// `KVCache.ropeOffset`. This default auto-dispatches through the
  /// batch-positioned refinement: if [`as_batch_positioned`](
  /// KvCache::as_batch_positioned) is `Some`, it yields
  /// `RopeOffset::Batch(batch_offset())`; otherwise the scalar
  /// [`offset`](KvCache::offset). This mirrors mlx-swift-lm's
  /// `BatchPositionedKVCache` `ropeOffset` protocol extension
  /// (`extension BatchPositionedKVCache { var ropeOffset { .batch(...) } }`),
  /// whose automatic conformance Rust lacks — so a P5 batch cache need only
  /// implement [`as_batch_positioned`](KvCache::as_batch_positioned) /
  /// [`BatchPositionedKvCache::batch_offset`] and inherits the correct
  /// `Batch` RoPE offset here without re-overriding `rope_offset`.
  ///
  /// Fallible because the `Batch` arm clones an owned [`Array`] and #33
  /// removes the infallible `impl Clone for Array` (only
  /// [`Array::try_clone`] remains); Swift's `ropeOffset` is non-throwing,
  /// but the user-approved fallible-now choice surfaces that as a `Result`.
  /// Today this is zero behavior change: no cache implements
  /// `as_batch_positioned` (its default returns `None`), so every current
  /// cache still yields `Ok(RopeOffset::Scalar(offset))`.
  fn rope_offset(&self) -> Result<RopeOffset> {
    match self.as_batch_positioned() {
      Some(bp) => Ok(RopeOffset::Batch(bp.batch_offset()?)),
      None => Ok(RopeOffset::Scalar(self.offset())),
    }
  }

  /// The maximum retained window length, if bounded (swift
  /// `KVCache.maxSize`; `None` for full-attention caches).
  fn max_size(&self) -> Option<usize> {
    None
  }

  /// Append `keys`/`values` and return the cache's current `(keys,
  /// values)` (mlx-lm `cache.update_and_fetch` / swift `update`).
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)>;

  /// The serializable array state (mlx-lm `cache.state`); `[]` when the
  /// cache holds nothing.
  fn state(&self) -> Result<Vec<Array>>;

  /// Restore the array state (mlx-lm `cache.state` setter).
  fn set_state(&mut self, state: Vec<Array>) -> Result<()>;

  /// Serializable scalar metadata as strings (mlx-lm `cache.meta_state`);
  /// empty for caches without metadata.
  fn meta_state(&self) -> Vec<String> {
    Vec::new()
  }

  /// Restore scalar metadata (mlx-lm `cache.meta_state` setter). The
  /// default (no metadata) is a no-op.
  fn set_meta_state(&mut self, _m: &[String]) -> Result<()> {
    Ok(())
  }

  /// Whether the cache can be trimmed (mlx-lm `cache.is_trimmable`).
  fn is_trimmable(&self) -> bool {
    false
  }

  /// Drop the most recent `min(offset, n)` tokens; returns the number
  /// trimmed (mlx-lm `cache.trim`). The default (non-trimmable) trims 0.
  fn trim(&mut self, _n: usize) -> Result<usize> {
    Ok(0)
  }

  /// Build the attention mask for `n` new tokens (mlx-lm
  /// `cache.make_mask` / swift `makeMask`).
  fn make_mask(&self, n: usize, window_size: Option<usize>, return_array: bool)
  -> Result<MaskMode>;

  /// Size of the cache in bytes (mlx-lm `cache.nbytes`).
  fn nbytes(&self) -> usize;

  /// Whether the cache holds no keys yet (mlx-lm `cache.empty()`).
  fn is_empty(&self) -> bool;

  /// An independent deep copy (mlx-lm `copy.deepcopy` / swift `copy()`).
  ///
  /// Swift's `copy()` is infallible (array COW); in Rust the underlying
  /// [`Array::try_clone`] is fallible (allocation / backend), so this
  /// surfaces that as a `Result` — the same no-implicit-eval /
  /// fallible-array-clone principle the rest of this module follows. A
  /// clone failure is propagated as an [`Error`], **never** swallowed into
  /// a half-populated cache (silent corruption) and **never** turned into a
  /// panic.
  fn copy(&self) -> Result<Box<dyn KvCache>>;

  /// Downcast to the quantized refinement, if this cache is quantized
  /// (swift `cache as? QuantizedKVCacheProtocol`). Default: not quantized.
  fn as_quantized(&self) -> Option<&dyn QuantizedKvCache> {
    None
  }

  /// Downcast to the batched-position refinement, if applicable (swift
  /// `cache as? BatchPositionedKVCache`). Default: scalar-positioned.
  fn as_batch_positioned(&self) -> Option<&dyn BatchPositionedKvCache> {
    None
  }
}

/// Caches that support efficient quantized operations — mlx-swift-lm's
/// `QuantizedKVCacheProtocol` (mlx-lm `QuantizedKVCache`). The concrete
/// implementation lands in a later PR.
pub trait QuantizedKvCache: KvCache {
  /// The quantization group size (mlx-lm `QuantizedKVCache.group_size`).
  fn group_size(&self) -> i32;

  /// The number of quantization bits (mlx-lm `QuantizedKVCache.bits`).
  fn bits(&self) -> i32;

  /// Update and return quantized `((w, scales, biases), …)` tuples
  /// (swift `updateQuantized`).
  fn update_quantized(&mut self, keys: &Array, values: &Array) -> Result<(QTriple, QTriple)>;

  /// The current quantized state without updating, or `None` if empty
  /// (swift `getQuantizedState`).
  fn quantized_state(&self) -> Result<Option<(QTriple, QTriple)>>;
}

/// Caches that expose per-sequence RoPE offsets — mlx-swift-lm's
/// `BatchPositionedKVCache`. The concrete batched caches land in a later
/// PR; this is the forward-compatible hook so `rope_offset` can dispatch.
pub trait BatchPositionedKvCache: KvCache {
  // `batch_offset` / `rope_offset` are fallible because they yield an owned
  // `Array` and #33 removes the infallible `impl Clone for Array` (only
  // `try_clone() -> Result` remains).
  /// Per-sequence RoPE offsets with shape `[B]` (swift `batchOffset`).
  fn batch_offset(&self) -> Result<Array>;
}

/// Reconstruct a cache from its serialized class name + state + metadata —
/// mlx-lm's `globals()[class].from_state(state, meta_state)` (the load path
/// of `load_prompt_cache`, `cache.py:79-82`).
///
/// `save_prompt_cache` writes the kind as `type(c).__name__` (the
/// **reference Python class name**, `cache.py:56`) and `load_prompt_cache`
/// reconstructs via `globals()[that_name]` (`cache.py:80`), so a prompt
/// cache produced by mlx-lm / mlx-swift names the cache by its reference
/// class — the match is keyed primarily on those source names, with our
/// Rust struct names kept as back-compat aliases:
///
/// - → [`StandardKvCache`]: `"KVCache"` | `"ConcatenateKVCache"` |
///   `"KVCacheSimple"` (swift) | `"StandardKvCache"` (Rust alias)
/// - → [`RotatingKvCache`]: `"RotatingKVCache"` | `"RotatingKvCache"`
///   (Rust alias)
///
/// The other cache kinds (`"ChunkedKVCache"`, `"QuantizedKVCache"`,
/// `"ArraysCache"`, `"CacheList"`, `"BatchKVCache"`,
/// `"BatchRotatingKVCache"`) are added by later PRs, so an unrecognized
/// `kind` is a recoverable [`Error::Backend`] and the match is left
/// extensible for those arms.
pub fn from_state(kind: &str, state: Vec<Array>, meta: &[String]) -> Result<Box<dyn KvCache>> {
  match kind {
    // mlx-lm `KVCache` / its documented twin `ConcatenateKVCache` /
    // mlx-swift-lm `KVCacheSimple`; `"StandardKvCache"` is our own
    // round-trip alias.
    "KVCache" | "ConcatenateKVCache" | "KVCacheSimple" | "StandardKvCache" => {
      let mut c = StandardKvCache::new();
      c.set_state(state)?;
      c.set_meta_state(meta)?;
      Ok(Box::new(c))
    }
    // mlx-lm / mlx-swift-lm `RotatingKVCache`; `"RotatingKvCache"` is our
    // own round-trip alias.
    "RotatingKVCache" | "RotatingKvCache" => {
      // mlx-lm reconstructs without __init__ then assigns state/meta_state
      // (`_BaseCache.from_state`, cache.py:170-175); the placeholder window
      // is overwritten by `set_meta_state`.
      let mut c = RotatingKvCache::new(0, 0);
      c.set_state(state)?;
      c.set_meta_state(meta)?;
      // mlx-lm/Swift rotating state setters require two arrays (a non-empty
      // state), so an empty buffer with a non-zero `offset`/`idx` is
      // unreachable there. Our `set_state` accepts an empty state and
      // `set_meta_state` restores `offset`/`idx` afterwards, which would
      // otherwise let `from_state` build an impossible cache (`keys=None`
      // but `offset>0`): the next `update` would treat that `offset` as
      // `prev`, grow a too-short zero buffer and surface placeholder zeros
      // as "prior context". Enforce the invariant `empty ⇒ offset==0 &&
      // idx==0` only here (so `set_state`/`set_meta_state` stay individually
      // 1:1 with cache.py:527-540), rejecting the inconsistent combination.
      if c.is_empty() && (c.offset() != 0 || c.idx() != 0) {
        return Err(Error::Backend {
          message: "RotatingKvCache: empty state with non-zero offset/idx is invalid".into(),
        });
      }
      Ok(Box::new(c))
    }
    // mlx-lm / mlx-swift-lm `ChunkedKVCache`; `"ChunkedKvCache"` is our own
    // round-trip alias. mlx-lm reconstructs via `_BaseCache.from_state`
    // (`cache.py:170-175`): `obj.state = state` THEN `obj.meta_state =
    // meta_state` — state first (it sets `offset = keys.shape[2]`), then
    // `set_meta_state` restores `chunk_size`/`start_position`. Unlike
    // `_BaseCache`, `ChunkedKVCache`'s state setter unpacks `keys, values =
    // v`, so an empty state is invalid (raises in mlx-lm) and `set_state`
    // surfaces that as a recoverable `Error`.
    "ChunkedKVCache" | "ChunkedKvCache" => {
      // `chunk_size` is overwritten by `set_meta_state`; the placeholder is
      // never observed.
      let mut c = ChunkedKvCache::new(None);
      c.set_state(state)?;
      c.set_meta_state(meta)?;
      Ok(Box::new(c))
    }
    other => Err(Error::Backend {
      message: format!("unknown cache kind: {other}"),
    }),
  }
}

/// The slice of the model `Config` the cache needs.
///
/// PR-1 is deliberately independent of the loader PR: `make_prompt_cache`
/// takes this minimal seam instead of importing the full `lm::load::Config`
/// (which lands in PR-2). PR-2's `Config` will provide a `CacheConfig` (via
/// the linear-stack rebase), so this type — not `Config` — is the stable
/// cache input and PR-1 stays buildable on its own.
pub struct CacheConfig {
  /// One [`KvCache`] is built per decoder layer.
  pub num_hidden_layers: usize,
  /// If set, the model uses sliding-window attention and every layer gets a
  /// [`RotatingKvCache`] of this window size; otherwise a [`StandardKvCache`].
  pub sliding_window: Option<i32>,
}

/// Build one boxed [`KvCache`] per decoder layer for `cfg`, mirroring
/// `mlx_lm.models.cache.make_prompt_cache`.
///
/// A [`RotatingKvCache`] (window = `cfg.sliding_window`, `keep =
/// ROTATING_DEFAULT_KEEP` = 4, matching mlx-lm) is used iff the model has a
/// sliding window; otherwise a [`StandardKvCache`]. The vector has exactly
/// `cfg.num_hidden_layers` entries.
pub fn make_prompt_cache(cfg: &CacheConfig) -> Vec<Box<dyn KvCache>> {
  (0..cfg.num_hidden_layers)
    .map(|_| -> Box<dyn KvCache> {
      match cfg.sliding_window {
        Some(window) => Box::new(RotatingKvCache::new(
          window.max(0) as usize,
          ROTATING_DEFAULT_KEEP.max(0) as usize,
        )),
        None => Box::new(StandardKvCache::new()),
      }
    })
    .collect()
}
