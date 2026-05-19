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
//! `ScaledDotProductAttentionMaskMode`) enums. Concrete caches:
//! [`StandardKvCache`], [`RotatingKvCache`], [`ChunkedKvCache`],
//! [`QuantizedKvCacheImpl`], plus the prompt-cache/persist surface; and the
//! composite [`CacheList`] (added by this PR). The remaining kinds
//! (Arrays / Batch) land in later PRs and [`from_state`] is left
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

mod cache_list;
mod chunked;
mod mask;
pub mod persist;
pub mod prompt;
mod quantized;
mod rotating;
mod standard;
mod util;

pub use cache_list::CacheList;
pub use chunked::*;
pub use mask::{create_attention_mask, create_causal_mask};
pub use persist::*;
pub use prompt::*;
pub use quantized::*;
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
  /// default mirrors mlx-lm `_BaseCache.meta_state` setter
  /// (`cache.py:142-145`): a no-meta cache that receives a non-empty
  /// `meta_state` raises (recoverable [`Error::Backend`] here, not a
  /// panic). An empty `m` is the no-op success path. Concrete caches with
  /// metadata (`RotatingKvCache`, `ChunkedKvCache`) override this with
  /// their own parsing logic.
  fn set_meta_state(&mut self, m: &[String]) -> Result<()> {
    if !m.is_empty() {
      return Err(Error::Backend {
        message: format!(
          "KvCache has no meta_state but {} value(s) were provided: {m:?} \
           (mirrors mlx-lm `_BaseCache.meta_state` setter cache.py:142-145)",
          m.len()
        ),
      });
    }
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

  /// **Mutable** downcast to the quantized refinement, if this cache is
  /// quantized — the `&mut` companion of [`as_quantized`](
  /// KvCache::as_quantized). mlx-swift-lm's quantized fast path needs
  /// *mutable* access (`cache as? QuantizedKVCacheProtocol` on a
  /// class-mutable cache, `KVCache.swift:101`), because the quantized
  /// cache's defining capability
  /// [`update_quantized`](QuantizedKvCache::update_quantized) takes `&mut
  /// self`; without this a generation loop holding a `Box<dyn KvCache>` /
  /// `&mut dyn KvCache` could never reach the quantized fast path through
  /// the generic API. Default: not quantized (every non-quantized cache
  /// inherits this `None`, so the addition is purely additive and
  /// backward-compatible — no sibling cache changes).
  fn as_quantized_mut(&mut self) -> Option<&mut dyn QuantizedKvCache> {
    None
  }

  /// Downcast to the batched-position refinement, if applicable (swift
  /// `cache as? BatchPositionedKVCache`). Default: scalar-positioned.
  fn as_batch_positioned(&self) -> Option<&dyn BatchPositionedKvCache> {
    None
  }

  /// Downcast to the [`CacheList`] composite, if this cache *is* a
  /// `CacheList` (swift `cache as? CacheList`). Default: not a CacheList.
  ///
  /// `CacheList::update` / `make_mask` are container-illegal (faithful to
  /// swift's `fatalError` and the absent mlx-lm methods); a hybrid model
  /// holding a `Box<dyn KvCache>` per layer needs this hook to reach the
  /// `CacheList`-inherent indexing API ([`CacheList::get`] /
  /// [`CacheList::get_mut`]) and delegate to the right child cache. Mirrors
  /// the structural-refinement downcasts above ([`as_quantized`](
  /// KvCache::as_quantized) / [`as_batch_positioned`](
  /// KvCache::as_batch_positioned)). Defaulted so every non-CacheList cache
  /// inherits `None` without change.
  fn as_cache_list(&self) -> Option<&CacheList> {
    None
  }

  /// `&mut` companion to [`as_cache_list`](KvCache::as_cache_list) — the
  /// indexing API a generation loop needs is mutating
  /// ([`CacheList::get_mut`] yields `&mut dyn KvCache` for the child's
  /// `update` / `make_mask`). Default: `None`.
  fn as_cache_list_mut(&mut self) -> Option<&mut CacheList> {
    None
  }

  /// The number of arrays [`state`](KvCache::state) would return, without
  /// materializing or cloning them. The default reads `self.state()?.len()`
  /// (preserving every concrete cache's existing behavior exactly); concrete
  /// caches with an O(1) shortcut (e.g. a fixed `(keys, values)` pair when
  /// populated) MAY override to avoid the per-call array clone the default
  /// does just to read its length.
  ///
  /// Added so [`CacheList::meta_state`] can frame each child's state-array
  /// count without `state().map(|s| s.len())` cloning every child's full
  /// state per call.
  fn state_count(&self) -> Result<usize> {
    self.state().map(|s| s.len())
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
/// - → [`ChunkedKvCache`]: `"ChunkedKVCache"` | `"ChunkedKvCache"`
///   (Rust alias)
/// - → [`QuantizedKvCacheImpl`]: `"QuantizedKVCache"` |
///   `"QuantizedKvCacheImpl"` (Rust alias)
/// - → [`CacheList`]: `"CacheList"` (the composite — rebuilds each child
///   recursively through this same dispatcher) — added by this PR
///
/// The other cache kinds (`"ArraysCache"`, `"BatchKVCache"`,
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
    // mlx-lm / mlx-swift-lm `QuantizedKVCache`; `"QuantizedKvCacheImpl"` is
    // our own round-trip alias.
    "QuantizedKVCache" | "QuantizedKvCacheImpl" => {
      // mlx-lm reconstructs without __init__ then assigns state/meta_state
      // (`_BaseCache.from_state`, cache.py:170-175): `set_state` loads the
      // (4 or 6) packed triple arrays and `set_meta_state` restores
      // `(offset, group_size, bits)` (cache.py:302-304) afterwards — the
      // placeholder `group_size`/`bits` here are overwritten by
      // `set_meta_state` (a serialized prompt cache always carries the
      // 3-value meta_state).
      let mut c = QuantizedKvCacheImpl::new(0, 0);
      c.set_state(state)?;
      c.set_meta_state(meta)?;
      // mlx-lm's `QuantizedKVCache.state` setter (cache.py:294-296:
      // `self.keys, self.values = v`) requires a non-empty `v` to unpack,
      // so an empty buffer with a non-zero restored `offset` is
      // unreachable there. Our `set_state` accepts an empty state (resets
      // to `None`) and `set_meta_state` restores `offset` afterwards,
      // which would otherwise let `from_state` build an impossible cache
      // (`keys=None` but `offset>0`): the next `update`/`update_quantized`
      // would take `prev = self.offset` yet the empty-storage arm of
      // `compute_appended` stores only the new triple, so `offset` and the
      // stored sequence length diverge (phantom context in
      // masks/RoPE/state slicing). Enforce `empty ⇒ offset==0` ONLY here
      // (so `set_state`/`set_meta_state` stay individually 1:1 with
      // cache.py:294-304), exactly mirroring the `RotatingKvCache` restore
      // guard above.
      if c.is_empty() && c.offset() != 0 {
        return Err(Error::Backend {
          message: "QuantizedKvCache: empty state with non-zero offset is invalid".into(),
        });
      }
      // P2 stores the quant triples **exactly `offset`-length** (the
      // documented `ConcatenateKVCache` / `StandardKvCache` equivalence;
      // `mlxrs::Array` is functional, no in-place buffer slice). `set_state`
      // and `set_meta_state` each stay individually 1:1 with mlx-lm
      // (cache.py:294-296 assigns the triples as-is; cache.py:302-304
      // restores `offset`), so a *forged / inconsistent* serialized cache
      // whose restored triple seq-len ≠ restored `offset` would otherwise
      // violate that invariant in BOTH directions — overlength (triple >
      // offset) → next `update_quantized` would `concat_seq` onto the
      // longer stored triple, surfacing stale tokens past the logical
      // `offset`; underlength (triple < offset) → next `update_quantized`
      // would land the new token past the storage end, leaving a phantom
      // gap between storage-len and `offset`. `enforce_offset_len_invariant`
      // converges both directions to the smaller of `offset` and the actual
      // stored seq-len (slice triples down to `offset`; then clamp `offset`
      // down to the post-trim seq-len, since `slice_seq` uses NumPy
      // `std::min(e, n)` clamping at `mlx/ops.cpp:685`). This is NOT new
      // validation the reference lacks: mlx-lm's `state` *getter* already
      // returns `[..., :offset, :]` (cache.py:285-292), which is
      // `[:min(offset, buf_len)]` under Python slice semantics — so this
      // converge makes P2's offset-length representation observably
      // IDENTICAL to mlx-lm's for ALL inputs (including forged ones in
      // either direction) — repr-equivalence maintenance mirroring mlx-lm's
      // `[:offset]`, not a reject. A faithfully saved consistent state
      // (seq-len already == offset, or the full buffer when offset == len)
      // is unaffected — both the triple slice and the offset clamp are
      // no-ops for it.
      c.enforce_offset_len_invariant()?;
      Ok(Box::new(c))
    }
    // mlx-lm `CacheList` (cache.py:814-902) / mlx-swift-lm `CacheList`
    // (KVCache.swift:1248-1370). Its flattened `state`/`meta_state`
    // (`[childCount, (className, stateCount, metaCount, ...meta)*]`)
    // rebuilds each child recursively through *this* dispatcher keyed on
    // the child's reference class name — so a nested `"CacheList"` child
    // recurses (exactly cache.py:898 `globals()["CacheList"]`).
    "CacheList" => cache_list::cache_list_from_state(state, meta),
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
