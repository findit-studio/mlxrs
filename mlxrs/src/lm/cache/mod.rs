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
//! [`StandardQuantizedKvCache`], plus the prompt-cache/persist surface; the
//! composite [`CacheList`]; the batch caches [`BatchKvCache`] /
//! [`BatchRotatingKvCache`] with the [`dynamic_roll`] helper; and the SSM
//! slot cache [`ArraysCache`] (added by this PR). [`from_state`] is left
//! extensible for future cache kinds.
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
  error::{
    Error, InvariantViolationPayload, LayerKeyedPayload, LengthMismatchPayload, OutOfRangePayload,
    Result, UnknownEnumValuePayload,
  },
};
use smol_str::format_smolstr;

pub mod arrays;
pub mod batch;
pub mod batch_rotating;
mod cache_list;
mod chunked;
mod mask;
pub mod persist;
pub mod prompt;
mod quantized;
mod rotating;
mod standard;
mod util;

pub use arrays::*;
pub use batch::*;
pub use batch_rotating::*;
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
#[derive(derive_more::IsVariant, derive_more::Unwrap, derive_more::TryUnwrap)]
#[unwrap(ref, ref_mut)]
#[try_unwrap(ref, ref_mut)]
pub enum RopeOffset {
  /// A single scalar RoPE position (the offset all sequences share).
  Scalar(usize),
  /// Per-sequence RoPE offsets with shape `[B]` (batched caches).
  Batch(Array),
}

/// Attention-mask mode returned by [`KvCache::make_mask`] — mlx-swift-lm's
/// `ScaledDotProductAttentionMaskMode`, equivalently mlx-lm's
/// `None | "causal" | array` triad.
#[derive(derive_more::IsVariant, derive_more::Unwrap, derive_more::TryUnwrap)]
#[unwrap(ref, ref_mut)]
#[try_unwrap(ref, ref_mut)]
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
///
/// # Per-layer fast-path convention (#110)
///
/// `Model::forward` calls `cache[i].update(k, v)?` once per layer per token
/// through the `Box<dyn KvCache>` vtable — `~32-80` vtables per token,
/// one per layer. Each layer's cache type is **fixed per model
/// architecture**, so the model statically knows which concrete type to
/// expect; the vtable is wasted information.
///
/// **Convention** (mirrors mlx-swift-lm's
/// `cache as? QuantizedKVCacheProtocol` idiom at
/// `KVCache.swift:101`): inside each per-model `Model::forward`,
/// downcast once via [`as_any_mut`](KvCache::as_any_mut) **before** the
/// per-layer hot loop, then dispatch every per-layer `update` /
/// `make_mask` / etc. on the concrete type — one vtable per layer
/// (the downcast), then static dispatch on every subsequent method:
///
/// ```ignore
/// fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
///     let mut x = self.embed(tokens)?;
///     for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
///         // ONE vtable per layer (downcast), then static dispatch on
///         // every update / make_mask / etc. inside the layer body.
///         let rot = c.as_any_mut()
///             .downcast_mut::<RotatingKvCache>()
///             .ok_or_else(|| Error::InvariantViolation(
///                 InvariantViolationPayload::new(
///                     "layer cache type",
///                     "must downcast to RotatingKvCache",
///                 ),
///             ))?;
///         let (k, v) = rot.update(&x_k, &x_v)?;       // static dispatch
///         let mask = rot.make_mask(s, None, false)?;  // static dispatch
///         x = layer.attention(&x, &k, &v, &mask)?;
///     }
///     Ok(x)
/// }
/// ```
///
/// # Breaking change (#110)
///
/// This trait now requires a `fn as_any_mut(&mut self) -> &mut dyn std::any::Any`
/// method with NO default. Out-of-tree implementations of [`KvCache`] MUST add
/// the following one-line override to compile:
///
/// ```rust,ignore
/// impl KvCache for MyCustomCache {
///     // ...
///     fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
/// }
/// ```
///
/// **Lifetime requirement**: `std::any::Any` requires `Self: 'static`, so the
/// `{ self }` one-liner only compiles for cache types that do not carry
/// non-`'static` borrows. Custom caches holding `&'a SomeRef` borrows hit a
/// lifetime error of the form `coercion requires 'a to outlive 'static` (no
/// E-code; the borrow checker rejects the upcast to `&mut dyn Any`) and need
/// a different downcast strategy (e.g., an `enum CacheRef<'a>` discriminator
/// or `&'a self` accessors that bypass `Any` entirely). E0310 can
/// additionally apply when the cache type takes an unconstrained generic
/// parameter `T` that the compiler cannot prove is `'static` (e.g.,
/// `impl<T> KvCache for MyCache<T>` without a `T: 'static` bound). All
/// in-tree caches (Standard, Rotating, Chunked, Quantized, Batch,
/// BatchRotating, Arrays, CacheList) are `'static` and use the one-liner
/// above.
///
/// This enables zero-cost downcast-once-per-layer typed cache access in the
/// hot loop, replacing per-layer vtable dispatch. The existing dynamic
/// `update` / `update_quantized` dispatch continues to work, but only after
/// implementers add this method (compile error E0046 otherwise; a
/// `coercion requires 'a to outlive 'static` lifetime error for caches with
/// non-`'static` borrowed fields, or E0310 for unconstrained generic
/// parameters lacking a `'static` bound).
///
/// The first per-model port (Qwen3-VL, LFM2, …) should land the downcast
/// pattern alongside its `forward` impl.
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
  /// whose automatic conformance Rust lacks — so a batch cache need only
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

  /// Push the serializable array state into a caller-provided buffer —
  /// the buffer-reuse companion to [`state`](KvCache::state) (#104).
  ///
  /// Python's `_BaseCache.state` getter (`cache.py:152-156`) returns a
  /// fresh tuple per call, and swift's `var state: [MLXArray]` allocates a
  /// fresh `Vec`; mlxrs's `state()` follows that pattern (`Vec<Array>` per
  /// call). Hot consumers that already own a working buffer (e.g.
  /// [`CacheList::state`], `persist::save_prompt_cache`) can avoid the
  /// per-call `Vec` allocation by reusing one through this method.
  ///
  /// The default implementation delegates to [`state`](KvCache::state) and
  /// `append`s its contents — semantically identical to a direct
  /// `buf.extend(self.state()?)`. Concrete caches with a fixed pair (e.g.
  /// `(keys, values)`) MAY override to `try_clone` straight into `buf`
  /// (saving the intermediate `Vec`); the contract is the same:
  /// **append** state arrays (do not clear `buf` — multi-child callers
  /// pass the same buffer through several caches).
  fn state_into(&self, buf: &mut Vec<Array>) -> Result<()> {
    buf.extend(self.state()?);
    Ok(())
  }

  /// Restore the array state (mlx-lm `cache.state` setter).
  fn set_state(&mut self, state: Vec<Array>) -> Result<()>;

  /// Force evaluation of the cache's **own stored arrays in place** — the
  /// per-chunk prefill memory barrier mlx-lm runs as
  /// `mx.eval([c.state for c in prompt_cache])` (`generate.py:442`).
  ///
  /// `mlxrs::Array` is lazy (an op only records a graph node), so without a
  /// per-chunk barrier the prefill loop would accumulate a single lazy graph
  /// spanning **every** chunk and force it only at the final save — peak
  /// memory would grow with the whole prompt and `prefill_step_size` would
  /// bound nothing (a long prompt could OOM/abort). This hook caps the live
  /// graph to one chunk's work by materializing the buffers the *next* chunk
  /// reuses.
  ///
  /// # Why a `&mut self` hook and not `mx.eval(self.state())`
  ///
  /// mlx-lm evals `c.state`, which for the plain append cache *is* the live
  /// buffer. But several caches' [`state`](KvCache::state) is a **serialized
  /// view**, not the stored buffer: a sliding-window / chunked / batched
  /// cache over-allocates its ring/step buffer and `state()` returns
  /// `seq_slice(self.keys, 0, offset)` (the `offset`-length serialization
  /// slice) whenever `offset < buffer_len` — exactly the regime an `S == 1`
  /// update reaches after growing the ring (and the `prefill_step_size == 1`
  /// path, also reachable via the `0` clamp). Evaluating that temporary slice
  /// is **not** the same operation as evaluating the `self.keys`/`self.values`
  /// arrays the next chunk's `update` actually reads and extends. Routing the
  /// barrier through a `&mut self` hook lets each concrete cache eval its
  /// genuine stored arrays directly (it owns the private fields), so the
  /// materialization is on the live buffers regardless of what the
  /// serialization `state()` happens to slice.
  ///
  /// # Contract for implementers
  ///
  /// Evaluate the concrete cache's **own** stored [`Array`] fields with the
  /// explicit `&mut` [`Array::eval`] step (the crate makes eval an explicit
  /// `&mut` operation; accessors never hidden-eval) — the real `self.keys`/
  /// `self.values` (or quantized triples / per-sequence position arrays /
  /// per-slot SSM state / child caches), **never** a `seq_slice`/`try_clone`
  /// of [`state`](KvCache::state). An empty cache (no stored arrays) is a
  /// no-op `Ok(())`. This is a required method (no default) precisely so a
  /// new cache kind cannot silently inherit a `state()`-based barrier that
  /// would evaluate the wrong (sliced) arrays.
  fn materialize(&mut self) -> Result<()>;

  /// Serializable scalar metadata as strings (mlx-lm `cache.meta_state`);
  /// empty for caches without metadata.
  fn meta_state(&self) -> Vec<String> {
    Vec::new()
  }

  /// Push the serializable scalar metadata into a caller-provided buffer —
  /// the buffer-reuse companion to [`meta_state`](KvCache::meta_state)
  /// (#103).
  ///
  /// Python's `_BaseCache.meta_state` getter (`cache.py:158-165`) returns
  /// a fresh tuple per call and swift's `metaState: [String]` allocates a
  /// fresh `Vec`; mlxrs's `meta_state()` follows the same pattern. Hot
  /// callers that frame multiple caches' meta — e.g. [`CacheList::
  /// meta_state`] (which calls `c.meta_state()` for every child on every
  /// invocation) or `persist::save_prompt_cache` — can avoid the per-call
  /// `Vec<String>` allocation by reusing a single buffer through this
  /// method.
  ///
  /// The default delegates to [`meta_state`](KvCache::meta_state) and
  /// `append`s its contents — semantically identical to a direct
  /// `buf.extend(self.meta_state())`. Concrete caches with a fixed-arity
  /// meta (e.g. `RotatingKvCache`'s 4 elements) MAY override to push
  /// directly into `buf` (saving the intermediate `Vec`); the contract is
  /// the same: **append** meta entries (do not clear `buf`).
  fn meta_state_into(&self, buf: &mut Vec<String>) {
    buf.extend(self.meta_state());
  }

  /// Restore scalar metadata (mlx-lm `cache.meta_state` setter). The
  /// default mirrors mlx-lm `_BaseCache.meta_state` setter
  /// (`cache.py:142-145`): a no-meta cache that receives a non-empty
  /// `meta_state` raises (recoverable [`Error::LengthMismatch`] here, not a
  /// panic). An empty `m` is the no-op success path. Concrete caches with
  /// metadata (`RotatingKvCache`, `ChunkedKvCache`) override this with
  /// their own parsing logic.
  fn set_meta_state(&mut self, m: &[String]) -> Result<()> {
    if !m.is_empty() {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "KvCache::set_meta_state: meta_state value count for a no-meta cache (mirrors mlx-lm `_BaseCache.meta_state` setter cache.py:142-145)",
        0,
        m.len(),
      )));
    }
    Ok(())
  }

  /// Restore the cache from a serialized `(state, meta)` pair.
  ///
  /// # Transactional default
  ///
  /// The default implementation **snapshots** the current `state()` +
  /// `meta_state()` BEFORE calling [`set_state`](KvCache::set_state) and
  /// [`set_meta_state`](KvCache::set_meta_state) — so on a
  /// `set_meta_state` failure *after* `set_state` has already mutated
  /// `self`, the default rolls back to the snapshot (best-effort, see
  /// "Rollback failure" below). This closes the half-restore window the
  /// earlier default left open (the verbatim
  /// `mlx_lm/models/cache.py:170-175` two-setter sequence).
  ///
  /// The snapshot is captured as `Vec<Array>` (via `state()`, which
  /// returns owned arrays — typically refcount-shared clones) and
  /// `Vec<String>` (via `meta_state()`). For a typical cache, this is a
  /// handful of small arrays + a few-element string vector — cheap and
  /// transient.
  ///
  /// # Rollback failure
  ///
  /// If the rollback itself fails (the snapshot's `set_state` or
  /// `set_meta_state` returns `Err`), the original `set_meta_state` error
  /// is returned with a `(rollback also failed: …)` suffix appended for
  /// diagnostics. In that pathological case `self` may be left
  /// half-restored — the same failure mode the earlier default had
  /// unconditionally. The rollback path runs only on the
  /// previously-valid snapshot, so this is a defense-in-depth annotation
  /// of an extreme failure scenario, not the common case.
  ///
  /// # When to override
  ///
  /// Concrete caches with non-trivial meta parsing / multi-field state
  /// mutation SHOULD still override this with the stronger
  /// stage-on-placeholder discipline [`ArraysCache::set_meta_state`]
  /// already follows: every fallible parse and allocation runs on locals
  /// while `self` is untouched, and `self` is committed by a single
  /// infallible block at the end. The contract for overrides is: if
  /// `from_serialized` returns `Err`, `self` is byte-identical to its
  /// pre-call state ("leaves self unchanged on error") — strictly
  /// stronger than the default's best-effort rollback.
  ///
  /// **Implementers — override unless your cache is meta-less AND has no
  /// state invariants**. If your impl has (a) any fallible meta parsing,
  /// OR (b) any post-setter invariant the canonical `super::from_state`
  /// loader enforces (e.g. `empty ⇒ offset==0`, `_idx <= L`,
  /// `enforce_offset_len_invariant`), you MUST override `from_serialized`
  /// to stage on a fresh placeholder + apply the same post-setter check +
  /// commit via `*self = staged` only on success. All 8 in-tree concrete
  /// caches do this; see e.g. `RotatingKvCache::from_serialized` for the
  /// canonical pattern.
  ///
  /// `state` is consumed (taken by value, like `set_state`); `meta` is
  /// borrowed because it's typically a small `Vec<String>` of decimal
  /// tokens that an impl may want to parse multiple times.
  #[allow(clippy::wrong_self_convention)] // mirrors mlx-lm `from_state` / swift `restoreFromMetaState`
  fn from_serialized(&mut self, state: Vec<Array>, meta: &[String]) -> Result<()> {
    // Issue #98: transactional default — snapshot
    // pre-call `state()` + `meta_state()`, attempt the 2-setter chain, and
    // on EITHER setter's failure roll back to the snapshot so `self` is
    // not left half-restored.
    //
    // The trait does NOT require `set_state` to be atomic — an impl
    // inheriting this default can mutate part of its state and then
    // return `Err`, leaving the cache corrupt unless we explicitly
    // restore on the `set_state` arm too. So both arms run the same
    // best-effort rollback closure (`set_state(snapshot_state)` then
    // `set_meta_state(snapshot_meta)`). The snapshot is a
    // previously-valid `(state, meta)` round-trip, so the rollback
    // setters should succeed; if either rollback step ITSELF fails
    // (defense-in-depth for a pathological scenario), surface the
    // original error with a rollback-failure suffix.
    let snapshot_state = self.state()?;
    let snapshot_meta = self.meta_state();
    // Forward the primary error `e`, attempting both rollback setters
    // best-effort; if EITHER rollback step itself errors, wrap with a
    // rollback-failed suffix. Snapshots are moved into this block once.
    let rollback = |cache: &mut Self,
                    e: Error,
                    snap_state: Vec<Array>,
                    snap_meta: Vec<String>|
     -> Error {
      if let Err(rb_state_err) = cache.set_state(snap_state) {
        // Preserve the primary error `e` as the typed inner and annotate
        // the rollback-step + failure via the runtime layer key. The
        // primary's typed structure (variant + payload) survives end-to-end
        // for downstream branching.
        return Error::LayerKeyed(LayerKeyedPayload::new(
          format_smolstr!(
            "KvCache::from_serialized: rollback failed (set_state on snapshot: {rb_state_err})"
          ),
          e,
        ));
      }
      if let Err(rb_meta_err) = cache.set_meta_state(&snap_meta) {
        return Error::LayerKeyed(LayerKeyedPayload::new(
          format_smolstr!(
            "KvCache::from_serialized: rollback failed (set_meta_state on snapshot: {rb_meta_err})"
          ),
          e,
        ));
      }
      e
    };
    match self.set_state(state) {
      Ok(()) => match self.set_meta_state(meta) {
        Ok(()) => Ok(()),
        Err(e) => Err(rollback(self, e, snapshot_state, snapshot_meta)),
      },
      Err(e) => Err(rollback(self, e, snapshot_state, snapshot_meta)),
    }
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

  /// Mutable [`Any`](std::any::Any) hook for the per-layer fast-path
  /// downcast (#110 — see the **Per-layer fast-path convention**
  /// section on the trait doc above).
  ///
  /// Each per-model `Model::forward` calls `cache[i].update(k, v)?`
  /// once per layer through the `Box<dyn KvCache>` vtable. The cache
  /// type is fixed per architecture, so the model can downcast once
  /// via `cache[i].as_any_mut().downcast_mut::<ConcreteCache>()` and
  /// then dispatch every subsequent per-layer method statically
  /// (mlx-swift-lm's `cache as? QuantizedKVCacheProtocol` idiom).
  ///
  /// This is the single trait method needed — all 8 in-tree caches
  /// override it with the one-line `self` return:
  ///
  /// ```ignore
  /// fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
  ///     self
  /// }
  /// ```
  ///
  /// **Required method, no default body.** The naive default of
  /// returning a `'static` placeholder (`Box::leak(Box::new(()))`)
  /// would allocate-and-leak per call, and a `&'static mut ()` via
  /// `OnceLock` would race on `&mut`; widening `self` itself would
  /// require `Self: 'static` on the trait (not enforced today and
  /// would propagate the bound to every existing `Box<dyn KvCache>`).
  /// Making this a required method instead surfaces the
  /// "did-not-opt-in" case as a compile error, not a silent
  /// fast-path miss — the right failure mode for a downcast target.
  ///
  /// **Lifetime requirement**: `std::any::Any` carries an implicit
  /// `Self: 'static` bound. The `{ self }` one-liner above therefore
  /// only compiles when the concrete cache type is `'static` (no
  /// non-`'static` borrows in its fields). Out-of-tree caches carrying
  /// `&'a SomeRef` borrows hit a lifetime error of the form
  /// `coercion requires 'a to outlive 'static` (no E-code; the borrow
  /// checker rejects the upcast to `&mut dyn Any`) with the one-liner
  /// and need a non-`Any` downcast strategy (e.g., a
  /// lifetime-parameterized discriminator enum or a typed `&'a self`
  /// accessor). E0310 can additionally apply when the cache type takes
  /// an unconstrained generic parameter `T` that the compiler cannot
  /// prove is `'static` (e.g., `impl<T> KvCache for MyCache<T>` without
  /// a `T: 'static` bound). All in-tree caches are `'static`. The trait
  /// itself is **not** declared `KvCache: Any` / `'static` to avoid
  /// propagating the bound to every existing `Box<dyn KvCache>`
  /// consumer.
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

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

  /// The mlx-lm reference class name (`type(c).__name__`) this cache
  /// serializes as — the **single switch** mlx-swift-lm's
  /// `cacheClassName(_:)` performs at `KVCache.swift:1381-1392`, lifted onto
  /// the trait so the kind label is a property of the concrete cache (no
  /// downstream `meta_state()` / `max_size()` heuristic). Used by
  /// [`persist::reference_class_name`] and [`from_state`] to round-trip the
  /// cache through the prompt-cache save/load surface; the returned string
  /// is exactly what mlx-lm `save_prompt_cache` writes (`cache.py:56`,
  /// `type(c).__name__`) and what [`from_state`] keys on.
  ///
  /// # REQUIRED method — no default
  ///
  /// This is intentionally a **required** trait method (no default).
  /// Forgetting to declare the class name on a new in-tree cache is a
  /// **compile error**, not a silent runtime label-loss (the pre-PR default
  /// `"KVCache"` would have round-tripped a new state-shape-different cache
  /// as a [`StandardKvCache`] on load — silent data loss). This diverges
  /// from mlx-swift-lm's `default: return "KVCache"` arm
  /// (`KVCache.swift:1390`) — a deliberate Rust-idiom upgrade that benefits
  /// compile-time safety while preserving the trait's open polymorphism for
  /// every concrete in-tree cache (each one declares its name in one
  /// `&'static str` literal). Out-of-tree wrappers MUST forward to
  /// `inner.reference_class_name()` explicitly (one-line method body) and
  /// in-crate wrappers / future cache kinds get compile-time guidance.
  ///
  /// # On the removed heuristic
  ///
  /// The pre-PR [`persist::reference_class_name`] was a *workaround*: it
  /// faked swift's class-identity-switch via
  /// `as_cache_list()` / `as_batch_positioned()` / `max_size() +
  /// meta_state().len()` probes — necessary precisely because there was
  /// **no** `reference_class_name` trait method. The trait method (faithful
  /// to swift's `cacheClassName` switch) makes the heuristic structurally
  /// **redundant**, not just stylistically removable. Reinstating it as a
  /// "fallback" when the trait method returns the default would re-create
  /// the very hack the trait method replaces — and miss the swift-faithful
  /// design point that the kind label IS a property of the concrete cache.
  fn reference_class_name(&self) -> &'static str;
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

/// The canonical typed dispatch tag for [`from_state`] — replaces the
/// mlx-lm `globals()[class].from_state(...)` string-keyed lookup
/// (`cache.py:898`) with a Rust enum whose variants are an exhaustive
/// match. Adding a new in-tree cache kind requires adding a variant AND
/// the dispatch arm in [`from_state`] — both compile-checked, so a
/// forgotten arm is a `non_exhaustive_patterns` warning/error.
///
/// `parse` accepts both the reference Python/Swift class names AND the
/// Rust struct-name round-trip aliases — every key the previous
/// string-keyed match accepted (back-compat).
#[derive(Debug, Clone, PartialEq, Eq, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
pub enum KvCacheKind {
  /// [`StandardKvCache`]: `"KVCache"` | `"ConcatenateKVCache"` |
  /// `"KVCacheSimple"` (swift) | `"StandardKvCache"` (Rust alias).
  KvCache,
  /// [`RotatingKvCache`]: `"RotatingKVCache"` | `"RotatingKvCache"`
  /// (Rust alias).
  RotatingKvCache,
  /// [`ChunkedKvCache`]: `"ChunkedKVCache"` | `"ChunkedKvCache"`
  /// (Rust alias).
  ChunkedKvCache,
  /// [`StandardQuantizedKvCache`]: `"QuantizedKVCache"` |
  /// `"StandardQuantizedKvCache"` (Rust alias).
  QuantizedKvCache,
  /// [`CacheList`]: `"CacheList"` (the composite).
  CacheList,
  /// [`BatchKvCache`]: `"BatchKVCache"` | `"BatchKvCache"` (Rust alias).
  BatchKvCache,
  /// [`BatchRotatingKvCache`]: `"BatchRotatingKVCache"` |
  /// `"BatchRotatingKvCache"` (Rust alias).
  BatchRotatingKvCache,
  /// [`ArraysCache`] (the generic SSM slot cache): `"ArraysCache"`.
  ArraysCache,
  /// [`ArraysCache`] aliased as `MambaCache` — mlx-swift-lm's
  /// `class MambaCache: ArraysCache` (`KVCache.swift:1229`, a 2-slot
  /// `ArraysCache`). Distinct variant from [`KvCacheKind::ArraysCache`]
  /// so the `is_mamba` provenance flag survives the round trip.
  MambaCache,
}

impl KvCacheKind {
  /// The canonical string tag for this cache kind — single source of truth for
  /// [`std::fmt::Display`], log messages, and the `reference_class_name` round
  /// trip. Unit-only enum (`KvCacheKind` has no data variants), so this is
  /// `const fn`.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::KvCache => "KVCache",
      Self::RotatingKvCache => "RotatingKVCache",
      Self::ChunkedKvCache => "ChunkedKVCache",
      Self::QuantizedKvCache => "QuantizedKVCache",
      Self::CacheList => "CacheList",
      Self::BatchKvCache => "BatchKVCache",
      Self::BatchRotatingKvCache => "BatchRotatingKVCache",
      Self::ArraysCache => "ArraysCache",
      Self::MambaCache => "MambaCache",
    }
  }

  /// Parse a reference class name (or Rust round-trip alias) into the
  /// typed dispatch tag, or `Err` for an unknown kind — the typed
  /// replacement for the earlier `match kind { ... other => Err(...) }`
  /// string-keyed dispatch.
  pub fn parse(kind: &str) -> Result<Self> {
    match kind {
      "KVCache" | "ConcatenateKVCache" | "KVCacheSimple" | "StandardKvCache" => Ok(Self::KvCache),
      "RotatingKVCache" | "RotatingKvCache" => Ok(Self::RotatingKvCache),
      "ChunkedKVCache" | "ChunkedKvCache" => Ok(Self::ChunkedKvCache),
      "QuantizedKVCache" | "StandardQuantizedKvCache" => Ok(Self::QuantizedKvCache),
      "CacheList" => Ok(Self::CacheList),
      "BatchKVCache" | "BatchKvCache" => Ok(Self::BatchKvCache),
      "BatchRotatingKVCache" | "BatchRotatingKvCache" => Ok(Self::BatchRotatingKvCache),
      "ArraysCache" => Ok(Self::ArraysCache),
      "MambaCache" => Ok(Self::MambaCache),
      other => Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
        "KvCacheKind",
        other,
        &[
          "KVCache",
          "ConcatenateKVCache",
          "KVCacheSimple",
          "StandardKvCache",
          "RotatingKVCache",
          "RotatingKvCache",
          "ChunkedKVCache",
          "ChunkedKvCache",
          "QuantizedKVCache",
          "StandardQuantizedKvCache",
          "CacheList",
          "BatchKVCache",
          "BatchKvCache",
          "BatchRotatingKVCache",
          "BatchRotatingKvCache",
          "ArraysCache",
          "MambaCache",
        ],
      ))),
    }
  }
}

/// Reconstruct a cache from its serialized class name + state + metadata —
/// mlx-lm's `globals()[class].from_state(state, meta_state)` (the load path
/// of `load_prompt_cache`, `cache.py:79-82`).
///
/// `save_prompt_cache` writes the kind as `type(c).__name__` (the
/// **reference Python class name**, `cache.py:56`) and `load_prompt_cache`
/// reconstructs via `globals()[that_name]` (`cache.py:80`), so a prompt
/// cache produced by mlx-lm / mlx-swift names the cache by its reference
/// class — the string-keyed `match kind { … }` is replaced with a
/// typed [`KvCacheKind`] enum dispatched via [`KvCacheKind::parse`]. The
/// match is keyed primarily on those source names, with our Rust struct
/// names kept as back-compat aliases (see [`KvCacheKind`]):
///
/// - → [`StandardKvCache`]: `"KVCache"` | `"ConcatenateKVCache"` |
///   `"KVCacheSimple"` (swift) | `"StandardKvCache"` (Rust alias)
/// - → [`RotatingKvCache`]: `"RotatingKVCache"` | `"RotatingKvCache"`
///   (Rust alias)
/// - → [`ChunkedKvCache`]: `"ChunkedKVCache"` | `"ChunkedKvCache"`
///   (Rust alias)
/// - → [`StandardQuantizedKvCache`]: `"QuantizedKVCache"` |
///   `"StandardQuantizedKvCache"` (Rust alias)
/// - → [`CacheList`]: `"CacheList"` (the composite — rebuilds each child
///   recursively through this same dispatcher)
/// - → [`BatchKvCache`]: `"BatchKVCache"` | `"BatchKvCache"` (Rust alias)
/// - → [`BatchRotatingKvCache`]: `"BatchRotatingKVCache"` |
///   `"BatchRotatingKvCache"` (Rust alias)
/// - → [`ArraysCache`]: `"ArraysCache"` (the generic SSM slot cache) |
///   `"MambaCache"` (mlx-swift-lm's `class MambaCache: ArraysCache`,
///   `KVCache.swift:1229` — a 2-slot `ArraysCache` adding **no** extra
///   state/metadata; swift saves this kind, `KVCache.swift:1384`, and
///   reconstructs it identically, `KVCache.swift:1531`); the
///   `MambaCache` provenance is preserved via [`ArraysCache::mamba`] so a
///   save-after-load emits `"MambaCache"` again rather than degrading to
///   `"ArraysCache"`.
///
/// An unrecognized `kind` is a recoverable [`Error::UnknownEnumValue`] (returned
/// by [`KvCacheKind::parse`]); adding a new variant to [`KvCacheKind`]
/// causes a `non_exhaustive_patterns` compile error here until the
/// matching arm is added (the structural-exhaustiveness benefit of the
/// typed dispatch).
pub fn from_state(kind: &str, state: Vec<Array>, meta: &[String]) -> Result<Box<dyn KvCache>> {
  match KvCacheKind::parse(kind)? {
    // mlx-lm `KVCache` / its documented twin `ConcatenateKVCache` /
    // mlx-swift-lm `KVCacheSimple`; `"StandardKvCache"` is our own
    // round-trip alias.
    KvCacheKind::KvCache => {
      let mut c = StandardKvCache::new();
      c.set_state(state)?;
      c.set_meta_state(meta)?;
      Ok(Box::new(c))
    }
    // mlx-lm / mlx-swift-lm `RotatingKVCache`; `"RotatingKvCache"` is our
    // own round-trip alias.
    KvCacheKind::RotatingKvCache => {
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
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "RotatingKvCache::from_state: empty state with non-zero offset/idx",
          "must satisfy offset=0 AND idx=0 when buffer is empty",
        )));
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
    KvCacheKind::ChunkedKvCache => {
      // `chunk_size` is overwritten by `set_meta_state`; the placeholder is
      // never observed.
      let mut c = ChunkedKvCache::new(None);
      c.set_state(state)?;
      c.set_meta_state(meta)?;
      Ok(Box::new(c))
    }
    // mlx-lm / mlx-swift-lm `QuantizedKVCache`; `"StandardQuantizedKvCache"` is
    // our own round-trip alias.
    KvCacheKind::QuantizedKvCache => {
      // mlx-lm reconstructs without __init__ then assigns state/meta_state
      // (`_BaseCache.from_state`, cache.py:170-175): `set_state` loads the
      // (4 or 6) packed triple arrays and `set_meta_state` restores
      // `(offset, group_size, bits)` (cache.py:302-304) afterwards — the
      // placeholder `group_size`/`bits` here are overwritten by
      // `set_meta_state` (a serialized prompt cache always carries the
      // 3-value meta_state).
      let mut c = StandardQuantizedKvCache::new_unchecked(0, 0);
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
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "QuantizedKvCache::from_state: empty state with non-zero offset",
          "must satisfy offset=0 when buffer is empty",
        )));
      }
      // The quant triples are stored **exactly `offset`-length** (the
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
      // converge makes the offset-length representation observably
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
    KvCacheKind::CacheList => cache_list::cache_list_from_state(state, meta),
    // mlx-lm `BatchKVCache` (cache.py:912). `_BaseCache.from_state`
    // (cache.py:170-175) reconstructs without `__init__` then assigns
    // `state` (`[keys, values, offset, left_padding]`, the setter derives
    // `_idx = keys.shape[2]`); `BatchKVCache` has no `meta_state` so `meta`
    // is the `_BaseCache` empty default. The placeholder `left_padding`
    // (`new(&[])`) is fully overwritten by `set_state`'s 4-array branch.
    KvCacheKind::BatchKvCache => {
      let mut c = BatchKvCache::new(&[]);
      c.set_state(state)?;
      c.set_meta_state(meta)?;
      // (Previous defensive `is_empty && offset != 0` check removed as
      // unreachable: `BatchKvCache::offset()` returns `_idx`, and
      // `BatchKvCache::set_state(Vec::new())` deterministically sets
      // `_idx = 0` (and now also resets `offset`/`right_padding` per the
      // empty-state-clear fix), while `set_meta_state` is a no-op for
      // `BatchKvCache`. So after `set_state(state)? + set_meta_state(meta)?`
      // the condition `is_empty() && offset() != 0` cannot hold — there is
      // no observable code path to it. The 4-array set_state branch's
      // rank-validation handles the only remaining recoverable-Err case.)
      Ok(Box::new(c))
    }
    // mlx-lm `BatchRotatingKVCache` (cache.py:1133). Reconstructed without
    // `__init__`, then `state` (`[keys, values, offset, left_padding]`) +
    // `meta_state` (`(max_size, _offset, _idx, rotated)`) are assigned; the
    // placeholder `max_size`/`left_padding` are overwritten by the setters.
    KvCacheKind::BatchRotatingKvCache => {
      let mut c = BatchRotatingKvCache::new(0, &[]);
      c.set_state(state)?;
      c.set_meta_state(meta)?;
      // STRUCTURAL hostile-restore guard (single chokepoint). `set_state`
      // /`set_meta_state` stay individually 1:1 with cache.py:1301-1315
      // (mlx-lm's setters do no validation — but mlx-lm is unbounded-int
      // numpy where a corrupt `_idx`/`rotated` errors at the *actual op*;
      // our functional port's `seq_slice` deliberately clamps, so a bad
      // restored cursor would silently mis-splice instead). Rather than
      // re-checking each downstream op (the overflow/lossy-cast/splice
      // symptoms), validate ONCE here that the restored `(state,
      // meta_state)` is one mlx-lm's own `state` getter
      // (cache.py:1294-1307) could have produced — closing the entire
      // corrupt-restored-`_idx`/`_offset`/`rotated` class:
      //
      //  * empty buffer ⇒ fully fresh: `_offset==0 && _idx==0 &&
      //    !rotated` (mlx-lm emits `keys=None` only for an untouched
      //    cache; a stale flag is NOT self-healing — the empty
      //    `_update_concat` branch keeps `rotated`, poisoning the next
      //    `make_mask`);
      //  * non-empty buffer (`L = keys.shape[-2]`) ⇒ `max_size >= 1`
      //    (a real rotating cache; `==0` only the pre-setter placeholder),
      //    `_idx <= L` (the ring write cursor never exceeds the physical
      //    buffer — rejects the out-of-range-`_idx` mis-splice),
      //    `rotated ⇒ L == max_size` (the ring only wraps once the buffer
      //    reached `max_size`), and `L <= _offset` — mlx-lm's getter
      //    (cache.py:1297-1298) emits `keys[..., :_offset, :]` whenever
      //    `_offset < buf_len`, so the SERIALIZED keys length is always
      //    `min(_offset, buf_len) <= _offset`; an `L > _offset` state is
      //    impossible from a real round-trip and would let
      //    `_update_in_place` skip growth and surface stale buffer rows.
      //    `_offset` itself stays uncapped (legitimately `> L`); only the
      //    `L <= _offset` direction is constrained (its overflow surface
      //    remains the in-op `checked_add`s). These four conjuncts are the
      //    EXACT characterization of states mlx-lm's `state` getter can
      //    produce — the structural class-kill, complete.
      // Report the EXACT conflict (which invariant failed and the offending
      // values) so a corrupt prompt-cache file is diagnosable from the
      // error message alone.
      if c.is_empty() {
        let offset = c.offset();
        let idx = c.ring_idx();
        let rotated = c.is_rotated();
        if offset != 0 || idx != 0 || rotated {
          // Empty buffer with non-fresh meta — surface the offending
          // (offset, idx, rotated) triple via OutOfRange whose `value`
          // carries the runtime triple (mirrors the sibling `_idx > L`
          // arm at lines 1044-1049 / `L > _offset` arm at 1056-1061).
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "BatchRotatingKvCache::from_serialized: empty buffer (keys=None) requires fully-fresh meta",
            "must satisfy offset=0 AND _idx=0 AND rotated=false",
            format_smolstr!("offset={offset}, _idx={idx}, rotated={rotated}"),
          )));
        }
      } else {
        let l = c.buf_seq_len()?.unwrap_or(0);
        let max_size = c.max_window();
        let idx = c.ring_idx();
        let offset = c.offset();
        let rotated = c.is_rotated();
        if max_size == 0 {
          return Err(Error::InvariantViolation(InvariantViolationPayload::new(
            "BatchRotatingKvCache::from_serialized: max_size",
            "must be >= 1 for a non-empty buffer (max_size=0 is only the pre-setter placeholder)",
          )));
        } else if idx > l {
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "BatchRotatingKvCache::from_serialized: _idx (write cursor must not exceed physical buffer seq-len L)",
            "must satisfy _idx <= L",
            format_smolstr!("_idx={idx}, L={l}"),
          )));
        } else if rotated && l != max_size {
          return Err(Error::LengthMismatch(LengthMismatchPayload::new(
            "BatchRotatingKvCache::from_serialized: rotated=true requires L == max_size",
            max_size,
            l,
          )));
        } else if l > offset {
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "BatchRotatingKvCache::from_serialized: L (keys seq-len; mlx-lm getter emits keys[:_offset,:], so L <= _offset always)",
            "must satisfy L <= _offset",
            format_smolstr!("L={l}, _offset={offset}"),
          )));
        }
      }
      Ok(Box::new(c))
    }
    // mlx-lm / mlx-swift-lm `ArraysCache` (the generic SSM slot cache).
    // `_BaseCache.from_state` rebuilds via `__new__` + `state`/`meta_state`
    // setters (cache.py:168-174).
    KvCacheKind::ArraysCache => arrays::from_state_arrays(state, meta, /* mamba */ false),
    // `"MambaCache"` is mlx-swift-lm's `class MambaCache: ArraysCache`
    // (`KVCache.swift:1229`) — it adds **no** extra state or metadata, only
    // fixing `size = 2`; swift *saves* the kind `"MambaCache"`
    // (`KVCache.swift:1384`) and its own load arm rebuilds it via the
    // identical `restoreFromMetaState` (`KVCache.swift:1531-1533`,
    // == `ArraysCache(size: 2)`). The provenance is preserved via a
    // constructor-time `is_mamba` flag on [`ArraysCache`] so a
    // save-after-load emits `"MambaCache"` again (not `"ArraysCache"`); the
    // slot state itself is identical (the no-per-model-arch-porting rule is
    // preserved — see `arrays::ArraysCache`'s `# MambaCache` note).
    KvCacheKind::MambaCache => arrays::from_state_arrays(state, meta, /* mamba */ true),
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
