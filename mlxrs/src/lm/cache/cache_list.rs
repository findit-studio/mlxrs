//! [`CacheList`] — a composite cache wrapping one child [`KvCache`] per
//! sub-component (e.g. an attention + a Mamba/SSM cache in a hybrid model).
//!
//! Faithful 1:1 port of `mlx_lm.models.cache.CacheList`
//! (`mlx_lm/models/cache.py:814-902`), cross-checked against mlx-swift-lm's
//! `MLXLMCommon.CacheList` (`Libraries/MLXLMCommon/KVCache.swift:1248-1370`).
//!
//! ## Why the serialization follows Swift, not Python
//!
//! Python's `CacheList.state` is a *nested* `[c.state for c in caches]` and
//! its `meta_state` is a tuple `([class_names], [child_meta_states])`
//! (cache.py:829-848). The merged [`KvCache`] trait signatures are *flat*
//! (`state() -> Vec<Array>`, `meta_state() -> Vec<String>`), which cannot
//! hold a nested list. mlx-swift-lm hit the exact same constraint (its
//! `state` is `[MLXArray]`, `metaState` is `[String]`) and resolved it by
//! **flattening**: `state` is `caches.flatMap { $0.state }` and `metaState`
//! is `[childCount, (className, stateCount, metaCount, ...meta)*]`
//! (KVCache.swift:1262-1369). This port mirrors that Swift design exactly —
//! it is the only representation compatible with the trait, it is
//! information-equivalent to Python's nested form (the per-child grouping is
//! fully recoverable from the embedded `stateCount`/`metaCount`), and it
//! makes a Swift-written `CacheList` prompt cache load here unchanged.
//!
//! ## Reference-class-name serialization
//!
//! Each child's `className` is its **reference** class name (mlx-lm
//! `type(c).__name__`, cache.py:841; Swift `cacheClassName`,
//! KVCache.swift:1381-1392), so [`from_state`](super::from_state) rebuilds
//! every child via the crate's [`from_state`](super::from_state) keyed on
//! those source names — including a child that is itself a `"CacheList"`
//! (recursively, exactly cache.py:898 `globals()["CacheList"]`).
//!
//! ## `update` / `make_mask` are container-illegal
//!
//! mlx-lm's `CacheList` defines **no** `update`/`make_mask` (and
//! `_BaseCache` defines no `make_mask` either, cache.py:127-175): a
//! composite is never masked or updated directly — callers index a child
//! via [`get`](CacheList::get) and use *that* child's `update`/`make_mask`.
//! Swift makes this explicit with `fatalError("CacheList should not use
//! update(keys:values:) - use subscript access instead")`
//! (KVCache.swift:1270-1272). The merged [`KvCache`] trait requires both
//! methods, so they are implemented as a **recoverable** typed [`Error`] variant
//! — the project's no-panic-on-recoverable-paths equivalent of Swift's
//! trap; never an `unwrap`/`panic!`.

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, Error, InvariantViolationPayload,
    LayerKeyedPayload, LengthMismatchPayload, ParsePayload, Result,
  },
  lm::cache::{KvCache, MaskMode, RopeOffset},
};
use smol_str::format_smolstr;

/// A composite cache delegating to an ordered list of child caches.
///
/// Port of `mlx_lm.models.cache.CacheList` (cache.py:814-902). Used by
/// hybrid models that need more than one cache kind per layer (e.g. a
/// sliding-window attention cache plus an SSM/Mamba state). All trait
/// methods that have a meaningful composite semantics delegate across the
/// children exactly as the reference does:
///
/// - [`is_trimmable`](KvCache::is_trimmable): **all** children
///   (cache.py:821-822).
/// - [`trim`](KvCache::trim): trims **every** child, returns the **last**
///   child's count (cache.py:824-827).
/// - [`state`](KvCache::state): the flattened concatenation of every
///   child's state (Swift KVCache.swift:1274-1275).
/// - [`meta_state`](KvCache::meta_state): per-child reference class name +
///   `stateCount`/`metaCount` framing (Swift KVCache.swift:1315-1327).
/// - [`offset`](KvCache::offset): `max` child offset (Python `size()` =
///   `max(c.size())`, cache.py:884-885; each child's `size()` is its
///   `offset`).
/// - [`copy`](KvCache::copy): deep-copies every child (cache.py
///   `copy.deepcopy`; Swift KVCache.swift:1287-1291).
/// - [`nbytes`](KvCache::nbytes): the **sum** over children
///   (cache.py:891-892).
/// - [`is_empty`](KvCache::is_empty): the **first** child's emptiness
///   (cache.py:887-888).
pub struct CacheList {
  caches: Vec<Box<dyn KvCache>>,
}

impl CacheList {
  /// A composite over the given ordered child caches — mlx-lm
  /// `CacheList(*caches)` (cache.py:815-816) / Swift `CacheList(_ caches:
  /// KVCache...)` (KVCache.swift:1251-1254). The list may be empty (the
  /// degenerate composite); per-method behavior on an empty list mirrors
  /// the reference's `all(...)`/`max(...)`/`sum(...)`/`caches[0]` where
  /// defined, and is a recoverable value (never a panic) where Python would
  /// raise (empty `max`, `caches[0]` IndexError).
  pub fn new(caches: Vec<Box<dyn KvCache>>) -> Self {
    Self { caches }
  }

  /// The number of child caches (`len(self.caches)` / Swift
  /// `caches.count`).
  ///
  /// `#[allow(clippy::len_without_is_empty)]`: the obvious companion name
  /// `is_empty` is **already taken** by the [`KvCache`] trait impl with a
  /// *different* meaning — mlx-lm `CacheList.empty()` is *the first
  /// child's* emptiness (`self.caches[0].empty()`, cache.py:887-888), **not**
  /// "zero children". Adding an inherent `is_empty` would shadow/contradict
  /// that faithful semantics, so the "no child caches" predicate is the
  /// distinctly-named [`is_child_list_empty`](CacheList::is_child_list_empty)
  /// instead.
  #[allow(clippy::len_without_is_empty)]
  pub fn len(&self) -> usize {
    self.caches.len()
  }

  /// Whether there are **no child caches** (an empty composite). Distinct
  /// from [`is_empty`](KvCache::is_empty), which is *the first child's*
  /// emptiness — mlx-lm `CacheList.empty()`, cache.py:887-888.
  pub fn is_child_list_empty(&self) -> bool {
    self.caches.is_empty()
  }

  /// The i-th child cache, or `None` if out of range — mlx-lm
  /// `CacheList.__getitem__` (cache.py:818-819) / Swift `subscript`
  /// (KVCache.swift:1266-1268). Python/Swift index unchecked (an
  /// out-of-range access is an `IndexError`/trap); this returns `None`
  /// instead so misuse stays a recoverable, non-panicking path.
  pub fn get(&self, idx: usize) -> Option<&dyn KvCache> {
    self.caches.get(idx).map(|b| b.as_ref())
  }

  /// The i-th child cache mutably, or `None` if out of range — the
  /// `&mut` companion to [`get`](CacheList::get) (the generation loop
  /// indexes a child and calls *its* `update`/`make_mask`, exactly as
  /// mlx-lm/Swift do; the composite itself never updates).
  pub fn get_mut(&mut self, idx: usize) -> Option<&mut (dyn KvCache + 'static)> {
    self.caches.get_mut(idx).map(|b| b.as_mut())
  }
}

impl KvCache for CacheList {
  /// Composite offset = `max` of the children's [`offset`](KvCache::offset).
  ///
  /// mlx-lm's `CacheList` has **no** `.offset` attribute and **no**
  /// `make_mask` — it is a pure container indexed via
  /// [`get`](CacheList::get); the only positional accessor it defines is
  /// `size()` = `max(c.size() for c in self.caches)` (cache.py:884-885).
  /// The merged [`KvCache`] trait, however, *requires* `offset()` and maps
  /// it to mlx-lm's raw `cache.offset` **attribute** (the uncapped position
  /// the attention mask / RoPE use — see [`KvCache::offset`] and
  /// `RotatingKvCache`'s contract), which is a *different* quantity from
  /// `RotatingKVCache.size()` = `min(offset, max_size)` (cache.py:517-518).
  /// So the faithful composite value is the trait-consistent aggregation —
  /// `max` of each child's trait `offset()` — mirroring the `max` structure
  /// of `CacheList.size()` while staying on the single quantity the trait
  /// exposes (rather than mixing `KVCache`'s `offset` attribute with
  /// `RotatingKVCache`'s capped `size()`). For an unbounded child these
  /// coincide; they differ only once a `RotatingKvCache` child's raw offset
  /// exceeds its `max_size` (then this is the raw max, not the capped one)
  /// — a deliberate consequence of the trait exposing `offset`, not
  /// `size()`. An empty list yields 0 (mlx-lm's `max` over no children
  /// would raise; 0 matches `_BaseCache.size()`'s documented "always 0"
  /// default, cache.py:149-156 — a recoverable value, never a panic).
  fn offset(&self) -> usize {
    self.caches.iter().map(|c| c.offset()).max().unwrap_or(0)
  }

  /// Composite RoPE offset is meaningless (each child positions
  /// independently — there is no single RoPE offset for a heterogeneous
  /// composite). mlx-lm/Swift never call `ropeOffset` on a `CacheList`
  /// (they index a child). Returning the scalar composite
  /// [`offset`](KvCache::offset) keeps the trait total and non-panicking;
  /// callers needing a child's RoPE offset use
  /// [`get`](CacheList::get)`.rope_offset()`.
  fn rope_offset(&self) -> Result<RopeOffset> {
    Ok(RopeOffset::Scalar(self.offset()))
  }

  /// **Container-illegal.** mlx-lm `CacheList` defines no `update`;
  /// Swift's is `fatalError("CacheList should not use update(keys:values:)
  /// - use subscript access instead")` (KVCache.swift:1270-1272). Callers
  /// must index a child via [`get`](CacheList::get) /
  /// [`get_mut`](CacheList::get_mut) and update *that* child. A recoverable
  /// typed [`Error`] variant — the no-panic equivalent of Swift's trap.
  fn update(&mut self, _keys: &Array, _values: &Array) -> Result<(Array, Array)> {
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "CacheList::update",
      "is invalid — index a child via CacheList::get_mut and update that child",
    )))
  }

  /// The flattened concatenation of every child's state — Swift
  /// `caches.flatMap { $0.state }` (KVCache.swift:1274-1275); Python's
  /// nested `[c.state for c in caches]` (cache.py:830-831) flattened to the
  /// trait's `Vec<Array>` (the per-child grouping is recoverable from
  /// [`meta_state`](KvCache::meta_state)'s `stateCount`). Empty list ->
  /// empty state.
  ///
  /// Routes through [`state_into`](KvCache::state_into) so each child can
  /// push directly into the composite buffer — saves one `Vec<Array>`
  /// allocation per child compared to the previous per-child `state()` +
  /// `extend` pattern (KVC-7, #104). Behavior is byte-identical.
  fn state(&self) -> Result<Vec<Array>> {
    let mut out = Vec::new();
    for c in &self.caches {
      c.state_into(&mut out)?;
    }
    Ok(out)
  }

  /// Push every child's state into the caller's buffer — the buffer-reuse
  /// variant that lets a parent composite ([`super::save_prompt_cache`],
  /// a nested `CacheList`, …) avoid the per-call `Vec<Array>` allocation
  /// the default trait method would pay (KVC-7, #104). Equivalent to
  /// `caches.iter().try_for_each(|c| c.state_into(buf))` — appends each
  /// child's state in order, never clears `buf`.
  fn state_into(&self, buf: &mut Vec<Array>) -> Result<()> {
    for c in &self.caches {
      c.state_into(buf)?;
    }
    Ok(())
  }

  /// Split the flattened arrays back per child by each child's *current*
  /// `state().len()` and assign — Swift `state` setter
  /// (KVCache.swift:1276-1285): `stateLengths = caches.map {
  /// $0.state.count }`; per child slice `[start, start+length)`. Mirrors
  /// Python's `for c, s in zip(self.caches, v): c.state = s`
  /// (cache.py:834-836) once flattened. The split must consume the input
  /// **exactly**; a length mismatch is a recoverable typed [`Error`] variant
  /// (never a slice panic / silent truncation).
  ///
  /// **Transactional**: the restore is staged onto *copies* of the
  /// children and `self.caches` is replaced only once **every** child's
  /// `set_state` (and the initial copy) has succeeded. If any child
  /// rejects its chunk (e.g. a later child's key array has a bad rank) the
  /// original `CacheList` is left **completely unchanged** — never a
  /// half-applied mix of old/new children (which would corrupt generation
  /// state and make retry/rollback unsafe). This mirrors the crate-wide
  /// "no partial mutation on a recoverable error" convention the sibling
  /// caches already follow (e.g. `RotatingKvCache::set_meta_state`
  /// parses+validates all fields before assigning any). Swift mutates the
  /// children in place (KVCache.swift:1279-1283); staging is the
  /// `Result`-faithful, corruption-free equivalent.
  fn set_state(&mut self, state: Vec<Array>) -> Result<()> {
    // Per-child state-array counts, taken from the children's *current*
    // state (Swift's `caches.map { $0.state.count }`). Uses the cheap
    // [`KvCache::state_count`] trait helper (added in this PR for exactly
    // this — Copilot flagged the prior `c.state()?.len()` as wasteful
    // because it cloned/materialized every child's full state just to
    // read its length).
    let mut lengths = Vec::with_capacity(self.caches.len());
    for c in &self.caches {
      lengths.push(c.state_count()?);
    }
    let total: usize = lengths.iter().sum();
    if total != state.len() {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "CacheList::set_state: flattened state array count vs sum of children state_count",
        total,
        state.len(),
      )));
    }
    // Stage onto copies first: copy every child, apply each per-child
    // chunk to the *staged* copy, and only swap `self.caches` after ALL
    // succeed. A copy failure or any child `set_state` error returns `Err`
    // with the original `CacheList` untouched (no partial mutation).
    let mut staged: Vec<Box<dyn KvCache>> = Vec::with_capacity(self.caches.len());
    for c in &self.caches {
      staged.push(c.copy()?);
    }
    // Consume `state` front-to-back into per-child chunks without cloning
    // (move each array exactly once into its staged child's `set_state`).
    let mut it = state.into_iter();
    for (c, &len) in staged.iter_mut().zip(lengths.iter()) {
      let chunk: Vec<Array> = it.by_ref().take(len).collect();
      // `take(len)` yields at most `len`; with the verified `total ==
      // state.len()` and front-to-back consumption every chunk is exactly
      // `len`, so no short-chunk can reach a child.
      c.set_state(chunk)?;
    }
    // All children restored successfully on the staged copies — commit.
    self.caches = staged;
    Ok(())
  }

  /// Force-evaluate every child cache's own stored arrays in place — the
  /// per-chunk prefill memory barrier (see [`KvCache::materialize`]).
  /// Delegates to each child's `materialize` (mirroring how
  /// [`state`](KvCache::state) flattens each child's `state()`), so each
  /// concrete child evals its genuine stored buffers rather than its
  /// serialization view. A no-op for an empty list.
  fn materialize(&mut self) -> Result<()> {
    for c in &mut self.caches {
      c.materialize()?;
    }
    Ok(())
  }

  /// The flattened per-child framing — Swift `metaState`
  /// (KVCache.swift:1315-1327): `[childCount, (className, stateCount,
  /// metaCount, ...childMeta)*]`. `className` is each child's **reference**
  /// class name (mlx-lm `type(c).__name__`, cache.py:841) so
  /// [`from_state`](super::from_state) rebuilds the right concrete kind;
  /// `stateCount`/`metaCount` let it slice the flattened
  /// [`state`](KvCache::state) / meta back per child. Information-equivalent
  /// to Python's `([class_names], [child_meta_states])` (cache.py:838-843).
  ///
  /// Routes through [`meta_state_into`](KvCache::meta_state_into) so each
  /// child pushes directly into the composite buffer — saves one
  /// `Vec<String>` allocation per child compared to the previous per-child
  /// `meta_state()` + `extend` pattern (KVC-6, #103). The `metaCount` slot
  /// is reserved before each child appends, then patched in place by
  /// snapshotting `buf.len()` before/after — preserves the swift-faithful
  /// framing byte-identically.
  fn meta_state(&self) -> Vec<String> {
    let mut out = Vec::new();
    self.meta_state_into(&mut out);
    out
  }

  /// Push the flattened per-child framing into a caller-provided buffer —
  /// the buffer-reuse variant ([`meta_state_into`](KvCache::meta_state_into))
  /// override for `CacheList`. A nested `CacheList` recurses through this
  /// same override so deep composites pay exactly **one** `Vec<String>`
  /// allocation at the outermost call (KVC-6, #103). Layout is byte-
  /// identical to [`meta_state`](KvCache::meta_state).
  fn meta_state_into(&self, buf: &mut Vec<String>) {
    buf.push(self.caches.len().to_string());
    for c in &self.caches {
      let class_name = c.reference_class_name();
      // `state_count()` may fail if a concrete cache falls back to the
      // trait default `Ok(self.state()?.len())` and `state()` itself
      // fails. The framing needs an accurate length — fall back to
      // `state()?.len()` (paying the re-clone the #82 optimization
      // normally avoids, but only on the rare edge case) so the framing
      // round-trips correctly even when `state_count` is fallible; if
      // BOTH fail, the round-trip is non-viable anyway and `0` is
      // detected by from_state's framing/payload-mismatch check.
      let state_count = c
        .state_count()
        .or_else(|_| c.state().map(|s| s.len()))
        .unwrap_or(0);
      buf.push(class_name.to_string());
      buf.push(state_count.to_string());
      // Reserve the `metaCount` slot, then push the child's meta directly
      // into `buf` via `meta_state_into` — no intermediate per-child
      // `Vec<String>`. Snapshot `buf.len()` before/after to compute the
      // count, then patch the reserved slot. Identical framing to the
      // pre-PR `child_meta.len()` value (a deterministic non-overflowing
      // subtraction since `meta_state_into` only appends).
      let count_slot = buf.len();
      buf.push(String::new());
      let before = buf.len();
      c.meta_state_into(buf);
      let appended = buf.len() - before;
      buf[count_slot] = appended.to_string();
    }
  }

  /// `set_meta_state` is not a valid direct operation on a `CacheList`:
  /// Swift's setter is `assertionFailure("CacheList.metaState should not
  /// be set directly. Use CacheList.fromState() instead")`
  /// (KVCache.swift:1328-1331). The round-trip path is
  /// [`from_state`](super::from_state)`("CacheList", state, meta)`, which
  /// rebuilds children atomically. A recoverable typed [`Error`] variant — the
  /// no-panic equivalent of Swift's `assertionFailure`.
  fn set_meta_state(&mut self, _m: &[String]) -> Result<()> {
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "CacheList::set_meta_state (direct call invalid)",
      "must reconstruct via from_state(\"CacheList\", state, meta) (Swift: CacheList.fromState)",
    )))
  }

  /// `all(c.is_trimmable() for c in self.caches)` — mlx-lm
  /// `CacheList.is_trimmable` (cache.py:821-822) / Swift
  /// `caches.allSatisfy { $0.isTrimmable }` (KVCache.swift:1293-1295).
  /// `all(...)` over an empty list is `true` (vacuous), matching mlx-lm.
  fn is_trimmable(&self) -> bool {
    self.caches.iter().all(|c| c.is_trimmable())
  }

  /// Trim **every** child by `n`, returning the **last** child's trimmed
  /// count — mlx-lm `CacheList.trim` (cache.py:824-827: the loop calls
  /// `c.trim(n)` for all children and `return m` is the *last* iteration's
  /// value) / Swift KVCache.swift:1297-1304 (`result = cache.trim(n)` in a
  /// loop, `return result`). An empty list returns 0 (mlx-lm's `m` is never
  /// assigned and `UnboundLocalError` would raise; 0 is the recoverable
  /// non-panicking equivalent — and matches `trim_prompt_cache`'s
  /// short-circuit-0 for an empty cache, cache.py:109-110).
  ///
  /// PRE-VALIDATED short-circuit (deviation from references' deliberate
  /// sequential non-transactional design, motivated by Rust's stricter
  /// fallibility vs Python's infallible int / Swift's infallible Int
  /// trim — same Rust-fallibility-gap pattern that motivated
  /// `set_state`'s transactional override). The pre-validation is
  /// *partial-atomicity*: it guarantees no child is mutated when ANY child
  /// is non-trimmable (the common adversarial / mis-configured-shape
  /// path), matching `cache.py:88-111`
  /// `can_trim_prompt_cache`/`trim_prompt_cache`'s `all(is_trimmable())`
  /// gate. It does NOT guarantee full transactional rollback after a
  /// child's per-`trim(n)?` returns `Err` (e.g., a rare allocation
  /// failure mid-loop): in that case earlier children are already
  /// trimmed when the `Err` surfaces, and recovery is via `from_state`
  /// from a prior serialized state (same recovery semantics as a failed
  /// `set_state`). NOT fully transactional — partial mutation on
  /// mid-loop `Err` is possible.
  fn trim(&mut self, n: usize) -> Result<usize> {
    // PRE-VALIDATED short-circuit: gate via `is_trimmable()` across *all*
    // children BEFORE mutating any. mlx-lm/swift trim is INFALLIBLE
    // (Python int / Swift Int) so they never observe a partial mutation;
    // mlxrs's `trim` is `Result<usize>` because `Array` ops are fallible.
    // The gate prevents partial mutation when ANY child is non-trimmable
    // (the common shape/state-mismatch path); it does NOT (and cannot
    // cheaply) prevent partial mutation when a per-child `trim(n)?` errors
    // mid-loop (e.g., a rare allocation failure). Recovery for the
    // mid-loop-Err case is `from_state` from a prior serialized state.
    //
    // Two-phase loop (partial-atomicity only):
    //   (1) PRE-VALIDATE: short-circuit-Ok(0) when ANY child is
    //       non-trimmable (matches mlx-lm's semantic that
    //       `cache.py:88-111` `can_trim_prompt_cache`/`trim_prompt_cache`
    //       gates on `all(is_trimmable())`). After this check, every child
    //       trim is — for the common case — guaranteed non-Err (rare
    //       allocation-failure aside).
    //   (2) APPLY: loop child trims; return the LAST child's trimmed count
    //       (mlx-lm `[c.trim(n) for c in self.caches][-1]` ≈ the for-loop's
    //       final iteration). If a rare allocation-failure `Err` slips
    //       through phase 1's screening, surface it; the cache is then in
    //       a PARTIALLY-trimmed state (earlier children already mutated)
    //       and the caller can rebuild from a serialized
    //       `state()`/`from_state()` snapshot — same recovery semantics
    //       as a failed `set_state`. This is NOT a full transactional
    //       rollback (a true rollback would require cloning each child's
    //       state pre-loop, which costs O(state) allocs the common case
    //       never needs).
    //
    // Common case: a real prompt-cache (every child returns the same trim
    // count) hits the loop just like before, faithful to mlx-lm.
    if !self.is_trimmable() {
      return Ok(0);
    }
    let mut last = 0;
    for c in &mut self.caches {
      last = c.trim(n)?;
    }
    Ok(last)
  }

  /// **Container-illegal.** Neither mlx-lm `CacheList` nor `_BaseCache`
  /// defines `make_mask` (cache.py:127-175, 814-902): a composite is never
  /// masked directly — callers index a child via [`get`](CacheList::get)
  /// and use *that* child's `make_mask`. A recoverable typed [`Error`] variant
  /// (the no-panic equivalent of the `AttributeError` a direct
  /// `CacheList.make_mask(...)` raises in mlx-lm).
  fn make_mask(
    &self,
    _n: usize,
    _window_size: Option<usize>,
    _return_array: bool,
  ) -> Result<MaskMode> {
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "CacheList::make_mask (composite is never masked directly)",
      "must mask per child via CacheList::get (mlx-lm CacheList/_BaseCache define no make_mask; masking is per child)",
    )))
  }

  /// The **sum** of children's `nbytes` — mlx-lm `CacheList.nbytes`
  /// (cache.py:891-892: `sum(c.nbytes for c in self.caches)`). Empty list
  /// -> 0 (`sum` of nothing).
  fn nbytes(&self) -> usize {
    self.caches.iter().map(|c| c.nbytes()).sum()
  }

  /// The **first** child's emptiness — mlx-lm `CacheList.empty()`
  /// (cache.py:887-888: `return self.caches[0].empty()`). An empty list is
  /// reported empty (mlx-lm's `self.caches[0]` would raise `IndexError`;
  /// `true` is the recoverable non-panicking equivalent — a list with no
  /// children holds nothing).
  fn is_empty(&self) -> bool {
    match self.caches.first() {
      Some(c) => c.is_empty(),
      None => true,
    }
  }

  /// A deep, independent copy — mlx-lm `copy.deepcopy(cache)` (the generic
  /// deep copy `copy_prompt_cache` uses) / Swift `caches.map { $0.copy() }`
  /// then `CacheList(caches:)` (KVCache.swift:1287-1291). Each child is
  /// deep-copied via its own [`copy`](KvCache::copy); a child clone failure
  /// is propagated (never swallowed into a partially-built composite, never
  /// panicked).
  fn copy(&self) -> Result<Box<dyn KvCache>> {
    let mut copied = Vec::with_capacity(self.caches.len());
    for c in &self.caches {
      copied.push(c.copy()?);
    }
    Ok(Box::new(Self { caches: copied }))
  }

  /// `Some(self)` — a hybrid model holding `Box<dyn KvCache>` per layer can
  /// downcast to reach the `CacheList`-inherent indexing API
  /// ([`CacheList::get`] / [`CacheList::get_mut`]) and delegate to the
  /// right child cache (faithful to swift's `cache as? CacheList` pattern).
  fn as_cache_list(&self) -> Option<&CacheList> {
    Some(self)
  }

  /// `Some(self)` — the `&mut` companion to [`as_cache_list`](
  /// KvCache::as_cache_list); the generation loop needs the mutating
  /// indexing API ([`CacheList::get_mut`] for a child's `update` /
  /// `make_mask`).
  fn as_cache_list_mut(&mut self) -> Option<&mut CacheList> {
    Some(self)
  }

  /// The flattened `state()` length without cloning every child's arrays
  /// — sum of each child's [`state_count`](KvCache::state_count). Each
  /// child delegates to its own trait method (a nested `CacheList` child
  /// recurses through this same override; non-CacheList children fall
  /// through the trait's `state()?.len()` default until they grow their own
  /// O(1) override). Preserves the behavior of [`state`](KvCache::state)
  /// (whose body is `caches.iter().map(|c| c.state()?).flatten()`) without
  /// the per-child full-state clone.
  fn state_count(&self) -> Result<usize> {
    let mut total = 0usize;
    for c in &self.caches {
      total = total
        .checked_add(c.state_count()?)
        .ok_or(Error::ArithmeticOverflow(ArithmeticOverflowPayload::new(
          "CacheList::state_count",
          "usize",
        )))?;
    }
    Ok(total)
  }

  /// `"CacheList"` — mlx-lm's `type(CacheList).__name__` (`cache.py:56`) /
  /// mlx-swift-lm `case is CacheList: return "CacheList"`
  /// (`KVCache.swift:1389`). [`super::from_state`] routes `"CacheList"` to
  /// the recursive `cache_list_from_state` dispatcher.
  fn reference_class_name(&self) -> &'static str {
    "CacheList"
  }

  /// P1 #110: per-layer fast-path downcast target — see the
  /// [`KvCache`]-trait doc's **Per-layer fast-path convention**.
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }

  /// Transactional override of [`KvCache::from_serialized`] — leaves `self`
  /// byte-identical to its pre-call state on every recoverable error
  /// (malformed framing, non-numeric child count / `stateCount` /
  /// `metaCount`, oversized child count, declared-vs-available
  /// `stateCount`/`metaCount` mismatch, nested-depth overflow, any failing
  /// child rebuild).
  ///
  /// **Highest-payoff override on this trait.** `CacheList` is the most
  /// error-prone meta consumer in the cache module: its flattened
  /// `[childCount, (className, stateCount, metaCount, ...meta)*]` framing
  /// must be parsed *and* every child must be rebuilt through
  /// [`super::from_state`] (which itself can fail per kind), all before any
  /// of `self.caches` is touched. The default trait impl would call
  /// `set_state` first (which itself stages onto copies — see
  /// [`KvCache::set_state`] for `CacheList` — and is internally
  /// transactional) and then `set_meta_state`, which is hard-coded to
  /// reject any direct call (Swift's `assertionFailure`); so the default
  /// impl is *unconditionally* broken for `CacheList` (every call would
  /// return `Err`). This override is what makes [`from_serialized`](
  /// KvCache::from_serialized) work on `CacheList` at all, AND it does so
  /// while preserving the leaves-self-unchanged contract: the entire
  /// children rebuild (`build_cache_list_children` — class-name dispatch,
  /// recursive nested `CacheList`s, depth-budget) runs into a local
  /// `Vec<Box<dyn KvCache>>` with `self.caches` untouched; only on
  /// `Ok(children)` is `self` committed via one infallible move.
  #[allow(clippy::wrong_self_convention)] // see KvCache::from_serialized
  fn from_serialized(&mut self, state: Vec<Array>, meta: &[String]) -> Result<()> {
    let children = build_cache_list_children(state, meta, CACHE_LIST_MAX_NESTING_DEPTH)?;
    *self = CacheList::new(children);
    Ok(())
  }
}

/// Reconstruct a [`CacheList`] from its flattened `state` + `meta_state` —
/// mlx-lm `CacheList.from_state` (cache.py:894-900: `obj.caches =
/// [globals()[c].from_state(s, m) for s, c, m in zip(state,
/// *meta_state)]`) / Swift `CacheList.fromState`
/// (KVCache.swift:1335-1369). The flattened `meta` is
/// `[childCount, (className, stateCount, metaCount, ...childMeta)*]`; for
/// each child we slice `stateCount` arrays off `state` and `metaCount`
/// strings off `meta`, then rebuild via the crate's
/// [`from_state`](super::from_state)`(className, childState, childMeta)`
/// keyed on the **reference** class name — so a child that is itself
/// `"CacheList"` recurses through this same function (exactly cache.py:898
/// `globals()["CacheList"]`).
///
/// Every malformed-framing case (missing/non-numeric child count,
/// truncated per-child fields, a declared `stateCount`/`metaCount`
/// exceeding what was provided) is a recoverable typed [`Error`] variant — never
/// an out-of-bounds slice panic. Unlike Swift (which clamps with
/// `min(...)`, KVCache.swift:1357-1361, silently shortening an
/// inconsistent child slice), this **rejects** the inconsistency so a
/// corrupt prompt cache cannot rebuild a child from a truncated state.
///
/// **Nesting-depth bounded.** A `"CacheList"` child recurses
/// (cache.py:898 `globals()["CacheList"]`); a forged prompt cache can
/// encode an arbitrarily deep single-child chain `CacheList -> CacheList
/// -> … -> []` using only metadata strings and **zero arrays**, so the
/// `child_count`/`stateCount` allocation and length guards never reject it
/// (every level is a well-formed `childCount=1` frame). Unbounded native
/// recursion on that chain exhausts the thread stack — a **process abort**,
/// not a recoverable `Error`, on the public [`from_state`](super::from_state)
/// load path. This is the same forged-prompt-cache defect class as the
/// `child_count` allocation-DoS bound above, along its *depth* dimension;
/// reconstruction therefore carries an explicit remaining-depth budget
/// ([`CACHE_LIST_MAX_NESTING_DEPTH`]) and a `CacheList`-into-`CacheList`
/// step that would exceed it is rejected as a recoverable
/// typed [`Error`] variant (never a stack-overflow abort). The budget is far
/// above any real hybrid-model nesting (which is a couple of levels), so
/// every faithful round-trip is unaffected — only a pathological forged
/// chain is rejected.
pub(crate) fn cache_list_from_state(
  state: Vec<Array>,
  meta: &[String],
) -> Result<Box<dyn KvCache>> {
  cache_list_from_state_bounded(state, meta, CACHE_LIST_MAX_NESTING_DEPTH)
}

/// Maximum `CacheList`-within-`CacheList` nesting depth accepted by
/// [`cache_list_from_state`].
///
/// mlx-lm / mlx-swift-lm impose **no** nesting limit (Python would raise a
/// `RecursionError`, Swift would crash), so — exactly like the
/// `child_count` allocation bound — there is no reference value to mirror:
/// this is purely the no-process-abort safety floor for a forged prompt
/// cache. Real hybrid models compose a *handful* of caches at most one
/// level deep (and a nested `CacheList` is itself rare), so this generous
/// ceiling never rejects a legitimate prompt cache while still bounding the
/// native recursion well before any reachable stack limit.
const CACHE_LIST_MAX_NESTING_DEPTH: usize = 64;

/// Depth-budgeted core of [`cache_list_from_state`]: identical framing
/// validation, but a child whose reference class name is `"CacheList"`
/// recurses **directly here** with a decremented `depth_budget` (rather
/// than through the public dispatcher) so the recursion is bounded. A
/// non-`CacheList` child is a leaf kind that cannot recurse, so it still
/// goes through the unchanged public [`from_state`](super::from_state).
/// `depth_budget == 0` on entry means the chain is one level deeper than
/// [`CACHE_LIST_MAX_NESTING_DEPTH`] allows — a recoverable
/// typed [`Error`] variant, never a stack-overflow abort.
fn cache_list_from_state_bounded(
  state: Vec<Array>,
  meta: &[String],
  depth_budget: usize,
) -> Result<Box<dyn KvCache>> {
  let children = build_cache_list_children(state, meta, depth_budget)?;
  Ok(Box::new(CacheList::new(children)))
}

/// The atomic, fallible inner build of a `CacheList`'s children
/// `Vec<Box<dyn KvCache>>` from its flattened `(state, meta)` — every
/// validation, allocation, and recursive child build happens here BEFORE
/// any caller's `self` is touched. Shared by:
///
/// - [`cache_list_from_state_bounded`] (which boxes the result into a
///   `CacheList`-as-`KvCache` for the [`super::from_state`] dispatch path);
/// - [`CacheList::from_serialized`] (the trait-method override, which
///   commits the children Vec into `self.caches` via a single infallible
///   `*self = CacheList::new(children)` only after this returns `Ok`).
///
/// Factoring this out is what makes the override's leaves-self-unchanged
/// guarantee load-bearing for the highest-payoff override on this trait:
/// the full multi-child framing parse / class-name dispatch / recursive
/// nested-CacheList build all run on locals; the caller's `self.caches`
/// is replaced atomically only on success. A malformed framing /
/// out-of-range `stateCount` / nested-depth overflow / failing child
/// rebuild leaves the parent `CacheList` byte-identical to its pre-call
/// state.
fn build_cache_list_children(
  state: Vec<Array>,
  meta: &[String],
  depth_budget: usize,
) -> Result<Vec<Box<dyn KvCache>>> {
  // Reject the over-deep chain BEFORE parsing this level's frame: a forged
  // single-child `CacheList -> CacheList -> …` consumes one budget unit per
  // level, and `0` here means reconstructing *this* CacheList already
  // exceeds `CACHE_LIST_MAX_NESTING_DEPTH` — a recoverable error rather
  // than another native recursion frame toward a stack-overflow abort.
  let Some(child_depth_budget) = depth_budget.checked_sub(1) else {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "CacheList::from_state: nesting depth (deeper chain rejected as a forged/corrupt prompt cache, not a stack-overflow abort)",
      "CACHE_LIST_MAX_NESTING_DEPTH",
      CACHE_LIST_MAX_NESTING_DEPTH as u64,
      CACHE_LIST_MAX_NESTING_DEPTH as u64,
    )));
  };

  let first = meta.first().ok_or_else(|| {
    Error::InvariantViolation(InvariantViolationPayload::new(
      "CacheList::from_state: meta_state",
      "must be non-empty (first element is child count)",
    ))
  })?;
  let child_count: usize = first.parse().map_err(|e: std::num::ParseIntError| {
    Error::Parse(ParsePayload::new(
      "CacheList::from_state: child count",
      "usize",
      Box::new(e),
    ))
  })?;

  // Bound `child_count` against the metadata length BEFORE any allocation.
  // Each child frame is at minimum 3 meta fields (`className`,
  // `stateCount`, `metaCount`) after the leading count — its own meta
  // values only add to that — so a well-formed framing necessarily has
  // `child_count <= (meta.len() - 1) / 3`. A corrupt/forged prompt cache
  // with a huge numeric first field (e.g. `usize::MAX`) would otherwise
  // reach `Vec::with_capacity(child_count)` below and panic on capacity
  // overflow / abort on OOM *before* the per-child truncation checks could
  // reject it — a panic/abort on the public `from_state` load path. Reject
  // it here as a recoverable `Error::CapExceeded` instead, and (since the count
  // is now bounded by `meta.len()`) grow the children `Vec` on demand
  // rather than pre-reserving an attacker-controlled capacity.
  let max_children = meta.len().saturating_sub(1) / 3;
  if child_count > max_children {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "CacheList::from_state: child count (3 framing fields per child)",
      "max_children_for_meta",
      max_children as u64,
      child_count as u64,
    )));
  }

  let mut children: Vec<Box<dyn KvCache>> = Vec::new();
  let mut meta_idx = 1usize; // skip childCount (Swift: `var metaIdx = 1`)
  // Move the flattened arrays through an iterator so each is consumed
  // exactly once (no `Array` clone) into the child it belongs to.
  let mut state_it = state.into_iter();
  let mut state_remaining = state_it.len();

  for child in 0..child_count {
    // Layer key naming the offending child index — wraps the typed inner
    // error so runtime `child` survives end-to-end without runtime String
    // payloads in the inner variant.
    let layer = |inner: Error| -> Error {
      Error::LayerKeyed(LayerKeyedPayload::new(
        format_smolstr!("child {child}"),
        inner,
      ))
    };
    // Need `className`, `stateCount`, `metaCount` (Swift guard
    // `metaIdx + 2 < metaState.count`, KVCache.swift:1345; we use
    // `+ 2 >= len` since Rust slices are 0-based half-open).
    if meta_idx + 2 >= meta.len() {
      return Err(layer(Error::InvariantViolation(
        InvariantViolationPayload::new(
          "CacheList::from_state: meta_state truncated at child frame (need class/state/meta counts)",
          "must have at least 3 meta entries remaining for each child frame",
        ),
      )));
    }
    // `class_name` is only consumed as `&str` (equality vs `"CacheList"` and
    // the `super::from_state(&class_name, ...)` call below) — borrow rather
    // than clone (the cache-allocation discipline; the borrow stays valid
    // across `meta_idx += 3` because `meta_idx` is a `usize` index, not a
    // borrow into `meta`, and `meta` is untouched for the remainder of this
    // iteration).
    let class_name: &str = &meta[meta_idx];
    let state_count: usize = meta[meta_idx + 1]
      .parse()
      .map_err(|e: std::num::ParseIntError| {
        layer(Error::Parse(ParsePayload::new(
          "CacheList::from_state: child stateCount",
          "usize",
          Box::new(e),
        )))
      })?;
    let meta_count: usize = meta[meta_idx + 2]
      .parse()
      .map_err(|e: std::num::ParseIntError| {
        layer(Error::Parse(ParsePayload::new(
          "CacheList::from_state: child metaCount",
          "usize",
          Box::new(e),
        )))
      })?;
    meta_idx += 3;

    // Slice `metaCount` child-meta strings. Reject (not clamp) an
    // out-of-range claim.
    let meta_end = meta_idx.checked_add(meta_count).ok_or_else(|| {
      layer(Error::ArithmeticOverflow(
        ArithmeticOverflowPayload::with_operands(
          "CacheList::from_state: meta_idx + metaCount",
          "usize",
          [
            ("meta_idx", meta_idx as u64),
            ("metaCount", meta_count as u64),
          ],
        ),
      ))
    })?;
    if meta_end > meta.len() {
      return Err(layer(Error::LengthMismatch(LengthMismatchPayload::new(
        "CacheList::from_state: child metaCount exceeds remaining meta values",
        meta.len().saturating_sub(meta_idx),
        meta_count,
      ))));
    }
    let child_meta = &meta[meta_idx..meta_end];
    meta_idx = meta_end;

    // Take `stateCount` arrays off the flattened state. Reject (not clamp,
    // unlike Swift's `min(...)`) a claim exceeding what remains, so a
    // corrupt cache cannot rebuild a child from a too-short state.
    if state_count > state_remaining {
      return Err(layer(Error::LengthMismatch(LengthMismatchPayload::new(
        "CacheList::from_state: child stateCount exceeds remaining state arrays",
        state_remaining,
        state_count,
      ))));
    }
    let child_state: Vec<Array> = state_it.by_ref().take(state_count).collect();
    state_remaining -= state_count;

    // Rebuild the concrete child via the crate's reference-name-keyed
    // dispatcher (cache.py:898 `globals()[c].from_state(...)`). A
    // `"CacheList"` child is the *only* recursive kind: recurse **directly**
    // into the depth-budgeted core with the decremented budget so a forged
    // deep chain is rejected before exhausting the stack. Every other
    // (leaf) kind cannot recurse, so it still goes through the unchanged
    // public `super::from_state` exactly as before — identical dispatch and
    // behavior for all non-CacheList children.
    let child_cache = if class_name == "CacheList" {
      cache_list_from_state_bounded(child_state, child_meta, child_depth_budget)?
    } else {
      super::from_state(class_name, child_state, child_meta)?
    };
    children.push(child_cache);
  }

  // A faithful round-trip consumes the framing exactly. Trailing
  // unconsumed state/meta means the framing disagrees with the payload —
  // reject rather than silently ignore (a corrupt/forged prompt cache).
  if state_remaining != 0 {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "CacheList::from_state: state array consumption after all children (framing/payload mismatch)",
      0,
      state_remaining,
    )));
  }
  if meta_idx != meta.len() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "CacheList::from_state: meta value consumption after all children (framing/payload mismatch)",
      meta.len(),
      meta_idx,
    )));
  }

  Ok(children)
}
