//! [`ArraysCache`] — the generic slot-state cache for SSM / Mamba models.

use std::str::FromStr;

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, Error, InvariantViolationPayload,
    OutOfRangePayload, ParsePayload, RankMismatchPayload, Result,
  },
  lm::cache::{KvCache, MaskMode, mask},
  ops,
};
use smol_str::format_smolstr;

/// Hard upper bound on the `slotCount` restored from a serialized
/// `ArraysCache` meta_state — mlx-swift-lm's `Array(repeating: nil, count:
/// slotCount)` (`KVCache.swift:1202`) and mlx-lm's equivalent
/// (`cache.py:642-665`) both allocate a `[None] * slotCount` list with no
/// upper bound, so a forged/corrupt prompt cache carrying `slotCount =
/// INT32_MAX` would attempt a multi-GB allocation that `try_reserve_exact`
/// rejects as OOM only **after** the allocator has been asked. Add an
/// explicit cap that fails fast on absurd values (#99 — derived from
/// realistic model maxes: Mamba's `MambaCache` is 2 slots,
/// `KVCache.swift:1230-1245`; SSM models typically ≤ 64 slots; the cap is
/// chosen ~4 orders of magnitude above that real-world ceiling so a
/// legitimately-large valid prompt cache is never rejected, while a forged
/// `INT32_MAX`-sized one is). Bounds the slot **allocation**; per the
/// project's bounded-memory-caps discipline, the cumulative work is still
/// bounded by the existing `try_reserve_exact` + `isize::MAX/elem_size`
/// checks that run after this gate (so an evader who passes
/// `MAX_SLOT_COUNT` still faces those secondary gates).
pub const MAX_SLOT_COUNT: usize = 1 << 20;

/// Parse a comma-separated meta_state value into `Vec<T>`, recoverable on
/// any parse error. Generic so `presentSlots` (slot indices, `usize`) and
/// `leftPadding` (signed offsets, `i32`) parse to their producer-native
/// types — see the call sites in [`ArraysCache::set_meta_state`].
///
/// **Bounded:** `max_elems` caps both the upfront capacity reservation and
/// the streamed parse — a forged CSV whose comma-count exceeds `max_elems`
/// is rejected with `Error::CapExceeded` BEFORE any `Vec<T>` allocation.
/// The producer side of
/// `meta_state` only ever emits `slotCount` indices for `presentSlots`
/// and at most one entry per slot for `leftPadding`, so a `max_elems =
/// slotCount` (already cap-gated by [`MAX_SLOT_COUNT`]) bounds the
/// entire parse to slot_count items — closing the "small `slotCount` +
/// huge CSV payload" evasion of the `MAX_SLOT_COUNT` gate. Comma-count
/// is computed by `bytes().filter()`, NOT `split().count()`, so the
/// pre-check itself allocates no `&str` slices.
fn parse_csv<T>(
  s: &str,
  what_context: &'static str,
  what_input_kind: &'static str,
  max_elems: usize,
) -> Result<Vec<T>>
where
  T: FromStr,
  T::Err: std::error::Error + Send + Sync + 'static,
{
  if s.is_empty() {
    return Ok(Vec::new());
  }
  // Cheap upper bound on element count via comma byte-counting — no
  // `&str` slice allocation, no parse work. `split(',')` yields
  // `commas + 1` substrings, so reject when that exceeds `max_elems`.
  let comma_count = s.bytes().filter(|&b| b == b',').count();
  let upper_bound = comma_count.saturating_add(1);
  if upper_bound > max_elems {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      what_context,
      "max_elems",
      max_elems as u64,
      upper_bound as u64,
    )));
  }
  // Pre-reserve so the streamed parse can fail fast on OOM for a hostile
  // (but in-bound) `max_elems` rather than amortized grow.
  let mut out: Vec<T> = Vec::new();
  out
    .try_reserve_exact(upper_bound)
    .map_err(|_| Error::OutOfMemory)?;
  for p in s.split(',') {
    let v = p.parse::<T>().map_err(|e| {
      Error::Parse(ParsePayload::new(
        what_context,
        what_input_kind,
        Box::new(e),
      ))
    })?;
    out.push(v);
  }
  Ok(out)
}

/// Generic array-slot cache — opaque per-slot state tensors, **not** 4-D
/// K/V. The slot count is fixed at construction
/// ([`new(size)`](ArraysCache::new) /
/// [`with_left_padding(size, ..)`](ArraysCache::with_left_padding)) but
/// can change via state restoration:
/// [`set_state`](KvCache::set_state) replaces `self.cache` with exactly
/// `state.len()` entries and [`set_meta_state`](KvCache::set_meta_state)
/// can later rebuild to the saved `slotCount` (the
/// `restoreFromMetaState` round-trip exposes this — mirrors
/// `KVCache.swift:1192-1207`).
///
/// Faithful 1:1 port of `mlx_lm.models.cache.ArraysCache`
/// (`mlx_lm/models/cache.py:594-730`, the authoritative spec), cross-checked
/// against mlx-swift-lm's `MLXLMCommon.ArraysCache`
/// (`KVCache.swift:1102-1227`).
///
/// State-space (Mamba-style) models keep their recurrent state here via the
/// slot accessors ([`get`](ArraysCache::get) / [`set`](ArraysCache::set),
/// mlx-lm `__getitem__` / `__setitem__`) and [`state`](KvCache::state) — not
/// `update_and_fetch`. mlx-lm's `ArraysCache` therefore has **no**
/// `update_and_fetch`; consequently the [`KvCache::update`] trait method here
/// is a recoverable [`Error::InvariantViolation`] ("ArraysCache::update is unsupported")
/// — accurately reporting the unsupported-operation condition rather than
/// misleadingly suggesting wrong-shaped tensors, exactly matching the
/// reference's absence of a K/V update path. It
/// has no `offset` / `size()` override either, so [`offset`](KvCache::offset)
/// is `0` (`_BaseCache.size()`); swift's `BaseKVCache.offset` likewise stays
/// `0` (the SSM models never advance it).
///
/// # `MambaCache`
///
/// mlx-swift-lm exposes `MambaCache: ArraysCache` whose only specialization
/// is `init { super.init(size: 2) }` (`KVCache.swift:1230-1245`). Per the
/// project's no-per-model-arch-porting rule (no Mamba architecture is ported
/// into `mlxrs`), there is **no** separate `MambaCache` type: a Mamba-style
/// architecture simply constructs [`ArraysCache::mamba`](ArraysCache::mamba)
/// (the 2-slot `(conv_state, ssm_state)` layout) — that is the documented
/// alias, nothing more.
///
/// ## Provenance flag
///
/// To preserve the original `"MambaCache"` class label across a
/// swift→Rust→swift round-trip, [`ArraysCache::mamba`] sets a constructor-
/// time `is_mamba` flag and [`reference_class_name`](KvCache::reference_class_name)
/// then returns `"MambaCache"` when the flag is set (`"ArraysCache"`
/// otherwise). The flag is plain bookkeeping (no state shape change, no Mamba
/// architecture) — exactly the swift-faithful `cacheClassName` switch
/// (`KVCache.swift:1381-1390`) where the `MambaCache` arm precedes the
/// `ArraysCache` arm. The [`super::from_state`] `"MambaCache"` arm
/// reconstructs with `is_mamba = true` so a swift-saved `"MambaCache"`
/// prompt cache round-trips as `"MambaCache"` — not `"ArraysCache"`.
///
/// # State serialization (slot identity preserved)
///
/// mlx-lm's `state` getter returns the **full** slot list including `None`
/// holes (`cache.py:624-626`) and its setter assigns it back verbatim
/// (`cache.py:628-630`), so a *sparse* cache (e.g. only slot 1 written)
/// round-trips with slot identity intact. The as-merged [`KvCache::state`]
/// is `Vec<Array>`, which structurally cannot carry a `None` slot — so a
/// naive compaction would silently re-pack slot 1 into slot 0 on restore
/// (wrong recurrent state).
///
/// This port therefore follows **mlx-swift-lm's `ArraysCache`** exactly
/// (`KVCache.swift:1112-1212`), which faces the identical "compacted array
/// list" constraint and solves it with slot-aware *metadata*:
///
/// - [`state`](KvCache::state) yields the present (non-`None`) slots in slot
///   order — swift `innerState()` / `state` getter `compactMap`
///   (`KVCache.swift:1112-1123`);
/// - [`meta_state`](KvCache::meta_state) carries `[slotCount,
///   presentSlotsCSV, leftPaddingCSV?]` — swift `metaState` getter
///   (`KVCache.swift:1173-1183`);
/// - [`set_state`](KvCache::set_state) + [`set_meta_state`](
///   KvCache::set_meta_state) together restore each compacted array into its
///   *original* slot index (and rebuild `left_padding`) — swift
///   `restoreFromMetaState(state:savedMetaState:)`
///   (`KVCache.swift:1192-1212`). In [`super::from_state`] `set_state` runs
///   first (stores the compacted arrays) and `set_meta_state` finalizes the
///   slot-aware placement, mirroring swift's combined restore and the
///   established [`RotatingKvCache`](super::RotatingKvCache) `set_state` →
///   `set_meta_state` ordering. The legacy/empty meta (`[]` / `[""]`,
///   `_BaseCache` default) keeps the compacted state as-is — swift's legacy
///   branch (`KVCache.swift:1208-1211`).
///
/// A fully-populated cache (the common SSM case) compacts to the full list,
/// so its round-trip is exact with or without the metadata.
///
/// No implicit eval: every op is a pure [`crate::ops`] / [`Array`]
/// composition returning a `Result`.
pub struct ArraysCache {
  /// `cache.py:602` `self.cache = [None] * size`. A `None` slot is a slot
  /// not yet written.
  cache: Vec<Option<Array>>,
  /// `cache.py:597/604` `left_padding` — per-sequence left-pad counts `[B]`,
  /// or `None`. Used (first) by [`make_mask`](KvCache::make_mask) /
  /// [`batch_size`](ArraysCache::batch_size) /
  /// [`meta_state`](KvCache::meta_state).
  ///
  /// Held as the integer values (Python stores `mx.array(left_padding)`,
  /// swift reads them back via `leftPaddingValues`,
  /// `KVCache.swift:1223-1226`): the broadcast `[B,1]` tensor is rebuilt
  /// on demand in `make_mask` — exactly Python's per-call
  /// `self.left_padding[:, None]` — so there is no array/values dual-state
  /// to keep in sync, `advance` is pure integer subtraction (Python's
  /// `self.left_padding -= N`), and `meta_state`'s CSV needs no eval (it
  /// must, the trait getter being `&self` + infallible).
  left_padding: Option<Vec<i32>>,
  /// `cache.py:598/679` `lengths` — per-sequence valid lengths `[B]` set via
  /// [`prepare`](ArraysCache::prepare), or `None`. Same integer-values
  /// rationale as `left_padding` (Python `self.lengths = mx.array(lengths)`
  /// / `self.lengths -= N`). Transient (never serialized — mlx-lm does not
  /// put it in `state`/`meta_state`).
  lengths: Option<Vec<i32>>,
  /// Provenance flag: `true` iff this `ArraysCache` was constructed
  /// via [`ArraysCache::mamba`] (i.e. it represents mlx-swift-lm's
  /// `MambaCache: ArraysCache`, `KVCache.swift:1229`). When set,
  /// [`reference_class_name`](KvCache::reference_class_name) returns
  /// `"MambaCache"` instead of `"ArraysCache"` so the swift→Rust→swift
  /// round-trip preserves the original class label. The flag is pure
  /// bookkeeping — no state shape change, no Mamba architecture port (the
  /// no-per-model-arch-porting rule is preserved).
  is_mamba: bool,
}

impl ArraysCache {
  /// A cache of `size` empty slots — mlx-lm `ArraysCache(size)`
  /// (`cache.py:601-602`), swift `ArraysCache(size:)`.
  pub fn new(size: usize) -> Self {
    Self {
      cache: (0..size).map(|_| None).collect(),
      left_padding: None,
      lengths: None,
      is_mamba: false,
    }
  }

  /// A 2-slot `ArraysCache` flagged with the `MambaCache` provenance
  /// — mlx-swift-lm's `class MambaCache: ArraysCache` whose only
  /// specialization is `init { super.init(size: 2) }`
  /// (`KVCache.swift:1230-1245`). State shape is identical to
  /// [`ArraysCache::new(2)`](ArraysCache::new); the only observable
  /// difference is [`reference_class_name`](KvCache::reference_class_name)
  /// returns `"MambaCache"`, so a save-after-load preserves the swift class
  /// label across the Rust round-trip.
  ///
  /// Construct this when porting a Mamba-style model (or restoring a
  /// `"MambaCache"`-kind prompt cache via [`super::from_state`]); use
  /// [`ArraysCache::new`] for the generic SSM slot cache.
  pub fn mamba() -> Self {
    Self {
      cache: vec![None, None],
      left_padding: None,
      lengths: None,
      is_mamba: true,
    }
  }

  /// `size` empty slots plus an initial `left_padding` `[B]` — mlx-lm
  /// `ArraysCache(size, left_padding)` (`cache.py:601-604`:
  /// `if left_padding: self.left_padding = mx.array(left_padding)`). An
  /// empty `left_padding` slice keeps it `None` (Python's falsy `[]`).
  pub fn with_left_padding(size: usize, left_padding: &[i32]) -> Self {
    let mut c = Self::new(size);
    if !left_padding.is_empty() {
      c.left_padding = Some(left_padding.to_vec());
    }
    c
  }

  /// `true` iff this `ArraysCache` carries the `MambaCache` provenance
  /// — see [`ArraysCache::mamba`].
  pub fn is_mamba(&self) -> bool {
    self.is_mamba
  }

  /// Reconstruct from a serialized state list + metadata — the
  /// `load_prompt_cache` path (`cache.py:79-82`). Mirrors mlx-swift-lm's
  /// `ArraysCache` restore (`KVCache.swift:1192-1212`
  /// `restoreFromMetaState(state:savedMetaState:)`): [`set_state`](
  /// KvCache::set_state) stores the compacted arrays, then
  /// [`set_meta_state`](KvCache::set_meta_state) places them back into their
  /// original slot indices and rebuilds `left_padding` (the slot-aware
  /// metaState format). `left_padding`/`lengths` start `None`
  /// (`cache.py:595-599`); `left_padding` is then restored from the
  /// metadata, `lengths` stays `None` (it is transient — set by
  /// [`prepare`](ArraysCache::prepare), never serialized; mlx-lm doesn't put
  /// it in `state`).
  ///
  /// Constructor-style helper (returns `Result<Self>`) shared with the
  /// trait-method override [`KvCache::from_serialized`] (which delegates
  /// here, then `*self = ..`-commits) and with the `from_state_arrays`
  /// dispatch entry (which wraps in `Box<dyn KvCache>`). Both fallible
  /// steps run on the freshly-`Self::new(0)`-built local, so the caller's
  /// own `self` (the trait-method override) is *never* mutated on an
  /// error path — the leaves-self-unchanged guarantee, end to end.
  ///
  /// `is_mamba` carries the provenance through the constructor —
  /// preserves [`ArraysCache::reference_class_name`] returning
  /// `"MambaCache"` across [`KvCache::from_serialized`] (so calling
  /// `from_serialized` on an `ArraysCache::mamba()` keeps it Mamba) and
  /// across [`super::from_state`]'s `"MambaCache"` arm.
  fn build_from_serialized(state: Vec<Array>, meta: &[String], is_mamba: bool) -> Result<Self> {
    let mut c = Self::new(0);
    c.set_state(state)?;
    c.set_meta_state(meta)?;
    c.is_mamba = is_mamba;
    Ok(c)
  }

  /// Borrow slot `idx` (`None` if unwritten **or** out of range) — mlx-lm
  /// `__getitem__` (`cache.py:621-622`). Python raises `IndexError` for an
  /// out-of-range index; this returns `None` instead of panicking
  /// (recoverable on every non-test path).
  pub fn get(&self, idx: usize) -> Option<&Array> {
    self.cache.get(idx).and_then(|s| s.as_ref())
  }

  /// Write slot `idx` — mlx-lm `__setitem__` (`cache.py:618-619`
  /// `self.cache[idx] = value`). An out-of-range `idx` is a recoverable
  /// [`Error::OutOfRange`] (Python's `IndexError`, surfaced as an indexing
  /// error rather than a tensor shape mismatch), never a panic.
  ///
  /// (Python's `__setitem__` also accepts `None`; SSM models only ever
  /// assign a real state array, and the no-arch-porting rule means no caller
  /// needs the `None` form, so this takes an [`Array`].)
  pub fn set(&mut self, idx: usize, value: Array) -> Result<()> {
    match self.cache.get_mut(idx) {
      Some(slot) => {
        *slot = Some(value);
        Ok(())
      }
      // Index-out-of-range is a range / indexing error, not a tensor shape
      // mismatch — use `Error::OutOfRange` so callers can distinguish "wrong
      // slot index" from "wrong tensor shape" via the variant alone.
      None => Err(Error::OutOfRange(OutOfRangePayload::new(
        "ArraysCache::set: slot index (must be < cache size)",
        "must be < cache size",
        format_smolstr!("idx={idx}, size={}", self.cache.len()),
      ))),
    }
  }

  /// The inferred batch size — mlx-lm `batch_size` (`cache.py:606-616`):
  /// the first non-`None` slot's leading axis, else `left_padding.size`,
  /// else `lengths.size`, else `1`.
  ///
  /// `c.shape[0]` on a rank-0 slot would be an `IndexError` in Python; here
  /// it is a recoverable error (never a raw `shape()[0]` panic on an
  /// un-validated tensor) — currently surfaced as [`Error::RankMismatch`]
  /// because the underlying condition IS a wrong tensor rank (a rank-0
  /// slot has no leading axis to read as batch size).
  pub fn batch_size(&self) -> Result<usize> {
    // mlx-lm `for c in self.cache: if c is not None: return c.shape[0]` —
    // the first present slot's leading axis.
    if let Some(slot) = self.cache.iter().flatten().next() {
      let shape = slot.shape();
      return match shape.first() {
        Some(&b) => Ok(b),
        None => Err(Error::RankMismatch(RankMismatchPayload::new(
          "ArraysCache::batch_size: slot must have rank >= 1 (a leading axis to read as batch size)",
          0,
          shape.to_vec(),
        ))),
      };
    }
    if let Some(lp) = &self.left_padding {
      return Ok(lp.len());
    }
    if let Some(l) = &self.lengths {
      return Ok(l.len());
    }
    Ok(1)
  }

  /// Set `lengths` from per-sequence valid lengths — mlx-lm `prepare`
  /// (`cache.py:678-679` `self.lengths = mx.array(lengths)`).
  pub fn prepare(&mut self, lengths: &[i32]) {
    self.lengths = Some(lengths.to_vec());
  }

  /// Clear `lengths` and `left_padding` — mlx-lm `finalize`
  /// (`cache.py:681-683`).
  pub fn finalize(&mut self) {
    self.lengths = None;
    self.left_padding = None;
  }

  /// Subtract `n` from `lengths` and `left_padding` (each only if set) —
  /// mlx-lm `advance` (`cache.py:685-689`: `self.lengths -= N` /
  /// `self.left_padding -= N`). Pure element-wise integer subtraction
  /// (wrapping like NumPy/MLX i32, so negatives are produced exactly as
  /// Python would).
  pub fn advance(&mut self, n: usize) -> Result<()> {
    // Integer-conversion / range error, not a tensor shape mismatch —
    // surface as `Error::ArithmeticOverflow` for the
    // same reason `set` and `update` switched: variants should reflect
    // the actual condition, not a misleading "shape" framing.
    let n = i32::try_from(n).map_err(|_| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "ArraysCache::advance: N exceeds i32::MAX",
        "i32",
        [("N", n as u64)],
      ))
    })?;
    if let Some(l) = &mut self.lengths {
      for v in l.iter_mut() {
        *v = v.wrapping_sub(n);
      }
    }
    if let Some(lp) = &mut self.left_padding {
      for v in lp.iter_mut() {
        *v = v.wrapping_sub(n);
      }
    }
    Ok(())
  }

  /// The `left_padding` values (`cache.py` `left_padding`), if set — swift
  /// `leftPaddingValues` (`KVCache.swift:1223-1226`).
  pub fn left_padding(&self) -> Option<&[i32]> {
    self.left_padding.as_deref()
  }

  /// The `lengths` values (`cache.py` `lengths`), if set.
  pub fn lengths(&self) -> Option<&[i32]> {
    self.lengths.as_deref()
  }
}

impl KvCache for ArraysCache {
  /// `0` — mlx-lm `ArraysCache` has no `offset` / `size()` override, so
  /// `_BaseCache.size()` (`cache.py:151-160`) returns `0`; swift's
  /// `BaseKVCache.offset` likewise stays `0` for this cache.
  fn offset(&self) -> usize {
    0
  }

  /// **Always an error** — mlx-lm `ArraysCache` has no `update_and_fetch`
  /// (it is a generic slot cache, not K/V; SSM models use `[]` / `state`).
  /// Recoverable, never a panic.
  ///
  /// Surfaced as [`Error::InvariantViolation`] (NOT `RankMismatch`/etc), so the variant
  /// reflects the actual condition — "this cache type doesn't support
  /// `update_and_fetch`" — rather than misleadingly suggesting the caller
  /// passed wrong-shaped tensors.
  fn update(&mut self, _keys: &Array, _values: &Array) -> Result<(Array, Array)> {
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "ArraysCache::update (generic slot cache, not K/V)",
      "must use get/set/state instead; update_and_fetch is unsupported",
    )))
  }

  /// Present (non-`None`) slots in slot order — swift `ArraysCache`
  /// `innerState()` / `state` getter `cache.compactMap` (`KVCache.swift:
  /// 1112-1123`). Slot identity is **not** lost: it is preserved out-of-band
  /// by [`meta_state`](KvCache::meta_state) (`presentSlots`), which
  /// [`set_meta_state`](KvCache::set_meta_state) uses to restore each array
  /// into its original slot. Each array is `try_clone`d (fallible per #33;
  /// never panicked).
  fn state(&self) -> Result<Vec<Array>> {
    self.cache.iter().flatten().map(|a| a.try_clone()).collect()
  }

  /// Force-evaluate the cache's own stored slot arrays in place — the
  /// per-chunk prefill memory barrier (see [`KvCache::materialize`]). Evals
  /// each present (`Some`) slot of `self.cache` directly via the explicit
  /// `&mut` [`Array::eval`] (`state()` already returns these live arrays
  /// un-sliced, but evaling the stored slots is the robust barrier). A no-op
  /// when every slot is empty.
  fn materialize(&mut self) -> Result<()> {
    for slot in self.cache.iter_mut().flatten() {
      slot.eval()?;
    }
    Ok(())
  }

  /// Replace the slots with `state`, compacted (`Some`) — swift
  /// `ArraysCache.state` setter `cache = newValue.map { $0 as MLXArray? }`
  /// (`KVCache.swift:1125-1127`). In [`super::from_state`] this runs
  /// *before* [`set_meta_state`](KvCache::set_meta_state), which (for the
  /// slot-aware metadata) then redistributes these compacted arrays back
  /// into their original slot indices — swift's combined `restoreFromMetaState`
  /// (`KVCache.swift:1192-1212`). An empty `state` is the "no slots" cache.
  fn set_state(&mut self, state: Vec<Array>) -> Result<()> {
    self.cache = state.into_iter().map(Some).collect();
    Ok(())
  }

  /// Slot-aware metadata `[slotCount, presentSlotsCSV, leftPaddingCSV?]` —
  /// swift `ArraysCache.metaState` getter (`KVCache.swift:1173-1183`). This
  /// is what makes a *sparse* cache round-trip (the present slots are
  /// serialized compactly by [`state`](KvCache::state); their original
  /// indices live here). `left_padding` is appended only when set (swift's
  /// optional third element).
  fn meta_state(&self) -> Vec<String> {
    let present: Vec<String> = self
      .cache
      .iter()
      .enumerate()
      .filter_map(|(i, s)| s.as_ref().map(|_| i.to_string()))
      .collect();
    let mut out = vec![self.cache.len().to_string(), present.join(",")];
    if let Some(lp) = &self.left_padding {
      out.push(lp.iter().map(i32::to_string).collect::<Vec<_>>().join(","));
    }
    out
  }

  /// Restore from the saved metadata — swift `ArraysCache`
  /// `restoreFromMetaState(state:savedMetaState:)` (`KVCache.swift:
  /// 1192-1212`). Two formats:
  ///
  /// - **slot-aware** (non-empty `m` where `m[0]` parses as `slotCount`):
  ///   requires `m.len() >= 2`, otherwise returns a recoverable error.
  ///   Rebuild `cache` as `slotCount` empty slots, then place the arrays
  ///   currently held by the preceding [`set_state`](KvCache::set_state)
  ///   (compacted, in slot order = the saved `presentSlots` order) back at
  ///   `presentSlots[i]`; restore `left_padding` from `m[2]` if present.
  ///   This is the exact inverse of [`meta_state`](KvCache::meta_state).
  /// - **legacy / empty** (`[]` / `[""]`, the `_BaseCache` default,
  ///   `KVCache.swift:158-165`): keep the compacted state as-is — swift's
  ///   legacy branch (`KVCache.swift:1208-1211`). Other malformed non-empty
  ///   metadata is not treated as legacy; it follows the slot-aware parse
  ///   path if `m[0]` parses, and otherwise returns a recoverable error.
  ///
  /// **Atomic on every error path.** All parsing *and* the (untrusted,
  /// possibly attacker-chosen) `slotCount` buffer allocation happen with
  /// `self` untouched; the receiver is then mutated by a single infallible
  /// block. So a non-numeric `slotCount`/slot index/left-padding is a
  /// recoverable [`Error::Parse`] / [`Error::CapExceeded`] / [`Error::ArithmeticOverflow`], and a hostile huge `slotCount` whose
  /// buffer cannot be allocated is a recoverable [`Error::OutOfMemory`]
  /// (via `try_reserve_exact`, **not** an aborting `vec![None; n]`) — in
  /// every failure case the cache is left exactly as the prior `set_state`
  /// produced it (never half-emptied), and never panics/aborts. An
  /// `arrayIdx`/`slotIdx` past the bounds is skipped — exactly swift's
  /// `where slotIdx < slotCount && arrayIdx < state.count` guard
  /// (`KVCache.swift:1203-1204`), so a truncated/oversized pairing degrades
  /// gracefully.
  fn set_meta_state(&mut self, m: &[String]) -> Result<()> {
    // Legacy / empty (`_BaseCache` default `[""]`, or `[]`): nothing to do;
    // the compacted `set_state` slots stand. (swift legacy branch.)
    if m.is_empty() || (m.len() == 1 && m[0].is_empty()) {
      return Ok(());
    }
    // Slot-aware: m[0]=slotCount, m[1]=presentSlots CSV, m[2]?=leftPadding.
    let slot_count: usize = m[0].parse().map_err(|e: std::num::ParseIntError| {
      Error::Parse(ParsePayload::new(
        "ArraysCache meta_state slotCount",
        "usize",
        Box::new(e),
      ))
    })?;
    if m.len() < 2 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "ArraysCache::set_meta_state: slot-aware meta_state",
        "must be [slotCount, presentSlots, leftPadding?] (length >= 2)",
      )));
    }
    // Atomicity is structural, not incidental: **every** fallible *and*
    // allocating step is staged here, with `self` still completely
    // untouched, so a malformed/hostile meta (bad parse, **or** an
    // attacker-chosen huge `slotCount` whose buffer can't be allocated)
    // returns `Err` with the cache exactly as the prior `set_state` left
    // it — the receiver is mutated by precisely one infallible block at the
    // end. Three failure modes for an untrusted `slot_count`:
    //
    // 0. **Hard cap exceeded** (`slot_count > MAX_SLOT_COUNT`) — surface as
    //    `Error::CapExceeded` ("slot_count exceeds MAX_SLOT_COUNT"). The
    //    **first** gate (#99) so a forged `INT32_MAX`-sized
    //    `slotCount` is rejected before either of the secondary gates
    //    (capacity-overflow / OOM) runs. Realistic SSM/Mamba caches have ≤
    //    64 slots (`MambaCache` is 2), so `MAX_SLOT_COUNT = 1 << 20` is far
    //    above any legitimate use while still well below `isize::MAX`.
    //
    //    This gate is placed BEFORE the `parse_csv` calls below so a forged
    //    `slotCount` rejects *before* either CSV's `Vec<T>` allocation —
    //    closing the "small slot_count + huge presentSlots/leftPadding
    //    payload" evasion. The CSVs themselves are then parsed with a
    //    `max_elems = slot_count` bound (see [`parse_csv`]) so a payload
    //    larger than slot_count is also rejected before any allocation,
    //    even though the cap above already protects the slot_count.
    // 1. **Capacity overflow** (`slot_count * size_of::<Option<Array>>()`
    //    exceeds `isize::MAX`, the Rust allocator's hard cap) — surface as
    //    `Error::ArithmeticOverflow` ("invalid slot_count: capacity overflow"). This
    //    is a logic-error / hostile-input distinction (the request is
    //    *intrinsically* invalid, not just larger than this machine).
    //    Pre-checked here because `TryReserveError::kind()` is nightly-only
    //    (#48043) — we can't distinguish OOM from capacity-overflow via
    //    the std `TryReserveError` accessors on stable, so we route the
    //    overflow case explicitly. Kept as a
    //    secondary gate after `MAX_SLOT_COUNT` so the hard cap can shrink
    //    without losing the allocator-level safety net.
    // 2. **Out-of-memory** (the allocator can't satisfy a valid request) —
    //    surface as `Error::OutOfMemory`. After the capacity-overflow
    //    pre-check, the residual `try_reserve_exact` failure is
    //    unambiguously OOM.
    if slot_count > MAX_SLOT_COUNT {
      return Err(Error::CapExceeded(CapExceededPayload::new(
        "ArraysCache::set_meta_state: slot_count (realistic SSM/Mamba caches use <= 64 slots; the cap fails fast on forged/corrupt meta_state)",
        "MAX_SLOT_COUNT",
        MAX_SLOT_COUNT as u64,
        slot_count as u64,
      )));
    }
    let elem_size = std::mem::size_of::<Option<Array>>().max(1);
    if slot_count > (isize::MAX as usize) / elem_size {
      return Err(Error::ArithmeticOverflow(
        ArithmeticOverflowPayload::with_operands(
          "ArraysCache::set_meta_state: slot_count * sizeof::<Option<Array>>() (capacity overflow exceeds isize::MAX)",
          "isize",
          [
            ("slot_count", slot_count as u64),
            ("elem_size", elem_size as u64),
          ],
        ),
      ));
    }
    // `presentSlots` is the producer's own `usize` slot index (emit at the
    // `meta_state` getter is `i.to_string()` where `i: usize`), so parse it
    // as `usize` to align producer/consumer types and drop the redundant
    // `>= 0` + `as usize` cast at the use site below. `leftPadding` stays
    // `i32` because mlx-lm/mlx-swift treat it as a signed offset and Swift
    // emits it as `Int`.
    //
    // Both CSVs are bounded by `slot_count`: the producer's `state_with_slots`
    // emits at most `slot_count` `presentSlots` indices (one per occupied
    // slot) and at most `slot_count` `leftPadding` entries (one per slot
    // entry, when populated), so `max_elems = slot_count` is the exact
    // legitimate upper bound — rejecting a forged small-slot-count +
    // huge-payload CSV before allocation.
    let present = parse_csv::<usize>(
      &m[1],
      "ArraysCache meta_state presentSlots",
      "CSV<usize>",
      slot_count,
    )?;
    let left_padding = match m.get(2) {
      Some(s) => Some(parse_csv::<i32>(
        s,
        "ArraysCache meta_state leftPadding",
        "CSV<i32>",
        slot_count,
      )?),
      None => None,
    };
    let mut rebuilt: Vec<Option<Array>> = Vec::new();
    rebuilt
      .try_reserve_exact(slot_count)
      .map_err(|_| Error::OutOfMemory)?;
    rebuilt.resize_with(slot_count, || None);
    // From here on: no `?`, no fallible/allocating call. Take the compacted
    // arrays the preceding `set_state` stored, in slot order (= the saved
    // `presentSlots` order, both ascending) — swift's `state` parameter to
    // `restoreFromMetaState`. Each is **moved** (not cloned) into its
    // original slot via `Option::take` (swift assigns existing arrays,
    // `KVCache.swift:1205` `self.cache[slotIdx] = state[arrayIdx]`, no deep
    // clone). This `mem::take` is the single point of mutation.
    let mut arrays: Vec<Option<Array>> = std::mem::take(&mut self.cache);
    for (array_idx, &slot_idx) in present.iter().enumerate() {
      // swift `where slotIdx < slotCount && arrayIdx < state.count` —
      // `slot_idx` is `usize` (see `parse_csv::<usize>` above), so the
      // `>= 0` half of swift's guard is vacuous in Rust.
      if slot_idx < slot_count
        && let Some(a) = arrays.get_mut(array_idx).and_then(Option::take)
      {
        rebuilt[slot_idx] = Some(a);
      }
    }
    self.cache = rebuilt;
    self.left_padding = left_padding;
    Ok(())
  }

  /// mlx-lm `ArraysCache.make_mask(self, N)` (`cache.py:691-699`) — note the
  /// reference signature takes **only** `N`: `window_size` / `return_array`
  /// are part of the uniform trait surface but, faithfully, this cache
  /// ignores them.
  ///
  /// - `left_padding` set → `mx.arange(N) >= left_padding[:, None]`
  ///   (`MaskMode::Array`, shape `[B, N]`);
  /// - else `lengths` set → `mx.arange(N) < lengths[:, None]`;
  /// - else → `None` (`MaskMode::None`).
  ///
  /// (mlx-swift-lm additionally gates on `cache[0] == nil` and only handles
  /// the `left_padding` branch — `KVCache.swift:1161-1167`; the authoritative
  /// mlx-lm form ported here has neither restriction.)
  fn make_mask(
    &self,
    n: usize,
    _window_size: Option<usize>,
    _return_array: bool,
  ) -> Result<MaskMode> {
    // `mx.array(values)[:, None]` -> `[B, 1]` (the values are the source of
    // truth; building `[B,1]` directly equals `mx.array(v)[:, None]` on the
    // `[B]` vector — Python rebuilds this per call too).
    let col = |v: &[i32]| -> Result<Array> { Array::from_slice::<i32>(v, &(v.len(), 1usize)) };
    // `mx.arange(N)` -> the exact integer `[0, N)` (I32) via the shared
    // guarded [`mask::iarange`]: this crate's `Array::arange` is f32-only,
    // so an `N > 2^24` exclusive stop would round and *silently* return a
    // wrong-length range — `iarange` rejects that with a recoverable
    // [`Error::OutOfRange`] instead of a corrupt mask (exactly the guard
    // the sibling `create_causal_mask` uses; mlx-lm's `mx.arange(N)` /
    // swift's `MLXArray(0..<N)` are integer-exact). Evaluated **lazily**,
    // only inside the branch that uses it — faithful to cache.py:691-699,
    // where `mx.arange(N)` is computed *within* the `left_padding` /
    // `lengths` branches and the `else` returns `None` *without* it (so a
    // huge `N` with no mask is `None`, not an error).
    if let Some(lp) = &self.left_padding {
      // `pos >= left_padding[:, None]` -> `[B, N]`.
      let pos = mask::iarange(0, n)?;
      return Ok(MaskMode::Array(ops::comparison::greater_equal(
        &pos,
        &col(lp)?,
      )?));
    }
    if let Some(l) = &self.lengths {
      let pos = mask::iarange(0, n)?;
      return Ok(MaskMode::Array(ops::comparison::less(&pos, &col(l)?)?));
    }
    Ok(MaskMode::None)
  }

  /// mlx-lm `ArraysCache.nbytes` (`cache.py:726-728`):
  /// `sum(c.nbytes for c in cache if c is not None)` — present slots only;
  /// `left_padding` / `lengths` are **not** counted (faithful). Pure
  /// metadata, no eval.
  fn nbytes(&self) -> usize {
    self
      .cache
      .iter()
      .flatten()
      .map(|a| super::util::nbytes(a).unwrap_or(0))
      .sum()
  }

  /// mlx-lm `ArraysCache.empty()` (`cache.py:723-724` `self.cache[0] is
  /// None`). Python `IndexError`s on a 0-slot cache; here a 0-slot cache
  /// (unreachable from any ported SSM constructor — Mamba uses 2 slots) is
  /// reported empty, the only non-panicking total answer for this
  /// non-`Result` signature.
  fn is_empty(&self) -> bool {
    match self.cache.first() {
      Some(slot) => slot.is_none(),
      None => true,
    }
  }

  /// Deep, independent copy — mlx-lm `copy.deepcopy` / swift `copy()`
  /// (`KVCache.swift:1130-1139`: new `ArraysCache(size)`, copy state +
  /// `offset` + `leftPadding`; here `lengths` is carried too, matching
  /// mlx-lm's `deepcopy` of the whole instance). Each slot is `try_clone`d
  /// (fallible per #33; a clone failure is propagated as an [`Error`],
  /// **never** swallowed into a partially-populated cache nor panicked);
  /// `left_padding` / `lengths` are owned `Vec<i32>`, plain-cloned.
  fn copy(&self) -> Result<Box<dyn KvCache>> {
    let mut cache = Vec::with_capacity(self.cache.len());
    for slot in &self.cache {
      cache.push(match slot {
        Some(a) => Some(a.try_clone()?),
        None => None,
      });
    }
    Ok(Box::new(Self {
      cache,
      left_padding: self.left_padding.clone(),
      lengths: self.lengths.clone(),
      // Preserve `MambaCache` provenance across `copy()` so the
      // deep-copied cache reports the same `reference_class_name`.
      is_mamba: self.is_mamba,
    }))
  }

  /// `"ArraysCache"` (or `"MambaCache"` when the `is_mamba` flag is set)
  /// — mlx-lm's `type(ArraysCache).__name__` (`cache.py:56`) /
  /// mlx-swift-lm `cacheClassName` switch (`KVCache.swift:1381-1390`),
  /// where the `MambaCache: ArraysCache` arm precedes the `ArraysCache`
  /// arm.
  ///
  /// **`MambaCache` provenance.** Constructing via [`ArraysCache::mamba`] (or
  /// loading via [`super::from_state`]'s `"MambaCache"` arm) sets the
  /// `is_mamba` flag, so the swift→Rust→swift round-trip preserves the
  /// `"MambaCache"` class label rather than degrading to `"ArraysCache"`.
  /// State shape is identical for both kinds (the only specialization is
  /// `super.init(size: 2)`, `KVCache.swift:1230-1245`), so this is pure
  /// bookkeeping — no Mamba architecture is ported and the
  /// no-per-model-arch-porting rule is preserved.
  fn reference_class_name(&self) -> &'static str {
    if self.is_mamba {
      "MambaCache"
    } else {
      "ArraysCache"
    }
  }

  /// Per-layer fast-path downcast target (#110) — see the
  /// [`KvCache`]-trait doc's **Per-layer fast-path convention**.
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }

  /// Transactional override of [`KvCache::from_serialized`] — leaves `self`
  /// byte-identical to its pre-call state on every recoverable error
  /// (`set_state` is infallible aside from `try_clone` paths it doesn't
  /// reach; `set_meta_state`'s slot-aware parse can fail on a malformed
  /// CSV, a non-numeric `slotCount`, or a hostile huge `slotCount` whose
  /// buffer can't be allocated — `Error::Parse` / `Error::CapExceeded` / `Error::OutOfMemory`,
  /// never a panic). The full restore is built into a fresh local via
  /// `ArraysCache::build_from_serialized` — the same constructor-style
  /// path `super::from_state`'s `"ArraysCache"` arm uses — and `self` is
  /// committed by a single infallible move only after that whole local
  /// build succeeds. `set_meta_state` is already itself transactional, but
  /// this override extends
  /// the leaves-self-unchanged guarantee across the *combined* `set_state`
  /// and `set_meta_state` sequence — closing the (today narrow, since
  /// the `set_state` body cannot fail without a `try_clone` it doesn't
  /// issue) 2-step window the default trait impl would leave open if
  /// `set_state` ever grew a fallible step.
  ///
  /// Preserves the cache's current `is_mamba` provenance flag
  /// across `from_serialized` (the trait-method override on `self`), so
  /// calling `from_serialized` on an [`ArraysCache::mamba`]-constructed
  /// cache keeps it Mamba. The [`super::from_state`] `"MambaCache"` arm
  /// uses `from_state_arrays(..., is_mamba: true)` (a crate-private
  /// helper) instead.
  #[allow(clippy::wrong_self_convention)] // see KvCache::from_serialized
  fn from_serialized(&mut self, state: Vec<Array>, meta: &[String]) -> Result<()> {
    *self = ArraysCache::build_from_serialized(state, meta, self.is_mamba)?;
    Ok(())
  }
}

/// `from_state` arm for the `"ArraysCache"` / `"MambaCache"` reference
/// class names (`cache.py:56` / the `load_prompt_cache` path
/// `cache.py:79-82`). Kept out of [`super::from_state`]'s body so this file
/// owns the whole port; the module's `match` adds two arms delegating here
/// with `is_mamba` set accordingly (provenance preservation).
pub(super) fn from_state_arrays(
  state: Vec<Array>,
  meta: &[String],
  is_mamba: bool,
) -> Result<Box<dyn KvCache>> {
  Ok(Box::new(ArraysCache::build_from_serialized(
    state, meta, is_mamba,
  )?))
}
