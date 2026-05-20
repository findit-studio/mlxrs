//! [`QuantizedKvCacheImpl`] — the on-the-fly quantized KV cache.

use crate::{
  array::Array,
  error::{Error, Result},
  lm::cache::{
    KvCache, MaskMode, QTriple, QuantizedKvCache, mask,
    util::{concat_seq, nbytes, seq_len, slice_seq},
  },
  ops,
};

/// mlx's quantization scheme used by `QuantizedKVCache`. mlx-lm calls
/// `mx.quantize(keys, group_size=..., bits=...)` (`cache.py:277-278`) /
/// mlx-swift-lm `quantized(keys, groupSize:, bits:)` with no `mode`
/// argument, i.e. mlx's default **affine** scheme — the only scheme that
/// produces the per-group `biases` mlx-lm's triple unconditionally indexes
/// (`self.keys[i]` for `i in 0..3`, `cache.py:279-281`). [`crate::ops`]'s
/// `quantize`/`dequantize` take `mode` explicitly, so it is pinned here.
const QUANT_MODE: &str = "affine";

/// mlx-lm `QuantizedKVCache.step` (`cache.py:233`) / mlx-swift-lm
/// `QuantizedKVCache.step` (`KVCache.swift:756`): the over-allocation batch
/// the reference's step buffer grows by. Identically to
/// [`StandardKvCache`](super::StandardKvCache), this is a **pure allocation
/// optimization with no observable effect** on what the cache returns:
/// mlx-lm slices the result to `[..., :offset, :]` on every exit
/// (`update_and_fetch`'s `tree_map(lambda x: x[..., :self.offset, :], ...)`
/// at `cache.py:283`, the `state` getter at `cache.py:285-292`, and
/// `getQuantizedState` at `KVCache.swift:820-823`). `mlxrs::Array` is
/// functional (no in-place buffer slicing), so this port stores the quant
/// triples always **exactly `offset`-length** and reproduces the observable
/// semantics via sequence-axis `concatenate` — exactly the
/// [`StandardKvCache`](super::StandardKvCache) / mlx-lm `ConcatenateKVCache`
/// equivalence already established in this module. The constant is kept for
/// documentation/parity only; it never changes a returned value, so it is
/// intentionally not otherwise referenced.
const _QUANT_STEP: usize = 256;

/// A stored quantized key/value tensor as the `(weight, scales, biases)`
/// triple mlx-lm holds in `self.keys` / `self.values` (`cache.py:236-237`;
/// mlx-swift-lm `(MLXArray, MLXArray, MLXArray?)`, `KVCache.swift:746-747`).
/// `biases` is optional because mlx's mode dispatch is bias-dependent (see
/// [`crate::ops::quantized::quantize`]); the affine scheme this cache uses
/// produces `Some`, but the optionality is modelled faithfully so the
/// shape of [`QTriple`] is preserved end to end (no implicit assumption
/// that biases exist — every per-element op is `Option`-aware).
type StoredTriple = (Array, Array, Option<Array>);

/// On-the-fly quantized KV cache — memory-efficient attention cache that
/// stores the keys/values in `bits`-bit grouped (affine) quantized form.
///
/// Faithful 1:1 port of `mlx_lm.models.cache.QuantizedKVCache`
/// (`cache.py:232-322`), cross-checked against mlx-swift-lm's
/// `MLXLMCommon.QuantizedKVCache` (`KVCache.swift:744-1005`) and its
/// `QuantizedKVCacheProtocol` (`KVCache.swift:111-136`).
///
/// Each [`update_quantized`](QuantizedKvCache::update_quantized) quantizes
/// the new keys/values (`mx.quantize`, reusing the merged
/// [`crate::ops::quantized`] — *not* a reimplementation), appends the
/// resulting `(weight, scales, biases)` triples on the sequence axis, and
/// returns the full accumulated quantized triples — exactly mlx-lm's
/// `update_and_fetch` (`cache.py:242-283`) / mlx-swift-lm's `updateQuantized`
/// (`KVCache.swift:833-906`). mlx-lm's `step`-sized over-allocated buffer
/// (`init_quant`/`expand_quant`) is a pure allocation optimization with no
/// observable effect on the returned (always `[..., :offset, :]`-sliced)
/// triples; with `mlxrs::Array` being functional this port reproduces those
/// observable semantics directly via `concatenate`, exactly the
/// [`StandardKvCache`](super::StandardKvCache) / `ConcatenateKVCache`
/// equivalence (see `_QUANT_STEP`).
///
/// No implicit eval: every op is a pure [`crate::ops`] composition
/// returning a `Result`; nothing on a recoverable path panics/unwraps.
pub struct QuantizedKvCacheImpl {
  keys: Option<StoredTriple>,
  values: Option<StoredTriple>,
  /// The cached sequence length — mlx-lm `QuantizedKVCache.offset`
  /// (`cache.py:238`); the raw position the attention mask / RoPE use,
  /// here always exactly the stored triples' sequence length.
  offset: usize,
  /// mlx-lm `QuantizedKVCache.group_size` (`cache.py:239`).
  group_size: i32,
  /// mlx-lm `QuantizedKVCache.bits` (`cache.py:240`).
  bits: i32,
}

impl Default for QuantizedKvCacheImpl {
  /// mlx-lm `QuantizedKVCache.__init__(group_size=64, bits=8)`
  /// (`cache.py:235`) / mlx-swift-lm `QuantizedKVCache(groupSize: 64,
  /// bits: 8)` (`KVCache.swift:753`).
  fn default() -> Self {
    Self::new(64, 8)
  }
}

impl QuantizedKvCacheImpl {
  /// A new, empty quantized cache with the given `group_size` / `bits`
  /// (mlx-lm `QuantizedKVCache(group_size, bits)`, `cache.py:235`).
  pub fn new(group_size: i32, bits: i32) -> Self {
    Self {
      keys: None,
      values: None,
      offset: 0,
      group_size,
      bits,
    }
  }

  /// `tree_map(transform, triple)` over a `(w, scales, biases?)` triple —
  /// mlx-lm's `tree_map(lambda x: ..., self.keys)` (`cache.py:265-291`) /
  /// mlx-swift-lm `treeMap` (`KVCache.swift:773-782`). Applies `f` to each
  /// present array; a `None` `biases` stays `None` (faithful: biases is
  /// only ever present-or-absent, never fabricated).
  fn tree_map(
    t: &StoredTriple,
    mut f: impl FnMut(&Array) -> Result<Array>,
  ) -> Result<StoredTriple> {
    Ok((
      f(&t.0)?,
      f(&t.1)?,
      match &t.2 {
        Some(b) => Some(f(b)?),
        None => None,
      },
    ))
  }

  /// Slice a stored triple's sequence axis (`-2`) to `[0, offset)` —
  /// mlx-lm's `tree_map(lambda x: x[..., : self.offset, :], ...)`
  /// (`cache.py:283`/`291`) / mlx-swift-lm `treeMap({ $0[.ellipsis,
  /// ..<offset, 0...] }, ...)` (`KVCache.swift:902-903`/`923-924`). Every
  /// triple array (weight `[B, H, S, dim/el_per_int]`, scales/biases `[B,
  /// H, S, dim/group_size]`) is the 4-D `[B, n_kv_heads, S, *]` KV layout,
  /// so the rank-safe `slice_seq` (axis `-2`, `KV_NDIM == 4`) applies
  /// directly; a wrong rank is a recoverable [`Error::ShapeMismatch`] from
  /// `seq_len`, never a raw `.shape()[N]` panic.
  fn trim_triple(t: &StoredTriple, offset: usize) -> Result<StoredTriple> {
    Self::tree_map(t, |a| {
      // Validate the 4-D KV rank (no blind shape indexing) before slicing.
      let _ = seq_len("quantized triple", a)?;
      slice_seq(a, 0, offset)
    })
  }

  /// Per-component-seq-len `(min, max)` of a stored triple — the seq-len
  /// extrema across `(weight, scales, biases-if-Some)`. mlx-lm's `state`
  /// getter applies `[..., :offset, :]` to **each** component independently
  /// (`tree_map(lambda x: x[..., :offset, :], self.keys)`,
  /// `cache.py:285-291`) — under NumPy/Python slice clamping
  /// (`mlx/ops.cpp:685` `std::min(e, n)`) each component clamps to its own
  /// `min(offset, own_len)`. So after [`trim_triple`](Self::trim_triple) on
  /// a forged state whose components have ASYMMETRIC seq-lens *within* the
  /// triple (e.g. `w` len 5, `scales` len 3, `biases` len 5 — `trim_triple(_,
  /// 5)` returns `w=5, scales=3, biases=5`), reading the post-trim seq-len
  /// from `w` only would commit `offset=5` while `scales` is len 3 —
  /// violating the offset-length invariant on the *components within* a
  /// triple. The within-triple `min` collapses that axis the same way the
  /// across-K/V `min` does — together they make P2's offset-length
  /// representation observably identical to mlx-lm's per-component
  /// `[:offset]` for all inputs (consistent or forged), in **every**
  /// asymmetry direction (across-K/V *and* within-(w, scales, biases)). A
  /// faithfully saved consistent triple (every component already same
  /// seq-len) is unaffected — `min == max == common_len` so the re-trim
  /// predicate `max > new_offset` is false.
  ///
  /// Returning `(min, max)` lets the caller (a) reduce across both triples
  /// with the min (the only converge value the offset can settle on
  /// without leaving any component shorter than `offset`), AND (b) detect
  /// — via `max > new_offset` — that re-trimming is actually needed (true
  /// iff at least one component is longer than the final `new_offset`,
  /// which can hold *within* a single triple even when that triple's `min`
  /// already equals `new_offset`, e.g. the `(w=5, scales=3, biases=5)`
  /// case above: `min=3=new_offset` but `max=5 > new_offset` so `w` and
  /// `biases` still need to be sliced down). Honest state: every
  /// component's seq-len equals the common `new_offset`, so `max ==
  /// new_offset` and the re-trim is a no-op.
  ///
  /// Rank-safe: each `seq_len` call validates the 4-D KV rank; a
  /// rank-invalid component is a recoverable [`Error::ShapeMismatch`].
  fn triple_component_len_range(name: &str, t: &StoredTriple) -> Result<(usize, usize)> {
    let lw = seq_len(name, &t.0)?;
    let ls = seq_len(name, &t.1)?;
    let lb = match &t.2 {
      Some(b) => Some(seq_len(name, b)?),
      None => None,
    };
    let mut lo = lw.min(ls);
    let mut hi = lw.max(ls);
    if let Some(b) = lb {
      lo = lo.min(b);
      hi = hi.max(b);
    }
    Ok((lo, hi))
  }

  /// Convert an owned `StoredTriple` into the public [`QTriple`] (identity:
  /// the public alias and the internal storage are the same
  /// `(Array, Array, Option<Array>)` shape — mlx-swift-lm's positional
  /// `(MLXArray, MLXArray, MLXArray?)`).
  fn into_qtriple(t: StoredTriple) -> QTriple {
    (t.0, t.1, t.2)
  }

  /// Independently clone a stored triple via the fallible
  /// [`Array::try_clone`] (#33 removed the infallible `impl Clone for
  /// Array`); a clone failure is propagated, never swallowed into a
  /// half-populated triple (silent corruption) and never panicked.
  fn clone_triple(t: &StoredTriple) -> Result<StoredTriple> {
    Self::tree_map(t, |a| a.try_clone())
  }

  /// `concat_seq` two stored triples on the sequence axis (`-2`) — mlx-lm's
  /// per-element `self.keys[i][..., prev:offset, :] = new[i]`
  /// (`cache.py:279-281`) / mlx-swift-lm's per-component assignment
  /// (`KVCache.swift:886-896`) under this port's always-`offset`-length
  /// storage. The `biases` Option must match (both present or both absent —
  /// the affine mode this cache uses always yields `Some`, so that is the
  /// taken arm; a mixed pairing means a bias-less state was loaded then an
  /// affine triple produced — a recoverable inconsistency, never a panic).
  fn concat_triple(prev: &StoredTriple, new: &StoredTriple) -> Result<StoredTriple> {
    let w = concat_seq(&prev.0, &new.0)?;
    let s = concat_seq(&prev.1, &new.1)?;
    let b = match (&prev.2, &new.2) {
      (Some(pb), Some(nb)) => Some(concat_seq(pb, nb)?),
      (None, None) => None,
      _ => {
        return Err(Error::ShapeMismatch {
          message:
            "QuantizedKvCache: biases present in only one of the stored / new quantized triple"
              .into(),
        });
      }
    };
    Ok((w, s, b))
  }

  /// All of `update_quantized`'s **fallible** work with **no** `self`
  /// mutation: validate ranks, checked-add the offset, `mx.quantize` the
  /// new keys/values (the merged `crate::ops::quantized::quantize`, NOT a
  /// reimpl), and `concat_seq` onto the existing triples. Returns the new
  /// `(stored_keys, stored_values, new_offset)` to be committed by the
  /// caller in one infallible block.
  ///
  /// Factored out so BOTH the quantized fast path
  /// ([`update_quantized`](QuantizedKvCache::update_quantized)) and the
  /// base dequant path ([`update`](KvCache::update)) can perform every
  /// fallible step (including their respective `clone`/`dequantize`)
  /// *before* touching `self`, then commit atomically — so a recoverable
  /// failure anywhere leaves the cache exactly as it was (the same
  /// no-partial-mutation invariant, uniformly on both `KvCache` entry
  /// points). The computed values are identical to the in-place version, so
  /// the observable mlx-lm result is unchanged.
  fn compute_appended(
    &self,
    keys: &Array,
    values: &Array,
  ) -> Result<(StoredTriple, StoredTriple, usize)> {
    // `B, n_kv_heads, num_steps, k_head_dim = keys.shape` (cache.py:243).
    // We only need `num_steps` (the sequence-axis length); `seq_len`
    // validates the 4-D KV rank (no blind shape indexing) and yields
    // `keys.shape[-2]`.
    let num_steps = seq_len("keys", keys)?;
    // Validate the values tensor's rank too (mlx-lm reads
    // `values.shape[-1]`); a wrong rank is a recoverable error, not a
    // panic. (mlx-lm/mlx-swift-lm don't cross-validate K/V *compatibility*
    // — they assign and let mlx error downstream — and neither do we; this
    // only rejects a non-4-D tensor before `concat_seq` would.)
    let _ = seq_len("values", values)?;

    let prev = self.offset;
    // `self.offset += num_steps` (cache.py:275) — checked (Python ints
    // never overflow; a corrupt restored `offset` could wrap/panic here).
    let new_offset = prev
      .checked_add(num_steps)
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "QuantizedKvCache update: offset ({prev}) + num_steps ({num_steps}) overflows usize"
        ),
      })?;

    // `keys = mx.quantize(keys, group_size, bits)` (cache.py:277);
    // `values = mx.quantize(values, ...)` (cache.py:278). The merged
    // `crate::ops::quantized::quantize` (affine mode) — NOT a reimpl.
    let (kw, ks, kb) =
      ops::quantized::quantize(keys, self.group_size, self.bits, QUANT_MODE, None)?;
    let (vw, vs, vb) =
      ops::quantized::quantize(values, self.group_size, self.bits, QUANT_MODE, None)?;
    let new_k: StoredTriple = (kw, ks, kb);
    let new_v: StoredTriple = (vw, vs, vb);

    // `self.keys[i][..., prev:offset, :] = keys[i]` (cache.py:279-281):
    // with always-`offset`-length storage this is `concat_seq(prev, new)`
    // on the sequence axis (`-2`), and just `new` when the cache was empty.
    let stored_k = match &self.keys {
      Some(pk) => Self::concat_triple(pk, &new_k)?,
      None => new_k,
    };
    let stored_v = match &self.values {
      Some(pv) => Self::concat_triple(pv, &new_v)?,
      None => new_v,
    };
    Ok((stored_k, stored_v, new_offset))
  }

  /// Re-establish this port's storage invariant (the stored triples are
  /// **exactly `offset`-length**) by **converging `self.offset` and the
  /// stored sequence length to the smaller of the two**: slice each stored
  /// triple's sequence axis (`-2`) down to `self.offset` (the overlength
  /// direction), AND clamp `self.offset` down to the actual resulting
  /// stored seq-len (the underlength direction; `slice_seq` follows NumPy
  /// clamping at `mlx/ops.cpp:685` `std::min(e, n)`, so an over-long `end`
  /// silently caps at the array length — without this clamp `self.offset`
  /// would remain at the larger forged value while the storage stayed
  /// shorter, and the next `update_quantized` would `concat_seq` onto a
  /// too-short storage, surfacing a phantom-slot gap).
  ///
  /// Used by [`from_state`](super::from_state) after `set_state` +
  /// `set_meta_state`: those two stay individually 1:1 with mlx-lm
  /// (`cache.py:294-296` assigns the triples as-is; `cache.py:302-304`
  /// restores `offset`), but a *forged / inconsistent* serialized prompt
  /// cache whose restored triple seq-len ≠ restored `offset` would then
  /// violate this port's offset-length representation — the next
  /// [`update_quantized`](QuantizedKvCache::update_quantized) would
  /// `concat_seq` onto a stored triple of the wrong size and silently
  /// surface stale tokens beyond the logical `offset` (overlength) or
  /// leave a phantom gap between the storage end and `offset`
  /// (underlength).
  ///
  /// This is **not** new validation the reference lacks: mlx-lm's `state`
  /// getter itself returns the triples sliced to `[..., :offset, :]`
  /// (`cache.py:285-292`), which is `[:min(offset, buf_len)]` under
  /// NumPy/Python slice semantics — so converging to the smaller of
  /// `offset` and the actual stored seq-len here makes this port's
  /// offset-length representation observably **identical** to mlx-lm's for
  /// *all* inputs (including forged ones, in both directions) — it
  /// maintains the faithful-observable-equivalence the representation
  /// exists to provide, mirroring mlx-lm's `[:offset]`. It is **not** a
  /// reject (the user-decided behavior is slice, not `Err`): a faithfully
  /// saved consistent state (seq-len already `== offset`, or the full
  /// buffer when `offset == len`) is **unaffected** — both the slice and
  /// the offset clamp are no-ops for it. Rank-safe (`trim_triple` and the
  /// post-trim seq-len read both validate the 4-D KV rank via `seq_len`;
  /// a wrong rank is a recoverable [`Error::ShapeMismatch`], never a
  /// panic).
  pub(crate) fn enforce_offset_len_invariant(&mut self) -> Result<()> {
    let offset = self.offset;
    let new_keys = match &self.keys {
      Some(k) => Some(Self::trim_triple(k, offset)?),
      None => None,
    };
    let new_values = match &self.values {
      Some(v) => Some(Self::trim_triple(v, offset)?),
      None => None,
    };
    // Symmetric underlength clamp + asymmetric K/V converge + within-triple
    // converge: `slice_seq` follows NumPy's `std::min(end, n)`
    // (mlx/ops.cpp:685), so when restored `offset > stored_len` the trim
    // above returns the full shorter array (its seq-len `== stored_len`,
    // NOT `offset`). And a forged state can have ASYMMETRIC stored seq-lens
    // (a) BETWEEN keys and values (e.g. keys stored 3, values stored 5, meta
    // offset 5) — both trim to their own `min(offset, own_len)`, leaving
    // K-len 3 and V-len 5 — AND/OR (b) WITHIN a single triple's `(weight,
    // scales, biases)` components (since `trim_triple` applies `slice_seq`
    // per-component, each clamps to its own `min(offset, own_len)`
    // independently). Without converging across BOTH the K/V axis AND the
    // within-triple components axis, `self.offset` and the longer side(s)
    // would commit out of sync and the next `update_quantized` would
    // surface stale tokens past the logical `offset` (overlength asymmetry)
    // / phantom-slot gap (underlength) / mismatched-shape `concat_seq` raise
    // (within-triple).
    //
    // Three-step converge: (1) trim each to `offset` above (per-component
    // NumPy clamp); (2) read post-trim `(min, max)` of each triple across
    // ALL components; (3) `new_offset = min(k_min, v_min)` (the only value
    // that leaves no component shorter than `offset`); re-trim each triple
    // whose `max > new_offset` (a longer component exists, within-triple
    // OR across-K/V). Empty/None storage clamps `offset` to 0. Faithful to
    // mlx-lm's `state` getter `[:offset]` semantics on EACH COMPONENT
    // independently — this just ensures every component of both triples
    // agrees post-clamp, which a non-forged round-trip already does (every
    // component updates by the same number of tokens, so honest state has
    // every component's seq-len == offset; both the across-K/V converge
    // and the within-triple re-trim are no-ops for it).
    // Per-triple `(min_len, max_len)` across ALL components (NOT just
    // `seq_len(w)`): a forged state can have ASYMMETRIC seq-lens *within* a
    // triple's components (e.g. `w` len 5, `scales` len 3, `biases` len 5;
    // offset 5 — `trim_triple(_, 5)` returns `w=5, scales=3, biases=5`
    // because NumPy `slice_seq` clamps each component independently at
    // `mlx/ops.cpp:685`). Reading `w` alone would commit `offset=5` while
    // `scales` is len 3 — phantom-slot gap on `scales`/`biases` the next
    // `update_quantized` would land into. The within-triple `min` is the
    // analog of the across-K/V `min` one level down (same defect class on
    // a different axis); the within-triple `max > new_offset` test detects
    // that an across-K/V re-trim is needed *even when* this triple's `min`
    // already equals the final `new_offset` (the within-triple asymmetry
    // case). Faithful to mlx-lm's `state` getter applying `[:offset]`
    // *per-component* (`cache.py:285-291` `tree_map`-over-each-array). See
    // [`triple_component_len_range`](Self::triple_component_len_range).
    let kr = match new_keys.as_ref() {
      Some(k) => Some(Self::triple_component_len_range("quantized keys", k)?),
      None => None,
    };
    let vr = match new_values.as_ref() {
      Some(v) => Some(Self::triple_component_len_range("quantized values", v)?),
      None => None,
    };
    let kl = kr.map(|(lo, _)| lo);
    let vl = vr.map(|(lo, _)| lo);
    let new_offset = match (kl, vl) {
      (Some(k), Some(v)) => k.min(v),
      (Some(k), None) => k,
      (None, Some(v)) => v,
      (None, None) => 0,
    };
    // Re-trim if ANY component of a triple is longer than `new_offset`
    // (asymmetric case — within-triple OR across-K/V). The within-triple
    // case (`min == new_offset` but `max > new_offset`) still needs the
    // longer components sliced down: predicate is `max > new_offset`, NOT
    // `min > new_offset`. `trim_triple` re-applies the per-component
    // `slice_seq` to `new_offset`, which is a no-op for any component
    // already at-or-below `new_offset` and forces longer ones down to
    // `new_offset` exactly. Result: every component of every triple has
    // seq-len == `new_offset`. Honest equal-length state: `max == min ==
    // common_len == new_offset`, both trims are no-ops.
    let new_keys = match (new_keys, kr) {
      (Some(k), Some((_, hi))) if hi > new_offset => Some(Self::trim_triple(&k, new_offset)?),
      (k, _) => k,
    };
    let new_values = match (new_values, vr) {
      (Some(v), Some((_, hi))) if hi > new_offset => Some(Self::trim_triple(&v, new_offset)?),
      (v, _) => v,
    };
    self.keys = new_keys;
    self.values = new_values;
    self.offset = new_offset;
    Ok(())
  }

  /// Dequantize a stored triple via the merged
  /// [`crate::ops::quantized::dequantize`] (the #19 op — **not** a
  /// reimplementation), using this cache's `group_size` / `bits` / affine
  /// `QUANT_MODE`.
  fn dequant_triple(&self, t: &StoredTriple) -> Result<Array> {
    ops::quantized::dequantize(
      &t.0,
      &t.1,
      t.2.as_ref(),
      self.group_size,
      self.bits,
      QUANT_MODE,
      None,
      None,
    )
  }
}

impl KvCache for QuantizedKvCacheImpl {
  /// The cached sequence length — mlx-lm `QuantizedKVCache.offset`
  /// (`cache.py:238`).
  fn offset(&self) -> usize {
    self.offset
  }

  /// The base [`KvCache::update`] for a quantized cache.
  ///
  /// mlx-lm's `QuantizedKVCache` defines only `update_and_fetch`
  /// (returning quantized triples, `cache.py:242`), and mlx-swift-lm's
  /// `QuantizedKVCache.update` deliberately `fatalError`s
  /// (`KVCache.swift:910-914`) to force callers onto `updateQuantized`.
  /// This crate's conventions forbid panicking on a recoverable path, and
  /// the merged [`KvCache`] trait requires `update -> (Array, Array)` for
  /// uniform `Box<dyn KvCache>` use; so this returns the **dequantized**
  /// accumulated keys/values — `update_quantized` followed by mlx
  /// `dequantize` (the merged [`crate::ops::quantized::dequantize`], *not*
  /// a reimplementation), which is exactly mlx-swift-lm's documented
  /// non-quantized fallback contract (`QuantizedKVCacheProtocol` usage
  /// example, `KVCache.swift:101-109`) and the same dequant mlx-swift-lm's
  /// `toUnquantized()` performs (`KVCache.swift:982-1004`). The quantized
  /// fast path remains [`QuantizedKvCache::update_quantized`]; a caller
  /// that wants the quantized triples downcasts via
  /// [`as_quantized`](KvCache::as_quantized).
  ///
  /// This is the one deliberate, documented deviation from mlx-swift-lm's
  /// `fatalError` (replacing an unrecoverable panic with the faithful
  /// observable dequantized equivalent); behavior is otherwise 1:1 with
  /// `update_and_fetch` + `dequantize`.
  ///
  /// Transactional like [`update_quantized`](QuantizedKvCache::update_quantized):
  /// the append (`compute_appended`) **and** both `dequantize` calls run
  /// into locals while the cache is untouched; only after every fallible
  /// step succeeds is `self` committed in one infallible block. So a
  /// recoverable failure (a `quantize`/`concat`/`dequantize` backend or
  /// allocation error) returns `Err` with the cache **unchanged** — the
  /// base `KvCache` path never advances `offset` / stores half state on a
  /// failed call (no double-append on retry), the same no-partial-mutation
  /// invariant the quantized path upholds.
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    let (stored_k, stored_v, new_offset) = self.compute_appended(keys, values)?;
    // Dequantize from the freshly-computed (not-yet-stored) triples — every
    // fallible op is done BEFORE any `self` mutation.
    let dk = self.dequant_triple(&stored_k)?;
    let dv = self.dequant_triple(&stored_v)?;
    // Infallible commit (one block, all-or-nothing).
    self.offset = new_offset;
    self.keys = Some(stored_k);
    self.values = Some(stored_v);
    Ok((dk, dv))
  }

  /// mlx-lm `QuantizedKVCache.state` getter (`cache.py:285-292`) /
  /// mlx-swift-lm `QuantizedKVCache.state` getter
  /// (`KVCache.swift:917-934`): the offset-sliced flattened triple arrays
  /// (`[k.w, k.scales, k.biases?, v.w, v.scales, v.biases?]` — 6 arrays, or
  /// 4 when biases are absent); `[]` when empty (mlx-swift-lm returns `[]`,
  /// `KVCache.swift:919`).
  ///
  /// mlx-lm only slices when `self.offset != self.keys[0].shape[2]`
  /// (`cache.py:287`); this port's triples are always exactly `offset`
  /// length, so the slice is the identity here — the observable serialized
  /// state is byte-identical to mlx-lm's.
  fn state(&self) -> Result<Vec<Array>> {
    match (&self.keys, &self.values) {
      (Some(k), Some(v)) => {
        let tk = Self::trim_triple(k, self.offset)?;
        let tv = Self::trim_triple(v, self.offset)?;
        let mut out = Vec::with_capacity(6);
        out.push(tk.0);
        out.push(tk.1);
        if let Some(b) = tk.2 {
          out.push(b);
        }
        out.push(tv.0);
        out.push(tv.1);
        if let Some(b) = tv.2 {
          out.push(b);
        }
        Ok(out)
      }
      _ => Ok(Vec::new()),
    }
  }

  /// mlx-lm `QuantizedKVCache.state` setter (`cache.py:294-296`:
  /// `self.keys, self.values = v`) / mlx-swift-lm
  /// (`KVCache.swift:935-949`): 6 arrays → `(w, scales, Some(biases))`
  /// triples; 4 arrays → `(w, scales, None)` triples (bias-less). An empty
  /// state resets to the fresh cache (`_BaseCache` "no state").
  ///
  /// Faithful to mlx-lm/mlx-swift-lm, this does **not** cross-validate the
  /// keys/values triples' shapes — it assigns and lets mlx error
  /// downstream — and `offset` is **not** derived here (mlx-lm restores it
  /// via `meta_state`, `cache.py:303-304`; the getter slices to whatever
  /// `offset` `set_meta_state` sets). A length other than 0/4/6 is a
  /// recoverable [`Error::Backend`] (mlx-swift-lm `fatalError`s,
  /// `KVCache.swift:945`; this crate forbids a panic on the recoverable
  /// load path).
  fn set_state(&mut self, mut state: Vec<Array>) -> Result<()> {
    match state.len() {
      0 => {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        Ok(())
      }
      4 => {
        // nil-biases case (mlx-swift-lm `KVCache.swift:937-940`).
        let v_s = state.pop().unwrap();
        let v_w = state.pop().unwrap();
        let k_s = state.pop().unwrap();
        let k_w = state.pop().unwrap();
        self.keys = Some((k_w, k_s, None));
        self.values = Some((v_w, v_s, None));
        Ok(())
      }
      6 => {
        // with-biases case (mlx-swift-lm `KVCache.swift:941-943`).
        let v_b = state.pop().unwrap();
        let v_s = state.pop().unwrap();
        let v_w = state.pop().unwrap();
        let k_b = state.pop().unwrap();
        let k_s = state.pop().unwrap();
        let k_w = state.pop().unwrap();
        self.keys = Some((k_w, k_s, Some(k_b)));
        self.values = Some((v_w, v_s, Some(v_b)));
        Ok(())
      }
      n => Err(Error::Backend {
        message: format!("QuantizedKvCache state must have 0, 4, or 6 arrays, got {n}"),
      }),
    }
  }

  /// mlx-lm `QuantizedKVCache.meta_state` getter (`cache.py:298-300`):
  /// `tuple(map(str, (self.offset, self.group_size, self.bits)))` — three
  /// decimal strings. (mlx-swift-lm additionally prepends `step`,
  /// `KVCache.swift:953`; mlx-lm is the authoritative spec, so three.)
  fn meta_state(&self) -> Vec<String> {
    vec![
      self.offset.to_string(),
      self.group_size.to_string(),
      self.bits.to_string(),
    ]
  }

  /// `QuantizedKVCache.meta_state` setter — accepts BOTH mlx-lm's and
  /// mlx-swift-lm's serialized forms so a prompt cache saved by either
  /// runtime loads into this one (cross-runtime portability, project
  /// decision 2026-05-20):
  /// - mlx-lm `[offset, group_size, bits]` — 3 strings, `cache.py:302-304`:
  ///   `self.offset, self.group_size, self.bits = map(int, v)`.
  /// - mlx-swift-lm `[step, offset, groupSize, bits]` — 4 strings,
  ///   `MLXLMCommon/KVCache.swift` `QuantizedKVCache.metaState` setter
  ///   (~line 952). `step` (the over-allocation tuning param at index 0)
  ///   is a pure allocation optimization with no observable effect on the
  ///   cache's contract (see `_QUANT_STEP`); the swift setter itself
  ///   restores ONLY `offset` from index `[1]` and ignores `step`,
  ///   `groupSize`, `bits`. This port restores `offset` (`[1]`),
  ///   `group_size` (`[2]`), and `bits` (`[3]`) from the same indices —
  ///   `group_size`/`bits` are restored (not ignored) so a cache restored
  ///   purely via `set_meta_state` after a fresh `new(_, _)` agrees on the
  ///   serialized values, but with `from_state` (the project entry point)
  ///   they match the placeholder `new(0, 0)` overwrite path identically.
  ///
  /// All three fields are parsed into locals before any `self` field is
  /// mutated, so a parse error on a later value leaves the cache
  /// untouched rather than partially corrupted (the same
  /// no-partial-mutation invariant the 3-string form already upheld).
  /// "Parsed" here means *successfully `usize`/`i32` parsed* — there is
  /// **no range / semantic validation** (no group_size > 0 check, no
  /// bits ∈ {2, 4, 8} check) per the NOTE in the body, since neither
  /// reference impl validates here either (Copilot review #3272690923).
  fn set_meta_state(&mut self, m: &[String]) -> Result<()> {
    // NOTE (Codex review needs-attention): No range-validation of
    // offset/group_size/bits is performed here. Neither mlx-lm
    // `cache.py:302-304` (`map(int, v)`) nor mlx-swift-lm
    // `KVCache.swift:952-961` (only restores `offset`, ignores
    // groupSize/bits entirely on restore) range-validates these fields.
    // Tightening beyond reference posture diverges from the contract per
    // [[feedback_match_official_binding_design]]. Downstream `mx.quantize`
    // calls error on invalid group_size/bits at the actual op (mlx-c's
    // contract).
    //
    // Sibling `RotatingKvCache::set_meta_state` (`rotating.rs:477-509`) has
    // the same posture — `offset` is parsed as an unbounded `usize` with no
    // range check. The pre-existing 3-string path here (since #40) parses
    // `m[0].parse::<usize>()` identically, flowing into the same
    // `enforce_offset_len_invariant` → `trim_triple` → `slice_seq` pipeline
    // (`util.rs:156-164`), whose `usize as i32` cast is the cross-cutting
    // boundary, not a Swift-form-specific defect. The 4-string path uses
    // the SAME parser/storage as the 3-string path — same exposure, faithful
    // to BOTH refs (which parse offset as an unbounded Python int / Swift
    // Int identically).
    //
    // mlx-lm 3-string form (`cache.py:302-304`): indices [offset,
    // group_size, bits]. mlx-swift-lm 4-string form
    // (`KVCache.swift:952-961`): indices [step, offset, groupSize, bits];
    // the leading `step` is dropped on restore (swift drops it too — it is
    // a pure over-allocation tuning param, not part of the observable
    // cache contract; see `_QUANT_STEP`).
    let (offset_idx, group_size_idx, bits_idx) = match m.len() {
      3 => (0, 1, 2),
      4 => (1, 2, 3),
      n => {
        return Err(Error::Backend {
          message: format!(
            "QuantizedKvCache meta_state must have 3 (mlx-lm form) or 4 \
             (mlx-swift-lm form) values, got {n}"
          ),
        });
      }
    };
    let offset = m[offset_idx].parse::<usize>().map_err(|e| Error::Backend {
      message: format!(
        "QuantizedKvCache meta_state offset ({:?}): {e}",
        m[offset_idx]
      ),
    })?;
    let group_size = m[group_size_idx]
      .parse::<i32>()
      .map_err(|e| Error::Backend {
        message: format!(
          "QuantizedKvCache meta_state group_size ({:?}): {e}",
          m[group_size_idx]
        ),
      })?;
    let bits = m[bits_idx].parse::<i32>().map_err(|e| Error::Backend {
      message: format!("QuantizedKvCache meta_state bits ({:?}): {e}", m[bits_idx]),
    })?;
    // Infallible commit tail — all fallible parsing done above.
    self.offset = offset;
    self.group_size = group_size;
    self.bits = bits;
    Ok(())
  }

  /// mlx-lm `QuantizedKVCache.is_trimmable` (`cache.py:306-307`): always
  /// trimmable.
  fn is_trimmable(&self) -> bool {
    true
  }

  /// mlx-lm `QuantizedKVCache.trim` (`cache.py:309-312`): `n = min(offset,
  /// n); offset -= n; return n`. Returns the number actually trimmed.
  ///
  /// mlx-lm only decrements `offset` because it keeps a `step`-over-
  /// allocated buffer and the next `update_and_fetch` *overwrites in place*
  /// at `prev = offset` (`self.keys[i][..., prev:offset, :] = ...`,
  /// `cache.py:280-281`) then returns `[..., :offset, :]` (`cache.py:283`)
  /// — so the trimmed-off tail is physically still there but provably
  /// overwritten/sliced-off before any observer. This port keeps the
  /// stored triples **exactly `offset`-length** (the documented
  /// `ConcatenateKVCache` / [`StandardKvCache`](super::StandardKvCache)
  /// equivalence; `mlxrs::Array` is functional, no in-place buffer slice),
  /// so to preserve that invariant — and the observable mlx-lm result —
  /// `trim` must also slice the stored triples to the new `offset`, exactly
  /// as [`StandardKvCache::trim`](super::StandardKvCache) does
  /// (`standard.rs`). Without this, a later
  /// [`update_quantized`](QuantizedKvCache::update_quantized) would
  /// `concat_seq` onto the *un-trimmed* triples (appending instead of
  /// overwriting at `prev`), so `quantized_state()` would surface the stale
  /// trimmed tokens instead of the new one — a faithful-semantics break in
  /// rollback/trim workflows. Slicing here keeps every per-method behavior
  /// 1:1 with the reference's *observable* result.
  fn trim(&mut self, n: usize) -> Result<usize> {
    let trimmed = n.min(self.offset);
    if trimmed == 0 {
      // mlx-lm `n = min(self.offset, n)`: a 0-token trim (incl. an empty
      // cache, `offset == 0`) is a no-op — nothing to slice, `offset`
      // unchanged.
      return Ok(0);
    }
    let new_offset = self.offset - trimmed;
    // Re-establish the "stored triples are exactly `offset`-length"
    // invariant by slicing the sequence axis (`-2`) to the new `offset`
    // (rank-safe via `trim_triple`; a wrong rank is a recoverable
    // `Error::ShapeMismatch`, never a panic).
    //
    // Transactional commit (same principle as `update_quantized`): compute
    // BOTH sliced triples into locals while the cache is untouched; only
    // once both fallible slices succeed do we mutate `self` in one
    // infallible block. A recoverable slice failure therefore leaves the
    // cache exactly as it was — never `offset` decremented with keys
    // sliced but values stale (silent corruption). `trim_triple` builds
    // fresh sliced arrays, leaving the originals intact until the move.
    let new_keys = match &self.keys {
      Some(k) => Some(Self::trim_triple(k, new_offset)?),
      None => None,
    };
    let new_values = match &self.values {
      Some(v) => Some(Self::trim_triple(v, new_offset)?),
      None => None,
    };
    self.offset = new_offset;
    self.keys = new_keys;
    self.values = new_values;
    Ok(trimmed)
  }

  /// mlx-lm `QuantizedKVCache.make_mask` (`cache.py:314-315`):
  /// `create_attention_mask(*args, offset=self.offset, **kwargs)` — the
  /// quantized cache forwards to the generic
  /// [`create_attention_mask`](mask::create_attention_mask) (verified
  /// against `cache.py:314-315`: it forwards, *unlike*
  /// `RotatingKVCache.make_mask`), passing the caller's `window_size`
  /// through unchanged.
  fn make_mask(
    &self,
    n: usize,
    window_size: Option<usize>,
    return_array: bool,
  ) -> Result<MaskMode> {
    mask::create_attention_mask(n, self.offset(), return_array, window_size)
  }

  /// mlx-lm `QuantizedKVCache.nbytes` (`cache.py:320-322`):
  /// `tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)`
  /// — the byte sum over **every** present triple array (weight, scales,
  /// and biases when present); 0 when empty. Pure metadata, no eval.
  fn nbytes(&self) -> usize {
    fn triple_bytes(t: &StoredTriple) -> usize {
      let mut total = nbytes(&t.0).unwrap_or(0) + nbytes(&t.1).unwrap_or(0);
      if let Some(b) = &t.2 {
        total += nbytes(b).unwrap_or(0);
      }
      total
    }
    let mut total = 0;
    if let Some(k) = &self.keys {
      total += triple_bytes(k);
    }
    if let Some(v) = &self.values {
      total += triple_bytes(v);
    }
    total
  }

  /// Whether the cache holds no keys yet — mlx-lm `QuantizedKVCache.empty`
  /// (`cache.py:317-318`: `return self.keys is None`).
  fn is_empty(&self) -> bool {
    self.keys.is_none()
  }

  /// An independent copy (mlx-lm `copy.deepcopy` / mlx-swift-lm `copy()`,
  /// `KVCache.swift:972-980`). Independence comes from MLX value semantics,
  /// not buffer duplication: the cache only ever *reassigns* the triples to
  /// freshly-computed arrays (never mutates a buffer in place), so although
  /// [`Array::try_clone`] is a refcount-sharing clone, the copy and the
  /// original evolve completely independently.
  ///
  /// mlx-swift-lm's `copy()` is infallible; here the fallible
  /// [`Array::try_clone`] is propagated as a `Result` — a clone failure is
  /// **never** mapped to a half-populated cache (silent corruption) and
  /// **never** panicked.
  fn copy(&self) -> Result<Box<dyn KvCache>> {
    Ok(Box::new(Self {
      keys: match &self.keys {
        Some(t) => Some(Self::clone_triple(t)?),
        None => None,
      },
      values: match &self.values {
        Some(t) => Some(Self::clone_triple(t)?),
        None => None,
      },
      offset: self.offset,
      group_size: self.group_size,
      bits: self.bits,
    }))
  }

  /// This cache *is* quantized — mlx-swift-lm `cache as?
  /// QuantizedKVCacheProtocol` (`KVCache.swift:101`). Returns `Some(self)`
  /// so the generation loop can take the quantized fast path.
  fn as_quantized(&self) -> Option<&dyn QuantizedKvCache> {
    Some(self)
  }

  /// This cache *is* quantized — the `&mut` companion of
  /// [`as_quantized`](KvCache::as_quantized). Returns `Some(self)` so a
  /// caller holding a `Box<dyn KvCache>` / `&mut dyn KvCache` can reach the
  /// quantized fast path's defining capability
  /// [`update_quantized`](QuantizedKvCache::update_quantized) (which takes
  /// `&mut self`), exactly mlx-swift-lm's mutable
  /// `QuantizedKVCacheProtocol` downcast (`KVCache.swift:101`).
  fn as_quantized_mut(&mut self) -> Option<&mut dyn QuantizedKvCache> {
    Some(self)
  }

  /// `"QuantizedKVCache"` — mlx-lm's `type(QuantizedKVCache).__name__`
  /// (`cache.py:56`) / mlx-swift-lm
  /// `case is QuantizedKVCache: return "QuantizedKVCache"`
  /// (`KVCache.swift:1387`).
  fn reference_class_name(&self) -> &'static str {
    "QuantizedKVCache"
  }
}

impl QuantizedKvCache for QuantizedKvCacheImpl {
  /// mlx-lm `QuantizedKVCache.group_size` (`cache.py:239`).
  fn group_size(&self) -> i32 {
    self.group_size
  }

  /// mlx-lm `QuantizedKVCache.bits` (`cache.py:240`).
  fn bits(&self) -> i32 {
    self.bits
  }

  /// Append `keys`/`values` and return the full accumulated **quantized**
  /// `((w, scales, biases?), (w, scales, biases?))` — a 1:1 port of mlx-lm
  /// `QuantizedKVCache.update_and_fetch` (`cache.py:242-283`) /
  /// mlx-swift-lm `QuantizedKVCache.updateQuantized`
  /// (`KVCache.swift:833-906`).
  ///
  /// Hand-trace of the reference (`cache.py:242-283`):
  /// - `B, n_kv_heads, num_steps, k_head_dim = keys.shape`;
  ///   `v_head_dim = values.shape[-1]`; `prev = self.offset`.
  /// - The `step`-over-allocated buffer (`el_per_int = 8 *
  ///   uint32.size // bits`; `init_quant` zero-fills `(*shape,
  ///   dim//el_per_int)` for the weight and `(*shape, dim//group_size)`
  ///   for scales/biases; `expand_quant` `concatenate`s a zero block on
  ///   `axis=-2`; the `prev % step != 0` re-trim to `[..., :prev, :]`) is
  ///   a **pure allocation optimization**: the return is *always*
  ///   `tree_map(lambda x: x[..., : self.offset, :], ...)`
  ///   (`cache.py:283`). `mlxrs::Array` is functional, so this port
  ///   instead keeps the stored triples exactly `offset`-length and
  ///   `concatenate`s the freshly-quantized new triples on the sequence
  ///   axis (`-2`) — observably identical to mlx-lm (the same
  ///   [`StandardKvCache`](super::StandardKvCache) / `ConcatenateKVCache`
  ///   equivalence; see `_QUANT_STEP`).
  /// - `self.offset += num_steps`.
  /// - `keys = mx.quantize(keys, group_size, bits)` /
  ///   `values = mx.quantize(values, ...)` then
  ///   `self.keys[i][..., prev:offset, :] = keys[i]` for each triple
  ///   element (`cache.py:277-281`) — i.e. the new quantized triple is
  ///   spliced over `[prev, offset)`. With always-`offset`-length storage
  ///   that is exactly: `new_triple = concat_seq(prev_triple,
  ///   quantize(new))` (and `= quantize(new)` when empty).
  /// - `return tree_map(lambda x: x[..., : self.offset, :], (self.keys,
  ///   self.values))` — the full accumulated triples (the slice is the
  ///   identity for `offset`-length storage).
  ///
  /// `mx.quantize` is the merged [`crate::ops::quantized::quantize`]
  /// (affine mode — the mlx default mlx-lm/mlx-swift-lm call with no
  /// `mode`; see `QUANT_MODE`) — **not** a reimplementation of
  /// quantization. Each triple array (weight / scales / biases) is the 4-D
  /// `[B, n_kv_heads, S, *]` KV layout, so the rank-safe `seq_len` /
  /// `concat_seq` (axis `-2`, `KV_NDIM == 4`) apply directly; a wrong
  /// rank is a recoverable [`Error::ShapeMismatch`], never a `.shape()[N]`
  /// panic. `offset` is bumped with [`usize::checked_add`] *before* any
  /// state mutation so a hostile/corrupt restored `offset` near
  /// `usize::MAX` is a recoverable error, not a wrap/panic, with no partial
  /// mutation (byte-identical to `offset + num_steps` for every
  /// non-overflowing input — the algorithm outcome is unchanged).
  fn update_quantized(&mut self, keys: &Array, values: &Array) -> Result<(QTriple, QTriple)> {
    // All fallible work (rank validation, checked offset, `mx.quantize`,
    // `concat_seq`) with NO `self` mutation — see `compute_appended`.
    let (stored_k, stored_v, new_offset) = self.compute_appended(keys, values)?;

    // Transactional commit: do EVERY remaining fallible op (the
    // `try_clone`-based `clone_triple` for the returned copies) into locals
    // FIRST, while the cache is still untouched. Only once both clones have
    // succeeded do we mutate `self` — and that final block is infallible
    // (three plain moves/assignments). So a recoverable failure anywhere
    // (#33 made `Array::try_clone` fallible; `quantize`/`concat` are
    // fallible too) returns `Err` with the cache **exactly as it was**:
    // never `offset` advanced with stale/half values, never keys replaced
    // while values lag (the silent-corruption hazard the sibling caches'
    // `copy` docs call out). The observable success result is unchanged
    // (same accumulated triples, same `offset`).
    let ret_k = Self::clone_triple(&stored_k)?;
    let ret_v = Self::clone_triple(&stored_v)?;
    self.offset = new_offset;
    self.keys = Some(stored_k);
    self.values = Some(stored_v);

    // `return tree_map(lambda x: x[..., : self.offset, :], (self.keys,
    // self.values))` (cache.py:283). Storage is already exactly `offset`
    // length, so the slice is the identity — return the accumulated
    // triples directly (the observable mlx-lm result).
    Ok((Self::into_qtriple(ret_k), Self::into_qtriple(ret_v)))
  }

  /// mlx-swift-lm `QuantizedKVCache.getQuantizedState`
  /// (`KVCache.swift:815-824`): the current quantized state without
  /// updating — `nil` if the cache is empty, else the triples sliced to
  /// `[..., :offset, :]`. (mlx-lm has no separate getter; its `state`
  /// property, `cache.py:285-292`, is the same offset-sliced triples — the
  /// `Option` here is mlx-swift-lm's `?` for the empty case.)
  fn quantized_state(&self) -> Result<Option<(QTriple, QTriple)>> {
    match (&self.keys, &self.values) {
      (Some(k), Some(v)) => {
        let tk = Self::trim_triple(k, self.offset)?;
        let tv = Self::trim_triple(v, self.offset)?;
        Ok(Some((Self::into_qtriple(tk), Self::into_qtriple(tv))))
      }
      _ => Ok(None),
    }
  }
}
