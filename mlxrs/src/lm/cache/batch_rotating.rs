//! [`BatchRotatingKvCache`] — the left-padded batched **sliding-window**
//! (ring) cache, ported 1:1 from
//! [`mlx_lm.models.cache`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
//! `BatchRotatingKVCache` (`cache.py:1133-1485`).
//!
//! mlx-swift-lm has **no** concrete `BatchRotatingKVCache` (only the
//! `BatchPositionedKVCache` protocol in `RoPEApplication.swift:13-22`), so
//! mlx-lm is the authoritative algorithm; the swift cross-check is just the
//! `batchOffset` → `.batch(...)` rope-offset contract (provided by the
//! merged [`KvCache::rope_offset`] default).
//!
//! This is the batched twin of [`RotatingKvCache`](super::RotatingKvCache)
//! and is **not** simplifiable: like the single-sequence rotating cache it
//! overwrites slots *in place* at a ring cursor, so the returned buffer is
//! in **physical ring order** (e.g. `max_size=4`, after ids `0..=4`
//! → `[4,1,2,3]`, *not* the temporal `[1,2,3,4]`), which its **own**
//! [`make_mask`](KvCache::make_mask) override (`cache.py:1330-1357`)
//! depends on. So it is a literal 1:1 port — `_idx`, `_offset`,
//! `_temporal_order`, `_trim`, the distinct `_update_in_place` (`S==1`) /
//! `_update_concat` (`S>1`) paths, the `rotated` flag, and the step buffer
//! (emulated with placeholder rows provably overwritten or sliced off
//! before any observer, exactly as in the single-sequence port).
//!
//! Critical differences from [`RotatingKvCache`](super::RotatingKvCache):
//! there is **no `keep`** prefix pin — `_trim` here is plain
//! `v[..., trim_size:, :]` (`cache.py:1152-1157`) — and `_temporal_order`
//! mutates `self` via a single full `mx.roll(keys, -_idx)` over all rows
//! (`cache.py:1159-1167`), not a pure per-call return. `offset` is a
//! per-sequence `[B]` array; `_offset` is the scalar monotone counter that
//! drives the grow/trim/rotate branches; `_idx` is the ring write cursor.
//!
//! No implicit eval: every op is a pure [`crate::ops`] composition.

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    ArithmeticOverflowPayload, DtypeMismatchPayload, Error, InvariantViolationPayload,
    LengthMismatchPayload, OutOfRangePayload, ParsePayload, RankMismatchPayload, Result,
    UnknownEnumValuePayload,
  },
  lm::cache::{
    BatchPositionedKvCache, KvCache, MaskMode,
    batch::{batch_head_dim, create_causal_mask_batched, dynamic_roll, ivec, validate_kv_compat},
    util::{concat_seq, nbytes, seq_len, seq_slice},
  },
  ops,
};
use smol_str::format_smolstr;

/// mlx-lm `BatchRotatingKVCache.step` — the in-place buffer growth batch
/// (`cache.py:1134`). Purely an allocation batch size: every grown row is
/// provably overwritten (or sliced off by the `_offset < max_size` return)
/// before the buffer is returned whole, so its value never reaches an
/// observer; mirrored as `256` so the buffer-length bookkeeping
/// (`keys.shape[2]`, which drives grow/trim/rotate and `_idx`) is
/// byte-for-byte mlx-lm's across every `S==1`/`S>1` mix.
const BATCH_ROTATING_STEP: usize = 256;

/// Port of `mx.roll(a, shift, axis=2)` for the 4-D `[B, n_kv_heads, S,
/// head_dim]` buffers `_temporal_order` rolls (`cache.py:1164-1165`).
/// `crate::ops` has no native `roll`; mlx defines `out[i] = a[(i - shift)
/// mod L]`. `_temporal_order` always rolls by `-self._idx`, i.e. elements
/// move toward *lower* indices by `idx`, which is exactly
/// `concat([a[idx:], a[:idx]])` along the sequence axis (`s = idx mod L`).
/// Built with the same slice/concatenate idioms the rest of the module
/// uses; no implicit eval.
fn roll_seq_neg(a: &Array, idx: usize) -> Result<Array> {
  let l = seq_len("keys/values", a)?;
  if l == 0 {
    return a.try_clone();
  }
  let s = idx % l;
  if s == 0 {
    return a.try_clone();
  }
  // mx.roll(a, -idx, axis=2): out = [ a[s:L] , a[0:s] ] on the seq axis.
  let head = seq_slice(a, s, l)?;
  let tail = seq_slice(a, 0, s)?;
  super::util::concat_parts(&[&head, &tail])
}

/// Emulate mlx-lm's in-place `buf[..., a:a+s, :] = new` on an immutable
/// `Array`: splice `new` over `[a, a+s)`, keeping the surrounding rows
/// (identical idiom to the single-sequence rotating port's `set_seq`).
fn set_seq(buf: &Array, a: usize, s: usize, new: &Array) -> Result<Array> {
  let l = seq_len("buffer", buf)?;
  let head = seq_slice(buf, 0, a)?;
  let tail = seq_slice(buf, a + s, l)?;
  super::util::concat_parts(&[&head, new, &tail])
}

/// Left-padded batched sliding-window KV cache — port of
/// `mlx_lm.models.cache.BatchRotatingKVCache` (`cache.py:1133-1485`).
pub struct BatchRotatingKvCache {
  keys: Option<Array>,
  values: Option<Array>,
  /// Per-sequence left-pad counts — mlx-lm `left_padding` (`[B]`, `I32`).
  left_padding: Array,
  /// Cached host-readable mirror of `left_padding`, kept in lockstep with
  /// the `Array` form (#101). See [`super::BatchKvCache::pad_lengths`]
  /// for the rationale — same per-call-`.item()` cost mlx-lm pays
  /// (cache.py:947-955 / KVCache.swift:1310-1325) is paid here exactly
  /// ONCE at construction / `set_state` time; consumers reach the host
  /// values via [`Self::pad_lengths`] as a borrowed slice.
  pad_lengths: Vec<i32>,
  /// Per-sequence raw position — mlx-lm `offset` (`[B]`, `I32`; starts at
  /// `-left_padding`, the per-seq RoPE offset / mask `left_padding`).
  offset: Array,
  /// Window length — mlx-lm `max_size`.
  max_size: usize,
  /// Physical ring write cursor — mlx-lm `_idx`. Wraps to `0` (no `keep`)
  /// once it reaches `max_size`.
  ///
  /// **Mutation discipline** (#102): `idx`, `off`, and `rotated`
  /// jointly form the ring's invariant `(rotated ⇔ off ≥ max_size && idx
  /// has wrapped)`. Every state-changing op MUST commit these three in
  /// the infallible commit tail of a stage-then-commit body — see
  /// [`Self::update_concat`] / [`Self::update_in_place`] for the canonical
  /// pattern. `rotated` SHOULD be the last mutation in the commit block
  /// so a half-committed cache (impossible in the current code, but a
  /// useful invariant to surface for future maintenance) is never reported
  /// as `rotated=true` against an old `idx`/`off`.
  idx: usize,
  /// Scalar monotone counter — mlx-lm `_offset` (drives grow/trim/rotate
  /// and is the trim/`is_trimmable` quantity; *not* capped at `max_size`).
  off: usize,
  /// Whether the ring has wrapped — mlx-lm `rotated` (the returned buffer
  /// is then in physical, not temporal, order).
  ///
  /// **Last-mutation rule** (#102): per the discipline above, this
  /// flag MUST be the **last** field assigned in any state-changing op's
  /// commit tail. Swift's reference `_update_in_place` (`KVCache.swift:
  /// 1330-1370`) mutates the buffer first and sets `rotated = false` late
  /// — a panic in the splice would leave `rotated = true` against a
  /// temporally-ordered buffer. The Rust port elevates this to a strict
  /// commit-tail-only assignment (the rest of the commit tail is
  /// infallible, so the flag is always coherent with the ring state).
  rotated: bool,
  /// Per-sequence true lengths for right-padded inputs, set by
  /// [`Self::prepare_right_padding`] — mlx-lm `_lengths` (so pad tokens do
  /// not evict valid tokens).
  lengths: Option<Array>,
}

impl BatchRotatingKvCache {
  /// A new empty left-padded batched rotating cache — mlx-lm
  /// `BatchRotatingKVCache(max_size, left_padding)` (`cache.py:1136-1150`):
  /// `offset = array([-l..])`, `_idx = _offset = 0`, `rotated = False`.
  ///
  /// The tiny `[B]` `mx.array(...)` builds are fallible only on
  /// allocation/backend; on the (unreachable) failure fall back to an
  /// empty array — still **no** panic / **no** heap leak (mirrors
  /// [`BatchKvCache::new`](super::BatchKvCache::new)).
  pub fn new(max_size: usize, left_padding: &[i32]) -> Self {
    let lp = ivec(left_padding).unwrap_or_else(|_| empty_ivec());
    let negated: Vec<i32> = left_padding.iter().map(|&l| -l).collect();
    let offset = ivec(&negated).unwrap_or_else(|_| empty_ivec());
    Self {
      keys: None,
      values: None,
      left_padding: lp,
      pad_lengths: left_padding.to_vec(),
      offset,
      max_size,
      idx: 0,
      off: 0,
      rotated: false,
      lengths: None,
    }
  }

  /// Per-sequence left-pad counts as a borrowed `&[i32]` — the cached
  /// host-readable mirror of [`left_padding_arr`](Self::left_padding_arr)
  /// (#101). Mirrors [`super::BatchKvCache::pad_lengths`] for
  /// cross-cache consistency; the value is updated in lockstep with the
  /// underlying `Array` form by [`set_state`](KvCache::set_state) /
  /// [`finalize`](Self::finalize) — never re-evaluated per call.
  pub fn pad_lengths(&self) -> &[i32] {
    &self.pad_lengths
  }

  /// The physical ring write cursor — mlx-lm `_idx`. Crate-internal so the
  /// sibling [`from_state`](super::from_state) can assert the
  /// post-reconstruction invariant `empty ⇒ _offset==0 && _idx==0 &&
  /// !rotated` (mirroring the merged single-seq
  /// [`RotatingKvCache::idx`](super::RotatingKvCache) precedent) without
  /// widening the public [`KvCache`] trait.
  pub(crate) fn ring_idx(&self) -> usize {
    self.idx
  }

  /// Whether the physical ring has wrapped — mlx-lm `rotated`.
  /// Crate-internal for the same `from_state` empty-state invariant.
  pub(crate) fn is_rotated(&self) -> bool {
    self.rotated
  }

  /// The window length — mlx-lm `max_size` (raw `usize`). Crate-internal
  /// so the sibling [`from_state`](super::from_state) can validate the
  /// restored-meta/buffer consistency invariant (the public
  /// [`max_size`](KvCache::max_size) returns `Option`).
  pub(crate) fn max_window(&self) -> usize {
    self.max_size
  }

  /// The restored buffer's sequence length (`keys.shape[-2]`), or `None`
  /// if empty. Crate-internal so [`from_state`](super::from_state) can
  /// reject a hostile prompt cache whose `meta_state`-injected
  /// `_idx`/`rotated` are inconsistent with the actually-restored buffer
  /// — the SINGLE structural chokepoint that closes the whole
  /// corrupt-restored-`_idx` class (rather than re-checking each
  /// downstream op). Returns the recoverable rank error rather than
  /// panicking on a non-4-D restored `keys`.
  pub(crate) fn buf_seq_len(&self) -> Result<Option<usize>> {
    match &self.keys {
      Some(k) => Ok(Some(seq_len("keys", k)?)),
      None => Ok(None),
    }
  }

  /// mlx-lm `BatchRotatingKVCache.prepare(lengths=..., right_padding=...)`
  /// (`cache.py:1282-1283`): when `max(right_padding) > 0`, store
  /// `_lengths = array(lengths) + offset`. Left-padding `prepare` is the
  /// constructor here.
  pub fn prepare_right_padding(&mut self, lengths: &[i32], right_padding: &[i32]) -> Result<()> {
    if right_padding.iter().copied().max().unwrap_or(0) > 0 {
      let l = ivec(lengths)?;
      self.lengths = Some(ops::arithmetic::add(&l, &self.offset)?);
    }
    Ok(())
  }

  /// mlx-lm `BatchRotatingKVCache.finalize` (`cache.py:1285-1292`): if
  /// `_lengths` is set, roll each sequence right by `max(0, offset -
  /// _lengths)`, fix `left_padding`/`offset`, clear `_lengths`.
  pub fn finalize(&mut self) -> Result<()> {
    if let Some(lengths) = &self.lengths {
      // Stage-then-commit: ALL fallible ops (`subtract`/`maximum`/
      // `expand_dims`/`dynamic_roll` ×2/`add`/`subtract`) compute into
      // locals FIRST; `self.*` is mutated only in the infallible tail. A
      // restored cache may (intentionally — `set_state` enforces only the
      // 4-D rank, not K/V batch compatibility, mirroring mlx-lm) have
      // batch-mismatched keys/values; the `values` roll would then fail
      // AFTER the keys roll. Computing into locals first means that `Err`
      // leaves keys/values/left_padding/offset/_lengths exactly as they
      // were (no keys-shifted-but-values-not desync, no retry double-roll)
      // — the Result-API no-partial-mutation contract, applied class-wide.
      let zero = ops::misc::astype(&Array::full::<f32>(&(1usize,), 0.0)?, Dtype::I32)?;
      let diff = ops::arithmetic::subtract(&self.offset, lengths)?;
      let roll = ops::arithmetic::maximum(&zero, &diff)?; // [B]
      let roll_col = ops::shape::expand_dims_axes(&roll, &[1])?; // [B,1]
      let rolled = match (&self.keys, &self.values) {
        (Some(k), Some(v)) => Some((
          dynamic_roll(k, &roll_col, 2)?,
          dynamic_roll(v, &roll_col, 2)?,
        )),
        _ => None,
      };
      let new_left_padding = ops::arithmetic::add(&self.left_padding, &roll)?;
      let new_offset = ops::arithmetic::subtract(&self.offset, &roll)?;
      // #101: refresh the cached host mirror by evaluating the
      // newly-computed `new_left_padding`. This is a finalize-time eval
      // (rare — once per request, not per token); the alternative
      // (host-side `max(0, off - lengths)` mirror) needs the `_lengths`
      // host values too, which would balloon the field set. Eval the
      // CLONED local (we already need `new_left_padding` for the commit).
      //
      // **Propagate extraction failures.** A naive implementation would
      // fall back to `self.pad_lengths.clone()` on a `to_vec`
      // error and then commit the new `left_padding` Array anyway,
      // leaving `pad_lengths()` permanently desynchronized from the
      // freshly-rolled Array state. Same class as
      // `BatchKvCache::set_state`/`BatchRotatingKvCache::set_state`:
      // propagate via `?` BEFORE the commit tail so an extraction/eval
      // failure leaves keys/values/offset/left_padding/_lengths fully
      // untouched.
      let mut lp_clone = new_left_padding.try_clone()?;
      let new_pad_lengths = lp_clone.to_vec::<i32>()?;
      // Infallible commit tail.
      if let Some((nk, nv)) = rolled {
        self.keys = Some(nk);
        self.values = Some(nv);
      }
      self.left_padding = new_left_padding;
      self.pad_lengths = new_pad_lengths;
      self.offset = new_offset;
      self.lengths = None;
    }
    Ok(())
  }

  /// mlx-lm `BatchRotatingKVCache.left_padding` — the per-sequence `[B]`
  /// left-pad counts (an owned clone; `Array::try_clone` is fallible per
  /// #33).
  pub fn left_padding_arr(&self) -> Result<Array> {
    self.left_padding.try_clone()
  }

  /// mlx-lm `_trim(trim_size, v, append)` (`cache.py:1152-1157`):
  /// `v[..., trim_size:, :]` then optional `concatenate([v, append], 2)`.
  /// **No `keep` prefix** (unlike the single-sequence rotating cache).
  fn trim_buf(&self, trim_size: usize, v: &Array, append: Option<&Array>) -> Result<Array> {
    let l = seq_len("buffer", v)?;
    let trimmed = if trim_size > 0 {
      seq_slice(v, trim_size, l)?
    } else {
      v.try_clone()?
    };
    match append {
      Some(a) => super::util::concat_parts(&[&trimmed, a]),
      None => Ok(trimmed),
    }
  }

  /// mlx-lm `_temporal_order` (`cache.py:1159-1167`): if `rotated`, roll
  /// `keys`/`values` by `-_idx` (axis 2) and report the reordered
  /// `(keys, values, new_idx)` (the post-roll `_idx = keys.shape[2]`,
  /// `rotated -> False`); `None` if not rotated (no-op).
  ///
  /// **Pure** (does not mutate `self`) so the sole caller
  /// ([`update_concat`](Self::update_concat)) can stage the entire update
  /// in locals and commit `self.*` atomically — an `Err` anywhere then
  /// leaves the cache fully unmutated (the Result-API no-partial-mutation
  /// contract). mlx-lm's `_temporal_order` mutates in place, but it is
  /// only ever invoked at the top of `_update_concat`, so a pure variant
  /// observed only through that one call site is behaviorally identical.
  fn temporal_order_parts(&self) -> Result<Option<(Array, Array, usize)>> {
    if self.rotated
      && let (Some(k), Some(v)) = (&self.keys, &self.values)
    {
      let nk = roll_seq_neg(k, self.idx)?;
      let nv = roll_seq_neg(v, self.idx)?;
      let new_idx = seq_len("keys", &nk)?;
      Ok(Some((nk, nv, new_idx)))
    } else {
      Ok(None)
    }
  }

  /// mlx-lm `_update_concat` — the `S > 1` path (`cache.py:1169-1206`).
  fn update_concat(&mut self, keys: &Array, values: &Array, s: usize) -> Result<(Array, Array)> {
    // mlx-lm `cache.py:1199-1200`: `self.offset += S; self._offset += S`.
    // Python ints never overflow; a corrupt/hostile prompt cache can
    // restore `_offset` near `usize::MAX` via `set_meta_state`, so this
    // bump would wrap (release) / panic (debug). Compute the post-update
    // `_offset` with `checked_add` BEFORE mutating ANY ring state (the
    // `temporal_order`/slice/roll/trim/`bump_offset` reassignments below)
    // so the overflow path performs NO partial mutation and the cache is
    // left exactly as it was (matching the merged single-seq
    // `RotatingKvCache` precedent). The value is byte-identical to
    // `self.off + s` for every non-overflowing input, so the ring
    // algorithm outcome is unchanged.
    //
    // CLASS-WIDE STAGE-THEN-COMMIT: the ENTIRE update is computed into
    // working locals (`wk`/`wv`/`w_idx`/`w_offset`/`w_left_padding`),
    // mirroring mlx-lm's `_update_concat` step-for-step; `self.*` is
    // mutated only in the infallible commit tail. So an `Err` from ANY
    // fallible op (overflow, `temporal_order`, slice, the `_lengths`
    // `dynamic_roll`, trim, `concat`) leaves the cache FULLY unmutated —
    // no keys-rolled-but-values-not / offset-advanced-but-buffer-not
    // desync, and a retry sees the original state. This kills the
    // partial-mutation-on-`Err` class structurally (not method-by-method)
    // for the `S>1` path.
    let new_off = self.off.checked_add(s).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "BatchRotatingKvCache::update: _offset + S",
        "usize",
        [("_offset", self.off as u64), ("S", s as u64)],
      ))
    })?;
    let (wk, wv): (Array, Array);
    let mut w_offset = self.offset.try_clone()?;
    let mut w_left_padding = self.left_padding.try_clone()?;
    // #101: track whether the host-side `pad_lengths` mirror
    // needs refreshing. The common decode case (empty cache OR `bk` fits
    // without trim/roll) leaves `w_left_padding` byte-identical to the
    // current `self.left_padding` (just a `try_clone`), so `pad_lengths`
    // is unchanged — no eval needed. Only when `lengths` roll or `trim`
    // mutates `w_left_padding` do we refresh.
    let mut lp_dirty = false;
    match (self.keys.as_ref(), self.values.as_ref()) {
      // mlx-lm: empty cache stores the inputs verbatim.
      (None, _) | (_, None) => {
        wk = keys.try_clone()?;
        wv = values.try_clone()?;
      }
      (Some(pk), Some(pv)) => {
        // Put the keys/values in temporal order to preserve context
        // (pure: returns the reordered parts + post-roll `_idx`).
        let (mut bk, mut bv, w_idx) = match self.temporal_order_parts()? {
          Some((nk, nv, ni)) => (nk, nv, ni),
          None => (pk.try_clone()?, pv.try_clone()?, self.idx),
        };

        // Slice off the end if needed (`shape[2] > _idx`).
        let cur = seq_len("keys", &bk)?;
        if cur > w_idx {
          bk = seq_slice(&bk, 0, w_idx)?;
          bv = seq_slice(&bv, 0, w_idx)?;
        }

        // Roll right sequences that are padded so we don't trim valid
        // entries: roll = max(0, offset - _lengths) (cache.py:1185-1190).
        if let Some(lengths) = &self.lengths {
          let zero = ops::misc::astype(&Array::full::<f32>(&(1usize,), 0.0)?, Dtype::I32)?;
          let diff = ops::arithmetic::subtract(&w_offset, lengths)?;
          let roll = ops::arithmetic::maximum(&zero, &diff)?;
          let roll_col = ops::shape::expand_dims_axes(&roll, &[1])?;
          bk = dynamic_roll(&bk, &roll_col, 2)?;
          bv = dynamic_roll(&bv, &roll_col, 2)?;
          w_left_padding = ops::arithmetic::add(&w_left_padding, &roll)?;
          w_offset = ops::arithmetic::subtract(&w_offset, &roll)?;
          lp_dirty = true;
        }

        // The largest size is max_size + S - 1 so every token gets at least
        // max_size context: trim_size = _idx - max_size + 1 (cache.py:1194).
        // `w_idx` derives from a restored `_idx` (`set_meta_state` parses a
        // `usize`, so a corrupt prompt cache can make it `usize::MAX`);
        // `w_idx + 1` would debug-panic / release-wrap (trim_size→0, a
        // wrong untrimmed buffer) — the SAME overflow class as `_offset +
        // S`. Compute it checked BEFORE any commit (we are still in the
        // staged region — `wk`/`wv` not yet assigned — so the `Err` leaves
        // the cache fully unmutated). For every non-overflowing input the
        // value is byte-identical to mlx-lm's unbounded `self._idx + 1`.
        let idx_plus_1 = w_idx.checked_add(1).ok_or_else(|| {
          Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
            "BatchRotatingKvCache::update_in_place: _idx + 1",
            "usize",
            [("_idx", w_idx as u64), ("one", 1u64)],
          ))
        })?;
        let trim_size = idx_plus_1.saturating_sub(self.max_size);
        if trim_size > 0 {
          // left_padding -= trim_size (cache.py:1196).
          let ts = ops::misc::astype(
            &Array::full::<f32>(&(1usize,), trim_size as f32)?,
            Dtype::I32,
          )?;
          w_left_padding = ops::arithmetic::subtract(&w_left_padding, &ts)?;
          lp_dirty = true;
        }
        // mlx-lm sets the final `self._idx = self.keys.shape[2]`
        // (cache.py:1201) — recomputed from `wk` in the commit tail; the
        // post-`temporal_order` `w_idx` was only the slice/trim pivot.
        wk = self.trim_buf(trim_size, &bk, Some(keys))?;
        wv = self.trim_buf(trim_size, &bv, Some(values))?;
      }
    }

    // offset += S (cache.py:1199); _idx = keys.shape[2] (cache.py:1201).
    let s_arr = ops::misc::astype(&Array::full::<f32>(&(1usize,), s as f32)?, Dtype::I32)?;
    let new_offset_arr = ops::arithmetic::add(&w_offset, &s_arr)?;
    let new_idx = seq_len("keys", &wk)?;
    let (rk, rv) = (wk.try_clone()?, wv.try_clone()?);
    // #101: If `lp_dirty`, eval the new left_padding ONCE here
    // (still in the staged region, before any commit). For the common
    // case (`lp_dirty == false` — no `lengths`, no trim), zero eval cost.
    //
    // **Propagate extraction failures.** A naive implementation would
    // fall back to `self.pad_lengths.clone()` on a `to_vec`
    // error and then commit the new `w_left_padding` Array anyway,
    // leaving `pad_lengths()` permanently desynchronized. Same class as
    // the finalize sibling above: propagate via `?` BEFORE the
    // commit tail so an extraction/eval failure leaves the cache fully
    // unmutated.
    let new_pad_lengths = if lp_dirty {
      let mut lp_clone = w_left_padding.try_clone()?;
      lp_clone.to_vec::<i32>()?
    } else {
      // Byte-identical to the current `self.pad_lengths` — `w_left_padding`
      // is just `self.left_padding.try_clone()?` along this path.
      self.pad_lengths.clone()
    };

    // ── Infallible commit tail (no `?` past this point) ──────────────
    self.keys = Some(wk);
    self.values = Some(wv);
    self.offset = new_offset_arr;
    self.left_padding = w_left_padding;
    self.pad_lengths = new_pad_lengths;
    self.off = new_off; // _offset += S (overflow already rejected above)
    self.idx = new_idx;
    // Clear `rotated`: `update_concat` ALWAYS produces a temporally-ordered
    // buffer — either `temporal_order_parts()` unwound a previously-rotated
    // ring (`mlx-lm` `cache.py:1183-1191`'s `self._rotated = False` after
    // the temporal_order roll), OR the buffer was already temporally
    // ordered (`rotated == false`) and stays so. Without this clear, a
    // mixed-path sequence (prefill → S==1 decodes that rotate → S>1 update
    // that temporal-orders) commits a temporally-ordered buffer while
    // `meta_state()` still reports `rotated == true`. Our own `from_state`
    // structural restore guard then rejects the cache's saved state
    // (`rotated && L != max_size` is impossible from mlx-lm's getter, so
    // the guard correctly screens it — but only because the producer side
    // must keep `rotated` in sync, which is precisely this fix).
    self.rotated = false;
    // mlx-lm's `mx.depends(...)` only forces eval ordering of the
    // `left_padding`/`offset` side metadata; `mlxrs::Array` is functional
    // with no implicit eval, so there is no in-place buffer to order
    // against — the data dependency is already explicit through the ops
    // above, making `depends` a no-op here (faithful: same values).
    Ok((rk, rv))
  }

  /// mlx-lm `_update_in_place` — the `S == 1` decode path
  /// (`cache.py:1208-1265`).
  fn update_in_place(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    if self.lengths.is_some() {
      // mlx-lm raises RuntimeError: finalize() must precede decoding.
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "BatchRotatingKvCache::update_in_place",
        "finalize() must be called before decoding",
      )));
    }
    // Rank validation is the public-`update` entry's responsibility
    // (`update` calls `validate_kv_compat(keys, values)?` before dispatching
    // to either `update_in_place` (S==1) or `update_concat` (S>1) — see
    // `update`'s body above). The previous in-method rank check here was
    // redundant with that centralized validation; remove to keep error
    // reporting consistent across both update paths. The `keys.shape()`
    // index reads below are now backed
    // by the public entry's pre-validation.
    let ks = keys.shape();
    let (b, h, s, k_hd) = (ks[0], ks[1], ks[2], ks[3]);
    let v_hd = batch_head_dim("values", values)?;
    let prev = self.off;

    // mlx-lm `cache.py:1252-1253`: `self._offset += S; self.offset += S`.
    // Python ints never overflow; a corrupt/hostile prompt cache can
    // restore `_offset` near `usize::MAX` via `set_meta_state`, so this
    // bump would wrap (release) / panic (debug). Compute the post-update
    // `_offset` with `checked_add` BEFORE mutating ANY ring state (the
    // grow/trim/rotate/`left_padding`/slot-overwrite reassignments below)
    // so the overflow path performs NO partial mutation and the cache is
    // left exactly as it was (matching the merged single-seq
    // `RotatingKvCache` precedent). The value is byte-identical to
    // `self.off + s` for every non-overflowing input, so the ring
    // algorithm outcome is unchanged.
    //
    // CLASS-WIDE STAGE-THEN-COMMIT (same contract as `update_concat`): the
    // whole decode step is computed into working locals mirroring mlx-lm's
    // `_update_in_place` step-for-step; `self.*` is mutated only in the
    // infallible commit tail. Any `Err` (overflow, grow `zeros`/`concat`,
    // trim, the `left_padding` ops, the in-place `set_seq` splice) leaves
    // the cache FULLY unmutated — no buffer-grown-but-offset-not /
    // keys-written-but-values-not desync, retry-safe.
    let new_off = self.off.checked_add(s).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "BatchRotatingKvCache::update: _offset + S",
        "usize",
        [("_offset", self.off as u64), ("S", s as u64)],
      ))
    })?;

    // Working copies of the mutable ring state (start = current `self`).
    let (mut bk, mut bv) = match (&self.keys, &self.values) {
      (Some(k), Some(v)) => (Some(k.try_clone()?), Some(v.try_clone()?)),
      _ => (None, None),
    };
    let mut w_idx = self.idx;
    let mut w_rotated = self.rotated;
    let mut w_left_padding = self.left_padding.try_clone()?;
    // #101: track whether `w_left_padding` diverges from
    // `self.left_padding`. The common decode case (no trim, no rotate)
    // leaves `pad_lengths` unchanged → zero eval cost.
    let mut lp_dirty = false;

    // Grow while below max_size: new_size = min(step, max_size - prev)
    // (cache.py:1218-1232).
    let cur_len = bk.as_ref().map_or(Ok(0), |k| seq_len("keys", k))?;
    if bk.is_none() || (prev >= cur_len && cur_len < self.max_size) {
      let new_size = BATCH_ROTATING_STEP.min(self.max_size.saturating_sub(prev));
      let zk = ops::misc::astype(
        &Array::zeros::<f32>(&(b, h, new_size, k_hd))?,
        keys.dtype()?,
      )?;
      let zv = ops::misc::astype(
        &Array::zeros::<f32>(&(b, h, new_size, v_hd))?,
        values.dtype()?,
      )?;
      let (nk, nv) = match (&bk, &bv) {
        (Some(pk), Some(pv)) => (concat_seq(pk, &zk)?, concat_seq(pv, &zv)?),
        _ => (zk, zv),
      };
      bk = Some(nk);
      bv = Some(nv);
      w_idx = prev;
    }

    // `bk`/`bv` are now `Some` (either grown above or pre-existing — the
    // grow branch's only false case requires `bk.is_some()`).
    let (kbuf, vbuf) = match (bk, bv) {
      (Some(k), Some(v)) => (k, v),
      _ => {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "BatchRotatingKvCache::update_in_place",
          "buffer is empty after grow (internal invariant violated)",
        )));
      }
    };
    let mut bk = kbuf;
    let mut bv = vbuf;

    // Trim if needed: trim_size = keys.shape[2] - max_size
    // (cache.py:1235-1240). `_trim` has no `keep` here.
    let cur_len = seq_len("keys", &bk)?;
    let trim_size = cur_len.saturating_sub(self.max_size);
    if trim_size > 0 {
      let tk = self.trim_buf(trim_size, &bk, None)?;
      let tv = self.trim_buf(trim_size, &bv, None)?;
      bk = tk;
      bv = tv;
      w_idx = self.max_size;
      let ts = ops::misc::astype(
        &Array::full::<f32>(&(1usize,), trim_size as f32)?,
        Dtype::I32,
      )?;
      w_left_padding = ops::arithmetic::subtract(&w_left_padding, &ts)?;
      lp_dirty = true;
    }

    // Rotate: if _idx == max_size -> rotated, _idx = 0; if rotated:
    // left_padding -= S (cache.py:1243-1247).
    if w_idx == self.max_size {
      w_rotated = true;
      w_idx = 0;
    }
    if w_rotated {
      let s_arr = ops::misc::astype(&Array::full::<f32>(&(1usize,), s as f32)?, Dtype::I32)?;
      w_left_padding = ops::arithmetic::subtract(&w_left_padding, &s_arr)?;
      lp_dirty = true;
    }

    // _idx += S (cache.py:1254). `w_idx` may still be a corrupt restored
    // `self.idx` (usize::MAX) if none of grow/trim/rotate reassigned it;
    // `w_idx + s` would debug-panic / release-wrap — same overflow class
    // as `_offset + S`. Compute it CHECKED here, BEFORE `set_seq`/the
    // commit tail, so the `Err` leaves the cache fully unmutated (and the
    // wrong out-of-range `set_seq` splice never even runs). Byte-identical
    // to mlx-lm's unbounded `self._idx += S` for every valid input.
    let new_w_idx = w_idx.checked_add(s).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "BatchRotatingKvCache::update_concat: _idx + S",
        "usize",
        [("_idx", w_idx as u64), ("S", s as u64)],
      ))
    })?;

    // Assign in place at [_idx, _idx+S) (cache.py:1250-1251); offset += S
    // (cache.py:1253).
    let nk = set_seq(&bk, w_idx, s, keys)?;
    let nv = set_seq(&bv, w_idx, s, values)?;
    let s_arr = ops::misc::astype(&Array::full::<f32>(&(1usize,), s as f32)?, Dtype::I32)?;
    let new_offset_arr = ops::arithmetic::add(&self.offset, &s_arr)?;

    // mlx-lm `cache.py:1259-1265`: if the buffer is not full slice off the
    // end. `_offset` after this update is `new_off`; compute the return
    // view from the staged `nk`/`nv` (no `self` read).
    let (rk, rv) = if new_off < self.max_size {
      (seq_slice(&nk, 0, new_off)?, seq_slice(&nv, 0, new_off)?)
    } else {
      (nk.try_clone()?, nv.try_clone()?)
    };
    // #101: refresh the host mirror only when `w_left_padding`
    // actually diverged from `self.left_padding` (trim and/or rotate
    // mutated it). Zero eval cost for the common decode steady-state
    // (no trim, no rotate).
    //
    // **Propagate extraction failures.** A naive implementation would
    // fall back to `self.pad_lengths.clone()` on a `to_vec`
    // error and then commit the new `w_left_padding` Array anyway,
    // leaving `pad_lengths()` permanently desynchronized. Same class as
    // the finalize/update_concat siblings above: propagate via
    // `?` BEFORE the commit tail so an extraction/eval failure leaves
    // `self` fully unmutated.
    let new_pad_lengths = if lp_dirty {
      let mut lp_clone = w_left_padding.try_clone()?;
      lp_clone.to_vec::<i32>()?
    } else {
      self.pad_lengths.clone()
    };

    // ── Infallible commit tail (no `?` past this point) ──────────────
    // #102: commit the ring state in the documented order — `idx`
    // and `off` first, `rotated` LAST. Swift's `_update_in_place` mutates
    // the buffer then sets `rotated = false` late (KVCache.swift:1330-
    // 1370), which would leave a half-committed `rotated=true` against a
    // temporally-ordered buffer if a panic interrupted. With every step
    // here infallible (post-`?`), the order is observably moot for a
    // well-formed run — but the discipline is codified so future
    // maintenance does not silently introduce a late-fallible step
    // between the buffer/idx commits and the flag.
    self.keys = Some(nk);
    self.values = Some(nv);
    self.offset = new_offset_arr;
    self.left_padding = w_left_padding;
    self.pad_lengths = new_pad_lengths;
    self.off = new_off; // _offset += S (overflow already rejected above)
    self.idx = new_w_idx;
    self.rotated = w_rotated; // LAST: enforces the commit-tail rule
    Ok((rk, rv))
  }
}

/// An empty `[0]`-length `I32` array — the unreachable
/// [`BatchRotatingKvCache::new`] allocation-failure fallback (panic-free,
/// no eval, no heap leak; mirrors [`super::batch`]'s `empty_ivec`).
fn empty_ivec() -> Array {
  Array::from_slice::<i32>(&[], &(0usize,)).unwrap_or_else(|_| {
    // Terminal, infallible, no-eval fresh empty handle (the `mlx_array_new`
    // out-param idiom). Reached only on the impossible double allocation
    // failure. SAFETY: `mlx_array_new()` returns a fresh owned empty handle
    // per the mlx-c convention; moved into the RAII `Array` newtype so it
    // is freed exactly once on drop.
    Array(unsafe { mlxrs_sys::mlx_array_new() })
  })
}

impl KvCache for BatchRotatingKvCache {
  /// mlx-lm `BatchRotatingKVCache.make_mask` builds its grid from
  /// `min(self.max_size - 1, self._offset)` (`cache.py:1335`); the scalar
  /// `offset()` (the mask/RoPE *scalar* position) is the monotone
  /// `_offset` (the per-sequence RoPE offset is the `[B]`
  /// [`BatchPositionedKvCache::batch_offset`] /
  /// [`rope_offset`](KvCache::rope_offset) `Batch`).
  fn offset(&self) -> usize {
    self.off
  }

  /// mlx-lm `BatchRotatingKVCache.max_size` — drives windowed masking.
  fn max_size(&self) -> Option<usize> {
    Some(self.max_size)
  }

  /// mlx-lm `BatchRotatingKVCache.update_and_fetch` (`cache.py:1267-1270`):
  /// dispatch on `S` — `_update_in_place` (`S==1`) / `_update_concat`
  /// (`S>1`). The two paths are **not** observably interchangeable (the
  /// in-place path returns the physical ring order its `make_mask`
  /// depends on).
  ///
  /// mlx-lm dispatches on `keys.shape[2]` (so a non-4-D `keys` is itself a
  /// Python error), `_update_in_place` raw-indexes `values.shape[3]`
  /// (`cache.py:1221`), and both paths end at
  /// `self.values[..., _idx:_idx+S, :] = values` / the V-buffer concat
  /// (built with **keys'** `B`/`n_kv_heads`) — which mlx-lm fails on a
  /// `B`/`n_kv_heads`/`S`-mismatched `values`. So `keys` *and* `values`
  /// are validated **here** (both 4-D; `values` `B`/`n_kv_heads`/`S` ==
  /// `keys`', head_dim free) before either path: a mismatch is a
  /// recoverable [`Error::RankMismatch`] / [`Error::ShapePairMismatch`] on
  /// **both** the `S==1` and `S>1` paths (never a panic, never a silent K/V
  /// desync), exactly mlx-lm's error semantics — observable behavior for
  /// every valid input is unchanged.
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    // Reject the constructor placeholder `max_size == 0` here too —
    // symmetric with `make_mask`. An update
    // against `max_size == 0` would otherwise drive `trim_size` /
    // `_idx - max_size + ...` arithmetic into degenerate / silently
    // wrong values; the from_state path never reaches here on the
    // placeholder (set_meta_state restores max_size before the first
    // update).
    if self.max_size == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "BatchRotatingKvCache::update",
        "max_size is 0 (from_state placeholder); call set_meta_state or construct with new(max_size > 0, left_padding) before calling update",
      )));
    }
    validate_kv_compat(keys, values)?;
    let s = seq_len("keys", keys)?;
    if s == 1 {
      self.update_in_place(keys, values)
    } else {
      self.update_concat(keys, values, s)
    }
  }

  /// mlx-lm `BatchRotatingKVCache.state` getter (`cache.py:1294-1299`):
  /// `[keys[:_offset], values[:_offset], offset, left_padding]` (sliced
  /// only when the buffer over-allocated); `[]` when empty.
  fn state(&self) -> Result<Vec<Array>> {
    match (&self.keys, &self.values) {
      (Some(k), Some(v)) => {
        let l = seq_len("keys", k)?;
        let (ks, vs) = if self.off < l {
          (seq_slice(k, 0, self.off)?, seq_slice(v, 0, self.off)?)
        } else {
          (k.try_clone()?, v.try_clone()?)
        };
        Ok(vec![
          ks,
          vs,
          self.offset.try_clone()?,
          self.left_padding.try_clone()?,
        ])
      }
      _ => Ok(Vec::new()),
    }
  }

  /// Force-evaluate the cache's own stored arrays in place — the per-chunk
  /// prefill memory barrier (see [`KvCache::materialize`]).
  ///
  /// Evals the genuine stored arrays via the explicit `&mut` [`Array::eval`]:
  /// the **full** `self.keys`/`self.values` ring buffers (the arrays the next
  /// `update` reads and extends — **not** the `seq_slice(k, 0, self.off)`
  /// views [`state`](KvCache::state) returns once the ring over-allocates),
  /// the per-sequence `self.offset`/`self.left_padding` position arrays (lazy
  /// `[B]` graphs that would otherwise chain across chunks), and the pending
  /// right-pad `self.lengths` when armed. Materializes every live buffer the
  /// next chunk reuses; `keys`/`values`/`lengths` are no-ops when absent.
  fn materialize(&mut self) -> Result<()> {
    if let Some(k) = self.keys.as_mut() {
      k.eval()?;
    }
    if let Some(v) = self.values.as_mut() {
      v.eval()?;
    }
    self.offset.eval()?;
    self.left_padding.eval()?;
    if let Some(lengths) = self.lengths.as_mut() {
      lengths.eval()?;
    }
    Ok(())
  }

  /// mlx-lm `BatchRotatingKVCache.state` setter (`cache.py:1301-1303`):
  /// `keys, values, offset, left_padding = v`. An empty state resets the
  /// buffer (`_offset`/`_idx`/`rotated`/`max_size` come from
  /// [`set_meta_state`](KvCache::set_meta_state)).
  fn set_state(&mut self, mut state: Vec<Array>) -> Result<()> {
    match state.len() {
      0 => {
        // Fully-fresh state, mirroring `BatchRotatingKvCache::new(self.max_size,
        // &self.left_padding_as_slice())`: an empty `set_state` MUST clear ALL
        // per-seq runtime state, not just `keys`/`values`. Otherwise `offset`
        // (per-seq `[B]` = current RoPE positions), `idx`/`off`/`rotated` (ring
        // cursor + write-monotone + wrapped flag), and `lengths` (pending
        // right-pad lengths) survive as stale metadata into a logically-fresh
        // cache — the next `update`/`make_mask`/`finalize` would treat absent
        // keys as prior context, mis-place the next write at a stale ring
        // cursor, or re-apply a dropped right-pad. `left_padding` and
        // `max_size` are preserved (constructor *inputs*). `offset =
        // -left_padding` is reproduced via a pure `ops::negative` (no eval, no
        // host extraction); the fallible op is staged BEFORE any `self.*`
        // mutation so a backend `Err` leaves the cache unmutated.
        let new_offset = ops::arithmetic::negative(&self.left_padding)?;
        // ── Infallible commit tail ──────────────────────────────────────
        self.keys = None;
        self.values = None;
        self.offset = new_offset;
        self.idx = 0;
        self.off = 0;
        self.rotated = false;
        self.lengths = None;
        Ok(())
      }
      4 => {
        let left_padding = state.pop().unwrap();
        let offset = state.pop().unwrap();
        let values = state.pop().unwrap();
        let keys = state.pop().unwrap();
        // mlx-lm's numpy `state` setter does no validation; here `keys`
        // *and* `values` are rank-validated (4-D `[B, n_kv_heads, S,
        // head_dim]`) BEFORE assignment: a hostile/corrupt prompt cache
        // could otherwise restore a rank-<3 buffer that a later
        // `state()`/`make_mask`/`offset` path raw-indexes on the seq axis
        // (`seq_slice`/`create_causal_mask`) and PANICs on the `Result`
        // API. We mirror mlx-lm's "no K/V *shape-compatibility*
        // validation" (head dim may differ; no B/H/S cross-check), only
        // enforcing the 4-D rank invariant so the failure is a recoverable
        // `Error::RankMismatch`, never a panic. Validate before assigning
        // any field so a bad buffer leaves the cache unmutated.
        seq_len("keys", &keys)?;
        batch_head_dim("values", &values)?;
        // #101: materialize the restored `left_padding` to the
        // host mirror ONCE here (same one-time eval cost as
        // `BatchKvCache::set_state` above). Staged on a local FIRST so an
        // eval failure leaves `self` fully unmutated.
        //
        // **Propagate extraction failures.** Same
        // class as `BatchKvCache::set_state`: a naive implementation would
        // fall back to `self.pad_lengths.clone()` on extraction failure and
        // then commit the new `left_padding` Array anyway, leaving
        // `pad_lengths()` permanently desynchronized from the actual
        // restored state (often at the empty placeholder from
        // `BatchRotatingKvCache::new(0, &[])`, which
        // `from_state("BatchRotatingKVCache")` opens with). So:
        // VALIDATE rank/dtype/length against the restored `keys` batch
        // dim BEFORE extraction, then propagate any `to_vec::<i32>`
        // error via `?` — cache stays untouched on every error path.
        let lp_shape = left_padding.shape();
        let kb = keys.shape()[0];
        if lp_shape.len() != 1 {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "BatchRotatingKvCache::set_state: restored left_padding must be 1-D [B]",
            lp_shape.len() as u32,
            lp_shape.to_vec(),
          )));
        }
        if lp_shape[0] != kb {
          return Err(Error::LengthMismatch(LengthMismatchPayload::new(
            "BatchRotatingKvCache::set_state: restored left_padding length vs keys batch dim",
            kb,
            lp_shape[0],
          )));
        }
        let lp_dtype = left_padding.dtype()?;
        if lp_dtype != Dtype::I32 {
          return Err(Error::DtypeMismatch(DtypeMismatchPayload::new(
            Dtype::I32,
            lp_dtype,
          )));
        }
        // `to_vec::<i32>` also enforces row-contiguity and re-checks
        // dtype, plus runs the single eval. Propagate every failure.
        let mut lp_clone = left_padding.try_clone()?;
        let new_pad_lengths = lp_clone.to_vec::<i32>()?;
        // ── Infallible commit tail ──────────────────────────────────────
        self.keys = Some(keys);
        self.values = Some(values);
        self.offset = offset;
        self.left_padding = left_padding;
        self.pad_lengths = new_pad_lengths;
        // Also clear `lengths` here, matching the empty-state branch above.
        // If `prepare_right_padding()` armed
        // `lengths` and a caller then `set_state`s with 4 arrays, the stale
        // `lengths` would (a) make `update_in_place` error
        // (`finalize() should be called…`) and/or (b) cause `finalize()`
        // to roll unexpectedly. mlx-lm doesn't have this problem because
        // its setter is part of `from_state`'s fresh-cache reconstruction
        // where `_lengths` is `None` by construction; mlxrs's setter is
        // callable out-of-band so we explicitly drop the stale field.
        // `_idx`/`_offset`/`rotated`/`max_size` come from
        // `set_meta_state` (separate setter), so they are NOT touched
        // here — `state` and `meta_state` setters stay individually 1:1
        // with mlx-lm's two-property contract (cache.py:1301-1315).
        self.lengths = None;
        Ok(())
      }
      n => Err(Error::OutOfRange(OutOfRangePayload::new(
        "BatchRotatingKvCache::set_state: state array count",
        "must be 0 or 4",
        format_smolstr!("{n}"),
      ))),
    }
  }

  /// mlx-lm `BatchRotatingKVCache.meta_state` (`cache.py:1305-1307`):
  /// `(max_size, _offset, _idx, rotated)` as strings (`rotated` is
  /// Python's `str(bool)` → `"True"`/`"False"`; we keep Rust's
  /// `"true"`/`"false"` and parse case-insensitively in
  /// [`set_meta_state`](KvCache::set_meta_state) so a round-trip and a
  /// Python-written cache both restore).
  fn meta_state(&self) -> Vec<String> {
    vec![
      self.max_size.to_string(),
      self.off.to_string(),
      self.idx.to_string(),
      self.rotated.to_string(),
    ]
  }

  /// mlx-lm `BatchRotatingKVCache.meta_state` setter
  /// (`cache.py:1309-1315`): `max_size, _offset, _idx = map(int, v[:3]);
  /// rotated = bool(v[3])`. Parsed atomically (all-or-nothing) so a
  /// malformed later field leaves the cache unmutated.
  fn set_meta_state(&mut self, m: &[String]) -> Result<()> {
    if m.len() != 4 {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "BatchRotatingKvCache::set_meta_state: meta_state values",
        4,
        m.len(),
      )));
    }
    let parse = |i: usize, name: &'static str| -> Result<usize> {
      m[i].parse::<usize>().map_err(|e: std::num::ParseIntError| {
        let context: &'static str = match name {
          "max_size" => "BatchRotatingKvCache::set_meta_state: max_size",
          "_offset" => "BatchRotatingKvCache::set_meta_state: _offset",
          "_idx" => "BatchRotatingKvCache::set_meta_state: _idx",
          _ => "BatchRotatingKvCache::set_meta_state",
        };
        Error::Parse(ParsePayload::new(context, "usize", Box::new(e)))
      })
    };
    let max_size = parse(0, "max_size")?;
    let off = parse(1, "_offset")?;
    let idx = parse(2, "_idx")?;
    // Python `bool(v[3])`: mlx-lm writes `str(bool)` → "True"/"False"; a
    // round-trip writes "true"/"false". Accept both; mlx-lm's `bool(...)`
    // of a non-empty string is `True`, but the only values ever serialized
    // are the two booleans, so a case-insensitive {true,false} parse is
    // faithful (an unexpected token is a recoverable error, not a panic).
    let rotated = match m[3].to_ascii_lowercase().as_str() {
      "true" => true,
      "false" => false,
      other => {
        return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
          "BatchRotatingKvCache::set_meta_state: rotated",
          other.to_string(),
          &["true", "false"],
        )));
      }
    };
    self.max_size = max_size;
    self.off = off;
    self.idx = idx;
    self.rotated = rotated;
    Ok(())
  }

  /// mlx-lm `BatchRotatingKVCache.is_trimmable` (`cache.py:1317-1318`):
  /// `_offset < max_size` (only before the ring fills).
  fn is_trimmable(&self) -> bool {
    self.off < self.max_size
  }

  /// mlx-lm `BatchRotatingKVCache.trim` (`cache.py:1320-1325`):
  /// `n = min(_offset, n); _offset -= n; _idx -= n; offset -= n; return
  /// n`.
  fn trim(&mut self, n: usize) -> Result<usize> {
    let trimmed = n.min(self.off);
    if trimmed == 0 {
      return Ok(0);
    }
    // Stage-then-commit (class-wide): the `offset -= n` array is computed
    // into a local FIRST; the scalar `_offset`/`_idx` decrements + the
    // `offset` array commit happen only in the infallible tail. A
    // `subtract` `Err` then leaves `_offset`/`_idx`/`offset` unchanged
    // (no scalar-decremented-but-array-not desync).
    let new_off = self.off - trimmed;
    let new_idx = self.idx.saturating_sub(trimmed);
    let ts = ops::misc::astype(&Array::full::<f32>(&(1usize,), trimmed as f32)?, Dtype::I32)?;
    let new_offset = ops::arithmetic::subtract(&self.offset, &ts)?;
    // ── Infallible commit tail ──────────────────────────────────────
    self.off = new_off;
    self.idx = new_idx;
    self.offset = new_offset;
    Ok(trimmed)
  }

  /// 1:1 port of mlx-lm `BatchRotatingKVCache.make_mask`
  /// (`cache.py:1330-1357`) — its **own** override, distinct from both the
  /// scalar `create_attention_mask` and
  /// [`BatchKvCache::make_mask`](super::BatchKvCache). Builds a windowed +
  /// per-sequence-left-padded mask, then (for a rotated `N==1` decode)
  /// `mx.roll`s it over the physical ring cursor:
  ///
  /// ```text
  /// window_size = window_size or self.max_size
  /// offset = min(self.max_size - 1, self._offset)
  /// rinds = arange(offset + N); linds = arange(offset, offset+N) if offset else rinds
  /// mask = (linds[:,None] >= rinds[None]) & (linds[:,None] < rinds[None] + window_size)
  /// trim_size = self._idx - self.max_size + int(N>1); if >0: left_padding -= trim_size
  /// rotated = N==1 and (self.rotated or self._idx >= self.max_size); if rotated: left_padding -= 1
  /// mask &= rinds >= expand_dims(left_padding, (1,2,3))
  /// if rotated: idx = self._idx; if idx>=max_size: idx=0; mask = roll(mask, idx+1, axis=-1)
  /// ```
  fn make_mask(
    &self,
    n: usize,
    window_size: Option<usize>,
    _return_array: bool,
  ) -> Result<MaskMode> {
    // `max_size == 0` is the constructor *placeholder* the `from_state`
    // dispatch passes before `set_meta_state` restores the real value
    // (`from_state` flow: `new(0, &[]) -> set_state -> set_meta_state` —
    // make_mask is never called on the placeholder by that path). A
    // `make_mask` against `max_size == 0` is therefore a USER-side misuse
    // (constructing `new(0, &[])` directly without restoring meta_state
    // afterwards) and would yield a degenerate / silently-wrong mask
    // (`ws == 0` and `offset == 0`). Reject as a recoverable error.
    if self.max_size == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "BatchRotatingKvCache::make_mask",
        "max_size is 0 (from_state placeholder); call set_meta_state or construct with new(max_size > 0, left_padding) before calling make_mask",
      )));
    }
    // window_size = window_size or self.max_size (Python truthiness: 0 is
    // falsy → falls back to max_size, like the single-seq rotating port).
    let ws = window_size.filter(|&w| w != 0).unwrap_or(self.max_size);
    // offset = min(self.max_size - 1, self._offset).
    let offset = self.max_size.saturating_sub(1).min(self.off);

    // mask = causal & windowed (no padding yet) via the shared batched
    // builder with right/left padding = None (left padding is folded in
    // *after* the trim/rotate adjustments below, exactly as mlx-lm). With
    // both paddings None this returns the rank-2 `[N, offset+N]` grid
    // (`linds[N,1] >= rinds[1,offset+N]`, optionally windowed) — exactly
    // mlx-lm's `mask = linds[:,None] >= rinds[None]` before the
    // `[B,1,1,1]` left-padding term broadcasts it to `[B,1,N,offset+N]`.
    let base = create_causal_mask_batched(n, offset, Some(ws), None, None)?;

    // trim_size = self._idx - self.max_size + int(N > 1) (cache.py:1342).
    // `self.idx` can be a corrupt restored `usize::MAX` (`set_meta_state`
    // parses a `usize`); `self.idx + int(N>1)` would debug-panic /
    // release-wrap (a wrong mask) — same overflow class as `_offset + S`.
    // Checked (this is `&self`, no mutation, so the `Err` is inherently
    // side-effect-free); byte-identical to mlx-lm's unbounded int for
    // every non-overflowing input.
    let idx_term = self.idx.checked_add(usize::from(n > 1)).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "BatchRotatingKvCache::make_mask: _idx + int(N>1)",
        "usize",
        [
          ("_idx", self.idx as u64),
          ("n_gt_1", usize::from(n > 1) as u64),
        ],
      ))
    })?;
    let trim_size = idx_term.saturating_sub(self.max_size);
    // rotated = N == 1 and (self.rotated or self._idx >= self.max_size).
    let rotated = n == 1 && (self.rotated || self.idx >= self.max_size);

    // left_padding - trim_size - (1 if rotated) (cache.py:1343/1347),
    // computed on a clone so the cache's own array is untouched.
    //
    // `trim_size` derives from a possibly-corrupt restored `_idx`; for
    // `N==1` the `checked_add(0)` above cannot catch it, so a hostile
    // `_idx==usize::MAX` makes `trim_size` ~`usize::MAX`. A raw
    // `trim_size as i64` / `delta as f32` would WRAP/lose precision
    // (e.g. `(usize::MAX-8) as i64 == -9`), flipping the `left_padding`
    // adjustment direction and returning a plausible-but-WRONG decode
    // mask instead of erroring (the same overflow class, at the
    // lossy cast rather than the add). Keep `delta` in `usize`
    // (`trim_size + int(rotated)`, checked), then require it to be
    // **exactly** representable through the `f32`-built I32 scalar the
    // MLX subtract uses (same `2^24` exact-int rationale as
    // `mask::scalar_i32`/`iarange`): for every real `_idx` `trim_size`
    // is tiny so this is byte-identical to mlx-lm's unbounded
    // `_idx - max_size + int(N>1)`; only the corrupt huge value is a
    // recoverable `ArithmeticOverflow` (this is `&self` — the `Err` is
    // inherently side-effect-free).
    let mut lp = self.left_padding.try_clone()?;
    let delta = trim_size.checked_add(usize::from(rotated)).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "BatchRotatingKvCache::make_mask: trim_size + int(rotated)",
        "usize",
        [
          ("trim_size", trim_size as u64),
          ("rotated", usize::from(rotated) as u64),
        ],
      ))
    })?;
    if delta != 0 {
      // `Array::full`/`scalar_i32` build the scalar through `f32`; an
      // `f32` represents every integer in `[0, 2^24]` exactly. A real
      // `delta` is far below this; a corrupt one beyond it would round
      // (silent wrong mask) — reject it instead.
      const F32_EXACT_INT_MAX: usize = 1usize << 24;
      if delta > F32_EXACT_INT_MAX {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "BatchRotatingKvCache::make_mask: trim/rotate delta (_idx restored too large for this path)",
          "must be <= 2^24 (f32 exact-integer limit)",
          format_smolstr!("{delta}"),
        )));
      }
      let d = ops::misc::astype(&Array::full::<f32>(&(1usize,), delta as f32)?, Dtype::I32)?;
      lp = ops::arithmetic::subtract(&lp, &d)?;
    }

    // mask &= rinds >= expand_dims(left_padding, (1,2,3)) (cache.py:1349).
    // rinds = arange(offset + N); rebuild it and broadcast lp to [B,1,1,1].
    let total = offset.checked_add(n).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "BatchRotatingKvCache::make_mask: offset + N",
        "usize",
        [("offset", offset as u64), ("N", n as u64)],
      ))
    })?;
    let rinds = super::mask::iarange(0, total)?;
    let rinds = ops::shape::expand_dims_axes(&rinds, &[0])?; // [1, total]
    let lp_b = ops::shape::expand_dims_axes(&lp, &[1, 2, 3])?; // [B,1,1,1]
    let pad_term = ops::comparison::greater_equal(&rinds, &lp_b)?; // rinds >= lp
    let mut mask = ops::logical::logical_and(&base, &pad_term)?;

    // if rotated: idx = self._idx; if idx >= max_size: idx = 0;
    // mask = mx.roll(mask, shift=idx + 1, axis=-1) (cache.py:1351-1355).
    if rotated {
      let idx = if self.idx >= self.max_size {
        0
      } else {
        self.idx
      };
      mask = roll_last_axis(&mask, idx + 1)?;
    }
    Ok(MaskMode::Array(mask))
  }

  /// mlx-lm `BatchRotatingKVCache.nbytes` (`cache.py:1480-1484`):
  /// `keys.nbytes + values.nbytes` (0 if empty).
  fn nbytes(&self) -> usize {
    let mut total = 0;
    if let Some(k) = &self.keys {
      total += nbytes(k).unwrap_or(0);
    }
    if let Some(v) = &self.values {
      total += nbytes(v).unwrap_or(0);
    }
    total
  }

  /// mlx-lm `BatchRotatingKVCache.empty` (`cache.py:1477-1478`):
  /// `keys is None`.
  fn is_empty(&self) -> bool {
    self.keys.is_none()
  }

  /// An independent copy (mlx-lm `copy.deepcopy`). MLX value semantics:
  /// arrays are immutable and the cache only ever *reassigns* its arrays
  /// (the in-place ring writes go through `set_seq`, which builds a fresh
  /// array — it never mutates a shared buffer), so a refcount-sharing
  /// `Array::try_clone` still yields a fully independent cache. The
  /// fallible clone is propagated as a `Result` — never swallowed (silent
  /// corruption) and never panicked.
  fn copy(&self) -> Result<Box<dyn KvCache>> {
    Ok(Box::new(Self {
      keys: match &self.keys {
        Some(a) => Some(a.try_clone()?),
        None => None,
      },
      values: match &self.values {
        Some(a) => Some(a.try_clone()?),
        None => None,
      },
      left_padding: self.left_padding.try_clone()?,
      pad_lengths: self.pad_lengths.clone(),
      offset: self.offset.try_clone()?,
      max_size: self.max_size,
      idx: self.idx,
      off: self.off,
      rotated: self.rotated,
      lengths: match &self.lengths {
        Some(a) => Some(a.try_clone()?),
        None => None,
      },
    }))
  }

  /// Batch-positioned (swift `cache as? BatchPositionedKVCache`); the
  /// merged [`rope_offset`](KvCache::rope_offset) default then yields
  /// `RopeOffset::Batch(batch_offset())`.
  fn as_batch_positioned(&self) -> Option<&dyn BatchPositionedKvCache> {
    Some(self)
  }

  /// `"BatchRotatingKVCache"` — mlx-lm's
  /// `type(BatchRotatingKVCache).__name__` (`cache.py:56`, written by
  /// `save_prompt_cache`; the load side accepts both this canonical name and
  /// the Rust alias `"BatchRotatingKvCache"`, see [`super::from_state`]).
  /// mlx-swift-lm has no `BatchRotatingKVCache` arm in its `cacheClassName`
  /// switch (`KVCache.swift:1381-1392`) — batch caches are mlx-lm-only — so
  /// the kind label is taken from mlx-lm verbatim.
  fn reference_class_name(&self) -> &'static str {
    "BatchRotatingKVCache"
  }

  /// Per-layer fast-path downcast target (#110) — see the
  /// [`KvCache`]-trait doc's **Per-layer fast-path convention**.
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }

  /// Transactional override of [`KvCache::from_serialized`] — leaves `self`
  /// byte-identical to its pre-call state on every recoverable error
  /// (`set_state` arity/rank failures; the 4-field meta parse of
  /// `max_size`/`_offset`/`_idx`/`rotated`). All fallible work runs on a
  /// fresh placeholder `BatchRotatingKvCache::new(0, &[])` (the exact
  /// placeholder the existing [`super::from_state`] dispatch uses; the
  /// `max_size`/`left_padding`/etc. are overwritten by the two setters);
  /// `self` is committed by a single infallible move only after both
  /// setters succeed. The default trait impl would mutate
  /// `self.keys`/`self.values`/`self.offset`/`self.left_padding` via
  /// `set_state` first, and a later malformed `rotated` boolean would then
  /// leave the cache half-restored.
  fn from_serialized(&mut self, state: Vec<Array>, meta: &[String]) -> Result<()> {
    let mut staged = BatchRotatingKvCache::new(0, &[]);
    staged.set_state(state)?;
    staged.set_meta_state(meta)?;
    // STRUCTURAL invariant guard — must match `super::from_state`'s
    // dispatcher arm (`cache/mod.rs:610-663`). The setters stay 1:1
    // with `cache.py:1301-1315` (no validation); the canonical loader
    // validates that the restored `(state, meta_state)` is one that
    // mlx-lm's own `state` getter (cache.py:1294-1307) could have
    // produced — closes the entire corrupt-restored-`_idx`/`_offset`/
    // `rotated` class. Empty buffer ⇒ fully fresh (offset=idx=0,
    // !rotated). Non-empty buffer ⇒ max_size>=1 ∧ idx<=L ∧
    // (rotated ⇒ L==max_size) ∧ L<=offset. Apply on `staged` so a
    // failure leaves `self` byte-identical to its pre-call state.
    if staged.is_empty() {
      let offset = staged.offset();
      let idx = staged.ring_idx();
      let rotated = staged.is_rotated();
      if offset != 0 || idx != 0 || rotated {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "BatchRotatingKvCache::from_serialized: empty buffer (keys=None) requires fully-fresh meta",
          "must satisfy offset=0 AND _idx=0 AND rotated=false",
          format_smolstr!("offset={offset}, _idx={idx}, rotated={rotated}"),
        )));
      }
    } else {
      let l = staged.buf_seq_len()?.unwrap_or(0);
      let max_size = staged.max_window();
      let idx = staged.ring_idx();
      let offset = staged.offset();
      let rotated = staged.is_rotated();
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
    *self = staged;
    Ok(())
  }
}

impl BatchPositionedKvCache for BatchRotatingKvCache {
  /// Per-sequence RoPE offsets `[B]` — mlx-lm
  /// `BatchRotatingKVCache.offset` (swift `batchOffset`); an owned clone
  /// (fallible per #33).
  fn batch_offset(&self) -> Result<Array> {
    self.offset.try_clone()
  }
}

/// Port of `mx.roll(a, shift, axis=-1)` for the mask
/// `BatchRotatingKVCache.make_mask` rolls (`cache.py:1355`). mlx defines
/// `out[..., i] = a[..., (i - shift) mod L]`, i.e. a positive shift moves
/// elements toward higher indices with wrap — exactly
/// `concat([a[..., L-s:], a[..., :L-s]], axis=-1)` for `s = shift mod L`
/// (identity when `s == 0`). The mask is at least 2-D
/// (`[ ..., N, offset+N]`); the roll is on the last axis. Built with the
/// generic slice/concatenate ops (rank-relative `axis=-1`); no implicit
/// eval.
fn roll_last_axis(a: &Array, shift: usize) -> Result<Array> {
  let shape = a.shape();
  let rank = shape.len();
  if rank == 0 {
    return a.try_clone();
  }
  let l = shape[rank - 1];
  if l == 0 {
    return a.try_clone();
  }
  let s = shift % l;
  if s == 0 {
    return a.try_clone();
  }
  // Slice the last axis: full on every other axis, [start,stop) on -1.
  let last = rank - 1;
  let slice_last = |start: usize, stop: usize| -> Result<Array> {
    let mut starts = vec![0i32; rank];
    let mut stops: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let strides = vec![1i32; rank];
    starts[last] = start as i32;
    stops[last] = stop as i32;
    ops::indexing::slice(a, &starts, &stops, &strides)
  };
  let tail = slice_last(l - s, l)?;
  let head = slice_last(0, l - s)?;
  ops::shape::concatenate(&[&tail, &head], -1)
}

#[cfg(test)]
mod tests {
  use super::*;

  // A `[B, n_heads, S, head_dim]` kv token batch (the cache's 4-D layout).
  fn kv(data: &[f32], b: usize, s: usize, d: usize) -> Array {
    Array::from_slice::<f32>(data, &(b, 1usize, s, d)).unwrap()
  }

  #[test]
  fn new_initial_state_is_empty_and_unrotated() {
    let c = BatchRotatingKvCache::new(4, &[0, 2]);
    assert_eq!(c.offset(), 0, "monotone _offset starts at 0");
    assert_eq!(c.max_size(), Some(4));
    assert_eq!(c.ring_idx(), 0, "ring write cursor starts at 0");
    assert!(!c.is_rotated(), "a fresh ring is not rotated");
    assert_eq!(c.max_window(), 4);
    assert_eq!(c.pad_lengths(), &[0, 2], "host mirror of left_padding");
    assert!(c.state().unwrap().is_empty(), "an empty cache yields []");
    assert_eq!(c.buf_seq_len().unwrap(), None, "no physical buffer yet");
    let mut lp = c.left_padding_arr().unwrap();
    assert_eq!(lp.to_vec::<i32>().unwrap(), vec![0, 2]);
  }

  #[test]
  fn reference_class_name_matches_mlx() {
    let c = BatchRotatingKvCache::new(2, &[0]);
    assert_eq!(c.reference_class_name(), "BatchRotatingKVCache");
  }

  #[test]
  fn update_against_zero_max_size_placeholder_errs() {
    // `max_size == 0` is the from_state placeholder; updating before
    // set_meta_state restores a real window must fail fast, not run the
    // degenerate ring arithmetic.
    let mut c = BatchRotatingKvCache::new(0, &[0]);
    let k = kv(&[1.0], 1, 1, 1);
    assert!(matches!(
      c.update(&k, &k),
      Err(Error::InvariantViolation(_))
    ));
  }

  #[test]
  fn update_rejects_kv_batch_mismatch() {
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let k = kv(&[1.0, 2.0], 1, 1, 2); // [B=1, H=1, S=1, D=2]
    let v = kv(&[1.0, 2.0, 3.0, 4.0], 2, 1, 2); // [B=2, …] — batch mismatch
    assert!(
      c.update(&k, &v).is_err(),
      "K/V batch mismatch must be rejected"
    );
  }

  #[test]
  fn single_token_update_advances_cursors() {
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let k = kv(&[1.0, 2.0], 1, 1, 2);
    let v = kv(&[3.0, 4.0], 1, 1, 2);
    let _ = c.update(&k, &v).unwrap();
    assert_eq!(c.offset(), 1, "_offset advances by S=1");
    assert_eq!(c.ring_idx(), 1, "write cursor advances by 1");
    assert!(!c.is_rotated(), "no wrap yet (idx 1 < max_size 4)");
    assert_eq!(
      c.state().unwrap().len(),
      4,
      "non-empty state is [keys, values, offset, left_padding]"
    );
  }

  #[test]
  fn ring_caps_and_rotates_past_max_size() {
    // Window of 2, fed three DISTINCT tokens with INDEPENDENT key and value
    // streams (keys 10/20/30, values 100/200/300), so the assertion is
    // load-bearing on data retention AND on the K and V buffers staying
    // distinct — a ring that overwrote the wrong slot, kept stale data,
    // zeroed a slot, or sourced values from keys would change the retained
    // contents, not merely the offset/rotated/shape metadata.
    let mut c = BatchRotatingKvCache::new(2, &[0]);
    let mut fetched = None;
    for (kval, vval) in [(10.0_f32, 100.0_f32), (20.0, 200.0), (30.0, 300.0)] {
      let k = kv(&[kval], 1, 1, 1);
      let v = kv(&[vval], 1, 1, 1);
      // `update` IS update_and_fetch — keep the LAST returned (K, V), the
      // buffers a decoder attends over immediately after the wrapping write.
      fetched = Some(c.update(&k, &v).unwrap());
    }
    assert_eq!(c.offset(), 3, "monotone _offset counts all 3 tokens");
    assert!(c.is_rotated(), "ring wrapped after exceeding max_size");
    assert_eq!(
      c.ring_idx(),
      1,
      "the 3rd write wrapped to slot 0, leaving the cursor at slot 1"
    );
    let st = c.state().unwrap();
    assert_eq!(
      st.len(),
      4,
      "non-empty state is [keys, values, offset, left_padding]"
    );
    // Assert the FULL [B, n_kv_heads, max_size, head_dim] shape, not just the
    // seq axis: the decoder attends over this exact 4-D shape, so a regression
    // that kept the right flat data but widened head_dim, dropped a rank, or
    // mis-placed the seq axis would corrupt attention while to_vec (which
    // flattens) stayed equal. [B,H,S,D] = [1,1,2,1] here (window of 2).
    let want_shape = vec![1usize, 1, 2, 1];
    assert_eq!(
      st[0].shape(),
      want_shape,
      "stored key ring shape [B,H,max_size,D]"
    );
    assert_eq!(
      st[1].shape(),
      want_shape,
      "stored value ring shape [B,H,max_size,D]"
    );
    // Once rotated (_offset >= physical buffer length) state() returns the
    // FULL physical ring (no logical 0..off slice). The 3rd token overwrote
    // slot 0 (evicting the 1st); slot 1 still holds the 2nd — physical order
    // [third, second]. K and V are asserted against their OWN independent
    // streams; the 1st token must NOT survive in either.
    let mut keys = st[0].try_clone().unwrap();
    let mut vals = st[1].try_clone().unwrap();
    assert_eq!(
      keys.to_vec::<f32>().unwrap(),
      vec![30.0, 20.0],
      "key ring holds [k3, k2]; k1 evicted"
    );
    assert_eq!(
      vals.to_vec::<f32>().unwrap(),
      vec![300.0, 200.0],
      "value ring holds [v3, v2]; v1 evicted"
    );
    // The wrap ALSO mutates the per-sequence RoPE offset and left_padding
    // that state() returns at [2]/[3] (and the batch_offset / left_padding_arr
    // / pad_lengths mirrors): offset += S each step (-> [3]) and left_padding
    // decrements by S on the rotate (-> [-1]). These drive RoPE + windowed
    // masking, so a regression leaving them stale would corrupt later batched
    // decoding while the K/V buffers above still looked correct.
    let mut off_arr = st[2].try_clone().unwrap();
    let mut lp_arr = st[3].try_clone().unwrap();
    assert_eq!(off_arr.shape(), vec![1usize], "per-seq RoPE offset is [B]");
    assert_eq!(
      off_arr.to_vec::<i32>().unwrap(),
      vec![3],
      "RoPE offset advanced by S three times"
    );
    assert_eq!(lp_arr.shape(), vec![1usize], "left_padding is [B]");
    assert_eq!(
      lp_arr.to_vec::<i32>().unwrap(),
      vec![-1],
      "left_padding decremented once, on the wrap"
    );
    let mut bo = c.batch_offset().unwrap();
    let mut lpa = c.left_padding_arr().unwrap();
    assert_eq!(
      bo.to_vec::<i32>().unwrap(),
      vec![3],
      "batch_offset() mirrors state()[2]"
    );
    assert_eq!(
      lpa.to_vec::<i32>().unwrap(),
      vec![-1],
      "left_padding_arr() mirrors state()[3]"
    );
    assert_eq!(
      c.pad_lengths(),
      &[-1],
      "pad_lengths() host mirror tracks left_padding"
    );
    // update_and_fetch's RETURN is the decode-time attention buffer — a
    // separate observable from the stored state() above. It must carry the
    // same independent K/V physical ring AND the same 4-D shape: a regression
    // that committed self.values correctly but returned the key buffer as
    // values, or returned a mis-shaped buffer, would corrupt attention while
    // leaving state() intact.
    let (mut ret_k, mut ret_v) = fetched.unwrap();
    assert_eq!(
      ret_k.shape(),
      want_shape,
      "fetched key shape matches the stored ring"
    );
    assert_eq!(
      ret_v.shape(),
      want_shape,
      "fetched value shape matches the stored ring"
    );
    assert_eq!(
      ret_k.to_vec::<f32>().unwrap(),
      vec![30.0, 20.0],
      "fetched keys mirror the key ring [k3, k2]"
    );
    assert_eq!(
      ret_v.to_vec::<f32>().unwrap(),
      vec![300.0, 200.0],
      "fetched values mirror the value ring [v3, v2]"
    );
  }

  // ── multi-token concat-then-decode: the `update_in_place` trim+rotate
  // branch (lines 607-633), unreachable from S==1-only streams ───────────
  #[test]
  fn concat_prefill_then_single_decode_trims_and_rotates() {
    // max_size=2, S=3 prefill over-allocates the buffer to max_size+S-1=3
    // via `_update_concat` (empty branch stores verbatim). The next S==1
    // decode then hits `_update_in_place`'s trim branch (buffer len 3 >
    // max_size 2) AND the rotate branch (post-trim _idx == max_size), a
    // path the S==1-only ring test cannot reach. DISTINCT K/V streams so
    // the value buffer cannot be silently sourced from keys.
    let mut c = BatchRotatingKvCache::new(2, &[0]);
    let pk = kv(&[10.0, 20.0, 30.0], 1, 3, 1);
    let pv = kv(&[100.0, 200.0, 300.0], 1, 3, 1);
    let _ = c.update(&pk, &pv).unwrap();
    assert_eq!(c.offset(), 3, "_offset = S after the prefill");
    assert_eq!(c.ring_idx(), 3, "_idx = buffer seq-len 3 after concat");
    assert!(!c.is_rotated(), "verbatim prefill is temporally ordered");

    // S==1 decode: trim_size = 3 - max_size 2 = 1 -> trim to [20,30];
    // _idx becomes max_size(2) -> rotate -> _idx=0, left_padding -= 1 -> -1
    // then rotate decrement -> -2; write slot 0 -> [40,30]/[400,300].
    let dk = kv(&[40.0], 1, 1, 1);
    let dv = kv(&[400.0], 1, 1, 1);
    let (mut rk, mut rv) = c.update(&dk, &dv).unwrap();
    assert_eq!(c.offset(), 4, "monotone _offset advanced to 4");
    assert!(
      c.is_rotated(),
      "post-trim _idx==max_size set the rotate flag"
    );
    assert_eq!(c.ring_idx(), 1, "wrapped to slot 0 then advanced to 1");
    let st = c.state().unwrap();
    let mut keys = st[0].try_clone().unwrap();
    let mut vals = st[1].try_clone().unwrap();
    assert_eq!(
      keys.shape(),
      vec![1usize, 1, 2, 1],
      "ring shrank to max_size"
    );
    assert_eq!(
      keys.to_vec::<f32>().unwrap(),
      vec![40.0, 30.0],
      "physical ring [k_new, k3]; the trimmed-off k1 is gone"
    );
    assert_eq!(
      vals.to_vec::<f32>().unwrap(),
      vec![400.0, 300.0],
      "value ring tracks its OWN stream [v_new, v3]"
    );
    // left_padding decremented twice (trim by 1, rotate by 1) -> -2.
    let mut lp = st[3].try_clone().unwrap();
    assert_eq!(lp.to_vec::<i32>().unwrap(), vec![-2], "trim(-1)+rotate(-1)");
    assert_eq!(
      c.pad_lengths(),
      &[-2],
      "host mirror tracked both decrements"
    );
    // The returned decode buffer mirrors the stored ring (buffer full now).
    assert_eq!(rk.to_vec::<f32>().unwrap(), vec![40.0, 30.0]);
    assert_eq!(rv.to_vec::<f32>().unwrap(), vec![400.0, 300.0]);
  }

  // ── update_in_place: the empty-cache grow branch on its OWN (S==1 onto a
  // never-concated cache) so the buffer-grow + zero-fill path is exercised
  // for a single batch row with distinct K/V ────────────────────────────
  #[test]
  fn single_decode_on_empty_grows_and_returns_logical_slice() {
    // A fresh cache + S==1 update goes straight through `_update_in_place`'s
    // grow branch (bk.is_none()): allocate min(step, max_size) rows, write
    // slot 0, and (since _offset 1 < max_size 4) return only the logical
    // [0:1] slice. DISTINCT K/V.
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let k = kv(&[7.0], 1, 1, 1);
    let v = kv(&[70.0], 1, 1, 1);
    let (rk, rv) = c.update(&k, &v).unwrap();
    assert_eq!(c.offset(), 1);
    assert_eq!(c.ring_idx(), 1);
    assert!(!c.is_rotated());
    // The returned view is sliced to the logical length (1), NOT the grown
    // physical buffer (the grow allocates up to max_size). The slice may be a
    // strided view, so route every host read through `contiguous` first.
    assert_eq!(
      rk.shape(),
      vec![1usize, 1, 1, 1],
      "returned the [0:_offset] slice"
    );
    let mut rk_c = ops::shape::contiguous(&rk, false).unwrap();
    let mut rv_c = ops::shape::contiguous(&rv, false).unwrap();
    assert_eq!(rk_c.to_vec::<f32>().unwrap(), vec![7.0]);
    assert_eq!(rv_c.to_vec::<f32>().unwrap(), vec![70.0]);
    // `state()` likewise slices to the logical length while _offset < L.
    let st = c.state().unwrap();
    let mut keys = ops::shape::contiguous(&st[0], false).unwrap();
    let mut vals = ops::shape::contiguous(&st[1], false).unwrap();
    assert_eq!(keys.to_vec::<f32>().unwrap(), vec![7.0]);
    assert_eq!(vals.to_vec::<f32>().unwrap(), vec![70.0]);
  }

  // ── prepare_right_padding + finalize SUCCESS path (lines 226-290): the
  // right-roll that repositions valid tokens, with a closed-form oracle ──
  #[test]
  fn finalize_rolls_buffer_and_fixes_padding() {
    // Order mirrors mlx-lm generation: prepare() (arms _lengths on the
    // fresh cache), THEN prefill (builds the buffer), THEN finalize() (rolls
    // it). max_size 8 is wide enough that the S=3 prefill never trims.
    let mut c = BatchRotatingKvCache::new(8, &[0]);
    // prepare(lengths=[2], right_padding=[1]): max(right_padding)=1>0 ->
    // _lengths = [2] + offset[0] = [2] (offset is -left_padding = 0 here).
    c.prepare_right_padding(&[2], &[1]).unwrap();
    // S=3 prefill: empty branch stores verbatim, offset -> [3]; _lengths is
    // NOT consumed by the empty concat branch.
    let pk = kv(&[10.0, 20.0, 30.0], 1, 3, 1);
    let pv = kv(&[100.0, 200.0, 300.0], 1, 3, 1);
    let _ = c.update(&pk, &pv).unwrap();
    assert_eq!(c.offset(), 3);
    // finalize: roll = max(0, offset[3] - _lengths[2]) = 1. Right-roll the
    // length-3 buffer by 1 (dynamic_roll: out[i] = x[(i-1) % 3]) ->
    // [x2, x0, x1]. left_padding += 1 -> [1]; offset -= 1 -> [2].
    c.finalize().unwrap();
    let st = c.state().unwrap();
    let mut keys = st[0].try_clone().unwrap();
    let mut vals = st[1].try_clone().unwrap();
    assert_eq!(
      keys.to_vec::<f32>().unwrap(),
      vec![30.0, 10.0, 20.0],
      "right-roll by 1 over the seq axis"
    );
    assert_eq!(
      vals.to_vec::<f32>().unwrap(),
      vec![300.0, 100.0, 200.0],
      "values rolled by the same per-row shift (own stream)"
    );
    let mut off_arr = st[2].try_clone().unwrap();
    let mut lp_arr = st[3].try_clone().unwrap();
    assert_eq!(off_arr.to_vec::<i32>().unwrap(), vec![2], "offset -= roll");
    assert_eq!(
      lp_arr.to_vec::<i32>().unwrap(),
      vec![1],
      "left_padding += roll"
    );
    assert_eq!(
      c.pad_lengths(),
      &[1],
      "host mirror refreshed after finalize"
    );
    // _lengths cleared: a subsequent S==1 decode no longer errors.
    let dk = kv(&[40.0], 1, 1, 1);
    let dv = kv(&[400.0], 1, 1, 1);
    assert!(
      c.update(&dk, &dv).is_ok(),
      "finalize cleared _lengths so decoding is allowed"
    );
  }

  // ── finalize with NO armed _lengths is a no-op; a roll of 0 still
  // round-trips (covers the `lengths` None early-return) ─────────────────
  #[test]
  fn finalize_without_lengths_is_noop() {
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let k = kv(&[1.0, 2.0], 1, 2, 1);
    let _ = c.update(&k, &k).unwrap();
    let before = c.offset();
    c.finalize().unwrap();
    assert_eq!(c.offset(), before, "no _lengths -> finalize is a no-op");
  }

  // ── prepare_right_padding with all-zero right_padding leaves _lengths
  // unarmed (the `max(right_padding) > 0` guard is false) ────────────────
  #[test]
  fn prepare_right_padding_zero_does_not_arm_lengths() {
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let k = kv(&[1.0], 1, 1, 1);
    let _ = c.update(&k, &k).unwrap();
    // right_padding all-zero -> _lengths stays None -> a later S==1 decode
    // does NOT trip the "finalize() must be called" guard.
    c.prepare_right_padding(&[1], &[0]).unwrap();
    let k2 = kv(&[2.0], 1, 1, 1);
    assert!(
      c.update(&k2, &k2).is_ok(),
      "unarmed _lengths must not block decoding"
    );
  }

  // ── update_in_place guard: a decode while _lengths is armed errors
  // (mlx-lm RuntimeError "finalize() must precede decoding") ─────────────
  #[test]
  fn decode_before_finalize_errors() {
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let k = kv(&[1.0, 2.0], 1, 2, 1);
    let _ = c.update(&k, &k).unwrap(); // prefill so the buffer is non-empty
    c.prepare_right_padding(&[2], &[1]).unwrap(); // arm _lengths
    let d = kv(&[3.0], 1, 1, 1);
    assert!(
      matches!(c.update(&d, &d), Err(Error::InvariantViolation(_))),
      "S==1 decode with _lengths armed must be an InvariantViolation"
    );
  }

  // ── multi-token update_concat onto a NON-empty, NON-rotated cache (the
  // temporal_order=None + slice-off-end + the lengths-roll branch) ───────
  #[test]
  fn concat_onto_nonrotated_with_lengths_rolls_and_advances() {
    // Wide window so nothing trims; arm _lengths, prefill S=2 (empty branch
    // verbatim), then a second S=2 update hits `_update_concat`'s
    // (Some,Some) arm: temporal_order returns None (not rotated -> covers
    // the None match), the slice-off-end check, and the `_lengths` roll.
    let mut c = BatchRotatingKvCache::new(16, &[0]);
    c.prepare_right_padding(&[1], &[1]).unwrap(); // _lengths = [1] + 0 = [1]
    let p1k = kv(&[1.0, 2.0], 1, 2, 1);
    let p1v = kv(&[11.0, 12.0], 1, 2, 1);
    let _ = c.update(&p1k, &p1v).unwrap(); // offset -> [2], _lengths=[1]
    assert_eq!(c.offset(), 2);
    let p2k = kv(&[3.0, 4.0], 1, 2, 1);
    let p2v = kv(&[13.0, 14.0], 1, 2, 1);
    // Second concat: temporal_order None; slice check (L==_idx, no slice);
    // _lengths roll = max(0, offset[2] - _lengths[1]) = 1 applied per-row to
    // the EXISTING [1,2] buffer before appending [3,4]. offset += S.
    let (mut rk, _) = c.update(&p2k, &p2v).unwrap();
    assert_eq!(c.offset(), 4, "_offset counts 2 + 2 tokens");
    assert!(!c.is_rotated(), "still under max_size -> not rotated");
    // The buffer holds the (rolled) prior context ++ the appended tokens; we
    // assert the length + the appended-tail tokens (closed-form on the tail,
    // which the roll never touches) and that the offset bookkeeping is right.
    assert_eq!(rk.shape(), vec![1usize, 1, 4, 1], "2 retained + 2 appended");
    let kvec = rk.to_vec::<f32>().unwrap();
    assert_eq!(
      &kvec[2..],
      &[3.0, 4.0],
      "appended tokens are verbatim at the tail"
    );
  }

  // ── state()/meta_state() <-> set_state()/set_meta_state() round-trip and
  // the empty-set_state reset (covers both set_state arms + meta parse) ──
  #[test]
  fn state_meta_set_state_round_trip() {
    let mut c = BatchRotatingKvCache::new(4, &[1, 0]);
    let pk = kv(&[1.0, 2.0], 2, 1, 1); // [B=2,H=1,S=1,D=1] -> 2 elements
    let pv = kv(&[10.0, 20.0], 2, 1, 1);
    let _ = c.update(&pk, &pv).unwrap(); // S==1 -> _offset/_idx = 1
    let meta = c.meta_state();
    assert_eq!(meta.len(), 4, "(max_size, _offset, _idx, rotated)");
    assert_eq!(meta[0], "4");
    assert_eq!(meta[1], "1", "_offset advanced by S=1");
    assert_eq!(meta[3], "false");
    let st = c.state().unwrap();
    assert_eq!(st.len(), 4);

    // Restore into a placeholder via the two setters (the from_serialized
    // sub-steps): set_state (4-arm) then set_meta_state.
    let mut restored = BatchRotatingKvCache::new(0, &[]);
    restored.set_state(st).unwrap();
    restored.set_meta_state(&meta).unwrap();
    assert_eq!(restored.offset(), 1, "restored _offset matches meta[1]");
    assert_eq!(
      restored.max_size(),
      Some(4),
      "restored max_size matches meta[0]"
    );
    assert!(!restored.is_rotated());
    assert_eq!(
      restored.pad_lengths(),
      &[1, 0],
      "left_padding host mirror restored"
    );

    // Empty set_state resets ALL per-seq runtime state but keeps left_padding
    // and recomputes offset = -left_padding.
    restored.set_meta_state(&meta).unwrap();
    restored.set_state(Vec::new()).unwrap();
    assert!(restored.is_empty(), "empty set_state clears the buffer");
    assert_eq!(restored.offset(), 0, "scalar _offset reset to 0");
    assert_eq!(restored.ring_idx(), 0, "ring cursor reset");
    assert!(!restored.is_rotated());
    let mut bo = restored.batch_offset().unwrap();
    assert_eq!(
      bo.to_vec::<i32>().unwrap(),
      vec![-1, 0],
      "offset reset to -left_padding"
    );
  }

  // ── set_state arity guard: any count other than 0 or 4 is OutOfRange ───
  #[test]
  fn set_state_wrong_arity_errs() {
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let a = kv(&[1.0], 1, 1, 1);
    let three = vec![
      a.try_clone().unwrap(),
      a.try_clone().unwrap(),
      a.try_clone().unwrap(),
    ];
    assert!(
      matches!(c.set_state(three), Err(Error::OutOfRange(_))),
      "a 3-array state is neither empty nor the 4-tuple"
    );
  }

  // ── set_meta_state guards: wrong length, unparseable int, bad bool ─────
  #[test]
  fn set_meta_state_guards() {
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    // Wrong length.
    assert!(matches!(
      c.set_meta_state(&["4".to_string(), "0".to_string()]),
      Err(Error::LengthMismatch(_))
    ));
    // Unparseable _offset.
    assert!(matches!(
      c.set_meta_state(&[
        "4".to_string(),
        "nope".to_string(),
        "0".to_string(),
        "false".to_string()
      ]),
      Err(Error::Parse(_))
    ));
    // Unknown rotated token.
    assert!(matches!(
      c.set_meta_state(&[
        "4".to_string(),
        "0".to_string(),
        "0".to_string(),
        "maybe".to_string()
      ]),
      Err(Error::UnknownEnumValue(_))
    ));
    // Valid round-trip accepts BOTH Rust "true" and Python-style "True".
    assert!(
      c.set_meta_state(&[
        "4".to_string(),
        "4".to_string(),
        "2".to_string(),
        "True".to_string()
      ])
      .is_ok()
    );
    assert!(c.is_rotated(), "case-insensitive True parsed as rotated");
  }

  // ── trim: the n==0 early return + the decrementing path ────────────────
  #[test]
  fn trim_zero_is_noop_and_positive_decrements() {
    let mut c = BatchRotatingKvCache::new(8, &[0]);
    let p = kv(&[1.0, 2.0, 3.0], 1, 3, 1);
    let _ = c.update(&p, &p).unwrap(); // _offset 3 (< 8 -> trimmable)
    assert!(c.is_trimmable(), "_offset 3 < max_size 8");
    // n==0 (and also n clamped to 0 when _offset is small) -> early Ok(0).
    assert_eq!(c.trim(0).unwrap(), 0, "trimming 0 is a no-op");
    assert_eq!(c.offset(), 3, "offset unchanged by a 0-trim");
    // Positive trim: n = min(_offset, n); _offset/_idx/offset -= n.
    assert_eq!(c.trim(2).unwrap(), 2);
    assert_eq!(c.offset(), 1, "_offset 3 - 2");
    let mut bo = c.batch_offset().unwrap();
    assert_eq!(bo.to_vec::<i32>().unwrap(), vec![1], "per-seq offset -= 2");
    // Fill past max_size -> no longer trimmable.
    let mut c2 = BatchRotatingKvCache::new(2, &[0]);
    let big = kv(&[1.0, 2.0, 3.0, 4.0], 1, 4, 1);
    let _ = c2.update(&big, &big).unwrap();
    assert!(!c2.is_trimmable(), "_offset 4 >= max_size 2");
  }

  // ── make_mask: max_size==0 placeholder is rejected before any arithmetic
  #[test]
  fn make_mask_against_zero_max_size_placeholder_errs() {
    let c = BatchRotatingKvCache::new(0, &[0]);
    assert!(matches!(
      c.make_mask(1, None, false),
      Err(Error::InvariantViolation(_))
    ));
  }

  // ── make_mask: the rotated N==1 path with idx>=max_size folds idx->0
  // before the roll (covers the `idx >= max_size { 0 }` branch) ──────────
  #[test]
  fn make_mask_rotated_with_idx_at_window_rolls() {
    // Restore a (corrupt-but-rank-free) rotated state where _idx == max_size
    // via set_meta_state: the rotate branch then folds idx -> 0 before the
    // last-axis roll. keys=None is fine (make_mask never reads the buffer).
    let mut c = BatchRotatingKvCache::new(2, &[0]);
    c.set_meta_state(&[
      "2".to_string(),
      "5".to_string(),
      "2".to_string(),
      "true".to_string(),
    ])
    .unwrap();
    match c.make_mask(1, Some(2), false).unwrap() {
      MaskMode::Array(mut m) => {
        // N=1, window 2, offset = min(max_size-1=1, _offset=5) = 1 ->
        // total = offset + N = 2; the mask's last axis is `total`.
        assert_eq!(m.shape()[m.shape().len() - 1], 2);
        // The rolled mask is built by `concatenate` (contiguous); reading it
        // back exercises the full rotated codepath without panicking.
        let _ = m.to_vec::<bool>().unwrap();
      }
      _ => panic!("rotated N==1 must return a rolled mask array"),
    }
  }

  // ── make_mask: trim/rotate delta exceeding the f32 exact-int limit is a
  // recoverable OutOfRange (corrupt restored _idx) ───────────────────────
  #[test]
  fn make_mask_huge_restored_idx_is_out_of_range() {
    // A restored _idx far beyond 2^24 makes the trim/rotate `delta` exceed
    // the f32 exact-integer limit; make_mask must reject it (not silently
    // build a wrong mask). N==1 so the `_idx + int(N>1)` add doesn't catch
    // it first; the delta-vs-2^24 guard does.
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let huge = (1usize << 25).to_string();
    c.set_meta_state(&["4".to_string(), huge.clone(), huge, "false".to_string()])
      .unwrap();
    assert!(matches!(
      c.make_mask(1, None, false),
      Err(Error::OutOfRange(_))
    ));
  }

  // ── make_mask: usize::MAX restored _idx with N>1 overflows `_idx + 1`
  // (the checked-add `idx_term`) -> ArithmeticOverflow ───────────────────
  #[test]
  fn make_mask_idx_max_with_n_gt_1_overflows() {
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let max = usize::MAX.to_string();
    c.set_meta_state(&["4".to_string(), max.clone(), max, "false".to_string()])
      .unwrap();
    assert!(matches!(
      c.make_mask(2, None, false),
      Err(Error::ArithmeticOverflow(_))
    ));
  }

  // ── nbytes: 0 when empty, keys.nbytes + values.nbytes once populated ───
  #[test]
  fn nbytes_tracks_buffers() {
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    assert_eq!(c.nbytes(), 0, "an empty cache has no buffer bytes");
    // S=2 prefill, [B=1,H=1,S=2,D=1] f32 = 2 elements * 4 bytes, for BOTH
    // keys and values -> 2*(2*4) = 16.
    let k = kv(&[1.0, 2.0], 1, 2, 1);
    let _ = c.update(&k, &k).unwrap();
    assert_eq!(
      c.nbytes(),
      16,
      "keys.nbytes + values.nbytes (2 elems each, f32)"
    );
  }

  // ── materialize: evals every live buffer in place; idempotent + no
  // observable state change (covers the keys/values/lengths Some arms) ───
  #[test]
  fn materialize_evals_live_buffers() {
    let mut c = BatchRotatingKvCache::new(8, &[0]);
    // Arm _lengths so the lengths Some-arm of materialize is exercised too.
    c.prepare_right_padding(&[1], &[1]).unwrap();
    let pk = kv(&[1.0, 2.0], 1, 2, 1);
    let pv = kv(&[11.0, 12.0], 1, 2, 1);
    let _ = c.update(&pk, &pv).unwrap();
    let before = c.state().unwrap();
    let mut kb = before[0].try_clone().unwrap();
    let kvec = kb.to_vec::<f32>().unwrap();
    c.materialize().unwrap();
    // No observable change: the same logical state survives the eval.
    let after = c.state().unwrap();
    let mut ka = after[0].try_clone().unwrap();
    assert_eq!(
      ka.to_vec::<f32>().unwrap(),
      kvec,
      "materialize preserves data"
    );
    assert_eq!(c.offset(), 2);
    // Materialize on an EMPTY cache (keys/values/lengths None arms) is a
    // no-op over the [B] position arrays only.
    let mut empty = BatchRotatingKvCache::new(4, &[0, 0]);
    assert!(empty.materialize().is_ok(), "empty materialize is a no-op");
  }

  // ── copy: an independent deep clone whose mutation does not touch the
  // original (covers every Some/None field arm + scalars) ────────────────
  #[test]
  fn copy_is_independent() {
    let mut c = BatchRotatingKvCache::new(4, &[2, 0]);
    // Populate keys/values; leave _lengths None (covers both arms across the
    // two clones below). [B=2,H=1,S=1,D=1] -> 2 elements.
    let pk = kv(&[1.0, 2.0], 2, 1, 1);
    let pv = kv(&[10.0, 20.0], 2, 1, 1);
    let _ = c.update(&pk, &pv).unwrap();
    let cloned = c.copy().unwrap();
    assert_eq!(cloned.offset(), c.offset(), "scalar _offset copied");
    assert_eq!(cloned.max_size(), c.max_size(), "max_size copied");
    assert!(!cloned.is_empty(), "buffers copied (not None)");
    let cst = cloned.state().unwrap();
    assert_eq!(cst.len(), 4);
    // Mutating the ORIGINAL after the copy must not change the clone.
    let d = kv(&[5.0, 6.0], 2, 1, 1);
    let _ = c.update(&d, &d).unwrap();
    assert_eq!(c.offset(), 2, "original advanced");
    assert_eq!(cloned.offset(), 1, "clone is independent of the original");

    // A copy of an EMPTY cache hits the keys/values/lengths None arms.
    let empty = BatchRotatingKvCache::new(4, &[0]);
    let ec = empty.copy().unwrap();
    assert!(ec.is_empty(), "empty cache copies to an empty cache");
  }

  // ── from_serialized SUCCESS round-trip + the empty-buffer structural
  // guard (empty state must carry fully-fresh meta) ──────────────────────
  #[test]
  fn from_serialized_round_trip_and_empty_guard() {
    let mut c = BatchRotatingKvCache::new(4, &[0, 0]);
    // B=2 (left_padding len 2), H=1, S=1, D=1 → 2 elements total.
    let pk = kv(&[1.0, 2.0], 2, 1, 1);
    let pv = kv(&[10.0, 20.0], 2, 1, 1);
    let _ = c.update(&pk, &pv).unwrap();
    let st = c.state().unwrap();
    let meta = c.meta_state();
    let mut dst = BatchRotatingKvCache::new(0, &[]);
    dst.from_serialized(st, &meta).unwrap();
    assert_eq!(dst.offset(), 1);
    assert_eq!(dst.max_size(), Some(4));
    assert!(!dst.is_empty());

    // Empty state (keys=None) with NON-fresh meta (offset != 0) is the
    // impossible combination -> rejected (OutOfRange), self left unchanged.
    let mut guard = BatchRotatingKvCache::new(4, &[0]);
    let prev_off = guard.offset();
    let bad_meta = vec![
      "4".to_string(),
      "2".to_string(),
      "0".to_string(),
      "false".to_string(),
    ];
    assert!(matches!(
      guard.from_serialized(Vec::new(), &bad_meta),
      Err(Error::OutOfRange(_))
    ));
    assert_eq!(
      guard.offset(),
      prev_off,
      "rejected restore leaves self unchanged"
    );
  }

  // ── from_serialized non-empty structural guards: rotated requires
  // L==max_size, and L must not exceed _offset ───────────────────────────
  #[test]
  fn from_serialized_nonempty_structural_guards() {
    // Build a valid non-empty (keys, values, offset, left_padding) state by
    // hand: B=1, L=2 physical buffer, _offset >= L.
    let keys = kv(&[1.0, 2.0], 1, 2, 1);
    let vals = kv(&[11.0, 12.0], 1, 2, 1);
    let off = Array::from_slice::<i32>(&[2], &(1usize,)).unwrap();
    let lp = Array::from_slice::<i32>(&[0], &(1usize,)).unwrap();
    let mk_state = || {
      vec![
        keys.try_clone().unwrap(),
        vals.try_clone().unwrap(),
        off.try_clone().unwrap(),
        lp.try_clone().unwrap(),
      ]
    };

    // rotated=true but L(2) != max_size(4) -> LengthMismatch.
    let mut a = BatchRotatingKvCache::new(0, &[]);
    let rotated_bad = vec![
      "4".to_string(),
      "2".to_string(),
      "2".to_string(),
      "true".to_string(),
    ];
    assert!(matches!(
      a.from_serialized(mk_state(), &rotated_bad),
      Err(Error::LengthMismatch(_))
    ));

    // L(2) > _offset(1) is impossible from mlx-lm's getter -> OutOfRange.
    let mut b = BatchRotatingKvCache::new(0, &[]);
    let l_gt_off = vec![
      "4".to_string(),
      "1".to_string(),
      "2".to_string(),
      "false".to_string(),
    ];
    assert!(matches!(
      b.from_serialized(mk_state(), &l_gt_off),
      Err(Error::OutOfRange(_))
    ));

    // A consistent rotated=true with L == max_size(2) and _idx <= L is
    // accepted (the happy non-empty rotated branch).
    let keys2 = kv(&[1.0, 2.0], 1, 2, 1);
    let vals2 = kv(&[11.0, 12.0], 1, 2, 1);
    let off2 = Array::from_slice::<i32>(&[5], &(1usize,)).unwrap();
    let lp2 = Array::from_slice::<i32>(&[0], &(1usize,)).unwrap();
    let mut ok = BatchRotatingKvCache::new(0, &[]);
    let good = vec![
      "2".to_string(),
      "5".to_string(),
      "2".to_string(),
      "true".to_string(),
    ];
    ok.from_serialized(vec![keys2, vals2, off2, lp2], &good)
      .unwrap();
    assert!(ok.is_rotated(), "consistent rotated state restored");
    assert_eq!(ok.max_size(), Some(2));
  }

  // ── corrupt restored `_idx == usize::MAX` (injected directly via the
  // two setters, bypassing the from_serialized structural guard) overflows
  // the checked `_idx + S` in BOTH update paths -> ArithmeticOverflow with
  // no partial mutation ──────────────────────────────────────────────────
  fn restore_corrupt_idx_cache() -> BatchRotatingKvCache {
    // A valid non-empty 4-D buffer (B=1, L=2) so the grow branch does NOT
    // fire (which would overwrite `w_idx`); `_offset` is small so the
    // `_offset + S` add does NOT overflow first, isolating the `_idx + S`
    // checked-add. `_idx` is the hostile `usize::MAX`.
    let mut c = BatchRotatingKvCache::new(4, &[0]);
    let keys = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 1, 2, 1)).unwrap();
    let vals = Array::from_slice::<f32>(&[11.0, 12.0], &(1usize, 1, 2, 1)).unwrap();
    let offset = Array::from_slice::<i32>(&[0], &(1usize,)).unwrap();
    let lp = Array::from_slice::<i32>(&[0], &(1usize,)).unwrap();
    c.set_state(vec![keys, vals, offset, lp]).unwrap();
    // off=0 (no _offset+S overflow), idx=usize::MAX, not rotated.
    let max = usize::MAX.to_string();
    c.set_meta_state(&["4".to_string(), "0".to_string(), max, "false".to_string()])
      .unwrap();
    c
  }

  #[test]
  fn corrupt_idx_overflow_rejected_update_in_place() {
    // S==1 -> `_update_in_place`: new_w_idx = w_idx(usize::MAX) + 1 overflows.
    let mut c = restore_corrupt_idx_cache();
    let d = kv(&[9.0], 1, 1, 1);
    assert!(matches!(
      c.update(&d, &d),
      Err(Error::ArithmeticOverflow(_))
    ));
    // No partial mutation: the restored corrupt _idx is still in place.
    assert_eq!(c.ring_idx(), usize::MAX, "rejected update mutated nothing");
    assert_eq!(c.offset(), 0);
  }

  #[test]
  fn corrupt_idx_overflow_rejected_update_concat() {
    // S>1 -> `_update_concat`: idx_plus_1 = w_idx(usize::MAX) + 1 overflows
    // (the slice/trim pivot add), before any commit.
    let mut c = restore_corrupt_idx_cache();
    let d = kv(&[9.0, 8.0], 1, 2, 1);
    assert!(matches!(
      c.update(&d, &d),
      Err(Error::ArithmeticOverflow(_))
    ));
    assert_eq!(c.ring_idx(), usize::MAX, "rejected update mutated nothing");
    assert_eq!(c.offset(), 0);
  }
}
