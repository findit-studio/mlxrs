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
  error::{Error, Result},
  lm::cache::{
    BatchPositionedKvCache, KvCache, MaskMode,
    batch::{batch_head_dim, create_causal_mask_batched, dynamic_roll, ivec, validate_kv_compat},
    util::{concat_seq, nbytes, seq_len, seq_slice},
  },
  ops,
};

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
  /// Per-sequence raw position — mlx-lm `offset` (`[B]`, `I32`; starts at
  /// `-left_padding`, the per-seq RoPE offset / mask `left_padding`).
  offset: Array,
  /// Window length — mlx-lm `max_size`.
  max_size: usize,
  /// Physical ring write cursor — mlx-lm `_idx`. Wraps to `0` (no `keep`)
  /// once it reaches `max_size`.
  idx: usize,
  /// Scalar monotone counter — mlx-lm `_offset` (drives grow/trim/rotate
  /// and is the trim/`is_trimmable` quantity; *not* capped at `max_size`).
  off: usize,
  /// Whether the ring has wrapped — mlx-lm `rotated` (the returned buffer
  /// is then in physical, not temporal, order).
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
      offset,
      max_size,
      idx: 0,
      off: 0,
      rotated: false,
      lengths: None,
    }
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
      // Infallible commit tail.
      if let Some((nk, nv)) = rolled {
        self.keys = Some(nk);
        self.values = Some(nv);
      }
      self.left_padding = new_left_padding;
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
    let new_off = self
      .off
      .checked_add(s)
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "BatchRotatingKvCache update: _offset ({}) + S ({s}) overflows usize",
          self.off
        ),
      })?;
    let (wk, wv): (Array, Array);
    let mut w_offset = self.offset.try_clone()?;
    let mut w_left_padding = self.left_padding.try_clone()?;
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
        let idx_plus_1 = w_idx.checked_add(1).ok_or_else(|| Error::ShapeMismatch {
          message: format!("BatchRotatingKvCache update: _idx ({w_idx}) + 1 overflows usize"),
        })?;
        let trim_size = idx_plus_1.saturating_sub(self.max_size);
        if trim_size > 0 {
          // left_padding -= trim_size (cache.py:1196).
          let ts = ops::misc::astype(
            &Array::full::<f32>(&(1usize,), trim_size as f32)?,
            Dtype::I32,
          )?;
          w_left_padding = ops::arithmetic::subtract(&w_left_padding, &ts)?;
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

    // ── Infallible commit tail (no `?` past this point) ──────────────
    self.keys = Some(wk);
    self.values = Some(wv);
    self.offset = new_offset_arr;
    self.left_padding = w_left_padding;
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
      return Err(Error::Backend {
        message: "finalize() should be called before decoding with BatchRotatingKvCache".into(),
      });
    }
    // Rank validation is the public-`update` entry's responsibility
    // (`update` calls `validate_kv_compat(keys, values)?` before dispatching
    // to either `update_in_place` (S==1) or `update_concat` (S>1) — see
    // `update`'s body above). The previous in-method rank check here was
    // redundant with that centralized validation; remove to keep error
    // reporting consistent across both update paths (Copilot review
    // #3271560177). The `keys.shape()` index reads below are now backed
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
    let new_off = self
      .off
      .checked_add(s)
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "BatchRotatingKvCache update: _offset ({}) + S ({s}) overflows usize",
          self.off
        ),
      })?;

    // Working copies of the mutable ring state (start = current `self`).
    let (mut bk, mut bv) = match (&self.keys, &self.values) {
      (Some(k), Some(v)) => (Some(k.try_clone()?), Some(v.try_clone()?)),
      _ => (None, None),
    };
    let mut w_idx = self.idx;
    let mut w_rotated = self.rotated;
    let mut w_left_padding = self.left_padding.try_clone()?;

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
        return Err(Error::Backend {
          message: "BatchRotatingKvCache: empty buffer after grow (unreachable)".into(),
        });
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
    }

    // _idx += S (cache.py:1254). `w_idx` may still be a corrupt restored
    // `self.idx` (usize::MAX) if none of grow/trim/rotate reassigned it;
    // `w_idx + s` would debug-panic / release-wrap — same overflow class
    // as `_offset + S`. Compute it CHECKED here, BEFORE `set_seq`/the
    // commit tail, so the `Err` leaves the cache fully unmutated (and the
    // wrong out-of-range `set_seq` splice never even runs). Byte-identical
    // to mlx-lm's unbounded `self._idx += S` for every valid input.
    let new_w_idx = w_idx.checked_add(s).ok_or_else(|| Error::ShapeMismatch {
      message: format!("BatchRotatingKvCache update: _idx ({w_idx}) + S ({s}) overflows usize"),
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

    // ── Infallible commit tail (no `?` past this point) ──────────────
    self.keys = Some(nk);
    self.values = Some(nv);
    self.offset = new_offset_arr;
    self.left_padding = w_left_padding;
    self.off = new_off; // _offset += S (overflow already rejected above)
    self.idx = new_w_idx;
    self.rotated = w_rotated;
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
  /// recoverable [`Error::ShapeMismatch`] on **both** the `S==1` and
  /// `S>1` paths (never a panic, never a silent K/V desync), exactly
  /// mlx-lm's error semantics — observable behavior for every valid input
  /// is unchanged.
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    // Reject the constructor placeholder `max_size == 0` here too —
    // symmetric with `make_mask` (Copilot review #3271308764). An update
    // against `max_size == 0` would otherwise drive `trim_size` /
    // `_idx - max_size + ...` arithmetic into degenerate / silently
    // wrong values; the from_state path never reaches here on the
    // placeholder (set_meta_state restores max_size before the first
    // update).
    if self.max_size == 0 {
      return Err(Error::Backend {
        message: "BatchRotatingKvCache::update: max_size is 0 (the constructor placeholder \
                  used by from_state); set max_size via set_meta_state or construct with \
                  new(max_size > 0, left_padding) before calling update"
          .into(),
      });
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
        // `Error::ShapeMismatch`, never a panic. Validate before assigning
        // any field so a bad buffer leaves the cache unmutated.
        seq_len("keys", &keys)?;
        batch_head_dim("values", &values)?;
        // ── Infallible commit tail ──────────────────────────────────────
        self.keys = Some(keys);
        self.values = Some(values);
        self.offset = offset;
        self.left_padding = left_padding;
        // Also clear `lengths` here, matching the empty-state branch above
        // (Copilot review #3271560146). If `prepare_right_padding()` armed
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
      n => Err(Error::Backend {
        message: format!("BatchRotatingKvCache state must have 0 or 4 arrays, got {n}"),
      }),
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
      return Err(Error::Backend {
        message: format!(
          "BatchRotatingKvCache meta_state must have 4 values, got {}",
          m.len()
        ),
      });
    }
    let parse = |i: usize, name: &str| -> Result<usize> {
      m[i].parse::<usize>().map_err(|e| Error::Backend {
        message: format!("BatchRotatingKvCache meta_state {name} ({:?}): {e}", m[i]),
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
        return Err(Error::Backend {
          message: format!("BatchRotatingKvCache meta_state rotated ({other:?}): expected a bool"),
        });
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
    // (`ws == 0` and `offset == 0`). Reject as a recoverable error
    // (Copilot review #3271308764).
    if self.max_size == 0 {
      return Err(Error::Backend {
        message: "BatchRotatingKvCache::make_mask: max_size is 0 (the constructor placeholder \
                  used by from_state); set max_size via set_meta_state or construct with \
                  new(max_size > 0, left_padding) before calling make_mask"
          .into(),
      });
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
    let idx_term =
      self
        .idx
        .checked_add(usize::from(n > 1))
        .ok_or_else(|| Error::ShapeMismatch {
          message: format!(
            "BatchRotatingKvCache::make_mask: _idx ({}) + int(N>1) overflows usize",
            self.idx
          ),
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
    // mask instead of erroring (the round-1/6 overflow class, at the
    // lossy cast rather than the add). Keep `delta` in `usize`
    // (`trim_size + int(rotated)`, checked), then require it to be
    // **exactly** representable through the `f32`-built I32 scalar the
    // MLX subtract uses (same `2^24` exact-int rationale as
    // `mask::scalar_i32`/`iarange`): for every real `_idx` `trim_size`
    // is tiny so this is byte-identical to mlx-lm's unbounded
    // `_idx - max_size + int(N>1)`; only the corrupt huge value is a
    // recoverable `ShapeMismatch` (this is `&self` — the `Err` is
    // inherently side-effect-free).
    let mut lp = self.left_padding.try_clone()?;
    let delta = trim_size.checked_add(usize::from(rotated)).ok_or_else(|| {
      Error::ShapeMismatch {
        message: format!(
          "BatchRotatingKvCache::make_mask: trim_size ({trim_size}) + int(rotated) overflows usize"
        ),
      }
    })?;
    if delta != 0 {
      // `Array::full`/`scalar_i32` build the scalar through `f32`; an
      // `f32` represents every integer in `[0, 2^24]` exactly. A real
      // `delta` is far below this; a corrupt one beyond it would round
      // (silent wrong mask) — reject it instead.
      const F32_EXACT_INT_MAX: usize = 1usize << 24;
      if delta > F32_EXACT_INT_MAX {
        return Err(Error::ShapeMismatch {
          message: format!(
            "BatchRotatingKvCache::make_mask: trim/rotate delta ({delta}) exceeds the exact f32 integer limit (2^24) — _idx restored too large for this path"
          ),
        });
      }
      let d = ops::misc::astype(&Array::full::<f32>(&(1usize,), delta as f32)?, Dtype::I32)?;
      lp = ops::arithmetic::subtract(&lp, &d)?;
    }

    // mask &= rinds >= expand_dims(left_padding, (1,2,3)) (cache.py:1349).
    // rinds = arange(offset + N); rebuild it and broadcast lp to [B,1,1,1].
    let total = offset.checked_add(n).ok_or_else(|| Error::ShapeMismatch {
      message: format!(
        "BatchRotatingKvCache::make_mask: offset ({offset}) + N ({n}) overflows usize"
      ),
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

  /// P1 #110: per-layer fast-path downcast target — see the
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
    let (invalid, reason) = if staged.is_empty() {
      let offset = staged.offset();
      let idx = staged.ring_idx();
      let rotated = staged.is_rotated();
      if offset != 0 || idx != 0 || rotated {
        (
          true,
          format!(
            "empty buffer (keys=None) requires fully-fresh meta: \
             offset={offset} _idx={idx} rotated={rotated} (need 0/0/false)"
          ),
        )
      } else {
        (false, String::new())
      }
    } else {
      let l = staged.buf_seq_len()?.unwrap_or(0);
      let max_size = staged.max_window();
      let idx = staged.ring_idx();
      let offset = staged.offset();
      let rotated = staged.is_rotated();
      if max_size == 0 {
        (
          true,
          format!("non-empty buffer requires max_size >= 1, got max_size=0 (L={l})"),
        )
      } else if idx > l {
        (
          true,
          format!("_idx ({idx}) > keys seq-len L ({l}): write cursor beyond physical buffer"),
        )
      } else if rotated && l != max_size {
        (
          true,
          format!("rotated=true requires L == max_size, got L={l} max_size={max_size}"),
        )
      } else if l > offset {
        (
          true,
          format!(
            "L ({l}) > _offset ({offset}): mlx-lm getter emits keys[:_offset, :], so L <= _offset always"
          ),
        )
      } else {
        (false, String::new())
      }
    };
    if invalid {
      return Err(Error::Backend {
        message: format!(
          "BatchRotatingKvCache: restored state/meta_state is inconsistent (not a state mlx-lm's own round-trip could produce): {reason}"
        ),
      });
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
