//! [`StandardKvCache`] — the full-attention append-and-fetch cache.

use crate::{
  array::Array,
  error::{Error, InvariantViolationPayload, OutOfRangePayload, Result},
  lm::cache::{
    KvCache, MaskMode, mask,
    util::{concat_seq, nbytes, seq_len, slice_seq},
  },
  ops::{indexing::slice_update, misc::astype},
};
use smol_str::format_smolstr;

/// Step-buffer growth granularity (mlx-lm `KVCache.step` / swift
/// `KVCacheSimple.step`, both `256`): the buffer's sequence axis is reallocated
/// in multiples of this many positions, then filled in place.
const STEP: usize = 256;

/// Append-and-fetch KV cache — the default cache for full-attention models.
///
/// Faithful port of mlx-lm `KVCache` (`mlx_lm/models/cache.py:325`) and swift
/// `KVCacheSimple` (`MLXLMCommon/KVCache.swift`), which are byte-identical: a
/// **pre-allocated step buffer**. [`keys`](Self::keys) / [`values`](Self::values)
/// are buffers whose sequence axis is `>= offset`, rounded up to a multiple of
/// [`STEP`]; `offset` is the logical length, so the live content is
/// `keys[.., :offset, :]`. Each [`update`](KvCache::update) writes the new `S`
/// keys/values into the `[offset, offset + S)` slot **in place** (`slice_update`,
/// which mlx evaluates as an in-place buffer donation when the input is unshared
/// — an O(S) write) and returns the `[.., :offset, :]` prefix. The buffer grows
/// (concatenate a fresh `ceil(S / STEP) * STEP` zero block) only when a write
/// would overflow it, so the per-token concatenate cost is amortized by `STEP`
/// — **not** the O(N²) of re-concatenating the whole cache every token (that is
/// mlx-lm's `ConcatenateKVCache`, whose own docstring says to prefer `KVCache`).
///
/// `trim(n)` drops the most recent `min(offset, n)` tokens by decrementing
/// `offset` and reusing the buffer (mlx-lm / swift `trim`); [`state`](KvCache::state)
/// returns the `[.., :offset, :]` prefix (mlx-lm / swift `state` getter), so the
/// serialized form is identical to the old append-and-fetch cache.
#[derive(Default)]
pub struct StandardKvCache {
  /// Key buffer; its sequence axis is `>= offset`, rounded to a [`STEP`]
  /// multiple. The logical keys are `keys[.., :offset, :]`.
  keys: Option<Array>,
  /// Value buffer (paired with [`Self::keys`]).
  values: Option<Array>,
  /// Logical cached sequence length (mlx-lm `KVCache.offset`).
  offset: usize,
}

impl StandardKvCache {
  /// A new, empty cache.
  pub fn new() -> Self {
    Self::default()
  }

  /// A `[B, n_kv_heads, block, head_dim]` zero block matching `template`'s
  /// non-sequence axes and dtype — the fresh chunk that EXTENDS an existing
  /// buffer. The grow path passes the current `keys`/`values` buffer as
  /// `template`, so the appended block carries the buffer's own
  /// batch/n_kv_heads/head_dim and `concat_seq` matches. `template` is an
  /// already-rank-checked 4-D array. The FIRST allocation instead uses
  /// [`first_blocks`], which derives the value buffer's batch/n_kv_heads from
  /// `keys` (not `values`).
  fn make_zero_block(template: &Array, block: usize) -> Result<Array> {
    let sh = template.shape();
    let zeros = Array::zeros::<f32>(&(sh[0], sh[1], block, sh[3]))?;
    astype(&zeros, template.dtype()?)
  }

  /// The fresh `(key, value)` zero blocks for the FIRST allocation (mlx-lm
  /// reset block with no prior buffer — the `k_shape`/`v_shape` in
  /// `update_and_fetch`). Both buffers share `keys`' batch + n_kv_heads —
  /// `k_shape = (B, n_kv_heads, block, k_head_dim)` and
  /// `v_shape = (B, n_kv_heads, block, v_head_dim)` with `B`/`n_kv_heads` from
  /// KEYS — while the key block takes keys' head_dim/dtype and the value block
  /// takes VALUES' head_dim/dtype. The `check_paired_kv` guard (run by `update`
  /// and `set_state` before this) already ensures keys and values agree on
  /// batch/n_kv_heads, so keys' and values' are equal here; deriving the value
  /// buffer from keys' is explicit (mirrors the reference `v_shape`) and
  /// defense-in-depth — it stays correct, never returning mismatched-batch/head
  /// K/V, even if that guard were weakened. Both args are already rank-checked
  /// 4-D.
  fn first_blocks(keys: &Array, values: &Array, block: usize) -> Result<(Array, Array)> {
    let k = keys.shape();
    let v = values.shape();
    let zk = astype(
      &Array::zeros::<f32>(&(k[0], k[1], block, k[3]))?,
      keys.dtype()?,
    )?;
    let zv = astype(
      &Array::zeros::<f32>(&(k[0], k[1], block, v[3]))?,
      values.dtype()?,
    )?;
    Ok((zk, zv))
  }

  /// Enforce the K/V pairing invariant and return the shared sequence length.
  /// `keys` and `values` are the K and V of one attention, so they must agree on
  /// batch (axis 0), n_kv_heads (axis 1), and sequence length (axis 2); head_dim
  /// (axis 3) and dtype may differ (`k_head_dim` vs `v_head_dim`). Both entry
  /// points that ingest a K/V pair — `update` (append) and `set_state` /
  /// `from_serialized` (restore) — call this before storing, so no consuming
  /// method ever sees an inconsistent pair (a shorter `values` would broadcast
  /// across positions or zero-pad on grow; a longer one would leak from
  /// `state`). DELIBERATELY diverges from mlx-lm, which assigns/broadcasts and
  /// relies on later errors — chosen to keep the cache sound. Each tensor is
  /// rank-checked (a deterministic `RankMismatch`) before the axis comparison.
  fn check_paired_kv(keys: &Array, values: &Array) -> Result<usize> {
    let sk = seq_len("keys", keys)?;
    let sv = seq_len("values", values)?;
    let (ksh, vsh) = (keys.shape(), values.shape());
    if ksh[0] != vsh[0] || ksh[1] != vsh[1] || sk != sv {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "StandardKvCache: keys/values pairing",
        "keys and values must be a paired K/V: equal batch, n_kv_heads, and sequence length",
      )));
    }
    Ok(sk)
  }

  /// `slice_update` `start`/`stop`/`strides` for the `[prev, new_offset)`
  /// sequence slot of the DESTINATION `buf` (`[B, n_kv_heads, L, head_dim]`),
  /// spanning every non-sequence axis in full. `B`/`n_kv_heads`/`head_dim` are
  /// read off `buf` (NOT the update tensor) so the slot is the whole buffer
  /// width — `slice_update` then broadcasts the update to fill it (mlx's
  /// `broadcast_to`), matching mlx-lm's `self.keys[..., prev:offset, :] = keys`:
  /// a narrower update (e.g. `B=1` into a `B=2` buffer) broadcasts over the full
  /// slot, and an incompatible width is a recoverable `slice_update` error — not
  /// a partial write that would leak stale (e.g. trimmed) buffer rows into the
  /// returned prefix.
  #[allow(clippy::type_complexity)]
  fn write_slot(
    buf: &Array,
    prev: usize,
    new_offset: usize,
  ) -> Result<([i32; 4], [i32; 4], [i32; 4])> {
    let sh = buf.shape();
    let to_i32 = |v: usize, what: &'static str| -> Result<i32> {
      i32::try_from(v).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          what,
          "must fit i32 for slice_update",
          format_smolstr!("{v}"),
        ))
      })
    };
    let (b, h, d) = (
      to_i32(sh[0], "StandardKvCache::update: buffer batch")?,
      to_i32(sh[1], "StandardKvCache::update: buffer n_kv_heads")?,
      to_i32(sh[3], "StandardKvCache::update: buffer head_dim")?,
    );
    let start = [0, 0, to_i32(prev, "StandardKvCache::update: offset")?, 0];
    let stop = [
      b,
      h,
      to_i32(new_offset, "StandardKvCache::update: offset + S")?,
      d,
    ];
    Ok((start, stop, [1, 1, 1, 1]))
  }
}

impl KvCache for StandardKvCache {
  /// The cached sequence length (mlx-lm `KVCache.offset` / `size()`).
  fn offset(&self) -> usize {
    self.offset
  }

  /// Append `keys`/`values` (`[B, n_kv_heads, S, head_dim]`) into the step
  /// buffer and return the live `[.., :offset, :]` prefix (mlx-lm
  /// `KVCache.update_and_fetch` / swift `KVCacheSimple.update`).
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    // `update` is one of the two entry points that ingest a K/V pair (the other
    // is `set_state`); enforce the same pairing invariant. Without it a
    // multi-token `keys` with a single-token `values` would broadcast the lone
    // value across all new positions (and a batch/head mismatch across batches)
    // — a shape-consistent but semantically corrupt cache the `set_state` guard
    // cannot catch after the fact. Rejected before any `self` mutation below.
    let s = Self::check_paired_kv(keys, values)?;
    let prev = self.offset;
    let new_offset = prev.checked_add(s).ok_or_else(|| {
      Error::OutOfRange(OutOfRangePayload::new(
        "StandardKvCache::update: offset + S",
        "must not overflow usize",
        format_smolstr!("offset={prev}, S={s}"),
      ))
    })?;
    let buf_len = match &self.keys {
      Some(k) => seq_len("keys", k)?,
      None => 0,
    };
    // Reallocate when empty or the write would overflow the buffer (mlx-lm
    // KVCache / swift KVCacheSimple `reset`).
    let reset = self.keys.is_none() || new_offset > buf_len;

    // Take the buffers so the in-place `slice_update` writes can DONATE (mlx
    // reuses an unshared input buffer in place — the O(S) write; `old_*` drops
    // at function exit on the success path, before the returned graph evals).
    // Restore `self` byte-identically on any error (stage-then-commit — the
    // transactional discipline shared with the other caches).
    let old_k = self.keys.take();
    let old_v = self.values.take();

    let staged: Result<(Array, Array, Array, Array)> = (|| {
      // Stage 1: the grown buffer as a fresh owned local (mlx-lm `reset` block:
      // trim a non-STEP-aligned prefix, concat a fresh `n_steps * STEP` zero
      // block; start from the block when empty). `None` ⇒ write into `old_*`.
      let grown: Option<(Array, Array)> = if reset {
        let n_steps = s.div_ceil(STEP); // (step + S - 1) // step (the refs)
        let block = n_steps.checked_mul(STEP).ok_or_else(|| {
          Error::OutOfRange(OutOfRangePayload::new(
            "StandardKvCache::update: n_steps * STEP",
            "must not overflow usize",
            format_smolstr!("n_steps={n_steps}, STEP={STEP}"),
          ))
        })?;
        match (&old_k, &old_v) {
          (Some(pk), Some(pv)) => {
            // Growing an EXISTING buffer: size the fresh zero block from the
            // buffer's own non-sequence axes (`pk`/`pv`), NOT the (possibly
            // narrower) incoming update — else a narrower broadcastable update
            // arriving at a STEP boundary (e.g. a B=1 decode step into a B=2
            // buffer) mismatches in `concat_seq` and is rejected, even though
            // the same update broadcasts fine inside existing capacity.
            // `slice_update` stays the single place that broadcasts the update
            // over the full grown slot.
            let zk = Self::make_zero_block(pk, block)?;
            let zv = Self::make_zero_block(pv, block)?;
            // `if prev % step != 0: keys = keys[..., :prev, :]` then concat; a
            // STEP-aligned (full) buffer concatenates whole.
            let (tk, tv) = if !prev.is_multiple_of(STEP) {
              (slice_seq(pk, 0, prev)?, slice_seq(pv, 0, prev)?)
            } else {
              (pk.try_clone()?, pv.try_clone()?)
            };
            Some((concat_seq(&tk, &zk)?, concat_seq(&tv, &zv)?))
          }
          // First allocation (no existing buffer): the KEY buffer takes the
          // incoming keys' non-sequence axes; the VALUE buffer shares those same
          // keys' batch/n_kv_heads (only its head_dim/dtype come from `values`,
          // see `first_blocks`), so a mismatched-width `values` broadcasts or is
          // rejected by `slice_update` against the key-shaped destination rather
          // than producing mismatched-batch/head K/V from the first write.
          _ => Some(Self::first_blocks(keys, values, block)?),
        }
      } else {
        None
      };

      // Stage 2: in-place write of the new tokens into the `[prev, new_offset)`
      // slot. The slot spans every non-sequence axis of the DESTINATION buffer
      // (`write_slot` reads `buf`'s shape), so `slice_update` broadcasts the
      // update to fill it — mlx-lm's `self.keys[..., prev:offset, :] = keys`.
      // unwrap: `!reset` ⇒ the `buf_len` read proved `self.keys`/`values` were
      // `Some`, so `old_*` are `Some`.
      let (buf_k, buf_v): (&Array, &Array) = match &grown {
        Some((gk, gv)) => (gk, gv),
        None => (old_k.as_ref().unwrap(), old_v.as_ref().unwrap()),
      };
      let (start_k, stop_k, strides) = Self::write_slot(buf_k, prev, new_offset)?;
      let (start_v, stop_v, _) = Self::write_slot(buf_v, prev, new_offset)?;
      let nk = slice_update(buf_k, keys, &start_k, &stop_k, &strides)?;
      let nv = slice_update(buf_v, values, &start_v, &stop_v, &strides)?;

      // Stage 3: the returned `[.., :new_offset, :]` prefix (mlx-lm returns the
      // sliced view; a fresh slice array here — bit-identical values).
      let rk = slice_seq(&nk, 0, new_offset)?;
      let rv = slice_seq(&nv, 0, new_offset)?;
      Ok((nk, nv, rk, rv))
    })();

    match staged {
      Ok((nk, nv, rk, rv)) => {
        self.keys = Some(nk);
        self.values = Some(nv);
        self.offset = new_offset;
        Ok((rk, rv))
      }
      Err(e) => {
        self.keys = old_k;
        self.values = old_v;
        Err(e)
      }
    }
  }

  /// mlx-lm / swift `state` getter: the logical `[.., :offset, :]` prefix of
  /// each stored tensor (drop the trailing step padding); `[]` when empty. So
  /// the serialized state is always the logical-length tensors, identical to the
  /// old append-and-fetch cache and to mlx-lm `KVCache.state`.
  ///
  /// The prefix is taken for `keys` and `values` INDEPENDENTLY, each against its
  /// own sequence length: a tensor exactly (or under) `offset` long has no step
  /// padding to drop and is cloned whole — the common no-padding fast path that
  /// keeps a contiguous full buffer — while a longer one is sliced. `set_state`
  /// enforces `keys.len == values.len` (the invariant guard), so for every
  /// reachable cache both tensors share their length and take the same branch;
  /// the per-tensor gating is defense-in-depth that cannot slice a shorter
  /// tensor out of range or leak a longer tensor's tail even if that guard were
  /// ever weakened.
  fn state(&self) -> Result<Vec<Array>> {
    // One stored tensor's logical prefix: clone whole when it has no padding
    // beyond `offset` (so the no-padding case stays a contiguous full buffer),
    // else slice off the trailing rows. Gating on each tensor's OWN length means
    // a shorter-than-`offset` tensor is returned whole (never sliced out of
    // range) and a longer one never leaks its tail.
    let prefix = |a: &Array, name: &'static str| -> Result<Array> {
      if self.offset >= seq_len(name, a)? {
        a.try_clone()
      } else {
        slice_seq(a, 0, self.offset)
      }
    };
    match (&self.keys, &self.values) {
      (Some(k), Some(v)) => Ok(vec![prefix(k, "keys")?, prefix(v, "values")?]),
      _ => Ok(Vec::new()),
    }
  }

  /// Force-evaluate the cache's own stored buffers in place — the per-chunk
  /// prefill memory barrier (see [`KvCache::materialize`]). Evals the genuine
  /// `self.keys`/`self.values` step buffers (the in-place `update` writes plus
  /// the trailing step padding) via the explicit `&mut` [`Array::eval`] — no
  /// `state()` clone, no slice. A no-op when empty.
  fn materialize(&mut self) -> Result<()> {
    if let Some(k) = self.keys.as_mut() {
      k.eval()?;
    }
    if let Some(v) = self.values.as_mut() {
      v.eval()?;
    }
    Ok(())
  }

  /// mlx-lm `KVCache.state` setter (cross-checked vs swift
  /// `KVCacheSimple.state`): `keys, values = v; offset = keys.shape[-2]`.
  /// An empty state resets to the fresh cache (`_BaseCache` "no state").
  fn set_state(&mut self, mut state: Vec<Array>) -> Result<()> {
    match state.len() {
      0 => {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        Ok(())
      }
      2 => {
        let values = state.pop().unwrap();
        let keys = state.pop().unwrap();
        // `set_state`/`from_serialized` is the only entry point that can RESTORE
        // an inconsistent pair (every state computed by `update` is consistent),
        // and an inconsistent pair corrupts downstream — a `values` shorter than
        // `keys` zero-pads on the next grow, a longer one leaks from `state()`.
        // Enforce the pairing invariant (shared with `update`) before any `self`
        // mutation, so a rejected restore leaves the cache byte-identical;
        // `offset` is the now-validated shared sequence length. mlx-lm assigns
        // with no cross-check — we deliberately diverge to keep the cache sound.
        let sk = Self::check_paired_kv(&keys, &values)?;
        self.keys = Some(keys);
        self.values = Some(values);
        self.offset = sk;
        Ok(())
      }
      n => Err(Error::OutOfRange(OutOfRangePayload::new(
        "StandardKvCache::set_state: state array count",
        "must be 0 or 2",
        format_smolstr!("{n}"),
      ))),
    }
  }

  fn is_trimmable(&self) -> bool {
    true
  }

  /// Drop the most recent `min(offset, n)` tokens; returns the number actually
  /// trimmed (mlx-lm / swift `trim`). Only `offset` is decremented — the buffer
  /// is reused, so a later [`update`](KvCache::update) overwrites the freed
  /// trailing slots in place (a non-`STEP`-aligned `offset` is re-aligned by
  /// `update`'s reset branch when the buffer next fills).
  fn trim(&mut self, n: usize) -> Result<usize> {
    let trimmed = n.min(self.offset);
    self.offset -= trimmed;
    Ok(trimmed)
  }

  /// mlx-lm `KVCache.make_mask` (`cache.py:393`):
  /// `create_attention_mask(*args, offset=self.offset, **kwargs)` — the
  /// caller's `window_size` is passed through unchanged (a full-attention
  /// cache never substitutes a window).
  fn make_mask(
    &self,
    n: usize,
    window_size: Option<usize>,
    return_array: bool,
  ) -> Result<MaskMode> {
    mask::create_attention_mask(n, self.offset(), return_array, window_size)
  }

  /// mlx-lm `KVCache.nbytes`: `keys.nbytes + values.nbytes` (0 if empty).
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

  /// Whether the cache holds no keys yet (mlx-lm `empty()`).
  fn is_empty(&self) -> bool {
    self.keys.is_none()
  }

  /// An independent copy (mlx-lm `copy.deepcopy` / swift `copy()`).
  /// Independence comes from MLX value semantics, not buffer duplication:
  /// arrays are immutable and this cache only ever *reassigns* `keys` /
  /// `values` to freshly-computed arrays (never mutates a buffer in place),
  /// so although `Array::try_clone` is a refcount-sharing clone, the copy
  /// and the original evolve completely independently.
  ///
  /// Swift's `copy()` is infallible; here the fallible [`Array::try_clone`]
  /// is propagated as a `Result` (`try_clone()?`) — a clone failure is
  /// **never** mapped to `None` (which would yield a cache with the right
  /// `offset` but missing keys/values: silent corruption) and **never**
  /// panicked.
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
      offset: self.offset,
    }))
  }

  /// `"KVCache"` — mlx-lm's `type(KVCache).__name__` (`cache.py:56`) /
  /// mlx-swift-lm `case is KVCacheSimple: return "KVCache"`
  /// (`KVCache.swift:1388`). Matches the trait default; overridden here
  /// explicitly so the kind label is co-located with the concrete cache
  /// (no inheritance of the generic `"KVCache"` fallback from the trait
  /// default — same pattern every other concrete cache follows).
  fn reference_class_name(&self) -> &'static str {
    "KVCache"
  }

  /// Per-layer fast-path downcast target (#110) — see the
  /// [`KvCache`]-trait doc's **Per-layer fast-path convention**.
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }

  /// Transactional override of [`KvCache::from_serialized`] — leaves
  /// `self` byte-identical to its pre-call state on every recoverable
  /// error. `StandardKvCache` has no meta (`meta_state() -> []` by
  /// default), so a caller passing non-empty `meta` here triggers the
  /// trait default `set_meta_state`'s rejection (mirrors mlx-lm
  /// `_BaseCache.meta_state` setter, `cache.py:142-145`). Without this
  /// override the default impl would call `set_state(state)?` first —
  /// mutating `self.keys`/`self.values`/`self.offset` to the new state —
  /// THEN error in `set_meta_state(meta)?`, leaving the cache holding
  /// the rejected serialized state. Stage on a fresh placeholder and
  /// commit only on success so the rollback contract holds for the
  /// most common cache kind too.
  #[allow(clippy::wrong_self_convention)] // see KvCache::from_serialized
  fn from_serialized(&mut self, state: Vec<Array>, meta: &[String]) -> Result<()> {
    let mut staged = StandardKvCache::new();
    staged.set_state(state)?;
    staged.set_meta_state(meta)?;
    *self = staged;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{lm::cache::KvCache, ops::shape::concatenate};

  /// `[1, 2, seq, 3]` f32 with distinct per-element values (`base + index`), so
  /// a value comparison catches a mis-placed token or a leaked zero pad.
  fn kv(seq: usize, base: f32) -> Array {
    let n = 2 * seq * 3;
    let data: Vec<f32> = (0..n).map(|i| base + i as f32).collect();
    Array::from_slice::<f32>(&data, &(1usize, 2, seq, 3)).unwrap()
  }

  /// Read a (possibly strided) array's values. The cache returns
  /// `keys[.., :offset, :]` slice / broadcast views, so reshape to a flat 1-D
  /// array — a non-contiguous reshape always copies — to get a contiguous
  /// buffer `to_vec` (which requires contiguity) can read.
  fn evaled(a: &Array) -> Vec<f32> {
    let n: usize = a.shape().iter().product();
    let mut c = crate::ops::shape::reshape(a, &(n,)).unwrap();
    c.eval().unwrap();
    c.to_vec::<f32>().unwrap()
  }

  /// `[2, 2, seq, 3]` f32 — batch 0 based at `base0`, batch 1 at `base1` — so a
  /// stale (un-broadcast) batch-1 write is detectable.
  fn b2_kv(seq: usize, base0: f32, base1: f32) -> Array {
    let per = 2 * seq * 3;
    let mut data: Vec<f32> = (0..per).map(|i| base0 + i as f32).collect();
    data.extend((0..per).map(|i| base1 + i as f32));
    Array::from_slice::<f32>(&data, &(2usize, 2, seq, 3)).unwrap()
  }

  /// The returned prefix equals `concatenate(all inputs so far, axis 2)`
  /// bit-for-bit — parity with mlx-lm `KVCache` / swift `KVCacheSimple` and the
  /// old append-and-fetch cache — including across the `STEP` realloc boundary.
  #[test]
  fn returned_prefix_equals_concatenated_inputs() {
    let mut c = StandardKvCache::new();
    let (mut ks, mut vs): (Vec<Array>, Vec<Array>) = (Vec::new(), Vec::new());
    for i in 0..(STEP + 8) {
      let k = kv(1, (i * 10) as f32);
      let v = kv(1, (i * 10 + 5000) as f32);
      let (rk, rv) = c.update(&k, &v).unwrap();
      ks.push(k);
      vs.push(v);
      assert_eq!(c.offset(), i + 1, "offset at step {i}");
      // Spot-check values at the start, across, and after the STEP boundary.
      if i < 2 || i == STEP - 1 || i == STEP || i == STEP + 7 {
        let ek = concatenate(&ks.iter().collect::<Vec<_>>(), 2).unwrap();
        let ev = concatenate(&vs.iter().collect::<Vec<_>>(), 2).unwrap();
        assert_eq!(evaled(&rk), evaled(&ek), "keys at step {i}");
        assert_eq!(evaled(&rv), evaled(&ev), "values at step {i}");
      }
    }
    // state() drops the step padding → exactly offset length.
    let st = c.state().unwrap();
    assert_eq!(st[0].shape(), vec![1, 2, STEP + 8, 3]);
    assert_eq!(st[1].shape(), vec![1, 2, STEP + 8, 3]);
  }

  /// Multi-token prefill (`S > STEP`) then single-token decode; the appended
  /// row equals the new token.
  #[test]
  fn prefill_then_decode() {
    let mut c = StandardKvCache::new();
    let (rk, _) = c.update(&kv(300, 0.0), &kv(300, 9000.0)).unwrap();
    assert_eq!(c.offset(), 300);
    assert_eq!(rk.shape(), vec![1, 2, 300, 3]);
    let (rk2, _) = c.update(&kv(1, 70_000.0), &kv(1, 80_000.0)).unwrap();
    assert_eq!(c.offset(), 301);
    assert_eq!(rk2.shape(), vec![1, 2, 301, 3]);
    assert_eq!(
      evaled(&slice_seq(&rk2, 300, 301).unwrap()),
      evaled(&kv(1, 70_000.0))
    );
  }

  /// `trim` decrements `offset` and reuses the buffer; the next `update`
  /// overwrites the freed slot (mlx-lm / swift `trim`).
  #[test]
  fn trim_reuses_buffer() {
    let mut c = StandardKvCache::new();
    for i in 0..10 {
      c.update(&kv(1, (i * 10) as f32), &kv(1, (i * 10 + 100) as f32))
        .unwrap();
    }
    assert_eq!(c.trim(3).unwrap(), 3);
    assert_eq!(c.offset(), 7);
    let (rk, _) = c.update(&kv(1, 42.0), &kv(1, 99.0)).unwrap();
    assert_eq!(c.offset(), 8);
    assert_eq!(rk.shape(), vec![1, 2, 8, 3]);
    assert_eq!(evaled(&slice_seq(&rk, 7, 8).unwrap()), evaled(&kv(1, 42.0)));
  }

  /// Codex round-1 (high): a narrower broadcastable update must fill the WHOLE
  /// destination slot, not just the leading sub-batch — otherwise stale (e.g.
  /// trimmed) buffer rows leak into the returned prefix. mlx-lm's
  /// `self.keys[..., prev:offset, :] = keys` broadcasts the RHS over the slot.
  #[test]
  fn narrow_update_broadcasts_over_full_buffer() {
    let mut c = StandardKvCache::new();
    // B=2 cache, 2 tokens, batch 1 distinct (100..) so a stale read is visible.
    c.update(&b2_kv(2, 0.0, 100.0), &b2_kv(2, 9000.0, 9100.0))
      .unwrap();
    c.trim(1).unwrap(); // offset 1; the B=2 buffer keeps its capacity
    // A B=1 update into the B=2 slot must broadcast, not partial-write batch 0.
    let k1 = kv(1, 42.0);
    let (rk, _) = c.update(&k1, &kv(1, 77.0)).unwrap();
    assert_eq!(c.offset(), 2);
    assert_eq!(rk.shape(), vec![2, 2, 2, 3], "B=2 preserved");
    // The appended row equals broadcast(k1) in BOTH batches — no stale batch 1.
    let appended = slice_seq(&rk, 1, 2).unwrap();
    let expected = crate::ops::shape::broadcast_to(&k1, &[2i32, 2, 1, 3]).unwrap();
    assert_eq!(
      evaled(&appended),
      evaled(&expected),
      "narrow update must broadcast over the full buffer slot, not leak stale rows"
    );
  }

  /// Codex round-2 (high): the round-1 fix made `write_slot` span the full
  /// destination buffer, but the GROW path still sized the fresh zero block
  /// from the incoming update — so a narrower broadcastable update arriving
  /// exactly at a STEP boundary (where the write must grow the buffer) hit a
  /// `concat_seq` batch mismatch and was rejected, even though the same update
  /// broadcasts fine inside existing capacity. Fill a B=2 cache to exactly STEP
  /// (offset == capacity), then a B=1 one-token update must still broadcast over
  /// the full B=2 slot rather than error at the boundary.
  #[test]
  fn narrow_update_at_step_boundary_broadcasts() {
    let mut c = StandardKvCache::new();
    // Fill to exactly STEP with a B=2 update → offset == buf_len == STEP, so the
    // next token forces a grow (reset). Batch 1 distinct (100..) for stale-read
    // visibility.
    c.update(&b2_kv(STEP, 0.0, 100.0), &b2_kv(STEP, 9000.0, 9100.0))
      .unwrap();
    assert_eq!(c.offset(), STEP);
    // A B=1 one-token update at the boundary must broadcast over the B=2 slot,
    // not be rejected by a B=2-vs-B=1 concat in the grow path.
    let k1 = kv(1, 42.0);
    let (rk, _) = c.update(&k1, &kv(1, 77.0)).unwrap();
    assert_eq!(c.offset(), STEP + 1);
    assert_eq!(rk.shape(), vec![2, 2, STEP + 1, 3], "B=2 preserved");
    let appended = slice_seq(&rk, STEP, STEP + 1).unwrap();
    let expected = crate::ops::shape::broadcast_to(&k1, &[2i32, 2, 1, 3]).unwrap();
    assert_eq!(
      evaled(&appended),
      evaled(&expected),
      "narrow update at the STEP boundary must broadcast over the full grown slot"
    );
  }

  /// Codex round-5 (high) + the invariant guard: `set_state` is the only entry
  /// point that can restore an inconsistent K/V pair (every state computed by
  /// `update` is consistent), and an inconsistent pair corrupts downstream — a
  /// shorter `values` is zero-padded into the next grow (R5), a longer one leaks
  /// rows from `state()` (R3). `set_state` rejects a pair that disagrees on
  /// batch, n_kv_heads, or sequence length (head_dim may differ), leaving the
  /// cache untouched — structurally preventing both corruption directions.
  #[test]
  fn set_state_rejects_mismatched_kv() {
    // Prime with a valid state so we can assert it is left untouched on reject.
    let mut c = StandardKvCache::new();
    c.update(&kv(2, 1.0), &kv(2, 2.0)).unwrap();

    // (a) values SHORTER than keys (the R5 zero-pad root).
    assert!(
      c.set_state(vec![kv(256, 0.0), kv(3, 100.0)]).is_err(),
      "values shorter than keys (seq mismatch) must be rejected"
    );
    // (b) values LONGER than keys (the R3 leak root).
    assert!(
      c.set_state(vec![kv(3, 0.0), kv(5, 100.0)]).is_err(),
      "values longer than keys (seq mismatch) must be rejected"
    );
    // (c) mismatched batch (axis 0), sequence equal.
    assert!(
      c.set_state(vec![b2_kv(2, 0.0, 1.0), kv(2, 100.0)]).is_err(),
      "mismatched batch must be rejected"
    );
    // Transactional: every rejected restore leaves the primed cache unchanged.
    assert_eq!(
      c.offset(),
      2,
      "rejected set_state leaves the cache unchanged"
    );

    // A consistent pair is accepted...
    assert!(c.set_state(vec![kv(4, 0.0), kv(4, 100.0)]).is_ok());
    assert_eq!(c.offset(), 4);
    // ...and head_dim (axis 3) may legitimately differ (k_head_dim vs v_head_dim).
    // Shapes (1,2,4,3) = 24 elems and (1,2,4,5) = 40 elems; head_dim 3 vs 5.
    let kd = Array::from_slice::<f32>(&[0.0f32; 24], &(1usize, 2, 4, 3)).unwrap();
    let vd = Array::from_slice::<f32>(&[1.0f32; 40], &(1usize, 2, 4, 5)).unwrap();
    assert!(
      c.set_state(vec![kd, vd]).is_ok(),
      "a consistent pair with differing head_dim (k vs v) is accepted"
    );
  }

  /// Codex round-6 (high): `update` is the other entry point that ingests a K/V
  /// pair, so it enforces the same pairing invariant as `set_state`. A
  /// multi-token `keys` with a single-token `values` would otherwise broadcast
  /// the lone value across all new positions, and a batch/head mismatch across
  /// batches — both shape-consistent but semantically corrupt and uncatchable by
  /// the `set_state` guard. `update` rejects a pair disagreeing on batch,
  /// n_kv_heads, or sequence length BEFORE any buffer mutation; a consistent pair
  /// (including one narrower than the buffer, which broadcasts into it) appends.
  #[test]
  fn update_rejects_mismatched_kv() {
    let mut c = StandardKvCache::new();
    c.update(&kv(2, 1.0), &kv(2, 2.0)).unwrap(); // prime with a consistent pair

    // (a) seq mismatch: 4 keys, 1 value — the broadcast-across-positions root.
    assert!(
      c.update(&kv(4, 0.0), &kv(1, 9.0)).is_err(),
      "multi-token keys with single-token values must be rejected"
    );
    // (b) batch mismatch: keys B=2, values B=1.
    assert!(
      c.update(&b2_kv(1, 0.0, 1.0), &kv(1, 9.0)).is_err(),
      "keys B=2 with values B=1 must be rejected"
    );
    // (c) inverse batch mismatch: keys B=1, values B=2.
    assert!(
      c.update(&kv(1, 0.0), &b2_kv(1, 9.0, 10.0)).is_err(),
      "keys B=1 with values B=2 must be rejected"
    );
    // Transactional: every rejected update leaves the primed cache unchanged.
    assert_eq!(c.offset(), 2, "rejected update leaves the cache unchanged");

    // A consistent pair appends normally.
    let (rk, _) = c.update(&kv(1, 5.0), &kv(1, 6.0)).unwrap();
    assert_eq!(c.offset(), 3);
    assert_eq!(rk.shape(), vec![1, 2, 3, 3]);
  }

  /// Donation check: the per-token in-place write stays O(S) (flat) as the
  /// cache grows — not O(N), which concat-every-step would be. `materialize`
  /// forces the write each step without reading the growing prefix.
  #[test]
  #[ignore = "perf: step-buffer write is O(S) via slice_update donation"]
  fn update_write_is_flat() {
    use std::time::Instant;
    let bench = |target: usize| -> f64 {
      let mut c = StandardKvCache::new();
      for i in 0..target {
        c.update(&kv(1, i as f32), &kv(1, (target + i) as f32))
          .unwrap();
        c.materialize().unwrap();
      }
      let inputs: Vec<(Array, Array)> = (0..64)
        .map(|i| {
          let mut k = kv(1, (2 * target + i) as f32);
          let mut v = kv(1, (3 * target + i) as f32);
          k.eval().unwrap();
          v.eval().unwrap();
          (k, v)
        })
        .collect();
      let mut ts = Vec::new();
      for (k, v) in &inputs {
        let t = Instant::now();
        c.update(k, v).unwrap();
        c.materialize().unwrap();
        ts.push(t.elapsed().as_secs_f64() * 1e6);
      }
      ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
      ts[0]
    };
    let short = bench(256);
    let long = bench(8192);
    eprintln!(
      "step-buffer write: len~256 {short:.2}us | len~8192 {long:.2}us | {:.2}x",
      long / short
    );
    assert!(
      long < short * 4.0,
      "per-token write grew {:.1}x — donation not working?",
      long / short
    );
  }
}
