//! [`ChunkedKvCache`] — the bounded-buffer cache for chunked prefill.

use crate::{
  array::Array,
  error::{Error, Result},
  lm::cache::{
    KvCache, MaskMode, mask,
    util::{KV_NDIM, concat_seq, nbytes, seq_len, seq_slice},
  },
  ops,
};

/// Faithful 1:1 port of `mlx_lm.models.cache.ChunkedKVCache`
/// (`cache.py:731-813`, `step = 256`), cross-checked against mlx-swift-lm
/// `ChunkedKVCache: KVCacheSimple` (`KVCache.swift:1008`).
///
/// Used by the llama4 model for chunked-attention prefill: between
/// chunks the model calls [`maybe_trim_front`](ChunkedKvCache::maybe_trim_front)
/// to keep the physical buffer at most `chunk_size` rows while a separate
/// `start_position` tracks how many leading tokens were dropped, so the
/// *raw* [`offset`](KvCache::offset) (the position the RoPE / chunk mask
/// use) keeps counting uncapped while `prev = offset - start_position`
/// indexes into the trimmed buffer.
///
/// Like [`RotatingKvCache`](super::RotatingKvCache), mlx-lm overwrites buffer
/// rows *in place* (`self.keys[..., prev:end, :] = keys`, `cache.py:769`),
/// and the returned slice `keys[..., :end, :]` depends on the physical
/// buffer layout, so this is a literal port — including mlx-lm's `step`-sized
/// zero-buffer over-allocation. `mlxrs::Array` is functional (no in-place
/// buffer slicing), so the in-place write is emulated by splicing the new
/// rows over `[prev, end)` of the buffer via [`crate::ops`]
/// concatenate/slice; every grown placeholder row is provably overwritten by
/// that splice (the write spans exactly `[prev, end)` with `end - prev == S`)
/// **or** sliced off by the `keys[..., :end, :]` return / the `state`
/// `keys[..., :offset, :]` slice before any observer, so `keys.shape[2]`
/// (the buffer length — which drives the realloc and
/// [`maybe_trim_front`](ChunkedKvCache::maybe_trim_front) branches and is
/// **not** the logical length) stays byte-for-byte identical to mlx-lm for
/// every `S` and trim mix.
///
/// `chunk_size` is `Option<usize>`: mlx-lm always constructs it with an int
/// (`ChunkedKVCache(chunk_size)`, `cache.py:734`), but mlx-swift-lm
/// (`KVCache.swift:1009-1015,1082-1098`) models it as an optional and a
/// reconstructed cache may carry the literal `"None"` — honored here so a
/// Swift-produced prompt cache round-trips and `maybe_trim_front` short-
/// circuits exactly as `KVCache.swift:1019-1021`.
///
/// No implicit eval: every op is a pure [`crate::ops`] composition returning
/// a `Result`; no recoverable path panics.
pub struct ChunkedKvCache {
  keys: Option<Array>,
  values: Option<Array>,
  /// Raw total tokens ever appended (monotone except via
  /// [`trim`](KvCache::trim)) — mlx-lm `ChunkedKVCache.offset`. This is the
  /// uncapped position the RoPE / chunk mask use, **not** a buffer length.
  offset: usize,
  /// The configured chunk size — mlx-lm `ChunkedKVCache.chunk_size`
  /// (`None` only via a Swift-reconstructed cache; see the type docs).
  chunk_size: Option<usize>,
  /// How many leading tokens [`maybe_trim_front`](
  /// ChunkedKvCache::maybe_trim_front) has dropped from the front of the
  /// physical buffer — mlx-lm `ChunkedKVCache.start_position`. `prev =
  /// offset - start_position` is the buffer-relative write cursor.
  start_position: usize,
}

/// mlx-lm `ChunkedKVCache.step` (`cache.py:732`) — how many rows the buffer
/// is over-allocated by. Purely an allocation batch size: every grown row is
/// provably overwritten by the `[prev, end)` splice or sliced off by the
/// `[..., :end, :]` return / `state` `[..., :offset, :]` slice before any
/// observer, so its value never reaches one; we mirror `256` so the
/// buffer-length bookkeeping (`keys.shape[2]`, which *does* drive the
/// realloc / `maybe_trim_front` branches) is byte-for-byte the same.
const CHUNKED_STEP: usize = 256;

impl ChunkedKvCache {
  /// A new, empty chunked cache with the given `chunk_size` — mlx-lm
  /// `ChunkedKVCache(chunk_size)` (`cache.py:734`). `None` mirrors a
  /// Swift-reconstructed optional `chunkSize` (`KVCache.swift:1012-1015`).
  pub fn new(chunk_size: Option<usize>) -> Self {
    Self {
      keys: None,
      values: None,
      offset: 0,
      chunk_size,
      start_position: 0,
    }
  }

  /// mlx-lm `ChunkedKVCache.maybe_trim_front` (`cache.py:741-746`):
  ///
  /// ```text
  /// if self.keys is not None and self.keys.shape[2] >= self.chunk_size:
  ///     self.start_position += self.keys.shape[2] - self.chunk_size
  ///     self.keys   = self.keys[..., -self.chunk_size:, :]
  ///     self.values = self.values[..., -self.chunk_size:, :]
  /// ```
  ///
  /// Keeps the **last** `chunk_size` *buffer* rows (`keys.shape[2]` is the
  /// physical buffer length, **not** the logical length) and adds the number
  /// of dropped rows to `start_position`; `offset` is **not** touched. With
  /// a `None` `chunk_size` it is unconditionally a no-op (mlx-swift-lm
  /// `KVCache.swift:1019-1021` `guard let chunkSize`). The llama4 model is
  /// responsible for only invoking this when the buffer is logically full
  /// (porting per-model arch is out of scope); the method itself reproduces
  /// the source's exact semantics.
  pub fn maybe_trim_front(&mut self) -> Result<()> {
    let chunk_size = match self.chunk_size {
      Some(c) => c,
      // Swift `guard let chunkSize else { return }` — no-op.
      None => return Ok(()),
    };
    let (k, v) = match (&self.keys, &self.values) {
      (Some(k), Some(v)) => (k, v),
      // `self.keys is not None` — nothing to trim.
      _ => return Ok(()),
    };
    let buf_len = seq_len("keys", k)?;
    // Rank-check `values` AND capture its OWN sequence length: mlx-lm's
    // `self.values = self.values[..., -chunk_size:, :]` (`cache.py:746`) is a
    // negative slice relative to the *values* tensor's own sequence axis, NOT
    // the keys length. This implementation intentionally accepts rank-valid
    // seq-*mismatched* restored K/V (the project's no-K/V-cross-validation
    // policy — #32), so `values` must be trimmed by its own last
    // `chunk_size` window independently; reusing the keys-derived window here
    // would silently keep the wrong `values` rows whenever the restored K/V
    // lengths differ. This is per-tensor faithfulness, NOT a K/V *cross-
    // comparison* (we never compare or require `keys.len == values.len`).
    let v_len = seq_len("values", v)?;
    if buf_len >= chunk_size {
      // `self.start_position += self.keys.shape[2] - self.chunk_size`
      // (`cache.py:744`). `buf_len >= chunk_size` here, so the subtraction
      // never underflows; a hostile restored `start_position` near
      // `usize::MAX` would wrap (release) / panic (debug) on the `+=`, so it
      // is a checked add (byte-identical for every non-overflowing input).
      let added = buf_len - chunk_size;
      let new_start =
        self
          .start_position
          .checked_add(added)
          .ok_or_else(|| Error::ShapeMismatch {
            message: format!(
              "ChunkedKvCache::maybe_trim_front: start_position ({}) + {added} overflows usize",
              self.start_position
            ),
          })?;
      // `self.keys = self.keys[..., -self.chunk_size:, :]` — keys' OWN last
      // `chunk_size` rows (`buf_len >= chunk_size`, so `start` is valid).
      // `self.values = self.values[..., -self.chunk_size:, :]` — VALUES' own
      // last `chunk_size` rows, computed from `v_len` independently (Python's
      // `[..., -chunk_size:, :]` on a `values` shorter than `chunk_size`
      // clamps the negative start to 0 — `saturating_sub` reproduces that
      // exactly; `seq_slice` then clamps `end` per-tensor). Python treats
      // `-0` as `0`, so `[..., -0:, :]` is the WHOLE tensor (not empty):
      // mirror that for `chunk_size == 0` by setting the slice start to 0
      // for both tensors — a faithful no-op trim that matches mlx-lm's
      // `keys[..., -0:, :] == keys` (even though `chunk_size == 0` is a
      // degenerate case mlx-lm itself doesn't guard against). Compute BOTH
      // slices into locals before mutating any `self` field so a failure on
      // the `values` slice cannot leave a split-brain cache (bumped
      // `start_position` / trimmed `keys` but stale `values`); the final
      // state is byte-identical to mlx-lm's three in-place assignments for
      // every restored state, including a seq-mismatched (yet rank-valid) one.
      let k_start = if chunk_size == 0 {
        0
      } else {
        buf_len - chunk_size
      };
      let v_start = if chunk_size == 0 {
        0
      } else {
        v_len.saturating_sub(chunk_size)
      };
      let new_keys = seq_slice(k, k_start, buf_len)?;
      let new_values = seq_slice(v, v_start, v_len)?;
      self.start_position = new_start;
      self.keys = Some(new_keys);
      self.values = Some(new_values);
    }
    Ok(())
  }

  /// Emulate mlx-lm's in-place `buf[..., a:a+s, :] = new` on an immutable
  /// `Array`: splice `new` over `[a, a+s)`, keeping the surrounding rows
  /// (identical idiom to [`RotatingKvCache`](super::RotatingKvCache)'s
  /// `set_seq`). `a + s` is checked so a corrupt restored `offset` /
  /// `start_position` cannot wrap the tail bound.
  ///
  /// `name` identifies the target buffer (`"keys"` / `"values"`) for the
  /// per-target bounds error.
  ///
  /// The write window `[a, a+s)` MUST lie inside the target buffer's own
  /// sequence axis: mlx-lm's `self.<buf>[..., a:a+s, :] = new` raises an
  /// IndexError if the slice extends past the buffer, NOT silently
  /// truncating. `seq_slice` clamps Python-style — fine for *reads* (the
  /// `[..., :end, :]` returns), but for a *write* it would silently swallow
  /// an out-of-bounds splice, dropping or extending rows and corrupting the
  /// cache. So check `end <= l` up front (per-target, no K/V cross-check)
  /// and surface a recoverable `ShapeMismatch`; the splice is then performed
  /// only on a provably-in-bounds window (every `head`/`tail`/`concat`
  /// produces an array of exactly the buffer length).
  fn set_seq(name: &str, buf: &Array, a: usize, s: usize, new: &Array) -> Result<Array> {
    let l = seq_len(name, buf)?;
    let end = a.checked_add(s).ok_or_else(|| Error::ShapeMismatch {
      message: format!("ChunkedKvCache: {name} write start ({a}) + S ({s}) overflows usize"),
    })?;
    if end > l {
      return Err(Error::ShapeMismatch {
        message: format!(
          "ChunkedKvCache: {name} write window [{a}, {end}) extends past buffer length {l}"
        ),
      });
    }
    // Structural class-kill (closes #78 P1 iter5): mlx-lm's
    // `self.<buf>[..., a:a+s, :] = new` slice-assignment routes through
    // `slice_update`, which broadcasts the RHS to the slice shape (`mlx/
    // ops.cpp:843` — `broadcast_to(update, upd_shape)`). Our write-emulation
    // (`concat_parts([head, new, tail])`) has a full-window shortcut that
    // returns `new` after only a rank check, bypassing both the
    // non-broadcastable-axes validation AND the size-1 broadcast itself
    // (e.g. a `[2, .., .., ..]` buffer with a `[1, .., .., ..]` `new` is
    // valid in mlx-lm — broadcast up to keep the buffer shape — but the
    // shortcut would silently SHRINK the buffer's batch axis). Route every
    // `set_seq` window — partial or full — through `broadcast_write_rhs`,
    // which builds the slice shape `[buf[0], buf[1], end-a, buf[3]]` and
    // broadcasts `new` to it exactly as mlx's `slice_update` does (single
    // helper, single tensor — NOT the fenced K/V cross-check). Identity
    // broadcasts are no-ops; size-1 broadcasts expand; non-broadcastable
    // axes are a recoverable `Err(ShapeMismatch)`. Faithful to mlx-lm for
    // every input shape.
    let new = super::util::broadcast_write_rhs(name, buf, a, end, new)?;
    let head = seq_slice(buf, 0, a)?;
    let tail = seq_slice(buf, end, l)?;
    super::util::concat_parts(&[&head, &new, &tail])
  }
}

impl KvCache for ChunkedKvCache {
  /// The raw total tokens ever appended — mlx-lm `ChunkedKVCache.offset`
  /// (the uncapped position the RoPE / chunk mask use; consistent with
  /// [`StandardKvCache`](super::StandardKvCache) /
  /// [`RotatingKvCache`](super::RotatingKvCache)).
  fn offset(&self) -> usize {
    self.offset
  }

  /// Append `keys`/`values` and return `keys[..., :end, :]` —
  /// mlx-lm `ChunkedKVCache.update_and_fetch` (`cache.py:748-771`),
  /// cross-checked vs mlx-swift-lm `KVCache.swift:1028-1062`:
  ///
  /// ```text
  /// prev = self.offset - self.start_position
  /// if self.keys is None or (prev + S) > self.keys.shape[2]:
  ///     n_steps = (step + S - 1) // step
  ///     new_k/new_v = zeros((B, n_kv_heads, n_steps*step, *_head_dim))
  ///     if self.keys is not None:
  ///         if prev % step != 0:
  ///             self.keys/values = self.keys/values[..., :prev, :]
  ///         self.keys/values = concat([self.keys/values, new_*], axis=2)
  ///     else:
  ///         self.keys, self.values = new_k, new_v
  /// self.offset += S
  /// end = self.offset - self.start_position
  /// self.keys[..., prev:end, :]   = keys
  /// self.values[..., prev:end, :] = values
  /// return self.keys[..., :end, :], self.values[..., :end, :]
  /// ```
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    // Rank-check both, then read every dim through the validated shape (no
    // raw `.shape()[N]` on a not-rank-validated tensor).
    let s = seq_len("keys", keys)?;
    // Rank-check `values` too (mirrors `keys`); mlx-lm reads
    // `values.shape[3]` (`cache.py:752`) and writes `values[..., prev:end,
    // :]` without cross-checking the K/V *sequence* lengths — we keep that
    // (no `vs == s` validation), only forcing the same 4-D rank guard so a
    // misuse is a recoverable error, never a raw-index panic.
    let _vs = seq_len("values", values)?;
    let ks = keys.shape();
    let vshape = values.shape();
    let (b, n_kv_heads, k_head_dim) = (ks[0], ks[1], ks[KV_NDIM - 1]);
    let v_head_dim = vshape[KV_NDIM - 1];

    // `prev = self.offset - self.start_position` (`cache.py:749`). Python
    // ints never go negative; a corrupt restored `start_position > offset`
    // would wrap (release) / panic (debug). Surface it as a recoverable
    // error (byte-identical `offset - start_position` for every valid
    // input where `start_position <= offset`, which all faithful traces
    // satisfy — `maybe_trim_front`/`trim` only ever keep it `<= offset`).
    let prev = self
      .offset
      .checked_sub(self.start_position)
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "ChunkedKvCache::update: start_position ({}) exceeds offset ({})",
          self.start_position, self.offset
        ),
      })?;

    let cur_buf = match &self.keys {
      Some(k) => Some(seq_len("keys", k)?),
      None => None,
    };
    // `self.keys is None or (prev + S) > self.keys.shape[2]` (`cache.py:750`).
    let prev_plus_s = prev.checked_add(s).ok_or_else(|| Error::ShapeMismatch {
      message: format!("ChunkedKvCache::update: prev ({prev}) + S ({s}) overflows usize"),
    })?;
    let need_alloc = match cur_buf {
      None => true,
      Some(buf_len) => prev_plus_s > buf_len,
    };
    // mlx-lm assigns `self.keys`/`self.values`/`self.offset` field-by-field
    // (`cache.py:760-770`) on mutable arrays; on `mlxrs::Array` (functional)
    // the realloc concat + the `[prev, end)` splice are all fallible
    // (backend/OOM). Assigning `self.keys` before the `self.values` concat /
    // the splices succeed would, on a recoverable `Err` from a *later* op,
    // leave a poisoned cache (grown `keys` buffer, stale `values`/`offset`);
    // a subsequent `maybe_trim_front` reads the grown `keys.shape[2]` as
    // authoritative and would trim `values` against a stale length. So the
    // whole `update` is transactional: every post-realloc + post-splice
    // tensor is computed into a local and the three `self` fields are
    // assigned only after ALL fallible work has succeeded (the exact
    // compute-locals-then-assign discipline `maybe_trim_front` uses;
    // byte-identical to mlx-lm for every success path).
    let (mut buf_k, mut buf_v) = match (&self.keys, &self.values) {
      (Some(k), Some(v)) => (k.try_clone()?, v.try_clone()?),
      // `self.keys is None`: no prior buffer (the realloc `else` arm below
      // installs `new_k`/`new_v`). A half-initialized (only one of keys /
      // values set) cache is unreachable via the public API; treat it as the
      // empty case rather than panicking.
      _ => {
        // Placeholder; overwritten unconditionally when `need_alloc`, and
        // `need_alloc` is always true here (`cur_buf == None`).
        (keys.try_clone()?, values.try_clone()?)
      }
    };
    if need_alloc {
      // `n_steps = (step + S - 1) // step` (`cache.py:753`); `S >= 1` for a
      // real KV update, but guard the subtraction for an empty-seq input
      // (saturating is exact for every `S >= 1`, mlx-lm's domain).
      let n_steps = (CHUNKED_STEP + s).saturating_sub(1) / CHUNKED_STEP;
      let total = n_steps
        .checked_mul(CHUNKED_STEP)
        .ok_or_else(|| Error::ShapeMismatch {
          message: format!("ChunkedKvCache::update: n_steps ({n_steps}) * step overflows usize"),
        })?;
      // `mx.zeros(shape, keys.dtype)` (`cache.py:756-757`): build f32 zeros
      // then cast (mirrors `RotatingKvCache`; `Array::zeros` is f32-only).
      let new_k = ops::misc::astype(
        &Array::zeros::<f32>(&(b, n_kv_heads, total, k_head_dim))?,
        keys.dtype()?,
      )?;
      let new_v = ops::misc::astype(
        &Array::zeros::<f32>(&(b, n_kv_heads, total, v_head_dim))?,
        values.dtype()?,
      )?;
      match (&self.keys, &self.values) {
        (Some(pk), Some(pv)) => {
          // `if prev % step != 0: self.keys = self.keys[..., :prev, :]`
          // (`cache.py:759-761`) — drop the partially-filled tail before
          // concatenating the fresh zero block.
          let (bk, bv) = if prev % CHUNKED_STEP != 0 {
            (seq_slice(pk, 0, prev)?, seq_slice(pv, 0, prev)?)
          } else {
            (pk.try_clone()?, pv.try_clone()?)
          };
          // `self.keys = mx.concatenate([self.keys, new_k], axis=2)`
          // (`cache.py:762-763`) — into locals, NOT `self` (see the
          // transactional note above).
          buf_k = concat_seq(&bk, &new_k)?;
          buf_v = concat_seq(&bv, &new_v)?;
        }
        // `else: self.keys, self.values = new_k, new_v` (`cache.py:765`).
        _ => {
          buf_k = new_k;
          buf_v = new_v;
        }
      }
    }

    // `self.offset += S` (`cache.py:767`). Python ints never overflow; a
    // corrupt restored `offset` near `usize::MAX` would wrap (release) /
    // panic (debug). Compute the post-update offset checked BEFORE the
    // in-place splice so the overflow path performs NO partial mutation
    // (byte-identical to `self.offset + S` for every non-overflowing input).
    let new_offset = self
      .offset
      .checked_add(s)
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "ChunkedKvCache::update: offset ({}) + S ({s}) overflows usize",
          self.offset
        ),
      })?;
    // `end = self.offset - self.start_position` AFTER `offset += S`
    // (`cache.py:768`). `new_offset >= self.offset >= start_position` (the
    // `prev` checked-sub above already established `offset >=
    // start_position`), so this never underflows.
    let end = new_offset
      .checked_sub(self.start_position)
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "ChunkedKvCache::update: start_position ({}) exceeds new offset ({new_offset})",
          self.start_position
        ),
      })?;
    // `self.keys[..., prev:end, :] = keys` (`cache.py:769-770`). `end - prev
    // == S` by construction, so the splice overwrites exactly the `S` new
    // rows and the `values` write uses the SAME `[prev, end)` window —
    // mirror mlx-lm and let a K/V seq-length disagreement surface from the
    // op, no extra cross-validation (faithful: cache.py does not cross-check
    // either). Splice the post-realloc *locals* (`buf_k`/`buf_v`), still NOT
    // `self`, so a failing splice cannot poison the cache.
    let nk = Self::set_seq("keys", &buf_k, prev, s, keys)?;
    let nv = Self::set_seq("values", &buf_v, prev, s, values)?;

    // `return self.keys[..., :end, :], self.values[..., :end, :]`
    // (`cache.py:771`). `seq_slice` clamps `end` to the buffer length with
    // Python/NumPy slicing semantics (a `maybe_trim_front`-shrunk buffer can
    // make `end > buf_len`; mlx-lm's `[..., :end, :]` clamps identically).
    // Compute the returned slices BEFORE any `self` mutation too: this is the
    // last fallible step, so the three `self` fields are committed only once
    // EVERY fallible op (realloc concat, splice, return slice) has succeeded
    // — the cache is byte-identically updated on success and completely
    // untouched on any recoverable `Err`.
    let ret_k = seq_slice(&nk, 0, end)?;
    let ret_v = seq_slice(&nv, 0, end)?;
    self.offset = new_offset;
    self.keys = Some(nk);
    self.values = Some(nv);
    Ok((ret_k, ret_v))
  }

  /// mlx-lm `ChunkedKVCache.state` getter (`cache.py:773-781`):
  ///
  /// ```text
  /// if self.offset == self.keys.shape[2]: return self.keys, self.values
  /// else: return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
  /// ```
  ///
  /// Note the slice bound is the raw `self.offset` (**not** `offset -
  /// start_position`); after `maybe_trim_front` the buffer is shorter than
  /// `offset`, and mlx-lm's `[..., :self.offset, :]` clamps to the whole
  /// (trimmed) buffer — `seq_slice` reproduces that clamp exactly. `[]`
  /// when empty (mlx-lm would `AttributeError` on `self.keys.shape` for a
  /// `None`/empty cache via `_BaseCache`-style serialization, so an empty
  /// cache serializing to no state is the faithful, non-panicking choice and
  /// matches every other cache in this module).
  fn state(&self) -> Result<Vec<Array>> {
    match (&self.keys, &self.values) {
      (Some(k), Some(v)) => {
        let buf_len = seq_len("keys", k)?;
        // Rank-check `values` too before the `else` arm slices it
        // (`seq_slice` reads `shape[-2]` raw); a recoverable `Error` for an
        // unusable restored tensor, never a raw-shape-index panic.
        seq_len("values", v)?;
        if self.offset == buf_len {
          Ok(vec![k.try_clone()?, v.try_clone()?])
        } else {
          Ok(vec![
            seq_slice(k, 0, self.offset)?,
            seq_slice(v, 0, self.offset)?,
          ])
        }
      }
      _ => Ok(Vec::new()),
    }
  }

  /// mlx-lm `ChunkedKVCache.state` setter (`cache.py:783-786`):
  /// `self.keys, self.values = v; self.offset = self.keys.shape[2]`.
  ///
  /// Unlike `_BaseCache` / [`StandardKvCache`](super::StandardKvCache),
  /// `ChunkedKVCache` defines its OWN setter that unpacks `self.keys,
  /// self.values = v` — an empty `v` raises (cannot unpack `[]`), so an
  /// empty state is **invalid** here (a recoverable [`Error::Backend`], not
  /// a silent reset); `start_position` is **not** restored by the state
  /// setter (it comes from [`set_meta_state`](KvCache::set_meta_state),
  /// matching `_BaseCache.from_state`'s state-then-meta order).
  ///
  /// No K/V *shape* cross-validation — mlx-lm only reads `keys.shape[2]`
  /// (`cache.py:786`) and never checks the keys/values *sequence lengths*
  /// agree; `offset` is derived from `keys`' own sequence axis, faithfully
  /// (we do **not** add a `values.seq == keys.seq` check mlx-lm lacks).
  /// Both tensors are still independently *rank*-checked (4-D `[B,
  /// n_kv_heads, S, head_dim]`), exactly as [`update`](KvCache::update)
  /// rank-checks both: without it a malformed/hostile restored prompt
  /// cache (rank-valid `keys`, rank-invalid `values`) would pass `from_state`
  /// and then **panic** in a later
  /// [`maybe_trim_front`](ChunkedKvCache::maybe_trim_front) / [`state`](
  /// KvCache::state) where `values` is sliced (`seq_slice` reads
  /// `shape[-2]`). Rank ≠ shape, so this is a recoverable [`Error`] for an
  /// unusable tensor — not the faithful-forbidden cross-check — keeping the
  /// "never raw-index `.shape()[N]` on a not-rank-validated tensor" rule.
  fn set_state(&mut self, mut state: Vec<Array>) -> Result<()> {
    match state.len() {
      2 => {
        let values = state.pop().unwrap();
        let keys = state.pop().unwrap();
        let sk = seq_len("keys", &keys)?;
        // Per-tensor rank guard on `values` too (NOT a K/V seq cross-check):
        // a rank-invalid `values` here would otherwise surface only as a
        // raw-shape-index panic when `maybe_trim_front`/`state` slices it.
        seq_len("values", &values)?;
        self.keys = Some(keys);
        self.values = Some(values);
        self.offset = sk;
        Ok(())
      }
      n => Err(Error::Backend {
        message: format!(
          "ChunkedKvCache state must have exactly 2 arrays (its setter unpacks keys, values = v; empty/other is invalid), got {n}"
        ),
      }),
    }
  }

  /// mlx-lm `ChunkedKVCache.meta_state` getter (`cache.py:796-798`):
  /// `tuple(map(str, (self.chunk_size, self.start_position)))`. A `None`
  /// `chunk_size` is the literal `"None"` (mlx-swift-lm
  /// `KVCache.swift:1084`).
  fn meta_state(&self) -> Vec<String> {
    let chunk = match self.chunk_size {
      Some(c) => c.to_string(),
      None => "None".to_string(),
    };
    vec![chunk, self.start_position.to_string()]
  }

  /// mlx-lm `ChunkedKVCache.meta_state` setter (`cache.py:800-802`):
  /// `self.chunk_size, self.start_position = map(int, v)`. mlx-swift-lm
  /// (`KVCache.swift:1087-1097`) requires exactly 2 values (`fatalError`
  /// otherwise — here a recoverable [`Error::Backend`]) and decodes a
  /// literal `"None"` `chunk_size` back to `None`. Both values are parsed
  /// into locals and validated before mutating ANY field, so a parse error
  /// on `start_position` leaves the cache exactly as it was (faithful:
  /// same two fields, same order/format as `cache.py:800-802`).
  fn set_meta_state(&mut self, m: &[String]) -> Result<()> {
    if m.len() != 2 {
      return Err(Error::Backend {
        message: format!(
          "ChunkedKvCache meta_state must have 2 values, got {}",
          m.len()
        ),
      });
    }
    // Swift `if newValue[0] == "None" { chunkSize = nil } else { Int(...) }`.
    let chunk_size = if m[0] == "None" {
      None
    } else {
      Some(m[0].parse::<usize>().map_err(|e| Error::Backend {
        message: format!("ChunkedKvCache meta_state chunk_size ({:?}): {e}", m[0]),
      })?)
    };
    let start_position = m[1].parse::<usize>().map_err(|e| Error::Backend {
      message: format!("ChunkedKvCache meta_state start_position ({:?}): {e}", m[1]),
    })?;
    self.chunk_size = chunk_size;
    self.start_position = start_position;
    Ok(())
  }

  /// mlx-lm `ChunkedKVCache.is_trimmable` (`cache.py:788-789`): always
  /// `True`.
  fn is_trimmable(&self) -> bool {
    true
  }

  /// mlx-lm `ChunkedKVCache.trim` (`cache.py:791-794`):
  /// `n = min(self.offset - self.start_position, n); self.offset -= n;
  /// return n`. Only `offset` is adjusted (`start_position` / the buffer are
  /// untouched — the next [`update`](KvCache::update) recomputes `prev`).
  fn trim(&mut self, n: usize) -> Result<usize> {
    // `self.offset - self.start_position`: a corrupt restored
    // `start_position > offset` would wrap (release) / panic (debug);
    // surface it (byte-identical to mlx-lm's subtraction for every faithful
    // input where `start_position <= offset`).
    let span = self
      .offset
      .checked_sub(self.start_position)
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "ChunkedKvCache::trim: start_position ({}) exceeds offset ({})",
          self.start_position, self.offset
        ),
      })?;
    let trimmed = span.min(n);
    self.offset -= trimmed;
    Ok(trimmed)
  }

  /// `ChunkedKVCache` defines **no** `make_mask`, and — unlike `KVCache`
  /// (`cache.py:219`) — neither does its base `_BaseCache`
  /// (`cache.py:127-175` has no `make_mask`). So mlx-lm
  /// `base.create_attention_mask` (`base.py:49`) finds
  /// `hasattr(cache, "make_mask")` **False** for a `ChunkedKVCache` and
  /// falls through to its own `N==1 -> None` / windowed-or-`return_array`
  /// `create_causal_mask` / `"causal"` triad.
  ///
  /// The Rust [`KvCache`] trait (mirroring the mlx-swift-lm `KVCache`
  /// protocol) makes `make_mask` **mandatory**, so the "Python-`hasattr`
  /// vs typed-protocol" tension must be resolved exactly as the other
  /// authoritative reference resolves it: mlx-swift-lm's `ChunkedKVCache`
  /// (`KVCache.swift:1008`) is `ChunkedKVCache: KVCacheSimple`, neither
  /// `ChunkedKVCache` nor `KVCacheSimple` overrides `makeMask`, so it
  /// inherits `BaseKVCache.makeMask` (`KVCache.swift:177-191`) — the
  /// standard offset-aware `create_attention_mask` (`offset = self.offset`).
  /// This is byte-identical to [`StandardKvCache::make_mask`](
  /// super::StandardKvCache) (mlx-lm `cache.py:114-126` with
  /// `offset=self.offset`), which is the faithful behavior for the typed
  /// trait. (The only place this differs from mlx-lm's `hasattr`-False
  /// fallthrough is the `offset` passed to a *materialized* mask when
  /// `offset > 0` and `window_size`/`return_array` forces an array;
  /// mlx-swift-lm — the typed reference this trait mirrors — definitively
  /// passes `self.offset`, and the symbolic `None`/`Causal` cases are
  /// identical to mlx-lm either way.)
  fn make_mask(
    &self,
    n: usize,
    window_size: Option<usize>,
    return_array: bool,
  ) -> Result<MaskMode> {
    mask::create_attention_mask(n, self.offset(), return_array, window_size)
  }

  /// mlx-lm `ChunkedKVCache.nbytes` (`cache.py:807-811`):
  /// `keys.nbytes + values.nbytes` (`0` if empty).
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

  /// mlx-lm `ChunkedKVCache.empty` (`cache.py:804-805`): `keys is None`.
  fn is_empty(&self) -> bool {
    self.keys.is_none()
  }

  /// An independent copy (mlx-lm `copy.deepcopy` / mlx-swift-lm
  /// `ChunkedKVCache.copy()`, `KVCache.swift:1071-1080`). Independence is
  /// from MLX value semantics, not buffer duplication: arrays are immutable
  /// and this cache only ever *reassigns* `keys`/`values` to freshly
  /// computed arrays (never mutates a buffer in place), so although
  /// [`Array::try_clone`] is a refcount-sharing clone, copy and original
  /// (including the scalar `offset` / `chunk_size` / `start_position`,
  /// copied by value) evolve completely independently.
  ///
  /// Swift's `copy()` is infallible; the fallible [`Array::try_clone`] is
  /// propagated as a `Result` — a clone failure is **never** mapped to a
  /// `None` buffer (silent corruption) and **never** panicked.
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
      chunk_size: self.chunk_size,
      start_position: self.start_position,
    }))
  }

  /// `"ChunkedKVCache"` — mlx-lm's `type(ChunkedKVCache).__name__`
  /// (`cache.py:56`) / mlx-swift-lm
  /// `case is ChunkedKVCache: return "ChunkedKVCache"`
  /// (`KVCache.swift:1383`).
  fn reference_class_name(&self) -> &'static str {
    "ChunkedKVCache"
  }
}
