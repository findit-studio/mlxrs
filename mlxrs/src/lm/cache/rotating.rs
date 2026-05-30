//! [`RotatingKvCache`] — the sliding-window (ring) cache.

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, Error, InvariantViolationPayload, LengthMismatchPayload,
    OutOfRangePayload, ParsePayload, Result,
  },
  lm::cache::{
    KvCache, MaskMode, mask,
    util::{KV_NDIM, concat_seq, head_dim, nbytes, seq_len, seq_slice},
  },
  ops,
};
use smol_str::format_smolstr;

/// Sliding-window KV cache — the cache for models with a bounded attention
/// window (`sliding_window` / `max_kv_size`).
///
/// Faithful 1:1 port of `mlx_lm.models.cache.RotatingKVCache`. mlx-lm has
/// two physical paths — `_update_in_place` (single-token decode, `S == 1`)
/// and `_update_concat` (multi-token prefill, `S > 1`) — and they are
/// **not** observably interchangeable: once the window is full,
/// `_update_in_place` overwrites the slot at the `_idx` ring cursor *in
/// place*, so the returned buffer is in **physical ring order** (e.g.
/// `max_size=8, keep=4` after ids `0..=8` → `[0,1,2,3,8,5,6,7]`, *not* the
/// temporal `[0,1,2,3,5,6,7,8]`), while `_update_concat` over-retains
/// `max_size + S - 1` so every new token still sees `max_size` of context.
/// An attention mask constructed the mlx-lm way relies on exactly this
/// layout, so the port mirrors `_idx`, `_temporal_order`, `_trim`, and both
/// update paths verbatim. Before the window fills, all `offset` tokens are
/// kept; `keep` pins the prompt prefix (BOS / system tokens) outside the
/// rotation. The step buffer is emulated with placeholder rows whose values
/// are provably overwritten or sliced off (`offset < max_size` return)
/// before any observer — so `keys.shape[2]`, which drives the
/// grow/trim/rotate branches and `_idx`, stays identical to mlx-lm for
/// every `S == 1` / `S > 1` mix, while `offset` is the raw uncapped counter.
pub struct RotatingKvCache {
  keys: Option<Array>,
  values: Option<Array>,
  /// Total tokens ever appended (monotone except via `trim`) — mlx-lm
  /// `RotatingKVCache.offset`. This is the raw position the attention
  /// mask / RoPE use; it is *not* capped at `max_size`.
  offset: usize,
  /// Physical ring write cursor — mlx-lm `RotatingKVCache._idx`. The next
  /// in-place single-token write lands at this slot; it wraps back to
  /// `keep` once it reaches `max_size`.
  idx: usize,
  /// Maximum retained window length — mlx-lm `RotatingKVCache.max_size`.
  max_size: usize,
  /// Leading tokens never evicted by rotation — mlx-lm
  /// `RotatingKVCache.keep`.
  keep: usize,
}

/// mlx-lm `RotatingKVCache.step` — how many rows the in-place buffer grows
/// by at a time. Purely an allocation batch size: every grown row is
/// provably overwritten (or sliced off by the `offset < max_size` return)
/// before the buffer is ever returned whole, so its value never reaches an
/// observer. We mirror mlx-lm's `256` so the buffer-length bookkeeping
/// (`keys.shape[2]`, which *does* drive the grow/trim/rotate branches and
/// `_idx`) is byte-for-byte the same across every S==1 / S>1 mix.
const ROTATING_STEP: usize = 256;

impl RotatingKvCache {
  /// A new, empty rotating cache with window `max_size`, pinning the first
  /// `keep` tokens (mlx-lm `RotatingKVCache(max_size, keep)`).
  pub fn new(max_size: usize, keep: usize) -> Self {
    Self {
      keys: None,
      values: None,
      offset: 0,
      idx: 0,
      max_size,
      keep,
    }
  }

  /// The physical ring write cursor — mlx-lm `RotatingKVCache._idx`.
  /// Crate-internal so the sibling [`from_state`](super::from_state) can
  /// assert the post-reconstruction invariant `empty ⇒ offset==0 &&
  /// idx==0` without widening the public [`KvCache`] trait.
  pub(crate) fn idx(&self) -> usize {
    self.idx
  }

  /// mlx-lm `RotatingKVCache._trim`: keep the first `keep` rows then drop
  /// `trim_size` of the next rows; optionally append `append`.
  ///
  /// Allocation-discipline (CORE-1): the no-trim and the append paths pass
  /// the source `&Array`s straight through to `concat_parts` (which takes
  /// `&[&Array]`) instead of cloning them into an owned `Vec<Array>`. The
  /// trim path stores its two slice results in stack-resident `Option<
  /// Array>` slots so the `&Array`s borrowed into `refs` outlive the
  /// `concat_parts` call — no heap `Vec<Array>` either.
  fn trim_buf(&self, trim_size: usize, v: &Array, append: Option<&Array>) -> Result<Array> {
    let l = v.shape()[KV_NDIM - 2];
    let (head_slice, tail_slice): (Option<Array>, Option<Array>) = if trim_size > 0 {
      (
        Some(seq_slice(v, 0, self.keep)?),
        Some(seq_slice(v, trim_size + self.keep, l)?),
      )
    } else {
      (None, None)
    };
    // Refs are populated either from the two owned slices (trim>0) or from
    // the source `v` itself (trim==0 — no clone needed). The optional
    // `append` is always added by-ref (never cloned).
    let mut refs: smallvec::SmallVec<[&Array; 3]> = smallvec::SmallVec::new();
    match (head_slice.as_ref(), tail_slice.as_ref()) {
      (Some(h), Some(t)) => {
        refs.push(h);
        refs.push(t);
      }
      _ => refs.push(v),
    }
    if let Some(a) = append {
      refs.push(a);
    }
    super::util::concat_parts(&refs)
  }

  /// mlx-lm `RotatingKVCache._temporal_order`: rearrange the physical ring
  /// back into arrival order, slicing off the unwritten tail.
  fn temporal_order(&self, v: &Array) -> Result<Array> {
    let l = v.shape()[KV_NDIM - 2];
    if self.idx == l {
      v.try_clone()
    } else if self.idx < self.offset {
      let head = seq_slice(v, 0, self.keep)?;
      let recent = seq_slice(v, self.idx, l)?;
      let mid = seq_slice(v, self.keep, self.idx)?;
      super::util::concat_parts(&[&head, &recent, &mid])
    } else {
      seq_slice(v, 0, self.idx)
    }
  }

  /// Emulate mlx-lm's in-place `buf[..., a:a+s, :] = new` on an immutable
  /// `Array`: splice `new` over `[a, a+s)`, keeping the surrounding rows.
  ///
  /// `name` identifies the target buffer (`"keys"` / `"values"`) for the
  /// non-seq-axes write-shape compatibility error message.
  ///
  /// Structural class-kill (closes #78 P1 iter5): mlx-lm's
  /// `self.<buf>[..., a:a+s, :] = new` slice-assignment routes through
  /// `slice_update`, which broadcasts the RHS to the slice shape
  /// (`mlx/ops.cpp:843` — `broadcast_to(update, upd_shape)`). Our
  /// write-emulation (`concat_parts([head, new, tail])`) has a full-window
  /// shortcut that returns `new` after only a rank check, bypassing both the
  /// non-broadcastable-axes validation AND the size-1 broadcast itself
  /// (e.g. a `[2, .., .., ..]` buffer with a `[1, .., .., ..]` `new` is
  /// valid in mlx-lm — broadcast up to keep the buffer shape — but the
  /// shortcut would silently SHRINK the buffer's batch axis). Route every
  /// `set_seq` window — partial or full — through `broadcast_write_rhs`,
  /// which builds the slice shape `[buf[0], buf[1], a+s-a, buf[3]]` and
  /// broadcasts `new` to it exactly as mlx's `slice_update` does (single
  /// helper, single tensor — NOT the fenced K/V cross-check). Identity
  /// broadcasts are no-ops; size-1 broadcasts expand; non-broadcastable
  /// axes are a recoverable `Err(ShapePairMismatch)`. Faithful to mlx-lm for
  /// every input shape.
  fn set_seq(name: &str, buf: &Array, a: usize, s: usize, new: &Array) -> Result<Array> {
    // Mirror `ChunkedKvCache::set_seq`'s rank-safe + overflow-safe entry:
    // `idx`/`offset` (which feed `a`/`s` at the call sites below) come from
    // the public `set_meta_state` and a hostile/invalid restored meta can
    // drive `a` out of range or `a + s` past `usize::MAX`. Without these
    // guards `seq_slice` would clamp-on-write (silent buffer-length change)
    // or `a + s` would wrap/panic — neither is the recoverable `Err` the
    // `Result` API promises. Use the rank-safe `seq_len` helper, compute
    // `end` via `checked_add`, and reject `end > l` before any splice.
    // Faithful for valid inputs (the slice/concat path is unchanged).
    let l = seq_len(name, buf)?;
    let end = a.checked_add(s).ok_or_else(|| {
      let context: &'static str = match name {
        "keys" => "RotatingKvCache::set_seq: keys write start + S",
        "values" => "RotatingKvCache::set_seq: values write start + S",
        _ => "RotatingKvCache::set_seq: write start + S",
      };
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        context,
        "usize",
        [("start", a as u64), ("S", s as u64)],
      ))
    })?;
    if end > l {
      let context: &'static str = match name {
        "keys" => "RotatingKvCache::set_seq: keys write window end (extends past buffer length)",
        "values" => {
          "RotatingKvCache::set_seq: values write window end (extends past buffer length)"
        }
        _ => "RotatingKvCache::set_seq: write window end (extends past buffer length)",
      };
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        context,
        "must be <= buffer length L",
        format_smolstr!("start={a}, end={end}, L={l}"),
      )));
    }
    let new = super::util::broadcast_write_rhs(name, buf, a, end, new)?;
    let head = seq_slice(buf, 0, a)?;
    let tail = seq_slice(buf, end, l)?;
    super::util::concat_parts(&[&head, &new, &tail])
  }

  /// mlx-lm `RotatingKVCache._update_concat` (the `S > 1` path): put the
  /// ring into temporal order, over-retain `max_size + S - 1` so every new
  /// token still sees `max_size` of context, then append.
  fn update_concat(&mut self, keys: &Array, values: &Array, s: usize) -> Result<(Array, Array)> {
    // mlx-lm `cache.py:464`: `self.offset += S`. Python ints never overflow;
    // a corrupt/hostile prompt cache can restore `offset` near `usize::MAX`
    // via `set_meta_state`, so a multi-token update here would wrap (release)
    // / panic (debug). Compute the post-update offset with `checked_add`
    // BEFORE mutating any ring state (the `self.idx` reassignment below) so
    // the overflow path performs NO partial mutation; the value is
    // byte-identical to `self.offset + s` for every non-overflowing input,
    // so the ring algorithm outcome is unchanged.
    let off = self.offset;
    let new_offset = off.checked_add(s).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "RotatingKvCache::update_concat: offset + S",
        "usize",
        [("offset", off as u64), ("S", s as u64)],
      ))
    })?;
    // `temporal_order` clones out so the `&self.keys` borrow ends before the
    // `self.idx` mutation mlx-lm does at `cache.py:458`.
    let reordered = match (&self.keys, &self.values) {
      (Some(pk), Some(pv)) => Some((self.temporal_order(pk)?, self.temporal_order(pv)?)),
      _ => None,
    };
    let (bk, bv) = if let Some((tk, tv)) = reordered {
      // CORE-1 v2 (Codex round-2 fix): compute `trim_size` from the
      // temporal-order length WITHOUT mutating `self.idx`. Mirrors
      // mlx-lm's two-step at `cache.py:458 + cache.py:462`
      // (`self._idx = self.keys.shape[2]`, then
      // `trim_size = self._idx - self.max_size + 1`) but stages the
      // length locally — the final `self.idx` assignment must wait until
      // all fallible ops below (`trim_buf` for K and V, the two return
      // `try_clone`s) succeed, otherwise a backend/OOM failure leaves
      // `self.idx` advanced to the temporal-order length with the buffer
      // unchanged (so a subsequent S==1 decode uses the wrong ring
      // cursor and over-retains stale context).
      let temporal_len = tk.shape()[KV_NDIM - 2];
      let trim_size = (temporal_len + 1).saturating_sub(self.max_size);
      (
        self.trim_buf(trim_size, &tk, Some(keys))?,
        self.trim_buf(trim_size, &tv, Some(values))?,
      )
    } else {
      (keys.try_clone()?, values.try_clone()?)
    };
    // CORE-1: stage-then-commit. Clone for the return BEFORE any `self.*`
    // mutation, then MOVE `bk`/`bv` into `self.keys`/`self.values` (the
    // same transactional discipline `update_in_place` uses class-wide).
    // The prior order mutated `self.offset`/`self.idx` first, then ran
    // two fallible `try_clone`s on top of them — a clone failure left
    // `offset`/`idx` advanced with the buffer not updated. Same total
    // allocation count (2 clones per side either way); failure no longer
    // poisons the cache.
    let new_idx = bk.shape()[KV_NDIM - 2];
    let (rk, rv) = (bk.try_clone()?, bv.try_clone()?);
    // All fallible ops have succeeded — commit infallibly. mlx-lm
    // `cache.py:466`: `self._idx = self.keys.shape[2]` (final length).
    self.offset = new_offset;
    self.idx = new_idx;
    self.keys = Some(bk);
    self.values = Some(bv);
    Ok((rk, rv))
  }

  /// mlx-lm `RotatingKVCache._update_in_place` (the `S == 1` decode path):
  /// grow the ring while it is below `max_size`, trim/rotate, overwrite the
  /// slot at `_idx`, and return the still-filling prefix or the full ring.
  ///
  /// Transactional (closes a #78 follow-up): every grow/trim/cursor-reset
  /// step is computed into a local; `self.keys`/`self.values`/`self.idx`/
  /// `self.offset` are committed only after ALL fallible ops (grow concat,
  /// trim concat, both `set_seq` splices including the new
  /// `broadcast_write_rhs` validation, and the return slice) succeed. A
  /// recoverable `Err` from any step leaves the cache byte-identical to its
  /// pre-update state — no partially-committed trim, no half-rewound `idx`,
  /// no dropped context that a retry would need (the exact compute-locals-
  /// then-assign discipline `ChunkedKvCache::update` uses). This becomes
  /// load-bearing with the hotfix because the new write-shape validation in
  /// `set_seq` can now fail on a non-broadcastable RHS — previously the
  /// `[one]` concat shortcut silently accepted any 4-D `new`, so the prior
  /// "trim then splice" sequence was infallible on the splice step. Faithful
  /// to mlx-lm for every success path (byte-identical state after a
  /// successful update; mlx-lm's slice-assignment also has no observable
  /// half-state on a failure — the model just propagates the IndexError up).
  fn update_in_place(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    // Both `keys` and `values` are already rank-validated: `update` runs
    // `seq_len("keys", keys)?` AND the symmetric standalone
    // `seq_len("values", values)?` before dispatching here, so each is
    // exactly 4-D and these indices cannot panic on any feature combo. The
    // rank-safe `head_dim` accessor below is kept as defense-in-depth (it
    // is byte-identical to `values.shape[3]` for the now-guaranteed 4-D
    // `values`, mirroring mlx-lm's `values.shape[3]` at `cache.py:478`); it
    // would still surface a recoverable `Error::RankMismatch` rather than
    // a slice OOB panic if this private method were ever reached directly.
    let ks = keys.shape();
    let (b, h, k_hd) = (ks[0], ks[1], ks[3]);
    let v_hd = head_dim("values", values)?;
    let prev = self.offset;

    // mlx-lm `cache.py:506`: `self.offset += 1` (the S==1 decode path).
    // Python ints never overflow; a corrupt/hostile prompt cache can restore
    // `offset` near `usize::MAX` via `set_meta_state`, so this single-token
    // bump would wrap (release) / panic (debug). Compute the post-update
    // offset with `checked_add` BEFORE mutating any ring state so the
    // overflow path performs NO partial mutation; the value is
    // byte-identical to `self.offset + 1` for every non-overflowing input,
    // so the ring algorithm outcome is unchanged.
    let new_offset = prev.checked_add(1).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "RotatingKvCache::update_in_place: offset + S (S=1)",
        "usize",
        [("offset", prev as u64), ("S", 1u64)],
      ))
    })?;

    // ZERO-CLONE TRANSACTIONAL STAGING. The prior pattern `try_cloned`
    // `self.keys`/`self.values` upfront to give grow/trim/splice mutable
    // locals — but `Array::try_clone` is a heap alloc + refcount bump
    // (`mlxrs/src/array/mod.rs:56-63` — "Never `try_clone` in hot paths"),
    // and this is THE hot path (S==1 per-token decode, per layer). Replace
    // it with read-only `&Array` borrows of `self.keys`/`self.values` for
    // every fallible stage: each of `concat_seq` (grow), `trim_buf`,
    // `set_seq` (splice), `seq_slice` (return) produces a NEW owned
    // `Array` into a local, so `self` is genuinely untouched until the
    // final commit — same transactional guarantee, zero upfront clones.
    // The "effective current buffer" is threaded via `Option<Array>` chains
    // (`grown_*.as_ref().or(self.<field>.as_ref())`), no allocation.

    // === Stage 1: GROW (read-only on self). ===
    let cur_len = self.keys.as_ref().map_or(0, |k| k.shape()[KV_NDIM - 2]);
    let need_grow = self.keys.is_none() || (prev >= cur_len && cur_len < self.max_size);
    let (grown_k, grown_v, idx_after_grow): (Option<Array>, Option<Array>, usize) = if need_grow {
      let new_size = ROTATING_STEP.min(self.max_size.saturating_sub(prev));
      let zk = ops::misc::astype(
        &Array::zeros::<f32>(&(b, h, new_size, k_hd))?,
        keys.dtype()?,
      )?;
      let zv = ops::misc::astype(
        &Array::zeros::<f32>(&(b, h, new_size, v_hd))?,
        values.dtype()?,
      )?;
      match (&self.keys, &self.values) {
        (Some(pk), Some(pv)) => (Some(concat_seq(pk, &zk)?), Some(concat_seq(pv, &zv)?), prev),
        _ => (Some(zk), Some(zv), prev),
      }
    } else {
      (None, None, self.idx)
    };

    // Effective buffer ref after Stage 1 (grown if Some, else self).
    let buf_k_after_grow: &Array = grown_k
      .as_ref()
      .or(self.keys.as_ref())
      .expect("buffer is grown-Some on the None-input path, otherwise self.keys is Some");
    let buf_v_after_grow: &Array = grown_v
      .as_ref()
      .or(self.values.as_ref())
      .expect("values mirrors keys");

    // === Stage 2: TRIM (read against post-grow ref, produces new owned). ===
    let cur_len = buf_k_after_grow.shape()[KV_NDIM - 2];
    let trim_size = cur_len.saturating_sub(self.max_size);
    let (trimmed_k, trimmed_v, idx_after_trim): (Option<Array>, Option<Array>, usize) =
      if trim_size > 0 {
        let tk = self.trim_buf(trim_size, buf_k_after_grow, None)?;
        let tv = self.trim_buf(trim_size, buf_v_after_grow, None)?;
        (Some(tk), Some(tv), self.max_size)
      } else {
        (None, None, idx_after_grow)
      };

    // Effective buffer ref after Stage 2 (trim > grow > self).
    let buf_k_ref: &Array = trimmed_k
      .as_ref()
      .or(grown_k.as_ref())
      .or(self.keys.as_ref())
      .unwrap();
    let buf_v_ref: &Array = trimmed_v
      .as_ref()
      .or(grown_v.as_ref())
      .or(self.values.as_ref())
      .unwrap();

    let idx = if idx_after_trim == self.max_size {
      self.keep
    } else {
      idx_after_trim
    };

    // === Stage 3: SPLICE (set_seq; fallible — `broadcast_write_rhs` may
    // reject a non-broadcastable RHS). Produces new owned arrays; `self`
    // is still untouched, so a recoverable `Err` here leaves the cache
    // byte-identical to its pre-update state — no committed trim, no
    // half-rewound `idx`. ===
    let nk = Self::set_seq("keys", buf_k_ref, idx, 1, keys)?;
    let nv = Self::set_seq("values", buf_v_ref, idx, 1, values)?;

    // mlx-lm `cache.py:506-518`: bump `_idx`, then return the still-filling
    // prefix or the full ring. The returned slice (`seq_slice` /
    // `try_clone`) is the final fallible step; compute it BEFORE the commit
    // too so any failure (e.g. OOM on the slice) also leaves `self`
    // untouched.
    let new_idx = idx + 1;
    let ret = if new_offset < self.max_size {
      (
        seq_slice(&nk, 0, new_offset)?,
        seq_slice(&nv, 0, new_offset)?,
      )
    } else {
      (nk.try_clone()?, nv.try_clone()?)
    };

    // All fallible work succeeded — commit the new state atomically.
    self.keys = Some(nk);
    self.values = Some(nv);
    self.offset = new_offset;
    self.idx = new_idx;
    Ok(ret)
  }
}

impl KvCache for RotatingKvCache {
  /// The raw total tokens ever appended — mlx-lm `RotatingKVCache.offset`
  /// (consistent with [`StandardKvCache::offset`](super::StandardKvCache);
  /// this is the value the attention mask / RoPE position use, **not** a
  /// `max_size` cap).
  fn offset(&self) -> usize {
    self.offset
  }

  /// mlx-lm `cache.RotatingKVCache.max_size` — drives windowed masking.
  fn max_size(&self) -> Option<usize> {
    Some(self.max_size)
  }

  /// Append `keys`/`values` and return the retained `(keys, values)`
  /// (mlx-lm `RotatingKVCache.update_and_fetch`, dispatching on `S`).
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    let s = seq_len("keys", keys)?;
    // Symmetric, STANDALONE per-tensor rank validation on `values` — the
    // exact faithful-equivalent of the `seq_len("keys", keys)?` rank check
    // above (mlx-lm `cache.py` implicitly requires a 4-D `[B, n_kv_heads, S,
    // head_dim]` `values`, indexing `values.shape[3]` at `cache.py:478`). It
    // is NOT the keys-vs-values seq-length cross-check the faithful revert
    // deliberately removed — `seq_len("values", values)` only checks
    // `values`'s OWN rank, never compares it to `keys`. Done BEFORE the S
    // dispatch so a rank-invalid `values` is a DETERMINISTIC recoverable
    // `Err(Error::RankMismatch)` on EVERY path (empty/non-empty cache,
    // S==1's `_update_in_place`, S>1's `_update_concat` including the
    // empty-cache `try_clone` branch) regardless of which downstream MLX op
    // would otherwise (feature-combo-dependently) catch or miss it.
    let _ = seq_len("values", values)?;
    if s == 1 {
      self.update_in_place(keys, values)
    } else {
      self.update_concat(keys, values, s)
    }
  }

  /// mlx-lm `RotatingKVCache.state` getter (cross-checked vs swift): the
  /// `offset`-length slice when the buffer over-allocated, else the buffer;
  /// `[]` when empty.
  fn state(&self) -> Result<Vec<Array>> {
    match (&self.keys, &self.values) {
      (Some(k), Some(v)) => {
        let l = k.shape()[KV_NDIM - 2];
        if self.offset < l {
          Ok(vec![
            seq_slice(k, 0, self.offset)?,
            seq_slice(v, 0, self.offset)?,
          ])
        } else {
          Ok(vec![k.try_clone()?, v.try_clone()?])
        }
      }
      _ => Ok(Vec::new()),
    }
  }

  /// Force-evaluate the cache's own stored ring buffers in place — the
  /// per-chunk prefill memory barrier (see [`KvCache::materialize`]).
  ///
  /// Evals the **full** `self.keys`/`self.values` ring buffers (the arrays
  /// the next chunk's `update_in_place`/`update_concat` actually reads and
  /// extends) via the explicit `&mut` [`Array::eval`] — deliberately **not**
  /// the `seq_slice(k, 0, self.offset)` views [`state`](KvCache::state)
  /// returns once the buffer over-allocates (`offset < buffer_len`, the
  /// regime an `S == 1` decode reaches after growing the ring). Evaluating
  /// those temporary slices would materialize the slice's output buffer, not
  /// the stored ring the next chunk reuses, so the stored lazy graph could
  /// chain across chunks and the prefill would not be memory-bounded — the
  /// exact bug this hook closes. A no-op when empty.
  fn materialize(&mut self) -> Result<()> {
    if let Some(k) = self.keys.as_mut() {
      k.eval()?;
    }
    if let Some(v) = self.values.as_mut() {
      v.eval()?;
    }
    Ok(())
  }

  /// mlx-lm `RotatingKVCache.state` setter: `keys, values = v` (offset/idx
  /// come from [`set_meta_state`](KvCache::set_meta_state), not the keys).
  /// An empty state resets the buffer.
  fn set_state(&mut self, mut state: Vec<Array>) -> Result<()> {
    match state.len() {
      0 => {
        self.keys = None;
        self.values = None;
        Ok(())
      }
      2 => {
        let values = state.pop().unwrap();
        let keys = state.pop().unwrap();
        // mlx-lm `RotatingKVCache.state` setter (cache.py:295): `self.keys,
        // self.values = v` — no K/V shape-COMPATIBILITY (cross) validation
        // (offset/idx come from `set_meta_state`, not the keys); we mirror
        // that: NO keys-vs-values comparison. But each stored array must
        // independently be the assumed 4-D `[B, n_kv_heads, S, head_dim]`:
        // unlike mlx-lm (where a later op raises a catchable error), our
        // `update_concat`/`temporal_order` read the cached buffer's
        // sequence axis with a RAW `v.shape()[KV_NDIM - 2]` index, so a
        // rank-invalid stored array would be a Rust slice OOB *panic* on a
        // later valid `update` — not a recoverable error. A STANDALONE
        // per-tensor rank check on each (symmetric to the `seq_len` rank
        // check at `update` entry; still NOT a K/V cross-check) makes a
        // rank-invalid loaded state a DETERMINISTIC recoverable
        // `Err(Error::RankMismatch)` here instead.
        let _ = seq_len("keys", &keys)?;
        let _ = seq_len("values", &values)?;
        self.keys = Some(keys);
        self.values = Some(values);
        Ok(())
      }
      n => Err(Error::OutOfRange(OutOfRangePayload::new(
        "RotatingKvCache::set_state: state array count",
        "must be 0 or 2",
        format_smolstr!("{n}"),
      ))),
    }
  }

  /// mlx-lm `RotatingKVCache.meta_state`:
  /// `(keep, max_size, offset, _idx)` as decimal strings.
  fn meta_state(&self) -> Vec<String> {
    vec![
      self.keep.to_string(),
      self.max_size.to_string(),
      self.offset.to_string(),
      self.idx.to_string(),
    ]
  }

  /// mlx-lm `RotatingKVCache.meta_state` setter:
  /// `self.keep, self.max_size, self.offset, self._idx = map(int, v)`.
  fn set_meta_state(&mut self, m: &[String]) -> Result<()> {
    if m.len() != 4 {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "RotatingKvCache::set_meta_state: meta_state values",
        4,
        m.len(),
      )));
    }
    let parse = |i: usize, name: &'static str| -> Result<usize> {
      m[i].parse::<usize>().map_err(|e: std::num::ParseIntError| {
        // Distinguish the field via static context strings so the typed
        // ParsePayload carries the per-field call site without runtime alloc.
        let context: &'static str = match name {
          "keep" => "RotatingKvCache::set_meta_state: keep",
          "max_size" => "RotatingKvCache::set_meta_state: max_size",
          "offset" => "RotatingKvCache::set_meta_state: offset",
          "idx" => "RotatingKvCache::set_meta_state: idx",
          _ => "RotatingKvCache::set_meta_state",
        };
        Error::Parse(ParsePayload::new(context, "usize", Box::new(e)))
      })
    };
    // Parse ALL four into locals and validate before mutating ANY field, so
    // a parse error on a later value (e.g. a non-numeric `offset`) leaves
    // the cache exactly as it was rather than partially corrupted (`keep`
    // changed though `max_size`/`offset`/`idx` failed). Faithful semantics
    // are unchanged: same four fields, same order/format as cache.py:531-533
    // (`self.keep, self.max_size, self.offset, self._idx = map(int, v)`); on
    // success all four are assigned, so `from_state`'s post-`set_state`
    // +`set_meta_state` `empty ⇒ offset==0 && idx==0` invariant still sees
    // the loaded `offset`/`idx`.
    let keep = parse(0, "keep")?;
    let max_size = parse(1, "max_size")?;
    let offset = parse(2, "offset")?;
    let idx = parse(3, "idx")?;
    self.keep = keep;
    self.max_size = max_size;
    self.offset = offset;
    self.idx = idx;
    Ok(())
  }

  /// Whether the cache can be trimmed — only before the window fills
  /// (`offset < max_size`), exactly mlx-lm `RotatingKVCache.is_trimmable`.
  fn is_trimmable(&self) -> bool {
    self.offset < self.max_size
  }

  /// Drop the most recent `min(offset, n)` tokens; returns the number
  /// trimmed (mlx-lm `RotatingKVCache.trim`: it only adjusts `offset` and
  /// `_idx` — the next in-place update rewrites from the moved cursor — and
  /// is only valid before the ring fills, see
  /// [`is_trimmable`](KvCache::is_trimmable)).
  fn trim(&mut self, n: usize) -> Result<usize> {
    let trimmed = n.min(self.offset);
    self.offset -= trimmed;
    self.idx = self.idx.saturating_sub(trimmed);
    Ok(trimmed)
  }

  /// 1:1 port of mlx-lm `RotatingKVCache.make_mask` (`cache.py:554-578`) —
  /// the rotating cache's **own** override, *not* the generic
  /// `create_attention_mask`. Two regimes:
  ///
  /// - `N > 1` (prefill): `window_size or self.max_size`, the offset capped
  ///   at `max_size - 1` (mlx-lm's `min(self.max_size-1, self.offset)`; the
  ///   struct's `offset` is the raw uncapped counter, see
  ///   [`offset`](KvCache::offset)). If `offset + N > window_size` or
  ///   `return_array`, a windowed `create_causal_mask`; else the symbolic
  ///   [`MaskMode::Causal`] (`cache.py:560-563`).
  /// - `N == 1` (decode): no mask unless a `window_size` is given AND
  ///   `self.offset >= window_size` AND `self.max_size > window_size`, in
  ///   which case the rolled physical-ring mask
  ///   `roll(arange(mask_size) >= mask_size - window_size, idx + 1)` over the
  ///   ring cursor (`cache.py:565-578`); any other path falls through to
  ///   [`MaskMode::None`] (mlx-lm's implicit `return None`).
  ///
  /// `crate::ops` has no native `roll`; the 1-D roll is composed faithfully
  /// by `mask::roll_1d` (`out[i] = a[(i-s) mod L]`). The `max_size - 1` /
  /// `mask_size - window_size` differences are guarded with `saturating_sub`
  /// — for every real rotating cache (`max_size >= 1`, and the branch
  /// guarantees `mask_size >= window_size + 1`) this is exactly mlx-lm's
  /// integer arithmetic; it only avoids an underflow panic on the degenerate
  /// `max_size == 0` `from_state` placeholder before `set_meta_state`.
  fn make_mask(
    &self,
    n: usize,
    window_size: Option<usize>,
    return_array: bool,
  ) -> Result<MaskMode> {
    if n > 1 {
      // `window_size = window_size or self.max_size` (cache.py:558).
      // Python `or` is truthiness, not None-coalescing: `0` is falsy, so
      // `Some(0)` must ALSO fall back to `self.max_size` (a plain
      // `unwrap_or` would keep `0` and yield a wrong all-windowed/empty N>1
      // mask). `None`/`Some(0)` -> `max_size`; `Some(w != 0)` -> `w`.
      let ws = window_size.filter(|&w| w != 0).unwrap_or(self.max_size);
      // `offset = min(self.max_size - 1, self.offset)` (cache.py:559)
      let offset = self.max_size.saturating_sub(1).min(self.offset);
      // `offset + N` (cache.py:560). Python ints never overflow; a corrupt
      // loaded `max_size`/`offset` near `usize::MAX` would here wrap
      // (release) / panic (debug) BEFORE `create_causal_mask`'s checked-add
      // can catch it, possibly flipping this decision. Compute it checked
      // (matching the round-2 `create_causal_mask` fix); the comparison
      // result is byte-identical to `offset + n` for every non-overflowing
      // input, so the decision outcome is unchanged.
      let offset_plus_n = offset.checked_add(n).ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "RotatingKvCache::make_mask: offset + N",
          "usize",
          [("offset", offset as u64), ("N", n as u64)],
        ))
      })?;
      if offset_plus_n > ws || return_array {
        Ok(MaskMode::Array(mask::create_causal_mask(
          n,
          offset,
          Some(ws),
        )?))
      } else {
        Ok(MaskMode::Causal)
      }
    } else {
      // N == 1
      match window_size {
        // `if window_size is None: return None` (cache.py:565-566)
        None => Ok(MaskMode::None),
        Some(ws) => {
          // `if self.offset >= window_size and self.max_size > window_size`
          // (cache.py:568)
          if self.offset >= ws && self.max_size > ws {
            // `idx = self._idx; if idx >= self.max_size: idx = 0`
            // (cache.py:569-571)
            let idx = if self.idx >= self.max_size {
              0
            } else {
              self.idx
            };
            // `mask_size = self.offset + 1 if self.offset < self.max_size
            //  else self.max_size` (cache.py:572-575)
            let mask_size = if self.offset < self.max_size {
              self.offset + 1
            } else {
              self.max_size
            };
            // `mask = mx.arange(mask_size) >= (mask_size - window_size)`
            // (cache.py:576) — built in mask.rs's I32 grid / Bool result
            // idiom (same as create_causal_mask).
            let inds = mask::iarange(0, mask_size)?;
            let bound = mask::scalar_i32((mask_size.saturating_sub(ws)) as i32)?;
            let m = ops::comparison::greater_equal(&inds, &bound)?;
            // `mask = mx.roll(mask, shift=idx + 1)` (cache.py:577)
            let m = mask::roll_1d(&m, idx + 1)?;
            Ok(MaskMode::Array(m))
          } else {
            // Python falls through with no `return` -> None.
            Ok(MaskMode::None)
          }
        }
      }
    }
  }

  /// mlx-lm `RotatingKVCache.nbytes`: `keys.nbytes + values.nbytes`
  /// (0 if empty).
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
  /// arrays are immutable and the cache only ever *reassigns* `keys` /
  /// `values` to freshly-computed arrays (never mutates a buffer in place),
  /// so although `Array::try_clone` is a refcount-sharing clone, the copy
  /// and the original (including the scalar ring fields, copied by value)
  /// evolve completely independently.
  ///
  /// Swift's `copy()` is infallible; here the fallible [`Array::try_clone`]
  /// is propagated as a `Result` (`try_clone()?`) — a clone failure is
  /// **never** mapped to `None` (which would yield a cache with the right
  /// `offset`/`idx` but missing keys/values: silent corruption) and
  /// **never** panicked.
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
      idx: self.idx,
      max_size: self.max_size,
      keep: self.keep,
    }))
  }

  /// `"RotatingKVCache"` — mlx-lm's `type(RotatingKVCache).__name__`
  /// (`cache.py:56`) / mlx-swift-lm
  /// `case is RotatingKVCache: return "RotatingKVCache"`
  /// (`KVCache.swift:1386`).
  fn reference_class_name(&self) -> &'static str {
    "RotatingKVCache"
  }

  /// P1 #110: per-layer fast-path downcast target — see the
  /// [`KvCache`]-trait doc's **Per-layer fast-path convention**.
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }

  /// Transactional override of [`KvCache::from_serialized`] — leaves `self`
  /// byte-identical to its pre-call state on every recoverable error
  /// (`set_state` rank failures, `set_meta_state` parse failures). All
  /// fallible work (rank validation in `set_state`; the 4-field meta parse
  /// in `set_meta_state` — `keep`, `max_size`, `offset`, `idx`) runs on a
  /// fresh placeholder `RotatingKvCache::new(0, 0)` (the exact placeholder
  /// the existing [`super::from_state`] dispatch uses), and `self` is
  /// committed by a single infallible move only after both succeed. The
  /// default trait impl would mutate `self.keys`/`self.values` via
  /// `set_state` first and could then leave them assigned while a malformed
  /// meta drops `offset`/`idx` to inconsistent values; this override closes
  /// that window.
  fn from_serialized(&mut self, state: Vec<Array>, meta: &[String]) -> Result<()> {
    let mut staged = RotatingKvCache::new(0, 0);
    staged.set_state(state)?;
    staged.set_meta_state(meta)?;
    // Post-setter invariant guard — must match `super::from_state`'s
    // dispatcher arm (`cache/mod.rs:453`). The setters individually stay
    // 1:1 with `mlx_lm/models/cache.py:527-540` (no validation), but
    // `set_state(Vec::new())` accepts an empty state and `set_meta_state`
    // can then restore non-zero `offset`/`idx` after it — an impossible
    // combination from a real round-trip (`keys=None ⇒ offset==idx==0`)
    // that would let the next `update` treat the bogus `offset` as `prev`
    // and surface placeholder zeros as "prior context". Reject the
    // forged combination here so the new `from_serialized` public API
    // is observably identical to the canonical `from_state` loader.
    if staged.is_empty() && (staged.offset() != 0 || staged.idx() != 0) {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "RotatingKvCache: empty state with non-zero offset/idx",
        "is invalid (empty state must have offset=0 and idx=0)",
      )));
    }
    *self = staged;
    Ok(())
  }
}
