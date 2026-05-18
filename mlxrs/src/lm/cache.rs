//! Key/value caches for incremental decoding, ported from
//! [`mlx_lm.models.cache`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
//! (`KVCache` / `ConcatenateKVCache` / `RotatingKVCache`) and cross-checked
//! against mlx-swift-lm's `MLXLMCommon` KV cache.
//!
//! A cache holds the per-layer attention keys/values seen so far so each
//! decode step only re-computes the new token. One [`KvCache`] exists per
//! decoder layer; [`make_prompt_cache`] builds the vector the
//! [`Model`](crate::lm::model::Model) mutates in place.
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
  ops,
};

/// The number of key/value heads + sequence axes a KV state must have:
/// `[B, n_kv_heads, S, head_dim]` (mlx-lm's `keys.shape == (B, n_kv_heads, S,
/// head_dim)` — the sequence axis is `-2`).
const KV_NDIM: usize = 4;
/// The sequence axis of a `[B, n_kv_heads, S, head_dim]` KV state, as a
/// negative (rank-relative) index — mlx-lm concatenates/slices keys on
/// `axis=-2`.
const SEQ_AXIS: i32 = -2;

/// mlx-lm's default `RotatingKVCache.keep` for sliding-window models
/// (`make_prompt_cache(... ) -> RotatingKVCache(max_size=..., keep=4)`).
pub const ROTATING_DEFAULT_KEEP: i32 = 4;

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

/// Validate a key/value tensor's rank and return its sequence length
/// (`shape[-2]`). mlx-lm assumes the 4-D `[B, n_kv_heads, S, head_dim]`
/// layout; we check it instead of indexing blindly so a misuse is a
/// recoverable [`Error::ShapeMismatch`], not a panic.
fn seq_len(name: &str, a: &Array) -> Result<usize> {
  let shape = a.shape();
  if shape.len() != KV_NDIM {
    return Err(Error::ShapeMismatch {
      message: format!(
        "KV cache expects 4-D {name} [B, n_kv_heads, S, head_dim], got shape {shape:?}"
      ),
    });
  }
  Ok(shape[KV_NDIM - 2])
}

/// Slice the sequence axis (`-2`) of a 4-D KV tensor to `[start, end)`,
/// keeping every other axis full. mlx-lm's `v[..., start:end, :]`.
fn slice_seq(a: &Array, start: usize, end: usize) -> Result<Array> {
  let shape = a.shape();
  let mut starts = vec![0i32; KV_NDIM];
  let mut stops: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
  let strides = vec![1i32; KV_NDIM];
  starts[KV_NDIM - 2] = start as i32;
  stops[KV_NDIM - 2] = end as i32;
  ops::indexing::slice(a, &starts, &stops, &strides)
}

/// Concatenate two 4-D KV tensors along the sequence axis (`-2`) — mlx-lm's
/// `mx.concatenate([a, b], axis=-2)`.
fn concat_seq(a: &Array, b: &Array) -> Result<Array> {
  ops::shape::concatenate(&[a, b], SEQ_AXIS)
}

/// Append-and-fetch KV cache — the default cache for full-attention models.
///
/// Port of `mlx_lm.models.cache.KVCache` (observable behavior of its
/// documented twin `ConcatenateKVCache`): each call concatenates the new
/// keys/values onto the running tensors along the sequence axis and returns
/// the full accumulated `(keys, values)`. `offset` tracks the sequence
/// length; `trim(n)` drops the most recent `min(offset, n)` tokens.
///
/// Unlike mlx-lm's step buffer, the stored tensors are always exactly the
/// logical `offset` length (slicing on `trim`), so the next
/// `update_and_fetch` concatenates onto the correct, trimmed prefix — the
/// observable result is identical to mlx-lm's `keys[..., :offset, :]`.
#[derive(Default)]
pub struct StandardKvCache {
  keys: Option<Array>,
  values: Option<Array>,
  offset: usize,
}

impl StandardKvCache {
  /// A new, empty cache.
  pub fn new() -> Self {
    Self::default()
  }

  /// Append `keys`/`values` (`[B, n_kv_heads, S, head_dim]`) and return the
  /// full accumulated `(keys, values)` (mlx-lm `KVCache.update_and_fetch`).
  pub fn update_and_fetch(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    let s = seq_len("keys", keys)?;
    seq_len("values", values)?;
    let (k, v) = match (&self.keys, &self.values) {
      (Some(pk), Some(pv)) => (concat_seq(pk, keys)?, concat_seq(pv, values)?),
      _ => (keys.try_clone()?, values.try_clone()?),
    };
    self.offset += s;
    self.keys = Some(k.try_clone()?);
    self.values = Some(v.try_clone()?);
    Ok((k, v))
  }

  /// The cached sequence length (mlx-lm `KVCache.offset` / `size()`).
  pub fn offset(&self) -> usize {
    self.offset
  }

  /// Drop the most recent `min(offset, n)` tokens; returns the number
  /// actually trimmed (mlx-lm `KVCache.trim`). Keeps the stored tensors in
  /// sync so a later `update_and_fetch` extends the trimmed prefix.
  pub fn trim(&mut self, n: usize) -> Result<usize> {
    let trimmed = n.min(self.offset);
    self.offset -= trimmed;
    if trimmed > 0
      && let (Some(k), Some(v)) = (&self.keys, &self.values)
    {
      let nk = slice_seq(k, 0, self.offset)?;
      let nv = slice_seq(v, 0, self.offset)?;
      self.keys = Some(nk);
      self.values = Some(nv);
    }
    Ok(trimmed)
  }

  /// Whether the cache holds no keys yet (mlx-lm `empty()`).
  pub fn is_empty(&self) -> bool {
    self.keys.is_none()
  }
}

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

/// Slice the sequence axis to `[start, end)` with Python/NumPy-style
/// clamping (`end` capped at the length, `start` capped at `end`) so an
/// over-long bound is the empty/whole slice mlx-lm's `v[..., a:b, :]`
/// would produce, never a panic.
fn seq_slice(a: &Array, start: usize, end: usize) -> Result<Array> {
  let l = a.shape()[KV_NDIM - 2];
  let end = end.min(l);
  let start = start.min(end);
  slice_seq(a, start, end)
}

/// Concatenate the non-empty parts along the sequence axis (an empty part
/// is a no-op in mlx-lm's `mx.concatenate`; dropping it avoids a redundant
/// op and any zero-length-concat edge). A single part is returned directly.
fn concat_parts(parts: &[&Array]) -> Result<Array> {
  let non_empty: Vec<&Array> = parts
    .iter()
    .copied()
    .filter(|a| a.shape()[KV_NDIM - 2] > 0)
    .collect();
  match non_empty.as_slice() {
    [] => parts[0].try_clone(),
    [one] => one.try_clone(),
    many => ops::shape::concatenate(many, SEQ_AXIS),
  }
}

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

  /// mlx-lm `RotatingKVCache._trim`: keep the first `keep` rows then drop
  /// `trim_size` of the next rows; optionally append `append`.
  fn trim_buf(&self, trim_size: usize, v: &Array, append: Option<&Array>) -> Result<Array> {
    let l = v.shape()[KV_NDIM - 2];
    let mut owned: Vec<Array> = Vec::new();
    if trim_size > 0 {
      owned.push(seq_slice(v, 0, self.keep)?);
      owned.push(seq_slice(v, trim_size + self.keep, l)?);
    } else {
      owned.push(v.try_clone()?);
    }
    if let Some(a) = append {
      owned.push(a.try_clone()?);
    }
    let refs: Vec<&Array> = owned.iter().collect();
    concat_parts(&refs)
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
      concat_parts(&[&head, &recent, &mid])
    } else {
      seq_slice(v, 0, self.idx)
    }
  }

  /// Emulate mlx-lm's in-place `buf[..., a:a+s, :] = new` on an immutable
  /// `Array`: splice `new` over `[a, a+s)`, keeping the surrounding rows.
  fn set_seq(buf: &Array, a: usize, s: usize, new: &Array) -> Result<Array> {
    let l = buf.shape()[KV_NDIM - 2];
    let head = seq_slice(buf, 0, a)?;
    let tail = seq_slice(buf, a + s, l)?;
    concat_parts(&[&head, new, &tail])
  }

  /// mlx-lm `RotatingKVCache._update_concat` (the `S > 1` path): put the
  /// ring into temporal order, over-retain `max_size + S - 1` so every new
  /// token still sees `max_size` of context, then append.
  fn update_concat(&mut self, keys: &Array, values: &Array, s: usize) -> Result<(Array, Array)> {
    // `temporal_order` clones out so the `&self.keys` borrow ends before the
    // `self.idx` mutation mlx-lm does at `cache.py:458`.
    let reordered = match (&self.keys, &self.values) {
      (Some(pk), Some(pv)) => Some((self.temporal_order(pk)?, self.temporal_order(pv)?)),
      _ => None,
    };
    let (bk, bv) = if let Some((tk, tv)) = reordered {
      // mlx-lm reassigns `self._idx = self.keys.shape[2]` to the
      // temporal-order length (cache.py:458) and ONLY THEN computes
      // `trim_size = self._idx - self.max_size + 1` (cache.py:462) — the
      // trim window must come from the reordered buffer, not the stale ring
      // cursor, or a still-active-ring S==1 then S>1 update over-retains.
      self.idx = tk.shape()[KV_NDIM - 2];
      let trim_size = (self.idx + 1).saturating_sub(self.max_size);
      (
        self.trim_buf(trim_size, &tk, Some(keys))?,
        self.trim_buf(trim_size, &tv, Some(values))?,
      )
    } else {
      (keys.try_clone()?, values.try_clone()?)
    };
    self.offset += s;
    // mlx-lm `cache.py:466`: `self._idx = self.keys.shape[2]` (final length).
    self.idx = bk.shape()[KV_NDIM - 2];
    self.keys = Some(bk.try_clone()?);
    self.values = Some(bv.try_clone()?);
    Ok((bk, bv))
  }

  /// mlx-lm `RotatingKVCache._update_in_place` (the `S == 1` decode path):
  /// grow the ring while it is below `max_size`, trim/rotate, overwrite the
  /// slot at `_idx`, and return the still-filling prefix or the full ring.
  fn update_in_place(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    let ks = keys.shape();
    let (b, h, k_hd) = (ks[0], ks[1], ks[3]);
    let v_hd = values.shape()[3];
    let prev = self.offset;

    let cur_len = self.keys.as_ref().map_or(0, |k| k.shape()[KV_NDIM - 2]);
    if self.keys.is_none() || (prev >= cur_len && cur_len < self.max_size) {
      let new_size = ROTATING_STEP.min(self.max_size.saturating_sub(prev));
      let zk = ops::misc::astype(
        &Array::zeros::<f32>(&(b, h, new_size, k_hd))?,
        keys.dtype()?,
      )?;
      let zv = ops::misc::astype(
        &Array::zeros::<f32>(&(b, h, new_size, v_hd))?,
        values.dtype()?,
      )?;
      let (nk, nv) = match (&self.keys, &self.values) {
        (Some(pk), Some(pv)) => (concat_seq(pk, &zk)?, concat_seq(pv, &zv)?),
        _ => (zk, zv),
      };
      self.keys = Some(nk);
      self.values = Some(nv);
      self.idx = prev;
    }

    let cur_len = self.keys.as_ref().map_or(0, |k| k.shape()[KV_NDIM - 2]);
    let trim_size = cur_len.saturating_sub(self.max_size);
    if trim_size > 0 {
      let tk = self.trim_buf(trim_size, self.keys.as_ref().unwrap(), None)?;
      let tv = self.trim_buf(trim_size, self.values.as_ref().unwrap(), None)?;
      self.keys = Some(tk);
      self.values = Some(tv);
      self.idx = self.max_size;
    }

    if self.idx == self.max_size {
      self.idx = self.keep;
    }

    let nk = Self::set_seq(self.keys.as_ref().unwrap(), self.idx, 1, keys)?;
    let nv = Self::set_seq(self.values.as_ref().unwrap(), self.idx, 1, values)?;
    self.keys = Some(nk);
    self.values = Some(nv);
    self.offset += 1;
    self.idx += 1;

    let k = self.keys.as_ref().unwrap();
    let v = self.values.as_ref().unwrap();
    if self.offset < self.max_size {
      Ok((seq_slice(k, 0, self.offset)?, seq_slice(v, 0, self.offset)?))
    } else {
      Ok((k.try_clone()?, v.try_clone()?))
    }
  }

  /// Append `keys`/`values` and return the retained `(keys, values)`
  /// (mlx-lm `RotatingKVCache.update_and_fetch`, dispatching on `S`).
  pub fn update_and_fetch(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    let s = seq_len("keys", keys)?;
    seq_len("values", values)?;
    if s == 1 {
      self.update_in_place(keys, values)
    } else {
      self.update_concat(keys, values, s)
    }
  }

  /// The raw total tokens ever appended — mlx-lm `RotatingKVCache.offset`
  /// (consistent with [`StandardKvCache::offset`]; this is the value the
  /// attention mask / RoPE position use, **not** a `max_size` cap).
  pub fn offset(&self) -> usize {
    self.offset
  }

  /// Drop the most recent `min(offset, n)` tokens; returns the number
  /// trimmed (mlx-lm `RotatingKVCache.trim`: it only adjusts `offset` and
  /// `_idx` — the next in-place update rewrites from the moved cursor — and
  /// is only valid before the ring fills, see [`is_trimmable`](
  /// RotatingKvCache::is_trimmable)).
  pub fn trim(&mut self, n: usize) -> Result<usize> {
    let trimmed = n.min(self.offset);
    self.offset -= trimmed;
    self.idx = self.idx.saturating_sub(trimmed);
    Ok(trimmed)
  }

  /// Whether the cache can be trimmed — only before the window fills
  /// (`offset < max_size`), exactly mlx-lm `RotatingKVCache.is_trimmable`.
  pub fn is_trimmable(&self) -> bool {
    self.offset < self.max_size
  }

  /// Whether the cache holds no keys yet (mlx-lm `empty()`).
  pub fn is_empty(&self) -> bool {
    self.keys.is_none()
  }
}

/// The per-layer KV cache, dispatched by kind.
///
/// `#[non_exhaustive]` so deferred variants (e.g. a quantized KV cache —
/// tracked M3 follow-up) are purely additive: callers use the inherent
/// methods ([`update_and_fetch`](KvCache::update_and_fetch), [`offset`](
/// KvCache::offset), [`trim`](KvCache::trim)) and never `match` it
/// exhaustively, so adding a variant is not a breaking change.
#[non_exhaustive]
pub enum KvCache {
  /// Full-attention append-and-fetch cache.
  Standard(StandardKvCache),
  /// Sliding-window rotating cache.
  Rotating(RotatingKvCache),
}

impl KvCache {
  /// Append `keys`/`values` and return the cache's current `(keys, values)`,
  /// dispatching to the active variant (mlx-lm `cache.update_and_fetch`).
  pub fn update_and_fetch(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    match self {
      KvCache::Standard(c) => c.update_and_fetch(keys, values),
      KvCache::Rotating(c) => c.update_and_fetch(keys, values),
    }
  }

  /// The raw total tokens ever appended — mlx-lm `cache.offset` (the
  /// attention-mask / RoPE position). Uniform across variants: a rotating
  /// cache returns the same raw counter as a standard one, **not** a
  /// `max_size`-capped window length.
  pub fn offset(&self) -> usize {
    match self {
      KvCache::Standard(c) => c.offset(),
      KvCache::Rotating(c) => c.offset(),
    }
  }

  /// Drop the most recent `min(offset, n)` tokens; returns the number
  /// trimmed (mlx-lm `cache.trim`).
  pub fn trim(&mut self, n: usize) -> Result<usize> {
    match self {
      KvCache::Standard(c) => c.trim(n),
      KvCache::Rotating(c) => c.trim(n),
    }
  }

  /// Whether the cache holds no keys yet (mlx-lm `cache.empty()`).
  pub fn is_empty(&self) -> bool {
    match self {
      KvCache::Standard(c) => c.is_empty(),
      KvCache::Rotating(c) => c.is_empty(),
    }
  }
}

/// Build one [`KvCache`] per decoder layer for `cfg`, mirroring
/// `mlx_lm.models.cache.make_prompt_cache`.
///
/// A [`RotatingKvCache`] (window = `cfg.sliding_window`, `keep =
/// ROTATING_DEFAULT_KEEP` = 4, matching mlx-lm) is used iff the model has a
/// sliding window; otherwise a [`StandardKvCache`]. The vector has exactly
/// `cfg.num_hidden_layers` entries.
pub fn make_prompt_cache(cfg: &CacheConfig) -> Vec<KvCache> {
  (0..cfg.num_hidden_layers)
    .map(|_| match cfg.sliding_window {
      Some(window) => KvCache::Rotating(RotatingKvCache::new(
        window.max(0) as usize,
        ROTATING_DEFAULT_KEEP.max(0) as usize,
      )),
      None => KvCache::Standard(StandardKvCache::new()),
    })
    .collect()
}
