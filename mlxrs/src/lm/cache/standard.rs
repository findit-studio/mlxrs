//! [`StandardKvCache`] — the full-attention append-and-fetch cache.

use crate::{
  array::Array,
  error::{Error, Result},
  lm::cache::{
    KvCache, MaskMode, mask,
    util::{concat_seq, nbytes, seq_len, slice_seq},
  },
};

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
/// [`update`](KvCache::update) concatenates onto the correct, trimmed prefix
/// — the observable result is identical to mlx-lm's `keys[..., :offset, :]`.
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
}

impl KvCache for StandardKvCache {
  /// The cached sequence length (mlx-lm `KVCache.offset` / `size()`).
  fn offset(&self) -> usize {
    self.offset
  }

  /// Append `keys`/`values` (`[B, n_kv_heads, S, head_dim]`) and return the
  /// full accumulated `(keys, values)` (mlx-lm `KVCache.update_and_fetch`).
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    let s = seq_len("keys", keys)?;
    // Symmetric, STANDALONE per-tensor rank validation on `values` — the
    // faithful-equivalent of the `seq_len("keys", keys)?` rank check above
    // (mlx-lm `cache.py` implicitly requires a 4-D `values`). NOT a
    // keys-vs-values seq-length cross-check (the faithful revert removed
    // that): `seq_len("values", values)` only checks `values`'s OWN rank.
    // The empty-cache branch below `try_clone`s `values` directly, so
    // without this a rank-invalid `values` on a fresh cache would be stored
    // raw and only surface (feature-combo-dependently) on a later op; the
    // guard makes it a DETERMINISTIC recoverable `Err(Error::ShapeMismatch)`
    // on every path (empty/non-empty cache) on entry.
    let _ = seq_len("values", values)?;
    let (k, v) = match (&self.keys, &self.values) {
      (Some(pk), Some(pv)) => (concat_seq(pk, keys)?, concat_seq(pv, values)?),
      _ => (keys.try_clone()?, values.try_clone()?),
    };
    // CORE-1: stage-then-commit. Compute the return clones BEFORE any
    // `self.*` mutation, then MOVE `k`/`v` into `self.keys`/`self.values`
    // (the same transactional discipline `RotatingKvCache::update_in_place`
    // and `ChunkedKvCache::update` use class-wide). The prior order
    // mutated `self.offset` first, then ran two fallible `try_clone`s on
    // top of it — a clone failure left `self.offset` advanced with the
    // buffer not updated. Same total allocation count (2 clones per side
    // either way); failure no longer poisons the cache.
    let (rk, rv) = (k.try_clone()?, v.try_clone()?);
    self.offset += s;
    self.keys = Some(k);
    self.values = Some(v);
    Ok((rk, rv))
  }

  /// mlx-lm `KVCache.state` getter: `(keys, values)` — here always exactly
  /// the logical `offset` length — or `[]` when empty.
  fn state(&self) -> Result<Vec<Array>> {
    match (&self.keys, &self.values) {
      (Some(k), Some(v)) => Ok(vec![k.try_clone()?, v.try_clone()?]),
      _ => Ok(Vec::new()),
    }
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
        // mlx-lm `KVCache.state` setter (cache.py:371): `self.keys,
        // self.values = v; self.offset = self.keys.shape[2]` — no K/V
        // shape-COMPATIBILITY (cross) validation; it assigns and lets MLX
        // error downstream. We mirror that (NO keys-vs-values comparison),
        // only deriving `offset` from `keys`' own sequence axis. Each
        // stored array is independently rank-validated (a STANDALONE
        // per-tensor 4-D check, symmetric — `keys` already was via the
        // `offset`-deriving `seq_len`; `values` likewise — still NOT a K/V
        // cross-check) so a rank-invalid loaded state is a DETERMINISTIC
        // recoverable `Err(Error::ShapeMismatch)` here on every feature
        // combo rather than a (combo-dependent) later op error.
        let sk = seq_len("keys", &keys)?;
        let _ = seq_len("values", &values)?;
        self.keys = Some(keys);
        self.values = Some(values);
        self.offset = sk;
        Ok(())
      }
      n => Err(Error::Backend {
        message: format!("StandardKvCache state must have 0 or 2 arrays, got {n}"),
      }),
    }
  }

  fn is_trimmable(&self) -> bool {
    true
  }

  /// Drop the most recent `min(offset, n)` tokens; returns the number
  /// actually trimmed (mlx-lm `KVCache.trim`). Keeps the stored tensors in
  /// sync so a later [`update`](KvCache::update) extends the trimmed prefix.
  fn trim(&mut self, n: usize) -> Result<usize> {
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
}
