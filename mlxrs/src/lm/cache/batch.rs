//! [`BatchKvCache`] + [`dynamic_roll`] — the left-padded batched
//! full-attention cache, ported 1:1 from
//! [`mlx_lm.models.cache`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
//! `dynamic_roll` (`cache.py:903-909`) and `BatchKVCache`
//! (`cache.py:912-1131`).
//!
//! mlx-swift-lm has **no** concrete `BatchKVCache` (only the
//! `BatchPositionedKVCache` protocol in `RoPEApplication.swift:13-22` and
//! `RoPEOffset.batch`), so mlx-lm is the authoritative algorithm; the swift
//! cross-check is the `batchOffset` → `.batch(batchOffset[.ellipsis])`
//! rope-offset contract (already provided by the merged
//! [`KvCache::rope_offset`] default once
//! [`as_batch_positioned`](super::KvCache::as_batch_positioned) is `Some`).
//!
//! Like [`StandardKvCache`](super::StandardKvCache) vs mlx-lm's
//! `KVCache` step buffer, `BatchKVCache`'s `step`-sized over-allocation is a
//! pure allocation optimization with **no** observable effect: every
//! returned token (`return self.keys[..., :self._idx, :]`,
//! `cache.py:965`) is a real written token (`self.keys[..., prev:self._idx,
//! :] = keys`, `cache.py:963`), the zero rows are always sliced off before
//! return, and there is **no** in-place ring overwrite an observer can see
//! (unlike [`RotatingKvCache`](super::RotatingKvCache) /
//! [`BatchRotatingKvCache`](super::BatchRotatingKvCache)). So
//! `mlxrs::Array` being functional, the buffer is reproduced exactly via
//! `concatenate`/`slice` — the observable `update_and_fetch` is a
//! sequence-axis concat of every update, with the per-sequence
//! `offset`/`left_padding` arrays carried as the RoPE/mask metadata.
//!
//! No implicit eval: every op is a pure [`crate::ops`] composition.

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    ArithmeticOverflowPayload, DtypeMismatchPayload, Error, InvariantViolationPayload,
    LengthMismatchPayload, OutOfRangePayload, RankMismatchPayload, Result,
    ShapePairMismatchPayload,
  },
  lm::cache::{
    BatchPositionedKvCache, KvCache, MaskMode,
    util::{KV_NDIM, concat_seq, nbytes, seq_len, slice_seq},
  },
  ops,
};
use smol_str::format_smolstr;

/// Rank-safe `keys`/`values` last-axis (head-dim) length, i.e. `shape[-1]`
/// of a 4-D `[B, n_kv_heads, S, head_dim]` KV state.
///
/// mlx-lm reads `values.shape[3]` directly (`cache.py:946`); on the
/// `mlxrs::Array` `Result` API a raw `.shape()[3]` would **panic** on a
/// wrong-rank input (the verified [medium] panic class of the merged
/// single-seq rotating cache). This validates the rank and returns a
/// recoverable [`Error::RankMismatch`] instead, never panicking. Kept
/// local so this PR is self-contained / panic-free; when the
/// `util::head_dim` hotfix lands this may switch to it (union-rebase).
pub(crate) fn batch_head_dim(name: &str, a: &Array) -> Result<usize> {
  let shape = a.shape();
  if shape.len() != KV_NDIM {
    let context: &'static str = match name {
      "keys" => "batch_head_dim: batched KV cache expects 4-D keys [B, n_kv_heads, S, head_dim]",
      "values" => {
        "batch_head_dim: batched KV cache expects 4-D values [B, n_kv_heads, S, head_dim]"
      }
      _ => "batch_head_dim: batched KV cache expects 4-D [B, n_kv_heads, S, head_dim]",
    };
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    )));
  }
  Ok(shape[KV_NDIM - 1])
}

/// Validate an `update`'s `keys`/`values` are a compatible KV pair, exactly
/// mirroring mlx-lm's implicit constraint: `BatchKVCache.update_and_fetch`
/// builds the V buffer with **keys'** `B`/`n_kv_heads` (only the head_dim
/// from `values`, `cache.py:945-949`) and then does
/// `self.values[..., prev:self._idx, :] = values` (`cache.py:964`) — that
/// in-place assignment **raises in mlx-lm** unless `values` matches the
/// slot's `[B_keys, n_kv_heads_keys, S_keys, *]`. mlx-lm therefore *does*
/// fail (inside `update_and_fetch`) on a `B`/`n_kv_heads`/`S`-mismatched
/// `values`; the only freedom is the head_dim (`v_head_dim =
/// values.shape[3]`). Our functional port's empty branch would otherwise
/// just clone the mismatched `values` and return `Ok`, silently
/// desynchronizing K/V — *less* faithful than mlx-lm. This restores
/// mlx-lm's exact error point as a recoverable [`Error::RankMismatch`] /
/// [`Error::ShapePairMismatch`]
/// (both 4-D; `values` `B`/`n_kv_heads`/`S` == `keys`'; head_dim free).
/// This is faithfulness parity, NOT extra validation beyond the reference.
pub(crate) fn validate_kv_compat(keys: &Array, values: &Array) -> Result<()> {
  let ks = keys.shape();
  let vs = values.shape();
  if ks.len() != KV_NDIM {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "batched KV cache expects 4-D keys [B, n_kv_heads, S, head_dim]",
      ks.len() as u32,
      ks.to_vec(),
    )));
  }
  if vs.len() != KV_NDIM {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "batched KV cache expects 4-D values [B, n_kv_heads, S, head_dim]",
      vs.len() as u32,
      vs.to_vec(),
    )));
  }
  // mlx-lm couples values to keys' B / n_kv_heads / S (head_dim free) via
  // the `new_v` buffer + `self.values[..., prev:_idx, :] = values` assign.
  if ks[0] != vs[0] || ks[1] != vs[1] || ks[2] != vs[2] {
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      "batched KV cache: values shape must match keys on [B, n_kv_heads, S] (head_dim free; mlx-lm raises at `self.values[..., prev:_idx, :] = values`)",
      vec![ks[0], ks[1], ks[2]],
      vec![vs[0], vs[1], vs[2]],
    )));
  }
  Ok(())
}

/// Build a 1-D `[B]` `I32` array from per-sequence ints (mlx-lm
/// `mx.array([... for ...])`). The padding/offset metadata are tiny
/// `[B]` integer vectors.
pub(crate) fn ivec(values: &[i32]) -> Result<Array> {
  Array::from_slice::<i32>(values, &(values.len(),))
}

/// `-l for l in left_padding` as an `I32` `[B]` array — mlx-lm
/// `self.offset = mx.array([-l for l in left_padding])` (`cache.py:937`).
fn neg_ivec(values: &[i32]) -> Result<Array> {
  let negated: Vec<i32> = values.iter().map(|&l| -l).collect();
  ivec(&negated)
}

/// Port of `mlx_lm.models.cache.dynamic_roll` (`cache.py:903-909`):
///
/// ```text
/// n = x.shape[axis]
/// expand_shifts  = (...,) + (None,) * (x.ndim - axis)
/// expand_indices = expand_shifts[:-1]
/// idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
/// rolled = mx.take_along_axis(x, idx, axis=axis)
/// ```
///
/// Every batched-cache caller passes a 4-D `x` (`[B, n_kv_heads, S,
/// head_dim]`), `axis = 2`, and `shifts` either shaped `[B, 1]` (per-row
/// shifts — `padding[:, None]` / `roll[:, None]`, `cache.py:983`/`1187`/
/// `1288`) OR shaped `[1, 1]` (scalar broadcast: every row gets the same
/// shift; arises from `BatchKvCache::finalize` arming a length-1
/// `right_padding` via `prepare_right_padding(&[k])` then `expand_dims`
/// to a `[1, 1]` `pad_col`). Then `expand_shifts = (..., None, None)`
/// makes `shifts[expand_shifts]` `[B, 1, 1, 1]` (or `[1, 1, 1, 1]` for
/// scalar broadcast) and `expand_indices = (..., None)` makes
/// `arange(n)[expand_indices]` `[S, 1]`, so `idx` broadcasts to
/// `[B, 1, S, 1]` and `take_along_axis(x, idx, 2)` (mlxrs broadcasts the
/// non-`axis` dims) yields per-row `out[b,:,i,:] = x[b,:,(i-shift[b])%S,:]`
/// — exactly `mx.roll` by `+shift[b]` per sequence.
///
/// Rank-validated (not raw-indexed): a non-4-D `x` or a mis-shaped
/// `shifts` is a recoverable [`Error::RankMismatch`] / [`Error::ShapePairMismatch`], never a panic.
/// `arange` is built through `f32` (the safe ops surface has no integer
/// `arange`) then cast to `I32`; `S` ≤ a tiny cache length, far below
/// `2^24`, so the round-trip is exact.
pub fn dynamic_roll(x: &Array, shifts: &Array, axis: i32) -> Result<Array> {
  // Callers are fixed (4-D, axis=2); validate rather than raw-index.
  let xshape = x.shape();
  if xshape.len() != KV_NDIM {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "dynamic_roll: x must be 4-D [B, n_kv_heads, S, head_dim]",
      xshape.len() as u32,
      xshape.to_vec(),
    )));
  }
  if axis != (KV_NDIM as i32) - 2 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "dynamic_roll: axis (must be the sequence axis)",
      "must equal KV_NDIM - 2 (the sequence axis = 2)",
      format_smolstr!("{axis}"),
    )));
  }
  let sshape = shifts.shape();
  // Split rank-first-then-shape (mirroring the `norm.rs` + `switch.rs`
  // patterns): a rank-1 or rank-3 `shifts`
  // would otherwise reach the collapsed guard and surface as
  // `ShapePairMismatch`, but `ShapePairMismatchPayload` is documented for
  // same-rank shape disagreement. Surface a divergent RANK as
  // `RankMismatch`; only after the rank is known to be 2 do we compare the
  // full `[B, 1]` shape.
  if sshape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "dynamic_roll: shifts must be rank 2 ([B, 1] or scalar broadcast [1, 1])",
      sshape.len() as u32,
      sshape.to_vec(),
    )));
  }
  // Accept the per-row shape `[B, 1]` OR the scalar broadcast `[1, 1]`:
  // `BatchKvCache::finalize` arms a length-1
  // `right_padding` via `prepare_right_padding(&[k])`, which becomes a
  // `[1, 1]` `pad_col` after the `expand_dims_axes` and must broadcast
  // across `keys`/`values`' batch dim — exactly the contract the existing
  // `batch_kv_finalize_with_scalar_right_padding_broadcasts_or_errs`
  // test pins. The leading dim must be `xshape[0]` OR `1`; the trailing
  // dim must be exactly `1` (matches mlx-lm `padding[:, None]`).
  let valid_b = sshape[0] == xshape[0] || sshape[0] == 1;
  if !valid_b || sshape[1] != 1 {
    // shifts must be `[B, 1]` (per-row) or `[1, 1]` (scalar broadcast).
    // Report the per-row expected shape — `[xshape[0], 1]` — as the
    // structured `expected` (the broadcast variant is a documented
    // relaxation, not an alternate expectation).
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      "dynamic_roll: shifts must be [B, 1] or [1, 1] (scalar broadcast)",
      vec![xshape[0], 1usize],
      sshape.to_vec(),
    )));
  }
  let n = xshape[KV_NDIM - 2];
  // `n == 0` is the empty-axis no-op: logically `roll([], k) == []` (mlx-lm
  // mx.roll on a zero-length axis also returns the input unchanged), and
  // computing `remainder(idx, 0)` below would be a divide-by-zero. Early
  // return a clone. This is symmetric with
  // the `n > 2^24` reject below: both are degenerate-`n` guards for cases
  // the reference's unbounded-int / overflow-defined semantics handle
  // implicitly but our finite-precision ops require explicit handling for.
  if n == 0 {
    return x.try_clone();
  }
  // The `Array::arange::<f32>(0.0, n as f32, 1.0)?` below builds the roll-index
  // range via f32 and casts to I32. f32 can represent consecutive integers
  // exactly only up to `2^24` (the mantissa precision limit); beyond that,
  // successive integers alias to the same f32 value and the cast back to
  // I32 silently produces wrong roll indices. Same aliasing class as
  // [`mask::iarange`](super::mask::iarange) and the local
  // `F32_EXACT_INT_MAX` guard in [`batch_rotating`](super::batch_rotating).
  // Reject `n > 2^24` up front with a recoverable `Error::OutOfRange`
  // rather than returning silently-wrong indices.
  const F32_EXACT_INT_MAX: usize = 1usize << 24;
  if n > F32_EXACT_INT_MAX {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "dynamic_roll: sequence axis n (arange/cast through f32 would silently alias indices and produce wrong rolls)",
      "must be <= 2^24 (f32 exact-integer limit)",
      format_smolstr!("{n}"),
    )));
  }
  // arange(n) -> [S]; expand_indices = (..., None) -> [S, 1].
  let ar = ops::misc::astype(&Array::arange::<f32>(0.0, n as f32, 1.0)?, Dtype::I32)?;
  let ar = ops::shape::expand_dims_axes(&ar, &[1])?; // [S, 1]
  // shifts[expand_shifts] = shifts[..., None, None]: [B,1] -> [B,1,1,1].
  let sh = ops::shape::expand_dims_axes(shifts, &[2, 3])?; // [B,1,1,1]
  // (arange - shifts) % n  -> broadcasts to [B, 1, S, 1].
  let diff = ops::arithmetic::subtract(&ar, &sh)?;
  let nscalar = ops::misc::astype(&Array::full::<f32>(&(1usize,), n as f32)?, Dtype::I32)?;
  let idx = ops::arithmetic::remainder(&diff, &nscalar)?; // [B,1,S,1]
  ops::indexing::take_along_axis(x, &idx, axis)
}

/// Left-padded batched full-attention KV cache — port of
/// `mlx_lm.models.cache.BatchKVCache` (`cache.py:912-1131`).
///
/// Expects inputs **left-padded** so every sequence shares the same length;
/// `left_padding[i]` is sequence `i`'s pad count. `offset` /
/// `left_padding` are per-sequence `[B]` arrays (the RoPE / mask metadata);
/// `_idx` is the scalar logical length. The step buffer is *not*
/// materialized (see the module docs): the observable `update_and_fetch`
/// is a sequence-axis concat of every update.
pub struct BatchKvCache {
  keys: Option<Array>,
  values: Option<Array>,
  /// Per-sequence left-pad counts — mlx-lm `BatchKVCache.left_padding`
  /// (`[B]`, `I32`).
  left_padding: Array,
  /// Cached host-readable mirror of `left_padding`, set in lockstep with
  /// the `Array` form (#101). mlx-lm's `int(self.left_padding[i]
  /// .item())` pattern (`cache.py:947-955` / swift `leftPaddingValues:
  /// asArray(Int.self)`, `KVCache.swift:1223-1226`) round-trips each
  /// scalar through the GPU→CPU boundary inside a hot loop. Caching the
  /// `Vec<i32>` once at construction / `set_state` time lets future
  /// consumers reach `pad_lengths() -> &[i32]` without re-evaling the
  /// array per call — exactly mirroring [`super::ArraysCache`]'s
  /// `left_padding: Option<Vec<i32>>` value-side representation (the only
  /// host-extracted view that pattern needs is a borrowed slice; broadcast
  /// `[B,1]` arrays are still rebuilt on demand from the cached values, no
  /// array/values dual-state to keep in sync). Always populated (the
  /// `Array` is the source of truth for ops; the `Vec<i32>` is a host-side
  /// projection updated together).
  pad_lengths: Vec<i32>,
  /// Per-sequence raw position — mlx-lm `BatchKVCache.offset` (`[B]`,
  /// `I32`; starts at `-left_padding`, the per-seq RoPE/mask offset).
  offset: Array,
  /// Scalar logical sequence length — mlx-lm `BatchKVCache._idx`.
  idx: usize,
  /// Deferred right-pad counts set by [`Self::prepare_right_padding`],
  /// applied by [`Self::finalize`] — mlx-lm `BatchKVCache._right_padding`.
  right_padding: Option<Array>,
  /// Cached host-readable mirror of `right_padding` (#101). Set in
  /// lockstep with the `Array` form by [`Self::prepare_right_padding`] so
  /// [`Self::finalize`] can update `pad_lengths` without re-evaling the
  /// array. Cleared together with `right_padding` (a `None` means there is
  /// no pending right_padding to apply, mirroring the `Array` field's
  /// optionality).
  right_padding_host: Option<Vec<i32>>,
}

impl BatchKvCache {
  /// A new empty left-padded batched cache — mlx-lm
  /// `BatchKVCache(left_padding)` (`cache.py:915-940`): `offset =
  /// array([-l..])`, `left_padding = array(left_padding)`, `_idx = 0`.
  ///
  /// `mx.array([...])` is fallible here only on allocation/backend; a tiny
  /// `[B]` integer vector cannot realistically fail, so `new` keeps
  /// mlx-lm's infallible `__init__` signature and on the (unreachable)
  /// failure falls back to an empty `[0]` array — still **no** panic and
  /// **no** heap leak on this constructor path (the `[-l..]` map reads the
  /// caller's own slice directly, never a re-read of the built array).
  pub fn new(left_padding: &[i32]) -> Self {
    let lp = ivec(left_padding).unwrap_or_else(|_| empty_ivec());
    let offset = neg_ivec(left_padding).unwrap_or_else(|_| empty_ivec());
    Self {
      keys: None,
      values: None,
      left_padding: lp,
      pad_lengths: left_padding.to_vec(),
      offset,
      idx: 0,
      right_padding: None,
      right_padding_host: None,
    }
  }

  /// Per-sequence left-pad counts as a borrowed `&[i32]` — the cached
  /// host-readable mirror of [`left_padding_arr`](Self::left_padding_arr)
  /// (#101). mlx-lm's `int(self.left_padding[i].item())` per-batch-
  /// entry GPU→CPU round-trip (`cache.py:947-955`) is replaced by a
  /// borrowed slice into the cached `Vec<i32>` — kept in lockstep with the
  /// underlying `Array` form. Reuses the exact same accessor name as
  /// [`super::ArraysCache::left_padding`] (which already returns
  /// `Option<&[i32]>` for the same value-side reason) for cross-cache
  /// consistency; the `BatchKvCache` variant is always populated (no
  /// `Option` wrapper) — the constructor always builds it.
  pub fn pad_lengths(&self) -> &[i32] {
    &self.pad_lengths
  }

  /// mlx-lm `BatchKVCache.prepare(right_padding=...)` (`cache.py:977-978`):
  /// store a non-zero `right_padding` to be applied by [`Self::finalize`]
  /// (mlx-lm only stores it when `max(right_padding) > 0`). Left-padding
  /// `prepare` (`cache.py:968-975`) is the constructor here.
  pub fn prepare_right_padding(&mut self, right_padding: &[i32]) -> Result<()> {
    if right_padding.iter().copied().max().unwrap_or(0) > 0 {
      // Build the Array FIRST (fallible) before any field mutation, so an
      // ivec allocation failure leaves both `right_padding` and
      // `right_padding_host` unchanged (no half-armed state).
      let rp = ivec(right_padding)?;
      self.right_padding = Some(rp);
      // #101: keep the cached host mirror in lockstep — `finalize`
      // uses it to update `pad_lengths` without re-evaling the array.
      self.right_padding_host = Some(right_padding.to_vec());
    }
    Ok(())
  }

  /// mlx-lm `BatchKVCache.finalize` (`cache.py:980-987`): if a
  /// `_right_padding` is pending, `dynamic_roll` keys/values right by it,
  /// `offset -= padding`, `left_padding += padding`, clear it.
  pub fn finalize(&mut self) -> Result<()> {
    // Borrow (do NOT `take`) the pending padding so a fallible step does
    // not consume it on the error path. Stage-then-commit: all fallible
    // ops compute into locals; `self.*` (including clearing
    // `right_padding`) is mutated only in the infallible tail. So an `Err`
    // (e.g. the `values` roll failing on a batch-mismatched restored
    // cache, after the `keys` roll) leaves keys/values/offset/
    // left_padding/right_padding EXACTLY as they were — retry-safe, no
    // keys-rolled-but-values-not desync and no lost `right_padding`.
    if let Some(padding) = &self.right_padding {
      // **Stale `pad_lengths` guard.** A naive implementation would
      // silently skip the host mirror update when
      // `right_padding_host.len() != self.pad_lengths.len()` and
      // continue committing `left_padding`/`right_padding=None`,
      // leaving `pad_lengths()` permanently stale (a length-1 padding
      // vector broadcasts across the `[B]` array op, so the commit
      // succeeds with the mirror frozen at the OLD values). So:
      // validate the host length FIRST — BEFORE any Array op work or
      // commit — so the failure path does ZERO wasted ops and leaves
      // the cache exactly as it was. Three supported cases:
      //
      //   * `rp_host.len() == pad_lengths.len()` (the common B==B case):
      //     elementwise add, byte-identical to the Array `add` below.
      //   * `rp_host.len() == 1` AND `pad_lengths.len() >= 1`: scalar
      //     broadcast — `pad_lengths[i] += rp_host[0]` for all i. This
      //     mirrors `ops::arithmetic::add(&left_padding[B], &padding[1])`
      //     on the Array side (MLX broadcasts length-1).
      //   * any other mismatch (`rp_host.len() != 1 && !=
      //     pad_lengths.len()`): return Err — the Array side may broadcast
      //     in ways the host mirror cannot reproduce, and we will NOT
      //     leave the cache with a desynchronized `pad_lengths`.
      //
      // `wrapping_add` keeps a corrupt-restored Vec arithmetic from
      // panicking on i32 overflow; the actual shape mismatch (if any)
      // surfaces through the Array op below, not the Vec.
      let new_pad_lengths = match self.right_padding_host.as_ref() {
        None => self.pad_lengths.clone(),
        Some(rp_host) if rp_host.len() == self.pad_lengths.len() => self
          .pad_lengths
          .iter()
          .zip(rp_host)
          .map(|(&a, &b)| a.wrapping_add(b))
          .collect::<Vec<i32>>(),
        Some(rp_host) if rp_host.len() == 1 => {
          // Scalar broadcast: every batch entry gets the same right-pad.
          let b = rp_host[0];
          self
            .pad_lengths
            .iter()
            .map(|&a| a.wrapping_add(b))
            .collect::<Vec<i32>>()
        }
        Some(rp_host) => {
          // The runtime length must either match pad_lengths.len() exactly OR
          // be 1 (scalar broadcast). Surface as LengthMismatch with the
          // expected = pad_lengths length and the actual = observed right_padding length.
          return Err(Error::LengthMismatch(LengthMismatchPayload::new(
            "BatchKvCache::finalize: right_padding length (must equal pad_lengths length or be a length-1 scalar broadcast — refusing to commit a desynchronized pad_lengths host mirror)",
            self.pad_lengths.len(),
            rp_host.len(),
          )));
        }
      };
      // padding[:, None] -> [B, 1].
      let pad_col = ops::shape::expand_dims_axes(padding, &[1])?;
      let rolled = match (&self.keys, &self.values) {
        (Some(k), Some(v)) => Some((dynamic_roll(k, &pad_col, 2)?, dynamic_roll(v, &pad_col, 2)?)),
        _ => None,
      };
      let new_offset = ops::arithmetic::subtract(&self.offset, padding)?;
      let new_left_padding = ops::arithmetic::add(&self.left_padding, padding)?;
      // ── Infallible commit tail ────────────────────────────────────
      if let Some((nk, nv)) = rolled {
        self.keys = Some(nk);
        self.values = Some(nv);
      }
      self.offset = new_offset;
      self.left_padding = new_left_padding;
      self.pad_lengths = new_pad_lengths;
      self.right_padding = None;
      self.right_padding_host = None;
    }
    Ok(())
  }

  /// mlx-lm `BatchKVCache.left_padding` — the per-sequence `[B]` left-pad
  /// counts (an owned clone; `Array::try_clone` is fallible per #33).
  pub fn left_padding_arr(&self) -> Result<Array> {
    self.left_padding.try_clone()
  }

  /// mlx-lm `BatchKVCache.state` getter `(keys, values)` pair (without the
  /// `offset`/`left_padding` metadata), `_idx`-sliced when the buffer
  /// over-allocated (`cache.py:991-995`). Test/inspection convenience.
  pub fn state_kv(&self) -> Result<(Array, Array)> {
    match (&self.keys, &self.values) {
      (Some(k), Some(v)) => Ok((slice_seq(k, 0, self.idx)?, slice_seq(v, 0, self.idx)?)),
      _ => Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "BatchKvCache::state_kv",
        "must be called on a non-empty cache (keys/values both Some)",
      ))),
    }
  }
}

/// An empty `[0]`-length `I32` array — the unreachable [`BatchKvCache::new`]
/// allocation-failure fallback (keeps the constructor panic-free without
/// changing observable behavior for any realistic input: a `[B]` int
/// vector build does not fail in practice).
fn empty_ivec() -> Array {
  Array::from_slice::<i32>(&[], &(0usize,)).unwrap_or_else(|_| {
    // Terminal, infallible, no-eval: a fresh empty handle (NULL ctx) — the
    // exact `mlx_array_new()` out-param idiom the ops use. Reached only on
    // the impossible double allocation failure; never panics, never evals.
    // SAFETY: `mlx_array_new()` returns a fresh owned empty handle per the
    // mlx-c convention; moved straight into the RAII `Array` newtype so it
    // is freed exactly once on drop.
    Array(unsafe { mlxrs_sys::mlx_array_new() })
  })
}

impl KvCache for BatchKvCache {
  /// mlx-lm `BatchKVCache.make_mask` uses `offset=self._idx`
  /// (`cache.py:1013`) and `create_causal_mask`'s scalar grid is built
  /// from `self._idx`; the scalar `offset()` is therefore `_idx` (the
  /// per-sequence RoPE position is the `[B]`
  /// [`BatchPositionedKvCache::batch_offset`] /
  /// [`rope_offset`](KvCache::rope_offset) `Batch`, not this scalar).
  fn offset(&self) -> usize {
    self.idx
  }

  /// mlx-lm `BatchKVCache.update_and_fetch` (`cache.py:942-965`). The
  /// step-buffer growth (`cache.py:944-959`) is a pure allocation
  /// optimization with no observable effect (module docs): every returned
  /// row is a written row, the zero rows are sliced off, and there is no
  /// in-place ring overwrite — so the observable result is the sequence
  /// concat of every update. `offset += S` / `_idx += S` mirror
  /// `cache.py:961-962`.
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    let s = seq_len("keys", keys)?;
    // Both 4-D AND `values` B/n_kv_heads/S == keys' (head_dim free) — the
    // exact constraint mlx-lm's `self.values[..., prev:_idx, :] = values`
    // (cache.py:964) implicitly asserts; restores mlx-lm's error point
    // (else the empty branch would clone a mismatched `values` and desync
    // K/V) and is also the rank-safety guard (no `.shape()[N]` panic).
    validate_kv_compat(keys, values)?;

    // Stage-then-commit (same class-wide contract as the batch-rotating
    // cache): every fallible op (`concat_seq`/`checked_add`/`astype`/
    // `add`/`try_clone`) computes into a local FIRST; `self.*` is mutated
    // only in the infallible tail. So an `Err` from any step leaves the
    // cache fully unmutated — no offset-advanced-but-buffer-not desync.
    let (k, v) = match (&self.keys, &self.values) {
      (Some(pk), Some(pv)) => (concat_seq(pk, keys)?, concat_seq(pv, values)?),
      _ => (keys.try_clone()?, values.try_clone()?),
    };

    // offset += keys.shape[2]; _idx += keys.shape[2] (cache.py:961-962).
    // `_idx` is bounded by the tiny test/decode lengths; a corrupt restored
    // `_idx` near usize::MAX could overflow on add, so guard it (the value
    // is byte-identical to `self.idx + s` for every realistic input).
    let new_idx = self.idx.checked_add(s).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "BatchKvCache::update: _idx + S",
        "usize",
        [("_idx", self.idx as u64), ("S", s as u64)],
      ))
    })?;
    let s_scalar = ops::misc::astype(&Array::full::<f32>(&(1usize,), s as f32)?, Dtype::I32)?;
    let new_offset = ops::arithmetic::add(&self.offset, &s_scalar)?;
    let (rk, rv) = (k.try_clone()?, v.try_clone()?);
    // ── Infallible commit tail ──────────────────────────────────────
    self.offset = new_offset;
    self.idx = new_idx;
    self.keys = Some(k);
    self.values = Some(v);
    Ok((rk, rv))
  }

  /// mlx-lm `BatchKVCache.state` getter (`cache.py:989-995`):
  /// `[keys[:_idx], values[:_idx], offset, left_padding]`; `[]` when empty
  /// (mlx-lm returns the four-tuple; an empty cache has `keys=None`).
  fn state(&self) -> Result<Vec<Array>> {
    match (&self.keys, &self.values) {
      (Some(k), Some(v)) => Ok(vec![
        slice_seq(k, 0, self.idx)?,
        slice_seq(v, 0, self.idx)?,
        self.offset.try_clone()?,
        self.left_padding.try_clone()?,
      ]),
      _ => Ok(Vec::new()),
    }
  }

  /// Force-evaluate the cache's own stored arrays in place — the per-chunk
  /// prefill memory barrier (see [`KvCache::materialize`]).
  ///
  /// Evals the genuine stored arrays via the explicit `&mut` [`Array::eval`]:
  /// the `self.keys`/`self.values` step buffers (the arrays the next
  /// `update`/`finalize` reads and splices into — **not** the
  /// `slice_seq(k, 0, self.idx)` views [`state`](KvCache::state) returns),
  /// plus the per-sequence `self.offset`/`self.left_padding`/
  /// `self.right_padding` position arrays (themselves lazy `[B]` graphs — e.g.
  /// `offset` is a lazy `negative(left_padding)` after an empty `set_state` —
  /// that would otherwise chain across chunks). Materializes every live
  /// buffer the next chunk reuses; `keys`/`values`/`right_padding` are no-ops
  /// when absent.
  fn materialize(&mut self) -> Result<()> {
    if let Some(k) = self.keys.as_mut() {
      k.eval()?;
    }
    if let Some(v) = self.values.as_mut() {
      v.eval()?;
    }
    self.offset.eval()?;
    self.left_padding.eval()?;
    if let Some(rp) = self.right_padding.as_mut() {
      rp.eval()?;
    }
    Ok(())
  }

  /// mlx-lm `BatchKVCache.state` setter (`cache.py:997-1000`):
  /// `keys, values, offset, left_padding = v; _idx = keys.shape[2]`. An
  /// empty state resets the cache.
  fn set_state(&mut self, mut state: Vec<Array>) -> Result<()> {
    match state.len() {
      0 => {
        // Fully-fresh state, mirroring `BatchKvCache::new(&self.left_padding)`:
        // an empty `set_state` MUST clear ALL per-seq runtime state, not just
        // the buffer + `_idx`. Otherwise `offset` (per-seq `[B]` = current
        // RoPE positions) and `right_padding` (pending finalize) survive as
        // stale metadata into a logically-fresh cache — the next `update`
        // mismatches its `[B]` against fresh inputs and the next `finalize`
        // re-applies a dropped right-pad. `left_padding` is preserved (it is
        // the constructor *input* — `new(&self.left_padding)` would feed the
        // same slice). `offset = -left_padding` is reproduced via a pure
        // `ops::negative` (no eval, no host extraction); the fallible op
        // is staged BEFORE any `self.*` mutation so a backend `Err` leaves
        // the cache unmutated.
        let new_offset = ops::arithmetic::negative(&self.left_padding)?;
        // ── Infallible commit tail ──────────────────────────────────────
        self.keys = None;
        self.values = None;
        self.idx = 0;
        self.offset = new_offset;
        self.right_padding = None;
        self.right_padding_host = None;
        // `pad_lengths` mirrors `left_padding`, which is preserved on the
        // empty-state path — so the cached host mirror is left untouched.
        Ok(())
      }
      4 => {
        // [keys, values, offset, left_padding]; pop in reverse.
        let left_padding = state.pop().unwrap();
        let offset = state.pop().unwrap();
        let values = state.pop().unwrap();
        let keys = state.pop().unwrap();
        // mlx-lm derives `_idx` from `keys.shape[2]` (cache.py:1000); a
        // rank-invalid `keys` is a recoverable error, not a panic.
        // `values` is rank-validated too (NOT done by mlx-lm's numpy
        // setter, but required here): a hostile/corrupt prompt cache could
        // otherwise restore a 4-D `keys` with a rank-<3 `values`, which a
        // later `state()`/`make_mask` would raw-index on the seq axis
        // (`slice_seq` / `create_causal_mask`) and PANIC on the `Result`
        // API. We mirror mlx-lm's "no K/V *shape-compatibility*
        // validation" (the head dim may legitimately differ; we do not
        // cross-check B/H/S), only enforcing the 4-D rank invariant the
        // rest of this module relies on so the failure is a recoverable
        // `Error::RankMismatch`, never a panic. Validate BEFORE assigning
        // any field so a bad `values` leaves the cache unmutated.
        let sk = seq_len("keys", &keys)?;
        batch_head_dim("values", &values)?;
        // #101: materialize the restored `left_padding` to a host
        // `Vec<i32>` mirror ONCE here, at restore time — replaces the
        // per-call `.item()` round-trip mlx-lm's `int(self.left_padding
        // [i].item())` would do on every consumer access. This is the
        // single eval pay-point for restored state; subsequent accesses
        // via [`Self::pad_lengths`] are zero-cost borrows. Staged on a
        // local FIRST (eval can fail on a backend error) so a failed
        // host extraction leaves the cache fully unmutated.
        //
        // **Propagate extraction failures.** A naive implementation would
        // fall back to `self.pad_lengths.clone()` on a
        // non-1-D / non-I32 / non-contiguous restored `left_padding`,
        // then commit `self.left_padding` to the new (corrupt) Array
        // anyway — leaving `pad_lengths()` permanently desynchronized
        // (and often at the empty placeholder from `BatchKvCache::new(
        // &[])`, which `from_state("BatchKVCache")` opens with). So:
        // VALIDATE rank/dtype against the restored `keys`'s batch dim
        // before extracting, then propagate any `to_vec::<i32>` error
        // via `?` — the cache is left fully unmutated on every error
        // path. The restored `left_padding` MUST be a 1-D I32 vector
        // whose length equals the `keys` batch dim (`keys.shape[0]`);
        // any deviation rejects the restore as a recoverable
        // `Error::RankMismatch` / `Error::LengthMismatch`.
        let lp_shape = left_padding.shape();
        let kb = keys.shape()[0];
        if lp_shape.len() != 1 {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "BatchKvCache::set_state: restored left_padding must be 1-D [B]",
            lp_shape.len() as u32,
            lp_shape.to_vec(),
          )));
        }
        if lp_shape[0] != kb {
          return Err(Error::LengthMismatch(LengthMismatchPayload::new(
            "BatchKvCache::set_state: restored left_padding length vs keys batch dim",
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
        // dtype, plus runs the single eval. Propagate every failure
        // (dtype-mismatch / non-contiguous / OOM / backend) — the
        // cache stays untouched.
        let mut lp_clone = left_padding.try_clone()?;
        let new_pad_lengths = lp_clone.to_vec::<i32>()?;
        // ── Infallible commit tail ──────────────────────────────────────
        self.keys = Some(keys);
        self.values = Some(values);
        self.offset = offset;
        self.left_padding = left_padding;
        self.pad_lengths = new_pad_lengths;
        self.idx = sk;
        // Also clear `right_padding` here, matching the empty-state branch
        // above. `set_state` fully defines
        // the cache's runtime state: leaving a previously-armed
        // `right_padding` from a prior `prepare_right_padding` call would
        // make the next `finalize()` unexpectedly roll the freshly-restored
        // buffers using stale padding. mlx-lm doesn't have this problem
        // because its `state` setter (cache.py:940) is called as part of
        // `from_state`'s fresh-cache reconstruction, so `_right_padding`
        // is `None` by construction; mlxrs's setter is callable
        // out-of-band so we explicitly drop the stale field.
        self.right_padding = None;
        self.right_padding_host = None;
        Ok(())
      }
      n => Err(Error::OutOfRange(OutOfRangePayload::new(
        "BatchKvCache::set_state: state array count",
        "must be 0 or 4",
        format_smolstr!("{n}"),
      ))),
    }
  }

  /// mlx-lm `BatchKVCache.is_trimmable` — always `True` (`cache.py:1002`).
  fn is_trimmable(&self) -> bool {
    true
  }

  /// mlx-lm `BatchKVCache.trim` (`cache.py:1005-1009`):
  /// `n = min(_idx, n); _idx -= n; offset -= n; return n`. Drops the
  /// stored buffer's tail too so a later [`update`](KvCache::update)
  /// extends the trimmed prefix (the over-allocation is invisible).
  fn trim(&mut self, n: usize) -> Result<usize> {
    let trimmed = n.min(self.idx);
    if trimmed == 0 {
      return Ok(0);
    }
    // Stage-then-commit: the new `_idx`, the `offset -= n` array, and the
    // sliced buffers are computed into locals FIRST; `self.*` is mutated
    // only in the infallible tail. So a fallible-op `Err` (e.g. the
    // `subtract`/`slice_seq`) leaves `_idx`/`offset`/`keys`/`values`
    // exactly as they were (no idx-decremented-but-buffer-not desync).
    let new_idx = self.idx - trimmed;
    let nscalar = ops::misc::astype(&Array::full::<f32>(&(1usize,), trimmed as f32)?, Dtype::I32)?;
    let new_offset = ops::arithmetic::subtract(&self.offset, &nscalar)?;
    let sliced = match (&self.keys, &self.values) {
      (Some(k), Some(v)) => Some((slice_seq(k, 0, new_idx)?, slice_seq(v, 0, new_idx)?)),
      _ => None,
    };
    // ── Infallible commit tail ──────────────────────────────────────
    self.idx = new_idx;
    self.offset = new_offset;
    if let Some((nk, nv)) = sliced {
      self.keys = Some(nk);
      self.values = Some(nv);
    }
    Ok(trimmed)
  }

  /// mlx-lm `BatchKVCache.make_mask` (`cache.py:1011-1014`) — its **own**
  /// override, NOT the generic `create_attention_mask`:
  /// `create_causal_mask(N, offset=self._idx, left_padding=self.left_padding,
  /// window_size=...)`. Returns the per-sequence left-padded causal
  /// (optionally windowed) `[B, 1, N, _idx + N]` mask, always materialized
  /// (the `left_padding` term needs the array). A single-token decode
  /// still produces the left-padded array (mlx-lm does not special-case
  /// `N == 1` here — it always calls `create_causal_mask`).
  fn make_mask(
    &self,
    n: usize,
    window_size: Option<usize>,
    _return_array: bool,
  ) -> Result<MaskMode> {
    Ok(MaskMode::Array(create_causal_mask_batched(
      n,
      self.idx,
      window_size,
      None,
      Some(&self.left_padding),
    )?))
  }

  /// mlx-lm `BatchKVCache.nbytes` (`cache.py:1126-1130`):
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

  /// mlx-lm `BatchKVCache.empty` (`cache.py:1123-1124`): `keys is None`.
  fn is_empty(&self) -> bool {
    self.keys.is_none()
  }

  /// An independent copy (mlx-lm `copy.deepcopy`). MLX value semantics:
  /// arrays are immutable and the cache only ever *reassigns* its arrays
  /// (never mutates a buffer in place), so a refcount-sharing
  /// `Array::try_clone` still yields a fully independent cache. The
  /// fallible clone is propagated as a `Result` — a failure is **never**
  /// swallowed (silent corruption) and **never** panicked.
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
      idx: self.idx,
      right_padding: match &self.right_padding {
        Some(a) => Some(a.try_clone()?),
        None => None,
      },
      right_padding_host: self.right_padding_host.clone(),
    }))
  }

  /// This cache is batch-positioned (swift
  /// `cache as? BatchPositionedKVCache`); the merged
  /// [`rope_offset`](KvCache::rope_offset) default then yields
  /// `RopeOffset::Batch(batch_offset())`.
  fn as_batch_positioned(&self) -> Option<&dyn BatchPositionedKvCache> {
    Some(self)
  }

  /// `"BatchKVCache"` — mlx-lm's `type(BatchKVCache).__name__`
  /// (`cache.py:56`, written by `save_prompt_cache`; the load side accepts
  /// both this canonical name and the Rust alias `"BatchKvCache"`, see
  /// [`super::from_state`]). mlx-swift-lm has no `BatchKVCache` arm in its
  /// `cacheClassName` switch (`KVCache.swift:1381-1392`) — batch caches are
  /// mlx-lm-only — so the kind label is taken from mlx-lm verbatim.
  fn reference_class_name(&self) -> &'static str {
    "BatchKVCache"
  }

  /// Per-layer fast-path downcast target (#110) — see the
  /// [`KvCache`]-trait doc's **Per-layer fast-path convention**.
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }

  /// Transactional override of [`KvCache::from_serialized`] — leaves `self`
  /// byte-identical to its pre-call state on every recoverable error
  /// (`set_state` arity/rank failures; the no-meta default rejects a
  /// non-empty `meta`). All fallible work runs on a fresh placeholder
  /// `BatchKvCache::new(&[])` (the exact placeholder the existing
  /// [`super::from_state`] dispatch uses; the per-seq `[B]` arrays
  /// `offset`/`left_padding` are overwritten by `set_state`'s 4-array
  /// branch). `self` is committed by a single infallible move only after
  /// both setters succeed. The default trait impl would mutate the fresh
  /// state arrays via `set_state` first; even though `BatchKvCache` has no
  /// custom meta parser (the default `set_meta_state` only errors on a
  /// non-empty meta), the override is still important so a corrupt prompt
  /// cache that hands `BatchKVCache` a non-empty `meta` cannot leave the
  /// (otherwise valid) restored state assigned while the cache is reported
  /// as having errored.
  fn from_serialized(&mut self, state: Vec<Array>, meta: &[String]) -> Result<()> {
    let mut staged = BatchKvCache::new(&[]);
    staged.set_state(state)?;
    staged.set_meta_state(meta)?;
    *self = staged;
    Ok(())
  }
}

impl BatchPositionedKvCache for BatchKvCache {
  /// Per-sequence RoPE offsets `[B]` — mlx-lm `BatchKVCache.offset`
  /// (swift `batchOffset`); an owned clone (fallible per #33).
  fn batch_offset(&self) -> Result<Array> {
    self.offset.try_clone()
  }
}

/// Port of `mlx_lm.models.base.create_causal_mask` (`base.py:24-42`) — the
/// **batched** form the batch caches' `make_mask` overrides need, i.e.
/// including the `left_padding` / `right_padding` per-sequence terms (the
/// scalar-only subset is `super::mask::create_causal_mask`):
///
/// ```text
/// rinds = mx.arange(offset + N)
/// linds = mx.arange(offset, offset + N) if offset else rinds
/// linds = linds[:, None]; rinds = rinds[None]
/// mask  = linds >= rinds
/// if window_size is not None:  mask &= linds < rinds + window_size
/// if right_padding is not None:
///     mask &= rinds < expand_dims((offset+N) - right_padding, (1,2,3))
/// if left_padding is not None:
///     mask &= expand_dims(left_padding, (1,2,3)) <= rinds
/// ```
///
/// Result `[B, 1, N, offset+N]` (broadcast from the `[1,N,offset+N]`
/// causal grid against the `[B,1,1,1]` padding terms). `offset + N` is
/// computed with [`usize::checked_add`] before any range is built so a
/// hostile restored `offset` is a recoverable [`Error::ArithmeticOverflow`],
/// never an overflow panic / silent wrong mask.
pub(crate) fn create_causal_mask_batched(
  n: usize,
  offset: usize,
  window_size: Option<usize>,
  right_padding: Option<&Array>,
  left_padding: Option<&Array>,
) -> Result<Array> {
  use crate::lm::cache::mask::{iarange, scalar_i32};
  let total = offset.checked_add(n).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "create_causal_mask_batched: offset + N",
      "usize",
      [("offset", offset as u64), ("N", n as u64)],
    ))
  })?;
  let rinds = iarange(0, total)?;
  let linds = if offset != 0 {
    iarange(offset, total)?
  } else {
    rinds.try_clone()?
  };
  // linds[:, None] / rinds[None].
  let linds = ops::shape::expand_dims_axes(&linds, &[1])?;
  let rinds = ops::shape::expand_dims_axes(&rinds, &[0])?;

  let mut mask = ops::comparison::greater_equal(&linds, &rinds)?;
  if let Some(w) = window_size
    && w < total
  {
    // mlx-lm: a `window_size >= total` is the unbounded-Python-int no-op
    // (every `linds < rinds + w` already holds); skip the term so a huge
    // `w` cannot lossily cast through `as i32`.
    //
    // For `w < total`, also guard against `w` itself exceeding `i32::MAX`
    // before the cast (`w` is `usize`, on 64-bit it can be > 2^31-1).
    // Use `i32::try_from(w)` to surface a recoverable `Error::ArithmeticOverflow`
    // instead of silently wrapping to a negative value, which would
    // produce a wrong (inverted) windowed mask. The `w < total <
    // i32::MAX` invariant *usually* holds (total derives from real seq
    // lengths), but the defensive cast costs nothing and closes the
    // wrap-on-cast hole.
    let w_i32 = i32::try_from(w).map_err(|_| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "create_causal_mask_batched: window_size exceeds i32::MAX (cannot fit into a scalar mask offset)",
        "i32",
        [("window_size", w as u64)],
      ))
    })?;
    let bound = ops::arithmetic::add(&rinds, &scalar_i32(w_i32)?)?;
    let windowed = ops::comparison::less(&linds, &bound)?;
    mask = ops::logical::logical_and(&mask, &windowed)?;
  }
  if let Some(rp) = right_padding {
    // rinds < expand_dims((offset+N) - right_padding, (1,2,3))
    //
    // Build the `total` scalar via the integer-exact `scalar_i32` helper
    // instead of round-tripping through `f32`.
    // For `total > 2^24`, an `f32` cast would lossily round (consecutive
    // integers alias) and silently produce a wrong right-padding bound,
    // hence a wrong mask. `scalar_i32` builds the I32 scalar directly with
    // no f32 intermediate — the same discipline `mask::iarange` and the
    // windowed `w_i32` cast above already use. Reject `total > i32::MAX`
    // with a recoverable `Error::ArithmeticOverflow` so an overflowing prefill
    // never silently corrupts the mask.
    let total_i32 = i32::try_from(total).map_err(|_| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "create_causal_mask_batched: total exceeds i32::MAX (cannot fit into a scalar mask offset)",
        "i32",
        [("total", total as u64)],
      ))
    })?;
    let total_s = scalar_i32(total_i32)?;
    let bound = ops::arithmetic::subtract(&total_s, rp)?; // [B]
    let bound = ops::shape::expand_dims_axes(&bound, &[1, 2, 3])?; // [B,1,1,1]
    let term = ops::comparison::less(&rinds, &bound)?;
    mask = ops::logical::logical_and(&mask, &term)?;
  }
  if let Some(lp) = left_padding {
    // expand_dims(left_padding, (1,2,3)) <= rinds
    let lp = ops::shape::expand_dims_axes(lp, &[1, 2, 3])?; // [B,1,1,1]
    let term = ops::comparison::greater_equal(&rinds, &lp)?; // lp <= rinds
    mask = ops::logical::logical_and(&mask, &term)?;
  }
  Ok(mask)
}
