//! Attention-mask construction, ported 1:1 from mlx-lm.
//!
//! - [`create_causal_mask`] is `mlx_lm.models.base.create_causal_mask`
//!   (the offset + optional sliding-`window_size` subset PR-0 needs; the
//!   `left/right_padding` batched args land with the batched caches in a
//!   later PR).
//! - [`create_attention_mask`] is `mlx_lm.models.cache.create_attention_mask`
//!   (`cache.py:114-126`): it returns the symbolic [`MaskMode::Causal`] when
//!   a materialized array is unnecessary, mirroring mlx-lm's `"causal"`
//!   sentinel, and cross-checked against mlx-swift-lm's
//!   `ScaledDotProductAttentionMaskMode`.
//!
//! No implicit eval: every op is a pure [`crate::ops`] composition.

use crate::{
  array::Array,
  dtype::Dtype,
  error::{ArithmeticOverflowPayload, Error, OutOfRangePayload, Result},
  lm::cache::MaskMode,
  ops,
};
use smol_str::format_smolstr;

/// The largest integer `f32` represents exactly. `f32` has a 24-bit
/// significand, so every integer in `[0, 2^24]` round-trips losslessly;
/// `2^24 + 1` is the first integer that aliases (it shares its bit pattern
/// with `2^24`, rounding *down* to `2^24`). [`iarange`] builds positions
/// through `f32` ([`Array::arange`] is `f32`-only) **and** casts its own
/// exclusive `stop` to `f32`, so `stop > 2^24` would round the bound and
/// silently corrupt (truncate) the causal/window mask.
const F32_EXACT_INT_MAX: usize = 1usize << 24;

/// A 1-element `I32` scalar (mlx-lm's weak Python int `window_size`),
/// broadcast against the index grids — built without eval.
pub(crate) fn scalar_i32(value: i32) -> Result<Array> {
  ops::misc::astype(&Array::full::<f32>(&(1usize,), value as f32)?, Dtype::I32)
}

/// 1-D `I32` `[start, stop)` index vector — mlx-lm's `mx.arange(...)`
/// (integer).
///
/// [`Array::arange`] is `f32`-only (the safe ops surface has no integer
/// `arange`; adding one is out of this PR's scope), so positions are built
/// through `f32` and cast back to `I32`. Crucially, the **exclusive `stop`
/// itself is cast to `f32`** to call `Array::arange::<f32>(start, stop, 1.0)`. `f32`
/// represents every integer in `[0, 2^24]` exactly (24-bit significand) and
/// rounds `2^24 + 1` *down* to `2^24`. So the bound rejects `stop > 2^24`
/// (strictly): for the maximum allowed `stop == 2^24`, the `stop` cast is
/// exact (so the element count `stop - start` is exact) **and** every
/// produced value lies in `[start, stop - 1] ⊆ [0, 2^24 - 1]`, each exactly
/// representable, so the `f32 -> I32` round-trip is lossless and the result
/// feeds an `I32`/`Bool` grid identical to mlx-lm's. Were the bound only on
/// the largest produced value (`stop - 1 > 2^24`), `stop == 2^24 + 1` would
/// pass yet `(2^24 + 1) as f32 == 2^24`, so `arange` would stop one element
/// short and silently emit a too-short (corrupt) mask. An out-of-range
/// `stop` is therefore **rejected** (a recoverable [`Error::OutOfRange`])
/// rather than truncated — a too-long cache context surfaces an error here
/// instead of a wrong mask.
pub(crate) fn iarange(start: usize, stop: usize) -> Result<Array> {
  if stop > F32_EXACT_INT_MAX {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "iarange: stop",
      "must be <= 2^24 (f32 exact-integer limit; exclusive stop is cast to f32 for the f32-only Array::arange, so stop>2^24 would round and silently truncate the range — cache context too long for this path)",
      format_smolstr!("{stop}"),
    )));
  }
  ops::misc::astype(
    &Array::arange::<f32>(start as f32, stop as f32, 1.0)?,
    Dtype::I32,
  )
}

/// Port of `mx.roll(a, shift=shift)` for the 1-D `[L]` arrays
/// `RotatingKVCache.make_mask` rolls (`cache.py:577`). `crate::ops` has no
/// native `roll`, so it is composed faithfully: mlx defines
/// `out[i] = a[(i - shift) mod L]`, i.e. a positive `shift` moves elements
/// toward higher indices with wrap, which is exactly
/// `concat([a[L-s:], a[:L-s]])` for `s = shift mod L` (and the identity when
/// `s == 0`). Built with the same `crate::ops` slice/concatenate idioms the
/// rest of this module uses; no implicit eval.
pub(crate) fn roll_1d(a: &Array, shift: usize) -> Result<Array> {
  let l = a.shape()[0];
  if l == 0 {
    return a.try_clone();
  }
  let s = shift % l;
  if s == 0 {
    return a.try_clone();
  }
  // out = [ a[L-s : L] , a[0 : L-s] ]  (1-D slice on axis 0)
  let tail = ops::indexing::slice(a, &[(l - s) as i32], &[l as i32], &[1])?;
  let head = ops::indexing::slice(a, &[0], &[(l - s) as i32], &[1])?;
  ops::shape::concatenate(&[&tail, &head], 0)
}

/// Port of `mlx_lm.models.base.create_causal_mask` (the offset + sliding
/// `window_size` subset).
///
/// ```text
/// rinds = mx.arange(offset + N)
/// linds = mx.arange(offset, offset + N) if offset else rinds
/// linds = linds[:, None]; rinds = rinds[None]
/// mask  = linds >= rinds
/// if window_size is not None:
///     mask = mask & (linds < rinds + window_size)
/// ```
///
/// Returns the boolean `[N, offset + N]` causal (optionally windowed) mask.
///
/// `offset + N` is computed with [`usize::checked_add`] *before* any range is
/// built: a hostile/corrupt loaded `offset` (mlx-lm's prompt-cache
/// `set_meta_state`) could otherwise overflow → a debug panic, or a release
/// wrap to a small value that then *passes* `iarange`'s `2^24` check and
/// silently produces a wrong mask. The overflow is a recoverable
/// [`Error::ArithmeticOverflow`] instead (behavior is identical for every valid
/// input — `offset + N ≤ 2^24` always reaches `iarange` unchanged).
///
/// `window_size` keeps mlx-lm's unbounded-Python-int semantics: there
/// `mask & (linds < rinds + window_size)` makes a `window_size` at least the
/// full index range a no-op (the term is always true). The largest
/// `rinds + window_size` compares against an `linds` in `[0, total)`, so a
/// `window_size >= total` cannot mask any valid position — the windowing term
/// is skipped entirely (the plain causal mask). Otherwise
/// `window_size < total ≤ 2^24 < i32::MAX`, so the `as i32` cast is exact;
/// this both mirrors mlx-lm and removes the lossy-cast hazard of a
/// `window_size > i32::MAX` wrapping.
pub fn create_causal_mask(n: usize, offset: usize, window_size: Option<usize>) -> Result<Array> {
  let total = offset.checked_add(n).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "create_causal_mask: offset + N",
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
  // linds[:, None] / rinds[None]
  let linds = ops::shape::expand_dims_axes(&linds, &[1])?;
  let rinds = ops::shape::expand_dims_axes(&rinds, &[0])?;

  let mut mask = ops::comparison::greater_equal(&linds, &rinds)?;
  if let Some(w) = window_size
    && w < total
  {
    // linds < rinds + window_size  (a `w >= total` is mlx-lm's
    // unbounded-int no-op: every `linds < rinds + w` already holds).
    let bound = ops::arithmetic::add(&rinds, &scalar_i32(w as i32)?)?;
    let windowed = ops::comparison::less(&linds, &bound)?;
    mask = ops::logical::logical_and(&mask, &windowed)?;
  }
  Ok(mask)
}

/// Port of `mlx_lm.models.cache.create_attention_mask` (`cache.py:114-126`):
///
/// ```text
/// if window_size is not None:           -> create_causal_mask(N, offset, window_size)
/// elif N == 1:                          -> None
/// elif return_array:                    -> create_causal_mask(N, offset, None)
/// else:                                 -> "causal"
/// ```
///
/// The `"causal"` sentinel maps to [`MaskMode::Causal`]; a materialized mask
/// to [`MaskMode::Array`]; the `N == 1` no-mask case to [`MaskMode::None`]
/// (mlx-swift-lm's `.none` / `.causal` / `.array`).
pub fn create_attention_mask(
  n: usize,
  offset: usize,
  return_array: bool,
  window_size: Option<usize>,
) -> Result<MaskMode> {
  if window_size.is_some() {
    Ok(MaskMode::Array(create_causal_mask(n, offset, window_size)?))
  } else if n == 1 {
    Ok(MaskMode::None)
  } else if return_array {
    Ok(MaskMode::Array(create_causal_mask(n, offset, None)?))
  } else {
    Ok(MaskMode::Causal)
  }
}
