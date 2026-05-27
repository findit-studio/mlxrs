//! Sequence-axis helpers shared by the KV cache implementations.
//!
//! mlx-lm treats a KV state as 4-D `[B, n_kv_heads, S, head_dim]` and
//! concatenates/slices on the sequence axis (`axis=-2`). These are the
//! `mlxrs::Array` (functional, no in-place buffer slicing) equivalents of the
//! `mx.concatenate([a, b], axis=-2)` / `v[..., a:b, :]` idioms the
//! [`StandardKvCache`](super::StandardKvCache) /
//! [`RotatingKvCache`](super::RotatingKvCache) ports use verbatim.

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    ArithmeticOverflowPayload, EmptyInputPayload, Error, InvariantViolationPayload,
    RankMismatchPayload, Result, ShapePairMismatchPayload,
  },
  ops,
};

/// The number of key/value heads + sequence axes a KV state must have:
/// `[B, n_kv_heads, S, head_dim]` (mlx-lm's `keys.shape == (B, n_kv_heads, S,
/// head_dim)` — the sequence axis is `-2`).
pub(crate) const KV_NDIM: usize = 4;
/// The sequence axis of a `[B, n_kv_heads, S, head_dim]` KV state, as a
/// negative (rank-relative) index — mlx-lm concatenates/slices keys on
/// `axis=-2`.
pub(crate) const SEQ_AXIS: i32 = -2;

/// Validate a key/value tensor's rank and return its sequence length
/// (`shape[-2]`). mlx-lm assumes the 4-D `[B, n_kv_heads, S, head_dim]`
/// layout; we check it instead of indexing blindly so a misuse is a
/// recoverable [`Error::RankMismatch`], not a panic.
pub(crate) fn seq_len(name: &str, a: &Array) -> Result<usize> {
  let shape = a.shape();
  if shape.len() != KV_NDIM {
    let context: &'static str = match name {
      "keys" => "seq_len: KV cache expects 4-D keys [B, n_kv_heads, S, head_dim]",
      "values" => "seq_len: KV cache expects 4-D values [B, n_kv_heads, S, head_dim]",
      _ => "seq_len: KV cache expects 4-D [B, n_kv_heads, S, head_dim]",
    };
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    )));
  }
  Ok(shape[KV_NDIM - 2])
}

/// Validate a key/value tensor's rank and return its head dimension
/// (`shape[-1]`, the last axis). mlx-lm reads `values.shape[3]` /
/// `values.shape[-1]` directly on the assumed 4-D `[B, n_kv_heads, S,
/// head_dim]` layout (`cache.py:337`/`cache.py:478`); we check the rank
/// instead of indexing blindly so a rank-invalid `values` is a recoverable
/// [`Error::RankMismatch`] (the faithful equivalent of Python's
/// `IndexError`), not a Rust slice out-of-bounds panic.
pub(crate) fn head_dim(name: &str, a: &Array) -> Result<usize> {
  let shape = a.shape();
  if shape.len() != KV_NDIM {
    let context: &'static str = match name {
      "keys" => "head_dim: KV cache expects 4-D keys [B, n_kv_heads, S, head_dim]",
      "values" => "head_dim: KV cache expects 4-D values [B, n_kv_heads, S, head_dim]",
      _ => "head_dim: KV cache expects 4-D [B, n_kv_heads, S, head_dim]",
    };
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    )));
  }
  Ok(shape[KV_NDIM - 1])
}

/// Prepare a write-emulation RHS tensor `new` for splicing over `[a, end)` of
/// the target KV buffer `buf`: broadcast `new` to the slice shape `[buf[0],
/// buf[1], end - a, buf[3]]` (the same shape mlx's `slice_update` builds for
/// `src[..., a:end, :] = new`, `ops.cpp:843`). This mirrors the implicit
/// broadcast + shape validation that mlx-lm's `self.<buf>[..., a:a+s, :] =
/// new` slice-assignment performs at the mlx level:
///
/// - a size-`d` `new` axis matches a size-`d` buffer axis (identity);
/// - a size-`1` `new` axis broadcasts up to a size-`d` buffer axis (size-1
///   broadcast — mlx `broadcast_to` semantics, called by `slice_update`);
/// - any other non-seq axis mismatch is non-broadcastable and raises
///   (mlx `broadcast_to` raises on a non-broadcastable dim mismatch).
///
/// `KV_NDIM-2` is the sequence axis: the seq-axis of `new` must equal
/// `end - a` (the slice window length) — mlx's `broadcast_to(update_shape,
/// upd_shape)` raises if `update_shape[seq] != upd_shape[seq]` and either
/// side isn't 1. Our `set_seq` always splices exactly `S` rows so the
/// caller's `s == new.shape[KV_NDIM-2]` invariant holds for every faithful
/// trace; we still check it here so a hostile/corrupt input is a recoverable
/// `Err`, not a downstream concat panic.
///
/// In mlxrs's `set_seq` write-emulation (which concatenates `[head, new,
/// tail]` via [`concat_parts`]), this is required at the entry — otherwise a
/// full-window write (empty head + empty tail) shortcuts to returning `new`
/// after only a rank check, BYPASSING both the non-seq-axes broadcast
/// validation AND the broadcast itself (e.g. a `[1, .., .., ..]` `new`
/// would silently SHRINK a `[2, .., .., ..]` `buf`'s batch axis on the
/// full-window fast path, while mlx-lm broadcasts the size-1 axis and keeps
/// the buffer's shape). Routing every full/partial window through this
/// helper keeps non-broadcastable mismatches as recoverable `Err` AND
/// broadcasts a size-1 RHS up exactly as mlx does — byte-identical to mlx-lm
/// for every faithful input.
///
/// `name` identifies the target buffer (`"keys"` / `"values"`) for the
/// per-target error message. This is a SINGLE-tensor check (`new` vs target
/// `buf`), NOT the fenced K/V cross-validation (keys vs values).
pub(crate) fn broadcast_write_rhs(
  name: &str,
  buf: &Array,
  a: usize,
  end: usize,
  new: &Array,
) -> Result<Array> {
  let bs = buf.shape();
  let ns = new.shape();
  if bs.len() != KV_NDIM {
    let context: &'static str = match name {
      "keys" => "broadcast_write_rhs: KV cache expects 4-D keys [B, n_kv_heads, S, head_dim]",
      "values" => "broadcast_write_rhs: KV cache expects 4-D values [B, n_kv_heads, S, head_dim]",
      _ => "broadcast_write_rhs: KV cache expects 4-D [B, n_kv_heads, S, head_dim]",
    };
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      bs.len() as u32,
      bs.to_vec(),
    )));
  }
  if ns.len() != KV_NDIM {
    let context: &'static str = match name {
      "keys" => {
        "broadcast_write_rhs: KV cache expects 4-D keys write RHS [B, n_kv_heads, S, head_dim]"
      }
      "values" => {
        "broadcast_write_rhs: KV cache expects 4-D values write RHS [B, n_kv_heads, S, head_dim]"
      }
      _ => "broadcast_write_rhs: KV cache expects 4-D write RHS [B, n_kv_heads, S, head_dim]",
    };
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      ns.len() as u32,
      ns.to_vec(),
    )));
  }
  // Slice window length on the sequence axis — the broadcast target's seq
  // dim (mlx `slice_update`'s `upd_shape[seq]` is exactly `stop - start`).
  let win = end.checked_sub(a).ok_or_else(|| {
    let context: &'static str = match name {
      "keys" => "set_seq: keys write end < start",
      "values" => "set_seq: values write end < start",
      _ => "set_seq: write end < start",
    };
    Error::InvariantViolation(InvariantViolationPayload::new(
      context,
      "must satisfy end >= start",
    ))
  })?;
  // Per-axis: identity (`d == d`) OR size-1-broadcast (`new == 1`). mlx
  // `broadcast_to` (called by `slice_update`, `ops.cpp:843`) accepts a size-1
  // `new` axis broadcast up to the buffer axis; any other mismatch raises.
  // The seq axis (`KV_NDIM-2`) is also validated — `new[seq]` must equal
  // `win` (or 1, which mlx broadcasts to `win`).
  for axis in 0..KV_NDIM {
    let target = if axis == KV_NDIM - 2 { win } else { bs[axis] };
    let got = ns[axis];
    if got != target && got != 1 {
      // Reconstruct the per-axis expected shape: target axes follow the buffer
      // shape except seq-axis which is the slice window length. This carries
      // the full pair (expected, actual) plus the call-site label naming
      // axis kind/name; per-axis breakdown is preserved structurally.
      let expected: Vec<usize> = (0..KV_NDIM)
        .map(|i| if i == KV_NDIM - 2 { win } else { bs[i] })
        .collect();
      let context: &'static str = match name {
        "keys" => {
          "broadcast_write_rhs: keys write RHS non-broadcastable (mlx-lm slice-assignment raises on non-broadcastable non-seq axes; seq-axis target is the slice window length)"
        }
        "values" => {
          "broadcast_write_rhs: values write RHS non-broadcastable (mlx-lm slice-assignment raises on non-broadcastable non-seq axes; seq-axis target is the slice window length)"
        }
        _ => {
          "broadcast_write_rhs: write RHS non-broadcastable (mlx-lm slice-assignment raises on non-broadcastable non-seq axes; seq-axis target is the slice window length)"
        }
      };
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        context,
        expected.as_slice(),
        ns.to_vec().as_slice(),
      )));
    }
  }
  // Build the broadcast target shape `[buf[0], buf[1], win, buf[3]]` and
  // broadcast `new` to it (mlx `slice_update`'s `broadcast_to(update,
  // upd_shape)`, `ops.cpp:843`). For a fully matching `new` this is the
  // identity broadcast (the same shape — mlx's `broadcast_to` no-ops); for a
  // size-1-broadcast `new` it expands the size-1 axes to match the buffer.
  let target_shape: Vec<usize> = (0..KV_NDIM)
    .map(|axis| if axis == KV_NDIM - 2 { win } else { bs[axis] })
    .collect();
  ops::shape::broadcast_to(new, &target_shape.as_slice())
}

/// Slice the sequence axis (`-2`) of a 4-D KV tensor to `[start, end)`,
/// keeping every other axis full. mlx-lm's `v[..., start:end, :]`.
///
/// `start` / `end` arrive as `usize` (callers pass offsets, seq positions,
/// or restored prompt-cache metadata); the mlx-c slice op takes `i32`
/// bounds. An unchecked `usize as i32` cast silently wraps on
/// `usize > i32::MAX` (potentially to a negative `i32`), producing a wrong
/// slice stop and a mis-spliced state. So we use the checked
/// `i32::try_from(end)` (and `start`) and surface overflow as a recoverable
/// [`Error::ArithmeticOverflow`] at this single integer-wrap boundary —
/// observably-equivalent for every valid input
/// (`start, end <= i32::MAX as usize`), which covers every real cache use
/// case. The shape dims come from an `Array` that mlx itself already
/// constructed, but the same checked cast is applied for defense-in-depth
/// and consistency (so any future call that hits this boundary fails
/// recoverably, never with a silent wrap).
pub(crate) fn slice_seq(a: &Array, start: usize, end: usize) -> Result<Array> {
  let shape = a.shape();
  // Rank check — surface a rank-misuse as recoverable `ShapeMismatch`
  // rather than panicking on the `stops[KV_NDIM - 2]` index below
  // (Copilot review #3273072304). Surrounding helpers (`seq_len`,
  // `head_dim`, `concat_parts`) all enforce `KV_NDIM` the same way; the
  // happy path through the existing callers (Standard/Rotating/Chunked/
  // Quantized/Batch/BatchRotating) all pre-validate rank before reaching
  // here, so this is a defense-in-depth guard, not a behavior change.
  if shape.len() != KV_NDIM {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "slice_seq: expects 4-D array [B, n_kv_heads, S, head_dim]",
      shape.len() as u32,
      shape.to_vec(),
    )));
  }
  let mut starts = vec![0i32; KV_NDIM];
  let mut stops: Vec<i32> = shape
    .iter()
    .map(|&d| {
      i32::try_from(d).map_err(|_| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "slice_seq: shape dim exceeds i32::MAX",
          "i32",
          [("dim", d as u64)],
        ))
      })
    })
    .collect::<Result<Vec<i32>>>()?;
  let strides = vec![1i32; KV_NDIM];
  starts[KV_NDIM - 2] = i32::try_from(start).map_err(|_| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "slice_seq: start offset exceeds i32::MAX",
      "i32",
      [("start", start as u64)],
    ))
  })?;
  stops[KV_NDIM - 2] = i32::try_from(end).map_err(|_| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "slice_seq: end offset exceeds i32::MAX",
      "i32",
      [("end", end as u64)],
    ))
  })?;
  ops::indexing::slice(a, &starts, &stops, &strides)
}

/// Concatenate two 4-D KV tensors along the sequence axis (`-2`) — mlx-lm's
/// `mx.concatenate([a, b], axis=-2)`.
pub(crate) fn concat_seq(a: &Array, b: &Array) -> Result<Array> {
  ops::shape::concatenate(&[a, b], SEQ_AXIS)
}

/// Slice the sequence axis to `[start, end)` with Python/NumPy-style
/// clamping (`end` capped at the length, `start` capped at `end`) so an
/// over-long bound is the empty/whole slice mlx-lm's `v[..., a:b, :]`
/// would produce, never a panic.
pub(crate) fn seq_slice(a: &Array, start: usize, end: usize) -> Result<Array> {
  let l = a.shape()[KV_NDIM - 2];
  let end = end.min(l);
  let start = start.min(end);
  slice_seq(a, start, end)
}

/// In-memory byte size of one `Dtype` element — mlx-c's `mlx_dtype_size`,
/// reproduced as a pure Rust mapping so [`nbytes`] needs no FFI/eval.
fn dtype_size(d: Dtype) -> usize {
  match d {
    Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
    Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 2,
    Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
    Dtype::U64 | Dtype::I64 | Dtype::F64 | Dtype::Complex64 => 8,
  }
}

/// Byte size of an array — `elem_count * dtype_size` (mlx-lm's
/// `array.nbytes`). Pure metadata read: no eval, no allocation.
pub(crate) fn nbytes(a: &Array) -> Result<usize> {
  Ok(a.size() * dtype_size(a.dtype()?))
}

/// Concatenate the non-empty parts along the sequence axis (an empty part
/// is a no-op in mlx-lm's `mx.concatenate`; dropping it avoids a redundant
/// op and any zero-length-concat edge). A single part is returned directly.
///
/// The `S > 1` `RotatingKvCache::update_concat` path routes the *external*,
/// not-yet-rank-validated `keys`/`values` through here (via `_trim`), so a
/// rank-invalid argument must NOT panic on the raw sequence-axis index.
/// Only a provably-empty *4-D* part is dropped; a part whose rank is not
/// `KV_NDIM` is **kept** and flowed into `ops::shape::concatenate`, which
/// returns a recoverable `Err` — the faithful equivalent of mlx-lm's
/// `mx.concatenate` itself raising a catchable error on a rank-mismatched
/// input. Behavior for valid 4-D parts is byte-identical (an empty 4-D part
/// is still dropped, a non-empty one still kept).
pub(crate) fn concat_parts(parts: &[&Array]) -> Result<Array> {
  let non_empty: Vec<&Array> = parts
    .iter()
    .copied()
    .filter(|a| {
      let shape = a.shape();
      // Drop only a provably-empty 4-D part; never index a rank-invalid
      // part's sequence axis (that would be a slice OOB panic) — keep it
      // and let `concatenate` surface a recoverable rank error.
      shape.len() != KV_NDIM || shape[KV_NDIM - 2] > 0
    })
    .collect();
  // The `[]` / `[one]` fast paths return a part directly *without* going
  // through `ops::shape::concatenate`. mlx-lm's `mx.concatenate(to_cat,
  // axis=2)` validates rank even for a single-element `to_cat` and raises
  // (catchably) on a rank-mismatched element, and the `update_concat` S>1
  // path can leave exactly the rank-invalid external `values` as the lone
  // surviving part (e.g. `max_size=1, keep=0`: the retained 4-D pieces are
  // empty and dropped). Returning that clone would (a) diverge from
  // `cache.py` and (b) store a rank-invalid buffer that a *later* valid
  // update would hit via a raw cached-shape read (`temporal_order` /
  // `set_seq`) → panic. So a fast-path part must be rank-checked: a
  // rank-invalid one is the same recoverable `Error::RankMismatch`
  // `mx.concatenate` would raise. A valid 4-D part is byte-identical
  // (`try_clone`) — `concatenate` of a single 4-D array is identity.
  let rank_checked = |a: &Array| -> Result<Array> {
    let shape = a.shape();
    if shape.len() != KV_NDIM {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "concat_parts: KV cache concat expects 4-D [B, n_kv_heads, S, head_dim] parts",
        shape.len() as u32,
        shape.to_vec(),
      )));
    }
    a.try_clone()
  };
  match non_empty.as_slice() {
    // Every part empty: mirror `mx.concatenate`'s result by returning the
    // first (empty) part. Internal callers always pass >= 1 part; an empty
    // `parts` slice has no defined concat result, so it is a recoverable
    // `Error` rather than an indexing panic.
    [] => match parts.first() {
      Some(first) => rank_checked(first),
      None => Err(Error::EmptyInput(EmptyInputPayload::new(
        "concat_parts: parts",
      ))),
    },
    [one] => rank_checked(one),
    many => ops::shape::concatenate(many, SEQ_AXIS),
  }
}

#[cfg(test)]
mod tests {
  //! Regression tests for the checked `usize -> i32` cast at the
  //! `slice_seq` boundary. A forged/corrupt prompt-cache restore can
  //! flow a `usize > i32::MAX` through here; the unchecked `as i32` cast
  //! previously wrapped silently (potentially to a negative `i32`),
  //! producing a wrong slice stop. The checked cast surfaces overflow as
  //! a recoverable `Error::ArithmeticOverflow` at this single source of
  //! truth — every cache (Standard / Rotating / Chunked / Quantized /
  //! Batch / BatchRotating) that flows restored offsets through
  //! `slice_seq` (via `enforce_offset_len_invariant` / `trim_triple` /
  //! direct callers) shares the same protection.
  use super::*;
  use crate::{Error, array::Array};
  // Minimum 4-D KV-shaped array all tests reuse — `slice_seq` only checks
  // the rank-implicit way (via `KV_NDIM`) so a `[1, 1, 1, 1]` is enough.
  fn kv1() -> Array {
    Array::from_slice::<f32>(&[0.0], &(1usize, 1, 1, 1)).unwrap()
  }

  #[test]
  fn slice_seq_rejects_end_above_i32_max() {
    let a = kv1();
    let bad_end = (i32::MAX as usize) + 1;
    let r = slice_seq(&a, 0, bad_end);
    match r {
      Err(Error::ArithmeticOverflow(payload)) => {
        assert!(
          payload.context().contains("end") && payload.context().contains("i32::MAX"),
          "expected context to name `end` and `i32::MAX`, got: {:?}",
          payload.context()
        );
        let has_value = payload
          .operands()
          .iter()
          .any(|(n, v)| *n == "end" && *v == bad_end as u64);
        assert!(
          has_value,
          "expected operands to include `end` = {bad_end}, got: {:?}",
          payload.operands()
        );
      }
      other => panic!("expected Err(ArithmeticOverflow), got {other:?}"),
    }
  }

  #[test]
  fn slice_seq_rejects_start_above_i32_max() {
    let a = kv1();
    let bad_start = (i32::MAX as usize) + 1;
    // `end` also overflows here; this test only asserts the start-bound
    // overflow is surfaced (not that it wins over the end-bound check) —
    // either error variant is correct, both name an offset > i32::MAX.
    let r = slice_seq(&a, bad_start, bad_start);
    match r {
      Err(Error::ArithmeticOverflow(payload)) => {
        assert!(
          payload.context().contains("i32::MAX"),
          "expected context to mention `i32::MAX`, got: {:?}",
          payload.context()
        );
        assert!(
          payload.context().contains("start") || payload.context().contains("end"),
          "expected context to name `start` or `end` offset, got: {:?}",
          payload.context()
        );
      }
      other => panic!("expected Err(ArithmeticOverflow), got {other:?}"),
    }
  }

  #[test]
  fn slice_seq_accepts_zero_window_at_origin() {
    // Sanity: the checked cast is observably-equivalent for valid inputs.
    // A `[0, 0)` window on the seq axis is a valid empty slice (mlx-lm's
    // `v[..., 0:0, :]`) and must succeed unchanged.
    let a = kv1();
    let r = slice_seq(&a, 0, 0);
    assert!(r.is_ok(), "valid zero-window slice must succeed, got {r:?}");
  }

  #[test]
  fn slice_seq_rejects_rank_mismatch() {
    // Defense-in-depth: a rank-misuse must surface as recoverable
    // RankMismatch rather than panicking on the `stops[KV_NDIM - 2]`
    // index (Copilot review #3273072304). All real callers pre-validate
    // rank, so this only fires on a programmer-error / misuse path.
    let a1: Array = Array::from_slice::<f32>(&[0.0, 1.0], &(2usize,)).unwrap(); // rank 1
    let r = slice_seq(&a1, 0, 0);
    match r {
      Err(Error::RankMismatch(payload)) => {
        assert!(
          payload.context().contains("4-D") || payload.context().contains("slice_seq"),
          "error context must name expected rank or call site; got: {:?}",
          payload.context()
        );
        assert_eq!(payload.actual(), 1, "expected actual rank 1");
        assert_eq!(
          payload.actual_shape(),
          &[2usize],
          "expected actual shape [2]"
        );
      }
      other => panic!("rank-1 must Err(RankMismatch), got {other:?}"),
    }
  }
}
