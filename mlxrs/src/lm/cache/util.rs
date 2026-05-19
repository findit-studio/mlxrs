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
  error::{Error, Result},
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
/// recoverable [`Error::ShapeMismatch`], not a panic.
pub(crate) fn seq_len(name: &str, a: &Array) -> Result<usize> {
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
pub(crate) fn slice_seq(a: &Array, start: usize, end: usize) -> Result<Array> {
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
pub(crate) fn concat_parts(parts: &[&Array]) -> Result<Array> {
  let non_empty: Vec<&Array> = parts
    .iter()
    .copied()
    .filter(|a| a.shape()[KV_NDIM - 2] > 0)
    .collect();
  match non_empty.as_slice() {
    // Every part empty: mirror `mx.concatenate`'s result by returning the
    // first (empty) part. Internal callers always pass >= 1 part; an empty
    // `parts` slice has no defined concat result, so it is a recoverable
    // `Error` rather than an indexing panic.
    [] => match parts.first() {
      Some(first) => first.try_clone(),
      None => Err(Error::ShapeMismatch {
        message: "concat_parts: no parts to concatenate".into(),
      }),
    },
    [one] => one.try_clone(),
    many => ops::shape::concatenate(many, SEQ_AXIS),
  }
}
