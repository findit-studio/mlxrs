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
  // index. All real callers pre-validate
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

// ── Shared builders for the closed-form-oracle tests below ───────────────

/// A `[1, 1, S, 1]` KV tensor whose only varying axis is the sequence axis,
/// so each row's marker value reads straight out of `to_vec` (mirrors the
/// sibling `chunked`/`batch_rotating` test builders). `S == vals.len()`.
fn kv(vals: &[f32]) -> Array {
  Array::from_slice::<f32>(vals, &(1usize, 1, vals.len(), 1)).unwrap()
}

/// A general 4-D `[B, H, S, D]` KV tensor with explicit marker data
/// (row-major). `vals.len()` must equal `b*h*s*d`.
fn kv4(b: usize, h: usize, s: usize, d: usize, vals: &[f32]) -> Array {
  Array::from_slice::<f32>(vals, &(b, h, s, d)).unwrap()
}

/// Row-major host read of a (possibly strided) 4-D KV array — route every
/// result through `contiguous` first (a `seq_slice` view / `broadcast_to`
/// result may be strided), mirroring the sibling cache tests.
fn rows(a: &Array) -> Vec<f32> {
  ops::shape::contiguous(a, false)
    .unwrap()
    .to_vec::<f32>()
    .unwrap()
}

// ── seq_len: the default-name `_ =>` context arm (line 39) ───────────────

#[test]
fn seq_len_rank_mismatch_default_name_context() {
  // A `name` that is neither "keys" nor "values" selects the generic
  // `_ =>` context arm (line 39). Rank-3 input → recoverable RankMismatch.
  let a = kv4(1, 1, 1, 1, &[0.0]); // 4-D
  let a3: Array = Array::from_slice::<f32>(&[0.0, 1.0, 2.0], &(1usize, 3, 1)).unwrap();
  // happy path stays 4-D (sanity): a proper 4-D array returns its seq len.
  assert_eq!(seq_len("anything", &a).unwrap(), 1);
  match seq_len("anything", &a3) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(
        p.context(),
        "seq_len: KV cache expects 4-D [B, n_kv_heads, S, head_dim]",
        "non-keys/values name must select the generic context arm"
      );
      assert_eq!(p.actual(), 3);
      assert_eq!(p.actual_shape(), &[1usize, 3, 1]);
    }
    other => panic!("expected Err(RankMismatch), got {other:?}"),
  }
}

// ── head_dim: rank-mismatch, all three name arms (lines 60-68) ───────────

#[test]
fn head_dim_returns_last_axis_for_valid_4d() {
  // Sanity: head_dim is shape[-1]. `[1,1,1,4]` → 4.
  let a = kv4(1, 1, 1, 4, &[0.0, 1.0, 2.0, 3.0]);
  assert_eq!(head_dim("keys", &a).unwrap(), 4);
}

#[test]
fn head_dim_rank_mismatch_keys_name_context() {
  let a3: Array = Array::from_slice::<f32>(&[0.0, 1.0], &(1usize, 1, 2)).unwrap(); // rank 3
  match head_dim("keys", &a3) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(
        p.context(),
        "head_dim: KV cache expects 4-D keys [B, n_kv_heads, S, head_dim]"
      );
      assert_eq!(p.actual(), 3);
      assert_eq!(p.actual_shape(), &[1usize, 1, 2]);
    }
    other => panic!("expected Err(RankMismatch) for keys, got {other:?}"),
  }
}

#[test]
fn head_dim_rank_mismatch_values_name_context() {
  let a5: Array = Array::from_slice::<f32>(&[0.0], &(1usize, 1, 1, 1, 1)).unwrap(); // rank 5
  match head_dim("values", &a5) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(
        p.context(),
        "head_dim: KV cache expects 4-D values [B, n_kv_heads, S, head_dim]"
      );
      assert_eq!(p.actual(), 5);
    }
    other => panic!("expected Err(RankMismatch) for values, got {other:?}"),
  }
}

#[test]
fn head_dim_rank_mismatch_default_name_context() {
  let a2: Array = Array::from_slice::<f32>(&[0.0, 1.0], &(2usize,)).unwrap(); // rank 1
  match head_dim("other", &a2) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(
        p.context(),
        "head_dim: KV cache expects 4-D [B, n_kv_heads, S, head_dim]"
      );
      assert_eq!(p.actual(), 1);
      assert_eq!(p.actual_shape(), &[2usize]);
    }
    other => panic!("expected Err(RankMismatch) for default name, got {other:?}"),
  }
}

// ── broadcast_write_rhs: buffer rank-mismatch (lines 120-128) ────────────

#[test]
fn broadcast_write_rhs_buf_rank_mismatch_keys() {
  // Rank-3 buffer → buf-rank branch, "keys" arm (lines 120-128).
  let buf3: Array = Array::from_slice::<f32>(&[0.0], &(1usize, 1, 1)).unwrap();
  let new = kv4(1, 1, 1, 1, &[5.0]);
  match broadcast_write_rhs("keys", &buf3, 0, 1, &new) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(
        p.context(),
        "broadcast_write_rhs: KV cache expects 4-D keys [B, n_kv_heads, S, head_dim]"
      );
      assert_eq!(p.actual(), 3);
      assert_eq!(p.actual_shape(), &[1usize, 1, 1]);
    }
    other => panic!("expected buf RankMismatch (keys), got {other:?}"),
  }
}

#[test]
fn broadcast_write_rhs_buf_rank_mismatch_values_and_default() {
  let buf3: Array = Array::from_slice::<f32>(&[0.0], &(1usize, 1, 1)).unwrap();
  let new = kv4(1, 1, 1, 1, &[5.0]);
  match broadcast_write_rhs("values", &buf3, 0, 1, &new) {
    Err(Error::RankMismatch(p)) => assert_eq!(
      p.context(),
      "broadcast_write_rhs: KV cache expects 4-D values [B, n_kv_heads, S, head_dim]"
    ),
    other => panic!("expected buf RankMismatch (values), got {other:?}"),
  }
  match broadcast_write_rhs("xyz", &buf3, 0, 1, &new) {
    Err(Error::RankMismatch(p)) => assert_eq!(
      p.context(),
      "broadcast_write_rhs: KV cache expects 4-D [B, n_kv_heads, S, head_dim]"
    ),
    other => panic!("expected buf RankMismatch (default), got {other:?}"),
  }
}

// ── broadcast_write_rhs: RHS (new) rank-mismatch (lines 132-144) ─────────

#[test]
fn broadcast_write_rhs_new_rank_mismatch_keys() {
  // 4-D buffer, rank-3 RHS → new-rank branch, "keys" arm (lines 132-144).
  let buf = kv4(1, 1, 1, 1, &[0.0]);
  let new3: Array = Array::from_slice::<f32>(&[5.0], &(1usize, 1, 1)).unwrap();
  match broadcast_write_rhs("keys", &buf, 0, 1, &new3) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(
        p.context(),
        "broadcast_write_rhs: KV cache expects 4-D keys write RHS [B, n_kv_heads, S, head_dim]"
      );
      assert_eq!(p.actual(), 3);
      assert_eq!(p.actual_shape(), &[1usize, 1, 1]);
    }
    other => panic!("expected RHS RankMismatch (keys), got {other:?}"),
  }
}

#[test]
fn broadcast_write_rhs_new_rank_mismatch_values_and_default() {
  let buf = kv4(1, 1, 1, 1, &[0.0]);
  let new3: Array = Array::from_slice::<f32>(&[5.0], &(1usize, 1, 1)).unwrap();
  match broadcast_write_rhs("values", &buf, 0, 1, &new3) {
    Err(Error::RankMismatch(p)) => assert_eq!(
      p.context(),
      "broadcast_write_rhs: KV cache expects 4-D values write RHS [B, n_kv_heads, S, head_dim]"
    ),
    other => panic!("expected RHS RankMismatch (values), got {other:?}"),
  }
  match broadcast_write_rhs("zzz", &buf, 0, 1, &new3) {
    Err(Error::RankMismatch(p)) => assert_eq!(
      p.context(),
      "broadcast_write_rhs: KV cache expects 4-D write RHS [B, n_kv_heads, S, head_dim]"
    ),
    other => panic!("expected RHS RankMismatch (default), got {other:?}"),
  }
}

// ── broadcast_write_rhs: end < start underflow (lines 150-157) ───────────

#[test]
fn broadcast_write_rhs_end_before_start_keys() {
  // Both buf + new are 4-D, but end (2) < start (5) → checked_sub underflow
  // → InvariantViolation, "keys" arm (lines 150-157).
  let buf = kv4(1, 1, 8, 1, &[0.0; 8]);
  let new = kv4(1, 1, 1, 1, &[5.0]);
  match broadcast_write_rhs("keys", &buf, 5, 2, &new) {
    Err(Error::InvariantViolation(p)) => {
      assert_eq!(p.context(), "set_seq: keys write end < start");
      assert_eq!(p.requirement(), "must satisfy end >= start");
    }
    other => panic!("expected InvariantViolation (keys), got {other:?}"),
  }
}

#[test]
fn broadcast_write_rhs_end_before_start_values_and_default() {
  let buf = kv4(1, 1, 8, 1, &[0.0; 8]);
  let new = kv4(1, 1, 1, 1, &[5.0]);
  match broadcast_write_rhs("values", &buf, 5, 2, &new) {
    Err(Error::InvariantViolation(p)) => {
      assert_eq!(p.context(), "set_seq: values write end < start")
    }
    other => panic!("expected InvariantViolation (values), got {other:?}"),
  }
  match broadcast_write_rhs("other", &buf, 5, 2, &new) {
    Err(Error::InvariantViolation(p)) => {
      assert_eq!(p.context(), "set_seq: write end < start")
    }
    other => panic!("expected InvariantViolation (default), got {other:?}"),
  }
}

// ── broadcast_write_rhs: non-broadcastable axis (lines 180-181, 184) ─────

#[test]
fn broadcast_write_rhs_non_broadcastable_batch_axis_keys() {
  // buf batch=2, RHS batch=3 (not 2, not 1) on a non-seq axis →
  // ShapePairMismatch. seq target is win = end - a = 1; expected shape is
  // the buffer shape with the seq axis replaced by win: [2, 1, 1, 1].
  let buf = kv4(2, 1, 4, 1, &[0.0; 8]);
  let new = kv4(3, 1, 1, 1, &[1.0, 2.0, 3.0]);
  match broadcast_write_rhs("keys", &buf, 0, 1, &new) {
    Err(Error::ShapePairMismatch(p)) => {
      assert!(
        p.context().contains("keys write RHS non-broadcastable"),
        "keys context arm (lines 180-181); got {:?}",
        p.context()
      );
      assert_eq!(p.expected(), &[2usize, 1, 1, 1]);
      assert_eq!(p.actual(), &[3usize, 1, 1, 1]);
    }
    other => panic!("expected ShapePairMismatch (keys), got {other:?}"),
  }
}

#[test]
fn broadcast_write_rhs_non_broadcastable_values_and_default() {
  let buf = kv4(2, 1, 4, 1, &[0.0; 8]);
  let new = kv4(3, 1, 1, 1, &[1.0, 2.0, 3.0]);
  match broadcast_write_rhs("values", &buf, 0, 1, &new) {
    Err(Error::ShapePairMismatch(p)) => assert!(
      p.context().contains("values write RHS non-broadcastable"),
      "values context arm; got {:?}",
      p.context()
    ),
    other => panic!("expected ShapePairMismatch (values), got {other:?}"),
  }
  match broadcast_write_rhs("kkk", &buf, 0, 1, &new) {
    Err(Error::ShapePairMismatch(p)) => assert!(
      p.context().contains("write RHS non-broadcastable")
        && !p.context().contains("keys")
        && !p.context().contains("values"),
      "default context arm (line 184); got {:?}",
      p.context()
    ),
    other => panic!("expected ShapePairMismatch (default), got {other:?}"),
  }
}

// ── broadcast_write_rhs: success paths (lines 165-203) ───────────────────

#[test]
fn broadcast_write_rhs_identity_returns_window_shape() {
  // RHS already matches the slice shape [B,H,win,D] exactly: identity
  // broadcast → same shape, same data. buf [1,1,4,1], window [1,3) (win=2),
  // RHS [1,1,2,1] = markers 7,8.
  let buf = kv4(1, 1, 4, 1, &[0.0, 0.0, 0.0, 0.0]);
  let new = kv4(1, 1, 2, 1, &[7.0, 8.0]);
  let out = broadcast_write_rhs("keys", &buf, 1, 3, &new).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 2, 1], "identity broadcast shape");
  assert_eq!(rows(&out), vec![7.0, 8.0], "identity broadcast data");
}

#[test]
fn broadcast_write_rhs_size1_axes_broadcast_up() {
  // Size-1 RHS axes broadcast up to the buffer's non-seq axes. buf
  // [2,1,4,3]; window [0,1) (win=1); RHS [1,1,1,1] = marker 9 → expands to
  // [2,1,1,3] all 9s (closed-form: 2*1*1*3 = 6 elements, every one == 9).
  let buf = kv4(2, 1, 4, 3, &[0.0; 24]);
  let new = kv4(1, 1, 1, 1, &[9.0]);
  let out = broadcast_write_rhs("values", &buf, 0, 1, &new).unwrap();
  assert_eq!(
    out.shape(),
    vec![2, 1, 1, 3],
    "size-1 axes broadcast to [B,1,win,D]"
  );
  assert_eq!(rows(&out), vec![9.0; 6], "all broadcast elements == marker");
}

// ── nbytes / dtype_size: every byte-size group (lines 289-290, 292) ──────

#[test]
fn nbytes_dtype_size_groups() {
  // nbytes = size * dtype_size(dtype). Each dtype exercises a distinct
  // `dtype_size` match arm with a hand-computed product oracle.
  // group 1 (1 byte): Bool — 4 elements * 1 = 4 (line 289).
  let b = Array::from_slice::<bool>(&[true, false, true, false], &(2usize, 2)).unwrap();
  assert_eq!(nbytes(&b).unwrap(), 4);
  // group 2 (2 bytes): U16 — 3 elements * 2 = 6 (line 290).
  let u16a = Array::from_slice::<u16>(&[1, 2, 3], &(3usize,)).unwrap();
  assert_eq!(nbytes(&u16a).unwrap(), 6);
  // group 4 (4 bytes): F32 — 6 elements * 4 = 24 (line 291).
  let f32a = kv4(1, 1, 6, 1, &[0.0; 6]);
  assert_eq!(nbytes(&f32a).unwrap(), 24);
  // group 8 (8 bytes): I64 — 2 elements * 8 = 16 (line 292).
  let i64a = Array::from_slice::<i64>(&[1, 2], &(2usize,)).unwrap();
  assert_eq!(nbytes(&i64a).unwrap(), 16);
}

// ── concat_seq / seq_slice: closed-form element oracle ───────────────────

#[test]
fn concat_seq_appends_on_sequence_axis() {
  // a = [10,20], b = [30,40,50] on the seq axis ([1,1,S,1]); the
  // concatenation along axis=-2 is [10,20,30,40,50] (hand-computed, NOT via
  // concat_seq).
  let a = kv(&[10.0, 20.0]);
  let b = kv(&[30.0, 40.0, 50.0]);
  let out = concat_seq(&a, &b).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 5, 1]);
  assert_eq!(rows(&out), vec![10.0, 20.0, 30.0, 40.0, 50.0]);
}

#[test]
fn seq_slice_clamps_overlong_end_to_length() {
  // seq_slice clamps end to the seq length (Python `v[..., a:b, :]`). A
  // [1,1,4,1] buffer [1,2,3,4], slice [1, 99) → clamped to [1,4) = [2,3,4].
  let a = kv(&[1.0, 2.0, 3.0, 4.0]);
  let out = seq_slice(&a, 1, 99).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 3, 1]);
  assert_eq!(rows(&out), vec![2.0, 3.0, 4.0]);
}

#[test]
fn seq_slice_start_clamped_to_end_is_empty() {
  // start (5) > clamped end (4) → start clamped to end → empty seq window.
  let a = kv(&[1.0, 2.0, 3.0, 4.0]);
  let out = seq_slice(&a, 5, 99).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 0, 1], "empty window after clamp");
}

// ── concat_parts: rank_checked branch (lines 343-346) ────────────────────

#[test]
fn concat_parts_single_rank_invalid_part_is_rank_mismatch() {
  // A single non-empty rank-3 part survives the empty-filter (its rank !=
  // KV_NDIM keeps it), lands in the `[one]` arm, and `rank_checked` rejects
  // it as RankMismatch (lines 343-346).
  let a3: Array = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1usize, 3, 1)).unwrap();
  match concat_parts(&[&a3]) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(
        p.context(),
        "concat_parts: KV cache concat expects 4-D [B, n_kv_heads, S, head_dim] parts"
      );
      assert_eq!(p.actual(), 3);
      assert_eq!(p.actual_shape(), &[1usize, 3, 1]);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
}

// ── concat_parts: all-empty `[]` arm → first part / EmptyInput (357-359) ─

#[test]
fn concat_parts_all_empty_returns_first_part() {
  // Every part is a provably-empty 4-D part → non_empty is `[]` → the `[]`
  // arm returns the FIRST part directly (rank-checked). `first` is a valid
  // 4-D empty array, so it round-trips unchanged (line 357).
  let e1 = kv(&[]); // [1,1,0,1]
  let e2 = kv(&[]); // [1,1,0,1]
  let out = concat_parts(&[&e1, &e2]).unwrap();
  assert_eq!(
    out.shape(),
    vec![1, 1, 0, 1],
    "returns the first empty part"
  );
}

#[test]
fn concat_parts_empty_slice_is_empty_input_error() {
  // An empty `parts` slice: non_empty is `[]` AND `parts.first()` is None →
  // EmptyInput (lines 358-359).
  match concat_parts(&[]) {
    Err(Error::EmptyInput(p)) => assert_eq!(p.context(), "concat_parts: parts"),
    other => panic!("expected EmptyInput, got {other:?}"),
  }
}

#[test]
fn concat_parts_drops_empty_keeps_nonempty_order() {
  // Mixed: empty + non-empty parts. The empty 4-D parts are dropped; the
  // surviving non-empty parts concatenate in order. parts = [empty, [1,2],
  // empty, [3]] → [1,2,3] (closed-form, NOT via concat_parts).
  let e = kv(&[]);
  let a = kv(&[1.0, 2.0]);
  let b = kv(&[3.0]);
  let out = concat_parts(&[&e, &a, &e, &b]).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 3, 1]);
  assert_eq!(rows(&out), vec![1.0, 2.0, 3.0]);
}

#[test]
fn concat_parts_single_valid_part_is_identity() {
  // A single non-empty 4-D part → `[one]` arm → rank_checked try_clone
  // (identity). [1,1,2,1] = [4,5] round-trips unchanged.
  let a = kv(&[4.0, 5.0]);
  let out = concat_parts(&[&a]).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 2, 1]);
  assert_eq!(rows(&out), vec![4.0, 5.0]);
}
