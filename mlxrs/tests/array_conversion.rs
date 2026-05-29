//! Introspection + conversion coverage for `array::conversion` (#260).
//!
//! `array/conversion.rs` had no dedicated test file. `error_paths.rs` covers
//! the `to_vec`/`as_slice` NON-contiguous guard and `array_explicit_eval.rs`
//! covers `try_item`; this file targets the remaining gaps on the `&mut self`
//! accessors and the metadata readers:
//!   * `ndim`/`size`/`shape`/`dtype` asserted directly on a multi-dim array
//!     (the `shape()` `Vec<usize>` mapping over `mlx_array_dim`).
//!   * `item` / `to_vec` / `as_slice` dtype-mismatch → typed `DtypeMismatch`
//!     (only `try_item`'s mismatch was previously tested).
//!   * `item` on a NON-scalar (size != 1) → `Error::MlxC` (mlx C++ throws the
//!     UNBRACKETED message `"item can only be called on arrays of size 1."`,
//!     which `MlxOpKind::parse_prefix` cannot classify, so the boundary emits
//!     `MlxC`, not `MlxOp`/`Backend` — see flag in the worker report).
//!   * `as_slice` HAPPY path on a contiguous array (only the non-contig
//!     rejection was tested in `error_paths.rs`).
//!   * `from_slice` → `as_slice` borrow round-trip, and `Debug` formatting.
//!
//! Accessor rule (`feedback_no_implicit_eval`): `item`/`to_vec`/`as_slice`
//! are `&mut self` and eval internally, so the build → read pattern of the
//! sibling test files is used directly.

use mlxrs::{Array, Dtype};

// ───────── ndim / size / shape / dtype ─────────

#[test]
fn metadata_on_3d_array() {
  let a = Array::zeros::<f32>(&(2, 3, 4)).unwrap();
  assert_eq!(a.ndim(), 3);
  assert_eq!(a.size(), 24);
  assert_eq!(a.shape(), vec![2, 3, 4]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
}

#[test]
fn metadata_on_rank0_scalar() {
  // Rank-0 scalar: ndim 0, size 1, empty shape vec.
  let empty: [i32; 0] = [];
  let a = Array::from_slice::<f32>(&[5.0], &empty).unwrap();
  assert_eq!(a.ndim(), 0);
  assert_eq!(a.size(), 1);
  assert_eq!(a.shape(), Vec::<usize>::new());
}

#[test]
fn dtype_reflects_element_type() {
  // dtype() reads the array's actual dtype, which is set by the constructor's
  // type parameter.
  assert_eq!(
    Array::zeros::<i32>(&(1,)).unwrap().dtype().unwrap(),
    Dtype::I32
  );
  assert_eq!(
    Array::zeros::<u32>(&(1,)).unwrap().dtype().unwrap(),
    Dtype::U32
  );
  assert_eq!(
    Array::zeros::<bool>(&(1,)).unwrap().dtype().unwrap(),
    Dtype::Bool
  );
}

// ───────── item ─────────

#[test]
fn item_reads_scalar() {
  let mut a = Array::full::<f32>(&(1,), 12.5).unwrap();
  assert_eq!(a.item::<f32>().unwrap(), 12.5);
}

#[test]
fn item_dtype_mismatch_is_typed_error() {
  // `item::<i32>` on an f32 array: the dtype check fires before any FFI/eval.
  // Payload carries expected=I32 (caller asserted) and got=F32 (actual).
  let mut a = Array::full::<f32>(&(1,), 1.0).unwrap();
  match a.item::<i32>() {
    Err(mlxrs::Error::DtypeMismatch(p)) => {
      assert_eq!(p.expected(), Dtype::I32);
      assert_eq!(p.got(), Dtype::F32);
    }
    other => panic!("expected Err(DtypeMismatch), got {other:?}"),
  }
}

#[test]
fn item_on_non_scalar_errors_not_aborts() {
  // mlx C++ `array::item()` throws `"item can only be called on arrays of
  // size 1."` for a multi-element array. That message has NO `[op]` bracket
  // prefix, so the FFI boundary classifies it as `Error::MlxC` (the
  // unparseable-prefix catch-all), NOT `MlxOp` and NOT `Backend`. The key
  // contract here is "returns Err, does not abort the process".
  let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let r = a.item::<f32>();
  assert!(
    matches!(r, Err(mlxrs::Error::MlxC(_))),
    "expected Err(MlxC) for item on a non-scalar, got {r:?}"
  );
}

// ───────── to_vec ─────────

#[test]
fn to_vec_round_trips_buffer() {
  let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn to_vec_dtype_mismatch_is_typed_error() {
  let mut a = Array::from_slice::<f32>(&[1.0, 2.0], &(2,)).unwrap();
  match a.to_vec::<i32>() {
    Err(mlxrs::Error::DtypeMismatch(p)) => {
      assert_eq!(p.expected(), Dtype::I32);
      assert_eq!(p.got(), Dtype::F32);
    }
    other => panic!("expected Err(DtypeMismatch), got {other:?}"),
  }
}

#[test]
fn to_vec_on_zero_element_array_is_empty() {
  // Zero-length short-circuit: a `[0]`-shaped array yields an empty Vec
  // without tripping the contiguity / null-pointer guards.
  let mut a = Array::from_slice::<f32>(&[], &[0i32]).unwrap();
  assert_eq!(a.size(), 0);
  assert_eq!(a.to_vec::<f32>().unwrap(), Vec::<f32>::new());
}

// ───────── as_slice ─────────

#[test]
fn as_slice_borrows_contiguous_buffer() {
  // Happy path for the borrow-relaxed view (error_paths.rs only covers the
  // non-contiguous rejection). A freshly-built `from_slice` array is
  // row-contiguous, so `as_slice` returns the buffer in order.
  let mut a = Array::from_slice::<f32>(&[10.0, 20.0, 30.0], &(3,)).unwrap();
  assert_eq!(a.as_slice::<f32>().unwrap(), &[10.0, 20.0, 30.0]);
}

#[test]
fn as_slice_dtype_mismatch_is_typed_error() {
  let mut a = Array::from_slice::<f32>(&[1.0], &(1,)).unwrap();
  match a.as_slice::<i32>() {
    Err(mlxrs::Error::DtypeMismatch(p)) => {
      assert_eq!(p.expected(), Dtype::I32);
      assert_eq!(p.got(), Dtype::F32);
    }
    other => panic!("expected Err(DtypeMismatch), got {other:?}"),
  }
}

#[test]
fn as_slice_on_zero_element_array_is_empty() {
  let mut a = Array::from_slice::<f32>(&[], &[0i32]).unwrap();
  assert_eq!(a.as_slice::<f32>().unwrap(), &[] as &[f32]);
}

// ───────── Debug / Display ─────────

#[test]
fn debug_reports_shape_and_dtype() {
  // `Debug` reads only metadata (no eval); the format string is fixed:
  // `Array(shape={shape:?}, dtype={dtype:?})` with dtype as `Option<Dtype>`.
  let a = Array::zeros::<f32>(&(2, 2)).unwrap();
  assert_eq!(format!("{a:?}"), "Array(shape=[2, 2], dtype=Some(F32))");
}

#[test]
fn display_renders_evaluated_values() {
  // `Display` routes through mlx's `tostring` (which evals). The exact
  // whitespace is mlx-internal, so we only assert the rendered scalar value
  // is present rather than pinning the full layout.
  let a = Array::full::<f32>(&(1,), 7.0).unwrap();
  let s = format!("{a}");
  assert!(
    s.contains('7'),
    "Display should render the scalar value 7, got {s:?}"
  );
}
