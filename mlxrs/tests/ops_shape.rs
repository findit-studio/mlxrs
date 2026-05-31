//! Happy-path tests for shape ops.

use std::ffi::CString;

use mlxrs::{Array, ops};

#[test]
fn transpose_2x3_swaps_to_3x2() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let t = a.transpose().unwrap();
  assert_eq!(t.shape(), vec![3, 2]);
}

#[test]
fn transpose_axes_3d_permutes() {
  // (2, 3, 4) with axes [2, 0, 1] -> (4, 2, 3)
  let a = Array::ones::<f32>(&(2usize, 3, 4)).unwrap();
  let t = a.transpose_axes(&[2, 0, 1]).unwrap();
  assert_eq!(t.shape(), vec![4, 2, 3]);
}

#[test]
fn transpose_axes_empty_for_scalar() {
  // 0-D scalar -> empty axes -> still scalar; exercises dim_ptr sentinel.
  let empty: [i32; 0] = [];
  let a = Array::from_slice::<f32>(&[7.0], &empty).unwrap();
  let mut t = a.transpose_axes(&[]).unwrap();
  assert_eq!(t.shape(), Vec::<usize>::new());
  assert_eq!(t.item::<f32>().unwrap(), 7.0);
}

#[test]
fn expand_dims_axes_inserts_dims() {
  let a = Array::ones::<f32>(&(3usize, 4)).unwrap();
  let e = a.expand_dims_axes(&[0, 2]).unwrap();
  // From (3, 4): insert at 0 -> (1, 3, 4); insert at 2 -> (1, 3, 1, 4).
  assert_eq!(e.shape(), vec![1, 3, 1, 4]);
}

#[test]
fn expand_dims_axes_empty_is_clone() {
  // Empty axes is a no-op identity (numpy semantics + cookbook archetype 2 rationale).
  let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.expand_dims_axes(&[]).unwrap();
  assert_eq!(r.shape(), a.shape());
  assert_eq!(r.to_vec::<f32>().unwrap(), a.to_vec::<f32>().unwrap());
}

#[test]
fn squeeze_axes_drops_size1() {
  // (1, 3, 1, 4) -> squeeze [0, 2] -> (3, 4).
  let a = Array::ones::<f32>(&(1usize, 3, 1, 4)).unwrap();
  let s = a.squeeze_axes(&[0, 2]).unwrap();
  assert_eq!(s.shape(), vec![3, 4]);
}

#[test]
fn squeeze_axes_empty_is_clone() {
  let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.squeeze_axes(&[]).unwrap();
  assert_eq!(r.shape(), a.shape());
  assert_eq!(r.to_vec::<f32>().unwrap(), a.to_vec::<f32>().unwrap());
}

#[test]
fn broadcast_to_expands_shape() {
  // (1, 3) broadcast to (4, 3) -> shape (4, 3); content all-ones.
  let a = Array::ones::<f32>(&(1usize, 3)).unwrap();
  let b = a.broadcast_to(&(4usize, 3)).unwrap();
  assert_eq!(b.shape(), vec![4, 3]);
  assert_eq!(b.size(), 12);
}

#[test]
fn stack_two_2x2_along_axis0() {
  let a = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let b = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let s = ops::shape::stack(&[&a, &b]).unwrap();
  // Stack inserts a new axis 0: (2, 2, 2).
  assert_eq!(s.shape(), vec![2, 2, 2]);
}

#[test]
fn stack_axis_two_2x2_along_axis2() {
  let a = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let b = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let s = ops::shape::stack_axis(&[&a, &b], 2).unwrap();
  assert_eq!(s.shape(), vec![2, 2, 2]);
}

#[test]
fn stack_with_method_form() {
  let a = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let b = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let c = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let s = a.stack_with(&[&b, &c], 0).unwrap();
  // 3 inputs stacked along axis 0 -> (3, 2, 2).
  assert_eq!(s.shape(), vec![3, 2, 2]);
}

#[test]
fn stack_rejects_empty_input() {
  let r = ops::shape::stack(&[]);
  assert!(matches!(r, Err(mlxrs::Error::EmptyInput(_))));
  let r2 = ops::shape::stack_axis(&[], 0);
  assert!(matches!(r2, Err(mlxrs::Error::EmptyInput(_))));
}

#[test]
fn split_sections_at_indices_yields_three_parts() {
  // arange(0, 10) = [0,1,2,3,4,5,6,7,8,9]; split at [3, 5] -> 3 parts.
  let a = Array::arange::<f32>(0.0, 10.0, 1.0).unwrap();
  let parts = a.split_sections(&[3, 5], 0).unwrap();
  assert_eq!(parts.len(), 3);
  assert_eq!(parts[0].shape(), vec![3]);
  assert_eq!(parts[1].shape(), vec![2]);
  assert_eq!(parts[2].shape(), vec![5]);
}

#[test]
fn split_sections_empty_indices_yields_single_part() {
  // Splitting at no indices = whole array as a single part. Exercises the
  // empty-slice dim_ptr sentinel.
  let a = Array::arange::<f32>(0.0, 4.0, 1.0).unwrap();
  let parts = a.split_sections(&[], 0).unwrap();
  assert_eq!(parts.len(), 1);
  assert_eq!(parts[0].shape(), vec![4]);
}

#[test]
fn flatten_2x3_to_6() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let mut f = a.flatten(0, -1).unwrap();
  assert_eq!(f.shape(), vec![6]);
  assert_eq!(
    f.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn flatten_partial_range() {
  // (2, 3, 4) flatten dims [1, 2] -> (2, 12).
  let a = Array::ones::<f32>(&(2usize, 3, 4)).unwrap();
  let f = a.flatten(1, 2).unwrap();
  assert_eq!(f.shape(), vec![2, 12]);
}

#[test]
fn swapaxes_swaps_axes() {
  // (2, 3, 4) swap (0, 2) -> (4, 3, 2).
  let a = Array::ones::<f32>(&(2usize, 3, 4)).unwrap();
  let s = a.swapaxes(0, 2).unwrap();
  assert_eq!(s.shape(), vec![4, 3, 2]);
}

#[test]
fn pad_constant_grows_axis() {
  // (3,) padded by 2 on the left and 1 on the right of axis 0 -> (6,).
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let zero = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let mode = CString::new("constant").unwrap();
  let p = ops::shape::pad(&a, &[0], &[2], &[1], &zero, &mode).unwrap();
  assert_eq!(p.shape(), vec![6]);
}

#[test]
fn pad_rejects_length_mismatch() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let zero = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let mode = CString::new("constant").unwrap();
  let r = ops::shape::pad(&a, &[0], &[2], &[1, 2], &zero, &mode);
  // `pad` returns `MultiLengthMismatch` carrying named
  // axes/low/high lengths so callers can identify which list diverged.
  assert!(
    matches!(
      r,
      Err(mlxrs::Error::MultiLengthMismatch(ref p))
        if p.context() == "pad: axes/low/high"
    ),
    "expected Err(MultiLengthMismatch), got {r:?}"
  );
}

#[test]
fn pad_rejects_negative_low() {
  // `low`/`high` are shape extents — negatives must be rejected before
  // reaching mlx::core::Shape construction. `validate_dims`
  // surfaces them as `Error::OutOfRange` with a `"dim[i]=<v>"` value.
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let zero = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let mode = CString::new("constant").unwrap();
  let r = ops::shape::pad(&a, &[0], &[-1], &[1], &zero, &mode);
  match r {
    Err(mlxrs::Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("validate_dims"),
        "context names the validator: {}",
        p.context()
      );
      assert_eq!(p.requirement(), "must be non-negative");
      assert!(
        p.value().contains("-1"),
        "value names the offending dim: {}",
        p.value()
      );
    }
    other => panic!("expected Err(OutOfRange) for negative low, got {other:?}"),
  }
}

#[test]
fn pad_rejects_negative_high() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let zero = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let mode = CString::new("constant").unwrap();
  let r = ops::shape::pad(&a, &[0], &[1], &[-2], &zero, &mode);
  match r {
    Err(mlxrs::Error::OutOfRange(p)) => {
      assert_eq!(p.requirement(), "must be non-negative");
      assert!(
        p.value().contains("-2"),
        "value names the offending dim: {}",
        p.value()
      );
    }
    other => panic!("expected Err(OutOfRange) for negative high, got {other:?}"),
  }
}

#[test]
fn as_strided_basic_view() {
  // [0, 1, 2, 3] reshaped via strides (2, 1) into a 2x2 -> [[0, 1], [2, 3]].
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &[4i32]).unwrap();
  // SAFETY: (2*2=4) elements with stride pattern (2,1) + offset 0 reach
  // index 0..=3 of a 4-element source; entirely in-bounds.
  let mut v = unsafe { ops::shape::as_strided(&a, &(2usize, 2), &[2, 1], 0) }.unwrap();
  assert_eq!(v.shape(), vec![2, 2]);
  assert_eq!(v.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn as_strided_with_offset() {
  // Same buffer, single-axis 2-element view starting at offset 1 -> [1, 2].
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &[4i32]).unwrap();
  // SAFETY: offset 1 + (2-1)*1 = index 2 < 4; in-bounds.
  let mut v = unsafe { ops::shape::as_strided(&a, &(2usize,), &[1], 1) }.unwrap();
  assert_eq!(v.shape(), vec![2]);
  assert_eq!(v.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn as_strided_accepts_slice_shape() {
  // The IntoShape pattern accepts `&[i32]` (and `&[usize]`) alongside the
  // tuple forms — locks in the canonical-shape-archetype ergonomics.
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &[4i32]).unwrap();
  let shape: &[i32] = &[2, 2];
  // SAFETY: same as `as_strided_basic_view`.
  let mut v = unsafe { ops::shape::as_strided(&a, &shape, &[2, 1], 0) }.unwrap();
  assert_eq!(v.shape(), vec![2, 2]);
  assert_eq!(v.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn as_strided_shape_strides_length_mismatch_errors() {
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &[4i32]).unwrap();
  // SAFETY: the length mismatch is rejected pre-FFI; no buffer access.
  let r = unsafe { ops::shape::as_strided(&a, &(2usize, 2), &[1i64], 0) };
  match r {
    Err(mlxrs::Error::LengthMismatch(p)) => {
      assert!(
        p.context().contains("as_strided") && p.context().contains("shape length"),
        "context names the as_strided shape-vs-strides check: {}",
        p.context()
      );
      assert_eq!(p.expected(), 2, "shape length");
      assert_eq!(p.actual(), 1, "strides length");
    }
    other => panic!("expected Err(LengthMismatch), got {other:?}"),
  }
}

#[test]
fn moveaxis_moves_first_to_last() {
  // (2, 3, 4) move axis 0 -> 2 keeps (1, 2) order -> (3, 4, 2).
  let a = Array::ones::<f32>(&(2usize, 3, 4)).unwrap();
  let m = a.moveaxis(0, 2).unwrap();
  assert_eq!(m.shape(), vec![3, 4, 2]);
}

#[test]
fn moveaxis_negative_axes() {
  // (2, 3, 4) move axis -1 (==2) -> 0 -> (4, 2, 3).
  let a = Array::ones::<f32>(&(2usize, 3, 4)).unwrap();
  let m = a.moveaxis(-1, 0).unwrap();
  assert_eq!(m.shape(), vec![4, 2, 3]);
}

#[test]
fn moveaxis_free_fn_form() {
  let a = Array::ones::<f32>(&(2usize, 3, 4)).unwrap();
  let m = ops::shape::moveaxis(&a, 1, 0).unwrap();
  // axis 1 -> 0 -> (3, 2, 4).
  assert_eq!(m.shape(), vec![3, 2, 4]);
}

#[test]
fn roll_flattened_right() {
  // flatten [0..6) -> roll right 2 -> [4,5,0,1,2,3], restore (2, 3).
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = a.roll(&[2]).unwrap();
  assert_eq!(r.shape(), vec![2, 3]);
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![4.0, 5.0, 0.0, 1.0, 2.0, 3.0]
  );
}

#[test]
fn roll_flattened_negative_shift() {
  // 1-D [0,1,2,3,4,5] roll left 1 -> [1,2,3,4,5,0].
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6i32]).unwrap();
  let mut r = ops::shape::roll(&a, &[-1]).unwrap();
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0]
  );
}

#[test]
fn roll_shift_larger_than_size_wraps() {
  // 1-D length 4, shift 6 == 6 mod 4 == 2 -> [2,3,0,1].
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &[4i32]).unwrap();
  let mut r = a.roll(&[6]).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 0.0, 1.0]);
}

#[test]
fn roll_axis_rolls_columns_then_rows() {
  // (2, 3) = [[0,1,2],[3,4,5]].
  // axis 1, shift 1 -> each row rolled right: [[2,0,1],[5,3,4]].
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r1 = a.roll_axis(&[1], 1).unwrap();
  assert_eq!(r1.shape(), vec![2, 3]);
  assert_eq!(
    r1.to_vec::<f32>().unwrap(),
    vec![2.0, 0.0, 1.0, 5.0, 3.0, 4.0]
  );
  // axis 0, shift 1 -> rows rolled down: [[3,4,5],[0,1,2]].
  let mut r0 = ops::shape::roll_axis(&a, &[1], 0).unwrap();
  assert_eq!(
    r0.to_vec::<f32>().unwrap(),
    vec![3.0, 4.0, 5.0, 0.0, 1.0, 2.0]
  );
}

#[test]
fn roll_axis_negative_axis() {
  // (2, 3); axis -1 (==1), shift 1 -> [[2,0,1],[5,3,4]] (same as axis 1).
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = a.roll_axis(&[1], -1).unwrap();
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![2.0, 0.0, 1.0, 5.0, 3.0, 4.0]
  );
}

#[test]
fn roll_axes_multi_shift() {
  // (2, 3) = [[0,1,2],[3,4,5]].
  // roll axis 0 by 1 -> [[3,4,5],[0,1,2]]; then axis 1 by 1 -> [[5,3,4],[2,0,1]].
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = a.roll_axes(&[1, 1], &[0, 1]).unwrap();
  assert_eq!(r.shape(), vec![2, 3]);
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![5.0, 3.0, 4.0, 2.0, 0.0, 1.0]
  );
}

#[test]
fn roll_axes_rejects_shift_axes_count_mismatch() {
  // Unlike the python `mx.roll` binding (which broadcasts a scalar shift over
  // an axis tuple *before* dispatching), `roll_axes` enforces exactly one shift
  // per axis IN RUST before the FFI call. MLX C++ only rejects the too-FEW case
  // and silently ignores extra shifts in the too-MANY case (`ops.cpp` roll
  // iterates over `axes`), so the wrapper rejects BOTH directions as a typed
  // `LengthMismatch` for a predictable 1:1 contract. (Callers wanting
  // scalar-broadcast semantics can repeat the shift, or use `roll`/`roll_axis`.)
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &(2, 3)).unwrap();

  // Too few shifts: 1 shift for 2 axes.
  let too_few = ops::shape::roll_axes(&a, &[1], &[0, 1]);
  assert!(
    matches!(too_few, Err(mlxrs::Error::LengthMismatch(_))),
    "expected Err(LengthMismatch) for too-few shifts, got {too_few:?}"
  );

  // Too many shifts: 3 shifts for 2 axes — MLX would silently drop the extra,
  // the wrapper rejects it.
  let too_many = ops::shape::roll_axes(&a, &[1, 2, 3], &[0, 1]);
  assert!(
    matches!(too_many, Err(mlxrs::Error::LengthMismatch(_))),
    "expected Err(LengthMismatch) for too-many shifts, got {too_many:?}"
  );
}

#[test]
fn tile_1d_doubles() {
  // [1,2,3] tiled by 2 -> [1,2,3,1,2,3].
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let mut t = a.tile(&[2]).unwrap();
  assert_eq!(t.shape(), vec![6]);
  assert_eq!(
    t.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
  );
}

#[test]
fn tile_reps_of_one_is_identity() {
  // reps == 1 leaves data and shape unchanged.
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let mut t = ops::shape::tile(&a, &[1]).unwrap();
  assert_eq!(t.shape(), vec![3]);
  assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn tile_multi_reps_2d() {
  // (2, 2) = [[1,2],[3,4]] tiled by [2, 1] -> (4, 2):
  // [[1,2],[3,4],[1,2],[3,4]] -> flat [1,2,3,4,1,2,3,4].
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut t = a.tile(&[2, 1]).unwrap();
  assert_eq!(t.shape(), vec![4, 2]);
  assert_eq!(
    t.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
  );
}

#[test]
fn tile_reps_longer_than_ndim_prepends_axis() {
  // (2,) tiled by [2, 2] -> reps longer than ndim prepends a leading dim:
  // shape (2, 4), rows are [1,2,1,2].
  let a = Array::from_slice::<f32>(&[1.0, 2.0], &[2i32]).unwrap();
  let mut t = a.tile(&[2, 2]).unwrap();
  assert_eq!(t.shape(), vec![2, 4]);
  assert_eq!(
    t.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
  );
}

// ---------------------------------------------------------------------------
// Bounded-soundness boundary tests (#259). Each asserts a SAFE
// Rust call cannot drive the underlying MLX C++ into signed-overflow UB, and
// instead returns a typed error. Hand-traced against ops.cpp:
//   - roll(Shape)/roll(Shape, axis) sum shift into an unchecked `int` (~6369).
//   - roll(...) negates a negative shift via `(-sh)` (~6351) — UB at i32::MIN.
//   - tile builds out dims as `reps[i]*shape[i]` in unchecked `int` (~1350).
// ---------------------------------------------------------------------------

#[test]
fn roll_multi_shift_sum_overflow_is_typed_error() {
  // i32::MAX + 1 overflows the C++ `int total_shift` accumulator. The wrapper
  // sums in Rust with checked_add and surfaces a typed ArithmeticOverflow
  // rather than wrapping (or invoking UB).
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &[4i32]).unwrap();
  let r = ops::shape::roll(&a, &[i32::MAX, 1]);
  assert!(
    matches!(r, Err(mlxrs::Error::ArithmeticOverflow(_))),
    "expected Err(ArithmeticOverflow) for overflowing shift sum, got {r:?}"
  );
  // Method form must guard identically (it is a thin forward).
  let r2 = a.roll(&[i32::MAX, 1]);
  assert!(matches!(r2, Err(mlxrs::Error::ArithmeticOverflow(_))));
}

#[test]
fn roll_axis_multi_shift_sum_overflow_is_typed_error() {
  // Same unchecked-sum path as `roll`, via the single-axis overload (~6390).
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &(2, 2)).unwrap();
  let r = ops::shape::roll_axis(&a, &[i32::MAX, i32::MAX], 0);
  assert!(
    matches!(r, Err(mlxrs::Error::ArithmeticOverflow(_))),
    "expected Err(ArithmeticOverflow) for overflowing shift sum, got {r:?}"
  );
}

#[test]
fn roll_int_min_shift_is_typed_range_error() {
  // A single i32::MIN shift reaches the `(-sh)` negation in MLX — UB. The
  // wrapper rejects it as a typed OutOfRange across all three roll forms.
  //
  // INTENTIONAL STRICTER-THAN-MLX CONTRACT (#259): mlx-core would
  // no-op an i32::MIN shift on a size-0 axis (its `size == 0` check at
  // ops.cpp ~6348 runs before the `(-sh)` negation), so this rejection is
  // slightly stricter than mlx-core for that degenerate empty case. We take
  // the simpler always-reject contract rather than axis-size-dependent logic:
  // i32::MIN is never needed since the shift is taken modulo the axis size,
  // and the rejection is a typed error, never UB. This test locks that in.
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &[4i32]).unwrap();

  let r = ops::shape::roll(&a, &[i32::MIN]);
  assert!(
    matches!(r, Err(mlxrs::Error::OutOfRange(_))),
    "roll(i32::MIN) should be OutOfRange, got {r:?}"
  );

  let r_axis = ops::shape::roll_axis(&a, &[i32::MIN], 0);
  assert!(
    matches!(r_axis, Err(mlxrs::Error::OutOfRange(_))),
    "roll_axis(i32::MIN) should be OutOfRange, got {r_axis:?}"
  );

  let r_axes = ops::shape::roll_axes(&a, &[i32::MIN], &[0]);
  assert!(
    matches!(r_axes, Err(mlxrs::Error::OutOfRange(_))),
    "roll_axes(i32::MIN) should be OutOfRange, got {r_axes:?}"
  );
}

#[test]
fn roll_sum_to_int_min_is_typed_range_error() {
  // Two shifts that sum to exactly i32::MIN (without overflowing the add):
  // i32::MIN/2 + i32::MIN/2 == i32::MIN. The checked sum succeeds but the total
  // is i32::MIN, whose negation is UB, so it is rejected as OutOfRange (not
  // ArithmeticOverflow — the addition itself did not overflow).
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &[4i32]).unwrap();
  let half = i32::MIN / 2; // -1073741824; 2*half == i32::MIN exactly
  let r = ops::shape::roll(&a, &[half, half]);
  assert!(
    matches!(r, Err(mlxrs::Error::OutOfRange(_))),
    "expected Err(OutOfRange) for shift sum == i32::MIN, got {r:?}"
  );
}

#[test]
fn tile_huge_rep_output_dim_overflow_is_typed_error() {
  // shape [2], reps [i32::MAX] -> out dim 2*i32::MAX overflows i32. The wrapper
  // bounds the product in i64 and returns a typed ArithmeticOverflow.
  let a = Array::from_slice::<f32>(&[1.0, 2.0], &[2i32]).unwrap();
  let r = ops::shape::tile(&a, &[i32::MAX]);
  assert!(
    matches!(r, Err(mlxrs::Error::ArithmeticOverflow(_))),
    "expected Err(ArithmeticOverflow) for overflowing tile out dim, got {r:?}"
  );
  let r2 = a.tile(&[i32::MAX]);
  assert!(matches!(r2, Err(mlxrs::Error::ArithmeticOverflow(_))));
}

#[test]
fn tile_negative_reps_is_typed_range_error() {
  // A negative rep would flow into the broadcast/output shape; rejected as a
  // typed OutOfRange before the FFI call.
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let r = ops::shape::tile(&a, &[-1]);
  assert!(
    matches!(r, Err(mlxrs::Error::OutOfRange(_))),
    "expected Err(OutOfRange) for negative reps, got {r:?}"
  );
  // Negative in a leading (reps longer than ndim) position is also caught.
  let r2 = ops::shape::tile(&a, &[-2, 2]);
  assert!(
    matches!(r2, Err(mlxrs::Error::OutOfRange(_))),
    "expected Err(OutOfRange) for leading negative reps, got {r2:?}"
  );
}

#[test]
fn tile_zero_reps_yields_empty_dim() {
  // reps == 0 is VALID in MLX (yields a size-0 output dim) and must NOT be
  // rejected by the overflow guard. shape [3] tiled by [0] -> shape [0].
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let t = ops::shape::tile(&a, &[0]).unwrap();
  assert_eq!(t.shape(), vec![0]);
}

#[test]
fn tile_multi_non_unit_reps_pass_intermediate_rank_guard() {
  // The intermediate-rank guard (#259) caps `max(reps.len(), ndim) +
  // count(reps != 1)` — the rank of tile's expand/broadcast intermediates, which
  // carry one extra dim per non-unit rep. This must NOT reject a legitimate
  // multi-non-unit-rep tile: here both reps are non-unit (2 extra dims), so the
  // guard sees intermediate rank 2 + 2 == 4, far under the cap, and the op runs
  // through to a correct materialised result.
  //
  // The genuine overflow path needs ~i32::MAX reps (an ~8GB+ slice that cannot
  // be allocated in a test), so the cap CROSSING is covered synthetically by the
  // `tile_intermediate_rank_boundary` unit test in src/ops/shape.rs (it feeds
  // crafted ndim/reps to `check_tile_intermediate_rank` with no array). This
  // integration test instead pins the no-false-reject contract end to end.
  // shape (2,) tiled by [2, 3] -> (2, 6); the leading rep prepends an axis and
  // both reps tile, so the result is contiguous and readable.
  let a = Array::from_slice::<f32>(&[1.0, 2.0], &[2i32]).unwrap();
  let mut t = ops::shape::tile(&a, &[2, 3]).unwrap();
  assert_eq!(t.shape(), vec![2, 6]);
  assert_eq!(
    t.to_vec::<f32>().unwrap(),
    vec![
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // row 0
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // row 1
    ]
  );
}

#[test]
fn as_strided_rejects_negative_dim() {
  // Per the docs, `validate_dims` rejects any negative dim before any FFI
  // call. Locks in the recoverable-error path so a regression that drops
  // the check (e.g. moves it past `with_shape`) is caught here.
  let a = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &[4i32]).unwrap();
  // Build the `&[i32]` slice with a negative dim — `&(usize, usize)`
  // can't express this since `usize` is unsigned, so we use the slice
  // impl of `IntoShape` directly.
  let shape: &[i32] = &[-1, 2];
  // SAFETY: rejection happens via `validate_dims` BEFORE any FFI call;
  // no buffer access on the error path.
  let r = unsafe { ops::shape::as_strided(&a, &shape, &[2i64, 1], 0) };
  match r {
    Err(mlxrs::Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("validate_dims"),
        "context names the validator: {}",
        p.context()
      );
      assert_eq!(p.requirement(), "must be non-negative");
      assert!(
        p.value().contains("-1"),
        "value names the offending dim: {}",
        p.value()
      );
    }
    other => panic!("negative dim must Err(OutOfRange), got {other:?}"),
  }
}
