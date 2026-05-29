//! Branch D misc happy-path tests: argmin, cum*, sort/argsort/topk/partition,
//! clip, *_like, astype.

use mlxrs::{Array, Dtype, ops};

// ───────── argmin (companion to argmax in archetypes.rs) ─────────

#[test]
fn argmin_arange_5_yields_0() {
  // argmin over [0, 1, 2, 3, 4] is index 0. mlx returns U32 for index outputs.
  let a = Array::arange(0.0, 5.0, 1.0).unwrap();
  let mut r = ops::misc::argmin(&a, None, false).unwrap();
  assert_eq!(r.item::<u32>().unwrap(), 0);
}

#[test]
fn argmin_axis_2x3_yields_per_row_index() {
  // 2×3 = [[5, 1, 3], [2, 4, 0]]; argmin along axis=1 → [1, 2]
  let data = [5.0_f32, 1.0, 3.0, 2.0, 4.0, 0.0];
  let a = Array::from_slice(&data, &(2, 3)).unwrap();
  let mut r = a.argmin(Some(1), false).unwrap();
  assert_eq!(r.shape(), vec![2]);
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![1, 2]);
}

// ───────── cumulative reductions ─────────

#[test]
fn cumsum_arange_5_yields_running_total() {
  // cumsum([0,1,2,3,4], axis=0, reverse=false, inclusive=true) = [0,1,3,6,10]
  let a = Array::arange(0.0, 5.0, 1.0).unwrap();
  let mut r = ops::misc::cumsum(&a, 0, false, true).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn cumprod_method_arange_1_to_4_yields_factorials() {
  // cumprod([1,2,3,4], axis=0, inclusive) = [1, 2, 6, 24]
  let a = Array::arange(1.0, 5.0, 1.0).unwrap();
  let mut r = a.cumprod(0, false, true).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 6.0, 24.0]);
}

#[test]
fn cummax_running_maximum() {
  // cummax([1, 3, 2, 5, 4]) = [1, 3, 3, 5, 5]
  let a = Array::from_slice(&[1.0_f32, 3.0, 2.0, 5.0, 4.0], &[5]).unwrap();
  let mut r = ops::misc::cummax(&a, 0, false, true).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 3.0, 3.0, 5.0, 5.0]);
}

#[test]
fn cummin_running_minimum() {
  // cummin([4, 2, 5, 1, 3]) = [4, 2, 2, 1, 1]
  let a = Array::from_slice(&[4.0_f32, 2.0, 5.0, 1.0, 3.0], &[5]).unwrap();
  let mut r = a.cummin(0, false, true).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![4.0, 2.0, 2.0, 1.0, 1.0]);
}

// ───────── sort / argsort ─────────

#[test]
fn sort_unsorted_1d_yields_ascending() {
  let a = Array::from_slice(&[3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], &[7]).unwrap();
  let mut r = ops::misc::sort(&a).unwrap();
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0]
  );
}

#[test]
fn sort_axis_2x3_per_row() {
  // [[3,1,2], [6,4,5]] → axis=1 → [[1,2,3], [4,5,6]]
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0, 6.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = a.sort_axis(1).unwrap();
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn argsort_yields_index_permutation_u32() {
  // [3, 1, 2] → argsort = [1, 2, 0]
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0], &[3]).unwrap();
  let mut r = ops::misc::argsort(&a).unwrap();
  assert_eq!(r.dtype().unwrap(), Dtype::U32);
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![1, 2, 0]);
}

// ───────── multi-dim no-axis: flatten contract ─────────
//
// The no-axis sort/argsort/topk/partition wrappers document "operating on the
// flattened array". 1-D inputs can't distinguish flatten-first from
// axis-default behavior; these tests use 2-D inputs and assert the output
// shape collapses to 1-D, locking in the flatten semantics. Copilot PR #8.

#[test]
fn sort_no_axis_flattens_2d() {
  // 2x3 input must produce a 1-D length-6 result, sorted globally.
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0, 6.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = ops::misc::sort(&a).unwrap();
  assert_eq!(r.shape(), vec![6]);
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn argsort_no_axis_flattens_2d() {
  // 2x3 input must produce a 1-D length-6 index permutation.
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0, 6.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = ops::misc::argsort(&a).unwrap();
  assert_eq!(r.shape(), vec![6]);
  assert_eq!(r.dtype().unwrap(), Dtype::U32);
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![1, 2, 0, 4, 5, 3]);
}

#[test]
fn topk_no_axis_flattens_2d() {
  // 2x3 input, k=3 → 1-D length-3 result containing the 3 largest globally
  // (set: {4,5,6}; topk doesn't sort within the result).
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0, 6.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = ops::misc::topk(&a, 3).unwrap();
  assert_eq!(r.shape(), vec![3]);
  let mut v = r.to_vec::<f32>().unwrap();
  v.sort_by(|x, y| x.partial_cmp(y).unwrap());
  assert_eq!(v, vec![4.0, 5.0, 6.0]);
}

#[test]
fn partition_no_axis_flattens_2d() {
  // 2x3 input, kth=3 (over flattened length-6) → 1-D length-6 result with
  // index 3 holding the 4th-smallest element of the whole array.
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0, 6.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = ops::misc::partition(&a, 3).unwrap();
  assert_eq!(r.shape(), vec![6]);
  let v = r.to_vec::<f32>().unwrap();
  assert_eq!(v[3], 4.0); // 4th-smallest of {1,2,3,4,5,6} = 4
  for x in &v[..3] {
    assert!(*x <= 4.0, "lower side must be ≤ pivot, got {x}");
  }
  for x in &v[4..] {
    assert!(*x >= 4.0, "upper side must be ≥ pivot, got {x}");
  }
}

#[test]
fn argsort_axis_per_row_u32() {
  // [[3,1,2],[6,4,5]] axis=1 → [[1,2,0],[1,2,0]]
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0, 6.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = a.argsort_axis(1).unwrap();
  assert_eq!(r.dtype().unwrap(), Dtype::U32);
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![1, 2, 0, 1, 2, 0]);
}

// ───────── topk / partition ─────────

#[test]
fn topk_returns_k_largest_unsorted() {
  // mlx topk returns the k largest values (unsorted among themselves).
  // For [1, 5, 2, 4, 3] with k=3 the SET of returned values must be {3,4,5}.
  let a = Array::from_slice(&[1.0_f32, 5.0, 2.0, 4.0, 3.0], &[5]).unwrap();
  let mut r = ops::misc::topk(&a, 3).unwrap();
  let mut v = r.to_vec::<f32>().unwrap();
  v.sort_by(|x, y| x.partial_cmp(y).unwrap());
  assert_eq!(v, vec![3.0, 4.0, 5.0]);
}

#[test]
fn topk_axis_per_row_largest_two() {
  // 2×4 = [[1,5,2,4],[8,6,7,3]]; axis=1, k=2 → each row's two largest as a set.
  // mlx's topk_axis returns a sliced (non-contiguous) view; verify via shape
  // and a per-row reduction that does not require materializing the buffer.
  let a = Array::from_slice(&[1.0_f32, 5.0, 2.0, 4.0, 8.0, 6.0, 7.0, 3.0], &(2, 4)).unwrap();
  let r = a.topk_axis(2, 1).unwrap();
  assert_eq!(r.shape(), vec![2, 2]);
  // Sum along axis=1 forces materialization and avoids the contiguity gate.
  // Row 0's top-2 are {4,5} (sum 9); row 1's are {7,8} (sum 15).
  let mut row_sums = r.sum_axes(&[1], false).unwrap();
  assert_eq!(row_sums.to_vec::<f32>().unwrap(), vec![9.0, 15.0]);
}

#[test]
fn partition_kth_element_is_in_position() {
  // After partition with kth=2 on [3,1,4,1,5,9,2,6], element at index 2 must
  // be the 3rd-smallest (= 2): elements at idx<2 are ≤ 2, elements at idx>2 are ≥ 2.
  let a = Array::from_slice(&[3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], &[8]).unwrap();
  let mut r = ops::misc::partition(&a, 2).unwrap();
  let v = r.to_vec::<f32>().unwrap();
  assert_eq!(v[2], 2.0);
  for x in &v[..2] {
    assert!(*x <= 2.0, "lower side must be ≤ pivot, got {x}");
  }
  for x in &v[3..] {
    assert!(*x >= 2.0, "upper side must be ≥ pivot, got {x}");
  }
}

#[test]
fn partition_axis_method_form() {
  // 2×3 = [[3,1,2], [6,4,5]]; partition along axis=1 with kth=1 →
  // each row's middle element is the 2nd-smallest of that row (= 2 and 5).
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0, 6.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = a.partition_axis(1, 1).unwrap();
  let v = r.to_vec::<f32>().unwrap();
  assert_eq!(v[1], 2.0);
  assert_eq!(v[4], 5.0);
}

// ───────── clip ─────────

#[test]
fn clip_with_scalar_clamps_into_range() {
  // [-2, -1, 0, 1, 2] clipped to [-1, 1] → [-1, -1, 0, 1, 1]
  let a = Array::from_slice(&[-2.0_f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
  let mut r = ops::misc::clip_with_scalar(&a, -1.0, 1.0).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn clip_with_array_bounds() {
  // [0,1,2,3,4] clipped by min=1 (scalar bcast), max=3 (scalar bcast), array form.
  let a = Array::arange(0.0, 5.0, 1.0).unwrap();
  let lo = Array::full::<f32>(&[1], 1.0).unwrap();
  let hi = Array::full::<f32>(&[1], 3.0).unwrap();
  let mut r = ops::misc::clip(&a, &lo, &hi).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 2.0, 3.0, 3.0]);
}

#[test]
fn clip_method_form_matches_freefn() {
  let a = Array::arange(0.0, 5.0, 1.0).unwrap();
  let mut r = a.clip_with_scalar(1.0, 3.0).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 2.0, 3.0, 3.0]);
}

// ───────── *_like constructors ─────────

#[test]
fn ones_like_inherits_shape_and_dtype() {
  let a = Array::zeros::<f32>(&(2, 3)).unwrap();
  let mut r = ops::misc::ones_like(&a).unwrap();
  assert_eq!(r.shape(), vec![2, 3]);
  assert_eq!(r.dtype().unwrap(), Dtype::F32);
  assert!(r.to_vec::<f32>().unwrap().iter().all(|&x| x == 1.0));
}

#[test]
fn zeros_like_method_form() {
  let a = Array::ones::<f32>(&(2, 2)).unwrap();
  let mut r = a.zeros_like().unwrap();
  assert_eq!(r.shape(), vec![2, 2]);
  assert!(r.to_vec::<f32>().unwrap().iter().all(|&x| x == 0.0));
}

#[test]
fn full_like_fills_with_value() {
  let a = Array::zeros::<f32>(&(2, 2)).unwrap();
  let mut r = ops::misc::full_like(&a, 7.5).unwrap();
  assert_eq!(r.shape(), vec![2, 2]);
  assert!(r.to_vec::<f32>().unwrap().iter().all(|&x| x == 7.5));
}

#[test]
fn full_like_method_form() {
  let a = Array::ones::<f32>(&(3,)).unwrap();
  let mut r = a.full_like(2.5).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.5, 2.5, 2.5]);
}

// ───────── astype ─────────

#[test]
fn astype_f32_to_i32_truncates() {
  // [0.0, 1.5, 2.9] → astype I32 → [0, 1, 2]
  let a = Array::from_slice(&[0.0_f32, 1.5, 2.9], &[3]).unwrap();
  let mut r = ops::misc::astype(&a, Dtype::I32).unwrap();
  assert_eq!(r.dtype().unwrap(), Dtype::I32);
  assert_eq!(r.to_vec::<i32>().unwrap(), vec![0, 1, 2]);
}

#[test]
fn astype_method_form_changes_dtype() {
  let a = Array::ones::<f32>(&[3]).unwrap();
  let mut r = a.astype(Dtype::U32).unwrap();
  assert_eq!(r.dtype().unwrap(), Dtype::U32);
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![1, 1, 1]);
}

// ───────── view (bit-preserving reinterpret) ─────────

/// Bit-preserving i32 → u32 reinterpret: a negative i32 value keeps its
/// high bit as the u32 sign-bit. The AWQ converter relies on this when
/// AutoAWQ checkpoints store `qweight` as signed `torch.int32` whose high
/// nibble's MSB is set (the converter then shifts-and-masks the resulting
/// u32 — `astype` would clamp negatives to 0, which is what we MUST avoid).
#[test]
fn view_i32_to_u32_preserves_bit_pattern() {
  // `0xF0FF_FFFF` as a signed 32-bit: -251_658_241 (the high nibble is 0xF,
  // which is what AWQ would unpack from the top 4 bits of the word).
  let raw: u32 = 0xF0FF_FFFF;
  let signed: i32 = raw as i32;
  assert!(
    signed < 0,
    "fixture must be a negative i32 to exercise the sign bit"
  );
  let a = Array::from_slice(&[signed], &[1]).unwrap();
  let mut r = ops::misc::view(&a, Dtype::U32).unwrap();
  assert_eq!(r.dtype().unwrap(), Dtype::U32);
  // Bit-preserving: the u32 value must equal the original raw pattern, NOT
  // a clamped / saturated cast (which astype would produce).
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![raw]);
}

/// Same-width view is shape-preserving (no last-axis rescale).
#[test]
fn view_same_width_preserves_shape() {
  let a = Array::from_slice(&[1_i32, 2, 3, 4], &(2, 2)).unwrap();
  let mut r = ops::misc::view(&a, Dtype::U32).unwrap();
  assert_eq!(r.shape(), vec![2, 2]);
  assert_eq!(r.dtype().unwrap(), Dtype::U32);
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![1, 2, 3, 4]);
}

// ───────── argpartition / softmax_axis (method-form bridges, #21) ─────────

#[test]
fn argpartition_method_form() {
  // [3,1,2], kth=0 → position 0 holds the index of the minimum (1 @ idx 1).
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0], &[3]).unwrap();
  let mut r = a.argpartition(0).unwrap();
  assert_eq!(r.dtype().unwrap(), Dtype::U32);
  assert_eq!(r.to_vec::<u32>().unwrap()[0], 1);
}

#[test]
fn argpartition_axis_method_form() {
  // 2×3 [[3,1,2],[6,4,5]], axis=1, kth=0 → each row's position-0 index
  // points to that row's minimum (column 1 in both rows).
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0, 6.0, 4.0, 5.0], &(2, 3)).unwrap();
  let mut r = a.argpartition_axis(0, 1).unwrap();
  assert_eq!(r.dtype().unwrap(), Dtype::U32);
  assert_eq!(r.shape(), vec![2, 3]);
  let v = r.to_vec::<u32>().unwrap();
  assert_eq!((v[0], v[3]), (1, 1));
}

#[test]
fn softmax_axis_method_form() {
  // softmax of equal logits along axis=1 is uniform: [0,0] → [0.5, 0.5].
  let a = Array::from_slice(&[0.0_f32, 0.0], &(1, 2)).unwrap();
  let mut r = a.softmax_axis(1, false).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![0.5, 0.5]);
}

#[test]
fn stop_gradient_is_forward_identity() {
  // stop_gradient is a forward no-op: shape, dtype, and values are preserved.
  // (It only severs the backward pass, which mlxrs cannot yet exercise without
  // a value_and_grad wrapper — autograd lands in a later milestone.)
  let a = Array::from_slice(&[1.5f32, -2.0, 3.25, 0.0], &[2, 2]).unwrap();
  let mut sg = a.stop_gradient().unwrap();
  assert_eq!(sg.shape(), &[2, 2]);
  assert_eq!(sg.dtype().unwrap(), Dtype::F32);
  assert_eq!(sg.to_vec::<f32>().unwrap(), vec![1.5, -2.0, 3.25, 0.0]);
}
