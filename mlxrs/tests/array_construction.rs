//! Value/shape/dtype coverage for `array::construction` (#260).
//!
//! `array/construction.rs` had ZERO direct coverage. The pre-existing
//! `error_paths.rs` already exercises `from_slice`'s negative-dim /
//! overflow / zero-element guards and `archetypes.rs` smoke-tests the
//! happy path of `ones`/`zeros`/`arange`, so this file deliberately
//! targets the GENUINE gaps:
//!   * `eye` and `linspace` — completely untested anywhere; plus `eye`'s
//!     non-square (`m`) and off-diagonal (`k`) parameters (#259 construction
//!     parity).
//!   * `full`'s exact-fill contract: `value` is a `T`, so the 0-d scalar handed
//!     to `mlx_full` is already dtype `T` and no value cast occurs — verify on
//!     an integer dtype and on a wide `u32` value the old f32/i32 path couldn't
//!     represent.
//!   * `arange`'s half-open `[start, stop)` semantics (non-unit / negative
//!     steps, empty result) and its `T`-driven output dtype, plus the #286/#287
//!     soundness guards: zero step, non-finite / over-`i32` / wrong-direction
//!     lengths, and integer seed values outside `T`'s range — all rejected (or
//!     returned empty) before the FFI rather than reaching mlx's `static_cast`
//!     UB.
//!   * `linspace`'s inclusive `[start, stop]` endpoints, the `num == 1` special
//!     case (C++ returns `astype(array({start}), dtype)`), its `T`-driven dtype,
//!     and the #286 integer-endpoint guard (rounded through `f32` to match mlx's
//!     inner ramp dtype, so it catches an endpoint at the dtype max).
//!   * `ones`/`zeros` dtype correctness on non-f32 element types.
//!
//! Crate accessor rule (`feedback_no_implicit_eval`): `item`/`to_vec`/
//! `as_slice` are `&mut self` and force an explicit eval inside themselves,
//! so the build → read pattern used by the sibling test files (`archetypes`,
//! `ops_arithmetic`) is followed verbatim — no separate `eval()` call is
//! needed before a `&mut self` accessor.

use mlxrs::{Array, Dtype};

// ───────── ones / zeros ─────────

#[test]
fn ones_values_shape_and_dtype() {
  let mut a = Array::ones::<f32>(&(2, 3)).unwrap();
  assert_eq!(a.shape(), vec![2, 3]);
  assert_eq!(a.size(), 6);
  assert_eq!(a.ndim(), 2);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![1.0; 6]);
}

#[test]
fn ones_i32_has_integer_dtype_and_values() {
  // dtype is driven purely by the `T` type parameter, independent of value.
  let mut a = Array::ones::<i32>(&(4,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![1_i32; 4]);
}

#[test]
fn zeros_values_shape_and_dtype() {
  let mut a = Array::zeros::<f32>(&(3, 2)).unwrap();
  assert_eq!(a.shape(), vec![3, 2]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0; 6]);
}

#[test]
fn zeros_u32_dtype_and_values() {
  let mut a = Array::zeros::<u32>(&(5,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U32);
  assert_eq!(a.to_vec::<u32>().unwrap(), vec![0_u32; 5]);
}

// ───────── full ─────────

#[test]
fn full_f32_fills_every_element() {
  let mut a = Array::full::<f32>(&(2, 2), 7.5).unwrap();
  assert_eq!(a.shape(), vec![2, 2]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![7.5; 4]);
}

#[test]
fn full_fills_requested_integer_dtype() {
  // `full::<i32>` takes an exact `i32` value; the 0-d scalar handed to
  // `mlx_full` is already I32, so no cast occurs and the result is the I32
  // value 3.
  let mut a = Array::full::<i32>(&(3,), 3).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![3_i32; 3]);
}

#[test]
fn full_i32_fill_keeps_dtype_and_value() {
  // #259: the fill value is an exact `i32`, so dtype + value round-trip exactly
  // (the 0-d scalar handed to `mlx_full` is already I32 — no float intermediate
  // that could lose precision above 2^24).
  let mut a = Array::full::<i32>(&(2,), 7).unwrap();
  assert_eq!(a.shape(), vec![2]);
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![7_i32, 7]);
}

#[test]
fn full_u32_wide_value_is_exact() {
  // #259: `value: T` makes the fill exact for any in-range `T` value. A `u32`
  // above i32::MAX (3e9) is faithfully represented — the old `impl Into<f64>` +
  // i32-scalar path could not (it rejected/clipped it). No cast, no wrap.
  let mut a = Array::full::<u32>(&(2,), 3_000_000_000u32).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U32);
  assert_eq!(
    a.to_vec::<u32>().unwrap(),
    vec![3_000_000_000, 3_000_000_000]
  );
}

// ───────── eye ─────────

#[test]
fn eye_3_is_identity_matrix() {
  // mlx `eye(n, n, k=0)` → row-major identity. Flattened, the 1.0s sit on
  // indices 0, 4, 8 (the i*(n+1) diagonal) and everything else is 0.0.
  let mut a = Array::eye::<f32>(3, None, 0).unwrap();
  assert_eq!(a.shape(), vec![3, 3]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(
    a.to_vec::<f32>().unwrap(),
    vec![
      1.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, //
      0.0, 0.0, 1.0,
    ]
  );
}

#[test]
fn eye_1_is_single_one() {
  let mut a = Array::eye::<f32>(1, None, 0).unwrap();
  assert_eq!(a.shape(), vec![1, 1]);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![1.0]);
}

#[test]
fn eye_non_square_uses_m_for_columns() {
  // #259: `m = Some(4)` → 3 rows, 4 columns; the main diagonal (k=0) holds
  // ones at flat indices 0, 5, 10 (i*(m+1)) and the rest are zero.
  let mut a = Array::eye::<f32>(3, Some(4), 0).unwrap();
  assert_eq!(a.shape(), vec![3, 4]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(
    a.to_vec::<f32>().unwrap(),
    vec![
      1.0, 0.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, 0.0, //
      0.0, 0.0, 1.0, 0.0,
    ]
  );
}

#[test]
fn eye_positive_k_is_super_diagonal() {
  // #259: `k = 1` shifts the ones one column above the main diagonal. For a
  // 3×3 that is flat indices 1 and 5; index 0 (main diagonal) is now 0.0.
  let mut a = Array::eye::<f32>(3, None, 1).unwrap();
  assert_eq!(a.shape(), vec![3, 3]);
  let v = a.to_vec::<f32>().unwrap();
  assert_eq!(v[1], 1.0);
  assert_eq!(v[5], 1.0);
  assert_eq!(v[0], 0.0);
  assert_eq!(v[8], 0.0);
}

#[test]
fn eye_negative_k_is_sub_diagonal() {
  // #259: `k = -1` shifts the ones one row below the main diagonal — a valid
  // negative offset. For a 3×3 that is flat indices 3 and 7.
  let mut a = Array::eye::<f32>(3, None, -1).unwrap();
  assert_eq!(a.shape(), vec![3, 3]);
  let v = a.to_vec::<f32>().unwrap();
  assert_eq!(v[3], 1.0);
  assert_eq!(v[7], 1.0);
  assert_eq!(v[0], 0.0);
  assert_eq!(v[4], 0.0);
}

#[test]
fn eye_k_i32_min_is_rejected_not_ub() {
  // #259 / Codex: mlx's eye evaluates `-k`, so k == i32::MIN would overflow in
  // C++ (UB). The wrapper rejects it with a typed error instead of calling FFI.
  let err = Array::eye::<f32>(3, None, i32::MIN).unwrap_err();
  assert!(matches!(err, mlxrs::Error::OutOfRange(_)), "got {err:?}");
}

// ───────── arange ─────────

#[test]
fn arange_unit_step_is_half_open() {
  // [0, 5) step 1 → 0,1,2,3,4 (stop is exclusive). Default f32 dtype.
  let mut a = Array::arange::<f32>(0.0, 5.0, 1.0).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn arange_non_unit_step() {
  // [0, 1) step 0.25 → 0, 0.25, 0.5, 0.75 (4 values, 1.0 excluded).
  let mut a = Array::arange::<f32>(0.0, 1.0, 0.25).unwrap();
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0, 0.25, 0.5, 0.75]);
}

#[test]
fn arange_negative_step_counts_down() {
  // [5, 0) step -1 → 5,4,3,2,1 (stop 0 excluded).
  let mut a = Array::arange::<f32>(5.0, 0.0, -1.0).unwrap();
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn arange_empty_when_start_equals_stop() {
  let mut a = Array::arange::<f32>(2.0, 2.0, 1.0).unwrap();
  assert_eq!(a.size(), 0);
  assert_eq!(a.to_vec::<f32>().unwrap(), Vec::<f32>::new());
}

#[test]
fn arange_i32_dtype_and_values() {
  // #286: the output dtype follows `T`. `arange::<i32>(0, 5, 1)` is an I32
  // array `[0, 1, 2, 3, 4]` (stop exclusive), not f32.
  let mut a = Array::arange::<i32>(0.0, 5.0, 1.0).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn arange_u32_wide_value_is_exact() {
  // #286: f64 bounds keep an integer range exact above the f32 2^24 window. A
  // u32 range starting at 3e9 (> i32::MAX, below 2^32 so in u32 range, and not
  // exactly representable in f32) yields the requested indices rather than the
  // empty/short array f32-rounded bounds would have produced.
  let mut a = Array::arange::<u32>(3_000_000_000.0, 3_000_000_005.0, 1.0).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U32);
  assert_eq!(
    a.to_vec::<u32>().unwrap(),
    vec![
      3_000_000_000,
      3_000_000_001,
      3_000_000_002,
      3_000_000_003,
      3_000_000_004,
    ]
  );
}

#[test]
fn arange_zero_step_is_rejected_not_ub() {
  // #287: step 0 makes mlx's length `ceil((stop - start) / step)` non-finite
  // (NaN when stop == start, -inf when descending), which it `static_cast`s to
  // int — C++ UB. Rejected pre-FFI for both the empty and descending shapes.
  assert!(matches!(
    Array::arange::<f32>(2.0, 2.0, 0.0).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
  assert!(matches!(
    Array::arange::<f32>(5.0, 0.0, 0.0).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_huge_wrong_direction_range_is_empty_not_ub() {
  // #287: a finite but huge-negative length (ceil(-1e20) is far below i32::MIN)
  // is exactly what mlx's one-sided `> INT_MAX` guard misses, so its
  // `static_cast<int>` is UB. The wrapper returns an empty array instead.
  let mut a = Array::arange::<f32>(0.0, -1.0e20, 1.0).unwrap();
  assert_eq!(a.size(), 0);
  assert_eq!(a.to_vec::<f32>().unwrap(), Vec::<f32>::new());
}

#[test]
fn arange_length_over_i32_max_is_rejected() {
  // #287: a length above i32::MAX (3e9 elements) is rejected with a typed error
  // rather than forwarded into mlx's int cast.
  assert!(matches!(
    Array::arange::<f32>(0.0, 3.0e9, 1.0).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_u8_seed_out_of_range_is_rejected_not_ub() {
  // #286: mlx `static_cast`s the seeds `start` and `start + step` into the C++
  // integer type. `start = 300` is outside u8, so the cast would be UB — the
  // wrapper rejects it.
  assert!(matches!(
    Array::arange::<u8>(300.0, 301.0, 1.0).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_i32_seed_out_of_range_is_rejected_not_ub() {
  // #286: a short range (length 1) whose start (3e9) is outside i32 still
  // reaches the seed cast → rejected, not UB.
  assert!(matches!(
    Array::arange::<i32>(3.0e9, 3_000_000_001.0, 1.0).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_empty_range_keeps_requested_dtype() {
  // The wrong-direction empty path returns a `T`-typed (here I32) zero-element
  // array, not an f32 one.
  let a = Array::arange::<i32>(5.0, 0.0, 1.0).unwrap();
  assert_eq!(a.size(), 0);
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
}

#[test]
fn arange_signed_accumulation_overflow_is_rejected() {
  // #286 / Codex R1: both seeds (2147483646, 2147483647) fit i32, but mlx's CPU
  // arange accumulates in the promoted `int` and the increment after writing
  // INT_MAX overflows it (UB). The post-last value 2147483650 is out of range,
  // so the signed-path guard rejects it.
  assert!(matches!(
    Array::arange::<i32>(2_147_483_646.0, 2_147_483_650.0, 1.0).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_signed_delta_subtraction_overflow_is_rejected() {
  // #286 / Codex R2: a one-element jump whose seeds (i32::MIN, i32::MAX) both fit
  // i32, but mlx forms `step = next - first` IN i32 = 2147483647 - (-2147483648)
  // = 4294967295, which overflows i32 at the subtraction itself (before any
  // accumulation). The exact i128 recurrence model rejects it on the `delta`.
  assert!(matches!(
    Array::arange::<i32>(-2_147_483_648.0, 0.0, 4_294_967_295.0).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_signed_fractional_step_overflow_is_rejected() {
  // #286 / Codex R2: mlx truncates the seeds, so start 0.5 / step 1.6 accumulate
  // with effective integer delta 2 (trunc(2.1) - trunc(0.5)). A ~1.2e9-element
  // i32 range then overflows the promoted int (post-last ≈ 2.4e9) even though the
  // naive f64 post-last (≈1.92e9) is in range — the i128 model catches it.
  assert!(matches!(
    Array::arange::<i32>(0.5, 1_920_000_000.0, 1.6).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_signed_fractional_boundary_seed_is_accepted() {
  // #286 / Codex R3: the cast truncates toward zero, so a seed of -2147483648.5
  // truncates to i32::MIN — a VALID cast. The truncation-aware guard must accept
  // it (the old raw-f64 bound wrongly rejected it as below i32::MIN).
  let mut a = Array::arange::<i32>(-2_147_483_648.5, -2_147_483_646.5, 1.0).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![i32::MIN, i32::MIN + 1]);
}

#[test]
fn arange_i16_large_delta_is_accepted() {
  // #286 / Codex R3: i8/i16 narrow back to `T` each step, so their promoted adds
  // never overflow `int` — the i32/i64 post-last guard must NOT apply. This i16
  // range has an un-narrowed post-last above i32::MAX but mlx runs it fine, so it
  // must be accepted (only Ok + dtype asserted; the values wrap, backend-defined).
  let a = Array::arange::<i16>(-32768.0, 2_147_549_182.0, 65535.0).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I16);
}

#[test]
fn arange_unsigned_wrap_is_allowed() {
  // #286: unsigned in-`T` arithmetic wraps (defined), so the guard checks only
  // the two seeds and PERMITS a range whose later values wrap past the dtype max
  // — it returns Ok rather than rejecting. u8 [254, 257) step 1 has in-range
  // seeds (254, 255). The exact wrapped values are backend-defined, so only the
  // success + shape/dtype are asserted.
  let a = Array::arange::<u8>(254.0, 257.0, 1.0).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U8);
  assert_eq!(a.size(), 3);
}

#[test]
fn arange_infinite_step_correct_direction_yields_start() {
  // #286 / Codex R1: mlx returns a single `[start]` for an infinite step in the
  // correct direction (NOT an empty array).
  let mut a = Array::arange::<f32>(0.0, 10.0, f32::INFINITY).unwrap();
  assert_eq!(a.size(), 1);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0]);
}

#[test]
fn arange_infinite_step_wrong_direction_is_empty() {
  // Infinite step in the wrong direction is an empty range.
  let a = Array::arange::<f32>(10.0, 0.0, f32::INFINITY).unwrap();
  assert_eq!(a.size(), 0);
}

#[test]
fn arange_infinite_step_integer_seed_out_of_range_is_rejected() {
  // The single `start` mlx emits for an infinite step is still cast into `T`;
  // a u8-out-of-range start is rejected, not UB.
  assert!(matches!(
    Array::arange::<u8>(300.0, 400.0, f32::INFINITY).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_nan_bound_is_rejected() {
  // mlx throws "Cannot compute length" on a NaN bound; the wrapper rejects it
  // with a typed NonFiniteScalar before the FFI.
  assert!(matches!(
    Array::arange::<f32>(0.0, f32::NAN, 1.0).unwrap_err(),
    mlxrs::Error::NonFiniteScalar(_)
  ));
}

#[test]
fn arange_bool_dtype_is_unsupported_even_when_empty() {
  // #286 / Codex R1: mlx rejects bool arange for EVERY range. The empty fast
  // path must not mask it — both a non-empty and a wrong-direction (empty)
  // bool range return UnsupportedDtype.
  assert!(matches!(
    Array::arange::<bool>(0.0, 5.0, 1.0).unwrap_err(),
    mlxrs::Error::UnsupportedDtype(_)
  ));
  assert!(matches!(
    Array::arange::<bool>(5.0, 0.0, 1.0).unwrap_err(),
    mlxrs::Error::UnsupportedDtype(_)
  ));
}

#[test]
fn arange_f32_seed_above_f32_max_is_rejected() {
  // #286 / Codex R4: the f64 bounds are also narrowed for FLOAT outputs. mlx
  // casts start + step (here f64::MAX) into `float`, which is out of f32's finite
  // range — C++ UB. The representability guard rejects it before the FFI.
  assert!(matches!(
    Array::arange::<f32>(0.0, f64::MAX, f64::MAX).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_f16_seed_above_f16_max_is_rejected() {
  // #286: f16's finite max is 65504, so a seed of 100000 narrows out of range.
  assert!(matches!(
    Array::arange::<half::f16>(0.0, 100_000.0, 100_000.0).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn arange_f32_in_range_is_unaffected() {
  // A normal f32 range is unaffected by the float guard.
  let mut a = Array::arange::<f32>(0.0, 5.0, 1.0).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

// ───────── linspace ─────────

#[test]
fn linspace_includes_both_endpoints() {
  // [0, 1] with 5 samples → 0, 0.25, 0.5, 0.75, 1.0 (BOTH ends inclusive,
  // unlike arange). Default f32 dtype.
  let mut a = Array::linspace::<f32>(0.0, 1.0, 5).unwrap();
  assert_eq!(a.size(), 5);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  let v = a.to_vec::<f32>().unwrap();
  let expected = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
  assert_eq!(v.len(), expected.len());
  for (got, want) in v.iter().zip(expected.iter()) {
    assert!((got - want).abs() < 1e-6, "linspace got {got}, want {want}");
  }
}

#[test]
fn linspace_num_1_returns_start_only() {
  // C++ special case: `num == 1` → `astype(array({start}), dtype)` — a
  // single-element array holding `start` (NOT `stop`).
  let mut a = Array::linspace::<f32>(3.0, 9.0, 1).unwrap();
  assert_eq!(a.size(), 1);
  assert_eq!(a.item::<f32>().unwrap(), 3.0);
}

#[test]
fn linspace_i32_dtype_and_values() {
  // #286: output dtype follows `T`. Use a GPU-safe integer dtype (f64 is
  // CPU-only on Metal). [0, 8] in 5 samples → 0, 2, 4, 6, 8.
  let mut a = Array::linspace::<i32>(0.0, 8.0, 5).unwrap();
  assert_eq!(a.size(), 5);
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![0, 2, 4, 6, 8]);
}

#[test]
fn linspace_i32_endpoint_at_dtype_max_is_rejected_not_ub() {
  // #286: linspace evaluates the ramp in f32 then `astype`s to T. i32::MAX
  // (2147483647) rounds UP to 2^31 in f32, which is out of i32 range, so the
  // astype `static_cast` would be UB. The guard rounds the endpoint through f32
  // to match mlx's inner dtype and rejects it.
  assert!(matches!(
    Array::linspace::<i32>(0.0, 2_147_483_647.0, 5).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn linspace_u8_endpoint_out_of_range_is_rejected_not_ub() {
  // #286: stop 300 is outside u8 → the astype cast would be UB; rejected.
  assert!(matches!(
    Array::linspace::<u8>(0.0, 300.0, 5).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn linspace_num_1_ignores_out_of_range_stop() {
  // num == 1 yields `[start]`; `stop` is never sampled, so an out-of-range stop
  // must NOT be rejected (only `start` is range-checked when num == 1).
  let mut a = Array::linspace::<u8>(5.0, 300.0, 1).unwrap();
  assert_eq!(a.size(), 1);
  assert_eq!(a.item::<u8>().unwrap(), 5);
}

#[test]
fn linspace_num_1_f32_rounding_out_of_range_is_rejected() {
  // #286 / Codex R5: num == 1 is astype(array({start}), dtype) where
  // array({start}) is FLOAT32 (TypeToDtype<double> -> float32, vendored
  // dtype.cpp), so start is narrowed f64 -> f32 -> T. i32::MAX (2147483647) rounds
  // UP to 2^31 in f32, which then astype's out of i32 range — UB — so it is
  // rejected (the earlier raw-f64 model wrongly accepted it).
  assert!(matches!(
    Array::linspace::<i32>(2_147_483_647.0, 0.0, 1).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn linspace_num_1_fractional_boundary_start_is_accepted() {
  // #286: under the f64 -> f32 -> i32 cast, start -2147483648.5 rounds to the
  // exact f32 -2147483648.0 and astype's to i32::MIN — a valid cast, accepted.
  let mut a = Array::linspace::<i32>(-2_147_483_648.5, 0.0, 1).unwrap();
  assert_eq!(a.size(), 1);
  assert_eq!(a.item::<i32>().unwrap(), i32::MIN);
}

#[test]
fn linspace_num_0_is_empty_without_narrowing() {
  // #286 / Codex R5: mlx's num == 0 still constructs array(start, f32) /
  // array(stop, f32) (a double -> float narrowing, UB for f64::MAX) before
  // producing the empty result. The wrapper returns the empty array directly, so
  // an out-of-f32-range endpoint is harmless.
  let a = Array::linspace::<f32>(f64::MAX, 0.0, 0).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.size(), 0);
}

#[test]
fn linspace_num_1_f64_output_still_narrows_through_f32() {
  // #286 / Codex R5: array({start}) is float32 even when the OUTPUT is f64, so a
  // num == 1 f64 linspace narrows start f64 -> f32. f64::MAX is out of f32 range,
  // so it is rejected — f64 outputs are NOT exempt for num == 1.
  assert!(matches!(
    Array::linspace::<f64>(f64::MAX, 0.0, 1).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn linspace_f32_endpoint_above_f32_max_is_rejected() {
  // #286 / Codex R4: for num >= 2 the ramp is built in an f32 inner dtype, so a
  // start of f64::MAX is narrowed f64 -> f32 out of range (C++ UB). Rejected.
  assert!(matches!(
    Array::linspace::<f32>(f64::MAX, 0.0, 2).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

#[test]
fn linspace_f16_ramp_in_range_works() {
  // A normal f16 linspace (f64 -> f32 inner -> f16 astype) is accepted and runs.
  let a = Array::linspace::<half::f16>(0.0, 1.0, 5).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::F16);
  assert_eq!(a.size(), 5);
}

#[test]
fn linspace_f16_interior_overshoot_at_max_is_rejected() {
  // #286 / Codex R6: both endpoints are exactly f16::MAX (65504), so an
  // endpoint-only check passes, but the f32 ramp `(1 - t) * 65504 + t * 65504`
  // rounds an INTERIOR sample to 65504.0039 > f16::MAX, which then astype's to f16
  // out of range (UB). The full-ramp margin bound rejects it before the FFI.
  assert!(matches!(
    Array::linspace::<half::f16>(65504.0, 65504.0, 8).unwrap_err(),
    mlxrs::Error::OutOfRange(_)
  ));
}

// ───────── from_slice ─────────

#[test]
fn from_slice_preserves_order_and_shape() {
  // Row-major flatten: the buffer order is preserved 1:1 for a contiguous
  // freshly-built array, and the shape product matches the buffer length.
  let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  assert_eq!(a.shape(), vec![2, 3]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(
    a.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn from_slice_i32_dtype_is_driven_by_element_type() {
  let mut a = Array::from_slice::<i32>(&[10, 20, 30], &(3,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn from_slice_scalar_rank0_round_trips() {
  // Empty shape `[]` → rank-0 scalar (size 1). `item` reads it back.
  let empty: [i32; 0] = [];
  let mut a = Array::from_slice::<f32>(&[42.0], &empty).unwrap();
  assert_eq!(a.ndim(), 0);
  assert_eq!(a.shape(), Vec::<usize>::new());
  assert_eq!(a.size(), 1);
  assert_eq!(a.item::<f32>().unwrap(), 42.0);
}

#[test]
fn from_slice_length_mismatch_is_typed_error() {
  // shape product (2*2 = 4) != data.len() (3) → typed LengthMismatch
  // carrying expected=4, actual=3 (NOT a panic, NOT a Backend string).
  let r = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(2, 2));
  match r {
    Err(mlxrs::Error::LengthMismatch(p)) => {
      assert_eq!(
        p.context(),
        "Array::from_slice: shape product vs data.len()"
      );
      assert_eq!(p.expected(), 4);
      assert_eq!(p.actual(), 3);
    }
    other => panic!("expected Err(LengthMismatch) on shape/len mismatch, got {other:?}"),
  }
}
