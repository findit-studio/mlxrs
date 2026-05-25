//! Drives the TLS error capture path across multiple threads, and the
//! contiguity guard on buffer-extracting methods.

use std::thread;

#[test]
fn shape_error_returns_err_not_abort() {
  // Reshaping a 4-element array to incompatible shape should produce Err, not abort.
  let r = mlxrs::Array::ones::<f32>(&(2, 2)).and_then(|a| a.reshape(&(3,)));
  assert!(
    matches!(r, Err(mlxrs::Error::Backend { .. })),
    "expected Err(Error::Backend), got {r:?}"
  );
}

#[test]
fn each_thread_has_independent_error_slot() {
  // Each thread should get its own TLS error capture, no cross-talk.
  // Source shape (2,2) has 4 elements; reshape targets must NOT equal 4.
  // Use {5, 6, 7, 8} so every thread's reshape is genuinely incompatible.
  let handles: Vec<_> = (0..4)
    .map(|tid| {
      thread::spawn(move || {
        let r = mlxrs::Array::ones::<f32>(&(2, 2)).and_then(|a| a.reshape(&(5 + tid,)));
        assert!(matches!(r, Err(mlxrs::Error::Backend { .. })));
      })
    })
    .collect();
  for h in handles {
    h.join().unwrap();
  }
}

#[test]
fn to_vec_rejects_non_contiguous_view() {
  // Regression test for the UB pathway Codex flagged: a strided view has the
  // same `mlx_array_size` as its source but reordered strides, so
  // `from_raw_parts(ptr, size)` reads in the wrong layout (and for broadcast
  // views, can read past the allocation entirely). The contiguity guard must
  // convert this into Err(NonContiguous).
  //
  // We construct the view via FFI + from_raw because the safe wrapper doesn't
  // expose transpose/broadcast yet (Phase 4). Going through from_raw is also
  // the exact pathway Codex identified as reachable from safe code.
  use mlxrs_sys::{mlx_array, mlx_array_new, mlx_default_gpu_stream_new, mlx_transpose};

  let src = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  // SAFETY: `Array::into_raw`'s contract — `src` is a valid owned Array;
  // ownership of the raw handle transfers to the caller and `Drop` will not
  // run (the handle is freed manually below).
  let raw_src = unsafe { src.into_raw() };
  // SAFETY: returns this thread's default GPU stream handle, mirroring
  // `stream::default_stream`; the test's `#[ctor]`-installed handler is live,
  // so a failed init surfaces rather than `printf+exit`.
  let stream = unsafe { mlx_default_gpu_stream_new() };
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL
  // ctx) per the mlx-c convention; it is populated by the `mlx_transpose`
  // call below before any use.
  let mut out: mlx_array = unsafe { mlx_array_new() };
  // SAFETY: `raw_src` and `stream` are valid handles (not retained by mlx
  // past the call); `out` is the fresh out-param allocated above; the rc is
  // asserted on the next line.
  let rc = unsafe { mlx_transpose(&mut out, raw_src, stream) };
  assert_eq!(rc, 0, "mlx_transpose failed");
  // SAFETY: `raw_src` is the handle this test owns via `into_raw` (freed
  // exactly once here); `mlx_transpose` does not retain it.
  unsafe {
    let _ = mlxrs_sys::mlx_array_free(raw_src);
  }

  // SAFETY: `Array::from_raw`'s contract — `out` is a valid handle freshly
  // produced by `mlx_transpose`, not aliased elsewhere; the safe `Array`
  // now owns it and frees it on `Drop`.
  let mut view = unsafe { mlxrs::Array::from_raw(out) };
  assert_eq!(view.shape(), vec![3, 2]);

  let r = view.to_vec::<f32>();
  assert!(
    matches!(r, Err(mlxrs::Error::NonContiguous)),
    "expected Err(NonContiguous), got {r:?}"
  );
  let r2 = view.as_slice::<f32>();
  assert!(
    matches!(r2, Err(mlxrs::Error::NonContiguous)),
    "expected Err(NonContiguous), got {r2:?}"
  );
}

#[test]
fn to_vec_works_on_contiguous_array() {
  // Sanity: the guard does not break the happy path.
  let mut a = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let v = a.to_vec::<f32>().unwrap();
  assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn from_slice_rejects_negative_i32_dims() {
  // Without the IntoShape negative-dim guard, `-1i32 as usize` becomes
  // usize::MAX and the shape-product check would multiply that into a value
  // that may match data.len() — handing mlx-c a buffer smaller than the
  // shape says. Must surface as ShapeMismatch.
  let r = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[-1i32, 3]);
  assert!(
    matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "expected Err(ShapeMismatch), got {r:?}"
  );
}

#[test]
fn from_slice_rejects_negative_i32_slice_dims() {
  // Same guard for the &[i32] IntoShape path (escape hatch for runtime shapes).
  let dims: &[i32] = &[2, -3];
  let r = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &dims);
  assert!(
    matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "expected Err(ShapeMismatch), got {r:?}"
  );
}

#[test]
fn from_slice_rejects_overflowing_shape_product() {
  // Three large positive dims whose usize product wraps in release builds.
  // `i32::MAX^3 ≈ 9.9e27` >> `usize::MAX ≈ 1.8e19`, so wrapping is guaranteed.
  // Without checked_mul, the wrapped value could match data.len() and pass.
  let r = mlxrs::Array::from_slice::<f32>(&[1.0], &[i32::MAX, i32::MAX, i32::MAX]);
  assert!(
    matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "expected Err(ShapeMismatch) on overflow, got {r:?}"
  );
}

#[test]
fn slice_rejects_len_ne_ndim() {
  // start/stop/strides length must equal a.ndim() — passing empty against a
  // 2-D array is the "len != ndim" failure mode, not the dangling-pointer
  // one. (The dangling-pointer concern is now handled by dim_ptr's sentinel,
  // so the safe-FFI boundary is closed without rejecting 0-D-scalar slicing.)
  let a = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let r = mlxrs::ops::indexing::slice(&a, &[], &[], &[]);
  assert!(
    matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "expected Err(ShapeMismatch) on len != ndim, got {r:?}"
  );
}

#[test]
fn slice_rejects_mismatched_lengths() {
  // start/stop/strides must agree on length (one entry per axis).
  let a = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let r = mlxrs::ops::indexing::slice(&a, &[0, 0], &[1], &[1, 1]);
  assert!(
    matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "expected Err(ShapeMismatch) on length mismatch, got {r:?}"
  );
}

#[test]
fn slice_accepts_empty_for_zero_dim_scalar() {
  // 0-D scalar input → all three slice arrays must be empty (one entry per
  // axis = zero entries). Empty inputs route through dim_ptr's sentinel,
  // not rejected. Copilot PR #5 finding.
  let empty: [i32; 0] = [];
  let a = mlxrs::Array::from_slice::<f32>(&[42.0], &empty).unwrap();
  assert_eq!(a.ndim(), 0);
  let mut r = mlxrs::ops::indexing::slice(&a, &[], &[], &[]).unwrap();
  assert_eq!(r.shape(), Vec::<usize>::new());
  assert_eq!(r.item::<f32>().unwrap(), 42.0);
}

#[test]
fn sum_axes_empty_returns_clone() {
  // Empty axes = sum over no axes = identity (numpy/mlx semantics). Must
  // short-circuit to clone instead of crossing FFI with a dangling pointer.
  // Codex PR #5 finding 2.
  let mut a = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = mlxrs::ops::reduction::sum_axes(&a, &[], false).unwrap();
  assert_eq!(r.shape(), a.shape());
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    a.to_vec::<f32>().unwrap(),
    "sum over no axes should be identity"
  );
}

#[test]
fn concatenate_rejects_empty_input() {
  // Concatenating zero arrays has no defined result; must reject before FFI.
  // Codex PR #5 finding 3 / dangling-pointer concern for empty Vec::as_ptr().
  let r = mlxrs::ops::shape::concatenate(&[], 0);
  assert!(
    matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "expected Err(ShapeMismatch) on empty input, got {r:?}"
  );
}

#[test]
fn from_slice_zero_element_uses_sentinel() {
  // Zero-element arrays are valid in numpy/mlx. The dangling-pointer concern
  // for Rust's `<&[T]>::as_ptr()` on an empty slice still needs a sentinel —
  // this exercises the data_ptr helper. Codex PR #5 round-2 finding.
  let mut a = mlxrs::Array::from_slice::<f32>(&[], &[0i32]).unwrap();
  assert_eq!(a.shape(), vec![0]);
  assert_eq!(a.size(), 0);
  // 2-D zero-element shape too.
  let b = mlxrs::Array::from_slice::<f32>(&[], &[2i32, 0]).unwrap();
  assert_eq!(b.shape(), vec![2, 0]);
  assert_eq!(b.size(), 0);
  // to_vec on a zero-element contiguous array is just an empty Vec.
  assert_eq!(a.to_vec::<f32>().unwrap(), Vec::<f32>::new());
}

#[test]
fn from_slice_zero_element_all_element_types() {
  // Every Element impl provides its own typed sentinel (Codex PR #5 round-3
  // finding). Verify each compiles + constructs without UB.
  let mut b = mlxrs::Array::from_slice::<bool>(&[], &[0i32]).unwrap();
  assert_eq!(b.shape(), vec![0]);
  assert_eq!(b.to_vec::<bool>().unwrap(), Vec::<bool>::new());

  let mut i = mlxrs::Array::from_slice::<i32>(&[], &[0i32]).unwrap();
  assert_eq!(i.shape(), vec![0]);
  assert_eq!(i.to_vec::<i32>().unwrap(), Vec::<i32>::new());

  let mut u = mlxrs::Array::from_slice::<u32>(&[], &[0i32]).unwrap();
  assert_eq!(u.shape(), vec![0]);
  assert_eq!(u.to_vec::<u32>().unwrap(), Vec::<u32>::new());

  let mut h = mlxrs::Array::from_slice::<half::f16>(&[], &[0i32]).unwrap();
  assert_eq!(h.shape(), vec![0]);
  assert_eq!(h.to_vec::<half::f16>().unwrap(), Vec::<half::f16>::new());
}
