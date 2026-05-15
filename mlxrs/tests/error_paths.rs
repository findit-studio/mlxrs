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
  // SAFETY: from_raw / into_raw round-trip; the stream init mirrors what the
  // safe layer does internally (stream::default_stream).
  let raw_src = unsafe { src.into_raw() };
  let stream = unsafe { mlx_default_gpu_stream_new() };
  let mut out: mlx_array = unsafe { mlx_array_new() };
  let rc = unsafe { mlx_transpose(&mut out, raw_src, stream) };
  assert_eq!(rc, 0, "mlx_transpose failed");
  unsafe {
    let _ = mlxrs_sys::mlx_array_free(raw_src);
  }

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
