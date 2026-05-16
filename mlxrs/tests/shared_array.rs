//! `SharedArray` — frozen, lock-free cross-thread sharing of an `Array`.

use mlxrs::{Array, SharedArray};

/// `SharedArray` must stay `Send + Sync + Clone`. The
/// `static_assertions::assert_impl_all!` inside the type's module is the
/// compile-time guard; this keeps it linked into the test binary.
#[test]
fn shared_array_send_sync_compile_assertions() {
  fn assert_send<T: Send>() {}
  fn assert_sync<T: Sync>() {}
  fn assert_clone<T: Clone>() {}
  assert_send::<SharedArray>();
  assert_sync::<SharedArray>();
  assert_clone::<SharedArray>();
}

#[test]
fn freeze_then_read_no_mut_no_eval() {
  // freeze() evals once and consumes the Array; reads are &self, no eval.
  let shared = Array::ones::<f32>(&(2, 2)).unwrap().freeze().unwrap();
  assert_eq!(shared.to_vec::<f32>().unwrap(), vec![1.0; 4]);
  assert_eq!(shared.as_slice::<f32>().unwrap(), &[1.0; 4]);
  assert_eq!(shared.shape(), vec![2, 2]);
  assert_eq!(shared.ndim(), 2);
  assert_eq!(shared.size(), 4);
  assert!(!shared.is_empty());
}

#[test]
fn freeze_item_scalar() {
  let shared = Array::from_slice(&[42.0f32], &(1,))
    .unwrap()
    .freeze()
    .unwrap();
  assert_eq!(shared.item::<f32>().unwrap(), 42.0);
}

#[test]
fn freeze_dtype_mismatch_errors() {
  let shared = Array::ones::<f32>(&(2,)).unwrap().freeze().unwrap();
  assert!(matches!(
    shared.to_vec::<i32>(),
    Err(mlxrs::Error::DtypeMismatch { .. })
  ));
}

#[test]
fn clone_is_cheap_handle_share() {
  let shared = Array::ones::<f32>(&(3,)).unwrap().freeze().unwrap();
  let c1 = shared.clone();
  let c2 = shared.clone();
  // All three observe the same frozen buffer.
  assert_eq!(c1.to_vec::<f32>().unwrap(), vec![1.0; 3]);
  assert_eq!(c2.to_vec::<f32>().unwrap(), vec![1.0; 3]);
  assert_eq!(shared.to_vec::<f32>().unwrap(), vec![1.0; 3]);
}

#[test]
fn freeze_on_worker_then_read_lock_free_cross_thread() {
  // `freeze` runs the eval on the worker (its own mlx TLS stream). The
  // frozen result is then read from other threads with no lock and no eval.
  let shared: SharedArray =
    std::thread::spawn(|| Array::ones::<f32>(&(2, 2)).unwrap().freeze().unwrap())
      .join()
      .expect("worker join");

  // Main thread read.
  assert_eq!(shared.to_vec::<f32>().unwrap(), vec![1.0; 4]);

  // Concurrent reads from multiple threads against the SAME shared buffer —
  // no Mutex, so these genuinely run in parallel.
  let handles: Vec<_> = (0..4)
    .map(|_| {
      let s = shared.clone();
      std::thread::spawn(move || s.to_vec::<f32>().unwrap())
    })
    .collect();
  for h in handles {
    assert_eq!(h.join().unwrap(), vec![1.0; 4]);
  }
}

#[test]
fn cross_thread_read_only_metadata() {
  // shape/dtype/ndim/size never touch a stream — readable from any thread.
  let shared = Array::ones::<f32>(&(3, 4)).unwrap().freeze().unwrap();
  let (shape, ndim, size) =
    std::thread::spawn(move || (shared.shape(), shared.ndim(), shared.size()))
      .join()
      .expect("thread join");
  assert_eq!(shape, vec![3, 4]);
  assert_eq!(ndim, 2);
  assert_eq!(size, 12);
}

#[test]
fn into_inner_happy_path() {
  let shared = Array::ones::<f32>(&(3,)).unwrap().freeze().unwrap();
  let mut arr = shared
    .into_inner()
    .expect("sole owner — into_inner must succeed");
  assert_eq!(arr.shape(), vec![3]);
  // Already evaluated by freeze; Array's own &mut reader still works.
  assert_eq!(arr.to_vec::<f32>().unwrap(), vec![1.0; 3]);
}

#[test]
fn into_inner_returns_none_when_aliased() {
  let shared = Array::ones::<f32>(&(2,)).unwrap().freeze().unwrap();
  let _alias = shared.clone();
  assert!(
    shared.into_inner().is_none(),
    "into_inner must return None while another clone is alive"
  );
}
