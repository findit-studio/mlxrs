//! Drives the TLS error capture path across multiple threads.

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
