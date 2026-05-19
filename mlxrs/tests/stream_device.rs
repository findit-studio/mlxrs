//! Public `Device` + `Stream` API smoke tests.

use mlxrs::{
  Device, DeviceKind, Stream,
  stream::{get_default_stream, set_default_stream},
};

/// Serializes the tests that mutate mlx's process-global default device.
/// `Device::{set_default,current}` are internally locked against data
/// races, but libtest runs `#[test]`s in parallel, so two tests that both
/// set + then assert the global default can still interleave logically
/// (test A sets CPU, test B sets GPU, test A asserts CPU → flake). Every
/// test that sets *and asserts* the global default holds this guard for
/// its critical section. `unwrap_or_else(PoisonError::into_inner)` so one
/// failing test doesn't cascade-fail the rest.
static DEFAULT_DEVICE_TEST_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ───────────────────────── Device ─────────────────────────

#[test]
fn device_cpu_constructs() {
  let dev = Device::cpu().expect("cpu device");
  assert_eq!(dev.kind().expect("kind"), DeviceKind::Cpu);
  assert_eq!(dev.index().expect("index"), 0);
}

#[test]
fn device_gpu_constructs() {
  let dev = Device::gpu().expect("gpu device");
  assert_eq!(dev.kind().expect("kind"), DeviceKind::Gpu);
  assert_eq!(dev.index().expect("index"), 0);
}

#[test]
fn device_with_index_round_trips_kind_and_index() {
  let dev = Device::with_index(DeviceKind::Cpu, 0).expect("cpu(0)");
  assert_eq!(dev.kind().unwrap(), DeviceKind::Cpu);
  assert_eq!(dev.index().unwrap(), 0);
}

#[test]
fn device_kind_count_returns_at_least_one_for_cpu() {
  let n = DeviceKind::Cpu.count().expect("cpu count");
  assert!(n >= 1, "expected at least one CPU device, got {n}");
}

#[test]
fn device_kind_count_returns_at_least_one_for_gpu_on_apple_silicon() {
  // On Apple-silicon CI, Metal exposes one GPU; on a non-Metal builder this
  // would be 0. We only assert non-negativity to keep the test portable.
  let n = DeviceKind::Gpu.count().expect("gpu count");
  let _ = n;
}

#[test]
fn device_current_returns_some_device() {
  let dev = Device::current().expect("current device");
  let _ = dev.kind().expect("current device has a kind");
}

#[test]
fn device_set_default_round_trip() {
  let _guard = DEFAULT_DEVICE_TEST_GUARD
    .lock()
    .unwrap_or_else(std::sync::PoisonError::into_inner);
  // Save the original default so we don't poison other tests in the binary.
  let original = Device::current().expect("current");

  let cpu = Device::cpu().expect("cpu");
  cpu.set_default().expect("set_default cpu");
  let after = Device::current().expect("current after");
  assert_eq!(after.kind().unwrap(), DeviceKind::Cpu);

  // Restore.
  original.set_default().expect("restore");
}

#[test]
fn device_is_available_for_cpu() {
  let dev = Device::cpu().unwrap();
  assert!(dev.is_available().expect("availability query"));
}

#[test]
fn device_equal_and_eq_agree() {
  let a = Device::cpu().unwrap();
  let b = Device::cpu().unwrap();
  assert!(a.equal(&b));
  assert_eq!(a, b);

  let g = Device::gpu().unwrap();
  assert_ne!(a, g);
}

#[test]
fn device_try_clone_produces_equal_handle() {
  let a = Device::cpu().unwrap();
  let b = a.try_clone().expect("test: device clone");
  assert_eq!(a, b);
}

#[test]
fn device_debug_prints_something() {
  let dev = Device::cpu().unwrap();
  let s = format!("{dev:?}");
  assert!(s.starts_with("Device("), "unexpected debug format: {s}");
}

// ───────────────────────── Stream ─────────────────────────

#[test]
fn stream_default_gpu_constructs() {
  let s = Stream::default_gpu().expect("default gpu stream");
  let _ = s.index().expect("stream index");
}

#[test]
fn stream_default_cpu_constructs() {
  let s = Stream::default_cpu().expect("default cpu stream");
  let dev = s.device().expect("device");
  assert_eq!(dev.kind().unwrap(), DeviceKind::Cpu);
}

#[test]
fn stream_new_on_cpu_targets_cpu() {
  let cpu = Device::cpu().unwrap();
  let s = Stream::new_on(&cpu).expect("stream on cpu");
  assert_eq!(s.device().unwrap().kind().unwrap(), DeviceKind::Cpu);
}

#[test]
fn stream_new_on_gpu_targets_gpu() {
  let gpu = Device::gpu().unwrap();
  let s = Stream::new_on(&gpu).expect("stream on gpu");
  assert_eq!(s.device().unwrap().kind().unwrap(), DeviceKind::Gpu);
}

#[test]
fn stream_synchronize_succeeds_on_idle_stream() {
  let s = Stream::default_cpu().expect("cpu stream");
  s.synchronize().expect("sync on idle stream");
}

#[test]
fn stream_try_clone_equals_source() {
  let s = Stream::default_cpu().unwrap();
  let t = s.try_clone().expect("test: stream clone");
  assert_eq!(s, t);
}

#[test]
fn stream_default_cpu_index_is_non_negative() {
  let s = Stream::default_cpu().unwrap();
  let i = s.index().unwrap();
  assert!(i >= 0, "stream index unexpectedly negative: {i}");
}

#[test]
fn stream_debug_prints_something() {
  let s = Stream::default_cpu().unwrap();
  let txt = format!("{s:?}");
  assert!(txt.starts_with("Stream("), "unexpected debug format: {txt}");
}

#[test]
fn get_default_stream_for_cpu_device() {
  let cpu = Device::cpu().unwrap();
  let s = get_default_stream(&cpu).expect("default stream for cpu");
  assert_eq!(s.device().unwrap().kind().unwrap(), DeviceKind::Cpu);
}

#[test]
fn set_default_stream_round_trip() {
  let cpu = Device::cpu().unwrap();
  let original = get_default_stream(&cpu).expect("original cpu default");

  let new_stream = Stream::new_on(&cpu).expect("new cpu stream");
  set_default_stream(&new_stream).expect("set default");

  let after = get_default_stream(&cpu).expect("after");
  assert_eq!(after, new_stream);

  // Restore so subsequent tests on this binary see the original.
  set_default_stream(&original).expect("restore");
}

// ───────────────────────── Send / Sync ─────────────────────────

#[test]
fn device_can_move_to_another_thread() {
  let dev = Device::cpu().unwrap();
  let handle = std::thread::spawn(move || dev.kind().unwrap());
  assert_eq!(handle.join().unwrap(), DeviceKind::Cpu);
}

// NOTE: there is intentionally NO "stream can move to another thread" test.
// `Stream` is `!Send + !Sync` (mlx-c++ keys CommandEncoders by thread; even
// the CPU path is not worth exposing as cross-thread-movable when the GPU
// path silently fails). Compile-fail coverage lives in
// `tests/ui-tests/stream_no_send.rs` + `stream_no_sync.rs`. Codex PR #13.

#[test]
fn device_can_be_shared_across_threads() {
  use std::sync::Arc;
  let dev = Arc::new(Device::cpu().unwrap());
  let d2 = Arc::clone(&dev);
  let handle = std::thread::spawn(move || d2.kind().unwrap());
  assert_eq!(handle.join().unwrap(), DeviceKind::Cpu);
  assert_eq!(dev.kind().unwrap(), DeviceKind::Cpu);
}

#[test]
fn concurrent_set_default_and_current_is_race_free() {
  // mlx-c++'s default device is a non-atomic function-static. Without the
  // crate-level lock, hammering set_default + current from many threads is
  // a C++ data race. With it, this completes deterministically and the
  // final default is always one of the two valid values. Codex PR #13.
  use std::thread;
  // Held for the whole test: this one mutates the process-global default
  // from 8 threads, so it must not interleave with the other tests that
  // set + assert it (see DEFAULT_DEVICE_TEST_GUARD).
  let _guard = DEFAULT_DEVICE_TEST_GUARD
    .lock()
    .unwrap_or_else(std::sync::PoisonError::into_inner);
  let cpu = Device::cpu().unwrap();
  let gpu_available = Device::gpu().is_ok();
  // Save/restore the original default so this test doesn't leave the
  // process default flipped to CPU for anything that runs afterwards.
  let original = Device::current().unwrap();

  let handles: Vec<_> = (0..8)
    .map(|i| {
      let cpu = cpu.try_clone().unwrap();
      thread::spawn(move || {
        for _ in 0..50 {
          // Alternate writers; readers interleave. Only set CPU if we
          // can't guarantee GPU is available (set_default(gpu) errors
          // without a gpu backend).
          if i % 2 == 0 {
            cpu.set_default().unwrap();
          }
          let _ = Device::current().unwrap();
        }
      })
    })
    .collect();
  for h in handles {
    h.join().unwrap();
  }
  // Whatever raced, the final default is still a coherent device.
  let kind = Device::current().unwrap().kind().unwrap();
  assert!(
    kind == DeviceKind::Cpu || (gpu_available && kind == DeviceKind::Gpu),
    "default device left in an incoherent state: {kind:?}",
  );
  // Restore so other tests in the binary see the original default.
  original.set_default().unwrap();
}

// ───────────────── clear_current_thread_streams (C++ shim) ─────────────────

#[test]
fn post_clear_array_display_also_panics_fast() {
  // `Display::fmt` → mlx_array_tostring → upstream streams via eval(), so it
  // re-enters eval and must honor the poison guard too (Codex PR #13 r7).
  let outcome = std::thread::spawn(|| {
    let a = mlxrs::Array::ones::<f32>(&(2usize, 2)).unwrap(); // lazy
    Stream::clear_current_thread_streams().unwrap(); // poison this thread
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let _ = format!("{a}"); // Display → tostring → eval → must panic
    }))
  })
  .join()
  .expect("spawned thread itself should not abort");

  let payload = outcome.expect_err("post-clear Display must panic");
  let msg = payload
    .downcast_ref::<String>()
    .map(String::as_str)
    .or_else(|| payload.downcast_ref::<&str>().copied())
    .unwrap_or("");
  assert!(
    msg.contains("clear_current_thread_streams"),
    "panic message should name the culprit API; got: {msg:?}"
  );
}

#[test]
fn post_clear_cpu_linalg_panics_fast() {
  // CPU-routed ops go through `linalg_cpu_stream()`; after this thread is
  // poisoned that helper must trip the cleared-thread guard and panic fast
  // instead of continuing into mlx with torn-down stream state
  // (M2 deferred closeout: handler/poison audit).
  let outcome = std::thread::spawn(|| {
    let a = mlxrs::Array::eye::<f32>(2).unwrap(); // build before poisoning
    Stream::clear_current_thread_streams().unwrap(); // poison this thread
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let _ = mlxrs::ops::linalg_full::svd(&a, false); // -> linalg_cpu_stream -> must panic
    }))
  })
  .join()
  .expect("spawned thread itself should not abort");

  let payload = outcome.expect_err("post-clear CPU linalg must panic");
  let msg = payload
    .downcast_ref::<String>()
    .map(String::as_str)
    .or_else(|| payload.downcast_ref::<&str>().copied())
    .unwrap_or("");
  assert!(
    msg.contains("clear_current_thread_streams"),
    "panic message should name the culprit API; got: {msg:?}"
  );
}

#[test]
fn post_clear_random_key_panics_fast() {
  // `random::key()` has no `default_stream()` on its path; after this thread
  // is poisoned it must trip the cleared-thread guard and panic fast instead
  // of entering mlx-c with torn-down stream state
  // (M2 deferred closeout: handler/poison audit).
  let outcome = std::thread::spawn(|| {
    Stream::clear_current_thread_streams().unwrap(); // poison this thread
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let _ = mlxrs::ops::random::key(0); // -> must panic
    }))
  })
  .join()
  .expect("spawned thread itself should not abort");

  let payload = outcome.expect_err("post-clear random::key must panic");
  let msg = payload
    .downcast_ref::<String>()
    .map(String::as_str)
    .or_else(|| payload.downcast_ref::<&str>().copied())
    .unwrap_or("");
  assert!(
    msg.contains("clear_current_thread_streams"),
    "panic message should name the culprit API; got: {msg:?}"
  );
}

#[test]
fn post_clear_random_seed_panics_fast() {
  // Same as `key`: `random::seed()` has no stream on its path and must fail
  // fast on a poisoned thread (M2 deferred closeout: handler/poison audit).
  let outcome = std::thread::spawn(|| {
    Stream::clear_current_thread_streams().unwrap(); // poison this thread
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let _ = mlxrs::ops::random::seed(0); // -> must panic
    }))
  })
  .join()
  .expect("spawned thread itself should not abort");

  let payload = outcome.expect_err("post-clear random::seed must panic");
  let msg = payload
    .downcast_ref::<String>()
    .map(String::as_str)
    .or_else(|| payload.downcast_ref::<&str>().copied())
    .unwrap_or("");
  assert!(
    msg.contains("clear_current_thread_streams"),
    "panic message should name the culprit API; got: {msg:?}"
  );
}

#[test]
fn clear_current_thread_streams_is_end_of_thread_cleanup() {
  // REALISTIC CONTRACT: clear_current_thread_streams() is an
  // end-of-thread-lifecycle primitive. A worker thread does mlx work, then
  // reclaims its Metal command encoders before finishing — instead of
  // leaking them until process exit. It is NOT "clear then keep using mlx
  // on this thread" (mlx does not re-bootstrap the thread's GPU stream
  // after clear_streams).
  let worker = std::thread::spawn(|| {
    let mut a = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
    let mut s = mlxrs::ops::reduction::sum(&a, false).unwrap();
    assert_eq!(s.item::<f32>().unwrap(), 10.0);
    let _ = a.to_vec::<f32>().unwrap(); // force materialization before clear
    // Done with mlx on this thread — reclaim its encoders deterministically.
    Stream::clear_current_thread_streams().unwrap();
    // Thread exits here; clearing must not abort/crash on the way out.
  });
  worker.join().expect("worker thread panicked / aborted");

  // The calling thread's mlx state is independent of the worker's and is
  // unaffected by the worker's clear_streams call.
  let mut b = mlxrs::Array::ones::<f32>(&(4usize, 4)).unwrap();
  b.eval().unwrap();
  assert_eq!(b.to_vec::<f32>().unwrap(), vec![1.0; 16]);
}

#[test]
fn clear_current_thread_streams_returns_ok_on_idle_thread() {
  // Calling it on a thread that has done no mlx work is still a valid no-op
  // (the encoder map is just empty). Run in a spawned thread so we never
  // poison the libtest worker thread. Locks in the rc=0 success path.
  std::thread::spawn(|| {
    Stream::clear_current_thread_streams().unwrap();
  })
  .join()
  .expect("idle clear should not panic");
}

#[test]
fn reusing_a_cleared_thread_panics_fast_with_actionable_message() {
  // After a successful clear, the thread is poisoned: the next mlxrs op
  // must panic IMMEDIATELY (in default_stream's guard) rather than fail
  // cryptically deep in eval. Done in a spawned thread so the panic + the
  // poisoned TLS stay contained.
  let outcome = std::thread::spawn(|| {
    Stream::clear_current_thread_streams().unwrap(); // poisons THIS thread
    // The next op funnels through default_stream() → must panic.
    std::panic::catch_unwind(|| {
      let _ = mlxrs::Array::ones::<f32>(&(2usize, 2));
    })
  })
  .join()
  .expect("spawned thread itself should not abort");

  let payload = outcome.expect_err("op on a cleared/poisoned thread must panic");
  let msg = payload
    .downcast_ref::<String>()
    .map(String::as_str)
    .or_else(|| payload.downcast_ref::<&str>().copied())
    .unwrap_or("");
  assert!(
    msg.contains("clear_current_thread_streams"),
    "panic message should name the culprit API; got: {msg:?}"
  );
}

#[test]
fn post_clear_every_public_stream_entry_point_panics_fast() {
  // The poison guard must cover EVERY safe public Stream API that touches
  // mlx stream FFI, not just eval/synchronize (Codex PR #13 r6). After a
  // successful clear, each of these must panic-fast.
  let outcome = std::thread::spawn(|| {
    // Construct a Device + a Stream BEFORE poisoning so the accessor checks
    // (device/index/equal) have a handle to call against post-clear.
    let dev = Device::gpu().expect("gpu device");
    let s = Stream::default_gpu().expect("default gpu stream");

    Stream::clear_current_thread_streams().unwrap(); // poison THIS thread

    let mut results: Vec<(&str, bool)> = Vec::new();
    macro_rules! check_panics {
      ($label:literal, $body:expr) => {
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
          let _ = $body;
        }))
        .is_err();
        results.push(($label, panicked));
      };
    }
    check_panics!("default_gpu", Stream::default_gpu());
    check_panics!("default_cpu", Stream::default_cpu());
    check_panics!("new_on", Stream::new_on(&dev));
    check_panics!("try_clone", s.try_clone());
    check_panics!("synchronize", s.synchronize());
    check_panics!("device", s.device());
    check_panics!("index", s.index());
    check_panics!("equal", s.equal(&s));
    check_panics!(
      "get_default_stream",
      mlxrs::stream::get_default_stream(&dev)
    );
    check_panics!("set_default_stream", mlxrs::stream::set_default_stream(&s));
    results
  })
  .join()
  .expect("spawned thread itself should not abort");

  for (label, panicked) in outcome {
    assert!(
      panicked,
      "post-clear `{label}` must panic-fast on a poisoned thread, but returned"
    );
  }
}

#[test]
fn post_clear_eval_of_existing_array_also_panics_fast() {
  // `eval`/`to_vec` reach mlx WITHOUT going through default_stream(), so the
  // poison guard must also cover that path (Codex PR #13 r5). Build a lazy
  // array BEFORE clearing, then assert materializing it after the clear
  // panics immediately rather than failing deep in the backend.
  let outcome = std::thread::spawn(|| {
    let mut a = mlxrs::Array::ones::<f32>(&(2usize, 2)).unwrap(); // lazy, not yet eval'd
    Stream::clear_current_thread_streams().unwrap(); // poisons this thread
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let _ = a.to_vec::<f32>(); // funnels through eval() → must panic
    }))
  })
  .join()
  .expect("spawned thread itself should not abort");

  let payload = outcome.expect_err("post-clear eval/to_vec must panic");
  let msg = payload
    .downcast_ref::<String>()
    .map(String::as_str)
    .or_else(|| payload.downcast_ref::<&str>().copied())
    .unwrap_or("");
  assert!(
    msg.contains("clear_current_thread_streams"),
    "panic message should name the culprit API; got: {msg:?}"
  );
}
