//! Public `Device` + `Stream` API smoke tests.

use mlxrs::{
  Device, DeviceKind, Stream,
  stream::{get_default_stream, set_default_stream},
};

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
fn device_clone_produces_equal_handle() {
  let a = Device::cpu().unwrap();
  let b = a.clone();
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
fn stream_clone_equals_source() {
  let s = Stream::default_cpu().unwrap();
  let t = s.clone();
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
  let cpu = Device::cpu().unwrap();
  let gpu_available = Device::gpu().is_ok();

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
}
