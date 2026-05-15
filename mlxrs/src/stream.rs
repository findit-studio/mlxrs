//! INTERNAL ONLY for M1: per-thread default GPU stream.
//!
//! Public `Stream` type lands in M2 (per spec §6.10 + §12 M2a).
//!
//! Per-thread (not process-wide) because mlx-c++ stores the default stream and
//! its `CommandEncoder` in `thread_local` storage on the C++ side
//! (see `mlx/stream.cpp::default_stream_storage` and
//! `mlx/backend/metal/device.cpp::get_command_encoders`). A handle obtained on
//! one thread cannot be used to eval on another — eval throws
//! "There is no Stream(gpu, N) in current thread." Caching the handle in a
//! process-wide `OnceLock` (Phase 3 design) only worked because Phase 3 had
//! no eval-driven tests; Phase 3.5's `to_vec/item` calls expose the issue.

use std::cell::Cell;

thread_local! {
  static DEFAULT_STREAM: Cell<Option<mlxrs_sys::mlx_stream>> = const { Cell::new(None) };
}

pub(crate) fn default_stream() -> mlxrs_sys::mlx_stream {
  // Most safe-layer FFI consumers funnel through here; install the error
  // handler before any mlx-c call so a stripped/disabled #[ctor] cannot let
  // the default printf+exit handler fire on the very first failure.
  crate::error::ensure_handler_installed();
  DEFAULT_STREAM.with(|cell| {
    if let Some(s) = cell.get() {
      return s;
    }
    // SAFETY: handler installed above; errors surface via TLS.
    let s = unsafe { mlxrs_sys::mlx_default_gpu_stream_new() };
    if s.ctx.is_null() {
      panic!(
        "mlxrs: mlx_default_gpu_stream_new returned NULL ctx — \
         GPU unavailable or initialization failed. Aborting."
      );
    }
    cell.set(Some(s));
    s
  })
}

// INTENTIONAL: never freed at thread/process exit. Metal frameworks tear down
// before destructors run, so calling mlx_stream_free at exit would crash.
// Instruments will flag this as a leak on shutdown — that's expected.
//
// USAGE GUIDANCE: each thread that ever calls into mlxrs allocates its own
// GPU stream that lives until process exit. mlxrs is intended to be driven
// from a small, long-lived set of worker threads (a fixed-size thread pool
// or the main thread). Patterns that spawn a fresh OS thread per request or
// per task — rayon-with-thread-recycling, std::thread per HTTP request,
// short-lived spawn loops — will accumulate one mlx_stream per worker over
// the process lifetime and grow without bound. M2's public `Stream` API
// will provide explicit lifetime control for those cases.
//
// On first-init failure, the panic propagates; the next call on that thread
// would re-panic in a hot loop. M1 contract: GPU init failure is a permanent
// abort. M2 may add a cached-failure path via AtomicBool.
