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

<<<<<<< HEAD
#[repr(transparent)]
pub(crate) struct RawStream(pub(crate) mlxrs_sys::mlx_stream);

// SAFETY: mlx_stream is an opaque handle to a refcounted C++ object.
// The Phase-3-entry refcount audit (docs/audits/send-soundness.md) confirmed
// that mlx::core::Stream is a trivial POD `{int, Device}`. Multiple threads
// reading the same ctx pointer is sound (no mutable state).
// Sync needed because OnceLock<T> requires T: Sync.
unsafe impl Send for RawStream {}
unsafe impl Sync for RawStream {}

static DEFAULT_STREAM: OnceLock<RawStream> = OnceLock::new();

pub(crate) fn default_stream() -> mlxrs_sys::mlx_stream {
  // Most safe-layer FFI consumers funnel through here; install the error
  // handler before any mlx-c call so a stripped/disabled #[ctor] cannot let
  // the default printf+exit handler fire on the very first failure.
  crate::error::ensure_handler_installed();
  DEFAULT_STREAM
    .get_or_init(|| {
      // SAFETY: handler installed above; errors surface via TLS.
      let s = unsafe { mlxrs_sys::mlx_default_gpu_stream_new() };
      if s.ctx.is_null() {
        panic!(
          "mlxrs: mlx_default_gpu_stream_new returned NULL ctx — \
           GPU unavailable or initialization failed. Aborting."
        );
      }
      RawStream(s)
    })
    .0
=======
thread_local! {
  static DEFAULT_STREAM: Cell<Option<mlxrs_sys::mlx_stream>> = const { Cell::new(None) };
>>>>>>> 964f3ec (feat(safe): 5 remaining archetype templates (sum/slice/concatenate/addmm/argmax))
}

pub(crate) fn default_stream() -> mlxrs_sys::mlx_stream {
  DEFAULT_STREAM.with(|cell| {
    if let Some(s) = cell.get() {
      return s;
    }
    // SAFETY: handler installed by ctor (see error.rs); errors surface via TLS.
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
// On first-init failure, the panic propagates; the next call on that thread
// would re-panic in a hot loop. M1 contract: GPU init failure is a permanent
// abort. M2 may add a cached-failure path via AtomicBool.
