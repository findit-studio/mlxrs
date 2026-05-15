//! INTERNAL ONLY for M1: process-wide default GPU stream singleton.
//!
//! Public `Stream` type lands in M2 (per spec §6.10 + §12 M2a).

use std::sync::OnceLock;

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
}

// INTENTIONAL: never freed at process exit. Metal frameworks tear down before
// static destructors, so calling mlx_stream_free at exit would crash.
// Instruments will flag this as a leak on shutdown — that's expected.
//
// On first-init failure, OnceLock::get_or_init does NOT poison after panic;
// subsequent calls would re-panic in a hot loop. M1 contract: GPU init failure
// is a permanent abort. M2 may add a cached-failure path via AtomicBool.
