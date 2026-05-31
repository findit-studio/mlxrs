//! Stream module: internal per-thread default GPU stream + public `Stream`
//! handle (M2).
//!
//! ## Internal singleton (M1 carry-over)
//!
//! `default_stream()` is a per-thread cache of the default GPU stream used by
//! every `ops::*` free function. It is intentionally process-lifetime-leaked
//! (Metal frameworks tear down before destructors run, so calling
//! `mlx_stream_free` at exit would crash).
//!
//! Per-thread (not process-wide) because mlx-c++ stores the default stream and
//! its `CommandEncoder` in `thread_local` storage on the C++ side
//! (see `mlx/stream.cpp::default_stream_storage` and
//! `mlx/backend/metal/device.cpp::get_command_encoders`). A handle obtained on
//! one thread cannot be used to eval on another — eval throws
//! "There is no Stream(gpu, N) in current thread."
//!
//! ## Public `Stream` (M2)
//!
//! [`Stream`] is a thread-affine handle, NOT a scoped RAII guard. Read the
//! type-level docs before using it; the short version:
//!
//! - `Stream` is `!Send + !Sync`. A GPU stream indexes mlx-c++ per-thread
//!   `CommandEncoder` state, so it cannot be moved/shared across threads
//!   (eval/synchronize on the wrong thread throws). Same class of constraint
//!   as `Array`.
//! - **`Drop` is NOT stream teardown.** It only frees the small C handle box
//!   (`delete (mlx::core::Stream*)ctx`). mlx has no per-stream destroy
//!   anywhere in its C++ API — verified — only the bulk, thread-wide
//!   `clear_streams()`. So per-value RAII is impossible at the source level.
//! - [`Stream::new_on`] permanently grows mlx's process-global stream
//!   registry (+ a GPU command encoder). Dropping it does NOT reclaim that.
//!   Allocate a bounded set at startup, never per request/task.
//! - [`Stream::clear_current_thread_streams`] bridges mlx's bulk
//!   `clear_streams()` via a first-party C++ shim. It is **end-of-thread
//!   cleanup only** — after a successful call the OS thread is "poisoned":
//!   any subsequent mlxrs op on it panics immediately with an actionable
//!   message (rather than failing cryptically deep in eval), because mlx
//!   does not re-bootstrap a thread's GPU stream.

use std::{cell::Cell, ffi::CStr};

use static_assertions::assert_not_impl_any;

use crate::{
  device::Device,
  error::{Result, check, ensure_handler_installed},
};

thread_local! {
  static DEFAULT_STREAM: Cell<Option<mlxrs_sys::mlx_stream>> = const { Cell::new(None) };
  /// Set true by `Stream::clear_current_thread_streams` after a successful
  /// bulk clear. mlx does not re-bootstrap a thread's GPU stream afterward,
  /// so this thread can no longer do mlxrs work. We check it in
  /// `default_stream()` (the funnel for every op) to turn the otherwise
  /// cryptic deep-in-eval "There is no Stream(gpu,0)" failure into an
  /// immediate, actionable panic at the first op. This is a logic/resource
  /// guard, not memory unsafety.
  static STREAMS_CLEARED: Cell<bool> = const { Cell::new(false) };
}

/// Panic IMMEDIATELY if this OS thread has had its streams cleared via
/// [`super::Stream::clear_current_thread_streams`]. Call this at the top of
/// every safe entry point that can touch mlx stream/TLS state — not just the
/// default-stream path. mlx will not re-bootstrap a cleared thread, so the
/// only useful behavior is a loud, self-explaining fast failure here instead
/// of a cryptic late failure deep in the mlx backend.
#[inline]
pub(crate) fn assert_streams_not_cleared() {
  if STREAMS_CLEARED.with(Cell::get) {
    panic!(
      "mlxrs: Stream::clear_current_thread_streams() was called on this \
       thread. That is end-of-thread cleanup — mlx does not re-bootstrap a \
       thread's GPU stream afterward, so mlxrs cannot run ops on this OS \
       thread again. If this fired inside a thread pool, the pool recycled \
       a cleared worker for new mlx work: only call \
       clear_current_thread_streams() as a worker's final action before the \
       thread truly exits. See the Stream docs."
    );
  }
}

pub(crate) fn default_stream() -> mlxrs_sys::mlx_stream {
  // Most safe-layer FFI consumers funnel through here; install the error
  // handler before any mlx-c call so a stripped/disabled #[ctor] cannot let
  // the default printf+exit handler fire on the very first failure.
  crate::error::ensure_handler_installed();
  assert_streams_not_cleared();
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

/// Mark this thread as stream-cleared and drop its cached default handle.
/// Called by [`super::Stream::clear_current_thread_streams`] after the bulk
/// `clear_streams()` shim runs.
///
/// Setting `STREAMS_CLEARED` makes the next `default_stream()` call panic
/// with an actionable message — mlx does NOT re-bootstrap a thread's GPU
/// stream after `clear_streams()`, so silently re-creating would only push
/// the failure deeper into eval. Dropping the cache too is belt-and-braces
/// (no dangling `{gpu,0}` handle is even reachable).
pub(crate) fn mark_streams_cleared() {
  STREAMS_CLEARED.with(|c| c.set(true));
  DEFAULT_STREAM.with(|cell| cell.set(None));
}

/// Whether this thread has had its streams cleared via
/// [`super::Stream::clear_current_thread_streams`]. Crate-internal probe
/// for `Drop` paths (e.g. [`super::memory::WiredLimitGuard`]) that need to
/// SKIP a stream-touching action when the thread is poisoned — calling
/// e.g. `Stream::default_gpu()` / `Stream::synchronize()` from a `Drop`
/// would panic (or double-panic on unwind), so the caller checks this and
/// silently skips that step instead.
#[inline]
pub(crate) fn current_thread_streams_cleared() -> bool {
  STREAMS_CLEARED.with(Cell::get)
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
// short-lived spawn loops — accumulate one mlx_stream per worker over the
// process lifetime and grow without bound.
//
// M2's public `Stream` API does NOT solve this with per-value lifetime
// control — it cannot (mlx has no per-stream teardown). `Stream` is a
// thread-affine, non-RAII handle; `Drop` frees only the C handle box. The
// ONLY reclaim path is `Stream::clear_current_thread_streams()`, called as
// a worker's final mlx action immediately before that OS thread terminates
// (NOT before returning the thread to a pool — see its docs).

// ───────────────────────── Public Stream API (M2) ─────────────────────────

/// MLX execution stream — an owned wrapper around the `mlxrs_sys::mlx_stream`
/// **C handle**. NOT a scoped, resource-reclaiming RAII guard: `Drop` frees
/// only the small mlx-c handle box, NOT the underlying mlx stream or its GPU
/// command encoder (mlx has no per-stream teardown — see the Lifetime
/// contract section below).
///
/// A stream targets a specific device and serializes work submitted to it.
/// Construct via [`Stream::default_gpu`], [`Stream::default_cpu`], or
/// [`Stream::new_on`].
///
/// ## Threading
/// `Stream` is intentionally **`!Send` and `!Sync`**.
///
/// The `mlx::core::Stream` struct is a `{DeviceType, int}` POD, so a
/// layout-only view would conclude Send/Sync is sound. That conclusion
/// is layout-only and is wrong in practice: a `Stream` is an *index into
/// per-thread state*. mlx-c++ stores the default-stream and the per-stream
/// `CommandEncoder` in C++ thread-local storage, so a GPU stream constructed
/// on thread A cannot be used to eval (or `synchronize`) on thread B —
/// mlx-c++ throws `"There is no Stream(gpu, N) in current thread."`. This
/// was confirmed empirically by the `SharedArray` cross-thread experiment.
///
/// This is the same class of bug as the M1 `Array` Send revision: a
/// trivially-copyable handle whose *referent* has thread-affine state.
/// Marking the wrapper `Send` would let safe code move the handle across a
/// thread boundary and hit that failure path. Until a thread-checked or
/// CPU/GPU-split API exists (future milestone), `Stream` stays single-thread
/// like `Array`. (`Device` IS `Send + Sync` — it is a pure `{kind, index}`
/// descriptor with no thread-local referent.)
///
/// # Lifetime contract — NOT per-value RAII
///
/// `Stream` is a `Drop` type, but **`Drop` only frees the small C handle
/// box** (`delete (mlx::core::Stream*)ctx`) — it does NOT reclaim the
/// underlying mlx stream. mlx's stream model:
/// - `mlx::core::new_stream` appends `{index, device}` to a process-global
///   `std::vector<Stream>` (no removal API) and, for GPU, registers a Metal
///   command encoder in *thread-local* storage.
/// - mlx's ONLY teardown primitive is `mlx::core::clear_streams()`, which
///   is **thread-wide and bulk** ("destroy all streams created on the
///   current thread" — it clears that thread's command-encoder map). There
///   is no per-stream free, so this fundamentally cannot map to Rust
///   per-value `Drop`. mlx-c does not expose it either; mlxrs bridges it
///   via a first-party shim — see [`Stream::clear_current_thread_streams`].
///
/// Consequences:
/// - [`Stream::default_gpu`] / [`Stream::default_cpu`] are cheap — they
///   return the pre-existing per-thread default; no registry growth.
/// - [`Stream::new_on`] permanently grows the global registry (+ a GPU
///   command encoder) on every call. `Drop` does NOT give that back.
///   Create a bounded set once at startup, never per request/task.
/// - To bound encoder memory in a worker-pool design, have each worker call
///   [`Stream::clear_current_thread_streams`] as its LAST mlx action before
///   the worker thread finishes (end-of-thread cleanup — mlx does not
///   re-bootstrap a thread's GPU stream afterward, so it is not a mid-life
///   "reset").
///
/// In short: streams are coarse, mostly-process-lifetime resources. Treat
/// `Stream` as a handle, not a scoped RAII guard.
#[repr(transparent)]
pub struct Stream(pub(crate) mlxrs_sys::mlx_stream);

// NO `unsafe impl Send/Sync for Stream`. The raw `mlx_stream` contains a
// `*mut c_void`, so the auto-traits are already absent; the assertion below
// locks that in against an accidental future `unsafe impl`.
assert_not_impl_any!(Stream: Send, Sync);

impl Drop for Stream {
  fn drop(&mut self) {
    // SAFETY: must NOT touch TLS or panic (drop runs during thread teardown).
    // Discard rc silently — same convention as Array::drop.
    //
    // IMPORTANT: this frees ONLY the small C handle box (`delete
    // (mlx::core::Stream*)ctx`). It does NOT reclaim the underlying mlx
    // stream. mlx-c++ has no stream-teardown API: `mlx::core::new_stream`
    // appends to a process-global `std::vector<Stream>` (and, for GPU,
    // allocates a Metal command queue) that lives until process exit. See
    // the `Stream` type docs for the lifetime contract — this is NOT
    // resource-reclaiming RAII.
    unsafe {
      let _ = mlxrs_sys::mlx_stream_free(self.0);
    }
  }
}

impl Stream {
  /// The per-thread default GPU stream. Wraps `mlx_default_gpu_stream_new`.
  /// Cheap and repeatable — returns the thread's existing default, so it
  /// does NOT grow mlx's global stream registry (unlike [`Stream::new_on`]).
  /// See the type-level "Lifetime contract" note: `Drop` frees only the C
  /// handle box.
  ///
  /// On a thread that never spun up Metal, this triggers GPU initialization;
  /// returns `Err(Backend { .. })` if the GPU is unavailable.
  pub fn default_gpu() -> Result<Self> {
    ensure_handler_installed();
    assert_streams_not_cleared();
    // SAFETY: `mlx_default_gpu_stream_new()` returns the thread's default GPU stream
    // handle; the error handler is installed first and the NULL-ctx case is
    // checked by the caller before the handle is used.
    let raw = unsafe { mlxrs_sys::mlx_default_gpu_stream_new() };
    if raw.ctx.is_null() {
      // A NULL ctx is handler-backed: mlx-c catches the C++ exception, records it
      // via `mlx_error`, then returns an empty handle. Drain that real error first
      // (also clearing the thread-local LAST slot so a later boundary failure is
      // not misattributed); fall back to a typed null-handle error only if none
      // was recorded.
      return Err(
        crate::error::take_last().unwrap_or(crate::Error::FfiNullHandle(
          crate::error::FfiNullHandlePayload::new("mlx_default_gpu_stream_new"),
        )),
      );
    }
    Ok(Self(raw))
  }

  /// New default-CPU stream. Wraps `mlx_default_cpu_stream_new`.
  pub fn default_cpu() -> Result<Self> {
    ensure_handler_installed();
    assert_streams_not_cleared();
    // SAFETY: `mlx_default_cpu_stream_new()` returns the thread's default CPU stream
    // handle; the error handler is installed first and the NULL-ctx case is
    // checked by the caller before the handle is cached/used.
    let raw = unsafe { mlxrs_sys::mlx_default_cpu_stream_new() };
    if raw.ctx.is_null() {
      // Handler-backed NULL ctx — drain the real mlx-c error first (see `default_gpu`).
      return Err(
        crate::error::take_last().unwrap_or(crate::Error::FfiNullHandle(
          crate::error::FfiNullHandlePayload::new("mlx_default_cpu_stream_new"),
        )),
      );
    }
    Ok(Self(raw))
  }

  /// New distinct stream targeting `device`, for op pipelining /
  /// concurrency. Wraps `mlx_stream_new_device`.
  ///
  /// **PERMANENT ALLOCATION — read before calling in a loop.** mlx-c++'s
  /// `new_stream` appends to a process-global `std::vector<Stream>` with no
  /// removal path, and for a GPU device it also allocates a Metal command
  /// queue that is never reclaimed. Dropping the returned `Stream` frees
  /// only the tiny C handle box — NOT the registry slot or the command
  /// queue. Every `new_on` call therefore costs process-lifetime memory
  /// (and a GPU queue). Create a *bounded* set of streams once at startup;
  /// never one per request/task. (`default_gpu`/`default_cpu` do not have
  /// this cost — they return the pre-existing per-thread default.)
  pub fn new_on(device: &Device) -> Result<Self> {
    ensure_handler_installed();
    assert_streams_not_cleared();
    // SAFETY: `mlx_stream_new_device(device.0)` takes a valid borrowed `mlx_device`
    // and returns a new stream handle; the error handler is installed first
    // and the NULL-ctx case is checked by the caller.
    let raw = unsafe { mlxrs_sys::mlx_stream_new_device(device.0) };
    if raw.ctx.is_null() {
      // Handler-backed NULL ctx — drain the real mlx-c error first (see `default_gpu`).
      return Err(
        crate::error::take_last().unwrap_or(crate::Error::FfiNullHandle(
          crate::error::FfiNullHandlePayload::new("mlx_stream_new_device"),
        )),
      );
    }
    Ok(Self(raw))
  }

  /// Handle duplication: allocates a fresh `mlx_stream` and copies
  /// `{kind, index}` via `mlx_stream_set` (a new independent handle with a
  /// copied payload — **not** a refcounted shared payload). Returns `Result`
  /// because the alloc/set can fail; `Stream` intentionally does not implement
  /// `Clone`.
  pub fn try_clone(&self) -> Result<Self> {
    ensure_handler_installed();
    assert_streams_not_cleared();
    // `mlx_stream_new` returns an empty handle (NULL ctx) intended to be
    // populated by `mlx_stream_set`/`mlx_get_default_stream` — same
    // out-param convention as `mlx_array_new`. Wrap in `Self` first so RAII
    // covers the fallible set.
    // SAFETY: `mlx_stream_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; wrapped in the RAII newtype FIRST so an early
    // return frees it, then populated by the following set/get call.
    let mut out = Self(unsafe { mlxrs_sys::mlx_stream_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_stream_set(&mut out.0, self.0) })?;
    Ok(out)
  }

  /// Block until all work submitted to this stream is complete. Wraps
  /// `mlx_synchronize`.
  pub fn synchronize(&self) -> Result<()> {
    ensure_handler_installed();
    // Synchronizing a stream whose thread was cleared touches dead encoder
    // state — fail fast with the actionable message instead.
    assert_streams_not_cleared();
    // SAFETY: `self.0` is a valid borrowed stream handle for the duration of
    // the call, mlx does not retain it past the call, and the backend rc is
    // surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_synchronize(self.0) })
  }

  /// Non-panicking, drop-safe variant of [`Self::synchronize`].
  ///
  /// Returns `Ok(())` and silently SKIPS the sync (without panicking) when
  /// this thread's streams have been cleared via
  /// [`Self::clear_current_thread_streams`] — mlx will not re-bootstrap the
  /// stream, so synchronizing would touch dead encoder state and is
  /// inappropriate. Crate-internal because the only legitimate caller is
  /// a `Drop` impl (which must be infallible / non-panicking — calling the
  /// public [`Self::synchronize`] from a `Drop` would `panic!` via the
  /// `assert_streams_not_cleared()` guard, and unwinding through a panic
  /// already in flight would abort the process).
  ///
  /// The returned `Result` carries only the underlying mlx-c rc; the
  /// caller (a `Drop`) discards it per the crate-wide Drop convention.
  ///
  /// Lets [`crate::memory::WiredLimitGuard::drop`]
  /// safely no-op the sync step when its scope ended after
  /// `clear_current_thread_streams`, while still completing the
  /// `set_wired_limit` restore step.
  pub(crate) fn try_synchronize(&self) -> Result<()> {
    if current_thread_streams_cleared() {
      return Ok(());
    }
    ensure_handler_installed();
    // SAFETY: `self.0` is a valid borrowed stream handle for the duration of
    // the call, mlx does not retain it past the call, and the backend rc is
    // surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_synchronize(self.0) })
  }

  /// Non-panicking, drop-safe variant of [`Self::default_gpu`].
  ///
  /// Returns `None` when this thread's streams have been cleared via
  /// [`Self::clear_current_thread_streams`] (the default GPU stream is no
  /// longer reachable on a poisoned thread). Returns `None` on any FFI
  /// failure instead of `Err`, since the only legitimate caller is a
  /// `Drop` impl that has no good way to surface an error.
  ///
  /// Companion to [`Self::try_synchronize`] — lets
  /// [`crate::memory::WiredLimitGuard::drop`] decide whether to sync the
  /// default stream (when no explicit streams were passed) or skip it
  /// entirely on a poisoned thread, without panicking.
  pub(crate) fn try_default_gpu() -> Option<Self> {
    if current_thread_streams_cleared() {
      return None;
    }
    ensure_handler_installed();
    // SAFETY: `mlx_default_gpu_stream_new()` returns the thread's default GPU
    // stream handle; the error handler is installed first; the NULL-ctx case
    // (e.g. no GPU at all) is treated as "skip" per the drop-safe contract.
    let raw = unsafe { mlxrs_sys::mlx_default_gpu_stream_new() };
    if raw.ctx.is_null() {
      return None;
    }
    Some(Self(raw))
  }

  /// Destroy **every** stream created on the *current thread*, reclaiming
  /// their Metal command encoders in bulk. This is mlx's only stream-
  /// teardown primitive (`mlx::core::clear_streams()`); mlx-c does not
  /// expose it, so this calls a first-party C++ shim
  /// ([`mlxrs_sys::mlxrs_shim_clear_streams`]).
  ///
  /// # This is END-OF-THREAD cleanup, not a mid-life "reset"
  ///
  /// mlx does NOT re-bootstrap a thread's GPU stream after `clear_streams()`
  /// — empirically, even a fresh `mlx_default_gpu_stream_new()` afterward
  /// still fails eval with "There is no Stream(gpu, 0) in current thread".
  /// So the contract is strictly: **call this once, as the last mlx action
  /// on a worker thread, right before that thread finishes.** Do NOT
  /// continue doing mlx work on the thread afterward.
  ///
  /// The intended pattern is **worker-thread shutdown**: a thread that is
  /// about to terminate calls this as its final mlx action to release its
  /// GPU encoder memory deterministically instead of leaking it until
  /// process exit (the otherwise-unavoidable cost of dynamic
  /// [`Stream::new_on`] usage). It is explicitly NOT for returning a worker
  /// to a pool — a successful clear poisons the OS thread, so the next job
  /// scheduled on a recycled worker panics immediately. Only call it when
  /// the thread itself is ending. It is
  /// an associated function (not `&self`) because the operation is
  /// thread-wide and bulk — it cannot be scoped to one `Stream`; every
  /// `Stream` previously obtained on this thread (including the per-thread
  /// default) is invalidated.
  ///
  /// # Misuse is loud, not silent
  ///
  /// After a successful call this thread is **poisoned**: a thread-local
  /// flag is set so that the very next mlxrs op (any op funnels through
  /// `default_stream()`) **panics immediately** with an actionable message,
  /// instead of failing cryptically deep inside `eval`. This makes the
  /// "thread pool recycled a cleared worker for new mlx work" bug fail
  /// fast and self-explain, rather than corrupt-looking late errors. It is
  /// a logic/resource guard — not memory unsafety — which is why this stays
  /// a safe `fn` rather than `unsafe`.
  ///
  /// "Subsequent mlxrs op" above means subsequent *work* — `eval`, ops,
  /// `Display`, the public `Stream` methods. **This function is itself
  /// exempt from the poison guard**: calling it again on an
  /// already-cleared thread is a deliberate, harmless idempotent no-op
  /// (mlx's `clear_streams()` just clears an already-empty map), not a
  /// panic. A defensive double-clear in cleanup code must not blow up.
  ///
  /// Returns `Err(Backend)` if the underlying C++ call threw (not expected
  /// in practice — it clears an `unordered_map`). The poison flag is set
  /// only on the success path; a thrown clear leaves the thread usable.
  pub fn clear_current_thread_streams() -> Result<()> {
    // NOTE: intentionally no `assert_streams_not_cleared()` here — see the
    // "exempt from the poison guard" paragraph in the doc above. Idempotent
    // by design.
    ensure_handler_installed();
    // SAFETY: first-party C++ shim with no arguments; the error handler is installed
    // first so a thrown `clear_streams()` surfaces as an rc rather than the
    // default `printf+exit`.
    let rc = unsafe { mlxrs_sys::mlxrs_shim_clear_streams() };
    if rc != 0 {
      // The C++ clear_streams() threw before tearing anything down (it just
      // clears a map; throwing is not expected). Leave the thread usable —
      // do NOT poison it — and surface the error.
      return Err(crate::Error::Backend(
        "mlxrs_shim_clear_streams: mlx::core::clear_streams() threw".into(),
      ));
    }
    // Success: this thread's encoders are gone and mlx will not re-bootstrap
    // them. Poison the thread so the next op panics with a clear message
    // (see `mark_streams_cleared` / `default_stream`) rather than failing
    // deep in eval with "There is no Stream(gpu,0)".
    mark_streams_cleared();
    Ok(())
  }

  /// Returns the [`Device`] this stream targets. Wraps `mlx_stream_get_device`.
  pub fn device(&self) -> Result<Device> {
    ensure_handler_installed();
    assert_streams_not_cleared();
    // SAFETY: `mlx_device_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; wrapped in the RAII newtype FIRST so an early
    // return frees it, then populated by the following set/get call.
    let mut dev = Device(unsafe { mlxrs_sys::mlx_device_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_stream_get_device(&mut dev.0, self.0) })?;
    Ok(dev)
  }

  /// Returns the index of this stream within its device. Wraps
  /// `mlx_stream_get_index`.
  pub fn index(&self) -> Result<i32> {
    ensure_handler_installed();
    assert_streams_not_cleared();
    let mut idx: i32 = 0;
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_stream_get_index(&mut idx, self.0) })?;
    Ok(idx)
  }

  /// Whether two streams refer to the same `{device, index}` pair. Wraps
  /// `mlx_stream_equal`.
  pub fn equal(&self, other: &Stream) -> bool {
    assert_streams_not_cleared();
    // SAFETY: pure comparison of two valid borrowed handles; mlx-c does not mutate
    // or retain either and returns a plain `bool`.
    unsafe { mlxrs_sys::mlx_stream_equal(self.0, other.0) }
  }

  /// Borrow the raw mlx-c handle (does not transfer ownership).
  ///
  /// # Safety
  /// Caller must not call `mlx_stream_free` on the returned handle and must
  /// not retain it past `self`'s lifetime.
  #[inline]
  pub unsafe fn as_raw(&self) -> mlxrs_sys::mlx_stream {
    self.0
  }
}

/// Returns the **calling thread's** default stream for `device`. Wraps
/// `mlx_get_default_stream`.
///
/// mlx stores default streams in `thread_local` storage (see the module
/// docs), so this is per-thread, NOT process-wide — a default set on one
/// thread is invisible to others.
pub fn get_default_stream(device: &Device) -> Result<Stream> {
  ensure_handler_installed();
  assert_streams_not_cleared();
  // SAFETY: `mlx_stream_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; wrapped in the RAII newtype FIRST so an early
  // return frees it, then populated by the following set/get call.
  let mut out = Stream(unsafe { mlxrs_sys::mlx_stream_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_get_default_stream(&mut out.0, device.0) })?;
  Ok(out)
}

/// Install `stream` as the **calling thread's** default for the device it
/// targets. Wraps `mlx_set_default_stream`.
///
/// mlx default streams are `thread_local`, so this has **no cross-thread
/// effect** — it does not change any other thread's default, and it does
/// NOT swap the per-thread default-GPU stream cached by `default_stream()`
/// (internal `ops::*` calls keep using their cached handle). Use this when
/// interoperating with raw mlx-c calls or `get_default_stream` on the same
/// thread.
pub fn set_default_stream(stream: &Stream) -> Result<()> {
  ensure_handler_installed();
  assert_streams_not_cleared();
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_set_default_stream(stream.0) })
}

impl PartialEq for Stream {
  fn eq(&self, other: &Self) -> bool {
    self.equal(other)
  }
}

impl Eq for Stream {}

impl std::fmt::Debug for Stream {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    // Reaches fallible mlx-c (mlx_stream_tostring); the error.rs contract
    // requires the handler be installed before any such call so a
    // stripped/disabled ctor cannot let mlx's default printf+exit abort.
    // Intentionally NOT poison-guarded: mlx_stream_tostring only formats
    // the {device, index} POD (no eval / no encoder access), and panicking
    // inside Debug is hostile — `{stream:?}` is exactly what you reach for
    // while debugging the poisoned-thread state, and a panic here while
    // formatting a panic message would double-panic → abort. Same rationale
    // as Array's Debug. Other Stream entry points (which DO touch encoder
    // state) remain poison-guarded.
    crate::error::ensure_handler_installed();
    // SAFETY: `mlx_string_new()` returns a fresh empty out-param `mlx_string`
    // (NULL ctx) per the mlx-c convention; populated by the following call
    // and freed via the local guard / explicit `mlx_string_free`.
    let mut s = unsafe { mlxrs_sys::mlx_string_new() };
    // SAFETY: `self.0` is a valid borrowed handle; `s` is a fresh `mlx_string`
    // out-param freed via the local guard/explicit free; mlx-c writes the
    // formatted string into it and the rc is surfaced (checked below).
    let rc = unsafe { mlxrs_sys::mlx_stream_tostring(&mut s, self.0) };
    let result = if rc == 0 {
      // SAFETY: `s` is a live `mlx_string` (freed only after this borrow); mlx-c
      // returns its internal NUL-terminated buffer, valid until the string is
      // freed. The returned pointer is NULL-checked before use.
      let p = unsafe { mlxrs_sys::mlx_string_data(s) };
      if p.is_null() {
        write!(f, "Stream(<unprintable>)")
      } else {
        // SAFETY: the pointer was NULL-checked just above and points into the live
        // `mlx_string` (still owned here, freed only after this borrow); the C
        // string is NUL-terminated by mlx-c.
        let cs = unsafe { CStr::from_ptr(p) };
        write!(f, "Stream({})", cs.to_string_lossy())
      }
    } else {
      write!(f, "Stream(<unprintable>)")
    };
    // SAFETY: frees a handle this guard owns exactly once. Runs during `Drop` /
    // thread teardown: must not touch TLS, call `check()`, panic, or unwind
    // across `extern "C"`; the rc is discarded silently per the crate's
    // Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_string_free(s);
    }
    result
  }
}
