//! Opt-in crash diagnostics for the failure modes that bypass `Result`.
//!
//! Synchronous mlx errors are already `Result<T, Error>`. Two failures are
//! not: (1) a Rust panic deep in the safe layer (e.g. the cleared-thread
//! poison guard); and (2) an **async Metal kernel failure** — the C++
//! runtime calls `std::terminate`/`abort()` *after* our rc already returned
//! `0`, so the process dies with `SIGABRT` and no `Result` is ever produced.
//!
//! [`install`] adds best-effort diagnostics for both. It does **not** and
//! cannot *recover* from an async Metal abort: mlx-c exposes no hook and the
//! Metal command-buffer state is undefined after failure. Diagnostics only.
//!
//! **Chaining, not clobbering.** Both hooks are *additive*: the panic hook
//! chains the previously-installed hook, and the `SIGABRT` handler chains
//! the previously-installed signal disposition (the application's own crash
//! reporter, or the default abort). The diagnostic side-effect is always
//! best-effort and never prevents the prior behaviour from running, and the
//! abort is always propagated — the process still terminates.
//!
//! **Opt-in by design.** A library must not unconditionally hijack the
//! global panic hook or process signal handlers — that is the application's
//! decision — so this is not auto-installed.
//!
//! **Concurrency contract (required call site).** [`install`] must be called
//! **once during single-threaded process initialisation**, before any other
//! component installs a competing `SIGABRT` disposition and before threads
//! that might trigger an abort are spawned. This is the same contract every
//! crash-handler library imposes (Breakpad, Crashpad, Sentry, backtrace-rs):
//! POSIX `sigaction` cannot both atomically capture-and-install *and* publish
//! the captured disposition before the handler is observable, so chaining is
//! only well-defined against dispositions installed *before* `install` runs.
//! If another thread races a `SIGABRT`-disposition change concurrently with
//! `install`, that change may be lost — an inherent property of the POSIX
//! signal API, not specific to this code. Used as documented (single-threaded
//! startup) there is no race and chaining is exact.

use std::{
  cell::UnsafeCell,
  io::Write,
  mem::MaybeUninit,
  ptr,
  sync::atomic::{AtomicBool, Ordering},
};

static INSTALLED: AtomicBool = AtomicBool::new(false);

/// The `SIGABRT` disposition that was installed before [`install`] ran.
///
/// SAFETY / publication-before-liveness invariant: written **exactly once**,
/// by the sole thread that wins the `INSTALLED.swap(true, SeqCst)`
/// single-init guard in [`install`]. The write is *published* — a separate,
/// non-installing `sigaction` query reads the current disposition into a
/// local, that local is written here, and a `SeqCst` fence is then executed
/// — **strictly before** our handler is registered with `sigaction` in a
/// later, distinct syscall. Our handler therefore cannot become live until
/// after this write + fence have completed in program order on the sole
/// init-guarded thread; the fence forbids the compiler/CPU from reordering
/// the non-atomic write past the install. After registration the cell is
/// read-only and is only ever read from [`abort_diag_handler`], which cannot
/// run until *after* our handler is registered (kernel registration in the
/// install syscall is the synchronization edge; the OS signal-delivery path
/// provides the cross-context ordering for the read). No runtime locking is
/// taken on the async-signal path.
struct PrevAction(UnsafeCell<MaybeUninit<libc::sigaction>>);
// SAFETY: access is serialized by the publication-before-liveness invariant
// documented above (one published write under the init guard, fenced strictly
// before the handler is registered; reads only from the signal handler, which
// cannot run before that registration). No concurrent mutation is possible.
unsafe impl Sync for PrevAction {}

static PREV_SIGABRT: PrevAction = PrevAction(UnsafeCell::new(MaybeUninit::uninit()));

/// Install best-effort crash diagnostics. Idempotent; opt-in.
///
/// - **Panic hook:** chains the existing hook. As a *best-effort,
///   non-panicking* prelude it writes this thread's most recent mlx backend
///   error ([`crate::error`]'s `LAST`) to stderr, so a panic that followed a
///   backend failure carries that context (this includes the cleared-thread
///   poison-guard panic). Building or writing that message can never prevent
///   the previous hook from running: `prev(info)` is invoked unconditionally
///   as the last action.
/// - **`SIGABRT` handler:** async Metal failures abort via `SIGABRT`. The
///   handler does *only* async-signal-safe work — one `write(2)` of a fixed
///   message to stderr — then **restores the previously-installed
///   disposition** (an application's own crash reporter, or the default
///   abort) and re-raises, so the prior behaviour still runs and the process
///   still aborts (we never swallow it). It deliberately does **not** read
///   `LAST` or capture a backtrace inside the handler: neither is
///   async-signal-safe. Richer detail comes from the panic hook above and
///   your own logging.
///
/// The previous `SIGABRT` disposition is captured by a *separate,
/// non-installing* `sigaction` query and published (write + `SeqCst` fence)
/// **strictly before** our handler is registered, so the handler provably
/// only ever observes a fully-initialised value (see `PrevAction`). If that
/// query fails, the `SIGABRT` handler is *not* installed (best-effort); the
/// panic hook is unaffected and remains active.
///
/// # Call-site contract
///
/// Call **once, during single-threaded process initialisation**, before any
/// other `SIGABRT` disposition is installed and before abort-capable threads
/// are spawned. The query-then-install protocol publishes the captured
/// disposition before the handler is observable (closing the uninitialised-read
/// window); the unavoidable consequence is a query→install gap. A *different*
/// thread installing a crash reporter inside that gap is not chained — an
/// inherent POSIX `sigaction` limitation shared by all crash-handler libraries
/// (see module docs). Under the documented single-threaded-startup call site
/// the gap cannot be raced and chaining is exact. Idempotent: a second call is
/// a no-op (the `INSTALLED` guard), so it never re-races.
pub fn install() {
  if INSTALLED.swap(true, Ordering::SeqCst) {
    return;
  }

  let prev = std::panic::take_hook();
  std::panic::set_hook(Box::new(move |info| {
    // Best-effort, non-panicking prelude. `writeln!` to stderr returns an
    // `io::Result`; ignore it — a failed diagnostic write must never panic
    // inside the panic hook (that would abort before `prev` runs).
    if let Some(msg) = crate::error::last_error_message() {
      let _ = writeln!(
        std::io::stderr(),
        "mlxrs: most recent mlx backend error before this panic: {msg}"
      );
    }
    // Always chain, unconditionally, as the final action — even if the
    // prelude above produced/wrote nothing.
    prev(info);
  }));

  // Step 1: query the CURRENT `SIGABRT` disposition WITHOUT installing
  // anything (`act == NULL`). This is the crux of the fix: we never use the
  // install syscall's `oldact` out-param to capture the previous action,
  // because that would atomically make our handler live *and* fill the
  // non-atomic `PREV_SIGABRT` in one syscall — a SIGABRT delivered to
  // another thread during that syscall could run `abort_diag_handler` and
  // read `PREV_SIGABRT` while it is still uninitialized/torn.
  //
  // SAFETY: pure query — `act` is NULL so no disposition is changed; `prev`
  // is a valid, zeroed, writable `libc::sigaction` out-param. Reentrant /
  // async-signal-safe regardless (it installs nothing).
  // SAFETY: all-zero is a valid initial `libc::sigaction` (a plain C struct
  // of integers/pointers + `sa_mask`); it is fully written by the query below
  // before any field is read.
  let mut prev: libc::sigaction = unsafe { std::mem::zeroed() };
  // SAFETY: pure query — the new-action ptr is NULL so no disposition is
  // changed; `prev` is the valid, zeroed, writable out-param above.
  // `sigaction` is async-signal-safe; failure is surfaced via `rc_q`.
  let rc_q = unsafe { libc::sigaction(libc::SIGABRT, ptr::null(), &mut prev) };
  if rc_q != 0 {
    // Query failed: do NOT install our SIGABRT handler (we have no trusted
    // previous disposition to chain to). Best-effort: the panic hook above
    // is already installed and remains active; this is the documented
    // early-return path. `PREV_SIGABRT` stays untouched/unused.
    return;
  }

  // Step 2: publish the captured previous disposition into `PREV_SIGABRT`,
  // then fence, BEFORE the handler can ever become live (step 3). The fence
  // forbids the non-atomic write from being reordered after the install
  // syscall; the `compiler_fence` additionally pins compiler ordering. On
  // this sole, init-guarded thread the write thus happens-before the
  // install in program order, and no other thread can invoke the handler
  // until the kernel registers it in step 3 (strictly after this published
  // write + fence). This is the ordering the SAFETY docs on `PrevAction`
  // now state.
  //
  // SAFETY: sole writer (won the `INSTALLED` single-init guard); the cell is
  // not yet observable by any reader because the handler is not yet live.
  unsafe {
    (*PREV_SIGABRT.0.get()).write(prev);
  }
  std::sync::atomic::compiler_fence(Ordering::SeqCst);
  std::sync::atomic::fence(Ordering::SeqCst);

  // A C signal handler must be passed as `sa_sigaction` (a pointer-sized
  // integer alias). Cast through a pointer (`as *const ()`) rather than a
  // direct fn-item-to-int cast: that is the idiom rustc's
  // `function_casts_as_integer` lint and clippy's `fn_to_numeric_cast*`
  // both point to, so no lint allow is needed.
  // SAFETY: all-zero is a valid initial `libc::sigaction`; every field used
  // (`sa_sigaction`, `sa_mask`, `sa_flags`) is explicitly set below before it
  // is passed to `sigaction`.
  let mut act: libc::sigaction = unsafe { std::mem::zeroed() };
  act.sa_sigaction = abort_diag_handler as *const () as libc::sighandler_t;
  // SAFETY: `sigemptyset` initializes the `sa_mask` of the local, exclusively
  // owned, writable `act`; it touches nothing else and cannot fail here.
  unsafe {
    libc::sigemptyset(&mut act.sa_mask);
  }
  act.sa_flags = libc::SA_RESTART;

  // Step 3: install our handler with `oldact == NULL` (we already captured
  // and published the previous disposition in steps 1-2). Kernel
  // registration here is the point at which the handler becomes live —
  // strictly after the published write + fence above.
  //
  // SAFETY: `abort_diag_handler` is a valid `extern "C" fn(c_int)` whose
  // body is async-signal-safe (single `write`, then `sigaction`+`raise` to
  // restore the previous disposition and re-abort). Registered for
  // `SIGABRT` only. `PREV_SIGABRT` is already fully published (steps 1-2,
  // see `PrevAction`); the handler can only run after this registration.
  let rc_i = unsafe { libc::sigaction(libc::SIGABRT, &act, ptr::null_mut()) };
  if rc_i != 0 {
    // Install failed: our handler simply isn't active. Best-effort and
    // harmless — the prior disposition is unchanged and the (now unused)
    // published `PREV_SIGABRT` is never read.
  }
}

extern "C" fn abort_diag_handler(sig: libc::c_int) {
  const MSG: &[u8] = b"\nmlxrs: process aborted (SIGABRT) \xe2\x80\x94 likely an \
async Metal kernel failure (not recoverable). If a panic preceded this, the \
last mlx error was printed above.\n";
  // SAFETY: async-signal-safe only. `write`, `sigaction`, and `raise` are all
  // on the POSIX.1-2008 (and 2017 §2.4.3) async-signal-safe function list; on
  // Darwin (this crate's only target) `raise(sig)` is `pthread_kill(
  // pthread_self(), sig)`, which is likewise async-signal-safe — so the
  // re-raise neither deadlocks nor re-enters the runtime. `write` to stderr
  // (fd 2), then restore the *previously-installed* `SIGABRT` disposition
  // (captured under the single-init guard before this handler went live —
  // read-only here) and re-raise so the prior crash reporter / default abort
  // still runs and the process still aborts. No allocation / TLS / backtrace
  // / locks (none are async-signal-safe). `PREV_SIGABRT` read is sound: it
  // was published
  // (written + `SeqCst` fence) strictly before our handler was registered
  // with the kernel, hence strictly before this handler could ever run.
  unsafe {
    libc::write(2, MSG.as_ptr() as *const libc::c_void, MSG.len());
    libc::sigaction(sig, (*PREV_SIGABRT.0.get()).as_ptr(), ptr::null_mut());
    libc::raise(sig);
  }
}
