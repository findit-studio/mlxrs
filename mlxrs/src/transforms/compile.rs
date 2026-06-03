//! Graph compilation: [`compile`] + the global [`set_compile_mode`] /
//! [`enable_compile`] / [`disable_compile`] controls.
//!
//! Mirrors `mlx.core.compile` (Python) and `mlx-swift`'s `compile(...)`
//! ([`Transforms+Compile.swift`](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Transforms%2BCompile.swift)),
//! both thin wrappers over the mlx-c `mlx_compile` entry point.
//!
//! ## What compilation does
//!
//! [`compile`] takes a function over arrays (`Fn(&[Array]) ->
//! Result<Vec<Array>>`) and returns a [`Compiled`] callable that, on first
//! application, *traces* the function once to build its operation graph,
//! then *caches* that graph (simplifying/fusing per the active [`CompileMode`]) in the mlx backend keyed by the
//! function's identity. Subsequent applications with matching input shapes
//! and dtypes reuse the cached graph rather than re-tracing — the win that
//! makes a per-token decode forward stop re-tracing every step (see the
//! crate-level note on the per-leaf compiled functions the mlx-lm / mlx-vlm
//! references apply to RoPE, norms, and the sampler stages).
//!
//! This trace/cache/fusion behavior — and the cache-hit elision of `f`'s
//! Rust-level side effects — applies to a [`Compiled`] built while compilation
//! is **enabled** (the default). If [`compile`] is called while compilation is
//! disabled ([`CompileMode::Disabled`] / [`disable_compile`]), mlx returns `f`
//! unchanged at construction: the [`Compiled`] calls `f` directly on every
//! application (no trace, cache, or fusion; side effects run every call), and a
//! later [`enable_compile`] does not convert it — the choice is fixed when the
//! wrapper is built.
//!
//! `shapeless`: when `true`, the cached graph is *not* re-traced when an
//! input's shape changes (only a change in the number of dimensions or in a
//! dtype forces a re-trace). Not every function can be compiled shapeless —
//! one that branches on a concrete dimension will surface the backend's
//! error from [`Compiled::call`]. Defaults to `false` (mlx parity).
//!
//! ## Ownership over FFI
//!
//! `mlx_compile(res, fun, shapeless)` calls `mlx_closure_set_(*res,
//! compile(get_(fun), shapeless))` (vendored `mlx-c/mlx/c/compile.cpp`):
//! `*res` enters as the `{ctx: NULL}` sentinel from `mlx_closure_new()`, and
//! `mlx_closure_set_` on a NULL ctx *allocates* a fresh `std::function` and
//! writes the pointer into `res->ctx` (vendored `private/closure.h`
//! `mlx_closure_set_`, line 30). [`Compiled`] therefore wraps the populated
//! handle (the local slot *after* the `mlx_compile` call) and frees it once
//! on [`Drop`] — the same "guard the post-set handle, not the NULL sentinel"
//! discipline as [`crate::transforms::autograd`]'s `build_value_and_grad`.
//!
//! The *source* closure passed to `mlx_compile` is dropped right after the
//! call returns (mirroring mlx-swift's `defer { mlx_closure_free(innerClosure)
//! }`). This is sound because the source closure's captured payload — the
//! boxed Rust callable — is held by a `std::shared_ptr<void>` *captured by
//! value* in the `std::function` lambda (vendored `closure.cpp`
//! `mlx_closure_new_func_payload`, line 76: `[fun, cpp_payload]`).
//! `mlx::core::compile` copies that `std::function` into the compiled graph's
//! own capture, incrementing the `shared_ptr` refcount, so the compiled
//! closure independently keeps the Rust callable alive for its whole lifetime
//! (it can re-invoke the trampoline on a shape/dtype-driven re-trace). The
//! boxed callable is reclaimed by `destroy_payload` only when the *last*
//! `shared_ptr` drops — i.e. after both the source closure and the
//! [`Compiled`] handle are freed.
//!
//! ## Concurrency / thread-safety
//!
//! MLX records "am I tracing?" in a single **process-global** stack
//! (`detail::InTracing::trace_stack_`) that is *not* `thread_local`. That one
//! stack is shared across **all** of mlx's transforms — [`compile`] as well as
//! [`grad`](crate::transforms::grad) /
//! [`vjp`](crate::transforms::vjp) / [`jvp`](crate::transforms::jvp) /
//! [`value_and_grad`](crate::transforms::value_and_grad) / `vmap` (each of which
//! constructs an `InTracing` guard) — and it is also *read* by ordinary ops
//! (mlx's `in_tracing()` / `in_dynamic_tracing()` checks).
//!
//! This crate serializes **compile-vs-compile** tracing internally: a
//! [`Compiled::call`] first-trace (or a shape/dtype re-trace) holds a private
//! process-wide lock, so two independent compiled closures cannot push/pop that
//! shared stack at the same time. That lock is *compile-private*, however — it
//! is not taken by `grad`/`vjp`/`jvp`/`value_and_grad`/`vmap` or by ordinary
//! ops. Tracing operations therefore should **not** be run concurrently across
//! threads with *other* tracing transforms or with ops, because the underlying
//! mlx tracing stack is process-global rather than thread-local: a
//! [`Compiled::call`] first-trace on one thread can still race a `grad` (or an
//! op) on another thread.
//!
//! This matches the crate's existing contract for the autograd transforms,
//! which are likewise *not* runtime-serialized against one another (see the
//! [`transforms`](crate::transforms) "Threading" note); tracing-sensitive tests
//! run in isolated processes. Fully closing the residual would require either a
//! global lock around every tracing-touching op (abandoning the thin-forward
//! design) or an upstream change making mlx's `trace_stack_` thread-local. The
//! upstream tracking issue is ml-explore/mlx#3620.

use std::{
  cell::Cell,
  sync::{
    LazyLock, Mutex, MutexGuard,
    atomic::{AtomicBool, Ordering},
  },
};

use crate::{
  Array,
  error::{
    InvariantViolationPayload, Result, check, check_vector_array_handle, ensure_handler_installed,
  },
  stream::assert_streams_not_cleared,
  transforms::closure::{Closure, VectorArrayGuard, drain_vector, vector_array_from_slice},
};

/// Process-wide lock serializing every entry into the mlx backend's *tracing*
/// path from this module.
///
/// MLX records "am I tracing?" in `detail::InTracing::trace_stack_`, a
/// **function-local `static std::vector`** — i.e. *process-global*, NOT
/// `thread_local` (see `mlxrs-sys/vendor/mlx/mlx/transforms.cpp::trace_stack()`
/// and the `InTracing` ctor in `transforms_impl.h`, which `push_back`s onto
/// it). Compilation pushes a frame onto that vector for the duration of a
/// trace. `mlx::core::detail::compile_trace` constructs an `InTracing` guard
/// (`compile.cpp:404`), and that trace runs lazily *inside the compiled
/// closure* on a cache-miss or shape/dtype re-trace (`compile.cpp:1126-1133`),
/// which is exactly what [`Compiled::call`] drives through
/// `mlx_closure_apply`.
///
/// Because safe Rust can build independent [`Compiled`] values and [`Array`]s
/// on several threads (`!Send` only blocks *moving one* closure across
/// threads, never two *independent* closures tracing at once), two first-calls
/// or re-traces could otherwise `push_back`/`pop_back` that one C++ vector
/// concurrently — a data race / UB. A safe function must never permit UB, so we
/// serialize the trace path here. `Mutex::new` is const (MSRV ≫ 1.63), so a
/// plain `static` suffices — the same idiom as `device.rs`'s default-device
/// lock.
///
/// **Scope of this lock — what it does and does NOT cover.** This lock closes
/// the *compile-vs-compile* race only: it is held across every entry into the
/// tracing path *from this module*, so two compiled-closure traces (a first
/// call or a shape/dtype re-trace) can never touch `trace_stack_` at once. It
/// does **not** cover the same process-global stack being touched by mlx's
/// *other* transforms (`grad`/`vjp`/`jvp`/`value_and_grad`/`vmap`, which build
/// their own `InTracing` guards) or *read* by ordinary ops (`in_tracing()` /
/// `in_dynamic_tracing()`) — none of those take this compile-private lock. A
/// compile first-trace on one thread can therefore still race a concurrent
/// `grad` or op on another thread; that residual is a process-global mlx-core
/// limitation (tracked upstream at ml-explore/mlx#3620), consistent with the
/// crate's existing contract that the autograd transforms are likewise not
/// runtime-serialized (see the module-level "Concurrency / thread-safety"
/// note). Do not over-trust this guard as a general cross-transform tracing
/// lock — it is scoped to this module's tracing entries by design.
static TRACE_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
  /// Re-entrancy depth for [`TRACE_LOCK`] on the *current* thread.
  ///
  /// A traced closure body can itself call [`compile`] / [`Compiled::call`]
  /// (nested compile), and that trace runs **synchronously on the same
  /// thread** that already holds [`TRACE_LOCK`]. A plain non-reentrant
  /// `Mutex` would self-deadlock there, so [`TraceGuard`] acquires the global
  /// lock only at the *outermost* entry on a thread (depth `0`) and lets
  /// nested same-thread entries pass straight through. Cross-thread callers
  /// are still fully serialized: another thread's outermost entry blocks on
  /// [`TRACE_LOCK`] until this thread's outermost guard drops.
  static TRACE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// RAII guard that serializes the mlx tracing path process-wide while tolerating
/// same-thread re-entrancy (nested compile).
///
/// At the outermost entry on a thread it holds [`TRACE_LOCK`]; nested entries
/// hold nothing but keep the depth raised. The lock is released on [`Drop`] —
/// including on unwind — so a closure that returns `Err` or panics (the
/// trampoline catches panics, converting them to a non-zero rc; see
/// `transforms::closure`) can never leave the lock held. Poison is recovered
/// via `into_inner()` so a panic that *did* poison the lock never wedges the
/// process permanently (matching `device.rs` / the compile-mode test guard).
struct TraceGuard {
  /// `Some` only for the outermost guard on this thread (which owns the lock);
  /// `None` for nested guards. Dropped before the depth is lowered.
  _lock: Option<MutexGuard<'static, ()>>,
}

impl TraceGuard {
  /// Enter the serialized tracing region, acquiring [`TRACE_LOCK`] iff this is
  /// the outermost entry on the current thread.
  fn enter() -> Self {
    let depth = TRACE_DEPTH.with(|d| {
      let n = d.get();
      d.set(n + 1);
      n
    });
    let lock = if depth == 0 {
      // Recover from poison: a previously-panicking trace must not block every
      // future compile for the rest of the process. The trampoline already
      // converts closure panics to an rc, so reaching a poisoned lock is the
      // defensive tail, not the common path.
      Some(
        TRACE_LOCK
          .lock()
          .unwrap_or_else(|poison| poison.into_inner()),
      )
    } else {
      None
    };
    Self { _lock: lock }
  }
}

impl Drop for TraceGuard {
  fn drop(&mut self) {
    // Drop the `MutexGuard` (releasing `TRACE_LOCK` for the outermost guard)
    // *before* lowering the depth, so the depth never reads `0` while the lock
    // is still held. `saturating_sub` is belt-and-suspenders against an
    // unbalanced decrement.
    self._lock = None;
    TRACE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
  }
}

/// Compilation mode for the global [`set_compile_mode`] control, mirroring
/// mlx-c's `mlx_compile_mode` enum (`mlx/c/compile.h`).
///
/// Controls which graph transformations the backend applies when compiling.
/// The default is [`CompileMode::Enabled`] (full simplification + fusion).
///
/// The variants act at two different times. [`Disabled`](Self::Disabled) is a
/// **construction-time** skip — it only governs whether [`compile`] builds a
/// compiled wrapper at all. The fusion levels [`NoSimplify`](Self::NoSimplify) /
/// [`NoFuse`](Self::NoFuse) / [`Enabled`](Self::Enabled) are instead sampled by
/// mlx when it fills a compiled closure's cache entry on the first call or a
/// shape/dtype re-trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompileMode {
  /// Disable compilation at **construction time** (equivalent to
  /// [`disable_compile`]): a [`Compiled`] that [`compile`] builds while this is
  /// the active mode just runs `f` directly — no graph, no caching or fusion.
  /// Unlike the fusion levels below, `Disabled` is checked only when the wrapper
  /// is constructed, NOT at first trace — setting it via [`set_compile_mode`] on
  /// an already-built [`Compiled`] does not turn that closure back into a direct
  /// call; mlx still traces, caches, and fuses it on its first call.
  Disabled,
  /// Compile and fuse, but skip the graph-simplification pass.
  NoSimplify,
  /// Compile and simplify, but skip operation fusion.
  NoFuse,
  /// Full compilation: simplify + fuse (the default).
  Enabled,
}

impl CompileMode {
  /// The lowercase mode name, matching the `mlx_compile_mode` enumerator
  /// (`"disabled"`, `"no_simplify"`, `"no_fuse"`, `"enabled"`).
  #[inline]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Disabled => "disabled",
      Self::NoSimplify => "no_simplify",
      Self::NoFuse => "no_fuse",
      Self::Enabled => "enabled",
    }
  }

  /// The raw `mlx_compile_mode` value this maps to.
  #[inline]
  const fn to_raw(self) -> mlxrs_sys::mlx_compile_mode {
    match self {
      Self::Disabled => mlxrs_sys::mlx_compile_mode__MLX_COMPILE_MODE_DISABLED,
      Self::NoSimplify => mlxrs_sys::mlx_compile_mode__MLX_COMPILE_MODE_NO_SIMPLIFY,
      Self::NoFuse => mlxrs_sys::mlx_compile_mode__MLX_COMPILE_MODE_NO_FUSE,
      Self::Enabled => mlxrs_sys::mlx_compile_mode__MLX_COMPILE_MODE_ENABLED,
    }
  }
}

/// Process-global mirror of whether mlx will *skip* compilation — read at
/// construction by [`build_compiled`] to record if a [`Compiled`] is cache-backed
/// (a real compiled graph) or a direct passthrough to `f`.
///
/// mlx decides this once, when [`compile`] runs, via its `skip_compile()`
/// (`mlxrs-sys/vendor/mlx/mlx/compile.cpp:1093-1095`):
/// `compile_mode() == disabled || !compile_available_for_device(default_device())`.
/// The device half is always `true` in mlxrs's build — the CPU backend's
/// `compile_available_for_device` returns `true` unconditionally
/// (`backend/cpu/compiled.cpp:52`); only the no-CPU GPU-only backend can return
/// `false` (`backend/no_cpu/compiled.cpp:12`), a configuration mlxrs does not
/// build. So in mlxrs `skip_compile()` reduces to the mode being `Disabled`.
///
/// mlx-c exposes **no getter** for the mode, so we mirror it. The seed matches
/// how mlx itself seeds `compile_mode()` (compile.cpp:217-226: `Disabled` iff the
/// `MLX_DISABLE_COMPILE` env var is set at first read, else `Enabled`), and the
/// only functions that can subsequently change mlx's mode — [`enable_compile`] /
/// [`disable_compile`] / [`set_compile_mode`] — update this flag in lockstep on
/// success. (A consumer reaching past the safe API to flip the mode via raw
/// `mlxrs-sys` FFI would desync the mirror; that unsafe path is out of contract.)
///
/// `Relaxed`: the only shared datum is this flag's own value (the derived
/// cache-backed bool is stored per-[`Compiled`]), so no cross-flag happens-before
/// is needed; a mode change that must affect a later compile on another thread
/// has to be externally ordered before it regardless — exactly mlx's own
/// process-global-mode contract.
static COMPILE_DISABLED: LazyLock<AtomicBool> =
  LazyLock::new(|| AtomicBool::new(std::env::var_os("MLX_DISABLE_COMPILE").is_some()));

/// Serializes a compile-mode change against a [`compile`] construction so the
/// [`COMPILE_DISABLED`] mirror and MLX's own mode stay an **atomic pair**.
///
/// Without this, [`build_compiled`] samples the mirror and then calls
/// `mlx_compile` (which makes the real passthrough-vs-compile decision *inside*,
/// via its `skip_compile`) as two separate steps; a concurrent
/// [`disable_compile`] / [`enable_compile`] / [`set_compile_mode`] landing
/// between them could change MLX's mode after our sample, so the recorded
/// `cache_backed` would disagree with what `mlx_compile` actually did — either
/// over-poisoning a passthrough or, worse, failing to poison a real cache (the
/// stale-empty-success hazard the flag exists to prevent). The mutators hold
/// this lock across their MLX update + mirror store, and [`build_compiled`]
/// holds it across the mirror load + `mlx_compile`, so MLX's mode cannot change
/// between our sample and the compile that consults it.
///
/// Lock order is total and deadlock-free: [`build_compiled`] takes
/// [`TRACE_LOCK`] (via [`TraceGuard`]) *then* `MODE_LOCK`. [`TRACE_LOCK`] must be
/// the OUTER lock because a traced closure body may nest-call [`compile`] while
/// [`Compiled::call`] already holds the *reentrant* [`TRACE_LOCK`], so a nested
/// construction re-takes `TRACE_LOCK` (reentrantly) before it takes `MODE_LOCK`;
/// making `MODE_LOCK` the outer lock would invert that order against this nested
/// path and risk a deadlock. The mutators take only `MODE_LOCK`; a cache-backed
/// [`Compiled::call`] takes only [`TRACE_LOCK`] (a passthrough call takes neither
/// lock — it never traces). No thread ever holds `MODE_LOCK`
/// while waiting on [`TRACE_LOCK`], so there is no cycle, and `MODE_LOCK` is
/// never re-entered on one thread (a construction takes it exactly once;
/// `mlx_compile` builds the caching lambda without running — hence without
/// re-entering — the closure), so a plain non-reentrant `Mutex` is sound. (A
/// consumer flipping the mode through raw `mlxrs-sys` FFI, bypassing the safe
/// mutators, would still desync — that unsafe path is out of contract, as for
/// [`COMPILE_DISABLED`].)
static MODE_LOCK: Mutex<()> = Mutex::new(());

/// Globally enable graph compilation (mlx-c `mlx_enable_compile`; mlx-swift's
/// `compile(enable: true)`). Compilation is enabled by default; call this to
/// re-enable after [`disable_compile`].
///
/// This is a process-global mlx backend toggle. The **disabled-vs-compiled**
/// decision is made at construction time: [`compile`] calls `mlx_compile` once
/// when it builds a [`Compiled`], so a [`Compiled`] built while compilation is
/// disabled runs `f` uncompiled (no graph caching or fusion) on every call, and
/// a later `enable`/`disable` does **not** flip an existing [`Compiled`] between
/// compiled and uncompiled. The specific simplify/fuse *mode* ([`CompileMode`]),
/// in contrast, is sampled by mlx when it fills the cache entry on the first
/// call or a shape/dtype re-trace — so changing the mode (see
/// [`set_compile_mode`]) before a compiled function's first call can still
/// affect that trace's fusion (the first-trace-sampled modes are
/// [`NoSimplify`](CompileMode::NoSimplify), [`NoFuse`](CompileMode::NoFuse), and
/// [`Enabled`](CompileMode::Enabled) — *not* [`Disabled`](CompileMode::Disabled),
/// which is the construction-time skip above). It is a performance switch,
/// never a correctness one.
pub fn enable_compile() -> Result<()> {
  // Serialize against a [`compile`] construction so this MLX-mode update and the
  // mirror store land atomically w.r.t. a build sampling them (see [`MODE_LOCK`]).
  let _mode = MODE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
  // SAFETY: a pure global backend toggle with no handle arguments; the rc is
  // surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_enable_compile() })?;
  // Keep the cache-backed mirror in lockstep with mlx's mode (see
  // [`COMPILE_DISABLED`]). Update only on success — a failed toggle left mlx's
  // mode unchanged.
  COMPILE_DISABLED.store(false, Ordering::Relaxed);
  Ok(())
}

/// Globally disable graph compilation (mlx-c `mlx_disable_compile`;
/// mlx-swift's `compile(enable: false)`). See [`enable_compile`] for the
/// construction-time semantics — a [`Compiled`] built while disabled runs `f`
/// uncompiled on every call (correct, just un-fused); [`Compiled`]s that already
/// exist are unaffected.
pub fn disable_compile() -> Result<()> {
  // Serialize against a [`compile`] construction (see [`MODE_LOCK`]).
  let _mode = MODE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
  // SAFETY: pure global backend toggle, no handle arguments; rc via `check()`.
  check(unsafe { mlxrs_sys::mlx_disable_compile() })?;
  // Mirror the mode (success only) — see [`COMPILE_DISABLED`].
  COMPILE_DISABLED.store(true, Ordering::Relaxed);
  Ok(())
}

/// Set the global compilation mode (mlx-c `mlx_set_compile_mode`).
///
/// Selects which graph transformations the backend applies — see
/// [`CompileMode`]. The default is [`CompileMode::Enabled`]. The fusion levels
/// ([`NoSimplify`](CompileMode::NoSimplify) / [`NoFuse`](CompileMode::NoFuse) /
/// [`Enabled`](CompileMode::Enabled)) are sampled by mlx when it fills a
/// [`Compiled`]'s cache entry on the first call or a shape/dtype re-trace — not
/// when [`compile`] builds the wrapper — so changing one before a compiled
/// function's first call still affects that trace, while one whose graph is
/// already cached is unaffected. [`CompileMode::Disabled`] is different: like
/// [`disable_compile`] it acts only at construction and does not un-compile an
/// existing [`Compiled`].
pub fn set_compile_mode(mode: CompileMode) -> Result<()> {
  // Serialize against a [`compile`] construction (see [`MODE_LOCK`]).
  let _mode = MODE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
  // SAFETY: pure global backend toggle taking a plain enum value (validated
  // by the exhaustive `CompileMode` → raw mapping); rc via `check()`.
  check(unsafe { mlxrs_sys::mlx_set_compile_mode(mode.to_raw()) })?;
  // Mirror the mode (success only). Only `Disabled` makes mlx skip compilation;
  // the fusion levels (`NoSimplify`/`NoFuse`/`Enabled`) all leave a wrapper
  // cache-backed — see [`COMPILE_DISABLED`].
  COMPILE_DISABLED.store(mode == CompileMode::Disabled, Ordering::Relaxed);
  Ok(())
}

/// A compiled function over arrays — the result of [`compile`].
///
/// Owns one reference to the backend-compiled `mlx_closure`. [`Compiled::call`]
/// applies the cached graph to a fresh input slice; the first call traces +
/// caches, later calls with matching shapes/dtypes reuse the cached graph.
/// (This holds for a `Compiled` built while compilation was enabled; one built
/// while [`CompileMode::Disabled`] is active wraps `f` directly — every call
/// runs `f` with no caching or fusion, and a later [`enable_compile`] does not
/// change that.)
///
/// `Compiled` is intentionally `!Send` + `!Sync`: the captured Rust callable
/// may reference [`Array`] handles (themselves `!Send`), and the backend's
/// evaluator is single-threaded (the same rationale as
/// [`crate::transforms::closure::Closure`]).
pub struct Compiled {
  inner: mlxrs_sys::mlx_closure,
  /// Set once a [`Compiled::call`] on a **cache-backed** wrapper fails, after
  /// which every later call errors instead of re-entering the backend. A
  /// passthrough wrapper (one built while compilation was disabled — see
  /// [`cache_backed`](Self::cache_backed)) never sets this: it owns no mlx cache
  /// to corrupt, so a failed call is just `f`'s own recoverable error.
  ///
  /// This guards against an mlx cache-poisoning hazard. On a cache-miss the
  /// backend marks the cache entry *non-empty before tracing it*
  /// (`mlxrs-sys/vendor/mlx/mlx/compile.cpp:1126-1133`: `entry.empty = false;`
  /// then `compile_trace(...)`). If that first trace fails — our Rust closure
  /// returns `Err` or panics, which the trampoline turns into a non-zero rc and
  /// surfaces as a C++ exception out of `compile_trace` — the entry is left
  /// marked non-empty but unfilled. A later matching `call` (e.g. a nullary
  /// function re-applied with the same inputs) would then `find` that entry,
  /// see `empty == false`, *skip the trace*, and `compile_replace` an empty
  /// tape — returning empty outputs as a spurious success. mlx-c exposes no
  /// way to evict just that one entry for the high-level `mlx_compile` path
  /// (`mlx_detail_compile_erase` needs the internal `fun_id`, which
  /// `mlx_compile` never hands back; `mlx_detail_compile_clear_cache` would
  /// nuke every *unrelated* compiled function process-wide), so we instead mark
  /// this wrapper poisoned on the first failing call and refuse all later ones.
  /// The observable contract: a failed trace never yields a later stale
  /// success. Upstream root cause (filed as ml-explore/mlx#3624): mlx should not
  /// set `entry.empty = false` until `compile_trace` returns successfully — once
  /// that lands and is vendored, this whole poison workaround can be removed.
  ///
  /// `AtomicBool` (not a plain `bool`) because [`Compiled::call`] is `&self` —
  /// poisoning needs interior mutability. `Relaxed` ordering suffices: a
  /// `Compiled` is `!Send` + `!Sync`, so the flag is only ever touched from the
  /// one thread that owns this value; no cross-thread happens-before is needed.
  poisoned: AtomicBool,
  /// Whether this wrapper owns a real compiled-graph cache (`true`) or is a
  /// direct passthrough to `f` (`false`).
  ///
  /// Captured at construction from [`COMPILE_DISABLED`]: mlx decides
  /// compile-vs-passthrough once, when [`compile`] runs (its `skip_compile()`),
  /// and never revisits it — a later [`enable_compile`] does not convert a
  /// passthrough into a compiled wrapper. Only a cache-backed wrapper can leave
  /// a half-filled mlx cache entry on a failed trace, so [`Compiled::call`]
  /// consults this before poisoning: a passthrough's `f` error is ordinary and
  /// recoverable (later valid calls must still run, per mlx's disabled
  /// contract), whereas a cache-backed failure must set
  /// [`poisoned`](Self::poisoned) to avoid a later stale empty success.
  cache_backed: bool,
}

impl Compiled {
  /// Apply the compiled function to `inputs`, returning its outputs.
  ///
  /// When this [`Compiled`] was built while compilation was enabled (the
  /// default), the first application traces `f` to build + cache the graph and
  /// subsequent applications with matching input shapes and dtypes reuse it; one
  /// built while compilation was disabled ([`CompileMode::Disabled`] /
  /// [`disable_compile`]) calls `f` directly every time (no trace, cache, or
  /// reuse). A backend error (e.g. a function that cannot be compiled
  /// `shapeless`) surfaces as the wrapped [`crate::Error`].
  ///
  /// Once a call on a **cache-backed** wrapper (one built while compilation was
  /// enabled) fails, that `Compiled` is **poisoned**: every subsequent call
  /// returns an [`crate::Error::InvariantViolation`] without re-entering the
  /// backend. This is deliberate — a first trace that failed can leave a
  /// half-filled mlx cache entry that would otherwise make a later matching call
  /// silently return empty outputs as success. A poisoned wrapper is not
  /// reusable; rebuild it via [`compile`] to retry. A wrapper built while
  /// compilation was disabled ([`CompileMode::Disabled`] / [`disable_compile`])
  /// is instead a direct passthrough with no cache, so it never poisons — a
  /// failed call there is just `f`'s recoverable error, and later valid calls
  /// still run.
  ///
  /// No implicit eval: the returned arrays are lazy graph nodes (consistent
  /// with the rest of mlxrs). Materialize them via [`Array::eval`] or
  /// [`crate::transforms::eval()`].
  pub fn call(&self, inputs: &[Array]) -> Result<Vec<Array>> {
    // A previous call already failed. Re-entering the backend now risks the mlx
    // cache-poisoning hazard (a half-filled cache entry returning empty outputs
    // as a spurious success — see [`Compiled::poisoned`]), so refuse instead
    // with a typed error. `Relaxed` is sound: this value is `!Send`/`!Sync`, so
    // the flag never crosses a thread boundary.
    if self.poisoned.load(Ordering::Relaxed) {
      return Err(crate::Error::InvariantViolation(
        InvariantViolationPayload::new(
          "transforms::compile: Compiled::call",
          "this compiled function failed to trace and is no longer usable",
        ),
      ));
    }
    assert_streams_not_cleared();
    let in_guard = vector_array_from_slice(inputs)?;
    // SAFETY: `mlx_vector_array_new()` returns a populated empty container
    // (non-null ctx on success; NULL on alloc failure caught immediately
    // below). Null-checked, then RAII-wrapped so the guard only ever holds a
    // valid handle.
    let mut out = unsafe { mlxrs_sys::mlx_vector_array_new() };
    check_vector_array_handle(out)?;
    let _out_guard = VectorArrayGuard(out);
    // Serialize the trace path, but ONLY for a cache-backed wrapper: on a
    // cache-miss / shape-or-dtype re-trace `mlx_closure_apply` runs
    // `compile_trace`, which pushes onto MLX's process-global `trace_stack_`
    // (see [`TRACE_LOCK`]). Without this two independent compiled closures could
    // trace concurrently on separate threads → data race / UB. The guard is
    // reentrancy-tolerant (a traced closure body may itself call `compile`/
    // `call`) and releases the lock on every exit path, including unwind. It
    // closes the *compile-vs-compile* race only — not a concurrent `grad`/op
    // touching the same global stack without this compile-private lock (see
    // [`TRACE_LOCK`]'s scope note and ml-explore/mlx#3620).
    //
    // A passthrough wrapper (built while compilation was disabled) never traces
    // — `mlx_closure_apply` just runs `f` directly — so it must NOT hold
    // `TRACE_LOCK`: that would run arbitrary user code under the process-global
    // lock, and a passthrough body that waits on another thread's `compile` /
    // `call` (which blocks on `TRACE_LOCK`) would deadlock. Gate the guard on
    // `cache_backed` so only the genuinely tracing path serializes; a passthrough
    // call takes no lock. `Option<TraceGuard>` holds the lock for the call's
    // scope when present and is a no-op when `None`.
    //
    // Residual (mlx-inherited; never occurs in inference — decode never feeds a
    // compiled closure tracer inputs): `cache_backed` is a construction-time
    // flag, but mlx runs `f` directly — without a `compile_trace`, before any
    // cache touch — when a *cache-backed* closure is applied to TRACER inputs
    // (vendored `mlx/compile.cpp` ~1116: `return fun(inputs)`). mlxrs cannot
    // detect that path (mlx-c does not bind `mlx::core::array::is_tracer`), so a
    // cache-backed call with tracer inputs (e.g. from inside `grad` / another
    // trace) carries TWO distinct residual hazards, with DIFFERENT scopes + fixes:
    //   * Deadlock — CONCURRENT tracing only: it still takes `TRACE_LOCK` and runs
    //     `f` under it, so if a second thread is entering `compile` / a
    //     cache-backed `call` the two can deadlock. Resolved by ml-explore/mlx#3620
    //     (thread-local `trace_stack_` → `TRACE_LOCK` can be dropped).
    //   * False poison — SINGLE-THREADED too: an ordinary `Err` from `f` on this
    //     path sets `poisoned` though no cache entry was touched, bricking the
    //     wrapper. #3620 does NOT fix this; only an `is_tracer` binding (skip the
    //     poison on the tracer-input path) or ml-explore/mlx#3624 (the upstream
    //     cache fix, after which the whole poison workaround is removed) does.
    // Both are accepted, documented limitations tracked at Findit-AI/mlxrs#363.
    let _trace = self.cache_backed.then(TraceGuard::enter);
    // SAFETY: `self.inner` is the owned compiled closure (alive for the call);
    // `in_guard.0` is a freshly built vector of borrowed handles live for the
    // call (mlx-c copies them in); `out` is a populated empty out-param mlx-c
    // mutates in place via its `mlx_vector_array_set_` semantics; the backend
    // rc is surfaced via `check()`.
    //
    // A non-zero rc here means either a real backend error or — the case that
    // motivates the flag — a failed first trace (our closure returned `Err` or
    // panicked, caught by the trampoline). For a cache-backed wrapper either
    // case may leave the mlx cache entry half-filled (`entry.empty == false`, no
    // tape), so poison it: every later call must error rather than risk a stale
    // empty success (see [`Compiled::poisoned`]). A passthrough wrapper owns no
    // cache entry, so its `f` error is ordinary and recoverable — never poison
    // it, or a single transient error would brick later valid calls.
    if let Err(e) = check(unsafe { mlxrs_sys::mlx_closure_apply(&mut out, self.inner, in_guard.0) })
    {
      if self.cache_backed {
        self.poisoned.store(true, Ordering::Relaxed);
      }
      return Err(e);
    }
    drain_vector(out)
  }
}

impl Drop for Compiled {
  fn drop(&mut self) {
    // SAFETY: frees the compiled closure this `Compiled` owns exactly once.
    // The closure's C++ shared_ptr refcount drops; the captured Rust callable
    // is reclaimed (`destroy_payload`) once the last shared_ptr — across the
    // already-dropped source closure and this handle — releases. Runs during
    // `Drop`, so must not panic / unwind across `extern "C"`; the rc is
    // discarded silently per the crate's `Drop` convention.
    unsafe {
      let _ = mlxrs_sys::mlx_closure_free(self.inner);
    }
  }
}

impl std::fmt::Debug for Compiled {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Compiled").finish_non_exhaustive()
  }
}

/// Compile `f` into a cached [`Compiled`] graph — its simplification and
/// operation fusion governed by the active [`CompileMode`] (mlx-c `mlx_compile`;
/// `mlx.core.compile`; mlx-swift's `compile(...)`).
///
/// `f`'s contract is the same as every other transform's:
/// `Fn(&[Array]) -> Result<Vec<Array>>`, required `+ 'static` so the backend
/// can re-invoke it on a shape/dtype-driven re-trace. The returned
/// [`Compiled`] is callable repeatedly via [`Compiled::call`].
///
/// `shapeless` (mlx parity, default `false` at the call site): when `true`,
/// the cached graph is not re-traced on an input *shape* change — only a
/// change in the number of dimensions or a dtype forces a re-trace. Functions
/// that branch on a concrete dimension cannot be compiled shapeless and will
/// surface the backend's error from [`Compiled::call`].
///
/// For a **pure** array function the returned [`Compiled`] produces results
/// identical to calling `f` directly — compilation is a performance
/// optimization, never a numeric change. Note, however, that `f`'s *Rust-level*
/// side effects (a captured `Cell`/`RefCell`, an RNG, a counter, logging, or a
/// read of external mutable state) execute only while the graph is being traced
/// — the first call plus any shape/dtype-driven re-trace — and are **not** re-run
/// on a cache hit. Compile pure, traceable array functions; an impure `f` will
/// diverge from a direct call.
///
/// All of the above — the cached graph and the cache-hit elision of `f`'s
/// Rust-level side effects — assumes compilation is **enabled** (the default)
/// when `compile` runs. Called while disabled ([`CompileMode::Disabled`] /
/// [`disable_compile`]), the returned [`Compiled`] is a direct passthrough to
/// `f` (no caching, fusion, or side-effect elision), permanently — a later
/// [`enable_compile`] does not convert it.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, transforms::compile};
/// // f(x) = x*x + x
/// let compiled = compile(
///   |xs| {
///     let sq = mlxrs::ops::arithmetic::square(&xs[0])?;
///     Ok(vec![mlxrs::ops::arithmetic::add(&sq, &xs[0])?])
///   },
///   false,
/// )?;
/// let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
/// let mut out = compiled.call(&[x])?;
/// assert_eq!(out[0].to_vec::<f32>()?, vec![2.0, 6.0, 12.0]);
/// # Ok(()) }
/// ```
pub fn compile<F>(f: F, shapeless: bool) -> Result<Compiled>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
  // Build the source closure around `f`. `Closure` owns the captured callable
  // for the FFI's lifetime via its payload box; we drop it at the end of this
  // function (mlx-swift's `defer { mlx_closure_free(innerClosure) }`), which is
  // sound because the compiled closure captures its own `shared_ptr` copy of
  // the payload (see the module-level "Ownership over FFI" note).
  let source = Closure::new(f)?;
  build_compiled(&source, shapeless)
}

/// Compile `f` and adapt it to the ergonomic `Fn(&[Array]) ->
/// Result<Vec<Array>>` shape (matching [`crate::transforms::value_and_grad`]'s
/// returned-closure surface).
///
/// Convenience over [`compile`] + [`Compiled::call`]: the returned closure owns
/// the [`Compiled`] wrapper and forwards each call to it, inheriting
/// [`compile`]'s construction-time mode semantics — including the
/// [`CompileMode::Disabled`] case, where the wrapper is a direct passthrough to
/// `f` rather than a cached compiled graph.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, transforms::compile_fn};
/// let g = compile_fn(|xs| Ok(vec![mlxrs::ops::arithmetic::square(&xs[0])?]), false)?;
/// let x = Array::from_slice(&[2.0f32, 3.0], &[2])?;
/// let mut out = g(&[x])?;
/// assert_eq!(out[0].to_vec::<f32>()?, vec![4.0, 9.0]);
/// # Ok(()) }
/// ```
pub fn compile_fn<F>(f: F, shapeless: bool) -> Result<impl Fn(&[Array]) -> Result<Vec<Array>>>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
  let compiled = compile(f, shapeless)?;
  Ok(move |inputs: &[Array]| compiled.call(inputs))
}

// ─────────────────────────── internal helper ───────────────────────────

/// Build the compiled `mlx_closure` from a source [`Closure`] + `shapeless`.
///
/// `mlx_closure_new()` returns the `{ctx: NULL}` sentinel; `mlx_compile`
/// internally calls `mlx_closure_set_(*res, …)` which (on NULL ctx) ALLOCATES
/// a fresh `std::function` and writes the pointer into `res->ctx`. So — exactly
/// like `build_value_and_grad` — the RAII-owning [`Compiled`] must wrap the
/// LOCAL `res` slot *after* the populating call, not the NULL sentinel from
/// `_new()` (which would leak the populated handle and free nothing).
fn build_compiled(source: &Closure, shapeless: bool) -> Result<Compiled> {
  ensure_handler_installed();
  // SAFETY: returns the documented `{ctx: NULL}` sentinel (infallible success
  // path; the catch arm returns the same `{nullptr}`). NO guard yet — a `Drop`
  // over a NULL ctx is a no-op (so no leak if the next `check(…)` short-
  // circuits), and wrapping the NULL copy now would hide the post-set ctx that
  // `mlx_compile` writes into THIS local slot.
  let mut res = unsafe { mlxrs_sys::mlx_closure_new() };
  // `mlx::core::compile` only *builds* the caching lambda — the trace fires
  // lazily on first apply (see [`Compiled::call`] / [`TRACE_LOCK`]) — so this
  // call does not itself touch `trace_stack_` today. We still take the
  // reentrant guard to serialize every tracing-capable entry from this module
  // (defense-in-depth if mlx-c ever traces eagerly here); reentrancy makes the
  // overhead a single uncontended lock and never deadlocks a nested compile.
  let _trace = TraceGuard::enter();
  // Hold MODE_LOCK (inner to TRACE_LOCK — see [`MODE_LOCK`]) across the mirror
  // sample AND `mlx_compile` so a concurrent mode mutator cannot change MLX's
  // mode between them.
  let _mode = MODE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
  // mlx decides compile-vs-passthrough NOW, inside `mlx_compile`, via its
  // `skip_compile()` — in mlxrs's build exactly "is compilation disabled" (see
  // [`COMPILE_DISABLED`]). With `MODE_LOCK` held the mode cannot change between
  // this sample and `mlx_compile`'s own read, so `cache_backed` matches exactly
  // what mlx did; it gates whether a later failed call may poison this wrapper
  // (a passthrough never owns a cache to corrupt, so it must not poison).
  let cache_backed = !COMPILE_DISABLED.load(Ordering::Relaxed);
  // SAFETY: `res` is a valid (NULL-ctx) closure slot mlx-c populates in place
  // via `mlx_closure_set_`; `source.as_raw()` is a valid borrowed handle live
  // for the call (mlx-c reads it via `mlx_closure_get_`, copying the underlying
  // `std::function` into the compiled graph — it does not retain our borrow);
  // `shapeless` is a plain bool; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_compile(&mut res, source.as_raw(), shapeless) })?;
  // `mlx_compile` leaves `res.ctx` non-null on success. Wrap the now-populated
  // slot so any later early return / the natural drop frees it exactly once.
  Ok(Compiled {
    inner: res,
    poisoned: AtomicBool::new(false),
    cache_backed,
  })
}
