# mlxrs Milestone 1 — FFI to mlx-c + Safe Array/Ops Core

**Status:** Draft for review
**Date:** 2026-05-14
**Owner:** FinDIT Studio
**Targets:** `aarch64-apple-darwin` (Apple silicon, Metal backend)
**Deliverable:** `mlxrs-sys = "0.1"` (full mlx-c FFI) + `mlxrs = "0.1"` (safe `Array` + `Dtype` + `Error` + all of `ops.h`)

---

## 1. Goals

1. **`mlxrs-sys`** ships pre-committed `extern "C"` bindings for **every** mlx-c v0.6.0 public header (~30 files, ~4 KLoC of C declarations). Built by `bindgen`, snapshot to `mlxrs-sys/src/generated/bindings.rs`. A `buildtime_bindgen` cargo feature exists for opt-in regeneration; default is off so consumers don't need libclang.
2. **`mlxrs`** (no features required) exposes:
   - `mlxrs::Array` — safe newtype around `sys::mlx_array` with `Drop`, manual `Clone` (refcount bump), conservative `unsafe impl Send + Sync`.
   - `mlxrs::Dtype` — Rust enum mirroring `mlx_dtype` 1:1, plus a sealed `Element` trait for typed constructors.
   - `mlxrs::Error` + `pub type Result<T> = std::result::Result<T, Error>`.
   - **All of `ops.h`** as both methods on `&Array` and free fns under `mlxrs::ops::*`.
   - `std::ops::{Add, Sub, Mul, Div}` impls for `&Array op &Array` (panic on shape mismatch).
   - `mlxrs::version() -> String`.
3. **Build pipeline:** `cargo build` on `aarch64-apple-darwin` with only Xcode CLT + cmake installed produces a working static-linked binary. **No `mlxrs::init()` call required** — error handler is installed lazily via `OnceLock` on first fallible call.
4. `cargo test -p mlxrs` runs and passes a meaningful suite (arithmetic, reductions, shape ops, error paths, Send/Sync compile checks) on a real Apple-silicon Mac.
5. CI green on a `macos-14` (or newer) GitHub-hosted Apple-silicon runner.

## 2. Non-Goals (deferred to later milestones)

- **Safe wrappers** for `fft.h`, `linalg.h` (beyond `matmul`/`addmm`), `random.h`, `transforms.h`, `compile.h`, `closure.h`, `distributed*.h`, `export.h`, `io.h`, `gguf` (private), `metal.h`, `cuda.h`, `fast.h`, `graph_utils.h`, `vector.h`/`map.h`. Raw FFI **is** shipped for these in `mlxrs-sys` (bindgen covers them automatically); only the safe layer is deferred.
- **`lm`, `vlm`, `audio` features** stay empty stubs (mirroring the current `lm`/`vlm` posture; `audio` is added as a new stub in this milestone).
- **Platforms other than `aarch64-apple-darwin`**: no x86_64 macOS, no Linux, no CUDA, no cross-targets. The build script asserts the target and panics otherwise (single cfg gate to relax later).
- **By-value operator overloads** (`Array + Array`, `Array + &Array`). Only `&Array op &Array` ships in M1; by-value impls are non-breaking additions later.
- **Publishing to crates.io.** Cargo metadata will be set up correctly; actual publish is M2+.
- **Benchmarks.** The `criterion` dev-dep already in `mlxrs/Cargo.toml` stays unused for M1.
- **Closures** (`mlx_closure_*`) — these need Rust↔C function-pointer interop and are tightly coupled to `transforms.h` (gradients, etc.). M2 work.

## 3. Workspace & Crate Layout

```
mlxrs/                                workspace root (already exists)
├── Cargo.toml                        workspace manifest + lints (already correct)
├── README.md                         install table updated for `audio` feature
├── ci/                               existing helper scripts; sanitizer/miri scripts unused in M1
├── .github/
│   ├── workflows/
│   │   ├── mlxrs-sys.yml             NEW — fmt/clippy/build/test/docs/bindings-drift for the sys crate (+ xtask)
│   │   ├── mlxrs.yml                 NEW — fmt/clippy/build/test/docs/coverage for the safe crate
│   │   └── loc.yml                   existing, untouched
│   ├── FUNDING.yml                   existing
│   └── dependabot.yml                existing
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-05-14-mlx-ffi-milestone-1-design.md  THIS FILE
├── xtask/                            NEW — small bin crate for maintainer commands
│   ├── Cargo.toml
│   └── src/main.rs                   subcommands: regen-bindings (more later)
├── mlxrs-sys/
│   ├── Cargo.toml                    deps: cmake, libc; build-deps: bindgen (cfg-gated)
│   ├── build.rs                      NEW — cmake invocation, link declarations, optional bindgen
│   ├── vendor/
│   │   └── mlx-c/                    NEW — git submodule pinned to v0.6.0
│   ├── wrapper.h                     NEW — `#include "mlx/c/mlx.h"` (single bindgen entry point)
│   └── src/
│       ├── lib.rs                    cfg-allows + `include!("generated/bindings.rs")` + smoke test
│       └── generated/
│           └── bindings.rs           NEW — pre-committed bindgen output (~4 KLoC)
└── mlxrs/
    ├── Cargo.toml                    `mlxrs-sys.workspace = true`; bumps version 0.0.0 → 0.1.0; no build.rs
    ├── src/
    │   ├── lib.rs                    re-exports + module wiring; cfg-gates `lm`, `vlm`, `audio` stubs
    │   ├── error.rs                  NEW — Error, Result, TLS+OnceLock handler, `check(rc)`
    │   ├── dtype.rs                  NEW — Dtype enum + `Element` sealed trait
    │   ├── version.rs                NEW — `mlxrs::version()`
    │   ├── shape.rs                  NEW — `IntoShape` trait, zero-alloc shape conversions
    │   ├── array/                    NEW — split for parallel work
    │   │   ├── mod.rs                struct Array, Drop, Clone, Send/Sync, basic accessors
    │   │   ├── construction.rs       ones, zeros, full, eye, arange, linspace, from_slice
    │   │   ├── conversion.rs         shape, ndim, size, dtype, as_slice, to_vec, to_string
    │   │   └── ops_impl.rs           method-form: `a.add(&b)`, `a.reshape(...)?` (bridges to ops/*)
    │   ├── ops/                      NEW — free-fn form of ops.h, split for parallel work
    │   │   ├── mod.rs                re-exports
    │   │   ├── arithmetic.rs         add, sub, mul, div, neg, pow, abs, sqrt, exp, log, sin, cos, tanh, …
    │   │   ├── reduction.rs          sum, mean, max, min, var, std, prod, …
    │   │   ├── comparison.rs         eq, ne, gt, ge, lt, le, allclose, isclose, …
    │   │   ├── logical.rs            logical_and, logical_or, logical_not, all, any
    │   │   ├── shape.rs              reshape, transpose, expand_dims, squeeze, broadcast_to, concatenate, stack, split, …
    │   │   ├── indexing.rs           slice, take, take_along_axis, gather, scatter
    │   │   ├── linalg_basic.rs       matmul, addmm  (full linalg.h is M2)
    │   │   └── misc.rs               clip, where_, sort, argsort, top_k, partition, cumsum, cumprod, …
    │   ├── ops_traits.rs             NEW — `std::ops::{Add,Sub,Mul,Div}` for `&Array`
    │   ├── lm/mod.rs                 existing stub (untouched)
    │   ├── vlm/mod.rs                existing stub (untouched)
    │   └── audio/mod.rs              NEW — empty stub mirroring lm/vlm (gated by `audio` feature)
    └── tests/
        ├── smoke.rs                  NEW — version() + array round-trip
        ├── arithmetic.rs             NEW
        ├── reductions.rs             NEW
        ├── shape.rs                  NEW
        ├── indexing.rs               NEW
        ├── error_paths.rs            NEW — drives the TLS error capture across threads
        └── send_sync.rs              NEW — compile-time `assert_send_sync::<Array>()` checks
```

### 3.1 Cleanup: existing `mlxrs/build.rs`

The current `mlxrs/build.rs` only does tarpaulin cfg detection. That logic moves to **`mlxrs-sys/build.rs`** (which has to exist anyway for cmake), and `mlxrs/build.rs` is **deleted**. The safe crate doesn't need a build script.

### 3.2 Cargo features

```toml
# mlxrs/Cargo.toml
[features]
default = []
lm = []
vlm = []
audio = ["lm"]                # mirrors mlx-audio's Python dep on mlx-lm
```

```toml
# mlxrs-sys/Cargo.toml
[features]
default = []
buildtime_bindgen = ["dep:bindgen"]
```

## 4. Build Pipeline (`mlxrs-sys/build.rs`)

Steps in order:

1. **Submodule check.** If `vendor/mlx-c/CMakeLists.txt` is missing, print a clear error (`run: git submodule update --init --recursive`) and exit non-zero. Don't try to fetch on the user's behalf.
2. **Target check.** Assert `CARGO_CFG_TARGET_OS == "macos"` and `CARGO_CFG_TARGET_ARCH == "aarch64"`. Panic with a helpful message otherwise. Single cfg gate to relax in M2.
3. **CMake invocation via the `cmake` crate** (a published crate, not custom):
   ```rust
   let dst = cmake::Config::new("vendor/mlx-c")
       .define("BUILD_SHARED_LIBS", "OFF")
       .define("MLX_C_BUILD_EXAMPLES", "OFF")
       .define("CMAKE_BUILD_TYPE", "Release")
       .build();
   ```
   This builds `libmlxc.a`, `libmlx.a`, and the various MLX object archives into `$OUT_DIR/build/`. mlx-c's own `FetchContent` pulls MLX from GitHub during the cmake configure step (one-time network on first build, cached after).
4. **Link declarations** (printed via `cargo:rustc-link-*` directives):
   - `cargo:rustc-link-search=native={OUT_DIR}/build`
   - `cargo:rustc-link-lib=static=mlxc`
   - `cargo:rustc-link-lib=static=mlx`
   - `cargo:rustc-link-lib=framework=Metal`
   - `cargo:rustc-link-lib=framework=MetalPerformanceShaders`
   - `cargo:rustc-link-lib=framework=Foundation`
   - `cargo:rustc-link-lib=framework=Accelerate`
   - `cargo:rustc-link-lib=c++`  *(mlx-c is C++20 internally)*
5. **Optional bindgen** (`cfg(feature = "buildtime_bindgen")`): if the feature is on, run bindgen against `wrapper.h`, write to `$OUT_DIR/bindings.rs`, and `lib.rs` `include!`s from `OUT_DIR` instead of `src/generated/`. Default feature is **off**.
6. **Tarpaulin cfg detection.** Move the existing tarpaulin cfg-detection logic from `mlxrs/build.rs` here (so coverage runs still work).
7. **Rerun-if-changed**: only `wrapper.h` and `vendor/mlx-c/`. The submodule pin guarantees stable inputs across machines.

## 5. FFI Bindings (`mlxrs-sys`)

### 5.1 Bindgen invocation (run by `xtask regen-bindings` or build.rs when `buildtime_bindgen` is on)

```rust
bindgen::Builder::default()
    .header("wrapper.h")
    .clang_arg("-Ivendor/mlx-c")
    .clang_arg("-std=c11")
    .allowlist_function("mlx_.*")
    .allowlist_type("mlx_.*")
    .allowlist_var("MLX_.*")
    .blocklist_item("_mlx_error")            // private; the macro mlx_error() handles it
    .blocklist_file(".*/private/.*")
    .layout_tests(true)
    .derive_default(false)                   // opaque handles shouldn't have Default
    .derive_debug(true)
    .generate_comments(true)                 // preserve doxygen on generated items
    .rustified_enum("mlx_dtype_")
    .formatter(bindgen::Formatter::Rustfmt)
    .generate()?
    .write_to_file("mlxrs-sys/src/generated/bindings.rs")?;
```

### 5.2 Type translation reference (governs the safe layer)

| mlx-c construct | sys type (bindgen output) | Safe layer translation |
|---|---|---|
| `typedef struct mlx_array_ { void* ctx; } mlx_array;` | `pub struct mlx_array { pub ctx: *mut c_void }` | `#[repr(transparent)] pub struct Array(sys::mlx_array)` with `Drop` calling `mlx_array_free` |
| `typedef enum mlx_dtype_ { MLX_BOOL, … }` | `pub enum mlx_dtype { … }` (rustified) | `pub enum Dtype { Bool, U8, U16, U32, U64, I8, I16, I32, I64, Float16, Float32, Float64, Bfloat16, Complex64 }` + `From`/`Into` |
| `int mlx_op(...)` | unchanged | `fn op(...) -> Result<()>` — non-zero → drain TLS into `Err` |
| `int mlx_op_returning(mlx_array* out, ...)` | unchanged | `fn op(...) -> Result<Array>` — `out` is owned by safe wrapper after `mlx_array_new` |
| `mlx_string` | opaque handle | converted to owned `String` via `mlx_string_data` + `mlx_string_size` + `mlx_string_free` (RAII) |
| `int* shape, size_t dim` parameter pair | `*const c_int, usize` | accepts `impl IntoShape` (zero-alloc; see §6.4) |
| `mlx_vector_array` | opaque handle | exposes `iter()` first; `to_vec()` adapter (alloc only if asked) |
| `mlx_optional_int` etc. | struct with `value` + `has_value` | translated to `Option<i32>` |
| `mlx_closure_*` | opaque + fn ptr | **deferred to M2** — out of scope |

### 5.3 Sys-crate hygiene

- `lib.rs`: `#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]` + `include!(...)` + `#[cfg(test)] mod smoke { use super::*; #[test] fn version() { unsafe { let mut s = mlx_string_new(); mlx_version(&mut s); mlx_string_free(s); } } }`
- `unsafe impl Send for mlx_array {}` is **not** added in -sys; Send/Sync for the safe `Array` is decided in §6.
- Pre-committed `bindings.rs` is checked in. CI verifies `xtask regen-bindings && git diff --exit-code` so the file can never silently drift from the headers.

## 6. Safe Wrapper (`mlxrs`)

### 6.1 Allocation discipline (applies to every safe-layer fn)

- Public APIs accept `&T` / `&[T]` / `impl AsRef<...>`, never owned `T` / `Vec<T>` / `String` for read-only inputs.
- Shape/axis parameters use stack arrays (`[i32; N]`) or borrowed slices, never `Vec` (see §6.4 `IntoShape`).
- Operator overloads bind `&Array` only, never `Array` by value.
- `Clone` for `Array` is manual (not derived) and documented as "cheap refcount bump (~ns), still not free."
- `mlx_vector_array` returns expose `iter()` first; collection (`to_vec()`) is opt-in.
- `mlx_string` returns produce owned `String` for M1 (note: `Cow<str>` is an M2 candidate optimization).
- Workspace lints add: `clippy::redundant_clone`, `clippy::needless_collect`, `clippy::unnecessary_to_owned`, `clippy::redundant_allocation`, all denied.

### 6.2 `Array` core (`array/mod.rs`)

```rust
#[repr(transparent)]
pub struct Array(sys::mlx_array);

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { sys::mlx_array_free(self.0); }   // status ignored: dropping
    }
}

impl Clone for Array {
    /// Cheap refcount bump on the C++-side `array` ctx. Not free (~ns), but
    /// does not copy data. Don't `.clone()` in hot paths; use `&Array` instead.
    fn clone(&self) -> Self {
        let mut new = unsafe { sys::mlx_array_new() };
        let rc = unsafe { sys::mlx_array_set(&mut new, self.0) };
        crate::error::check(rc).expect("mlx_array_set failed");
        Self(new)
    }
}

// Conservative Send/Sync: mlx C++ arrays are immutable refcounted values,
// thread-safe per upstream docs and mlx-swift's Sendable conformance.
unsafe impl Send for Array {}
unsafe impl Sync for Array {}
```

### 6.3 Error model (`error.rs`)

```rust
use std::cell::RefCell;
use std::sync::OnceLock;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr;

#[derive(Debug, thiserror::Error)]
#[error("mlx: {message}")]
pub struct Error { pub message: String }

pub type Result<T> = std::result::Result<T, Error>;

thread_local! { static LAST: RefCell<Option<Error>> = const { RefCell::new(None) }; }
static INIT: OnceLock<()> = OnceLock::new();

extern "C" fn handler(msg: *const c_char, _data: *mut c_void) {
    let s = unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned();
    LAST.with(|c| *c.borrow_mut() = Some(Error { message: s }));
}

#[inline]
fn ensure_init() {
    INIT.get_or_init(|| unsafe {
        sys::mlx_set_error_handler(Some(handler), ptr::null_mut(), None);
    });
}

#[inline]
pub(crate) fn check(rc: c_int) -> Result<()> {
    ensure_init();
    if rc == 0 { Ok(()) }
    else {
        Err(LAST.with(|c| c.borrow_mut().take()).unwrap_or(Error {
            message: format!("mlx returned {rc} with no message"),
        }))
    }
}
```

Every fallible safe-wrapper method becomes a 3-liner: call FFI → `check(rc)?` → return.

### 6.4 `IntoShape` (`shape.rs`) — zero-alloc shape conversions

```rust
pub trait IntoShape {
    fn with_shape<R>(&self, f: impl FnOnce(*const c_int, usize) -> R) -> R;
}

impl IntoShape for &[i32] {
    fn with_shape<R>(&self, f: impl FnOnce(*const c_int, usize) -> R) -> R {
        f(self.as_ptr(), self.len())
    }
}

impl<const N: usize> IntoShape for [i32; N] {
    fn with_shape<R>(&self, f: impl FnOnce(*const c_int, usize) -> R) -> R {
        f(self.as_ptr(), N)
    }
}

impl IntoShape for (usize, usize) {
    fn with_shape<R>(&self, f: impl FnOnce(*const c_int, usize) -> R) -> R {
        let s: [i32; 2] = [self.0 as i32, self.1 as i32];   // stack
        f(s.as_ptr(), 2)
    }
}
// repeat for (usize,), (usize,usize,usize), (usize,usize,usize,usize) — covers 1D-4D scalars
```

The `with_shape` callback pattern guarantees no heap allocation regardless of input form.

Tuple-form impls cast `usize` → `i32` because mlx-c's shape API uses `int*`. Values exceeding `i32::MAX` (~2.1 billion elements per dim) are pathological for any realistic tensor and will silently wrap; if this becomes a real concern in M2 we add `i32::try_from(...)` with `Result` propagation. For M1 the cast is accepted.

### 6.5 Operator overloading (`ops_traits.rs`)

```rust
impl<'a, 'b> std::ops::Add<&'b Array> for &'a Array {
    type Output = Array;
    fn add(self, rhs: &'b Array) -> Array {
        crate::ops::arithmetic::add(self, rhs)
            .expect("Array + Array: shape mismatch or dtype error")
    }
}
// repeat for Sub, Mul, Div with the same pattern
```

By-value impls deferred to M2.

### 6.6 `Dtype` (`dtype.rs`)

Simple Rust enum mirroring `mlx_dtype` 1:1, plus a sealed `Element` trait for `bool, i8, i16, i32, i64, u8, …, f32, f64` so generic constructors like `Array::from_slice::<f32>(&[1.0, 2.0], (1, 2))?` are typed without users naming `Dtype` explicitly.

### 6.7 `mlxrs::version()`

Wraps `mlx_version()`, returns `String`. Used in §7 phase 1 to validate the build pipeline before anything else exists.

## 7. Phasing & Parallelism Plan

| Phase | Mode | Branch(es) | Wall-clock | Outcome |
|---|---|---|---|---|
| **1. Skeleton** | sequential | `m1-skeleton` | ~1d | Hand-written `extern "C" fn mlx_version`, `mlxrs-sys/build.rs` invokes cmake, links libmlxc + Metal/Accelerate. `mlxrs::version() -> String` works. CI workflow files landed. One green smoke test. |
| **2. Bindgen pipeline** | sequential | `m1-bindgen` | ~1d | `wrapper.h`, `xtask regen-bindings`, committed `src/generated/bindings.rs`, `buildtime_bindgen` feature, CI bindings-drift gate. Hand-written `mlx_version` replaced by generated symbol; smoke test still passes. |
| **3. Safe-layer foundation** | sequential | `m1-safe-foundation` | ~1d | `error.rs`, `dtype.rs`, `shape.rs`, `array/mod.rs` (Drop/Clone/Send/Sync), `array/construction.rs`, `array/conversion.rs`, **one** template op (`add`) fully wired as the canonical example for phase 4. send_sync compile test, smoke integration test passing. |
| **4. Ops fan-out** | **parallel** | `m1-ops-arithmetic`, `m1-ops-reduction`, `m1-ops-comparison`, `m1-ops-logical`, `m1-ops-shape`, `m1-ops-indexing`, `m1-ops-linalg`, `m1-ops-misc` (8 branches) | ~2d wall-clock with parallelism (vs ~5d sequential) | All of `ops.h` covered. Each subagent owns one file, never crosses module boundaries — zero merge conflict surface. |
| **5. Polish** | sequential | `m1-polish` | ~1d | Operator overloading (`ops_traits.rs`), `audio = ["lm"]` feature + module stub, README install table updated, final `cargo doc`, full CI green. |

**Total:** ~6 days calendar with parallelism (vs ~9-10 sequential).

### 7.1 Phase 4 mechanics (the parallel part)

1. Main thread creates 8 worktrees via the `superpowers:using-git-worktrees` skill, all branched off `main` after phase 3 merges.
2. Main thread dispatches 8 subagents in parallel via `superpowers:dispatching-parallel-agents`. Each subagent gets:
   - Its branch + worktree path
   - Its target file (`mlxrs/src/ops/<group>.rs`) and tests file (`mlxrs/tests/<group>.rs`)
   - The corresponding mlx-c symbols to wrap (extracted from `bindings.rs` once, distributed)
   - The `add` template (a code excerpt) showing the canonical wrapper shape: borrow inputs, allocation-discipline comments, `check(rc)?`, return owned `Array`
   - Test requirements: ≥1 happy-path test per fn, ≥1 error-path test per group
3. Each subagent runs `cargo test -p mlxrs --test <its-suite>` before declaring done.
4. Main thread merges branches in any order as subagents finish — they touch disjoint files.

### 7.2 Concurrency safety constraint

Every parallel branch only writes to **one** file under `mlxrs/src/ops/<group>.rs` plus that group's test file. No subagent edits `lib.rs`, `array/`, `error.rs`, `shape.rs`, or any other shared file. The `pub mod arithmetic;` etc. lines in `ops/mod.rs` are added once during phase 3.

If a subagent discovers a missing primitive in the foundation (e.g., needs an `Array::reshape_into` helper that doesn't exist), it pauses and reports rather than editing shared code. Main thread either adds the helper or instructs the subagent to inline the workaround.

## 8. Testing & CI

### 8.1 Test layers

1. **Bindgen layout tests** (auto-generated): `bindgen` emits `#[test] fn bindgen_test_layout_mlx_array()` etc. for every struct, asserting Rust's `size_of`/`align_of` matches C's. Free safety net.
2. **Sys-crate smoke** (`mlxrs-sys/src/lib.rs` `#[cfg(test)]`): `mlx_version()` round-trip + one `mlx_array_new` / `mlx_array_free` cycle to prove RAII works.
3. **Safe-wrapper unit tests** (per `ops/<group>.rs` `#[cfg(test)] mod tests`): correctness of each fn (shape, dtype, values vs known answers).
4. **Safe-wrapper integration tests** (`mlxrs/tests/*.rs`): cross-module flows — arithmetic round-trips, reductions, shape ops, error paths drive the TLS error capture across multiple threads, send_sync compile-time checks.
5. **Bindings-drift gate** (CI step): `cargo run -p xtask -- regen-bindings && git diff --exit-code mlxrs-sys/src/generated/`.

### 8.2 CI workflows (per-crate split, replacing `ci.yml`)

| File | Trigger paths | Runner | Jobs |
|---|---|---|---|
| `mlxrs-sys.yml` | `mlxrs-sys/**`, `vendor/mlx-c/**`, `wrapper.h`, `xtask/**`, root `Cargo.{toml,lock}` | `macos-14` | fmt (scope: `mlxrs-sys/` + `xtask/`), clippy `-p mlxrs-sys -p xtask`, build `-p mlxrs-sys`, test `-p mlxrs-sys`, docs `-p mlxrs-sys`, bindings-drift |
| `mlxrs.yml` | `mlxrs/**`, `mlxrs-sys/**`, `vendor/mlx-c/**`, root `Cargo.{toml,lock}` | `macos-14` | fmt (scope: `mlxrs/`), clippy `-p mlxrs --all-features`, build `-p mlxrs` over feature matrix `[default, lm, vlm, audio, all]`, test `-p mlxrs` over the same matrix, docs `-p mlxrs --all-features`, scheduled coverage (`cargo tarpaulin -p mlxrs`) |
| `loc.yml` | (existing) | (existing) | (untouched) |

**Old `ci.yml` is deleted.** The cross/miri/sanitizer/loom/ubuntu/windows jobs are removed because they cannot apply to an FFI to libmlxc on Apple silicon.

Cache keys per workflow:
- `~/.cargo/registry`, `~/.cargo/git`, `target/` keyed on `Cargo.lock` hash
- `cmake` build dir under `${OUT_DIR}` keyed on `vendor/mlx-c` HEAD (otherwise every CI run re-fetches MLX from GitHub via FetchContent)

`concurrency:` group per workflow so a force-push cancels in-flight runs of the same workflow on the same ref.

Both workflows trigger on -sys changes (since the safe wrapper depends on -sys via cargo). Both run in parallel; GitHub's required-status-checks list both.

### 8.3 Out-of-scope for M1 testing

- Benchmarks (criterion is in dev-deps, unused).
- Fuzzing.
- miri (mlx-c is C++ — miri can't run it).
- valgrind / ASAN (Apple silicon support is poor; Instruments leaks check is a manual M2 task).
- Cross-platform tests (no targets beyond `aarch64-apple-darwin` in M1).

## 9. Known Costs & Risks

- **First `cargo build` is slow (~minutes).** mlx-c's CMakeLists `FetchContent`s MLX from GitHub during the cmake configure step. Subsequent builds are cache hits. CI cache mitigates this for repeat runs.
- **CI requires Apple-silicon hosted runners.** `macos-14` is GitHub's default macOS runner and is arm64. If GitHub policies change, falling back to `macos-13` (Intel) breaks the build because we explicitly require `aarch64-apple-darwin`.
- **`Send + Sync` for `Array` is asserted, not proven.** mlx C++ arrays are documented as immutable refcounted values, and mlx-swift marks them `Sendable`. We're trusting that. If a future mlx version regresses on this, we'd need to wrap in `Mutex` or downgrade to `Send`-only. M2 should add concrete multi-threaded tests.
- **`mlx_string` allocation cost.** Every `to_string()` / debug-format call allocates. Acceptable for M1; revisit with `Cow<str>` in M2 if profiling shows it matters.
- **Operator overloads panic on shape mismatch.** Convenience for prototyping, sharp edge for production. Documented; users wanting `Result` use `a.add(&b)?`.

## 10. Out-of-Scope (explicit list to prevent scope creep)

- Anything Linux, Windows, x86_64, CUDA, or wasm.
- Safe wrappers for any header beyond `array.h` + `ops.h` + `error.h` + `version.h` + `string.h` + (subset of) `linalg.h`.
- LM, VLM, or audio model implementation (feature stubs only).
- Closures, transforms, gradients, JIT compile, distributed.
- Operator overloads by value.
- `crates.io` publish.
- Benchmarks, fuzzing, miri, sanitizers.

## 11. Hand-off

After this spec is approved, the **writing-plans** skill produces the per-phase implementation plan with concrete tasks, dependencies, and exit criteria. The plan will explicitly mark phase 4's branches as parallel work, ready to be dispatched via `superpowers:dispatching-parallel-agents` after phase 3 merges to `main`.
