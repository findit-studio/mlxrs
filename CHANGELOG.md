# Changelog

## [0.1.0] — 2026-05-16 (M1: FFI + safe Array/ops core)

### Added

- **`mlxrs-sys` 0.1.0** — pre-committed bindgen output for `mlx-c` (post-v0.6.0 main HEAD; ~30 KLoC). Builds vendored mlx-c via cmake-rs; links libmlxc + libmlx + Metal/Accelerate. `aarch64-apple-darwin` only.
- **`mlxrs` 0.1.0** — safe-layer over the C FFI:
  - `Array` RAII handle, `Dtype` enum (14 variants), `Element` trait with impls for `bool`, `i32`, `u32`, `f32`, `half::f16` (extended types in M2).
  - Lazy evaluation (`Array::eval`); implicit eval in data accessors (`item`, `to_vec`, `as_slice`).
  - `Array` is `!Send + !Sync` — single-thread use only. M2 will provide `SharedArray` for cross-thread.
  - Internal per-thread default-stream singleton; M2 lifts `Stream` to public API.
  - 76 ops wrapped across `arithmetic`, `reduction`, `comparison`, `logical`, `shape`, `indexing`, `linalg_basic`, `misc`. Long-tail ops (var/std/all/any/logsumexp/etc.) deferred to M2.
  - Operator overloads (`&a + &b`, `-&a`, …) gated behind `unstable-ops-overload` feature, OFF by default; panic on shape/dtype error. Library authors must NEVER enable transitively.
  - Feature stubs: `lm` (M3), `vlm` (M4), `audio` (M5; implies `lm`), `embeddings` (M3). Per-model architectures land per-usecase, not bulk-ported from the upstream Python projects.
- **xtask `regen-bindings`** subcommand for re-running bindgen against the vendored mlx-c headers.
- **CI** — per-crate workflows (`mlxrs-sys.yml`, `mlxrs.yml`) including matrix feature builds, clippy/fmt/docs gates, and a bindings-drift gate. Plus `dep-watch.yml` weekly Mon 06:00 UTC cron — `cargo update --recursive` + workspace test, with auto-managed tracking issue on failure.
- **Goal-7 perf-floor smoke test** (`mlxrs/tests/perf_floor.rs`) — `#[ignore]` by default; explicit `cargo test --release --test perf_floor -- --ignored` invocation.

### Architecture decisions

- **No dependency pinning** in any manifest (loose semver only); `Cargo.lock` is gitignored. Reproducibility check is the weekly `dep-watch.yml` cron, not a checked-in lockfile.
- **Pre-committed bindings + drift CI gate** anchor binding stability — `regen-bindings` must be re-run + committed when the mlx-c submodule moves.
- **Async Metal kernel failures intentionally abort the process.** The `rc`/sentinel chain only catches synchronous errors. M2 will add a `set_terminate` shim for recovery.
- **No per-model architecture porting** from `mlx-lm`/`mlx-vlm`/`mlx-audio`/`mlx-embeddings` — M3-M5 ship the support surface (loaders, tokenizers, processors, generation loops); model implementations are added per-usecase.

### Safety audits

- Phase-3 entry refcount audit (`docs/audits/send-soundness.md`, local-only) — verified MLX `array_desc_` is `std::shared_ptr` with atomic refcount, but per-clone Send is unsound because `set_status` mutates non-atomic state through `const`. Final design is `!Send + !Sync`.
- Codex adversarial reviews on every PR (4-9): caught and fixed empty-slice dangling-pointer UB across `slice`/`sum_axes`/`concatenate`/`gather`/`pad`/shape-taking ops; introduced `dim_ptr`/`data_ptr`/per-`Element` `sentinel_ptr` helpers; sealed `IntoShape`; centralized `validate_dims` at every FFI boundary; `to_vec`/`as_slice` short-circuit on zero-element arrays; `mean_axes`/`max_axes`/`min_axes` route empty-axes through MLX (dtype/zero-size contract); `clip_with_scalar`/`full_like` checked-scalar guard.

[0.1.0]: https://github.com/Findit-AI/mlxrs/releases/tag/m1-complete
