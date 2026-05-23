// Architecture and the f64 dot/reduction kernel adapted from the `dia`
// project (github — MIT/Apache-2.0), src/ops/.

//! Hand-written `core::arch` SIMD kernels for the host-CPU numeric
//! loops mlxrs runs *itself* (not through MLX FFI).
//!
//! The overwhelming majority of mlxrs tensor math is delegated to MLX
//! and runs on MLX's own SIMD/Metal kernels — those are out of scope
//! here. This module covers the small set of Rust-side `&[f64]` /
//! `&[f32]` loops that run on the CPU regardless (audio DSP,
//! preprocessing), where a `core::arch` kernel is a genuine win.
//!
//! ## Layered architecture
//!
//! Mirrors the `dia` project's `src/ops/` four-layer shape:
//!
//! - [`scalar`](crate::simd::scalar) — bit-exact scalar reference
//!   kernels. **Always compiled**, independent of `target_arch`. The
//!   math contract is anchored here; it is also the differential-test
//!   oracle and the fallback path.
//! - `arch` — architecture-specific SIMD backends, gated behind
//!   `#[cfg(target_arch = "aarch64")]` (so not linkable from these
//!   always-rendered docs). `arch::neon` holds
//!   `#[target_feature(enable = "neon")] unsafe fn` kernels.
//! - `dispatch` — runtime-detection routers. Each public dispatcher
//!   asserts its slice-length preconditions **unconditionally**, then
//!   picks NEON (if available) or the scalar fallback.
//! - this module (`simd`) — module doc + the public dispatcher
//!   re-exports + the [`neon_available`](crate::simd::neon_available)
//!   detector.
//!
//! ## Public surface
//!
//! - [`dot`](crate::simd::dot) — `Σ a[i] * b[i]`, f64.
//! - [`sum_of_squares`](crate::simd::sum_of_squares) — `Σ v[i]²`,
//!   f64. Used by [`crate::audio`]'s `integrated_loudness` for the
//!   per-block K-weighted mean-square.
//!
//! ## Always on — no cargo feature
//!
//! SIMD is **unconditional**: there is no `simd` cargo feature. Whether
//! the NEON backend runs is gated purely on `#[cfg(target_arch =
//! "aarch64")]` plus runtime CPU detection
//! ([`neon_available`](crate::simd::neon_available)); on every other
//! target the dispatchers route to [`scalar`](crate::simd::scalar)
//! automatically. The [`scalar`](crate::simd::scalar) and `dispatch`
//! layers therefore compile on **all** targets — only the `arch`
//! module is `aarch64`-gated (the
//! [`neon_available`](crate::simd::neon_available) detector is a
//! `const false` stub elsewhere).
//! This matches the `dia` reference (no simd feature). A pure-scalar
//! build for bisecting a numeric regression — even on a NEON-capable
//! host — is available via the `--cfg mlxrs_force_scalar` build escape
//! (see [`neon_available`](crate::simd::neon_available)).
//!
//! ## Cross-path determinism
//!
//! On `aarch64`, [`scalar`](crate::simd::scalar) and the
//! `arch::neon` kernels produce **bit-identical** results. This is
//! deliberate:
//!
//! 1. both use `f64::mul_add` for each per-element FMA — one IEEE 754
//!    rounding, identical to `vfmaq_f64`;
//! 2. [`scalar`](crate::simd::scalar)'s reduction tree mirrors NEON's
//!    — 4 partial sums over modulo-4 indices, combined
//!    `((s00 + s10) + (s01 + s11))`.
//!
//! Verified by the `differential_tests` module below
//! (`assert_eq!` on `f64::to_bits()`).
//!
//! ## Adding a new kernel — the per-kernel triple
//!
//! Every new kernel ships as **three pieces** that follow the in-tree
//! [`dot`](crate::simd::dot) worked example exactly:
//!
//! 1. **Scalar reference** — `pub fn foo_scalar(x: &[T]) -> R` under
//!    [`scalar`](crate::simd::scalar). Always compiled, independent of
//!    `target_arch`. Anchors the math contract, is the
//!    differential-test oracle, and is the fallback path. See
//!    [`scalar::dot`](crate::simd::scalar::dot) for the worked
//!    example.
//! 2. **NEON kernel** — `#[target_feature(enable = "neon")] unsafe fn
//!    foo(x: &[T]) -> R` under `arch::neon` (the whole module is
//!    `#[cfg(target_arch = "aarch64")]`-gated). See `arch::neon::dot`
//!    for the worked example — note the `pub(crate)` visibility and
//!    the `# Safety` doc-comment that names every caller obligation
//!    (NEON availability + every input precondition).
//! 3. **Dispatcher** — `pub fn foo(x: &[T]) -> R` under `dispatch::`,
//!    re-exported from this module. Asserts its slice-length
//!    preconditions **unconditionally** (release-too — the NEON
//!    kernel's `debug_assert!` is a no-op in release and would
//!    OOB-read), then
//!    `if neon_available() { unsafe { neon::foo(x) } } else { scalar::foo(x) }`.
//!    See the re-exported [`dot`](crate::simd::dot) for the worked
//!    example.
//!
//! ## Picking a differential-test class — `Exact` vs `Tolerance`
//!
//! Every new kernel also ships a scalar-vs-dispatcher differential
//! test using the helpers in [`diff`](crate::simd::diff):
//!
//! - **`Exact`** — call
//!   [`diff::assert_eq_over_lane_sweep`](crate::simd::diff::assert_eq_over_lane_sweep).
//!   Use for data-movement / lossless-widening kernels (C1 integer
//!   arms, C3, C4, C5, C6) — the SIMD output **must be bit-identical**
//!   to scalar. For fp outputs that are deliberately bit-identical
//!   (the in-tree `dot` is one — matched reduction tree), the
//!   `differential_tests` module below compares on `f64::to_bits()`
//!   instead.
//! - **`Tolerance { abs, rel }` — scalar output** — call
//!   [`diff::assert_close_over_lane_sweep`](crate::simd::diff::assert_close_over_lane_sweep).
//!   Use for fp-reduction / FMA-rounding kernels that fold the input
//!   to a single `f64` (C2 loudness sum-of-squares; any future fp
//!   reduction without a matched scalar reduction tree).
//! - **`Tolerance { abs, rel }` — vector output** — call
//!   [`diff::assert_close_slice_over_lane_sweep`](crate::simd::diff::assert_close_slice_over_lane_sweep).
//!   Use for fp kernels that return a `Vec<f64>` (C5 `rotate_buf`
//!   permutation, C10 `mel_filter_bank` triangle construction, C12
//!   window generation — the vector-producing fp candidates
//!   documented under `simd::audio` / `simd::vlm`). Asserts
//!   dispatcher and scalar outputs have the same length **and** every
//!   element pair satisfies the same `abs.max(rel * |s|)` tolerance
//!   as the scalar twin.
//!
//! All three helpers share the same length sweep
//! ([`diff::lane_sweep_lengths`](crate::simd::diff::lane_sweep_lengths)
//! — 9 lengths covering every boundary class: empty / singleton /
//! single-block-just-below / single-block-clean / single-block-plus-tail /
//! post-body large-tail / multi-block-clean ×2 / multi-block-clean ×3 /
//! multi-block-plus-tail), so coverage is uniform across `Exact` and
//! both `Tolerance` flavours.
//!
//! See the [`diff`](crate::simd::diff) module doc for the full class
//! catalog and the length-sweep rationale.
//!
//! ## Verify-before-claim (§5.4 of the SIMD doc)
//!
//! Before committing a hand-written NEON kernel for any new
//! candidate, **benchmark scalar vs SIMD** in the `mlxrs-m2-benches`
//! worktree under release. Some candidates (C3 RGB widen, the cold
//! one-time table builds C10/C11/C12) are *suspected already
//! auto-vectorized by LLVM* or *too cold to matter* — shipping a
//! hand-rolled intrinsic kernel that LLVM already emits is dead
//! weight (extra `unsafe`, extra maintenance, no perf win). The
//! benchmark is the gate, not intuition or "it should be faster". See
//! `docs/core-arch-simd-candidates.md` §5.4 and the
//! [project memory rule "Verify review premise empirically"].
//!
//! ## Suggested execution order (§5.5 of the SIMD doc)
//!
//! The cross-cutting doc orders the candidate work by
//! risk × benefit:
//!
//! 1. **C6** — `pad_to_square` fill (quick win, lowest risk).
//! 2. **C1** — PCM decode widen (highest steady benefit, exact
//!    semantics).
//! 3. **C4** — BGR `vld3`/`vst3` de-interleave widen (clean NEON
//!    fit, exact, the arm LLVM most likely misses).
//! 4. **C3** — RGB widen, **only if** disassembly shows LLVM is not
//!    already vectorizing.
//! 5. **C2** — loudness sum-of-squares (fp-associativity / LUFS
//!    bit-exactness risk; needs existing loudness test tolerances
//!    re-baselined; do *after* C1/C3/C4).
//! 6. **Defer C5, C7, C8, C10, C11, C12** — cold, gather-bound, or
//!    transcendental-bound; revisit only if a benchmark proves a
//!    real cost.
//! 7. **Never** — C9 (`lfilter` recurrence — serial by
//!    construction).
//!
//! ## No cargo feature — always on (project memory override)
//!
//! Per the project memory rule **"SIMD always-on"**, mlxrs's SIMD
//! infrastructure has **no `simd` cargo feature**. Whether the NEON
//! backend runs is gated purely on `#[cfg(target_arch = "aarch64")]`
//! plus runtime CPU detection
//! ([`is_neon_available`](crate::simd::is_neon_available)); on every
//! other target the dispatchers route to the always-compiled scalar
//! path. This explicitly overrides §5.2 of the SIMD doc, which had
//! recommended a Cargo-feature gate before the project rule was set.

#[cfg(target_arch = "aarch64")]
pub mod arch;
#[cfg(target_arch = "aarch64")]
pub(crate) mod audio;
pub mod diff;
mod dispatch;
pub mod scalar;
// `simd::vlm` is **not** `aarch64`-gated: the scalar reference inside
// each kernel triple (e.g. `pad_canvas_fill_scalar`) must compile on
// every target so the dispatcher's scalar-fallback branch is a real,
// linkable function (per the project memory rule "SIMD always-on" —
// scalar fallback compiles on all targets, only the `arch` module is
// `aarch64`-gated). The NEON kernels inside each kernel triple are
// individually `#[cfg(target_arch = "aarch64")]`-gated at the
// function level. The module is `vlm`-feature-gated so the
// `--no-default-features` / per-feature CI builds don't compile a
// dead-code dispatcher (the only call site is
// [`crate::vlm::image::pad_to_square`], itself behind the same
// feature).
//
// `pub` (rather than `pub(crate)`) so the in-tree
// `benches/simd_pad_canvas_fill.rs` micro-benchmark — a separate
// binary that only sees the public API — can drive the dispatcher
// and scalar reference directly per the verify-before-claim rule
// (§5.4 of the SIMD doc). Per-kernel items inside are individually
// `pub`/`pub(crate)` (only the dispatcher + the scalar reference
// are exposed; the `unsafe` NEON kernel stays `pub(crate)`).
#[cfg(feature = "vlm")]
#[doc(hidden)]
pub mod vlm;

pub use dispatch::{dot, sum_of_squares};

// ─── runtime CPU-feature detection ───────────────────────────────────
//
// `--cfg mlxrs_force_scalar` overrides detection so the scalar path
// can be exercised even on a NEON-capable host — set it via
// `RUSTFLAGS="--cfg mlxrs_force_scalar"`.

/// Whether the NEON SIMD backend is usable on the executing CPU.
///
/// `true` when NEON is reported by the CPU **and** `--cfg
/// mlxrs_force_scalar` is not set. NEON is part of the AArch64
/// baseline, so on a normal `aarch64` host this is effectively always
/// `true`; the explicit check keeps the scalar fallback a real,
/// reachable branch and honours the force-scalar escape.
///
/// On every non-`aarch64` target there is no NEON backend to gate, so
/// this is a `const false` stub: every dispatcher then routes to
/// [`scalar`]. The stub keeps the symbol present on all targets so
/// intra-doc links resolve in a non-`aarch64` rustdoc build.
#[cfg(target_arch = "aarch64")]
pub fn neon_available() -> bool {
  if cfg!(mlxrs_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

/// Whether the NEON SIMD backend is usable on the executing CPU.
///
/// `true` when NEON is reported by the CPU **and** `--cfg
/// mlxrs_force_scalar` is not set. NEON is part of the AArch64
/// baseline, so on a normal `aarch64` host this is effectively always
/// `true`; the explicit check keeps the scalar fallback a real,
/// reachable branch and honours the force-scalar escape.
///
/// On every non-`aarch64` target there is no NEON backend to gate, so
/// this is a `const false` stub: every dispatcher then routes to
/// [`scalar`]. The stub keeps the symbol present on all targets so
/// intra-doc links resolve in a non-`aarch64` rustdoc build.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_available() -> bool {
  false
}

/// Whether the NEON SIMD backend is usable on the executing CPU.
///
/// Canonical name matching the [`std::arch::is_aarch64_feature_detected`]
/// idiom — wraps `is_aarch64_feature_detected!("neon")` on `aarch64`
/// and is a `const false` stub on every other target. New per-kernel
/// dispatchers (the C1–C12 follow-ups landing under `simd::audio` /
/// `simd::vlm`) call this directly so the runtime-detection branch
/// has a uniform shape across the crate.
///
/// Functionally identical to [`neon_available`] (which the in-tree
/// [`dot`] dispatcher uses); the duplicate name exists so callers can
/// pick the idiom that reads best at the call site
/// (`if is_neon_available()` mirrors `is_aarch64_feature_detected!`;
/// `if neon_available()` reads naturally as a noun-style query). Both
/// honour the `--cfg mlxrs_force_scalar` build escape.
#[inline]
pub fn is_neon_available() -> bool {
  neon_available()
}

#[cfg(test)]
mod differential_tests {
  //! Scalar vs NEON differential tests.
  //!
  //! Contract: on `aarch64` (the deployment target), [`super::scalar`]
  //! and [`super::arch::neon`] produce **bit-identical** results for
  //! every primitive. Achieved by (1) `f64::mul_add` per-element FMA
  //! on both, and (2) a scalar reduction tree that mirrors NEON's
  //! (4 partial sums over modulo-4 indices, then `((s00 + s10) +
  //! (s01 + s11))`).
  //!
  //! On non-`aarch64` targets the dispatcher routes to scalar, so the
  //! differential check is a scalar-vs-scalar identity — still a
  //! useful regression guard, and trivially bit-equal.

  /// `sum_of_squares` scalar vs the SIMD dispatcher, over lengths that
  /// straddle the 2-lane / 4-wide-unroll boundaries (the scalar-tail
  /// paths). On `aarch64` the matched reduction tree makes this
  /// bit-identical — asserted on `to_bits()`.
  #[test]
  fn sum_of_squares_scalar_matches_simd() {
    // Lengths straddle the 4-wide unroll (mod 4 ∈ {0,1,2,3}) and the
    // 2-wide tail; 0 exercises the empty case.
    for n in [0usize, 1, 3, 7, 8, 9, 17, 33, 129, 1024] {
      // A spread of magnitudes / signs — fully deterministic (no rng
      // dep): a sign-alternating, slowly-growing sequence.
      let v: Vec<f64> = (0..n)
        .map(|i| {
          let mag = 0.5 + (i as f64) * 0.013_f64;
          if i % 2 == 0 { mag } else { -mag }
        })
        .collect();
      let s = super::scalar::sum_of_squares(&v);
      let d = super::sum_of_squares(&v);
      assert_eq!(
        s.to_bits(),
        d.to_bits(),
        "sum_of_squares n={n}: scalar/SIMD not bit-identical (s={s}, d={d})"
      );
    }
  }

  /// `dot` scalar vs the SIMD dispatcher, same length sweep. The
  /// `sum_of_squares` kernel is the `b ≡ a` specialization of `dot`;
  /// covering `dot` directly locks the general kernel's reduction
  /// tree too.
  #[test]
  fn dot_scalar_matches_simd() {
    for n in [0usize, 1, 3, 7, 8, 9, 17, 33, 129, 1024] {
      let a: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64) * 0.013_f64).collect();
      let b: Vec<f64> = (0..n)
        .map(|i| {
          let mag = 0.25 + (i as f64) * 0.007_f64;
          if i % 3 == 0 { -mag } else { mag }
        })
        .collect();
      let s = super::scalar::dot(&a, &b);
      let d = super::dot(&a, &b);
      assert_eq!(
        s.to_bits(),
        d.to_bits(),
        "dot n={n}: scalar/SIMD not bit-identical (s={s}, d={d})"
      );
    }
  }

  /// A self-dot (`Σ a[i]²` via `dot(a, a)`) must equal
  /// `sum_of_squares(a)` bit-for-bit — the specialization claim. Both
  /// run the same reduction tree; this pins that they agree.
  #[test]
  fn sum_of_squares_equals_self_dot() {
    for n in [0usize, 1, 7, 8, 33, 129] {
      let a: Vec<f64> = (0..n)
        .map(|i| {
          let mag = 0.5 + (i as f64) * 0.011_f64;
          if i % 2 == 0 { mag } else { -mag }
        })
        .collect();
      let via_dot = super::dot(&a, &a);
      let via_ss = super::sum_of_squares(&a);
      assert_eq!(
        via_dot.to_bits(),
        via_ss.to_bits(),
        "n={n}: sum_of_squares != dot(a, a) (dot={via_dot}, ss={via_ss})"
      );
    }
  }

  /// Large-magnitude / catastrophic-cancellation-style band test.
  /// `dia` does the analogous check on `dot`. `sum_of_squares` cannot
  /// cancel (every term `v²` is ≥ 0), but a `1e16` element dwarfs the
  /// unit-scale ones — reduction order still matters. The matched
  /// scalar/NEON trees keep them bit-identical; the absolute value is
  /// pinned so a future kernel rewrite that changes the order
  /// surfaces here.
  #[test]
  fn sum_of_squares_large_magnitude_band() {
    let v: Vec<f64> = vec![1e16, 1.0, 1.0, 1.0, 1e16, 1.0, 1.0];
    let s = super::scalar::sum_of_squares(&v);
    let d = super::sum_of_squares(&v);
    assert_eq!(
      s.to_bits(),
      d.to_bits(),
      "large-magnitude sum_of_squares: scalar/SIMD not bit-identical (s={s}, d={d})"
    );
    // Two 1e16 terms dominate: `2 * (1e16)² = 2e32`. The five unit
    // terms (`+5`) are far below the f64 ulp at 2e32 and are lost —
    // this is expected, not a bug; it just pins the reduction's
    // large-magnitude behaviour.
    assert!(
      (s - 2e32).abs() < 1e17,
      "sum_of_squares large-magnitude result drifted from ~2e32: {s}"
    );
  }

  /// Mismatched `dot` lengths must `panic!` — not OOB-read. The
  /// dispatcher asserts `a.len() == b.len()` **unconditionally**
  /// before routing to the unsafe NEON kernel; this test would
  /// silently read past `b` if that guard were `debug_assert!`-only.
  #[test]
  #[should_panic(expected = "simd::dot")]
  fn dot_dispatch_panics_on_length_mismatch() {
    let a = vec![1.0_f64; 8];
    let b = vec![1.0_f64; 4];
    let _ = super::dot(&a, &b);
  }

  /// `scalar::dot` (the public scalar API, not the dispatcher) must
  /// `panic!` on mismatched lengths with the **shorter slice first**.
  /// Pre-fix this branch used a `debug_assert!`, so in release it
  /// silently returned the dot over `a.len()` (the shorter length) —
  /// a wrong result, no panic. The unconditional `assert_eq!` makes
  /// it fail loud, matching the dispatcher.
  #[test]
  #[should_panic(expected = "scalar::dot")]
  fn scalar_dot_panics_on_length_mismatch_shorter_first() {
    let a = vec![1.0_f64; 4];
    let b = vec![1.0_f64; 8];
    let _ = super::scalar::dot(&a, &b);
  }

  /// Symmetric case: `scalar::dot` with the **longer slice first**.
  /// This direction always panicked (out-of-bounds indexing of `b`),
  /// but it now panics via the explicit length `assert_eq!` instead —
  /// the contract is symmetric, fail-loud in both orders.
  #[test]
  #[should_panic(expected = "scalar::dot")]
  fn scalar_dot_panics_on_length_mismatch_longer_first() {
    let a = vec![1.0_f64; 8];
    let b = vec![1.0_f64; 4];
    let _ = super::scalar::dot(&a, &b);
  }

  /// On `aarch64`, NEON is baseline-mandatory — every `aarch64` core
  /// has it (ARMv8.0+) — so [`super::is_neon_available`] must report
  /// `true` here on the deployment target. This pins the
  /// runtime-detection contract: a flip to `false` (e.g. someone
  /// accidentally inverting `is_aarch64_feature_detected!`) would
  /// silently route every dispatcher to the scalar path and quietly
  /// halve perf.
  ///
  /// Skipped under `--cfg mlxrs_force_scalar` — the force-scalar build
  /// escape deliberately returns `false` to bisect SIMD-vs-scalar
  /// numeric regressions; asserting `true` there would defeat the
  /// escape.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn is_neon_available_true_on_aarch64() {
    if cfg!(mlxrs_force_scalar) {
      // Force-scalar build: the detector deliberately returns false
      // (see [`super::neon_available`]); skip the positive assertion
      // so the escape stays a real, exercised branch.
      return;
    }
    assert!(
      super::is_neon_available(),
      "NEON is baseline-mandatory on aarch64; is_neon_available() must report true"
    );
  }
}
