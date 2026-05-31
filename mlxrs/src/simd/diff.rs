//! Reusable scalar-vs-SIMD differential-test helpers.
//!
//! For every SIMD kernel triple (`dispatcher` + `arch::neon` +
//! `scalar`), the contract anchor is the scalar reference. The
//! correctness check runs the scalar and the dispatcher (which on
//! `aarch64` routes to the NEON kernel) over the same input and
//! compares — but the comparison shape depends on whether the kernel is
//! lossless or fp-reducing.
//!
//! # Two correctness classes
//!
//! Split kernels into:
//!
//! - **[`assert_eq_over_lane_sweep`]** — the `Exact` class. Used for
//!   data-movement / lossless-widening kernels:
//!     - integer arms of PCM decode-widen ([#146](https://github.com/Findit-AI/mlxrs/issues/146)),
//!     - u8→f32 RGB widen ([#148](https://github.com/Findit-AI/mlxrs/issues/148)),
//!     - BGR R↔B swap widen ([#149](https://github.com/Findit-AI/mlxrs/issues/149)),
//!     - `pad_to_square` fill ([#151](https://github.com/Findit-AI/mlxrs/issues/151)),
//!     - `rotate_buf` permutation ([#150](https://github.com/Findit-AI/mlxrs/issues/150)).
//!
//!   The SIMD output **must be bit-identical** to scalar; the helper
//!   asserts `scalar_out == simd_out`. For `f32`/`f64` outputs you
//!   typically want bit-level equality via `to_bits()` instead — see
//!   the worked `dot` differential test in [`crate::simd`] for the
//!   bit-exact reduction-tree contract.
//!
//! - **[`assert_close_over_lane_sweep`]** — the `Tolerance` class,
//!   *scalar* output. Used for fp-reduction / FMA-rounding kernels
//!   that fold the input to a single `f64` and where SIMD changes
//!   summation order or rounding (in general — note the in-tree
//!   `dot`/`sum_of_squares` are deliberately bit-identical because the
//!   scalar tree mirrors NEON's, but the generic `Tolerance` shape is
//!   the right shape for *new* fp reductions where that property has
//!   not been engineered):
//!     - loudness sum-of-squares ([#147](https://github.com/Findit-AI/mlxrs/issues/147)),
//!     - any future fp reduction without a matched reduction tree.
//!
//!   The helper asserts
//!   `(scalar - simd).abs() <= abs.max(rel * scalar.abs())`.
//!
//! - **[`assert_close_slice_over_lane_sweep`]** — the `Tolerance`
//!   class, *vector* output. The elementwise twin of
//!   [`assert_close_over_lane_sweep`] for fp kernels that return a
//!   `Vec<f64>`:
//!     - `rotate_buf` permutation, vector output ([#150](https://github.com/Findit-AI/mlxrs/issues/150)),
//!     - `mel_filter_bank` triangle construction ([#155](https://github.com/Findit-AI/mlxrs/issues/155)),
//!     - window generation, Hann / Hamming ([#157](https://github.com/Findit-AI/mlxrs/issues/157)),
//!     - any future fp kernel that emits a vector rather than a scalar.
//!
//!   The helper asserts the dispatcher and scalar outputs have the
//!   same length, and that every element pair satisfies the same
//!   `(s - d).abs() <= abs.max(rel * s.abs())` shape as the scalar
//!   twin.
//!
//! # Length sweep
//!
//! All three helpers call the input generator at lengths
//! `{0, 1, lanes-1, lanes, lanes+1, 2*lanes-1, 2*lanes, 3*lanes, 3*lanes+1}`
//! so the kernel's head / body / tail paths are *all* exercised
//! (covering the tail / remainder). The two clean-multi-block
//! lengths (`2*lanes`, `3*lanes`) specifically catch off-by-one bugs
//! on the chunk-loop bound that a sweep without them would miss (a
//! kernel that mis-handles `len == k * lanes && tail == 0` can pass a
//! sweep that only carries multi-block-plus-tail). The `2*lanes-1`
//! length is the post-body large-tail case: a kernel whose
//! post-vector tail loop only handles a single remainder element
//! would silently pass a sweep limited to `lanes+1` / `3*lanes+1`
//! (both `tail == 1`) but fail at `2*lanes-1` (`full body +
//! (lanes-1)`-element remainder). `lanes` is parameterized at the
//! call site (e.g. `lanes = 2` for `float64x2_t`-based kernels,
//! `lanes = 4` for `float32x4_t`, `lanes = 16` for `uint8x16_t`).
//!
//! # Signature shape
//!
//! All three helpers take:
//! 1. the scalar reference function pointer (`fn(&[T]) -> R`),
//! 2. the public dispatcher function pointer (`fn(&[T]) -> R`) —
//!    internally selects NEON or scalar,
//! 3. an input generator (`fn(usize) -> Vec<T>` — receives the length
//!    and returns a deterministic input of that length).
//!
//! The output type `R` is `R: PartialEq + core::fmt::Debug` for
//! [`assert_eq_over_lane_sweep`], `f64` for
//! [`assert_close_over_lane_sweep`], and `Vec<f64>` for
//! [`assert_close_slice_over_lane_sweep`]. The two tolerance helpers
//! share `f64` as the comparison type for symmetry — call sites with
//! `f32` outputs widen via `as f64` (lossless).
//!
//! The generator returning a fresh `Vec` for each length is
//! deliberate: it keeps the helper trivially `Send` / `Sync`-free
//! (helpful for `--test-threads=1` runs) and lets the call site decide
//! the data distribution (random-seeded, alternating-sign, edge values
//! like `i16::MIN`/`MAX`, etc.).

/// Returns the canonical SIMD lane-sweep coverage for `lanes`-wide
/// kernels.
///
/// Covers nine boundary classes the helpers
/// [`assert_eq_over_lane_sweep`], [`assert_close_over_lane_sweep`],
/// and [`assert_close_slice_over_lane_sweep`] drive their generator
/// at:
///
/// 1. `0`               — empty input (no body, no tail).
/// 2. `1`               — singleton (degenerate-tail-only).
/// 3. `lanes - 1`       — single-block-just-below (pure-tail, no body).
/// 4. `lanes`           — single-block-clean (one body, no tail).
/// 5. `lanes + 1`       — single-block-plus-tail (body + tail).
/// 6. `2 * lanes - 1`   — post-body large-tail (full body + `lanes-1`-element remainder).
/// 7. `2 * lanes`       — multi-block-clean ×2 (two bodies, no tail).
/// 8. `3 * lanes`       — multi-block-clean ×3 (three bodies, no tail).
/// 9. `3 * lanes + 1`   — multi-block-plus-tail (three bodies + tail).
///
/// The two clean-multi-block lengths catch chunk-loop off-by-one bugs
/// that a sweep carrying only `3 * lanes + 1` would miss — a kernel
/// that mis-handles `len == k * lanes && tail == 0` (e.g. an
/// inclusive vs exclusive chunk bound) can still pass the
/// multi-block-plus-tail length. The `2 * lanes - 1` length is the
/// post-body large-tail case: a kernel whose post-vector tail loop
/// only handles a single remainder element would silently pass a
/// sweep limited to `lanes + 1` / `3 * lanes + 1` (both `tail == 1`)
/// but fail at `2 * lanes - 1` (full body + `lanes-1` remainder).
///
/// `lanes` is internally clamped to `>= 1` so `lanes == 0` is
/// well-defined: it collapses (1 → 1, 0 → 0, 1 → 1, 2 → 2, 1 → 1,
/// 3 → 3, 4 → 4) but still covers the empty + singleton + small body
/// cases. Some entries collide for very small `lanes` (e.g.
/// `lanes == 1` gives `[0, 1, 0, 1, 2, 1, 2, 3, 4]`); that is fine —
/// duplicates are harmless (re-running a length is cheap) and the
/// helper loop has no dedup requirement.
///
/// Returning a fixed-size array (`[usize; 9]`) keeps the signature
/// stack-only and avoids any heap allocation — the sweep is
/// constant-shape, callers iterate by `for &n in &lane_sweep_lengths(l)`.
///
/// Exposed as `pub` so a test that needs a non-standard shape (e.g. a
/// kernel taking two slices `dot(a, b)`) can build its own loop using
/// the same length set, instead of reimplementing the sweep.
#[inline]
pub fn lane_sweep_lengths(lanes: usize) -> [usize; 9] {
  // Clamp to >= 1 so `lanes == 0` doesn't degenerate every multi-block
  // length to 0 (which would silently strip coverage). `lanes - 1`
  // still uses `saturating_sub` on the clamped value for clarity.
  // `2 * l - 1` uses `saturating_sub` so the `l == 1` case stays safe
  // (collapses to `1`, harmlessly duplicating the singleton entry).
  let l = lanes.max(1);
  [
    0,
    1,
    l.saturating_sub(1),
    l,
    l + 1,
    (2 * l).saturating_sub(1),
    2 * l,
    3 * l,
    3 * l + 1,
  ]
}

/// `Exact` differential-test class — asserts the SIMD dispatcher's
/// output equals the scalar reference output, **bit-for-bit** (`==`)
/// at every length in [`lane_sweep_lengths`].
///
/// Use for data-movement / lossless-widening kernels — the SIMD output
/// must be bit-identical to scalar (no rounding, no FMA, no
/// re-association). See the module doc for the kernel-class catalog
/// (the integer-widen arms, RGB and BGR widen, `rotate_buf`, and
/// `pad_to_square` fill).
///
/// For `f32`/`f64` outputs, prefer comparing on `.to_bits()` at the
/// call site (the existing `dot` / `sum_of_squares` differential tests
/// in [`crate::simd`] do this) — `R: PartialEq` accepts both, but
/// `f64::PartialEq` treats `NaN != NaN`, which can mask a genuine
/// bit-equal-NaN regression.
///
/// # Parameters
///
/// - `lanes` — the kernel's lane width (e.g. 2 for `float64x2_t`),
///   used to drive the length sweep.
/// - `scalar_fn` — the scalar reference (`fn(&[T]) -> R`).
/// - `dispatcher_fn` — the public dispatcher (`fn(&[T]) -> R`) — on
///   `aarch64` routes to the NEON kernel, elsewhere to `scalar_fn`.
/// - `gen_input` — deterministic input factory (`fn(usize) -> Vec<T>`).
///
/// # Panics
///
/// On the first length where `scalar_fn(input) != dispatcher_fn(input)`
/// — the message includes the failing length.
pub fn assert_eq_over_lane_sweep<T, R, S, D, G>(
  lanes: usize,
  scalar_fn: S,
  dispatcher_fn: D,
  gen_input: G,
) where
  R: PartialEq + core::fmt::Debug,
  S: Fn(&[T]) -> R,
  D: Fn(&[T]) -> R,
  G: Fn(usize) -> Vec<T>,
{
  for n in lane_sweep_lengths(lanes) {
    let input = gen_input(n);
    let s = scalar_fn(&input);
    let d = dispatcher_fn(&input);
    assert_eq!(
      s, d,
      "Exact differential check failed at n={n} (lanes={lanes}): scalar != dispatcher"
    );
  }
}

/// `Tolerance` differential-test class — asserts the SIMD dispatcher's
/// output is within `abs.max(rel * scalar.abs())` of the scalar
/// reference at every length in [`lane_sweep_lengths`].
///
/// Use for fp-reduction / FMA-rounding kernels (loudness
/// sum-of-squares; any future fp reduction without a deliberately
/// matched reduction tree). The check shape mirrors `numpy.isclose` /
/// `approx::abs_diff_eq` / `assert_relative_eq!`: pass if
/// `|s - d| <= abs OR |s - d| <= rel * |s|` (combined as
/// `|s - d| <= abs.max(rel * |s|)` to be lenient at both small and
/// large magnitudes).
///
/// # Parameters
///
/// - `lanes`, `scalar_fn`, `dispatcher_fn`, `gen_input` — as for
///   [`assert_eq_over_lane_sweep`].
/// - `abs` — absolute tolerance (use when `|scalar|` may be ~0).
/// - `rel` — relative tolerance (use when `|scalar|` is large).
///
/// # Panics
///
/// On the first length where the tolerance is exceeded — the message
/// includes the failing length, both outputs, and the tolerance used.
pub fn assert_close_over_lane_sweep<T, S, D, G>(
  lanes: usize,
  scalar_fn: S,
  dispatcher_fn: D,
  gen_input: G,
  abs: f64,
  rel: f64,
) where
  S: Fn(&[T]) -> f64,
  D: Fn(&[T]) -> f64,
  G: Fn(usize) -> Vec<T>,
{
  for n in lane_sweep_lengths(lanes) {
    let input = gen_input(n);
    let s = scalar_fn(&input);
    let d = dispatcher_fn(&input);
    let tol = abs.max(rel * s.abs());
    assert!(
      (s - d).abs() <= tol,
      "Tolerance differential check failed at n={n} (lanes={lanes}): \
       scalar={s} dispatcher={d} |diff|={diff} tol={tol} (abs={abs}, rel={rel})",
      diff = (s - d).abs(),
    );
  }
}

/// `Tolerance` differential-test class for kernels that return a
/// slice / `Vec` of fp values — the elementwise twin of
/// [`assert_close_over_lane_sweep`].
///
/// Asserts at every length in [`lane_sweep_lengths`]:
///
/// - the dispatcher's output length equals the scalar reference's
///   output length (length-contract regression guard);
/// - each elementwise pair `(s, d)` satisfies
///   `(s - d).abs() <= abs.max(rel * s.abs())` — the same combined
///   abs/rel shape as [`assert_close_over_lane_sweep`].
///
/// Use for vector-producing fp kernels — the `rotate_buf`
/// permutation, `mel_filter_bank` triangle construction, and
/// window-generation kernels documented under `simd::audio` /
/// `simd::vlm` (both modules are `pub(crate)` so are not
/// intra-doc-linked from this public helper). The existing scalar
/// [`assert_close_over_lane_sweep`] only covers fp *scalar*
/// reductions (loudness sum-of-squares); this sibling covers their
/// vector-output counterparts.
///
/// # Output type
///
/// Returns `Vec<f64>` deliberately, matching the scalar twin's `f64`
/// return — keeping both `Tolerance` helpers a symmetric API pair
/// rather than introducing a `ToF64` trait or a parallel `_f32`
/// variant for marginal gain. Call sites with `Vec<f32>` kernels
/// promote via `.iter().map(|&x| x as f64).collect()` (an exact
/// widening) the same way the scalar twin's `f64` return already
/// requires; the cost is one allocation per length-sweep step, in
/// test code only.
///
/// # Parameters
///
/// - `lanes`, `scalar_fn`, `dispatcher_fn`, `gen_input`, `abs`,
///   `rel` — semantics identical to [`assert_close_over_lane_sweep`],
///   except `scalar_fn` and `dispatcher_fn` return `Vec<f64>`.
///
/// # Panics
///
/// - On the first length where the dispatcher and scalar outputs have
///   different lengths — the message includes the failing length and
///   both lengths.
/// - On the first elementwise pair that exceeds the tolerance — the
///   message includes the failing length, the element index, both
///   scalar values, the diff, and the tolerance used.
pub fn assert_close_slice_over_lane_sweep<T, S, D, G>(
  lanes: usize,
  scalar_fn: S,
  dispatcher_fn: D,
  gen_input: G,
  abs: f64,
  rel: f64,
) where
  S: Fn(&[T]) -> Vec<f64>,
  D: Fn(&[T]) -> Vec<f64>,
  G: Fn(usize) -> Vec<T>,
{
  for n in lane_sweep_lengths(lanes) {
    let input = gen_input(n);
    let scalar_out = scalar_fn(&input);
    let simd_out = dispatcher_fn(&input);
    assert_eq!(
      scalar_out.len(),
      simd_out.len(),
      "Tolerance-slice length mismatch at n={n} (lanes={lanes}): \
       scalar.len={s_len} dispatcher.len={d_len}",
      s_len = scalar_out.len(),
      d_len = simd_out.len(),
    );
    for (i, (&s, &d)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
      let tol = abs.max(rel * s.abs());
      assert!(
        (s - d).abs() <= tol,
        "Tolerance-slice differential check failed at n={n} i={i} \
         (lanes={lanes}): scalar={s} dispatcher={d} |diff|={diff} \
         tol={tol} (abs={abs}, rel={rel})",
        diff = (s - d).abs(),
      );
    }
  }
}

#[cfg(test)]
mod tests {
  //! Self-tests of the differential helpers, using the in-tree `dot`
  //! dispatcher (which already ships scalar + NEON) as the SIMD pair.
  //!
  //! These tests **do not** validate `dot` itself — that is the
  //! responsibility of the per-kernel differential tests in
  //! [`crate::simd`]. They validate that the helper drives a passing
  //! test across the lane-sweep set for both correctness classes.
  use super::{
    assert_close_over_lane_sweep, assert_close_slice_over_lane_sweep, assert_eq_over_lane_sweep,
    lane_sweep_lengths,
  };

  /// `lane_sweep_lengths` produces the documented 9-length set across
  /// the canonical NEON lane widths the in-tree kernels use
  /// (`lanes ∈ {1, 2, 4, 8, 16, 32}`). Locking the exact array per
  /// lane width pins the coverage contract — every boundary class
  /// (empty / singleton / single-block-just-below /
  /// single-block-clean / single-block-plus-tail / post-body
  /// large-tail / multi-block-clean ×2 / multi-block-clean ×3 /
  /// multi-block-plus-tail) is present at every lane width, and
  /// adding/removing/shuffling an entry surfaces here loudly.
  #[test]
  fn lane_sweep_lengths_full_coverage() {
    assert_eq!(lane_sweep_lengths(1), [0, 1, 0, 1, 2, 1, 2, 3, 4]);
    assert_eq!(lane_sweep_lengths(2), [0, 1, 1, 2, 3, 3, 4, 6, 7]); // `float64x2_t`.
    assert_eq!(lane_sweep_lengths(4), [0, 1, 3, 4, 5, 7, 8, 12, 13]); // `float32x4_t`.
    assert_eq!(lane_sweep_lengths(8), [0, 1, 7, 8, 9, 15, 16, 24, 25]); // `int16x8_t`.
    assert_eq!(lane_sweep_lengths(16), [0, 1, 15, 16, 17, 31, 32, 48, 49]); // `uint8x16_t`.
    assert_eq!(lane_sweep_lengths(32), [0, 1, 31, 32, 33, 63, 64, 96, 97]); // 32-wide composite.
    // Edge: `lanes == 0` is well-defined — the helper clamps `l = lanes.max(1)`
    // so the sweep collapses to the `lanes == 1` shape rather than degenerating
    // every multi-block entry to 0.
    assert_eq!(lane_sweep_lengths(0), [0, 1, 0, 1, 2, 1, 2, 3, 4]);
  }

  /// `Exact` class self-test using a trivial pass-through (`sum-by-fold`)
  /// kernel — both the "scalar" and "dispatcher" are the same function,
  /// so equality is tautological; this exercises the helper's
  /// length-sweep + assert-shape, not a real SIMD kernel. The bit-exact
  /// dispatcher contract for `dot` itself is checked separately in
  /// [`crate::simd`]'s `differential_tests`.
  #[test]
  fn exact_helper_passes_on_passthrough_kernel() {
    // A trivial Exact-class kernel: sum the integer slice as `i64`.
    // Identical "scalar" and "dispatcher" closures => guaranteed equal.
    fn sum_i32(xs: &[i32]) -> i64 {
      xs.iter().map(|&x| x as i64).sum()
    }
    assert_eq_over_lane_sweep(
      4, // pretend lanes=4 for a `int32x4_t`-shaped kernel.
      sum_i32,
      sum_i32,
      |n| (0..n as i32).collect::<Vec<i32>>(),
    );
  }

  /// `Tolerance` class self-test using the real in-tree
  /// [`crate::simd::dot`] dispatcher against
  /// [`crate::simd::scalar::dot`] — both are fp-reductions, so this
  /// is the kernel class the helper is designed for. On `aarch64`
  /// they happen to be bit-identical (so any positive tolerance
  /// trivially passes); on other targets the dispatcher routes to
  /// scalar (so equality is exact). The helper still exercises its
  /// length-sweep + tolerance comparison shape.
  #[test]
  fn tolerance_helper_passes_on_dot_dispatcher() {
    // Need a 2-arg kernel (`dot(a, b)`) but the helper takes
    // `fn(&[T]) -> R`. Wrap by splitting the input slice in half.
    fn dot_split_scalar(xs: &[f64]) -> f64 {
      let mid = xs.len() / 2;
      let n = mid; // both halves get n elements; tail (if odd) is dropped.
      crate::simd::scalar::dot(&xs[..n], &xs[mid..mid + n])
    }
    fn dot_split_dispatch(xs: &[f64]) -> f64 {
      let mid = xs.len() / 2;
      let n = mid;
      crate::simd::dot(&xs[..n], &xs[mid..mid + n])
    }
    assert_close_over_lane_sweep(
      2, // dot's NEON kernel is `float64x2_t` — 2 lanes.
      dot_split_scalar,
      dot_split_dispatch,
      |n| {
        // 2 halves of `n` each ⇒ caller-side slice length `2 * n`.
        // Deterministic, mildly-signed magnitudes.
        (0..2 * n)
          .map(|i| {
            let mag = 0.5 + (i as f64) * 0.013_f64;
            if i % 2 == 0 { mag } else { -mag }
          })
          .collect()
      },
      // Bit-identical on aarch64 ⇒ any positive tolerance passes; on
      // other targets routes to scalar and is exact. Use a small but
      // non-zero abs so a future fp-reorder regression in `dot` would
      // surface here too.
      1e-12,
      1e-12,
    );
  }

  /// `Tolerance`-slice self-test using a trivial pass-through kernel:
  /// both the "scalar" and "dispatcher" produce
  /// `(0..n).map(|i| i as f64).collect()`. Asserts the helper drives
  /// the length sweep, the per-length length-equality check, and the
  /// elementwise tolerance comparison without spurious panics — the
  /// vector-output twin of [`exact_helper_passes_on_passthrough_kernel`]
  /// + [`tolerance_helper_passes_on_dot_dispatcher`].
  #[test]
  fn slice_tolerance_helper_passes_on_passthrough_kernel() {
    fn identity_vec_f64(xs: &[i32]) -> Vec<f64> {
      xs.iter().map(|&x| x as f64).collect()
    }
    assert_close_slice_over_lane_sweep(
      4, // pretend lanes=4 for a `float32x4_t`-shaped vector-producing kernel.
      identity_vec_f64,
      identity_vec_f64,
      |n| (0..n as i32).collect::<Vec<i32>>(),
      // Identical closures ⇒ pointwise diff is exactly 0.0; any
      // positive tolerance passes. Use a small but non-zero abs/rel
      // so the comparison shape is exercised, not short-circuited.
      1e-12,
      1e-12,
    );
  }

  /// `Tolerance`-slice self-test that the helper panics with the
  /// documented message shape when the dispatcher's output diverges
  /// outside tolerance. The dispatcher adds `1.0` to every element
  /// (well above the `1e-12` abs / rel tolerance), so the helper must
  /// trip on the elementwise comparison at the first non-zero index.
  /// `should_panic(expected = "Tolerance-slice differential check failed")`
  /// pins the failure-message contract — a future helper rename or
  /// reword would surface here.
  #[test]
  #[should_panic(expected = "Tolerance-slice differential check failed")]
  fn slice_tolerance_helper_fails_on_divergent_pair() {
    fn identity_vec_f64(xs: &[i32]) -> Vec<f64> {
      xs.iter().map(|&x| x as f64).collect()
    }
    fn divergent_vec_f64(xs: &[i32]) -> Vec<f64> {
      xs.iter().map(|&x| x as f64 + 1.0).collect()
    }
    assert_close_slice_over_lane_sweep(
      4,
      identity_vec_f64,
      divergent_vec_f64,
      // Non-empty length needed for the divergence to bite — the
      // helper sweeps `n == 0` first (no elements to compare ⇒ no
      // panic at the empty length), then `n == 1` where the
      // single-element output differs by 1.0.
      |n| (0..n.max(1) as i32).collect::<Vec<i32>>(),
      1e-12,
      1e-12,
    );
  }
}
