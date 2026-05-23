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
//! Per `docs/core-arch-simd-candidates.md` §5.3, split kernels into:
//!
//! - **[`assert_eq_over_lane_sweep`]** — the `Exact` class. Used for
//!   data-movement / lossless-widening kernels:
//!     - **C1**  ([#146](https://github.com/Findit-AI/mlxrs/issues/146)) integer arms of PCM decode-widen,
//!     - **C3**  ([#148](https://github.com/Findit-AI/mlxrs/issues/148)) u8→f32 RGB widen,
//!     - **C4**  ([#149](https://github.com/Findit-AI/mlxrs/issues/149)) BGR R↔B swap widen,
//!     - **C6**  ([#151](https://github.com/Findit-AI/mlxrs/issues/151)) `pad_to_square` fill,
//!     - **C5**  ([#150](https://github.com/Findit-AI/mlxrs/issues/150)) `rotate_buf` permutation.
//!
//!   The SIMD output **must be bit-identical** to scalar; the helper
//!   asserts `scalar_out == simd_out`. For `f32`/`f64` outputs you
//!   typically want bit-level equality via `to_bits()` instead — see
//!   the worked `dot` differential test in [`crate::simd`] for the
//!   bit-exact reduction-tree contract.
//!
//! - **[`assert_close_over_lane_sweep`]** — the `Tolerance` class.
//!   Used for fp-reduction / FMA-rounding kernels where SIMD changes
//!   summation order or rounding (in general — note the in-tree
//!   `dot`/`sum_of_squares` are deliberately bit-identical because the
//!   scalar tree mirrors NEON's, but the generic `Tolerance` shape is
//!   the right shape for *new* fp reductions where that property has
//!   not been engineered):
//!     - **C2**  ([#147](https://github.com/Findit-AI/mlxrs/issues/147)) loudness sum-of-squares,
//!     - any future fp reduction without a matched reduction tree.
//!
//!   The helper asserts
//!   `(scalar - simd).abs() <= abs.max(rel * scalar.abs())`.
//!
//! # Length sweep
//!
//! Both helpers call the input generator at lengths
//! `{0, 1, lanes-1, lanes, lanes+1, 3*lanes+1}` so the kernel's
//! head / body / tail paths are *all* exercised (§5.3 — "Cover the
//! tail / remainder"). `lanes` is parameterized at the call site (e.g.
//! `lanes = 2` for `float64x2_t`-based kernels, `lanes = 4` for
//! `float32x4_t`, `lanes = 16` for `uint8x16_t`).
//!
//! # Signature shape
//!
//! Both helpers take:
//! 1. the scalar reference function pointer (`fn(&[T]) -> R`),
//! 2. the public dispatcher function pointer (`fn(&[T]) -> R`) —
//!    internally selects NEON or scalar,
//! 3. an input generator (`fn(usize) -> Vec<T>` — receives the length
//!    and returns a deterministic input of that length).
//!
//! The generator returning a fresh `Vec` for each length is
//! deliberate: it keeps the helper trivially `Send` / `Sync`-free
//! (helpful for `--test-threads=1` runs) and lets the call site decide
//! the data distribution (random-seeded, alternating-sign, edge values
//! like `i16::MIN`/`MAX`, etc.).

/// Lane-sweep length set. See the module doc for rationale.
///
/// Returns inputs at length `0`, `1`, `lanes - 1`, `lanes`,
/// `lanes + 1`, and `3 * lanes + 1` — the 6 lengths the helpers
/// [`assert_eq_over_lane_sweep`] and [`assert_close_over_lane_sweep`]
/// drive their generator at. The `lanes - 1` case `saturating_sub`s
/// to `0` when `lanes <= 1` (re-running an earlier length is harmless).
///
/// Exposed as `pub` so a test that needs a non-standard shape (e.g. a
/// kernel taking two slices `dot(a, b)`) can build its own loop using
/// the same length set, instead of reimplementing the sweep.
#[inline]
pub fn lane_sweep_lengths(lanes: usize) -> [usize; 6] {
  let l = lanes;
  // `saturating_sub` keeps `lanes <= 1` from underflowing.
  [0, 1, l.saturating_sub(1), l, l + 1, 3 * l + 1]
}

/// `Exact` differential-test class — asserts the SIMD dispatcher's
/// output equals the scalar reference output, **bit-for-bit** (`==`)
/// at every length in [`lane_sweep_lengths`].
///
/// Use for data-movement / lossless-widening kernels — the SIMD output
/// must be bit-identical to scalar (no rounding, no FMA, no
/// re-association). See the module doc for the kernel-class catalog
/// (C1 integer arms, C3, C4, C5, C6).
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
/// Use for fp-reduction / FMA-rounding kernels (C2 loudness
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

#[cfg(test)]
mod tests {
  //! Self-tests of the differential helpers, using the in-tree `dot`
  //! dispatcher (which already ships scalar + NEON) as the SIMD pair.
  //!
  //! These tests **do not** validate `dot` itself — that is the
  //! responsibility of the per-kernel differential tests in
  //! [`crate::simd`]. They validate that the helper drives a passing
  //! test across the lane-sweep set for both correctness classes.
  use super::{assert_close_over_lane_sweep, assert_eq_over_lane_sweep, lane_sweep_lengths};

  /// `lane_sweep_lengths` produces the documented 6-length set for a
  /// representative NEON lane width (`lanes = 4` for `float32x4_t`),
  /// and handles the `lanes <= 1` underflow case via
  /// `saturating_sub`.
  #[test]
  fn lane_sweep_lengths_shape() {
    assert_eq!(lane_sweep_lengths(4), [0, 1, 3, 4, 5, 13]);
    assert_eq!(lane_sweep_lengths(2), [0, 1, 1, 2, 3, 7]); // dot's `float64x2_t` lane width.
    assert_eq!(lane_sweep_lengths(16), [0, 1, 15, 16, 17, 49]); // `uint8x16_t`.
    assert_eq!(lane_sweep_lengths(1), [0, 1, 0, 1, 2, 4]); // edge: saturating_sub kicks in.
    assert_eq!(lane_sweep_lengths(0), [0, 1, 0, 0, 1, 1]); // edge: lanes=0 still well-defined.
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
}
