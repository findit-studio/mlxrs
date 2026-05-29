//! `IntoShape` trait — zero-allocation shape conversion via slice callback.
//!
//! Returns `Result` so we propagate `usize > i32::MAX` as `Error::ShapeMismatch`
//! instead of silently saturating.
//!
//! `IntoShape` is **sealed**: downstream crates cannot implement it. The
//! single-source-of-truth validation lives at the FFI boundary (callers use
//! [`validate_dims`] inside their `with_shape` callback before passing the
//! slice to mlx-c). Sealing prevents a downstream impl from supplying a
//! callback slice that bypasses the call-site check.

use std::ffi::c_int;

use smol_str::format_smolstr;

use crate::error::{Error, OutOfRangePayload, Result};

mod sealed {
  pub trait Sealed {}
  impl Sealed for &[i32] {}
  impl<const N: usize> Sealed for [i32; N] {}
  impl Sealed for &[usize] {}
  impl Sealed for Vec<i32> {}
  impl Sealed for Vec<usize> {}
  impl Sealed for (usize,) {}
  impl Sealed for (usize, usize) {}
  impl Sealed for (usize, usize, usize) {}
  impl Sealed for (usize, usize, usize, usize) {}
  impl Sealed for (usize, usize, usize, usize, usize) {}
  impl Sealed for (usize, usize, usize, usize, usize, usize) {}
  impl Sealed for (usize, usize, usize, usize, usize, usize, usize) {}
  impl Sealed for (usize, usize, usize, usize, usize, usize, usize, usize) {}
}

/// Types that can supply an mlx-c-shaped `&[c_int]` to a callback without
/// heap allocation (for ranks ≤ 8). Sealed; not implementable downstream.
///
/// # Supported shape sources
///
/// - **Slices:** `&[i32]` (passed through verbatim) and `&[usize]` (each dim
///   range-checked into `i32`). Any rank.
/// - **Owned vectors:** `Vec<i32>` and `Vec<usize>`, with the same semantics
///   as the corresponding slice (#257 M6). Any rank.
/// - **Arrays:** `[i32; N]` for any const `N`.
/// - **Tuples:** `(usize,)` through the 8-tuple
///   `(usize, usize, usize, usize, usize, usize, usize, usize)` — i.e. ranks
///   **1 through 8** (#257 M5). 8 is the max because the zero-alloc `&[usize]`
///   path uses a fixed `[i32; 8]` stack buffer; ranks above 8 are expressible
///   via a slice/`Vec` (which spills to the heap past 8).
pub trait IntoShape: sealed::Sealed {
  /// Invoke `f` with the shape as `&[c_int]`. The caller MUST run
  /// [`validate_dims`] on the slice before any unsafe FFI use, even though
  /// the built-in impls validate eagerly — sealing is defense in depth, not
  /// a substitute for the boundary check.
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R>;
}

/// FFI-boundary safety check. Reject negative dimensions before any unsafe
/// mlx-c call. A negative `i32` silently sign-extends when cast to `usize`
/// (`-1i32 as usize == usize::MAX`), so an unchecked dim crossing into
/// mlx-c can produce malformed allocation/copy behavior or invalid array
/// metadata. Every safe-layer entry point that consumes a shape slice MUST
/// call this on the slice produced by `IntoShape::with_shape` before
/// passing to FFI.
pub fn validate_dims(s: &[c_int]) -> Result<()> {
  for (i, &d) in s.iter().enumerate() {
    if d < 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "shape::validate_dims: dim",
        "must be non-negative",
        format_smolstr!("dim[{i}]={d}"),
      )));
    }
  }
  Ok(())
}

/// Static sentinel for FFI calls that would otherwise pass a Rust dangling
/// pointer (the well-defined `NonNull::dangling()` returned by `<&[i32]>::as_ptr`
/// on an empty slice). The C++ side calls `std::vector<int>(p, p + n)`; while
/// `p + 0` is well-defined for any pointer in C++17+, constructing a vector
/// from a "singular" iterator (one not associated with any allocation) is
/// strictly UB per `[iterator.requirements.general]`. Routing empty-len cases
/// through a real static i32 keeps the iterator non-singular without changing
/// observed behavior — the value is never read because `n == 0`.
static EMPTY_DIM_SENTINEL: c_int = 0;

/// Returns `s.as_ptr()` for non-empty slices; otherwise a non-singular pointer
/// into [`EMPTY_DIM_SENTINEL`]. Use at every FFI call site that passes a
/// `(*const c_int, len)` pair to mlx-c.
#[inline]
pub(crate) fn dim_ptr(s: &[c_int]) -> *const c_int {
  if s.is_empty() {
    &EMPTY_DIM_SENTINEL as *const c_int
  } else {
    s.as_ptr()
  }
}

/// `i64` sibling of [`EMPTY_DIM_SENTINEL`] for stride slices. Same rationale:
/// avoids the singular dangling pointer that `<&[i64]>::as_ptr` returns on an
/// empty slice when the FFI feeds it into a C++ `std::vector<int64_t>(p, p + n)`
/// constructor.
static EMPTY_STRIDE_SENTINEL: i64 = 0;

/// `i64` sibling of [`dim_ptr`] for stride slices passed to mlx-c via a
/// `(*const i64, len)` pair (e.g. `mlx_as_strided`).
#[inline]
pub(crate) fn stride_ptr(s: &[i64]) -> *const i64 {
  if s.is_empty() {
    &EMPTY_STRIDE_SENTINEL as *const i64
  } else {
    s.as_ptr()
  }
}

impl IntoShape for &[i32] {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    f(self)
  }
}

impl<const N: usize> IntoShape for [i32; N] {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    f(&self[..])
  }
}

fn convert_dim(d: usize) -> Result<c_int> {
  i32::try_from(d).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "shape::convert_dim",
      "must fit in i32",
      format_smolstr!("{d}"),
    ))
  })
}

impl IntoShape for &[usize] {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    if self.len() <= 8 {
      let mut buf = [0i32; 8];
      for (i, &d) in self.iter().enumerate() {
        buf[i] = convert_dim(d)?;
      }
      f(&buf[..self.len()])
    } else {
      let v: Vec<c_int> = self
        .iter()
        .map(|&d| convert_dim(d))
        .collect::<Result<_>>()?;
      f(&v)
    }
  }
}

// Owned-vector siblings of the slice impls (#257 M6). Each forwards to the
// corresponding `&[_]` impl so the conversion/validation lives in one place.
impl IntoShape for Vec<i32> {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    // `self.as_slice(): &[i32]`; method-call autoref hands the `&[i32]` impl
    // its `&&[i32]` receiver.
    self.as_slice().with_shape(f)
  }
}

impl IntoShape for Vec<usize> {
  fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
    self.as_slice().with_shape(f)
  }
}

/// Generates an `IntoShape` impl for a `usize` tuple of a given arity (#257
/// M5). Each `$idx` is the tuple field index; the body builds a fixed-size
/// `[c_int; N]` on the stack (zero-alloc) by range-checking every dim through
/// [`convert_dim`], then invokes the callback. Covers ranks 1 through 8 below.
macro_rules! tuple_into_shape {
  ($($T:ty),+ => $($idx:tt),+) => {
    impl IntoShape for ($($T,)+) {
      fn with_shape<R>(&self, f: impl FnOnce(&[c_int]) -> Result<R>) -> Result<R> {
        let s = [$(convert_dim(self.$idx)?),+];
        f(&s)
      }
    }
  };
}

tuple_into_shape!(usize => 0);
tuple_into_shape!(usize, usize => 0, 1);
tuple_into_shape!(usize, usize, usize => 0, 1, 2);
tuple_into_shape!(usize, usize, usize, usize => 0, 1, 2, 3);
tuple_into_shape!(usize, usize, usize, usize, usize => 0, 1, 2, 3, 4);
tuple_into_shape!(usize, usize, usize, usize, usize, usize => 0, 1, 2, 3, 4, 5);
tuple_into_shape!(usize, usize, usize, usize, usize, usize, usize => 0, 1, 2, 3, 4, 5, 6);
tuple_into_shape!(usize, usize, usize, usize, usize, usize, usize, usize => 0, 1, 2, 3, 4, 5, 6, 7);

#[cfg(test)]
mod tests {
  use super::*;

  /// Capture the `&[c_int]` the impl hands to the callback.
  fn collect(s: &impl IntoShape) -> Vec<c_int> {
    s.with_shape(|dims| Ok(dims.to_vec())).expect("with_shape")
  }

  #[test]
  fn tuple_ranks_1_through_8() {
    // #257 M5: tuple `IntoShape` now covers ranks 1..=8.
    assert_eq!(collect(&(1usize,)), vec![1]);
    assert_eq!(collect(&(1usize, 2)), vec![1, 2]);
    assert_eq!(collect(&(1usize, 2, 3)), vec![1, 2, 3]);
    assert_eq!(collect(&(1usize, 2, 3, 4)), vec![1, 2, 3, 4]);
    assert_eq!(collect(&(1usize, 2, 3, 4, 5)), vec![1, 2, 3, 4, 5]);
    assert_eq!(collect(&(1usize, 2, 3, 4, 5, 6)), vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(
      collect(&(1usize, 2, 3, 4, 5, 6, 7)),
      vec![1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
      collect(&(1usize, 2, 3, 4, 5, 6, 7, 8)),
      vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
  }

  #[test]
  fn vec_usize_and_i32_match_slices() {
    // #257 M6: owned-vector `IntoShape` impls.
    let vu: Vec<usize> = vec![2, 3, 4];
    assert_eq!(collect(&vu), vec![2, 3, 4]);
    let vi: Vec<i32> = vec![5, 6];
    assert_eq!(collect(&vi), vec![5, 6]);
    // Empty vec → empty dims (rank-0 / scalar shape).
    assert_eq!(collect(&Vec::<usize>::new()), Vec::<c_int>::new());
    assert_eq!(collect(&Vec::<i32>::new()), Vec::<c_int>::new());
  }

  #[test]
  fn vec_usize_matches_equivalent_slice() {
    let vu: Vec<usize> = vec![7, 8, 9, 10, 11];
    let su: &[usize] = &[7, 8, 9, 10, 11];
    assert_eq!(collect(&vu), collect(&su));
  }

  #[test]
  fn slice_usize_rank_above_8_spills_to_heap_path() {
    // 9 dims exceeds the fixed [i32; 8] stack buffer → the Vec branch.
    let dims: Vec<usize> = (1..=9).collect();
    assert_eq!(collect(&dims), (1..=9).collect::<Vec<c_int>>());
  }

  #[test]
  fn tuple_dim_overflowing_i32_is_rejected() {
    // usize > i32::MAX must surface OutOfRange, not silently saturate.
    let big = (i32::MAX as usize) + 1;
    let err = (big,).with_shape(|_| Ok(())).unwrap_err();
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn vec_dim_overflowing_i32_is_rejected() {
    let big = (i32::MAX as usize) + 1;
    let v: Vec<usize> = vec![1, big];
    let err = v.with_shape(|_| Ok(())).unwrap_err();
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn validate_dims_rejects_negative() {
    assert!(validate_dims(&[1, 2, 3]).is_ok());
    let err = validate_dims(&[1, -2, 3]).unwrap_err();
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }
}
