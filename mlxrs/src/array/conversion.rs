//! Array introspection: shape, dtype, scalar/buffer extraction.

use std::ffi::CStr;

use crate::{
  array::Array,
  dtype::{Dtype, Element},
  error::{Error, Result},
};

impl Array {
  /// Number of dimensions.
  pub fn ndim(&self) -> usize {
    // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
    // it, and the call returns a plain scalar (no out-param, no rc).
    unsafe { mlxrs_sys::mlx_array_ndim(self.0) }
  }

  /// Total number of elements.
  pub fn size(&self) -> usize {
    // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
    // it, and the call returns a plain scalar (no out-param, no rc).
    unsafe { mlxrs_sys::mlx_array_size(self.0) }
  }

  /// Element type.
  pub fn dtype(&self) -> Result<Dtype> {
    // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
    // it, and the call returns a plain scalar (no out-param, no rc).
    Dtype::try_from(unsafe { mlxrs_sys::mlx_array_dtype(self.0) })
  }

  /// Shape as a `Vec<usize>`.
  pub fn shape(&self) -> Vec<usize> {
    let n = self.ndim();
    (0..n)
      // SAFETY: pure read of a valid borrowed handle for `0 <= i < ndim`; mlx-c does
      // not mutate or retain the handle and returns a plain scalar.
      .map(|i| unsafe { mlxrs_sys::mlx_array_dim(self.0, i as std::ffi::c_int) as usize })
      .collect()
  }

  /// Scalar extraction. Implicitly evaluates the array (mlx requires the
  /// underlying buffer to be materialized for data access), which is why the
  /// signature is `&mut self` — the eval mutates non-atomic
  /// `array_desc->status` and would race a shared `&Array` (see
  /// `array/mod.rs` `!Sync` rationale).
  ///
  /// **CORE-2 audit (#118).** This is the `&mut self` accessor that exercises
  /// the lazy→materialized transition. The [`Array::try_item`] parallel
  /// relaxes the borrow to `&self` (useful when the caller holds an `&Array`)
  /// but does **not** enforce the strict no-implicit-eval contract from
  /// `feedback_no_implicit_eval` — see the `try_item` doc for the audit
  /// finding and the binding work that would be needed to enforce it.
  pub fn item<T: Element>(&mut self) -> Result<T> {
    let actual = self.dtype()?;
    if actual != T::DTYPE {
      return Err(Error::DtypeMismatch {
        expected: T::DTYPE,
        got: actual,
      });
    }
    self.eval()?;
    // SAFETY: `self.0` was evaluated (`self.eval()` above) and its dtype verified
    // `== T::DTYPE` above, satisfying `Element::item`'s # Safety contract.
    unsafe { T::item(self.0) }
  }

  /// Materialize the underlying buffer as `Vec<T>`. Forces eval. Errors with
  /// `Error::NonContiguous` if the array is strided/broadcast: `mlx_array_size`
  /// (logical element count) can exceed the contiguous storage reachable from
  /// the data pointer for views, so reading `size` elements would read past
  /// the allocation. M2 will add `.contiguous()` to materialize strided views.
  ///
  /// **CORE-2 audit (#118).** No `try_to_vec(&self)` parallel is provided:
  /// `mlx_array_data_*` segfaults on an unscheduled array (see the
  /// [`Array::try_item`] doc for the C++ status-check gap). A safe
  /// borrow-relaxed variant requires either a binding for the internal
  /// `_mlx_array_is_available` or an upstream mlx-c entry point that routes
  /// through C++ const overloads — both out of scope for this polish PR.
  pub fn to_vec<T: Element>(&mut self) -> Result<Vec<T>> {
    let actual = self.dtype()?;
    if actual != T::DTYPE {
      return Err(Error::DtypeMismatch {
        expected: T::DTYPE,
        got: actual,
      });
    }
    self.eval()?;
    if !is_row_contiguous(self.0) {
      return Err(Error::NonContiguous);
    }
    // SAFETY: array materialized by the prior `eval()`, dtype verified `== T::DTYPE`
    // and row-contiguity checked above; the NULL/zero-length case is guarded
    // before this call, so `(ptr, len)` is a valid non-null slice.
    unsafe {
      let (ptr, len) = T::data(self.0);
      // Zero-element arrays (shape `[0]`, `[2,0]`, ...) yield NULL from mlx;
      // `from_raw_parts(NULL, 0)` is UB per Rust's slice contract, so return
      // an empty Vec without touching the pointer.
      if len == 0 {
        return Ok(Vec::new());
      }
      assert!(!ptr.is_null(), "mlx data pointer NULL after eval");
      Ok(std::slice::from_raw_parts(ptr, len).to_vec())
    }
  }

  /// Borrow the underlying buffer as `&[T]`. Forces eval. Errors with
  /// `Error::NonContiguous` if the array is strided (post-transpose, etc.).
  ///
  /// **CORE-2 audit (#118).** Same caveat as [`Array::to_vec`]: no
  /// `try_as_slice(&self)` parallel — `mlx_array_data_*` is not safe on
  /// unscheduled arrays. See [`Array::try_item`] doc.
  pub fn as_slice<T: Element>(&mut self) -> Result<&[T]> {
    let actual = self.dtype()?;
    if actual != T::DTYPE {
      return Err(Error::DtypeMismatch {
        expected: T::DTYPE,
        got: actual,
      });
    }
    self.eval()?;
    if !is_row_contiguous(self.0) {
      return Err(Error::NonContiguous);
    }
    // SAFETY: array materialized by the prior `eval()`, dtype verified `== T::DTYPE`
    // and row-contiguity checked above; the NULL/zero-length case is guarded
    // before this call, so `(ptr, len)` is a valid non-null slice.
    unsafe {
      let (ptr, len) = T::data(self.0);
      // Same zero-element guard as `to_vec`: NULL data ptr is legitimate
      // when `len == 0`, and `from_raw_parts(NULL, 0)` is still UB.
      if len == 0 {
        return Ok(&[]);
      }
      assert!(!ptr.is_null(), "mlx data pointer NULL after eval");
      Ok(std::slice::from_raw_parts(ptr, len))
    }
  }

  /// `&self` scalar extraction — borrow-relaxation parallel of
  /// [`Array::item`]. Lets the caller read a scalar through a shared `&Array`
  /// reference (the canonical motivation of `feedback_no_implicit_eval`:
  /// reading shouldn't require an `&mut` borrow).
  ///
  /// ```ignore
  /// a.eval()?;                       // explicit eval (recommended pattern)
  /// let v: f32 = a.try_item()?;      // works through `&Array`
  /// ```
  ///
  /// ## CORE-2 audit finding (#118): no-implicit-eval not enforceable here
  ///
  /// The strict "error if unscheduled" contract from
  /// `feedback_no_implicit_eval` cannot be honored under the current
  /// mlx-c binding set: `mlx_array_item_*` internally dispatches to the C++
  /// **non-const** `array::item()` overload (because
  /// `mlx-c/mlx/c/private/array.h:42` exposes `array&`, not `const array&`,
  /// so overload resolution picks the non-const overload at
  /// `vendor/mlx/mlx/array.h:574-579`). That overload calls `eval()`
  /// unconditionally and does NOT check status — only the `const` overload
  /// at line 574-585 does. So a `try_item` call on an unscheduled array
  /// still triggers an implicit eval inside mlx-c. The same is true for
  /// `mlx_array_data_*` (which would back a `try_to_vec`/`try_as_slice`),
  /// but worse: that path **segfaults** on unscheduled rather than
  /// implicitly evaluating, because C++ `array::data<T>() const`
  /// (`vendor/mlx/mlx/array.h:379-381`) `const_cast`s to the non-checking
  /// non-const variant which dereferences `array_desc_->data->buffer`
  /// (which is null when unscheduled). The mlx-c header comment
  /// "Array must be evaluated, otherwise returns NULL"
  /// (`vendor/mlx-c/mlx/c/array.h:309`) does not reflect the actual C++
  /// behavior.
  ///
  /// **Enforcing the strict contract** therefore requires either:
  ///   1. Allowlisting and binding `_mlx_array_is_available` (currently
  ///      excluded by the `xtask` bindgen allowlist `mlx_.*` because of the
  ///      underscore prefix) so we can pre-check status from Rust;
  ///   2. Upstream mlx-c adding `mlx_array_item_*_const` /
  ///      `mlx_array_data_*_const` entry points that route through the C++
  ///      const overloads.
  ///
  /// Both are out of scope for a polish PR. `try_item` is shipped now as
  /// **just the borrow-relaxation** — the `&self` signature lets callers pass
  /// `&Array` (no `&mut`) and is sound on a single thread because `Array` is
  /// `!Sync` (no cross-thread shared `&Array` is possible). The "no implicit
  /// eval" guarantee is a follow-up that needs the binding work above.
  ///
  /// `try_to_vec` / `try_as_slice` are deliberately NOT added in this PR —
  /// they would have the same borrow-relaxation value but with a SEGV
  /// failure mode on unscheduled input, which is strictly worse than the
  /// current `&mut self` accessors' "force the caller to materialize first"
  /// guarantee.
  ///
  /// ## Errors
  /// - `Error::DtypeMismatch` if `T::DTYPE != self.dtype()`.
  /// - `Error::Backend` if mlx's `item` throws (e.g. `size() != 1`).
  pub fn try_item<T: Element>(&self) -> Result<T> {
    // CRITICAL: must be the first call in this function. If removed,
    // a stripped-ctor environment (where the process-global mlx error handler
    // wasn't installed by #[ctor]) would cause mlx-c's default handler to
    // exit(-1) on the first FFI failure here, instead of returning Err.
    // See issue #215 for the structural-test spiral history. Covered by
    // the runtime regression test
    // `stripped_ctor_try_item::try_item_survives_stripped_ctor_environment`
    // (issue #223): it spawns a child with `MLXRS_DISABLE_CTOR_FOR_TEST=1`
    // (suppressing the eager `#[ctor]` install) and calls `try_item` on a
    // non-scalar; removing this `ensure_handler_installed()` reproducibly
    // flips the child's exit code from 0 (Err returned) to non-zero (mlx-c
    // `exit(-1)` aborted before `check()` could observe the rc).
    //
    // Defense-in-depth handler install, identical to `Array::eval` and the
    // constructors. `try_item` is a public safe entry point that can call
    // `mlx_array_item_*`, which may throw (non-scalar arrays, eval failure,
    // OOM); the rc-pattern `check()` in `Element::item` assumes the handler
    // is installed first, otherwise mlx-c's default handler can `exit(-1)`
    // before Rust observes the error. Required because `try_item` is
    // reachable on an `Array` constructed via `from_raw` without any prior
    // `mlxrs` constructor / `eval` having run on this thread (the ctor
    // install is process-global but the `INIT_VIA_CTOR` flag may be false
    // if the static-constructor entry was stripped, e.g. an `objcopy`-d or
    // dlopen'd build that disables `__attribute__((constructor))`).
    crate::error::ensure_handler_installed();
    let actual = self.dtype()?;
    if actual != T::DTYPE {
      return Err(Error::DtypeMismatch {
        expected: T::DTYPE,
        got: actual,
      });
    }
    // Cleared-thread poison guard, identical to `Array::eval` — mlx-c's
    // `item` reaches the backend and triggers eval internally (see the
    // audit-finding doc above); without this guard, a `try_item` on a
    // cleared-stream thread would fail cryptically inside mlx instead of
    // panicking immediately.
    crate::stream::assert_streams_not_cleared();
    // SAFETY: dtype verified `== T::DTYPE` above.
    //
    // **Contract reconciliation (#118 R1).** `Element::item`'s documented
    // `# Safety` precondition is "`arr` must be evaluated and have dtype
    // `DTYPE`" (see `dtype::Element::item`). `try_item` intentionally
    // relaxes the "must be evaluated" half — and that relaxation is sound
    // *only* under an impl-specific guarantee, not the trait contract: all
    // current `Element` impls route through `mlx_array_item_*`, which in
    // turn dispatches to the C++ non-const `array::item()` overload at
    // `vendor/mlx/mlx/array.h:574-579`, and that overload performs its own
    // internal `eval()` before reading. An unscheduled handle therefore
    // does NOT dereference a null `array_desc_->data->buffer` here — mlx-c
    // evaluates it inside the FFI call.
    //
    // **Forward-compat invariant for future `Element` implementors / refactors:**
    // any new `Element::item` impl MUST preserve the "internal-eval-on-
    // lazy" routing (use `mlx_array_item_*`, not a hypothetical
    // `*_const`/`*_strict` variant that would skip the implicit eval and
    // deref a null buffer). If that routing ever changes (e.g. mlx-c adds
    // a const overload and an impl switches to it), this call site MUST
    // add an explicit `self.eval()` *before* `T::item(self.0)` — but
    // `try_item` is `&self`, so that would require either changing the
    // signature to `&mut self` or introducing an `_mlx_array_is_available`
    // binding to check + bail (see audit-finding doc above). Until then,
    // the soundness of `try_item` over a lazy array is anchored by the
    // doc + the `try_item_currently_implicitly_evaluates_lazy_graph`
    // regression in `tests/array_explicit_eval.rs`.
    //
    // Soundness of the `&self` signature itself relies on `Array: !Sync`
    // preventing any concurrent `&Array` from another thread (the
    // `mlx::core::array_desc->status` write inside the FFI eval is
    // non-atomic, see `array/mod.rs` `!Sync` rationale).
    unsafe { T::item(self.0) }
  }
}

/// Compute row-major contiguity from shape + strides. mlx-c does not expose
/// `mlx_array_is_contiguous` directly, so we replicate the standard check:
/// for each dim from innermost to outermost, the stride must equal the running
/// product of trailing dims. Dims of size 1 are skipped (any stride is fine).
fn is_row_contiguous(arr: mlxrs_sys::mlx_array) -> bool {
  // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
  // it, and the call returns a plain scalar (no out-param, no rc).
  let ndim = unsafe { mlxrs_sys::mlx_array_ndim(arr) };
  if ndim == 0 {
    return true;
  }
  // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
  // it, and the call returns a plain scalar (no out-param, no rc).
  let shape_ptr = unsafe { mlxrs_sys::mlx_array_shape(arr) };
  // SAFETY: pure read of a valid borrowed handle; mlx-c does not mutate or retain
  // it, and the call returns a plain scalar (no out-param, no rc).
  let strides_ptr = unsafe { mlxrs_sys::mlx_array_strides(arr) };
  if shape_ptr.is_null() || strides_ptr.is_null() {
    return false;
  }
  // SAFETY: `arr` is a valid borrowed handle and `ndim > 0` was checked above; the
  // shape/strides pointers were NULL-checked, and mlx-c guarantees each
  // spans `ndim` elements, so the `(ptr, ndim)` slice is in bounds.
  let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim) };
  // SAFETY: `arr` is a valid borrowed handle and `ndim > 0` was checked above; the
  // shape/strides pointers were NULL-checked, and mlx-c guarantees each
  // spans `ndim` elements, so the `(ptr, ndim)` slice is in bounds.
  let strides = unsafe { std::slice::from_raw_parts(strides_ptr, ndim) };
  let mut expected: usize = 1;
  for i in (0..ndim).rev() {
    let dim = shape[i] as usize;
    if dim == 1 {
      continue;
    }
    if strides[i] != expected {
      return false;
    }
    expected = expected.saturating_mul(dim);
  }
  true
}

impl std::fmt::Debug for Array {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let shape = self.shape();
    let dtype = self.dtype().ok();
    write!(f, "Array(shape={shape:?}, dtype={dtype:?})")
  }
}

/// RAII guard for a temporary `mlx_string` handle (e.g. the Display buffer).
struct StringGuard(mlxrs_sys::mlx_string);
impl Drop for StringGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. Runs during `Drop` /
    // thread teardown: must not touch TLS, call `check()`, panic, or unwind
    // across `extern "C"`; the rc is discarded silently per the crate's
    // Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_string_free(self.0);
    }
  }
}

impl std::fmt::Display for Array {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    crate::error::ensure_handler_installed();
    // mlx_array_tostring → upstream `operator<<(ostream, array)` calls
    // `a.eval()` before printing, so Display re-enters eval. It must honor
    // the cleared-thread poison guard like Array::eval does, otherwise
    // formatting a lazy array on a recycled-cleared worker silently
    // degrades to `Array(<tostring failed>)` instead of failing fast.
    // (Debug only reads shape/dtype metadata — no eval — so it is not
    // guarded; panicking in Debug during a debugger session is hostile.)
    crate::stream::assert_streams_not_cleared();
    // SAFETY: `mlx_string_new()` returns a fresh empty out-param `mlx_string`
    // (NULL ctx) per the mlx-c convention; populated by the following call
    // and freed via the local guard / explicit `mlx_string_free`.
    let mut s = StringGuard(unsafe { mlxrs_sys::mlx_string_new() });
    // SAFETY: `self.0` is a valid borrowed handle; `s` is a fresh `mlx_string`
    // out-param freed via the local guard/explicit free; mlx-c writes the
    // formatted string into it and the rc is surfaced (checked below).
    let rc = unsafe { mlxrs_sys::mlx_array_tostring(&mut s.0, self.0) };
    if rc != 0 {
      return write!(f, "Array(<tostring failed: rc={rc}>)");
    }
    // SAFETY: `s` is a live `mlx_string` (freed only after this borrow); mlx-c
    // returns its internal NUL-terminated buffer, valid until the string is
    // freed. The returned pointer is NULL-checked before use.
    let cstr = unsafe { CStr::from_ptr(mlxrs_sys::mlx_string_data(s.0)) };
    write!(f, "{}", cstr.to_string_lossy())
  }
}
