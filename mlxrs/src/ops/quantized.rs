//! Quantization ops: `quantize` / `dequantize` / `quantized_matmul`
//! (plus `gather_qmm`), the thin wrappers behind MLX's grouped affine
//! quantization scheme.
//!
//! Signatures mirror `mlx.core.quantize` / `dequantize` /
//! `quantized_matmul` (python `python/mlx/nn/layers/quantized.py`) and
//! `mlx-swift`'s `Source/MLX/Quantized.swift`: `group_size` / `bits`
//! integers, a `mode` string (default `"affine"`), and an optional
//! per-tensor `global_scale`. Mode-defaults (`group_size=64, bits=4` for
//! `"affine"`) live in mlx-c; we forward `group_size` / `bits` as present
//! `mlx_optional_int`s and let mlx-c validate.
//!
//! `quantize` returns `(w_q, scales, Option<biases>)` (mlx-c packs the
//! outputs in an `mlx_vector_array`, drained here like
//! `ops::linalg_full::svd`). The `biases` output is mode-dependent: the
//! `"affine"` scheme yields a 3-array `(w_q, scales, biases)` result, while
//! the bias-less floating-point schemes (`"mxfp4"` / `"mxfp8"` / `"nvfp4"`)
//! yield a 2-array `(w_q, scales)` result — so `biases` is `Option<Array>`,
//! mirroring the optional-bias shape the sibling wrappers (`dequantize` /
//! `quantized_matmul` / `gather_qmm`) already use for their `biases` input.
//! Optional input arrays (`biases`, `global_scale`, gather indices) map to
//! `Option<&Array>`; `None` forwards a NULL-ctx `mlx_array` which mlx-c's
//! "may be null" parameters accept.
//!
//! See [mlx quantization docs](https://ml-explore.github.io/mlx/build/html/python/ops.html#quantization).

use std::ffi::CString;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Result, check},
  stream::default_stream,
};

/// Build a present `mlx_optional_int` (these ops always take explicit
/// `group_size` / `bits`; mode-defaults are resolved inside mlx-c).
#[inline]
fn opt_int(v: i32) -> mlxrs_sys::mlx_optional_int {
  mlxrs_sys::mlx_optional_int {
    value: v,
    has_value: true,
  }
}

/// Raw handle for an `Option<&Array>`: the inner handle when `Some`, or a
/// fresh NULL-ctx `mlx_array` when `None`. The returned guard owns the
/// placeholder so its `Drop` frees it; keep it alive across the FFI call.
#[inline]
fn opt_array(a: Option<&Array>) -> (mlxrs_sys::mlx_array, Option<Array>) {
  match a {
    Some(arr) => (arr.0, None),
    None => {
      // SAFETY: `mlx_array_new()` returns a fresh empty handle (NULL ctx) per the
      // mlx-c convention; it is wrapped in the RAII newtype FIRST so an early
      // return / panic frees it. It is not an out-param here: the NULL-ctx
      // placeholder is what mlx-c's "may be null" parameters accept for an
      // absent optional input, and the returned guard keeps it alive across
      // the FFI call.
      let placeholder = Array(unsafe { mlxrs_sys::mlx_array_new() });
      (placeholder.0, Some(placeholder))
    }
  }
}

/// Convert the `mode` string into a C string. An interior NUL is a caller
/// bug (mode names are short ASCII tags like `"affine"`); surface it as a
/// backend-style error rather than panicking across the FFI boundary.
#[inline]
fn mode_cstring(mode: &str) -> Result<CString> {
  CString::new(mode).map_err(|_| crate::Error::Backend {
    message: format!("mlxrs::ops::quantized: `mode` contains an interior NUL byte: {mode:?}"),
  })
}

/// Drain an `mlx_vector_array` into a `Vec<Array>`, copying out each handle.
fn drain_vector(vec: mlxrs_sys::mlx_vector_array) -> Result<Vec<Array>> {
  // SAFETY: pure read of a valid populated `mlx_vector_array`; mlx-c does not
  // mutate or retain it and returns a plain length.
  let n = unsafe { mlxrs_sys::mlx_vector_array_size(vec) };
  let mut parts = Vec::with_capacity(n);
  for i in 0..n {
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut part = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_vector_array_get(&mut part.0, vec, i) })?;
    parts.push(part);
  }
  Ok(parts)
}

/// RAII guard for a temporary `mlx_vector_array`.
struct VectorArrayGuard(mlxrs_sys::mlx_vector_array);
impl Drop for VectorArrayGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. Runs during `Drop` /
    // thread teardown: must not touch TLS, call `check()`, panic, or unwind
    // across `extern "C"`; the rc is discarded silently per the crate's
    // Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_vector_array_free(self.0);
    }
  }
}

/// Quantize the matrix `w` using `bits`-bit grouped quantization over groups
/// of `group_size` elements. `mode` selects the scheme (`"affine"` is the
/// mlx default); `global_scale` is an optional per-tensor scale.
///
/// Returns `(w_q, scales, biases)`: the packed quantized weights, the
/// per-group scales, and the per-group biases. `biases` is `Option<Array>`
/// because mlx's mode dispatch is bias-dependent — the `"affine"` scheme
/// produces a per-group `biases` (`Some`), while the bias-less
/// floating-point schemes (`"mxfp4"` / `"mxfp8"` / `"nvfp4"`) produce only
/// `(w_q, scales)` (`None`). At the pinned mlx (v0.31.2) `affine` →
/// 3-array, `mxfp4`/`mxfp8`/`nvfp4` → 2-array; any other arity is surfaced
/// as a recoverable error rather than a panic.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html).
pub fn quantize(
  w: &Array,
  group_size: i32,
  bits: i32,
  mode: &str,
  global_scale: Option<&Array>,
) -> Result<(Array, Array, Option<Array>)> {
  let mode_c = mode_cstring(mode)?;
  let (gs, _gs_guard) = opt_array(global_scale);
  // Resolve the stream FIRST — `default_stream()` runs the cleared-thread
  // poison guard (`assert_streams_not_cleared`) and installs the error
  // handler (`ensure_handler_installed`) before the fallible
  // `mlx_vector_array_new()` allocation. Mirrors `ops::linalg_full::svd`/`lu`:
  // a poisoned thread must fail fast (panic) here rather than return `Err` if
  // the subsequent alloc fails under allocator pressure. No alloc-failure
  // injection hook exists, so guard order — not a test — enforces the
  // fail-fast contract.
  let s = default_stream();
  // SAFETY: `mlx_vector_array_new()` returns a fresh empty out-param handle (NULL
  // ctx) per the mlx-c convention; the RAII guard captures it (below) before the
  // populating call so a partial / early-return vector is still freed.
  let mut vec_out = unsafe { mlxrs_sys::mlx_vector_array_new() };
  // `mlx_vector_array_new` is fallible: a null `ctx` means allocation failed
  // and an error sits in TLS. Validate (draining handler state) BEFORE the
  // guard so it only ever wraps a non-null handle (no leak / double-free).
  crate::error::check_vector_array_handle(vec_out)?;
  let _vec_guard = VectorArrayGuard(vec_out);
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it) — `gs` is either `global_scale`'s handle or the
  // NULL-ctx placeholder kept alive by `_gs_guard`, which `mlx_quantize` accepts
  // for the optional per-tensor scale; the out-param `vec_out` was freshly
  // allocated above and is written by this call; the backend rc is surfaced
  // via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_quantize(
      &mut vec_out,
      w.0,
      opt_int(group_size),
      opt_int(bits),
      mode_c.as_ptr(),
      gs,
      s,
    )
  })?;
  // mlx v0.31.2 `quantize` dispatches on `mode`: `affine` →
  // `affine_quantize` returns `{w_q, scales, biases}` (3); the bias-less
  // float schemes `mxfp4`/`mxfp8`/`nvfp4` → `fp_quantize` returns
  // `{w_q, scales}` (2). Validate the exact arity (don't index a fixed 3)
  // and surface anything else as a recoverable `Err`, not a panic.
  let mut parts = drain_vector(vec_out)?;
  let (w_q, scales, biases) = match parts.len() {
    2 => {
      // Bias-less float mode (`mxfp4`/`mxfp8`/`nvfp4`): `{w_q, scales}`.
      let scales = parts.pop().expect("len checked == 2");
      let w_q = parts.pop().expect("len checked == 2");
      (w_q, scales, None)
    }
    3 => {
      // Affine mode: `{w_q, scales, biases}`.
      let biases = parts.pop().expect("len checked == 3");
      let scales = parts.pop().expect("len checked == 3");
      let w_q = parts.pop().expect("len checked == 3");
      (w_q, scales, Some(biases))
    }
    n => return Err(unexpected_arity(n)),
  };
  Ok((w_q, scales, biases))
}

/// Inverse of [`quantize`]: reconstruct the dense matrix from the quantized
/// weights `w`, per-group `scales`, and optional `biases`. `dtype` requests
/// the output element type (default left to mlx-c).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html).
#[allow(clippy::too_many_arguments)]
pub fn dequantize(
  w: &Array,
  scales: &Array,
  biases: Option<&Array>,
  group_size: i32,
  bits: i32,
  mode: &str,
  global_scale: Option<&Array>,
  dtype: Option<Dtype>,
) -> Result<Array> {
  let mode_c = mode_cstring(mode)?;
  let (biases_h, _biases_guard) = opt_array(biases);
  let (gs, _gs_guard) = opt_array(global_scale);
  let dtype_opt = mlxrs_sys::mlx_optional_dtype {
    value: dtype
      .map(Into::into)
      .unwrap_or(mlxrs_sys::mlx_dtype__MLX_FLOAT32),
    has_value: dtype.is_some(),
  };
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it) — `biases_h` / `gs` are either the borrowed
  // optional handles or NULL-ctx placeholders kept alive by their guards, which
  // `mlx_dequantize` accepts for the optional `biases` / `global_scale`; the
  // out-param was freshly allocated above and is written by this call; the
  // backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_dequantize(
      &mut out.0,
      w.0,
      scales.0,
      biases_h,
      opt_int(group_size),
      opt_int(bits),
      mode_c.as_ptr(),
      gs,
      dtype_opt,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Matrix-multiply `x` by the quantized matrix (`w`, `scales`, optional
/// `biases`). `transpose` multiplies by `w`'s transpose (the common case for
/// quantized `Linear` layers). `group_size` / `bits` / `mode` must match the
/// quantization used to produce `w`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html).
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul(
  x: &Array,
  w: &Array,
  scales: &Array,
  biases: Option<&Array>,
  transpose: bool,
  group_size: i32,
  bits: i32,
  mode: &str,
) -> Result<Array> {
  let mode_c = mode_cstring(mode)?;
  let (biases_h, _biases_guard) = opt_array(biases);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it) — `biases_h` is either `biases`'s handle or the
  // NULL-ctx placeholder kept alive by `_biases_guard`, which
  // `mlx_quantized_matmul` accepts for the optional `biases`; the out-param was
  // freshly allocated above and is written by this call; the backend rc is
  // surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_quantized_matmul(
      &mut out.0,
      x.0,
      w.0,
      scales.0,
      biases_h,
      transpose,
      opt_int(group_size),
      opt_int(bits),
      mode_c.as_ptr(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Batched/gathered quantized matmul: like [`quantized_matmul`] but selects
/// rows of `x` / `w` via optional `lhs_indices` / `rhs_indices` (used by
/// quantized mixture-of-experts and gather-style layers). `sorted_indices`
/// promises `rhs_indices` is sorted, enabling a faster kernel.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html).
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
  x: &Array,
  w: &Array,
  scales: &Array,
  biases: Option<&Array>,
  lhs_indices: Option<&Array>,
  rhs_indices: Option<&Array>,
  transpose: bool,
  group_size: i32,
  bits: i32,
  mode: &str,
  sorted_indices: bool,
) -> Result<Array> {
  let mode_c = mode_cstring(mode)?;
  let (biases_h, _biases_guard) = opt_array(biases);
  let (lhs_h, _lhs_guard) = opt_array(lhs_indices);
  let (rhs_h, _rhs_guard) = opt_array(rhs_indices);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it) — `biases_h` / `lhs_h` / `rhs_h` are either the
  // borrowed optional handles or NULL-ctx placeholders kept alive by their
  // guards, which `mlx_gather_qmm` accepts for the optional `biases` /
  // `lhs_indices` / `rhs_indices`; the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_gather_qmm(
      &mut out.0,
      x.0,
      w.0,
      scales.0,
      biases_h,
      lhs_h,
      rhs_h,
      transpose,
      opt_int(group_size),
      opt_int(bits),
      mode_c.as_ptr(),
      sorted_indices,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Error for a `quantize` output vector whose length is neither 2
/// (bias-less float modes: `{w_q, scales}`) nor 3 (affine:
/// `{w_q, scales, biases}`) — the only arities mlx's `quantize` produces.
fn unexpected_arity(n: usize) -> crate::Error {
  crate::Error::Backend {
    message: format!(
      "mlxrs::ops::quantized::quantize: mlx_quantize returned {n} outputs; \
       expected 2 (bias-less float modes) or 3 (affine)"
    ),
  }
}
