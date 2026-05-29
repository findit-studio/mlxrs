//! Pseudo-random ops using mlx's split-key (JAX-style) PRNG.
//!
//! Every sampling op takes a `key: &Array` returned by [`key`] (or split off
//! from one via [`split`] / [`split_num`]). Re-using a key produces identical
//! output by design — split before each draw.
//!
//! Shape-taking ops follow the canonical `dim_ptr(s)` pattern: empty shape is
//! routed through the static sentinel rather than the Rust dangling pointer
//! returned from `<&[i32]>::as_ptr` for empty slices.
//!
//! See [mlx random docs](https://ml-explore.github.io/mlx/build/html/python/random.html).

use std::{cell::Cell, ffi::c_int};

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Result, check},
  shape::{IntoShape, dim_ptr, validate_dims},
  stream::default_stream,
};

thread_local! {
  static CPU_STREAM: Cell<Option<mlxrs_sys::mlx_stream>> = const { Cell::new(None) };
}

/// Per-thread CPU stream for random ops that fall through to a CPU-only linalg
/// kernel inside mlx (e.g. `multivariate_normal` calls SVD on the covariance,
/// which mlx-c rejects on the GPU). Mirrors [`crate::ops::linalg_full`]'s
/// per-thread CPU stream pattern.
fn random_cpu_stream() -> mlxrs_sys::mlx_stream {
  crate::error::ensure_handler_installed();
  // Honor the #13 cleared-thread poison contract (as `default_stream()` /
  // `Stream::default_cpu()` do): a CPU-routed op on a poisoned thread must
  // fail fast, not continue into mlx with torn-down stream state.
  crate::stream::assert_streams_not_cleared();
  CPU_STREAM.with(|cell| {
    if let Some(s) = cell.get() {
      return s;
    }
    // SAFETY: `mlx_default_cpu_stream_new()` returns the thread's default CPU stream
    // handle; the error handler is installed first and the NULL-ctx case is
    // checked by the caller before the handle is cached/used.
    let s = unsafe { mlxrs_sys::mlx_default_cpu_stream_new() };
    if s.ctx.is_null() {
      panic!(
        "mlxrs::ops::random: mlx_default_cpu_stream_new returned NULL ctx — \
         CPU stream initialization failed. Aborting."
      );
    }
    cell.set(Some(s));
    s
  })
}

/// Construct a PRNG key from `seed`. Returns a `U32[2]` key array.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.key.html).
pub fn key(seed: u64) -> Result<Array> {
  // No `default_stream()` on this path, so the eager-#[ctor] handler is not
  // guaranteed installed; install it before the fallible mlx-c call (else a
  // stripped ctor → mlx-c's default `printf+exit(-1)` aborts from safe Rust).
  crate::error::ensure_handler_installed();
  // Honor the #13 cleared-thread poison contract (as `random_cpu_stream()` /
  // `linalg_cpu_stream()` do): a safe op on a poisoned thread must fail fast,
  // not enter mlx-c with torn-down stream state.
  crate::stream::assert_streams_not_cleared();
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_random_key(&mut out.0, seed) })?;
  Ok(out)
}

/// Set the global RNG seed. Subsequent ops that elide their `key` argument
/// (none in this Rust API — always pass an explicit key) read from this state.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.seed.html).
pub fn seed(seed: u64) -> Result<()> {
  // Same as `key`: no stream on this path — install the handler first, then
  // honor the #13 cleared-thread poison contract before entering mlx-c.
  crate::error::ensure_handler_installed();
  crate::stream::assert_streams_not_cleared();
  // SAFETY: `mlx_random_seed` takes a scalar `seed` by value (no handles, no
  // out-param); it mutates only backend-global RNG state and the backend rc
  // is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_random_seed(seed) })
}

/// Split `key` into two independent subkeys (commonly used as `(key, subkey)`).
/// The output is a `U32[2, 2]` array; the first row is the new key, the second
/// is the subkey for sampling.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.split.html).
pub fn split(key: &Array) -> Result<(Array, Array)> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut k0 = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut k1 = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_random_split(&mut k0.0, &mut k1.0, key.0, default_stream()) })?;
  Ok((k0, k1))
}

/// Split `key` into `num` independent subkeys. Returns a `U32[num, 2]` array
/// (each row is a subkey).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.split.html).
pub fn split_num(key: &Array, num: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_random_split_num(&mut out.0, key.0, num as c_int, default_stream())
  })?;
  Ok(out)
}

/// Bernoulli draws with per-element probability `p` (broadcast to `shape`).
/// Output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.bernoulli.html).
pub fn bernoulli(p: &Array, shape: &impl IntoShape, key: &Array) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_bernoulli(
        &mut out.0,
        p.0,
        dim_ptr(s),
        s.len(),
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Uniform draws in `[low, high)` of the given `shape` and `dtype`. `low` and
/// `high` are arrays (broadcast to `shape`).
///
/// mlx does **not** validate `low <= high`. With `low > high` the computed
/// range `high - low` is negative, so samples fall in the reversed half-open
/// interval `(high, low]` instead of `[low, high)` (no error is raised). This
/// is upstream behavior, preserved here; see [`randint`] for why no value-based
/// guard is added (it would force an implicit `eval` on the borrowed bounds).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.uniform.html).
pub fn uniform(
  low: &Array,
  high: &Array,
  shape: &impl IntoShape,
  dtype: Dtype,
  key: &Array,
) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_uniform(
        &mut out.0,
        low.0,
        high.0,
        dim_ptr(s),
        s.len(),
        mlxrs_sys::mlx_dtype::from(dtype),
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Normal (Gaussian) draws with mean `loc` and standard deviation `scale` of
/// the given `shape` and `dtype`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.normal.html).
pub fn normal(
  shape: &impl IntoShape,
  dtype: Dtype,
  loc: f32,
  scale: f32,
  key: &Array,
) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_normal(
        &mut out.0,
        dim_ptr(s),
        s.len(),
        mlxrs_sys::mlx_dtype::from(dtype),
        loc,
        scale,
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Normal draws with broadcast `loc` and `scale` arrays.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.normal.html).
pub fn normal_broadcast(
  shape: &impl IntoShape,
  dtype: Dtype,
  loc: &Array,
  scale: &Array,
  key: &Array,
) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_normal_broadcast(
        &mut out.0,
        dim_ptr(s),
        s.len(),
        mlxrs_sys::mlx_dtype::from(dtype),
        loc.0,
        scale.0,
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Random integer draws in `[low, high)` of the given `shape` and integer
/// `dtype`. `low` and `high` are arrays (broadcast to `shape`).
///
/// # Inverted bounds (`low > high`)
///
/// mlx does **not** validate that `low <= high`. Internally `randint` is
/// `astype(maximum(uniform(low, high), low), dtype)` (`mlx/mlx/random.cpp`):
/// when `low > high` the underlying `uniform` draws from the reversed/empty
/// range `(high, low]` and the `maximum(·, low)` then clamps every sample up to
/// `low`, so the result is silently a constant array of `low` rather than an
/// error. This is upstream behavior (mlx-core / mlx-python both forward without
/// a guard), faithfully preserved here.
///
/// This wrapper deliberately does **not** add a value-based `low <= high` guard:
/// `low`/`high` are borrowed (`&Array`) and may be lazy/unevaluated, so reading
/// them to compare would force an implicit `eval` — a behavior change that
/// violates the crate's "no implicit eval on `&self`" contract. Callers that
/// need strict bounds should validate their scalar `low`/`high` before the call.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.randint.html).
pub fn randint(
  low: &Array,
  high: &Array,
  shape: &impl IntoShape,
  dtype: Dtype,
  key: &Array,
) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_randint(
        &mut out.0,
        low.0,
        high.0,
        dim_ptr(s),
        s.len(),
        mlxrs_sys::mlx_dtype::from(dtype),
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Sample one categorical index per logit row along `axis`. Output dtype is U32.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.categorical.html).
pub fn categorical(logits: &Array, axis: i32, key: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_random_categorical(&mut out.0, logits.0, axis as c_int, key.0, default_stream())
  })?;
  Ok(out)
}

/// Sample categorical indices into a custom output `shape`. Output dtype is U32.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.categorical.html).
pub fn categorical_shape(
  logits: &Array,
  axis: i32,
  shape: &impl IntoShape,
  key: &Array,
) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_categorical_shape(
        &mut out.0,
        logits.0,
        axis as c_int,
        dim_ptr(s),
        s.len(),
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Sample `num_samples` categorical indices per logit row along `axis`. Output
/// dtype is U32.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.categorical.html).
pub fn categorical_num_samples(
  logits: &Array,
  axis: i32,
  num_samples: i32,
  key: &Array,
) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_random_categorical_num_samples(
      &mut out.0,
      logits.0,
      axis as c_int,
      num_samples as c_int,
      key.0,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Standard Gumbel draws of the given `shape` and `dtype`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.gumbel.html).
pub fn gumbel(shape: &impl IntoShape, dtype: Dtype, key: &Array) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_gumbel(
        &mut out.0,
        dim_ptr(s),
        s.len(),
        mlxrs_sys::mlx_dtype::from(dtype),
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Truncated normal draws restricted to `[lower, upper]` (broadcast arrays).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.truncated_normal.html).
pub fn truncated_normal(
  lower: &Array,
  upper: &Array,
  shape: &impl IntoShape,
  dtype: Dtype,
  key: &Array,
) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_truncated_normal(
        &mut out.0,
        lower.0,
        upper.0,
        dim_ptr(s),
        s.len(),
        mlxrs_sys::mlx_dtype::from(dtype),
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Multivariate normal draws with the given `mean` (shape `[..., k]`) and
/// covariance `cov` (shape `[..., k, k]`). Runs on the per-thread CPU stream
/// because mlx implements `multivariate_normal` via SVD on the covariance,
/// which is not yet supported on the Metal GPU backend.
///
/// # Empty covariance (a zero-length last-two dimension)
///
/// Because mlx computes the covariance square-root via `linalg::svd(cov, ...)`
/// (`mlx/mlx/random.cpp`), a covariance with a zero-sized trailing matrix
/// dimension (`0×0`, etc.) hits the same SVD-kernel divide-by-zero as
/// [`crate::ops::linalg_full::svd`]: mlx's `multivariate_normal` only checks
/// `cov.ndim() < 2` and that `cov` is square, both of which a `0×0` cov passes.
/// This safe wrapper rejects such a `cov` first with a recoverable
/// [`crate::error::Error::EmptyInput`] (a cheap shape check, no `eval`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.multivariate_normal.html).
pub fn multivariate_normal(
  mean: &Array,
  cov: &Array,
  shape: &impl IntoShape,
  dtype: Dtype,
  key: &Array,
) -> Result<Array> {
  // Guard the SVD divide-by-zero on the covariance: mlx forwards `cov` to
  // `linalg::svd` and only checks `ndim < 2` / squareness, so a `0×0` (or
  // `0×n` / `m×0`) `cov` would reach the kernel's `a.size() / (m * n)` (`0 / 0`,
  // UB / SIGFPE). Reuse the shared SVD-input guard before entering mlx.
  crate::ops::linalg_full::reject_empty_matrix(
    cov,
    "multivariate_normal: covariance matrix has a zero-length row or column dimension",
  )?;
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_multivariate_normal(
        &mut out.0,
        mean.0,
        cov.0,
        dim_ptr(s),
        s.len(),
        mlxrs_sys::mlx_dtype::from(dtype),
        key.0,
        random_cpu_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Laplace (double-exponential) draws with location `loc` and scale `scale`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.laplace.html).
pub fn laplace(
  shape: &impl IntoShape,
  dtype: Dtype,
  loc: f32,
  scale: f32,
  key: &Array,
) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_laplace(
        &mut out.0,
        dim_ptr(s),
        s.len(),
        mlxrs_sys::mlx_dtype::from(dtype),
        loc,
        scale,
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Raw uniform `width`-byte integer bits.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.bits.html).
pub fn bits(shape: &impl IntoShape, width: i32, key: &Array) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_random_bits(
        &mut out.0,
        dim_ptr(s),
        s.len(),
        width as c_int,
        key.0,
        default_stream(),
      )
    })?;
    Ok(out)
  })
}

/// Random permutation of `x` along `axis`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.permutation.html).
pub fn permutation(x: &Array, axis: i32, key: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_random_permutation(&mut out.0, x.0, axis as c_int, key.0, default_stream())
  })?;
  Ok(out)
}

/// Random permutation of `arange(x)`. Output dtype is U32 (mlx index-output
/// convention; matches `argmax`/`take`/`searchsorted`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.permutation.html).
pub fn permutation_arange(x: i32, key: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_random_permutation_arange(&mut out.0, x as c_int, key.0, default_stream())
  })?;
  Ok(out)
}
