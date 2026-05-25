//! FFT ops: forward/inverse 1-D, 2-D, N-D, real-input variants, and shifts.
//!
//! All FFT ops accept an [`FftNorm`] strategy. The default in mlx-python is
//! `FftNorm::Backward` (no scaling on the forward, `1/N` on the inverse). The
//! one-axis ops also accept an `n` length (for zero-pad/truncate to a target
//! transform length) and an `axis` index.
//!
//! Multi-axis ops (`fft2`, `fftn`, etc.) take parallel `n` and `axes`
//! slices. Passing **empty** slices selects mlx-python's defaults, resolved
//! in this layer exactly like `mlx-swift` (`Source/MLX/FFT.swift`): empty
//! `axes` → all dims (the last two for the `*2` variants); empty `n` →
//! the size of each transformed axis. mlx-c only binds the explicit-axes
//! overload (which returns the input unchanged for empty axes), so the
//! default is materialized here rather than forwarded as empty.
//!
//! See [mlx FFT docs](https://ml-explore.github.io/mlx/build/html/python/fft.html).

use std::ffi::c_int;

use crate::{
  array::Array,
  error::{Result, check},
  shape::dim_ptr,
  stream::default_stream,
};

/// Normalization mode for FFT ops. Mirrors `mlx.core.fft`'s `norm=` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FftNorm {
  /// No scaling on forward, `1/N` on inverse. Matches numpy/mlx-python default.
  #[default]
  Backward,
  /// `1/sqrt(N)` on both forward and inverse (unitary FFT).
  Ortho,
  /// `1/N` on forward, no scaling on inverse.
  Forward,
}

impl From<FftNorm> for mlxrs_sys::mlx_fft_norm {
  fn from(n: FftNorm) -> Self {
    match n {
      FftNorm::Backward => mlxrs_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD,
      FftNorm::Ortho => mlxrs_sys::mlx_fft_norm__MLX_FFT_NORM_ORTHO,
      FftNorm::Forward => mlxrs_sys::mlx_fft_norm__MLX_FFT_NORM_FORWARD,
    }
  }
}

/// Resolve `(n, axes)` for the multi-axis FFTs the way mlx-python and
/// `mlx-swift` (`Source/MLX/FFT.swift`) do, so callers can pass empty
/// slices for the documented defaults instead of hitting mlx-c's
/// explicit-overload no-op:
/// - both given: forward as-is.
/// - axes given, `n` empty: `n` = size of each transformed axis.
/// - `n` given, axes empty: `axes` = the rightmost `n.len()` dims.
/// - both empty: `axes` = all dims (the last two for the `*2` variants),
///   `n` = their sizes.
///
/// Returns owned vectors — the FFI needs concrete arrays, and the official
/// bindings likewise materialize the default axes/n here. An out-of-range
/// explicit axis is forwarded unchanged so mlx-c emits its own precise
/// error instead of this panicking.
fn resolve_fft(a: &Array, n: &[i32], axes: &[i32], last_two: bool) -> (Vec<i32>, Vec<i32>) {
  let ndim = a.ndim() as i32;
  let norm = |ax: i32| if ax < 0 { ax + ndim } else { ax };
  if !axes.is_empty() {
    if n.is_empty() {
      let shape = a.shape();
      let ok = axes.iter().all(|&ax| {
        let r = norm(ax);
        r >= 0 && (r as usize) < shape.len()
      });
      if ok {
        let nn = axes
          .iter()
          .map(|&ax| shape[norm(ax) as usize] as i32)
          .collect();
        return (nn, axes.to_vec());
      }
    }
    return (n.to_vec(), axes.to_vec());
  }
  if !n.is_empty() {
    let cnt = n.len() as i32;
    return (n.to_vec(), ((ndim - cnt).max(0)..ndim).collect());
  }
  let ax: Vec<i32> = if last_two {
    ((ndim - 2).max(0)..ndim).collect()
  } else {
    (0..ndim).collect()
  };
  let shape = a.shape();
  let nn = ax
    .iter()
    .map(|&x| shape.get(x as usize).copied().unwrap_or(0) as i32)
    .collect();
  (nn, ax)
}

/// 1-D discrete Fourier transform along `axis`. `n` is the transform length
/// (zero-pad or truncate `a` along `axis` to this length before transforming).
/// Output dtype is Complex64.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fft.html).
pub fn fft(a: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_fft(
      &mut out.0,
      a.0,
      n as c_int,
      axis as c_int,
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 1-D inverse discrete Fourier transform along `axis`. See [`fft`] for the
/// semantics of `n` and `norm`. Output dtype is Complex64.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifft.html).
pub fn ifft(a: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_ifft(
      &mut out.0,
      a.0,
      n as c_int,
      axis as c_int,
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 1-D real-input FFT along `axis`. Input is real-valued; output is complex
/// with the redundant negative-frequency half dropped (length `n/2 + 1`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfft.html).
pub fn rfft(a: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_rfft(
      &mut out.0,
      a.0,
      n as c_int,
      axis as c_int,
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 1-D inverse of [`rfft`]: complex-valued one-sided spectrum -> real signal.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfft.html).
pub fn irfft(a: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_irfft(
      &mut out.0,
      a.0,
      n as c_int,
      axis as c_int,
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// N-D FFT. Empty `axes` ⇒ all dims; empty `n` ⇒ each transformed axis's
/// size (see module docs / `resolve_fft`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fftn.html).
pub fn fftn(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let (n, axes) = resolve_fft(a, n, axes, false);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_fftn(
      &mut out.0,
      a.0,
      dim_ptr(&n),
      n.len(),
      dim_ptr(&axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// N-D inverse of [`fftn`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifftn.html).
pub fn ifftn(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let (n, axes) = resolve_fft(a, n, axes, false);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_ifftn(
      &mut out.0,
      a.0,
      dim_ptr(&n),
      n.len(),
      dim_ptr(&axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D FFT. Defaults to the last two axes (empty `axes`); empty `n` ⇒
/// those axes' sizes.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fft2.html).
pub fn fft2(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let (n, axes) = resolve_fft(a, n, axes, true);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_fft2(
      &mut out.0,
      a.0,
      dim_ptr(&n),
      n.len(),
      dim_ptr(&axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D inverse of [`fft2`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifft2.html).
pub fn ifft2(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let (n, axes) = resolve_fft(a, n, axes, true);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_ifft2(
      &mut out.0,
      a.0,
      dim_ptr(&n),
      n.len(),
      dim_ptr(&axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// N-D real-input FFT.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfftn.html).
pub fn rfftn(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let (n, axes) = resolve_fft(a, n, axes, false);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_rfftn(
      &mut out.0,
      a.0,
      dim_ptr(&n),
      n.len(),
      dim_ptr(&axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// N-D inverse of [`rfftn`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfftn.html).
pub fn irfftn(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let (n, axes) = resolve_fft(a, n, axes, false);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_irfftn(
      &mut out.0,
      a.0,
      dim_ptr(&n),
      n.len(),
      dim_ptr(&axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D real-input FFT. Defaults to the last two axes (empty `axes`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfft2.html).
pub fn rfft2(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let (n, axes) = resolve_fft(a, n, axes, true);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_rfft2(
      &mut out.0,
      a.0,
      dim_ptr(&n),
      n.len(),
      dim_ptr(&axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D inverse of [`rfft2`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfft2.html).
pub fn irfft2(a: &Array, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
  let (n, axes) = resolve_fft(a, n, axes, true);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_irfft2(
      &mut out.0,
      a.0,
      dim_ptr(&n),
      n.len(),
      dim_ptr(&axes),
      axes.len(),
      norm.into(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Sample frequencies for [`fft`] of length `n` and sample spacing `d`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fftfreq.html).
pub fn fftfreq(n: i32, d: f64) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_fft_fftfreq(&mut out.0, n as c_int, d, default_stream()) })?;
  Ok(out)
}

/// Sample frequencies for [`rfft`] of length `n` and sample spacing `d`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.rfftfreq.html).
pub fn rfftfreq(n: i32, d: f64) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_fft_rfftfreq(&mut out.0, n as c_int, d, default_stream()) })?;
  Ok(out)
}

/// Shift the zero-frequency component to the center. Empty `axes` shifts
/// all dims (mlx-python default), expanded here like `mlx-swift`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.fftshift.html).
pub fn fftshift(a: &Array, axes: &[i32]) -> Result<Array> {
  let (_, axes) = resolve_fft(a, &[], axes, false);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_fftshift(
      &mut out.0,
      a.0,
      dim_ptr(&axes),
      axes.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Inverse of [`fftshift`].
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.ifftshift.html).
pub fn ifftshift(a: &Array, axes: &[i32]) -> Result<Array> {
  let (_, axes) = resolve_fft(a, &[], axes, false);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fft_ifftshift(
      &mut out.0,
      a.0,
      dim_ptr(&axes),
      axes.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}
