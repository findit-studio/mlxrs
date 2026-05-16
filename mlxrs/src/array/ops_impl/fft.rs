//! Method-form FFT bridges.

use crate::{array::Array, error::Result, ops::fft::FftNorm};

impl Array {
  /// 1-D FFT along `axis`. See [`crate::ops::fft::fft`].
  pub fn fft(&self, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
    crate::ops::fft::fft(self, n, axis, norm)
  }

  /// 1-D inverse FFT along `axis`. See [`crate::ops::fft::ifft`].
  pub fn ifft(&self, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
    crate::ops::fft::ifft(self, n, axis, norm)
  }

  /// 1-D real-input FFT along `axis`. See [`crate::ops::fft::rfft`].
  pub fn rfft(&self, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
    crate::ops::fft::rfft(self, n, axis, norm)
  }

  /// 1-D inverse real-input FFT along `axis`. See [`crate::ops::fft::irfft`].
  pub fn irfft(&self, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
    crate::ops::fft::irfft(self, n, axis, norm)
  }

  /// N-D FFT. See [`crate::ops::fft::fftn`].
  pub fn fftn(&self, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
    crate::ops::fft::fftn(self, n, axes, norm)
  }

  /// N-D inverse FFT. See [`crate::ops::fft::ifftn`].
  pub fn ifftn(&self, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
    crate::ops::fft::ifftn(self, n, axes, norm)
  }

  /// 2-D FFT. See [`crate::ops::fft::fft2`].
  pub fn fft2(&self, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
    crate::ops::fft::fft2(self, n, axes, norm)
  }

  /// 2-D inverse FFT. See [`crate::ops::fft::ifft2`].
  pub fn ifft2(&self, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
    crate::ops::fft::ifft2(self, n, axes, norm)
  }

  /// N-D real-input FFT. See [`crate::ops::fft::rfftn`].
  pub fn rfftn(&self, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
    crate::ops::fft::rfftn(self, n, axes, norm)
  }

  /// N-D inverse real-input FFT. See [`crate::ops::fft::irfftn`].
  pub fn irfftn(&self, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
    crate::ops::fft::irfftn(self, n, axes, norm)
  }

  /// 2-D real-input FFT. See [`crate::ops::fft::rfft2`].
  pub fn rfft2(&self, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
    crate::ops::fft::rfft2(self, n, axes, norm)
  }

  /// 2-D inverse real-input FFT. See [`crate::ops::fft::irfft2`].
  pub fn irfft2(&self, n: &[i32], axes: &[i32], norm: FftNorm) -> Result<Array> {
    crate::ops::fft::irfft2(self, n, axes, norm)
  }

  /// Shift zero-frequency component to center. See [`crate::ops::fft::fftshift`].
  pub fn fftshift(&self, axes: &[i32]) -> Result<Array> {
    crate::ops::fft::fftshift(self, axes)
  }

  /// Inverse of `fftshift`. See [`crate::ops::fft::ifftshift`].
  pub fn ifftshift(&self, axes: &[i32]) -> Result<Array> {
    crate::ops::fft::ifftshift(self, axes)
  }
}
