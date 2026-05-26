//! Incremental mel spectrogram with overlap-save framing.
//!
//! Faithful port of
//! [`mlx-audio-swift/Sources/MLXAudioSTT/Streaming/IncrementalMelSpectrogram.swift`][swift-ref]:
//! computes mel spectrograms over a stream of audio chunks, carrying
//! `n_fft - hop_length` samples of overlap between successive
//! [`IncrementalMelSpectrogram::process`] calls so STFT frames spanning
//! chunk boundaries are computed correctly.
//!
//! The first chunk uses reflect padding at the start (per Whisper /
//! mlx-audio convention); subsequent chunks overlap with the tail of
//! the previous chunk. [`IncrementalMelSpectrogram::flush`] drains
//! whatever overlap remains at session end, padding with zeros + a
//! reflect-pad tail so the final partial frame is emitted.
//!
//! Output frames are log-mel features with the same running-max log
//! normalization the Swift reference uses
//! (`(log10(mel) + 4) / 4`, clamped at `runningLogMax - 8`).
//!
//! [swift-ref]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioSTT/Streaming/IncrementalMelSpectrogram.swift

use smol_str::format_smolstr;

use crate::{
  Array,
  audio::dsp::{hann_window, mel_filter_bank},
  error::{Error, InvariantViolationPayload, OutOfRangePayload, Result},
  ops::{
    arithmetic::{abs, add, divide, log10, maximum, multiply, square},
    fft::{FftNorm, rfft},
    linalg_basic::matmul,
    reduction,
    shape::as_strided,
  },
};

/// Incremental mel-spectrogram extractor — feed audio chunks, get mel
/// frames out, plus a [`flush`](Self::flush) at end-of-stream.
///
/// Maintains a rolling buffer of `n_fft - hop_length` samples between
/// calls. Sample rate, FFT size, hop, and mel-bin count are fixed at
/// construction.
#[derive(Debug)]
pub struct IncrementalMelSpectrogram {
  n_fft: usize,
  hop_length: usize,
  /// Overlap kept between chunks (`n_fft - hop_length`).
  overlap_size: usize,

  /// Pre-computed Hann window of length `n_fft`.
  window: Array,
  /// Pre-computed mel filterbank, shape `(n_mels, n_freqs)`.
  filters: Array,

  /// Rolling buffer of leftover samples from the previous chunk.
  overlap_buffer: Vec<f32>,
  /// `true` until the first non-empty chunk is processed.
  is_first_chunk: bool,
  /// Running maximum of the log-mel-spectrogram values (monotonic).
  running_log_max: f32,
  /// Total mel frames produced since construction / `reset`.
  total_frames: usize,

  /// Test-only error-injection counter: while `> 0`, the next
  /// [`flush`](Self::flush) returns [`Error::Backend`] and decrements
  /// the counter. Used by retry-state regression tests to script a
  /// recoverable `flush()` failure — there is no real-world Err the
  /// pure-MLX compute pipeline reliably surfaces, so a deterministic
  /// counter is the only way to exercise the cross-call retry path.
  ///
  /// Crucially, the injection fires BEFORE `overlap_buffer` is
  /// touched, so the transactional contract (overlap preserved on Err)
  /// holds for the injected error path too.
  #[cfg(test)]
  pub(crate) flush_err_inject_count: usize,
}

impl IncrementalMelSpectrogram {
  /// Build a new extractor with the given parameters.
  ///
  /// Defaults are the Whisper preset: `n_fft = 400`, `hop_length = 160`,
  /// `n_mels = 128`, `sample_rate = 16000`. Mirrors the Swift
  /// reference's `init` default arguments.
  ///
  /// # Errors
  /// [`Error::Backend`] if `n_fft` is zero, odd, or `n_fft <
  /// hop_length` (cannot maintain the overlap-save invariant); or if
  /// the underlying [`hann_window`] / [`mel_filter_bank`] construction
  /// errors. Mel-filterbank parameter validation (e.g. `sample_rate >
  /// 0`, `n_mels > 0`) is delegated to [`mel_filter_bank`].
  pub fn new(sample_rate: u32, n_fft: usize, hop_length: usize, n_mels: usize) -> Result<Self> {
    if n_fft == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "IncrementalMelSpectrogram::new: n_fft",
        "must be > 0",
      )));
    }
    if !n_fft.is_multiple_of(2) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "IncrementalMelSpectrogram::new: n_fft",
        "must be even (odd n_fft is unsupported because the one-sided rfft is not invertible)",
        format_smolstr!("{n_fft}"),
      )));
    }
    if hop_length == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "IncrementalMelSpectrogram::new: hop_length",
        "must be > 0",
      )));
    }
    if hop_length > n_fft {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "IncrementalMelSpectrogram::new: hop_length",
        "must be <= n_fft (overlap-save framing requires `n_fft - hop_length >= 0`)",
        format_smolstr!("hop_length={hop_length}, n_fft={n_fft}"),
      )));
    }
    let window = hann_window(n_fft)?;
    let filters = mel_filter_bank(n_mels, n_fft, sample_rate, 0.0, None)?;
    Ok(Self {
      n_fft,
      hop_length,
      overlap_size: n_fft - hop_length,
      window,
      filters,
      overlap_buffer: Vec::new(),
      is_first_chunk: true,
      running_log_max: f32::NEG_INFINITY,
      total_frames: 0,
      #[cfg(test)]
      flush_err_inject_count: 0,
    })
  }

  /// Total mel frames produced since construction / last
  /// [`reset`](Self::reset).
  pub fn total_frames(&self) -> usize {
    self.total_frames
  }

  /// Length of the rolling overlap buffer — exposed for in-crate tests
  /// that need to assert the transactional `flush` contract (overlap
  /// preserved on Err, cleared on Ok).
  #[doc(hidden)]
  #[cfg(test)]
  pub(crate) fn overlap_buffer_len(&self) -> usize {
    self.overlap_buffer.len()
  }

  /// Process new audio samples and return mel frames, shape
  /// `(num_frames, n_mels)`.
  ///
  /// Returns `Ok(None)` when fewer than `n_fft` samples have
  /// accumulated (overlap + new samples); they're held in the rolling
  /// buffer for the next call. Returns `Ok(Some(arr))` with the new
  /// mel frames otherwise.
  ///
  /// # Errors
  /// Propagates from the underlying STFT / matmul / log10 ops, or
  /// from [`Array::from_slice`] when the input signal cannot be
  /// materialized.
  pub fn process(&mut self, samples: &[f32]) -> Result<Option<Array>> {
    if samples.is_empty() {
      return Ok(None);
    }

    let signal = self.build_signal_for_chunk(samples);

    // How many complete frames we can compute.
    let num_frames = if signal.len() >= self.n_fft {
      (signal.len() - self.n_fft) / self.hop_length + 1
    } else {
      0
    };

    if num_frames == 0 {
      // Not enough samples yet — save everything as overlap.
      self.overlap_buffer = signal;
      return Ok(None);
    }

    // Save leftover samples for next chunk: keep the last
    // `overlap_size` samples of whatever the frames consumed.
    let consumed = (num_frames - 1) * self.hop_length + self.n_fft;
    self.overlap_buffer = if consumed < signal.len() {
      // Keep `overlap_size` ending at `consumed` (= n_fft - hop_length
      // last samples of the last frame), then append the trailing
      // post-`consumed` samples for the next chunk's framing.
      let start = consumed.saturating_sub(self.overlap_size);
      signal[start..].to_vec()
    } else {
      // Frames consumed the whole signal (or more) — keep the tail
      // overlap exactly.
      let tail_start = signal.len().saturating_sub(self.overlap_size);
      signal[tail_start..].to_vec()
    };

    let mel = self.compute_mel(&signal, num_frames)?;
    self.total_frames += num_frames;
    Ok(Some(mel))
  }

  /// Drain any remaining samples at session end. Pads with zeros to
  /// fill the last frame if needed + appends reflect padding at the
  /// tail (matching the Swift reference + the offline STFT's
  /// `reflect` pad convention).
  ///
  /// `flush` is **transactional**: the overlap buffer is cloned into a
  /// local stage, the fallible mel computation runs against the stage,
  /// and only after `compute_mel` returns `Ok` is `self.overlap_buffer`
  /// cleared. On `Err` the buffer is preserved so a retry recomputes
  /// the same `mel_frames` from the same input.
  ///
  /// # Errors
  /// Same propagation as [`process`](Self::process).
  pub fn flush(&mut self) -> Result<Option<Array>> {
    // Test-only error injection — fires BEFORE we touch
    // `overlap_buffer` so the transactional contract holds (overlap
    // preserved on Err → retry sees identical input).
    #[cfg(test)]
    if self.flush_err_inject_count > 0 {
      self.flush_err_inject_count -= 1;
      return Err(Error::Backend(
        "IncrementalMelSpectrogram::flush: scripted test injection".into(),
      ));
    }

    if self.overlap_buffer.is_empty() {
      return Ok(None);
    }

    // Stage: clone the overlap buffer into a local. self.overlap_buffer
    // stays intact so a fallible compute_mel below can be retried.
    let mut signal: Vec<f32> = self.overlap_buffer.clone();

    // Pad with zeros so we have at least `n_fft` samples.
    if signal.len() < self.n_fft {
      signal.resize(self.n_fft, 0.0);
    }

    // Reflect padding at the tail (n_fft / 2 samples).
    let pad_size = self.n_fft / 2;
    let signal_len = signal.len();
    let reflect_len = pad_size.min(signal_len.saturating_sub(1));
    if reflect_len > 0 {
      let lower = signal_len - 1 - reflect_len;
      let upper = signal_len - 1;
      let mut suffix: Vec<f32> = signal[lower..upper].iter().copied().rev().collect();
      signal.append(&mut suffix);
    }

    let num_frames = if signal.len() >= self.n_fft {
      (signal.len() - self.n_fft) / self.hop_length + 1
    } else {
      0
    };
    if num_frames == 0 {
      // No fallible work happens — commit the clear immediately so a
      // double-flush on an unconsumable tail doesn't re-enter the work.
      self.overlap_buffer.clear();
      return Ok(None);
    }
    // FALLIBLE: any Err below MUST leave self.overlap_buffer intact
    // for retry. Runs against the staged clone, not self.*.
    let mel = self.compute_mel(&signal, num_frames)?;
    // COMMIT: compute_mel succeeded — only now clear the source.
    self.overlap_buffer.clear();
    self.total_frames += num_frames;
    Ok(Some(mel))
  }

  /// Reset all internal state for a new session.
  pub fn reset(&mut self) {
    self.overlap_buffer.clear();
    self.is_first_chunk = true;
    self.running_log_max = f32::NEG_INFINITY;
    self.total_frames = 0;
  }

  // -------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------

  /// Build the chunk's signal — first chunk uses reflect-padding at
  /// the start, subsequent chunks prepend the rolling overlap.
  fn build_signal_for_chunk(&mut self, samples: &[f32]) -> Vec<f32> {
    if self.is_first_chunk {
      // Reflect padding at the start: copy `samples[1..=reflect_len]`
      // in reverse, where `reflect_len = min(n_fft / 2, samples.len() - 1)`.
      let pad_size = self.n_fft / 2;
      let mut prefix: Vec<f32> = Vec::with_capacity(pad_size);
      if samples.len() > 1 {
        let reflect_len = pad_size.min(samples.len() - 1);
        if reflect_len > 0 {
          prefix.extend(samples[1..=reflect_len].iter().rev().copied());
        }
      }
      if prefix.is_empty() {
        let fill = samples.first().copied().unwrap_or(0.0);
        prefix.resize(pad_size, fill);
      } else if prefix.len() < pad_size {
        // Samples shorter than pad_size — repeat the reflected prefix.
        while prefix.len() < pad_size {
          let needed = pad_size - prefix.len();
          let snapshot: Vec<f32> = prefix.iter().copied().take(needed).collect();
          prefix.extend(snapshot);
        }
      }
      let mut signal = prefix;
      signal.extend_from_slice(samples);
      self.is_first_chunk = false;
      signal
    } else {
      let mut signal = std::mem::take(&mut self.overlap_buffer);
      signal.extend_from_slice(samples);
      signal
    }
  }

  /// Compute the mel features for the given signal + frame count, then
  /// apply the running-log-max normalization in-place.
  fn compute_mel(&mut self, signal: &[f32], num_frames: usize) -> Result<Array> {
    let n_fft_i32 = i32::try_from(self.n_fft)
      .map_err(|_| Error::Backend("IncrementalMelSpectrogram: n_fft does not fit i32".into()))?;
    let num_frames_i32 = i32::try_from(num_frames).map_err(|_| {
      Error::Backend("IncrementalMelSpectrogram: num_frames does not fit i32".into())
    })?;

    let signal_array = Array::from_slice::<f32>(signal, &[signal.len() as i32])?;
    // SAFETY: `(num_frames, n_fft)` frame view: stride `(hop_length, 1)`
    // and offset `0`. Frame `i` covers element range `[i*hop, i*hop +
    // n_fft)`. The last frame ends at `(num_frames-1)*hop + n_fft`
    // which is `<= signal.len()` by the `num_frames` derivation above.
    let frames_stacked = unsafe {
      as_strided(
        &signal_array,
        &[num_frames_i32, n_fft_i32],
        &[self.hop_length as i64, 1],
        0,
      )?
    };

    let windowed = multiply(&frames_stacked, &self.window)?;
    let fft = rfft(&windowed, n_fft_i32, 1, FftNorm::Backward)?;
    let magnitudes = square(&abs(&fft)?)?;
    // `power` is `(num_frames, n_freqs)`. Mel layout used here mirrors
    // the Swift reference: `(num_frames, n_mels)` = `power @ filters.T`.
    // mlxrs's `mel_filter_bank` returns `(n_mels, n_freqs)` already, so
    // transpose it once to `(n_freqs, n_mels)` for the matmul.
    let filters_t = self.filters.transpose()?;
    let mut mel = matmul(&magnitudes, &filters_t)?;

    let floor = Array::from_slice::<f32>(&[1e-10], &[1i32])?;
    mel = maximum(&mel, &floor)?;
    mel = log10(&mel)?;

    // Update running log-max using a scalar item from the reduction.
    let mut chunk_max_arr = reduction::max(&mel, false)?;
    let chunk_max: f32 = chunk_max_arr.item::<f32>()?;
    if chunk_max > self.running_log_max {
      self.running_log_max = chunk_max;
    }

    let floor_log = self.running_log_max - 8.0;
    let floor_log_arr = Array::from_slice::<f32>(&[floor_log], &[1i32])?;
    mel = maximum(&mel, &floor_log_arr)?;

    let four = Array::from_slice::<f32>(&[4.0_f32], &[1i32])?;
    mel = divide(&add(&mel, &four)?, &four)?;
    Ok(mel)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Test fixture: a small but realistic configuration.
  fn make_extractor() -> IncrementalMelSpectrogram {
    IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap()
  }

  #[test]
  fn new_rejects_zero_n_fft() {
    let err = IncrementalMelSpectrogram::new(16_000, 0, 16, 8).unwrap_err();
    assert!(matches!(err, Error::InvariantViolation(ref p)
        if p.context().contains("n_fft") && p.requirement().contains("must be > 0")));
  }

  #[test]
  fn new_rejects_odd_n_fft() {
    let err = IncrementalMelSpectrogram::new(16_000, 33, 16, 8).unwrap_err();
    assert!(matches!(err, Error::OutOfRange(ref p)
        if p.context().contains("n_fft") && p.requirement().contains("must be even")));
  }

  #[test]
  fn new_rejects_zero_hop_length() {
    let err = IncrementalMelSpectrogram::new(16_000, 32, 0, 8).unwrap_err();
    assert!(matches!(err, Error::InvariantViolation(ref p)
        if p.context().contains("hop_length") && p.requirement().contains("must be > 0")));
  }

  #[test]
  fn new_rejects_hop_larger_than_n_fft() {
    let err = IncrementalMelSpectrogram::new(16_000, 32, 64, 8).unwrap_err();
    assert!(matches!(err, Error::OutOfRange(ref p)
        if p.context().contains("hop_length") && p.requirement().contains("<= n_fft")));
  }

  #[test]
  fn process_empty_input_returns_none() {
    let mut mel = make_extractor();
    let out = mel.process(&[]).unwrap();
    assert!(out.is_none());
    assert_eq!(mel.total_frames(), 0);
  }

  #[test]
  fn process_emits_mel_frames_with_expected_shape() {
    let mut mel = make_extractor();
    // 128 samples: with n_fft=32, hop=16, reflect-pad = 16, signal = 144,
    // num_frames = (144 - 32) / 16 + 1 = 8.
    let samples: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut out = mel.process(&samples).unwrap().expect("expected frames");
    let shape = out.shape();
    assert_eq!(shape.len(), 2, "expected 2-D output, got shape {shape:?}");
    assert_eq!(shape[1], 8, "n_mels axis should be 8, got {shape:?}");
    assert!(shape[0] > 0, "num_frames axis must be > 0, got {shape:?}");
    assert_eq!(mel.total_frames(), shape[0]);
    // Output should be finite values (no NaN / inf from the log
    // normalization path).
    let vals = out.to_vec::<f32>().unwrap();
    for v in &vals {
      assert!(v.is_finite(), "non-finite mel value: {v}");
    }
  }

  #[test]
  fn streaming_then_flush_consumes_all_samples_deterministically() {
    // Feed in two chunks + flush.
    let mut mel = make_extractor();
    let samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.005).sin()).collect();

    let _ = mel.process(&samples[..128]).unwrap();
    let _ = mel.process(&samples[128..]).unwrap();
    let after = mel.total_frames();

    let flushed = mel.flush().unwrap();
    let final_total = mel.total_frames();
    if let Some(_arr) = flushed {
      assert!(final_total > after, "flush should emit at least 1 frame");
    } else {
      assert_eq!(final_total, after);
    }
  }

  #[test]
  fn reset_clears_state_so_second_session_starts_fresh() {
    let mut mel = make_extractor();
    let samples: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin()).collect();
    let _ = mel.process(&samples).unwrap();
    let total_before_reset = mel.total_frames();
    assert!(total_before_reset > 0);

    mel.reset();
    assert_eq!(mel.total_frames(), 0);
    // Running max + first-chunk flag reset.
    let _ = mel.process(&samples).unwrap();
    let total_after_reset = mel.total_frames();
    // Re-processing the same samples after reset yields the same frame
    // count (deterministic re-emission).
    assert_eq!(total_after_reset, total_before_reset);
  }
}
