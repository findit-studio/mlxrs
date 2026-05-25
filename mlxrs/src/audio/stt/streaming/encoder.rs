//! Streaming encoder window accumulator + per-architecture backend
//! trait.
//!
//! Faithful port of
//! [`mlx-audio-swift/Sources/MLXAudioSTT/Streaming/StreamingEncoder.swift`][swift-ref]:
//! accumulates mel frames until a full window (e.g. 800 frames =
//! ~8 s of audio at 100 fps) is ready, then encodes the window via
//! the per-model
//! [`StreamingEncoderBackend::encode_window`] hook. Consecutive
//! windows can overlap by a configurable number of mel frames.
//!
//! Per the project's [no per-model arch porting][noarch] rule, mlxrs
//! does **not** ship concrete encoder bodies (Qwen3ASR / Whisper /
//! Voxtral / Moshi / etc.). Per-architecture code implements
//! [`StreamingEncoderBackend`] and constructs a [`StreamingEncoder`]
//! over it.
//!
//! [swift-ref]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioSTT/Streaming/StreamingEncoder.swift
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

use crate::{
  Array,
  error::{Error, Result},
  ops::shape::{concatenate, pad},
};

/// Per-architecture streaming-encoder hook.
///
/// Implementations encode ONE window of mel frames (shape `(window_size,
/// n_mels)`) and return encoder hidden states (shape `(num_tokens,
/// hidden_dim)`). Block-attention encoders (the Swift `Qwen3ASR`
/// reference's `nWindowInfer = 800`) need no cross-window state, so the
/// returned tokens for each window can be concatenated independently —
/// this trait deliberately surfaces no per-call state.
///
/// `window_size` carries the encoder's mel-frame budget per window.
///
/// # Contract
///
/// [`encode_window`](Self::encode_window) is **always** called with a
/// mel-window padded to exactly `(window_size, n_mels)`. For partial
/// windows (early-feedback decode at session start + the trailing
/// short window flushed at session stop) the accumulator
/// zero-pads the buffer to `window_size` rows along axis 0 BEFORE the
/// call, and supplies `valid_frames < window_size` to signal how many
/// leading rows are real audio. Backends that need to attend only over
/// the real frames (causal masks, attention masks) MUST honor
/// `valid_frames`. Backends that don't care (block-attention with no
/// mask, the Swift `Qwen3ASR` default) can ignore it — the zero padding
/// is a no-op for their math.
pub trait StreamingEncoderBackend {
  /// Mel-frame window size — `n_window_infer` in the Swift reference
  /// (Qwen3ASR = `800`).
  fn window_size(&self) -> usize;

  /// Encode a zero-padded `(window_size, n_mels)` mel-frame window into
  /// encoder hidden states of shape `(num_tokens, hidden_dim)`.
  ///
  /// `valid_frames` (always `<= window_size`) is the number of leading
  /// rows in `mel_window` that are real audio; the remainder is
  /// zero-padding the accumulator added so the encoder always sees a
  /// uniform shape. Implementations that build an attention mask on the
  /// encoder side should use `valid_frames` to mask out the padding;
  /// implementations that don't (block-attention encoders without a
  /// mask, e.g. Qwen3ASR's default) can ignore it.
  ///
  /// # Errors
  /// Implementation-defined — propagate via [`Result`].
  fn encode_window(&self, mel_window: &Array, valid_frames: usize) -> Result<Array>;
}

/// Streaming wrapper around a [`StreamingEncoderBackend`] that
/// accumulates mel frames into complete windows + caches the encoded
/// output, with bounded cache size.
pub struct StreamingEncoder<B> {
  encoder: B,
  window_size: usize,
  window_stride: usize,
  max_cached_windows: usize,

  /// Cached encoded window outputs.
  cached_windows: Vec<Array>,
  /// Newly encoded full windows not yet consumed by the session.
  newly_encoded_windows: Vec<Array>,
  /// Total number of *completed* windows encoded since reset.
  total_encoded_windows: usize,

  /// Pending mel frames not yet forming a full window (held as a
  /// concatenated `Array`).
  pending_frames: Option<Array>,
  /// Number of pending mel frames.
  pending_frame_count: usize,
}

impl<B: StreamingEncoderBackend> StreamingEncoder<B> {
  /// Build a streaming wrapper around `encoder` with the given cache
  /// size + cross-window mel-frame overlap.
  ///
  /// `overlap_frames` is clamped to `[0, window_size - 1]` so the
  /// derived `window_stride = window_size - overlap_frames` is always
  /// `>= 1`. Matches the Swift reference's clamping semantics 1:1.
  pub fn new(encoder: B, max_cached_windows: usize, overlap_frames: usize) -> Self {
    let window_size = encoder.window_size();
    let clamped_overlap = overlap_frames.min(window_size.saturating_sub(1));
    let window_stride = window_size.saturating_sub(clamped_overlap).max(1);
    Self {
      encoder,
      window_size,
      window_stride,
      max_cached_windows,
      cached_windows: Vec::new(),
      newly_encoded_windows: Vec::new(),
      total_encoded_windows: 0,
      pending_frames: None,
      pending_frame_count: 0,
    }
  }

  /// Borrow the underlying [`StreamingEncoderBackend`] (for example to
  /// query model-specific config).
  #[inline(always)]
  pub fn backend(&self) -> &B {
    &self.encoder
  }

  /// Window size in mel frames.
  #[inline(always)]
  pub fn window_size(&self) -> usize {
    self.window_size
  }

  /// Window stride (`window_size - overlap`).
  #[inline(always)]
  pub fn window_stride(&self) -> usize {
    self.window_stride
  }

  /// Number of fully encoded windows since the last [`reset`](Self::reset).
  #[inline(always)]
  pub fn encoded_window_count(&self) -> usize {
    self.total_encoded_windows
  }

  /// Whether pending mel frames are waiting for a full window.
  #[inline(always)]
  pub fn has_pending_frames(&self) -> bool {
    self.pending_frame_count > 0
  }

  /// Total encoder tokens across all currently-cached windows.
  pub fn total_cached_tokens(&self) -> usize {
    self
      .cached_windows
      .iter()
      .map(|a| a.shape().first().copied().unwrap_or(0))
      .sum()
  }

  /// Feed mel frames to the encoder. Full windows are encoded
  /// immediately.
  ///
  /// `mel_frames` must be 2-D with shape `(num_frames, n_mels)`.
  ///
  /// `feed` is **transactional**: every fallible step (concatenate,
  /// slice, [`StreamingEncoderBackend::encode_window`], `eval`,
  /// `try_clone`) executes against local staging variables; `self.*`
  /// is mutated only after the WHOLE call succeeds. On `Err`, the
  /// caller may retry `feed` with the same `mel_frames` and observe
  /// the same starting state — no pending frames are consumed and no
  /// partial windows are committed to the cache.
  ///
  /// # Errors
  /// [`Error::Backend`] for a non-2-D input or a propagated error from
  /// the backend's [`StreamingEncoderBackend::encode_window`].
  pub fn feed(&mut self, mel_frames: &Array) -> Result<usize> {
    if mel_frames.ndim() != 2 {
      return Err(Error::Backend {
        message: format!(
          "StreamingEncoder::feed: expected 2-D mel_frames input, got {}-D",
          mel_frames.ndim()
        ),
      });
    }

    let window_size_i32 = i32::try_from(self.window_size).map_err(|_| Error::Backend {
      message: "StreamingEncoder::feed: window_size does not fit i32".into(),
    })?;
    let stride_i32 = i32::try_from(self.window_stride).map_err(|_| Error::Backend {
      message: "StreamingEncoder::feed: window_stride does not fit i32".into(),
    })?;

    // STAGE the merged pending buffer in a LOCAL. Use try_clone on the
    // existing self.pending_frames (refcount clone, no full copy)
    // instead of take so a concatenate error leaves self.pending_frames
    // intact for retry.
    let combined = match self.pending_frames.as_ref() {
      Some(existing) => concatenate(&[existing, mel_frames], 0)?,
      None => mel_frames.try_clone()?,
    };
    let mut staged_count = combined.shape().first().copied().unwrap_or(0);
    let mut staged_pending: Option<Array> = Some(combined);

    // Encode-loop output is staged in these locals — they are only
    // appended to self.cached_windows / self.newly_encoded_windows
    // AFTER the loop completes without error.
    let mut staged_cached: Vec<Array> = Vec::new();
    let mut staged_newly: Vec<Array> = Vec::new();

    while staged_count >= self.window_size {
      let frames = staged_pending
        .take()
        .expect("staged_pending was non-empty when staged_count >= window_size");
      // Take rows [0, window_size) for the encode pass — full window,
      // valid_frames == window_size, no padding needed.
      let window = frames.slice(&[0i32, 0i32], &[window_size_i32, i32::MAX], &[1i32, 1i32])?;
      let mut encoded = self.encoder.encode_window(&window, self.window_size)?;
      encoded.eval()?;
      // try_clone BEFORE pushing into both staging vecs: a clone Err
      // must not leave one Vec one-ahead of the other.
      let cached_handle = encoded.try_clone()?;
      staged_cached.push(cached_handle);
      staged_newly.push(encoded);

      // Trim staged pending to leave staged_count - window_stride rows.
      if staged_count > self.window_stride {
        let remainder = frames.slice(&[stride_i32, 0i32], &[i32::MAX, i32::MAX], &[1i32, 1i32])?;
        staged_count = remainder.shape().first().copied().unwrap_or(0);
        staged_pending = Some(remainder);
      } else {
        staged_pending = None;
        staged_count = 0;
      }
    }

    // COMMIT. Every fallible step succeeded — only now mutate self.*.
    // Append cached windows one-by-one so the max_cached_windows eviction
    // matches the prior call-order semantics (oldest dropped first).
    let new_windows = staged_cached.len();
    self.pending_frames = staged_pending;
    self.pending_frame_count = staged_count;
    for window in staged_cached {
      self.cached_windows.push(window);
      if self.cached_windows.len() > self.max_cached_windows {
        self.cached_windows.remove(0);
      }
    }
    self.newly_encoded_windows.extend(staged_newly);
    self.total_encoded_windows = self.total_encoded_windows.saturating_add(new_windows);

    Ok(new_windows)
  }

  /// Encode any partial window remaining at session end.
  /// Returns `Ok(1)` if a window was encoded, `Ok(0)` otherwise.
  ///
  /// # Errors
  /// Same as [`feed`](Self::feed).
  pub fn flush_partial(&mut self) -> Result<usize> {
    let Some(frames) = self.pending_frames.take() else {
      return Ok(0);
    };
    let valid_frames = self.pending_frame_count;
    if valid_frames == 0 {
      return Ok(0);
    }
    let padded = pad_to_window_size(&frames, valid_frames, self.window_size)?;
    let mut encoded = self.encoder.encode_window(&padded, valid_frames)?;
    encoded.eval()?;
    self.cached_windows.push(encoded);

    self.pending_frame_count = 0;
    if self.cached_windows.len() > self.max_cached_windows {
      self.cached_windows.remove(0);
    }
    Ok(1)
  }

  /// All cached encoded windows concatenated along axis 0.
  pub fn cached_encoder_output(&self) -> Result<Option<Array>> {
    self.cached_encoder_output_from_window(0)
  }

  /// Cached encoded windows starting from `start_window`, concatenated.
  pub fn cached_encoder_output_from_window(&self, start_window: usize) -> Result<Option<Array>> {
    if start_window >= self.cached_windows.len() {
      return Ok(None);
    }
    let slice = &self.cached_windows[start_window..];
    if slice.is_empty() {
      return Ok(None);
    }
    if slice.len() == 1 {
      return Ok(Some(slice[0].try_clone()?));
    }
    let refs: Vec<&Array> = slice.iter().collect();
    Ok(Some(concatenate(&refs, 0)?))
  }

  /// Encode the current pending partial window for early feedback.
  ///
  /// Does NOT consume the pending frames — they stay in the buffer and
  /// will be re-encoded as part of the full window when it completes.
  /// Returns `None` when there are no pending frames.
  ///
  /// The partial buffer is zero-padded to `window_size` before being
  /// passed to the backend, with `valid_frames` set to the number of
  /// real (non-padding) rows — see
  /// [`StreamingEncoderBackend::encode_window`]'s contract.
  pub fn encode_pending(&self) -> Result<Option<Array>> {
    let Some(frames) = self.pending_frames.as_ref() else {
      return Ok(None);
    };
    let valid_frames = self.pending_frame_count;
    if valid_frames == 0 {
      return Ok(None);
    }
    let padded = pad_to_window_size(frames, valid_frames, self.window_size)?;
    let mut encoded = self.encoder.encode_window(&padded, valid_frames)?;
    encoded.eval()?;
    Ok(Some(encoded))
  }

  /// Cached + pending output concatenated, with optional
  /// `from_window` clip.
  pub fn full_encoder_output(&self, from_window: Option<usize>) -> Result<Option<Array>> {
    let cached = match from_window {
      Some(start) => self.cached_encoder_output_from_window(start)?,
      None => self.cached_encoder_output()?,
    };
    let pending = self.encode_pending()?;
    match (cached, pending) {
      (None, None) => Ok(None),
      (Some(c), None) => Ok(Some(c)),
      (None, Some(p)) => Ok(Some(p)),
      (Some(c), Some(p)) => Ok(Some(concatenate(&[&c, &p], 0)?)),
    }
  }

  /// Drain newly-encoded windows since the last drain. Used by the
  /// session to schedule one-shot per-window finalize decodes.
  pub fn drain_newly_encoded_windows(&mut self) -> Vec<Array> {
    std::mem::take(&mut self.newly_encoded_windows)
  }

  /// Reset all state for a new session.
  pub fn reset(&mut self) {
    self.cached_windows.clear();
    self.newly_encoded_windows.clear();
    self.total_encoded_windows = 0;
    self.pending_frames = None;
    self.pending_frame_count = 0;
  }
}

/// Zero-pad a `(valid_frames, n_mels)` mel buffer along axis 0 to
/// `(window_size, n_mels)`. Returns the input unchanged when
/// `valid_frames == window_size` (the buffer is already the right size).
///
/// Lives free-standing (no `B` parameter) so [`flush_partial`] and
/// [`encode_pending`] can share the same padding path without monomorphizing
/// twice per backend.
fn pad_to_window_size(frames: &Array, valid_frames: usize, window_size: usize) -> Result<Array> {
  if valid_frames >= window_size {
    return frames.try_clone();
  }
  let high = window_size - valid_frames;
  let high_i32 = i32::try_from(high).map_err(|_| Error::Backend {
    message: format!(
      "StreamingEncoder: pad-high count {high} exceeds i32::MAX for window-size padding"
    ),
  })?;
  // 0-D scalar zero — `pad` casts it to the input dtype for the constant
  // fill (matches the convention used by `audio::dsp::place_window`).
  let pad_value = Array::zeros::<f32>(&[0i32; 0])?;
  pad(
    frames,
    &[0_i32],
    &[0_i32],
    &[high_i32],
    &pad_value,
    c"constant",
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::Mutex;

  /// Mock encoder that records `(rows, valid_frames)` of every window it
  /// is asked to encode and returns a deterministic `(rows, 2)` array
  /// where each row is filled with its row-index in the window.
  struct MockEncoder {
    window_size: usize,
    calls: Mutex<Vec<(usize, usize)>>,
  }

  impl MockEncoder {
    fn new(window_size: usize) -> Self {
      Self {
        window_size,
        calls: Mutex::new(Vec::new()),
      }
    }

    fn call_count(&self) -> usize {
      self.calls.lock().unwrap().len()
    }

    fn last_call_rows(&self) -> Option<usize> {
      self.calls.lock().unwrap().last().map(|(rows, _)| *rows)
    }

    fn last_call_valid_frames(&self) -> Option<usize> {
      self.calls.lock().unwrap().last().map(|(_, valid)| *valid)
    }
  }

  impl StreamingEncoderBackend for MockEncoder {
    fn window_size(&self) -> usize {
      self.window_size
    }

    fn encode_window(&self, mel_window: &Array, valid_frames: usize) -> Result<Array> {
      let rows = mel_window.shape().first().copied().unwrap_or(0);
      self.calls.lock().unwrap().push((rows, valid_frames));
      // Output `(rows, 2)` with each row = `[row_index, 0]`.
      let mut buf: Vec<f32> = Vec::with_capacity(rows * 2);
      for i in 0..rows {
        buf.push(i as f32);
        buf.push(0.0);
      }
      Array::from_slice::<f32>(&buf, &[rows as i32, 2i32])
    }
  }

  /// Build a `(rows, n_mels)` zero mel-frame array.
  fn zero_mel(rows: usize, n_mels: usize) -> Array {
    let buf = vec![0.0_f32; rows * n_mels];
    Array::from_slice::<f32>(&buf, &[rows as i32, n_mels as i32]).unwrap()
  }

  #[test]
  fn feed_accumulates_until_window_full_then_calls_backend_once() {
    let encoder = MockEncoder::new(16);
    let mut stream = StreamingEncoder::new(encoder, 4, 0);

    // First chunk: 8 frames — short of the window, no encode.
    assert_eq!(stream.feed(&zero_mel(8, 4)).unwrap(), 0);
    assert_eq!(stream.backend().call_count(), 0);
    assert!(stream.has_pending_frames());

    // Second chunk: 8 more — fills the window, one encode with the full
    // contract: rows == window_size AND valid_frames == window_size.
    assert_eq!(stream.feed(&zero_mel(8, 4)).unwrap(), 1);
    assert_eq!(stream.backend().call_count(), 1);
    assert_eq!(stream.backend().last_call_rows(), Some(16));
    assert_eq!(stream.backend().last_call_valid_frames(), Some(16));
    assert!(!stream.has_pending_frames());
    assert_eq!(stream.encoded_window_count(), 1);
  }

  #[test]
  fn feed_emits_multiple_windows_when_input_exceeds_one_window() {
    let encoder = MockEncoder::new(8);
    let mut stream = StreamingEncoder::new(encoder, 4, 0);
    // 24 rows / window 8 = 3 full windows.
    let new_windows = stream.feed(&zero_mel(24, 4)).unwrap();
    assert_eq!(new_windows, 3);
    assert_eq!(stream.encoded_window_count(), 3);
    assert!(!stream.has_pending_frames());
  }

  #[test]
  fn feed_with_overlap_advances_by_stride_not_full_window() {
    let encoder = MockEncoder::new(8);
    let mut stream = StreamingEncoder::new(encoder, 4, 2); // stride = 6
    assert_eq!(stream.window_stride(), 6);

    // 14 rows: window 1 covers [0..8), stride advances to 6,
    // pending = [6..14) = 8 rows → window 2 covers [6..14), stride
    // advances to 12, pending = [12..14) = 2 rows.
    let n = stream.feed(&zero_mel(14, 4)).unwrap();
    assert_eq!(n, 2);
    assert!(stream.has_pending_frames());
  }

  #[test]
  fn feed_rejects_non_2d_input() {
    let encoder = MockEncoder::new(8);
    let mut stream = StreamingEncoder::new(encoder, 4, 0);
    let one_d = Array::from_slice::<f32>(&[0.0_f32; 8], &[8i32]).unwrap();
    let err = stream.feed(&one_d).unwrap_err();
    assert!(matches!(err, Error::Backend { ref message } if message.contains("2-D")));
  }

  #[test]
  fn flush_partial_encodes_remaining_pending_frames() {
    let encoder = MockEncoder::new(8);
    let mut stream = StreamingEncoder::new(encoder, 4, 0);
    // 5 rows: no window encoded, all pending.
    assert_eq!(stream.feed(&zero_mel(5, 4)).unwrap(), 0);
    let flushed = stream.flush_partial().unwrap();
    assert_eq!(flushed, 1);
    assert_eq!(stream.backend().call_count(), 1);
    // Backend always sees a window_size-row buffer (padded);
    // valid_frames signals the real prefix length.
    assert_eq!(stream.backend().last_call_rows(), Some(8));
    assert_eq!(stream.backend().last_call_valid_frames(), Some(5));
    assert!(!stream.has_pending_frames());
  }

  #[test]
  fn flush_partial_on_empty_buffer_is_noop() {
    let encoder = MockEncoder::new(8);
    let mut stream = StreamingEncoder::new(encoder, 4, 0);
    assert_eq!(stream.flush_partial().unwrap(), 0);
    assert_eq!(stream.backend().call_count(), 0);
  }

  #[test]
  fn cache_evicts_oldest_window_when_max_exceeded() {
    let encoder = MockEncoder::new(8);
    // max_cached_windows = 2, but encoded_window_count tracks the
    // monotonic full count.
    let mut stream = StreamingEncoder::new(encoder, 2, 0);
    // 24 rows = 3 full windows.
    let n = stream.feed(&zero_mel(24, 4)).unwrap();
    assert_eq!(n, 3);
    assert_eq!(stream.encoded_window_count(), 3);
    // Only the last 2 windows are kept in the cache.
    let cached = stream.cached_encoder_output().unwrap().unwrap();
    assert_eq!(cached.shape()[0], 16); // 2 windows × 8 tokens each
  }

  #[test]
  fn drain_newly_encoded_windows_returns_each_window_once() {
    let encoder = MockEncoder::new(8);
    let mut stream = StreamingEncoder::new(encoder, 10, 0);
    let _ = stream.feed(&zero_mel(16, 4)).unwrap(); // 2 windows
    let first = stream.drain_newly_encoded_windows();
    assert_eq!(first.len(), 2);
    let second = stream.drain_newly_encoded_windows();
    assert_eq!(second.len(), 0);
    let _ = stream.feed(&zero_mel(8, 4)).unwrap(); // 1 window
    let third = stream.drain_newly_encoded_windows();
    assert_eq!(third.len(), 1);
  }

  #[test]
  fn encode_pending_does_not_consume_pending_frames() {
    let encoder = MockEncoder::new(8);
    let mut stream = StreamingEncoder::new(encoder, 4, 0);
    let _ = stream.feed(&zero_mel(5, 4)).unwrap();
    let pending_before = stream.has_pending_frames();
    assert!(pending_before);

    let out = stream.encode_pending().unwrap().unwrap();
    // Backend now receives a window_size-row padded buffer; the
    // (rows, 2) passthrough mock reflects that shape.
    assert_eq!(out.shape()[0], 8);
    // Buffer is still intact.
    assert!(stream.has_pending_frames());
    // Encode was called once (for the partial), no full-window encodes
    // yet.
    assert_eq!(stream.backend().call_count(), 1);
    assert_eq!(stream.backend().last_call_rows(), Some(8));
    assert_eq!(stream.backend().last_call_valid_frames(), Some(5));
  }

  #[test]
  fn reset_clears_state_for_new_session() {
    let encoder = MockEncoder::new(8);
    let mut stream = StreamingEncoder::new(encoder, 4, 0);
    let _ = stream.feed(&zero_mel(16, 4)).unwrap();
    assert_eq!(stream.encoded_window_count(), 2);
    stream.reset();
    assert_eq!(stream.encoded_window_count(), 0);
    assert_eq!(stream.total_cached_tokens(), 0);
    assert!(!stream.has_pending_frames());
  }

  /// Strict-contract mock: asserts shape == `(window_size, n_mels)` on
  /// every call. Fails fast if the accumulator ever passes a non-padded
  /// partial buffer through.
  struct StrictContractEncoder {
    window_size: usize,
    n_mels: usize,
    calls: Mutex<Vec<(usize, usize)>>,
  }

  impl StrictContractEncoder {
    fn new(window_size: usize, n_mels: usize) -> Self {
      Self {
        window_size,
        n_mels,
        calls: Mutex::new(Vec::new()),
      }
    }
  }

  impl StreamingEncoderBackend for StrictContractEncoder {
    fn window_size(&self) -> usize {
      self.window_size
    }

    fn encode_window(&self, mel_window: &Array, valid_frames: usize) -> Result<Array> {
      let shape = mel_window.shape();
      assert_eq!(
        shape.len(),
        2,
        "strict contract: window must be 2-D, got shape={shape:?}"
      );
      assert_eq!(
        shape[0], self.window_size,
        "strict contract: window row count must equal window_size={}, got {}",
        self.window_size, shape[0]
      );
      assert_eq!(
        shape[1], self.n_mels,
        "strict contract: window col count must equal n_mels={}, got {}",
        self.n_mels, shape[1]
      );
      assert!(
        valid_frames <= self.window_size,
        "strict contract: valid_frames {valid_frames} must be <= window_size {}",
        self.window_size
      );
      self.calls.lock().unwrap().push((shape[0], valid_frames));
      Array::from_slice::<f32>(&vec![0.0_f32; shape[0] * 2], &[shape[0] as i32, 2i32])
    }
  }

  /// F1 regression: a strict-contract backend that demands shape ==
  /// `(window_size, n_mels)` MUST accept a partial-window flush — the
  /// accumulator zero-pads the buffer and signals the real prefix length
  /// via `valid_frames`.
  #[test]
  fn streaming_encoder_pads_partial_window_with_zeros_and_signals_valid_frames() {
    let mut stream = StreamingEncoder::new(StrictContractEncoder::new(8, 4), 4, 0);
    // Feed 3 partial rows (< window_size == 8).
    let _ = stream.feed(&zero_mel(3, 4)).unwrap();
    // encode_pending must pad to 8 rows + report valid_frames == 3.
    let _ = stream.encode_pending().unwrap().unwrap();
    let calls = stream.backend().calls.lock().unwrap();
    let (rows, valid_frames) = calls[0];
    assert_eq!(rows, 8, "rows must be padded to window_size");
    assert_eq!(valid_frames, 3, "valid_frames must equal the partial count");
    assert!(
      valid_frames < rows,
      "partial case: valid_frames < rows; got valid_frames={valid_frames} rows={rows}"
    );
  }

  /// F1 regression: the full-window path passes `valid_frames ==
  /// window_size` so backends can disambiguate "padded partial" from
  /// "real full window".
  #[test]
  fn streaming_encoder_full_window_passes_valid_frames_equals_window_size() {
    let mut stream = StreamingEncoder::new(StrictContractEncoder::new(8, 4), 4, 0);
    // Feed exactly one full window.
    let new_windows = stream.feed(&zero_mel(8, 4)).unwrap();
    assert_eq!(new_windows, 1);
    let calls = stream.backend().calls.lock().unwrap();
    let (rows, valid_frames) = calls[0];
    assert_eq!(rows, 8);
    assert_eq!(
      valid_frames, 8,
      "full-window path must pass valid_frames == window_size"
    );
  }
}
