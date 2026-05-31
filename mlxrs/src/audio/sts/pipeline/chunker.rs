//! [`AudioChunker`] + [`FixedSizeAudioChunker`] + [`PreRollBuffer`] —
//! sample-buffer primitives the pipeline composes on the input side.
//!
//! Ports the three helpers `mlx_audio.sts.voice_pipeline` carries inside
//! its module:
//!
//! - [`FixedSizeAudioChunker`] ([`voice_pipeline.py:144-160`][vp-chunker]):
//!   accumulates incoming samples in a `Vec<f32>` and emits every
//!   `chunk_size`-element window in order, retaining the unaligned tail.
//!   The Voxtral STT / Silero VAD frontends consume fixed-sized frames,
//!   so the pipeline funnels every mic burst through this chunker before
//!   dispatching to VAD / STT.
//! - [`PreRollBuffer`] ([`voice_pipeline.py:162-194`][vp-preroll]): a
//!   bounded ring of the most recent `max_samples` samples mlx-audio
//!   prepends to the first transcriber feed when a speech turn starts,
//!   so the STT model sees a small leading context the VAD's start-of-
//!   speech detector ran past.
//! - [`AudioChunker`] (trait): the shared shape both the fixed-size
//!   chunker and any caller-supplied alternative (e.g. a variable-frame
//!   chunker for an STT model with a different frame contract)
//!   implement, so [`super::orchestrator::VoiceSession`] can take a
//!   chunker generically rather than pinning to
//!   [`FixedSizeAudioChunker`].
//!
//! [vp-chunker]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L144-L160
//! [vp-preroll]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L162-L194

use crate::error::Result;

/// The chunker contract: accept arbitrary-length sample bursts and emit
/// model-aligned fixed-size (or otherwise) chunks in order.
///
/// `push_samples` returns the chunks ready for downstream consumption
/// (VAD / STT); residual samples shorter than one chunk stay buffered
/// inside the implementor. An empty `samples` push is a no-op
/// (`Ok(vec![])`).
///
/// Mirrors the duck-typed `chunker.push(samples) -> list[np.ndarray]`
/// shape `voice_pipeline.py` consumes; the trait makes that interface
/// explicit so [`super::orchestrator::VoiceSession`] can be generic
/// over the chunker rather than depending on a concrete struct.
pub trait AudioChunker {
  /// Push samples into the chunker and return any newly-completed
  /// chunks (in arrival order). Residual samples too small for a
  /// chunk stay buffered.
  ///
  /// # Errors
  /// Implementor-defined; the default [`FixedSizeAudioChunker`]
  /// never fails (`Ok`), but a chunker that does its own resampling
  /// or format conversion may.
  fn push_samples(&mut self, samples: &[f32]) -> Result<Vec<Vec<f32>>>;

  /// Drain any residual buffered samples (shorter than one full
  /// chunk) and return them, leaving the chunker empty. Used by
  /// [`super::orchestrator::VoiceSession::flush_in_progress_turn`]
  /// at mic-EOF so the trailing partial-chunk audio still reaches
  /// the STT.
  ///
  /// Implementors that never buffer (e.g. a pass-through chunker
  /// that immediately re-emits whatever it received) may return an
  /// empty `Vec`. The default [`FixedSizeAudioChunker`] returns the
  /// tail and clears its internal buffer.
  fn drain_residual(&mut self) -> Vec<f32>;

  /// Reset internal state (drop any buffered residual). Used when
  /// the pipeline tears a turn down and starts fresh (mirror of the
  /// `chunker = FixedSizeAudioChunker(...)` reset in mlx-audio's
  /// `SileroSpeechGate.reset`).
  fn reset(&mut self);
}

/// Accumulates incoming samples and emits every `chunk_size`-element
/// window in order — direct port of
/// [`mlx_audio.sts.voice_pipeline.FixedSizeAudioChunker`][vp-chunker].
///
/// The internal buffer is a `Vec<f32>` (mlx-audio uses `np.ndarray`
/// `concat` + slice; mlxrs uses `Vec<f32>` `drain` to avoid the
/// per-push allocation the numpy concat path incurs).
///
/// - `push_samples(&[]).unwrap()` → `vec![]` (no-op).
/// - `push_samples(samples)` returns one chunk per multiple of
///   `chunk_size` the cumulative buffer crosses; residual samples
///   (`buffer.len() % chunk_size`) stay for the next push.
/// - `chunk_size` must be `> 0`; the constructor panics on `0`
///   (mlx-audio's `int(chunk_size)` does not validate either, and
///   the downstream VAD / STT contract has no sensible behavior at
///   `chunk_size = 0`).
///
/// [vp-chunker]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L144-L160
#[derive(Debug, Clone)]
pub struct FixedSizeAudioChunker {
  chunk_size: usize,
  buffer: Vec<f32>,
}

impl FixedSizeAudioChunker {
  /// Build a chunker that emits `chunk_size`-sample chunks.
  ///
  /// # Panics
  /// Panics on `chunk_size == 0` — the chunking loop is undefined
  /// for a zero-size window (and mlx-audio's downstream VAD / STT
  /// frame contract has no sensible behavior at zero).
  #[must_use]
  pub fn new(chunk_size: usize) -> Self {
    assert!(
      chunk_size > 0,
      "FixedSizeAudioChunker requires chunk_size > 0 (got 0)"
    );
    Self {
      chunk_size,
      buffer: Vec::new(),
    }
  }

  /// The configured chunk size (samples).
  #[inline(always)]
  #[must_use]
  pub fn chunk_size(&self) -> usize {
    self.chunk_size
  }

  /// Currently-buffered tail length (samples). The next chunk will
  /// be emitted when `buffered_len() + next_push.len() >= chunk_size`.
  #[inline(always)]
  #[must_use]
  pub fn buffered_len(&self) -> usize {
    self.buffer.len()
  }
}

impl AudioChunker for FixedSizeAudioChunker {
  /// Push samples; emit every newly-aligned chunk in order.
  /// Empty-push is a no-op (`Ok(vec![])`) — mirrors mlx-audio's
  /// `if samples.size == 0: return []`.
  fn push_samples(&mut self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
    if samples.is_empty() {
      return Ok(Vec::new());
    }
    self.buffer.extend_from_slice(samples);
    let n_chunks = self.buffer.len() / self.chunk_size;
    if n_chunks == 0 {
      return Ok(Vec::new());
    }
    let mut chunks = Vec::with_capacity(n_chunks);
    let consumed = n_chunks * self.chunk_size;
    {
      let mut drain = self.buffer.drain(..consumed);
      for _ in 0..n_chunks {
        let chunk: Vec<f32> = (&mut drain).take(self.chunk_size).collect();
        chunks.push(chunk);
      }
    }
    Ok(chunks)
  }

  fn drain_residual(&mut self) -> Vec<f32> {
    std::mem::take(&mut self.buffer)
  }

  fn reset(&mut self) {
    self.buffer.clear();
  }
}

/// Bounded ring buffer of the most recent `max_samples` mono `f32`
/// samples — direct port of
/// [`mlx_audio.sts.voice_pipeline.PreRollBuffer`][vp-preroll].
///
/// The pipeline feeds **every** incoming mic burst into the pre-roll
/// while the transcriber is idle; when speech actually starts the
/// pre-roll's contents are prepended to the first STT feed so the
/// model sees the leading sounds the VAD ran past. After the speech
/// turn completes the buffer is cleared.
///
/// Storage is a single contiguous `Vec<f32>` (mlx-audio uses a list
/// of numpy arrays + sliced front-drop; mlxrs's contiguous-`Vec`
/// representation is simpler and the front-drop is one `drain` call).
///
/// [vp-preroll]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L162-L194
#[derive(Debug, Clone)]
pub struct PreRollBuffer {
  max_samples: usize,
  buffer: Vec<f32>,
}

impl PreRollBuffer {
  /// Build a pre-roll capped at `max_samples` (the most recent N
  /// samples are kept). `max_samples == 0` is allowed and turns the
  /// buffer into a permanent no-op (matches mlx-audio's
  /// `if self.max_samples <= 0: return`).
  #[must_use]
  pub const fn new(max_samples: usize) -> Self {
    Self {
      max_samples,
      buffer: Vec::new(),
    }
  }

  /// Append samples; trim the front so the buffer holds at most
  /// `max_samples`. Empty-push is a no-op.
  pub fn append(&mut self, samples: &[f32]) {
    if self.max_samples == 0 || samples.is_empty() {
      return;
    }
    self.buffer.extend_from_slice(samples);
    if self.buffer.len() > self.max_samples {
      let excess = self.buffer.len() - self.max_samples;
      self.buffer.drain(..excess);
    }
  }

  /// Return a snapshot of the current pre-roll contents (the
  /// caller owns the `Vec`).
  #[must_use]
  pub fn snapshot(&self) -> Vec<f32> {
    self.buffer.clone()
  }

  /// Clear the buffer (post-turn-end reset; mirror of mlx-audio's
  /// `PreRollBuffer.clear`).
  pub fn clear(&mut self) {
    self.buffer.clear();
  }

  /// Currently-buffered length (samples).
  #[inline(always)]
  #[must_use]
  pub fn len(&self) -> usize {
    self.buffer.len()
  }

  /// Whether the buffer holds no samples.
  #[inline(always)]
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.buffer.is_empty()
  }

  /// Configured capacity (max retained samples).
  #[inline(always)]
  #[must_use]
  pub fn max_samples(&self) -> usize {
    self.max_samples
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 16 kHz × 20 ms chunks = 320 samples per chunk; pushing 1000
  /// samples should emit 3 chunks (3 × 320 = 960) and leave 40 in the
  /// buffer. The doc-spec example fence.
  #[test]
  fn fixed_size_chunker_emits_aligned_chunks() {
    let chunk_size = (16_000 * 20) / 1_000;
    assert_eq!(chunk_size, 320);
    let mut chunker = FixedSizeAudioChunker::new(chunk_size);
    let samples: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let chunks = chunker.push_samples(&samples).unwrap();

    assert_eq!(chunks.len(), 3);
    for c in &chunks {
      assert_eq!(c.len(), 320);
    }
    // Tail length = 1000 - 3*320 = 40.
    assert_eq!(chunker.buffered_len(), 40);

    // Chunk content is in order (0..320, 320..640, 640..960).
    assert_eq!(chunks[0].first().copied(), Some(0.0));
    assert_eq!(chunks[0].last().copied(), Some(319.0));
    assert_eq!(chunks[1].first().copied(), Some(320.0));
    assert_eq!(chunks[2].last().copied(), Some(959.0));
  }

  /// Empty push is a no-op (matches mlx-audio's `if samples.size ==
  /// 0: return []`).
  #[test]
  fn fixed_size_chunker_empty_push_is_noop() {
    let mut chunker = FixedSizeAudioChunker::new(512);
    let chunks = chunker.push_samples(&[]).unwrap();
    assert!(chunks.is_empty());
    assert_eq!(chunker.buffered_len(), 0);
  }

  /// Pushes that don't span a chunk boundary buffer the samples and
  /// emit nothing — until the cumulative buffer crosses `chunk_size`.
  #[test]
  fn fixed_size_chunker_accumulates_across_pushes() {
    let mut chunker = FixedSizeAudioChunker::new(100);
    // 30 + 30 + 30 = 90 < 100 → no chunk emitted yet.
    for _ in 0..3 {
      let chunks = chunker.push_samples(&[1.0; 30]).unwrap();
      assert!(chunks.is_empty());
    }
    assert_eq!(chunker.buffered_len(), 90);
    // 90 + 20 = 110 → one 100-chunk + 10 tail.
    let chunks = chunker.push_samples(&[2.0; 20]).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].len(), 100);
    assert_eq!(chunker.buffered_len(), 10);
  }

  /// `reset()` drops the buffered tail.
  #[test]
  fn fixed_size_chunker_reset_drops_tail() {
    let mut chunker = FixedSizeAudioChunker::new(512);
    chunker.push_samples(&vec![0.0; 200]).unwrap();
    assert_eq!(chunker.buffered_len(), 200);
    chunker.reset();
    assert_eq!(chunker.buffered_len(), 0);
  }

  /// `drain_residual()` returns the buffered tail and empties the
  /// chunker — the EOF-finalize hook the orchestrator needs to avoid
  /// dropping the partial chunk before STT.
  #[test]
  fn fixed_size_chunker_drain_residual_returns_and_clears_tail() {
    let mut chunker = FixedSizeAudioChunker::new(512);
    let _ = chunker.push_samples(&[1.0_f32; 200]).unwrap();
    assert_eq!(chunker.buffered_len(), 200);

    let drained = chunker.drain_residual();
    assert_eq!(drained.len(), 200);
    assert!(drained.iter().all(|&s| s == 1.0));
    assert_eq!(chunker.buffered_len(), 0);

    // Second drain on an empty buffer is a no-op (empty Vec).
    let drained2 = chunker.drain_residual();
    assert!(drained2.is_empty());
  }

  /// `drain_residual()` on a chunker that has emitted all its samples
  /// returns an empty `Vec` (no tail buffered).
  #[test]
  fn fixed_size_chunker_drain_residual_empty_when_aligned() {
    let mut chunker = FixedSizeAudioChunker::new(100);
    let _ = chunker.push_samples(&[0.0_f32; 200]).unwrap();
    assert_eq!(chunker.buffered_len(), 0);
    let drained = chunker.drain_residual();
    assert!(drained.is_empty());
  }

  /// `chunk_size == 0` panics at construction.
  #[test]
  #[should_panic(expected = "chunk_size > 0")]
  fn fixed_size_chunker_zero_size_panics() {
    let _ = FixedSizeAudioChunker::new(0);
  }

  /// `PreRollBuffer` trims to capacity, never grows past `max_samples`.
  #[test]
  fn preroll_trims_to_capacity() {
    let mut preroll = PreRollBuffer::new(8);
    preroll.append(&[1.0, 2.0, 3.0]);
    preroll.append(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    // After append: total written = 10, capped at 8 → drops [1, 2].
    assert_eq!(preroll.len(), 8);
    assert_eq!(
      preroll.snapshot(),
      vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    );
  }

  /// `clear()` empties the buffer.
  #[test]
  fn preroll_clear_empties() {
    let mut preroll = PreRollBuffer::new(8);
    preroll.append(&[1.0, 2.0, 3.0]);
    assert!(!preroll.is_empty());
    preroll.clear();
    assert!(preroll.is_empty());
  }

  /// `max_samples == 0` is a permanent no-op (mlx-audio parity).
  #[test]
  fn preroll_zero_capacity_is_noop() {
    let mut preroll = PreRollBuffer::new(0);
    preroll.append(&[1.0, 2.0, 3.0]);
    assert!(preroll.is_empty());
    assert_eq!(preroll.snapshot(), Vec::<f32>::new());
  }
}
