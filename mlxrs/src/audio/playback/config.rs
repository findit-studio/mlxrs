//! Playback configuration types ([`PlaybackConfig`], [`SampleFormat`],
//! [`ChannelLayout`]) — the cpal-equivalent of the
//! `AVAudioFormat(standardFormatWithSampleRate: …, channels: …)`
//! call that `MLXAudioCore.AudioPlayer.startStreaming` issues before
//! attaching the player node.
//!
//! Mirrors the parameters the Swift `AudioPlayer` negotiates with
//! `AVAudioEngine`:
//! - sample rate (the streaming format's `sampleRate`),
//! - channel count (mono / stereo / N-channel; Swift defaults to
//!   `channels: 1` for streaming PCM),
//! - optional buffer-size hint (cpal's `BufferSize::Fixed(N)` if set,
//!   `BufferSize::Default` otherwise — equivalent to letting
//!   `AVAudioEngine` pick the I/O buffer size).
//!
//! Sample-format selection is a separate axis from the Swift port:
//! Swift's streaming path is implicitly `f32` (AVAudioPCMBuffer's
//! `floatChannelData`); we keep the same default but expose the
//! [`SampleFormat`] enum so a future caller can target an `i16` /
//! `u16` device without rewriting the player. The minimum viable
//! shipping path here is `f32` — the [`AudioPlayer`] only emits `f32`
//! to its cpal stream today.
//!
//! [`AudioPlayer`]: super::player::AudioPlayer

/// Sample format the player's cpal stream emits.
///
/// Swift's `MLXAudioCore.AudioPlayer` streams 32-bit float PCM
/// (`AVAudioPCMBuffer.floatChannelData`). We expose the cpal-supported
/// formats so a future caller can negotiate a different format with
/// the device, but the shipping [`super::player::AudioPlayer`] path
/// only emits [`SampleFormat::F32`]; the variants are reserved for
/// future format-conversion work (out of scope for A11).
#[derive(
  Debug, Clone, Copy, PartialEq, Eq, Default, derive_more::Display, derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
pub enum SampleFormat {
  /// 32-bit interleaved float (the Swift `AudioPlayer` default; the
  /// only variant currently constructed by [`super::player::AudioPlayer`]).
  #[default]
  F32,
  /// 16-bit signed interleaved integer. Reserved — A11 does not
  /// convert to this format; callers must supply f32 PCM.
  I16,
  /// 16-bit unsigned interleaved integer. Reserved — A11 does not
  /// convert to this format; callers must supply f32 PCM.
  U16,
}

impl SampleFormat {
  /// The canonical lowercase string representation (`f32`/`i16`/`u16`).
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::F32 => "f32",
      Self::I16 => "i16",
      Self::U16 => "u16",
    }
  }
}

/// Channel layout the player's cpal stream emits.
///
/// Mirrors the `channels:` argument the Swift
/// `AudioPlayer.startStreaming(sampleRate:)` passes to
/// `AVAudioFormat(standardFormatWithSampleRate:channels:)`. The Swift
/// streaming default is mono (`channels: 1`); we expose the common
/// shapes plus a raw `Channels(N)` escape hatch for >2 channel
/// configurations cpal supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, derive_more::IsVariant)]
pub enum ChannelLayout {
  /// One-channel (mono) — the Swift `AudioPlayer.startStreaming`
  /// default.
  #[default]
  Mono,
  /// Two-channel (left, right) interleaved.
  Stereo,
  /// Arbitrary channel count. cpal supports devices with >2
  /// channels; this variant lets a caller target one without adding
  /// per-shape enum variants. Must be `>= 1` — zero-channel
  /// configurations are rejected by [`PlaybackConfig::cpal_config`].
  Channels(u16),
}

impl ChannelLayout {
  /// Numeric channel count this layout maps to (cpal's
  /// `StreamConfig::channels`).
  #[inline(always)]
  #[must_use]
  pub fn count(self) -> u16 {
    match self {
      Self::Mono => 1,
      Self::Stereo => 2,
      Self::Channels(n) => n,
    }
  }
}

/// Playback parameters for [`super::player::AudioPlayer`] — the input
/// to `AudioPlayer::new` / `AudioPlayer::with_device`.
///
/// Mirrors the parameters Swift's `AudioPlayer.startStreaming(sampleRate:)`
/// hardcodes into its `AVAudioFormat(standardFormatWithSampleRate:channels:)`:
/// `sampleRate` + `channels` are the streaming-format inputs;
/// `buffer_size_frames` is the cpal-side hint for the device callback
/// buffer (no direct Swift analog — the Swift path lets `AVAudioEngine`
/// pick the I/O buffer size, equivalent to leaving this `None`).
///
/// `queue_capacity_frames` bounds the player's internal sample queue
/// (the cpal equivalent of `AVAudioPlayerNode.scheduleBuffer`'s
/// pending-buffer chain). A bounded capacity is required so
/// [`super::output_stream::AudioOutputStream::write_samples`] can
/// surface a recoverable `Err` on producer-side overflow instead of
/// growing the queue unboundedly (the unbounded-growth case is
/// neither a Swift nor a Rust contract; see
/// [`crate::audio::playback::player::AudioPlayer`] for the overflow
/// policy).
#[derive(Debug, Clone, Copy)]
pub struct PlaybackConfig {
  /// Output sample rate (Hz). The Swift `AudioPlayer.startStreaming`
  /// API takes this as `sampleRate: Double`; mlxrs constrains it to
  /// `u32` to match cpal's `StreamConfig::sample_rate` field.
  ///
  /// Default: 24000 (matches the Swift `MLXAudioUI` voice-pipeline
  /// default).
  sample_rate: u32,
  /// Channel layout. Default: [`ChannelLayout::Mono`] (the Swift
  /// streaming default).
  channels: ChannelLayout,
  /// Sample format. Default: [`SampleFormat::F32`] (the Swift
  /// streaming default; the only format
  /// [`super::player::AudioPlayer`] emits today).
  sample_format: SampleFormat,
  /// Cpal device callback buffer-size hint (frames). `None` lets the
  /// platform pick (cpal's `BufferSize::Default` — equivalent to
  /// `AVAudioEngine`'s automatic I/O buffer sizing).
  buffer_size_frames: Option<u32>,
  /// Maximum queued *FRAMES* (NOT samples) before
  /// [`super::output_stream::AudioOutputStream::write_samples`]
  /// returns [`crate::error::Error::Backend`]. Bounds memory; bound
  /// is enforced in **frames** (one frame = `channels.count()`
  /// interleaved samples). The
  /// [`super::player::AudioPlayer::with_device`] constructor does the
  /// single frame-to-sample conversion via `* channels.count()` —
  /// callers MUST NOT pre-multiply by channel count (doing so
  /// double-counts the bound). The unit here is constant across all
  /// channel layouts.
  ///
  /// Default: `sample_rate as usize * 4` — four seconds of audio at
  /// the configured sample rate, large enough that a moderately
  /// bursty producer (TTS chunk arriving in 100ms windows) doesn't
  /// trip overflow, small enough that an unbounded-producer bug is
  /// caught instead of OOMing.
  queue_capacity_frames: usize,
}

impl Default for PlaybackConfig {
  fn default() -> Self {
    Self::new(24_000, ChannelLayout::Mono, SampleFormat::F32)
  }
}

impl PlaybackConfig {
  /// Build a [`PlaybackConfig`] with the given `sample_rate`, `channels`, and
  /// `sample_format`. Sets `buffer_size_frames = None` (platform default) and
  /// `queue_capacity_frames = sample_rate * 4` (four seconds of frames).
  ///
  /// Mirrors the three-parameter shape of Swift's
  /// `AVAudioFormat(standardFormatWithSampleRate:channels:)` call.
  #[must_use]
  pub fn new(sample_rate: u32, channels: ChannelLayout, sample_format: SampleFormat) -> Self {
    Self {
      sample_rate,
      channels,
      sample_format,
      buffer_size_frames: None,
      queue_capacity_frames: (sample_rate as usize) * 4,
    }
  }

  /// Override the cpal device callback buffer-size hint (in frames).
  /// `None` (the default) lets the platform pick (`BufferSize::Default`).
  #[must_use]
  pub fn with_buffer_size_frames(mut self, frames: u32) -> Self {
    self.buffer_size_frames = Some(frames);
    self
  }

  /// Override the producer queue capacity (in frames). The player's
  /// `with_device` constructor converts to samples via `× channels.count()`.
  #[must_use]
  pub fn with_queue_capacity_frames(mut self, frames: usize) -> Self {
    self.queue_capacity_frames = frames;
    self
  }

  /// Build a config with the default channel layout
  /// ([`ChannelLayout::Mono`]) + format ([`SampleFormat::F32`]) + a
  /// 4-second queue capacity.
  ///
  /// Mirrors the Swift `AudioPlayer.startStreaming(sampleRate:)`
  /// single-argument constructor.
  #[must_use]
  pub fn mono(sample_rate: u32) -> Self {
    Self::new(sample_rate, ChannelLayout::Mono, SampleFormat::F32)
  }

  /// Build a stereo config — a convenience helper. Swift's
  /// `AudioPlayer` doesn't expose this directly (its streaming path
  /// is mono); ported for parity with cpal's `default_output_config`
  /// which is typically stereo.
  ///
  /// `queue_capacity_frames` is `sample_rate * 4` (four seconds of
  /// frames), the same unit as [`PlaybackConfig::mono`] — the
  /// frame-to-sample fan-out by channel count is the player's
  /// responsibility (see [`PlaybackConfig::queue_capacity_frames`]).
  #[must_use]
  pub fn stereo(sample_rate: u32) -> Self {
    Self::new(sample_rate, ChannelLayout::Stereo, SampleFormat::F32)
  }

  /// Output sample rate (Hz).
  #[inline(always)]
  pub fn sample_rate(&self) -> u32 {
    self.sample_rate
  }

  /// Channel layout.
  #[inline(always)]
  pub fn channels(&self) -> ChannelLayout {
    self.channels
  }

  /// Sample format.
  #[inline(always)]
  pub fn sample_format(&self) -> SampleFormat {
    self.sample_format
  }

  /// Cpal device callback buffer-size hint (frames). `None` = platform default.
  #[inline(always)]
  pub fn buffer_size_frames(&self) -> Option<u32> {
    self.buffer_size_frames
  }

  /// Maximum queued frames before `write_samples` returns `Err`.
  #[inline(always)]
  pub fn queue_capacity_frames(&self) -> usize {
    self.queue_capacity_frames
  }

  /// Cpal `StreamConfig` equivalent of `self`. Returns
  /// [`crate::error::Error::Backend`] if the channel count is zero
  /// (cpal would reject the stream build, but we surface a typed
  /// error pre-build).
  ///
  /// # Errors
  /// - [`crate::error::Error::Backend`] if `self.channels.count() == 0`.
  pub fn cpal_config(&self) -> crate::error::Result<cpal::StreamConfig> {
    let channels = self.channels.count();
    if channels == 0 {
      return Err(crate::error::Error::Backend {
        message: "PlaybackConfig: channel count must be >= 1".to_string(),
      });
    }
    let buffer_size = match self.buffer_size_frames {
      Some(n) => cpal::BufferSize::Fixed(n),
      None => cpal::BufferSize::Default,
    };
    // Note: `cpal::SampleRate` is a `pub type SampleRate = u32` alias
    // in cpal 0.17 (the older `SampleRate(u32)` newtype was flattened
    // into a raw `u32` upstream), so we pass the rate directly.
    Ok(cpal::StreamConfig {
      channels,
      sample_rate: self.sample_rate,
      buffer_size,
    })
  }
}
