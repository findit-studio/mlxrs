//! [`AudioPlayer`] — cpal-backed device playback, the mlxrs port of
//! `mlx-audio-swift`'s
//! [`MLXAudioCore.AudioPlayer.startStreaming(sampleRate:)`][swift-ap] +
//! `scheduleAudioChunk(_:withCrossfade:)` streaming path.
//!
//! ## API mirror
//!
//! Swift `AudioPlayer` exposes a roughly six-method streaming surface
//! over `AVAudioEngine` + `AVAudioPlayerNode`:
//!
//! | Swift                                       | mlxrs                                              |
//! | ------------------------------------------- | -------------------------------------------------- |
//! | `startStreaming(sampleRate:)`               | [`AudioPlayer::new`] / [`AudioPlayer::with_device`] + [`AudioPlayer::start`] |
//! | `scheduleAudioChunk(_:withCrossfade:)`      | [`AudioPlayer::write_samples`] (via [`super::output_stream::AudioOutputStream`]) |
//! | `pause()` (streaming branch)                | [`AudioPlayer::pause`]                              |
//! | `togglePlayPause()` (streaming branch)      | [`AudioPlayer::resume`]                             |
//! | `stopStreaming()` / `stop()`                | [`AudioPlayer::stop`] / `Drop`                      |
//! | `isPlaying` / `isStreamingMode`             | [`AudioPlayer::is_running`]                         |
//! | `finishStreamingInput()`                    | [`super::output_stream::AudioOutputStream::flush`]  |
//!
//! Swift's volume is read off `AVAudioPlayerNode.volume`; mlxrs
//! exposes the equivalent via [`AudioPlayer::store_volume`] +
//! [`AudioPlayer::volume`] backed by an `AtomicU32` (f32 bits) the
//! cpal callback reads each invocation.
//!
//! ## Cpal callback + buffer-queue plumbing
//!
//! The Swift path is buffer-by-buffer (`AVAudioPlayerNode.scheduleBuffer`
//! per `[Float]` chunk, each carrying its own completion handler).
//! The cpal equivalent inverts the polarity: cpal owns the I/O
//! thread, calls back into us with a pre-sized `&mut [f32]` to fill,
//! and we pull samples from a thread-safe queue. Concretely:
//!
//! ```text
//! producer thread (e.g. STS pipeline)            cpal I/O thread
//! ───────────────────────────────────            ─────────────────
//!   write_samples(&[f32]) ──┐                          │
//!                           ▼                          │
//!                 SampleQueue::push                    │
//!                           │                          │
//!                           ▼                          ▼
//!                       Arc<Mutex<VecDeque<f32>>> ── callback fills &mut [f32]
//!                           │                          │
//!                           │                          ▼
//!                           │              for s in out: s = pop_or_zero() * volume
//!                           ▼
//!                  AudioPlayer::buffer_depth
//! ```
//!
//! - **Producer side.** [`AudioPlayer::write_samples`] locks the
//!   shared `VecDeque<f32>` (capped at
//!   [`super::config::PlaybackConfig::queue_capacity_frames`] × channel
//!   count). Returns `Err` on overflow (a recoverable
//!   [`crate::error::Error::Backend`] — no producer surprise OOM).
//!   This is the cpal-equivalent of `AVAudioPlayerNode.scheduleBuffer`
//!   returning even though the underlying scheduling chain is
//!   bounded.
//! - **Cpal callback.** Runs on cpal's audio I/O thread; locks the
//!   queue (a short critical section — only `pop_front` calls under
//!   the lock), reads the current volume from the `AtomicU32`,
//!   writes `pop * volume` per sample. On underrun (queue empty)
//!   the callback writes `0.0` — silence — instead of panicking or
//!   blocking. This matches the Swift behavior: the player node
//!   sits idle if no buffer is scheduled.
//! - **State.** Stored as `Arc<AtomicU8>` so both producer and cpal
//!   callback can observe transitions without holding the queue
//!   lock; values [`STATE_STOPPED`], [`STATE_RUNNING`],
//!   [`STATE_PAUSED`] map to Swift's `isPlaying` /
//!   `isStreamingMode` distinction (we collapse them into a single
//!   tri-state to make the cpal-side check a single atomic load).
//!
//! ## Concurrency
//!
//! The cpal callback runs on a real-time audio I/O thread with a
//! hard deadline: missing the device's callback period yields an
//! audible underrun even if the silence-fill path is correct
//! (CoreAudio fills with whatever was left in the buffer). The
//! callback **must not block** on producer-held mutexes.
//!
//! Concrete contract:
//!
//! - **Callback uses `try_lock`, not `lock`.** If the producer is
//!   mid-extend on the queue mutex, the callback emits silence for
//!   the current period instead of blocking past the device
//!   deadline. The cost (one underrun period) is bounded; blocking
//!   the audio thread is not.
//! - **Producer chunks large writes.** [`AudioPlayer::write_samples`]
//!   splits writes larger than [`WRITE_CHUNK_MAX`] into per-chunk
//!   lock acquisitions, so the callback's `try_lock` window is
//!   bounded by the duration of one chunk's `extend` (microseconds
//!   at 4096 f32 samples) rather than the duration of the full
//!   producer payload (which can be hundreds of milliseconds for a
//!   multi-second TTS chunk).
//! - **Future migration.** If audible underruns persist under
//!   profiling, swap the `Mutex<VecDeque<f32>>` for a lock-free
//!   `crossbeam-queue::ArrayQueue<f32>` (no new dep today; the
//!   `try_lock` + chunking pattern stays within the existing
//!   surface).
//!
//! ## Scope cuts (explicit, A11)
//!
//! The Swift `AudioPlayer` exposes a few capabilities A11 deliberately
//! does NOT port; each is a separate follow-up issue per the
//! `[[feedback_match_official_binding_design]]` rule:
//!
//! - **Audio input / recording.** A11 is playback-only.
//!   `AVAudioPlayer` / `AVAudioPlayerDelegate` are not mirrored.
//! - **File I/O (`loadAudio(from: URL)`).** A11 plays raw PCM. WAV /
//!   MP3 / FLAC loading already lives in [`crate::audio::io`]; a
//!   caller that wants to play a file decodes there and pipes the
//!   resulting samples through [`AudioPlayer::write_samples`].
//! - **Format conversion (`PCMStreamConverter`).** A11 expects the
//!   caller to supply samples at the configured
//!   [`super::config::PlaybackConfig::sample_rate`] /
//!   [`super::config::PlaybackConfig::channels`]. Resampling +
//!   format-conversion is a separate concern (already partially
//!   covered by [`crate::audio::io::load_audio`]'s resampling, fully
//!   covered by a future polyphase resampler follow-up).
//! - **Crossfade / fade-in (`scheduleAudioChunk(_:withCrossfade:)`'s
//!   `withCrossfade: true` branch).** Crossfade is an
//!   application-level concern; A11 plays exactly the samples the
//!   caller pushes. A future helper module can wrap `AudioPlayer`
//!   with a fade-in/crossfade transform without touching the
//!   playback core.
//! - **Per-buffer completion callbacks.** Swift schedules each
//!   buffer with a `completionCallbackType: .dataConsumed` to track
//!   queued-buffer drain; cpal has no per-buffer-completion hook —
//!   instead [`super::output_stream::AudioOutputStream::flush`]
//!   blocks until [`AudioPlayer::buffer_depth`] reaches zero, which
//!   is the same end-state contract (`onDidFinishStreaming` fires
//!   when `queuedBuffers == 0`).
//! - **Timer-driven `currentTime` publishing.** Swift uses
//!   `Timer.scheduledTimer` (every 100ms) + Combine to publish
//!   `currentTime` for UI binding. mlxrs is a Rust library, not a
//!   SwiftUI ObservableObject; no `@Published` properties / no
//!   Combine equivalent. Callers that want positional readback can
//!   maintain their own sample counter against
//!   [`AudioPlayer::buffer_depth`].
//!
//! [swift-ap]: https://github.com/fintit-ai/mlx-audio-swift/blob/main/Sources/MLXAudioCore/AudioPlayer.swift

use std::{
  collections::VecDeque,
  sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering},
  },
  thread,
  time::{Duration, Instant},
};

use cpal::{
  Stream, StreamError,
  traits::{DeviceTrait, HostTrait, StreamTrait},
};

use super::{
  config::{PlaybackConfig, SampleFormat},
  output_stream::AudioOutputStream,
};
use crate::error::{Error, Result};

/// Stopped — the cpal stream is built but not playing (Swift's
/// `!isStreaming && !isPlaying`).
pub const STATE_STOPPED: u8 = 0;
/// Running — the cpal stream is `play()`ing and producer writes are
/// accepted (Swift's `isStreaming && isPlaying`).
pub const STATE_RUNNING: u8 = 1;
/// Paused — the cpal stream is `pause()`d but the queue retains its
/// contents (Swift's `playerNode.pause()` branch of `pause()`).
pub const STATE_PAUSED: u8 = 2;

/// Spin-wait granularity for [`AudioPlayer::flush`]. Picked to match
/// Swift's `Timer.scheduledTimer(withTimeInterval: 0.1, ...)` poll
/// cadence so flush latency under tight contention is bounded by the
/// same order of magnitude as the Swift implementation.
const FLUSH_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Default [`AudioPlayer::flush`] timeout. Defensive cap so a stalled
/// cpal device doesn't block the producer forever; long enough that a
/// realistic 4-second queue (the [`PlaybackConfig`] default) can drain
/// at real-time playback speeds with safety margin.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum number of f32 samples the producer extends into the
/// shared queue per lock acquisition. Bounds the duration the
/// callback's `try_lock` would have to wait if it raced the producer
/// (~microseconds at 4096 samples on a modern CPU), so the cpal
/// callback's silence-on-contention fallback never lingers across
/// multiple device periods.
///
/// Picked at 4096 f32 samples (170ms at 24 kHz mono, 85ms at 48 kHz
/// stereo) — far larger than any realistic cpal callback buffer
/// (typically 64–1024 frames) but small enough that the producer's
/// per-chunk `extend` finishes well inside one device period.
pub const WRITE_CHUNK_MAX: usize = 4096;

/// Sanitize a caller-supplied volume scalar to `[0.0, 1.0]`. Public
/// helper so the policy can be unit-tested without constructing an
/// [`AudioPlayer`] (which opens a cpal device — not available in
/// CI). Used by [`AudioPlayer::store_volume`].
///
/// - **Non-finite (`NaN`, `±∞`) maps to `0.0`.** `f32::clamp`
///   preserves NaN bits, which would cause the cpal callback's
///   `sample * volume` arithmetic to emit NaN PCM (audible as
///   full-scale noise on most DACs).
/// - **Finite out-of-range is clamped to `[0.0, 1.0]`.** Matches
///   `AVAudioPlayerNode.volume`'s documented range.
#[must_use]
pub fn sanitize_volume(vol: f32) -> f32 {
  if vol.is_finite() {
    vol.clamp(0.0, 1.0)
  } else {
    0.0
  }
}

/// Thread-shared callback context. Lives behind an `Arc` so the cpal
/// stream's callback (which gets a `'static` closure) and the
/// producer-side [`AudioPlayer`] can both read/write the same state.
///
/// Kept as a dedicated struct (rather than five sibling `Arc<…>`
/// fields on `AudioPlayer`) so the `Drop` impl on [`AudioPlayer`]
/// can drop the cpal stream first (which joins the callback thread)
/// without an interleaved-drop hazard on the queue / state /
/// volume atomics.
struct SharedState {
  /// Producer-consumer queue of interleaved f32 samples. Bounded at
  /// `PlaybackConfig::queue_capacity_frames * channels` total
  /// samples; the cap is enforced in [`AudioPlayer::write_samples`].
  ///
  /// `Mutex` (not `parking_lot::Mutex`, not lock-free `ringbuf`) is
  /// chosen for A11 because:
  /// - cpal's audio thread takes the lock for a single
  ///   `pop_front`-loop per callback (microseconds at typical 64-1024
  ///   frame callback buffers),
  /// - the producer holds the lock only across `extend` +
  ///   capacity-check arithmetic,
  /// - a future migration to `ringbuf` (one of the cpal docs'
  ///   recommended low-latency choices) is a local refactor behind
  ///   the same trait surface if profiling shows the lock matters.
  queue: Mutex<VecDeque<f32>>,
  /// Bound on `queue.lock().unwrap().len()`; computed once from
  /// `PlaybackConfig::queue_capacity_frames * channels.count()` so
  /// the producer doesn't recompute it per `write_samples` call.
  queue_capacity_samples: usize,
  /// Current state. Loaded by the cpal callback on every invocation
  /// (single atomic load is the lightweight check that gates the
  /// pop loop); written by the producer (`start`, `pause`, `resume`,
  /// `stop`).
  state: AtomicU8,
  /// **One-way terminal latch**, set by [`AudioPlayer::stop`] and
  /// never cleared. Independent of [`SharedState::state`] (which is a
  /// tri-state playback flag the cpal callback gates on every
  /// invocation) because the terminal contract is asymmetric: once
  /// `stop()` returns, NO subsequent `start()` / `pause()` /
  /// `resume()` / `write_samples()` may "rehydrate" the player to a
  /// live state and accept further producer writes. Without this
  /// separate latch, `start()` storing `STATE_RUNNING` after `stop()`
  /// would mask the terminated condition from the producer-side
  /// `STATE_STOPPED` gate in `write_samples()` and let post-stop
  /// chunks accumulate + play on the re-started cpal stream — a
  /// silent violation of [`super::output_stream::AudioOutputStream::stop`]'s
  /// "MUST NOT silently accept post-stop writes" clause.
  terminated: AtomicBool,
  /// Current volume scalar, stored as `f32::to_bits` in an
  /// `AtomicU32`. Read by the cpal callback every sample; written
  /// by [`AudioPlayer::store_volume`]. Default is 1.0 (unity gain) —
  /// matches Swift's `AVAudioPlayerNode.volume` default.
  volume_bits: AtomicU32,
  /// Captured first error from the cpal stream's `err_fn`. The
  /// callback can't bubble up `Result`, so we stash it here and
  /// surface it on the next producer call (`write_samples`, `flush`,
  /// `pause`, `resume`).
  ///
  /// `Mutex<Option<String>>` (string-typed, not `Error`-typed) so
  /// errors aren't lost if multiple device events fire — we keep the
  /// first one. Cleared by [`AudioPlayer::stop`].
  callback_error: Mutex<Option<String>>,
}

impl SharedState {
  /// Build a [`SharedState`] with a queue pre-allocated to its full
  /// `queue_capacity_samples` bound. Pre-allocation at construction
  /// (rather than on first producer write) keeps the producer-loop
  /// lock window in [`AudioPlayer::write_samples`] to a pure
  /// `VecDeque::extend` (O(chunk) memcpy with NO realloc possible),
  /// so the cpal callback's `try_lock` window can't be inflated by
  /// allocator time on a growing queue.
  ///
  /// # Errors
  /// - [`Error::Backend`] if `try_reserve_exact` fails on the bounded
  ///   queue capacity (e.g. tiny-RAM or fragmented-heap on the
  ///   caller's host).
  fn new(queue_capacity_samples: usize) -> Result<Self> {
    let mut queue = VecDeque::new();
    queue
      .try_reserve_exact(queue_capacity_samples)
      .map_err(|e| Error::Backend {
        message: format!(
          "AudioPlayer::with_device failed to pre-allocate queue capacity \
           ({queue_capacity_samples} samples): {e}"
        ),
      })?;
    Ok(Self {
      queue: Mutex::new(queue),
      queue_capacity_samples,
      state: AtomicU8::new(STATE_STOPPED),
      terminated: AtomicBool::new(false),
      volume_bits: AtomicU32::new(1.0_f32.to_bits()),
      callback_error: Mutex::new(None),
    })
  }

  #[inline(always)]
  fn load_volume(&self) -> f32 {
    f32::from_bits(self.volume_bits.load(Ordering::Relaxed))
  }

  /// Unconditionally drain the producer-visible state (queue +
  /// captured callback error) under poison-recovering locks. Called
  /// by [`AudioPlayer::stop`] AFTER the latch + state writes and
  /// AFTER the cpal `Stream::pause()` attempt (whose result is
  /// captured separately) — see the comment on `stop()` for why this
  /// must run even when pause fails.
  ///
  /// Poison-recover (`into_inner`) on both locks: a panicked cpal
  /// callback could have poisoned either, and on stop we MUST still
  /// be able to clear them so a subsequent observer (e.g. the
  /// integration test's `buffer_depth()`) sees an empty queue and
  /// the post-stop producer-call gate doesn't surface a stale
  /// captured error.
  fn stop_cleanup(&self) {
    {
      let mut q = match self.queue.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
      };
      q.clear();
    }
    {
      let mut e = match self.callback_error.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
      };
      *e = None;
    }
  }
}

/// Cpal-backed device player.
///
/// See the module-level docs for the cpal-callback + buffer-queue
/// plumbing diagram and the explicit list of Swift-side capabilities
/// A11 scopes out (input, file I/O, format conversion, crossfade,
/// per-buffer completions, `@Published` properties).
pub struct AudioPlayer {
  /// The cpal output stream. `None` only between
  /// [`AudioPlayer::stop`] + `Drop` (we tear the stream down on
  /// `stop` so we can rebuild a fresh one on a subsequent `start`,
  /// matching Swift's `startStreaming` ↔ `stopStreaming` lifecycle).
  ///
  /// The current implementation builds the stream once at
  /// construction time and re-uses it for the full lifetime of the
  /// player (cpal Streams support `play()` / `pause()`); we still
  /// keep this `Option` so `Drop` can take + drop it explicitly
  /// before the `SharedState` so the cpal callback thread is joined
  /// while the queue + atomics are still live.
  ///
  /// `cpal::Stream` is `Send + Sync` (per cpal 0.17.x docs) so the
  /// `AudioPlayer` can cross thread boundaries — the A8 pipeline can
  /// drive a player from any thread.
  stream: Option<Stream>,
  /// Shared callback + producer state. See [`SharedState`].
  shared: Arc<SharedState>,
  /// Stored config; consulted by [`AudioPlayer::config`] introspection
  /// + the [`AudioOutputStream`] impl.
  config: PlaybackConfig,
}

impl AudioPlayer {
  /// Build an [`AudioPlayer`] bound to the default output device on
  /// the default cpal host. Mirrors the Swift
  /// `AudioPlayer.startStreaming(sampleRate:)` entry point (which
  /// implicitly uses `AVAudioEngine`'s default output node).
  ///
  /// The cpal stream is **built but not started** — call
  /// [`AudioPlayer::start`] before pushing samples. This matches the
  /// Swift split between `startStreaming` (engine prep) and
  /// `playerNode.play()` (actual playback).
  ///
  /// # Errors
  /// - [`Error::Backend`] if cpal has no default host, no default
  ///   output device, the config rejects, or the cpal stream build
  ///   fails (CoreAudio init failure, unsupported sample rate,
  ///   etc.).
  pub fn new(config: PlaybackConfig) -> Result<Self> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or_else(|| Error::Backend {
      message: "AudioPlayer: no default cpal output device available".to_string(),
    })?;
    Self::with_device(&device, config)
  }

  /// Build an [`AudioPlayer`] bound to an explicit cpal device.
  /// Useful when the caller has already enumerated cpal devices and
  /// wants to target a specific one (the Swift API has no direct
  /// analog — `AVAudioEngine` always uses the system default — but
  /// cpal's multi-device support is a natural extension here).
  ///
  /// # Errors
  /// - [`Error::Backend`] if [`PlaybackConfig::cpal_config`] rejects
  ///   the config (zero channels) or the cpal stream build fails.
  pub fn with_device(device: &cpal::Device, config: PlaybackConfig) -> Result<Self> {
    if !matches!(config.sample_format(), SampleFormat::F32) {
      return Err(Error::Backend {
        message: format!(
          "AudioPlayer: only SampleFormat::F32 is currently supported (got {:?}); \
           non-F32 device negotiation is reserved for a follow-up",
          config.sample_format()
        ),
      });
    }

    let stream_config = config.cpal_config()?;

    let queue_capacity_samples = config
      .queue_capacity_frames()
      .checked_mul(usize::from(config.channels().count()))
      .ok_or_else(|| Error::Backend {
        message: "AudioPlayer: queue_capacity_frames * channels overflows usize".to_string(),
      })?;

    let shared = Arc::new(SharedState::new(queue_capacity_samples)?);

    // cpal callback (audio I/O thread). Pulls from the queue, scales
    // by current volume, writes silence on underrun. Cloned `Arc`
    // moved into the `'static` closure cpal requires.
    let cb_shared = Arc::clone(&shared);
    let data_callback = move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
      let state = cb_shared.state.load(Ordering::Acquire);
      if state != STATE_RUNNING {
        // Paused / stopped — emit silence. (Cpal pauses the
        // callback on `Stream::pause()`, but the producer may also
        // toggle our `state` flag; the dual gate is intentional.)
        for s in out.iter_mut() {
          *s = 0.0;
        }
        return;
      }
      let volume = cb_shared.load_volume();
      // Single short critical section: drain into the cpal buffer.
      // We don't hold the lock across `*s = ...` arithmetic outside
      // this scope.
      //
      // `try_lock` (not blocking `lock`): the callback runs on the
      // real-time audio I/O thread and MUST NOT block on a
      // producer-held mutex past the device callback deadline. On
      // contention, emit silence for this period — the cost is one
      // underrun, the alternative (blocking) is unbounded latency
      // that compounds across device periods. See the module-level
      // `## Concurrency` doc-comment for the full contract.
      let mut q = match cb_shared.queue.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::WouldBlock) => {
          // Producer holds the lock; emit silence rather than block
          // past the device deadline.
          for s in out.iter_mut() {
            *s = 0.0;
          }
          return;
        }
        Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
      };
      let drain_n = out.len().min(q.len());
      for slot in out.iter_mut().take(drain_n) {
        // pop_front is O(1) for VecDeque; the loop is the cpal
        // equivalent of the Swift `AVAudioPCMBuffer` per-buffer copy.
        let sample = q.pop_front().unwrap_or(0.0);
        *slot = sample * volume;
      }
      // Drop the lock before zeroing the tail — silence-on-underrun
      // doesn't need the queue.
      drop(q);
      for slot in out.iter_mut().skip(drain_n) {
        *slot = 0.0;
      }
    };

    // cpal `err_fn`. Stash the first error; surface it on the next
    // producer call. We don't have a logger dep in mlxrs, so silent
    // capture is the chosen behavior (the producer will see it).
    let err_shared = Arc::clone(&shared);
    let err_callback = move |err: StreamError| {
      let mut slot = match err_shared.callback_error.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
      };
      if slot.is_none() {
        *slot = Some(format!("cpal stream error: {err}"));
      }
    };

    let stream = device
      .build_output_stream(&stream_config, data_callback, err_callback, None)
      .map_err(|e| Error::Backend {
        message: format!("AudioPlayer: cpal build_output_stream failed: {e}"),
      })?;

    Ok(Self {
      stream: Some(stream),
      shared,
      config,
    })
  }

  /// The config the player was built with. Returns by value (Copy).
  #[inline(always)]
  #[must_use]
  pub fn config(&self) -> PlaybackConfig {
    self.config
  }

  /// Number of samples currently queued for playback (the cpal
  /// equivalent of Swift's `queuedBuffers * buffer.frameLength` sum,
  /// in samples not frames).
  #[inline(always)]
  #[must_use]
  pub fn buffer_depth(&self) -> usize {
    match self.shared.queue.lock() {
      Ok(g) => g.len(),
      Err(poisoned) => poisoned.into_inner().len(),
    }
  }

  /// `true` if [`AudioPlayer::start`] has been called and neither
  /// [`AudioPlayer::pause`] nor [`AudioPlayer::stop`] has run since.
  /// Mirrors the Swift `isPlaying` getter on the streaming branch.
  #[inline(always)]
  #[must_use]
  pub fn is_running(&self) -> bool {
    self.shared.state.load(Ordering::Acquire) == STATE_RUNNING
  }

  /// `true` if the player is in [`STATE_PAUSED`] (cpal stream is
  /// `pause()`d, queue retains samples; Swift's `playerNode.pause()`
  /// branch of `AudioPlayer.pause()`).
  #[inline(always)]
  #[must_use]
  pub fn is_paused(&self) -> bool {
    self.shared.state.load(Ordering::Acquire) == STATE_PAUSED
  }

  /// Current output volume, default 1.0. Mirrors
  /// `AVAudioPlayerNode.volume`.
  #[inline(always)]
  #[must_use]
  pub fn volume(&self) -> f32 {
    self.shared.load_volume()
  }

  /// Atomically store the output volume. Clamped to `[0.0, 1.0]` — values
  /// outside the range are clamped silently (matches the
  /// `AVAudioPlayerNode.volume` 0..1 documented range).
  ///
  /// **Non-finite inputs (NaN, ±∞) are mapped to `0.0`** rather than
  /// propagated. `f32::clamp` preserves NaN bits, which would cause
  /// the callback's `sample * volume` arithmetic to emit NaN samples
  /// (audible as full-scale noise on most DACs); silent safe-default
  /// matches the unsigned-clamp idiom used elsewhere in mlxrs for
  /// user-supplied scalars.
  ///
  /// Named `store_volume` (not `set_volume`) to signal atomic-write
  /// semantics rather than fluent-builder semantics — this is a global
  /// side-effect on the player, not a `&mut self` field setter.
  ///
  /// Takes `&self` (not `&mut self`) so the volume can be adjusted
  /// concurrently with [`AudioPlayer::write_samples`] without
  /// shadowing the producer borrow — useful when a UI thread tweaks
  /// volume while a worker thread is pumping samples.
  pub fn store_volume(&self, vol: f32) {
    let sanitized = sanitize_volume(vol);
    self
      .shared
      .volume_bits
      .store(sanitized.to_bits(), Ordering::Release);
  }

  /// Start the cpal stream — samples written via
  /// [`AudioPlayer::write_samples`] start flowing to the device.
  /// Mirrors the Swift `playerNode.play()` call inside
  /// `startStreaming`.
  ///
  /// Idempotent: calling `start` on a running player is a no-op
  /// (returns `Ok(())`). Calling `start` after `pause` resumes
  /// playback (equivalent to [`AudioPlayer::resume`]).
  ///
  /// **Terminal-state contract.** [`AudioPlayer::stop`] is a
  /// one-way terminal transition; once `stop()` returns, `start()`
  /// rejects with [`Error::Backend`] ("...called on terminated
  /// player...") rather than re-arming the producer surface. The
  /// caller MUST construct a fresh [`AudioPlayer`] to resume
  /// playback. The cpal stream is preserved across `stop()` only so
  /// `Drop` can join the I/O thread cleanly — it is NOT a hook for
  /// restarting the producer pipeline.
  ///
  /// # Errors
  /// - [`Error::Backend`] if [`AudioPlayer::stop`] has already been
  ///   called on this player (one-way terminal latch).
  /// - [`Error::Backend`] if the cpal `Stream::play()` call fails,
  ///   or if the stream has already been dropped by a prior `stop`.
  pub fn start(&mut self) -> Result<()> {
    // FIX 1: one-way terminal latch. Checked FIRST so a post-stop
    // `start()` doesn't re-arm `state = STATE_RUNNING` and let
    // subsequent `write_samples()` slip past its `STATE_STOPPED`
    // gate. Acquire-load pairs with the Release-store in `stop()`.
    if self.shared.terminated.load(Ordering::Acquire) {
      return Err(Error::Backend {
        message: "AudioPlayer::start called on terminated player — construct a new AudioPlayer"
          .to_string(),
      });
    }
    self.take_callback_error()?;
    let stream = self.stream.as_ref().ok_or_else(|| Error::Backend {
      message: "AudioPlayer::start: stream has been dropped (post-stop)".to_string(),
    })?;
    stream.play().map_err(|e| Error::Backend {
      message: format!("AudioPlayer::start: cpal play() failed: {e}"),
    })?;
    self.shared.state.store(STATE_RUNNING, Ordering::Release);
    Ok(())
  }

  /// Pause playback. The cpal stream is `pause()`d and the queue
  /// retains its samples; subsequent [`AudioPlayer::write_samples`]
  /// calls still buffer into the queue but no audio is emitted.
  /// Mirrors `MLXAudioCore.AudioPlayer.pause()` (streaming branch).
  ///
  /// **Terminal-state contract.** Rejects on a terminated player
  /// (see [`AudioPlayer::start`] / [`AudioPlayer::stop`]).
  ///
  /// # Errors
  /// - [`Error::Backend`] if [`AudioPlayer::stop`] has already been
  ///   called on this player (one-way terminal latch).
  /// - [`Error::Backend`] if the cpal `Stream::pause()` call fails.
  pub fn pause(&mut self) -> Result<()> {
    if self.shared.terminated.load(Ordering::Acquire) {
      return Err(Error::Backend {
        message: "AudioPlayer::pause called on terminated player — construct a new AudioPlayer"
          .to_string(),
      });
    }
    self.take_callback_error()?;
    let stream = self.stream.as_ref().ok_or_else(|| Error::Backend {
      message: "AudioPlayer::pause: stream has been dropped (post-stop)".to_string(),
    })?;
    stream.pause().map_err(|e| Error::Backend {
      message: format!("AudioPlayer::pause: cpal pause() failed: {e}"),
    })?;
    self.shared.state.store(STATE_PAUSED, Ordering::Release);
    Ok(())
  }

  /// Resume from [`AudioPlayer::pause`]. Mirrors Swift's
  /// `togglePlayPause()` resuming branch (`playerNode.play()` +
  /// `isPlaying = true`).
  ///
  /// **Terminal-state contract.** Rejects on a terminated player
  /// (see [`AudioPlayer::start`] / [`AudioPlayer::stop`]). The
  /// dedicated `resume`-named error message keeps the call-site
  /// signal clear (the producer that called `resume()` after a
  /// stop got the same one-way-latch rejection `start()` would
  /// have surfaced).
  ///
  /// # Errors
  /// - [`Error::Backend`] if [`AudioPlayer::stop`] has already been
  ///   called on this player (one-way terminal latch).
  /// - [`Error::Backend`] if the cpal `Stream::play()` call fails.
  pub fn resume(&mut self) -> Result<()> {
    if self.shared.terminated.load(Ordering::Acquire) {
      return Err(Error::Backend {
        message: "AudioPlayer::resume called on terminated player — construct a new AudioPlayer"
          .to_string(),
      });
    }
    self.start()
  }

  /// Stop playback immediately. Drops every queued sample, pauses
  /// the cpal stream, and clears any captured callback error.
  /// Mirrors `stopStreaming()`.
  ///
  /// **Terminal-state contract.** `stop()` is a **one-way terminal
  /// transition**: after it returns, every subsequent producer-side
  /// method ([`AudioPlayer::start`], [`AudioPlayer::pause`],
  /// [`AudioPlayer::resume`], [`AudioPlayer::write_samples`])
  /// rejects with [`Error::Backend`] containing
  /// "terminated"/"after stop()" so late producer chunks MUST NOT
  /// accumulate silently and replay on a later `start()` — honoring
  /// the [`super::output_stream::AudioOutputStream::stop`] contract.
  /// The one-way latch (`SharedState::terminated`) is checked BEFORE
  /// `state` on every entry so a `start(); stop(); start();`
  /// sequence cannot re-arm `state = STATE_RUNNING` and slip
  /// post-stop writes past the producer gate.
  ///
  /// The cpal stream is preserved across `stop()` only so `Drop` can
  /// join the I/O thread cleanly — it is NOT a hook for restarting
  /// the producer pipeline. The caller MUST construct a fresh
  /// [`AudioPlayer`] to resume playback. Pause is the soft-state
  /// alternative (see [`AudioPlayer::pause`] — pause-state writes
  /// still buffer for [`AudioPlayer::resume`]).
  ///
  /// `stop()` itself is idempotent and ALWAYS succeeds at moving
  /// the player to the terminated state — even if the underlying
  /// cpal `Stream::pause()` returns an error, the latch is set
  /// FIRST so re-entry on a half-stopped player is consistently
  /// rejected.
  ///
  /// # Errors
  /// - [`Error::Backend`] if the cpal `Stream::pause()` call fails.
  ///   Note: the terminal latch is already set by this point, so
  ///   subsequent producer calls still reject correctly.
  pub fn stop(&mut self) -> Result<()> {
    // Set the one-way terminal latch FIRST (and unconditionally).
    // Any subsequent producer-side call (including a re-entrant
    // `stop()` on a poisoned half-stopped player) checks this latch
    // BEFORE `state`, so the terminal contract holds even if the
    // cpal `Stream::pause()` below returns an error. Release-store
    // pairs with Acquire-loads in `start` / `pause` / `resume` /
    // `write_samples`.
    self.shared.terminated.store(true, Ordering::Release);
    self.shared.state.store(STATE_STOPPED, Ordering::Release);

    // Capture the cpal pause result WITHOUT `?`-propagating — we
    // must still run the unconditional queue + callback-error
    // cleanup below even if pause fails. Pre-R3, an `?` here would
    // skip cleanup: the latch + state were terminated but the
    // VecDeque still held queued samples, so the cpal callback
    // (running until `Drop`) would keep emitting silence-on-stopped
    // while the queue lingered. The R3 contract is: stop() always
    // tears down the queue + error slot; the pause error (if any)
    // is reported AFTER the cleanup runs.
    let pause_result = if let Some(stream) = self.stream.as_ref() {
      stream.pause().map_err(|e| Error::Backend {
        message: format!("AudioPlayer::stop: cpal pause() failed: {e}"),
      })
    } else {
      Ok(())
    };

    // UNCONDITIONAL queue + callback-error cleanup. Poison-recover
    // (into_inner) rather than fail-silent on a poisoned lock so a
    // panicked callback can't leave stale samples / errors behind
    // post-stop. This MUST run regardless of the pause result above.
    self.shared.stop_cleanup();

    pause_result
  }

  /// Push interleaved PCM samples into the playback queue. Returns
  /// the number of samples accepted (`= samples.len()` on success).
  ///
  /// Surfaces a pending callback error (cpal `err_fn` capture) if
  /// one is queued — the next producer call after a device error
  /// receives the error report.
  ///
  /// Internally splits `samples` into [`WRITE_CHUNK_MAX`]-sized
  /// inner-loop chunks, each taking the queue lock for its own
  /// `extend`. This bounds the duration the audio callback's
  /// `try_lock` would have to wait on the producer to ~one chunk's
  /// extend (microseconds), so a multi-second producer payload
  /// can't stall the cpal callback past the device deadline. See
  /// the module-level `## Concurrency` doc-comment.
  ///
  /// # Errors
  /// - [`Error::Backend`] if [`AudioPlayer::stop`] has been called
  ///   on this player. `stop()` is a one-way terminal latch — writes
  ///   after stop are rejected outright to honor the
  ///   [`super::output_stream::AudioOutputStream::stop`] contract
  ///   ("any queued samples MUST be dropped; subsequent
  ///   `write_samples` calls MUST return `Err`"). Pause-state writes
  ///   are still accepted (they buffer for `resume`).
  /// - [`Error::Backend`] if the queue would overflow
  ///   [`PlaybackConfig::queue_capacity_frames`] × channel count.
  ///   The write is rejected wholesale — no partial accept on
  ///   overflow (the caller has no way to know how many fit, and a
  ///   partial accept would invite torn audio at the chunk
  ///   boundary). The overflow check uses the total payload length
  ///   so chunking can't mask the cap.
  /// - [`Error::Backend`] if a prior cpal callback error was
  ///   captured.
  pub fn write_samples(&mut self, samples: &[f32]) -> Result<usize> {
    // FIX 1: one-way terminal latch FIRST, before reading `state`.
    // A naive `state == STATE_STOPPED` gate is insufficient because
    // `start()` unconditionally re-arms `state = STATE_RUNNING`; the
    // sequence `start(); stop(); start(); write_samples(...)` would
    // otherwise slip past the gate and silently replay post-stop
    // chunks on the re-armed stream. Acquire-load pairs with the
    // Release-store in `stop()`.
    if self.shared.terminated.load(Ordering::Acquire) {
      return Err(Error::Backend {
        message: "AudioPlayer::write_samples called after stop() — player is terminated"
          .to_string(),
      });
    }
    self.take_callback_error()?;

    // Pre-`start()` writes are also rejected — the cpal stream isn't
    // play()ing yet, so accumulating samples here would replay on
    // the first `start()` rather than start cleanly.
    let state = self.shared.state.load(Ordering::Acquire);
    if state == STATE_STOPPED {
      return Err(Error::Backend {
        message: "AudioPlayer::write_samples called after stop()".to_string(),
      });
    }

    // Whole-payload overflow check up front (one lock acquisition).
    // Chunking is purely a concurrency-shaping concern; it must not
    // weaken the cap.
    {
      let q = match self.shared.queue.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
      };
      let projected_len = q
        .len()
        .checked_add(samples.len())
        .ok_or_else(|| Error::Backend {
          message: "AudioPlayer::write_samples: queue length + new samples overflows usize"
            .to_string(),
        })?;
      if projected_len > self.shared.queue_capacity_samples {
        return Err(Error::Backend {
          message: format!(
            "AudioPlayer::write_samples: queue overflow (capacity {} samples, current {} \
             samples, tried to push {})",
            self.shared.queue_capacity_samples,
            q.len(),
            samples.len()
          ),
        });
      }
    }

    // FIX 2: per-chunk lock acquisition with NO per-chunk
    // `reserve_exact`. The queue is pre-allocated to
    // `queue_capacity_samples` at construction (see
    // `SharedState::new` via `try_reserve_exact`), and the
    // whole-payload overflow check above guarantees
    // `q.len() + samples.len() <= queue_capacity_samples` — so
    // `VecDeque::extend` here is a pure O(chunk.len()) memcpy with
    // NO realloc possible. Allocator time CANNOT inflate the lock
    // window and the cpal callback's `try_lock` window stays
    // bounded by the per-chunk `extend` duration (microseconds at
    // WRITE_CHUNK_MAX = 4096 samples).
    for chunk in samples.chunks(WRITE_CHUNK_MAX) {
      let mut q = match self.shared.queue.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
      };
      q.extend(chunk.iter().copied());
    }
    Ok(samples.len())
  }

  /// Block until the playback queue has drained. The cpal callback
  /// continues to consume samples while this method polls; when the
  /// queue empties, [`AudioPlayer::flush`] returns. Mirrors Swift's
  /// `finishStreamingInput()` → `finishStreamIfDrained()` path.
  ///
  /// The implementation is a bounded poll loop (10ms granularity,
  /// 30s timeout) — cpal has no per-buffer-completion hook so we
  /// can't park on a condvar tied to the callback. The poll cadence
  /// matches Swift's `Timer.scheduledTimer(withTimeInterval: 0.1)`
  /// order of magnitude; the timeout prevents an indefinite block
  /// on a stalled device.
  ///
  /// If the player is not [`STATE_RUNNING`] (stopped or paused) and
  /// the queue is non-empty, this method returns immediately with a
  /// [`Error::Backend`] — flushing a stopped/paused player would
  /// block forever (the callback doesn't drain unless running).
  ///
  /// # Errors
  /// - [`Error::Backend`] if the flush times out, the player is not
  ///   running and the queue is non-empty, or a cpal callback error
  ///   surfaced mid-drain.
  pub fn flush(&mut self) -> Result<()> {
    self.take_callback_error()?;
    let start = Instant::now();
    loop {
      let depth = self.buffer_depth();
      if depth == 0 {
        return Ok(());
      }
      let state = self.shared.state.load(Ordering::Acquire);
      if state != STATE_RUNNING {
        return Err(Error::Backend {
          message: format!(
            "AudioPlayer::flush: queue has {depth} samples but state is {state} (not running) — \
             call start() before flush()"
          ),
        });
      }
      if start.elapsed() > FLUSH_TIMEOUT {
        return Err(Error::Backend {
          message: format!(
            "AudioPlayer::flush: timed out after {:?} with {depth} samples still queued",
            FLUSH_TIMEOUT
          ),
        });
      }
      thread::sleep(FLUSH_POLL_INTERVAL);
      self.take_callback_error()?;
    }
  }

  /// Pull the captured cpal `err_fn` message (if any) and surface it
  /// as a [`Error::Backend`]. Called at the head of every public
  /// producer method.
  fn take_callback_error(&self) -> Result<()> {
    let mut slot = match self.shared.callback_error.lock() {
      Ok(g) => g,
      Err(poisoned) => poisoned.into_inner(),
    };
    if let Some(msg) = slot.take() {
      return Err(Error::Backend { message: msg });
    }
    Ok(())
  }
}

impl AudioOutputStream for AudioPlayer {
  fn write_samples(&mut self, samples: &[f32]) -> Result<usize> {
    AudioPlayer::write_samples(self, samples)
  }

  fn flush(&mut self) -> Result<()> {
    AudioPlayer::flush(self)
  }

  fn stop(&mut self) -> Result<()> {
    AudioPlayer::stop(self)
  }

  fn is_running(&self) -> bool {
    AudioPlayer::is_running(self)
  }
}

impl Drop for AudioPlayer {
  fn drop(&mut self) {
    // Mark stopped first so the callback sees STATE_STOPPED on its
    // next invocation and stops draining.
    self.shared.state.store(STATE_STOPPED, Ordering::Release);
    // Drop the stream explicitly. `cpal::Stream`'s `Drop` joins the
    // I/O thread (so the data callback is guaranteed dead after
    // this line); doing it explicitly + first means the callback
    // can't observe a half-dropped `SharedState`.
    if let Some(stream) = self.stream.take() {
      // Best-effort pause before drop — on macOS CoreAudio,
      // `Stream::drop` already stops the unit, but pausing first
      // avoids one extra callback hit on `STATE_STOPPED` silence.
      let _ = stream.pause();
      drop(stream);
    }
  }
}

#[cfg(test)]
mod tests {
  //! In-crate unit tests for `AudioPlayer` invariants that need
  //! access to private state (`SharedState::queue`,
  //! `SharedState::queue_capacity_samples`, `SharedState::terminated`,
  //! `SharedState::callback_error`).
  //!
  //! These were previously reachable from `tests/audio_playback.rs`
  //! via `pub #[doc(hidden)] _test_*` accessors on `AudioPlayer`,
  //! which leaked into the crate's public API surface (downstream
  //! crates could call them in release builds, and removing them
  //! later would be a SemVer break). Moving the tests here keeps the
  //! private invariant under in-crate test coverage without any
  //! `pub` accessor.
  //!
  //! Device-touching tests (those that call `AudioPlayer::new`,
  //! which opens a cpal output stream) remain
  //! `#[ignore = "requires real default audio output device"]` so CI
  //! without an audio device still passes — they run locally via
  //! `cargo test --features audio -- --ignored`.
  use super::*;
  use crate::audio::playback::config::{ChannelLayout, PlaybackConfig, SampleFormat};

  /// F1 (R3 MEDIUM) — `stop()`'s unconditional cleanup path:
  /// `SharedState::stop_cleanup` MUST drain a non-empty queue + any
  /// captured callback error to None, regardless of how it was
  /// invoked. This is the cleanup branch `AudioPlayer::stop` runs
  /// AFTER capturing the cpal pause result; if pause errs, this
  /// still runs (the `Result` is returned at the end, not via `?`).
  ///
  /// Constructs `SharedState` directly (no cpal stream needed) so
  /// the test runs in CI without an audio device — this is the
  /// "struct mock" path the R3 spec calls for when injecting a real
  /// cpal pause failure is impractical (cpal `Stream::pause()` on
  /// macOS CoreAudio doesn't err on a healthy device, and there's
  /// no public hook to swap in a faulty stream).
  #[test]
  fn shared_state_stop_cleanup_drains_queue_and_clears_error_unconditionally() {
    let shared = SharedState::new(4096).expect("pre-allocate test queue");

    // Pre-populate the queue + callback_error to non-empty so the
    // cleanup's effect is observable.
    {
      let mut q = shared.queue.lock().unwrap();
      q.extend([0.1_f32, 0.2, 0.3, 0.4]);
    }
    {
      let mut e = shared.callback_error.lock().unwrap();
      *e = Some("simulated prior cpal err_fn capture".to_string());
    }

    // Mirror `AudioPlayer::stop`'s ordering: latch FIRST, then
    // state, then (in stop() — captured pause result, then) the
    // unconditional cleanup. We exercise the cleanup branch
    // directly here.
    shared.terminated.store(true, Ordering::Release);
    shared.state.store(STATE_STOPPED, Ordering::Release);
    shared.stop_cleanup();

    // Latch + state were set before cleanup — confirm they survive.
    assert!(
      shared.terminated.load(Ordering::Acquire),
      "terminated latch must be set (stop ordering: latch -> state -> cleanup)"
    );
    assert_eq!(
      shared.state.load(Ordering::Acquire),
      STATE_STOPPED,
      "state must be STOPPED post-stop"
    );

    // The unconditional cleanup MUST have drained the queue + cleared the
    // captured callback error — both observable to the producer-side gates
    // (`buffer_depth()` reads `queue.len()`; `take_callback_error()` reads
    // `callback_error`).
    assert_eq!(
      shared.queue.lock().unwrap().len(),
      0,
      "stop_cleanup must drain the queue unconditionally (R3: this branch \
       runs even when cpal pause errs — pre-R3, an early `?` on pause would \
       skip this and leave samples lingering until Drop)"
    );
    assert!(
      shared.callback_error.lock().unwrap().is_none(),
      "stop_cleanup must clear captured callback_error unconditionally"
    );
  }

  /// F1 corollary: `stop_cleanup` is also poison-safe — a panicked
  /// callback that poisoned the queue or callback_error lock must
  /// not prevent stop from draining. We simulate poisoning by
  /// running a closure that panics while holding the lock.
  #[test]
  fn shared_state_stop_cleanup_recovers_from_poisoned_locks() {
    use std::{panic, sync::Arc};

    let shared = Arc::new(SharedState::new(64).expect("pre-allocate test queue"));

    // Poison the queue lock from another thread by panicking while
    // holding it. The Mutex transitions to Poisoned state; subsequent
    // `lock()` returns `Err(PoisonError)`.
    let queue_poisoner = Arc::clone(&shared);
    let _ = std::thread::spawn(move || {
      let _g = queue_poisoner.queue.lock().unwrap();
      panic!("simulated callback panic poisoning queue lock");
    })
    .join();

    // Poison the callback_error lock similarly.
    let err_poisoner = Arc::clone(&shared);
    let _ = std::thread::spawn(move || {
      let _g = err_poisoner.callback_error.lock().unwrap();
      panic!("simulated callback panic poisoning callback_error lock");
    })
    .join();

    assert!(
      shared.queue.is_poisoned(),
      "test setup: queue lock should be poisoned"
    );
    assert!(
      shared.callback_error.is_poisoned(),
      "test setup: callback_error lock should be poisoned"
    );

    // Now exercise stop_cleanup: it MUST NOT propagate the
    // PoisonError (no `unwrap`) and MUST still drain.
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
      shared.stop_cleanup();
    }));
    assert!(
      result.is_ok(),
      "stop_cleanup must not panic on poisoned locks (R3: poison-recover via into_inner)"
    );
  }

  /// F2 (R3 MEDIUM, moved from `tests/audio_playback.rs`) — queue
  /// is pre-allocated to its full `queue_capacity_samples` bound at
  /// construction time, NOT on first producer write. Asserts the
  /// `try_reserve_exact` contract ("AT LEAST `additional` more").
  ///
  /// Lives in-crate so it can read the private
  /// `shared.queue.capacity()` + `shared.queue_capacity_samples`
  /// without needing the previously-leaked `pub _test_*` accessors.
  /// Device-gated: `AudioPlayer::new` opens a cpal output stream.
  #[cfg(target_os = "macos")]
  #[test]
  #[ignore = "requires real default audio output device"]
  fn audio_player_pre_allocates_queue_capacity_at_construction() {
    let cfg = PlaybackConfig::new(16_000, ChannelLayout::Mono, SampleFormat::F32)
      .with_queue_capacity_frames(4096);
    let player = AudioPlayer::new(cfg).unwrap();

    assert_eq!(player.buffer_depth(), 0);
    let cap_samples = player.shared.queue_capacity_samples;
    assert_eq!(
      cap_samples, 4096,
      "queue cap = frames * channels = 4096 * 1"
    );
    let underlying = match player.shared.queue.lock() {
      Ok(g) => g.capacity(),
      Err(poisoned) => poisoned.into_inner().capacity(),
    };
    assert!(
      underlying >= cap_samples,
      "VecDeque underlying capacity ({underlying}) must be >= bounded cap ({cap_samples}) per \
       try_reserve_exact contract"
    );
  }

  /// F2 (R3 MEDIUM, moved from `tests/audio_playback.rs`) —
  /// producer-side `write_samples` MUST NOT grow the underlying
  /// VecDeque capacity. The bound is pre-allocated at construction,
  /// so `extend` is a pure O(chunk) memcpy with no realloc inside
  /// the producer's lock window (the cpal callback's `try_lock`
  /// can't be inflated by allocator time).
  #[cfg(target_os = "macos")]
  #[test]
  #[ignore = "requires real default audio output device"]
  fn audio_player_write_samples_does_not_grow_queue_capacity_during_playback() {
    let cfg = PlaybackConfig::new(16_000, ChannelLayout::Mono, SampleFormat::F32)
      .with_queue_capacity_frames(4096);
    let mut player = AudioPlayer::new(cfg).unwrap();
    player.start().unwrap();
    // pause() so the callback doesn't drain the just-pushed samples
    // and trigger a buffer-depth race in the post-write capacity
    // read.
    player.pause().unwrap();

    let cap_before = match player.shared.queue.lock() {
      Ok(g) => g.capacity(),
      Err(poisoned) => poisoned.into_inner().capacity(),
    };
    let cap_samples = player.shared.queue_capacity_samples;

    // Push 1024 samples — well under the 4096 bound, so `extend`
    // is a pure memcpy with no realloc.
    player.write_samples(&[0.25_f32; 1024]).unwrap();

    let cap_after = match player.shared.queue.lock() {
      Ok(g) => g.capacity(),
      Err(poisoned) => poisoned.into_inner().capacity(),
    };
    assert_eq!(
      cap_after, cap_before,
      "queue capacity grew during write_samples (before={cap_before}, after={cap_after}) — \
       producer-loop `extend` must not realloc because the queue is pre-allocated to \
       queue_capacity_samples ({cap_samples}) at construction"
    );

    let _ = player.stop();
  }
}
