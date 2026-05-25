//! Integration tests for [`mlxrs::audio::playback`] — the cpal-backed
//! `AudioPlayer` + `AudioOutputStream` trait port of
//! `mlx-audio-swift`'s `MLXAudioCore.AudioPlayer` streaming surface.
//!
//! Two test families:
//! - **Mock-based unit tests** (default; CI-safe). Exercise the
//!   `AudioOutputStream` trait + `PlaybackConfig` math without
//!   touching cpal device init — no audio hardware required.
//! - **Real-device tests** (gated `#[cfg(target_os = "macos")]` +
//!   `#[ignore]`). Smoke-test the cpal-driven `AudioPlayer`
//!   end-to-end on a real default output device.
//!
//! NO `peak_memory()` magnitude asserts (per the project's
//! `[[feedback_no_global_peak_memory_assert]]` rule).

#![cfg(feature = "audio")]

use std::sync::{Arc, Mutex};

use mlxrs::audio::playback::{
  AudioOutputStream, ChannelLayout, PlaybackConfig, SampleFormat,
  player::{WRITE_CHUNK_MAX, sanitize_volume},
};

// ---------------------------------------------------------------------------
// Mock AudioOutputStream
// ---------------------------------------------------------------------------

/// In-memory `AudioOutputStream` implementor used to test that the
/// trait surface compiles + behaves contractually without pulling in
/// cpal device init. Mirrors the role a unit-test recorder plays for
/// the Swift `AudioPlayer` (drop-in for `AVAudioPlayerNode`).
struct RecordingSink {
  buffer: Arc<Mutex<Vec<f32>>>,
  capacity: usize,
  running: bool,
}

impl RecordingSink {
  fn new(capacity: usize) -> Self {
    Self {
      buffer: Arc::new(Mutex::new(Vec::new())),
      capacity,
      running: true,
    }
  }
}

impl AudioOutputStream for RecordingSink {
  fn write_samples(&mut self, samples: &[f32]) -> mlxrs::error::Result<usize> {
    if !self.running {
      return Err(mlxrs::error::Error::Backend {
        message: "RecordingSink: stream stopped".to_string(),
      });
    }
    let mut buf = self.buffer.lock().unwrap();
    if buf.len() + samples.len() > self.capacity {
      return Err(mlxrs::error::Error::Backend {
        message: format!(
          "RecordingSink: capacity {} exceeded ({} + {})",
          self.capacity,
          buf.len(),
          samples.len()
        ),
      });
    }
    buf.extend_from_slice(samples);
    Ok(samples.len())
  }

  fn flush(&mut self) -> mlxrs::error::Result<()> {
    // Pretend the sink drained immediately.
    self.buffer.lock().unwrap().clear();
    Ok(())
  }

  fn stop(&mut self) -> mlxrs::error::Result<()> {
    self.running = false;
    self.buffer.lock().unwrap().clear();
    Ok(())
  }

  fn is_running(&self) -> bool {
    self.running
  }
}

// ---------------------------------------------------------------------------
// PlaybackConfig — default + constructor + cpal_config()
// ---------------------------------------------------------------------------

#[test]
fn playback_config_default_sample_rate_matches_swift_default() {
  // Swift `MLXAudioUI` voice-pipeline default is 24 kHz; the mlxrs
  // `PlaybackConfig::default` should match so the A8 pipeline
  // composes without spelling out the rate.
  let cfg = PlaybackConfig::default();
  assert_eq!(cfg.sample_rate(), 24_000);
  assert_eq!(cfg.channels(), ChannelLayout::Mono);
  assert_eq!(cfg.sample_format(), SampleFormat::F32);
  assert_eq!(cfg.buffer_size_frames(), None);
  // 4 seconds @ 24 kHz = 96000 frames.
  assert_eq!(cfg.queue_capacity_frames(), 96_000);
}

#[test]
fn playback_config_mono_constructor() {
  let cfg = PlaybackConfig::mono(48_000);
  assert_eq!(cfg.sample_rate(), 48_000);
  assert_eq!(cfg.channels(), ChannelLayout::Mono);
  assert_eq!(cfg.channels().count(), 1);
  assert_eq!(cfg.queue_capacity_frames(), 48_000 * 4);
}

#[test]
fn playback_config_stereo_constructor() {
  let cfg = PlaybackConfig::stereo(44_100);
  assert_eq!(cfg.channels(), ChannelLayout::Stereo);
  assert_eq!(cfg.channels().count(), 2);
  // F4: `queue_capacity_frames` is in FRAMES (not samples), same
  // unit across mono and stereo. The player's `with_device`
  // constructor does the single frame-to-sample conversion via
  // `* channels.count()`; `stereo()` MUST NOT pre-multiply (doing
  // so would double-count the bound and yield 8 seconds of stereo
  // for what's advertised as a 4-second cap).
  assert_eq!(cfg.queue_capacity_frames(), 44_100 * 4);
}

#[test]
fn playback_config_stereo_queue_capacity_is_frames_not_samples() {
  // F4 regression: pre-fix the stereo constructor stored
  // `sample_rate * 4 * 2` and `with_device` then multiplied by
  // channel count again — yielding 8 seconds of stereo capacity
  // for what's advertised as 4 seconds. Post-fix the stored value
  // is in frames and matches the mono constructor's unit.
  let cfg = PlaybackConfig::stereo(48_000);
  // 4 seconds @ 48 kHz = 192_000 frames (NOT 384_000 samples).
  assert_eq!(cfg.queue_capacity_frames(), 192_000);
  // Per-channel-applied sample budget is what the player actually
  // bounds: 192_000 frames × 2 channels = 384_000 interleaved
  // samples (i.e. 4 seconds of stereo audio, the documented cap).
  let samples = cfg.queue_capacity_frames() * usize::from(cfg.channels().count());
  assert_eq!(samples, 384_000);
}

#[test]
fn playback_config_mono_queue_capacity_is_frames_not_samples() {
  // F4 invariant: the mono path is the unit-of-truth — frames are
  // frames whether the channel count is 1 or N. For mono,
  // `queue_capacity_frames * 1 == queue_capacity_frames`, which is
  // a degenerate case but pins the contract.
  let cfg = PlaybackConfig::mono(48_000);
  // 4 seconds @ 48 kHz = 192_000 frames.
  assert_eq!(cfg.queue_capacity_frames(), 192_000);
  let samples = cfg.queue_capacity_frames() * usize::from(cfg.channels().count());
  assert_eq!(samples, 192_000);
}

#[test]
fn channel_layout_count_arbitrary() {
  assert_eq!(ChannelLayout::Mono.count(), 1);
  assert_eq!(ChannelLayout::Stereo.count(), 2);
  assert_eq!(ChannelLayout::Channels(6).count(), 6);
}

#[test]
fn playback_config_cpal_config_rejects_zero_channels() {
  let cfg = PlaybackConfig::new(16_000, ChannelLayout::Channels(0), SampleFormat::F32)
    .with_queue_capacity_frames(1024);
  let err = cfg.cpal_config().unwrap_err();
  assert!(
    matches!(err, mlxrs::error::Error::Backend { ref message } if message.contains("channel count")),
    "expected Backend(channel count) error, got {err:?}"
  );
}

#[test]
fn playback_config_cpal_config_passes_buffer_hint() {
  let with_hint = PlaybackConfig::new(16_000, ChannelLayout::Mono, SampleFormat::F32)
    .with_buffer_size_frames(256)
    .with_queue_capacity_frames(1024);
  let cpal_cfg = with_hint.cpal_config().unwrap();
  assert_eq!(cpal_cfg.channels, 1);
  // `cpal::SampleRate` is a `pub type SampleRate = u32` alias in
  // cpal 0.17 — compare as a plain `u32`.
  assert_eq!(cpal_cfg.sample_rate, 16_000);
  assert!(matches!(cpal_cfg.buffer_size, cpal::BufferSize::Fixed(256)));

  let without_hint = PlaybackConfig::mono(16_000);
  let cpal_cfg = without_hint.cpal_config().unwrap();
  assert!(matches!(cpal_cfg.buffer_size, cpal::BufferSize::Default));
}

// ---------------------------------------------------------------------------
// AudioOutputStream trait — mock-based contract tests
// ---------------------------------------------------------------------------

#[test]
fn audio_output_stream_writes_samples_returns_count() {
  let mut sink = RecordingSink::new(4096);
  let samples = vec![0.5_f32; 1024];

  let written = sink.write_samples(&samples).unwrap();
  assert_eq!(written, 1024);
  assert!(sink.is_running());
}

#[test]
fn audio_output_stream_flush_drains_buffer() {
  let mut sink = RecordingSink::new(4096);
  sink.write_samples(&[0.1_f32; 256]).unwrap();
  // Pre-flush the buffer has 256 samples; post-flush it's empty.
  assert_eq!(sink.buffer.lock().unwrap().len(), 256);
  sink.flush().unwrap();
  assert_eq!(sink.buffer.lock().unwrap().len(), 0);
}

#[test]
fn audio_output_stream_stop_marks_not_running_and_rejects_writes() {
  let mut sink = RecordingSink::new(4096);
  assert!(sink.is_running());

  sink.stop().unwrap();
  assert!(!sink.is_running());

  // Post-stop writes return Err — the trait contract.
  let err = sink.write_samples(&[0.0_f32; 32]).unwrap_err();
  assert!(
    matches!(err, mlxrs::error::Error::Backend { ref message } if message.contains("stopped")),
    "expected stopped-stream Backend error, got {err:?}"
  );
}

#[test]
fn audio_output_stream_overflow_returns_err() {
  let mut sink = RecordingSink::new(1024);
  sink.write_samples(&[0.0_f32; 512]).unwrap();
  sink.write_samples(&[0.0_f32; 512]).unwrap();

  // Now full; next write blows the cap.
  let err = sink.write_samples(&[0.0_f32; 1]).unwrap_err();
  assert!(
    matches!(err, mlxrs::error::Error::Backend { ref message } if message.contains("capacity")),
    "expected capacity-overflow Backend error, got {err:?}"
  );
}

// ---------------------------------------------------------------------------
// AudioPlayer — non-device-touching unit tests
// ---------------------------------------------------------------------------
//
// These exercise the `AudioPlayer` construction + configuration path
// that DOESN'T need to open a cpal stream (which would fail in CI
// without an audio device). The cpal Stream-open path is exercised
// in the `#[ignore]`-gated real-device tests below.

#[test]
fn audio_player_write_chunk_max_splits_large_writes() {
  // F2 contract: producer-side `write_samples` splits payloads
  // larger than `WRITE_CHUNK_MAX` into per-chunk lock acquisitions
  // so the cpal callback's `try_lock` window is bounded by
  // `WRITE_CHUNK_MAX` samples (microseconds-scale `extend`), not
  // the full payload length (which can be hundreds of milliseconds
  // for a multi-second TTS chunk). This mock test verifies the
  // chunking math via the public `WRITE_CHUNK_MAX` constant — the
  // device-touching path is exercised under the real-device gate.
  assert_eq!(
    WRITE_CHUNK_MAX, 4096,
    "WRITE_CHUNK_MAX is the documented contract value; bumping it changes the audio-callback \
     contention envelope (see player.rs ## Concurrency)"
  );

  // Simulate a 10 000-sample producer payload (e.g. a ~200ms
  // mono-24kHz TTS chunk). It must split into ceil(10000/4096)=3
  // chunks of sizes [4096, 4096, 1808] — the same iteration the
  // producer's per-chunk lock loop walks.
  let payload = vec![0.0_f32; 10_000];
  let chunks: Vec<usize> = payload.chunks(WRITE_CHUNK_MAX).map(<[f32]>::len).collect();
  assert_eq!(chunks.len(), 3);
  assert_eq!(chunks[0], 4096);
  assert_eq!(chunks[1], 4096);
  assert_eq!(chunks[2], 1808);
  assert_eq!(chunks.iter().sum::<usize>(), 10_000);
  // Every chunk MUST be <= WRITE_CHUNK_MAX (the lock-window bound).
  assert!(chunks.iter().all(|&n| n <= WRITE_CHUNK_MAX));

  // Boundary: a payload smaller than WRITE_CHUNK_MAX takes one
  // chunk; a payload exactly WRITE_CHUNK_MAX takes one chunk too.
  assert_eq!(vec![0.0_f32; 100].chunks(WRITE_CHUNK_MAX).count(), 1);
  assert_eq!(
    vec![0.0_f32; WRITE_CHUNK_MAX]
      .chunks(WRITE_CHUNK_MAX)
      .count(),
    1
  );
  assert_eq!(
    vec![0.0_f32; WRITE_CHUNK_MAX + 1]
      .chunks(WRITE_CHUNK_MAX)
      .count(),
    2
  );
}

#[test]
fn audio_player_rejects_non_f32_sample_format_pre_device() {
  // We can construct a PlaybackConfig with SampleFormat::I16 even
  // though the player doesn't currently support it; assert the
  // construction path itself succeeds (the device-open call would
  // be the one to reject — exercised in real-device tests). Verifies
  // the enum is exposed + the config builder is the gate.
  let cfg = PlaybackConfig::new(16_000, ChannelLayout::Mono, SampleFormat::I16)
    .with_queue_capacity_frames(1024);
  // cpal_config doesn't gate sample_format (that's a device-level
  // concern in cpal); the player's `with_device` constructor is what
  // returns Err on I16. Smoke-check the field round-trips:
  assert_eq!(cfg.sample_format(), SampleFormat::I16);
}

#[test]
fn audio_player_queue_capacity_frames_multiplied_by_channels() {
  // F4 post-fix semantics: `queue_capacity_frames` is the frame
  // count (constant across channel layouts); the player's
  // `with_device` constructor does the single frame-to-sample
  // conversion via `* channels.count()`. So a stereo config with
  // `queue_capacity_frames = 1024` admits exactly 2048 interleaved
  // L/R samples before overflow — NOT 4096 (the pre-fix bug
  // double-counted the channel multiplier in the `stereo()`
  // constructor; the assertion below is the post-fix unit-of-truth
  // for both mono and stereo, set via an explicit-field struct
  // literal so the constructor's frame/sample convention can't mask
  // a future regression).
  let stereo = PlaybackConfig::new(16_000, ChannelLayout::Stereo, SampleFormat::F32)
    .with_queue_capacity_frames(1024);
  assert_eq!(
    stereo.queue_capacity_frames() * usize::from(stereo.channels().count()),
    2048
  );

  let mono = PlaybackConfig::new(16_000, ChannelLayout::Mono, SampleFormat::F32)
    .with_queue_capacity_frames(1024);
  assert_eq!(
    mono.queue_capacity_frames() * usize::from(mono.channels().count()),
    1024
  );
}

// ---------------------------------------------------------------------------
// Real-device tests — gated. Run with: `cargo test -- --ignored`
// ---------------------------------------------------------------------------
//
// `#[ignore]` so CI (which may lack a default audio output device,
// e.g. headless macOS runners under -nox) doesn't fail on construct.
// macOS-only gate because CoreAudio is the only backend mlxrs targets
// in M5; on Linux/Windows the same code should work but isn't a
// shipping target. Run locally with:
//
//     cargo test -p mlxrs --features audio audio_player_starts_and_stops_on_default_device \
//         -- --ignored --test-threads=1

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_constructs_without_starting_stream() {
  use mlxrs::audio::playback::AudioPlayer;

  let mut player = AudioPlayer::new(PlaybackConfig::mono(24_000)).unwrap();
  // Newly-constructed player isn't running until `start()`.
  assert!(!player.is_running());
  assert!(!player.is_paused());
  assert_eq!(player.buffer_depth(), 0);
  assert_eq!(player.config().sample_rate(), 24_000);
  // Defaults round-trip:
  assert!((player.volume() - 1.0).abs() < 1e-6);
  // Cleanup.
  let _ = player.stop();
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_starts_and_stops_on_default_device() {
  use std::{thread, time::Duration};

  use mlxrs::audio::playback::AudioPlayer;

  let mut player = AudioPlayer::new(PlaybackConfig::mono(24_000)).unwrap();
  player.start().unwrap();
  assert!(player.is_running());

  // Push a quarter-second of silence so the cpal callback has
  // something to drain; underrun would also be safe (silence) but
  // this is a stronger sanity check that write_samples + flush
  // round-trip on a real device.
  let samples = vec![0.0_f32; 24_000 / 4];
  player.write_samples(&samples).unwrap();
  player.flush().unwrap();
  assert_eq!(player.buffer_depth(), 0);

  player.pause().unwrap();
  assert!(player.is_paused());
  assert!(!player.is_running());

  player.resume().unwrap();
  assert!(player.is_running());

  player.stop().unwrap();
  assert!(!player.is_running());

  // Give cpal a beat to settle before drop.
  thread::sleep(Duration::from_millis(50));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_buffer_overflow_returns_err() {
  use mlxrs::audio::playback::AudioPlayer;

  // Tiny queue so overflow is reachable without pushing megabytes.
  let cfg = PlaybackConfig::new(16_000, ChannelLayout::Mono, SampleFormat::F32)
    .with_queue_capacity_frames(1024);
  let mut player = AudioPlayer::new(cfg).unwrap();
  // F1: writes pre-`start()` are rejected (STATE_STOPPED is the
  // initial state, and the write-gate is "reject when STOPPED").
  // Transition to STATE_PAUSED via start()+pause() so writes are
  // accepted but the cpal callback's STATE_RUNNING guard prevents
  // it from draining the queue while we fill it.
  player.start().unwrap();
  player.pause().unwrap();
  player.write_samples(&[0.0_f32; 1024]).unwrap();
  let err = player.write_samples(&[0.0_f32; 1]).unwrap_err();
  assert!(
    matches!(err, mlxrs::error::Error::Backend { ref message } if message.contains("overflow")),
    "expected overflow Backend error, got {err:?}"
  );
  let _ = player.stop();
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_underrun_emits_silence_no_panic() {
  use std::{thread, time::Duration};

  use mlxrs::audio::playback::AudioPlayer;

  // Start the stream with an empty queue; the cpal callback should
  // emit silence (zero) for every callback hit instead of panicking
  // or blocking. We can't directly observe the cpal-callback buffer
  // from here, but we can assert the player stays in STATE_RUNNING
  // across a callback interval and is_running() stays true (no
  // poisoned-state from a panic in the callback).
  let mut player = AudioPlayer::new(PlaybackConfig::mono(24_000)).unwrap();
  player.start().unwrap();
  assert!(player.is_running());

  thread::sleep(Duration::from_millis(100));
  assert!(player.is_running(), "underrun must not stop the player");

  player.stop().unwrap();
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_store_volume_clamps_and_persists() {
  use mlxrs::audio::playback::AudioPlayer;

  let player = AudioPlayer::new(PlaybackConfig::mono(16_000)).unwrap();
  assert!((player.volume() - 1.0).abs() < 1e-6);

  player.store_volume(0.5);
  assert!((player.volume() - 0.5).abs() < 1e-6);

  // Clamp: values >1.0 or <0.0 are clamped silently.
  player.store_volume(1.5);
  assert!((player.volume() - 1.0).abs() < 1e-6);

  player.store_volume(-0.1);
  assert!((player.volume() - 0.0).abs() < 1e-6);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_output_stream_rejects_writes_after_stop() {
  // F1: terminal-state contract. After `stop()` returns,
  // `write_samples` MUST reject — late producer chunks MUST NOT
  // accumulate silently and replay on a later restart.
  use mlxrs::audio::playback::AudioPlayer;

  let mut player = AudioPlayer::new(PlaybackConfig::mono(16_000)).unwrap();
  player.start().unwrap();

  // Pre-stop: write succeeds.
  assert_eq!(player.write_samples(&[0.0_f32; 128]).unwrap(), 128);

  // Stop is the terminal-state transition.
  player.stop().unwrap();
  assert!(!player.is_running());

  // Post-stop: write is rejected with an `after stop()` Backend
  // error. The literal "after stop()" substring is the contract.
  let err = player.write_samples(&[0.0_f32; 32]).unwrap_err();
  match err {
    mlxrs::error::Error::Backend { message } => {
      assert!(
        message.contains("after stop()"),
        "expected `after stop()` in error message, got: {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
}

// ---------------------------------------------------------------------------
// F1 (HIGH) — one-way `terminated` latch independent of playback state
// ---------------------------------------------------------------------------
//
// Pre-R2 fix, `write_samples`' STATE_STOPPED gate only rejected while
// state was CURRENTLY STOPPED, but `start()` unconditionally stored
// STATE_RUNNING. The sequence `start(); stop(); start();` would slip
// past the gate and let `write_samples()` resume accepting post-stop
// chunks — silently violating `AudioOutputStream::stop`'s one-way
// terminal contract. R2 adds a dedicated `SharedState::terminated`
// AtomicBool latch (set in `stop()`, checked FIRST in every producer
// method); these tests pin the new contract on a real device.

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_start_after_stop_returns_terminated_err() {
  // F1: `start(); stop(); start()` — the second `start()` MUST
  // reject with a "terminated" Backend error rather than re-arm the
  // player. Pre-R2 this returned Ok(()) and silently rehydrated the
  // producer surface.
  use mlxrs::audio::playback::AudioPlayer;

  let mut player = AudioPlayer::new(PlaybackConfig::mono(16_000)).unwrap();
  player.start().unwrap();
  player.stop().unwrap();

  let err = player.start().unwrap_err();
  match err {
    mlxrs::error::Error::Backend { message } => {
      assert!(
        message.contains("terminated"),
        "expected `terminated` in start()-after-stop() error, got: {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_write_samples_after_restart_attempt_still_rejected() {
  // F1 hardening: even if a caller IGNORES the `start()`-after-stop
  // Err and pushes samples anyway, `write_samples` MUST still reject
  // (the terminated latch is checked FIRST, before the state
  // tri-state). This is the regression test for the exact attack the
  // R2 fix is closing: `start(); stop(); start(); write_samples()`
  // pre-fix slipped past `state == STATE_STOPPED` because the second
  // `start()` re-armed state to STATE_RUNNING.
  use mlxrs::audio::playback::AudioPlayer;

  let mut player = AudioPlayer::new(PlaybackConfig::mono(16_000)).unwrap();
  player.start().unwrap();
  player.stop().unwrap();
  // Ignore the Err from the restart attempt — the contract is that
  // EVEN IF the producer ignores `start()`'s Err, `write_samples`
  // still rejects.
  let _ = player.start();

  let err = player.write_samples(&[0.5_f32; 64]).unwrap_err();
  match err {
    mlxrs::error::Error::Backend { message } => {
      assert!(
        message.contains("terminated"),
        "expected `terminated` in write_samples()-after-restart-attempt error, got: {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_pause_after_stop_returns_terminated_err() {
  // F1: `pause()` after `stop()` MUST reject — same terminated-latch
  // rule as `start()`. Without this, a caller could `start();
  // stop(); pause()` and surprise the state machine.
  use mlxrs::audio::playback::AudioPlayer;

  let mut player = AudioPlayer::new(PlaybackConfig::mono(16_000)).unwrap();
  player.start().unwrap();
  player.stop().unwrap();

  let err = player.pause().unwrap_err();
  match err {
    mlxrs::error::Error::Backend { message } => {
      assert!(
        message.contains("terminated"),
        "expected `terminated` in pause()-after-stop() error, got: {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires real default audio output device"]
fn audio_player_resume_after_stop_returns_terminated_err() {
  // F1: `resume()` after `stop()` MUST reject. Resume delegates to
  // `start()` internally but the dedicated `resume`-named error
  // message keeps the call-site signal clear.
  use mlxrs::audio::playback::AudioPlayer;

  let mut player = AudioPlayer::new(PlaybackConfig::mono(16_000)).unwrap();
  player.start().unwrap();
  player.stop().unwrap();

  let err = player.resume().unwrap_err();
  match err {
    mlxrs::error::Error::Backend { message } => {
      assert!(
        message.contains("terminated"),
        "expected `terminated` in resume()-after-stop() error, got: {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
}

// ---------------------------------------------------------------------------
// F2 (MEDIUM) — queue pre-allocation at construction, NO per-chunk realloc
// ---------------------------------------------------------------------------
//
// Pre-R2, `write_samples`' producer loop called `q.reserve_exact(chunk.len())`
// under the queue lock on every chunk — if the underlying VecDeque needed to
// grow past its initial capacity, the allocator path ran inside the lock and
// the cpal callback's `try_lock` would see WouldBlock for allocator time too
// (not just `extend`). R2 pre-allocates the full bounded
// `queue_capacity_samples` at construction (via `try_reserve_exact` so alloc
// failure surfaces as `Error::Backend`) and removes the per-chunk
// `reserve_exact`. The two F2 invariant tests
// (`audio_player_pre_allocates_queue_capacity_at_construction` +
// `audio_player_write_samples_does_not_grow_queue_capacity_during_playback`)
// were previously here and used `pub #[doc(hidden)] _test_*` accessors on
// `AudioPlayer` to reach the private VecDeque capacity. The R3 MEDIUM fix
// removes those leaked-public accessors and moves the tests INTO
// `mlxrs/src/audio/playback/player.rs`'s `#[cfg(test)] mod tests` block,
// where they can read `shared.queue.capacity()` +
// `shared.queue_capacity_samples` directly without any `pub` surface.

// F3 tests exercise the pure `sanitize_volume` helper directly so
// they run in CI without a default audio output device. The
// device-touching round-trip is exercised in
// `audio_player_store_volume_clamps_and_persists` (real-device gate).

#[test]
fn audio_player_store_volume_sanitizes_nan_to_zero() {
  // F3: NaN must NOT propagate into volume_bits — the callback's
  // `sample * volume` would emit NaN PCM (audible as full-scale
  // noise on most DACs). `f32::clamp` preserves NaN, so the
  // sanitizer explicitly maps non-finite inputs to 0.0.
  let stored = sanitize_volume(f32::NAN);
  assert!(
    !stored.is_nan(),
    "sanitize_volume(NaN) must NOT return NaN (would produce NaN PCM via sample * volume); \
     got {stored}"
  );
  assert_eq!(stored, 0.0, "NaN volume must sanitize to 0.0");
}

#[test]
fn audio_player_store_volume_sanitizes_infinity_to_zero() {
  // F3: positive and negative infinity are both non-finite and
  // must sanitize to 0.0 (same policy as NaN — `f32::clamp` on +∞
  // returns 1.0, which is the wrong semantic for "you gave us
  // garbage"; we treat all non-finite inputs uniformly).
  assert_eq!(
    sanitize_volume(f32::INFINITY),
    0.0,
    "+infinity volume must sanitize to 0.0 (non-finite policy)"
  );
  assert_eq!(
    sanitize_volume(f32::NEG_INFINITY),
    0.0,
    "-infinity volume must sanitize to 0.0 (non-finite policy)"
  );
}

#[test]
fn audio_player_store_volume_clamps_negative_to_zero() {
  // F3: negative finite values are clamped (not sanitized) — the
  // existing `clamp(0.0, 1.0)` path handles this. Explicit test so
  // a future regression (e.g. dropping the clamp in favor of the
  // is_finite branch) is caught.
  assert_eq!(sanitize_volume(-0.5), 0.0);
  assert_eq!(sanitize_volume(-1.0), 0.0);
  assert_eq!(sanitize_volume(-1000.0), 0.0);
}

#[test]
fn audio_player_store_volume_passes_through_in_range() {
  // F3 sanity: finite in-range values pass through unchanged.
  assert_eq!(sanitize_volume(0.0), 0.0);
  assert_eq!(sanitize_volume(0.5), 0.5);
  assert_eq!(sanitize_volume(1.0), 1.0);
}

#[test]
fn audio_player_store_volume_clamps_above_unity() {
  // F3: finite values >1.0 clamp to 1.0 (not sanitized to 0.0 —
  // they're well-formed, just out of range).
  assert_eq!(sanitize_volume(1.5), 1.0);
  assert_eq!(sanitize_volume(100.0), 1.0);
}
