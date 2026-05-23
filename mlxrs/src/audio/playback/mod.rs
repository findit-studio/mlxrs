//! Device playback — the cpal-backed mlxrs port of
//! `mlx-audio-swift`'s `MLXAudioCore.AudioPlayer` streaming surface.
//!
//! AUDIO-A11 (issue #174) ports the *streaming* branch of Swift's
//! `AudioPlayer` (the `startStreaming(sampleRate:)` /
//! `scheduleAudioChunk(_:withCrossfade:)` /
//! `stopStreaming()` triad) over [cpal][cpal], the de-facto pure-Rust
//! cross-platform audio I/O crate (CoreAudio on macOS, WASAPI on
//! Windows, ALSA on Linux). The cpal equivalent of Swift's
//! `AVAudioEngine` + `AVAudioPlayerNode` pattern is:
//!
//! ```text
//! cpal::default_host() ── Host ── default_output_device() ── Device
//!                                                              │
//!                                                  build_output_stream(config, data_cb, err_cb)
//!                                                              │
//!                                                              ▼
//!                                                         cpal::Stream  ── play() / pause()
//! ```
//!
//! [`AudioPlayer`] owns the [`cpal::Stream`] and a shared queue the
//! producer fills; the cpal callback pulls from the queue on the
//! audio I/O thread. [`AudioOutputStream`] is the producer-side
//! trait the A8 speech-to-speech pipeline ([`crate::audio::sts`])
//! consumes — same shape whether the sink is a real device, a file,
//! or a unit-test recorder.
//!
//! ## Module layout
//!
//! - [`mod@config`] — [`PlaybackConfig`], [`SampleFormat`],
//!   [`ChannelLayout`]. The cpal-equivalent of Swift's
//!   `AVAudioFormat(standardFormatWithSampleRate:channels:)`.
//! - [`mod@output_stream`] — [`AudioOutputStream`] trait. The narrow
//!   producer-side surface the upstream pipeline consumes.
//! - [`mod@player`] — [`AudioPlayer`] struct. The cpal-backed
//!   default implementor of [`AudioOutputStream`].
//!
//! ## Scope cuts (explicit, A11)
//!
//! Per the `[[feedback_match_official_binding_design]]` rule the
//! mlxrs port intentionally omits a few Swift-side capabilities; the
//! full enumeration lives in the [`mod@player`] docstring. Quick
//! summary:
//!
//! - **Audio input / recording** — playback-only.
//! - **File I/O** — raw PCM only; file decoding lives in
//!   [`crate::audio::io`].
//! - **Format conversion / resampling** — caller-supplied PCM at the
//!   configured [`PlaybackConfig::sample_rate`] /
//!   [`PlaybackConfig::channels`].
//! - **Crossfade / fade-in** — application-level concern.
//! - **`@Published` properties / Combine bindings** — Rust library,
//!   no `ObservableObject`.
//!
//! [cpal]: https://crates.io/crates/cpal

pub mod config;
pub mod output_stream;
pub mod player;

pub use config::{ChannelLayout, PlaybackConfig, SampleFormat};
pub use output_stream::AudioOutputStream;
pub use player::AudioPlayer;
