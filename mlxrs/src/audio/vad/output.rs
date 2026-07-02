//! The shared VAD inference-result struct, ported from
//! [`mlx_audio.vad.models.silero_vad.silero_vad.VADOutput`][vad-output].
//!
//! Every VAD architecture mlx-audio ships exposes its
//! `Model.generate(audio, …)` result as this 3-field bundle: the speech
//! timestamps (start/end pairs), the per-frame speech probabilities, and
//! the inference sample rate. The struct is reproduced verbatim here so a
//! per-architecture VAD model (silero_vad, sortformer / diarization,
//! smart_turn endpoint, …) can return one [`VadOutput`] that the
//! downstream caller can consume uniformly (the [`VoicePipeline`-style
//! consumer][sts-pipeline] mlx-audio's `sts/voice_pipeline.py` builds).
//!
//! [vad-output]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L21-L25
//! [sts-pipeline]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py

use crate::array::Array;

/// One speech segment in a [`VadOutput`] — the start / end pair mlx-audio
/// emits as `{"start": int, "end": int}` dictionaries
/// ([silero_vad.py:163-176][vad-segment]).
///
/// `start` and `end` are sample indices into the input waveform (the
/// `return_seconds=False` path; mlx-audio's `return_seconds=True` path
/// multiplies by `1/sample_rate` — that conversion is left to the
/// caller). `start < end` by construction; an empty / silent input yields
/// an empty `timestamps` vector.
///
/// [vad-segment]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L163-L176
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct SpeechSegment {
  /// Start sample index (inclusive) of the speech segment.
  start: u64,
  /// End sample index (exclusive) of the speech segment.
  end: u64,
}

impl SpeechSegment {
  /// Construct a [`SpeechSegment`] from a start/end sample-index pair.
  ///
  /// `start` is inclusive, `end` is exclusive. Both are sample indices
  /// into the input waveform (`return_seconds=False` path).
  pub const fn new(start: u64, end: u64) -> Self {
    Self { start, end }
  }

  /// Start sample index (inclusive) of the speech segment.
  #[inline(always)]
  pub fn start(&self) -> u64 {
    self.start
  }

  /// End sample index (exclusive) of the speech segment.
  #[inline(always)]
  pub fn end(&self) -> u64 {
    self.end
  }

  /// Start time in seconds — mlx-audio's `return_seconds=True` view
  /// (`round(start / sample_rate, 3)`, silero_vad.py:419-426). `sample_rate` is
  /// the inference rate ([`VadOutput::sample_rate`]); the result is rounded to
  /// milliseconds (3 decimals) to match the reference.
  #[inline]
  pub fn start_seconds(&self, sample_rate: u32) -> f64 {
    round_to_ms(self.start as f64 / f64::from(sample_rate.max(1)))
  }

  /// End time in seconds — the `return_seconds=True` view of [`Self::end`]
  /// (see [`Self::start_seconds`]).
  #[inline]
  pub fn end_seconds(&self, sample_rate: u32) -> f64 {
    round_to_ms(self.end as f64 / f64::from(sample_rate.max(1)))
  }
}

/// Round seconds to milliseconds (3 decimals) — mlx-audio's `round(x, 3)` for
/// the `return_seconds` view (silero_vad.py:422-423).
#[inline]
fn round_to_ms(seconds: f64) -> f64 {
  (seconds * 1000.0).round() / 1000.0
}

/// The result of one VAD inference pass — port of
/// [`mlx_audio.vad.models.silero_vad.silero_vad.VADOutput`][vad-output].
///
/// Faithful 1:1 of mlx-audio's 3-field dataclass:
///
/// - `timestamps: List[dict]` → [`VadOutput::timestamps`] as
///   `Vec<SpeechSegment>` (the `{"start": …, "end": …}` dictionaries are
///   spelled as a typed [`SpeechSegment`] here rather than free-form
///   maps — see the per-`segment` doc).
/// - `probabilities: mx.array` → [`VadOutput::probabilities`] as an
///   [`Array`] (the same `(n_frames,)` shape mlx-audio's
///   `_predict_proba_array` returns).
/// - `sample_rate: int` → [`VadOutput::sample_rate`] as `u32` (the input
///   waveform's sample rate; matches mlx-audio's `int`).
///
/// [`Array`] is `!Send`, so this struct is `!Send` — matching every
/// other audio-domain struct in mlxrs (`crate::audio::stt`'s
/// `EncoderState`, the `crate::lm::generate::GenStep` envelope, …).
///
/// Serde lifecycle: only the typed [`VadOutput::timestamps`] +
/// [`VadOutput::sample_rate`] fields are derivable (the [`Array`]
/// probabilities are a backend handle and cannot be Serde'd directly);
/// the [`SpeechSegment`] type ships full serde derives so a caller that
/// only needs the timestamps (the common `VoicePipeline` consumer) can
/// round-trip them without touching the array.
///
/// [vad-output]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L21-L25
#[derive(Debug)]
pub struct VadOutput {
  /// Speech segments detected in the input waveform — port of mlx-audio's
  /// `timestamps: List[dict]`. Empty when the input has no speech.
  pub timestamps: Vec<SpeechSegment>,
  /// The per-frame speech probabilities mlx-audio's
  /// `_predict_proba_array` returns — typically `(n_frames,)`-shaped
  /// floats in `[0, 1]`. Carried verbatim so a caller can apply a
  /// different threshold without re-running inference.
  pub probabilities: Array,
  /// The input waveform sample rate (Hz). mlx-audio's
  /// `Model.generate(audio, sample_rate=…)` records the resolved rate
  /// here so a downstream consumer (the `VoicePipeline` end-silence
  /// computation) does not have to plumb it separately.
  pub sample_rate: u32,
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A minimal [`VadOutput`] can be constructed and its fields read —
  /// the constructor-round-trip smoke test (the [`Array`] probabilities
  /// field is not serde-checked because mlx [`Array`]s are backend
  /// handles).
  #[test]
  fn vad_output_struct_round_trips() {
    let segments = vec![SpeechSegment::new(0, 1600), SpeechSegment::new(3200, 4800)];
    // Probabilities: shape (3,), the 3-frame mock the test exercises.
    let probabilities = Array::from_slice::<f32>(&[0.1, 0.9, 0.85], &(3,)).unwrap();
    let out = VadOutput {
      timestamps: segments.clone(),
      probabilities,
      sample_rate: 16_000,
    };

    assert_eq!(out.timestamps, segments);
    assert_eq!(out.sample_rate, 16_000);
    assert_eq!(out.probabilities.shape(), vec![3]);

    // Serde sanity on the typed-only timestamp field (probabilities is
    // an Array handle and not serde-able).
    let s = serde_json::to_string(&out.timestamps).unwrap();
    let back: Vec<SpeechSegment> = serde_json::from_str(&s).unwrap();
    assert_eq!(back, segments);
  }
}
