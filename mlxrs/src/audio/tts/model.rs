//! The architecture-agnostic [`TtsModel`] seam for `mlxrs::audio::tts` — the
//! text-to-speech analogue of [`crate::audio::stt::model::Model`], mirroring
//! mlx-audio's TTS model surface (the per-model `Model.generate` shape every
//! `tts/models/*` architecture exposes — kokoro, csm, bark, qwen3-tts, …)
//! and mlx-audio-swift's [`SpeechGenerationModel`][swift-gen] protocol.
//!
//! Per the project's no-per-model-arch rule
//! (`project_no_per_model_arch_porting`), mlxrs ships **no
//! concrete TTS model implementations**: the per-model token decoder +
//! vocoder / codec (kokoro's istftnet decoder, csm's RVQ + mimi codec,
//! bark's coarse/fine transformers, …) live in user code on top of this
//! trait. [`TtsModel`] is the *shape* per-model code must conform to so the
//! [`crate::audio::tts::generate::tts_generate`] driver can synthesize from
//! any architecture uniformly — the same "trait + generic loop" seam the
//! [`crate::audio::stt`] STT loop and the [`crate::lm::generate`] LM loop
//! use.
//!
//! ## What the trait abstracts
//!
//! mlx-audio's TTS architectures differ wildly internally (autoregressive
//! token LMs + neural codecs vs. non-autoregressive duration-predictor +
//! iSTFT vocoders vs. diffusion decoders), but their **public synthesis
//! contract** is uniform: `text → list/stream of audio chunks`, each chunk a
//! span of `f32` PCM samples in `[-1, 1]` at the model's
//! [`TtsModel::sample_rate`]. mlx-audio expresses one chunk per *text
//! segment* (kokoro's `split_pattern` split) and, under `stream=True`, one
//! chunk per *streaming interval* (the per-model `streaming_token_interval`
//! cadence). mlxrs mirrors that with a single
//! [`TtsModel::synthesize_segment`] hook the
//! [`super::generate::tts_generate`] driver calls once per
//! [`super::generate::TtsSegment`].
//!
//! [swift-gen]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/Generation.swift

use crate::{
  array::Array,
  error::{Error, InvariantViolationPayload, Result},
};

use super::generate::{TtsGenConfig, TtsSegment};

/// A text-to-speech model: the architecture-agnostic seam every concrete TTS
/// architecture (kokoro, csm, bark, qwen3-tts, …) implements so
/// [`super::generate::tts_generate`] can synthesize from it uniformly.
///
/// Mirrors mlx-audio's per-model `Model.generate` shape and
/// mlx-audio-swift's [`SpeechGenerationModel`][swift-gen] protocol: the
/// per-model token decoder + vocoder is wired behind
/// [`TtsModel::synthesize_segment`]; the driver composes text segmentation,
/// audio-chunk assembly, and the streaming-chunk envelope around it.
///
/// - `&self` everywhere — weights are immutable after load, so TTS synthesis
///   never needs `&mut` on the model (matching mlx-audio's `nn.Module` for
///   inference, and the same `&self` choice
///   [`crate::audio::stt::model::Model`] makes). One model can back many
///   concurrent synthesis runs.
/// - [`TtsModel::synthesize_segment`] runs **once per text segment** — the
///   mlx-audio per-model `generate` loop's `for segment_idx, … in
///   enumerate(pipeline(text, …))` body (kokoro `kokoro.py`, llama
///   `llama.py`). The driver handles splitting `text` into
///   [`TtsSegment`]s and assembling the per-segment outputs.
///
/// [swift-gen]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/Generation.swift
pub trait TtsModel {
  /// Synthesize one text segment into a 1-D `f32` PCM waveform.
  ///
  /// `segment` carries the segment text plus the resolved per-segment
  /// synthesis knobs ([`TtsSegment`] — voice / speed / language / the
  /// optional reference-audio voice-clone inputs). The returned [`Array`]
  /// is the segment's audio: a **rank-1** `[samples]` tensor of `f32` PCM
  /// in `[-1, 1]` at [`TtsModel::sample_rate`] (the mlx-audio
  /// `GenerationResult.audio` layout — every per-model `generate` yields
  /// `audio[0]` / a `[samples]` slice).
  ///
  /// Per-model code wires whichever synthesis path its architecture uses:
  /// - kokoro: G2P → ALBERT prosody encoder → duration predictor → iSTFT
  ///   decoder (`kokoro/kokoro.py`),
  /// - csm / bark / qwen3-tts: autoregressive backbone token LM → neural
  ///   codec decode (`sesame/`, `bark/`, `qwen3_tts/`).
  ///
  /// The text-preprocessing step (G2P / phonemization / normalization) is
  /// model-specific — a model that needs phonemized IPA input runs its own
  /// G2P here, optionally driven by a caller-supplied
  /// [`TextProcessor`](super::TextProcessor) hook. The driver passes the
  /// segment text through unchanged; it does not phonemize.
  ///
  /// Default impl is unsupported (`Err(Error::InvariantViolation)`) — every concrete
  /// TTS model MUST override it, mirroring
  /// [`crate::audio::stt::model::Model::decode_step`]'s
  /// per-model-override default.
  fn synthesize_segment(&self, segment: &TtsSegment<'_>) -> Result<Array> {
    let _ = segment;
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "TtsModel::synthesize_segment",
      "needs `synthesize_segment` override (per-model)",
    )))
  }

  /// The output PCM sample rate in Hz.
  ///
  /// mlx-audio per-model `Model.sample_rate` — kokoro `24_000`, most codec-
  /// based models `24_000`, some `16_000` / `44_100`. Used by
  /// [`super::generate::tts_generate`] to stamp
  /// [`super::generate::AudioChunk::sample_rate`] and to derive chunk
  /// durations; the trait has no default because there is no architecture-
  /// independent "correct" rate (unlike STT's whisper-default mel config,
  /// every TTS model declares its own).
  fn sample_rate(&self) -> u32;

  /// The default synthesis config this model recommends.
  ///
  /// mlx-audio-swift's `SpeechGenerationModel.defaultGenerationParameters`
  /// — the per-model preset a caller gets when they do not pass an explicit
  /// [`TtsGenConfig`]. The default impl returns [`TtsGenConfig::default`]
  /// (mlx-audio's `generate_audio` defaults: `voice = "af_heart"`,
  /// `speed = 1.0`, `temperature = 0.7`, `max_tokens = 1200`); a model with
  /// different recommended knobs (e.g. a different default voice) overrides
  /// it.
  fn default_config(&self) -> TtsGenConfig {
    TtsGenConfig::default()
  }
}
