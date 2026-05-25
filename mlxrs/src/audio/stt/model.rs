//! The architecture-agnostic [`Model`] seam for `mlxrs::audio::stt` — the
//! STT analogue of [`crate::lm::model::Model`] / [`crate::vlm`]'s multimodal
//! shape, mirroring mlx-audio's [`STTModel` base][stt-base] (and the per-
//! model subclasses' `encode_audio` / `decode_step` shape: whisper's encoder
//! plus cross-attention decoder, parakeet's conformer encoder plus RNN-T
//! joint network, etc.).
//!
//! Per the project's no-per-model-arch rule
//! ([`project_no_per_model_arch_porting`][noarch]), mlxrs ships **no
//! concrete STT model implementations**. This trait is the *shape* per-model
//! code (whisper, parakeet, canary, …) must conform to so the
//! [`crate::audio::stt::generate::stt_generate`] iterator can drive any
//! architecture uniformly — the same "trait + generic loop" seam the
//! [`crate::lm::generate`] loop uses for text-only LMs.
//!
//! [stt-base]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/base.py
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

use crate::{
  array::Array,
  error::{Error, Result},
  lm::cache::KvCache,
};

/// A speech-to-text model: extends [`crate::lm::model::Model`] with an
/// audio-encoder front-end and a cross-attention decode step.
///
/// Mirrors mlx-audio's `STTModel` shape: each concrete architecture (whisper,
/// parakeet, canary, …) ports its encoder + cross-attention/joint-network
/// decoder behind these two hooks; [`crate::audio::stt::generate::stt_generate`]
/// composes them uniformly.
///
/// - `&self` everywhere — weights are immutable after load, so STT
///   generation never needs `&mut` on the model (matching mlx-audio's
///   `nn.Module` for inference). One model can back many concurrent decode
///   loops via independent KV caches.
/// - `encode_audio` runs **once** per utterance (mlx-audio whisper's
///   `AudioEncoder.__call__`); the returned encoder hidden states are reused
///   across every decode step.
/// - `decode_step` runs **once per output token** (mlx-audio's per-step
///   decoder forward + cross-attention); the per-layer KV cache is the same
///   heterogeneous-by-layer `Box<dyn KvCache>` slice the [LM
///   loop][crate::lm::generate] uses, so a multi-cache-kind model (e.g.
///   sliding-window encoder cache + standard self-attention decoder cache)
///   composes naturally.
///
/// `bos_token` / `eos_token` mirror mlx-audio's whisper
/// `<|startoftranscript|>` / `<|endoftext|>` identification: per-model
/// special-token identities live with the model implementation rather than
/// the loop's [`super::generate::SttGenConfig`].
pub trait Model: crate::lm::model::Model {
  /// Encode a mel-spectrogram into encoder hidden states.
  ///
  /// `mel` is the [`crate::audio::dsp::log_mel_spectrogram`] output of shape
  /// `(n_mels, T)` (the mlx-audio / Whisper canonical layout — `n_mels`
  /// frequency bins along axis 0, `T` time frames along axis 1). The
  /// returned encoder states' shape (`[T', D]` for whisper-style models,
  /// `[T', H, D]` for split-head variants) is the per-model choice;
  /// [`super::generate::stt_generate`] treats it opaquely and forwards it
  /// untouched to every [`Model::decode_step`] call.
  ///
  /// Per-model encoder (conv subsampling + transformer for whisper;
  /// conformer for parakeet; etc.) lives in user code.
  fn encode_audio(&self, mel: &Array) -> Result<Array>;

  /// Cross-attend the decoder over `encoder_states` (the output of
  /// [`Model::encode_audio`]) for the current decode step, conditioned on
  /// the previously sampled `token`.
  ///
  /// Per-model code wires whichever cross-attention shape its architecture
  /// uses:
  /// - whisper: a self-attention decoder block followed by a
  ///   cross-attention block that projects K/V from `encoder_states` (the
  ///   `ResidualAttentionBlock` `xa` path in mlx-audio's
  ///   `whisper/whisper.py`),
  /// - parakeet (RNN-T): a joint network conditioning the predictor on
  ///   `encoder_states` per time-step.
  ///
  /// Returns next-token logits `[1, V]` (the same `[B=1, V]` shape the
  /// [LM loop's][crate::lm::generate] last-position slice produces). The
  /// [`super::generate::stt_generate`] loop normalizes via the LM's exact
  /// `logits - logsumexp` and samples via the LM's sampler chain.
  ///
  /// Default impl is unsupported (`Err(Error::Backend)`) — every concrete
  /// STT model MUST override it. The trait does **not** route through
  /// [`crate::lm::model::Model::forward`] because the STT decode step's
  /// signature is fundamentally different (a single token id + the encoder
  /// states the LM loop has no notion of); overriding `decode_step`
  /// keeps the [LM][crate::lm::model::Model] / STT seams cleanly separated
  /// the same way [`crate::lm::model::Model::forward_embeddings`] separates
  /// the VLM seam.
  fn decode_step(
    &self,
    token: u32,
    encoder_states: &Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<Array> {
    let _ = (token, encoder_states, cache);
    Err(Error::Backend {
      message: "STT model needs `decode_step` override (per-model)".into(),
    })
  }

  /// The mel-spectrogram extraction config this model expects.
  ///
  /// The default is the Whisper preset
  /// ([`MelConfig::whisper_default`]): `n_fft=400`, `hop_length=160`,
  /// `n_mels=80`, `sample_rate=16000`. Per-model code overrides for
  /// architectures with different feature-extractor configs (e.g.
  /// parakeet's `n_mels=80, hop=160` but `sample_rate=16000`; canary's
  /// `n_mels=128`).
  fn mel_config(&self) -> MelConfig {
    MelConfig::whisper_default()
  }

  /// The BOS token id the decode loop starts from
  /// (e.g. `<|startoftranscript|>` for whisper).
  ///
  /// This is the **first token fed into [`Model::decode_step`]** — mlx-
  /// audio's whisper `DecodingTask` initializes `tokens = list(sot_sequence)`
  /// where the first element is `tokenizer.sot`; mlxrs mirrors that as a
  /// single `bos_token()` seed (per-model wrapping of the full
  /// `sot_sequence` — task / language / timestamps tokens — happens inside
  /// the per-model code by feeding them through [`Model::decode_step`]
  /// before yielding the first user-visible token).
  fn bos_token(&self) -> u32;

  /// The EOS token id the decode loop stops on (e.g. `<|endoftext|>`).
  ///
  /// Generation ends — the EOS token IS yielded as the final
  /// [`crate::lm::generate::GenStep`] (mirroring the LM loop's
  /// "yield-then-fuse" eos handling) — once
  /// [`Model::decode_step`] samples this id.
  fn eos_token(&self) -> u32;
}

/// Mel-spectrogram extraction config — the argument bundle
/// [`crate::audio::dsp::log_mel_spectrogram`] consumes, returned by
/// [`Model::mel_config`].
///
/// [`MelConfig::whisper_default`] is the Whisper preset (the only one
/// mlx-audio currently bundles as a "default" — every other architecture
/// declares its own feature-extractor config); per-model overrides supply
/// custom values for architectures with different mel front-ends.
///
/// `Copy` because the fields are all trivially-copyable primitives —
/// matches the rest of mlxrs's small-config bundles (e.g. the lm
/// [`crate::lm::cache::CacheConfig`] is also a plain struct, though that
/// one isn't `Copy` because it carries an `Option<i32>`; here every field
/// is `Copy`).
#[derive(Debug, Clone, Copy)]
pub struct MelConfig {
  /// FFT length (mlx-audio whisper default `400`).
  n_fft: usize,
  /// STFT hop length in samples (mlx-audio whisper default `160`).
  hop_length: usize,
  /// Window length in samples; `None` ⇒ `n_fft` (mlx-audio default).
  /// Per §1 scalar Option-with-meaningful-sentinel: kept as `Option`.
  win_length: Option<usize>,
  /// Number of mel filterbank bins (mlx-audio whisper default `80`; canary
  /// uses `128`).
  n_mels: usize,
  /// Target audio sample rate in Hz (mlx-audio whisper default `16_000`).
  /// [`super::generate::stt_generate`] resamples the input via
  /// [`crate::audio::io::resample_linear`] when the source sample rate
  /// differs (gated on [`super::generate::SttGenConfig::auto_resample`]).
  sample_rate: u32,
  /// Lower mel band edge (Hz; mlx-audio default `0.0`).
  f_min: f32,
  /// Upper mel band edge (Hz); `None` ⇒ `sample_rate / 2` (Nyquist), the
  /// `mel_filter_bank` default.
  /// Per §1 scalar Option-with-meaningful-sentinel: kept as `Option`.
  f_max: Option<f32>,
  /// Numerical floor applied to mel energies before the **natural log**
  /// (`ln`, the `log_mel_spectrogram_with` `.log()` step)
  /// ([`crate::audio::dsp::LogFloor`]). Whisper frontends expect
  /// `1e-10` ([`LogFloor::Whisper`](crate::audio::dsp::LogFloor::Whisper),
  /// the default); Kaldi-style frontends expect `1e-8`
  /// ([`LogFloor::Kaldi`](crate::audio::dsp::LogFloor::Kaldi)). A model
  /// whose feature extractor uses the Kaldi floor MUST set this in its
  /// `mel_config` override — otherwise low-energy / silence bins are
  /// shifted by `ln(1e-8) - ln(1e-10) = ln(100) ≈ 4.6` natural-log
  /// units, silently degrading transcription quality. Threaded through
  /// [`super::generate::stt_generate`]'s `audio_path_to_mel` via
  /// [`crate::audio::dsp::log_mel_spectrogram_with`].
  log_floor: crate::audio::dsp::LogFloor,
}

impl MelConfig {
  /// Construct a [`MelConfig`] from all fields.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    n_fft: usize,
    hop_length: usize,
    win_length: Option<usize>,
    n_mels: usize,
    sample_rate: u32,
    f_min: f32,
    f_max: Option<f32>,
    log_floor: crate::audio::dsp::LogFloor,
  ) -> Self {
    Self {
      n_fft,
      hop_length,
      win_length,
      n_mels,
      sample_rate,
      f_min,
      f_max,
      log_floor,
    }
  }

  /// The Whisper preset: `n_fft=400`, `hop_length=160`, `n_mels=80`,
  /// `sample_rate=16_000`, `f_min=0`, `f_max=None` (Nyquist). Matches
  /// mlx-audio's whisper [`audio.py`][whisper-audio] feature-extractor
  /// defaults.
  ///
  /// [whisper-audio]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/whisper/audio.py
  pub const fn whisper_default() -> Self {
    Self {
      n_fft: 400,
      hop_length: 160,
      win_length: None,
      n_mels: 80,
      sample_rate: 16_000,
      f_min: 0.0,
      f_max: None,
      log_floor: crate::audio::dsp::LogFloor::Whisper,
    }
  }

  /// FFT length.
  #[inline(always)]
  pub fn n_fft(&self) -> usize {
    self.n_fft
  }

  /// STFT hop length in samples.
  #[inline(always)]
  pub fn hop_length(&self) -> usize {
    self.hop_length
  }

  /// Window length in samples; `None` ⇒ `n_fft`.
  #[inline(always)]
  pub fn win_length(&self) -> Option<usize> {
    self.win_length
  }

  /// Number of mel filterbank bins.
  #[inline(always)]
  pub fn n_mels(&self) -> usize {
    self.n_mels
  }

  /// Target audio sample rate in Hz.
  #[inline(always)]
  pub fn sample_rate(&self) -> u32 {
    self.sample_rate
  }

  /// Lower mel band edge in Hz.
  #[inline(always)]
  pub fn f_min(&self) -> f32 {
    self.f_min
  }

  /// Upper mel band edge in Hz; `None` ⇒ Nyquist.
  #[inline(always)]
  pub fn f_max(&self) -> Option<f32> {
    self.f_max
  }

  /// Numerical log floor applied before the mel's natural log.
  #[inline(always)]
  pub fn log_floor(&self) -> crate::audio::dsp::LogFloor {
    self.log_floor
  }

  /// Return a copy with `n_fft` overridden.
  #[inline(always)]
  pub fn with_n_fft(self, n_fft: usize) -> Self {
    Self { n_fft, ..self }
  }

  /// Return a copy with `hop_length` overridden.
  #[inline(always)]
  pub fn with_hop_length(self, hop_length: usize) -> Self {
    Self { hop_length, ..self }
  }

  /// Return a copy with `win_length` overridden.
  #[inline(always)]
  pub fn with_win_length(self, win_length: Option<usize>) -> Self {
    Self { win_length, ..self }
  }

  /// Return a copy with `n_mels` overridden.
  #[inline(always)]
  pub fn with_n_mels(self, n_mels: usize) -> Self {
    Self { n_mels, ..self }
  }

  /// Return a copy with `sample_rate` overridden.
  #[inline(always)]
  pub fn with_sample_rate(self, sample_rate: u32) -> Self {
    Self {
      sample_rate,
      ..self
    }
  }

  /// Return a copy with `f_min` overridden.
  #[inline(always)]
  pub fn with_f_min(self, f_min: f32) -> Self {
    Self { f_min, ..self }
  }

  /// Return a copy with `f_max` overridden.
  #[inline(always)]
  pub fn with_f_max(self, f_max: Option<f32>) -> Self {
    Self { f_max, ..self }
  }

  /// Return a copy with `log_floor` overridden.
  #[inline(always)]
  pub fn with_log_floor(self, log_floor: crate::audio::dsp::LogFloor) -> Self {
    Self { log_floor, ..self }
  }
}
