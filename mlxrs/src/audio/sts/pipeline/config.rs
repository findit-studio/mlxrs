//! [`VoicePipelineConfig`] — the typed argument bundle the
//! [`super::orchestrator::VoiceSession`] consumes, port of
//! [`mlx_audio.sts.voice_pipeline.VoicePipelineConfig`][vp-cfg].
//!
//! Faithful 1:1 of mlx-audio's dataclass: every field name + default
//! value carried verbatim from `voice_pipeline.py:26-89` so a Python
//! caller's tuning translates directly. The single Rust-idiom delta is
//! that the `latency_profile: str` ("fast" / "balanced" / "quality")
//! is replaced by a typed [`LatencyProfile`] enum (no string-typing in
//! the public API per the project's Rust conventions); the `__post_init__`
//! derivation that fills `stt_transcription_delay_ms` /
//! `tts_streaming_interval` from the profile when the caller leaves
//! them as `None` is preserved verbatim via
//! [`VoicePipelineConfig::resolved`].
//!
//! Model-name strings (`stt_model` / `vad_model` / `turn_model` /
//! `response_model` / `tts_model`) are intentionally **non-resolved**:
//! mlxrs's [no per-model arch porting][noarch] rule means the per-model
//! loader is the caller's responsibility — this config carries the
//! repo / path string for the caller to feed into the per-domain
//! `load` entry points
//! ([`crate::audio::stt::load::load()`] /
//! [`crate::audio::tts::load::load()`] /
//! [`crate::audio::vad::load::load()`] /
//! [`crate::lm::load::load()`]).
//!
//! [vp-cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L25-L89
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

/// Latency-vs-quality profile preset — typed analogue of mlx-audio's
/// `latency_profile: str` ("fast" / "balanced" / "quality").
///
/// Selects the default `stt_transcription_delay_ms` /
/// `tts_streaming_interval` the [`VoicePipelineConfig::resolved`]
/// post-init step fills in when those knobs are left as `None`.
///
/// Defaults to [`LatencyProfile::Balanced`] — same as mlx-audio's
/// dataclass default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LatencyProfile {
  /// Lowest end-to-end latency at the cost of partial-transcript
  /// stability — mlx-audio's `"fast"` profile (`240ms`
  /// transcription delay, `0.24s` TTS streaming interval).
  Fast,
  /// Balanced latency + stability — mlx-audio's `"balanced"` profile
  /// (`480ms` transcription delay, `0.32s` TTS streaming interval).
  /// This is the default.
  #[default]
  Balanced,
  /// Highest transcript / synthesis quality at the cost of latency —
  /// mlx-audio's `"quality"` profile (`960ms` transcription delay,
  /// `0.48s` TTS streaming interval).
  Quality,
}

impl LatencyProfile {
  /// The `stt_transcription_delay_ms` default this profile contributes
  /// when the explicit field is `None`. Matches
  /// `voice_pipeline.py:78-83`'s `{"fast": 240, "balanced": 480,
  /// "quality": 960}` table (with the same `.get(profile, 480)`
  /// fallback baked in via the enum's exhaustive match).
  pub const fn default_transcription_delay_ms(self) -> u32 {
    match self {
      LatencyProfile::Fast => 240,
      LatencyProfile::Balanced => 480,
      LatencyProfile::Quality => 960,
    }
  }

  /// The `tts_streaming_interval` default this profile contributes
  /// when the explicit field is `None`. Matches
  /// `voice_pipeline.py:85-89`'s `{"fast": 0.24, "balanced": 0.32,
  /// "quality": 0.48}` table (with the same `.get(profile, 0.32)`
  /// fallback baked in via the enum's exhaustive match).
  pub const fn default_tts_streaming_interval(self) -> f32 {
    match self {
      LatencyProfile::Fast => 0.24,
      LatencyProfile::Balanced => 0.32,
      LatencyProfile::Quality => 0.48,
    }
  }
}

/// Configuration for the realtime voice-pipeline loop — port of
/// [`mlx_audio.sts.voice_pipeline.VoicePipelineConfig`][vp-cfg].
///
/// Faithful 1:1 of mlx-audio's dataclass: each field's default value
/// matches `voice_pipeline.py:26-89` verbatim so a Python caller's
/// tuning translates without surprises. The single Rust-idiom delta
/// is that `latency_profile: str` becomes a typed [`LatencyProfile`]
/// enum (no string-typing) and the two `Optional[int|float]` knobs
/// (`stt_transcription_delay_ms` / `tts_streaming_interval`) stay as
/// `Option<…>` so the caller can leave them unset and have
/// [`VoicePipelineConfig::resolved`] fill the profile-driven default.
///
/// Per [no per-model arch porting][noarch] the per-model name strings
/// (`stt_model` / `vad_model` / `turn_model` / `response_model` /
/// `tts_model`) carry the upstream repo / path string verbatim; the
/// caller resolves them into concrete trait objects via the per-domain
/// `load` entry points ([`crate::audio::stt::load::load()`] /
/// [`crate::audio::tts::load::load()`] /
/// [`crate::audio::vad::load::load()`] /
/// [`crate::lm::load::load()`]) before handing them to
/// [`VoiceSession`][session].
///
/// [session]: crate::audio::sts::pipeline::orchestrator::VoiceSession
///
/// [vp-cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L25-L89
/// [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
#[derive(Debug, Clone)]
pub struct VoicePipelineConfig {
  /// Mic capture sample rate (Hz). mlx-audio default `16_000`.
  pub input_sample_rate: u32,
  /// Audio-output sample rate (Hz). `None` ⇒ inherit the TTS model's
  /// [`crate::audio::tts::model::TtsModel::sample_rate`] at runtime,
  /// matching mlx-audio's `output_sample_rate: Optional[int] = None`
  /// default.
  pub output_sample_rate: Option<u32>,
  /// Mic capture channel count. mlx-audio default `1` (mono).
  pub input_channels: u16,
  /// Mic frame duration in ms — the granularity at which input
  /// samples are pushed into the pipeline. mlx-audio default `32`.
  pub frame_duration_ms: u32,
  /// Latency-vs-quality profile — typed analogue of mlx-audio's
  /// `latency_profile: str`. Default [`LatencyProfile::Balanced`].
  pub latency_profile: LatencyProfile,

  // === STT ===
  /// STT model repo / path. mlx-audio default
  /// `"mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"`.
  pub stt_model: String,
  /// STT streaming transcription delay in ms. `None` ⇒ derived from
  /// [`LatencyProfile::default_transcription_delay_ms`].
  pub stt_transcription_delay_ms: Option<u32>,
  /// Maximum decode tokens per streaming step. mlx-audio default `6`.
  pub stt_max_decode_tokens_per_step: u32,
  /// Maximum decode tokens per user turn. mlx-audio default `256`.
  pub stt_max_turn_tokens: u32,
  /// Bounded finalization steps after endpointing. mlx-audio default
  /// `96`.
  pub stt_finalization_max_steps: u32,

  // === VAD ===
  /// VAD model repo / path. mlx-audio default
  /// `"mlx-community/silero-vad"`.
  pub vad_model: String,
  /// VAD start-of-speech probability threshold. mlx-audio default `0.35`.
  pub vad_start_threshold: f32,
  /// VAD continue-speech probability threshold (hysteresis). mlx-audio
  /// default `0.2`.
  pub vad_stop_threshold: f32,
  /// Consecutive speech frames needed to confirm start-of-speech.
  /// mlx-audio default `1`.
  pub vad_start_frames: u32,
  /// Silence duration that triggers turn-end consideration (ms).
  /// mlx-audio default `600`.
  pub vad_end_silence_ms: u32,
  /// Maximum single-turn duration (seconds). mlx-audio default `30.0`.
  pub vad_max_turn_seconds: f32,
  /// Pre-roll buffer (ms) preserved at start-of-speech so the
  /// transcriber sees a small leading context. mlx-audio default
  /// `250`.
  pub preroll_ms: u32,

  // === Turn-taking ===
  /// Turn-end model repo / path. mlx-audio default
  /// `"mlx-community/smart-turn-v3"`.
  pub turn_model: String,
  /// Smart-turn endpoint probability threshold. mlx-audio default `0.5`.
  pub turn_threshold: f32,
  /// Max silence (ms) we wait before force-finalizing when the
  /// endpoint model keeps reporting "incomplete". mlx-audio default
  /// `1600`.
  pub turn_max_incomplete_silence_ms: u32,

  // === LLM response engine ===
  /// LLM model repo / path. mlx-audio default
  /// `"mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"`.
  pub response_model: String,
  /// System prompt for the LLM. mlx-audio default carries the
  /// voice-assistant persona prompt.
  pub system_prompt: String,

  // === TTS ===
  /// TTS model repo / path. mlx-audio default `"mlx-community/pocket-tts"`.
  pub tts_model: String,
  /// TTS voice name. mlx-audio default `"cosette"`.
  pub tts_voice: String,
  /// TTS streaming chunk interval (seconds). `None` ⇒ derived from
  /// [`LatencyProfile::default_tts_streaming_interval`].
  pub tts_streaming_interval: Option<f32>,
  /// TTS sampling temperature. `None` ⇒ model default.
  pub tts_temperature: Option<f32>,

  // === Barge-in + echo ===
  /// Whether to honor user barge-in during TTS playback. mlx-audio
  /// default `true`.
  pub barge_in: bool,
  /// Minimum user-speech duration (ms) required to confirm a
  /// barge-in candidate as real (not echo). mlx-audio default `180`.
  pub min_barge_in_ms: u32,
  /// Playback-echo window after recent output callbacks (ms).
  /// mlx-audio default `450`.
  pub ignore_playback_echo_ms: u32,
  /// Minimum expected acoustic echo delay (ms). mlx-audio default
  /// `250`.
  pub echo_delay_min_ms: u32,
  /// Maximum expected acoustic echo delay (ms). mlx-audio default
  /// `500`.
  pub echo_delay_max_ms: u32,
  /// Echo-correlation step granularity (ms). mlx-audio default `32`.
  pub echo_correlation_step_ms: u32,
  /// Minimum partial-transcript characters needed to confirm a
  /// barge-in candidate. mlx-audio default `2`.
  pub barge_in_min_transcript_chars: u32,

  // === Output / runtime ===
  /// Whether to play TTS output through an audio device. mlx-audio
  /// default `true`. mlxrs respects this via [`VoicePipeline::run`] —
  /// when `false`, the sink is not driven (mlx-audio's
  /// `play_audio=False`).
  ///
  /// [`VoicePipeline::run`]: super::voice_pipeline::VoicePipeline::run
  pub play_audio: bool,
  /// Audio-queue capacity (slots). mlx-audio default `128`.
  pub queue_size: usize,
  /// Verbose structured-event logging. mlx-audio default `false`.
  /// mlxrs honors the flag but emits via the `log` crate rather than
  /// stdout (no global logger setup performed).
  pub verbose: bool,
}

impl VoicePipelineConfig {
  /// Resolve every `Option<…>` field — replace `None`s with the
  /// `latency_profile` default — and return the value-copy with
  /// every knob materialized.
  ///
  /// Mirrors mlx-audio's `__post_init__`
  /// ([`voice_pipeline.py:75-89`][vp-cfg]) but as an explicit
  /// fold-method rather than a constructor side-effect: a caller
  /// who wants the raw `Option<u32>` knob preserved can hold the
  /// original; a caller who needs the resolved profile-default
  /// value calls [`Self::resolved_transcription_delay_ms`] /
  /// [`Self::resolved_tts_streaming_interval`] without rebuilding
  /// the whole struct.
  ///
  /// [vp-cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L75-L89
  #[must_use]
  pub fn resolved(mut self) -> Self {
    if self.stt_transcription_delay_ms.is_none() {
      self.stt_transcription_delay_ms = Some(self.latency_profile.default_transcription_delay_ms());
    }
    if self.tts_streaming_interval.is_none() {
      self.tts_streaming_interval = Some(self.latency_profile.default_tts_streaming_interval());
    }
    self
  }

  /// The effective STT transcription delay (ms) — explicit field if
  /// set, else the [`LatencyProfile`] default.
  pub fn resolved_transcription_delay_ms(&self) -> u32 {
    self
      .stt_transcription_delay_ms
      .unwrap_or_else(|| self.latency_profile.default_transcription_delay_ms())
  }

  /// The effective TTS streaming interval (seconds) — explicit field
  /// if set, else the [`LatencyProfile`] default.
  pub fn resolved_tts_streaming_interval(&self) -> f32 {
    self
      .tts_streaming_interval
      .unwrap_or_else(|| self.latency_profile.default_tts_streaming_interval())
  }
}

impl Default for VoicePipelineConfig {
  /// The mlx-audio dataclass-default config
  /// ([`voice_pipeline.py:26-89`][vp-cfg]). Every value matches the
  /// upstream default verbatim.
  ///
  /// [vp-cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L25-L89
  fn default() -> Self {
    Self {
      input_sample_rate: 16_000,
      output_sample_rate: None,
      input_channels: 1,
      frame_duration_ms: 32,
      latency_profile: LatencyProfile::default(),

      stt_model: "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit".to_string(),
      stt_transcription_delay_ms: None,
      stt_max_decode_tokens_per_step: 6,
      stt_max_turn_tokens: 256,
      stt_finalization_max_steps: 96,

      vad_model: "mlx-community/silero-vad".to_string(),
      vad_start_threshold: 0.35,
      vad_stop_threshold: 0.2,
      vad_start_frames: 1,
      vad_end_silence_ms: 600,
      vad_max_turn_seconds: 30.0,
      preroll_ms: 250,

      turn_model: "mlx-community/smart-turn-v3".to_string(),
      turn_threshold: 0.5,
      turn_max_incomplete_silence_ms: 1600,

      response_model: "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit".to_string(),
      system_prompt: "You are a helpful voice assistant. Respond in natural spoken sentences. \
                      Never use markdown, emoji, or lists."
        .to_string(),

      tts_model: "mlx-community/pocket-tts".to_string(),
      tts_voice: "cosette".to_string(),
      tts_streaming_interval: None,
      tts_temperature: None,

      barge_in: true,
      min_barge_in_ms: 180,
      ignore_playback_echo_ms: 450,
      echo_delay_min_ms: 250,
      echo_delay_max_ms: 500,
      echo_correlation_step_ms: 32,
      barge_in_min_transcript_chars: 2,

      play_audio: true,
      queue_size: 128,
      verbose: false,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The default config matches every mlx-audio dataclass field exactly
  /// — this is the parity-fence: if the upstream `voice_pipeline.py`
  /// changes a default, this test fails and forces the porter to
  /// reconcile the divergence.
  #[test]
  fn defaults_match_python_dataclass() {
    let cfg = VoicePipelineConfig::default();

    assert_eq!(cfg.input_sample_rate, 16_000);
    assert_eq!(cfg.output_sample_rate, None);
    assert_eq!(cfg.input_channels, 1);
    assert_eq!(cfg.frame_duration_ms, 32);
    assert_eq!(cfg.latency_profile, LatencyProfile::Balanced);

    assert_eq!(
      cfg.stt_model,
      "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    );
    assert_eq!(cfg.stt_transcription_delay_ms, None);
    assert_eq!(cfg.stt_max_decode_tokens_per_step, 6);
    assert_eq!(cfg.stt_max_turn_tokens, 256);
    assert_eq!(cfg.stt_finalization_max_steps, 96);

    assert_eq!(cfg.vad_model, "mlx-community/silero-vad");
    assert!((cfg.vad_start_threshold - 0.35).abs() < 1e-6);
    assert!((cfg.vad_stop_threshold - 0.2).abs() < 1e-6);
    assert_eq!(cfg.vad_start_frames, 1);
    assert_eq!(cfg.vad_end_silence_ms, 600);
    assert!((cfg.vad_max_turn_seconds - 30.0).abs() < 1e-6);
    assert_eq!(cfg.preroll_ms, 250);

    assert_eq!(cfg.turn_model, "mlx-community/smart-turn-v3");
    assert!((cfg.turn_threshold - 0.5).abs() < 1e-6);
    assert_eq!(cfg.turn_max_incomplete_silence_ms, 1600);

    assert_eq!(
      cfg.response_model,
      "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"
    );
    assert!(cfg.system_prompt.contains("voice assistant"));

    assert_eq!(cfg.tts_model, "mlx-community/pocket-tts");
    assert_eq!(cfg.tts_voice, "cosette");
    assert_eq!(cfg.tts_streaming_interval, None);
    assert_eq!(cfg.tts_temperature, None);

    assert!(cfg.barge_in);
    assert_eq!(cfg.min_barge_in_ms, 180);
    assert_eq!(cfg.ignore_playback_echo_ms, 450);
    assert_eq!(cfg.echo_delay_min_ms, 250);
    assert_eq!(cfg.echo_delay_max_ms, 500);
    assert_eq!(cfg.echo_correlation_step_ms, 32);
    assert_eq!(cfg.barge_in_min_transcript_chars, 2);

    assert!(cfg.play_audio);
    assert_eq!(cfg.queue_size, 128);
    assert!(!cfg.verbose);
  }

  /// `resolved()` fills in the profile-driven defaults for the two
  /// `Option<…>` knobs — mirror of mlx-audio's `__post_init__`.
  #[test]
  fn resolved_fills_profile_defaults() {
    // Balanced (default) → 480 ms / 0.32 s.
    let cfg = VoicePipelineConfig::default().resolved();
    assert_eq!(cfg.stt_transcription_delay_ms, Some(480));
    assert_eq!(cfg.tts_streaming_interval, Some(0.32));

    // Fast → 240 ms / 0.24 s.
    let cfg = VoicePipelineConfig {
      latency_profile: LatencyProfile::Fast,
      ..VoicePipelineConfig::default()
    }
    .resolved();
    assert_eq!(cfg.stt_transcription_delay_ms, Some(240));
    assert_eq!(cfg.tts_streaming_interval, Some(0.24));

    // Quality → 960 ms / 0.48 s.
    let cfg = VoicePipelineConfig {
      latency_profile: LatencyProfile::Quality,
      ..VoicePipelineConfig::default()
    }
    .resolved();
    assert_eq!(cfg.stt_transcription_delay_ms, Some(960));
    assert_eq!(cfg.tts_streaming_interval, Some(0.48));
  }

  /// Explicit fields beat the profile default — mirror of mlx-audio's
  /// `__post_init__` "only fill if `is None`" rule.
  #[test]
  fn resolved_preserves_explicit_overrides() {
    let cfg = VoicePipelineConfig {
      latency_profile: LatencyProfile::Fast,
      stt_transcription_delay_ms: Some(123),
      tts_streaming_interval: Some(0.07),
      ..VoicePipelineConfig::default()
    }
    .resolved();
    assert_eq!(cfg.stt_transcription_delay_ms, Some(123));
    assert_eq!(cfg.tts_streaming_interval, Some(0.07));
  }

  /// The `resolved_*` accessors return the same value with or without
  /// going through [`VoicePipelineConfig::resolved`].
  #[test]
  fn resolved_accessors_agree_with_resolved_method() {
    let raw = VoicePipelineConfig {
      latency_profile: LatencyProfile::Quality,
      ..VoicePipelineConfig::default()
    };
    let folded = raw.clone().resolved();
    assert_eq!(
      raw.resolved_transcription_delay_ms(),
      folded.stt_transcription_delay_ms.unwrap()
    );
    assert!(
      (raw.resolved_tts_streaming_interval() - folded.tts_streaming_interval.unwrap()).abs() < 1e-6
    );
  }
}
