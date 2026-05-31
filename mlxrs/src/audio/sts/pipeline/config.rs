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
//! mlxrs's no per-model arch porting rule means the per-model
//! loader is the caller's responsibility — this config carries the
//! repo / path string for the caller to feed into the per-domain
//! `load` entry points
//! ([`crate::audio::stt::load::load()`] /
//! [`crate::audio::tts::load::load()`] /
//! [`crate::audio::vad::load::load()`] /
//! [`crate::lm::load::load()`]).
//!
//! [vp-cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L25-L89

use derive_more::Display;

/// Latency-vs-quality profile preset — typed analogue of mlx-audio's
/// `latency_profile: str` ("fast" / "balanced" / "quality").
///
/// Selects the default `stt_transcription_delay_ms` /
/// `tts_streaming_interval` the [`VoicePipelineConfig::resolved`]
/// post-init step fills in when those knobs are left as `None`.
///
/// Defaults to [`LatencyProfile::Balanced`] — same as mlx-audio's
/// dataclass default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Display)]
#[display("{}", self.as_str())]
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
  /// The lowercase snake-case string label for this profile —
  /// `"low_latency"` / `"balanced"` / `"high_quality"`.
  ///
  /// Matches mlx-audio's `latency_profile: str` values after the
  /// Rust-to-Python name mapping.
  pub const fn as_str(self) -> &'static str {
    match self {
      LatencyProfile::Fast => "low_latency",
      LatencyProfile::Balanced => "balanced",
      LatencyProfile::Quality => "high_quality",
    }
  }

  /// Whether this is the [`LatencyProfile::Fast`] variant.
  #[inline(always)]
  #[must_use]
  pub fn is_fast(self) -> bool {
    matches!(self, LatencyProfile::Fast)
  }

  /// Whether this is the [`LatencyProfile::Balanced`] variant.
  #[inline(always)]
  #[must_use]
  pub fn is_balanced(self) -> bool {
    matches!(self, LatencyProfile::Balanced)
  }

  /// Whether this is the [`LatencyProfile::Quality`] variant.
  #[inline(always)]
  #[must_use]
  pub fn is_quality(self) -> bool {
    matches!(self, LatencyProfile::Quality)
  }

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
/// Construct via [`VoicePipelineConfig::new`] (= [`Default::default`])
/// and tune via the `with_*` builder methods (each returns `Self`
/// and is `#[must_use]`).
///
/// Per no per-model arch porting the per-model name strings
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
#[derive(Debug, Clone)]
pub struct VoicePipelineConfig {
  /// Mic capture sample rate (Hz). mlx-audio default `16_000`.
  input_sample_rate: u32,
  /// Audio-output sample rate (Hz). `None` ⇒ inherit the TTS model's
  /// [`crate::audio::tts::model::TtsModel::sample_rate`] at runtime,
  /// matching mlx-audio's `output_sample_rate: Optional[int] = None`
  /// default.
  output_sample_rate: Option<u32>,
  /// Mic capture channel count. mlx-audio default `1` (mono).
  input_channels: u16,
  /// Mic frame duration in ms — the granularity at which input
  /// samples are pushed into the pipeline. mlx-audio default `32`.
  frame_duration_ms: u32,
  /// Latency-vs-quality profile — typed analogue of mlx-audio's
  /// `latency_profile: str`. Default [`LatencyProfile::Balanced`].
  latency_profile: LatencyProfile,

  // === STT ===
  /// STT model repo / path. mlx-audio default
  /// `"mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"`.
  stt_model: String,
  /// STT streaming transcription delay in ms. `None` ⇒ derived from
  /// [`LatencyProfile::default_transcription_delay_ms`].
  stt_transcription_delay_ms: Option<u32>,
  /// Maximum decode tokens per streaming step. mlx-audio default `6`.
  stt_max_decode_tokens_per_step: u32,
  /// Maximum decode tokens per user turn. mlx-audio default `256`.
  stt_max_turn_tokens: u32,
  /// Bounded finalization steps after endpointing. mlx-audio default
  /// `96`.
  stt_finalization_max_steps: u32,

  // === VAD ===
  /// VAD model repo / path. mlx-audio default
  /// `"mlx-community/silero-vad"`.
  vad_model: String,
  /// VAD start-of-speech probability threshold. mlx-audio default `0.35`.
  vad_start_threshold: f32,
  /// VAD continue-speech probability threshold (hysteresis). mlx-audio
  /// default `0.2`.
  vad_stop_threshold: f32,
  /// Consecutive speech frames needed to confirm start-of-speech.
  /// mlx-audio default `1`.
  vad_start_frames: u32,
  /// Silence duration that triggers turn-end consideration (ms).
  /// mlx-audio default `600`.
  vad_end_silence_ms: u32,
  /// Maximum single-turn duration (seconds). mlx-audio default `30.0`.
  vad_max_turn_seconds: f32,
  /// Pre-roll buffer (ms) preserved at start-of-speech so the
  /// transcriber sees a small leading context. mlx-audio default
  /// `250`.
  preroll_ms: u32,

  // === Turn-taking ===
  /// Turn-end model repo / path. mlx-audio default
  /// `"mlx-community/smart-turn-v3"`.
  turn_model: String,
  /// Smart-turn endpoint probability threshold. mlx-audio default `0.5`.
  turn_threshold: f32,
  /// Max silence (ms) we wait before force-finalizing when the
  /// endpoint model keeps reporting "incomplete". mlx-audio default
  /// `1600`.
  turn_max_incomplete_silence_ms: u32,

  // === LLM response engine ===
  /// LLM model repo / path. mlx-audio default
  /// `"mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"`.
  response_model: String,
  /// System prompt for the LLM. mlx-audio default carries the
  /// voice-assistant persona prompt.
  system_prompt: String,

  // === TTS ===
  /// TTS model repo / path. mlx-audio default `"mlx-community/pocket-tts"`.
  tts_model: String,
  /// TTS voice name. mlx-audio default `"cosette"`.
  tts_voice: String,
  /// TTS streaming chunk interval (seconds). `None` ⇒ derived from
  /// [`LatencyProfile::default_tts_streaming_interval`].
  tts_streaming_interval: Option<f32>,
  /// TTS sampling temperature. `None` ⇒ model default.
  tts_temperature: Option<f32>,

  // === Barge-in + echo ===
  /// Whether to honor user barge-in during TTS playback. mlx-audio
  /// default `true`.
  barge_in: bool,
  /// Minimum user-speech duration (ms) required to confirm a
  /// barge-in candidate as real (not echo). mlx-audio default `180`.
  min_barge_in_ms: u32,
  /// Playback-echo window after recent output callbacks (ms).
  /// mlx-audio default `450`.
  ignore_playback_echo_ms: u32,
  /// Minimum expected acoustic echo delay (ms). mlx-audio default
  /// `250`.
  echo_delay_min_ms: u32,
  /// Maximum expected acoustic echo delay (ms). mlx-audio default
  /// `500`.
  echo_delay_max_ms: u32,
  /// Echo-correlation step granularity (ms). mlx-audio default `32`.
  echo_correlation_step_ms: u32,
  /// Minimum partial-transcript characters needed to confirm a
  /// barge-in candidate. mlx-audio default `2`.
  barge_in_min_transcript_chars: u32,

  // === Output / runtime ===
  /// Whether to play TTS output through an audio device. mlx-audio
  /// default `true`. mlxrs respects this via [`VoicePipeline::run`] —
  /// when `false`, the sink is not driven (mlx-audio's
  /// `play_audio=False`).
  ///
  /// [`VoicePipeline::run`]: super::voice_pipeline::VoicePipeline::run
  play_audio: bool,
  /// Audio-queue capacity (slots). mlx-audio default `128`.
  queue_size: usize,
  /// Verbose structured-event logging. mlx-audio default `false`.
  /// mlxrs honors the flag but emits via the `log` crate rather than
  /// stdout (no global logger setup performed).
  verbose: bool,
}

impl VoicePipelineConfig {
  /// Construct a config with all mlx-audio dataclass defaults
  /// ([`voice_pipeline.py:26-89`][vp-cfg]). Identical to
  /// [`Default::default`]; provided as an explicit constructor so
  /// callers can write `VoicePipelineConfig::new().with_*(…)` chains.
  ///
  /// [vp-cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L25-L89
  #[must_use]
  pub fn new() -> Self {
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

  // ── Accessors ────────────────────────────────────────────────────────

  /// Mic capture sample rate (Hz).
  #[inline(always)]
  #[must_use]
  pub fn input_sample_rate(&self) -> u32 {
    self.input_sample_rate
  }

  /// Audio-output sample rate override (Hz), or `None` to inherit from
  /// the TTS model at runtime.
  #[inline(always)]
  #[must_use]
  pub fn output_sample_rate(&self) -> Option<u32> {
    self.output_sample_rate
  }

  /// Mic capture channel count.
  #[inline(always)]
  #[must_use]
  pub fn input_channels(&self) -> u16 {
    self.input_channels
  }

  /// Mic frame duration (ms).
  #[inline(always)]
  #[must_use]
  pub fn frame_duration_ms(&self) -> u32 {
    self.frame_duration_ms
  }

  /// Latency-vs-quality profile preset.
  #[inline(always)]
  #[must_use]
  pub fn latency_profile(&self) -> LatencyProfile {
    self.latency_profile
  }

  /// STT model repo / path.
  #[inline(always)]
  #[must_use]
  pub fn stt_model(&self) -> &str {
    &self.stt_model
  }

  /// STT streaming transcription delay override (ms), or `None`.
  #[inline(always)]
  #[must_use]
  pub fn stt_transcription_delay_ms(&self) -> Option<u32> {
    self.stt_transcription_delay_ms
  }

  /// Maximum decode tokens per streaming step.
  #[inline(always)]
  #[must_use]
  pub fn stt_max_decode_tokens_per_step(&self) -> u32 {
    self.stt_max_decode_tokens_per_step
  }

  /// Maximum decode tokens per user turn.
  #[inline(always)]
  #[must_use]
  pub fn stt_max_turn_tokens(&self) -> u32 {
    self.stt_max_turn_tokens
  }

  /// Bounded finalization steps after endpointing.
  #[inline(always)]
  #[must_use]
  pub fn stt_finalization_max_steps(&self) -> u32 {
    self.stt_finalization_max_steps
  }

  /// VAD model repo / path.
  #[inline(always)]
  #[must_use]
  pub fn vad_model(&self) -> &str {
    &self.vad_model
  }

  /// VAD start-of-speech probability threshold.
  #[inline(always)]
  #[must_use]
  pub fn vad_start_threshold(&self) -> f32 {
    self.vad_start_threshold
  }

  /// VAD continue-speech probability threshold (hysteresis).
  #[inline(always)]
  #[must_use]
  pub fn vad_stop_threshold(&self) -> f32 {
    self.vad_stop_threshold
  }

  /// Consecutive speech frames needed to confirm start-of-speech.
  #[inline(always)]
  #[must_use]
  pub fn vad_start_frames(&self) -> u32 {
    self.vad_start_frames
  }

  /// Silence duration that triggers turn-end consideration (ms).
  #[inline(always)]
  #[must_use]
  pub fn vad_end_silence_ms(&self) -> u32 {
    self.vad_end_silence_ms
  }

  /// Maximum single-turn duration (seconds).
  #[inline(always)]
  #[must_use]
  pub fn vad_max_turn_seconds(&self) -> f32 {
    self.vad_max_turn_seconds
  }

  /// Pre-roll buffer (ms) preserved at start-of-speech.
  #[inline(always)]
  #[must_use]
  pub fn preroll_ms(&self) -> u32 {
    self.preroll_ms
  }

  /// Turn-end model repo / path.
  #[inline(always)]
  #[must_use]
  pub fn turn_model(&self) -> &str {
    &self.turn_model
  }

  /// Smart-turn endpoint probability threshold.
  #[inline(always)]
  #[must_use]
  pub fn turn_threshold(&self) -> f32 {
    self.turn_threshold
  }

  /// Max silence (ms) before force-finalizing an incomplete turn.
  #[inline(always)]
  #[must_use]
  pub fn turn_max_incomplete_silence_ms(&self) -> u32 {
    self.turn_max_incomplete_silence_ms
  }

  /// LLM model repo / path.
  #[inline(always)]
  #[must_use]
  pub fn response_model(&self) -> &str {
    &self.response_model
  }

  /// System prompt for the LLM.
  #[inline(always)]
  #[must_use]
  pub fn system_prompt(&self) -> &str {
    &self.system_prompt
  }

  /// TTS model repo / path.
  #[inline(always)]
  #[must_use]
  pub fn tts_model(&self) -> &str {
    &self.tts_model
  }

  /// TTS voice name.
  #[inline(always)]
  #[must_use]
  pub fn tts_voice(&self) -> &str {
    &self.tts_voice
  }

  /// TTS streaming chunk interval override (seconds), or `None`.
  #[inline(always)]
  #[must_use]
  pub fn tts_streaming_interval(&self) -> Option<f32> {
    self.tts_streaming_interval
  }

  /// TTS sampling temperature override, or `None` for model default.
  #[inline(always)]
  #[must_use]
  pub fn tts_temperature(&self) -> Option<f32> {
    self.tts_temperature
  }

  /// Whether barge-in is enabled.
  #[inline(always)]
  #[must_use]
  pub fn barge_in(&self) -> bool {
    self.barge_in
  }

  /// Minimum user-speech duration (ms) to confirm barge-in.
  #[inline(always)]
  #[must_use]
  pub fn min_barge_in_ms(&self) -> u32 {
    self.min_barge_in_ms
  }

  /// Playback-echo window after recent output callbacks (ms).
  #[inline(always)]
  #[must_use]
  pub fn ignore_playback_echo_ms(&self) -> u32 {
    self.ignore_playback_echo_ms
  }

  /// Minimum expected acoustic echo delay (ms).
  #[inline(always)]
  #[must_use]
  pub fn echo_delay_min_ms(&self) -> u32 {
    self.echo_delay_min_ms
  }

  /// Maximum expected acoustic echo delay (ms).
  #[inline(always)]
  #[must_use]
  pub fn echo_delay_max_ms(&self) -> u32 {
    self.echo_delay_max_ms
  }

  /// Echo-correlation step granularity (ms).
  #[inline(always)]
  #[must_use]
  pub fn echo_correlation_step_ms(&self) -> u32 {
    self.echo_correlation_step_ms
  }

  /// Minimum partial-transcript characters to confirm barge-in.
  #[inline(always)]
  #[must_use]
  pub fn barge_in_min_transcript_chars(&self) -> u32 {
    self.barge_in_min_transcript_chars
  }

  /// Whether to play TTS output through an audio device.
  #[inline(always)]
  #[must_use]
  pub fn play_audio(&self) -> bool {
    self.play_audio
  }

  /// Audio-queue capacity (slots).
  #[inline(always)]
  #[must_use]
  pub fn queue_size(&self) -> usize {
    self.queue_size
  }

  /// Verbose structured-event logging.
  #[inline(always)]
  #[must_use]
  pub fn verbose(&self) -> bool {
    self.verbose
  }

  // ── Builder methods ──────────────────────────────────────────────────

  /// Set the mic capture sample rate (Hz).
  #[must_use]
  pub fn with_input_sample_rate(mut self, v: u32) -> Self {
    self.input_sample_rate = v;
    self
  }

  /// Override the audio-output sample rate (Hz). Pass `None` to
  /// inherit from the TTS model at runtime.
  #[must_use]
  pub fn with_output_sample_rate(mut self, v: Option<u32>) -> Self {
    self.output_sample_rate = v;
    self
  }

  /// Set the mic capture channel count.
  #[must_use]
  pub fn with_input_channels(mut self, v: u16) -> Self {
    self.input_channels = v;
    self
  }

  /// Set the mic frame duration (ms).
  #[must_use]
  pub fn with_frame_duration_ms(mut self, v: u32) -> Self {
    self.frame_duration_ms = v;
    self
  }

  /// Set the latency-vs-quality profile preset.
  #[must_use]
  pub fn with_latency_profile(mut self, v: LatencyProfile) -> Self {
    self.latency_profile = v;
    self
  }

  /// Set the STT model repo / path.
  #[must_use]
  pub fn with_stt_model(mut self, v: impl Into<String>) -> Self {
    self.stt_model = v.into();
    self
  }

  /// Override the STT streaming transcription delay (ms). Pass `None`
  /// to derive from the latency profile.
  #[must_use]
  pub fn with_stt_transcription_delay_ms(mut self, v: Option<u32>) -> Self {
    self.stt_transcription_delay_ms = v;
    self
  }

  /// Set the maximum decode tokens per streaming step.
  #[must_use]
  pub fn with_stt_max_decode_tokens_per_step(mut self, v: u32) -> Self {
    self.stt_max_decode_tokens_per_step = v;
    self
  }

  /// Set the maximum decode tokens per user turn.
  #[must_use]
  pub fn with_stt_max_turn_tokens(mut self, v: u32) -> Self {
    self.stt_max_turn_tokens = v;
    self
  }

  /// Set the bounded finalization steps after endpointing.
  #[must_use]
  pub fn with_stt_finalization_max_steps(mut self, v: u32) -> Self {
    self.stt_finalization_max_steps = v;
    self
  }

  /// Set the VAD model repo / path.
  #[must_use]
  pub fn with_vad_model(mut self, v: impl Into<String>) -> Self {
    self.vad_model = v.into();
    self
  }

  /// Set the VAD start-of-speech probability threshold.
  #[must_use]
  pub fn with_vad_start_threshold(mut self, v: f32) -> Self {
    self.vad_start_threshold = v;
    self
  }

  /// Set the VAD continue-speech probability threshold (hysteresis).
  #[must_use]
  pub fn with_vad_stop_threshold(mut self, v: f32) -> Self {
    self.vad_stop_threshold = v;
    self
  }

  /// Set the consecutive speech frames needed to confirm start-of-speech.
  #[must_use]
  pub fn with_vad_start_frames(mut self, v: u32) -> Self {
    self.vad_start_frames = v;
    self
  }

  /// Set the silence duration that triggers turn-end consideration (ms).
  #[must_use]
  pub fn with_vad_end_silence_ms(mut self, v: u32) -> Self {
    self.vad_end_silence_ms = v;
    self
  }

  /// Set the maximum single-turn duration (seconds).
  #[must_use]
  pub fn with_vad_max_turn_seconds(mut self, v: f32) -> Self {
    self.vad_max_turn_seconds = v;
    self
  }

  /// Set the pre-roll buffer duration (ms).
  #[must_use]
  pub fn with_preroll_ms(mut self, v: u32) -> Self {
    self.preroll_ms = v;
    self
  }

  /// Set the turn-end model repo / path.
  #[must_use]
  pub fn with_turn_model(mut self, v: impl Into<String>) -> Self {
    self.turn_model = v.into();
    self
  }

  /// Set the smart-turn endpoint probability threshold.
  #[must_use]
  pub fn with_turn_threshold(mut self, v: f32) -> Self {
    self.turn_threshold = v;
    self
  }

  /// Set the max silence (ms) before force-finalizing an incomplete turn.
  #[must_use]
  pub fn with_turn_max_incomplete_silence_ms(mut self, v: u32) -> Self {
    self.turn_max_incomplete_silence_ms = v;
    self
  }

  /// Set the LLM model repo / path.
  #[must_use]
  pub fn with_response_model(mut self, v: impl Into<String>) -> Self {
    self.response_model = v.into();
    self
  }

  /// Set the system prompt for the LLM.
  #[must_use]
  pub fn with_system_prompt(mut self, v: impl Into<String>) -> Self {
    self.system_prompt = v.into();
    self
  }

  /// Set the TTS model repo / path.
  #[must_use]
  pub fn with_tts_model(mut self, v: impl Into<String>) -> Self {
    self.tts_model = v.into();
    self
  }

  /// Set the TTS voice name.
  #[must_use]
  pub fn with_tts_voice(mut self, v: impl Into<String>) -> Self {
    self.tts_voice = v.into();
    self
  }

  /// Override the TTS streaming chunk interval (seconds). Pass `None`
  /// to derive from the latency profile.
  #[must_use]
  pub fn with_tts_streaming_interval(mut self, v: Option<f32>) -> Self {
    self.tts_streaming_interval = v;
    self
  }

  /// Override the TTS sampling temperature. Pass `None` for model
  /// default.
  #[must_use]
  pub fn with_tts_temperature(mut self, v: Option<f32>) -> Self {
    self.tts_temperature = v;
    self
  }

  /// Enable or disable barge-in.
  #[must_use]
  pub fn with_barge_in(mut self, v: bool) -> Self {
    self.barge_in = v;
    self
  }

  /// Set the minimum user-speech duration (ms) to confirm barge-in.
  #[must_use]
  pub fn with_min_barge_in_ms(mut self, v: u32) -> Self {
    self.min_barge_in_ms = v;
    self
  }

  /// Set the playback-echo window (ms).
  #[must_use]
  pub fn with_ignore_playback_echo_ms(mut self, v: u32) -> Self {
    self.ignore_playback_echo_ms = v;
    self
  }

  /// Set the minimum expected acoustic echo delay (ms).
  #[must_use]
  pub fn with_echo_delay_min_ms(mut self, v: u32) -> Self {
    self.echo_delay_min_ms = v;
    self
  }

  /// Set the maximum expected acoustic echo delay (ms).
  #[must_use]
  pub fn with_echo_delay_max_ms(mut self, v: u32) -> Self {
    self.echo_delay_max_ms = v;
    self
  }

  /// Set the echo-correlation step granularity (ms).
  #[must_use]
  pub fn with_echo_correlation_step_ms(mut self, v: u32) -> Self {
    self.echo_correlation_step_ms = v;
    self
  }

  /// Set the minimum partial-transcript characters to confirm barge-in.
  #[must_use]
  pub fn with_barge_in_min_transcript_chars(mut self, v: u32) -> Self {
    self.barge_in_min_transcript_chars = v;
    self
  }

  /// Enable or disable playing TTS output through an audio device.
  #[must_use]
  pub fn with_play_audio(mut self, v: bool) -> Self {
    self.play_audio = v;
    self
  }

  /// Set the audio-queue capacity (slots).
  #[must_use]
  pub fn with_queue_size(mut self, v: usize) -> Self {
    self.queue_size = v;
    self
  }

  /// Enable or disable verbose structured-event logging.
  #[must_use]
  pub fn with_verbose(mut self, v: bool) -> Self {
    self.verbose = v;
    self
  }

  // ── Resolution helpers ───────────────────────────────────────────────

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
  /// upstream default verbatim. Delegates to [`VoicePipelineConfig::new`].
  ///
  /// [vp-cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L25-L89
  fn default() -> Self {
    Self::new()
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
    let cfg = VoicePipelineConfig::new();

    assert_eq!(cfg.input_sample_rate(), 16_000);
    assert_eq!(cfg.output_sample_rate(), None);
    assert_eq!(cfg.input_channels(), 1);
    assert_eq!(cfg.frame_duration_ms(), 32);
    assert_eq!(cfg.latency_profile(), LatencyProfile::Balanced);

    assert_eq!(
      cfg.stt_model(),
      "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    );
    assert_eq!(cfg.stt_transcription_delay_ms(), None);
    assert_eq!(cfg.stt_max_decode_tokens_per_step(), 6);
    assert_eq!(cfg.stt_max_turn_tokens(), 256);
    assert_eq!(cfg.stt_finalization_max_steps(), 96);

    assert_eq!(cfg.vad_model(), "mlx-community/silero-vad");
    assert!((cfg.vad_start_threshold() - 0.35).abs() < 1e-6);
    assert!((cfg.vad_stop_threshold() - 0.2).abs() < 1e-6);
    assert_eq!(cfg.vad_start_frames(), 1);
    assert_eq!(cfg.vad_end_silence_ms(), 600);
    assert!((cfg.vad_max_turn_seconds() - 30.0).abs() < 1e-6);
    assert_eq!(cfg.preroll_ms(), 250);

    assert_eq!(cfg.turn_model(), "mlx-community/smart-turn-v3");
    assert!((cfg.turn_threshold() - 0.5).abs() < 1e-6);
    assert_eq!(cfg.turn_max_incomplete_silence_ms(), 1600);

    assert_eq!(
      cfg.response_model(),
      "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"
    );
    assert!(cfg.system_prompt().contains("voice assistant"));

    assert_eq!(cfg.tts_model(), "mlx-community/pocket-tts");
    assert_eq!(cfg.tts_voice(), "cosette");
    assert_eq!(cfg.tts_streaming_interval(), None);
    assert_eq!(cfg.tts_temperature(), None);

    assert!(cfg.barge_in());
    assert_eq!(cfg.min_barge_in_ms(), 180);
    assert_eq!(cfg.ignore_playback_echo_ms(), 450);
    assert_eq!(cfg.echo_delay_min_ms(), 250);
    assert_eq!(cfg.echo_delay_max_ms(), 500);
    assert_eq!(cfg.echo_correlation_step_ms(), 32);
    assert_eq!(cfg.barge_in_min_transcript_chars(), 2);

    assert!(cfg.play_audio());
    assert_eq!(cfg.queue_size(), 128);
    assert!(!cfg.verbose());
  }

  /// `resolved()` fills in the profile-driven defaults for the two
  /// `Option<…>` knobs — mirror of mlx-audio's `__post_init__`.
  #[test]
  fn resolved_fills_profile_defaults() {
    // Balanced (default) → 480 ms / 0.32 s.
    let cfg = VoicePipelineConfig::new().resolved();
    assert_eq!(cfg.stt_transcription_delay_ms(), Some(480));
    assert_eq!(cfg.tts_streaming_interval(), Some(0.32));

    // Fast → 240 ms / 0.24 s.
    let cfg = VoicePipelineConfig::new()
      .with_latency_profile(LatencyProfile::Fast)
      .resolved();
    assert_eq!(cfg.stt_transcription_delay_ms(), Some(240));
    assert_eq!(cfg.tts_streaming_interval(), Some(0.24));

    // Quality → 960 ms / 0.48 s.
    let cfg = VoicePipelineConfig::new()
      .with_latency_profile(LatencyProfile::Quality)
      .resolved();
    assert_eq!(cfg.stt_transcription_delay_ms(), Some(960));
    assert_eq!(cfg.tts_streaming_interval(), Some(0.48));
  }

  /// Explicit fields beat the profile default — mirror of mlx-audio's
  /// `__post_init__` "only fill if `is None`" rule.
  #[test]
  fn resolved_preserves_explicit_overrides() {
    let cfg = VoicePipelineConfig::new()
      .with_latency_profile(LatencyProfile::Fast)
      .with_stt_transcription_delay_ms(Some(123))
      .with_tts_streaming_interval(Some(0.07))
      .resolved();
    assert_eq!(cfg.stt_transcription_delay_ms(), Some(123));
    assert_eq!(cfg.tts_streaming_interval(), Some(0.07));
  }

  /// The `resolved_*` accessors return the same value with or without
  /// going through [`VoicePipelineConfig::resolved`].
  #[test]
  fn resolved_accessors_agree_with_resolved_method() {
    let raw = VoicePipelineConfig::new().with_latency_profile(LatencyProfile::Quality);
    let folded = raw.clone().resolved();
    assert_eq!(
      raw.resolved_transcription_delay_ms(),
      folded.stt_transcription_delay_ms().unwrap()
    );
    assert!(
      (raw.resolved_tts_streaming_interval() - folded.tts_streaming_interval().unwrap()).abs()
        < 1e-6
    );
  }

  /// `LatencyProfile::as_str` returns the expected snake-case labels.
  #[test]
  fn latency_profile_as_str() {
    assert_eq!(LatencyProfile::Fast.as_str(), "low_latency");
    assert_eq!(LatencyProfile::Balanced.as_str(), "balanced");
    assert_eq!(LatencyProfile::Quality.as_str(), "high_quality");
  }

  /// `Display` for `LatencyProfile` delegates to `as_str`.
  #[test]
  fn latency_profile_display() {
    assert_eq!(LatencyProfile::Fast.to_string(), "low_latency");
    assert_eq!(LatencyProfile::Balanced.to_string(), "balanced");
    assert_eq!(LatencyProfile::Quality.to_string(), "high_quality");
  }

  /// `is_*` variant predicate methods.
  #[test]
  fn latency_profile_is_variant_predicates() {
    assert!(LatencyProfile::Fast.is_fast());
    assert!(!LatencyProfile::Fast.is_balanced());
    assert!(!LatencyProfile::Fast.is_quality());

    assert!(!LatencyProfile::Balanced.is_fast());
    assert!(LatencyProfile::Balanced.is_balanced());
    assert!(!LatencyProfile::Balanced.is_quality());

    assert!(!LatencyProfile::Quality.is_fast());
    assert!(!LatencyProfile::Quality.is_balanced());
    assert!(LatencyProfile::Quality.is_quality());
  }

  /// `with_*` builder chain round-trips all settable fields.
  #[test]
  fn builder_chain_sets_fields() {
    let cfg = VoicePipelineConfig::new()
      .with_input_sample_rate(8_000)
      .with_output_sample_rate(Some(24_000))
      .with_input_channels(2)
      .with_frame_duration_ms(16)
      .with_latency_profile(LatencyProfile::Fast)
      .with_stt_model("my-stt")
      .with_stt_transcription_delay_ms(Some(100))
      .with_stt_max_decode_tokens_per_step(3)
      .with_stt_max_turn_tokens(128)
      .with_stt_finalization_max_steps(48)
      .with_vad_model("my-vad")
      .with_vad_start_threshold(0.5)
      .with_vad_stop_threshold(0.3)
      .with_vad_start_frames(2)
      .with_vad_end_silence_ms(300)
      .with_vad_max_turn_seconds(15.0)
      .with_preroll_ms(100)
      .with_turn_model("my-turn")
      .with_turn_threshold(0.7)
      .with_turn_max_incomplete_silence_ms(800)
      .with_response_model("my-llm")
      .with_system_prompt("be concise")
      .with_tts_model("my-tts")
      .with_tts_voice("alice")
      .with_tts_streaming_interval(Some(0.1))
      .with_tts_temperature(Some(0.8))
      .with_barge_in(false)
      .with_min_barge_in_ms(90)
      .with_ignore_playback_echo_ms(200)
      .with_echo_delay_min_ms(100)
      .with_echo_delay_max_ms(300)
      .with_echo_correlation_step_ms(16)
      .with_barge_in_min_transcript_chars(5)
      .with_play_audio(false)
      .with_queue_size(64)
      .with_verbose(true);

    assert_eq!(cfg.input_sample_rate(), 8_000);
    assert_eq!(cfg.output_sample_rate(), Some(24_000));
    assert_eq!(cfg.input_channels(), 2);
    assert_eq!(cfg.frame_duration_ms(), 16);
    assert_eq!(cfg.latency_profile(), LatencyProfile::Fast);
    assert_eq!(cfg.stt_model(), "my-stt");
    assert_eq!(cfg.stt_transcription_delay_ms(), Some(100));
    assert_eq!(cfg.stt_max_decode_tokens_per_step(), 3);
    assert_eq!(cfg.stt_max_turn_tokens(), 128);
    assert_eq!(cfg.stt_finalization_max_steps(), 48);
    assert_eq!(cfg.vad_model(), "my-vad");
    assert!((cfg.vad_start_threshold() - 0.5).abs() < 1e-6);
    assert!((cfg.vad_stop_threshold() - 0.3).abs() < 1e-6);
    assert_eq!(cfg.vad_start_frames(), 2);
    assert_eq!(cfg.vad_end_silence_ms(), 300);
    assert!((cfg.vad_max_turn_seconds() - 15.0).abs() < 1e-6);
    assert_eq!(cfg.preroll_ms(), 100);
    assert_eq!(cfg.turn_model(), "my-turn");
    assert!((cfg.turn_threshold() - 0.7).abs() < 1e-6);
    assert_eq!(cfg.turn_max_incomplete_silence_ms(), 800);
    assert_eq!(cfg.response_model(), "my-llm");
    assert_eq!(cfg.system_prompt(), "be concise");
    assert_eq!(cfg.tts_model(), "my-tts");
    assert_eq!(cfg.tts_voice(), "alice");
    assert_eq!(cfg.tts_streaming_interval(), Some(0.1));
    assert_eq!(cfg.tts_temperature(), Some(0.8));
    assert!(!cfg.barge_in());
    assert_eq!(cfg.min_barge_in_ms(), 90);
    assert_eq!(cfg.ignore_playback_echo_ms(), 200);
    assert_eq!(cfg.echo_delay_min_ms(), 100);
    assert_eq!(cfg.echo_delay_max_ms(), 300);
    assert_eq!(cfg.echo_correlation_step_ms(), 16);
    assert_eq!(cfg.barge_in_min_transcript_chars(), 5);
    assert!(!cfg.play_audio());
    assert_eq!(cfg.queue_size(), 64);
    assert!(cfg.verbose());
  }
}
