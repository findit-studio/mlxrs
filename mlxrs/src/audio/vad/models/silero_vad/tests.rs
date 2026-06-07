//! Silero VAD parity tests — mirror of mlx-audio's
//! [`tests/test_silero_vad.py`][t]. The probability / segment oracles compare
//! against the reference's own expected values (an independent oracle, not the
//! code under test): the `probs_to_timestamps` hysteresis is checked against
//! the verbatim reference test vector, and the forward / streaming tests assert
//! the reference's documented output shapes and `[0, 1]` probability range.
//!
//! No real Silero checkpoint is bundled (it lives in the gitignored root
//! `/models`), so the forward tests build a deterministic synthetic model whose
//! per-layer weight tensors have exactly the shapes `mlx.nn.Conv1d` / `nn.LSTM`
//! produce — the same all-fresh-init `Model(ModelConfig())` the reference's
//! shape tests use, which likewise assert only shapes / ranges (random init
//! gives no fixed numeric oracle).
//!
//! [t]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/tests/test_silero_vad.py

use std::collections::HashMap;

use super::*;
use crate::{
  array::Array,
  audio::vad::{load::VadModel, output::SpeechSegment},
  dtype::Dtype,
  error::Result,
};

// ─────────────────────────── synthetic weights ───────────────────────────

/// Build a zero weight tensor of the given shape (the synthetic-checkpoint
/// stand-in; a real Silero forward needs trained weights, but the reference
/// shape tests run on fresh-init weights and assert only shapes / ranges).
fn zeros(shape: &[i32]) -> Array {
  Array::zeros::<f32>(&shape).expect("zeros")
}

/// As [`zeros`] but in the given dtype — a real Silero `dtype == "float16"`
/// checkpoint ships f16 weights, so the synthetic fixture mirrors that (the
/// reference does not re-cast weights, so f16 parity requires f16 params).
fn zeros_dtype(shape: &[i32], dtype: Dtype) -> Array {
  Array::zeros::<f32>(&shape)
    .and_then(|a| a.astype(dtype))
    .expect("zeros_dtype")
}

/// Assemble the per-branch weight tensors for one sample-rate branch under the
/// `<branch>.` prefix, with the exact shapes `mlx.nn.Conv1d` (`(C_out, K,
/// C_in)`) / `nn.LSTM` (`Wx (4H, D)`, `Wh (4H, H)`, `bias (4H,)`) produce, in
/// the model `dtype` (mirroring a real f16 / f32 checkpoint).
fn insert_branch_weights(
  weights: &mut HashMap<String, Array>,
  cfg: &BranchConfig,
  branch: &str,
  dtype: Dtype,
) {
  let cutoff = cfg.cutoff();
  let put = |w: &mut HashMap<String, Array>, suffix: &str, shape: &[i32]| {
    w.insert(format!("{branch}.{suffix}"), zeros_dtype(shape, dtype));
  };
  // stft_conv: (2*cutoff, filter_length, 1), no bias.
  put(
    weights,
    "stft_conv.weight",
    &[2 * cutoff, cfg.filter_length(), 1],
  );
  // conv1: cutoff -> 128 (k=3); conv2: 128 -> 64; conv3: 64 -> 64; conv4: 64 -> 128.
  put(weights, "conv1.weight", &[128, 3, cutoff]);
  put(weights, "conv1.bias", &[128]);
  put(weights, "conv2.weight", &[64, 3, 128]);
  put(weights, "conv2.bias", &[64]);
  put(weights, "conv3.weight", &[64, 3, 64]);
  put(weights, "conv3.bias", &[64]);
  put(weights, "conv4.weight", &[128, 3, 64]);
  put(weights, "conv4.bias", &[128]);
  // lstm(128, 128): Wx (512, 128), Wh (512, 128), bias (512,).
  put(weights, "lstm.Wx", &[512, 128]);
  put(weights, "lstm.Wh", &[512, 128]);
  put(weights, "lstm.bias", &[512]);
  // final_conv: 128 -> 1 (k=1).
  put(weights, "final_conv.weight", &[1, 1, 128]);
  put(weights, "final_conv.bias", &[1]);
}

/// A deterministic synthetic [`SileroVadModel`] for the given config — both
/// branches built from correctly-shaped zero weights.
fn synthetic_model(config: ModelConfig) -> SileroVadModel {
  let mut weights = HashMap::new();
  let dtype = config.dtype();
  insert_branch_weights(&mut weights, config.branch_16k(), "vad_16k", dtype);
  insert_branch_weights(&mut weights, config.branch_8k(), "vad_8k", dtype);
  SileroVadModel::from_weights(config, weights).expect("synthetic model")
}

// ─────────────────────────── config parity ───────────────────────────

/// The default config matches the reference dataclass defaults
/// (`test_silero_vad.py::test_default_config`).
#[test]
fn default_config_matches_reference() {
  let cfg = ModelConfig::default();
  assert_eq!(cfg.branch_16k().sample_rate(), 16_000);
  assert_eq!(cfg.branch_16k().chunk_size(), 512);
  assert_eq!(cfg.branch_8k().sample_rate(), 8_000);
  assert_eq!(cfg.branch_8k().chunk_size(), 256);
  assert_eq!(cfg.dtype(), Dtype::F32);
  assert_eq!(cfg.threshold(), 0.5);
  assert_eq!(cfg.min_speech_duration_ms(), 250);
  assert_eq!(cfg.min_silence_duration_ms(), 100);
  assert_eq!(cfg.speech_pad_ms(), 30);
}

/// The full 8 kHz branch defaults match the reference `__post_init__`
/// fallback (`config.py:43-52`).
#[test]
fn default_8k_branch_matches_reference() {
  let b = BranchConfig::default_8k();
  assert_eq!(b.sample_rate(), 8_000);
  assert_eq!(b.filter_length(), 128);
  assert_eq!(b.hop_length(), 64);
  assert_eq!(b.pad(), 32);
  assert_eq!(b.cutoff(), 65);
  assert_eq!(b.context_size(), 32);
  assert_eq!(b.chunk_size(), 256);
}

/// `from_json` with a partial config overlays present keys onto the branch
/// defaults and resolves `dtype` (`test_silero_vad.py::test_config_from_dict`).
#[test]
fn config_from_json_overlays_and_resolves_dtype() {
  let cfg = ModelConfig::from_json(
    r#"{
      "dtype": "float16",
      "branch_16k": {"chunk_size": 512, "context_size": 64},
      "branch_8k": {"sample_rate": 8000, "filter_length": 128}
    }"#,
  )
  .expect("parse");
  assert_eq!(cfg.dtype(), Dtype::F16);
  // branch_16k present keys overlaid; absent keys keep the 16k default.
  assert_eq!(cfg.branch_16k().chunk_size(), 512);
  assert_eq!(cfg.branch_16k().context_size(), 64);
  assert_eq!(cfg.branch_16k().cutoff(), 129);
  // branch_8k PRESENT (partial) → the omitted keys fill from the 16 kHz
  // dataclass defaults (`BranchConfig.from_dict`, config.py:41-42), NOT the
  // 8 kHz overrides. So `filter_length` is the overlaid 128, but `chunk_size`
  // and `context_size` are the 16 kHz defaults (512 / 64) — a present partial
  // branch_8k does NOT inherit the 8 kHz 256 / 32.
  assert_eq!(cfg.branch_8k().filter_length(), 128);
  assert_eq!(cfg.branch_8k().chunk_size(), 512);
  assert_eq!(cfg.branch_8k().context_size(), 64);
}

/// A present-but-partial `branch_8k` fills omitted fields from the 16 kHz
/// dataclass defaults, while an ABSENT `branch_8k` uses the 8 kHz overrides —
/// the exact reference `__post_init__` semantics (`config.py:41-52`).
#[test]
fn partial_branch_8k_fills_from_16k_defaults_absent_uses_8k() {
  // Present (partial): omitted keys → 16 kHz defaults.
  let present = ModelConfig::from_json(r#"{"branch_8k": {"hop_length": 999}}"#).expect("parse");
  assert_eq!(present.branch_8k().hop_length(), 999); // overlaid
  assert_eq!(present.branch_8k().chunk_size(), 512); // 16k default (not 8k=256)
  assert_eq!(present.branch_8k().context_size(), 64); // 16k default (not 8k=32)
  assert_eq!(present.branch_8k().pad(), 64); // 16k default (not 8k=32)
  assert_eq!(present.branch_8k().cutoff(), 129); // 16k default (not 8k=65)

  // Absent: the full 8 kHz overrides.
  let absent = ModelConfig::from_json("{}").expect("parse");
  assert_eq!(absent.branch_8k(), &BranchConfig::default_8k());
  assert_eq!(absent.branch_8k().chunk_size(), 256);
  assert_eq!(absent.branch_8k().context_size(), 32);
}

/// A present branch as JSON `null` is treated as ABSENT (faithful to the
/// reference's `branch is None` path → the per-rate default); a present branch
/// that is neither an object nor null (array / string / number) is a malformed
/// config and is rejected with a typed error (fail closed), not silently
/// defaulted.
#[test]
fn branch_null_is_absent_and_malformed_branch_is_rejected() {
  // null → absent → the per-rate defaults.
  let null8 = ModelConfig::from_json(r#"{"branch_8k": null}"#).expect("null branch_8k parses");
  assert_eq!(null8.branch_8k(), &BranchConfig::default_8k());
  let null16 = ModelConfig::from_json(r#"{"branch_16k": null}"#).expect("null branch_16k parses");
  assert_eq!(null16.branch_16k(), &BranchConfig::default_16k());

  // present non-object (array / string / number) → typed error, not a default.
  assert!(ModelConfig::from_json(r#"{"branch_8k": []}"#).is_err());
  assert!(ModelConfig::from_json(r#"{"branch_8k": "x"}"#).is_err());
  assert!(ModelConfig::from_json(r#"{"branch_16k": 5}"#).is_err());
}

/// An empty `config.json` body resolves to the full dataclass defaults (every
/// key absent → every default kept).
#[test]
fn config_from_empty_json_is_defaults() {
  let cfg = ModelConfig::from_json("{}").expect("parse");
  assert_eq!(cfg, ModelConfig::default());
}

/// A non-`"float16"` dtype string resolves to f32 (the reference's
/// `else mx.float32`, `silero_vad.py:117`).
#[test]
fn config_dtype_non_float16_is_f32() {
  let cfg = ModelConfig::from_json(r#"{"dtype": "bfloat16"}"#).expect("parse");
  assert_eq!(cfg.dtype(), Dtype::F32);
}

/// A non-positive branch dim is rejected by the eager validator (the reference
/// performs no validation; this is the audio-config convention).
#[test]
fn config_rejects_non_positive_dim() {
  let err = ModelConfig::from_json(r#"{"branch_16k": {"cutoff": 0}}"#);
  assert!(err.is_err(), "cutoff=0 must be rejected");
}

// ─────────────────────────── forward shape parity ───────────────────────────

/// 16 kHz forward over a `(2, 576)` window returns a `(2, 1)` probability and a
/// `(2, 2, 128)` state, with the probability in `[0, 1]`
/// (`test_silero_vad.py::test_forward_shape_and_state_16k`).
#[test]
fn forward_shape_and_state_16k() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  // 576 = context_size(64) + chunk_size(512): the low-level window the
  // reference documents for 16 kHz (`silero_vad.py:108-111`).
  let x = Array::zeros::<f32>(&[2, 576])?;
  let (mut out, mut state) = model.forward(&x, None, 16_000)?;
  out.eval()?;
  state.eval()?;
  assert_eq!(out.shape(), vec![2, 1]);
  assert_eq!(state.shape(), vec![2, 2, 128]);
  let lo = out.min(false)?.astype(Dtype::F32)?.to_vec::<f32>()?[0];
  let hi = out.max(false)?.astype(Dtype::F32)?.to_vec::<f32>()?[0];
  assert!((0.0..=1.0).contains(&lo), "min {lo}");
  assert!((0.0..=1.0).contains(&hi), "max {hi}");
  Ok(())
}

/// 8 kHz forward over a `(1, 288)` window returns a `(1, 1)` probability and a
/// `(2, 1, 128)` state (`test_silero_vad.py::test_forward_shape_and_state_8k`).
#[test]
fn forward_shape_and_state_8k() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  // 288 = context_size(32) + chunk_size(256): the 8 kHz low-level window.
  let x = Array::zeros::<f32>(&[1, 288])?;
  let (mut out, mut state) = model.forward(&x, None, 8_000)?;
  out.eval()?;
  state.eval()?;
  assert_eq!(out.shape(), vec![1, 1]);
  assert_eq!(state.shape(), vec![2, 1, 128]);
  Ok(())
}

/// Feeding one 512-sample frame returns a `(1, 1)` probability and a state
/// whose carried context is `(1, 64)`
/// (`test_silero_vad.py::test_feed_updates_streaming_context`).
#[test]
fn feed_updates_streaming_context() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let chunk = Array::zeros::<f32>(&[512])?;
  let (mut out, state) = model.feed(&chunk, None, 16_000)?;
  out.eval()?;
  assert_eq!(out.shape(), vec![1, 1]);
  assert_eq!(state.context().shape(), vec![1, 64]);
  assert_eq!(state.sample_rate(), 16_000);
  Ok(())
}

/// `predict_proba` over 1024 samples at 16 kHz yields exactly 2 frames
/// (`test_silero_vad.py::test_predict_proba_chunks`).
#[test]
fn predict_proba_chunk_count() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[1024])?;
  let mut probs = model.predict_proba(&audio, 16_000)?;
  probs.eval()?;
  assert_eq!(probs.shape(), vec![2]);
  Ok(())
}

/// A long multi-chunk input drives the reference's periodic `async_eval`
/// graph-bounding path (`eval_every = 16`, `silero_vad.py:312-316`) and yields
/// the right frame count: 18 chunks at 16 kHz (chunk_size 512) is > 16 (so the
/// in-loop step-16 `async_eval` fires) and `18 % 16 != 0` (so the tail
/// `async_eval` fires), then concatenation yields 18 frames.
#[test]
fn predict_proba_long_input_periodic_eval() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[18 * 512])?;
  let mut probs = model.predict_proba(&audio, 16_000)?;
  probs.eval()?;
  assert_eq!(probs.shape(), vec![18]);
  Ok(())
}

/// An empty waveform yields an empty probability array (the reference's
/// early-return at `silero_vad.py:290-295`).
#[test]
fn predict_proba_empty_is_empty() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[0])?;
  let mut probs = model.predict_proba(&audio, 16_000)?;
  probs.eval()?;
  assert_eq!(probs.shape(), vec![0]);
  Ok(())
}

/// `predict_proba` rejects a rank it cannot handle (rank-0 scalar / rank-3+)
/// with a typed [`crate::error::Error::RankMismatch`] instead of panicking on
/// the empty-shape index — the reference accepts only mono / batched audio.
#[test]
fn predict_proba_rejects_bad_rank() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  // rank-0 scalar: would otherwise fall through the empty-input branch and
  // panic on `shape()[0]` of an empty shape vector.
  let scalar = Array::zeros::<f32>(&[])?;
  assert!(matches!(
    model.predict_proba(&scalar, 16_000),
    Err(crate::error::Error::RankMismatch(_))
  ));
  // rank-3 is also outside the (T,) / (B, T) contract.
  let rank3 = Array::zeros::<f32>(&[2, 2, 512])?;
  assert!(matches!(
    model.predict_proba(&rank3, 16_000),
    Err(crate::error::Error::RankMismatch(_))
  ));
  Ok(())
}

/// `generate` returns a [`crate::audio::vad::output::VadOutput`] with the
/// resolved sample rate and a 1-frame probability for a single-chunk input
/// (`test_silero_vad.py::test_generate_returns_output`).
#[test]
fn generate_returns_output() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[512])?;
  let out = model.generate(&audio, 16_000)?;
  assert_eq!(out.sample_rate, 16_000);
  assert_eq!(out.probabilities.shape(), vec![1]);
  Ok(())
}

/// A sample rate other than 8 kHz / 16 kHz is rejected (the reference's
/// `_branch` ValueError, `silero_vad.py:353-358`).
#[test]
fn unsupported_sample_rate_is_rejected() {
  let model = synthetic_model(ModelConfig::default());
  assert!(model.branch(44_100).is_err());
}

// ─────────────────────── segment-extraction oracle ───────────────────────

/// The hysteresis state machine reproduces the reference's exact expected
/// segment for its test vector (`test_silero_vad.py::test_probs_to_timestamps`):
/// `probs = [0.1, 0.8, 0.85, 0.1, 0.1]` → `[{"start": 512, "end": 1536}]`.
///
/// This is an INDEPENDENT oracle — the expected `(512, 1536)` is the
/// reference's own asserted value, not a value derived from this code.
#[test]
fn probs_to_timestamps_matches_reference_vector() {
  let probs = [0.1_f32, 0.8, 0.85, 0.1, 0.1];
  let segs = probs_to_timestamps(
    &probs,
    5 * 512, // audio_len
    16_000,  // sample_rate
    0.5,     // threshold
    30,      // min_speech_duration_ms
    30,      // min_silence_duration_ms
    0,       // speech_pad_ms
  );
  assert_eq!(segs, vec![SpeechSegment::new(512, 1536)]);
}

/// Two raw speeches whose pads overlap COALESCE into one segment — faithful to
/// mlx-audio's pad-and-merge (`silero_vad.py:410-417`: `start <= padded[-1].end`
/// → extend the previous end). This is INTENTIONALLY mlx-audio's behavior, NOT
/// the upstream PyTorch snakers4 Silero `get_speech_timestamps`, which splits
/// short inter-segment silence between neighbors; the directive is 1:1 mlx-audio.
#[test]
fn probs_to_timestamps_coalesces_padded_overlap_like_mlx_audio() {
  // Two speeches {512,1536} and {2560,3584} separated by a closing silence.
  let probs = [0.1_f32, 0.8, 0.85, 0.1, 0.1, 0.8, 0.85, 0.1, 0.1];
  let segs = probs_to_timestamps(
    &probs,
    9 * 512, // audio_len
    16_000,  // sample_rate
    0.5,     // threshold
    30,      // min_speech_duration_ms
    30,      // min_silence_duration_ms
    100,     // speech_pad_ms → pad 1600 samples each side, so the pads overlap
  );
  // Pads: {0,3136} and {960,4608}; 960 <= 3136 ⇒ coalesce into a single segment.
  assert_eq!(segs, vec![SpeechSegment::new(0, 4608)]);
}

/// All-silence probabilities produce no segments.
#[test]
fn probs_to_timestamps_all_silence_is_empty() {
  let probs = [0.0_f32, 0.1, 0.05, 0.0];
  let segs = probs_to_timestamps(&probs, 4 * 512, 16_000, 0.5, 30, 30, 0);
  assert!(segs.is_empty());
}

/// A segment still open at the end of the stream is closed at
/// `min(audio_len, n_frames * chunk_size)` and kept if long enough (the
/// reference's trailing-segment branch, `silero_vad.py:405-408`).
#[test]
fn probs_to_timestamps_closes_trailing_segment() {
  // Four speech frames, never dropping below neg_threshold → still triggered at
  // the end. n_frames*chunk = 4*512 = 2048; audio_len 2048.
  let probs = [0.9_f32, 0.9, 0.9, 0.9];
  let segs = probs_to_timestamps(&probs, 2048, 16_000, 0.5, 30, 30, 0);
  // current_start = 0 (first frame triggers); end = min(2048, 2048) = 2048;
  // 2048 - 0 = 2048 >= min_speech(480) → kept.
  assert_eq!(segs, vec![SpeechSegment::new(0, 2048)]);
}

/// `speech_pad` widens a segment on each side and clamps to `[0, audio_len]`.
#[test]
fn probs_to_timestamps_applies_speech_pad() {
  // Reuse the reference vector but with 10 ms of speech pad: 16000*10/1000 = 160
  // samples each side. Segment (512, 1536) → (352, 1696), both in range.
  let probs = [0.1_f32, 0.8, 0.85, 0.1, 0.1];
  let segs = probs_to_timestamps(&probs, 5 * 512, 16_000, 0.5, 30, 30, 10);
  assert_eq!(segs, vec![SpeechSegment::new(352, 1696)]);
}

/// The 8 kHz path uses a 256-sample chunk for the index→sample mapping
/// (`silero_vad.py:372`), so the same frame index maps to half the sample
/// offset of the 16 kHz path.
#[test]
fn probs_to_timestamps_8k_uses_256_chunk() {
  let probs = [0.1_f32, 0.8, 0.85, 0.1, 0.1];
  // min_speech/min_silence at 8 kHz: 8000*30/1000 = 240 samples each.
  let segs = probs_to_timestamps(&probs, 5 * 256, 8_000, 0.5, 30, 30, 0);
  // Same hysteresis as the 16 kHz vector but chunk = 256: start = 1*256 = 256,
  // temp_end = 3*256 = 768; 768-256 = 512 >= 240 → kept → (256, 768).
  assert_eq!(segs, vec![SpeechSegment::new(256, 768)]);
}

// ─────────────────────── sanitize + streaming ───────────────────────

/// `sanitize` drops the reference's non-model `val_*` keys
/// (`silero_vad.py:429-431`) and keeps everything else.
#[test]
fn sanitize_drops_val_prefixed_keys() {
  let mut weights = HashMap::new();
  weights.insert("vad_16k.conv1.weight".to_string(), zeros(&[1]));
  weights.insert("val_loss".to_string(), zeros(&[1]));
  weights.insert("val_acc.running".to_string(), zeros(&[1]));
  let kept = sanitize(weights);
  assert!(kept.contains_key("vad_16k.conv1.weight"));
  assert!(!kept.keys().any(|k| k.starts_with("val_")));
  assert_eq!(kept.len(), 1);
}

/// Streaming `feed` over consecutive frames threads the LSTM state and the
/// carried context: each step returns a `(1, 1)` probability, the context stays
/// `(1, context_size)`, and the state carries an LSTM recurrent state after the
/// first frame.
#[test]
fn feed_threads_state_across_frames() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let mut state = None;
  for _ in 0..3 {
    let chunk = Array::zeros::<f32>(&[512])?;
    let (mut out, next) = model.feed(&chunk, state, 16_000)?;
    out.eval()?;
    assert_eq!(out.shape(), vec![1, 1]);
    assert_eq!(next.context().shape(), vec![1, 64]);
    assert!(next.state().is_some(), "feed must carry an LSTM state");
    state = Some(next);
  }
  Ok(())
}

/// Feeding a wrong-width chunk is rejected (the reference's
/// `Expected … samples` ValueError, `silero_vad.py:176-180`).
#[test]
fn feed_rejects_wrong_chunk_width() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let chunk = Array::zeros::<f32>(&[400])?; // not 512
  assert!(model.feed(&chunk, None, 16_000).is_err());
  Ok(())
}

// ─────────────────────── dtype preservation ───────────────────────

/// An f16 model keeps half precision through the forward — the probability and
/// state come back as f16, not promoted to f32 (the dtype-preservation
/// discipline; the reference's `self.dtype` threading, `silero_vad.py:117-148`).
#[test]
fn forward_preserves_f16_dtype() -> Result<()> {
  let cfg = ModelConfig::from_json(r#"{"dtype": "float16"}"#)?;
  assert_eq!(cfg.dtype(), Dtype::F16);
  let model = synthetic_model(cfg);
  // Feed an f32 input: the model casts it to its f16 dtype (`__call__`).
  let x = Array::zeros::<f32>(&[1, 576])?;
  let (out, state) = model.forward(&x, None, 16_000)?;
  assert_eq!(out.dtype()?, Dtype::F16);
  assert_eq!(state.dtype()?, Dtype::F16);
  Ok(())
}

/// `predict_proba` over a multi-chunk waveform in f16 returns f16
/// probabilities with the right frame count (the per-chunk recurrence keeps the
/// model dtype).
#[test]
fn predict_proba_preserves_f16_dtype() -> Result<()> {
  let cfg = ModelConfig::from_json(r#"{"dtype": "float16"}"#)?;
  let model = synthetic_model(cfg);
  let audio = Array::zeros::<f32>(&[1536])?; // 3 chunks
  let mut probs = model.predict_proba(&audio, 16_000)?;
  assert_eq!(probs.dtype()?, Dtype::F16);
  probs.eval()?;
  assert_eq!(probs.shape(), vec![3]);
  Ok(())
}

// ─────────────────── batched / 8 kHz / boundary coverage ───────────────────

/// A batched `(B, T)` input returns per-row frame probabilities `(B, n_frames)`
/// (the reference's batched `_predict_proba_array` path).
#[test]
fn predict_proba_batched_returns_per_row_frames() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[2, 1024])?; // (B=2, T=1024)
  let mut probs = model.predict_proba(&audio, 16_000)?;
  probs.eval()?;
  assert_eq!(probs.shape(), vec![2, 2]); // (B, n_frames) with 1024/512 = 2
  Ok(())
}

/// The 8 kHz path uses a 256-sample chunk: 1024 samples → 4 frames.
#[test]
fn predict_proba_8k_chunk_count() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[1024])?;
  let mut probs = model.predict_proba(&audio, 8_000)?;
  probs.eval()?;
  assert_eq!(probs.shape(), vec![4]); // 1024 / 256
  Ok(())
}

/// Exactly `EVAL_EVERY` (16) chunks: the 1-based `async_eval` cadence fires the
/// final in-loop eval on the last chunk (the off-by-one the fix guards — a
/// 0-based `step` would never reach 16 and the tail also skips the
/// exact-multiple case). The frame count is invariant to the eval cadence, so
/// this exercises the exact-multiple boundary without asserting eval timing.
#[test]
fn predict_proba_exact_eval_every_multiple() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[16 * 512])?;
  let mut probs = model.predict_proba(&audio, 16_000)?;
  probs.eval()?;
  assert_eq!(probs.shape(), vec![16]);
  Ok(())
}

// ─────────────────────── loader robustness (#2, #5/#8) ──────────────────────

/// A checkpoint carrying a `<prefix>.scales` sibling (a quantized layout) is
/// rejected — Silero is dense-only, so loading it dense would misinterpret the
/// packed weights (the `has_relevant_scales` fail-closed gate).
#[test]
fn from_weights_rejects_quantized_checkpoint() {
  let config = ModelConfig::default();
  let mut weights = HashMap::new();
  insert_branch_weights(&mut weights, config.branch_16k(), "vad_16k", config.dtype());
  insert_branch_weights(&mut weights, config.branch_8k(), "vad_8k", config.dtype());
  // A `.scales` sibling for an existing `.weight` ⇒ looks quantized.
  weights.insert("vad_16k.conv1.scales".to_string(), zeros(&[128, 1]));
  assert!(matches!(
    SileroVadModel::from_weights(config, weights),
    Err(crate::error::Error::OutOfRange(_))
  ));
}

/// An LSTM `Wx` whose leading dim is not a positive multiple of 4 (the `4*H`
/// gate stack) is rejected by `build_branch` — the hidden size is inferred from
/// `Wx.shape[0] / 4`, so a malformed shape fails closed instead of mis-splitting
/// the gates.
#[test]
fn from_weights_rejects_malformed_lstm_shape() {
  let config = ModelConfig::default();
  let mut weights = HashMap::new();
  insert_branch_weights(&mut weights, config.branch_16k(), "vad_16k", config.dtype());
  insert_branch_weights(&mut weights, config.branch_8k(), "vad_8k", config.dtype());
  // 513 is not a multiple of 4.
  weights.insert(
    "vad_16k.lstm.Wx".to_string(),
    zeros_dtype(&[513, 128], config.dtype()),
  );
  assert!(SileroVadModel::from_weights(config, weights).is_err());
}

// ───────────────────── prepare_audio (#3) + seconds (#4) ─────────────────────

/// `prepare_audio` downmixes a `(T, C)` stereo/multichannel clip (`C <= 8 < T`)
/// to mono `(T,)`, but leaves a `(B, T)` batch (trailing `T > 8`) untouched —
/// the reference's `ndim == 2 and shape[-1] <= 8 < shape[0]` discriminator.
#[test]
fn prepare_audio_downmixes_stereo_and_keeps_batch() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let stereo = Array::zeros::<f32>(&[1000, 2])?;
  let (prepared, sr) = model.prepare_audio(&stereo, 16_000)?;
  assert_eq!(prepared.shape(), vec![1000]);
  assert_eq!(sr, 16_000);

  let batched = Array::zeros::<f32>(&[2, 1000])?;
  let (prepared_b, _) = model.prepare_audio(&batched, 16_000)?;
  assert_eq!(prepared_b.shape(), vec![2, 1000]);
  Ok(())
}

/// `prepare_audio` resamples a non-8/16 kHz input to 16 kHz (the reference's
/// `target_sr = sr if sr in (8000, 16000) else 16000` net), resolving the rate
/// and shrinking a 32 kHz clip to roughly half the samples.
#[test]
fn prepare_audio_resamples_unsupported_rate_to_16k() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[1024])?;
  let (prepared, sr) = model.prepare_audio(&audio, 32_000)?;
  assert_eq!(sr, 16_000);
  let n = prepared.shape()[0];
  assert!(
    (400..=600).contains(&n),
    "32k→16k resample of 1024 → ~512, got {n}"
  );
  Ok(())
}

/// [`SpeechSegment::start_seconds`] / [`SpeechSegment::end_seconds`] convert
/// sample indices to seconds at the inference rate (the reference's
/// `return_seconds=True` view).
#[test]
fn speech_segment_seconds_accessors() {
  let seg = SpeechSegment::new(16_000, 32_000);
  assert_eq!(seg.start_seconds(16_000), 1.0);
  assert_eq!(seg.end_seconds(16_000), 2.0);
  assert_eq!(seg.start_seconds(8_000), 2.0);
}

// ───────────── get_speech_timestamps + predict / reset_state (#6/#7/#10) ─────

/// `get_speech_timestamps` runs the full prepare → predict → segment pipeline
/// and honors a per-call threshold override (the reference's keyword args). A
/// zero-weight model's sigmoid head is 0.5 everywhere: threshold 0.5 (default)
/// ⇒ all-speech (one segment), threshold 0.6 ⇒ silence (empty).
#[test]
fn get_speech_timestamps_default_and_override() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[16 * 512])?;

  let segs = model.get_speech_timestamps(&audio, 16_000, SpeechTimestampOptions::default())?;
  assert!(!segs.is_empty(), "all-0.5 probs at threshold 0.5 → speech");

  let opts = SpeechTimestampOptions {
    threshold: Some(0.6),
    ..Default::default()
  };
  let none = model.get_speech_timestamps(&audio, 16_000, opts)?;
  assert!(none.is_empty(), "all-0.5 probs at threshold 0.6 → silence");
  Ok(())
}

/// `predict` (the public prepare + probabilities entry) and `reset_state` (the
/// `initial_state` alias) are wired through correctly.
#[test]
fn predict_and_reset_state_smoke() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[1024])?;
  let mut probs = model.predict(&audio, 16_000)?;
  probs.eval()?;
  assert_eq!(probs.shape(), vec![2]); // 1024 / 512

  let state = model.reset_state(1, 16_000)?;
  assert_eq!(state.context().shape(), vec![1, 64]);
  assert!(state.state().is_none());
  Ok(())
}

// ─────────────────── per-call override + gate hardening ───────────────────

/// A negative per-call duration/padding override is rejected with a typed error
/// (the override path validates like `ModelConfig` does at load), so it cannot
/// produce a `start > end` segment from the padding step.
#[test]
fn get_speech_timestamps_rejects_negative_override() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[4 * 512])?;
  let opts = SpeechTimestampOptions {
    speech_pad_ms: Some(-100),
    ..Default::default()
  };
  assert!(matches!(
    model.get_speech_timestamps(&audio, 16_000, opts),
    Err(crate::error::Error::OutOfRange(_))
  ));
  Ok(())
}

/// A quantized LSTM checkpoint (a `lstm.Wx.scales` sibling — NOT a `*.weight`
/// key) is still rejected: the dense-only gate detects ANY `.scales` tensor.
#[test]
fn from_weights_rejects_quantized_lstm_scales() {
  let config = ModelConfig::default();
  let mut weights = HashMap::new();
  insert_branch_weights(&mut weights, config.branch_16k(), "vad_16k", config.dtype());
  insert_branch_weights(&mut weights, config.branch_8k(), "vad_8k", config.dtype());
  weights.insert("vad_16k.lstm.Wx.scales".to_string(), zeros(&[512, 1]));
  assert!(matches!(
    SileroVadModel::from_weights(config, weights),
    Err(crate::error::Error::OutOfRange(_))
  ));
}

/// A rank-2 zero-row batch `(0, T)` at an unsupported sample rate returns an
/// empty prepared batch at the resolved 16 kHz rate instead of crashing in the
/// resample row-loop's empty concatenate.
#[test]
fn prepare_audio_zero_row_batch_at_unsupported_rate() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let empty_batch = Array::zeros::<f32>(&[0, 1000])?;
  let (prepared, sr) = model.prepare_audio(&empty_batch, 32_000)?;
  assert_eq!(sr, 16_000);
  assert_eq!(prepared.shape()[0], 0);
  Ok(())
}

/// Empty batches of EITHER layout — zero rows `(0, T)` or zero width `(B, 0)`
/// — flow through the PUBLIC timestamp paths end-to-end: `generate` and
/// `get_speech_timestamps` return no timestamps instead of failing on the
/// batched `probs[0]` row-take (a zero-row rank-2 probability array has no
/// row 0) or on a zero-length downmix reduction. Covers supported and
/// unsupported rates (the resample path).
#[test]
fn empty_batches_yield_empty_timestamps_end_to_end() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  for (r, c) in [(0usize, 1000usize), (2, 0)] {
    for rate in [16_000u32, 32_000] {
      let audio = Array::zeros::<f32>(&[r as i32, c as i32])?;
      let out = model.generate(&audio, rate)?;
      assert!(
        out.timestamps.is_empty(),
        "generate on ({r}, {c}) at {rate} Hz must yield no timestamps"
      );
      let segs = model.get_speech_timestamps(&audio, rate, SpeechTimestampOptions::default())?;
      assert!(
        segs.is_empty(),
        "get_speech_timestamps on ({r}, {c}) at {rate} Hz must yield no segments"
      );
    }
  }
  Ok(())
}

/// A zero-ROW batch with a nonzero width at an UNSUPPORTED rate resolves its
/// time axis to the resampled width, so the probability frame count reflects
/// the resolved 16 kHz timeline (1000 samples at 32 kHz resample to ~500 at
/// 16 kHz → zero frames of 512 would differ from the un-resampled 1000's one
/// frame — the shape contract must track the resolved rate, not the original).
#[test]
fn zero_row_unsupported_rate_resolves_resampled_width() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let audio = Array::zeros::<f32>(&[0, 1000])?;
  let (prepared, sr) = model.prepare_audio(&audio, 32_000)?;
  assert_eq!(sr, 16_000);
  let w = prepared.shape()[1];
  assert!(
    (400..=600).contains(&w),
    "the empty batch's width must be the RESAMPLED length (~500), got {w}"
  );
  // End-to-end: the probability frame count follows the resolved timeline.
  let out = model.generate(&audio, 32_000)?;
  let frames = *out.probabilities.shape().last().unwrap_or(&usize::MAX);
  assert_eq!(
    frames,
    w.div_ceil(512),
    "probability frames must chunk the resampled width"
  );
  assert!(out.timestamps.is_empty());
  Ok(())
}

/// The empty-batch resampled width is pure shape ARITHMETIC: a data-less
/// zero-row batch with a huge width must resolve instantly (no
/// width-proportional dummy allocation) to the arithmetically-resampled
/// width.
#[test]
fn zero_row_huge_width_is_shape_arithmetic_only() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let huge = 1_000_000_000i32; // 1e9 samples — 4 GB if materialized as f32
  let audio = Array::zeros::<f32>(&[0, huge])?;
  let (prepared, sr) = model.prepare_audio(&audio, 32_000)?;
  assert_eq!(sr, 16_000);
  assert_eq!(
    prepared.shape(),
    vec![0, (huge as usize) / 2],
    "width must be the arithmetic in*to/from resample length"
  );
  Ok(())
}

/// A zero-row batch short-circuits `predict_proba` to the reference's
/// `(0, ceil(total / chunk))` frame contract WITHOUT running the chunk loop
/// (a `(0, huge)` input would otherwise iterate millions of empty windows).
#[test]
fn zero_row_predict_proba_short_circuits_frame_shape() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let huge = 1_000_000_000i32;
  let audio = Array::zeros::<f32>(&[0, huge])?;
  let probs = model.predict_proba(&audio, 16_000)?;
  assert_eq!(
    probs.shape(),
    vec![0, (huge as usize).div_ceil(512)],
    "zero-row probabilities must carry the reference frame count"
  );
  Ok(())
}

/// A zero sample rate is a typed error from `prepare_audio` — never an
/// integer divide-by-zero — for empty AND non-empty inputs alike.
#[test]
fn zero_sample_rate_is_typed_error() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  for shape in [[0i32, 1000], [2, 1000]] {
    let audio = Array::zeros::<f32>(&shape)?;
    assert!(
      matches!(
        model.prepare_audio(&audio, 0),
        Err(crate::error::Error::OutOfRange(_))
      ),
      "sample_rate 0 on {shape:?} must be a typed error"
    );
  }
  Ok(())
}

/// The stereo-downmix predicate is the EXACT reference heuristic
/// (`shape[-1] <= 8 < shape[0]`, silero_vad.py:338): a small `(3, 2)` batch —
/// whose row count does NOT exceed 8 — stays a batch, it is not collapsed to
/// a mono `(3,)` signal even though its column count is small.
#[test]
fn prepare_audio_small_batch_is_not_downmixed() -> Result<()> {
  let model = synthetic_model(ModelConfig::default());
  let small = Array::zeros::<f32>(&[3, 2])?;
  let (prepared, _) = model.prepare_audio(&small, 16_000)?;
  assert_eq!(
    prepared.shape(),
    vec![3, 2],
    "rows <= 8 must not trigger the stereo downmix (reference: 8 < rows)"
  );
  Ok(())
}
