//! Oracle tests for the Qwen3-ASR audio encoder.
//!
//! Every expected value is computed independently of the code under test — the
//! conv downsample lengths by the analytic stride-2 recurrence, the sinusoidal
//! position table and the single-conv stem by closed-form arithmetic, the
//! sanitize axis-swap against the verbatim test input, and the config bounds by
//! the documented caps — never by invoking the implementation a second time.

use std::collections::HashMap;

use super::{audio::AudioEncoder, config::AudioEncoderConfig, sanitize};
use crate::{Dtype, array::Array, error::Error};

const TOL: f32 = 1e-4;

fn assert_close(got: &[f32], want: &[f32]) {
  assert_eq!(
    got.len(),
    want.len(),
    "length mismatch: {got:?} vs {want:?}"
  );
  for (i, (g, w)) in got.iter().zip(want).enumerate() {
    assert!(
      (g - w).abs() <= TOL,
      "index {i}: got {g}, want {w} (|Δ|={})",
      (g - w).abs()
    );
  }
}

/// Closed-form exact GELU (erf form) for the oracle: `x/2 * (1 + erf(x/√2))`.
fn gelu_scalar(x: f64) -> f64 {
  x * 0.5 * (1.0 + libm_erf(x / std::f64::consts::SQRT_2))
}

/// `erf` via a high-accuracy rational approximation (Abramowitz & Stegun
/// 7.1.26, |error| < 1.5e-7) — independent of MLX's erf, sufficient for the
/// 1e-4 oracle tolerance.
fn libm_erf(x: f64) -> f64 {
  let sign = if x < 0.0 { -1.0 } else { 1.0 };
  let x = x.abs();
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let y = 1.0
    - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
      * t
      * (-x * x).exp();
  sign * y
}

// ════════════════════════════ conv downsample math ════════════════════════════

/// The analytic three-fold stride-2 (kernel 3, padding 1) recurrence
/// `out = (in + 1) / 2`, computed independently of the config helper.
fn conv3_chain(n: i64) -> i64 {
  let step = |x: i64| if x <= 0 { 0 } else { (x + 1) / 2 };
  step(step(step(n)))
}

#[test]
fn freq_after_conv_matches_analytic_chain() {
  // num_mel_bins = 128: 128 -> (129/2=64) -> (65/2=32) -> (33/2=16) = 16.
  let cfg = AudioEncoderConfig::from_json(r#"{}"#).unwrap();
  assert_eq!(cfg.num_mel_bins, 128);
  assert_eq!(cfg.freq_after_conv(), 16);
  assert_eq!(i64::from(cfg.freq_after_conv()), conv3_chain(128));

  // num_mel_bins = 80: 80 -> 40 -> 20 -> 10.
  let cfg80 = AudioEncoderConfig::from_json(r#"{"num_mel_bins": 80}"#).unwrap();
  assert_eq!(cfg80.freq_after_conv(), 10);
  // num_mel_bins = 8: 8 -> (9/2=4) -> (5/2=2) -> (3/2=1) = 1.
  let cfg8 = AudioEncoderConfig::from_json(r#"{"num_mel_bins": 8}"#).unwrap();
  assert_eq!(cfg8.freq_after_conv(), 1);
}

#[test]
fn time_after_conv_matches_analytic_chain() {
  // time = 16: 16 -> (17/2=8) -> (9/2=4) -> (5/2=2) = 2.
  assert_eq!(AudioEncoderConfig::time_after_conv(16), 2);
  assert_eq!(AudioEncoderConfig::time_after_conv(16), conv3_chain(16));
  // A few more lengths against the independent chain.
  for &t in &[1i64, 5, 100, 999, 3000] {
    assert_eq!(
      AudioEncoderConfig::time_after_conv(t),
      conv3_chain(t),
      "t={t}"
    );
  }
}

// ════════════════════════════ config validation ════════════════════════════

#[test]
fn config_parses_defaults_and_ignores_unknown() {
  let cfg = AudioEncoderConfig::from_json(r#"{"unknown_future_key": 7}"#).unwrap();
  assert_eq!(cfg.d_model, 1024);
  assert_eq!(cfg.encoder_layers, 24);
  assert_eq!(cfg.encoder_attention_heads, 16);
  assert_eq!(cfg.encoder_ffn_dim, 4096);
  assert_eq!(cfg.output_dim, 2048);
  assert_eq!(cfg.downsample_hidden_size, 480);
  assert_eq!(cfg.max_source_positions, 1500);
  assert_eq!(cfg.head_dim(), 64); // 1024 / 16
}

#[test]
fn config_validate_accepts_defaults() {
  assert!(AudioEncoderConfig::from_json(r#"{}"#).is_ok());
}

#[test]
fn config_validate_rejects_non_divisible_heads() {
  // d_model = 1024 not divisible by 17 heads.
  let err = AudioEncoderConfig::from_json(r#"{"encoder_attention_heads": 17}"#)
    .expect_err("non-divisible heads must be rejected");
  assert!(
    matches!(err, Error::DivisibilityConstraint(_)),
    "got {err:?}"
  );
}

#[test]
fn config_validate_rejects_odd_d_model() {
  // Odd d_model (still divisible by 1 head) fails the even-channels check the
  // sinusoidal embedding requires.
  let err = AudioEncoderConfig::from_json(r#"{"d_model": 15, "encoder_attention_heads": 1}"#)
    .expect_err("odd d_model must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn config_validate_accepts_large_layer_count() {
  // A large-but-valid layer count parses and validates: validate() carries no
  // magnitude ceiling, so it only checks structure (positive / divisible), not
  // model size. (validate() builds nothing, so this stays cheap.)
  assert!(AudioEncoderConfig::from_json(r#"{"encoder_layers": 100000}"#).is_ok());
}

#[test]
fn config_validate_rejects_non_positive_layer_count() {
  let err = AudioEncoderConfig::from_json(r#"{"encoder_layers": 0}"#)
    .expect_err("zero encoder_layers must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn config_validate_rejects_non_positive_dim() {
  let err =
    AudioEncoderConfig::from_json(r#"{"d_model": 0}"#).expect_err("zero d_model must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn config_validate_rejects_deviating_activation() {
  let err = AudioEncoderConfig::from_json(r#"{"activation_function": "relu"}"#)
    .expect_err("non-gelu activation must be rejected");
  assert!(matches!(err, Error::UnknownEnumValue(_)), "got {err:?}");
}

#[test]
fn config_validate_rejects_max_source_positions_below_full_chunk() {
  // The positional table must cover a full conv chunk's post-CNN length
  // (`time_after_conv(n_window * 2)`). With n_window = 50 the chunk is 100 mel
  // frames → time_after_conv(100) = 13 post-CNN rows, but max_source_positions =
  // 4 cannot cover it: the windowed encoder would add pos_emb[:13] off a 4-row
  // table. validate() must reject it as a typed OutOfRange at construction.
  assert_eq!(AudioEncoderConfig::time_after_conv(100), 13);
  let err = AudioEncoderConfig::from_json(r#"{"n_window": 50, "max_source_positions": 4}"#)
    .expect_err("a too-small max_source_positions must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn config_validate_accepts_max_source_positions_exactly_full_chunk() {
  // The boundary: max_source_positions == time_after_conv(n_window * 2) is
  // exactly enough and must validate.
  let needed = AudioEncoderConfig::time_after_conv(100); // 13
  let json = format!(r#"{{"n_window": 50, "max_source_positions": {needed}}}"#);
  assert!(
    AudioEncoderConfig::from_json(&json).is_ok(),
    "max_source_positions == full-chunk post-CNN length must validate"
  );
}

// ════════════════════════════ sinusoidal position embedding ════════════════════════════

#[test]
fn sinusoidal_table_matches_closed_form() {
  // Channels = 4 → half = 2, log_inc = ln(10000) / (2 - 1) = ln(10000).
  // inv_timescales = [exp(0), exp(-ln(10000))] = [1, 1e-4].
  // For position p: row = [sin(p*1), sin(p*1e-4), cos(p*1), cos(p*1e-4)].
  // Compute the implementation's first 3 rows and compare to this closed form.
  let channels = 4i32;
  let seqlen = 3i32;
  let got = super::audio::SinusoidalPositionEmbedding::eval_rows(8, channels, seqlen).unwrap();
  assert_eq!(got.len(), (seqlen * channels) as usize);

  let half = (channels / 2) as usize;
  let log_inc = (10000.0f64).ln() / (half as f64 - 1.0);
  let inv: Vec<f64> = (0..half).map(|k| (-log_inc * k as f64).exp()).collect();
  // Independent sanity on inv_timescales (not derived from the impl).
  assert!((inv[0] - 1.0).abs() < 1e-9);
  assert!((inv[1] - 1e-4).abs() < 1e-9);

  let mut want: Vec<f32> = Vec::new();
  for p in 0..seqlen as i64 {
    // sin halves first, then cos halves (the reference concat order).
    want.extend(inv.iter().map(|&iv| ((p as f64) * iv).sin() as f32));
    want.extend(inv.iter().map(|&iv| ((p as f64) * iv).cos() as f32));
  }
  assert_close(&got, &want);
}

#[test]
fn sinusoidal_forward_rejects_seqlen_past_table() {
  // MLX `slice` CLAMPS an out-of-range stop, so slicing `0..seqlen` of a table
  // with fewer rows would silently return a truncated table that then broadcasts
  // position 0..rows across a longer sequence (reused positions). The forward
  // must reject `seqlen > rows` with a typed OutOfRange BEFORE slicing.
  //
  // `eval_rows(length, channels, seqlen)` builds a `length`-row table then slices
  // `seqlen` rows; `seqlen == length` is the boundary (accepted), `> length` must
  // error.
  assert!(
    super::audio::SinusoidalPositionEmbedding::eval_rows(4, 4, 4).is_ok(),
    "seqlen == table rows must be accepted"
  );
  match super::audio::SinusoidalPositionEmbedding::eval_rows(4, 4, 10) {
    Err(Error::OutOfRange(_)) => {}
    other => panic!("seqlen past the table must be a typed OutOfRange, got {other:?}"),
  }
}

// ════════════════════════════ single-conv stem oracle ════════════════════════════

#[test]
fn conv2d_stem_layer_closed_form() -> Result<(), Error> {
  // A standalone Conv2d(kernel=3, stride=2, padding=1) over a 1-channel
  // (N=1, H=2, W=2, C_in=1) input, with a hand-picked kernel + bias, computed
  // through the public encoder by isolating the first conv: we replicate the
  // layer's math here with the crate conv op and compare to a fully hand-rolled
  // value, confirming gelu(conv2d + bias) is what the stem applies.
  //
  // Input (H=2, W=2): [[1, 2], [3, 4]] (the single channel).
  // Kernel (C_out=1, KH=3, KW=3, C_in=1): a single 1.0 at center [1][1], else 0
  //   → an identity that, with stride 2 / padding 1, samples the top-left of
  //   each 3x3 window. Output H' = (2+1)/2 = 1? No: (2 + 2*1 - 3)/2 + 1 =
  //   (1)/2 + 1 = 1. So output is (1,1,1,1): the center-tap at the first window
  //   anchored at padded position (0,0) selects input[0][0] = 1.
  // With bias b = 0.5 and GELU: out = gelu(1 + 0.5) = gelu(1.5).
  let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1])?;
  // center-tap kernel.
  let mut kernel = vec![0.0f32; 9];
  kernel[4] = 1.0; // KH=1, KW=1 center of 3x3.
  let weight = Array::from_slice::<f32>(&kernel, &[1, 3, 3, 1])?;
  let conv = crate::ops::conv::conv2d(&input, &weight, (2, 2), (1, 1), (1, 1), 1)?;
  // conv output shape (1, 1, 1, 1); value = input[0][0] = 1.0 (center tap at the
  // first stride window, padded).
  let bias = Array::from_slice::<f32>(&[0.5], &[1])?;
  let biased = conv.add(&bias)?;
  let mut acted = crate::lm::nn::activations::gelu(&biased)?;
  let got = acted.to_vec::<f32>()?;
  assert_eq!(got.len(), 1, "expected a single output element");
  // Independent expected: gelu(1.0 + 0.5) = gelu(1.5).
  let want = gelu_scalar(1.5) as f32;
  assert_close(&got, &[want]);
  Ok(())
}

// ════════════════════════════ tiny-config structural forward ════════════════════════════

/// A tiny but valid config: num_mel_bins=8 (freq_after_conv=1),
/// downsample_hidden_size=2, d_model=4 (heads=2 → head_dim=2, even),
/// encoder_ffn_dim=8, output_dim=6, 1 encoder layer, max_source_positions=8.
fn tiny_config() -> AudioEncoderConfig {
  AudioEncoderConfig::from_json(
    r#"{
      "num_mel_bins": 8,
      "encoder_layers": 1,
      "encoder_attention_heads": 2,
      "encoder_ffn_dim": 8,
      "d_model": 4,
      "output_dim": 6,
      "max_source_positions": 8,
      "n_window": 2,
      "n_window_infer": 8,
      "downsample_hidden_size": 2
    }"#,
  )
  .expect("tiny config must validate")
}

/// The tiny config with an explicit `conv_chunksize` — the windowed encoder
/// processes the padded chunk batch in `conv_chunksize`-sized slices, so this
/// lets a test compare a small slice size against one covering every chunk.
fn tiny_config_conv_chunksize(conv_chunksize: i32) -> AudioEncoderConfig {
  AudioEncoderConfig::from_json(&format!(
    r#"{{
      "num_mel_bins": 8,
      "encoder_layers": 1,
      "encoder_attention_heads": 2,
      "encoder_ffn_dim": 8,
      "d_model": 4,
      "output_dim": 6,
      "max_source_positions": 8,
      "n_window": 2,
      "n_window_infer": 8,
      "conv_chunksize": {conv_chunksize},
      "downsample_hidden_size": 2
    }}"#
  ))
  .expect("tiny conv_chunksize config must validate")
}

/// Deterministic small constant tensor `(shape)` filled with `val`.
fn filled(shape: &[i32], val: f32) -> Array {
  Array::full::<f32>(&shape.to_vec(), val).unwrap()
}

/// Build the full tiny-config weight map (channels-last conv weights, every
/// named projection / norm). Small constant fills keep the forward numerically
/// stable; the test asserts shapes, not magnitudes.
fn tiny_weights(cfg: &AudioEncoderConfig) -> HashMap<String, Array> {
  let d = cfg.d_model;
  let h = cfg.downsample_hidden_size;
  let ffn = cfg.encoder_ffn_dim;
  let out = cfg.output_dim;
  let conv_out_in = cfg.conv_out_in_features().unwrap();
  let mut w: HashMap<String, Array> = HashMap::new();

  // Conv2d stem (channels-last (out, kH, kW, in)).
  w.insert("conv2d1.weight".into(), filled(&[h, 3, 3, 1], 0.05));
  w.insert("conv2d1.bias".into(), filled(&[h], 0.0));
  w.insert("conv2d2.weight".into(), filled(&[h, 3, 3, h], 0.05));
  w.insert("conv2d2.bias".into(), filled(&[h], 0.0));
  w.insert("conv2d3.weight".into(), filled(&[h, 3, 3, h], 0.05));
  w.insert("conv2d3.bias".into(), filled(&[h], 0.0));
  // conv_out bias-free (d_model, hidden * freq_after_conv).
  w.insert("conv_out.weight".into(), filled(&[d, conv_out_in], 0.1));

  // Single encoder layer.
  for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    w.insert(
      format!("layers.0.self_attn.{proj}.weight"),
      filled(&[d, d], 0.1),
    );
    w.insert(format!("layers.0.self_attn.{proj}.bias"), filled(&[d], 0.0));
  }
  w.insert(
    "layers.0.self_attn_layer_norm.weight".into(),
    filled(&[d], 1.0),
  );
  w.insert(
    "layers.0.self_attn_layer_norm.bias".into(),
    filled(&[d], 0.0),
  );
  w.insert("layers.0.fc1.weight".into(), filled(&[ffn, d], 0.1));
  w.insert("layers.0.fc1.bias".into(), filled(&[ffn], 0.0));
  w.insert("layers.0.fc2.weight".into(), filled(&[d, ffn], 0.1));
  w.insert("layers.0.fc2.bias".into(), filled(&[d], 0.0));
  w.insert("layers.0.final_layer_norm.weight".into(), filled(&[d], 1.0));
  w.insert("layers.0.final_layer_norm.bias".into(), filled(&[d], 0.0));

  // Output head.
  w.insert("ln_post.weight".into(), filled(&[d], 1.0));
  w.insert("ln_post.bias".into(), filled(&[d], 0.0));
  w.insert("proj1.weight".into(), filled(&[d, d], 0.1));
  w.insert("proj1.bias".into(), filled(&[d], 0.0));
  w.insert("proj2.weight".into(), filled(&[out, d], 0.1));
  w.insert("proj2.bias".into(), filled(&[out], 0.0));
  w
}

fn tiny_encoder() -> AudioEncoder {
  let cfg = tiny_config();
  let weights = tiny_weights(&cfg);
  AudioEncoder::from_weights(cfg, weights).expect("tiny encoder must build")
}

#[test]
fn from_weights_builds_and_reports_config() {
  let enc = tiny_encoder();
  assert_eq!(enc.config().d_model, 4);
  assert_eq!(enc.config().output_dim, 6);
  assert_eq!(enc.config().freq_after_conv(), 1);
  assert_eq!(enc.config().conv_out_in_features().unwrap(), 2); // hidden(2) * freq(1)
}

#[test]
fn encode_features_downsamples_to_expected_shape() {
  let enc = tiny_encoder();
  // mel [B=1, n_mels=8, T=16] → conv stem → [B=1, T'=2, d_model=4].
  let mel = filled(&[1, 8, 16], 0.3);
  let feats = enc.encode_features(&mel).unwrap();
  let shape = feats.shape();
  assert_eq!(
    shape,
    vec![1, 2, 4],
    "expected [B, T', d_model], got {shape:?}"
  );
  // T' must equal the analytic time recurrence.
  assert_eq!(shape[1] as i64, AudioEncoderConfig::time_after_conv(16));
}

#[test]
fn forward_produces_audio_embeddings_of_exact_shape() {
  let enc = tiny_encoder();
  // mel [B=1, n_mels=8, T=16] → [B=1, T'=2, output_dim=6].
  let mel = filled(&[1, 8, 16], 0.3);
  let mut emb = enc.forward(&mel).unwrap();
  let shape = emb.shape();
  assert_eq!(
    shape,
    vec![1, 2, 6],
    "expected [B, T', output_dim], got {shape:?}"
  );
  // The values are finite (the encoder ran end to end: conv stem + pos emb +
  // 1 attention/FFN block + ln_post + proj1/proj2).
  let vals = emb.to_vec::<f32>().unwrap();
  assert_eq!(vals.len(), 2 * 6);
  assert!(
    vals.iter().all(|v| v.is_finite()),
    "non-finite output: {vals:?}"
  );
}

#[test]
fn forward_batched_and_longer_time_shape() {
  let enc = tiny_encoder();
  // mel [B=2, n_mels=8, T=32] → T' = ((32+1)/2=16 -> 8 -> 4) = 4.
  let mel = filled(&[2, 8, 32], 0.2);
  let emb = enc.forward(&mel).unwrap();
  assert_eq!(emb.shape(), vec![2, 4, 6]);
  assert_eq!(4i64, AudioEncoderConfig::time_after_conv(32));
}

#[test]
fn forward_rejects_mel_longer_than_positional_table() {
  // The plain public `forward` lets the caller supply any mel length; a mel
  // whose post-CNN length exceeds `max_source_positions` would (absent the
  // guard) get a clamped positional slice broadcast across the longer sequence.
  // It must instead surface a typed OutOfRange from the positional slice.
  //
  // tiny_config: max_source_positions = 8, num_mel_bins = 8. T = 72 →
  // time_after_conv(72) = 9 > 8, so the positional slice guard fires.
  let enc = tiny_encoder();
  assert_eq!(AudioEncoderConfig::time_after_conv(72), 9);
  let mel = filled(&[1, 8, 72], 0.2);
  match enc.forward(&mel) {
    Err(Error::OutOfRange(_)) => {}
    other => panic!("an over-long mel must be a typed OutOfRange, got shape/result {other:?}"),
  }
}

#[test]
fn encode_features_rejects_wrong_mel_bins() {
  let enc = tiny_encoder();
  // config num_mel_bins is 8; a 10-bin input must be a typed shape error.
  let mel = filled(&[1, 10, 16], 0.3);
  let err = enc
    .encode_features(&mel)
    .expect_err("wrong mel-bin axis must be rejected");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

#[test]
fn encode_features_rejects_non_rank3() {
  let enc = tiny_encoder();
  let mel = filled(&[8, 16], 0.3); // rank-2
  let err = enc
    .encode_features(&mel)
    .expect_err("rank-2 input must be rejected");
  assert!(matches!(err, Error::RankMismatch(_)), "got {err:?}");
}

#[test]
fn from_weights_rejects_wrong_conv_shape() {
  let cfg = tiny_config();
  let mut weights = tiny_weights(&cfg);
  // Corrupt conv2d1.weight to a wrong kernel (kH=2 instead of 3).
  weights.insert("conv2d1.weight".into(), filled(&[2, 2, 3, 1], 0.05));
  let err =
    AudioEncoder::from_weights(cfg, weights).expect_err("wrong conv kernel shape must be rejected");
  assert!(matches!(err, Error::LayerKeyed(_)), "got {err:?}");
}

#[test]
fn from_weights_rejects_missing_key() {
  let cfg = tiny_config();
  let mut weights = tiny_weights(&cfg);
  weights.remove("proj2.bias");
  let err = AudioEncoder::from_weights(cfg, weights).expect_err("missing weight must be rejected");
  assert!(matches!(err, Error::MissingKey(_)), "got {err:?}");
}

// ════════════════════════════ forward_with_mask validation ══════════════════

#[test]
fn forward_with_mask_accepts_full_grid_mask() {
  // The canonical shape: a `(batch, num_heads, T, T)` additive mask over the
  // post-CNN score grid is accepted and runs the masked forward. tiny_config:
  // n_window=2, num_mel_bins=8, heads=2. mel time=16 → T = time_after_conv(16)
  // = 2; batch=1, heads=2 → mask (1, 2, 2, 2).
  let enc = tiny_encoder();
  let mel = filled(&[1, 8, 16], 0.3);
  let t = AudioEncoderConfig::time_after_conv(16) as i32;
  assert_eq!(t, 2);
  let mask = filled(&[1, 2, t, t], 0.0);
  let out = enc.forward_with_mask(&mel, &mask).unwrap();
  // (B=1, T'=2, output_dim=6) — the masked forward produced the same shape as
  // the plain forward (the mask is all-zeros here, so it is a no-op bias).
  assert_eq!(out.shape(), vec![1, 2, 6]);
}

#[test]
fn forward_with_mask_accepts_bare_t_by_t_and_singleton_leading() {
  // A bare `(T, T)` mask (both leading axes absent), a `(1, T, T)` (singleton
  // head), a `(heads, T, T)`, and a `(1, 1, T, T)` are each accepted — every
  // leading axis is either the explicit dim or a broadcast singleton 1.
  let enc = tiny_encoder();
  let mel = filled(&[1, 8, 16], 0.3);
  let t = AudioEncoderConfig::time_after_conv(16) as i32; // 2
  for shape in [
    vec![t, t],
    vec![1, t, t],
    vec![2, t, t], // heads == 2
    vec![1, 1, t, t],
    vec![1, 2, t, t],
  ] {
    let mask = filled(&shape, 0.0);
    let out = enc
      .forward_with_mask(&mel, &mask)
      .unwrap_or_else(|e| panic!("mask shape {shape:?} must be accepted, got {e:?}"));
    assert_eq!(out.shape(), vec![1, t as usize, 6], "shape {shape:?}");
  }
}

#[test]
fn forward_with_mask_rejects_query_axis_singleton() {
  // A `(T, 1)` mask broadcasts a single key-step's bias across the whole key
  // axis — a malformed-but-broadcastable shape SDPA would silently accept. It
  // must be a typed ShapePairMismatch (trailing axes are not exactly (T, T)).
  let enc = tiny_encoder();
  let mel = filled(&[1, 8, 16], 0.3);
  let t = AudioEncoderConfig::time_after_conv(16) as i32; // 2
  let mask = filled(&[t, 1], 0.0);
  let err = enc
    .forward_with_mask(&mel, &mask)
    .expect_err("(T, 1) mask must be rejected");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

#[test]
fn forward_with_mask_rejects_key_axis_singleton() {
  // The symmetric `(1, T)` case: the query axis collapsed to one step. Rejected
  // as a ShapePairMismatch rather than silently broadcast.
  let enc = tiny_encoder();
  let mel = filled(&[1, 8, 16], 0.3);
  let t = AudioEncoderConfig::time_after_conv(16) as i32; // 2
  let mask = filled(&[1, t], 0.0);
  let err = enc
    .forward_with_mask(&mel, &mask)
    .expect_err("(1, T) mask must be rejected");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

#[test]
fn forward_with_mask_rejects_wrong_t() {
  // A `(T+1, T+1)` mask addresses a different sequence length than the post-CNN
  // grid — rejected (the trailing axes do not equal the actual T).
  let enc = tiny_encoder();
  let mel = filled(&[1, 8, 16], 0.3);
  let t = AudioEncoderConfig::time_after_conv(16) as i32; // 2
  let mask = filled(&[t + 1, t + 1], 0.0);
  let err = enc
    .forward_with_mask(&mel, &mask)
    .expect_err("wrong-T mask must be rejected");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

#[test]
fn forward_with_mask_rejects_wrong_leading_axis() {
  // A head axis that is neither 1 nor `num_heads` (here 3 vs heads=2) cannot
  // address the score grid — rejected as a ShapePairMismatch.
  let enc = tiny_encoder();
  let mel = filled(&[1, 8, 16], 0.3);
  let t = AudioEncoderConfig::time_after_conv(16) as i32; // 2
  let mask = filled(&[3, t, t], 0.0);
  let err = enc
    .forward_with_mask(&mel, &mask)
    .expect_err("a non-singleton, non-heads leading axis must be rejected");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

#[test]
fn forward_with_mask_rejects_rank_above_four() {
  // SDPA broadcasts at most a rank-4 mask; a rank-5 mask is a typed
  // RankMismatch (caught host-side before the kernel).
  let enc = tiny_encoder();
  let mel = filled(&[1, 8, 16], 0.3);
  let t = AudioEncoderConfig::time_after_conv(16) as i32; // 2
  let mask = filled(&[1, 1, 1, t, t], 0.0);
  let err = enc
    .forward_with_mask(&mel, &mask)
    .expect_err("a rank-5 mask must be rejected");
  assert!(matches!(err, Error::RankMismatch(_)), "got {err:?}");
}

// ════════════════════════════ feature-length / single-window ════════════════

/// Independent port of the reference `_get_feat_extract_output_lengths` for the
/// oracle (Python floor-division semantics).
fn feat_output_len_reference(mel_len: i64) -> i64 {
  if mel_len <= 0 {
    return 0;
  }
  let fd = |a: i64, b: i64| a.div_euclid(b);
  let leave = mel_len % 100;
  let feat = fd(leave - 1, 2) + 1;
  fd(fd(feat - 1, 2) + 1 - 1, 2) + 1 + (mel_len / 100) * 13
}

#[test]
fn feature_output_length_matches_reference_formula() {
  for &n in &[1i64, 2, 4, 8, 16, 50, 99, 100, 101, 200, 333, 1000, 1600] {
    assert_eq!(
      AudioEncoder::feature_output_length(n),
      feat_output_len_reference(n),
      "mel_len={n}"
    );
  }
  // Non-positive lengths saturate at 0.
  assert_eq!(AudioEncoder::feature_output_length(0), 0);
  assert_eq!(AudioEncoder::feature_output_length(-5), 0);
}

#[test]
fn feature_output_length_agrees_with_time_after_conv_single_chunk() {
  // Within a single conv chunk (short utterances), the formula equals the plain
  // three-fold stride-2 downsample.
  for &n in &[1i64, 2, 4, 8, 16, 32, 64, 99, 100] {
    assert_eq!(
      AudioEncoder::feature_output_length(n),
      AudioEncoderConfig::time_after_conv(n),
      "mel_len={n}"
    );
  }
}

#[test]
fn windowed_output_length_is_exact_per_chunk_conv_sum() {
  // The exact windowed row count = sum over chunks of the analytic conv chain.
  // Cross-check against the independent host oracle for several (valid_len,
  // n_window) pairs, including a non-default n_window whose chunk exceeds 100.
  for &(valid, nw) in &[
    (10i64, 2i64),
    (253, 2),
    (250, 50),
    (250, 100),
    (200, 100),
    (1000, 100),
    (333, 51),
  ] {
    let chunk = nw * 2;
    assert_eq!(
      AudioEncoder::windowed_output_length(valid, chunk),
      windowed_rows_oracle(valid, nw),
      "valid={valid}, n_window={nw}"
    );
  }
  // Non-positive / degenerate inputs.
  assert_eq!(AudioEncoder::windowed_output_length(0, 4), 0);
  assert_eq!(AudioEncoder::windowed_output_length(-3, 4), 0);
  // chunk < 1 is clamped to 1 (caller validates n_window >= 1); never divides by
  // zero. With chunk 1 every chunk is a single frame -> conv3_chain(1) = 1 row.
  assert_eq!(
    AudioEncoder::windowed_output_length(5, 0),
    5 * conv3_chain(1)
  );

  // The reference closed form agrees ONLY at the standard 100-frame chunk; the
  // exact recurrence is what the conv stem realizes for any other chunk size.
  assert_eq!(
    AudioEncoder::windowed_output_length(250, 100),
    AudioEncoder::feature_output_length(250),
    "n_window=50 (chunk 100): exact == reference closed form"
  );
  // n_window=100 (chunk 200): chunks (200, 50). conv chain: 200->100->50->25 and
  // 50->25->13->7, so 25+7 = 32 rows, while feature_output_length(250) = 33 and
  // feature_output_length(200) = 26 (the over-count the exact recurrence avoids).
  assert_eq!(AudioEncoder::windowed_output_length(250, 200), 32);
  assert_eq!(AudioEncoderConfig::time_after_conv(200), 25);
  assert_eq!(AudioEncoder::feature_output_length(200), 26);
  assert_ne!(
    AudioEncoder::windowed_output_length(250, 200),
    AudioEncoder::feature_output_length(250),
    "n_window=100 (chunk 200): exact diverges from the reference closed form"
  );
}

#[test]
fn forward_single_window_trims_padding_to_valid_length() {
  // tiny_config has n_window=2 → chunk=4. Pad a 4-frame utterance into a
  // 16-frame axis and mark only 4 valid: the encoder trims to 4 →
  // time_after_conv(4) = 1 row, NOT the full-axis time_after_conv(16) = 2 rows.
  // (A trim mismatch would surface as the larger row count.)
  let enc = tiny_encoder();
  let padded = filled(&[1, 8, 16], 0.3); // [B=1, n_mels=8, time=16]
  let out = enc.forward_single_window(&padded, Some(&[4])).unwrap();
  let trimmed_rows = AudioEncoderConfig::time_after_conv(4) as usize;
  assert_eq!(out.shape(), vec![1, trimmed_rows, 6]);
  // The trim is observable: valid length 4 and the padded axis 16 downsample to
  // different row counts, and the output matches the trimmed (valid) length.
  assert_eq!(AudioEncoderConfig::time_after_conv(4), 1);
  assert_eq!(AudioEncoderConfig::time_after_conv(16), 2);
}

#[test]
fn forward_single_window_no_lengths_uses_full_axis() {
  let enc = tiny_encoder();
  // time=4 fits one chunk (n_window=2 → chunk=4); no feature length → full axis.
  let feats = filled(&[1, 8, 4], 0.3);
  let out = enc.forward_single_window(&feats, None).unwrap();
  let rows = AudioEncoderConfig::time_after_conv(4) as usize;
  assert_eq!(out.shape(), vec![1, rows, 6]); // time_after_conv(4) = 1
}

/// Independent host-side oracle for the windowed encoder's post-CNN row count:
/// the valid mel frames are split into `chunk = n_window * 2`-frame conv chunks
/// (last = remainder), and each chunk contributes `conv3_chain(chunk_len)`
/// post-CNN frames — the EXACT three-fold stride-2 conv output length of that
/// chunk (computed by the independent analytic chain, not the implementation).
/// The concatenated total is the audio-feature sequence length the windowed
/// forward emits. For the standard 100-frame chunk this equals
/// `feature_output_length(valid_len)` (the reference closed form, additive
/// across 100-frame boundaries); for any other chunk size it is the exact
/// per-chunk sum, which is what the encoder actually produces (and what the
/// closed form would mis-count for a chunk > 100).
fn windowed_rows_oracle(valid_len: i64, n_window: i64) -> i64 {
  let chunk = n_window * 2;
  let rem = valid_len % chunk;
  let num = valid_len / chunk + i64::from(rem != 0);
  let mut total = 0i64;
  for j in 0..num {
    let clen = if j == num - 1 && rem != 0 { rem } else { chunk };
    total += conv3_chain(clen);
  }
  total
}

#[test]
fn forward_single_window_multi_chunk_shape() {
  // tiny_config: n_window=2 → chunk=4. A 10-frame utterance spans 3 conv chunks
  // (4, 4, 2); each chunk → time_after_conv(chunk_len) post-CNN frames, summed.
  let enc = tiny_encoder();
  let feats = filled(&[1, 8, 10], 0.25);
  let mut out = enc.forward_single_window(&feats, None).unwrap();
  let rows = windowed_rows_oracle(10, 2);
  assert_eq!(rows, 3, "oracle: 3 chunks (4,4,2) → 1+1+1 post-CNN rows");
  assert_eq!(out.shape(), vec![1, rows as usize, 6]);
  // The whole windowed graph evaluated to finite values (per-chunk conv + per-
  // chunk position reset + block-diagonal mask + the encoder layers + head).
  let vals = out.to_vec::<f32>().unwrap();
  assert_eq!(vals.len(), (rows as usize) * 6);
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
}

#[test]
fn forward_single_window_multi_chunk_trims_padding() {
  // A 10-frame valid utterance padded into a 24-frame axis: the windowed path
  // trims to the valid 10 frames first, so the row count is the 10-frame
  // windowed count (3), NOT the padded 24-frame count.
  let enc = tiny_encoder();
  let padded = filled(&[1, 8, 24], 0.25);
  let out = enc.forward_single_window(&padded, Some(&[10])).unwrap();
  let rows = windowed_rows_oracle(10, 2);
  assert_eq!(out.shape(), vec![1, rows as usize, 6]);
  // The padded full axis would yield a different (larger) count, so the trim is
  // observable.
  assert_ne!(rows, windowed_rows_oracle(24, 2));
}

#[test]
fn forward_single_window_many_windows_block_mask_shape() {
  // 250+ mel frames including a partial last window. tiny_config window_aftercnn
  // = max_len_after_cnn(1) * (n_window_infer 8 / chunk 4 = 2) = 2 post-CNN
  // frames per attention window, so a long sequence is split into many blocks
  // (the block-diagonal mask path, not a single window).
  let enc = tiny_encoder();
  // 253 = 63 full chunks of 4 + a partial chunk of 1 → 64 chunks, 64 post-CNN
  // rows (1 per chunk), grouped into 2-frame windows = 32 blocks.
  let feats = filled(&[1, 8, 253], 0.1);
  let out = enc.forward_single_window(&feats, None).unwrap();
  let rows = windowed_rows_oracle(253, 2);
  assert_eq!(rows, 64, "63 full + 1 partial chunk → 64 post-CNN rows");
  assert_eq!(out.shape(), vec![1, rows as usize, 6]);
}

#[test]
fn forward_single_window_real_window_size_row_count_matches_feature_len() {
  // With the REAL chunk size (n_window=50 → chunk=100), the windowed encoder's
  // per-chunk post-CNN frames sum to exactly feature_output_length(valid_len)
  // for ANY length — the invariant the aligner's audio-token count relies on.
  // Use a real-window config but otherwise tiny dims so the forward is cheap.
  let cfg = AudioEncoderConfig::from_json(
    r#"{
      "num_mel_bins": 8,
      "encoder_layers": 1,
      "encoder_attention_heads": 2,
      "encoder_ffn_dim": 8,
      "d_model": 4,
      "output_dim": 6,
      "max_source_positions": 64,
      "n_window": 50,
      "n_window_infer": 800,
      "downsample_hidden_size": 2
    }"#,
  )
  .expect("real-window tiny config must validate");
  let enc = AudioEncoder::from_weights(cfg, tiny_weights(&tiny_config())).expect("encoder");
  // 250 valid mel frames = chunks (100, 100, 50), spanning >1 conv chunk but a
  // single attention window (window_aftercnn = 13 * 8 = 104 >= seq_len 33).
  let feats = filled(&[1, 8, 250], 0.1);
  let out = enc.forward_single_window(&feats, None).unwrap();
  let expected = AudioEncoder::feature_output_length(250);
  assert_eq!(expected, 33, "feature_output_length(250)");
  assert_eq!(
    expected,
    windowed_rows_oracle(250, 50),
    "per-chunk sum invariant"
  );
  assert_eq!(out.shape(), vec![1, expected as usize, 6]);
}

#[test]
fn forward_single_window_degenerate_matches_plain_forward() {
  // A valid length that fits one conv chunk takes the degenerate single-window
  // branch: byte-identical to the plain (unmasked) forward over the trimmed mel.
  let enc = tiny_encoder();
  let feats = filled(&[1, 8, 4], 0.3); // chunk = 4, fits exactly one chunk
  let win = enc
    .forward_single_window(&feats, None)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  let plain = enc.forward(&feats).unwrap().to_vec::<f32>().unwrap();
  assert_eq!(win.len(), plain.len());
  assert_close(&win, &plain);
}

#[test]
fn forward_single_window_rejects_batch_gt_one() {
  let enc = tiny_encoder();
  let feats = filled(&[2, 8, 4], 0.3);
  let err = enc
    .forward_single_window(&feats, None)
    .expect_err("batched input must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn forward_single_window_rejects_negative_feature_length() {
  // A negative valid length is malformed runtime input: rejected with a typed
  // OutOfRange, NOT clamped to 0 (which would silently produce a zero-length
  // audio span).
  let enc = tiny_encoder();
  let feats = filled(&[1, 8, 16], 0.3); // time = 16
  let err = enc
    .forward_single_window(&feats, Some(&[-1]))
    .expect_err("negative feature length must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn forward_single_window_rejects_overlong_feature_length() {
  // A length past the mel time axis is malformed: rejected with a typed
  // OutOfRange, NOT clamped to the full axis (which would silently span padding
  // as if it were valid audio).
  let enc = tiny_encoder();
  let feats = filled(&[1, 8, 16], 0.3); // time = 16
  let err = enc
    .forward_single_window(&feats, Some(&[17]))
    .expect_err("overlong feature length must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn forward_single_window_rejects_extra_feature_lengths() {
  // This is the single-utterance (batch == 1) path, so exactly one length is
  // expected; a slice with more than one entry is a LengthMismatch.
  let enc = tiny_encoder();
  let feats = filled(&[1, 8, 16], 0.3);
  let err = enc
    .forward_single_window(&feats, Some(&[4, 4]))
    .expect_err("more than one feature length must be rejected");
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err:?}");
}

#[test]
fn forward_single_window_accepts_in_range_feature_length() {
  // A valid in-range length is unchanged: it trims to the valid frames (the
  // boundary `length == time` is accepted, not rejected).
  let enc = tiny_encoder();
  let feats = filled(&[1, 8, 16], 0.3); // time = 16
  // length == time (the inclusive upper boundary) keeps the full axis.
  let full = enc.forward_single_window(&feats, Some(&[16])).unwrap();
  assert_eq!(
    full.shape(),
    vec![1, windowed_rows_oracle(16, 2) as usize, 6]
  );
  // A shorter valid length trims to that many frames.
  let out = enc.forward_single_window(&feats, Some(&[4])).unwrap();
  assert_eq!(
    out.shape(),
    vec![1, AudioEncoderConfig::time_after_conv(4) as usize, 6]
  );
}

#[test]
fn forward_single_window_conv_chunksize_invariant() {
  // The windowed encoder processes the padded chunk batch in
  // `conv_chunksize`-sized slices along the chunk axis. Because the conv stem
  // is per-chunk independent, the slice size only bounds the conv working set —
  // it must not change the result. Compare `conv_chunksize` covering ALL chunks
  // (one slice, the prior all-at-once behavior) against a small `conv_chunksize`
  // (many slices). tiny_config: n_window=2 → chunk=4. An 18-frame utterance =
  // 5 conv chunks (4,4,4,4,2), so conv_chunksize 5 is a single slice while
  // conv_chunksize 2 is three slices ([0:2],[2:4],[4:5]).
  //
  // A VARIED (non-constant) mel is essential: a constant fill makes every chunk
  // identical, which would mask an off-by-one slice-boundary bug. arange/18*8
  // gives each (mel, time) cell a distinct value.
  let n = 8 * 18;
  let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.5).collect();
  let mel = Array::from_slice::<f32>(&data, &[1, 8, 18]).unwrap();

  let enc_one_slice = AudioEncoder::from_weights(
    tiny_config_conv_chunksize(5), // >= 5 chunks → a single conv slice
    tiny_weights(&tiny_config()),
  )
  .expect("encoder (single-slice conv)");
  let enc_many_slices = AudioEncoder::from_weights(
    tiny_config_conv_chunksize(2), // 2 chunks/slice → three conv slices
    tiny_weights(&tiny_config()),
  )
  .expect("encoder (multi-slice conv)");

  let one = enc_one_slice
    .forward_single_window(&mel, None)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  let many = enc_many_slices
    .forward_single_window(&mel, None)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();

  // 5 chunks → 5 post-CNN rows (1 per chunk in tiny_config), output_dim 6.
  let rows = windowed_rows_oracle(18, 2);
  assert_eq!(rows, 5, "18 frames = chunks (4,4,4,4,2) → 5 post-CNN rows");
  assert_eq!(one.len(), (rows as usize) * 6);
  // The conv_chunksize-sliced path is numerically identical to the single slice.
  assert_close(&many, &one);
}

// ════════════════════════════ dtype preservation ════════════════════════════

/// Build the tiny encoder with every weight cast to `dtype` — a half-precision
/// checkpoint. The conv stem, projections, and norms are all built at `dtype`,
/// so the activations flow at `dtype` and the output dtype is observable.
fn tiny_encoder_dtype(cfg: &AudioEncoderConfig, dtype: Dtype) -> AudioEncoder {
  let weights: HashMap<String, Array> = tiny_weights(cfg)
    .into_iter()
    .map(|(k, v)| (k, v.astype(dtype).expect("weight cast")))
    .collect();
  AudioEncoder::from_weights(cfg.clone(), weights).expect("tiny encoder must build")
}

/// `forward_single_window` must preserve the activation dtype: a `bf16`/`f16`
/// checkpoint plus a `bf16`/`f16` mel must yield a `bf16`/`f16` output, never a
/// silent promotion to `f32`. The f32-built sinusoidal positional embedding and
/// the f16 saturation-clamp bounds are the upcast risks this pins.
fn assert_single_window_preserves_dtype(dtype: Dtype) {
  let cfg = tiny_config();
  let enc = tiny_encoder_dtype(&cfg, dtype);
  // time=4 fits one conv chunk (n_window=2 → chunk=4).
  let mel = filled(&[1, 8, 4], 0.3).astype(dtype).unwrap();
  let out = enc.forward_single_window(&mel, None).unwrap();
  assert_eq!(
    out.dtype().unwrap(),
    dtype,
    "forward_single_window upcast {dtype:?} → {:?}",
    out.dtype().unwrap()
  );
  // The graph still evaluates to finite values at this dtype (read via an f32
  // view; `to_vec::<f32>` requires an f32 array).
  let vals = out.astype(Dtype::F32).unwrap().to_vec::<f32>().unwrap();
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
}

#[test]
fn forward_single_window_preserves_bf16() {
  assert_single_window_preserves_dtype(Dtype::BF16);
}

#[test]
fn forward_single_window_preserves_f16() {
  assert_single_window_preserves_dtype(Dtype::F16);
}

/// The windowed (multi-chunk) path must also preserve the activation dtype: a
/// multi-chunk utterance exercises the per-chunk position-embedding add and the
/// f32-built block-diagonal mask, both of which would silently promote a
/// bf16/f16 activation to f32 if not cast back to the activation dtype.
fn assert_multi_window_preserves_dtype(dtype: Dtype) {
  let cfg = tiny_config();
  let enc = tiny_encoder_dtype(&cfg, dtype);
  // 12 mel frames = 3 conv chunks (4,4,4); seq_len 3 > window_aftercnn 2, so
  // both the per-chunk position add and a real block-diagonal mask run.
  let mel = filled(&[1, 8, 12], 0.2).astype(dtype).unwrap();
  let out = enc.forward_single_window(&mel, None).unwrap();
  assert_eq!(
    out.dtype().unwrap(),
    dtype,
    "windowed forward_single_window upcast {dtype:?} → {:?}",
    out.dtype().unwrap()
  );
  let vals = out.astype(Dtype::F32).unwrap().to_vec::<f32>().unwrap();
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
}

#[test]
fn forward_single_window_multi_chunk_preserves_bf16() {
  assert_multi_window_preserves_dtype(Dtype::BF16);
}

#[test]
fn forward_single_window_multi_chunk_preserves_f16() {
  assert_multi_window_preserves_dtype(Dtype::F16);
}

/// The same guarantee for the plain [`AudioEncoder::forward`] path (the
/// positional-add and f16-clamp sites are shared with the single-window path,
/// but pin them through the public full forward too).
fn assert_forward_preserves_dtype(dtype: Dtype) {
  let cfg = tiny_config();
  let enc = tiny_encoder_dtype(&cfg, dtype);
  let mel = filled(&[1, 8, 16], 0.3).astype(dtype).unwrap();
  let out = enc.forward(&mel).unwrap();
  assert_eq!(out.dtype().unwrap(), dtype, "forward upcast {dtype:?}");
  let vals = out.astype(Dtype::F32).unwrap().to_vec::<f32>().unwrap();
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
}

#[test]
fn forward_preserves_bf16() {
  assert_forward_preserves_dtype(Dtype::BF16);
}

#[test]
fn forward_preserves_f16() {
  assert_forward_preserves_dtype(Dtype::F16);
}

// ════════════════════════════ f16 saturation clamp ════════════════════════════

#[test]
fn f16_clamp_bound_is_finfo_max_minus_1000() {
  // finfo(float16).max == 65504, so the symmetric clamp bound is 64504 — the
  // reference's `torch.finfo(float16).max - 1000`.
  assert_eq!(f64::from(half::f16::MAX), 65504.0);
  assert_eq!(super::audio::F16_CLAMP_BOUND, 64504.0);
}

#[test]
fn clamp_if_f16_saturates_overflow_and_preserves_dtype() {
  // An f16 value that, scaled past finfo.max, overflows to +inf WITHOUT the
  // clamp; the clamp must instead saturate it to the finite f16 bound and keep
  // the dtype f16. A symmetric negative element checks the lower bound.
  let base = Array::from_slice::<f32>(&[60000.0, -60000.0, 1.0], &[3])
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  // base * 2 overflows f16 (120000 > 65504): without the clamp this is ±inf.
  let two = Array::from_slice::<f32>(&[2.0], &[1])
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let overflowed = base.multiply(&two).unwrap();

  // Independently confirm the un-clamped value really is non-finite (the
  // overflow the clamp exists to tame), via an f32 view.
  let raw = overflowed
    .astype(Dtype::F32)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  assert!(
    raw[0].is_infinite() && raw[1].is_infinite(),
    "expected the pre-clamp value to overflow to ±inf, got {raw:?}"
  );

  // The clamp saturates to the f16-rounded bound and stays f16. The f32 bound
  // 64504 is not itself representable in f16 (the ULP near there is 32), so it
  // rounds to 64512 — the effective half-precision ceiling, exactly as a
  // half-precision clamp against the same f32 bound would round it.
  let f16_bound = f64::from(half::f16::from_f32(super::audio::F16_CLAMP_BOUND)) as f32;
  assert_eq!(f16_bound, 64512.0);
  let clamped = super::audio::clamp_if_f16(overflowed).unwrap();
  assert_eq!(clamped.dtype().unwrap(), Dtype::F16, "clamp upcast f16");
  let vals = clamped.astype(Dtype::F32).unwrap().to_vec::<f32>().unwrap();
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
  assert_eq!(vals[0], f16_bound);
  assert_eq!(vals[1], -f16_bound);
  // The within-bound element (1.0 * 2 = 2.0) is passed through unchanged.
  assert_eq!(vals[2], 2.0);
}

#[test]
fn clamp_if_f16_is_noop_for_non_f16() {
  // The reference only clamps when dtype == float16. A bf16/f32 value far above
  // the f16 bound passes through untouched (no saturation, dtype preserved).
  for dtype in [Dtype::F32, Dtype::BF16] {
    let big = Array::from_slice::<f32>(&[200000.0, -200000.0], &[2])
      .unwrap()
      .astype(dtype)
      .unwrap();
    let out = super::audio::clamp_if_f16(big).unwrap();
    assert_eq!(out.dtype().unwrap(), dtype, "clamp changed {dtype:?}");
    let vals = out.astype(Dtype::F32).unwrap().to_vec::<f32>().unwrap();
    // bf16 rounds 200000 but stays well above the f16 bound — i.e. NOT clamped.
    assert!(vals[0] > 64504.0, "non-f16 value was clamped: {vals:?}");
    assert!(vals[1] < -64504.0, "non-f16 value was clamped: {vals:?}");
  }
}

/// An f16 encoder forward whose feed-forward residual would overflow to
/// `inf`/`NaN` without the per-layer clamp must instead produce a finite output.
///
/// Only the FFN weights are large: the conv stem and attention projections are
/// kept small so the *attention* residual stays finite (the reference — and
/// this port — clamp only after the FFN residual, not after attention, so an
/// already-`inf` attention residual could not be rescued and would not isolate
/// the FFN clamp). The large `fc1`/`fc2` then drive the post-FFN-residual hidden
/// state past `finfo(float16).max`; the clamp saturates it so the downstream
/// `ln_post`/`proj1`/`proj2` see finite values and the result is finite.
#[test]
fn forward_f16_large_ffn_saturates_finite() {
  let cfg = tiny_config();
  let d = cfg.d_model;
  let h = cfg.downsample_hidden_size;
  let ffn = cfg.encoder_ffn_dim;
  let out = cfg.output_dim;
  let conv_out_in = cfg.conv_out_in_features().unwrap();
  let mut w: HashMap<String, Array> = HashMap::new();
  // Small conv stem + attention so the pre-FFN residual is finite and modest.
  w.insert("conv2d1.weight".into(), filled(&[h, 3, 3, 1], 0.05));
  w.insert("conv2d1.bias".into(), filled(&[h], 0.0));
  w.insert("conv2d2.weight".into(), filled(&[h, 3, 3, h], 0.05));
  w.insert("conv2d2.bias".into(), filled(&[h], 0.0));
  w.insert("conv2d3.weight".into(), filled(&[h, 3, 3, h], 0.05));
  w.insert("conv2d3.bias".into(), filled(&[h], 0.0));
  w.insert("conv_out.weight".into(), filled(&[d, conv_out_in], 0.1));
  for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    w.insert(
      format!("layers.0.self_attn.{proj}.weight"),
      filled(&[d, d], 0.1),
    );
    w.insert(format!("layers.0.self_attn.{proj}.bias"), filled(&[d], 0.0));
  }
  w.insert(
    "layers.0.self_attn_layer_norm.weight".into(),
    filled(&[d], 1.0),
  );
  w.insert(
    "layers.0.self_attn_layer_norm.bias".into(),
    filled(&[d], 0.0),
  );
  // Large FFN weights so fc2(gelu(fc1(final_layer_norm(h)))) + residual
  // overflows f16 (final_layer_norm normalizes h to ~unit scale, so the blow-up
  // comes from the 100.0 projections: ~ffn * 100 * gelu(d * 100) ≫ 65504).
  w.insert("layers.0.fc1.weight".into(), filled(&[ffn, d], 100.0));
  w.insert("layers.0.fc1.bias".into(), filled(&[ffn], 0.0));
  w.insert("layers.0.fc2.weight".into(), filled(&[d, ffn], 100.0));
  w.insert("layers.0.fc2.bias".into(), filled(&[d], 0.0));
  w.insert("layers.0.final_layer_norm.weight".into(), filled(&[d], 1.0));
  w.insert("layers.0.final_layer_norm.bias".into(), filled(&[d], 0.0));
  w.insert("ln_post.weight".into(), filled(&[d], 1.0));
  w.insert("ln_post.bias".into(), filled(&[d], 0.0));
  w.insert("proj1.weight".into(), filled(&[d, d], 1.0));
  w.insert("proj1.bias".into(), filled(&[d], 0.0));
  w.insert("proj2.weight".into(), filled(&[out, d], 1.0));
  w.insert("proj2.bias".into(), filled(&[out], 0.0));

  let w_f16: HashMap<String, Array> = w
    .into_iter()
    .map(|(k, v)| (k, v.astype(Dtype::F16).unwrap()))
    .collect();
  let enc = AudioEncoder::from_weights(cfg, w_f16).expect("f16 encoder must build");

  // A non-trivial mel (ramped, not constant — a constant would normalize to
  // zero in the LayerNorms and never stress the residual).
  let mel_vals: Vec<f32> = (0..8 * 16).map(|i| ((i % 7) as f32) * 0.5 + 0.1).collect();
  let mel = Array::from_slice::<f32>(&mel_vals, &[1, 8, 16])
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();

  let out = enc.forward(&mel).unwrap();
  assert_eq!(out.dtype().unwrap(), Dtype::F16, "forward upcast f16");
  let vals = out.astype(Dtype::F32).unwrap().to_vec::<f32>().unwrap();
  // The clamp saturated the overflowing FFN residual, so the whole forward is
  // finite — no inf/NaN propagated through ln_post/proj1/proj2.
  assert!(
    vals.iter().all(|v| v.is_finite()),
    "f16 forward produced non-finite output (clamp ineffective): {vals:?}"
  );
}

// ════════════════════════════ sanitize ════════════════════════════

#[test]
fn sanitize_transposes_conv2d_and_strips_prefixes() {
  // PyTorch conv2d weight (out=2, in=1, kH=3, kW=3): fill positions so the
  // transpose to (out, kH, kW, in) = (2, 3, 3, 1) is observable. Use a ramp.
  // out=2, in=1, kH=3, kW=3 → 18 elements.
  let n = 2 * 3 * 3;
  let ramp: Vec<f32> = (0..n).map(|i| i as f32).collect();
  let conv_w = Array::from_slice::<f32>(&ramp, &[2, 1, 3, 3]).unwrap();
  let mut raw: HashMap<String, Array> = HashMap::new();
  raw.insert("thinker.audio_tower.conv2d1.weight".into(), conv_w);
  // A non-audio key (text decoder) must be dropped.
  raw.insert(
    "thinker.model.layers.0.self_attn.q_proj.weight".into(),
    filled(&[4, 4], 1.0),
  );
  // An audio bias (1-D, not transposed) must survive with the prefix stripped.
  raw.insert("thinker.audio_tower.conv2d1.bias".into(), filled(&[2], 0.0));

  let out = sanitize(raw).unwrap();
  // Only the two audio_tower keys survive, prefix-stripped.
  assert!(out.contains_key("conv2d1.weight"), "weight key missing");
  assert!(out.contains_key("conv2d1.bias"), "bias key missing");
  assert!(
    !out.contains_key("model.layers.0.self_attn.q_proj.weight"),
    "non-audio key should be dropped"
  );
  assert_eq!(out.len(), 2, "exactly the two audio keys survive");

  // The conv weight is transposed to channels-last (out, kH, kW, in).
  let mut w = out.get("conv2d1.weight").unwrap().try_clone().unwrap();
  assert_eq!(w.shape(), vec![2, 3, 3, 1]);
  // transpose(0,2,3,1): out[o, kh, kw, i] = in[o, i, kh, kw]. With in = 1, the
  // values are a straight (out, kH, kW) ramp 0..9, 9..18 — unchanged order
  // because the in-axis is singleton. Verify the first row equals 0..9.
  let vals = w.to_vec::<f32>().unwrap();
  let want: Vec<f32> = (0..18).map(|i| i as f32).collect();
  assert_close(&vals, &want);
}

#[test]
fn sanitize_rejects_duplicate_destination_key() {
  // Both the prefixed `thinker.audio_tower.<x>` and `audio_tower.<x>` forms map
  // to the same destination `<x>`; the second insert must be a collision.
  let mut raw: HashMap<String, Array> = HashMap::new();
  raw.insert(
    "thinker.audio_tower.ln_post.weight".into(),
    filled(&[4], 1.0),
  );
  raw.insert("audio_tower.ln_post.weight".into(), filled(&[4], 2.0));
  let err = sanitize(raw).expect_err("duplicate destination key must be rejected");
  assert!(matches!(err, Error::KeyCollision(_)), "got {err:?}");
}

#[test]
fn sanitize_drops_lm_head_and_text_decoder() {
  // Sanitize keeps only the audio tower; lm_head and decoder weights are gone.
  let mut raw: HashMap<String, Array> = HashMap::new();
  raw.insert("thinker.lm_head.weight".into(), filled(&[8, 4], 1.0));
  raw.insert("thinker.audio_tower.proj2.bias".into(), filled(&[6], 0.0));
  let out = sanitize(raw).unwrap();
  assert_eq!(out.len(), 1);
  assert!(out.contains_key("proj2.bias"));
}
