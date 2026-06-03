//! Oracles for the SenseVoice-Small file-loading factory: `from_weights` /
//! `from_weights_quantized`, the `.scales` quantization discriminator + the
//! `has_relevant_scales` pre-scan, the safetensors shard walk + merge, the
//! `am.mvn` / tokenizer asset loading, and the `load()` -> `Box<dyn Transcribe>`
//! factory + the `model_type` dispatch.
//!
//! Every fixture is synthetic (an in-memory weight map or a tmpdir model
//! directory) — no real checkpoint. The dense / quantized weight maps and the
//! tiny config reuse the same shapes as the `model` oracles so the built model
//! forwards at trivial size; the shard-walk and `load()` tests write real
//! `.safetensors` / `am.mvn` / `tokens.json` files to a per-process tmpdir.

use std::{collections::HashMap, fs, path::PathBuf};

use super::*;
use crate::{
  array::Array,
  audio::stt::model::{CtcModel, Transcribe, TranscribeOptions},
  error::Error,
  lm::quant::{PerLayerQuantization, Quantization, QuantizationOption},
};

// ───────────────────────────── synthetic dims + weights ─────────────────────────────

const D: i32 = 8; // input_size == n_mels (lfr_m = 1, no stacking)
const H: i32 = 4; // output_size (hidden)
const F: i32 = 6; // linear_units
const V: i32 = 32; // vocab_size
const K: i32 = 3; // FSMN kernel size

/// A `(rows, cols)` array filled by `f(r, c)`.
fn filled(rows: i32, cols: i32, f: impl Fn(i32, i32) -> f32) -> Array {
  let mut data = Vec::with_capacity((rows * cols) as usize);
  for r in 0..rows {
    for c in 0..cols {
      data.push(f(r, c));
    }
  }
  Array::from_slice::<f32>(&data, &[rows, cols]).unwrap()
}

/// A 1-D `(n,)` array filled by `f(i)`.
fn filled1(n: i32, f: impl Fn(i32) -> f32) -> Array {
  let data: Vec<f32> = (0..n).map(f).collect();
  Array::from_slice::<f32>(&data, &[n]).unwrap()
}

/// The tiny config JSON: 1 `encoders0` block, 0 `encoders`, 0 `tp`, 1 head,
/// `n_mels = D`, `lfr_m = lfr_n = 1` (no LFR stacking, so the fbank width is D).
fn tiny_config_json() -> String {
  format!(
    r#"{{
      "model_type": "sensevoice",
      "vocab_size": {V},
      "input_size": {D},
      "encoder_conf": {{
        "output_size": {H},
        "attention_heads": 1,
        "linear_units": {F},
        "num_blocks": 1,
        "tp_blocks": 0,
        "kernel_size": {K}
      }},
      "frontend_conf": {{ "n_mels": {D}, "lfr_m": 1, "lfr_n": 1 }}
    }}"#
  )
}

fn tiny_config() -> Config {
  let c: Config = serde_json::from_str(&tiny_config_json()).unwrap();
  c.validate().unwrap();
  c
}

/// The full dense weight map for the tiny encoder + head (`embed` row i = i).
fn tiny_weights() -> HashMap<String, Array> {
  let mut w: HashMap<String, Array> = HashMap::new();
  w.insert(
    "encoder.encoders0.0.self_attn.linear_q_k_v.weight".to_string(),
    filled(3 * H, D, |r, c| 0.01 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.self_attn.linear_out.weight".to_string(),
    filled(H, H, |r, c| if r == c { 0.5 } else { 0.0 }),
  );
  w.insert(
    "encoder.encoders0.0.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (H * K) as usize], &[H, K, 1]).unwrap(),
  );
  w.insert(
    "encoder.encoders0.0.feed_forward.w_1.weight".to_string(),
    filled(F, H, |r, c| 0.02 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.feed_forward.w_2.weight".to_string(),
    filled(H, F, |r, c| 0.02 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.norm1.weight".to_string(),
    filled1(D, |_| 1.0),
  );
  w.insert(
    "encoder.encoders0.0.norm1.bias".to_string(),
    filled1(D, |_| 0.0),
  );
  w.insert(
    "encoder.encoders0.0.norm2.weight".to_string(),
    filled1(H, |_| 1.0),
  );
  w.insert(
    "encoder.encoders0.0.norm2.bias".to_string(),
    filled1(H, |_| 0.0),
  );
  w.insert("encoder.after_norm.weight".to_string(), filled1(H, |_| 1.0));
  w.insert("encoder.after_norm.bias".to_string(), filled1(H, |_| 0.0));
  w.insert("encoder.tp_norm.weight".to_string(), filled1(H, |_| 1.0));
  w.insert("encoder.tp_norm.bias".to_string(), filled1(H, |_| 0.0));
  w.insert(
    "ctc_lo.weight".to_string(),
    filled(V, H, |r, c| 0.03 * ((r + c) as f32)),
  );
  w.insert("ctc_lo.bias".to_string(), filled1(V, |_| 0.0));
  w.insert("embed.weight".to_string(), filled(16, D, |r, _| r as f32));
  w
}

/// The tiny weight map in the **pre-sanitize** on-disk torch layout — the form a
/// real checkpoint stores, which the `load()` / shard-walk path runs through
/// [`sanitize`]. The only difference from [`tiny_weights`] is the depthwise FSMN
/// conv weight: torch stores `(C_out, C_in/groups, K) = (H, 1, K)`, which
/// `sanitize`'s `transpose(0, 2, 1)` turns into the MLX `(H, K, 1)` the encoder
/// consumes (`sensevoice.py:561-562`). The all-zero fsmn weight makes the values
/// identical post-transpose; only the shape must be the pre-sanitize one so the
/// transpose lands the right layout (a post-sanitize `(H, K, 1)` on disk would be
/// double-transposed to `(H, 1, K)` and break the depthwise grouping).
fn tiny_weights_raw() -> HashMap<String, Array> {
  let mut w = tiny_weights();
  w.insert(
    "encoder.encoders0.0.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (H * K) as usize], &[H, 1, K]).unwrap(),
  );
  w
}

/// A 1-D mono waveform of `n` samples (a ramp) — enough for a few LFR frames.
fn ramp_waveform(n: i32) -> Array {
  let data: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();
  Array::from_slice::<f32>(&data, &[n]).unwrap()
}

/// A per-process tmpdir for the on-disk fixtures.
fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_sensevoice_loader_{}_{}",
    std::process::id(),
    name
  ));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

// ───────────────────────────── from_weights (dense) ─────────────────────────────

#[test]
fn from_weights_builds_dense_model_and_forwards() {
  // A dense weight map + config builds a model whose forward produces the
  // (1, T+4, V) log-probs (shape-pinned through the real encoder + head).
  let config = tiny_config();
  let model =
    SenseVoiceModel::from_weights(config, tiny_weights(), SenseVoiceTokenizer::id_join(), None)
      .unwrap();

  let t = 5i32;
  let feats = filled(t, D, |r, c| 0.1 * ((r + c) as f32));
  let feats = crate::ops::shape::reshape(&feats, &[1, t, D]).unwrap();
  let mut log_probs = model.forward(&feats, "auto", false).unwrap();
  assert_eq!(log_probs.shape(), vec![1, (t + 4) as usize, V as usize]);
  // Each frame is a normalized log-softmax distribution.
  let lp = log_probs.to_vec::<f32>().unwrap();
  let frame0: f32 = (0..V as usize).map(|v| lp[v].exp()).sum();
  assert!(
    (frame0 - 1.0).abs() < 1e-4,
    "frame 0 softmax sum = {frame0}"
  );
}

#[test]
fn from_weights_validates_config_first() {
  // A malformed config (output_size not divisible by heads) is rejected before
  // any tensor is built.
  let json = format!(
    r#"{{ "vocab_size": {V}, "input_size": {D},
         "encoder_conf": {{ "output_size": 5, "attention_heads": 2, "num_blocks": 1, "tp_blocks": 0 }} }}"#
  );
  let config: Config = serde_json::from_str(&json).unwrap();
  assert!(matches!(
    SenseVoiceModel::from_weights(config, tiny_weights(), SenseVoiceTokenizer::id_join(), None),
    Err(Error::DivisibilityConstraint(_))
  ));
}

#[test]
fn from_weights_missing_weight_errors() {
  let config = tiny_config();
  let mut w = tiny_weights();
  w.remove("ctc_lo.weight");
  assert!(matches!(
    SenseVoiceModel::from_weights(config, w, SenseVoiceTokenizer::id_join(), None),
    Err(Error::MissingKey(_))
  ));
}

#[test]
fn from_weights_carries_cmvn_into_the_model() {
  // The (means, istd) pair handed to from_weights is applied in extract_features.
  let config = tiny_config();
  let means = filled1(D, |_| 1.0);
  let istd = filled1(D, |_| 2.0);
  let model = SenseVoiceModel::from_weights(
    config,
    tiny_weights(),
    SenseVoiceTokenizer::id_join(),
    Some((means, istd)),
  )
  .unwrap();

  let wav = ramp_waveform(3000);
  let with = model
    .extract_features(&wav)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  let plain = SenseVoiceModel::from_weights(
    tiny_config(),
    tiny_weights(),
    SenseVoiceTokenizer::id_join(),
    None,
  )
  .unwrap()
  .extract_features(&wav)
  .unwrap()
  .to_vec::<f32>()
  .unwrap();
  assert_eq!(with.len(), plain.len());
  assert!(
    with.iter().zip(&plain).any(|(a, b)| (a - b).abs() > 1e-6),
    "CMVN must change the features"
  );
}

// ───────────────────────────── quantized build ─────────────────────────────

/// mlx supports only group sizes 32 / 64 / 128, so the quantized fixtures use a
/// larger config whose quantized-tensor input widths are multiples of QGROUP.
const QGROUP: i32 = 32;
const QH: i32 = 32; // output_size (ctc_lo input width)
const QD: i32 = 64; // input_size (embed input width)
const QF: i32 = 64; // linear_units
const QV: i32 = 40; // vocab

fn quant_config_json() -> String {
  format!(
    r#"{{
      "model_type": "sensevoice",
      "vocab_size": {QV},
      "input_size": {QD},
      "encoder_conf": {{
        "output_size": {QH},
        "attention_heads": 4,
        "linear_units": {QF},
        "num_blocks": 1,
        "tp_blocks": 0,
        "kernel_size": {K}
      }},
      "frontend_conf": {{ "n_mels": {QD}, "lfr_m": 1, "lfr_n": 1 }}
    }}"#
  )
}

fn quant_config() -> Config {
  let c: Config = serde_json::from_str(&quant_config_json()).unwrap();
  c.validate().unwrap();
  c
}

fn quant_weights() -> HashMap<String, Array> {
  let mut w: HashMap<String, Array> = HashMap::new();
  w.insert(
    "encoder.encoders0.0.self_attn.linear_q_k_v.weight".to_string(),
    filled(3 * QH, QD, |r, c| 0.001 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.self_attn.linear_out.weight".to_string(),
    filled(QH, QH, |r, c| if r == c { 0.5 } else { 0.0 }),
  );
  w.insert(
    "encoder.encoders0.0.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (QH * K) as usize], &[QH, K, 1]).unwrap(),
  );
  w.insert(
    "encoder.encoders0.0.feed_forward.w_1.weight".to_string(),
    filled(QF, QH, |r, c| 0.001 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.feed_forward.w_2.weight".to_string(),
    filled(QH, QF, |r, c| 0.001 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.norm1.weight".to_string(),
    filled1(QD, |_| 1.0),
  );
  w.insert(
    "encoder.encoders0.0.norm1.bias".to_string(),
    filled1(QD, |_| 0.0),
  );
  w.insert(
    "encoder.encoders0.0.norm2.weight".to_string(),
    filled1(QH, |_| 1.0),
  );
  w.insert(
    "encoder.encoders0.0.norm2.bias".to_string(),
    filled1(QH, |_| 0.0),
  );
  w.insert(
    "encoder.after_norm.weight".to_string(),
    filled1(QH, |_| 1.0),
  );
  w.insert("encoder.after_norm.bias".to_string(), filled1(QH, |_| 0.0));
  w.insert("encoder.tp_norm.weight".to_string(), filled1(QH, |_| 1.0));
  w.insert("encoder.tp_norm.bias".to_string(), filled1(QH, |_| 0.0));
  w.insert(
    "ctc_lo.weight".to_string(),
    filled(QV, QH, |r, c| 0.001 * ((r + c) as f32)),
  );
  w.insert("ctc_lo.bias".to_string(), filled1(QV, |_| 0.0));
  w.insert("embed.weight".to_string(), filled(16, QD, |r, _| r as f32));
  w
}

/// Replace `<prefix>.weight` with the real 8-bit affine quantized triple.
fn quantize_in_place(w: &mut HashMap<String, Array>, prefix: &str, group_size: i32) {
  let dense = w.remove(&format!("{prefix}.weight")).unwrap();
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, group_size, 8, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

/// The global affine quantization config (no per-layer overrides).
fn quant_block() -> PerLayerQuantization {
  PerLayerQuantization::from_global(Quantization::affine(QGROUP, 8))
}

#[test]
fn from_weights_quantized_builds_quantized_head_and_encoder() {
  // Quantize the whole transformer (every linear + ctc_lo + embed) and assert
  // from_weights_quantized builds them quantized and the forward runs.
  let config = quant_config();
  let mut w = quant_weights();
  for prefix in [
    "encoder.encoders0.0.self_attn.linear_q_k_v",
    "encoder.encoders0.0.self_attn.linear_out",
    "encoder.encoders0.0.feed_forward.w_1",
    "encoder.encoders0.0.feed_forward.w_2",
    "ctc_lo",
    "embed",
  ] {
    quantize_in_place(&mut w, prefix, QGROUP);
  }

  let model = SenseVoiceModel::from_weights_quantized(
    config,
    w,
    SenseVoiceTokenizer::id_join(),
    None,
    Some(&quant_block()),
  )
  .unwrap();

  // The quantized forward runs end-to-end and produces speech-only logits.
  let wav = ramp_waveform(4000);
  let logits = CtcModel::logits(&model, &wav).unwrap();
  assert_eq!(logits.shape().len(), 2);
  assert_eq!(logits.shape()[1], QV as usize);
}

#[test]
fn from_weights_quantized_scales_without_scheme_errors() {
  // A `.scales` sibling present (has_relevant_scales true) but no resolvable
  // global scheme (quantization = None) is a typed InvariantViolation.
  let config = quant_config();
  let mut w = quant_weights();
  quantize_in_place(&mut w, "ctc_lo", QGROUP);
  assert!(matches!(
    SenseVoiceModel::from_weights_quantized(config, w, SenseVoiceTokenizer::id_join(), None, None),
    Err(Error::InvariantViolation(_))
  ));
}

#[test]
fn from_weights_quantized_dense_ignores_stale_quant_block() {
  // A DENSE checkpoint (no `.scales`) loads dense even when a quantization
  // block is supplied — the has_relevant_scales pre-scan gates the scheme so the
  // stale block is never consumed.
  let config = quant_config();
  let model = SenseVoiceModel::from_weights_quantized(
    config,
    quant_weights(),
    SenseVoiceTokenizer::id_join(),
    None,
    Some(&quant_block()),
  )
  .unwrap();
  let wav = ramp_waveform(4000);
  let logits = CtcModel::logits(&model, &wav).unwrap();
  assert_eq!(logits.shape()[1], QV as usize);
}

/// The full set of quantizable prefixes the tiny quant config loads — the head
/// (`ctc_lo` / `embed`) plus the single `encoders0.0` block's four linears.
const QUANT_PREFIXES: &[&str] = &[
  "encoder.encoders0.0.self_attn.linear_q_k_v",
  "encoder.encoders0.0.self_attn.linear_out",
  "encoder.encoders0.0.feed_forward.w_1",
  "encoder.encoders0.0.feed_forward.w_2",
  "ctc_lo",
  "embed",
];

#[test]
fn from_weights_quantized_per_layer_only_no_global_default_loads() {
  // A per-layer-only quantization config (NO global default) loads through
  // `from_weights_quantized`: every quantized layer carries an explicit override
  // and `quantization` (the global default) is `None`. The OLD code collapsed
  // `quantization` to a single global tuple and wrongly REJECTED this (no global
  // → `None` → present `.scales` → InvariantViolation). Per-prefix resolution via
  // `quantization_for` loads it.
  let config = quant_config();
  let mut w = quant_weights();
  let mut per_layer = HashMap::new();
  for prefix in QUANT_PREFIXES {
    quantize_in_place(&mut w, prefix, QGROUP);
    per_layer.insert(
      (*prefix).to_string(),
      QuantizationOption::Quantize(Quantization::affine(QGROUP, 8)),
    );
  }
  // `quantization = None` — only the explicitly-listed layers are quantized.
  let quant = PerLayerQuantization::new(None, per_layer);

  let model = SenseVoiceModel::from_weights_quantized(
    config,
    w,
    SenseVoiceTokenizer::id_join(),
    None,
    Some(&quant),
  )
  .expect("a per-layer-only config (no global default) must load, not be rejected");
  let wav = ramp_waveform(4000);
  let logits = CtcModel::logits(&model, &wav).unwrap();
  assert_eq!(logits.shape().len(), 2);
  assert_eq!(logits.shape()[1], QV as usize);
}

#[test]
fn from_weights_quantized_per_layer_skip_builds_dense_layer_dense() {
  // A global default PLUS a per-layer `Skip` for `ctc_lo`, where `ctc_lo` is
  // DENSE on disk (no `.scales`) and every OTHER quantizable layer is quantized.
  // The pre-scan sees the other layers' `.scales` and threads `quant`; `ctc_lo`'s
  // prefix then resolves to `None` (Skip) and, with no `.scales`, the dense arm
  // builds it. A single collapsed global tuple could not carry the per-layer
  // `Skip`. The model builds + forwards.
  let config = quant_config();
  let mut w = quant_weights();
  for prefix in QUANT_PREFIXES {
    if *prefix == "ctc_lo" {
      continue; // leave `ctc_lo` DENSE on disk
    }
    quantize_in_place(&mut w, prefix, QGROUP);
  }
  let mut per_layer = HashMap::new();
  per_layer.insert("ctc_lo".to_string(), QuantizationOption::Skip);
  let quant = PerLayerQuantization::new(Some(Quantization::affine(QGROUP, 8)), per_layer);

  let model = SenseVoiceModel::from_weights_quantized(
    config,
    w,
    SenseVoiceTokenizer::id_join(),
    None,
    Some(&quant),
  )
  .expect("a per-layer Skip on a dense-on-disk ctc_lo builds it dense, the rest quantized");
  let wav = ramp_waveform(4000);
  let logits = CtcModel::logits(&model, &wav).unwrap();
  assert_eq!(logits.shape().len(), 2);
  assert_eq!(logits.shape()[1], QV as usize);
}

#[test]
fn from_weights_quantized_per_layer_skip_on_scales_layer_errors() {
  // The dual: a `Skip` override on a layer that DOES carry `.scales` on disk is a
  // checkpoint/config inconsistency. The per-prefix resolution yields `None` for
  // `ctc_lo` while its `.scales` is present, which the shared
  // `MaybeQuantizedLinear` rejects with a typed `InvariantViolation` (the
  // per-prefix `None` reached the leaf — the qwen3 contract).
  let config = quant_config();
  let mut w = quant_weights();
  for prefix in QUANT_PREFIXES {
    quantize_in_place(&mut w, prefix, QGROUP);
  }
  let mut per_layer = HashMap::new();
  per_layer.insert("ctc_lo".to_string(), QuantizationOption::Skip);
  let quant = PerLayerQuantization::new(Some(Quantization::affine(QGROUP, 8)), per_layer);
  assert!(
    matches!(
      SenseVoiceModel::from_weights_quantized(
        config,
        w,
        SenseVoiceTokenizer::id_join(),
        None,
        Some(&quant),
      ),
      Err(Error::InvariantViolation(_))
    ),
    "a Skip on a `.scales`-bearing ctc_lo is an InvariantViolation, not a silent quantized build"
  );
}

#[test]
fn from_weights_quantized_per_layer_parameter_override() {
  // A global default group_size = 32 PLUS a per-layer override for `embed` to
  // group_size = 64 (QD = 64 is divisible by 64): `embed` is quantized at 64 and
  // every other layer at the global 32. Resolving the wrong (global = 32) tuple
  // for the group_size = 64 packed `embed` would fail the triple validator, so
  // the build succeeding proves `embed` resolved its OWN override per prefix.
  const EMBED_GROUP: i32 = 64;
  let config = quant_config();
  let mut w = quant_weights();
  for prefix in QUANT_PREFIXES {
    let g = if *prefix == "embed" {
      EMBED_GROUP
    } else {
      QGROUP
    };
    quantize_in_place(&mut w, prefix, g);
  }
  let mut per_layer = HashMap::new();
  per_layer.insert(
    "embed".to_string(),
    QuantizationOption::Quantize(Quantization::affine(EMBED_GROUP, 8)),
  );
  let quant = PerLayerQuantization::new(Some(Quantization::affine(QGROUP, 8)), per_layer);

  let model = SenseVoiceModel::from_weights_quantized(
    config,
    w,
    SenseVoiceTokenizer::id_join(),
    None,
    Some(&quant),
  )
  .expect("the per-layer group_size override for embed must be used, not the global default");
  let wav = ramp_waveform(4000);
  let logits = CtcModel::logits(&model, &wav).unwrap();
  assert_eq!(logits.shape()[1], QV as usize);

  // Cross-check: the global tuple (group_size = 32) mis-decodes the group_size =
  // 64 packed `embed` — the schemes are genuinely distinguishable.
  let mut w_wrong = quant_weights();
  for prefix in QUANT_PREFIXES {
    let g = if *prefix == "embed" {
      EMBED_GROUP
    } else {
      QGROUP
    };
    quantize_in_place(&mut w_wrong, prefix, g);
  }
  assert!(
    SenseVoiceModel::from_weights_quantized(
      quant_config(),
      w_wrong,
      SenseVoiceTokenizer::id_join(),
      None,
      Some(&quant_block()), // global group_size = 32 for ALL — wrong for embed
    )
    .is_err(),
    "the global group_size must mis-decode the group_size=64 packed embed"
  );
}

// ───────────────────────────── has_relevant_scales pre-scan ─────────────────────────────

#[test]
fn has_relevant_scales_false_for_dense_map() {
  let config = tiny_config();
  assert!(!has_relevant_scales(&config, &tiny_weights()));
}

#[test]
fn has_relevant_scales_detects_each_quantizable_prefix() {
  let config = quant_config();
  for prefix in [
    "ctc_lo",
    "embed",
    "encoder.encoders0.0.self_attn.linear_q_k_v",
    "encoder.encoders0.0.self_attn.linear_out",
    "encoder.encoders0.0.feed_forward.w_1",
    "encoder.encoders0.0.feed_forward.w_2",
  ] {
    let mut w = quant_weights();
    quantize_in_place(&mut w, prefix, QGROUP);
    assert!(
      has_relevant_scales(&config, &w),
      "scales on {prefix} must be detected"
    );
  }
}

#[test]
fn has_relevant_scales_ignores_foreign_and_out_of_range_scales() {
  // A `.scales` on a never-quantized layer (the FSMN conv / a norm), a foreign
  // key, or an out-of-range block index is NOT a relevant scale.
  let config = tiny_config();
  let mut w = tiny_weights();
  w.insert(
    "encoder.encoders0.0.self_attn.fsmn_block.scales".to_string(),
    filled1(2, |_| 1.0),
  );
  w.insert("encoder.after_norm.scales".to_string(), filled1(2, |_| 1.0));
  w.insert("foreign.linear.scales".to_string(), filled1(2, |_| 1.0));
  // num_blocks = 1 -> `encoders.0` is out of range; tp_blocks = 0 -> any tp idx.
  w.insert(
    "encoder.encoders.0.self_attn.linear_q_k_v.scales".to_string(),
    filled1(2, |_| 1.0),
  );
  w.insert(
    "encoder.tp_encoders.0.feed_forward.w_1.scales".to_string(),
    filled1(2, |_| 1.0),
  );
  assert!(
    !has_relevant_scales(&config, &w),
    "foreign / never-quantized / out-of-range scales are ignored"
  );
}

// ───────────────────────────── shard walk ─────────────────────────────

#[test]
fn shard_walk_merges_two_shards() {
  // Split the tiny weight map across two `.safetensors` shards and assert the
  // merged build produces a working model.
  let dir = temp_dir("shard_walk");
  let w = tiny_weights_raw();
  // Partition the keys roughly in half across two shards.
  let mut a: HashMap<String, Array> = HashMap::new();
  let mut b: HashMap<String, Array> = HashMap::new();
  for (i, (k, v)) in w.into_iter().enumerate() {
    if i % 2 == 0 {
      a.insert(k, v);
    } else {
      b.insert(k, v);
    }
  }
  crate::io::save_safetensors(&dir.join("model-00001-of-00002.safetensors"), &a).unwrap();
  crate::io::save_safetensors(&dir.join("model-00002-of-00002.safetensors"), &b).unwrap();

  let merged = load_all_safetensors(&dir).unwrap();
  // The merged map has every key from both shards (ctc_lo + embed + the block).
  assert!(merged.contains_key("ctc_lo.weight"));
  assert!(merged.contains_key("embed.weight"));
  assert!(merged.contains_key("encoder.encoders0.0.self_attn.linear_q_k_v.weight"));

  // It builds a working model.
  let sanitized = sanitize(merged).unwrap();
  let model = SenseVoiceModel::from_weights(
    tiny_config(),
    sanitized,
    SenseVoiceTokenizer::id_join(),
    None,
  )
  .unwrap();
  let wav = ramp_waveform(3000);
  assert_eq!(
    CtcModel::logits(&model, &wav).unwrap().shape()[1],
    V as usize
  );
}

#[test]
fn shard_walk_no_safetensors_errors() {
  let dir = temp_dir("no_shards");
  assert!(matches!(
    load_all_safetensors(&dir),
    Err(Error::MissingKey(_))
  ));
}

#[test]
fn shard_walk_duplicate_key_across_shards_errors() {
  // Two shards defining the same tensor key fail closed (LayerKeyed wrapping a
  // KeyCollision) rather than silently overwriting.
  let dir = temp_dir("dup_shards");
  let mut a: HashMap<String, Array> = HashMap::new();
  a.insert("ctc_lo.weight".to_string(), filled(V, H, |_, _| 1.0));
  let mut b: HashMap<String, Array> = HashMap::new();
  b.insert("ctc_lo.weight".to_string(), filled(V, H, |_, _| 2.0));
  crate::io::save_safetensors(&dir.join("a.safetensors"), &a).unwrap();
  crate::io::save_safetensors(&dir.join("b.safetensors"), &b).unwrap();
  assert!(matches!(
    load_all_safetensors(&dir),
    Err(Error::LayerKeyed(_))
  ));
}

// ───────────────────────────── am.mvn asset loading ─────────────────────────────

/// A minimal `am.mvn` Kaldi-MVN text fixture with `n` per-dim shift / rescale
/// values (the `<AddShift>` means and the `<Rescale>` inverse-stddev).
fn am_mvn_text(means: &[f32], istd: &[f32]) -> String {
  let m: Vec<String> = means.iter().map(|v| format!("{v}")).collect();
  let s: Vec<String> = istd.iter().map(|v| format!("{v}")).collect();
  format!(
    "<Nnet>\n<AddShift> {n} {n}\n<LearnRateCoef> 0 [ {means} ]\n\
     <Rescale> {n} {n}\n<LearnRateCoef> 0 [ {istd} ]\n</Nnet>\n",
    n = means.len(),
    means = m.join(" "),
    istd = s.join(" "),
  )
}

#[test]
fn load_cmvn_parses_am_mvn() {
  // An am.mvn in the dir -> the parsed (means, istd) pair (D-wide for the tiny
  // config). The config-fallback is NOT consulted when am.mvn is present.
  let dir = temp_dir("am_mvn");
  let means: Vec<f32> = (0..D).map(|i| -(i as f32)).collect();
  let istd: Vec<f32> = (0..D).map(|i| 1.0 + i as f32).collect();
  fs::write(dir.join("am.mvn"), am_mvn_text(&means, &istd)).unwrap();

  let (m, s) = load_cmvn(&dir, &tiny_config()).unwrap().unwrap();
  assert_eq!(m.shape(), vec![D as usize]);
  assert_eq!(s.shape(), vec![D as usize]);
  let mut m = m;
  let mut s = s;
  assert_eq!(m.to_vec::<f32>().unwrap(), means);
  assert_eq!(s.to_vec::<f32>().unwrap(), istd);
}

#[test]
fn load_cmvn_falls_back_to_config_when_no_am_mvn() {
  // No am.mvn but an in-config CMVN pair -> the config fallback.
  let dir = temp_dir("cmvn_config");
  let means: Vec<f32> = (0..D).map(|i| i as f32).collect();
  let istd: Vec<f32> = (0..D).map(|_| 2.0).collect();
  let json = format!(
    r#"{{ "model_type": "sensevoice", "vocab_size": {V}, "input_size": {D},
         "cmvn_means": {means:?}, "cmvn_istd": {istd:?},
         "encoder_conf": {{ "output_size": {H}, "attention_heads": 1, "num_blocks": 1, "tp_blocks": 0 }},
         "frontend_conf": {{ "n_mels": {D}, "lfr_m": 1, "lfr_n": 1 }} }}"#
  );
  let config: Config = serde_json::from_str(&json).unwrap();
  config.validate().unwrap();

  let (m, _s) = load_cmvn(&dir, &config).unwrap().unwrap();
  let mut m = m;
  assert_eq!(m.to_vec::<f32>().unwrap(), means);
}

#[test]
fn load_cmvn_none_when_no_am_mvn_and_no_config_cmvn() {
  let dir = temp_dir("cmvn_absent");
  assert!(load_cmvn(&dir, &tiny_config()).unwrap().is_none());
}

#[test]
fn load_cmvn_malformed_am_mvn_errors() {
  let dir = temp_dir("cmvn_malformed");
  fs::write(dir.join("am.mvn"), "garbage with no AddShift block").unwrap();
  assert!(matches!(
    load_cmvn(&dir, &tiny_config()),
    Err(Error::MalformedData(_))
  ));
}

#[test]
fn load_cmvn_rejects_length_mismatched_am_mvn() {
  // A length-1 CMVN against `input_size = D (8)` would broadcast silently with
  // the wrong stats; require the parsed lengths to equal `input_size` -> a typed
  // LengthMismatch.
  let dir = temp_dir("cmvn_len_mismatch");
  fs::write(dir.join("am.mvn"), am_mvn_text(&[0.0], &[1.0])).unwrap();
  assert!(matches!(
    load_cmvn(&dir, &tiny_config()),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn load_cmvn_rejects_length_mismatched_config_cmvn() {
  // The in-config CMVN fallback is length-checked too (a length-1 pair against
  // `input_size = D`). `Config::validate` would also catch this, but `load_cmvn`
  // re-asserts so it never builds a mis-sized CMVN on its own.
  let dir = temp_dir("cmvn_config_len");
  let json = format!(
    r#"{{ "model_type": "sensevoice", "vocab_size": {V}, "input_size": {D},
         "cmvn_means": [0.0], "cmvn_istd": [1.0],
         "encoder_conf": {{ "output_size": {H}, "attention_heads": 1, "num_blocks": 1, "tp_blocks": 0 }},
         "frontend_conf": {{ "n_mels": {D}, "lfr_m": 1, "lfr_n": 1 }} }}"#
  );
  // Build the config WITHOUT `validate()` (it would reject the length-1 pair
  // first); the test targets `load_cmvn`'s own re-assertion.
  let config: Config = serde_json::from_str(&json).unwrap();
  assert!(matches!(
    load_cmvn(&dir, &config),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn load_cmvn_rejects_oversized_am_mvn() {
  // The `am.mvn` body is read through the shared bounded reader (the 1 MiB
  // config-read convention); a file over the cap is rejected rather than read
  // into memory unbounded.
  let dir = temp_dir("cmvn_oversized");
  // A valid-looking header followed by > 1 MiB of padding inside the bracket ->
  // over the cap. The bounded read fails before the parse runs.
  let mut body = String::from("<AddShift> 1 1\n<LearnRateCoef> 0 [ 0.0 ");
  body.push_str(&" ".repeat(1024 * 1024 + 1));
  body.push_str("]\n<Rescale> 1 1\n<LearnRateCoef> 0 [ 1.0 ]\n");
  fs::write(dir.join("am.mvn"), &body).unwrap();
  assert!(
    load_cmvn(&dir, &tiny_config()).is_err(),
    "oversized am.mvn must be rejected by the bounded read"
  );
}

// ───────────────────────────── tokenizer asset loading ─────────────────────────────

#[test]
fn load_tokenizer_reads_tokens_json() {
  // A tokens.json piece list -> a TokenList tokenizer that decodes a known id
  // sequence (`"".join(pieces).replace("▁", " ").strip()`).
  let dir = temp_dir("tokens_json");
  let tokens = ["<blank>", "\u{2581}hello", "\u{2581}world", "!"];
  let body = serde_json::to_string(&tokens).unwrap();
  fs::write(dir.join("tokens.json"), body).unwrap();

  let tok = load_tokenizer(&dir).unwrap();
  // ids [1, 2, 3] -> "▁hello▁world!" -> " hello world!" -> "hello world!".
  assert_eq!(tok.decode(&[1, 2, 3]), "hello world!");
}

#[test]
fn load_tokenizer_id_join_when_no_assets() {
  let dir = temp_dir("tok_absent");
  let tok = load_tokenizer(&dir).unwrap();
  assert!(tok.is_id_join());
  assert_eq!(tok.decode(&[7, 8]), "7 8");
}

#[test]
fn load_tokenizer_rejects_malformed_tokens_json() {
  let dir = temp_dir("tok_malformed");
  fs::write(dir.join("tokens.json"), "{ not a list }").unwrap();
  assert!(matches!(load_tokenizer(&dir), Err(Error::Parse(_))));
}

// ───────────────────────────── bounded config read ─────────────────────────────

#[test]
fn load_rejects_oversized_config() {
  // A config.json over the shared 1 MiB bound is rejected by load_config's
  // bounded read (the established convention reused via read_bounded_config_file).
  let dir = temp_dir("big_config");
  // 1 MiB + 1 of whitespace padding inside a JSON object -> over the cap.
  let mut body = String::from("{\"model_type\":\"sensevoice\"");
  body.push_str(&" ".repeat(1024 * 1024 + 1));
  body.push('}');
  fs::write(dir.join("config.json"), &body).unwrap();
  // Write a shard so the failure is the config read, not a missing weights file.
  crate::io::save_safetensors(&dir.join("model.safetensors"), &tiny_weights()).unwrap();

  let err = SenseVoiceModel::load(dir.to_str().unwrap());
  assert!(err.is_err(), "oversized config must be rejected");
}

// ───────────────────────────── load() end-to-end + factory ─────────────────────────────

/// Write a full synthetic model directory: config.json + one safetensors shard
/// + am.mvn + tokens.json, returning the dir.
fn write_model_dir(name: &str, model_type: &str) -> PathBuf {
  let dir = temp_dir(name);
  // config.json with the requested model_type.
  let json = tiny_config_json().replace("\"sensevoice\"", &format!("\"{model_type}\""));
  fs::write(dir.join("config.json"), json).unwrap();
  // one weights shard (pre-sanitize layout — load() runs sanitize).
  crate::io::save_safetensors(&dir.join("model.safetensors"), &tiny_weights_raw()).unwrap();
  // am.mvn (D-wide, identity-ish: shift 0, rescale 1).
  let means: Vec<f32> = (0..D).map(|_| 0.0).collect();
  let istd: Vec<f32> = (0..D).map(|_| 1.0).collect();
  fs::write(dir.join("am.mvn"), am_mvn_text(&means, &istd)).unwrap();
  // tokens.json.
  let tokens = ["<blank>", "\u{2581}ok"];
  fs::write(
    dir.join("tokens.json"),
    serde_json::to_string(&tokens).unwrap(),
  )
  .unwrap();
  dir
}

#[test]
fn model_load_end_to_end_transcribes() {
  // A full synthetic model dir -> SenseVoiceModel::load -> a working transcriber
  // with the CMVN + tokenizer assets wired in.
  let dir = write_model_dir("load_e2e", "sensevoice");
  let model = SenseVoiceModel::load(dir.to_str().unwrap()).unwrap();
  // The tokenizer asset was loaded (a TokenList, not id-join).
  assert!(!model.tokenizer_ref().is_id_join());

  let wav = ramp_waveform(4000);
  let out = model
    .transcribe(&wav, &TranscribeOptions::default())
    .unwrap();
  // One full-utterance segment; the rich path also yields all three tags.
  assert_eq!(out.segments_slice().len(), 1);
  let rich = model.transcribe_rich(&wav, "auto", false).unwrap();
  assert!(!rich.rich().language().is_empty());
  assert!(!rich.rich().emotion().is_empty());
  assert!(!rich.rich().event().is_empty());
}

#[test]
fn dense_checkpoint_with_stale_quant_block_loads_dense() {
  // A DENSE checkpoint (no `.scales`) whose config.json carries a malformed /
  // stale `quantization` block must load as dense: `load()` runs the
  // `has_relevant_scales` pre-scan BEFORE `apply_quantization`, so the stale
  // block is never parsed (it would otherwise be a load-failing OutOfRange /
  // Parse). The model builds and transcribes normally.
  let dir = temp_dir("stale_quant_dense");
  // A config.json with a non-object `quantization` block (which
  // `apply_quantization` rejects with OutOfRange if it ever parses it).
  let json = tiny_config_json().replace(
    "\"model_type\": \"sensevoice\",",
    "\"model_type\": \"sensevoice\",\n      \"quantization\": \"stale-garbage-block\",",
  );
  // Sanity: the injection actually landed (otherwise the test proves nothing).
  assert!(json.contains("\"quantization\": \"stale-garbage-block\""));
  fs::write(dir.join("config.json"), json).unwrap();
  // A purely DENSE weights shard (no `.scales` anywhere), pre-sanitize layout.
  crate::io::save_safetensors(&dir.join("model.safetensors"), &tiny_weights_raw()).unwrap();

  // Loads dense (the stale block is gated out), not a parse / OutOfRange error.
  let model = SenseVoiceModel::load(dir.to_str().unwrap())
    .expect("dense checkpoint must load despite the stale quantization block");
  // The head is dense (the stale block never produced a quantization scheme).
  let wav = ramp_waveform(3500);
  let logits = CtcModel::logits(&model, &wav).unwrap();
  assert_eq!(logits.shape().len(), 2);
  assert_eq!(logits.shape()[1], V as usize);
}

#[test]
fn factory_load_dispatches_sensevoice_to_box_transcribe() {
  // The factory load() reads config, accepts model_type "sensevoice", and
  // returns a Box<dyn Transcribe> producing text + a segment.
  let dir = write_model_dir("factory_ok", "sensevoice");
  let model: Box<dyn Transcribe> = load(dir.to_str().unwrap()).unwrap();
  let wav = ramp_waveform(3500);
  let out = model
    .transcribe(&wav, &TranscribeOptions::default())
    .unwrap();
  assert_eq!(out.segments_slice().len(), 1);
  assert_eq!(out.segments_slice()[0].text(), out.text());
}

#[test]
fn factory_load_rejects_wrong_model_type() {
  // A non-sensevoice model_type is rejected with a typed UnknownEnumValue.
  let dir = write_model_dir("factory_wrong", "whisper");
  assert!(matches!(
    load(dir.to_str().unwrap()),
    Err(Error::UnknownEnumValue(_))
  ));
}

#[test]
fn model_load_missing_directory_errors() {
  let missing = std::env::temp_dir().join(format!(
    "mlxrs_sensevoice_loader_missing_{}",
    std::process::id()
  ));
  let _ = fs::remove_dir_all(&missing);
  assert!(SenseVoiceModel::load(missing.to_str().unwrap()).is_err());
}
