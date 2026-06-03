//! Tests for the Qwen3 forced aligner.
//!
//! Coverage:
//!
//! - [`fix_timestamp`] — independent hand-computed LIS-repair oracles (a
//!   monotone passthrough, a short anomalous run with the nearest-neighbor
//!   tie-break, and a long run with linear interpolation), each verified
//!   against a value worked out by hand, within `1e-9`.
//! - The timestamp **decode** — a hand-built `(1, L, classify_num)` logits
//!   tensor whose argmax class at each `<timestamp>` input position is a known
//!   value; the decoded spans must equal `class * timestamp_segment_time /
//!   1000` seconds, with the monotonicity repair applied across the markers.
//! - The structural forward — a tiny aligner built from synthetic weights
//!   produces `(B, L, classify_num)` logits; `build_input_ids` lays out the
//!   audio markers + interleaved `<timestamp>` pairs; `align` runs end to end
//!   to per-word spans.
//! - Config validation — count caps, token-id sign, and a non-finite /
//!   non-positive segment quantum map to typed errors.

use std::collections::HashMap;

use super::*;
use crate::{
  array::Array,
  audio::stt::{
    model::{AlignOptions, ForcedAligner as ForcedAlignerTrait},
    models::qwen3_asr::Qwen3AsrTextConfig,
  },
  error::Error,
  tokenizer::Tokenizer,
};

// ════════════════════════════ fix_timestamp oracles ════════════════════════

fn assert_fixed(got: &[i64], want: &[i64]) {
  assert_eq!(got, want, "fix_timestamp mismatch");
}

#[test]
fn fix_timestamp_monotone_is_passthrough() {
  // An already non-decreasing sequence is its own LIS — nothing is repaired.
  let data = [0.0, 80.0, 160.0, 240.0, 240.0];
  assert_fixed(&fix_timestamp(&data), &[0, 80, 160, 240, 240]);
}

#[test]
fn fix_timestamp_empty_is_empty() {
  assert!(fix_timestamp(&[]).is_empty());
}

#[test]
fn fix_timestamp_short_anomaly_uses_nearest_neighbor() {
  // data = [0, 80, 40, 160]. The non-strict LIS is [0, 80, 160] (indices
  // 0,1,3); index 2 (value 40) is the lone anomaly (run length 1 <= 2).
  // Neighbors: left anchor = 80 (index 1), right anchor = 160 (index 3).
  // Tie-break for k=2: dist_left = (k+1)-i = 3-2 = 1, dist_right = j-k = 3-2 = 1
  // → 1 <= 1 picks the LEFT value 80. Result: [0, 80, 80, 160].
  let data = [0.0, 80.0, 40.0, 160.0];
  assert_fixed(&fix_timestamp(&data), &[0, 80, 80, 160]);
}

#[test]
fn fix_timestamp_short_anomaly_picks_right_when_closer() {
  // data = [0, 160, 80, 240, 320]. Non-strict LIS = [0, 160, 240, 320]
  // (indices 0,1,3,4); index 2 (value 80) is the lone anomaly. Run [2,3):
  // i=2, j=3. left anchor = result[1] = 160, right anchor = result[3] = 240.
  // k=2: dist_left = (2+1)-2 = 1, dist_right = 3-2 = 1 → tie → LEFT (160).
  // (Confirms the documented `<=` tie-break: equal distances pick left.)
  let data = [0.0, 160.0, 80.0, 240.0, 320.0];
  assert_fixed(&fix_timestamp(&data), &[0, 160, 160, 240, 320]);
}

#[test]
fn fix_timestamp_long_run_linear_interpolation() {
  // data = [0, 1000, 10, 20, 30, 400]. The non-strict LIS keeping the most
  // points is [0, 10, 20, 30, 400] (indices 0,2,3,4,5, length 5); index 1
  // (value 1000) is a lone anomaly though — run length 1, NOT > 2.
  //
  // To force the long-run (interpolation) branch we need a run of >= 3
  // anomalies. data = [0, 5, 6, 7, 100] with a leading large value:
  // [100, 0, 5, 6, 7]? Build it so the LIS brackets a 3-long gap.
  //
  // data = [0, 99, 99, 99, 40]: LIS (non-strict) = [0, 40] (indices 0,4) OR
  // [0,99,99,99] (indices 0,1,2,3, length 4) — the longer one wins, so the
  // anomaly is index 4 alone. Not a long run either.
  //
  // Use a clean bracketed gap: data = [0, 7, 7, 7, 40]. Hmm still LIS picks
  // the plateau. We want the *bracketing* anchors normal and the middle
  // abnormal. data = [0, 50, 60, 70, 8] gives LIS [0,50,60,70] and a lone
  // tail. Instead force a dip: data = [0, 50, 9, 9, 9, 80].
  //   Non-strict LIS keeping the most points: [0, 9, 9, 9, 80] (indices
  //   0,2,3,4,5, length 5) → index 1 (value 50) is the lone anomaly. Still
  //   short.
  //
  // The reliable long-run construction: a single big spike-down run of 3
  // between two anchors that the LIS keeps as the anchors. data =
  // [0, 100, 1, 1, 1, 100]:
  //   Candidate LIS [0,1,1,1,100] (indices 0,2,3,4,5) length 5 vs
  //   [0,100,100] (indices 0,1,5) length 3 → the length-5 wins, anomaly =
  //   index 1 (lone). Argh — the plateau is always kept.
  //
  // Break the plateau so the dip is genuinely non-monotone *internally*:
  // data = [0, 100, 3, 2, 1, 100]. Non-strict LIS:
  //   [0, 3, ...]? 3,2,1 is decreasing, so from index 2 the best is length 2
  //   ([0,3] or [0,2] or [0,1], then +100). Compare:
  //     [0, 100, 100] (idx 0,1,5) = length 3
  //     [0, 3, 100]   (idx 0,2,5) = length 3
  //     [0, 2, 100]   (idx 0,3,5) = length 3
  //     [0, 1, 100]   (idx 0,4,5) = length 3
  //   `dp.index(max)` picks the FIRST index achieving the max length. The max
  //   dp value is 3, first achieved at index 1 (the [0,100,...] chain ending
  //   at 100? no — dp[i] is the LIS *ending at i*). dp ending at index 1
  //   (value 100) = 2 ([0,100]); dp[5] = 3. So max_length = 3, first index
  //   with dp==3 is index 5. Backtrack from 5: parent[5] is the j<5 with
  //   data[j]<=100 and best dp — j=1 (dp 2) is the first such with dp 2, so
  //   parent[5]=1, parent[1]=0. LIS indices = {0,1,5}. Normal = {0,1,5};
  //   anomalies = indices 2,3,4 (a run of 3 > 2 → LINEAR INTERP).
  //   left = result[1] = 100, right = result[5] = 100, count = 3,
  //   step = (100-100)/4 = 0 → all three become 100.
  //   Result = [0, 100, 100, 100, 100, 100].
  let data = [0.0, 100.0, 3.0, 2.0, 1.0, 100.0];
  assert_fixed(&fix_timestamp(&data), &[0, 100, 100, 100, 100, 100]);
}

#[test]
fn fix_timestamp_long_run_interpolates_between_distinct_anchors() {
  // Distinct bracketing anchors so the interpolation step is non-zero.
  // data = [0, 40, 3, 2, 1, 80]. By the same reasoning as above the LIS is
  // {0,1,5} (dp max 3 first reached at index 5, backtracking to 1 then 0),
  // anomalies = {2,3,4} (run of 3 → interp). left = result[1] = 40, right =
  // result[5] = 80, count = 3, step = (80-40)/4 = 10 →
  //   index 2 → 40 + 10*1 = 50
  //   index 3 → 40 + 10*2 = 60
  //   index 4 → 40 + 10*3 = 70
  // Result = [0, 40, 50, 60, 70, 80].
  let data = [0.0, 40.0, 3.0, 2.0, 1.0, 80.0];
  assert_fixed(&fix_timestamp(&data), &[0, 40, 50, 60, 70, 80]);
}

/// Brute-force O(n^2) reference of the whole `fix_timestamp` repair, mirroring
/// `ForceAlignProcessor.fix_timestamp` line for line. Used as an independent
/// oracle for the O(n log n) implementation on long, noisy inputs.
fn fix_timestamp_reference(data: &[f64]) -> Vec<i64> {
  let n = data.len();
  if n == 0 {
    return Vec::new();
  }
  let mut dp = vec![1usize; n];
  let mut parent = vec![usize::MAX; n];
  for i in 1..n {
    for j in 0..i {
      if data[j] <= data[i] && dp[j] + 1 > dp[i] {
        dp[i] = dp[j] + 1;
        parent[i] = j;
      }
    }
  }
  let max_length = *dp.iter().max().unwrap();
  let max_idx = dp.iter().position(|&d| d == max_length).unwrap();
  let mut is_normal = vec![false; n];
  let mut idx = max_idx;
  loop {
    is_normal[idx] = true;
    if parent[idx] == usize::MAX {
      break;
    }
    idx = parent[idx];
  }
  let mut result = data.to_vec();
  let mut i = 0;
  while i < n {
    if !is_normal[i] {
      let mut j = i;
      while j < n && !is_normal[j] {
        j += 1;
      }
      let anomaly_count = j - i;
      let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
      let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);
      if anomaly_count <= 2 {
        for (offset, slot) in result[i..j].iter_mut().enumerate() {
          let k = i + offset;
          *slot = match (left_val, right_val) {
            (None, Some(r)) => r,
            (Some(l), None) => l,
            (Some(l), Some(r)) => {
              if (k + 1) - i <= j - k {
                l
              } else {
                r
              }
            }
            (None, None) => *slot,
          };
        }
      } else {
        match (left_val, right_val) {
          (Some(l), Some(r)) => {
            let step = (r - l) / (anomaly_count as f64 + 1.0);
            for (offset, slot) in result[i..j].iter_mut().enumerate() {
              *slot = l + step * (offset as f64 + 1.0);
            }
          }
          (Some(l), None) => result[i..j].iter_mut().for_each(|s| *s = l),
          (None, Some(r)) => result[i..j].iter_mut().for_each(|s| *s = r),
          (None, None) => {}
        }
      }
      i = j;
    } else {
      i += 1;
    }
  }
  result.into_iter().map(|v| v as i64).collect()
}

#[test]
fn fix_timestamp_long_monotone_with_noise_matches_brute_force() {
  // A long predominantly-monotone marker sequence with periodic dips/spikes:
  // the case the O(n log n) reconstruction must handle identically to the
  // reference's O(n^2) DP. 4000 markers (= 2000 words), value = 50*i ms with a
  // deterministic perturbation every few positions (both dips and spikes,
  // including ties and equal-distance anomalies).
  let n = 4000usize;
  let mut data = vec![0.0f64; n];
  for (i, slot) in data.iter_mut().enumerate() {
    let base = 50.0 * i as f64;
    *slot = match i % 7 {
      0 => base - 130.0, // dip below the local trend
      3 => base + 90.0,  // spike above
      5 => base,         // on-trend (creates ties with neighbors' repairs)
      _ => base,
    };
  }
  // Add a longer (>2) contiguous anomalous run to exercise the interpolation
  // branch, bracketed by on-trend anchors.
  data[1000..1004].fill(17.0); // a flat low plateau dropped into the middle
  let got = fix_timestamp(&data);
  let want = fix_timestamp_reference(&data);
  assert_eq!(got.len(), n);
  assert_eq!(
    got, want,
    "O(n log n) fix_timestamp diverged from brute force"
  );
}

#[test]
fn fix_timestamp_lis_lengths_matches_brute_force_dp() {
  // The Fenwick-based LIS-length pass must reproduce the reference's per-index
  // non-decreasing-LIS lengths exactly (including on duplicate values).
  let data = [0.0, 100.0, 3.0, 2.0, 1.0, 100.0, 100.0, 50.0, 200.0, 200.0];
  let got = super::lis_lengths(&data);
  // Brute-force the same recurrence.
  let n = data.len();
  let mut want = vec![1usize; n];
  for i in 1..n {
    for j in 0..i {
      if data[j] <= data[i] && want[j] + 1 > want[i] {
        want[i] = want[j] + 1;
      }
    }
  }
  assert_eq!(got, want);
}

// ════════════════════════════ tiny aligner construction ════════════════════

/// Tiny decoder dims: hidden=4, head_dim=2, n_heads=2, n_kv_heads=1,
/// intermediate=6, vocab=40, 1 layer. The audio tower's `output_dim` is set to
/// this `hidden` so the splice (which replaces a hidden-width embedding row
/// with an audio row) is shape-valid.
const HIDDEN: i32 = 4;
const VOCAB: i32 = 40;
const CLASSIFY_NUM: i32 = 8;

/// The aligner config JSON wiring the tiny audio tower (output_dim == HIDDEN)
/// to the tiny decoder, with a small `classify_num` and the reference token
/// ids / segment quantum — in the released nested `thinker_config` shape.
fn tiny_aligner_config() -> ForcedAlignerConfig {
  let json = format!(
    r#"{{
      "model_type": "qwen3_asr",
      "thinker_config": {{
        "model_type": "qwen3_forced_aligner",
        "audio_config": {{
          "num_mel_bins": 8,
          "encoder_layers": 1,
          "encoder_attention_heads": 2,
          "encoder_ffn_dim": 8,
          "d_model": 4,
          "output_dim": {HIDDEN},
          "max_source_positions": 8,
          "n_window": 4,
          "n_window_infer": 8,
          "downsample_hidden_size": 2
        }},
        "text_config": {{
          "hidden_size": {HIDDEN}, "head_dim": 2, "num_attention_heads": 2,
          "num_key_value_heads": 1, "num_hidden_layers": 1, "intermediate_size": 6,
          "vocab_size": {VOCAB}, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
          "tie_word_embeddings": true
        }},
        "classify_num": {CLASSIFY_NUM},
        "audio_token_id": 30,
        "audio_start_token_id": 31,
        "audio_end_token_id": 32,
        "timestamp_token_id": 33,
        "timestamp_segment_time": 80.0
      }}
    }}"#
  );
  ForcedAlignerConfig::from_json(&json).expect("tiny aligner config must validate")
}

/// Deterministic small constant tensor `(shape)` filled with `val`.
fn filled(shape: &[i32], val: f32) -> Array {
  Array::full::<f32>(&shape.to_vec(), val).unwrap()
}

/// The audio-tower weight map (channels-last conv weights, every named
/// projection / norm) for the tiny audio config — mirrors the audio module's
/// own tiny-weights helper.
fn tiny_audio_weights(cfg: &super::super::config::AudioEncoderConfig) -> HashMap<String, Array> {
  let d = cfg.d_model;
  let h = cfg.downsample_hidden_size;
  let ffn = cfg.encoder_ffn_dim;
  let out = cfg.output_dim;
  let conv_out_in = cfg.conv_out_in_features().unwrap();
  let mut w: HashMap<String, Array> = HashMap::new();

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
  w.insert("layers.0.fc1.weight".into(), filled(&[ffn, d], 0.1));
  w.insert("layers.0.fc1.bias".into(), filled(&[ffn], 0.0));
  w.insert("layers.0.fc2.weight".into(), filled(&[d, ffn], 0.1));
  w.insert("layers.0.fc2.bias".into(), filled(&[d], 0.0));
  w.insert("layers.0.final_layer_norm.weight".into(), filled(&[d], 1.0));
  w.insert("layers.0.final_layer_norm.bias".into(), filled(&[d], 0.0));

  w.insert("ln_post.weight".into(), filled(&[d], 1.0));
  w.insert("ln_post.bias".into(), filled(&[d], 0.0));
  w.insert("proj1.weight".into(), filled(&[d, d], 0.1));
  w.insert("proj1.bias".into(), filled(&[d], 0.0));
  w.insert("proj2.weight".into(), filled(&[out, d], 0.1));
  w.insert("proj2.bias".into(), filled(&[out], 0.0));
  w
}

/// The decoder (`model.*`) + timestamp-head (`lm_head.weight`) weight map for
/// the tiny text config. Small constant fills keep the forward stable; the
/// tests assert shapes / decode, not magnitudes.
fn tiny_decoder_weights(cfg: &Qwen3AsrTextConfig) -> HashMap<String, Array> {
  let hidden = cfg.hidden_size;
  let head_dim = cfg.head_dim;
  let n_heads = cfg.num_attention_heads;
  let n_kv = cfg.num_key_value_heads;
  let inter = cfg.intermediate_size;
  let vocab = cfg.vocab_size;
  let mut m: HashMap<String, Array> = HashMap::new();

  m.insert(
    "model.embed_tokens.weight".into(),
    filled(&[vocab, hidden], 0.02),
  );
  m.insert("model.norm.weight".into(), filled(&[hidden], 1.0));
  let p = "model.layers.0";
  m.insert(
    format!("{p}.self_attn.q_proj.weight"),
    filled(&[n_heads * head_dim, hidden], 0.05),
  );
  m.insert(
    format!("{p}.self_attn.k_proj.weight"),
    filled(&[n_kv * head_dim, hidden], 0.05),
  );
  m.insert(
    format!("{p}.self_attn.v_proj.weight"),
    filled(&[n_kv * head_dim, hidden], 0.05),
  );
  m.insert(
    format!("{p}.self_attn.o_proj.weight"),
    filled(&[hidden, n_heads * head_dim], 0.05),
  );
  m.insert(
    format!("{p}.self_attn.q_norm.weight"),
    filled(&[head_dim], 1.0),
  );
  m.insert(
    format!("{p}.self_attn.k_norm.weight"),
    filled(&[head_dim], 1.0),
  );
  m.insert(
    format!("{p}.mlp.gate_proj.weight"),
    filled(&[inter, hidden], 0.05),
  );
  m.insert(
    format!("{p}.mlp.up_proj.weight"),
    filled(&[inter, hidden], 0.05),
  );
  m.insert(
    format!("{p}.mlp.down_proj.weight"),
    filled(&[hidden, inter], 0.05),
  );
  m.insert(
    format!("{p}.input_layernorm.weight"),
    filled(&[hidden], 1.0),
  );
  m.insert(
    format!("{p}.post_attention_layernorm.weight"),
    filled(&[hidden], 1.0),
  );
  // The timestamp head: Linear(hidden -> classify_num), weight (classify_num, hidden).
  m.insert(
    "lm_head.weight".into(),
    filled(&[CLASSIFY_NUM, hidden], 0.1),
  );
  m
}

fn tiny_aligner() -> ForcedAligner {
  let cfg = tiny_aligner_config();
  let audio = tiny_audio_weights(&cfg.audio_config);
  let dec = tiny_decoder_weights(&cfg.text_config);
  ForcedAligner::from_weights(cfg, audio, dec).expect("tiny aligner must build")
}

// ════════════════════════════ from_weights ════════════════════════════

#[test]
fn from_weights_builds_and_reports_config() {
  let a = tiny_aligner();
  assert_eq!(a.config().classify_num, CLASSIFY_NUM);
  assert_eq!(a.config().timestamp_token_id, 33);
  assert_eq!(a.decoder().num_layers(), 1);
  assert_eq!(a.audio_tower().config().output_dim, HIDDEN);
}

#[test]
fn from_weights_missing_head_is_typed_error() {
  let cfg = tiny_aligner_config();
  let audio = tiny_audio_weights(&cfg.audio_config);
  let mut dec = tiny_decoder_weights(&cfg.text_config);
  assert!(dec.remove("lm_head.weight").is_some());
  assert!(matches!(
    ForcedAligner::from_weights(cfg, audio, dec),
    Err(Error::MissingKey(_))
  ));
}

#[test]
fn from_weights_wrong_head_shape_is_typed_error() {
  let cfg = tiny_aligner_config();
  let audio = tiny_audio_weights(&cfg.audio_config);
  let mut dec = tiny_decoder_weights(&cfg.text_config);
  // A head of the wrong width (classify_num + 1) must be rejected.
  dec.insert(
    "lm_head.weight".into(),
    filled(&[CLASSIFY_NUM + 1, HIDDEN], 0.1),
  );
  assert!(matches!(
    ForcedAligner::from_weights(cfg, audio, dec),
    Err(Error::ShapePairMismatch(_))
  ));
}

// ════════════════════════════ build_input_ids ════════════════════════════

#[test]
fn build_input_ids_lays_out_markers_and_timestamp_pairs() {
  let a = tiny_aligner();
  // Two words: token ids [10, 11] and [12]. num_audio_tokens = 3.
  let transcript = [
    AlignWord::new("ab", vec![10, 11]),
    AlignWord::new("c", vec![12]),
  ];
  let mut ids = a.build_input_ids(&transcript, 3).unwrap();
  // Expected: [start=31, pad=30,30,30, end=32, 10, 11, ts=33,33, 12, ts=33,33].
  let want = vec![31, 30, 30, 30, 32, 10, 11, 33, 33, 12, 33, 33];
  assert_eq!(ids.shape(), vec![1, want.len()]);
  assert_eq!(ids.to_vec::<i32>().unwrap(), want);
}

// ════════════════════════════ decode oracle ════════════════════════════

/// Build a `(1, L, classify_num)` logits tensor whose argmax over the last
/// axis at position `p` is `classes[p]` (a large value placed at that class,
/// the rest small + distinct so the argmax is unambiguous).
fn logits_with_argmax(classes: &[u32], classify_num: i32) -> Array {
  let l = classes.len();
  let c = classify_num as usize;
  let mut flat = vec![0.0f32; l * c];
  for (p, &cls) in classes.iter().enumerate() {
    for j in 0..c {
      // Distinct small ramp; spike the chosen class well above the rest.
      flat[p * c + j] = j as f32 * 0.001;
    }
    flat[p * c + cls as usize] = 100.0;
  }
  Array::from_slice::<f32>(&flat, &(1usize, l, c)).unwrap()
}

#[test]
fn decode_maps_timestamp_classes_to_second_spans() {
  let a = tiny_aligner();
  // input_ids: [start, pad*2, end, w0, ts, ts, w1, ts, ts] — two words, four
  // <timestamp> (id 33) positions. Non-timestamp positions get arbitrary ids.
  let ts = 33;
  let ids = vec![31, 30, 30, 32, 10, ts, ts, 12, ts, ts];
  let input_ids = Array::from_slice::<i32>(&ids, &(1usize, ids.len())).unwrap();

  // Argmax class at EVERY position; only the four ts-position classes are read.
  // Place classes so the four timestamp markers are [1, 2, 3, 4] (already
  // monotone → no repair). class * 80 ms → 80,160,240,320 ms → seconds
  // 0.08,0.16,0.24,0.32. Word0 = (0.08, 0.16), word1 = (0.24, 0.32).
  let classes = vec![0, 0, 0, 0, 0, 1, 2, 0, 3, 4];
  let logits = logits_with_argmax(&classes, CLASSIFY_NUM);

  let transcript = [
    AlignWord::new("hello", vec![10]),
    AlignWord::new("world", vec![12]),
  ];
  let align = a
    .decode_alignment(&input_ids, &logits, &transcript, Some("english".into()))
    .unwrap();

  assert_eq!(align.language(), Some("english"));
  let spans = align.spans();
  assert_eq!(spans.len(), 2);
  assert_eq!(spans[0].text(), "hello");
  assert!((spans[0].start_time() - 0.08).abs() < 1e-4);
  assert!((spans[0].end_time() - 0.16).abs() < 1e-4);
  assert_eq!(spans[1].text(), "world");
  assert!((spans[1].start_time() - 0.24).abs() < 1e-4);
  assert!((spans[1].end_time() - 0.32).abs() < 1e-4);
}

#[test]
fn decode_applies_monotonicity_repair_across_markers() {
  let a = tiny_aligner();
  let ts = 33;
  // Three words → six <timestamp> markers.
  let ids = vec![10, ts, ts, 11, ts, ts, 12, ts, ts];
  let input_ids = Array::from_slice::<i32>(&ids, &(1usize, ids.len())).unwrap();

  // Marker classes (in ts-position order): [1, 2, 1, 3, 4, 5].
  // class*80 ms = [80, 160, 80, 240, 320, 400]. The lone dip at marker index 2
  // (value 80) is repaired: non-strict LIS = [80,160,240,320,400] (indices
  // 0,1,3,4,5) → index 2 anomalous, run length 1, neighbors left=160 right=240,
  // dist_left=(2+1)-2=1 dist_right=3-2=1 → left=160. Fixed ms =
  // [80,160,160,240,320,400] → seconds [0.08,0.16,0.16,0.24,0.32,0.40].
  // word0=(0.08,0.16) word1=(0.16,0.24) word2=(0.32,0.40).
  let classes = vec![0, 1, 2, 0, 1, 3, 0, 4, 5];
  let logits = logits_with_argmax(&classes, CLASSIFY_NUM);

  let transcript = [
    AlignWord::new("a", vec![10]),
    AlignWord::new("b", vec![11]),
    AlignWord::new("c", vec![12]),
  ];
  let align = a
    .decode_alignment(&input_ids, &logits, &transcript, None)
    .unwrap();
  let s = align.spans();
  assert_eq!(s.len(), 3);
  let got: Vec<(f64, f64)> = s.iter().map(|x| (x.start_time(), x.end_time())).collect();
  let want = [(0.08, 0.16), (0.16, 0.24), (0.32, 0.40)];
  for (g, w) in got.iter().zip(want) {
    assert!(
      (g.0 - w.0).abs() < 1e-4 && (g.1 - w.1).abs() < 1e-4,
      "got {g:?} want {w:?}"
    );
  }
}

#[test]
fn decode_rejects_marker_count_mismatch() {
  let a = tiny_aligner();
  let ts = 33;
  // Only TWO markers but the transcript has two words (needs four).
  let ids = vec![10, ts, ts, 11];
  let input_ids = Array::from_slice::<i32>(&ids, &(1usize, ids.len())).unwrap();
  let classes = vec![0, 1, 2, 0];
  let logits = logits_with_argmax(&classes, CLASSIFY_NUM);
  let transcript = [AlignWord::new("a", vec![10]), AlignWord::new("b", vec![11])];
  assert!(matches!(
    a.decode_alignment(&input_ids, &logits, &transcript, None),
    Err(Error::LengthMismatch(_))
  ));
}

// ─────────────── decode_alignment adversarial shape mismatches (Finding 1) ───

/// Build a `(1, l, classify_num)` logits tensor for an explicit sequence length
/// `l` (every position's argmax class is `0`; only the shape matters here).
fn flat_logits(l: usize, classify_num: i32) -> Array {
  let c = classify_num as usize;
  let flat = vec![0.0f32; l * c];
  Array::from_slice::<f32>(&flat, &(1usize, l, c)).unwrap()
}

#[test]
fn decode_rejects_batched_input_ids_no_cross_row_read() {
  // The exact Finding-1 cross-row case: a rank-2 `input_ids` of shape (2, 3)
  // (6 flattened ids) paired with a `logits` whose sequence (6) is longer than
  // input_ids ROW 0 (3) but equals the flattened total (6). The old decode
  // flattened input_ids and sliced `[..6]`, reading across into batch row 1 and
  // returning plausible-but-wrong spans. It must now be rejected (batch != 1),
  // never a cross-row read.
  let a = tiny_aligner();
  let ts = 33;
  // Two rows of length 3; the flattened length (6) is what the old code sliced.
  let ids = vec![10, ts, ts, 11, ts, ts];
  let input_ids = Array::from_slice::<i32>(&ids, &(2usize, 3usize)).unwrap();
  let logits = flat_logits(6, CLASSIFY_NUM); // seq 6 > row-0 len 3
  let transcript = [AlignWord::new("a", vec![10])];
  let err = a
    .decode_alignment(&input_ids, &logits, &transcript, None)
    .expect_err("batched input_ids in decode must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn decode_rejects_input_ids_seq_shorter_than_logits_seq() {
  // Batch-1 mismatch: input_ids row length (3) disagrees with the logits seq
  // (5). The decode must reject the shape relation (a ShapePairMismatch), rather
  // than read past the input_ids row.
  let a = tiny_aligner();
  let ts = 33;
  let ids = vec![10, ts, ts];
  let input_ids = Array::from_slice::<i32>(&ids, &(1usize, 3usize)).unwrap();
  let logits = flat_logits(5, CLASSIFY_NUM);
  let transcript = [AlignWord::new("a", vec![10])];
  let err = a
    .decode_alignment(&input_ids, &logits, &transcript, None)
    .expect_err("seq mismatch must be rejected");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

#[test]
fn decode_rejects_logits_last_dim_not_classify_num() {
  // The logits last dim must equal classify_num: a head of the wrong width
  // would make the argmax class index a different label space. Reject it as a
  // ShapePairMismatch rather than decode plausible-but-wrong classes.
  let a = tiny_aligner();
  let ts = 33;
  let ids = vec![10, ts, ts];
  let input_ids = Array::from_slice::<i32>(&ids, &(1usize, 3usize)).unwrap();
  // seq matches (3) but the last dim is classify_num + 1.
  let logits = flat_logits(3, CLASSIFY_NUM + 1);
  let transcript = [AlignWord::new("a", vec![10])];
  let err = a
    .decode_alignment(&input_ids, &logits, &transcript, None)
    .expect_err("wrong classify_num must be rejected");
  assert!(matches!(err, Error::ShapePairMismatch(_)), "got {err:?}");
}

#[test]
fn decode_rejects_too_low_rank_logits() {
  // A rank-1 logits tensor must be a typed RankMismatch, never an argmax/axis
  // panic on a missing dimension.
  let a = tiny_aligner();
  let ids = vec![10, 33, 33];
  let input_ids = Array::from_slice::<i32>(&ids, &(1usize, 3usize)).unwrap();
  let logits = Array::full::<f32>(&[CLASSIFY_NUM], 0.0).unwrap(); // rank 1
  let transcript = [AlignWord::new("a", vec![10])];
  let err = a
    .decode_alignment(&input_ids, &logits, &transcript, None)
    .expect_err("rank-1 logits must be rejected");
  assert!(matches!(err, Error::RankMismatch(_)), "got {err:?}");
}

// ─────────────── pre-tokenized id range guards (Finding 2 — UB) ───

#[test]
fn build_input_ids_rejects_negative_token_id() {
  // A negative pre-tokenized id reaches MLX `take` in embed_tokens, whose gather
  // reads `id + vocab` (a wrong, possibly out-of-bounds row) — undefined
  // behavior. build_input_ids must reject it with a typed OutOfRange.
  let a = tiny_aligner();
  let transcript = [AlignWord::new("x", vec![-1])];
  let err = a
    .build_input_ids(&transcript, 3)
    .expect_err("negative token id must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn build_input_ids_rejects_token_id_at_or_above_vocab() {
  // An id == vocab_size (and one above) reads past the embedding table in the
  // unchecked MLX gather — out-of-bounds (UB). Reject both with OutOfRange.
  let a = tiny_aligner();
  for bad in [VOCAB, VOCAB + 100] {
    let transcript = [AlignWord::new("x", vec![bad])];
    let err = a
      .build_input_ids(&transcript, 3)
      .expect_err("out-of-range token id must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?} for {bad}");
  }
}

#[test]
fn build_input_ids_accepts_in_range_ids() {
  // The positive control: ids in [0, vocab) build fine (the guard is fail-fast,
  // not over-strict).
  let a = tiny_aligner();
  let transcript = [AlignWord::new("x", vec![0, VOCAB - 1, 10])];
  assert!(a.build_input_ids(&transcript, 3).is_ok());
}

#[test]
fn forward_rejects_negative_input_id_before_embed() {
  // The direct-forward path: an `input_ids` value of -1 must be rejected before
  // embed_tokens gathers (the unchecked `take` would otherwise read `id + vocab`
  // — UB), as a typed OutOfRange.
  let a = tiny_aligner();
  let feats = filled(&[1, 8, 8], 0.1);
  let n_audio = a.num_audio_tokens_for_test(&feats);
  // A correctly-sized id layout, then corrupt one non-marker position to -1.
  let transcript = [AlignWord::new("x", vec![10])];
  let mut ids_vec = a
    .build_input_ids(&transcript, n_audio)
    .unwrap()
    .to_vec::<i32>()
    .unwrap();
  *ids_vec.last_mut().unwrap() = -1;
  let l = ids_vec.len();
  let input_ids = Array::from_slice::<i32>(&ids_vec, &(1usize, l)).unwrap();
  let err = a
    .forward(&input_ids, &feats)
    .expect_err("negative input id must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn forward_rejects_out_of_range_input_id_before_embed() {
  // The direct-forward path: an `input_ids` value >= vocab reads past the
  // embedding table in the unchecked gather — UB. Reject it before embed.
  let a = tiny_aligner();
  let feats = filled(&[1, 8, 8], 0.1);
  let n_audio = a.num_audio_tokens_for_test(&feats);
  let transcript = [AlignWord::new("x", vec![10])];
  let mut ids_vec = a
    .build_input_ids(&transcript, n_audio)
    .unwrap()
    .to_vec::<i32>()
    .unwrap();
  *ids_vec.last_mut().unwrap() = VOCAB; // == vocab_size, out of [0, vocab)
  let l = ids_vec.len();
  let input_ids = Array::from_slice::<i32>(&ids_vec, &(1usize, l)).unwrap();
  let err = a
    .forward(&input_ids, &feats)
    .expect_err("out-of-range input id must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

// ════════════════════════════ forward shape ════════════════════════════

#[test]
fn forward_returns_classify_logits_shape() {
  let a = tiny_aligner();
  // input_features: (batch=1, n_mels=8, time=8) → the encoder downsamples; the
  // forward splices audio rows into the <audio_pad> positions and applies the
  // timestamp head. We assert the (B, L, classify_num) logits shape.
  let feats = filled(&[1, 8, 8], 0.1);
  // Build input_ids sized to the encoder output so the splice count matches.
  let n_audio = a.num_audio_tokens_for_test(&feats);
  let transcript = [AlignWord::new("x", vec![10]), AlignWord::new("y", vec![11])];
  let ids = a.build_input_ids(&transcript, n_audio).unwrap();
  let l = ids.shape()[1];

  let logits = a.forward(&ids, &feats).unwrap();
  assert_eq!(logits.shape(), vec![1, l, CLASSIFY_NUM as usize]);
}

#[test]
fn forward_rejects_too_few_audio_placeholders() {
  // One fewer `<audio_pad>` than the encoder emits: the splice cannot merge one
  // audio row per placeholder, so the count mismatch is a typed error rather
  // than a silently partial splice.
  let a = tiny_aligner();
  let feats = filled(&[1, 8, 8], 0.1);
  let n_audio = a.num_audio_tokens_for_test(&feats);
  assert!(
    n_audio >= 1,
    "tiny encoder must emit at least one audio row"
  );
  let transcript = [AlignWord::new("x", vec![10])];
  let ids = a.build_input_ids(&transcript, n_audio - 1).unwrap();
  let err = a
    .forward(&ids, &feats)
    .expect_err("too few audio placeholders must be rejected");
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err:?}");
}

#[test]
fn forward_rejects_too_many_audio_placeholders() {
  // One extra `<audio_pad>` than the encoder emits: the surplus placeholder
  // would be left as a normal audio_pad embedding (an unspliced audio position),
  // so the count mismatch is rejected.
  let a = tiny_aligner();
  let feats = filled(&[1, 8, 8], 0.1);
  let n_audio = a.num_audio_tokens_for_test(&feats);
  let transcript = [AlignWord::new("x", vec![10])];
  let ids = a.build_input_ids(&transcript, n_audio + 1).unwrap();
  let err = a
    .forward(&ids, &feats)
    .expect_err("too many audio placeholders must be rejected");
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err:?}");
}

#[test]
fn forward_rejects_batched_input_ids() {
  // The aligner is a single-utterance path. A `(2, L)` input_ids tensor paired
  // with a single `(1, n_mels, T)` audio feature tensor is rejected before any
  // embedding or splice, consistent with the audio tower's batch reject.
  let a = tiny_aligner();
  let feats = filled(&[1, 8, 8], 0.1);
  let n_audio = a.num_audio_tokens_for_test(&feats);
  let transcript = [AlignWord::new("x", vec![10])];
  // Build a correct single-row id layout, then stack it into a 2-row batch so
  // the per-row pad count is right but the batch dimension is 2.
  let mut row = a.build_input_ids(&transcript, n_audio).unwrap();
  let l = row.shape()[1];
  let flat = row.to_vec::<i32>().unwrap();
  let mut batched = Vec::with_capacity(flat.len() * 2);
  batched.extend_from_slice(&flat);
  batched.extend_from_slice(&flat);
  let ids = Array::from_slice::<i32>(&batched, &(2usize, l)).unwrap();

  let err = a
    .forward(&ids, &feats)
    .expect_err("batched input_ids must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

// ════════════════════════════ feature-length path ════════════════════════════

/// A real-window aligner config (`n_window = 50` → conv chunk = 100 mel frames,
/// `n_window_infer = 800`) with otherwise tiny dims. With the real chunk size,
/// the windowed encoder's per-chunk post-CNN frames sum to exactly
/// `feature_output_length(valid_len)` for any number of chunks, so the aligner's
/// `<audio_pad>` count (derived from that formula) matches the encoder rows —
/// the invariant a multi-window forward / align relies on. `max_source_positions
/// = 16` covers the 13 post-CNN frames a full 100-frame chunk produces.
fn real_window_aligner_config() -> ForcedAlignerConfig {
  let json = format!(
    r#"{{
      "model_type": "qwen3_asr",
      "thinker_config": {{
        "model_type": "qwen3_forced_aligner",
        "audio_config": {{
          "num_mel_bins": 8,
          "encoder_layers": 1,
          "encoder_attention_heads": 2,
          "encoder_ffn_dim": 8,
          "d_model": 4,
          "output_dim": {HIDDEN},
          "max_source_positions": 16,
          "n_window": 50,
          "n_window_infer": 800,
          "downsample_hidden_size": 2
        }},
        "text_config": {{
          "hidden_size": {HIDDEN}, "head_dim": 2, "num_attention_heads": 2,
          "num_key_value_heads": 1, "num_hidden_layers": 1, "intermediate_size": 6,
          "vocab_size": {VOCAB}, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
          "tie_word_embeddings": true
        }},
        "classify_num": {CLASSIFY_NUM},
        "audio_token_id": 30,
        "audio_start_token_id": 31,
        "audio_end_token_id": 32,
        "timestamp_token_id": 33,
        "timestamp_segment_time": 80.0
      }}
    }}"#
  );
  ForcedAlignerConfig::from_json(&json).expect("real-window aligner config must validate")
}

fn real_window_aligner() -> ForcedAligner {
  let cfg = real_window_aligner_config();
  let audio = tiny_audio_weights(&cfg.audio_config);
  let dec = tiny_decoder_weights(&cfg.text_config);
  ForcedAligner::from_weights(cfg, audio, dec).expect("real-window aligner must build")
}

#[test]
fn forward_multi_window_feature_length_produces_logits() {
  // 250 valid mel frames span 3 conv chunks (100, 100, 50) — the multi-window
  // audio path. The audio-token count (feature_output_length(250) = 33) matches
  // the windowed encoder's row count, so the splice succeeds and the timestamp
  // head produces the (B, L, classify_num) logits.
  let a = real_window_aligner();
  let feats = filled(&[1, 8, 250], 0.1);
  let n_audio = a.num_audio_tokens_for_test(&feats);
  assert_eq!(
    n_audio,
    usize::try_from(super::super::audio::AudioEncoder::feature_output_length(
      250
    ))
    .unwrap()
  );
  assert_eq!(n_audio, 33, "feature_output_length(250) = 33");
  let transcript = [AlignWord::new("x", vec![10]), AlignWord::new("y", vec![11])];
  let ids = a.build_input_ids(&transcript, n_audio).unwrap();
  let l = ids.shape()[1];
  let logits = a.forward(&ids, &feats).unwrap();
  assert_eq!(logits.shape(), vec![1, l, CLASSIFY_NUM as usize]);
}

#[test]
fn forward_multi_window_trims_padding_for_splice() {
  // Pad a 250-frame valid utterance into a 320-frame axis (3.2 chunks). The
  // audio-token count is taken from the valid 250, the windowed forward trims
  // the padding, and the spliced row count matches.
  let a = real_window_aligner();
  let feats = filled(&[1, 8, 320], 0.1);
  let n_audio = usize::try_from(super::super::audio::AudioEncoder::feature_output_length(
    250,
  ))
  .unwrap();
  let transcript = [AlignWord::new("x", vec![10])];
  let ids = a.build_input_ids(&transcript, n_audio).unwrap();
  let l = ids.shape()[1];
  let logits = a
    .forward_with_feature_length(&ids, &feats, Some(250))
    .unwrap();
  assert_eq!(logits.shape(), vec![1, l, CLASSIFY_NUM as usize]);
}

#[test]
fn forward_with_feature_length_trims_padding_for_splice() {
  // Pad an utterance to time=16 but mark only 8 frames valid (== one chunk).
  // The audio-token count is computed from the valid length, and the trimmed
  // forward produces matching rows, so the (B, L, classify_num) shape holds.
  let a = tiny_aligner();
  let feats = filled(&[1, 8, 16], 0.1);
  // valid length 8 → feature_output_length(8) audio tokens.
  let n_audio =
    usize::try_from(super::super::audio::AudioEncoder::feature_output_length(8)).unwrap();
  let transcript = [AlignWord::new("x", vec![10]), AlignWord::new("y", vec![11])];
  let ids = a.build_input_ids(&transcript, n_audio).unwrap();
  let l = ids.shape()[1];
  let logits = a
    .forward_with_feature_length(&ids, &feats, Some(8))
    .unwrap();
  assert_eq!(logits.shape(), vec![1, l, CLASSIFY_NUM as usize]);
}

// ═══════════════════════ splice gather-index overflow ═══════════════════════

#[test]
fn splice_gather_index_maps_pads_and_passthrough() {
  // The k-th audio-pad flat position maps to `seq_elems_i + k` (the k-th
  // appended audio row); every other position maps to its own flat embed row.
  // ids = [7, PAD, 7, PAD] with audio_token_id = 99, seq_elems_i = 4 → the two
  // pads (k = 0, 1) become rows 4 and 5, the non-pads stay at their indices.
  let ids = [7, 99, 7, 99];
  let gather = splice_gather_index(&ids, 99, 4).expect("normal index must build");
  assert_eq!(gather, vec![0, 4, 2, 5]);
}

#[test]
fn splice_gather_index_rejects_seq_plus_k_overflow() {
  // The soundness boundary: `seq_elems_i + k` must not wrap past `i32::MAX`
  // into a negative / out-of-range row (MLX `take` does not bound-check its
  // indices, so a wrapped index is an out-of-bounds row read — UB). With
  // seq_elems_i == i32::MAX, the first audio-pad position (k = 0) is in range
  // (it is exactly i32::MAX), but the SECOND pad (k = 1) would overflow → a
  // typed OutOfRange, not a panic and not a wrapped index.
  let ids = [99, 99]; // two audio-pad positions
  let err =
    splice_gather_index(&ids, 99, i32::MAX).expect_err("seq_elems_i + k overflow must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

#[test]
fn splice_gather_index_rejects_embed_index_overflow() {
  // A non-pad flat position maps to its own index `i`; an `i` past `i32::MAX`
  // is rejected (a typed OutOfRange) rather than truncated by an `as` cast.
  // Building an actual `(i32::MAX + 2)`-length slice is infeasible, so this is
  // covered by the checked-arithmetic boundary in the pad path above and the
  // `combined_rows` guard in `splice_audio`; here we assert the passthrough
  // path still produces the identity index for an in-range non-pad run.
  let ids = [1, 2, 3];
  let gather = splice_gather_index(&ids, 99, 3).expect("identity index must build");
  assert_eq!(gather, vec![0, 1, 2]);
}

// ════════════════════════════ align end-to-end ════════════════════════════

#[test]
fn align_pretokenized_produces_one_span_per_word() {
  let a = tiny_aligner();
  let feats = filled(&[1, 8, 8], 0.1);
  let transcript = [
    AlignWord::new("alpha", vec![10, 11]),
    AlignWord::new("beta", vec![12]),
  ];
  let opts = AlignOptions::new().with_language("english");
  // The pre-tokenized input path: no tokenizer needed.
  let align = a
    .align(&feats, PreTokenizedTranscript::new(&transcript), &opts)
    .unwrap();
  assert_eq!(align.language(), Some("english"));
  let spans = align.spans();
  assert_eq!(spans.len(), 2);
  assert_eq!(spans[0].text(), "alpha");
  assert_eq!(spans[1].text(), "beta");
  // Times are finite seconds (class * 80 ms / 1000), and start <= end after the
  // monotonicity repair guarantees a non-decreasing marker sequence.
  for s in spans {
    assert!(s.start_time().is_finite() && s.end_time().is_finite());
    assert!(
      s.start_time() <= s.end_time(),
      "start {} > end {}",
      s.start_time(),
      s.end_time()
    );
  }
}

#[test]
fn align_pretokenized_multi_window_produces_one_span_per_word() {
  // A multi-window utterance (250 mel frames = 3 conv chunks) aligns end to end
  // through the windowed audio path: `align` derives the audio-token count from
  // the valid length, the windowed encoder emits a matching row count, and the
  // decoder + timestamp head produce one span per word.
  let a = real_window_aligner();
  let feats = filled(&[1, 8, 250], 0.1);
  let transcript = [
    AlignWord::new("alpha", vec![10, 11]),
    AlignWord::new("beta", vec![12]),
  ];
  let opts = AlignOptions::new().with_language("english");
  let align = a
    .align(&feats, PreTokenizedTranscript::new(&transcript), &opts)
    .unwrap();
  let spans = align.spans();
  assert_eq!(spans.len(), 2);
  assert_eq!(spans[0].text(), "alpha");
  assert_eq!(spans[1].text(), "beta");
  for s in spans {
    assert!(s.start_time().is_finite() && s.end_time().is_finite());
    assert!(s.start_time() <= s.end_time());
  }
}

// ════════════════════════════ word splitting ════════════════════════════

/// No pluggable Japanese/Korean segmenter — the default for the inline-language
/// `split_words` tests (only the Japanese/Korean branch consults it).
const NO_SEG: Option<&dyn JpKoSegmenter> = None;

#[test]
fn split_words_space_lang_splits_on_whitespace_and_cleans() {
  // English / default branch: whitespace split, punctuation dropped by
  // clean_token, apostrophes kept.
  let got = split_words("Hello, world! it's", "English", NO_SEG).unwrap();
  assert_eq!(got, vec!["Hello", "world", "it's"]);
}

#[test]
fn split_words_chinese_uses_space_lang_path() {
  // The reference only special-cases Japanese/Korean; Chinese routes through the
  // space-separated path. A single whitespace segment is cleaned then has each
  // CJK ideograph broken out, with a Latin run kept whole: "ab你好c" → ["ab",
  // "你", "好", "c"].
  let got = split_words("ab你好c", "Chinese", NO_SEG).unwrap();
  assert_eq!(got, vec!["ab", "你", "好", "c"]);
}

#[test]
fn split_words_chinese_cleans_segment_before_splitting_cjk() {
  // Mixed Chinese/Latin with internal punctuation: the space-lang path cleans
  // each whitespace segment (dropping the hyphen) before breaking out CJK, so
  // "COVID-19" stays joined as "COVID19" while "你好" splits per character.
  // Reference: tokenize_space_lang → clean_token + split_segment_with_chinese.
  let got = split_words("COVID-19 你好", "Chinese", NO_SEG).unwrap();
  assert_eq!(got, vec!["COVID19", "你", "好"]);
}

#[test]
fn split_words_space_lang_breaks_out_embedded_cjk() {
  // A whitespace segment containing CJK is split so each ideograph is its own
  // token (split_segment_with_chinese), even on the space-separated path.
  let got = split_words("hi你 there", "English", NO_SEG).unwrap();
  assert_eq!(got, vec!["hi", "你", "there"]);
}

#[test]
fn split_words_case_insensitive_language() {
  // The language is matched case-insensitively (reference `language.lower()`).
  assert_eq!(split_words("你", "CHINESE", NO_SEG).unwrap(), vec!["你"]);
}

#[test]
fn split_words_all_punctuation_is_empty() {
  // Every character is dropped by clean_token, leaving no tokens.
  assert!(
    split_words("!!! ---  ???", "English", NO_SEG)
      .unwrap()
      .is_empty()
  );
}

#[test]
fn split_words_empty_transcript_is_empty() {
  // An empty transcript yields no words.
  assert!(split_words("", "English", NO_SEG).unwrap().is_empty());
  assert!(split_words("", "Chinese", NO_SEG).unwrap().is_empty());
}

#[test]
fn split_words_japanese_korean_without_segmenter_are_typed_errors() {
  // With no pluggable segmenter supplied, Japanese/Korean raw splitting is a
  // typed Tokenizer error rather than a wrong split. The message must direct the
  // caller to BOTH the pluggable segmenter hook AND the pre-tokenized path
  // (honest scope, not a claim of full-language raw-text faithfulness).
  for (text, lang) in [("こんにちは", "Japanese"), ("안녕하세요", "Korean")] {
    let err =
      split_words(text, lang, NO_SEG).expect_err("JP/KO raw split without a segmenter must error");
    let Error::Tokenizer(msg) = err else {
      panic!("expected Error::Tokenizer for {lang}, got {err:?}");
    };
    let msg = msg.to_string();
    assert!(
      msg.contains("pre-tokenized"),
      "{lang} error must point to the pre-tokenized path: {msg}"
    );
    assert!(
      msg.contains("PreTokenizedTranscript"),
      "{lang} error must name PreTokenizedTranscript: {msg}"
    );
    assert!(
      msg.contains("RawAlignOptions::with_segmenter"),
      "{lang} error must point to the pluggable segmenter hook in the align options: {msg}"
    );
  }
}

/// A mock [`JpKoSegmenter`] for the tests: it splits on a chosen delimiter
/// (whitespace by default) into word units, recording the `language` it was
/// called with so a test can confirm the dispatch reached it. A faithful stand
/// -in for an external `nagisa` / `soynlp` segmenter — the aligner only needs
/// the ordered word labels back.
struct MockJpKoSegmenter;

impl JpKoSegmenter for MockJpKoSegmenter {
  fn segment(&self, text: &str, language: &str) -> Result<Vec<String>> {
    // Only Japanese/Korean should ever reach the segmenter.
    assert!(
      language == "japanese" || language == "korean",
      "segmenter called with unexpected language {language:?}"
    );
    // Split into one unit per non-space character — a deterministic stand-in for
    // a real morphological segmenter (each CJK character becomes a word unit).
    Ok(
      text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_string())
        .collect(),
    )
  }
}

#[test]
fn split_words_japanese_korean_with_segmenter_splits_through_it() {
  // With a pluggable segmenter attached, Japanese/Korean raw splitting routes
  // through it (one unit per character here) instead of erroring.
  let seg = MockJpKoSegmenter;
  let seg_ref: Option<&dyn JpKoSegmenter> = Some(&seg);
  let ja = split_words("こんにちは", "Japanese", seg_ref).unwrap();
  assert_eq!(ja, vec!["こ", "ん", "に", "ち", "は"]);
  let ko = split_words("안녕", "Korean", seg_ref).unwrap();
  assert_eq!(ko, vec!["안", "녕"]);
}

#[test]
fn split_words_segmenter_not_consulted_for_other_languages() {
  // The segmenter is only for Japanese/Korean; English still routes through the
  // inline space-lang path even when a segmenter is attached (the mock would
  // panic on a non-JP/KO language, so reaching the inline path is the assertion).
  let seg = MockJpKoSegmenter;
  let seg_ref: Option<&dyn JpKoSegmenter> = Some(&seg);
  let got = split_words("hello world", "English", seg_ref).unwrap();
  assert_eq!(got, vec!["hello", "world"]);
}

/// A [`JpKoSegmenter`] that returns a fixed, caller-supplied unit list, ignoring
/// the input text — a stand-in for a degraded/buggy external segmenter so the
/// validation of its output can be exercised directly.
struct FixedUnitsSegmenter(Vec<String>);

impl JpKoSegmenter for FixedUnitsSegmenter {
  fn segment(&self, _text: &str, _language: &str) -> Result<Vec<String>> {
    Ok(self.0.clone())
  }
}

#[test]
fn split_words_segmenter_empty_units_for_nonempty_text_is_typed_error() {
  // A degraded segmenter returning only empty / whitespace-only units for a
  // non-empty Japanese/Korean transcript must surface as a typed Tokenizer error
  // (nothing alignable) — never a successful blank-span alignment.
  let seg = FixedUnitsSegmenter(vec![String::new(), "   ".to_string(), "\t".to_string()]);
  let seg_ref: Option<&dyn JpKoSegmenter> = Some(&seg);
  let err = split_words("こんにちは", "Japanese", seg_ref)
    .expect_err("blank-only segmenter output for non-empty text must be a typed error");
  assert!(
    matches!(err, Error::Tokenizer(_)),
    "expected Error::Tokenizer, got {err:?}"
  );
}

#[test]
fn split_words_segmenter_unit_with_reserved_marker_is_typed_error() {
  // A unit equal to / containing a reserved audio or timestamp marker would
  // inject a special token into the tokenizer input; reject it as a typed error.
  // Cover each marker literal assemble_input_text relies on.
  for marker in [
    "<|audio_start|>",
    "<|audio_pad|>",
    "<|audio_end|>",
    "<timestamp>",
  ] {
    // A unit that *contains* the marker (with a real-looking prefix) is rejected,
    // not just an exact-equal unit.
    let seg = FixedUnitsSegmenter(vec!["안".to_string(), format!("녕{marker}")]);
    let seg_ref: Option<&dyn JpKoSegmenter> = Some(&seg);
    let err = split_words("안녕", "Korean", seg_ref)
      .expect_err("a unit containing a reserved marker must be a typed error");
    assert!(
      matches!(err, Error::Tokenizer(_)),
      "expected Error::Tokenizer for marker {marker:?}, got {err:?}"
    );
  }
}

#[test]
fn split_words_segmenter_valid_units_align_unchanged() {
  // A well-behaved segmenter's units pass through validation unchanged (the
  // normal-case regression: trimming a unit with no surrounding whitespace and
  // dropping no non-empty unit leaves the list as-is).
  let seg = FixedUnitsSegmenter(vec!["안".to_string(), "녕".to_string()]);
  let seg_ref: Option<&dyn JpKoSegmenter> = Some(&seg);
  let got = split_words("안녕", "Korean", seg_ref).unwrap();
  assert_eq!(got, vec!["안", "녕"]);
}

#[test]
fn split_words_segmenter_mixed_units_drop_empties_keep_valid() {
  // A mix of valid and empty/whitespace-only units drops the empties (mirroring
  // the built-in clean path) and keeps the valid words trimmed, in order.
  let seg = FixedUnitsSegmenter(vec![
    " 안 ".to_string(),
    String::new(),
    "녕".to_string(),
    "   ".to_string(),
    "하세요".to_string(),
  ]);
  let seg_ref: Option<&dyn JpKoSegmenter> = Some(&seg);
  let got = split_words("안녕하세요", "Korean", seg_ref).unwrap();
  assert_eq!(got, vec!["안", "녕", "하세요"]);
}

#[test]
fn split_words_segmenter_non_alignable_only_units_are_typed_error() {
  // A degraded segmenter returning units with no alignable character — a
  // zero-width space (U+200B, General_Category Cf) and a punctuation-only unit
  // (the CJK full stop U+3002, Po) — for a non-empty transcript must surface as
  // a typed Tokenizer error: these clean to empty in the built-in path, so they
  // are dropped here too and the no-alignable-units guard fires, never a
  // successful blank / non-alignable span (issue #322).
  let seg = FixedUnitsSegmenter(vec!["\u{200B}".to_string(), "\u{3002}".to_string()]);
  let seg_ref: Option<&dyn JpKoSegmenter> = Some(&seg);
  let err = split_words("こんにちは。", "Japanese", seg_ref)
    .expect_err("non-alignable-only segmenter output for non-empty text must be a typed error");
  assert!(
    matches!(err, Error::Tokenizer(_)),
    "expected Error::Tokenizer, got {err:?}"
  );
}

#[test]
fn split_words_segmenter_punctuation_plus_real_char_keeps_unit() {
  // A unit that is punctuation PLUS a real CJK character is kept (it has an
  // alignable character): the keep/drop predicate only gates emptiness, so the
  // trimmed original — punctuation included — survives as the label, exactly as
  // the built-in path keeps a segment with at least one kept character. The CJK
  // full stop (U+3002) attached to a real ideograph keeps the whole unit.
  let seg = FixedUnitsSegmenter(vec!["안".to_string(), "녕\u{3002}".to_string()]);
  let seg_ref: Option<&dyn JpKoSegmenter> = Some(&seg);
  let got = split_words("안녕。", "Korean", seg_ref).unwrap();
  assert_eq!(got, vec!["안", "녕\u{3002}"]);
}

#[test]
fn is_kept_char_keeps_letters_numbers_and_apostrophe() {
  // The kept set is exactly Unicode General_Category L* / N* plus the apostrophe
  // (reference is_kept_char: category startswith "L"/"N", or ch == "'").
  assert!(is_kept_char('a')); // Ll
  assert!(is_kept_char('Z')); // Lu
  assert!(is_kept_char('5')); // Nd
  assert!(is_kept_char('好')); // Lo (CJK ideograph)
  assert!(is_kept_char('\'')); // apostrophe, kept explicitly
  assert!(is_kept_char('Ⅷ')); // Nl (U+2167 ROMAN NUMERAL EIGHT)
  assert!(is_kept_char('²')); // No (U+00B2 SUPERSCRIPT TWO)
}

#[test]
fn is_kept_char_drops_combining_marks_punctuation_and_symbols() {
  // General_Category-exact parity: combining marks (Mn/Mc/Me) are dropped,
  // unlike the broader char::is_alphabetic / is_numeric derived properties that
  // would keep them. Also punctuation, symbols, and separators are dropped.
  assert!(!is_kept_char('\u{064E}')); // ARABIC FATHA (Mn)
  assert!(!is_kept_char('\u{0650}')); // ARABIC KASRA (Mn)
  assert!(!is_kept_char('\u{0301}')); // COMBINING ACUTE ACCENT (Mn)
  assert!(!is_kept_char('-')); // hyphen-minus (Pd)
  assert!(!is_kept_char('!')); // exclamation mark (Po)
  assert!(!is_kept_char('+')); // plus sign (Sm)
  assert!(!is_kept_char(' ')); // space (Zs)
}

#[test]
fn clean_token_drops_arabic_combining_marks_to_base_letter() {
  // A base Arabic letter followed by a diacritic combining mark cleans to just
  // the base letter — the General_Category check drops FATHA/KASRA (Mn) that the
  // old is_alphabetic()/is_numeric() approximation would have wrongly kept.
  // ARABIC LETTER BEH (U+0628, Lo) + FATHA (U+064E, Mn) → just the BEH.
  assert_eq!(clean_token("\u{0628}\u{064E}"), "\u{0628}");
  // ARABIC LETTER SEEN (U+0633, Lo) + KASRA (U+0650, Mn) → just the SEEN.
  assert_eq!(clean_token("\u{0633}\u{0650}"), "\u{0633}");
}

#[test]
fn clean_token_diacritized_word_matches_canonical_clean() {
  // Parity case: a diacritized Arabic token cleans to the bare base-letter
  // string the canonical clean_token yields. "بَتِ" — BEH+FATHA, TEH+KASRA —
  // cleans to "بت" (BEH then TEH), the same string Python clean_token produces.
  assert_eq!(
    clean_token("\u{0628}\u{064E}\u{062A}\u{0650}"),
    "\u{0628}\u{062A}"
  );
}

#[test]
fn split_words_drops_combining_marks_in_segment() {
  // Through the public split path: a whitespace segment of a base letter plus a
  // combining mark yields just the base letter, and a Latin word with a
  // combining accent keeps only its base letters.
  let got = split_words("\u{0628}\u{064E} cafe\u{0301}", "English", NO_SEG).unwrap();
  assert_eq!(got, vec!["\u{0628}", "cafe"]);
}

#[test]
fn assemble_input_text_lays_out_markers() {
  // Two words, 3 audio tokens → the marker string the tokenizer receives.
  let words = vec!["hello".to_string(), "world".to_string()];
  let s = assemble_input_text(&words, 3).unwrap();
  assert_eq!(
    s,
    "<|audio_start|><|audio_pad|><|audio_pad|><|audio_pad|><|audio_end|>\
     hello<timestamp><timestamp>world<timestamp><timestamp>"
  );
}

// ════════════════════════════ raw-text align ════════════════════════════

/// Build a tiny HF tokenizer whose vocab places the aligner's special markers at
/// the tiny config's ids (`<|audio_start|>`=31, `<|audio_pad|>`=30,
/// `<|audio_end|>`=32, `<timestamp>`=33) and the test words at their own ids,
/// registered as special tokens so they encode as single units. Saved and
/// loaded through the feature-combo-agnostic [`Tokenizer::from_path`] (the
/// wrapper-test idiom).
fn build_aligner_tokenizer(dir: &std::path::Path) {
  use tokenizers::{
    AddedToken, Tokenizer as HfTokenizer, models::wordlevel::WordLevel,
    pre_tokenizers::whitespace::Whitespace,
  };
  let vocab: Vec<(&str, u32)> = vec![
    ("<unk>", 0),
    ("hello", 10),
    ("world", 11),
    ("<|audio_pad|>", 30),
    ("<|audio_start|>", 31),
    ("<|audio_end|>", 32),
    ("<timestamp>", 33),
  ];
  let map = vocab.iter().map(|(w, i)| ((*w).to_string(), *i)).collect();
  let wl = WordLevel::builder()
    .vocab(map)
    .unk_token("<unk>".to_string())
    .build()
    .unwrap();
  let mut hf = HfTokenizer::new(wl);
  hf.with_pre_tokenizer(Some(Whitespace {}));
  // Register the markers as special tokens so they are matched as single units
  // anywhere in the string; the ids come from the vocab entries above.
  let _added = hf.add_special_tokens([
    AddedToken::from("<|audio_start|>", true),
    AddedToken::from("<|audio_pad|>", true),
    AddedToken::from("<|audio_end|>", true),
    AddedToken::from("<timestamp>", true),
  ]);
  hf.save(dir.join("tokenizer.json"), false).unwrap();
}

/// A fresh, unique temp directory per call (process id + monotonic counter),
/// matching the `fresh_dir` idiom in the wrapper tests.
fn fresh_dir(tag: &str) -> std::path::PathBuf {
  use std::sync::atomic::{AtomicU64, Ordering};
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-qwen3-aligner-{tag}-{}-{n}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// The tiny aligner with the marker-aware tokenizer attached.
fn tiny_aligner_with_tokenizer(tag: &str) -> ForcedAligner {
  let dir = fresh_dir(tag);
  build_aligner_tokenizer(&dir);
  let tok = Tokenizer::from_path(&dir, None).expect("load fixture tokenizer");
  let cfg = tiny_aligner_config();
  let audio = tiny_audio_weights(&cfg.audio_config);
  let dec = tiny_decoder_weights(&cfg.text_config);
  ForcedAligner::from_weights_with_tokenizer(cfg, audio, dec, tok)
    .expect("tiny aligner with tokenizer must build")
}

#[test]
fn align_raw_text_tokenizes_internally_and_produces_spans() {
  // The primary path: raw transcript text in, the aligner splits + tokenizes
  // internally (no caller-supplied token ids), and emits one span per word.
  let a = tiny_aligner_with_tokenizer("raw-spans");
  let feats = filled(&[1, 8, 8], 0.1);
  let input = RawTranscript::english("hello world");
  let opts = RawAlignOptions::new();
  let align = a.align(&feats, input, &opts).unwrap();
  // The result language defaults to the transcript language when opts is unset.
  assert_eq!(align.language(), Some("English"));
  let spans = align.spans();
  assert_eq!(spans.len(), 2, "one span per split word");
  assert_eq!(spans[0].text(), "hello");
  assert_eq!(spans[1].text(), "world");
  for s in spans {
    assert!(s.start_time().is_finite() && s.end_time().is_finite());
    assert!(s.start_time() <= s.end_time());
  }
}

#[test]
fn align_raw_text_opts_language_overrides_result_label() {
  // An explicit AlignOptions language overrides the transcript language as the
  // result label (the splitting still uses the transcript's language).
  let a = tiny_aligner_with_tokenizer("raw-override");
  let feats = filled(&[1, 8, 8], 0.1);
  let input = RawTranscript::new("hello world", "English");
  let opts = RawAlignOptions::new().with_language("en-US");
  let align = a.align(&feats, input, &opts).unwrap();
  assert_eq!(align.language(), Some("en-US"));
  assert_eq!(align.spans().len(), 2);
}

#[test]
fn align_raw_text_without_tokenizer_is_typed_error() {
  // The raw-text path needs the model tokenizer; an aligner built without one
  // returns a typed Tokenizer error rather than panicking.
  let a = tiny_aligner(); // from_weights → no tokenizer
  let feats = filled(&[1, 8, 8], 0.1);
  let input = RawTranscript::english("hello world");
  let err = ForcedAlignerTrait::<RawTranscript>::align(&a, &feats, input, &RawAlignOptions::new())
    .expect_err("raw-text align without a tokenizer must error");
  assert!(matches!(err, Error::Tokenizer(_)), "got {err:?}");
}

/// [`RawAlignOptions`] carrying the [`MockJpKoSegmenter`] (one word unit per
/// non-space character) — the segmenter now flows through the per-align options,
/// not a stored aligner field (issue #322).
fn opts_with_segmenter() -> RawAlignOptions {
  RawAlignOptions::new().with_segmenter(Box::new(MockJpKoSegmenter))
}

#[test]
fn raw_align_options_carries_language_and_segmenter() {
  // The segmenter is now a typed part of the align contract (RawAlignOptions),
  // not a bolted-on aligner field: default options carry neither a language nor
  // a segmenter, `with_language` projects through the shared AlignOptions, and
  // `with_segmenter` attaches the owned hook the raw-text path reads.
  let bare = RawAlignOptions::new();
  assert_eq!(bare.language(), None);
  assert!(bare.segmenter().is_none(), "default carries no segmenter");

  let labelled = RawAlignOptions::new().with_language("ja");
  assert_eq!(labelled.language(), Some("ja"));
  assert!(
    labelled.segmenter().is_none(),
    "a language label alone adds no segmenter"
  );

  let full = opts_with_segmenter().with_language("ko");
  assert_eq!(full.language(), Some("ko"));
  assert!(
    full.segmenter().is_some(),
    "with_segmenter attaches the owned hook"
  );

  // The in-place setters chain and mutate through &mut.
  let mut m = RawAlignOptions::new();
  m.set_language("en")
    .set_segmenter(Box::new(MockJpKoSegmenter));
  assert_eq!(m.language(), Some("en"));
  assert!(m.segmenter().is_some());
}

#[test]
fn align_raw_text_japanese_through_segmenter_in_opts_produces_spans() {
  // The pluggable-hook path (issue #322): a Japanese raw transcript splits
  // through the JpKoSegmenter supplied IN the RawAlignOptions (one span per
  // character here) and aligns end to end to per-word spans — no caller-supplied
  // tokenization, no error. The segmenter is sourced from opts, not the struct.
  let a = tiny_aligner_with_tokenizer("jp-spans");
  let feats = filled(&[1, 8, 8], 0.1);
  let input = RawTranscript::new("こん", "Japanese");
  let align = a.align(&feats, input, &opts_with_segmenter()).unwrap();
  // Result language defaults to the transcript language.
  assert_eq!(align.language(), Some("Japanese"));
  let spans = align.spans();
  assert_eq!(spans.len(), 2, "one span per segmented character");
  assert_eq!(spans[0].text(), "こ");
  assert_eq!(spans[1].text(), "ん");
  for s in spans {
    assert!(s.start_time().is_finite() && s.end_time().is_finite());
    assert!(s.start_time() <= s.end_time());
  }
}

#[test]
fn align_raw_text_korean_through_segmenter_in_opts_produces_spans() {
  // The symmetric Korean case through the same opts-supplied pluggable hook.
  let a = tiny_aligner_with_tokenizer("ko-spans");
  let feats = filled(&[1, 8, 8], 0.1);
  let input = RawTranscript::new("안녕", "Korean");
  let align = a.align(&feats, input, &opts_with_segmenter()).unwrap();
  assert_eq!(align.language(), Some("Korean"));
  let spans = align.spans();
  assert_eq!(spans.len(), 2);
  assert_eq!(spans[0].text(), "안");
  assert_eq!(spans[1].text(), "녕");
}

#[test]
fn align_raw_text_japanese_korean_without_segmenter_in_opts_is_typed_error() {
  // With no segmenter supplied in the RawAlignOptions, a Japanese/Korean
  // RawTranscript still returns the typed Tokenizer error through the full align
  // path (the issue #322 behavior, now sourced from opts rather than a stored
  // field — `RawAlignOptions::new()` carries no segmenter).
  let a = tiny_aligner_with_tokenizer("jp-ko-no-seg");
  let feats = filled(&[1, 8, 8], 0.1);
  for lang in ["Japanese", "Korean"] {
    let input = RawTranscript::new("こん", lang);
    let err =
      ForcedAlignerTrait::<RawTranscript>::align(&a, &feats, input, &RawAlignOptions::new())
        .expect_err("JP/KO align without a segmenter in opts must error");
    assert!(matches!(err, Error::Tokenizer(_)), "got {err:?} for {lang}");
  }
}

#[test]
fn raw_transcript_aligner_is_object_safe() {
  // `Box<dyn ForcedAligner<RawTranscript, Options = RawAlignOptions>>` must be
  // constructible and usable — the generic-over-input trait stays object-safe
  // for a chosen input type once the associated Options is named at the dyn site.
  let a = tiny_aligner_with_tokenizer("obj-safe");
  let boxed: Box<dyn ForcedAlignerTrait<RawTranscript, Options = RawAlignOptions>> = Box::new(a);
  let feats = filled(&[1, 8, 8], 0.1);
  let align = boxed
    .align(
      &feats,
      RawTranscript::english("hello world"),
      &RawAlignOptions::new(),
    )
    .unwrap();
  assert_eq!(align.spans().len(), 2);
  assert_eq!(align.spans()[0].text(), "hello");
}

#[test]
fn pretokenized_transcript_aligner_is_object_safe() {
  // The pre-tokenized input is likewise reachable object-safely (no tokenizer
  // required on this path), naming its `Options = AlignOptions` at the dyn site.
  let a = tiny_aligner();
  let boxed: Box<
    dyn for<'b> ForcedAlignerTrait<PreTokenizedTranscript<'b>, Options = AlignOptions>,
  > = Box::new(a);
  let feats = filled(&[1, 8, 8], 0.1);
  let transcript = [AlignWord::new("x", vec![10]), AlignWord::new("y", vec![11])];
  let align = boxed
    .align(
      &feats,
      PreTokenizedTranscript::new(&transcript),
      &AlignOptions::new(),
    )
    .unwrap();
  assert_eq!(align.spans().len(), 2);
}

// ════════════════════════════ config validation ════════════════════════════

/// Wrap a `thinker_config` body in the nested root shape `from_json` requires.
fn nested(thinker_body: &str) -> String {
  format!(r#"{{"thinker_config": {{{thinker_body}}}}}"#)
}

#[test]
fn config_defaults_parse_and_validate() {
  // An empty thinker object → all aligner fields take their reference defaults.
  let cfg = ForcedAlignerConfig::from_json(&nested("")).unwrap();
  assert_eq!(cfg.classify_num, 5000);
  assert_eq!(cfg.audio_token_id, 151676);
  assert_eq!(cfg.timestamp_token_id, 151705);
  assert_eq!(cfg.timestamp_segment_time, 80.0);
  assert!(cfg.validate().is_ok());
}

#[test]
fn config_requires_thinker_config() {
  // A flat (un-nested) root is rejected with a clear missing-key error rather
  // than silently parsed as all-defaults.
  assert!(matches!(
    ForcedAlignerConfig::from_json("{}"),
    Err(Error::MissingKey(_))
  ));
  assert!(matches!(
    ForcedAlignerConfig::from_json(r#"{"classify_num": 3000}"#),
    Err(Error::MissingKey(_))
  ));
}

#[test]
fn config_rejects_non_object_thinker_config() {
  assert!(matches!(
    ForcedAlignerConfig::from_json(r#"{"thinker_config": 7}"#),
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn config_parses_real_nested_thinker_config() {
  // The released Qwen3-ForcedAligner shape: audio/text sub-configs, audio
  // marker ids, classify_num, and the MRoPE text rope_scaling, all under
  // thinker_config. timestamp_token_id / timestamp_segment_time at the root are
  // a fallback the thinker here overrides.
  let json = r#"{
    "model_type": "qwen3_asr",
    "timestamp_token_id": 999,
    "timestamp_segment_time": 40.0,
    "thinker_config": {
      "model_type": "qwen3_forced_aligner",
      "audio_config": {"num_mel_bins": 128, "d_model": 1024, "encoder_attention_heads": 16},
      "text_config": {
        "hidden_size": 2048, "head_dim": 128, "num_attention_heads": 16,
        "num_key_value_heads": 8, "num_hidden_layers": 4, "vocab_size": 151936,
        "rope_theta": 1000000.0,
        "rope_scaling": {"mrope_section": [24, 20, 20], "interleaved": true, "rope_type": "default"}
      },
      "classify_num": 3000,
      "audio_token_id": 151676,
      "audio_start_token_id": 151669,
      "audio_end_token_id": 151670,
      "timestamp_token_id": 151705,
      "timestamp_segment_time": 80.0
    }
  }"#;
  let cfg = ForcedAlignerConfig::from_json(json).unwrap();
  // Values come from the thinker object, not the (overridden) root fallbacks.
  assert_eq!(cfg.classify_num, 3000);
  assert_eq!(cfg.audio_token_id, 151676);
  assert_eq!(cfg.timestamp_token_id, 151705);
  assert_eq!(cfg.timestamp_segment_time, 80.0);
  assert_eq!(cfg.text_config.hidden_size, 2048);
  assert_eq!(cfg.audio_config.num_mel_bins, 128);
  // The non-null MRoPE rope_scaling is parsed (the dense Qwen3 config would
  // reject it); section sums to head_dim/2 = 64, interleaved.
  let mrope = cfg.text_config.mrope().unwrap();
  assert_eq!(mrope.section, [24, 20, 20]);
  assert!(mrope.interleaved);
}

#[test]
fn config_root_timestamp_fallback_applies_when_thinker_omits() {
  // When the thinker object omits the timestamp params, the root-level values
  // are used.
  let json = r#"{
    "timestamp_token_id": 4242,
    "timestamp_segment_time": 100.0,
    "thinker_config": {"classify_num": 3000}
  }"#;
  let cfg = ForcedAlignerConfig::from_json(json).unwrap();
  assert_eq!(cfg.timestamp_token_id, 4242);
  assert_eq!(cfg.timestamp_segment_time, 100.0);
  assert_eq!(cfg.classify_num, 3000);
}

#[test]
fn config_rejects_oversized_classify_num() {
  let json = nested(r#""classify_num": 33554433"#); // > 2^24
  assert!(matches!(
    ForcedAlignerConfig::from_json(&json),
    Err(Error::CapExceeded(_)) | Err(Error::OutOfRange(_))
  ));
}

#[test]
fn config_rejects_nonpositive_classify_num() {
  let json = nested(r#""classify_num": 0"#);
  assert!(matches!(
    ForcedAlignerConfig::from_json(&json),
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn config_rejects_negative_token_id() {
  let json = nested(r#""timestamp_token_id": -1"#);
  assert!(matches!(
    ForcedAlignerConfig::from_json(&json),
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn config_rejects_nonfinite_and_nonpositive_segment_time() {
  let mut cfg = ForcedAlignerConfig::from_json(&nested("")).unwrap();
  cfg.timestamp_segment_time = f64::NAN;
  assert!(matches!(cfg.validate(), Err(Error::NonFiniteScalar(_))));

  let mut cfg = ForcedAlignerConfig::from_json(&nested("")).unwrap();
  cfg.timestamp_segment_time = -80.0;
  assert!(matches!(cfg.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn config_propagates_subconfig_validation() {
  // A bad audio sub-config (odd d_model) must surface through the aligner
  // config's validate.
  let json = nested(r#""audio_config": {"d_model": 5}"#);
  assert!(ForcedAlignerConfig::from_json(&json).is_err());
}

#[test]
fn config_rejects_malformed_mrope_section() {
  // A non-null rope_scaling whose mrope_section does not sum to head_dim/2 is
  // rejected (head_dim 128 → half 64, but 10+10+10 = 30).
  let json =
    nested(r#""text_config": {"head_dim": 128, "rope_scaling": {"mrope_section": [10, 10, 10]}}"#);
  assert!(matches!(
    ForcedAlignerConfig::from_json(&json),
    Err(Error::OutOfRange(_))
  ));
}
