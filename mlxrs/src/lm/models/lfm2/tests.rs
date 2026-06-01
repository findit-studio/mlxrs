//! Oracle tests for the LFM2 LM blocks.
//!
//! The load-bearing check is the [`ShortConv`] round-trip: the prefill
//! (left-pad) path must equal an independent hand-computed causal depthwise
//! convolution, and feeding the same total sequence through the cache one
//! chunk at a time must reproduce the prefill output. Every oracle is
//! derived from the documented `lfm2.py` math (an independent reimplementation
//! in `f64`), never a call to the code under test.

use super::{linear::Linear, *};
use crate::array::Array;

const TOL: f32 = 1e-4;

/// A dense `(rows, cols)` weight matrix, row-major.
type Mat = Vec<Vec<f64>>;

/// `(in_proj, conv, out_proj, hidden, K)` — the [`ShortConv`] weight bundle
/// the round-trip tests share.
type ConvWeights = (Mat, Mat, Mat, usize, usize);

fn assert_close(got: &[f32], want: &[f32]) {
  assert_eq!(got.len(), want.len(), "length mismatch");
  for (i, (g, w)) in got.iter().zip(want).enumerate() {
    assert!(
      (g - w).abs() <= TOL,
      "index {i}: got {g}, want {w} (|Δ|={})",
      (g - w).abs()
    );
  }
}

// ───────────────────────── ShortConv oracle ─────────────────────────

/// An independent `f64` reimplementation of the `ShortConv` forward pass for
/// the **prefill** (no-cache, no-mask) path, derived directly from
/// `lfm2.py:112-170`. NOT a call into `ShortConv::forward`.
///
/// Shapes: `x` is `[batch][L][hidden]`; `in_w` is the `(3*hidden, hidden)`
/// `in_proj` weight; `conv_w` is the `(hidden, K, 1)` depthwise weight;
/// `out_w` is the `(hidden, hidden)` `out_proj` weight. All biases are zero.
// The convolution math is naturally index-addressed (`conv_out[t]` over the
// kernel taps `conv_w[c][j]`); rewriting it with iterator adapters would
// obscure the closed form, so the range-loop lint is silenced here.
#[allow(clippy::needless_range_loop)]
fn shortconv_prefill_oracle(
  x: &[Vec<Vec<f64>>],
  in_w: &[Vec<f64>],
  conv_w: &[Vec<f64>],
  out_w: &[Vec<f64>],
  hidden: usize,
  k: usize,
) -> Vec<Vec<Vec<f64>>> {
  let batch = x.len();
  let len = x[0].len();
  // in_proj: y[b][t][o] = sum_i x[b][t][i] * in_w[o][i]; split o into B,C,x.
  // B = chunk 0, C = chunk 1, x_split = chunk 2 (each `hidden` wide).
  let mut bx = vec![vec![vec![0.0_f64; hidden]; len]; batch]; // B * x_split
  let mut c_gate = vec![vec![vec![0.0_f64; hidden]; len]; batch];
  for b in 0..batch {
    for t in 0..len {
      let proj = |o: usize| -> f64 { (0..hidden).map(|i| x[b][t][i] * in_w[o][i]).sum() };
      for c in 0..hidden {
        let b_val = proj(c);
        let c_val = proj(hidden + c);
        let x_val = proj(2 * hidden + c);
        bx[b][t][c] = b_val * x_val;
        c_gate[b][t][c] = c_val;
      }
    }
  }
  // Left-pad Bx by K-1 zeros on the time axis, then depthwise conv1d
  // (cross-correlation): conv_out[b][t][c] = sum_{j=0}^{K-1}
  // padded[b][t+j][c] * conv_w[c][j]. With K-1 left padding the output
  // length is exactly `len`, and conv_out[b][t] depends on Bx[b][t-K+1..=t].
  let pad = k - 1;
  let mut conv_out = vec![vec![vec![0.0_f64; hidden]; len]; batch];
  for b in 0..batch {
    for t in 0..len {
      for c in 0..hidden {
        let mut acc = 0.0;
        for j in 0..k {
          // padded index t+j; padded[0..pad] are zero, padded[pad+s] = Bx[s].
          let pidx = t + j;
          if pidx >= pad {
            acc += bx[b][pidx - pad][c] * conv_w[c][j];
          }
        }
        conv_out[b][t][c] = acc;
      }
    }
  }
  // y = C * conv_out, then out_proj.
  let mut out = vec![vec![vec![0.0_f64; hidden]; len]; batch];
  for b in 0..batch {
    for t in 0..len {
      let y: Vec<f64> = (0..hidden)
        .map(|c| c_gate[b][t][c] * conv_out[b][t][c])
        .collect();
      for o in 0..hidden {
        out[b][t][o] = (0..hidden).map(|i| y[i] * out_w[o][i]).sum();
      }
    }
  }
  out
}

/// Build a [`ShortConv`] from explicit dense weights. `in_w`/`out_w` are
/// `(out, in)` row-major; `conv_w` is `(hidden, K, 1)` row-major.
fn make_shortconv(
  in_w: &[Vec<f64>],
  conv_w: &[Vec<f64>],
  out_w: &[Vec<f64>],
  hidden: usize,
  k: usize,
) -> ShortConv {
  let flat =
    |w: &[Vec<f64>]| -> Vec<f32> { w.iter().flat_map(|r| r.iter().map(|&v| v as f32)).collect() };
  let in_rows = in_w.len();
  let in_cols = in_w[0].len();
  let in_proj = Linear::new(
    Array::from_slice::<f32>(&flat(in_w), &(in_rows, in_cols)).unwrap(),
    None,
  );
  let out_rows = out_w.len();
  let out_cols = out_w[0].len();
  let out_proj = Linear::new(
    Array::from_slice::<f32>(&flat(out_w), &(out_rows, out_cols)).unwrap(),
    None,
  );
  let conv_flat: Vec<f32> = conv_w
    .iter()
    .flat_map(|r| r.iter().map(|&v| v as f32))
    .collect();
  let conv_weight = Array::from_slice::<f32>(&conv_flat, &(hidden, k, 1usize)).unwrap();
  ShortConv {
    hidden_size: hidden as i32,
    l_cache: k as i32,
    conv_weight,
    conv_bias: None,
    in_proj,
    out_proj,
  }
}

fn x_to_array(x: &[Vec<Vec<f64>>]) -> Array {
  let batch = x.len();
  let len = x[0].len();
  let hidden = x[0][0].len();
  let flat: Vec<f32> = x
    .iter()
    .flat_map(|b| b.iter().flat_map(|t| t.iter().map(|&v| v as f32)))
    .collect();
  Array::from_slice::<f32>(&flat, &(batch, len, hidden)).unwrap()
}

/// Sample weights/input for the multi-channel round-trip tests.
fn sample_weights() -> ConvWeights {
  let hidden = 2;
  let k = 3;
  // in_proj: (3*hidden, hidden) = (6, 2).
  let in_w = vec![
    vec![0.5, -0.2],
    vec![0.1, 0.3],
    vec![-0.4, 0.6],
    vec![0.2, 0.7],
    vec![0.8, -0.1],
    vec![-0.3, 0.5],
  ];
  // conv: (hidden, K, 1) = (2, 3, 1).
  let conv_w = vec![vec![0.3, -0.5, 0.7], vec![0.9, 0.2, -0.4]];
  // out_proj: (hidden, hidden) = (2, 2).
  let out_w = vec![vec![1.1, -0.3], vec![0.4, 0.8]];
  (in_w, conv_w, out_w, hidden, k)
}

fn sample_input() -> Vec<Vec<Vec<f64>>> {
  // batch 1, L 5, hidden 2.
  vec![vec![
    vec![1.0, -1.0],
    vec![0.5, 2.0],
    vec![-0.5, 0.3],
    vec![2.0, 1.5],
    vec![-1.0, 0.8],
  ]]
}

#[test]
fn shortconv_prefill_matches_manual_causal_conv() {
  let (in_w, conv_w, out_w, hidden, k) = sample_weights();
  let conv = make_shortconv(&in_w, &conv_w, &out_w, hidden, k);
  let x = sample_input();
  let arr = x_to_array(&x);

  let mut got = conv.forward(&arr, None, None).unwrap();
  let want = shortconv_prefill_oracle(&x, &in_w, &conv_w, &out_w, hidden, k);
  let want_flat: Vec<f32> = want
    .iter()
    .flat_map(|b| b.iter().flat_map(|t| t.iter().map(|&v| v as f32)))
    .collect();
  assert_eq!(got.shape(), vec![1, 5, hidden]);
  assert_close(&got.to_vec::<f32>().unwrap(), &want_flat);
}

#[test]
fn shortconv_decode_with_cache_matches_prefill() {
  let (in_w, conv_w, out_w, hidden, k) = sample_weights();
  let conv = make_shortconv(&in_w, &conv_w, &out_w, hidden, k);
  let x = sample_input();
  let arr = x_to_array(&x);

  // Prefill reference over the whole sequence (no cache).
  let mut prefill = conv.forward(&arr, None, None).unwrap();
  let prefill_v = prefill.to_vec::<f32>().unwrap();

  // Decode: one token at a time through a fresh one-slot ArraysCache.
  let mut cache = ArraysCache::new(1);
  let mut decoded: Vec<f32> = Vec::new();
  for token in &x[0] {
    // This token as a [1, 1, hidden] step.
    let step: Vec<f32> = token.iter().map(|&v| v as f32).collect();
    let step_arr = Array::from_slice::<f32>(&step, &(1usize, 1usize, hidden)).unwrap();
    let mut out = conv.forward(&step_arr, None, Some(&mut cache)).unwrap();
    decoded.extend(out.to_vec::<f32>().unwrap());
  }
  // The decode stream (concatenated per-step outputs) must equal the prefill
  // output position-for-position.
  assert_close(&decoded, &prefill_v);
}

#[test]
fn shortconv_decode_in_two_chunks_matches_prefill() {
  // The same round-trip but with an uneven chunking (a 3-token prefill chunk
  // followed by a 2-token decode chunk) — exercises the cached-state prefix
  // path with `L > 1` chunks, not just single-token decode.
  let (in_w, conv_w, out_w, hidden, k) = sample_weights();
  let conv = make_shortconv(&in_w, &conv_w, &out_w, hidden, k);
  let x = sample_input();
  let arr = x_to_array(&x);

  let mut prefill = conv.forward(&arr, None, None).unwrap();
  let prefill_v = prefill.to_vec::<f32>().unwrap();

  let mut cache = ArraysCache::new(1);
  let chunk = |rows: &[Vec<f64>]| -> Array {
    let flat: Vec<f32> = rows
      .iter()
      .flat_map(|t| t.iter().map(|&v| v as f32))
      .collect();
    Array::from_slice::<f32>(&flat, &(1usize, rows.len(), hidden)).unwrap()
  };
  let mut decoded: Vec<f32> = Vec::new();
  let mut c0 = conv
    .forward(&chunk(&x[0][0..3]), None, Some(&mut cache))
    .unwrap();
  decoded.extend(c0.to_vec::<f32>().unwrap());
  let mut c1 = conv
    .forward(&chunk(&x[0][3..5]), None, Some(&mut cache))
    .unwrap();
  decoded.extend(c1.to_vec::<f32>().unwrap());

  assert_close(&decoded, &prefill_v);
}

// ───────────────────────── sanitize ─────────────────────────

#[test]
fn sanitize_transposes_pytorch_conv_weight() {
  use std::collections::HashMap;
  // PyTorch depthwise conv weight (C=2, 1, K=3) — last axis (3) > middle (1).
  let pt =
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2usize, 1usize, 3usize)).unwrap();
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("model.layers.0.conv.conv.weight".to_string(), pt);
  weights.insert(
    // A non-conv weight must pass through untouched.
    "model.layers.0.feed_forward.w1.weight".to_string(),
    Array::from_slice::<f32>(&[7.0, 8.0], &(1usize, 2usize)).unwrap(),
  );
  Lfm2::sanitize(&mut weights).unwrap();

  let mut conv = weights.remove("model.layers.0.conv.conv.weight").unwrap();
  // Transposed to MLX (C, K, 1) = (2, 3, 1).
  assert_eq!(conv.shape(), vec![2, 3, 1]);
  // transpose(0, 2, 1) of [[ [1,2,3] ]] layout: channel 0 -> [1,2,3], so the
  // (C,K,1) data is the same flat order here ([1,2,3,4,5,6]).
  assert_eq!(
    conv.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
  let w1 = weights
    .remove("model.layers.0.feed_forward.w1.weight")
    .unwrap();
  assert_eq!(w1.shape(), vec![1, 2]);
}

#[test]
fn sanitize_leaves_mlx_layout_conv_weight_untouched() {
  use std::collections::HashMap;
  // Already MLX layout (C=2, K=3, 1): last axis (1) < middle (3) — no change.
  let mlx =
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2usize, 3usize, 1usize)).unwrap();
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("model.layers.1.conv.conv.weight".to_string(), mlx);
  Lfm2::sanitize(&mut weights).unwrap();
  let conv = &weights["model.layers.1.conv.conv.weight"];
  assert_eq!(conv.shape(), vec![2, 3, 1]);
}

// ───────────────────────── config ─────────────────────────

#[test]
fn text_config_defaults_and_overrides() {
  // Empty object → all reference defaults.
  let cfg = TextConfig::from_json("{}").unwrap();
  assert_eq!(cfg.hidden_size, 1024);
  assert_eq!(cfg.num_hidden_layers, 16);
  assert_eq!(cfg.num_attention_heads, 16);
  assert_eq!(cfg.num_key_value_heads, 8);
  assert_eq!(cfg.vocab_size, 65536);
  assert_eq!(cfg.conv_l_cache, 3);
  assert!(!cfg.conv_bias);
  assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
  assert_eq!(cfg.head_dim(), 64);

  // Unmodeled keys are ignored (from_dict semantics); overrides applied.
  let json = r#"{"hidden_size": 8, "num_attention_heads": 2,
    "num_hidden_layers": 3, "conv_L_cache": 4, "some_future_key": 99}"#;
  let cfg = TextConfig::from_json(json).unwrap();
  assert_eq!(cfg.hidden_size, 8);
  assert_eq!(cfg.num_attention_heads, 2);
  assert_eq!(cfg.conv_l_cache, 4);
  assert_eq!(cfg.head_dim(), 4);
}

#[test]
fn attention_indices_from_layer_types() {
  let json = r#"{"num_hidden_layers": 4,
    "layer_types": ["conv", "full_attention", "conv", "full_attention"]}"#;
  let cfg = TextConfig::from_json(json).unwrap();
  assert_eq!(cfg.attention_layer_indices().unwrap(), vec![1, 3]);
}

#[test]
fn attention_indices_explicit_full_attn_idxs_wins() {
  // When full_attn_idxs is present it is authoritative (mlx-lm __post_init__
  // only derives from layer_types when full_attn_idxs is None).
  let json = r#"{"num_hidden_layers": 4, "full_attn_idxs": [2],
    "layer_types": ["full_attention", "conv", "conv", "conv"]}"#;
  let cfg = TextConfig::from_json(json).unwrap();
  assert_eq!(cfg.attention_layer_indices().unwrap(), vec![2]);
}

#[test]
fn attention_indices_none_means_all_conv() {
  let json = r#"{"num_hidden_layers": 3}"#;
  let cfg = TextConfig::from_json(json).unwrap();
  assert!(cfg.attention_layer_indices().unwrap().is_empty());
}

#[test]
fn attention_index_out_of_range_rejected() {
  let json = r#"{"num_hidden_layers": 2, "full_attn_idxs": [5]}"#;
  let cfg = TextConfig::from_json(json).unwrap();
  assert!(cfg.attention_layer_indices().is_err());
}

#[test]
fn adjusted_ff_dim_matches_reference_formula() {
  // block_ff_dim=6656, multiple_of=256, multiplier=1.0, adjust=true:
  //   int(2*6656/3)=4437; int(1.0*4437)=4437;
  //   256 * ceil(4437/256) = 256 * 18 = 4608.
  assert_eq!(adjusted_ff_dim(6656, 256, true, 1.0), 4608);
  // adjust=false returns ff_dim unchanged.
  assert_eq!(adjusted_ff_dim(6656, 256, false, 1.0), 6656);
  // A non-unit multiplier: int(2*3000/3)=2000; int(0.5*2000)=1000;
  //   128 * ceil(1000/128) = 128 * 8 = 1024.
  assert_eq!(adjusted_ff_dim(3000, 128, true, 0.5), 1024);
}
