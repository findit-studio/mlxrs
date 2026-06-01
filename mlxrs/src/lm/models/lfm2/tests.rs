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

#[test]
fn shortconv_decode_with_cache_lengths_matches_prefill() {
  // Exercises the `cache.lengths`-set decode branch — the one that builds the
  // host `lengths_positions` index and stashes via `take_along_axis` (the path
  // the cap + checked host alloc protect). With a single batch row whose length
  // covers the whole sequence, `clip(length, 0, t)` keeps exactly the trailing
  // `n_keep` frames, so the per-step decode stream must still equal the prefill
  // output position-for-position.
  let (in_w, conv_w, out_w, hidden, k) = sample_weights();
  let conv = make_shortconv(&in_w, &conv_w, &out_w, hidden, k);
  let x = sample_input();
  let arr = x_to_array(&x);

  let mut prefill = conv.forward(&arr, None, None).unwrap();
  let prefill_v = prefill.to_vec::<f32>().unwrap();

  let mut cache = ArraysCache::new(1);
  let mut decoded: Vec<f32> = Vec::new();
  // `prepare` sets `cache.lengths`; mlx-lm's `advance` decrements it per step,
  // so seed it with the full remaining length before each token.
  let total = x[0].len() as i32;
  for (step, token) in x[0].iter().enumerate() {
    cache.prepare(&[total - step as i32]);
    let s: Vec<f32> = token.iter().map(|&v| v as f32).collect();
    let step_arr = Array::from_slice::<f32>(&s, &(1usize, 1usize, hidden)).unwrap();
    let mut out = conv.forward(&step_arr, None, Some(&mut cache)).unwrap();
    decoded.extend(out.to_vec::<f32>().unwrap());
  }
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

  // Unmodeled keys are ignored (from_dict semantics); overrides applied. The
  // overridden head counts stay GQA-consistent (2 query heads / 2 kv heads) so
  // the config also passes validation.
  let json = r#"{"hidden_size": 8, "num_attention_heads": 2,
    "num_key_value_heads": 2, "num_hidden_layers": 3, "conv_L_cache": 4,
    "some_future_key": 99}"#;
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
fn attention_indices_reject_over_cap_full_attn_idxs() {
  // A tiny `num_hidden_layers` paired with a `full_attn_idxs` longer than the
  // realistic `MAX_CONFIG_CARDINALITY` cap would drive a large host allocation
  // (the wholesale clone) before any index is range-checked. The length is
  // bounded up front and rejected as a recoverable CapExceeded. Every entry is
  // in range (`0`), so this isolates the *length* guard from the per-index one.
  let over = (MAX_CONFIG_CARDINALITY as usize) + 1;
  let idxs = vec!["0"; over].join(",");
  let json = format!(r#"{{"num_hidden_layers": 2, "full_attn_idxs": [{idxs}]}}"#);
  let cfg = TextConfig::from_json(&json).unwrap();
  assert!(matches!(
    cfg.attention_layer_indices(),
    Err(Error::CapExceeded(_))
  ));
  // Exactly at the cap is admitted (the length guard alone passes); every entry
  // is index 0 so the per-index range check also passes.
  let at = vec!["0"; MAX_CONFIG_CARDINALITY as usize].join(",");
  let at_json = format!(r#"{{"num_hidden_layers": 2, "full_attn_idxs": [{at}]}}"#);
  let at_cfg = TextConfig::from_json(&at_json).unwrap();
  assert!(at_cfg.attention_layer_indices().is_ok());
}

#[test]
fn attention_indices_reject_over_cap_layer_types() {
  // The `layer_types` collect path is bounded the same way: a per-layer list
  // longer than the cap is a recoverable CapExceeded before the
  // `"full_attention"` filter allocates. Entries are all `"conv"` so no index
  // is ever produced — this isolates the source-length guard.
  let over = (MAX_CONFIG_CARDINALITY as usize) + 1;
  let types = vec!["\"conv\""; over].join(",");
  let json = format!(r#"{{"num_hidden_layers": 2, "layer_types": [{types}]}}"#);
  let cfg = TextConfig::from_json(&json).unwrap();
  assert!(matches!(
    cfg.attention_layer_indices(),
    Err(Error::CapExceeded(_))
  ));
}

#[test]
fn from_json_rejects_oversized_full_attn_idxs_during_deserialization() {
  // An array strictly longer than `MAX_CONFIG_CARDINALITY + 1` must be rejected
  // *while parsing* (the `BoundedSeq` visitor stops growing the `Vec` at the cap
  // rather than draining millions of elements into memory), so `from_json`
  // itself fails with a typed `Error::Parse` — the `attention_layer_indices`
  // `CapExceeded` path is never reached because the value never fully
  // deserializes. The element count is `cap + 2` so it is the smallest array the
  // deserializer rejects; entries are `0` (in range) to isolate the *length*
  // bound from any per-index check.
  let over = (MAX_CONFIG_CARDINALITY as usize) + 2;
  let idxs = vec!["0"; over].join(",");
  let json = format!(r#"{{"num_hidden_layers": 2, "full_attn_idxs": [{idxs}]}}"#);
  assert!(matches!(TextConfig::from_json(&json), Err(Error::Parse(_))));
  // The boundary the accessor tests (`cap + 1`) still deserializes — the
  // deserializer admits exactly `cap + 1`, leaving the over-cap rejection to the
  // `require_cardinality` check in `attention_layer_indices` (a recoverable
  // `CapExceeded`), so the two guards compose without a gap.
  let boundary = (MAX_CONFIG_CARDINALITY as usize) + 1;
  let b_idxs = vec!["0"; boundary].join(",");
  let b_json = format!(r#"{{"num_hidden_layers": 2, "full_attn_idxs": [{b_idxs}]}}"#);
  let b_cfg = TextConfig::from_json(&b_json).unwrap();
  assert!(matches!(
    b_cfg.attention_layer_indices(),
    Err(Error::CapExceeded(_))
  ));
}

#[test]
fn from_json_rejects_oversized_layer_types_during_deserialization() {
  // The `layer_types` field is bounded the same way: an array longer than
  // `cap + 1` is rejected by the `BoundedSeq` visitor during parsing, so
  // `from_json` returns a typed `Error::Parse` before the `"full_attention"`
  // collect can run. Entries are `"conv"` so no attention index would ever be
  // produced — this isolates the deserialize-time length bound.
  let over = (MAX_CONFIG_CARDINALITY as usize) + 2;
  let types = vec!["\"conv\""; over].join(",");
  let json = format!(r#"{{"num_hidden_layers": 2, "layer_types": [{types}]}}"#);
  assert!(matches!(TextConfig::from_json(&json), Err(Error::Parse(_))));
  // `cap + 1` still parses, deferring the over-cap rejection to the
  // `require_cardinality` check in `attention_layer_indices`.
  let boundary = (MAX_CONFIG_CARDINALITY as usize) + 1;
  let b_types = vec!["\"conv\""; boundary].join(",");
  let b_json = format!(r#"{{"num_hidden_layers": 2, "layer_types": [{b_types}]}}"#);
  let b_cfg = TextConfig::from_json(&b_json).unwrap();
  assert!(matches!(
    b_cfg.attention_layer_indices(),
    Err(Error::CapExceeded(_))
  ));
}

#[test]
fn bounded_seq_at_cap_parses_and_capacity_is_bounded() {
  // Exactly `MAX_CONFIG_SEQ_LEN` elements deserialize successfully through the
  // `BoundedSeq` visitor, and — the regression Codex asked for — the backing
  // `Vec`'s capacity never exceeds the cap even though `serde_json` supplies no
  // `size_hint` (the visitor pins `with_capacity(MAX_CONFIG_SEQ_LEN)`, so `push`
  // cannot drive geometric doubling past the ceiling).
  let idxs = vec!["0"; MAX_CONFIG_SEQ_LEN].join(",");
  let json = format!("[{idxs}]");
  let parsed: BoundedSeq<i32> = serde_json::from_str(&json).unwrap();
  assert_eq!(parsed.0.len(), MAX_CONFIG_SEQ_LEN);
  assert!(
    parsed.0.capacity() <= MAX_CONFIG_SEQ_LEN,
    "capacity {} exceeded the cap {MAX_CONFIG_SEQ_LEN}",
    parsed.0.capacity()
  );

  // Same guarantee for the `String` element type whose over-cap surplus the fix
  // must avoid materializing: capacity stays pinned at the ceiling.
  let types = vec!["\"conv\""; MAX_CONFIG_SEQ_LEN].join(",");
  let s_json = format!("[{types}]");
  let s_parsed: BoundedSeq<String> = serde_json::from_str(&s_json).unwrap();
  assert_eq!(s_parsed.0.len(), MAX_CONFIG_SEQ_LEN);
  assert!(
    s_parsed.0.capacity() <= MAX_CONFIG_SEQ_LEN,
    "string-seq capacity {} exceeded the cap {MAX_CONFIG_SEQ_LEN}",
    s_parsed.0.capacity()
  );
}

#[test]
fn bounded_seq_rejects_surplus_without_materializing() {
  // One element past the ceiling (`MAX_CONFIG_SEQ_LEN + 1`, i.e. `cap + 2`) is
  // rejected at parse by the visitor's `IgnoredAny` probe — the surplus element
  // is *never* deserialized as `T`. Deserializing `BoundedSeq` directly surfaces
  // the visitor's typed serde error (which `from_json` maps to `Error::Parse`).
  let over = MAX_CONFIG_SEQ_LEN + 1;
  let idxs = vec!["0"; over].join(",");
  let json = format!("[{idxs}]");
  assert!(serde_json::from_str::<BoundedSeq<i32>>(&json).is_err());

  // The surplus `String` is not allocated: a value `serde_json` could not parse
  // *as a `String`* (a bare number) in the surplus slot is still rejected as an
  // over-length array, not a type error — proof the probe runs as `IgnoredAny`
  // before any `String` materialization is attempted at the ceiling.
  let mut elems = vec!["\"conv\""; MAX_CONFIG_SEQ_LEN];
  elems.push("123");
  let s_json = format!("[{}]", elems.join(","));
  assert!(serde_json::from_str::<BoundedSeq<String>>(&s_json).is_err());
}

#[test]
fn bounded_opt_fields_handle_null_and_absent() {
  // Explicit JSON `null` → `None` through the `Option` shim (the
  // `deserialize_bounded_opt_vec` wrapper handles `null` without entering the
  // sequence visitor or panicking).
  let null_json = r#"{"num_hidden_layers": 2,
    "full_attn_idxs": null, "layer_types": null}"#;
  let null_cfg = TextConfig::from_json(null_json).unwrap();
  assert!(null_cfg.full_attn_idxs.is_none());
  assert!(null_cfg.layer_types.is_none());

  // Absent fields → `None` via `#[serde(default)]` (the `deserialize_with` shim
  // is not even invoked for a missing key).
  let absent_cfg = TextConfig::from_json(r#"{"num_hidden_layers": 2}"#).unwrap();
  assert!(absent_cfg.full_attn_idxs.is_none());
  assert!(absent_cfg.layer_types.is_none());
}

#[test]
fn validate_rejects_zero_negative_nondivisible_and_oversized() {
  // A divisible base (hidden 8 / heads 2 = head_dim 4; heads 2 / kv 2) so each
  // case below isolates a single malformed field on an otherwise-sound config.
  let base = r#"{"hidden_size": 8, "num_attention_heads": 2,
    "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 16}"#;
  assert!(TextConfig::from_json(base).unwrap().validate().is_ok());

  // Zero dimension.
  let zero = r#"{"hidden_size": 0, "num_attention_heads": 2,
    "num_key_value_heads": 2, "num_hidden_layers": 2}"#;
  assert!(matches!(
    TextConfig::from_json(zero),
    Err(Error::OutOfRange(_))
  ));

  // Negative dimension.
  let neg = r#"{"num_hidden_layers": -3}"#;
  assert!(matches!(
    TextConfig::from_json(neg),
    Err(Error::OutOfRange(_))
  ));

  // hidden_size not divisible by num_attention_heads (head_dim would truncate).
  let nondiv = r#"{"hidden_size": 10, "num_attention_heads": 4,
    "num_key_value_heads": 2, "num_hidden_layers": 2}"#;
  assert!(matches!(
    TextConfig::from_json(nondiv),
    Err(Error::DivisibilityConstraint(_))
  ));

  // num_attention_heads not divisible by num_key_value_heads (GQA grouping).
  let nondiv_gqa = r#"{"hidden_size": 12, "num_attention_heads": 6,
    "num_key_value_heads": 4, "num_hidden_layers": 2}"#;
  assert!(matches!(
    TextConfig::from_json(nondiv_gqa),
    Err(Error::DivisibilityConstraint(_))
  ));

  // Oversized / overflow-prone dimension (above the 2^24 cap).
  let oversized = r#"{"vocab_size": 33554432, "hidden_size": 8,
    "num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 2}"#;
  assert!(matches!(
    TextConfig::from_json(oversized),
    Err(Error::OutOfRange(_))
  ));

  // block_multiple_of == 0 would divide-by-zero in adjusted_ff_dim.
  let zero_multiple = r#"{"block_multiple_of": 0, "hidden_size": 8,
    "num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 2}"#;
  assert!(matches!(
    TextConfig::from_json(zero_multiple),
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn default_config_passes_validation() {
  // The reference defaults must themselves be a valid configuration.
  assert!(TextConfig::from_json("{}").unwrap().validate().is_ok());
}

#[test]
fn validate_rejects_huge_layer_and_head_counts() {
  // A cardinality field past the realistic 4096 cap would size a multi-GB
  // per-layer Vec / cache before the first missing-key error; the shared
  // `require_cardinality` guard rejects it as a recoverable CapExceeded at
  // config time.
  let huge_layers = r#"{"num_hidden_layers": 16777216, "hidden_size": 8,
    "num_attention_heads": 2, "num_key_value_heads": 2}"#;
  assert!(matches!(
    TextConfig::from_json(huge_layers),
    Err(Error::CapExceeded(_))
  ));
  // The head counts are cardinalities too (they bound per-head reshapes and
  // the cap keeps them realistic). 8192 > 4096.
  let huge_heads = r#"{"num_attention_heads": 8192, "num_key_value_heads": 8192,
    "hidden_size": 8192, "num_hidden_layers": 2}"#;
  assert!(matches!(
    TextConfig::from_json(huge_heads),
    Err(Error::CapExceeded(_))
  ));
  // The cap boundary: exactly 4096 layers is accepted (within cap), 4097 is
  // not — isolating the off-by-one. `hidden_size`/heads kept sound + even
  // head_dim.
  let at_cap = r#"{"num_hidden_layers": 4096, "hidden_size": 8,
    "num_attention_heads": 2, "num_key_value_heads": 2}"#;
  assert!(TextConfig::from_json(at_cap).unwrap().validate().is_ok());
  let over_cap = r#"{"num_hidden_layers": 4097, "hidden_size": 8,
    "num_attention_heads": 2, "num_key_value_heads": 2}"#;
  assert!(matches!(
    TextConfig::from_json(over_cap),
    Err(Error::CapExceeded(_))
  ));
}

#[test]
fn validate_rejects_odd_rope_head_dim() {
  // hidden=6, heads=2 -> head_dim=3 (odd). RoPE pairs feature k with
  // k + head_dim/2, so an odd head_dim loads but only fails in the forward
  // pass; validate must reject it eagerly. (6 % 2 == 0 so the divisibility
  // check passes — this isolates the even-head_dim rule.)
  let odd = r#"{"hidden_size": 6, "num_attention_heads": 2,
    "num_key_value_heads": 2, "num_hidden_layers": 2}"#;
  assert!(matches!(
    TextConfig::from_json(odd),
    Err(Error::OutOfRange(_))
  ));
  // An even head_dim (hidden=8, heads=2 -> 4) passes.
  let even = r#"{"hidden_size": 8, "num_attention_heads": 2,
    "num_key_value_heads": 2, "num_hidden_layers": 2}"#;
  assert!(TextConfig::from_json(even).unwrap().validate().is_ok());
}

#[test]
fn validate_rejects_unbounded_ffn_multiplier() {
  // A huge `block_ffn_dim_multiplier` overflows the i32 MLP-width arithmetic
  // in adjusted_ff_dim (which `from_weights` calls). It must be a recoverable
  // config-time error, not a saturating cast + overflow panic. (Sound base
  // config so the multiplier is the only malformed field.)
  let huge = r#"{"block_ffn_dim_multiplier": 1e30, "hidden_size": 8,
    "num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 2}"#;
  assert!(matches!(
    TextConfig::from_json(huge),
    Err(Error::OutOfRange(_))
  ));
  // A non-finite multiplier (NaN / Inf) must be a NonFiniteScalar. serde's
  // JSON grammar has no NaN/Inf literal, so the field is set directly on an
  // otherwise-valid parsed config and validate() called.
  let base = r#"{"hidden_size": 8, "num_attention_heads": 2,
    "num_key_value_heads": 2, "num_hidden_layers": 2}"#;
  let mut nan_cfg = TextConfig::from_json(base).unwrap();
  nan_cfg.block_ffn_dim_multiplier = f32::NAN;
  assert!(matches!(nan_cfg.validate(), Err(Error::NonFiniteScalar(_))));
  let mut inf_cfg = TextConfig::from_json(base).unwrap();
  inf_cfg.block_ffn_dim_multiplier = f32::INFINITY;
  assert!(matches!(inf_cfg.validate(), Err(Error::NonFiniteScalar(_))));
  // A non-positive multiplier is rejected too (it would zero / negate the MLP
  // width).
  let neg = r#"{"block_ffn_dim_multiplier": -1.0, "hidden_size": 8,
    "num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 2}"#;
  assert!(matches!(
    TextConfig::from_json(neg),
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn validate_bounds_conv_l_cache_as_a_cardinality() {
  // `conv_L_cache` sizes runtime allocations (the conv-state array's middle
  // axis, the prefill left-pad, and a host `B * (conv_L_cache - 1)` index Vec),
  // so it takes the tight `MAX_CONV_L_CACHE` (256) cardinality cap, NOT the
  // loose 2^24 width cap. A value past the cap would otherwise drive a ~16M-wide
  // pad / a `B`-multiplied gigabyte host allocation before any typed error.
  let with_conv = |v: &str| {
    format!(
      r#"{{"conv_L_cache": {v}, "hidden_size": 8, "num_attention_heads": 2,
        "num_key_value_heads": 2, "num_hidden_layers": 2}}"#
    )
  };

  // The cap boundary: exactly MAX_CONV_L_CACHE is accepted, one past it is a
  // recoverable CapExceeded — isolating the off-by-one.
  assert!(
    TextConfig::from_json(&with_conv(&MAX_CONV_L_CACHE.to_string()))
      .unwrap()
      .validate()
      .is_ok()
  );
  assert!(matches!(
    TextConfig::from_json(&with_conv(&(MAX_CONV_L_CACHE + 1).to_string())),
    Err(Error::CapExceeded(_))
  ));
  // A value far past the old 2^24 width cap (which used to admit it) is now
  // rejected as CapExceeded, not OutOfRange.
  assert!(matches!(
    TextConfig::from_json(&with_conv("1000000")),
    Err(Error::CapExceeded(_))
  ));
  // The cardinality contract also rejects a non-positive window as OutOfRange
  // (a zero / negative kernel is structurally invalid, not merely over-cap).
  assert!(matches!(
    TextConfig::from_json(&with_conv("0")),
    Err(Error::OutOfRange(_))
  ));
  assert!(matches!(
    TextConfig::from_json(&with_conv("-1")),
    Err(Error::OutOfRange(_))
  ));
  // The realistic default (3) and a small explicit window stay valid.
  assert!(
    TextConfig::from_json(&with_conv("3"))
      .unwrap()
      .validate()
      .is_ok()
  );
}

// ───────────────────────── conv host index build ─────────────────────────

#[test]
fn lengths_positions_builds_clamped_index_rows() {
  // `lengths_positions(lengths, t, n_keep)` materializes the `[B, n_keep, 1]`
  // index `(clip(lengths, 0, t)[:, None] + arange(n_keep))[..., None]` consumed
  // by the cached-state `take_along_axis`. Verify the closed form directly
  // (independent of the code under test): row `b` is
  // `clip(lengths[b],0,t), .., clip(lengths[b],0,t)+n_keep-1`. Here `t == 5` is
  // at/above every length, so the clamp is a no-op.
  let lengths = [0_i32, 2, 5];
  let t = 5;
  let n_keep = 2;
  let mut idx = lengths_positions(&lengths, t, n_keep).unwrap();
  assert_eq!(idx.shape(), vec![3, 2, 1]);
  assert_eq!(idx.to_vec::<i32>().unwrap(), vec![0, 1, 2, 3, 5, 6]);

  // The `clip(_, 0, t)` is folded into the build: a length above `t` is clamped
  // down to `t`, a negative one up to `0`. With `t == 3`, `5 -> 3` and `-1 -> 0`.
  let clamped = [-1_i32, 2, 5];
  let mut clipped = lengths_positions(&clamped, 3, n_keep).unwrap();
  assert_eq!(clipped.to_vec::<i32>().unwrap(), vec![0, 1, 2, 3, 3, 4]);

  // `n_keep == 0` (a `conv_L_cache == 1` pointwise kernel) yields an empty
  // `[B, 0, 1]` index with no host allocation driven past zero.
  let mut empty = lengths_positions(&lengths, t, 0).unwrap();
  assert_eq!(empty.shape(), vec![3, 0, 1]);
  assert!(empty.to_vec::<i32>().unwrap().is_empty());
}

#[test]
fn lengths_positions_host_build_is_bounded_by_the_cap() {
  // The host index Vec is sized `B * (conv_L_cache - 1)`. The `conv_L_cache`
  // cardinality cap keeps the per-batch multiplier a small constant, so even a
  // large (but realistic) batch cannot drive an unbounded allocation: the
  // capacity computation is a checked `usize` multiply and the buffer is
  // fallibly reserved. Drive the largest within-cap window over a non-trivial
  // batch and confirm the closed-form shape / first-row contents.
  let n_keep = MAX_CONV_L_CACHE - 1; // the widest window the cap admits
  let batch = 64;
  let lengths: Vec<i32> = (0..batch).collect();
  // `t` at/above every length so the folded clamp is a no-op for this shape check.
  let mut idx = lengths_positions(&lengths, batch, n_keep).unwrap();
  assert_eq!(idx.shape(), vec![batch as usize, n_keep as usize, 1]);
  let flat = idx.to_vec::<i32>().unwrap();
  assert_eq!(flat.len(), batch as usize * n_keep as usize);
  // Row 0 is `0, 1, .., n_keep-1`; row 1 starts at `ends[1] == 1`.
  assert_eq!(flat[0], 0);
  assert_eq!(flat[n_keep as usize - 1], n_keep - 1);
  assert_eq!(flat[n_keep as usize], 1);
}

#[test]
fn shortconv_lengths_mismatch_with_batch_is_rejected() {
  // `cache.lengths` must carry exactly one entry per batch row. A mismatched
  // length would otherwise build a host index whose batch axis disagrees with
  // `Bx` and fail deep inside `take_along_axis`; `ShortConv::forward` rejects it
  // up front as a typed LengthMismatch. The input is batch 1, so a 2-entry (and
  // a 0-entry) `lengths` are both mismatches.
  let (in_w, conv_w, out_w, hidden, k) = sample_weights();
  let conv = make_shortconv(&in_w, &conv_w, &out_w, hidden, k);
  let x = sample_input(); // batch 1, L 5, hidden 2
  let arr = x_to_array(&x);
  let t = x[0].len() as i32;

  let mut too_many = ArraysCache::new(1);
  too_many.prepare(&[t, t]); // 2 entries vs batch 1
  assert!(matches!(
    conv.forward(&arr, None, Some(&mut too_many)),
    Err(Error::LengthMismatch(_))
  ));

  let mut too_few = ArraysCache::new(1);
  too_few.prepare(&[]); // 0 entries vs batch 1
  assert!(matches!(
    conv.forward(&arr, None, Some(&mut too_few)),
    Err(Error::LengthMismatch(_))
  ));

  // The matching-length path (1 entry, batch 1) is accepted.
  let mut ok = ArraysCache::new(1);
  ok.prepare(&[t]);
  assert!(conv.forward(&arr, None, Some(&mut ok)).is_ok());
}

// ───────────────────────── cache cardinality ─────────────────────────

/// The tiny all-conv config JSON (`hidden=4`, `vocab=8`, `K=3`, `ff=8`, no
/// `auto_adjust_ff_dim`, 2 layers) shared by the cardinality / conv-bias load
/// tests. `conv_bias` is templated in by [`tiny_all_conv_weights`].
const TINY_ALL_CONV_DIMS: (usize, usize, usize, usize) = (4, 8, 8, 3);

/// Build the flat weight map for a tiny all-conv 2-layer LFM2. When
/// `with_conv_bias` is set, each conv layer also carries `conv.bias`,
/// `in_proj.bias`, and `out_proj.bias` (the three `conv_bias`-gated tensors).
fn tiny_all_conv_weights(with_conv_bias: bool) -> std::collections::HashMap<String, Array> {
  use std::collections::HashMap;
  let (hidden, vocab, ff, k) = TINY_ALL_CONV_DIMS;

  // A deterministic `(rows, cols)` weight whose entries are tiny and distinct.
  let mat = |rows: usize, cols: usize| -> Array {
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01 - 0.1).collect();
    Array::from_slice::<f32>(&data, &(rows, cols)).unwrap()
  };
  let vecn = |n: usize| -> Array {
    let data: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.1).collect();
    Array::from_slice::<f32>(&data, &(n,)).unwrap()
  };

  let mut w: HashMap<String, Array> = HashMap::new();
  w.insert("model.embed_tokens.weight".to_string(), mat(vocab, hidden));
  w.insert("model.embedding_norm.weight".to_string(), vecn(hidden));
  for i in 0..2 {
    let p = format!("model.layers.{i}");
    w.insert(format!("{p}.operator_norm.weight"), vecn(hidden));
    w.insert(format!("{p}.ffn_norm.weight"), vecn(hidden));
    w.insert(format!("{p}.feed_forward.w1.weight"), mat(ff, hidden));
    w.insert(format!("{p}.feed_forward.w3.weight"), mat(ff, hidden));
    w.insert(format!("{p}.feed_forward.w2.weight"), mat(hidden, ff));
    // Conv layer (all-conv model): (hidden, K, 1) depthwise weight + projs.
    let conv_flat: Vec<f32> = (0..hidden * k).map(|i| (i as f32) * 0.02).collect();
    w.insert(
      format!("{p}.conv.conv.weight"),
      Array::from_slice::<f32>(&conv_flat, &(hidden, k, 1usize)).unwrap(),
    );
    w.insert(format!("{p}.conv.in_proj.weight"), mat(3 * hidden, hidden));
    w.insert(format!("{p}.conv.out_proj.weight"), mat(hidden, hidden));
    if with_conv_bias {
      // `conv.bias` is per-channel `(hidden,)`; the projection biases match
      // their output widths (`in_proj` -> 3*hidden, `out_proj` -> hidden).
      w.insert(format!("{p}.conv.conv.bias"), vecn(hidden));
      w.insert(format!("{p}.conv.in_proj.bias"), vecn(3 * hidden));
      w.insert(format!("{p}.conv.out_proj.bias"), vecn(hidden));
    }
  }
  w
}

/// The tiny all-conv config with the given `conv_bias` flag.
fn tiny_all_conv_config(conv_bias: bool) -> TextConfig {
  let json = format!(
    r#"{{"hidden_size": 4, "num_attention_heads": 2,
    "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 8,
    "conv_L_cache": 3, "block_auto_adjust_ff_dim": false, "block_ff_dim": 8,
    "conv_bias": {conv_bias}}}"#
  );
  TextConfig::from_json(&json).unwrap()
}

/// Build a tiny all-conv 2-layer [`Lfm2`] (`conv_bias=false`, no biases).
/// All-conv keeps the per-layer weight set to the conv + MLP + norms (no
/// attention QK-norm / RoPE projections), which is enough to exercise the
/// forward-pass cache-cardinality guard.
fn tiny_all_conv_model() -> Lfm2 {
  Lfm2::from_weights(tiny_all_conv_config(false), tiny_all_conv_weights(false)).unwrap()
}

#[test]
fn forward_rejects_wrong_cardinality_cache() {
  use crate::lm::model::Model as _;
  let model = tiny_all_conv_model();
  // A `[1, 2]` token window — two valid ids into the 8-row embedding table.
  let tokens = Array::from_slice::<i32>(&[1, 3], &(1usize, 2usize)).unwrap();

  // The correct cache has one entry per layer (2). A 1-entry cache must be a
  // recoverable LengthMismatch, NOT an out-of-bounds index panic.
  let mut short = model.make_cache();
  short.truncate(1);
  assert_eq!(short.len(), 1);
  assert!(matches!(
    model.forward(&tokens, &mut short),
    Err(Error::LengthMismatch(_))
  ));

  // An over-long cache (3 entries) is likewise rejected rather than silently
  // truncated by the layer `zip`.
  let mut long = model.make_cache();
  long.push(Box::new(ArraysCache::new(1)));
  assert_eq!(long.len(), 3);
  assert!(matches!(
    model.forward(&tokens, &mut long),
    Err(Error::LengthMismatch(_))
  ));

  // The matching-cardinality cache (2 entries) runs without error — the guard
  // only rejects the mismatched counts.
  let mut ok = model.make_cache();
  assert_eq!(ok.len(), 2);
  let logits = model.forward(&tokens, &mut ok).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, 8]);
}

#[test]
fn adjusted_ff_dim_matches_reference_formula() {
  // block_ff_dim=6656, multiple_of=256, multiplier=1.0, adjust=true:
  //   int(2*6656/3)=4437; int(1.0*4437)=4437;
  //   256 * ceil(4437/256) = 256 * 18 = 4608.
  assert_eq!(adjusted_ff_dim(6656, 256, true, 1.0).unwrap(), 4608);
  // adjust=false returns ff_dim unchanged.
  assert_eq!(adjusted_ff_dim(6656, 256, false, 1.0).unwrap(), 6656);
  // A non-unit multiplier: int(2*3000/3)=2000; int(0.5*2000)=1000;
  //   128 * ceil(1000/128) = 128 * 8 = 1024.
  assert_eq!(adjusted_ff_dim(3000, 128, true, 0.5).unwrap(), 1024);
}

#[test]
fn adjusted_ff_dim_rejects_overflowing_multiplier() {
  // A huge multiplier would saturate `(multiplier * base) as i32` to i32::MAX
  // and then overflow the `ff + multiple_of - 1` round-up. It must instead be
  // a recoverable OutOfRange, never a panic or a silently-wrapped width.
  assert!(matches!(
    adjusted_ff_dim(6656, 256, true, 1e30),
    Err(Error::OutOfRange(_))
  ));
  // A multiplier just past the i32 ceiling for this base (base = 2*6656/3 =
  // 4437; 4437 * multiplier > i32::MAX) is likewise rejected.
  let just_over = (i32::MAX as f32 / 4437.0) * 2.0;
  assert!(matches!(
    adjusted_ff_dim(6656, 256, true, just_over),
    Err(Error::OutOfRange(_))
  ));
}

// ───────────────────────── conv_bias gating ─────────────────────────

#[test]
fn conv_bias_true_loads_with_all_biases() {
  use crate::lm::model::Model as _;
  // conv_bias=true + all three bias tensors present per conv layer: loads and
  // runs (the bias is honored, not silently dropped).
  let model = Lfm2::from_weights(tiny_all_conv_config(true), tiny_all_conv_weights(true)).unwrap();
  let tokens = Array::from_slice::<i32>(&[1, 3], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, 8]);
}

#[test]
fn conv_bias_true_missing_bias_is_missing_key() {
  // conv_bias=true but a required bias tensor is absent: must be a typed
  // MissingKey (a silent run-without-bias would diverge from the reference).
  // Drop one bias from an otherwise-complete biased weight map.
  let mut w = tiny_all_conv_weights(true);
  assert!(w.remove("model.layers.0.conv.conv.bias").is_some());
  assert!(matches!(
    Lfm2::from_weights(tiny_all_conv_config(true), w),
    Err(Error::MissingKey(_))
  ));

  // Likewise for a missing projection bias.
  let mut w = tiny_all_conv_weights(true);
  assert!(w.remove("model.layers.1.conv.in_proj.bias").is_some());
  assert!(matches!(
    Lfm2::from_weights(tiny_all_conv_config(true), w),
    Err(Error::MissingKey(_))
  ));
}

#[test]
fn conv_bias_false_extra_bias_is_key_collision() {
  // conv_bias=false but a bias tensor IS present: must be a typed KeyCollision
  // (the stray bias would otherwise be silently applied). Add one bias to an
  // otherwise-biasless weight map.
  let mut w = tiny_all_conv_weights(false);
  let vecn = |n: usize| -> Array {
    let data: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.1).collect();
    Array::from_slice::<f32>(&data, &(n,)).unwrap()
  };
  w.insert("model.layers.0.conv.conv.bias".to_string(), vecn(4));
  assert!(matches!(
    Lfm2::from_weights(tiny_all_conv_config(false), w),
    Err(Error::KeyCollision(_))
  ));

  // A stray projection bias is likewise rejected.
  let mut w = tiny_all_conv_weights(false);
  w.insert("model.layers.1.conv.out_proj.bias".to_string(), vecn(4));
  assert!(matches!(
    Lfm2::from_weights(tiny_all_conv_config(false), w),
    Err(Error::KeyCollision(_))
  ));
}

#[test]
fn conv_bias_false_no_bias_loads() {
  // The complement of the collision case: conv_bias=false + no bias tensors is
  // the normal biasless path and loads cleanly.
  assert!(Lfm2::from_weights(tiny_all_conv_config(false), tiny_all_conv_weights(false)).is_ok());
}
