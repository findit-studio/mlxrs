//! Oracle tests for the Qwen3 LM blocks.
//!
//! Every numeric oracle is an independent `f64` reimplementation of the
//! documented `qwen3.py` math — never a call into the code under test:
//!
//! - [`attention_oracle`] hand-computes the full GQA attention for a tiny
//!   `head_dim=2` config (per-head Q/K RMSNorm, RoPE at positions 0/1 with
//!   `traditional=false`, the causal softmax, the GQA value repeat, and
//!   `o_proj`) — the load-bearing Qwen3-specific check.
//! - [`single_token_model_oracle`] hand-computes a whole 1-layer forward for a
//!   single token at position 0 (where RoPE is the identity and the single-key
//!   softmax is `1.0`, so the attention output is exactly the value
//!   projection), exercising the projections, residuals, SwiGLU MLP, final
//!   norm, and the tied LM head end to end.
//!
//! The structure tests drive the real public forward on a tiny config to
//! confirm the logits shape and that the GQA-repeat / Q-K-norm / RoPE /
//! tie-embeddings paths all execute on a multi-token sequence.

use std::collections::HashMap;

use super::{config::Qwen3Config, *};
use crate::{array::Array, error::Error};

const TOL: f32 = 1e-4;

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

// ════════════════════════════ small f64 helpers ════════════════════════════

/// `y[o] = sum_i x[i] * w[o][i]` — a dense `(out, in)` `nn.Linear` (no bias),
/// applied to a single feature vector.
fn linear_vec(x: &[f64], w: &[Vec<f64>]) -> Vec<f64> {
  w.iter()
    .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
    .collect()
}

/// RMSNorm over a single vector: `x / sqrt(mean(x^2) + eps) * weight`.
fn rms_norm_vec(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
  let n = x.len() as f64;
  let ms = x.iter().map(|v| v * v).sum::<f64>() / n;
  let inv = 1.0 / (ms + eps).sqrt();
  x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

/// Non-traditional RoPE on a single `head_dim`-vector at integer position `m`:
/// pairs feature `k` with `k + d/2`, rotating by `m * base^(-2k/d)`.
fn rope_vec(x: &[f64], pos: usize, base: f64) -> Vec<f64> {
  let d = x.len();
  let half = d / 2;
  let mut out = x.to_vec();
  for k in 0..half {
    let freq = base.powf(-2.0 * (k as f64) / (d as f64));
    let theta = (pos as f64) * freq;
    let (c, s) = (theta.cos(), theta.sin());
    let a = x[k];
    let b = x[k + half];
    out[k] = a * c - b * s;
    out[k + half] = b * c + a * s;
  }
  out
}

/// Softmax of a score row (f64).
fn softmax(row: &[f64]) -> Vec<f64> {
  let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  let exps: Vec<f64> = row.iter().map(|v| (v - m).exp()).collect();
  let sum: f64 = exps.iter().sum();
  exps.iter().map(|v| v / sum).collect()
}

// ════════════════════════════ attention oracle ════════════════════════════

/// Weights for a single [`Attention`] block: the four projections (all
/// `(out, in)` row-major, no bias) and the two per-head norms (`(head_dim,)`).
struct AttnWeights {
  q_proj: Vec<Vec<f64>>,
  k_proj: Vec<Vec<f64>>,
  v_proj: Vec<Vec<f64>>,
  o_proj: Vec<Vec<f64>>,
  q_norm: Vec<f64>,
  k_norm: Vec<f64>,
}

/// An independent `f64` reimplementation of [`Attention::forward`] for a single
/// batch row (`x` is `[L][hidden]`), derived directly from `qwen3.py:59-89`:
/// per-head reshape, q/k RMSNorm over `head_dim`, RoPE (`traditional=false`),
/// the causal softmax, the GQA value repeat, and `o_proj`. NOT a call into
/// `Attention::forward`.
#[allow(clippy::too_many_arguments)]
fn attention_oracle(
  x: &[Vec<f64>],
  w: &AttnWeights,
  n_heads: usize,
  n_kv_heads: usize,
  head_dim: usize,
  eps: f64,
  rope_base: f64,
) -> Vec<Vec<f64>> {
  let l = x.len();
  let scale = (head_dim as f64).powf(-0.5);
  let group = n_heads / n_kv_heads;

  // Project + per-head split + q/k-norm + RoPE for every position.
  // q[h][t] / k[kv][t] / v[kv][t] are head_dim-vectors.
  let mut q = vec![vec![vec![0.0; head_dim]; l]; n_heads];
  let mut k = vec![vec![vec![0.0; head_dim]; l]; n_kv_heads];
  let mut v = vec![vec![vec![0.0; head_dim]; l]; n_kv_heads];
  for (t, xt) in x.iter().enumerate() {
    let q_full = linear_vec(xt, &w.q_proj); // n_heads * head_dim
    let k_full = linear_vec(xt, &w.k_proj); // n_kv_heads * head_dim
    let v_full = linear_vec(xt, &w.v_proj); // n_kv_heads * head_dim
    for h in 0..n_heads {
      let head = &q_full[h * head_dim..(h + 1) * head_dim];
      let normed = rms_norm_vec(head, &w.q_norm, eps);
      q[h][t] = rope_vec(&normed, t, rope_base);
    }
    for kv in 0..n_kv_heads {
      let kh = &k_full[kv * head_dim..(kv + 1) * head_dim];
      let normed = rms_norm_vec(kh, &w.k_norm, eps);
      k[kv][t] = rope_vec(&normed, t, rope_base);
      v[kv][t] = v_full[kv * head_dim..(kv + 1) * head_dim].to_vec();
    }
  }

  // Per-head causal attention, GQA: head `h` reads kv-group `h / group`.
  // out_heads[h][t] is a head_dim-vector; concat over heads -> o_proj input.
  let mut concat = vec![vec![0.0; n_heads * head_dim]; l];
  for h in 0..n_heads {
    let kv = h / group;
    for t in 0..l {
      // Causal: query t attends to keys 0..=t.
      let scores: Vec<f64> = (0..=t)
        .map(|s| {
          q[h][t]
            .iter()
            .zip(&k[kv][s])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            * scale
        })
        .collect();
      let probs = softmax(&scores);
      let mut acc = vec![0.0; head_dim];
      for (s, p) in probs.iter().enumerate() {
        for d in 0..head_dim {
          acc[d] += p * v[kv][s][d];
        }
      }
      concat[t][h * head_dim..(h + 1) * head_dim].copy_from_slice(&acc);
    }
  }

  // o_proj per position.
  concat.iter().map(|c| linear_vec(c, &w.o_proj)).collect()
}

/// A deterministic `(rows, cols)` f64 weight whose entries are tiny + distinct.
fn mat(rows: usize, cols: usize, scale: f64, off: f64) -> Vec<Vec<f64>> {
  (0..rows)
    .map(|r| {
      (0..cols)
        .map(|c| ((r * cols + c) as f64) * scale + off)
        .collect()
    })
    .collect()
}

/// A deterministic `(n,)` f64 vector.
fn vecn(n: usize, base: f64, step: f64) -> Vec<f64> {
  (0..n).map(|i| base + i as f64 * step).collect()
}

/// `(rows, cols)` f64 matrix -> a `(rows, cols)` f32 [`Array`].
fn mat_to_array(w: &[Vec<f64>]) -> Array {
  let rows = w.len();
  let cols = w[0].len();
  let flat: Vec<f32> = w.iter().flat_map(|r| r.iter().map(|&v| v as f32)).collect();
  Array::from_slice::<f32>(&flat, &(rows, cols)).unwrap()
}

/// `(n,)` f64 vector -> a `(n,)` f32 [`Array`].
fn vec_to_array(v: &[f64]) -> Array {
  let data: Vec<f32> = v.iter().map(|&x| x as f32).collect();
  Array::from_slice::<f32>(&data, &(v.len(),)).unwrap()
}

/// `[L][hidden]` f64 -> a `[1, L, hidden]` f32 [`Array`] (batch 1).
fn seq_to_array(x: &[Vec<f64>]) -> Array {
  let l = x.len();
  let hidden = x[0].len();
  let flat: Vec<f32> = x.iter().flat_map(|t| t.iter().map(|&v| v as f32)).collect();
  Array::from_slice::<f32>(&flat, &(1usize, l, hidden)).unwrap()
}

#[test]
fn attention_matches_independent_gqa_oracle() {
  // head_dim = 2 (one RoPE rotation pair), n_heads = 4, n_kv_heads = 2 (group
  // 2), hidden = 3, L = 2. Exercises per-head Q/K-norm, RoPE at positions 0/1,
  // the causal softmax, and the GQA repeat — all under numeric scrutiny.
  let (hidden, head_dim, n_heads, n_kv_heads, l) = (3usize, 2usize, 4usize, 2usize, 2usize);
  let eps = 1e-6_f64;
  let rope_base = 1_000_000.0_f64;

  let w = AttnWeights {
    q_proj: mat(n_heads * head_dim, hidden, 0.03, -0.1), // (8, 3)
    k_proj: mat(n_kv_heads * head_dim, hidden, 0.05, -0.2), // (4, 3)
    v_proj: mat(n_kv_heads * head_dim, hidden, -0.04, 0.15), // (4, 3)
    o_proj: mat(hidden, n_heads * head_dim, 0.02, -0.05), // (3, 8)
    q_norm: vecn(head_dim, 1.1, 0.2),
    k_norm: vecn(head_dim, 0.9, -0.1),
  };
  let x = vec![vec![0.5, -0.3, 0.8], vec![-0.2, 0.6, 0.1]];

  let want = attention_oracle(&x, &w, n_heads, n_kv_heads, head_dim, eps, rope_base);

  let attn = Attention {
    n_heads: n_heads as i32,
    n_kv_heads: n_kv_heads as i32,
    head_dim: head_dim as i32,
    scale: (head_dim as f32).powf(-0.5),
    q_proj: Linear::new(mat_to_array(&w.q_proj), None),
    k_proj: Linear::new(mat_to_array(&w.k_proj), None),
    v_proj: Linear::new(mat_to_array(&w.v_proj), None),
    o_proj: Linear::new(mat_to_array(&w.o_proj), None),
    q_norm: RMSNorm::new(vec_to_array(&w.q_norm), eps as f32),
    k_norm: RMSNorm::new(vec_to_array(&w.k_norm), eps as f32),
    rope: Rope::new(head_dim as i32, false, rope_base as f32, 1.0),
  };
  let mut cache = StandardKvCache::new();
  // Prefill (offset 0) -> causal mask over the L=2 window.
  let mask = cache.make_mask(l, None, false).unwrap();
  let mut out = attn.forward(&seq_to_array(&x), &mask, &mut cache).unwrap();

  let want_flat: Vec<f32> = want
    .iter()
    .flat_map(|t| t.iter().map(|&v| v as f32))
    .collect();
  assert_eq!(out.shape(), vec![1, l, hidden]);
  assert_close(&out.to_vec::<f32>().unwrap(), &want_flat);
  // The cache advanced by the prefill length.
  assert_eq!(cache.offset(), l);
}

// ════════════════════════════ single-token model oracle ════════════════════

/// Tiny single-layer config dimensions: `(hidden, head_dim, n_heads,
/// n_kv_heads, intermediate, vocab)`.
const TINY: (usize, usize, usize, usize, usize, usize) = (4, 2, 2, 1, 6, 5);

/// The tied-head tiny config JSON (1 layer, the [`TINY`] dims).
fn tiny_config(tie: bool) -> Qwen3Config {
  let (hidden, head_dim, n_heads, n_kv_heads, inter, vocab) = TINY;
  let json = format!(
    r#"{{"hidden_size": {hidden}, "head_dim": {head_dim},
    "num_attention_heads": {n_heads}, "num_key_value_heads": {n_kv_heads},
    "num_hidden_layers": 1, "intermediate_size": {inter}, "vocab_size": {vocab},
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
    "tie_word_embeddings": {tie}}}"#
  );
  Qwen3Config::from_json(&json).unwrap()
}

/// All weights for a tiny single-layer Qwen3, returned both as the f32 weight
/// map (for the model) and as the f64 originals (for the oracle).
struct TinyWeights {
  embed: Vec<Vec<f64>>,           // (vocab, hidden)
  q_proj: Vec<Vec<f64>>,          // (n_heads*head_dim, hidden)
  k_proj: Vec<Vec<f64>>,          // (n_kv*head_dim, hidden)
  v_proj: Vec<Vec<f64>>,          // (n_kv*head_dim, hidden)
  o_proj: Vec<Vec<f64>>,          // (hidden, n_heads*head_dim)
  q_norm: Vec<f64>,               // (head_dim,)
  k_norm: Vec<f64>,               // (head_dim,)
  gate: Vec<Vec<f64>>,            // (inter, hidden)
  up: Vec<Vec<f64>>,              // (inter, hidden)
  down: Vec<Vec<f64>>,            // (hidden, inter)
  input_ln: Vec<f64>,             // (hidden,)
  post_ln: Vec<f64>,              // (hidden,)
  final_norm: Vec<f64>,           // (hidden,)
  lm_head: Option<Vec<Vec<f64>>>, // (vocab, hidden) when untied
}

fn tiny_weights(tie: bool) -> TinyWeights {
  let (hidden, head_dim, n_heads, n_kv_heads, inter, vocab) = TINY;
  TinyWeights {
    embed: mat(vocab, hidden, 0.07, -0.2),
    q_proj: mat(n_heads * head_dim, hidden, 0.03, -0.1),
    k_proj: mat(n_kv_heads * head_dim, hidden, 0.05, -0.15),
    v_proj: mat(n_kv_heads * head_dim, hidden, -0.04, 0.12),
    o_proj: mat(hidden, n_heads * head_dim, 0.02, -0.05),
    q_norm: vecn(head_dim, 1.1, 0.2),
    k_norm: vecn(head_dim, 0.9, -0.1),
    gate: mat(inter, hidden, 0.02, -0.08),
    up: mat(inter, hidden, -0.03, 0.1),
    down: mat(hidden, inter, 0.04, -0.06),
    input_ln: vecn(hidden, 1.0, 0.05),
    post_ln: vecn(hidden, 0.95, 0.03),
    final_norm: vecn(hidden, 1.05, -0.02),
    lm_head: if tie {
      None
    } else {
      Some(mat(vocab, hidden, -0.05, 0.2))
    },
  }
}

/// Build the flat f32 weight map from the f64 originals.
fn tiny_weight_map(w: &TinyWeights) -> HashMap<String, Array> {
  let mut m = HashMap::new();
  m.insert(
    "model.embed_tokens.weight".to_string(),
    mat_to_array(&w.embed),
  );
  m.insert("model.norm.weight".to_string(), vec_to_array(&w.final_norm));
  let p = "model.layers.0";
  m.insert(
    format!("{p}.self_attn.q_proj.weight"),
    mat_to_array(&w.q_proj),
  );
  m.insert(
    format!("{p}.self_attn.k_proj.weight"),
    mat_to_array(&w.k_proj),
  );
  m.insert(
    format!("{p}.self_attn.v_proj.weight"),
    mat_to_array(&w.v_proj),
  );
  m.insert(
    format!("{p}.self_attn.o_proj.weight"),
    mat_to_array(&w.o_proj),
  );
  m.insert(
    format!("{p}.self_attn.q_norm.weight"),
    vec_to_array(&w.q_norm),
  );
  m.insert(
    format!("{p}.self_attn.k_norm.weight"),
    vec_to_array(&w.k_norm),
  );
  m.insert(format!("{p}.mlp.gate_proj.weight"), mat_to_array(&w.gate));
  m.insert(format!("{p}.mlp.up_proj.weight"), mat_to_array(&w.up));
  m.insert(format!("{p}.mlp.down_proj.weight"), mat_to_array(&w.down));
  m.insert(
    format!("{p}.input_layernorm.weight"),
    vec_to_array(&w.input_ln),
  );
  m.insert(
    format!("{p}.post_attention_layernorm.weight"),
    vec_to_array(&w.post_ln),
  );
  if let Some(head) = &w.lm_head {
    m.insert("lm_head.weight".to_string(), mat_to_array(head));
  }
  m
}

/// `silu(x) = x * sigmoid(x)`.
fn silu(x: f64) -> f64 {
  x / (1.0 + (-x).exp())
}

/// Independent f64 forward of the whole 1-layer model for a SINGLE token at
/// position 0. With one position the causal softmax over a single key is `1.0`,
/// so the per-head attention output is exactly the value projection (q/k-norm +
/// RoPE do not change a length-1 softmax) — letting the rest of the pipeline
/// (projections, residuals, SwiGLU, final norm, tied/untied head) be checked in
/// closed form. Returns the `(vocab,)` logits.
fn single_token_model_oracle(token: usize, w: &TinyWeights) -> Vec<f64> {
  let (_hidden, head_dim, n_heads, n_kv_heads, _inter, _vocab) = TINY;
  let eps = 1e-6;
  let group = n_heads / n_kv_heads;

  // Embed.
  let x = w.embed[token].clone();

  // Attention: input norm, project, single-key softmax => attn head = v head.
  let normed = rms_norm_vec(&x, &w.input_ln, eps);
  let v_full = linear_vec(&normed, &w.v_proj);
  let mut concat = vec![0.0; n_heads * head_dim];
  for h in 0..n_heads {
    let kv = h / group;
    let v_head = &v_full[kv * head_dim..(kv + 1) * head_dim];
    concat[h * head_dim..(h + 1) * head_dim].copy_from_slice(v_head);
  }
  let attn = linear_vec(&concat, &w.o_proj);
  let hidden_state: Vec<f64> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();

  // MLP: post norm, SwiGLU, residual.
  let normed2 = rms_norm_vec(&hidden_state, &w.post_ln, eps);
  let gate = linear_vec(&normed2, &w.gate);
  let up = linear_vec(&normed2, &w.up);
  let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
  let mlp = linear_vec(&act, &w.down);
  let out: Vec<f64> = hidden_state.iter().zip(&mlp).map(|(a, b)| a + b).collect();

  // Final norm, then the LM head (tied = embed, untied = lm_head).
  let final_h = rms_norm_vec(&out, &w.final_norm, eps);
  let head = w.lm_head.as_ref().unwrap_or(&w.embed);
  linear_vec(&final_h, head)
}

#[test]
fn single_token_forward_matches_oracle_tied() {
  let w = tiny_weights(true);
  let model = Qwen3::from_weights(tiny_config(true), tiny_weight_map(&w)).unwrap();
  let token = 3usize;
  let tokens = Array::from_slice::<i32>(&[token as i32], &(1usize, 1usize)).unwrap();
  let mut cache = model.make_cache();
  let mut logits = model.forward(&tokens, &mut cache).unwrap();

  assert_eq!(logits.shape(), vec![1, 1, TINY.5]);
  let want: Vec<f32> = single_token_model_oracle(token, &w)
    .iter()
    .map(|&v| v as f32)
    .collect();
  assert_close(&logits.to_vec::<f32>().unwrap(), &want);
}

#[test]
fn single_token_forward_matches_oracle_untied() {
  // The untied head must use `lm_head.weight`, NOT the embedding table.
  let w = tiny_weights(false);
  let model = Qwen3::from_weights(tiny_config(false), tiny_weight_map(&w)).unwrap();
  let token = 2usize;
  let tokens = Array::from_slice::<i32>(&[token as i32], &(1usize, 1usize)).unwrap();
  let mut cache = model.make_cache();
  let mut logits = model.forward(&tokens, &mut cache).unwrap();

  let want: Vec<f32> = single_token_model_oracle(token, &w)
    .iter()
    .map(|&v| v as f32)
    .collect();
  assert_close(&logits.to_vec::<f32>().unwrap(), &want);
}

// ════════════════════════════ structure / shape ════════════════════════════

#[test]
fn forward_returns_expected_logits_shape() {
  // A multi-token (L=2) prefill on the tiny tied config: the GQA-repeat,
  // per-head Q/K-norm, RoPE, causal mask, and tied-head paths all execute and
  // the logits shape is exactly `[B, L, vocab]`.
  let w = tiny_weights(true);
  let model = Qwen3::from_weights(tiny_config(true), tiny_weight_map(&w)).unwrap();
  let tokens = Array::from_slice::<i32>(&[1, 4], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  assert_eq!(cache.len(), 1);
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, TINY.5]);
  // Every layer's cache advanced by the prefill length.
  assert!(cache.iter().all(|c| c.offset() == 2));
}

/// The dense Qwen3 forward must preserve the activation dtype: a `bf16`/`f16`
/// checkpoint must yield `bf16`/`f16` logits, never a silent promotion to
/// `f32`. The standard-RoPE path uses the fused dtype-preserving rotary, so this
/// is a regression guard that nothing in the dense path reintroduces an f32
/// constant combined with the activations (the class fixed in the ASR towers).
fn assert_dense_forward_preserves_dtype(dtype: crate::Dtype) {
  let w = tiny_weights(true);
  let weights: HashMap<String, Array> = tiny_weight_map(&w)
    .into_iter()
    .map(|(k, v)| (k, v.astype(dtype).expect("weight cast")))
    .collect();
  let model = Qwen3::from_weights(tiny_config(true), weights).unwrap();
  let tokens = Array::from_slice::<i32>(&[1, 4], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(
    logits.dtype().unwrap(),
    dtype,
    "dense Qwen3 forward upcast {dtype:?} → {:?}",
    logits.dtype().unwrap()
  );
  // Read via an f32 view (`to_vec::<f32>` requires an f32 array).
  let vals = logits
    .astype(crate::Dtype::F32)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
}

#[test]
fn forward_preserves_bf16() {
  assert_dense_forward_preserves_dtype(crate::Dtype::BF16);
}

#[test]
fn forward_preserves_f16() {
  assert_dense_forward_preserves_dtype(crate::Dtype::F16);
}

#[test]
fn prefill_then_step_advances_cache() {
  // Prefill 2 tokens, then a single decode step: the cache offset advances
  // 0 -> 2 -> 3 and each forward returns the expected per-call logits shape.
  let w = tiny_weights(true);
  let model = Qwen3::from_weights(tiny_config(true), tiny_weight_map(&w)).unwrap();
  let mut cache = model.make_cache();

  let prefill = Array::from_slice::<i32>(&[1, 4], &(1usize, 2usize)).unwrap();
  let logits = model.forward(&prefill, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, TINY.5]);
  assert!(cache.iter().all(|c| c.offset() == 2));

  let step = Array::from_slice::<i32>(&[2], &(1usize, 1usize)).unwrap();
  let logits = model.forward(&step, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 1, TINY.5]);
  assert!(cache.iter().all(|c| c.offset() == 3));
}

#[test]
fn multi_layer_two_kv_groups_forward_runs() {
  // A 2-layer config with n_heads=4 / n_kv_heads=2 (GQA group 2), batch 2.
  // Confirms the per-layer loop + GQA grouping + batched embedding lookup run
  // and produce `[B, L, vocab]` logits.
  let json = r#"{"hidden_size": 8, "head_dim": 4, "num_attention_heads": 4,
    "num_key_value_heads": 2, "num_hidden_layers": 2, "intermediate_size": 16,
    "vocab_size": 12, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
    "tie_word_embeddings": true}"#;
  let cfg = Qwen3Config::from_json(json).unwrap();

  let (hidden, head_dim, n_heads, n_kv, inter, vocab) = (8, 4, 4, 2, 16, 12);
  let mut m = HashMap::new();
  let mat_f =
    |rows: usize, cols: usize, off: f64| -> Array { mat_to_array(&mat(rows, cols, 0.01, off)) };
  let vec_f = |n: usize| -> Array { vec_to_array(&vecn(n, 1.0, 0.01)) };
  m.insert(
    "model.embed_tokens.weight".to_string(),
    mat_f(vocab, hidden, -0.1),
  );
  m.insert("model.norm.weight".to_string(), vec_f(hidden));
  for i in 0..2 {
    let p = format!("model.layers.{i}");
    m.insert(
      format!("{p}.self_attn.q_proj.weight"),
      mat_f(n_heads * head_dim, hidden, -0.05),
    );
    m.insert(
      format!("{p}.self_attn.k_proj.weight"),
      mat_f(n_kv * head_dim, hidden, -0.04),
    );
    m.insert(
      format!("{p}.self_attn.v_proj.weight"),
      mat_f(n_kv * head_dim, hidden, 0.03),
    );
    m.insert(
      format!("{p}.self_attn.o_proj.weight"),
      mat_f(hidden, n_heads * head_dim, -0.02),
    );
    m.insert(format!("{p}.self_attn.q_norm.weight"), vec_f(head_dim));
    m.insert(format!("{p}.self_attn.k_norm.weight"), vec_f(head_dim));
    m.insert(
      format!("{p}.mlp.gate_proj.weight"),
      mat_f(inter, hidden, -0.03),
    );
    m.insert(
      format!("{p}.mlp.up_proj.weight"),
      mat_f(inter, hidden, 0.02),
    );
    m.insert(
      format!("{p}.mlp.down_proj.weight"),
      mat_f(hidden, inter, -0.01),
    );
    m.insert(format!("{p}.input_layernorm.weight"), vec_f(hidden));
    m.insert(
      format!("{p}.post_attention_layernorm.weight"),
      vec_f(hidden),
    );
  }
  let model = Qwen3::from_weights(cfg, m).unwrap();

  // Batch 2, L 3 — two distinct rows of valid token ids.
  let tokens = Array::from_slice::<i32>(&[0, 1, 2, 3, 4, 5], &(2usize, 3usize)).unwrap();
  let mut cache = model.make_cache();
  assert_eq!(cache.len(), 2);
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![2, 3, vocab]);
}

// ════════════════════════════ cache cardinality ════════════════════════════

#[test]
fn forward_rejects_wrong_cardinality_cache() {
  use crate::lm::cache::StandardKvCache;
  let w = tiny_weights(true);
  let model = Qwen3::from_weights(tiny_config(true), tiny_weight_map(&w)).unwrap();
  let tokens = Array::from_slice::<i32>(&[1, 3], &(1usize, 2usize)).unwrap();

  // The correct cache has one entry per layer (1). An empty cache must be a
  // recoverable LengthMismatch, NOT an out-of-bounds index panic.
  let mut empty: Vec<Box<dyn KvCache>> = Vec::new();
  assert!(matches!(
    model.forward(&tokens, &mut empty),
    Err(Error::LengthMismatch(_))
  ));

  // An over-long cache (2 entries vs 1 layer) is likewise rejected.
  let mut long = model.make_cache();
  long.push(Box::new(StandardKvCache::new()));
  assert_eq!(long.len(), 2);
  assert!(matches!(
    model.forward(&tokens, &mut long),
    Err(Error::LengthMismatch(_))
  ));
}

// ════════════════════════════ weight loading ════════════════════════════

#[test]
fn from_weights_missing_key_is_typed_error() {
  let w = tiny_weights(true);
  let mut map = tiny_weight_map(&w);
  assert!(
    map
      .remove("model.layers.0.self_attn.q_norm.weight")
      .is_some()
  );
  assert!(matches!(
    Qwen3::from_weights(tiny_config(true), map),
    Err(Error::MissingKey(_))
  ));
}

#[test]
fn untied_missing_lm_head_is_missing_key() {
  // An untied config without `lm_head.weight` must be a typed MissingKey.
  let w = tiny_weights(false);
  let mut map = tiny_weight_map(&w);
  assert!(map.remove("lm_head.weight").is_some());
  assert!(matches!(
    Qwen3::from_weights(tiny_config(false), map),
    Err(Error::MissingKey(_))
  ));
}

#[test]
fn sanitize_drops_tied_lm_head() {
  // sanitize drops a stray `lm_head.weight` when tied (the tied head reuses the
  // embedding table), and leaves it when untied.
  let cfg_tied = tiny_config(true);
  let mut map = tiny_weight_map(&tiny_weights(true));
  map.insert(
    "lm_head.weight".to_string(),
    mat_to_array(&mat(TINY.5, TINY.0, 0.01, 0.0)),
  );
  Qwen3::sanitize(&cfg_tied, &mut map);
  assert!(!map.contains_key("lm_head.weight"));
  // The sanitized tied map still loads (the stray head was the only extra key).
  assert!(Qwen3::from_weights(cfg_tied, map).is_ok());

  let cfg_untied = tiny_config(false);
  let mut map2 = tiny_weight_map(&tiny_weights(false));
  Qwen3::sanitize(&cfg_untied, &mut map2);
  assert!(map2.contains_key("lm_head.weight"));
}

// ════════════════════════════ config validation ════════════════════════════

#[test]
fn config_defaults_and_overrides() {
  let cfg = Qwen3Config::from_json("{}").unwrap();
  assert_eq!(cfg.model_type, "qwen3");
  assert_eq!(cfg.hidden_size, 2048);
  assert_eq!(cfg.num_attention_heads, 16);
  assert_eq!(cfg.num_key_value_heads, 8);
  assert_eq!(cfg.head_dim, 128);
  assert_eq!(cfg.vocab_size, 151936);
  assert_eq!(cfg.max_position_embeddings, 65536);
  assert_eq!(cfg.rope_theta, 1_000_000.0);
  assert!(cfg.tie_word_embeddings);

  let json = r#"{"hidden_size": 1024, "num_attention_heads": 8,
    "num_key_value_heads": 2, "head_dim": 64, "tie_word_embeddings": false}"#;
  let cfg = Qwen3Config::from_json(json).unwrap();
  assert_eq!(cfg.hidden_size, 1024);
  assert_eq!(cfg.num_key_value_heads, 2);
  assert_eq!(cfg.head_dim, 64);
  assert!(!cfg.tie_word_embeddings);
}

#[test]
fn default_config_passes_validation() {
  assert!(Qwen3Config::from_json("{}").unwrap().validate().is_ok());
}

#[test]
fn validate_rejects_zero_negative_and_nondivisible() {
  // Zero / negative width.
  let zero = r#"{"hidden_size": 0}"#;
  assert!(matches!(
    Qwen3Config::from_json(zero),
    Err(Error::OutOfRange(_))
  ));
  let neg = r#"{"vocab_size": -8}"#;
  assert!(matches!(
    Qwen3Config::from_json(neg),
    Err(Error::OutOfRange(_))
  ));
  // GQA: n_heads not divisible by n_kv_heads.
  let nondiv = r#"{"num_attention_heads": 6, "num_key_value_heads": 4}"#;
  assert!(matches!(
    Qwen3Config::from_json(nondiv),
    Err(Error::DivisibilityConstraint(_))
  ));
}

#[test]
fn validate_rejects_odd_head_dim() {
  // RoPE pairs k with k+head_dim/2, so an odd head_dim is rejected.
  let odd = r#"{"head_dim": 65}"#;
  assert!(matches!(
    Qwen3Config::from_json(odd),
    Err(Error::OutOfRange(_))
  ));
  let even = r#"{"head_dim": 64}"#;
  assert!(Qwen3Config::from_json(even).unwrap().validate().is_ok());
}

#[test]
fn validate_rejects_huge_layer_and_head_counts() {
  let huge_layers = r#"{"num_hidden_layers": 100000}"#;
  assert!(matches!(
    Qwen3Config::from_json(huge_layers),
    Err(Error::CapExceeded(_))
  ));
  let huge_heads = r#"{"num_attention_heads": 100000, "num_key_value_heads": 100000}"#;
  assert!(matches!(
    Qwen3Config::from_json(huge_heads),
    Err(Error::CapExceeded(_))
  ));
  // Oversized width (> 2^24).
  let oversized = r#"{"hidden_size": 33554433}"#;
  assert!(matches!(
    Qwen3Config::from_json(oversized),
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn validate_rejects_overflowing_query_width() {
  // num_attention_heads * head_dim must stay within the width cap even when
  // each factor alone is within its own cap (heads <= 4096 cardinality,
  // head_dim <= 2^24 width): 4096 heads * 8192 head_dim = 2^25 > 2^24, a
  // recoverable OutOfRange (NOT a wrapping multiply or an overflow panic).
  let json = r#"{"num_attention_heads": 4096, "num_key_value_heads": 4096,
    "head_dim": 8192}"#;
  assert!(matches!(
    Qwen3Config::from_json(json),
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn validate_rejects_nonfinite_and_nonpositive_scalars() {
  // rope_theta must be finite + positive; rms_norm_eps must be finite.
  let mut cfg = Qwen3Config::from_json("{}").unwrap();
  cfg.rope_theta = f32::NAN;
  assert!(matches!(cfg.validate(), Err(Error::NonFiniteScalar(_))));

  let mut cfg = Qwen3Config::from_json("{}").unwrap();
  cfg.rope_theta = -1.0;
  assert!(matches!(cfg.validate(), Err(Error::OutOfRange(_))));

  let mut cfg = Qwen3Config::from_json("{}").unwrap();
  cfg.rms_norm_eps = f32::INFINITY;
  assert!(matches!(cfg.validate(), Err(Error::NonFiniteScalar(_))));
}

#[test]
fn validate_rejects_nonnull_rope_scaling() {
  // A non-null rope_scaling is a later-phase feature; reject it rather than
  // silently applying unscaled RoPE.
  let json = r#"{"rope_scaling": {"type": "linear", "factor": 2.0}}"#;
  assert!(matches!(
    Qwen3Config::from_json(json),
    Err(Error::OutOfRange(_))
  ));
  // null / absent rope_scaling is accepted.
  let null_json = r#"{"rope_scaling": null}"#;
  assert!(Qwen3Config::from_json(null_json).is_ok());
}

#[test]
fn from_json_rejects_malformed_json() {
  assert!(matches!(
    Qwen3Config::from_json("{not json"),
    Err(Error::Parse(_))
  ));
}
