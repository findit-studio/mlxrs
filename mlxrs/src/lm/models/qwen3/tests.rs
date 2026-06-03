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
    q_proj: Linear::dense(mat_to_array(&w.q_proj)),
    k_proj: Linear::dense(mat_to_array(&w.k_proj)),
    v_proj: Linear::dense(mat_to_array(&w.v_proj)),
    o_proj: Linear::dense(mat_to_array(&w.o_proj)),
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

// ════════════════════════ token-id range / embed table ════════════════════

#[test]
fn embed_tokens_rejects_out_of_range_ids() {
  // The `embed_tokens` gather (MLX `take`) does not bound-check, so a negative id
  // or an id `>= vocab` is an out-of-bounds embedding read (UB). Both must be a
  // typed OutOfRange before the gather, and a valid id must still embed.
  let w = tiny_weights(true);
  let model = Qwen3::from_weights(tiny_config(true), tiny_weight_map(&w)).unwrap();
  let vocab = TINY.5 as i32; // 5

  // A negative id.
  let neg = Array::from_slice::<i32>(&[-1], &(1usize, 1usize)).unwrap();
  assert!(matches!(
    model.model().embed_tokens(&neg),
    Err(Error::OutOfRange(_))
  ));

  // An id exactly one past the end (== vocab).
  let past = Array::from_slice::<i32>(&[vocab], &(1usize, 1usize)).unwrap();
  assert!(matches!(
    model.model().embed_tokens(&past),
    Err(Error::OutOfRange(_))
  ));

  // The same out-of-range id reached through the public `forward` (which embeds
  // via the same gather) is likewise a typed error, never an OOB read.
  let mut cache = model.make_cache();
  assert!(matches!(
    model.forward(&past, &mut cache),
    Err(Error::OutOfRange(_))
  ));

  // A valid id (in `[0, vocab)`) still embeds to `(B, L, hidden)`.
  let ok = Array::from_slice::<i32>(&[vocab - 1], &(1usize, 1usize)).unwrap();
  let mut embedded = model.model().embed_tokens(&ok).unwrap();
  assert_eq!(embedded.shape(), vec![1, 1, TINY.0]);
  assert!(
    embedded
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite())
  );
}

#[test]
fn from_weights_rejects_undersized_embedding_table() {
  // A checkpoint whose embedding table has FEWER rows than `config.vocab_size`
  // must be rejected on load (else an id in `[rows, vocab_size)` — which
  // `embed_tokens` admits against `vocab_size` — would gather out of bounds).
  let w = tiny_weights(true);
  let mut map = tiny_weight_map(&w);
  // Replace the embedding table with one short of `vocab_size` rows.
  let short = mat(TINY.5 - 1, TINY.0, 0.07, -0.2); // (vocab-1, hidden)
  map.insert(
    "model.embed_tokens.weight".to_string(),
    mat_to_array(&short),
  );
  assert!(matches!(
    Qwen3::from_weights(tiny_config(true), map),
    Err(Error::LayerKeyed(_))
  ));
}

#[test]
fn from_weights_rejects_wrong_embedding_width() {
  // The embedding table's axis-1 (hidden) must equal `config.hidden_size`.
  let w = tiny_weights(true);
  let mut map = tiny_weight_map(&w);
  let wide = mat(TINY.5, TINY.0 + 1, 0.07, -0.2); // (vocab, hidden+1)
  map.insert("model.embed_tokens.weight".to_string(), mat_to_array(&wide));
  assert!(matches!(
    Qwen3::from_weights(tiny_config(true), map),
    Err(Error::LayerKeyed(_))
  ));
}

// ════════════════════════ forward_embeddings rank / width ═══════════════════

#[test]
fn forward_hidden_rejects_non_rank3() {
  // `forward_hidden` (reachable via `forward_embeddings`, whose
  // `supports_input_embeddings` is true) is public and accepts an arbitrary
  // `Array`; a rank-0 / rank-1 / rank-2 input must surface a typed RankMismatch,
  // never an index-out-of-bounds panic on `shape[1]`.
  let w = tiny_weights(true);
  let model = Qwen3::from_weights(tiny_config(true), tiny_weight_map(&w)).unwrap();
  let hidden = TINY.0 as i32;
  for shape in [vec![], vec![hidden], vec![1, hidden]] {
    let mut cache = model.make_cache();
    let h = Array::full::<f32>(&shape, 0.1).unwrap();
    assert!(
      matches!(
        model.forward_embeddings(&h, &mut cache),
        Err(Error::RankMismatch(_))
      ),
      "rank-{} embeddings must be a RankMismatch",
      shape.len()
    );
  }
}

#[test]
fn forward_hidden_rejects_wrong_width() {
  // A rank-3 input whose hidden axis disagrees with the token-embedding width
  // must be a typed ShapePairMismatch, not a downstream matmul panic.
  let w = tiny_weights(true);
  let model = Qwen3::from_weights(tiny_config(true), tiny_weight_map(&w)).unwrap();
  let mut cache = model.make_cache();
  let h = Array::full::<f32>(&[1, 2, TINY.0 as i32 + 1], 0.1).unwrap();
  assert!(matches!(
    model.forward_embeddings(&h, &mut cache),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn forward_hidden_accepts_correct_rank3() {
  // A correct rank-3 `[batch, seq, hidden]` input still runs end to end and
  // returns `[batch, seq, vocab]` logits.
  let w = tiny_weights(true);
  let model = Qwen3::from_weights(tiny_config(true), tiny_weight_map(&w)).unwrap();
  let mut cache = model.make_cache();
  let h = Array::full::<f32>(&[1, 2, TINY.0 as i32], 0.1).unwrap();
  let logits = model.forward_embeddings(&h, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, TINY.5]);
}

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

// ════════════════════ decoder weight shape validation ═══════════════════════

/// Replace one weight in a tied tiny map with `array`, then assert
/// [`Qwen3::from_weights`] rejects it with a typed [`Error::LayerKeyed`] —
/// i.e. the malformed shape is caught at load, not deferred into a later
/// matmul / reshape / RMSNorm (or run as a different broadcastable graph).
fn assert_tied_weight_shape_rejected(key: &str, array: Array) {
  let mut map = tiny_weight_map(&tiny_weights(true));
  assert!(
    map.insert(key.to_string(), array).is_some(),
    "{key} not in map"
  );
  assert!(
    matches!(
      Qwen3::from_weights(tiny_config(true), map),
      Err(Error::LayerKeyed(_))
    ),
    "{key} with a bad shape must be a typed LayerKeyed load error"
  );
}

#[test]
fn from_weights_rejects_wrong_dimension_decoder_weights() {
  // A representative sample of every dense decoder weight class — a q_proj, a
  // layer norm, an MLP gate, an MLP down, and the final norm — each with a
  // wrong DIMENSION (right rank) must be a typed load-time error, not a deferred
  // MLX failure. (The same `take_shaped` pins every other layer tensor too.)
  let (hidden, head_dim, n_heads, _n_kv, inter, _vocab) = TINY;
  let p = "model.layers.0";

  // q_proj: rows must be n_heads * head_dim; widen the rows by one head_dim.
  assert_tied_weight_shape_rejected(
    &format!("{p}.self_attn.q_proj.weight"),
    mat_to_array(&mat((n_heads + 1) * head_dim, hidden, 0.03, -0.1)),
  );
  // input_layernorm: must be (hidden,); make it one wider.
  assert_tied_weight_shape_rejected(
    &format!("{p}.input_layernorm.weight"),
    vec_to_array(&vecn(hidden + 1, 1.0, 0.05)),
  );
  // mlp gate_proj: rows must be intermediate; make it one short.
  assert_tied_weight_shape_rejected(
    &format!("{p}.mlp.gate_proj.weight"),
    mat_to_array(&mat(inter - 1, hidden, 0.02, -0.08)),
  );
  // mlp down_proj: cols must be intermediate; make them one wider.
  assert_tied_weight_shape_rejected(
    &format!("{p}.mlp.down_proj.weight"),
    mat_to_array(&mat(hidden, inter + 1, 0.04, -0.06)),
  );
  // final norm: must be (hidden,); make it one short.
  assert_tied_weight_shape_rejected(
    "model.norm.weight",
    vec_to_array(&vecn(hidden - 1, 1.05, -0.02)),
  );
}

#[test]
fn from_weights_rejects_wrong_rank_decoder_weights() {
  // The same representative sample, each with a wrong RANK, must be rejected on
  // load by the shape check rather than reaching the forward.
  let (hidden, head_dim, n_heads, _n_kv, inter, _vocab) = TINY;
  let p = "model.layers.0";

  // q_proj as a rank-1 vector instead of (n_heads*head_dim, hidden).
  assert_tied_weight_shape_rejected(
    &format!("{p}.self_attn.q_proj.weight"),
    vec_to_array(&vecn(n_heads * head_dim, 0.03, -0.1)),
  );
  // input_layernorm as a rank-2 matrix instead of (hidden,).
  assert_tied_weight_shape_rejected(
    &format!("{p}.input_layernorm.weight"),
    mat_to_array(&mat(hidden, 1, 1.0, 0.05)),
  );
  // mlp gate_proj as a rank-1 vector instead of (intermediate, hidden).
  assert_tied_weight_shape_rejected(
    &format!("{p}.mlp.gate_proj.weight"),
    vec_to_array(&vecn(inter, 0.02, -0.08)),
  );
  // mlp down_proj as a rank-1 vector instead of (hidden, intermediate).
  assert_tied_weight_shape_rejected(
    &format!("{p}.mlp.down_proj.weight"),
    vec_to_array(&vecn(hidden, 0.04, -0.06)),
  );
  // final norm as a rank-2 matrix instead of (hidden,).
  assert_tied_weight_shape_rejected(
    "model.norm.weight",
    mat_to_array(&mat(hidden, 1, 1.05, -0.02)),
  );
}

#[test]
fn from_weights_accepts_fully_correct_tied_checkpoint() {
  // A fully correctly-shaped tied checkpoint still loads and forwards to
  // `[B, S, vocab]` — the shape pins reject only malformed weights.
  let w = tiny_weights(true);
  let model = Qwen3::from_weights(tiny_config(true), tiny_weight_map(&w)).unwrap();
  let mut cache = model.make_cache();
  let tokens = Array::from_slice::<i32>(&[1, 4, 2], &(1usize, 3usize)).unwrap();
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 3, TINY.5]);
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
fn from_weights_rejects_oversized_untied_lm_head() {
  // An untied `lm_head.weight` with MORE rows than `config.vocab_size` would emit
  // `(B, S, vocab_size + 1)` logits — token ids outside the configured vocab —
  // breaking the `Model` contract; it must be rejected on load like the
  // embedding table.
  let w = tiny_weights(false);
  let mut map = tiny_weight_map(&w);
  let tall = mat(TINY.5 + 1, TINY.0, -0.05, 0.2); // (vocab+1, hidden)
  map.insert("lm_head.weight".to_string(), mat_to_array(&tall));
  assert!(matches!(
    Qwen3::from_weights(tiny_config(false), map),
    Err(Error::LayerKeyed(_))
  ));
}

#[test]
fn from_weights_rejects_undersized_untied_lm_head() {
  // An untied `lm_head.weight` with FEWER rows than `config.vocab_size` must be
  // rejected on load (the head's row count is the logits' vocab axis).
  let w = tiny_weights(false);
  let mut map = tiny_weight_map(&w);
  let short = mat(TINY.5 - 1, TINY.0, -0.05, 0.2); // (vocab-1, hidden)
  map.insert("lm_head.weight".to_string(), mat_to_array(&short));
  assert!(matches!(
    Qwen3::from_weights(tiny_config(false), map),
    Err(Error::LayerKeyed(_))
  ));
}

#[test]
fn from_weights_rejects_wrong_untied_lm_head_width() {
  // The untied head's axis-1 (hidden) must equal `config.hidden_size`; a wrong
  // width must be a typed load-time error here, not a deferred matmul failure in
  // `Linear::forward`.
  let w = tiny_weights(false);
  let mut map = tiny_weight_map(&w);
  let wide = mat(TINY.5, TINY.0 + 1, -0.05, 0.2); // (vocab, hidden+1)
  map.insert("lm_head.weight".to_string(), mat_to_array(&wide));
  assert!(matches!(
    Qwen3::from_weights(tiny_config(false), map),
    Err(Error::LayerKeyed(_))
  ));
}

#[test]
fn from_weights_rejects_wrong_rank_untied_lm_head() {
  // A `lm_head.weight` that is not rank 2 must be rejected on load by the same
  // shape check, not reach `Linear::forward`.
  let w = tiny_weights(false);
  let mut map = tiny_weight_map(&w);
  // A rank-1 `(vocab,)` tensor.
  map.insert(
    "lm_head.weight".to_string(),
    vec_to_array(&vecn(TINY.5, -0.05, 0.2)),
  );
  assert!(matches!(
    Qwen3::from_weights(tiny_config(false), map),
    Err(Error::LayerKeyed(_))
  ));
}

#[test]
fn from_weights_accepts_correct_untied_lm_head() {
  // A correctly shaped `(vocab_size, hidden_size)` untied head loads and forwards
  // to `(B, S, vocab_size)` logits.
  let w = tiny_weights(false);
  let model = Qwen3::from_weights(tiny_config(false), tiny_weight_map(&w)).unwrap();
  let mut cache = model.make_cache();
  let tokens = Array::from_slice::<i32>(&[1, 2], &(1usize, 2usize)).unwrap();
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, TINY.5]);
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

// ════════════ public-constructor config gate (bypassing from_json) ══════════

#[test]
fn model_from_weights_rejects_zero_layers_before_weights() {
  // `Qwen3Model::from_weights` is public and `Qwen3Config`'s fields are public,
  // so a caller can mutate a parsed (valid) config to `num_hidden_layers == 0`
  // and call the constructor directly. Without an up-front `validate` this would
  // load a norm-only decoder skipping every per-layer required weight; with it,
  // the config is rejected with a typed `OutOfRange` BEFORE any weight is
  // consulted. The empty weight map proves the rejection precedes the first
  // `take_shaped` (otherwise the missing embedding table would surface a
  // `MissingKey` instead).
  let mut cfg = tiny_config(true);
  cfg.num_hidden_layers = 0;
  let mut weights: HashMap<String, Array> = HashMap::new();
  assert!(matches!(
    Qwen3Model::from_weights(&cfg, &mut weights, None),
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn model_from_weights_rejects_zero_head_counts_before_weights() {
  // A zero attention- or kv-head count derives zero-width projections that would
  // otherwise fail later in `reshape` / SDPA; the config gate rejects each with a
  // typed `OutOfRange` at load, ahead of consulting any weight (empty map).
  for mutate in [
    (|c: &mut Qwen3Config| c.num_attention_heads = 0) as fn(&mut Qwen3Config),
    (|c: &mut Qwen3Config| c.num_key_value_heads = 0) as fn(&mut Qwen3Config),
  ] {
    let mut cfg = tiny_config(true);
    mutate(&mut cfg);
    let mut weights: HashMap<String, Array> = HashMap::new();
    assert!(matches!(
      Qwen3Model::from_weights(&cfg, &mut weights, None),
      Err(Error::OutOfRange(_))
    ));
  }
}

#[test]
fn model_from_weights_accepts_valid_config() {
  // The gate does not reject a valid config: a well-formed tiny config with a
  // matching weight map still builds the head-less decoder.
  let w = tiny_weights(true);
  let mut map = tiny_weight_map(&w);
  assert!(Qwen3Model::from_weights(&tiny_config(true), &mut map, None).is_ok());
}

#[test]
fn from_json_rejects_malformed_json() {
  assert!(matches!(
    Qwen3Config::from_json("{not json"),
    Err(Error::Parse(_))
  ));
}

// ════════════════════════════ quantized load ════════════════════════════
//
// A small synthetic 8-bit affine-quantized Qwen3 checkpoint, built by running
// the dense weights through the real `ops::quantized::quantize` op (exactly how
// an mlx-community quantized Qwen3 bundle stores a quantized `nn.Linear` /
// `nn.Embedding`), then asserting it loads through `Qwen3::from_weights` and
// forwards to the right shape. Mirrors the Whisper / EmbeddingGemma
// quantized-load tests.
//
// The dims are chosen so every quantized weight's last axis (the `in`
// dimension) is a whole number of affine groups (`group_size = 32`), which
// mlx's affine `quantize` requires: every projection / the embedding / the
// untied head contracts over 32 or 64.

/// Affine group size for the synthetic quantized checkpoint (divides every
/// quantized weight's last axis: 32, 64).
const QGROUP: i32 = 32;
/// Bit depth for the synthetic quantized checkpoint.
const QBITS: i32 = 8;

const Q_HIDDEN: i32 = 32;
const Q_HEADS: i32 = 2;
const Q_KV_HEADS: i32 = 1;
const Q_HEAD_DIM: i32 = 16; // Q_HEADS * Q_HEAD_DIM = 32 = Q_HIDDEN
const Q_INTER: i32 = 64; // multiple of QGROUP
const Q_VOCAB: i32 = 64;

/// A tiny `Qwen3Config` (with a `quantization` block) whose every quantized
/// weight's last axis is a multiple of `QGROUP`. `tie` selects the tied embedding
/// head vs a dedicated `lm_head`.
fn quant_config(tie: bool) -> Qwen3Config {
  let json = format!(
    r#"{{"hidden_size": {Q_HIDDEN}, "head_dim": {Q_HEAD_DIM},
    "num_attention_heads": {Q_HEADS}, "num_key_value_heads": {Q_KV_HEADS},
    "num_hidden_layers": 1, "intermediate_size": {Q_INTER}, "vocab_size": {Q_VOCAB},
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
    "tie_word_embeddings": {tie},
    "quantization": {{ "group_size": {QGROUP}, "bits": {QBITS} }}}}"#
  );
  let cfg = Qwen3Config::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// A deterministic `(rows, cols)` f32 weight with small distinct values.
fn qmat(rows: i32, cols: i32, off: f32) -> Array {
  let (r, c) = (rows as usize, cols as usize);
  let data: Vec<f32> = (0..r * c)
    .map(|n| ((n % 13) as f32) * 0.01 - 0.05 + off)
    .collect();
  Array::from_slice::<f32>(&data, &(r, c)).unwrap()
}

/// A deterministic `(n,)` f32 vector (RMSNorm weights stay dense).
fn qvec(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize)
    .map(|i| 1.0 + ((i % 5) as f32) * 0.01)
    .collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

/// Build the dense `quant_config`-sized weight map (sanitized layout) of the f32
/// originals — the DENSE checkpoint, before any quantization. `tie` adds a
/// dense `lm_head.weight` when untied.
fn dense_quant_sized_map(tie: bool) -> HashMap<String, Array> {
  let mut w = HashMap::new();
  w.insert(
    "model.embed_tokens.weight".to_string(),
    qmat(Q_VOCAB, Q_HIDDEN, 0.0),
  );
  w.insert("model.norm.weight".to_string(), qvec(Q_HIDDEN));
  let p = "model.layers.0";
  let q_out = Q_HEADS * Q_HEAD_DIM; // 32
  let kv_out = Q_KV_HEADS * Q_HEAD_DIM; // 16
  w.insert(
    format!("{p}.self_attn.q_proj.weight"),
    qmat(q_out, Q_HIDDEN, 0.01),
  );
  w.insert(
    format!("{p}.self_attn.k_proj.weight"),
    qmat(kv_out, Q_HIDDEN, 0.02),
  );
  w.insert(
    format!("{p}.self_attn.v_proj.weight"),
    qmat(kv_out, Q_HIDDEN, 0.03),
  );
  w.insert(
    format!("{p}.self_attn.o_proj.weight"),
    qmat(Q_HIDDEN, q_out, 0.04),
  );
  w.insert(format!("{p}.self_attn.q_norm.weight"), qvec(Q_HEAD_DIM));
  w.insert(format!("{p}.self_attn.k_norm.weight"), qvec(Q_HEAD_DIM));
  w.insert(
    format!("{p}.mlp.gate_proj.weight"),
    qmat(Q_INTER, Q_HIDDEN, 0.05),
  );
  w.insert(
    format!("{p}.mlp.up_proj.weight"),
    qmat(Q_INTER, Q_HIDDEN, 0.06),
  );
  w.insert(
    format!("{p}.mlp.down_proj.weight"),
    qmat(Q_HIDDEN, Q_INTER, 0.07),
  );
  w.insert(format!("{p}.input_layernorm.weight"), qvec(Q_HIDDEN));
  w.insert(
    format!("{p}.post_attention_layernorm.weight"),
    qvec(Q_HIDDEN),
  );
  if !tie {
    w.insert("lm_head.weight".to_string(), qmat(Q_VOCAB, Q_HIDDEN, 0.08));
  }
  w
}

/// Replace the dense `<prefix>.weight` in `w` with the real
/// `ops::quantized::quantize` affine triple (`<prefix>.weight` packed +
/// `<prefix>.scales` + `<prefix>.biases`), mirroring how an mlx-community
/// quantized checkpoint stores a quantized `nn.Linear` / `nn.Embedding`.
fn quantize_weight_in_place(w: &mut HashMap<String, Array>, prefix: &str) {
  let dense = w
    .remove(&format!("{prefix}.weight"))
    .expect("dense weight present");
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, QGROUP, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

/// The list of quantizable layer prefixes (every `nn.Linear` + the embedding;
/// the untied `lm_head` when present). RMSNorm weights stay dense.
fn quant_prefixes(tie: bool) -> Vec<String> {
  let p = "model.layers.0";
  let mut prefixes = vec![
    "model.embed_tokens".to_string(),
    format!("{p}.self_attn.q_proj"),
    format!("{p}.self_attn.k_proj"),
    format!("{p}.self_attn.v_proj"),
    format!("{p}.self_attn.o_proj"),
    format!("{p}.mlp.gate_proj"),
    format!("{p}.mlp.up_proj"),
    format!("{p}.mlp.down_proj"),
  ];
  if !tie {
    prefixes.push("lm_head".to_string());
  }
  prefixes
}

/// The dense map with every `nn.Linear` projection, the token embedding, and
/// (untied) the `lm_head` quantized to the 8-bit affine triple — mirroring an
/// mlx-community quantized Qwen3 checkpoint.
fn quant_weights(tie: bool) -> HashMap<String, Array> {
  let mut w = dense_quant_sized_map(tie);
  for prefix in quant_prefixes(tie) {
    quantize_weight_in_place(&mut w, &prefix);
  }
  w
}

#[test]
fn quantized_checkpoint_loads_and_builds_quantized_layers() {
  // With a quantization config and a checkpoint whose Linear/Embedding weights
  // carry `.scales`/`.biases`, the model builds quantized layers throughout. A
  // packed `uint32` weight of a DIFFERENT shape than the dense `(out, in)` would
  // otherwise be rejected by the dense shape gate, so a successful load is itself
  // proof the quantized path ran — but assert the introspection too.
  let model = Qwen3::from_weights(quant_config(true), quant_weights(true)).expect("load");
  assert!(
    model.model().embedding_is_quantized(),
    "the token embedding must load quantized"
  );
  assert!(
    model.model().all_projections_quantized(),
    "every attention + MLP projection must load quantized"
  );
}

#[test]
fn quantized_untied_lm_head_loads_quantized() {
  // An untied quantized checkpoint loads the dedicated `lm_head` through the
  // quantized path too (its `lm_head.scales` sibling is the signal).
  let model = Qwen3::from_weights(quant_config(false), quant_weights(false)).expect("load");
  assert!(
    model.model().embedding_is_quantized(),
    "the token embedding must load quantized"
  );
  assert!(
    model.model().all_projections_quantized(),
    "every attention + MLP projection must load quantized"
  );
  assert!(
    model.untied_lm_head_is_quantized(),
    "the untied lm_head must load quantized"
  );
}

#[test]
fn quantized_checkpoint_forwards_to_expected_logits_shape() {
  // The full quantized forward executes (the quantized attention/MLP
  // `quantized_matmul`, the quantized token-embedding gather + dequantize, and
  // the tied quantized `as_linear` head) and returns finite `[B, L, vocab]`
  // logits.
  let model = Qwen3::from_weights(quant_config(true), quant_weights(true)).expect("load");
  let tokens = Array::from_slice::<i32>(&[1, 7, 30], &(1usize, 3usize)).unwrap();
  let mut cache = model.make_cache();
  let mut logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 3, Q_VOCAB as usize]);
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite()),
    "quantized forward produced non-finite logits"
  );
}

#[test]
fn quantized_forward_matches_dequantized_dense_oracle() {
  // Independent oracle: dequantize every triple back to a dense weight via the
  // real `ops::quantized::dequantize` op, build a DENSE Qwen3 from those weights,
  // and run it. The quantized model's logits must match the dense-from-
  // dequantized-weights logits — the only difference is the round-trip quant
  // error, which `dequantize(quantize(w))` reproduces exactly on both sides, so
  // the two are bit-for-bit equal here (both consume the same dequantized
  // weight). This confirms the quantized `quantized_matmul` / gather path is the
  // faithful counterpart of the dense matmul, not an unrelated graph.
  let qmap = quant_weights(true);

  // Build the dense oracle map: dequantize each quantized triple, keep the dense
  // (RMSNorm) weights verbatim.
  let prefixes = quant_prefixes(true);
  let mut dense: HashMap<String, Array> = HashMap::new();
  for (key, arr) in &qmap {
    // Skip the `.scales` / `.biases` siblings; reconstruct from the `.weight`.
    if key.ends_with(".scales") || key.ends_with(".biases") {
      continue;
    }
    let prefix = key.strip_suffix(".weight");
    let is_quant = prefix.is_some_and(|p| prefixes.iter().any(|qp| qp == p));
    if is_quant {
      let p = prefix.unwrap();
      let scales = qmap.get(&format!("{p}.scales")).unwrap();
      let biases = qmap.get(&format!("{p}.biases")).unwrap();
      let deq = crate::ops::quantized::dequantize(
        arr,
        scales,
        Some(biases),
        QGROUP,
        QBITS,
        "affine",
        None,
        None,
      )
      .unwrap();
      dense.insert(key.clone(), deq);
    } else {
      dense.insert(key.clone(), arr.try_clone().unwrap());
    }
  }

  // Dense model: a config WITHOUT the quantization block, dense weights.
  let dense_cfg = Qwen3Config::from_json(&format!(
    r#"{{"hidden_size": {Q_HIDDEN}, "head_dim": {Q_HEAD_DIM},
    "num_attention_heads": {Q_HEADS}, "num_key_value_heads": {Q_KV_HEADS},
    "num_hidden_layers": 1, "intermediate_size": {Q_INTER}, "vocab_size": {Q_VOCAB},
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true}}"#
  ))
  .unwrap();
  let dense_model = Qwen3::from_weights(dense_cfg, dense).expect("dense load");
  let quant_model = Qwen3::from_weights(quant_config(true), qmap).expect("quant load");

  let tokens = Array::from_slice::<i32>(&[2, 11, 40], &(1usize, 3usize)).unwrap();
  let mut dc = dense_model.make_cache();
  let mut qc = quant_model.make_cache();
  let mut dense_logits = dense_model.forward(&tokens, &mut dc).unwrap();
  let mut quant_logits = quant_model.forward(&tokens, &mut qc).unwrap();
  assert_eq!(quant_logits.shape(), dense_logits.shape());

  let want = dense_logits.to_vec::<f32>().unwrap();
  let got = quant_logits.to_vec::<f32>().unwrap();
  // Tolerance for the dequantize/quantized_matmul accumulation-order difference
  // (the quantized kernel and the dense matmul accumulate in a different order on
  // identical dequantized values), scaled to the logit magnitude.
  let scale = want.iter().fold(0.0_f32, |m, v| m.max(v.abs())).max(1.0);
  for (i, (g, w)) in got.iter().zip(&want).enumerate() {
    assert!(
      (g - w).abs() <= 2e-3 * scale,
      "index {i}: quant {g} vs dense-from-dequant {w} (|Δ|={})",
      (g - w).abs()
    );
  }
}

/// The quantized Qwen3 forward must preserve the activation dtype: a `bf16`/`f16`
/// activation must yield `bf16`/`f16` logits, never a silent promotion to `f32`.
/// The quantized `quantized_matmul` / dequantize gather follow the activation
/// dtype, so this is the quantized-path counterpart of the dense
/// `forward_preserves_*` guards. The scales/biases are kept f32 (mlx quantizes
/// from an f32 master here); the activation enters bf16/f16 via the embedding
/// dequantize → the `scales` dtype drives it, so quantize the dense weights from
/// a `dtype`-cast master so the dequantize yields `dtype`.
fn assert_quantized_forward_preserves_dtype(dtype: crate::Dtype) {
  // Build the dense master in `dtype`, then quantize: mlx's `quantize` writes
  // `scales` in the input dtype, so the embedding dequantize + every
  // `quantized_matmul` produce `dtype` activations.
  let mut w: HashMap<String, Array> = dense_quant_sized_map(true)
    .into_iter()
    .map(|(k, v)| (k, v.astype(dtype).expect("cast master")))
    .collect();
  for prefix in quant_prefixes(true) {
    quantize_weight_in_place(&mut w, &prefix);
  }
  let model = Qwen3::from_weights(quant_config(true), w).expect("load");
  let tokens = Array::from_slice::<i32>(&[1, 7], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(
    logits.dtype().unwrap(),
    dtype,
    "quantized Qwen3 forward upcast {dtype:?} → {:?}",
    logits.dtype().unwrap()
  );
  let vals = logits
    .astype(crate::Dtype::F32)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
}

#[test]
fn quantized_forward_preserves_bf16() {
  assert_quantized_forward_preserves_dtype(crate::Dtype::BF16);
}

#[test]
fn quantized_forward_preserves_f16() {
  assert_quantized_forward_preserves_dtype(crate::Dtype::F16);
}

#[test]
fn dense_checkpoint_loads_dense_even_with_quant_config() {
  // A dense checkpoint (no `.scales` siblings) loads DENSE even when the config
  // carries a `quantization` block — the `.scales` sibling, not the config, is
  // the load-bearing per-layer signal (the `class_predicate` gate).
  let model = Qwen3::from_weights(quant_config(true), dense_quant_sized_map(true)).expect("load");
  assert!(
    !model.model().embedding_is_quantized(),
    "a dense embedding must NOT load quantized just because the config has a quant block"
  );
  assert!(
    !model.model().all_projections_quantized(),
    "dense projections must NOT load quantized without `.scales` siblings"
  );
  // And it still forwards.
  let tokens = Array::from_slice::<i32>(&[3, 9], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, Q_VOCAB as usize]);
}

#[test]
fn quantized_scales_without_config_is_rejected() {
  // A checkpoint that carries `.scales` siblings but a config with NO
  // quantization block does NOT silently mis-load: the `.scales` sibling ALONE
  // is the load-bearing "this layer is quantized" signal (the shared
  // `.scales`-presence discriminator), so the layer takes the quantized branch
  // and — finding no scheme to interpret the packed weight — fails fast with a
  // typed `InvariantViolation` (a mixed / malformed checkpoint), NEVER a silent
  // dense reinterpret of the packed `uint32` weight. The first consumed
  // quantized weight (the token embedding) fires it.
  let model = Qwen3::from_weights(tiny_quant_sized_dense_config(), quant_weights(true));
  assert!(matches!(model, Err(Error::InvariantViolation(_))));
}

/// Insert a stale `<prefix>.scales` next to an otherwise-dense, correctly-shaped
/// `<prefix>.weight` in `w`. The `.scales` shape is irrelevant: the
/// `.scales`-presence gate enters the quantized branch and (with no resolvable
/// scheme) errors BEFORE any `.scales` shape check, so a tiny placeholder
/// suffices. Used to prove a DENSE-shaped weight carrying a stale `.scales` is
/// rejected rather than silently loaded dense (the dense shape gate would have
/// admitted the dense-shaped weight, masking the stale quant metadata).
fn add_stale_scales(w: &mut HashMap<String, Array>, prefix: &str) {
  assert!(
    w.contains_key(&format!("{prefix}.weight")),
    "{prefix}.weight must be a dense, correctly-shaped weight"
  );
  w.insert(format!("{prefix}.scales"), qvec(QGROUP));
}

#[test]
fn dense_shaped_weight_with_stale_scales_projection_is_invariant_violation() {
  // A DENSE-shaped `q_proj.weight` (the dense shape gate would accept it) that
  // ALSO carries a stale `q_proj.scales`, under a config with NO quantization
  // block, must NOT silently load dense (ignoring the quant metadata): the
  // `.scales` presence alone routes it to the quantized branch, which — finding
  // no scheme — fails with a typed `InvariantViolation`. The embedding is left
  // clean dense so the projection (not the embedding) is the layer that fires.
  let mut w = dense_quant_sized_map(true);
  add_stale_scales(&mut w, "model.layers.0.self_attn.q_proj");
  let model = Qwen3::from_weights(tiny_quant_sized_dense_config(), w);
  assert!(matches!(model, Err(Error::InvariantViolation(_))));
}

#[test]
fn dense_shaped_weight_with_stale_scales_embedding_is_invariant_violation() {
  // The token embedding: a dense-shaped `model.embed_tokens.weight` carrying a
  // stale `model.embed_tokens.scales`, no quant config — the same
  // `.scales`-presence gate rejects it with `InvariantViolation` rather than
  // silently gathering against the dense-reinterpreted table.
  let mut w = dense_quant_sized_map(true);
  add_stale_scales(&mut w, "model.embed_tokens");
  let model = Qwen3::from_weights(tiny_quant_sized_dense_config(), w);
  assert!(matches!(model, Err(Error::InvariantViolation(_))));
}

#[test]
fn dense_shaped_weight_with_stale_scales_untied_lm_head_is_invariant_violation() {
  // The untied `lm_head`: a dense-shaped `lm_head.weight` carrying a stale
  // `lm_head.scales`, no quant config. The embedding + every projection are
  // clean dense, so loading reaches the (last-consumed) untied head, whose
  // `.scales` presence routes it to the quantized branch → `InvariantViolation`,
  // not a silent dense head.
  let mut w = dense_quant_sized_map(false);
  add_stale_scales(&mut w, "lm_head");
  let model = Qwen3::from_weights(tiny_quant_sized_dense_config_untied(), w);
  assert!(matches!(model, Err(Error::InvariantViolation(_))));
}

#[test]
fn quantized_scales_with_skip_override_is_invariant_violation() {
  // A config WITH a quantization block but an explicit per-layer `false` (Skip)
  // for a layer that nonetheless carries `.scales` cannot resolve scheme
  // parameters for it — a typed InvariantViolation, never a guessed scheme nor a
  // silent dense reinterpret of the packed weight. Mark the token embedding
  // `Skip`; it is the first consumed quantized layer.
  let json = format!(
    r#"{{"hidden_size": {Q_HIDDEN}, "head_dim": {Q_HEAD_DIM},
    "num_attention_heads": {Q_HEADS}, "num_key_value_heads": {Q_KV_HEADS},
    "num_hidden_layers": 1, "intermediate_size": {Q_INTER}, "vocab_size": {Q_VOCAB},
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true,
    "quantization": {{ "group_size": {QGROUP}, "bits": {QBITS},
      "model.embed_tokens": false }}}}"#
  );
  let cfg = Qwen3Config::from_json(&json).unwrap();
  let model = Qwen3::from_weights(cfg, quant_weights(true));
  assert!(matches!(model, Err(Error::InvariantViolation(_))));
}

/// The `quant_config` dims WITHOUT a quantization block — used to prove a
/// `.scales`-bearing checkpoint with no config quant block is rejected.
fn tiny_quant_sized_dense_config() -> Qwen3Config {
  Qwen3Config::from_json(&format!(
    r#"{{"hidden_size": {Q_HIDDEN}, "head_dim": {Q_HEAD_DIM},
    "num_attention_heads": {Q_HEADS}, "num_key_value_heads": {Q_KV_HEADS},
    "num_hidden_layers": 1, "intermediate_size": {Q_INTER}, "vocab_size": {Q_VOCAB},
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true}}"#
  ))
  .unwrap()
}

/// The `tiny_quant_sized_dense_config` dims but UNTIED (a dedicated `lm_head`),
/// still WITHOUT a quantization block — used to prove a stale-`.scales` untied
/// head is rejected with no config quant block.
fn tiny_quant_sized_dense_config_untied() -> Qwen3Config {
  Qwen3Config::from_json(&format!(
    r#"{{"hidden_size": {Q_HIDDEN}, "head_dim": {Q_HEAD_DIM},
    "num_attention_heads": {Q_HEADS}, "num_key_value_heads": {Q_KV_HEADS},
    "num_hidden_layers": 1, "intermediate_size": {Q_INTER}, "vocab_size": {Q_VOCAB},
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": false}}"#
  ))
  .unwrap()
}

#[test]
fn config_quantization_parses_block_and_defaults_none() {
  // The `quantization` block deserializes to a PerLayerQuantization; absent block
  // is None.
  let plq = quant_config(true).quantization().unwrap().expect("Some");
  let q = plq
    .quantization_for("model.layers.0.self_attn.q_proj")
    .unwrap();
  assert_eq!(q.group_size, QGROUP);
  assert_eq!(q.bits, QBITS);
  // A config with no quantization block → None.
  assert!(
    Qwen3Config::from_json("{}")
      .unwrap()
      .quantization()
      .unwrap()
      .is_none()
  );
  // A null block → None.
  assert!(
    Qwen3Config::from_json(r#"{"quantization": null}"#)
      .unwrap()
      .quantization()
      .unwrap()
      .is_none()
  );
}

/// A `quant_config`-sized config carrying a `quantization` block that is NOT a
/// valid `PerLayerQuantization` — `quant_block_json` is spliced in verbatim.
/// `from_json` parses it (the block is retained as an opaque JSON value and
/// `validate` never inspects it), but [`Qwen3Config::quantization`] would fail
/// to deserialize it: a malformed / foreign / partial block becomes fatal ONLY
/// if the scheme is actually resolved. `tie` selects the tied vs untied head.
fn dense_config_with_quant_block(tie: bool, quant_block_json: &str) -> Qwen3Config {
  Qwen3Config::from_json(&format!(
    r#"{{"hidden_size": {Q_HIDDEN}, "head_dim": {Q_HEAD_DIM},
    "num_attention_heads": {Q_HEADS}, "num_key_value_heads": {Q_KV_HEADS},
    "num_hidden_layers": 1, "intermediate_size": {Q_INTER}, "vocab_size": {Q_VOCAB},
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": {tie},
    "quantization": {quant_block_json}}}"#
  ))
  .unwrap()
}

#[test]
fn dense_checkpoint_with_malformed_quant_block_loads_dense() {
  // The completing-finding case: a DENSE checkpoint (no `.scales` on any layer)
  // whose config carries a stale / foreign / partial `quantization` block that
  // is NOT a valid mlx `PerLayerQuantization`. The block becoming fatal would
  // contradict the `.scales`-presence discriminator (unused quant metadata must
  // not break a dense load), so `from_weights` must load DENSE and forward —
  // the scheme is never resolved because no layer is quantized.
  //
  // A `{ "quant_method": "gptq", "version": 2 }`-style block has no
  // `group_size`, so it is the kind that WOULD be fatal if parsed — assert that
  // directly first, so this test stays honest if the parser ever loosens.
  let malformed = r#"{ "quant_method": "gptq", "version": 2, "bits": 4 }"#;
  let probe = dense_config_with_quant_block(true, malformed);
  assert!(
    matches!(probe.quantization(), Err(Error::Parse(_))),
    "the malformed block must be fatal WHEN the scheme is resolved (control)"
  );

  // Dense weights (no `.scales`) + this malformed block → loads DENSE, no parse.
  let model = Qwen3::from_weights(
    dense_config_with_quant_block(true, malformed),
    dense_quant_sized_map(true),
  )
  .expect("dense checkpoint must load despite the unused malformed quant block");
  assert!(
    !model.model().embedding_is_quantized(),
    "no `.scales` ⇒ dense embedding"
  );
  assert!(
    !model.model().all_projections_quantized(),
    "no `.scales` ⇒ dense projections"
  );
  // And the dense forward runs to the expected logits shape.
  let tokens = Array::from_slice::<i32>(&[4, 12], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, Q_VOCAB as usize]);
}

#[test]
fn dense_untied_checkpoint_with_partial_quant_block_loads_dense() {
  // The untied-head counterpart: an incomplete block (only `bits`, no required
  // `group_size`) on an UNTIED dense checkpoint (dedicated `lm_head.weight`, no
  // `.scales` anywhere). The dedicated head + every projection load dense; the
  // partial block is never parsed.
  let partial = r#"{ "bits": 8 }"#;
  let probe = dense_config_with_quant_block(false, partial);
  assert!(
    matches!(probe.quantization(), Err(Error::Parse(_))),
    "a block missing `group_size` must be fatal WHEN resolved (control)"
  );
  let model = Qwen3::from_weights(
    dense_config_with_quant_block(false, partial),
    dense_quant_sized_map(false),
  )
  .expect("dense untied checkpoint must load despite the unused partial quant block");
  assert!(!model.model().all_projections_quantized());
  assert!(
    !model.untied_lm_head_is_quantized(),
    "the dedicated dense `lm_head` must NOT load quantized"
  );
  let tokens = Array::from_slice::<i32>(&[5, 20], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  let logits = model.forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, Q_VOCAB as usize]);
}

#[test]
fn malformed_quant_block_with_present_scales_still_errors() {
  // The symmetry guard: when a relevant `.scales` IS present, the scheme is
  // still required, so a malformed block remains fatal — the pre-scan gates the
  // parse, it never SUPPRESSES it. A dense-shaped `q_proj.weight` carrying a
  // stale `q_proj.scales`, under a config with a malformed quant block, surfaces
  // the block's parse failure (the `.scales` pre-scan finds the sibling and
  // resolves the scheme, which fails) rather than silently loading dense.
  let mut w = dense_quant_sized_map(true);
  add_stale_scales(&mut w, "model.layers.0.self_attn.q_proj");
  let cfg = dense_config_with_quant_block(true, r#"{ "quant_method": "gptq" }"#);
  assert!(matches!(Qwen3::from_weights(cfg, w), Err(Error::Parse(_))));
}

#[test]
fn stale_scales_on_never_quantized_layer_does_not_resolve_scheme() {
  // A `.scales` sibling on a layer the model NEVER loads quantized (the final
  // `model.norm`, which loads dense via `take_shaped`) is NOT a relevant scale:
  // the pre-scan must ignore it, so a dense checkpoint with a malformed quant
  // block still loads DENSE. This pins the leaf-name boundary of the pre-scan —
  // only the projections / embedding / `lm_head` carry a load-bearing `.scales`.
  let mut w = dense_quant_sized_map(true);
  w.insert("model.norm.scales".to_string(), qvec(QGROUP));
  let model = Qwen3::from_weights(
    dense_config_with_quant_block(true, r#"{ "quant_method": "awq" }"#),
    w,
  )
  .expect("an irrelevant `.scales` on `model.norm` must not resolve the scheme");
  assert!(!model.model().all_projections_quantized());
}

// ════════════ exact, config-aware `.scales` pre-scan ════════════
//
// The pre-scan probes the EXACT `<prefix>.scales` keys the `Linear` / `Embedding`
// loaders consume under the validated config — not a suffix / `ends_with` match.
// A foreign key, an out-of-range layer index, or a tied `lm_head.scales` is then
// IGNORED, so a dense checkpoint carrying one (plus a malformed/unused quant
// block) still loads DENSE; only a scale on a layer the loaders actually consume
// resolves the scheme.

#[test]
fn foreign_scales_key_is_ignored_dense_load() {
  // A `.scales` whose prefix merely ENDS WITH a quantizable leaf (`foreign.q_proj`)
  // is NOT a key any loader consults: the exact-key pre-scan ignores it, so a
  // dense checkpoint with an unused malformed quant block still loads DENSE.
  let mut w = dense_quant_sized_map(true);
  w.insert("foreign.q_proj.scales".to_string(), qvec(QGROUP));
  let model = Qwen3::from_weights(
    dense_config_with_quant_block(true, r#"{ "quant_method": "gptq" }"#),
    w,
  )
  .expect("a foreign `.scales` key must not resolve the scheme — dense load");
  assert!(
    !model.model().embedding_is_quantized() && !model.model().all_projections_quantized(),
    "no consumed `.scales` ⇒ fully dense"
  );
  let tokens = Array::from_slice::<i32>(&[2, 8], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  assert_eq!(
    model.forward(&tokens, &mut cache).unwrap().shape(),
    vec![1, 2, Q_VOCAB as usize]
  );
}

#[test]
fn out_of_range_layer_scales_key_is_ignored_dense_load() {
  // An in-shape `.scales` on a layer index BEYOND `num_hidden_layers` (which is 1
  // here, so layer 99 is never built) is not consumed by any loader: the
  // config-aware pre-scan iterates only `0..num_hidden_layers`, so a dense
  // checkpoint with an unused malformed quant block still loads DENSE.
  let mut w = dense_quant_sized_map(true);
  w.insert(
    "model.layers.99.self_attn.q_proj.scales".to_string(),
    qvec(QGROUP),
  );
  let model = Qwen3::from_weights(
    dense_config_with_quant_block(true, r#"{ "quant_method": "gptq" }"#),
    w,
  )
  .expect("an out-of-range layer `.scales` must not resolve the scheme — dense load");
  assert!(!model.model().all_projections_quantized());
  let tokens = Array::from_slice::<i32>(&[3, 11], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  assert_eq!(
    model.forward(&tokens, &mut cache).unwrap().shape(),
    vec![1, 2, Q_VOCAB as usize]
  );
}

#[test]
fn tied_lm_head_scales_is_ignored_dense_load() {
  // A TIED model reuses the embedding table as the head and never consumes
  // `lm_head.scales`; the pre-scan gates `lm_head` on `!tie_word_embeddings`, so
  // a stale `lm_head.scales` on a tied dense checkpoint (plus a malformed unused
  // block) is IGNORED and the checkpoint loads DENSE.
  let mut w = dense_quant_sized_map(true);
  w.insert("lm_head.scales".to_string(), qvec(QGROUP));
  let model = Qwen3::from_weights(
    dense_config_with_quant_block(true, r#"{ "quant_method": "gptq" }"#),
    w,
  )
  .expect("a tied `lm_head.scales` must not resolve the scheme — dense load");
  assert!(
    !model.model().embedding_is_quantized() && !model.model().all_projections_quantized(),
    "tied dense head ⇒ fully dense"
  );
  let tokens = Array::from_slice::<i32>(&[6, 21], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  assert_eq!(
    model.forward(&tokens, &mut cache).unwrap().shape(),
    vec![1, 2, Q_VOCAB as usize]
  );
}

#[test]
fn real_consumed_scales_still_resolves_scheme_and_errors() {
  // The symmetry guard for the exact pre-scan: a `.scales` on a layer the
  // loaders DO consume (`model.layers.0.self_attn.q_proj`, layer 0 is in range)
  // is relevant, so the scheme IS resolved — a malformed quant block then remains
  // fatal (the R4 InvariantViolation/Parse contract stays green), proving the
  // exact match did not regress the real-scale detection.
  let mut w = dense_quant_sized_map(true);
  add_stale_scales(&mut w, "model.layers.0.self_attn.q_proj");
  // No config quant block ⇒ no scheme resolves ⇒ typed InvariantViolation.
  assert!(matches!(
    Qwen3::from_weights(tiny_quant_sized_dense_config(), w),
    Err(Error::InvariantViolation(_))
  ));
}

#[test]
fn untied_lm_head_scales_is_relevant_and_resolves_scheme() {
  // The untied counterpart to the tied-ignore test: an UNTIED model DOES consume
  // `lm_head.scales`, so the pre-scan must treat it as relevant. A stale
  // `lm_head.scales` on an otherwise-dense untied checkpoint with no resolvable
  // scheme is therefore a typed InvariantViolation, not a silent dense head.
  let mut w = dense_quant_sized_map(false);
  add_stale_scales(&mut w, "lm_head");
  assert!(matches!(
    Qwen3::from_weights(tiny_quant_sized_dense_config_untied(), w),
    Err(Error::InvariantViolation(_))
  ));
}

// ════════════ `quantization` / `quantization_config` two-field precedence ════════════
//
// A mlx-saved quantized config mirrors `quantization` into `quantization_config`
// (`utils.py:914-915`, reproduced by mlxrs `save_config`), so a round-tripped
// config carries BOTH keys. The two separate `#[serde(default)]` fields parse
// either or both without a serde duplicate-field error, and `quantization()`
// resolves precedence (canonical `quantization` first, else the mirror).

/// A `quant_config`-sized config carrying BOTH a `quantization` and a
/// `quantization_config` block — each spliced verbatim — mirroring a
/// mlxrs-saved (round-tripped) quantized Qwen3 `config.json`.
fn config_with_both_quant_keys(
  tie: bool,
  quantization_json: &str,
  quantization_config_json: &str,
) -> Qwen3Config {
  Qwen3Config::from_json(&format!(
    r#"{{"hidden_size": {Q_HIDDEN}, "head_dim": {Q_HEAD_DIM},
    "num_attention_heads": {Q_HEADS}, "num_key_value_heads": {Q_KV_HEADS},
    "num_hidden_layers": 1, "intermediate_size": {Q_INTER}, "vocab_size": {Q_VOCAB},
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": {tie},
    "quantization": {quantization_json},
    "quantization_config": {quantization_config_json}}}"#
  ))
  .unwrap()
}

#[test]
fn config_with_both_quant_keys_parses_without_duplicate_field_error() {
  // The core round-trip fix: a config carrying BOTH keys must deserialize (the
  // single aliased field would reject it as a serde duplicate-field error). Both
  // fields are populated and the chosen block resolves.
  let cfg = config_with_both_quant_keys(
    true,
    &format!(r#"{{ "group_size": {QGROUP}, "bits": {QBITS} }}"#),
    &format!(r#"{{ "group_size": {QGROUP}, "bits": {QBITS} }}"#),
  );
  assert!(cfg.quantization.is_some(), "`quantization` field populated");
  assert!(
    cfg.quantization_config.is_some(),
    "`quantization_config` field populated"
  );
  let plq = cfg.quantization().unwrap().expect("a block resolves");
  let q = plq
    .quantization_for("model.layers.0.self_attn.q_proj")
    .unwrap();
  assert_eq!((q.group_size, q.bits), (QGROUP, QBITS));
}

#[test]
fn both_keys_precedence_prefers_quantization_block() {
  // Precedence: when both keys are present, the canonical `quantization` block
  // wins (mlx-lm mirrors `quantization_config` INTO `quantization`). Give the two
  // blocks DISTINCT bit depths and confirm the resolved scheme is `quantization`'s.
  let cfg = config_with_both_quant_keys(
    true,
    r#"{ "group_size": 32, "bits": 8 }"#,
    r#"{ "group_size": 64, "bits": 4 }"#,
  );
  let q = cfg
    .quantization()
    .unwrap()
    .expect("resolves")
    .quantization_for("model.embed_tokens")
    .unwrap();
  assert_eq!(
    (q.group_size, q.bits),
    (32, 8),
    "the canonical `quantization` block must take precedence over `quantization_config`"
  );
}

#[test]
fn quantization_config_mirror_is_fallback_when_quantization_absent() {
  // Fallback: a config with `quantization` absent + a `quantization_config`
  // mirror present resolves the mirror (HF model-tree interop), so a
  // quantization_config-only checkpoint config is not silently treated as dense.
  let json = format!(
    r#"{{"hidden_size": {Q_HIDDEN}, "head_dim": {Q_HEAD_DIM},
    "num_attention_heads": {Q_HEADS}, "num_key_value_heads": {Q_KV_HEADS},
    "num_hidden_layers": 1, "intermediate_size": {Q_INTER}, "vocab_size": {Q_VOCAB},
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true,
    "quantization_config": {{ "group_size": 64, "bits": 4 }}}}"#
  );
  let cfg = Qwen3Config::from_json(&json).unwrap();
  let q = cfg
    .quantization()
    .unwrap()
    .expect("the mirror resolves")
    .quantization_for("model.embed_tokens")
    .unwrap();
  assert_eq!((q.group_size, q.bits), (64, 4));
}

#[test]
fn quantization_null_falls_back_to_quantization_config_mirror() {
  // A config carrying `quantization: null` (present but null) alongside a real
  // `quantization_config` mirror still resolves the mirror — the null canonical
  // block is treated as absent for precedence.
  let cfg = config_with_both_quant_keys(
    true,
    "null",
    &format!(r#"{{ "group_size": {QGROUP}, "bits": {QBITS} }}"#),
  );
  let q = cfg
    .quantization()
    .unwrap()
    .expect("the mirror resolves when `quantization` is null")
    .quantization_for("model.embed_tokens")
    .unwrap();
  assert_eq!((q.group_size, q.bits), (QGROUP, QBITS));
}

#[test]
fn both_keys_quantized_round_trip_loads_quantized() {
  // A genuinely quantized round-trip: a both-keys config (the mlxrs-saved layout)
  // + a `.scales`-bearing checkpoint loads QUANTIZED throughout — the two-field
  // parse does not break the real quantized path, and the resolved scheme drives
  // every layer.
  let cfg = config_with_both_quant_keys(
    true,
    &format!(r#"{{ "group_size": {QGROUP}, "bits": {QBITS} }}"#),
    &format!(r#"{{ "group_size": {QGROUP}, "bits": {QBITS} }}"#),
  );
  let model = Qwen3::from_weights(cfg, quant_weights(true)).expect("quantized load");
  assert!(
    model.model().embedding_is_quantized() && model.model().all_projections_quantized(),
    "a both-keys quantized checkpoint must load quantized throughout"
  );
}

#[test]
fn both_keys_malformed_dense_checkpoint_loads_dense() {
  // The gated-pre-scan × two-field combination: a DENSE checkpoint (no consumed
  // `.scales`) whose both-keys config carries a malformed block under BOTH keys
  // still loads DENSE — the config parses (no duplicate-field error) and the
  // unused malformed scheme is never resolved.
  let model = Qwen3::from_weights(
    config_with_both_quant_keys(
      true,
      r#"{ "quant_method": "gptq", "version": 2 }"#,
      r#"{ "quant_method": "gptq", "version": 2 }"#,
    ),
    dense_quant_sized_map(true),
  )
  .expect("a dense both-keys-malformed checkpoint must load dense");
  assert!(
    !model.model().embedding_is_quantized() && !model.model().all_projections_quantized(),
    "no consumed `.scales` ⇒ fully dense despite the both-keys malformed block"
  );
  let tokens = Array::from_slice::<i32>(&[7, 19], &(1usize, 2usize)).unwrap();
  let mut cache = model.make_cache();
  assert_eq!(
    model.forward(&tokens, &mut cache).unwrap().shape(),
    vec![1, 2, Q_VOCAB as usize]
  );
}
