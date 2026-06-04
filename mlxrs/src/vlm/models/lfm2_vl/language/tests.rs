//! Tests for the LFM2.5-VL language adapter — the embed/forward-from-embeddings
//! entry points + their boundary guards.
//!
//! Builds a tiny all-conv LFM2 LM (hidden = 4, vocab = 8, 2 conv layers) — the
//! same minimal arch the LFM2 LM's own tests use — wraps it in the adapter, and
//! pins: the rank/width preflight on `forward_embeddings` (rank-0/1/2 + wrong
//! hidden width rejected, the rank-3 correct-width path runs), the token-id
//! range guard on `embed_tokens` / `forward` (out-of-vocab + negative ids
//! rejected, valid ids embed), and the embed output shape.

use std::collections::HashMap;

use super::*;
use crate::lm::models::lfm2::TextConfig;

/// (hidden, vocab, ff, conv_kernel) of the tiny all-conv LM.
const DIMS: (usize, usize, usize, usize) = (4, 8, 8, 3);

/// A tiny all-conv 2-layer LFM2 config (hidden = 4, vocab = 8, no biases).
fn tiny_config() -> TextConfig {
  let json = r#"{"hidden_size": 4, "num_attention_heads": 2,
    "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 8,
    "conv_L_cache": 3, "block_auto_adjust_ff_dim": false, "block_ff_dim": 8,
    "conv_bias": false}"#;
  TextConfig::from_json(json).unwrap()
}

/// The tiny all-conv weight map (mirrors the LFM2 LM's own test fixture).
fn tiny_weights() -> HashMap<String, Array> {
  let (hidden, vocab, ff, k) = DIMS;
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
    let conv_flat: Vec<f32> = (0..hidden * k).map(|i| (i as f32) * 0.02).collect();
    w.insert(
      format!("{p}.conv.conv.weight"),
      Array::from_slice::<f32>(&conv_flat, &(hidden, k, 1usize)).unwrap(),
    );
    w.insert(format!("{p}.conv.in_proj.weight"), mat(3 * hidden, hidden));
    w.insert(format!("{p}.conv.out_proj.weight"), mat(hidden, hidden));
  }
  w
}

/// Build the tiny adapter.
fn tiny_adapter() -> LanguageModel {
  let lm = Lfm2::from_weights(tiny_config(), tiny_weights()).unwrap();
  LanguageModel::new(lm)
}

// ───────────────────── forward_embeddings rank/width guard ─────────────────────

#[test]
fn forward_embeddings_rank3_correct_width_runs() {
  let adapter = tiny_adapter();
  let mut cache = adapter.make_cache();
  // (B=1, T=2, hidden=4) merged embeddings -> (1, 2, vocab=8) logits.
  let emb =
    Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &(1usize, 2, 4)).unwrap();
  let logits = adapter.forward_embeddings(&emb, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, 8]);
}

#[test]
fn forward_embeddings_rejects_rank0() {
  let adapter = tiny_adapter();
  let mut cache = adapter.make_cache();
  let scalar = Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap();
  // Reshape to rank-0 via an empty shape.
  let scalar = crate::ops::shape::reshape(&scalar, &([] as [i32; 0])).unwrap();
  let err = adapter.forward_embeddings(&scalar, &mut cache).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn forward_embeddings_rejects_rank1() {
  let adapter = tiny_adapter();
  let mut cache = adapter.make_cache();
  let emb = Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &(4usize,)).unwrap();
  let err = adapter.forward_embeddings(&emb, &mut cache).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn forward_embeddings_rejects_rank2() {
  let adapter = tiny_adapter();
  let mut cache = adapter.make_cache();
  let emb = Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &(1usize, 4)).unwrap();
  let err = adapter.forward_embeddings(&emb, &mut cache).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn forward_embeddings_rejects_wrong_hidden_width() {
  let adapter = tiny_adapter();
  let mut cache = adapter.make_cache();
  // (1, 2, 5) — width 5 != hidden_size 4.
  let emb = Array::zeros::<f32>(&(1usize, 2, 5)).unwrap();
  let err = adapter.forward_embeddings(&emb, &mut cache).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

// ───────────────────── embed_tokens token-id range guard ─────────────────────

#[test]
fn embed_tokens_valid_ids_shape() {
  let adapter = tiny_adapter();
  // (1, 3) ids all in [0, vocab=8) -> (1, 3, hidden=4).
  let ids = Array::from_slice::<i32>(&[0, 3, 7], &(1usize, 3)).unwrap();
  let mut out = adapter.embed_tokens(&ids).unwrap();
  assert_eq!(out.shape(), vec![1, 3, 4]);
  out.eval().unwrap();
}

#[test]
fn embed_tokens_rejects_out_of_vocab_id() {
  let adapter = tiny_adapter();
  // id 8 == vocab_size -> out of [0, 8).
  let ids = Array::from_slice::<i32>(&[0, 8], &(1usize, 2)).unwrap();
  let err = adapter.embed_tokens(&ids).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn embed_tokens_rejects_negative_id() {
  let adapter = tiny_adapter();
  let ids = Array::from_slice::<i32>(&[0, -1], &(1usize, 2)).unwrap();
  let err = adapter.embed_tokens(&ids).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

// ───────────────────── forward (text-only) ─────────────────────

#[test]
fn forward_text_only_valid_ids_runs() {
  let adapter = tiny_adapter();
  let mut cache = adapter.make_cache();
  let ids = Array::from_slice::<i32>(&[1, 3], &(1usize, 2)).unwrap();
  let logits = adapter.forward(&ids, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, 8]);
}

#[test]
fn forward_text_only_rejects_out_of_vocab() {
  let adapter = tiny_adapter();
  let mut cache = adapter.make_cache();
  let ids = Array::from_slice::<i32>(&[1, 99], &(1usize, 2)).unwrap();
  let err = adapter.forward(&ids, &mut cache).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn inner_exposes_wrapped_lm() {
  let adapter = tiny_adapter();
  assert_eq!(adapter.inner().config().hidden_size, 4);
}

// ───────────────────── dtype preservation (f16 / bf16) ─────────────────────

/// The tiny weight map with every tensor cast to `dt` — a synthetic bf16 / f16
/// dense checkpoint for the language path.
fn tiny_weights_in_dtype(dt: Dtype) -> HashMap<String, Array> {
  let mut w = tiny_weights();
  for v in w.values_mut() {
    *v = crate::ops::misc::astype(v, dt).unwrap();
  }
  w
}

/// The LFM2.5-VL language path (the wrapped LFM2 LM) must keep the activation
/// dtype: a `dt`-dtype checkpoint forwards token ids AND merged embeddings to
/// `dt` logits, never silently promoted to f32 (the preserve-activation-dtype
/// contract; the other language tests are f32-only).
fn assert_language_path_preserves_dtype(dt: Dtype) {
  let lm = Lfm2::from_weights(tiny_config(), tiny_weights_in_dtype(dt)).unwrap();
  let adapter = LanguageModel::new(lm);

  // Text-only forward (ids -> logits) stays in-dtype.
  let mut cache = adapter.make_cache();
  let ids = Array::from_slice::<i32>(&[1, 3], &(1usize, 2)).unwrap();
  let logits = adapter.forward(&ids, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 2, 8]);
  assert_eq!(
    logits.dtype().unwrap(),
    dt,
    "language forward must keep {dt:?} (no f32 promotion)"
  );

  // The merged-embeddings forward (the image-splice path) likewise: a `dt`
  // `(B, T, hidden)` embedding -> `dt` logits.
  let mut cache2 = adapter.make_cache();
  let emb_f32 =
    Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &(1usize, 2, 4)).unwrap();
  let emb = crate::ops::misc::astype(&emb_f32, dt).unwrap();
  let logits2 = adapter.forward_embeddings(&emb, &mut cache2).unwrap();
  assert_eq!(logits2.shape(), vec![1, 2, 8]);
  assert_eq!(
    logits2.dtype().unwrap(),
    dt,
    "language forward_embeddings must keep {dt:?}"
  );
}

#[test]
fn language_path_preserves_bf16_dtype() {
  assert_language_path_preserves_dtype(Dtype::BF16);
}

#[test]
fn language_path_preserves_f16_dtype() {
  assert_language_path_preserves_dtype(Dtype::F16);
}
