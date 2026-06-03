//! Tests for the Qwen3-ASR MRoPE text decoder.
//!
//! Coverage:
//!
//! - [`MRope::axis_selector`] — the interleaved and chunked frequency-to-axis
//!   selectors against hand-computed expectations.
//! - [`MRope::cos_sin`] — for genuine 3-D positions, the per-axis frequency
//!   selection against a closed-form oracle; for text-only positions, the
//!   collapse to standard non-traditional RoPE.
//! - [`MRope::apply`] — `x*cos + rotate_half(x)*sin` against an independent RoPE
//!   rotation on text-only positions.
//! - The head-less decoder forward — a tiny decoder produces `(B, L, hidden)`
//!   hidden states; the text-only MRoPE result equals an independent
//!   standard-RoPE attention oracle.
//! - Config — the MRoPE section parsing and validation.

use std::collections::HashMap;

use super::{config::MRopeConfig, *};
use crate::array::Array;

const TOL: f32 = 1e-4;

fn assert_close(got: &[f32], want: &[f32]) {
  assert_eq!(got.len(), want.len(), "len: {got:?} vs {want:?}");
  for (i, (g, w)) in got.iter().zip(want).enumerate() {
    assert!(
      (g - w).abs() <= TOL,
      "index {i}: got {g} want {w} (|Δ|={})",
      (g - w).abs()
    );
  }
}

// ════════════════════════════ axis selector ════════════════════════════

#[test]
fn axis_selector_interleaved_matches_reference_pattern() {
  // half = 9, section = [3, 2, 2] (sums above-half is fine for the selector;
  // the reference's interleaved layout only consults section[1]/section[2]).
  // H at slots 1,4 (offset 1 step 3, < section[1]*3 = 6); W at slots 2,5 (offset
  // 2 step 3, < section[2]*3 = 6); T elsewhere.
  let sel = MRope::axis_selector(
    9,
    MRopeConfig {
      section: [3, 2, 2],
      interleaved: true,
    },
  );
  assert_eq!(sel, vec![0, 1, 2, 0, 1, 2, 0, 0, 0]);
}

#[test]
fn axis_selector_chunked_matches_reference_pattern() {
  // half = 7, section = [3, 2, 2]: first 3 temporal, next 2 height, next 2 width.
  let sel = MRope::axis_selector(
    7,
    MRopeConfig {
      section: [3, 2, 2],
      interleaved: false,
    },
  );
  assert_eq!(sel, vec![0, 0, 0, 1, 1, 2, 2]);
}

#[test]
fn axis_selector_text_only_fallback_is_all_temporal() {
  // The standard-RoPE fallback section [half, 0, 0]: every slot is temporal.
  let sel = MRope::axis_selector(
    4,
    MRopeConfig {
      section: [4, 0, 0],
      interleaved: false,
    },
  );
  assert_eq!(sel, vec![0, 0, 0, 0]);
}

// ════════════════════════════ cos/sin oracle ════════════════════════════

/// Independent inverse frequencies `base^(-(2d)/dim)` for `d in [0, dim/2)`.
fn inv_freq(dim: i32, base: f64) -> Vec<f64> {
  (0..dim / 2)
    .map(|d| base.powf(-(2.0 * d as f64) / dim as f64))
    .collect()
}

#[test]
fn cos_sin_3d_selects_per_axis_positions() {
  // dim = 6 → half = 3. chunked section [1, 1, 1]: slot 0 temporal, slot 1
  // height, slot 2 width. Distinct per-axis positions so the selection is
  // observable: T=[2], H=[5], W=[9] at a single (B=1, L=1) position.
  let dim = 6i32;
  let base = 1_000_000.0f64;
  let mrope = MRopeConfig {
    section: [1, 1, 1],
    interleaved: false,
  };
  let rope = MRope::new(dim, base as f32, mrope).unwrap();

  // position_ids (3, B=1, L=1): axis 0 = 2, axis 1 = 5, axis 2 = 9.
  let pos = Array::from_slice::<i32>(&[2, 5, 9], &[3, 1, 1]).unwrap();
  let (mut cos, mut sin) = rope.cos_sin(&pos).unwrap();
  let cos = cos.to_vec::<f32>().unwrap();
  let sin = sin.to_vec::<f32>().unwrap();

  // freqs_combined[d] = pos[selector[d]] * inv_freq[d]; selector = [0,1,2].
  let inv = inv_freq(dim, base);
  let combined = [2.0 * inv[0], 5.0 * inv[1], 9.0 * inv[2]];
  // emb = concat([combined, combined]) → cos/sin over 6 dims.
  let mut want_cos = Vec::new();
  let mut want_sin = Vec::new();
  for half in [&combined, &combined] {
    for &f in half {
      want_cos.push(f.cos() as f32);
      want_sin.push(f.sin() as f32);
    }
  }
  assert_eq!(cos.len(), 6);
  assert_close(&cos, &want_cos);
  assert_close(&sin, &want_sin);
}

#[test]
fn cos_sin_text_only_equals_standard_rope_angles() {
  // For text-only positions (all 3 axes equal), the interleave/chunk selection
  // is irrelevant — every slot uses the same scalar position. The angles must
  // equal standard RoPE: pos * inv_freq, duplicated.
  let dim = 8i32;
  let base = 1_000_000.0f64;
  let rope = MRope::new(
    dim,
    base as f32,
    MRopeConfig {
      section: [2, 1, 1], // interleaved layout, but text-only positions collapse it
      interleaved: true,
    },
  )
  .unwrap();

  // position_ids (3, 1, 2): both positions p=3 and p=7 on every axis.
  let pos = Array::from_slice::<i32>(&[3, 7, 3, 7, 3, 7], &[3, 1, 2]).unwrap();
  let (mut cos, _) = rope.cos_sin(&pos).unwrap();
  let cos = cos.to_vec::<f32>().unwrap();

  let inv = inv_freq(dim, base);
  let mut want = Vec::new();
  for &p in &[3.0f64, 7.0] {
    let combined: Vec<f64> = inv.iter().map(|&f| p * f).collect();
    for half in [&combined, &combined] {
      for &f in half {
        want.push(f.cos() as f32);
      }
    }
  }
  assert_close(&cos, &want);
}

// ════════════════════════════ apply (rotate_half) ════════════════════════════

#[test]
fn apply_matches_independent_rope_rotation() {
  // A single head, single position: rotate a known vector and compare to a
  // hand-rolled non-traditional RoPE (pairs d with d + dim/2).
  let dim = 4i32;
  let base = 10_000.0f64;
  let rope = MRope::new(
    dim,
    base as f32,
    MRopeConfig {
      section: [2, 0, 0],
      interleaved: false,
    },
  )
  .unwrap();

  // x (B=1, heads=1, L=1, dim=4) = [1, 2, 3, 4].
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
  // position 5 on the (text-only) temporal axis.
  let pos = Array::from_slice::<i32>(&[5, 5, 5], &[3, 1, 1]).unwrap();
  let (cos, sin) = rope.cos_sin(&pos).unwrap();
  let mut rotated = rope.apply(&x, &cos, &sin).unwrap();
  let got = rotated.to_vec::<f32>().unwrap();

  // Hand-rolled: half = 2; inv_freq = [1, base^(-1/2)]. angle_d = 5 * inv[d].
  // rotate_half(x) = [-x2, -x3, x0, x1] = [-3, -4, 1, 2].
  // out[d] = x[d]*cos(angle_{d%2}) + rh[d]*sin(angle_{d%2}).
  let inv = inv_freq(dim, base);
  let ang: Vec<f64> = inv.iter().map(|&f| 5.0 * f).collect();
  let x_v = [1.0f64, 2.0, 3.0, 4.0];
  let rh = [-3.0f64, -4.0, 1.0, 2.0];
  let mut want = Vec::new();
  for d in 0..4usize {
    let a = ang[d % 2];
    want.push((x_v[d] * a.cos() + rh[d] * a.sin()) as f32);
  }
  assert_close(&got, &want);
}

// ════════════════════════════ tiny decoder ════════════════════════════

const HIDDEN: i32 = 4;
const HEAD_DIM: i32 = 2;
const N_HEADS: i32 = 2;
const N_KV: i32 = 1;
const VOCAB: i32 = 12;

/// A tiny ASR text config: hidden=4, head_dim=2 (→ half=1, mrope fallback
/// [1,0,0]), 2 heads, 1 kv head, 1 layer.
fn tiny_config() -> Qwen3AsrTextConfig {
  let json = format!(
    r#"{{
      "hidden_size": {HIDDEN}, "head_dim": {HEAD_DIM}, "num_attention_heads": {N_HEADS},
      "num_key_value_heads": {N_KV}, "num_hidden_layers": 1, "intermediate_size": 6,
      "vocab_size": {VOCAB}, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
      "tie_word_embeddings": true
    }}"#
  );
  Qwen3AsrTextConfig::from_json(&json).expect("tiny text config must validate")
}

fn filled(shape: &[i32], val: f32) -> Array {
  Array::full::<f32>(&shape.to_vec(), val).unwrap()
}

fn tiny_weights(cfg: &Qwen3AsrTextConfig) -> HashMap<String, Array> {
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
  m
}

fn tiny_model() -> Qwen3AsrTextModel {
  let cfg = tiny_config();
  let mut w = tiny_weights(&cfg);
  Qwen3AsrTextModel::from_weights(&cfg, &mut w).expect("tiny model must build")
}

#[test]
fn forward_returns_hidden_shape_and_finite() {
  let model = tiny_model();
  assert_eq!(model.num_layers(), 1);
  let ids = Array::from_slice::<i32>(&[1, 2, 3], &[1, 3]).unwrap();
  let mut cache = model.make_cache();
  let mut hidden = model.forward_tokens(&ids, &mut cache).unwrap();
  assert_eq!(hidden.shape(), vec![1, 3, HIDDEN as usize]);
  let vals = hidden.to_vec::<f32>().unwrap();
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
}

/// Build the tiny text decoder with every weight cast to `dtype` — a
/// half-precision checkpoint whose embedding table sets the activation dtype.
fn tiny_model_dtype(dtype: crate::Dtype) -> Qwen3AsrTextModel {
  let cfg = tiny_config();
  let mut w: HashMap<String, Array> = tiny_weights(&cfg)
    .into_iter()
    .map(|(k, v)| (k, v.astype(dtype).expect("weight cast")))
    .collect();
  Qwen3AsrTextModel::from_weights(&cfg, &mut w).expect("tiny model must build")
}

/// The MRoPE text forward (`forward_hidden`) must preserve the activation dtype:
/// a `bf16`/`f16` checkpoint must yield `bf16`/`f16` hidden states, never a
/// silent promotion to `f32`. The f32-built rotary `cos`/`sin` are the upcast
/// risk this pins (the manual MRoPE must cast them back, like the fused
/// `nn.RoPE` the reference applies).
fn assert_forward_hidden_preserves_dtype(dtype: crate::Dtype) {
  let model = tiny_model_dtype(dtype);
  let ids = Array::from_slice::<i32>(&[1, 2, 3], &[1, 3]).unwrap();
  let mut cache = model.make_cache();
  let h = model.embed_tokens(&ids).unwrap();
  assert_eq!(h.dtype().unwrap(), dtype, "embed_tokens upcast {dtype:?}");
  let hidden = model.forward_hidden(&h, &mut cache).unwrap();
  assert_eq!(
    hidden.dtype().unwrap(),
    dtype,
    "forward_hidden upcast {dtype:?} → {:?}",
    hidden.dtype().unwrap()
  );
  // Read via an f32 view (`to_vec::<f32>` requires an f32 array).
  let vals = hidden
    .astype(crate::Dtype::F32)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
}

#[test]
fn forward_hidden_preserves_bf16() {
  assert_forward_hidden_preserves_dtype(crate::Dtype::BF16);
}

#[test]
fn forward_hidden_preserves_f16() {
  assert_forward_hidden_preserves_dtype(crate::Dtype::F16);
}

#[test]
fn forward_rejects_wrong_cache_cardinality() {
  let model = tiny_model();
  let ids = Array::from_slice::<i32>(&[1, 2], &[1, 2]).unwrap();
  let h = model.embed_tokens(&ids).unwrap();
  // Two caches for a one-layer model.
  let mut cache = model.make_cache();
  cache.push(Box::new(crate::lm::cache::StandardKvCache::new()));
  assert!(matches!(
    model.forward_hidden(&h, &mut cache),
    Err(crate::error::Error::LengthMismatch(_))
  ));
}

#[test]
fn forward_rejects_non_rank3_hidden_states() {
  // The decoder forward is public and accepts an arbitrary `Array`; a rank-0 /
  // rank-1 / rank-2 input must surface a typed RankMismatch, never an
  // index-out-of-bounds panic on `shape[0]`/`shape[1]`.
  let model = tiny_model();
  for shape in [vec![], vec![HIDDEN], vec![1, HIDDEN]] {
    let mut cache = model.make_cache();
    let h = filled(&shape, 0.1);
    assert!(
      matches!(
        model.forward_hidden(&h, &mut cache),
        Err(crate::error::Error::RankMismatch(_))
      ),
      "rank-{} hidden states must be a RankMismatch",
      shape.len()
    );
  }
}

#[test]
fn forward_rejects_wrong_hidden_width() {
  // A rank-3 input whose hidden axis disagrees with the token-embedding width
  // must be a typed ShapePairMismatch, not a downstream matmul panic.
  let model = tiny_model();
  let mut cache = model.make_cache();
  let h = filled(&[1, 2, HIDDEN + 1], 0.1);
  assert!(matches!(
    model.forward_hidden(&h, &mut cache),
    Err(crate::error::Error::ShapePairMismatch(_))
  ));
}

#[test]
fn forward_with_positions_rejects_non_rank3_position_ids() {
  // `forward_hidden_with_positions` is public and accepts a caller-supplied
  // `position_ids`. A rank-0 / rank-1 / rank-2 explicit position tensor must
  // surface a typed RankMismatch BEFORE `cos_sin` reaches any MLX op, never an
  // opaque MLX take/transpose error.
  let model = tiny_model();
  let h = filled(&[1, 3, HIDDEN], 0.1); // valid rank-3 (batch 1, seq 3)
  for shape in [vec![], vec![3], vec![3, 3]] {
    let mut cache = model.make_cache();
    let pos = Array::full::<i32>(&shape, 0).unwrap();
    assert!(
      matches!(
        model.forward_hidden_with_positions(&h, Some(&pos), &mut cache),
        Err(crate::error::Error::RankMismatch(_))
      ),
      "rank-{} position_ids must be a RankMismatch",
      shape.len()
    );
  }
}

#[test]
fn forward_with_positions_rejects_wrong_leading_dim() {
  // A rank-3 `position_ids` whose leading axis is not 3 (the temporal/height/
  // width MRoPE axes) must be a typed ShapePairMismatch, not a downstream MLX
  // gather error.
  let model = tiny_model();
  let h = filled(&[1, 3, HIDDEN], 0.1); // batch 1, seq 3
  // (2, 1, 3) — wrong leading dim (2 instead of 3).
  let pos = Array::full::<i32>(&[2, 1, 3], 0).unwrap();
  let mut cache = model.make_cache();
  assert!(matches!(
    model.forward_hidden_with_positions(&h, Some(&pos), &mut cache),
    Err(crate::error::Error::ShapePairMismatch(_))
  ));
}

#[test]
fn forward_with_positions_rejects_broadcastable_wrong_batch() {
  // The silent-corruption case: `position_ids` of `(3, 1, seq)` for batch > 1
  // would broadcast one position row across the whole batch (wrong rotations).
  // It must be rejected as a typed ShapePairMismatch before `cos_sin`.
  let model = tiny_model();
  // h: batch 2, seq 3.
  let h = filled(&[2, 3, HIDDEN], 0.1);
  // (3, 1, 3) — broadcastable but wrong batch (1 instead of 2).
  let pos = Array::full::<i32>(&[3, 1, 3], 0).unwrap();
  let mut cache = model.make_cache();
  assert!(matches!(
    model.forward_hidden_with_positions(&h, Some(&pos), &mut cache),
    Err(crate::error::Error::ShapePairMismatch(_))
  ));
}

#[test]
fn forward_with_positions_rejects_wrong_seq() {
  // A rank-3 `position_ids` whose sequence axis disagrees with `h`'s seq must be
  // a typed ShapePairMismatch (not an MLX error from a later op).
  let model = tiny_model();
  let h = filled(&[1, 3, HIDDEN], 0.1); // batch 1, seq 3
  // (3, 1, 4) — wrong seq (4 instead of 3).
  let pos = Array::full::<i32>(&[3, 1, 4], 0).unwrap();
  let mut cache = model.make_cache();
  assert!(matches!(
    model.forward_hidden_with_positions(&h, Some(&pos), &mut cache),
    Err(crate::error::Error::ShapePairMismatch(_))
  ));
}

#[test]
fn forward_with_positions_accepts_exact_shape() {
  // The positive control: an exact `(3, batch, seq)` text-only position tensor
  // is accepted and produces the documented `(B, L, hidden)` hidden states.
  let model = tiny_model();
  let h = filled(&[1, 3, HIDDEN], 0.1); // batch 1, seq 3
  // (3, 1, 3): every MRoPE axis equal to the sequence index.
  let pos = Array::from_slice::<i32>(&[0, 1, 2, 0, 1, 2, 0, 1, 2], &[3, 1, 3]).unwrap();
  let mut cache = model.make_cache();
  let mut hidden = model
    .forward_hidden_with_positions(&h, Some(&pos), &mut cache)
    .expect("exact-shape position_ids must be accepted");
  assert_eq!(hidden.shape(), vec![1, 3, HIDDEN as usize]);
  let vals = hidden.to_vec::<f32>().unwrap();
  assert!(vals.iter().all(|v| v.is_finite()), "non-finite: {vals:?}");
}

#[test]
fn from_weights_missing_key_is_typed_error() {
  let cfg = tiny_config();
  let mut w = tiny_weights(&cfg);
  w.remove("model.norm.weight");
  assert!(matches!(
    Qwen3AsrTextModel::from_weights(&cfg, &mut w),
    Err(crate::error::Error::MissingKey(_))
  ));
}

// ─────────────── malformed decoder weight shapes (Finding 3) ───

/// Assert that replacing decoder weight `key` with a tensor of the wrong
/// `shape` makes `from_weights` reject it as a typed shape error (a
/// `LayerKeyed` wrapping a `ShapePairMismatch`), naming the key — rather than
/// storing it and failing later as an opaque MLX error (or silently changing
/// the accepted hidden width via the embedding table).
fn assert_wrong_shape_rejected(key: &str, shape: &[i32]) {
  let cfg = tiny_config();
  let mut w = tiny_weights(&cfg);
  assert!(w.contains_key(key), "fixture must carry {key}");
  w.insert(key.to_string(), filled(shape, 0.05));
  match Qwen3AsrTextModel::from_weights(&cfg, &mut w) {
    Err(crate::error::Error::LayerKeyed(p)) => {
      assert_eq!(p.layer(), key, "the error must name the offending key");
      assert!(
        matches!(p.inner(), crate::error::Error::ShapePairMismatch(_)),
        "inner must be a ShapePairMismatch, got {:?}",
        p.inner()
      );
    }
    other => panic!("expected LayerKeyed(ShapePairMismatch) for {key}, got {other:?}"),
  }
}

#[test]
fn from_weights_rejects_wrong_embed_tokens_shape() {
  // A wrong embedding width would silently change the accepted hidden axis
  // (forward reads embed_tokens.shape()[1]); reject it at load.
  assert_wrong_shape_rejected("model.embed_tokens.weight", &[VOCAB, HIDDEN + 1]);
  // A wrong vocab axis is likewise rejected.
  assert_wrong_shape_rejected("model.embed_tokens.weight", &[VOCAB + 1, HIDDEN]);
}

#[test]
fn from_weights_rejects_wrong_norm_shape() {
  assert_wrong_shape_rejected("model.norm.weight", &[HIDDEN + 1]);
}

#[test]
fn from_weights_rejects_wrong_attn_projection_shapes() {
  // q/k/v/o projections and the q/k norms each have a config-derived shape.
  assert_wrong_shape_rejected(
    "model.layers.0.self_attn.q_proj.weight",
    &[N_HEADS * HEAD_DIM + 1, HIDDEN],
  );
  assert_wrong_shape_rejected(
    "model.layers.0.self_attn.k_proj.weight",
    &[N_KV * HEAD_DIM, HIDDEN + 1],
  );
  assert_wrong_shape_rejected(
    "model.layers.0.self_attn.v_proj.weight",
    &[N_KV * HEAD_DIM + 1, HIDDEN],
  );
  assert_wrong_shape_rejected(
    "model.layers.0.self_attn.o_proj.weight",
    &[HIDDEN, N_HEADS * HEAD_DIM + 1],
  );
  assert_wrong_shape_rejected("model.layers.0.self_attn.q_norm.weight", &[HEAD_DIM + 1]);
  assert_wrong_shape_rejected("model.layers.0.self_attn.k_norm.weight", &[HEAD_DIM + 1]);
}

#[test]
fn from_weights_rejects_wrong_mlp_shapes() {
  let inter = tiny_config().intermediate_size;
  assert_wrong_shape_rejected("model.layers.0.mlp.gate_proj.weight", &[inter + 1, HIDDEN]);
  assert_wrong_shape_rejected("model.layers.0.mlp.up_proj.weight", &[inter, HIDDEN + 1]);
  assert_wrong_shape_rejected("model.layers.0.mlp.down_proj.weight", &[HIDDEN, inter + 1]);
}

#[test]
fn from_weights_rejects_wrong_layernorm_shapes() {
  assert_wrong_shape_rejected("model.layers.0.input_layernorm.weight", &[HIDDEN + 1]);
  assert_wrong_shape_rejected(
    "model.layers.0.post_attention_layernorm.weight",
    &[HIDDEN + 1],
  );
}

#[test]
fn from_weights_rejects_wrong_rank_weight() {
  // A rank mismatch (a 2-D norm instead of 1-D) is caught by the same shape
  // check, not an opaque later error.
  assert_wrong_shape_rejected("model.norm.weight", &[HIDDEN, 1]);
}

// ════════════════════════════ config ════════════════════════════

#[test]
fn config_default_mrope_is_standard_rope_fallback() {
  let cfg = Qwen3AsrTextConfig::default();
  let mrope = cfg.mrope().unwrap();
  // head_dim 128 → half 64; fallback puts everything on temporal.
  assert_eq!(mrope.section, [64, 0, 0]);
  assert!(!mrope.interleaved);
}

#[test]
fn config_parses_interleaved_mrope_section() {
  let json = r#"{
    "head_dim": 128,
    "rope_scaling": {"mrope_section": [24, 20, 20], "interleaved": true}
  }"#;
  let cfg = Qwen3AsrTextConfig::from_json(json).unwrap();
  let mrope = cfg.mrope().unwrap();
  assert_eq!(mrope.section, [24, 20, 20]);
  assert!(mrope.interleaved);
}

#[test]
fn config_accepts_mrope_interleaved_alias() {
  let json = r#"{
    "head_dim": 128,
    "rope_scaling": {"mrope_section": [24, 20, 20], "mrope_interleaved": true}
  }"#;
  let cfg = Qwen3AsrTextConfig::from_json(json).unwrap();
  assert!(cfg.mrope().unwrap().interleaved);
}

#[test]
fn config_rejects_mrope_section_wrong_length() {
  let json = r#"{"head_dim": 128, "rope_scaling": {"mrope_section": [32, 32]}}"#;
  assert!(matches!(
    Qwen3AsrTextConfig::from_json(json),
    Err(crate::error::Error::OutOfRange(_))
  ));
}

#[test]
fn config_rejects_mrope_section_wrong_sum() {
  // head_dim 128 → half 64, but 10+10+10 = 30.
  let json = r#"{"head_dim": 128, "rope_scaling": {"mrope_section": [10, 10, 10]}}"#;
  assert!(matches!(
    Qwen3AsrTextConfig::from_json(json),
    Err(crate::error::Error::OutOfRange(_))
  ));
}

#[test]
fn config_rejects_nonnull_rope_scaling_without_section() {
  let json = r#"{"head_dim": 128, "rope_scaling": {"rope_type": "default"}}"#;
  assert!(matches!(
    Qwen3AsrTextConfig::from_json(json),
    Err(crate::error::Error::OutOfRange(_))
  ));
}

#[test]
fn config_accepts_released_default_mrope_rope_type() {
  // The released Qwen3-ASR text_config: an explicit default-MRoPE rope_scaling
  // (`rope_type: "default"` + a 3-section mrope_section summing to head_dim/2).
  // This is the only path the decoder implements, so it must load.
  let json = r#"{
    "head_dim": 128,
    "rope_scaling": {
      "rope_type": "default",
      "mrope_section": [24, 20, 20],
      "interleaved": true
    }
  }"#;
  let cfg = Qwen3AsrTextConfig::from_json(json).unwrap();
  let mrope = cfg.mrope().unwrap();
  assert_eq!(mrope.section, [24, 20, 20]);
  assert!(mrope.interleaved);
}

#[test]
fn config_accepts_mrope_rope_type_via_type_alias() {
  // The Qwen VL / Omni configs spell the field `type` (not `rope_type`) and use
  // the value `"mrope"`; both must be accepted as the default base-theta path.
  let json = r#"{
    "head_dim": 128,
    "rope_scaling": {"type": "mrope", "mrope_section": [32, 16, 16]}
  }"#;
  let cfg = Qwen3AsrTextConfig::from_json(json).unwrap();
  assert_eq!(cfg.mrope().unwrap().section, [32, 16, 16]);
}

#[test]
fn config_rejects_non_default_rope_type() {
  // A scaling rope_type the port does not implement (yarn introduces a non-unity
  // attention_scaling and a different inverse-frequency formula). It must fail
  // fast with the typed UnknownEnumValue error, never load as plain base-theta.
  let json = r#"{
    "head_dim": 128,
    "rope_scaling": {
      "rope_type": "yarn",
      "factor": 4.0,
      "mrope_section": [24, 20, 20]
    }
  }"#;
  match Qwen3AsrTextConfig::from_json(json) {
    Err(crate::error::Error::UnknownEnumValue(p)) => {
      assert_eq!(p.value(), "yarn");
      assert!(p.supported().contains(&"default"));
      assert!(p.supported().contains(&"mrope"));
    }
    other => panic!("expected UnknownEnumValue for a yarn rope_type, got {other:?}"),
  }
}

#[test]
fn config_rejects_non_default_rope_type_via_type_alias() {
  // The `type` alias must be gated identically (and take precedence over
  // `rope_type` per the reference `initialize_rope`).
  let json = r#"{
    "head_dim": 128,
    "rope_scaling": {"type": "linear", "factor": 2.0, "mrope_section": [24, 20, 20]}
  }"#;
  match Qwen3AsrTextConfig::from_json(json) {
    Err(crate::error::Error::UnknownEnumValue(p)) => assert_eq!(p.value(), "linear"),
    other => panic!("expected UnknownEnumValue for a linear rope_type, got {other:?}"),
  }
}

#[test]
fn config_null_rope_scaling_is_fallback() {
  let json = r#"{"head_dim": 64, "rope_scaling": null}"#;
  let cfg = Qwen3AsrTextConfig::from_json(json).unwrap();
  assert_eq!(cfg.mrope().unwrap().section, [32, 0, 0]);
}

#[test]
fn config_rejects_odd_head_dim() {
  let json = r#"{"head_dim": 5, "num_attention_heads": 1, "num_key_value_heads": 1}"#;
  assert!(Qwen3AsrTextConfig::from_json(json).is_err());
}

// ───────────── from_weights validates the config ───
//
// `Qwen3AsrTextModel::from_weights` takes a `&Qwen3AsrTextConfig` whose fields
// are public, so a caller can mutate a parsed/default config into a structurally
// invalid one and call the constructor directly (bypassing `from_json`'s
// validation). Each such config must be rejected with a typed error at the START
// of `from_weights` — BEFORE any head-count / projection-width arithmetic — not
// admitted into a model whose forward later divides by zero in SDPA.

// Each test hands `from_weights` an EMPTY weight map: the config gate runs
// FIRST (before any weight is consulted or any head/shape arithmetic), so a
// valid config would fail later with a MissingKey while a rejected config
// errors on `validate` up front — and an empty map also avoids the fixture
// itself panicking when it tries to build a negative-shape weight from the
// malformed head count.

#[test]
fn from_weights_rejects_zero_kv_heads() {
  // num_key_value_heads == 0: the derived kv projection width is 0, and a later
  // forward would feed zero K/V heads to SDPA (`n_q_heads % 0` — a mod-by-zero /
  // UB path). from_weights must reject it as a typed OutOfRange (not a panic /
  // crash), before any weight is consulted.
  let mut cfg = tiny_config();
  cfg.num_key_value_heads = 0;
  let mut w: HashMap<String, Array> = HashMap::new();
  assert!(matches!(
    Qwen3AsrTextModel::from_weights(&cfg, &mut w),
    Err(crate::error::Error::OutOfRange(_))
  ));
}

#[test]
fn from_weights_rejects_negative_head_count() {
  // A negative num_attention_heads is malformed (it sizes per-head reshapes and
  // is a divisor); from_weights must reject it as a typed OutOfRange, never
  // panic on the negative-count arithmetic.
  let mut cfg = tiny_config();
  cfg.num_attention_heads = -2;
  let mut w: HashMap<String, Array> = HashMap::new();
  assert!(matches!(
    Qwen3AsrTextModel::from_weights(&cfg, &mut w),
    Err(crate::error::Error::OutOfRange(_))
  ));
}

#[test]
fn from_weights_rejects_non_finite_rms_norm_eps() {
  // A non-finite rms_norm_eps would poison every RMSNorm; from_weights must
  // reject it with a typed NonFiniteScalar error, not build the model.
  let mut cfg = tiny_config();
  cfg.rms_norm_eps = f32::NAN;
  let mut w: HashMap<String, Array> = HashMap::new();
  assert!(matches!(
    Qwen3AsrTextModel::from_weights(&cfg, &mut w),
    Err(crate::error::Error::NonFiniteScalar(_))
  ));
}

#[test]
fn from_weights_rejects_non_divisible_gqa_grouping() {
  // num_attention_heads not divisible by num_key_value_heads breaks the GQA
  // head grouping; from_weights must reject it with a typed error rather than
  // build a model whose K/V head repeat is ill-defined.
  let mut cfg = tiny_config();
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 3; // 4 % 3 != 0
  let mut w: HashMap<String, Array> = HashMap::new();
  assert!(
    Qwen3AsrTextModel::from_weights(&cfg, &mut w).is_err(),
    "a non-divisible GQA grouping must be rejected"
  );
}
