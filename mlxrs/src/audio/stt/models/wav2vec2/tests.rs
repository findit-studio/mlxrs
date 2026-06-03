//! Oracle tests for the Wav2Vec2 CTC port.
//!
//! Every expected value is computed independently of the code under test —
//! by hand-collapsing reference CTC sequences, by the analytic conv
//! time-dimension recurrence, by closed-form GroupNorm / weight-norm
//! arithmetic, or against the verbatim test inputs (sanitize / vocab) — never
//! by invoking the implementation a second time.

use super::*;
use crate::error::Error;

// ───────────────────────── test 1: CTC greedy collapse ─────────────────────────

#[test]
fn ctc_collapse_drops_blanks_and_dedups_runs() {
  // Reference (mms.py:33-45): emit when `token != prev && token != 0`, update
  // `prev` every frame. Hand-collapsed:
  //   [5,5,0,5,3,3] -> 5 (emit), 5 (==prev skip), 0 (blank), 5 (emit again,
  //                    prev was 0), 3 (emit), 3 (==prev skip) = [5,5,3]
  assert_eq!(ctc_greedy_collapse(&[5, 5, 0, 5, 3, 3]), vec![5, 5, 3]);
  //   [5,5,5] -> [5]
  assert_eq!(ctc_greedy_collapse(&[5, 5, 5]), vec![5]);
  //   [0,0,5,0] -> only the single 5 survives = [5]
  assert_eq!(ctc_greedy_collapse(&[0, 0, 5, 0]), vec![5]);
}

#[test]
fn ctc_collapse_edge_cases() {
  // Empty input -> empty.
  assert_eq!(ctc_greedy_collapse(&[]), Vec::<u32>::new());
  // All blanks -> empty.
  assert_eq!(ctc_greedy_collapse(&[0, 0, 0]), Vec::<u32>::new());
  // Leading non-blank emits immediately (prev sentinel != 0).
  assert_eq!(ctc_greedy_collapse(&[7]), vec![7]);
  // A blank between identical tokens splits the run into two emissions.
  assert_eq!(ctc_greedy_collapse(&[4, 0, 4]), vec![4, 4]);
  // No blanks, alternating: every transition emits.
  assert_eq!(ctc_greedy_collapse(&[1, 2, 1, 2]), vec![1, 2, 1, 2]);
}

// ───────────────────────── test 5: vocabulary ─────────────────────────

/// A miniature `vocab.json` body covering the structural cases: blank id 0,
/// the word-delimiter `|`, and ordinary letters. Mirrors the `base-960h`
/// vocab.json shape `{token: id}`.
fn mini_vocab_json() -> &'static str {
  r#"{"<pad>": 0, "|": 1, "H": 2, "I": 3}"#
}

#[test]
fn vocab_parses_and_inverts() {
  let vocab = Vocab::from_json(mini_vocab_json()).unwrap();
  // Highest id is 3 -> 4 slots (0..=3).
  assert_eq!(vocab.len(), 4);
  assert!(!vocab.is_empty());
  // Inverted id -> token (compared against the literal test input, not the
  // implementation).
  assert_eq!(vocab.token(0), Some("<pad>"));
  assert_eq!(vocab.token(1), Some("|"));
  assert_eq!(vocab.token(2), Some("H"));
  assert_eq!(vocab.token(3), Some("I"));
  // Out-of-range id -> None.
  assert_eq!(vocab.token(4), None);
}

#[test]
fn vocab_tokens_to_text_maps_pipe_to_space() {
  let vocab = Vocab::from_json(mini_vocab_json()).unwrap();
  // Decoded ids [2,3,1,2,3] -> "HI HI": "H"+"I"+"|"+"H"+"I" then |->space.
  assert_eq!(vocab.tokens_to_text(&[2, 3, 1, 2, 3]), "HI HI");
  // Unknown id (4) contributes nothing.
  assert_eq!(vocab.tokens_to_text(&[2, 4, 3]), "HI");
  // The blank id 0 maps to its literal token here ("<pad>") since
  // tokens_to_text does not itself filter blanks — that is ctc_greedy_collapse's
  // job. This documents the separation of concerns.
  assert_eq!(vocab.tokens_to_text(&[2, 3]), "HI");
}

#[test]
fn vocab_rejects_nested_multilingual_json() {
  // MMS multilingual `{lang: {token: id}}` is unsupported for base-960h; a
  // nested object fails to deserialize as `{string: i64}`.
  let nested = r#"{"eng": {"<pad>": 0, "A": 1}}"#;
  assert!(matches!(Vocab::from_json(nested), Err(Error::Parse(_))));
}

#[test]
fn vocab_empty_is_empty() {
  let vocab = Vocab::from_json("{}").unwrap();
  assert!(vocab.is_empty());
  assert_eq!(vocab.len(), 0);
  assert_eq!(vocab.tokens_to_text(&[0, 1, 2]), "");
}

#[test]
fn vocab_accepts_large_but_valid_id() {
  // The library imposes no magnitude ceiling on a token id: a large-but-valid
  // id allocates its dense `id → token` table and parses. (Bounding a
  // legitimately large vocabulary is the caller's concern; a pathological id is
  // bounded by fallible allocation, not a fixed cap.) 2^20 is a ~1M-slot table,
  // trivially within memory yet well past any pinned ceiling.
  let high = 1i64 << 20;
  let json = format!(r#"{{"A": {high}}}"#);
  let vocab = Vocab::from_json(&json).unwrap();
  // Highest id is exactly 2^20 -> 2^20 + 1 slots, only the top one populated.
  assert_eq!(vocab.len(), (1usize << 20) + 1);
  assert_eq!(vocab.token(1 << 20), Some("A"));
}

#[test]
fn vocab_rejects_all_negative_ids() {
  // A NON-EMPTY map whose every id is negative is malformed: inverting it
  // would silently drop the entire vocabulary. It must be a typed
  // MalformedData, distinct from the legitimately-empty `{}` (which is Ok).
  let all_neg = r#"{"A": -1, "B": -3}"#;
  assert!(matches!(
    Vocab::from_json(all_neg),
    Err(Error::MalformedData(_))
  ));
  // A single negative id is likewise malformed when it is the only entry.
  let one_neg = r#"{"A": -1}"#;
  assert!(matches!(
    Vocab::from_json(one_neg),
    Err(Error::MalformedData(_))
  ));
}

#[test]
fn vocab_rejects_negative_id_mixed_with_valid() {
  // A negative id alongside valid ones (so max_id >= 0, the table IS
  // allocated) is rejected per-entry with OutOfRange rather than silently
  // skipped or panicking on a wrapped index.
  let mixed = r#"{"<pad>": 0, "A": 1, "BAD": -2}"#;
  assert!(matches!(Vocab::from_json(mixed), Err(Error::OutOfRange(_))));
}

#[test]
fn vocab_rejects_duplicate_ids() {
  // Two distinct tokens mapping to the SAME id is malformed: the source is a
  // HashMap, so a bare slot overwrite would keep an arbitrary (per-run
  // nondeterministic) survivor and silently corrupt the vocabulary. It must be
  // a typed MalformedData, never a silent overwrite.
  let dup = r#"{"A": 1, "B": 1}"#;
  match Vocab::from_json(dup) {
    Err(Error::MalformedData(p)) => {
      assert_eq!(p.context(), "Vocab::from_json");
      assert!(
        p.detail().contains("same id"),
        "detail should describe the duplicate-id condition, got {:?}",
        p.detail()
      );
    }
    other => panic!("expected MalformedData for duplicate ids, got {other:?}"),
  }
  // A duplicate at id 0 (two tokens both claiming the blank slot) is likewise
  // rejected, regardless of which token the HashMap happens to visit first.
  let dup0 = r#"{"<pad>": 0, "<s>": 0, "A": 1}"#;
  assert!(matches!(
    Vocab::from_json(dup0),
    Err(Error::MalformedData(_))
  ));
}

// ───────────────────────── test 6: sanitize ─────────────────────────

#[test]
fn sanitize_swaps_conv_axes_renames_params_drops_training_keeps_lm_head() {
  let mut weights: HashMap<String, Array> = HashMap::new();
  // A conv weight (out=2, in=1, k=3) — HF layout; sanitize swaps to (out, k, in).
  weights.insert(
    "wav2vec2.feature_extractor.conv_layers.0.conv.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]).unwrap(),
  );
  // A weight-norm parametrization pair (out=2, in=1, k=3) on the pos conv.
  weights.insert(
    "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0".to_string(),
    Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 1, 3]).unwrap(),
  );
  weights.insert(
    "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]).unwrap(),
  );
  // The CTC head — must be KEPT and NOT prefix-stripped.
  weights.insert(
    "lm_head.weight".to_string(),
    Array::from_slice::<f32>(&[0.0, 1.0], &[2, 1]).unwrap(),
  );
  // Training-only keys — must be DROPPED.
  weights.insert(
    "quantizer.codevectors".to_string(),
    Array::from_slice::<f32>(&[0.0], &[1]).unwrap(),
  );
  weights.insert(
    "project_hid.weight".to_string(),
    Array::from_slice::<f32>(&[0.0], &[1]).unwrap(),
  );
  weights.insert(
    "masked_spec_embed".to_string(),
    Array::from_slice::<f32>(&[0.0], &[1]).unwrap(),
  );

  let out = sanitize(weights).unwrap();

  // Backbone prefix stripped; conv weight axis-swapped to (out=2, k=3, in=1).
  let conv = out
    .get("feature_extractor.conv_layers.0.conv.weight")
    .expect("conv key present with prefix stripped");
  assert_eq!(conv.shape(), vec![2, 3, 1]);

  // Parametrization renamed and axis-swapped.
  let wg = out
    .get("encoder.pos_conv_embed.conv.weight_g")
    .expect("original0 renamed to weight_g");
  assert_eq!(wg.shape(), vec![2, 3, 1]);
  let wv = out
    .get("encoder.pos_conv_embed.conv.weight_v")
    .expect("original1 renamed to weight_v");
  assert_eq!(wv.shape(), vec![2, 3, 1]);

  // lm_head kept verbatim (NOT prefix-stripped — it has no wav2vec2. prefix).
  assert!(out.contains_key("lm_head.weight"));

  // Training-only keys dropped.
  assert!(!out.contains_key("quantizer.codevectors"));
  assert!(!out.contains_key("project_hid.weight"));
  assert!(!out.contains_key("masked_spec_embed"));
}

#[test]
fn sanitize_strips_supported_backbone_prefixes() {
  // Each `*ForCTC` nests its backbone under a different prefix: Wav2Vec2ForCTC
  // under `wav2vec2.`, HubertForCTC under `hubert.`. sanitize must strip BOTH
  // to the same unprefixed `feature_extractor.*` / `encoder.*` keys the
  // builders expect, while leaving the top-level `lm_head.*` untouched. (The
  // earlier tests fed already-unprefixed synthetic keys; this pins the strip
  // itself for both backbones.)
  for prefix in ["wav2vec2.", "hubert."] {
    let mut weights: HashMap<String, Array> = HashMap::new();
    // A non-conv backbone tensor (no axis swap) under the backbone prefix.
    weights.insert(
      format!("{prefix}encoder.layer_norm.weight"),
      Array::from_slice::<f32>(&[1.0, 2.0], &[2]).unwrap(),
    );
    // A conv weight under the backbone prefix (also exercises the axis swap on
    // the post-strip key).
    weights.insert(
      format!("{prefix}feature_extractor.conv_layers.0.conv.weight"),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]).unwrap(),
    );
    // The CTC head is top-level (no backbone prefix) and must be kept verbatim.
    weights.insert(
      "lm_head.weight".to_string(),
      Array::from_slice::<f32>(&[0.0, 1.0], &[2, 1]).unwrap(),
    );

    let out = sanitize(weights).unwrap();

    // Prefix stripped from both backbone keys.
    assert!(
      out.contains_key("encoder.layer_norm.weight"),
      "{prefix}: encoder.layer_norm.weight must lose its backbone prefix"
    );
    let conv = out
      .get("feature_extractor.conv_layers.0.conv.weight")
      .unwrap_or_else(|| panic!("{prefix}: conv key must lose its backbone prefix"));
    // (out=2, in=1, k=3) -> (out=2, k=3, in=1) after the axis swap.
    assert_eq!(conv.shape(), vec![2, 3, 1], "{prefix}: conv axis-swapped");
    // The prefixed forms must be gone.
    assert!(!out.contains_key(&format!("{prefix}encoder.layer_norm.weight")));
    assert!(!out.contains_key(&format!(
      "{prefix}feature_extractor.conv_layers.0.conv.weight"
    )));
    // lm_head untouched.
    assert!(out.contains_key("lm_head.weight"));
  }
}

#[test]
fn sanitize_rejects_hubert_prefixed_unprefixed_collision() {
  // The collision guard covers the hubert backbone too: a checkpoint carrying
  // both `hubert.lm_head.weight` (rule 1 strips to `lm_head.weight`) and the
  // unprefixed `lm_head.weight` maps two source keys onto one destination. The
  // source is a `HashMap`, so a bare overwrite would keep an arbitrary survivor;
  // sanitize must reject it with a typed KeyCollision regardless of order.
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert(
    "hubert.lm_head.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0], &[2, 1]).unwrap(),
  );
  weights.insert(
    "lm_head.weight".to_string(),
    Array::from_slice::<f32>(&[3.0, 4.0], &[2, 1]).unwrap(),
  );
  match sanitize(weights) {
    Err(Error::KeyCollision(p)) => {
      assert_eq!(p.context(), "Wav2Vec2 sanitize");
      assert_eq!(p.key(), "lm_head.weight");
    }
    other => panic!("expected KeyCollision for hubert-prefixed+unprefixed lm_head, got {other:?}"),
  }
}

#[test]
fn sanitize_conv_axis_swap_values() {
  // (out=1, in=2, k=2) HF tensor, row-major:
  //   [[ [a,b], [c,d] ]]  with values [1,2,3,4] meaning in0=[1,2], in1=[3,4].
  // After swapaxes(1,2) -> (out=1, k=2, in=2): element (0,j,i) = old (0,i,j).
  //   new[0,0,0]=old[0,0,0]=1, new[0,0,1]=old[0,1,0]=3,
  //   new[0,1,0]=old[0,0,1]=2, new[0,1,1]=old[0,1,1]=4.
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert(
    "feature_extractor.conv_layers.1.conv.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap(),
  );
  let mut out = sanitize(weights).unwrap();
  let w = out
    .remove("feature_extractor.conv_layers.1.conv.weight")
    .unwrap();
  assert_eq!(w.shape(), vec![1, 2, 2]);
  // `swapaxes` yields a non-contiguous strided view; materialize a
  // row-contiguous copy before reading the flat buffer. (The production
  // path hands the lazy view straight to conv1d and never reads it flat,
  // so this copy is a test-only concern.)
  let mut contiguous = ops::shape::contiguous(&w, false).unwrap();
  assert_eq!(
    contiguous.to_vec::<f32>().unwrap(),
    vec![1.0, 3.0, 2.0, 4.0]
  );
}

#[test]
fn sanitize_rejects_prefixed_unprefixed_collision() {
  // A checkpoint carrying BOTH the prefixed (`wav2vec2.lm_head.weight`, which
  // rule 1 strips to `lm_head.weight`) and the unprefixed (`lm_head.weight`)
  // form maps two source keys onto the same destination. The source is a
  // `HashMap`, so a bare overwrite would keep an arbitrary (per-run
  // nondeterministic) survivor; sanitize must instead reject it with a typed
  // KeyCollision regardless of HashMap visitation order.
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert(
    "wav2vec2.lm_head.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0], &[2, 1]).unwrap(),
  );
  weights.insert(
    "lm_head.weight".to_string(),
    Array::from_slice::<f32>(&[3.0, 4.0], &[2, 1]).unwrap(),
  );
  match sanitize(weights) {
    Err(Error::KeyCollision(p)) => {
      assert_eq!(p.context(), "Wav2Vec2 sanitize");
      assert_eq!(p.key(), "lm_head.weight");
    }
    other => panic!("expected KeyCollision for prefixed+unprefixed lm_head, got {other:?}"),
  }
}

#[test]
fn sanitize_rejects_parametrization_legacy_weight_g_collision() {
  // A checkpoint carrying both `...conv.parametrizations.weight.original0`
  // (rule 3 renames it to `...conv.weight_g`) and a legacy `...conv.weight_g`
  // (rule 2, axis-swapped) also collides on the same destination key. Both
  // forms are rank-3 (they take the conv axis-swap path); the collision is a
  // typed KeyCollision, never a silent overwrite.
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert(
    "encoder.pos_conv_embed.conv.parametrizations.weight.original0".to_string(),
    Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 1, 3]).unwrap(),
  );
  weights.insert(
    "encoder.pos_conv_embed.conv.weight_g".to_string(),
    Array::from_slice::<f32>(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], &[2, 1, 3]).unwrap(),
  );
  match sanitize(weights) {
    Err(Error::KeyCollision(p)) => {
      assert_eq!(p.context(), "Wav2Vec2 sanitize");
      assert_eq!(p.key(), "encoder.pos_conv_embed.conv.weight_g");
    }
    other => panic!("expected KeyCollision for original0 + legacy weight_g, got {other:?}"),
  }
}

// ───────────────────────── config parse ─────────────────────────

#[test]
fn config_parses_base_960h_defaults_and_ignores_unknown() {
  // Minimal config carrying an unmodeled key — must parse, ignore the extra
  // key, and fall back to base-960h defaults for absent fields.
  let json = r#"{ "model_type": "wav2vec2", "future_unknown_key": 123 }"#;
  let config = Config::from_json(json).unwrap();
  assert_eq!(config.model_type(), "wav2vec2");
  assert_eq!(config.hidden_size, 768);
  assert_eq!(config.num_hidden_layers, 12);
  assert_eq!(config.num_attention_heads, 12);
  assert_eq!(config.intermediate_size, 3072);
  assert_eq!(config.vocab_size, 32);
  assert_eq!(config.conv_stride, vec![5, 2, 2, 2, 2, 2, 2]);
  assert_eq!(config.conv_kernel, vec![10, 3, 3, 3, 3, 2, 2]);
  assert_eq!(config.num_conv_pos_embeddings, 128);
  assert_eq!(config.num_conv_pos_embedding_groups, 16);
  assert!(config.is_group_norm());
  assert_eq!(config.hidden_act, "gelu");
  assert_eq!(config.feat_extract_activation, "gelu");
  assert!((config.layer_norm_eps - 1e-5).abs() < 1e-12);
  assert!(!config.do_stable_layer_norm);
  assert!(!config.conv_bias);
  assert!(!config.add_adapter);
  assert_eq!(config.adapter_attn_dim, None);
  assert_eq!(config.pad_token_id, 0);
  // HuBERT-only flags fall back to their HF defaults (the wired arm) when
  // absent — feat_proj_layer_norm = true (apply the projection LayerNorm),
  // conv_pos_batch_norm = false (weight-norm positional conv).
  assert!(config.feat_proj_layer_norm);
  assert!(!config.conv_pos_batch_norm);
}

#[test]
fn config_head_dim() {
  let config = Config::from_json(r#"{"hidden_size": 768, "num_attention_heads": 12}"#).unwrap();
  assert_eq!(config.head_dim().unwrap(), 64);
}

#[test]
fn config_validate_accepts_base_960h() {
  // The base-960h defaults (feat_extract_norm == "group",
  // do_stable_layer_norm == false) are the one supported arm.
  let config = Config::from_json(r#"{"model_type": "wav2vec2"}"#).unwrap();
  assert!(config.validate().is_ok());
}

#[test]
fn config_validate_accepts_both_feat_extract_norm_arms_rejects_other() {
  // (a) BOTH feature-encoder norm arms are now wired: "group" (the default) and
  // "layer" (used by large-960h-lv60-self) must each validate.
  for norm in ["group", "layer"] {
    let cfg = Config::from_json(&format!(r#"{{"feat_extract_norm": "{norm}"}}"#)).unwrap();
    assert!(
      cfg.validate().is_ok(),
      "feat_extract_norm == {norm:?} must validate (both arms are wired)"
    );
    assert_eq!(
      cfg.feat_extract_norm_scheme().unwrap(),
      if norm == "group" {
        FeatExtractNorm::Group
      } else {
        FeatExtractNorm::Layer
      }
    );
  }

  // (b) Any OTHER feat_extract_norm value is rejected -> UnknownEnumValue, and
  // the payload carries the rejected value + the (group/layer) supported set.
  let bad = Config::from_json(r#"{"feat_extract_norm": "instance"}"#).unwrap();
  match bad.validate() {
    Err(Error::UnknownEnumValue(p)) => {
      assert_eq!(p.value(), "instance");
      assert_eq!(p.supported(), &["group", "layer"]);
    }
    other => panic!("expected UnknownEnumValue for an unknown feat_extract_norm, got {other:?}"),
  }

  // (c) The stable-layer-norm arm is likewise SUPPORTED: a `do_stable_layer_norm`
  // config (otherwise default) must validate.
  let stable = Config::from_json(r#"{"do_stable_layer_norm": true}"#).unwrap();
  assert!(stable.do_stable_layer_norm);
  assert!(
    stable.validate().is_ok(),
    "the stable-layer-norm arm must validate (it is now wired)"
  );
}

#[test]
fn config_validate_accepts_conv_bias() {
  // `conv_bias == true` is now wired (every ConvLayer loads and adds its
  // conv.bias), so a `conv_bias` config (otherwise default) must validate.
  let biased = Config::from_json(r#"{"conv_bias": true}"#).unwrap();
  assert!(biased.conv_bias);
  assert!(
    biased.validate().is_ok(),
    "a conv_bias config must validate (the bias is now wired)"
  );
}

#[test]
fn config_validate_accepts_within_cap_layer_count_rejects_over_cap() {
  // A layer count up to the config-cardinality cap is a valid (if deep) variant:
  // `num_hidden_layers` sizes eager per-layer `Vec`s (and, on the MMS path, an
  // `O(layers)` adapter-key `Vec` + set), so it is bounded by
  // `MAX_CONFIG_CARDINALITY` (matching qwen3 / lfm2), not merely checked
  // positive. A count AT the cap validates; the within-cap reservations are made
  // fallibly by the builders / overlay, so a within-cap-but-heavyweight count
  // surfaces later as a typed `AllocFailure`, never an abort.
  let at_cap = Config::from_json(&format!(
    r#"{{"num_hidden_layers": {MAX_CONFIG_CARDINALITY}}}"#
  ))
  .unwrap();
  assert!(
    at_cap.validate().is_ok(),
    "num_hidden_layers == MAX_CONFIG_CARDINALITY must validate (the cap is inclusive)"
  );
  // An OVER-cap count is rejected as a recoverable `CapExceeded` (the
  // config-cardinality bound — NOT a DoS cap on valid input: a billion-layer
  // transformer is not a real checkpoint), so an `O(layers)` allocation is never
  // even attempted.
  let over = Config::from_json(r#"{"num_hidden_layers": 1000000}"#).unwrap();
  assert!(
    matches!(over.validate(), Err(Error::CapExceeded(_))),
    "an over-cap num_hidden_layers must be a CapExceeded"
  );
  let over_feat = Config::from_json(r#"{"num_feat_extract_layers": 1000000}"#).unwrap();
  assert!(
    matches!(over_feat.validate(), Err(Error::CapExceeded(_))),
    "an over-cap num_feat_extract_layers must be a CapExceeded"
  );
  // A zero / negative count is still rejected as malformed (OutOfRange) — that
  // is a positivity/soundness check, distinct from the cardinality cap.
  let zero = Config::from_json(r#"{"num_hidden_layers": 0}"#).unwrap();
  assert!(matches!(zero.validate(), Err(Error::OutOfRange(_))));
  let negative = Config::from_json(r#"{"num_feat_extract_layers": -1}"#).unwrap();
  assert!(matches!(negative.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn config_validate_relaxes_dimensions_but_enforces_structure() {
  // A wider hidden_size is no longer rejected as a "deviation" — it is a valid
  // larger variant. The full large config (1024 hidden, 16 heads, 24 layers,
  // 4096 intermediate, stable-LN) must validate.
  let large = Config::from_json(
    r#"{"hidden_size": 1024, "num_attention_heads": 16, "num_hidden_layers": 24,
        "intermediate_size": 4096, "do_stable_layer_norm": true}"#,
  )
  .unwrap();
  assert!(
    large.validate().is_ok(),
    "a large stable-LN config must validate"
  );

  // But the structural invariants are still enforced: a hidden_size not
  // divisible by num_attention_heads is a DivisibilityConstraint (the per-head
  // split would not be exact), and a non-positive width is OutOfRange.
  let indivisible =
    Config::from_json(r#"{"hidden_size": 1000, "num_attention_heads": 12}"#).unwrap();
  match indivisible.validate() {
    Err(Error::DivisibilityConstraint(p)) => {
      assert!(
        p.name_dividend().contains("hidden_size")
          && p.name_divisor().contains("num_attention_heads"),
        "the constraint should name hidden_size / num_attention_heads, got {p:?}"
      );
    }
    other => {
      panic!("expected DivisibilityConstraint for an indivisible hidden_size, got {other:?}")
    }
  }
  let nonpos = Config::from_json(r#"{"hidden_size": 0}"#).unwrap();
  assert!(matches!(nonpos.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn config_validate_relaxes_conv_arrays_but_enforces_length_and_positivity() {
  // A conv-stack array whose values merely DIFFER from base-960h (but are
  // positive and the right length) is now accepted — different strides are a
  // different valid variant, not a rejection.
  let diff_stride = Config::from_json(r#"{"conv_stride": [5, 3, 2, 2, 2, 2, 2]}"#).unwrap();
  assert!(
    diff_stride.validate().is_ok(),
    "a conv_stride array with positive entries of the right length must validate"
  );

  // (a) An array shorter than num_feat_extract_layers is still LengthMismatch
  // (the builder would index past the end).
  let short = Config::from_json(r#"{"conv_kernel": [10, 3, 3, 3, 3, 2]}"#).unwrap();
  match short.validate() {
    Err(Error::LengthMismatch(p)) => {
      assert!(
        p.context().contains("conv_kernel"),
        "context should name conv_kernel, got {:?}",
        p.context()
      );
      assert_eq!(p.expected(), 7);
      assert_eq!(p.actual(), 6);
    }
    other => panic!("expected LengthMismatch for a short conv_kernel, got {other:?}"),
  }

  // (b) A non-positive conv entry is still OutOfRange naming the index + value
  // (a zero / negative width, kernel, or stride is structurally invalid).
  let nonpos = Config::from_json(r#"{"conv_stride": [5, 0, 2, 2, 2, 2, 2]}"#).unwrap();
  match nonpos.validate() {
    Err(Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("conv_stride"),
        "context should name conv_stride, got {:?}",
        p.context()
      );
      assert!(
        p.value().contains("element 1") && p.value().contains("= 0"),
        "value should name the offending index and value, got {:?}",
        p.value()
      );
    }
    other => panic!("expected OutOfRange for a non-positive conv_stride entry, got {other:?}"),
  }
}

#[test]
fn config_validate_rejects_conv_array_longer_than_num_feat_extract_layers() {
  // The conv arrays must have EXACTLY `num_feat_extract_layers` entries. A
  // LONGER array (here a trailing extra entry, length 8 vs the default 7) used
  // to slip through the old `len < n` check while only the first n entries were
  // consumed — desyncing the feature encoder (which builds n layers, output
  // width conv_dim[n-1]) from build_feature_projection (which reads
  // conv_dim.last(), a LATER trailing entry of a different width). It must now
  // be rejected with a typed LengthMismatch naming the field and the
  // expected/actual lengths.
  let long_dim =
    Config::from_json(r#"{"conv_dim": [512, 512, 512, 512, 512, 512, 512, 128]}"#).unwrap();
  assert_eq!(long_dim.conv_dim.len(), 8);
  assert_eq!(long_dim.num_feat_extract_layers, 7);
  match long_dim.validate() {
    Err(Error::LengthMismatch(p)) => {
      assert!(
        p.context().contains("conv_dim"),
        "context should name conv_dim, got {:?}",
        p.context()
      );
      assert_eq!(
        p.expected(),
        7,
        "expected length is num_feat_extract_layers"
      );
      assert_eq!(p.actual(), 8, "actual length is the over-long array");
    }
    other => panic!("expected LengthMismatch for an over-long conv_dim, got {other:?}"),
  }

  // The same exact-length rule applies to conv_stride and conv_kernel: a
  // trailing extra entry on either is likewise rejected.
  let long_stride = Config::from_json(r#"{"conv_stride": [5, 2, 2, 2, 2, 2, 2, 2]}"#).unwrap();
  assert!(matches!(
    long_stride.validate(),
    Err(Error::LengthMismatch(_))
  ));
  let long_kernel = Config::from_json(r#"{"conv_kernel": [10, 3, 3, 3, 3, 2, 2, 2]}"#).unwrap();
  assert!(matches!(
    long_kernel.validate(),
    Err(Error::LengthMismatch(_))
  ));

  // The exact-length boundary: shrinking num_feat_extract_layers to match the
  // longer arrays makes the SAME arrays valid (proving the rule is exact
  // equality, not a one-sided floor) — 8-entry arrays with
  // num_feat_extract_layers = 8 must validate. (The matching transformer dims
  // are kept default; only the conv stack + layer count change.)
  let exact = Config::from_json(
    r#"{"num_feat_extract_layers": 8,
        "conv_dim": [512, 512, 512, 512, 512, 512, 512, 512],
        "conv_stride": [5, 2, 2, 2, 2, 2, 2, 2],
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2, 2]}"#,
  )
  .unwrap();
  assert!(
    exact.validate().is_ok(),
    "8-entry conv arrays with num_feat_extract_layers = 8 must validate (exact-length match)"
  );
}

#[test]
fn config_validate_rejects_non_positive_or_non_finite_layer_norm_eps() {
  // `layer_norm_eps` varies across the family, so it is no longer pinned to a
  // magnitude — a different positive finite value (e.g. the lv60 1e-5, or any
  // other positive eps) is accepted. Only a non-finite or non-positive value
  // is rejected (it would drive a non-finite / degenerate denominator).
  let smaller = Config::from_json(r#"{"layer_norm_eps": 1e-6}"#).unwrap();
  assert!(
    smaller.validate().is_ok(),
    "a different positive finite eps must validate (eps is not pinned)"
  );
  // Zero eps is OutOfRange.
  let zero = Config::from_json(r#"{"layer_norm_eps": 0.0}"#).unwrap();
  match zero.validate() {
    Err(Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("layer_norm_eps"),
        "context should name layer_norm_eps, got {:?}",
        p.context()
      );
    }
    other => panic!("expected OutOfRange for a zero layer_norm_eps, got {other:?}"),
  }
  // Negative eps is OutOfRange.
  let neg = Config::from_json(r#"{"layer_norm_eps": -1e-5}"#).unwrap();
  assert!(matches!(neg.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn config_validate_rejects_non_finite_layer_norm_eps() {
  // An eps that overflows f32 to a non-finite value must never be accepted: it
  // would otherwise drive a non-finite normalization denominator. Whether the
  // over-range literal is caught at parse time (serde) or at validate time
  // (the helper's NonFiniteScalar branch, when the f64→f32 cast saturates to
  // infinity), the config must be rejected — never produce a usable model.
  match Config::from_json(r#"{"layer_norm_eps": 1e40}"#) {
    Err(Error::Parse(_)) => {}
    Ok(cfg) => {
      // Parsed: the cast saturated to a non-finite f32; validate rejects it.
      assert!(
        !cfg.layer_norm_eps.is_finite(),
        "1e40 should overflow f32 to a non-finite value, got {}",
        cfg.layer_norm_eps
      );
      assert!(matches!(cfg.validate(), Err(Error::NonFiniteScalar(_))));
    }
    other => panic!("expected Parse error or a non-finite eps rejected by validate, got {other:?}"),
  }
}

#[test]
fn config_validate_rejects_unsupported_hidden_act() {
  // An activation the port does not wire (e.g. relu) is rejected with a typed
  // UnknownEnumValue carrying the offending value and the supported set.
  let relu = Config::from_json(r#"{"hidden_act": "relu"}"#).unwrap();
  assert_eq!(relu.hidden_act, "relu");
  match relu.validate() {
    Err(Error::UnknownEnumValue(p)) => {
      assert_eq!(p.value(), "relu");
      assert_eq!(
        p.supported(),
        &["gelu", "gelu_new", "gelu_pytorch_tanh", "silu", "swish"]
      );
    }
    other => panic!("expected UnknownEnumValue for an unsupported hidden_act, got {other:?}"),
  }
}

#[test]
fn config_validate_rejects_unsupported_feat_extract_activation() {
  // Same dispatch for the feature-encoder activation: an unsupported name is a
  // typed UnknownEnumValue carrying the value + supported set.
  let relu = Config::from_json(r#"{"feat_extract_activation": "relu"}"#).unwrap();
  assert_eq!(relu.feat_extract_activation, "relu");
  match relu.validate() {
    Err(Error::UnknownEnumValue(p)) => {
      assert_eq!(p.value(), "relu");
      assert_eq!(
        p.supported(),
        &["gelu", "gelu_new", "gelu_pytorch_tanh", "silu", "swish"]
      );
    }
    other => {
      panic!("expected UnknownEnumValue for an unsupported feat_extract_activation, got {other:?}")
    }
  }
}

#[test]
fn config_validate_accepts_supported_activations() {
  // Every supported HF activation name on both fields validates.
  for act in ["gelu", "gelu_new", "gelu_pytorch_tanh", "silu", "swish"] {
    let hidden = Config::from_json(&format!(r#"{{"hidden_act": "{act}"}}"#)).unwrap();
    assert!(hidden.validate().is_ok(), "hidden_act={act} must validate");
    let feat = Config::from_json(&format!(r#"{{"feat_extract_activation": "{act}"}}"#)).unwrap();
    assert!(
      feat.validate().is_ok(),
      "feat_extract_activation={act} must validate"
    );
  }
}

#[test]
fn activation_resolve_maps_hf_names() {
  // The HF activation-name mapping (ACT2FN): gelu -> exact GELU; gelu_new /
  // gelu_pytorch_tanh -> tanh-approx GELU; silu / swish -> SiLU. An unsupported
  // name is a typed UnknownEnumValue carrying the field + value.
  assert_eq!(
    Activation::resolve("gelu", "test").unwrap(),
    Activation::Gelu
  );
  assert_eq!(
    Activation::resolve("gelu_new", "test").unwrap(),
    Activation::GeluApprox
  );
  assert_eq!(
    Activation::resolve("gelu_pytorch_tanh", "test").unwrap(),
    Activation::GeluApprox
  );
  assert_eq!(
    Activation::resolve("silu", "test").unwrap(),
    Activation::Silu
  );
  assert_eq!(
    Activation::resolve("swish", "test").unwrap(),
    Activation::Silu
  );
  match Activation::resolve("relu", "test field") {
    Err(Error::UnknownEnumValue(p)) => {
      assert_eq!(p.type_name(), "test field");
      assert_eq!(p.value(), "relu");
    }
    other => panic!("expected UnknownEnumValue for an unsupported name, got {other:?}"),
  }
}

#[test]
fn activation_forward_dispatches_to_primitive() {
  // Each Activation::forward must match the corresponding free function in
  // `lm::nn::activations` exactly (the dispatch wires the right kernel). The
  // oracle is the standalone primitive over the same input.
  use crate::lm::nn::activations;
  let probe = Array::from_slice::<f32>(&[-2.0, -0.5, 0.0, 0.7, 2.0], &[5]).unwrap();
  for act in [Activation::Gelu, Activation::GeluApprox, Activation::Silu] {
    let mut got = act.forward(&probe).unwrap();
    let mut want = match act {
      Activation::Gelu => activations::gelu(&probe),
      Activation::GeluApprox => activations::gelu_approx(&probe),
      Activation::Silu => activations::silu(&probe),
    }
    .unwrap();
    let g = got.to_vec::<f32>().unwrap();
    let w = want.to_vec::<f32>().unwrap();
    for (a, b) in g.iter().zip(w.iter()) {
      assert!((a - b).abs() < 1e-6, "{act:?}: {a} vs {b}");
    }
  }
}

#[test]
fn config_validate_rejects_add_adapter() {
  // `add_adapter == true` stacks a post-encoder conv adapter (re-shaping the
  // CTC head's input dim) that this port does not wire, so the head would read
  // the wrong dimension. `validate` must reject it with a typed
  // InvariantViolation naming the field, BEFORE any tensor is built.
  let cfg = Config::from_json(r#"{"add_adapter": true}"#).unwrap();
  assert!(cfg.add_adapter);
  match cfg.validate() {
    Err(Error::InvariantViolation(p)) => {
      assert!(
        p.context().contains("add_adapter"),
        "context should name add_adapter, got {:?}",
        p.context()
      );
    }
    other => panic!("expected InvariantViolation for add_adapter = true, got {other:?}"),
  }
}

#[test]
fn config_validate_accepts_positive_adapter_attn_dim_rejects_non_positive() {
  // A POSITIVE `adapter_attn_dim` is now wired (the MMS per-attention-block
  // language adapter): a config with `adapter_attn_dim` set (and the stable-LN
  // arm MMS uses) must validate.
  let cfg = Config::from_json(r#"{"adapter_attn_dim": 16, "do_stable_layer_norm": true}"#).unwrap();
  assert_eq!(cfg.adapter_attn_dim, Some(16));
  assert!(
    cfg.validate().is_ok(),
    "a positive adapter_attn_dim must validate (the MMS adapter is now wired)"
  );
  // It also validates with the (default) post-norm arm — the reference attaches
  // the adapter only to the stable-LN layer, so a post-norm checkpoint carrying
  // adapter_attn_dim simply runs no adapter; the value is still accepted.
  let post = Config::from_json(r#"{"adapter_attn_dim": 16}"#).unwrap();
  assert!(post.validate().is_ok());

  // A NON-POSITIVE `adapter_attn_dim` (it sizes the adapter Linears) is
  // malformed -> typed OutOfRange naming the field.
  for bad in ["0", "-4"] {
    let cfg = Config::from_json(&format!(r#"{{"adapter_attn_dim": {bad}}}"#)).unwrap();
    match cfg.validate() {
      Err(Error::OutOfRange(p)) => {
        assert!(
          p.context().contains("adapter_attn_dim"),
          "context should name adapter_attn_dim, got {:?}",
          p.context()
        );
      }
      other => panic!("expected OutOfRange for adapter_attn_dim = {bad}, got {other:?}"),
    }
  }
  // An explicit `null` is equivalent to absent and validates (the base-960h
  // form): no adapter at all.
  let explicit_null = Config::from_json(r#"{"adapter_attn_dim": null}"#).unwrap();
  assert_eq!(explicit_null.adapter_attn_dim, None);
  assert!(explicit_null.validate().is_ok());
}

#[test]
fn config_validate_rejects_deviating_pad_token_id() {
  // The greedy CTC decoder hardcodes blank id 0 (`CTC_BLANK`); a checkpoint
  // declaring a different `pad_token_id` would collapse the argmax against the
  // wrong token. `validate` must reject it with a typed OutOfRange naming the
  // field + the offending and expected (0) values.
  let cfg = Config::from_json(r#"{"pad_token_id": 1}"#).unwrap();
  assert_eq!(cfg.pad_token_id, 1);
  match cfg.validate() {
    Err(Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("pad_token_id"),
        "context should name pad_token_id, got {:?}",
        p.context()
      );
      assert!(
        p.value().contains('1') && p.value().contains('0'),
        "value should name the offending id and the expected one, got {:?}",
        p.value()
      );
    }
    other => panic!("expected OutOfRange for a deviating pad_token_id, got {other:?}"),
  }
}

#[test]
fn config_validate_feat_proj_layer_norm_false_is_hubert_only() {
  // `feat_proj_layer_norm` is a HuBERT-ONLY flag (HF default `true`): HF's
  // `Wav2Vec2FeatureProjection` has no such field and ALWAYS applies the
  // projection LayerNorm, while only `HubertFeatureProjection` gates it. So the
  // no-LayerNorm arm (`false`) is valid ONLY for `model_type == "hubert"`.

  // (a) wav2vec2 (the default model_type) + `false` is REJECTED — honoring it
  // would build a wav2vec2 graph with its projection LayerNorm silently dropped.
  let w2v2_no_ln = Config::from_json(r#"{"feat_proj_layer_norm": false}"#).unwrap();
  assert!(!w2v2_no_ln.feat_proj_layer_norm);
  assert!(!w2v2_no_ln.is_hubert());
  match w2v2_no_ln.validate() {
    Err(Error::InvariantViolation(_)) => {}
    Ok(()) => panic!(
      "feat_proj_layer_norm = false on a wav2vec2 model_type must be REJECTED (the no-LayerNorm \
       projection is a HuBERT-only arm)"
    ),
    Err(e) => panic!(
      "expected a typed InvariantViolation for wav2vec2 + feat_proj_layer_norm=false, got {e:?}"
    ),
  }
  // An explicit `"model_type": "wav2vec2"` + `false` is likewise rejected.
  let w2v2_explicit =
    Config::from_json(r#"{"model_type": "wav2vec2", "feat_proj_layer_norm": false}"#).unwrap();
  assert!(w2v2_explicit.validate().is_err());

  // (b) hubert + `false` is HONORED (HuBERT's no-LayerNorm arm) — validates.
  let hubert_no_ln =
    Config::from_json(r#"{"model_type": "hubert", "feat_proj_layer_norm": false}"#).unwrap();
  assert!(!hubert_no_ln.feat_proj_layer_norm);
  assert!(hubert_no_ln.is_hubert());
  assert!(
    hubert_no_ln.validate().is_ok(),
    "feat_proj_layer_norm = false on a hubert model_type is the no-LayerNorm arm and must validate"
  );

  // (c) `true` (the LayerNorm arm) validates for BOTH model_types — the HF
  // default for HuBERT and the implicit value for every wav2vec2 config.
  let w2v2_true = Config::from_json(r#"{"feat_proj_layer_norm": true}"#).unwrap();
  assert!(w2v2_true.feat_proj_layer_norm);
  assert!(w2v2_true.validate().is_ok());
  let hubert_true =
    Config::from_json(r#"{"model_type": "hubert", "feat_proj_layer_norm": true}"#).unwrap();
  assert!(hubert_true.validate().is_ok());

  // (d) Absent (the common case: wav2vec2 configs never carry the field, HuBERT
  // defaults it true) falls back to the LayerNorm arm and validates for both.
  let absent = Config::from_json("{}").unwrap();
  assert!(
    absent.feat_proj_layer_norm,
    "feat_proj_layer_norm must default to true when absent"
  );
  assert!(absent.validate().is_ok());
  let hubert_absent = Config::from_json(r#"{"model_type": "hubert"}"#).unwrap();
  assert!(hubert_absent.feat_proj_layer_norm);
  assert!(hubert_absent.validate().is_ok());
}

#[test]
fn config_validate_rejects_non_default_conv_pos_batch_norm() {
  // `conv_pos_batch_norm` is a HuBERT-only flag (HF default `false`); the wired
  // PositionalConvEmbedding reconstructs the fused kernel from the weight-norm
  // `weight_g` / `weight_v` pair, so the `true` (batch-norm) arm selects a
  // different module (a BatchNorm1d over a plain conv whose checkpoint carries
  // no weight_g/weight_v) that is not implemented this phase. A `true` value
  // must be rejected with a typed InvariantViolation naming the field, BEFORE
  // any tensor is built.
  let cfg = Config::from_json(r#"{"conv_pos_batch_norm": true}"#).unwrap();
  assert!(cfg.conv_pos_batch_norm);
  match cfg.validate() {
    Err(Error::InvariantViolation(p)) => {
      assert!(
        p.context().contains("conv_pos_batch_norm"),
        "context should name conv_pos_batch_norm, got {:?}",
        p.context()
      );
    }
    other => panic!("expected InvariantViolation for conv_pos_batch_norm = true, got {other:?}"),
  }
  // The default (`false`, the weight-norm arm) validates — the HF default for
  // HuBERT and the implicit value for every wav2vec2 config.
  let default_false = Config::from_json(r#"{"conv_pos_batch_norm": false}"#).unwrap();
  assert!(!default_false.conv_pos_batch_norm);
  assert!(
    default_false.validate().is_ok(),
    "the default conv_pos_batch_norm = false (the weight-norm arm) must validate"
  );
  // Absent falls back to the wired (weight-norm) arm and validates.
  let absent = Config::from_json("{}").unwrap();
  assert!(
    !absent.conv_pos_batch_norm,
    "conv_pos_batch_norm must default to false when absent"
  );
  assert!(absent.validate().is_ok());
}

#[test]
fn default_hubert_flags_build_and_forward() {
  // A HuBERT config that spells out both HuBERT-only flags at their HF defaults
  // (feat_proj_layer_norm = true, conv_pos_batch_norm = false) selects the wired
  // graph (LayerNorm feature projection + weight-norm positional conv), so it
  // must build and forward to the right logits shape — the both-directions
  // proof that the pins reject ONLY the non-default arm, while a default HuBERT
  // checkpoint is faithfully served end to end.
  let config = tiny_config_json(
    r#", "model_type": "hubert", "feat_proj_layer_norm": true, "conv_pos_batch_norm": false"#,
  );
  assert_eq!(config.model_type(), "hubert");
  assert!(config.feat_proj_layer_norm);
  assert!(!config.conv_pos_batch_norm);
  assert_forward_shape(config, 400);
}

/// Parametric guard: family variants and dimension overrides that are merely
/// *different* from base-960h now VALIDATE (the port is generic), while the
/// genuinely out-of-scope arms and structurally-invalid configs are rejected.
#[test]
fn config_validate_accepts_family_variants_rejects_out_of_scope() {
  // (a) Different-but-valid family configs — each must validate.
  let accepted: &[&str] = &[
    // model_type family members (plain self-attention only; wavlm is NOT here —
    // its gated relative-position-bias attention is not wired this phase).
    r#"{"model_type": "hubert"}"#,
    // A larger vocab (different CTC head width).
    r#"{"vocab_size": 33}"#,
    // A different positive eps.
    r#"{"layer_norm_eps": 1e-6}"#,
    // The stable-layer-norm arm and conv_bias.
    r#"{"do_stable_layer_norm": true}"#,
    r#"{"conv_bias": true}"#,
    // The supported non-gelu activations.
    r#"{"hidden_act": "gelu_new"}"#,
    r#"{"feat_extract_activation": "silu"}"#,
    // The HuBERT-only flags spelled out at their HF defaults (the wired arm).
    r#"{"model_type": "hubert", "feat_proj_layer_norm": true}"#,
    r#"{"model_type": "hubert", "conv_pos_batch_norm": false}"#,
    // The newly-unlocked variant arms: the "layer" feature-encoder norm
    // (large-960h-lv60-self), the HuBERT no-LayerNorm feature projection (only
    // valid on `model_type == "hubert"`), and the MMS per-attention-block adapter
    // (positive bottleneck width).
    r#"{"feat_extract_norm": "layer"}"#,
    r#"{"model_type": "hubert", "feat_proj_layer_norm": false}"#,
    r#"{"adapter_attn_dim": 16}"#,
    r#"{"adapter_attn_dim": 16, "do_stable_layer_norm": true}"#,
    // A consistent larger transformer (hidden divisible by heads + groups).
    r#"{"hidden_size": 1024, "num_attention_heads": 16,
        "num_conv_pos_embedding_groups": 16}"#,
    r#"{"intermediate_size": 4096}"#,
    r#"{"num_hidden_layers": 24}"#,
    // A different-but-positive conv stack of the right length.
    r#"{"conv_dim": [512, 512, 512, 512, 512, 512, 256]}"#,
    r#"{"conv_stride": [5, 3, 2, 2, 2, 2, 2]}"#,
    r#"{"conv_kernel": [10, 3, 3, 3, 3, 2, 3]}"#,
  ];
  for json in accepted {
    let cfg = Config::from_json(json).unwrap();
    assert!(
      cfg.validate().is_ok(),
      "a valid family variant must pass validate(), but this one was rejected: {json}"
    );
  }

  // (b) Out-of-scope arms / structurally-invalid configs — each must error.
  let rejected: &[&str] = &[
    // Out-of-scope (not wired this phase).
    r#"{"model_type": "unknown_arch"}"#,
    // WavLM needs gated relative-position-bias attention (not wired this
    // phase); admitting it would run its rel-pos tensors through the plain
    // attention path unconsumed (silent corruption), so it is rejected.
    r#"{"model_type": "wavlm"}"#,
    // An UNKNOWN feat_extract_norm (neither "group" nor "layer") is rejected.
    r#"{"feat_extract_norm": "instance"}"#,
    r#"{"hidden_act": "relu"}"#,
    r#"{"feat_extract_activation": "relu"}"#,
    r#"{"add_adapter": true}"#,
    // A non-positive adapter bottleneck width is malformed (it sizes the
    // adapter Linears) even though the adapter itself is now wired.
    r#"{"adapter_attn_dim": 0}"#,
    r#"{"adapter_attn_dim": -8}"#,
    r#"{"pad_token_id": 1}"#,
    // The HuBERT-only batch-norm positional conv arm is not wired this phase.
    r#"{"conv_pos_batch_norm": true}"#,
    // The no-LayerNorm feature projection is HuBERT-only: a wav2vec2 model_type
    // (default + explicit) with feat_proj_layer_norm=false is rejected (HF's
    // Wav2Vec2FeatureProjection always applies the projection LayerNorm).
    r#"{"feat_proj_layer_norm": false}"#,
    r#"{"model_type": "wav2vec2", "feat_proj_layer_norm": false}"#,
    // Structurally invalid.
    r#"{"hidden_size": 0}"#,
    r#"{"hidden_size": 1000, "num_attention_heads": 12}"#,
    r#"{"num_attention_heads": 0}"#,
    r#"{"intermediate_size": -1}"#,
    r#"{"vocab_size": 0}"#,
    r#"{"num_hidden_layers": 0}"#,
    r#"{"num_feat_extract_layers": 0}"#,
    r#"{"layer_norm_eps": 0.0}"#,
    r#"{"num_conv_pos_embedding_groups": 7}"#,
    r#"{"conv_kernel": [10, 3, 3, 3, 3, 2]}"#,
    r#"{"conv_stride": [5, 0, 2, 2, 2, 2, 2]}"#,
  ];
  for json in rejected {
    let cfg = Config::from_json(json).unwrap();
    assert!(
      cfg.validate().is_err(),
      "an out-of-scope / invalid config must be rejected by validate(), but this one passed: {json}"
    );
  }

  // The all-default baseline must itself validate, proving the rejections are
  // caused by the override, not a baseline that already fails.
  assert!(
    Config::from_json("{}").unwrap().validate().is_ok(),
    "the all-default config must pass validate()"
  );
}

#[test]
fn defaults_match_base_960h_and_validate() {
  // The serde `default_*` fns describe `facebook/wav2vec2-base-960h`, and the
  // default config must pass validate() (the all-default checkpoint loads).
  let defaults = Config::from_json("{}").unwrap();
  assert!(
    defaults.validate().is_ok(),
    "the all-default (base-960h) config must pass validate()"
  );
  // Spot-check the base-960h default values.
  assert_eq!(defaults.hidden_size, 768);
  assert_eq!(defaults.num_hidden_layers, 12);
  assert_eq!(defaults.num_attention_heads, 12);
  assert_eq!(defaults.intermediate_size, 3072);
  assert_eq!(defaults.vocab_size, 32);
  assert_eq!(defaults.num_conv_pos_embeddings, 128);
  assert_eq!(defaults.num_conv_pos_embedding_groups, 16);
  assert_eq!(defaults.num_feat_extract_layers, 7);
  assert_eq!(defaults.conv_dim, vec![512, 512, 512, 512, 512, 512, 512]);
  assert_eq!(defaults.conv_stride, vec![5, 2, 2, 2, 2, 2, 2]);
  assert_eq!(defaults.conv_kernel, vec![10, 3, 3, 3, 3, 2, 2]);
  assert_eq!(defaults.model_type(), "wav2vec2");
  assert_eq!(defaults.hidden_act, "gelu");
  assert_eq!(defaults.feat_extract_activation, "gelu");
  assert!((defaults.layer_norm_eps - 1e-5).abs() < 1e-12);
  assert!(!defaults.do_stable_layer_norm);
  assert!(!defaults.conv_bias);
  assert!(!defaults.add_adapter);
  assert_eq!(defaults.adapter_attn_dim, None);
  assert_eq!(defaults.pad_token_id, 0);
  // HuBERT-only flag defaults match the wired graph.
  assert!(defaults.feat_proj_layer_norm);
  assert!(!defaults.conv_pos_batch_norm);
}

// ───────────────────────── test 2: feature-encoder time chain ─────────────────────────

/// The conv output length recurrence (no padding, dilation 1):
/// `L_out = (L_in - kernel) / stride + 1`. Applied with the base-960h
/// strides/kernels to a 16000-sample input, the chain lands on ~49 frames.
fn conv_out_len(l_in: i64, kernel: i64, stride: i64) -> i64 {
  (l_in - kernel) / stride + 1
}

#[test]
fn feature_encoder_time_chain_analytic() {
  // Hand-roll the analytic chain that build_feature_encoder's conv1d stack
  // produces for a 1-second 16 kHz clip.
  let kernels = [10i64, 3, 3, 3, 3, 2, 2];
  let strides = [5i64, 2, 2, 2, 2, 2, 2];
  let mut l = 16_000i64;
  for (k, s) in kernels.iter().zip(strides.iter()) {
    l = conv_out_len(l, *k, *s);
  }
  // The canonical wav2vec2 output for 1 s @ 16 kHz is 49 frames.
  assert_eq!(l, 49);
}

#[test]
fn feature_encoder_conv_stack_matches_analytic_shape() {
  // Build a synthetic 7-layer channels-last conv stack with the base-960h
  // strides/kernels (channels collapsed to 1 for a cheap shape probe) and
  // confirm the time dimension matches the analytic recurrence. This exercises
  // the same conv1d + stride path build_feature_encoder wires, on a short
  // input so the test stays fast.
  let kernels = [10i32, 3, 3, 3, 3, 2, 2];
  let strides = [5i32, 2, 2, 2, 2, 2, 2];
  let l_in: i32 = 1024;
  // (B=1, L=1024, C=1) channels-last input.
  let mut x = Array::zeros::<f32>(&[1, l_in, 1]).unwrap();
  let mut expected = l_in as i64;
  for (k, s) in kernels.iter().zip(strides.iter()) {
    // (C_out=1, K, C_in=1) all-ones kernel — shape probe only.
    let w = Array::from_slice::<f32>(&vec![1.0f32; *k as usize], &[1, *k, 1]).unwrap();
    x = ops::conv::conv1d(&x, &w, *s, 0, 1, 1).unwrap();
    expected = conv_out_len(expected, *k as i64, *s as i64);
  }
  let shape = x.shape();
  assert_eq!(shape[0], 1); // batch
  assert_eq!(shape[2], 1); // channels
  assert_eq!(shape[1] as i64, expected);
}

// ───────────────────────── test 3: GroupNorm per-channel (num_groups==dims) ─────────────────────────

#[test]
fn group_norm_per_channel_zero_mean_unit_var() {
  // With num_groups == dims, each channel is its own group: GroupNorm
  // normalizes every channel independently to zero-mean / unit-variance over
  // the spatial (time) axis. This mirrors the L0 Wav2Vec2GroupNormConvLayer
  // (num_groups == dims == 512); here we use dims = 2 for a hand-checkable case.
  //
  // Channels-last input (B=1, L=3, C=2):
  //   channel 0 over time: [1, 2, 3]  -> mean 2, var 2/3
  //   channel 1 over time: [10, 20, 30] -> mean 20, var 200/3
  let x = Array::from_slice::<f32>(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], &[1, 3, 2]).unwrap();
  let gn = GroupNorm::new(2, 2, 1e-5, true, true).unwrap();
  let mut out = gn.forward(&x).unwrap();
  let v = out.to_vec::<f32>().unwrap();
  // Reconstruct per-channel by striding the (L=3, C=2) row-major buffer.
  let ch0: Vec<f32> = vec![v[0], v[2], v[4]];
  let ch1: Vec<f32> = vec![v[1], v[3], v[5]];
  for ch in [&ch0, &ch1] {
    let mean: f32 = ch.iter().sum::<f32>() / 3.0;
    let var: f32 = ch.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 3.0;
    assert!(mean.abs() < 1e-4, "channel mean ~0, got {mean}");
    assert!((var - 1.0).abs() < 1e-3, "channel var ~1, got {var}");
  }
  // The two channels are normalized identically up to their own scale, so the
  // normalized values must match between channels (both are the same affine
  // image of [-, 0, +]).
  for (a, b) in ch0.iter().zip(ch1.iter()) {
    assert!(
      (a - b).abs() < 1e-3,
      "channels normalize identically: {a} vs {b}"
    );
  }
}

// ───────────────────────── test 4: WNConv1d weight reconstruction ─────────────────────────

#[test]
fn wn_weight_reconstruction_reduces_over_kernel_complement() {
  // weight = weight_g * weight_v / ‖weight_v‖, with the norm of weight_v taken
  // over every axis EXCEPT the kernel axis (axis 1 in MLX (out, k, in) layout),
  // keepdims so it broadcasts.
  //
  // Take (out=2, k=2, in=1) weight_v:
  //   v[o, kk, 0]:  o0 -> [3, 4]   (kernel positions)
  //                 o1 -> [0, 5]
  // Norm over axes (0, 2) keepdims -> shape (1, 2, 1), per KERNEL position k:
  //   k=0: sqrt(v[0,0,0]^2 + v[1,0,0]^2) = sqrt(9 + 0) = 3
  //   k=1: sqrt(v[0,1,0]^2 + v[1,1,0]^2) = sqrt(16 + 25) = sqrt(41)
  // weight_g broadcast as all-2s.
  // Expected fused weight[o,k,0] = 2 * v[o,k,0] / norm[k]:
  //   [0,0,0] = 2*3/3        = 2
  //   [0,1,0] = 2*4/sqrt(41)
  //   [1,0,0] = 2*0/3        = 0
  //   [1,1,0] = 2*5/sqrt(41)
  let weight_v = Array::from_slice::<f32>(&[3.0, 4.0, 0.0, 5.0], &[2, 2, 1]).unwrap();
  let weight_g = Array::from_slice::<f32>(&[2.0, 2.0, 2.0, 2.0], &[2, 2, 1]).unwrap();
  let mut fused = reconstruct_wn_weight(&weight_g, &weight_v).unwrap();
  assert_eq!(fused.shape(), vec![2, 2, 1]);
  let got = fused.to_vec::<f32>().unwrap();
  let sqrt41 = 41.0f32.sqrt();
  let want = [2.0, 2.0 * 4.0 / sqrt41, 0.0, 2.0 * 5.0 / sqrt41];
  for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
    assert!((g - w).abs() < 1e-5, "fused[{i}]: got {g}, want {w}");
  }
}

#[test]
fn wn_weight_reconstruction_rejects_non_rank3() {
  let weight_v = Array::from_slice::<f32>(&[1.0, 2.0], &[2]).unwrap();
  let weight_g = Array::from_slice::<f32>(&[1.0, 2.0], &[2]).unwrap();
  assert!(matches!(
    reconstruct_wn_weight(&weight_g, &weight_v),
    Err(Error::RankMismatch(_))
  ));
}

// ───────────────────────── waveform normalization ─────────────────────────

#[test]
fn normalize_waveform_zero_mean_unit_var() {
  // x = [1, 2, 3, 4] over the last axis. mean = 2.5, var (population) = 1.25.
  // normalized = (x - 2.5) / sqrt(1.25 + 1e-7) ~ (x - 2.5)/1.1180340.
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
  let mut out = normalize_waveform(&x).unwrap();
  let v = out.to_vec::<f32>().unwrap();
  let denom = (1.25f32 + 1e-7).sqrt();
  let want = [
    (1.0 - 2.5) / denom,
    (2.0 - 2.5) / denom,
    (3.0 - 2.5) / denom,
    (4.0 - 2.5) / denom,
  ];
  for (i, (g, w)) in v.iter().zip(want.iter()).enumerate() {
    assert!((g - w).abs() < 1e-5, "normalized[{i}]: got {g}, want {w}");
  }
  // Result is zero-mean / unit-variance by construction.
  let mean: f32 = v.iter().sum::<f32>() / 4.0;
  assert!(mean.abs() < 1e-5, "normalized mean ~0, got {mean}");
}

#[test]
fn normalize_waveform_promotes_1d_to_2d() {
  // A 1-D (T,) input is promoted to (1, T) before normalization.
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
  let out = normalize_waveform(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 4]);
}

// ───────────────────────── linear wrapper ─────────────────────────

#[test]
fn dense_linear_with_and_without_bias() {
  // The dense `Linear` wrapper (the adoption of the shared `MaybeQuantizedLinear`)
  // computes `y = x @ wᵀ (+ bias)`. x (1, 2) = [1, 2]; weight (out=2, in=2) =
  // [[1,0],[0,1]] (identity); y = x @ wᵀ = [1, 2]. With bias [10, 20] -> [11, 22].
  let x = Array::from_slice::<f32>(&[1.0, 2.0], &[1, 2]).unwrap();
  let w = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
  let no_bias = Linear::new(w.try_clone().unwrap(), None);
  let mut y = no_bias.forward(&x).unwrap();
  assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
  assert!(!no_bias.is_quantized(), "a bare-weight Linear is dense");
  let bias = Array::from_slice::<f32>(&[10.0, 20.0], &[2]).unwrap();
  let with_bias = Linear::new(w, Some(bias));
  let mut yb = with_bias.forward(&x).unwrap();
  assert_eq!(yb.to_vec::<f32>().unwrap(), vec![11.0, 22.0]);
}

// ───────────────────────── weight-shape validation ─────────────────────────

/// A lazy zero tensor of the given shape — cheap (no materialization) because
/// `from_weights` only reads `shape()` and composes lazy ops; nothing is
/// evaluated during construction.
fn zeros(shape: &[i32]) -> Array {
  Array::zeros::<f32>(&shape).unwrap()
}

/// Build a complete `base-960h` **post-sanitize** weight map (the exact layout
/// `from_weights` consumes), every tensor at its correct base-960h shape. Used
/// as the baseline the drift tests mutate one tensor of. All tensors are lazy
/// zeros, so the whole map costs only metadata.
///
/// Shapes are written out longhand (not read from the code under test) so a
/// regression in the production expected-shape derivation cannot also silently
/// shift this oracle.
fn base_960h_weights() -> HashMap<String, Array> {
  let mut w: HashMap<String, Array> = HashMap::new();
  // Feature encoder: L0 is (512, 10, 1) + a (512,) GroupNorm affine; L1-6 are
  // (512, k, 512) with kernels [3,3,3,3,2,2].
  let kernels = [10i32, 3, 3, 3, 3, 2, 2];
  for (i, &k) in kernels.iter().enumerate() {
    let in_ch = if i == 0 { 1 } else { 512 };
    w.insert(
      format!("feature_extractor.conv_layers.{i}.conv.weight"),
      zeros(&[512, k, in_ch]),
    );
  }
  w.insert(
    "feature_extractor.conv_layers.0.layer_norm.weight".to_string(),
    zeros(&[512]),
  );
  w.insert(
    "feature_extractor.conv_layers.0.layer_norm.bias".to_string(),
    zeros(&[512]),
  );
  // Feature projection: LayerNorm(512), Linear(512 -> 768).
  w.insert(
    "feature_projection.layer_norm.weight".to_string(),
    zeros(&[512]),
  );
  w.insert(
    "feature_projection.layer_norm.bias".to_string(),
    zeros(&[512]),
  );
  w.insert(
    "feature_projection.projection.weight".to_string(),
    zeros(&[768, 512]),
  );
  w.insert(
    "feature_projection.projection.bias".to_string(),
    zeros(&[768]),
  );
  // Positional conv: weight_g (1, 128, 1), weight_v (768, 128, 48), bias (768,).
  w.insert(
    "encoder.pos_conv_embed.conv.weight_g".to_string(),
    zeros(&[1, 128, 1]),
  );
  w.insert(
    "encoder.pos_conv_embed.conv.weight_v".to_string(),
    zeros(&[768, 128, 48]),
  );
  w.insert(
    "encoder.pos_conv_embed.conv.bias".to_string(),
    zeros(&[768]),
  );
  w.insert("encoder.layer_norm.weight".to_string(), zeros(&[768]));
  w.insert("encoder.layer_norm.bias".to_string(), zeros(&[768]));
  // 12 encoder layers.
  for i in 0..12 {
    let p = format!("encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      w.insert(format!("{p}.attention.{proj}.weight"), zeros(&[768, 768]));
      w.insert(format!("{p}.attention.{proj}.bias"), zeros(&[768]));
    }
    w.insert(format!("{p}.layer_norm.weight"), zeros(&[768]));
    w.insert(format!("{p}.layer_norm.bias"), zeros(&[768]));
    w.insert(
      format!("{p}.feed_forward.intermediate_dense.weight"),
      zeros(&[3072, 768]),
    );
    w.insert(
      format!("{p}.feed_forward.intermediate_dense.bias"),
      zeros(&[3072]),
    );
    w.insert(
      format!("{p}.feed_forward.output_dense.weight"),
      zeros(&[768, 3072]),
    );
    w.insert(format!("{p}.feed_forward.output_dense.bias"), zeros(&[768]));
    w.insert(format!("{p}.final_layer_norm.weight"), zeros(&[768]));
    w.insert(format!("{p}.final_layer_norm.bias"), zeros(&[768]));
  }
  // CTC head: Linear(768 -> 32).
  w.insert("lm_head.weight".to_string(), zeros(&[32, 768]));
  w.insert("lm_head.bias".to_string(), zeros(&[32]));
  w
}

fn base_960h_config() -> Config {
  Config::from_json("{}").unwrap()
}

/// Assert a `from_weights` result is the shape gate's typed rejection — an
/// [`Error::LayerKeyed`] naming `key` whose inner error is an
/// [`Error::ShapePairMismatch`] — and return that inner payload for further
/// shape assertions. `Model` is not `Debug` (it holds `Array`s), so the
/// `Ok` arm is reported without formatting the model.
fn expect_shape_rejection(
  result: Result<Model<Standard>>,
  key: &str,
) -> crate::error::ShapePairMismatchPayload {
  match result {
    Err(Error::LayerKeyed(p)) => {
      assert_eq!(p.layer(), key, "rejection should name the offending tensor");
      match p.inner() {
        Error::ShapePairMismatch(sp) => sp.clone(),
        other => panic!("expected inner ShapePairMismatch for `{key}`, got {other:?}"),
      }
    }
    Err(other) => panic!("expected LayerKeyed(ShapePairMismatch) for `{key}`, got {other:?}"),
    Ok(_) => panic!("expected a shape rejection for `{key}`, but from_weights built a model"),
  }
}

#[test]
fn from_weights_accepts_correctly_shaped_base_960h() {
  // The complete, correctly-shaped base-960h weight set must build a model
  // (the both-directions baseline for the drift rejections below). Construction
  // is lazy, so this never materializes the ~hundreds-of-MB graph.
  let model = Model::from_weights(base_960h_config(), base_960h_weights(), Vocab::default());
  assert!(
    model.is_ok(),
    "a fully base-960h-shaped weight set must build a model"
  );
}

#[test]
fn forward_rejects_over_cap_waveform_length() {
  // The inherent `forward` path has no STT-pipeline `max_audio_seconds` cap, so
  // an over-long waveform would otherwise drive the O(T) conv feature maps and
  // — after the ~320x downsampling — the transformer's O(T'^2) self-attention
  // without bound (a process-level OOM / DoS). The length guard must reject a
  // waveform whose last axis exceeds `MAX_INPUT_SAMPLES` up front, BEFORE the
  // conv stack, with a recoverable typed `OutOfRange` (so the ~3.8 MB zeros
  // input below never reaches any forward allocation).
  let model =
    Model::from_weights(base_960h_config(), base_960h_weights(), Vocab::default()).unwrap();

  // One sample over the cap: rejected at the guard (a shape read), not run.
  let over_cap = zeros(&[(Model::<Standard>::MAX_INPUT_SAMPLES + 1) as i32]);
  let err = model.forward(&over_cap);
  assert!(
    matches!(err, Err(Error::OutOfRange(_))),
    "an over-cap waveform must be rejected with OutOfRange"
  );

  // A normal 1 s waveform (16 000 samples) is well under the cap, so the guard
  // must NOT trip: `forward` builds its lazy graph and returns without the cap
  // error (the result is left lazy — no eval — so this stays cheap).
  let one_second = zeros(&[Model::<Standard>::SAMPLE_RATE as i32]);
  assert!(
    !matches!(model.forward(&one_second), Err(Error::OutOfRange(_))),
    "a 1 s waveform must not be rejected by the length cap"
  );
}

#[test]
fn forward_rejects_over_cap_batched_waveform() {
  // Last axis == cap, but a batch dimension pushes the TOTAL over the cap: the
  // old per-axis check missed this; the total-element cap catches it.
  let model =
    Model::from_weights(base_960h_config(), base_960h_weights(), Vocab::default()).unwrap();
  let over = zeros(&[2, Model::<Standard>::MAX_INPUT_SAMPLES as i32]);
  assert!(matches!(model.forward(&over), Err(Error::OutOfRange(_))));
}

#[test]
fn from_weights_rejects_wrong_lm_head_output_dim() {
  // A hostile `lm_head.weight` with a huge output dim passes the config gate
  // (vocab_size is a config field, pinned to 32 — but the tensor itself is read
  // from the checkpoint) yet would drive a huge logits allocation at forward
  // time. It must be rejected by the shape gate, before any forward, with a
  // typed ShapePairMismatch naming the offending tensor via LayerKeyed.
  let mut weights = base_960h_weights();
  // 1,000,000 output rows instead of vocab_size (32); inner dim still correct.
  weights.insert("lm_head.weight".to_string(), zeros(&[1_000_000, 768]));
  let result = Model::from_weights(base_960h_config(), weights, Vocab::default());
  let sp = expect_shape_rejection(result, "lm_head.weight");
  // The expected shape is the base-960h (vocab_size, hidden_size); the observed
  // is the hostile oversized one. Both computed here, not by the code under test.
  assert_eq!(sp.expected(), &[32usize, 768]);
  assert_eq!(sp.actual(), &[1_000_000usize, 768]);
}

#[test]
fn from_weights_rejects_wrong_conv_kernel_size() {
  // A feature-encoder conv weight whose kernel axis differs from the base-960h
  // value silently changes the receptive field. L0's kernel is 10; a tensor
  // with kernel 7 must be rejected by the shape gate (the feature encoder is
  // built first, so this fails before any other tensor / any forward).
  let mut weights = base_960h_weights();
  // (512, 7, 1) instead of the base-960h (512, 10, 1).
  weights.insert(
    "feature_extractor.conv_layers.0.conv.weight".to_string(),
    zeros(&[512, 7, 1]),
  );
  let result = Model::from_weights(base_960h_config(), weights, Vocab::default());
  let sp = expect_shape_rejection(result, "feature_extractor.conv_layers.0.conv.weight");
  assert_eq!(sp.expected(), &[512usize, 10, 1]);
  assert_eq!(sp.actual(), &[512usize, 7, 1]);
}

#[test]
fn from_weights_rejects_wrong_rank_tensor() {
  // A consumed tensor with the wrong RANK (here a 2-D attention weight given as
  // 3-D) is rejected by the same gate — the length comparison pins the rank, so
  // a rank drift never slips through as a "close enough" shape.
  let mut weights = base_960h_weights();
  weights.insert(
    "encoder.layers.0.attention.q_proj.weight".to_string(),
    zeros(&[768, 768, 1]),
  );
  let result = Model::from_weights(base_960h_config(), weights, Vocab::default());
  let sp = expect_shape_rejection(result, "encoder.layers.0.attention.q_proj.weight");
  // Rank pinned: expected the rank-2 (hidden, hidden), observed the rank-3 drift.
  assert_eq!(sp.expected(), &[768usize, 768]);
  assert_eq!(sp.actual(), &[768usize, 768, 1]);
}

#[test]
fn from_weights_rejects_wrong_pos_conv_weight_v() {
  // The positional conv weight_v controls the positional receptive field; a
  // deviating kernel/channel axis must be rejected (here kernel 64 instead of
  // the base-960h 128).
  let mut weights = base_960h_weights();
  weights.insert(
    "encoder.pos_conv_embed.conv.weight_v".to_string(),
    zeros(&[768, 64, 48]),
  );
  let result = Model::from_weights(base_960h_config(), weights, Vocab::default());
  let sp = expect_shape_rejection(result, "encoder.pos_conv_embed.conv.weight_v");
  assert_eq!(sp.expected(), &[768usize, 128, 48]);
  assert_eq!(sp.actual(), &[768usize, 64, 48]);
}

#[test]
fn from_weights_rejects_wrong_feed_forward_dim() {
  // The feed-forward intermediate weight is (intermediate_size, hidden_size) =
  // (3072, 768); a deviating intermediate dim runs a different MLP and must be
  // rejected.
  let mut weights = base_960h_weights();
  weights.insert(
    "encoder.layers.3.feed_forward.intermediate_dense.weight".to_string(),
    zeros(&[4096, 768]),
  );
  let result = Model::from_weights(base_960h_config(), weights, Vocab::default());
  let sp = expect_shape_rejection(
    result,
    "encoder.layers.3.feed_forward.intermediate_dense.weight",
  );
  assert_eq!(sp.expected(), &[3072usize, 768]);
  assert_eq!(sp.actual(), &[4096usize, 768]);
}

#[test]
fn expect_shape_matches_and_mismatches() {
  // Direct unit coverage of the shape-check helper: a correct shape passes; a
  // wrong dim, a wrong rank, and the OOM-relevant oversized output dim each
  // produce a LayerKeyed(ShapePairMismatch) carrying both full shapes.
  let t = zeros(&[32, 768]);
  assert!(expect_shape(&t, "lm_head.weight", "ctc head", &[32, 768]).is_ok());
  // Wrong inner dim.
  match expect_shape(&t, "lm_head.weight", "ctc head", &[32, 769]) {
    Err(Error::LayerKeyed(p)) => {
      assert_eq!(p.layer(), "lm_head.weight");
      match p.inner() {
        Error::ShapePairMismatch(sp) => {
          assert_eq!(sp.expected(), &[32usize, 769]);
          assert_eq!(sp.actual(), &[32usize, 768]);
        }
        other => panic!("expected ShapePairMismatch, got {other:?}"),
      }
    }
    other => panic!("expected LayerKeyed for a wrong dim, got {other:?}"),
  }
  // Wrong rank (expecting rank-3 for a rank-2 tensor).
  assert!(matches!(
    expect_shape(&t, "lm_head.weight", "ctc head", &[32, 768, 1]),
    Err(Error::LayerKeyed(_))
  ));
  // Oversized output dim (the OOM guard).
  assert!(matches!(
    expect_shape(&t, "lm_head.weight", "ctc head", &[1_000_000, 768]),
    Err(Error::LayerKeyed(_))
  ));
}

// ───────────────────────── loader error paths ─────────────────────────

#[test]
fn load_rejects_missing_local_directory() {
  // A non-existent local path is a clear MissingKey, never a panic / network
  // attempt.
  let missing = format!("/nonexistent/mlxrs_wav2vec2_{}/model", std::process::id());
  assert!(matches!(Model::load(&missing), Err(Error::MissingKey(_))));
}

#[test]
fn load_errors_when_safetensors_absent() {
  // A directory with a valid config.json but no model.safetensors is a clear
  // MissingKey (sharded checkpoints are not handled by this single-file path).
  let dir = std::env::temp_dir().join(format!("mlxrs_wav2vec2_load_no_st_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(dir.join("config.json"), r#"{"model_type": "wav2vec2"}"#).unwrap();
  let err = Model::load(&dir.to_string_lossy());
  let _ = std::fs::remove_dir_all(&dir);
  // `Model` is not `Debug` (it holds `Array`s), so assert on the variant
  // without formatting the `Ok` payload.
  assert!(
    matches!(err, Err(Error::MissingKey(_))),
    "expected MissingKey for a dir with no model.safetensors"
  );
}

/// Build a complete **HF pre-sanitize** checkpoint map for `config`, with every
/// backbone key carrying `prefix` (`"wav2vec2."` / `"hubert."`) and `lm_head.*`
/// top-level — exactly the on-disk `*ForCTC` layout the PUBLIC
/// [`Model::load`] path feeds through [`sanitize`]. This is the faithful
/// inverse of what `from_weights` consumes: the conv weights are in HF
/// `(out, in, k)` order (sanitize swaps them to MLX `(out, k, in)`), and the
/// positional weight-norm pair is stored under the PyTorch parametrization names
/// `...parametrizations.weight.original0` / `original1` in HF order (sanitize
/// renames + axis-swaps them to `weight_g` / `weight_v`). Every other tensor
/// (biases, LayerNorm affines, the projection and CTC-head Linears) passes
/// through sanitize unchanged, so it is stored at the same shape `from_weights`
/// expects, merely prefixed.
///
/// Shapes are written longhand (not read from the code under test) so a
/// regression in the production sanitize/shape derivation cannot also shift this
/// oracle.
fn hf_layout_prefixed_weights(c: &Config, prefix: &str) -> HashMap<String, Array> {
  let mut w: HashMap<String, Array> = HashMap::new();
  let bb = |k: &str| format!("{prefix}{k}"); // backbone-prefixed key
  let hs = c.hidden_size;
  let inter = c.intermediate_size;
  let groups = c.num_conv_pos_embedding_groups;
  let kpos = c.num_conv_pos_embeddings;
  // Feature encoder. HF conv weight is (out, in, k) — sanitize swaps (1,2) to
  // the MLX (out, k, in) the builder pins.
  for i in 0..(c.num_feat_extract_layers as usize) {
    let out = c.conv_dim[i];
    let k = c.conv_kernel[i];
    let in_ch = if i == 0 { 1 } else { c.conv_dim[i - 1] };
    w.insert(
      bb(&format!("feature_extractor.conv_layers.{i}.conv.weight")),
      ramp(&[out, in_ch, k], 0.3),
    );
    if c.conv_bias {
      w.insert(
        bb(&format!("feature_extractor.conv_layers.{i}.conv.bias")),
        filled(&[out], 0.0),
      );
    }
    if i == 0 {
      w.insert(
        bb("feature_extractor.conv_layers.0.layer_norm.weight"),
        filled(&[out], 1.0),
      );
      w.insert(
        bb("feature_extractor.conv_layers.0.layer_norm.bias"),
        filled(&[out], 0.0),
      );
    }
  }
  let last = *c.conv_dim.last().unwrap();
  // Feature projection (no sanitize transform — stored verbatim, prefixed).
  w.insert(
    bb("feature_projection.layer_norm.weight"),
    filled(&[last], 1.0),
  );
  w.insert(
    bb("feature_projection.layer_norm.bias"),
    filled(&[last], 0.0),
  );
  w.insert(
    bb("feature_projection.projection.weight"),
    ramp(&[hs, last], 0.3),
  );
  w.insert(bb("feature_projection.projection.bias"), filled(&[hs], 0.0));
  // Positional conv: HF stores the weight-norm reparametrization as
  // `...parametrizations.weight.original0` (the magnitude → weight_g) and
  // `original1` (the direction → weight_v). sanitize renames them and swaps
  // (1,2), so the HF shapes are the (1,2)-swap of the MLX shapes the builder
  // pins: weight_g MLX (1, kpos, 1) ⟸ HF (1, 1, kpos); weight_v MLX
  // (hs, kpos, hs/groups) ⟸ HF (hs, hs/groups, kpos).
  w.insert(
    bb("encoder.pos_conv_embed.conv.parametrizations.weight.original0"),
    filled(&[1, 1, kpos], 1.0),
  );
  w.insert(
    bb("encoder.pos_conv_embed.conv.parametrizations.weight.original1"),
    ramp(&[hs, hs / groups, kpos], 0.3),
  );
  w.insert(bb("encoder.pos_conv_embed.conv.bias"), filled(&[hs], 0.0));
  w.insert(bb("encoder.layer_norm.weight"), filled(&[hs], 1.0));
  w.insert(bb("encoder.layer_norm.bias"), filled(&[hs], 0.0));
  // Transformer layers (no sanitize transform — stored verbatim, prefixed).
  for i in 0..(c.num_hidden_layers as usize) {
    let p = format!("encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      w.insert(
        bb(&format!("{p}.attention.{proj}.weight")),
        ramp(&[hs, hs], 0.3),
      );
      w.insert(
        bb(&format!("{p}.attention.{proj}.bias")),
        filled(&[hs], 0.0),
      );
    }
    w.insert(bb(&format!("{p}.layer_norm.weight")), filled(&[hs], 1.0));
    w.insert(bb(&format!("{p}.layer_norm.bias")), filled(&[hs], 0.0));
    w.insert(
      bb(&format!("{p}.feed_forward.intermediate_dense.weight")),
      ramp(&[inter, hs], 0.3),
    );
    w.insert(
      bb(&format!("{p}.feed_forward.intermediate_dense.bias")),
      filled(&[inter], 0.0),
    );
    w.insert(
      bb(&format!("{p}.feed_forward.output_dense.weight")),
      ramp(&[hs, inter], 0.3),
    );
    w.insert(
      bb(&format!("{p}.feed_forward.output_dense.bias")),
      filled(&[hs], 0.0),
    );
    w.insert(
      bb(&format!("{p}.final_layer_norm.weight")),
      filled(&[hs], 1.0),
    );
    w.insert(
      bb(&format!("{p}.final_layer_norm.bias")),
      filled(&[hs], 0.0),
    );
  }
  // CTC head: top-level (NOT backbone-prefixed), no sanitize transform.
  w.insert("lm_head.weight".to_string(), ramp(&[c.vocab_size, hs], 0.3));
  w.insert("lm_head.bias".to_string(), filled(&[c.vocab_size], 0.0));
  w
}

/// Drive the PUBLIC `load()` path end to end for a backbone-prefixed checkpoint:
/// write a `config.json` + a prefixed HF-layout `model.safetensors` to a temp
/// dir, then `Model::load` it (exercising `sanitize` on real prefixed,
/// HF-ordered keys, not `from_weights` with pre-sanitized ones) and forward to
/// the right logits shape. `config_json` selects the `model_type`; `prefix` is
/// its backbone prefix.
fn assert_load_path_strips_prefix(config_json: &str, prefix: &str) {
  let config = Config::from_json(config_json).unwrap();
  // The on-disk checkpoint carries the backbone prefix + HF tensor order that
  // load()'s sanitize must strip and swap.
  let prefixed = hf_layout_prefixed_weights(&config, prefix);

  let dir = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_loadpath_{}_{}",
    prefix.trim_end_matches('.'),
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(dir.join("config.json"), config_json).unwrap();
  // Save the PREFIXED weights so load()'s sanitize must strip the backbone
  // prefix — the path the synthetic from_weights tests bypass.
  crate::io::save_safetensors(&dir.join("model.safetensors"), &prefixed).unwrap();

  let loaded = Model::load(&dir.to_string_lossy());
  let _ = std::fs::remove_dir_all(&dir);

  let model = match loaded {
    Ok(m) => m,
    Err(e) => panic!("public load() must sanitize the {prefix} prefix and build, got {e:?}"),
  };
  // A real forward proves the prefixed backbone keys were correctly stripped to
  // the unprefixed names the builders consume (a stale prefix would have failed
  // load() with MissingKey above).
  let waveform = filled(&[1, 400], 0.1);
  let mut logits = model
    .forward(&waveform)
    .expect("forward after load must succeed");
  let expected_t = feature_out_len(&config, 400);
  assert_eq!(
    logits.shape(),
    vec![1, expected_t as usize, config.vocab_size as usize],
    "logits must be (1, T'={expected_t}, vocab={})",
    config.vocab_size
  );
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite()),
    "all logits must be finite after the public load path"
  );
}

#[test]
fn load_path_strips_hubert_backbone_prefix() {
  // A real HubertForCTC checkpoint nests its backbone under `hubert.`. The
  // PUBLIC load() path must strip it via sanitize so the builders find the
  // unprefixed `feature_extractor.*` / `encoder.*` keys — without this, load()
  // would fail with MissingKey. This exercises load()/sanitize end to end
  // (config.json + a prefixed model.safetensors on disk), not from_weights with
  // pre-sanitized synthetic keys.
  let json = r#"{
    "model_type": "hubert",
    "hidden_size": 32, "num_attention_heads": 4, "intermediate_size": 64,
    "num_hidden_layers": 2, "vocab_size": 12,
    "num_feat_extract_layers": 3,
    "conv_dim": [16, 16, 16], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
    "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4
  }"#;
  assert_load_path_strips_prefix(json, "hubert.");
}

#[test]
fn load_path_strips_wav2vec2_backbone_prefix() {
  // The wav2vec2 backbone prefix (`wav2vec2.`) is likewise stripped by the
  // public load() path — the same end-to-end round-trip for the Wav2Vec2ForCTC
  // layout, so neither backbone family relies on pre-sanitized inputs.
  let json = r#"{
    "model_type": "wav2vec2",
    "hidden_size": 32, "num_attention_heads": 4, "intermediate_size": 64,
    "num_hidden_layers": 2, "vocab_size": 12,
    "num_feat_extract_layers": 3,
    "conv_dim": [16, 16, 16], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
    "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4
  }"#;
  assert_load_path_strips_prefix(json, "wav2vec2.");
}

// ───────────────── newly-unlocked variants: build + forward shape ─────────────────

/// A small constant-filled tensor — non-zero so the forward isn't numerically
/// degenerate (the norms still well-behave: a constant input has zero variance,
/// eps-stabilized).
fn filled(shape: &[i32], v: f32) -> Array {
  Array::full::<f32>(&shape, v).unwrap()
}

/// A deterministic, **element-varying** tensor of the given shape (a bounded
/// pseudo-random ramp). Varied values across the feature axis are what make the
/// post-norm vs stable-LN block orderings produce different outputs: a single
/// constant fill collapses both arms to the same fixed point (the LayerNorms
/// become trivial on a feature-uniform tensor), masking the structural
/// difference. Values lie in roughly `[-scale, scale]`.
fn ramp(shape: &[i32], scale: f32) -> Array {
  let n: usize = shape.iter().map(|&d| d.max(0) as usize).product();
  let mut data = Vec::with_capacity(n);
  // A simple deterministic LCG-style sequence folded into [-scale, scale];
  // no RNG dependency, fully reproducible.
  let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
  for _ in 0..n {
    state = state
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    let frac = ((state >> 33) as f32) / ((1u64 << 31) as f32); // [0, 2)
    data.push((frac - 1.0) * scale);
  }
  Array::from_slice::<f32>(&data, &shape).unwrap()
}

/// Build a complete **post-sanitize** weight map for an arbitrary
/// [`Config`] — the exact layout `from_weights` consumes, every tensor
/// at the shape the config implies. Norm affines are ones (weight) / zeros
/// (bias); every other tensor is a small constant. Used to drive a real
/// forward over newly-unlocked (large / conv_bias / non-gelu) configs at tiny
/// synthetic dims. Shapes are written longhand (not read from the code under
/// test) so a regression in the production shape derivation cannot also shift
/// this oracle.
fn synthetic_weights(c: &Config) -> HashMap<String, Array> {
  let mut w: HashMap<String, Array> = HashMap::new();
  let hs = c.hidden_size;
  let inter = c.intermediate_size;
  let groups = c.num_conv_pos_embedding_groups;
  let kpos = c.num_conv_pos_embeddings;
  // Feature encoder.
  for i in 0..(c.num_feat_extract_layers as usize) {
    let out = c.conv_dim[i];
    let k = c.conv_kernel[i];
    let in_ch = if i == 0 { 1 } else { c.conv_dim[i - 1] };
    w.insert(
      format!("feature_extractor.conv_layers.{i}.conv.weight"),
      ramp(&[out, k, in_ch], 0.3),
    );
    if c.conv_bias {
      w.insert(
        format!("feature_extractor.conv_layers.{i}.conv.bias"),
        filled(&[out], 0.0),
      );
    }
    if i == 0 {
      // L0 GroupNorm affine (ones weight, zeros bias).
      w.insert(
        "feature_extractor.conv_layers.0.layer_norm.weight".to_string(),
        filled(&[out], 1.0),
      );
      w.insert(
        "feature_extractor.conv_layers.0.layer_norm.bias".to_string(),
        filled(&[out], 0.0),
      );
    }
  }
  let last = *c.conv_dim.last().unwrap();
  // Feature projection.
  w.insert(
    "feature_projection.layer_norm.weight".to_string(),
    filled(&[last], 1.0),
  );
  w.insert(
    "feature_projection.layer_norm.bias".to_string(),
    filled(&[last], 0.0),
  );
  w.insert(
    "feature_projection.projection.weight".to_string(),
    ramp(&[hs, last], 0.3),
  );
  w.insert(
    "feature_projection.projection.bias".to_string(),
    filled(&[hs], 0.0),
  );
  // Positional conv (weight-norm pair) + encoder LayerNorm.
  w.insert(
    "encoder.pos_conv_embed.conv.weight_g".to_string(),
    filled(&[1, kpos, 1], 1.0),
  );
  w.insert(
    "encoder.pos_conv_embed.conv.weight_v".to_string(),
    ramp(&[hs, kpos, hs / groups], 0.3),
  );
  w.insert(
    "encoder.pos_conv_embed.conv.bias".to_string(),
    filled(&[hs], 0.0),
  );
  w.insert("encoder.layer_norm.weight".to_string(), filled(&[hs], 1.0));
  w.insert("encoder.layer_norm.bias".to_string(), filled(&[hs], 0.0));
  // Transformer layers.
  for i in 0..(c.num_hidden_layers as usize) {
    let p = format!("encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      w.insert(format!("{p}.attention.{proj}.weight"), ramp(&[hs, hs], 0.3));
      w.insert(format!("{p}.attention.{proj}.bias"), filled(&[hs], 0.0));
    }
    w.insert(format!("{p}.layer_norm.weight"), filled(&[hs], 1.0));
    w.insert(format!("{p}.layer_norm.bias"), filled(&[hs], 0.0));
    w.insert(
      format!("{p}.feed_forward.intermediate_dense.weight"),
      ramp(&[inter, hs], 0.3),
    );
    w.insert(
      format!("{p}.feed_forward.intermediate_dense.bias"),
      filled(&[inter], 0.0),
    );
    w.insert(
      format!("{p}.feed_forward.output_dense.weight"),
      ramp(&[hs, inter], 0.3),
    );
    w.insert(
      format!("{p}.feed_forward.output_dense.bias"),
      filled(&[hs], 0.0),
    );
    w.insert(format!("{p}.final_layer_norm.weight"), filled(&[hs], 1.0));
    w.insert(format!("{p}.final_layer_norm.bias"), filled(&[hs], 0.0));
  }
  // CTC head.
  w.insert("lm_head.weight".to_string(), ramp(&[c.vocab_size, hs], 0.3));
  w.insert("lm_head.bias".to_string(), filled(&[c.vocab_size], 0.0));
  w
}

/// The conv-output length recurrence (no padding, dilation 1) over the config's
/// conv stack, computed independently of the model.
fn feature_out_len(c: &Config, t_in: i64) -> i64 {
  let mut l = t_in;
  for i in 0..(c.num_feat_extract_layers as usize) {
    l = (l - i64::from(c.conv_kernel[i])) / i64::from(c.conv_stride[i]) + 1;
  }
  l
}

/// A tiny synthetic config (overriding the small dims onto the JSON the test
/// supplies) so a full forward is cheap. Uses 3 feat-extract layers, 2
/// transformer layers, hidden 32 / heads 4 / inter 64, kpos 16 / groups 4.
fn tiny_config_json(extra: &str) -> Config {
  let json = format!(
    r#"{{
      "hidden_size": 32, "num_attention_heads": 4, "intermediate_size": 64,
      "num_hidden_layers": 2, "vocab_size": 12,
      "num_feat_extract_layers": 3,
      "conv_dim": [16, 16, 16], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
      "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4
      {extra}
    }}"#
  );
  Config::from_json(&json).unwrap()
}

/// Forward a tiny model over a `(1, t_in)` waveform and assert the logits shape
/// is `(1, T', vocab)` where `T'` is the analytic feature-encoder output length
/// — exercising the full build + forward graph for a newly-unlocked config.
fn assert_forward_shape(config: Config, t_in: i32) {
  let expected_t = feature_out_len(&config, i64::from(t_in));
  assert!(expected_t > 0, "choose a longer input: T' = {expected_t}");
  let vocab = config.vocab_size as usize;
  let weights = synthetic_weights(&config);
  let model = Model::from_weights(config, weights, Vocab::default())
    .expect("a valid family config must build");
  let waveform = filled(&[1, t_in], 0.1);
  let mut logits = model.forward(&waveform).expect("forward must succeed");
  let shape = logits.shape();
  assert_eq!(
    shape,
    vec![1, expected_t as usize, vocab],
    "logits must be (1, T'={expected_t}, vocab={vocab})"
  );
  // The graph must produce finite logits (no NaN/Inf from the norms / convs).
  let data = logits.to_vec::<f32>().unwrap();
  assert!(
    data.iter().all(|v| v.is_finite()),
    "all logits must be finite"
  );
}

#[test]
fn large_stable_layer_norm_config_builds_and_forwards() {
  // A large-style stable-LN variant (do_stable_layer_norm = true) at tiny
  // synthetic dims: it must parse, build the pre-norm encoder arm, and forward
  // to the right logits shape. Exercises the stable-LN block ordering + the
  // encoder-level LayerNorm-at-the-end placement end to end.
  let config = tiny_config_json(r#", "do_stable_layer_norm": true"#);
  assert!(config.do_stable_layer_norm);
  assert_forward_shape(config, 400);
}

#[test]
fn full_large_lv60_style_config_parses_and_validates() {
  // The real large-960h-lv60-self transformer dims (1024 hidden / 16 heads / 24
  // layers / 4096 intermediate, stable-LN) with its defining `feat_extract_norm
  // = "layer"` extractor must parse, validate, and resolve to the LayerNorm
  // feature-extractor scheme (the arm this checkpoint actually uses).
  let json = r#"{
    "model_type": "wav2vec2",
    "hidden_size": 1024, "num_attention_heads": 16, "num_hidden_layers": 24,
    "intermediate_size": 4096, "vocab_size": 32,
    "feat_extract_norm": "layer",
    "do_stable_layer_norm": true,
    "num_conv_pos_embeddings": 128, "num_conv_pos_embedding_groups": 16
  }"#;
  let config = Config::from_json(json).unwrap();
  assert_eq!(config.hidden_size, 1024);
  assert_eq!(config.num_hidden_layers, 24);
  assert!(config.do_stable_layer_norm);
  assert_eq!(
    config.feat_extract_norm_scheme().unwrap(),
    FeatExtractNorm::Layer
  );
  assert!(config.validate().is_ok());
}

#[test]
fn conv_bias_config_builds_and_forwards() {
  // A conv_bias = true variant: every ConvLayer must load and add its
  // conv.bias. The model must build (the bias tensors are consumed, not
  // dropped) and forward to the right shape.
  let config = tiny_config_json(r#", "conv_bias": true"#);
  assert!(config.conv_bias);
  assert_forward_shape(config, 400);
}

#[test]
fn conv_bias_config_requires_bias_tensors() {
  // With conv_bias = true, the builder consumes a `.conv.bias` per layer; a
  // checkpoint missing one is a clear MissingKey (not a silent drop).
  let config = tiny_config_json(r#", "conv_bias": true"#);
  let mut weights = synthetic_weights(&config);
  weights.remove("feature_extractor.conv_layers.1.conv.bias");
  match Model::from_weights(config, weights, Vocab::default()) {
    Err(Error::MissingKey(p)) => {
      assert!(
        p.key().contains("conv_layers.1.conv.bias"),
        "the missing key should name the absent conv bias, got {:?}",
        p.key()
      );
    }
    Err(other) => panic!("expected MissingKey for an absent conv bias, got {other:?}"),
    Ok(_) => panic!("expected MissingKey for an absent conv bias, but the model built"),
  }
}

#[test]
fn non_gelu_activation_config_builds_and_forwards() {
  // A non-gelu activation variant (silu feed-forward + tanh-approx-gelu feature
  // encoder) must parse, build, and forward to the right shape — the dispatch
  // wires the configured activations rather than hardcoding GELU.
  let config = tiny_config_json(r#", "hidden_act": "silu", "feat_extract_activation": "gelu_new""#);
  assert_eq!(config.hidden_act, "silu");
  assert_eq!(config.feat_extract_activation, "gelu_new");
  assert_forward_shape(config, 400);
}

#[test]
fn hubert_model_type_builds_and_forwards() {
  // The hubert CTC checkpoint shares the plain self-attention transformer wired
  // here (HuBERT reuses the wav2vec2 encoder architecture); a `hubert` config
  // must build and forward to the right shape.
  let config = tiny_config_json(r#", "model_type": "hubert""#);
  assert_eq!(config.model_type(), "hubert");
  assert_forward_shape(config, 400);
}

#[test]
fn wavlm_model_type_is_rejected() {
  // WavLM's defining feature is gated relative-position-bias attention, which
  // this phase does not implement, and it has no plain-attention variant. So a
  // `wavlm` checkpoint cannot be run faithfully through the plain attention
  // path (its relative-position tensors would go unconsumed = silent
  // corruption). `validate` must REJECT it with a typed UnknownEnumValue
  // carrying the offending value and the (wavlm-free) supported set — never
  // silently accept a model this phase cannot serve.
  let config = tiny_config_json(r#", "model_type": "wavlm""#);
  assert_eq!(config.model_type(), "wavlm");
  match config.validate() {
    Err(Error::UnknownEnumValue(p)) => {
      assert_eq!(p.value(), "wavlm");
      assert_eq!(
        p.supported(),
        &["wav2vec2", "hubert"],
        "the supported model_type set must be wav2vec2 + hubert only (no wavlm)"
      );
    }
    other => panic!("expected UnknownEnumValue rejecting wavlm, got {other:?}"),
  }
  // The rejection is also enforced at the construction boundary: from_weights
  // runs `validate` first, so a wavlm config never builds a model (even with an
  // otherwise-complete weight set).
  let weights = synthetic_weights(&tiny_config_json(r#", "model_type": "wavlm""#));
  assert!(
    matches!(
      Model::from_weights(
        tiny_config_json(r#", "model_type": "wavlm""#),
        weights,
        Vocab::default()
      ),
      Err(Error::UnknownEnumValue(_))
    ),
    "from_weights must reject a wavlm config via the validate gate"
  );
}

#[test]
fn post_norm_and_stable_layer_norm_differ() {
  // The two encoder arms must produce DIFFERENT logits from the same weights
  // and input — confirming the stable-LN arm is a distinct graph (different
  // block ordering + encoder-LayerNorm placement), not an accidental alias of
  // the post-norm arm.
  let post = tiny_config_json("");
  let stable = tiny_config_json(r#", "do_stable_layer_norm": true"#);
  assert!(!post.do_stable_layer_norm);
  assert!(stable.do_stable_layer_norm);

  let waveform = filled(&[1, 400], 0.1);
  let mut post_logits = Model::from_weights(
    post,
    synthetic_weights(&tiny_config_json("")),
    Vocab::default(),
  )
  .unwrap()
  .forward(&waveform)
  .unwrap();
  let mut stable_logits = Model::from_weights(
    stable,
    synthetic_weights(&tiny_config_json(r#", "do_stable_layer_norm": true"#)),
    Vocab::default(),
  )
  .unwrap()
  .forward(&waveform)
  .unwrap();
  // Same shape, different values.
  assert_eq!(post_logits.shape(), stable_logits.shape());
  let a = post_logits.to_vec::<f32>().unwrap();
  let b = stable_logits.to_vec::<f32>().unwrap();
  assert!(
    a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
    "post-norm and stable-LN arms must differ"
  );
}

// ───────────────────── positional-conv activation ─────────────────────

/// VALUE-LEVEL: the positional conv embedding must apply the configured
/// `feat_extract_activation` (HF's `ACT2FN[config.feat_extract_activation]`),
/// **not** a hardcoded GELU. For a non-gelu activation (silu / gelu_new) the
/// output must equal that activation applied to the conv+bias+SamePad result —
/// a hardcoded GELU would diverge from this oracle.
///
/// The oracle recomputes the pre-activation tensor through the **public** conv
/// op (`ops::conv::conv1d` + the bias add + the SamePad crop), independently of
/// `PositionalConvEmbedding::forward`, then applies the standalone activation
/// primitive — so it pins which activation `forward` applies (a shape assertion
/// alone cannot, since every activation preserves shape).
#[test]
fn positional_conv_applies_configured_activation() {
  use crate::lm::nn::activations;

  // A tiny grouped positional conv: hidden 8, groups 2 (in/group = 4), an EVEN
  // kernel 4 (so SamePad crops one trailing frame, exercising that branch too),
  // padding = kernel/2 = 2. Channels-last (B, T, C) input.
  let hidden = 8i32;
  let groups = 2i32;
  let kernel = 4i32;
  let in_per_group = hidden / groups;
  // Deterministic, element-varying weight/bias/input so the activation's
  // nonlinearity is actually exercised (a constant fill would not distinguish
  // silu from gelu well).
  let weight = ramp(&[hidden, kernel, in_per_group], 0.4);
  let bias = ramp(&[hidden], 0.2);
  let x = ramp(&[1, 12, hidden], 0.5);

  for (name, act) in [
    ("silu", Activation::Silu),
    ("gelu_new", Activation::GeluApprox),
  ] {
    let pos = PositionalConvEmbedding {
      weight: weight.try_clone().unwrap(),
      bias: bias.try_clone().unwrap(),
      groups,
      padding: kernel / 2,
      num_pad_remove: 1, // even kernel
      activation: act,
    };
    let mut got = pos.forward(&x).unwrap();

    // Oracle pre-activation: conv1d + bias + SamePad crop, via the public ops
    // (NOT through pos.forward), then the standalone activation primitive.
    let conv = ops::conv::conv1d(&x, &weight, 1, kernel / 2, 1, groups).unwrap();
    let pre = conv.add(&bias).unwrap();
    // SamePad: drop the single trailing time frame (axis 1 in (B, T, C)).
    let shape = pre.shape();
    let stop: Vec<i32> = shape
      .iter()
      .enumerate()
      .map(|(ax, &d)| if ax == 1 { d as i32 - 1 } else { d as i32 })
      .collect();
    let start = vec![0i32; shape.len()];
    let strides = vec![1i32; shape.len()];
    let cropped = ops::indexing::slice(&pre, &start, &stop, &strides).unwrap();
    let mut want = match act {
      Activation::Silu => activations::silu(&cropped),
      Activation::GeluApprox => activations::gelu_approx(&cropped),
      Activation::Gelu => activations::gelu(&cropped),
    }
    .unwrap();

    assert_eq!(
      got.shape(),
      want.shape(),
      "{name}: positional-conv output shape must match the oracle"
    );
    let g = got.to_vec::<f32>().unwrap();
    let w = want.to_vec::<f32>().unwrap();
    for (a, b) in g.iter().zip(w.iter()) {
      assert!(
        (a - b).abs() < 1e-5,
        "{name}: positional conv must apply the configured activation \
         (got {a}, want {b}) — a hardcoded GELU would diverge here"
      );
    }

    // Cross-check: the same input through the EXACT-GELU embedding must NOT
    // match the silu/gelu_new oracle (proving the activation is honoured, not a
    // GELU that happens to be close). Skipped for the gelu_new case only where
    // the tanh-approx is numerically near exact GELU; silu is unambiguously
    // distinct.
    if matches!(act, Activation::Silu) {
      let gelu_pos = PositionalConvEmbedding {
        weight: weight.try_clone().unwrap(),
        bias: bias.try_clone().unwrap(),
        groups,
        padding: kernel / 2,
        num_pad_remove: 1,
        activation: Activation::Gelu,
      };
      let mut gelu_out = gelu_pos.forward(&x).unwrap();
      let go = gelu_out.to_vec::<f32>().unwrap();
      assert!(
        go.iter().zip(w.iter()).any(|(x, y)| (x - y).abs() > 1e-4),
        "a hardcoded exact-GELU positional conv must differ from the silu oracle"
      );
    }
  }
}

// ───────────────────── dtype preservation ─────────────────────

/// Cast every tensor in a (post-sanitize) weight map to `dtype`, for the
/// half-precision dtype-preservation tests. Norm affines / biases / weights all
/// move to the target dtype, mirroring a real F16 / BF16 checkpoint.
fn cast_weights(
  weights: HashMap<String, Array>,
  dtype: crate::dtype::Dtype,
) -> HashMap<String, Array> {
  weights
    .into_iter()
    .map(|(k, v)| (k, v.astype(dtype).unwrap()))
    .collect()
}

/// ISOLATED ATTENTION: the attention block's query scale must be
/// built in the operand dtype, so a half-precision (F16 / BF16) input stays
/// half-precision through attention. A bare F32 scale would promote the whole
/// block to F32 under MLX type promotion (breaking mixed-precision numerics +
/// inflating memory / the KV-cache). Asserts the attention OUTPUT dtype equals
/// the input dtype (i.e. does NOT silently become F32).
#[test]
fn attention_preserves_half_precision_dtype() {
  let hidden = 8i32;
  let heads = 2i32;
  let head_dim = hidden / heads;
  for dtype in [crate::dtype::Dtype::F16, crate::dtype::Dtype::BF16] {
    let mk = |shape: &[i32], scale: f32| ramp(shape, scale).astype(dtype).unwrap();
    // Build each projection as a dense quantize-aware `Linear` (the adoption of
    // the shared `MaybeQuantizedLinear`); the dtype-preservation contract is on
    // the dense path here.
    let mk_proj = |scale_w: f32, scale_b: f32| {
      Linear::new(mk(&[hidden, hidden], scale_w), Some(mk(&[hidden], scale_b)))
    };
    let attn = Attention {
      q_proj: mk_proj(0.3, 0.1),
      k_proj: mk_proj(0.3, 0.1),
      v_proj: mk_proj(0.3, 0.1),
      out_proj: mk_proj(0.3, 0.1),
      num_heads: heads,
      head_dim,
      scaling: (head_dim as f32).powf(-0.5),
    };
    let x = ramp(&[1, 5, hidden], 0.5).astype(dtype).unwrap();
    let out = attn.forward(&x).unwrap();
    assert_eq!(
      out.dtype().unwrap(),
      dtype,
      "attention must preserve {dtype:?} — a bare F32 query scale would promote it to F32"
    );
  }
}

/// WHOLE-FORWARD dtype preservation: a tiny model built
/// from F16 / BF16 weights, forwarded over a same-dtype waveform, must produce
/// logits of that same dtype — the final output dtype must equal the
/// weight/input dtype, never silently F32. Covers every dtype-meets-activation
/// site on the real forward path (attention scale, positional conv,
/// activations, the norms), which the f32-only suite misses.
#[test]
fn forward_preserves_half_precision_dtype() {
  for dtype in [crate::dtype::Dtype::F16, crate::dtype::Dtype::BF16] {
    let config = tiny_config_json("");
    let weights = cast_weights(synthetic_weights(&tiny_config_json("")), dtype);
    let model = Model::from_weights(config, weights, Vocab::default()).unwrap();
    let waveform = ramp(&[1, 400], 0.5).astype(dtype).unwrap();
    let logits = model.forward(&waveform).unwrap();
    assert_eq!(
      logits.dtype().unwrap(),
      dtype,
      "the whole forward must preserve {dtype:?} (no f32-built scalar / scale / \
       eps may promote the activations) — final logits dtype must match the weights"
    );
    // The half-precision graph must still be finite (no NaN/Inf). Read back via
    // an explicit f32 upcast (the logits are half-precision, so a direct
    // `to_vec::<f32>` would dtype-mismatch — the upcast is test-only).
    let data = logits
      .astype(crate::dtype::Dtype::F32)
      .unwrap()
      .to_vec::<f32>()
      .unwrap();
    assert!(
      data.iter().all(|v| v.is_finite()),
      "all {dtype:?} logits must be finite"
    );
  }
}

/// WHOLE-FORWARD, stable-LN arm: the pre-norm encoder must likewise preserve
/// half precision (it runs the same attention + positional conv + activations,
/// only reordered), so cover it too.
#[test]
fn forward_stable_ln_preserves_half_precision_dtype() {
  for dtype in [crate::dtype::Dtype::F16, crate::dtype::Dtype::BF16] {
    let extra = r#", "do_stable_layer_norm": true"#;
    let config = tiny_config_json(extra);
    let weights = cast_weights(synthetic_weights(&tiny_config_json(extra)), dtype);
    let model = Model::from_weights(config, weights, Vocab::default()).unwrap();
    let waveform = ramp(&[1, 400], 0.5).astype(dtype).unwrap();
    let logits = model.forward(&waveform).unwrap();
    assert_eq!(
      logits.dtype().unwrap(),
      dtype,
      "the stable-LN forward must preserve {dtype:?}"
    );
    let data = logits
      .astype(crate::dtype::Dtype::F32)
      .unwrap()
      .to_vec::<f32>()
      .unwrap();
    assert!(
      data.iter().all(|v| v.is_finite()),
      "all stable-LN {dtype:?} logits must be finite"
    );
  }
}

// ───────────────────── quantized-checkpoint loading ─────────────────────
//
// No local 8-bit wav2vec2 checkpoint is available, so the quantized load path
// is covered by a SYNTHETIC quantized checkpoint: a tiny config whose every
// quantized Linear's input width (`hidden_size`, `intermediate_size`,
// `conv_dim[-1]`) is divisible by the affine `group_size` (mlx requires
// `group_size ∈ {32, 64, 128}`), with every encoder attention / feed-forward
// projection, the feature projection, and the CTC `lm_head` weight replaced by
// the real `ops::quantized::quantize` `(weight, scales, biases)` triple — the
// exact on-disk layout an mlx-community 8-bit checkpoint ships. The model must
// then construct (building `MaybeQuantizedLinear::Quantized` layers) and run a
// full forward to the right CTC-logits shape with finite output. The
// convolutional feature extractor + positional conv stay DENSE (the
// `class_predicate` only quantizes `nn.Linear`).

/// A valid mlx affine group size (`group_size ∈ {32, 64, 128}`).
const QGROUP: i32 = 32;
/// 8-bit affine — the common mlx-community quantized scheme.
const QBITS: i32 = 8;

/// A tiny quant-friendly config: every quantized Linear's input width
/// (`hidden_size = 64`, `intermediate_size = 128`, `conv_dim[-1] = 64`) is a
/// whole number of `QGROUP` groups. 3 feat-extract layers, 2 transformer layers.
fn quant_config_json(extra: &str) -> Config {
  let json = format!(
    r#"{{
      "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 128,
      "num_hidden_layers": 2, "vocab_size": 32,
      "num_feat_extract_layers": 3,
      "conv_dim": [64, 64, 64], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
      "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4
      {extra}
    }}"#
  );
  Config::from_json(&json).unwrap()
}

/// Replace the dense `<prefix>.weight` in `w` with the real
/// `ops::quantized::quantize` affine triple (`<prefix>.weight` packed +
/// `<prefix>.scales` + `<prefix>.biases`) at the given affine `group_size`,
/// mirroring how an mlx-community quantized checkpoint stores a quantized
/// `Linear`. The `<prefix>.bias` (dense output bias), if any, is left untouched.
/// `group_size` must divide `<prefix>.weight`'s last (input) axis.
fn quantize_weight_in_place(w: &mut HashMap<String, Array>, prefix: &str, group_size: i32) {
  let dense = w
    .remove(&format!("{prefix}.weight"))
    .unwrap_or_else(|| panic!("dense weight {prefix}.weight present"));
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, group_size, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

/// Build a synthetic post-sanitize quantized wav2vec2 checkpoint at an EXPLICIT
/// affine `group_size`: the dense `synthetic_weights` for `config`, then every
/// quantized-eligible Linear (encoder attention `q/k/v/out`, feed-forward
/// `intermediate` / `output`, the feature projection, and the CTC `lm_head`)
/// replaced by the 8-bit affine triple. The conv feature extractor + positional
/// conv stay DENSE — only `nn.Linear` is quantized, matching mlx-audio / MLX.
/// `group_size` must divide every quantized Linear's input width
/// (`hidden_size` / `intermediate_size` / `conv_dim[-1]`).
fn quant_weights_at(config: &Config, group_size: i32) -> HashMap<String, Array> {
  let mut w = synthetic_weights(config);
  // Feature projection Linear.
  quantize_weight_in_place(&mut w, "feature_projection.projection", group_size);
  // Per-layer attention + feed-forward projections.
  for i in 0..(config.num_hidden_layers as usize) {
    let p = format!("encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      quantize_weight_in_place(&mut w, &format!("{p}.attention.{proj}"), group_size);
    }
    quantize_weight_in_place(
      &mut w,
      &format!("{p}.feed_forward.intermediate_dense"),
      group_size,
    );
    quantize_weight_in_place(
      &mut w,
      &format!("{p}.feed_forward.output_dense"),
      group_size,
    );
  }
  // CTC head Linear.
  quantize_weight_in_place(&mut w, "lm_head", group_size);
  w
}

/// The synthetic quantized checkpoint at the module-default `QGROUP` — the
/// shape an mlx-community 8-bit checkpoint ships for these dims.
fn quant_weights(config: &Config) -> HashMap<String, Array> {
  quant_weights_at(config, QGROUP)
}

/// The parsed global 8-bit affine quantization config for the synthetic
/// checkpoint (the analogue of the `config.json` `quantization` block).
fn quant_config() -> PerLayerQuantization {
  PerLayerQuantization::from_global(crate::lm::quant::Quantization::affine(QGROUP, QBITS))
}

#[test]
fn from_weights_quantized_builds_quantized_layers() {
  // With a quantization config and a checkpoint whose Linear weights carry
  // `.scales`/`.biases`, the model builds quantized layers — and still builds
  // (a packed `uint32` weight of a DIFFERENT shape than the dense `(out, in)`
  // would otherwise be rejected by the dense shape gate).
  let config = quant_config_json("");
  let model = Model::from_weights_quantized(
    config,
    quant_weights(&quant_config_json("")),
    Vocab::default(),
    Some(&quant_config()),
  )
  .expect("an 8-bit quantized checkpoint must build through the quantized path");
  // The CTC head is the quantized variant (a dense `.weight` would have made
  // `is_quantized()` false).
  assert!(
    model.lm_head.is_quantized(),
    "the CTC lm_head must load as a quantized projection"
  );
  // The conv feature extractor stays dense (it is not a `nn.Linear`); its first
  // conv weight is still a plain f32 tensor, never quantized.
  let conv0 = &model.feature_encoder.conv_layers[0].weight;
  assert_eq!(
    conv0.dtype().unwrap(),
    crate::dtype::Dtype::F32,
    "the conv feature extractor must stay dense (never quantized)"
  );
}

#[test]
fn from_weights_quantized_runs_forward_to_finite_ctc_logits() {
  // The real GOAL contract on a synthetic stand-in: an 8-bit checkpoint loads
  // AND runs the full forward to FINITE per-frame CTC logits of the right shape
  // (the quantized attention / feed-forward / projection / lm_head
  // `quantized_matmul` all execute through mlx-c, the conv front-end runs dense).
  let config = quant_config_json("");
  let expected_t = feature_out_len(&config, 400);
  assert!(expected_t > 0, "choose a longer input: T' = {expected_t}");
  let vocab = config.vocab_size as usize;
  let model = Model::from_weights_quantized(
    quant_config_json(""),
    quant_weights(&config),
    Vocab::default(),
    Some(&quant_config()),
  )
  .unwrap();

  let waveform = filled(&[1, 400], 0.1);
  let mut logits = model
    .forward(&waveform)
    .expect("quantized forward must succeed");
  assert_eq!(
    logits.shape(),
    vec![1, expected_t as usize, vocab],
    "quantized logits must be (1, T'={expected_t}, vocab={vocab})"
  );
  let data = logits.to_vec::<f32>().unwrap();
  assert!(
    data.iter().all(|v| v.is_finite()),
    "all quantized CTC logits must be finite"
  );
}

#[test]
fn from_weights_quantized_stable_ln_runs_forward() {
  // The stable-LN (pre-norm) encoder arm must likewise load + forward a
  // quantized checkpoint (it runs the same quantized projections, only
  // reordered) to finite logits of the right shape.
  let config = quant_config_json(r#", "do_stable_layer_norm": true"#);
  assert!(config.do_stable_layer_norm);
  let expected_t = feature_out_len(&config, 400);
  let vocab = config.vocab_size as usize;
  let model = Model::from_weights_quantized(
    quant_config_json(r#", "do_stable_layer_norm": true"#),
    quant_weights(&config),
    Vocab::default(),
    Some(&quant_config()),
  )
  .unwrap();
  let waveform = filled(&[1, 400], 0.1);
  let mut logits = model.forward(&waveform).unwrap();
  assert_eq!(logits.shape(), vec![1, expected_t as usize, vocab]);
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite()),
    "stable-LN quantized logits must be finite"
  );
}

#[test]
fn from_weights_quantized_dense_checkpoint_unchanged() {
  // A NON-quantized checkpoint loads identically whether or not a quantization
  // config is threaded (the `.scales` sibling is the load-bearing signal; a
  // dense checkpoint has none, so the dense path runs regardless). Produces the
  // same logits shape as the plain `from_weights` path.
  let config = quant_config_json("");
  let expected_t = feature_out_len(&config, 400);
  let vocab = config.vocab_size as usize;
  let model = Model::from_weights_quantized(
    quant_config_json(""),
    synthetic_weights(&config),
    Vocab::default(),
    Some(&quant_config()),
  )
  .expect("a dense checkpoint loads even when a quantization config is supplied");
  // None of its projections is quantized (no `.scales` siblings present).
  assert!(
    !model.lm_head.is_quantized(),
    "a dense checkpoint's lm_head must stay dense even with a quant config"
  );
  let waveform = filled(&[1, 400], 0.1);
  let logits = model.forward(&waveform).unwrap();
  assert_eq!(logits.shape(), vec![1, expected_t as usize, vocab]);
}

#[test]
fn from_weights_quantized_scales_without_config_errors() {
  // Weights say quantized (`.scales` present) but no quantization config
  // resolved scheme params → a typed InvariantViolation, not a silent wrong
  // load. Thread an empty per-layer config with no global default so
  // `quantization_for` returns None for the quantized layer.
  let config = quant_config_json("");
  let empty_cfg = PerLayerQuantization::new(None, HashMap::new());
  // `Model` is not `Debug` (it holds `Array`s), so match rather than
  // `.unwrap_err()` (which would need the `Ok` value to be `Debug`).
  match Model::from_weights_quantized(
    config,
    quant_weights(&quant_config_json("")),
    Vocab::default(),
    Some(&empty_cfg),
  ) {
    Err(Error::InvariantViolation(_)) => {}
    Err(other) => {
      panic!(
        "expected InvariantViolation for `.scales` present but no resolved params, got {other:?}"
      )
    }
    Ok(_) => panic!("expected an error for a `.scales` sibling with no resolved scheme params"),
  }
}

#[test]
fn from_weights_quantized_dense_path_identical_to_from_weights() {
  // The dense path is byte-for-byte unchanged: `from_weights` (the public
  // non-quantized entry) and `from_weights_quantized(.., None)` over the same
  // dense checkpoint must produce identical logits (same graph, same weights).
  let config = quant_config_json("");
  let waveform = ramp(&[1, 400], 0.5);

  let model_plain = Model::from_weights(
    quant_config_json(""),
    synthetic_weights(&config),
    Vocab::default(),
  )
  .unwrap();
  let mut logits_plain = model_plain.forward(&waveform).unwrap();

  let model_none = Model::from_weights_quantized(
    quant_config_json(""),
    synthetic_weights(&config),
    Vocab::default(),
    None,
  )
  .unwrap();
  let mut logits_none = model_none.forward(&waveform).unwrap();

  assert_eq!(
    logits_plain.to_vec::<f32>().unwrap(),
    logits_none.to_vec::<f32>().unwrap(),
    "from_weights and from_weights_quantized(.., None) must be identical on a dense checkpoint"
  );
}

// ───────────────── on-disk quantized `load()` path ─────────────────
//
// The `from_weights_quantized` tests above thread an already-parsed
// `PerLayerQuantization` straight into the constructor, so they never exercise
// how the public `load()` entry resolves the `config.json` quantization block.
// `load()` must resolve it through the shared audio resolver
// `crate::audio::load::apply_quantization`, which (mirroring mlx-audio) accepts
// EITHER a top-level `quantization` block OR the HF `quantization_config` key
// and defaults a missing `group_size` to 64. These tests drive a real on-disk
// checkpoint (a `config.json` + a quantized `model.safetensors`) through the
// full `load()` path — `sanitize` included — for exactly those two shapes.

/// A whole-checkpoint **HF pre-sanitize** layout with every quantized-eligible
/// `nn.Linear` replaced by its `ops::quantized::quantize` affine triple at
/// `group_size`. Starts from [`hf_layout_prefixed_weights`] (so the conv /
/// positional weights are in the on-disk HF order [`sanitize`] swaps) and
/// quantizes only the Linear weights — which `sanitize` passes through
/// unchanged, so quantizing them in their on-disk (prefixed / top-level) form is
/// exactly what an mlx-community quantized `*ForCTC` checkpoint ships. The conv
/// feature extractor and positional conv stay DENSE (only `nn.Linear`
/// quantizes). `group_size` must divide every quantized Linear's input width.
fn hf_quant_layout_weights(c: &Config, prefix: &str, group_size: i32) -> HashMap<String, Array> {
  let mut w = hf_layout_prefixed_weights(c, prefix);
  // Backbone Linears keep the backbone prefix on disk; `lm_head` is top-level.
  quantize_weight_in_place(
    &mut w,
    &format!("{prefix}feature_projection.projection"),
    group_size,
  );
  for i in 0..(c.num_hidden_layers as usize) {
    let p = format!("{prefix}encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      quantize_weight_in_place(&mut w, &format!("{p}.attention.{proj}"), group_size);
    }
    quantize_weight_in_place(
      &mut w,
      &format!("{p}.feed_forward.intermediate_dense"),
      group_size,
    );
    quantize_weight_in_place(
      &mut w,
      &format!("{p}.feed_forward.output_dense"),
      group_size,
    );
  }
  quantize_weight_in_place(&mut w, "lm_head", group_size);
  w
}

/// Drive the PUBLIC `load()` path for a quantized checkpoint whose `config.json`
/// expresses its quantization scheme via the verbatim `quant_block` JSON
/// fragment (e.g. `"quantization_config": {...}` or a `"quantization"` block
/// without `group_size`). The on-disk `model.safetensors` is the HF pre-sanitize
/// layout with its Linears quantized at `weight_group_size` (the group size the
/// resolved scheme must agree on). Writes the `config.json` + weights to a temp
/// dir, calls `Model::load`, and asserts the quantized path was taken (the
/// CTC head loaded quantized) and a real forward produces finite CTC logits of
/// the right shape. The model dims are pinned here (mirroring
/// `quant_config_json("")`) so the saved quantized weights line up with the
/// config the loader parses.
fn assert_quant_load_path(quant_block: &str, tag: &str, weight_group_size: i32) {
  let config_json = format!(
    r#"{{
      "model_type": "wav2vec2",
      "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 128,
      "num_hidden_layers": 2, "vocab_size": 32,
      "num_feat_extract_layers": 3,
      "conv_dim": [64, 64, 64], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
      "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4,
      {quant_block}
    }}"#
  );
  let config = Config::from_json(&config_json).unwrap();
  let weights = hf_quant_layout_weights(&config, "wav2vec2.", weight_group_size);

  let dir = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_quantload_{tag}_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(dir.join("config.json"), &config_json).unwrap();
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();

  let loaded = Model::load(&dir.to_string_lossy());
  let _ = std::fs::remove_dir_all(&dir);

  let model = match loaded {
    Ok(m) => m,
    Err(e) => {
      panic!("public load() must resolve the {tag} quantization block and build, got {e:?}")
    }
  };
  // A dense `.weight` would have made `is_quantized()` false — proving the
  // quantization block was resolved (not silently dropped, the bug this guards)
  // is exactly that the CTC head loaded as a quantized projection.
  assert!(
    model.lm_head.is_quantized(),
    "the {tag} checkpoint must load its CTC lm_head through the quantized path"
  );
  let expected_t = feature_out_len(&config, 400);
  let waveform = filled(&[1, 400], 0.1);
  let mut logits = model
    .forward(&waveform)
    .expect("quantized forward after load must succeed");
  assert_eq!(
    logits.shape(),
    vec![1, expected_t as usize, config.vocab_size as usize],
    "{tag} quantized logits must be (1, T'={expected_t}, vocab={})",
    config.vocab_size
  );
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite()),
    "all {tag} quantized CTC logits must be finite after the public load path"
  );
}

#[test]
fn load_path_resolves_hf_quantization_config_key() {
  // The exact case the LM-only parser dropped: a VALID 8-bit checkpoint whose
  // `config.json` expresses quantization under the HF `quantization_config` key
  // (NOT a top-level `quantization` block). The shared audio resolver falls back
  // to that key, so `load()` must take the quantized path; the LM parser would
  // have returned `None` and the dense shape gate would then reject the packed
  // uint32 weight, failing a valid checkpoint to load.
  let block = format!(r#""quantization_config": {{ "group_size": {QGROUP}, "bits": {QBITS} }}"#);
  assert_quant_load_path(&block, "hf_quantization_config", QGROUP);
}

#[test]
fn load_path_defaults_missing_group_size_to_64() {
  // A `quantization` block that OMITS `group_size` must default it to 64 (the
  // mlx-audio convention the shared resolver applies); the LM parser rejects a
  // missing `group_size` outright. The on-disk weights are quantized at
  // group_size 64 so they line up with the defaulted scheme — a successful load
  // + forward proves the default was applied (a wrong default would mis-shape
  // the `.scales` and be rejected).
  let block = format!(r#""quantization": {{ "bits": {QBITS} }}"#);
  assert_quant_load_path(&block, "missing_group_size", 64);
}

/// The quantization `group_size` a built [`Linear`] (the wav2vec2 wrapper) was
/// loaded with, or `None` when the layer is dense. Reads the wrapper's inner
/// [`MaybeQuantizedLinear`] (a private field reachable from this child module),
/// proving which scheme a layer actually resolved — the exact fact the per-layer
/// override reprojection must get right.
fn loaded_group_size(layer: &super::Linear) -> Option<i32> {
  match &layer.inner {
    crate::nn::MaybeQuantizedLinear::Quantized(q) => Some(q.group_size()),
    crate::nn::MaybeQuantizedLinear::Dense(_) => None,
  }
}

/// The transformer layer stack of a built [`Encoder`], regardless of its
/// (post-norm / stable-LN) arm — both arms share the same `EncoderInner` layout.
fn encoder_layers(encoder: &super::StandardEncoder) -> &[super::EncoderLayer] {
  match encoder {
    super::StandardEncoder::PostNorm(inner) | super::StandardEncoder::StableLayerNorm(inner) => {
      &inner.layers
    }
  }
}

#[test]
fn load_path_applies_hf_keyed_per_layer_quant_override() {
  // The finding this guards: a per-layer quantization override keyed in the
  // on-disk HF form (carrying the `wav2vec2.` backbone prefix) must reach the
  // layer it names. `load()` runs the weights through `sanitize` (which strips
  // that prefix) and the builders then resolve a layer's scheme by its SANITIZED
  // prefix, so an override keyed `wav2vec2.encoder.layers.0.attention.q_proj`
  // would never match the `encoder.layers.0.attention.q_proj` lookup and the
  // layer would silently fall back to the GLOBAL scheme. With the reprojection,
  // the override's per-layer `group_size` must win for exactly that one layer
  // while every other quantized layer keeps the global `group_size`.
  const GLOBAL_GROUP: i32 = 64;
  const OVERRIDE_GROUP: i32 = 32;
  // Sanity: the two group sizes differ, so reading back `OVERRIDE_GROUP` on the
  // overridden layer can only mean the override was applied (not the global).
  assert_ne!(GLOBAL_GROUP, OVERRIDE_GROUP);

  // The override is keyed in the HF on-disk form (with the backbone prefix); its
  // `group_size` differs from the global default.
  let config_json = format!(
    r#"{{
      "model_type": "wav2vec2",
      "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 128,
      "num_hidden_layers": 2, "vocab_size": 32,
      "num_feat_extract_layers": 3,
      "conv_dim": [64, 64, 64], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
      "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4,
      "quantization": {{
        "group_size": {GLOBAL_GROUP}, "bits": {QBITS},
        "wav2vec2.encoder.layers.0.attention.q_proj": {{ "group_size": {OVERRIDE_GROUP}, "bits": {QBITS} }}
      }}
    }}"#
  );
  let config = Config::from_json(&config_json).unwrap();

  // On-disk HF-layout weights: start from the DENSE prefixed layout, then pack
  // each quantized-eligible Linear at its scheme's group size — the single
  // overridden layer at the OVERRIDE size, every other eligible Linear at the
  // GLOBAL size — quantizing each layer exactly once from its dense weight so
  // the on-disk `.scales` line up with the per-layer scheme the config declares
  // (a mismatch would be caught by the shape gate). The conv feature extractor +
  // positional conv stay dense (only `nn.Linear` quantizes).
  let mut weights = hf_layout_prefixed_weights(&config, "wav2vec2.");
  const OVERRIDE_LAYER: &str = "wav2vec2.encoder.layers.0.attention.q_proj";
  quantize_weight_in_place(
    &mut weights,
    "wav2vec2.feature_projection.projection",
    GLOBAL_GROUP,
  );
  for i in 0..(config.num_hidden_layers as usize) {
    let p = format!("wav2vec2.encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      let prefix = format!("{p}.attention.{proj}");
      let group = if prefix == OVERRIDE_LAYER {
        OVERRIDE_GROUP
      } else {
        GLOBAL_GROUP
      };
      quantize_weight_in_place(&mut weights, &prefix, group);
    }
    quantize_weight_in_place(
      &mut weights,
      &format!("{p}.feed_forward.intermediate_dense"),
      GLOBAL_GROUP,
    );
    quantize_weight_in_place(
      &mut weights,
      &format!("{p}.feed_forward.output_dense"),
      GLOBAL_GROUP,
    );
  }
  quantize_weight_in_place(&mut weights, "lm_head", GLOBAL_GROUP);

  let dir = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_perlayer_override_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(dir.join("config.json"), &config_json).unwrap();
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();

  let loaded = Model::load(&dir.to_string_lossy());
  let _ = std::fs::remove_dir_all(&dir);

  let model = match loaded {
    Ok(m) => m,
    Err(e) => panic!("public load() must apply the HF-keyed per-layer override, got {e:?}"),
  };

  let layers = encoder_layers(&model.encoder);
  let layer0 = &layers[0];
  // The overridden layer 0 `q_proj` must load quantized at the OVERRIDE group
  // size — proving the HF-keyed override reached the sanitized lookup (the bug:
  // it would otherwise have silently fallen back to GLOBAL_GROUP here).
  assert_eq!(
    loaded_group_size(&layer0.attention.q_proj),
    Some(OVERRIDE_GROUP),
    "the HF-keyed per-layer override must apply its group_size to encoder.layers.0.attention.q_proj"
  );
  // Its siblings in the SAME layer carry no override, so they keep the global
  // scheme — confirming the override is scoped to exactly the named layer.
  assert_eq!(
    loaded_group_size(&layer0.attention.k_proj),
    Some(GLOBAL_GROUP),
    "a sibling with no override must keep the global group_size"
  );
  assert_eq!(
    loaded_group_size(&layer0.attention.out_proj),
    Some(GLOBAL_GROUP),
    "a sibling with no override must keep the global group_size"
  );
  // A different layer is likewise untouched by the layer-0 override.
  assert_eq!(
    loaded_group_size(&layers[1].attention.q_proj),
    Some(GLOBAL_GROUP),
    "a different layer must keep the global group_size"
  );
  // The CTC head (top-level `lm_head`, never backbone-prefixed) stays global.
  assert_eq!(
    loaded_group_size(&model.lm_head),
    Some(GLOBAL_GROUP),
    "the top-level lm_head must keep the global group_size"
  );

  // And the model still runs the full forward to finite CTC logits.
  let expected_t = feature_out_len(&config, 400);
  let waveform = filled(&[1, 400], 0.1);
  let mut logits = model
    .forward(&waveform)
    .expect("mixed-quant forward after load must succeed");
  assert_eq!(
    logits.shape(),
    vec![1, expected_t as usize, config.vocab_size as usize],
  );
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite()),
    "all mixed-quant CTC logits must be finite after the public load path"
  );
}

#[test]
fn reproject_quant_keys_strips_backbone_prefix_and_keeps_global() {
  // Each per-layer override key has its backbone prefix (`wav2vec2.` / `hubert.`)
  // stripped to the sanitized lookup namespace; an already-unprefixed key (the
  // top-level `lm_head`) is left as-is; the global default is carried through.
  let global = crate::lm::quant::Quantization::affine(64, QBITS);
  let layer_scheme = crate::lm::quant::Quantization::affine(32, QBITS);
  let mut per_layer = HashMap::new();
  per_layer.insert(
    "wav2vec2.encoder.layers.0.attention.q_proj".to_string(),
    QuantizationOption::Quantize(layer_scheme),
  );
  per_layer.insert(
    "hubert.encoder.layers.1.feed_forward.output_dense".to_string(),
    QuantizationOption::Skip,
  );
  per_layer.insert("lm_head".to_string(), QuantizationOption::Quantize(global));

  let out = reproject_quant_keys(&PerLayerQuantization::new(Some(global), per_layer))
    .expect("a conflict-free per-layer map must reproject");

  // The global default is preserved.
  assert_eq!(out.quantization, Some(global));
  let reprojected = out.per_layer_ref();
  // The backbone prefixes are stripped to the sanitized lookup form.
  assert_eq!(
    reprojected.get("encoder.layers.0.attention.q_proj"),
    Some(&QuantizationOption::Quantize(layer_scheme)),
    "the wav2vec2.-prefixed override must reproject to the sanitized key"
  );
  assert_eq!(
    reprojected.get("encoder.layers.1.feed_forward.output_dense"),
    Some(&QuantizationOption::Skip),
    "the hubert.-prefixed Skip override must reproject to the sanitized key"
  );
  // The already-unprefixed top-level key is unchanged.
  assert_eq!(
    reprojected.get("lm_head"),
    Some(&QuantizationOption::Quantize(global)),
    "an already-sanitized key (lm_head) must pass through unchanged"
  );
  // No backbone-prefixed key survives the reprojection.
  assert!(
    !reprojected
      .keys()
      .any(|k| k.starts_with("wav2vec2.") || k.starts_with("hubert.")),
    "no reprojected key may retain a backbone prefix"
  );
}

#[test]
fn reproject_quant_keys_dedups_identical_collision() {
  // Two source keys (a prefixed + an unprefixed form of the SAME layer) that
  // reproject to the same sanitized key with the IDENTICAL scheme are a benign
  // duplicate — deduplicated to a single entry, not an error.
  let scheme = crate::lm::quant::Quantization::affine(32, QBITS);
  let mut per_layer = HashMap::new();
  per_layer.insert(
    "wav2vec2.encoder.layers.0.attention.q_proj".to_string(),
    QuantizationOption::Quantize(scheme),
  );
  per_layer.insert(
    "encoder.layers.0.attention.q_proj".to_string(),
    QuantizationOption::Quantize(scheme),
  );

  let out = reproject_quant_keys(&PerLayerQuantization::new(None, per_layer))
    .expect("identical reprojected schemes must deduplicate, not error");
  let reprojected = out.per_layer_ref();
  assert_eq!(
    reprojected.len(),
    1,
    "the identical duplicate must collapse to one entry"
  );
  assert_eq!(
    reprojected.get("encoder.layers.0.attention.q_proj"),
    Some(&QuantizationOption::Quantize(scheme)),
  );
}

#[test]
fn reproject_quant_keys_rejects_conflicting_collision() {
  // Two source keys that reproject to the same sanitized key with CONFLICTING
  // schemes are a genuine config contradiction — a typed `KeyCollision`, never a
  // silent arbitrary-survivor overwrite.
  let mut per_layer = HashMap::new();
  per_layer.insert(
    "wav2vec2.encoder.layers.0.attention.q_proj".to_string(),
    QuantizationOption::Quantize(crate::lm::quant::Quantization::affine(32, QBITS)),
  );
  per_layer.insert(
    "encoder.layers.0.attention.q_proj".to_string(),
    QuantizationOption::Quantize(crate::lm::quant::Quantization::affine(64, QBITS)),
  );

  match reproject_quant_keys(&PerLayerQuantization::new(None, per_layer)) {
    Err(Error::KeyCollision(p)) => {
      assert_eq!(
        p.key(),
        "encoder.layers.0.attention.q_proj",
        "the collision error must name the conflicting sanitized key"
      );
    }
    Err(other) => {
      panic!("expected KeyCollision for conflicting reprojected schemes, got {other:?}")
    }
    Ok(_) => panic!("conflicting reprojected schemes must not silently overwrite"),
  }
}

#[test]
fn from_weights_quantized_applies_hf_keyed_per_layer_override_via_public_api() {
  // The finding this guards: a caller using the PUBLIC `from_weights_quantized`
  // constructor directly — `sanitize(raw_weights)` + the parsed
  // `PerLayerQuantization` from `apply_quantization(config_json)` — with a
  // per-layer override keyed in the on-disk HF form (carrying the `wav2vec2.`
  // backbone prefix) must still reach the layer it names. `sanitize` strips that
  // prefix from the weight keys, and the builders resolve a layer's scheme by
  // its SANITIZED prefix, so an override keyed
  // `wav2vec2.encoder.layers.0.attention.q_proj` would never match the
  // `encoder.layers.0.attention.q_proj` lookup and the layer would silently fall
  // back to the GLOBAL scheme — unless `from_weights_quantized` itself
  // normalizes the override keys (the single reprojection boundary, exercised
  // here WITHOUT the `load()` wrapper). With that boundary fix, the override's
  // per-layer `group_size` wins for exactly that one layer while every other
  // quantized layer keeps the global `group_size`.
  const GLOBAL_GROUP: i32 = 64;
  const OVERRIDE_GROUP: i32 = 32;
  // Sanity: the two group sizes differ, so reading back `OVERRIDE_GROUP` on the
  // overridden layer can only mean the override was applied (not the global).
  assert_ne!(GLOBAL_GROUP, OVERRIDE_GROUP);

  // The override is keyed in the HF on-disk form (with the backbone prefix); its
  // `group_size` differs from the global default.
  let config_json = format!(
    r#"{{
      "model_type": "wav2vec2",
      "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 128,
      "num_hidden_layers": 2, "vocab_size": 32,
      "num_feat_extract_layers": 3,
      "conv_dim": [64, 64, 64], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
      "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4,
      "quantization": {{
        "group_size": {GLOBAL_GROUP}, "bits": {QBITS},
        "wav2vec2.encoder.layers.0.attention.q_proj": {{ "group_size": {OVERRIDE_GROUP}, "bits": {QBITS} }}
      }}
    }}"#
  );
  let config = Config::from_json(&config_json).unwrap();

  // On-disk HF-layout weights: start from the DENSE prefixed layout, then pack
  // each quantized-eligible Linear at its scheme's group size — the single
  // overridden layer at the OVERRIDE size, every other eligible Linear at the
  // GLOBAL size — quantizing each layer exactly once from its dense weight so
  // the `.scales` line up with the per-layer scheme the config declares. The
  // conv feature extractor + positional conv stay dense (only `nn.Linear`
  // quantizes).
  let mut raw_weights = hf_layout_prefixed_weights(&config, "wav2vec2.");
  const OVERRIDE_LAYER: &str = "wav2vec2.encoder.layers.0.attention.q_proj";
  quantize_weight_in_place(
    &mut raw_weights,
    "wav2vec2.feature_projection.projection",
    GLOBAL_GROUP,
  );
  for i in 0..(config.num_hidden_layers as usize) {
    let p = format!("wav2vec2.encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      let prefix = format!("{p}.attention.{proj}");
      let group = if prefix == OVERRIDE_LAYER {
        OVERRIDE_GROUP
      } else {
        GLOBAL_GROUP
      };
      quantize_weight_in_place(&mut raw_weights, &prefix, group);
    }
    quantize_weight_in_place(
      &mut raw_weights,
      &format!("{p}.feed_forward.intermediate_dense"),
      GLOBAL_GROUP,
    );
    quantize_weight_in_place(
      &mut raw_weights,
      &format!("{p}.feed_forward.output_dense"),
      GLOBAL_GROUP,
    );
  }
  quantize_weight_in_place(&mut raw_weights, "lm_head", GLOBAL_GROUP);

  // Drive the PUBLIC API exactly as a direct caller would (NOT via `load()`):
  // sanitize the raw HF-keyed weight map ourselves (stripping the backbone
  // prefix off the weight keys), parse the quantization config ourselves (the
  // override is still HF-prefixed), and thread BOTH straight into
  // `from_weights_quantized`.
  let weights = sanitize(raw_weights).expect("the HF-keyed weight map must sanitize");
  let quantization = crate::audio::load::apply_quantization(&config_json)
    .expect("the quantization block must parse")
    .expect("a `quantization` block is present, so the parse is Some");
  // The parsed override is still keyed in the HF form — `from_weights_quantized`
  // is responsible for normalizing it (the exact gap a direct caller hits).
  assert!(
    quantization.per_layer_ref().contains_key(OVERRIDE_LAYER),
    "precondition: the parsed config carries the HF-prefixed override key the public constructor must normalize"
  );

  let model = match Model::from_weights_quantized(
    config.clone(),
    weights,
    Vocab::default(),
    Some(&quantization),
  ) {
    Ok(m) => m,
    Err(e) => panic!(
      "the public from_weights_quantized constructor must apply the HF-keyed per-layer override, got {e:?}"
    ),
  };

  let layers = encoder_layers(&model.encoder);
  let layer0 = &layers[0];
  // The overridden layer 0 `q_proj` must load quantized at the OVERRIDE group
  // size — proving the HF-keyed override reached the sanitized lookup THROUGH
  // THE PUBLIC CONSTRUCTOR (the bug: it would otherwise have silently fallen
  // back to GLOBAL_GROUP here for a direct caller).
  assert_eq!(
    loaded_group_size(&layer0.attention.q_proj),
    Some(OVERRIDE_GROUP),
    "the HF-keyed per-layer override must apply its group_size via the public constructor"
  );
  // Its siblings in the SAME layer carry no override, so they keep the global
  // scheme — confirming the override is scoped to exactly the named layer.
  assert_eq!(
    loaded_group_size(&layer0.attention.k_proj),
    Some(GLOBAL_GROUP),
    "a sibling with no override must keep the global group_size"
  );
  assert_eq!(
    loaded_group_size(&layer0.attention.out_proj),
    Some(GLOBAL_GROUP),
    "a sibling with no override must keep the global group_size"
  );
  // A different layer is likewise untouched by the layer-0 override.
  assert_eq!(
    loaded_group_size(&layers[1].attention.q_proj),
    Some(GLOBAL_GROUP),
    "a different layer must keep the global group_size"
  );
  // The CTC head (top-level `lm_head`, never backbone-prefixed) stays global.
  assert_eq!(
    loaded_group_size(&model.lm_head),
    Some(GLOBAL_GROUP),
    "the top-level lm_head must keep the global group_size"
  );

  // And the model still runs the full forward to finite CTC logits.
  let expected_t = feature_out_len(&config, 400);
  let waveform = filled(&[1, 400], 0.1);
  let mut logits = model
    .forward(&waveform)
    .expect("mixed-quant forward via the public constructor must succeed");
  assert_eq!(
    logits.shape(),
    vec![1, expected_t as usize, config.vocab_size as usize],
  );
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite()),
    "all mixed-quant CTC logits must be finite via the public constructor"
  );
}

// ───────────────────── Family / Transcribe wiring ─────────────────────

/// A `vocab.json` body covering every id `0..vocab_size` of [`tiny_config_json`]
/// (12 tokens): id `0` is the CTC blank `<pad>` (collapsed out), the rest are
/// single characters including the `|` word-delimiter at id `9`. With every
/// class mapped, whatever the tiny model's per-frame argmax selects decodes to a
/// known character — so the boxed and inherent transcription paths render real
/// (non-empty) text, not the empty-default-`Vocab` placeholder.
fn full_tiny_vocab() -> Vocab {
  Vocab::from_json(
    r#"{
      "<pad>": 0, "A": 1, "B": 2, "C": 3, "D": 4,
      "E": 5, "F": 6, "G": 7, "H": 8, "|": 9, "I": 10, "J": 11
    }"#,
  )
  .expect("a well-formed 12-token vocab must parse")
}

#[test]
fn model_standard_is_usable_as_dyn_transcribe() {
  // The golden CTC trait wiring: `Model<Standard>` implements `CtcModel` (so the
  // blanket-style `Transcribe` delegation to `greedy_ctc_transcribe` applies),
  // making a loaded model usable through the object-safe `Box<dyn Transcribe>`
  // seam every STT pipeline depends on. Build a tiny model with a real (non-empty)
  // vocab, erase it behind the trait object, and drive a real transcription over a
  // mono waveform — the dynamic dispatch must run the shared normalize → forward →
  // greedy-collapse → decode path to a single-segment `Transcription`.
  use crate::audio::stt::model::{Transcribe, TranscribeOptions};

  let config = tiny_config_json("");
  let model = Model::from_weights(
    config,
    synthetic_weights(&tiny_config_json("")),
    full_tiny_vocab(),
  )
  .expect("a valid Standard config must build");

  // Erase the concrete dialect at the trait-object boundary (the one dyn point).
  let erased: Box<dyn Transcribe> = Box::new(model);
  // A mono (T,) waveform — `greedy_ctc_transcribe` validates a non-empty mono
  // input, then reads the model's `(T', vocab)` logits.
  let waveform = filled(&[400], 0.1);
  let transcription = erased
    .transcribe(&waveform, &TranscribeOptions::new())
    .expect("transcription through Box<dyn Transcribe> must succeed");
  // CTC emits a single segment spanning the whole utterance; the language is
  // unreported (CTC has no language conditioning).
  assert_eq!(
    transcription.segments_slice().len(),
    1,
    "a CTC transcription carries exactly one segment"
  );
  assert_eq!(transcription.language(), None);
  // The single segment's text equals the top-level text (both run through the
  // shared trimmed decode seam) — the reference sets `text=` and the segment to
  // the same `text.strip()`.
  assert_eq!(
    transcription.segments_slice()[0].text(),
    transcription.text(),
    "the CTC segment text and the top-level text are the one trimmed decode"
  );
}

#[test]
fn boxed_transcribe_rejects_empty_vocabulary() {
  // A model loaded without `vocab.json` carries the empty default `Vocab`. The
  // forward is well-defined, but there is no id → token map, so transcription
  // must be rejected — NOT silently succeed with empty text. The boxed
  // `Box<dyn Transcribe>` path (which delegates to the infallible greedy driver)
  // must return the SAME typed error the inherent `Model::transcribe` raises, via
  // the one shared `ensure_decodable` guard.
  use crate::audio::stt::model::{Transcribe, TranscribeOptions};

  let config = tiny_config_json("");
  let model = Model::from_weights(
    config,
    synthetic_weights(&tiny_config_json("")),
    Vocab::default(),
  )
  .expect("a valid Standard config must build");
  let erased: Box<dyn Transcribe> = Box::new(model);
  let waveform = filled(&[400], 0.1);

  match erased.transcribe(&waveform, &TranscribeOptions::new()) {
    Err(Error::InvariantViolation(p)) => {
      assert_eq!(p.context(), "Model::transcribe");
      assert_eq!(
        p.requirement(),
        "model was built without a vocabulary (use forward + ctc_greedy_collapse)"
      );
    }
    other => {
      panic!("empty-vocab boxed transcribe must reject with InvariantViolation, got {other:?}")
    }
  }

  // The inherent path raises the identical error (the shared guard) — the two
  // are byte-identical, not merely both-failing.
  let inherent = Model::from_weights(
    tiny_config_json(""),
    synthetic_weights(&tiny_config_json("")),
    Vocab::default(),
  )
  .expect("a valid Standard config must build");
  match inherent.transcribe(&waveform) {
    Err(Error::InvariantViolation(p)) => {
      assert_eq!(p.context(), "Model::transcribe");
      assert_eq!(
        p.requirement(),
        "model was built without a vocabulary (use forward + ctc_greedy_collapse)"
      );
    }
    other => {
      panic!("empty-vocab inherent transcribe must reject with InvariantViolation, got {other:?}")
    }
  }
}

#[test]
fn decode_ids_trims_pipe_mapped_edge_spaces() {
  // The shared decode seam (`CtcModel::decode_ids`) every transcription path runs:
  // it maps ids → text (the `|` word-delimiter → a space) and then trims the
  // leading / trailing whitespace the delimiter mapping leaves at the utterance
  // edges — the reference's `"".join(...).replace("|", " ").strip()`. A collapsed
  // id stream that begins and ends with the `|` delimiter (id 9) must decode to
  // the interior word with NO surrounding spaces.
  use crate::audio::stt::model::CtcModel;

  let model = Model::from_weights(
    tiny_config_json(""),
    synthetic_weights(&tiny_config_json("")),
    full_tiny_vocab(),
  )
  .expect("a valid Standard config must build");

  // ids: | H I | → raw "".join = " HI ", .replace already done, .strip() → "HI".
  let decoded = model.decode_ids(&[9, 8, 10, 9]);
  assert_eq!(
    decoded, "HI",
    "leading/trailing |-mapped spaces must be trimmed"
  );

  // An interior `|` is a real word break and is preserved; only the edges trim.
  // ids: | A | B | → " A B " → strip → "A B".
  let two_words = model.decode_ids(&[9, 1, 9, 2, 9]);
  assert_eq!(
    two_words, "A B",
    "interior |-mapped spaces are word breaks and stay; only the edges trim"
  );

  // The empty collapse (the all-blank / empty-time CTC case) decodes to "" — the
  // driver routes both through `decode_ids(&[])`, so the trim makes it empty.
  assert_eq!(model.decode_ids(&[]), "");
}

#[test]
fn boxed_and_inherent_transcribe_yield_identical_text() {
  // The boxed `Box<dyn Transcribe>` path and the inherent `Model::transcribe`
  // must produce BYTE-IDENTICAL text on the same model + waveform. Both reach the
  // same `(T', vocab)` logits (one shared normalize → forward), collapse
  // identically (the local `ctc_greedy_collapse` and the driver's inline collapse
  // are the same algorithm), and render through the one shared `decode_ids` seam
  // (the `|`→space mapping + edge trim), so the two decodes never diverge.
  use crate::audio::stt::model::{Transcribe, TranscribeOptions};

  let waveform = ramp(&[512], 0.5);

  let inherent_model = Model::from_weights(
    tiny_config_json(""),
    synthetic_weights(&tiny_config_json("")),
    full_tiny_vocab(),
  )
  .expect("a valid Standard config must build");
  let inherent_text = inherent_model
    .transcribe(&waveform)
    .expect("inherent transcribe must succeed with a real vocab");

  // A separately-built but identical model, erased behind the trait object.
  let boxed_model = Model::from_weights(
    tiny_config_json(""),
    synthetic_weights(&tiny_config_json("")),
    full_tiny_vocab(),
  )
  .expect("a valid Standard config must build");
  let erased: Box<dyn Transcribe> = Box::new(boxed_model);
  let boxed = erased
    .transcribe(&waveform, &TranscribeOptions::new())
    .expect("boxed transcribe must succeed with a real vocab");

  assert_eq!(
    boxed.text(),
    inherent_text,
    "the boxed Transcribe path and the inherent path must decode identically"
  );
  // And the segment text matches too (the segment carries the same trimmed text).
  assert_eq!(boxed.segments_slice()[0].text(), inherent_text);
}

#[test]
fn direct_greedy_ctc_transcribe_rejects_empty_vocabulary() {
  // The third public text-producing route: a caller can invoke the shared CTC
  // driver `greedy_ctc_transcribe(&model, …)` DIRECTLY (the `CtcModel` path),
  // bypassing the boxed `Box<dyn Transcribe>` wrapper entirely. A model loaded
  // without a `vocab.json` carries the empty default `Vocab`: the forward is
  // well-defined, but there is no id → token map, so this route would otherwise
  // silently succeed with empty text. The driver now calls
  // `CtcModel::ensure_decodable` at its single chokepoint, so this DIRECT route
  // must reject the empty vocabulary with the SAME typed `InvariantViolation`
  // the boxed and inherent paths raise — closing the empty-vocab class at the
  // one seam every route funnels through.
  use crate::audio::stt::model::TranscribeOptions;

  let model = Model::from_weights(
    tiny_config_json(""),
    synthetic_weights(&tiny_config_json("")),
    Vocab::default(),
  )
  .expect("a valid Standard config must build");
  let waveform = filled(&[400], 0.1);

  // Reach the CtcModel driver directly — NOT via `Box<dyn Transcribe>`.
  match greedy_ctc_transcribe(&model, &waveform, &TranscribeOptions::new()) {
    Err(Error::InvariantViolation(p)) => {
      assert_eq!(p.context(), "Model::transcribe");
      assert_eq!(
        p.requirement(),
        "model was built without a vocabulary (use forward + ctc_greedy_collapse)"
      );
    }
    other => panic!(
      "a direct greedy_ctc_transcribe on an empty-vocab model must reject with \
         InvariantViolation (not empty-string success), got {other:?}"
    ),
  }
}

#[test]
fn direct_greedy_ctc_transcribe_succeeds_with_real_vocabulary() {
  // The companion to the empty-vocab rejection: the SAME direct `CtcModel`
  // driver route, on a model carrying a real (non-empty) vocab, transcribes
  // normally — the `ensure_decodable` chokepoint guard does not regress the
  // happy path. CTC emits a single segment spanning the whole utterance, with
  // no language reported, and the segment text equals the top-level text (both
  // run through the one shared trimmed `decode_ids` seam).
  use crate::audio::stt::model::TranscribeOptions;

  let model = Model::from_weights(
    tiny_config_json(""),
    synthetic_weights(&tiny_config_json("")),
    full_tiny_vocab(),
  )
  .expect("a valid Standard config must build");
  let waveform = filled(&[400], 0.1);

  let transcription = greedy_ctc_transcribe(&model, &waveform, &TranscribeOptions::new())
    .expect("a direct greedy_ctc_transcribe with a real vocab must succeed");
  assert_eq!(
    transcription.segments_slice().len(),
    1,
    "a CTC transcription carries exactly one segment"
  );
  assert_eq!(transcription.language(), None);
  assert_eq!(
    transcription.segments_slice()[0].text(),
    transcription.text(),
    "the CTC segment text and the top-level text are the one trimmed decode"
  );
}

// ═════════════ feat_extract_norm == "layer" feature extractor ═════════════

/// A complete post-sanitize weight map for a `feat_extract_norm == "layer"`
/// config — like [`synthetic_weights`] but EVERY conv layer carries a
/// `layer_norm.{weight,bias}` affine over its conv output width (the
/// `Wav2Vec2LayerNormConvLayer` extractor), instead of only L0 carrying one (the
/// GroupNorm extractor). Shapes written longhand.
fn layer_norm_extractor_weights(c: &Config) -> HashMap<String, Array> {
  let mut w = synthetic_weights(c);
  // synthetic_weights only inserts conv_layers.0.layer_norm.*; the "layer" arm
  // needs one per conv layer (each over conv_dim[i]). Insert the missing
  // per-layer LayerNorm affines (L0's is already present from synthetic_weights,
  // sized conv_dim[0], which is correct here too).
  for i in 0..(c.num_feat_extract_layers as usize) {
    let out = c.conv_dim[i];
    w.insert(
      format!("feature_extractor.conv_layers.{i}.layer_norm.weight"),
      filled(&[out], 1.0),
    );
    w.insert(
      format!("feature_extractor.conv_layers.{i}.layer_norm.bias"),
      filled(&[out], 0.0),
    );
  }
  w
}

#[test]
fn layer_feat_extract_norm_builds_and_forwards() {
  // A `feat_extract_norm = "layer"` config (the large-960h-lv60-self extractor)
  // must build the all-LayerNorm feature encoder and forward to the right logits
  // shape — the large-960h-lv60-self extractor.
  let config = tiny_config_json(r#", "feat_extract_norm": "layer""#);
  assert_eq!(
    config.feat_extract_norm_scheme().unwrap(),
    FeatExtractNorm::Layer
  );
  let expected_t = feature_out_len(&config, 400);
  let vocab = config.vocab_size as usize;
  let weights = layer_norm_extractor_weights(&config);
  let model = Model::from_weights(config, weights, Vocab::default())
    .expect("a feat_extract_norm=layer config must build");
  // Every conv layer must carry a LayerNorm (and none a GroupNorm) — the
  // structural proof the "layer" arm was built, not the group arm.
  for (i, layer) in model.feature_encoder.conv_layers.iter().enumerate() {
    assert!(
      layer.layer_norm.is_some(),
      "conv layer {i} must carry a LayerNorm in the \"layer\" extractor"
    );
    assert!(
      layer.group_norm.is_none(),
      "conv layer {i} must NOT carry a GroupNorm in the \"layer\" extractor"
    );
  }
  let waveform = filled(&[1, 400], 0.1);
  let mut logits = model.forward(&waveform).expect("forward must succeed");
  assert_eq!(
    logits.shape(),
    vec![1, expected_t as usize, vocab],
    "layer-norm-extractor logits must be (1, T'={expected_t}, vocab={vocab})"
  );
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite()),
    "all logits must be finite"
  );
}

#[test]
fn group_arm_has_only_l0_groupnorm() {
  // The both-directions counterpart: the default "group" extractor must have a
  // GroupNorm at L0 ONLY, and no LayerNorm anywhere — distinguishing it from the
  // "layer" arm built above.
  let config = tiny_config_json("");
  assert_eq!(
    config.feat_extract_norm_scheme().unwrap(),
    FeatExtractNorm::Group
  );
  let model = Model::from_weights(
    config,
    synthetic_weights(&tiny_config_json("")),
    Vocab::default(),
  )
  .expect("the default group config must build");
  for (i, layer) in model.feature_encoder.conv_layers.iter().enumerate() {
    assert_eq!(
      layer.group_norm.is_some(),
      i == 0,
      "conv layer {i}: GroupNorm present iff it is L0 (the \"group\" extractor)"
    );
    assert!(
      layer.layer_norm.is_none(),
      "conv layer {i}: no LayerNorm in the \"group\" extractor"
    );
  }
}

#[test]
fn layer_and_group_feature_extractors_differ() {
  // The "layer" and "group" extractors must produce DIFFERENT features (hence
  // logits) from the same conv weights + input — proving the LayerNorm arm is a
  // distinct graph (per-layer LayerNorm at every layer vs a single L0 GroupNorm),
  // not an accidental alias.
  let group_cfg = tiny_config_json("");
  let layer_cfg = tiny_config_json(r#", "feat_extract_norm": "layer""#);
  // Share the SAME conv weights (synthetic_weights is deterministic for a given
  // config and the two configs share every conv dim), so the only difference is
  // the normalization scheme.
  let group_w = synthetic_weights(&tiny_config_json(""));
  let layer_w =
    layer_norm_extractor_weights(&tiny_config_json(r#", "feat_extract_norm": "layer""#));
  let waveform = filled(&[1, 400], 0.3);
  let mut g = Model::from_weights(group_cfg, group_w, Vocab::default())
    .unwrap()
    .forward(&waveform)
    .unwrap();
  let mut l = Model::from_weights(layer_cfg, layer_w, Vocab::default())
    .unwrap()
    .forward(&waveform)
    .unwrap();
  assert_eq!(g.shape(), l.shape(), "same dims, same logits shape");
  let ga = g.to_vec::<f32>().unwrap();
  let la = l.to_vec::<f32>().unwrap();
  assert!(
    ga.iter().zip(la.iter()).any(|(x, y)| (x - y).abs() > 1e-5),
    "the group and layer feature-extractor norm arms must produce different outputs"
  );
}

#[test]
fn layer_feat_extract_norm_requires_per_layer_layernorm() {
  // The "layer" extractor consumes a `layer_norm.{weight,bias}` for EVERY conv
  // layer (not just L0). A checkpoint missing a non-L0 layer's LayerNorm is a
  // clear MissingKey — the both-directions proof the per-layer norms are
  // actually consumed.
  let config = tiny_config_json(r#", "feat_extract_norm": "layer""#);
  let mut weights = layer_norm_extractor_weights(&config);
  // Drop layer 1's LayerNorm weight (present in the "layer" arm, absent in
  // "group").
  weights.remove("feature_extractor.conv_layers.1.layer_norm.weight");
  match Model::from_weights(config, weights, Vocab::default()) {
    Err(Error::MissingKey(p)) => assert!(
      p.key().contains("conv_layers.1.layer_norm.weight"),
      "the missing key should name the absent per-layer LayerNorm, got {:?}",
      p.key()
    ),
    // `Model` is not `Debug`, so the Ok arm is reported without formatting it.
    Err(other) => panic!("expected MissingKey for an absent per-layer LayerNorm, got {other:?}"),
    Ok(_) => panic!("expected MissingKey for an absent per-layer LayerNorm, but the model built"),
  }
}

// ═════════════ feat_proj_layer_norm == false (HuBERT no-LN projection) ═════════════

#[test]
fn feat_proj_no_layernorm_builds_and_skips_projection_ln() {
  // A `feat_proj_layer_norm = false` config (HuBERT's no-LayerNorm projection
  // arm) must build with NO projection LayerNorm and forward correctly — the
  // HuBERT no-LayerNorm feature-projection arm.
  let config = tiny_config_json(r#", "model_type": "hubert", "feat_proj_layer_norm": false"#);
  assert!(!config.feat_proj_layer_norm);
  // The synthetic weight map carries the projection LayerNorm affine; the
  // builder must NOT consume it when the flag is false (so the model still
  // builds — the unused tensor is simply left in the map).
  let weights = synthetic_weights(&config);
  let model = Model::from_weights(config, weights, Vocab::default())
    .expect("a feat_proj_layer_norm=false config must build");
  assert!(
    model.feature_projection.layer_norm.is_none(),
    "the projection LayerNorm must be absent when feat_proj_layer_norm = false"
  );
  let waveform = filled(&[1, 400], 0.1);
  let mut logits = model.forward(&waveform).expect("forward must succeed");
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite()),
    "all logits must be finite for the no-LayerNorm projection arm"
  );
}

#[test]
fn feat_proj_layernorm_present_when_flag_true() {
  // The both-directions counterpart: the default (feat_proj_layer_norm = true)
  // builds WITH the projection LayerNorm — so the flag genuinely gates it.
  let config = tiny_config_json("");
  assert!(config.feat_proj_layer_norm);
  let model = Model::from_weights(
    config,
    synthetic_weights(&tiny_config_json("")),
    Vocab::default(),
  )
  .expect("the default config must build");
  assert!(
    model.feature_projection.layer_norm.is_some(),
    "the projection LayerNorm must be present when feat_proj_layer_norm = true"
  );
}

#[test]
fn feat_proj_with_and_without_layernorm_differ() {
  // The LayerNorm and no-LayerNorm projection arms must produce DIFFERENT logits
  // from the same weights + input — proving the `false` arm actually skips the
  // normalization (an un-normalized projection input vs a normalized one).
  let with_ln = tiny_config_json(r#", "model_type": "hubert""#); // default true
  let no_ln = tiny_config_json(r#", "model_type": "hubert", "feat_proj_layer_norm": false"#);
  assert!(with_ln.feat_proj_layer_norm);
  assert!(!no_ln.feat_proj_layer_norm);
  // Use a NON-trivial projection LayerNorm affine so dropping it actually
  // changes the result (the default synthetic affine is ones/zeros, which on a
  // feature-uniform input could be near-identity; ramp the input so the
  // pre-projection features vary across the channel axis).
  let waveform = ramp(&[1, 400], 0.5);
  let mut a = Model::from_weights(
    with_ln,
    synthetic_weights(&tiny_config_json(r#", "model_type": "hubert""#)),
    Vocab::default(),
  )
  .unwrap()
  .forward(&waveform)
  .unwrap();
  let mut b = Model::from_weights(
    no_ln,
    synthetic_weights(&tiny_config_json(
      r#", "model_type": "hubert", "feat_proj_layer_norm": false"#,
    )),
    Vocab::default(),
  )
  .unwrap()
  .forward(&waveform)
  .unwrap();
  assert_eq!(a.shape(), b.shape());
  let av = a.to_vec::<f32>().unwrap();
  let bv = b.to_vec::<f32>().unwrap();
  assert!(
    av.iter().zip(bv.iter()).any(|(x, y)| (x - y).abs() > 1e-5),
    "the LayerNorm and no-LayerNorm projection arms must produce different outputs"
  );
}

#[test]
fn feat_proj_no_layernorm_is_hubert_only_on_build_path() {
  // The no-LayerNorm feature projection is HuBERT-only — HF's
  // `Wav2Vec2FeatureProjection` ALWAYS applies the projection LayerNorm. So a
  // wav2vec2 `model_type` with `feat_proj_layer_norm = false` must be REJECTED on
  // the build path (Model::from_weights runs Config::validate), never load a
  // silently-wrong graph (a wav2vec2 model with its projection LayerNorm dropped).

  // (a) wav2vec2 (default model_type) + false → rejected by the build path.
  let w2v2_no_ln = tiny_config_json(r#", "feat_proj_layer_norm": false"#);
  assert!(!w2v2_no_ln.feat_proj_layer_norm);
  assert!(!w2v2_no_ln.is_hubert());
  let weights = synthetic_weights(&w2v2_no_ln);
  match Model::from_weights(w2v2_no_ln, weights, Vocab::default()) {
    Err(Error::InvariantViolation(_)) => {}
    Ok(_) => panic!(
      "a wav2vec2 model_type with feat_proj_layer_norm=false must be REJECTED on the build path \
       (the no-LayerNorm projection is HuBERT-only)"
    ),
    Err(e) => panic!("expected a typed InvariantViolation, got {e:?}"),
  }

  // (b) An explicit wav2vec2 model_type + false → likewise rejected.
  let w2v2_explicit =
    tiny_config_json(r#", "model_type": "wav2vec2", "feat_proj_layer_norm": false"#);
  let weights = synthetic_weights(&w2v2_explicit);
  assert!(Model::from_weights(w2v2_explicit, weights, Vocab::default()).is_err());

  // (c) wav2vec2 with the flag absent (the common case) builds WITH the
  // projection LayerNorm — the wav2vec2 graph always applies it.
  let w2v2_default = tiny_config_json("");
  assert!(w2v2_default.feat_proj_layer_norm);
  let weights = synthetic_weights(&w2v2_default);
  let model = Model::from_weights(w2v2_default, weights, Vocab::default())
    .expect("a default wav2vec2 config must build");
  assert!(
    model.feature_projection.layer_norm.is_some(),
    "the wav2vec2 projection LayerNorm must always be present"
  );

  // (d) hubert + false is the no-LayerNorm arm and is honored (builds, no LN) —
  // proving the gate is on model_type, not a blanket rejection of false.
  let hubert_no_ln = tiny_config_json(r#", "model_type": "hubert", "feat_proj_layer_norm": false"#);
  assert!(hubert_no_ln.is_hubert());
  let weights = synthetic_weights(&hubert_no_ln);
  let hubert_model = Model::from_weights(hubert_no_ln, weights, Vocab::default()).expect(
    "a hubert model_type with feat_proj_layer_norm=false must build (its no-LayerNorm arm)",
  );
  assert!(
    hubert_model.feature_projection.layer_norm.is_none(),
    "the hubert no-LayerNorm arm must drop the projection LayerNorm"
  );
}

// ═══════════════════ MMS attention adapter ═══════════════════

/// A tiny MMS config: the tiny dims + the stable-LN arm (which MMS uses) + a
/// small `adapter_attn_dim` bottleneck. The `extra` is appended to the JSON.
fn mms_config_json(extra: &str) -> Config {
  tiny_config_json(&format!(
    r#", "do_stable_layer_norm": true, "adapter_attn_dim": 8{extra}"#
  ))
}

/// Add the per-layer MMS adapter weights (`encoder.layers.{i}.adapter_layer.*`)
/// to a base synthetic weight map for `c`: a `norm` LayerNorm affine over
/// hidden, a `linear_1` (hidden → adapter_attn_dim) + a `linear_2`
/// (adapter_attn_dim → hidden), each with a bias. Used for the build/forward
/// tests; `value` scales the adapter Linear ramps (0.0 ⇒ a zero adapter, so the
/// adapter add is a no-op — handy for the "adapter present but inert" check).
fn add_adapter_weights(w: &mut HashMap<String, Array>, c: &Config, value: f32) {
  let hs = c.hidden_size;
  let d = c
    .adapter_attn_dim
    .expect("an MMS config sets adapter_attn_dim");
  for i in 0..(c.num_hidden_layers as usize) {
    let p = format!("encoder.layers.{i}.adapter_layer");
    w.insert(format!("{p}.norm.weight"), filled(&[hs], 1.0));
    w.insert(format!("{p}.norm.bias"), filled(&[hs], 0.0));
    w.insert(format!("{p}.linear_1.weight"), ramp(&[d, hs], value));
    w.insert(format!("{p}.linear_1.bias"), filled(&[d], 0.0));
    w.insert(format!("{p}.linear_2.weight"), ramp(&[hs, d], value));
    w.insert(format!("{p}.linear_2.bias"), filled(&[hs], 0.0));
  }
}

/// A complete MMS synthetic weight map (the base synthetic weights + the
/// per-layer adapter weights).
fn mms_weights(c: &Config, adapter_value: f32) -> HashMap<String, Array> {
  let mut w = synthetic_weights(c);
  add_adapter_weights(&mut w, c, adapter_value);
  w
}

#[test]
fn attn_adapter_layer_forward_oracle() {
  // VALUE-LEVEL oracle for the adapter branch: norm → linear_1 → relu →
  // linear_2 (Wav2Vec2AttnAdapterLayer.__call__). The oracle recomputes each
  // step through the PUBLIC primitives (LayerNorm, the Linear forward, the relu
  // = max(x,0)), independently of AttnAdapterLayer::forward, then compares.
  use crate::lm::nn::norm::LayerNorm as Ln;

  let hidden = 4i32;
  let adapter_dim = 3i32;
  // Deterministic, element-varying weights/input so relu actually clips some
  // values (a constant fill would not exercise the nonlinearity).
  let norm_w = filled(&[hidden], 1.0);
  let norm_b = filled(&[hidden], 0.0);
  let l1_w = ramp(&[adapter_dim, hidden], 0.7);
  let l1_b = ramp(&[adapter_dim], 0.5);
  let l2_w = ramp(&[hidden, adapter_dim], 0.7);
  let l2_b = ramp(&[hidden], 0.3);
  let x = ramp(&[1, 5, hidden], 0.9);

  let adapter = AttnAdapterLayer {
    norm: Ln::new(
      Some(norm_w.try_clone().unwrap()),
      Some(norm_b.try_clone().unwrap()),
      1e-5,
    ),
    linear_1: Linear::new(l1_w.try_clone().unwrap(), Some(l1_b.try_clone().unwrap())),
    linear_2: Linear::new(l2_w.try_clone().unwrap(), Some(l2_b.try_clone().unwrap())),
  };
  let mut got = adapter.forward(&x).unwrap();

  // Oracle: norm(x) → linear_1 → relu → linear_2, all via public ops.
  let oracle_norm = Ln::new(Some(norm_w), Some(norm_b), 1e-5);
  let n = oracle_norm.forward(&x).unwrap();
  let h1 = Linear::new(l1_w, Some(l1_b)).forward(&n).unwrap();
  // relu = max(h1, 0).
  let zero = Array::full::<f32>(&[0i32; 0], 0.0).unwrap();
  let r = ops::arithmetic::maximum(&h1, &zero).unwrap();
  let mut want = Linear::new(l2_w, Some(l2_b)).forward(&r).unwrap();

  assert_eq!(
    got.shape(),
    want.shape(),
    "adapter output shape (B, T, hidden)"
  );
  assert_eq!(got.shape(), vec![1, 5, hidden as usize]);
  let g = got.to_vec::<f32>().unwrap();
  let wv = want.to_vec::<f32>().unwrap();
  for (i, (a, b)) in g.iter().zip(wv.iter()).enumerate() {
    assert!(
      (a - b).abs() < 1e-5,
      "adapter[{i}]: got {a}, want {b} (norm→linear_1→relu→linear_2)"
    );
  }
}

#[test]
fn mms_config_builds_adapter_in_every_stable_ln_layer() {
  // An MMS config (stable-LN + adapter_attn_dim) must build an adapter in EVERY
  // encoder layer and forward to the right shape — the structural proof the
  // per-layer adapter is wired.
  let config = mms_config_json("");
  assert_eq!(config.adapter_attn_dim, Some(8));
  assert!(config.do_stable_layer_norm);
  let weights = mms_weights(&config, 0.3);
  let model = Model::from_weights(config, weights, Vocab::default())
    .expect("an MMS (stable-LN + adapter_attn_dim) config must build");
  let layers = encoder_layers(&model.encoder);
  assert!(!layers.is_empty());
  for (i, layer) in layers.iter().enumerate() {
    assert!(
      layer.adapter_layer.is_some(),
      "stable-LN layer {i} must carry the MMS adapter when adapter_attn_dim is set"
    );
  }
  let waveform = filled(&[1, 400], 0.1);
  let mut logits = model.forward(&waveform).expect("MMS forward must succeed");
  let expected_t = feature_out_len(&mms_config_json(""), 400);
  assert_eq!(
    logits.shape(),
    vec![
      1,
      expected_t as usize,
      mms_config_json("").vocab_size as usize
    ]
  );
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite())
  );
}

#[test]
fn post_norm_config_with_adapter_attn_dim_builds_no_adapter() {
  // The reference attaches the adapter ONLY to the stable-LN layer
  // (Wav2Vec2EncoderLayerStableLayerNorm). A POST-norm config that nonetheless
  // carries adapter_attn_dim must build NO adapter (and consume no
  // adapter_layer.* weights) — faithful to the reference's post-norm
  // Wav2Vec2EncoderLayer, which has no adapter.
  let config = tiny_config_json(r#", "adapter_attn_dim": 8"#); // post-norm (default)
  assert_eq!(config.adapter_attn_dim, Some(8));
  assert!(!config.do_stable_layer_norm);
  // The base synthetic weights carry NO adapter_layer.* keys; if the post-norm
  // builder tried to consume them it would MissingKey. It must build cleanly.
  let model = Model::from_weights(
    config,
    synthetic_weights(&tiny_config_json(r#", "adapter_attn_dim": 8"#)),
    Vocab::default(),
  )
  .expect("a post-norm config with adapter_attn_dim must build with no adapter");
  for (i, layer) in encoder_layers(&model.encoder).iter().enumerate() {
    assert!(
      layer.adapter_layer.is_none(),
      "post-norm layer {i} must have NO adapter (the reference attaches it only to stable-LN)"
    );
  }
}

#[test]
fn mms_adapter_changes_output_vs_no_adapter() {
  // A non-zero MMS adapter must CHANGE the logits vs the same stable-LN model
  // with a zero adapter (whose `h + adapter(h)` add is a no-op) — proving the
  // adapter branch is actually summed into the hidden states, not dropped.
  let cfg_active = mms_config_json("");
  let cfg_zero = mms_config_json("");
  let waveform = ramp(&[1, 400], 0.4);
  // Active adapter (non-zero Linear ramps) vs an all-zero adapter (linear_1 /
  // linear_2 weights = 0, biases = 0 ⇒ adapter(h) = 0 ⇒ h + 0 = h).
  let mut active = Model::from_weights(
    cfg_active,
    mms_weights(&mms_config_json(""), 0.5),
    Vocab::default(),
  )
  .unwrap()
  .forward(&waveform)
  .unwrap();
  let mut zero = Model::from_weights(
    cfg_zero,
    mms_weights(&mms_config_json(""), 0.0),
    Vocab::default(),
  )
  .unwrap()
  .forward(&waveform)
  .unwrap();
  assert_eq!(active.shape(), zero.shape());
  let a = active.to_vec::<f32>().unwrap();
  let z = zero.to_vec::<f32>().unwrap();
  assert!(
    a.iter().zip(z.iter()).any(|(x, y)| (x - y).abs() > 1e-5),
    "a non-zero MMS adapter must change the output (its branch is added to the hidden states)"
  );
}

#[test]
fn mms_config_requires_adapter_weights() {
  // With adapter_attn_dim set on a stable-LN config, the builder consumes the
  // `adapter_layer.*` weights per layer; a checkpoint missing one is a clear
  // MissingKey (not a silent drop of the adapter graph term).
  let config = mms_config_json("");
  let mut weights = mms_weights(&config, 0.3);
  weights.remove("encoder.layers.0.adapter_layer.linear_1.weight");
  match Model::from_weights(config, weights, Vocab::default()) {
    Err(Error::MissingKey(p)) => assert!(
      p.key().contains("layers.0.adapter_layer.linear_1.weight"),
      "the missing key should name the absent adapter weight, got {:?}",
      p.key()
    ),
    // `Model` is not `Debug`, so the Ok arm is reported without formatting it.
    Err(other) => panic!("expected MissingKey for an absent adapter weight, got {other:?}"),
    Ok(_) => panic!("expected MissingKey for an absent adapter weight, but the model built"),
  }
}

// ─────────────── MMS per-language adapter loading ───────────────

/// The dense lm_head weight of a loaded model (for asserting the adapter file's
/// lm_head replaced the base one). Panics if the lm_head is quantized (these
/// fixtures are dense).
fn lm_head_weight(model: &Model<Standard>) -> Vec<f32> {
  match &model.lm_head.inner {
    crate::nn::MaybeQuantizedLinear::Dense(l) => {
      l.weight_ref().try_clone().unwrap().to_vec::<f32>().unwrap()
    }
    crate::nn::MaybeQuantizedLinear::Quantized(_) => panic!("dense lm_head expected"),
  }
}

/// A per-language MMS adapter safetensors map (the on-disk
/// `adapter.{lang}.safetensors` content): the per-layer `adapter_layer.*`
/// weights + the per-language `lm_head.*`, all at scale `value` so different
/// languages produce distinguishable weights. Keys are backbone-relative (no
/// `wav2vec2.` prefix — real MMS adapter files store them this way); `lm_head`
/// is top-level. No conv tensors (sanitize leaves these unchanged).
fn adapter_file_weights(c: &Config, value: f32) -> HashMap<String, Array> {
  let mut w: HashMap<String, Array> = HashMap::new();
  let hs = c.hidden_size;
  let d = c
    .adapter_attn_dim
    .expect("MMS config sets adapter_attn_dim");
  for i in 0..(c.num_hidden_layers as usize) {
    let p = format!("encoder.layers.{i}.adapter_layer");
    w.insert(format!("{p}.norm.weight"), filled(&[hs], 1.0));
    w.insert(format!("{p}.norm.bias"), filled(&[hs], 0.0));
    w.insert(format!("{p}.linear_1.weight"), ramp(&[d, hs], value));
    w.insert(format!("{p}.linear_1.bias"), filled(&[d], 0.0));
    w.insert(format!("{p}.linear_2.weight"), ramp(&[hs, d], value));
    w.insert(format!("{p}.linear_2.bias"), filled(&[hs], 0.0));
  }
  // The per-language CTC head (a DISTINCT ramp so it is distinguishable from the
  // base init's lm_head).
  w.insert(
    "lm_head.weight".to_string(),
    ramp(&[c.vocab_size, hs], value),
  );
  w.insert("lm_head.bias".to_string(), filled(&[c.vocab_size], value));
  w
}

/// Write an MMS checkpoint to a fresh temp dir: the base `model.safetensors`
/// (HF `wav2vec2.`-prefixed, with a language-agnostic adapter init scaled by
/// `base_value`) plus an `adapter.{lang}.safetensors` per `(lang, value)` pair.
/// Returns the dir; the caller removes it.
fn write_mms_checkpoint(
  config_json: &str,
  config: &Config,
  base_value: f32,
  adapters: &[(&str, f32)],
  tag: &str,
) -> std::path::PathBuf {
  // Base HF-prefixed weights + the language-agnostic adapter init (also
  // prefixed). hf_layout_prefixed_weights covers the backbone; add the adapter
  // init under the same prefix.
  let mut base = hf_layout_prefixed_weights(config, "wav2vec2.");
  let hs = config.hidden_size;
  let d = config.adapter_attn_dim.unwrap();
  for i in 0..(config.num_hidden_layers as usize) {
    let p = format!("wav2vec2.encoder.layers.{i}.adapter_layer");
    base.insert(format!("{p}.norm.weight"), filled(&[hs], 1.0));
    base.insert(format!("{p}.norm.bias"), filled(&[hs], 0.0));
    base.insert(format!("{p}.linear_1.weight"), ramp(&[d, hs], base_value));
    base.insert(format!("{p}.linear_1.bias"), filled(&[d], 0.0));
    base.insert(format!("{p}.linear_2.weight"), ramp(&[hs, d], base_value));
    base.insert(format!("{p}.linear_2.bias"), filled(&[hs], 0.0));
  }
  let dir = std::env::temp_dir().join(format!("mlxrs_wav2vec2_mms_{tag}_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(dir.join("config.json"), config_json).unwrap();
  crate::io::save_safetensors(&dir.join("model.safetensors"), &base).unwrap();
  for (lang, value) in adapters {
    let adapter = adapter_file_weights(config, *value);
    crate::io::save_safetensors(&dir.join(format!("adapter.{lang}.safetensors")), &adapter)
      .unwrap();
  }
  dir
}

#[test]
fn mms_load_overlays_per_language_adapter_and_lm_head() {
  // The MMS per-language overlay (mms.py post_load_hook): loading a checkpoint
  // with an adapter.{lang}.safetensors must REPLACE the base's adapter-layer
  // weights AND lm_head with the per-language ones. Proven two ways:
  //   (1) the loaded lm_head weight equals the ADAPTER file's lm_head (not the
  //       base init's) — a direct, byte-level proof of the overlay;
  //   (2) loading the same base with TWO DIFFERENT per-language adapters
  //       (eng vs fra) produces DIFFERENT logits (both go through the identical
  //       on-disk → sanitize → build path, differing only in which adapter was
  //       overlaid) — proving the adapter weights flow into the forward, not
  //       just the lm_head. (An MMS config now REQUIRES an adapter, so a
  //       no-overlay baseline is no longer a valid load — see
  //       `mms_load_requires_adapter_file_for_mms_config`.)
  let config_json = mms_config_json_str();
  let config = Config::from_json(config_json).unwrap();
  const BASE_V: f32 = 0.2;
  const ENG_V: f32 = 0.6;
  const FRA_V: f32 = 0.35;
  // Two dirs sharing the SAME base init; one ships adapter.eng, the other ships a
  // DIFFERENT adapter.fra (a distinct per-language ramp).
  let eng_dir = write_mms_checkpoint(
    config_json,
    &config,
    BASE_V,
    &[("eng", ENG_V)],
    "overlay_eng",
  );
  let fra_dir = write_mms_checkpoint(
    config_json,
    &config,
    BASE_V,
    &[("fra", FRA_V)],
    "overlay_fra",
  );

  // `Model` is not `Debug`, so unwrap via `match` (not `.expect`).
  let eng_model =
    match Model::<Standard>::load_with_target_lang(&eng_dir.to_string_lossy(), Some("eng")) {
      Ok(m) => m,
      Err(e) => panic!("loading an MMS checkpoint with adapter.eng must succeed, got {e:?}"),
    };
  let fra_model =
    match Model::<Standard>::load_with_target_lang(&fra_dir.to_string_lossy(), Some("fra")) {
      Ok(m) => m,
      Err(e) => panic!("loading an MMS checkpoint with adapter.fra must succeed, got {e:?}"),
    };

  // (1) The eng-adapter model's lm_head must be the ADAPTER's (ENG_V ramp), not
  // the base init's (BASE_V ramp). Compare against the adapter file's lm_head
  // weight computed independently.
  let got_head = lm_head_weight(&eng_model);
  let want_head = ramp(&[config.vocab_size, config.hidden_size], ENG_V)
    .to_vec::<f32>()
    .unwrap();
  assert_eq!(got_head.len(), want_head.len());
  assert!(
    got_head
      .iter()
      .zip(want_head.iter())
      .all(|(a, b)| (a - b).abs() < 1e-5),
    "the loaded lm_head must be the per-language adapter's, not the base init's"
  );
  // The base init lm_head (BASE_V ramp) must differ from the adapter's (ENG_V) —
  // proof the overlay is what changed lm_head, and that BASE_V and ENG_V actually
  // differ (so (1) discriminates).
  let base_head = ramp(&[config.vocab_size, config.hidden_size], BASE_V)
    .to_vec::<f32>()
    .unwrap();
  assert!(
    base_head
      .iter()
      .zip(want_head.iter())
      .any(|(a, b)| (a - b).abs() > 1e-5),
    "the base-init and adapter lm_heads must differ (else the test cannot discriminate)"
  );

  // (2) The eng vs fra models' logits must DIFFER — both built from the identical
  // on-disk base via the same sanitize/build path, differing ONLY in which
  // per-language adapter was overlaid. So the adapter-layer weights (not just
  // lm_head) reach the forward.
  let waveform = ramp(&[1, 400], 0.3);
  let mut eng_logits = eng_model.forward(&waveform).unwrap();
  let mut fra_logits = fra_model.forward(&waveform).unwrap();
  let _ = std::fs::remove_dir_all(&eng_dir);
  let _ = std::fs::remove_dir_all(&fra_dir);
  assert_eq!(eng_logits.shape(), fra_logits.shape());
  let a = eng_logits.to_vec::<f32>().unwrap();
  let b = fra_logits.to_vec::<f32>().unwrap();
  assert!(
    a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-4),
    "two different per-language adapters must produce different logits (the overlay reaches the forward)"
  );
}

/// The config JSON string for the MMS fixture (must match `mms_config_json`'s
/// dims so the saved weights line up with the parsed config).
fn mms_config_json_str() -> &'static str {
  r#"{
    "model_type": "wav2vec2",
    "hidden_size": 32, "num_attention_heads": 4, "intermediate_size": 64,
    "num_hidden_layers": 2, "vocab_size": 12,
    "num_feat_extract_layers": 3,
    "conv_dim": [16, 16, 16], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
    "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4,
    "do_stable_layer_norm": true, "adapter_attn_dim": 8
  }"#
}

#[test]
fn mms_adapter_file_discovery_exact_then_fallback() {
  // adapter_file_for: the EXACT `adapter.{target_lang}.safetensors` is preferred;
  // when it is absent, it falls back to the lexicographically-smallest
  // `adapter.*.safetensors`. Drive both branches against a real temp dir.
  let config = Config::from_json(mms_config_json_str()).unwrap();
  // Two adapter files (fra, deu) but NOT the requested eng.
  let dir = write_mms_checkpoint(
    mms_config_json_str(),
    &config,
    0.2,
    &[("fra", 0.5), ("deu", 0.6)],
    "discovery",
  );

  // (a) Requesting "fra" finds the exact adapter.fra.safetensors AND reports the
  // selected language as the requested "fra".
  let exact = adapter_file_for(&dir, "fra")
    .unwrap()
    .expect("fra adapter present");
  assert!(
    exact.path.file_name().unwrap().to_str().unwrap() == "adapter.fra.safetensors",
    "the exact requested-language file must be chosen, got {:?}",
    exact.path
  );
  assert_eq!(
    exact.lang, "fra",
    "an exact hit reports the requested language as selected"
  );
  // (b) Requesting "eng" (absent) falls back to the smallest adapter.* —
  // "adapter.deu.safetensors" < "adapter.fra.safetensors" lexicographically —
  // and reports the FALLBACK's language ("deu"), not the requested "eng", so the
  // caller can align the vocab to it.
  let fallback = adapter_file_for(&dir, "eng")
    .unwrap()
    .expect("a fallback adapter.* must be found");
  assert_eq!(
    fallback.path.file_name().unwrap().to_str().unwrap(),
    "adapter.deu.safetensors",
    "the fallback must be the lexicographically-smallest adapter.* file"
  );
  assert_eq!(
    fallback.lang, "deu",
    "the fallback reports the selected file's language (deu), not the requested eng"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mms_adapter_discovery_none_when_no_adapter_files() {
  // A checkpoint dir with NO adapter.*.safetensors (a plain wav2vec2 / HuBERT
  // model) yields None — not an error, just the absence of an overlay.
  let dir = std::env::temp_dir().join(format!("mlxrs_wav2vec2_noadapter_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  // Write an unrelated file so the dir is non-empty but has no adapter.
  std::fs::write(dir.join("config.json"), "{}").unwrap();
  let found = adapter_file_for(&dir, "eng").unwrap();
  let _ = std::fs::remove_dir_all(&dir);
  assert!(found.is_none(), "no adapter.* files ⇒ None (no overlay)");
}

#[test]
fn mms_adapter_file_for_returns_canonical_in_dir_path() {
  // No-escape / TOCTOU closure: `adapter_file_for` returns the CANONICALIZED path
  // that passed the under-`dir` no-escape check — NOT the raw `entry.path()` —
  // so the loader opens exactly the validated path (no swap-between-check-and-
  // open re-resolution of the unchecked original). The returned `SelectedAdapter`
  // path must equal the canonicalized in-`dir` adapter file.
  let config = Config::from_json(mms_config_json_str()).unwrap();
  let dir = write_mms_checkpoint(
    mms_config_json_str(),
    &config,
    0.2,
    &[("fra", 0.5)],
    "canon_path",
  );
  let selected = adapter_file_for(&dir, "fra")
    .unwrap()
    .expect("fra adapter present");
  let expected_canon = dir.join("adapter.fra.safetensors").canonicalize().unwrap();
  assert_eq!(
    selected.path, expected_canon,
    "the returned adapter path must be the canonicalized in-dir file (the validated path), \
     so the loader opens exactly what was checked"
  );
  // It must be absolute (a canonicalized path always is) — a further proof it is
  // the resolved path, not a relative `entry.path()` artifact.
  assert!(
    selected.path.is_absolute(),
    "a canonicalized adapter path is absolute"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mms_adapter_file_for_resolves_in_dir_symlink_to_canonical_target() {
  // Canonical-path resolution (defense in depth): an `adapter.{lang}.safetensors`
  // that is a SYMLINK to a real adapter file located ELSEWHERE UNDER the model
  // dir is accepted, and the returned path is the canonical (resolved) target —
  // demonstrating the loader opens the resolved path that passed the under-`dir`
  // check, not the symlink's own `entry.path()`.
  use std::os::unix::fs::symlink;
  let config = Config::from_json(mms_config_json_str()).unwrap();
  // Build a checkpoint whose real adapter lives under a `real/` subdir; expose it
  // at the top level via a symlink named `adapter.fra.safetensors`.
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_canon_symlink_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(dir.join("real")).unwrap();
  let target = dir.join("real").join("adapter.fra.safetensors");
  let adapter = adapter_file_weights(&config, 0.5);
  crate::io::save_safetensors(&target, &adapter).unwrap();
  let link = dir.join("adapter.fra.safetensors");
  symlink(&target, &link).unwrap();

  let selected = adapter_file_for(&dir, "fra")
    .unwrap()
    .expect("the symlinked in-dir adapter must be discovered");
  // The returned path is the canonical TARGET (resolved through the symlink),
  // which stays under the (canonicalized) model dir.
  let canon_target = target.canonicalize().unwrap();
  assert_eq!(
    selected.path, canon_target,
    "the returned path must be the symlink's canonical target (the validated path the loader opens)"
  );
  assert!(
    selected.path.starts_with(dir.canonicalize().unwrap()),
    "the resolved adapter path stays under the model dir"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mms_adapter_file_for_rejects_symlink_escaping_dir() {
  // No-escape invariant: an `adapter.{lang}.safetensors` symlink that resolves
  // OUTSIDE the model dir is rejected with a typed error — never overlaid. (And
  // because discovery now returns the canonical path, a symlink that passes the
  // check could not later be swapped for an escaping target: the checked path IS
  // the opened path.)
  use std::os::unix::fs::symlink;
  let config = Config::from_json(mms_config_json_str()).unwrap();
  // The real adapter target lives in a SIBLING dir (outside the model dir).
  let base = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_escape_symlink_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&base);
  let model_dir = base.join("model");
  let outside_dir = base.join("outside");
  std::fs::create_dir_all(&model_dir).unwrap();
  std::fs::create_dir_all(&outside_dir).unwrap();
  let outside_target = outside_dir.join("secret.safetensors");
  let adapter = adapter_file_weights(&config, 0.5);
  crate::io::save_safetensors(&outside_target, &adapter).unwrap();
  // Inside the model dir, an adapter.fra.safetensors symlink pointing OUT.
  let link = model_dir.join("adapter.fra.safetensors");
  symlink(&outside_target, &link).unwrap();

  let result = adapter_file_for(&model_dir, "fra");
  let _ = std::fs::remove_dir_all(&base);
  // `SelectedAdapter` is not `Debug`, so discriminate without formatting it.
  match result {
    Err(Error::InvariantViolation(_)) => {}
    Err(other) => {
      panic!("an escaping adapter symlink must be an InvariantViolation, got {other:?}")
    }
    Ok(_) => panic!("an adapter symlink escaping the model dir must be rejected, not accepted"),
  }
}

#[test]
fn mms_adapter_file_for_rejects_in_dir_symlink_to_other_language() {
  // Adapter/vocab language consistency: an in-`dir` `adapter.eng.safetensors`
  // that is a SYMLINK to a real `adapter.fra.safetensors` (also under the dir)
  // passes the under-`dir` escape check, but opening it would load the FRENCH
  // weights while the entry name "eng" selects the ENGLISH vocab — a silent
  // wrong-language transcription. `adapter_file_for` must reject it: the
  // canonical basename's language (`fra`) differs from the discovered entry's
  // (`eng`), so it is a typed InvariantViolation, never a SelectedAdapter whose
  // weights and `lang` describe different languages.
  use std::os::unix::fs::symlink;
  let config = Config::from_json(mms_config_json_str()).unwrap();
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_lang_desync_symlink_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  // A real French adapter, and an `adapter.eng.safetensors` symlink retargeting
  // to it (both top-level under the model dir, so the no-escape check passes).
  let fra_target = dir.join("adapter.fra.safetensors");
  let adapter = adapter_file_weights(&config, 0.5);
  crate::io::save_safetensors(&fra_target, &adapter).unwrap();
  let eng_link = dir.join("adapter.eng.safetensors");
  symlink(&fra_target, &eng_link).unwrap();

  // Requesting "eng" discovers the `adapter.eng.safetensors` entry, but it
  // canonicalizes to `adapter.fra.safetensors` — a language mismatch.
  let result = adapter_file_for(&dir, "eng");
  let _ = std::fs::remove_dir_all(&dir);
  // `SelectedAdapter` is not `Debug`, so discriminate without formatting an `Ok`.
  match result {
    Err(Error::InvariantViolation(_)) => {}
    Err(other) => panic!(
      "an adapter symlink retargeting to another language must be an InvariantViolation, got \
       {other:?}"
    ),
    Ok(_) => panic!(
      "an `adapter.eng.safetensors` symlinked to `adapter.fra.safetensors` must be rejected (its \
       weights and selected vocab would describe different languages), not accepted"
    ),
  }
}

#[test]
fn expected_adapter_keys_fallible_path_builds_full_key_set() {
  // Fallible allocation: `expected_adapter_keys` reserves its
  // `O(num_hidden_layers)` key `Vec` FALLIBLY (via `reserve_or_error`) and
  // returns `Result`. A within-cap config must build the full key set with no
  // panic: 6 adapter-layer tensors per layer + the 2 lm_head keys, every key
  // present and the count exact.
  let config = Config::from_json(mms_config_json_str()).unwrap();
  let n = config.num_hidden_layers as usize;
  let keys = expected_adapter_keys(&config).expect("within-cap key set reserves successfully");
  assert_eq!(
    keys.len(),
    n.saturating_mul(6).saturating_add(2),
    "6 adapter-layer tensors per layer + 2 lm_head keys"
  );
  for i in 0..n {
    let p = format!("encoder.layers.{i}.adapter_layer");
    for suffix in [
      "norm.weight",
      "norm.bias",
      "linear_1.weight",
      "linear_1.bias",
      "linear_2.weight",
      "linear_2.bias",
    ] {
      assert!(
        keys.contains(&format!("{p}.{suffix}")),
        "expected key {p}.{suffix} present"
      );
    }
  }
  assert!(keys.contains(&"lm_head.weight".to_string()));
  assert!(keys.contains(&"lm_head.bias".to_string()));
}

#[test]
fn expected_adapter_keys_at_cardinality_cap_does_not_panic() {
  // A config AT the cardinality cap (the largest count `validate` accepts) must
  // build its key set through the fallible path WITHOUT panicking or aborting.
  // The cap bounds the request and the `try_reserve_exact` path would surface any
  // genuine OOM as a typed `AllocFailure` — here the in-cap allocation succeeds,
  // returning `Ok` with the full key count.
  let json = format!(
    r#"{{
      "model_type": "wav2vec2",
      "hidden_size": 32, "num_attention_heads": 4, "intermediate_size": 64,
      "num_hidden_layers": {MAX_CONFIG_CARDINALITY}, "vocab_size": 12,
      "num_feat_extract_layers": 3,
      "conv_dim": [16, 16, 16], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
      "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4,
      "do_stable_layer_norm": true, "adapter_attn_dim": 8
    }}"#
  );
  let config = Config::from_json(&json).unwrap();
  // `validate` accepts the at-cap count (the cap is inclusive).
  assert!(config.validate().is_ok(), "at-cap config validates");
  let keys = expected_adapter_keys(&config).expect("at-cap key set reserves successfully");
  assert_eq!(
    keys.len(),
    (MAX_CONFIG_CARDINALITY as usize)
      .saturating_mul(6)
      .saturating_add(2)
  );
}

#[test]
fn overlay_uses_fallible_allowed_set_and_rejects_foreign_key() {
  // Fallible allocation + structural-contract regression: `overlay_adapter_weights`
  // builds its allowed-key `HashSet` FALLIBLY (via `reserve_or_error`) from the
  // (`Result`-returning) `expected_adapter_keys`, and reserves the overlay into
  // `base` fallibly from the adapter file's key count. The structural contract is
  // unchanged: a complete, allowed overlay succeeds; a foreign key is still a
  // typed `KeyCollision`. (A within-cap count never panics on the reservation —
  // it succeeds here.)
  let config = Config::from_json(mms_config_json_str()).unwrap();
  // `Array` is not `Clone`, so rebuild a fresh base map (backbone + the
  // language-agnostic adapter init + lm_head the overlay replaces) per call.
  let fresh_base = || -> HashMap<String, Array> {
    let mut base = sanitize(hf_layout_prefixed_weights(&config, "wav2vec2.")).unwrap();
    for (k, v) in adapter_file_weights(&config, 0.2) {
      base.insert(k, v);
    }
    base
  };
  // Write a valid adapter file and overlay it through the real function.
  let dir = std::env::temp_dir().join(format!("mlxrs_w2v_overlay_fallible_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  let good = adapter_file_weights(&config, 0.7);
  let good_path = dir.join("adapter.fra.safetensors");
  crate::io::save_safetensors(&good_path, &good).unwrap();
  let mut base_ok = fresh_base();
  overlay_adapter_weights(&mut base_ok, &good_path, &config, true)
    .expect("a complete allowed overlay must succeed (fallible reservations OK in-cap)");

  // A foreign key (a stray attention sidecar the adapter must not carry) is still
  // rejected as a typed KeyCollision — the allocation-discipline change did not
  // weaken the structural contract.
  let mut bad = adapter_file_weights(&config, 0.7);
  bad.insert(
    "encoder.layers.0.attention.q_proj.scales".to_string(),
    filled(&[4], 1.0),
  );
  let bad_path = dir.join("adapter.deu.safetensors");
  crate::io::save_safetensors(&bad_path, &bad).unwrap();
  let mut base_bad = fresh_base();
  let err = overlay_adapter_weights(&mut base_bad, &bad_path, &config, true);
  let _ = std::fs::remove_dir_all(&dir);
  assert!(
    matches!(err, Err(Error::KeyCollision(_))),
    "a foreign adapter key must still be a typed KeyCollision, got {err:?}"
  );
}

#[test]
fn overlay_rejects_layernorm_sidecar_and_orphan_biases() {
  // Exact sidecar contract: quant sidecars (`.scales`/`.biases`) are admitted
  // ONLY for the `build_linear`-loaded prefixes (the adapter `linear_1` /
  // `linear_2` projections and `lm_head`) — never for the `take_shaped` adapter
  // `LayerNorm` (`adapter_layer.norm`), and a `.biases` is valid only ALONGSIDE
  // its `.scales`. Two malformed adapters are rejected:
  //   (1) `encoder.layers.0.adapter_layer.norm.scales` — a sidecar on a
  //       non-quantizable LayerNorm prefix. `norm` is never quantized, so this is
  //       a key `build_linear` would never read; it must be a foreign key
  //       (KeyCollision), NOT admitted merely because `norm.weight` is present;
  //   (2) `encoder.layers.0.adapter_layer.linear_1.biases` with no matching
  //       `.scales` — a `.biases` is the affine half of a quantized triple, read
  //       only with its `.scales`; a lone `.biases` (even next to its `.weight`)
  //       is a MissingKey naming the absent `.scales`.
  let config = Config::from_json(mms_config_json_str()).unwrap();
  // `Array` is not `Clone`, so rebuild a fresh base map (backbone + the
  // language-agnostic adapter init + lm_head the overlay replaces) per call.
  let fresh_base = || -> HashMap<String, Array> {
    let mut base = sanitize(hf_layout_prefixed_weights(&config, "wav2vec2.")).unwrap();
    for (k, v) in adapter_file_weights(&config, 0.2) {
      base.insert(k, v);
    }
    base
  };
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_w2v_overlay_sidecar_contract_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();

  // (1) A LayerNorm `.scales` sidecar — a foreign key on a non-quantizable
  // prefix. The `norm.weight` it shadows IS present (the full adapter is
  // complete), so this proves the sidecar is rejected on the PREFIX being
  // non-quantizable, not on a missing companion weight.
  let mut ln_sidecar = adapter_file_weights(&config, 0.7);
  ln_sidecar.insert(
    "encoder.layers.0.adapter_layer.norm.scales".to_string(),
    filled(&[config.hidden_size], 1.0),
  );
  let ln_path = dir.join("adapter.fra.safetensors");
  crate::io::save_safetensors(&ln_path, &ln_sidecar).unwrap();
  let mut base_ln = fresh_base();
  let err_ln = overlay_adapter_weights(&mut base_ln, &ln_path, &config, true);
  assert!(
    matches!(err_ln, Err(Error::KeyCollision(_))),
    "a `.scales` on the non-quantizable adapter LayerNorm prefix must be a foreign-key \
     KeyCollision, got {err_ln:?}"
  );

  // (2) A linear-projection `.biases` with no `.scales` sibling (its `.weight` is
  // present in the complete adapter) — an incomplete affine triple, rejected as a
  // MissingKey naming the absent `.scales`.
  let mut orphan_biases = adapter_file_weights(&config, 0.7);
  let d = config.adapter_attn_dim.unwrap();
  orphan_biases.insert(
    "encoder.layers.0.adapter_layer.linear_1.biases".to_string(),
    filled(&[d], 0.0),
  );
  let ob_path = dir.join("adapter.deu.safetensors");
  crate::io::save_safetensors(&ob_path, &orphan_biases).unwrap();
  let mut base_ob = fresh_base();
  let err_ob = overlay_adapter_weights(&mut base_ob, &ob_path, &config, true);
  let _ = std::fs::remove_dir_all(&dir);
  assert!(
    matches!(err_ob, Err(Error::MissingKey(_))),
    "a `.biases` with no `.scales` sibling on a linear prefix must be a MissingKey, got {err_ob:?}"
  );
}

#[test]
fn mms_load_falls_back_to_first_adapter_when_requested_lang_absent() {
  // When the requested target_lang has no adapter file, load() falls back to the
  // first adapter.* (mms.py's `adapters[0]`). Here only adapter.fra exists; a
  // request for the (absent) default "eng" must still load — applying the fra
  // adapter (the only one) — so the loaded lm_head is fra's, not the base init's.
  let config_json = mms_config_json_str();
  let config = Config::from_json(config_json).unwrap();
  const BASE_V: f32 = 0.2;
  const FRA_V: f32 = 0.7;
  let dir = write_mms_checkpoint(config_json, &config, BASE_V, &[("fra", FRA_V)], "fallback");

  // Request the default language (eng) — absent; the loader must fall back to
  // adapter.fra.
  let loaded = Model::<Standard>::load(&dir.to_string_lossy());
  let _ = std::fs::remove_dir_all(&dir);
  let model = match loaded {
    Ok(m) => m,
    Err(e) => panic!("loading must fall back to the only adapter.* file, got {e:?}"),
  };
  let got_head = lm_head_weight(&model);
  let want_fra = ramp(&[config.vocab_size, config.hidden_size], FRA_V)
    .to_vec::<f32>()
    .unwrap();
  assert!(
    got_head
      .iter()
      .zip(want_fra.iter())
      .all(|(a, b)| (a - b).abs() < 1e-5),
    "the fallback must apply the only adapter.* (fra), so the lm_head is fra's"
  );
}

#[test]
fn mms_multilingual_vocab_selects_target_language() {
  // Vocab::from_json_for_lang on a nested {lang: {token: id}} MMS vocab must
  // select the requested language's map (then eng/en, then the smallest key).
  let nested = r#"{
    "eng": {"<pad>": 0, "A": 1, "B": 2},
    "fra": {"<pad>": 0, "E": 1, "T": 2},
    "deu": {"<pad>": 0, "X": 1, "Y": 2}
  }"#;
  // Requested language wins.
  let fra = Vocab::from_json_for_lang(nested, "fra").unwrap();
  assert_eq!(fra.token(1), Some("E"));
  assert_eq!(fra.token(2), Some("T"));
  // An absent requested language falls back to eng.
  let missing = Vocab::from_json_for_lang(nested, "spa").unwrap();
  assert_eq!(
    missing.token(1),
    Some("A"),
    "fallback to eng when target absent"
  );
  // A flat monolingual vocab still parses unchanged (the base-960h shape).
  let flat = Vocab::from_json_for_lang(r#"{"<pad>": 0, "Z": 1}"#, "eng").unwrap();
  assert_eq!(flat.token(1), Some("Z"));
}

#[test]
fn mms_multilingual_vocab_fallback_without_eng() {
  // With no eng/en key, the nested-vocab selection falls back to the
  // lexicographically-smallest language key (a deterministic substitute for the
  // reference's insertion-order first). Keys "deu" < "fra", so "deu" wins.
  let nested = r#"{
    "fra": {"<pad>": 0, "E": 1},
    "deu": {"<pad>": 0, "X": 1}
  }"#;
  let v = Vocab::from_json_for_lang(nested, "spa").unwrap();
  assert_eq!(
    v.token(1),
    Some("X"),
    "with no eng/en, the smallest language key (deu) is selected"
  );
}

// ─────────── MMS adapter loading: completeness + vocab-alignment + quant ───────────
// Three correctness properties of the MMS per-language adapter loader:
//   1. an MMS config (adapter_attn_dim on the stable-LN arm) REQUIRES a complete
//      adapter — a missing file or a missing tensor/lm_head half is a typed load
//      failure, not a silent base build;
//   2. the vocab follows the SELECTED adapter language (a fallback adapter is
//      decoded with its own token table, not the requested language's);
//   3. a DENSE per-language overlay over a QUANTIZED base removes the base's
//      stale `.scales`/`.biases` so the overlaid layer loads dense (not as a
//      mis-read packed-quantized layer).

#[test]
fn mms_load_requires_adapter_file_for_mms_config() {
  // Property 1 (adapter required): an MMS config (adapter_attn_dim + stable-LN)
  // with NO adapter.*.safetensors must be a typed load failure — never a silent
  // build from the language-agnostic base init (which would transcribe WRONG).
  let config_json = mms_config_json_str();
  let config = Config::from_json(config_json).unwrap();
  // A base checkpoint with the language-agnostic adapter init but NO adapter file.
  let dir = write_mms_checkpoint(config_json, &config, 0.2, &[], "require_none");
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir);
  match loaded {
    Err(Error::MissingKey(p)) => assert!(
      p.key().contains("adapter.eng.safetensors"),
      "the error must name the missing adapter file, got {:?}",
      p.key()
    ),
    Err(other) => panic!("an MMS config with no adapter file must MissingKey, got {other:?}"),
    Ok(_) => {
      panic!("an MMS config with no adapter file must NOT silently build from the base init")
    }
  }
}

#[test]
fn mms_load_rejects_adapter_missing_required_key() {
  // Property 1 (adapter complete): an adapter file present but MISSING a
  // required key (an adapter tensor, or an lm_head half) must be a typed load
  // failure naming the absent key — the overlay is validated COMPLETE before any
  // merge.
  let config_json = mms_config_json_str();
  let config = Config::from_json(config_json).unwrap();

  // Case (a): missing a per-layer adapter tensor.
  let dir_a = write_mms_checkpoint(config_json, &config, 0.2, &[("eng", 0.6)], "miss_layer");
  {
    // Rewrite adapter.eng.safetensors WITHOUT one adapter-layer tensor.
    let mut adapter = adapter_file_weights(&config, 0.6);
    adapter.remove("encoder.layers.0.adapter_layer.linear_2.weight");
    crate::io::save_safetensors(&dir_a.join("adapter.eng.safetensors"), &adapter).unwrap();
  }
  let loaded_a = Model::<Standard>::load_with_target_lang(&dir_a.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir_a);
  match loaded_a {
    Err(Error::MissingKey(p)) => assert!(
      p.key().contains("layers.0.adapter_layer.linear_2.weight"),
      "the error must name the missing adapter tensor, got {:?}",
      p.key()
    ),
    Err(other) => panic!("a truncated adapter (missing tensor) must MissingKey, got {other:?}"),
    Ok(_) => panic!("a truncated adapter must NOT build a hybrid model"),
  }

  // Case (b): missing lm_head.bias (an lm_head half).
  let dir_b = write_mms_checkpoint(config_json, &config, 0.2, &[("eng", 0.6)], "miss_head");
  {
    let mut adapter = adapter_file_weights(&config, 0.6);
    adapter.remove("lm_head.bias");
    crate::io::save_safetensors(&dir_b.join("adapter.eng.safetensors"), &adapter).unwrap();
  }
  let loaded_b = Model::<Standard>::load_with_target_lang(&dir_b.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir_b);
  match loaded_b {
    Err(Error::MissingKey(p)) => assert!(
      p.key().contains("lm_head.bias"),
      "the error must name the missing lm_head half, got {:?}",
      p.key()
    ),
    Err(other) => panic!("an adapter missing lm_head.bias must MissingKey, got {other:?}"),
    Ok(_) => panic!("an adapter missing lm_head.bias must NOT build"),
  }
}

#[test]
fn mms_load_complete_adapter_builds_and_non_mms_no_adapter_is_noop() {
  // Property 1 (positive cases): a COMPLETE adapter overlays + builds; and a
  // NON-MMS config (no adapter_attn_dim) with no adapter files still loads (the
  // absence-of-adapter no-op is unchanged for plain wav2vec2 / HuBERT).
  let config_json = mms_config_json_str();
  let config = Config::from_json(config_json).unwrap();
  // (a) Complete adapter → builds.
  let dir = write_mms_checkpoint(config_json, &config, 0.2, &[("eng", 0.6)], "complete");
  let built = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir);
  match built {
    Ok(_) => {}
    Err(e) => panic!("a complete MMS adapter must overlay + build, got {e:?}"),
  }

  // (b) A plain (non-MMS) checkpoint with no adapter files: write a dense
  // backbone with NO adapter_attn_dim and NO adapter.*.safetensors. It must load
  // (no adapter requirement, no overlay).
  let plain_json = r#"{
    "model_type": "wav2vec2",
    "hidden_size": 32, "num_attention_heads": 4, "intermediate_size": 64,
    "num_hidden_layers": 2, "vocab_size": 12,
    "num_feat_extract_layers": 3,
    "conv_dim": [16, 16, 16], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
    "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4,
    "do_stable_layer_norm": true
  }"#;
  let plain_config = Config::from_json(plain_json).unwrap();
  assert_eq!(plain_config.adapter_attn_dim, None);
  let base = hf_layout_prefixed_weights(&plain_config, "wav2vec2.");
  let plain_dir = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_plain_noadapter_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&plain_dir);
  std::fs::create_dir_all(&plain_dir).unwrap();
  std::fs::write(plain_dir.join("config.json"), plain_json).unwrap();
  crate::io::save_safetensors(&plain_dir.join("model.safetensors"), &base).unwrap();
  let plain_loaded = Model::<Standard>::load_with_target_lang(&plain_dir.to_string_lossy(), None);
  let _ = std::fs::remove_dir_all(&plain_dir);
  match plain_loaded {
    Ok(_) => {}
    Err(e) => panic!("a plain (non-MMS) checkpoint with no adapter files must load, got {e:?}"),
  }
}

#[test]
fn mms_load_vocab_follows_selected_adapter_language() {
  // Property 2 (vocab follows adapter): requesting "eng" when only adapter.fra is
  // present must OVERLAY the fra adapter AND select the FRA vocab map (aligned
  // with the overlaid adapter + lm_head), NOT the eng vocab — so the French
  // logits are decoded with the French token table.
  let config_json = mms_config_json_str();
  let config = Config::from_json(config_json).unwrap();
  const FRA_V: f32 = 0.4;
  let dir = write_mms_checkpoint(config_json, &config, 0.2, &[("fra", FRA_V)], "vocab_align");
  // A nested multilingual vocab whose eng/fra maps assign DIFFERENT tokens to the
  // same ids (so the selected language is observable).
  let nested_vocab = r#"{
    "eng": {"<pad>": 0, "A": 1, "B": 2},
    "fra": {"<pad>": 0, "E": 1, "T": 2}
  }"#;
  std::fs::write(dir.join("vocab.json"), nested_vocab).unwrap();

  // Request eng (absent) → falls back to the only adapter (fra).
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let model = match loaded {
    Ok(m) => m,
    Err(e) => panic!("a fallback-to-fra load must succeed, got {e:?}"),
  };
  // (1) The lm_head is fra's (the overlaid adapter) — adapter aligned to fra.
  let got_head = lm_head_weight(&model);
  let want_fra = ramp(&[config.vocab_size, config.hidden_size], FRA_V)
    .to_vec::<f32>()
    .unwrap();
  assert!(
    got_head
      .iter()
      .zip(want_fra.iter())
      .all(|(a, b)| (a - b).abs() < 1e-5),
    "the overlaid lm_head must be the fallback (fra) adapter's"
  );
  // (2) The vocab follows fra (token 1 = "E", 2 = "T"), NOT eng ("A"/"B") — the
  // load-bearing assertion: adapter language and vocab language MATCH.
  assert_eq!(
    model.vocab.token(1),
    Some("E"),
    "the vocab must follow the SELECTED adapter language (fra: id 1 = E), not the requested eng (A)"
  );
  assert_eq!(model.vocab.token(2), Some("T"), "fra vocab id 2 = T");
  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mms_load_exact_language_uses_that_adapter_and_vocab() {
  // Property 2 (exact case): an exact-language request uses THAT language's
  // adapter AND vocab — adapter and vocab languages match on the exact path too.
  let config_json = mms_config_json_str();
  let config = Config::from_json(config_json).unwrap();
  const ENG_V: f32 = 0.55;
  const FRA_V: f32 = 0.35;
  // Both adapters present; request fra exactly.
  let dir = write_mms_checkpoint(
    config_json,
    &config,
    0.2,
    &[("eng", ENG_V), ("fra", FRA_V)],
    "exact_align",
  );
  let nested_vocab = r#"{
    "eng": {"<pad>": 0, "A": 1},
    "fra": {"<pad>": 0, "E": 1}
  }"#;
  std::fs::write(dir.join("vocab.json"), nested_vocab).unwrap();
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("fra"));
  let model = match loaded {
    Ok(m) => m,
    Err(e) => panic!("an exact fra request must load, got {e:?}"),
  };
  // lm_head is fra's, and the vocab is fra's (id 1 = E, not eng's A).
  let got_head = lm_head_weight(&model);
  let want_fra = ramp(&[config.vocab_size, config.hidden_size], FRA_V)
    .to_vec::<f32>()
    .unwrap();
  assert!(
    got_head
      .iter()
      .zip(want_fra.iter())
      .all(|(a, b)| (a - b).abs() < 1e-5),
    "an exact fra request must overlay the fra adapter"
  );
  assert_eq!(
    model.vocab.token(1),
    Some("E"),
    "an exact fra request must select the fra vocab (id 1 = E)"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Group size for the quantized-MMS fixtures — a valid mlx affine group size
/// (`{32, 64, 128}`). 32 divides every quantized Linear's input width at the
/// [`mms_quant_config_json`] dims (hidden 64, intermediate 128, conv_dim[-1] 64,
/// adapter_attn_dim 32, vocab 64) — base + adapter + lm_head.
const MMS_QGROUP: i32 = 32;

/// A quantized MMS checkpoint's `config.json`: an MMS stable-LN config sized so
/// every quantized Linear's input width is a whole number of [`MMS_QGROUP`]
/// groups (the adapter `linear_2`'s input is `adapter_attn_dim = 32`, so a
/// smaller bottleneck would not be group-aligned), with an 8-bit affine
/// `quantization` block at [`MMS_QGROUP`].
fn mms_quant_config_json() -> String {
  format!(
    r#"{{
      "model_type": "wav2vec2",
      "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 128,
      "num_hidden_layers": 2, "vocab_size": 64,
      "num_feat_extract_layers": 3,
      "conv_dim": [64, 64, 64], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
      "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4,
      "do_stable_layer_norm": true, "adapter_attn_dim": 32,
      "quantization": {{ "group_size": {MMS_QGROUP}, "bits": {QBITS} }}
    }}"#
  )
}

/// Write a QUANTIZED MMS checkpoint to a temp dir: the base `model.safetensors`
/// has every quantized-eligible Linear packed (incl. the per-layer adapter
/// `linear_{1,2}` + `lm_head`), plus an `adapter.{lang}.safetensors`. The
/// adapter's adapter-layer linears + lm_head are DENSE when `dense_adapter`,
/// else quantized (its own `.scales`/`.biases`). Returns the dir.
fn write_quant_mms_checkpoint(
  config: &Config,
  config_json: &str,
  base_value: f32,
  lang: &str,
  adapter_value: f32,
  dense_adapter: bool,
  tag: &str,
) -> std::path::PathBuf {
  let hs = config.hidden_size;
  let d = config.adapter_attn_dim.unwrap();
  // Base: HF-prefixed backbone, then add the (prefixed) adapter init, then pack
  // every quantized-eligible Linear (backbone + adapter linears + lm_head).
  let mut base = hf_layout_prefixed_weights(config, "wav2vec2.");
  for i in 0..(config.num_hidden_layers as usize) {
    let p = format!("wav2vec2.encoder.layers.{i}.adapter_layer");
    base.insert(format!("{p}.norm.weight"), filled(&[hs], 1.0));
    base.insert(format!("{p}.norm.bias"), filled(&[hs], 0.0));
    base.insert(format!("{p}.linear_1.weight"), ramp(&[d, hs], base_value));
    base.insert(format!("{p}.linear_1.bias"), filled(&[d], 0.0));
    base.insert(format!("{p}.linear_2.weight"), ramp(&[hs, d], base_value));
    base.insert(format!("{p}.linear_2.bias"), filled(&[hs], 0.0));
  }
  // Pack the backbone Linears.
  quantize_weight_in_place(
    &mut base,
    "wav2vec2.feature_projection.projection",
    MMS_QGROUP,
  );
  for i in 0..(config.num_hidden_layers as usize) {
    let p = format!("wav2vec2.encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      quantize_weight_in_place(&mut base, &format!("{p}.attention.{proj}"), MMS_QGROUP);
    }
    quantize_weight_in_place(
      &mut base,
      &format!("{p}.feed_forward.intermediate_dense"),
      MMS_QGROUP,
    );
    quantize_weight_in_place(
      &mut base,
      &format!("{p}.feed_forward.output_dense"),
      MMS_QGROUP,
    );
    // Pack the per-layer adapter linears in the base too.
    let ap = format!("{p}.adapter_layer");
    quantize_weight_in_place(&mut base, &format!("{ap}.linear_1"), MMS_QGROUP);
    quantize_weight_in_place(&mut base, &format!("{ap}.linear_2"), MMS_QGROUP);
  }
  quantize_weight_in_place(&mut base, "lm_head", MMS_QGROUP);

  // Adapter file: the adapter-layer linears + lm_head, dense or quantized.
  let mut adapter = adapter_file_weights(config, adapter_value);
  if !dense_adapter {
    for i in 0..(config.num_hidden_layers as usize) {
      let ap = format!("encoder.layers.{i}.adapter_layer");
      quantize_weight_in_place(&mut adapter, &format!("{ap}.linear_1"), MMS_QGROUP);
      quantize_weight_in_place(&mut adapter, &format!("{ap}.linear_2"), MMS_QGROUP);
    }
    quantize_weight_in_place(&mut adapter, "lm_head", MMS_QGROUP);
  }

  let dir = std::env::temp_dir().join(format!("mlxrs_wav2vec2_qmms_{tag}_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(dir.join("config.json"), config_json).unwrap();
  crate::io::save_safetensors(&dir.join("model.safetensors"), &base).unwrap();
  crate::io::save_safetensors(&dir.join(format!("adapter.{lang}.safetensors")), &adapter).unwrap();
  dir
}

/// The adapter-layer linear of a built encoder layer (panics if absent).
fn adapter_linear(layer: &super::EncoderLayer, which: u8) -> &super::Linear {
  let ad = layer
    .adapter_layer
    .as_ref()
    .expect("MMS layer has an adapter");
  match which {
    1 => &ad.linear_1,
    _ => &ad.linear_2,
  }
}

#[test]
fn mms_quant_base_dense_adapter_overlay_loads_dense() {
  // Property 3 (dense overlay clears stale sidecars): a QUANTIZED MMS base + a
  // DENSE adapter overlay (adapter-layer linears + lm_head) must load those
  // overlaid prefixes DENSE — the base's stale `.scales`/`.biases` are removed so
  // build_linear does NOT mis-read the dense F32 weight as a packed-quantized
  // triple (which would dtype/shape-fail).
  let config_json = mms_quant_config_json();
  let config = Config::from_json(&config_json).unwrap();
  let dir = write_quant_mms_checkpoint(
    &config,
    &config_json,
    0.2,
    "eng",
    0.5,
    true,
    "dense_overlay",
  );
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir);
  let model = match loaded {
    Ok(m) => m,
    Err(e) => {
      panic!(
        "a quantized base with a dense adapter overlay must load (stale sidecars removed), got {e:?}"
      )
    }
  };
  // The overlaid lm_head must be DENSE (the adapter supplied a plain .weight).
  assert!(
    !model.lm_head.is_quantized(),
    "a dense lm_head overlay must load DENSE (the stale base .scales must be removed)"
  );
  // Every overlaid adapter-layer linear must be DENSE.
  for (i, layer) in encoder_layers(&model.encoder).iter().enumerate() {
    assert!(
      !adapter_linear(layer, 1).is_quantized(),
      "layer {i} adapter linear_1 must load DENSE after a dense overlay"
    );
    assert!(
      !adapter_linear(layer, 2).is_quantized(),
      "layer {i} adapter linear_2 must load DENSE after a dense overlay"
    );
  }
  // The NON-overlaid base layers keep their quantization (the attention q_proj
  // was packed in the base and not touched by the adapter).
  let layer0 = &encoder_layers(&model.encoder)[0];
  assert!(
    layer0.attention.q_proj.is_quantized(),
    "a non-overlaid base attention projection must stay quantized"
  );
  // A forward must run finite (no dtype/shape blow-up from a mis-read layer).
  let waveform = ramp(&[1, 400], 0.2);
  let mut logits = model
    .forward(&waveform)
    .expect("the hybrid model must forward");
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite())
  );
}

/// Write a QUANTIZED MMS checkpoint whose adapter file is quantized BUT has its
/// per-group affine `.biases` stripped from every overlaid prefix — a truncated
/// quantized adapter (it supplies `.weight` + `.scales` but no `.biases`). The
/// base remains a complete quantized checkpoint (every base prefix keeps its own
/// `.scales` + `.biases`). Returns the dir.
fn write_quant_mms_checkpoint_adapter_missing_biases(
  config: &Config,
  config_json: &str,
  lang: &str,
  tag: &str,
) -> std::path::PathBuf {
  let hs = config.hidden_size;
  let d = config.adapter_attn_dim.unwrap();
  // Base: a COMPLETE quantized checkpoint (the language-agnostic backbone +
  // adapter init + lm_head, every Linear packed with its own scales + biases).
  let mut base = hf_layout_prefixed_weights(config, "wav2vec2.");
  for i in 0..(config.num_hidden_layers as usize) {
    let p = format!("wav2vec2.encoder.layers.{i}.adapter_layer");
    base.insert(format!("{p}.norm.weight"), filled(&[hs], 1.0));
    base.insert(format!("{p}.norm.bias"), filled(&[hs], 0.0));
    base.insert(format!("{p}.linear_1.weight"), ramp(&[d, hs], 0.2));
    base.insert(format!("{p}.linear_1.bias"), filled(&[d], 0.0));
    base.insert(format!("{p}.linear_2.weight"), ramp(&[hs, d], 0.2));
    base.insert(format!("{p}.linear_2.bias"), filled(&[hs], 0.0));
  }
  quantize_weight_in_place(
    &mut base,
    "wav2vec2.feature_projection.projection",
    MMS_QGROUP,
  );
  for i in 0..(config.num_hidden_layers as usize) {
    let p = format!("wav2vec2.encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      quantize_weight_in_place(&mut base, &format!("{p}.attention.{proj}"), MMS_QGROUP);
    }
    quantize_weight_in_place(
      &mut base,
      &format!("{p}.feed_forward.intermediate_dense"),
      MMS_QGROUP,
    );
    quantize_weight_in_place(
      &mut base,
      &format!("{p}.feed_forward.output_dense"),
      MMS_QGROUP,
    );
    let ap = format!("{p}.adapter_layer");
    quantize_weight_in_place(&mut base, &format!("{ap}.linear_1"), MMS_QGROUP);
    quantize_weight_in_place(&mut base, &format!("{ap}.linear_2"), MMS_QGROUP);
  }
  quantize_weight_in_place(&mut base, "lm_head", MMS_QGROUP);

  // Adapter: quantize every overlaid Linear, then STRIP its `.biases` — leaving a
  // truncated affine triple (`.weight` + `.scales`, no `.biases`).
  let mut adapter = adapter_file_weights(config, 0.5);
  let strip_biases = |w: &mut HashMap<String, Array>, prefix: &str| {
    quantize_weight_in_place(w, prefix, MMS_QGROUP);
    assert!(
      w.remove(&format!("{prefix}.biases")).is_some(),
      "the quantized adapter prefix must have produced a .biases to strip"
    );
  };
  for i in 0..(config.num_hidden_layers as usize) {
    let ap = format!("encoder.layers.{i}.adapter_layer");
    strip_biases(&mut adapter, &format!("{ap}.linear_1"));
    strip_biases(&mut adapter, &format!("{ap}.linear_2"));
  }
  strip_biases(&mut adapter, "lm_head");

  let dir = std::env::temp_dir().join(format!("mlxrs_wav2vec2_qmms_{tag}_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(dir.join("config.json"), config_json).unwrap();
  crate::io::save_safetensors(&dir.join("model.safetensors"), &base).unwrap();
  crate::io::save_safetensors(&dir.join(format!("adapter.{lang}.safetensors")), &adapter).unwrap();
  dir
}

#[test]
fn mms_quant_adapter_missing_biases_is_typed_error() {
  // Quant-sidecar self-containment: a QUANTIZED adapter that supplies `.weight` +
  // `.scales` but is MISSING `.biases` must NOT inherit the base prefix's
  // `.biases` (a silent hybrid = adapter weight+scales over BASE biases). With
  // both stale base sidecars removed per overlaid `.weight`, the truncated affine
  // triple has no
  // `.biases` anywhere, so it fails the downstream quant-triple validation
  // (build_linear -> QuantizedLinear::from_parts) with a typed error.
  let config_json = mms_quant_config_json();
  let config = Config::from_json(&config_json).unwrap();
  let dir = write_quant_mms_checkpoint_adapter_missing_biases(
    &config,
    &config_json,
    "eng",
    "missing_biases",
  );
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir);
  match loaded {
    Ok(_) => panic!(
      "a quantized adapter missing .biases must FAIL as incomplete (not silently build a hybrid \
       with the base's .biases)"
    ),
    // `affine` mode requires per-group biases — from_parts rejects the truncated
    // triple. The exact variant is InvariantViolation (mlx's `if (!biases) throw`).
    Err(Error::InvariantViolation(_)) => {}
    Err(e) => {
      panic!("expected a typed InvariantViolation for the missing affine .biases, got {e:?}")
    }
  }
}

#[test]
fn mms_quant_base_quant_adapter_overlay_loads_quantized() {
  // Property 3 (the converse): a QUANTIZED MMS base + a QUANTIZED adapter overlay
  // (the adapter supplies its OWN `.scales`/`.biases`) must load the overlaid
  // prefixes QUANTIZED — the overlay's quant identity wins, the layers are NOT
  // forced dense.
  let config_json = mms_quant_config_json();
  let config = Config::from_json(&config_json).unwrap();
  let dir = write_quant_mms_checkpoint(
    &config,
    &config_json,
    0.2,
    "eng",
    0.5,
    false,
    "quant_overlay",
  );
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir);
  let model = match loaded {
    Ok(m) => m,
    Err(e) => panic!("a quantized base with a quantized adapter overlay must load, got {e:?}"),
  };
  // The overlaid lm_head + adapter-layer linears load QUANTIZED at MMS_QGROUP.
  assert!(
    model.lm_head.is_quantized(),
    "a quantized lm_head overlay must load QUANTIZED"
  );
  for (i, layer) in encoder_layers(&model.encoder).iter().enumerate() {
    assert_eq!(
      loaded_group_size(adapter_linear(layer, 1)),
      Some(MMS_QGROUP),
      "layer {i} adapter linear_1 must load quantized at MMS_QGROUP"
    );
    assert_eq!(
      loaded_group_size(adapter_linear(layer, 2)),
      Some(MMS_QGROUP),
      "layer {i} adapter linear_2 must load quantized at MMS_QGROUP"
    );
  }
  let waveform = ramp(&[1, 400], 0.2);
  let mut logits = model
    .forward(&waveform)
    .expect("the quantized model must forward");
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite())
  );
}

#[test]
fn mms_quant_adapter_extra_foreign_key_is_typed_error() {
  // Exact allowed key set: an MMS adapter that carries an
  // EXTRA/foreign key — one NOT in the adapter-layer + lm_head weights/biases
  // (+ optional matching sidecars) — must be REJECTED with a typed error naming
  // the offending key, BEFORE any merge. Concretely a stray
  // `encoder.layers.0.attention.q_proj.scales` (an attention-projection sidecar
  // the adapter never overlays): a merely-required-keys check would blindly
  // insert it, clobbering the base packed q_proj's `.scales` while leaving the
  // base `.weight` + `.biases` — a silent quantized hybrid from mismatched parts.
  let config_json = mms_quant_config_json();
  let config = Config::from_json(&config_json).unwrap();
  // A QUANTIZED base + a complete QUANTIZED adapter (so the foreign key is the
  // ONLY defect, not a missing/dense-mismatch one).
  let dir = write_quant_mms_checkpoint(&config, &config_json, 0.2, "eng", 0.5, false, "extra_key");
  {
    // Rebuild the (quantized) adapter file exactly, then inject a foreign key:
    // an `encoder.layers.0.attention.q_proj.scales` (a real-looking quant sidecar
    // for a layer the adapter does NOT overlay — not in the expected key set).
    let mut adapter = adapter_file_weights(&config, 0.5);
    for i in 0..(config.num_hidden_layers as usize) {
      let ap = format!("encoder.layers.{i}.adapter_layer");
      quantize_weight_in_place(&mut adapter, &format!("{ap}.linear_1"), MMS_QGROUP);
      quantize_weight_in_place(&mut adapter, &format!("{ap}.linear_2"), MMS_QGROUP);
    }
    quantize_weight_in_place(&mut adapter, "lm_head", MMS_QGROUP);
    // The foreign key (a stray attention-projection sidecar). Its exact shape is
    // irrelevant — it must be rejected purely for being outside the allowed set.
    adapter.insert(
      "encoder.layers.0.attention.q_proj.scales".to_string(),
      filled(&[config.hidden_size, 1], 0.01),
    );
    crate::io::save_safetensors(&dir.join("adapter.eng.safetensors"), &adapter).unwrap();
  }
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir);
  match loaded {
    Ok(_) => panic!(
      "an MMS adapter carrying a foreign key (a sidecar for a layer it does not overlay) must be \
       REJECTED, not silently build a quantized hybrid by clobbering a base tensor"
    ),
    Err(Error::KeyCollision(p)) => assert!(
      p.key().contains("attention.q_proj.scales"),
      "the error must name the offending foreign key, got {:?}",
      p.key()
    ),
    Err(e) => panic!("expected a typed KeyCollision naming the foreign key, got {e:?}"),
  }
}

#[test]
fn mms_quant_adapter_orphan_sidecar_is_typed_error() {
  // Orphan sidecar: an MMS adapter whose `<prefix>.scales` has NO
  // matching `<prefix>.weight` in the overlay is an orphan — it would define a
  // quant identity for a layer the overlay does not actually replace. It must be
  // REJECTED with a typed error naming the absent companion `.weight`. Built by
  // dropping `lm_head.weight` while leaving an `lm_head.scales` behind (the
  // sidecar without its weight).
  let config_json = mms_quant_config_json();
  let config = Config::from_json(&config_json).unwrap();
  let dir = write_quant_mms_checkpoint(
    &config,
    &config_json,
    0.2,
    "eng",
    0.5,
    false,
    "orphan_sidecar",
  );
  {
    // A complete quantized adapter, then strip lm_head.weight + lm_head.biases
    // but KEEP lm_head.scales → an orphan sidecar (scales without its weight).
    let mut adapter = adapter_file_weights(&config, 0.5);
    for i in 0..(config.num_hidden_layers as usize) {
      let ap = format!("encoder.layers.{i}.adapter_layer");
      quantize_weight_in_place(&mut adapter, &format!("{ap}.linear_1"), MMS_QGROUP);
      quantize_weight_in_place(&mut adapter, &format!("{ap}.linear_2"), MMS_QGROUP);
    }
    quantize_weight_in_place(&mut adapter, "lm_head", MMS_QGROUP);
    // Drop the weight + biases, leaving lm_head.scales orphaned.
    assert!(adapter.remove("lm_head.weight").is_some());
    assert!(adapter.remove("lm_head.biases").is_some());
    assert!(
      adapter.contains_key("lm_head.scales"),
      "lm_head.scales must remain as the orphan sidecar"
    );
    crate::io::save_safetensors(&dir.join("adapter.eng.safetensors"), &adapter).unwrap();
  }
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir);
  match loaded {
    Ok(_) => panic!(
      "an MMS adapter with an orphan sidecar (lm_head.scales without lm_head.weight) must be \
       REJECTED, not silently apply a quant identity to a layer it does not overlay"
    ),
    Err(Error::MissingKey(p)) => assert!(
      p.key().contains("lm_head.weight"),
      "the error must name the absent companion weight, got {:?}",
      p.key()
    ),
    Err(e) => panic!("expected a typed MissingKey naming the absent companion weight, got {e:?}"),
  }
}

#[test]
fn mms_quant_complete_adapter_with_sidecars_still_overlays() {
  // Well-formed adapter still overlays: a complete adapter — EXACTLY the expected
  // adapter-layer + lm_head weights/biases plus their matching `.scales`/`.biases`
  // sidecars, and nothing else — must still overlay + load under the exact-key
  // validation (the structural close rejects only EXTRA/orphan keys, never a
  // conforming quantized adapter).
  let config_json = mms_quant_config_json();
  let config = Config::from_json(&config_json).unwrap();
  let dir = write_quant_mms_checkpoint(&config, &config_json, 0.2, "eng", 0.5, false, "exact_ok");
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir);
  let model = match loaded {
    Ok(m) => m,
    Err(e) => panic!("a well-formed complete quantized adapter must overlay + load, got {e:?}"),
  };
  // The overlaid prefixes loaded quantized from their OWN sidecars (the exact-key
  // path did not strip or reject the conforming sidecars).
  assert!(
    model.lm_head.is_quantized(),
    "the conforming quantized lm_head overlay must load quantized"
  );
  let waveform = ramp(&[1, 400], 0.2);
  let mut logits = model.forward(&waveform).expect("the model must forward");
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite())
  );
}

// ───────────── MMS strict vocab for the selected adapter language ─────────────
// Vocab/adapter language alignment: when an adapter file is selected (the MMS
// per-language path) the selected language MUST have an exact entry in a nested
// {lang:{token:id}} vocab.json — no silent fallback to another language's token
// table. The lenient eng/en/smallest fallback is kept ONLY for the no-adapter /
// flat-vocab path.

#[test]
fn vocab_from_json_for_lang_strict_requires_exact_nested_entry() {
  // A nested {lang: {token: id}} vocab that LACKS the requested language must be
  // a typed MissingKey under the strict parser (no fallback to eng/en/smallest).
  let nested = r#"{
    "eng": {"<pad>": 0, "A": 1, "B": 2},
    "deu": {"<pad>": 0, "X": 1, "Y": 2}
  }"#;
  match Vocab::from_json_for_lang_strict(nested, "fra") {
    Err(Error::MissingKey(_)) => {}
    Ok(_) => panic!(
      "the strict parser must NOT fall back to another language's table when the selected lang \
       (fra) is absent from the nested vocab"
    ),
    Err(e) => panic!("expected a typed MissingKey for the absent selected lang, got {e:?}"),
  }
  // The exact requested language is used when present.
  let fra = Vocab::from_json_for_lang_strict(
    r#"{
      "eng": {"<pad>": 0, "A": 1},
      "fra": {"<pad>": 0, "E": 1, "T": 2}
    }"#,
    "fra",
  )
  .unwrap();
  assert_eq!(fra.token(1), Some("E"), "strict fra entry: id 1 = E");
  assert_eq!(fra.token(2), Some("T"), "strict fra entry: id 2 = T");
}

#[test]
fn vocab_from_json_for_lang_strict_flat_vocab_used_as_is() {
  // A FLAT (non-nested) vocab.json is language-agnostic and is used as-is under
  // the strict parser regardless of the requested language (unchanged behavior).
  let flat = Vocab::from_json_for_lang_strict(r#"{"<pad>": 0, "Z": 1}"#, "fra").unwrap();
  assert_eq!(
    flat.token(1),
    Some("Z"),
    "a flat vocab is language-agnostic and used as-is even under strict"
  );
}

#[test]
fn vocab_from_json_for_lang_lenient_still_falls_back() {
  // The lenient (no-adapter path) parser keeps the eng/en/smallest fallback —
  // an absent requested language falls back to eng (the prior behavior, kept
  // green for the no-adapter / plain-checkpoint path).
  let nested = r#"{
    "eng": {"<pad>": 0, "A": 1},
    "deu": {"<pad>": 0, "X": 1}
  }"#;
  let v = Vocab::from_json_for_lang(nested, "fra").unwrap();
  assert_eq!(
    v.token(1),
    Some("A"),
    "the lenient parser still falls back to eng when the requested lang is absent"
  );
}

#[test]
fn mms_load_strict_vocab_rejects_missing_selected_lang() {
  // Language alignment (load path): adapter.fra selected + a nested vocab.json
  // LACKING a "fra" entry must FAIL with a typed error — NOT silently decode the
  // French adapter's logits with another language's token table.
  let config_json = mms_config_json_str();
  let config = Config::from_json(config_json).unwrap();
  let dir = write_mms_checkpoint(
    config_json,
    &config,
    0.2,
    &[("fra", 0.4)],
    "strict_vocab_miss",
  );
  // A nested multilingual vocab that has eng/deu but NO fra (the selected lang).
  let nested_vocab = r#"{
    "eng": {"<pad>": 0, "A": 1, "B": 2},
    "deu": {"<pad>": 0, "X": 1, "Y": 2}
  }"#;
  std::fs::write(dir.join("vocab.json"), nested_vocab).unwrap();
  // Request fra exactly → the fra adapter is selected → vocab_lang = "fra".
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("fra"));
  let _ = std::fs::remove_dir_all(&dir);
  match loaded {
    Ok(_) => panic!(
      "a selected adapter language absent from the nested vocab must FAIL (not silently use \
       another language's vocab)"
    ),
    Err(Error::MissingKey(_)) => {}
    Err(e) => panic!("expected a typed MissingKey for the missing fra vocab entry, got {e:?}"),
  }
}

#[test]
fn mms_load_strict_vocab_uses_exact_selected_lang() {
  // Language alignment (positive): adapter.fra selected + a nested vocab WITH
  // "fra" must load and decode with the fra map (the strict path accepts an exact
  // entry).
  let config_json = mms_config_json_str();
  let config = Config::from_json(config_json).unwrap();
  let dir = write_mms_checkpoint(
    config_json,
    &config,
    0.2,
    &[("fra", 0.4)],
    "strict_vocab_hit",
  );
  let nested_vocab = r#"{
    "eng": {"<pad>": 0, "A": 1, "B": 2},
    "fra": {"<pad>": 0, "E": 1, "T": 2}
  }"#;
  std::fs::write(dir.join("vocab.json"), nested_vocab).unwrap();
  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("fra"));
  let model = match loaded {
    Ok(m) => m,
    Err(e) => panic!("an exact fra request with a matching nested vocab must load, got {e:?}"),
  };
  assert_eq!(model.vocab.token(1), Some("E"), "fra vocab id 1 = E");
  assert_eq!(model.vocab.token(2), Some("T"), "fra vocab id 2 = T");
  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────── adapter-overlay key-restriction applies to EVERY overlay ───────────
// The structural close of the sidecar-hybrid class is not limited to the MMS
// (`require_complete`) path: the foreign-key + orphan-sidecar checks run for ANY
// overlay regardless of `require_complete`, so a sidecar-only stray adapter can
// never clobber a base packed weight's `.scales`/`.biases` and build a silent
// quantized hybrid. (In the LOADER, discovery+overlay are additionally gated on
// an MMS config, so a non-MMS checkpoint never even reads a stray adapter file —
// faithful to mms.py, where only the MMS `Model` defines `post_load_hook`. These
// function-level tests pin the in-function defense in depth directly.)

#[test]
fn overlay_adapter_weights_key_restricts_foreign_key_when_not_complete() {
  // A sidecar-only stray adapter — an `encoder.layers.0.attention.q_proj.scales`
  // with NO matching `q_proj.weight` — overlaid with `require_complete = false`
  // (the non-MMS / not-required path) must STILL be rejected as a foreign key,
  // BEFORE any merge. Previously the foreign-key check was gated on
  // `require_complete`, so this stray sidecar would have been blindly inserted —
  // clobbering the base packed q_proj's `.scales` while leaving the base
  // `.weight` + `.biases` (the silent quantized hybrid this closes).
  let config = quant_config_json(""); // non-MMS (no adapter_attn_dim), post-norm
  assert_eq!(config.adapter_attn_dim, None);
  // A base map carrying a packed q_proj (weight + scales + biases) — the tensor a
  // stray `q_proj.scales` overlay would partially clobber.
  let mut base: HashMap<String, Array> = HashMap::new();
  base.insert(
    "encoder.layers.0.attention.q_proj.weight".to_string(),
    ramp(&[config.hidden_size, config.hidden_size], 0.3),
  );
  quantize_weight_in_place(&mut base, "encoder.layers.0.attention.q_proj", MMS_QGROUP);
  // The stray adapter: ONLY `q_proj.scales` (a sidecar with no matching .weight),
  // a foreign key for a layer no adapter overlays.
  let stray: HashMap<String, Array> = HashMap::from([(
    "encoder.layers.0.attention.q_proj.scales".to_string(),
    filled(&[config.hidden_size, 1], 0.01),
  )]);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_stray_overlay_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  let stray_path = dir.join("adapter.eng.safetensors");
  crate::io::save_safetensors(&stray_path, &stray).unwrap();
  // require_complete = false (the non-MMS path) must STILL reject the foreign key.
  let res = overlay_adapter_weights(&mut base, &stray_path, &config, false);
  let _ = std::fs::remove_dir_all(&dir);
  match res {
    Ok(()) => panic!(
      "a sidecar-only stray adapter must be rejected even when require_complete = false (the \
       key-restriction applies to EVERY overlay, not just the MMS path)"
    ),
    Err(Error::KeyCollision(p)) => assert!(
      p.key().contains("attention.q_proj.scales"),
      "the error must name the foreign key, got {:?}",
      p.key()
    ),
    Err(e) => panic!("expected a typed KeyCollision for the foreign sidecar, got {e:?}"),
  }
}

#[test]
fn non_mms_quant_checkpoint_with_stray_adapter_loads_base_not_hybrid() {
  // Loader gating (non-MMS): a QUANTIZED NON-MMS (post-norm, no
  // adapter_attn_dim) checkpoint that happens to carry a stray sidecar-only
  // `adapter.eng.safetensors` must load the BASE unchanged — the stray adapter is
  // NOT discovered/overlaid (discovery is gated on an MMS config), so there is no
  // silent quantized hybrid. Faithful to mms.py: a plain `Wav2Vec2ForCTC` has no
  // `post_load_hook` adapter overlay.
  let config_json = format!(
    r#"{{
      "model_type": "wav2vec2",
      "hidden_size": 64, "num_attention_heads": 4, "intermediate_size": 128,
      "num_hidden_layers": 2, "vocab_size": 64,
      "num_feat_extract_layers": 3,
      "conv_dim": [64, 64, 64], "conv_kernel": [10, 3, 3], "conv_stride": [5, 2, 2],
      "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4,
      "quantization": {{ "group_size": {MMS_QGROUP}, "bits": {QBITS} }}
    }}"#
  );
  let config = Config::from_json(&config_json).unwrap();
  assert_eq!(
    config.adapter_attn_dim, None,
    "this fixture is a NON-MMS checkpoint"
  );
  // A quantized non-MMS backbone (every eligible Linear packed). No adapter init.
  let mut base = hf_layout_prefixed_weights(&config, "wav2vec2.");
  quantize_weight_in_place(
    &mut base,
    "wav2vec2.feature_projection.projection",
    MMS_QGROUP,
  );
  for i in 0..(config.num_hidden_layers as usize) {
    let p = format!("wav2vec2.encoder.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      quantize_weight_in_place(&mut base, &format!("{p}.attention.{proj}"), MMS_QGROUP);
    }
    quantize_weight_in_place(
      &mut base,
      &format!("{p}.feed_forward.intermediate_dense"),
      MMS_QGROUP,
    );
    quantize_weight_in_place(
      &mut base,
      &format!("{p}.feed_forward.output_dense"),
      MMS_QGROUP,
    );
  }
  quantize_weight_in_place(&mut base, "lm_head", MMS_QGROUP);

  let dir = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_nonmms_stray_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(dir.join("config.json"), &config_json).unwrap();
  crate::io::save_safetensors(&dir.join("model.safetensors"), &base).unwrap();
  // A stray sidecar-only adapter (ONLY a q_proj.scales — no matching .weight).
  // If it were discovered+overlaid, it would clobber the base packed q_proj's
  // scales and build a hybrid (or fail). It must instead be IGNORED.
  let stray: HashMap<String, Array> = HashMap::from([(
    "encoder.layers.0.attention.q_proj.scales".to_string(),
    filled(&[config.hidden_size, 1], 0.01),
  )]);
  crate::io::save_safetensors(&dir.join("adapter.eng.safetensors"), &stray).unwrap();

  let loaded = Model::<Standard>::load_with_target_lang(&dir.to_string_lossy(), Some("eng"));
  let _ = std::fs::remove_dir_all(&dir);
  let model = match loaded {
    Ok(m) => m,
    Err(e) => panic!(
      "a non-MMS quantized checkpoint with a stray adapter must load the BASE unchanged (the stray \
       adapter is not overlaid), got {e:?}"
    ),
  };
  // The base q_proj loaded QUANTIZED at the base group size — i.e. the base's own
  // packed triple, NOT a hybrid built from the stray sidecar. (A hybrid would
  // have replaced the scales with the stray's mis-shaped one; here the base is
  // intact.)
  for (i, layer) in encoder_layers(&model.encoder).iter().enumerate() {
    assert_eq!(
      loaded_group_size(&layer.attention.q_proj),
      Some(MMS_QGROUP),
      "layer {i} base q_proj must load quantized at the base group size (stray adapter ignored)"
    );
  }
  // And the model forwards to finite logits (a corrupt hybrid would not).
  let waveform = ramp(&[1, 400], 0.2);
  let mut logits = model
    .forward(&waveform)
    .expect("the base-only quantized model must forward");
  assert!(
    logits
      .to_vec::<f32>()
      .unwrap()
      .iter()
      .all(|v| v.is_finite())
  );
}

// ───────────── adapter discovery never builds a path from target_lang ─────────────
// Enumerate-and-match (no interpolation): the exact-adapter selection enumerates
// the real on-disk `adapter.*.safetensors` files and matches `target_lang`
// against the EXTRACTED `<lang>` (a single path component), instead of
// interpolating `target_lang` into `adapter.{target_lang}.safetensors`. So a
// `target_lang` carrying path separators or `..` cannot select a file outside the
// model directory.

#[test]
fn adapter_file_for_path_traversal_target_lang_cannot_escape_dir() {
  // A model dir with exactly ONE in-dir adapter (adapter.eng.safetensors). A
  // hostile `target_lang` ("../evil" / "a/b" / "..") must NEVER resolve to a file
  // outside the dir: enumerate-and-match means such a target_lang matches no
  // extracted `<lang>`, so discovery falls back to the in-dir eng adapter (a real
  // directory entry), never an interpolated out-of-dir path.
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_traversal_in_{}",
    std::process::id()
  ));
  // A SIBLING dir holding a decoy adapter the traversal would target if a path
  // were built from target_lang (../<sibling-relative> style). It must NOT be
  // selected.
  let sibling = std::env::temp_dir().join(format!(
    "mlxrs_wav2vec2_traversal_evil_{}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  let _ = std::fs::remove_dir_all(&sibling);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::create_dir_all(&sibling).unwrap();
  // The single legitimate in-dir adapter.
  let one: HashMap<String, Array> =
    HashMap::from([("lm_head.bias".to_string(), filled(&[2], 0.0))]);
  crate::io::save_safetensors(&dir.join("adapter.eng.safetensors"), &one).unwrap();
  // A decoy outside the model dir, named so a naive
  // `dir.join("adapter.{target_lang}.safetensors")` with target_lang
  // `"../<sibling>/adapter.x"`-style could have reached it. (We do not even need
  // it to be reachable; its presence proves selection stays in-dir.)
  crate::io::save_safetensors(&sibling.join("adapter.evil.safetensors"), &one).unwrap();

  for hostile in ["..", "../x", "a/b", "eng/../../x", "x\\y"] {
    let selected = adapter_file_for(&dir, hostile)
      .unwrap_or_else(|e| panic!("discovery for target_lang {hostile:?} must not error: {e:?}"));
    let selected = selected.unwrap_or_else(|| {
      panic!("discovery must fall back to the in-dir eng adapter for target_lang {hostile:?}")
    });
    // The selected file is the in-dir eng adapter (the only real entry), NOT a
    // path built from the hostile target_lang.
    assert_eq!(
      selected.path.file_name().and_then(|n| n.to_str()),
      Some("adapter.eng.safetensors"),
      "a hostile target_lang {hostile:?} must select the in-dir fallback, never an interpolated path"
    );
    assert_eq!(
      selected.lang, "eng",
      "the selected language is the in-dir file's extracted <lang>, not the hostile target_lang"
    );
    // The selected path is the CANONICALIZED in-dir adapter: the returned path is
    // the validated canonical path, so it lives under the canonicalized model dir
    // — compare against the canonicalized dir because the macOS temp dir is itself
    // a symlink, e.g. /var → /private/var.
    let canon_dir = dir.canonicalize().unwrap();
    assert_eq!(
      selected.path.parent(),
      Some(canon_dir.as_path()),
      "the selected adapter must live in the (canonicalized) model dir for target_lang {hostile:?}"
    );
  }
  let _ = std::fs::remove_dir_all(&dir);
  let _ = std::fs::remove_dir_all(&sibling);
}

#[test]
fn adapter_file_for_legit_lang_selects_its_adapter_via_enumeration() {
  // The enumerate-and-match path still selects an EXACT legitimate language and
  // still falls back to the smallest when the requested language is absent —
  // unchanged behavior, now without interpolating target_lang into a path.
  let dir = std::env::temp_dir().join(format!("mlxrs_wav2vec2_enum_match_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  let one: HashMap<String, Array> =
    HashMap::from([("lm_head.bias".to_string(), filled(&[2], 0.0))]);
  crate::io::save_safetensors(&dir.join("adapter.eng.safetensors"), &one).unwrap();
  crate::io::save_safetensors(&dir.join("adapter.fra.safetensors"), &one).unwrap();
  // Exact match on fra → the fra file, lang "fra".
  let fra = adapter_file_for(&dir, "fra").unwrap().unwrap();
  assert_eq!(
    fra.path.file_name().and_then(|n| n.to_str()),
    Some("adapter.fra.safetensors")
  );
  assert_eq!(fra.lang, "fra");
  // Absent language (deu) → fallback to the lexicographically-smallest (eng).
  let fallback = adapter_file_for(&dir, "deu").unwrap().unwrap();
  assert_eq!(
    fallback.path.file_name().and_then(|n| n.to_str()),
    Some("adapter.eng.safetensors"),
    "an absent requested language falls back to the smallest in-dir adapter"
  );
  assert_eq!(fallback.lang, "eng");
  let _ = std::fs::remove_dir_all(&dir);
}
