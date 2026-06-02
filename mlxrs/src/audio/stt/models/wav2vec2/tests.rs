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
  let config = Wav2Vec2Config::from_json(json).unwrap();
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
  let config =
    Wav2Vec2Config::from_json(r#"{"hidden_size": 768, "num_attention_heads": 12}"#).unwrap();
  assert_eq!(config.head_dim().unwrap(), 64);
}

#[test]
fn config_validate_accepts_base_960h() {
  // The base-960h defaults (feat_extract_norm == "group",
  // do_stable_layer_norm == false) are the one supported arm.
  let config = Wav2Vec2Config::from_json(r#"{"model_type": "wav2vec2"}"#).unwrap();
  assert!(config.validate().is_ok());
}

#[test]
fn config_validate_rejects_out_of_scope_arms() {
  // (a) The "layer" feature-encoder arm is out of scope -> UnknownEnumValue,
  // and the payload carries the rejected value + the supported set.
  let layer = Wav2Vec2Config::from_json(r#"{"feat_extract_norm": "layer"}"#).unwrap();
  match layer.validate() {
    Err(Error::UnknownEnumValue(p)) => {
      assert_eq!(p.value(), "layer");
      assert_eq!(p.supported(), &["group"]);
    }
    other => panic!("expected UnknownEnumValue for feat_extract_norm, got {other:?}"),
  }

  // (b) The stable-layer-norm arm is now SUPPORTED: a `do_stable_layer_norm`
  // config (otherwise default) must validate (the both-directions check that
  // the relaxation actually took effect, not just that the rejection moved).
  let stable = Wav2Vec2Config::from_json(r#"{"do_stable_layer_norm": true}"#).unwrap();
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
  let biased = Wav2Vec2Config::from_json(r#"{"conv_bias": true}"#).unwrap();
  assert!(biased.conv_bias);
  assert!(
    biased.validate().is_ok(),
    "a conv_bias config must validate (the bias is now wired)"
  );
}

#[test]
fn config_validate_accepts_large_positive_layer_count() {
  // A large positive layer count is a valid (if deep) variant, not something
  // `validate` rejects: `validate` only checks positivity (it allocates no
  // per-layer `Vec`), and the builder reserves the layer `Vec`s fallibly, so a
  // pathological count surfaces later as a typed allocation error, never a
  // magnitude cap here. (`validate` itself does no allocation, so this check is
  // cheap regardless of the count.)
  let deep = Wav2Vec2Config::from_json(r#"{"num_hidden_layers": 1000000}"#).unwrap();
  assert!(
    deep.validate().is_ok(),
    "a large positive num_hidden_layers must validate (no magnitude cap)"
  );
  // A zero / negative count is still rejected as malformed (OutOfRange) — that
  // is a positivity/soundness check, not a magnitude cap.
  let zero = Wav2Vec2Config::from_json(r#"{"num_hidden_layers": 0}"#).unwrap();
  assert!(matches!(zero.validate(), Err(Error::OutOfRange(_))));
  let negative = Wav2Vec2Config::from_json(r#"{"num_feat_extract_layers": -1}"#).unwrap();
  assert!(matches!(negative.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn config_validate_relaxes_dimensions_but_enforces_structure() {
  // A wider hidden_size is no longer rejected as a "deviation" — it is a valid
  // larger variant. The full large config (1024 hidden, 16 heads, 24 layers,
  // 4096 intermediate, stable-LN) must validate.
  let large = Wav2Vec2Config::from_json(
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
    Wav2Vec2Config::from_json(r#"{"hidden_size": 1000, "num_attention_heads": 12}"#).unwrap();
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
  let nonpos = Wav2Vec2Config::from_json(r#"{"hidden_size": 0}"#).unwrap();
  assert!(matches!(nonpos.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn config_validate_relaxes_conv_arrays_but_enforces_length_and_positivity() {
  // A conv-stack array whose values merely DIFFER from base-960h (but are
  // positive and the right length) is now accepted — different strides are a
  // different valid variant, not a rejection.
  let diff_stride = Wav2Vec2Config::from_json(r#"{"conv_stride": [5, 3, 2, 2, 2, 2, 2]}"#).unwrap();
  assert!(
    diff_stride.validate().is_ok(),
    "a conv_stride array with positive entries of the right length must validate"
  );

  // (a) An array shorter than num_feat_extract_layers is still LengthMismatch
  // (the builder would index past the end).
  let short = Wav2Vec2Config::from_json(r#"{"conv_kernel": [10, 3, 3, 3, 3, 2]}"#).unwrap();
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
  let nonpos = Wav2Vec2Config::from_json(r#"{"conv_stride": [5, 0, 2, 2, 2, 2, 2]}"#).unwrap();
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
    Wav2Vec2Config::from_json(r#"{"conv_dim": [512, 512, 512, 512, 512, 512, 512, 128]}"#).unwrap();
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
  let long_stride =
    Wav2Vec2Config::from_json(r#"{"conv_stride": [5, 2, 2, 2, 2, 2, 2, 2]}"#).unwrap();
  assert!(matches!(
    long_stride.validate(),
    Err(Error::LengthMismatch(_))
  ));
  let long_kernel =
    Wav2Vec2Config::from_json(r#"{"conv_kernel": [10, 3, 3, 3, 3, 2, 2, 2]}"#).unwrap();
  assert!(matches!(
    long_kernel.validate(),
    Err(Error::LengthMismatch(_))
  ));

  // The exact-length boundary: shrinking num_feat_extract_layers to match the
  // longer arrays makes the SAME arrays valid (proving the rule is exact
  // equality, not a one-sided floor) — 8-entry arrays with
  // num_feat_extract_layers = 8 must validate. (The matching transformer dims
  // are kept default; only the conv stack + layer count change.)
  let exact = Wav2Vec2Config::from_json(
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
  let smaller = Wav2Vec2Config::from_json(r#"{"layer_norm_eps": 1e-6}"#).unwrap();
  assert!(
    smaller.validate().is_ok(),
    "a different positive finite eps must validate (eps is not pinned)"
  );
  // Zero eps is OutOfRange.
  let zero = Wav2Vec2Config::from_json(r#"{"layer_norm_eps": 0.0}"#).unwrap();
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
  let neg = Wav2Vec2Config::from_json(r#"{"layer_norm_eps": -1e-5}"#).unwrap();
  assert!(matches!(neg.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn config_validate_rejects_non_finite_layer_norm_eps() {
  // An eps that overflows f32 to a non-finite value must never be accepted: it
  // would otherwise drive a non-finite normalization denominator. Whether the
  // over-range literal is caught at parse time (serde) or at validate time
  // (the helper's NonFiniteScalar branch, when the f64→f32 cast saturates to
  // infinity), the config must be rejected — never produce a usable model.
  match Wav2Vec2Config::from_json(r#"{"layer_norm_eps": 1e40}"#) {
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
  let relu = Wav2Vec2Config::from_json(r#"{"hidden_act": "relu"}"#).unwrap();
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
  let relu = Wav2Vec2Config::from_json(r#"{"feat_extract_activation": "relu"}"#).unwrap();
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
    let hidden = Wav2Vec2Config::from_json(&format!(r#"{{"hidden_act": "{act}"}}"#)).unwrap();
    assert!(hidden.validate().is_ok(), "hidden_act={act} must validate");
    let feat =
      Wav2Vec2Config::from_json(&format!(r#"{{"feat_extract_activation": "{act}"}}"#)).unwrap();
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
  let cfg = Wav2Vec2Config::from_json(r#"{"add_adapter": true}"#).unwrap();
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
fn config_validate_rejects_adapter_attn_dim() {
  // A set `adapter_attn_dim` adds a per-layer attention adapter (its output
  // added to the hidden states) that this port omits, so a non-null value
  // would silently drop a graph term. Only the absent (`None`) form is
  // supported; a set value is a typed InvariantViolation naming the field.
  let cfg = Wav2Vec2Config::from_json(r#"{"adapter_attn_dim": 16}"#).unwrap();
  assert_eq!(cfg.adapter_attn_dim, Some(16));
  match cfg.validate() {
    Err(Error::InvariantViolation(p)) => {
      assert!(
        p.context().contains("adapter_attn_dim"),
        "context should name adapter_attn_dim, got {:?}",
        p.context()
      );
    }
    other => panic!("expected InvariantViolation for a set adapter_attn_dim, got {other:?}"),
  }
  // An explicit `null` is equivalent to absent and validates (the base-960h
  // form), so a checkpoint that spells out the default is still accepted.
  let explicit_null = Wav2Vec2Config::from_json(r#"{"adapter_attn_dim": null}"#).unwrap();
  assert_eq!(explicit_null.adapter_attn_dim, None);
  assert!(explicit_null.validate().is_ok());
}

#[test]
fn config_validate_rejects_deviating_pad_token_id() {
  // The greedy CTC decoder hardcodes blank id 0 (`CTC_BLANK`); a checkpoint
  // declaring a different `pad_token_id` would collapse the argmax against the
  // wrong token. `validate` must reject it with a typed OutOfRange naming the
  // field + the offending and expected (0) values.
  let cfg = Wav2Vec2Config::from_json(r#"{"pad_token_id": 1}"#).unwrap();
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
fn config_validate_rejects_non_default_feat_proj_layer_norm() {
  // `feat_proj_layer_norm` is a HuBERT-only flag (HF default `true`); the wired
  // FeatureProjection unconditionally applies the LayerNorm, so the `false`
  // (no-LayerNorm) arm is not implemented this phase. A `false` value must be
  // rejected with a typed InvariantViolation naming the field, BEFORE any
  // tensor is built — never silently run the normalizing graph on a config that
  // asked for the un-normalized one.
  let cfg = Wav2Vec2Config::from_json(r#"{"feat_proj_layer_norm": false}"#).unwrap();
  assert!(!cfg.feat_proj_layer_norm);
  match cfg.validate() {
    Err(Error::InvariantViolation(p)) => {
      assert!(
        p.context().contains("feat_proj_layer_norm"),
        "context should name feat_proj_layer_norm, got {:?}",
        p.context()
      );
    }
    other => panic!("expected InvariantViolation for feat_proj_layer_norm = false, got {other:?}"),
  }
  // The default (`true`, the wired arm) validates — the HF default for HuBERT
  // and the implicit value for every wav2vec2 config — proving the gate rejects
  // only the non-default arm, not the supported one.
  let default_true = Wav2Vec2Config::from_json(r#"{"feat_proj_layer_norm": true}"#).unwrap();
  assert!(default_true.feat_proj_layer_norm);
  assert!(
    default_true.validate().is_ok(),
    "the default feat_proj_layer_norm = true (the wired arm) must validate"
  );
  // Absent (the common case: wav2vec2 configs never carry the field, HuBERT
  // defaults it true) falls back to the wired arm and validates.
  let absent = Wav2Vec2Config::from_json("{}").unwrap();
  assert!(
    absent.feat_proj_layer_norm,
    "feat_proj_layer_norm must default to true when absent"
  );
  assert!(absent.validate().is_ok());
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
  let cfg = Wav2Vec2Config::from_json(r#"{"conv_pos_batch_norm": true}"#).unwrap();
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
  let default_false = Wav2Vec2Config::from_json(r#"{"conv_pos_batch_norm": false}"#).unwrap();
  assert!(!default_false.conv_pos_batch_norm);
  assert!(
    default_false.validate().is_ok(),
    "the default conv_pos_batch_norm = false (the weight-norm arm) must validate"
  );
  // Absent falls back to the wired (weight-norm) arm and validates.
  let absent = Wav2Vec2Config::from_json("{}").unwrap();
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
    let cfg = Wav2Vec2Config::from_json(json).unwrap();
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
    r#"{"feat_extract_norm": "layer"}"#,
    r#"{"hidden_act": "relu"}"#,
    r#"{"feat_extract_activation": "relu"}"#,
    r#"{"add_adapter": true}"#,
    r#"{"adapter_attn_dim": 16}"#,
    r#"{"pad_token_id": 1}"#,
    // The HuBERT-only graph arms not wired this phase: the no-LayerNorm feature
    // projection and the batch-norm positional conv.
    r#"{"feat_proj_layer_norm": false}"#,
    r#"{"conv_pos_batch_norm": true}"#,
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
    let cfg = Wav2Vec2Config::from_json(json).unwrap();
    assert!(
      cfg.validate().is_err(),
      "an out-of-scope / invalid config must be rejected by validate(), but this one passed: {json}"
    );
  }

  // The all-default baseline must itself validate, proving the rejections are
  // caused by the override, not a baseline that already fails.
  assert!(
    Wav2Vec2Config::from_json("{}").unwrap().validate().is_ok(),
    "the all-default config must pass validate()"
  );
}

#[test]
fn defaults_match_base_960h_and_validate() {
  // The serde `default_*` fns describe `facebook/wav2vec2-base-960h`, and the
  // default config must pass validate() (the all-default checkpoint loads).
  let defaults = Wav2Vec2Config::from_json("{}").unwrap();
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

// ───────────────────────── linear helper ─────────────────────────

#[test]
fn linear_with_and_without_bias() {
  // x (1, 2) = [1, 2]; weight (out=2, in=2) = [[1,0],[0,1]] (identity);
  // y = x @ wᵀ = [1, 2]. With bias [10, 20] -> [11, 22].
  let x = Array::from_slice::<f32>(&[1.0, 2.0], &[1, 2]).unwrap();
  let w = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
  let mut y = linear(&x, &w, None).unwrap();
  assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
  let bias = Array::from_slice::<f32>(&[10.0, 20.0], &[2]).unwrap();
  let mut yb = linear(&x, &w, Some(&bias)).unwrap();
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

fn base_960h_config() -> Wav2Vec2Config {
  Wav2Vec2Config::from_json("{}").unwrap()
}

/// Assert a `from_weights` result is the shape gate's typed rejection — an
/// [`Error::LayerKeyed`] naming `key` whose inner error is an
/// [`Error::ShapePairMismatch`] — and return that inner payload for further
/// shape assertions. `Wav2Vec2Ctc` is not `Debug` (it holds `Array`s), so the
/// `Ok` arm is reported without formatting the model.
fn expect_shape_rejection(
  result: Result<Wav2Vec2Ctc>,
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
  let model = Wav2Vec2Ctc::from_weights(base_960h_config(), base_960h_weights(), Vocab::default());
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
    Wav2Vec2Ctc::from_weights(base_960h_config(), base_960h_weights(), Vocab::default()).unwrap();

  // One sample over the cap: rejected at the guard (a shape read), not run.
  let over_cap = zeros(&[(Wav2Vec2Ctc::MAX_INPUT_SAMPLES + 1) as i32]);
  let err = model.forward(&over_cap);
  assert!(
    matches!(err, Err(Error::OutOfRange(_))),
    "an over-cap waveform must be rejected with OutOfRange"
  );

  // A normal 1 s waveform (16 000 samples) is well under the cap, so the guard
  // must NOT trip: `forward` builds its lazy graph and returns without the cap
  // error (the result is left lazy — no eval — so this stays cheap).
  let one_second = zeros(&[Wav2Vec2Ctc::SAMPLE_RATE as i32]);
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
    Wav2Vec2Ctc::from_weights(base_960h_config(), base_960h_weights(), Vocab::default()).unwrap();
  let over = zeros(&[2, Wav2Vec2Ctc::MAX_INPUT_SAMPLES as i32]);
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
  let result = Wav2Vec2Ctc::from_weights(base_960h_config(), weights, Vocab::default());
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
  let result = Wav2Vec2Ctc::from_weights(base_960h_config(), weights, Vocab::default());
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
  let result = Wav2Vec2Ctc::from_weights(base_960h_config(), weights, Vocab::default());
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
  let result = Wav2Vec2Ctc::from_weights(base_960h_config(), weights, Vocab::default());
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
  let result = Wav2Vec2Ctc::from_weights(base_960h_config(), weights, Vocab::default());
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
  assert!(matches!(
    Wav2Vec2Ctc::load(&missing),
    Err(Error::MissingKey(_))
  ));
}

#[test]
fn load_errors_when_safetensors_absent() {
  // A directory with a valid config.json but no model.safetensors is a clear
  // MissingKey (sharded checkpoints are not handled by this single-file path).
  let dir = std::env::temp_dir().join(format!("mlxrs_wav2vec2_load_no_st_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(dir.join("config.json"), r#"{"model_type": "wav2vec2"}"#).unwrap();
  let err = Wav2Vec2Ctc::load(&dir.to_string_lossy());
  let _ = std::fs::remove_dir_all(&dir);
  // `Wav2Vec2Ctc` is not `Debug` (it holds `Array`s), so assert on the variant
  // without formatting the `Ok` payload.
  assert!(
    matches!(err, Err(Error::MissingKey(_))),
    "expected MissingKey for a dir with no model.safetensors"
  );
}

/// Build a complete **HF pre-sanitize** checkpoint map for `config`, with every
/// backbone key carrying `prefix` (`"wav2vec2."` / `"hubert."`) and `lm_head.*`
/// top-level — exactly the on-disk `*ForCTC` layout the PUBLIC
/// [`Wav2Vec2Ctc::load`] path feeds through [`sanitize`]. This is the faithful
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
fn hf_layout_prefixed_weights(c: &Wav2Vec2Config, prefix: &str) -> HashMap<String, Array> {
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
/// dir, then `Wav2Vec2Ctc::load` it (exercising `sanitize` on real prefixed,
/// HF-ordered keys, not `from_weights` with pre-sanitized ones) and forward to
/// the right logits shape. `config_json` selects the `model_type`; `prefix` is
/// its backbone prefix.
fn assert_load_path_strips_prefix(config_json: &str, prefix: &str) {
  let config = Wav2Vec2Config::from_json(config_json).unwrap();
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

  let loaded = Wav2Vec2Ctc::load(&dir.to_string_lossy());
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
/// [`Wav2Vec2Config`] — the exact layout `from_weights` consumes, every tensor
/// at the shape the config implies. Norm affines are ones (weight) / zeros
/// (bias); every other tensor is a small constant. Used to drive a real
/// forward over newly-unlocked (large / conv_bias / non-gelu) configs at tiny
/// synthetic dims. Shapes are written longhand (not read from the code under
/// test) so a regression in the production shape derivation cannot also shift
/// this oracle.
fn synthetic_weights(c: &Wav2Vec2Config) -> HashMap<String, Array> {
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
fn feature_out_len(c: &Wav2Vec2Config, t_in: i64) -> i64 {
  let mut l = t_in;
  for i in 0..(c.num_feat_extract_layers as usize) {
    l = (l - i64::from(c.conv_kernel[i])) / i64::from(c.conv_stride[i]) + 1;
  }
  l
}

/// A tiny synthetic config (overriding the small dims onto the JSON the test
/// supplies) so a full forward is cheap. Uses 3 feat-extract layers, 2
/// transformer layers, hidden 32 / heads 4 / inter 64, kpos 16 / groups 4.
fn tiny_config_json(extra: &str) -> Wav2Vec2Config {
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
  Wav2Vec2Config::from_json(&json).unwrap()
}

/// Forward a tiny model over a `(1, t_in)` waveform and assert the logits shape
/// is `(1, T', vocab)` where `T'` is the analytic feature-encoder output length
/// — exercising the full build + forward graph for a newly-unlocked config.
fn assert_forward_shape(config: Wav2Vec2Config, t_in: i32) {
  let expected_t = feature_out_len(&config, i64::from(t_in));
  assert!(expected_t > 0, "choose a longer input: T' = {expected_t}");
  let vocab = config.vocab_size as usize;
  let weights = synthetic_weights(&config);
  let model = Wav2Vec2Ctc::from_weights(config, weights, Vocab::default())
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
  // layers / 4096 intermediate, stable-LN) must parse and validate (the
  // feat_extract_norm stays "group" here — the "layer" arm is out of scope).
  let json = r#"{
    "model_type": "wav2vec2",
    "hidden_size": 1024, "num_attention_heads": 16, "num_hidden_layers": 24,
    "intermediate_size": 4096, "vocab_size": 32,
    "do_stable_layer_norm": true,
    "num_conv_pos_embeddings": 128, "num_conv_pos_embedding_groups": 16
  }"#;
  let config = Wav2Vec2Config::from_json(json).unwrap();
  assert_eq!(config.hidden_size, 1024);
  assert_eq!(config.num_hidden_layers, 24);
  assert!(config.do_stable_layer_norm);
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
  match Wav2Vec2Ctc::from_weights(config, weights, Vocab::default()) {
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
      Wav2Vec2Ctc::from_weights(
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
  let mut post_logits = Wav2Vec2Ctc::from_weights(
    post,
    synthetic_weights(&tiny_config_json("")),
    Vocab::default(),
  )
  .unwrap()
  .forward(&waveform)
  .unwrap();
  let mut stable_logits = Wav2Vec2Ctc::from_weights(
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
    let attn = Attention {
      q_weight: mk(&[hidden, hidden], 0.3),
      q_bias: mk(&[hidden], 0.1),
      k_weight: mk(&[hidden, hidden], 0.3),
      k_bias: mk(&[hidden], 0.1),
      v_weight: mk(&[hidden, hidden], 0.3),
      v_bias: mk(&[hidden], 0.1),
      out_weight: mk(&[hidden, hidden], 0.3),
      out_bias: mk(&[hidden], 0.1),
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
    let model = Wav2Vec2Ctc::from_weights(config, weights, Vocab::default()).unwrap();
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
    let model = Wav2Vec2Ctc::from_weights(config, weights, Vocab::default()).unwrap();
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
