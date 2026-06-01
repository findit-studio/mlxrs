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
fn vocab_rejects_enormous_id_before_allocating() {
  // A single enormous id (here i64::MAX) would, if used as a dense-table
  // length, drive a multi-exabyte `vec![None; len]` and abort the process.
  // It must instead be rejected with a typed CapExceeded — and the observed
  // value carried in the payload must equal the offending id, computed here
  // independently of the implementation.
  let json = format!(r#"{{"<pad>": 0, "X": {}}}"#, i64::MAX);
  match Vocab::from_json(&json) {
    Err(Error::CapExceeded(p)) => {
      assert_eq!(p.observed(), i64::MAX as u64);
      assert_eq!(p.cap(), (1u64 << 20));
    }
    other => panic!("expected CapExceeded for an enormous id, got {other:?}"),
  }
  // An id one past the cap is rejected; the cap itself (2^20) is accepted.
  let over = format!(r#"{{"A": {}}}"#, (1i64 << 20) + 1);
  assert!(matches!(
    Vocab::from_json(&over),
    Err(Error::CapExceeded(_))
  ));
  let at_cap = format!(r#"{{"A": {}}}"#, 1i64 << 20);
  let vocab = Vocab::from_json(&at_cap).unwrap();
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
  assert!((config.layer_norm_eps - 1e-5).abs() < 1e-12);
  assert!(!config.do_stable_layer_norm);
  assert!(!config.conv_bias);
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
fn config_validate_rejects_unsupported_arms() {
  // (a) The pre-norm stable-layer-norm arm is not ported -> InvariantViolation.
  let stable = Wav2Vec2Config::from_json(r#"{"do_stable_layer_norm": true}"#).unwrap();
  match stable.validate() {
    Err(Error::InvariantViolation(p)) => {
      assert!(p.context().contains("do_stable_layer_norm"));
    }
    other => panic!("expected InvariantViolation for stable layer norm, got {other:?}"),
  }

  // (b) A non-"group" feat_extract_norm is not ported -> UnknownEnumValue,
  // and the payload carries the rejected value + the supported set.
  let layer = Wav2Vec2Config::from_json(r#"{"feat_extract_norm": "layer"}"#).unwrap();
  match layer.validate() {
    Err(Error::UnknownEnumValue(p)) => {
      assert_eq!(p.value(), "layer");
      assert_eq!(p.supported(), &["group"]);
    }
    other => panic!("expected UnknownEnumValue for feat_extract_norm, got {other:?}"),
  }
}

#[test]
fn config_validate_rejects_conv_bias() {
  // The port's ConvLayer stores no bias and `forward` adds none, so a
  // `conv_bias == true` checkpoint would load (its bias tensors silently
  // dropped) and run wrong. `validate` must reject it BEFORE any tensor is
  // built, with a typed InvariantViolation naming the field.
  let biased = Wav2Vec2Config::from_json(r#"{"conv_bias": true}"#).unwrap();
  assert!(biased.conv_bias);
  match biased.validate() {
    Err(Error::InvariantViolation(p)) => {
      assert!(
        p.context().contains("conv_bias"),
        "context should name conv_bias, got {:?}",
        p.context()
      );
    }
    other => panic!("expected InvariantViolation for conv_bias = true, got {other:?}"),
  }
}

#[test]
fn config_validate_rejects_oversized_count() {
  // An unbounded count must not pass validate only to blow up later at the
  // per-layer allocation loop. Pinning num_hidden_layers to its base-960h
  // value (12) bounds it: an oversized count is rejected here, before the
  // builder's `Vec::with_capacity(num_layers)` / weight-fetch loop. The
  // OutOfRange payload names the offending value and the expected one.
  let oversized = Wav2Vec2Config::from_json(r#"{"num_hidden_layers": 1000000}"#).unwrap();
  match oversized.validate() {
    Err(Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("num_hidden_layers"),
        "context should name num_hidden_layers, got {:?}",
        p.context()
      );
      // The value carries both the offending count and the base-960h expectation.
      assert!(
        p.value().contains("1000000") && p.value().contains("12"),
        "value should name the offending count and the expected one, got {:?}",
        p.value()
      );
    }
    other => panic!("expected OutOfRange for an oversized num_hidden_layers, got {other:?}"),
  }
}

#[test]
fn config_validate_rejects_deviating_dimension() {
  // A scalar architecture field that deviates from base-960h is a different
  // (unsupported) architecture — the port reads weight shapes from the
  // checkpoint, so it would load-and-run silently wrong. Reject with a typed
  // OutOfRange naming the field + value.
  let wide = Wav2Vec2Config::from_json(r#"{"hidden_size": 1024}"#).unwrap();
  match wide.validate() {
    Err(Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("hidden_size"),
        "context should name hidden_size, got {:?}",
        p.context()
      );
      assert!(
        p.value().contains("1024") && p.value().contains("768"),
        "value should name the offending dim and the expected one, got {:?}",
        p.value()
      );
    }
    other => panic!("expected OutOfRange for a deviating hidden_size, got {other:?}"),
  }
}

#[test]
fn config_validate_rejects_deviating_conv_array() {
  // A conv-stack array that deviates from base-960h is rejected: a wrong
  // length is LengthMismatch; a wrong element is OutOfRange naming the index +
  // value + expectation.
  //
  // (a) Deviating element (same length, one stride changed 2 -> 3 at index 1).
  let bad_elem = Wav2Vec2Config::from_json(r#"{"conv_stride": [5, 3, 2, 2, 2, 2, 2]}"#).unwrap();
  match bad_elem.validate() {
    Err(Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("conv_stride"),
        "context should name conv_stride, got {:?}",
        p.context()
      );
      // index 1, offending value 3, expected 2.
      assert!(
        p.value().contains("element 1")
          && p.value().contains("= 3")
          && p.value().contains("expected 2"),
        "value should name the index, value, and expectation, got {:?}",
        p.value()
      );
    }
    other => panic!("expected OutOfRange for a deviating conv_stride element, got {other:?}"),
  }

  // (b) Wrong length (a 6-element conv_kernel instead of 7) -> LengthMismatch.
  let bad_len = Wav2Vec2Config::from_json(r#"{"conv_kernel": [10, 3, 3, 3, 3, 2]}"#).unwrap();
  match bad_len.validate() {
    Err(Error::LengthMismatch(p)) => {
      assert!(
        p.context().contains("conv_kernel"),
        "context should name conv_kernel, got {:?}",
        p.context()
      );
      assert_eq!(p.expected(), 7);
      assert_eq!(p.actual(), 6);
    }
    other => panic!("expected LengthMismatch for a wrong-length conv_kernel, got {other:?}"),
  }
}

#[test]
fn config_validate_rejects_deviating_layer_norm_eps() {
  // `layer_norm_eps` is shared by every LayerNorm and the L0 GroupNorm, so a
  // deviating value silently runs a numerically different graph. `validate`
  // must reject it with a typed OutOfRange naming the field + the offending
  // and expected values (the helper widens the f32 field to f64 for the
  // compare, so the message carries the f64 forms).
  let bad = Wav2Vec2Config::from_json(r#"{"layer_norm_eps": 1e-6}"#).unwrap();
  // The parsed field is the deviating value.
  assert!((bad.layer_norm_eps - 1e-6).abs() < 1e-12);
  match bad.validate() {
    Err(Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("layer_norm_eps"),
        "context should name layer_norm_eps, got {:?}",
        p.context()
      );
      // The value names both the offending eps and the expected base-960h one.
      assert!(
        p.value().contains("expected"),
        "value should name the expected eps, got {:?}",
        p.value()
      );
    }
    other => panic!("expected OutOfRange for a deviating layer_norm_eps, got {other:?}"),
  }
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
fn config_validate_rejects_deviating_hidden_act() {
  // The port hardcodes GELU in every block, so a config whose `hidden_act`
  // names a different activation would silently run a different graph.
  // `validate` must reject it with a typed UnknownEnumValue carrying the
  // offending value and the supported set.
  let relu = Wav2Vec2Config::from_json(r#"{"hidden_act": "relu"}"#).unwrap();
  assert_eq!(relu.hidden_act, "relu");
  match relu.validate() {
    Err(Error::UnknownEnumValue(p)) => {
      assert_eq!(p.value(), "relu");
      assert_eq!(p.supported(), &["gelu"]);
    }
    other => panic!("expected UnknownEnumValue for a deviating hidden_act, got {other:?}"),
  }
}

#[test]
fn base_960h_constants_match_defaults() {
  // The validate() pin-constants and the serde `default_*` fns must agree:
  // a default-constructed config (the base-960h checkpoint) must pass
  // validate(). This guards against the two sources of truth drifting — if
  // they did, the all-defaults config below would fail to validate.
  let defaults = Wav2Vec2Config::from_json("{}").unwrap();
  assert!(
    defaults.validate().is_ok(),
    "the all-default (base-960h) config must pass the validate() pins"
  );
  // Spot-check the exact base-960h values the pins enforce.
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
  assert!((defaults.layer_norm_eps - 1e-5).abs() < 1e-12);
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
