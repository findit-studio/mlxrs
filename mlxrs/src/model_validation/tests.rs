//! Unit tests for the shared model-validation toolkit: a happy path plus
//! every typed-error path for each helper, asserting the exact [`Error`]
//! variant (not merely `is_err`) and, where load-bearing, the payload fields.

use std::collections::HashMap;

use crate::{error::Error, model_validation::*};

// ──────────────────────────── 1. field pinning ────────────────────────────

#[test]
fn pin_i32_accepts_match_rejects_mismatch() {
  assert!(pin_i32("hidden_size", 768, 768).is_ok());
  let err = pin_i32("hidden_size", 512, 768).unwrap_err();
  assert!(err.is_out_of_range(), "got {err:?}");
  match err {
    Error::OutOfRange(p) => {
      assert_eq!(p.context(), "hidden_size");
      // The offending value AND the expected value are both surfaced.
      assert!(p.value().contains("512"));
      assert!(p.value().contains("768"));
    }
    other => panic!("expected OutOfRange, got {other:?}"),
  }
}

#[test]
fn pin_usize_accepts_match_rejects_mismatch() {
  assert!(pin_usize("num_layers", 12, 12).is_ok());
  let err = pin_usize("num_layers", 13, 12).unwrap_err();
  match err {
    Error::OutOfRange(p) => {
      assert_eq!(p.context(), "num_layers");
      assert!(p.value().contains("13"));
    }
    other => panic!("expected OutOfRange, got {other:?}"),
  }
}

#[test]
fn pin_bool_accepts_match_rejects_mismatch_both_directions() {
  assert!(pin_bool("conv_bias", false, false).is_ok());
  assert!(pin_bool("do_stable_layer_norm", true, true).is_ok());

  // expected=false, actual=true → "must be false"
  match pin_bool("conv_bias", true, false).unwrap_err() {
    Error::InvariantViolation(p) => {
      assert_eq!(p.context(), "conv_bias");
      assert_eq!(p.requirement(), "must be false");
    }
    other => panic!("expected InvariantViolation, got {other:?}"),
  }
  // expected=true, actual=false → "must be true"
  match pin_bool("do_stable_layer_norm", false, true).unwrap_err() {
    Error::InvariantViolation(p) => {
      assert_eq!(p.requirement(), "must be true");
    }
    other => panic!("expected InvariantViolation, got {other:?}"),
  }
}

#[test]
fn pin_str_accepts_allowed_rejects_others() {
  // single pinned value
  assert!(pin_str("model_type", "wav2vec2", &["wav2vec2"]).is_ok());
  // multi-arm allowed set — any member matches
  assert!(pin_str("feat_extract_norm", "group", &["group", "layer"]).is_ok());
  assert!(pin_str("feat_extract_norm", "layer", &["group", "layer"]).is_ok());

  match pin_str("model_type", "hubert", &["wav2vec2"]).unwrap_err() {
    Error::UnknownEnumValue(p) => {
      assert_eq!(p.type_name(), "model_type");
      assert_eq!(p.value(), "hubert");
      assert_eq!(p.supported(), &["wav2vec2"]);
    }
    other => panic!("expected UnknownEnumValue, got {other:?}"),
  }
  // not in a multi-arm set
  assert!(pin_str("feat_extract_norm", "instance", &["group", "layer"]).is_err());
}

#[test]
fn pin_f64_accepts_exact_rejects_mismatch_and_non_finite() {
  assert!(pin_f64("layer_norm_eps", 1e-5, 1e-5).is_ok());

  // finite but different → OutOfRange
  match pin_f64("layer_norm_eps", 1e-6, 1e-5).unwrap_err() {
    Error::OutOfRange(p) => assert_eq!(p.context(), "layer_norm_eps"),
    other => panic!("expected OutOfRange, got {other:?}"),
  }

  // non-finite is rejected BEFORE the equality compare → NonFiniteScalar
  for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
    match pin_f64("layer_norm_eps", bad, 1e-5).unwrap_err() {
      Error::NonFiniteScalar(p) => assert_eq!(p.context(), "layer_norm_eps"),
      other => panic!("expected NonFiniteScalar for {bad}, got {other:?}"),
    }
  }
}

#[test]
fn pin_i32_slice_accepts_match_rejects_length_then_element() {
  let expected = [5, 2, 2, 2, 2, 2, 2];
  assert!(pin_i32_slice("conv_stride", &expected, &expected).is_ok());

  // wrong length → LengthMismatch (checked before per-element)
  match pin_i32_slice("conv_stride", &[5, 2, 2], &expected).unwrap_err() {
    Error::LengthMismatch(p) => {
      assert_eq!(p.context(), "conv_stride");
      assert_eq!(p.expected(), 7);
      assert_eq!(p.actual(), 3);
    }
    other => panic!("expected LengthMismatch, got {other:?}"),
  }

  // right length, one deviating element → OutOfRange naming the index
  let mut wrong = expected;
  wrong[3] = 9;
  match pin_i32_slice("conv_stride", &wrong, &expected).unwrap_err() {
    Error::OutOfRange(p) => {
      assert!(p.value().contains("element 3"));
      assert!(p.value().contains('9'));
    }
    other => panic!("expected OutOfRange, got {other:?}"),
  }

  // empty == empty is a vacuous match
  assert!(pin_i32_slice("empty", &[], &[]).is_ok());
}

// ──────────────────────────────── 2. bounds ───────────────────────────────

#[test]
fn require_positive_accepts_positive_rejects_zero_and_negative() {
  assert!(require_positive("hidden_size", 1).is_ok());
  assert!(require_positive("hidden_size", i32::MAX).is_ok());
  for bad in [0, -1, i32::MIN] {
    match require_positive("hidden_size", bad).unwrap_err() {
      Error::OutOfRange(p) => {
        assert_eq!(p.context(), "hidden_size");
        assert!(p.value().contains(&bad.to_string()));
      }
      other => panic!("expected OutOfRange for {bad}, got {other:?}"),
    }
  }
}

#[test]
fn require_in_range_accepts_endpoints_rejects_outside() {
  assert!(require_in_range("groups", 1, 1, 64).is_ok()); // low endpoint
  assert!(require_in_range("groups", 64, 1, 64).is_ok()); // high endpoint
  assert!(require_in_range("groups", 16, 1, 64).is_ok());

  match require_in_range("groups", 0, 1, 64).unwrap_err() {
    Error::OutOfRange(p) => assert!(p.value().contains('0')),
    other => panic!("expected OutOfRange, got {other:?}"),
  }
  match require_in_range("groups", 65, 1, 64).unwrap_err() {
    Error::OutOfRange(p) => assert!(p.value().contains("65")),
    other => panic!("expected OutOfRange, got {other:?}"),
  }
  // negative band is supported
  assert!(require_in_range("offset", -3, -5, -1).is_ok());
  assert!(require_in_range("offset", -6, -5, -1).is_err());
}

#[test]
fn require_cardinality_accepts_within_cap_rejects_nonpositive_and_overflow() {
  assert!(require_cardinality("num_hidden_layers", 12, 4096).is_ok());
  assert!(require_cardinality("num_hidden_layers", 4096, 4096).is_ok()); // == cap

  // non-positive → OutOfRange
  for bad in [0i64, -1, i64::MIN] {
    match require_cardinality("num_hidden_layers", bad, 4096).unwrap_err() {
      Error::OutOfRange(p) => assert_eq!(p.context(), "num_hidden_layers"),
      other => panic!("expected OutOfRange for {bad}, got {other:?}"),
    }
  }

  // over cap → CapExceeded carrying cap + observed; a near-i64::MAX value must
  // not wrap.
  match require_cardinality("num_hidden_layers", 1 << 30, 4096).unwrap_err() {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap(), 4096);
      assert_eq!(p.observed(), 1 << 30);
      assert_eq!(p.cap_name(), "num_hidden_layers");
    }
    other => panic!("expected CapExceeded, got {other:?}"),
  }
  match require_cardinality("shards", i64::MAX, 4096).unwrap_err() {
    Error::CapExceeded(p) => assert_eq!(p.observed(), i64::MAX as u64),
    other => panic!("expected CapExceeded, got {other:?}"),
  }
}

#[test]
fn require_divisible_accepts_multiple_rejects_remainder_and_bad_divisor() {
  assert!(require_divisible("hidden_size", 768, "num_heads", 12).is_ok());
  assert!(require_divisible("a", 0, "b", 4).is_ok()); // 0 is divisible by anything

  // non-multiple → DivisibilityConstraint carrying both operands
  match require_divisible("hidden_size", 768, "num_heads", 5).unwrap_err() {
    Error::DivisibilityConstraint(p) => {
      assert_eq!(p.name_dividend(), "hidden_size");
      assert_eq!(p.name_divisor(), "num_heads");
      assert_eq!(p.dividend(), 768);
      assert_eq!(p.divisor(), 5);
    }
    other => panic!("expected DivisibilityConstraint, got {other:?}"),
  }

  // zero / negative divisor is rejected FIRST as OutOfRange (no `% 0` panic)
  for bad in [0, -3] {
    match require_divisible("hidden_size", 768, "num_heads", bad).unwrap_err() {
      Error::OutOfRange(p) => assert_eq!(p.context(), "num_heads"),
      other => panic!("expected OutOfRange for divisor {bad}, got {other:?}"),
    }
  }
}

#[test]
fn require_even_accepts_even_rejects_odd_incl_negative() {
  assert!(require_even("head_dim", 0).is_ok());
  assert!(require_even("head_dim", 64).is_ok());
  assert!(require_even("head_dim", -4).is_ok());

  match require_even("head_dim", 65).unwrap_err() {
    Error::OutOfRange(p) => {
      assert_eq!(p.context(), "head_dim");
      assert!(p.value().contains("65"));
    }
    other => panic!("expected OutOfRange, got {other:?}"),
  }
  // negative odd is also rejected
  assert!(require_even("head_dim", -3).is_err());
}

// ─────────────────────────── 3. checked arithmetic ─────────────────────────

#[test]
fn checked_mul_returns_product_or_overflow() {
  assert_eq!(
    checked_mul("embed", "heads", 12, "head_dim", 64).unwrap(),
    768
  );
  assert_eq!(checked_mul("z", "a", 0, "b", i32::MAX).unwrap(), 0);
  assert_eq!(checked_mul("neg", "a", -2, "b", 3).unwrap(), -6);

  match checked_mul("embed", "heads", i32::MAX, "head_dim", 2).unwrap_err() {
    Error::ArithmeticOverflow(p) => {
      assert_eq!(p.context(), "embed");
      assert_eq!(p.op_type(), "i32");
      let ops = p.operands();
      assert_eq!(ops.len(), 2);
      assert_eq!(ops[0].0, "heads");
      assert_eq!(ops[0].1, i32::MAX as i64 as u64);
      assert_eq!(ops[1], ("head_dim", 2));
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
}

#[test]
fn checked_add_returns_sum_or_overflow() {
  assert_eq!(
    checked_add("vocab", "base", 32000, "added", 100).unwrap(),
    32100
  );
  match checked_add("vocab", "base", i32::MAX, "added", 1).unwrap_err() {
    Error::ArithmeticOverflow(p) => {
      assert_eq!(p.context(), "vocab");
      assert_eq!(p.operands().len(), 2);
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
  // negative-side overflow is caught too
  assert!(checked_add("x", "a", i32::MIN, "b", -1).is_err());
}

// ─────────────────────────── 4. fallible allocation ────────────────────────

#[test]
fn reserve_or_error_reserves_vec_capacity() {
  let mut v: Vec<u32> = Vec::new();
  assert!(reserve_or_error(&mut v, "layers", 12).is_ok());
  assert!(v.capacity() >= 12);
  assert!(v.is_empty()); // reservation only — no elements added
  // zero reservation is a no-op success
  assert!(reserve_or_error(&mut v, "layers", 0).is_ok());
}

#[test]
fn reserve_or_error_reserves_hashmap_capacity() {
  let mut m: HashMap<String, u32> = HashMap::new();
  assert!(reserve_or_error(&mut m, "weights", 32).is_ok());
  assert!(m.capacity() >= 32);
}

#[test]
fn reserve_or_error_oversize_request_is_typed_alloc_failure() {
  // A request whose byte size overflows `isize` cannot be satisfied; the
  // collection's own `try_reserve_exact` returns `Err`, which must map to a
  // typed `AllocFailure` (NOT an abort).
  let mut v: Vec<u64> = Vec::new();
  match reserve_or_error(&mut v, "samples", usize::MAX) {
    Err(Error::AllocFailure(p)) => {
      assert_eq!(p.item(), "samples");
      assert_eq!(p.count(), usize::MAX as u64);
    }
    other => panic!("expected AllocFailure, got {other:?}"),
  }
}

// ────────────────────────────── 5. key collision ───────────────────────────

#[test]
fn insert_unique_inserts_first_rejects_duplicate() {
  let mut m: HashMap<String, u32> = HashMap::new();
  assert!(insert_unique(&mut m, "encoder.weight".to_string(), 1, "sanitize").is_ok());
  assert!(insert_unique(&mut m, "decoder.weight".to_string(), 2, "sanitize").is_ok());

  match insert_unique(&mut m, "encoder.weight".to_string(), 99, "sanitize").unwrap_err() {
    Error::KeyCollision(p) => {
      assert_eq!(p.context(), "sanitize");
      assert_eq!(p.key(), "encoder.weight");
    }
    other => panic!("expected KeyCollision, got {other:?}"),
  }
  // the original value is preserved; the colliding value is dropped
  assert_eq!(m["encoder.weight"], 1);
  assert_eq!(m.len(), 2);
}

// ──────────────────── 6. config-gated optional weight ──────────────────────

#[test]
fn require_if_present_enforces_flag_contract() {
  // agreeing cases
  assert!(require_if_present("conv_bias", true, "conv.bias", true).is_ok());
  assert!(require_if_present("conv_bias", false, "conv.bias", false).is_ok());

  // flag true but absent → MissingKey
  match require_if_present("conv_bias", true, "conv.bias", false).unwrap_err() {
    Error::MissingKey(p) => {
      assert_eq!(p.context(), "conv_bias");
      assert_eq!(p.key(), "conv.bias");
    }
    other => panic!("expected MissingKey, got {other:?}"),
  }

  // flag false but present → KeyCollision
  match require_if_present("conv_bias", false, "conv.bias", true).unwrap_err() {
    Error::KeyCollision(p) => {
      assert_eq!(p.context(), "conv_bias");
      assert_eq!(p.key(), "conv.bias");
    }
    other => panic!("expected KeyCollision, got {other:?}"),
  }
}

#[test]
fn take_if_drains_required_present_and_absent_optional() {
  // flag true + present → Some(value), drained from the map
  let mut m: HashMap<String, u32> = HashMap::from([("conv.bias".to_string(), 7)]);
  assert_eq!(
    take_if(&mut m, "conv_bias", true, "conv.bias").unwrap(),
    Some(7)
  );
  assert!(m.is_empty());

  // flag false + absent → None, map untouched
  let mut empty: HashMap<String, u32> = HashMap::new();
  assert_eq!(
    take_if(&mut empty, "conv_bias", false, "conv.bias").unwrap(),
    None
  );
}

#[test]
fn take_if_rejects_required_absent_and_forbidden_present() {
  // flag true + absent → MissingKey
  let mut empty: HashMap<String, u32> = HashMap::new();
  match take_if(&mut empty, "conv_bias", true, "conv.bias").unwrap_err() {
    Error::MissingKey(p) => assert_eq!(p.key(), "conv.bias"),
    other => panic!("expected MissingKey, got {other:?}"),
  }

  // flag false + present → KeyCollision; the forbidden tensor IS removed so a
  // later strict "map fully drained" check still passes.
  let mut m: HashMap<String, u32> = HashMap::from([("conv.bias".to_string(), 7)]);
  match take_if(&mut m, "conv_bias", false, "conv.bias").unwrap_err() {
    Error::KeyCollision(p) => assert_eq!(p.key(), "conv.bias"),
    other => panic!("expected KeyCollision, got {other:?}"),
  }
  assert!(m.is_empty());
}
