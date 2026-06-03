//! Config-parse + validation tests for EmbeddingGemma.

use super::*;
use crate::error::Error;

/// A `config.json` body with the `google/embeddinggemma-300m` defaults.
fn full_config_json() -> &'static str {
  r#"{
    "model_type": "gemma3_text",
    "vocab_size": 262144,
    "hidden_size": 768,
    "num_hidden_layers": 24,
    "intermediate_size": 1152,
    "num_attention_heads": 3,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "num_key_value_heads": 1,
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    "query_pre_attn_scalar": 256.0,
    "sliding_window": 512,
    "sliding_window_pattern": 6,
    "max_position_embeddings": 2048
  }"#
}

#[test]
fn parses_full_config_and_validates() {
  let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
  cfg.validate().expect("validate");
  assert_eq!(cfg.model_type(), "gemma3_text");
  assert_eq!(cfg.vocab_size, 262144);
  assert_eq!(cfg.hidden_size, 768);
  assert_eq!(cfg.num_hidden_layers, 24);
  assert_eq!(cfg.intermediate_size, 1152);
  assert_eq!(cfg.num_attention_heads, 3);
  assert_eq!(cfg.head_dim, 256);
  assert_eq!(cfg.num_key_value_heads, 1);
  assert_eq!(cfg.sliding_window_pattern, 6);
}

#[test]
fn defaults_match_the_300m_checkpoint() {
  // An empty object falls back to every default — they must be the
  // `embeddinggemma-300m` values (the `*_constants_match_defaults` discipline).
  let cfg = Gemma3Config::from_json("{}").expect("empty object parses to defaults");
  assert_eq!(cfg.model_type(), "gemma3_text");
  assert_eq!(cfg.vocab_size, 262144);
  assert_eq!(cfg.hidden_size, 768);
  assert_eq!(cfg.num_hidden_layers, 24);
  assert_eq!(cfg.intermediate_size, 1152);
  assert_eq!(cfg.num_attention_heads, 3);
  assert_eq!(cfg.head_dim, 256);
  assert_eq!(cfg.num_key_value_heads, 1);
  assert_eq!(cfg.rope_theta, 1_000_000.0);
  assert_eq!(cfg.rope_local_base_freq, 10_000.0);
  assert_eq!(cfg.query_pre_attn_scalar, 256.0);
  assert_eq!(cfg.sliding_window, 512);
  assert_eq!(cfg.sliding_window_pattern, 6);
  assert_eq!(cfg.max_position_embeddings, 2048);
  cfg.validate().expect("default config validates");
}

#[test]
fn forward_compatible_ignores_unknown_keys() {
  let json = r#"{ "model_type": "gemma3_text", "hidden_size": 768, "some_future_key": [1, 2, 3] }"#;
  let cfg = Gemma3Config::from_json(json).expect("unknown keys are ignored");
  assert_eq!(cfg.hidden_size, 768);
}

#[test]
fn rejects_wrong_model_type() {
  let json = r#"{ "model_type": "bert" }"#;
  let cfg = Gemma3Config::from_json(json).expect("parse");
  let err = cfg.validate().unwrap_err();
  // pin_str surfaces an UnknownEnumValue on a mismatched architecture id.
  assert!(
    matches!(err, Error::UnknownEnumValue(_)),
    "wrong model_type must be rejected, got {err:?}"
  );
}

#[test]
fn rejects_non_positive_width() {
  let json = r#"{ "model_type": "gemma3_text", "hidden_size": 0 }"#;
  let cfg = Gemma3Config::from_json(json).expect("parse");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "hidden_size = 0 must be OutOfRange, got {err:?}"
  );
}

#[test]
fn rejects_non_divisible_head_split() {
  // num_attention_heads not divisible by num_key_value_heads.
  let json =
    r#"{ "model_type": "gemma3_text", "num_attention_heads": 3, "num_key_value_heads": 2 }"#;
  let cfg = Gemma3Config::from_json(json).expect("parse");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::DivisibilityConstraint(_) | Error::OutOfRange(_)),
    "non-divisible head split must be rejected, got {err:?}"
  );
}

#[test]
fn rejects_non_positive_query_pre_attn_scalar() {
  let json = r#"{ "model_type": "gemma3_text", "query_pre_attn_scalar": 0.0 }"#;
  let cfg = Gemma3Config::from_json(json).expect("parse");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "query_pre_attn_scalar = 0 must be OutOfRange, got {err:?}"
  );
}

#[test]
fn rejects_query_pre_attn_scalar_that_breaks_under_f32_narrowing() {
  // The SDPA scale is `(query_pre_attn_scalar as f32).powf(-0.5)`. A value that
  // is finite-and-positive in f64 but overflows f32 to +Inf (or underflows to
  // 0.0) would install an infinite / zero scale — reject the EXACT f32 value.
  // Huge f64 → +Inf in f32 → NonFiniteScalar.
  for huge in [1e39, f64::MAX] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      query_pre_attn_scalar: huge,
      ..cfg
    };
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::NonFiniteScalar(_)),
      "query_pre_attn_scalar = {huge} overflows f32 → NonFiniteScalar, got {err:?}"
    );
  }
  // Tiny positive f64 → 0.0 in f32 → OutOfRange.
  for tiny in [1e-50, f64::MIN_POSITIVE] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      query_pre_attn_scalar: tiny,
      ..cfg
    };
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "query_pre_attn_scalar = {tiny} underflows f32 to 0 → OutOfRange, got {err:?}"
    );
  }
  // A normal value still passes.
  let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
  let cfg = Gemma3Config {
    query_pre_attn_scalar: 256.0,
    ..cfg
  };
  cfg
    .validate()
    .expect("normal query_pre_attn_scalar validates");
}

#[test]
fn rejects_non_positive_or_non_finite_rms_norm_eps() {
  // rms_norm_eps feeds every RMSNorm; zero / negative is OutOfRange, NaN / Inf
  // is NonFiniteScalar. A corrupt value would otherwise yield NaNs at inference.
  for bad in [0.0, -1e-6] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rms_norm_eps: bad,
      ..cfg
    };
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "rms_norm_eps = {bad} must be OutOfRange, got {err:?}"
    );
  }
  for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rms_norm_eps: bad,
      ..cfg
    };
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::NonFiniteScalar(_)),
      "rms_norm_eps = {bad} must be NonFiniteScalar, got {err:?}"
    );
  }
  // rms_norm_eps is narrowed to f32 for the RMSNorm; a value that overflows f32
  // to +Inf (huge) or underflows to 0.0 (tiny positive) would install an
  // Inf / zero eps. Validate the narrowed value.
  for huge in [1e39, f64::MAX] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rms_norm_eps: huge,
      ..cfg
    };
    assert!(
      matches!(cfg.validate(), Err(Error::NonFiniteScalar(_))),
      "rms_norm_eps = {huge} overflows f32 → NonFiniteScalar"
    );
  }
  for tiny in [1e-50, f64::MIN_POSITIVE] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rms_norm_eps: tiny,
      ..cfg
    };
    assert!(
      matches!(cfg.validate(), Err(Error::OutOfRange(_))),
      "rms_norm_eps = {tiny} underflows f32 to 0 → OutOfRange"
    );
  }
}

#[test]
fn rejects_non_positive_or_non_finite_rope_theta() {
  // rope_theta is the global-layer RoPE base; a non-positive / non-finite base
  // yields invalid inverse frequencies.
  for bad in [0.0, -1000.0] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rope_theta: bad,
      ..cfg
    };
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "rope_theta = {bad} must be OutOfRange, got {err:?}"
    );
  }
  for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rope_theta: bad,
      ..cfg
    };
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::NonFiniteScalar(_)),
      "rope_theta = {bad} must be NonFiniteScalar, got {err:?}"
    );
  }
  // rope_theta is narrowed to f32 for the RoPE base; a value that overflows f32
  // to +Inf or underflows to 0.0 would install an invalid base. Validate the
  // narrowed value (a real Gemma rope_theta is 1e6, well within f32).
  for huge in [1e39, f64::MAX] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rope_theta: huge,
      ..cfg
    };
    assert!(
      matches!(cfg.validate(), Err(Error::NonFiniteScalar(_))),
      "rope_theta = {huge} overflows f32 → NonFiniteScalar"
    );
  }
  for tiny in [1e-50, f64::MIN_POSITIVE] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rope_theta: tiny,
      ..cfg
    };
    assert!(
      matches!(cfg.validate(), Err(Error::OutOfRange(_))),
      "rope_theta = {tiny} underflows f32 to 0 → OutOfRange"
    );
  }
}

#[test]
fn rejects_non_positive_or_non_finite_rope_local_base_freq() {
  // rope_local_base_freq is the local-layer RoPE base; same positivity /
  // finiteness contract as rope_theta.
  for bad in [0.0, -1.0] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rope_local_base_freq: bad,
      ..cfg
    };
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "rope_local_base_freq = {bad} must be OutOfRange, got {err:?}"
    );
  }
  for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rope_local_base_freq: bad,
      ..cfg
    };
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::NonFiniteScalar(_)),
      "rope_local_base_freq = {bad} must be NonFiniteScalar, got {err:?}"
    );
  }
  // rope_local_base_freq is narrowed to f32 for the local-layer RoPE base; same
  // f32 overflow / underflow contract as rope_theta.
  for huge in [1e39, f64::MAX] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rope_local_base_freq: huge,
      ..cfg
    };
    assert!(
      matches!(cfg.validate(), Err(Error::NonFiniteScalar(_))),
      "rope_local_base_freq = {huge} overflows f32 → NonFiniteScalar"
    );
  }
  for tiny in [1e-50, f64::MIN_POSITIVE] {
    let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
    let cfg = Gemma3Config {
      rope_local_base_freq: tiny,
      ..cfg
    };
    assert!(
      matches!(cfg.validate(), Err(Error::OutOfRange(_))),
      "rope_local_base_freq = {tiny} underflows f32 to 0 → OutOfRange"
    );
  }
}

#[test]
fn quantized_config_is_accepted_and_resolves_scheme_params() {
  // A quantized EmbeddingGemma bundle (e.g. mlx-community/embeddinggemma-300m-8bit)
  // declares a `quantization` block and carries packed .weight/.scales/.biases
  // weight triples. Quantized checkpoints are SUPPORTED (loaded via the shared
  // MaybeQuantizedLinear), so the config must validate, and the block must
  // resolve into a PerLayerQuantization carrying the scheme parameters. The
  // HF-style `quantization_config` key is an alias for the same field, so it
  // must be accepted + resolved identically.
  for key in ["quantization", "quantization_config"] {
    let json =
      format!(r#"{{ "model_type": "gemma3_text", "{key}": {{ "group_size": 64, "bits": 8 }} }}"#);
    let cfg = Gemma3Config::from_json(&json).expect("parse");
    cfg
      .validate()
      .expect("a quantized config is supported and must validate");
    let plq = cfg
      .quantization()
      .expect("resolve")
      .expect("a present quantization block resolves to Some");
    // The global scheme is the affine default with the declared group_size/bits;
    // any layer (no per-layer override) resolves to it.
    let q = plq
      .quantization_for("model.layers.0.self_attn.q_proj")
      .expect("global default applies to every eligible layer");
    assert_eq!(q.group_size, 64, "resolved group_size for `{key}`");
    assert_eq!(q.bits, 8, "resolved bits for `{key}`");
    assert!(
      matches!(q.mode, crate::lm::quant::QuantMode::Affine),
      "a `{{group_size, bits}}` block defaults to affine mode"
    );
  }
}

#[test]
fn quantization_does_not_short_circuit_other_field_checks() {
  // Quantized is no longer rejected, so a quantized config with an ALSO-invalid
  // field (here `hidden_size: 0`) must surface that field's typed error — the
  // width check still fires (it is not pre-empted by any quant guard).
  let json = r#"{ "model_type": "gemma3_text", "hidden_size": 0,
                  "quantization": { "group_size": 64, "bits": 4 } }"#;
  let cfg = Gemma3Config::from_json(json).expect("parse");
  assert!(
    matches!(cfg.validate(), Err(Error::OutOfRange(_))),
    "a zero hidden_size must be an OutOfRange width error (quant no longer short-circuits)"
  );
}

#[test]
fn present_but_null_quantization_is_allowed_and_resolves_to_none() {
  // A present-but-null `quantization` carries no quantization (the dense path);
  // it must NOT be rejected, and must resolve to `None` (so the loader takes the
  // dense path). A dense config with no quantization key likewise validates +
  // resolves to `None` (the `full_config_json` / empty-object cases above
  // already cover the absent-key path).
  let json = r#"{ "model_type": "gemma3_text", "quantization": null }"#;
  let cfg = Gemma3Config::from_json(json).expect("parse");
  cfg
    .validate()
    .expect("a null quantization block is the dense path and must validate");
  assert!(
    cfg.quantization().expect("resolve").is_none(),
    "a null quantization block resolves to None (dense path)"
  );

  // The absent-key dense config also resolves to None.
  let dense = Gemma3Config::from_json(full_config_json()).expect("parse");
  assert!(
    dense.quantization().expect("resolve").is_none(),
    "no quantization key resolves to None"
  );
}

#[test]
fn is_global_layer_follows_the_pattern() {
  let cfg = Gemma3Config::from_json(full_config_json()).expect("parse");
  // pattern = 6: layers 5, 11, 17, 23 are global; 0..=4, 6..=10, etc. are local.
  assert!(!cfg.is_global_layer(0));
  assert!(!cfg.is_global_layer(4));
  assert!(cfg.is_global_layer(5));
  assert!(!cfg.is_global_layer(6));
  assert!(cfg.is_global_layer(11));
  assert!(cfg.is_global_layer(23));
}

#[test]
fn dense_config_derives_four_x_hidden() {
  let dense = DenseConfig::from_hidden(768).expect("derive");
  assert_eq!(dense.hidden_size, 768);
  assert_eq!(dense.intermediate_size, 3072);
}

#[test]
fn dense_config_overflow_is_a_typed_error() {
  // hidden * 4 overflowing i32 must be a typed error, not a wrap / panic.
  let err = DenseConfig::from_hidden(i32::MAX).unwrap_err();
  assert!(
    matches!(err, Error::ArithmeticOverflow(_)),
    "hidden*4 overflow must be ArithmeticOverflow, got {err:?}"
  );
}
