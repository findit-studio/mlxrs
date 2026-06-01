//! Tests for the SigLIP2 NaFlex config parsing + validation.

use super::*;

/// A minimal but realistic `google/siglip2-base-patch16-naflex`
/// `config.json` (only the fields the port reads; unmodeled keys are
/// included to exercise the forward-compatible parse).
const BASE_CONFIG_JSON: &str = r#"{
  "model_type": "siglip2",
  "num_labels": 0,
  "some_unmodeled_top_level_key": [1, 2, 3],
  "text_config": {
    "model_type": "siglip2_text_model",
    "vocab_size": 32000,
    "max_position_embeddings": 64,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "unmodeled_text_key": "ignored"
  },
  "vision_config": {
    "model_type": "siglip2_vision_model",
    "image_size": 256,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "vision_use_head": true,
    "num_patches": 256,
    "max_num_patches": 256
  }
}"#;

#[test]
fn from_json_round_trip_base_naflex() {
  let cfg = Siglip2NaflexConfig::from_json(BASE_CONFIG_JSON).unwrap();
  assert_eq!(cfg.model_type(), "siglip2");
  assert_eq!(cfg.num_labels, 0);

  let t = &cfg.text_config;
  assert_eq!(t.model_type(), "siglip2_text_model");
  assert_eq!(t.vocab_size, 32000);
  assert_eq!(t.max_position_embeddings, 64);
  assert_eq!(t.hidden_size, 768);
  assert_eq!(t.intermediate_size, 3072);
  assert_eq!(t.num_attention_heads, 12);
  assert_eq!(t.num_hidden_layers, 12);
  assert_eq!(t.layer_norm_eps, 1e-6);
  // projection_size absent ⇒ resolves to hidden_size (the __post_init__
  // rule).
  assert_eq!(t.projection_size(), 768);

  let v = &cfg.vision_config;
  assert_eq!(v.model_type(), "siglip2_vision_model");
  assert_eq!(v.image_size, 256);
  assert_eq!(v.patch_size, 16);
  assert_eq!(v.num_channels, 3);
  assert_eq!(v.hidden_size, 768);
  assert_eq!(v.num_attention_heads, 12);
  assert_eq!(v.num_hidden_layers, 12);
  assert!(v.vision_use_head);
  assert_eq!(v.num_patches().unwrap(), 256);
  assert_eq!(v.max_num_patches(), 256);
  assert_eq!(v.patch_feature_dim().unwrap(), 3 * 16 * 16); // 768

  cfg.validate().unwrap();
}

#[test]
fn defaults_fill_absent_fields() {
  // Both towers present but empty objects → every field defaults to the
  // base-naflex value.
  let cfg =
    Siglip2NaflexConfig::from_json(r#"{ "text_config": {}, "vision_config": {} }"#).unwrap();
  assert_eq!(cfg.model_type(), "siglip2");
  assert_eq!(cfg.text_config.hidden_size, 768);
  assert_eq!(cfg.text_config.vocab_size, 32000);
  assert_eq!(cfg.vision_config.patch_size, 16);
  assert_eq!(cfg.vision_config.num_channels, 3);
  assert!(cfg.vision_config.vision_use_head);
  // num_patches absent ⇒ (image_size / patch_size)^2 = (256/16)^2 = 256.
  assert_eq!(cfg.vision_config.num_patches().unwrap(), 256);
  // max_num_patches absent ⇒ DEFAULT_NUM_PATCHES.
  assert_eq!(cfg.vision_config.max_num_patches(), DEFAULT_NUM_PATCHES);
  cfg.validate().unwrap();
}

#[test]
fn defaults_match_named_constants() {
  // The `default_*` fns are the single source of truth; pin them so a
  // drift is caught here rather than silently shipping a wrong default.
  assert_eq!(default_hidden_size(), 768);
  assert_eq!(default_intermediate_size(), 3072);
  assert_eq!(default_num_attention_heads(), 12);
  assert_eq!(default_num_hidden_layers(), 12);
  assert_eq!(default_layer_norm_eps(), 1e-6);
  assert_eq!(default_text_vocab_size(), 32000);
  assert_eq!(default_text_max_position_embeddings(), 64);
  assert_eq!(default_image_size(), 256);
  assert_eq!(default_patch_size(), 16);
  assert_eq!(default_num_channels(), 3);
  assert_eq!(DEFAULT_NUM_PATCHES, 256);
}

#[test]
fn validate_rejects_wrong_top_level_model_type() {
  let json = BASE_CONFIG_JSON.replace(r#""model_type": "siglip2""#, r#""model_type": "clip""#);
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::UnknownEnumValue(_)), "got {err}");
}

#[test]
fn validate_rejects_wrong_vision_model_type() {
  let json = BASE_CONFIG_JSON.replace(
    r#""model_type": "siglip2_vision_model""#,
    r#""model_type": "clip_vision_model""#,
  );
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::UnknownEnumValue(_)), "got {err}");
}

#[test]
fn validate_rejects_non_rgb_channels() {
  let json = BASE_CONFIG_JSON.replace(r#""num_channels": 3"#, r#""num_channels": 4"#);
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  // pin_i32 mismatch → OutOfRange.
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_hidden_not_divisible_by_heads() {
  // 770 is not divisible by 12.
  let json = BASE_CONFIG_JSON.replace(r#""hidden_size": 768"#, r#""hidden_size": 770"#);
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::DivisibilityConstraint(_)), "got {err}");
}

#[test]
fn validate_rejects_non_positive_dimension() {
  let json = BASE_CONFIG_JSON.replace(r#""vocab_size": 32000"#, r#""vocab_size": 0"#);
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_layer_count() {
  let json = BASE_CONFIG_JSON.replace(
    r#""num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "unmodeled_text_key": "ignored""#,
    r#""num_hidden_layers": 100000,
    "layer_norm_eps": 1e-6,
    "unmodeled_text_key": "ignored""#,
  );
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::CapExceeded(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_max_position_embeddings() {
  // A hostile `max_position_embeddings` sizes the text position table; the
  // cardinality cap rejects an over-cap value (here 1 << 20).
  let json = BASE_CONFIG_JSON.replace(
    r#""max_position_embeddings": 64"#,
    r#""max_position_embeddings": 1048576"#,
  );
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::CapExceeded(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_text_vocab_size() {
  // A hostile `vocab_size` sizes the token-embedding table rows; the width cap
  // (`1 << 20`) rejects an over-cap value as `OutOfRange` (positivity alone is
  // not a DoS boundary for an attacker-controlled checkpoint).
  let json = BASE_CONFIG_JSON.replace(r#""vocab_size": 32000"#, r#""vocab_size": 1048577"#);
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_text_hidden_size() {
  // A hostile text `hidden_size` names a matmul / embedding axis; the width cap
  // rejects it before any tensor is shaped. `1048584` is over-cap (and the
  // width check runs before the divisibility check, so divisibility is moot).
  let json = BASE_CONFIG_JSON.replace(
    r#""hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "unmodeled_text_key": "ignored""#,
    r#""hidden_size": 1048584,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "unmodeled_text_key": "ignored""#,
  );
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_text_intermediate_size() {
  // A hostile text `intermediate_size` names the feed-forward matmul axis.
  let json = BASE_CONFIG_JSON.replace(
    r#""intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "unmodeled_text_key": "ignored""#,
    r#""intermediate_size": 1048577,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "unmodeled_text_key": "ignored""#,
  );
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_text_projection_size() {
  // `projection_size` (the contrastive head's output width) is `Option`;
  // an explicit over-cap value must be rejected by the width cap. A standalone
  // text config keeps the other defaults realistic.
  let t = TextConfig::from_json(r#"{ "projection_size": 1048577 }"#).unwrap();
  assert_eq!(t.projection_size(), 1_048_577);
  let err = t.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_vision_hidden_size() {
  // A hostile vision `hidden_size` names the ViT matmul axis.
  let json = BASE_CONFIG_JSON.replace(
    r#""hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "vision_use_head": true"#,
    r#""hidden_size": 1048584,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "vision_use_head": true"#,
  );
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_vision_intermediate_size() {
  // A hostile vision `intermediate_size` names the ViT feed-forward axis.
  let json = BASE_CONFIG_JSON.replace(
    r#""intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "vision_use_head": true"#,
    r#""intermediate_size": 1048577,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "layer_norm_eps": 1e-6,
    "vision_use_head": true"#,
  );
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_vision_patch_size() {
  // A hostile `patch_size` (the Conv2d kernel / flattened-patch stride) is
  // bounded by the width cap directly (before the `patch_feature_dim` product).
  let json = BASE_CONFIG_JSON.replace(r#""patch_size": 16"#, r#""patch_size": 1048577"#);
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_patch_feature_dim() {
  // `patch_size` within the per-field cap but whose derived
  // `num_channels * patch_size^2` product exceeds the width cap must still be
  // rejected: 3 * 1000^2 = 3_000_000 > 1_048_576, while patch_size 1000 alone
  // passes the per-field width check. (num_patches is set explicitly to 256 so
  // the `(image_size/patch_size)^2` fallback is not exercised.)
  let json = BASE_CONFIG_JSON.replace(r#""patch_size": 16"#, r#""patch_size": 1000"#);
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn validate_rejects_oversize_image_size() {
  // A hostile `image_size` would inflate the `(image_size/patch_size)^2`
  // num_patches fallback; the cardinality cap rejects an over-cap value.
  let json = BASE_CONFIG_JSON.replace(r#""image_size": 256"#, r#""image_size": 1048576"#);
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::CapExceeded(_)), "got {err}");
}

#[test]
fn validate_rejects_negative_num_labels() {
  let json = BASE_CONFIG_JSON.replace(r#""num_labels": 0"#, r#""num_labels": -1"#);
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn malformed_json_maps_to_parse_error() {
  let err = Siglip2NaflexConfig::from_json("{ not json").unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err}");
}

#[test]
fn sub_config_from_json_parses_standalone() {
  let v =
    VisionConfig::from_json(r#"{ "model_type": "siglip2_vision_model", "num_patches": 256 }"#)
      .unwrap();
  assert_eq!(v.num_patches().unwrap(), 256);
  v.validate().unwrap();

  let t = TextConfig::from_json(r#"{ "projection_size": 512 }"#).unwrap();
  assert_eq!(t.projection_size(), 512);
  t.validate().unwrap();
}
