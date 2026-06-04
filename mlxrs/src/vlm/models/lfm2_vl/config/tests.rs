//! Tests for the LFM2.5-VL config structs (`VisionConfig` / `ModelConfig`):
//! reference defaults, JSON parse, and the `validate()` accept / reject gates.

use super::*;
use crate::error::Error;

/// A full `config.json` body matching `LiquidAI/LFM2.5-VL-450M-MLX-8bit`'s
/// shape (the fields the port reads; unmodeled keys are ignored).
fn full_config_json() -> String {
  r#"{
    "model_type": "lfm2-vl",
    "downsample_factor": 2,
    "image_token_index": 396,
    "projector_hidden_size": 2560,
    "projector_use_layernorm": true,
    "projector_bias": true,
    "vision_feature_layer": -1,
    "max_num_patches": 1024,
    "tile_size": 512,
    "eos_token_id": 7,
    "quantization": {"group_size": 64, "bits": 8},
    "text_config": {
      "model_type": "lfm2",
      "hidden_size": 1024,
      "num_hidden_layers": 16,
      "num_attention_heads": 16,
      "num_key_value_heads": 8,
      "vocab_size": 65536,
      "conv_L_cache": 3,
      "full_attn_idxs": [2, 5, 8, 10, 12, 14],
      "layer_types": ["conv","conv","full_attention"]
    },
    "vision_config": {
      "model_type": "lfm2_vl",
      "hidden_size": 768,
      "intermediate_size": 3072,
      "num_hidden_layers": 12,
      "num_attention_heads": 12,
      "num_channels": 3,
      "image_size": 224,
      "patch_size": 16,
      "num_patches": 256,
      "layer_norm_eps": 1e-6
    }
  }"#
    .to_string()
}

#[test]
fn vision_config_defaults_match_reference() {
  // An empty object falls back to every reference default (config.py's
  // VisionConfig).
  let cfg = VisionConfig::from_json("{}").unwrap();
  assert_eq!(cfg.model_type(), "lfm2_vl");
  assert_eq!(cfg.hidden_size, 768);
  assert_eq!(cfg.intermediate_size, 3072);
  assert_eq!(cfg.num_hidden_layers, 12);
  assert_eq!(cfg.num_attention_heads, 12);
  assert_eq!(cfg.num_channels, 3);
  assert_eq!(cfg.image_size, 224);
  assert_eq!(cfg.patch_size, 16);
  assert_eq!(cfg.num_patches, 256);
  assert!((cfg.layer_norm_eps - 1e-6).abs() < 1e-12);
  // The default vision activation (`config.py:65`) is the tanh GELU the MLP runs.
  assert_eq!(cfg.hidden_act, "gelu_pytorch_tanh");
  // `3 * 16^2 = 768` flattened-patch width.
  assert_eq!(cfg.patch_feature_dim().unwrap(), 768);
  cfg.validate().unwrap();
}

#[test]
fn vision_config_default_hidden_act_validates() {
  // The default `hidden_act` (`gelu_pytorch_tanh`, the tanh GELU the MLP
  // implements) clears `validate`.
  let cfg = VisionConfig::from_json(r#"{"hidden_act": "gelu_pytorch_tanh"}"#).unwrap();
  cfg.validate().unwrap();
  assert_eq!(cfg.hidden_act, "gelu_pytorch_tanh");
}

#[test]
fn vision_config_rejects_non_tanh_hidden_act() {
  // The vision MLP forward (`vision.rs`) hard-codes the tanh GELU
  // (`nn.GELU(approx="precise")` → `gelu_approx`, `vision.py:67`); a checkpoint
  // declaring any other `hidden_act` must fail loudly rather than silently
  // running the tanh GELU under a mismatched declared activation. `gelu`
  // (the erf GELU) is the most adversarial near-miss — it must still be rejected.
  for bad in [
    r#"{"hidden_act": "gelu"}"#,
    r#"{"hidden_act": "gelu_new"}"#,
    r#"{"hidden_act": "relu"}"#,
    r#"{"hidden_act": "silu"}"#,
  ] {
    let cfg = VisionConfig::from_json(bad).unwrap();
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::UnknownEnumValue(_)),
      "expected UnknownEnumValue for {bad}, got {err}"
    );
  }
}

#[test]
fn vision_config_accepts_siglip2_model_type() {
  // vision.py's VisionModel guard accepts both ids.
  let cfg = VisionConfig::from_json(r#"{"model_type": "siglip2_vision_model"}"#).unwrap();
  cfg.validate().unwrap();
  assert_eq!(cfg.model_type(), "siglip2_vision_model");
}

#[test]
fn vision_config_rejects_wrong_model_type() {
  let cfg = VisionConfig::from_json(r#"{"model_type": "clip_vision_model"}"#).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::UnknownEnumValue(_)), "got {err}");
}

#[test]
fn vision_config_rejects_non_divisible_head_split() {
  // hidden_size not divisible by num_attention_heads.
  let cfg = VisionConfig::from_json(r#"{"hidden_size": 768, "num_attention_heads": 7}"#).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::DivisibilityConstraint(_)), "got {err}");
}

#[test]
fn vision_config_rejects_non_square_num_patches() {
  // 255 is not a perfect square ⇒ the trained position grid is not square.
  let cfg = VisionConfig::from_json(r#"{"num_patches": 255}"#).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn vision_config_rejects_nonpositive_dim() {
  let cfg = VisionConfig::from_json(r#"{"hidden_size": 0}"#).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn model_config_full_parses_and_validates() {
  let cfg = ModelConfig::from_json(&full_config_json()).unwrap();
  assert_eq!(cfg.model_type(), "lfm2-vl");
  assert_eq!(cfg.downsample_factor, 2);
  assert_eq!(cfg.image_token_index, 396);
  assert_eq!(cfg.projector_hidden_size, 2560);
  assert!(cfg.projector_use_layernorm);
  assert!(cfg.projector_bias);
  assert_eq!(cfg.vision_feature_layer, -1);
  assert_eq!(cfg.max_num_patches, 1024);
  assert_eq!(cfg.tile_size, 512);
  assert_eq!(cfg.eos_token_id, 7);
  assert!(cfg.quantization().is_some(), "8-bit block present");
  // Both towers parsed.
  assert_eq!(cfg.vision_config.hidden_size, 768);
  assert_eq!(cfg.text_config.hidden_size, 1024);
  cfg.validate().unwrap();
  // `-1` keeps all 12 vision layers.
  assert_eq!(cfg.vision_feature_layers_kept().unwrap(), 12);
}

#[test]
fn model_config_defaults_fill_top_level_fields() {
  // Only the two required tower configs are supplied; every top-level field
  // falls back to its reference default.
  let json = r#"{"text_config": {}, "vision_config": {}}"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  assert_eq!(cfg.model_type(), "lfm2-vl");
  assert_eq!(cfg.downsample_factor, 2);
  assert_eq!(cfg.image_token_index, 396);
  assert_eq!(cfg.projector_hidden_size, 2560);
  assert_eq!(cfg.vision_feature_layer, -1);
  assert_eq!(cfg.max_num_patches, 1024);
  assert_eq!(cfg.tile_size, 512);
  assert_eq!(cfg.eos_token_id, 7);
  // Image-splitting / tiling config defaults (config.py:76-88).
  assert!(cfg.do_image_splitting);
  assert_eq!(cfg.encoder_patch_size, 16);
  assert_eq!(cfg.max_image_tokens, 256);
  assert_eq!(cfg.min_image_tokens, 64);
  assert_eq!(cfg.max_tiles, 10);
  assert_eq!(cfg.min_tiles, 2);
  assert_eq!(cfg.max_pixels_tolerance, 2.0);
  assert!(!cfg.use_thumbnail);
  assert!(cfg.use_image_special_tokens);
  assert_eq!(cfg.projector_hidden_act, "gelu");
  // The nested vision activation default (`config.py:65`).
  assert_eq!(cfg.vision_config.hidden_act, "gelu_pytorch_tanh");
  assert!(cfg.quantization().is_none(), "dense by default");
  cfg.validate().unwrap();
}

#[test]
fn model_config_parses_explicit_tiling_fields() {
  // The image-splitting / tiling config fields (config.py:76-88) parse from JSON
  // and validate. These are carried for config parity (the mlx-vlm processor
  // path this port mirrors runs with splitting disabled), so this only pins the
  // parse + validate, not any splitting behavior.
  let json = r#"{
    "text_config": {}, "vision_config": {},
    "do_image_splitting": false,
    "encoder_patch_size": 14,
    "max_image_tokens": 300,
    "min_image_tokens": 32,
    "max_tiles": 12,
    "min_tiles": 4,
    "max_pixels_tolerance": 1.5,
    "tile_size": 448,
    "use_thumbnail": true,
    "use_image_special_tokens": false,
    "projector_hidden_act": "gelu"
  }"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  cfg.validate().unwrap();
  assert!(!cfg.do_image_splitting);
  assert_eq!(cfg.encoder_patch_size, 14);
  assert_eq!(cfg.max_image_tokens, 300);
  assert_eq!(cfg.min_image_tokens, 32);
  assert_eq!(cfg.max_tiles, 12);
  assert_eq!(cfg.min_tiles, 4);
  assert_eq!(cfg.max_pixels_tolerance, 1.5);
  assert_eq!(cfg.tile_size, 448);
  assert!(cfg.use_thumbnail);
  assert!(!cfg.use_image_special_tokens);
  assert_eq!(cfg.projector_hidden_act, "gelu");
}

#[test]
fn model_config_rejects_nonpositive_tiling_cardinality() {
  for bad in [
    r#"{"text_config":{},"vision_config":{},"encoder_patch_size":0}"#,
    r#"{"text_config":{},"vision_config":{},"max_image_tokens":0}"#,
    r#"{"text_config":{},"vision_config":{},"min_image_tokens":0}"#,
    r#"{"text_config":{},"vision_config":{},"max_tiles":0}"#,
    r#"{"text_config":{},"vision_config":{},"min_tiles":-1}"#,
  ] {
    let cfg = ModelConfig::from_json(bad).unwrap();
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange for {bad}, got {err}"
    );
  }
}

#[test]
fn model_config_rejects_inverted_tiling_ranges() {
  // min_tiles > max_tiles is an empty band.
  let bad_tiles = r#"{"text_config":{},"vision_config":{},"min_tiles":6,"max_tiles":4}"#;
  let cfg = ModelConfig::from_json(bad_tiles).unwrap();
  assert!(matches!(cfg.validate().unwrap_err(), Error::OutOfRange(_)));

  // min_image_tokens > max_image_tokens is an empty band.
  let bad_tokens =
    r#"{"text_config":{},"vision_config":{},"min_image_tokens":300,"max_image_tokens":256}"#;
  let cfg = ModelConfig::from_json(bad_tokens).unwrap();
  assert!(matches!(cfg.validate().unwrap_err(), Error::OutOfRange(_)));
}

#[test]
fn model_config_rejects_nonfinite_pixels_tolerance() {
  // A non-positive tolerance is OutOfRange; a non-finite one is NonFiniteScalar.
  let zero = r#"{"text_config":{},"vision_config":{},"max_pixels_tolerance":0.0}"#;
  let cfg = ModelConfig::from_json(zero).unwrap();
  assert!(matches!(cfg.validate().unwrap_err(), Error::OutOfRange(_)));

  let nan = r#"{"text_config":{},"vision_config":{},"max_pixels_tolerance":"not a number"}"#;
  // A non-numeric value fails at parse, not validate.
  assert!(matches!(
    ModelConfig::from_json(nan).unwrap_err(),
    Error::Parse(_)
  ));
}

#[test]
fn model_config_vision_feature_layer_explicit_in_range() {
  // vision_feature_layer = 5 keeps the first 6 layers (5 + 1).
  let json = r#"{"text_config": {}, "vision_config": {}, "vision_feature_layer": 5}"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  cfg.validate().unwrap();
  assert_eq!(cfg.vision_feature_layers_kept().unwrap(), 6);
}

#[test]
fn model_config_rejects_out_of_range_vision_feature_layer() {
  // 12 keeps 13 layers but the vision tower only has 12.
  let json = r#"{"text_config": {}, "vision_config": {}, "vision_feature_layer": 12}"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn model_config_rejects_wrong_model_type() {
  let json = r#"{"text_config": {}, "vision_config": {}, "model_type": "qwen2_vl"}"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::UnknownEnumValue(_)), "got {err}");
}

#[test]
fn model_config_accepts_both_model_type_spellings() {
  // `config.py`'s default is `"lfm2-vl"` (hyphen, `config.py:75`), but the
  // released mlx-community checkpoints (e.g. `mlx-community/LFM2.5-VL-450M-6bit`
  // / `-8bit`) ship `model_type: "lfm2_vl"` (underscore). Both must validate so
  // either checkpoint spelling loads; a bogus value still errors.
  let underscore = r#"{"text_config": {}, "vision_config": {}, "model_type": "lfm2_vl"}"#;
  let cfg = ModelConfig::from_json(underscore).unwrap();
  cfg
    .validate()
    .expect("underscore model_type lfm2_vl must validate (mlx-community checkpoints)");
  assert_eq!(cfg.model_type(), "lfm2_vl");

  let hyphen = r#"{"text_config": {}, "vision_config": {}, "model_type": "lfm2-vl"}"#;
  let cfg = ModelConfig::from_json(hyphen).unwrap();
  cfg
    .validate()
    .expect("hyphen model_type lfm2-vl must still validate (config.py default)");
  assert_eq!(cfg.model_type(), "lfm2-vl");

  let bogus = r#"{"text_config": {}, "vision_config": {}, "model_type": "lfm2vl"}"#;
  let cfg = ModelConfig::from_json(bogus).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::UnknownEnumValue(_)), "got {err}");
}

#[test]
fn model_config_rejects_non_gelu_projector_hidden_act() {
  // The projector forward hard-codes erf GELU (`projector.rs`); a checkpoint
  // declaring any other `projector_hidden_act` must fail loudly rather than
  // silently running GELU with the wrong declared activation.
  let gelu = r#"{"text_config": {}, "vision_config": {}, "projector_hidden_act": "gelu"}"#;
  ModelConfig::from_json(gelu)
    .unwrap()
    .validate()
    .expect("projector_hidden_act gelu must validate");

  let relu = r#"{"text_config": {}, "vision_config": {}, "projector_hidden_act": "relu"}"#;
  let cfg = ModelConfig::from_json(relu).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::UnknownEnumValue(_)), "got {err}");
}

#[test]
fn model_config_rejects_negative_image_token_index() {
  let json = r#"{"text_config": {}, "vision_config": {}, "image_token_index": -1}"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn model_config_rejects_malformed_json() {
  let err = ModelConfig::from_json("{not json").unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err}");
}

#[test]
fn model_config_propagates_invalid_tower_config() {
  // A structurally invalid vision sub-config must fail the top-level validate.
  let json = r#"{"text_config": {}, "vision_config": {"num_attention_heads": 0}}"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  let err = cfg.validate().unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn model_config_applies_rope_parameters_override_on_nested_text_config() {
  // `__post_init__` (`lfm2.py:41-42`): a `text_config.rope_parameters.rope_theta`
  // overrides the top-level `text_config.rope_theta`. The VLM `ModelConfig`
  // deserializes `text_config` via serde derive (NOT `TextConfig::from_json`), so
  // `ModelConfig::from_json` must apply the override on the nested config too —
  // otherwise the wrapped LM attention RoPE is built with the wrong base. Here
  // the nested top-level is 1000 and the override is 5000, so the effective
  // `text_config.rope_theta` on the VLM path must be 5000.
  let json = r#"{
    "vision_config": {},
    "text_config": {
      "hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 2,
      "num_hidden_layers": 2, "rope_theta": 1000.0,
      "rope_parameters": {"rope_theta": 5000.0}
    }
  }"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  assert_eq!(
    cfg.text_config.rope_theta, 5000.0,
    "text_config.rope_parameters.rope_theta must win on the VLM nested path"
  );
  // The override leaves an otherwise-valid config valid.
  cfg.validate().unwrap();
}
