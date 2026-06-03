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
  // `3 * 16^2 = 768` flattened-patch width.
  assert_eq!(cfg.patch_feature_dim().unwrap(), 768);
  cfg.validate().unwrap();
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
  assert!(cfg.quantization().is_none(), "dense by default");
  cfg.validate().unwrap();
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
