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
  assert_eq!(cfg.eos_token_id(), 7);
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
  // Neither a top-level nor a nested `text_config.eos_token_id` is present, so
  // the resolver backstops with `DEFAULT_EOS_TOKEN_ID`.
  assert_eq!(cfg.eos_token_id(), DEFAULT_EOS_TOKEN_ID);
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
fn model_config_rejects_oversized_max_tiles() {
  // `max_tiles` is the cardinality of the tile-grid candidate set, whose
  // builder reserves / iterates `max_tiles^2`; a value above the cap is
  // rejected at load so a malformed checkpoint cannot drive quadratic work.
  let over = MAX_TILES + 1;
  let bad = format!(r#"{{"text_config":{{}},"vision_config":{{}},"max_tiles":{over}}}"#);
  let cfg = ModelConfig::from_json(&bad).unwrap();
  assert!(
    matches!(cfg.validate().unwrap_err(), Error::OutOfRange(_)),
    "max_tiles above the cap must be OutOfRange"
  );

  // The cap value itself is in-bound (a faithful, if generous, config).
  let at_cap =
    format!(r#"{{"text_config":{{}},"vision_config":{{}},"min_tiles":2,"max_tiles":{MAX_TILES}}}"#);
  ModelConfig::from_json(&at_cap)
    .unwrap()
    .validate()
    .expect("max_tiles == cap validates");
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
fn model_config_pins_vision_num_channels_to_3() {
  // The full LFM2.5-VL model + its image processor are RGB-only: the processor
  // hard-wires `num_channels = RGB_CHANNELS` (`processor::Lfm2VlProcessorConfig::new`)
  // and `preprocess_image` rejects any config whose `num_channels != 3` (it uses
  // the channel count as the patchify stride into an always-3-channel buffer),
  // and the patch-embed Linear width derives from `num_channels * patch_size^2`.
  // So a full `ModelConfig` whose `vision_config.num_channels != 3` is a
  // wrong-architecture / malformed checkpoint and must be rejected at LOAD (a
  // typed `OutOfRange`), not run a mismatched architecture or fail late at a
  // vision matmul shape check. `1` (grayscale) and `4` (RGBA) are the most
  // plausible near-misses; both must be rejected.
  for bad_channels in [1, 2, 4, 6] {
    let json =
      format!(r#"{{"text_config":{{}},"vision_config":{{"num_channels":{bad_channels}}}}}"#);
    let cfg = ModelConfig::from_json(&json).unwrap();
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "vision_config.num_channels={bad_channels} must be OutOfRange at ModelConfig::validate, got {err}"
    );
  }
  // The RGB value `3` (and the default, which is 3) validates.
  let rgb = r#"{"text_config":{},"vision_config":{"num_channels":3}}"#;
  ModelConfig::from_json(rgb)
    .unwrap()
    .validate()
    .expect("num_channels == 3 (RGB) must validate");
  let defaulted = r#"{"text_config":{},"vision_config":{}}"#;
  let cfg = ModelConfig::from_json(defaulted).unwrap();
  assert_eq!(
    cfg.vision_config.num_channels, 3,
    "default num_channels is RGB"
  );
  cfg
    .validate()
    .expect("default (RGB) num_channels must validate");
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

#[test]
fn model_config_resolves_eos_from_text_config_when_top_level_null() {
  // The canonical `LiquidAI/LFM2-VL-450M` `config.json` carries a TOP-LEVEL
  // `eos_token_id: null` (the real value lives under `text_config`). A defaulted
  // bare `i32` rejected the present `null` (`invalid type: null, expected i32`),
  // which blocked loading the real checkpoint. The hand-written `Deserialize`
  // must (a) parse the present `null` and (b) resolve the eos from the nested
  // `text_config.eos_token_id` (`7`).
  let json = r#"{
    "model_type": "lfm2-vl",
    "eos_token_id": null,
    "bos_token_id": null,
    "pad_token_id": null,
    "text_config": {
      "model_type": "lfm2",
      "hidden_size": 1024,
      "num_hidden_layers": 16,
      "num_attention_heads": 16,
      "num_key_value_heads": 8,
      "vocab_size": 65536,
      "bos_token_id": 1,
      "pad_token_id": 0,
      "eos_token_id": 7
    },
    "vision_config": {"num_channels": 3}
  }"#;
  let cfg = ModelConfig::from_json(json).expect("top-level eos_token_id: null must parse");
  // Resolved from `text_config.eos_token_id`, NOT the top-level null.
  assert_eq!(
    cfg.eos_token_id(),
    7,
    "a top-level null eos must resolve from text_config.eos_token_id"
  );
  cfg
    .validate()
    .expect("the canonical-shape config validates");
}

#[test]
fn model_config_top_level_eos_wins_over_text_config() {
  // Precedence: a present, non-null TOP-LEVEL `eos_token_id` wins over the nested
  // `text_config.eos_token_id`. Here top-level `9` must beat nested `7`.
  let json = r#"{
    "eos_token_id": 9,
    "text_config": {"eos_token_id": 7},
    "vision_config": {}
  }"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  assert_eq!(
    cfg.eos_token_id(),
    9,
    "a present top-level eos_token_id must take precedence over the nested value"
  );
  cfg.validate().unwrap();
}

#[test]
fn model_config_eos_falls_back_to_default_when_absent_everywhere() {
  // Neither a top-level nor a nested `text_config.eos_token_id` is present
  // (absent OR an explicit top-level null with no nested value) ⇒ the resolver
  // backstops with `DEFAULT_EOS_TOKEN_ID` (`7`).
  for json in [
    r#"{"text_config": {}, "vision_config": {}}"#,
    r#"{"eos_token_id": null, "text_config": {}, "vision_config": {}}"#,
  ] {
    let cfg = ModelConfig::from_json(json).unwrap();
    assert_eq!(
      cfg.eos_token_id(),
      DEFAULT_EOS_TOKEN_ID,
      "absent eos everywhere must fall back to DEFAULT_EOS_TOKEN_ID for {json}"
    );
    cfg.validate().unwrap();
  }
}

// ──────────────────── config-controlled-allocation caps ─────────────────────
//
// Every config field that multiplies into a Vec reservation or a tensor
// allocation is bounded at `validate()` so a hostile-but-parseable config cannot
// drive an oversized allocation. Width-like fields take `MAX_CONFIG_DIM`,
// cardinality-like fields `MAX_CARDINALITY`, and the per-image patch budget
// `MAX_PATCH_BUDGET` — each rejecting both a non-positive and an oversized value.

#[test]
fn model_config_rejects_oversized_max_num_patches() {
  // `max_num_patches` is the leading dimension of the `pixel_values` allocation
  // the patchify paths zero-fill; a value near `i32::MAX` would drive a multi-TB
  // buffer. Bounded by `MAX_PATCH_BUDGET` at load (an over-cap value is the
  // ranged `OutOfRange`).
  let over = MAX_PATCH_BUDGET + 1;
  let bad = format!(r#"{{"text_config":{{}},"vision_config":{{}},"max_num_patches":{over}}}"#);
  let cfg = ModelConfig::from_json(&bad).unwrap();
  assert!(
    matches!(cfg.validate().unwrap_err(), Error::OutOfRange(_)),
    "max_num_patches above the patch-budget cap must be OutOfRange"
  );
  // A near-`i32::MAX` `max_num_patches` would size a multi-TB `pixel_values`
  // buffer; it is rejected at load.
  let huge = format!(
    r#"{{"text_config":{{}},"vision_config":{{}},"max_num_patches":{}}}"#,
    i32::MAX
  );
  assert!(matches!(
    ModelConfig::from_json(&huge)
      .unwrap()
      .validate()
      .unwrap_err(),
    Error::OutOfRange(_)
  ));
  // The cap value itself is in-bound (generous but faithful).
  let at_cap =
    format!(r#"{{"text_config":{{}},"vision_config":{{}},"max_num_patches":{MAX_PATCH_BUDGET}}}"#);
  ModelConfig::from_json(&at_cap)
    .unwrap()
    .validate()
    .expect("max_num_patches == cap validates");
}

#[test]
fn model_config_rejects_oversized_patch_budget_fields() {
  // The HF tile-grid budgets index the same per-image patch space as
  // `max_num_patches`, so each is bounded by `MAX_PATCH_BUDGET`.
  let over = MAX_PATCH_BUDGET + 1;
  for field in ["tile_size", "encoder_patch_size", "max_image_tokens"] {
    let bad = format!(r#"{{"text_config":{{}},"vision_config":{{}},"{field}":{over}}}"#);
    let cfg = ModelConfig::from_json(&bad).unwrap();
    assert!(
      matches!(cfg.validate().unwrap_err(), Error::OutOfRange(_)),
      "{field} above the patch-budget cap must be OutOfRange"
    );
  }
}

#[test]
fn model_config_rejects_oversized_width_fields() {
  // The projector width fields name a matmul axis / fold factor, bounded by
  // `MAX_CONFIG_DIM`.
  let over = i64::from(MAX_CONFIG_DIM) + 1;
  for field in ["downsample_factor", "projector_hidden_size"] {
    let bad = format!(r#"{{"text_config":{{}},"vision_config":{{}},"{field}":{over}}}"#);
    let cfg = ModelConfig::from_json(&bad).unwrap();
    assert!(
      matches!(cfg.validate().unwrap_err(), Error::OutOfRange(_)),
      "{field} above the width cap must be OutOfRange"
    );
  }
}

#[test]
fn vision_config_rejects_oversized_width_fields() {
  // Vision width fields (matmul axis / position-table column) are bounded by
  // `MAX_CONFIG_DIM` — both a non-positive and an oversized value is `OutOfRange`.
  // `hidden_size` is divisible by `num_attention_heads` (12), so use a multiple.
  let over_w = i64::from(MAX_CONFIG_DIM) + 12;
  for field in [
    "hidden_size",
    "intermediate_size",
    "image_size",
    "patch_size",
  ] {
    let bad = format!(r#"{{"model_type":"lfm2_vl","{field}":{over_w}}}"#);
    let cfg = VisionConfig::from_json(&bad).unwrap();
    assert!(
      matches!(cfg.validate().unwrap_err(), Error::OutOfRange(_)),
      "vision {field} above the width cap must be OutOfRange"
    );
  }
  // `num_patches` (the position-table row count) is bounded by the width cap too;
  // `MAX_CONFIG_DIM == 1024^2` is itself a perfect square, so the next larger
  // perfect square (`1025^2`) is over-cap and rejected by the range check (which
  // runs before the perfect-square check).
  let over_sq = 1025i64 * 1025;
  let bad = format!(r#"{{"model_type":"lfm2_vl","num_patches":{over_sq}}}"#);
  assert!(matches!(
    VisionConfig::from_json(&bad)
      .unwrap()
      .validate()
      .unwrap_err(),
    Error::OutOfRange(_)
  ));
}

#[test]
fn vision_config_rejects_oversized_cardinality_fields() {
  // Vision cardinality fields size an eager per-unit `Vec` / loop, bounded by
  // `MAX_CARDINALITY`; an over-cap value is `CapExceeded` (distinct from the
  // `OutOfRange` a non-positive value yields).
  let over = i64::from(MAX_CARDINALITY) + 1;
  for field in ["num_hidden_layers", "num_channels"] {
    let bad = format!(r#"{{"model_type":"lfm2_vl","{field}":{over}}}"#);
    let cfg = VisionConfig::from_json(&bad).unwrap();
    assert!(
      matches!(cfg.validate().unwrap_err(), Error::CapExceeded(_)),
      "vision {field} above the cardinality cap must be CapExceeded"
    );
  }
  // `num_attention_heads` over the cap: keep `hidden_size` divisible by it so the
  // cap check (not the head-split divisibility) is what fires. `MAX_CARDINALITY +
  // 1 == 4097` is prime-ish; use a `hidden_size` that is its multiple.
  let heads = MAX_CARDINALITY + 1;
  let hidden = i64::from(heads); // hidden_size == heads ⇒ divisible, head_dim == 1
  let bad =
    format!(r#"{{"model_type":"lfm2_vl","num_attention_heads":{heads},"hidden_size":{hidden}}}"#);
  assert!(matches!(
    VisionConfig::from_json(&bad)
      .unwrap()
      .validate()
      .unwrap_err(),
    Error::CapExceeded(_)
  ));
}

#[test]
fn vision_config_at_caps_validates() {
  // A generous-but-in-bound config validates: widths at `MAX_CONFIG_DIM`,
  // cardinalities at `MAX_CARDINALITY`. `num_patches == 1024^2 == MAX_CONFIG_DIM`
  // is a perfect square at the width cap; `hidden_size` must stay divisible by
  // `num_attention_heads` (use `MAX_CARDINALITY` heads ⇒ head_dim 256). Keep
  // `patch_size`/`num_channels` small so `patch_feature_dim` does not overflow.
  let hidden = MAX_CONFIG_DIM; // divisible by MAX_CARDINALITY (1<<20 / 4096 == 256)
  let json = format!(
    r#"{{"model_type":"lfm2_vl","hidden_size":{hidden},"intermediate_size":{MAX_CONFIG_DIM},
        "num_hidden_layers":{MAX_CARDINALITY},"num_attention_heads":{MAX_CARDINALITY},
        "num_channels":3,"image_size":{MAX_CONFIG_DIM},"patch_size":16,
        "num_patches":{MAX_CONFIG_DIM}}}"#
  );
  VisionConfig::from_json(&json)
    .unwrap()
    .validate()
    .expect("config at the caps validates");
}
