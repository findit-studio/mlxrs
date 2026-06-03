//! Tests for the CLAP configuration: defaults pin against the named HTSAT /
//! RoBERTa architecture constants, JSON round-trips, and `validate()` accepts
//! the checkpoint defaults while rejecting structurally invalid configs.

use super::*;

/// The `laion/clap-htsat-unfused` defaults match the HTSAT-base + RoBERTa
/// architecture constants the plan cites (HF `ClapAudioConfig` /
/// `ClapTextConfig` / `ClapConfig` + `textclap/src/mel.rs`).
#[test]
fn defaults_match_architecture_constants() {
  let cfg = ClapConfig::default_for_test();
  assert_eq!(cfg.model_type(), "clap");
  assert_eq!(cfg.projection_dim, 512);
  // Top-level projection contract (HF `ClapConfig`): the projection-input
  // `hidden_size` equals the tower hidden (`768`), and the projection MLP uses
  // ReLU (NOT the towers' GELU).
  assert_eq!(cfg.hidden_size, 768);
  assert_eq!(cfg.projection_hidden_act(), "relu");

  let a = &cfg.audio_config;
  assert_eq!(a.model_type(), "clap_audio_model");
  assert_eq!(a.patch_embeds_hidden_size, 96);
  assert_eq!(a.depths, [2, 2, 6, 2]);
  assert_eq!(a.num_attention_heads, [4, 8, 16, 32]);
  assert_eq!(a.window_size, 8);
  assert_eq!(a.patch_size, 4);
  assert_eq!(a.patch_embed_input_channels, 1);
  assert_eq!(a.spec_size, 256);
  assert_eq!(a.freq_ratio, 4);
  assert_eq!(a.hidden_size, 768);
  assert!(!a.enable_fusion);
  // mel / spectrogram front-end params (also pinned by textclap/src/mel.rs).
  assert_eq!(a.sampling_rate, 48_000);
  assert_eq!(a.num_mel_bins, 64);

  let t = &cfg.text_config;
  assert_eq!(t.model_type(), "clap_text_model");
  assert_eq!(t.vocab_size, 50265);
  assert_eq!(t.hidden_size, 768);
  assert_eq!(t.num_hidden_layers, 12);
  assert_eq!(t.num_attention_heads, 12);
  assert_eq!(t.intermediate_size, 3072);
  assert_eq!(t.max_position_embeddings, 514);
  assert_eq!(t.type_vocab_size, 1);
  assert_eq!(t.pad_token_id, 1);
  assert_eq!(t.hidden_act(), "gelu");
}

/// A minimal `{}` config parses to all-defaults (forward-compatible:
/// `#[serde(default)]`, not `deny_unknown_fields`).
#[test]
fn empty_json_is_all_defaults() {
  let cfg = ClapConfig::from_json("{}").expect("empty config parses");
  assert_eq!(cfg.model_type(), "clap");
  assert_eq!(cfg.audio_config.depths, [2, 2, 6, 2]);
  assert_eq!(cfg.text_config.vocab_size, 50265);
  cfg.validate().expect("default config validates");
}

/// Unmodeled keys are ignored; modeled nested keys override defaults.
#[test]
fn parses_nested_and_ignores_unknown_keys() {
  let json = r#"{
    "model_type": "clap",
    "projection_dim": 512,
    "some_future_key": [1, 2, 3],
    "audio_config": { "hidden_size": 768, "another_unknown": true },
    "text_config": { "vocab_size": 50265 }
  }"#;
  let cfg = ClapConfig::from_json(json).expect("parses");
  assert_eq!(cfg.audio_config.hidden_size, 768);
  assert_eq!(cfg.text_config.vocab_size, 50265);
  cfg.validate().expect("validates");
}

/// `validate()` accepts the real checkpoint defaults end-to-end.
#[test]
fn validate_accepts_checkpoint_defaults() {
  ClapConfig::default_for_test()
    .validate()
    .expect("checkpoint-default config validates");
}

/// A wrong top-level `model_type` is rejected (pinned to `"clap"`).
#[test]
fn validate_rejects_wrong_model_type() {
  let json = r#"{ "model_type": "not_clap" }"#;
  let cfg = ClapConfig::from_json(json).expect("parses");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::UnknownEnumValue(_)),
    "expected a pin failure, got {err:?}"
  );
}

/// A non-default-but-positive dimension is rejected: every architecture field
/// is pinned, so a different (positive) `hidden_size` is an unsupported model,
/// not a valid config (`pin_i32` → `OutOfRange`).
#[test]
fn validate_rejects_non_default_dim() {
  let json = r#"{ "text_config": { "hidden_size": 0 } }"#;
  let cfg = ClapConfig::from_json(json).expect("parses");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange, got {err:?}"
  );
}

/// `enable_fusion = true` is rejected (only the unfused path is supported).
#[test]
fn validate_rejects_enable_fusion() {
  let json = r#"{ "audio_config": { "enable_fusion": true } }"#;
  let cfg = ClapConfig::from_json(json).expect("parses");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange for enable_fusion, got {err:?}"
  );
}

/// A wrong per-stage `depths` list is rejected (pinned to the HTSAT-base
/// layout).
#[test]
fn validate_rejects_wrong_depths() {
  let json = r#"{ "audio_config": { "depths": [2, 2, 2, 2] } }"#;
  let cfg = ClapConfig::from_json(json).expect("parses");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected a depths pin failure, got {err:?}"
  );
}

// ── fixed-arity parse rejection ──
//
// `depths` / `num_attention_heads` are the fixed 4-stage HTSAT layout, modeled
// as `[i32; 4]`. A `config.json` whose array is not exactly four elements is an
// invalid HTSAT config: serde's fixed-array deserializer fails on the length
// mismatch, so it is rejected at PARSE (a typed [`Error::Parse`]) BEFORE any Vec
// is allocated — no OOM-before-validate path for a hostile / corrupt config.

/// A `depths` array that is too long (5 elements) is rejected at parse time, not
/// allocated and then validated. Proves the fixed-arity `[i32; 4]` typing closes
/// the unbounded-`Vec`-before-`validate` allocation path.
#[test]
fn from_json_rejects_oversized_depths_at_parse() {
  let json = r#"{ "audio_config": { "depths": [2, 2, 6, 2, 2] } }"#;
  let err = ClapConfig::from_json(json).unwrap_err();
  assert!(
    matches!(err, Error::Parse(_)),
    "expected a parse-time length rejection for oversized depths, got {err:?}"
  );
}

/// A `depths` array that is too short (3 elements) is likewise rejected at parse
/// time (the fixed-array deserializer fails on any length != 4).
#[test]
fn from_json_rejects_undersized_depths_at_parse() {
  let json = r#"{ "audio_config": { "depths": [2, 2, 6] } }"#;
  let err = ClapConfig::from_json(json).unwrap_err();
  assert!(
    matches!(err, Error::Parse(_)),
    "expected a parse-time length rejection for undersized depths, got {err:?}"
  );
}

/// A hostile, very large `depths` array is rejected at parse time WITHOUT first
/// allocating it (the fixed-array deserializer stops at the length mismatch on
/// the fifth element). Regression for the OOM-before-validate finding: a corrupt
/// config can no longer drive an unbounded allocation ahead of validation.
#[test]
fn from_json_rejects_huge_depths_at_parse_without_oom() {
  // A 100_000-element JSON array — would be a large `Vec<i32>` if modeled as a
  // growable sequence; the `[i32; 4]` deserializer rejects it at the length
  // boundary instead.
  let mut elems = String::from("0");
  for _ in 1..100_000 {
    elems.push_str(",0");
  }
  let json = format!(r#"{{ "audio_config": {{ "depths": [{elems}] }} }}"#);
  let err = ClapConfig::from_json(&json).unwrap_err();
  assert!(
    matches!(err, Error::Parse(_)),
    "expected a parse-time length rejection for a huge depths array, got {err:?}"
  );
}

/// A `num_attention_heads` array that is the wrong length (5 elements) is
/// rejected at parse time, same as `depths`.
#[test]
fn from_json_rejects_oversized_num_attention_heads_at_parse() {
  let json = r#"{ "audio_config": { "num_attention_heads": [4, 8, 16, 32, 64] } }"#;
  let err = ClapConfig::from_json(json).unwrap_err();
  assert!(
    matches!(err, Error::Parse(_)),
    "expected a parse-time length rejection for oversized num_attention_heads, got {err:?}"
  );
}

/// A hostile, very large `num_attention_heads` array is rejected at parse time
/// without an unbounded allocation (the OOM-before-validate finding, for the
/// head-count field).
#[test]
fn from_json_rejects_huge_num_attention_heads_at_parse_without_oom() {
  let mut elems = String::from("0");
  for _ in 1..100_000 {
    elems.push_str(",0");
  }
  let json = format!(r#"{{ "audio_config": {{ "num_attention_heads": [{elems}] }} }}"#);
  let err = ClapConfig::from_json(&json).unwrap_err();
  assert!(
    matches!(err, Error::Parse(_)),
    "expected a parse-time length rejection for a huge num_attention_heads array, got {err:?}"
  );
}

/// The genuine 4-element HTSAT-base arrays still parse cleanly and validate
/// end-to-end (the positive half of the fixed-arity fix: a correct config is
/// unaffected by the length rejection).
#[test]
fn from_json_accepts_genuine_four_element_arrays() {
  let json = r#"{
    "audio_config": {
      "depths": [2, 2, 6, 2],
      "num_attention_heads": [4, 8, 16, 32]
    }
  }"#;
  let cfg = ClapConfig::from_json(json).expect("genuine 4-element arrays parse");
  assert_eq!(cfg.audio_config.depths, [2, 2, 6, 2]);
  assert_eq!(cfg.audio_config.num_attention_heads, [4, 8, 16, 32]);
  cfg
    .validate()
    .expect("genuine 4-element arrays validate end-to-end");
}

// ── per-field pin rejections ──
//
// Each test overrides ONE fixed field to a non-default-but-otherwise-valid
// value and asserts `validate()` rejects it with a typed error. Without the
// pins these would all PASS the advertised fail-fast gate while running a
// wrong / oversized model against the hard-coded unfused CLAP-HTSAT contract.

/// Every fixed `i32` audio field, set to a distinct non-default positive value,
/// is rejected by its pin (`pin_i32` → `OutOfRange`).
#[test]
fn validate_rejects_non_default_audio_i32_fields() {
  // (override-key, non-default-but-positive value). 16 kHz / 80-mel is the
  // finding's worked example (a different, real CLAP-incompatible audio setup).
  let cases: &[(&str, i32)] = &[
    ("patch_embeds_hidden_size", 64),
    ("window_size", 7),
    ("patch_size", 16),
    ("patch_embed_input_channels", 3),
    ("spec_size", 224),
    ("freq_ratio", 2),
    ("hidden_size", 512),
    ("sampling_rate", 16_000),
    ("num_mel_bins", 80),
  ];
  for (key, value) in cases {
    let json = format!(r#"{{ "audio_config": {{ "{key}": {value} }} }}"#);
    let cfg = ClapConfig::from_json(&json).expect("parses");
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange for audio {key}={value}, got {err:?}"
    );
  }
}

/// The fixed audio float fields (`mlp_ratio`, `layer_norm_eps`), set to a
/// non-default finite value, are rejected by `pin_f64` (`OutOfRange`).
#[test]
fn validate_rejects_non_default_audio_f64_fields() {
  for (key, value) in [("mlp_ratio", "2.0"), ("layer_norm_eps", "1e-6")] {
    let json = format!(r#"{{ "audio_config": {{ "{key}": {value} }} }}"#);
    let cfg = ClapConfig::from_json(&json).expect("parses");
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange for audio {key}={value}, got {err:?}"
    );
  }
}

/// A wrong per-stage `num_attention_heads` list is rejected (pinned to the
/// HTSAT-base `[4, 8, 16, 32]` layout).
#[test]
fn validate_rejects_wrong_audio_heads() {
  let json = r#"{ "audio_config": { "num_attention_heads": [2, 4, 8, 16] } }"#;
  let cfg = ClapConfig::from_json(json).expect("parses");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected a heads pin failure, got {err:?}"
  );
}

/// Every fixed `i32` text field, set to a distinct non-default positive value,
/// is rejected by its pin (`pin_i32` → `OutOfRange`).
#[test]
fn validate_rejects_non_default_text_i32_fields() {
  let cases: &[(&str, i32)] = &[
    ("vocab_size", 30522),
    ("hidden_size", 1024),
    ("num_hidden_layers", 24),
    ("num_attention_heads", 16),
    ("intermediate_size", 4096),
    ("max_position_embeddings", 512),
    ("type_vocab_size", 2),
    ("pad_token_id", 0),
  ];
  for (key, value) in cases {
    let json = format!(r#"{{ "text_config": {{ "{key}": {value} }} }}"#);
    let cfg = ClapConfig::from_json(&json).expect("parses");
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange for text {key}={value}, got {err:?}"
    );
  }
}

/// The fixed text `layer_norm_eps`, set to a non-default finite value, is
/// rejected by `pin_f64` (`OutOfRange`).
#[test]
fn validate_rejects_non_default_text_layer_norm_eps() {
  let json = r#"{ "text_config": { "layer_norm_eps": 1e-6 } }"#;
  let cfg = ClapConfig::from_json(json).expect("parses");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange for text layer_norm_eps, got {err:?}"
  );
}

/// A non-`gelu` text `hidden_act` is rejected (pinned to exact GELU;
/// `pin_str` → `UnknownEnumValue`).
#[test]
fn validate_rejects_non_default_text_hidden_act() {
  let json = r#"{ "text_config": { "hidden_act": "relu" } }"#;
  let cfg = ClapConfig::from_json(json).expect("parses");
  let err = cfg.validate().unwrap_err();
  assert!(
    matches!(err, Error::UnknownEnumValue(_)),
    "expected UnknownEnumValue for text hidden_act, got {err:?}"
  );
}

/// A non-default top-level `projection_dim` / `hidden_size` is rejected. Both
/// are real serialized top-level keys in the checkpoint `config.json`
/// (`projection_dim=512`, `hidden_size=768`); the projection MLP is
/// `Linear(hidden_size=768 → projection_dim=512)` → ReLU → `Linear(512 → 512)`,
/// so a deviating width is an incompatible projection (`pin_i32` →
/// `OutOfRange`).
#[test]
fn validate_rejects_non_default_projection_widths() {
  // (override-key, non-default-but-positive value). Each must be REJECTED.
  let cases: &[(&str, i32)] = &[
    ("projection_dim", 256),
    // `hidden_size=1024` is a real, CLAP-incompatible value (e.g. a
    // RoBERTa-large hidden); the checkpoint top-level `hidden_size` is `768`.
    ("hidden_size", 1024),
  ];
  for (key, value) in cases {
    let json = format!(r#"{{ "{key}": {value} }}"#);
    let cfg = ClapConfig::from_json(&json).expect("parses");
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange for {key}={value}, got {err:?}"
    );
  }
}

/// The real checkpoint serializes a top-level `projection_hidden_act` (`"relu"`)
/// that the struct must READ and PIN, not silently discard back to a default. A
/// non-default top-level `projection_hidden_act` (CLAP uses ReLU, NOT the
/// towers' GELU) is rejected with a typed error — proving it is now modeled +
/// pinned rather than dropped (the unmodeled-field finding). `pin_str` →
/// `UnknownEnumValue`.
#[test]
fn validate_rejects_non_default_projection_hidden_act() {
  // `"gelu"` is the WRONG projection activation (it is the towers' activation,
  // not the projection's); the checkpoint projection activation is `"relu"`.
  for act in ["gelu", "gelu_new", "silu"] {
    let json = format!(r#"{{ "projection_hidden_act": "{act}" }}"#);
    let cfg = ClapConfig::from_json(&json).expect("parses");
    let err = cfg.validate().unwrap_err();
    assert!(
      matches!(err, Error::UnknownEnumValue(_)),
      "expected UnknownEnumValue for projection_hidden_act={act:?}, got {err:?}"
    );
  }
}

/// The real checkpoint top-level fields parse into the modeled fields under
/// their REAL serialized keys (`hidden_size`, `projection_hidden_act`) — not
/// dropped — and the genuine checkpoint values still validate. This is the
/// positive half of the unmodeled-field fix: a config carrying the real
/// top-level `hidden_size=768` + `projection_hidden_act="relu"` is read +
/// accepted (and would be REJECTED if the keys were ignored and a wrong default
/// pinned).
#[test]
fn parses_and_validates_top_level_projection_fields() {
  let json = r#"{
    "model_type": "clap",
    "projection_dim": 512,
    "hidden_size": 768,
    "projection_hidden_act": "relu"
  }"#;
  let cfg = ClapConfig::from_json(json).expect("parses");
  assert_eq!(cfg.hidden_size, 768);
  assert_eq!(cfg.projection_hidden_act(), "relu");
  cfg
    .validate()
    .expect("real top-level projection fields validate");
}

/// The pinned mel-front-end audio fields equal the constants the [`super::mel`]
/// front-end hard-codes, so the validated config and the front-end agree on the
/// fixed `(1, 1, 1001, 64)` shape contract.
#[test]
fn pinned_mel_fields_match_front_end_constants() {
  let a = &ClapConfig::default_for_test().audio_config;
  assert_eq!(a.sampling_rate, super::super::mel::SAMPLE_RATE as i32);
  assert_eq!(a.num_mel_bins, super::super::mel::N_MELS as i32);
}

impl ClapConfig {
  /// The all-defaults config (the `laion/clap-htsat-unfused` checkpoint
  /// defaults), built without going through JSON.
  fn default_for_test() -> Self {
    Self::from_json("{}").expect("empty config parses to defaults")
  }
}
