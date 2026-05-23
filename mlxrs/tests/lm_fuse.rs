//! LM-A3 — `fuse()` library orchestrator (`lm::fuse`).
//!
//! Integration tests for the port of `mlx_lm/fuse.py::main` — the
//! adapter-fusion driver that ties `load_adapters` + per-layer
//! [`LoraLayer::fuse`](mlxrs::lm::lora::LoraLayer::fuse) + the F6 save into
//! a one-call pipeline. Mirrors `lm_convert.rs` style: gated on the `lm`
//! umbrella, fixture-built synthetic source / adapter directories under
//! `temp_dir()`, hand-traced assertions.
//!
//! Test list (per the A3 spec, #162):
//! - `fuse_round_trips_lora_into_dense_weights`
//! - `fuse_dequantize_strips_quantization_keys_from_config`
//! - `fuse_dequantize_false_preserves_quantization_keys`
//! - `fuse_rejects_missing_model_path`
//! - `fuse_rejects_missing_adapter_path`
//! - `fuse_rejects_hf_hub_url_in_model_path`
//! - `fuse_rejects_hf_hub_url_in_adapter_path`
//! - `fuse_with_no_adapter_layers_is_err` — load_adapters' completeness
//!   postcondition surfaces here (a fuse where no base layer matched the
//!   adapter selection is an error, not a silent no-op save — same
//!   contract A3 inherits from F5's adapter loader).
//! - `fuse_save_path_is_created_when_absent`
//!
//! No `peak_memory()` magnitude asserts (per project memory).
#![cfg(feature = "lm")]

use std::{
  collections::HashMap,
  fs::{self, File},
  io::Write,
  path::{Path, PathBuf},
  process,
};

use mlxrs::{
  Array, Error, io,
  lm::{fuse, load},
};

// ────────────────────────── test scaffolding ──────────────────────────

/// Fresh per-test temp directory, the no-`tempfile`-crate convention used by
/// the rest of the suite (process-scoped + named so parallel cases never
/// collide; collision-resistant counter for multiple dirs per test).
fn temp_dir(name: &str) -> PathBuf {
  use std::sync::atomic::{AtomicU64, Ordering};
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!("mlxrs_lm_fuse_{}_{name}_{n}", process::id()));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

/// A minimal valid `Config` JSON with no quantization block. The model_type
/// / hidden_size / etc are immaterial — `fuse` does not consult them beyond
/// `num_hidden_layers` (which drives `load_adapters`' trailing-block window).
fn plain_config_json(num_hidden_layers: i32) -> String {
  format!(
    r#"{{
      "model_type": "qwen3",
      "hidden_size": 16,
      "num_hidden_layers": {num_hidden_layers},
      "num_attention_heads": 2,
      "num_key_value_heads": 2,
      "head_dim": 8,
      "rope_theta": 10000.0,
      "vocab_size": 128,
      "tie_word_embeddings": false
    }}"#
  )
}

/// A `Config` JSON carrying a `quantization` AND a `quantization_config`
/// block — exactly the dual-key shape the F6 `save_config` mirror emits
/// (`quantization → quantization_config`), so a quantized base loaded from
/// an F6-saved checkpoint carries both top-level keys.
fn config_json_with_quant_blocks(num_hidden_layers: i32) -> String {
  format!(
    r#"{{
      "model_type": "qwen3",
      "hidden_size": 16,
      "num_hidden_layers": {num_hidden_layers},
      "num_attention_heads": 2,
      "num_key_value_heads": 2,
      "head_dim": 8,
      "rope_theta": 10000.0,
      "vocab_size": 128,
      "tie_word_embeddings": false,
      "quantization": {{ "group_size": 64, "bits": 4 }},
      "quantization_config": {{ "group_size": 64, "bits": 4 }}
    }}"#
  )
}

/// Write a synthetic base-model directory: `config.json` + a single
/// `model.safetensors` carrying `weights`.
fn write_base_dir(name: &str, weights: &HashMap<String, Array>, config_json: &str) -> PathBuf {
  let dir = temp_dir(name);
  let mut f = File::create(dir.join("config.json")).unwrap();
  f.write_all(config_json.as_bytes()).unwrap();
  io::save_safetensors(&dir.join("model.safetensors"), weights).unwrap();
  dir
}

/// `[output_dims=2, input_dims=3]` identity-on-first-two-coords base
/// weight, the same hand-traced fixture the lora unit tests reuse.
fn base_weight() -> Array {
  Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &(2, 3)).unwrap()
}

/// `[input_dims=3, r=2]` LoRA `lora_a` factor — picks the first two coords
/// of `x`.
fn lora_a() -> Array {
  Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &(3, 2)).unwrap()
}

/// `[r=2, output_dims=2]` identity LoRA `lora_b` factor.
fn lora_b() -> Array {
  Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap()
}

/// Write an mlx-lm-native adapter dir with `keys=["self_attn.q_proj"]` and
/// `rank=2`, `scale=2.0`, carrying factors for the q_proj of every block in
/// `blocks`. The factors are the hand-traced `lora_a` / `lora_b` above.
fn write_mlxlm_adapter(name: &str, blocks: &[i32], scale: f32) -> PathBuf {
  let dir = temp_dir(name);
  let config = format!(
    r#"{{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": {{ "rank": 2, "scale": {scale}, "keys": ["self_attn.q_proj"] }}
    }}"#
  );
  fs::write(dir.join("adapter_config.json"), config).unwrap();
  let mut arrays: HashMap<String, Array> = HashMap::new();
  for &b in blocks {
    let path = format!("model.layers.{b}.self_attn.q_proj");
    arrays.insert(format!("{path}.lora_a"), lora_a());
    arrays.insert(format!("{path}.lora_b"), lora_b());
  }
  io::save_safetensors(&dir.join("adapters.safetensors"), &arrays).unwrap();
  dir
}

/// A toy base weight map: 2 decoder blocks each carrying a single
/// `self_attn.q_proj.weight` (rank-2, `[2, 3]`).
fn toy_base_weights(num_blocks: i32) -> HashMap<String, Array> {
  let mut w = HashMap::new();
  for b in 0..num_blocks {
    w.insert(
      format!("model.layers.{b}.self_attn.q_proj.weight"),
      base_weight(),
    );
  }
  w
}

// ─────────────────── round-trip: LoRA → dense fused weights ───────────────────

#[test]
fn fuse_round_trips_lora_into_dense_weights() {
  // Build a tiny base (2 decoder blocks, one q_proj each) + an adapter
  // shipping `lora_a` / `lora_b` for both q_projs. Fuse and reload the
  // saved model; the fused weight must equal `W + scale * (lora_b^T @
  // lora_a^T)` to f32 precision.
  let weights = toy_base_weights(2);
  let model_dir = write_base_dir("rt_lora_model", &weights, &plain_config_json(2));
  let adapter_dir = write_mlxlm_adapter("rt_lora_adapter", &[0, 1], /* scale */ 2.0);
  let save_dir = temp_dir("rt_lora_save");
  // F6's `save_model` happily overwrites into a created-but-empty dir;
  // remove the temp_dir-created scaffold so we can also exercise the
  // "save_path created when absent" branch implicitly.
  fs::remove_dir_all(&save_dir).unwrap();

  fuse::fuse(
    &model_dir,
    &adapter_dir,
    &save_dir,
    /* dequantize */ false,
  )
  .unwrap();

  // Reload the saved fused weights through the F6 index loader.
  let mut reloaded = load::load_weights(&save_dir).unwrap();
  assert_eq!(reloaded.len(), 2, "only the two q_proj weights survive");

  // Hand-traced expected fused weight for the toy fixture:
  //   W                = [[1,0,0],
  //                       [0,1,0]]                 (2×3)
  //   lora_b           = [[1,0],
  //                       [0,1]]                   (2×2, identity)
  //   lora_a           = [[1,0],
  //                       [0,1],
  //                       [0,0]]                   (3×2)
  //   scale * lora_b^T = [[2,0],
  //                       [0,2]]                   (2×2)
  //   delta = (scale * lora_b^T) @ lora_a^T
  //         = [[2,0],   @   [[1,0,0],     = [[2,0,0],
  //            [0,2]]        [0,1,0]]        [0,2,0]]
  //   W_fused = W + delta
  //         = [[3,0,0],
  //            [0,3,0]]
  let expected = vec![3.0_f32, 0.0, 0.0, 0.0, 3.0, 0.0];
  for b in 0..2 {
    let key = format!("model.layers.{b}.self_attn.q_proj.weight");
    let mut got = reloaded.remove(&key).expect("fused weight present");
    let vals = got.to_vec::<f32>().unwrap();
    for (i, (g, e)) in vals.iter().zip(expected.iter()).enumerate() {
      assert!(
        (g - e).abs() <= 1e-5,
        "block {b} fused weight element {i}: got {g}, expected {e} \
         (full: {vals:?} vs {expected:?})"
      );
    }
  }
}

// ─────────────────── dequantize=true strips quant config keys ───────────────────

#[test]
fn fuse_dequantize_strips_quantization_keys_from_config() {
  // Start with a config that carries BOTH `quantization` AND
  // `quantization_config` (the F6 `save_config` dual-key shape). The
  // base weights are dense (no quantized triples to actually
  // dequantize — exercises the config-strip path in isolation, plus
  // the `dequantize_weights` walk over a quant-free map which is a
  // no-op pass-through).
  let weights = toy_base_weights(2);
  let model_dir = write_base_dir(
    "deq_strip_model",
    &weights,
    &config_json_with_quant_blocks(2),
  );
  let adapter_dir = write_mlxlm_adapter("deq_strip_adapter", &[0, 1], 2.0);
  let save_dir = temp_dir("deq_strip_save");
  fs::remove_dir_all(&save_dir).unwrap();

  fuse::fuse(
    &model_dir,
    &adapter_dir,
    &save_dir,
    /* dequantize */ true,
  )
  .unwrap();

  let saved_cfg_text = fs::read_to_string(save_dir.join("config.json")).unwrap();
  let saved_cfg: serde_json::Value = serde_json::from_str(&saved_cfg_text).unwrap();
  assert!(
    saved_cfg.get("quantization").is_none(),
    "`quantization` stripped from saved config: {saved_cfg_text}"
  );
  assert!(
    saved_cfg.get("quantization_config").is_none(),
    "`quantization_config` stripped from saved config: {saved_cfg_text}"
  );
  // Other keys must be preserved.
  assert_eq!(
    saved_cfg.get("model_type").and_then(|v| v.as_str()),
    Some("qwen3")
  );
}

// ─────────────────── dequantize=false preserves quant keys ───────────────────

#[test]
fn fuse_dequantize_false_preserves_quantization_keys() {
  // Same dual-key source config, but `dequantize=false`. The F6
  // `save_config` mirror preserves the original `quantization` key AND
  // mirrors it to `quantization_config` (the save side's two-key
  // contract), so a `dequantize=false` fuse must leave BOTH keys
  // present on disk.
  let weights = toy_base_weights(2);
  let model_dir = write_base_dir(
    "deq_keep_model",
    &weights,
    &config_json_with_quant_blocks(2),
  );
  let adapter_dir = write_mlxlm_adapter("deq_keep_adapter", &[0, 1], 2.0);
  let save_dir = temp_dir("deq_keep_save");
  fs::remove_dir_all(&save_dir).unwrap();

  fuse::fuse(
    &model_dir,
    &adapter_dir,
    &save_dir,
    /* dequantize */ false,
  )
  .unwrap();

  let saved_cfg_text = fs::read_to_string(save_dir.join("config.json")).unwrap();
  let saved_cfg: serde_json::Value = serde_json::from_str(&saved_cfg_text).unwrap();
  assert!(
    saved_cfg.get("quantization").is_some(),
    "`quantization` preserved by dequantize=false: {saved_cfg_text}"
  );
  // F6 `save_config` mirrors `quantization → quantization_config`, so
  // the second key is also present.
  assert!(
    saved_cfg.get("quantization_config").is_some(),
    "`quantization_config` preserved by F6 mirror: {saved_cfg_text}"
  );
}

// ─────────────────── reject missing model_path ───────────────────

#[test]
fn fuse_rejects_missing_model_path() {
  // A nonexistent model_path bubbles up as `Error::Backend` from
  // `load_config`'s file-not-found arm. The error message must name
  // the missing path so the user can recover.
  let bogus = temp_dir("missing_model_path_root");
  let model_dir = bogus.join("does_not_exist");
  // Build a real adapter dir so the failure attribution is unambiguous.
  let adapter_dir = write_mlxlm_adapter("missing_model_adapter", &[0], 2.0);
  let save_dir = temp_dir("missing_model_save");
  fs::remove_dir_all(&save_dir).unwrap();

  let err = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false).unwrap_err();
  match err {
    Error::Backend { message } => {
      assert!(
        message.contains("does_not_exist") || message.contains("config"),
        "error mentions the missing path / file: {message}"
      );
    }
    other => panic!("expected Error::Backend, got {other:?}"),
  }
}

// ─────────────────── reject missing adapter_path ───────────────────

#[test]
fn fuse_rejects_missing_adapter_path() {
  let weights = toy_base_weights(2);
  let model_dir = write_base_dir("missing_adapter_model", &weights, &plain_config_json(2));
  let bogus = temp_dir("missing_adapter_root");
  let adapter_dir = bogus.join("not_an_adapter");
  let save_dir = temp_dir("missing_adapter_save");
  fs::remove_dir_all(&save_dir).unwrap();

  let err = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false).unwrap_err();
  match err {
    Error::Backend { message } => {
      assert!(
        message.contains("not_an_adapter") || message.contains("adapter"),
        "error mentions the missing adapter path: {message}"
      );
    }
    other => panic!("expected Error::Backend, got {other:?}"),
  }
}

// ─────────────────── reject hf hub URL in model_path ───────────────────

#[test]
fn fuse_rejects_hf_hub_url_in_model_path() {
  let adapter_dir = write_mlxlm_adapter("hub_url_model_adapter", &[0], 2.0);
  let save_dir = temp_dir("hub_url_model_save");
  fs::remove_dir_all(&save_dir).unwrap();

  let err = fuse::fuse(
    Path::new("hf://mlx-community/Qwen3-4B-bf16"),
    &adapter_dir,
    &save_dir,
    false,
  )
  .unwrap_err();
  match err {
    Error::Backend { message } => {
      assert!(
        message.contains("huggingface-cli download mlx-community/Qwen3-4B-bf16"),
        "actionable workaround names the bare repo-id: {message}"
      );
      assert!(
        message.contains("model_path"),
        "error names the rejected arg: {message}"
      );
    }
    other => panic!("expected Error::Backend, got {other:?}"),
  }
}

// ─────────────────── reject hf hub URL in adapter_path ───────────────────

#[test]
fn fuse_rejects_hf_hub_url_in_adapter_path() {
  let weights = toy_base_weights(2);
  let model_dir = write_base_dir("hub_url_adapter_model", &weights, &plain_config_json(2));
  let save_dir = temp_dir("hub_url_adapter_save");
  fs::remove_dir_all(&save_dir).unwrap();

  let err = fuse::fuse(
    &model_dir,
    Path::new("https://huggingface.co/owner/adapter-repo"),
    &save_dir,
    false,
  )
  .unwrap_err();
  match err {
    Error::Backend { message } => {
      assert!(
        message.contains("huggingface-cli download owner/adapter-repo"),
        "actionable workaround names the bare repo-id (no protocol): {message}"
      );
      assert!(
        message.contains("adapter_path"),
        "error names the rejected arg: {message}"
      );
    }
    other => panic!("expected Error::Backend, got {other:?}"),
  }
}

// ─────────────────── no matching adapter layers ───────────────────

#[test]
fn fuse_with_no_adapter_layers_is_err() {
  // An adapter whose `keys` selection matches no base layer in the
  // model's weight map is rejected by `load_adapters`' completeness
  // postcondition (see `linear_to_lora_layers::check_adapter_completeness`
  // case (b): "adapter factor group(s) match no base layer"). `fuse`
  // bubbles that error up — the operation is NOT silently a no-op save.
  let weights = toy_base_weights(2);
  let model_dir = write_base_dir("no_match_model", &weights, &plain_config_json(2));
  // Adapter ships factors for q_proj of block 99 — present in the
  // adapter, absent in the base model — which is a "factor group(s) match
  // no base layer" violation per `load_adapters`' postcondition.
  let adapter_dir = write_mlxlm_adapter("no_match_adapter", &[99], 2.0);
  let save_dir = temp_dir("no_match_save");
  fs::remove_dir_all(&save_dir).unwrap();

  let err = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false).unwrap_err();
  match err {
    Error::Backend { message } => {
      // The postcondition's diagnostic fires from one of the three arms
      // in `check_adapter_completeness`. Here `keys=["self_attn.q_proj"]`
      // is an EXPLICIT selection that matches blocks 0 + 1 in the base
      // (both are q_projs in the trailing window), but the adapter
      // supplies factors for block 99 only — so arm (a) fires
      // ("adapter is missing factors for N explicitly-selected
      // target(s)"). Either the missing-factors phrase OR the
      // unused-factor / nothing-adapted phrases are acceptable
      // postcondition-violation signals.
      assert!(
        message.contains("missing factors")
          || message.contains("no base layer")
          || message.contains("match no")
          || message.contains("matched nothing"),
        "diagnostic names the postcondition violation: {message}"
      );
    }
    other => panic!("expected Error::Backend, got {other:?}"),
  }
}

// ─────────────────── save_path created when absent ───────────────────

#[test]
fn fuse_save_path_is_created_when_absent() {
  // The F6 `save` path creates `save_path` if absent (`create_dir_all`).
  // `fuse` inherits that contract — passing a never-created destination
  // must succeed and produce a valid directory.
  let weights = toy_base_weights(2);
  let model_dir = write_base_dir("absent_save_model", &weights, &plain_config_json(2));
  let adapter_dir = write_mlxlm_adapter("absent_save_adapter", &[0, 1], 2.0);
  // Build a path under a parent dir that DOES exist, but the leaf
  // doesn't — exactly the F6 contract.
  let save_parent = temp_dir("absent_save_parent");
  let save_dir = save_parent.join("brand_new_destination");
  assert!(!save_dir.exists(), "precondition: save_dir starts absent");

  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false).unwrap();

  assert!(save_dir.is_dir(), "save_dir created by fuse");
  assert!(
    save_dir.join("config.json").is_file(),
    "config.json written"
  );
  assert!(
    save_dir.join("model.safetensors.index.json").is_file(),
    "index.json written"
  );
}
