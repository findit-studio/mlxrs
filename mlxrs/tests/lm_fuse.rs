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
//! - `fuse_preserves_fan_in_fan_out_layout_for_non_square_peft_target` —
//!   PEFT `fan_in_fan_out: true` on a non-square base ([in=4, out=3]); the
//!   fused weight must be re-transposed back to the persisted `[in, out]`
//!   layout before save so a load through the same loader round-trips.
//! - `fuse_preserves_fan_in_fan_out_layout_for_square_target` — a square
//!   PEFT `fan_in_fan_out: true` base with a NON-symmetric LoRA delta
//!   (asymmetric upper-triangular); a silent transpose would produce a
//!   different matrix, so this catches the corruption a square-base
//!   numerical check would miss.
//! - `fuse_output_is_loadable_through_default_lm_loader` — fuse output dir
//!   loads end-to-end through `lm::load::load` (which constructs the
//!   tokenizer from the same dir); a fused dir that shipped weights +
//!   config only would silently fail to load.
//! - `fuse_copies_all_tokenizer_extras_present_in_source` — every
//!   `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` /
//!   `generation_config.json` present at `model_path` lands at `save_path`
//!   with byte-identical content (the verbatim-copy contract).
//! - `fuse_load_adapters_with_config_skips_second_adapter_config_read` (R2
//!   Finding 1) — `load_adapters_with_config` consumes a pre-parsed
//!   [`mlxrs::lm::lora::LoraConfig`] and must NOT re-read
//!   `adapter_config.json` on the load side. The test parses the config,
//!   clobbers the on-disk file, then calls `load_adapters_with_config`
//!   (success — proves no second read) AND `load_adapters` (failure —
//!   proves the wrapper still does its own parse), closing the TOCTOU
//!   window `fuse` previously had.
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
  lm::{fuse, load, lora},
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

// ─────────────────────────── PEFT fan_in_fan_out ───────────────────────────

/// Write a PEFT `adapter_config.json` + `adapter_model.safetensors` for the
/// given block list with `fan_in_fan_out: true`, rank `r` / `lora_alpha` /
/// `target_modules`. The factor tensors are supplied in PEFT's on-disk
/// orientation (`lora_A.weight`: `[r, in_features]`, `lora_B.weight`:
/// `[out_features, r]`) — `translate_peft_keys` transposes them to the
/// mlxrs scheme on load.
fn write_peft_fifo_adapter(
  name: &str,
  blocks: &[i32],
  r: i32,
  lora_alpha: f32,
  lora_a_disk: &Array,
  lora_b_disk: &Array,
) -> PathBuf {
  let dir = temp_dir(name);
  let config = format!(
    r#"{{
      "peft_type": "LORA",
      "r": {r},
      "lora_alpha": {lora_alpha},
      "target_modules": ["q_proj"],
      "fan_in_fan_out": true
    }}"#
  );
  fs::write(dir.join("adapter_config.json"), config).unwrap();
  let mut arrays: HashMap<String, Array> = HashMap::new();
  for &b in blocks {
    let path = format!("model.layers.{b}.self_attn.q_proj");
    arrays.insert(
      format!("base_model.model.{path}.lora_A.weight"),
      lora_a_disk.try_clone().unwrap(),
    );
    arrays.insert(
      format!("base_model.model.{path}.lora_B.weight"),
      lora_b_disk.try_clone().unwrap(),
    );
  }
  io::save_safetensors(&dir.join("adapter_model.safetensors"), &arrays).unwrap();
  dir
}

#[test]
fn fuse_preserves_fan_in_fan_out_layout_for_non_square_peft_target() {
  // PEFT `fan_in_fan_out: true` on a NON-square base — `[in=4, out=3]`
  // persisted in Conv1D layout. The fused weight must be saved in the
  // SAME persisted `[in, out]` orientation so a `load::load_weights`
  // reload yields a tensor the same loader transposes back to canonical
  // `[out, in]` — i.e. the round-trip is consistent. Without the fix the
  // fused weight is saved canonical `[out, in]` and a downstream reader
  // either errors (non-square: shape mismatch) or silently transposes
  // (square: semantic corruption — the next test covers that).
  //
  // We assert two contracts:
  //  (1) The save succeeds and the reloaded `<path>.weight` has the
  //      persisted shape `[in=4, out=3]` (NOT `[out=3, in=4]`).
  //  (2) The reloaded weight equals `base_persisted + transpose(canonical
  //      delta)` where `canonical delta = scale * lora_b @ lora_a` in
  //      `[out, in]`.
  //
  // Fixture:
  //   base canonical `W` ([out=3, in=4]) — picks elements 0,1,2 of x:
  //     W = [[1,0,0,0],
  //          [0,1,0,0],
  //          [0,0,1,0]]
  //   base persisted `W_p = W^T` ([in=4, out=3])
  //   lora_A (PEFT disk) ([r=2, in=4]):
  //     A_disk = [[1,0,0,0],
  //               [0,1,0,0]]
  //   lora_B (PEFT disk) ([out=3, r=2]):
  //     B_disk = [[1,0],
  //               [0,1],
  //               [0,0]]
  //   scale = lora_alpha / r = 4.0 / 2 = 2.0
  //   canonical delta = scale * B_disk @ A_disk ([out, in])
  //                   = 2 * [[1,0,0,0],
  //                          [0,1,0,0],
  //                          [0,0,0,0]]
  //                   = [[2,0,0,0],
  //                      [0,2,0,0],
  //                      [0,0,0,0]]
  //   canonical fused = W + delta
  //                   = [[3,0,0,0],
  //                      [0,3,0,0],
  //                      [0,0,1,0]]
  //   persisted fused = canonical fused^T ([in=4, out=3])
  //                   = [[3,0,0],
  //                      [0,3,0],
  //                      [0,0,1],
  //                      [0,0,0]]
  let canonical_w = Array::from_slice::<f32>(
    &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    &(3, 4),
  )
  .unwrap();
  let persisted_w = canonical_w.transpose().unwrap(); // [4, 3]
  // The persisted shape is what the loader stores on disk for `fan_in_fan_out`.
  assert_eq!(persisted_w.shape(), &[4, 3]);

  let mut weights = HashMap::new();
  weights.insert(
    "model.layers.0.self_attn.q_proj.weight".to_string(),
    persisted_w,
  );
  let model_dir = write_base_dir("fifo_nonsq_model", &weights, &plain_config_json(1));

  // PEFT-disk lora_A: [r=2, in=4]; lora_B: [out=3, r=2].
  let lora_a_disk = Array::from_slice::<f32>(
    &[
      1.0, 0.0, 0.0, 0.0, // r=0
      0.0, 1.0, 0.0, 0.0, // r=1
    ],
    &(2, 4),
  )
  .unwrap();
  let lora_b_disk = Array::from_slice::<f32>(
    &[
      1.0, 0.0, // out=0
      0.0, 1.0, // out=1
      0.0, 0.0, // out=2
    ],
    &(3, 2),
  )
  .unwrap();
  let adapter_dir = write_peft_fifo_adapter(
    "fifo_nonsq_adapter",
    &[0],
    /* r */ 2,
    /* lora_alpha */ 4.0,
    &lora_a_disk,
    &lora_b_disk,
  );
  let save_dir = temp_dir("fifo_nonsq_save");
  fs::remove_dir_all(&save_dir).unwrap();

  // The fuse must succeed end-to-end (shape would mismatch without the
  // transpose-back fix because the loader's [in, out] expectation could not
  // match a [out, in]-saved fused weight on a re-load).
  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false).unwrap();

  let mut reloaded = load::load_weights(&save_dir).unwrap();
  let mut got = reloaded
    .remove("model.layers.0.self_attn.q_proj.weight")
    .expect("fused q_proj weight present");
  // (1) Persisted shape: [in=4, out=3].
  assert_eq!(
    got.shape(),
    &[4, 3],
    "saved fused weight stays in the persisted [in, out] layout"
  );
  // (2) Hand-traced expected persisted fused weight (see fixture above).
  let expected_persisted: Vec<f32> = vec![
    3.0, 0.0, 0.0, // in=0
    0.0, 3.0, 0.0, // in=1
    0.0, 0.0, 1.0, // in=2
    0.0, 0.0, 0.0, // in=3
  ];
  let vals = got.to_vec::<f32>().unwrap();
  for (i, (g, e)) in vals.iter().zip(expected_persisted.iter()).enumerate() {
    assert!(
      (g - e).abs() <= 1e-5,
      "persisted fused weight elt {i}: got {g}, expected {e} \
       (full: {vals:?} vs {expected_persisted:?})"
    );
  }
}

#[test]
fn fuse_preserves_fan_in_fan_out_layout_for_square_target() {
  // Square PEFT `fan_in_fan_out: true` base — the silent-transpose
  // corruption mode that a shape check cannot catch. We use a
  // NON-symmetric LoRA delta (a rank-1 lower-triangular update on the
  // bottom output row) so a transposed save produces a different matrix
  // (an updated right COLUMN instead).
  //
  // Fixture:
  //   base canonical `W` ([out=3, in=3]) — identity:
  //     W = [[1,0,0],
  //          [0,1,0],
  //          [0,0,1]]
  //   base persisted W_p = W^T = W (identity is symmetric — keeps the
  //                                  base in BOTH orientations equal so
  //                                  the asymmetry lives in the DELTA).
  //   lora_A (PEFT disk) ([r=1, in=3]):
  //     A_disk = [[1,1,1]]
  //   lora_B (PEFT disk) ([out=3, r=1]):
  //     B_disk = [[0],[0],[1]]   // only the last output row is updated
  //   scale = lora_alpha / r = 2.0 / 1 = 2.0
  //   canonical delta = scale * B_disk @ A_disk ([out=3, in=3])
  //                   = 2 * [[0,0,0],
  //                          [0,0,0],
  //                          [1,1,1]]
  //                   = [[0,0,0],
  //                      [0,0,0],
  //                      [2,2,2]]
  //   canonical fused = W + delta
  //                   = [[1,0,0],
  //                      [0,1,0],
  //                      [2,2,3]]                     ← bottom row updated
  //   persisted fused = canonical fused^T
  //                   = [[1,0,2],
  //                      [0,1,2],
  //                      [0,0,3]]                     ← right column updated
  //                   (clearly NOT equal to the canonical layout — the
  //                    silent-transpose bug would save the canonical
  //                    layout and the loader would re-transpose it back,
  //                    giving a model whose UPDATED ROW became an
  //                    UPDATED COLUMN — wrong output.)
  let canonical_w =
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &(3, 3)).unwrap();
  let persisted_w = canonical_w.transpose().unwrap(); // [3, 3] but stored as W^T

  let mut weights = HashMap::new();
  weights.insert(
    "model.layers.0.self_attn.q_proj.weight".to_string(),
    persisted_w,
  );
  let model_dir = write_base_dir("fifo_sq_model", &weights, &plain_config_json(1));

  let lora_a_disk = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(1, 3)).unwrap();
  let lora_b_disk = Array::from_slice::<f32>(&[0.0, 0.0, 1.0], &(3, 1)).unwrap();
  let adapter_dir = write_peft_fifo_adapter(
    "fifo_sq_adapter",
    &[0],
    /* r */ 1,
    /* lora_alpha */ 2.0,
    &lora_a_disk,
    &lora_b_disk,
  );
  let save_dir = temp_dir("fifo_sq_save");
  fs::remove_dir_all(&save_dir).unwrap();

  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false).unwrap();

  let mut reloaded = load::load_weights(&save_dir).unwrap();
  let mut got = reloaded
    .remove("model.layers.0.self_attn.q_proj.weight")
    .expect("fused square q_proj weight present");
  assert_eq!(
    got.shape(),
    &[3, 3],
    "shape preserved (square; the only signal is values)"
  );
  // Expected PERSISTED `[in, out]` (= canonical W_fused^T per the trace above).
  let expected_persisted: Vec<f32> = vec![
    1.0, 0.0, 2.0, // in=0
    0.0, 1.0, 2.0, // in=1
    0.0, 0.0, 3.0, // in=2
  ];
  let vals = got.to_vec::<f32>().unwrap();
  for (i, (g, e)) in vals.iter().zip(expected_persisted.iter()).enumerate() {
    assert!(
      (g - e).abs() <= 1e-5,
      "square persisted fused elt {i}: got {g}, expected {e} \
       (full: {vals:?} vs {expected_persisted:?}); a silent transpose \
       would produce the canonical layout `[[1,0,0],[0,1,0],[2,2,3]]` \
       which differs at multiple cells"
    );
  }
}

// ─────────────────────── tokenizer + extras copy ───────────────────────

/// A minimal valid `tokenizer.json` for the rust-tokenizers parser — a
/// WordLevel model with three tokens. Small (sub-KB) so the helper
/// stays self-contained and the parse cost stays negligible.
fn minimal_tokenizer_json() -> &'static str {
  r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "decoder": null,
    "model": {
      "type": "WordLevel",
      "vocab": { "<pad>": 0, "hello": 1, "world": 2 },
      "unk_token": "<pad>"
    }
  }"#
}

/// Write a model dir that ALSO carries the union of tokenizer extras the
/// loader / `copy_tokenizer_and_extras` cover — used by the loadable-output
/// and extras-copied tests. Returns the dir + a map of `basename → bytes`
/// so the caller can assert byte-identical content on the destination.
fn write_base_dir_with_tokenizer_extras(
  name: &str,
  weights: &HashMap<String, Array>,
  config_json: &str,
  extras: &[(&str, &[u8])],
) -> (PathBuf, HashMap<String, Vec<u8>>) {
  let dir = write_base_dir(name, weights, config_json);
  let mut written: HashMap<String, Vec<u8>> = HashMap::new();
  for (basename, bytes) in extras {
    let path = dir.join(basename);
    fs::write(&path, bytes).unwrap();
    written.insert((*basename).to_string(), bytes.to_vec());
  }
  (dir, written)
}

#[test]
fn fuse_output_is_loadable_through_default_lm_loader() {
  // `lm::load::load(dir)` constructs the tokenizer from the same dir
  // immediately after the weights / config (`load.rs:683-685`). A fused
  // dir that shipped weights + config only would silently fail this load
  // with a tokenizer-not-found error — the F2 fix is to copy tokenizer
  // files from the original model dir to the destination so the dir loads
  // end-to-end.
  let weights = toy_base_weights(2);
  let extras: &[(&str, &[u8])] = &[
    ("tokenizer.json", minimal_tokenizer_json().as_bytes()),
    (
      "tokenizer_config.json",
      br#"{"model_max_length": 32, "padding_side": "right"}"#,
    ),
  ];
  let (model_dir, _written) =
    write_base_dir_with_tokenizer_extras("loadable_model", &weights, &plain_config_json(2), extras);
  let adapter_dir = write_mlxlm_adapter("loadable_adapter", &[0, 1], 2.0);
  let save_dir = temp_dir("loadable_save");
  fs::remove_dir_all(&save_dir).unwrap();

  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false).unwrap();

  // The default LM loader: config + weights + tokenizer.
  let (_cfg, _w, tokenizer) =
    load::load(&save_dir).expect("fused dir loads end-to-end through the default LM loader");

  // The tokenizer was loaded from the COPIED file — same vocab as source.
  // Re-load directly from source to compare vocab-as-token-encoding.
  let (_src_cfg, _src_w, src_tokenizer) =
    load::load(&model_dir).expect("source dir loads (sanity)");
  let probe = "hello world";
  let dst_ids = tokenizer.encode(probe, true).expect("dst encode");
  let src_ids = src_tokenizer.encode(probe, true).expect("src encode");
  assert_eq!(
    dst_ids, src_ids,
    "copied tokenizer encodes {probe:?} the same way as the source"
  );
}

#[test]
fn fuse_copies_all_tokenizer_extras_present_in_source() {
  // All 4 extras-family files at `model_path` must land at `save_path`
  // with byte-identical content. We supply distinct, fingerprint-able
  // bytes per file so a mistakenly mixed-up copy (or a missed file) is
  // caught by the per-file content comparison.
  let weights = toy_base_weights(1);
  let chat_template_bytes = br#"{%- for m in messages %}{{ m.content }}{% endfor %}"#;
  let extras: &[(&str, &[u8])] = &[
    ("tokenizer.json", minimal_tokenizer_json().as_bytes()),
    (
      "tokenizer_config.json",
      br#"{"model_max_length": 64, "padding_side": "left"}"#,
    ),
    ("chat_template.jinja", chat_template_bytes),
    (
      "generation_config.json",
      br#"{"max_new_tokens": 256, "temperature": 0.7}"#,
    ),
  ];
  let (model_dir, written) =
    write_base_dir_with_tokenizer_extras("extras_model", &weights, &plain_config_json(1), extras);
  let adapter_dir = write_mlxlm_adapter("extras_adapter", &[0], 2.0);
  let save_dir = temp_dir("extras_save");
  fs::remove_dir_all(&save_dir).unwrap();

  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false).unwrap();

  for (basename, expected_bytes) in &written {
    let dst_path = save_dir.join(basename);
    assert!(
      dst_path.is_file(),
      "{basename}: present at destination after fuse"
    );
    let got_bytes = fs::read(&dst_path).unwrap();
    assert_eq!(
      &got_bytes, expected_bytes,
      "{basename}: byte-identical copy from source"
    );
  }
}

// ───────────────── R2 Finding 1 — single adapter-config snapshot ─────────────────

#[test]
fn fuse_load_adapters_with_config_skips_second_adapter_config_read() {
  // R2 Finding 1 — TOCTOU window between `read_adapter_config` (save-side
  // `fan_in_fan_out` decision) and `load_adapters`' INTERNAL second read of
  // the same `adapter_config.json` (load-side transpose / quant-arm
  // decisions). The fix is `load_adapters_with_config`, a variant that
  // takes a PRE-PARSED `LoraConfig` and skips the internal re-read.
  //
  // Empirical proof of single-snapshot:
  //  1. Parse `LoraConfig` from a valid adapter dir.
  //  2. CLOBBER `adapter_config.json` on disk with garbage that would FAIL
  //     a re-parse (so a sneaky second read can't pass as a no-op).
  //  3. Call `load_adapters_with_config` with the parsed config — MUST
  //     succeed (no second read happens).
  //  4. Call `load_adapters` (the wrapper) — MUST fail (it does its own
  //     read at the top, which now sees the corrupted body).
  //
  // Together (3) + (4) prove the two functions diverge exactly on the
  // adapter-config-read count: `_with_config` zero, the wrapper one. The
  // `fuse()` orchestrator funnels through `_with_config` (using the same
  // parsed config it consulted for `fan_in_fan_out`), so the load side
  // and the save side share a single snapshot.
  let weights = toy_base_weights(2);
  let _model_dir = write_base_dir("toctou_model", &weights, &plain_config_json(2));
  let adapter_dir = write_mlxlm_adapter("toctou_adapter", &[0, 1], 2.0);

  // (1) parse once
  let parsed_cfg = lora::read_adapter_config(&adapter_dir).expect("initial parse");

  // (2) replace the on-disk config with garbage — proves the in-memory
  // parsed_cfg is the ONLY source of truth on the load side. We keep the
  // dir + weights file intact and only corrupt the config (deletion would
  // also work, but `linear_to_lora_layers` depends on the adapter weights
  // file being present).
  fs::write(adapter_dir.join("adapter_config.json"), b"not json {{{")
    .expect("clobber adapter_config.json");

  // (3) load_adapters_with_config holds the pre-parsed config and does
  // NOT re-read the on-disk file → succeeds.
  let layers = lora::load_adapters_with_config(
    &weights,
    &adapter_dir,
    &parsed_cfg,
    /* quant */ None,
    /* num_blocks */ 2,
  )
  .expect("load_adapters_with_config must not re-read adapter_config.json");
  assert_eq!(
    layers.len(),
    2,
    "both q_proj layers reached the adapter pipeline via the pre-parsed config"
  );

  // (4) the wrapper re-parses internally — clobbered body must surface
  // as a JSON parse error from `LoraConfig::from_json`.
  let err = lora::load_adapters(
    &weights,
    &adapter_dir,
    /* quant */ None,
    /* num_blocks */ 2,
  )
  .expect_err("load_adapters wrapper re-reads adapter_config.json and must surface the corruption");
  match err {
    Error::Backend { message } => {
      assert!(
        !message.is_empty(),
        "wrapper error must carry a parse diagnostic: {message}"
      );
    }
    other => panic!("expected Error::Backend from the wrapper's re-parse, got {other:?}"),
  }
}
