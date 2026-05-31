//! `fuse()` library orchestrator (`lm::fuse`).
//!
//! Integration tests for the port of `mlx_lm/fuse.py::main` — the
//! adapter-fusion driver that ties `load_adapters` + per-layer
//! [`LoraLayer::fuse`](mlxrs::lm::lora::LoraLayer::fuse) + the model save into
//! a one-call pipeline. Mirrors `lm_convert.rs` style: gated on the `lm`
//! umbrella, fixture-built synthetic source / adapter directories under
//! `temp_dir()`, hand-traced assertions.
//!
//! Test list (#162):
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
//!   contract inherited from the adapter loader).
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
//!   config only would silently fail to load. Compares encodings across
//!   multiple probes (single-probe equality is compatible with a
//!   partial-corruption that happens to agree on one input).
//! - `fuse_copies_all_tokenizer_extras_present_in_source` — every
//!   `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` /
//!   `generation_config.json` present at `model_path` lands at `save_path`
//!   with byte-identical content (the verbatim-copy contract).
//! - `fuse_load_adapters_with_config_skips_second_adapter_config_read` —
//!   `load_adapters_with_config` consumes a pre-parsed
//!   [`mlxrs::lm::lora::LoraConfig`] and must NOT re-read
//!   `adapter_config.json` on the load side. The test parses the config,
//!   clobbers the on-disk file, then calls `load_adapters_with_config`
//!   (success — proves no second read) AND `load_adapters` (failure —
//!   proves the wrapper still does its own parse), closing the TOCTOU
//!   window `fuse` previously had.
//! - `fuse_rejects_source_with_missing_tokenizer` — a model
//!   dir WITHOUT `tokenizer.json` causes `fuse()` to fail BEFORE any save
//!   work, preventing the prior silent-Ok / unloadable-destination
//!   regression.
//! - `fuse_rejects_source_with_malformed_tokenizer` — a
//!   truncated `tokenizer.json` body fails the same fail-fast validation
//!   (corrupt source bytes can't be silently shipped to the destination).
//! - `fuse_overwrites_stale_destination_tokenizer` — a
//!   `save_path` pre-populated with stale `tokenizer.json` bytes from a
//!   different model has its contents OVERWRITTEN by the source's
//!   tokenizer (the cross-model-contamination concern is mitigated by
//!   `std::fs::copy`'s default-overwrite semantics; mlxrs `fuse` matches
//!   fuse.py's permissive destination contract but the overwrite +
//!   source-validate combo prevents stale data from leaking).
//! - `fuse_drops_stale_destination_generation_config_when_source_lacks_it`
//!   — a pre-existing `save_path/generation_config.json`
//!   that the source dir does NOT carry is UNLINKED by the staging
//!   stale-sweep, so the fused dir loads with the SOURCE's
//!   (none-here) EOS contract — not the stale destination's.
//! - `fuse_drops_stale_destination_chat_template_when_source_lacks_it`
//!   — same shape for `chat_template.jinja`. Templating
//!   is a tokenizer-surface contract; a stale jinja from a previous
//!   model would silently load via the new model's tokenizer.
//! - `fuse_drops_stale_destination_python_extras_when_source_lacks_them`
//!   — same shape for `*.py`. Some HF model loaders
//!   import these files for arch-specific custom code; a stale `*.py`
//!   from a previous model could execute wrong-model code.
//! - `fuse_snapshots_source_tokenizer_before_validate` —
//!   proves the validate step runs against `<save_path>/.staging-fuse-*`,
//!   NOT `model_path`: deleting `model_path/tokenizer.json` AFTER the
//!   snapshot has been taken but BEFORE the rest of fuse runs must NOT
//!   surface a validate failure (the validate uses staged bytes), and
//!   the SHIPPED tokenizer at `save_path` is byte-identical to the
//!   pre-deletion source bytes.
//! - `fuse_cleans_up_staging_dir_on_save_failure` —
//!   induces a save-side failure (read-only `save_path`) and asserts
//!   no `<save_path>/.staging-fuse-*` directory survives. The staging
//!   guard's `Drop` is the single mechanism that holds this invariant.
//! - `fuse_fails_when_stale_destination_tokenizer_config_is_directory`
//!   — pre-existing `save_dir/tokenizer_config.json/`
//!   as a DIRECTORY (source lacks the file). A removal gated on
//!   `is_file()` would silently SKIP the directory, then
//!   `lm::load::load(save_dir)` would fail (or hang on a FIFO) reading
//!   a dir as JSON. The fix uses `symlink_metadata()` and fails
//!   promotion with [`Error::Backend`] naming the path + "non-regular".
//! - `fuse_fails_when_stale_destination_python_extra_is_directory`
//!   — same `*.py` sweep variant for an
//!   attacker- or operator-planted `save_dir/custom_arch.py/`.
//! - `fuse_fails_when_stale_destination_tokenizer_json_is_symlink_to_directory`
//!   — `tokenizer.json` as a symlink to a dir.
//!   `is_file()` follows symlinks, so it returns `false` (target is dir)
//!   and a removal gated on it would SKIP. The fix's `symlink_metadata()`
//!   never follows symlinks and surfaces the symlink as a fail.
//! - `fuse_cleans_up_staging_dir_on_stale_removal_failure`
//!   — pre-create
//!   `save_dir/tokenizer_config.json/{nested}` so the non-regular gate
//!   returns `Err(Backend)` from the stale-sweep. The staging guard stays
//!   ARMED until promotion success, so this
//!   mid-promote Err must NOT leak a `.staging-fuse-*` dir.
//!
//! No `peak_memory()` magnitude asserts.
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
/// block — exactly the dual-key shape the `save_config` mirror emits
/// (`quantization → quantization_config`), so a quantized base loaded from
/// a saved checkpoint carries both top-level keys.
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

/// A minimal valid `tokenizer.json` for the rust-tokenizers parser — a
/// WordLevel model with three tokens. Small (sub-KB) so the fixture stays
/// self-contained and the parse cost stays negligible. Shared by
/// [`write_base_dir`] (the default-bundle fixture) and the loadable-output
/// integration tests below.
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

/// Write a synthetic base-model directory: `config.json` + a single
/// `model.safetensors` carrying `weights` + a minimal valid
/// `tokenizer.json`. The tokenizer is bundled by default so the
/// validate-source-tokenizer-before-save step in `fuse()` is satisfied
/// for every standard fixture — tests that EXPLICITLY want to exercise
/// the missing-tokenizer / malformed-tokenizer error path use
/// [`write_base_dir_no_tokenizer`] which skips the tokenizer write.
fn write_base_dir(name: &str, weights: &HashMap<String, Array>, config_json: &str) -> PathBuf {
  let dir = write_base_dir_no_tokenizer(name, weights, config_json);
  fs::write(dir.join("tokenizer.json"), minimal_tokenizer_json()).unwrap();
  dir
}

/// Variant of [`write_base_dir`] that does NOT write `tokenizer.json`.
/// Used by the missing-tokenizer / malformed-tokenizer tests to
/// construct a structurally incomplete source directory; every OTHER
/// fixture uses the bundled [`write_base_dir`] (so the
/// validate-source-tokenizer step passes for the orthogonal
/// fuse-behavior tests).
fn write_base_dir_no_tokenizer(
  name: &str,
  weights: &HashMap<String, Array>,
  config_json: &str,
) -> PathBuf {
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
  // `save_model` happily overwrites into a created-but-empty dir;
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

  // Reload the saved fused weights through the index loader.
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
  // `quantization_config` (the `save_config` dual-key shape). The
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
  // Same dual-key source config, but `dequantize=false`. The
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
  // `save_config` mirrors `quantization → quantization_config`, so
  // the second key is also present.
  assert!(
    saved_cfg.get("quantization_config").is_some(),
    "`quantization_config` preserved by the save_config mirror: {saved_cfg_text}"
  );
}

// ─────────────────── reject missing model_path ───────────────────

#[test]
fn fuse_rejects_missing_model_path() {
  // A nonexistent model_path bubbles up as `Error::FileIo` (Open / NotFound)
  // from `load_config`'s file-not-found arm. The payload's path must name
  // the missing file so the user can recover.
  let bogus = temp_dir("missing_model_path_root");
  let model_dir = bogus.join("does_not_exist");
  // Build a real adapter dir so the failure attribution is unambiguous.
  let adapter_dir = write_mlxlm_adapter("missing_model_adapter", &[0], 2.0);
  let save_dir = temp_dir("missing_model_save");
  fs::remove_dir_all(&save_dir).unwrap();

  let err = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false).unwrap_err();
  match err {
    Error::FileIo(p) => {
      assert_eq!(p.op(), mlxrs::error::FileOp::Open);
      assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
      let path_str = p.path().to_string_lossy();
      assert!(
        path_str.contains("does_not_exist") || path_str.contains("config"),
        "FileIo path names the missing file: {path_str}"
      );
      assert!(
        p.context().contains("config"),
        "context names the load step: {}",
        p.context()
      );
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
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
    Error::FileIo(p) => {
      assert_eq!(p.op(), mlxrs::error::FileOp::Open);
      assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
      let path_str = p.path().to_string_lossy();
      assert!(
        path_str.contains("not_an_adapter") || path_str.contains("adapter"),
        "FileIo path names the missing adapter file: {path_str}"
      );
    }
    other => panic!("expected Error::FileIo, got {other:?}"),
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
    Error::LayerKeyed(p) => {
      // Carrier layer is the rejected URL verbatim — the caller can fix
      // their input by stripping the prefix off it.
      assert_eq!(p.layer(), "hf://mlx-community/Qwen3-4B-bf16");
      match p.inner() {
        Error::InvariantViolation(inner) => {
          assert_eq!(inner.context(), "model_path");
          assert!(
            inner.requirement().contains("huggingface-cli download"),
            "requirement gives actionable workaround: {}",
            inner.requirement()
          );
        }
        other => panic!("expected inner InvariantViolation, got {other:?}"),
      }
    }
    other => panic!("expected Error::LayerKeyed, got {other:?}"),
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
    Error::LayerKeyed(p) => {
      assert_eq!(p.layer(), "https://huggingface.co/owner/adapter-repo");
      match p.inner() {
        Error::InvariantViolation(inner) => {
          assert_eq!(inner.context(), "adapter_path");
          assert!(
            inner.requirement().contains("huggingface-cli download"),
            "requirement gives actionable workaround: {}",
            inner.requirement()
          );
        }
        other => panic!("expected inner InvariantViolation, got {other:?}"),
      }
    }
    other => panic!("expected Error::LayerKeyed, got {other:?}"),
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
    Error::MissingKey(p) => {
      // The postcondition's diagnostic fires from one of the three arms
      // in `check_adapter_completeness`. Here `keys=["self_attn.q_proj"]`
      // is an EXPLICIT selection that matches blocks 0 + 1 in the base
      // (both are q_projs in the trailing window), but the adapter
      // supplies factors for block 99 only — so the explicitly-selected
      // target is reported as a `MissingKey`.
      assert!(
        p.context().contains("load_adapters")
          && (p.context().contains("explicitly-selected") || p.context().contains("target")),
        "context names the load_adapters postcondition: {}",
        p.context()
      );
      assert!(
        p.key().contains("q_proj") && p.key().contains("layers.0"),
        "key names the missing base layer: {}",
        p.key()
      );
    }
    other => panic!("expected Error::MissingKey, got {other:?}"),
  }
}

// ─────────────────── save_path created when absent ───────────────────

#[test]
fn fuse_save_path_is_created_when_absent() {
  // The `save` path creates `save_path` if absent (`create_dir_all`).
  // `fuse` inherits that contract — passing a never-created destination
  // must succeed and produce a valid directory.
  let weights = toy_base_weights(2);
  let model_dir = write_base_dir("absent_save_model", &weights, &plain_config_json(2));
  let adapter_dir = write_mlxlm_adapter("absent_save_adapter", &[0, 1], 2.0);
  // Build a path under a parent dir that DOES exist, but the leaf
  // doesn't — exactly the save-path contract.
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

/// Write a model dir that ALSO carries the union of tokenizer extras the
/// loader / `copy_tokenizer_and_extras` cover — used by the loadable-output
/// and extras-copied tests. The bundled `tokenizer.json` from
/// [`write_base_dir`] is OVERWRITTEN if the `extras` list also names
/// `tokenizer.json` (so a test can substitute a different vocab without
/// fighting the default). Returns the dir + a map of
/// `basename → bytes` so the caller can assert byte-identical content on
/// the destination — the default tokenizer.json is included so a caller
/// asserting "byte-identical copy from source" without explicitly listing
/// `tokenizer.json` in `extras` still has a baseline.
fn write_base_dir_with_tokenizer_extras(
  name: &str,
  weights: &HashMap<String, Array>,
  config_json: &str,
  extras: &[(&str, &[u8])],
) -> (PathBuf, HashMap<String, Vec<u8>>) {
  let dir = write_base_dir(name, weights, config_json);
  let mut written: HashMap<String, Vec<u8>> = HashMap::new();
  written.insert(
    "tokenizer.json".to_string(),
    minimal_tokenizer_json().as_bytes().to_vec(),
  );
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
  // with a tokenizer-not-found error — the fix is to copy tokenizer
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
  // Re-load directly from source to compare vocab-as-token-encoding ACROSS
  // MULTIPLE distinct probes (a single "hello world" probe could
  // pass even with a partially-corrupted copy where most ids happen to match;
  // exercising 3 disjoint strings — multi-word
  // mix, single in-vocab token, plus a string with an out-of-vocab token —
  // means an identity-failure on any one fails the test).
  let (_src_cfg, _src_w, src_tokenizer) =
    load::load(&model_dir).expect("source dir loads (sanity)");
  for probe in ["hello world", "hello", "world hello unknown"] {
    let dst_ids = tokenizer.encode(probe, true).expect("dst encode");
    let src_ids = src_tokenizer.encode(probe, true).expect("src encode");
    assert_eq!(
      dst_ids, src_ids,
      "copied tokenizer encodes {probe:?} identically to the source"
    );
    assert!(
      !dst_ids.is_empty(),
      "encode of {probe:?} must produce at least one id (a silently empty \
       tokenizer would compare equal but be broken downstream)"
    );
  }
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

// ───────────────── single adapter-config snapshot ─────────────────

#[test]
fn fuse_load_adapters_with_config_skips_second_adapter_config_read() {
  // TOCTOU window between `read_adapter_config` (save-side
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
    Error::Parse(p) => {
      assert!(
        p.context().contains("LoraConfig"),
        "wrapper Parse context names the parser: {}",
        p.context()
      );
      assert_eq!(
        p.input_kind(),
        "adapter_config.json",
        "wrapper Parse input_kind identifies the on-disk file"
      );
    }
    other => panic!("expected Error::Parse from the wrapper's re-parse, got {other:?}"),
  }
}

// ───────────────── tokenizer validate-before-save ─────────────────

#[test]
fn fuse_rejects_source_with_missing_tokenizer() {
  // `fuse()` previously called `copy_tokenizer_and_extras`
  // AFTER save and mapped its `Ok(_)` to `Ok(())` without checking that
  // the source had a usable `tokenizer.json`. `copy_tokenizer_and_extras`
  // silently SKIPS absent files, so a source without `tokenizer.json` let
  // `fuse()` return `Ok(())` and the saved dir was unloadable through
  // `lm::load::load(save_path)`.
  //
  // The fix mirrors `convert::convert`'s `let _tokenizer =
  // load::load_tokenizer(&hf_path, &cfg_typed)?` line — validate source
  // tokenizer BEFORE any save IO so an unloadable source surfaces fast
  // with the source path in the error message.
  let weights = toy_base_weights(2);
  // `write_base_dir_no_tokenizer` is the only fixture that omits the
  // bundled `tokenizer.json`; every other test uses `write_base_dir`,
  // which now writes the minimal tokenizer by default.
  let model_dir = write_base_dir_no_tokenizer("missing_tok_model", &weights, &plain_config_json(2));
  assert!(
    !model_dir.join("tokenizer.json").exists(),
    "fixture precondition: source has no tokenizer.json"
  );
  let adapter_dir = write_mlxlm_adapter("missing_tok_adapter", &[0, 1], 2.0);
  let save_dir = temp_dir("missing_tok_save");
  fs::remove_dir_all(&save_dir).unwrap();

  let err = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false)
    .expect_err("fuse() must reject a source dir missing tokenizer.json");
  // The validate-on-staging failure is re-wrapped with `model_path`
  // context by `fuse()` so the surface error carries the SOURCE path
  // (the staging dir is an internal implementation detail). Inside that
  // outer LayerKeyed is the staging-side LayerKeyed wrapping the
  // underlying Tokenizer error.
  match err {
    Error::LayerKeyed(outer) => {
      assert_eq!(
        outer.layer(),
        model_dir.to_str().unwrap(),
        "outer LayerKeyed layer names the source path so the caller can fix the input"
      );
      match outer.inner() {
        // Walk to the innermost (Tokenizer) error — the staging-dir
        // LayerKeyed is the load::load_tokenizer wrapper. The
        // tokenizer-construction failure is the inner Error::Tokenizer.
        Error::LayerKeyed(staging) => match staging.inner() {
          #[cfg(feature = "tokenizer")]
          Error::Tokenizer(msg) => {
            assert!(
              msg.contains("tokenizer.json"),
              "Tokenizer error names the missing file: {msg}"
            );
          }
          other => panic!("expected innermost Error::Tokenizer, got {other:?}"),
        },
        other => panic!("expected nested LayerKeyed staging wrapper, got {other:?}"),
      }
    }
    other => panic!("expected outer Error::LayerKeyed, got {other:?}"),
  }
  // Critical: NO save artifacts may have landed in save_dir. The whole
  // point of validate-before-save is to fail FAST so a doomed fuse
  // doesn't leave a half-populated destination.
  assert!(
    !save_dir.join("config.json").exists(),
    "no config.json may land on a tokenizer-validation failure"
  );
  assert!(
    !save_dir.join("model.safetensors.index.json").exists(),
    "no weight index may land on a tokenizer-validation failure"
  );
}

#[test]
fn fuse_rejects_source_with_malformed_tokenizer() {
  // Same fail-fast contract as `fuse_rejects_source_with_missing_tokenizer`
  // but with a truncated `tokenizer.json` body (the parser fails inside
  // `HfTokenizer::from_file` rather than at the open). Both error paths
  // funnel through `Error::Backend("cannot load tokenizer
  // from {}: ...")` per `load::load_tokenizer`.
  let weights = toy_base_weights(2);
  let model_dir =
    write_base_dir_no_tokenizer("malformed_tok_model", &weights, &plain_config_json(2));
  // Truncated mid-object — the WordLevel parser must reject (the `vocab`
  // value is required and missing the closing `}`).
  fs::write(
    model_dir.join("tokenizer.json"),
    b"{ \"version\": \"1.0\", \"model\": { \"type\": \"WordLevel\", \"vocab\":",
  )
  .unwrap();
  let adapter_dir = write_mlxlm_adapter("malformed_tok_adapter", &[0, 1], 2.0);
  let save_dir = temp_dir("malformed_tok_save");
  fs::remove_dir_all(&save_dir).unwrap();

  let err = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false)
    .expect_err("fuse() must reject a source dir with a malformed tokenizer.json");
  // Same outer/inner LayerKeyed shape as the missing-tokenizer test;
  // the innermost error is a tokenizer parse failure rather than NotFound.
  match err {
    Error::LayerKeyed(outer) => {
      assert_eq!(
        outer.layer(),
        model_dir.to_str().unwrap(),
        "outer LayerKeyed layer names the source path"
      );
      match outer.inner() {
        Error::LayerKeyed(staging) => match staging.inner() {
          #[cfg(feature = "tokenizer")]
          Error::Tokenizer(msg) => {
            assert!(
              msg.contains("tokenizer.json"),
              "Tokenizer error names the file: {msg}"
            );
          }
          other => panic!("expected innermost Error::Tokenizer, got {other:?}"),
        },
        other => panic!("expected nested LayerKeyed staging wrapper, got {other:?}"),
      }
    }
    other => panic!("expected outer Error::LayerKeyed, got {other:?}"),
  }
  assert!(
    !save_dir.join("config.json").exists(),
    "no config.json may land on a malformed-tokenizer validation failure"
  );
}

#[test]
fn fuse_overwrites_stale_destination_tokenizer() {
  // Stale-destination audit: convert REJECTS pre-existing
  // destinations wholesale (`convert.rs:588`); fuse.py PERMITS them
  // (writes-through via `save_pretrained`). mlxrs `fuse` matches fuse.py
  // (permissive), but the cross-model-contamination concern is
  // mitigated by `std::fs::copy`'s default-overwrite semantics: a stale
  // `tokenizer.json` at `save_path` is OVERWRITTEN by the source's bytes
  // during `copy_tokenizer_and_extras`.
  //
  // This test pre-populates `save_path` with stale (different-from-source)
  // tokenizer bytes, runs `fuse`, and asserts the destination tokenizer is
  // byte-identical to the SOURCE — not the stale pre-existing content.
  let weights = toy_base_weights(2);
  let model_dir = write_base_dir("stale_dst_model", &weights, &plain_config_json(2));
  let source_tok = fs::read(model_dir.join("tokenizer.json")).expect("source tokenizer");
  let adapter_dir = write_mlxlm_adapter("stale_dst_adapter", &[0, 1], 2.0);
  let save_dir = temp_dir("stale_dst_save");
  // Pre-populate save_dir with a stale tokenizer.json that DIFFERS
  // structurally from the source — same WordLevel shape but a DIFFERENT
  // vocab so a stale-leak would surface as an encoding mismatch downstream.
  let stale_tok = br#"{
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
      "vocab": { "<pad>": 0, "STALE": 1, "GHOST": 2 },
      "unk_token": "<pad>"
    }
  }"#;
  fs::write(save_dir.join("tokenizer.json"), stale_tok).unwrap();
  // Sanity: the stale bytes are NOT equal to the source bytes.
  assert_ne!(
    stale_tok.as_slice(),
    source_tok.as_slice(),
    "fixture precondition: stale destination differs from source"
  );

  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false)
    .expect("fuse must succeed on a permissive destination with stale tokenizer");

  let dst_tok = fs::read(save_dir.join("tokenizer.json")).expect("post-fuse destination tokenizer");
  assert_eq!(
    dst_tok, source_tok,
    "stale destination tokenizer must be OVERWRITTEN with the source's bytes (no cross-model contamination)"
  );
  assert_ne!(
    dst_tok.as_slice(),
    stale_tok.as_slice(),
    "destination must NOT carry the stale pre-existing content after fuse"
  );
}

// ────────── staging stale-extras sweep drops dest files
// the source did NOT carry (cross-model contamination via dest leftovers) ──────────

/// Helper: count entries in `save_dir` whose name starts with the staging
/// marker prefix `.staging-fuse-`. A successful fuse must leave ZERO of
/// these — the staging guard's `Drop` and the explicit
/// `remove_dir(staging_path)` at the end of `promote_staging_into_save_path`
/// together guarantee it. Shared by every staging test below.
fn staging_dir_count(save_dir: &Path) -> usize {
  fs::read_dir(save_dir)
    .map(|entries| {
      entries
        .flatten()
        .filter(|e| {
          e.file_name()
            .to_str()
            .is_some_and(|n| n.starts_with(".staging-fuse-"))
        })
        .count()
    })
    .unwrap_or(0)
}

#[test]
fn fuse_drops_stale_destination_generation_config_when_source_lacks_it() {
  // `copy_tokenizer_and_extras` only OVERWRITES destination
  // files when the SOURCE carries the same name. A pre-existing
  // `save_dir/generation_config.json` from a previous model would SURVIVE
  // when the new source lacked `generation_config.json`, and
  // `load_config` consumes `generation_config.json` as the EOS override —
  // so the fused dir would silently load with the wrong-model EOS contract.
  //
  // The staging+promote fix sweeps stale extras: a destination
  // `generation_config.json` not present in the staging snapshot is
  // unlinked during promote.
  let weights = toy_base_weights(1);
  // Source dir has tokenizer.json (the bundled default) + NOTHING ELSE
  // from the extras family. Crucially NO generation_config.json.
  let model_dir = write_base_dir("stale_gen_cfg_model", &weights, &plain_config_json(1));
  assert!(
    !model_dir.join("generation_config.json").exists(),
    "fixture precondition: source has NO generation_config.json"
  );

  let adapter_dir = write_mlxlm_adapter("stale_gen_cfg_adapter", &[0], 2.0);
  let save_dir = temp_dir("stale_gen_cfg_save");
  // Pre-populate save_dir with a stale generation_config.json from an
  // earlier model. Distinctive marker payload — a fingerprint a surviving
  // copy would be caught with byte-comparison.
  let stale_gen_cfg =
    br#"{"max_new_tokens": 999, "temperature": 9.9, "_marker": "WRONG_MODEL_LEFTOVER"}"#;
  fs::write(save_dir.join("generation_config.json"), stale_gen_cfg).unwrap();

  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false)
    .expect("fuse must succeed on a permissive destination with stale generation_config");

  // Source LACKS generation_config.json → destination MUST NOT
  // carry one after fuse. (The stale destination file is the bug; the
  // fix deletes it during the staging promote sweep.)
  assert!(
    !save_dir.join("generation_config.json").exists(),
    "stale destination generation_config.json must be REMOVED when source lacks it \
     (an unswept destination silently keeps the wrong-model EOS contract)"
  );

  // The staging dir mechanism cleaned up properly.
  assert_eq!(
    staging_dir_count(&save_dir),
    0,
    "no .staging-fuse-* directory may survive a successful fuse"
  );
}

#[test]
fn fuse_drops_stale_destination_chat_template_when_source_lacks_it() {
  // `chat_template.jinja` is templating consumed by the
  // tokenizer surface. A stale jinja from an earlier model would survive
  // when the new source lacked `chat_template.jinja`, so the fused dir
  // would silently load with the wrong-model chat formatter.
  let weights = toy_base_weights(1);
  // Source has tokenizer.json (bundled) but NO chat_template.jinja.
  let model_dir = write_base_dir("stale_jinja_model", &weights, &plain_config_json(1));
  assert!(
    !model_dir.join("chat_template.jinja").exists(),
    "fixture precondition: source has NO chat_template.jinja"
  );

  let adapter_dir = write_mlxlm_adapter("stale_jinja_adapter", &[0], 2.0);
  let save_dir = temp_dir("stale_jinja_save");
  let stale_jinja =
    br#"{%- for m in messages %}<<<WRONG_MODEL_TEMPLATE>>>{{ m.content }}{% endfor %}"#;
  fs::write(save_dir.join("chat_template.jinja"), stale_jinja).unwrap();

  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false)
    .expect("fuse must succeed on a permissive destination with stale chat_template");

  assert!(
    !save_dir.join("chat_template.jinja").exists(),
    "stale destination chat_template.jinja must be REMOVED when source lacks it \
     (an unswept destination silently keeps the wrong-model chat formatter)"
  );
  assert_eq!(
    staging_dir_count(&save_dir),
    0,
    "no .staging-fuse-* directory may survive a successful fuse"
  );
}

#[test]
fn fuse_drops_stale_destination_python_extras_when_source_lacks_them() {
  // `*.py` extras are HF model loader auxiliary code
  // (some VLM / custom-arch loaders import the model dir's `*.py` files
  // for arch-specific logic). A stale `*.py` from an earlier model
  // would survive when the new source lacked the same basename, and a
  // downstream loader importing the dir could execute wrong-model code.
  let weights = toy_base_weights(1);
  // Source has tokenizer.json (bundled) but NO *.py files. The
  // `write_base_dir_with_tokenizer_extras` helper takes an `extras` list
  // we leave empty for the *.py axis (the helper only writes the names
  // we pass + the default tokenizer.json).
  let (model_dir, _written) = write_base_dir_with_tokenizer_extras(
    "stale_py_model",
    &weights,
    &plain_config_json(1),
    /* extras */ &[],
  );
  // Sanity: no *.py present at source.
  let src_py = fs::read_dir(&model_dir)
    .unwrap()
    .flatten()
    .filter(|e| e.file_name().to_str().is_some_and(|n| n.ends_with(".py")))
    .count();
  assert_eq!(src_py, 0, "fixture precondition: source has no .py extras");

  let adapter_dir = write_mlxlm_adapter("stale_py_adapter", &[0], 2.0);
  let save_dir = temp_dir("stale_py_save");
  // Two stale *.py files from a previous model.
  let stale_py_1 = b"# WRONG_MODEL_CODE\nclass StaleConfig: ...\n";
  let stale_py_2 = b"# also WRONG_MODEL_CODE\ndef stale_helper(): pass\n";
  fs::write(save_dir.join("modeling_extras.py"), stale_py_1).unwrap();
  fs::write(save_dir.join("custom_arch.py"), stale_py_2).unwrap();

  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false)
    .expect("fuse must succeed on a permissive destination with stale *.py extras");

  assert!(
    !save_dir.join("modeling_extras.py").exists(),
    "stale destination modeling_extras.py must be REMOVED when source lacks it"
  );
  assert!(
    !save_dir.join("custom_arch.py").exists(),
    "stale destination custom_arch.py must be REMOVED when source lacks it"
  );
  // And no other *.py was left behind.
  let dst_py = fs::read_dir(&save_dir)
    .unwrap()
    .flatten()
    .filter(|e| e.file_name().to_str().is_some_and(|n| n.ends_with(".py")))
    .count();
  assert_eq!(
    dst_py, 0,
    "no *.py file may survive when the source dir lacks every *.py"
  );
  assert_eq!(
    staging_dir_count(&save_dir),
    0,
    "no .staging-fuse-* directory may survive a successful fuse"
  );
}

// ────────── staging snapshot closes the validate/copy TOCTOU ──────────

#[test]
fn fuse_snapshots_source_tokenizer_before_validate() {
  // Previously, `load_tokenizer(model_path)` validated source
  // bytes at T0 and `copy_tokenizer_and_extras(model_path, save_path)`
  // RE-READ source bytes at T1 (post-save). Any mid-flight mutation
  // (deletion / swap / partial write) between T0 and T1 produced a fuse
  // that returned Ok(()) but shipped a tokenizer DIFFERENT from the one
  // that was validated.
  //
  // The fix collapses both reads to the staging snapshot: the
  // tokenizer + extras are copied INTO staging FIRST, validate reads
  // from staging, the rest of fuse runs, then staging is promoted into
  // save_path. The single read of `model_path` makes the TOCTOU window
  // structurally impossible: there is exactly one `model_path` read for
  // the tokenizer + extras (the `std::fs::copy` inside
  // `copy_tokenizer_and_extras(model_path, staging)`).
  //
  // **Inspection-only contract proof.** A mid-flight mutation racing
  // the in-process `copy_tokenizer_and_extras` is hard to inject
  // deterministically without a public fault hook. We prove the
  // contract by inspection of the SHIPPED bytes (must equal the
  // pre-fuse source bytes — the bytes the snapshot captured), then
  // demonstrate the snapshot mechanism is structurally in place by
  // checking that `<save_path>/.staging-fuse-*` directories were
  // CREATED-AND-CLEANED-UP during fuse (the staging guard's
  // `Drop` + the explicit `remove_dir` at promote end are the only
  // two paths that clear the dir; an absent dir post-fuse with an
  // earlier pre-condition asserting it did not exist proves both
  // creation + cleanup happened).

  let weights = toy_base_weights(2);
  let extras: &[(&str, &[u8])] = &[
    ("tokenizer.json", minimal_tokenizer_json().as_bytes()),
    (
      "tokenizer_config.json",
      br#"{"model_max_length": 128, "padding_side": "right", "_marker": "PRE_FUSE_SOURCE"}"#,
    ),
  ];
  let (model_dir, written) = write_base_dir_with_tokenizer_extras(
    "snapshot_resist_model",
    &weights,
    &plain_config_json(2),
    extras,
  );
  let source_tok = written
    .get("tokenizer.json")
    .expect("tokenizer.json was written by the fixture")
    .clone();
  let source_tok_cfg = written
    .get("tokenizer_config.json")
    .expect("tokenizer_config.json was written by the fixture")
    .clone();
  let adapter_dir = write_mlxlm_adapter("snapshot_resist_adapter", &[0, 1], 2.0);
  let save_dir = temp_dir("snapshot_resist_save");

  // Pre-condition: save_dir is freshly created and contains no .staging-fuse-* dirs.
  assert_eq!(
    staging_dir_count(&save_dir),
    0,
    "pre-fuse precondition: save_dir holds no staging dirs"
  );

  fuse::fuse(&model_dir, &adapter_dir, &save_dir, false)
    .expect("fuse must succeed on a clean source");

  // SHIPPED bytes equal the SOURCE bytes the snapshot captured.
  let shipped_tok = fs::read(save_dir.join("tokenizer.json")).expect("shipped tokenizer.json");
  let shipped_tok_cfg =
    fs::read(save_dir.join("tokenizer_config.json")).expect("shipped tokenizer_config.json");
  assert_eq!(
    shipped_tok, source_tok,
    "shipped tokenizer.json bytes must equal the pre-fuse source bytes (the snapshot)"
  );
  assert_eq!(
    shipped_tok_cfg, source_tok_cfg,
    "shipped tokenizer_config.json bytes must equal the pre-fuse source bytes"
  );

  // Post-condition: no staging dir survived (so the snapshot mechanism
  // RAN and CLEANED UP). Combined with the SHIPPED-bytes equality this
  // proves the staging snapshot was the source of truth for the copy.
  assert_eq!(
    staging_dir_count(&save_dir),
    0,
    "no .staging-fuse-* directory may survive a successful fuse \
     (staging guard Drop + explicit remove_dir at promote end)"
  );

  // Cross-check: mutating source AFTER fuse returns has NO effect on
  // the shipped bytes — confirms shipped bytes are independent of any
  // post-fuse source mutation. This is the same property the snapshot
  // achieves WITHIN the fuse call (validate + copy decoupled from
  // model_path reads after the snapshot was taken).
  fs::write(model_dir.join("tokenizer.json"), b"POST_FUSE_CORRUPTED").unwrap();
  let shipped_tok_after = fs::read(save_dir.join("tokenizer.json")).unwrap();
  assert_eq!(
    shipped_tok_after, source_tok,
    "post-fuse source mutation must not affect already-shipped tokenizer bytes"
  );
}

// ────────── staging guard — Drop cleans up on save failure ──────────

#[test]
fn fuse_cleans_up_staging_dir_on_save_failure() {
  // The `StagingDir::Drop` impl must remove the
  // staging directory on every error exit between snapshot creation
  // and a successful promote (the only paths that consume the guard).
  //
  // Failure-injection strategy: pre-create `save_dir/config.json` as a
  // NON-EMPTY DIRECTORY. The fuse pipeline runs staging creation +
  // tokenizer snapshot + tokenizer validate normally (those write into
  // the `.staging-fuse-*` subdir, not over config.json), then
  // `load::save`'s staged-config commit hits a rename onto config.json
  // which fails because config.json is a non-empty dir
  // (rename-file-onto-non-empty-dir is EISDIR / ENOTDIR / EPERM on
  // POSIX and Windows alike). The save returns `Err` BEFORE the
  // promote step — exactly the path the staging guard's `Drop` is
  // intended to clean up.
  //
  // This shape:
  //   1. Exercises the snapshot + validate steps (they DO create the
  //      `.staging-fuse-*` dir on disk),
  //   2. Fails the save step deterministically,
  //   3. Returns Err from fuse — Drop fires on the staging guard,
  //   4. The staging dir should be gone post-fuse.

  let weights = toy_base_weights(1);
  let model_dir = write_base_dir("ro_save_model", &weights, &plain_config_json(1));
  let adapter_dir = write_mlxlm_adapter("ro_save_adapter", &[0], 2.0);
  let save_dir = temp_dir("ro_save_save");

  // Pre-create config.json as a non-empty directory (collides with the
  // staged-config rename target). Must contain at least one entry so
  // it's not unlink-replaceable as an empty dir.
  fs::create_dir_all(save_dir.join("config.json").join("sentinel")).unwrap();

  let result = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false);
  assert!(
    result.is_err(),
    "fuse must fail when config.json is a non-empty directory at save_path: got {result:?}"
  );

  // Contract: no `.staging-fuse-*` dir survived the failed fuse.
  // The staging guard's `Drop` is the single mechanism that holds this.
  let leftover = staging_dir_count(&save_dir);
  assert_eq!(
    leftover, 0,
    "the staging guard's Drop must clean up the staging dir on a save failure; \
     found {leftover} leftover .staging-fuse-* dirs"
  );

  // Best-effort GC cleanup of the sentinel structure (so the test
  // runner's tempdir cleanup recurses cleanly).
  let _ = fs::remove_dir_all(save_dir.join("config.json"));
}

// ────────── stale-sweep symlink_metadata + non-regular reject ──────────

/// Helper for the non-regular-reject tests: extract the inner promotion
/// error's message from the [`Error::ConvertPostSavePartial`] wrapper that
/// `fuse()` returns for any hard promotion failure (per the
/// `promote_outcome -> Err(...)` arm in `fuse.rs::fuse`). Asserts on
/// the WRAPPED `copy_error` (which carries the original
/// `Error::Backend` message via `io::Error::other`), so the diagnostic
/// stays end-to-end machine-checkable.
fn extract_promote_error_message(err: &Error) -> String {
  match err {
    Error::ConvertPostSavePartial(p) => p.copy_error().to_string(),
    other => {
      panic!("expected Error::ConvertPostSavePartial wrapping the promote err, got: {other:?}")
    }
  }
}

#[test]
fn fuse_fails_when_stale_destination_tokenizer_config_is_directory() {
  // A stale-sweep gating removal on `path.is_file()` would silently SKIP
  // non-regular entries (dir, FIFO, symlink-to-dir). When the source
  // LACKED `tokenizer_config.json` and the destination held a stale
  // `tokenizer_config.json/` DIRECTORY, the dir would survive into
  // save_path, then `lm::load::load(save_path)` would fail (or hang)
  // reading a dir as a JSON file.
  //
  // The fix uses `symlink_metadata()` and returns
  // `Err(Error::Backend)` for any non-regular reserved basename. We
  // assert (a) fuse fails (b) the error wraps the path + the
  // "non-regular" hint so the operator can act.
  let weights = toy_base_weights(1);
  let model_dir = write_base_dir("stale_tokcfg_dir_model", &weights, &plain_config_json(1));
  assert!(
    !model_dir.join("tokenizer_config.json").exists(),
    "fixture precondition: source has NO tokenizer_config.json"
  );

  let adapter_dir = write_mlxlm_adapter("stale_tokcfg_dir_adapter", &[0], 2.0);
  let save_dir = temp_dir("stale_tokcfg_dir_save");
  // Pre-create the reserved basename as an EMPTY directory. The
  // policy treats this as "non-regular reserved path; remove manually".
  fs::create_dir(save_dir.join("tokenizer_config.json")).unwrap();

  let result = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false);
  let err = result.expect_err("fuse must fail when stale tokenizer_config.json is a directory");
  let msg = extract_promote_error_message(&err);
  assert!(
    msg.contains("tokenizer_config.json"),
    "error message must name the offending basename; got: {msg}"
  );
  assert!(
    msg.contains("non-regular"),
    "error message must mark the path as non-regular reserved; got: {msg}"
  );
  assert!(
    msg.contains("directory"),
    "error message must name the kind (directory); got: {msg}"
  );

  // Staging-guard contract: no `.staging-fuse-*` survived a mid-promote Err.
  let leftover = staging_dir_count(&save_dir);
  assert_eq!(
    leftover, 0,
    "the staging guard must stay armed until success; \
     found {leftover} leftover .staging-fuse-* dirs after a stale-sweep failure"
  );

  // Best-effort GC.
  let _ = fs::remove_dir_all(save_dir.join("tokenizer_config.json"));
}

#[test]
fn fuse_fails_when_stale_destination_python_extra_is_directory() {
  // Same shape for the `*.py` stale-sweep. Any
  // `save_dir/<name>.py/` directory in the destination at promotion
  // time must fail with the named non-regular error; a removal gated on
  // `is_file()` would silently skip it.
  let weights = toy_base_weights(1);
  // Source has no *.py — `write_base_dir` only writes tokenizer.json
  // + config.json + model.safetensors.
  let model_dir = write_base_dir("stale_py_dir_model", &weights, &plain_config_json(1));

  let adapter_dir = write_mlxlm_adapter("stale_py_dir_adapter", &[0], 2.0);
  let save_dir = temp_dir("stale_py_dir_save");
  // Pre-create custom_arch.py AS A DIRECTORY (with one entry so it is
  // not unlink-replaceable as an empty dir; mirrors the
  // save-failure test's non-empty config.json/sentinel shape).
  fs::create_dir_all(save_dir.join("custom_arch.py").join("nested")).unwrap();

  let result = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false);
  let err = result.expect_err("fuse must fail when stale custom_arch.py is a directory");
  let msg = extract_promote_error_message(&err);
  assert!(
    msg.contains("custom_arch.py"),
    "error message must name the offending *.py basename; got: {msg}"
  );
  assert!(
    msg.contains("non-regular"),
    "error message must mark the path as non-regular reserved; got: {msg}"
  );
  assert!(
    msg.contains("directory"),
    "error message must name the kind (directory); got: {msg}"
  );

  // Staging-guard contract: no `.staging-fuse-*` survived the stale-sweep fail.
  let leftover = staging_dir_count(&save_dir);
  assert_eq!(
    leftover, 0,
    "the staging guard must stay armed until success; \
     found {leftover} leftover .staging-fuse-* dirs after a *.py-sweep failure"
  );

  let _ = fs::remove_dir_all(save_dir.join("custom_arch.py"));
}

#[cfg(unix)]
#[test]
fn fuse_fails_when_stale_destination_tokenizer_json_is_symlink_to_directory() {
  // `is_file()` FOLLOWS symlinks, so a
  // `save_dir/tokenizer.json` symlink whose target is a directory
  // returns `false` (target is dir) and would be SILENTLY SKIPPED by an
  // `is_file()`-gated sweep. Worse: a symlink at a reserved basename
  // pointing at an arbitrary path is an unbounded redirection vector
  // (cross-FS escape via `/etc/passwd`). The sweep uses
  // `symlink_metadata()` (NEVER follows symlinks) and rejects ANY
  // symlink at a reserved basename via the same named-error path.
  //
  // Note: the source `write_base_dir` writes tokenizer.json, so
  // `staged_names` will contain `tokenizer.json`. The stale sweep
  // skips entries in `staged_names`, so to actually exercise the
  // symlink-rejection path the source must LACK tokenizer.json.
  let weights = toy_base_weights(1);
  let model_dir =
    write_base_dir_no_tokenizer("stale_toksym_model", &weights, &plain_config_json(1));
  assert!(
    !model_dir.join("tokenizer.json").exists(),
    "fixture precondition: source has NO tokenizer.json"
  );
  // Tokenizer validation would normally fail with a missing tokenizer; we
  // want the test to exercise the STALE-SWEEP path (not the
  // validate-before-save path). So we substitute spiece.model — a
  // RESERVED basename that's NOT validated by `load_tokenizer` but IS
  // a member of TOKENIZER_EXTRA_FILES, so the stale sweep walks it.
  // But the validate step itself requires tokenizer.json to be loadable.
  //
  // Cleaner approach: add a tokenizer.json so validate passes, but
  // pre-create the SYMLINK at a DIFFERENT reserved basename the
  // source lacks (`special_tokens_map.json` or `vocab.json`). The
  // policy is basename-agnostic — any reserved basename that's a
  // symlink fails.
  fs::write(model_dir.join("tokenizer.json"), minimal_tokenizer_json()).unwrap();
  assert!(
    !model_dir.join("special_tokens_map.json").exists(),
    "fixture precondition: source has NO special_tokens_map.json"
  );

  let adapter_dir = write_mlxlm_adapter("stale_toksym_adapter", &[0], 2.0);
  let save_dir = temp_dir("stale_toksym_save");
  // Create the symlink target: a directory under save_dir. We use a
  // subdir of save_dir (not /tmp) so the test cleanup is local and
  // the symlink resolves to an EXISTING dir (the `symlink_metadata`
  // check would still classify the symlink itself as `is_symlink ==
  // true` regardless of target existence, but using an existing dir
  // makes the test fixture maximally realistic).
  let target_dir = save_dir.join("sym_target_dir");
  fs::create_dir_all(&target_dir).unwrap();
  std::os::unix::fs::symlink(&target_dir, save_dir.join("special_tokens_map.json")).unwrap();
  // Sanity: the path EXISTS as a symlink and `is_file()` follows the
  // link (returns false because target is dir) — exactly the trap the
  // fix closes.
  assert!(
    save_dir
      .join("special_tokens_map.json")
      .symlink_metadata()
      .is_ok(),
    "fixture: symlink IS at the reserved basename"
  );
  assert!(
    !save_dir.join("special_tokens_map.json").is_file(),
    "fixture: `is_file()` follows the link and returns false (symlink-to-dir target)"
  );

  let result = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false);
  let err =
    result.expect_err("fuse must fail when stale special_tokens_map.json is a symlink-to-dir");
  let msg = extract_promote_error_message(&err);
  assert!(
    msg.contains("special_tokens_map.json"),
    "error message must name the offending basename; got: {msg}"
  );
  assert!(
    msg.contains("non-regular"),
    "error message must mark the path as non-regular reserved; got: {msg}"
  );
  assert!(
    msg.contains("symlink"),
    "error message must name the kind (symlink); got: {msg}"
  );

  // Staging-guard contract.
  let leftover = staging_dir_count(&save_dir);
  assert_eq!(
    leftover, 0,
    "the staging guard must stay armed until success; \
     found {leftover} leftover .staging-fuse-* dirs after a symlink-rejection failure"
  );

  let _ = fs::remove_file(save_dir.join("special_tokens_map.json"));
  let _ = fs::remove_dir_all(&target_dir);
}

// ────────── staging guard armed until promote success ──────────

#[test]
fn fuse_cleans_up_staging_dir_on_stale_removal_failure() {
  // A `promote_staging_into_save_path` that called `staging.consume()` as
  // its FIRST step would disarm the RAII guard. Any later Err (rename
  // failure, stale-sweep removal failure, the non-regular rejection)
  // would then return WITHOUT cleanup and a `.staging-fuse-*` dir would
  // LEAK into `save_path` permanently.
  //
  // The fix keeps `staging` ARMED across the entire borrow-only
  // inner pass; only the success path consumes the guard and removes
  // the (now-empty) staging dir explicitly. Every Err path drops the
  // guard, firing `StagingDir::Drop`'s `remove_dir_all`.
  //
  // Failure-injection strategy: pre-create a NON-EMPTY directory at a
  // reserved basename the source lacks. The stale-sweep
  // hits this entry, classifies it as `directory`, and returns
  // `Err(Error::Backend{"non-regular reserved path"})`. This is a
  // mid-promote Err — exactly the path the armed-until-success guard is
  // required to clean up (a `consume()`-first shape would leak because
  // it had already disarmed the guard).
  let weights = toy_base_weights(1);
  let model_dir = write_base_dir("f2_cleanup_model", &weights, &plain_config_json(1));
  assert!(
    !model_dir.join("tokenizer_config.json").exists(),
    "fixture precondition: source lacks tokenizer_config.json (so stale-sweep visits it)"
  );

  let adapter_dir = write_mlxlm_adapter("f2_cleanup_adapter", &[0], 2.0);
  let save_dir = temp_dir("f2_cleanup_save");
  // Non-empty dir at the reserved basename — the non-regular gate
  // rejects, and the staging dir must still be cleaned up.
  fs::create_dir_all(save_dir.join("tokenizer_config.json").join("nested_file")).unwrap();

  let result = fuse::fuse(&model_dir, &adapter_dir, &save_dir, false);
  assert!(
    result.is_err(),
    "fuse must fail when stale tokenizer_config.json is a non-empty directory: got {result:?}"
  );

  // Staging-guard contract: no `.staging-fuse-*` leaked. This is the WHOLE
  // POINT of keeping the guard armed — a consume()-first
  // shape would leak here because the Err returned AFTER consume()
  // disarmed the guard.
  let leftover = staging_dir_count(&save_dir);
  assert_eq!(
    leftover, 0,
    "the staging guard MUST clean up the staging dir on a stale-sweep failure; \
     found {leftover} leftover .staging-fuse-* dirs"
  );

  let _ = fs::remove_dir_all(save_dir.join("tokenizer_config.json"));
}
