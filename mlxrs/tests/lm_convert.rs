//! `convert()` library driver (`lm::convert`).
//!
//! Integration tests for the port of `mlx_lm/convert.py::convert` (the
//! conversion driver that ties load + quantize/dequantize + save together).
//! Mirrors `lm_load.rs` style: gated on the `lm` umbrella, fixture-built
//! synthetic HF directories under `temp_dir()`, hand-traced assertions.
//!
//! Test list:
//! - `convert_pass_through_no_quantize_no_dequantize`
//! - `convert_quantize_int4_group64`
//! - `convert_dequantize_round_trip`
//! - `convert_mixed_predicate_skips_embedding_and_lm_head`
//! - `convert_rejects_upload_repo`
//! - `convert_rejects_revision`
//! - `convert_copies_tokenizer_files`
//! - `convert_rename_in_place_is_handled`
//! - `mixed_quant_predicate_default_heuristics` — table test
//!
//! No `peak_memory()` magnitude asserts.
#![cfg(feature = "lm")]

use std::{collections::HashMap, fs, io::Write, path::PathBuf, process};

use mlxrs::{
  Array, Dtype, io,
  lm::{
    convert::{self, ConvertArgs, MixedQuantPredicate, MixedQuantRecipe, mixed_quant_predicate},
    load,
    quant::{QuantMode, Quantization},
  },
};

// ────────────────────────── test scaffolding ──────────────────────────

const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizer.json");
const TOKENIZER_CONFIG_JSON: &str = include_str!("fixtures/tokenizer_config.json");

/// Fresh per-test temp directory, the no-`tempfile`-crate convention used by
/// the rest of the suite (process-scoped + named so parallel cases never
/// collide; collision-resistant counter for multiple dirs per test).
fn temp_dir(name: &str) -> PathBuf {
  use std::sync::atomic::{AtomicU64, Ordering};
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!("mlxrs_lm_convert_{}_{name}_{n}", process::id()));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

fn f32_array(v: &[f32], shape: (usize, usize)) -> Array {
  Array::from_slice(v, &shape).unwrap()
}

/// A minimal valid `Config` JSON with no quantization block. The model_type
/// / hidden_size / etc are immaterial — `convert` does not consult them.
const PLAIN_CONFIG_JSON: &str = r#"{
  "model_type": "qwen3",
  "hidden_size": 16,
  "num_hidden_layers": 1,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 8,
  "rope_theta": 10000.0,
  "vocab_size": 128,
  "tie_word_embeddings": false
}"#;

/// Build a synthetic HF-style source directory with `config.json`,
/// `tokenizer.json`, `tokenizer_config.json`, and one safetensors file
/// holding `weights`. Returns the directory path.
fn write_src_dir(name: &str, weights: &HashMap<String, Array>, config_json: &str) -> PathBuf {
  let dir = temp_dir(name);
  fs::write(dir.join("config.json"), config_json).unwrap();
  io::save_safetensors(&dir.join("model.safetensors"), weights).unwrap();
  let mut tj = fs::File::create(dir.join("tokenizer.json")).unwrap();
  tj.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
  let mut tc = fs::File::create(dir.join("tokenizer_config.json")).unwrap();
  tc.write_all(TOKENIZER_CONFIG_JSON.as_bytes()).unwrap();
  dir
}

/// Helper: the smallest valid float weight set (rank-2, last-axis multiple
/// of 64 so the `int4 + group_size=64` test can actually quantize it).
fn make_quantizable_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();
  // 2×64 = 128 f32 values; last-axis (64) is divisible by group_size=64.
  let big = (0..128).map(|i| (i as f32) * 0.01).collect::<Vec<_>>();
  w.insert("layer.weight".to_string(), f32_array(&big, (2, 64)));
  w
}

// ──────────────────────── pass-through (no quant) ────────────────────────

#[test]
fn convert_pass_through_no_quantize_no_dequantize() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("pass_through_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("pass_through_dst");
  // dst is created by temp_dir; convert must reject pre-existing dst per
  // the python reference (`convert.py:105-109`). Remove it so convert can
  // create it itself.
  fs::remove_dir_all(&dst).unwrap();

  convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst.clone(),
    ..Default::default()
  })
  .unwrap();

  // The output dir has config.json + a single shard + index.
  assert!(dst.join("config.json").is_file(), "config.json written");
  // The save layer always writes the index (even for a single shard).
  assert!(
    dst.join("model.safetensors.index.json").is_file(),
    "index.json written"
  );

  // Reload the weights through `load_weights` (which follows the index)
  // and check they survived the pass-through round-trip byte-equal.
  let mut reloaded = load::load_weights(&dst).unwrap();
  assert_eq!(reloaded.len(), 1);
  let mut got = reloaded.remove("layer.weight").unwrap();
  let got_vals = got.to_vec::<f32>().unwrap();
  let expected = (0..128).map(|i| (i as f32) * 0.01).collect::<Vec<_>>();
  assert_eq!(got_vals, expected, "pass-through weights byte-equal");
}

// ──────────────────────── quantize int4/group64 ────────────────────────

#[test]
fn convert_quantize_int4_group64() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("quant_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("quant_dst");
  fs::remove_dir_all(&dst).unwrap();

  convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst.clone(),
    quantize: true,
    q_bits: Some(4),
    q_group_size: Some(64),
    q_mode: QuantMode::Affine,
    ..Default::default()
  })
  .unwrap();

  // Output has the quantized triple (`.weight`, `.scales`, `.biases` for
  // affine mode).
  let mut reloaded = load::load_weights(&dst).unwrap();
  assert!(
    reloaded.contains_key("layer.scales"),
    "scales emitted (affine)"
  );
  assert!(
    reloaded.contains_key("layer.biases"),
    "biases emitted (affine)"
  );
  // The packed `.weight` is uint32 (mlx-quantized layout). Remove from
  // the map so we own the `Array` mutably (it's `!Clone`).
  let packed = reloaded
    .remove("layer.weight")
    .expect("quantized weight present");
  assert_eq!(packed.dtype().unwrap(), Dtype::U32);
}

// ──────────────────────── dequantize round-trip ────────────────────────

#[test]
fn convert_dequantize_round_trip() {
  // Step 1: produce a quantized source dir by quantizing a known dense
  // input through the same `convert` driver. (Using the convert→reload
  // path keeps the test invariant to the saved-shard layout.)
  let weights = make_quantizable_weights();
  let src = write_src_dir("dequant_src_dense", &weights, PLAIN_CONFIG_JSON);
  let quant_dir = temp_dir("dequant_src_quantized");
  fs::remove_dir_all(&quant_dir).unwrap();
  convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: quant_dir.clone(),
    quantize: true,
    q_bits: Some(4),
    q_group_size: Some(64),
    q_mode: QuantMode::Affine,
    ..Default::default()
  })
  .unwrap();
  // Now `quant_dir/config.json` carries a `quantization_config` block (per
  // `save_config`'s `quantization → quantization_config` mirror). For
  // `dequantize_weights` to find the per-layer config, we patch the
  // `quantization` block back into the on-disk config — that's what mlx-lm
  // does in convert's own dequantize path (`config.pop("quantization_config")`
  // and `dequantize_model` reads from module attrs). In our port `convert`'s
  // dequantize branch reads `quantization` off the source config (the
  // `quant_dir`'s saved config carries it under `quantization` too, since
  // `save_config` mirrors `quantization`→`quantization_config` but
  // PRESERVES the original `quantization` key).

  // Step 2: convert with dequantize=true.
  let dst = temp_dir("dequant_dst");
  fs::remove_dir_all(&dst).unwrap();
  convert::convert(ConvertArgs {
    hf_path: quant_dir,
    mlx_path: dst.clone(),
    dequantize: true,
    ..Default::default()
  })
  .unwrap();

  // The output is dense: no `.scales` / `.biases`, only a `.weight` whose
  // dtype is float (not uint32 — that would be the still-packed case).
  let mut reloaded = load::load_weights(&dst).unwrap();
  assert!(
    !reloaded.contains_key("layer.scales"),
    "scales removed by dequantize"
  );
  assert!(
    !reloaded.contains_key("layer.biases"),
    "biases removed by dequantize"
  );
  let w = reloaded
    .remove("layer.weight")
    .expect("dequantized .weight present");
  let dt = w.dtype().unwrap();
  assert!(
    matches!(dt, Dtype::F32 | Dtype::F16 | Dtype::BF16),
    "dequantized weight is float, got {dt:?}"
  );
}

// ──────────────────────── mixed predicate skips embed / lm_head ────────────────────────

#[test]
fn convert_mixed_predicate_skips_embedding_and_lm_head() {
  // Build a synthetic model: embed + lm_head (both should be skipped) +
  // a linear layer (should be quantized). All last-axis = 64 so they pass
  // the structural shape gate.
  let mut weights = HashMap::new();
  let blob = (0..128).map(|i| (i as f32) * 0.01).collect::<Vec<_>>();
  weights.insert(
    "model.embed_tokens.weight".to_string(),
    f32_array(&blob, (2, 64)),
  );
  weights.insert("lm_head.weight".to_string(), f32_array(&blob, (2, 64)));
  weights.insert(
    "model.layers.0.self_attn.q_proj.weight".to_string(),
    f32_array(&blob, (2, 64)),
  );
  let src = write_src_dir("mixed_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("mixed_dst");
  fs::remove_dir_all(&dst).unwrap();

  /// A test-only predicate that mirrors the standard mlx-lm convention of
  /// skipping the embedding and lm_head layers.
  struct SkipEmbedAndHead;
  impl MixedQuantPredicate for SkipEmbedAndHead {
    fn decide(&self, layer_name: &str, _weight: &Array) -> Option<Quantization> {
      if layer_name == "model.embed_tokens" || layer_name == "lm_head" {
        None // skip
      } else {
        Some(Quantization::affine(64, 4))
      }
    }
  }

  convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst.clone(),
    quantize: true,
    q_bits: Some(4),
    q_group_size: Some(64),
    q_mode: QuantMode::Affine,
    quant_predicate: Some(Box::new(SkipEmbedAndHead)),
    ..Default::default()
  })
  .unwrap();

  let reloaded = load::load_weights(&dst).unwrap();
  // Embedding + lm_head are still dense (no `.scales` sibling).
  assert!(
    !reloaded.contains_key("model.embed_tokens.scales"),
    "embedding not quantized"
  );
  assert!(
    !reloaded.contains_key("lm_head.scales"),
    "lm_head not quantized"
  );
  // q_proj got quantized (has `.scales` sibling).
  assert!(
    reloaded.contains_key("model.layers.0.self_attn.q_proj.scales"),
    "q_proj quantized — has .scales"
  );
}

// ──────────────────────── reject hub upload ────────────────────────

#[test]
fn convert_rejects_upload_repo() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("reject_upload_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("reject_upload_dst");
  fs::remove_dir_all(&dst).unwrap();

  let err = convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst,
    upload_repo: Some("user/repo".to_string()),
    ..Default::default()
  })
  .unwrap_err();
  match err {
    mlxrs::Error::InvariantViolation(p) => {
      assert!(
        p.context().contains("upload_repo"),
        "context names the rejected field: {}",
        p.context()
      );
    }
    other => panic!("expected Error::InvariantViolation, got {other:?}"),
  }
}

// ──────────────────────── reject revision ────────────────────────

#[test]
fn convert_rejects_revision() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("reject_rev_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("reject_rev_dst");
  fs::remove_dir_all(&dst).unwrap();

  let err = convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst,
    revision: Some("main".to_string()),
    ..Default::default()
  })
  .unwrap_err();
  match err {
    mlxrs::Error::InvariantViolation(p) => {
      assert!(
        p.context().contains("revision"),
        "context names the rejected field: {}",
        p.context()
      );
    }
    other => panic!("expected Error::InvariantViolation, got {other:?}"),
  }
}

// ──────────────────────── reject quantize + dequantize ────────────────────────

#[test]
fn convert_rejects_quantize_and_dequantize() {
  // mlx-lm's `convert.py:146-147` — `Choose either quantize or
  // dequantize, not both.`
  let weights = make_quantizable_weights();
  let src = write_src_dir("reject_both_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("reject_both_dst");
  fs::remove_dir_all(&dst).unwrap();

  let err = convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst,
    quantize: true,
    dequantize: true,
    ..Default::default()
  })
  .unwrap_err();
  match err {
    mlxrs::Error::InvariantViolation(p) => {
      let ctx = p.context().to_lowercase();
      assert!(
        ctx.contains("quantize") && ctx.contains("dequantize"),
        "context mentions both flags: {}",
        p.context()
      );
    }
    other => panic!("expected Error::InvariantViolation, got {other:?}"),
  }
}

// ──────────────────────── reject existing destination ────────────────────────

#[test]
fn convert_rejects_existing_destination() {
  // mlx-lm's `convert.py:105-109` — `Cannot save to the path {mlx_path}
  // as it already exists.` This is the FIRST check in the driver, before
  // load.
  let weights = make_quantizable_weights();
  let src = write_src_dir("reject_existing_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("reject_existing_dst");
  // Leave dst in place (temp_dir created it); convert should refuse.

  let err = convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst,
    ..Default::default()
  })
  .unwrap_err();
  match err {
    mlxrs::Error::FileIo(p) => {
      assert_eq!(p.op(), mlxrs::error::FileOp::Stat);
      assert_eq!(p.inner().kind(), std::io::ErrorKind::AlreadyExists);
      assert!(
        p.context().contains("destination must not already exist"),
        "context names the rejected destination: {}",
        p.context()
      );
    }
    other => panic!("expected Error::FileIo (AlreadyExists), got {other:?}"),
  }
}

// ──────────────────────── copies tokenizer + extras ────────────────────────

#[test]
fn convert_copies_tokenizer_files() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("copy_tok_src", &weights, PLAIN_CONFIG_JSON);
  // Plant the extra tokenizer-family files the python reference's
  // `save_pretrained` would emit (special_tokens_map.json,
  // chat_template.jinja) plus a `generation_config.json` (the explicit
  // copy in `utils.save`).
  fs::write(
    src.join("special_tokens_map.json"),
    br#"{"eos_token":"</s>"}"#,
  )
  .unwrap();
  fs::write(src.join("chat_template.jinja"), b"{{ messages }}").unwrap();
  fs::write(src.join("generation_config.json"), br#"{"max_length": 32}"#).unwrap();
  // A `.py` extra (HF model code some loaders need — `utils.save:946`).
  fs::write(src.join("model_code.py"), b"# pretend impl").unwrap();

  let dst = temp_dir("copy_tok_dst");
  fs::remove_dir_all(&dst).unwrap();

  convert::convert(ConvertArgs {
    hf_path: src.clone(),
    mlx_path: dst.clone(),
    ..Default::default()
  })
  .unwrap();

  // Every tokenizer / template / python-helper file present at dst,
  // byte-equal to the source.
  for name in [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "model_code.py",
  ] {
    let a = fs::read(src.join(name)).unwrap();
    let b = fs::read(dst.join(name)).unwrap();
    assert_eq!(a, b, "{name} byte-equal at dst");
  }
}

// ──────────────────────── rename-in-place (src == dst) ────────────────────────

#[test]
fn convert_rename_in_place_is_handled() {
  // mlx-lm itself rejects `mlx_path.exists()` (convert.py:105-109), so the
  // rename-in-place case requires either bypassing that check or having
  // the destination pre-exist == source. Since `convert` MUST reject
  // existing destinations to match the reference, this test asserts that
  // contract: rename-in-place returns the same `already exists` error.
  // (The `copy_tokenizer_and_extras` helper independently no-ops when
  // src == dst — separately exercised below.)
  let weights = make_quantizable_weights();
  let src = write_src_dir("rename_src", &weights, PLAIN_CONFIG_JSON);
  let err = convert::convert(ConvertArgs {
    hf_path: src.clone(),
    mlx_path: src.clone(),
    ..Default::default()
  })
  .unwrap_err();
  match err {
    mlxrs::Error::FileIo(p) => {
      assert_eq!(p.op(), mlxrs::error::FileOp::Stat);
      assert_eq!(p.inner().kind(), std::io::ErrorKind::AlreadyExists);
      assert!(
        p.context().contains("destination must not already exist"),
        "context names the rejected destination: {}",
        p.context()
      );
    }
    other => panic!("expected Error::FileIo (AlreadyExists), got {other:?}"),
  }

  // The helper itself is independently a no-op when src == dst (rename-in-
  // place is what the save path + this helper combine to handle for the
  // on-disk-only partial-convert scenarios). Calling the helper with the same path
  // both ways must complete successfully without overwriting/clobbering.
  convert::copy_tokenizer_and_extras(&src, &src).unwrap();
  // Source files still present and unchanged.
  assert!(src.join("tokenizer.json").is_file());
  assert_eq!(
    fs::read_to_string(src.join("tokenizer.json")).unwrap(),
    TOKENIZER_JSON
  );
}

// ──────────────────────── builder heuristics — table test ────────────────────────

#[test]
fn mixed_quant_predicate_default_heuristics() {
  // Hand-trace the mlx-lm `mixed_quant_predicate_builder` heuristic
  // (convert.py:48-77) against a small synthetic model with the standard
  // `model.layers.{idx}.{...}` path shape.
  //
  // Recipe `mixed_3_6` → low=3, high=6 (group_size=64, mode=affine).
  // For a 16-layer model, `use_more_bits` is true when:
  //   - idx < num_layers // 8  (i.e. idx < 2 → indices 0, 1), OR
  //   - idx >= 7 * num_layers // 8  (i.e. idx >= 14 → indices 14, 15), OR
  //   - (idx - num_layers // 8) % 3 == 2  (i.e. (idx - 2) % 3 == 2 →
  //     idx ∈ {4, 7, 10, 13}).
  //
  // Build a synthetic Weights map carrying a `down_proj` at one of the
  // typical paths so the builder's path-introspection can find the
  // `layer_location` index (it scans for the first numeric segment of
  // the first `down_proj` key — `convert.py:42-45`).
  let mut weights: HashMap<String, Array> = HashMap::new();
  for i in 0..16 {
    weights.insert(
      format!("model.layers.{i}.mlp.down_proj.weight"),
      f32_array(&[0.0; 64], (1, 64)),
    );
    weights.insert(
      format!("model.layers.{i}.self_attn.q_proj.weight"),
      f32_array(&[0.0; 64], (1, 64)),
    );
    weights.insert(
      format!("model.layers.{i}.self_attn.v_proj.weight"),
      f32_array(&[0.0; 64], (1, 64)),
    );
  }
  weights.insert("lm_head.weight".to_string(), f32_array(&[0.0; 64], (1, 64)));

  let pred = mixed_quant_predicate(MixedQuantRecipe::Mixed3_6, &weights, 64).unwrap();
  let probe = f32_array(&[0.0; 64], (1, 64));

  // lm_head ALWAYS gets high_bits regardless of layer index
  // (convert.py:72-73).
  let q = pred.decide("lm_head", &probe).unwrap();
  assert_eq!(q.bits, 6, "lm_head gets high_bits=6");
  assert_eq!(q.group_size, 64);
  assert_eq!(q.mode, QuantMode::Affine);

  // Indices in the use_more_bits set (idx=4, v_proj → high_bits=6).
  let q = pred
    .decide("model.layers.4.self_attn.v_proj", &probe)
    .unwrap();
  assert_eq!(q.bits, 6, "v_proj at idx=4 is in use_more_bits set");

  // down_proj at idx=4 is also in use_more_bits → high_bits=6.
  let q = pred.decide("model.layers.4.mlp.down_proj", &probe).unwrap();
  assert_eq!(q.bits, 6, "down_proj at idx=4 gets high_bits");

  // Outside use_more_bits set: idx=3 (3 >= 2, 3 < 14, (3-2)%3=1, not 2).
  let q = pred
    .decide("model.layers.3.self_attn.v_proj", &probe)
    .unwrap();
  assert_eq!(q.bits, 3, "v_proj at idx=3 NOT in use_more_bits → low_bits");

  // Plain q_proj never gets high_bits (the use_more_bits high-bits arms
  // only fire for v_proj / down_proj / lm_head).
  let q = pred
    .decide("model.layers.4.self_attn.q_proj", &probe)
    .unwrap();
  assert_eq!(q.bits, 3, "q_proj is never high_bits → low_bits=3");

  // idx=0: in the < num_layers//8 set → use_more_bits=true. v_proj/0 →
  // high; q_proj/0 → low.
  let q = pred
    .decide("model.layers.0.self_attn.v_proj", &probe)
    .unwrap();
  assert_eq!(q.bits, 6, "v_proj at idx=0 in use_more_bits");
  let q = pred
    .decide("model.layers.0.self_attn.q_proj", &probe)
    .unwrap();
  assert_eq!(q.bits, 3, "q_proj at idx=0 NEVER high_bits");

  // idx=15 (>= 7*16//8 = 14) → use_more_bits=true; down_proj → high.
  let q = pred
    .decide("model.layers.15.mlp.down_proj", &probe)
    .unwrap();
  assert_eq!(q.bits, 6, "down_proj at idx=15 in trailing use_more_bits");
}

#[test]
fn mixed_quant_predicate_recipes_table() {
  // Hand-trace the recipe → (low_bits, high_bits) table from
  // `convert.py:26-36`:
  //   mixed_2_6 → (2, 6)
  //   mixed_3_4 → (3, 4)
  //   mixed_3_6 → (3, 6)
  //   mixed_4_6 → (4, 6)
  let mut weights: HashMap<String, Array> = HashMap::new();
  for i in 0..16 {
    weights.insert(
      format!("model.layers.{i}.mlp.down_proj.weight"),
      f32_array(&[0.0; 64], (1, 64)),
    );
  }

  let probe = f32_array(&[0.0; 64], (1, 64));
  // idx=3 is OUT of use_more_bits (low bits); idx=4 is IN (high bits).
  for (recipe, low, high) in [
    (MixedQuantRecipe::Mixed2_6, 2, 6),
    (MixedQuantRecipe::Mixed3_4, 3, 4),
    (MixedQuantRecipe::Mixed3_6, 3, 6),
    (MixedQuantRecipe::Mixed4_6, 4, 6),
  ] {
    let pred = mixed_quant_predicate(recipe, &weights, 64).unwrap();
    let q_low = pred.decide("model.layers.3.mlp.gate_proj", &probe).unwrap();
    assert_eq!(q_low.bits, low, "recipe {recipe:?}: idx=3 → low_bits={low}");
    let q_high = pred.decide("model.layers.4.mlp.down_proj", &probe).unwrap();
    assert_eq!(
      q_high.bits, high,
      "recipe {recipe:?}: idx=4 down_proj → high_bits={high}"
    );
  }
}

#[test]
fn mixed_quant_predicate_no_down_proj_is_error() {
  // `convert.py:39-40` — `if len(down_keys) == 0: raise ValueError(...)`.
  let weights: HashMap<String, Array> = HashMap::new();
  // `DefaultMixedQuantPredicate` doesn't implement `Debug` (it has no
  // need to — it's a trait-object carrier), so go through `match` rather
  // than `unwrap_err`.
  match mixed_quant_predicate(MixedQuantRecipe::Mixed3_6, &weights, 64) {
    Ok(_) => panic!("expected Err for empty weights"),
    Err(mlxrs::Error::EmptyInput(p)) => {
      assert!(
        p.context().to_lowercase().contains("down_proj"),
        "context mentions missing down_proj keys: {}",
        p.context()
      );
    }
    Err(other) => panic!("expected Error::EmptyInput, got {other:?}"),
  }
}

// ──────────────────────── explicit-dtype gate ────────────────────────

/// An explicit `Some(Dtype::I32)` MUST be a hard
/// error from `convert`, not silently cast every floating weight to
/// int. The reference's string-typed `if dtype in MODEL_CONVERSION_DTYPES`
/// gate (`convert.py:133`) silently drops unknown spellings; the typed
/// `Dtype` enum would otherwise silently accept `I32` and forward it to
/// `cast_floats_to_dtype`, wrecking every weight. End-to-end through
/// `convert`.
#[test]
fn convert_rejects_explicit_dtype_i32() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("reject_dtype_i32_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("reject_dtype_i32_dst");
  fs::remove_dir_all(&dst).unwrap();

  let err = convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst,
    dtype: Some(Dtype::I32),
    ..Default::default()
  })
  .unwrap_err();
  match err {
    mlxrs::Error::UnsupportedDtype(p) => {
      assert_eq!(p.dtype(), Dtype::I32);
      assert_eq!(p.supported(), &[Dtype::F16, Dtype::BF16, Dtype::F32]);
      assert!(
        p.context().to_lowercase().contains("dtype"),
        "context names the dtype field: {}",
        p.context()
      );
    }
    other => panic!("expected Error::UnsupportedDtype, got {other:?}"),
  }
}

#[test]
fn convert_rejects_explicit_dtype_f64() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("reject_dtype_f64_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("reject_dtype_f64_dst");
  fs::remove_dir_all(&dst).unwrap();

  let err = convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst,
    dtype: Some(Dtype::F64),
    ..Default::default()
  })
  .unwrap_err();
  match err {
    mlxrs::Error::UnsupportedDtype(p) => {
      assert_eq!(p.dtype(), Dtype::F64);
      assert_eq!(p.supported(), &[Dtype::F16, Dtype::BF16, Dtype::F32]);
    }
    other => panic!("expected Error::UnsupportedDtype, got {other:?}"),
  }
}

#[test]
fn convert_rejects_explicit_dtype_complex() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("reject_dtype_complex_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("reject_dtype_complex_dst");
  fs::remove_dir_all(&dst).unwrap();

  let err = convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst,
    dtype: Some(Dtype::Complex64),
    ..Default::default()
  })
  .unwrap_err();
  match err {
    mlxrs::Error::UnsupportedDtype(p) => {
      assert_eq!(p.dtype(), Dtype::Complex64);
      assert_eq!(p.supported(), &[Dtype::F16, Dtype::BF16, Dtype::F32]);
    }
    other => panic!("expected Error::UnsupportedDtype, got {other:?}"),
  }
}

// ──────────────────────── q_group_size / q_bits zero falsy ────────────────────────

/// `q_group_size: Some(0)` is python-falsy
/// (`utils.py:808`: `group_size or default_group_size`) and MUST fall
/// back to the per-mode default. The previous `unwrap_or` shape would
/// have written a `quantization.group_size = 0` block to disk against
/// dense weights (because `quantize_weights`' `last % 0 != 0` arm skips
/// every layer) — convert would have silently produced a broken
/// checkpoint. This test asserts the destination ACTUALLY quantizes
/// (`.scales` is emitted) when `Some(0)` is passed.
#[test]
fn convert_q_group_size_zero_falls_back_to_default() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("q_group_size_zero_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("q_group_size_zero_dst");
  fs::remove_dir_all(&dst).unwrap();

  // q_group_size = Some(0) — python-falsy; convert MUST fall back to
  // the affine default (64). The weights are sized for 64-element
  // groups (last axis = 64) so the quantize step succeeds.
  convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst.clone(),
    quantize: true,
    q_bits: Some(4),
    q_group_size: Some(0),
    q_mode: QuantMode::Affine,
    ..Default::default()
  })
  .unwrap();

  let reloaded = load::load_weights(&dst).unwrap();
  // If `Some(0)` had survived, `quantize_weights`'s shape gate would
  // have skipped every layer → no `.scales` emitted → the assertion
  // below would fire.
  assert!(
    reloaded.contains_key("layer.scales"),
    "Some(0) group_size fell back to default 64 → quantization actually ran"
  );
  assert!(
    reloaded.contains_key("layer.biases"),
    "Some(0) group_size fell back to default 64 → biases emitted"
  );

  // The saved config carries group_size = 64 (NOT 0).
  let cfg_text = fs::read_to_string(dst.join("config.json")).unwrap();
  let cfg: serde_json::Value = serde_json::from_str(&cfg_text).unwrap();
  let q = cfg.get("quantization").unwrap();
  assert_eq!(
    q.get("group_size").and_then(|v| v.as_i64()),
    Some(64),
    "config.quantization.group_size is the fallback default, not 0"
  );
}

#[test]
fn convert_q_bits_zero_falls_back_to_default() {
  let weights = make_quantizable_weights();
  let src = write_src_dir("q_bits_zero_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("q_bits_zero_dst");
  fs::remove_dir_all(&dst).unwrap();

  // q_bits = Some(0) — python-falsy; convert MUST fall back to the
  // affine default (4).
  convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst.clone(),
    quantize: true,
    q_bits: Some(0),
    q_group_size: Some(64),
    q_mode: QuantMode::Affine,
    ..Default::default()
  })
  .unwrap();

  let reloaded = load::load_weights(&dst).unwrap();
  assert!(
    reloaded.contains_key("layer.scales"),
    "Some(0) bits fell back to default 4 → quantization ran"
  );
  // The saved config carries bits = 4 (NOT 0).
  let cfg_text = fs::read_to_string(dst.join("config.json")).unwrap();
  let cfg: serde_json::Value = serde_json::from_str(&cfg_text).unwrap();
  let q = cfg.get("quantization").unwrap();
  assert_eq!(
    q.get("bits").and_then(|v| v.as_i64()),
    Some(4),
    "config.quantization.bits is the fallback default, not 0"
  );
}

// ──────────────────────── predicate called once per layer ────────────────────────

/// The user-supplied `MixedQuantPredicate` MUST be
/// called exactly ONCE per structurally-eligible layer across the full
/// convert pipeline. The reference's `nn.quantize` invokes
/// `wrapped_predicate` exactly once per module (`utils.py:837-843`),
/// and its single return value flows BOTH into `quantized_config`
/// (`utils.py:831-834`) AND back to `nn.quantize`'s decision — a
/// stateful predicate is therefore consistent across both views.
///
/// The previous mlxrs shape evaluated the predicate twice (once in
/// `build_quantize_config`, again in the `eligible` closure of
/// `convert`), so a stateful or non-deterministic predicate could write
/// one decision into the saved config and apply a different one to the
/// weights. This test asserts max-count <= 1 per path across the full
/// convert pipeline.
#[test]
fn convert_stateful_predicate_called_exactly_once_per_eligible_layer() {
  use std::{cell::RefCell, rc::Rc};

  // Predicate that records every call site (path → count).
  // `Rc` not `Arc`: `MixedQuantPredicate` is `!Send` / `!Sync` (per
  // the trait's doc-comment), so single-threaded ref-counting is the
  // right shape (clippy's `arc_with_non_send_sync` would reject `Arc`).
  struct CountingPredicate {
    counts: Rc<RefCell<HashMap<String, u32>>>,
  }
  impl MixedQuantPredicate for CountingPredicate {
    fn decide(&self, layer_name: &str, _weight: &Array) -> Option<Quantization> {
      *self
        .counts
        .borrow_mut()
        .entry(layer_name.to_string())
        .or_insert(0) += 1;
      // Returning `Some(_)` is more strenuous than `None` — it forces
      // BOTH the config-builder write arm and the `eligible` closure
      // accept arm to fire (i.e. both downstream consumers that
      // previously re-invoked the predicate).
      Some(Quantization::affine(64, 4))
    }
  }

  // Build a synthetic model with multiple structurally-eligible
  // layers (rank 2, last axis = 64). All paths should be quantized.
  let mut weights: HashMap<String, Array> = HashMap::new();
  let blob: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
  let eligible_paths = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.mlp.down_proj",
  ];
  for path in eligible_paths {
    weights.insert(format!("{path}.weight"), f32_array(&blob, (2, 64)));
  }
  let src = write_src_dir("predicate_once_src", &weights, PLAIN_CONFIG_JSON);
  let dst = temp_dir("predicate_once_dst");
  fs::remove_dir_all(&dst).unwrap();

  let counts = Rc::new(RefCell::new(HashMap::<String, u32>::new()));
  let pred = CountingPredicate {
    counts: Rc::clone(&counts),
  };

  convert::convert(ConvertArgs {
    hf_path: src,
    mlx_path: dst.clone(),
    quantize: true,
    q_bits: Some(4),
    q_group_size: Some(64),
    q_mode: QuantMode::Affine,
    quant_predicate: Some(Box::new(pred)),
    ..Default::default()
  })
  .unwrap();

  // Read back the per-path counts. The key assertion: max count across
  // all eligible layers is <= 1.
  let counts_ro = counts.borrow();
  assert!(
    !counts_ro.is_empty(),
    "the predicate was invoked at all (sanity)"
  );
  let max = counts_ro.values().copied().max().unwrap_or(0);
  assert!(
    max <= 1,
    "predicate must be invoked at most once per eligible layer; max count was {max}; counts={:?}",
    *counts_ro,
  );
  // And each eligible path was visited exactly once (== 1, not 0).
  for path in eligible_paths {
    assert_eq!(
      counts_ro.get(path).copied(),
      Some(1),
      "{path} visited exactly once"
    );
  }

  // Sanity: the weights actually quantized (so we know we drove the
  // "predicate's decision flows to the quantizer" path, not just the
  // "predicate skipped everything" no-op).
  let reloaded = load::load_weights(&dst).unwrap();
  for path in eligible_paths {
    assert!(
      reloaded.contains_key(&format!("{path}.scales")),
      "{path} got quantized"
    );
  }
}
