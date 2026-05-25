//! M3 WS-C PR-2 — model-load support surface (`lm::load`).
//!
//! Mirrors the existing `mlxrs/tests` style: integration tests reachable from
//! outside the crate, gated on the `lm` umbrella (which pulls the
//! `serde`/`serde_json` graph `Config` reuses and the `Tokenizer` surface
//! `load` wires). Covers `Config` serde (minimal + forward-compatible +
//! missing-required), sharded-safetensors merge keeping quantized triples,
//! the single-file fallback, and the `load(dir) -> (Config, Weights,
//! Tokenizer)` wiring against the committed tokenizer fixtures.
#![cfg(feature = "lm")]

use std::{collections::HashMap, fs, io::Write, path::PathBuf, process};

use mlxrs::{
  Array, io,
  lm::load::{self, Config},
};

/// A unique temp directory for one test (process-scoped + named so parallel
/// test binaries / cases never collide). Created fresh.
fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_lm_load_{}_{}", process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizer.json");
const TOKENIZER_CONFIG_JSON: &str = include_str!("fixtures/tokenizer_config.json");

/// A minimal, valid `config.json` body covering every required `Config`
/// field plus extra unknown keys (forward-compat) and the optional
/// `sliding_window` / `quantization` block.
const FULL_CONFIG_JSON: &str = r#"{
  "model_type": "qwen3",
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 64,
  "rope_theta": 1000000.0,
  "vocab_size": 151936,
  "tie_word_embeddings": true,
  "sliding_window": 4096,
  "quantization": { "group_size": 64, "bits": 4 },
  "max_position_embeddings": 32768,
  "some_future_key": [1, 2, 3]
}"#;

// ───────────────────────── Task 2.1: Config serde ─────────────────────────

#[test]
fn config_parses_minimal_and_ignores_extra() {
  let cfg = Config::from_json(FULL_CONFIG_JSON).unwrap();
  assert_eq!(cfg.model_type(), "qwen3");
  assert_eq!(cfg.hidden_size, 1024);
  assert_eq!(cfg.num_hidden_layers, 24);
  assert_eq!(cfg.num_attention_heads, 16);
  assert_eq!(cfg.num_key_value_heads, 8);
  assert_eq!(cfg.head_dim, 64);
  assert_eq!(cfg.rope_theta, 1_000_000.0);
  assert_eq!(cfg.vocab_size, 151936);
  assert!(cfg.tie_word_embeddings);
  assert_eq!(cfg.sliding_window, Some(4096));
  let q = cfg.quantization.expect("quantization block present");
  assert_eq!(q.group_size, 64);
  assert_eq!(q.bits, 4);
}

#[test]
fn config_optionals_default_when_absent() {
  // Same required fields, but no `sliding_window` / `quantization` keys, and
  // a still-unknown extra key → both optionals default to `None`.
  let json = r#"{
    "model_type": "llama",
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "rope_theta": 10000.0,
    "vocab_size": 32000,
    "tie_word_embeddings": false,
    "unrelated": "ignored"
  }"#;
  let cfg = Config::from_json(json).unwrap();
  assert_eq!(cfg.model_type(), "llama");
  assert!(!cfg.tie_word_embeddings);
  assert_eq!(cfg.sliding_window, None);
  assert!(cfg.quantization.is_none());
}

#[test]
fn config_missing_required_is_backend_error() {
  // `num_hidden_layers` omitted → serde fails → mapped to Error::Backend.
  let json = r#"{
    "model_type": "qwen3",
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "rope_theta": 1000000.0,
    "vocab_size": 151936,
    "tie_word_embeddings": true
  }"#;
  let err = Config::from_json(json).unwrap_err();
  assert!(
    matches!(err, mlxrs::Error::Backend { .. }),
    "expected Error::Backend, got {err:?}"
  );
}

#[test]
fn config_invalid_json_is_backend_error() {
  let err = Config::from_json("{ not json").unwrap_err();
  assert!(matches!(err, mlxrs::Error::Backend { .. }));
}

// ─────────── Task 2.2: Weights — sharded merge + quantized triples ───────────

fn small(v: &[f32], shape: (usize, usize)) -> Array {
  Array::from_slice(v, &shape).unwrap()
}

#[test]
fn weights_merges_shards_and_keeps_quant_triples() {
  let dir = temp_dir("shards");

  // Shard 1 carries a quantized triple for `a`; shard 2 carries `b.weight`.
  let mut s1 = HashMap::new();
  s1.insert("a.weight".to_string(), small(&[1.0, 2.0, 3.0, 4.0], (2, 2)));
  s1.insert("a.scales".to_string(), small(&[0.5, 0.25], (1, 2)));
  s1.insert("a.biases".to_string(), small(&[0.1, 0.2], (1, 2)));
  let mut s2 = HashMap::new();
  s2.insert("b.weight".to_string(), small(&[9.0, 8.0], (1, 2)));

  io::save_safetensors(&dir.join("model-00001-of-00002.safetensors"), &s1).unwrap();
  io::save_safetensors(&dir.join("model-00002-of-00002.safetensors"), &s2).unwrap();
  // The HF/safetensors sharded convention: an authoritative
  // `model.safetensors.index.json` lists every key's owning shard. The
  // index-honoring `load_weights` follows it — a shard not in the index
  // is invisible (the structural fix that makes the `save_model`
  // index-rename single-commit-point safe). Hand-written JSON so this
  // integration test doesn't depend on `serde_json` being a dev-dep.
  fs::write(
    dir.join("model.safetensors.index.json"),
    br#"{
  "metadata": { "total_size": 32, "total_parameters": 8 },
  "weight_map": {
    "a.weight": "model-00001-of-00002.safetensors",
    "a.scales": "model-00001-of-00002.safetensors",
    "a.biases": "model-00001-of-00002.safetensors",
    "b.weight": "model-00002-of-00002.safetensors"
  }
}"#,
  )
  .unwrap();

  let mut w = load::load_weights(&dir).unwrap();
  assert_eq!(w.len(), 4, "all four keys merged");

  // Quantized triple kept verbatim (no key remap / sanitize).
  let mut aw = w.remove("a.weight").unwrap();
  assert_eq!(aw.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
  let mut asc = w.remove("a.scales").unwrap();
  assert_eq!(asc.to_vec::<f32>().unwrap(), vec![0.5, 0.25]);
  let mut ab = w.remove("a.biases").unwrap();
  assert_eq!(ab.to_vec::<f32>().unwrap(), vec![0.1, 0.2]);
  let mut bw = w.remove("b.weight").unwrap();
  assert_eq!(bw.to_vec::<f32>().unwrap(), vec![9.0, 8.0]);
}

#[test]
fn weights_single_unsharded_safetensors() {
  let dir = temp_dir("single");
  let mut m = HashMap::new();
  m.insert(
    "tok_embeddings.weight".to_string(),
    small(&[1.0, 2.0], (1, 2)),
  );
  io::save_safetensors(&dir.join("model.safetensors"), &m).unwrap();

  let w = load::load_weights(&dir).unwrap();
  assert_eq!(w.len(), 1);
  assert!(w.contains_key("tok_embeddings.weight"));
}

#[test]
fn weights_missing_is_backend_error() {
  let dir = temp_dir("empty");
  let err = load::load_weights(&dir).unwrap_err();
  assert!(
    matches!(err, mlxrs::Error::Backend { .. }),
    "expected Error::Backend for a dir with no weights, got {err:?}"
  );
}

// ───────────────────────── Task 2.3: load() wiring ─────────────────────────

fn write_model_dir(name: &str) -> PathBuf {
  let dir = temp_dir(name);
  fs::write(dir.join("config.json"), FULL_CONFIG_JSON).unwrap();

  let mut m = HashMap::new();
  m.insert(
    "model.embed_tokens.weight".to_string(),
    small(&[1.0, 2.0, 3.0, 4.0], (2, 2)),
  );
  io::save_safetensors(&dir.join("model.safetensors"), &m).unwrap();

  let mut tj = fs::File::create(dir.join("tokenizer.json")).unwrap();
  tj.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
  let mut tc = fs::File::create(dir.join("tokenizer_config.json")).unwrap();
  tc.write_all(TOKENIZER_CONFIG_JSON.as_bytes()).unwrap();
  dir
}

#[test]
fn load_returns_config_weights_tokenizer() {
  let dir = write_model_dir("full");
  let (cfg, weights, tok) = load::load(&dir).unwrap();

  assert_eq!(cfg.model_type(), "qwen3");
  assert_eq!(cfg.num_hidden_layers, 24);
  assert_eq!(cfg.sliding_window, Some(4096));

  assert!(weights.contains_key("model.embed_tokens.weight"));

  let ids = tok.encode("hello world", false).unwrap();
  assert!(!ids.is_empty());
  assert_eq!(ids, vec![3, 4]);
}

#[test]
fn load_missing_config_is_backend_error() {
  // Weights + tokenizer present, but no config.json.
  let dir = temp_dir("no_config");
  let mut m = HashMap::new();
  m.insert("w".to_string(), small(&[1.0], (1, 1)));
  io::save_safetensors(&dir.join("model.safetensors"), &m).unwrap();
  let mut tj = fs::File::create(dir.join("tokenizer.json")).unwrap();
  tj.write_all(TOKENIZER_JSON.as_bytes()).unwrap();

  // `Tokenizer` isn't `Debug`, so the `(Config, Weights, Tokenizer)` Ok
  // variant can't go through `unwrap_err()`; match the result directly.
  match load::load(&dir) {
    Err(mlxrs::Error::Backend { .. }) => {}
    Err(other) => panic!("expected Error::Backend when config.json absent, got {other:?}"),
    Ok(_) => panic!("expected Err when config.json absent, got Ok"),
  }
}

/// Write the non-weight loadable parts: a minimal valid `config.json` plus
/// the committed tokenizer fixtures.
fn write_meta(dir: &std::path::Path) {
  fs::write(
    dir.join("config.json"),
    br#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"rope_theta":10000.0,"vocab_size":32,"tie_word_embeddings":false}"#,
  )
  .unwrap();
  fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
  fs::write(dir.join("tokenizer_config.json"), TOKENIZER_CONFIG_JSON).unwrap();
}

/// `write_meta` + a tiny plain `model.safetensors`.
fn loadable(name: &str) -> PathBuf {
  let d = temp_dir(name);
  write_meta(&d);
  let mut m = HashMap::new();
  m.insert("w".to_string(), small(&[1.0], (1, 1)));
  io::save_safetensors(&d.join("model.safetensors"), &m).unwrap();
  d
}

/// Codex #30 [high]: HF Hub snapshot dirs store `model*.safetensors` as
/// symlinks into `blobs/<hash>`. `collect_sorted` must resolve the symlink
/// (via `fs::metadata`, which follows links) and load it — not skip it as a
/// non-regular `DirEntry::file_type()`.
#[cfg(unix)]
#[test]
fn load_follows_symlinked_weights_hf_snapshot_layout() {
  let dir = temp_dir("symlink_weights");
  write_meta(&dir);
  let blobs = dir.join("blobs");
  fs::create_dir_all(&blobs).unwrap();
  let mut m = HashMap::new();
  m.insert(
    "blk.0.weight".to_string(),
    small(&[1.0, 2.0, 3.0, 4.0], (2, 2)),
  );
  // The blob is a normal safetensors file (written via the same proven path
  // pattern as every other test); `model.safetensors` is a symlink INTO it,
  // mirroring a HF Hub snapshot dir (`snapshots/<rev>/model.safetensors` ->
  // `../../blobs/<hash>`). The blob's own name is irrelevant — what matters
  // is that the globbed `model.safetensors` ENTRY is a symlink.
  io::save_safetensors(&blobs.join("blob.safetensors"), &m).unwrap();
  std::os::unix::fs::symlink(
    blobs.join("blob.safetensors"),
    dir.join("model.safetensors"),
  )
  .unwrap();

  let (_c, w, _t) = load::load(&dir)
    .expect("a HF-snapshot-style dir whose model.safetensors is a symlink must load");
  let arr = w
    .get("blk.0.weight")
    .expect("symlinked model.safetensors must be resolved & loaded, not skipped");
  assert_eq!(arr.shape(), vec![2, 2]);
}

/// Codex #30: faithful mlx-lm eos resolution. `mlx_lm.utils.load_config`
/// uses `config.json`'s `eos_token_id` as the base, a *truthy*
/// `generation_config.json` `eos_token_id` OVERWRITES it, and the result is
/// passed to `TokenizerWrapper` as the COMPLETE set — `set(eos_token_ids)`
/// REPLACES the tokenizer-config default (it is NOT unioned); absent ⇒ the
/// tokenizer's own `eos_token`. Exact-set assertions (not `contains`) so the
/// replace-not-merge contract and the base/precedence are actually pinned.
#[test]
fn load_resolves_eos_set_replace_not_merge() {
  use std::collections::BTreeSet;
  let set = |ids: &[u32]| ids.iter().copied().collect::<BTreeSet<u32>>();

  use load::EosTokenId::{Many, Single};

  // Baseline: no generation_config, and `write_meta`'s config.json has no
  // `eos_token_id` → resolved eos is `None` → the tokenizer's OWN default,
  // and the returned `Config.eos_token_id` is `None`.
  let d0 = loadable("eos_base");
  let (c0, _w, t0) = load::load(&d0).expect("baseline loads");
  let base: BTreeSet<u32> = t0.eos_token_ids_iter().collect();
  assert_eq!(c0.eos_token_id, None, "no config/gen eos ⇒ Config eos None");
  assert!(
    !base.contains(&4242)
      && !base.contains(&4243)
      && !base.contains(&4244)
      && !base.contains(&7)
      && !base.contains(&8)
      && !base.contains(&9)
      && !base.contains(&10)
      && !base.contains(&0),
    "test ids must be outside the fixture's base eos set: {base:?}"
  );
  assert!(
    !base.is_empty(),
    "fixture tokenizer must have its own eos for the replace guard"
  );
  // Codex #30 r4: assert the tokenizer eos set AND the returned
  // `Config.eos_token_id` TOGETHER for every case — Python overwrites
  // `config["eos_token_id"]` in place, so they must never disagree.

  // A config.json carrying every required field plus a chosen `eos_token_id`.
  let cfg_with_eos = |eos: &str| {
    format!(
      r#"{{"model_type":"llama","hidden_size":8,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"rope_theta":10000.0,"vocab_size":32,"tie_word_embeddings":false,"eos_token_id":{eos}}}"#
    )
  };

  // generation_config list → eos is EXACTLY {4242,4243}; base is DROPPED.
  let d1 = loadable("eos_gen_list");
  fs::write(
    d1.join("generation_config.json"),
    br#"{"eos_token_id":[4242,4243]}"#,
  )
  .unwrap();
  let (c1, _w, t1) = load::load(&d1).unwrap();
  assert_eq!(
    t1.eos_token_ids_iter().collect::<BTreeSet<u32>>(),
    set(&[4242, 4243]),
    "list REPLACES, not merges"
  );
  assert_eq!(
    c1.eos_token_id,
    Some(Many(vec![4242, 4243])),
    "Config eos overwritten (list, shape preserved)"
  );

  // generation_config scalar (truthy) → EXACTLY {4244}.
  let d2 = loadable("eos_gen_int");
  fs::write(
    d2.join("generation_config.json"),
    br#"{"eos_token_id":4244}"#,
  )
  .unwrap();
  let (c2, _w, t2) = load::load(&d2).unwrap();
  assert_eq!(
    t2.eos_token_ids_iter().collect::<BTreeSet<u32>>(),
    set(&[4244]),
    "scalar REPLACES, not merges"
  );
  assert_eq!(
    c2.eos_token_id,
    Some(Single(4244)),
    "Config eos overwritten (scalar, shape preserved)"
  );

  // generation_config list containing 0 → list is truthy regardless of
  // contents, so 0 is KEPT (the scalar-0 falsy rule is scalar-only).
  let dl0 = loadable("eos_gen_list0");
  fs::write(
    dl0.join("generation_config.json"),
    br#"{"eos_token_id":[0,4242]}"#,
  )
  .unwrap();
  let (cl0, _w, tl0) = load::load(&dl0).unwrap();
  assert_eq!(
    tl0.eos_token_ids_iter().collect::<BTreeSet<u32>>(),
    set(&[0, 4242]),
    "list [0,..] keeps 0"
  );
  assert_eq!(
    cl0.eos_token_id,
    Some(Many(vec![0, 4242])),
    "Config eos list keeps 0"
  );

  // generation_config scalar 0 is FALSY → not copied; no config.json eos →
  // falls back to the tokenizer default (EXACTLY the baseline).
  let dz = loadable("eos_gen_zero");
  fs::write(dz.join("generation_config.json"), br#"{"eos_token_id":0}"#).unwrap();
  let (cz, _w, tz) = load::load(&dz).unwrap();
  assert_eq!(
    tz.eos_token_ids_iter().collect::<BTreeSet<u32>>(),
    base,
    "falsy scalar 0 ⇒ tokenizer default"
  );
  assert_eq!(
    cz.eos_token_id, None,
    "falsy scalar 0 ⇒ Config eos untouched"
  );

  // generation_config empty list is FALSY → falls back to tokenizer default.
  let de = loadable("eos_gen_empty");
  fs::write(de.join("generation_config.json"), br#"{"eos_token_id":[]}"#).unwrap();
  let (ce, _w, te) = load::load(&de).unwrap();
  assert_eq!(
    te.eos_token_ids_iter().collect::<BTreeSet<u32>>(),
    base,
    "empty list is falsy ⇒ tokenizer default"
  );
  assert_eq!(ce.eos_token_id, None, "empty list ⇒ Config eos untouched");

  // No generation_config, config.json `eos_token_id` present → that REPLACES
  // the tokenizer default (EXACTLY {7}; scalar form).
  let dc = loadable("eos_cfg_int");
  fs::write(dc.join("config.json"), cfg_with_eos("7")).unwrap();
  let (cc, _w, tc) = load::load(&dc).unwrap();
  assert_eq!(
    tc.eos_token_ids_iter().collect::<BTreeSet<u32>>(),
    set(&[7]),
    "config.json eos REPLACES default"
  );
  assert_eq!(
    cc.eos_token_id,
    Some(Single(7)),
    "Config eos = config.json (scalar, no gen)"
  );

  // No generation_config, config.json `eos_token_id` list → EXACTLY {9,10}.
  let dcl = loadable("eos_cfg_list");
  fs::write(dcl.join("config.json"), cfg_with_eos("[9,10]")).unwrap();
  let (ccl, _w, tcl) = load::load(&dcl).unwrap();
  assert_eq!(
    tcl.eos_token_ids_iter().collect::<BTreeSet<u32>>(),
    set(&[9, 10]),
    "config.json list REPLACES default"
  );
  assert_eq!(
    ccl.eos_token_id,
    Some(Many(vec![9, 10])),
    "Config eos = config.json (list, no gen)"
  );

  // Precedence: truthy generation_config OVERRIDES config.json (both the
  // config.json eos AND the tokenizer default are dropped) → EXACTLY {8}.
  let dp = loadable("eos_precedence");
  fs::write(dp.join("config.json"), cfg_with_eos("7")).unwrap();
  fs::write(dp.join("generation_config.json"), br#"{"eos_token_id":8}"#).unwrap();
  let (cp, _w, tp) = load::load(&dp).unwrap();
  assert_eq!(
    tp.eos_token_ids_iter().collect::<BTreeSet<u32>>(),
    set(&[8]),
    "generation_config overrides config.json"
  );
  assert_eq!(
    cp.eos_token_id,
    Some(Single(8)),
    "Config eos overwritten by truthy generation_config (precedence)"
  );

  // Malformed generation_config is tolerated (mlx-lm `except
  // json.JSONDecodeError: pass`) and, with no config.json eos, falls back to
  // the tokenizer default.
  let db = loadable("eos_bad");
  fs::write(db.join("generation_config.json"), b"{ not json").unwrap();
  let (cb, _w, tb) = load::load(&db).expect("malformed generation_config is tolerated");
  assert_eq!(
    tb.eos_token_ids_iter().collect::<BTreeSet<u32>>(),
    base,
    "malformed ⇒ tokenizer default"
  );
  assert_eq!(cb.eos_token_id, None, "malformed ⇒ Config eos untouched");
}
