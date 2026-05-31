use super::*;
use std::fs;

/// A unique temp directory for one test (process-scoped + named so
/// parallel test binaries / cases never collide).
fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_audio_load_{}_{}", std::process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

/// `get_model_path("/abs/path/that/exists")` returns the canonical
/// local PathBuf — mirror of mlx-audio's
/// `if model_path.exists(): return model_path` early return.
#[test]
fn get_model_path_resolves_local_path() {
  let dir = temp_dir("resolves_local");
  let s = dir.to_string_lossy().into_owned();
  let resolved = get_model_path(&s).expect("local existing path resolves");
  assert_eq!(resolved, dir);
}

/// A repo-id-shaped input ("org/name") that does NOT exist locally is
/// REJECTED with a clear no-Hub message — the no-network policy.
#[test]
fn get_model_path_rejects_hf_hub_path() {
  let err = get_model_path("mlx-community/silero-vad")
    .expect_err("non-local repo id must be rejected, not silently fetched");
  let msg = err.to_string();
  assert!(
    msg.contains("local on-disk directory"),
    "error should explain the no-network policy, got: {msg}"
  );
  assert!(
    msg.contains("huggingface-cli download"),
    "error should point at the out-of-process workaround, got: {msg}"
  );
}

/// A local-shaped path that does NOT exist surfaces a clear "not
/// found" error rather than being treated as a Hub id.
#[test]
fn get_model_path_local_missing_is_clear_error() {
  let err = get_model_path("/definitely/does/not/exist/mlxrs-a9-missing")
    .expect_err("missing local path must error, not fetch");
  let msg = err.to_string();
  assert!(
    msg.contains("local path not found"),
    "error should name the local-not-found case, got: {msg}"
  );
}

/// `load_config` reads a small synthetic `config.json` and returns the
/// verbatim body.
#[test]
fn load_config_reads_small_json() {
  let dir = temp_dir("load_config_small");
  let body = r#"{ "model_type": "silero_vad", "hidden_size": 128 }"#;
  fs::write(dir.join("config.json"), body).unwrap();
  let text = load_config(&dir).expect("config.json reads");
  assert_eq!(text, body);
}

/// A missing `config.json` is a recoverable Backend error naming the
/// offending path.
#[test]
fn load_config_missing_is_clear_error() {
  let dir = temp_dir("load_config_missing");
  let err = load_config(&dir).expect_err("missing config.json must error");
  let msg = err.to_string();
  assert!(
    msg.contains("audio model config not found"),
    "error should name the missing-config case, got: {msg}"
  );
}

/// A `config.json` without a `quantization` block is the dense-model
/// path: `apply_quantization` returns `Ok(None)`, matching mlx-audio's
/// `if quantization is None: return` early return.
#[test]
fn apply_quantization_passes_through_unquantized_model() {
  let body = r#"{ "model_type": "silero_vad", "hidden_size": 128 }"#;
  let q = apply_quantization(body).expect("dense config parses");
  assert!(
    q.is_none(),
    "no quantization block → Ok(None), got Some(_) — broke the dense-model path"
  );
}

/// A `config.json` with a global `quantization` block parses into a
/// [`PerLayerQuantization`] carrying the default.
#[test]
fn apply_quantization_parses_global_block() {
  let body = r#"{
      "model_type": "silero_vad",
      "quantization": { "group_size": 64, "bits": 4 }
    }"#;
  let q = apply_quantization(body).expect("quantized config parses");
  let plq = q.expect("Some(PerLayerQuantization) for quantized config");
  let global = plq.quantization.expect("global default present");
  assert_eq!(global.group_size, 64);
  assert_eq!(global.bits, 4);
}

/// `base_load_model` chains the three steps on a synthetic local dir
/// (an empty `config.json` + a path that exists). The returned bundle
/// carries the path, the verbatim JSON body, and the parsed
/// (here-`None`) quantization.
#[test]
fn base_load_model_local_path_resolves() {
  let dir = temp_dir("base_load_local");
  let body = r#"{ "model_type": "silero_vad" }"#;
  fs::write(dir.join("config.json"), body).unwrap();
  let bundle = base_load_model(&dir.to_string_lossy()).expect("local dir loads");
  assert_eq!(bundle.model_path(), dir);
  assert_eq!(bundle.config_json(), body);
  assert!(bundle.quantization().is_none());
}

/// HF post-quantize artifact: `"quantization_config"` (the longer key)
/// is the fallback mlx-audio's `utils.py:221-223` recognizes when the
/// shorter `"quantization"` is absent. Both should parse identically.
#[test]
fn apply_quantization_parses_quantization_config_key() {
  let body = r#"{
      "model_type": "voxtral",
      "quantization_config": { "bits": 4, "group_size": 64 }
    }"#;
  let q = apply_quantization(body).expect("HF-key config parses");
  let plq = q.expect("Some(PerLayerQuantization) for HF-key config");
  let global = plq.quantization.expect("global default present");
  assert_eq!(global.group_size, 64);
  assert_eq!(global.bits, 4);
}

/// mlx-audio's `quantization.get("group_size", 64)` ([utils.py:226])
/// silently defaults a missing `group_size` to 64. The LM-side parser
/// would reject this; the audio parser injects the default.
#[test]
fn apply_quantization_defaults_missing_group_size_to_64() {
  let body = r#"{
      "model_type": "voxtral",
      "quantization": { "bits": 4 }
    }"#;
  let q = apply_quantization(body).expect("missing-group_size config parses");
  let plq = q.expect("Some(PerLayerQuantization) for default-injected config");
  let global = plq.quantization.expect("global default present");
  assert_eq!(global.group_size, 64, "audio default group_size is 64");
  assert_eq!(global.bits, 4);
}

/// When BOTH `"quantization"` and `"quantization_config"` are present,
/// the top-level key (`"quantization"`) wins — matching mlx-audio's
/// `config.get("quantization", None)` precedence at utils.py:221.
#[test]
fn apply_quantization_top_level_takes_precedence_over_quantization_config() {
  let body = r#"{
      "model_type": "voxtral",
      "quantization": { "bits": 8, "group_size": 32 },
      "quantization_config": { "bits": 4, "group_size": 64 }
    }"#;
  let q = apply_quantization(body).expect("both-keys config parses");
  let plq = q.expect("Some(PerLayerQuantization) for both-keys config");
  let global = plq.quantization.expect("global default present");
  assert_eq!(global.bits, 8, "top-level `quantization` wins");
  assert_eq!(global.group_size, 32, "top-level `quantization` wins");
}

/// Python's `config.get("quantization", None)` ([utils.py:221]) treats
/// a missing key and an explicit `null` value identically — both fall
/// through to the `quantization_config` retry. A `{"quantization":
/// null, "quantization_config": {...}}` config must therefore select
/// the non-null `quantization_config` block, not error on the null.
#[test]
fn apply_quantization_null_primary_falls_back_to_quantization_config() {
  let body = r#"{
      "model_type": "voxtral",
      "quantization": null,
      "quantization_config": { "bits": 4, "group_size": 64 }
    }"#;
  let q = apply_quantization(body).expect("null-primary config falls back");
  let plq = q.expect("Some(PerLayerQuantization) from quantization_config fallback");
  let global = plq.quantization.expect("global default present");
  assert_eq!(global.bits, 4, "fallback block's `bits` selected");
  assert_eq!(
    global.group_size, 64,
    "fallback block's `group_size` selected"
  );
}

/// `{"quantization_config": null}` is the dense-model no-op (Python's
/// `config.get("quantization_config", None)` is `None`, the early
/// return at [utils.py:222-225] fires), not an error on the null
/// value.
#[test]
fn apply_quantization_only_null_quantization_config_returns_none() {
  let body = r#"{ "model_type": "voxtral", "quantization_config": null }"#;
  let q = apply_quantization(body).expect("null-only quantization_config parses as dense");
  assert!(
    q.is_none(),
    "null quantization_config → Ok(None), matches upstream's no-op early return"
  );
}

/// `{"quantization": null}` — same rationale as the
/// `quantization_config: null` case: Python's `dict.get(_, None)` on a
/// null value yields `None`, falling through to the no-op early
/// return, not erroring on the null.
#[test]
fn apply_quantization_only_null_quantization_returns_none() {
  let body = r#"{ "model_type": "voxtral", "quantization": null }"#;
  let q = apply_quantization(body).expect("null-only quantization parses as dense");
  assert!(
    q.is_none(),
    "null quantization → Ok(None), matches upstream's no-op early return"
  );
}

/// Both keys explicitly `null` is the conjunction of the two
/// preceding cases — the null-aware fallback must still reach the
/// dense no-op early return, not error on either null block.
#[test]
fn apply_quantization_both_null_returns_none() {
  let body = r#"{
      "model_type": "voxtral",
      "quantization": null,
      "quantization_config": null
    }"#;
  let q = apply_quantization(body).expect("both-null config parses as dense");
  assert!(
    q.is_none(),
    "both keys null → Ok(None), matches upstream's no-op early return"
  );
}

/// `hf://org/model` repo-id-shaped input: the rejection message's
/// **CLI workaround segment** must strip the `hf://` prefix so
/// `huggingface-cli download <repo_id>` is actionable.
///
/// The message echoes the user's raw input for context (e.g. `path
/// "hf://org/model" is not a local on-disk directory`), but the CLI
/// suggestion segment after "Fetch the model directory out of process"
/// must contain only the clean repo id.
#[test]
fn get_model_path_hf_url_prefix_yields_clean_repo_id_in_error() {
  let err = get_model_path("hf://mlx-community/silero-vad")
    .expect_err("hf:// repo id must be rejected with a clean workaround");
  let Error::OutOfRange(payload) = &err else {
    panic!("hf:// rejection must be OutOfRange, got: {err:?}");
  };
  // The structured `value` field carries `path=..., repo_id=...`: the
  // repo_id MUST be the prefix-stripped clean form (used by the CLI
  // workaround), even though the verbatim `path=` echoes the user's raw
  // input.
  let value = payload.value();
  assert!(
    value.contains("repo_id=mlx-community/silero-vad"),
    "value should embed the clean repo id (repo_id=...), got: {value}"
  );
  // Split on `repo_id=` so the assertion only inspects the repo_id
  // segment — the `path=` echo deliberately preserves the user's raw
  // input (including the `hf://` prefix).
  let repo_id_segment = value
    .split_once("repo_id=")
    .map(|(_, after)| after)
    .expect("repo_id= segment present in value");
  assert!(
    !repo_id_segment.contains("hf://"),
    "clean repo_id must not embed the `hf://` prefix, got: {repo_id_segment}"
  );
  // The context references the CLI workaround using the `<repo>` placeholder
  // (the runtime repo id is in `value`, not interpolated into the static
  // context string).
  assert!(
    payload
      .context()
      .contains("huggingface-cli download <repo>"),
    "context must reference the huggingface-cli workaround, got: {}",
    payload.context()
  );
}

/// `https://huggingface.co/org/model` URL: same — strip the URL
/// prefix so the CLI workaround is correct.
#[test]
fn get_model_path_https_huggingface_url_yields_clean_repo_id_in_error() {
  let err = get_model_path("https://huggingface.co/mlx-community/silero-vad")
    .expect_err("https://huggingface.co/ URL must be rejected with a clean workaround");
  let Error::OutOfRange(payload) = &err else {
    panic!("https://huggingface.co/ rejection must be OutOfRange, got: {err:?}");
  };
  let value = payload.value();
  assert!(
    value.contains("repo_id=mlx-community/silero-vad"),
    "value should embed the clean repo id (repo_id=...), got: {value}"
  );
  let repo_id_segment = value
    .split_once("repo_id=")
    .map(|(_, after)| after)
    .expect("repo_id= segment present in value");
  assert!(
    !repo_id_segment.contains("https://huggingface.co/"),
    "clean repo_id must not embed the full URL, got: {repo_id_segment}"
  );
  assert!(
    payload
      .context()
      .contains("huggingface-cli download <repo>"),
    "context must reference the huggingface-cli workaround, got: {}",
    payload.context()
  );
}

/// Every per-domain `audio::<domain>::load` module exposes its
/// `MODEL_REMAPPING` table under the same uniform name (no
/// per-domain prefix) so generic caller code can read
/// `audio::<domain>::load::MODEL_REMAPPING` without a per-domain
/// branch. Codec's table is empty (mlx-audio's `codec/__init__.py`
/// ships no remapping); the others mirror their upstream
/// `*-utils.py:MODEL_REMAPPING` tables.
#[test]
#[allow(non_snake_case)]
fn per_domain_load_modules_expose_uniform_MODEL_REMAPPING() {
  let tts: &[(&str, &str)] = crate::audio::tts::load::MODEL_REMAPPING;
  let stt: &[(&str, &str)] = crate::audio::stt::load::MODEL_REMAPPING;
  let sts: &[(&str, &str)] = crate::audio::sts::load::MODEL_REMAPPING;
  let vad: &[(&str, &str)] = crate::audio::vad::load::MODEL_REMAPPING;
  let lid: &[(&str, &str)] = crate::audio::lid::load::MODEL_REMAPPING;
  let codec: &[(&str, &str)] = crate::audio::codec::load::MODEL_REMAPPING;

  assert!(
    codec.is_empty(),
    "codec's MODEL_REMAPPING must be empty per upstream's no-remapping shape, got: {codec:?}"
  );
  assert!(
    !tts.is_empty(),
    "TTS MODEL_REMAPPING must mirror upstream's non-empty alias table"
  );
  assert!(
    !stt.is_empty(),
    "STT MODEL_REMAPPING must mirror upstream's non-empty alias table"
  );
  assert!(
    !sts.is_empty(),
    "STS MODEL_REMAPPING must mirror upstream's non-empty alias table"
  );
  assert!(
    !vad.is_empty(),
    "VAD MODEL_REMAPPING must mirror upstream's non-empty alias table"
  );
  assert!(
    !lid.is_empty(),
    "LID MODEL_REMAPPING must mirror upstream's non-empty alias table"
  );
}
