//! `fuse()` — the LoRA/DoRA adapter-fusion driver, ported from
//! [`mlx_lm/fuse.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/fuse.py).
//!
//! Wires the inference-time adapter loader ([`crate::lm::lora::load_adapters`] —
//! which itself returns a [`LoraLayers`](crate::lm::lora::LoraLayers) map keyed
//! by base-weight path), the per-layer
//! [`fuse`](crate::lm::lora::LoraLayer::fuse) (fold low-rank factors into the
//! base weight), and the save side ([`crate::lm::load::save`] +
//! [`crate::lm::load::save_model`]) into a one-call pipeline: read a base
//! model directory + adapter directory, fuse each LoRA/DoRA-wrapped layer
//! into its base, optionally also fully dequantize, and write the result to
//! `save_path` as a self-contained adapter-free model.
//!
//! ## Pipeline (mirrors `mlx_lm/fuse.py:62-93`)
//!
//! ```text
//!   fuse(model_path, adapter_path, save_path, dequantize)
//!      │
//!      ▼
//!   validate paths (model_path / adapter_path exist, not hf hub urls)
//!      │
//!      ▼
//!   load base config + weights                  (fuse.py:64-66 → load(model_path, return_config=True))
//!      │  (cfg_typed, config_json_text)  =  load::load_config(model_path)
//!      │   weights                       =  load::load_weights(model_path)
//!      │   quant                         =  parse_quantization(config_json_text)
//!      ▼
//!   load adapters → LoraLayers map               (fuse.py:64-66 → load(adapter_path=...))
//!      │   lora::load_adapters(&weights, adapter_path, quant.as_ref(), num_blocks)
//!      ▼
//!   for (path, layer) in layers:                 (fuse.py:68-75 → fused_linears + update_modules)
//!      │   fused = layer.fuse(dequantize)?
//!      │   replace `<path>.weight` / `.scales` / `.biases` / `.bias` in `weights`
//!      ▼
//!   if dequantize:                                (fuse.py:77-81)
//!      │   strip `quantization` + `quantization_config` from config_json_text
//!      │   dequantize_weights(weights, &quant)   (analogue of dequantize_model — mlxrs
//!      │                                          has no module tree, so the
//!      │                                          remaining quantized triples in the
//!      │                                          weight map are dequantized by name)
//!      ▼
//!   save(save_path, &weights, &config_json, &per_layer)   (fuse.py:83-91 → save(...))
//!      │
//!      ▼
//!   Ok(())
//! ```
//!
//! ## Scope decisions (deliberately NOT ported)
//!
//! Mirrors the same fences as the rest of `lm::*`:
//!
//! - **CLI / `argparse`** (`fuse.py:15-57` + `__main__`) — application
//!   surface, excluded. Callers invoke [`fuse`] directly with paths.
//! - **HuggingFace Hub upload** (`fuse.py:102-103` → `upload_to_hub`) —
//!   library does not upload. Hub-URL `model_path` / `adapter_path`
//!   (`hf://...` / `https://huggingface.co/...`) are rejected with an
//!   actionable [`Error::Backend`] pointing at `huggingface-cli download`
//!   (mirroring [`crate::audio::load::get_model_path`]'s pattern).
//! - **GGUF export** (`fuse.py:93-100` → `convert_to_gguf`) — a separate
//!   future task; out of scope here. Callers wanting GGUF run the dedicated
//!   GGUF-export driver against the fused model directory after this call
//!   returns.
//! - **Tokenizer load/save** (`fuse.py:64` reads the tokenizer; `fuse.py:88`
//!   passes it to `save`) — mlxrs's tokenizer surface is load-only and the
//!   tokenizer files on disk are NOT modified by the fuse operation, so the
//!   `save_path` carries weights + the cleaned `config.json` only. Callers
//!   wanting the full HF-style directory copy the source tokenizer files
//!   (`tokenizer.json` / `tokenizer_config.json` / etc.) alongside, the
//!   same convention [`crate::lm::convert::copy_tokenizer_and_extras`]
//!   handles for [`crate::lm::convert::convert`]. (We deliberately do NOT
//!   invoke that helper here: `fuse.py` does not copy tokenizer extras
//!   either — its `save(..., tokenizer, ...)` call only writes the
//!   in-memory tokenizer's own `save_pretrained` output, which mlxrs's
//!   load-only tokenizer has no analogue for.)
//!
//! ## API style
//!
//! Per [project memory `feedback_api_style`] the python kwarg surface
//! becomes a Rust function with explicit `&Path` arguments and `bool` —
//! `fuse(model_path, adapter_path, save_path, dequantize)`. No struct
//! wrapper (only four args, all required).
//!
//! [`Error::Backend`]: crate::Error::Backend

use std::path::Path;

use crate::{
  error::{Error, Result},
  lm::{
    load::{self, Weights},
    lora::{self, BaseEmbedding, BaseLinear, LoraLayer},
    quant::{self, PerLayerQuantization},
  },
};

/// Fuse LoRA/DoRA adapter weights into the base model + save the result as
/// an adapter-free model directory.
///
/// Mirrors `mlx_lm/fuse.py::main` orchestrator (lines 62-93). Library
/// function — no CLI parsing, no HuggingFace Hub upload, no GGUF export
/// (each excluded per [the module docs](self#scope-decisions-deliberately-not-ported)).
///
/// # References
///
/// - Python: [`mlx_lm/fuse.py:62-93`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/fuse.py)
/// - Swift: [`Adapters/LoRA/LoRAContainer.fuse`](https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/MLXLMCommon/Adapters/LoRA/LoRAContainer.swift)
///
/// # Arguments
///
/// - `model_path` — local directory with the base model: `config.json` +
///   tokenizer + weights (sharded `model*.safetensors[.index.json]` or
///   single `model.safetensors`). Per the project's local-only policy a
///   `hf://...` / `https://huggingface.co/...` URL is rejected with an
///   actionable error.
/// - `adapter_path` — local directory with `adapter_config.json` +
///   `adapters.safetensors` (mlx-lm-native) or `adapter_model.safetensors`
///   (HuggingFace PEFT). Same hub-URL rejection.
/// - `save_path` — destination directory for the fused model. Created if
///   absent. The fused output carries the rewritten weights + a cleaned
///   `config.json` (with `quantization` / `quantization_config` stripped
///   iff `dequantize`); tokenizer files are NOT copied — see the
///   [module-level scope decisions](self#scope-decisions-deliberately-not-ported).
/// - `dequantize` — if `true`, additionally produce a fully dense model:
///   strip `quantization` + `quantization_config` from the saved config,
///   and dequantize any remaining quantized triples in the weight map
///   (mirrors `dequantize_model`, `fuse.py:77-81`). When `false`, a
///   quantized base stays quantized — fused QLoRA / QDoRA layers are
///   re-quantized with the source's scheme.
///
/// # Errors (recoverable)
///
/// - Hub-URL `model_path` / `adapter_path` → [`Error::Backend`] with an
///   actionable `huggingface-cli download <repo_id>` hint.
/// - Missing `model_path` / missing or unreadable `config.json` / no
///   safetensors → [`Error::Backend`] from [`load::load_config`] /
///   [`load::load_weights`].
/// - Missing `adapter_path` / missing `adapter_config.json` / missing
///   adapter weights file / config drift → [`Error::Backend`] from
///   [`lora::load_adapters`] (see that function's docs for the full list).
/// - A LoRA factor shape that doesn't match its base layer →
///   [`Error::ShapeMismatch`] from the layer constructor.
/// - Save-side failure (directory create, shard write, index commit) →
///   [`Error::Backend`] / [`crate::Error::ShardPathCollision`] from
///   [`load::save`]; a post-commit `fsync_dir` warning surfaces as
///   [`crate::Error::DurabilityWarning`] with `committed: true` (the
///   on-disk model is loadable, only directory metadata durability is
///   uncertain).
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use mlxrs::lm::fuse;
///
/// fuse::fuse(
///   Path::new("./Qwen3-4B"),
///   Path::new("./qwen3-4b-lora-adapter"),
///   Path::new("./qwen3-4b-fused"),
///   /* dequantize: */ false,
/// )?;
/// # Ok::<(), mlxrs::Error>(())
/// ```
pub fn fuse(
  model_path: &Path,
  adapter_path: &Path,
  save_path: &Path,
  dequantize: bool,
) -> Result<()> {
  // (1) Reject hub URLs up front — local-only policy. Same rejection shape
  // as `audio::load::get_model_path`. The two checks fire BEFORE any disk
  // IO so a `hf://...` typo can't briefly stat anything.
  reject_hub_url("model_path", model_path)?;
  reject_hub_url("adapter_path", adapter_path)?;

  // (2) Load base config (typed + raw text) + weights. `load_config` /
  // `load_weights` handle missing-directory / missing-file with clear
  // `Error::Backend` messages naming the offending path — no need to
  // re-stat here.
  let (cfg_typed, config_json_text) = load::load_config(model_path)?;
  let weights = load::load_weights(model_path)?;

  // (3) Parse the on-disk per-layer quantization block (the loaded
  // quantized triples' scheme). Carried through `load_adapters` (so
  // QLoRA / QDoRA wrappers route the correct base) AND through
  // `dequantize_weights` (so remaining quantized triples find their
  // per-layer scheme); also threaded into the fused-model
  // `PerLayerQuantization` so `save_model::get_total_parameters` counts
  // params correctly.
  let parsed_quant = quant::parse_quantization(&config_json_text)?;

  // (4) Build the LoraLayers map (path → wrapped layer) by reading
  // `adapter_config.json` + the adapter weights, then matching factor
  // groups to base layers. `load_adapters` does the explicit-target
  // completeness check (an `Error::Backend` when an `adapter_config.json`
  // `keys` / `target_modules` selection misses factors, or when an
  // adapter factor group matches no base layer) so a partial / empty
  // fuse cannot silently succeed.
  let layers = lora::load_adapters(
    &weights,
    adapter_path,
    parsed_quant.as_ref(),
    cfg_typed.num_hidden_layers,
  )?;

  // (5) Walk the layer map, fuse each, and rewrite the weight map. Take
  // ownership of `weights` so each replaced Array is dropped (not
  // cloned) before the fused replacement lands.
  let mut weights = weights;
  for (path, layer) in &layers {
    apply_fuse_to_weights(&mut weights, path, layer, dequantize)?;
  }

  // (6) Per-layer quantization for the SAVED checkpoint:
  //
  // - `dequantize=true`: the fused output is fully dense — drop the
  //   block AND walk any remaining quantized triples through
  //   `dequantize_weights` (mirrors `dequantize_model`).
  // - `dequantize=false`: preserve the source's quantization scheme;
  //   `apply_fuse_to_weights` already re-quantized fused QLoRA / QDoRA
  //   layers with the same scheme, so untouched quantized layers stay
  //   quantized and the per-layer block is unchanged.
  let (out_weights, out_config_json, save_quant) = if dequantize {
    let stripped_config = strip_quantization_blocks(&config_json_text)?;
    let walk_quant = parsed_quant.unwrap_or_default();
    let dense_weights = quant::dequantize_weights(weights, &walk_quant)?;
    (
      dense_weights,
      stripped_config,
      PerLayerQuantization::default(),
    )
  } else {
    let save_quant = parsed_quant.unwrap_or_default();
    (weights, config_json_text, save_quant)
  };

  // (7) Save — atomic, fsync-disciplined (F6). `save` does the
  // config-stage / weights-shard / index-commit sequence; a post-commit
  // `fsync_dir` warning surfaces as `Error::DurabilityWarning {
  // committed: true, .. }` so the caller can distinguish "saved but
  // durability uncertain" from a hard pre-commit failure.
  load::save(save_path, &out_weights, &out_config_json, &save_quant)
}

/// Replace the `<path>.weight` / `.scales` / `.biases` / `.bias` entries in
/// `weights` with the result of fusing `layer` (folding the LoRA/DoRA
/// adapter into the base weight). For a Dense fused output we drop any
/// `.scales` / `.biases` siblings (the source may have been quantized and
/// is now dense); for a Quantized fused output we write the full triple.
/// For a `DoraEmbedding` we use [`LoraLayer::fuse_embedding`] which returns
/// a [`BaseEmbedding`] (no bias / no quantized variant).
fn apply_fuse_to_weights(
  weights: &mut Weights,
  path: &str,
  layer: &LoraLayer,
  dequantize: bool,
) -> Result<()> {
  // Drop the source layer's weight-map entries up front so the per-variant
  // insert below is the only writer (no stale `.scales` / `.biases` left
  // over when we transition a quantized source to a dense fused output).
  let weight_key = format!("{path}.weight");
  let scales_key = format!("{path}.scales");
  let biases_key = format!("{path}.biases");
  let bias_key = format!("{path}.bias");
  weights.remove(&weight_key);
  weights.remove(&scales_key);
  weights.remove(&biases_key);
  weights.remove(&bias_key);

  match layer {
    LoraLayer::Lora(_) | LoraLayer::Dora(_) => {
      let fused = layer.fuse(dequantize)?;
      insert_base_linear(weights, path, fused);
    }
    LoraLayer::DoraEmbedding(_) => {
      let fused = layer.fuse_embedding()?;
      insert_base_embedding(weights, path, fused);
    }
  }
  Ok(())
}

/// Insert a fused [`BaseLinear`] back into the weight map under `path`:
/// the dense `[output_dims, input_dims]` weight at `<path>.weight`, the
/// optional output bias at `<path>.bias`, and (for a re-quantized base)
/// the `<path>.scales` + (`affine`-only) `<path>.biases` triple.
fn insert_base_linear(weights: &mut Weights, path: &str, fused: BaseLinear) {
  match fused {
    BaseLinear::Dense { weight, bias } => {
      weights.insert(format!("{path}.weight"), weight);
      if let Some(b) = bias {
        weights.insert(format!("{path}.bias"), b);
      }
    }
    BaseLinear::Quantized {
      weight,
      scales,
      quant_biases,
      bias,
      ..
    } => {
      weights.insert(format!("{path}.weight"), weight);
      weights.insert(format!("{path}.scales"), scales);
      if let Some(qb) = quant_biases {
        weights.insert(format!("{path}.biases"), qb);
      }
      if let Some(b) = bias {
        weights.insert(format!("{path}.bias"), b);
      }
    }
  }
}

/// Insert a fused [`BaseEmbedding`] back into the weight map under `path`.
/// `BaseEmbedding` is dense-only (mlx-lm's `DoRAEmbedding` does not
/// support a quantized base — `tuner/dora.py:141-142`), so this writes
/// just the `[num_embeddings, dims]` weight under `<path>.weight`.
fn insert_base_embedding(weights: &mut Weights, path: &str, fused: BaseEmbedding) {
  match fused {
    BaseEmbedding::Dense { weight } => {
      weights.insert(format!("{path}.weight"), weight);
    }
  }
}

/// Strip `quantization` and `quantization_config` keys from a `config.json`
/// body — mirrors `fuse.py:80-81` (`config.pop("quantization", None);
/// config.pop("quantization_config", None)`) and is structurally identical
/// to [`crate::lm::convert`]'s helper of the same name (kept local rather
/// than re-exported so each module's helper stays adjacent to its sole
/// caller — they cannot diverge: both delete the same two top-level keys).
fn strip_quantization_blocks(config_json: &str) -> Result<String> {
  let value: serde_json::Value = serde_json::from_str(config_json).map_err(|e| Error::Backend {
    message: format!("fuse: source config is not valid JSON: {e}"),
  })?;
  let serde_json::Value::Object(mut map) = value else {
    return Err(Error::Backend {
      message: "fuse: source config JSON must be an object".into(),
    });
  };
  map.remove("quantization");
  map.remove("quantization_config");
  let stripped = serde_json::Value::Object(map);
  serde_json::to_string(&stripped).map_err(|e| Error::Backend {
    message: format!("fuse: cannot re-serialize stripped config: {e}"),
  })
}

/// Reject a hub-style URL (`hf://...` / `https://huggingface.co/...` /
/// `http://huggingface.co/...`) passed for a local path argument. Mirrors
/// [`crate::audio::load::get_model_path`]'s rejection: strip the URL
/// prefix before interpolating the repo-id into the actionable hint, so
/// the user sees a copy-pasteable `huggingface-cli download <repo_id>`
/// rather than `huggingface-cli download hf://<repo_id>` (broken advice).
fn reject_hub_url(arg_name: &str, path: &Path) -> Result<()> {
  let Some(s) = path.to_str() else {
    return Ok(());
  };
  let repo_id = s
    .strip_prefix("hf://")
    .or_else(|| s.strip_prefix("https://huggingface.co/"))
    .or_else(|| s.strip_prefix("http://huggingface.co/"));
  if let Some(repo_id) = repo_id {
    return Err(Error::Backend {
      message: format!(
        "fuse: `{arg_name}` is a HuggingFace Hub URL ({s}); mlxrs is local-only \
         and does not download from the Hub. Fetch the model directory out of \
         process (e.g. `huggingface-cli download {repo_id}` or \
         `hf download {repo_id}`) and pass the resulting local path."
      ),
    });
  }
  Ok(())
}

// ─────────────────────────── unit tests ───────────────────────────

#[cfg(test)]
mod tests {
  //! Unit tests for the local helpers: helper-shape checks (hub-URL
  //! rejection, config-strip, base-linear insert). Integration tests for
  //! the public `fuse()` orchestrator live in `tests/lm_fuse.rs` (the
  //! same convention `lm_convert.rs` follows for `convert()`).

  use super::*;

  #[test]
  fn reject_hub_url_strips_hf_prefix_in_hint() {
    let err =
      reject_hub_url("model_path", Path::new("hf://mlx-community/Qwen3-4B-bf16")).unwrap_err();
    let Error::Backend { message } = err else {
      panic!("expected Error::Backend");
    };
    assert!(
      message.contains("huggingface-cli download mlx-community/Qwen3-4B-bf16"),
      "hint should carry the bare repo-id: {message}"
    );
    assert!(
      message.contains("model_path"),
      "hint names the rejected arg: {message}"
    );
  }

  #[test]
  fn reject_hub_url_strips_https_prefix_in_hint() {
    let err = reject_hub_url(
      "adapter_path",
      Path::new("https://huggingface.co/owner/repo"),
    )
    .unwrap_err();
    let Error::Backend { message } = err else {
      panic!("expected Error::Backend");
    };
    assert!(
      message.contains("huggingface-cli download owner/repo"),
      "hint should carry the bare repo-id (no protocol): {message}"
    );
  }

  #[test]
  fn reject_hub_url_passes_through_local_paths() {
    assert!(reject_hub_url("model_path", Path::new("/tmp/model")).is_ok());
    assert!(reject_hub_url("model_path", Path::new("./relative/path")).is_ok());
    assert!(reject_hub_url("model_path", Path::new("~/local/path")).is_ok());
    assert!(reject_hub_url("model_path", Path::new("local")).is_ok());
  }

  #[test]
  fn strip_quantization_blocks_removes_both_keys() {
    let src = r#"{
      "model_type": "qwen3",
      "quantization": { "group_size": 64, "bits": 4 },
      "quantization_config": { "group_size": 64, "bits": 4 }
    }"#;
    let stripped = strip_quantization_blocks(src).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stripped).unwrap();
    assert!(
      parsed.get("quantization").is_none(),
      "`quantization` removed"
    );
    assert!(
      parsed.get("quantization_config").is_none(),
      "`quantization_config` removed"
    );
    // Other keys preserved.
    assert_eq!(
      parsed.get("model_type").and_then(|v| v.as_str()),
      Some("qwen3")
    );
  }

  #[test]
  fn strip_quantization_blocks_passes_through_without_keys() {
    let src = r#"{ "model_type": "qwen3", "hidden_size": 16 }"#;
    let stripped = strip_quantization_blocks(src).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stripped).unwrap();
    assert_eq!(
      parsed.get("model_type").and_then(|v| v.as_str()),
      Some("qwen3")
    );
    assert_eq!(parsed.get("hidden_size").and_then(|v| v.as_i64()), Some(16));
  }

  #[test]
  fn strip_quantization_blocks_rejects_non_object_root() {
    let src = "[1, 2, 3]";
    let err = strip_quantization_blocks(src).unwrap_err();
    let Error::Backend { message } = err else {
      panic!("expected Error::Backend");
    };
    assert!(
      message.contains("must be an object"),
      "diagnostic names the constraint: {message}"
    );
  }
}
