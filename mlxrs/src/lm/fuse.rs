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
//!   read adapter config (for fan_in_fan_out)     (fuse.py:65 → load_adapters reads this)
//!      │   lora_cfg = lora::read_adapter_config(adapter_path)
//!      │   fifo     = lora_cfg.fan_in_fan_out()  // PEFT-only, false on mlx-lm-native
//!      ▼
//!   load adapters → LoraLayers map               (fuse.py:64-66 → load(adapter_path=...))
//!      │   lora::load_adapters(&weights, adapter_path, quant.as_ref(), num_blocks)
//!      ▼
//!   for (path, layer) in layers:                 (fuse.py:68-75 → fused_linears + update_modules)
//!      │   fused = layer.fuse(dequantize)?
//!      │   if fifo: fused.weight = transpose(fused.weight)  // [out, in] → [in, out]
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
//!   copy_tokenizer_and_extras(model_path, save_path)      (fuse.py:88 → save(..., tokenizer, ...))
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
//!
//! ## Tokenizer + extras: copied from the source model directory
//!
//! `fuse.py:64,88` reads the tokenizer with the base model and hands it to
//! `save(..., tokenizer, ...)` so the saved directory is a self-contained,
//! loadable HF-style model dir. mlxrs's tokenizer surface is load-only, so
//! the on-disk tokenizer / extras files are copied verbatim from
//! `model_path` via [`crate::lm::convert::copy_tokenizer_and_extras`] —
//! the same union [`crate::lm::convert::convert`] writes
//! (`tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` /
//! `added_tokens.json` / `spiece.model` / `tokenizer.model` / `vocab.json` /
//! `merges.txt` / `chat_template.jinja` / `generation_config.json` / `*.py`).
//! The output dir therefore loads end-to-end through
//! [`crate::lm::load::load`] (which immediately constructs the tokenizer
//! after the weights / config — `load.rs:683-685`); a fused directory that
//! shipped weights + config only would silently fail to load.
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
    convert::{self, CopyDurabilityWarnings, CopyOutcome},
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
///   absent. The fused output is a self-contained, loadable HF-style model
///   dir: the rewritten weights + a cleaned `config.json` (with
///   `quantization` / `quantization_config` stripped iff `dequantize`) +
///   the tokenizer / `*.py` / `generation_config.json` extras copied
///   verbatim from `model_path` (see [the module-level tokenizer + extras
///   note](self#tokenizer--extras-copied-from-the-source-model-directory)).
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
/// - Tokenizer-extras copy from `model_path` → `save_path` failure
///   (post-save, after weights+config commit): a hard
///   [`std::fs::copy`] error surfaces as
///   [`crate::Error::ConvertPostSavePartial`] with `committed: true` (the
///   weights + config landed but at least one tokenizer file did NOT —
///   the destination is structurally incomplete and a
///   [`load::load`] would fail or load against the wrong tokenizer); a
///   post-copy `fsync` warning surfaces as [`crate::Error::DurabilityWarning`]
///   (single) or [`crate::Error::ConvertDurabilityWarnings`] (multi) —
///   same "data on disk, durability uncertain" contract
///   [`crate::lm::convert::convert`] uses.
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

  // (4) Read the adapter's typed config separately so the PEFT
  // `fan_in_fan_out` flag is available BEFORE we walk the fused layers
  // — we need it to know whether the SAVED weight should be re-transposed
  // back to the persisted `[in, out]` layout (see step 5). `load_adapters`
  // already parses the same file internally; the duplicate parse is
  // intentional and cheap (`adapter_config.json` is bounded by
  // [`crate::lm::load::MAX_CONFIG_BYTES`] and small in practice — single-
  // -digit-KB JSON).
  let lora_cfg = lora::read_adapter_config(adapter_path)?;
  let fan_in_fan_out = lora_cfg.fan_in_fan_out();

  // (5) Build the LoraLayers map (path → wrapped layer) by reading
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

  // (6) Walk the layer map, fuse each, and rewrite the weight map. Take
  // ownership of `weights` so each replaced Array is dropped (not
  // cloned) before the fused replacement lands.
  //
  // PEFT `fan_in_fan_out: true` (`lora.rs:3185-3243` documents the load
  // side): on disk the base weight is persisted `[in, out]` (Conv1D-style);
  // `build_base_linear` transposes it back to canonical `[out, in]` for the
  // forward + the fuse math. `LoraLayer::fuse` therefore returns a fused
  // [`BaseLinear`] in the canonical `[out, in]` orientation — but the
  // persisted-orientation contract on disk for any downstream PEFT-aware
  // reader is `[in, out]`. So when `fan_in_fan_out` is set we transpose the
  // fused weight back to `[in, out]` BEFORE insertion (the loader's
  // transpose-on-read inverts it). Quantized + `fan_in_fan_out` is rejected
  // at load time (`build_base_linear` `lora.rs:3212-3221`) — a packed
  // quantized weight cannot be transposed without corrupting bit-packing —
  // so the fused output we see when `fan_in_fan_out` is set is always
  // [`BaseLinear::Dense`].
  let mut weights = weights;
  for (path, layer) in &layers {
    apply_fuse_to_weights(&mut weights, path, layer, dequantize, fan_in_fan_out)?;
  }

  // (7) Per-layer quantization for the SAVED checkpoint:
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

  // (8) Save — atomic, fsync-disciplined (F6). `save` does the
  // config-stage / weights-shard / index-commit sequence; a post-commit
  // `fsync_dir` warning surfaces as `Error::DurabilityWarning {
  // committed: true, .. }` so the caller can distinguish "saved but
  // durability uncertain" from a hard pre-commit failure.
  //
  // The save side returns one of three shapes:
  //   - `Ok(())` — fully durable (every fsync passed).
  //   - `Err(DurabilityWarning { committed: true, .. })` — weights + config
  //      ARE on disk; only a parent-dir fsync warned. We REMEMBER this
  //      warning and proceed to the tokenizer-extras copy (skipping the
  //      copy on a durability-only warning would leave the destination
  //      structurally incomplete — same routing `convert::convert` uses).
  //   - any other `Err` — pre-commit failure; propagate immediately
  //      (weights / config not durably on disk; copying tokenizer extras
  //      now would only mask the real cause).
  let save_warning: Option<std::io::Error> =
    match load::save(save_path, &out_weights, &out_config_json, &save_quant) {
      Ok(()) => None,
      Err(Error::DurabilityWarning { committed, source }) if committed => Some(source),
      Err(e) => return Err(e),
    };

  // (9) Copy tokenizer + extras from the ORIGINAL `model_path` to
  // `save_path` so the fused directory is a self-contained, loadable
  // HF-style model dir (`fuse.py:88` — `save(..., tokenizer, ...)`).
  // The helper covers the same `tokenizer.json` / `tokenizer_config.json`
  // / `special_tokens_map.json` / SentencePiece-family / BPE-family /
  // `chat_template.jinja` / `generation_config.json` / `*.py` set
  // `convert::convert` writes (see the source-side `TOKENIZER_EXTRA_FILES`
  // constant + the `*.py` glob).
  //
  // We mirror `convert::convert`'s shape: route a hard copy failure to
  // `ConvertPostSavePartial` (the destination is structurally incomplete)
  // and route per-boundary fsync warnings to the typed `DurabilityWarning`
  // (single) / `ConvertDurabilityWarnings` (multi) aggregate. This keeps
  // the public error shape uniform across `convert::convert` and
  // `fuse::fuse` — both produce HF-style dirs from a base model dir; both
  // share the same post-save extras-copy contract.
  match convert::copy_tokenizer_and_extras(model_path, save_path) {
    Ok(copy_outcome) => {
      let copy_warnings = match copy_outcome {
        CopyOutcome::Committed => CopyDurabilityWarnings::default(),
        CopyOutcome::CommittedWithDurabilityWarning(w) => w,
      };
      let aggregate = crate::error::ConvertDurabilityWarnings {
        committed: true,
        save: save_warning,
        post_copy_file: copy_warnings.post_copy_file,
        post_copy_dir: copy_warnings.post_copy_dir,
      };
      match aggregate.count() {
        // 0 — fully durable end-to-end.
        0 => Ok(()),
        // 1 — surface via the existing single-warning shape
        // (`DurabilityWarning`) so the one-source contract is unchanged.
        1 => {
          let crate::error::ConvertDurabilityWarnings {
            committed: _,
            save,
            post_copy_file,
            post_copy_dir,
          } = aggregate;
          let source = save
            .or(post_copy_file)
            .or(post_copy_dir)
            .expect("count() == 1 guarantees exactly one Some field");
          Err(Error::DurabilityWarning {
            committed: true,
            source,
          })
        }
        // 2+ — typed multi-warning aggregate so each warning is reachable
        // via direct destructuring (no string fold).
        _ => Err(Error::ConvertDurabilityWarnings(aggregate)),
      }
    }
    // A hard copy failure: at least one extras file did NOT reach disk —
    // the destination dir is structurally incomplete. The save side IS
    // committed (weights + config landed before we reached the copy
    // step), so route through `ConvertPostSavePartial` with
    // `committed: true` + the save-side fsync warning (if any) carried in
    // `save_warning` and the underlying copy failure in `copy_error`.
    Err(copy_err) => Err(Error::ConvertPostSavePartial {
      committed: true,
      save_warning,
      copy_error: std::io::Error::other(copy_err.to_string()),
    }),
  }
}

/// Replace the `<path>.weight` / `.scales` / `.biases` / `.bias` entries in
/// `weights` with the result of fusing `layer` (folding the LoRA/DoRA
/// adapter into the base weight). For a Dense fused output we drop any
/// `.scales` / `.biases` siblings (the source may have been quantized and
/// is now dense); for a Quantized fused output we write the full triple.
/// For a `DoraEmbedding` we use [`LoraLayer::fuse_embedding`] which returns
/// a [`BaseEmbedding`] (no bias / no quantized variant).
///
/// `fan_in_fan_out` (PEFT `LoraConfig.fan_in_fan_out`): when `true` the base
/// weights on disk are persisted `[in_features, out_features]` (Conv1D-style)
/// rather than canonical `[out_features, in_features]`. The fuse math runs
/// on the canonical layout (`LoraLayer::fuse` returns canonical `[out, in]`),
/// but the SAVED weight must round-trip through any PEFT-aware reader: the
/// persisted-orientation contract is `[in, out]`. So before insertion we
/// transpose the fused dense weight back to `[in, out]` — `transpose` on a
/// 2-D Array swaps the two axes (equivalent to `transpose_axes(&[1, 0])`).
///
/// `fan_in_fan_out` over a quantized base is rejected at the LOAD side
/// (`lora.rs::build_base_linear`, lines 3212-3221 — transposing a packed
/// quantized weight would corrupt the bit-packing), so a fused Quantized
/// variant cannot reach the `fan_in_fan_out: true` branch. A debug-only
/// assertion in [`insert_base_linear`] guards that invariant.
fn apply_fuse_to_weights(
  weights: &mut Weights,
  path: &str,
  layer: &LoraLayer,
  dequantize: bool,
  fan_in_fan_out: bool,
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
      insert_base_linear(weights, path, fused, fan_in_fan_out)?;
    }
    LoraLayer::DoraEmbedding(_) => {
      // PEFT `fan_in_fan_out` is a *Linear* concept (the Conv1D-style
      // transposed weight only appears on linear layers — embeddings have
      // their own `[num_embeddings, dims]` layout and PEFT does not
      // transpose them). The flag is ignored for the embedding fuse path.
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
///
/// When `fan_in_fan_out` is `true` the dense `weight` is transposed back to
/// the persisted `[in_features, out_features]` orientation before insertion
/// — see [`apply_fuse_to_weights`] for the round-trip rationale. The
/// Quantized branch is unreachable on the `fan_in_fan_out: true` path
/// (rejected at load time, `lora.rs::build_base_linear` 3212-3221); a
/// debug-only assertion confirms the invariant so a future refactor that
/// loosens the load-side rejection without revisiting the fuse side fails
/// loudly in debug builds rather than silently corrupting on disk.
fn insert_base_linear(
  weights: &mut Weights,
  path: &str,
  fused: BaseLinear,
  fan_in_fan_out: bool,
) -> Result<()> {
  match fused {
    BaseLinear::Dense { weight, bias } => {
      let persisted = if fan_in_fan_out {
        weight.transpose()?
      } else {
        weight
      };
      weights.insert(format!("{path}.weight"), persisted);
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
      debug_assert!(
        !fan_in_fan_out,
        "insert_base_linear: fan_in_fan_out=true reached a Quantized fused output for \
         {path:?}; the load side rejects this combination (lora.rs::build_base_linear \
         3212-3221) — a packed quantized weight cannot be transposed without corrupting \
         the bit-packing"
      );
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
  Ok(())
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
