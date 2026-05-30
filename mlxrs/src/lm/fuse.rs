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
//! ## Pipeline
//!
//! Mirrors `mlx_lm/fuse.py:62-93`, hardened past fuse.py's permissive
//! post-save copy with a staging-dir tokenizer snapshot.
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
//!   create save_path + a unique `<save_path>/.staging-fuse-<pid>-<nanos>-<ctr>/`
//!      │  staging dir (the snapshot bay — survives the fuse pipeline and is
//!      │  guard-deleted on every exit path via [`StagingDir::Drop`])
//!      ▼
//!   snapshot tokenizer + extras INTO staging  (Codex R3 Findings 1+2 fix)
//!      │   copy_tokenizer_and_extras(model_path, &staging_dir)?;
//!      │   // freezes the EXACT bytes the fused dir will ship — eliminates
//!      │   // the TOCTOU window where a re-read of `model_path` between
//!      │   // validate (T0) and copy (T1) could surface different tokenizer
//!      │   // bytes
//!      ▼
//!   validate the SNAPSHOT is loadable           (fuse.py validate-source equivalent,
//!      │   _ = load::load_tokenizer(&staging_dir, &cfg_typed)?;     re-pointed at
//!      ▼                                                            the staged bytes
//!                                                                   so validate +
//!                                                                   copy see the
//!                                                                   SAME bytes)
//!   read adapter config ONCE (shared by load + save) (fuse.py:65 → load_adapters reads this)
//!      │   lora_cfg = lora::read_adapter_config(adapter_path)
//!      │   fifo     = lora_cfg.fan_in_fan_out()  // PEFT-only, false on mlx-lm-native
//!      ▼
//!   load adapters → LoraLayers map               (fuse.py:64-66 → load(adapter_path=...))
//!      │   lora::load_adapters_with_config(&weights, adapter_path,
//!      │                                  &lora_cfg, quant.as_ref(), num_blocks)
//!      │   // single-snapshot: same parsed `lora_cfg` the save side uses
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
//!   promote staging → save_path                  (replaces fuse.py:88 `save(..., tokenizer, ...)`
//!      │   for each staged file: rename into save_path (overwrites)         with the snapshot+
//!      │   for each TOKENIZER_EXTRA_FILES name + every `*.py` at save_path: promote+stale-walk
//!      │     if NOT present in staging → unlink (Codex R3 F1 fix —          shape that
//!      │     stale extras the source didn't carry are dropped)              eliminates both
//!      │   remove staging dir                                               findings)
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
//! ## Tokenizer + extras: snapshot-and-promote from the source dir
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
//! ### Staging-dir snapshot (Codex R3 — two HIGH findings fixed structurally)
//!
//! The pre-R3 shape called [`crate::lm::convert::copy_tokenizer_and_extras`]
//! directly into `save_path` and validated `model_path`'s tokenizer at a
//! DIFFERENT time. Two related defects fell out:
//!
//! 1. **F1 — stale-extras retention.** A permissive destination (the
//!    fuse.py contract) only had files OVERWRITTEN when the source carried
//!    them. So a `save_path/generation_config.json` /
//!    `save_path/chat_template.jinja` / `save_path/*.py` left over from a
//!    PREVIOUS model SURVIVED the fuse when the new source lacked the same
//!    file. [`crate::lm::load::load_config`] consumes the leftover
//!    `generation_config.json` as the EOS override and the tokenizer surface
//!    consumes the leftover `tokenizer_config.json` / `chat_template.jinja`,
//!    so the output dir silently loads with wrong-model semantics — the
//!    EXACT cross-model-contamination concern R2 raised but only mitigated
//!    on the present-at-source axis (a `std::fs::copy` overwrite).
//! 2. **F2 — validate vs copy TOCTOU.** `load_tokenizer(model_path)`
//!    validated at `T0`; `copy_tokenizer_and_extras(model_path, save_path)`
//!    re-read `model_path` at `T1`. Any mid-flight mutation (the source dir
//!    gets a deleted / swapped / partially-written `tokenizer.json`)
//!    surfaces as a fuse that returns `Ok(())` with the validated-tokenizer
//!    contract met but the actually-shipped tokenizer different from (or
//!    missing relative to) what was validated.
//!
//! The R3 fix collapses both defects with one structural change: copy
//! tokenizer + extras into a unique `<save_path>/.staging-fuse-*` staging
//! directory FIRST (freezes a snapshot of the source bytes), validate the
//! SNAPSHOT (the same bytes that will be shipped), run the rest of the
//! pipeline, then PROMOTE staging → `save_path` with a stale-extras
//! sweep — unlink any [`TOKENIZER_EXTRA_FILES`]-family name or `*.py` at
//! `save_path` that the snapshot does NOT carry. The promote is the
//! single ship point: a failure BEFORE promote drops the staging dir on
//! `Drop`, leaving `save_path` untouched (or, if `load::save` already
//! committed the weights+config, the partial state is reported via
//! [`Error::ConvertPostSavePartial`]); a failure AFTER promote is
//! reported via [`Error::ConvertPostSavePartial`] too.
//!
//! [`TOKENIZER_EXTRA_FILES`]: crate::lm::convert
//! [`Error::ConvertPostSavePartial`]: crate::Error::ConvertPostSavePartial
//!
//! ## API style
//!
//! Per [project memory `feedback_api_style`] the python kwarg surface
//! becomes a Rust function with explicit `&Path` arguments and `bool` —
//! `fuse(model_path, adapter_path, save_path, dequantize)`. No struct
//! wrapper (only four args, all required).
//!
//! [`Error::Backend`]: crate::Error::Backend

use std::path::{Path, PathBuf};

use crate::{
  error::{
    ConvertPostSavePartialPayload, DurabilityWarningPayload, Error, FileIoPayload, FileOp,
    InvariantViolationPayload, LayerKeyedPayload, ParsePayload, Result,
  },
  lm::{
    convert::{self, TOKENIZER_EXTRA_FILES},
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
/// - Missing or malformed `model_path/tokenizer.json` (or unparseable
///   `tokenizer_config.json` if present) → [`Error::Backend`] from
///   [`load::load_tokenizer`]. Validated BEFORE any save-side IO so a
///   source without a usable tokenizer fails fast — the alternative
///   (leaving the validation to the post-save `copy_tokenizer_and_extras`
///   step) would silently skip absent files and produce an unloadable
///   `save_path` while returning `Ok(())`.
/// - Missing `adapter_path` / missing `adapter_config.json` / missing
///   adapter weights file / config drift → [`Error::Backend`] from
///   [`lora::load_adapters`] (see that function's docs for the full list).
/// - A LoRA factor shape that doesn't match its base layer →
///   [`Error::RankMismatch`] / [`Error::LengthMismatch`] / [`Error::ShapePairMismatch`]
///   from the layer constructor.
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

  // (2a) **Snapshot tokenizer + extras into a staging dir.** Codex R3
  // Findings 1+2 fix.
  //
  // The pre-R3 shape did `let _ = load::load_tokenizer(model_path, ...)`
  // here AND `copy_tokenizer_and_extras(model_path, save_path)` AFTER
  // save — TWO `model_path` reads at two different times (TOCTOU window:
  // F2), and the post-save copy only OVERWROTE files PRESENT at the
  // source so any stale extras at the destination survived (F1).
  //
  // The R3 fix collapses both: copy `model_path`'s tokenizer + extras
  // into a unique `<save_path>/.staging-fuse-*` directory FIRST so the
  // bytes that will be shipped are frozen at a single point in time,
  // then validate THAT SNAPSHOT (no second read of `model_path`), then
  // (after the rest of the pipeline runs) promote the snapshot into
  // `save_path` with a stale-extras sweep that unlinks any
  // [`TOKENIZER_EXTRA_FILES`]-family name or `*.py` at `save_path` the
  // snapshot does NOT carry.
  //
  // `save_path` is created up front so the staging directory has a
  // parent to land in (the F6 `load::save` would create it later, but
  // we need it now so the same-fs `std::fs::rename` of staged files
  // into `save_path` during promote stays atomic — cross-fs renames
  // silently degrade to copy+unlink and lose atomicity). The staging
  // guard's [`Drop`] removes the staging dir on every error exit so a
  // failure between here and promotion never leaves `<save_path>/.staging-fuse-*`
  // junk on disk.
  std::fs::create_dir_all(save_path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "fuse: cannot create save_path",
      FileOp::Create,
      save_path.to_path_buf(),
      e,
    ))
  })?;
  let staging = StagingDir::create(save_path)?;
  match convert::copy_tokenizer_and_extras(model_path, staging.path()) {
    Ok(_outcome) => {
      // The `_outcome` carries best-effort post-copy fsync warnings on
      // the STAGING dir. Those warnings describe durability of the
      // staging copy ONLY — we are about to re-rename every staged file
      // into `save_path` and fsync the destination's parent dir during
      // promotion, so the staging-dir's fsync warnings are intermediate
      // (the staging dir gets removed after promote anyway). Discard.
    }
    Err(snapshot_err) => {
      // The staging guard's `Drop` will clean up the (partial) staging
      // dir on the way out. Propagate the snapshot error verbatim so
      // the caller sees the actionable "cannot copy tokenizer file"
      // diagnostic. No save artifacts have been written yet.
      return Err(snapshot_err);
    }
  }

  // (2b) Validate the STAGED snapshot is loadable BEFORE any save work
  // begins. The `tokenizer` itself is unused on this side (mlxrs's
  // tokenizer surface is load-only); the call is a side-effect parse to
  // prove the snapshot directory has a usable `tokenizer.json` (+
  // optional `tokenizer_config.json` schema). Reading from
  // `staging.path()` rather than `model_path` is the F2 fix: a re-read
  // of `model_path` here would re-open the TOCTOU window the snapshot
  // closed.
  //
  // Failure here means the SOURCE tokenizer bytes don't parse — same
  // "missing / malformed source tokenizer" contract the pre-R3 shape
  // had, but routed through the snapshot so the validate and copy see
  // the SAME bytes. The staging guard's `Drop` cleans the snapshot up;
  // no save artifacts have landed. The validate failure is re-wrapped
  // with `model_path` context so the diagnostic still names the SOURCE
  // path the caller needs to fix (the snapshot path is an internal
  // implementation detail).
  if let Err(e) = load::load_tokenizer(staging.path(), &cfg_typed) {
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      model_path.display().to_string(),
      e,
    )));
  }

  // (3) Parse the on-disk per-layer quantization block (the loaded
  // quantized triples' scheme). Carried through `load_adapters` (so
  // QLoRA / QDoRA wrappers route the correct base) AND through
  // `dequantize_weights` (so remaining quantized triples find their
  // per-layer scheme); also threaded into the fused-model
  // `PerLayerQuantization` so `save_model::get_total_parameters` counts
  // params correctly.
  let parsed_quant = quant::parse_quantization(&config_json_text)?;

  // (4) Read the adapter's typed config ONCE and share it across both the
  // load side (via [`lora::load_adapters_with_config`]) and the save side
  // (the `fan_in_fan_out` decision below). The previous shape called
  // [`lora::read_adapter_config`] here AND let [`lora::load_adapters`]
  // re-open + re-parse the same file — two reads = two snapshots = a
  // TOCTOU window where a hostile or just-flipped `adapter_config.json`
  // could send the load side and the save side down divergent paths:
  //
  // - Square-target `fan_in_fan_out` flag flip ⇒ the load side would
  //   build a canonical `[out, in]` fused weight, then the save side's
  //   stale-snapshot decision would transpose it to `[in, out]` (or
  //   skip the transpose), silently corrupting the saved orientation.
  //   The R1-fix square-target test (`fuse_preserves_fan_in_fan_out_layout
  //   _for_square_target`) is exactly this failure mode masked by a
  //   silent corruption — re-opening that window between two reads
  //   reopens the same bug.
  // - Quantized + `fan_in_fan_out` flag flip ⇒ the load side passes
  //   (`build_base_linear` rejects ONLY when load-side sees
  //   `fan_in_fan_out: true`), and the save side's
  //   [`insert_base_linear`] quantized arm fires its debug-only
  //   `debug_assert!(!fan_in_fan_out, ...)`, panicking debug builds.
  //
  // Holding ONE parsed [`LoraConfig`] across both phases collapses both
  // reads to the same snapshot. The cost is one fewer ~single-digit-KB
  // disk read for the load side (cheaper, not more expensive).
  let lora_cfg = lora::read_adapter_config(adapter_path)?;
  let fan_in_fan_out = lora_cfg.fan_in_fan_out();

  // (5) Build the LoraLayers map (path → wrapped layer) by loading the
  // adapter weights against the SHARED parsed [`LoraConfig`] (`lora_cfg`
  // above) — no second `adapter_config.json` read. `load_adapters_with_config`
  // does the explicit-target completeness check (an `Error::Backend` when
  // an `adapter_config.json` `keys` / `target_modules` selection misses
  // factors, or when an adapter factor group matches no base layer) so a
  // partial / empty fuse cannot silently succeed.
  let layers = lora::load_adapters_with_config(
    &weights,
    adapter_path,
    &lora_cfg,
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
  // `fsync_dir` warning surfaces as `Error::DurabilityWarning(DurabilityWarningPayload::new(true, ))` so the caller can distinguish "saved but
  // durability uncertain" from a hard pre-commit failure.
  //
  // The save side returns one of three shapes:
  //   - `Ok(())` — fully durable (every fsync passed).
  //   - `Err(DurabilityWarning { committed: true, .. })` — weights + config
  //      ARE on disk; only a parent-dir fsync warned. We REMEMBER this
  //      warning and proceed to the staging-promote step (skipping the
  //      promote on a durability-only warning would leave the destination
  //      structurally incomplete — same routing `convert::convert` uses).
  //   - any other `Err` — pre-commit failure; propagate immediately
  //      (weights / config not durably on disk; promoting tokenizer
  //      extras now would only mask the real cause). The staging
  //      guard's `Drop` cleans up the snapshot.
  let save_warning: Option<std::io::Error> =
    match load::save(save_path, &out_weights, &out_config_json, &save_quant) {
      Ok(()) => None,
      Err(Error::DurabilityWarning(p)) if p.committed() => Some(p.into_source()),
      Err(e) => return Err(e),
    };

  // (9) **Promote staging → save_path** with a stale-extras sweep.
  //
  // This is the single ship point for the tokenizer + extras. It runs
  // ONLY after `load::save` committed the weights+config (so the
  // save-side state is definitive) and reads bytes ONLY from `staging`
  // (so the F2 TOCTOU window is closed: validate at step 2b and
  // promotion here both consume the SAME on-disk bytes). The stale
  // sweep handles F1: a [`TOKENIZER_EXTRA_FILES`] name or `*.py` at
  // `save_path` that the snapshot did NOT carry is unlinked, so a
  // permissive destination (the fuse.py contract this orchestrator
  // mirrors) cannot ship a stale `generation_config.json` /
  // `chat_template.jinja` / `tokenizer_config.json` / `*.py` from an
  // earlier model.
  //
  // Any hard IO failure inside promotion routes through
  // [`Error::ConvertPostSavePartial`] (the destination is structurally
  // incomplete — weights+config are committed but tokenizer transfer
  // didn't finish) with `committed: true` and the save-side fsync
  // warning carried separately in `save_warning`. A best-effort
  // staging-dir cleanup failure during promotion is recorded as a
  // warning (see `StagingDir::Drop`) but does NOT fail the operation:
  // the destination is correct; only a stray `<save_path>/.staging-fuse-*`
  // dir survived.
  let promote_outcome = promote_staging_into_save_path(staging, save_path);
  match promote_outcome {
    Ok(post_promote) => {
      // Aggregate durability warnings from save + the post-promote
      // per-file and per-dir fsyncs into the same typed shape
      // `convert::convert` uses. 0 → Ok(()), 1 → DurabilityWarning,
      // 2+ → ConvertDurabilityWarnings. The PRE-R3 shape funneled
      // `copy_tokenizer_and_extras`'s CopyDurabilityWarnings here;
      // post-R3 we carry the equivalent fsync boundaries observed
      // during the staged-file rename + post-promote dir fsync.
      let aggregate = crate::error::ConvertDurabilityWarnings {
        committed: true,
        save: save_warning,
        post_copy_file: post_promote.post_promote_file,
        post_copy_dir: post_promote.post_promote_dir,
      };
      match aggregate.count() {
        // 0 — fully durable end-to-end.
        0 => Ok(()),
        // 1 — surface via the existing single-warning shape
        // (`DurabilityWarning`) so the one-source contract is unchanged.
        1 => {
          let (_, save, post_copy_file, post_copy_dir) = aggregate.into_parts();
          let source = save
            .or(post_copy_file)
            .or(post_copy_dir)
            .expect("count() == 1 guarantees exactly one Some field");
          Err(Error::DurabilityWarning(DurabilityWarningPayload::new(
            true, source,
          )))
        }
        // 2+ — typed multi-warning aggregate so each warning is reachable
        // via direct destructuring (no string fold).
        _ => Err(Error::ConvertDurabilityWarnings(aggregate)),
      }
    }
    // A hard promotion failure: at least one staged file did NOT reach
    // `save_path`, or a stale-extras unlink failed. The destination dir
    // is structurally incomplete. The save side IS committed (weights +
    // config landed before promotion), so route through
    // `ConvertPostSavePartial` with `committed: true` + the save-side
    // fsync warning (if any) carried in `save_warning` and the
    // underlying promotion failure in `copy_error`.
    // `promote_err` passed through as the typed `crate::Error` so
    // `Error::FileIo(FileIoPayload { .. })` survives end-to-end for
    // recovery code (R-final fix: no `io::Error::other(...)`
    // stringification of the structured copy failure).
    Err(promote_err) => Err(Error::ConvertPostSavePartial(
      ConvertPostSavePartialPayload::new(true, save_warning, promote_err),
    )),
  }
}

// ─────────────────────── staging-dir guard ───────────────────────

/// A scoped, named temporary subdirectory of `<save_path>/` that holds
/// the tokenizer + extras SNAPSHOT until promotion. Created via
/// [`StagingDir::create`], promoted via [`StagingDir::consume`], and
/// removed on `Drop` if neither was called (every error exit between
/// staging and promotion).
///
/// **Naming:** `<save_path>/.staging-fuse-<pid>-<nanos>-<ctr>/`. Mirrors
/// the F6 [`crate::lm::load::save_model`] tempfile naming pattern
/// (`open_excl_temp_shard`'s `<filename>.<pid>.<rand>.tmp.safetensors`):
/// pid + monotonic-nanos + process-unique atomic counter give a
/// collision-resistant name that two concurrent `fuse()` calls into the
/// same `save_path` won't share. The leading `.staging-fuse-` prefix
/// keeps the name outside the [`TOKENIZER_EXTRA_FILES`] family and
/// outside the `*.py` glob the stale-sweep walks, so concurrent
/// promotions can't unlink each other's staging directories.
///
/// **Sibling rationale (vs `tempfile::tempdir_in`):** the `tempfile`
/// crate is NOT a workspace dependency (per `Cargo.toml`), and the
/// existing F6 [`open_excl_temp_shard`] / test [`fresh_dir`] code uses
/// the same hand-rolled pid+nanos+counter pattern. Matching that pattern
/// avoids introducing a new crate dependency for a one-call use site.
struct StagingDir {
  /// Absolute (or save-path-relative) path of the staging directory.
  /// `None` after [`consume`](Self::consume) so [`Drop`] is a no-op for
  /// the successful-promote case.
  ///
  /// On every error exit between [`create`](Self::create) and `consume`
  /// the path is `Some`, and `Drop` removes the directory recursively.
  path: Option<PathBuf>,
}

impl StagingDir {
  /// Create a fresh staging directory under `parent` and return the
  /// guard. The directory itself is created with `create_dir` (not
  /// `create_dir_all`) so a name collision (extraordinarily unlikely
  /// with the pid+nanos+counter pattern) surfaces as `AlreadyExists`
  /// and we retry with a new name. The retry cap matches
  /// [`open_excl_temp_shard`]'s `MAX_RETRIES = 16`.
  fn create(parent: &Path) -> Result<Self> {
    use std::{
      fs::create_dir,
      io::ErrorKind,
      sync::atomic::{AtomicU64, Ordering},
      time::{SystemTime, UNIX_EPOCH},
    };
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    const MAX_RETRIES: u32 = 16;

    let pid = std::process::id();
    let mut last_err: Option<std::io::Error> = None;
    for _ in 0..MAX_RETRIES {
      let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
      let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
      let rand = nanos ^ counter.rotate_left(17);
      let candidate = parent.join(format!(".staging-fuse-{pid}-{rand:016x}"));
      match create_dir(&candidate) {
        Ok(()) => {
          return Ok(StagingDir {
            path: Some(candidate),
          });
        }
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
          last_err = Some(e);
          continue;
        }
        Err(e) => {
          return Err(Error::FileIo(FileIoPayload::new(
            "fuse: cannot create staging dir",
            FileOp::Create,
            parent.to_path_buf(),
            e,
          )));
        }
      }
    }
    Err(Error::FileIo(FileIoPayload::new(
      "fuse: exhausted staging-dir create_dir retries (MAX_RETRIES collisions — likely a \
        hostile staging-dir race or a filesystem refusing mkdir)",
      FileOp::Create,
      parent.to_path_buf(),
      last_err.unwrap_or_else(|| std::io::Error::from(std::io::ErrorKind::AlreadyExists)),
    )))
  }

  /// The on-disk path of the staging directory.
  fn path(&self) -> &Path {
    self
      .path
      .as_deref()
      .expect("StagingDir::path called after consume — should be unreachable")
  }

  /// Consume the guard, returning the on-disk path WITHOUT removing it.
  /// The caller is responsible for the eventual cleanup (after a
  /// successful promotion the staging dir is empty and removed by
  /// `promote_staging_into_save_path`'s final `remove_dir`).
  fn consume(mut self) -> PathBuf {
    self
      .path
      .take()
      .expect("StagingDir::consume called twice — should be unreachable")
  }
}

impl Drop for StagingDir {
  /// Best-effort cleanup. Called on every error exit between
  /// [`StagingDir::create`] and a successful [`StagingDir::consume`].
  /// A failure to remove the staging dir is recorded as an `eprintln!`
  /// warning (no `tracing` dep in `mlxrs`) — the destination dir is
  /// correct; only a stray `<save_path>/.staging-fuse-*` survived.
  fn drop(&mut self) {
    if let Some(path) = self.path.take()
      && let Err(e) = std::fs::remove_dir_all(&path)
    {
      eprintln!(
        "fuse: warning — could not remove staging dir {}: {e}",
        path.display()
      );
    }
  }
}

// ─────────────────────── promote staging → save_path ───────────────────────

/// Post-promotion durability warnings surfaced by
/// [`promote_staging_into_save_path`]. Mirrors
/// [`crate::lm::convert::CopyDurabilityWarnings`]'s shape so the
/// `fuse()` aggregate can plug straight into the same
/// [`crate::error::ConvertDurabilityWarnings`] structure.
struct PostPromoteWarnings {
  /// First per-file `fsync` warning observed AFTER a successful rename
  /// of a staged file into `save_path`. The data IS on disk after the
  /// rename (and observable by a subsequent reader); only durability
  /// across a power loss is uncertain.
  post_promote_file: Option<std::io::Error>,
  /// Post-promotion `fsync_dir(save_path)` warning. The new directory
  /// entries (the renamed-in staged files + the unlinked stale extras)
  /// ARE observable by a reader; only durability is uncertain.
  post_promote_dir: Option<std::io::Error>,
}

/// Move every staged file at `staging` into `save_path` (overwriting
/// any existing same-name destination), unlink any
/// [`TOKENIZER_EXTRA_FILES`] / `*.py` at `save_path` the snapshot did
/// NOT carry, then remove the (now-empty) staging directory.
///
/// **Same-fs invariant.** `staging` is created under `save_path` by
/// [`StagingDir::create`], so every per-file [`std::fs::rename`] below
/// is single-fs (atomic on POSIX/Windows; a cross-fs rename silently
/// degrades to copy+unlink and loses atomicity). The shared parent fs
/// is the same property [`open_excl_temp_shard`] enforces for the
/// shard tempfiles.
///
/// **Stale-extras sweep.** The walk covers EXACTLY the file family
/// [`copy_tokenizer_and_extras`] writes: every name in
/// [`TOKENIZER_EXTRA_FILES`] (the fixed list — `tokenizer.json` /
/// `tokenizer_config.json` / `special_tokens_map.json` /
/// `added_tokens.json` / `spiece.model` / `tokenizer.model` /
/// `vocab.json` / `merges.txt` / `chat_template.jinja` /
/// `generation_config.json`) PLUS every `*.py` at `save_path`. Any name
/// in that family present at `save_path` but NOT present in the staging
/// snapshot is unlinked. Files OUTSIDE the family — `config.json`,
/// `model.safetensors*`, `model.safetensors.index.json` — are NOT
/// touched (those are F6's `load::save`-owned artifacts and the F1
/// finding is specifically scoped to the same family
/// `copy_tokenizer_and_extras` writes).
///
/// **Returns** the post-promotion fsync warnings (see
/// [`PostPromoteWarnings`]). A hard IO failure — a rename failure, an
/// unlink failure on a stale extra, a `read_dir(save_path)` failure
/// during the stale-sweep, a non-regular reserved basename (F1 R4) —
/// short-circuits with `Err(Error::Backend)` the caller routes through
/// [`Error::ConvertPostSavePartial`]. The stale-sweep's
/// `read_dir(staging)` failure also short-circuits the same way.
///
/// **F2 staging-guard contract** (Codex R4 MEDIUM). The `staging`
/// guard stays ARMED through the entire borrow-only inner pass; only
/// the success path consumes it and explicitly `remove_dir`s the
/// now-empty staging directory. Every `Err` path drops the guard
/// in-frame, and [`StagingDir::Drop`] `remove_dir_all`s the staging
/// dir — so a promotion that fails halfway through (rename failure,
/// stale-sweep failure, F1 R4 non-regular reserved basename) never
/// leaks a `.staging-fuse-*` dir under `save_path`.
fn promote_staging_into_save_path(
  staging: StagingDir,
  save_path: &Path,
) -> Result<PostPromoteWarnings> {
  // **F2 fix (Codex R4 MEDIUM).** `staging` stays ARMED for the entire
  // borrow-only run of [`promote_staging_inner`]. Any `?` early-return
  // from the inner helper (a `read_dir`/`rename` failure, a stale-sweep
  // unlink failure, the F1 non-regular reserved-path rejection)
  // propagates out of THIS function with `staging` still owned by this
  // frame, so the `StagingDir::Drop` impl fires and `remove_dir_all`s
  // the staging directory. Pre-R4, `staging.consume()` was the FIRST
  // call inside the function: any later error returned WITHOUT cleanup
  // and a `.staging-fuse-*` dir leaked under `save_path`.
  let post_promote_file = promote_staging_inner(staging.path(), save_path)?;

  // Success path: disarm the guard, then explicitly `remove_dir` the
  // (now empty) staging directory. We use `remove_dir` (not
  // `remove_dir_all`) on the success branch so an unexpected stray file
  // in the staging dir surfaces as an `ENOTEMPTY` warning rather than
  // being silently nuked. (The error branch above still uses
  // `remove_dir_all` via `Drop` — that's correct because an in-flight
  // failure may have left partial-write artifacts and we must not
  // refuse cleanup of our own scratch space on the error path.)
  let staging_path = staging.consume();
  if let Err(e) = std::fs::remove_dir(&staging_path) {
    eprintln!(
      "fuse: warning — could not remove empty staging dir {}: {e}",
      staging_path.display()
    );
  }

  // Post-promotion directory fsync — makes the new directory entries
  // (the renamed-in staged files + any unlinked stale extras) durable.
  // Same shape as the post-rename `fsync_dir` `crate::lm::load::save`
  // uses. A failure is a durability-only warning (the entries ARE
  // observable on a running kernel; only a power loss could revert).
  let post_promote_dir = crate::lm::load::fsync_dir(save_path).err();

  Ok(PostPromoteWarnings {
    post_promote_file,
    post_promote_dir,
  })
}

/// Borrow-only worker for [`promote_staging_into_save_path`].
///
/// **F2 contract** (Codex R4 MEDIUM): this helper takes `staging` as a
/// borrowed `&Path` so the parent's [`StagingDir`] RAII guard stays
/// ARMED across every operation here. Any `Err` returned from this
/// function propagates up to the parent, drops the guard, and the
/// `StagingDir::Drop` impl `remove_dir_all`s the staging directory. No
/// cleanup is needed on this side of the call.
///
/// Returns the first post-rename `fsync_path_io` warning (or `None`).
/// The post-promotion `fsync_dir(save_path)` warning is taken in the
/// parent after this returns.
fn promote_staging_inner(staging: &Path, save_path: &Path) -> Result<Option<std::io::Error>> {
  use std::collections::HashSet;

  // Snapshot the staged file names BEFORE we move them — we need the
  // set for the stale-sweep below.
  let mut staged_names: HashSet<String> = HashSet::new();
  let entries = std::fs::read_dir(staging).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "fuse: cannot read staging dir",
      FileOp::Read,
      staging.to_path_buf(),
      e,
    ))
  })?;
  let mut staged_paths: Vec<PathBuf> = Vec::new();
  for entry in entries {
    let entry = entry.map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "fuse: cannot read entry in staging dir",
        FileOp::Read,
        staging.to_path_buf(),
        e,
      ))
    })?;
    let path = entry.path();
    if !path.is_file() {
      // Defensive: `copy_tokenizer_and_extras` only writes regular
      // files, but if anything else ever lands here, skip it — the
      // promotion is over the file family only.
      continue;
    }
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
      continue;
    };
    staged_names.insert(name.to_string());
    staged_paths.push(path);
  }

  // Move each staged file into save_path (overwrite any existing
  // same-name destination via `std::fs::rename`). Same-fs by
  // construction (staging is a subdir of save_path). The per-file
  // `fsync_path_io` AFTER the rename is best-effort: a warning records
  // into `post_promote_file` but does NOT stop the promotion.
  let mut post_promote_file: Option<std::io::Error> = None;

  for staged_path in &staged_paths {
    let Some(name) = staged_path.file_name() else {
      continue;
    };
    let dst = save_path.join(name);
    // Typed-error preservation (R-final+1 Codex finding): preserve the
    // io::ErrorKind through `Error::FileIo` so callers reading
    // `ConvertPostSavePartialPayload::copy_error` can branch on
    // `FileOp::Rename` + `path` + `inner.kind()`. The promote-rename is
    // the moral equivalent of `copy_tokenizer_and_extras`'s std::fs::copy
    // — both feed the same payload via `Err(promote_err)` below.
    std::fs::rename(staged_path, &dst).map_err(|e| {
      crate::Error::FileIo(FileIoPayload::new(
        "fuse: cannot promote staged file to",
        crate::error::FileOp::Rename,
        dst.clone(),
        e,
      ))
    })?;
    if let Err(e) = crate::lm::load::fsync_path_io(&dst) {
      // Wrap with operation + destination-path context (mirrors
      // `copy_tokenizer_and_extras`'s F7 R7 wrap shape). Preserves
      // `ErrorKind` via `std::io::Error::new`. First-failure preserved
      // so the surfaced warning names the EARLIEST file whose fsync
      // dropped durability.
      if post_promote_file.is_none() {
        post_promote_file = Some(std::io::Error::new(
          e.kind(),
          format!("fuse: fsync {} failed: {e}", dst.display()),
        ));
      }
    }
  }

  // **Stale-extras sweep.** For each fixed name in
  // `TOKENIZER_EXTRA_FILES`: if it exists at `save_path` AND the
  // snapshot did NOT carry it → unlink. For each `*.py` at `save_path`:
  // same rule. The walk over the dir does NOT touch any file outside
  // the family.
  //
  // F1 fix: the pre-R3 shape only OVERWROTE files present at the
  // source. A `save_path` pre-populated with e.g. `generation_config.json`
  // / `chat_template.jinja` / `*.py` from an EARLIER model survived the
  // fuse when the new source lacked them, and downstream loaders
  // consumed the stale bytes as wrong-model semantics
  // (`load_config` consumes `generation_config.json` as EOS override;
  // tokenizer surface consumes `tokenizer_config.json` + chat
  // metadata).
  //
  // **F1 R4 fix (Codex R4 HIGH).** Pre-R4, both stale-sweep loops gated
  // removal on `is_file()` which (a) follows symlinks (a symlink whose
  // TARGET is a directory returns `false`) and (b) silently skipped
  // every non-regular entry (directory, FIFO, socket, symlink-to-dir).
  // The skipped entries SURVIVED into `save_path`, then downstream
  // `lm::load::load(save_path)` either failed when it tried to read a
  // dir as a JSON file (Err — opaque error from the loader, not the
  // fuse driver), hung when it tried to read a FIFO, or — worse — read
  // an attacker-planted symlink's target (cross-FS escape via
  // `save_path/tokenizer_config.json` pointing at `/etc/passwd`).
  //
  // The R4 fix uses `symlink_metadata` (NEVER follows symlinks) and:
  // - regular file → `remove_file` (the ordinary stale-extras case)
  // - any other kind (dir, symlink-of-any-target, FIFO, socket, …) →
  //   fail promotion with [`Error::Backend`] naming the offending path
  //   and the kind, instructing the operator to "remove manually or
  //   use a fresh save destination". We deliberately do NOT
  //   `remove_dir_all` a non-regular reserved path — too destructive
  //   for a stale-sweep gated on a basename match; an operator-visible
  //   error is the correct boundary.
  for name in TOKENIZER_EXTRA_FILES {
    if staged_names.contains(*name) {
      continue;
    }
    let candidate = save_path.join(name);
    remove_stale_reserved_path(&candidate, name)?;
  }

  // `*.py` sweep. We walk save_path; for each `*.py`, if NOT in
  // staged_names, unlink. (The staged_names set already contains the
  // copied `*.py` basenames per the snapshot — see
  // `copy_tokenizer_and_extras`'s `*.py` glob.)
  let entries = std::fs::read_dir(save_path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "fuse: cannot read save_path",
      FileOp::Read,
      save_path.to_path_buf(),
      e,
    ))
  })?;
  for entry in entries {
    let entry = entry.map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "fuse: cannot read entry in save_path",
        FileOp::Read,
        save_path.to_path_buf(),
        e,
      ))
    })?;
    let path = entry.path();
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
      continue;
    };
    if !name.ends_with(".py") {
      continue;
    }
    if staged_names.contains(name) {
      continue;
    }
    // F1 R4: route every `*.py` reserved basename through the same
    // symlink_metadata-gated remover as the TOKENIZER_EXTRA_FILES loop
    // above. A `save_path/foo.py/` DIRECTORY (or symlink, or FIFO)
    // that the snapshot did not carry must fail promotion, not be
    // silently skipped by `path.is_file() == false`.
    remove_stale_reserved_path(&path, name)?;
  }

  Ok(post_promote_file)
}

/// Remove a single stale reserved-basename entry at `path`, where
/// `name` is the basename used for diagnostics.
///
/// **F1 R4 contract** (Codex R4 HIGH). The pre-R4 stale-sweep used
/// `path.is_file()` which (1) follows symlinks (a symlink whose target
/// is a directory returns `false`) and (2) silently skipped every
/// non-regular entry. Both behaviors let stale non-regular reserved
/// paths survive into `save_path`, where downstream
/// `lm::load::load(save_path)` failed (read-dir-as-json) or hung (FIFO)
/// or escaped (attacker-planted symlink to `/etc/passwd`).
///
/// The R4 sweep uses [`std::fs::symlink_metadata`] (NEVER follows
/// symlinks) and routes by `file_type`:
/// - **regular file** → [`std::fs::remove_file`] (ordinary stale extra).
/// - **directory** → `Err(Error::Backend)` naming the path + kind. We
///   deliberately do NOT `remove_dir_all` — too destructive for a
///   sweep keyed on a fixed basename family; the operator should
///   resolve the conflict manually (or pick a fresh `save_path`).
/// - **symlink** (regardless of target) → `Err(Error::Backend)`. A
///   symlink at a reserved basename is a security smell (cross-fs
///   escape); same operator-visible boundary as the directory case.
/// - **other** (FIFO, socket, block/char device, …) →
///   `Err(Error::Backend)` with the kind named so the operator can
///   diagnose without re-stat'ing.
/// - **`NotFound`** → no-op (the basename simply isn't there; the
///   normal absent case).
fn remove_stale_reserved_path(path: &Path, name: &str) -> Result<()> {
  let meta = match std::fs::symlink_metadata(path) {
    Ok(m) => m,
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
    Err(e) => {
      return Err(Error::FileIo(FileIoPayload::new(
        "fuse: cannot stat stale destination path",
        FileOp::Stat,
        path.to_path_buf(),
        e,
      )));
    }
  };
  let ft = meta.file_type();
  if ft.is_file() {
    std::fs::remove_file(path).map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "fuse: cannot remove stale destination file",
        FileOp::Remove,
        path.to_path_buf(),
        e,
      ))
    })?;
    return Ok(());
  }
  // Diagnose the offending kind so the operator can act without a
  // re-stat. is_symlink is checked BEFORE is_dir because symlink_metadata
  // never follows symlinks; a symlink-to-dir has is_symlink() == true
  // and is_dir() == false here.
  let kind: &'static str = if ft.is_symlink() {
    "symlink"
  } else if ft.is_dir() {
    "directory"
  } else {
    "non-regular file (FIFO, socket, or device)"
  };
  // `name` is a known static reserved-basename string from the caller
  // (the closed set of fuse-output basenames); embed it in `LayerKeyed.layer`
  // alongside the path for machine inspection.
  Err(Error::LayerKeyed(LayerKeyedPayload::new(
    format!("{} ({})", path.display(), name),
    Error::InvariantViolation(InvariantViolationPayload::new(
      "fuse: stale destination path",
      // `kind` is one of a closed set of static strings; the typed
      // diagnostic carries the path + name via LayerKeyed and the
      // requirement is "regular file or absent" — caller's branch is
      // path + kind via the carried layer.
      match kind {
        "symlink" => {
          "must not be a symlink (non-regular reserved path; remove manually or use a fresh save destination)"
        }
        "directory" => {
          "must not be a directory (non-regular reserved path; remove manually or use a fresh save destination)"
        }
        _ => {
          "must not be a non-regular file (FIFO, socket, or device) (non-regular reserved path; remove manually or use a fresh save destination)"
        }
      },
    )),
  )))
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
  let value: serde_json::Value = serde_json::from_str(config_json)
    .map_err(|e| Error::Parse(ParsePayload::new("fuse: source config", "JSON", e)))?;
  let serde_json::Value::Object(mut map) = value else {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "fuse: source config JSON",
      "must be an object",
    )));
  };
  map.remove("quantization");
  map.remove("quantization_config");
  let stripped = serde_json::Value::Object(map);
  serde_json::to_string(&stripped).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "fuse: cannot re-serialize stripped config",
      "JSON",
      e,
    ))
  })
}

/// Reject a hub-style URL (`hf://...` / `https://huggingface.co/...` /
/// `http://huggingface.co/...`) passed for a local path argument. Mirrors
/// [`crate::audio::load::get_model_path`]'s rejection: strip the URL
/// prefix before interpolating the repo-id into the actionable hint, so
/// the user sees a copy-pasteable `huggingface-cli download <repo_id>`
/// rather than `huggingface-cli download hf://<repo_id>` (broken advice).
fn reject_hub_url(arg_name: &'static str, path: &Path) -> Result<()> {
  let Some(s) = path.to_str() else {
    return Ok(());
  };
  let repo_id = s
    .strip_prefix("hf://")
    .or_else(|| s.strip_prefix("https://huggingface.co/"))
    .or_else(|| s.strip_prefix("http://huggingface.co/"));
  if let Some(_repo_id) = repo_id {
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      s.to_string(),
      Error::InvariantViolation(InvariantViolationPayload::new(
        arg_name,
        "must be a LOCAL path, not a HuggingFace Hub URL (mlxrs is local-only and does \
          not download from the Hub; fetch the model directory out of process — e.g. \
          `huggingface-cli download <repo_id>` or `hf download <repo_id>` — and pass the \
          resulting local path)",
      )),
    )));
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
    let Error::LayerKeyed(payload) = err else {
      panic!("expected Error::LayerKeyed");
    };
    assert_eq!(
      payload.layer(),
      "hf://mlx-community/Qwen3-4B-bf16",
      "carrier layer must be the rejected URL",
    );
    let Error::InvariantViolation(inner) = payload.inner() else {
      panic!("expected inner Error::InvariantViolation");
    };
    assert_eq!(
      inner.context(),
      "model_path",
      "inner context must name the rejected arg"
    );
  }

  #[test]
  fn reject_hub_url_strips_https_prefix_in_hint() {
    let err = reject_hub_url(
      "adapter_path",
      Path::new("https://huggingface.co/owner/repo"),
    )
    .unwrap_err();
    let Error::LayerKeyed(payload) = err else {
      panic!("expected Error::LayerKeyed");
    };
    assert_eq!(
      payload.layer(),
      "https://huggingface.co/owner/repo",
      "carrier layer must be the rejected URL"
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
    let Error::InvariantViolation(p) = err else {
      panic!("expected Error::InvariantViolation, got {err:?}");
    };
    assert_eq!(p.context(), "fuse: source config JSON");
    assert_eq!(p.requirement(), "must be an object");
  }
}
