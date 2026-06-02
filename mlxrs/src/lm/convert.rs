//! `convert()` — the model-conversion driver, ported from
//! [`mlx_lm/convert.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/convert.py).
//!
//! Wires the load-side ([`crate::lm::load`]), the quantize / dequantize side
//! ([`crate::lm::quant`]) and the save-side ([`crate::lm::load::save`] +
//! [`crate::lm::load::save_model`] + [`crate::lm::load::save_config`]) into a
//! one-call pipeline: read an HF-style checkpoint at `hf_path`, optionally
//! apply a quantization (with an optional per-layer predicate) or its
//! inverse, and write the result to `mlx_path` — exactly mirroring
//! `mlx_lm/convert.py::convert`.
//!
//! ## Pipeline (mirrors `convert.py:85-175`)
//!
//! ```text
//!   ConvertArgs
//!      │
//!      ▼
//!   validate args                    (convert.py:101-109, 121-127, 146-147)
//!      │   ─ existing destination?
//!      │   ─ quantize && dequantize? (mutually exclusive)
//!      │   ─ upload_repo / revision? (REJECTED — local-only)
//!      ▼
//!   load(hf_path)                    (convert.py:111-118)
//!      → (Config, Weights, Tokenizer, raw_config_json)
//!      │
//!      ▼
//!   resolve dtype, cast              (convert.py:129-144)
//!      │   ─ explicit override OR  config["torch_dtype"] OR  text_config["dtype"]
//!      ▼
//!   branch:
//!      │  quantize  → quantize_weights(weights, …)         (convert.py:149-158)
//!      │             + patch config "quantization" block   (utils.py:813-845)
//!      │  dequantize→ dequantize_weights(weights, cfg)     (convert.py:160-164)
//!      │             + strip "quantization" / "quantization_config"
//!      │  neither   → pass through unchanged
//!      ▼
//!   save(mlx_path, weights, config, per_layer_q)           (convert.py:166-172)
//!      │
//!      ▼
//!   copy_tokenizer_and_extras(hf_path, mlx_path)           (utils.py:944-948)
//!      │
//!      ▼
//!   Ok(())                                                 (no Hub upload — local-only)
//! ```
//!
//! ## Scope decisions (deliberately NOT ported)
//!
//! Mirrors the same fences as [`crate::lm::load`]:
//!
//! - **HuggingFace Hub upload** (`upload_to_hub` / `share.py`, `convert.py:174-175`,
//!   `utils.py:648-714`) — mlxrs is local-path-only. `upload_repo = Some(_)`
//!   returns [`Error::Backend`] with a clear message.
//! - **HuggingFace Hub download** (`hf_repo_to_path` / `_download` and the
//!   `revision` kwarg, `convert.py:94`) — same fence. `revision = Some(_)`
//!   returns [`Error::Backend`].
//! - **CLI / `argparse`** (`configure_parser` / `main` / `__main__`,
//!   `convert.py:178-267`) — application surface, excluded. Callers
//!   construct [`ConvertArgs`] directly.
//! - **`trust_remote_code`** (`convert.py:99`, `utils.py:439-446`) — mlxrs's
//!   tokenizer load surface (#18) carries no equivalent. A no-op in this
//!   port: a planted `trust_remote_code = true` is accepted but ignored
//!   (and the loader applies its own bounded / non-regular-reject discipline
//!   uniformly to every checkpoint regardless).
//! - **Distributed / multi-host pipelines** (`sharded_load`, `pipeline_load`)
//!   — out of scope (same fence as the load side).
//!
//! ## API style
//!
//! The keyword-arg surface
//! (`convert.py:85-100`) becomes the Rust-idiomatic [`ConvertArgs`] struct
//! with [`Default`]; the python `Callable[[str, nn.Module, dict], …]`
//! closure becomes the [`MixedQuantPredicate`] trait. Every public item
//! carries the cited reference line-ref in its doc-comment.

use std::{
  collections::HashMap,
  path::{Path, PathBuf},
};

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    ConvertPostSavePartialPayload, DurabilityWarningPayload, EmptyInputPayload, Error,
    FileIoPayload, FileOp, InvariantViolationPayload, LayerKeyedPayload, ParsePayload, Result,
    UnsupportedDtypePayload,
  },
  lm::{
    load::{self, Weights},
    quant::{self, PerLayerQuantization, QuantMode, Quantization, QuantizationOption},
  },
};

// ─────────────────────────── ConvertArgs ───────────────────────────

/// Driver arguments for [`convert`] — the Rust-idiomatic analogue of
/// `mlx_lm/convert.py::convert`'s keyword-arg surface (`convert.py:85-100`).
///
/// Every field maps 1:1 to the python kwarg of the same lowered-camelCase
/// name; see the per-field cite for the reference line. [`Default`] mirrors
/// python's keyword defaults so the common-case `ConvertArgs { hf_path, mlx_path, .. Default::default() }`
/// invocation matches `convert(hf_path, mlx_path)` byte-for-byte.
///
/// **Local-only fences** (per the module-level "Scope decisions"):
/// `upload_repo` and `revision` are accepted in the struct shape so
/// callers can be source-compatible with the python API, but a non-`None`
/// value at [`convert`]-call time returns [`Error::Backend`] —
/// HuggingFace Hub upload + download are excluded surface.
///
/// **`!Send`-compatible.** [`Array`] is `!Send` /
/// `!Sync`; the trait object [`Box<dyn MixedQuantPredicate>`](MixedQuantPredicate)
/// uses no `Send`/`Sync` bound, so [`ConvertArgs`] is `!Send`-compatible too.
pub struct ConvertArgs {
  /// `hf_path` (`convert.py:86`). Source directory — already-downloaded
  /// HF-style checkpoint (`config.json` + weights + tokenizer files).
  pub hf_path: PathBuf,

  /// `mlx_path` (`convert.py:87`). Destination directory. Per the
  /// reference (`convert.py:105-109`), [`convert`] refuses to overwrite
  /// an existing path — the caller deletes / renames first.
  ///
  /// Python default is `"mlx_model"`; this is left to the caller (no
  /// implicit relative path) since `&Path` is unambiguous.
  pub mlx_path: PathBuf,

  /// `quantize` (`convert.py:88`). Apply quantization. Mutually exclusive
  /// with [`dequantize`](Self::dequantize) (`convert.py:146-147`).
  pub quantize: bool,

  /// `q_group_size` (`convert.py:89`). Elements per quantization group.
  /// Defaults from `quantize_model`'s `defaults_for_mode` table
  /// (`utils.py:800-808`): `affine`→64, `mxfp4`→32, `nvfp4`→16,
  /// `mxfp8`→32. Resolved at [`convert`]-time per the active
  /// [`q_mode`](Self::q_mode).
  pub q_group_size: Option<i32>,

  /// `q_bits` (`convert.py:90`). Bits per weight. Defaults per
  /// [`q_group_size`](Self::q_group_size) per-mode table (4 for affine /
  /// mxfp4 / nvfp4, 8 for mxfp8).
  pub q_bits: Option<i32>,

  /// `q_mode` (`convert.py:91`). The quantization scheme — see
  /// [`QuantMode`]. Default matches python's `"affine"`.
  pub q_mode: QuantMode,

  /// `dtype` (`convert.py:92`). Override the loaded weights' floating
  /// dtype. `None` falls back to `config.json` `torch_dtype` then
  /// `text_config.dtype` (`convert.py:129-132`); a still-`None` dtype is a
  /// no-op (weights are written in their loaded dtype).
  ///
  /// **Supported set.** Only the three values in
  /// `MODEL_CONVERSION_DTYPES` (`convert.py:82`) are honored:
  /// [`Dtype::F16`] / [`Dtype::BF16`] / [`Dtype::F32`]. An explicit
  /// `Some(_)` outside that set ([`Dtype::I32`] / [`Dtype::F64`] /
  /// [`Dtype::Bool`] / [`Dtype::Complex64`] / any other integer /
  /// boolean variant) is an [`Error::Backend`] at [`convert`]-call time
  /// — matching the reference's silent `if dtype in MODEL_CONVERSION_DTYPES`
  /// gate (`convert.py:133`), where any other parsed string falls
  /// through to "no cast" and never casts weights into an unsupported
  /// type. The Rust port surfaces this as a hard error so a caller
  /// passing e.g. [`Dtype::I32`] cannot silently destroy every floating
  /// weight by casting it to a non-floating dtype (the python `or`-arm
  /// fallback chain at `convert.py:129-132` is string-typed and silently
  /// `None` for any unknown spelling; an explicit `mx.<dtype>` enum
  /// value at the python call site would similarly slip past the gate
  /// — mlxrs forecloses the foot-gun).
  ///
  /// `None` (the default) parses the config's `torch_dtype` /
  /// `text_config.dtype` strings exactly per the reference; unknown
  /// strings are still a silent no-cast (the `if dtype in
  /// MODEL_CONVERSION_DTYPES` gate, faithfully ported).
  pub dtype: Option<Dtype>,

  /// `upload_repo` (`convert.py:93`). HuggingFace Hub destination repo.
  /// **REJECTED.** A non-`None` value at [`convert`]-call time returns
  /// [`Error::Backend`] — mlxrs is local-only per the module-level
  /// "Scope decisions".
  pub upload_repo: Option<String>,

  /// `revision` (`convert.py:94`). HuggingFace Hub `git` rev for the
  /// download. **REJECTED.** Same fence as
  /// [`upload_repo`](Self::upload_repo): a non-`None` value returns
  /// [`Error::Backend`].
  pub revision: Option<String>,

  /// `dequantize` (`convert.py:95`). Inverse of
  /// [`quantize`](Self::quantize): reconstruct dense weights from
  /// already-quantized triples. Mutually exclusive with
  /// [`quantize`](Self::quantize) (`convert.py:146-147`).
  pub dequantize: bool,

  /// `quant_predicate` (`convert.py:96-98`). Per-layer override deciding
  /// whether and how each Linear-like layer is quantized. Python passes a
  /// `Callable[(str, nn.Module, dict)]` returning `bool | dict | None`;
  /// the dict form is `{group_size, bits, mode}` (a per-layer
  /// [`Quantization`]). Rust analogue: a [`MixedQuantPredicate`]
  /// trait-object returning [`Option<Quantization>`] (`Some(q)` → use
  /// these params for the layer; `None` → skip).
  ///
  /// Python also accepts a `str` recipe name and routes it through
  /// [`mixed_quant_predicate_builder`](self::mixed_quant_predicate)
  /// (`convert.py:120-127`). In Rust the caller does that explicitly
  /// (build the predicate via [`mixed_quant_predicate`], box it, attach
  /// it here) — no implicit string-to-predicate coercion.
  pub quant_predicate: Option<Box<dyn MixedQuantPredicate>>,

  /// `trust_remote_code` (`convert.py:99`). **No-op** in this port: the
  /// mlxrs tokenizer surface (#18) carries no remote-code execution
  /// path, so a planted `true` is accepted but unused. Kept in the
  /// struct shape so callers can be source-compatible with the python
  /// kwarg.
  pub trust_remote_code: bool,
}

impl Default for ConvertArgs {
  /// `convert(...)` python kwarg defaults: empty paths must be set by
  /// the caller; everything else matches the python signature
  /// (`convert.py:85-100`).
  fn default() -> Self {
    Self {
      hf_path: PathBuf::new(),
      mlx_path: PathBuf::new(),
      quantize: false,
      q_group_size: None,
      q_bits: None,
      q_mode: QuantMode::Affine,
      dtype: None,
      upload_repo: None,
      revision: None,
      dequantize: false,
      quant_predicate: None,
      trust_remote_code: false,
    }
  }
}

// ─────────────────────── MixedQuantPredicate ───────────────────────

/// A per-layer quantization decider, the Rust analogue of python's
/// `Callable[[str, nn.Module], Union[bool, dict]]` (`convert.py:22,49-51`).
///
/// Called for every Linear-like `<layer_path>.weight` key the
/// quantization pass would otherwise apply the global default to. Returns
/// `Some(q)` to use these per-layer params (`{group_size, bits, mode}`),
/// `None` to skip this layer (the python `False` arm of `wrapped_predicate`,
/// `utils.py:823-835`).
///
/// **`!Send`-compatible.** No `Send` / `Sync` bound — both [`Array`] and
/// the trait-objects flowing through it are `!Send`. The trait is also
/// not `Clone` (mirroring python closure semantics).
pub trait MixedQuantPredicate {
  /// Decide quantization for `layer_name` (the layer path with the
  /// `.weight` suffix stripped — the same key mlx-lm's
  /// `class_predicate(path, module)` receives, `utils.py:349-355`).
  ///
  /// `weight` is the layer's dense `.weight` [`Array`] — the per-layer
  /// shape probe the python `wrapped_predicate` does via
  /// `module.weight.shape[-1]` (`utils.py:826`). The predicate may inspect
  /// it (e.g. to gate on dimensions) but MUST NOT eval / clone it.
  fn decide(&self, layer_name: &str, weight: &Array) -> Option<Quantization>;
}

/// Recipe id for [`mixed_quant_predicate`], mirroring python's
/// `QUANT_RECIPES` list (`convert.py:80`). The variant set is closed:
/// adding a new recipe is a deliberate API change (matching python's
/// `if recipe == ...: ... else: raise ValueError`,
/// `convert.py:26-36`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MixedQuantRecipe {
  /// `"mixed_2_6"` — low=2, high=6, mode=affine (`convert.py:26-27`).
  Mixed2_6,
  /// `"mixed_3_4"` — low=3, high=4, mode=affine (`convert.py:28-30`).
  Mixed3_4,
  /// `"mixed_3_6"` — low=3, high=6, mode=affine (`convert.py:31-32`).
  Mixed3_6,
  /// `"mixed_4_6"` — low=4, high=6, mode=affine (`convert.py:33-34`).
  Mixed4_6,
}

impl MixedQuantRecipe {
  /// `(low_bits, high_bits)` — the per-recipe table from
  /// `convert.py:26-34`. `high_bits` defaults to `6` for every recipe
  /// except `mixed_3_4` which overrides to `4` (`convert.py:24,30`).
  fn bits(self) -> (i32, i32) {
    match self {
      MixedQuantRecipe::Mixed2_6 => (2, 6),
      MixedQuantRecipe::Mixed3_4 => (3, 4),
      MixedQuantRecipe::Mixed3_6 => (3, 6),
      MixedQuantRecipe::Mixed4_6 => (4, 6),
    }
  }
}

/// The runtime predicate [`mixed_quant_predicate`] returns — the Rust
/// analogue of python's nested `mixed_quant_predicate` closure
/// (`convert.py:48-77`). Carries the resolved per-recipe `(low, high)`
/// bits, the `group_size`, and the introspected `layer_location` /
/// `num_layers` derived from the source weight map.
///
/// Hidden as `pub` because [`mixed_quant_predicate`] returns it through
/// the [`MixedQuantPredicate`] trait, but the concrete type is kept
/// accessible for callers who want to introspect / re-use it.
#[derive(Debug)]
pub struct DefaultMixedQuantPredicate {
  low_bits: i32,
  high_bits: i32,
  group_size: i32,
  /// Index of the numeric layer-index segment in `down_proj` paths
  /// (`convert.py:42-45`). Computed once from the source weight map at
  /// builder time; resolved against each path's `.split(".")` at
  /// decide-time.
  layer_location: usize,
  /// `len(model.layers)` (`convert.py:46`). For mlxrs (no module tree)
  /// this is `max_idx + 1` over every `down_proj`-bearing layer in the
  /// source weight map.
  num_layers: i32,
}

impl MixedQuantPredicate for DefaultMixedQuantPredicate {
  fn decide(&self, layer_name: &str, _weight: &Array) -> Option<Quantization> {
    // Hand-port of `mixed_quant_predicate` (`convert.py:48-75`).
    // `index = int(path.split(".")[layer_location]) if len > layer_location else 0`
    let index: i32 = layer_name
      .split('.')
      .nth(self.layer_location)
      .and_then(|s| s.parse().ok())
      .unwrap_or(0);

    // `use_more_bits = (index < num_layers // 8) or (index >= 7 * num_layers // 8)
    //   or ((index - num_layers // 8) % 3 == 2)` (`convert.py:61-65`).
    // Python `//` is floor-div; for non-negative `num_layers` it matches
    // Rust integer `/`.
    let q8 = self.num_layers / 8;
    let use_more_bits =
      index < q8 || index >= 7 * self.num_layers / 8 || (index - q8).rem_euclid(3) == 2;

    // `if ("v_proj" in path or "v_a_proj" in path or "v_b_proj" in path) and use_more_bits: high`
    // (`convert.py:66-69`).
    if use_more_bits
      && (layer_name.contains("v_proj")
        || layer_name.contains("v_a_proj")
        || layer_name.contains("v_b_proj"))
    {
      return Some(Quantization {
        group_size: self.group_size,
        bits: self.high_bits,
        mode: QuantMode::Affine,
      });
    }
    // `if "down_proj" in path and use_more_bits: high` (`convert.py:70-71`).
    if use_more_bits && layer_name.contains("down_proj") {
      return Some(Quantization {
        group_size: self.group_size,
        bits: self.high_bits,
        mode: QuantMode::Affine,
      });
    }
    // `if "lm_head" in path: high` (`convert.py:72-73`) — ALWAYS high
    // regardless of use_more_bits.
    if layer_name.contains("lm_head") {
      return Some(Quantization {
        group_size: self.group_size,
        bits: self.high_bits,
        mode: QuantMode::Affine,
      });
    }
    // Otherwise the recipe's low_bits (`convert.py:75`).
    Some(Quantization {
      group_size: self.group_size,
      bits: self.low_bits,
      mode: QuantMode::Affine,
    })
  }
}

/// Build the default mixed-bit quantization predicate, port of
/// `mlx_lm/convert.py::mixed_quant_predicate_builder` (`convert.py:20-77`).
///
/// **Reference signature.** Python's builder takes `(recipe: str, model:
/// nn.Module, group_size: int)`. mlxrs has no model-module tree, so this
/// port takes the source [`Weights`] map (the structural analogue of
/// `model.named_modules()`) and introspects it for the same two things
/// the python builder consults:
///
/// - The numeric layer-index segment position in `down_proj` paths
///   (`convert.py:42-45`). Computed by splitting the first
///   `down_proj`-bearing key on `.` and scanning for the first
///   all-digit segment.
/// - `len(model.layers)` (`convert.py:46`). Computed as the max index
///   observed across every `down_proj`-bearing key in the map, plus one.
///
/// `weights` must carry at least one `down_proj`-bearing key, mirroring
/// the python `if len(down_keys) == 0: raise ValueError(...)` check
/// (`convert.py:39-40`).
///
/// The returned predicate dispatches per the recipe's `(low, high)` bits
/// and the heuristic block at `convert.py:61-75` — see
/// [`DefaultMixedQuantPredicate`] for the line-for-line breakdown.
///
/// Returns an error mirroring python:
///
/// - No `down_proj`-bearing key → `Error::Backend` quoting `convert.py:40`.
///
/// (Recipes are an enum so the python `raise ValueError(f"Invalid quant
/// recipe ...")` arm at `convert.py:36` is replaced by Rust exhaustive
/// match — unrepresentable recipe strings cannot reach this function.)
pub fn mixed_quant_predicate(
  recipe: MixedQuantRecipe,
  weights: &Weights,
  group_size: i32,
) -> Result<DefaultMixedQuantPredicate> {
  let (low_bits, high_bits) = recipe.bits();

  // `down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]`
  // (`convert.py:38`). We have a weight MAP not a module tree, so scan
  // the keys; the `.weight` suffix strip mirrors `class_predicate`'s
  // path semantics (`utils.py:349-355`).
  let mut down_keys: Vec<&str> = weights
    .keys()
    .filter_map(|k| {
      if k.contains("down_proj") {
        // Strip the trailing `.weight` if present so the path matches
        // python's `named_modules()` key (which excludes the parameter
        // name).
        Some(k.strip_suffix(".weight").unwrap_or(k.as_str()))
      } else {
        None
      }
    })
    .collect();

  if down_keys.is_empty() {
    // `convert.py:40` — `raise ValueError("Model does not have expected
    // keys for mixed quant.")`.
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "mixed_quant_predicate: model down_proj keys",
    )));
  }

  // Sort `down_keys` so the `[0]` choice is deterministic regardless of
  // HashMap iteration order — the python builder iterates the dict in
  // python 3.7+ insertion order, which here is undefined.
  down_keys.sort();

  // `for layer_location, k in enumerate(down_keys[0].split(".")):
  //     if k.isdigit(): break` (`convert.py:43-45`). The first numeric
  // segment in the first key wins.
  let first = down_keys[0];
  let layer_location: usize = first
    .split('.')
    .position(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
    .ok_or_else(|| {
      Error::LayerKeyed(LayerKeyedPayload::new(
        first.to_string(),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "mixed_quant_predicate: `down_proj` path",
          "must contain a numeric layer-index segment (mirroring \
            convert.py:43-45's `if k.isdigit(): break`)",
        )),
      ))
    })?;

  // `num_layers = len(model.layers)` (`convert.py:46`). We compute it as
  // `max_idx + 1` over every `down_proj` key — the structural analogue
  // when there's no module tree.
  let mut max_idx: i32 = -1;
  for key in &down_keys {
    if let Some(seg) = key.split('.').nth(layer_location)
      && let Ok(idx) = seg.parse::<i32>()
      && idx > max_idx
    {
      max_idx = idx;
    }
  }
  // At least one down_proj key was found, but if none of them yielded a
  // parseable numeric segment at `layer_location`, fall back to 1
  // (matching python's behavior where `model.layers` is always at least 1
  // for any model that has `down_proj` modules at all).
  let num_layers = if max_idx >= 0 { max_idx + 1 } else { 1 };

  Ok(DefaultMixedQuantPredicate {
    low_bits,
    high_bits,
    group_size,
    layer_location,
    num_layers,
  })
}

// ─────────────────────────── convert ───────────────────────────

/// The model-conversion driver, port of `mlx_lm/convert.py::convert`
/// (`convert.py:85-175`).
///
/// See the [module-level pipeline diagram](self#pipeline-mirrors-convertpy85-175).
/// Each step's reference line-ref is cited inline in the source so a
/// review against `convert.py` can trace the port edge-for-edge.
///
/// ## Returns
///
/// `Ok(())` — the conversion fully succeeded AND is durable on disk:
/// weights + shard index + config + tokenizer extras all landed on
/// disk AND every fsync along the way (save-side parent-directory
/// fsync, post-copy per-file fsync for each tokenizer / `*.py`, and
/// the post-copy destination-directory fsync) reported success.
/// `Ok = durable across power loss` end-to-end.
///
/// Every recoverable failure is an [`Error`] with a meaning that lets the
/// caller decide whether the on-disk destination is usable as-is, needs a
/// follow-up recovery step, or should be treated as failed and removed:
///
/// - [`Error::Backend`] — **pre-save** failure: argument validation
///   (existing destination, mutually-exclusive flags, rejected
///   `upload_repo` / `revision`), load failures (missing / oversized /
///   invalid `config.json` or weights or tokenizer — see
///   [`crate::lm::load::load`]), quantize / dequantize failures (see
///   [`crate::lm::quant`]) or pre-commit save failures (see
///   [`crate::lm::load::save`]). The destination directory is **not
///   committed**: either it was never created (validation/load failures)
///   or [`crate::lm::load::save`]'s atomic index rename never happened
///   (so any pre-commit shard / config tempfiles are still labeled
///   tempfiles and won't be observed by a future
///   [`crate::lm::load::load`]).
///
/// - [`Error::DurabilityWarning`] with `committed: true` — the save is
///   **logically complete**: weights + index + config + the tokenizer
///   extras copy all landed on disk and would be observed by a
///   subsequent [`crate::lm::load::load`]. EXACTLY ONE of the following
///   fsync boundaries returned an error (so a power loss before the FS
///   internally drains could revert the corresponding rename / write):
///   the save-side parent-directory fsync, a post-copy per-file fsync,
///   or the post-copy destination-directory fsync. The caller may
///   proceed (the convert is logically complete; only durability is
///   uncertain). When TWO OR MORE fsync boundaries warn in the same
///   convert, the typed aggregate [`Error::ConvertDurabilityWarnings`]
///   is returned instead (see below).
///
/// - [`Error::ConvertDurabilityWarnings`] with `committed: true` — the
///   save is **logically complete** (same on-disk shape as
///   [`Error::DurabilityWarning`] above) and TWO OR MORE fsync
///   boundaries warned in the same convert. The inner aggregate
///   ([`crate::error::ConvertDurabilityWarnings`]) carries each
///   boundary's [`std::io::Error`] in a separate `Option` field
///   (`save`, `post_copy_file`, `post_copy_dir`) so the caller can
///   machine-detect WHICH boundaries warned via direct destructuring
///   (no string parse). The [`std::error::Error::source`] chain points
///   at the first non-`None` warning in
///   `save -> post_copy_file -> post_copy_dir` priority order — the
///   most-actionable for a chain walker.
///
/// - [`Error::ConvertPostSavePartial`] with `committed: true` — the save
///   committed (weights + index + config on disk) but the post-save
///   [`copy_tokenizer_and_extras`] step's [`std::fs::copy`] ACTUALLY
///   FAILED for at least one tokenizer / `*.py` / `generation_config.json`
///   file (the file did NOT reach disk — distinct from a post-copy
///   fsync warning, which surfaces as [`Error::DurabilityWarning`]
///   above). The on-disk destination is **structurally incomplete**
///   and a downstream [`crate::lm::load::load`] would either fail
///   (missing tokenizer.json) or silently produce a checkpoint with
///   the wrong tokenizer. The caller MUST decide whether to retry the
///   copy, copy the missing files by hand, or treat the whole convert
///   as failed and delete the destination.
///
///   The variant's typed fields (`save_warning: Option<io::Error>`,
///   `copy_error: io::Error`) carry the two failure signals
///   separately — `save_warning = Some(_)` means the save side ALSO
///   raised a [`Error::DurabilityWarning`] (committed + fsync warning);
///   `save_warning = None` means the save was clean and only the
///   extras-copy failed. Both shapes leave the destination structurally
///   incomplete, so both surface the same variant for the caller's
///   uniform "incomplete-destination" recovery path.
pub fn convert(args: ConvertArgs) -> Result<()> {
  let ConvertArgs {
    hf_path,
    mlx_path,
    quantize,
    q_group_size,
    q_bits,
    q_mode,
    dtype,
    upload_repo,
    revision,
    dequantize,
    quant_predicate,
    trust_remote_code: _, // no-op per module docs
  } = args;

  // ─── 1. Validate args ───
  //
  // `convert.py:105-109` — `if mlx_path.exists(): raise ValueError(...)`.
  // The reference does this FIRST (before load) so a doomed convert
  // doesn't waste a load on a destination it can't write. Symlink-to-
  // anywhere counts as "exists" too (matches python's
  // `pathlib.Path.exists()` — follows symlinks).
  if mlx_path.exists() {
    return Err(Error::FileIo(FileIoPayload::new(
      "convert: destination must not already exist (delete the file/directory or specify \
        a new path to save to)",
      FileOp::Stat,
      mlx_path,
      std::io::Error::from(std::io::ErrorKind::AlreadyExists),
    )));
  }

  // mlxrs is local-only; the python Hub upload / download surface is
  // out of scope per the module-level "Scope decisions". Reject AFTER
  // the `exists()` check so destination validation always runs first
  // (mirrors the reference's check order at `convert.py:101-109`).
  if upload_repo.is_some() {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "convert: `upload_repo`",
      "must be None in mlxrs (HuggingFace Hub upload is out of scope — mlxrs is \
        local-path-only; drop the kwarg or upload the result directory yourself)",
    )));
  }
  if revision.is_some() {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "convert: `revision`",
      "must be None in mlxrs (HuggingFace Hub download is out of scope — mlxrs is \
        local-path-only; download the checkpoint yourself and pass its local path as `hf_path`)",
    )));
  }

  // `convert.py:146-147` — `if quantize and dequantize: raise ValueError(...)`.
  if quantize && dequantize {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "convert: `quantize` && `dequantize` flags",
      "must not both be true — choose either quantize or dequantize, not both \
        (convert.py:146-147)",
    )));
  }

  // ─── 2. Load (`convert.py:111-118`) ───
  //
  // Python: `model, tokenizer, config = load(hf_path, revision=..., return_config=True,
  //   tokenizer_config={"trust_remote_code": ...}, lazy=True)`.
  // The mlxrs equivalent returns the raw `config.json` body alongside the
  // typed [`Config`] (`load_config`), so we can mutate it (cast keys / strip
  // quantization block) and round-trip it through `save_config` which
  // handles the python `_name_or_path` / `vision_config` cleanup +
  // `quantization`→`quantization_config` mirror itself.
  let (cfg_typed, config_json_text) = load::load_config(&hf_path)?;
  let weights = load::load_weights(&hf_path)?;
  // Tokenizer is loaded for the side-effect of validating it exists +
  // is parseable. The actual on-disk tokenizer files are copied by
  // `copy_tokenizer_and_extras` after `save` — mirroring python's
  // `tokenizer.save_pretrained` + the explicit `*.py` /
  // `generation_config.json` copy.
  let _tokenizer = load::load_tokenizer(&hf_path, &cfg_typed)?;

  // ─── 3. Resolve dtype + cast (`convert.py:129-144`) ───
  //
  // Python:
  //   if dtype is None:
  //       dtype = config.get("torch_dtype", None)
  //   if dtype is None and (text_config := config.get("text_config", None)):
  //       dtype = text_config.get("dtype", None)
  //   if dtype in MODEL_CONVERSION_DTYPES:
  //       dtype = getattr(mx, dtype)
  //       cast_predicate = getattr(model, "cast_predicate", lambda _: True)
  //       def set_dtype(k, v):
  //           if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
  //               return v.astype(dtype)
  //           else:
  //               return v
  //       model.update(tree_map_with_path(set_dtype, model.parameters()))
  //
  // mlxrs port: resolve the dtype, then cast every floating weight in-
  // place to it. `cast_predicate` defaults to "always true" (the python
  // `getattr(...,  lambda _: True)`); we have no nn.Module to consult
  // for an architecture-specific override, so the always-true default is
  // the only branch (mirroring the python default arm exactly).
  let resolved_dtype = resolve_target_dtype(dtype, &config_json_text)?;
  let weights = if let Some(d) = resolved_dtype {
    cast_floats_to_dtype(weights, d)?
  } else {
    weights
  };

  // ─── 4. Quantize / dequantize / pass-through ───

  // Determine the [`PerLayerQuantization`] to pass into the save path.
  // It carries the global+per-layer quantization config the save side
  // needs for `get_total_parameters` / `compute_bits_per_weight`. The
  // default (`PerLayerQuantization::default()`) is the "no quantization"
  // pass-through case.
  let (out_weights, out_config_json, per_layer_cfg) = if quantize {
    // `convert.py:149-158` — `quantize_model(model, config, q_group_size,
    //   q_bits, mode=q_mode, quant_predicate=quant_predicate)`.
    let (gs, bits) = defaults_for_mode(q_mode, q_group_size, q_bits);

    // Single-evaluation closure: evaluate the predicate ONCE per
    // structurally-eligible layer into a decision map BEFORE walking
    // either the config builder OR the `quantize_weights` eligibility
    // closure. The python reference's `wrapped_predicate` is called
    // exactly once per module by `nn.quantize` (`utils.py:837-843`),
    // and its single return value flows BOTH into `quantized_config`
    // (`utils.py:831-834`) AND back to `nn.quantize`'s decision —
    // a stateful predicate is therefore consistent across both views.
    //
    // Evaluating the predicate twice (once in `build_quantize_config`,
    // again in the `eligible` closure) would let a stateful /
    // non-deterministic predicate write one decision into the saved
    // config and apply a different one to the weights. Caching the
    // per-eligible-layer decision in a `HashMap` collapses both reads to
    // the same source-of-truth, matching the reference's
    // call-the-callable-once semantics.
    let decisions = build_predicate_decisions(quant_predicate.as_deref(), &weights, gs);

    let (cfg, cfg_json) =
      build_quantize_config(&config_json_text, gs, bits, q_mode, &decisions, &weights)?;
    let eligible = |path: &str, _weight: &Array| -> bool {
      // The "structural analogue of mlx-lm's `hasattr(module,
      // 'to_quantized')`" predicate. When a predicate is supplied, the
      // pre-computed [`PredicateDecisions`] map is the single source of
      // truth: `Some(Some(_))` ⇒ quantize, `Some(None)` ⇒ predicate
      // explicitly skipped, `None` ⇒ layer never reached the predicate
      // (structurally ineligible — same arms `build_predicate_decisions`
      // filtered out, so `quantize_weights`'s downstream shape gate
      // would skip too). The match collapses to "do we have a Some
      // decision for this path?".
      //
      // No user predicate? Then every layer that passes the downstream
      // shape gates is eligible (the python `quant_predicate=None` arm
      // of `wrapped_predicate` defaults `bool_or_params=True`,
      // `utils.py:828`).
      match (quant_predicate.is_some(), decisions.get(path)) {
        // Predicate supplied + a `Some(q)` decision on file → quantize.
        (true, Some(Some(_))) => true,
        // Predicate supplied + an explicit `None` skip on file → skip.
        (true, Some(None)) => false,
        // Predicate supplied + this path never reached the predicate
        // (structurally ineligible). Fall through to the downstream
        // shape gate (which will skip too) by returning false here.
        (true, None) => false,
        // No predicate at all → every eligible layer goes through.
        (false, _) => true,
      }
    };
    let w = quant::quantize_weights(weights, &cfg, &eligible)?;
    (w, cfg_json, cfg)
  } else if dequantize {
    // `convert.py:160-164` — `config.pop("quantization", None);
    //   config.pop("quantization_config", None); model = dequantize_model(model)`.
    // Use the source config's quantization block to resolve per-layer
    // params for the dequantize call (the python `dequantize_model`
    // reads from module attrs, which were populated from the same
    // `config["quantization"]` at load-time). After the strip, the saved
    // config carries no quantization block at all — `save_config`'s
    // mirror is then a no-op.
    let cfg = quant::parse_quantization(&config_json_text)?.unwrap_or_default();
    let stripped = strip_quantization_blocks(&config_json_text)?;
    let w = quant::dequantize_weights(weights, &cfg)?;
    (w, stripped, PerLayerQuantization::default())
  } else {
    // Pass-through: no quantization params change, no weight change.
    // The source config may already carry a `quantization` block (if
    // converting an already-quantized checkpoint pass-through) so
    // parse + carry it forward so `save_model`'s
    // `get_total_parameters` sees the right per-layer block.
    let cfg = quant::parse_quantization(&config_json_text)?.unwrap_or_default();
    (weights, config_json_text, cfg)
  };

  // ─── 5. Save (`convert.py:166-172`) ───
  //
  // `save(mlx_path, hf_path, model, tokenizer, config)`. mlxrs's `save`
  // doesn't carry the tokenizer (no `tokenizer.save_pretrained` is
  // ported — the tokenizer surface is load-only) or the source-`*.py` /
  // `generation_config.json` copy: those are this driver's
  // `copy_tokenizer_and_extras` step below.
  //
  // A [`Error::DurabilityWarning`] with `committed:
  // true` from [`load::save`] is NOT a hard failure — the weights +
  // index + config are already visible on disk (only the post-rename
  // parent-directory `fsync` returned an error). A plain `?` early-
  // return would skip the tokenizer copy, leaving a destination that
  // PASSES the [`mlx_path.exists()`] gate of any future
  // [`convert`] retry while MISSING tokenizer files — a non-fatal
  // durability warning would become a partial, hard-to-recover
  // conversion. Match the warning explicitly: stash the underlying
  // error, continue with the tokenizer / extras copy (so the on-disk
  // dir is COMPLETE), then re-raise the warning to the caller.
  let committed_warning: Option<std::io::Error> =
    match load::save(&mlx_path, &out_weights, &out_config_json, &per_layer_cfg) {
      Ok(()) => None,
      Err(Error::DurabilityWarning(p)) if p.committed() => Some(p.into_source()),
      Err(other) => return Err(other),
    };

  // ─── 6. Copy tokenizer + extras (the deliberately-deferred portion
  //         of `utils.save`, `utils.py:944-948`) ───
  //
  // Runs unconditionally on a committed save — including the committed-
  // durability-warning branch — so the destination dir is fully
  // populated before we propagate the warning to the caller.
  //
  // A post-save copy failure must NOT be folded into the
  // DurabilityWarning's `source` (e.g. via [`std::io::Error::other(format!
  // (...))`]) because that conflates two semantically-different cases:
  //
  //   (a) save committed + only durability uncertain (the documented
  //       [`Error::DurabilityWarning`] contract = logically-complete
  //       checkpoint), and
  //   (b) save committed + extras copy partially failed (destination
  //       MAY be incomplete — tokenizer files missing).
  //
  // Such a fold would also HIDE the copy failure inside a free-form
  // `source.to_string()` — callers couldn't machine-detect it via
  // `ErrorKind` / typed accessors. Callers treating
  // `committed=true DurabilityWarning` as "success-with-warning" per
  // the documented contract could then consume an INCOMPLETE checkpoint.
  //
  // So (b) (and the symmetric clean-save + copy-fail
  // case) routes to the structured variant [`Error::ConvertPostSavePartial`]
  // so the two cases are machine-detectable at the type level: the
  // typed `save_warning: Option<io::Error>` field disambiguates the
  // save side, the typed `copy_error: io::Error` field carries the
  // actually-actionable failure, and the variant's distinct
  // [`std::mem::discriminant`] tells the caller the destination is
  // structurally incomplete (tokenizer files missing) — NOT merely
  // committed-with-fsync-warning.
  //
  // [`copy_tokenizer_and_extras`] fsyncs each
  // copied file + the dst dir, so the documented "Ok = durable"
  // contract holds for tokenizer extras too. The fsync step
  // distinguishes a POST-COPY fsync warning (data on disk, durability
  // uncertain — same shape as the save-side fsync warning) from a
  // hard copy failure (file did NOT reach disk). The return type is
  // now [`CopyOutcome`], whose [`CopyOutcome::CommittedWithDurabilityWarning`]
  // arm carries the post-copy fsync warning so it can be folded into
  // the existing [`Error::DurabilityWarning`] variant (rather than
  // [`Error::ConvertPostSavePartial`], which stays reserved for the
  // genuine "copy actually failed" case).
  let copy_result = copy_tokenizer_and_extras(&hf_path, &mlx_path);

  // ─── 7. (Hub upload — `convert.py:174-175`) — REJECTED at step 1. ───

  // Assemble per-boundary durability warnings into a
  // single typed aggregate so the caller can machine-detect WHICH
  // boundary(ies) warned via direct destructuring — no string parse
  // (folding multi-warnings into a free-form
  // `std::io::Error::other(format!(...))` message inside
  // `DurabilityWarning.source` would hide the typed errors). The
  // 0/1/2+-non-None counting routes the result through the right
  // shape:
  //   - 0 non-None warnings → `Ok(())` (happy path).
  //   - 1 non-None warning  → `Err(DurabilityWarning { source })`
  //                           (single-warning shape unchanged — the
  //                           existing one-source contract).
  //   - 2+ non-None warnings → `Err(ConvertDurabilityWarnings { ... })`
  //                           (new typed aggregate; each field is
  //                           reachable via destructuring and the
  //                           first non-None is reachable via
  //                           `std::error::Error::source()` for
  //                           chain walkers).
  // A hard copy failure (`copy_result == Err`) bypasses the aggregate
  // and routes to `ConvertPostSavePartial` (the structurally-
  // incomplete-destination contract) — orthogonal to the
  // durability-only multi-warning case below.
  match (committed_warning, copy_result) {
    // copy_result == Ok: 0 / 1 / 2+ durability-warning routing.
    (save, Ok(copy_outcome)) => {
      let copy_warnings = match copy_outcome {
        CopyOutcome::Committed => CopyDurabilityWarnings::default(),
        CopyOutcome::CommittedWithDurabilityWarning(w) => w,
      };
      let aggregate = crate::error::ConvertDurabilityWarnings {
        committed: true,
        save,
        post_copy_file: copy_warnings.post_copy_file,
        post_copy_dir: copy_warnings.post_copy_dir,
      };
      match aggregate.count() {
        // 0 — happy path: weights + index + config + tokenizer
        // extras all landed and EVERY fsync (save-side parent-dir,
        // post-copy per-file, post-copy dst-dir) returned success.
        // `Ok = durable` contract holds end-to-end.
        0 => Ok(()),
        // 1 — single fsync boundary warned (could be the save-side
        // parent-dir, the post-copy per-file, or the post-copy
        // dst-dir). Same "logically committed, durability uncertain"
        // shape as before — surface via the existing
        // [`Error::DurabilityWarning`] so the single-source contract
        // is unchanged. The unwrap is safe: `count() == 1` guarantees
        // exactly one non-None field and `first_warning()` returns
        // that field in priority order.
        //
        // We MOVE the underlying io::Error out of `aggregate` (rather
        // than clone — `io::Error` is not Clone). The destructure
        // pattern below is exhaustive over `aggregate`'s shape so
        // every field is named even when only one is `Some`.
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
        // 2+ — multi-warning case: surface the
        // typed aggregate so the caller can destructure each
        // boundary's [`std::io::Error`] separately (no string fold).
        // The first non-None is also reachable via
        // [`std::error::Error::source()`] (priority:
        // save -> post_copy_file -> post_copy_dir).
        _ => Err(Error::ConvertDurabilityWarnings(aggregate)),
      }
    }

    // (None, Err) — save was clean but the copy step's
    // [`std::fs::copy`] returned `Err` for a file (it did NOT reach
    // disk) → the destination dir is structurally incomplete. Surface
    // the structured [`Error::ConvertPostSavePartial`] variant with
    // `save_warning: None` so the caller can machine-detect the
    // incomplete-destination contract. The save IS committed (the
    // index rename succeeded before we even reached the copy step) —
    // weights + index + config are on disk; only the extras are
    // missing.
    (None, Err(copy_err)) => Err(Error::ConvertPostSavePartial(
      ConvertPostSavePartialPayload::new(true, None, copy_err),
    )),

    // (Some, Err) — save raised a DurabilityWarning AND the copy
    // step's [`std::fs::copy`] failed for a file → both signals
    // matter. Surface the structured variant with the save warning
    // in `save_warning` and the actual copy failure in `copy_error`.
    // Both stay machine-readable; the variant tells the caller the
    // destination is structurally incomplete (not just fsync-warned).
    // `copy_err` is passed through as the typed `crate::Error` so
    // `Error::FileIo(FileIoPayload { .. })` survives intact for recovery
    // code (no `io::Error::other(...)` stringification).
    (Some(save_source), Err(copy_err)) => Err(Error::ConvertPostSavePartial(
      ConvertPostSavePartialPayload::new(true, Some(save_source), copy_err),
    )),
  }
}

// ─────────────────────── helpers ───────────────────────

/// `defaults_for_mode` (`utils.py:800-808`) — per-mode `(group_size,
/// bits)` fallbacks when the kwarg is `None`. mlx-lm's hard-coded table.
///
/// **Zero is falsy** (`utils.py:808`): python evaluates `group_size or
/// default_group_size, bits or default_bits` — `0` triggers the `or`-arm
/// fallback because it's falsy. The Rust port mirrors that: `Some(0)`
/// MUST fall back to the per-mode default, not survive as `0`. A
/// surviving `0` would skip every layer at the `last % group_size == 0`
/// gate (`quantize_weights` would write dense weights against an invalid
/// `group_size: 0` quantization block on disk).
///
/// We mirror with `Option::filter(|&v| v > 0).unwrap_or(default)` —
/// faithful one-line transcription of `value or default` for the
/// positive-integer arithmetic the table demands. Negative values would
/// also be invalid (mlx's `quantize` asserts positive `group_size` /
/// `bits`); the filter rejects those too so a `Some(-1)` doesn't
/// pretend the caller actually wanted `-1`.
pub(crate) fn defaults_for_mode(mode: QuantMode, gs: Option<i32>, bits: Option<i32>) -> (i32, i32) {
  let (default_gs, default_bits) = match mode {
    QuantMode::Affine => (64, 4),
    QuantMode::Mxfp4 => (32, 4),
    QuantMode::Nvfp4 => (16, 4),
    QuantMode::Mxfp8 => (32, 8),
  };
  // `Some(v).filter(|&v| v > 0)` is the Rust transcription of python's
  // `v or default` truthiness (where `0`/`-1` are falsy for positive
  // integer arithmetic) — see fn-doc.
  (
    gs.filter(|&v| v > 0).unwrap_or(default_gs),
    bits.filter(|&v| v > 0).unwrap_or(default_bits),
  )
}

/// Resolve the target floating dtype the cast step should use, mirroring
/// the python fallback chain at `convert.py:129-133`:
///   1. explicit kwarg (`Some(d)` — gated to the supported set)
///   2. `config.json` `torch_dtype` (string)
///   3. `config.json` `text_config.dtype` (string; the VLM-config
///      fallback)
///   4. `None` (no cast)
///
/// Only the three [`MODEL_CONVERSION_DTYPES`] (`convert.py:82`) are
/// honored — any other parsed string falls through to `None` (no cast),
/// matching python's silent `if dtype in MODEL_CONVERSION_DTYPES` gate.
///
/// **Explicit-kwarg gate** (this fn's `explicit` arg): the reference's
/// gate is string-typed and silently falls through to "no cast" for any
/// unknown spelling. mlxrs's [`Dtype`] is an enum that includes integer /
/// boolean / complex variants that the reference's `if dtype in
/// MODEL_CONVERSION_DTYPES` gate would silently drop — a caller passing
/// e.g. [`Dtype::I32`] would NEVER cast in python but WOULD destroy
/// every floating weight in Rust. The port forecloses the foot-gun by
/// surfacing an [`Error::Backend`] for any explicit `Some(d)` outside
/// the supported set, matching the reference's effective semantics (no
/// unsupported cast) while telling the caller why.
fn resolve_target_dtype(explicit: Option<Dtype>, config_json: &str) -> Result<Option<Dtype>> {
  if let Some(d) = explicit {
    // Gate the explicit override to the same set the reference resolves
    // via `getattr(mx, dtype)` from `MODEL_CONVERSION_DTYPES`
    // (`convert.py:82`, `convert.py:133-135`). Anything else is a hard
    // error — see the [`ConvertArgs::dtype`] field-doc for the why.
    return match d {
      Dtype::F16 | Dtype::BF16 | Dtype::F32 => Ok(Some(d)),
      other => Err(Error::UnsupportedDtype(UnsupportedDtypePayload::new(
        "convert: `dtype` (matches mlx_lm/convert.py:82 supported set MODEL_CONVERSION_DTYPES)",
        other,
        &[Dtype::F16, Dtype::BF16, Dtype::F32],
      ))),
    };
  }
  let parsed: serde_json::Value = match serde_json::from_str(config_json) {
    Ok(v) => v,
    Err(_) => return Ok(None), // config parsed once already; this path is unreachable in practice
  };
  if let Some(s) = parsed.get("torch_dtype").and_then(|v| v.as_str())
    && let Some(d) = parse_conversion_dtype(s)
  {
    return Ok(Some(d));
  }
  if let Some(text_cfg) = parsed.get("text_config")
    && let Some(s) = text_cfg.get("dtype").and_then(|v| v.as_str())
    && let Some(d) = parse_conversion_dtype(s)
  {
    return Ok(Some(d));
  }
  Ok(None)
}

/// `MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]`
/// (`convert.py:82`). The python `getattr(mx, dtype)` only resolves to a
/// known mlx dtype when the string is in this list; any other value is a
/// silent no-op.
fn parse_conversion_dtype(s: &str) -> Option<Dtype> {
  match s {
    "float16" => Some(Dtype::F16),
    "bfloat16" => Some(Dtype::BF16),
    "float32" => Some(Dtype::F32),
    _ => None,
  }
}

/// `if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
///     return v.astype(dtype)` (`convert.py:138-142`).
///
/// Cast every floating-dtype [`Array`] in `weights` to `target`, leaving
/// non-floating arrays (quantized triples' `uint32` packed `.weight` and
/// integer indices) unchanged. The `cast_predicate` defaults to the
/// always-true python lambda; mlxrs has no module to consult for an
/// override, so the always-true arm is the only one (per the module-level
/// "cast_predicate" decision).
fn cast_floats_to_dtype(weights: Weights, target: Dtype) -> Result<Weights> {
  let mut out: Weights = HashMap::with_capacity(weights.len());
  for (k, arr) in weights {
    let dt = arr.dtype()?;
    // mlx's `issubdtype(v.dtype, mx.floating)` — only the IEEE-754 +
    // bfloat16 dtypes (mlx-c's `mlx_dtype` floating set). Integer +
    // bool + complex pass through.
    let is_floating = matches!(dt, Dtype::F16 | Dtype::F32 | Dtype::BF16);
    if is_floating && dt != target {
      out.insert(k, arr.astype(target)?);
    } else {
      out.insert(k, arr);
    }
  }
  Ok(out)
}

/// Per-eligible-layer cached predicate decision — the
/// "evaluate the predicate exactly once per layer" data structure.
///
/// A path appears in the map iff it cleared the python `wrapped_predicate`'s
/// structural gates (`hasattr(module, 'to_quantized')` + last-axis-divisible-
/// by-group-size, `utils.py:824-827`). The value records the predicate's
/// single return — `Some(Quantization)` ⇒ python's dict-arm
/// (`utils.py:831-832`); `None` ⇒ python's `False` arm
/// (`utils.py:823-835` falls through to `return False`).
///
/// Both `build_quantize_config` (the saved-config writer) and the
/// `eligible` closure (the runtime weights-quantization gate) read from
/// this map. The predicate itself is invoked ONCE per path inside
/// [`build_predicate_decisions`]; downstream call sites never re-invoke
/// it. This mirrors python's `nn.quantize` calling `wrapped_predicate`
/// exactly once per module (`utils.py:837-843`) — a stateful or
/// non-deterministic predicate yields one consistent decision per
/// layer.
type PredicateDecisions = HashMap<String, Option<Quantization>>;

/// Walk `weights`, run the structural eligibility gate
/// (`utils.py:824-827`) per layer, and call the predicate exactly once
/// for each surviving path. The returned map keys the eligible paths
/// onto the predicate's single decision.
///
/// When `predicate` is `None`, the returned map is empty — the
/// downstream `eligible` closure short-circuits to "every layer is
/// eligible" in that case (matching the python `quant_predicate=None`
/// arm's `bool_or_params = True` default at `utils.py:828`).
///
/// `group_size <= 0` is a defensive guard — `defaults_for_mode`
/// clamps `Some(0)` to the mode default, so a 0 here
/// would mean the global default itself is 0 (which the config
/// builder's `if last % (group_size as usize) != 0` guard would catch
/// anyway). We early-return an empty map in that case so the divisor
/// is never used and the python `if module.weight.shape[-1] %
/// group_size != 0` arm's behavior (skip everything) is preserved.
fn build_predicate_decisions(
  predicate: Option<&dyn MixedQuantPredicate>,
  weights: &Weights,
  group_size: i32,
) -> PredicateDecisions {
  let mut decisions: PredicateDecisions = HashMap::new();
  let Some(pred) = predicate else {
    return decisions;
  };
  if group_size <= 0 {
    return decisions;
  }
  let gs_usize = group_size as usize;
  for (key, arr) in weights {
    let Some(path) = key.strip_suffix(".weight") else {
      continue;
    };
    // mlx-lm `class_predicate` only fires for layers that pass the
    // structural shape gate (`weight.shape[-1] % group_size == 0`,
    // `utils.py:826-827`). Mirror so a predicate returning `Some(q)`
    // for an ineligible layer doesn't end up in the saved config (and
    // is never asked about — matching `nn.quantize`'s single-call
    // semantics).
    let shape = arr.shape();
    if shape.len() < 2 {
      continue;
    }
    let last = *shape.last().expect("rank>=2");
    if last % gs_usize != 0 {
      continue;
    }
    // The single predicate invocation per eligible layer. Cached into
    // the decision map for both downstream consumers (config builder
    // + `eligible` closure) — neither re-invokes the predicate.
    let decision = pred.decide(path, arr);
    decisions.insert(path.to_string(), decision);
  }
  decisions
}

/// Build the [`PerLayerQuantization`] config the quantization pass will
/// honor + emit the saved `config.json` text with the right
/// `"quantization"` block in place. Mirrors `quantize_model`'s config
/// mutation at `utils.py:810-845`:
///
/// - If the source config already carries a `quantization` block, treat
///   the call as "fine-grained" — every per-layer predicate decision
///   that returns params is written as a per-layer override
///   (`utils.py:832`), every "use defaults" decision (truthy bool in
///   python) also writes an override (`utils.py:833-834`).
/// - Otherwise install the global `{group_size, bits, mode}` block
///   (`utils.py:821`) and only per-layer-DICT decisions add explicit
///   overrides.
///
/// `quantization_config` is mirrored to `quantization` per `utils.py:845`
/// — `save_config` already does that mirror at write-time, so the
/// returned config text carries only the `quantization` key (the mirror
/// happens inside `save_config`).
///
/// **Predicate decisions are pre-computed.** The `decisions` arg is the
/// single-evaluation map (see
/// [`build_predicate_decisions`]); the builder NEVER re-invokes the
/// predicate. An empty map means "no user predicate / nothing eligible
/// for an override" — the global block alone is written. The `weights`
/// arg is still threaded through for the bookkeeping pass that iterates
/// `.weight`-keys; it does NOT trigger any predicate call.
fn build_quantize_config(
  config_json: &str,
  group_size: i32,
  bits: i32,
  mode: QuantMode,
  decisions: &PredicateDecisions,
  weights: &Weights,
) -> Result<(PerLayerQuantization, String)> {
  let value: serde_json::Value = serde_json::from_str(config_json)
    .map_err(|e| Error::Parse(ParsePayload::new("convert: source config", "JSON", e)))?;
  let serde_json::Value::Object(mut map) = value else {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "convert: source config JSON",
      "must be an object",
    )));
  };

  // Python's `fine_grained_config` flag (`utils.py:815-820`).
  let fine_grained = map.contains_key("quantization");

  // Build the live [`PerLayerQuantization`] the `quantize_weights` call
  // will read for per-layer (and global) params. The global default is
  // always the call's (group_size, bits, mode); per-layer overrides
  // come from the pre-computed predicate decision map.
  let global = Quantization {
    group_size,
    bits,
    mode,
  };
  let mut per_layer_overrides: HashMap<String, QuantizationOption> = HashMap::new();

  // Walk every `.weight`-bearing weight key; for each, read the cached
  // decision. The python `wrapped_predicate(path, module)` runs at
  // `nn.quantize(...)` time over every Linear/Embedding/SwitchLinear
  // module ONCE; we cached that single return in `decisions`. The
  // decision's two arms (`Some(q)` / `None`) map directly onto the
  // python dict-arm and `False`-arm of `wrapped_predicate`. We also
  // need to walk `weights` here (not just iterate `decisions`) so the
  // ineligible-but-fine-grained "use defaults under this path" arm
  // (`utils.py:833-834`) can fire — though in the current shape
  // `decisions` only contains the eligible set, so the iteration
  // remains a faithful one-pass.
  if !decisions.is_empty() {
    for key in weights.keys() {
      let Some(path) = key.strip_suffix(".weight") else {
        continue;
      };
      // Look up the cached decision. `None` here means the layer never
      // passed the structural gate inside `build_predicate_decisions`
      // (or no predicate was supplied) — nothing to write.
      let Some(decision) = decisions.get(path) else {
        continue;
      };
      match decision {
        Some(q) => {
          // The python `wrapped_predicate`'s dict-arm
          // (`utils.py:831-832`) writes the per-layer dict; the
          // truthy-bool arm (`utils.py:833-834`) writes the global
          // defaults under the layer key when `fine_grained_config`
          // is on. Both arms emit a `Quantize(q)` override for us
          // (rust's `Some(q)` collapses both). For the global-default
          // case we only emit an override when fine_grained (matching
          // python's `elif fine_grained_config and bool_or_params:
          // ...`).
          if *q != global || fine_grained {
            per_layer_overrides.insert(path.to_string(), QuantizationOption::Quantize(*q));
          }
        }
        None => {
          // Python returns `False` from the predicate → the layer is
          // simply NOT in `class_predicate`'s accept set, so
          // `nn.quantize` doesn't visit it. mlx-lm writes nothing into
          // `quantized_config["quantization"][path]` for skipped layers
          // (only the dict / truthy-bool arms write; the `False` arm
          // falls through `wrapped_predicate` returning `False`). The
          // result is that a skipped layer stays dense AND there's no
          // per-layer config entry for it on save.
          //
          // For mlxrs that means: in the per-layer-OVERRIDE map we
          // record a `Skip` ONLY when fine_grained — so a later load
          // honors the explicit "don't quantize this" decision even
          // though the layer's `.weight` is dense (matches python's
          // behavior: it's a no-op for the writer either way, but a
          // future read picks up the override).
          if fine_grained {
            per_layer_overrides.insert(path.to_string(), QuantizationOption::Skip);
          }
        }
      }
    }
  }

  // Write the `"quantization"` block back into the in-memory config JSON
  // so `save_config` (in `save`) emits it. The shape mirrors mlx's
  // on-disk schema (`BaseConfiguration.swift:103-118`): global keys
  // (`group_size`, `bits`, `mode`) live at the top of the block, with
  // per-layer keys interleaved alongside. Build the block from
  // `per_layer_overrides` BEFORE moving them into [`PerLayerQuantization`],
  // so we don't pay a `HashMap` clone.
  let mut quant_block = serde_json::Map::new();
  quant_block.insert(
    "group_size".to_string(),
    serde_json::Value::Number(group_size.into()),
  );
  quant_block.insert("bits".to_string(), serde_json::Value::Number(bits.into()));
  quant_block.insert(
    "mode".to_string(),
    serde_json::Value::String(mode.as_str().to_string()),
  );
  for (path, opt) in &per_layer_overrides {
    match opt {
      QuantizationOption::Skip => {
        quant_block.insert(path.clone(), serde_json::Value::Bool(false));
      }
      QuantizationOption::Quantize(q) => {
        let mut nested = serde_json::Map::new();
        nested.insert(
          "group_size".to_string(),
          serde_json::Value::Number(q.group_size.into()),
        );
        nested.insert("bits".to_string(), serde_json::Value::Number(q.bits.into()));
        nested.insert(
          "mode".to_string(),
          serde_json::Value::String(q.mode.as_str().to_string()),
        );
        quant_block.insert(path.clone(), serde_json::Value::Object(nested));
      }
    }
  }
  map.insert(
    "quantization".to_string(),
    serde_json::Value::Object(quant_block),
  );

  let updated_text = serde_json::to_string(&serde_json::Value::Object(map)).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "convert: cannot re-serialize patched config",
      "JSON",
      e,
    ))
  })?;

  let live_cfg = PerLayerQuantization::new(Some(global), per_layer_overrides);
  Ok((live_cfg, updated_text))
}

/// Strip `quantization` and `quantization_config` from the in-memory
/// config text, mirroring `convert.py:162-163` (`config.pop(...)`).
/// Used by the dequantize branch so the saved config doesn't carry a
/// stale quant block.
fn strip_quantization_blocks(config_json: &str) -> Result<String> {
  let value: serde_json::Value = serde_json::from_str(config_json)
    .map_err(|e| Error::Parse(ParsePayload::new("convert: source config", "JSON", e)))?;
  let serde_json::Value::Object(mut map) = value else {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "convert: source config JSON",
      "must be an object",
    )));
  };
  map.remove("quantization");
  map.remove("quantization_config");
  let stripped = serde_json::Value::Object(map);
  serde_json::to_string(&stripped).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "convert: cannot re-serialize stripped config",
      "JSON",
      e,
    ))
  })
}

// ─────────────────── copy_tokenizer_and_extras ───────────────────

/// File-name families [`copy_tokenizer_and_extras`] copies from the
/// source dir to the destination dir. Mirrors the union of:
///
/// - HF [`tokenizer.save_pretrained`]'s typical output set (the python
///   `utils.save:944` line `tokenizer.save_pretrained(dst_path)` emits
///   these from the in-memory tokenizer; mlxrs's tokenizer is
///   load-only, so the on-disk files are copied verbatim instead).
/// - The explicit `*.py` + `generation_config.json` glob at
///   `utils.save:946-948`.
///
/// `config.json` is intentionally NOT in the list — `save_config`
/// emits the cleaned (`_name_or_path` / `vision_config` removed,
/// `quantization` mirrored to `quantization_config`, sorted) version
/// inside `save`. Weight files (`*.safetensors` / `*.bin` / `*.gguf`)
/// are NOT copied — `save_model` writes the new sharded layout.
///
/// [`tokenizer.save_pretrained`]: https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/tokenizer
///
/// `pub(crate)` so [`crate::lm::fuse::fuse`]'s staging-promote step can
/// walk the SAME fixed-name family to drop stale destination files that
/// the source dir does not carry (a destination
/// pre-populated with e.g. a stale `generation_config.json` /
/// `chat_template.jinja` would otherwise survive the fuse, since the
/// permissive `copy_tokenizer_and_extras` only OVERWRITES files present
/// at the source — the `_skipped` arm leaves any absent name alone, so
/// stale destination bytes silently ship as the fused checkpoint's
/// tokenizer / generation config). Keeping the constant single-source
/// guarantees the snapshot+stale-walk and the copy operate over the
/// IDENTICAL file family.
pub(crate) const TOKENIZER_EXTRA_FILES: &[&str] = &[
  // Core tokenizer.
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "added_tokens.json",
  // SentencePiece-family vocab artifacts.
  "spiece.model",
  "tokenizer.model",
  // Byte-pair-encoding vocab artifacts.
  "vocab.json",
  "merges.txt",
  // Templating.
  "chat_template.jinja",
  // mlx-lm explicit extras (`utils.save:946-948`).
  "generation_config.json",
];

/// Per-fsync-boundary durability warnings observed by a single
/// [`copy_tokenizer_and_extras`] call.
///
/// Each field is `Some(_)` iff the corresponding post-copy fsync
/// boundary returned `Err` AFTER its underlying [`std::fs::copy`]
/// succeeded; the data is on disk either way, only durability across
/// a power loss is uncertain.
///
/// Returned (via [`CopyOutcome::CommittedWithDurabilityWarning`]) so
/// [`convert`] can route the multi-warning case (`save` warned + at
/// least one post-copy fsync warned, or both post-copy fsyncs warned)
/// to the typed [`Error::ConvertDurabilityWarnings`] aggregate instead
/// of an `std::io::Error::other(format!(...))` fold (which would lose
/// typed access to the individual warnings).
#[derive(Debug, Default)]
pub struct CopyDurabilityWarnings {
  /// First per-file `fsync_path_io` warning observed (preserves the
  /// earliest-failure information so the user can pinpoint which file's
  /// fsync was the first to lose durability). `None` if every per-file
  /// fsync passed.
  ///
  /// The carried [`std::io::Error`] preserves the raw
  /// underlying [`std::io::ErrorKind`] (ENOSPC / EIO /
  /// PermissionDenied / ...). Calling the (crate-
  /// internal) `fsync_path` helper instead (which returns the crate-wide
  /// [`crate::Error::Backend`] variant — string-wrapping the
  /// underlying io::Error and losing its kind) and then re-wrapping the
  /// message via [`std::io::Error::other`] would collapse every kind to
  /// [`std::io::ErrorKind::Other`] and force callers to parse the
  /// message text. Routing through the kind-preserving
  /// `fsync_path_io` sibling (returns [`std::io::Result`]) instead lets callers
  /// branch on `.kind()` directly.
  ///
  /// The message is also wrapped at the call site with
  /// `"copy_tokenizer_and_extras: fsync <destination-path> failed:
  /// <inner>"` so callers (and humans reading the warning) can pinpoint
  /// WHICH copied tokenizer file warned — without this a real OS
  /// failure (where the inner [`std::io::Error`] message is the
  /// context-free OS text like `"No such file or directory (os error
  /// 2)"`) would surface no path information. The wrap preserves the
  /// underlying `kind()` (uses [`std::io::Error::new`], not
  /// [`std::io::Error::other`]) so the kind-preservation contract is intact.
  pub post_copy_file: Option<std::io::Error>,
  /// Post-copy `fsync_dir(dst)` warning, after every per-file fsync
  /// has been observed. `None` if the directory fsync passed.
  pub post_copy_dir: Option<std::io::Error>,
}

/// Outcome of a successful [`copy_tokenizer_and_extras`] call —
/// disambiguates "all files are durable on disk" from "all files'
/// content reached disk but at least one post-copy `fsync` warned".
///
/// A copy is considered **logically complete** the moment
/// [`std::fs::copy`] returns `Ok`: the file's bytes are in the page
/// cache and a subsequent reader (including a process restart on a
/// running kernel) will observe them. The post-copy `fsync_path` /
/// `fsync_dir` calls only convert "logically complete" into "durable
/// across a power loss". A fsync error AFTER the content is in place
/// therefore does NOT mean the data is missing — only that durability
/// is uncertain.
///
/// The [`Error::ConvertPostSavePartial`] variant stays reserved for
/// the genuine "copy actually failed" case where [`std::fs::copy`]
/// itself returned `Err` (the file did NOT reach disk); the
/// `CommittedWithDurabilityWarning` variant carries the post-copy
/// fsync warnings so [`convert`] can re-surface them via the existing
/// [`Error::DurabilityWarning`] shape (single-warning) or the typed
/// [`Error::ConvertDurabilityWarnings`] aggregate (multi-warning).
///
/// Used internally by [`convert`] (the public surface still returns
/// [`Result<()>`]) and surfaced via [`copy_tokenizer_and_extras`] for
/// callers that drive the helper standalone (e.g. partial-convert
/// recovery flows) — those callers need the same machine-detectable
/// "data on disk, durability uncertain" signal.
///
/// Folding the per-file + dir fsync warnings into a single
/// [`std::io::Error`] (via `std::io::Error::other(format!(...))`) would
/// lose typed access to the two underlying errors. This variant carries
/// each warning separately in [`CopyDurabilityWarnings`] so the caller
/// (and the convert()-side aggregate) can destructure WHICH boundary
/// warned.
#[derive(Debug)]
pub enum CopyOutcome {
  /// All [`std::fs::copy`] calls returned `Ok` AND all post-copy
  /// `fsync_path` + post-copy `fsync_dir(dst)` returned `Ok`. The
  /// destination directory is fully durable.
  Committed,
  /// All [`std::fs::copy`] calls returned `Ok` (so the file content
  /// reached disk and is observable by a subsequent reader) BUT at
  /// least one post-copy `fsync_path` or the post-copy
  /// `fsync_dir(dst)` returned `Err`. The data IS on disk; only
  /// durability across a power loss is uncertain. Each fsync boundary's
  /// error is carried separately in [`CopyDurabilityWarnings`] so the
  /// caller can machine-detect WHICH boundary warned (no string parse).
  CommittedWithDurabilityWarning(CopyDurabilityWarnings),
}

/// Copy the tokenizer + extras family of files from `src` to `dst`,
/// mirroring `utils.save:944-948` minus the `tokenizer.save_pretrained`
/// in-memory re-serialization (mlxrs's tokenizer surface is load-only;
/// the on-disk files are copied verbatim).
///
/// Files copied:
///
/// - Every basename in the fixed tokenizer-extras list (see the source
///   constant `TOKENIZER_EXTRA_FILES`) that exists at `src` — the union
///   of `tokenizer.save_pretrained`'s typical output set +
///   mlx-lm's explicit `generation_config.json`. The set spans the core
///   tokenizer files (`tokenizer.json`, `tokenizer_config.json`,
///   `special_tokens_map.json`, `added_tokens.json`), the
///   SentencePiece-family vocab artifacts (`spiece.model`,
///   `tokenizer.model`), the BPE artifacts (`vocab.json`, `merges.txt`),
///   templating (`chat_template.jinja`), and `generation_config.json`.
/// - Every `*.py` at `src` (the python `glob("*.py")` at
///   `utils.save:946-947` — HF model code some loaders need).
///
/// Files explicitly NOT copied (per the module-level "scope decisions"):
///
/// - `config.json` — `save_config` already writes the cleaned
///   version inside `convert`'s save step.
/// - `*.safetensors` / `*.bin` / `*.gguf` / `model.safetensors.index.json`
///   — `save_model` writes the new sharded layout.
///
/// **Rename-in-place** (`src == dst`): no-op. Same-path copies would
/// either truncate-to-zero or no-op depending on `std::fs::copy`'s
/// implementation; the explicit guard short-circuits the entire walk.
/// `src`'s files are left untouched.
///
/// **Post-copy durability:** after each successful
/// [`std::fs::copy`], the copied file is `fsync`ed (via the
/// `crate::lm::load::fsync_path_io` helper — the `io::Result<()>`
/// variant of `fsync_path` that preserves the underlying
/// [`std::io::ErrorKind`]) so its bytes are durable on
/// disk. After ALL copies complete, the destination directory is
/// `fsync`ed (via the `crate::lm::load::fsync_dir` helper) so the new
/// directory entries are durable. Without these, an `Ok` return would
/// lie: a crash AFTER `convert() → Ok(())` could leave weights /
/// config durable (the save side already fsyncs them) but tokenizer
/// files torn / missing — breaking the documented "Ok = durable"
/// contract.
///
/// Failure semantics:
///
/// - A missing source file is silently skipped (the python `for file
///   in glob(...)` is naturally absent-tolerant).
/// - An IO failure on the [`std::fs::copy`] of an existing source file
///   is a recoverable [`Error::Backend`] naming the offending file and
///   the underlying error — the file did NOT reach disk, so the
///   destination dir is structurally incomplete.
/// - An IO failure on a post-copy `fsync_path` / `fsync_dir` AFTER the
///   file content reached disk is folded into the returned
///   [`CopyOutcome::CommittedWithDurabilityWarning`] (the data IS on
///   disk; only durability is uncertain). Distinguishable from a hard
///   copy failure at the type level.
///
/// [`tokenizer.save_pretrained`]: https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/tokenizer
pub fn copy_tokenizer_and_extras(src: &Path, dst: &Path) -> Result<CopyOutcome> {
  // Rename-in-place: nothing to do (mirrors python's natural behavior —
  // `tokenizer.save_pretrained(dst)` is a no-op when the tokenizer was
  // loaded from `dst`, and `shutil.copy` on the same path is at best a
  // no-op at worst a truncate). Short-circuit so we don't accidentally
  // unlink-then-recreate. No fsync needed: nothing changed on disk.
  if paths_are_same(src, dst) {
    return Ok(CopyOutcome::Committed);
  }

  // Per-boundary warnings captured rather than early-returned so the
  // WHOLE batch of copies runs (data durability is best-effort
  // post-copy) and so the caller (convert()) can route the multi-
  // warning case to the typed [`Error::ConvertDurabilityWarnings`]
  // aggregate.
  let mut warnings = CopyDurabilityWarnings::default();

  // Record a per-file fsync warning iff none has been observed yet —
  // preserves FIRST-failure information so the user can pinpoint the
  // earliest file whose fsync lost durability. We do NOT fold a later
  // file-fsync warning into the first (callers can re-run the copy to
  // collect the rest if needed); the dir-fsync boundary is recorded
  // separately below.
  let mut record_file_fsync_warning = |e: std::io::Error| {
    if warnings.post_copy_file.is_none() {
      warnings.post_copy_file = Some(e);
    }
  };

  // Wrap `fsync_path_io`'s raw [`std::io::Error`] with
  // operation + destination-path context BEFORE recording. Without this
  // a REAL post-copy fsync failure (where [`std::fs::File::open`] /
  // [`std::fs::File::sync_all`] return an [`std::io::Error`] whose
  // message is the OS-level text only — `"No such file or directory
  // (os error 2)"`, `"Input/output error (os error 5)"`, ...) gives the
  // caller no way to identify WHICH copied tokenizer file warned or
  // whether the failure was the reopen vs the sync_all. The injected-
  // error tests pass the path-context assertion by accident (the
  // test-only injector formats the path into its message), but a real
  // failure in production would surface a context-free OS string.
  //
  // [`std::io::Error::new`] preserves the underlying [`ErrorKind`]
  // (callers branch on `.kind()` to disambiguate
  // ENOSPC / EIO / PermissionDenied / NotFound / ...) while the wrap
  // adds the missing operation + path context to the message. The
  // helper is closed over `dst`'s display lifetime per-call so each
  // entry's warning names its OWN file.
  //
  // Takes `e` by reference: we read `kind()` (`&self`) and Display-
  // interpolate (`{e}` via `Display::fmt(&self, _)`) and DON'T need to
  // consume the value — the new wrapped [`std::io::Error`] is
  // constructed from these two views plus the path-context message.
  // Avoids clippy::needless_pass_by_value.
  fn wrap_fsync_err(dst: &Path, e: &std::io::Error) -> std::io::Error {
    std::io::Error::new(
      e.kind(),
      format!(
        "copy_tokenizer_and_extras: fsync {} failed: {e}",
        dst.display()
      ),
    )
  }

  // The fixed-set extras.
  for name in TOKENIZER_EXTRA_FILES {
    let s = src.join(name);
    if !s.is_file() {
      continue;
    }
    let d = dst.join(name);
    // Typed-error preservation: the underlying
    // `std::io::ErrorKind` (PermissionDenied / NoSpace / ReadOnlyFilesystem
    // / …) must reach `ConvertPostSavePartialPayload::copy_error` intact
    // so callers can branch on `FileOp::Copy` + `path` + `inner.kind()`
    // without parsing message text. Stringifying into `Error::Backend`
    // here defeats the typed pass-through that the convert() handler
    // sets up via `Box<Error>`.
    std::fs::copy(&s, &d).map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "copy_tokenizer_and_extras",
        FileOp::Copy,
        d.clone(),
        e,
      ))
    })?;
    // Post-copy file fsync. The data IS on disk
    // after `std::fs::copy` returns Ok; this only converts "logically
    // complete" into "durable across a power loss". A failure here
    // does NOT mean the file is missing — record + continue.
    //
    // Use the kind-preserving
    // [`crate::lm::load::fsync_path_io`] variant so the underlying
    // [`std::io::ErrorKind`] (ENOSPC / EIO / PermissionDenied / ...)
    // carries through to the structured aggregate intact. The previous
    // shape called [`crate::lm::load::fsync_path`] (returns crate
    // [`Error::Backend`]) and then re-wrapped the message via
    // `io::Error::other(...)` — collapsing every real kind to
    // [`std::io::ErrorKind::Other`] and forcing callers to parse the
    // message text to disambiguate quota / permission / disk failures.
    //
    // Wrap with operation + destination-path context at
    // the call site (kind preserved). See `wrap_fsync_err` above.
    if let Err(e) = crate::lm::load::fsync_path_io(&d) {
      record_file_fsync_warning(wrap_fsync_err(&d, &e));
    }
  }

  // The python `*.py` glob (HF model code; some loaders need it).
  let entries = std::fs::read_dir(src).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "copy_tokenizer_and_extras",
      FileOp::Read,
      src.to_path_buf(),
      e,
    ))
  })?;
  for entry in entries {
    let entry = entry.map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "copy_tokenizer_and_extras: cannot read entry in",
        FileOp::Read,
        src.to_path_buf(),
        e,
      ))
    })?;
    let path = entry.path();
    if !path.is_file() {
      continue;
    }
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
      continue;
    };
    if !name.ends_with(".py") {
      continue;
    }
    let d = dst.join(name);
    // Typed-error preservation: preserve the
    // io::ErrorKind through the Error::FileIo variant for the *.py
    // glob copies — same rationale as the TOKENIZER_EXTRA_FILES loop
    // above.
    std::fs::copy(&path, &d).map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "copy_tokenizer_and_extras",
        FileOp::Copy,
        d.clone(),
        e,
      ))
    })?;
    // Post-copy file fsync — same rationale + kind-preservation as
    // the TOKENIZER_EXTRA_FILES loop above. Wrapped
    // with operation + destination-path context.
    if let Err(e) = crate::lm::load::fsync_path_io(&d) {
      record_file_fsync_warning(wrap_fsync_err(&d, &e));
    }
  }

  // Post-copy directory fsync — makes the new directory entries
  // (every `dst/<name>` created above) durable. Same shape as
  // [`crate::lm::load::save`]'s post-rename `fsync_dir`. A failure
  // here is also a durability-only warning (the entries are observable
  // by a reader on this running kernel; only a power loss could revert
  // them). Carried in the typed `post_copy_dir` field of
  // [`CopyDurabilityWarnings`] (separate from `post_copy_file`) so the
  // convert()-side aggregate can route the multi-warning case to
  // [`Error::ConvertDurabilityWarnings`] with each warning machine-
  // readable via destructuring (no string fold).
  if let Err(dir_err) = crate::lm::load::fsync_dir(dst) {
    warnings.post_copy_dir = Some(dir_err);
  }

  Ok(
    if warnings.post_copy_file.is_none() && warnings.post_copy_dir.is_none() {
      CopyOutcome::Committed
    } else {
      CopyOutcome::CommittedWithDurabilityWarning(warnings)
    },
  )
}

/// Resolve `src` and `dst` to canonical absolute paths and compare. Used
/// by [`copy_tokenizer_and_extras`] for the rename-in-place fast-path.
/// If `dst` doesn't exist yet (the common case — convert created it
/// fresh), we can't canonicalize it; fall back to a textual compare of
/// the original args. A spurious "same" classification on a textual
/// mismatch would only no-op the copies, which is recoverable; a
/// spurious "different" classification on the rename-in-place case
/// would truncate-then-write the source files (the python case
/// `tokenizer.save_pretrained` rewrites them with the in-memory state,
/// which is byte-equal — but our `std::fs::copy` on the same path is
/// undefined; canonicalization with the fallback is the safe shape).
fn paths_are_same(src: &Path, dst: &Path) -> bool {
  match (std::fs::canonicalize(src), std::fs::canonicalize(dst)) {
    (Ok(a), Ok(b)) => a == b,
    _ => src == dst,
  }
}

// ─────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod unit {
  //! Internal-helper unit tests. The integration / end-to-end suite is
  //! in `mlxrs/tests/lm_convert.rs`.

  use super::*;

  #[test]
  fn defaults_for_mode_table_matches_utils_py_800_808() {
    // `utils.py:800-808` — the hard-coded `mode_defaults` table.
    assert_eq!(defaults_for_mode(QuantMode::Affine, None, None), (64, 4));
    assert_eq!(defaults_for_mode(QuantMode::Mxfp4, None, None), (32, 4));
    assert_eq!(defaults_for_mode(QuantMode::Nvfp4, None, None), (16, 4));
    assert_eq!(defaults_for_mode(QuantMode::Mxfp8, None, None), (32, 8));
    // Explicit kwargs override the per-mode default.
    assert_eq!(
      defaults_for_mode(QuantMode::Affine, Some(128), Some(8)),
      (128, 8)
    );
  }

  /// `Some(0)` is python-falsy and MUST fall
  /// back to the per-mode default (`utils.py:808`: `group_size or
  /// default_group_size, bits or default_bits`). A surviving `0` would
  /// later make `quantize_weights` skip every layer at the
  /// `last % group_size == 0` gate (and `0 % 0` is undefined) — yielding
  /// dense weights on disk against an invalid `group_size: 0` block.
  #[test]
  fn defaults_for_mode_zero_group_size_is_falsy() {
    assert_eq!(
      defaults_for_mode(QuantMode::Affine, Some(0), None),
      (64, 4),
      "Some(0) group_size falls back to mode default"
    );
    assert_eq!(
      defaults_for_mode(QuantMode::Mxfp4, Some(0), None),
      (32, 4),
      "Some(0) group_size falls back to mxfp4 default"
    );
  }

  #[test]
  fn defaults_for_mode_zero_bits_is_falsy() {
    assert_eq!(
      defaults_for_mode(QuantMode::Affine, None, Some(0)),
      (64, 4),
      "Some(0) bits falls back to mode default"
    );
    assert_eq!(
      defaults_for_mode(QuantMode::Mxfp8, None, Some(0)),
      (32, 8),
      "Some(0) bits falls back to mxfp8 default"
    );
  }

  #[test]
  fn defaults_for_mode_negative_also_falls_back() {
    // Defensive: a negative `group_size` / `bits` is invalid per mlx's
    // `quantize` op anyway; the `filter(|&v| v > 0)` arm of the python-
    // truthiness transcription rejects those too.
    assert_eq!(
      defaults_for_mode(QuantMode::Affine, Some(-1), Some(-2)),
      (64, 4)
    );
  }

  #[test]
  fn parse_conversion_dtype_table_matches_convert_py_82() {
    assert_eq!(parse_conversion_dtype("float16"), Some(Dtype::F16));
    assert_eq!(parse_conversion_dtype("bfloat16"), Some(Dtype::BF16));
    assert_eq!(parse_conversion_dtype("float32"), Some(Dtype::F32));
    // Anything else: silent `None` (matches python's
    // `if dtype in MODEL_CONVERSION_DTYPES` gate).
    assert_eq!(parse_conversion_dtype("float64"), None);
    assert_eq!(parse_conversion_dtype("int32"), None);
    assert_eq!(parse_conversion_dtype(""), None);
  }

  #[test]
  fn resolve_target_dtype_explicit_wins() {
    let cfg = r#"{"torch_dtype":"float32"}"#;
    assert_eq!(
      resolve_target_dtype(Some(Dtype::BF16), cfg).unwrap(),
      Some(Dtype::BF16)
    );
  }

  #[test]
  fn resolve_target_dtype_falls_back_to_torch_dtype() {
    let cfg = r#"{"torch_dtype":"bfloat16"}"#;
    assert_eq!(resolve_target_dtype(None, cfg).unwrap(), Some(Dtype::BF16));
  }

  #[test]
  fn resolve_target_dtype_falls_back_to_text_config_dtype() {
    let cfg = r#"{"text_config":{"dtype":"float16"}}"#;
    assert_eq!(resolve_target_dtype(None, cfg).unwrap(), Some(Dtype::F16));
  }

  #[test]
  fn resolve_target_dtype_unknown_is_none() {
    let cfg = r#"{"torch_dtype":"float64"}"#;
    assert_eq!(resolve_target_dtype(None, cfg).unwrap(), None);
  }

  /// An explicit `Some(Dtype::I32)` (or any
  /// non-floating dtype) is a hard `Error::Backend`, NOT a silent
  /// "cast every float to int" wrecking-ball. The reference's
  /// string-typed `if dtype in MODEL_CONVERSION_DTYPES` gate
  /// (`convert.py:133`) silently drops unknown strings; the typed
  /// `Dtype` enum could silently accept e.g. `Dtype::I32` and cast
  /// every weight to int — the port forecloses that foot-gun.
  #[test]
  fn resolve_target_dtype_explicit_i32_is_error() {
    let cfg = r#"{"torch_dtype":"float32"}"#;
    match resolve_target_dtype(Some(Dtype::I32), cfg) {
      Err(Error::UnsupportedDtype(p)) => {
        assert_eq!(p.dtype(), Dtype::I32);
        assert_eq!(p.supported(), &[Dtype::F16, Dtype::BF16, Dtype::F32]);
        assert!(
          p.context().contains("MODEL_CONVERSION_DTYPES"),
          "context names the reference set: {}",
          p.context()
        );
      }
      other => panic!("expected Err(UnsupportedDtype), got {other:?}"),
    }
  }

  #[test]
  fn resolve_target_dtype_explicit_f64_is_error() {
    let cfg = r#"{}"#;
    match resolve_target_dtype(Some(Dtype::F64), cfg) {
      Err(Error::UnsupportedDtype(p)) => {
        assert_eq!(p.dtype(), Dtype::F64);
        assert_eq!(p.supported(), &[Dtype::F16, Dtype::BF16, Dtype::F32]);
      }
      other => panic!("expected Err(UnsupportedDtype), got {other:?}"),
    }
  }

  #[test]
  fn resolve_target_dtype_explicit_complex_is_error() {
    let cfg = r#"{}"#;
    match resolve_target_dtype(Some(Dtype::Complex64), cfg) {
      Err(Error::UnsupportedDtype(p)) => {
        assert_eq!(p.dtype(), Dtype::Complex64);
        assert_eq!(p.supported(), &[Dtype::F16, Dtype::BF16, Dtype::F32]);
      }
      other => panic!("expected Err(UnsupportedDtype), got {other:?}"),
    }
  }

  #[test]
  fn resolve_target_dtype_explicit_bool_is_error() {
    let cfg = r#"{}"#;
    match resolve_target_dtype(Some(Dtype::Bool), cfg) {
      Err(Error::UnsupportedDtype(p)) => {
        assert_eq!(p.dtype(), Dtype::Bool);
        assert_eq!(p.supported(), &[Dtype::F16, Dtype::BF16, Dtype::F32]);
      }
      other => panic!("expected Err(UnsupportedDtype), got {other:?}"),
    }
  }

  #[test]
  fn strip_quantization_blocks_removes_both_keys() {
    let cfg = r#"{
      "model_type":"qwen3",
      "quantization":{"group_size":64,"bits":4},
      "quantization_config":{"group_size":64,"bits":4}
    }"#;
    let out = strip_quantization_blocks(cfg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(parsed.get("quantization").is_none());
    assert!(parsed.get("quantization_config").is_none());
    // Unrelated keys preserved.
    assert_eq!(
      parsed.get("model_type").and_then(|v| v.as_str()),
      Some("qwen3")
    );
  }

  // ─────────── single-evaluation predicate ───────────

  use std::cell::RefCell;

  /// Test-only predicate that counts how many times `decide` was called
  /// per layer path. Used by the single-evaluation closure test to assert
  /// `nn.quantize`'s single-call-per-module semantics
  /// (`utils.py:837-843`).
  struct CountingPredicate {
    /// `RefCell<HashMap>` because `MixedQuantPredicate::decide` is
    /// `&self`; the trait is `!Send` so single-threaded interior
    /// mutability is the right shape.
    counts: RefCell<HashMap<String, u32>>,
    /// `RefCell<u32>` cycle counter for the alternating decision
    /// (used by the "non-deterministic predicate desyncs config from
    /// weights" half of the test). Each call increments it; the
    /// predicate returns `Some(_)` on odd cycles and `None` on even.
    cycle: RefCell<u32>,
  }

  impl CountingPredicate {
    fn new() -> Self {
      Self {
        counts: RefCell::new(HashMap::new()),
        cycle: RefCell::new(0),
      }
    }

    fn max_count(&self) -> u32 {
      self.counts.borrow().values().copied().max().unwrap_or(0)
    }

    fn paths_seen(&self) -> Vec<String> {
      let mut paths: Vec<String> = self.counts.borrow().keys().cloned().collect();
      paths.sort();
      paths
    }
  }

  impl MixedQuantPredicate for CountingPredicate {
    fn decide(&self, layer_name: &str, _weight: &Array) -> Option<Quantization> {
      *self
        .counts
        .borrow_mut()
        .entry(layer_name.to_string())
        .or_insert(0) += 1;
      // Bump cycle — used to make the predicate flip-flop. The
      // desync (config-says-quantize, weights-say-skip) would manifest
      // if the predicate were called twice and saw different cycle
      // values.
      *self.cycle.borrow_mut() += 1;
      Some(Quantization {
        group_size: 64,
        bits: 4,
        mode: QuantMode::Affine,
      })
    }
  }

  /// `build_predicate_decisions` must call
  /// the predicate exactly ONCE per structurally-eligible layer.
  /// Evaluating the predicate twice (once in `build_quantize_config`,
  /// again in the `eligible` closure of `convert`) would let a stateful
  /// predicate record one decision in the saved config and apply a
  /// different one to weights. This test counts invocations per path and
  /// asserts max <= 1, mirroring the python `nn.quantize`'s
  /// single-call-per-module contract (`utils.py:837-843`).
  #[test]
  fn build_predicate_decisions_calls_predicate_once_per_eligible_layer() {
    let mut weights: Weights = HashMap::new();
    // Three structurally-eligible layers (rank 2, last axis 64 ⇒
    // last % 64 == 0).
    for path in ["layer.a", "layer.b", "layer.c"] {
      weights.insert(
        format!("{path}.weight"),
        Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
      );
    }
    // One structurally-INeligible layer (rank 1) — must NEVER reach
    // the predicate (it fails the `shape.len() >= 2` gate).
    weights.insert(
      "layer.d.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 64], &(64usize,)).unwrap(),
    );

    let pred = CountingPredicate::new();
    let decisions = build_predicate_decisions(Some(&pred), &weights, 64);

    // Eligible layers each got exactly one call.
    let counts = pred.counts.borrow();
    assert_eq!(counts.len(), 3, "exactly the 3 eligible layers visited");
    for path in ["layer.a", "layer.b", "layer.c"] {
      assert_eq!(
        counts.get(path).copied(),
        Some(1),
        "{path} called exactly once",
      );
    }
    // The ineligible layer was never called.
    assert!(
      !counts.contains_key("layer.d"),
      "structurally-ineligible layer never reaches the predicate"
    );

    // The decision map records each eligible layer with the predicate's
    // single return.
    assert_eq!(decisions.len(), 3, "decision map has 3 eligible entries");
    for path in ["layer.a", "layer.b", "layer.c"] {
      assert!(
        matches!(decisions.get(path), Some(Some(_))),
        "{path} decision recorded as Some(Some(_))",
      );
    }
  }

  /// Re-call after the map is built must NOT
  /// re-invoke the predicate. The full convert pipeline runs the
  /// predicate once (in `build_predicate_decisions`); both downstream
  /// consumers (`build_quantize_config` + the `eligible` closure) read
  /// from the cached map. This test models the pipeline by calling
  /// `build_quantize_config` after `build_predicate_decisions` and
  /// asserts the counter never moves past the single invocation.
  #[test]
  fn build_quantize_config_does_not_reinvoke_predicate() {
    let mut weights: Weights = HashMap::new();
    for path in ["layer.a", "layer.b"] {
      weights.insert(
        format!("{path}.weight"),
        Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
      );
    }

    let pred = CountingPredicate::new();
    let decisions = build_predicate_decisions(Some(&pred), &weights, 64);
    // Snapshot the post-decisions counter for each path.
    let after_decisions = pred.counts.borrow().clone();

    // Now run the config builder.
    let _ = build_quantize_config("{}", 64, 4, QuantMode::Affine, &decisions, &weights).unwrap();

    // The counter MUST NOT have moved — `build_quantize_config` reads
    // from `decisions`, never from the predicate.
    assert_eq!(
      *pred.counts.borrow(),
      after_decisions,
      "build_quantize_config must not re-invoke the predicate"
    );
    assert_eq!(pred.max_count(), 1, "every layer's count is still 1");
    let paths = pred.paths_seen();
    assert_eq!(paths, vec!["layer.a".to_string(), "layer.b".to_string()]);
  }

  // ─────────── DurabilityWarning still copies tokenizer ───────────

  /// When `load::save` returns
  /// [`Error::DurabilityWarning`] with `committed: true`, the weights +
  /// config are visible on disk; `convert` MUST continue with the
  /// tokenizer / extras copy (so the destination dir is COMPLETE)
  /// before re-surfacing the warning. Using `?` on the `load::save`
  /// call would early-return and SKIP the `copy_tokenizer_and_extras`
  /// step — turning a non-fatal durability warning into a partial,
  /// hard-to-recover conversion (the destination dir would exist so the
  /// `mlx_path.exists()` gate of a retry would reject it, but tokenizer
  /// files would be missing).
  ///
  /// This test:
  ///   (a) arms the `fsync_dir` fault injector to fire AFTER the
  ///       shard fsync (skip=1, fires on the index-fsync) — driving
  ///       `save` into the `CommittedWithDurabilityWarning` branch.
  ///   (b) runs `convert` end-to-end through this driver.
  ///   (c) asserts: the dst dir is COMPLETE (config + index + a
  ///       tokenizer-extras file ARE present), AND convert's final
  ///       return is `Err(DurabilityWarning { committed: true, .. })`.
  #[test]
  fn convert_durability_warning_still_copies_tokenizer_and_returns_warning() {
    // Build a synthetic source dir with config + weights + a tokenizer
    // file we can assert was copied.
    let dir = std::env::temp_dir().join(format!("mlxrs_convert_durability_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let src = dir.join("src");
    let dst = dir.join("dst");
    std::fs::create_dir_all(&src).unwrap();

    let plain_config = r#"{
      "model_type":"qwen3","hidden_size":16,"num_hidden_layers":1,
      "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
      "rope_theta":10000.0,"vocab_size":128,"tie_word_embeddings":false
    }"#;
    std::fs::write(src.join("config.json"), plain_config).unwrap();
    let blob: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "layer.weight".to_string(),
      Array::from_slice::<f32>(&blob, &(2usize, 64usize)).unwrap(),
    );
    crate::io::save_safetensors(&src.join("model.safetensors"), &weights).unwrap();

    // Plant the tokenizer-extras files (we don't actually need a real
    // tokenizer to load; convert's tokenizer-loading is independent of
    // copy_tokenizer_and_extras's copy list, and we use a real
    // tokenizer.json + tokenizer_config.json from the integration
    // suite's fixtures so `load_tokenizer` succeeds).
    let tokenizer_json = include_str!("../../tests/fixtures/tokenizer.json");
    let tokenizer_config_json = include_str!("../../tests/fixtures/tokenizer_config.json");
    std::fs::write(src.join("tokenizer.json"), tokenizer_json).unwrap();
    std::fs::write(src.join("tokenizer_config.json"), tokenizer_config_json).unwrap();
    // A "marker" extras file that the helper MUST copy.
    std::fs::write(
      src.join("special_tokens_map.json"),
      br#"{"eos_token":"</s>"}"#,
    )
    .unwrap();
    std::fs::write(src.join("generation_config.json"), br#"{"max_length":32}"#).unwrap();

    // Arm the fault injector to fire on the index-fsync (skip=1: shard
    // fsync passes, index fsync fails → save_model returns
    // CommittedWithDurabilityWarning; save() surfaces a final
    // Err(DurabilityWarning) AFTER committing the config too).
    let _guard = crate::lm::load::arm_fsync_dir_fault(1);

    let r = convert(ConvertArgs {
      hf_path: src.clone(),
      mlx_path: dst.clone(),
      ..Default::default()
    });
    drop(_guard);

    // (a) convert's final return is `Err(DurabilityWarning{committed:true})`.
    match r {
      Err(Error::DurabilityWarning(p)) => {
        assert!(
          p.committed(),
          "convert's DurabilityWarning must carry committed=true"
        );
        assert!(
          p.source()
            .to_string()
            .contains("injected fsync_dir failure"),
          "underlying io::Error preserved: got {}",
          p.source()
        );
      }
      other => panic!("expected Err(DurabilityWarning), got {other:?}"),
    }

    // (b) The destination dir IS complete: weights + index + config +
    // tokenizer files all present, byte-equal to source where
    // applicable.
    assert!(dst.join("config.json").is_file(), "config.json on disk");
    assert!(
      dst.join("model.safetensors.index.json").is_file(),
      "index.json on disk"
    );
    for name in [
      "tokenizer.json",
      "tokenizer_config.json",
      "special_tokens_map.json",
      "generation_config.json",
    ] {
      assert!(
        dst.join(name).is_file(),
        "{name} copied despite the DurabilityWarning"
      );
      // Byte-equal to the source copy.
      let a = std::fs::read(src.join(name)).unwrap();
      let b = std::fs::read(dst.join(name)).unwrap();
      assert_eq!(a, b, "{name} byte-equal at dst");
    }

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
  }

  // ─── DurabilityWarning preserved across
  //     a post-save tokenizer-copy failure ────────────────────────────
  //
  // Stashing the committed-DurabilityWarning into a local and then
  // calling `copy_tokenizer_and_extras(...)?` would let the `?`
  // discard the stashed warning on copy failure, surfacing the copy
  // IO error instead. The caller would then lose the only signal that
  // the checkpoint is already committed: `mlx_path` exists on disk, and
  // a retry's `mlx_path.exists()` gate would reject — silently dropping
  // the already-committed save.
  //
  // This test exercises BOTH faults together:
  //   (1) Arm the `fsync_dir` injector (`skip=1`) so save returns
  //       `Err(DurabilityWarning { committed: true, .. })` — the
  //       committed-durability-warning branch.
  //   (2) `chmod 000` one of the tokenizer-extras files in `src` so
  //       `copy_tokenizer_and_extras` hits an EACCES on `fs::copy` for
  //       that file (a real OS-level copy failure, not an injector).
  //       The chosen file is `special_tokens_map.json` because
  //       [`Tokenizer::from_path`] does NOT read it (it reads
  //       `tokenizer.json` + `tokenizer_config.json`) — so the load
  //       step earlier in `convert` doesn't trip over the permission.
  //
  // Asserts:
  //   (a) The final return is `Err(DurabilityWarning { committed:
  //       true, .. })` — NOT a plain `Error::Backend` from the copy.
  //   (b) The folded message names BOTH `fsync_dir` AND
  //       `copy_tokenizer_and_extras` so the caller can disambiguate
  //       the two failures (and tooling can match on either marker).
  //   (c) The destination dir still has the committed-before-warning
  //       artifacts (weights + index + config). The tokenizer-extras
  //       copy is best-effort post-commit — some tokenizer files MAY
  //       have been copied (depending on iteration order), but the
  //       chmod-000 file itself MUST NOT have been copied (the
  //       source's perm prevents the read).
  //
  // Unix-only: relies on `chmod 000` to produce the EACCES; the
  // `fsync_dir` injector is also a Unix-only meaningful path (the
  // injector is `#[cfg(test)]` but `fsync_dir` itself is a no-op on
  // non-Unix). Keep the test gated to `#[cfg(unix)]` to match.
  #[cfg(unix)]
  #[test]
  fn convert_durability_warning_then_tokenizer_copy_failure_preserves_committed_signal() {
    use std::os::unix::fs::PermissionsExt;

    let dir = std::env::temp_dir().join(format!(
      "mlxrs_convert_durability_then_copyfail_{}",
      std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let src = dir.join("src");
    let dst = dir.join("dst");
    std::fs::create_dir_all(&src).unwrap();

    let plain_config = r#"{
      "model_type":"qwen3","hidden_size":16,"num_hidden_layers":1,
      "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
      "rope_theta":10000.0,"vocab_size":128,"tie_word_embeddings":false
    }"#;
    std::fs::write(src.join("config.json"), plain_config).unwrap();
    let blob: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "layer.weight".to_string(),
      Array::from_slice::<f32>(&blob, &(2usize, 64usize)).unwrap(),
    );
    crate::io::save_safetensors(&src.join("model.safetensors"), &weights).unwrap();

    // Same tokenizer fixtures the other tests use — `load_tokenizer`
    // succeeds because both files are readable.
    let tokenizer_json = include_str!("../../tests/fixtures/tokenizer.json");
    let tokenizer_config_json = include_str!("../../tests/fixtures/tokenizer_config.json");
    std::fs::write(src.join("tokenizer.json"), tokenizer_json).unwrap();
    std::fs::write(src.join("tokenizer_config.json"), tokenizer_config_json).unwrap();
    // A plain tokenizer-extras file that `copy_tokenizer_and_extras`
    // WILL try to copy. `Tokenizer::from_path` does NOT read it, so
    // the chmod-000 below doesn't break the earlier `load_tokenizer`
    // step inside `convert`.
    let chmod_target = src.join("special_tokens_map.json");
    std::fs::write(&chmod_target, br#"{"eos_token":"</s>"}"#).unwrap();
    // Also plant `generation_config.json` (readable) so we can sanity-
    // check that the iteration ORDER doesn't matter — at least one
    // readable extras file may or may not have been copied; the
    // failing file is the assertion target.
    std::fs::write(src.join("generation_config.json"), br#"{"max_length":32}"#).unwrap();

    // Make the chosen extras file unreadable. `fs::copy` will hit
    // EACCES on the read side; `is_file()` still returns true (perm-
    // ission bits don't affect a stat).
    let mut perm = std::fs::metadata(&chmod_target).unwrap().permissions();
    perm.set_mode(0o000);
    std::fs::set_permissions(&chmod_target, perm).unwrap();

    // Drop guard restores permissions even on test panic so cleanup +
    // any future test run isn't blocked by an undeletable file.
    struct PermRestore(std::path::PathBuf);
    impl Drop for PermRestore {
      fn drop(&mut self) {
        if let Ok(meta) = std::fs::metadata(&self.0) {
          let mut p = meta.permissions();
          p.set_mode(0o644);
          let _ = std::fs::set_permissions(&self.0, p);
        }
      }
    }
    let _perm_guard = PermRestore(chmod_target);

    // Arm fsync-dir fault: `skip=1` → shard fsync passes, index fsync
    // fails → save_model returns CommittedWithDurabilityWarning,
    // save() surfaces a final Err(DurabilityWarning { committed:true }).
    let _guard = crate::lm::load::arm_fsync_dir_fault(1);

    let r = convert(ConvertArgs {
      hf_path: src,
      mlx_path: dst.clone(),
      ..Default::default()
    });
    drop(_guard);

    // (a) Final return is the structured
    //     `Err(ConvertPostSavePartial { committed: true, save_warning:
    //     Some(_), copy_error: _ })` — NOT a free-form
    //     [`DurabilityWarning`] with the copy error folded into its
    //     `source` string. The contract: a post-commit copy failure
    //     MUST surface a distinct variant so the caller can
    //     machine-detect "destination structurally incomplete" vs
    //     "logically-complete checkpoint with fsync warning", AND
    //     BOTH the save-side warning AND the copy failure stay
    //     individually accessible via typed fields (no string parse).
    match &r {
      Err(Error::ConvertPostSavePartial(p)) => {
        assert!(
          p.committed(),
          "ConvertPostSavePartial must carry committed=true (variant is \
           reachable only after the observable commit point)"
        );
        // (b) `save_warning` is `Some(_)` because the save side raised
        //     a `DurabilityWarning` (the fsync-dir injector fired on
        //     skip=1). The underlying IO error is machine-readable
        //     via `.kind()` and the verbatim original message
        //     (`injected fsync_dir failure ...`) is preserved (no
        //     string fold).
        let save_warning = p
          .save_warning()
          .expect("save_warning must be Some — the fsync-dir injector fired");
        // `.kind()` is the machine-readable accessor — assert it's a
        // real IO error category (not the catch-all `Other` that
        // `io::Error::other(format!(..))` produces). `fsync_dir`
        // returns the OS-level errno via `Errno::result(...)?.into()`,
        // so the kind is whatever the OS reported (commonly `Other`
        // for ad-hoc fault-injector strings, but the source string
        // below is the verbatim assertion).
        let _ = save_warning.kind();
        assert!(
          save_warning
            .to_string()
            .contains("injected fsync_dir failure"),
          "save_warning preserves the verbatim fsync_dir io::Error \
           message: got {save_warning}"
        );
        // (c) `copy_error` carries the actual tokenizer-copy failure.
        //     It's typed `Error::FileIo` so
        //     callers can branch on `op` / `path` / `inner.kind()`
        //     without string parsing; its Display names
        //     `copy_tokenizer_and_extras` (the function that returned
        //     it). The two errors are NOT folded into a single
        //     free-form string anymore.
        let copy_error = p.copy_error();
        assert!(
          matches!(copy_error, Error::FileIo(_)),
          "copy_error is the typed FileIo variant (machine-readable, \
           not stringified); got: {copy_error:?}"
        );
        let copy_msg = copy_error.to_string();
        assert!(
          copy_msg.contains("copy_tokenizer_and_extras"),
          "copy_error names copy_tokenizer_and_extras; got: {copy_msg}"
        );
        assert!(
          copy_msg.contains("special_tokens_map.json"),
          "copy_error names the failing file (special_tokens_map.json); \
           got: {copy_msg}"
        );
      }
      other => panic!(
        "expected Err(ConvertPostSavePartial), got {other:?} — the post-save \
         copy failure must surface the structured variant so the caller can \
         machine-detect 'destination structurally incomplete'"
      ),
    }

    // (c) Destination dir has the committed-before-warning artifacts:
    //     weights + index + config — these are what `save` actually
    //     committed before the index-fsync warning fired.
    assert!(dst.join("config.json").is_file(), "config.json on disk");
    assert!(
      dst.join("model.safetensors.index.json").is_file(),
      "index.json on disk"
    );
    let any_shard = std::fs::read_dir(&dst)
      .unwrap()
      .filter_map(|e| e.ok())
      .any(|e| {
        e.path()
          .file_name()
          .and_then(|n| n.to_str())
          .map(|n| n.ends_with(".safetensors"))
          .unwrap_or(false)
      });
    assert!(any_shard, "at least one shard committed on disk");

    // (d) The chmod-000 source file MUST NOT have been copied — its
    //     read failed before any byte landed at dst. The OTHER
    //     extras files MAY or may not have been copied depending on
    //     iteration order: `copy_tokenizer_and_extras` walks the
    //     TOKENIZER_EXTRA_FILES const-array in order, so any file
    //     iterated BEFORE the chmod-000 entry WAS copied; any file
    //     iterated AFTER it was skipped by the early-return — both
    //     are best-effort post-commit. This is intentionally NOT
    //     asserted (it's iteration-order-dependent and not part of
    //     the committed-signal contract).
    assert!(
      !dst.join("special_tokens_map.json").is_file(),
      "the chmod-000 source file MUST NOT have been copied (its \
       read failed before any bytes were written to dst)"
    );

    // Restore perms BEFORE the dir-wide remove so the cleanup
    // succeeds even if the guard hasn't run yet.
    drop(_perm_guard);
    let _ = std::fs::remove_dir_all(&dir);
  }

  // ─── clean-save + tokenizer-copy failure surfaces
  //     `ConvertPostSavePartial` with `save_warning = None` ───
  //
  // The handler routes BOTH the "committed + durability warning + copy
  // failure" AND the "committed + clean save + copy failure" cases to
  // the structured [`Error::ConvertPostSavePartial`] variant. A bare
  // copy `Error::Backend` on the (None, Err) arm would be correct for
  // "not committed" but the save HAS committed by this point (the index
  // rename succeeded BEFORE the copy_tokenizer_and_extras step runs) —
  // so the destination dir is structurally incomplete, and the caller
  // needs the structured variant's recovery contract.
  //
  // This test exercises the (None, Err) arm in isolation:
  //   (1) No fsync-dir fault injector — `save` returns plain `Ok(())`.
  //   (2) `chmod 000` `special_tokens_map.json` so
  //       `copy_tokenizer_and_extras` hits EACCES.
  //
  // Asserts:
  //   (a) `Err(ConvertPostSavePartial { committed: true,
  //       save_warning: None, copy_error: _ })` — the `None` arm of
  //       `save_warning` is the machine-readable signal that the save
  //       was clean.
  //   (b) `copy_error.kind()` is meaningful and its message names
  //       `copy_tokenizer_and_extras` + the failing file.
  //   (c) The destination dir has the committed artifacts (weights +
  //       index + config) but NOT the chmod-000 file.
  //
  // Unix-only for the same reason as the durability-warning test:
  // `chmod 000` to produce EACCES.
  #[cfg(unix)]
  #[test]
  fn convert_no_durability_warning_then_tokenizer_copy_failure_returns_partial_with_no_save_warning()
   {
    use std::os::unix::fs::PermissionsExt;

    let dir = std::env::temp_dir().join(format!(
      "mlxrs_convert_clean_save_then_copyfail_{}",
      std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let src = dir.join("src");
    let dst = dir.join("dst");
    std::fs::create_dir_all(&src).unwrap();

    let plain_config = r#"{
      "model_type":"qwen3","hidden_size":16,"num_hidden_layers":1,
      "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
      "rope_theta":10000.0,"vocab_size":128,"tie_word_embeddings":false
    }"#;
    std::fs::write(src.join("config.json"), plain_config).unwrap();
    let blob: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "layer.weight".to_string(),
      Array::from_slice::<f32>(&blob, &(2usize, 64usize)).unwrap(),
    );
    crate::io::save_safetensors(&src.join("model.safetensors"), &weights).unwrap();

    let tokenizer_json = include_str!("../../tests/fixtures/tokenizer.json");
    let tokenizer_config_json = include_str!("../../tests/fixtures/tokenizer_config.json");
    std::fs::write(src.join("tokenizer.json"), tokenizer_json).unwrap();
    std::fs::write(src.join("tokenizer_config.json"), tokenizer_config_json).unwrap();
    let chmod_target = src.join("special_tokens_map.json");
    std::fs::write(&chmod_target, br#"{"eos_token":"</s>"}"#).unwrap();
    std::fs::write(src.join("generation_config.json"), br#"{"max_length":32}"#).unwrap();

    let mut perm = std::fs::metadata(&chmod_target).unwrap().permissions();
    perm.set_mode(0o000);
    std::fs::set_permissions(&chmod_target, perm).unwrap();

    struct PermRestore(std::path::PathBuf);
    impl Drop for PermRestore {
      fn drop(&mut self) {
        if let Ok(meta) = std::fs::metadata(&self.0) {
          let mut p = meta.permissions();
          p.set_mode(0o644);
          let _ = std::fs::set_permissions(&self.0, p);
        }
      }
    }
    let _perm_guard = PermRestore(chmod_target);

    // NO fault injector — `save` returns plain `Ok(())`.
    let r = convert(ConvertArgs {
      hf_path: src,
      mlx_path: dst.clone(),
      ..Default::default()
    });

    // (a) `Err(ConvertPostSavePartial { committed: true, save_warning:
    //     None, copy_error: _ })`. The `None` arm of `save_warning`
    //     is the machine-readable signal that the save was clean
    //     (no fsync warning) — distinct from the durability-warning
    //     test which has `save_warning: Some(_)`.
    match &r {
      Err(Error::ConvertPostSavePartial(p)) => {
        assert!(
          p.committed(),
          "ConvertPostSavePartial must carry committed=true (variant is \
           reachable only after the observable commit point)"
        );
        assert!(
          p.save_warning().is_none(),
          "save_warning must be None — the save returned plain Ok(()) \
           with no fsync warning; got: {:?}",
          p.save_warning()
        );
        // (b) `copy_error` is the typed FileIo variant
        //     (machine-readable: branch on `op`/`path`/`inner.kind()`).
        let copy_error = p.copy_error();
        assert!(
          matches!(copy_error, Error::FileIo(_)),
          "copy_error is the typed FileIo variant; got: {copy_error:?}"
        );
        let copy_msg = copy_error.to_string();
        assert!(
          copy_msg.contains("copy_tokenizer_and_extras"),
          "copy_error names copy_tokenizer_and_extras; got: {copy_msg}"
        );
        assert!(
          copy_msg.contains("special_tokens_map.json"),
          "copy_error names the failing file (special_tokens_map.json); \
           got: {copy_msg}"
        );
      }
      other => panic!(
        "expected Err(ConvertPostSavePartial), got {other:?} — a clean \
         save + post-save copy failure MUST surface the structured \
         variant (the destination IS committed, structurally \
         incomplete)"
      ),
    }

    // (c) Destination dir has the committed artifacts but NOT the
    //     chmod-000 file.
    assert!(dst.join("config.json").is_file(), "config.json on disk");
    assert!(
      dst.join("model.safetensors.index.json").is_file(),
      "index.json on disk"
    );
    let any_shard = std::fs::read_dir(&dst)
      .unwrap()
      .filter_map(|e| e.ok())
      .any(|e| {
        e.path()
          .file_name()
          .and_then(|n| n.to_str())
          .map(|n| n.ends_with(".safetensors"))
          .unwrap_or(false)
      });
    assert!(any_shard, "at least one shard committed on disk");
    assert!(
      !dst.join("special_tokens_map.json").is_file(),
      "the chmod-000 source file MUST NOT have been copied"
    );

    drop(_perm_guard);
    let _ = std::fs::remove_dir_all(&dir);
  }

  // ─── `ConvertPostSavePartial` error chain is iterable via
  //     `std::error::Error::source()` ─────────────────
  //
  // The variant uses `#[source]` on `copy_error` so callers walking
  // the error chain via [`std::error::Error::source`] reach the
  // tokenizer-copy failure (the actually-actionable signal the caller
  // needs to retry or recover from). This test asserts the chain is
  // present and that `.source()` points at the `copy_error` (not
  // `save_warning`, which is exposed via direct field access — see
  // the variant doc-comment for the rationale).
  #[test]
  fn convert_post_save_partial_error_chain_iterable() {
    // Construct the variant directly so this test runs on every
    // platform (the (Some, Err) and (None, Err) paths above are
    // Unix-only because they rely on `chmod 000`).
    let copy_err = Error::FileIo(FileIoPayload::new(
      "copy_tokenizer_and_extras",
      crate::error::FileOp::Copy,
      std::path::PathBuf::from("special_tokens_map.json"),
      std::io::Error::new(std::io::ErrorKind::PermissionDenied, "EACCES on copy"),
    ));
    let save_warning_inner = std::io::Error::other("fsync_dir warning");
    let err = Error::ConvertPostSavePartial(ConvertPostSavePartialPayload::new(
      true,
      Some(save_warning_inner),
      copy_err,
    ));

    // The error implements `std::error::Error` (free check: trait
    // object coercion only compiles if the trait is implemented).
    let e: &dyn std::error::Error = &err;
    // The top-level `Display` carries the structured message
    // (committed + incomplete-destination hint).
    let top = e.to_string();
    assert!(
      top.contains("committed=true"),
      "Display carries the structured committed=true tag; got: {top}"
    );
    assert!(
      top.contains("destination directory may be incomplete"),
      "Display carries the structurally-incomplete hint; got: {top}"
    );

    // `.source()` points at `copy_error` (the actually-actionable
    // failure). Its message matches what we constructed.
    let source = e.source().expect(
      "ConvertPostSavePartial has a #[source]-annotated chain — \
       calling .source() must return the copy_error",
    );
    let source_msg = source.to_string();
    assert!(
      source_msg.contains("EACCES on copy"),
      ".source() returns the copy_error (the actionable failure); \
       got: {source_msg}"
    );

    // The same field is independently reachable via destructuring
    // (machine-readable typed access, no string parse).
    if let Error::ConvertPostSavePartial(p) = &err {
      assert!(p.committed());
      assert_eq!(
        p.save_warning().map(|e| e.to_string()).as_deref(),
        Some("fsync_dir warning"),
        "save_warning is reachable via direct field access (typed accessor)"
      );
      // copy_error is the typed FileIo variant — its inner io::Error
      // carries the originating ErrorKind for machine-readable branching.
      let Error::FileIo(io_p) = p.copy_error() else {
        unreachable!("copy_error is FileIo (constructed above)");
      };
      assert_eq!(io_p.inner().kind(), std::io::ErrorKind::PermissionDenied);
      assert!(p.copy_error().to_string().contains("EACCES on copy"));
    } else {
      unreachable!("constructed ConvertPostSavePartial above");
    }
  }

  // ──────────────────────────────────────────────────────────────────
  // Post-copy durability closures (fsync_path + fsync_dir AFTER
  // tokenizer/extras `std::fs::copy`).
  //
  // Without these `copy_tokenizer_and_extras` would not fsync the
  // copied files or the dst dir, so a crash AFTER `convert() → Ok(())`
  // could leave weights+config durable but tokenizer files torn/missing
  // — breaking the documented "Ok = durable" contract for extras.
  //
  // The post-copy durability step:
  //   (1) fsyncs each copied file via [`crate::lm::load::fsync_path`]
  //   (2) fsyncs the dst dir via [`crate::lm::load::fsync_dir`] after
  //       all copies complete
  //   (3) routes post-copy fsync warnings (data IS on disk, only
  //       durability uncertain) into the [`Error::DurabilityWarning`]
  //       variant — distinguishable from a hard
  //       [`Error::ConvertPostSavePartial`] (file did NOT reach disk).
  //
  // These tests drive the (1)/(2)/(3) branches via the
  // [`crate::lm::load::arm_fsync_path_fault`] +
  // [`crate::lm::load::arm_fsync_dir_fault`] injectors. Both are
  // [`#[cfg(test)] pub(crate)`] (sibling-test access) and `Drop`-guarded
  // so a test panic leaves the thread clean.
  //
  // The shared fixture is the same shape as the other convert tests:
  // build a synthetic src with config + weights + a tokenizer the
  // load step accepts, then arm the relevant injector and call
  // `convert`. The skip counts target the post-copy fsync(s): the
  // save side makes 3 `fsync_path` + 3 `fsync_dir` calls before
  // copy_tokenizer_and_extras runs (config-stage fsync_path, shard
  // fsync_path, index fsync_path; shard-dir fsync_dir, index-dir
  // fsync_dir, config-commit fsync_dir), so skip=3 lets every
  // save-side fsync pass and fires on the first post-copy call.
  // ──────────────────────────────────────────────────────────────────

  /// Build the synthetic src+dst dir pair used by the post-copy
  /// fsync tests. Mirrors the other convert fixtures but extracted so
  /// the four tests don't each duplicate ~30 lines. Returns
  /// `(workdir, src, dst)` so the caller controls cleanup via
  /// `remove_dir_all(workdir)`.
  fn build_save_fixture(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let workdir = std::env::temp_dir().join(format!("mlxrs_convert_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&workdir);
    let src = workdir.join("src");
    let dst = workdir.join("dst");
    std::fs::create_dir_all(&src).unwrap();

    let plain_config = r#"{
      "model_type":"qwen3","hidden_size":16,"num_hidden_layers":1,
      "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
      "rope_theta":10000.0,"vocab_size":128,"tie_word_embeddings":false
    }"#;
    std::fs::write(src.join("config.json"), plain_config).unwrap();
    let blob: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "layer.weight".to_string(),
      Array::from_slice::<f32>(&blob, &(2usize, 64usize)).unwrap(),
    );
    crate::io::save_safetensors(&src.join("model.safetensors"), &weights).unwrap();

    // Tokenizer fixtures + extras the helper will copy.
    let tokenizer_json = include_str!("../../tests/fixtures/tokenizer.json");
    let tokenizer_config_json = include_str!("../../tests/fixtures/tokenizer_config.json");
    std::fs::write(src.join("tokenizer.json"), tokenizer_json).unwrap();
    std::fs::write(src.join("tokenizer_config.json"), tokenizer_config_json).unwrap();
    std::fs::write(
      src.join("special_tokens_map.json"),
      br#"{"eos_token":"</s>"}"#,
    )
    .unwrap();
    std::fs::write(src.join("generation_config.json"), br#"{"max_length":32}"#).unwrap();

    (workdir, src, dst)
  }

  /// Post-copy FILE fsync failure surfaces
  /// `Err(DurabilityWarning)` (NOT `ConvertPostSavePartial`).
  ///
  /// The post-copy fsync runs AFTER `std::fs::copy` returns Ok, so the
  /// file content IS on disk; only durability is uncertain. This
  /// matches the documented `DurabilityWarning { committed: true }`
  /// contract (the destination is logically complete). The
  /// `ConvertPostSavePartial` variant stays reserved for the case
  /// where `std::fs::copy` itself failed (file did NOT reach disk).
  ///
  /// Skip count: 3 — save makes 3 `fsync_path` calls (config-stage,
  /// shard tmp, index tmp) before copy_tokenizer_and_extras starts,
  /// so skip=3 lets every save-side fsync pass and fires on the
  /// first post-copy file fsync (the first
  /// TOKENIZER_EXTRA_FILES-resident file in the src, which is
  /// `tokenizer.json` by the const-array's order).
  #[test]
  fn convert_post_copy_file_fsync_failure_returns_durability_warning() {
    let (workdir, src, dst) = build_save_fixture("file_fsync_fail");

    // Arm the fsync_path injector: skip the 3 save-side calls, fail
    // on the 4th (first post-copy per-file fsync). The dir injector
    // is NOT armed, so the post-copy fsync_dir(dst) passes.
    let _guard = crate::lm::load::arm_fsync_path_fault(3);

    let r = convert(ConvertArgs {
      hf_path: src.clone(),
      mlx_path: dst.clone(),
      ..Default::default()
    });
    drop(_guard);

    // (b) Result is `Err(DurabilityWarning { committed: true })` —
    //     NOT `ConvertPostSavePartial`. A post-copy fsync warning
    //     means data IS on disk; the variant distinction matters
    //     because callers' recovery contract differs.
    match &r {
      Err(Error::DurabilityWarning(p)) => {
        assert!(
          p.committed(),
          "post-copy fsync warning carries committed=true (data IS on disk)"
        );
        // (c) The source message references the post-copy fsync (so
        //     the user can pinpoint the boundary that warned).
        let msg = p.source().to_string();
        assert!(
          msg.contains("injected fsync_path failure") || msg.contains("post-copy"),
          "source message references the post-copy fsync; got: {msg}"
        );
      }
      Err(Error::ConvertPostSavePartial(_)) => panic!(
        "post-copy FSYNC warning must NOT surface as ConvertPostSavePartial — \
         that variant is reserved for `std::fs::copy` itself failing (file \
         did NOT reach disk). A post-copy fsync warning means data IS on \
         disk; only durability is uncertain (DurabilityWarning contract)."
      ),
      other => panic!("expected Err(DurabilityWarning), got {other:?}"),
    }

    // (a) The copied file IS on disk byte-equal to its source. The
    //     fsync failed but `std::fs::copy` ran to completion BEFORE
    //     fsync was attempted, so the data is in the page cache and
    //     visible to any reader on this running kernel.
    for name in [
      "tokenizer.json",
      "tokenizer_config.json",
      "special_tokens_map.json",
      "generation_config.json",
    ] {
      assert!(
        dst.join(name).is_file(),
        "{name} IS on disk despite the post-copy fsync warning"
      );
      let a = std::fs::read(src.join(name)).unwrap();
      let b = std::fs::read(dst.join(name)).unwrap();
      assert_eq!(a, b, "{name} byte-equal at dst");
    }

    let _ = std::fs::remove_dir_all(&workdir);
  }

  /// Post-copy DIR fsync failure surfaces
  /// `Err(DurabilityWarning)` (NOT `ConvertPostSavePartial`).
  ///
  /// Same shape as the file-fsync test above — the only difference
  /// is WHICH fsync warns. The post-copy `fsync_dir(dst)` runs after
  /// EVERY per-file fsync has succeeded; a failure here means the
  /// new directory entries are visible (the renames committed) but
  /// the dir-inode metadata may not yet be durable. Data IS on
  /// disk — `DurabilityWarning` contract.
  ///
  /// Skip count: 3 — save makes 3 `fsync_dir` calls (shard publish,
  /// index publish, config commit) before copy_tokenizer_and_extras
  /// starts. skip=3 lets each pass and fires on the 4th call (the
  /// post-copy `fsync_dir(dst)`).
  #[test]
  fn convert_post_copy_dir_fsync_failure_returns_durability_warning() {
    let (workdir, src, dst) = build_save_fixture("dir_fsync_fail");

    // Arm the fsync_dir injector: skip the 3 save-side calls, fail
    // on the 4th (post-copy fsync_dir(dst)). The path injector is
    // NOT armed, so every per-file fsync_path passes.
    let _guard = crate::lm::load::arm_fsync_dir_fault(3);

    let r = convert(ConvertArgs {
      hf_path: src.clone(),
      mlx_path: dst.clone(),
      ..Default::default()
    });
    drop(_guard);

    match &r {
      Err(Error::DurabilityWarning(p)) => {
        assert!(
          p.committed(),
          "post-copy dir-fsync warning carries committed=true (data IS on disk)"
        );
        let msg = p.source().to_string();
        // The source message references the post-copy dir fsync —
        // either via the verbatim injector marker or the "post-copy"
        // tag added by the convert-side fold.
        assert!(
          msg.contains("injected fsync_dir failure") || msg.contains("post-copy fsync_dir"),
          "source message references the post-copy dir fsync; got: {msg}"
        );
      }
      Err(Error::ConvertPostSavePartial(_)) => panic!(
        "post-copy DIR fsync warning must NOT surface as ConvertPostSavePartial — \
         that variant is reserved for `std::fs::copy` itself failing. A \
         post-copy dir-fsync warning means data IS on disk (every file's \
         own fsync passed); only the dir-entry durability is uncertain."
      ),
      other => panic!("expected Err(DurabilityWarning), got {other:?}"),
    }

    // Files are on disk byte-equal — the per-file fsyncs all passed
    // and the renames committed.
    for name in [
      "tokenizer.json",
      "tokenizer_config.json",
      "special_tokens_map.json",
      "generation_config.json",
    ] {
      assert!(dst.join(name).is_file(), "{name} on disk");
      let a = std::fs::read(src.join(name)).unwrap();
      let b = std::fs::read(dst.join(name)).unwrap();
      assert_eq!(a, b, "{name} byte-equal at dst");
    }

    let _ = std::fs::remove_dir_all(&workdir);
  }

  /// BOTH the per-file fsync AND the post-copy dir fsync warn.
  /// Surfaces the typed aggregate
  /// [`Error::ConvertDurabilityWarnings`] with each warning carried
  /// in a separate `Option<std::io::Error>` field (no
  /// `io::Error::other(format!(...))` fold) so the caller can
  /// machine-detect WHICH boundaries warned via destructuring (a
  /// string-folded `DurabilityWarning` would hide typed access to the
  /// individual warnings).
  ///
  /// Skip counts: 3 on each injector. The path injector fires on
  /// the first post-copy file fsync; the dir injector fires on the
  /// post-copy `fsync_dir(dst)` call AFTER every per-file fsync
  /// completes. `copy_tokenizer_and_extras` records the first file
  /// fsync warning, runs to completion through every other file,
  /// then triggers the dir fsync — observing its warning too —
  /// and returns both in the typed `CopyDurabilityWarnings` shape.
  #[test]
  fn convert_post_copy_both_fsyncs_fail_combined_message() {
    let (workdir, src, dst) = build_save_fixture("both_fsyncs_fail");

    // Arm BOTH injectors. The path injector skips the 3 save-side
    // fsync_path calls (config-stage, shard tmp, index tmp) and
    // fires on the 4th (first post-copy file fsync). The dir
    // injector skips the 3 save-side fsync_dir calls (shard publish,
    // index publish, config commit) and fires on the 4th (post-copy
    // fsync_dir(dst)).
    let _path_guard = crate::lm::load::arm_fsync_path_fault(3);
    let _dir_guard = crate::lm::load::arm_fsync_dir_fault(3);

    let r = convert(ConvertArgs {
      hf_path: src,
      mlx_path: dst,
      ..Default::default()
    });
    drop(_path_guard);
    drop(_dir_guard);

    match &r {
      Err(Error::ConvertDurabilityWarnings(agg)) => {
        // (a) committed=true (variant is reachable only after the
        //     observable commit point — the index rename).
        assert!(agg.committed, "committed=true even when both fsyncs warn");

        // (b) Save side did NOT warn (no fsync_dir fault armed
        //     before save's own fsync calls — the dir injector's
        //     skip=3 deliberately steps past all 3 save-side
        //     fsync_dir calls).
        assert!(
          agg.save.is_none(),
          "save-side fsync passed (skip count steps past save's 3 \
           fsync_dir calls); got: {:?}",
          agg.save
        );

        // (c) Per-file fsync warning IS present + machine-readable
        //     via direct destructuring. `.kind()` returns a real
        //     io::ErrorKind (no string parse needed).
        let post_copy_file = agg
          .post_copy_file
          .as_ref()
          .expect("post_copy_file fsync warned (path injector fired on the 4th call)");
        let _ = post_copy_file.kind();
        assert!(
          post_copy_file
            .to_string()
            .contains("injected fsync_path failure"),
          "post_copy_file preserves the verbatim file-fsync io::Error \
           message (no string fold); got: {post_copy_file}"
        );

        // (d) Dir fsync warning IS present + machine-readable via
        //     direct destructuring.
        let post_copy_dir = agg
          .post_copy_dir
          .as_ref()
          .expect("post_copy_dir fsync warned (dir injector fired on the 4th call)");
        let _ = post_copy_dir.kind();
        assert!(
          post_copy_dir
            .to_string()
            .contains("injected fsync_dir failure"),
          "post_copy_dir preserves the verbatim dir-fsync io::Error \
           message (no string fold); got: {post_copy_dir}"
        );

        // (e) count() reports the multi-warning shape (exactly 2
        //     for this test).
        assert_eq!(
          agg.count(),
          2,
          "two non-None warning fields (post_copy_file + post_copy_dir)"
        );

        // (f) `std::error::Error::source()` walks the chain and
        //     reaches the FIRST non-None warning (priority order:
        //     save -> post_copy_file -> post_copy_dir). Here save is
        //     None, so the source is post_copy_file.
        let e: &dyn std::error::Error = r.as_ref().err().unwrap();
        let source = e.source().expect(
          "ConvertDurabilityWarnings has a source chain via the \
           inner aggregate's std::error::Error impl",
        );
        assert!(
          source.to_string().contains("injected fsync_path failure"),
          ".source() returns the FIRST non-None warning \
           (post_copy_file when save is None); got: {source}"
        );
      }
      Err(Error::DurabilityWarning(_)) => panic!(
        "both-fsyncs-warn surfaces the multi-warning aggregate \
         ConvertDurabilityWarnings, NOT the single-warning \
         DurabilityWarning shape (typed access to each \
         boundary)"
      ),
      Err(Error::ConvertPostSavePartial(_)) => panic!(
        "both-fsyncs-warn surfaces ConvertDurabilityWarnings, NOT \
         ConvertPostSavePartial (data IS on disk; only durability \
         uncertain on two boundaries)"
      ),
      other => panic!("expected Err(ConvertDurabilityWarnings), got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&workdir);
  }

  /// Happy-path: `convert() → Ok(())` implies every
  /// post-copy fsync was actually invoked. Asserted by reading the
  /// on-disk state: every copied file's bytes match the source AND
  /// the destination dir is observable. A regression that silently
  /// SKIPPED the post-copy fsyncs would still pass the on-disk
  /// byte-equality check (the fsync is durability-only, not content);
  /// to catch a silent-skip we ALSO verify that the fsync_path
  /// injector — armed with skip=0 to fire on the FIRST call — converts
  /// the otherwise-Ok happy path into a `DurabilityWarning`. If the
  /// post-copy fsyncs were never called (e.g. someone removed the
  /// fsync_path loop), the injector would only fire during save and
  /// the test would observe the save-side fsync, not the post-copy
  /// one. To isolate, we use skip=3 (past every save-side call) so
  /// the injector only has a chance to fire IF the post-copy fsync
  /// loop runs.
  #[test]
  fn convert_ok_implies_post_copy_fsyncs_called() {
    // ── Sub-test A: no injector → happy path returns Ok(()) AND
    //                every copied file is on disk byte-equal.
    {
      let (workdir, src, dst) = build_save_fixture("happy_path");
      let r = convert(ConvertArgs {
        hf_path: src.clone(),
        mlx_path: dst.clone(),
        ..Default::default()
      });
      assert!(
        matches!(r, Ok(())),
        "happy path returns Ok(()) — every fsync passes; got: {r:?}"
      );
      for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
      ] {
        assert!(
          dst.join(name).is_file(),
          "{name} on disk after happy-path convert"
        );
        let a = std::fs::read(src.join(name)).unwrap();
        let b = std::fs::read(dst.join(name)).unwrap();
        assert_eq!(a, b, "{name} byte-equal");
      }
      let _ = std::fs::remove_dir_all(&workdir);
    }

    // ── Sub-test B: arm fsync_path with skip=3 (past every save-side
    //                call). If the post-copy file fsync loop is
    //                actually called, the injector fires and convert
    //                returns DurabilityWarning. If a regression
    //                silently REMOVED the post-copy fsync_path call,
    //                the injector would still be armed at end-of-
    //                convert and the result would be Ok(()) — a
    //                clear signal that the fsync was skipped.
    {
      let (workdir, src, dst) = build_save_fixture("happy_path_spy_file");
      let _guard = crate::lm::load::arm_fsync_path_fault(3);
      let r = convert(ConvertArgs {
        hf_path: src,
        mlx_path: dst,
        ..Default::default()
      });
      drop(_guard);
      assert!(
        matches!(r, Err(Error::DurabilityWarning(_))),
        "fsync_path injector armed past every save-side call must be \
         observed by the post-copy file fsync loop — a silent removal \
         of that loop would leave the injector unfired and the result \
         Ok(()); got: {r:?}"
      );
      let _ = std::fs::remove_dir_all(&workdir);
    }

    // ── Sub-test C: arm fsync_dir with skip=3 (past every save-side
    //                call). Same reasoning as sub-test B but for the
    //                post-copy `fsync_dir(dst)` call. If a regression
    //                silently REMOVED the post-copy dir fsync, the
    //                injector would not fire and the result would be
    //                Ok(()).
    {
      let (workdir, src, dst) = build_save_fixture("happy_path_spy_dir");
      let _guard = crate::lm::load::arm_fsync_dir_fault(3);
      let r = convert(ConvertArgs {
        hf_path: src,
        mlx_path: dst,
        ..Default::default()
      });
      drop(_guard);
      assert!(
        matches!(r, Err(Error::DurabilityWarning(_))),
        "fsync_dir injector armed past every save-side call must be \
         observed by the post-copy `fsync_dir(dst)` call — a silent \
         removal of that call would leave the injector unfired and \
         the result Ok(()); got: {r:?}"
      );
      let _ = std::fs::remove_dir_all(&workdir);
    }
  }

  // ──────────────────────────────────────────────────────────────────
  // Multi-warning save+post-copy aggregate tests.
  //
  // A naive (save warned, post-copy fsync warned) branch would fold
  // the two io::Errors into a single free-form
  // `std::io::Error::other(format!(...))` message inside
  // `DurabilityWarning.source`, losing typed access. Instead,
  // multi-warning convert()s are routed through the typed
  // `Error::ConvertDurabilityWarnings(ConvertDurabilityWarnings { ... })`
  // aggregate so the caller can destructure each warning by field.
  //
  // Skip counts: the save side makes 3 `fsync_path` + 3 `fsync_dir`
  // calls before `copy_tokenizer_and_extras` runs. To make the save
  // SIDE warn AND a specific post-copy fsync also warn we arm the
  // fsync_dir injector at skip=2 (so save's 3rd fsync_dir call — the
  // config-commit dir fsync — fires the warning, surfacing the save's
  // save_model() through the DurabilityWarning path) PLUS the
  // post-copy injector at skip=3 (which targets the first post-copy
  // fsync_{path,dir} call).
  // ──────────────────────────────────────────────────────────────────

  /// Aggregate routing — save warned AND the post-copy dir fsync
  /// warned. Asserts the typed aggregate
  /// `Err(ConvertDurabilityWarnings { save: Some, post_copy_file:
  /// None, post_copy_dir: Some })` is surfaced (NOT a folded
  /// single-source `DurabilityWarning`).
  ///
  /// The fault injector's contract (`arm_fsync_dir_fault(skip)` fires
  /// on the (skip+1)-th call, single-shot drop guard) makes a
  /// two-specific-dir-fsync-failure topology impossible to set up via
  /// injection alone: arming `skip=2` fires once at save's 3rd
  /// fsync_dir call (the config commit) and the post-copy `fsync_dir(dst)`
  /// then runs unimpeded, since the guard has already fired. Failing
  /// TWO specific dir-fsync calls would require re-arming mid-convert,
  /// which isn't possible.
  ///
  /// So the (save: Some, post_copy_dir: Some) shape is exercised by
  /// direct construction here (and the convert()-side aggregate-count
  /// routing it drives), while the (save: Some, post_copy_file: Some)
  /// shape — realizable with two independent injectors (fsync_dir
  /// skip=2 for save + fsync_path skip=3 for post-copy file) — is
  /// covered end-to-end by
  /// `convert_save_and_post_copy_file_warn_returns_aggregate` below.
  #[test]
  fn convert_save_and_post_copy_dir_warn_returns_aggregate() {
    // Direct construction (no injector — the fault model can't
    // produce two specific fsync_dir failures with a single-shot
    // guard). Exercises the convert()-side ROUTING: aggregate's
    // count() == 2 must surface ConvertDurabilityWarnings, not
    // DurabilityWarning, regardless of which fields are Some.
    let save = std::io::Error::other("save-side fsync_dir warning");
    let post_copy_dir = std::io::Error::other("post-copy fsync_dir warning");
    let agg = crate::error::ConvertDurabilityWarnings {
      committed: true,
      save: Some(save),
      post_copy_file: None,
      post_copy_dir: Some(post_copy_dir),
    };
    assert_eq!(agg.count(), 2, "two non-None fields");
    let err: Error = agg.into();
    match &err {
      Error::ConvertDurabilityWarnings(agg) => {
        assert!(agg.committed);
        let save = agg
          .save
          .as_ref()
          .expect("save warning present (direct destructure)");
        assert!(save.to_string().contains("save-side fsync_dir warning"));
        assert!(
          agg.post_copy_file.is_none(),
          "post_copy_file is None: {:?}",
          agg.post_copy_file
        );
        let post_copy_dir = agg
          .post_copy_dir
          .as_ref()
          .expect("post_copy_dir warning present (direct destructure)");
        assert!(
          post_copy_dir
            .to_string()
            .contains("post-copy fsync_dir warning")
        );
        // first_warning() returns save (priority order:
        // save -> post_copy_file -> post_copy_dir).
        assert!(
          agg
            .first_warning()
            .unwrap()
            .to_string()
            .contains("save-side fsync_dir warning"),
          "first_warning() returns save when save is Some"
        );
      }
      other => panic!(
        "the aggregate-count routing must produce \
         ConvertDurabilityWarnings, NOT {other:?}"
      ),
    }
  }

  /// Save warned (save-side fsync_dir at skip=2 fires on save's 3rd
  /// fsync_dir call — the config-commit dir fsync) AND the post-copy
  /// file fsync warned (fsync_path at skip=3 fires on the first
  /// post-copy per-file fsync — the first TOKENIZER_EXTRA_FILES entry,
  /// `tokenizer.json`). Asserts the typed aggregate
  /// `Err(ConvertDurabilityWarnings { save: Some, post_copy_file: Some,
  /// post_copy_dir: None })` is surfaced (a folded shape would return a
  /// single `DurabilityWarning` with a string-folded source).
  #[test]
  fn convert_save_and_post_copy_file_warn_returns_aggregate() {
    let (workdir, src, dst) = build_save_fixture("save_and_postcopy_file_warn");

    // Arm fsync_dir at skip=2: save's 3 fsync_dir calls are
    // (shard publish, index publish, config commit) — skip 2 lets
    // the first two pass and fires on the 3rd (config commit) →
    // save_model surfaces a save-side DurabilityWarning that
    // save() then re-raises through the convert pipeline.
    // Arm fsync_path at skip=3: save's 3 fsync_path calls
    // (config-stage, shard tmp, index tmp) all pass, and the 4th
    // call — the first post-copy per-file fsync inside
    // copy_tokenizer_and_extras — fires.
    let _dir_guard = crate::lm::load::arm_fsync_dir_fault(2);
    let _path_guard = crate::lm::load::arm_fsync_path_fault(3);

    let r = convert(ConvertArgs {
      hf_path: src.clone(),
      mlx_path: dst.clone(),
      ..Default::default()
    });
    drop(_dir_guard);
    drop(_path_guard);

    match &r {
      Err(Error::ConvertDurabilityWarnings(agg)) => {
        assert!(agg.committed, "committed=true");
        // save warning present + machine-readable.
        let save = agg
          .save
          .as_ref()
          .expect("save warning is Some (fsync_dir injector skip=2 fired on save's config-commit dir fsync)");
        let _ = save.kind();
        assert!(
          save.to_string().contains("injected fsync_dir failure"),
          "save preserves the verbatim save-side fsync_dir io::Error \
           message; got: {save}"
        );
        // post_copy_file warning present + machine-readable.
        let post_copy_file = agg.post_copy_file.as_ref().expect(
          "post_copy_file warning is Some (fsync_path injector skip=3 \
           fired on the first post-copy per-file fsync)",
        );
        let _ = post_copy_file.kind();
        assert!(
          post_copy_file
            .to_string()
            .contains("injected fsync_path failure"),
          "post_copy_file preserves the verbatim post-copy fsync_path \
           io::Error message; got: {post_copy_file}"
        );
        // post_copy_dir is None (no dir injector armed past the
        // save-side calls — its guard already fired at skip=2).
        assert!(
          agg.post_copy_dir.is_none(),
          "post_copy_dir is None (single-shot fsync_dir guard fired \
           during save and is not re-armed for the post-copy dir fsync); \
           got: {:?}",
          agg.post_copy_dir
        );
        assert_eq!(agg.count(), 2, "two non-None warnings");
        // first_warning() returns save (priority order).
        assert!(
          agg
            .first_warning()
            .unwrap()
            .to_string()
            .contains("injected fsync_dir failure"),
          "first_warning() returns save when save is Some"
        );
      }
      Err(Error::DurabilityWarning(_)) => panic!(
        "save warned + post-copy file fsync warned MUST surface the \
         typed aggregate ConvertDurabilityWarnings, NOT the \
         single-source DurabilityWarning (which would fold the two via \
         io::Error::other(format!(...)))"
      ),
      other => panic!("expected Err(ConvertDurabilityWarnings), got {other:?}"),
    }

    // Tokenizer files ARE on disk byte-equal (`std::fs::copy`
    // succeeded; only fsync warned).
    for name in [
      "tokenizer.json",
      "tokenizer_config.json",
      "special_tokens_map.json",
      "generation_config.json",
    ] {
      assert!(dst.join(name).is_file(), "{name} on disk");
      let a = std::fs::read(src.join(name)).unwrap();
      let b = std::fs::read(dst.join(name)).unwrap();
      assert_eq!(a, b, "{name} byte-equal at dst");
    }

    let _ = std::fs::remove_dir_all(&workdir);
  }

  /// The aggregate's `std::error::Error::source()`
  /// chain is walkable and returns the FIRST non-None warning in
  /// deterministic `save -> post_copy_file -> post_copy_dir`
  /// priority order. Asserts (a) the chain is non-empty when any
  /// field is `Some`; (b) the first non-None per priority is what
  /// `.source()` returns; (c) every field is reachable via direct
  /// destructuring + machine-readable via `.kind()`.
  ///
  /// Constructed directly (no fixture) so this runs on every
  /// platform and exercises the trait impl in isolation.
  #[test]
  fn convert_durability_aggregate_error_chain_walkable() {
    // ── Case 1: save Some + post_copy_file Some + post_copy_dir Some
    //            → .source() returns save (highest priority).
    {
      let save = std::io::Error::other("SAVE warning");
      let pcf = std::io::Error::other("PCF warning");
      let pcd = std::io::Error::other("PCD warning");
      let agg = crate::error::ConvertDurabilityWarnings {
        committed: true,
        save: Some(save),
        post_copy_file: Some(pcf),
        post_copy_dir: Some(pcd),
      };
      assert_eq!(agg.count(), 3);
      let err: Error = agg.into();
      let e: &dyn std::error::Error = &err;
      let source = e
        .source()
        .expect("source chain non-empty (any non-None warning)");
      assert!(
        source.to_string().contains("SAVE warning"),
        ".source() returns the save warning (highest priority when \
         present); got: {source}"
      );
      // Destructure: every field directly reachable + .kind() works.
      if let Error::ConvertDurabilityWarnings(agg) = &err {
        assert_eq!(agg.save.as_ref().unwrap().kind(), std::io::ErrorKind::Other);
        assert_eq!(
          agg.post_copy_file.as_ref().unwrap().kind(),
          std::io::ErrorKind::Other
        );
        assert_eq!(
          agg.post_copy_dir.as_ref().unwrap().kind(),
          std::io::ErrorKind::Other
        );
        assert!(agg.save.as_ref().unwrap().to_string().contains("SAVE"));
        assert!(
          agg
            .post_copy_file
            .as_ref()
            .unwrap()
            .to_string()
            .contains("PCF")
        );
        assert!(
          agg
            .post_copy_dir
            .as_ref()
            .unwrap()
            .to_string()
            .contains("PCD")
        );
      } else {
        unreachable!("constructed ConvertDurabilityWarnings");
      }
    }

    // ── Case 2: save None + post_copy_file Some + post_copy_dir Some
    //            → .source() returns post_copy_file (next priority).
    {
      let pcf = std::io::Error::other("PCF only warning");
      let pcd = std::io::Error::other("PCD only warning");
      let agg = crate::error::ConvertDurabilityWarnings {
        committed: true,
        save: None,
        post_copy_file: Some(pcf),
        post_copy_dir: Some(pcd),
      };
      let err: Error = agg.into();
      let e: &dyn std::error::Error = &err;
      let source = e.source().expect("source chain non-empty");
      assert!(
        source.to_string().contains("PCF only warning"),
        ".source() returns post_copy_file when save is None (next \
         priority); got: {source}"
      );
    }

    // ── Case 3: save None + post_copy_file None + post_copy_dir Some
    //            → .source() returns post_copy_dir (last priority).
    {
      let pcd = std::io::Error::other("PCD lone warning");
      let agg = crate::error::ConvertDurabilityWarnings {
        committed: true,
        save: None,
        post_copy_file: None,
        post_copy_dir: Some(pcd),
      };
      let err: Error = agg.into();
      let e: &dyn std::error::Error = &err;
      let source = e.source().expect("source chain non-empty");
      assert!(
        source.to_string().contains("PCD lone warning"),
        ".source() returns post_copy_dir when both higher-priority \
         fields are None; got: {source}"
      );
    }

    // ── Case 4: all None → .source() returns None. The aggregate's
    //            Display still works (the committed=true tag is in
    //            the message).
    {
      let agg = crate::error::ConvertDurabilityWarnings {
        committed: true,
        save: None,
        post_copy_file: None,
        post_copy_dir: None,
      };
      assert_eq!(agg.count(), 0);
      let err: Error = agg.into();
      let e: &dyn std::error::Error = &err;
      assert!(
        e.source().is_none(),
        ".source() is None when every field is None"
      );
      assert!(
        err.to_string().contains("committed=true"),
        "Display carries the committed=true tag; got: {err}"
      );
    }
  }

  // ─── io::Error kind preserved end-to-end ─────────
  //
  // The convert()-side aggregate carries an
  // `Option<std::io::Error>` per fsync boundary that is advertised as
  // machine-readable (callers branch on `.kind()` to disambiguate
  // ENOSPC / EIO / PermissionDenied / ...). Routing through
  // [`crate::lm::load::fsync_path`] (which returns
  // [`crate::Error::Backend`] — string-wrapping the underlying
  // io::Error and losing its kind) and then re-wrapping the message via
  // [`std::io::Error::other`] would collapse EVERY post_copy_file
  // warning's kind to [`std::io::ErrorKind::Other`]. Instead we route
  // through the kind-preserving sibling
  // [`crate::lm::load::fsync_path_io`] (returns
  // [`std::io::Result<()>`]) so the underlying kind survives intact.
  //
  // The save-side warning already preserves kind end-to-end
  // ([`crate::lm::load::fsync_dir`] returns [`std::io::Result<()>`]
  // natively and the save → save_warning path carries the raw
  // io::Error via [`crate::lm::load::CommitOutcome::CommittedWithDurabilityWarning`]).
  // The post-copy DIR warning also already preserves kind (the
  // post-copy `fsync_dir(dst)` returns raw io::Error directly into
  // `warnings.post_copy_dir`). The three tests below verify ALL THREE
  // boundaries (save / post_copy_file / post_copy_dir) preserve the
  // injected kind so a regression on ANY one is caught.
  //
  // Driven via the kind-injecting [`arm_fsync_path_fault_with_kind`] /
  // [`arm_fsync_dir_fault_with_kind`] sibling injectors — they extend
  // the single-arg variants with an [`std::io::ErrorKind`]
  // override so the test can inject a SPECIFIC non-`Other` kind
  // (defaulting to `Other` would mask the bug — a kind-collapsing code
  // path also produces `Other`, so a kind-equality assertion would pass
  // even with the regression).

  /// Post-copy per-file fsync warning preserves the
  /// underlying [`std::io::ErrorKind`] (NOT collapsed to `Other`).
  ///
  /// Drives the same path as
  /// `convert_post_copy_file_fsync_failure_returns_durability_warning`
  /// but injects [`std::io::ErrorKind::PermissionDenied`] specifically
  /// — the assertion `agg.post_copy_file.unwrap().kind() ==
  /// PermissionDenied` (NOT `Other`) is the regression detector. A
  /// kind-collapsing build would unwrap the injected io::Error via
  /// `Err(Error::Backend(message)) => Err(io::Error::other(message))`
  /// inside `copy_tokenizer_and_extras`'s `fsync_copied` closure,
  /// collapsing the kind to `Other`.
  ///
  /// Skip count: 3 — past every save-side fsync_path call (config-
  /// stage, shard tmp, index tmp), fires on the 4th call (first post-
  /// copy per-file fsync). The dir injector is NOT armed so the only
  /// non-None field is `post_copy_file` and the single-warning
  /// surface is the existing [`Error::DurabilityWarning`] shape
  /// (count() == 1 takes the single-source branch).
  #[test]
  fn convert_post_copy_file_warning_preserves_io_error_kind() {
    let (workdir, src, dst) = build_save_fixture("post_copy_file_kind");

    // Arm the path injector with skip=3 + a SPECIFIC non-`Other` kind.
    // A kind-collapsing path would force `Other` regardless of what the
    // injector produces (the convert()-side `fsync_copied` re-wrapped
    // via `io::Error::other(message)` — kind unconditionally `Other`).
    // Routing through `fsync_path_io` lets the kind survive into the
    // typed aggregate's `post_copy_file` field.
    let _guard =
      crate::lm::load::arm_fsync_path_fault_with_kind(3, std::io::ErrorKind::PermissionDenied);

    let r = convert(ConvertArgs {
      hf_path: src,
      mlx_path: dst,
      ..Default::default()
    });
    drop(_guard);

    // Single warning → existing single-source DurabilityWarning shape.
    // The `source` field is the raw io::Error from `post_copy_file` —
    // its `.kind()` must equal the INJECTED kind, NOT `Other`.
    match &r {
      Err(Error::DurabilityWarning(p)) => {
        assert!(p.committed(), "committed=true (data IS on disk)");
        assert_eq!(
          p.source().kind(),
          std::io::ErrorKind::PermissionDenied,
          "post_copy_file warning preserves the injected ErrorKind \
           (PermissionDenied) end-to-end — an \
           `io::Error::other(message)` re-wrap would collapse this to \
           ErrorKind::Other; got: {:?} ({})",
          p.source().kind(),
          p.source()
        );
        assert!(
          p.source()
            .to_string()
            .contains("injected fsync_path failure"),
          "source preserves the verbatim injector message: {}",
          p.source()
        );
      }
      other => panic!("expected Err(DurabilityWarning), got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&workdir);
  }

  /// Post-copy directory fsync warning preserves the
  /// underlying [`std::io::ErrorKind`]. The dir path already returned
  /// raw [`std::io::Result`] (no `Error::Backend` wrap) so this code
  /// path is kind-preserving HERE — this test guards
  /// against a regression that adds a fold (e.g. a future "uniform
  /// shape with fsync_path" refactor that wraps in
  /// `io::Error::other(message)`).
  ///
  /// Skip count: 3 — past every save-side fsync_dir call (shard
  /// publish, index publish, config commit), fires on the 4th call
  /// (post-copy `fsync_dir(dst)`).
  #[test]
  fn convert_post_copy_dir_warning_preserves_io_error_kind() {
    let (workdir, src, dst) = build_save_fixture("post_copy_dir_kind");

    // Use `StorageFull` (commonly ENOSPC) — a kind a real durability
    // recovery flow MUST be able to distinguish from PermissionDenied
    // to decide whether to retry / free space / surface to the user.
    let _guard = crate::lm::load::arm_fsync_dir_fault_with_kind(3, std::io::ErrorKind::StorageFull);

    let r = convert(ConvertArgs {
      hf_path: src,
      mlx_path: dst,
      ..Default::default()
    });
    drop(_guard);

    match &r {
      Err(Error::DurabilityWarning(p)) => {
        assert!(p.committed(), "committed=true (data IS on disk)");
        assert_eq!(
          p.source().kind(),
          std::io::ErrorKind::StorageFull,
          "post_copy_dir warning preserves the injected ErrorKind \
           (StorageFull / ENOSPC) end-to-end; got: {:?} ({})",
          p.source().kind(),
          p.source()
        );
        assert!(
          p.source()
            .to_string()
            .contains("injected fsync_dir failure"),
          "source preserves the verbatim injector message: {}",
          p.source()
        );
      }
      other => panic!("expected Err(DurabilityWarning), got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&workdir);
  }

  /// Save-side warning preserves the underlying
  /// [`std::io::ErrorKind`]. Drives the same path as
  /// `convert_durability_warning_still_copies_tokenizer_and_returns_warning`
  /// but asserts the SPECIFIC injected kind (rather than just the
  /// "DurabilityWarning observed" branch). The save side already
  /// preserves kind end-to-end (the
  /// [`crate::lm::load::CommitOutcome::CommittedWithDurabilityWarning`]
  /// variant carries the raw io::Error from [`crate::lm::load::fsync_dir`]);
  /// this test guards against a regression that adds a fold (e.g. a
  /// "uniform shape" refactor in [`crate::lm::load::save`] that wraps
  /// via `io::Error::other(message)`).
  ///
  /// Skip count: 1 — shard fsync passes, the index-fsync (save's 2nd
  /// fsync_dir call) fails → save_model returns
  /// `CommittedWithDurabilityWarning` → save() surfaces a final
  /// `Err(DurabilityWarning)` after also committing the config.
  /// No post-copy fsync is faulted, so the only non-None field is
  /// `save` and the single-warning surface is the existing
  /// `Error::DurabilityWarning` shape (count() == 1).
  #[test]
  fn convert_save_warning_preserves_io_error_kind() {
    let (workdir, src, dst) = build_save_fixture("save_kind");

    // Inject ConnectionReset just to ensure the test is asserting a
    // SPECIFIC kind that's neither `Other` (the default) nor
    // PermissionDenied / StorageFull (used by the other two kind tests)
    // — a fold that collapsed every kind to `Other` (or any
    // non-ConnectionReset default) would fail this assertion. The
    // kind is otherwise arbitrary — fsync errors in real life are
    // commonly `Other` (errno EIO with no narrower std mapping), so
    // the realism of the kind isn't the point; preservation is.
    let _guard =
      crate::lm::load::arm_fsync_dir_fault_with_kind(1, std::io::ErrorKind::ConnectionReset);

    let r = convert(ConvertArgs {
      hf_path: src,
      mlx_path: dst,
      ..Default::default()
    });
    drop(_guard);

    match &r {
      Err(Error::DurabilityWarning(p)) => {
        assert!(p.committed(), "committed=true (save committed)");
        assert_eq!(
          p.source().kind(),
          std::io::ErrorKind::ConnectionReset,
          "save-side warning preserves the injected ErrorKind \
           (ConnectionReset) end-to-end through \
           CommitOutcome::CommittedWithDurabilityWarning → \
           Error::DurabilityWarning → convert's committed_warning \
           stash → ConvertDurabilityWarnings.save; got: {:?} ({})",
          p.source().kind(),
          p.source()
        );
        assert!(
          p.source()
            .to_string()
            .contains("injected fsync_dir failure"),
          "source preserves the verbatim injector message: {}",
          p.source()
        );
      }
      other => panic!("expected Err(DurabilityWarning), got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&workdir);
  }

  // ─── call-site wrap adds path + operation context ──
  //
  // `fsync_path_io` returns the raw [`std::io::Error`]
  // from [`std::fs::File::open`] / [`std::fs::File::sync_all`] without
  // adding path/operation context. For REAL failures (not the injected
  // ones whose message happens to include the path), callers get only
  // OS-level text like `"No such file or directory (os error 2)"` or
  // `"Input/output error (os error 5)"` — no way to tell WHICH copied
  // tokenizer file warned or whether the failure was the reopen vs the
  // sync_all.
  //
  // The call site (in `copy_tokenizer_and_extras`) wraps the error
  // with `"copy_tokenizer_and_extras: fsync <dst-path> failed: <inner>"`
  // via [`std::io::Error::new`] (kind preserved).
  //
  // The two tests below cover BOTH halves of the assurance:
  //
  // 1. `convert_post_copy_file_warning_includes_destination_path` —
  //    drives the injected-kind path and asserts the wrap
  //    added BOTH the destination filename AND the operation tag. This
  //    verifies the wrap fires on the standard injected path
  //    (regression detector: if the wrap is removed, the assertion
  //    that the message contains the operation tag fails because the
  //    injector's own message doesn't include `"copy_tokenizer_and_extras:
  //    fsync"`).
  //
  // 2. `convert_post_copy_file_real_failure_includes_path_and_kind` —
  //    drives the "real failure" injector (which removes the
  //    target file then falls through to the natural
  //    [`std::fs::File::open`] for an AUTHENTIC OS-level
  //    [`std::io::ErrorKind::NotFound`] with NO path in the message).
  //    Asserts the wrap added path + operation context to a context-
  //    free OS error — proving the path-context assertion isn't passing
  //    only because the injector pre-formats the path into its message.

  /// The post-copy per-file fsync warning message
  /// contains the destination filename + operation tag (added by the
  /// call-site wrap in `copy_tokenizer_and_extras`).
  ///
  /// Drives the same injected-kind path as
  /// `convert_post_copy_file_warning_preserves_io_error_kind` but adds
  /// the path-context assertions. A regression that removes the
  /// call-site wrap would still pass the kind assertion (because
  /// `fsync_path_io` returns the raw io::Error directly) but would FAIL
  /// the operation-tag assertion because the injector's own message
  /// (`"injected fsync_path failure for ..."`) does NOT contain
  /// `"copy_tokenizer_and_extras: fsync"`.
  ///
  /// Skip count: 3 — same shape as the kind test. The first post-copy
  /// per-file fsync is for `tokenizer.json` (first
  /// `TOKENIZER_EXTRA_FILES` entry that exists at src). Asserts the
  /// warning message contains BOTH `"tokenizer.json"` and the
  /// operation tag `"copy_tokenizer_and_extras: fsync"`.
  #[test]
  fn convert_post_copy_file_warning_includes_destination_path() {
    let (workdir, src, dst) = build_save_fixture("post_copy_file_path_ctx");

    let _guard =
      crate::lm::load::arm_fsync_path_fault_with_kind(3, std::io::ErrorKind::PermissionDenied);

    let r = convert(ConvertArgs {
      hf_path: src,
      mlx_path: dst.clone(),
      ..Default::default()
    });
    drop(_guard);

    match &r {
      Err(Error::DurabilityWarning(p)) => {
        assert!(p.committed(), "committed=true (data IS on disk)");
        // Kind-preservation contract intact: the wrap uses
        // io::Error::new (kind preserved), not io::Error::other (which
        // would collapse to `Other`).
        assert_eq!(
          p.source().kind(),
          std::io::ErrorKind::PermissionDenied,
          "the wrap preserves the underlying ErrorKind; \
           got: {:?} ({})",
          p.source().kind(),
          p.source()
        );
        let msg = p.source().to_string();
        // The wrap added the operation tag — the injector's own message
        // does not contain this string, so this assertion fails if the
        // call-site wrap is removed.
        assert!(
          msg.contains("copy_tokenizer_and_extras: fsync"),
          "the wrap adds the operation tag `copy_tokenizer_and_extras: \
           fsync ...`; got: {msg}"
        );
        // The wrap added the destination filename (tokenizer.json is
        // the first entry of TOKENIZER_EXTRA_FILES present in src per
        // build_save_fixture).
        assert!(
          msg.contains("tokenizer.json"),
          "wrap names the destination filename (tokenizer.json); got: \
           {msg}"
        );
        // The wrap names the FULL destination path (not just the
        // basename) so users can `cd` to the parent dir.
        let expected_dst = dst.join("tokenizer.json");
        assert!(
          msg.contains(&expected_dst.display().to_string()),
          "wrap names the full destination path ({}); got: {msg}",
          expected_dst.display()
        );
        // Kind-preservation contract intact: the wrap PRESERVES the
        // inner injector message via the trailing `: {e}` interpolation.
        assert!(
          msg.contains("injected fsync_path failure"),
          "wrap preserves the verbatim inner io::Error message; got: \
           {msg}"
        );
      }
      other => panic!("expected Err(DurabilityWarning), got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&workdir);
  }

  /// REAL OS-level fsync failure (not the synthesized
  /// "injected fsync_path failure for {path}" string from the standard
  /// injector) carries path + operation context via the call-site wrap.
  ///
  /// Drives the "remove_then_fail" injector — on the 4th
  /// `fsync_path_inner` call (past the 3 save-side calls), the injector
  /// removes the target file then FALLS THROUGH to the natural
  /// [`std::fs::File::open`] which returns the AUTHENTIC OS-level
  /// [`std::io::ErrorKind::NotFound`] error. That error's message is
  /// the platform OS text (`"No such file or directory (os error 2)"`
  /// on Unix) — it does NOT include the path. Without the call-site
  /// wrap, the recorded warning would surface this context-free
  /// string to the caller, who would have no way to tell WHICH copied
  /// tokenizer file warned.
  ///
  /// Asserts:
  ///   (a) result is `Err(DurabilityWarning)` (single warning →
  ///       single-source shape) with `committed: true`;
  ///   (b) source.kind() is the REAL OS kind (NotFound) — not the
  ///       injector-default Other (proves we're observing the natural
  ///       failure, not a synthesized one);
  ///   (c) source.to_string() contains the destination filename
  ///       (tokenizer.json — added by the wrap);
  ///   (d) source.to_string() does NOT contain `"injected fsync_path
  ///       failure"` (the standard injector's marker — proves this is
  ///       a real OS failure path, not the synthesized one);
  ///   (e) source.to_string() contains `"copy_tokenizer_and_extras:
  ///       fsync"` (the operation tag — added by the wrap).
  #[test]
  fn convert_post_copy_file_real_failure_includes_path_and_kind() {
    let (workdir, src, dst) = build_save_fixture("post_copy_file_real_fail");

    // Skip 3 (save-side fsync_path calls); fire on the 4th (first
    // post-copy per-file fsync — tokenizer.json). The injector removes
    // tokenizer.json then falls through to File::open which returns
    // the real OS NotFound error WITHOUT path in the message.
    let _guard = crate::lm::load::arm_fsync_path_fault_remove_then_fail(3);

    let r = convert(ConvertArgs {
      hf_path: src,
      mlx_path: dst.clone(),
      ..Default::default()
    });
    drop(_guard);

    match &r {
      Err(Error::DurabilityWarning(p)) => {
        // (a) committed: true.
        assert!(
          p.committed(),
          "post-copy fsync warning carries committed=true (the file's \
           bytes reached disk via std::fs::copy BEFORE the fsync ran; \
           the durability-uncertain window is between copy returning \
           Ok and the fsync completing)"
        );
        // (b) Real OS kind (NotFound), not the injector-default Other.
        // This is the proof-of-real-failure: the standard
        // `arm_fsync_path_fault` injector synthesizes a
        // `ErrorKind::Other` failure; only the natural File::open path
        // produces ErrorKind::NotFound.
        assert_eq!(
          p.source().kind(),
          std::io::ErrorKind::NotFound,
          "real OS failure path produces the natural File::open kind \
           (NotFound) — observed {:?} ({})",
          p.source().kind(),
          p.source()
        );
        let msg = p.source().to_string();
        // (c) Destination filename is in the wrap.
        assert!(
          msg.contains("tokenizer.json"),
          "wrap names the destination filename (tokenizer.json) so the \
           caller can pinpoint WHICH copied file warned; got: {msg}"
        );
        let expected_dst = dst.join("tokenizer.json");
        assert!(
          msg.contains(&expected_dst.display().to_string()),
          "wrap names the full destination path ({}) so the caller can \
           navigate to the failing file directly; got: {msg}",
          expected_dst.display()
        );
        // (d) Operation tag is in the wrap.
        assert!(
          msg.contains("copy_tokenizer_and_extras: fsync"),
          "wrap adds the operation tag so a real OS error (which has no \
           path embedded — the message is OS-level text like `No such \
           file or directory (os error 2)`) can be traced back to the \
           post-copy fsync step in copy_tokenizer_and_extras; got: {msg}"
        );
        // (e) Confirms this is the REAL OS path: the standard injector's
        // marker is absent. Without this assertion the test could pass
        // even if the remove_then_fail injector regressed to using the
        // synthesized marker (which would coincidentally include the
        // path and mask a missing call-site wrap).
        assert!(
          !msg.contains("injected fsync_path failure"),
          "this test drives the REAL OS failure path (File::open on a \
           removed file); the standard injector's synthesized marker \
           `injected fsync_path failure` must NOT appear — its presence \
           would mean the remove_then_fail injector regressed to using \
           the synthesized error path; got: {msg}"
        );
      }
      other => panic!(
        "expected Err(DurabilityWarning) carrying the real OS NotFound \
         (kind from File::open on a removed file) wrapped with the operation path + \
         operation context, got {other:?}"
      ),
    }

    let _ = std::fs::remove_dir_all(&workdir);
  }

  // ──────────────────────────────────────────────────────────────────
  // Helper-level coverage: error / continue / pass-through branches in
  // `mixed_quant_predicate`, `resolve_target_dtype`,
  // `cast_floats_to_dtype`, `build_predicate_decisions`,
  // `build_quantize_config`, `strip_quantization_blocks`,
  // `copy_tokenizer_and_extras`, and `paths_are_same`.
  //
  // Each oracle is independent + closed-form: known JSON / path / shape
  // inputs drive the branch, and the expected output is computed by hand
  // (typed-error matching for the error arms). No fn-under-test is used
  // to produce its own expected.
  // ──────────────────────────────────────────────────────────────────

  /// `mixed_quant_predicate` — a `down_proj`-bearing key with NO numeric
  /// layer-index segment trips the `.ok_or_else(...)` arm
  /// (`convert.rs` ~452-457): an [`Error::LayerKeyed`] wrapping an
  /// [`Error::InvariantViolation`]. Mirrors `convert.py:43-45`'s
  /// `if k.isdigit(): break` finding no digit segment.
  #[test]
  fn mixed_quant_predicate_no_numeric_segment_is_layer_keyed_error() {
    // One down_proj key whose every `.`-segment is non-numeric — the
    // `position(... all ascii_digit)` scan returns None.
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "model.decoder.down_proj.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
    );

    match mixed_quant_predicate(MixedQuantRecipe::Mixed3_6, &weights, 64) {
      Err(Error::LayerKeyed(p)) => {
        // The offending path is recorded as the layer key (the
        // `.weight` suffix is stripped by the builder before the scan).
        assert_eq!(
          p.layer(),
          "model.decoder.down_proj",
          "LayerKeyed names the offending down_proj path"
        );
        // The wrapped sub-error is the structural InvariantViolation.
        match p.inner() {
          Error::InvariantViolation(iv) => {
            assert!(
              iv.context().contains("down_proj"),
              "inner InvariantViolation names the down_proj path context: {}",
              iv.context()
            );
            assert!(
              iv.requirement().contains("numeric layer-index segment"),
              "inner requirement quotes the numeric-segment rule: {}",
              iv.requirement()
            );
          }
          other => panic!("expected inner InvariantViolation, got {other:?}"),
        }
      }
      other => panic!("expected Err(LayerKeyed), got {other:?}"),
    }
  }

  /// `resolve_target_dtype` — a config string that is NOT valid JSON
  /// takes the `Err(_) => Ok(None)` arm (`convert.rs` ~1014). Documented
  /// as "unreachable in practice" (the config is parsed once already)
  /// but the defensive arm must still resolve to "no cast". `explicit`
  /// is `None` so the dtype-gate above is skipped and the parse arm is
  /// reached.
  #[test]
  fn resolve_target_dtype_unparseable_config_is_none() {
    // `{` alone is not valid JSON → serde_json::from_str returns Err.
    assert_eq!(resolve_target_dtype(None, "{ not json").unwrap(), None);
    // A bare non-JSON token too.
    assert_eq!(resolve_target_dtype(None, "@@@").unwrap(), None);
  }

  /// `resolve_target_dtype` — `torch_dtype` is present but an UNKNOWN
  /// spelling, so `parse_conversion_dtype` returns `None`; the
  /// `text_config.dtype` fallback (also absent/unknown) then yields the
  /// final `Ok(None)`. Exercises the fall-through past both `if let`
  /// chains (the `&& let Some(d) = ...` short-circuits on the parse).
  #[test]
  fn resolve_target_dtype_known_key_unknown_value_falls_through() {
    // torch_dtype present, value unknown → first chain's parse is None.
    // text_config present, dtype unknown → second chain's parse is None.
    let cfg = r#"{"torch_dtype":"int8","text_config":{"dtype":"qint4"}}"#;
    assert_eq!(resolve_target_dtype(None, cfg).unwrap(), None);
    // text_config present but WITHOUT a `dtype` key → the inner
    // `.get("dtype")` is None, second chain short-circuits.
    let cfg2 = r#"{"text_config":{"hidden_size":16}}"#;
    assert_eq!(resolve_target_dtype(None, cfg2).unwrap(), None);
  }

  /// `cast_floats_to_dtype` (`convert.rs` ~1052-1066) — floating arrays
  /// whose dtype differs from `target` are cast; floating arrays already
  /// at `target` AND non-floating arrays are passed through untouched.
  ///
  /// Oracle: build a known mix — an F32 weight (must cast to F16), a
  /// pre-F16 weight (must stay, the `dt != target` arm is false), and a
  /// non-floating (U32) weight (the `is_floating` arm is false). Assert
  /// each output array's dtype against the hand-computed expectation.
  #[test]
  fn cast_floats_to_dtype_casts_floats_passes_through_rest() {
    let mut weights: Weights = HashMap::new();
    // F32 → must be cast to F16.
    weights.insert(
      "a.weight".to_string(),
      Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &(2usize, 2usize)).unwrap(),
    );
    // Already F16 → must be left as F16 (the `dt != target` guard skips
    // the cast — exercises the `else` arm for a floating array).
    weights.insert(
      "b.weight".to_string(),
      Array::from_slice::<f32>(&[5.0_f32, 6.0], &(1usize, 2usize))
        .unwrap()
        .astype(Dtype::F16)
        .unwrap(),
    );
    // Non-floating (U32) → `is_floating` is false, passed through.
    weights.insert(
      "c.weight".to_string(),
      Array::from_slice::<f32>(&[7.0_f32, 8.0], &(1usize, 2usize))
        .unwrap()
        .astype(Dtype::U32)
        .unwrap(),
    );

    let out = cast_floats_to_dtype(weights, Dtype::F16).unwrap();
    assert_eq!(out.len(), 3, "every key is preserved");
    assert_eq!(
      out.get("a.weight").unwrap().dtype().unwrap(),
      Dtype::F16,
      "F32 weight cast to the F16 target"
    );
    assert_eq!(
      out.get("b.weight").unwrap().dtype().unwrap(),
      Dtype::F16,
      "already-F16 weight left at F16 (no redundant cast)"
    );
    assert_eq!(
      out.get("c.weight").unwrap().dtype().unwrap(),
      Dtype::U32,
      "non-floating U32 weight passed through untouched"
    );
  }

  /// `build_predicate_decisions` — `group_size <= 0` short-circuits to an
  /// EMPTY decision map (`convert.rs` ~1116) so the divisor is never
  /// used. A predicate IS supplied (so we pass the `let Some(pred)`
  /// guard) and the weights ARE structurally eligible — proving the
  /// early-return is the `group_size <= 0` guard, not the no-predicate
  /// or no-eligible-layer path.
  #[test]
  fn build_predicate_decisions_nonpositive_group_size_returns_empty() {
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "layer.a.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
    );
    let pred = CountingPredicate::new();

    // group_size == 0 → empty map, predicate never invoked.
    let d0 = build_predicate_decisions(Some(&pred), &weights, 0);
    assert!(d0.is_empty(), "group_size==0 yields an empty decision map");
    // group_size < 0 → same.
    let dn = build_predicate_decisions(Some(&pred), &weights, -8);
    assert!(dn.is_empty(), "negative group_size yields an empty map");
    // The predicate was NEVER called (the guard returns before the walk).
    assert_eq!(
      pred.max_count(),
      0,
      "the predicate is not invoked when group_size <= 0"
    );
  }

  /// `build_predicate_decisions` — the two per-key `continue` arms
  /// (`convert.rs` ~1121, ~1135): a key WITHOUT a `.weight` suffix is
  /// skipped (strip_suffix → None), and a `.weight` key whose last axis
  /// is NOT divisible by `group_size` is skipped (the structural shape
  /// gate). Only the eligible key reaches the predicate.
  #[test]
  fn build_predicate_decisions_skips_non_weight_and_indivisible_keys() {
    let mut weights: Weights = HashMap::new();
    // (1) No `.weight` suffix → skipped at the strip_suffix `continue`.
    weights.insert(
      "layer.a.bias".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 64], &(1usize, 64usize)).unwrap(),
    );
    // (2) `.weight` but last axis 50 is NOT divisible by group_size 64 →
    //     skipped at the `last % gs != 0` `continue`.
    weights.insert(
      "layer.b.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 100], &(2usize, 50usize)).unwrap(),
    );
    // (3) `.weight`, rank < 2 → skipped at the `shape.len() < 2`
    //     `continue` (covered alongside).
    weights.insert(
      "layer.c.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 64], &(64usize,)).unwrap(),
    );
    // (4) Eligible: `.weight`, rank 2, last axis 64 divisible by 64.
    weights.insert(
      "layer.d.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
    );

    let pred = CountingPredicate::new();
    let decisions = build_predicate_decisions(Some(&pred), &weights, 64);

    // ONLY layer.d reached the predicate / decision map.
    assert_eq!(decisions.len(), 1, "exactly the one eligible layer");
    assert!(
      matches!(decisions.get("layer.d"), Some(Some(_))),
      "layer.d recorded with the predicate's decision"
    );
    assert_eq!(pred.paths_seen(), vec!["layer.d".to_string()]);
    // The skipped keys never appear.
    for skipped in ["layer.a", "layer.b", "layer.c"] {
      assert!(
        !decisions.contains_key(skipped),
        "{skipped} skipped before reaching the predicate"
      );
    }
  }

  /// `build_quantize_config` — a source config JSON that is NOT an object
  /// (e.g. a top-level array) trips the
  /// `let serde_json::Value::Object(..) else` arm (`convert.rs`
  /// ~1183-1185): [`Error::InvariantViolation`] "must be an object".
  #[test]
  fn build_quantize_config_non_object_config_is_error() {
    let weights: Weights = HashMap::new();
    let decisions: PredicateDecisions = HashMap::new();
    match build_quantize_config("[1,2,3]", 64, 4, QuantMode::Affine, &decisions, &weights) {
      Err(Error::InvariantViolation(p)) => {
        assert!(
          p.context().contains("source config JSON"),
          "context names the source config: {}",
          p.context()
        );
        assert_eq!(p.requirement(), "must be an object");
      }
      other => panic!("expected Err(InvariantViolation), got {other:?}"),
    }
  }

  /// `build_quantize_config` — the `Quantize(q)` per-layer override arm
  /// (`convert.rs` ~1237 + the nested-object block ~1286-1297). With a
  /// NON-fine-grained source config (no `"quantization"` key), a
  /// per-layer decision whose params DIFFER from the global default is
  /// written as an explicit nested override; the emitted `quantization`
  /// block carries the nested `{group_size,bits,mode}`.
  ///
  /// Oracle: decisions = {"layer.a": Some(q_diff)} where q_diff differs
  /// from the global (group_size 64, bits 4) by bits=8. Expect the live
  /// `PerLayerQuantization` to carry exactly that override, and the
  /// re-serialized config's `quantization["layer.a"]` to be the nested
  /// object with bits=8.
  #[test]
  fn build_quantize_config_quantize_override_emits_nested_object() {
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "layer.a.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
    );
    let q_diff = Quantization {
      group_size: 64,
      bits: 8, // differs from the global bits=4 → override is emitted
      mode: QuantMode::Affine,
    };
    let mut decisions: PredicateDecisions = HashMap::new();
    decisions.insert("layer.a".to_string(), Some(q_diff));

    // NON-fine-grained source (no "quantization" key).
    let (live, text) =
      build_quantize_config("{}", 64, 4, QuantMode::Affine, &decisions, &weights).unwrap();

    // Live config: global default + the per-layer override.
    assert_eq!(
      live.quantization,
      Some(Quantization {
        group_size: 64,
        bits: 4,
        mode: QuantMode::Affine
      }),
      "global default is the call's (group_size, bits, mode)"
    );
    assert_eq!(
      live.per_layer_ref().get("layer.a"),
      Some(&QuantizationOption::Quantize(q_diff)),
      "layer.a carries the differing-params Quantize override"
    );

    // Re-serialized config text: global keys + nested override object.
    let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
    let block = parsed.get("quantization").unwrap();
    assert_eq!(block.get("group_size").and_then(|v| v.as_i64()), Some(64));
    assert_eq!(block.get("bits").and_then(|v| v.as_i64()), Some(4));
    assert_eq!(block.get("mode").and_then(|v| v.as_str()), Some("affine"));
    let nested = block.get("layer.a").expect("per-layer override emitted");
    assert_eq!(
      nested.get("bits").and_then(|v| v.as_i64()),
      Some(8),
      "nested override carries the differing bits=8"
    );
    assert_eq!(nested.get("group_size").and_then(|v| v.as_i64()), Some(64));
    assert_eq!(nested.get("mode").and_then(|v| v.as_str()), Some("affine"));
  }

  /// `build_quantize_config` — a per-layer `Some(q)` decision EQUAL to
  /// the global default writes NO override when NOT fine-grained
  /// (`convert.rs` ~1236: `if *q != global || fine_grained`). So the
  /// live map is empty and the emitted block has only the global keys.
  #[test]
  fn build_quantize_config_equal_default_no_override_when_not_fine_grained() {
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "layer.a.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
    );
    let global = Quantization {
      group_size: 64,
      bits: 4,
      mode: QuantMode::Affine,
    };
    let mut decisions: PredicateDecisions = HashMap::new();
    decisions.insert("layer.a".to_string(), Some(global)); // EQUAL to global

    let (live, text) =
      build_quantize_config("{}", 64, 4, QuantMode::Affine, &decisions, &weights).unwrap();

    assert!(
      live.per_layer_ref().is_empty(),
      "no override when the decision equals the global and config is not fine-grained"
    );
    let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
    let block = parsed.get("quantization").unwrap().as_object().unwrap();
    // Only the 3 global keys — no per-layer key.
    assert_eq!(
      block.len(),
      3,
      "block carries only the global keys: {block:?}"
    );
    assert!(block.contains_key("group_size"));
    assert!(block.contains_key("bits"));
    assert!(block.contains_key("mode"));
  }

  /// `build_quantize_config` — fine-grained source config (already
  /// carries a `"quantization"` key) makes BOTH the `Some(q)`-equal-to-
  /// global arm (`convert.rs` ~1237, via `|| fine_grained`) AND the
  /// `None` decision arm (`convert.rs` ~1257, the `Skip` write) fire.
  /// Also covers the `Skip` → JSON `false` emission (`convert.rs`
  /// ~1282-1285).
  ///
  /// Oracle: decisions = {"keep": Some(global), "drop": None}; fine-
  /// grained source. Expect `keep` → Quantize(global) override (written
  /// because fine_grained), `drop` → Skip override (written because
  /// fine_grained), and the emitted block has `keep` = nested object,
  /// `drop` = literal `false`.
  #[test]
  fn build_quantize_config_fine_grained_writes_skip_and_default_overrides() {
    let mut weights: Weights = HashMap::new();
    for path in ["keep", "drop"] {
      weights.insert(
        format!("{path}.weight"),
        Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
      );
    }
    let global = Quantization {
      group_size: 64,
      bits: 4,
      mode: QuantMode::Affine,
    };
    let mut decisions: PredicateDecisions = HashMap::new();
    decisions.insert("keep".to_string(), Some(global)); // equal-to-global
    decisions.insert("drop".to_string(), None); // explicit skip

    // Fine-grained: the source config ALREADY carries a quantization
    // block (any shape — `contains_key("quantization")` is the gate).
    let src = r#"{"quantization":{"group_size":64,"bits":4}}"#;
    let (live, text) =
      build_quantize_config(src, 64, 4, QuantMode::Affine, &decisions, &weights).unwrap();

    // Live map: equal-to-global override IS written (fine_grained), and
    // the skip override IS written.
    assert_eq!(
      live.per_layer_ref().get("keep"),
      Some(&QuantizationOption::Quantize(global)),
      "equal-to-global decision still written under fine_grained"
    );
    assert_eq!(
      live.per_layer_ref().get("drop"),
      Some(&QuantizationOption::Skip),
      "None decision written as Skip under fine_grained"
    );

    // Emitted block: `keep` is a nested object, `drop` is literal false.
    let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
    let block = parsed.get("quantization").unwrap();
    assert!(
      block.get("keep").map(|v| v.is_object()).unwrap_or(false),
      "keep emitted as a nested object: {block:?}"
    );
    assert_eq!(
      block.get("drop"),
      Some(&serde_json::Value::Bool(false)),
      "Skip emitted as a literal false (BaseConfiguration shape)"
    );
  }

  /// `build_quantize_config` — a weight key WITHOUT a `.weight` suffix is
  /// skipped in the override-emit walk (`convert.rs` ~1217), and a
  /// `.weight` key with NO cached decision is skipped (`convert.rs`
  /// ~1223). Both `continue` arms are exercised with a non-empty
  /// `decisions` map (so the `if !decisions.is_empty()` walk runs).
  #[test]
  fn build_quantize_config_skips_non_weight_and_undecided_keys() {
    let mut weights: Weights = HashMap::new();
    // (1) No `.weight` suffix → skipped at the strip_suffix `continue`.
    weights.insert(
      "layer.bias".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 64], &(1usize, 64usize)).unwrap(),
    );
    // (2) `.weight` but absent from `decisions` → skipped at the
    //     `decisions.get(path)` None `continue`.
    weights.insert(
      "layer.undecided.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
    );
    // (3) `.weight` WITH a decision → the override IS written.
    weights.insert(
      "layer.decided.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 128], &(2usize, 64usize)).unwrap(),
    );

    let q = Quantization {
      group_size: 64,
      bits: 8,
      mode: QuantMode::Affine,
    };
    let mut decisions: PredicateDecisions = HashMap::new();
    // Non-empty so the walk runs; only `layer.decided` has an entry.
    decisions.insert("layer.decided".to_string(), Some(q));

    let (live, _text) =
      build_quantize_config("{}", 64, 4, QuantMode::Affine, &decisions, &weights).unwrap();

    // Exactly one override — the decided key. The non-`.weight` key and
    // the undecided `.weight` key both hit a `continue`.
    assert_eq!(
      live.per_layer_ref().len(),
      1,
      "only the decided key produced an override"
    );
    assert_eq!(
      live.per_layer_ref().get("layer.decided"),
      Some(&QuantizationOption::Quantize(q))
    );
    assert!(!live.per_layer_ref().contains_key("layer.bias"));
    assert!(!live.per_layer_ref().contains_key("layer.undecided"));
  }

  /// `strip_quantization_blocks` — a non-object config JSON trips the
  /// `let serde_json::Value::Object(..) else` arm (`convert.rs`
  /// ~1326-1328): [`Error::InvariantViolation`] "must be an object".
  #[test]
  fn strip_quantization_blocks_non_object_config_is_error() {
    match strip_quantization_blocks("[\"not\",\"an\",\"object\"]") {
      Err(Error::InvariantViolation(p)) => {
        assert!(
          p.context().contains("source config JSON"),
          "context names the source config: {}",
          p.context()
        );
        assert_eq!(p.requirement(), "must be an object");
      }
      other => panic!("expected Err(InvariantViolation), got {other:?}"),
    }
  }

  /// `copy_tokenizer_and_extras` — the `*.py` glob branch (`convert.rs`
  /// ~1664-1700): a `.py` file in `src` is copied to `dst`, a
  /// non-`.py` regular file is NOT, and a sub-DIRECTORY (which fails the
  /// `path.is_file()` gate ~1675) is skipped. Drives the helper directly
  /// (no MLX) over a real temp dir pair, and asserts a clean
  /// [`CopyOutcome::Committed`] (no fsync faults armed).
  #[test]
  fn copy_tokenizer_and_extras_copies_py_skips_non_py_and_dirs() {
    let dir = std::env::temp_dir().join(format!("mlxrs_convert_pyglob_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let src = dir.join("src");
    let dst = dir.join("dst");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(&dst).unwrap();

    // A `.py` file (must be copied via the glob).
    std::fs::write(src.join("modeling_custom.py"), b"# hf model code\n").unwrap();
    // A second `.py` file (the loop handles >1 entry).
    std::fs::write(src.join("configuration_custom.py"), b"# hf config code\n").unwrap();
    // A non-`.py` regular file NOT in TOKENIZER_EXTRA_FILES → skipped at
    // the `!name.ends_with(".py")` `continue`.
    std::fs::write(src.join("README.md"), b"readme\n").unwrap();
    // A sub-directory whose name ends with `.py` — must be skipped at
    // the `!path.is_file()` gate (a dir is not a file).
    std::fs::create_dir_all(src.join("package.py")).unwrap();

    let outcome = copy_tokenizer_and_extras(&src, &dst).unwrap();
    assert!(
      matches!(outcome, CopyOutcome::Committed),
      "no fsync faults → fully-durable Committed; got {outcome:?}"
    );

    // The two `.py` files were copied byte-equal.
    for name in ["modeling_custom.py", "configuration_custom.py"] {
      assert!(dst.join(name).is_file(), "{name} copied via the *.py glob");
      assert_eq!(
        std::fs::read(src.join(name)).unwrap(),
        std::fs::read(dst.join(name)).unwrap(),
        "{name} byte-equal at dst"
      );
    }
    // The non-`.py` file was NOT copied.
    assert!(
      !dst.join("README.md").is_file(),
      "non-.py regular file is not copied by the glob"
    );
    // The `.py`-named DIRECTORY was NOT copied (skipped at is_file gate).
    assert!(
      !dst.join("package.py").exists(),
      ".py-named sub-directory is skipped (not a file)"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// `copy_tokenizer_and_extras` — a NON-EXISTENT `src` directory makes
  /// the `std::fs::read_dir(src)` call fail, tripping the
  /// `.map_err(...)` arm (`convert.rs` ~1656-1663): an
  /// [`Error::FileIo`] with [`FileOp::Read`] naming `src`. The
  /// TOKENIZER_EXTRA_FILES loop above it is a no-op (every `is_file()`
  /// is false), so the read_dir error is the first failure reached.
  #[test]
  fn copy_tokenizer_and_extras_missing_src_dir_is_read_error() {
    let dir =
      std::env::temp_dir().join(format!("mlxrs_convert_missing_src_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // `src` deliberately does NOT exist; `dst` does.
    let src = dir.join("does_not_exist_src");
    let dst = dir.join("dst");
    std::fs::create_dir_all(&dst).unwrap();

    match copy_tokenizer_and_extras(&src, &dst) {
      Err(Error::FileIo(p)) => {
        assert_eq!(p.op(), FileOp::Read, "read_dir failure is a Read op");
        assert_eq!(p.path(), src.as_path(), "the failing path is src");
        assert!(
          p.context().contains("copy_tokenizer_and_extras"),
          "context names the helper: {}",
          p.context()
        );
      }
      other => panic!("expected Err(FileIo{{op:Read}}), got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// `paths_are_same` — when BOTH paths fail to canonicalize (neither
  /// exists), the fallback arm `_ => src == dst` (`convert.rs` ~1741)
  /// compares the original args textually. Two distinct non-existent
  /// paths → `false`; the same non-existent path → `true`.
  #[test]
  fn paths_are_same_fallback_textual_compare_when_uncanonicalizable() {
    let base =
      std::env::temp_dir().join(format!("mlxrs_convert_paths_same_{}", std::process::id()));
    // Neither sub-path exists, so canonicalize() errors for both →
    // the `_ => src == dst` fallback runs.
    let a = base.join("nonexistent_a");
    let b = base.join("nonexistent_b");
    assert!(
      !paths_are_same(&a, &b),
      "distinct uncanonicalizable paths compare unequal via the fallback"
    );
    assert!(
      paths_are_same(&a, &a),
      "identical uncanonicalizable path compares equal via the fallback"
    );
  }

  /// `convert` end-to-end — the `eligible` closure's `(true, None)` arm
  /// (`convert.rs` ~734): a predicate IS supplied AND a `.weight` key is
  /// structurally INELIGIBLE (it never entered `build_predicate_decisions`'s
  /// map), so `decisions.get(path)` is `None` and the closure returns
  /// `false` (skip). The eligible rank-2 layer is still quantized.
  ///
  /// Fixture mirrors the inline durability tests. The ineligible key is a
  /// rank-1 `.weight` (fails the `shape.len() >= 2` gate inside
  /// `build_predicate_decisions`), so it is never offered to the
  /// predicate yet IS walked by `quantize_weights`'s pass-1 eligibility
  /// closure — the exact path that reaches the `(true, None)` arm.
  #[test]
  fn convert_predicate_with_ineligible_weight_skips_via_true_none_arm() {
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_convert_true_none_arm_{}",
      std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let src = dir.join("src");
    let dst = dir.join("dst");
    std::fs::create_dir_all(&src).unwrap();

    let plain_config = r#"{
      "model_type":"qwen3","hidden_size":16,"num_hidden_layers":1,
      "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
      "rope_theta":10000.0,"vocab_size":128,"tie_word_embeddings":false
    }"#;
    std::fs::write(src.join("config.json"), plain_config).unwrap();

    let blob: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let mut weights: Weights = HashMap::new();
    // Eligible: rank-2, last axis 64 → divisible by group_size 64.
    weights.insert(
      "model.layers.0.self_attn.q_proj.weight".to_string(),
      Array::from_slice::<f32>(&blob, &(2usize, 64usize)).unwrap(),
    );
    // INELIGIBLE: rank-1 `.weight` → fails the `shape.len() >= 2` gate in
    // build_predicate_decisions (never offered to the predicate), but IS
    // walked by quantize_weights's eligibility closure → the (true, None)
    // arm returns false (stays dense).
    weights.insert(
      "model.norm.weight".to_string(),
      Array::from_slice::<f32>(&[0.0_f32; 64], &(64usize,)).unwrap(),
    );
    crate::io::save_safetensors(&src.join("model.safetensors"), &weights).unwrap();

    let tokenizer_json = include_str!("../../tests/fixtures/tokenizer.json");
    let tokenizer_config_json = include_str!("../../tests/fixtures/tokenizer_config.json");
    std::fs::write(src.join("tokenizer.json"), tokenizer_json).unwrap();
    std::fs::write(src.join("tokenizer_config.json"), tokenizer_config_json).unwrap();

    /// Predicate that quantizes every layer it is offered. The rank-1
    /// `model.norm.weight` is never offered (ineligible), so it can only
    /// be skipped via the eligible closure's `(true, None)` arm.
    struct QuantizeAll;
    impl MixedQuantPredicate for QuantizeAll {
      fn decide(&self, _layer_name: &str, _weight: &Array) -> Option<Quantization> {
        Some(Quantization::affine(64, 4))
      }
    }

    convert(ConvertArgs {
      hf_path: src,
      mlx_path: dst.clone(),
      quantize: true,
      q_bits: Some(4),
      q_group_size: Some(64),
      q_mode: QuantMode::Affine,
      quant_predicate: Some(Box::new(QuantizeAll)),
      ..Default::default()
    })
    .unwrap();

    let reloaded = load::load_weights(&dst).unwrap();
    // The eligible q_proj was quantized (has a `.scales` sibling).
    assert!(
      reloaded.contains_key("model.layers.0.self_attn.q_proj.scales"),
      "eligible rank-2 layer quantized"
    );
    // The ineligible rank-1 norm stayed dense (the (true, None) arm
    // returned false → quantize_weights skipped it → no `.scales`).
    assert!(
      reloaded.contains_key("model.norm.weight"),
      "ineligible rank-1 weight still present"
    );
    assert!(
      !reloaded.contains_key("model.norm.scales"),
      "ineligible rank-1 weight NOT quantized (skipped via the (true, None) arm)"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }
}
