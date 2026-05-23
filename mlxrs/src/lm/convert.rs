//! `convert()` — the model-conversion driver, ported from
//! [`mlx_lm/convert.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/convert.py).
//!
//! Wires the load-side (F2 [`crate::lm::load`]), the quantize / dequantize side
//! (F3 [`crate::lm::quant`]) and the save-side (F6 [`crate::lm::load::save`] +
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
//!   load(hf_path)                    (convert.py:111-118)  →  F2
//!      → (Config, Weights, Tokenizer, raw_config_json)
//!      │
//!      ▼
//!   resolve dtype, cast              (convert.py:129-144)
//!      │   ─ explicit override OR  config["torch_dtype"] OR  text_config["dtype"]
//!      ▼
//!   branch:
//!      │  quantize  → quantize_weights(weights, …)         (convert.py:149-158)  →  F3
//!      │             + patch config "quantization" block   (utils.py:813-845)
//!      │  dequantize→ dequantize_weights(weights, cfg)     (convert.py:160-164)  →  F3
//!      │             + strip "quantization" / "quantization_config"
//!      │  neither   → pass through unchanged
//!      ▼
//!   save(mlx_path, weights, config, per_layer_q)           (convert.py:166-172)  →  F6
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
//! Per [project memory `feedback_api_style`] the keyword-arg surface
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
  error::{Error, Result},
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
/// **`!Send`-compatible.** Per project memory, [`Array`] is `!Send` /
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
  /// no-op (weights are written in their loaded dtype). Only the three
  /// values in `MODEL_CONVERSION_DTYPES` (`convert.py:82`) are honored:
  /// [`Dtype::F16`] / [`Dtype::BF16`] / [`Dtype::F32`]. Any other
  /// `Some(_)` is accepted (no-op pass-through, matching python's silent
  /// `if dtype in MODEL_CONVERSION_DTYPES` gate at `convert.py:133`).
  pub dtype: Option<Dtype>,

  /// `upload_repo` (`convert.py:93`). HuggingFace Hub destination repo.
  /// **REJECTED.** A non-`None` value at [`convert`]-call time returns
  /// [`Error::Backend`] — mlxrs is local-only per the module-level
  /// "Scope decisions" + [project memory `project_no_model_arch_porting`].
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
    return Err(Error::Backend {
      message: "mixed_quant_predicate: model does not have expected keys for mixed quant \
                (no `down_proj`-bearing layer in the weight map)"
        .into(),
    });
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
    .ok_or_else(|| Error::Backend {
      message: format!(
        "mixed_quant_predicate: cannot locate the layer-index segment in `down_proj` \
         path {first:?} (mirroring `convert.py:43-45`'s `if k.isdigit(): break`)"
      ),
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
/// `Ok(())` on a successful conversion. Every recoverable failure is an
/// [`Error`] surfaced as-is from the called primitive:
///
/// - [`Error::Backend`] for argument validation (existing destination,
///   mutually-exclusive flags, rejected `upload_repo` / `revision`),
///   load failures (missing / oversized / invalid `config.json` or
///   weights or tokenizer — see [`crate::lm::load::load`]), quantize /
///   dequantize failures (see [`crate::lm::quant`]) and save failures
///   (see [`crate::lm::load::save`]).
/// - [`Error::DurabilityWarning`] when the save committed but the
///   post-rename `fsync_dir` failed — the new checkpoint IS visible on
///   disk but a power loss before the FS drains could lose the directory
///   entry (see [`crate::lm::load::save`]'s contract). The on-disk
///   convert is logically complete in this case.
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
    return Err(Error::Backend {
      message: format!(
        "convert: cannot save to the path {} as it already exists. Please delete \
         the file/directory or specify a new path to save to.",
        mlx_path.display()
      ),
    });
  }

  // mlxrs is local-only; the python Hub upload / download surface is
  // out of scope per the module-level "Scope decisions". Reject AFTER
  // the `exists()` check so destination validation always runs first
  // (mirrors the reference's check order at `convert.py:101-109`).
  if upload_repo.is_some() {
    return Err(Error::Backend {
      message: "convert: `upload_repo` is unsupported in mlxrs (HuggingFace Hub upload \
                is out of scope — mlxrs is local-path-only). Drop the kwarg or upload \
                the result directory yourself."
        .into(),
    });
  }
  if revision.is_some() {
    return Err(Error::Backend {
      message: "convert: `revision` is unsupported in mlxrs (HuggingFace Hub download \
                is out of scope — mlxrs is local-path-only). Download the checkpoint \
                yourself and pass its local path as `hf_path`."
        .into(),
    });
  }

  // `convert.py:146-147` — `if quantize and dequantize: raise ValueError(...)`.
  if quantize && dequantize {
    return Err(Error::Backend {
      message: "convert: choose either `quantize` or `dequantize`, not both \
                (convert.py:146-147)."
        .into(),
    });
  }

  // ─── 2. Load (`convert.py:111-118` → F2) ───
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
  let resolved_dtype = resolve_target_dtype(dtype, &config_json_text);
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
    let (cfg, cfg_json) = build_quantize_config(
      &config_json_text,
      gs,
      bits,
      q_mode,
      quant_predicate.as_deref(),
      &weights,
    )?;
    let eligible = |path: &str, weight: &Array| -> bool {
      // The "structural analogue of mlx-lm's `hasattr(module,
      // 'to_quantized')`" predicate, deferring to the user-supplied
      // [`MixedQuantPredicate`] when one is set so a `None` decision
      // from it (the python `False` arm of `wrapped_predicate`,
      // `utils.py:823-835`) actually skips the layer at the
      // `quantize_weights` call site.
      if let Some(p) = quant_predicate.as_deref() {
        return p.decide(path, weight).is_some();
      }
      true
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

  // ─── 5. Save (`convert.py:166-172` → F6) ───
  //
  // `save(mlx_path, hf_path, model, tokenizer, config)`. mlxrs's `save`
  // doesn't carry the tokenizer (no `tokenizer.save_pretrained` is
  // ported — the tokenizer surface is load-only) or the source-`*.py` /
  // `generation_config.json` copy: those are this F7 driver's
  // `copy_tokenizer_and_extras` step below.
  load::save(&mlx_path, &out_weights, &out_config_json, &per_layer_cfg)?;

  // ─── 6. Copy tokenizer + extras (the deliberately-deferred portion
  //         of `utils.save`, `utils.py:944-948`) ───
  copy_tokenizer_and_extras(&hf_path, &mlx_path)?;

  // ─── 7. (Hub upload — `convert.py:174-175`) — REJECTED at step 1. ───

  Ok(())
}

// ─────────────────────── helpers ───────────────────────

/// `defaults_for_mode` (`utils.py:800-808`) — per-mode `(group_size,
/// bits)` fallbacks when the kwarg is `None`. mlx-lm's hard-coded table.
fn defaults_for_mode(mode: QuantMode, gs: Option<i32>, bits: Option<i32>) -> (i32, i32) {
  let (default_gs, default_bits) = match mode {
    QuantMode::Affine => (64, 4),
    QuantMode::Mxfp4 => (32, 4),
    QuantMode::Nvfp4 => (16, 4),
    QuantMode::Mxfp8 => (32, 8),
  };
  (gs.unwrap_or(default_gs), bits.unwrap_or(default_bits))
}

/// Resolve the target floating dtype the cast step should use, mirroring
/// the python fallback chain at `convert.py:129-133`:
///   1. explicit kwarg (`Some(d)` returned as-is)
///   2. `config.json` `torch_dtype` (string)
///   3. `config.json` `text_config.dtype` (string; the VLM-config
///      fallback)
///   4. `None` (no cast)
///
/// Only the three [`MODEL_CONVERSION_DTYPES`] (`convert.py:82`) are
/// honored — any other parsed string falls through to `None` (no cast),
/// matching python's silent `if dtype in MODEL_CONVERSION_DTYPES` gate.
fn resolve_target_dtype(explicit: Option<Dtype>, config_json: &str) -> Option<Dtype> {
  if let Some(d) = explicit {
    // The python `if dtype in MODEL_CONVERSION_DTYPES` gate
    // (`convert.py:133`) implicitly accepts any explicit `mx.*` dtype
    // the user passes too — mlxrs's `Dtype` enum is already a fixed
    // set; pass it through.
    return Some(d);
  }
  let parsed: serde_json::Value = match serde_json::from_str(config_json) {
    Ok(v) => v,
    Err(_) => return None, // config parsed once already; this path is unreachable in practice
  };
  if let Some(s) = parsed.get("torch_dtype").and_then(|v| v.as_str())
    && let Some(d) = parse_conversion_dtype(s)
  {
    return Some(d);
  }
  if let Some(text_cfg) = parsed.get("text_config")
    && let Some(s) = text_cfg.get("dtype").and_then(|v| v.as_str())
    && let Some(d) = parse_conversion_dtype(s)
  {
    return Some(d);
  }
  None
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
/// — F6's `save_config` already does that mirror at write-time, so the
/// returned config text carries only the `quantization` key (the mirror
/// happens inside `save_config`).
fn build_quantize_config(
  config_json: &str,
  group_size: i32,
  bits: i32,
  mode: QuantMode,
  predicate: Option<&dyn MixedQuantPredicate>,
  weights: &Weights,
) -> Result<(PerLayerQuantization, String)> {
  let value: serde_json::Value = serde_json::from_str(config_json).map_err(|e| Error::Backend {
    message: format!("convert: source config is not valid JSON: {e}"),
  })?;
  let serde_json::Value::Object(mut map) = value else {
    return Err(Error::Backend {
      message: "convert: source config JSON must be an object".into(),
    });
  };

  // Python's `fine_grained_config` flag (`utils.py:815-820`).
  let fine_grained = map.contains_key("quantization");

  // Build the live [`PerLayerQuantization`] the `quantize_weights` call
  // will read for per-layer (and global) params. The global default is
  // always the call's (group_size, bits, mode); per-layer overrides
  // come from the predicate.
  let global = Quantization {
    group_size,
    bits,
    mode,
  };
  let mut per_layer_overrides: HashMap<String, QuantizationOption> = HashMap::new();

  // Walk every `.weight`-bearing weight key; for each, ask the predicate
  // what to do. The python `wrapped_predicate(path, module)` runs at
  // `nn.quantize(...)` time over every Linear/Embedding/SwitchLinear
  // module; we walk the weight MAP keys to get the same shape (path,
  // weight). The predicate's decision flows two ways: into the runtime
  // `per_layer` overrides (so `quantize_weights` sees them), and into
  // the saved config block (so a later load can reconstruct the
  // per-layer layout, mirroring `utils.py:832-834`).
  if let Some(pred) = predicate {
    for (key, arr) in weights {
      let Some(path) = key.strip_suffix(".weight") else {
        continue;
      };
      // mlx-lm `class_predicate` only fires for layers that pass the
      // structural shape gate (`weight.shape[-1] % group_size == 0`,
      // `utils.py:826-827`). Mirror so a predicate returning `Some(q)`
      // for an ineligible layer doesn't end up in the saved config.
      let shape = arr.shape();
      if shape.len() < 2 {
        continue;
      }
      let last = *shape.last().expect("rank>=2");
      if group_size <= 0 || last % (group_size as usize) != 0 {
        continue;
      }
      match pred.decide(path, arr) {
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
          if q != global || fine_grained {
            per_layer_overrides.insert(path.to_string(), QuantizationOption::Quantize(q));
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
    serde_json::Value::String(mode.as_mlx_str().to_string()),
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
          serde_json::Value::String(q.mode.as_mlx_str().to_string()),
        );
        quant_block.insert(path.clone(), serde_json::Value::Object(nested));
      }
    }
  }
  map.insert(
    "quantization".to_string(),
    serde_json::Value::Object(quant_block),
  );

  let updated_text =
    serde_json::to_string(&serde_json::Value::Object(map)).map_err(|e| Error::Backend {
      message: format!("convert: cannot re-serialize patched config: {e}"),
    })?;

  let live_cfg = PerLayerQuantization {
    quantization: Some(global),
    per_layer: per_layer_overrides,
  };
  Ok((live_cfg, updated_text))
}

/// Strip `quantization` and `quantization_config` from the in-memory
/// config text, mirroring `convert.py:162-163` (`config.pop(...)`).
/// Used by the dequantize branch so the saved config doesn't carry a
/// stale quant block.
fn strip_quantization_blocks(config_json: &str) -> Result<String> {
  let value: serde_json::Value = serde_json::from_str(config_json).map_err(|e| Error::Backend {
    message: format!("convert: source config is not valid JSON: {e}"),
  })?;
  let serde_json::Value::Object(mut map) = value else {
    return Err(Error::Backend {
      message: "convert: source config JSON must be an object".into(),
    });
  };
  map.remove("quantization");
  map.remove("quantization_config");
  let stripped = serde_json::Value::Object(map);
  serde_json::to_string(&stripped).map_err(|e| Error::Backend {
    message: format!("convert: cannot re-serialize stripped config: {e}"),
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
/// `config.json` is intentionally NOT in the list — F6's `save_config`
/// emits the cleaned (`_name_or_path` / `vision_config` removed,
/// `quantization` mirrored to `quantization_config`, sorted) version
/// inside `save`. Weight files (`*.safetensors` / `*.bin` / `*.gguf`)
/// are NOT copied — F6's `save_model` writes the new sharded layout.
///
/// [`tokenizer.save_pretrained`]: https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/tokenizer
const TOKENIZER_EXTRA_FILES: &[&str] = &[
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
/// - `config.json` — F6's `save_config` already writes the cleaned
///   version inside `convert`'s save step.
/// - `*.safetensors` / `*.bin` / `*.gguf` / `model.safetensors.index.json`
///   — F6's `save_model` writes the new sharded layout.
///
/// **Rename-in-place** (`src == dst`): no-op. Same-path copies would
/// either truncate-to-zero or no-op depending on `std::fs::copy`'s
/// implementation; the explicit guard short-circuits the entire walk.
/// `src`'s files are left untouched.
///
/// Failure semantics: a missing source file is silently skipped (the
/// python `for file in glob(...)` is naturally absent-tolerant); an IO
/// failure on an existing source file is a recoverable
/// [`Error::Backend`] naming the offending file and the underlying
/// error.
pub fn copy_tokenizer_and_extras(src: &Path, dst: &Path) -> Result<()> {
  // Rename-in-place: nothing to do (mirrors python's natural behavior —
  // `tokenizer.save_pretrained(dst)` is a no-op when the tokenizer was
  // loaded from `dst`, and `shutil.copy` on the same path is at best a
  // no-op at worst a truncate). Short-circuit so we don't accidentally
  // unlink-then-recreate.
  if paths_are_same(src, dst) {
    return Ok(());
  }

  // The fixed-set extras.
  for name in TOKENIZER_EXTRA_FILES {
    let s = src.join(name);
    if !s.is_file() {
      continue;
    }
    let d = dst.join(name);
    std::fs::copy(&s, &d).map_err(|e| Error::Backend {
      message: format!(
        "copy_tokenizer_and_extras: cannot copy {} -> {}: {e}",
        s.display(),
        d.display()
      ),
    })?;
  }

  // The python `*.py` glob (HF model code; some loaders need it).
  let entries = match std::fs::read_dir(src) {
    Ok(e) => e,
    Err(e) => {
      return Err(Error::Backend {
        message: format!(
          "copy_tokenizer_and_extras: cannot read source dir {}: {e}",
          src.display()
        ),
      });
    }
  };
  for entry in entries {
    let entry = entry.map_err(|e| Error::Backend {
      message: format!(
        "copy_tokenizer_and_extras: cannot read entry in {}: {e}",
        src.display()
      ),
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
    std::fs::copy(&path, &d).map_err(|e| Error::Backend {
      message: format!(
        "copy_tokenizer_and_extras: cannot copy {} -> {}: {e}",
        path.display(),
        d.display()
      ),
    })?;
  }

  Ok(())
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
      resolve_target_dtype(Some(Dtype::BF16), cfg),
      Some(Dtype::BF16)
    );
  }

  #[test]
  fn resolve_target_dtype_falls_back_to_torch_dtype() {
    let cfg = r#"{"torch_dtype":"bfloat16"}"#;
    assert_eq!(resolve_target_dtype(None, cfg), Some(Dtype::BF16));
  }

  #[test]
  fn resolve_target_dtype_falls_back_to_text_config_dtype() {
    let cfg = r#"{"text_config":{"dtype":"float16"}}"#;
    assert_eq!(resolve_target_dtype(None, cfg), Some(Dtype::F16));
  }

  #[test]
  fn resolve_target_dtype_unknown_is_none() {
    let cfg = r#"{"torch_dtype":"float64"}"#;
    assert_eq!(resolve_target_dtype(None, cfg), None);
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
}
