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

    // F7 R1 Finding-3 closure: evaluate the predicate ONCE per
    // structurally-eligible layer into a decision map BEFORE walking
    // either the config builder OR the `quantize_weights` eligibility
    // closure. The python reference's `wrapped_predicate` is called
    // exactly once per module by `nn.quantize` (`utils.py:837-843`),
    // and its single return value flows BOTH into `quantized_config`
    // (`utils.py:831-834`) AND back to `nn.quantize`'s decision —
    // a stateful predicate is therefore consistent across both views.
    //
    // The mlxrs port previously evaluated the predicate twice (once in
    // `build_quantize_config`, again in the `eligible` closure), so a
    // stateful / non-deterministic predicate could write one decision
    // into the saved config and apply a different one to the weights.
    // Caching the per-eligible-layer decision in a `HashMap` collapses
    // both reads to the same source-of-truth, matching the reference's
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

  // ─── 5. Save (`convert.py:166-172` → F6) ───
  //
  // `save(mlx_path, hf_path, model, tokenizer, config)`. mlxrs's `save`
  // doesn't carry the tokenizer (no `tokenizer.save_pretrained` is
  // ported — the tokenizer surface is load-only) or the source-`*.py` /
  // `generation_config.json` copy: those are this F7 driver's
  // `copy_tokenizer_and_extras` step below.
  //
  // F7 R1 Finding-4: a [`Error::DurabilityWarning`] with `committed:
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
      Err(Error::DurabilityWarning {
        committed: true,
        source,
      }) => Some(source),
      Err(other) => return Err(other),
    };

  // ─── 6. Copy tokenizer + extras (the deliberately-deferred portion
  //         of `utils.save`, `utils.py:944-948`) ───
  //
  // Runs unconditionally on a committed save — including the committed-
  // durability-warning branch — so the destination dir is fully
  // populated before we propagate the warning to the caller.
  copy_tokenizer_and_extras(&hf_path, &mlx_path)?;

  // ─── 7. (Hub upload — `convert.py:174-175`) — REJECTED at step 1. ───

  // Re-surface any committed-DurabilityWarning AFTER the tokenizer /
  // extras copy ran (step 6). The on-disk dir is logically complete;
  // the caller's contract on `Err(DurabilityWarning { committed: true,
  // .. })` is "the save IS visible but the parent-dir fsync didn't
  // return success" — same shape [`load::save`] surfaces.
  if let Some(source) = committed_warning {
    return Err(Error::DurabilityWarning {
      committed: true,
      source,
    });
  }
  Ok(())
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
/// `group_size: 0` quantization block on disk — see F7 R1 Finding-2).
///
/// We mirror with `Option::filter(|&v| v > 0).unwrap_or(default)` —
/// faithful one-line transcription of `value or default` for the
/// positive-integer arithmetic the table demands. Negative values would
/// also be invalid (mlx's `quantize` asserts positive `group_size` /
/// `bits`); the filter rejects those too so a `Some(-1)` doesn't
/// pretend the caller actually wanted `-1`.
fn defaults_for_mode(mode: QuantMode, gs: Option<i32>, bits: Option<i32>) -> (i32, i32) {
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
      other => Err(Error::Backend {
        message: format!(
          "convert: `dtype` must be one of float16, bfloat16, float32 — got {other:?}; \
           matches mlx_lm/convert.py:82 supported set (MODEL_CONVERSION_DTYPES)"
        ),
      }),
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

/// Per-eligible-layer cached predicate decision — the F7 R1 Finding-3
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
/// onto the predicate's single decision. F7 R1 Finding-3.
///
/// When `predicate` is `None`, the returned map is empty — the
/// downstream `eligible` closure short-circuits to "every layer is
/// eligible" in that case (matching the python `quant_predicate=None`
/// arm's `bool_or_params = True` default at `utils.py:828`).
///
/// `group_size <= 0` is a defensive guard — `defaults_for_mode` now
/// clamps `Some(0)` to the mode default (F7 R1 Finding-2), so a 0 here
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
/// — F6's `save_config` already does that mirror at write-time, so the
/// returned config text carries only the `quantization` key (the mirror
/// happens inside `save_config`).
///
/// **Predicate decisions are pre-computed.** The `decisions` arg is the
/// F7 R1 Finding-3 single-evaluation map (see
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

  /// Finding 2 (Codex F7 R1) — `Some(0)` is python-falsy and MUST fall
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

  /// Finding 1 (Codex F7 R1) — an explicit `Some(Dtype::I32)` (or any
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
      Err(Error::Backend { message }) => {
        assert!(
          message.contains("float16")
            && message.contains("bfloat16")
            && message.contains("float32"),
          "error names the supported set: {message}"
        );
        assert!(
          message.contains("I32") || message.contains("got"),
          "error names the rejected dtype: {message}"
        );
      }
      other => panic!("expected Err(Backend), got {other:?}"),
    }
  }

  #[test]
  fn resolve_target_dtype_explicit_f64_is_error() {
    let cfg = r#"{}"#;
    match resolve_target_dtype(Some(Dtype::F64), cfg) {
      Err(Error::Backend { message }) => {
        assert!(message.contains("float16"));
      }
      other => panic!("expected Err(Backend), got {other:?}"),
    }
  }

  #[test]
  fn resolve_target_dtype_explicit_complex_is_error() {
    let cfg = r#"{}"#;
    match resolve_target_dtype(Some(Dtype::Complex64), cfg) {
      Err(Error::Backend { message }) => {
        assert!(message.contains("float16"));
      }
      other => panic!("expected Err(Backend), got {other:?}"),
    }
  }

  #[test]
  fn resolve_target_dtype_explicit_bool_is_error() {
    let cfg = r#"{}"#;
    match resolve_target_dtype(Some(Dtype::Bool), cfg) {
      Err(Error::Backend { message }) => {
        assert!(message.contains("float16"));
      }
      other => panic!("expected Err(Backend), got {other:?}"),
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

  // ─────────── Finding 3 — single-evaluation predicate ───────────

  use std::cell::RefCell;

  /// Test-only predicate that counts how many times `decide` was called
  /// per layer path. Used by the F7 R1 Finding-3 closure test to assert
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
      // Finding-3 desync (config-says-quantize, weights-say-skip)
      // would manifest if the predicate were called twice and saw
      // different cycle values.
      *self.cycle.borrow_mut() += 1;
      Some(Quantization {
        group_size: 64,
        bits: 4,
        mode: QuantMode::Affine,
      })
    }
  }

  /// Finding 3 (Codex F7 R1) — `build_predicate_decisions` must call
  /// the predicate exactly ONCE per structurally-eligible layer. The
  /// previous shape evaluated the predicate twice (once in
  /// `build_quantize_config`, again in the `eligible` closure of
  /// `convert`); a stateful predicate could record one decision in the
  /// saved config and apply a different one to weights. This test
  /// counts invocations per path and asserts max <= 1, mirroring the
  /// python `nn.quantize`'s single-call-per-module contract
  /// (`utils.py:837-843`).
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

  /// Finding 3 followup — re-call after the map is built must NOT
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

  // ─────────── Finding 4 — DurabilityWarning still copies tokenizer ───────────

  /// Finding 4 (Codex F7 R1) — when `load::save` returns
  /// [`Error::DurabilityWarning`] with `committed: true`, the weights +
  /// config are visible on disk; `convert` MUST continue with the
  /// tokenizer / extras copy (so the destination dir is COMPLETE)
  /// before re-surfacing the warning. The previous shape used `?` on
  /// the `load::save` call, which early-returned and SKIPPED the
  /// `copy_tokenizer_and_extras` step — a non-fatal durability warning
  /// became a partial, hard-to-recover conversion (the destination dir
  /// existed so the `mlx_path.exists()` gate of a retry would reject
  /// it, but tokenizer files were missing).
  ///
  /// This test:
  ///   (a) arms the F6 `fsync_dir` fault injector to fire AFTER the
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
      Err(Error::DurabilityWarning { committed, source }) => {
        assert!(
          committed,
          "convert's DurabilityWarning must carry committed=true"
        );
        assert!(
          source.to_string().contains("injected fsync_dir failure"),
          "underlying io::Error preserved: got {source}"
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
}
