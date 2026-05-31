//! Inference-time **LoRA / DoRA adapter loading** — the runtime surface that
//! takes a *pre-trained* low-rank adapter and runs it against a base model.
//!
//! Port of the inference-relevant half of mlx-lm's `mlx_lm/tuner/` and
//! mlx-swift-lm's `MLXLMCommon/Adapters/LoRA/`:
//!
//! - The LoRA-wrapped linear layer [`LoRALinear`] and the weight-decomposed
//!   [`DoRALinear`], each wrapping a [`BaseLinear`] that is **either** a dense
//!   weight **or** an MLX-quantized triple ([`BaseLinear::Quantized`]) — the
//!   QLoRA / QDoRA case (swift's separate `QLoRALinear` / `QDoRALinear`
//!   classes; here one type covers both bases, mirroring mlx-lm's `LoRALinear`
//!   which wraps `Linear` and `QuantizedLinear` alike, `tuner/lora.py:22-23`).
//!   These mirror mlx-lm `tuner/lora.py::LoRALinear` +
//!   `tuner/dora.py::DoRALinear` and the swift `LoRA+Layers.swift` /
//!   `DoRA+Layers.swift` classes. Each ports the FORWARD pass and the
//!   [`fuse`](LoraLayer::fuse) method (fold the adapter into the base weight so
//!   a fused model needs no adapter at runtime).
//! - [`linear_to_lora_layers`] — the layer-selection step: wrap the targeted
//!   linear weights of a [`Weights`] map (the `keys` / `num_layers` predicate
//!   from `adapter_config.json`), mirroring mlx-lm
//!   `tuner/utils.py::linear_to_lora_layers`.
//! - [`load_adapters`] — the load-time entry: read `adapter_config.json` +
//!   `adapters.safetensors` from a **local** directory (no HuggingFace Hub, per
//!   the project's local-path-only scope), build the LoRA/DoRA layers, and bind
//!   their parameters, mirroring mlx-lm `tuner/utils.py::load_adapters` +
//!   swift `LoRAContainer.from(directory:)` / `load(into:)`.
//! - [`LoraConfig`] — the `adapter_config.json` schema (rank `r`, `alpha` /
//!   `scale`, target `keys`, `fine_tune_type` lora|dora|full, `num_layers`),
//!   mirroring swift `LoRAConfiguration` (`LoRAContainer.swift:27-66`).
//!
//! # Scope — inference adapter loading ONLY
//!
//! This is the surface for **running** a pre-trained adapter, NOT training one.
//! Deliberately excluded (training, out of project scope): the optimizer /
//! loss / dataset / `trainer.py` surface (`tuner/trainer.py`,
//! `tuner/datasets.py`, `tuner/losses.py`), `print_trainable_parameters`, the
//! `dropout` (an inference adapter has no dropout — mlx-lm passes `dropout=0.0`
//! at load and the layer's `nn.Dropout(p=0)` is the identity; this port omits
//! the dropout module entirely, so [`LoRALinear::forward`] applies the
//! low-rank term to `x` directly), and the random `lora_a` / zero `lora_b`
//! *initializers* (training only — at inference both come from
//! `adapters.safetensors`).
//!
//! The MoE `LoRASwitchLinear` / `LoRAEmbedding` variants
//! (`tuner/lora.py:101,198`) are deferred follow-ups — they need the
//! `gather_mm`/embedding-as-linear adapter wiring layered on top of this base
//! `Linear` surface, exactly as [`crate::lm::nn::switch`] deferred `SwitchMLP`.
//!
//! # No module tree — the weight-map model
//!
//! mlx-lm / mlx-swift apply LoRA by walking a live `nn.Module` tree, replacing
//! `Linear` leaves with `LoRALinear` wrappers. mlxrs has **no** model-module
//! tree (that is per-usecase), so — exactly as
//! [`crate::lm::quant`] walks the [`Weights`] name-map instead of an
//! `nn.Module` — this module builds [`LoRALinear`] objects keyed by their
//! base-weight **path** in the loaded [`Weights`] map. [`linear_to_lora_layers`]
//! returns a [`LoraLayers`] map (path → wrapped layer); the per-usecase
//! architecture, which already routes a `model.layers.N.self_attn.q_proj` path
//! to its forward call, dispatches through the wrapped layer for adapted paths.
//!
//! # The LoRA forward math
//!
//! For a base linear `W` (shape `[output_dims, input_dims]`), low-rank factors
//! `lora_a` (`[input_dims, r]`) and `lora_b` (`[r, output_dims]`), and a scalar
//! `scale` (`scale = alpha / r` when `alpha`/`lora_alpha` is present — the PEFT
//! convention, which WINS over a literal `scale` — else the literal `scale`
//! field, else the `20.0` default):
//!
//! ```text
//! LoRA:  y = x @ Wᵀ (+ bias)
//!        z = (x @ lora_a) @ lora_b
//!        out = y + (scale · z)
//!
//! fuse:  Δ = (scale · lora_bᵀ) @ lora_aᵀ      // shape [output_dims, input_dims]
//!        W_fused = W + Δ
//! ```
//!
//! DoRA (weight-decomposed) additionally carries a per-output-row magnitude
//! `m = ‖W‖₂ along axis 1` (`[output_dims]`) and renormalizes:
//!
//! ```text
//! DoRA:  adapted = W + (scale · lora_bᵀ) @ lora_aᵀ
//!        denom   = ‖adapted‖₂ along axis 1
//!        out     = (m / denom) · (y + scale · z) (+ bias)
//!
//! fuse:  W_adapted = W + (scale · lora_bᵀ) @ lora_aᵀ
//!        W_fused   = (m / ‖W_adapted‖₂)[:, None] · W_adapted
//! ```
//!
//! These match mlx-lm `tuner/lora.py::LoRALinear.{__call__,fuse}` /
//! `tuner/dora.py::DoRALinear.{__call__,fuse}` and swift
//! `LoRA+Layers.swift` / `DoRA+Layers.swift` exactly.
//!
//! Conventions mirror [`crate::lm::quant`] / [`crate::lm::load`]:
//! `Result`-fallible, no implicit eval (the returned `Array`s are lazy — no
//! `eval`/`item`/`to_vec`), recoverable IO / parse / shape failures map to
//! [`Error::Backend`] / [`Error::RankMismatch`] / [`Error::LengthMismatch`] /
//! [`Error::ShapePairMismatch`] with a clear message, and the
//! `adapter_config.json` read is bounded against an untrusted adapter directory
//! exactly as [`crate::lm::load::load_config`].
//!
//! [`Error::Backend`]: crate::Error::Backend
//! [`feedback_no_per_model_arch_porting`]: crate::lm

use std::{
  collections::{HashMap, HashSet},
  path::Path,
};

use regex::Regex;

use crate::{
  array::Array,
  error::{
    CapExceededPayload, Error, FileIoPayload, FileOp, InvariantViolationPayload, LayerKeyedPayload,
    LengthMismatchPayload, MissingFieldPayload, MissingKeyPayload, OutOfRangePayload, ParsePayload,
    RankMismatchPayload, Result, ShapePairMismatchPayload, UnknownEnumValuePayload,
  },
  lm::{
    load::Weights,
    quant::{PerLayerQuantization, Quantization},
  },
  ops,
};

/// mlx-lm's default LoRA `scale` (`tuner/lora.py:17,73`: `scale: float =
/// 20.0`) and swift's (`LoRALinear` init `scale: Float = 20.0`). Applied when
/// neither `scale` nor `alpha` is present in `adapter_config.json`.
pub const DEFAULT_LORA_SCALE: f32 = 20.0;

/// mlx-lm's default LoRA rank (`tuner/lora.py:16,71`: `r: int = 8`). Also the
/// HuggingFace PEFT `LoraConfig.r` default (`peft` `lora/config.py`: `r: int =
/// 8`) — the two coincide.
pub const DEFAULT_LORA_RANK: i32 = 8;

/// HuggingFace PEFT `LoraConfig.lora_alpha`'s default (`peft` `lora/config.py`:
/// `lora_alpha: int = 8`). A PEFT adapter that omits `lora_alpha` is scaled by
/// `8 / r` (or `8 / sqrt(r)` under rsLoRA) — NOT by the mlx-lm `20.0` literal
/// (PEFT has no literal-scale concept). Used only on the PEFT-flat path.
pub const DEFAULT_PEFT_LORA_ALPHA: f32 = 8.0;

/// mlx-lm's default `num_layers` for a LoRA config — the number of trailing
/// decoder blocks adapted (`mlx_lm` adapter configs commonly carry it
/// explicitly; swift `LoRAConfiguration` defaults to `16`,
/// `LoRAContainer.swift:52`).
pub const DEFAULT_NUM_LAYERS: i32 = 16;

/// Upper bound on the `adapters.safetensors` file [`load_adapters`] will hand
/// to [`crate::io::load_safetensors`]. A LoRA/DoRA adapter is **low-rank** —
/// only the `lora_a` / `lora_b` (and DoRA `m`) factors of the targeted
/// projections — so even a wide, high-rank adapter over a large model is well
/// under this bound; a file beyond it is not a plausible adapter. The cap
/// bounds the damage an untrusted adapter dir can do (a hostile
/// `adapters.safetensors` pointing at an oversized blob ⇒ a clear recoverable
/// error, not an OOM). Generous (2 GiB) because the budget is a safety ceiling,
/// not a tight fit — distinct from the 1-MiB `lm::load`-internal JSON-config
/// cap (`MAX_CONFIG_BYTES`).
pub const MAX_ADAPTER_SAFETENSORS_BYTES: u64 = 2 << 30;

// ───────────────────────────── config ─────────────────────────────

/// How a checkpoint was fine-tuned — mlx-lm `fine_tune_type`
/// (`tuner/utils.py:129`, one of `"lora"` / `"dora"` / `"full"`) and swift
/// `LoRAConfiguration.FineTuneType` (`LoRAContainer.swift:29-32`, `lora` /
/// `dora`).
///
/// `Full` (a full-weight fine-tune, no low-rank factorization) is recognized
/// for parity with mlx-lm `load_adapters` (which skips
/// `linear_to_lora_layers` entirely for `"full"` and just loads the dense
/// weight delta) but is **not** an adapter-wrapping mode — [`load_adapters`]
/// reports it as unsupported here, since mlxrs has no module tree to load a
/// full-weight delta into (the per-usecase architecture would merge a full
/// fine-tune at the weight-map level via [`crate::lm::load::load_weights`]).
#[derive(
  Debug,
  Clone,
  Copy,
  PartialEq,
  Eq,
  serde::Deserialize,
  serde::Serialize,
  derive_more::Display,
  derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum FineTuneType {
  /// Low-Rank Adaptation — the `lora_a` / `lora_b` factors only
  /// (mlx-lm `LoRALinear`).
  Lora,
  /// Weight-Decomposed Low-Rank Adaptation — LoRA plus a learned per-row
  /// magnitude `m` (mlx-lm `DoRALinear`).
  Dora,
  /// Full-weight fine-tune (no low-rank factorization). Recognized but not an
  /// adapter-wrapping mode here — see the enum docs.
  Full,
}

impl Default for FineTuneType {
  /// `lora` — mlx-lm `getattr(config, "fine_tune_type", "lora")`
  /// (`tuner/utils.py:129`).
  fn default() -> Self {
    FineTuneType::Lora
  }
}

impl FineTuneType {
  /// The lowercase tag string used in `adapter_config.json` and
  /// mlx-lm's `fine_tune_type` field.
  pub const fn as_str(self) -> &'static str {
    match self {
      FineTuneType::Lora => "lora",
      FineTuneType::Dora => "dora",
      FineTuneType::Full => "full",
    }
  }
}

/// The in-memory low-rank *parameter* block — `rank` / `scale` / `alpha` /
/// `dropout` plus (mlx-lm-native only) the `keys` allowlist. The normalized
/// representation of an `adapter_config.json`'s low-rank scalars, regardless
/// of which on-disk *shape* the config used.
///
/// # Two on-disk shapes, one in-memory form
///
/// `adapter_config.json` comes in **two** structurally different shapes, and
/// [`LoraConfig`]'s custom [`Deserialize`](serde::Deserialize) bridges both
/// into this single [`LoraParameters`]:
///
/// - **mlx-lm-native** — the low-rank settings live in a *nested*
///   `lora_parameters` object: `{ "fine_tune_type": …, "num_layers": N,
///   "lora_parameters": { "rank": R, "scale": S, "dropout": D, "keys": […] } }`
///   (mlx-lm `config.lora_parameters`, `tuner/utils.py:133`; swift
///   `LoRAConfiguration.LoRAParameters`, `LoRAContainer.swift:34-45`). Its keys
///   are `rank` / `scale` / `alpha` / `keys` / `dropout`.
/// - **PEFT / HuggingFace** — a real `peft` `LoraConfig` has **no**
///   `lora_parameters` nesting; its fields are *flat at the top level*. The
///   PEFT scalar mapping into this block is `r` → [`rank`] (default
///   [`DEFAULT_LORA_RANK`]), `lora_alpha` → [`alpha`] (default
///   [`DEFAULT_PEFT_LORA_ALPHA`]), `lora_dropout` → [`dropout`]. PEFT has no
///   literal-`scale` field, so [`scale`] is left `None`; PEFT's `target_modules`
///   selection does **not** land in `keys` — it lives in [`PeftSelection`]
///   ([`LoraConfig::selection`]), because PEFT's regex / `layers_to_transform`
///   selection is richer than a `keys` suffix list.
///
/// `scale` is the literal low-rank scale (mlx-lm `config["scale"]`). When
/// `alpha` is present (the convention `scale = alpha / rank`), it **takes
/// precedence** over the literal `scale` ([`LoraParameters::resolved_scale`]).
/// `keys` is the mlx-lm-native explicit target-projection allowlist (e.g.
/// `["self_attn.q_proj", "self_attn.v_proj"]`); `None` means "every eligible
/// linear" (mlx-lm's auto-discovery, `tuner/utils.py:85-101`). `dropout` is
/// carried for config round-trip fidelity but **ignored at inference** (an
/// inference adapter's dropout is the identity — see the [module docs](self)).
///
/// [`rank`]: LoraParameters::rank
/// [`scale`]: LoraParameters::scale
/// [`alpha`]: LoraParameters::alpha
/// [`dropout`]: LoraParameters::dropout
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct LoraParameters {
  /// Low-rank dimension `r`. In the mlx-lm-native nested shape this is the
  /// `lora_parameters.rank` key; in the PEFT flat shape it is the top-level
  /// `r` key — [`LoraConfig`]'s `Deserialize` maps whichever applies into here.
  /// Defaults to [`DEFAULT_LORA_RANK`] when neither shape supplies it. On a
  /// PEFT config this is the config-wide default; per-module `rank_pattern`
  /// overrides are in [`PeftSelection`].
  #[serde(default = "default_rank")]
  pub rank: i32,
  /// Literal low-rank scale (mlx-lm `config["scale"]`). Used when `alpha` is
  /// absent; defaults to [`DEFAULT_LORA_SCALE`] when neither `scale` nor
  /// `alpha` is present. The PEFT shape has no literal-scale field (PEFT always
  /// derives the scale from `lora_alpha / r`), so a PEFT-bridged config leaves
  /// this `None`.
  #[serde(default)]
  pub scale: Option<f32>,
  /// The alpha — if present, the effective scale is `alpha / rank` and this
  /// **takes precedence** over a literal `scale`. In the mlx-lm-native nested
  /// shape this is `lora_parameters.alpha`; in the PEFT flat shape it carries
  /// `lora_alpha` (defaulting to [`DEFAULT_PEFT_LORA_ALPHA`] when PEFT omits
  /// it). On a PEFT config this is the config-wide default; per-module
  /// `alpha_pattern` overrides and the `use_rslora` scaling are in
  /// [`PeftSelection`].
  #[serde(default)]
  pub alpha: Option<f32>,
  /// **mlx-lm-native** explicit target-projection allowlist (suffix paths like
  /// `"self_attn.q_proj"`). Empty ⇒ adapt every eligible linear (an empty
  /// allowlist is treated as absent).
  /// This is the `lora_parameters.keys` key. A PEFT config leaves this empty —
  /// PEFT's `target_modules` / `exclude_modules` selection lives in
  /// [`PeftSelection`].
  ///
  /// Private: access via [`keys_slice`](Self::keys_slice).
  #[serde(default)]
  keys: Vec<String>,
  /// Training dropout — carried for round-trip fidelity, **ignored at
  /// inference** (see module docs). The mlx-lm-native `lora_parameters.dropout`
  /// or the PEFT top-level `lora_dropout`.
  #[serde(default)]
  pub dropout: Option<f32>,
}

fn default_rank() -> i32 {
  DEFAULT_LORA_RANK
}

impl Default for LoraParameters {
  fn default() -> Self {
    Self {
      rank: DEFAULT_LORA_RANK,
      scale: None,
      alpha: None,
      keys: Vec::new(),
      dropout: None,
    }
  }
}

impl LoraParameters {
  /// The mlx-lm-native explicit target-projection allowlist. Empty ⇒ adapt
  /// every eligible linear (an empty allowlist is treated as absent — the
  /// auto-discovery path).
  #[inline(always)]
  pub fn keys_slice(&self) -> &[String] {
    &self.keys
  }
}

impl LoraParameters {
  /// The effective low-rank scale, resolving the PEFT/HF precedence:
  /// `alpha` (`lora_alpha`) **wins** when present → `alpha / rank` (the HF
  /// convention an adapter trained with `lora_alpha` carries); else the literal
  /// `scale` field; else [`DEFAULT_LORA_SCALE`]. This matches PEFT's `scaling =
  /// lora_alpha / r` taking precedence over a stored scalar, and the
  /// [module docs](self) (`scale = alpha / r` when built from `alpha`, else the
  /// literal `scale`, else `20.0`).
  ///
  /// A non-positive `rank` with an `alpha` present cannot form `alpha / rank`,
  /// so it falls back to the literal `scale` (then the default) — the
  /// [`LoraConfig`]/[`load_adapters`] path rejects `rank <= 0` before a layer is
  /// ever built, so this is a defensive floor, not a live path.
  pub fn resolved_scale(&self) -> f32 {
    // `alpha` wins — but only when `rank > 0` can form `alpha / rank`. A
    // non-positive `rank` (or an absent `alpha`) falls through to the literal
    // `scale`, then the default.
    if let Some(a) = self.alpha
      && self.rank > 0
    {
      return a / self.rank as f32;
    }
    if let Some(s) = self.scale {
      s
    } else {
      DEFAULT_LORA_SCALE
    }
  }
}

/// PEFT's `target_modules` / `exclude_modules` selector — **either** a list of
/// module-name suffixes **or** a single regex string (`peft`
/// `LoraConfig.target_modules: Optional[Union[list[str], str]]`, same for
/// `exclude_modules`).
///
/// Reproduces PEFT's `check_target_module_exists`
/// (`peft/tuners/tuners_utils.py`) match semantics **faithfully**:
///
/// - **list** — a module key matches when it equals a list entry **or** ends
///   with `".{entry}"` (PEFT: `key in target_modules or any(key.endswith(f".
///   {t}") for t in target_modules)`).
/// - **regex** — a *full* match against the whole module key (PEFT:
///   `match_target_against_key` is `re.fullmatch(target_pattern, key)`).
///
/// The regex form is modeled exactly via the `regex` crate, so a real PEFT
/// config with a regex `target_modules` loads.
///
/// # The `"all-linear"` sentinel
///
/// PEFT treats the *string* `"all-linear"` (`INCLUDE_LINEAR_LAYERS_SHORTHAND`,
/// case-insensitive) as a **special shorthand**, NOT a regex: PEFT's
/// `_maybe_include_all_linear_layers` (`tuners/tuners_utils.py`) expands it to
/// *every* eligible `nn.Linear` / `Conv1D` module, minus the output head
/// (`model.get_output_embeddings()`). This is the [`ModuleMatcher::AllLinear`]
/// arm — see `peft_module_is_selected` for how the eligibility (rank-2
/// `.weight`) + head exclusion are applied without a module tree.
#[derive(Debug, Clone, derive_more::IsVariant)]
pub enum ModuleMatcher {
  /// `["q_proj", "v_proj"]` — exact-or-`.endswith` suffix match.
  List(Vec<String>),
  /// `"...regex..."` — a full-match regex over the whole module key. Boxed
  /// because a compiled [`Regex`] is large relative to the small `Vec` arm.
  Regex(Box<Regex>),
  /// The PEFT `"all-linear"` sentinel — *every* eligible linear module minus the
  /// output head. The eligibility (rank-2 `.weight`) + head exclusion are
  /// applied in `peft_module_is_selected` (this arm has no per-name pattern);
  /// [`matches`](Self::matches) returns `true` only for the head-exclusion
  /// check, since "is this a linear?" needs the weight tensor, not just the key.
  AllLinear,
}

impl ModuleMatcher {
  /// Whether `module_key` matches — PEFT `check_target_module_exists` semantics
  /// (`re.fullmatch` for the regex arm; exact-or-`.endswith(".{entry}")` for
  /// the list arm).
  ///
  /// The [`AllLinear`](Self::AllLinear) arm matches **every** key *except* the
  /// output head (PEFT's `all-linear` excludes `model.get_output_embeddings()`).
  /// The rank-2 "is a linear" half of the predicate needs the weight tensor, so
  /// it lives in `peft_module_is_selected`, not here — `matches` alone is the
  /// key-only (head-exclusion) half.
  pub fn matches(&self, module_key: &str) -> bool {
    match self {
      ModuleMatcher::List(names) => names
        .iter()
        .any(|n| module_key == n || module_key.ends_with(&format!(".{n}"))),
      // PEFT `match_target_against_key` is `re.fullmatch` — `Regex::is_match`
      // is a `search`, so anchor it to the whole string explicitly.
      ModuleMatcher::Regex(re) => re
        .find(module_key)
        .is_some_and(|m| m.start() == 0 && m.end() == module_key.len()),
      // `all-linear` matches everything that is not the output head; the
      // rank-2 linear check is applied alongside in `peft_module_is_selected`.
      ModuleMatcher::AllLinear => !is_output_head_path(module_key),
    }
  }
}

/// Whether `path` is the model's **output head** (`lm_head`), which PEFT's
/// `all-linear` shorthand excludes (`model.get_output_embeddings()` — for HF
/// causal LMs the `lm_head` `nn.Linear`).
///
/// mlxrs sees only a flat weight map (no `nn.Module` tree), so the head cannot
/// be located via `get_output_embeddings()`; this approximates it by PEFT's own
/// naming convention — `EMBEDDING_LAYER_NAMES = ["embed_tokens", "lm_head"]`,
/// of which `lm_head` is the output projection (the input `embed_tokens` is an
/// `nn.Embedding`, never an `nn.Linear`, so it is excluded by the rank-2 check
/// anyway). A path is the head when its final dotted component is exactly
/// `lm_head` (so `lm_head` and `model.lm_head` match, but a `…q_proj` does not).
/// Documented approximation: a model with a non-`lm_head` output head (rare) is
/// not detected — see [`peft_module_is_selected`].
fn is_output_head_path(path: &str) -> bool {
  path.rsplit('.').next() == Some("lm_head")
}

/// The selection / scale half of a PEFT `LoraConfig` — everything PEFT uses to
/// decide *which* modules get a LoRA layer and *how strongly* each is scaled
/// (`peft/tuners/lora/{config,layer,model}.py` +
/// `peft/tuners/tuners_utils.py`). Built only on the PEFT-flat path; the
/// mlx-lm-native path keeps its own `num_layers` + `LoraParameters::keys`
/// selection (see [`AdapterSelection`]).
///
/// Faithful to PEFT, NOT to mlx-lm: PEFT has **no** `num_layers` trailing
/// window — it adapts EVERY block whose modules match `target_modules`, unless
/// `layers_to_transform` / `layers_pattern` restrict to explicit block indices.
#[derive(Debug, Clone)]
pub struct PeftSelection {
  /// PEFT `target_modules` (`Optional[Union[list[str], str]]`) — the module
  /// allowlist. `None` is represented as an empty [`ModuleMatcher::List`] (PEFT
  /// `None` means "auto-detect linears"; mlxrs has no module tree, so an empty
  /// allowlist with no restriction falls back to the rank-2 auto-discovery in
  /// [`linear_to_lora_layers`] — see its docs).
  pub target_modules: Option<ModuleMatcher>,
  /// PEFT `exclude_modules` (`Optional[Union[list[str], str]]`) — modules to
  /// remove from the target set. Checked *before* `target_modules` (PEFT's
  /// early `_ExcludedModule` return).
  pub exclude_modules: Option<ModuleMatcher>,
  /// PEFT `layers_to_transform` (`Optional[Union[list[int], int]]`) — when set,
  /// only these decoder-block indices are adapted. `None` ⇒ every block. An
  /// `int` is normalized to a one-element list.
  pub layers_to_transform: Option<Vec<i32>>,
  /// PEFT `layers_pattern` (`Optional[Union[list[str], str]]`) — the
  /// `ModuleList` attribute name(s) the block index sits under (e.g. `"layers"`
  /// / `"h"`). Empty ⇒ PEFT's default index regex `.*\.[^.]*\.(\d+)\.`. Only
  /// consulted when `layers_to_transform` is set.
  pub layers_pattern: Vec<String>,
  /// PEFT `rank_pattern` (`dict[str, int]`) — per-module rank overrides. Each
  /// key is matched against a module path by PEFT's `get_pattern_key`
  /// (`re.match(rf"(.*\.)?({key})$", module)`); a match overrides `r` for that
  /// module. Empty ⇒ every module uses `r`. Kept in the config's **JSON
  /// insertion order** (the deserializer preserves it) so the resolver's
  /// first-match-wins matches PEFT's in-order `get_pattern_key`.
  pub rank_pattern: Vec<(String, i32)>,
  /// PEFT `alpha_pattern` (`dict[str, int]`) — per-module alpha overrides, same
  /// `get_pattern_key` matching as `rank_pattern`. Empty ⇒ every module uses
  /// `lora_alpha`. Insertion-ordered like `rank_pattern`.
  pub alpha_pattern: Vec<(String, f32)>,
  /// PEFT `use_rslora` — rank-stabilized scaling. `true` ⇒ the per-module scale
  /// is `alpha / sqrt(r)`; `false` ⇒ `alpha / r` (`peft/tuners/lora/layer.py`
  /// `update_layer`).
  pub use_rslora: bool,
  /// PEFT `fan_in_fan_out` — `true` ⇒ the base weight is stored transposed,
  /// `[in_features, out_features]` (Conv1D-style) rather than the standard
  /// `[out_features, in_features]`. The forward / fuse math transposes it back
  /// (`peft/tuners/lora/layer.py`'s `transpose(...)`).
  pub fan_in_fan_out: bool,
}

impl PeftSelection {
  /// The rank for `module_path` — `rank_pattern[key]` when a pattern key
  /// matches (PEFT `get_pattern_key`), else the config-wide `default_rank`
  /// (PEFT `r`). PEFT `_create_and_replace`:
  /// `r = lora_config.rank_pattern.get(r_key, lora_config.r)`.
  pub fn rank_for(&self, module_path: &str, default_rank: i32) -> i32 {
    pattern_lookup(&self.rank_pattern, module_path).unwrap_or(default_rank)
  }

  /// The alpha for `module_path` — `alpha_pattern[key]` when a pattern key
  /// matches, else the config-wide `default_alpha` (PEFT `lora_alpha`). PEFT
  /// `_create_and_replace`:
  /// `alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)`.
  pub fn alpha_for(&self, module_path: &str, default_alpha: f32) -> f32 {
    pattern_lookup(&self.alpha_pattern, module_path).unwrap_or(default_alpha)
  }

  /// The effective LoRA scale for `module_path` — PEFT `update_layer`'s
  /// `scaling = lora_alpha / r` (or `lora_alpha / sqrt(r)` under `use_rslora`),
  /// with the per-module `rank_pattern` / `alpha_pattern` overrides applied
  /// first. A non-positive resolved rank yields `0.0` (degenerate; rejected
  /// upstream before a layer is built).
  pub fn scale_for(&self, module_path: &str, default_rank: i32, default_alpha: f32) -> f32 {
    let r = self.rank_for(module_path, default_rank);
    let alpha = self.alpha_for(module_path, default_alpha);
    if r <= 0 {
      return 0.0;
    }
    if self.use_rslora {
      alpha / (r as f32).sqrt()
    } else {
      alpha / r as f32
    }
  }
}

/// PEFT `get_pattern_key` (`peft/utils/other.py`): match `module_path` against
/// each `(pattern, value)` entry — a hit is `re.match(rf"(.*\.)?({pattern})$",
/// module_path)`, i.e. an optional dotted prefix then the literal pattern key
/// anchored at the end. Returns the first matching value, or `None`.
///
/// PEFT's pattern keys are themselves regex fragments; this reproduces the
/// `re.match` anchoring by full-matching `(.*\.)?(pattern)` and requiring the
/// match to reach the string end. A pattern that fails to compile is skipped
/// (a malformed `rank_pattern` key cannot crash adapter loading — it simply
/// never matches, exactly as a regex that matches nothing).
fn pattern_lookup<T: Copy>(patterns: &[(String, T)], module_path: &str) -> Option<T> {
  for (pattern, value) in patterns {
    // `(.*\.)?(pattern)$` anchored at end — equivalent to PEFT's
    // `re.match(rf"(.*\.)?({key})$", key_to_match)`.
    let Ok(re) = Regex::new(&format!(r"(.*\.)?({pattern})$")) else {
      continue;
    };
    if re
      .find(module_path)
      .is_some_and(|m| m.start() == 0 && m.end() == module_path.len())
    {
      return Some(*value);
    }
  }
  None
}

/// An `adapter_config.json` `rank_pattern` / `alpha_pattern` object,
/// deserialized **preserving JSON insertion order**.
///
/// PEFT's `get_pattern_key` (`peft/utils/other.py`) iterates the pattern dict
/// *in order* and returns the **first** key whose `re.match(rf"(.*\.)?({key})$",
/// module)` hits — so for overlapping pattern keys the tie-break is the dict's
/// insertion order, NOT a lexicographic sort. A previous port deserialized
/// these dicts into a [`HashMap`] and sorted the resulting `Vec` by key, which
/// diverged from PEFT whenever two keys could both match the same module path
/// (different winner ⇒ wrong rank/alpha, or a spurious `validate_config_rank`
/// rejection). This newtype's [`Deserialize`](serde::Deserialize) visits the
/// JSON object's entries in order and collects them into a `Vec<(key, value)>`,
/// so [`pattern_lookup`]'s first-match-wins reproduces PEFT's `get_pattern_key`
/// exactly. (`serde_json` is built with `preserve_order` for the rest of the
/// crate; this visitor preserves order regardless of the underlying map, so the
/// behavior does not depend on that feature being on.)
struct OrderedPattern<T>(Vec<(String, T)>);

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for OrderedPattern<T> {
  fn deserialize<D: serde::Deserializer<'de>>(
    deserializer: D,
  ) -> std::result::Result<Self, D::Error> {
    struct OrderedVisitor<T>(std::marker::PhantomData<T>);

    impl<'de, T: serde::Deserialize<'de>> serde::de::Visitor<'de> for OrderedVisitor<T> {
      type Value = Vec<(String, T)>;

      fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a `rank_pattern` / `alpha_pattern` JSON object {pattern: value}")
      }

      fn visit_map<M: serde::de::MapAccess<'de>>(
        self,
        mut access: M,
      ) -> std::result::Result<Self::Value, M::Error> {
        // `MapAccess` yields entries in the deserializer's iteration order — for
        // a JSON object that is the on-disk (insertion) order, so the `Vec`
        // preserves it (PEFT's `get_pattern_key` first-match-in-order semantics).
        let mut out = Vec::with_capacity(access.size_hint().unwrap_or(0));
        while let Some((k, v)) = access.next_entry::<String, T>()? {
          out.push((k, v));
        }
        Ok(out)
      }
    }

    deserializer
      .deserialize_map(OrderedVisitor(std::marker::PhantomData))
      .map(OrderedPattern)
  }
}

/// Unwrap an optional [`OrderedPattern`] into its **insertion-ordered**
/// `Vec<(pattern, value)>` (an absent field ⇒ empty), preserving the on-disk
/// order so [`pattern_lookup`]'s first-match-wins matches PEFT's
/// `get_pattern_key`. (Replaces an earlier key-sorted collection, which broke
/// PEFT's insertion-order tie-break for overlapping pattern keys.)
fn ordered_pattern<T>(pattern: Option<OrderedPattern<T>>) -> Vec<(String, T)> {
  pattern.map(|p| p.0).unwrap_or_default()
}

/// Which on-disk shape an `adapter_config.json` was — and therefore which
/// layer-selection rule [`linear_to_lora_layers`] applies.
///
/// The two shapes select layers by **structurally different** rules, so the
/// normalized config carries which one applies rather than flattening them:
///
/// - [`AdapterSelection::MlxLm`] — mlx-lm's *trailing-`num_layers`-block window*
///   plus the `lora_parameters.keys` suffix allowlist.
/// - [`AdapterSelection::Peft`] — PEFT's `target_modules` / `exclude_modules` /
///   `layers_to_transform` / `layers_pattern` selection (NO trailing window —
///   every matching block).
#[derive(Debug, Clone)]
pub enum AdapterSelection {
  /// mlx-lm-native: the trailing-`num_layers`-block window. A **non-positive**
  /// value selects ALL blocks (mlx-lm's `model.layers[-max(num_layers, 0):]`
  /// reduces to `layers[0:]` via the Python `-0` slice quirk).
  MlxLm {
    /// Number of trailing decoder blocks adapted.
    num_layers: i32,
  },
  /// PEFT-flat: target/exclude/layer-index selection (see [`PeftSelection`]).
  Peft(PeftSelection),
}

/// The `adapter_config.json` schema — mlx-lm's adapter config
/// (`tuner/utils.py:127-136`: `fine_tune_type`, `num_layers`,
/// `lora_parameters`) and swift `LoRAConfiguration` (`LoRAContainer.swift:27-66`),
/// **plus** the full HuggingFace `peft` `LoraConfig` shape.
///
/// # Two on-disk shapes
///
/// A LoRA `adapter_config.json` is written in one of **two** structurally
/// different shapes, and this type's custom [`Deserialize`](serde::Deserialize)
/// accepts both:
///
/// - **mlx-lm-native** — the low-rank settings are *nested* under a
///   `lora_parameters` object:
///   `{ "fine_tune_type": …, "num_layers": N, "lora_parameters": { "rank": R,
///   "scale": S, "dropout": D, "keys": […] } }`.
/// - **PEFT / HuggingFace** — a real `peft` `LoraConfig` has **no**
///   `lora_parameters` nesting; its fields sit *flat at the top level* under
///   PEFT's own names: `{ "peft_type": "LORA", "r": R, "lora_alpha": A,
///   "target_modules": …, "exclude_modules": …, "use_rslora": …, "use_dora":
///   …, "rank_pattern": {…}, "alpha_pattern": {…}, "layers_to_transform": …,
///   "layers_pattern": …, "fan_in_fan_out": …, "lora_dropout": D, "bias": … }`.
///
/// `Deserialize` (see its own docs) **detects** the shape: a top-level
/// `lora_parameters` object ⇒ the mlx-lm-native read; top-level `r` /
/// `lora_alpha` / `target_modules` / `peft_type` (with no `lora_parameters`) ⇒
/// the PEFT read. The resolved [`selection`](LoraConfig::selection) records
/// which rule applies — PEFT has **no** `num_layers` trailing window (the
/// bug where a PEFT config wrongly inherited `num_layers=16`); it selects by
/// `target_modules` / `layers_to_transform` instead.
///
/// # PEFT scale — rsLoRA + rank/alpha patterns
///
/// On the PEFT path the scale is **per-module**: PEFT's
/// `scaling = lora_alpha / r` (or `lora_alpha / sqrt(r)` when `use_rslora`),
/// with `rank_pattern` / `alpha_pattern` overriding `r` / `lora_alpha` for
/// modules whose path matches a pattern key. See [`PeftSelection::scale_for`].
/// On the mlx-lm-native path the scale is the single `LoraParameters`-resolved
/// value (`alpha / rank`, or the literal `scale`).
///
/// # Training-only PEFT fields — accept-and-ignore (an explicit allowlist)
///
/// A real PEFT `LoraConfig` carries training-only / metadata fields with **no**
/// inference effect on already-saved factors: `init_lora_weights` (a factor
/// *seed* for the pure-seed values `true` / `false` / `gaussian` / `eva` /
/// `orthogonal` — but the base-weight-mutating modes `olora` / `pissa*` /
/// `corda` / `loftq` / `lora_ga` are **rejected**, see below),
/// `loftq_config`, `eva_config`, `corda_config`, `lora_ga_config`, `task_type`
/// and the other inherited `PeftConfig` metadata (`auto_mapping`,
/// `base_model_name_or_path`, `revision`, `inference_mode`, `peft_version`),
/// `megatron_config`, `megatron_core`, `runtime_config`, `qalora_group_size`,
/// `ensure_weight_tying`, … These are accepted and ignored even when set — see
/// the **explicit allowlist** `is_benign_ignore_field`. (`modules_to_save` and
/// `bias`, by contrast, *do* carry inference-affecting tensors and are
/// **rejected** — see below.)
///
/// # Reject-unknown-active — the structural backstop (PEFT-flat)
///
/// PEFT's `LoraConfig` is a 75-field, ever-growing dataclass; many fields switch
/// the inference forward (`resolve_lora_variant`, `lora/layer.py`). Rather than
/// chase each new variant with a reactive reject-list (whack-a-mole — a new PEFT
/// release can add a forward-switching field this loader would then silently run
/// as vanilla LoRA), the PEFT-flat `Deserialize` takes a **reject-unknown-active**
/// posture: it captures *every* top-level key (`#[serde(flatten)]` into the
/// private `RawLoraConfig::extra` catch-all) and rejects any key that is neither
/// modeled nor on the benign allowlist and is set to an **active** value
/// (anything other than `null` / `false` — PEFT's "off" default for its variant
/// fields; see the private `reject_unknown_active_peft_fields` /
/// `is_active_config_value`). So a
/// *future* unmodeled variant fails loudly with **no per-field code change**,
/// while a config that merely carries a defaulted (`null` / `false`) future
/// field still loads. This is the backstop that catches forward-switching fields
/// not enumerated below — e.g. `arrow_config`, `use_bdlora`, `layer_replication`,
/// `trainable_token_indices`, `target_parameters`. (The mlx-lm-native nested
/// shape is our own small, well-defined format and keeps its existing
/// accept-and-ignore behavior — the rule applies to the PEFT-flat branch only.)
///
/// # Rejected PEFT fields — bias, `modules_to_save`, and the exotic LoRA variants
///
/// Some PEFT fields, when set, change an adapter's **inference** forward — so
/// they cannot be silently dropped like the training-only fields above. The
/// `Deserialize` rejects each with a clear, recoverable error (the explicit ones
/// keep a tailored message; everything else is caught by the structural backstop
/// above) rather than loading the adapter at the wrong behavior:
///
/// - **`lora_bias: true`** — ships a bias on the `lora_B` projection that PEFT
///   adds (scaled) in the forward. mlxrs's [`LoRALinear`] is a faithful port of
///   mlx-lm's `tuner/lora.py`, which has no `lora_B`-bias term, so a silent
///   drop would give wrong inference. `lora_bias: false` — the default,
///   carried by ~every adapter — is fine.
/// - **`bias: "all"` / `"lora_only"`** — PEFT trains and saves base/adapter
///   `.bias` tensors (`utils/save_and_load.py` keeps `"bias" in k`) it adds in
///   the forward; mlxrs's [`LoRALinear`] has no adapted-bias slot, so a
///   non-`"none"` value is rejected (a silent drop of the bias tensors would be
///   wrong inference). `bias: "none"` — the default — is fine. (The sidecar
///   `.bias` tensors are also rejected at the weights file during PEFT
///   weight-key translation.)
/// - **`modules_to_save` (non-empty)** — PEFT trains and saves these modules in
///   *full* (e.g. a resized `embed_tokens` / classifier head) alongside the
///   low-rank factors; mlxrs's low-rank loader has no saved-full-module slot, so
///   a non-empty list is rejected (a silent drop of the full module weights
///   would be wrong inference). (The saved full-module tensors are also rejected
///   at the weights file during PEFT weight-key translation.)
/// - **`use_qalora: true`** — Quantization-Aware LoRA average-pools the
///   `lora_A` input in groups before the low-rank matmul; a different forward.
///   (PEFT's companion `qalora_group_size` is meaningful only with
///   `use_qalora`, so it is not modeled — it parses and is dropped.)
/// - **`alora_invocation_tokens`** (non-`None`) — Activated-LoRA gates the
///   adapter by token position (applied only at/after an invocation sequence).
/// - **`velora_config`** (non-`None`) — VeLoRA alters the adapter numerics via
///   a custom compressed-activation backward.
/// - **`monteclora_config`** (non-`None`) — MonteCLoRA adds variational
///   Monte-Carlo sampling over the LoRA adapters.
///
/// # Field disposition summary
///
/// - **Modeled** (inference-affecting, fully ported): `r`, `lora_alpha`,
///   `lora_dropout` (carried, inference-ignored — an inference adapter's
///   dropout is the identity), `fan_in_fan_out`, `use_rslora`, `use_dora`,
///   `target_modules`, `exclude_modules`, `layers_to_transform`,
///   `layers_pattern`, `rank_pattern`, `alpha_pattern`, `peft_type`.
/// - **Accepted and ignored** (training/init-only / metadata, no inference
///   effect on saved factors) — the **explicit allowlist** in the private
///   `is_benign_ignore_field`: `megatron_config`/`megatron_core`,
///   `loftq_config`, `eva_config`, `corda_config`, `lora_ga_config`,
///   `init_lora_weights` (pure-seed values only — the base-weight-mutating
///   modes `olora` / `pissa*` / `corda` / `loftq` / `lora_ga` are rejected),
///   `task_type`, `auto_mapping`,
///   `base_model_name_or_path`, `revision`, `inference_mode`, `peft_version`,
///   `runtime_config`, `qalora_group_size`, `ensure_weight_tying`.
/// - **Rejected loudly** (set values that *do* change inference / model
///   structure): the explicitly-named `lora_bias`, `bias` (`!= "none"`),
///   `modules_to_save` (non-empty), and the exotic variants `use_qalora`,
///   `alora_invocation_tokens`, `velora_config`, `monteclora_config`; **plus**,
///   via the structural backstop, any *other* un-modeled non-benign field set to
///   an active value — including the base-weight-mutating `init_lora_weights`
///   modes (`olora` / `pissa*` / `corda` / `loftq` / `lora_ga`), `arrow_config`,
///   `use_bdlora`, `layer_replication`, `trainable_token_indices`,
///   `target_parameters`, and any future forward-switching field.
#[derive(Debug, Clone)]
pub struct LoraConfig {
  /// `lora` / `dora` / `full` (mlx-lm `fine_tune_type`). Defaults to
  /// [`FineTuneType::Lora`] (mlx-lm's `getattr(..., "lora")`). A PEFT config
  /// has no `fine_tune_type` key, so a PEFT-bridged config is always
  /// [`FineTuneType::Lora`] here (DoRA is signalled by PEFT's `use_dora`).
  pub fine_tune_type: FineTuneType,
  /// The normalized low-rank parameter block — built from `config.lora_parameters`
  /// (mlx-lm-native) **or** from the flat PEFT top-level fields (see the type
  /// docs and [`LoraParameters`]). On the PEFT path, `keys` is left `None`
  /// (PEFT selection lives in [`selection`](Self::selection)) and `alpha`
  /// carries `lora_alpha` (defaulting to [`DEFAULT_PEFT_LORA_ALPHA`]).
  pub lora_parameters: LoraParameters,
  /// PEFT/HF `use_dora` flag — some adapters carry the DoRA signal here
  /// instead of `fine_tune_type: "dora"`. Either signal selects DoRA (see
  /// [`LoraConfig::is_dora`]). Read from the top level in **both** shapes.
  pub use_dora: bool,
  /// Which layer-selection rule applies — the mlx-lm trailing-`num_layers`
  /// window, or the PEFT `target_modules` / `layers_to_transform` selection.
  pub selection: AdapterSelection,
}

fn default_num_layers() -> i32 {
  DEFAULT_NUM_LAYERS
}

/// The permissive deserialization target that captures the **union** of both
/// `adapter_config.json` shapes — mlx-lm-native (nested `lora_parameters`) and
/// PEFT-flat (the full top-level PEFT `LoraConfig` surface). Every field is
/// optional; [`LoraConfig`]'s [`Deserialize`](serde::Deserialize) reads this,
/// then normalizes it into the typed [`LoraConfig`].
///
/// This is the "permissive deserialize then normalize" step: serde cannot
/// branch on which keys are present from a `#[derive]`, so the raw form
/// captures *all* keys non-fatally and the normalization picks the shape.
/// Training-only / metadata PEFT fields (`init_lora_weights`, `loftq_config`,
/// `task_type`, `megatron_*`, …) are *not* listed as named fields — they fall
/// into the [`extra`](Self::extra) `#[serde(flatten)]` catch-all, where the
/// PEFT-flat normalization either accepts-and-ignores them (if benign — see
/// [`is_benign_ignore_field`]) or, under the **reject-unknown-active** rule,
/// rejects them when set to an active value (see
/// [`reject_unknown_active_peft_fields`]). The **exotic LoRA variants**
/// (`use_qalora`, `alora_invocation_tokens`, `velora_config`,
/// `monteclora_config`) *are* listed as named fields — not to honor them, but so
/// the shape-independent [`reject_exotic_variants`] guard can **reject** an
/// adapter that sets them with a tailored message (each changes the inference
/// forward; see [`LoraConfig`]).
#[derive(serde::Deserialize)]
struct RawLoraConfig {
  /// mlx-lm `fine_tune_type` (`"lora"` / `"dora"` / `"full"`). Absent in PEFT
  /// configs. An *unknown* string here is still a hard parse error (the
  /// `FineTuneType` enum has no catch-all variant).
  #[serde(default)]
  fine_tune_type: Option<FineTuneType>,
  /// mlx-lm `num_layers`. Absent in PEFT configs.
  #[serde(default)]
  num_layers: Option<i32>,
  /// The mlx-lm-native nested `lora_parameters` object. Its **presence** is the
  /// signal that this is the mlx-lm-native shape (vs PEFT-flat).
  #[serde(default)]
  lora_parameters: Option<LoraParameters>,
  /// PEFT/HF `use_dora` — top-level in both shapes.
  #[serde(default)]
  use_dora: bool,
  /// PEFT `peft_type` (`"LORA"` for a LoRA adapter). Present only in PEFT
  /// configs; used both as a shape signal and to reject a non-LoRA PEFT kind.
  #[serde(default)]
  peft_type: Option<String>,
  /// PEFT top-level rank `r`.
  #[serde(default)]
  r: Option<i32>,
  /// PEFT top-level `lora_alpha`.
  #[serde(default)]
  lora_alpha: Option<f32>,
  /// PEFT top-level `target_modules` — a list of module-name suffixes or a
  /// single regex string.
  #[serde(default)]
  target_modules: Option<StrOrList>,
  /// PEFT top-level `exclude_modules` — same list-or-regex shape.
  #[serde(default)]
  exclude_modules: Option<StrOrList>,
  /// PEFT top-level `lora_dropout` (inference: ignored).
  #[serde(default)]
  lora_dropout: Option<f32>,
  /// PEFT `use_rslora` — rank-stabilized scaling (`alpha / sqrt(r)`).
  #[serde(default)]
  use_rslora: bool,
  /// PEFT `fan_in_fan_out` — base weight stored transposed.
  #[serde(default)]
  fan_in_fan_out: bool,
  /// PEFT `lora_bias` — a bias on the `lora_B` projection
  /// (`nn.Linear(r, out_features, bias=lora_bias)`). When `true` the adapter
  /// ships a `lora_B.bias` tensor that PEFT adds (scaled) in the forward — a
  /// surface mlxrs's mlx-lm-faithful `LoRALinear` does not carry, so a `true`
  /// value is rejected by the `Deserialize` (a silent drop would give wrong
  /// inference). Defaults `false` (the overwhelmingly common case).
  #[serde(default)]
  lora_bias: bool,
  /// PEFT `bias` (`Literal["none", "all", "lora_only"]`, default `"none"`) —
  /// which base/adapter bias terms are *trained and saved* with the adapter
  /// (`peft` `lora/config.py` `bias`; `utils/save_and_load.py` keeps `"bias" in
  /// k` tensors when this is `"all"` / `"lora_only"`). A non-`"none"` value
  /// ships trained `.bias` tensors that affect inference — mlxrs's
  /// mlx-lm-faithful [`LoRALinear`] has no adapted-bias slot, so the
  /// `Deserialize` rejects it (a silent drop would give wrong inference). Absent
  /// / `"none"` (the overwhelmingly common case) is fine.
  #[serde(default)]
  bias: Option<String>,
  /// PEFT `modules_to_save` (`Optional[list[str]]`, default `None`) — extra full
  /// modules (e.g. a resized `embed_tokens` / classifier head) trained and saved
  /// *in full* alongside the low-rank factors (`peft` `lora/config.py`
  /// `modules_to_save`). A non-empty list ships full module weights that affect
  /// inference — mlxrs's low-rank loader has no saved-full-module slot, so the
  /// `Deserialize` rejects it rather than silently dropping those weights.
  /// Absent / `[]` is fine.
  #[serde(default)]
  modules_to_save: Option<Vec<String>>,
  /// PEFT `layers_to_transform` — an int or a list of ints.
  #[serde(default)]
  layers_to_transform: Option<IntOrList>,
  /// PEFT `layers_pattern` — a string or a list of strings.
  #[serde(default)]
  layers_pattern: Option<StrOrList>,
  /// PEFT `rank_pattern` — `{module-pattern: rank}`. Deserialized into an
  /// insertion-ordered [`OrderedPattern`] (not a `HashMap`) so [`pattern_lookup`]
  /// reproduces PEFT `get_pattern_key`'s first-match-in-dict-order tie-break.
  #[serde(default)]
  rank_pattern: Option<OrderedPattern<i32>>,
  /// PEFT `alpha_pattern` — `{module-pattern: alpha}`. Insertion-ordered like
  /// `rank_pattern` (same `get_pattern_key` matching).
  #[serde(default)]
  alpha_pattern: Option<OrderedPattern<f32>>,
  /// PEFT `use_qalora` (`bool`, default `false`) — Quantization-Aware LoRA: the
  /// `lora_A` input is **average-pooled** in groups of `qalora_group_size`
  /// before the low-rank matmul (`peft` `lora/config.py` `use_qalora`). This
  /// changes the forward, so a `true` value cannot be silently ignored — the
  /// `Deserialize` rejects it. `false` (the default) is the normal LoRA path.
  /// PEFT's companion `qalora_group_size` (`int`, default `16`) is *not* a
  /// field here: it is meaningful only when `use_qalora` is `true` (already
  /// rejected), so it is left to parse-and-drop like the training-only fields.
  #[serde(default)]
  use_qalora: bool,
  /// PEFT `alora_invocation_tokens` (`Optional[list[int]]`, default `None`) —
  /// Activated-LoRA: the adapter is applied **only** to tokens at/after the
  /// invocation sequence (`peft` `lora/config.py` `alora_invocation_tokens`).
  /// A non-`None` value makes the adapter token-position-dependent — wrong if
  /// applied unconditionally — so the `Deserialize` rejects it.
  #[serde(default)]
  alora_invocation_tokens: Option<serde_json::Value>,
  /// PEFT `velora_config` (`Optional[...]`, default `None`) — VeLoRA swaps in a
  /// custom backward storing compressed activations (`peft` `lora/config.py`
  /// `velora_config`). It alters the adapter's numerics, so a non-`None` value
  /// is rejected by the `Deserialize`.
  #[serde(default)]
  velora_config: Option<serde_json::Value>,
  /// PEFT `monteclora_config` (`Optional[...]`, default `None`) — MonteCLoRA
  /// adds variational Monte-Carlo sampling on top of the LoRA adapters (`peft`
  /// `lora/config.py` `monteclora_config`). It changes the adapter's forward,
  /// so a non-`None` value is rejected by the `Deserialize`.
  #[serde(default)]
  monteclora_config: Option<serde_json::Value>,
  /// **Every remaining top-level key**, captured verbatim (`#[serde(flatten)]`)
  /// rather than dropped. This is the backstop for the *reject-unknown-active*
  /// posture: PEFT's `LoraConfig` is a 75-field, ever-growing zoo, and any field
  /// not explicitly modeled above lands here. On the PEFT-flat path the
  /// `Deserialize` walks this map and **rejects** any key that is neither
  /// modeled nor on the [`benign-ignore`](is_benign_ignore_field) allowlist and
  /// is set to an [`active`](is_active_config_value) value (anything other than
  /// `null` / `false`), so a *future* forward-switching variant fails loudly
  /// instead of silently running as vanilla LoRA — without a code change per new
  /// field. The modeled fields above are consumed by serde first, so they never
  /// appear here; only un-modeled keys do.
  #[serde(flatten)]
  extra: HashMap<String, serde_json::Value>,
}

/// A PEFT field that is **either** a single string **or** a list of strings
/// (`target_modules` / `exclude_modules` / `layers_pattern` —
/// `Optional[Union[list[str], str]]`).
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum StrOrList {
  /// `["q_proj", "v_proj"]`.
  List(Vec<String>),
  /// `"...single string / regex..."`.
  One(String),
}

/// A PEFT field that is **either** a single int **or** a list of ints
/// (`layers_to_transform` — `Optional[Union[list[int], int]]`).
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum IntOrList {
  /// `[0, 1, 5]`.
  List(Vec<i32>),
  /// `3`.
  One(i32),
}

/// PEFT's `INCLUDE_LINEAR_LAYERS_SHORTHAND` (`utils/constants.py`) — the
/// `target_modules` string that means "all eligible linear modules", expanded
/// by `_maybe_include_all_linear_layers` rather than matched as a regex.
const ALL_LINEAR_SENTINEL: &str = "all-linear";

/// Build a [`ModuleMatcher`] from a PEFT `target_modules` / `exclude_modules`
/// value: the list form is a [`ModuleMatcher::List`]; the single-string form is
/// a *regex* — compiled here, and a compile failure is a recoverable
/// deserialize error (a malformed regex must not silently match nothing).
///
/// The string `"all-linear"` (case-insensitive — PEFT lowercases before the
/// compare) is the **sentinel** [`ModuleMatcher::AllLinear`], NOT a regex:
/// `is_target_modules` gates it to `target_modules` only, mirroring PEFT's
/// `_maybe_include_all_linear_layers` (which rewrites `target_modules` alone).
/// An `"all-linear"` in `exclude_modules` has no PEFT meaning, so it falls
/// through to the regex path (it would match the literal string `all-linear`,
/// i.e. nothing — exactly PEFT's behavior, which never special-cases it there).
fn module_matcher_from<E: serde::de::Error>(
  value: StrOrList,
  field: &str,
  is_target_modules: bool,
) -> std::result::Result<ModuleMatcher, E> {
  match value {
    StrOrList::List(names) => Ok(ModuleMatcher::List(names)),
    StrOrList::One(pattern) => {
      // PEFT's `all-linear` shorthand — `target_modules.lower() ==
      // INCLUDE_LINEAR_LAYERS_SHORTHAND`. Special-cased BEFORE the regex
      // compile so it is NOT matched literally (a literal full-match would
      // select nothing).
      if is_target_modules && pattern.eq_ignore_ascii_case(ALL_LINEAR_SENTINEL) {
        return Ok(ModuleMatcher::AllLinear);
      }
      let re = Regex::new(&pattern).map_err(|e| {
        E::custom(format!(
          "adapter_config.json `{field}` is the regex string {pattern:?}, which failed to \
           compile: {e}"
        ))
      })?;
      Ok(ModuleMatcher::Regex(Box::new(re)))
    }
  }
}

/// Whether a JSON value for an *un-modeled* PEFT config field counts as
/// **active** — i.e. it might switch the inference forward and so must be
/// rejected rather than silently dropped.
///
/// PEFT's variant-gating fields default to `None` (serde → JSON `null`) or
/// `False` when *off*; `resolve_lora_variant` (`peft` `lora/layer.py`) only
/// dispatches a variant when such a field is non-`None` / truthy. So a `null`
/// or `false` value for an un-modeled field is provably the inactive default and
/// is safe to ignore; **anything else** (a number, a string, a non-empty list,
/// an object, `true`) is treated as active and rejected. An empty list / empty
/// object is conservatively treated as active too — it is not a PEFT default for
/// any forward-switching field, so a real config never carries one, and erring
/// toward rejection keeps the backstop loud.
fn is_active_config_value(value: &serde_json::Value) -> bool {
  !matches!(
    value,
    serde_json::Value::Null | serde_json::Value::Bool(false)
  )
}

/// Whether `field` is a PEFT `LoraConfig` (or inherited `PeftConfig`) key that
/// is **safe to accept and ignore even when set** — metadata, training-only, or
/// factor-initialization-only fields that never change how *already-saved* LoRA
/// factors run at inference.
///
/// This is the allowlist half of the *reject-unknown-active* rule: a key that is
/// neither modeled (a `RawLoraConfig` struct field) nor on this list is unknown,
/// and an unknown key set to an [`active`](is_active_config_value) value is
/// rejected (see [`reject_unknown_active_peft_fields`]). Grounded in
/// `config.py` docstrings and whether `lora/layer.py`'s forward /
/// `resolve_lora_variant` consults the field:
///
/// - **Inherited `PeftConfig` metadata** (`config.py` parent): `task_type`,
///   `auto_mapping`, `peft_version`, `base_model_name_or_path`, `revision`,
///   `inference_mode` — provenance / bookkeeping, never read in the forward.
/// - **Factor-initialization strategies** — *pure factor seeds* applied at
///   training time; the saved factors already embody them, so they are no-ops
///   at load. The benign `init_lora_weights` values are `true` / `false` /
///   `gaussian` / `eva` / `orthogonal`, plus the companion config blocks
///   `eva_config`, `corda_config`, `lora_ga_config`, `loftq_config` (carried,
///   never read). **`init_lora_weights` string values are an allowlist:** only
///   the pure factor seeds `gaussian` / `eva` / `orthogonal` (see
///   [`is_factor_only_init_mode`]) load; every OTHER string is REJECTED before
///   this allowlist is consulted (see [`reject_unknown_active_peft_fields`]).
///   The base-weight-mutating modes `olora` / `pissa*` / `corda*` / `loftq` /
///   `lora_ga` each subtract a low-rank residual from the *base layer weight*
///   at init (so raw factors pair with a modified base), and any unknown/future
///   mode rejects by default. The bare `init_lora_weights` key is benign here
///   only for the booleans `true` / `false` (PEFT's conversion path rewrites
///   converted adapters to `init_lora_weights = true`, which loads).
/// - **Training-only / runtime knobs**: `megatron_config`, `megatron_core`
///   (Megatron parallel-linear construction at train time), `runtime_config`
///   (explicitly *not saved or restored* — `config.py`), `qalora_group_size`
///   (meaningful only with `use_qalora`, itself rejected), `ensure_weight_tying`
///   (re-ties weights for `modules_to_save` / `target_modules` — both rejected
///   when active; tying alone does not change a pure-LoRA forward on saved
///   factors).
///
/// Fields that *do* switch inference (`arrow_config`, `use_bdlora`,
/// `layer_replication`, `trainable_token_indices`, `target_parameters`, …) are
/// deliberately **absent** so the generic rule rejects them when active.
fn is_benign_ignore_field(field: &str) -> bool {
  matches!(
    field,
    // Inherited PeftConfig metadata (config.py parent).
    "task_type"
      | "auto_mapping"
      | "peft_version"
      | "base_model_name_or_path"
      | "revision"
      | "inference_mode"
      // Factor-init strategies + their config blocks (train-time seed only;
      // saved factors already reflect them). `init_lora_weights` STRING values
      // are an allowlist (only gaussian / eva / orthogonal) enforced as a REJECT
      // before this benign pass (see `reject_unknown_active_peft_fields` /
      // `is_factor_only_init_mode`); the bare key is benign here only for the
      // booleans true / false.
      | "init_lora_weights"
      | "eva_config"
      | "corda_config"
      | "lora_ga_config"
      | "loftq_config"
      // Training-only / runtime knobs (no inference effect on saved factors).
      | "megatron_config"
      | "megatron_core"
      | "runtime_config"
      | "qalora_group_size"
      | "ensure_weight_tying"
  )
}

/// Whether an `init_lora_weights` **string** names a *pure factor seed* — an
/// init mode that touches only the LoRA factors and leaves the base weight
/// untouched, so the saved factors load correctly against this loader's
/// **unmodified** base.
///
/// This is an **allowlist** (closed set), deliberately the inverse of a
/// reject-list. PEFT's `reset_lora_parameters` (peft `tuners/lora/layer.py`
/// `:225-273`) dispatches init modes against a closed set and **raises on any
/// unrecognized string**; within that set the base-*mutating* modes are
/// `pissa`/`corda` (prefix-dispatched: `startswith("pissa")` `:225`,
/// `startswith("corda")` `:228`), `olora` (`:231`), `loftq` (`:234`),
/// `lora_ga` (`:242`) — each subtracts a low-rank residual from
/// `base_layer.weight.data` (`olora_init:361`, `pissa_init:396`,
/// `corda_init:477`, `loftq_init:499`, `lora_ga_init:631`). The *only* pure
/// factor seeds are `eva` (`:237`, which PEFT then rewrites to
/// `init_lora_weights = true`, `eva.py:526`), `orthogonal` (`:239`), and
/// `gaussian` (`:273`).
///
/// Returning `false` for **everything else** — the base-mutating modes, their
/// prefixed variants (`pissa_niter_<N>`, any `corda*`), and any unknown/future
/// string — means [`reject_unknown_active_peft_fields`] rejects them loudly
/// with no code change, instead of silently running raw factors against the
/// wrong base. (Booleans `true` / `false` are handled separately by the benign
/// allowlist; a converted checkpoint reports `init_lora_weights = true`.)
/// Matching is case-insensitive (PEFT writes these lowercase; be lenient).
fn is_factor_only_init_mode(mode: &str) -> bool {
  matches!(
    mode.to_ascii_lowercase().as_str(),
    "gaussian" | "eva" | "orthogonal"
  )
}

/// The *reject-unknown-active* backstop for the **PEFT-flat** path: reject any
/// top-level key that is neither modeled nor [`benign`](is_benign_ignore_field)
/// and is set to an [`active`](is_active_config_value) value.
///
/// # Why this exists
///
/// PEFT's `LoraConfig` is a 75-field, forever-growing dataclass; many fields
/// switch the inference forward (`resolve_lora_variant`, `lora/layer.py`). A
/// reactive *reject-list* (name each unsupported variant) is whack-a-mole —
/// every new PEFT release can add a forward-switching field that this loader
/// would then silently run as **vanilla LoRA**, at the wrong behavior. This rule
/// flips the posture: anything not explicitly understood, when set to an active
/// value, fails loudly. New unmodeled variants are caught with *no code change*.
///
/// # The rule
///
/// `extra` ([`RawLoraConfig::extra`], a `#[serde(flatten)]` catch-all) holds
/// exactly the keys serde did **not** bind to a modeled struct field. For each
/// such key: if it is on the benign allowlist, skip it; otherwise, if its value
/// is active (not `null` / `false`), return [`Error::Backend`](crate::Error)
/// (via the deserializer's error type) naming the field as an unsupported /
/// unmodeled PEFT field. Inactive unknowns (`null` / `false` — PEFT's "off"
/// default for its variant fields) are ignored, so a config that merely *carries*
/// a defaulted future field still loads.
///
/// `init_lora_weights` string values are an **allowlist** ([`is_factor_only_init_mode`]):
/// only the pure factor seeds (`gaussian` / `eva` / `orthogonal`) load; every
/// other string is a hard reject here. The base-weight-mutating modes (`olora`,
/// `pissa` incl. `pissa_niter_<N>`, `corda` incl. prefixed variants, `loftq`,
/// `lora_ga`) subtract a low-rank residual from the base layer weight at init,
/// so a raw checkpoint trained with one is not interchangeable with a plain-LoRA
/// load on the unmodified base; unknown/future modes reject by default. The
/// booleans `true` / `false` are not strings and stay benign.
fn reject_unknown_active_peft_fields<E: serde::de::Error>(
  raw: &RawLoraConfig,
) -> std::result::Result<(), E> {
  // `init_lora_weights` may be a STRING naming an init mode. Only the pure
  // factor seeds (`gaussian` / `eva` / `orthogonal`) leave the base weight
  // untouched and are safe to load as plain LoRA — this is an ALLOWLIST
  // ([`is_factor_only_init_mode`]). Every OTHER string is rejected: the
  // base-weight-mutating modes (`olora`, `pissa*`, `corda*`, `loftq`,
  // `lora_ga`) subtract a low-rank residual from `base_layer.weight.data` at
  // init (peft `lora/layer.py`: `olora_init:361`, `pissa_init:396`,
  // `corda_init:477`, `loftq_init:499`, `lora_ga_init:631`), so their raw saved
  // factors are paired with a *modified* base — applying them to this loader's
  // UNMODIFIED base is silently wrong inference; and any unknown/future mode is
  // rejected by default (PEFT itself only accepts a closed set and raises on the
  // rest, `reset_lora_parameters` `:225-273`). The allowlist means a new PEFT
  // init mode fails loud here with NO code change — unlike a reject-list, which
  // missed prefixed variants like `corda_v1`. (Booleans `true` / `false` are
  // not strings, so they fall through to the benign allowlist; PEFT's CONVERSION
  // path rewrites `init_lora_weights = True`, `eva.py:526`, so converted
  // checkpoints report `true` and stay loadable.)
  if let Some(init) = raw.extra.get("init_lora_weights")
    && let Some(mode) = init.as_str()
    && !is_factor_only_init_mode(mode)
  {
    return Err(E::custom(format!(
      "adapter_config.json sets `init_lora_weights: {mode:?}` — this loader only supports the \
       pure factor-seed init modes (`gaussian` / `eva` / `orthogonal`) and the booleans \
       `true` / `false`. Other modes either mutate the base model weight at init (`olora`, \
       `pissa` incl. `pissa_niter_<N>`, `corda` incl. prefixed variants, `loftq`, `lora_ga` — \
       they subtract a low-rank residual from `base_layer.weight`, pairing the raw saved factors \
       with a *modified* base) or are not understood; applying them to this loader's unmodified \
       base would be silently wrong, so they are rejected. (A checkpoint converted via PEFT's \
       conversion path reports `init_lora_weights: true` and loads fine.)"
    )));
  }

  // Generic backstop: every key serde could not bind to a modeled field is in
  // `extra`. Reject any that is neither benign nor inactive.
  for (field, value) in &raw.extra {
    if is_benign_ignore_field(field) {
      continue;
    }
    if is_active_config_value(value) {
      return Err(E::custom(format!(
        "adapter_config.json sets the unsupported / unmodeled PEFT field {field:?} to an active \
         value; this loader models only a known subset of PEFT `LoraConfig` and rejects any \
         other field that is set (not `null` / `false`), so a future forward-switching variant \
         fails loudly instead of silently running as vanilla LoRA. If {field:?} does not affect \
         inference, it must be added to the benign-ignore allowlist; otherwise it needs explicit \
         support."
      )));
    }
  }
  Ok(())
}

/// Reject any PEFT *exotic LoRA variant* that is set to a non-default — the
/// shape-independent guard run before [`LoraConfig`]'s shape detection.
///
/// The four exotic variants (`peft` `lora/config.py`) each change the
/// adapter's **inference** forward:
///
/// - `use_qalora` — Quantization-Aware LoRA average-pools the `lora_A` input
///   in groups before the low-rank matmul.
/// - `alora_invocation_tokens` — Activated-LoRA applies the adapter only to
///   tokens at/after an invocation sequence (token-position-dependent).
/// - `velora_config` — VeLoRA alters the adapter numerics via a custom
///   compressed-activation backward.
/// - `monteclora_config` — MonteCLoRA adds variational Monte-Carlo sampling
///   over the LoRA adapters.
///
/// Loading such an adapter as plain LoRA would run it at the *wrong behavior*,
/// so each is a recoverable `Deserialize` error naming the variant — never a
/// silent drop (the treatment the *training-only* fields get). The check is
/// **shape-independent**: it runs on the raw config before the `lora_parameters`
/// early return and the `is_peft` gate, so an exotic field is rejected whether
/// the config is mlx-lm-native, PEFT-flat, or carries no shape marker at all.
///
/// At their PEFT defaults (`use_qalora: false`; the three optionals `None` —
/// including a JSON `null`, which serde maps to `None` for `Option<_>`) none of
/// these trip. `qalora_group_size` is *not* checked: it is meaningful only when
/// `use_qalora` is `true` (already rejected), so a stray default `16` is
/// harmless and must still parse.
fn reject_exotic_variants<E: serde::de::Error>(raw: &RawLoraConfig) -> std::result::Result<(), E> {
  if raw.use_qalora {
    return Err(E::custom(
      "adapter_config.json sets `use_qalora: true` — Quantization-Aware LoRA pools the lora_A \
       input before the low-rank matmul, a forward this loader does not implement; a QALoRA \
       adapter is not supported (loading it as plain LoRA would be wrong)",
    ));
  }
  if raw.alora_invocation_tokens.is_some() {
    return Err(E::custom(
      "adapter_config.json sets `alora_invocation_tokens` — Activated-LoRA applies the adapter \
       only to tokens at/after an invocation sequence, a token-position-dependent forward this \
       loader does not implement; an aLoRA adapter is not supported (applying it \
       unconditionally would be wrong)",
    ));
  }
  if raw.velora_config.is_some() {
    return Err(E::custom(
      "adapter_config.json carries a `velora_config` — VeLoRA alters the adapter's numerics \
       with a custom compressed-activation backward; a VeLoRA adapter is not supported by this \
       loader (loading it as plain LoRA would be wrong)",
    ));
  }
  if raw.monteclora_config.is_some() {
    return Err(E::custom(
      "adapter_config.json carries a `monteclora_config` — MonteCLoRA adds variational \
       Monte-Carlo sampling over the LoRA adapters, changing the forward; a MonteCLoRA adapter \
       is not supported by this loader (loading it as plain LoRA would be wrong)",
    ));
  }
  Ok(())
}

impl<'de> serde::Deserialize<'de> for LoraConfig {
  /// Deserialize an `adapter_config.json` of **either** on-disk shape into the
  /// single normalized [`LoraConfig`].
  ///
  /// This is the dual-shape bridge the [type docs](LoraConfig) describe. It
  /// runs a *permissive* deserialize into a private `RawLoraConfig` (which
  /// captures the union of both shapes' keys, all optional), then a
  /// **shape-detection** normalization:
  ///
  /// - **mlx-lm-native** — when the raw form has a `lora_parameters` object,
  ///   that object IS the [`LoraParameters`]; `fine_tune_type` / `num_layers` /
  ///   `use_dora` are read from the top level; [`selection`](LoraConfig::selection)
  ///   is [`AdapterSelection::MlxLm`]. The flat PEFT keys are ignored — a
  ///   config carrying `lora_parameters` is unambiguously mlx-lm-native.
  /// - **PEFT / HuggingFace** — when there is **no** `lora_parameters` object
  ///   but the raw form carries top-level `r` / `lora_alpha` /
  ///   `target_modules` / `peft_type`, the full PEFT surface is mapped: `r` →
  ///   [`rank`](LoraParameters::rank) (default [`DEFAULT_LORA_RANK`]),
  ///   `lora_alpha` → [`alpha`](LoraParameters::alpha) (default
  ///   [`DEFAULT_PEFT_LORA_ALPHA`]), `lora_dropout` →
  ///   [`dropout`](LoraParameters::dropout); `target_modules` /
  ///   `exclude_modules` / `use_rslora` / `fan_in_fan_out` /
  ///   `layers_to_transform` / `layers_pattern` / `rank_pattern` /
  ///   `alpha_pattern` into a [`PeftSelection`]. [`selection`](LoraConfig::selection)
  ///   is [`AdapterSelection::Peft`] — **no** `num_layers` window.
  /// - **neither** — a bare / training-only-keys config takes every default
  ///   ([`AdapterSelection::MlxLm`] with [`DEFAULT_NUM_LAYERS`]).
  ///
  /// # Errors
  ///
  /// - A `peft_type` other than `"LORA"` (case-insensitive) is a *different
  ///   adapter kind* (`LOHA`, `LOKR`, `IA3`, prompt-tuning, …) — this module
  ///   loads LoRA/DoRA adapters only, so a non-LoRA `peft_type` is a
  ///   recoverable deserialize error.
  /// - A `target_modules` / `exclude_modules` regex string that fails to
  ///   compile is a recoverable deserialize error.
  ///
  /// [`rank`]: LoraParameters::rank
  /// [`alpha`]: LoraParameters::alpha
  /// [`dropout`]: LoraParameters::dropout
  fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    use serde::de::Error as _;

    let raw = RawLoraConfig::deserialize(deserializer)?;

    // ── exotic-variant rejection (SHAPE-INDEPENDENT — must run first) ──
    // PEFT's exotic LoRA variants — `use_qalora`, `alora_invocation_tokens`,
    // `velora_config`, `monteclora_config` — each, when set, change the
    // adapter's *inference* forward (QALoRA pools the `lora_A` input, aLoRA
    // gates by token position, VeLoRA/MonteCLoRA alter the adapter numerics).
    // Unlike the training/init-only fields (`init_lora_weights`, `loftq_config`,
    // … — silently dropped, accept-and-ignore), these cannot be ignored:
    // loading such an adapter as plain LoRA would run it at the *wrong
    // behavior*. The check is performed HERE, before the `lora_parameters`
    // early return and the `is_peft` gate, so it is shape-independent — an
    // adapter that carries an exotic field is rejected whether it is written in
    // the mlx-lm-native nested shape, the PEFT-flat shape, or a shape with no
    // PEFT markers at all. (`qalora_group_size` is meaningful only with
    // `use_qalora`, so it is not a signal on its own — a default `16` alone
    // parses harmlessly; only `use_qalora: true` rejects.)
    reject_exotic_variants::<D::Error>(&raw)?;

    // ── shape detection ──
    // A `lora_parameters` object is the unambiguous mlx-lm-native marker; with
    // it present, the flat PEFT keys are NOT consulted (a real PEFT config has
    // no `lora_parameters` nesting, so this branch only fires for mlx-lm
    // configs). Without it, top-level `r` / `lora_alpha` / `target_modules` /
    // `peft_type` marks the PEFT-flat shape.
    if let Some(lora_parameters) = raw.lora_parameters {
      // mlx-lm-native: the nested object is the parameter block verbatim, and
      // selection is the trailing-`num_layers` window.
      return Ok(LoraConfig {
        fine_tune_type: raw.fine_tune_type.unwrap_or_default(),
        lora_parameters,
        use_dora: raw.use_dora,
        selection: AdapterSelection::MlxLm {
          num_layers: raw.num_layers.unwrap_or_else(default_num_layers),
        },
      });
    }

    let is_peft = raw.peft_type.is_some()
      || raw.r.is_some()
      || raw.lora_alpha.is_some()
      || raw.target_modules.is_some();

    if is_peft {
      // PEFT-flat: validate `peft_type` (LoRA only) and map the full surface.
      if let Some(peft_type) = &raw.peft_type
        && !peft_type.eq_ignore_ascii_case("LORA")
      {
        return Err(D::Error::custom(format!(
          "adapter_config.json `peft_type` is {peft_type:?}, but this loader handles only \
           LoRA/DoRA adapters (`peft_type` \"LORA\"); a different PEFT method (LOHA, LOKR, \
           IA3, prompt-tuning, …) is not supported"
        )));
      }
      // `lora_bias: true` ships a `lora_B.bias` tensor PEFT adds in the
      // forward — mlxrs's mlx-lm-faithful `LoRALinear` has no B-bias slot, so
      // reject it (a silent drop would give wrong inference). `false` (the
      // default, ~every adapter) is fine.
      if raw.lora_bias {
        return Err(D::Error::custom(
          "adapter_config.json sets `lora_bias: true` (a bias on the lora_B projection); this \
           loader's LoRALinear has no lora_B-bias term, so a `lora_bias` adapter is not \
           supported (it would silently drop the bias)",
        ));
      }
      // PEFT `bias` (`"none"` / `"all"` / `"lora_only"`) — a non-`"none"` value
      // trains+saves `.bias` tensors (`utils/save_and_load.py` keeps `"bias" in
      // k`) that PEFT adds in the forward; mlxrs's LoRALinear has no
      // adapted-bias slot, so reject it loudly rather than silently drop the
      // bias tensors (which `translate_peft_keys` would otherwise discard).
      // Absent / `"none"` is the default (~every adapter). An *unknown* value is
      // also rejected (PEFT's `Literal` would; we mirror that).
      if let Some(bias) = &raw.bias
        && !bias.eq_ignore_ascii_case("none")
      {
        return Err(D::Error::custom(format!(
          "adapter_config.json sets `bias: {bias:?}` (PEFT trains+saves base/adapter `.bias` \
           tensors for `\"all\"` / `\"lora_only\"`); this loader's LoRALinear has no adapted-bias \
           slot, so a PEFT `bias` adapter is not supported (it would silently drop the bias \
           tensors)"
        )));
      }
      // PEFT `modules_to_save` — a non-empty list trains+saves *full* modules
      // (e.g. a resized `embed_tokens` / classifier head) alongside the low-rank
      // factors; mlxrs's low-rank loader has no saved-full-module slot, so
      // reject it rather than silently drop those full weights (which
      // `translate_peft_keys` would otherwise discard). Absent / `[]` is fine.
      if raw.modules_to_save.as_ref().is_some_and(|m| !m.is_empty()) {
        return Err(D::Error::custom(
          "adapter_config.json sets a non-empty `modules_to_save` (PEFT trains+saves these \
           modules in full alongside the LoRA factors); this loader applies only the low-rank \
           factors and has no saved-full-module slot, so a `modules_to_save` adapter is not \
           supported (it would silently drop the saved module weights)",
        ));
      }
      // ── reject-unknown-active backstop (PEFT-flat only) ──
      // The explicit named rejects above keep their clearer messages, but PEFT's
      // `LoraConfig` is a 75-field, ever-growing zoo: any forward-switching field
      // NOT modeled above would otherwise be silently dropped and the adapter run
      // as vanilla LoRA. This generic rule rejects ANY un-modeled, non-benign key
      // that is set to an active value (and `init_lora_weights: "loftq"`), so a
      // future variant fails loudly without a per-field code change. Scope is the
      // PEFT-flat branch — the mlx-lm-native nested shape (handled by the
      // `lora_parameters` early return) is our own small, well-defined format and
      // keeps its existing accept-and-ignore behavior.
      reject_unknown_active_peft_fields::<D::Error>(&raw)?;

      let target_modules = match raw.target_modules {
        None => None,
        Some(v) => Some(module_matcher_from::<D::Error>(v, "target_modules", true)?),
      };
      let exclude_modules = match raw.exclude_modules {
        None => None,
        Some(v) => Some(module_matcher_from::<D::Error>(
          v,
          "exclude_modules",
          false,
        )?),
      };
      // `layers_to_transform`: an int normalizes to a one-element list (PEFT's
      // `layer_index == layer_indexes` int case ≡ membership in `[idx]`).
      let layers_to_transform = raw.layers_to_transform.map(|v| match v {
        IntOrList::List(xs) => xs,
        IntOrList::One(x) => vec![x],
      });
      // `layers_pattern`: a string normalizes to a one-element list.
      let layers_pattern = match raw.layers_pattern {
        None => Vec::new(),
        Some(StrOrList::List(xs)) => xs,
        Some(StrOrList::One(x)) => vec![x],
      };
      let peft = PeftSelection {
        target_modules,
        exclude_modules,
        layers_to_transform,
        layers_pattern,
        rank_pattern: ordered_pattern(raw.rank_pattern),
        alpha_pattern: ordered_pattern(raw.alpha_pattern),
        use_rslora: raw.use_rslora,
        fan_in_fan_out: raw.fan_in_fan_out,
      };
      let lora_parameters = LoraParameters {
        rank: raw.r.unwrap_or_else(default_rank),
        // PEFT has no literal-`scale` field — the scale is `lora_alpha / r`
        // (or `/ sqrt(r)`), resolved per-module by `PeftSelection::scale_for`.
        scale: None,
        // PEFT `lora_alpha` defaults to 8 (NOT the mlx-lm `20.0` literal).
        alpha: Some(raw.lora_alpha.unwrap_or(DEFAULT_PEFT_LORA_ALPHA)),
        // PEFT selection lives in `PeftSelection`, not `keys`.
        keys: Vec::new(),
        dropout: raw.lora_dropout,
      };
      return Ok(LoraConfig {
        // PEFT configs have no `fine_tune_type` — LoRA by default; DoRA is
        // carried by `use_dora`.
        fine_tune_type: FineTuneType::Lora,
        lora_parameters,
        use_dora: raw.use_dora,
        selection: AdapterSelection::Peft(peft),
      });
    }

    // Neither shape: a bare / training-only-keys config — every default.
    Ok(LoraConfig {
      fine_tune_type: raw.fine_tune_type.unwrap_or_default(),
      lora_parameters: LoraParameters::default(),
      use_dora: raw.use_dora,
      selection: AdapterSelection::MlxLm {
        num_layers: raw.num_layers.unwrap_or_else(default_num_layers),
      },
    })
  }
}

impl LoraConfig {
  /// Parse a [`LoraConfig`] from an in-memory `adapter_config.json` string.
  ///
  /// Mirrors mlx-lm `json.load(adapter_config.json)` restricted to the typed
  /// subset. A serde failure (malformed JSON) maps to [`Error::Backend`] with
  /// the underlying cause — the codebase's config-parse error convention (twin
  /// of [`crate::lm::load::Config::from_json`]).
  pub fn from_json(json: &str) -> Result<LoraConfig> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "LoraConfig::from_json",
        "adapter_config.json",
        e,
      ))
    })
  }

  /// Whether this config selects DoRA (weight-decomposed) — either
  /// `fine_tune_type: "dora"` or the PEFT `use_dora: true` flag, mirroring
  /// mlx-lm `use_dora=(fine_tune_type == "dora")` (`tuner/utils.py:135`) plus
  /// the HF `use_dora` convention.
  pub fn is_dora(&self) -> bool {
    self.fine_tune_type == FineTuneType::Dora || self.use_dora
  }

  /// The resolved low-rank scale **for the mlx-lm-native path**
  /// (see [`LoraParameters::resolved_scale`]). On a PEFT config the scale is
  /// per-module — use [`scale_for`](Self::scale_for) — so this returns the
  /// `lora_parameters`-resolved value, which for a PEFT config is the
  /// no-pattern, non-rsLoRA baseline (`lora_alpha / r`).
  pub fn scale(&self) -> f32 {
    self.lora_parameters.resolved_scale()
  }

  /// The effective low-rank scale **for one module path** — the value that
  /// scales that module's low-rank term.
  ///
  /// - **mlx-lm-native** — the single config-wide scale (`alpha / rank`, or the
  ///   literal `scale`); `module_path` is unused.
  /// - **PEFT** — PEFT's per-module `update_layer` scaling: `lora_alpha / r`,
  ///   or `lora_alpha / sqrt(r)` under `use_rslora`, with `rank_pattern` /
  ///   `alpha_pattern` overriding `r` / `lora_alpha` for a matching module
  ///   (see [`PeftSelection::scale_for`]).
  pub fn scale_for(&self, module_path: &str) -> f32 {
    match &self.selection {
      AdapterSelection::MlxLm { .. } => self.lora_parameters.resolved_scale(),
      AdapterSelection::Peft(peft) => peft.scale_for(
        module_path,
        self.lora_parameters.rank,
        self
          .lora_parameters
          .alpha
          .unwrap_or(DEFAULT_PEFT_LORA_ALPHA),
      ),
    }
  }

  /// The effective low-rank **rank** for one module path — the config-wide
  /// `rank` for the mlx-lm-native path, or PEFT's `rank_pattern`-overridden
  /// rank for a PEFT config (see [`PeftSelection::rank_for`]). This is the rank
  /// the module's `lora_a` / `lora_b` factors must agree with (the
  /// config-vs-tensor cross-check).
  pub fn rank_for(&self, module_path: &str) -> i32 {
    match &self.selection {
      AdapterSelection::MlxLm { .. } => self.lora_parameters.rank,
      AdapterSelection::Peft(peft) => peft.rank_for(module_path, self.lora_parameters.rank),
    }
  }

  /// The config-wide low-rank dimension `r` (PEFT `r` / mlx-lm
  /// `lora_parameters.rank`) — the default before any `rank_pattern` override.
  pub fn rank(&self) -> i32 {
    self.lora_parameters.rank
  }

  /// The PEFT selection block, when this config is PEFT-shaped; `None` for an
  /// mlx-lm-native config.
  pub fn peft(&self) -> Option<&PeftSelection> {
    match &self.selection {
      AdapterSelection::Peft(p) => Some(p),
      AdapterSelection::MlxLm { .. } => None,
    }
  }

  /// Whether the base weights this adapter targets are stored transposed
  /// (`[in_features, out_features]`) — PEFT `fan_in_fan_out`. Always `false`
  /// for an mlx-lm-native config (mlx-lm stores weights `[out, in]`).
  pub fn fan_in_fan_out(&self) -> bool {
    self.peft().is_some_and(|p| p.fan_in_fan_out)
  }
}

// ───────────────────────── adapter weights ─────────────────────────

/// The per-layer adapter parameters loaded from `adapters.safetensors` for one
/// target path: the low-rank factors plus (DoRA only) the magnitude.
///
/// These are the *named* arrays mlx-lm's `LoRALinear` registers (`lora_a` /
/// `lora_b`) plus DoRA's `m` (`tuner/dora.py:90`). At inference they come
/// entirely from the safetensors file — there is no random/zero init here.
///
/// Does **not** derive `Clone` ([`Array`] deliberately doesn't — see
/// [`Array::try_clone`]); use [`AdapterParams::try_clone`] for the
/// refcount-sharing dup.
#[derive(Debug)]
pub struct AdapterParams {
  /// `lora_a` — shape `[input_dims, r]` (mlx-lm `tuner/lora.py:88-92`).
  pub lora_a: Array,
  /// `lora_b` — shape `[r, output_dims]` (mlx-lm `tuner/lora.py:93`).
  pub lora_b: Array,
  /// DoRA magnitude `m` — shape `[output_dims]` (mlx-lm `tuner/dora.py:90`);
  /// `None` for plain LoRA.
  pub magnitude: Option<Array>,
}

impl AdapterParams {
  /// Refcount-sharing dup of all three slots (a fresh mlx handle over the same
  /// buffer per [`Array::try_clone`]; no data copy). Fallible because the
  /// mlx-c handle alloc can fail.
  pub fn try_clone(&self) -> Result<Self> {
    Ok(Self {
      lora_a: self.lora_a.try_clone()?,
      lora_b: self.lora_b.try_clone()?,
      magnitude: match &self.magnitude {
        Some(m) => Some(m.try_clone()?),
        None => None,
      },
    })
  }
}

// ──────────────────────────── base layer ────────────────────────────

/// The base linear a LoRA/DoRA layer wraps: either a **dense** weight (+
/// optional bias) or an MLX-**quantized** triple. Mirrors the
/// `Linear` / `QuantizedLinear` split mlx-lm's `LoRALinear.from_base` branches
/// on (`tuner/lora.py:22-23`) and swift's `LoRALinear` / `QLoRALinear`.
///
/// Constructed via [`BaseLinear::dense`] / [`BaseLinear::quantized`] (the
/// quantized ctor validates the `affine`/`fp`-mode bias arity, mirroring
/// [`crate::lm::nn::switch::QuantizedSwitchLinear::from_parts`]); the inner
/// arrays are read-only thereafter so the `(weight, scales, biases)` triple
/// stays internally consistent.
#[derive(Debug)]
pub enum BaseLinear {
  /// Dense base: `weight` is `[output_dims, input_dims]`, `bias` optional
  /// `[output_dims]`.
  Dense {
    /// `[output_dims, input_dims]` dense weight.
    weight: Array,
    /// Optional `[output_dims]` bias (`None` ⇒ `bias=False`).
    bias: Option<Array>,
  },
  /// Quantized base: the MLX `(weight, scales, biases)` packed triple plus the
  /// scheme parameters, and an optional dense `[output_dims]` output bias.
  Quantized {
    /// Packed `uint32` quantized weight.
    weight: Array,
    /// Per-group scales.
    scales: Array,
    /// Per-group biases (`affine` only; `None` for `mxfp4`/`mxfp8`/`nvfp4`).
    quant_biases: Option<Array>,
    /// Optional `[output_dims]` output bias.
    bias: Option<Array>,
    /// Quantization group size.
    group_size: i32,
    /// Quantization bit depth.
    bits: i32,
    /// Quantization mode (`"affine"` / `"mxfp4"` / …).
    mode: String,
  },
}

impl BaseLinear {
  /// Build a dense base from a `[output_dims, input_dims]` weight (+ optional
  /// `[output_dims]` bias). Verifies rank-2 weight and matching bias shape.
  pub fn dense(weight: Array, bias: Option<Array>) -> Result<Self> {
    let w_shape = weight.shape();
    let w_rank = w_shape.len();
    let w_output_dims = w_shape.first().copied().unwrap_or(0);
    if w_rank != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "BaseLinear::dense: weight must be 2-D [output_dims, input_dims]",
        w_rank as u32,
        w_shape,
      )));
    }
    if let Some(b) = &bias {
      let b_shape = b.shape();
      if b_shape.len() != 1 || b_shape[0] != w_output_dims {
        return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
          "BaseLinear::dense: bias must be [output_dims]",
          vec![w_output_dims],
          b_shape,
        )));
      }
    }
    Ok(BaseLinear::Dense { weight, bias })
  }

  /// Build a quantized base from the MLX `(weight, scales, biases)` triple plus
  /// the scheme parameters. Validates the per-mode bias arity (mirroring
  /// [`crate::lm::nn::switch::QuantizedSwitchLinear::from_parts`]): `affine`
  /// REQUIRES `quant_biases`; the float schemes (`mxfp4`/`mxfp8`/`nvfp4`)
  /// forbid it.
  pub fn quantized(
    weight: Array,
    scales: Array,
    quant_biases: Option<Array>,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: String,
  ) -> Result<Self> {
    match (mode.as_str(), &quant_biases) {
      ("affine", None) => {
        return Err(Error::MissingField(MissingFieldPayload::new(
          "BaseLinear::quantized",
          "quant_biases (affine mode requires it; mlx affine_quantize writes {w_q, scales, biases})",
        )));
      }
      ("mxfp4" | "mxfp8" | "nvfp4", Some(_)) => {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "BaseLinear::quantized: quant_biases",
          "must be None for scale-only modes (mxfp4/mxfp8/nvfp4 — mlx fp_quantize writes {w_q, scales})",
        )));
      }
      ("affine", Some(_)) | ("mxfp4" | "mxfp8" | "nvfp4", None) => {}
      (other, _) => {
        return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
          "BaseLinear::quantized: mode",
          other.to_string(),
          &["affine", "mxfp4", "mxfp8", "nvfp4"],
        )));
      }
    }
    if bits <= 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "BaseLinear::quantized: bits",
        "must be > 0",
        bits.to_string(),
      )));
    }
    if group_size <= 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "BaseLinear::quantized: group_size",
        "must be > 0",
        group_size.to_string(),
      )));
    }
    Ok(BaseLinear::Quantized {
      weight,
      scales,
      quant_biases,
      bias,
      group_size,
      bits,
      mode,
    })
  }

  /// The optional output bias (the dense `[output_dims]` addend, NOT the
  /// quantization `biases`). `None` matches `bias=False`.
  pub fn bias(&self) -> Option<&Array> {
    match self {
      BaseLinear::Dense { bias, .. } => bias.as_ref(),
      BaseLinear::Quantized { bias, .. } => bias.as_ref(),
    }
  }

  /// The dense `[output_dims, input_dims]` weight, **dequantizing** if this is
  /// a quantized base (mlx-lm `_dequantized_weight`, `tuner/dora.py:92-106`).
  /// Used by `fuse` and the DoRA forward (which need the float weight to form
  /// the adapted magnitude).
  pub fn dequantized_weight(&self) -> Result<Array> {
    match self {
      BaseLinear::Dense { weight, .. } => weight.try_clone(),
      BaseLinear::Quantized {
        weight,
        scales,
        quant_biases,
        group_size,
        bits,
        mode,
        ..
      } => ops::quantized::dequantize(
        weight,
        scales,
        quant_biases.as_ref(),
        *group_size,
        *bits,
        mode,
        None,
        None,
      ),
    }
  }

  /// The base linear's output **without** the output bias: `x @ Wᵀ` for a dense
  /// base, a fused [`ops::quantized::quantized_matmul`] (`transpose=true`) for a
  /// quantized base. This is the bias-free base-output route the DoRA forward
  /// needs (mlx-lm `tuner/dora.py:113-114` / swift `QDoRALinear` `y = quantizedMM
  /// (...)` then `DoRALinear` `y = matmul(x, weight.T)`, `DoRA+Layers.swift:111,
  /// 172-174` — both bias-free, the bias is re-added after the magnitude renorm).
  ///
  /// Crucially, the quantized branch routes through `quantized_matmul` rather
  /// than dequantizing the full weight, so a QDoRA forward never materializes a
  /// dense `[output_dims, input_dims]` weight just to compute the base output.
  fn base_output_no_bias(&self, x: &Array) -> Result<Array> {
    match self {
      BaseLinear::Dense { weight, .. } => {
        let wt = weight.transpose()?;
        x.matmul(&wt)
      }
      BaseLinear::Quantized {
        weight,
        scales,
        quant_biases,
        group_size,
        bits,
        mode,
        ..
      } => {
        // `transpose=true` matches mlx-lm's QuantizedLinear (the packed weight
        // is laid out for the `output_dims x input_dims` orientation).
        ops::quantized::quantized_matmul(
          x,
          weight,
          scales,
          quant_biases.as_ref(),
          true,
          *group_size,
          *bits,
          mode,
        )
      }
    }
  }

  /// The base linear's output `y = x @ Wᵀ (+ bias)` — [`base_output_no_bias`]
  /// plus the optional output bias. Mirrors mlx-lm `self.linear(x)`
  /// (`tuner/lora.py:96`) / swift `super.callAsFunction(x)`. Does NOT add the
  /// low-rank term — that is [`LoRALinear::forward`]'s job.
  ///
  /// [`base_output_no_bias`]: BaseLinear::base_output_no_bias
  fn base_output(&self, x: &Array) -> Result<Array> {
    let y = self.base_output_no_bias(x)?;
    match self.bias() {
      Some(b) => y.add(b),
      None => Ok(y),
    }
  }

  /// Re-quantize a fused dense weight back into a [`BaseLinear::Quantized`]
  /// with this base's scheme — mlx-lm `nn.QuantizedLinear.from_linear(...)`
  /// in `fuse` (`tuner/lora.py:57-63`). Only meaningful for a quantized base;
  /// a dense base returns the dense fused linear unchanged.
  fn requantize_fused(&self, fused_weight: Array, fused_bias: Option<Array>) -> Result<BaseLinear> {
    match self {
      BaseLinear::Dense { .. } => BaseLinear::dense(fused_weight, fused_bias),
      BaseLinear::Quantized {
        group_size,
        bits,
        mode,
        ..
      } => {
        let (w_q, scales, q_biases) =
          ops::quantized::quantize(&fused_weight, *group_size, *bits, mode, None)?;
        BaseLinear::quantized(
          w_q,
          scales,
          q_biases,
          fused_bias,
          *group_size,
          *bits,
          mode.clone(),
        )
      }
    }
  }

  /// Whether this base is quantized (drives the `fuse(dequantize)` re-quantize
  /// decision).
  fn is_quantized(&self) -> bool {
    matches!(self, BaseLinear::Quantized { .. })
  }
}

// ─────────────────────── scalar-multiply helper ───────────────────────

/// `scale · arr`, broadcasting a scalar. MLX broadcasts a `[1]`-shaped array
/// against any shape, so a single `from_slice(&[scale], (1,))` × `multiply`
/// reproduces python's `scale * z` without an operator overload (mlxrs exposes
/// no `impl Mul`). Lazy — does not evaluate.
///
/// Mirrors mlx-lm's `to_array(v, a.dtype())` scalar-coercion behavior: the
/// `scale` operand is cast to `arr`'s dtype BEFORE the multiply, so a
/// uniform-half input (e.g. an f16 `lora_a` × the Python scalar `self.scale`)
/// stays at the input's narrow precision — mlx's promotion-on-mix would
/// otherwise quietly upcast the result to f32, silently diverging from mlx-lm
/// for uniform-half adapters (mlx-lm `tuner/lora.py:97`, `tuner/dora.py:200`).
/// For mixed-precision (e.g. f16 base + f32 adapter), `arr` is the f32 adapter
/// side so the scalar stays f32 and the surrounding promotion is unchanged.
fn scaled(arr: &Array, scale: f32) -> Result<Array> {
  let s = Array::from_slice::<f32>(&[scale], &(1usize,))?;
  let s = match arr.dtype() {
    Ok(dt) => s.astype(dt)?,
    Err(_) => s,
  };
  arr.multiply(&s)
}

/// `(scale · lora_bᵀ) @ lora_aᵀ` — the dense low-rank delta `Δ` of shape
/// `[output_dims, input_dims]`, the additive update shared by LoRA `fuse`
/// (mlx-lm `tuner/lora.py:52`) and the DoRA `adapted` weight
/// (`tuner/dora.py:120`). `lora_b` is `[r, output_dims]` → `lora_bᵀ` is
/// `[output_dims, r]`; `lora_a` is `[input_dims, r]` → `lora_aᵀ` is
/// `[r, input_dims]`; the product is `[output_dims, input_dims]`, matching the
/// base weight. Lazy.
fn lora_delta(params: &AdapterParams, scale: f32) -> Result<Array> {
  let lb_t = params.lora_b.transpose()?; // [output_dims, r]
  let la_t = params.lora_a.transpose()?; // [r, input_dims]
  let lb_t_scaled = scaled(&lb_t, scale)?;
  lb_t_scaled.matmul(&la_t)
}

/// The shared low-rank forward term `z = (x @ lora_a) @ lora_b`
/// (mlx-lm `tuner/lora.py:97` / `tuner/dora.py:116`), pre-scale. `x @ lora_a`
/// is `[..., r]`; `@ lora_b` is `[..., output_dims]`. Lazy.
fn lora_z(x: &Array, params: &AdapterParams) -> Result<Array> {
  let xa = x.matmul(&params.lora_a)?;
  xa.matmul(&params.lora_b)
}

// ──────────────────────────── LoRALinear ────────────────────────────

/// A LoRA-wrapped linear layer — mlx-lm `tuner/lora.py::LoRALinear` (dense
/// base) / its `QuantizedLinear` branch (quantized base, swift `QLoRALinear`).
///
/// Holds the [`BaseLinear`] (dense or quantized), the [`AdapterParams`]
/// (`lora_a` / `lora_b`), and the scalar `scale`. [`forward`](Self::forward)
/// adds the scaled low-rank update to the base output; [`fuse`](Self::fuse)
/// folds the update into the base weight.
///
/// Construct via [`LoRALinear::new`] (validates the factor shapes against the
/// base). The same type covers QLoRA (LoRA over a quantized base) — the
/// [`BaseLinear::Quantized`] variant routes the base output through a fused
/// quantized matmul (mlx-lm wraps `QuantizedLinear` with the *same* `LoRALinear`
/// class, `tuner/lora.py:22-23`).
#[derive(Debug)]
pub struct LoRALinear {
  base: BaseLinear,
  params: AdapterParams,
  scale: f32,
}

impl LoRALinear {
  /// Wrap `base` with the low-rank `params` and `scale`. Validates the factor
  /// shapes against the base dims: `lora_a` is `[input_dims, r]`, `lora_b` is
  /// `[r, output_dims]` (mlx-lm `tuner/lora.py:88-93`). A magnitude in
  /// `params` is ignored (use [`DoRALinear`] for the weight-decomposed forward).
  pub fn new(base: BaseLinear, params: AdapterParams, scale: f32) -> Result<Self> {
    validate_factor_shapes(&base, &params, LinearValidationContext::LoraLinear)?;
    Ok(Self {
      base,
      params,
      scale,
    })
  }

  /// The low-rank `scale` (mlx-lm `self.scale`).
  pub fn scale(&self) -> f32 {
    self.scale
  }

  /// The wrapped base linear.
  pub fn base(&self) -> &BaseLinear {
    &self.base
  }

  /// Forward pass `out = base(x) + scale · ((x @ lora_a) @ lora_b)` — mlx-lm
  /// `tuner/lora.py::LoRALinear.__call__` (`tuner/lora.py:95-98`) / swift
  /// `LoRALinear.callAsFunction`. The base output `base(x)` is `x @ Wᵀ (+
  /// bias)` (dense matmul or fused quantized matmul); the low-rank term adds
  /// `scale · z`. Lazy — does not evaluate.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let y = self.base.base_output(x)?;
    let z = lora_z(x, &self.params)?;
    let scaled_z = scaled(&z, self.scale)?;
    // mlx-lm casts the low-rank term back to x's dtype before the add
    // (`(self.scale * z).astype(x.dtype)`); replicate so a mixed-precision
    // base (e.g. fp16 weight, fp32 accumulation in the factors) matches.
    let scaled_z = match x.dtype() {
      Ok(dt) => scaled_z.astype(dt)?,
      Err(_) => scaled_z,
    };
    y.add(&scaled_z)
  }

  /// Fold the adapter into the base weight, returning a plain [`BaseLinear`]
  /// whose forward equals this layer's forward (the fusion is a no-op on the
  /// math). Mirrors mlx-lm `tuner/lora.py::LoRALinear.fuse` (`tuner/lora.py:34-65`)
  /// / swift `LoRALinear.fused()`.
  ///
  /// `W_fused = W + (scale · lora_bᵀ) @ lora_aᵀ`. For a quantized base the
  /// weight is dequantized, the delta added, then re-quantized with the same
  /// scheme unless `dequantize` is `true` (mlx-lm's `fuse(dequantize=...)`
  /// argument, `tuner/lora.py:34,57`), in which case the fused base is left
  /// dense.
  pub fn fuse(&self, dequantize: bool) -> Result<BaseLinear> {
    let weight = self.base.dequantized_weight()?;
    let delta = lora_delta(&self.params, self.scale)?;
    // mlx-lm casts the delta to the (dequantized) weight dtype before the add.
    let delta = match weight.dtype() {
      Ok(dt) => delta.astype(dt)?,
      Err(_) => delta,
    };
    let fused_weight = weight.add(&delta)?;
    let fused_bias = match self.base.bias() {
      Some(b) => Some(b.try_clone()?),
      None => None,
    };
    if self.base.is_quantized() && !dequantize {
      self.base.requantize_fused(fused_weight, fused_bias)
    } else {
      BaseLinear::dense(fused_weight, fused_bias)
    }
  }
}

// ──────────────────────────── DoRALinear ────────────────────────────

/// A DoRA-wrapped linear layer — mlx-lm `tuner/dora.py::DoRALinear` (dense
/// base) / its `QuantizedLinear` branch (quantized base, swift `QDoRALinear`).
///
/// DoRA (Weight-Decomposed Low-Rank Adaptation) augments LoRA with a learned
/// per-output-row magnitude `m = ‖W‖₂ (axis 1)`, decoupling the weight's
/// direction (the LoRA-adapted, renormalized weight) from its magnitude.
/// Holds the [`BaseLinear`], the [`AdapterParams`] (`lora_a` / `lora_b` plus
/// the **required** `m`), and `scale`.
///
/// Construct via [`DoRALinear::new`] (validates the factor shapes AND requires
/// a magnitude). The same type covers QDoRA (DoRA over a quantized base) — the
/// base output runs through a fused quantized matmul (swift `QDoRALinear`,
/// `DoRA+Layers.swift:172-174`), and the dequantized weight is materialized
/// **only** for the adapted-weight L2-norm + fuse path (mlx-lm
/// `tuner/dora.py:92-106,120`).
#[derive(Debug)]
pub struct DoRALinear {
  base: BaseLinear,
  params: AdapterParams,
  magnitude: Array,
  scale: f32,
}

impl DoRALinear {
  /// Wrap `base` with the low-rank `params` and `scale` for the DoRA forward.
  /// Validates the factor shapes against the base dims and requires a
  /// magnitude `m` of shape `[output_dims]` in `params` (mlx-lm
  /// `tuner/dora.py:90`). The `m` is taken from `params.magnitude` (loaded
  /// from `adapters.safetensors`).
  pub fn new(base: BaseLinear, params: AdapterParams, scale: f32) -> Result<Self> {
    validate_factor_shapes(&base, &params, LinearValidationContext::DoraLinear)?;
    let magnitude = match &params.magnitude {
      Some(m) => m.try_clone()?,
      None => {
        return Err(Error::MissingField(MissingFieldPayload::new(
          "DoRALinear::new",
          "magnitude `m` (loaded from adapters.safetensors; DoRA requires it)",
        )));
      }
    };
    // `m` is the per-output-row norm: shape [output_dims].
    let output_dims = base_output_dims(&base)?;
    let m_shape = magnitude.shape();
    if m_shape.len() != 1 || m_shape[0] != output_dims {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "DoRALinear::new: magnitude `m` must be [output_dims]",
        vec![output_dims],
        m_shape,
      )));
    }
    Ok(Self {
      base,
      params,
      magnitude,
      scale,
    })
  }

  /// The low-rank `scale`.
  pub fn scale(&self) -> f32 {
    self.scale
  }

  /// The wrapped base linear.
  pub fn base(&self) -> &BaseLinear {
    &self.base
  }

  /// The DoRA magnitude `m` (`[output_dims]`).
  pub fn magnitude(&self) -> &Array {
    &self.magnitude
  }

  /// Forward pass — mlx-lm `tuner/dora.py::DoRALinear.__call__`
  /// (`tuner/dora.py:111-128`) / swift `DoRALinear.callAsFunction` →
  /// `DoRA+Layers.swift::forward`:
  ///
  /// ```text
  /// y       = x @ Wᵀ            (base output, NO bias — quantized_matmul for a
  ///                              quantized base, never a dense dequantize)
  /// z       = (x @ lora_a) @ lora_b
  /// out     = y + (scale · z)
  /// w       = dequantized_weight (ONLY for the norm below)
  /// adapted = w + (scale · lora_bᵀ) @ lora_aᵀ
  /// denom   = ‖adapted‖₂ (axis 1)
  /// out     = (m / denom) · out  (+ bias)
  /// ```
  ///
  /// The renormalization `(m / denom)` is the weight-decomposition step that
  /// distinguishes DoRA from LoRA. For a quantized (QDoRA) base the base output
  /// `y` runs through [`ops::quantized::quantized_matmul`] (matching swift's
  /// `QDoRALinear` `y = quantizedMM(...)`, `DoRA+Layers.swift:172-174`) — the
  /// full weight is dequantized **only** to compute the adapted-weight L2-norm,
  /// never to form the base output, so a forward never materializes a dense
  /// `[output_dims, input_dims]` weight for the matmul. Lazy — does not evaluate.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    // y = base(x) WITHOUT the base bias (the bias is re-added at the very end,
    // mlx-lm `tuner/dora.py:113,126-127`, AFTER the magnitude renorm so it is
    // not scaled). Quantized base ⇒ quantized_matmul, NOT a dense dequantize.
    let y = self.base.base_output_no_bias(x)?;

    let z = lora_z(x, &self.params)?;
    let scaled_z = scaled(&z, self.scale)?;
    let scaled_z = match x.dtype() {
      Ok(dt) => scaled_z.astype(dt)?,
      Err(_) => scaled_z,
    };
    let out = y.add(&scaled_z)?;

    // adapted = w + (scale · lora_bᵀ) @ lora_aᵀ; denom = ‖adapted‖₂ (axis 1).
    // The dense weight is needed HERE (and only here) for the row-wise norm.
    let w = self.base.dequantized_weight()?;
    let delta = lora_delta(&self.params, self.scale)?;
    let delta = match w.dtype() {
      Ok(dt) => delta.astype(dt)?,
      Err(_) => delta,
    };
    let adapted = w.add(&delta)?;
    // norm along axis 1 → [output_dims]; broadcasts against out's last axis.
    let denom = ops::linalg_full::norm(&adapted, 2.0, &[1], false)?;
    let norm_scale = self.magnitude.divide(&denom)?;
    let norm_scale = match x.dtype() {
      Ok(dt) => norm_scale.astype(dt)?,
      Err(_) => norm_scale,
    };
    let mut out = out.multiply(&norm_scale)?;

    // Re-add the base bias AFTER the renorm (mlx-lm `tuner/dora.py:126-127`).
    if let Some(bias) = self.base.bias() {
      out = out.add(bias)?;
    }
    Ok(out)
  }

  /// Fold the DoRA adapter into the base weight — mlx-lm
  /// `tuner/dora.py::DoRALinear.fuse` (`tuner/dora.py:32-56`) / swift
  /// `DoRA+Layers.swift::fuse`:
  ///
  /// ```text
  /// W_adapted = w + (scale · lora_bᵀ) @ lora_aᵀ
  /// W_fused   = (m / ‖W_adapted‖₂)[:, None] · W_adapted
  /// ```
  ///
  /// The fused linear has **no** bias term folded into the weight (DoRA's
  /// `fuse` builds `nn.Linear(..., bias=False)` then re-attaches the original
  /// bias — `tuner/dora.py:38,46-47`). For a quantized base the weight is
  /// dequantized, fused, then re-quantized unless `dequantize` is `true`.
  pub fn fuse(&self, dequantize: bool) -> Result<BaseLinear> {
    let weight = self.base.dequantized_weight()?;
    let delta = lora_delta(&self.params, self.scale)?;
    let delta = match weight.dtype() {
      Ok(dt) => delta.astype(dt)?,
      Err(_) => delta,
    };
    let adapted = weight.add(&delta)?;
    let denom = ops::linalg_full::norm(&adapted, 2.0, &[1], false)?;
    let norm_scale = self.magnitude.divide(&denom)?;
    // norm_scale[:, None] — reshape [output_dims] → [output_dims, 1] so it
    // broadcasts down each weight row (mlx-lm `norm_scale[:, None] * weight`,
    // `tuner/dora.py:44`).
    let norm_scale_col = norm_scale.expand_dims_axes(&[-1])?;
    let fused_weight = norm_scale_col.multiply(&adapted)?;
    let fused_bias = match self.base.bias() {
      Some(b) => Some(b.try_clone()?),
      None => None,
    };
    if self.base.is_quantized() && !dequantize {
      self.base.requantize_fused(fused_weight, fused_bias)
    } else {
      BaseLinear::dense(fused_weight, fused_bias)
    }
  }
}

// ──────────────────────── BaseEmbedding / DoRAEmbedding ────────────────────────

/// The base embedding a [`DoRAEmbedding`] wraps — the *dense* `Embedding` weight
/// `[num_embeddings, dims]` (the per-token lookup table). Mirrors mlx-lm's
/// `nn.Embedding` (the `tuner/dora.py::DoRAEmbedding` base, `tuner/dora.py:179`).
///
/// **Dense-only by design** — mlx-lm's `tuner/dora.py::DoRAEmbedding.from_base`
/// raises `ValueError("DoRAEmbedding does not yet support quantization.")`
/// (`tuner/dora.py:141-142`), so a *quantized* embedding base is not a valid
/// DoRA target upstream and is not modeled here either; the enum carries only a
/// [`BaseEmbedding::Dense`] arm to keep parity with mlx-lm's surface.
///
/// Distinct from [`BaseLinear`] because (a) the forward path is a row gather
/// (`take_axis(weight, ids, 0)`), not a matmul, and (b) the LoRA factor
/// orientation is swapped: `lora_a` is `[num_embeddings, r]` (gathered by token
/// id), `lora_b` is `[r, dims]` — the opposite convention from the
/// [`BaseLinear`]-flavored [`AdapterParams`] (`lora_a` `[input, r]`,
/// `lora_b` `[r, output]`). The embedding-orientation factor shapes are
/// validated by a `validate_embedding_factor_shapes` helper paralleling the
/// linear `validate_factor_shapes` cross-check.
#[derive(Debug)]
pub enum BaseEmbedding {
  /// Dense embedding: `weight` is `[num_embeddings, dims]`.
  Dense {
    /// `[num_embeddings, dims]` dense embedding lookup table.
    weight: Array,
  },
}

impl BaseEmbedding {
  /// Build a dense embedding base from a `[num_embeddings, dims]` weight.
  /// Verifies the weight is rank-2.
  pub fn dense(weight: Array) -> Result<Self> {
    let shape = weight.shape();
    let rank = shape.len();
    if rank != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "BaseEmbedding::dense: weight must be 2-D [num_embeddings, dims]",
        rank as u32,
        shape,
      )));
    }
    Ok(BaseEmbedding::Dense { weight })
  }

  /// The `[num_embeddings, dims]` embedding weight.
  pub fn weight(&self) -> &Array {
    match self {
      BaseEmbedding::Dense { weight } => weight,
    }
  }

  /// `num_embeddings` — the embedding table's leading axis.
  fn num_embeddings(&self) -> Result<usize> {
    let shape = self.weight().shape();
    shape.first().copied().ok_or_else(|| {
      Error::RankMismatch(RankMismatchPayload::new(
        "BaseEmbedding: weight must be rank-2 [num_embeddings, dims] to determine num_embeddings",
        shape.len() as u32,
        shape.clone(),
      ))
    })
  }

  /// `dims` — the embedding table's trailing axis (the per-token vector width).
  fn dims(&self) -> Result<usize> {
    let shape = self.weight().shape();
    shape.get(1).copied().ok_or_else(|| {
      Error::RankMismatch(RankMismatchPayload::new(
        "BaseEmbedding: weight must be rank-2 [num_embeddings, dims] to determine dims",
        shape.len() as u32,
        shape.clone(),
      ))
    })
  }

  /// Per-token lookup: `take_axis(weight, ids, axis=0)`, mirroring
  /// mlx-lm `self.embedding(x)` (`tuner/dora.py:199`) / `nn.Embedding.__call__`.
  /// `ids` carries integer token indices; output dtype matches `weight`.
  fn lookup(&self, ids: &Array) -> Result<Array> {
    self.weight().take_axis(ids, 0)
  }

  /// Embedding-as-linear: `x @ weightᵀ`, the tied-weight LM-head path
  /// (mlx-lm `nn.Embedding.as_linear`, used by `DoRAEmbedding.as_linear`,
  /// `tuner/dora.py:213`). Output shape `[..., num_embeddings]`.
  fn as_linear(&self, x: &Array) -> Result<Array> {
    let wt = self.weight().transpose()?;
    x.matmul(&wt)
  }
}

/// A DoRA-wrapped embedding layer — mlx-lm
/// `tuner/dora.py::DoRAEmbedding` (`tuner/dora.py:131-225`).
///
/// DoRA on an `Embedding` augments the lookup table with a learned per-row
/// magnitude `m = ‖W‖₂ along axis 1` (one magnitude per token row) and the
/// LoRA factors `lora_a` `[num_embeddings, r]` (gathered by token id) /
/// `lora_b` `[r, dims]` — note this is the **opposite** factor orientation from
/// [`DoRALinear`] (where `lora_a` is `[input_dims, r]`), exactly mirroring
/// mlx-lm's `tuner/dora.py:187-192` vs `tuner/dora.py:78-83`.
///
/// Construct via [`DoRAEmbedding::new`]; the layer exposes the two distinct
/// forward modes [`forward`](Self::forward) (token-id lookup) and
/// [`as_linear`](Self::as_linear) (the tied-weight LM-head matmul,
/// `tuner/dora.py:212-224`), plus [`fuse`](Self::fuse) which folds the adapter
/// into a fresh dense [`BaseEmbedding`].
///
/// **Quantized base is intentionally rejected at construction** — mlx-lm's own
/// `DoRAEmbedding.from_base` (`tuner/dora.py:141-142`) raises for a
/// `QuantizedEmbedding` ("DoRAEmbedding does not yet support quantization."),
/// so this surface stays a faithful 1:1 port and offers no
/// `BaseEmbedding::Quantized` arm.
#[derive(Debug)]
pub struct DoRAEmbedding {
  base: BaseEmbedding,
  params: AdapterParams,
  magnitude: Array,
  scale: f32,
}

impl DoRAEmbedding {
  /// Wrap `base` with the low-rank `params` and `scale` for the DoRA
  /// embedding forward. Validates the **embedding-orientation** factor shapes
  /// (`lora_a` `[num_embeddings, r]`, `lora_b` `[r, dims]` — cross-checked
  /// against `base.num_embeddings()` / `base.dims()`) and requires a magnitude
  /// `m` of shape `[num_embeddings]` in `params`, taken from
  /// `params.magnitude` (loaded from `adapters.safetensors`).
  pub fn new(base: BaseEmbedding, params: AdapterParams, scale: f32) -> Result<Self> {
    validate_embedding_factor_shapes(&base, &params, EmbeddingValidationContext::DoraEmbedding)?;
    let magnitude = match &params.magnitude {
      Some(m) => m.try_clone()?,
      None => {
        return Err(Error::MissingField(MissingFieldPayload::new(
          "DoRAEmbedding::new",
          "magnitude `m` (loaded from adapters.safetensors; DoRA requires it)",
        )));
      }
    };
    // `m` is the per-row norm: shape [num_embeddings].
    let num_embeddings = base.num_embeddings()?;
    let m_shape = magnitude.shape();
    if m_shape.len() != 1 || m_shape[0] != num_embeddings {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "DoRAEmbedding::new: magnitude `m` must be [num_embeddings]",
        vec![num_embeddings],
        m_shape,
      )));
    }
    Ok(Self {
      base,
      params,
      magnitude,
      scale,
    })
  }

  /// The low-rank `scale`.
  pub fn scale(&self) -> f32 {
    self.scale
  }

  /// The wrapped base embedding.
  pub fn base(&self) -> &BaseEmbedding {
    &self.base
  }

  /// The DoRA per-row magnitude `m` (`[num_embeddings]`).
  pub fn magnitude(&self) -> &Array {
    &self.magnitude
  }

  /// Token-id lookup forward — mlx-lm
  /// `tuner/dora.py::DoRAEmbedding.__call__` (`tuner/dora.py:198-210`):
  ///
  /// ```text
  /// y       = embedding(x) = weight[x]           // [..., dims]   (per-token lookup)
  /// z       = scale · (lora_a[x] @ lora_b)       // [..., dims]   (per-token low-rank)
  /// out     = y + z                              // (dropout is identity at inference)
  /// adapted = y + (z / scale)·scale = y + z      // PER-TOKEN adapted embedding (`adapted = y + z`)
  /// denom   = ‖adapted‖₂  along axis = -1        // [...]
  /// out     = (m[x] / denom)[..., None] · out
  /// ```
  ///
  /// The renorm divisor is the L2 norm of each token's *adapted* embedding
  /// (per-token, NOT a global row-norm of the table — that is the [`fuse`]
  /// path). `x` carries integer token ids; the output dtype is the mlx
  /// promotion of `y.dtype` and the adapter's dtype (matching mlx-lm
  /// `tuner/dora.py:208` which returns `(m[x]/denom)[..., None] * out`
  /// directly — no final astype). So with an f16/bf16 base and f32 adapter,
  /// `forward` returns f32; callers that want the narrow dtype must cast.
  /// Lazy.
  ///
  /// [`fuse`]: DoRAEmbedding::fuse
  pub fn forward(&self, x: &Array) -> Result<Array> {
    // y = weight[x] — per-token lookup, shape [..., dims].
    let y = self.base.lookup(x)?;
    // lora_a[x] — gather rows of [num_embeddings, r] by x, shape [..., r].
    let la_gathered = self.params.lora_a.take_axis(x, 0)?;
    // z = scale · (lora_a[x] @ lora_b) — UNCAST in the adapter's natural
    // dtype (mlx-lm `tuner/dora.py:200`: `z = self.scale * self.lora_a[x]
    // @ self.lora_b`, never cast before the renorm compute). Critical for
    // mixed-precision: with an f16 base and f32 adapter, upcasting z to
    // y.dtype HERE would drop ~16 bits before the L2 norm, silently shifting
    // the renorm divisor — see the
    // `dora_embedding_forward_mixed_precision_*` regression tests.
    let mut z = la_gathered.matmul(&self.params.lora_b)?;
    z = scaled(&z, self.scale)?;
    // mlx-lm casts the low-rank term to y's dtype ONLY for the `out`
    // accumulator (`out = y + self.dropout(z).astype(y.dtype)`,
    // `tuner/dora.py:201`) — NOT for the adapted-norm compute below.
    let z_for_out = match y.dtype() {
      Ok(dt) => z.astype(dt)?,
      Err(_) => z.try_clone()?,
    };
    let out = y.add(&z_for_out)?;

    // adapted = y + z (UNCAST z — mlx-lm `tuner/dora.py:204`); mlx promotes
    // y/z to the higher-precision dtype on add. denom = ‖adapted‖₂ along
    // axis=-1 (per-token norm, `tuner/dora.py:205` — `axis=-1`
    // distinguishes this from the global `axis=1` row-norm of `fuse`).
    let adapted = y.add(&z)?;
    let denom = ops::linalg_full::norm(&adapted, 2.0, &[-1], false)?;
    // m[x] — gather the per-row magnitude by x, shape [...].
    let m_gathered = self.magnitude.take_axis(x, 0)?;
    // norm_scale = m[x] / denom — UNCAST (mlx-lm `tuner/dora.py:208`'s
    // `(self.m[x] / denom)[..., None]` is never cast; mlx promotes the
    // subsequent multiply against `out` per usual).
    let norm_scale = m_gathered.divide(&denom)?;
    // norm_scale[..., None] — append a trailing size-1 axis so it broadcasts
    // against the [..., dims] `out` (mlx-lm `(m[x] / denom)[..., None] * out`,
    // `tuner/dora.py:208`).
    let norm_scale = norm_scale.expand_dims_axes(&[-1])?;
    // (m[x]/denom)[..., None] * out — mlx promotes to the higher-precision
    // dtype on multiply, exactly mirroring mlx-lm `tuner/dora.py:208`:
    // `out = (self.m[x] / denom)[..., None] * out` (RETURNED DIRECTLY, NO
    // astype). With f16/bf16 base + f32 adapter, mlx promotion leaves the
    // result in f32; a final `astype(y.dtype)` here would silently narrow
    // back to f16/bf16 and drop the promoted precision, diverging from
    // mlx-lm at the forward boundary.
    norm_scale.multiply(&out)
  }

  /// Tied-weight LM-head forward (`embedding.as_linear`) — mlx-lm
  /// `tuner/dora.py::DoRAEmbedding.as_linear` (`tuner/dora.py:212-224`):
  ///
  /// ```text
  /// y       = x @ weightᵀ                        // [..., num_embeddings]
  /// z       = (x @ lora_bᵀ) @ lora_aᵀ            // [..., num_embeddings]
  /// out     = y + (scale · z)
  /// adapted = weight + (scale · lora_a) @ lora_b // [num_embeddings, dims] (GLOBAL)
  /// denom   = ‖adapted‖₂ along axis = 1          // [num_embeddings]
  /// out     = (m / denom) · out
  /// ```
  ///
  /// Distinct from [`forward`](Self::forward) in two ways: (a) the renorm
  /// divisor is the GLOBAL row-norm of the adapted embedding table (mlx-lm
  /// `axis=1` on `adapted = self.embedding.weight + (scale · lora_a) @ lora_b`,
  /// `tuner/dora.py:218-219`), and (b) `m` is broadcast (NOT gathered) — the
  /// last axis is `num_embeddings`, matching `m`'s shape exactly. Lazy.
  pub fn as_linear(&self, x: &Array) -> Result<Array> {
    // y = x @ weightᵀ — shape [..., num_embeddings].
    let y = self.base.as_linear(x)?;
    // z = (x @ lora_bᵀ) @ lora_aᵀ (mlx-lm `tuner/dora.py:214`).
    let lb_t = self.params.lora_b.transpose()?;
    let la_t = self.params.lora_a.transpose()?;
    let xb = x.matmul(&lb_t)?;
    let z = xb.matmul(&la_t)?;
    // scaled_z = scale · z — kept in its natural dtype (typically the
    // adapter's f32) and ONLY cast to x.dtype for the `out` accumulator
    // below, mirroring mlx-lm `tuner/dora.py:215`:
    // `out = y + (self.scale * z).astype(x.dtype)`.
    let scaled_z = scaled(&z, self.scale)?;
    let scaled_z_for_out = match x.dtype() {
      Ok(dt) => scaled_z.astype(dt)?,
      Err(_) => scaled_z.try_clone()?,
    };
    let out = y.add(&scaled_z_for_out)?;

    // adapted = weight + (scale · lora_a) @ lora_b — GLOBAL [num_embeddings,
    // dims]. delta is UNCAST (mlx-lm `tuner/dora.py:218` — `self.embedding
    // .weight + (self.scale * self.lora_a) @ self.lora_b` has NO astype on
    // the delta; mlx promotes the add to the higher-precision dtype). With
    // f16 weight + f32 adapter, downcasting delta to weight.dtype HERE
    // would drop ~16 bits before the row-norm and silently shift the
    // adapted-row magnitudes — the analogous mixed-precision bug to
    // `forward`'s.
    let scaled_la = scaled(&self.params.lora_a, self.scale)?;
    let delta = scaled_la.matmul(&self.params.lora_b)?;
    let w = self.base.weight();
    let adapted = w.add(&delta)?;
    let denom = ops::linalg_full::norm(&adapted, 2.0, &[1], false)?;
    // norm_scale = m / denom — UNCAST (mlx-lm `tuner/dora.py:222`:
    // `out = (self.m / denom) * out` has no astype on `m / denom`; mlx
    // promotes the multiply against `out` per usual). Matches `forward`'s
    // pattern: only cast where mlx-lm casts, NOT at every step.
    let norm_scale = self.magnitude.divide(&denom)?;
    out.multiply(&norm_scale)
  }

  /// Fold the DoRA embedding adapter into the base weight — mlx-lm
  /// `tuner/dora.py::DoRAEmbedding.fuse` (`tuner/dora.py:153-166`):
  ///
  /// ```text
  /// W_adapted = weight + (scale · lora_a) @ lora_b           // [num_embeddings, dims]
  /// W_fused   = (m / ‖W_adapted‖₂)[:, None] · W_adapted
  /// ```
  ///
  /// Returns a plain dense [`BaseEmbedding`] whose lookup (and `as_linear`)
  /// equals this layer's `as_linear` within fp tolerance. The
  /// per-token `forward` does NOT equal the fused-base lookup because
  /// `forward`'s renorm is per-token (`axis=-1` on `y + z`), distinct from
  /// `fuse`'s global row-norm — exactly mirroring mlx-lm's split.
  pub fn fuse(&self) -> Result<BaseEmbedding> {
    let scaled_la = scaled(&self.params.lora_a, self.scale)?;
    let delta = scaled_la.matmul(&self.params.lora_b)?;
    let w = self.base.weight();
    let delta = match w.dtype() {
      Ok(dt) => delta.astype(dt)?,
      Err(_) => delta,
    };
    let adapted = w.add(&delta)?;
    let denom = ops::linalg_full::norm(&adapted, 2.0, &[1], false)?;
    let norm_scale = self.magnitude.divide(&denom)?;
    // norm_scale[:, None] — append a trailing size-1 axis so it broadcasts
    // down each weight row (mlx-lm `norm_scale[:, None] * weight`,
    // `tuner/dora.py:164`).
    let norm_scale_col = norm_scale.expand_dims_axes(&[-1])?;
    let fused_weight = norm_scale_col.multiply(&adapted)?;
    BaseEmbedding::dense(fused_weight)
  }
}

/// Validate `lora_a` / `lora_b` against an embedding base — the
/// **embedding-orientation** factor shapes (`lora_a` `[num_embeddings, r]`,
/// `lora_b` `[r, dims]`). Cross-checks the shared rank axis (`a[1] == b[0]`),
/// `lora_a`'s leading axis against `num_embeddings`, and `lora_b`'s last axis
/// against `dims` — so a wrong-shape factor is a recoverable
/// [`Error::RankMismatch`] / [`Error::LengthMismatch`] at construct/load time,
/// not an opaque mlx-c failure on the first lookup. Mirrors
/// [`validate_factor_shapes`] for the linear side.
fn validate_embedding_factor_shapes(
  base: &BaseEmbedding,
  params: &AdapterParams,
  who: EmbeddingValidationContext,
) -> Result<()> {
  let a_shape = params.lora_a.shape();
  let b_shape = params.lora_b.shape();
  let a_rank = a_shape.len();
  let b_rank = b_shape.len();
  // Snapshot the rank-axis values BEFORE the rank-2 early-returns can move
  // the shape vecs; both later cross-checks need them.
  let a_rank_axis = a_shape.get(1).copied().unwrap_or_default();
  let b_rank_axis = b_shape.first().copied().unwrap_or_default();
  let a_leading_axis = a_shape.first().copied().unwrap_or_default();
  let b_last_axis = b_shape.get(1).copied().unwrap_or_default();
  if a_rank != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      who.lora_a_rank2(),
      a_rank as u32,
      a_shape,
    )));
  }
  if b_rank != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      who.lora_b_rank2(),
      b_rank as u32,
      b_shape,
    )));
  }
  if a_rank_axis != b_rank_axis {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      who.shared_rank(),
      b_rank_axis,
      a_rank_axis,
    )));
  }
  let num_embeddings = base.num_embeddings()?;
  if a_leading_axis != num_embeddings {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      who.a_leading_vs_num_embeddings(),
      num_embeddings,
      a_leading_axis,
    )));
  }
  let dims = base.dims()?;
  if b_last_axis != dims {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      who.b_last_vs_dims(),
      dims,
      b_last_axis,
    )));
  }
  Ok(())
}

/// Closed-set caller label for [`validate_embedding_factor_shapes`]: maps a
/// construction site onto the static `context()` strings the typed
/// [`Error::RankMismatch`] / [`Error::LengthMismatch`] payloads carry. Each
/// method returns `&'static str` (the typed payloads' `context()` accessor
/// type), avoiding a `format!`-built runtime context string.
#[derive(Debug, Clone, Copy)]
enum EmbeddingValidationContext {
  DoraEmbedding,
}

impl EmbeddingValidationContext {
  /// Context label for the rank-2 lora_a check (`a.shape().len() == 2`).
  const fn lora_a_rank2(self) -> &'static str {
    match self {
      Self::DoraEmbedding => "DoRAEmbedding: lora_a must be 2-D [num_embeddings, r]",
    }
  }
  /// Context label for the rank-2 lora_b check (`b.shape().len() == 2`).
  const fn lora_b_rank2(self) -> &'static str {
    match self {
      Self::DoraEmbedding => "DoRAEmbedding: lora_b must be 2-D [r, dims]",
    }
  }
  /// Context label for the shared-rank cross-check (`a[1] == b[0]`).
  const fn shared_rank(self) -> &'static str {
    match self {
      Self::DoraEmbedding => {
        "DoRAEmbedding: lora_a last axis vs lora_b leading axis (shared rank `r`)"
      }
    }
  }
  /// Context label for the lora_a leading axis vs base num_embeddings cross-check.
  const fn a_leading_vs_num_embeddings(self) -> &'static str {
    match self {
      Self::DoraEmbedding => "DoRAEmbedding: lora_a leading axis vs base num_embeddings",
    }
  }
  /// Context label for the lora_b last axis vs base dims cross-check.
  const fn b_last_vs_dims(self) -> &'static str {
    match self {
      Self::DoraEmbedding => "DoRAEmbedding: lora_b last axis vs base dims",
    }
  }
}

// ──────────────────────────── LoraLayer ────────────────────────────

/// A wrapped LoRA/DoRA layer — the unified runtime surface a per-usecase
/// architecture dispatches an adapted weight-path through. Mirrors swift's
/// `LoRALayer` protocol (`LoRA+Layers.swift` / `DoRA+Layers.swift` both
/// conform), which the [`LoraLayers`] map stores polymorphically.
///
/// One of [`LoraLayer::Lora`], [`LoraLayer::Dora`], or [`LoraLayer::DoraEmbedding`],
/// each carrying the concrete wrapped layer ([`LoRALinear`] / [`DoRALinear`] /
/// [`DoRAEmbedding`]); [`forward`](Self::forward) dispatches to the variant.
/// The two linear variants additionally expose [`fuse`](Self::fuse) (folding
/// into a [`BaseLinear`]); the embedding variant exposes
/// [`fuse_embedding`](Self::fuse_embedding).
///
/// The [`DoraEmbedding`](Self::DoraEmbedding) variant is the
/// `mlx_lm/tuner/dora.py::DoRAEmbedding` port; the parallel LoRA-embedding
/// variant (`mlx_lm/tuner/lora.py::LoRAEmbedding`) is the deferred follow-up
/// described in the [module docs](self).
#[derive(Debug)]
pub enum LoraLayer {
  /// A LoRA-wrapped linear.
  Lora(LoRALinear),
  /// A DoRA-wrapped linear.
  Dora(DoRALinear),
  /// A DoRA-wrapped embedding.
  DoraEmbedding(DoRAEmbedding),
}

impl LoraLayer {
  /// Forward through the wrapped layer (LoRA-linear / DoRA-linear / DoRA-embedding).
  /// For [`DoraEmbedding`](Self::DoraEmbedding) `x` carries integer token ids
  /// (see [`DoRAEmbedding::forward`]); for the linear variants `x` is the
  /// activation matrix. Lazy.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    match self {
      LoraLayer::Lora(l) => l.forward(x),
      LoraLayer::Dora(d) => d.forward(x),
      LoraLayer::DoraEmbedding(d) => d.forward(x),
    }
  }

  /// Fuse a *linear* LoRA/DoRA layer into a plain [`BaseLinear`] (see
  /// [`LoRALinear::fuse`] / [`DoRALinear::fuse`]). Returns
  /// `Err(Error::Backend)` when the variant is [`DoraEmbedding`](Self::DoraEmbedding)
  /// — embedding fuse returns a [`BaseEmbedding`], so use
  /// [`fuse_embedding`](Self::fuse_embedding) for that.
  pub fn fuse(&self, dequantize: bool) -> Result<BaseLinear> {
    match self {
      LoraLayer::Lora(l) => l.fuse(dequantize),
      LoraLayer::Dora(d) => d.fuse(dequantize),
      LoraLayer::DoraEmbedding(_) => {
        Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "LoraLayer::fuse: variant",
          "is a DoRA embedding layer; call `fuse_embedding` to obtain a `BaseEmbedding`",
        )))
      }
    }
  }

  /// Fuse a [`DoraEmbedding`](Self::DoraEmbedding) into a plain
  /// [`BaseEmbedding`] (see [`DoRAEmbedding::fuse`]). Returns
  /// `Err(Error::InvariantViolation)` for the linear variants — use
  /// [`fuse`](Self::fuse) for those.
  pub fn fuse_embedding(&self) -> Result<BaseEmbedding> {
    match self {
      LoraLayer::DoraEmbedding(d) => d.fuse(),
      LoraLayer::Lora(_) | LoraLayer::Dora(_) => {
        Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "LoraLayer::fuse_embedding: variant",
          "is a linear LoRA/DoRA layer; call `fuse` to obtain a `BaseLinear`",
        )))
      }
    }
  }

  /// The wrapped base *linear* — `Some(&BaseLinear)` for the linear variants,
  /// `None` for [`DoraEmbedding`](Self::DoraEmbedding) (whose base is a
  /// [`BaseEmbedding`], not a [`BaseLinear`] — use [`base_embedding`](Self::base_embedding)).
  pub fn base(&self) -> Option<&BaseLinear> {
    match self {
      LoraLayer::Lora(l) => Some(l.base()),
      LoraLayer::Dora(d) => Some(d.base()),
      LoraLayer::DoraEmbedding(_) => None,
    }
  }

  /// The wrapped base *embedding* — `Some(&BaseEmbedding)` for
  /// [`DoraEmbedding`](Self::DoraEmbedding), `None` for the linear variants.
  pub fn base_embedding(&self) -> Option<&BaseEmbedding> {
    match self {
      LoraLayer::DoraEmbedding(d) => Some(d.base()),
      LoraLayer::Lora(_) | LoraLayer::Dora(_) => None,
    }
  }
}

/// The map a [`linear_to_lora_layers`] / [`load_adapters`] run produces: the
/// base-weight **path** (e.g. `"model.layers.27.self_attn.q_proj"`) → its
/// wrapped [`LoraLayer`].
///
/// This is the weight-map analogue of the in-place `nn.Module` replacement
/// mlx-lm / swift perform — a per-usecase architecture that already routes a
/// path to its forward call dispatches through the wrapped layer for any path
/// present in this map (and leaves un-adapted paths on their base forward).
pub type LoraLayers = HashMap<String, LoraLayer>;

// ──────────────────── shape validation helpers ────────────────────

/// The base linear's `output_dims` (the leading weight dim for a dense base;
/// for a quantized base the *packed* weight's leading dim still equals
/// `output_dims` — MLX packs along the last axis only).
fn base_output_dims(base: &BaseLinear) -> Result<usize> {
  let shape = match base {
    BaseLinear::Dense { weight, .. } => weight.shape(),
    BaseLinear::Quantized { weight, .. } => weight.shape(),
  };
  shape.first().copied().ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      "base linear: weight must be rank-2 [output_dims, input_dims] to determine output_dims",
      shape.len() as u32,
      shape.clone(),
    ))
  })
}

/// The base linear's `input_dims` — the contraction dimension `lora_a`'s leading
/// axis must equal. For a **dense** base it is the weight's trailing axis
/// (`weight` is `[output_dims, input_dims]`). For a **quantized** base the
/// *packed* weight's trailing axis is `input_dims * bits / 32` (MLX packs
/// `32 / bits` weights per `uint32` along the last axis), so the logical input
/// width is `packed_last_axis * 32 / bits` — exactly mlx-lm's `from_base`
/// recovery `input_dims = input_dims * 32 // bits` (`tuner/lora.py:23`,
/// `tuner/dora.py:21`). `bits` is validated `> 0` by [`BaseLinear::quantized`].
fn base_input_dims(base: &BaseLinear) -> Result<usize> {
  match base {
    BaseLinear::Dense { weight, .. } => {
      let shape = weight.shape();
      let rank = shape.len() as u32;
      shape.get(1).copied().ok_or_else(|| {
        Error::RankMismatch(RankMismatchPayload::new(
          "dense base weight must be 2-D [output_dims, input_dims]",
          rank,
          shape.clone(),
        ))
      })
    }
    BaseLinear::Quantized { weight, bits, .. } => {
      let shape = weight.shape();
      let rank = shape.len() as u32;
      let packed = shape.get(1).copied().ok_or_else(|| {
        Error::RankMismatch(RankMismatchPayload::new(
          "quantized base weight must be 2-D [output_dims, input_dims*bits/32]",
          rank,
          shape.clone(),
        ))
      })?;
      // `bits > 0` is guaranteed by `BaseLinear::quantized`; recover the logical
      // input width `packed * 32 / bits` (e.g. 4-bit packs 8 weights / u32).
      Ok(packed * 32 / (*bits as usize))
    }
  }
}

/// Validate `lora_a` / `lora_b` against the base dims. `lora_a` is
/// `[input_dims, r]`, so its leading axis must equal the base `input_dims`
/// (recovered from the packed width for a quantized base — see
/// [`base_input_dims`]) and its last axis (`r`) must match `lora_b`'s leading
/// axis (`r`); `lora_b` is `[r, output_dims]`, so its last axis must equal the
/// base `output_dims`. Cross-checking the `input_dims` axis here means a wrong
/// `lora_a` width is a recoverable [`Error::RankMismatch`] / [`Error::LengthMismatch`]
/// at validate/load time (not an opaque mlx-c matmul failure on the first forward).
fn validate_factor_shapes(
  base: &BaseLinear,
  params: &AdapterParams,
  who: LinearValidationContext,
) -> Result<()> {
  let a_shape = params.lora_a.shape();
  let b_shape = params.lora_b.shape();
  let a_rank = a_shape.len();
  let b_rank = b_shape.len();
  // Snapshot the rank-axis values BEFORE the rank-2 early-returns can move
  // the shape vecs; the linear-side cross-checks all need them.
  let a_rank_axis = a_shape.get(1).copied().unwrap_or_default();
  let b_rank_axis = b_shape.first().copied().unwrap_or_default();
  let a_leading_axis = a_shape.first().copied().unwrap_or_default();
  let b_last_axis = b_shape.get(1).copied().unwrap_or_default();
  if a_rank != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      who.lora_a_rank2(),
      a_rank as u32,
      a_shape,
    )));
  }
  if b_rank != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      who.lora_b_rank2(),
      b_rank as u32,
      b_shape,
    )));
  }
  // r consistency: lora_a's last axis == lora_b's leading axis.
  if a_rank_axis != b_rank_axis {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      who.shared_rank(),
      b_rank_axis,
      a_rank_axis,
    )));
  }
  // input_dims consistency: lora_a's leading axis == base input_dims.
  let input_dims = base_input_dims(base)?;
  if a_leading_axis != input_dims {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      who.a_leading_vs_input_dims(),
      input_dims,
      a_leading_axis,
    )));
  }
  // output_dims consistency: lora_b's last axis == base output_dims.
  let output_dims = base_output_dims(base)?;
  if b_last_axis != output_dims {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      who.b_last_vs_output_dims(),
      output_dims,
      b_last_axis,
    )));
  }
  Ok(())
}

/// Closed-set caller label for [`validate_factor_shapes`] — mirror of
/// [`EmbeddingValidationContext`] for the LINEAR side. Each method returns
/// `&'static str` so the typed payloads' `context()` accessor stays `'static`
/// without a `format!`-built runtime context string.
#[derive(Debug, Clone, Copy)]
enum LinearValidationContext {
  LoraLinear,
  DoraLinear,
}

impl LinearValidationContext {
  /// Context label for the rank-2 lora_a check.
  const fn lora_a_rank2(self) -> &'static str {
    match self {
      Self::LoraLinear => "LoRALinear: lora_a must be 2-D [input_dims, r]",
      Self::DoraLinear => "DoRALinear: lora_a must be 2-D [input_dims, r]",
    }
  }
  /// Context label for the rank-2 lora_b check.
  const fn lora_b_rank2(self) -> &'static str {
    match self {
      Self::LoraLinear => "LoRALinear: lora_b must be 2-D [r, output_dims]",
      Self::DoraLinear => "DoRALinear: lora_b must be 2-D [r, output_dims]",
    }
  }
  /// Context label for the shared-rank cross-check.
  const fn shared_rank(self) -> &'static str {
    match self {
      Self::LoraLinear => "LoRALinear: lora_a last axis vs lora_b leading axis (shared rank `r`)",
      Self::DoraLinear => "DoRALinear: lora_a last axis vs lora_b leading axis (shared rank `r`)",
    }
  }
  /// Context label for the lora_a leading axis vs base input_dims cross-check.
  const fn a_leading_vs_input_dims(self) -> &'static str {
    match self {
      Self::LoraLinear => "LoRALinear: lora_a leading axis vs base input_dims",
      Self::DoraLinear => "DoRALinear: lora_a leading axis vs base input_dims",
    }
  }
  /// Context label for the lora_b last axis vs base output_dims cross-check.
  const fn b_last_vs_output_dims(self) -> &'static str {
    match self {
      Self::LoraLinear => "LoRALinear: lora_b last axis vs base output_dims",
      Self::DoraLinear => "DoRALinear: lora_b last axis vs base output_dims",
    }
  }
  /// Context label for the adapter_config.json `rank` vs lora_a actual rank check.
  const fn config_rank_vs_lora_a_rank(self) -> &'static str {
    match self {
      Self::LoraLinear => "LoRALinear: adapter_config.json rank vs lora_a actual rank axis",
      Self::DoraLinear => "DoRALinear: adapter_config.json rank vs lora_a actual rank axis",
    }
  }
  /// Context label for the adapter_config.json `rank` vs lora_b actual rank check.
  const fn config_rank_vs_lora_b_rank(self) -> &'static str {
    match self {
      Self::LoraLinear => "LoRALinear: adapter_config.json rank vs lora_b actual rank axis",
      Self::DoraLinear => "DoRALinear: adapter_config.json rank vs lora_b actual rank axis",
    }
  }
}

/// Check the adapter factor tensors' rank axis against the rank declared in
/// `adapter_config.json` (`config.rank()`) — the *config-vs-tensor* rank
/// cross-check.
///
/// [`validate_factor_shapes`] only verifies `lora_a` and `lora_b` agree with
/// **each other** on the shared rank axis; it cannot see the config. But the
/// layer SCALE is `alpha / config.rank()` when an `alpha` (`lora_alpha`) is
/// present, so a config whose `rank` has drifted from the tensors' rank — a
/// stale `adapter_config.json` whose declared rank no longer matches the
/// shipped factors — would otherwise build rank-`R` tensors while scaling by
/// `alpha / config.rank()` (the wrong divisor): silently wrong strength on
/// every adapted projection.
///
/// Requiring `lora_a`'s rank axis (`[input_dims, r]`, last axis) and
/// `lora_b`'s rank axis (`[r, output_dims]`, leading axis) to both equal
/// `config_rank` makes that drift a loud, recoverable [`Error::LengthMismatch`]
/// at load time instead. Indexing is defensive (a non-2-D factor reads as a
/// `0` rank axis), so this is safe to call independently of
/// [`validate_factor_shapes`].
fn validate_config_rank(
  params: &AdapterParams,
  config_rank: usize,
  who: LinearValidationContext,
) -> Result<()> {
  let a_shape = params.lora_a.shape();
  let b_shape = params.lora_b.shape();
  // A well-formed `lora_a` is `[input_dims, r]` and `lora_b` is
  // `[r, output_dims]`; a non-2-D factor reads as a `0` rank axis here and
  // fails the equality below (it also fails `validate_factor_shapes`).
  let a_rank = a_shape.get(1).copied().unwrap_or_default();
  let b_rank = b_shape.first().copied().unwrap_or_default();
  if a_rank != config_rank {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      who.config_rank_vs_lora_a_rank(),
      config_rank,
      a_rank,
    )));
  }
  if b_rank != config_rank {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      who.config_rank_vs_lora_b_rank(),
      config_rank,
      b_rank,
    )));
  }
  Ok(())
}

// ─────────────────────── linear_to_lora_layers ───────────────────────

/// Apply LoRA/DoRA wrapping to the targeted linear layers of a [`Weights`]
/// map — mlx-lm `tuner/utils.py::linear_to_lora_layers` (`tuner/utils.py:38-110`)
/// **and** HuggingFace PEFT's `LoraModel._create_and_replace` /
/// `check_target_module_exists`, adapted to the weight-map model (see the
/// [module docs](self)).
///
/// For each base-weight path the predicate selects, this builds a [`LoraLayer`]
/// (LoRA or DoRA per `config`) over the path's [`BaseLinear`] — a dense base
/// from `<path>.weight` (+ optional `<path>.bias`), or a quantized base from
/// the `<path>.weight` / `<path>.scales` / `<path>.biases` triple when `quant`
/// resolves a [`Quantization`] for that path (the QLoRA case). The returned
/// [`LoraLayers`] map carries the wrapped layers; un-targeted paths are not
/// touched.
///
/// # Layer selection
///
/// The rule depends on [`config.selection`](LoraConfig::selection):
///
/// - **[`AdapterSelection::MlxLm`]** — mlx-lm's two-part predicate: the
///   trailing-`num_layers`-block window (a non-positive `num_layers` selects
///   ALL blocks — the Python `-0` slice quirk) **plus** the
///   `lora_parameters.keys` suffix allowlist (`None` ⇒ every rank-2 linear in
///   the window — mlx-lm's auto-discovery).
/// - **[`AdapterSelection::Peft`]** — PEFT's selection
///   (`check_target_module_exists`): `exclude_modules` removes a match first,
///   then `target_modules` (exact-or-`.endswith` list, or `re.fullmatch`
///   regex) selects, then `layers_to_transform` + `layers_pattern` restrict to
///   explicit block indices. **No** trailing-`num_layers` window — PEFT adapts
///   EVERY matching block (this is why a PEFT config must NOT inherit mlx-lm's
///   `num_layers=16`). A PEFT config with no `target_modules` falls back to the
///   rank-2 auto-discovery (mlxrs has no module tree for PEFT's "auto-detect
///   linear layers" — see the module docs).
///
/// # Per-module scale
///
/// The low-rank scale is resolved **per module** ([`LoraConfig::scale_for`]):
/// the single config-wide scale for mlx-lm-native, or PEFT's
/// `lora_alpha / r` — `/ sqrt(r)` under `use_rslora` — with `rank_pattern` /
/// `alpha_pattern` overrides for a PEFT config.
///
/// `adapter_params` supplies the per-path [`AdapterParams`] (loaded from
/// `adapters.safetensors`). `num_blocks` is the model's decoder-block count
/// (needed for the mlx-lm trailing window; ignored on the PEFT path).
///
/// # Completeness postcondition
///
/// After wrapping, the result is checked (`check_adapter_completeness`) so a
/// path-prefix mismatch / missing tensor group / empty safetensors /
/// `adapter_config.json` drift cannot silently return a partially- or
/// un-adapted model. It is a recoverable [`Error::Backend`] when (a) an
/// explicit target selection (mlx-lm `keys` / PEFT `target_modules`) is missing
/// factors for a selected target, (b) an `adapter_params` factor group matches
/// no base layer, or (c) nothing was adapted at all.
///
/// A selected path whose factor shapes don't match the base (or a DoRA path
/// with no magnitude) is a recoverable [`Error::RankMismatch`] /
/// [`Error::LengthMismatch`] / [`Error::ShapePairMismatch`] /
/// [`Error::Backend`]. A selected path whose factor tensors' rank axis
/// disagrees with the module's resolved rank ([`LoraConfig::rank_for`] — the
/// `rank_pattern`-overridden rank for a PEFT module) is a recoverable
/// [`Error::LengthMismatch`] (`validate_config_rank`) — caught before the
/// scale is applied, so a rank drift cannot silently scale by the wrong
/// divisor.
pub fn linear_to_lora_layers(
  weights: &Weights,
  config: &LoraConfig,
  adapter_params: &HashMap<String, AdapterParams>,
  quant: Option<&PerLayerQuantization>,
  num_blocks: i32,
) -> Result<LoraLayers> {
  let mut out: LoraLayers = HashMap::new();
  let is_dora = config.is_dora();
  let fan_in_fan_out = config.fan_in_fan_out();

  // The mlx-lm trailing window's first adapted block index — `tuner/utils.py:103`
  // `model.layers[-max(num_layers, 0):]`, with the Python `-0` quirk
  // (`num_layers <= 0` ⇒ `layers[0:]` == ALL blocks). Unused on the PEFT path.
  let first_adapted = match &config.selection {
    AdapterSelection::MlxLm { num_layers } if *num_layers > 0 => (num_blocks - num_layers).max(0),
    _ => 0,
  };

  // Completeness tracking (the postcondition below): every adapter factor
  // group MUST be applied to a base layer, and an explicit target selection
  // MUST find its factors — otherwise a path-prefix mismatch / missing tensor
  // group / config drift would silently yield a partially- or un-adapted model.
  let mut consumed: HashSet<&str> = HashSet::with_capacity(adapter_params.len());
  // Targets the predicate selected but for which no factors were supplied —
  // only an *error* when the selection is explicit (with auto-discovery, an
  // unmatched linear is expected: the adapter trains only a subset).
  let mut selected_without_factors: Vec<&str> = Vec::new();
  // Whether the target selection is *explicit* (an allowlist the adapter is
  // expected to fully cover) vs auto-discovery. mlx-lm: `keys` is set. PEFT: an
  // explicit `target_modules` list / regex is set. The `all-linear` sentinel is
  // NOT explicit — it is a "discover all linears" shorthand, and mlxrs's
  // head-exclusion is approximate, so a discovered linear the adapter did not
  // train must be skipped (like auto-discovery), not flagged missing.
  let explicit_selection = match &config.selection {
    AdapterSelection::MlxLm { .. } => !config.lora_parameters.keys.is_empty(),
    AdapterSelection::Peft(peft) => {
      matches!(&peft.target_modules, Some(m) if !matches!(m, ModuleMatcher::AllLinear))
    }
  };

  for (key, weight) in weights {
    let Some(path) = key.strip_suffix(".weight") else {
      continue;
    };

    if !module_is_selected(path, weight, config, first_adapted) {
      continue;
    }

    // `path` is now a SELECTED target (predicate-matched). Build a layer only
    // when we actually have factors for it; record a missing-factor target so
    // the postcondition can reject an incomplete explicit selection.
    let Some(params) = adapter_params.get(path) else {
      selected_without_factors.push(path);
      continue;
    };
    consumed.insert(path);

    // The module's resolved rank / scale (PEFT `rank_pattern` / `alpha_pattern`
    // overrides applied; mlx-lm: the config-wide values). Cross-check the
    // factor tensors' rank axis against the resolved rank BEFORE building the
    // layer: a config/tensor rank drift must fail loudly, not silently scale by
    // the wrong divisor.
    let module_rank = config.rank_for(path);
    let scale = config.scale_for(path);
    if let Some(rank) = usize::try_from(module_rank).ok().filter(|&r| r > 0) {
      let who = if is_dora {
        LinearValidationContext::DoraLinear
      } else {
        LinearValidationContext::LoraLinear
      };
      validate_config_rank(params, rank, who)?;
    }

    let base = build_base_linear(weights, path, weight, quant, fan_in_fan_out)?;
    let layer = if is_dora {
      LoraLayer::Dora(DoRALinear::new(base, params.try_clone()?, scale)?)
    } else {
      LoraLayer::Lora(LoRALinear::new(base, params.try_clone()?, scale)?)
    };
    out.insert(path.to_string(), layer);
  }

  check_adapter_completeness(
    &out,
    adapter_params,
    &consumed,
    &selected_without_factors,
    explicit_selection,
  )?;
  Ok(out)
}

/// Whether the base-weight path `path` (weight `weight`) is selected for
/// adaptation under `config` — the per-shape predicate `linear_to_lora_layers`
/// applies to every `<path>.weight` in the weight map. `first_adapted` is the
/// mlx-lm trailing window's first block index (unused on the PEFT path).
fn module_is_selected(path: &str, weight: &Array, config: &LoraConfig, first_adapted: i32) -> bool {
  match &config.selection {
    AdapterSelection::MlxLm { .. } => {
      // mlx-lm: `keys` suffix allowlist, or rank-2 auto-discovery, then the
      // trailing-`num_layers`-block window.
      if !config.lora_parameters.keys.is_empty() {
        // explicit allowlist: path must match a key suffix
        if !config
          .lora_parameters
          .keys
          .iter()
          .any(|k| path_matches_key(path, k))
        {
          return false;
        }
      } else {
        // auto-discovery: only rank-2 weights are "linears".
        if weight.shape().len() != 2 {
          return false;
        }
      }
      // A path inside a decoder block is adapted only when its block index is
      // in the trailing window; a non-block path (no `layers.N`) has no block
      // index to gate, so `keys` alone governs it.
      match parse_block_index(path) {
        Some(block) => block >= first_adapted,
        None => true,
      }
    }
    AdapterSelection::Peft(peft) => peft_module_is_selected(path, weight, peft),
  }
}

/// PEFT's `check_target_module_exists` + `layers_to_transform` selection for
/// one module path (`peft/tuners/tuners_utils.py`):
///
/// 1. `exclude_modules` — if it matches, the module is excluded (PEFT's early
///    `_ExcludedModule` return, checked *first*).
/// 2. `target_modules` — must match (list exact-or-`.endswith`, or
///    `re.fullmatch` regex). `None` ⇒ rank-2 auto-discovery (mlxrs has no
///    module tree for PEFT's `_maybe_include_all_linear_layers`).
///    [`ModuleMatcher::AllLinear`] (the `"all-linear"` sentinel) ⇒ the same
///    rank-2 eligibility as auto-discovery, **minus the output head**
///    ([`is_output_head_path`]) — PEFT's `_maybe_include_all_linear_layers`
///    expands `all-linear` to every linear/`Conv1D` except
///    `model.get_output_embeddings()`. Lacking a module tree, mlxrs
///    approximates the head by name (`lm_head`); a non-`lm_head` output head is
///    not excluded (documented approximation). `all-linear` is treated as
///    auto-discovery for the completeness postcondition (a discovered linear
///    that the adapter did not train is silently skipped, not an error — see
///    [`linear_to_lora_layers`]).
/// 3. `layers_to_transform` — when set, the module's decoder-block index (from
///    `layers_pattern`, or PEFT's default index regex) must be in the list.
fn peft_module_is_selected(path: &str, weight: &Array, peft: &PeftSelection) -> bool {
  // (1) exclude first.
  if let Some(exclude) = &peft.exclude_modules
    && exclude.matches(path)
  {
    return false;
  }

  // (2) target match — or rank-2 auto-discovery when `target_modules` is None.
  match &peft.target_modules {
    // `all-linear`: rank-2 eligibility (the "is a linear" half) AND not the
    // output head (`matches` is the head-exclusion half). Both are required.
    Some(ModuleMatcher::AllLinear) => {
      if weight.shape().len() != 2 || !ModuleMatcher::AllLinear.matches(path) {
        return false;
      }
    }
    Some(target) => {
      if !target.matches(path) {
        return false;
      }
    }
    None => {
      if weight.shape().len() != 2 {
        return false;
      }
    }
  }

  // (3) layers_to_transform: restrict to explicit block indices. PEFT only
  // applies this when `layers_to_transform` is non-empty AND the target
  // matched; a module with no extractable block index is then NOT selected
  // (PEFT sets `target_module_found = False` when `layer_index is None`).
  if let Some(layers) = &peft.layers_to_transform
    && !layers.is_empty()
  {
    match peft_layer_index(path, &peft.layers_pattern) {
      Some(idx) => return layers.contains(&idx),
      None => return false,
    }
  }

  true
}

/// Extract a module's decoder-block index the way PEFT's
/// `check_target_module_exists` does for `layers_to_transform`
/// (`peft/tuners/tuners_utils.py`):
///
/// - `layers_pattern` empty ⇒ PEFT's default `re.match(r".*\.[^.]*\.(\d+)\.",
///   key)` — the digits between two dots after some prefix.
/// - `layers_pattern` non-empty ⇒ for each `pattern`, PEFT's
///   `re.match(rf".*\.{pattern}\.(\d+)\.", key)` — the digits right after the
///   named `ModuleList` attribute.
///
/// `None` when no pattern extracts an index (PEFT then de-selects the module).
fn peft_layer_index(path: &str, layers_pattern: &[String]) -> Option<i32> {
  if layers_pattern.is_empty() {
    // PEFT default: `.*\.[^.]*\.(\d+)\.` — anchored at start by `re.match`.
    let re = Regex::new(r"^.*\.[^.]*\.(\d+)\.").ok()?;
    let caps = re.captures(path)?;
    return caps.get(1)?.as_str().parse::<i32>().ok();
  }
  for pattern in layers_pattern {
    // PEFT: `.*\.{pattern}\.(\d+)\.` — `{pattern}` is a literal attribute name
    // (e.g. "layers" / "h"), escaped so a dotted name cannot inject regex.
    let escaped = regex::escape(pattern);
    let Ok(re) = Regex::new(&format!(r"^.*\.{escaped}\.(\d+)\.")) else {
      continue;
    };
    if let Some(caps) = re.captures(path)
      && let Some(m) = caps.get(1)
      && let Ok(idx) = m.as_str().parse::<i32>()
    {
      return Some(idx);
    }
  }
  None
}

/// The adapter-completeness postcondition for [`linear_to_lora_layers`]:
/// reject a result that would leave inference silently-wrong.
///
/// A base path matching the selection predicate but carrying no
/// [`AdapterParams`] must not be skipped silently — a path-prefix mismatch,
/// missing tensor group, empty `adapters.safetensors`, or `adapter_config.json`
/// drift would otherwise return `Ok` with a partially- or un-adapted model.
/// This catches all three failure modes:
///
/// - **(a) explicitly-selected target with no factors** — when the selection
///   is explicit (`explicit_selection`: an mlx-lm `keys` list or a PEFT
///   `target_modules`), every selected path is a target the adapter is
///   expected to provide; a missing factor group is config drift. (With
///   auto-discovery an unmatched linear is *expected* — the adapter trains only
///   a subset — so this is not checked there.)
/// - **(b) unused adapter factor group** — every path present in
///   `adapter_params` (i.e. every `<path>.lora_a`/`lora_b` group in the
///   safetensors) MUST have matched a base layer; one that matched nothing is a
///   path-prefix mismatch. This is the analogue of swift's
///   `model.update(parameters:, verify: .noUnusedKeys)` (`LoRAContainer.swift:152`).
/// - **(c) empty result** — no layer adapted at all ⇒ the adapter did nothing.
///
/// Each violation is a recoverable [`Error::Backend`] naming the offending
/// paths.
fn check_adapter_completeness(
  applied: &LoraLayers,
  adapter_params: &HashMap<String, AdapterParams>,
  consumed: &HashSet<&str>,
  selected_without_factors: &[&str],
  explicit_selection: bool,
) -> Result<()> {
  // (a) explicit selection (mlx-lm `keys` / PEFT `target_modules`) missing
  // factors. The typed [`Error::MissingKey`] carries the FIRST (sorted)
  // missing target so a programmatic caller can branch on it; remaining
  // missing keys would each surface here on re-run after that one is fixed
  // (the [`AdapterParams`] map is iteration-order-stable post-sort).
  if explicit_selection && !selected_without_factors.is_empty() {
    let mut missing: Vec<&str> = selected_without_factors.to_vec();
    missing.sort_unstable();
    return Err(Error::MissingKey(MissingKeyPayload::new(
      "load_adapters: explicitly-selected adapter target (adapter_config.json target \
        selection does not match adapters.safetensors contents)",
      missing[0].to_string(),
    )));
  }

  // (b) adapter factor groups that matched no base layer (unused). Wraps a
  // typed `InvariantViolation` ("must match a base layer") in a
  // [`LayerKeyed`] keyed by the FIRST (sorted) unused path — a programmatic
  // caller can branch on `LayerKeyed.layer()` for the offending path and on
  // the inner variant for the violated rule.
  let mut unused: Vec<&str> = adapter_params
    .keys()
    .map(String::as_str)
    .filter(|p| !consumed.contains(p))
    .collect();
  if !unused.is_empty() {
    unused.sort_unstable();
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      unused[0].to_string(),
      Error::InvariantViolation(InvariantViolationPayload::new(
        "load_adapters: adapter factor group",
        "must match a base layer (the adapters.safetensors paths do not align with \
          the base model weights — path-prefix mismatch or config drift)",
      )),
    )));
  }

  // (c) nothing adapted at all.
  if applied.is_empty() {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "load_adapters: adapted-layer count",
      "must be >= 1 (the adapter_config.json target selection — mlx-lm `keys`/`num_layers` \
        or PEFT `target_modules`/`layers_to_transform` — matched nothing in the base model, \
        or adapters.safetensors carried no factors)",
    )));
  }

  Ok(())
}

/// Build the [`BaseLinear`] for `path` from the weight map: a quantized base
/// (from the `<path>.weight` / `.scales` / `.biases` triple) when `quant`
/// resolves a [`Quantization`] for `path` AND a `<path>.scales` sibling
/// exists; otherwise a dense base (from `<path>.weight` + optional
/// `<path>.bias`).
///
/// `fan_in_fan_out` (PEFT `LoraConfig.fan_in_fan_out`): when `true` the stored
/// dense `<path>.weight` is laid out `[in_features, out_features]` (a
/// `transformers.Conv1D`-style transposed weight) rather than the standard
/// `[out_features, in_features]`. The weight is transposed back here so every
/// downstream consumer ([`BaseLinear`]'s forward, the factor-shape
/// cross-check, `fuse`) sees the standard `[out, in]` orientation — exactly
/// PEFT's `transpose(weight, fan_in_fan_out)` at the matmul boundary. A
/// `fan_in_fan_out` *quantized* base is rejected: transposing a packed
/// quantized weight would corrupt the bit-packing, and the combination does
/// not occur in practice (PEFT's `fan_in_fan_out` targets `Conv1D`, which is
/// never the `(weight, scales, biases)` quantized triple).
fn build_base_linear(
  weights: &Weights,
  path: &str,
  weight: &Array,
  quant: Option<&PerLayerQuantization>,
  fan_in_fan_out: bool,
) -> Result<BaseLinear> {
  let scales_key = format!("{path}.scales");
  let biases_key = format!("{path}.biases");
  let bias_key = format!("{path}.bias");

  // The QLoRA case: a resolvable Quantization for this path AND a `.scales`
  // sibling (the load-bearing quantized-layout signal — mlx-lm's
  // `f"{p}.scales" in weights` check, `utils.py:349-355`).
  let q: Option<Quantization> = quant.and_then(|c| c.quantization_for(path));
  if let (Some(q), Some(scales)) = (q, weights.get(&scales_key)) {
    if fan_in_fan_out {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        path.to_string(),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "load_adapters: quantized base + adapter_config.json `fan_in_fan_out`",
          "must not be combined (a packed quantized weight cannot be transposed without \
            corrupting the bit-packing; `fan_in_fan_out` applies only to a dense Conv1D-style \
            base)",
        )),
      )));
    }
    let quant_biases = weights.get(&biases_key).map(Array::try_clone).transpose()?;
    let bias = weights.get(&bias_key).map(Array::try_clone).transpose()?;
    return BaseLinear::quantized(
      weight.try_clone()?,
      scales.try_clone()?,
      quant_biases,
      bias,
      q.group_size,
      q.bits,
      q.mode.as_str().to_string(),
    );
  }

  // Dense base. With `fan_in_fan_out` the stored weight is `[in, out]` — undo
  // the transpose so `BaseLinear::dense` receives the standard `[out, in]`.
  let bias = weights.get(&bias_key).map(Array::try_clone).transpose()?;
  let dense_weight = if fan_in_fan_out {
    weight.transpose()?
  } else {
    weight.try_clone()?
  };
  BaseLinear::dense(dense_weight, bias)
}

/// Whether `path` should match the adapter key `key`: mlx-lm matches a module
/// **name** (the path tail). A `key` matches when `path` equals it or ends
/// with `".{key}"` (so `"self_attn.q_proj"` matches
/// `"model.layers.27.self_attn.q_proj"` but not `"…xself_attn.q_proj"`).
fn path_matches_key(path: &str, key: &str) -> bool {
  path == key || path.ends_with(&format!(".{key}"))
}

/// Parse the decoder-block index from a `…layers.N.…` (or trailing
/// `…layers.N`) path segment, mirroring mlx-lm's per-block iteration over
/// `model.layers`. `None` when there is no `layers.<int>` segment (a
/// non-block path — e.g. `model.embed_tokens` / `lm_head`).
fn parse_block_index(path: &str) -> Option<i32> {
  // Find a "layers." segment and parse the following integer up to the next
  // '.' or end of string.
  let marker = "layers.";
  let idx = path.find(marker)? + marker.len();
  let rest = &path[idx..];
  let end = rest.find('.').unwrap_or(rest.len());
  rest[..end].parse::<i32>().ok()
}

// ───────────────────────────── load_adapters ─────────────────────────────

/// Load a pre-trained adapter from a **local** directory and apply it to a
/// base model's [`Weights`] map — mlx-lm `tuner/utils.py::load_adapters`
/// (`tuner/utils.py:113-138`) + swift `LoRAContainer.from(directory:)` /
/// `load(into:)`, restricted to the local-path, no-network surface.
///
/// Reads `<dir>/adapter_config.json` (bounded, untrusted-dir-safe — same
/// discipline as [`crate::lm::load::load_config`]) and the adapter weights
/// file (via [`crate::io::load_safetensors`]), splits it into per-path
/// [`AdapterParams`], then runs [`linear_to_lora_layers`] over `base_weights`
/// to build the [`LoraLayers`] map.
///
/// # mlx-lm-native and HuggingFace PEFT adapters
///
/// Both adapter formats are supported (the format is detected from the
/// `adapter_config.json` shape — see [`LoraConfig`]):
///
/// - **mlx-lm-native** — weights in `adapters.safetensors`, keyed
///   `<path>.lora_a` / `.lora_b` / `.m`; selection by the
///   trailing-`num_layers` window + `lora_parameters.keys`.
/// - **PEFT** — weights in `adapter_model.safetensors`, keyed
///   `base_model.model.<path>.lora_A.weight` / `.lora_B.weight` /
///   `.lora_magnitude_vector` (the PEFT key scheme is translated to the mlxrs
///   scheme, transposing the factor tensors — PEFT stores them transposed);
///   selection by PEFT's `target_modules` / `exclude_modules` /
///   `layers_to_transform`; per-module rsLoRA + `rank_pattern` / `alpha_pattern`
///   scaling.
///
/// `base_weights` is the loaded base-model weight map ([`crate::lm::load::load_weights`]);
/// `quant` is the base model's [`PerLayerQuantization`] (from
/// [`crate::lm::quant::parse_quantization`] on the base `config.json`) so a
/// quantized base routes through the QLoRA path — pass `None` for a dense base.
/// `num_blocks` is the base model's decoder-block count.
///
/// # Errors (recoverable)
///
/// - Missing adapter dir / `adapter_config.json` / adapter weights file,
///   oversized / non-regular / non-UTF-8 config → [`Error::Backend`].
/// - An adapter weights file that is not a regular file (FIFO / device /
///   directory) or exceeds [`MAX_ADAPTER_SAFETENSORS_BYTES`] → [`Error::Backend`]
///   (the file is stat-checked before mlx-c mmaps it).
/// - `fine_tune_type: "full"` (a full-weight fine-tune, not an adapter — see
///   [`FineTuneType::Full`]) → [`Error::Backend`] (unsupported here).
///   An **unknown** `fine_tune_type` string is a serde parse error →
///   [`Error::Backend`] from [`LoraConfig::from_json`].
/// - A target path with a magnitude-less DoRA factor, or factor shapes that
///   don't match the base → [`Error::RankMismatch`] / [`Error::LengthMismatch`] /
///   [`Error::ShapePairMismatch`] / [`Error::Backend`].
/// - The completeness postcondition of [`linear_to_lora_layers`]: an explicit
///   target selection (mlx-lm `keys` / PEFT `target_modules`) missing factors,
///   an unused adapter factor group, or an empty result → [`Error::Backend`].
pub fn load_adapters(
  base_weights: &Weights,
  dir: &Path,
  quant: Option<&PerLayerQuantization>,
  num_blocks: i32,
) -> Result<LoraLayers> {
  // Single parse of `adapter_config.json` (bounded, untrusted-dir-safe) ⇒
  // forward to the with-config variant. Callers that already hold a parsed
  // [`LoraConfig`] (e.g. [`crate::lm::fuse::fuse`], which needs the PEFT
  // `fan_in_fan_out` flag BEFORE walking the fused layers) should call
  // [`load_adapters_with_config`] directly so the same on-disk file isn't
  // parsed twice — two reads = two snapshots = a TOCTOU window where a
  // hostile or just-flipped `adapter_config.json` could send the load side
  // and the save side down divergent paths (a quantized + `fan_in_fan_out`
  // flag flip would panic the debug-only `insert_base_linear` assertion;
  // a square-target `fan_in_fan_out` flag flip would silently transpose
  // the saved weight against the load-side decision).
  let config_text = read_bounded_adapter_config(dir)?;
  let config = LoraConfig::from_json(&config_text)?;
  load_adapters_with_config(base_weights, dir, &config, quant, num_blocks)
}

/// Like [`load_adapters`] but takes a pre-parsed [`LoraConfig`] instead of
/// reading + parsing `<dir>/adapter_config.json` internally. The on-disk
/// adapter weights file is still located + loaded from `dir`, but the
/// config is **not** re-read — eliminating a TOCTOU window for callers that
/// also need the typed config for their own decisions (e.g.
/// [`crate::lm::fuse::fuse`] needs the PEFT `fan_in_fan_out` flag BEFORE
/// walking the fused layers).
///
/// All other arguments + errors + postconditions match [`load_adapters`]
/// — the body of the two is structurally identical from "validate config"
/// onward; [`load_adapters`] is a thin wrapper that parses once then forwards.
///
/// # Single-snapshot contract
///
/// The shared parsed config is the **single source of truth** for both the
/// load side (transpose-on-read for PEFT `fan_in_fan_out`, build_base_linear's
/// quantized-rejection) and the save side (re-transpose to persisted
/// orientation, quant-arm debug-assert). A separate parse on the save side
/// could observe a different snapshot (process raced the file replacement,
/// or a hostile re-write) — that would either silently corrupt the saved
/// weight orientation (square-target case: a load-canonical fused weight
/// would be written without the matching transpose-back-to-persisted) or
/// fire the debug-only `insert_base_linear` quantized-arm assertion in
/// debug builds (the quantized + `fan_in_fan_out` combination is rejected
/// at load time; a flag flip between parses would let the load side see
/// `fan_in_fan_out: false` and proceed, while the save side sees
/// `fan_in_fan_out: true` and trips the invariant). Callers must hold ONE
/// parsed config across both reads.
pub fn load_adapters_with_config(
  base_weights: &Weights,
  dir: &Path,
  config: &LoraConfig,
  quant: Option<&PerLayerQuantization>,
  num_blocks: i32,
) -> Result<LoraLayers> {
  // mlx-lm skips linear_to_lora_layers for "full" and loads a dense delta;
  // mlxrs has no module tree to merge a full fine-tune into here, so reject it
  // as unsupported (recoverable) — the per-usecase architecture merges a full
  // fine-tune at the weight-map level instead.
  if config.fine_tune_type == FineTuneType::Full {
    return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
      "load_adapters: fine_tune_type (supported adapter types only — `full` is a full-weight \
        fine-tune; merge it at the weight-map level via lm::load::load_weights)",
      "full",
      &["lora", "dora"],
    )));
  }

  // Reject a non-positive rank early (a degenerate config that would build
  // empty factors).
  if config.rank() <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "load_adapters: adapter rank",
      "must be > 0",
      format!("{}", config.rank()),
    )));
  }

  // 2) Locate the adapter weights file. mlx-lm names it `adapters.safetensors`;
  // HuggingFace PEFT names it `adapter_model.safetensors`. Pick by the config
  // shape, but fall back to whichever file is actually on disk (some exporters
  // pair a PEFT config with the mlx-lm filename, or vice versa).
  let st_path = locate_adapter_safetensors(dir, config)?;
  // Stat the file FIRST (regular-file + size-budget) so an untrusted adapter
  // dir cannot point us at a FIFO/device (hang/opaque error) or an oversized
  // blob (OOM) — the safetensors path is otherwise handed straight to mlx-c,
  // which would mmap whatever it is given.
  check_adapter_safetensors(&st_path)?;
  let adapter_arrays = crate::io::load_safetensors(&st_path)
    .map_err(|e| Error::LayerKeyed(LayerKeyedPayload::new(st_path.display().to_string(), e)))?;
  // PEFT keys (`base_model.model.<path>.lora_A.weight`, …) → the mlxrs
  // `<path>.lora_a` / `.lora_b` / `.m` scheme, transposing the factor tensors
  // (PEFT stores them in the opposite orientation — see `translate_peft_keys`).
  // mlx-lm-native keys are already in the mlxrs scheme — used verbatim.
  let adapter_arrays = match config.selection {
    AdapterSelection::Peft(_) => translate_peft_keys(adapter_arrays)?,
    AdapterSelection::MlxLm { .. } => adapter_arrays,
  };
  let adapter_params = split_adapter_params(adapter_arrays, config.is_dora())?;

  // 3) Build + apply the LoRA/DoRA layers over the base weight map.
  linear_to_lora_layers(base_weights, config, &adapter_params, quant, num_blocks)
}

/// Read + parse an adapter directory's `adapter_config.json` and return the
/// typed [`LoraConfig`] — the same bounded read + serde parse
/// [`load_adapters`] uses internally, exposed so callers that need only the
/// config metadata (e.g. [`crate::lm::fuse::fuse`] needs the PEFT
/// `fan_in_fan_out` flag to carry the persisted-orientation contract through
/// fusion) don't have to load the full weight map + build the layer table
/// first.
///
/// Same untrusted-dir safety discipline as [`load_adapters`]'s internal read:
/// non-blocking open with `O_CLOEXEC`, regular-file check before any read,
/// body capped at the crate-internal `MAX_CONFIG_BYTES` (1 MiB) via
/// `Read::take`. A missing directory / file / non-regular target / oversized
/// body / malformed JSON is a recoverable [`Error::Backend`] whose message
/// names the offending path (twin of [`load_adapters`]'s error shape).
pub fn read_adapter_config(dir: &Path) -> Result<LoraConfig> {
  let config_text = read_bounded_adapter_config(dir)?;
  LoraConfig::from_json(&config_text)
}

/// mlx-lm's adapter weights filename (`tuner/utils.py:137`
/// `adapter_path / "adapters.safetensors"`).
const MLX_LM_ADAPTER_FILE: &str = "adapters.safetensors";

/// HuggingFace PEFT's adapter weights filename — `peft`'s
/// `SAFETENSORS_WEIGHTS_NAME`, written by `PeftModel.save_pretrained`.
const PEFT_ADAPTER_FILE: &str = "adapter_model.safetensors";

/// Locate the adapter weights file in `dir`. The *preferred* name follows the
/// config shape — mlx-lm-native ⇒ [`MLX_LM_ADAPTER_FILE`], PEFT ⇒
/// [`PEFT_ADAPTER_FILE`] — but if the preferred file is absent and the other
/// name is present, that is used (an exporter pairing one project's config
/// with the other's filename is not a hard error). Only when BOTH candidates
/// are confirmed absent (a real `NotFound`) does this synthesize a
/// `FileIo(NotFound)` naming the preferred path. Any *other* stat failure —
/// `PermissionDenied`, symlink loop, etc. — surfaces as a typed `Error::FileIo`
/// carrying the actual `io::Error` rather than being silently rebranded as
/// "missing" by a bare `Path::exists()` check.
fn locate_adapter_safetensors(dir: &Path, config: &LoraConfig) -> Result<std::path::PathBuf> {
  let (preferred, fallback) = match config.selection {
    AdapterSelection::Peft(_) => (PEFT_ADAPTER_FILE, MLX_LM_ADAPTER_FILE),
    AdapterSelection::MlxLm { .. } => (MLX_LM_ADAPTER_FILE, PEFT_ADAPTER_FILE),
  };
  let preferred_path = dir.join(preferred);
  if adapter_candidate_present(&preferred_path)? {
    return Ok(preferred_path);
  }
  let fallback_path = dir.join(fallback);
  if adapter_candidate_present(&fallback_path)? {
    return Ok(fallback_path);
  }
  // Both candidates returned a real `NotFound`. Synthesize the canonical
  // missing-file error naming the preferred candidate (the one the config
  // shape asked for).
  Err(Error::FileIo(FileIoPayload::new(
    "load_adapters: no adapter weights file (expected adapters.safetensors or \
      adapter_model.safetensors)",
    FileOp::Open,
    preferred_path,
    std::io::Error::from(std::io::ErrorKind::NotFound),
  )))
}

/// Exhaustive classification of the outcomes of stating an adapter-candidate
/// path. Each variant maps to one of the four contract obligations of
/// [`adapter_candidate_present`] — the typed lift makes the mapping a
/// `match` on the enum rather than nested if-let / `is_file` checks, so any
/// future change that omits one of the four outcomes is a non-exhaustive-match
/// compile error rather than a silent fallthrough.
enum CandidateProbe {
  /// `metadata()` resolved to a genuine `NotFound` (the candidate file is
  /// absent at the resolved target, OR a parent component is missing).
  /// Caller should fall through to the next candidate.
  Absent,
  /// `metadata()` resolved (via symlink-following) to a *regular* file. This
  /// is the only outcome that should pin the candidate as the adapter weights.
  Present,
  /// `metadata()` resolved to a non-regular path (directory, FIFO, socket,
  /// block device, …). NOT a loadable adapter weights file — fail fast rather
  /// than silently falling through, because silently falling through here
  /// masks a misconfiguration where the user pointed mlxrs at a directory
  /// with `adapters.safetensors/` as a subdir or similar.
  NonRegular,
  /// Any other I/O failure during `metadata()` — `PermissionDenied`,
  /// `FilesystemLoop`/`ELOOP`, `Uncategorized`, etc. Must be propagated as a
  /// typed [`Error::FileIo`], not coerced to `Ok(false)`.
  IoError(std::io::Error),
}

/// Stat `path` and classify the outcome into the exhaustive
/// [`CandidateProbe`] set. `metadata()` (NOT `symlink_metadata()`) is used so
/// that the *target* of any symlink is what's classified — broken symlinks
/// surface as [`CandidateProbe::Absent`] from the target, not `Present` on the
/// link object itself.
fn probe_candidate(path: &Path) -> CandidateProbe {
  match std::fs::metadata(path) {
    Ok(m) if m.is_file() => CandidateProbe::Present,
    Ok(_) => CandidateProbe::NonRegular,
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => CandidateProbe::Absent,
    Err(e) => CandidateProbe::IoError(e),
  }
}

/// Does the adapter candidate file exist *as a regular file*?
///
/// # Contract
///
/// The four outcomes are exhaustively classified via [`CandidateProbe`] and
/// mapped to a 4-way `match`, killing the entire defect class where a
/// non-`NotFound` failure (or a non-regular resolved path) silently collapses
/// to `Ok(false)` and lets the fallback candidate quietly suppress the real
/// signal:
///
/// | Outcome of `metadata(path)`                | Return                       | Caller behavior                |
/// |--------------------------------------------|------------------------------|--------------------------------|
/// | Regular file                               | `Ok(true)`                   | Use this candidate.            |
/// | `NotFound` (or parent missing)             | `Ok(false)`                  | Try the next candidate.        |
/// | Non-regular (dir / FIFO / socket / …)      | `Err(FileIo(InvalidInput))`  | **Fail fast — no fallback.**   |
/// | Any other I/O error (`PermissionDenied`,   | `Err(FileIo(<real kind>))`   | Propagate the typed io::Error. |
/// | `FilesystemLoop`/`ELOOP`, `Uncategorized`) |                              |                                |
///
/// # Why the typed lift
///
/// Three simpler shapes are *rejected*:
/// - **`Path::exists()`** collapses NotFound, PermissionDenied, and ELOOP into
///   a single `false`, which would rebrand a permission failure as "missing
///   file" — diagnostic loss this dedicated probe avoids.
/// - **`symlink_metadata()`** does NOT follow the symlink — a *broken*
///   preferred symlink (target missing) would `Ok(_)` on the link object
///   itself, so `Ok(true)` would short-circuit fallback and the later `open`
///   would fail instead of trying the valid fallback file. `metadata()`
///   follows the link, so a broken link surfaces as `NotFound` from the
///   *target* and the fallback path is correctly tried.
/// - **`Ok(m) => Ok(m.is_file())`** silently downgrades a directory / FIFO /
///   socket at the preferred path to `Ok(false)` → the fallback candidate is
///   tried even though the user clearly *wanted* `adapters.safetensors` at the
///   preferred location. A directory at the preferred slot is a
///   misconfiguration that must surface immediately, not be papered over by
///   reading a different file with potentially incompatible weights.
fn adapter_candidate_present(path: &Path) -> Result<bool> {
  match probe_candidate(path) {
    CandidateProbe::Present => Ok(true),
    CandidateProbe::Absent => Ok(false),
    CandidateProbe::NonRegular => Err(Error::FileIo(FileIoPayload::new(
      "load_adapters: adapter weights candidate exists but is not a regular file",
      FileOp::Stat,
      path.to_path_buf(),
      std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "adapter candidate path exists but is not a regular file",
      ),
    ))),
    CandidateProbe::IoError(e) => Err(Error::FileIo(FileIoPayload::new(
      "load_adapters: cannot stat adapter weights candidate",
      FileOp::Stat,
      path.to_path_buf(),
      e,
    ))),
  }
}

/// PEFT adapter-weight key prefix — every tensor in a `peft`
/// `adapter_model.safetensors` is named `base_model.model.<module_path>.<...>`
/// (`peft/utils/save_and_load.py`: the non-prompt-learning `prefix =
/// "base_model.model."`).
const PEFT_KEY_PREFIX: &str = "base_model.model.";

/// Translate a PEFT-keyed `adapter_model.safetensors` array map into the mlxrs
/// `<path>.lora_a` / `<path>.lora_b` / `<path>.m` scheme that
/// [`split_adapter_params`] consumes.
///
/// # PEFT's on-disk key scheme
///
/// `peft` (`utils/save_and_load.py`) saves a LoRA tensor as
/// `base_model.model.<module_path>.lora_A.weight` (and `lora_B.weight`,
/// `lora_magnitude_vector` for DoRA). The adapter-name segment PEFT carries
/// in-memory (`lora_A.<adapter>.weight`) is **stripped before saving** — on
/// disk there is no adapter name (`peft` `remove_adapter_name`: "the adapter
/// name is just an arbitrary name that can be changed when loading"). So this
/// strips the `base_model.model.` prefix and maps the trailing component:
///
/// | PEFT on-disk suffix          | mlxrs suffix |
/// |------------------------------|--------------|
/// | `.lora_A.weight`             | `.lora_a`    |
/// | `.lora_B.weight`             | `.lora_b`    |
/// | `.lora_magnitude_vector` / `.lora_magnitude_vector.weight` | `.m` |
///
/// # Factor-tensor transpose
///
/// PEFT's `lora_A` is an `nn.Linear(in_features, r)`, so `lora_A.weight` is
/// stored `[r, in_features]`; `lora_B` is `nn.Linear(r, out_features)`, so
/// `lora_B.weight` is `[out_features, r]`. mlxrs's [`AdapterParams`] uses the
/// **opposite** orientation — `lora_a` is `[input_dims, r]`, `lora_b` is
/// `[r, output_dims]` (mlx-lm's `tuner/lora.py` layout). So each factor tensor
/// is transposed during translation. The DoRA `lora_magnitude_vector` is a
/// length-`out_features` vector in both — copied without transpose.
///
/// # Rejected — PEFT-prefixed non-LoRA tensors (bias / `modules_to_save`)
///
/// A PEFT adapter trained with `bias != "none"` ships `.bias` tensors
/// (`utils/save_and_load.py` keeps `"bias" in k`, saved as
/// `base_model.model.<path>.bias`); `modules_to_save` ships *full* module
/// weights (the `modules_to_save.<adapter>.` prefix is stripped on save, so on
/// disk they are `base_model.model.<module>.weight`). Both carry the PEFT
/// prefix but a suffix that is **not** a recognized low-rank factor, and both
/// affect inference — so a key under the PEFT prefix whose suffix is none of
/// `.lora_A.weight` / `.lora_B.weight` / `.lora_magnitude_vector[.weight]` is a
/// recoverable [`Error::Backend`] naming the offending key, **not** a silent
/// drop. (The config-level `bias` / `modules_to_save` rejection in
/// [`LoraConfig`]'s `Deserialize` is the first line of defense; this guards the
/// weights file directly, catching a sidecar tensor even when the config does
/// not declare it.) A key *without* the PEFT prefix is still skipped — it is
/// simply not a PEFT adapter tensor.
fn translate_peft_keys(arrays: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
  let mut out: HashMap<String, Array> = HashMap::with_capacity(arrays.len());
  for (key, arr) in arrays {
    // Strip the `base_model.model.` prefix. A key without it is not a PEFT
    // adapter tensor (e.g. a stray base weight) — skip it.
    let Some(rest) = key.strip_prefix(PEFT_KEY_PREFIX) else {
      continue;
    };
    if let Some(path) = rest.strip_suffix(".lora_A.weight") {
      // [r, in] → [in, r].
      out.insert(format!("{path}.lora_a"), arr.transpose()?);
    } else if let Some(path) = rest.strip_suffix(".lora_B.weight") {
      // [out, r] → [r, out].
      out.insert(format!("{path}.lora_b"), arr.transpose()?);
    } else if let Some(path) = rest
      .strip_suffix(".lora_magnitude_vector.weight")
      .or_else(|| rest.strip_suffix(".lora_magnitude_vector"))
    {
      // DoRA magnitude — [out_features] in both schemes, no transpose.
      out.insert(format!("{path}.m"), arr);
    } else if rest.contains(".lora_embedding_A") || rest.contains(".lora_embedding_B") {
      // Embedding LoRA — PEFT's `LoraLayer.adapter_layer_names`
      // (`lora/layer.py:105`) registers `lora_embedding_A` / `lora_embedding_B`
      // ParameterDicts for `nn.Embedding` targets; `utils/save_and_load.py`
      // saves them (key matches `"lora_" in k`) as
      // `base_model.model.<path>.lora_embedding_A` (optionally `.weight`).
      // These ARE legitimate low-rank factors, NOT a `bias` / `modules_to_save`
      // tensor — so they must NOT fall through to the generic message below
      // (which would misclassify them). Embedding-LoRA *application* is not
      // implemented (deferred), so reject with a precise, distinct error rather
      // than load it wrong. (`.contains` rather than `.strip_suffix` because the
      // suffix may be bare or carry a trailing `.weight`.)
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        key,
        Error::InvariantViolation(InvariantViolationPayload::new(
          "load_adapters: PEFT adapter tensor",
          "is an embedding LoRA factor (`lora_embedding_A` / `lora_embedding_B`); embedding \
            LoRA is not supported by this loader (only linear-layer `lora_A` / `lora_B` \
            low-rank factors are applied)",
        )),
      )));
    } else {
      // A PEFT-prefixed tensor that is NOT a low-rank factor — a `.bias`
      // (PEFT `bias != "none"`) or a `modules_to_save` full-module weight. Both
      // affect inference; dropping them would silently corrupt the adapter, so
      // reject loudly naming the key (mlxrs's LoRALinear has no adapted-bias /
      // saved-module slot — same stance as the config-level rejection).
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        key,
        Error::InvariantViolation(InvariantViolationPayload::new(
          "load_adapters: PEFT adapter tensor",
          "must be one of `.lora_A.weight` / `.lora_B.weight` / `.lora_magnitude_vector` \
            (this is a PEFT `bias` or `modules_to_save` tensor, which affects inference and \
            this low-rank loader has no slot for — dropping it would silently corrupt the \
            adapter)",
        )),
      )));
    }
  }
  Ok(out)
}

/// Split the flat `adapters.safetensors` array map into per-path
/// [`AdapterParams`]. mlx-lm's `LoRALinear` registers its factors at
/// `<module>.lora_a` / `<module>.lora_b` (and DoRA's `<module>.m`), so the
/// safetensors keys are `<path>.lora_a` / `<path>.lora_b` / `<path>.m`. Groups
/// by stripping those suffixes; a `.lora_a` without a matching `.lora_b` (or
/// vice versa) is a recoverable [`Error::Backend`]. When `expect_dora`, a
/// group missing its `.m` is an error; when not, a stray `.m` is ignored.
fn split_adapter_params(
  arrays: HashMap<String, Array>,
  expect_dora: bool,
) -> Result<HashMap<String, AdapterParams>> {
  // Collect the three slots per path.
  let mut a_map: HashMap<String, Array> = HashMap::new();
  let mut b_map: HashMap<String, Array> = HashMap::new();
  let mut m_map: HashMap<String, Array> = HashMap::new();

  for (key, arr) in arrays {
    if let Some(path) = key.strip_suffix(".lora_a") {
      a_map.insert(path.to_string(), arr);
    } else if let Some(path) = key.strip_suffix(".lora_b") {
      b_map.insert(path.to_string(), arr);
    } else if let Some(path) = key.strip_suffix(".m") {
      m_map.insert(path.to_string(), arr);
    }
    // Any other key (e.g. a saved base weight in a "full" checkpoint) is
    // ignored — this path only handles low-rank adapters.
  }

  let mut out: HashMap<String, AdapterParams> = HashMap::with_capacity(a_map.len());
  for (path, lora_a) in a_map {
    let lora_b = b_map.remove(&path).ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "load_adapters: adapter has `lora_a` but no matching `lora_b`",
        format!("{path}.lora_b"),
      ))
    })?;
    let magnitude = m_map.remove(&path);
    if expect_dora && magnitude.is_none() {
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "load_adapters: DoRA adapter is missing its magnitude `m`",
        format!("{path}.m"),
      )));
    }
    out.insert(
      path,
      AdapterParams {
        lora_a,
        lora_b,
        magnitude,
      },
    );
  }

  // Any `lora_b` left without a matching `lora_a` is an error.
  if let Some((path, _)) = b_map.into_iter().next() {
    return Err(Error::MissingKey(MissingKeyPayload::new(
      "load_adapters: adapter has `lora_b` but no matching `lora_a`",
      format!("{path}.lora_a"),
    )));
  }

  Ok(out)
}

/// Read `<dir>/adapter_config.json` with the same bounded, untrusted-dir-safe
/// discipline as [`crate::lm::load::load_config`]: open once (closing the
/// stat-then-read TOCTOU window), reject a non-regular file before any read,
/// cap the body at [`crate::lm::load::MAX_CONFIG_BYTES`] via `Read::take`, and
/// on Unix carry `O_NONBLOCK | O_CLOEXEC` so a planted FIFO returns
/// immediately. Every failure (missing dir/file, non-regular, oversized,
/// unreadable, non-UTF-8) is a recoverable [`Error::Backend`].
fn read_bounded_adapter_config(dir: &Path) -> Result<String> {
  use std::io::Read;

  // No `dir.exists()` precheck: `Path::exists()` collapses `NotFound` /
  // `PermissionDenied` / symlink-loop into a single `false`, which would
  // synthesize a fabricated `NotFound` for a real `PermissionDenied`. The
  // downstream `OpenOptions::open` on `adapter_config.json` surfaces the
  // actual underlying `io::Error` via `FileIoPayload`, so a missing dir
  // becomes `FileIo(NotFound, .../adapter_config.json)` and a permission
  // failure stays a permission failure.
  let path = dir.join("adapter_config.json");

  #[cfg(unix)]
  let file = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(&path)
      .map_err(|e| {
        Error::FileIo(FileIoPayload::new(
          "load_adapters",
          FileOp::Open,
          path.to_path_buf(),
          e,
        ))
      })?
  };
  #[cfg(not(unix))]
  let file = std::fs::File::open(&path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "load_adapters",
      FileOp::Open,
      path.to_path_buf(),
      e,
    ))
  })?;

  let meta = file.metadata().map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "load_adapters",
      FileOp::Stat,
      path.to_path_buf(),
      e,
    ))
  })?;
  if !meta.is_file() {
    return Err(Error::FileIo(FileIoPayload::new(
      "load_adapters: adapter_config.json must be a regular file",
      FileOp::Stat,
      path,
      std::io::Error::from(std::io::ErrorKind::InvalidInput),
    )));
  }

  let cap = crate::lm::load::MAX_CONFIG_BYTES;
  let mut bytes = Vec::new();
  file.take(cap + 1).read_to_end(&mut bytes).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "load_adapters",
      FileOp::Read,
      path.clone(),
      e,
    ))
  })?;
  if bytes.len() as u64 > cap {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "load_adapters: adapter_config.json body",
      "MAX_CONFIG_BYTES",
      cap,
      bytes.len() as u64,
    )));
  }
  String::from_utf8(bytes).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "load_adapters: adapter_config.json",
      "UTF-8",
      e,
    ))
  })
}

/// Stat `<dir>/adapters.safetensors` before it is handed to
/// [`crate::io::load_safetensors`] (which mmaps whatever path it is given,
/// performing no validation). Mirrors the regular-file discipline of
/// [`read_bounded_adapter_config`] / [`crate::lm::load`]'s shard discovery:
///
/// - Open once with `O_NONBLOCK | O_CLOEXEC` on Unix so a planted **FIFO**
///   returns immediately instead of blocking the caller (symlinks are followed
///   — a cached-model layout may symlink the file — but the post-open `fstat`
///   below checks the *resolved target*).
/// - `fstat` the opened handle and require a **regular file**: a FIFO / device
///   / directory / symlink-to-any-of-those is rejected before `load_safetensors`
///   can mmap it.
/// - Enforce the [`MAX_ADAPTER_SAFETENSORS_BYTES`] budget on the reported size
///   so an oversized blob is a clear recoverable error, not an OOM.
///
/// Every violation (missing file, non-regular, oversized, unstattable) is a
/// recoverable [`Error::Backend`]. The handle is closed on return; the
/// subsequent [`crate::io::load_safetensors`] re-opens via mlx-c. (This leaves
/// a narrow TOCTOU window between the check and mlx-c's open — acceptable here,
/// matching `lm::load`'s shard discovery, since `load_safetensors` cannot be
/// handed a pre-opened descriptor; the budget still bounds a same-size swap and
/// `O_NONBLOCK` is moot once a regular file has been confirmed.)
fn check_adapter_safetensors(path: &Path) -> Result<()> {
  #[cfg(unix)]
  let file = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(path)
      .map_err(|e| {
        Error::FileIo(FileIoPayload::new(
          "load_adapters",
          FileOp::Open,
          path.to_path_buf(),
          e,
        ))
      })?
  };
  #[cfg(not(unix))]
  let file = std::fs::File::open(path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "load_adapters",
      FileOp::Open,
      path.to_path_buf(),
      e,
    ))
  })?;

  let meta = file.metadata().map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "load_adapters",
      FileOp::Stat,
      path.to_path_buf(),
      e,
    ))
  })?;
  if !meta.is_file() {
    return Err(Error::FileIo(FileIoPayload::new(
      "load_adapters: adapter safetensors must be a regular file",
      FileOp::Stat,
      path.to_path_buf(),
      std::io::Error::from(std::io::ErrorKind::InvalidInput),
    )));
  }
  if meta.len() > MAX_ADAPTER_SAFETENSORS_BYTES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "load_adapters: adapter safetensors body",
      "MAX_ADAPTER_SAFETENSORS_BYTES",
      MAX_ADAPTER_SAFETENSORS_BYTES,
      meta.len(),
    )));
  }
  Ok(())
}

#[cfg(test)]
mod tests;
