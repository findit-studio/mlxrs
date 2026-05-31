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
mod tests {
  use super::*;
  use crate::Dtype;

  // ───────────────────── hand-traced fixtures ─────────────────────

  /// Base weight `W` of shape [output_dims=2, input_dims=3]:
  /// ```text
  /// [[1, 0, 0],
  ///  [0, 1, 0]]
  /// ```
  /// so `x @ Wᵀ` projects `x=[x0,x1,x2]` to `[x0, x1]`.
  fn base_weight() -> Array {
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &(2, 3)).unwrap()
  }

  /// `lora_a` of shape [input_dims=3, r=2]:
  /// ```text
  /// [[1, 0],
  ///  [0, 1],
  ///  [0, 0]]
  /// ```
  fn lora_a() -> Array {
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &(3, 2)).unwrap()
  }

  /// `lora_b` of shape [r=2, output_dims=2]:
  /// ```text
  /// [[1, 0],
  ///  [0, 1]]
  /// ```
  fn lora_b() -> Array {
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap()
  }

  fn plain_params() -> AdapterParams {
    AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: None,
    }
  }

  /// Build an mlx-lm-native [`LoraConfig`] (LoRA, the given `num_layers`
  /// trailing-block window and `lora_parameters`).
  fn mlxlm_config(num_layers: i32, lora_parameters: LoraParameters) -> LoraConfig {
    LoraConfig {
      fine_tune_type: FineTuneType::Lora,
      lora_parameters,
      use_dora: false,
      selection: AdapterSelection::MlxLm { num_layers },
    }
  }

  /// The mlx-lm trailing-block window count of a config — asserts the config
  /// is mlx-lm-native (NOT PEFT, which has no `num_layers`). Test-only.
  fn mlxlm_num_layers(cfg: &LoraConfig) -> i32 {
    match &cfg.selection {
      AdapterSelection::MlxLm { num_layers } => *num_layers,
      AdapterSelection::Peft(_) => panic!("expected an mlx-lm-native config, got PEFT"),
    }
  }

  /// The `keys`-allowlisted rank-2 `LoraParameters` the layer-selection tests
  /// reuse (`scale = 2.0`, the given `keys`; empty = auto-discovery).
  fn keyed_params(keys: Vec<String>) -> LoraParameters {
    LoraParameters {
      rank: 2,
      scale: Some(2.0),
      alpha: None,
      keys,
      dropout: None,
    }
  }

  fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b.iter()) {
      assert!((x - y).abs() <= tol, "‖{x} - {y}‖ > {tol} ({a:?} vs {b:?})");
    }
  }

  // ───────────────────── LoRALinear forward ─────────────────────

  #[test]
  fn lora_linear_forward_hand_traced() {
    // x = [1, 2, 3]; scale = 2.0.
    // base(x)  = x @ Wᵀ = [1, 2]
    // x @ a    = [1, 2]  (a picks first two coords)
    // (x@a)@b  = [1, 2]
    // out      = base + scale*z = [1 + 2*1, 2 + 2*2] = [3, 6]
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);
  }

  #[test]
  fn lora_linear_forward_with_bias() {
    // bias = [10, 20]; out = [3, 6] + [10, 20] = [13, 26].
    let bias = Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap();
    let base = BaseLinear::dense(base_weight(), Some(bias)).unwrap();
    let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[13.0, 26.0], 1e-5);
  }

  #[test]
  fn lora_linear_zero_b_is_identity() {
    // lora_b all zeros ⇒ the low-rank term vanishes ⇒ out == base(x).
    // (This is the just-loaded-before-training state; an inference adapter has
    // a trained, non-zero lora_b, but the math must reduce correctly.)
    let zero_b = Array::zeros::<f32>(&(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: zero_b,
      magnitude: None,
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = LoRALinear::new(base, params, 20.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[1.0, 2.0], 1e-5);
  }

  // ───────────────────── fuse == forward ─────────────────────

  #[test]
  fn lora_fuse_matches_forward() {
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();

    let mut via_forward = layer.forward(&x).unwrap();
    // Fuse, then run the fused base's plain forward — must match.
    let fused = layer.fuse(false).unwrap();
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      1e-5,
    );
  }

  #[test]
  fn lora_fuse_with_bias_matches_forward() {
    let bias = Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap();
    let base = BaseLinear::dense(base_weight(), Some(bias)).unwrap();
    let layer = LoRALinear::new(base, plain_params(), 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut via_forward = layer.forward(&x).unwrap();
    let fused = layer.fuse(false).unwrap();
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      1e-5,
    );
  }

  // ───────────────────── DoRA forward ─────────────────────

  #[test]
  fn dora_linear_forward_hand_traced() {
    // DoRA with m chosen to equal ‖adapted‖₂ so the renorm is the identity,
    // making the expected output the same [3, 6] as the LoRA case — this
    // isolates the renorm wiring (m/denom == 1 row-wise).
    //
    // adapted = W + scale*(lora_bᵀ @ lora_aᵀ); with scale=2,
    //   lora_bᵀ = [[1,0],[0,1]], lora_aᵀ = [[1,0,0],[0,1,0]]
    //   lora_bᵀ @ lora_aᵀ = [[1,0,0],[0,1,0]]
    //   adapted = [[1,0,0],[0,1,0]] + 2*[[1,0,0],[0,1,0]] = [[3,0,0],[0,3,0]]
    //   ‖adapted‖₂ row-wise = [3, 3]
    // Set m = [3, 3] ⇒ m/denom = [1, 1] ⇒ out == LoRA out == [3, 6].
    let m = Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: Some(m),
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = DoRALinear::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);
  }

  #[test]
  fn dora_linear_forward_renorm_halves() {
    // Same adapted norm [3, 3], but m = [1.5, 1.5] ⇒ m/denom = [0.5, 0.5] ⇒
    // out = 0.5 * [3, 6] = [1.5, 3.0].
    let m = Array::from_slice::<f32>(&[1.5, 1.5], &(2usize,)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: Some(m),
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = DoRALinear::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[1.5, 3.0], 1e-5);
  }

  #[test]
  fn dora_fuse_matches_forward() {
    let m = Array::from_slice::<f32>(&[1.5, 2.5], &(2usize,)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: Some(m),
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = DoRALinear::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut via_forward = layer.forward(&x).unwrap();
    let fused = layer.fuse(false).unwrap();
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      1e-4,
    );
  }

  #[test]
  fn dora_requires_magnitude() {
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let err = DoRALinear::new(base, plain_params(), 2.0).unwrap_err();
    assert!(matches!(err, Error::MissingField(_)));
  }

  // ───────────────────── QLoRA (quantized base) ─────────────────────

  #[test]
  fn qlora_forward_matches_dense_within_quant_error() {
    // Quantize a dense base, wrap with LoRA, and assert the QLoRA forward is
    // close to the dense LoRA forward (within affine-quant error). Use a
    // group_size that divides input_dims and a wide-ish weight so the quant
    // error stays small.
    //
    // input_dims must be divisible by group_size; use input_dims=64,
    // output_dims=2, group_size=32, bits=8 (low error).
    let input_dims = 64usize;
    let output_dims = 2usize;
    // Dense weight: row 0 = 1.0s, row 1 = 0.5s (well-represented at 8 bits).
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();

    // lora_a [input_dims, r=2] small constant; lora_b [r=2, output_dims].
    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: None,
    };

    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

    // Dense LoRA forward.
    let dense_base = BaseLinear::dense(dense_w.try_clone().unwrap(), None).unwrap();
    let dense_layer = LoRALinear::new(dense_base, params.try_clone().unwrap(), 2.0).unwrap();
    let mut dense_out = dense_layer.forward(&x).unwrap();

    // Quantized base (affine, group_size=32, bits=8).
    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let q_layer = LoRALinear::new(q_base, params, 2.0).unwrap();
    let mut q_out = q_layer.forward(&x).unwrap();

    // Within affine-quant error (8-bit, uniform weights → small).
    approx_eq(
      &q_out.to_vec::<f32>().unwrap(),
      &dense_out.to_vec::<f32>().unwrap(),
      1e-2,
    );
  }

  #[test]
  fn qlora_fuse_dequantize_matches_forward() {
    // fuse(dequantize=true) on a quantized base yields a dense fused linear
    // whose forward matches the QLoRA forward within quant error.
    let input_dims = 64usize;
    let output_dims = 2usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();
    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: None,
    };
    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let q_layer = LoRALinear::new(q_base, params, 2.0).unwrap();
    let mut via_forward = q_layer.forward(&x).unwrap();

    let fused = q_layer.fuse(true).unwrap();
    assert!(matches!(fused, BaseLinear::Dense { .. }));
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      1e-2,
    );
  }

  // ───────────────────── config parsing ─────────────────────

  #[test]
  fn config_parse_lora_basic() {
    let json = r#"{
      "fine_tune_type": "lora",
      "num_layers": 4,
      "lora_parameters": { "rank": 16, "scale": 20.0 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.fine_tune_type, FineTuneType::Lora);
    assert_eq!(mlxlm_num_layers(&cfg), 4);
    assert_eq!(cfg.rank(), 16);
    assert_eq!(cfg.scale(), 20.0);
    assert!(!cfg.is_dora());
  }

  #[test]
  fn config_parse_peft_flat_shape() {
    // A REAL PEFT adapter_config.json: NO `lora_parameters` nesting — `r`,
    // `lora_alpha`, `target_modules`, `lora_dropout`, `peft_type` all flat at
    // the top level. The dual-shape `Deserialize` must detect the PEFT shape
    // and map the flat fields, so a PEFT-trained adapter does NOT silently
    // fall back to the default rank/scale.
    let json = r#"{
      "peft_type": "LORA",
      "r": 16,
      "lora_alpha": 32.0,
      "target_modules": ["q_proj", "v_proj"],
      "lora_dropout": 0.05,
      "bias": "none"
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.rank(), 16, "PEFT top-level `r` must populate `rank`");
    // alpha/rank = 32/16 = 2.0 — `lora_alpha`/`r` resolves the scale.
    assert_eq!(cfg.scale(), 2.0);
    // `target_modules` lands in the PEFT selection (NOT `keys` — PEFT
    // selection is the richer `PeftSelection`).
    assert!(cfg.lora_parameters.keys_slice().is_empty());
    let peft = cfg.peft().expect("PEFT config must carry a PeftSelection");
    match &peft.target_modules {
      Some(ModuleMatcher::List(names)) => {
        assert_eq!(names, &["q_proj".to_string(), "v_proj".to_string()]);
      }
      other => panic!("expected a target_modules List, got {other:?}"),
    }
    // `lora_dropout` maps to `dropout` (carried, ignored at inference).
    assert_eq!(cfg.lora_parameters.dropout, Some(0.05));
    assert_eq!(cfg.fine_tune_type, FineTuneType::Lora);
    // A PEFT config must NOT inherit mlx-lm's `num_layers` window
    // — PEFT adapts EVERY matching block. The selection is `Peft`, never
    // `MlxLm { num_layers }`.
    assert!(
      matches!(cfg.selection, AdapterSelection::Peft(_)),
      "a PEFT config must select via PeftSelection, never the mlx-lm num_layers window"
    );
    assert!(!cfg.is_dora());
  }

  #[test]
  fn config_parse_peft_use_dora() {
    // A PEFT config with `use_dora: true` selects DoRA (PEFT carries the DoRA
    // signal in `use_dora`, not a `fine_tune_type`).
    let json = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "use_dora": true
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert!(cfg.is_dora(), "PEFT `use_dora` must select DoRA");
    assert_eq!(cfg.scale(), 2.0);
  }

  #[test]
  fn config_parse_peft_no_peft_type_still_detected() {
    // A flat config with top-level `r` / `lora_alpha` / `target_modules` but
    // NO `peft_type` is still recognized as the PEFT shape (some exporters
    // omit `peft_type`) — the absence of a `lora_parameters` nesting plus the
    // flat PEFT keys is the signal.
    let json = r#"{
      "r": 4,
      "lora_alpha": 8.0,
      "target_modules": ["o_proj"]
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.rank(), 4);
    assert_eq!(cfg.scale(), 2.0);
    let peft = cfg
      .peft()
      .expect("PEFT shape detected ⇒ PeftSelection present");
    assert!(matches!(
      &peft.target_modules,
      Some(ModuleMatcher::List(n)) if n == &["o_proj".to_string()]
    ));
  }

  #[test]
  fn config_parse_peft_default_rank_when_r_absent() {
    // PEFT `r` defaults to 8 in `peft` itself — a PEFT config without `r` but
    // with another PEFT marker must still parse and use DEFAULT_LORA_RANK.
    let json = r#"{ "peft_type": "LORA", "lora_alpha": 16.0, "target_modules": ["q_proj"] }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.rank(), DEFAULT_LORA_RANK);
  }

  #[test]
  fn config_parse_peft_non_lora_peft_type_is_err() {
    // A non-LoRA PEFT method (LOHA / LOKR / IA3 / prompt-tuning / …) is a
    // different adapter kind — this loader handles LoRA/DoRA only, so a
    // `peft_type` other than "LORA" is a recoverable parse error.
    for kind in ["LOHA", "LOKR", "IA3", "PROMPT_TUNING"] {
      let json = format!(r#"{{ "peft_type": "{kind}", "r": 8, "target_modules": ["q_proj"] }}"#);
      assert!(
        LoraConfig::from_json(&json).is_err(),
        "peft_type {kind:?} must be rejected"
      );
    }
  }

  #[test]
  fn config_parse_peft_type_case_insensitive() {
    // PEFT writes `peft_type` upper-case ("LORA"); accept any case.
    let json =
      r#"{ "peft_type": "Lora", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"] }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.rank(), 8);
  }

  #[test]
  fn config_parse_peft_target_modules_regex() {
    // PEFT `target_modules` may be a single regex string — modeled faithfully
    // via the `regex` crate (`re.fullmatch` semantics), NOT rejected.
    let json = r#"{ "peft_type": "LORA", "r": 8, "target_modules": ".*\\.(q|v)_proj" }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let peft = cfg.peft().unwrap();
    let target = match &peft.target_modules {
      Some(ModuleMatcher::Regex(re)) => re,
      other => panic!("expected a target_modules Regex, got {other:?}"),
    };
    // `re.fullmatch` — the whole module key must match.
    assert!(target.is_match("model.layers.0.self_attn.q_proj"));
    assert!(target.is_match("model.layers.7.self_attn.v_proj"));
    assert!(!target.is_match("model.layers.0.self_attn.k_proj"));
  }

  #[test]
  fn config_parse_peft_invalid_regex_target_modules_is_err() {
    // A `target_modules` regex string that fails to compile is a recoverable
    // parse error (a malformed regex must not silently match nothing).
    let json = r#"{ "peft_type": "LORA", "r": 8, "target_modules": "(unclosed" }"#;
    assert!(
      LoraConfig::from_json(json).is_err(),
      "an uncompilable `target_modules` regex must be rejected"
    );
  }

  #[test]
  fn config_lora_parameters_nesting_wins_over_flat_keys() {
    // A `lora_parameters` object is the unambiguous mlx-lm-native marker: when
    // it is present the flat PEFT keys are NOT consulted (a real config never
    // mixes the two shapes — this just pins the detection precedence).
    let json = r#"{
      "fine_tune_type": "lora",
      "num_layers": 3,
      "lora_parameters": { "rank": 64, "scale": 8.0 },
      "r": 1, "lora_alpha": 999.0
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.rank(), 64, "nested `lora_parameters.rank` wins");
    assert_eq!(
      cfg.scale(),
      8.0,
      "nested literal `scale` wins, flat keys ignored"
    );
    assert_eq!(mlxlm_num_layers(&cfg), 3);
  }

  #[test]
  fn config_parse_dora_and_alpha_scale() {
    // mlx-lm-native nested shape — its alpha key is `alpha`. alpha/rank scale:
    // alpha=32, rank=8 ⇒ scale=4.0. fine_tune_type dora.
    let json = r#"{
      "fine_tune_type": "dora",
      "num_layers": 2,
      "lora_parameters": { "rank": 8, "alpha": 32.0 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert!(cfg.is_dora());
    assert_eq!(cfg.scale(), 4.0);
  }

  #[test]
  fn config_use_dora_flag() {
    let json = r#"{
      "fine_tune_type": "lora",
      "use_dora": true,
      "lora_parameters": { "rank": 8, "scale": 10.0 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert!(cfg.is_dora());
  }

  #[test]
  fn config_defaults_and_unknown_keys_ignored() {
    // Minimal config + extra training-only keys → parses, defaults applied.
    let json = r#"{ "optimizer": "adam", "learning_rate": 1e-4 }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.fine_tune_type, FineTuneType::Lora);
    assert_eq!(mlxlm_num_layers(&cfg), DEFAULT_NUM_LAYERS);
    assert_eq!(cfg.rank(), DEFAULT_LORA_RANK);
    assert_eq!(cfg.scale(), DEFAULT_LORA_SCALE);
  }

  #[test]
  fn config_unknown_fine_tune_type_is_err() {
    let json = r#"{ "fine_tune_type": "bogus" }"#;
    assert!(LoraConfig::from_json(json).is_err());
  }

  // ───────────────────── path/key helpers ─────────────────────

  #[test]
  fn path_key_matching() {
    assert!(path_matches_key(
      "model.layers.27.self_attn.q_proj",
      "self_attn.q_proj"
    ));
    assert!(path_matches_key("self_attn.q_proj", "self_attn.q_proj"));
    assert!(!path_matches_key(
      "model.layers.27.self_attn.k_proj",
      "q_proj"
    ));
    // Must match on a segment boundary, not a substring.
    assert!(!path_matches_key("model.xq_proj", "q_proj"));
  }

  #[test]
  fn block_index_parsing() {
    assert_eq!(
      parse_block_index("model.layers.27.self_attn.q_proj"),
      Some(27)
    );
    assert_eq!(parse_block_index("model.layers.0.mlp.down_proj"), Some(0));
    assert_eq!(parse_block_index("model.embed_tokens"), None);
    assert_eq!(parse_block_index("lm_head"), None);
  }

  // ───────────────────── linear_to_lora_layers ─────────────────────

  /// Build a tiny weight map with 4 decoder blocks, each carrying a single
  /// `self_attn.q_proj.weight` (and one block also a `k_proj`), plus a
  /// top-level `lm_head.weight`.
  fn toy_weights() -> Weights {
    let mut w = Weights::new();
    for b in 0..4 {
      w.insert(
        format!("model.layers.{b}.self_attn.q_proj.weight"),
        base_weight(),
      );
    }
    w.insert(
      "model.layers.0.self_attn.k_proj.weight".to_string(),
      base_weight(),
    );
    w.insert("lm_head.weight".to_string(), base_weight());
    w
  }

  /// Adapter params for every q_proj path in the toy map (4 blocks).
  fn toy_adapter_params() -> HashMap<String, AdapterParams> {
    toy_adapter_params_for(&[0, 1, 2, 3])
  }

  /// Adapter params for the q_proj paths of the given block indices only.
  /// Used to keep an adapter's factor set aligned with the `num_layers` window
  /// under test — the completeness postcondition rejects factors for a path
  /// outside the selection, so a windowed test must supply only in-window
  /// factors.
  fn toy_adapter_params_for(blocks: &[i32]) -> HashMap<String, AdapterParams> {
    let mut m = HashMap::new();
    for &b in blocks {
      m.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    }
    m
  }

  #[test]
  fn lora_layers_keys_and_num_layers_window() {
    // keys=["self_attn.q_proj"], num_layers=2 ⇒ only blocks 2,3's q_proj wrap.
    // The adapter supplies factors for exactly those two blocks (an adapter
    // that also carried block-0/1 factors would now be a config mismatch — see
    // `lora_layers_extra_factors_outside_window_is_err`).
    let weights = toy_weights();
    let params = toy_adapter_params_for(&[2, 3]);
    let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    // Only blocks 2 and 3 are inside the trailing-2 window.
    assert!(layers.contains_key("model.layers.2.self_attn.q_proj"));
    assert!(layers.contains_key("model.layers.3.self_attn.q_proj"));
    assert!(!layers.contains_key("model.layers.0.self_attn.q_proj"));
    assert!(!layers.contains_key("model.layers.1.self_attn.q_proj"));
    // k_proj never matches the key.
    assert!(!layers.contains_key("model.layers.0.self_attn.k_proj"));
    // lm_head is a non-block path and not in keys → untouched.
    assert!(!layers.contains_key("lm_head"));
    assert_eq!(layers.len(), 2);
  }

  #[test]
  fn lora_layers_covers_all_blocks_when_num_layers_large() {
    let weights = toy_weights();
    let params = toy_adapter_params();
    // num_layers 16 > 4 blocks ⇒ all q_proj blocks wrap.
    let cfg = mlxlm_config(16, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
  }

  // ───────────────────── load_adapters end-to-end ─────────────────────

  /// Write a mock adapter dir: adapter_config.json + adapters.safetensors with
  /// factors for two q_proj paths.
  fn write_mock_adapter(dir: &Path, fine_tune_type: &str, with_m: bool) {
    let config = format!(
      r#"{{
        "fine_tune_type": "{fine_tune_type}",
        "num_layers": 16,
        "lora_parameters": {{ "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }}
      }}"#
    );
    std::fs::write(dir.join("adapter_config.json"), config).unwrap();

    let mut arrays: HashMap<String, Array> = HashMap::new();
    for b in 0..4 {
      let path = format!("model.layers.{b}.self_attn.q_proj");
      arrays.insert(format!("{path}.lora_a"), lora_a());
      arrays.insert(format!("{path}.lora_b"), lora_b());
      if with_m {
        // m = ‖adapted‖₂ (so renorm is identity) → [3, 3] for these factors.
        arrays.insert(
          format!("{path}.m"),
          Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap(),
        );
      }
    }
    crate::io::save_safetensors(&dir.join("adapters.safetensors"), &arrays).unwrap();
  }

  #[test]
  fn load_adapters_lora_end_to_end() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_lora_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    write_mock_adapter(&tmp, "lora", false);

    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    // 4 q_proj blocks adapted.
    assert_eq!(layers.len(), 4);
    assert!(matches!(
      layers.get("model.layers.0.self_attn.q_proj"),
      Some(LoraLayer::Lora(_))
    ));

    // Forward through an adapted layer matches the hand-traced LoRA result.
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layers
      .get("model.layers.0.self_attn.q_proj")
      .unwrap()
      .forward(&x)
      .unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);

    std::fs::remove_dir_all(&tmp).ok();
  }

  /// Write a mock adapter dir whose `adapter_config.json` is `config_json`
  /// (caller-supplied, so a test can vary `rank`/`r`/`alpha`) and whose
  /// `adapters.safetensors` carries rank-`r` factors for the 4 q_proj paths
  /// over the toy `[2, 3]` base: `lora_a` is `[3, r]`, `lora_b` is `[r, 2]`.
  fn write_mock_adapter_rank(dir: &Path, config_json: &str, r: usize) {
    std::fs::write(dir.join("adapter_config.json"), config_json).unwrap();
    let la = Array::full::<f32>(&(3usize, r), 0.01).unwrap();
    let lb = Array::full::<f32>(&(r, 2usize), 0.01).unwrap();
    let mut arrays: HashMap<String, Array> = HashMap::new();
    for b in 0..4 {
      let path = format!("model.layers.{b}.self_attn.q_proj");
      arrays.insert(format!("{path}.lora_a"), la.try_clone().unwrap());
      arrays.insert(format!("{path}.lora_b"), lb.try_clone().unwrap());
    }
    crate::io::save_safetensors(&dir.join("adapters.safetensors"), &arrays).unwrap();
  }

  /// Write a mock **PEFT** adapter dir: a caller-supplied
  /// `adapter_config.json` plus a PEFT-keyed `adapter_model.safetensors`.
  /// `paths` are the base-module paths (without `.weight`) to ship factors
  /// for; for each, the PEFT tensors `base_model.model.<path>.lora_A.weight`
  /// (`[r, in=3]`) and `.lora_B.weight` (`[out=2, r]`) are written — the PEFT
  /// orientation (transposed vs the mlxrs scheme). When `with_dora`, a
  /// `.lora_magnitude_vector` (`[out=2]`) is added per path. The PEFT factor
  /// values are `value` (so the post-translation `lora_a`/`lora_b` are
  /// constant — handy for hand-traced math).
  fn write_mock_peft_adapter(
    dir: &Path,
    config_json: &str,
    paths: &[&str],
    r: usize,
    with_dora: bool,
    value: f32,
  ) {
    std::fs::write(dir.join("adapter_config.json"), config_json).unwrap();
    // PEFT `lora_A.weight` is `[r, in_features]`; `lora_B.weight` is
    // `[out_features, r]` (the transpose of the mlxrs `lora_a` / `lora_b`).
    let lora_a_peft = Array::full::<f32>(&(r, 3usize), value).unwrap();
    let lora_b_peft = Array::full::<f32>(&(2usize, r), value).unwrap();
    let mut arrays: HashMap<String, Array> = HashMap::new();
    for path in paths {
      arrays.insert(
        format!("base_model.model.{path}.lora_A.weight"),
        lora_a_peft.try_clone().unwrap(),
      );
      arrays.insert(
        format!("base_model.model.{path}.lora_B.weight"),
        lora_b_peft.try_clone().unwrap(),
      );
      if with_dora {
        // DoRA magnitude — [out_features=2], no transpose in either scheme.
        arrays.insert(
          format!("base_model.model.{path}.lora_magnitude_vector"),
          Array::from_slice::<f32>(&[1.0, 1.0], &(2usize,)).unwrap(),
        );
      }
    }
    crate::io::save_safetensors(&dir.join("adapter_model.safetensors"), &arrays).unwrap();
  }

  #[test]
  fn load_adapters_peft_flat_shape_rank16_end_to_end() {
    // A REAL PEFT-shaped adapter_config.json — flat top-level `peft_type` /
    // `r` / `lora_alpha` / `target_modules` / `lora_dropout`, NO
    // `lora_parameters` nesting — plus rank-16 factor tensors. The dual-shape
    // `Deserialize` must read `r:16` (so the rank-16 factors pass the
    // config-rank cross-check), resolve the scale to `lora_alpha/r = 32/16 =
    // 2.0`, and use `target_modules` to drive layer selection.
    let tmp = std::env::temp_dir().join(format!("mlxrs_peft_flat16_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "peft_type": "LORA",
      "r": 16,
      "lora_alpha": 32.0,
      "target_modules": ["self_attn.q_proj"],
      "lora_dropout": 0.0,
      "bias": "none"
    }"#;
    let q_paths: Vec<String> = (0..4)
      .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
      .collect();
    let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
    write_mock_peft_adapter(&tmp, cfg, &q_refs, 16, false, 0.01);
    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    // `target_modules: ["self_attn.q_proj"]` selects the 4 q_proj paths (and
    // NOT the lone k_proj) — i.e. PEFT's `target_modules` drove the selection.
    assert_eq!(layers.len(), 4);
    assert!(layers.contains_key("model.layers.0.self_attn.q_proj"));
    assert!(!layers.contains_key("model.layers.0.self_attn.k_proj"));
    // The resolved scale is lora_alpha/r = 32/16 = 2.0.
    if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.0.self_attn.q_proj") {
      assert_eq!(l.scale(), 2.0, "PEFT scale must be lora_alpha/r");
    } else {
      panic!("expected a LoRA layer");
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_peft_flat_shape_rank8_scale_not_default() {
    // A PEFT config with `r:8` + rank-8 factors must load and resolve the
    // scale to `lora_alpha/8`, NOT the literal-`scale` default of 20.0 (PEFT
    // has no literal-`scale` key — the prior nested-only alias would have left
    // a real flat PEFT config defaulting both rank and scale).
    let tmp = std::env::temp_dir().join(format!("mlxrs_peft_flat8_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 32.0,
      "target_modules": ["self_attn.q_proj"]
    }"#;
    let q_paths: Vec<String> = (0..4)
      .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
      .collect();
    let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
    write_mock_peft_adapter(&tmp, cfg, &q_refs, 8, false, 0.01);
    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.0.self_attn.q_proj") {
      // lora_alpha/r = 32/8 = 4.0 — explicitly NOT DEFAULT_LORA_SCALE (20.0).
      assert_eq!(l.scale(), 4.0);
      assert_ne!(l.scale(), DEFAULT_LORA_SCALE);
    } else {
      panic!("expected a LoRA layer");
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_rank_drift_is_shape_mismatch() {
    // mlx-lm-native config declares rank 8 with `alpha` present, but the
    // factor tensors are rank 16 (a stale `adapter_config.json` drift).
    // Without the config-vs-tensor rank cross-check this silently builds
    // rank-16 factors and scales by alpha/8 instead of alpha/16 — wrong
    // strength. It must fail loudly at load with a LengthMismatch.
    let tmp = std::env::temp_dir().join(format!("mlxrs_rankdrift_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 8, "alpha": 32.0, "keys": ["self_attn.q_proj"] }
    }"#;
    write_mock_adapter_rank(&tmp, cfg, 16);
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::LengthMismatch(_)),
      "rank drift must be a LengthMismatch, got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_peft_rank_drift_is_shape_mismatch() {
    // The rank-drift guard must also catch a PEFT-flat config: `r:8`
    // declared but rank-16 factors shipped. The dual-shape `Deserialize`
    // reads `r` correctly, then `validate_config_rank` rejects the drift.
    let tmp =
      std::env::temp_dir().join(format!("mlxrs_peft_rankdrift_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 32.0,
      "target_modules": ["self_attn.q_proj"]
    }"#;
    // `r:8` declared, but rank-16 factors shipped.
    let q_paths: Vec<String> = (0..4)
      .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
      .collect();
    let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
    write_mock_peft_adapter(&tmp, cfg, &q_refs, 16, false, 0.01);
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::LengthMismatch(_)),
      "PEFT rank drift must be a LengthMismatch, got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_dora_end_to_end() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_dora_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    write_mock_adapter(&tmp, "dora", true);

    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    assert!(matches!(
      layers.get("model.layers.0.self_attn.q_proj"),
      Some(LoraLayer::Dora(_))
    ));
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_dora_missing_magnitude_is_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_dora_nom_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    // fine_tune_type dora but no `.m` arrays → recoverable Err.
    write_mock_adapter(&tmp, "dora", false);
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::MissingKey(_)),
      "missing DoRA magnitude must be MissingKey, got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_full_is_unsupported_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_full_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    write_mock_adapter(&tmp, "full", false);
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::UnknownEnumValue(_)),
      "fine_tune_type=full rejection must be UnknownEnumValue, got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_unknown_fine_tune_type_is_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_bogus_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    write_mock_adapter(&tmp, "bogus", false);
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::Parse(_)),
      "unknown fine_tune_type must be a serde Parse error, got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_missing_config_is_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_nocfg_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    // Only write the safetensors, no config.
    let arrays: HashMap<String, Array> = HashMap::new();
    crate::io::save_safetensors(&tmp.join("adapters.safetensors"), &arrays).unwrap();
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::FileIo(_)),
      "missing adapter_config.json must be a FileIo error, got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_missing_dir_is_err() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_nodir_test_{}", std::process::id()));
    // Do NOT create the dir.
    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::FileIo(_)),
      "missing adapter dir must be a FileIo error, got {err:?}"
    );
  }

  // ───────────────────── factor-shape validation ─────────────────────

  #[test]
  fn lora_rejects_mismatched_output_dims() {
    // lora_b last axis (3) != base output_dims (2).
    let bad_b = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &(2, 3)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: bad_b,
      magnitude: None,
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let err = LoRALinear::new(base, params, 2.0).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch(_)));
  }

  #[test]
  fn lora_rejects_rank_mismatch() {
    // lora_a [3, 2] but lora_b [3, 2] (leading 3 != a's r=2).
    let bad_b = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &(3, 2)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: bad_b,
      magnitude: None,
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let err = LoRALinear::new(base, params, 2.0).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch(_)));
  }

  // ───────── lora_a input-dim cross-check ─────────

  #[test]
  fn lora_rejects_wrong_lora_a_input_dim_dense() {
    // Dense base W is [output_dims=2, input_dims=3]; a lora_a with leading axis
    // 2 (≠ input_dims 3) must be rejected at construction, not deferred to a
    // mlx-c matmul failure on the first forward.
    let bad_a = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: bad_a,
      lora_b: lora_b(),
      magnitude: None,
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let err = LoRALinear::new(base, params, 2.0).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch(_)));
  }

  #[test]
  fn lora_rejects_wrong_lora_a_input_dim_quantized() {
    // Quantized base: dense [2, 64] affine-quantized at 8 bits ⇒ packed [2, 16];
    // base_input_dims recovers 16 * 32 / 8 = 64. A lora_a with leading axis 32
    // (≠ 64) must be rejected at construction.
    let input_dims = 64usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(2, input_dims)).unwrap();
    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();

    // input_dims should be 64 — supply a wrong-width lora_a [32, 2].
    let bad_a = Array::full::<f32>(&(32usize, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: bad_a,
      lora_b: lb,
      magnitude: None,
    };
    let err = LoRALinear::new(q_base, params, 2.0).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch(_)));
  }

  #[test]
  fn lora_a_correct_input_dim_quantized_ok() {
    // The positive companion: a correctly-sized lora_a [64, 2] over the same
    // quantized base constructs cleanly (base_input_dims == 64 == lora_a[0]).
    let input_dims = 64usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(2, input_dims)).unwrap();
    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let params = AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: None,
    };
    assert!(LoRALinear::new(q_base, params, 2.0).is_ok());
  }

  // ───────── scale precedence (alpha wins) ─────────

  #[test]
  fn resolved_scale_alpha_only() {
    // alpha present, no scale ⇒ alpha / rank.
    let p = LoraParameters {
      rank: 8,
      scale: None,
      alpha: Some(32.0),
      keys: Vec::new(),
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), 4.0);
  }

  #[test]
  fn resolved_scale_scale_only() {
    // scale present, no alpha ⇒ the literal scale.
    let p = LoraParameters {
      rank: 8,
      scale: Some(7.5),
      alpha: None,
      keys: Vec::new(),
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), 7.5);
  }

  #[test]
  fn resolved_scale_alpha_wins_over_scale() {
    // BOTH present ⇒ alpha / rank WINS over the literal scale (PEFT precedence).
    // alpha=64, rank=16 ⇒ 4.0, NOT the literal 99.0.
    let p = LoraParameters {
      rank: 16,
      scale: Some(99.0),
      alpha: Some(64.0),
      keys: Vec::new(),
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), 4.0);
  }

  #[test]
  fn resolved_scale_neither_is_default() {
    // Neither present ⇒ DEFAULT_LORA_SCALE.
    let p = LoraParameters {
      rank: 8,
      scale: None,
      alpha: None,
      keys: Vec::new(),
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), DEFAULT_LORA_SCALE);
  }

  #[test]
  fn resolved_scale_alpha_with_nonpositive_rank_falls_back() {
    // Defensive floor: alpha present but rank <= 0 ⇒ `alpha / rank` is
    // undefined ⇒ fall through to the literal scale, then the default.
    let p = LoraParameters {
      rank: 0,
      scale: Some(5.0),
      alpha: Some(32.0),
      keys: Vec::new(),
      dropout: None,
    };
    assert_eq!(p.resolved_scale(), 5.0);
    let p_no_scale = LoraParameters {
      rank: -1,
      scale: None,
      alpha: Some(32.0),
      keys: Vec::new(),
      dropout: None,
    };
    assert_eq!(p_no_scale.resolved_scale(), DEFAULT_LORA_SCALE);
  }

  #[test]
  fn config_both_scale_and_alpha_alpha_wins() {
    // mlx-lm-native config carrying BOTH a literal `scale` and `alpha` ⇒
    // alpha/rank wins over the literal scale.
    let json = r#"{
      "fine_tune_type": "lora",
      "lora_parameters": { "rank": 8, "scale": 50.0, "alpha": 16.0 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.scale(), 2.0); // 16 / 8, not the literal 50.0
  }

  // ───────── num_layers <= 0 selects ALL blocks ─────────

  #[test]
  fn lora_layers_num_layers_negative_one_selects_all_blocks() {
    // mlx-lm `model.layers[-max(-1,0):]` == `layers[-0:]` == `layers[0:]` ⇒
    // num_layers: -1 adapts EVERY decoder block, not none.
    let weights = toy_weights();
    let params = toy_adapter_params(); // factors for all 4 q_proj blocks
    let cfg = mlxlm_config(-1, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 4, "num_layers=-1 must adapt all 4 blocks");
    for b in 0..4 {
      assert!(layers.contains_key(&format!("model.layers.{b}.self_attn.q_proj")));
    }
  }

  #[test]
  fn lora_layers_num_layers_zero_selects_all_blocks() {
    // num_layers: 0 ⇒ `max(0,0)=0` ⇒ `layers[-0:]` == all blocks too.
    let weights = toy_weights();
    let params = toy_adapter_params();
    let cfg = mlxlm_config(0, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 4, "num_layers=0 must adapt all 4 blocks");
  }

  // ───────── adapter-completeness postcondition ─────────

  #[test]
  fn lora_layers_explicit_key_missing_factors_is_err() {
    // keys=["self_attn.q_proj"], num_layers covers all 4 blocks, but the
    // adapter only supplies factors for blocks 0,1 ⇒ blocks 2,3 are selected
    // targets with no factors ⇒ typed `Error::MissingKey` (case a) keyed on
    // the FIRST (sorted) missing target.
    let weights = toy_weights();
    let params = toy_adapter_params_for(&[0, 1]);
    let cfg = mlxlm_config(16, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let err = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap_err();
    match err {
      Error::MissingKey(p) => {
        assert!(
          p.context().contains("explicitly-selected adapter target"),
          "context names the explicit-selection rule: {}",
          p.context()
        );
        assert_eq!(p.key(), "model.layers.2.self_attn.q_proj");
      }
      other => panic!("expected Error::MissingKey, got {other:?}"),
    }
  }

  #[test]
  fn lora_layers_unused_adapter_factor_is_err() {
    // The adapter carries a factor group for a path that exists in NO base
    // weight (a path-prefix mismatch / config drift) ⇒ Err (case b): typed
    // `Error::LayerKeyed` keyed on the unused path wrapping a typed
    // `Error::InvariantViolation` calling out the "must match a base layer"
    // rule.
    let weights = toy_weights();
    let mut params = toy_adapter_params(); // all 4 q_proj blocks (all match)
    params.insert(
      "model.layers.99.self_attn.q_proj".to_string(),
      plain_params(),
    );
    let cfg = mlxlm_config(16, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let err = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap_err();
    match err {
      Error::LayerKeyed(p) => {
        assert_eq!(p.layer(), "model.layers.99.self_attn.q_proj");
        let Error::InvariantViolation(iv) = p.inner() else {
          panic!(
            "expected inner Error::InvariantViolation, got {:?}",
            p.inner()
          );
        };
        assert!(
          iv.context().contains("adapter factor group")
            && iv.requirement().contains("must match a base layer"),
          "inner violation should call out base-layer matching: {iv:?}"
        );
      }
      other => panic!("expected Error::LayerKeyed, got {other:?}"),
    }
  }

  #[test]
  fn lora_layers_empty_result_is_err() {
    // keys names a projection that exists in NO base weight, and there are no
    // factors ⇒ nothing adapted ⇒ typed `Error::InvariantViolation` (case c).
    let weights = toy_weights();
    let params: HashMap<String, AdapterParams> = HashMap::new();
    let cfg = mlxlm_config(
      16,
      keyed_params(vec!["self_attn.nonexistent_proj".to_string()]),
    );
    let err = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap_err();
    match err {
      Error::InvariantViolation(p) => {
        assert_eq!(p.context(), "load_adapters: adapted-layer count");
        assert!(p.requirement().contains("must be >= 1"));
      }
      other => panic!("expected Error::InvariantViolation, got {other:?}"),
    }
  }

  #[test]
  fn lora_layers_autodiscovery_partial_factors_is_ok() {
    // keys: None (auto-discovery) ⇒ a base linear without factors is EXPECTED
    // (the adapter trains only a subset); only the unused-factor (b) and
    // empty-result (c) checks apply. Factors for 2 of the 4 q_proj blocks ⇒ Ok.
    let weights = toy_weights();
    let params = toy_adapter_params_for(&[2, 3]);
    let cfg = mlxlm_config(16, keyed_params(Vec::new()));
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 2);
  }

  #[test]
  fn load_adapters_unused_factor_end_to_end_is_err() {
    // End-to-end: an adapters.safetensors carrying a factor group for a path
    // absent from the base model ⇒ load_adapters rejects it.
    let tmp = std::env::temp_dir().join(format!("mlxrs_unused_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
    std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
    let mut arrays: HashMap<String, Array> = HashMap::new();
    for b in 0..4 {
      let path = format!("model.layers.{b}.self_attn.q_proj");
      arrays.insert(format!("{path}.lora_a"), lora_a());
      arrays.insert(format!("{path}.lora_b"), lora_b());
    }
    // A factor group for a path that is NOT in toy_weights().
    arrays.insert(
      "model.layers.42.self_attn.q_proj.lora_a".to_string(),
      lora_a(),
    );
    arrays.insert(
      "model.layers.42.self_attn.q_proj.lora_b".to_string(),
      lora_b(),
    );
    crate::io::save_safetensors(&tmp.join("adapters.safetensors"), &arrays).unwrap();

    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::LayerKeyed(ref p) if matches!(p.inner(), Error::InvariantViolation(_))),
      "unused factor group must be LayerKeyed(InvariantViolation), got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_empty_safetensors_is_err() {
    // An empty adapters.safetensors (no factor groups at all) ⇒ nothing adapted
    // ⇒ Err (case c), instead of a silently-unadapted Ok.
    let tmp = std::env::temp_dir().join(format!("mlxrs_emptyst_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
    std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
    let arrays: HashMap<String, Array> = HashMap::new();
    crate::io::save_safetensors(&tmp.join("adapters.safetensors"), &arrays).unwrap();

    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    assert!(
      matches!(err, Error::MissingKey(_)),
      "explicit-selection w/o factors must be MissingKey, got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
  }

  // ───────── QDoRA forward via quantized_matmul ─────────

  #[test]
  fn qdora_forward_matches_dense_within_quant_error() {
    // QDoRA (DoRA over a quantized base) + bias: the forward must match the
    // dense DoRA forward within affine-quant error. By construction the
    // quantized base output runs through quantized_matmul (base_output_no_bias),
    // never a full dense-weight matmul — the dequantized weight is materialized
    // only for the adapted-weight L2-norm.
    let input_dims = 64usize;
    let output_dims = 2usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();

    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    // m = ‖adapted‖₂ row-wise of the DENSE adapted weight (so dense + quantized
    // share the same magnitude vector — the renorm is identical).
    let bias = Array::from_slice::<f32>(&[3.0, -1.0], &(output_dims,)).unwrap();

    let dense_params = AdapterParams {
      lora_a: la.try_clone().unwrap(),
      lora_b: lb.try_clone().unwrap(),
      magnitude: None,
    };
    // Build a DoRALinear over the dense base to read back its computed adapted
    // norm via fuse? Simpler: pick m = norm of (dense_w + scale*delta).
    let scale = 2.0f32;
    let delta = lora_delta(&dense_params, scale).unwrap();
    let adapted = dense_w.add(&delta).unwrap();
    let m = ops::linalg_full::norm(&adapted, 2.0, &[1], false).unwrap();

    let dense_base = BaseLinear::dense(
      dense_w.try_clone().unwrap(),
      Some(bias.try_clone().unwrap()),
    )
    .unwrap();
    let dense_layer = DoRALinear::new(
      dense_base,
      AdapterParams {
        lora_a: la.try_clone().unwrap(),
        lora_b: lb.try_clone().unwrap(),
        magnitude: Some(m.try_clone().unwrap()),
      },
      scale,
    )
    .unwrap();
    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();
    let mut dense_out = dense_layer.forward(&x).unwrap();

    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base = BaseLinear::quantized(
      w_q,
      scales,
      biases,
      Some(bias.try_clone().unwrap()),
      32,
      8,
      "affine".to_string(),
    )
    .unwrap();
    let q_layer = DoRALinear::new(
      q_base,
      AdapterParams {
        lora_a: la,
        lora_b: lb,
        magnitude: Some(m),
      },
      scale,
    )
    .unwrap();
    let mut q_out = q_layer.forward(&x).unwrap();

    approx_eq(
      &q_out.to_vec::<f32>().unwrap(),
      &dense_out.to_vec::<f32>().unwrap(),
      2e-2,
    );
  }

  #[test]
  fn qdora_forward_matches_fuse() {
    // QDoRA forward must equal its own fuse path within quant error — exercises
    // the quantized_matmul base output against the fused (renormalized) weight.
    let input_dims = 64usize;
    let output_dims = 2usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();
    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let m = Array::from_slice::<f32>(&[1.5, 2.5], &(output_dims,)).unwrap();
    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let q_layer = DoRALinear::new(
      q_base,
      AdapterParams {
        lora_a: la,
        lora_b: lb,
        magnitude: Some(m),
      },
      2.0,
    )
    .unwrap();
    let mut via_forward = q_layer.forward(&x).unwrap();
    let fused = q_layer.fuse(true).unwrap();
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      2e-2,
    );
  }

  // ───────── adapters.safetensors hardening ─────────

  #[test]
  fn load_adapters_non_regular_safetensors_is_err() {
    // A directory planted where adapters.safetensors should be is not a regular
    // file ⇒ `adapter_candidate_present` classifies the probe outcome as
    // `CandidateProbe::NonRegular` and surfaces a typed `Error::FileIo` with
    // `ErrorKind::InvalidInput` from the `Stat` op. The structural fix makes
    // a non-regular candidate fail-fast (never falls through to the fallback),
    // so a directory at the preferred slot can NOT be silently masked by an
    // adjacent valid `adapter_model.safetensors` — misconfigurations of the
    // user's adapter directory surface immediately rather than being papered
    // over by reading a different file.
    let tmp = std::env::temp_dir().join(format!("mlxrs_nonreg_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
    std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
    // adapters.safetensors is a DIRECTORY, not a file.
    std::fs::create_dir_all(tmp.join("adapters.safetensors")).unwrap();

    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    match err {
      Error::FileIo(p) => {
        // Fail-fast: the error names the non-regular preferred path with
        // `FileOp::Stat` (the probe), NOT the fallback (which is absent).
        assert_eq!(p.path(), tmp.join("adapters.safetensors").as_path());
        assert_eq!(p.op(), FileOp::Stat);
        assert_eq!(p.inner().kind(), std::io::ErrorKind::InvalidInput);
      }
      other => panic!("expected Error::FileIo(InvalidInput, Stat), got {other:?}"),
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn load_adapters_oversized_safetensors_is_err() {
    // A sparse file reporting a length beyond MAX_ADAPTER_SAFETENSORS_BYTES is
    // rejected on the stat, before any mmap. set_len makes a sparse file on
    // APFS/most filesystems — the on-disk footprint stays ~0.
    let tmp = std::env::temp_dir().join(format!("mlxrs_oversize_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let config = r#"{
      "fine_tune_type": "lora",
      "num_layers": 16,
      "lora_parameters": { "rank": 2, "scale": 2.0, "keys": ["self_attn.q_proj"] }
    }"#;
    std::fs::write(tmp.join("adapter_config.json"), config).unwrap();
    let f = std::fs::File::create(tmp.join("adapters.safetensors")).unwrap();
    f.set_len(MAX_ADAPTER_SAFETENSORS_BYTES + 1).unwrap();
    drop(f);

    let weights = toy_weights();
    let err = load_adapters(&weights, &tmp, None, 4).unwrap_err();
    match err {
      Error::CapExceeded(p) => {
        assert_eq!(p.cap_name(), "MAX_ADAPTER_SAFETENSORS_BYTES");
        assert_eq!(p.cap(), MAX_ADAPTER_SAFETENSORS_BYTES);
        assert_eq!(p.observed(), MAX_ADAPTER_SAFETENSORS_BYTES + 1);
      }
      other => panic!("expected Error::CapExceeded, got {other:?}"),
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  // ═════════════════ HuggingFace PEFT — full surface ═════════════════

  /// A weight map with `n` decoder blocks, each carrying a `self_attn.q_proj`
  /// and a `self_attn.v_proj`, plus a top-level `lm_head.weight`.
  fn peft_toy_weights(n: usize) -> Weights {
    let mut w = Weights::new();
    for b in 0..n {
      w.insert(
        format!("model.layers.{b}.self_attn.q_proj.weight"),
        base_weight(),
      );
      w.insert(
        format!("model.layers.{b}.self_attn.v_proj.weight"),
        base_weight(),
      );
    }
    w.insert("lm_head.weight".to_string(), base_weight());
    w
  }

  // ───────────── PEFT config: fields + defaults ─────────────

  #[test]
  fn peft_config_lora_alpha_defaults_to_8() {
    // PEFT `LoraConfig.lora_alpha` defaults to 8 (NOT the mlx-lm 20.0 literal).
    // A PEFT config omitting `lora_alpha` with `r:16` ⇒ scale 8/16 = 0.5.
    let json = r#"{ "peft_type": "LORA", "r": 16, "target_modules": ["q_proj"] }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.lora_parameters.alpha, Some(DEFAULT_PEFT_LORA_ALPHA));
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 0.5);
  }

  #[test]
  fn peft_config_accepts_and_ignores_training_only_fields() {
    // A real PEFT `LoraConfig` carries training-only / metadata fields with no
    // inference effect on already-saved factors — these BENIGN fields must parse
    // cleanly (accept-and-ignore), not error, even when set to real values.
    // (`layer_replication` / `trainable_token_indices` / `target_parameters` —
    // formerly in this list — are forward/structure-switching and are now
    // rejected by the reject-unknown-active backstop; see
    // `peft_config_structural_reject_examples_*`.)
    let json = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "init_lora_weights": "gaussian",
      "loftq_config": {},
      "eva_config": null,
      "corda_config": null,
      "task_type": "CAUSAL_LM",
      "megatron_config": null,
      "megatron_core": "megatron.core",
      "revision": null,
      "base_model_name_or_path": "meta-llama/Llama-3-8B"
    }"#;
    let cfg = LoraConfig::from_json(json).expect("training-only fields must not error");
    assert_eq!(cfg.rank(), 8);
    assert_eq!(cfg.scale_for("q_proj"), 2.0);
  }

  #[test]
  fn peft_config_lora_bias_true_is_err() {
    // `lora_bias: true` puts a bias on lora_B that PEFT adds in the forward —
    // mlxrs's LoRALinear has no such term, so a silent drop would be wrong
    // inference. It must be a recoverable parse error.
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "lora_bias": true
    }"#;
    assert!(
      LoraConfig::from_json(json).is_err(),
      "`lora_bias: true` must be rejected (no lora_B-bias term in LoRALinear)"
    );
    // `lora_bias: false` (the default) is accepted.
    let ok = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "lora_bias": false }"#;
    assert!(LoraConfig::from_json(ok).is_ok());
  }

  #[test]
  fn peft_config_bias_all_or_lora_only_is_err() {
    // PEFT `bias: "all"` / `"lora_only"` trains+saves `.bias` tensors that PEFT
    // adds in the forward (`utils/save_and_load.py` keeps `"bias" in k`);
    // mlxrs's LoRALinear has no adapted-bias slot, so a non-`"none"` value must
    // be a recoverable parse error (a silent drop would be wrong inference).
    for bias in ["all", "lora_only"] {
      let json = format!(
        r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], "bias": {bias:?} }}"#
      );
      let err =
        LoraConfig::from_json(&json).expect_err(&format!("PEFT `bias: {bias:?}` must be rejected"));
      assert!(
        matches!(err, Error::Parse(_)),
        "expected Error::Parse for `bias: {bias:?}`, got {err:?}"
      );
    }
    // `bias: "none"` (the default) — and no `bias` key at all — are fine.
    let none = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "bias": "none" }"#;
    assert!(LoraConfig::from_json(none).is_ok());
    let absent = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"] }"#;
    assert!(LoraConfig::from_json(absent).is_ok());
  }

  #[test]
  fn peft_config_nonempty_modules_to_save_is_err() {
    // PEFT `modules_to_save` trains+saves full modules alongside the low-rank
    // factors; mlxrs's low-rank loader has no saved-full-module slot, so a
    // non-empty list must be rejected (a silent drop of the full module weights
    // would be wrong inference).
    let json = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "modules_to_save": ["embed_tokens", "lm_head"] }"#;
    let err =
      LoraConfig::from_json(json).expect_err("non-empty `modules_to_save` must be rejected");
    assert!(
      matches!(err, Error::Parse(_)),
      "expected Error::Parse, got {err:?}"
    );
    // An empty `modules_to_save` (or absent) is fine — it ships no full modules.
    let empty = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "modules_to_save": [] }"#;
    assert!(LoraConfig::from_json(empty).is_ok());
  }

  #[test]
  fn peft_key_translation_rejects_sidecar_bias_and_modules_to_save_tensors() {
    // A PEFT-prefixed tensor whose suffix is NOT a low-rank factor is a `.bias`
    // (PEFT `bias != "none"`) or a `modules_to_save` full-module weight — both
    // affect inference, so `translate_peft_keys` must REJECT (naming the key),
    // never silently drop. (Defense-in-depth at the weights file, mirroring the
    // config-level `bias` / `modules_to_save` rejection.)

    // (a) a `.bias` tensor adjacent to a LoRA path (PEFT `bias: "all"` /
    // `"lora_only"` saves `base_model.model.<path>.bias`).
    let bias_key = "base_model.model.model.layers.0.self_attn.q_proj.bias";
    let mut with_bias: HashMap<String, Array> = HashMap::new();
    with_bias.insert(
      "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
      Array::zeros::<f32>(&(2, 3)).unwrap(),
    );
    with_bias.insert(
      "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
      Array::zeros::<f32>(&(4, 2)).unwrap(),
    );
    with_bias.insert(
      bias_key.to_string(),
      Array::zeros::<f32>(&(4usize,)).unwrap(),
    );
    let err = translate_peft_keys(with_bias)
      .expect_err("a PEFT-prefixed `.bias` tensor must be rejected, not silently dropped");
    match err {
      Error::LayerKeyed(ref payload) => {
        assert_eq!(
          payload.layer(),
          bias_key,
          "the rejection must name the dropped key"
        );
        assert!(matches!(payload.inner(), Error::InvariantViolation(_)));
      }
      other => panic!("expected Error::LayerKeyed, got {other:?}"),
    }

    // (b) a `modules_to_save` full-module weight (the `modules_to_save.<adapter>.`
    // prefix is stripped on save → `base_model.model.<module>.weight`).
    let saved_key = "base_model.model.lm_head.weight";
    let mut with_saved: HashMap<String, Array> = HashMap::new();
    with_saved.insert(saved_key.to_string(), base_weight());
    let err = translate_peft_keys(with_saved)
      .expect_err("a PEFT-prefixed `modules_to_save` weight must be rejected");
    let Error::LayerKeyed(payload) = err else {
      panic!("expected LayerKeyed");
    };
    assert_eq!(payload.layer(), saved_key);
    assert!(matches!(payload.inner(), Error::InvariantViolation(_)));
  }

  #[test]
  fn peft_config_exotic_variants_are_rejected() {
    // PEFT's exotic LoRA variants each CHANGE the inference forward — loading
    // such an adapter as plain LoRA would run it at the wrong behavior, so the
    // `Deserialize` must REJECT them loudly (not silently drop, as it does the
    // training-only fields). One non-default exotic field per config, on a
    // normal PEFT-flat config.
    let base = r#""peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"]"#;
    for (field, value) in [
      ("use_qalora", "true"),
      ("alora_invocation_tokens", "[1, 2, 3]"),
      ("velora_config", r#"{"rank": 4}"#),
      ("monteclora_config", r#"{"num_samples": 8}"#),
    ] {
      let json = format!("{{ {base}, {field:?}: {value} }}");
      let err = LoraConfig::from_json(&json).expect_err(&format!(
        "a PEFT adapter setting `{field}` must be rejected (it changes inference)"
      ));
      // The error is a typed `Error::Parse` whose inner serde error's `Display`
      // names the offending field (the `E::custom(...)` rejection string).
      let Error::Parse(p) = &err else {
        panic!("expected Error::Parse for `{field}`, got {err:?}");
      };
      assert_eq!(p.context(), "LoraConfig::from_json");
      let msg = p.inner().to_string();
      assert!(
        msg.contains(field),
        "the rejection error for `{field}` should name the field; got: {msg}"
      );
    }
  }

  #[test]
  fn peft_config_exotic_variant_rejection_is_shape_independent() {
    // The exotic-variant rejection MUST run before the shape-detection branches
    // — it cannot be gated behind the PEFT-shape markers (`peft_type` / `r` /
    // `lora_alpha` / `target_modules`) or the `lora_parameters` early return.
    // An adapter that carries an exotic field but NO PEFT marker, or that uses
    // the mlx-lm-native `lora_parameters` nesting, must still be rejected —
    // otherwise it silently loads as plain/mlx-lm LoRA at the wrong behavior.
    for (label, json) in [
      // (a) exotic field, NO PEFT markers at all (would otherwise fall through
      // to the bare-config default mlx-lm path).
      ("no-marker use_qalora", r#"{ "use_qalora": true }"#),
      (
        "no-marker alora",
        r#"{ "alora_invocation_tokens": [7, 8] }"#,
      ),
      ("no-marker velora", r#"{ "velora_config": {"rank": 2} }"#),
      (
        "no-marker monteclora",
        r#"{ "monteclora_config": {"k": 1} }"#,
      ),
      // (b) exotic field alongside the mlx-lm-native `lora_parameters` nesting
      // (would otherwise hit the early `lora_parameters` return).
      (
        "mlx-lm-shape use_qalora",
        r#"{ "lora_parameters": { "rank": 8 }, "use_qalora": true }"#,
      ),
      (
        "mlx-lm-shape velora",
        r#"{ "fine_tune_type": "lora", "num_layers": 4,
            "lora_parameters": { "rank": 8 }, "velora_config": {"x": 1} }"#,
      ),
    ] {
      assert!(
        LoraConfig::from_json(json).is_err(),
        "exotic-field config {label:?} must be rejected regardless of on-disk shape"
      );
    }
  }

  #[test]
  fn peft_config_exotic_variant_defaults_are_accepted() {
    // The exotic fields at their PEFT DEFAULTS are NOT a signal — a config that
    // carries `use_qalora: false` (the default) or sets the others to `null`
    // parses cleanly as a normal LoRA adapter. `qalora_group_size` (no longer a
    // modeled field — only meaningful with `use_qalora: true`) is left to
    // parse-and-drop, so a stray default `16` is likewise harmless.
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"],
      "use_qalora": false, "qalora_group_size": 16,
      "alora_invocation_tokens": null, "velora_config": null, "monteclora_config": null
    }"#;
    let cfg = LoraConfig::from_json(json)
      .expect("exotic fields at their defaults must not trip the rejection");
    assert_eq!(cfg.rank(), 8);
    assert!(!cfg.is_dora());

    // A bare mlx-lm-native config carrying only the exotic *defaults* is also
    // fine — the shape-independent guard must not false-positive on `null`s.
    let mlx = r#"{ "lora_parameters": { "rank": 4 },
      "use_qalora": false, "velora_config": null, "monteclora_config": null }"#;
    assert!(
      LoraConfig::from_json(mlx).is_ok(),
      "exotic defaults on an mlx-lm-shaped config must not trip the guard"
    );
  }

  // ───────── reject-unknown-active: the structural backstop (PEFT-flat) ─────────

  #[test]
  fn peft_config_arrow_config_is_err() {
    // `arrow_config` switches the forward (PEFT `resolve_lora_variant` returns
    // an `ArrowLinearVariant`); it is NOT a modeled field, so the structural
    // backstop must reject it when set to an object — BEFORE any tensor
    // translation. (Caught generically, no per-field code.)
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"],
      "arrow_config": { "top_k": 3 }
    }"#;
    let err = LoraConfig::from_json(json)
      .expect_err("`arrow_config` set must be rejected (forward variant)");
    let Error::Parse(p) = &err else {
      panic!("expected Error::Parse, got {err:?}");
    };
    let msg = p.inner().to_string();
    assert!(
      msg.contains("arrow_config"),
      "the rejection should name `arrow_config`; got: {msg}"
    );
  }

  #[test]
  fn peft_config_use_bdlora_is_err() {
    // `use_bdlora` switches the forward (PEFT `resolve_lora_variant` returns a
    // `BdLoraLinearVariant`); un-modeled, so the structural backstop rejects an
    // object value.
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"],
      "use_bdlora": { "nblocks": 2 }
    }"#;
    let err = LoraConfig::from_json(json).expect_err("`use_bdlora` set must be rejected");
    let Error::Parse(p) = &err else {
      panic!("expected Error::Parse, got {err:?}");
    };
    let msg = p.inner().to_string();
    assert!(
      msg.contains("use_bdlora"),
      "the rejection should name `use_bdlora`; got: {msg}"
    );
  }

  #[test]
  fn peft_config_invented_unknown_active_field_is_err() {
    // The whole point of the structural posture: a field that does not exist in
    // *today's* PEFT, set to an active value, must be rejected by name with NO
    // code change. Proves the backstop catches genuinely NEW fields (object and
    // scalar forms both).
    for (field, value) in [
      ("some_future_variant", r#"{ "k": 1 }"#),
      ("another_future_knob", "7"),
      ("yet_another_variant", "true"),
      ("a_future_string_variant", r#""enabled""#),
    ] {
      let json = format!(
        r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], {field:?}: {value} }}"#
      );
      let err = LoraConfig::from_json(&json).expect_err(&format!(
        "an active unknown field `{field}` must be rejected by the structural backstop"
      ));
      let Error::Parse(p) = &err else {
        panic!("expected Error::Parse for `{field}`, got {err:?}");
      };
      let msg = p.inner().to_string();
      assert!(
        msg.contains(field),
        "the rejection for `{field}` should name the field; got: {msg}"
      );
    }
  }

  #[test]
  fn peft_config_unknown_field_inactive_value_is_accepted() {
    // PEFT's variant-gating fields default to None (→ JSON null) or False when
    // off. An unknown field set to `null` or `false` is provably the inactive
    // default, so it must be IGNORED (loads fine) — otherwise merely carrying a
    // defaulted future field would spuriously fail.
    for value in ["null", "false"] {
      let json = format!(
        r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], "some_future_variant": {value} }}"#
      );
      let cfg = LoraConfig::from_json(&json).unwrap_or_else(|e| {
        panic!("an inactive (`{value}`) unknown field must be ignored, got: {e:?}")
      });
      assert_eq!(cfg.rank(), 8);
    }
    // Several inactive unknowns at once — still fine.
    let json = r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "future_a": null, "future_b": false, "future_c": null }"#;
    assert!(
      LoraConfig::from_json(json).is_ok(),
      "multiple inactive unknown fields must all be ignored"
    );
  }

  #[test]
  fn peft_config_benign_fields_with_real_values_are_accepted() {
    // BENIGN-IGNORE fields carry metadata / training-only info with no effect on
    // already-saved factors at inference. Set to real (active) values they must
    // still load — they are on the explicit allowlist, not unknown.
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"],
      "task_type": "CAUSAL_LM",
      "revision": "main",
      "base_model_name_or_path": "meta-llama/Llama-3-8B",
      "auto_mapping": { "base_model_class": "LlamaForCausalLM" },
      "inference_mode": true,
      "peft_version": "0.19.2.dev0",
      "megatron_core": "megatron.core",
      "megatron_config": { "tensor_model_parallel_size": 1 },
      "runtime_config": { "ephemeral_gpu_offload": true },
      "eva_config": { "rho": 2.0 },
      "corda_config": { "corda_method": "ipm" },
      "lora_ga_config": { "scale": "stable" },
      "loftq_config": { "loftq_bits": 4 },
      "qalora_group_size": 16,
      "ensure_weight_tying": true
    }"#;
    let cfg = LoraConfig::from_json(json)
      .expect("benign metadata / training-only fields must load even when set");
    assert_eq!(cfg.rank(), 8);
    assert_eq!(cfg.lora_parameters.alpha, Some(16.0));
  }

  #[test]
  fn peft_config_init_lora_weights_allowlist_rejects_non_factor_modes() {
    // `init_lora_weights` is an ALLOWLIST: only the pure factor seeds
    // (gaussian/eva/orthogonal + booleans) load; every other string rejects.
    // The base-weight-MUTATING modes subtract a low-rank residual from
    // `base_layer.weight` at init (peft `lora/layer.py`:
    // olora_init/pissa_init/corda_init/loftq_init/lora_ga_init), so a RAW
    // checkpoint saved with one pairs its factors with a modified base —
    // applying them to the unmodified base is silently wrong. `pissa_niter_<N>`
    // and prefixed `corda*` (PEFT dispatches BOTH via `startswith`, layer.py
    // :225/:228) must reject, AND an unknown/future mode must reject by default
    // — the allowlist's whole point, since a reject-list missed `corda_v1`. The
    // message names the offending mode (actionable); matching is
    // case-insensitive.
    for mode in [
      "pissa",
      "pissa_niter_4",
      "PISSA_NITER_16",
      "olora",
      "corda",
      "corda_v1",
      "lora_ga",
      "loftq",
      "some_future_init_mode",
    ] {
      let json = format!(
        r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], "init_lora_weights": "{mode}" }}"#
      );
      match LoraConfig::from_json(&json) {
        Err(Error::Parse(p)) => {
          let msg = p.inner().to_string();
          assert!(
            msg.contains(mode),
            "the rejection should name the mode `{mode}`; got: {msg}"
          );
        }
        Ok(_) => {
          panic!("`init_lora_weights: \"{mode}\"` must be rejected (mutates base weight at init)")
        }
        Err(other) => panic!("expected Error::Parse for `{mode}`, got {other:?}"),
      }
    }
    // The pure factor SEEDS only seed the LoRA factors (or are overwritten at
    // load) and leave the base untouched, so they must still load. PEFT's
    // conversion path also rewrites converted adapters to `true`.
    for init in ["\"gaussian\"", "\"eva\"", "\"orthogonal\"", "true", "false"] {
      let json = format!(
        r#"{{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
           "target_modules": ["q_proj"], "init_lora_weights": {init} }}"#
      );
      assert!(
        LoraConfig::from_json(&json).is_ok(),
        "`init_lora_weights: {init}` is a pure factor seed and must load"
      );
    }
  }

  #[test]
  fn peft_config_structural_reject_examples_layer_replication_and_token_indices() {
    // These forward/structure-switching fields are deliberately NOT on the
    // benign allowlist, so the structural backstop rejects them when active —
    // even though there is no per-field check for them.
    let cases = [
      (
        "layer_replication",
        r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
          "target_modules": ["q_proj"], "layer_replication": [[0, 4], [2, 5]] }"#,
      ),
      (
        "trainable_token_indices",
        r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
          "target_modules": ["q_proj"], "trainable_token_indices": [0, 1, 2] }"#,
      ),
      (
        "target_parameters",
        r#"{ "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
          "target_modules": [], "target_parameters": ["feed_forward.experts.gate_up_proj"] }"#,
      ),
    ];
    for (field, json) in cases {
      match LoraConfig::from_json(json) {
        Ok(_) => panic!("`{field}` set must be rejected by the structural backstop"),
        Err(Error::Parse(p)) => {
          let msg = p.inner().to_string();
          assert!(
            msg.contains(field),
            "the rejection should name `{field}`; got: {msg}"
          );
        }
        Err(other) => panic!("expected Error::Parse for `{field}`, got {other:?}"),
      }
    }
  }

  #[test]
  fn peft_config_valid_flat_fixture_still_loads() {
    // Regression: a realistic, fully-populated PEFT-flat config — exactly the
    // shape `LoraConfig.save_pretrained` writes, where EVERY field is serialized
    // including the forward-switching ones at their inactive (`null` / `false`)
    // defaults — must still load after the structural rule. The reject-if-active
    // fields below (`layer_replication`, `trainable_token_indices`,
    // `target_parameters`, `use_bdlora`, `arrow_config`, …) are present but
    // inactive, so the backstop must ignore them; the rule must not regress this
    // common case.
    let json = r#"{
      "peft_type": "LORA",
      "task_type": "CAUSAL_LM",
      "auto_mapping": null,
      "peft_version": "0.19.2.dev0",
      "base_model_name_or_path": "meta-llama/Llama-3-8B",
      "revision": null,
      "inference_mode": true,
      "r": 16,
      "lora_alpha": 32.0,
      "lora_dropout": 0.05,
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
      "exclude_modules": null,
      "bias": "none",
      "use_rslora": false,
      "use_dora": false,
      "fan_in_fan_out": false,
      "lora_bias": false,
      "modules_to_save": null,
      "init_lora_weights": true,
      "layers_to_transform": null,
      "layers_pattern": null,
      "rank_pattern": {},
      "alpha_pattern": {},
      "megatron_config": null,
      "megatron_core": "megatron.core",
      "use_qalora": false,
      "qalora_group_size": 16,
      "alora_invocation_tokens": null,
      "loftq_config": {},
      "eva_config": null,
      "corda_config": null,
      "lora_ga_config": null,
      "velora_config": null,
      "monteclora_config": null,
      "layer_replication": null,
      "trainable_token_indices": null,
      "target_parameters": null,
      "use_bdlora": null,
      "arrow_config": null,
      "ensure_weight_tying": false,
      "runtime_config": {"ephemeral_gpu_offload": false}
    }"#;
    let cfg = LoraConfig::from_json(json).expect("a realistic PEFT-flat config must still load");
    assert_eq!(cfg.rank(), 16);
    assert_eq!(cfg.lora_parameters.alpha, Some(32.0));
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 2.0); // 32/16
    let peft = cfg.peft().expect("PEFT selection");
    assert!(matches!(&peft.target_modules, Some(ModuleMatcher::List(_))));
  }

  #[test]
  fn mlx_lm_native_fixture_still_loads_with_unknown_keys() {
    // Regression + scope: the mlx-lm-NATIVE nested shape keeps its existing
    // accept-and-ignore behavior — the reject-unknown-active rule applies to the
    // PEFT-flat branch ONLY. An mlx-lm-native config (the `lora_parameters`
    // early return) with an extra unknown key still loads.
    let json = r#"{
      "fine_tune_type": "lora",
      "num_layers": 8,
      "lora_parameters": { "rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["q_proj"] },
      "some_native_extra_key": { "whatever": 1 }
    }"#;
    let cfg = LoraConfig::from_json(json)
      .expect("mlx-lm-native shape must keep accept-and-ignore for unknown keys");
    assert_eq!(cfg.rank(), 8);
    assert!(matches!(
      cfg.selection,
      AdapterSelection::MlxLm { num_layers: 8 }
    ));
  }

  #[test]
  fn peft_key_translation_embedding_lora_precise_reject() {
    // PEFT embedding-LoRA saves `lora_embedding_A` / `lora_embedding_B` factors
    // (`adapter_layer_names`, `lora/layer.py:105`). These ARE legitimate
    // low-rank factors, NOT a bias / modules_to_save tensor — so the translation
    // must reject them with a PRECISE "embedding" message, not the generic
    // bias/modules_to_save one (which would misclassify them). Embedding-LoRA
    // application is deferred, so reject (don't load) — but correctly named.
    for suffix in [
      ".lora_embedding_A",
      ".lora_embedding_B",
      ".lora_embedding_A.weight",
      ".lora_embedding_B.weight",
    ] {
      let key = format!("base_model.model.model.embed_tokens{suffix}");
      let mut arrays: HashMap<String, Array> = HashMap::new();
      arrays.insert(key.clone(), Array::zeros::<f32>(&(2, 3)).unwrap());
      match translate_peft_keys(arrays) {
        Ok(_) => panic!("embedding-LoRA key {key:?} must be rejected, not accepted"),
        Err(Error::LayerKeyed(p)) => {
          assert_eq!(p.layer(), key, "LayerKeyed must name the offending key");
          let Error::InvariantViolation(iv) = p.inner() else {
            panic!(
              "expected inner Error::InvariantViolation, got {:?}",
              p.inner()
            );
          };
          assert!(
            iv.requirement().to_lowercase().contains("embedding"),
            "the rejection requirement must mention embedding; got: {}",
            iv.requirement()
          );
          assert!(
            !iv.requirement().contains("bias") && !iv.requirement().contains("modules_to_save"),
            "embedding-LoRA must NOT be misclassified as bias/modules_to_save; got: {}",
            iv.requirement()
          );
        }
        Err(other) => panic!("expected Error::LayerKeyed, got {other:?}"),
      }
    }
  }

  #[test]
  fn peft_config_all_selection_fields_parse() {
    // Every inference-affecting PEFT selection field on one config.
    let json = r#"{
      "peft_type": "LORA",
      "r": 8,
      "lora_alpha": 16.0,
      "target_modules": ["q_proj", "v_proj"],
      "exclude_modules": ["lm_head"],
      "use_rslora": true,
      "use_dora": false,
      "fan_in_fan_out": true,
      "layers_to_transform": [0, 2, 4],
      "layers_pattern": "layers",
      "rank_pattern": { "q_proj": 16 },
      "alpha_pattern": { "v_proj": 64 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let peft = cfg.peft().unwrap();
    assert!(matches!(&peft.target_modules, Some(ModuleMatcher::List(_))));
    assert!(matches!(
      &peft.exclude_modules,
      Some(ModuleMatcher::List(_))
    ));
    assert!(peft.use_rslora);
    assert!(peft.fan_in_fan_out);
    assert_eq!(peft.layers_to_transform.as_deref(), Some(&[0, 2, 4][..]));
    assert_eq!(peft.layers_pattern, vec!["layers".to_string()]);
    assert!(cfg.fan_in_fan_out());
  }

  // ───────────── PEFT scale: rsLoRA + rank/alpha patterns ─────────────

  #[test]
  fn peft_rslora_scale_is_alpha_over_sqrt_r() {
    // use_rslora=true ⇒ scale = lora_alpha / sqrt(r). r=16, alpha=32 ⇒
    // 32/sqrt(16) = 32/4 = 8.0. Non-rsLoRA would be 32/16 = 2.0.
    let json = r#"{
      "peft_type": "LORA", "r": 16, "lora_alpha": 32.0,
      "target_modules": ["q_proj"], "use_rslora": true
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 8.0);
  }

  #[test]
  fn peft_non_rslora_scale_is_alpha_over_r() {
    // use_rslora absent ⇒ scale = lora_alpha / r = 32/16 = 2.0.
    let json = r#"{
      "peft_type": "LORA", "r": 16, "lora_alpha": 32.0,
      "target_modules": ["q_proj"]
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert!(!cfg.peft().unwrap().use_rslora);
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 2.0);
  }

  #[test]
  fn peft_rank_pattern_overrides_rank_per_module() {
    // `rank_pattern: {"q_proj": 32}` ⇒ a q_proj module resolves rank 32; a
    // v_proj module (no pattern) keeps the config-wide `r:8`.
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj", "v_proj"],
      "rank_pattern": { "q_proj": 32 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.rank_for("model.layers.0.self_attn.q_proj"), 32);
    assert_eq!(cfg.rank_for("model.layers.0.self_attn.v_proj"), 8);
    // The scale follows the overridden rank: q_proj is 16/32 = 0.5; v_proj
    // is the config-wide 16/8 = 2.0.
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 0.5);
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.v_proj"), 2.0);
  }

  #[test]
  fn peft_alpha_pattern_overrides_alpha_per_module() {
    // `alpha_pattern: {"q_proj": 64}` ⇒ a q_proj module scales by 64/r; a
    // v_proj module keeps the config-wide `lora_alpha:16`.
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj", "v_proj"],
      "alpha_pattern": { "q_proj": 64 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    // q_proj: alpha 64 / r 8 = 8.0; v_proj: alpha 16 / r 8 = 2.0.
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 8.0);
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.v_proj"), 2.0);
  }

  #[test]
  fn peft_rank_and_alpha_pattern_with_rslora() {
    // rank_pattern + alpha_pattern + rsLoRA compose: q_proj resolves rank 16,
    // alpha 64 ⇒ rsLoRA scale 64/sqrt(16) = 64/4 = 16.0.
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"], "use_rslora": true,
      "rank_pattern": { "q_proj": 16 }, "alpha_pattern": { "q_proj": 64 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 16.0);
  }

  #[test]
  fn peft_pattern_lookup_anchors_at_segment_boundary() {
    // PEFT `get_pattern_key` is `re.match(rf"(.*\.)?({key})$", module)` — the
    // pattern key matches a dotted suffix, NOT a mid-string substring.
    let patterns = vec![("q_proj".to_string(), 99i32)];
    assert_eq!(
      pattern_lookup(&patterns, "model.layers.0.self_attn.q_proj"),
      Some(99)
    );
    assert_eq!(pattern_lookup(&patterns, "q_proj"), Some(99));
    // a substring `xq_proj` must NOT match (the `(.*\.)?` needs a dot).
    assert_eq!(pattern_lookup(&patterns, "model.xq_proj"), None);
    // no match ⇒ None (caller falls back to the default).
    assert_eq!(pattern_lookup(&patterns, "model.layers.0.mlp.down"), None);
  }

  #[test]
  fn peft_pattern_lookup_regex_key() {
    // PEFT pattern keys are themselves regex fragments — a `layers.0.*q_proj`
    // pattern keys block 0 only.
    let patterns = vec![("layers\\.0\\..*q_proj".to_string(), 64i32)];
    assert_eq!(
      pattern_lookup(&patterns, "model.layers.0.self_attn.q_proj"),
      Some(64)
    );
    assert_eq!(
      pattern_lookup(&patterns, "model.layers.1.self_attn.q_proj"),
      None
    );
  }

  #[test]
  fn peft_rank_pattern_resolves_in_json_insertion_order_not_sorted() {
    // PEFT `get_pattern_key` returns the FIRST dict key (in insertion order)
    // whose `re.match(rf"(.*\.)?({key})$", module)` matches. For OVERLAPPING
    // pattern keys this tie-break is the JSON order — NOT a lexicographic sort.
    // Both keys below match `…self_attn.q_proj`, but `".*\.q_proj"` sorts BEFORE
    // `"self_attn.q_proj"` lexicographically ('.' 0x2E < 's' 0x73). With the
    // keys written `self_attn.q_proj` FIRST, insertion order must win (rank 11),
    // proving the resolver preserves JSON order rather than sorting (which would
    // wrongly pick 22).
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "rank_pattern": { "self_attn.q_proj": 11, ".*\\.q_proj": 22 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert_eq!(
      cfg.rank_for("model.layers.0.self_attn.q_proj"),
      11,
      "first-in-JSON-order key must win (a lexicographic sort would pick 22)"
    );

    // Reversing the JSON order flips the winner — confirming order, not value
    // or specificity, is the tie-break.
    let reversed = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "rank_pattern": { ".*\\.q_proj": 22, "self_attn.q_proj": 11 }
    }"#;
    let cfg2 = LoraConfig::from_json(reversed).unwrap();
    assert_eq!(
      cfg2.rank_for("model.layers.0.self_attn.q_proj"),
      22,
      "with the order reversed the other key wins — pure insertion-order tie-break"
    );
  }

  #[test]
  fn peft_alpha_pattern_resolves_in_json_insertion_order_not_sorted() {
    // Same insertion-order tie-break for `alpha_pattern`. Two overlapping keys;
    // `".*\.q_proj"` sorts first lexicographically, but `q_proj` is written
    // first, so its alpha (40) wins over the other (80) — a sort would pick 80.
    let json = r#"{
      "peft_type": "LORA", "r": 8, "lora_alpha": 16.0,
      "target_modules": ["q_proj"],
      "alpha_pattern": { "q_proj": 40, ".*\\.q_proj": 80 }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    // scale = alpha / r = 40 / 8 = 5.0 (NOT 80/8 = 10.0).
    assert_eq!(cfg.scale_for("model.layers.0.self_attn.q_proj"), 5.0);
  }

  // ───────────── PEFT selection: target / exclude / layers ─────────────

  #[test]
  fn peft_select_target_modules_list() {
    // PEFT `target_modules` list: every block's q_proj wraps (NO num_layers
    // window — the historical bug), v_proj does not.
    let weights = peft_toy_weights(4);
    let mut params = HashMap::new();
    for b in 0..4 {
      params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    }
    let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["q_proj"] }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    assert!(layers.contains_key("model.layers.3.self_attn.q_proj"));
    assert!(!layers.contains_key("model.layers.0.self_attn.v_proj"));
  }

  #[test]
  fn peft_select_target_modules_regex() {
    // PEFT `target_modules` as a regex string — `re.fullmatch` over the whole
    // module path. `.*self_attn\.q_proj` matches only the q_proj paths.
    let weights = peft_toy_weights(3);
    let mut params = HashMap::new();
    for b in 0..3 {
      params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    }
    let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ".*self_attn\\.q_proj" }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 3).unwrap();
    assert_eq!(layers.len(), 3);
    assert!(layers.contains_key("model.layers.2.self_attn.q_proj"));
    assert!(!layers.contains_key("model.layers.0.self_attn.v_proj"));
  }

  #[test]
  fn peft_select_exclude_modules_list() {
    // PEFT `exclude_modules` removes a target match. target=regex matching
    // both q and v proj; exclude=["v_proj"] ⇒ only q_proj wraps.
    let weights = peft_toy_weights(2);
    let mut params = HashMap::new();
    for b in 0..2 {
      params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    }
    let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ".*_proj", "exclude_modules": ["v_proj"] }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 2).unwrap();
    assert_eq!(layers.len(), 2);
    assert!(layers.contains_key("model.layers.0.self_attn.q_proj"));
    assert!(!layers.contains_key("model.layers.0.self_attn.v_proj"));
  }

  #[test]
  fn peft_select_exclude_modules_regex() {
    // `exclude_modules` as a regex (`re.fullmatch`): exclude every v_proj.
    let weights = peft_toy_weights(2);
    let mut params = HashMap::new();
    for b in 0..2 {
      params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    }
    let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ".*_proj", "exclude_modules": ".*\\.v_proj" }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 2).unwrap();
    assert_eq!(layers.len(), 2);
    assert!(!layers.contains_key("model.layers.1.self_attn.v_proj"));
  }

  #[test]
  fn peft_select_layers_to_transform_int() {
    // `layers_to_transform: 1` (a bare int) ⇒ only block 1's q_proj wraps.
    let weights = peft_toy_weights(4);
    let mut params = HashMap::new();
    params.insert(
      "model.layers.1.self_attn.q_proj".to_string(),
      plain_params(),
    );
    let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["q_proj"], "layers_to_transform": 1 }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    assert_eq!(layers.len(), 1);
    assert!(layers.contains_key("model.layers.1.self_attn.q_proj"));
  }

  #[test]
  fn peft_select_layers_to_transform_list() {
    // `layers_to_transform: [0, 3]` ⇒ only blocks 0 and 3 wrap.
    let weights = peft_toy_weights(5);
    let mut params = HashMap::new();
    for b in [0, 3] {
      params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    }
    let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["q_proj"], "layers_to_transform": [0, 3] }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 5).unwrap();
    assert_eq!(layers.len(), 2);
    assert!(layers.contains_key("model.layers.0.self_attn.q_proj"));
    assert!(layers.contains_key("model.layers.3.self_attn.q_proj"));
    assert!(!layers.contains_key("model.layers.1.self_attn.q_proj"));
  }

  #[test]
  fn peft_select_layers_pattern_custom_attr() {
    // `layers_pattern: "h"` extracts the block index after a `.h.` attribute
    // (GPT-2-style `transformer.h.0.…`) instead of `.layers.`.
    let mut weights = Weights::new();
    for b in 0..3 {
      weights.insert(
        format!("transformer.h.{b}.attn.c_attn.weight"),
        base_weight(),
      );
    }
    let mut params = HashMap::new();
    params.insert("transformer.h.2.attn.c_attn".to_string(), plain_params());
    let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["c_attn"], "layers_to_transform": [2],
      "layers_pattern": "h" }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 3).unwrap();
    assert_eq!(layers.len(), 1);
    assert!(layers.contains_key("transformer.h.2.attn.c_attn"));
  }

  #[test]
  fn peft_select_no_restriction_adapts_all_blocks_over_16() {
    // A PEFT config with no `layers_to_transform` must adapt
    // EVERY matching block — including blocks 16..19 on a 20-block model.
    // Applying mlx-lm's `num_layers=16` trailing window here would be
    // wrong: it would drop blocks 0..3.
    let weights = peft_toy_weights(20);
    let mut params = HashMap::new();
    for b in 0..20 {
      params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
    }
    let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["q_proj"] }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 20).unwrap();
    assert_eq!(layers.len(), 20, "PEFT must adapt ALL 20 blocks, no window");
    // Block 0 (which a trailing-16 window would drop) IS adapted.
    assert!(layers.contains_key("model.layers.0.self_attn.q_proj"));
    assert!(layers.contains_key("model.layers.19.self_attn.q_proj"));
  }

  #[test]
  fn peft_target_modules_all_linear_string_is_sentinel_not_regex() {
    // PEFT's `"all-linear"` string is a SENTINEL (expand to all linears minus
    // the output head), NOT a regex. The literal string compiles as a regex
    // that full-matches only "all-linear" (i.e. nothing), so a regex read would
    // select nothing — `all-linear` must instead select all rank-2 linears.
    let json = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": "all-linear" }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    let peft = match &cfg.selection {
      AdapterSelection::Peft(p) => p,
      other => panic!("expected a PEFT selection, got {other:?}"),
    };
    assert!(
      matches!(peft.target_modules, Some(ModuleMatcher::AllLinear)),
      "the `all-linear` string must parse to the AllLinear sentinel, not a regex"
    );

    // 3 blocks of q_proj + v_proj (all rank-2) plus a top-level `lm_head` (also
    // rank-2). `all-linear` selects every rank-2 linear EXCEPT the output head;
    // the `lm_head` weight is in the map but must NOT be adapted. (Factors are
    // shipped only for the q/v linears — `all-linear` is auto-discovery, so a
    // discovered-but-untrained linear is simply skipped, and a non-selected
    // `lm_head` with no factors is correctly never touched.)
    let weights = peft_toy_weights(3);
    let mut params = HashMap::new();
    for b in 0..3 {
      params.insert(format!("model.layers.{b}.self_attn.q_proj"), plain_params());
      params.insert(format!("model.layers.{b}.self_attn.v_proj"), plain_params());
    }

    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 3).unwrap();
    // 3 q_proj + 3 v_proj = 6; lm_head excluded by the head filter.
    assert_eq!(
      layers.len(),
      6,
      "all-linear adapts every linear minus the head"
    );
    for b in 0..3 {
      assert!(layers.contains_key(&format!("model.layers.{b}.self_attn.q_proj")));
      assert!(layers.contains_key(&format!("model.layers.{b}.self_attn.v_proj")));
    }
    assert!(
      !layers.contains_key("lm_head"),
      "all-linear must EXCLUDE the output head (lm_head)"
    );
    // The unit-level selector check (with lm_head factors present) lives in
    // `peft_target_modules_all_linear_excludes_head_and_non_rank2` — it proves
    // the head is excluded by the *selector*, not merely by missing factors.
  }

  #[test]
  fn peft_target_modules_all_linear_is_case_insensitive() {
    // PEFT lowercases `target_modules` before the sentinel compare
    // (`target_modules.lower() == "all-linear"`).
    for s in ["All-Linear", "ALL-LINEAR"] {
      let json = format!(r#"{{ "peft_type": "LORA", "r": 2, "target_modules": {s:?} }}"#);
      let cfg = LoraConfig::from_json(&json).unwrap();
      assert!(
        matches!(
          &cfg.selection,
          AdapterSelection::Peft(p) if matches!(p.target_modules, Some(ModuleMatcher::AllLinear))
        ),
        "`{s}` must be recognized as the all-linear sentinel (case-insensitive)"
      );
    }
  }

  #[test]
  fn peft_target_modules_all_linear_excludes_head_and_non_rank2() {
    // The AllLinear selector applies BOTH halves of the predicate: rank-2
    // ("is a linear") AND not-the-output-head. A rank-1 weight (e.g. a norm
    // gain) and the `lm_head` are both excluded; a normal rank-2 linear is in.
    let q_w = base_weight(); // rank-2
    let norm_w = Array::zeros::<f32>(&(8usize,)).unwrap(); // rank-1
    let head_w = base_weight(); // rank-2 but it IS the head
    let peft = PeftSelection {
      target_modules: Some(ModuleMatcher::AllLinear),
      exclude_modules: None,
      layers_to_transform: None,
      layers_pattern: Vec::new(),
      rank_pattern: Vec::new(),
      alpha_pattern: Vec::new(),
      use_rslora: false,
      fan_in_fan_out: false,
    };
    assert!(peft_module_is_selected(
      "model.layers.0.self_attn.q_proj",
      &q_w,
      &peft
    ));
    assert!(
      !peft_module_is_selected("model.layers.0.input_layernorm", &norm_w, &peft),
      "a rank-1 weight is not a linear — all-linear must skip it"
    );
    assert!(
      !peft_module_is_selected("lm_head", &head_w, &peft),
      "the output head is excluded by all-linear even though it is rank-2"
    );
    // A nested `lm_head` (e.g. `model.lm_head`) is also the head.
    assert!(!peft_module_is_selected("model.lm_head", &head_w, &peft));
  }

  // ───────────── ModuleMatcher / peft_layer_index units ─────────────

  #[test]
  fn module_matcher_list_is_exact_or_dotted_suffix() {
    let m = ModuleMatcher::List(vec!["q_proj".to_string()]);
    assert!(m.matches("model.layers.0.self_attn.q_proj"));
    assert!(m.matches("q_proj"));
    // a substring without a dot boundary must NOT match.
    assert!(!m.matches("model.xq_proj"));
    assert!(!m.matches("q_proj_extra"));
  }

  #[test]
  fn module_matcher_regex_is_full_match() {
    let m = ModuleMatcher::Regex(Box::new(Regex::new(r".*\.q_proj").unwrap()));
    assert!(m.matches("model.layers.0.self_attn.q_proj"));
    // `re.fullmatch` — a trailing extra segment must NOT match (the `.*\.q_proj`
    // pattern cannot consume the trailing `.bias`).
    assert!(!m.matches("model.layers.0.self_attn.q_proj.bias"));
    // A regex anchored to a specific suffix only — `re.fullmatch` requires the
    // WHOLE key to match, so a key with extra leading content is rejected (a
    // `search`-style match would wrongly accept it).
    let suffix = ModuleMatcher::Regex(Box::new(Regex::new(r"q_proj").unwrap()));
    assert!(suffix.matches("q_proj"));
    assert!(!suffix.matches("model.layers.0.self_attn.q_proj"));
  }

  #[test]
  fn peft_layer_index_default_and_custom_pattern() {
    // default pattern: digits between dots after a prefix.
    assert_eq!(
      peft_layer_index("model.layers.7.self_attn.q_proj", &[]),
      Some(7)
    );
    // custom attribute name.
    assert_eq!(
      peft_layer_index("transformer.h.3.attn.c_attn", &["h".to_string()]),
      Some(3)
    );
    // no extractable index ⇒ None.
    assert_eq!(peft_layer_index("lm_head", &[]), None);
  }

  // ───────────── PEFT weight-key translation ─────────────

  #[test]
  fn peft_key_translation_strips_prefix_maps_suffix_transposes() {
    // `base_model.model.<path>.lora_A.weight` → `<path>.lora_a`, transposed
    // ([r,in] → [in,r]); `lora_B.weight` → `.lora_b` ([out,r] → [r,out]);
    // `lora_magnitude_vector` → `.m` (no transpose).
    let mut raw: HashMap<String, Array> = HashMap::new();
    // PEFT lora_A: [r=2, in=3]; lora_B: [out=4, r=2]; magnitude: [out=4].
    raw.insert(
      "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
      Array::zeros::<f32>(&(2, 3)).unwrap(),
    );
    raw.insert(
      "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
      Array::zeros::<f32>(&(4, 2)).unwrap(),
    );
    raw.insert(
      "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector".to_string(),
      Array::zeros::<f32>(&(4usize,)).unwrap(),
    );
    // A non-PEFT key (no `base_model.model.` prefix) is dropped.
    raw.insert("some.stray.weight".to_string(), base_weight());

    let out = translate_peft_keys(raw).unwrap();
    assert_eq!(out.len(), 3, "3 LoRA tensors, the stray key dropped");
    let path = "model.layers.0.self_attn.q_proj";
    // lora_a: PEFT [2,3] transposed → [3,2].
    assert_eq!(out[&format!("{path}.lora_a")].shape(), &[3, 2]);
    // lora_b: PEFT [4,2] transposed → [2,4].
    assert_eq!(out[&format!("{path}.lora_b")].shape(), &[2, 4]);
    // m: PEFT [4] unchanged.
    assert_eq!(out[&format!("{path}.m")].shape(), &[4]);
  }

  #[test]
  fn peft_key_translation_magnitude_vector_dot_weight_variant() {
    // PEFT may store the DoRA magnitude as `lora_magnitude_vector.weight`
    // (the in-memory `ModuleDict` form) — both spellings map to `.m`.
    let mut raw: HashMap<String, Array> = HashMap::new();
    raw.insert(
      "base_model.model.q_proj.lora_magnitude_vector.weight".to_string(),
      Array::zeros::<f32>(&(2usize,)).unwrap(),
    );
    let out = translate_peft_keys(raw).unwrap();
    assert!(out.contains_key("q_proj.m"));
  }

  // ───────────── PEFT end-to-end ─────────────

  #[test]
  fn peft_end_to_end_rslora_scale_and_all_blocks() {
    // A real PEFT adapter dir (config + adapter_model.safetensors): rsLoRA
    // scale, all matching blocks adapted.
    let tmp = std::env::temp_dir().join(format!("mlxrs_peft_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "peft_type": "LORA", "r": 16, "lora_alpha": 32.0,
      "target_modules": ["self_attn.q_proj"], "use_rslora": true
    }"#;
    let q_paths: Vec<String> = (0..4)
      .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
      .collect();
    let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
    write_mock_peft_adapter(&tmp, cfg, &q_refs, 16, false, 0.01);
    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    // rsLoRA scale = lora_alpha / sqrt(r) = 32 / 4 = 8.0.
    if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.0.self_attn.q_proj") {
      assert_eq!(l.scale(), 8.0, "rsLoRA scale must be alpha/sqrt(r)");
    } else {
      panic!("expected a LoRA layer");
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn peft_end_to_end_dora_with_magnitude_vector() {
    // A PEFT DoRA adapter: `use_dora: true` + a `lora_magnitude_vector` tensor
    // per module. The DoRA layer must build (the magnitude is loaded from the
    // PEFT-keyed safetensors).
    let tmp = std::env::temp_dir().join(format!("mlxrs_peft_dora_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["self_attn.q_proj"], "use_dora": true
    }"#;
    let q_paths: Vec<String> = (0..4)
      .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
      .collect();
    let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
    write_mock_peft_adapter(&tmp, cfg, &q_refs, 2, true, 0.01);
    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    assert!(matches!(
      layers.get("model.layers.0.self_attn.q_proj"),
      Some(LoraLayer::Dora(_))
    ));
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn peft_end_to_end_rank_pattern_per_module_scale() {
    // A PEFT adapter where `rank_pattern` overrides one block's rank. Block 0
    // gets rank 4 (via the pattern, factors shipped at rank 4); blocks 1..3
    // get the config-wide rank 2. Each module's scale follows its rank.
    let tmp = std::env::temp_dir().join(format!("mlxrs_peft_rankpat_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "peft_type": "LORA", "r": 2, "lora_alpha": 8.0,
      "target_modules": ["self_attn.q_proj"],
      "rank_pattern": { "layers\\.0\\..*q_proj": 4 }
    }"#;
    std::fs::write(tmp.join("adapter_config.json"), cfg).unwrap();
    // Block 0: rank-4 PEFT factors; blocks 1..3: rank-2.
    let mut arrays: HashMap<String, Array> = HashMap::new();
    for b in 0..4 {
      let r = if b == 0 { 4 } else { 2 };
      let path = format!("model.layers.{b}.self_attn.q_proj");
      arrays.insert(
        format!("base_model.model.{path}.lora_A.weight"),
        Array::full::<f32>(&(r, 3usize), 0.01).unwrap(),
      );
      arrays.insert(
        format!("base_model.model.{path}.lora_B.weight"),
        Array::full::<f32>(&(2usize, r), 0.01).unwrap(),
      );
    }
    crate::io::save_safetensors(&tmp.join("adapter_model.safetensors"), &arrays).unwrap();

    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    // Block 0: alpha 8 / rank 4 = 2.0 (the rank_pattern override).
    if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.0.self_attn.q_proj") {
      assert_eq!(l.scale(), 2.0, "rank_pattern block-0 scale = alpha/4");
    } else {
      panic!("expected a LoRA layer at block 0");
    }
    // Block 1: alpha 8 / rank 2 = 4.0 (the config-wide rank).
    if let Some(LoraLayer::Lora(l)) = layers.get("model.layers.1.self_attn.q_proj") {
      assert_eq!(l.scale(), 4.0, "default-rank block-1 scale = alpha/2");
    } else {
      panic!("expected a LoRA layer at block 1");
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn peft_end_to_end_exclude_modules() {
    // A PEFT adapter targeting `.*_proj` but excluding v_proj — the v_proj
    // base layers must NOT be adapted (and the adapter ships no v_proj
    // factors, so a wrong selection would also trip the completeness check).
    let tmp = std::env::temp_dir().join(format!("mlxrs_peft_excl_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{
      "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ".*_proj", "exclude_modules": ".*\\.v_proj"
    }"#;
    let weights = peft_toy_weights(3);
    let q_paths: Vec<String> = (0..3)
      .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
      .collect();
    let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
    write_mock_peft_adapter(&tmp, cfg, &q_refs, 2, false, 0.01);
    let layers = load_adapters(&weights, &tmp, None, 3).unwrap();
    assert_eq!(layers.len(), 3);
    for b in 0..3 {
      assert!(layers.contains_key(&format!("model.layers.{b}.self_attn.q_proj")));
      assert!(!layers.contains_key(&format!("model.layers.{b}.self_attn.v_proj")));
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  // ───────────── fan_in_fan_out ─────────────

  #[test]
  fn peft_fan_in_fan_out_transposes_base_weight() {
    // With `fan_in_fan_out: true` the base weight is stored `[in, out]`.
    // `build_base_linear` transposes it back to `[out, in]` so the LoRA
    // forward matches the same adapter applied to a standard `[out, in]` base.
    //
    // Standard base: W = [[1,0,0],[0,1,0]] ([out=2, in=3]).
    // fan_in_fan_out base: Wᵀ = [[1,0],[0,1],[0,0]] ([in=3, out=2]).
    let standard_w = base_weight();
    let fifo_w = standard_w.transpose().unwrap(); // [3, 2] — the [in, out] layout
    let mut std_weights = Weights::new();
    std_weights.insert(
      "model.layers.0.self_attn.q_proj.weight".to_string(),
      standard_w,
    );
    let mut fifo_weights = Weights::new();
    fifo_weights.insert("model.layers.0.self_attn.q_proj.weight".to_string(), fifo_w);

    let mut params = HashMap::new();
    params.insert(
      "model.layers.0.self_attn.q_proj".to_string(),
      plain_params(),
    );

    let std_cfg = LoraConfig::from_json(
      r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
        "target_modules": ["q_proj"], "fan_in_fan_out": false }"#,
    )
    .unwrap();
    let fifo_cfg = LoraConfig::from_json(
      r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
        "target_modules": ["q_proj"], "fan_in_fan_out": true }"#,
    )
    .unwrap();

    let std_layers = linear_to_lora_layers(&std_weights, &std_cfg, &params, None, 1).unwrap();
    let fifo_layers = linear_to_lora_layers(&fifo_weights, &fifo_cfg, &params, None, 1).unwrap();

    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut std_out = std_layers["model.layers.0.self_attn.q_proj"]
      .forward(&x)
      .unwrap();
    let mut fifo_out = fifo_layers["model.layers.0.self_attn.q_proj"]
      .forward(&x)
      .unwrap();
    // The fan_in_fan_out base, after the transpose, must give the SAME forward
    // as the standard base.
    approx_eq(
      &fifo_out.to_vec::<f32>().unwrap(),
      &std_out.to_vec::<f32>().unwrap(),
      1e-5,
    );
  }

  #[test]
  fn peft_fan_in_fan_out_quantized_is_err() {
    // `fan_in_fan_out` over a quantized base is rejected — transposing a
    // packed quantized weight would corrupt the bit-packing.
    let weight = Array::zeros::<u32>(&(8, 4)).unwrap();
    let scales = Array::zeros::<f32>(&(8, 4)).unwrap();
    let qbiases = Array::zeros::<f32>(&(8, 4)).unwrap();
    let mut weights = Weights::new();
    weights.insert("model.layers.0.self_attn.q_proj.weight".to_string(), weight);
    weights.insert("model.layers.0.self_attn.q_proj.scales".to_string(), scales);
    weights.insert(
      "model.layers.0.self_attn.q_proj.biases".to_string(),
      qbiases,
    );

    let quant =
      crate::lm::quant::PerLayerQuantization::from_global(crate::lm::quant::Quantization {
        group_size: 32,
        bits: 4,
        mode: crate::lm::quant::QuantMode::Affine,
      });
    let err = build_base_linear(
      &weights,
      "model.layers.0.self_attn.q_proj",
      &weights["model.layers.0.self_attn.q_proj.weight"],
      Some(&quant),
      true, // fan_in_fan_out
    )
    .unwrap_err();
    assert!(matches!(
      err,
      Error::LayerKeyed(ref payload)
        if matches!(payload.inner(), Error::InvariantViolation(_))
    ));
  }

  // ───────────── safetensors filename + neither-shape ─────────────

  #[test]
  fn peft_load_uses_adapter_model_safetensors_filename() {
    // A PEFT config pairs with `adapter_model.safetensors` (not mlx-lm's
    // `adapters.safetensors`). `load_adapters` picks the file by config shape.
    let tmp = std::env::temp_dir().join(format!("mlxrs_peft_fname_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = r#"{ "peft_type": "LORA", "r": 2, "lora_alpha": 4.0,
      "target_modules": ["self_attn.q_proj"] }"#;
    let q_paths: Vec<String> = (0..4)
      .map(|b| format!("model.layers.{b}.self_attn.q_proj"))
      .collect();
    let q_refs: Vec<&str> = q_paths.iter().map(String::as_str).collect();
    // write_mock_peft_adapter writes `adapter_model.safetensors` — NOT
    // `adapters.safetensors`. Confirm load still succeeds.
    write_mock_peft_adapter(&tmp, cfg, &q_refs, 2, false, 0.01);
    assert!(!tmp.join("adapters.safetensors").exists());
    assert!(tmp.join("adapter_model.safetensors").exists());
    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    std::fs::remove_dir_all(&tmp).ok();
  }

  #[test]
  fn mlxlm_native_path_unchanged_by_peft_work() {
    // A faithful mlx-lm-native config still parses to the MlxLm selection and
    // loads via `adapters.safetensors` — the PEFT additions did not regress
    // the native path.
    let json = r#"{
      "fine_tune_type": "lora", "num_layers": 8,
      "lora_parameters": { "rank": 4, "scale": 16.0, "keys": ["q_proj"] }
    }"#;
    let cfg = LoraConfig::from_json(json).unwrap();
    assert!(matches!(
      cfg.selection,
      AdapterSelection::MlxLm { num_layers: 8 }
    ));
    assert!(cfg.peft().is_none());
    assert_eq!(cfg.scale_for("anything"), 16.0);
    assert_eq!(cfg.rank_for("anything"), 4);
    assert!(!cfg.fan_in_fan_out());
  }

  // ═════════════════════════════ DoRA — spec-named tests ═════════════════════════════
  //
  // Tests with the names called out by the DoRA spec (#161). Some of these are
  // (renamed) duplicates of pre-existing hand-traced tests; keeping both
  // preserves the existing coverage *and* surfaces the spec-named tests in the
  // test report (the spec asked for these exact names).

  /// `dora_linear_forward_matches_python_reference` — assert the
  /// [`DoRALinear::forward`] output matches a hand-traced scalar reference
  /// derived from mlx-lm `tuner/dora.py::DoRALinear.__call__`
  /// (`tuner/dora.py:111-128`).
  ///
  /// Setup: base `W = I_{[2,3]}` truncated, `lora_a = I_{[3,2]}` truncated,
  /// `lora_b = I_2`, `scale = 2.0`, `x = [1, 2, 3]`. Picks `m = [3, 3]` so the
  /// `m / ‖adapted‖₂` renorm is the identity, isolating the DoRA wiring against
  /// the LoRA arithmetic; expected `out = [3, 6]`.
  #[test]
  fn dora_linear_forward_matches_python_reference() {
    // adapted = W + scale·(lora_bᵀ @ lora_aᵀ) = [[3,0,0],[0,3,0]], ‖·‖₂ = [3,3].
    // m = [3, 3] ⇒ renorm = identity. base(x) = [1, 2], scale·z = [2, 4] ⇒ [3, 6].
    let m = Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: Some(m),
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = DoRALinear::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.forward(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-5);
  }

  /// `dora_linear_fuse_into_base_round_trip` — fuse the DoRA adapter into the
  /// base, run the fused base's plain forward, assert it matches the un-fused
  /// DoRA forward within fp tolerance (mlx-lm `tuner/dora.py:32-56` /
  /// `DoRA+Layers.swift::fuse`).
  #[test]
  fn dora_linear_fuse_into_base_round_trip() {
    let m = Array::from_slice::<f32>(&[1.5, 2.5], &(2usize,)).unwrap();
    let params = AdapterParams {
      lora_a: lora_a(),
      lora_b: lora_b(),
      magnitude: Some(m),
    };
    let base = BaseLinear::dense(base_weight(), None).unwrap();
    let layer = DoRALinear::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut via_forward = layer.forward(&x).unwrap();
    // Fuse, then run the fused base's plain forward — must match.
    let fused = layer.fuse(false).unwrap();
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      1e-4,
    );
  }

  /// `dora_embedding_forward_matches_python_reference` — assert
  /// [`DoRAEmbedding::forward`] matches a hand-traced scalar reference for
  /// mlx-lm `tuner/dora.py::DoRAEmbedding.__call__` (`tuner/dora.py:198-210`).
  ///
  /// Setup: `weight = I_{[3, 3]}` (3 token rows, 3 dims), so for `x = [0, 2]`:
  /// `y[0] = [1,0,0]`, `y[1] = [0,0,1]`. `lora_a = zeros([3, 2])` and
  /// `lora_b = zeros([2, 3])` ⇒ `z = 0` ⇒ `adapted == y` ⇒ `denom = ‖y‖₂ = [1, 1]`.
  /// Setting `m = [1, 1, 1]` gives `m[x] / denom = [1, 1]` ⇒ `out == y`,
  /// which validates the gather + per-token renorm wiring against a known
  /// fixed point of the DoRA computation.
  #[test]
  fn dora_embedding_forward_matches_python_reference() {
    let num_embeddings = 3usize;
    let dims = 3usize;
    let r = 2usize;
    // weight = I_3 (one-hot rows).
    #[rustfmt::skip]
    let weight = Array::from_slice::<f32>(
      &[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
      ],
      &(num_embeddings, dims),
    ).unwrap();
    let base = BaseEmbedding::dense(weight).unwrap();

    let lora_a = Array::zeros::<f32>(&(num_embeddings, r)).unwrap();
    let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
    let m = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, 2.0).unwrap();

    // Gather rows 0 and 2 ⇒ [[1,0,0], [0,0,1]].
    let ids = Array::from_slice::<i32>(&[0, 2], &(2usize,)).unwrap();
    let mut out = layer.forward(&ids).unwrap();
    approx_eq(
      &out.to_vec::<f32>().unwrap(),
      &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
      1e-5,
    );
  }

  /// Companion to [`dora_embedding_forward_matches_python_reference`] for a
  /// **non-identity** DoRA renorm — set `m` to *half* the per-token adapted
  /// norm so the per-token renorm halves the output. With `lora_*` zero,
  /// `adapted = y` and `‖y‖₂ = [1, 1]`; `m = [0.5, 0.5, 0.5]` ⇒ `m[x] / denom
  /// = [0.5, 0.5]` ⇒ `out = 0.5 · y`. Validates the per-token renorm wiring
  /// distinguishes from the global `as_linear` renorm path.
  #[test]
  fn dora_embedding_forward_per_token_renorm_halves() {
    let num_embeddings = 3usize;
    let dims = 3usize;
    let r = 2usize;
    #[rustfmt::skip]
    let weight = Array::from_slice::<f32>(
      &[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
      ],
      &(num_embeddings, dims),
    ).unwrap();
    let base = BaseEmbedding::dense(weight).unwrap();
    let lora_a = Array::zeros::<f32>(&(num_embeddings, r)).unwrap();
    let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
    let m = Array::from_slice::<f32>(&[0.5, 0.5, 0.5], &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, 1.0).unwrap();
    let ids = Array::from_slice::<i32>(&[1], &(1usize,)).unwrap();
    let mut out = layer.forward(&ids).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[0.0, 0.5, 0.0], 1e-5);
  }

  /// DoRAEmbedding's `as_linear` is the tied-weight LM-head path
  /// (`tuner/dora.py:212-224`) — for a one-hot embedding table with zero
  /// adapter, `as_linear(x) == x @ Iᵀ = x` modulo the global renorm
  /// `(m / ‖weight‖₂)` which is `[1, 1, 1]` here ⇒ identity output.
  #[test]
  fn dora_embedding_as_linear_one_hot_identity() {
    let num_embeddings = 3usize;
    let dims = 3usize;
    let r = 2usize;
    #[rustfmt::skip]
    let weight = Array::from_slice::<f32>(
      &[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
      ],
      &(num_embeddings, dims),
    ).unwrap();
    let base = BaseEmbedding::dense(weight).unwrap();
    let lora_a = Array::zeros::<f32>(&(num_embeddings, r)).unwrap();
    let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
    // m = ‖weight‖₂ row-wise = [1, 1, 1] ⇒ renorm = identity globally.
    let m = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, 2.0).unwrap();
    // x = [[1, 2, 3]] ⇒ x @ Iᵀ = [1, 2, 3] ⇒ renormed = [1, 2, 3].
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let mut out = layer.as_linear(&x).unwrap();
    approx_eq(&out.to_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0], 1e-5);
  }

  /// [`DoRAEmbedding::fuse`] round-trip — fuse the adapter into a fresh dense
  /// embedding and assert the fused weight's `as_linear` matches the un-fused
  /// `as_linear` within fp tolerance (mlx-lm `tuner/dora.py:153-166`). The
  /// `forward` path is per-token-renormed and intentionally distinct from
  /// `fuse`; `as_linear` is the global-renorm path that fuse mirrors.
  #[test]
  fn dora_embedding_fuse_round_trip() {
    let num_embeddings = 3usize;
    let dims = 3usize;
    let r = 2usize;
    #[rustfmt::skip]
    let weight = Array::from_slice::<f32>(
      &[
        1.0, 0.5, 0.0,
        0.0, 1.0, 0.5,
        0.5, 0.0, 1.0,
      ],
      &(num_embeddings, dims),
    ).unwrap();
    let base = BaseEmbedding::dense(weight).unwrap();
    let lora_a =
      Array::from_slice::<f32>(&[0.1, 0.0, 0.0, 0.1, 0.1, 0.1], &(num_embeddings, r)).unwrap();
    let lora_b = Array::from_slice::<f32>(&[0.2, 0.0, 0.1, 0.0, 0.1, 0.2], &(r, dims)).unwrap();
    let m = Array::from_slice::<f32>(&[1.5, 2.0, 1.2], &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 0.5], &(1, dims)).unwrap();
    let mut via_aslinear = layer.as_linear(&x).unwrap();
    let fused = layer.fuse().unwrap();
    let mut via_fused_aslinear = fused.as_linear(&x).unwrap();
    approx_eq(
      &via_fused_aslinear.to_vec::<f32>().unwrap(),
      &via_aslinear.to_vec::<f32>().unwrap(),
      1e-4,
    );
  }

  /// DoRAEmbedding rejects a magnitude-less `AdapterParams` (LoRA-flavored
  /// factors) at construction — same contract as [`DoRALinear`].
  #[test]
  fn dora_embedding_requires_magnitude() {
    let num_embeddings = 3usize;
    let dims = 3usize;
    let r = 2usize;
    let weight = Array::zeros::<f32>(&(num_embeddings, dims)).unwrap();
    let base = BaseEmbedding::dense(weight).unwrap();
    let lora_a = Array::zeros::<f32>(&(num_embeddings, r)).unwrap();
    let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: None,
    };
    let err = DoRAEmbedding::new(base, params, 1.0).unwrap_err();
    assert!(
      matches!(&err, Error::MissingField(p)
        if p.type_name() == "DoRAEmbedding::new" && p.field().contains("magnitude")),
      "expected Error::MissingField naming `magnitude`, got {err:?}"
    );
  }

  /// DoRAEmbedding rejects a `lora_a` whose leading axis is not
  /// `num_embeddings` (the embedding-orientation factor cross-check).
  #[test]
  fn dora_embedding_rejects_wrong_factor_shape() {
    let num_embeddings = 3usize;
    let dims = 3usize;
    let r = 2usize;
    let weight = Array::zeros::<f32>(&(num_embeddings, dims)).unwrap();
    let base = BaseEmbedding::dense(weight).unwrap();
    // bad: lora_a is [2, r] instead of [num_embeddings=3, r].
    let bad_a = Array::zeros::<f32>(&(2usize, r)).unwrap();
    let lora_b = Array::zeros::<f32>(&(r, dims)).unwrap();
    let m = Array::zeros::<f32>(&(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a: bad_a,
      lora_b,
      magnitude: Some(m),
    };
    let err = DoRAEmbedding::new(base, params, 1.0).unwrap_err();
    // `validate_embedding_factor_shapes` hits the leading-axis cross-check
    // (`a_leading_axis != num_embeddings`) and returns the typed
    // `Error::LengthMismatch` (expected = num_embeddings, actual = 2).
    assert!(
      matches!(&err, Error::LengthMismatch(p)
        if p.expected() == num_embeddings && p.actual() == 2
          && p.context().contains("lora_a")),
      "expected Error::LengthMismatch for wrong leading axis, got {err:?}"
    );
  }

  /// `qdora_linear_forward_matches_python_reference` — assert the QDoRA
  /// forward (DoRA over a quantized base) matches the dense DoRA forward
  /// within affine-quantization error, exercising the `quantized_matmul` base
  /// path against the dense baseline.
  #[test]
  fn qdora_linear_forward_matches_python_reference() {
    let input_dims = 64usize;
    let output_dims = 2usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();

    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();

    // m derived from the *dense* adapted weight so dense and quantized share an
    // identical magnitude vector — any difference is then quantization error
    // alone (not a magnitude mismatch).
    let dense_params_no_m = AdapterParams {
      lora_a: la.try_clone().unwrap(),
      lora_b: lb.try_clone().unwrap(),
      magnitude: None,
    };
    let scale = 2.0f32;
    let delta = lora_delta(&dense_params_no_m, scale).unwrap();
    let adapted = dense_w.add(&delta).unwrap();
    let m = ops::linalg_full::norm(&adapted, 2.0, &[1], false).unwrap();

    let dense_base = BaseLinear::dense(dense_w.try_clone().unwrap(), None).unwrap();
    let dense_layer = DoRALinear::new(
      dense_base,
      AdapterParams {
        lora_a: la.try_clone().unwrap(),
        lora_b: lb.try_clone().unwrap(),
        magnitude: Some(m.try_clone().unwrap()),
      },
      scale,
    )
    .unwrap();
    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();
    let mut dense_out = dense_layer.forward(&x).unwrap();

    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let q_layer = DoRALinear::new(
      q_base,
      AdapterParams {
        lora_a: la,
        lora_b: lb,
        magnitude: Some(m),
      },
      scale,
    )
    .unwrap();
    let mut q_out = q_layer.forward(&x).unwrap();

    approx_eq(
      &q_out.to_vec::<f32>().unwrap(),
      &dense_out.to_vec::<f32>().unwrap(),
      2e-2,
    );
  }

  /// `qdora_linear_fuse_round_trip` — fuse a QDoRA layer (`dequantize=true`)
  /// into a dense base, assert the fused base's plain forward matches the
  /// un-fused QDoRA forward within quantization error.
  #[test]
  fn qdora_linear_fuse_round_trip() {
    let input_dims = 64usize;
    let output_dims = 2usize;
    let mut wdata = vec![1.0f32; input_dims];
    wdata.extend(vec![0.5f32; input_dims]);
    let dense_w = Array::from_slice::<f32>(&wdata, &(output_dims, input_dims)).unwrap();
    let la = Array::full::<f32>(&(input_dims, 2usize), 0.01).unwrap();
    let lb = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let m = Array::from_slice::<f32>(&[1.5, 2.5], &(output_dims,)).unwrap();
    let x = Array::full::<f32>(&(1usize, input_dims), 1.0).unwrap();

    let (w_q, scales, biases) = ops::quantized::quantize(&dense_w, 32, 8, "affine", None).unwrap();
    let q_base =
      BaseLinear::quantized(w_q, scales, biases, None, 32, 8, "affine".to_string()).unwrap();
    let q_layer = DoRALinear::new(
      q_base,
      AdapterParams {
        lora_a: la,
        lora_b: lb,
        magnitude: Some(m),
      },
      2.0,
    )
    .unwrap();
    let mut via_forward = q_layer.forward(&x).unwrap();
    let fused = q_layer.fuse(true).unwrap();
    assert!(matches!(fused, BaseLinear::Dense { .. }));
    let mut via_fused = fused.base_output(&x).unwrap();
    approx_eq(
      &via_fused.to_vec::<f32>().unwrap(),
      &via_forward.to_vec::<f32>().unwrap(),
      2e-2,
    );
  }

  /// `load_dora_adapter_from_safetensors` — write a small adapter directory
  /// (`adapter_config.json` with `fine_tune_type: "dora"`, plus
  /// `adapters.safetensors` carrying `lora_a` / `lora_b` / `m` for each
  /// targeted path), load via the existing [`load_adapters`] entry, and
  /// verify the resulting layers are [`LoraLayer::Dora`] with the right
  /// magnitude shape.
  #[test]
  fn load_dora_adapter_from_safetensors() {
    let tmp = std::env::temp_dir().join(format!("mlxrs_a2_dora_load_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    write_mock_adapter(&tmp, "dora", true);

    let weights = toy_weights();
    let layers = load_adapters(&weights, &tmp, None, 4).unwrap();
    assert_eq!(layers.len(), 4);
    for b in 0..4 {
      let key = format!("model.layers.{b}.self_attn.q_proj");
      match layers.get(&key) {
        Some(LoraLayer::Dora(d)) => {
          // magnitude must be shape [output_dims=2] per `write_mock_adapter`'s
          // `m = [3, 3]` fixture.
          assert_eq!(d.magnitude().shape(), &[2]);
        }
        other => panic!("expected DoRA layer at {key}, got {other:?}"),
      }
    }
    std::fs::remove_dir_all(&tmp).ok();
  }

  /// `linear_to_dora_layers_grafts_correctly` — graft DoRA adapters into the
  /// targeted linear paths of a synthetic model and verify only the targeted
  /// layers are wrapped (and as the `Dora` variant), others are untouched.
  /// Uses [`linear_to_lora_layers`] with a `fine_tune_type: "dora"` config —
  /// the existing entrypoint is the "sibling" referenced in the DoRA spec
  /// (dispatches to [`DoRALinear`] via `LoraConfig::is_dora()`).
  #[test]
  fn linear_to_dora_layers_grafts_correctly() {
    let weights = toy_weights();
    // mlx-lm-native DoRA config: keys=["self_attn.q_proj"], rank=2.
    let cfg = LoraConfig {
      fine_tune_type: FineTuneType::Dora,
      lora_parameters: LoraParameters {
        rank: 2,
        scale: Some(2.0),
        alpha: None,
        keys: vec!["self_attn.q_proj".to_string()],
        dropout: None,
      },
      use_dora: false,
      selection: AdapterSelection::MlxLm { num_layers: 16 },
    };

    // DoRA AdapterParams for each q_proj path — m chosen so the renorm is
    // identity (‖adapted‖₂ = [3, 3] for these factors at scale 2.0).
    let mut params = HashMap::new();
    for b in 0..4 {
      let path = format!("model.layers.{b}.self_attn.q_proj");
      params.insert(
        path,
        AdapterParams {
          lora_a: lora_a(),
          lora_b: lora_b(),
          magnitude: Some(Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,)).unwrap()),
        },
      );
    }

    let layers = linear_to_lora_layers(&weights, &cfg, &params, None, 4).unwrap();
    // Exactly 4 q_proj paths wrapped (one per block); k_proj and lm_head left
    // untouched.
    assert_eq!(layers.len(), 4);
    for b in 0..4 {
      let key = format!("model.layers.{b}.self_attn.q_proj");
      assert!(
        matches!(layers.get(&key), Some(LoraLayer::Dora(_))),
        "expected DoRA layer at {key}"
      );
    }
    assert!(!layers.contains_key("model.layers.0.self_attn.k_proj"));
    assert!(!layers.contains_key("lm_head"));
  }

  // ════════════════ DoRAEmbedding mixed-precision dtype ════════════════
  //
  // Regression coverage for the dtype flow in
  // `DoRAEmbedding::forward` and `DoRAEmbedding::as_linear`: the low-rank
  // product (`z` / `delta`) and the renorm scale (`m[x]/denom`,
  // `m/denom`) must stay uncast through the L2 norm and the final
  // multiply, matching mlx-lm `tuner/dora.py:198-224` (which keeps them
  // uncast for the norm-and-scale compute and only casts the `out`
  // accumulator). With f16 base and f32 adapter, casting them to the base
  // / input dtype upfront silently drops ~16 bits of precision through the
  // renorm divisor (and ~7 bits with bf16 base, where rounding is much
  // coarser).
  //
  // Strategy: an `y ≈ -z` cancellation fixture so the f16/bf16 rounding of `z`
  // perturbs ‖y + z‖ by a *relative* amount well above the fp16/bf16 tolerance
  // floor — the uncast pipeline matches the f64 scalar reference; an
  // upfront-cast pipeline would not. The companion regression-oracle
  // test asserts this directly by computing both reference paths and
  // confirming the real output is closer to the uncast one.

  /// f16 round-trip on an f32 fixture — `f64(f16::from_f32(x))`. Models the
  /// `astype(F16)` rounding mlx applies when an f32 source is cast to f16,
  /// so the scalar reference operates on the SAME bit patterns the kernel
  /// does.
  fn f16_rt(x: f32) -> f64 {
    half::f16::from_f32(x).to_f64()
  }

  /// bf16 round-trip on an f32 fixture — `f64(bf16::from_f32(x))`.
  fn bf16_rt(x: f32) -> f64 {
    half::bf16::from_f32(x).to_f64()
  }

  /// Cancellation fixture inputs reused by the four mixed-precision tests.
  /// `y ≈ -z` per token (with a small `eps` perturbation so `denom > 0`) so
  /// the per-token L2 norm of `adapted = y + z` is small and any rounding
  /// error in `z` to fp16/bf16 shows up as a large *relative* change in the
  /// renorm divisor. Coupled with order-magnitude `m`, the resulting
  /// amplified `m/denom` multiplier makes the upfront-cast bug visible at
  /// the fp16 tolerance floor.
  ///
  /// All weight values are chosen to be near (but NOT all exactly) on the f16
  /// representable grid — picking values like 1.0 (exact) for `y` and 0.99 …
  /// fractions for the cancelling `z` means the f32-precision `z` carries
  /// mantissa bits that f16 rounds away, exactly the scenario an
  /// upfront-cast divergence magnifies through `‖adapted‖₂`.
  #[allow(clippy::type_complexity)] // 5-tuple of nested Vec<Vec<f32>> is just
  // the fixture's "5 input tensors" shape; aliasing each would obscure more
  // than it'd clarify in a test fixture.
  fn mp_fixture() -> (
    Vec<Vec<f32>>, // weight_f32 [4][4]
    Vec<Vec<f32>>, // lora_a_f32 [4][2]
    Vec<Vec<f32>>, // lora_b_f32 [2][4]
    Vec<f32>,      // m_f32 [4]
    f32,           // scale
  ) {
    // y = weight[tid] — chosen to be exactly representable in f16 so the
    // round-trip is the identity on y (isolating the divergence to z's
    // rounding).
    // Row 0: [1,1,1,1] — uniform, simplest cancellation.
    // Row 1: [0.5, 0.5, 0.5, 0.5] — half-scale, exact in f16/bf16.
    // Row 2: [0.25, 0.25, 0.25, 0.25] — quarter-scale.
    // Row 3: [-0.75, -0.75, -0.75, -0.75] — negative direction.
    let weight_f32 = vec![
      vec![1.0, 1.0, 1.0, 1.0],
      vec![0.5, 0.5, 0.5, 0.5],
      vec![0.25, 0.25, 0.25, 0.25],
      vec![-0.75, -0.75, -0.75, -0.75],
    ];
    // lora_a[tid] picks ONE adversarial column of lora_b per token (so z is
    // dominated by a single per-row contribution, easier to reason about).
    let lora_a_f32 = vec![
      vec![1.0, 0.0],
      vec![0.5, 0.0],
      vec![0.25, 0.0],
      vec![-0.75, 0.0],
    ];
    // lora_b row 0 carries the cancelling magnitude: `-0.99853` (≈ -1) for
    // all dims. Combined with lora_a (which picks the matching scalar), z
    // for each token is `-0.99853 * weight_scale`, so adapted = y + z ≈
    // tiny. The exact value -0.99853 is OFF the f16 grid (f16 ULP near 1
    // is ~9.77e-4), so f16-rounding z shifts it by an absolute amount
    // comparable to ‖adapted‖ itself.
    let lora_b_f32 = vec![
      vec![-0.99853, -0.99853, -0.99853, -0.99853],
      vec![0.0, 0.0, 0.0, 0.0],
    ];
    // m chosen so the renorm scale `m/denom` is large enough (~100s) that
    // the per-token cast-vs-uncast divergence in `denom` (relative ~25-30%
    // under this cancellation fixture, since z's f16-rounding error is a
    // sizeable fraction of ‖adapted‖) MULTIPLIES `out_pre` into a final-out
    // delta well above the fp16 tolerance floor (5e-3). m roughly tracks
    // each token's |y| so the final output stays in fp16 range (~order 1).
    let m_f32 = vec![1.0, 0.5, 0.25, 0.75];
    let scale = 1.0f32;
    (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale)
  }

  /// Build the scalar f64 reference for `DoRAEmbedding::forward` matching the
  /// mlx-lm pipeline (`tuner/dora.py:198-210`) with optional `cast_z_upfront`
  /// to model the divergent (cast-upfront) computation. The reference
  /// operates on `rt(x)` — a
  /// pre-rounded version of the f32 source (f16 or bf16 round-trip) so it
  /// reflects the exact bits the kernel sees.
  ///
  /// Returns the kernel-equivalent promoted-dtype outputs for each token in
  /// `ids`, flattened to a `Vec<f32>` for direct comparison against the
  /// kernel output (which we extract via `astype(F32)`). The final value is
  /// NOT round-tripped to the narrow dtype — `forward` now returns the
  /// promoted dtype directly (mlx-lm `tuner/dora.py:208` returns
  /// `(self.m[x] / denom)[..., None] * out` with no astype; the port
  /// mirrors that exactly).
  #[allow(clippy::too_many_arguments)]
  fn forward_scalar_reference(
    weight_f32: &[Vec<f32>],
    lora_a_f32: &[Vec<f32>],
    lora_b_f32: &[Vec<f32>],
    m_f32: &[f32],
    scale: f32,
    ids: &[usize],
    rt: fn(f32) -> f64,
    cast_z_upfront: bool,
  ) -> Vec<f32> {
    let dims = weight_f32[0].len();
    let r = lora_a_f32[0].len();
    let scale_f64 = scale as f64;
    let mut out = Vec::with_capacity(ids.len() * dims);
    for &tid in ids {
      // y = round(weight[tid]) — what the kernel sees after the f16/bf16 cast.
      let y_rt: Vec<f64> = weight_f32[tid].iter().map(|&w| rt(w)).collect();
      // z_uncast[d] = scale * sum_r lora_a[tid][r] * lora_b[r][d] — f32→f64
      // (no rounding; the f32 source bits are exactly representable in f64).
      let mut z_uncast = vec![0.0f64; dims];
      for d in 0..dims {
        let mut acc = 0.0f64;
        for k in 0..r {
          acc += (lora_a_f32[tid][k] as f64) * (lora_b_f32[k][d] as f64);
        }
        z_uncast[d] = scale_f64 * acc;
      }
      // z_cast = round(z_uncast) — what `astype(y.dtype)` produces.
      let z_cast: Vec<f64> = z_uncast.iter().map(|&v| rt(v as f32)).collect();
      // out_pre = round(y + z_cast) — what `out = y + dropout(z).astype(y.dtype)`
      // produces (the add itself runs at y.dtype because both operands are now
      // f16/bf16).
      let out_pre: Vec<f64> = (0..dims)
        .map(|d| rt((y_rt[d] + z_cast[d]) as f32))
        .collect();
      // adapted = y + z_for_norm. The divergent path = `cast_z_upfront`
      // true; the correct path = false (uncast). mlx promotes
      // y(f16/bf16) + z(f32) to f32; we work at f64 to give the reference
      // bounded round-off well below the fp16/bf16 tolerance.
      let z_for_norm = if cast_z_upfront { &z_cast } else { &z_uncast };
      let adapted: Vec<f64> = (0..dims).map(|d| y_rt[d] + z_for_norm[d]).collect();
      let denom = adapted.iter().map(|v| v * v).sum::<f64>().sqrt();
      let norm_scale = (m_f32[tid] as f64) / denom;
      // scaled_out = norm_scale * out_pre — at f64; mlx runs at f32
      // (promotion from f16/bf16 * f32). NO final cast to y.dtype — `forward`
      // returns the promoted dtype directly (mlx-lm `tuner/dora.py:208`),
      // so the reference is returned at the promoted dtype too (f32 for the
      // mixed-precision fixture). f64 → f32 narrowing is fine here: the f64
      // reference's round-off well below the fp16/bf16 tolerance floor used
      // by the assertions.
      for &op in &out_pre {
        let scaled = norm_scale * op;
        out.push(scaled as f32);
      }
    }
    out
  }

  /// Build the scalar f64 reference for `DoRAEmbedding::as_linear` matching
  /// mlx-lm `tuner/dora.py:212-224`. With `cast_delta_upfront`, models the
  /// divergent path (delta cast to weight.dtype before the row-norm);
  /// without it, the uncast path. Output is the kernel-equivalent
  /// `[batch, num_embeddings]` flattened to `Vec<f32>` for direct comparison
  /// (the kernel returns the promoted dtype — for f16 base × f32 adapter →
  /// f32 — so we DON'T round-trip the final value to fp16, matching the new
  /// code's "no final astype" choice).
  #[allow(clippy::too_many_arguments)]
  fn as_linear_scalar_reference(
    weight_f32: &[Vec<f32>],
    lora_a_f32: &[Vec<f32>],
    lora_b_f32: &[Vec<f32>],
    m_f32: &[f32],
    scale: f32,
    x_f32: &[Vec<f32>],
    rt: fn(f32) -> f64,
    cast_delta_upfront: bool,
  ) -> Vec<f32> {
    let num_embeddings = weight_f32.len();
    let dims = weight_f32[0].len();
    let r = lora_a_f32[0].len();
    let batch = x_f32.len();
    let scale_f64 = scale as f64;
    // delta_uncast[e][d] = scale * sum_r lora_a[e][r] * lora_b[r][d] (f64).
    let mut delta_uncast = vec![vec![0.0f64; dims]; num_embeddings];
    for e in 0..num_embeddings {
      for d in 0..dims {
        let mut acc = 0.0f64;
        for k in 0..r {
          acc += (lora_a_f32[e][k] as f64) * (lora_b_f32[k][d] as f64);
        }
        delta_uncast[e][d] = scale_f64 * acc;
      }
    }
    // delta_cast[e][d] = round(delta_uncast[e][d]) — what astype(weight.dtype)
    // produces; only used by the buggy-cast reference.
    let delta_cast: Vec<Vec<f64>> = delta_uncast
      .iter()
      .map(|row| row.iter().map(|&v| rt(v as f32)).collect())
      .collect();
    let delta_for_norm = if cast_delta_upfront {
      &delta_cast
    } else {
      &delta_uncast
    };
    // adapted[e][d] = weight_rt[e][d] + delta_for_norm[e][d] (f64; mlx promotes
    // to f32 — f64 reference is precise enough for the fp tolerance).
    let mut adapted = vec![vec![0.0f64; dims]; num_embeddings];
    for e in 0..num_embeddings {
      for d in 0..dims {
        adapted[e][d] = rt(weight_f32[e][d]) + delta_for_norm[e][d];
      }
    }
    // denom[e] = ‖adapted[e]‖₂, axis=1 (`tuner/dora.py:219`).
    let denom: Vec<f64> = adapted
      .iter()
      .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
      .collect();
    // norm_scale[e] = m[e] / denom[e] (UNCAST, `tuner/dora.py:222`).
    let norm_scale: Vec<f64> = (0..num_embeddings)
      .map(|e| (m_f32[e] as f64) / denom[e])
      .collect();
    // y[b][e] = sum_d x_rt[b][d] * weight_rt[e][d] — x@weightᵀ at the base
    // dtype. f16+f32 promote to f32; f64 reference is more than enough.
    let mut out = Vec::with_capacity(batch * num_embeddings);
    for x_row in x_f32 {
      let x_rt: Vec<f64> = x_row.iter().map(|&v| rt(v)).collect();
      for e in 0..num_embeddings {
        let mut y_be = 0.0f64;
        for d in 0..dims {
          y_be += x_rt[d] * rt(weight_f32[e][d]);
        }
        // scaled_z_be = scale * (x_rt @ lora_b.T @ lora_a.T)[e]
        //            = sum_d x_rt[d] * delta_uncast[e][d] / something...
        // Cleanest: z_be = sum_k (x @ lora_b.T)[k] * lora_a[e][k]
        //                = sum_k (sum_d x_rt[d] * lora_b[k][d]) * lora_a[e][k]
        let xb: Vec<f64> = (0..r)
          .map(|k| {
            (0..dims)
              .map(|d| x_rt[d] * (lora_b_f32[k][d] as f64))
              .sum::<f64>()
          })
          .collect();
        let z_be: f64 = (0..r).map(|k| xb[k] * (lora_a_f32[e][k] as f64)).sum();
        let scaled_z_be = scale_f64 * z_be;
        // out_pre = y + round(scaled_z, x.dtype). Cast scaled_z to x.dtype
        // first (mirrors mlx-lm `(self.scale * z).astype(x.dtype)`).
        let scaled_z_cast = rt(scaled_z_be as f32);
        let out_pre = y_be + scaled_z_cast;
        // Final: norm_scale[e] * out_pre. mlx promotes f32*f16 → f32; f64
        // reference returned as f32 for direct compare against the kernel
        // output extracted via `astype(F32)`. No final astype to base dtype
        // — mlx-lm doesn't cast here and the port doesn't either.
        out.push((norm_scale[e] * out_pre) as f32);
      }
    }
    out
  }

  /// `dora_embedding_forward_mixed_precision_matches_reference_f16_base_f32_adapter`
  /// — exercise the dtype fix: with an f16 embedding weight and f32
  /// adapter factors + magnitude, the renorm divisor must be computed at the
  /// UNCAST dtype (`forward`'s `adapted = y + z` uses uncast `z`,
  /// mirroring mlx-lm `tuner/dora.py:204`). Adversarial `y ≈ -z` fixture so
  /// the f16 rounding of `z` perturbs ‖adapted‖ by a relative amount above
  /// the fp16 tolerance — an upfront-cast computation would mismatch the
  /// scalar reference by orders of magnitude.
  ///
  /// Also asserts the output dtype is **f32** — `forward` carries no
  /// trailing `astype(y.dtype)`, so it returns mlx's promoted dtype
  /// directly (mlx-lm `tuner/dora.py:208` returns `(m[x]/denom)[..., None] *
  /// out` with no astype; f16 base × f32 adapter promotes to f32).
  #[test]
  fn dora_embedding_forward_mixed_precision_matches_reference_f16_base_f32_adapter() {
    let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
    let num_embeddings = weight_f32.len();
    let dims = weight_f32[0].len();
    let r = lora_a_f32[0].len();
    let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
    let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
    let weight_f16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let base = BaseEmbedding::dense(weight_f16).unwrap();
    let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
    let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
    let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, scale).unwrap();
    // Stress all four tokens — the adversarial cancellation is per-token.
    let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
    let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
    let out = layer.forward(&ids).unwrap();
    // Final dtype must be f32 — mlx promotes f16 × f32 → f32 on the final
    // `(m[x]/denom)[..., None] * out` multiply, and there is no narrowing
    // astype pinning the return to y.dtype.
    assert_eq!(
      out.dtype().unwrap(),
      Dtype::F32,
      "forward must return the promoted dtype = f32 (f16 base × f32 adapter)"
    );
    let mut out_f32 = out.astype(Dtype::F32).unwrap();
    let got = out_f32.to_vec::<f32>().unwrap();
    let ids_usize: Vec<usize> = (0..num_embeddings).collect();
    let want = forward_scalar_reference(
      &weight_f32,
      &lora_a_f32,
      &lora_b_f32,
      &m_f32,
      scale,
      &ids_usize,
      f16_rt,
      false, // uncast-z pipeline for the renorm.
    );
    // Promoted-dtype output. Tolerance still ~5e-3: the cancellation-fixture
    // f16 rounding of `y` (which enters `adapted = y + z`) dominates the
    // residual error; the dropped final-narrowing cast does not buy a
    // tighter fit because the f16 round-off was already absorbed upstream.
    // Keeping the original tolerance preserves the test's defect-detection
    // power against the upfront-cast bug.
    approx_eq(&got, &want, 5e-3);
  }

  /// `dora_embedding_forward_mixed_precision_matches_reference_bf16_base_f32_adapter`
  /// — bf16 sibling of the f16 test. bf16 has only ~7 mantissa bits, so the
  /// upfront-cast bug's per-element error is ~16× the f16 case; tolerance is
  /// loosened accordingly (`5e-2` per-element).
  #[test]
  fn dora_embedding_forward_mixed_precision_matches_reference_bf16_base_f32_adapter() {
    let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
    let num_embeddings = weight_f32.len();
    let dims = weight_f32[0].len();
    let r = lora_a_f32[0].len();
    let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
    let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
    let weight_bf16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
      .unwrap()
      .astype(Dtype::BF16)
      .unwrap();
    let base = BaseEmbedding::dense(weight_bf16).unwrap();
    let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
    let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
    let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, scale).unwrap();
    let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
    let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
    let out = layer.forward(&ids).unwrap();
    // Promoted dtype = f32 (bf16 × f32 → f32 under mlx promotion); the
    // narrowing astype was removed.
    assert_eq!(
      out.dtype().unwrap(),
      Dtype::F32,
      "forward must return the promoted dtype = f32 (bf16 base × f32 adapter)"
    );
    let mut out_f32 = out.astype(Dtype::F32).unwrap();
    let got = out_f32.to_vec::<f32>().unwrap();
    let ids_usize: Vec<usize> = (0..num_embeddings).collect();
    let want = forward_scalar_reference(
      &weight_f32,
      &lora_a_f32,
      &lora_b_f32,
      &m_f32,
      scale,
      &ids_usize,
      bf16_rt,
      false,
    );
    // bf16 tolerance: looser, matching its narrower mantissa (the bf16
    // round-off on y dominates; same reasoning as the f16 sibling).
    approx_eq(&got, &want, 5e-2);
  }

  /// `dora_embedding_as_linear_mixed_precision_matches_reference_f16_base_f32_adapter`
  /// — analogous mixed-precision test for `as_linear`: with f16 weight + f32
  /// adapter, the global adapted-row norm must be computed at the UNCAST
  /// delta (mlx-lm `tuner/dora.py:218`'s `weight + (scale·lora_a) @ lora_b`).
  /// Casting delta to weight.dtype before the row-norm would diverge; the
  /// uncast path doesn't. Returned dtype is f32 (mlx promotes f32·f16 — no
  /// final astype, mlx-lm doesn't cast either).
  #[test]
  fn dora_embedding_as_linear_mixed_precision_matches_reference_f16_base_f32_adapter() {
    let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
    let num_embeddings = weight_f32.len();
    let dims = weight_f32[0].len();
    let r = lora_a_f32[0].len();
    let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
    let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
    let weight_f16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let base = BaseEmbedding::dense(weight_f16).unwrap();
    let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
    let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
    let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, scale).unwrap();
    // x rows tuned so x @ weightᵀ varies across the batch; passed as f16 to
    // match the embedding base dtype (typical LM-head call site).
    let x_f32 = vec![vec![1.0, 1.0, 1.0, 1.0], vec![0.5, -0.25, 0.75, -0.125]];
    let flat_x: Vec<f32> = x_f32.iter().flatten().copied().collect();
    let x_arr = Array::from_slice::<f32>(&flat_x, &(x_f32.len(), dims))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let out = layer.as_linear(&x_arr).unwrap();
    let mut out_f32 = out.astype(Dtype::F32).unwrap();
    let got = out_f32.to_vec::<f32>().unwrap();
    let want = as_linear_scalar_reference(
      &weight_f32,
      &lora_a_f32,
      &lora_b_f32,
      &m_f32,
      scale,
      &x_f32,
      f16_rt,
      false,
    );
    approx_eq(&got, &want, 5e-3);
  }

  /// `dora_embedding_as_linear_mixed_precision_matches_reference_bf16_base_f32_adapter`
  /// — bf16 sibling of `as_linear`'s mixed-precision test.
  #[test]
  fn dora_embedding_as_linear_mixed_precision_matches_reference_bf16_base_f32_adapter() {
    let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
    let num_embeddings = weight_f32.len();
    let dims = weight_f32[0].len();
    let r = lora_a_f32[0].len();
    let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
    let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
    let weight_bf16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
      .unwrap()
      .astype(Dtype::BF16)
      .unwrap();
    let base = BaseEmbedding::dense(weight_bf16).unwrap();
    let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
    let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
    let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, scale).unwrap();
    let x_f32 = vec![vec![1.0, 1.0, 1.0, 1.0], vec![0.5, -0.25, 0.75, -0.125]];
    let flat_x: Vec<f32> = x_f32.iter().flatten().copied().collect();
    let x_arr = Array::from_slice::<f32>(&flat_x, &(x_f32.len(), dims))
      .unwrap()
      .astype(Dtype::BF16)
      .unwrap();
    let out = layer.as_linear(&x_arr).unwrap();
    let mut out_f32 = out.astype(Dtype::F32).unwrap();
    let got = out_f32.to_vec::<f32>().unwrap();
    let want = as_linear_scalar_reference(
      &weight_f32,
      &lora_a_f32,
      &lora_b_f32,
      &m_f32,
      scale,
      &x_f32,
      bf16_rt,
      false,
    );
    approx_eq(&got, &want, 5e-2);
  }

  /// `dora_embedding_forward_loses_precision_with_upfront_cast_regression_oracle`
  /// — assert the (uncast-z, uncast-norm-scale, no-final-astype)
  /// pipeline matches the f64 scalar reference WAY MORE TIGHTLY than an
  /// upfront-cast pipeline would. Cancellation fixture: with f16 base +
  /// f32 adapter and `y ≈ -z`, the f16 rounding of `z` perturbs ‖adapted‖
  /// by a relative amount that flows through `m/denom` and ends up well
  /// above the fp16 tolerance floor on the final output — so the uncast
  /// code matches the scalar reference at ≤ `5e-3`, while comparing against
  /// the upfront-cast reference mismatches by ≥ `1e-2` on at least one
  /// element. Also asserts the promoted return dtype (f32) — direct guard
  /// against both the upfront-cast and final-narrowing-astype regressions.
  #[test]
  fn dora_embedding_forward_loses_precision_with_upfront_cast_regression_oracle() {
    let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
    let num_embeddings = weight_f32.len();
    let dims = weight_f32[0].len();
    let r = lora_a_f32[0].len();
    let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
    let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
    let weight_f16 = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let base = BaseEmbedding::dense(weight_f16).unwrap();
    let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
    let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
    let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, scale).unwrap();
    let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
    let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
    let out = layer.forward(&ids).unwrap();
    // Guard: forward returns the promoted dtype (f32), not the
    // base's f16. Re-introducing the final `astype(y.dtype)` would flip this.
    assert_eq!(
      out.dtype().unwrap(),
      Dtype::F32,
      "regression-oracle: forward must return the promoted dtype = f32"
    );
    let mut out_f32 = out.astype(Dtype::F32).unwrap();
    let got = out_f32.to_vec::<f32>().unwrap();
    let ids_usize: Vec<usize> = (0..num_embeddings).collect();
    let want_new = forward_scalar_reference(
      &weight_f32,
      &lora_a_f32,
      &lora_b_f32,
      &m_f32,
      scale,
      &ids_usize,
      f16_rt,
      false, // uncast pipeline reference
    );
    let want_old = forward_scalar_reference(
      &weight_f32,
      &lora_a_f32,
      &lora_b_f32,
      &m_f32,
      scale,
      &ids_usize,
      f16_rt,
      true, // upfront-cast pipeline reference
    );
    let new_max_err = got
      .iter()
      .zip(want_new.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0f32, f32::max);
    let old_max_err = got
      .iter()
      .zip(want_old.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0f32, f32::max);
    assert!(
      new_max_err <= 5e-3,
      "uncast pipeline must match scalar reference at fp16 tol; got max err {new_max_err}",
    );
    assert!(
      old_max_err >= 1e-2,
      "upfront-cast pipeline must mismatch the scalar reference noticeably; got max err {old_max_err} (cancellation fixture may need re-tuning)",
    );
    // Sanity gap: the uncast pipeline matches the reference at least 5×
    // tighter than the upfront-cast one — the dtype flow is the difference.
    assert!(
      new_max_err * 5.0 <= old_max_err,
      "regression-oracle expected ≥5× tighter uncast-vs-upfront-cast fit; got uncast={new_max_err}, upfront-cast={old_max_err}",
    );
  }

  /// `dora_embedding_forward_returns_promoted_dtype_for_mixed_precision` —
  /// explicit, focused dtype guard for the promoted-return-dtype fix. Asserts
  /// that `forward` returns the mlx-promoted dtype (f32) for both `f16 base × f32
  /// adapter` and `bf16 base × f32 adapter`, NOT the embedding's narrow
  /// dtype. mlx-lm `tuner/dora.py:208` returns `(self.m[x] / denom)[...,
  /// None] * out` directly — no astype — and the port now mirrors that.
  /// Re-introducing a final `astype(y.dtype)` would flip these assertions.
  ///
  /// This test does NOT exercise value parity (the
  /// `*_matches_reference_*_base_f32_adapter` tests do that); it is a pure
  /// dtype contract test.
  #[test]
  fn dora_embedding_forward_returns_promoted_dtype_for_mixed_precision() {
    for (narrow, label) in [(Dtype::F16, "f16"), (Dtype::BF16, "bf16")] {
      let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
      let num_embeddings = weight_f32.len();
      let dims = weight_f32[0].len();
      let r = lora_a_f32[0].len();
      let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
      let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
      let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
      let weight_narrow = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
        .unwrap()
        .astype(narrow)
        .unwrap();
      let base = BaseEmbedding::dense(weight_narrow).unwrap();
      // Adapter factors + magnitude stay f32 — mlx will promote on the
      // final multiply.
      let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r)).unwrap();
      let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims)).unwrap();
      let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,)).unwrap();
      let params = AdapterParams {
        lora_a,
        lora_b,
        magnitude: Some(m),
      };
      let layer = DoRAEmbedding::new(base, params, scale).unwrap();
      let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
      let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
      let out = layer.forward(&ids).unwrap();
      assert_eq!(
        out.dtype().unwrap(),
        Dtype::F32,
        "forward must return promoted dtype f32 for {label} base × f32 adapter (no final narrowing astype)",
      );
    }
  }

  /// `dora_embedding_forward_preserves_base_dtype_for_uniform_precision` —
  /// sanity sibling of the mixed-precision dtype test: when base AND adapter
  /// share a dtype, `forward` returns THAT dtype because no operand triggers
  /// mlx's promotion-on-mix rule. Direct guard against a defensive "just to
  /// be safe" re-introduction of the final astype — a forward that always
  /// did `.astype(y.dtype)` would also pass this test (it's the no-op case),
  /// but combined with `*_returns_promoted_dtype_for_mixed_precision`'s
  /// "must be f32 for f16/bf16 base × f32 adapter" assertion, the pair
  /// triangulates: a re-introduced final astype would pass THIS test and
  /// fail THAT one, pinpointing the regression.
  ///
  /// Covers `(f32, f32)`, `(f16, f16)`, and `(bf16, bf16)`. The half-precision
  /// cases exercise [`scaled`]'s coercion: the scalar `self.scale` is
  /// coerced to `arr.dtype()` (mirroring mlx-lm `to_array(v, a.dtype())`) so
  /// `z = scale · (lora_a[x] @ lora_b)` stays in the adapter's narrow dtype
  /// instead of promoting to f32. An f32 mlx scalar would silently upcast
  /// uniform-half adapters to f32 — this triple-test pins the helper's
  /// behavior across all three uniform precisions.
  #[test]
  fn dora_embedding_forward_preserves_base_dtype_for_uniform_precision() {
    for (uniform, label) in [
      (Dtype::F32, "f32"),
      (Dtype::F16, "f16"),
      (Dtype::BF16, "bf16"),
    ] {
      let (weight_f32, lora_a_f32, lora_b_f32, m_f32, scale) = mp_fixture();
      let num_embeddings = weight_f32.len();
      let dims = weight_f32[0].len();
      let r = lora_a_f32[0].len();
      let flat_w: Vec<f32> = weight_f32.iter().flatten().copied().collect();
      let flat_a: Vec<f32> = lora_a_f32.iter().flatten().copied().collect();
      let flat_b: Vec<f32> = lora_b_f32.iter().flatten().copied().collect();
      // All operands at the same dtype — no promotion at any step; `forward`
      // returns `uniform`.
      let weight = Array::from_slice::<f32>(&flat_w, &(num_embeddings, dims))
        .unwrap()
        .astype(uniform)
        .unwrap();
      let base = BaseEmbedding::dense(weight).unwrap();
      let lora_a = Array::from_slice::<f32>(&flat_a, &(num_embeddings, r))
        .unwrap()
        .astype(uniform)
        .unwrap();
      let lora_b = Array::from_slice::<f32>(&flat_b, &(r, dims))
        .unwrap()
        .astype(uniform)
        .unwrap();
      let m = Array::from_slice::<f32>(&m_f32, &(num_embeddings,))
        .unwrap()
        .astype(uniform)
        .unwrap();
      let params = AdapterParams {
        lora_a,
        lora_b,
        magnitude: Some(m),
      };
      let layer = DoRAEmbedding::new(base, params, scale).unwrap();
      let ids_vec: Vec<i32> = (0..num_embeddings as i32).collect();
      let ids = Array::from_slice::<i32>(&ids_vec, &(num_embeddings,)).unwrap();
      let out = layer.forward(&ids).unwrap();
      assert_eq!(
        out.dtype().unwrap(),
        uniform,
        "forward must return {label} when base AND adapter are uniform {label} (no promotion)",
      );
    }
  }

  // ───────────────────── scaled() coercion ─────────────────────

  /// `scaled_helper_coerces_scalar_to_array_dtype` — unit test on the
  /// [`scaled`] helper: the scalar `scale` operand is cast to `arr`'s dtype
  /// BEFORE the multiply, mirroring mlx-lm's `to_array(v, a.dtype())`
  /// scalar-coercion (mlx-lm `lora.py:97`, `dora.py:200`).
  ///
  /// If the helper created an f32 mlx scalar, `scaled(f16_arr, …)` would
  /// silently return an f32 array (mlx promotes f16 × f32 → f32) —
  /// silently diverging from mlx-lm for uniform-half adapters. This test
  /// triangulates the coercion across all three float dtypes the helper
  /// is expected to round-trip preserving precision.
  #[test]
  fn scaled_helper_coerces_scalar_to_array_dtype() {
    for (dt, label) in [
      (Dtype::F16, "f16"),
      (Dtype::BF16, "bf16"),
      (Dtype::F32, "f32"),
    ] {
      let arr = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,))
        .unwrap()
        .astype(dt)
        .unwrap();
      let out = scaled(&arr, 0.5).unwrap();
      assert_eq!(
        out.dtype().unwrap(),
        dt,
        "scaled must coerce the scalar to the array's dtype and keep the {label} result in {label}",
      );
    }
  }

  /// `dora_embedding_forward_uniform_f16_adapter_returns_f16` — the
  /// `scaled` coercion: with a uniform-f16 base + adapter, `DoRAEmbedding::forward`
  /// must return f16 (mlx-lm `to_array(scale, a.dtype())` keeps the scalar
  /// at f16, so `z = scale · lora_a[x] @ lora_b` stays at f16 and no
  /// downstream op promotes). If `scaled` minted an f32 scalar, the
  /// final `out` would be silently f32 — divergent from mlx-lm.
  ///
  /// Hand-constructed deterministic fixture (all values exact in f16/bf16):
  /// num_embeddings=2, dims=2, r=1; weight=[[1,0],[0,1]], lora_a=[[1],[0]],
  /// lora_b=[[1,0]], m=[1,1], scale=1.0. Per-token math:
  /// - x=0: y=[1,0], z=1·[1,0]=[1,0], adapted=[2,0], ‖·‖=2, m/denom=0.5,
  ///   out_pre=[2,0], out=0.5·[2,0]=[1,0].
  /// - x=1: y=[0,1], z=1·[0,0]=[0,0], adapted=[0,1], ‖·‖=1, m/denom=1,
  ///   out_pre=[0,1], out=1·[0,1]=[0,1].
  ///
  /// Expected for ids=[0,1] is [[1,0],[0,1]] — exact in f16/bf16.
  #[test]
  fn dora_embedding_forward_uniform_f16_adapter_returns_f16() {
    dora_embedding_forward_uniform_dtype_case(Dtype::F16, "f16");
  }

  /// `dora_embedding_forward_uniform_bf16_adapter_returns_bf16` — bf16
  /// sibling of the f16 uniform-dtype contract test. Same fixture (all
  /// values exact in bf16); asserts dtype = bf16 and value parity.
  #[test]
  fn dora_embedding_forward_uniform_bf16_adapter_returns_bf16() {
    dora_embedding_forward_uniform_dtype_case(Dtype::BF16, "bf16");
  }

  /// Shared driver for the uniform-dtype `forward` dtype + value contract.
  /// See [`dora_embedding_forward_uniform_f16_adapter_returns_f16`]'s docstring
  /// for the hand-traced fixture math.
  fn dora_embedding_forward_uniform_dtype_case(uniform: Dtype, label: &str) {
    let num_embeddings = 2usize;
    let dims = 2usize;
    let r = 1usize;
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(num_embeddings, dims))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let base = BaseEmbedding::dense(weight).unwrap();
    let lora_a = Array::from_slice::<f32>(&[1.0, 0.0], &(num_embeddings, r))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let lora_b = Array::from_slice::<f32>(&[1.0, 0.0], &(r, dims))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let m = Array::from_slice::<f32>(&[1.0, 1.0], &(num_embeddings,))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, 1.0f32).unwrap();
    let ids = Array::from_slice::<i32>(&[0, 1], &(2usize,)).unwrap();
    let out = layer.forward(&ids).unwrap();
    assert_eq!(
      out.dtype().unwrap(),
      uniform,
      "forward must return {label} for uniform-{label} base + adapter (scaled() coerces scalar to arr.dtype)",
    );
    let mut out_f32 = out.astype(Dtype::F32).unwrap();
    let got = out_f32.to_vec::<f32>().unwrap();
    // [[1, 0], [0, 1]] — exact in f16/bf16; zero tolerance would also pass,
    // but a tight 1e-3 leaves headroom against any future kernel-order shift.
    approx_eq(&got, &[1.0, 0.0, 0.0, 1.0], 1e-3);
  }

  /// `dora_embedding_as_linear_uniform_f16_adapter_returns_f16` — the
  /// `scaled` coercion for `as_linear`: with uniform-f16 base + adapter, the tied-weight
  /// LM-head forward must also return f16. The same `scaled` helper is on the
  /// hot path (the scale·lora_a delta), so the coercion applies symmetrically.
  ///
  /// Hand-constructed fixture (all values exact in f16/bf16) with x=[[1, 1]]:
  /// - y = x @ weightᵀ = [1, 1]
  /// - z = (x @ lora_bᵀ) @ lora_aᵀ = [1, 0]
  /// - adapted = weight + scale · lora_a @ lora_b = [[2,0],[0,1]]
  /// - denom (axis=1) = [2, 1], norm_scale = [0.5, 1]
  /// - out_pre = y + scale·z = [2, 1], out = norm_scale · out_pre = [1, 1].
  ///
  /// Expected = [[1, 1]] — exact in f16/bf16.
  #[test]
  fn dora_embedding_as_linear_uniform_f16_adapter_returns_f16() {
    dora_embedding_as_linear_uniform_dtype_case(Dtype::F16, "f16");
  }

  /// `dora_embedding_as_linear_uniform_bf16_adapter_returns_bf16` — bf16
  /// sibling of the `as_linear` uniform-dtype contract test.
  #[test]
  fn dora_embedding_as_linear_uniform_bf16_adapter_returns_bf16() {
    dora_embedding_as_linear_uniform_dtype_case(Dtype::BF16, "bf16");
  }

  /// Shared driver for the uniform-dtype `as_linear` dtype + value contract.
  /// See [`dora_embedding_as_linear_uniform_f16_adapter_returns_f16`]'s
  /// docstring for the hand-traced fixture math.
  fn dora_embedding_as_linear_uniform_dtype_case(uniform: Dtype, label: &str) {
    let num_embeddings = 2usize;
    let dims = 2usize;
    let r = 1usize;
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(num_embeddings, dims))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let base = BaseEmbedding::dense(weight).unwrap();
    let lora_a = Array::from_slice::<f32>(&[1.0, 0.0], &(num_embeddings, r))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let lora_b = Array::from_slice::<f32>(&[1.0, 0.0], &(r, dims))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let m = Array::from_slice::<f32>(&[1.0, 1.0], &(num_embeddings,))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let params = AdapterParams {
      lora_a,
      lora_b,
      magnitude: Some(m),
    };
    let layer = DoRAEmbedding::new(base, params, 1.0f32).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 1.0], &(1usize, dims))
      .unwrap()
      .astype(uniform)
      .unwrap();
    let out = layer.as_linear(&x).unwrap();
    assert_eq!(
      out.dtype().unwrap(),
      uniform,
      "as_linear must return {label} for uniform-{label} base + adapter (scaled() coerces scalar to arr.dtype)",
    );
    let mut out_f32 = out.astype(Dtype::F32).unwrap();
    let got = out_f32.to_vec::<f32>().unwrap();
    approx_eq(&got, &[1.0, 1.0], 1e-3);
  }

  /// `dora_linear_forward_uniform_f16_adapter_returns_f16` — sibling
  /// for [`DoRALinear`]: the same [`scaled`] helper is on its hot path, so
  /// the coercion propagates. DoRALinear's `forward` has an explicit trailing
  /// `astype(x.dtype)` on the low-rank term (mlx-lm `tuner/lora.py:97` casts
  /// `(scale * z).astype(x.dtype)`), so the dtype contract here is "out
  /// matches x.dtype" — the scaled() coercion doesn't change THAT contract for
  /// DoRALinear (the trailing astype already enforces it), but the test
  /// pins the contract as a regression oracle against a future refactor
  /// that elides the trailing astype.
  ///
  /// Hand-traced fixture (all values exact in f16): input_dims=3,
  /// output_dims=2, r=2; reuses [`base_weight`], [`lora_a`], [`lora_b`]
  /// (the LoRA `[3, 6]` hand-trace) with m chosen so renorm = identity
  /// (m = ‖adapted‖₂ row-wise = [3, 3], same as
  /// [`dora_linear_forward_hand_traced`]) — expected out = [3, 6].
  #[test]
  fn dora_linear_forward_uniform_f16_adapter_returns_f16() {
    let weight = base_weight().astype(Dtype::F16).unwrap();
    let la = lora_a().astype(Dtype::F16).unwrap();
    let lb = lora_b().astype(Dtype::F16).unwrap();
    let m = Array::from_slice::<f32>(&[3.0, 3.0], &(2usize,))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let params = AdapterParams {
      lora_a: la,
      lora_b: lb,
      magnitude: Some(m),
    };
    let base = BaseLinear::dense(weight, None).unwrap();
    let layer = DoRALinear::new(base, params, 2.0).unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(
      out.dtype().unwrap(),
      Dtype::F16,
      "DoRALinear::forward must return f16 for uniform-f16 base + adapter (trailing astype + scaled() coercion both contribute)",
    );
    let mut out_f32 = out.astype(Dtype::F32).unwrap();
    approx_eq(&out_f32.to_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-3);
  }

  // ──── locate_adapter_safetensors symlink-following regression ────
  //
  // `adapter_candidate_present` must use `metadata()` (NOT
  // `symlink_metadata()`) so that a broken preferred symlink falls through
  // to the fallback candidate, and a symlink loop surfaces as a typed
  // `Error::FileIo` rather than short-circuiting on the link object itself.
  // Unix-gated because they use `std::os::unix::fs::symlink` directly.

  #[cfg(unix)]
  #[test]
  fn locate_adapter_safetensors_falls_back_when_preferred_is_broken_symlink() {
    use std::os::unix::fs::symlink;
    let tmp = std::env::temp_dir().join(format!(
      "mlxrs_lora_broken_symlink_{}_{}",
      std::process::id(),
      std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Broken preferred symlink: adapters.safetensors -> does_not_exist
    symlink(tmp.join("does_not_exist"), tmp.join(MLX_LM_ADAPTER_FILE)).unwrap();
    // Valid fallback regular file (contents irrelevant — locate only stats).
    std::fs::write(tmp.join(PEFT_ADAPTER_FILE), b"valid bytes").unwrap();

    // mlx-lm-native config => preferred = adapters.safetensors (broken
    // symlink), fallback = adapter_model.safetensors (valid). Locate must
    // return the fallback.
    let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let found = locate_adapter_safetensors(&tmp, &cfg)
      .expect("expected fallback to be located when preferred is a broken symlink");
    assert_eq!(found, tmp.join(PEFT_ADAPTER_FILE));

    let _ = std::fs::remove_dir_all(&tmp);
  }

  // ──── locate_adapter_safetensors non-regular path fail-fast ────
  //
  // Two structural-classification tests pin the **NonRegular → fail-fast**
  // contract of [`adapter_candidate_present`] / [`probe_candidate`]: a
  // directory (or FIFO / socket / …) sitting at either the preferred or
  // fallback adapter weights path must surface as a typed `Error::FileIo`
  // with `ErrorKind::InvalidInput` rather than silently being treated as
  // "absent" and falling through. Combined with the broken-symlink and
  // symlink-loop regressions above, the suite now exhaustively pins all four
  // outcomes of the `CandidateProbe` classification (Absent / Present /
  // NonRegular / IoError) at both preferred and fallback positions — any
  // future change that re-introduces a silent collapse will be caught.

  #[cfg(unix)]
  #[test]
  fn locate_adapter_safetensors_rejects_non_regular_preferred_path_even_with_valid_fallback() {
    let tmp = std::env::temp_dir().join(format!(
      "mlxrs_lora_nonreg_preferred_{}_{}",
      std::process::id(),
      std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Preferred slot (mlx-lm-native ⇒ `adapters.safetensors`) is a
    // DIRECTORY — non-regular path the user clearly wanted as a file.
    std::fs::create_dir(tmp.join(MLX_LM_ADAPTER_FILE)).unwrap();
    // Valid fallback present: must NOT be silently used.
    std::fs::write(tmp.join(PEFT_ADAPTER_FILE), b"valid bytes").unwrap();

    let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let err = locate_adapter_safetensors(&tmp, &cfg)
      .expect_err("expected fail-fast Error::FileIo for non-regular preferred path");
    match err {
      Error::FileIo(p) => {
        assert_eq!(
          p.path(),
          tmp.join(MLX_LM_ADAPTER_FILE).as_path(),
          "path round-trips through FileIoPayload"
        );
        assert_eq!(
          p.op(),
          FileOp::Stat,
          "non-regular surfaces from the stat probe"
        );
        assert_eq!(
          p.inner().kind(),
          std::io::ErrorKind::InvalidInput,
          "non-regular candidates surface with InvalidInput",
        );
      }
      other => panic!("expected Error::FileIo for non-regular preferred path, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&tmp);
  }

  #[cfg(unix)]
  #[test]
  fn locate_adapter_safetensors_rejects_non_regular_fallback_path() {
    let tmp = std::env::temp_dir().join(format!(
      "mlxrs_lora_nonreg_fallback_{}_{}",
      std::process::id(),
      std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Preferred (mlx-lm-native ⇒ `adapters.safetensors`) is genuinely
    // absent. Fallback (`adapter_model.safetensors`) is a DIRECTORY.
    std::fs::create_dir(tmp.join(PEFT_ADAPTER_FILE)).unwrap();

    let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let err = locate_adapter_safetensors(&tmp, &cfg)
      .expect_err("expected fail-fast Error::FileIo for non-regular fallback path");
    match err {
      Error::FileIo(p) => {
        assert_eq!(
          p.path(),
          tmp.join(PEFT_ADAPTER_FILE).as_path(),
          "path round-trips through FileIoPayload"
        );
        assert_eq!(
          p.op(),
          FileOp::Stat,
          "non-regular surfaces from the stat probe"
        );
        assert_eq!(
          p.inner().kind(),
          std::io::ErrorKind::InvalidInput,
          "non-regular candidates surface with InvalidInput",
        );
      }
      other => panic!("expected Error::FileIo for non-regular fallback path, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&tmp);
  }

  #[cfg(unix)]
  #[test]
  fn locate_adapter_safetensors_surfaces_symlink_loop_as_typed_file_io() {
    use std::os::unix::fs::symlink;
    let tmp = std::env::temp_dir().join(format!(
      "mlxrs_lora_symlink_loop_{}_{}",
      std::process::id(),
      std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Self-referential loop: adapters.safetensors -> adapters.safetensors.
    // `metadata()` on this resolves via the symlink, hits ELOOP, and returns
    // an `io::Error` whose kind is `FilesystemLoop` (Linux) or similar
    // (macOS surfaces `Uncategorized` for ELOOP on some toolchain versions).
    // The contract we assert is: the helper returns `Err(Error::FileIo(...))`
    // (NOT `Ok(true)` / `Ok(false)`) with the candidate path + FileOp::Stat.
    let preferred = tmp.join(MLX_LM_ADAPTER_FILE);
    symlink(&preferred, &preferred).unwrap();

    let cfg = mlxlm_config(2, keyed_params(vec!["self_attn.q_proj".to_string()]));
    let err = locate_adapter_safetensors(&tmp, &cfg)
      .expect_err("expected typed FileIo error for symlink loop");
    match err {
      Error::FileIo(p) => {
        assert_eq!(
          p.path(),
          preferred.as_path(),
          "path round-trips through FileIoPayload"
        );
        assert_eq!(
          p.op(),
          FileOp::Stat,
          "loop surfaces from the stat probe, not open"
        );
      }
      other => panic!("expected Error::FileIo for symlink loop, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&tmp);
  }
}
