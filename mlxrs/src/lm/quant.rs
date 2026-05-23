//! Weight-map (de)quantization + per-layer [`Quantization`] config schema.
//!
//! Port of mlx-lm's `quantize_model` / `dequantize_model` (`mlx_lm/utils.py`)
//! and mlx-swift-lm's `MLXLMCommon.BaseConfiguration.Quantization` /
//! `PerLayerQuantization` (`Libraries/MLXLMCommon/BaseConfiguration.swift`),
//! adapted to mlxrs's per-project scope: mlxrs has no model-module tree
//! (that is per-usecase, [see project memory:
//! `feedback_no_per_model_arch_porting`](../index.html)), so where mlx-lm
//! walks `nn.Module` leaves replacing `Linear` / `Embedding` with their
//! quantized counterparts, this module walks the [`Weights`] **name map**
//! (the loaded `HashMap<String, Array>` from
//! [`crate::lm::load::load_weights`]) and applies the merged
//! [`crate::ops::quantized::quantize`] / [`crate::ops::quantized::dequantize`]
//! (the #19 ops — **not** a re-implementation) to weights matching a
//! per-layer predicate.
//!
//! The schema mirrors the swift `BaseConfiguration.Quantization` struct
//! verbatim (group_size / bits / mode) and the `PerLayerQuantization`
//! container that lets a per-layer path either skip quantization (an
//! explicit `false` in the config JSON) or override the global parameters
//! (a nested `{ group_size, bits, [mode] }` object). The deserializer
//! handles the interleaved-key JSON shape mlx checkpoints actually emit
//! (global keys side-by-side with `model.layers.…` per-layer keys —
//! `BaseConfiguration.swift:103-118`).
//!
//! ## Predicate (which weight keys get quantized)
//!
//! A faithful adaptation of `mlx_lm.utils.py`'s `wrapped_predicate`
//! (`utils.py:823-835`), translated from the module tree to the weight map.
//! Pass 1 is the caller-supplied **eligibility** check (the structural
//! analogue of `hasattr(module, "to_quantized")` — see [`quantize_weights`]
//! for the closure signature); passes 2–4 are the secondary structural
//! guards mlx-lm runs after `to_quantized`:
//!
//! 1. The architecture-supplied [`Eligible`] closure returns `true`
//!    (the analogue of `hasattr(module, "to_quantized")` —
//!    `utils.py:824`). mlx-lm uses python-class membership; mlxrs has no
//!    module tree, so the caller's closure is the source of truth for which
//!    weight paths are quantization targets. For the historical "every
//!    `.weight` is a candidate" behavior, pass [`default_eligible`].
//! 2. The key ends in `.weight` — mlx's `Linear` / `Embedding` /
//!    `SwitchLinear` all store the dense matrix as the module's `weight`
//!    parameter. The layer **path** is the key with the `.weight` suffix
//!    stripped (mlx-lm passes
//!    `path = "model.layers.0.self_attn.q_proj"` to the predicate; the
//!    weight lives at `"model.layers.0.self_attn.q_proj.weight"`).
//! 3. The array has rank ≥ 2 (mlx-lm `module.weight.shape[-1]` indexes the
//!    last axis; a scalar or 1-D bias is not quantizable).
//! 4. The last axis is divisible by `group_size` (mlx-lm `if
//!    module.weight.shape[-1] % group_size != 0: return False`,
//!    `utils.py:826-827`).
//! 5. The per-layer override (if any) is consulted (mlx-lm
//!    `quant_predicate(path, module)`, `utils.py:829-830`): a
//!    [`QuantizationOption::Skip`] turns this weight off; a
//!    [`QuantizationOption::Quantize`] overrides the global `group_size` /
//!    `bits` / `mode` for this one weight (a "fine-grained" / "mixed
//!    precision" quant — `BaseConfiguration.swift:69-71`).
//!
//! Weights that fail any check pass through **unchanged** — exactly mlx-lm,
//! and exactly the swift `PerLayerQuantization.quantization(layer:)`
//! semantics (`BaseConfiguration.swift:86-100`). When a weight IS
//! quantized, its `<path>.weight` entry is replaced by the packed
//! [`crate::ops::quantized::quantize`] output, and two new entries
//! (`<path>.scales` plus `<path>.biases` for `affine`; `<path>.scales` only
//! for the bias-less float schemes) are inserted — the exact layout
//! mlx-lm's `QuantizedLinear` writes (`mlx/python/mlx/nn/layers/quantized.py:134-137`).
//! Already-quantized triples in the input map pass through verbatim.
//!
//! ## Inverse
//!
//! [`dequantize_weights`] is the inverse: it walks the map looking for the
//! triple shape (`<path>.weight` + `<path>.scales` [+ `<path>.biases`]) and
//! replaces it with the reconstructed dense weight via
//! [`crate::ops::quantized::dequantize`]. Non-triple entries pass through
//! unchanged.
//!
//! Conventions mirror [`crate::lm::sample`] / [`crate::lm::load`]:
//! `Result`-fallible, no implicit eval (the weight `Array`s are returned
//! lazily — no `eval`/`item`/`to_vec` here), recoverable failures map to
//! [`Error::Backend`] / [`Error::ShapeMismatch`] with a clear message.
//!
//! ## Validation contract
//!
//! The already-quantized-triple classifier (`classify_triple`, private to
//! this module) does basic shape-sanity checks (weight dtype, rank ≥ 2,
//! leading-dims-match, mode arity); it does **not** validate per-mode
//! bits/group_size pairings, the scales-last-axis invariant, or scale dtypes.
//! Those per-mode contracts are enforced by `mlx-c`'s
//! `validate_quantized_input` (`mlx/mlx/ops.cpp:75-115`) at the
//! [`crate::ops::quantized::quantize`] / [`crate::ops::quantized::dequantize`]
//! call site and surface as recoverable [`Error::Backend`] from mlx-c with a
//! precise message. Mirroring mlx-c-internal validation here would duplicate
//! work mlx-c already does and would diverge from the reference behavior of
//! mlx-lm's `quantize_module_predicate` (`mlx_lm/utils.py:823-835`, which only
//! checks `hasattr(module, "to_quantized")` and last-axis-divisible-by-group_size)
//! and mlx-swift's `QuantizationContainer.decode`
//! (`Libraries/MLXLMCommon/BaseConfiguration.swift:139-171`, which only decodes
//! group_size/bits/mode + per-layer overrides) — both trust mlx-c.
//!
//! See [project memory `feedback_match_official_binding_design`]: mlxrs
//! wrappers are thin forwards mirroring mlx-swift/python; we do not chase
//! mlx-core-internal hardening. Per-mode contracts (e.g. `bits ∈ {2,3,4,5,6,8}`
//! for affine — `mlx/mlx/ops.cpp:4745-4750`; `mxfp4` requires `(32, 4)`,
//! `nvfp4` requires `(16, 4)` — `mlx/mlx/ops.cpp:4808-4823`) are upstream of
//! this module.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer};

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Error, Result},
  lm::load::Weights,
  ops,
};

/// The set of MLX quantization modes — mlx-swift's `QuantizationMode`
/// (`mlx-swift/Source/MLX/Ops.swift:1097-1124`), serialized as the lowercase
/// tag string mlx-c expects (`"affine"` / `"mxfp4"` / `"mxfp8"` / `"nvfp4"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantMode {
  /// Asymmetric grouped affine quantization (mlx default,
  /// `mlx-swift/Ops.swift:1109`). Per-group `scale` + `bias`; the only
  /// mode that yields a `biases` output.
  Affine,
  /// MX (Microscaling) FP4 — `mlx-swift/Ops.swift:1115`.
  Mxfp4,
  /// MX (Microscaling) FP8 — `mlx-swift/Ops.swift:1121`.
  Mxfp8,
  /// NVIDIA FP4 — `mlx-swift/Ops.swift:1123`.
  Nvfp4,
}

impl Default for QuantMode {
  /// `affine` — mlx-swift `Quantization.mode` default
  /// (`BaseConfiguration.swift:46`: `_mode ?? .affine`).
  fn default() -> Self {
    QuantMode::Affine
  }
}

impl QuantMode {
  /// The mlx-c mode tag string (the wire format mlx-c expects). Stable
  /// snake-case lower — matches the `serde(rename_all = "lowercase")` form
  /// in [`QuantMode`]'s `Deserialize` impl, so serialize/deserialize roundtrip.
  pub fn as_mlx_str(self) -> &'static str {
    match self {
      QuantMode::Affine => "affine",
      QuantMode::Mxfp4 => "mxfp4",
      QuantMode::Mxfp8 => "mxfp8",
      QuantMode::Nvfp4 => "nvfp4",
    }
  }
}

/// Quantization parameters for one (set of) weight(s) — mlx-swift
/// `BaseConfiguration.Quantization` (`BaseConfiguration.swift:22-56`).
///
/// Mirrors the swift struct verbatim: `group_size` and `bits` are required;
/// `mode` is optional in the on-disk JSON (a missing `"mode"` key defaults
/// to [`QuantMode::Affine`], swift's `_mode ?? .affine`). Extra keys in the
/// JSON block (e.g. legacy `quant_method`) are ignored — the deserializer
/// for the container ([`PerLayerQuantization`]) strips them before the
/// per-layer scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub struct Quantization {
  /// Elements per quantization group (`mlx.core.quantize` `group_size`).
  pub group_size: i32,
  /// Bits per weight (`mlx.core.quantize` `bits`).
  pub bits: i32,
  /// The quantization scheme — swift `BaseConfiguration.Quantization._mode`
  /// (`BaseConfiguration.swift:40`); defaults to [`QuantMode::Affine`].
  #[serde(default)]
  pub mode: QuantMode,
}

impl Quantization {
  /// A convenience builder for the common `affine` case (mlx-lm's default).
  pub fn affine(group_size: i32, bits: i32) -> Self {
    Self {
      group_size,
      bits,
      mode: QuantMode::Affine,
    }
  }
}

/// The per-layer override the [`PerLayerQuantization`] map carries — mlx-swift
/// `BaseConfiguration.QuantizationOption` (`BaseConfiguration.swift:58-64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationOption {
  /// "Do not quantize this layer." Encoded in `config.json` as a literal
  /// `false` for the layer path (`BaseConfiguration.swift:60-61`,
  /// `BaseConfiguration.swift:157-161`).
  Skip,
  /// "Quantize this layer with these specific parameters." Encoded as a
  /// nested `{ group_size, bits, [mode] }` object
  /// (`BaseConfiguration.swift:62-63`, `BaseConfiguration.swift:162-166`).
  Quantize(Quantization),
}

/// A container for per-layer [`Quantization`] settings — mlx-swift
/// `BaseConfiguration.PerLayerQuantization`
/// (`BaseConfiguration.swift:66-101`).
///
/// `quantization` is the **default** applied to any layer not explicitly
/// named in `per_layer`; `None` means "no default — only the explicitly
/// listed layers are quantized" (swift's optional default,
/// `BaseConfiguration.swift:71-73`). `per_layer` maps the layer path (e.g.
/// `"model.embed_tokens"`) to the override
/// ([`Skip`](QuantizationOption::Skip) or
/// [`Quantize`](QuantizationOption::Quantize) — `BaseConfiguration.swift:75-77`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PerLayerQuantization {
  /// The default quantization for any layer not explicitly named in
  /// `per_layer` — swift `quantization` (`BaseConfiguration.swift:72-73`).
  pub quantization: Option<Quantization>,
  /// Path → override. Empty when the on-disk JSON only carried the global
  /// `{ group_size, bits, [mode] }` (no per-layer keys — the common case).
  pub per_layer: HashMap<String, QuantizationOption>,
}

impl PerLayerQuantization {
  /// Build a flat-default [`PerLayerQuantization`] from a single global
  /// [`Quantization`] (no per-layer overrides). Convenience for callers
  /// that already have a [`Quantization`] in hand and want the default
  /// "quantize every eligible layer with these params" behavior.
  pub fn from_global(q: Quantization) -> Self {
    Self {
      quantization: Some(q),
      per_layer: HashMap::new(),
    }
  }

  /// Resolve the [`Quantization`] for one layer path — mlx-swift
  /// `PerLayerQuantization.quantization(layer:)`
  /// (`BaseConfiguration.swift:86-100`).
  ///
  /// Returns `Some(q)` to quantize this layer with `q` (an explicit
  /// override OR the global default), `None` to skip it (an explicit
  /// [`Skip`](QuantizationOption::Skip) override OR no global default).
  pub fn quantization_for(&self, layer: &str) -> Option<Quantization> {
    match self.per_layer.get(layer) {
      Some(QuantizationOption::Skip) => None,
      Some(QuantizationOption::Quantize(q)) => Some(*q),
      None => self.quantization,
    }
  }
}

/// Deserialize a [`PerLayerQuantization`] from the interleaved JSON
/// shape mlx checkpoints emit — mlx-swift
/// `BaseConfiguration.QuantizationContainer.init(from:)`
/// (`BaseConfiguration.swift:139-171`).
///
/// The on-disk shape (`BaseConfiguration.swift:103-118`):
///
/// ```json
/// "quantization": {
///     "group_size": 64,
///     "bits": 4,
///     "mode": "affine",
///     "model.embed_tokens": { "group_size": 32, "bits": 4 },
///     "model.layers.0.self_attn.q_norm": false
/// }
/// ```
///
/// Global keys (`group_size` / `bits` / `mode`) live side-by-side with
/// arbitrary per-layer keys; the deserializer separates them by name. The
/// recognized non-layer keys are exactly the swift list
/// (`BaseConfiguration.swift:148-154`): the three [`Quantization`] keys
/// plus the legacy HF interop tags (`quant_method`, `linear_class`,
/// `quantization_mode`). Any other key is a layer path. A `false` value
/// becomes a [`Skip`](QuantizationOption::Skip); a JSON object becomes a
/// [`Quantize`](QuantizationOption::Quantize) with a nested
/// [`Quantization`].
impl<'de> Deserialize<'de> for PerLayerQuantization {
  fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    use serde::de::Error as _;
    use serde_json::Value;

    let value = Value::deserialize(deserializer)?;
    let Value::Object(map) = value else {
      return Err(D::Error::custom("quantization block must be a JSON object"));
    };

    // mlx-swift `QuantizationContainer.init(from:)` reserved non-layer keys
    // (`BaseConfiguration.swift:148-154`): the three `Quantization` keys
    // plus the HF/MLX-community interop tags that some checkpoints planted
    // alongside the quant block (`mlx-community/bitnet-b1.58-2B-4T-4bit`).
    const RESERVED: &[&str] = &[
      "group_size",
      "bits",
      "mode",
      "quant_method",
      "linear_class",
      "quantization_mode",
    ];
    let is_reserved = |k: &str| RESERVED.contains(&k);

    // 1) Parse the global `Quantization` from the same level. Mirroring
    //    swift `QuantizationContainer.init(from:)` which calls
    //    `Quantization(from: decoder)` unconditionally and lets it throw if
    //    the keys are missing (`BaseConfiguration.swift:141`; `bits` /
    //    `group_size` are `let` non-optional in the swift struct,
    //    `BaseConfiguration.swift:34-37`), both keys are REQUIRED at the
    //    top level of any `"quantization"` block — a missing key here is a
    //    deserialize error, not a silent `None`. Per-layer-only configs
    //    without a global default are simply not a valid swift /
    //    mlx-checkpoint shape.
    if !map.contains_key("group_size") {
      return Err(D::Error::custom(
        "`quantization` block is missing required key `group_size`",
      ));
    }
    if !map.contains_key("bits") {
      return Err(D::Error::custom(
        "`quantization` block is missing required key `bits`",
      ));
    }
    // Build a stripped object that contains ONLY the global keys, so
    // serde_json's `Quantization` deserializer doesn't choke on per-layer
    // keys it doesn't expect (the swift `Quantization(from: decoder)` only
    // reads its own three keys via `CodingKeys`).
    let mut globals = serde_json::Map::new();
    for k in ["group_size", "bits", "mode"] {
      if let Some(v) = map.get(k) {
        globals.insert(k.to_string(), v.clone());
      }
    }
    let quantization = Some(
      serde_json::from_value::<Quantization>(Value::Object(globals)).map_err(D::Error::custom)?,
    );

    // 2) Per-layer keys: everything that is not reserved.
    let mut per_layer: HashMap<String, QuantizationOption> = HashMap::new();
    for (key, v) in &map {
      if is_reserved(key) {
        continue;
      }
      // mlx-swift `if let f = try? container.decode(Bool.self, forKey: key)
      // { if !f { ... .skip } }` — only `false` becomes Skip; a `true` is
      // silently ignored (swift falls through the `if !f` branch); this
      // port mirrors that.
      let opt = match v {
        Value::Bool(false) => QuantizationOption::Skip,
        Value::Bool(true) => continue,
        // mlx-swift `else { perLayerQuantization[key] = .quantize(try
        // container.decode(Quantization.self, forKey: key)) }`.
        Value::Object(_) => {
          let q = serde_json::from_value::<Quantization>(v.clone()).map_err(D::Error::custom)?;
          QuantizationOption::Quantize(q)
        }
        // mlx-swift's `try? container.decode(Bool.self, ...)` returns nil
        // for any non-Bool / non-Object value, falling into the `else`
        // arm which `try`s a `Quantization.decode` — that throws on
        // anything but an object, propagating as a decode error. Mirror:
        // an unrecognized scalar (number / string / null / array) at a
        // layer key is a decode error.
        other => {
          return Err(D::Error::custom(format!(
            "per-layer quantization value at {key:?} must be `false` or a quantization object, got {other:?}"
          )));
        }
      };
      per_layer.insert(key.clone(), opt);
    }

    Ok(PerLayerQuantization {
      quantization,
      per_layer,
    })
  }
}

/// Parse the `"quantization"` block out of an in-memory `config.json` text.
///
/// Mirrors mlx-swift `BaseConfiguration.quantizationContainer`
/// (`BaseConfiguration.swift:188-189` + `CodingKeys.quantizationContainer`
/// at `BaseConfiguration.swift:207`): the top-level `config.json` key the
/// container is decoded from is `"quantization"`. Returns `Ok(None)` if
/// the config is valid JSON but has no `"quantization"` key (the
/// non-quantized checkpoint case); `Err(Backend)` if the config is not
/// valid JSON or the `"quantization"` block fails to deserialize as a
/// [`PerLayerQuantization`].
///
/// This is the read-path entry point for callers that want the
/// per-layer-aware [`PerLayerQuantization`] (the swift-faithful schema);
/// [`crate::lm::load::Config::quantization`] also exposes the global
/// [`Quantization`] for the simpler "just the defaults" case.
pub fn parse_quantization(config_json: &str) -> Result<Option<PerLayerQuantization>> {
  let value: serde_json::Value = serde_json::from_str(config_json).map_err(|e| Error::Backend {
    message: format!("parse_quantization: invalid config JSON: {e}"),
  })?;
  let Some(block) = value.get("quantization") else {
    return Ok(None);
  };
  let plq: PerLayerQuantization =
    serde_json::from_value(block.clone()).map_err(|e| Error::Backend {
      message: format!("parse_quantization: invalid `quantization` block: {e}"),
    })?;
  Ok(Some(plq))
}

/// The `.weight` suffix mlx stores Linear / Embedding / SwitchLinear dense
/// matrices under in the flat weight map. Stripping it from a weight-map
/// key yields the layer **path** the per-layer predicate / per-layer
/// override is keyed by (mlx-lm's `path` arg to `class_predicate(path,
/// module)`, `utils.py:349`).
const WEIGHT_SUFFIX: &str = ".weight";
const SCALES_SUFFIX: &str = ".scales";
const BIASES_SUFFIX: &str = ".biases";

/// The architecture-supplied eligibility predicate
/// [`quantize_weights`] consults to decide which weight paths are
/// quantization targets — the structural analogue of mlx-lm's
/// `hasattr(module, "to_quantized")` check (`utils.py:824`).
///
/// Called with `(layer_path, weight_array)` for every key ending in
/// `.weight` (the layer path is the key with the `.weight` suffix
/// stripped). Returning `false` makes that weight pass through
/// unchanged — even if its shape / `group_size` / per-layer override
/// would otherwise make it eligible. mlxrs has no module tree to
/// consult, so this caller-supplied predicate is the source of truth
/// for which weights belong to a quantizable module class
/// (Linear / Embedding / SwitchLinear in mlx-lm).
///
/// See [`default_eligible`] for the unconditional-true fallback that
/// reproduces the historical "every `.weight` is a candidate" behavior.
pub type Eligible<'a> = dyn Fn(&str, &Array) -> bool + 'a;

/// The "every `.weight` is a candidate" eligibility predicate — the
/// pre-Codex-fix default behavior. Pass this to [`quantize_weights`]
/// when the caller does not have an architecture-specific allowlist
/// and wants every weight that passes the structural guards
/// (suffix / rank ≥ 2 / `last_dim % group_size == 0`) to be quantized.
///
/// Prefer a tighter caller-supplied closure when one is available;
/// mlx-lm's `wrapped_predicate` (`utils.py:823`) only returns true
/// for modules that expose `to_quantized` (the Linear / Embedding /
/// SwitchLinear set), so any future architecture weight named
/// `*.weight` that is not in that module class will be quantized
/// anyway under this default — producing a checkpoint no dense layer
/// can load. Use the explicit allowlist whenever the architecture is
/// known.
pub fn default_eligible(_path: &str, _weight: &Array) -> bool {
  true
}

/// Classification of a `<layer_path>.weight` key's quantization siblings
/// (`.scales` / `.biases`) in the input weight map.
///
/// [`quantize_weights`] consults [`classify_triple`] BEFORE the
/// eligibility / per-layer / shape gates, so the sibling-collision check
/// fires uniformly for every prospective quantization target, not only
/// the ones the rest of the chain happens to select.
enum TripleClass {
  /// No `.scales` or `.biases` sibling — this is a fresh dense weight;
  /// proceed to the eligibility predicate + structural guards +
  /// quantize.
  Absent,
  /// A structurally valid already-quantized triple. mlx-lm
  /// `class_predicate` (`utils.py:349-355`) gates on `f"{p}.scales" in
  /// weights` as the signal that the checkpoint already pre-quantized
  /// this layer. Per the [module-level validation contract](self#validation-contract),
  /// this performs only basic shape sanity (the checks below); per-mode
  /// bits/group_size pairings, the scales-last-axis invariant
  /// (`mlx/mlx/ops.cpp:107`) and scale dtypes are validated by mlx-c at
  /// the [`crate::ops::quantized::quantize`] /
  /// [`crate::ops::quantized::dequantize`] call site and surface as
  /// recoverable [`Error::Backend`].
  ///
  /// Checks enforced here (faithful to mlx-lm / mlx-swift loader paths,
  /// which validate only the structural shape):
  ///
  /// - `.weight` dtype is `uint32` (packed quantized — both `affine`
  ///   and the `fp` modes write a `uint32` packed matrix; a float
  ///   `.weight` next to a `.scales` is the orphan case).
  /// - `.weight` rank ≥ 2 (rank-0/1 next to a `.scales` is not a layout
  ///   mlx's `quantize` can have produced).
  /// - `.scales` rank equals `.weight` rank, and the leading dims (all
  ///   but the last) match — mlx preserves the leading shape across
  ///   `quantize` and `mlx-c`'s `validate_quantized_input` enforces it
  ///   (`mlx/mlx/ops.cpp:97-105`).
  /// - `.biases` arity matches the resolved mode (`mlx/ops.cpp:4908-4951`):
  ///   `affine` REQUIRES `.biases` (3-output `affine_quantize`,
  ///   `mlx/ops.cpp:4793-4798`); `mxfp4` / `mxfp8` / `nvfp4` MUST NOT
  ///   carry `.biases` (2-output `fp_quantize`, `mlx/ops.cpp:4890,4898-4904`).
  ///   This arity is a *layout* contract (which keys exist), not a
  ///   per-mode params contract; mlx-c does not check it because the
  ///   shape of the call (`dequantize(w, scales, biases?, ...)`) already
  ///   encodes it at the Rust binding site.
  ///
  /// Pass-through unchanged.
  Valid,
  /// A `.scales` and/or `.biases` sibling exists but does NOT match
  /// mlx's quantized layout — an orphan or a mismatch from a corrupted
  /// / out-of-sync checkpoint. The message names the offending path and
  /// the specific inconsistency; the caller surfaces it as
  /// [`Error::Backend`].
  Invalid(String),
}

/// Inspect `<layer_path>.scales` and `<layer_path>.biases` in `weights`
/// and classify the triple as [`Absent`](TripleClass::Absent),
/// [`Valid`](TripleClass::Valid) (mlx-quantized layout) or
/// [`Invalid`](TripleClass::Invalid) (orphan / shape / dtype mismatch).
///
/// `layer_weight` is the `<layer_path>.weight` array (the caller has
/// already stripped the suffix); `cfg` carries the global +
/// per-layer-override [`Quantization`] parameters used to resolve the
/// triple's mode (which determines the `.biases` arity, see
/// [`TripleClass::Valid`]).
///
/// **Precondition.** `cfg` must carry a resolvable [`Quantization`] for
/// `layer_path` whenever a triple is present: either via the global
/// `cfg.quantization` (the common case — Fix 2 enforces that any parsed
/// `"quantization"` block contains `group_size` + `bits`) or via a
/// per-layer [`QuantizationOption::Quantize`] override. A per-layer
/// [`QuantizationOption::Skip`] for `layer_path` means the layer was
/// intentionally not quantized — any sibling `.scales` / `.biases` at
/// that path is therefore a stale collision (returned as
/// [`Invalid`](TripleClass::Invalid), not [`Valid`](TripleClass::Valid)).
/// A `cfg.quantization == None` with no per-layer override leaves no
/// way to resolve the triple's mode; this should not arise in production
/// (it would mean `quantize_weights` was called without any quantization
/// params at all, in which case there is nothing for it to do), but it
/// is treated as [`Invalid`](TripleClass::Invalid) defensively.
///
/// See [`TripleClass`] for the exact invariants enforced, the
/// [validation contract](self#validation-contract) for what is delegated
/// to mlx-c, and the [Sibling-name reservation](self#sibling-name-reservation)
/// section for the surrounding contract.
fn classify_triple(
  weights: &Weights,
  layer_path: &str,
  layer_weight: &Array,
  cfg: &PerLayerQuantization,
) -> TripleClass {
  let scales_key = format!("{layer_path}{SCALES_SUFFIX}");
  let biases_key = format!("{layer_path}{BIASES_SUFFIX}");
  let scales = weights.get(&scales_key);
  let biases = weights.get(&biases_key);

  match (scales, biases) {
    // No siblings at all — a fresh dense weight. Proceed to the rest of
    // the chain (eligibility / shape gates / quantize).
    (None, None) => TripleClass::Absent,
    // Orphan `.biases` with no `.scales`. mlx `affine_quantize`
    // always writes `.scales` alongside `.biases`
    // (`mlx/ops.cpp:4793-4798`); a `.biases` alone is never a valid
    // mlx-produced triple. The `fp` schemes (`mxfp4`/`mxfp8`/`nvfp4`)
    // don't write `.biases` at all (`mlx/ops.cpp:4898-4900`), so this
    // can't be a non-affine triple either.
    (None, Some(_)) => TripleClass::Invalid(format!(
      "quantize_weights: layer {layer_path}: input has a stale `{biases_key}` \
       with no matching `{scales_key}` (mlx `quantize` always writes `.scales` \
       alongside `.biases`); refusing to silently overwrite the generated bias"
    )),
    // `.scales` present (with or without `.biases`). Validate the
    // layout matches what mlx's `quantize` produces — if not, it's
    // an orphan `.scales` next to a dense weight, or a corrupted
    // shape/dtype mismatch.
    (Some(s), b_opt) => {
      // Resolve the per-layer `Quantization` for this path. A per-layer
      // `Skip` (or a missing global default with no override) leaves no
      // valid quantization params for the layer; a pre-existing triple
      // at that path is therefore a stale collision — Invalid, not
      // Valid (see the function-level "Precondition" doc above).
      let q = match cfg.per_layer.get(layer_path) {
        Some(QuantizationOption::Skip) => {
          return TripleClass::Invalid(format!(
            "quantize_weights: layer {layer_path}: input carries `{scales_key}` \
             but the per-layer config marks this layer as `Skip` (not \
             quantized) — refusing to silently treat the stale triple as a \
             valid already-quantized layer"
          ));
        }
        Some(QuantizationOption::Quantize(q)) => *q,
        None => match cfg.quantization {
          Some(q) => q,
          None => {
            return TripleClass::Invalid(format!(
              "quantize_weights: layer {layer_path}: input carries \
               `{scales_key}` but `cfg` has no global `Quantization` and no \
               per-layer override for this layer — cannot resolve expected \
               `.scales` shape (this should not arise in production: any \
               parsed `quantization` block carries `group_size` + `bits`)"
            ));
          }
        },
      };
      // Per-mode `.biases` arity: mlx `quantize` dispatches on mode and
      // the resulting triple's bias slot is fully determined by it
      // (`mlx/ops.cpp:4908-4951`):
      //   - `affine` → `affine_quantize` returns `{w_q, scales, biases}`
      //     (3 outputs, `mlx/ops.cpp:4793-4798`); `.biases` is REQUIRED.
      //   - `mxfp4` / `mxfp8` / `nvfp4` → `fp_quantize` returns
      //     `{w_q, scales}` (2 outputs, `mlx/ops.cpp:4890,4898-4904`);
      //     `.biases` MUST be absent — these are scale-only formats.
      // A shape/dtype-aligned `.biases` next to a `mxfp*`/`nvfp4` triple
      // is a stale sibling from a different mode and would silently
      // corrupt `dequantize`; an `affine` triple with no `.biases` is
      // structurally incomplete and would crash mlx's `affine_dequantize`.
      // Validate the arity BEFORE the per-array shape/dtype checks so the
      // failure cites the offending mode, not a downstream shape mismatch.
      match (q.mode, b_opt) {
        (QuantMode::Affine, None) => {
          return TripleClass::Invalid(format!(
            "quantize_weights: layer {layer_path}: `affine` mode \
             (bits={}, group_size={}) requires `{biases_key}` alongside \
             `{scales_key}` (mlx `affine_quantize` always writes \
             `{{w_q, scales, biases}}`, `mlx/ops.cpp:4793-4798`), but the \
             input carries no `.biases` — this is a structurally incomplete \
             affine triple",
            q.bits, q.group_size
          ));
        }
        (QuantMode::Mxfp4 | QuantMode::Mxfp8 | QuantMode::Nvfp4, Some(_)) => {
          return TripleClass::Invalid(format!(
            "quantize_weights: layer {layer_path}: `{}` mode is scale-only \
             (mlx `fp_quantize` writes `{{w_q, scales}}` with no biases, \
             `mlx/ops.cpp:4890,4898-4904`), but the input carries a stale \
             `{biases_key}` — refusing to silently retain a bias from a \
             different (affine) mode",
            q.mode.as_mlx_str()
          ));
        }
        // `(Affine, Some(_))` falls through to the existing
        // shape/dtype validation against `.scales` below.
        // `(Mxfp4 | Mxfp8 | Nvfp4, None)` is the expected scale-only
        // layout and proceeds to the `.weight` / `.scales` checks below.
        _ => {}
      }
      // `.weight` dtype must be `uint32` — both `affine_quantize`
      // (`mlx/ops.cpp:4795`) and `fp_quantize` (`mlx/ops.cpp:4900`)
      // write a `uint32` packed matrix. A float `.weight` means this
      // is a dense weight with a stale `.scales` orphan next to it.
      let w_dtype = match layer_weight.dtype() {
        Ok(d) => d,
        Err(e) => {
          return TripleClass::Invalid(format!(
            "quantize_weights: layer {layer_path}: cannot read `.weight` dtype \
             (required to validate already-quantized triple): {e}"
          ));
        }
      };
      if w_dtype != Dtype::U32 {
        return TripleClass::Invalid(format!(
          "quantize_weights: layer {layer_path}: input has `{scales_key}` but \
           `.weight` dtype is {w_dtype:?} (mlx-quantized `.weight` is always \
           uint32 — `mlx/ops.cpp:4795,4900`); this is a stale `.scales` orphan \
           next to a dense `.weight`, not a valid already-quantized triple"
        ));
      }
      // `.weight` rank must be ≥ 2 — mlx `quantize` requires rank ≥ 2
      // inputs (`mlx/ops.cpp:4925-4929`), so a rank-0/1 `.weight` next
      // to a `.scales` cannot be a real quantized triple even when the
      // dtype is `uint32` and the leading dims happen to match.
      let w_shape = layer_weight.shape();
      if w_shape.len() < 2 {
        return TripleClass::Invalid(format!(
          "quantize_weights: layer {layer_path}: `.weight` has rank {} \
           (shape {:?}), but mlx `quantize` requires rank ≥ 2 inputs \
           (`mlx/ops.cpp:4925-4929`); this is a malformed triple (a \
           uint32 1-D / scalar `.weight` next to a `.scales` is not a \
           layout mlx's `quantize` can have produced)",
          w_shape.len(),
          w_shape
        ));
      }
      // `.scales` rank == `.weight` rank, and the leading dims (all
      // but the last) match — mlx `quantize` preserves the leading
      // shape (`mlx/ops.cpp:4789-4798`).
      let s_shape = s.shape();
      if s_shape.len() != w_shape.len() {
        return TripleClass::Invalid(format!(
          "quantize_weights: layer {layer_path}: `{scales_key}` rank ({}) \
           does not match `.weight` rank ({}) — mlx `quantize` preserves the \
           leading shape across the packed `.weight` / `.scales` / `.biases` \
           outputs (`mlx/ops.cpp:4789-4798`)",
          s_shape.len(),
          w_shape.len()
        ));
      }
      // Leading dims (all but the last) must match. Rank ≥ 2 is
      // already enforced above, so both slices are non-empty and the
      // index is safe. This is the structural shape mlx `quantize`
      // preserves and `mlx-c`'s `validate_quantized_input` enforces
      // (`mlx/mlx/ops.cpp:97-105`); checking it here surfaces the
      // mismatch with a layer-named error before mlx-c sees it.
      if s_shape[..s_shape.len() - 1] != w_shape[..w_shape.len() - 1] {
        return TripleClass::Invalid(format!(
          "quantize_weights: layer {layer_path}: `{scales_key}` leading dims \
           {:?} do not match `.weight` leading dims {:?} — mlx `quantize` \
           preserves all-but-last dims",
          &s_shape[..s_shape.len() - 1],
          &w_shape[..w_shape.len() - 1],
        ));
      }
      // Per the [module-level validation contract](self#validation-contract):
      // per-mode bits/group_size pairings, the scales-last-axis invariant
      // (`mlx/mlx/ops.cpp:107`), and scale dtypes are validated by mlx-c
      // at the [`crate::ops::quantized::quantize`] /
      // [`crate::ops::quantized::dequantize`] call site. Faithful-port to
      // mlx-lm (`utils.py:823-835`) / mlx-swift (`BaseConfiguration.swift:139-171`)
      // does NOT duplicate those checks in the loader path.
      TripleClass::Valid
    }
  }
}

/// Quantize the eligible weights in `weights` per `cfg`, returning a new
/// [`Weights`] map.
///
/// Port of the **weight-map equivalent** of mlx-lm's `quantize_model`
/// (`mlx_lm/utils.py:774-850`): mlx-lm walks `nn.Module` leaves replacing
/// each `Linear` / `Embedding` / `SwitchLinear` with its quantized
/// counterpart (so the resulting model's `state_dict` carries
/// `<layer>.weight` (packed) + `<layer>.scales` (+ `<layer>.biases` for
/// `affine`)); mlxrs has no model-module tree, so this walks the loaded
/// weight MAP, applies the merged [`crate::ops::quantized::quantize`] (the
/// #19 op — **not** a re-implementation) to every key matching the
/// predicate (see [module docs](self#predicate-which-weight-keys-get-quantized)),
/// and writes out the resulting `(w_q, scales, biases?)` triple under the
/// same `<path>.weight` / `<path>.scales` / `<path>.biases` names mlx's
/// `QuantizedLinear` uses (`mlx/python/mlx/nn/layers/quantized.py:134-137`).
///
/// ## Eligibility predicate
///
/// `eligible` is the caller-supplied architecture allowlist — the
/// structural analogue of mlx-lm's `hasattr(module, "to_quantized")`
/// check (`utils.py:824`). mlxrs has no module tree, so the caller's
/// closure is the source of truth for which weight paths are
/// quantization targets. Use [`default_eligible`] to reproduce the
/// historical "every `.weight` is a candidate" behavior; prefer a
/// tighter architecture-specific closure when available (the historical
/// default may quantize a future `.weight` that is not a Linear /
/// Embedding / SwitchLinear target, producing a checkpoint no dense
/// layer can load — mirroring mlx-lm's wrapped_predicate is the
/// recommended pattern).
///
/// Per-layer overrides: a [`QuantizationOption::Skip`] passes that
/// weight through unchanged; a [`QuantizationOption::Quantize`]
/// substitutes its own `group_size` / `bits` / `mode` for the global
/// default — swift's `PerLayerQuantization.quantization(layer:)`
/// semantics (`BaseConfiguration.swift:86-100`).
///
/// Already-quantized weights pass through unchanged (mlx-lm gates on
/// `f"{p}.scales" in weights`, `utils.py:349-355`); their existing
/// `.scales` / `.biases` siblings are preserved by the verbatim map copy.
///
/// ## Sibling-name reservation
///
/// When a path `P` is selected for quantization, the generated triple
/// (`P.weight` / `P.scales` / `P.biases`) reserves those names. Before
/// the eligibility / per-layer / shape gates fire, every `<path>.weight`
/// key is classified against the layout mlx's `quantize` actually
/// produces (`<path>.weight` is `uint32`; `.scales` rank matches
/// `.weight` rank with the same leading dims; `.biases` — if present —
/// has the same shape and dtype as `.scales`):
///
/// - `Valid` (a structurally consistent already-quantized triple) →
///   pass through unchanged, mlx-lm `class_predicate` semantics
///   (`utils.py:349-355`).
/// - `Invalid` (an orphan `.scales` / `.biases` with no matching
///   sibling, a `.scales` next to a dense `.weight`, or a shape/dtype
///   mismatch) → return [`Error::Backend`] naming the offending path
///   and the inconsistency. A non-deterministic overwrite by HashMap
///   iteration order — or a downstream [`dequantize_weights`]
///   corrupt-triple crash — is worse than a clear early failure.
/// - `Absent` (no siblings) → proceed to the rest of the chain.
///
/// **Failure handling.** Every quantization op is fallible
/// ([`crate::ops::quantized::quantize`] propagates mlx-c's error); a
/// failure mid-walk drops the partially-built result map and returns
/// `Err` — the input `weights` is consumed but no partial output escapes.
pub fn quantize_weights(
  weights: Weights,
  cfg: &PerLayerQuantization,
  eligible: &Eligible<'_>,
) -> Result<Weights> {
  // Out-map sized for "at most everything got quantized" (adds up to one
  // `.scales` + one `.biases` per `.weight` quantized, i.e. ≤ 3× the input
  // — a conservative upper bound).
  let mut out: Weights = HashMap::with_capacity(weights.len());

  // Two passes so the predicate sees the COMPLETE input map for the
  // triple-classification check (sibling `.scales` / `.biases` need the
  // full map up front). Pass 1 chooses which keys to quantize without
  // mutating; pass 2 does the work. mlx-lm's `tree_map_with_path` on
  // `leaf_modules()` is the module-tree analog of this two-pass shape.
  let mut to_quantize: Vec<(String, Quantization)> = Vec::new();
  for (key, arr) in &weights {
    let Some(layer_path) = key.strip_suffix(WEIGHT_SUFFIX) else {
      continue;
    };
    // FIRST: classify the prospective triple against mlx's quantized
    // layout. This runs BEFORE the eligibility / per-layer / shape
    // gates so the orphan-sibling collision check fires uniformly for
    // every `<path>.weight` key (catches the case where a dense
    // `.weight` has a stale `.scales` / `.biases` orphan next to it
    // — see [`TripleClass`] for the exact invariants). The
    // `is_already_quantized` presence-only gate (mlx-lm
    // `utils.py:349-355`) is subsumed by the
    // [`Valid`](TripleClass::Valid) branch. `cfg` is passed in so the
    // expected `.scales` last-axis can be computed from the per-layer
    // (`bits`, `group_size`) — mlx's invariant
    // `w.shape(-1) * 32 / bits == scales.shape(-1) * group_size`
    // (`mlx/ops.cpp:107`).
    match classify_triple(&weights, layer_path, arr, cfg) {
      TripleClass::Absent => {}
      TripleClass::Valid => continue,
      TripleClass::Invalid(message) => return Err(Error::Backend { message }),
    }
    // Caller-supplied eligibility — the structural analogue of mlx-lm's
    // `hasattr(module, "to_quantized")` (`utils.py:824`). Pass 1 of the
    // wrapped_predicate translation; fails the rest of the chain
    // immediately and the weight passes through unchanged.
    if !eligible(layer_path, arr) {
      continue;
    }
    // Per-layer-aware resolution (Skip wins; Quantize override wins over
    // the global default; None ⇒ skip).
    let Some(q) = cfg.quantization_for(layer_path) else {
      continue;
    };
    // mlx-lm `utils.py:826-827`: shape-rank ≥ 2 with last axis divisible by
    // `group_size`. Anything else (scalars, 1-D biases, last-axis ≠ 0 mod
    // group_size) passes through verbatim.
    let shape = arr.shape();
    if shape.len() < 2 {
      continue;
    }
    let last = *shape.last().expect("len >= 2");
    let gs = usize::try_from(q.group_size).map_err(|_| Error::ShapeMismatch {
      message: format!(
        "quantize_weights: layer {layer_path}: group_size ({}) must be a non-negative i32",
        q.group_size
      ),
    })?;
    if gs == 0 || last % gs != 0 {
      continue;
    }
    to_quantize.push((layer_path.to_string(), q));
  }

  let quantize_set: HashMap<String, Quantization> = to_quantize.into_iter().collect();

  // Pass 2: walk again, quantize the chosen ones, copy everything else
  // verbatim. Generated triple names (`<path>.weight` / `<path>.scales` /
  // `<path>.biases` for each path in `quantize_set`) were reserved by the
  // sibling-collision check in pass 1, so there is no input key that can
  // collide with them.
  for (key, arr) in weights {
    let layer_path = key.strip_suffix(WEIGHT_SUFFIX);
    let quant_match = layer_path.and_then(|p| quantize_set.get(p).map(|q| (p, *q)));
    if let Some((path, q)) = quant_match {
      let (w_q, scales, biases) =
        ops::quantized::quantize(&arr, q.group_size, q.bits, q.mode.as_mlx_str(), None)?;
      // mlx's `QuantizedLinear` stores the packed weight at
      // `<path>.weight`, the scales at `<path>.scales`, and (for `affine`)
      // the biases at `<path>.biases` —
      // `mlx/python/mlx/nn/layers/quantized.py:134-137`. Preserve the
      // names so the resulting map round-trips with mlx-lm's saved layout.
      out.insert(format!("{path}{WEIGHT_SUFFIX}"), w_q);
      out.insert(format!("{path}{SCALES_SUFFIX}"), scales);
      if let Some(b) = biases {
        out.insert(format!("{path}{BIASES_SUFFIX}"), b);
      }
    } else {
      out.insert(key, arr);
    }
  }
  Ok(out)
}

/// Inverse of [`quantize_weights`]: reconstruct dense weights from any
/// quantized triples (`<path>.weight` + `<path>.scales` [+ `<path>.biases`])
/// in `weights`, returning a new [`Weights`] map.
///
/// Port of the **weight-map equivalent** of mlx-lm's `dequantize_model`
/// (`mlx_lm/utils.py:853-896`): mlx-lm walks `nn.Module` leaves replacing
/// each `QuantizedLinear` / `QuantizedEmbedding` / `QuantizedSwitchLinear`
/// with its dense counterpart (calling `mx.dequantize(module.weight,
/// module.scales, module.biases, module.group_size, module.bits,
/// module.mode)`); this walks the weight MAP, detects triples by the
/// sibling-key shape (a `.scales` is the load-bearing indicator —
/// mlx-lm's class-isinstance check), and applies the merged
/// [`crate::ops::quantized::dequantize`] (the #19 op).
///
/// `cfg` carries the global `group_size` / `bits` / `mode` (and any
/// per-layer overrides); a triple's parameters come from
/// [`PerLayerQuantization::quantization_for`] for its layer path. A
/// missing global default with no per-layer override is a recoverable
/// [`Error::Backend`] for that triple — there is no way to dequantize
/// without parameters.
///
/// **Mode arity.** Symmetric with [`quantize_weights`]: after resolving
/// `q` for each triple, the resolved mode dictates the bias slot —
/// `affine` requires `.biases`, `mxfp4` / `mxfp8` / `nvfp4` forbid it
/// (`mlx/ops.cpp:5085-5099,5198-5210`). A mode/bias mismatch returns
/// [`Error::Backend`] for that triple. Per the
/// [module-level validation contract](self#validation-contract), other
/// per-mode checks (bits/group_size pairings, scales-last-axis, scale
/// dtypes) are delegated to mlx-c at the
/// [`crate::ops::quantized::dequantize`] call site.
///
/// Non-triple entries (no `.scales` sibling) pass through verbatim — a
/// `.weight` without a matching `.scales` is an already-dense weight, and
/// stray `.scales` / `.biases` without a `.weight` are passed through too
/// (a hostile / corrupt checkpoint shape; mlx-lm leaves them in place too —
/// `dequantize_model` only replaces *modules*, never deletes parameters).
/// Symmetric with the orphan-`.biases` guard in
/// [`quantize_weights`]'s triple classifier, the one exception is a
/// layer carrying a `uint32`-packed `.weight` plus `.biases` with NO
/// `.scales` — that combination is never a valid mlx-produced
/// quantized triple (mlx `affine_quantize` always writes `.scales`
/// alongside `.biases`, `mlx/ops.cpp:4793-4798`) and would otherwise
/// leave the `uint32`-packed `.weight` as a pass-through in the
/// dequantized output; it returns [`Error::Backend`] naming the layer
/// and the missing `.scales` instead. The guard is narrowed to the
/// `uint32` dtype signal (`mlx/ops.cpp:4795,4900`) so that a normal
/// dense Linear layer (`P.weight` F32 plus `P.biases` F32 with no
/// `P.scales`) passes through verbatim — there's no quantization
/// involvement and nothing to dequantize.
pub fn dequantize_weights(weights: Weights, cfg: &PerLayerQuantization) -> Result<Weights> {
  let mut out: Weights = HashMap::with_capacity(weights.len());

  // Symmetric with [`classify_triple`]'s orphan-`.biases` guard
  // (`(None, Some(_))` arm above): a layer with `.weight` + `.biases`
  // but NO `.scales` is never a valid mlx-produced QUANTIZED triple — mlx
  // `affine_quantize` always writes `.scales` alongside `.biases`
  // (`mlx/ops.cpp:4793-4798`) and the `fp_*` schemes write no biases
  // at all (`mlx/ops.cpp:4898-4900`). Without this guard the `.biases`
  // would fall into the pass-through branch (no triple → not staged)
  // and the `.weight` (still `uint32` packed) would ALSO pass through,
  // handing the caller a packed weight in an output it expects dense.
  //
  // The guard MUST be narrowed to the U32-packed `.weight` signal,
  // otherwise it over-rejects a perfectly normal dense Linear layer
  // (`P.weight` F32 + `P.biases` F32, no `P.scales`) — that combination
  // is a standard dense+bias layer with no quantization involvement at
  // all, and there is nothing to dequantize. We only flag when `.weight`
  // is `uint32` (the mlx-quantization signal: `mlx/ops.cpp:4795,4900`),
  // matching the dtype check in [`classify_triple`]. A dense `.weight`
  // with a sibling `.biases` and no `.scales` passes through verbatim
  // — the orphan-`.biases` concern only applies when there's a packed
  // weight that would otherwise leak unconverted.
  for key in weights.keys() {
    if let Some(path) = key.strip_suffix(BIASES_SUFFIX) {
      let scales_key = format!("{path}{SCALES_SUFFIX}");
      let weight_key = format!("{path}{WEIGHT_SUFFIX}");
      if weights.contains_key(&scales_key) {
        continue;
      }
      let Some(weight_arr) = weights.get(&weight_key) else {
        continue;
      };
      let w_dtype = weight_arr.dtype().map_err(|e| Error::Backend {
        message: format!(
          "dequantize_weights: layer {path}: cannot read `{weight_key}` dtype \
           (required to classify orphan `.biases` against packed `.weight`): {e}"
        ),
      })?;
      // Mirror `classify_triple`: only `uint32` `.weight` is the
      // mlx-packed signal (`mlx/ops.cpp:4795,4900`). Any other dtype
      // (`F32` and friends) is a dense layer — pass through.
      if w_dtype != Dtype::U32 {
        continue;
      }
      // Mirror `classify_triple` shape symmetry: a rank<2 U32 `.weight`
      // is not a real mlx-packed matrix (`mlx/ops.cpp:4925-4929` requires
      // rank ≥ 2), so don't flag the orphan `.biases` as a quantization
      // hazard against it.
      if weight_arr.shape().len() < 2 {
        continue;
      }
      return Err(Error::Backend {
        message: format!(
          "dequantize_weights: layer {path}: input has a stale \
           `{path}{BIASES_SUFFIX}` with no matching `{path}{SCALES_SUFFIX}` \
           (mlx `quantize` always writes `.scales` alongside `.biases`, \
           `mlx/ops.cpp:4793-4798`); this is a structurally incomplete \
           triple — refusing to silently leave the `uint32`-packed \
           `{path}{WEIGHT_SUFFIX}` as a pass-through in the dequantized output"
        ),
      });
    }
  }

  // Identify the triples upfront so the SECOND walk can consume the input
  // map without rechecking sibling presence per key.
  let mut triple_set: HashMap<String, ()> = HashMap::new();
  for key in weights.keys() {
    if let Some(path) = key.strip_suffix(".scales") {
      let weight_key = format!("{path}.weight");
      if weights.contains_key(&weight_key) {
        triple_set.insert(path.to_string(), ());
      }
    }
  }

  // Stage the triple components by path, then dequantize once we have all
  // three (or two, for bias-less). Consume the input map so each Array is
  // moved exactly once (no clone).
  type StagedTriple = (Option<Array>, Option<Array>, Option<Array>);
  let mut staged: HashMap<String, StagedTriple> = HashMap::new();
  for (key, arr) in weights {
    // Try to attribute `key` to a triple component first; if not, it's a
    // pass-through. Each branch consumes `key`/`arr` only on match.
    let component = if let Some(path) = key.strip_suffix(WEIGHT_SUFFIX) {
      triple_set.contains_key(path).then(|| (path.to_string(), 0))
    } else if let Some(path) = key.strip_suffix(".scales") {
      triple_set.contains_key(path).then(|| (path.to_string(), 1))
    } else if let Some(path) = key.strip_suffix(".biases") {
      triple_set.contains_key(path).then(|| (path.to_string(), 2))
    } else {
      None
    };
    if let Some((path, slot)) = component {
      let entry = staged.entry(path).or_insert((None, None, None));
      match slot {
        0 => entry.0 = Some(arr),
        1 => entry.1 = Some(arr),
        2 => entry.2 = Some(arr),
        _ => unreachable!(),
      }
    } else {
      // Not part of any triple — pass through verbatim.
      out.insert(key, arr);
    }
  }

  for (path, (w_opt, s_opt, b_opt)) in staged {
    let w = w_opt.ok_or_else(|| Error::Backend {
      message: format!("dequantize_weights: layer {path} is missing `.weight`"),
    })?;
    let scales = s_opt.ok_or_else(|| Error::Backend {
      message: format!("dequantize_weights: layer {path} is missing `.scales`"),
    })?;
    let q = cfg.quantization_for(&path).ok_or_else(|| Error::Backend {
      message: format!(
        "dequantize_weights: no quantization parameters for layer {path} \
           (no global default and no per-layer override)"
      ),
    })?;
    // Symmetric mode-arity check with [`quantize_weights`] /
    // [`classify_triple`]: mlx `dequantize` dispatches on mode and the
    // expected bias slot is fully determined by it (`mlx/ops.cpp:4908-4951`).
    // `affine` requires `.biases` (3-input `affine_dequantize`,
    // `mlx/ops.cpp:5085-5099`); `mxfp4` / `mxfp8` / `nvfp4` forbid `.biases`
    // (2-input `fp_dequantize`, `mlx/ops.cpp:5198-5210`). Forwarding a
    // mode/bias mismatch to mlx-c would corrupt the dequantized output (an
    // `affine` triple with no `.biases` reconstructs without the zero-point,
    // and an `fp_*` triple with a stale `.biases` retains a bias from a
    // different mode); a clear early failure is better than silent corruption.
    match (q.mode, b_opt.as_ref()) {
      (QuantMode::Affine, None) => {
        return Err(Error::Backend {
          message: format!(
            "dequantize_weights: layer {path}: `affine` mode \
             (bits={}, group_size={}) requires `{path}.biases` alongside \
             `{path}.scales` (mlx `affine_dequantize` takes \
             `{{w_q, scales, biases}}`, `mlx/ops.cpp:5085-5099`), but the \
             input carries no `.biases` — this is a structurally incomplete \
             affine triple",
            q.bits, q.group_size
          ),
        });
      }
      (QuantMode::Mxfp4 | QuantMode::Mxfp8 | QuantMode::Nvfp4, Some(_)) => {
        return Err(Error::Backend {
          message: format!(
            "dequantize_weights: layer {path}: `{}` mode is scale-only \
             (mlx `fp_dequantize` takes `{{w_q, scales}}` with no biases, \
             `mlx/ops.cpp:5198-5210`), but the input carries a stale \
             `{path}.biases` — refusing to silently retain a bias from a \
             different (affine) mode",
            q.mode.as_mlx_str()
          ),
        });
      }
      // `(Affine, Some(_))` / `(Mxfp4 | Mxfp8 | Nvfp4, None)` are the
      // valid arity arms — fall through to the `dequantize` call.
      _ => {}
    }
    let dense = ops::quantized::dequantize(
      &w,
      &scales,
      b_opt.as_ref(),
      q.group_size,
      q.bits,
      q.mode.as_mlx_str(),
      None,
      None,
    )?;
    out.insert(format!("{path}.weight"), dense);
  }

  Ok(out)
}

// ─────────────────────────── AutoAWQ / GPTQ on-load conversion ───────────────────────────

/// The AutoAWQ / GPTQ on-load quantization parameters carried by the
/// `quantization_config` block of the upstream `config.json` —
/// mirroring the dict shape `_transform_awq_weights` consumes
/// (`mlx-lm/mlx_lm/utils.py:83-172`) and the swift `ParoQuantConfig`
/// reader (`mlx-swift-lm/Libraries/MLXLMCommon/ParoQuant/ParoQuantLoader.swift:24-40`).
///
/// Layout the upstream checkpoint emits:
///
/// ```json
/// {
///   "quantization_config": {
///     "quant_method": "awq",
///     "bits": 4,
///     "group_size": 128,
///     "zero_point": true,
///     "version": "gemm"
///   }
/// }
/// ```
///
/// `bits`, `group_size`, `zero_point`, `version` follow the AutoAWQ
/// convention (`autoawq` library's `quantization_config` writer).
/// Only `bits = 4` is supported by mlx-lm's converter
/// (`utils.py:88-89`); other values are rejected at
/// [`transform_awq_weights`] time. `zero_point` defaults to `true`
/// (asymmetric, the AutoAWQ default; the python converter handles
/// both paths — `utils.py:135-151`). `version` is interop metadata
/// preserved from the checkpoint (e.g. `"gemm"`); the converter does
/// not switch on it (mlx-lm consumes the same packed layout
/// regardless), but it round-trips through [`AwqLoadConfig`] so a
/// downstream caller can inspect it.
///
/// The `quant_method` discriminator (`"awq"` / `"gptq"` /
/// `"paroquant"`) is **not** carried by this struct — the caller
/// (loader) inspects it before deciding to invoke
/// [`transform_awq_weights`] (`utils.py:370-391`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct AwqLoadConfig {
  /// Bits per weight. mlx-lm rejects anything other than `4`
  /// (`utils.py:88-89`).
  #[serde(default = "AwqLoadConfig::default_bits")]
  pub bits: u32,
  /// Elements per quantization group. AutoAWQ default is `128`
  /// (`utils.py:90`).
  #[serde(default = "AwqLoadConfig::default_group_size")]
  pub group_size: u32,
  /// `true` = asymmetric (per-group `qzeros` carried alongside
  /// `scales`); `false` = symmetric (implicit `2^(bits-1)` zero,
  /// `utils.py:149-151`). AutoAWQ default is `true`.
  #[serde(default = "AwqLoadConfig::default_zero_point")]
  pub zero_point: bool,
  /// AutoAWQ checkpoint version tag (`"gemm"` / `"gemv"` / ...). Not
  /// consumed by the converter; preserved for caller inspection.
  #[serde(default)]
  pub version: String,
}

impl AwqLoadConfig {
  fn default_bits() -> u32 {
    4
  }
  fn default_group_size() -> u32 {
    128
  }
  fn default_zero_point() -> bool {
    true
  }
}

impl Default for AwqLoadConfig {
  /// mlx-lm / AutoAWQ defaults (`utils.py:87,90`,
  /// `mlx-swift-lm/.../ParoQuantLoader.swift:36-39`).
  fn default() -> Self {
    Self {
      bits: Self::default_bits(),
      group_size: Self::default_group_size(),
      zero_point: Self::default_zero_point(),
      version: String::new(),
    }
  }
}

/// AutoAWQ packing constants — fixed at `bits = 4` (the only width
/// mlx-lm's on-load converter accepts, `utils.py:88-89`). Splitting
/// the constants out keeps the `transform_awq_weights` body focused
/// on the layout logic rather than the magic numbers.
const AWQ_BITS: u32 = 4;
/// `pack_factor = 32 // bits` (`utils.py:74,115`) — 8 nibbles per `uint32`.
const AWQ_PACK_FACTOR: usize = 32 / (AWQ_BITS as usize);
/// `mask = (1 << bits) - 1` (`utils.py:77`) — the nibble extractor.
const AWQ_NIBBLE_MASK: u32 = (1 << AWQ_BITS) - 1;
/// AutoAWQ's per-nibble bit positions inside each packed `uint32`,
/// `[0, 4, 1, 5, 2, 6, 3, 7] * bits` (`utils.py:78`).
///
/// AutoAWQ stores the 8 nibbles of each output element in the scrambled
/// order `[0, 2, 4, 6, 1, 3, 5, 7]` (the forward "AWQ reorder").
/// Reading them out via this shift table — the inverse permutation
/// `[0, 4, 1, 5, 2, 6, 3, 7]` scaled by `bits` — places each nibble back
/// at its natural sequential position. So the single `(qweight >> shifts)
/// & mask` step in [`unpack_awq_weights`] both unpacks AND undoes the
/// AWQ scramble in one pass — no follow-up `take`/`gather` is needed.
/// (The swift `ParoQuantLoader.unpackAndReorder` does it in two steps
/// — unpack with `arange(8) * bits`, then `take(inverseReorder)` — but
/// the algebraic result is identical.)
// Spelled-out vs computed: `[0,4,1,5,2,6,3,7].map(|i| i * AWQ_BITS)`,
// inlined so clippy's `identity_op` doesn't fire on the `0 * X` term.
// AWQ_BITS = 4, so `i * 4` for the inverse-permutation indices.
const AWQ_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];
// Compile-time assertion that the spelled-out table tracks `AWQ_BITS`.
// If `AWQ_BITS` ever changes from 4, this will fail to compile and force
// the table to be regenerated.
const _: () = assert!(AWQ_BITS == 4 && AWQ_SHIFTS[1] == 4 * AWQ_BITS);

/// Unpack an AutoAWQ-packed 4-bit `qweight` (`uint32`, 8 nibbles per
/// element) into the dense natural-order layout — port of
/// `_unpack_awq_weights` (`mlx-lm/mlx_lm/utils.py:72-82`).
///
/// `qweight` must be a 2-D `uint32` array of shape `[rows, packed_cols]`.
/// Output is the same dtype, shape `[rows, packed_cols * 8]`, with each
/// position holding the 4-bit nibble in `[0, 15]`. The output dtype
/// preserves the input's (`uint32`) — mlx-lm leaves the cast to the
/// caller (`utils.py:79-80`); `transform_awq_weights` casts to `uint32`
/// explicitly before its repack.
///
/// The shift table [`AWQ_SHIFTS`] folds the AutoAWQ packing-reorder
/// into the unpack: a single `(qweight >> shifts) & mask` over the
/// scaled inverse permutation yields the nibbles in their natural
/// sequential order. See [`AWQ_SHIFTS`] for the algebra and the
/// equivalent two-step swift form.
///
/// Mirrors the mlx-lm 2-D contract verbatim; non-2D inputs are a
/// [`Error::ShapeMismatch`] (the python version would `ValueError`
/// during the trailing `.reshape(out_features, in_features)`).
pub fn unpack_awq_weights(qweight: &Array) -> Result<Array> {
  let shape = qweight.shape();
  if shape.len() != 2 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "unpack_awq_weights: expected 2-D qweight `[rows, packed_cols]`, got shape {shape:?}"
      ),
    });
  }
  let dtype = qweight.dtype()?;
  if dtype != Dtype::U32 {
    return Err(Error::Backend {
      message: format!(
        "unpack_awq_weights: AutoAWQ stores `qweight` as `uint32`-packed nibbles \
         (`utils.py:72-82`); got dtype {dtype:?}"
      ),
    });
  }
  let rows = shape[0];
  let packed_cols = shape[1];
  let cols = packed_cols
    .checked_mul(AWQ_PACK_FACTOR)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!(
        "unpack_awq_weights: unpacked col count `packed_cols * 8` overflows usize \
         (packed_cols={packed_cols})"
      ),
    })?;

  // Build the shift vector + mask scalar as broadcastable inputs. mlx-c
  // promotes mixed-width integer rhs to the lhs dtype, so we hand it
  // u32 directly — no astype dance needed.
  let shifts = Array::from_slice::<u32>(&AWQ_SHIFTS, &(AWQ_SHIFTS.len(),))?;
  let mask = Array::from_slice::<u32>(&[AWQ_NIBBLE_MASK], &(1usize,))?;
  // `qweight[..., None]` → shape `[rows, packed_cols, 1]`; broadcast
  // against `shifts` of shape `[8]` yields `[rows, packed_cols, 8]`.
  let expanded = ops::shape::expand_dims_axes(qweight, &[2])?;
  let shifted = ops::arithmetic::right_shift(&expanded, &shifts)?;
  let nibbles = ops::arithmetic::bitwise_and(&shifted, &mask)?;
  // Collapse the trailing pair `[packed_cols, 8]` → `[cols]`.
  ops::shape::reshape(&nibbles, &(rows, cols))
}

/// Resolve every layer's transformed `model_dtype` consistently
/// (`utils.py:156,163-165`): a single floating dtype shared by all the
/// post-transform floating weights, so heterogeneous-precision
/// checkpoints settle onto one type before the MLX quantize pass.
///
/// mlx-lm takes the LAST iterated layer's `scales.dtype` as the target
/// (`utils.py:156` overwrites `model_dtype` each iteration; the
/// `dict.keys()` order is insertion order, so it's the last inserted
/// `.qweight` layer). mlxrs honors the same contract: we pick the dtype
/// of the `scales` for the lexicographically last `.qweight` prefix —
/// the result is stable across runs (HashMap iteration order is not),
/// and matches the python ref's "last-wins" intent.
fn resolve_awq_model_dtype(
  weights: &Weights,
  qweight_prefixes: &[String],
) -> Result<Option<Dtype>> {
  let Some(last_prefix) = qweight_prefixes.iter().max() else {
    return Ok(None);
  };
  let scales_key = format!("{last_prefix}.scales");
  let scales = weights.get(&scales_key).ok_or_else(|| Error::Backend {
    message: format!(
      "transform_awq_weights: layer `{last_prefix}.qweight` is missing its companion \
         `{scales_key}` (AutoAWQ writes `.qweight` / `.scales` / `.qzeros` as a triple); \
         refusing to silently drop the layer"
    ),
  })?;
  Ok(Some(scales.dtype()?))
}

/// `true` for `F16` / `F32` / `F64` / `BF16` — the mlx-python
/// `mx.issubdtype(dtype, mx.floating)` set
/// (`utils.py:164`).
fn is_floating(d: Dtype) -> bool {
  matches!(d, Dtype::F16 | Dtype::F32 | Dtype::F64 | Dtype::BF16)
}

/// Convert an AutoAWQ / GPTQ on-disk weight map into MLX's quantized-triple
/// layout — port of `_transform_awq_weights`
/// (`mlx-lm/mlx_lm/utils.py:83-172`).
///
/// For every `<prefix>.qweight` in `weights`:
///
/// 1. **Unpack + reorder** `qweight` via [`unpack_awq_weights`]
///    (`utils.py:121`). The unpack folds the AutoAWQ scramble into the
///    shift table, so the result is the dense nibble matrix in natural
///    order. Shape goes `[in_features, packed_out] → [in_features, out_features]`.
/// 2. **Transpose** `[in_features, out_features] → [out_features, in_features]`
///    (mlx stores `Linear`'s weight as `[out, in]`, `utils.py:122-123`).
/// 3. **Re-pack** with MLX's sequential shift table `arange(pack_factor) * bits`
///    (`utils.py:128-131`); output is `[out_features, in_features // pack_factor]`,
///    dtype `uint32` — the exact `mlx.core.QuantizedLinear` layout.
/// 4. **`scales`**: AutoAWQ stores `[n_groups, out_features]`; transpose to
///    `[out_features, n_groups]` and materialise via `contiguous`
///    (`utils.py:133`).
/// 5. **`biases`**: from `qzeros` (asymmetric, `utils.py:136-147`) or
///    implicit-zero (symmetric, `utils.py:148-151`). MLX dequantization is
///    `w * scale + bias`; AWQ's is `(w - zero) * scale`. The algebra makes
///    `bias = -zero * scale`.
/// 6. **Floating-dtype unification** (`utils.py:163-165`): every transformed
///    floating weight is cast to the resolved `model_dtype` (the last
///    iterated layer's `scales.dtype` — see [`resolve_awq_model_dtype`]).
///
/// Returns the converted [`Weights`] map plus a [`PerLayerQuantization`]
/// carrying the resolved `(group_size, bits)` MLX quant params
/// (`utils.py:167-170`). Non-AWQ keys (anything that is not
/// `.qweight` / `.qzeros` / `.scales`) pass through verbatim
/// (`utils.py:158-161`); a `.g_idx` key (non-contiguous GPTQ group
/// indices) is rejected as [`Error::Backend`] (`utils.py:95-100`).
///
/// `config.bits` must be `4`; mlx-lm rejects other widths
/// (`utils.py:88-89`). The caller (loader) is responsible for routing
/// only AWQ / GPTQ checkpoints into this function — the `quant_method`
/// discriminator is read at the loader level (`utils.py:370-391`), not
/// here.
pub fn transform_awq_weights(
  weights: Weights,
  config: &AwqLoadConfig,
) -> Result<(Weights, PerLayerQuantization)> {
  // Faithful to mlx-lm's `if bits != 4: raise ValueError` (`utils.py:88`).
  if config.bits != AWQ_BITS {
    return Err(Error::Backend {
      message: format!(
        "transform_awq_weights: only bits=4 is supported for AutoAWQ/GPTQ models \
         (`mlx-lm/mlx_lm/utils.py:88-89`); got bits={}",
        config.bits
      ),
    });
  }
  let group_size = config.group_size;
  let group_size_i32 = i32::try_from(group_size).map_err(|_| Error::ShapeMismatch {
    message: format!("transform_awq_weights: group_size {group_size} exceeds i32::MAX"),
  })?;

  // Reject GPTQ `g_idx` upfront — `utils.py:95-100`. mlxrs's port does not
  // implement the non-contiguous-group reorder path; the caller must
  // re-quantize via `mlx_lm.convert` or pick a model without `g_idx`.
  for key in weights.keys() {
    if key.ends_with(".g_idx") {
      return Err(Error::Backend {
        message: format!(
          "transform_awq_weights: found `{key}` in weights. Models with non-contiguous \
           group indices (`g_idx`) are not supported by mlx-lm's AutoAWQ on-load \
           converter (`mlx-lm/mlx_lm/utils.py:95-100`). Please use a model without `g_idx` \
           or re-quantize the model via `mlx_lm.convert`."
        ),
      });
    }
  }

  // Collect every prefix of a `.qweight` key (sorted, for the lex-last
  // `model_dtype` resolution + deterministic iteration in tests).
  let mut qweight_prefixes: Vec<String> = weights
    .keys()
    .filter_map(|k| k.strip_suffix(".qweight").map(str::to_string))
    .collect();
  qweight_prefixes.sort();

  let model_dtype = resolve_awq_model_dtype(&weights, &qweight_prefixes)?;

  // Sibling validation pass (deferred until after we know every prefix):
  // refuse on-disk shapes that mlx-lm's converter would NameError /
  // KeyError on, but surface a clear message instead of a panic. AutoAWQ
  // always writes `.scales` alongside `.qweight` (`autoawq` source; the
  // python converter assumes it, `utils.py:109`); `.qzeros` is optional
  // (symmetric mode falls back to `2^(bits-1)`, `utils.py:148-151`).
  for prefix in &qweight_prefixes {
    let qweight_key = format!("{prefix}.qweight");
    let scales_key = format!("{prefix}.scales");
    let Some(qweight) = weights.get(&qweight_key) else {
      // Should be unreachable (we built the prefix list FROM the keys),
      // but guard defensively.
      return Err(Error::Backend {
        message: format!("transform_awq_weights: missing `{qweight_key}` after prefix scan"),
      });
    };
    let Some(scales) = weights.get(&scales_key) else {
      return Err(Error::Backend {
        message: format!(
          "transform_awq_weights: layer `{qweight_key}` is missing its companion `{scales_key}` \
           (AutoAWQ writes `.qweight` / `.scales` / `.qzeros` as a triple); refusing to silently \
           drop the layer"
        ),
      });
    };
    // Shape validation: `qweight: [in_features, packed_out]` /
    // `scales: [n_groups, out_features]`. The two must agree on
    // out_features (post-pack-factor), and `in_features` must be a
    // multiple of `group_size` (`utils.py:118` `n_groups = in_features //
    // group_size`).
    let q_shape = qweight.shape();
    let s_shape = scales.shape();
    if q_shape.len() != 2 {
      return Err(Error::ShapeMismatch {
        message: format!(
          "transform_awq_weights: layer `{qweight_key}`: qweight must be 2-D \
           `[in_features, packed_out]`, got shape {q_shape:?}"
        ),
      });
    }
    if s_shape.len() != 2 {
      return Err(Error::ShapeMismatch {
        message: format!(
          "transform_awq_weights: layer `{scales_key}`: scales must be 2-D \
           `[n_groups, out_features]`, got shape {s_shape:?}"
        ),
      });
    }
    let in_features = q_shape[0];
    let packed_out = q_shape[1];
    let out_features =
      packed_out
        .checked_mul(AWQ_PACK_FACTOR)
        .ok_or_else(|| Error::ShapeMismatch {
          message: format!(
            "transform_awq_weights: layer `{qweight_key}`: out_features overflows usize \
           (packed_out={packed_out})"
          ),
        })?;
    if group_size as usize == 0 || in_features % (group_size as usize) != 0 {
      return Err(Error::ShapeMismatch {
        message: format!(
          "transform_awq_weights: layer `{qweight_key}`: in_features {in_features} is not a \
           multiple of group_size {group_size} (`utils.py:118`: n_groups = in_features // group_size)"
        ),
      });
    }
    let n_groups = in_features / (group_size as usize);
    if s_shape[0] != n_groups || s_shape[1] != out_features {
      return Err(Error::ShapeMismatch {
        message: format!(
          "transform_awq_weights: layer `{prefix}`: scales shape {s_shape:?} does not match \
           the expected `[n_groups={n_groups}, out_features={out_features}]` derived from \
           qweight shape {q_shape:?} with group_size={group_size}"
        ),
      });
    }
    let qzeros_key = format!("{prefix}.qzeros");
    if let Some(qzeros) = weights.get(&qzeros_key) {
      let z_shape = qzeros.shape();
      if z_shape.len() != 2 || z_shape[0] != n_groups || z_shape[1] != packed_out {
        return Err(Error::ShapeMismatch {
          message: format!(
            "transform_awq_weights: layer `{qzeros_key}`: qzeros shape {z_shape:?} does not match \
             the expected `[n_groups={n_groups}, packed_out={packed_out}]` derived from qweight \
             shape {q_shape:?} with group_size={group_size}"
          ),
        });
      }
    }
  }

  // Now do the conversion. Move every key out of the input map exactly
  // once — non-AWQ keys flow straight through.
  let mut new_weights: Weights = HashMap::with_capacity(weights.len());
  // First, pull out every AWQ triple component so the pass-through walk
  // can move the remainder verbatim. Mirrors mlx-lm's
  // `if key.endswith(".qweight"): ... elif not any(key.endswith(...)): ...`
  // structure (`utils.py:102-161`).
  // Triple: (qweight, scales, qzeros) — `qzeros` is optional (symmetric mode).
  type AwqTriple = (Option<Array>, Option<Array>, Option<Array>);
  let mut awq_components: HashMap<String, AwqTriple> =
    HashMap::with_capacity(qweight_prefixes.len());
  let mut remainder: Weights = HashMap::with_capacity(weights.len());
  for (key, arr) in weights {
    if let Some(prefix) = key.strip_suffix(".qweight") {
      awq_components
        .entry(prefix.to_string())
        .or_insert((None, None, None))
        .0 = Some(arr);
    } else if let Some(prefix) = key.strip_suffix(".scales") {
      if qweight_prefixes.binary_search(&prefix.to_string()).is_ok() {
        awq_components
          .entry(prefix.to_string())
          .or_insert((None, None, None))
          .1 = Some(arr);
      } else {
        // Orphan `.scales` not tied to an AWQ triple — pass through.
        remainder.insert(key, arr);
      }
    } else if let Some(prefix) = key.strip_suffix(".qzeros") {
      if qweight_prefixes.binary_search(&prefix.to_string()).is_ok() {
        awq_components
          .entry(prefix.to_string())
          .or_insert((None, None, None))
          .2 = Some(arr);
      } else {
        // Orphan `.qzeros` not tied to an AWQ triple — pass through.
        remainder.insert(key, arr);
      }
    } else {
      remainder.insert(key, arr);
    }
  }

  // Convert each prefix in lexicographic order (deterministic for tests
  // and shaves no observable behavior — mlx-lm's order is HashMap
  // insertion order, which is unspecified).
  for prefix in &qweight_prefixes {
    let (qw_opt, sc_opt, qz_opt) = awq_components
      .remove(prefix)
      .ok_or_else(|| Error::Backend {
        message: format!(
          "transform_awq_weights: layer `{prefix}` lost its components mid-pipeline"
        ),
      })?;
    let qweight = qw_opt.ok_or_else(|| Error::Backend {
      message: format!("transform_awq_weights: layer `{prefix}.qweight` disappeared mid-pipeline"),
    })?;
    let scales = sc_opt.ok_or_else(|| Error::Backend {
      message: format!("transform_awq_weights: layer `{prefix}.scales` disappeared mid-pipeline"),
    })?;

    let q_shape = qweight.shape();
    let in_features = q_shape[0];
    let packed_out = q_shape[1];
    let out_features = packed_out * AWQ_PACK_FACTOR; // checked above.
    let packed_in = in_features / AWQ_PACK_FACTOR;

    // 1. Unpack + reorder `qweight` → `[in_features, out_features]` u32.
    let unpacked = unpack_awq_weights(&qweight)?;
    // 2. Transpose → `[out_features, in_features]`.
    let unpacked_t = ops::shape::transpose(&unpacked)?;
    // 3. Re-pack via MLX's sequential shift table.
    //    reshape → `[out_features, packed_in, pack_factor]`.
    let reshaped = ops::shape::reshape(&unpacked_t, &(out_features, packed_in, AWQ_PACK_FACTOR))?;
    // Build the mlx repack shifts: arange(pack_factor) * bits as u32 vec.
    // `utils.py:128` does `mx.arange(pack_factor) * bits`.
    let pack_shifts_data: Vec<u32> = (0..AWQ_PACK_FACTOR as u32).map(|i| i * AWQ_BITS).collect();
    let pack_shifts = Array::from_slice::<u32>(&pack_shifts_data, &(pack_shifts_data.len(),))?;
    // Force u32 dtype on the reshaped nibble matrix so `<<` doesn't surprise
    // us (it's already u32 from `unpack_awq_weights`, but be explicit —
    // matches `utils.py:130` `repacked.astype(mx.uint32)`).
    let reshaped_u32 = ops::misc::astype(&reshaped, Dtype::U32)?;
    let shifted = ops::arithmetic::left_shift(&reshaped_u32, &pack_shifts)?;
    // sum_axes axis=-1 (the pack_factor axis) → `[out_features, packed_in]` u32.
    let repacked = ops::reduction::sum_axes(&shifted, &[2_i32], false)?;
    // mlx-lm explicitly casts back to u32 in case the `sum` promoted
    // (`utils.py:131`). On mlx the reduction keeps int dtype, but be
    // safe + match the python.
    let new_weight = ops::misc::astype(&repacked, Dtype::U32)?;

    // 4. Scales: transpose `[n_groups, out_features] → [out_features, n_groups]`
    //    then materialize via contiguous (`utils.py:133`). Keep dtype.
    let scales_t = ops::shape::transpose(&scales)?;
    let scales_c = ops::shape::contiguous(&scales_t, false)?;

    // 5. Biases.
    let scales_dtype = scales.dtype()?;
    let biases = if config.zero_point {
      match qz_opt {
        Some(qzeros) => {
          // 5a. Unpack zeros, transpose to `[out_features, n_groups]`.
          let unpacked_zeros = unpack_awq_weights(&qzeros)?;
          let unpacked_zeros_t = ops::shape::transpose(&unpacked_zeros)?;
          // 5b. Promote to f32 for the multiply, then cast back. mlx-lm
          //     does `unpacked_zeros.astype(mx.float32) * scales`
          //     (`utils.py:147`) — note `scales` is the post-transpose
          //     contiguous one.
          let zeros_f32 = ops::misc::astype(&unpacked_zeros_t, Dtype::F32)?;
          let scales_f32 = ops::misc::astype(&scales_c, Dtype::F32)?;
          let prod = ops::arithmetic::multiply(&zeros_f32, &scales_f32)?;
          let neg = ops::arithmetic::negative(&prod)?;
          // Cast to scales dtype (matches `utils.py:155` /
          // `biases.astype(scales.dtype)`).
          ops::misc::astype(&neg, scales_dtype)?
        }
        None => {
          // Asymmetric requested but no qzeros on disk — mlx-lm
          // `utils.py:136` checks `qzeros_key in weights` so the python
          // ref silently falls through to the symmetric path. Mirror
          // that: zero_point=true means "use qzeros if present, else
          // implicit zero".
          symmetric_biases(&scales_c, scales_dtype)?
        }
      }
    } else {
      // 5c. Symmetric: implicit zero `2^(bits-1)` (`utils.py:149-151`).
      symmetric_biases(&scales_c, scales_dtype)?
    };

    new_weights.insert(format!("{prefix}.weight"), new_weight);
    new_weights.insert(format!("{prefix}.scales"), scales_c);
    new_weights.insert(format!("{prefix}.biases"), biases);
  }

  // Pass-through pass for non-AWQ keys (`utils.py:158-161`). After the
  // AWQ pass, apply the floating-dtype unification (`utils.py:163-165`)
  // to every floating weight — but only when we resolved a model_dtype
  // (i.e. at least one AWQ triple was converted; otherwise this is a
  // no-op input and there's no target dtype to enforce).
  for (key, arr) in remainder {
    new_weights.insert(key, arr);
  }
  if let Some(target) = model_dtype {
    let keys: Vec<String> = new_weights.keys().cloned().collect();
    for key in keys {
      let arr = new_weights
        .get(&key)
        .expect("key just enumerated from new_weights");
      let d = arr.dtype()?;
      if is_floating(d) && d != target {
        let cast = ops::misc::astype(arr, target)?;
        new_weights.insert(key, cast);
      }
    }
  }

  let mlx_quantization = PerLayerQuantization::from_global(Quantization {
    group_size: group_size_i32,
    bits: i32::try_from(config.bits).map_err(|_| Error::ShapeMismatch {
      message: format!(
        "transform_awq_weights: bits {} exceeds i32::MAX",
        config.bits
      ),
    })?,
    mode: QuantMode::Affine,
  });

  Ok((new_weights, mlx_quantization))
}

/// Build the symmetric-quantization biases array for one layer
/// (`mlx-lm/mlx_lm/utils.py:149-151`):
///
/// ```text
/// biases = -2^(bits-1) * scales   (cast to scales.dtype)
/// ```
fn symmetric_biases(scales_c: &Array, scales_dtype: Dtype) -> Result<Array> {
  let zero_point = (1u32 << (AWQ_BITS - 1)) as f32; // 8.0 for bits=4
  // `scales * -zero_point` via `multiply` + `full_like` (avoids needing a
  // scalar broadcast helper — `full_like(scales_c, -zero_point)` matches
  // scales dtype shape exactly).
  let factor = ops::misc::full_like(scales_c, -zero_point)?;
  let biases = ops::arithmetic::multiply(scales_c, &factor)?;
  if biases.dtype()? != scales_dtype {
    ops::misc::astype(&biases, scales_dtype)
  } else {
    Ok(biases)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn arr_f32(data: &[f32], shape: &[usize]) -> Array {
    Array::from_slice::<f32>(data, &shape).expect("from_slice")
  }

  /// Construct a packed-quantized `.weight` array (dtype `uint32`,
  /// the layout mlx's `quantize` writes). The new
  /// [`TripleClass`]-based already-quantized detector validates that
  /// `.weight` is `uint32` before passing a triple through, so the
  /// "already quantized" test fixtures need to use this — a dense
  /// `f32` `.weight` next to a `.scales` is now classified as an
  /// orphan, not a valid triple.
  fn arr_u32(data: &[u32], shape: &[usize]) -> Array {
    Array::from_slice::<u32>(data, &shape).expect("from_slice")
  }

  // ──────────────── Quantization parse (schema) ────────────────

  #[test]
  fn quantization_parses_minimal_block() {
    // The simplest mlx-lm form: just `{ group_size, bits }`, no `mode`.
    let cfg_json = r#"{ "quantization": { "group_size": 64, "bits": 4 } }"#;
    let plq = parse_quantization(cfg_json).unwrap().unwrap();
    let q = plq.quantization.expect("global quant present");
    assert_eq!(q.group_size, 64);
    assert_eq!(q.bits, 4);
    assert_eq!(q.mode, QuantMode::Affine);
    assert!(plq.per_layer.is_empty());
  }

  #[test]
  fn quantization_parses_mode_explicit() {
    let cfg_json = r#"{ "quantization": { "group_size": 32, "bits": 4, "mode": "mxfp4" } }"#;
    let q = parse_quantization(cfg_json)
      .unwrap()
      .unwrap()
      .quantization
      .unwrap();
    assert_eq!(q.mode, QuantMode::Mxfp4);
  }

  #[test]
  fn quantization_parses_per_layer_overrides() {
    // Mirrors the mlx-swift `BaseConfiguration.swift:103-118` doc example.
    let cfg_json = r#"{
      "quantization": {
        "group_size": 64,
        "bits": 4,
        "model.embed_tokens": { "group_size": 32, "bits": 4 },
        "model.layers.0.self_attn.q_norm": false
      }
    }"#;
    let plq = parse_quantization(cfg_json).unwrap().unwrap();
    let q = plq.quantization.unwrap();
    assert_eq!(q.group_size, 64);
    assert_eq!(q.bits, 4);
    assert_eq!(plq.per_layer.len(), 2);
    match plq.per_layer.get("model.embed_tokens") {
      Some(QuantizationOption::Quantize(q2)) => {
        assert_eq!(q2.group_size, 32);
        assert_eq!(q2.bits, 4);
      }
      other => panic!("expected Quantize override, got {other:?}"),
    }
    assert_eq!(
      plq
        .per_layer
        .get("model.layers.0.self_attn.q_norm")
        .copied(),
      Some(QuantizationOption::Skip)
    );
    // `quantization_for` resolves correctly for each case.
    assert_eq!(
      plq.quantization_for("model.embed_tokens"),
      Some(Quantization {
        group_size: 32,
        bits: 4,
        mode: QuantMode::Affine,
      })
    );
    assert_eq!(
      plq.quantization_for("model.layers.0.self_attn.q_norm"),
      None
    );
    // An unlisted layer falls back to the global default.
    assert_eq!(
      plq.quantization_for("model.layers.5.mlp.gate_proj"),
      Some(q)
    );
  }

  #[test]
  fn quantization_ignores_legacy_hf_keys() {
    // mlx-swift strips `quant_method` / `linear_class` / `quantization_mode`
    // before the per-layer scan (`BaseConfiguration.swift:152-154`).
    let cfg_json = r#"{
      "quantization": {
        "group_size": 64,
        "bits": 4,
        "quant_method": "awq",
        "linear_class": "QuantizedLinear",
        "quantization_mode": "affine"
      }
    }"#;
    let plq = parse_quantization(cfg_json).unwrap().unwrap();
    assert!(plq.per_layer.is_empty());
    assert_eq!(plq.quantization.unwrap().group_size, 64);
  }

  #[test]
  fn quantization_absent_returns_none() {
    // A valid config.json with no `quantization` key.
    let cfg_json = r#"{ "model_type": "qwen3", "hidden_size": 1024 }"#;
    let plq = parse_quantization(cfg_json).unwrap();
    assert!(plq.is_none());
  }

  #[test]
  fn quantization_invalid_json_errors() {
    let plq = parse_quantization("{ not json");
    assert!(plq.is_err());
  }

  // ──────────────── quantize_weights ────────────────

  /// Tiny canned weight map: two `*.weight` keys eligible for quantization,
  /// one already-quantized triple, one 1-D bias, one weight whose last
  /// axis is not a multiple of `group_size`. Confirms the predicate
  /// (rank / last-axis / `.scales`-sibling-presence) selects exactly the
  /// two eligible weights.
  #[test]
  fn quantize_weights_applies_to_eligible_and_skips_rest() {
    let group_size = 64_usize;
    let n_rows = 3_usize;
    // Two eligible weights: [3, 64].
    let w1 = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
    let w2 = arr_f32(&vec![-0.25_f32; n_rows * group_size], &[n_rows, group_size]);
    // Already-quantized layer: a STRUCTURALLY-VALID affine triple
    // (`<path>.weight` uint32 + `<path>.scales` (+ `<path>.biases`)
    // f32 of matching leading dims). Classified as
    // [`TripleClass::Valid`] → skipped + passed through verbatim (per
    // mlx-lm `utils.py:349-355`, sharpened to the actual mlx layout
    // — `mlx/ops.cpp:4789-4798`).
    // Packed shape: bits=4 packs 8 elements per uint32 → last axis is
    // `group_size / 8 = 8` for group_size=64.
    let already_w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    let already_scales = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    let already_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    // A bias (1-D) — not quantizable (rank < 2).
    let bias = arr_f32(&[1.0_f32, 2.0, 3.0], &[3]);
    // A weight whose last axis (63) is not a multiple of group_size 64.
    let odd_last = arr_f32(&vec![0.0_f32; 3 * 63], &[3, 63]);
    // A non-`.weight` key — should pass through verbatim.
    let other = arr_f32(&[42.0_f32], &[1]);

    let mut weights: Weights = HashMap::new();
    weights.insert("model.layers.0.q_proj.weight".to_string(), w1);
    weights.insert("model.layers.0.k_proj.weight".to_string(), w2);
    weights.insert("model.layers.1.v_proj.weight".to_string(), already_w);
    weights.insert("model.layers.1.v_proj.scales".to_string(), already_scales);
    weights.insert("model.layers.1.v_proj.biases".to_string(), already_biases);
    weights.insert("model.layers.0.q_proj.bias".to_string(), bias);
    weights.insert("model.layers.2.bad.weight".to_string(), odd_last);
    weights.insert("model.norm.gamma".to_string(), other);
    let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));

    let out = quantize_weights(weights, &cfg, &default_eligible).expect("quantize");

    // Eligible weights: replaced with quantized triples (.weight + .scales
    // + .biases for affine).
    for path in ["model.layers.0.q_proj", "model.layers.0.k_proj"] {
      let w_q = out.get(&format!("{path}.weight")).expect(".weight");
      let scales = out.get(&format!("{path}.scales")).expect(".scales");
      let biases = out
        .get(&format!("{path}.biases"))
        .expect(".biases (affine)");
      // mlx `affine_quantize` packs `bits=4` elements 8-per-uint32 along
      // the last axis, so the packed shape is `[N, dim / (32/bits)]` =
      // `[3, 8]` for `[3, 64]` at 4 bits. `scales` / `biases` shape is
      // `[N, dim / group_size]` = `[3, 1]` for one group per row.
      assert_eq!(w_q.shape(), vec![3, 8]);
      assert_eq!(w_q.dtype().unwrap(), crate::dtype::Dtype::U32);
      assert_eq!(scales.shape(), vec![3, 1]);
      assert_eq!(scales.dtype().unwrap(), crate::dtype::Dtype::F32);
      assert_eq!(biases.shape(), vec![3, 1]);
      assert_eq!(biases.dtype().unwrap(), crate::dtype::Dtype::F32);
    }

    // Skipped: already-quantized layer's triple passes through unchanged
    // (uint32 packed `.weight`, f32 `.scales` / `.biases` of matching
    // leading dims — exactly the layout mlx's `affine_quantize` writes).
    let pre_q_w = out.get("model.layers.1.v_proj.weight").expect("already-w");
    assert_eq!(pre_q_w.shape(), vec![n_rows, 8]);
    assert_eq!(pre_q_w.dtype().unwrap(), crate::dtype::Dtype::U32);
    assert!(out.contains_key("model.layers.1.v_proj.scales"));
    assert!(out.contains_key("model.layers.1.v_proj.biases"));

    // Skipped: 1-D bias and ragged-last-axis weight pass through.
    assert_eq!(
      out.get("model.layers.0.q_proj.bias").unwrap().shape(),
      vec![3]
    );
    assert_eq!(
      out.get("model.layers.2.bad.weight").unwrap().shape(),
      vec![3, 63]
    );

    // Skipped: non-`.weight` keys pass through verbatim.
    assert_eq!(out.get("model.norm.gamma").unwrap().shape(), vec![1]);

    // Skipped layers do NOT acquire a stray `.scales`/`.biases`.
    assert!(!out.contains_key("model.layers.0.q_proj.scales.scales"));
    assert!(!out.contains_key("model.layers.2.bad.scales"));
    assert!(!out.contains_key("model.layers.2.bad.biases"));
  }

  #[test]
  fn quantize_then_dequantize_roundtrips_within_tolerance() {
    let group_size = 64_usize;
    let n_rows = 4_usize;
    // Modestly-varying f32 weights so the quantization grid actually
    // covers a useful range (a constant tensor quantizes / dequantizes
    // exactly with zero error, so this catches the lossy path).
    let data: Vec<f32> = (0..n_rows * group_size)
      .map(|i| (i as f32 / 128.0) - 1.0)
      .collect();
    let w = arr_f32(&data, &[n_rows, group_size]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.linear.weight".to_string(), w);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));

    let quantized = quantize_weights(weights, &cfg, &default_eligible).unwrap();
    let dequantized = dequantize_weights(quantized, &cfg).unwrap();

    let mut deq = dequantized
      .get("model.linear.weight")
      .expect("round-tripped .weight")
      .try_clone()
      .unwrap();
    assert_eq!(deq.shape(), vec![n_rows, group_size]);
    let deq_vec: Vec<f32> = deq.to_vec().unwrap();
    // `affine` at 4 bits is lossy; mlx's grouped affine over 64 elements
    // with a [-1, 1) range typically reconstructs within ~ a few %. Use a
    // generous tolerance — the test is for the round-trip plumbing
    // (predicate, triple writeback, dequantize_weights inverse), not the
    // quantizer's exact accuracy (which is mlx-c's job and is tested
    // elsewhere).
    let max_abs_err = data
      .iter()
      .zip(deq_vec.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f32, f32::max);
    assert!(
      max_abs_err < 0.05,
      "round-trip max abs err = {max_abs_err}; expected < 0.05 for 4-bit affine"
    );
  }

  #[test]
  fn quantize_weights_per_layer_skip_passes_through() {
    let group_size = 64_usize;
    let n_rows = 2_usize;
    let w = arr_f32(&vec![0.1_f32; n_rows * group_size], &[n_rows, group_size]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.embed_tokens.weight".to_string(), w);

    // Global default would quantize, but a per-layer Skip turns it off.
    let mut per_layer = HashMap::new();
    per_layer.insert("model.embed_tokens".to_string(), QuantizationOption::Skip);
    let cfg = PerLayerQuantization {
      quantization: Some(Quantization::affine(group_size as i32, 4)),
      per_layer,
    };

    let out = quantize_weights(weights, &cfg, &default_eligible).unwrap();
    let pass = out.get("model.embed_tokens.weight").expect(".weight");
    assert_eq!(pass.shape(), vec![n_rows, group_size]);
    assert_eq!(pass.dtype().unwrap(), crate::dtype::Dtype::F32);
    assert!(!out.contains_key("model.embed_tokens.scales"));
    assert!(!out.contains_key("model.embed_tokens.biases"));
  }

  #[test]
  fn quantize_weights_per_layer_override_uses_override_params() {
    let n_rows = 2_usize;
    // Eligible only at group_size 32 (last axis 32; the global default
    // would be group_size 64, which fails the `% group_size == 0` gate —
    // but the per-layer override at 32 makes it eligible).
    let last = 32_usize;
    let w = arr_f32(&vec![0.1_f32; n_rows * last], &[n_rows, last]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.embed_tokens.weight".to_string(), w);

    let mut per_layer = HashMap::new();
    per_layer.insert(
      "model.embed_tokens".to_string(),
      QuantizationOption::Quantize(Quantization::affine(32, 4)),
    );
    let cfg = PerLayerQuantization {
      quantization: Some(Quantization::affine(64, 4)),
      per_layer,
    };

    let out = quantize_weights(weights, &cfg, &default_eligible).unwrap();
    // Quantized at group_size 32: scales / biases have one group per row
    // (last / group_size = 32 / 32 = 1).
    let scales = out.get("model.embed_tokens.scales").expect(".scales");
    assert_eq!(scales.shape(), vec![n_rows, 1]);
    let w_q = out.get("model.embed_tokens.weight").expect(".weight");
    // bits=4 packs 8 elements per uint32 → last axis is 32 / 8 = 4.
    assert_eq!(w_q.shape(), vec![n_rows, 4]);
  }

  // ──────────────── new Codex-review fixtures ────────────────

  /// Fix 1: a weight whose key ends in `.weight` AND meets every
  /// structural guard (rank ≥ 2, last-axis divisible by group_size) but
  /// the caller-supplied eligibility predicate rejects → passes through
  /// unchanged (no `.scales` / `.biases` emitted). Mirrors mlx-lm's
  /// `wrapped_predicate` returning `False` for a non-Linear /
  /// Embedding / SwitchLinear module (`utils.py:824`).
  #[test]
  fn quantize_weights_predicate_rejected_passes_through() {
    let group_size = 64_usize;
    let n_rows = 2_usize;
    let w = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.some_future_module.weight".to_string(), w);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));
    // Predicate that rejects this specific architecture's "future" module.
    let reject_all: &Eligible<'_> = &|_path: &str, _arr: &Array| false;

    let out = quantize_weights(weights, &cfg, reject_all).unwrap();
    let pass = out.get("model.some_future_module.weight").expect(".weight");
    assert_eq!(pass.shape(), vec![n_rows, group_size]);
    assert_eq!(pass.dtype().unwrap(), crate::dtype::Dtype::F32);
    assert!(!out.contains_key("model.some_future_module.scales"));
    assert!(!out.contains_key("model.some_future_module.biases"));
  }

  /// Fix 1: a predicate that selects a SPECIFIC path AND every other
  /// structural guard passes → that path IS quantized (.weight replaced,
  /// .scales / .biases emitted), while a sibling path the predicate
  /// rejects passes through unchanged. Confirms the predicate is the
  /// PRIMARY filter and the structural guards run after.
  #[test]
  fn quantize_weights_predicate_approved_quantizes() {
    let group_size = 64_usize;
    let n_rows = 2_usize;
    let w_yes = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
    let w_no = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.linear_class.weight".to_string(), w_yes);
    weights.insert("model.other_class.weight".to_string(), w_no);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));
    let only_linear: &Eligible<'_> = &|path: &str, _arr: &Array| path == "model.linear_class";

    let out = quantize_weights(weights, &cfg, only_linear).unwrap();
    // Selected: quantized triple.
    assert_eq!(
      out
        .get("model.linear_class.scales")
        .expect("scales for approved layer")
        .shape(),
      vec![n_rows, 1]
    );
    // Rejected: pass-through (no .scales emitted).
    assert_eq!(
      out
        .get("model.other_class.weight")
        .expect("rejected layer .weight kept")
        .shape(),
      vec![n_rows, group_size]
    );
    assert!(!out.contains_key("model.other_class.scales"));
    assert!(!out.contains_key("model.other_class.biases"));
  }

  // Fix 2: schema-required keys.

  #[test]
  fn quantization_missing_bits_errors() {
    let cfg_json = r#"{ "quantization": { "group_size": 64 } }"#;
    let err = parse_quantization(cfg_json).unwrap_err();
    let msg = format!("{err}");
    assert!(
      msg.contains("bits"),
      "error should mention the missing `bits` key, got: {msg}"
    );
  }

  #[test]
  fn quantization_missing_group_size_errors() {
    let cfg_json = r#"{ "quantization": { "bits": 4 } }"#;
    let err = parse_quantization(cfg_json).unwrap_err();
    let msg = format!("{err}");
    assert!(
      msg.contains("group_size"),
      "error should mention the missing `group_size` key, got: {msg}"
    );
  }

  #[test]
  fn quantization_both_present_ok() {
    let cfg_json = r#"{ "quantization": { "group_size": 32, "bits": 4 } }"#;
    let plq = parse_quantization(cfg_json).unwrap().unwrap();
    let q = plq.quantization.expect("global quant present");
    assert_eq!(q.group_size, 32);
    assert_eq!(q.bits, 4);
  }

  // Fix 3: stale sibling collision.

  #[test]
  fn quantize_weights_orphan_biases_collision_errors() {
    let group_size = 64_usize;
    let n_rows = 2_usize;
    let w = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
    // Orphan biases — NO matching `.scales`, so not a valid
    // already-quantized triple (mlx `affine_quantize` always writes
    // `.scales` alongside `.biases`, `mlx/ops.cpp:4793-4798`). The
    // `classify_triple` check runs BEFORE the eligibility predicate, so
    // this fires unconditionally for every `.weight` key with an orphan
    // `.biases` sibling.
    let stale_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.foo.weight".to_string(), w);
    weights.insert("model.foo.biases".to_string(), stale_biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));
    let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.foo"),
          "error should name the colliding layer, got: {message}"
        );
        assert!(
          message.contains(".biases"),
          "error should name the colliding sibling, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 3: a VALID already-quantized triple (`.weight` uint32 packed +
  /// `.scales` (+ `.biases`) of matching leading dims, the exact layout
  /// mlx's `affine_quantize` writes — `mlx/ops.cpp:4789-4798`) STILL
  /// passes through unchanged. The new [`TripleClass`] validation must
  /// not regress the already-quantized skip.
  #[test]
  fn quantize_weights_valid_existing_triple_still_skipped() {
    let n_rows = 2_usize;
    // Packed `.weight`: bits=4 packs 8 elements per uint32 → last axis
    // is `group_size / 8 = 8` for group_size=64.
    let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    let biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.already.weight".to_string(), w);
    weights.insert("model.already.scales".to_string(), scales);
    weights.insert("model.already.biases".to_string(), biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let out = quantize_weights(weights, &cfg, &default_eligible).expect("valid triple passes");
    // `.weight` is the packed [N, 8] uint32 we inserted — not re-quantized.
    let w_out = out.get("model.already.weight").unwrap();
    assert_eq!(w_out.shape(), vec![n_rows, 8]);
    assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
    assert!(out.contains_key("model.already.scales"));
    assert!(out.contains_key("model.already.biases"));
  }

  /// Fix 4 (this PR): a dense `.weight` (float dtype) next to a stale
  /// `.scales` orphan (no valid quantized layout) → [`TripleClass::Invalid`]
  /// → `Err(Backend)` naming the layer and the offending `.scales`. This is
  /// the case the Codex review surfaced: the old presence-only
  /// `is_already_quantized` check would have classified this as "already
  /// quantized" and silently passed through, leaving a dense `.weight` next
  /// to a corrupt `.scales` for `dequantize_weights` to choke on.
  ///
  /// `.biases` is included so the triple advances past Fix 6's affine-arity
  /// check and reaches the `.weight` dtype check (the regression this
  /// fixture is asserting); a separate fixture covers the missing-`.biases`
  /// arity case under `affine`.
  #[test]
  fn quantize_weights_orphan_scales_with_dense_weight_errors() {
    let group_size = 64_usize;
    let n_rows = 2_usize;
    // Dense f32 `.weight` (NOT a quantized uint32 packed matrix).
    let w = arr_f32(&vec![0.5_f32; n_rows * group_size], &[n_rows, group_size]);
    // Stale orphan `.scales` + matching `.biases` next to it.
    let stale_scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    let stale_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.foo.weight".to_string(), w);
    weights.insert("model.foo.scales".to_string(), stale_scales);
    weights.insert("model.foo.biases".to_string(), stale_biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));
    let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.foo"),
          "error should name the colliding layer, got: {message}"
        );
        assert!(
          message.contains(".scales"),
          "error should name the colliding sibling, got: {message}"
        );
        // The message should call out the dense-`.weight` mismatch
        // (the load-bearing signal that this is an orphan, not a real
        // already-quantized triple).
        assert!(
          message.contains("uint32") || message.contains("dense") || message.contains("F32"),
          "error should explain the `.weight` dtype mismatch, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 4 (this PR): a `.weight` + `.scales` with MISMATCHED leading
  /// dims (the `.weight` claims to be uint32 packed, but its rank or
  /// leading shape doesn't match `.scales` as mlx's `quantize` would
  /// produce). Classified as [`TripleClass::Invalid`] → `Err(Backend)`.
  #[test]
  fn quantize_weights_mismatched_scales_shape_errors() {
    let n_rows = 2_usize;
    // Packed `.weight` (uint32) at shape [N=2, 8].
    let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    // `.scales` with a different leading dim ([3, 1] vs `.weight`
    // leading dim of [2]).
    let bad_scales = arr_f32(&[1.0_f32; 3], &[3, 1]);
    // `.biases` matching `.scales` so the triple advances past Fix 6's
    // affine-arity check and reaches the leading-dim mismatch check.
    let biases = arr_f32(&[0.0_f32; 3], &[3, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.foo.weight".to_string(), w);
    weights.insert("model.foo.scales".to_string(), bad_scales);
    weights.insert("model.foo.biases".to_string(), biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.foo"),
          "error should name the colliding layer, got: {message}"
        );
        assert!(
          message.contains("leading"),
          "error should explain the leading-dim mismatch, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  // R5 structural pivot: the `quantize_weights_mismatched_biases_dtype_errors`
  // test from R4 asserted a `.biases`/`.scales` dtype-equality check we are
  // intentionally removing to match the mlx-lm / mlx-swift reference loader
  // paths (which trust mlx-c to validate scale dtypes at the
  // `quantize` / `dequantize` call site — `mlx/mlx/ops.cpp:75-115`). The
  // dtype-mismatched triple is now passed through to mlx-c, which surfaces
  // a precise `[dequantize] ...` error. See the module-level "Validation
  // contract" section.

  // ──────────────── Structural shape sanity ────────────────

  /// Fix 5: a uint32 rank-1 `.weight` next to a uint32 rank-1 `.scales`
  /// (rank-equal, even leading-dim-equal trivially since both have only
  /// a last axis). Pre-fix `classify_triple` would have classified this
  /// as [`TripleClass::Valid`] (dtype `uint32` + ranks equal + no
  /// rank ≥ 2 check). The fix rejects it because mlx `quantize` requires
  /// rank ≥ 2 inputs (`mlx/ops.cpp:4925-4929`).
  #[test]
  fn quantize_weights_rank1_uint32_triple_errors() {
    // Both `.weight` and `.scales` are rank-1 uint32 — would slip past
    // the dtype + rank-equality check, but mlx never emits a rank-1
    // quantized triple.
    let w = arr_u32(&[0_u32, 0, 0, 0], &[4]);
    let scales = arr_u32(&[1_u32], &[1]);
    // `.biases` matching `.scales` shape/dtype so the triple advances past
    // Fix 6's affine-arity check and reaches the rank-≥-2 check.
    let biases = arr_u32(&[0_u32], &[1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.bad.weight".to_string(), w);
    weights.insert("model.bad.scales".to_string(), scales);
    weights.insert("model.bad.biases".to_string(), biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.bad"),
          "error should name the malformed layer, got: {message}"
        );
        assert!(
          message.contains("rank"),
          "error should call out the `.weight` rank, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// R5 structural pivot: a `.weight` + `.scales` triple whose
  /// `.scales` last-axis does NOT match the mlx invariant
  /// `w.shape(-1) * 32 / bits == scales.shape(-1) * group_size`
  /// (`mlx/ops.cpp:107`) now passes `classify_triple` (which only
  /// checks dtype/rank/leading-dims, see the module-level "Validation
  /// contract" section). The mismatch is caught downstream by mlx-c
  /// at the `dequantize` call — the loader path no longer rejects it
  /// upfront, mirroring mlx-lm's `quantize_module_predicate`
  /// (`utils.py:823-835`) and mlx-swift's `QuantizationContainer.decode`
  /// (`BaseConfiguration.swift:139-171`), which both trust mlx-c.
  ///
  /// This test asserts the new pass-through behavior: an
  /// already-quantized triple with structurally-sound dtype/rank/leading
  /// dims is preserved verbatim regardless of the per-mode bits /
  /// group_size pairing (mlx-c will validate when the user later
  /// invokes `dequantize_weights` or any quantized matmul).
  #[test]
  fn quantize_weights_pre_quantized_triple_passes_through_to_mlxc() {
    // Packed `.weight` `[2, 8]` u32 + `.scales` `[2, 2]` f32 (+ `.biases`
    // matching). Under the OLD R4 check, `bits=4, group_size=64` would
    // have rejected this (expected scales-last = 8 * 32 / 4 / 64 = 1, not
    // 2). Under R5, this passes through — mlx-c is the validator.
    let n_rows = 2_usize;
    let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    let scales = arr_f32(&vec![1.0_f32; n_rows * 2], &[n_rows, 2]);
    let biases = arr_f32(&vec![0.0_f32; n_rows * 2], &[n_rows, 2]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.foo.weight".to_string(), w);
    weights.insert("model.foo.scales".to_string(), scales);
    weights.insert("model.foo.biases".to_string(), biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let out = quantize_weights(weights, &cfg, &default_eligible)
      .expect("triple now passes through; mlx-c validates per-mode params at call time");
    // Triple preserved verbatim.
    let w_out = out.get("model.foo.weight").expect(".weight");
    assert_eq!(w_out.shape(), vec![n_rows, 8]);
    assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
    let s_out = out.get("model.foo.scales").expect(".scales");
    assert_eq!(s_out.shape(), vec![n_rows, 2]);
    assert!(out.contains_key("model.foo.biases"));
  }

  /// R5 faithful-port: an affine triple with `bits=3` (mlx-supported,
  /// `mlx/ops.cpp:4745-4750`: bits ∈ {2,3,4,5,6,8}) passes through.
  /// The old R4 `32 % bits == 0` guard incorrectly rejected `bits ∈
  /// {3, 5, 6}`; per the new validation contract, per-mode bits
  /// validation is delegated to mlx-c.
  #[test]
  fn quantize_weights_pre_quantized_bits3_triple_passes_through() {
    // A structurally-sound triple with `bits=3` per the per-layer
    // override. `classify_triple` only checks `.weight` is u32, rank
    // ≥ 2, leading-dims match — none of which depend on the bit width.
    // (The exact packed last-axis would depend on mlx's bits=3 packing,
    // but the loader path does not compute it.)
    let n_rows = 2_usize;
    let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    let biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.foo.weight".to_string(), w);
    weights.insert("model.foo.scales".to_string(), scales);
    weights.insert("model.foo.biases".to_string(), biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 3));
    let out = quantize_weights(weights, &cfg, &default_eligible)
      .expect("bits=3 triple passes through; mlx supports bits ∈ {2,3,4,5,6,8}");
    let w_out = out.get("model.foo.weight").expect(".weight");
    assert_eq!(w_out.shape(), vec![n_rows, 8]);
    assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
  }

  /// R5 structural-shape regression: a CORRECT `.weight` `[2, 8]`
  /// packed at `bits=4, group_size=64` with `.scales` `[2, 1]` (+
  /// `.biases` matching `.scales` shape — affine-arity holds). Still
  /// passes through (the basic shape-sanity checks all hold).
  #[test]
  fn quantize_weights_valid_triple_skipped() {
    let n_rows = 2_usize;
    let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    let biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.foo.weight".to_string(), w);
    weights.insert("model.foo.scales".to_string(), scales);
    weights.insert("model.foo.biases".to_string(), biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let out =
      quantize_weights(weights, &cfg, &default_eligible).expect("valid triple passes through");
    let w_out = out.get("model.foo.weight").expect(".weight");
    assert_eq!(w_out.shape(), vec![n_rows, 8]);
    assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
    let s_out = out.get("model.foo.scales").expect(".scales");
    assert_eq!(s_out.shape(), vec![n_rows, 1]);
    assert!(out.contains_key("model.foo.biases"));
  }

  /// Fix 5: a triple at a path that the per-layer config marks as
  /// `Skip`. The layer was intentionally not quantized — a pre-existing
  /// triple at that path is a stale collision. Classified as
  /// [`TripleClass::Invalid`] (the doc-level "Precondition" branch).
  #[test]
  fn quantize_weights_triple_on_skip_path_errors() {
    let n_rows = 2_usize;
    let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.embed_tokens.weight".to_string(), w);
    weights.insert("model.embed_tokens.scales".to_string(), scales);

    let mut per_layer = HashMap::new();
    per_layer.insert("model.embed_tokens".to_string(), QuantizationOption::Skip);
    let cfg = PerLayerQuantization {
      quantization: Some(Quantization::affine(64, 4)),
      per_layer,
    };

    let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.embed_tokens"),
          "error should name the Skip layer, got: {message}"
        );
        assert!(
          message.contains("Skip"),
          "error should call out the Skip override, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  // ──────────────── Fix 6 (this PR): per-mode bias arity ────────────────

  /// Fix 6: an `affine` triple with NO `.biases` (only `.weight` + `.scales`)
  /// is structurally incomplete. mlx `affine_quantize` emits
  /// `{w_q, scales, biases}` unconditionally (`mlx/ops.cpp:4793-4798`); a
  /// matching shape/dtype on `.scales` is not enough — the resolved mode
  /// dictates the bias arity. Classified as [`TripleClass::Invalid`].
  #[test]
  fn quantize_weights_affine_triple_missing_biases_errors() {
    let n_rows = 2_usize;
    // Packed `.weight` `[2, 8]` u32 + `.scales` `[2, 1]` f32 — a layout
    // that matches the affine weight/scales invariant except for the
    // missing `.biases`.
    let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.affine_missing.weight".to_string(), w);
    weights.insert("model.affine_missing.scales".to_string(), scales);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.affine_missing"),
          "error should name the incomplete layer, got: {message}"
        );
        assert!(
          message.contains(".biases"),
          "error should name the missing `.biases` sibling, got: {message}"
        );
        assert!(
          message.contains("affine"),
          "error should call out the `affine` mode requirement, got: {message}"
        );
        // The arity message also names `bits` / `group_size` (the cfg
        // that fixed the resolved mode).
        assert!(
          message.contains("bits=4"),
          "error should name the resolved bits, got: {message}"
        );
        assert!(
          message.contains("group_size=64"),
          "error should name the resolved group_size, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 6: an `mxfp4` triple with `.biases` present is a stale sibling
  /// from a different mode. mlx `fp_quantize` emits `{w_q, scales}`
  /// only — never `.biases` (`mlx/ops.cpp:4890,4898-4904`). Even if
  /// shape/dtype happen to align with `.scales`, the bias slot MUST be
  /// absent. Classified as [`TripleClass::Invalid`].
  #[test]
  fn quantize_weights_mxfp4_triple_with_stale_biases_errors() {
    let n_rows = 2_usize;
    // `mxfp4` requires `group_size=32`, `bits=4` (`mlx/ops.cpp:4808-4823`).
    // Unpacked last = packed_last * 32 / bits = 4 * 8 = 32 = group_size,
    // so scales last-axis = 32 / 32 = 1 — a structurally well-formed
    // `mxfp4` `.weight`/`.scales` pair.
    let w = arr_u32(&vec![0_u32; n_rows * 4], &[n_rows, 4]);
    let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    // Stale `.biases` from a different (affine) mode — same shape/dtype
    // as `.scales` so it looks valid to a shape-only check.
    let stale_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.mxfp4_stale.weight".to_string(), w);
    weights.insert("model.mxfp4_stale.scales".to_string(), scales);
    weights.insert("model.mxfp4_stale.biases".to_string(), stale_biases);

    let cfg = PerLayerQuantization::from_global(Quantization {
      group_size: 32,
      bits: 4,
      mode: QuantMode::Mxfp4,
    });
    let err = quantize_weights(weights, &cfg, &default_eligible).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.mxfp4_stale"),
          "error should name the offending layer, got: {message}"
        );
        assert!(
          message.contains("mxfp4"),
          "error should call out the offending `mxfp4` mode, got: {message}"
        );
        assert!(
          message.contains(".biases"),
          "error should name the stale `.biases` sibling, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 6 regression: a structurally valid `mxfp4` triple
  /// (`.weight` u32 + `.scales` matching, NO `.biases`) — the scale-only
  /// layout `fp_quantize` actually writes (`mlx/ops.cpp:4890,4898-4904`).
  /// Must pass through unchanged: the new arity check accepts the
  /// `(Mxfp4 | Mxfp8 | Nvfp4, None)` arm.
  #[test]
  fn quantize_weights_valid_mxfp4_scales_only_triple_passes() {
    let n_rows = 2_usize;
    // `mxfp4` invariants: group_size=32, bits=4. Packed `.weight` `[2, 4]`
    // u32 → unpacks to `[2, 32]` (1 group per row) → `.scales` `[2, 1]`.
    let w = arr_u32(&vec![0_u32; n_rows * 4], &[n_rows, 4]);
    let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.mxfp4_ok.weight".to_string(), w);
    weights.insert("model.mxfp4_ok.scales".to_string(), scales);

    let cfg = PerLayerQuantization::from_global(Quantization {
      group_size: 32,
      bits: 4,
      mode: QuantMode::Mxfp4,
    });
    let out = quantize_weights(weights, &cfg, &default_eligible)
      .expect("scale-only mxfp4 triple passes through");
    // `.weight` and `.scales` preserved verbatim; `.biases` is NOT
    // synthesized (scale-only mode).
    let w_out = out.get("model.mxfp4_ok.weight").expect(".weight");
    assert_eq!(w_out.shape(), vec![n_rows, 4]);
    assert_eq!(w_out.dtype().unwrap(), crate::dtype::Dtype::U32);
    let s_out = out.get("model.mxfp4_ok.scales").expect(".scales");
    assert_eq!(s_out.shape(), vec![n_rows, 1]);
    assert!(!out.contains_key("model.mxfp4_ok.biases"));
  }

  // ──────────────── R5 dequantize_weights mode-arity symmetry ────────────────

  /// R5 Finding 1: `dequantize_weights` is symmetric with
  /// `quantize_weights`'s mode-arity check (the `affine`-requires-biases
  /// / `mxfp*|nvfp4`-forbids-biases contract). An affine triple WITHOUT
  /// `.biases` was previously forwarded to mlx-c's `dequantize`, which
  /// would silently reconstruct without the zero-point. The arity check
  /// now catches this upfront and returns a clear error naming the layer
  /// and the resolved `affine` mode.
  #[test]
  fn dequantize_weights_affine_missing_biases_errors() {
    let n_rows = 2_usize;
    // Structurally-valid affine `.weight` + `.scales` pair, but no
    // `.biases` — incomplete affine triple.
    let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.affine_no_bias.weight".to_string(), w);
    weights.insert("model.affine_no_bias.scales".to_string(), scales);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let err = dequantize_weights(weights, &cfg).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.affine_no_bias"),
          "error should name the layer, got: {message}"
        );
        assert!(
          message.contains("affine"),
          "error should name the resolved `affine` mode, got: {message}"
        );
        assert!(
          message.contains(".biases"),
          "error should name the missing `.biases` sibling, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// R5 Finding 1: an `mxfp4` triple WITH a stale `.biases` would be
  /// forwarded to mlx-c, which silently dequantizes (ignoring the
  /// biases). The arity check now catches this upfront and returns a
  /// clear error naming the layer and the offending `mxfp4` mode.
  #[test]
  fn dequantize_weights_mxfp4_with_stale_biases_errors() {
    let n_rows = 2_usize;
    let w = arr_u32(&vec![0_u32; n_rows * 4], &[n_rows, 4]);
    let scales = arr_f32(&vec![1.0_f32; n_rows], &[n_rows, 1]);
    let stale_biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.mxfp4_stale.weight".to_string(), w);
    weights.insert("model.mxfp4_stale.scales".to_string(), scales);
    weights.insert("model.mxfp4_stale.biases".to_string(), stale_biases);

    let cfg = PerLayerQuantization::from_global(Quantization {
      group_size: 32,
      bits: 4,
      mode: QuantMode::Mxfp4,
    });
    let err = dequantize_weights(weights, &cfg).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.mxfp4_stale"),
          "error should name the layer, got: {message}"
        );
        assert!(
          message.contains("mxfp4"),
          "error should name the offending `mxfp4` mode, got: {message}"
        );
        assert!(
          message.contains(".biases"),
          "error should name the stale `.biases` sibling, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// R6 Finding: `dequantize_weights` is symmetric with
  /// `classify_triple`'s orphan-`.biases` guard. A map carrying
  /// `.weight` (`uint32` packed) + `.biases` but NO `.scales` is never
  /// a valid mlx-produced triple (mlx `affine_quantize` always writes
  /// `.scales` alongside `.biases`, `mlx/ops.cpp:4793-4798`). Pre-fix
  /// the orphan would fall through the discovery walk (which only
  /// indexed `.scales` keys) and the `uint32` packed `.weight` would
  /// pass through to the dequantized output as-is. The new orphan-bias
  /// guard catches this upfront with the same exit point + message
  /// style as the dequantize arity check.
  #[test]
  fn dequantize_weights_orphan_biases_with_packed_weight_errors() {
    let n_rows = 2_usize;
    // `uint32`-packed `.weight` shaped [2, 8] + `.biases` [2, 1], NO `.scales`.
    let w = arr_u32(&vec![0_u32; n_rows * 8], &[n_rows, 8]);
    let biases = arr_f32(&vec![0.0_f32; n_rows], &[n_rows, 1]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.orphan_bias.weight".to_string(), w);
    weights.insert("model.orphan_bias.biases".to_string(), biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let err = dequantize_weights(weights, &cfg).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.orphan_bias"),
          "error should name the layer, got: {message}"
        );
        assert!(
          message.contains(".scales"),
          "error should name the missing `.scales` sibling, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// R7 Finding: the R6 orphan-bias guard over-rejected a normal dense
  /// Linear layer carrying `P.weight` (F32) + `P.biases` (F32) with no
  /// `P.scales` — that combination is a standard dense+bias layer, not a
  /// malformed quantized triple. The narrowed guard only fires when
  /// `P.weight` is `uint32` (the mlx-quantization signal,
  /// `mlx/ops.cpp:4795,4900`); a dense (non-`uint32`) `.weight` passes
  /// through verbatim, both keys preserved.
  #[test]
  fn dequantize_weights_dense_weight_with_biases_passes_through() {
    let n_rows = 2_usize;
    let n_cols = 8_usize;
    // Dense F32 `.weight` shaped [2, 8] + F32 `.biases` [8], NO `.scales`.
    let w = arr_f32(
      &(0..n_rows * n_cols).map(|i| i as f32).collect::<Vec<_>>(),
      &[n_rows, n_cols],
    );
    let biases = arr_f32(&vec![0.5_f32; n_cols], &[n_cols]);
    let mut weights: Weights = HashMap::new();
    weights.insert("model.dense.weight".to_string(), w);
    weights.insert("model.dense.biases".to_string(), biases);

    let cfg = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let out = dequantize_weights(weights, &cfg)
      .expect("dense `.weight` (F32) + `.biases` (F32) with no `.scales` must pass through");

    // Both keys preserved verbatim, dtypes unchanged.
    let mut w_out = out
      .get("model.dense.weight")
      .expect("passed-through .weight")
      .try_clone()
      .unwrap();
    let mut b_out = out
      .get("model.dense.biases")
      .expect("passed-through .biases")
      .try_clone()
      .unwrap();
    assert_eq!(w_out.dtype().unwrap(), Dtype::F32);
    assert_eq!(b_out.dtype().unwrap(), Dtype::F32);
    assert_eq!(w_out.shape(), vec![n_rows, n_cols]);
    assert_eq!(b_out.shape(), vec![n_cols]);
    let w_vec: Vec<f32> = w_out.to_vec().unwrap();
    let b_vec: Vec<f32> = b_out.to_vec().unwrap();
    assert_eq!(
      w_vec,
      (0..n_rows * n_cols).map(|i| i as f32).collect::<Vec<_>>(),
      "dense `.weight` data must be passed through verbatim"
    );
    assert_eq!(
      b_vec,
      vec![0.5_f32; n_cols],
      "`.biases` data must be passed through verbatim"
    );
  }

  // ──────────────── AutoAWQ on-load conversion ────────────────

  /// `0xFFFF` packed at the AWQ bit positions for `[0xF, 0, 0xF, 0, 0xF, 0, 0xF, 0]`.
  /// See [`AWQ_SHIFTS`] for the bit-layout algebra. Verifying this exact pattern
  /// pins the inverse-permutation step and catches a swap to `[0..8] * bits`
  /// (the swift `unpackAndReorder` form without the `take` step).
  #[test]
  fn unpack_awq_weights_single_int32_gives_8_nibbles() {
    // `0xFFFF` = `0xF | (0xF << 4) | (0xF << 8) | (0xF << 12)` — four 0xF
    // nibbles at AWQ shift positions [0, 4, 8, 12] = logical positions
    // [0, 2, 4, 6] (even). The shift table places them at output positions
    // 0, 2, 4, 6; the zero nibbles at AWQ positions [16, 20, 24, 28] land
    // at output positions 1, 3, 5, 7.
    let packed = Array::from_slice::<u32>(&[0xFFFF_u32], &(1usize, 1)).unwrap();
    let mut unpacked = unpack_awq_weights(&packed).unwrap();
    assert_eq!(unpacked.shape(), vec![1, 8]);
    assert_eq!(unpacked.dtype().unwrap(), Dtype::U32);
    assert_eq!(
      unpacked.to_vec::<u32>().unwrap(),
      vec![0xF, 0, 0xF, 0, 0xF, 0, 0xF, 0]
    );
  }

  /// Verify the inverse permutation: packing nibbles `[0, 1, 2, 3, 4, 5, 6, 7]`
  /// at AWQ bit positions produces an int32 that unpacks to that natural order.
  /// This is the load-bearing assertion — if the shift table were sequential
  /// (`[0..8] * bits`) the output would be `[0, 2, 4, 6, 1, 3, 5, 7]` (the
  /// AWQ-native scrambled order).
  #[test]
  fn unpack_awq_weights_reverses_awq_scramble() {
    // logical-pos → bit-pos: [0→0, 1→16, 2→4, 3→20, 4→8, 5→24, 6→12, 7→28].
    // The 0-nibble at bit 0 contributes nothing — drop the explicit `0_u32 |`
    // to avoid clippy's `identity_op` lint.
    let packed_val: u32 =
      (1_u32 << 16) | (2 << 4) | (3 << 20) | (4 << 8) | (5 << 24) | (6 << 12) | (7 << 28);
    assert_eq!(packed_val, 0x7531_6420);
    let packed = Array::from_slice::<u32>(&[packed_val], &(1usize, 1)).unwrap();
    let mut unpacked = unpack_awq_weights(&packed).unwrap();
    assert_eq!(unpacked.shape(), vec![1, 8]);
    assert_eq!(
      unpacked.to_vec::<u32>().unwrap(),
      vec![0, 1, 2, 3, 4, 5, 6, 7]
    );
  }

  /// 2-D `[rows, packed_cols]` qweight → `[rows, packed_cols * 8]`. Mirrors
  /// the python ref's strict 2-D contract (`utils.py:75` `out_features,
  /// packed_in = qweight.shape`).
  #[test]
  fn unpack_awq_weights_preserves_row_count_expands_cols_8x() {
    // 3 rows × 2 packed_cols = 6 int32. Use all zeros (the only shape we're
    // checking here).
    let packed = Array::from_slice::<u32>(&[0u32; 6], &(3usize, 2)).unwrap();
    let mut unpacked = unpack_awq_weights(&packed).unwrap();
    assert_eq!(unpacked.shape(), vec![3, 16]);
    assert_eq!(unpacked.to_vec::<u32>().unwrap(), vec![0u32; 48]);
  }

  /// All-zero packed input → all-zero unpacked output of correct shape.
  #[test]
  fn unpack_awq_weights_handles_zero_input() {
    let packed = Array::from_slice::<u32>(&[0u32, 0, 0, 0], &(2usize, 2)).unwrap();
    let mut unpacked = unpack_awq_weights(&packed).unwrap();
    assert_eq!(unpacked.shape(), vec![2, 16]);
    assert_eq!(unpacked.to_vec::<u32>().unwrap(), vec![0u32; 32]);
  }

  /// 1-D / 3-D / 0-D inputs are rejected with a clear shape error. Mirrors the
  /// python ref's strict 2-D contract (`utils.py:75`).
  #[test]
  fn unpack_awq_weights_rejects_non_2d() {
    let r1 = Array::from_slice::<u32>(&[0u32; 4], &(4usize,)).unwrap();
    let err = unpack_awq_weights(&r1).unwrap_err();
    assert!(
      matches!(err, Error::ShapeMismatch { .. }),
      "1-D should be ShapeMismatch, got {err:?}"
    );
    let r3 = Array::from_slice::<u32>(&[0u32; 8], &(2usize, 2, 2)).unwrap();
    assert!(matches!(
      unpack_awq_weights(&r3).unwrap_err(),
      Error::ShapeMismatch { .. }
    ));
  }

  /// Non-`u32` dtype is rejected (AutoAWQ's `qweight` is always `uint32`-packed;
  /// any other dtype is a layout mismatch the caller should fix upstream).
  #[test]
  fn unpack_awq_weights_rejects_non_u32_dtype() {
    let r = Array::from_slice::<i32>(&[0i32; 4], &(2usize, 2)).unwrap();
    let err = unpack_awq_weights(&r).unwrap_err();
    assert!(
      matches!(err, Error::Backend { .. }),
      "i32 dtype should be Backend, got {err:?}"
    );
  }

  // ──────────────── transform_awq_weights ────────────────

  /// Build a 1-element AWQ qweight (`[1, 1]` u32) whose 8 nibbles, in logical
  /// order, equal `nibbles`.
  fn awq_pack_one_row(nibbles: [u32; 8]) -> u32 {
    let mut packed = 0u32;
    for (k, &n) in nibbles.iter().enumerate() {
      packed |= (n & 0xF) << AWQ_SHIFTS[k];
    }
    packed
  }

  /// Compute the AutoAWQ-dequantize value for a single nibble:
  /// `(nibble - zero) * scale` (`utils.py:144-147` comment).
  fn awq_dequant(nibble: u32, zero: u32, scale: f32) -> f32 {
    (nibble as i32 - zero as i32) as f32 * scale
  }

  /// Compute the MLX-affine-dequantize value for a single nibble:
  /// `nibble * scale + bias` (`mlx/ops.cpp` affine_dequantize convention).
  fn mlx_dequant(nibble: u32, scale: f32, bias: f32) -> f32 {
    nibble as f32 * scale + bias
  }

  /// End-to-end round-trip: pick known AWQ qweight/qzeros/scales, run
  /// `transform_awq_weights`, then verify that re-dequantizing the MLX-format
  /// output (via the literal `nibble * scale + bias`) matches the original
  /// AWQ-format dequant (`(nibble - zero) * scale`) at every output position.
  /// This is the load-bearing semantic guarantee of the converter.
  #[test]
  fn transform_awq_weights_round_trips_known_fixture() {
    // in_features = 8, out_features = 8, group_size = 4, bits = 4.
    // → packed_out = 1, packed_in = 2, n_groups = 2.
    // qweight shape: [in_features, packed_out] = [8, 1]
    // scales  shape: [n_groups,    out_features] = [2, 8]
    // qzeros  shape: [n_groups,    packed_out] = [2, 1]
    let in_features = 8usize;
    let out_features = 8usize;
    let group_size = 4u32;
    let n_groups = 2usize;

    // Choose distinct nibbles per (in, out) so we can verify the transpose.
    // unpacked_awq[in, out] = ((in + 1) * 3 + out) % 16
    let awq_unpacked: Vec<Vec<u32>> = (0..in_features)
      .map(|i| {
        (0..out_features)
          .map(|o| (((i + 1) * 3 + o) % 16) as u32)
          .collect()
      })
      .collect();
    // Pack each row's 8 nibbles into one u32 → flat [in_features] u32 buffer.
    let qweight_data: Vec<u32> = (0..in_features)
      .map(|i| {
        let row: [u32; 8] = awq_unpacked[i].to_vec().try_into().unwrap();
        awq_pack_one_row(row)
      })
      .collect();
    let qweight = Array::from_slice::<u32>(&qweight_data, &(in_features, 1)).unwrap();

    // qzeros: per (group, out). Choose nibble = (group + out) % 16.
    let qzero_unpacked: Vec<Vec<u32>> = (0..n_groups)
      .map(|g| (0..out_features).map(|o| ((g + o) % 16) as u32).collect())
      .collect();
    let qzeros_data: Vec<u32> = (0..n_groups)
      .map(|g| {
        let row: [u32; 8] = qzero_unpacked[g].to_vec().try_into().unwrap();
        awq_pack_one_row(row)
      })
      .collect();
    let qzeros = Array::from_slice::<u32>(&qzeros_data, &(n_groups, 1)).unwrap();

    // scales: per (group, out). Distinct positive floats.
    let scales_data: Vec<f32> = (0..n_groups * out_features)
      .map(|i| 0.1_f32 * (i as f32 + 1.0))
      .collect();
    let scales = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();

    let mut weights: Weights = HashMap::new();
    weights.insert("layer.qweight".to_string(), qweight);
    weights.insert("layer.qzeros".to_string(), qzeros);
    weights.insert("layer.scales".to_string(), scales);

    let config = AwqLoadConfig {
      bits: 4,
      group_size,
      zero_point: true,
      version: "gemm".into(),
    };
    let (out, plq) = transform_awq_weights(weights, &config).expect("transform");

    // PerLayerQuantization carries the resolved (group_size=4, bits=4, affine).
    let g = plq.quantization.expect("global quant");
    assert_eq!(g.group_size, group_size as i32);
    assert_eq!(g.bits, 4);
    assert_eq!(g.mode, QuantMode::Affine);

    // Output keys: `layer.weight` (u32 [out, packed_in]),
    //              `layer.scales` (f32 [out, n_groups]),
    //              `layer.biases` (f32 [out, n_groups]).
    let mut weight_arr = out
      .get("layer.weight")
      .expect("layer.weight")
      .try_clone()
      .unwrap();
    let mut scales_arr = out
      .get("layer.scales")
      .expect("layer.scales")
      .try_clone()
      .unwrap();
    let mut biases_arr = out
      .get("layer.biases")
      .expect("layer.biases")
      .try_clone()
      .unwrap();
    assert!(
      !out.contains_key("layer.qweight"),
      "qweight key must be replaced by .weight"
    );
    assert!(
      !out.contains_key("layer.qzeros"),
      "qzeros key must be replaced by .biases"
    );
    assert_eq!(weight_arr.dtype().unwrap(), Dtype::U32);
    assert_eq!(weight_arr.shape(), vec![out_features, in_features / 8]);
    assert_eq!(scales_arr.shape(), vec![out_features, n_groups]);
    assert_eq!(biases_arr.shape(), vec![out_features, n_groups]);
    assert_eq!(scales_arr.dtype().unwrap(), Dtype::F32);
    assert_eq!(biases_arr.dtype().unwrap(), Dtype::F32);

    // Unpack the MLX-format weight back to natural nibbles for the assertion.
    // MLX-packed shifts: arange(8) * 4 = [0, 4, 8, 12, 16, 20, 24, 28].
    let weight_packed: Vec<u32> = weight_arr.to_vec().unwrap();
    let mut mlx_nibbles = vec![vec![0u32; in_features]; out_features];
    for o in 0..out_features {
      for pi in 0..(in_features / 8) {
        let word = weight_packed[o * (in_features / 8) + pi];
        for k in 0..8 {
          mlx_nibbles[o][pi * 8 + k] = (word >> (k as u32 * AWQ_BITS)) & AWQ_NIBBLE_MASK;
        }
      }
    }

    // Verify: mlx_nibbles[o][i] == awq_unpacked[i][o] (the transpose).
    for o in 0..out_features {
      for i in 0..in_features {
        assert_eq!(
          mlx_nibbles[o][i], awq_unpacked[i][o],
          "MLX-format nibble at (o={o}, i={i}) must equal AWQ-format nibble at (i={i}, o={o})"
        );
      }
    }

    // Verify MLX-dequant matches AWQ-dequant at every (i, o, group).
    let scales_flat: Vec<f32> = scales_arr.to_vec().unwrap();
    let biases_flat: Vec<f32> = biases_arr.to_vec().unwrap();
    for o in 0..out_features {
      for g in 0..n_groups {
        // Per-group scale + bias (MLX layout: [o, g]).
        let mlx_scale = scales_flat[o * n_groups + g];
        let mlx_bias = biases_flat[o * n_groups + g];
        // AWQ scale + zero for this (group, out) — AWQ scales/zeros are
        // per group (n_groups, out_features).
        let awq_scale = scales_data[g * out_features + o];
        let awq_zero = qzero_unpacked[g][o];
        // Every nibble in this group must dequantize identically.
        for i_in in 0..(group_size as usize) {
          let i = g * (group_size as usize) + i_in;
          let nibble = awq_unpacked[i][o];
          let awq_dq = awq_dequant(nibble, awq_zero, awq_scale);
          let mlx_dq = mlx_dequant(nibble, mlx_scale, mlx_bias);
          assert!(
            (awq_dq - mlx_dq).abs() < 1e-4,
            "AWQ dequant {awq_dq} != MLX dequant {mlx_dq} at (o={o}, g={g}, i={i}, nibble={nibble})"
          );
        }
      }
    }
  }

  /// Multiple AWQ-formatted layers in one input map: all transform correctly,
  /// the PerLayerQuantization is a single global entry, and the non-AWQ keys
  /// pass through verbatim.
  #[test]
  fn transform_awq_weights_handles_multiple_layers() {
    let group_size = 4u32;
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;
    // Two layers with all-zero qweight/qzeros + nonzero scales — verify both
    // exist + the pass-through key is preserved.
    let make_weights = |prefix: &str| -> Vec<(String, Array)> {
      let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
      let qz = Array::from_slice::<u32>(&vec![0u32; n_groups], &(n_groups, 1)).unwrap();
      let scales_data: Vec<f32> = (0..n_groups * out_features)
        .map(|i| 0.1_f32 * (i as f32 + 1.0))
        .collect();
      let sc = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();
      vec![
        (format!("{prefix}.qweight"), qw),
        (format!("{prefix}.qzeros"), qz),
        (format!("{prefix}.scales"), sc),
      ]
    };
    let mut weights: Weights = HashMap::new();
    for (k, v) in make_weights("layer0.q") {
      weights.insert(k, v);
    }
    for (k, v) in make_weights("layer1.q") {
      weights.insert(k, v);
    }
    // Pass-through key (e.g. `embed_tokens.weight`).
    let passthrough = Array::from_slice::<f32>(&[1.0_f32; 16], &(2usize, 8)).unwrap();
    weights.insert("embed_tokens.weight".to_string(), passthrough);

    let config = AwqLoadConfig {
      bits: 4,
      group_size,
      zero_point: true,
      version: String::new(),
    };
    let (out, plq) = transform_awq_weights(weights, &config).expect("transform");

    // Both layers transformed.
    assert!(out.contains_key("layer0.q.weight"));
    assert!(out.contains_key("layer0.q.scales"));
    assert!(out.contains_key("layer0.q.biases"));
    assert!(out.contains_key("layer1.q.weight"));
    assert!(out.contains_key("layer1.q.scales"));
    assert!(out.contains_key("layer1.q.biases"));
    // Originals gone.
    assert!(!out.contains_key("layer0.q.qweight"));
    assert!(!out.contains_key("layer1.q.qzeros"));
    // Pass-through preserved.
    let mut pt = out
      .get("embed_tokens.weight")
      .expect("pass-through")
      .try_clone()
      .unwrap();
    assert_eq!(pt.shape(), vec![2, 8]);
    assert_eq!(pt.to_vec::<f32>().unwrap(), vec![1.0_f32; 16]);
    // PerLayerQuantization global is set, per-layer empty.
    let g = plq.quantization.unwrap();
    assert_eq!(g.group_size, group_size as i32);
    assert_eq!(g.bits, 4);
    assert!(plq.per_layer.is_empty());
  }

  /// A `.qweight` with no `.scales` companion is rejected with a clear
  /// message (mirrors mlx-lm's implicit `KeyError`, `utils.py:109`).
  #[test]
  fn transform_awq_weights_rejects_missing_scales() {
    let in_features = 8usize;
    let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert("layer.qweight".to_string(), qw);
    // No scales, no qzeros.
    let config = AwqLoadConfig::default();
    let err = transform_awq_weights(weights, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
      msg.contains("scales") && msg.contains("layer.qweight"),
      "error must name the missing scales + offending qweight key, got: {msg}"
    );
  }

  /// qweight/scales shape mismatch is rejected with a clear ShapeMismatch.
  #[test]
  fn transform_awq_weights_rejects_mismatched_shapes() {
    let qw = Array::from_slice::<u32>(&[0u32; 8], &(8usize, 1)).unwrap();
    // Mismatched scales: should be [n_groups=2, out_features=8] but we give [4, 8]
    let sc = Array::from_slice::<f32>(&[0.1_f32; 32], &(4usize, 8)).unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert("layer.qweight".to_string(), qw);
    weights.insert("layer.scales".to_string(), sc);
    let config = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: false,
      version: String::new(),
    };
    let err = transform_awq_weights(weights, &config).unwrap_err();
    assert!(
      matches!(err, Error::ShapeMismatch { .. }),
      "expected ShapeMismatch, got {err:?}"
    );
  }

  /// A `.g_idx` key (GPTQ non-contiguous-group reorder) is rejected upfront.
  #[test]
  fn transform_awq_weights_rejects_g_idx() {
    let qw = Array::from_slice::<u32>(&[0u32; 8], &(8usize, 1)).unwrap();
    let sc = Array::from_slice::<f32>(&[0.1_f32; 16], &(2usize, 8)).unwrap();
    let gidx = Array::from_slice::<i32>(&[0i32, 1, 0, 1, 0, 1, 0, 1], &(8usize,)).unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert("layer.qweight".to_string(), qw);
    weights.insert("layer.scales".to_string(), sc);
    weights.insert("layer.g_idx".to_string(), gidx);
    let config = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: false,
      version: String::new(),
    };
    let err = transform_awq_weights(weights, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
      msg.contains("g_idx"),
      "error must mention g_idx, got: {msg}"
    );
  }

  /// Non-4 bits is rejected with a clear message.
  #[test]
  fn transform_awq_weights_rejects_non_4_bits() {
    let weights: Weights = HashMap::new();
    let config = AwqLoadConfig {
      bits: 8,
      ..AwqLoadConfig::default()
    };
    let err = transform_awq_weights(weights, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("bits=4"), "error must say bits=4, got: {msg}");
  }

  /// Symmetric quantization (`zero_point: false`): biases are computed from
  /// the implicit `2^(bits-1) = 8` zero point, NOT from any qzeros that
  /// might happen to be present.
  #[test]
  fn transform_awq_weights_symmetric_uses_implicit_zero() {
    let in_features = 8usize;
    let out_features = 8usize;
    let group_size = 4u32;
    let n_groups = 2usize;
    let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    // Scale = 1.0 everywhere so the bias check is trivial: bias = -8 * 1 = -8.
    let scales_data: Vec<f32> = vec![1.0_f32; n_groups * out_features];
    let sc = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert("layer.qweight".to_string(), qw);
    weights.insert("layer.scales".to_string(), sc);

    let config = AwqLoadConfig {
      bits: 4,
      group_size,
      zero_point: false,
      version: String::new(),
    };
    let (out, _) = transform_awq_weights(weights, &config).expect("transform");
    let mut biases_arr = out
      .get("layer.biases")
      .expect("layer.biases")
      .try_clone()
      .unwrap();
    let biases: Vec<f32> = biases_arr.to_vec().unwrap();
    // Every entry must be exactly -8.0.
    for &b in &biases {
      assert!(
        (b + 8.0_f32).abs() < 1e-5,
        "symmetric bias must be -2^(bits-1) * scale = -8.0, got {b}"
      );
    }
  }

  /// Empty input (no `.qweight` keys) is a no-op: pass-through verbatim plus
  /// a `PerLayerQuantization` with the requested global params.
  #[test]
  fn transform_awq_weights_empty_input_is_noop() {
    let pt = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0], &(3usize,)).unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert("layer.weight".to_string(), pt);

    let config = AwqLoadConfig::default();
    let (out, plq) = transform_awq_weights(weights, &config).expect("transform");
    // Pass-through preserved.
    let mut got = out
      .get("layer.weight")
      .expect("pass-through")
      .try_clone()
      .unwrap();
    assert_eq!(got.to_vec::<f32>().unwrap(), vec![1.0_f32, 2.0, 3.0]);
    // Global quant set from config defaults.
    let g = plq.quantization.unwrap();
    assert_eq!(g.bits, 4);
    assert_eq!(g.group_size, 128);
    assert_eq!(g.mode, QuantMode::Affine);
  }

  // ──────────────── AwqLoadConfig ────────────────

  /// AwqLoadConfig round-trips through serde from a typical AutoAWQ
  /// `quantization_config` JSON block.
  #[test]
  fn awq_load_config_parses_quantization_json() {
    let json = r#"{
      "bits": 4,
      "group_size": 128,
      "zero_point": true,
      "version": "gemm"
    }"#;
    let cfg: AwqLoadConfig = serde_json::from_str(json).expect("parse");
    assert_eq!(cfg.bits, 4);
    assert_eq!(cfg.group_size, 128);
    assert!(cfg.zero_point);
    assert_eq!(cfg.version, "gemm");
  }

  /// Defaults populate when keys are absent (AutoAWQ omitted-field convention).
  #[test]
  fn awq_load_config_defaults_when_keys_absent() {
    let cfg: AwqLoadConfig = serde_json::from_str("{}").expect("parse");
    assert_eq!(cfg.bits, 4);
    assert_eq!(cfg.group_size, 128);
    assert!(cfg.zero_point);
    assert_eq!(cfg.version, "");
  }

  /// Default impl matches the JSON-deserialized defaults (audit cross-check).
  #[test]
  fn awq_load_config_default_matches_serde_default() {
    let from_default = AwqLoadConfig::default();
    let from_serde: AwqLoadConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(from_default, from_serde);
  }
}
