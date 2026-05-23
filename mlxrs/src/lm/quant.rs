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

use std::collections::{BTreeSet, HashMap};

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
//
// # Scope
//
// This section ports the on-load CONVERSION half of mlx-lm's AutoAWQ /
// GPTQ pipeline (`mlx-lm/mlx_lm/utils.py:72-172`): an already-quantized
// AutoAWQ checkpoint's `qweight` / `qzeros` / `scales` triple is
// transformed into MLX's native `weight` / `scales` / `biases` layout
// so the standard MLX quant load path can consume it.
//
// **Out of scope:**
//
// - **AWQ search-and-quantize** (`mlx-lm/mlx_lm/quant/awq.py`,
//   ~585 lines: `awq_quantize`, `search_best_scale`, `apply_scale`,
//   `scale_block`, `search_best_clip`, `clip_block`, `run_layer`, ...).
//   That is the TRAINING half — searching for per-channel scale + clip
//   ranges to minimize quantization error. Callers that need to mint a
//   fresh AutoAWQ checkpoint must do so upstream (via the python
//   `mlx_lm.quant.awq` module) and feed the resulting on-disk
//   checkpoint through [`transform_awq_weights`] at load time. A
//   future port of the search path is tracked separately; flagged here
//   as a follow-up.
//
// - **ParoQuant loader + `RotateQuantizedLinear`** (mlx-swift-lm
//   `Libraries/MLXLMCommon/ParoQuant/ParoQuantLoader.swift` +
//   `.../RotateQuantizedLinear.swift`). These ARE present in the
//   swift ref but are intentionally NOT ported here:
//
//   1. `ParoQuantLoader.loadParoQuantModel` is a **per-model loader**
//      gated on `architectures == ["Qwen3_5ForConditionalGeneration"]`
//      (`ParoQuantLoader.swift:52`). It is not a general-purpose
//      loader, just a Qwen3.5-PARO entry point that reuses the AWQ
//      unpack-and-repack we DO port (see `convertAutoAWQ` in the
//      swift ref, which is a re-implementation of
//      `_transform_awq_weights` parameterized on the PARO key set).
//      Per `[[project_no_per_model_arch_porting]]`: mlxrs ports
//      loaders/tokenizers/pooling — not per-usecase
//      model-architecture loaders.
//
//   2. `RotateQuantizedLinear` is a `QuantizedLinear` subclass that
//      overrides `callAsFunction` to fuse a pairwise-Givens rotation
//      into the forward pass via a **Metal compute kernel** built
//      with `MLXFast.metalKernel(name:inputNames:outputNames:source:)`.
//      mlxrs does not currently expose a Module/Linear hierarchy
//      (per the project scope above) and does not wrap mlx-c's
//      general `mlx_fast_metal_kernel_config` surface (only
//      `mlx_fast_layer_norm` / `mlx_fast_rms_norm` are bound, in
//      `crate::embeddings::fast`). Porting the layer would require
//      both substrates first and is therefore deferred to the
//      per-usecase consumer that needs a Qwen3.5-PARO inference
//      pipeline.
//
//   The AWQ on-load conversion this section ports IS the load-time
//   substrate ParoQuant relies on (`convertAutoAWQ` in the swift ref
//   is the same algorithm with the same inverse-permutation +
//   transpose + repack steps); a future ParoQuant port will sit on
//   top of [`transform_awq_weights`] rather than re-implementing it.

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

/// Unpack an AutoAWQ-packed 4-bit `qweight` (32-bit packed nibbles, 8 per
/// element) into the dense natural-order layout — port of
/// `_unpack_awq_weights` (`mlx-lm/mlx_lm/utils.py:72-82`).
///
/// `qweight` must be a 2-D array of dtype `uint32` OR `int32` (AutoAWQ's
/// `WQLinear_GEMM` allocates its packed buffer with `torch.int32`, so on-disk
/// safetensors carry the SIGNED dtype — mlx-lm's python `_unpack_awq_weights`
/// performs the shift-and-mask without any unsigned-only gate, so we accept
/// both). For `int32` inputs the buffer is bit-preservingly reinterpreted
/// via [`ops::misc::view`] (the MLX `mx.view` primitive — `mlx/ops.cpp`
/// `array view(const array& a, const Dtype& dtype, ...)`) — this keeps a
/// negative i32's sign bit as the u32 MSB, so the subsequent shift-and-mask
/// over `AWQ_SHIFTS` (which contains a `28`-bit shift) recovers the top
/// nibble. A value-preserving `astype` would clamp negatives to `0` and lose
/// the high nibble entirely — that is the bug this gate prevents.
///
/// Output shape is `[rows, packed_cols * 8]`, dtype `uint32`, with each
/// position holding the 4-bit nibble in `[0, 15]`. `transform_awq_weights`
/// then re-casts to `uint32` explicitly before its repack
/// (`utils.py:130`) — the dtype of the output here is already `uint32`
/// since the view/already-u32 input flows through the bit-ops without
/// dtype promotion.
///
/// The internal `AWQ_SHIFTS` table folds the AutoAWQ packing-reorder
/// into the unpack: a single `(qweight >> shifts) & mask` over the
/// scaled inverse permutation `[0, 4, 1, 5, 2, 6, 3, 7] * bits` yields
/// the nibbles in their natural sequential order. The swift
/// `ParoQuantLoader.unpackAndReorder` form does it in two steps
/// (`unpack with arange(8) * bits` then `take(inverseReorder)`) — the
/// algebraic result is identical.
///
/// Mirrors the mlx-lm 2-D contract verbatim; non-2D inputs are a
/// [`Error::ShapeMismatch`] (the python version would `ValueError`
/// during the trailing `.reshape(out_features, in_features)`). Dtypes
/// other than `uint32` / `int32` are rejected as [`Error::Backend`].
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
  // AutoAWQ allocates `qweight` / `qzeros` with `torch.int32` (signed) —
  // mlx-lm's `_unpack_awq_weights` (`utils.py:72-82`) does the shift-and-mask
  // without an unsigned-only gate, so we accept both dtypes. Other dtypes
  // (floats, etc.) are still a layout error and rejected.
  //
  // For I32 input we bit-preservingly reinterpret to U32 via the MLX `view`
  // primitive (NOT `astype`). For same-width dtypes `view` keeps the
  // underlying bit-pattern intact, so a negative i32 (e.g. `0xF0FF_FFFF`
  // = -251_658_241) becomes the u32 with the same MSB. `astype` would do
  // a value-preserving cast that clamps negatives to 0 — losing the high
  // nibble. See `mlx/ops.cpp` `array view(...)` for the source semantics.
  let owned_view;
  let packed_u32: &Array = match dtype {
    Dtype::U32 => qweight,
    Dtype::I32 => {
      owned_view = ops::misc::view(qweight, Dtype::U32)?;
      &owned_view
    }
    other => {
      return Err(Error::Backend {
        message: format!(
          "unpack_awq_weights: AutoAWQ stores `qweight` as 32-bit packed nibbles \
           (`utils.py:72-82`) — accept `uint32` (mlx-lm canonical) or `int32` \
           (AutoAWQ `WQLinear_GEMM`'s default `torch.int32` allocation); got dtype {other:?}"
        ),
      });
    }
  };
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
  let expanded = ops::shape::expand_dims_axes(packed_u32, &[2])?;
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
/// **Fix 3 [HIGH]**: mlx-lm takes the LAST iterated layer's `scales.dtype`
/// as the target (`utils.py:156` overwrites `model_dtype` each iteration).
/// In Python that's `dict` insertion order, which for AutoAWQ checkpoints
/// usually means the last weight in the safetensors file. mlxrs originally
/// picked the LEX-LAST prefix to get a stable choice across HashMap
/// iteration orders, but for HETEROGENEOUS-PRECISION checkpoints
/// (e.g. some layers f16, some bf16) the lex-last pick is arbitrary and
/// would silently downcast the higher-precision layers.
///
/// Resolution policy: **highest precision wins** in HIERARCHICAL cases —
/// `F64 > F32 > BF16 / F16` (a wider format is a superset, so the cast
/// is lossless from the lower formats up). For ties at the same rank
/// (e.g. all bf16) the result is the first dtype with that rank — stable
/// across runs.
///
/// **F5 [HIGH] — F16 + BF16 mixed escalation**: F16 and BF16 are NOT
/// mutually-convertible without loss. Neither is a superset of the other:
/// - F16 has 10 mantissa bits + a 5-bit exponent (high precision, narrow range).
/// - BF16 has 7 mantissa bits + an 8-bit exponent (F32-equivalent range, low precision).
///
/// Example: the F16 value `1.0009765625` (= 1 + 2⁻¹⁰, exactly representable
/// in F16) rounds to `1.0` in BF16 — BF16's smallest delta near 1 is
/// 2⁻⁷ ≈ 0.0078. Conversely, BF16 magnitudes outside ±65 504 overflow F16.
///
/// So when a checkpoint mixes F16 and BF16 `.scales` with no F32/F64
/// present, neither half can losslessly hold the other; we **escalate to
/// F32** (a superset of both). This is a deliberate divergence from
/// mlx-lm's "last layer wins" — mlx-lm doesn't address heterogeneous
/// precision at all (`utils.py:156` overwrites unconditionally), so the
/// choice falls to mlxrs. F32 is chosen because it is the only common
/// superset that preserves every value from both formats.
///
/// Tested by `resolve_awq_model_dtype_escalates_f16_plus_bf16_to_f32`
/// (resolution path) and
/// `transform_awq_weights_preserves_f16_precision_when_mixed_with_bf16`
/// (end-to-end value preservation via the unification cast).
///
/// **F5 \[MEDIUM\] R3 — scope**: the dtype this fn resolves applies to the
/// AWQ-generated `.scales` / `.biases` outputs ONLY, **not** to the
/// pass-through floating tensors in the same checkpoint (embeddings, LM
/// head, norms, etc.). Earlier revisions ran the unification cast over
/// every floating key in `new_weights`, which for a large quantized model
/// with BF16/F16 embeddings + one mixed-half AWQ pair could DOUBLE the
/// resident size of those pass-through tensors and add a full-size cast
/// allocation during load — capable of turning a fitting model into OOM.
/// [`transform_awq_weights`] now iterates the unification loop over a
/// `BTreeSet<String>` of generated keys only; pass-through tensors retain
/// their on-disk dtype. See
/// `transform_awq_weights_does_not_widen_passthrough_bf16_tensor` and
/// siblings for the regression coverage.
///
/// Validation: assumes every `.scales` dtype was already gated as
/// floating by [`validate_awq_scales_are_floating`] — this fn does NOT
/// re-check (its caller MUST call the validator first).
fn resolve_awq_model_dtype(
  weights: &Weights,
  qweight_prefixes: &[String],
) -> Result<Option<Dtype>> {
  if qweight_prefixes.is_empty() {
    return Ok(None);
  }
  // Walk every `.scales` (preflight validator already gated them as
  // floating + present) and pick the highest-precision one. Deterministic
  // tiebreaker via `floating_dtype_precision_rank` (higher rank = more
  // precision; F64 > F32 > BF16 > F16). For ties at the same rank
  // (e.g. all bf16) the result is the first dtype with that rank —
  // stable across runs.
  //
  // Also track whether BOTH F16 and BF16 are present: in that case the
  // hierarchical "highest rank" answer (BF16) is LOSSY for the F16
  // layers, so we escalate to F32 unless an even-wider format
  // (F32/F64) is already present and wins on rank.
  let mut best: Option<Dtype> = None;
  let mut has_f16 = false;
  let mut has_bf16 = false;
  for prefix in qweight_prefixes {
    let scales_key = format!("{prefix}.scales");
    let scales = weights.get(&scales_key).ok_or_else(|| Error::Backend {
      message: format!(
        "transform_awq_weights: layer `{prefix}.qweight` is missing its companion \
           `{scales_key}` (AutoAWQ writes `.qweight` / `.scales` / `.qzeros` as a triple); \
           refusing to silently drop the layer"
      ),
    })?;
    let d = scales.dtype()?;
    match d {
      Dtype::F16 => has_f16 = true,
      Dtype::BF16 => has_bf16 = true,
      _ => {}
    }
    match best {
      None => best = Some(d),
      Some(prev) => {
        if floating_dtype_precision_rank(d) > floating_dtype_precision_rank(prev) {
          best = Some(d);
        }
      }
    }
  }
  // F5 escalation: F16+BF16 mixed without F32/F64 → promote to F32 to
  // avoid the lossy F16 → BF16 cast (see doc-comment above for the
  // bit-layout reason). When F32 or F64 is already present, its higher
  // rank wins via the loop above and is a superset of both halves —
  // no escalation needed.
  if let Some(b) = best
    && has_f16
    && has_bf16
    && b != Dtype::F32
    && b != Dtype::F64
  {
    best = Some(Dtype::F32);
  }
  Ok(best)
}

/// Fix 3 [HIGH]: precision rank for the floating dtypes that may appear as
/// AWQ `.scales`. Higher rank = more precision.
///
/// Order: `F64 > F32 > BF16 > F16 > anything-else (sentinel 0)`.
///
/// **Caveat (F5 [HIGH])**: this rank treats BF16 > F16 because BF16 has
/// the wider exponent (F32-equivalent dynamic range), but BF16 has
/// FEWER mantissa bits (7 vs F16's 10). Neither half is a superset of
/// the other, so when both appear together [`resolve_awq_model_dtype`]
/// must NOT just take the higher-rank one — it escalates to F32. The
/// rank order remains useful for hierarchical cases (F32 > F16,
/// F64 > BF16, etc., where each step IS a true superset).
fn floating_dtype_precision_rank(d: Dtype) -> u8 {
  match d {
    Dtype::F64 => 4,
    Dtype::F32 => 3,
    Dtype::BF16 => 2,
    Dtype::F16 => 1,
    _ => 0,
  }
}

/// Fix 3 [HIGH]: validate that every AWQ `.scales` tensor is a SUPPORTED
/// FLOATING dtype. Without this gate, a hostile/malformed checkpoint with
/// integer `.scales` (e.g. `i32`, `u8`) would propagate that dtype through
/// [`resolve_awq_model_dtype`] and the unification loop would then CAST
/// every model floating weight to that integer dtype — corrupting the
/// entire model.
///
/// "Supported floating" = the mlx-python `mx.issubdtype(dtype, mx.floating)`
/// set (`utils.py:164`): `F16`, `F32`, `F64`, `BF16`. (F64 is included for
/// upstream parity even though Metal has no native f64 support — the cast
/// would still happen losslessly via the CPU path.)
///
/// On the first non-floating dtype encountered, returns `Err(Backend)`
/// naming the offending layer + dtype.
fn validate_awq_scales_are_floating(weights: &Weights, qweight_prefixes: &[String]) -> Result<()> {
  for prefix in qweight_prefixes {
    let scales_key = format!("{prefix}.scales");
    // Missing `.scales` is a different error class (caught by the
    // sibling-validation pass); skip silently here so the rejection
    // message stays scoped to the dtype contract.
    let Some(scales) = weights.get(&scales_key) else {
      continue;
    };
    let d = scales.dtype()?;
    if !is_floating(d) {
      return Err(Error::Backend {
        message: format!(
          "transform_awq_weights: layer `{scales_key}` has non-floating dtype {d:?}; \
           AutoAWQ `.scales` MUST be a floating type (F16 / F32 / F64 / BF16) — \
           any other dtype would corrupt the dtype-unification cast"
        ),
      });
    }
  }
  Ok(())
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
/// 6. **Floating-dtype unification** (`utils.py:163-165`): every AWQ-
///    generated `.scales` / `.biases` is cast to the resolved
///    `model_dtype` (see `resolve_awq_model_dtype` in this module).
///    **F5 \[MEDIUM\] R3** — scope: the cast walks only the keys this
///    function INSERTED into `new_weights` (the generated `.scales` +
///    `.biases` per converted prefix), tracked in a `BTreeSet<String>`
///    during the conversion pass. Pass-through floating tensors
///    (embeddings, LM head, norms, etc.) keep their original on-disk
///    dtype; they are not touched by the unification cast and so do not
///    inflate their resident size when a mixed-half checkpoint escalates
///    to F32.
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
  // Fix 2 [HIGH]: reject `version = "gemv"` and any other non-GEMM version
  // BEFORE any processing. AwqLoadConfig advertises {"gemm" / "gemv"} but
  // the converter unconditionally assumes GEMM layout. GEMV has different
  // qweight shape + scales layout + sequential packing — a GEMV checkpoint
  // either rejects at an unrelated shape-check OR silently mis-converts.
  // Gate it here so the failure mode is a clear "not supported" error
  // instead of corrupt inference. Empty `""` accepts the serde default
  // (the field's `#[serde(default)]` is `String::new()` — older AutoAWQ
  // checkpoints + mlxrs-internal construction both leave it empty, and
  // historically that has meant "any" / "gemm").
  match config.version.as_str() {
    "" | "gemm" => { /* proceed */ }
    "gemv" => {
      return Err(Error::Backend {
        message: "AWQ version 'gemv' not yet supported — only 'gemm' is implemented. \
                  GEMV checkpoints use a different qweight shape, scales layout, and \
                  sequential packing; converting one through the GEMM path would silently \
                  produce corrupt weights. See upstream AutoAWQ for GEMV-layout details. \
                  Re-quantize with `awq --version gemm` if possible."
          .to_string(),
      });
    }
    other => {
      return Err(Error::Backend {
        message: format!(
          "transform_awq_weights: AWQ version '{other}' not recognized (expected 'gemm' or empty)"
        ),
      });
    }
  }
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

  // Collect every prefix of a `.qweight` key (sorted, for deterministic
  // iteration in tests; `resolve_awq_model_dtype` uses a precision rank
  // independent of order — see Fix 3 below).
  let mut qweight_prefixes: Vec<String> = weights
    .keys()
    .filter_map(|k| k.strip_suffix(".qweight").map(str::to_string))
    .collect();
  qweight_prefixes.sort();

  // Fix 3 [HIGH]: gate every `.scales` dtype as floating BEFORE resolving
  // model_dtype. Without this, an integer `.scales` would propagate to the
  // unification loop and cast every model float to that integer — silently
  // corrupting the model. Spec'd `validate_awq_scales_are_floating`.
  validate_awq_scales_are_floating(&weights, &qweight_prefixes)?;

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
    // Fix 4 [HIGH]: collision check with `<prefix>.weight` / `<prefix>.biases`
    // siblings. The converter emits `<prefix>.weight` + `<prefix>.scales`
    // (+ `<prefix>.biases` for affine) and then unconditionally inserts the
    // remainder keys into the output map. If the input ALSO carries a stale
    // `<prefix>.weight` (a dense weight left over from a partial conversion,
    // a misnamed pair of files, or hostile content) the remainder pass would
    // OVERWRITE the freshly-generated AWQ output — yielding corrupt inference
    // or a deferred load failure. The pre-existing non-AWQ `quantize_weights`
    // path already refuses analogous orphan/stale collisions (see
    // `classify_triple` / `TripleClass::Invalid` — "refusing to silently
    // overwrite the generated bias"). Mirror that contract here so the
    // failure mode is a clear preflight error, not silent corruption.
    let weight_key = format!("{prefix}.weight");
    if weights.contains_key(&weight_key) {
      return Err(Error::Backend {
        message: format!(
          "AWQ conversion collision: input contains both `{qweight_key}` (to be converted) and \
           `{weight_key}` (would be overwritten by the generated AWQ output). Remove the stale \
           dense weight before fusing. (Precedent: the non-AWQ `quantize_weights` path also \
           refuses analogous orphan/stale collisions — see `classify_triple` in this module.)"
        ),
      });
    }
    let biases_key = format!("{prefix}.biases");
    if weights.contains_key(&biases_key) {
      return Err(Error::Backend {
        message: format!(
          "AWQ conversion collision: input contains both `{qweight_key}` (to be converted) and \
           `{biases_key}` (would be overwritten by the generated AWQ output). Remove the stale \
           biases before fusing. (Precedent: the non-AWQ `quantize_weights` path also refuses \
           analogous orphan/stale collisions — see `classify_triple` in this module.)"
        ),
      });
    }
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
    // Fix 1 [CRITICAL]: dtype preflight. AutoAWQ's `WQLinear_GEMM`
    // allocates `qweight` / `qzeros` as `torch.int32` (signed); mlx-lm's
    // canonical converter expects `uint32`. Accept BOTH — but reject other
    // dtypes (floats, narrower ints, etc.) here with a clear message, so a
    // hostile/malformed checkpoint cannot slip past to mid-pipeline.
    // `unpack_awq_weights` performs the bit-preserving I32 → U32 view
    // internally; this gate just surfaces the wrong-dtype case as a
    // preflight rather than a per-layer error during the conversion pass.
    let qw_dtype = qweight.dtype()?;
    if !matches!(qw_dtype, Dtype::U32 | Dtype::I32) {
      return Err(Error::Backend {
        message: format!(
          "transform_awq_weights: layer `{qweight_key}`: qweight dtype {qw_dtype:?} not supported \
           — AutoAWQ stores packed nibbles as `uint32` (mlx-lm canonical) or `int32` \
           (AutoAWQ `WQLinear_GEMM` default `torch.int32` allocation); reject all other dtypes"
        ),
      });
    }
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
      // Same dtype gate as qweight — accept U32 (mlx-lm canonical) or I32
      // (AutoAWQ `torch.int32`), reject other dtypes here at preflight.
      let qz_dtype = qzeros.dtype()?;
      if !matches!(qz_dtype, Dtype::U32 | Dtype::I32) {
        return Err(Error::Backend {
          message: format!(
            "transform_awq_weights: layer `{qzeros_key}`: qzeros dtype {qz_dtype:?} not supported \
             — accept `uint32` (mlx-lm canonical) or `int32` (AutoAWQ default `torch.int32`)"
          ),
        });
      }
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
  // F5 R3 [MEDIUM] — track the AWQ-generated `.scales` / `.biases` keys
  // INSERTED by the conversion pass below. The post-loop unification cast
  // walks only this set, so pass-through floating tensors (embeddings, LM
  // head, norms, etc.) keep their on-disk dtype and are not widened when a
  // mixed-half checkpoint escalates `model_dtype` to F32. Without this
  // scoping, a single F16+BF16 AWQ pair could double the resident size of
  // every BF16/F16 pass-through tensor in the checkpoint — capable of
  // turning a fitting model into OOM during load. (The generated
  // `.weight` is U32, not floating, so it is never a unification target
  // either way; we still leave it out of the set to make the
  // "only generated FLOATING outputs" contract explicit.)
  let mut awq_generated_floating_keys: BTreeSet<String> = BTreeSet::new();
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

    let scales_key = format!("{prefix}.scales");
    let biases_key = format!("{prefix}.biases");
    new_weights.insert(format!("{prefix}.weight"), new_weight);
    new_weights.insert(scales_key.clone(), scales_c);
    new_weights.insert(biases_key.clone(), biases);
    // F5 R3 [MEDIUM]: record the AWQ-generated floating outputs so the
    // unification cast below can scope itself to them. (`.weight` is U32,
    // not in scope for floating unification.)
    awq_generated_floating_keys.insert(scales_key);
    awq_generated_floating_keys.insert(biases_key);
  }

  // Pass-through pass for non-AWQ keys (`utils.py:158-161`). The
  // remainder flows verbatim; we deliberately do NOT touch its dtype.
  for (key, arr) in remainder {
    new_weights.insert(key, arr);
  }
  // F5 R3 [MEDIUM] — Floating-dtype unification (`utils.py:163-165`)
  // scoped to AWQ-generated `.scales` / `.biases` ONLY. mlx-lm runs this
  // cast over every floating key in the resulting dict, but doing so in a
  // Rust port (where the input map carries every pass-through tensor —
  // embeddings, LM head, norms, etc.) means a single mixed-half AWQ pair
  // (F16+BF16 → F32 escalation, see `resolve_awq_model_dtype` F5 [HIGH])
  // can double the resident size of every BF16/F16 pass-through tensor
  // plus allocate a full-size cast buffer per tensor during load. For
  // large quantized models that turns a fitting checkpoint into OOM.
  // Walk only the keys this fn inserted; pass-through tensors keep their
  // on-disk dtype.
  if let Some(target) = model_dtype {
    for key in &awq_generated_floating_keys {
      let arr = new_weights
        .get(key)
        .expect("AWQ-generated key inserted moments ago must still be present");
      let d = arr.dtype()?;
      if is_floating(d) && d != target {
        let cast = ops::misc::astype(arr, target)?;
        new_weights.insert(key.clone(), cast);
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

  /// Non-32-bit-int dtype is rejected. AutoAWQ allocates `qweight` /
  /// `qzeros` as `torch.int32` (signed) — we accept both `u32` AND `i32`
  /// (see [`unpack_awq_weights_accepts_i32_input`]), but anything else
  /// (floats, narrower ints, etc.) is a layout mismatch the caller should
  /// fix upstream.
  #[test]
  fn unpack_awq_weights_rejects_non_32bit_int_dtype() {
    // f32 is the canonical "wrong" dtype to test against — narrow ints,
    // bool, and floats all hit the same `other => Err(Backend)` arm.
    let r = Array::from_slice::<f32>(&[0.0_f32; 4], &(2usize, 2)).unwrap();
    let err = unpack_awq_weights(&r).unwrap_err();
    assert!(
      matches!(err, Error::Backend { .. }),
      "f32 dtype should be Backend, got {err:?}"
    );
  }

  /// Fix 1 [CRITICAL]: i32 input is accepted (AutoAWQ's `WQLinear_GEMM`
  /// allocates packed buffers as `torch.int32`, so standard on-disk
  /// checkpoints carry the signed dtype). Output matches what the equivalent
  /// u32 input would produce — verifying the bit-preserving reinterpret.
  #[test]
  fn unpack_awq_weights_accepts_i32_input() {
    // Pick a packed value whose high bit is SET — this is the case the
    // bug would corrupt: a value-preserving cast would clamp the negative
    // i32 to 0 (or saturate), losing the high nibble. The bit-preserving
    // view keeps `0xF` in the MSB nibble.
    let raw: u32 = 0xF0FF_FFFF;
    let signed: i32 = raw as i32;
    assert!(
      signed < 0,
      "fixture must be negative to exercise the sign bit"
    );
    let i32_packed = Array::from_slice::<i32>(&[signed], &(1usize, 1)).unwrap();
    let u32_packed = Array::from_slice::<u32>(&[raw], &(1usize, 1)).unwrap();

    let mut from_i32 = unpack_awq_weights(&i32_packed).expect("i32 input should be accepted");
    let mut from_u32 = unpack_awq_weights(&u32_packed).expect("u32 input still accepted");
    assert_eq!(from_i32.shape(), vec![1, 8]);
    assert_eq!(from_u32.shape(), vec![1, 8]);
    assert_eq!(from_i32.dtype().unwrap(), Dtype::U32);
    let i32_nibbles = from_i32.to_vec::<u32>().unwrap();
    let u32_nibbles = from_u32.to_vec::<u32>().unwrap();
    assert_eq!(
      i32_nibbles, u32_nibbles,
      "i32 input must produce the SAME nibbles as the equivalent u32 input (bit-preserving)"
    );
  }

  /// Fix 1: existing u32 inputs continue to work (regression guard for the
  /// `Cow::Borrowed(qweight)` short-circuit path).
  #[test]
  fn unpack_awq_weights_accepts_u32_input() {
    let raw: u32 = 0xF0FF_FFFF;
    let packed = Array::from_slice::<u32>(&[raw], &(1usize, 1)).unwrap();
    let out = unpack_awq_weights(&packed).expect("u32 input accepted");
    assert_eq!(out.shape(), vec![1, 8]);
    assert_eq!(out.dtype().unwrap(), Dtype::U32);
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

  // ──────────────── Fix 2 [HIGH]: version validation ────────────────

  /// Helper: build a minimal valid GEMM-shaped weights map (in=8, out=8,
  /// gs=4, ng=2). Lets the F2 tests focus on the version field without
  /// re-deriving the shape arithmetic each time.
  fn awq_gemm_fixture_weights() -> Weights {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;
    let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let qz = Array::from_slice::<u32>(&vec![0u32; n_groups], &(n_groups, 1)).unwrap();
    let scales_data: Vec<f32> = vec![1.0_f32; n_groups * out_features];
    let sc = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();
    let mut w: Weights = HashMap::new();
    w.insert("layer.qweight".to_string(), qw);
    w.insert("layer.qzeros".to_string(), qz);
    w.insert("layer.scales".to_string(), sc);
    w
  }

  /// Fix 2: `version = "gemv"` is REJECTED at the top of transform_awq_weights
  /// (before any conversion work). The error message must name the offending
  /// version and call out "not yet supported" — the spec-required signal.
  #[test]
  fn transform_awq_weights_rejects_gemv_version() {
    let config = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: "gemv".into(),
    };
    let err = transform_awq_weights(awq_gemm_fixture_weights(), &config).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("gemv"),
          "error must name the offending 'gemv' version, got: {message}"
        );
        assert!(
          message.contains("not yet supported"),
          "error must say 'not yet supported', got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 2: an unknown version string is REJECTED with the version named
  /// in the message.
  #[test]
  fn transform_awq_weights_rejects_unknown_version() {
    let config = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: "unsupported".into(),
    };
    let err = transform_awq_weights(awq_gemm_fixture_weights(), &config).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("unsupported"),
          "error must name the offending version, got: {message}"
        );
        assert!(
          message.contains("not recognized"),
          "error must say 'not recognized', got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 2: empty version (the serde default) is ACCEPTED — older AutoAWQ
  /// checkpoints + mlxrs-internal construction both leave it empty.
  #[test]
  fn transform_awq_weights_accepts_empty_version() {
    let config = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: String::new(),
    };
    transform_awq_weights(awq_gemm_fixture_weights(), &config)
      .expect("empty version (serde default) must be accepted");
  }

  /// Fix 2: explicit `"gemm"` is ACCEPTED.
  #[test]
  fn transform_awq_weights_accepts_gemm_version() {
    let config = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: "gemm".into(),
    };
    transform_awq_weights(awq_gemm_fixture_weights(), &config)
      .expect("explicit 'gemm' version must be accepted");
  }

  // ──────────────── Fix 1 [CRITICAL]: I32 qweight/qzeros acceptance ────────────────

  /// Fix 1: a full I32 fixture (both qweight + qzeros allocated as `torch.int32`,
  /// as AutoAWQ's `WQLinear_GEMM` does) round-trips through `transform_awq_weights`.
  /// Includes a qweight value with the high bit SET — the bit-pattern that
  /// would corrupt under a value-preserving `astype`.
  #[test]
  fn transform_awq_weights_accepts_i32_qweight_and_qzeros() {
    // Same shapes as the round-trip fixture: in=8, out=8, gs=4, ng=2.
    let in_features = 8usize;
    let out_features = 8usize;
    let group_size = 4u32;
    let n_groups = 2usize;
    // Pack a row with the high nibble set so the resulting u32 word's MSB
    // is `0xF` — when allocated as i32 this is a negative number.
    let qweight_data_u32: Vec<u32> = (0..in_features)
      .map(|i| {
        let nibbles = [
          (i % 16) as u32,
          ((i + 1) % 16) as u32,
          ((i + 2) % 16) as u32,
          ((i + 3) % 16) as u32,
          ((i + 4) % 16) as u32,
          ((i + 5) % 16) as u32,
          ((i + 6) % 16) as u32,
          0xF_u32, // high nibble = 0xF → MSB set when packed at AWQ_SHIFTS[7]=28
        ];
        awq_pack_one_row(nibbles)
      })
      .collect();
    let qweight_data_i32: Vec<i32> = qweight_data_u32.iter().map(|&u| u as i32).collect();
    assert!(
      qweight_data_i32.iter().any(|&v| v < 0),
      "fixture must contain a negative i32 to exercise the high-bit case"
    );

    // qzeros: also int32, with same fixture as the round-trip test.
    let qzero_unpacked: Vec<Vec<u32>> = (0..n_groups)
      .map(|g| (0..out_features).map(|o| ((g + o) % 16) as u32).collect())
      .collect();
    let qzeros_data_u32: Vec<u32> = (0..n_groups)
      .map(|g| {
        let row: [u32; 8] = qzero_unpacked[g].to_vec().try_into().unwrap();
        awq_pack_one_row(row)
      })
      .collect();
    let qzeros_data_i32: Vec<i32> = qzeros_data_u32.iter().map(|&u| u as i32).collect();

    let scales_data: Vec<f32> = (0..n_groups * out_features)
      .map(|i| 0.1_f32 * (i as f32 + 1.0))
      .collect();

    // Build the I32 weights map.
    let qw_i32 = Array::from_slice::<i32>(&qweight_data_i32, &(in_features, 1)).unwrap();
    let qz_i32 = Array::from_slice::<i32>(&qzeros_data_i32, &(n_groups, 1)).unwrap();
    let sc = Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap();
    let mut weights_i32: Weights = HashMap::new();
    weights_i32.insert("layer.qweight".to_string(), qw_i32);
    weights_i32.insert("layer.qzeros".to_string(), qz_i32);
    weights_i32.insert("layer.scales".to_string(), sc);

    let config = AwqLoadConfig {
      bits: 4,
      group_size,
      zero_point: true,
      version: "gemm".into(),
    };
    let (out, plq) =
      transform_awq_weights(weights_i32, &config).expect("i32 qweight + qzeros accepted");

    // The transformed `.weight` must be u32 (the MLX quantized output dtype).
    let weight_arr = out.get("layer.weight").expect("layer.weight");
    assert_eq!(weight_arr.dtype().unwrap(), Dtype::U32);
    // PLQ unchanged.
    let g = plq.quantization.expect("global quant");
    assert_eq!(g.bits, 4);
    assert_eq!(g.group_size, group_size as i32);
  }

  /// Fix 1: pack a known-negative i32 fixture and verify the resulting
  /// MLX-format output bit-pattern matches what the equivalent U32 input
  /// produces — confirming the i32 path is bit-preserving end-to-end
  /// (NOT value-preserving via `astype`, which would clamp negatives to 0).
  #[test]
  fn transform_awq_weights_preserves_bit_pattern_on_i32_input() {
    let in_features = 8usize;
    let out_features = 8usize;
    let group_size = 4u32;
    let n_groups = 2usize;

    // Build identical fixtures, one allocated as u32, the other as the
    // bitwise-equal i32 — feed both through and compare the .weight output.
    let qweight_data_u32: Vec<u32> = (0..in_features)
      .map(|i| {
        // Same scrambled nibbles with high bit set in MSB slot.
        let nibbles = [
          (i % 16) as u32,
          ((i + 7) % 16) as u32,
          ((i + 3) % 16) as u32,
          ((i + 5) % 16) as u32,
          ((i + 2) % 16) as u32,
          ((i + 6) % 16) as u32,
          ((i + 1) % 16) as u32,
          0xF_u32,
        ];
        awq_pack_one_row(nibbles)
      })
      .collect();
    let qweight_data_i32: Vec<i32> = qweight_data_u32.iter().map(|&u| u as i32).collect();

    let qzeros_data_u32: Vec<u32> = vec![0_u32; n_groups];
    let qzeros_data_i32: Vec<i32> = vec![0_i32; n_groups];
    let scales_data: Vec<f32> = (0..n_groups * out_features)
      .map(|i| 0.5_f32 + (i as f32) * 0.01)
      .collect();

    let build = |qw_dtype_i32: bool| -> Weights {
      let mut w: Weights = HashMap::new();
      if qw_dtype_i32 {
        w.insert(
          "layer.qweight".to_string(),
          Array::from_slice::<i32>(&qweight_data_i32, &(in_features, 1)).unwrap(),
        );
        w.insert(
          "layer.qzeros".to_string(),
          Array::from_slice::<i32>(&qzeros_data_i32, &(n_groups, 1)).unwrap(),
        );
      } else {
        w.insert(
          "layer.qweight".to_string(),
          Array::from_slice::<u32>(&qweight_data_u32, &(in_features, 1)).unwrap(),
        );
        w.insert(
          "layer.qzeros".to_string(),
          Array::from_slice::<u32>(&qzeros_data_u32, &(n_groups, 1)).unwrap(),
        );
      }
      w.insert(
        "layer.scales".to_string(),
        Array::from_slice::<f32>(&scales_data, &(n_groups, out_features)).unwrap(),
      );
      w
    };

    let cfg = AwqLoadConfig {
      bits: 4,
      group_size,
      zero_point: true,
      version: "gemm".into(),
    };
    let (out_u32, _) = transform_awq_weights(build(false), &cfg).expect("u32 path");
    let (out_i32, _) = transform_awq_weights(build(true), &cfg).expect("i32 path");

    let mut w_u32 = out_u32.get("layer.weight").unwrap().try_clone().unwrap();
    let mut w_i32 = out_i32.get("layer.weight").unwrap().try_clone().unwrap();
    let u32_buf: Vec<u32> = w_u32.to_vec().unwrap();
    let i32_buf: Vec<u32> = w_i32.to_vec().unwrap();
    assert_eq!(
      u32_buf, i32_buf,
      "i32 qweight must produce the SAME .weight bit-pattern as the equivalent u32 input"
    );
  }

  // ──────────────── Fix 3 [HIGH]: .scales dtype validation ────────────────

  /// Fix 3: integer `.scales` (`i32`) is REJECTED — a hostile/malformed
  /// checkpoint with integer scales would silently CAST every model float
  /// to that integer through the dtype-unification loop. The validator
  /// fires first and names the offending layer + the rejection reason.
  #[test]
  fn transform_awq_weights_rejects_integer_scales_dtype() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;
    let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let qz = Array::from_slice::<u32>(&vec![0u32; n_groups], &(n_groups, 1)).unwrap();
    // INTEGER `.scales` — the bug class.
    let sc_int = Array::from_slice::<i32>(
      &vec![1_i32; n_groups * out_features],
      &(n_groups, out_features),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert("model.layer0.qweight".to_string(), qw);
    weights.insert("model.layer0.qzeros".to_string(), qz);
    weights.insert("model.layer0.scales".to_string(), sc_int);

    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: "gemm".into(),
    };
    let err = transform_awq_weights(weights, &cfg).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.layer0.scales"),
          "error must name the offending layer's `.scales` key, got: {message}"
        );
        assert!(
          message.contains("non-floating"),
          "error must call out the 'non-floating' rejection, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 3: `u8` (unsigned narrow int) `.scales` is REJECTED with the same
  /// error shape. Confirms the gate fires for narrow ints too — not just
  /// the canonical `i32` case.
  #[test]
  fn transform_awq_weights_rejects_uint_scales_dtype() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;
    let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let qz = Array::from_slice::<u32>(&vec![0u32; n_groups], &(n_groups, 1)).unwrap();
    let sc_u8 = Array::from_slice::<u8>(
      &vec![1_u8; n_groups * out_features],
      &(n_groups, out_features),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert("model.layer0.qweight".to_string(), qw);
    weights.insert("model.layer0.qzeros".to_string(), qz);
    weights.insert("model.layer0.scales".to_string(), sc_u8);

    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: "gemm".into(),
    };
    let err = transform_awq_weights(weights, &cfg).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("model.layer0.scales"),
          "error must name the offending layer's `.scales`, got: {message}"
        );
        assert!(
          message.contains("non-floating"),
          "error must call out 'non-floating', got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 3 / F5: HIERARCHICAL heterogeneous-precision `.scales` (mixing
  /// dtypes where one IS a true superset of the others) must resolve to
  /// the higher-precision target. This covers F32+F16 → F32 and F64+BF16
  /// → F64. The F5 fix carved out the F16+BF16 case (no superset relation,
  /// see `..._escalates_f16_plus_bf16_to_f32`) — this test guards the
  /// remaining cases where the simple "highest rank wins" answer IS still
  /// correct (and lossless).
  #[test]
  fn resolve_awq_model_dtype_uses_highest_when_hierarchical() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    // Case 1: F32 + F16 → F32 (F32 is a strict superset of F16).
    let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
      .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
      .collect();
    let f32_scales_data: Vec<f32> = (0..n_groups * out_features)
      .map(|i| 0.5 + 0.01 * (i as f32))
      .collect();

    let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_b = Array::from_slice::<f32>(&f32_scales_data, &(n_groups, out_features)).unwrap();

    let weights: Weights = HashMap::from([
      ("layer_a.qweight".to_string(), qw_a),
      ("layer_a.scales".to_string(), sc_a),
      ("layer_b.qweight".to_string(), qw_b),
      ("layer_b.scales".to_string(), sc_b),
    ]);
    let mut prefixes: Vec<String> = vec!["layer_a".to_string(), "layer_b".to_string()];
    prefixes.sort();

    validate_awq_scales_are_floating(&weights, &prefixes).expect("both floating, must pass");
    let resolved = resolve_awq_model_dtype(&weights, &prefixes)
      .unwrap()
      .expect("some dtype");
    assert_eq!(
      resolved,
      Dtype::F32,
      "F32+F16 hierarchical must resolve to F32 (superset), got {resolved:?}"
    );

    // Case 2: F64 + BF16 → F64 (F64 is a strict superset of BF16:
    // more mantissa bits AND more exponent bits).
    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();
    let f64_scales_data: Vec<f64> = (0..n_groups * out_features)
      .map(|i| 0.5 + 0.001 * (i as f64))
      .collect();

    let qw_c = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_c =
      Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_d = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_d = Array::from_slice::<f64>(&f64_scales_data, &(n_groups, out_features)).unwrap();

    let weights2: Weights = HashMap::from([
      ("layer_c.qweight".to_string(), qw_c),
      ("layer_c.scales".to_string(), sc_c),
      ("layer_d.qweight".to_string(), qw_d),
      ("layer_d.scales".to_string(), sc_d),
    ]);
    let mut prefixes2: Vec<String> = vec!["layer_c".to_string(), "layer_d".to_string()];
    prefixes2.sort();

    validate_awq_scales_are_floating(&weights2, &prefixes2).expect("both floating, must pass");
    let resolved2 = resolve_awq_model_dtype(&weights2, &prefixes2)
      .unwrap()
      .expect("some dtype");
    assert_eq!(
      resolved2,
      Dtype::F64,
      "F64+BF16 hierarchical must resolve to F64 (superset), got {resolved2:?}"
    );
  }

  /// F5 [HIGH]: F16 and BF16 mixed alone (no F32/F64 present) must
  /// escalate to F32. Neither half-float is a superset of the other —
  /// F16 has more mantissa bits, BF16 has more exponent bits — so any
  /// pick within the halves would be lossy for one side. The escalation
  /// to F32 is order-independent (HashMap iteration may visit them in
  /// either order via `prefixes`).
  #[test]
  fn resolve_awq_model_dtype_escalates_f16_plus_bf16_to_f32() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
      .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
      .collect();
    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();

    let build = || {
      let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
      let sc_a =
        Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
      let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
      let sc_b =
        Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
      let weights: Weights = HashMap::from([
        ("layer_a.qweight".to_string(), qw_a),
        ("layer_a.scales".to_string(), sc_a),
        ("layer_b.qweight".to_string(), qw_b),
        ("layer_b.scales".to_string(), sc_b),
      ]);
      weights
    };

    // Forward order: [layer_a (F16), layer_b (BF16)].
    let weights = build();
    let prefixes: Vec<String> = vec!["layer_a".to_string(), "layer_b".to_string()];
    validate_awq_scales_are_floating(&weights, &prefixes).expect("both floating, must pass");
    let resolved = resolve_awq_model_dtype(&weights, &prefixes)
      .unwrap()
      .expect("some dtype");
    assert_eq!(
      resolved,
      Dtype::F32,
      "F16+BF16 must escalate to F32 (no half is a superset), got {resolved:?}"
    );

    // Reverse order: [layer_b (BF16), layer_a (F16)]. Result must be
    // identical — escalation does not depend on iteration order.
    let weights_r = build();
    let prefixes_r: Vec<String> = vec!["layer_b".to_string(), "layer_a".to_string()];
    validate_awq_scales_are_floating(&weights_r, &prefixes_r).expect("both floating, must pass");
    let resolved_r = resolve_awq_model_dtype(&weights_r, &prefixes_r)
      .unwrap()
      .expect("some dtype");
    assert_eq!(
      resolved_r,
      Dtype::F32,
      "F16+BF16 reversed order must still escalate to F32, got {resolved_r:?}"
    );
  }

  /// F5: when F32 is already present alongside F16+BF16, it short-circuits
  /// the escalation — F32 wins on rank and is already a superset of both
  /// halves, no need to "escalate" further.
  #[test]
  fn resolve_awq_model_dtype_escalates_f16_plus_bf16_plus_f32_stays_at_f32() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
      .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
      .collect();
    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();
    let f32_scales_data: Vec<f32> = (0..n_groups * out_features)
      .map(|i| 0.25 + 0.001 * (i as f32))
      .collect();

    let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_b =
      Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_c = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_c = Array::from_slice::<f32>(&f32_scales_data, &(n_groups, out_features)).unwrap();

    let weights: Weights = HashMap::from([
      ("layer_a.qweight".to_string(), qw_a),
      ("layer_a.scales".to_string(), sc_a),
      ("layer_b.qweight".to_string(), qw_b),
      ("layer_b.scales".to_string(), sc_b),
      ("layer_c.qweight".to_string(), qw_c),
      ("layer_c.scales".to_string(), sc_c),
    ]);
    let prefixes: Vec<String> = vec![
      "layer_a".to_string(),
      "layer_b".to_string(),
      "layer_c".to_string(),
    ];
    validate_awq_scales_are_floating(&weights, &prefixes).expect("all floating, must pass");
    let resolved = resolve_awq_model_dtype(&weights, &prefixes)
      .unwrap()
      .expect("some dtype");
    assert_eq!(
      resolved,
      Dtype::F32,
      "F16+BF16+F32 must stay at F32 (F32 already > BF16 rank, no escalation), got {resolved:?}"
    );
  }

  /// F5: when F64 is already present alongside F16+BF16, it stays at F64
  /// (F64 outranks F32; F64 is also a superset of both halves so no
  /// escalation is needed).
  #[test]
  fn resolve_awq_model_dtype_escalates_f16_plus_bf16_with_f64_stays_at_f64() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
      .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
      .collect();
    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();
    let f64_scales_data: Vec<f64> = (0..n_groups * out_features)
      .map(|i| 0.25 + 0.001 * (i as f64))
      .collect();

    let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_b =
      Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_c = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_c = Array::from_slice::<f64>(&f64_scales_data, &(n_groups, out_features)).unwrap();

    let weights: Weights = HashMap::from([
      ("layer_a.qweight".to_string(), qw_a),
      ("layer_a.scales".to_string(), sc_a),
      ("layer_b.qweight".to_string(), qw_b),
      ("layer_b.scales".to_string(), sc_b),
      ("layer_c.qweight".to_string(), qw_c),
      ("layer_c.scales".to_string(), sc_c),
    ]);
    let prefixes: Vec<String> = vec![
      "layer_a".to_string(),
      "layer_b".to_string(),
      "layer_c".to_string(),
    ];
    validate_awq_scales_are_floating(&weights, &prefixes).expect("all floating, must pass");
    let resolved = resolve_awq_model_dtype(&weights, &prefixes)
      .unwrap()
      .expect("some dtype");
    assert_eq!(
      resolved,
      Dtype::F64,
      "F16+BF16+F64 must stay at F64 (F64 already > BF16 rank, no escalation), got {resolved:?}"
    );
  }

  /// F5 [HIGH] END-TO-END value preservation: a checkpoint with F16
  /// `.scales` carrying the value `1.0009765625` (= 1 + 2⁻¹⁰, exactly
  /// representable in F16 but NOT in BF16 — BF16's smallest delta near
  /// 1 is 2⁻⁷ ≈ 0.0078) and a sibling BF16 `.scales` layer must round-
  /// trip through `transform_awq_weights` with that F16 value PRESERVED.
  ///
  /// Under the pre-fix policy, the resolver returned BF16, the unification
  /// loop cast F16 → BF16, and `1.0009765625` collapsed to `1.0`
  /// (silently corrupting every F16 scale value). Under F5 the resolver
  /// escalates to F32, the cast is F16 → F32 (lossless), and the original
  /// value survives.
  #[test]
  fn transform_awq_weights_preserves_f16_precision_when_mixed_with_bf16() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    // F16 layer: every scale = 1.0009765625 (= 1 + 2⁻¹⁰), exactly
    // representable in F16 (bits 0x3C01). BF16 has 7 mantissa bits so
    // its smallest delta near 1.0 is 2⁻⁷ ≈ 0.0078125 — the value
    // would round to 1.0 if cast to BF16.
    let f16_value = half::f16::from_bits(0x3C01);
    assert_eq!(
      f16_value.to_f32(),
      1.0 + (2.0_f32).powi(-10),
      "F16 fixture value must be exactly 1 + 2^-10"
    );
    // Sanity-check the BF16 truncation the test catches:
    let bf_round = half::bf16::from_f32(f16_value.to_f32());
    assert_eq!(
      bf_round.to_f32(),
      1.0,
      "pre-condition: casting F16 1.0009765625 → BF16 must truncate to 1.0 \
       (this is the lossy behavior F5 prevents)"
    );

    let f16_scales_data: Vec<half::f16> = vec![f16_value; n_groups * out_features];
    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();

    let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_b =
      Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

    let weights: Weights = HashMap::from([
      ("layer_a.qweight".to_string(), qw_a),
      ("layer_a.scales".to_string(), sc_a),
      ("layer_b.qweight".to_string(), qw_b),
      ("layer_b.scales".to_string(), sc_b),
    ]);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: false, // symmetric — no qzeros required.
      version: "gemm".into(),
    };

    let (out, _) = transform_awq_weights(weights, &cfg).expect("transform must succeed");

    // The resolved/unified dtype must be F32 (escalation kicked in).
    let mut sc_a_out = out
      .get("layer_a.scales")
      .expect("converted layer_a.scales present")
      .try_clone()
      .unwrap();
    assert_eq!(
      sc_a_out.dtype().unwrap(),
      Dtype::F32,
      "unified dtype must be F32 under F5 escalation (was BF16 pre-fix)"
    );

    // Read back as F32 and verify EVERY element still holds 1.0009765625.
    let vals: Vec<f32> = sc_a_out.to_vec().expect("read back as F32");
    for (i, &v) in vals.iter().enumerate() {
      assert_eq!(
        v,
        1.0 + (2.0_f32).powi(-10),
        "layer_a.scales[{i}] = {v} (bits 0x{:08X}) — F16 1.0009765625 must NOT have \
         been truncated through BF16 (would land at 1.0 == 0x3F800000)",
        v.to_bits()
      );
    }

    // layer_b (originally BF16) was also unified to F32 (lossless from
    // BF16 → F32 — BF16 mantissa fits in F32's 23 bits trivially).
    let sc_b_out = out
      .get("layer_b.scales")
      .expect("converted layer_b.scales present");
    assert_eq!(
      sc_b_out.dtype().unwrap(),
      Dtype::F32,
      "layer_b.scales must also be unified to F32"
    );
  }

  /// F5 [HIGH] END-TO-END order-independence: same as the preservation
  /// test above, but with prefix names swapped lexicographically (BF16
  /// layer named to sort BEFORE the F16 layer). Guards against any
  /// regression that would reintroduce the lex-last-wins behavior — the
  /// resolver must still escalate to F32 and the F16 value must still
  /// survive regardless of which prefix iterates last.
  #[test]
  fn transform_awq_weights_preserves_f16_precision_with_bf16_in_reversed_prefix_order() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    // Same F16 fixture value as the forward-order test (= 1 + 2⁻¹⁰).
    let f16_value = half::f16::from_bits(0x3C01);

    let f16_scales_data: Vec<half::f16> = vec![f16_value; n_groups * out_features];
    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();

    // Naming: "alpha" (BF16) sorts BEFORE "zeta" (F16). Under the old
    // lex-last policy this would have picked F16 (zeta last); under
    // the rank-only policy it would pick BF16. Under F5 it MUST be F32.
    let qw_alpha = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_alpha =
      Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_zeta = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_zeta =
      Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();

    let weights: Weights = HashMap::from([
      ("alpha.qweight".to_string(), qw_alpha),
      ("alpha.scales".to_string(), sc_alpha),
      ("zeta.qweight".to_string(), qw_zeta),
      ("zeta.scales".to_string(), sc_zeta),
    ]);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: false,
      version: "gemm".into(),
    };

    let (out, _) = transform_awq_weights(weights, &cfg).expect("transform must succeed");

    let mut sc_zeta_out = out
      .get("zeta.scales")
      .expect("converted zeta.scales present")
      .try_clone()
      .unwrap();
    assert_eq!(
      sc_zeta_out.dtype().unwrap(),
      Dtype::F32,
      "unified dtype must be F32 regardless of prefix order"
    );
    let vals: Vec<f32> = sc_zeta_out.to_vec().expect("read back as F32");
    for (i, &v) in vals.iter().enumerate() {
      assert_eq!(
        v,
        1.0 + (2.0_f32).powi(-10),
        "zeta.scales[{i}] = {v} — F16 precision must be preserved in reversed-order layout"
      );
    }
  }

  // ──────────────── Fix 4 [HIGH]: collision with stale `.weight`/`.biases` ────────────────

  /// Fix 4: input carries `<prefix>.qweight + .scales + .qzeros + .weight` —
  /// a stale dense `.weight` next to a valid AWQ triple. The converter would
  /// emit `<prefix>.weight` from the AWQ conversion, then the remainder pass
  /// would OVERWRITE it with the stale input. Preflight collision check
  /// must REJECT this with a clear message naming the prefix + "collision".
  #[test]
  fn transform_awq_weights_rejects_collision_with_stale_weight() {
    let mut weights = awq_gemm_fixture_weights();
    // Add a stale dense `.weight` next to the AWQ triple. The exact shape
    // doesn't matter — the collision check fires at preflight, before any
    // shape validation.
    let stale = Array::from_slice::<f32>(&[0.0_f32; 16], &(2usize, 8)).unwrap();
    weights.insert("layer.weight".to_string(), stale);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: "gemm".into(),
    };
    let err = transform_awq_weights(weights, &cfg).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("layer.qweight"),
          "error must name the offending qweight, got: {message}"
        );
        assert!(
          message.contains("collision"),
          "error must call out 'collision', got: {message}"
        );
        assert!(
          message.contains("layer.weight"),
          "error must name the colliding stale .weight, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 4: same collision, but with `<prefix>.biases` instead of `.weight`.
  #[test]
  fn transform_awq_weights_rejects_collision_with_stale_biases() {
    let mut weights = awq_gemm_fixture_weights();
    let stale = Array::from_slice::<f32>(&[0.0_f32; 8], &(8usize,)).unwrap();
    weights.insert("layer.biases".to_string(), stale);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: "gemm".into(),
    };
    let err = transform_awq_weights(weights, &cfg).unwrap_err();
    match err {
      Error::Backend { message } => {
        assert!(
          message.contains("layer.qweight"),
          "error must name the offending qweight, got: {message}"
        );
        assert!(
          message.contains("collision"),
          "error must call out 'collision', got: {message}"
        );
        assert!(
          message.contains("layer.biases"),
          "error must name the colliding stale .biases, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  /// Fix 4: an UNRELATED `.weight` key (different prefix) must NOT trigger
  /// the collision check — the conversion proceeds and the unrelated dense
  /// key passes through verbatim.
  #[test]
  fn transform_awq_weights_accepts_unrelated_weight_keys() {
    let mut weights = awq_gemm_fixture_weights();
    // Distinct prefix — embed_tokens.weight is the canonical pass-through.
    let pt = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap();
    weights.insert("embed_tokens.weight".to_string(), pt);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: "gemm".into(),
    };
    let (out, _) = transform_awq_weights(weights, &cfg).expect("unrelated .weight must pass");
    // AWQ output present + pass-through preserved.
    assert!(
      out.contains_key("layer.weight"),
      "AWQ-converted .weight must be present"
    );
    assert!(
      out.contains_key("embed_tokens.weight"),
      "unrelated .weight must be preserved"
    );
  }

  /// Fix 4: BOTH stale `.weight` + `.biases` present → still errors (the
  /// first detected one is fine; this confirms the second one wouldn't
  /// somehow be quiet either, by removing the first and re-running).
  #[test]
  fn transform_awq_weights_rejects_collision_with_both_stale_keys() {
    let mut weights = awq_gemm_fixture_weights();
    let stale_w = Array::from_slice::<f32>(&[0.0_f32; 16], &(2usize, 8)).unwrap();
    let stale_b = Array::from_slice::<f32>(&[0.0_f32; 8], &(8usize,)).unwrap();
    weights.insert("layer.weight".to_string(), stale_w);
    weights.insert("layer.biases".to_string(), stale_b);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: true,
      version: "gemm".into(),
    };
    let err = transform_awq_weights(weights, &cfg).unwrap_err();
    assert!(
      matches!(err, Error::Backend { .. }),
      "must reject with Backend, got: {err:?}"
    );

    // Now drop the .weight collision; the .biases collision alone must
    // still fire. (Confirms the gate is per-sibling, not "first-only".)
    let mut weights2 = awq_gemm_fixture_weights();
    let stale_b2 = Array::from_slice::<f32>(&[0.0_f32; 8], &(8usize,)).unwrap();
    weights2.insert("layer.biases".to_string(), stale_b2);
    let err2 = transform_awq_weights(weights2, &cfg).unwrap_err();
    match err2 {
      Error::Backend { message } => {
        assert!(
          message.contains("layer.biases"),
          "must name the .biases collision when .weight is absent, got: {message}"
        );
      }
      other => panic!("expected Error::Backend, got: {other:?}"),
    }
  }

  // ──────────────── F5 R3 [MEDIUM]: scoped unification ────────────────

  /// F5 R3: a BF16 pass-through tensor (e.g. `embed_tokens.weight`) sitting
  /// next to a single AWQ-quantized layer with BF16 `.scales` must keep its
  /// ORIGINAL dtype (BF16) after `transform_awq_weights`. The unification
  /// cast applies only to the AWQ-generated `.scales` / `.biases`, not to
  /// pass-through floating tensors. (Pre-fix, the unification loop walked
  /// every floating key in the output map, so for a checkpoint whose
  /// pass-through tensors were already at the resolved dtype this was a
  /// no-op — but the *bytes-equivalence* contract is what we want: the
  /// pass-through value is not touched at all.)
  #[test]
  fn transform_awq_weights_does_not_widen_passthrough_bf16_tensor() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    // BF16 scales — resolves to BF16 (single dtype, no escalation).
    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();
    let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

    // Large-ish BF16 pass-through tensor (`embed_tokens.weight`). The
    // exact value pattern doesn't matter — we just need it READABLE so the
    // post-transform comparison can confirm byte-equivalence.
    let pt_shape = (100usize, 100usize);
    let pt_data: Vec<half::bf16> = (0..pt_shape.0 * pt_shape.1)
      .map(|i| half::bf16::from_f32(0.001 * (i as f32 % 1000.0)))
      .collect();
    let pt = Array::from_slice::<half::bf16>(&pt_data, &pt_shape).unwrap();

    let weights: Weights = HashMap::from([
      ("layer.qweight".to_string(), qw),
      ("layer.scales".to_string(), sc),
      ("embed_tokens.weight".to_string(), pt),
    ]);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: false,
      version: "gemm".into(),
    };

    let (out, _) = transform_awq_weights(weights, &cfg).expect("transform");

    // Generated `.scales` is BF16 (resolved model_dtype, no escalation).
    let sc_out = out.get("layer.scales").expect("layer.scales generated");
    assert_eq!(
      sc_out.dtype().unwrap(),
      Dtype::BF16,
      "BF16-only AWQ scales must resolve to BF16, got {:?}",
      sc_out.dtype().unwrap()
    );

    // Pass-through embed_tokens.weight keeps BF16 dtype + shape.
    let mut pt_out = out
      .get("embed_tokens.weight")
      .expect("pass-through embed_tokens.weight preserved")
      .try_clone()
      .unwrap();
    assert_eq!(
      pt_out.dtype().unwrap(),
      Dtype::BF16,
      "pass-through BF16 tensor must NOT be widened by unification"
    );
    assert_eq!(
      pt_out.shape(),
      vec![pt_shape.0, pt_shape.1],
      "pass-through shape preserved"
    );
    // Byte-equivalence: every value identical to the source.
    let pt_back: Vec<half::bf16> = pt_out.to_vec().expect("read pass-through as BF16");
    assert_eq!(
      pt_back.len(),
      pt_data.len(),
      "pass-through element count preserved"
    );
    for (i, (&got, &want)) in pt_back.iter().zip(pt_data.iter()).enumerate() {
      assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "pass-through value at index {i} must be byte-identical (got 0x{:04X}, want 0x{:04X})",
        got.to_bits(),
        want.to_bits()
      );
    }
  }

  /// F5 R3: a F16 pass-through `lm_head.weight` next to TWO AWQ layers
  /// (one F16 scales, one BF16 scales — triggers the F5 [HIGH] F32
  /// escalation per `resolve_awq_model_dtype`) must STILL be F16 after
  /// `transform_awq_weights`. The escalation only applies to the
  /// AWQ-generated `.scales` / `.biases`, NOT to pass-through tensors.
  /// Pre-fix this would have cast `lm_head.weight` from F16 to F32,
  /// doubling its resident size + adding a full-size cast allocation.
  #[test]
  fn transform_awq_weights_does_not_widen_passthrough_f16_tensor_when_mixed_with_bf16_awq_scales() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
      .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
      .collect();
    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();

    let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_b =
      Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

    // F16 pass-through `lm_head.weight`.
    let lm_head_shape = (32usize, 16usize);
    let lm_head_data: Vec<half::f16> = (0..lm_head_shape.0 * lm_head_shape.1)
      .map(|i| half::f16::from_f32(0.01 * (i as f32 % 100.0)))
      .collect();
    let lm_head = Array::from_slice::<half::f16>(&lm_head_data, &lm_head_shape).unwrap();

    let weights: Weights = HashMap::from([
      ("layer_a.qweight".to_string(), qw_a),
      ("layer_a.scales".to_string(), sc_a),
      ("layer_b.qweight".to_string(), qw_b),
      ("layer_b.scales".to_string(), sc_b),
      ("lm_head.weight".to_string(), lm_head),
    ]);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: false,
      version: "gemm".into(),
    };

    let (out, _) = transform_awq_weights(weights, &cfg).expect("transform");

    // AWQ-generated outputs ESCALATED to F32 (per F5 [HIGH]).
    let sc_a_out = out.get("layer_a.scales").expect("layer_a.scales");
    assert_eq!(
      sc_a_out.dtype().unwrap(),
      Dtype::F32,
      "AWQ-generated layer_a.scales must be cast to F32 under mixed-half escalation"
    );
    let sc_b_out = out.get("layer_b.scales").expect("layer_b.scales");
    assert_eq!(
      sc_b_out.dtype().unwrap(),
      Dtype::F32,
      "AWQ-generated layer_b.scales must be cast to F32 under mixed-half escalation"
    );
    let bi_a_out = out.get("layer_a.biases").expect("layer_a.biases");
    assert_eq!(
      bi_a_out.dtype().unwrap(),
      Dtype::F32,
      "AWQ-generated layer_a.biases must be cast to F32 under mixed-half escalation"
    );
    let bi_b_out = out.get("layer_b.biases").expect("layer_b.biases");
    assert_eq!(
      bi_b_out.dtype().unwrap(),
      Dtype::F32,
      "AWQ-generated layer_b.biases must be cast to F32 under mixed-half escalation"
    );

    // BUT pass-through lm_head.weight is STILL F16 (NOT cast to F32).
    let mut lm_head_out = out
      .get("lm_head.weight")
      .expect("pass-through lm_head.weight preserved")
      .try_clone()
      .unwrap();
    assert_eq!(
      lm_head_out.dtype().unwrap(),
      Dtype::F16,
      "pass-through F16 tensor must NOT be widened to F32 by the AWQ \
       mixed-half escalation — only the AWQ-generated .scales/.biases get widened"
    );
    assert_eq!(
      lm_head_out.shape(),
      vec![lm_head_shape.0, lm_head_shape.1],
      "pass-through lm_head shape preserved"
    );
    // Byte-equivalence: every F16 value identical to the source.
    let lm_back: Vec<half::f16> = lm_head_out.to_vec().expect("read lm_head as F16");
    for (i, (&got, &want)) in lm_back.iter().zip(lm_head_data.iter()).enumerate() {
      assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "lm_head.weight[{i}] must be byte-identical (got 0x{:04X}, want 0x{:04X})",
        got.to_bits(),
        want.to_bits()
      );
    }
  }

  /// F5 R3: explicit contrast — with 1 AWQ layer (BF16 scales, no
  /// escalation: resolves to BF16) + 1 F16 pass-through key, the
  /// AWQ-generated `.scales` IS cast (to the resolved BF16) but the
  /// pass-through F16 tensor is left at F16. Confirms the cast is
  /// **scoped** to AWQ-generated keys and not blanket-applied to every
  /// floating output.
  #[test]
  fn transform_awq_weights_widens_only_generated_scales_and_biases() {
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();
    let qw = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc = Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

    // F16 pass-through.
    let pt_shape = (16usize, 8usize);
    let pt_data: Vec<half::f16> = (0..pt_shape.0 * pt_shape.1)
      .map(|i| half::f16::from_f32(0.01 * (i as f32)))
      .collect();
    let pt = Array::from_slice::<half::f16>(&pt_data, &pt_shape).unwrap();

    let weights: Weights = HashMap::from([
      ("layer.qweight".to_string(), qw),
      ("layer.scales".to_string(), sc),
      ("model.norm.weight".to_string(), pt),
    ]);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: false,
      version: "gemm".into(),
    };

    let (out, _) = transform_awq_weights(weights, &cfg).expect("transform");

    // AWQ-generated .scales: BF16 (resolved model_dtype, no escalation).
    let sc_out = out.get("layer.scales").expect("layer.scales");
    assert_eq!(
      sc_out.dtype().unwrap(),
      Dtype::BF16,
      "AWQ-generated .scales is at the resolved BF16 model_dtype"
    );
    let bi_out = out.get("layer.biases").expect("layer.biases");
    assert_eq!(
      bi_out.dtype().unwrap(),
      Dtype::BF16,
      "AWQ-generated .biases is at the resolved BF16 model_dtype"
    );

    // Pass-through F16: still F16, NOT widened to BF16.
    let mut pt_out = out
      .get("model.norm.weight")
      .expect("pass-through model.norm.weight")
      .try_clone()
      .unwrap();
    assert_eq!(
      pt_out.dtype().unwrap(),
      Dtype::F16,
      "pass-through F16 tensor must NOT be cast to the resolved BF16 — \
       unification is scoped to AWQ-generated outputs only"
    );
    // Byte-equivalence.
    let pt_back: Vec<half::f16> = pt_out.to_vec().expect("read pass-through as F16");
    for (i, (&got, &want)) in pt_back.iter().zip(pt_data.iter()).enumerate() {
      assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "pass-through value at index {i} must be byte-identical"
      );
    }
  }

  /// F5 R3: resident-size proxy — a large-ish pass-through tensor next to
  /// a single AWQ layer triggering F32 escalation (via a mixed F16+BF16
  /// pair). The pass-through `Array::size()` × `dtype_size()` must be
  /// IDENTICAL pre- vs post-transform (same shape, same dtype → identical
  /// resident bytes). Pre-fix, the pass-through would have been cast from
  /// BF16 → F32, doubling its resident size.
  #[test]
  fn transform_awq_weights_preserves_resident_size_for_passthrough() {
    fn dtype_size(d: Dtype) -> usize {
      match d {
        Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
        Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::U64 | Dtype::I64 | Dtype::F64 | Dtype::Complex64 => 8,
      }
    }
    let in_features = 8usize;
    let out_features = 8usize;
    let n_groups = 2usize;

    // Mixed F16+BF16 AWQ pair → resolver escalates to F32 (per F5 [HIGH]).
    let f16_scales_data: Vec<half::f16> = (0..n_groups * out_features)
      .map(|i| half::f16::from_f32(0.1 * (i + 1) as f32))
      .collect();
    let bf16_scales_data: Vec<half::bf16> = (0..n_groups * out_features)
      .map(|i| half::bf16::from_f32(0.5 + 0.01 * (i as f32)))
      .collect();
    let qw_a = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_a = Array::from_slice::<half::f16>(&f16_scales_data, &(n_groups, out_features)).unwrap();
    let qw_b = Array::from_slice::<u32>(&vec![0u32; in_features], &(in_features, 1)).unwrap();
    let sc_b =
      Array::from_slice::<half::bf16>(&bf16_scales_data, &(n_groups, out_features)).unwrap();

    // Large-ish BF16 pass-through. Record pre-transform resident size.
    let pt_shape = (256usize, 256usize);
    let pt_data: Vec<half::bf16> = (0..pt_shape.0 * pt_shape.1)
      .map(|i| half::bf16::from_f32((i as f32) * 1e-4))
      .collect();
    let pt = Array::from_slice::<half::bf16>(&pt_data, &pt_shape).unwrap();
    let pt_size_pre = pt.size() * dtype_size(pt.dtype().unwrap());
    assert_eq!(
      pt_size_pre,
      pt_shape.0 * pt_shape.1 * 2,
      "pre-transform BF16 pass-through resident size sanity"
    );

    let weights: Weights = HashMap::from([
      ("layer_a.qweight".to_string(), qw_a),
      ("layer_a.scales".to_string(), sc_a),
      ("layer_b.qweight".to_string(), qw_b),
      ("layer_b.scales".to_string(), sc_b),
      ("embed_tokens.weight".to_string(), pt),
    ]);
    let cfg = AwqLoadConfig {
      bits: 4,
      group_size: 4,
      zero_point: false,
      version: "gemm".into(),
    };

    let (out, _) = transform_awq_weights(weights, &cfg).expect("transform");

    let pt_out = out
      .get("embed_tokens.weight")
      .expect("pass-through preserved");
    // Same dtype + same shape ⇒ same resident size. (BF16 = 2 bytes;
    // had it been cast to F32 the size would have doubled to 4 bytes/elem.)
    assert_eq!(
      pt_out.dtype().unwrap(),
      Dtype::BF16,
      "pass-through must remain BF16 (not widened to F32 by the mixed-half escalation)"
    );
    assert_eq!(
      pt_out.shape(),
      vec![pt_shape.0, pt_shape.1],
      "pass-through shape preserved"
    );
    let pt_size_post = pt_out.size() * dtype_size(pt_out.dtype().unwrap());
    assert_eq!(
      pt_size_post,
      pt_size_pre,
      "pass-through resident size must be IDENTICAL post-transform \
       (pre-fix this would have doubled from {pt_size_pre} to {} bytes)",
      pt_size_pre * 2
    );
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
