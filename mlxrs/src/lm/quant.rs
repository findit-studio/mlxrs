//! Weight-map (de)quantization + per-layer [`Quantization`] config schema.
//!
//! Port of mlx-lm's `quantize_model` / `dequantize_model` (`mlx_lm/utils.py`)
//! and mlx-swift-lm's `MLXLMCommon.BaseConfiguration.Quantization` /
//! `PerLayerQuantization` (`Libraries/MLXLMCommon/BaseConfiguration.swift`),
//! adapted to mlxrs's per-project scope: mlxrs has no model-module tree
//! (that is per-usecase), so where mlx-lm
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
//! [`Error::Backend`] / [`Error::RankMismatch`] / [`Error::ShapePairMismatch`]
//! with a clear message.
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
//! mlxrs wrappers are thin forwards mirroring mlx-swift/python; we do not
//! chase mlx-core-internal hardening. Per-mode contracts (e.g. `bits ∈ {2,3,4,5,6,8}`
//! for affine — `mlx/mlx/ops.cpp:4745-4750`; `mxfp4` requires `(32, 4)`,
//! `nvfp4` requires `(16, 4)` — `mlx/mlx/ops.cpp:4808-4823`) are upstream of
//! this module.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Deserializer};

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    ArithmeticOverflowPayload, DivisibilityConstraintPayload, Error, InvariantViolationPayload,
    KeyCollisionPayload, LayerKeyedPayload, LengthMismatchPayload, MissingKeyPayload,
    OutOfRangePayload, ParsePayload, RankMismatchPayload, Result, ShapePairMismatchPayload,
    UnknownEnumValuePayload, UnsupportedDtypePayload,
  },
  lm::load::Weights,
  ops,
};

/// The set of MLX quantization modes — mlx-swift's `QuantizationMode`
/// (`mlx-swift/Source/MLX/Ops.swift:1097-1124`), serialized as the lowercase
/// tag string mlx-c expects (`"affine"` / `"mxfp4"` / `"mxfp8"` / `"nvfp4"`).
#[derive(
  Debug,
  Clone,
  Copy,
  PartialEq,
  Eq,
  Hash,
  serde::Deserialize,
  serde::Serialize,
  derive_more::Display,
  derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
#[non_exhaustive]
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
  pub const fn as_str(self) -> &'static str {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::IsVariant)]
#[non_exhaustive]
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
  ///
  /// Private: access via [`per_layer_ref`](Self::per_layer_ref).
  per_layer: HashMap<String, QuantizationOption>,
}

impl PerLayerQuantization {
  /// Build a [`PerLayerQuantization`] from a global [`Quantization`] (or
  /// `None` for skip-by-default) and an explicit per-layer override map.
  pub fn new(
    quantization: Option<Quantization>,
    per_layer: HashMap<String, QuantizationOption>,
  ) -> Self {
    Self {
      quantization,
      per_layer,
    }
  }

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

  /// The per-layer override map (path → [`QuantizationOption`]).
  ///
  /// Empty when the on-disk JSON only carried the global
  /// `{ group_size, bits, [mode] }` (no per-layer keys — the common case).
  #[inline(always)]
  pub fn per_layer_ref(&self) -> &HashMap<String, QuantizationOption> {
    &self.per_layer
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
  let value: serde_json::Value = serde_json::from_str(config_json)
    .map_err(|e| Error::Parse(ParsePayload::new("parse_quantization", "config JSON", e)))?;
  let Some(block) = value.get("quantization") else {
    return Ok(None);
  };
  let plq: PerLayerQuantization = serde_json::from_value(block.clone()).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "parse_quantization",
      "`quantization` block",
      e,
    ))
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
/// permissive default behavior. Pass this to [`quantize_weights`]
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
  /// / out-of-sync checkpoint. The wrapped typed [`Error`] carries the
  /// structured diagnostic; the caller surfaces it directly.
  Invalid(Error),
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
/// `cfg.quantization` (the common case — the deserializer enforces that any
/// parsed `"quantization"` block contains `group_size` + `bits`) or via a
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
    (None, Some(_)) => TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
      layer_path.to_string(),
      Error::MissingKey(MissingKeyPayload::new(
        "quantize_weights: stale `.biases` with no matching `.scales` \
          (mlx `quantize` always writes `.scales` alongside `.biases`; refusing to \
          silently overwrite the generated bias)",
        scales_key.clone(),
      )),
    ))),
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
          return TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
            layer_path.to_string(),
            Error::KeyCollision(KeyCollisionPayload::new(
              "quantize_weights: input carries `.scales` but the per-layer config marks this \
                layer as `Skip` (not quantized) — refusing to silently treat the stale triple \
                as a valid already-quantized layer",
              scales_key.clone(),
            )),
          )));
        }
        Some(QuantizationOption::Quantize(q)) => *q,
        None => match cfg.quantization {
          Some(q) => q,
          None => {
            return TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
              layer_path.to_string(),
              Error::InvariantViolation(InvariantViolationPayload::new(
                "quantize_weights: input carries `.scales` but `cfg` has no global \
                  `Quantization` and no per-layer override for this layer — cannot resolve \
                  expected `.scales` shape (defensive: any parsed `quantization` block \
                  carries group_size + bits)",
                "quantization parameters must be resolvable",
              )),
            )));
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
          return TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
            layer_path.to_string(),
            Error::MissingKey(MissingKeyPayload::new(
              "quantize_weights: `affine` mode requires `.biases` alongside `.scales` \
                (mlx `affine_quantize` always writes `{w_q, scales, biases}`, \
                mlx/ops.cpp:4793-4798); this is a structurally incomplete affine triple",
              biases_key.clone(),
            )),
          )));
        }
        (QuantMode::Mxfp4 | QuantMode::Mxfp8 | QuantMode::Nvfp4, Some(_)) => {
          return TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
            layer_path.to_string(),
            Error::KeyCollision(KeyCollisionPayload::new(
              "quantize_weights: scale-only mode (mxfp4 / mxfp8 / nvfp4) must not carry \
                `.biases` (mlx `fp_quantize` writes `{w_q, scales}` with no biases, \
                mlx/ops.cpp:4890,4898-4904); refusing to silently retain a bias from a \
                different (affine) mode",
              biases_key.clone(),
            )),
          )));
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
          return TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
            layer_path.to_string(),
            e,
          )));
        }
      };
      if w_dtype != Dtype::U32 {
        return TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
          layer_path.to_string(),
          Error::UnsupportedDtype(UnsupportedDtypePayload::new(
            "quantize_weights: input has `.scales` but `.weight` dtype must be uint32 \
              (mlx-quantized `.weight` is always uint32 — mlx/ops.cpp:4795,4900); this is \
              a stale `.scales` orphan next to a dense `.weight`, not a valid \
              already-quantized triple",
            w_dtype,
            &[Dtype::U32],
          )),
        )));
      }
      // `.weight` rank must be ≥ 2 — mlx `quantize` requires rank ≥ 2
      // inputs (`mlx/ops.cpp:4925-4929`), so a rank-0/1 `.weight` next
      // to a `.scales` cannot be a real quantized triple even when the
      // dtype is `uint32` and the leading dims happen to match.
      let w_shape = layer_weight.shape();
      let w_rank = w_shape.len();
      if w_rank < 2 {
        return TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
          layer_path.to_string(),
          Error::RankMismatch(RankMismatchPayload::new(
            "quantize_weights: `.weight` next to `.scales` must be rank-2 (mlx `quantize` \
              requires rank >= 2 inputs — mlx/ops.cpp:4925-4929; a uint32 1-D / scalar \
              `.weight` next to a `.scales` is not a layout mlx's `quantize` can have produced)",
            w_rank as u32,
            w_shape,
          )),
        )));
      }
      // `.scales` rank == `.weight` rank, and the leading dims (all
      // but the last) match — mlx `quantize` preserves the leading
      // shape (`mlx/ops.cpp:4789-4798`).
      let s_shape = s.shape();
      if s_shape.len() != w_shape.len() {
        return TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
          layer_path.to_string(),
          Error::LengthMismatch(LengthMismatchPayload::new(
            "quantize_weights: `.scales` rank vs `.weight` rank — mlx `quantize` preserves \
              the leading shape across the packed `.weight` / `.scales` / `.biases` outputs \
              (mlx/ops.cpp:4789-4798)",
            w_shape.len(),
            s_shape.len(),
          )),
        )));
      }
      // Leading dims (all but the last) must match. Rank ≥ 2 is
      // already enforced above, so both slices are non-empty and the
      // index is safe. This is the structural shape mlx `quantize`
      // preserves and `mlx-c`'s `validate_quantized_input` enforces
      // (`mlx/mlx/ops.cpp:97-105`); checking it here surfaces the
      // mismatch with a layer-named error before mlx-c sees it.
      if s_shape[..s_shape.len() - 1] != w_shape[..w_shape.len() - 1] {
        return TripleClass::Invalid(Error::LayerKeyed(LayerKeyedPayload::new(
          layer_path.to_string(),
          Error::ShapePairMismatch(ShapePairMismatchPayload::new(
            "quantize_weights: `.scales` leading dims (all but the last) must match \
              `.weight` leading dims — mlx `quantize` preserves all-but-last dims",
            w_shape[..w_shape.len() - 1].to_vec(),
            s_shape[..s_shape.len() - 1].to_vec(),
          )),
        )));
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
      TripleClass::Invalid(err) => return Err(err),
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
    let gs = usize::try_from(q.group_size).map_err(|_| {
      Error::LayerKeyed(LayerKeyedPayload::new(
        layer_path.to_string(),
        Error::OutOfRange(OutOfRangePayload::new(
          "quantize_weights: group_size",
          "must be a non-negative i32",
          q.group_size.to_string(),
        )),
      ))
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
        ops::quantized::quantize(&arr, q.group_size, q.bits, q.mode.as_str(), None)?;
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
      let w_dtype = weight_arr
        .dtype()
        .map_err(|e| Error::LayerKeyed(LayerKeyedPayload::new(path.to_string(), e)))?;
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
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        path.to_string(),
        Error::MissingKey(MissingKeyPayload::new(
          "dequantize_weights: stale `.biases` with no matching `.scales` \
            (mlx `quantize` always writes `.scales` alongside `.biases`, \
            `mlx/ops.cpp:4793-4798`); this is a structurally incomplete triple, \
            refusing to silently leave the `uint32`-packed `.weight` as a \
            pass-through in the dequantized output",
          format!("{path}{SCALES_SUFFIX}"),
        )),
      )));
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
    let w = w_opt.ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "dequantize_weights: triple missing `.weight`",
        format!("{path}{WEIGHT_SUFFIX}"),
      ))
    })?;
    let scales = s_opt.ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "dequantize_weights: triple missing `.scales`",
        format!("{path}{SCALES_SUFFIX}"),
      ))
    })?;
    let q = cfg.quantization_for(&path).ok_or_else(|| {
      Error::LayerKeyed(LayerKeyedPayload::new(
        path.to_string(),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "dequantize_weights: quantization parameters",
          "must be resolvable (no global default and no per-layer override)",
        )),
      ))
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
        return Err(Error::LayerKeyed(LayerKeyedPayload::new(
          path.to_string(),
          Error::MissingKey(MissingKeyPayload::new(
            "dequantize_weights: `affine` mode requires `.biases` alongside `.scales` \
              (mlx `affine_dequantize` takes `{w_q, scales, biases}`, mlx/ops.cpp:5085-5099)",
            format!("{path}.biases"),
          )),
        )));
      }
      (QuantMode::Mxfp4 | QuantMode::Mxfp8 | QuantMode::Nvfp4, Some(_)) => {
        return Err(Error::LayerKeyed(LayerKeyedPayload::new(
          path.to_string(),
          Error::KeyCollision(KeyCollisionPayload::new(
            "dequantize_weights: scale-only mode (mxfp4 / mxfp8 / nvfp4) must not carry \
              `.biases` (mlx `fp_dequantize` takes `{w_q, scales}` with no biases, \
              mlx/ops.cpp:5198-5210); refusing to silently retain a bias from a \
              different (affine) mode",
            format!("{path}.biases"),
          )),
        )));
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
      q.mode.as_str(),
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
//      mlxrs ports loaders/tokenizers/pooling — not per-usecase
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
/// [`Error::RankMismatch`] (the python version would `ValueError`
/// during the trailing `.reshape(out_features, in_features)`). Dtypes
/// other than `uint32` / `int32` are rejected as [`Error::Backend`].
pub fn unpack_awq_weights(qweight: &Array) -> Result<Array> {
  let shape = qweight.shape();
  let shape_len = shape.len();
  if shape_len != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "unpack_awq_weights: qweight must be 2-D [rows, packed_cols]",
      shape_len as u32,
      shape,
    )));
  }
  let shape = qweight.shape();
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
      return Err(Error::UnsupportedDtype(UnsupportedDtypePayload::new(
        "unpack_awq_weights: qweight (AutoAWQ stores qweight as 32-bit packed nibbles, \
          utils.py:72-82; accepts uint32 (mlx-lm canonical) or int32 (AutoAWQ WQLinear_GEMM's \
          default torch.int32 allocation))",
        other,
        &[Dtype::U32, Dtype::I32],
      )));
    }
  };
  let rows = shape[0];
  let packed_cols = shape[1];
  let cols = packed_cols.checked_mul(AWQ_PACK_FACTOR).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "unpack_awq_weights: unpacked col count `packed_cols * 8`",
      "usize",
      [
        ("packed_cols", packed_cols as u64),
        ("multiplier", AWQ_PACK_FACTOR as u64),
      ],
    ))
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
/// mlx-lm takes the LAST iterated layer's `scales.dtype`
/// as the target (`utils.py:156` overwrites `model_dtype` each iteration).
/// In Python that's `dict` insertion order, which for AutoAWQ checkpoints
/// usually means the last weight in the safetensors file. Picking the
/// LEX-LAST prefix would give a stable choice across HashMap iteration
/// orders, but for HETEROGENEOUS-PRECISION checkpoints
/// (e.g. some layers f16, some bf16) the lex-last pick is arbitrary and
/// would silently downcast the higher-precision layers — so mlxrs
/// resolves by precision instead.
///
/// Resolution policy: **highest precision wins** in HIERARCHICAL cases —
/// `F64 > F32 > BF16 / F16` (a wider format is a superset, so the cast
/// is lossless from the lower formats up). For ties at the same rank
/// (e.g. all bf16) the result is the first dtype with that rank — stable
/// across runs.
///
/// **F16 + BF16 mixed escalation**: F16 and BF16 are NOT
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
/// **Scope**: the dtype this fn resolves applies to the
/// AWQ-generated `.scales` / `.biases` outputs ONLY, **not** to the
/// pass-through floating tensors in the same checkpoint (embeddings, LM
/// head, norms, etc.). Running the unification cast over every floating
/// key in `new_weights` would, for a large quantized model with BF16/F16
/// embeddings + one mixed-half AWQ pair, DOUBLE the resident size of
/// those pass-through tensors and add a full-size cast allocation during
/// load — capable of turning a fitting model into OOM.
/// [`transform_awq_weights`] therefore iterates the unification loop over
/// a `BTreeSet<String>` of generated keys only; pass-through tensors
/// retain their on-disk dtype. See
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
    let scales = weights.get(&scales_key).ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "transform_awq_weights: AWQ `.qweight` missing its `.scales` companion \
          (AutoAWQ writes `.qweight` / `.scales` / `.qzeros` as a triple — refusing to \
          silently drop the layer)",
        scales_key.clone(),
      ))
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
  // Escalation: F16+BF16 mixed without F32/F64 → promote to F32 to
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

/// Precision rank for the floating dtypes that may appear as
/// AWQ `.scales`. Higher rank = more precision.
///
/// Order: `F64 > F32 > BF16 > F16 > anything-else (sentinel 0)`.
///
/// **Caveat**: this rank treats BF16 > F16 because BF16 has
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

/// Validate that every AWQ `.scales` tensor is a SUPPORTED
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
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        scales_key.clone(),
        Error::UnsupportedDtype(UnsupportedDtypePayload::new(
          "transform_awq_weights: AutoAWQ `.scales` (any other dtype would corrupt the \
            dtype-unification cast)",
          d,
          &[Dtype::F16, Dtype::F32, Dtype::F64, Dtype::BF16],
        )),
      )));
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
///    Scope: the cast walks only the keys this
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
  // Reject `version = "gemv"` and any other non-GEMM version
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
    other => {
      return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
        "transform_awq_weights: AWQ version (only `gemm` is implemented — GEMV checkpoints \
          use a different qweight shape, scales layout, and sequential packing; converting one \
          through the GEMM path would silently produce corrupt weights — re-quantize with \
          `awq --version gemm` if possible)",
        other.to_string(),
        &["gemm", ""],
      )));
    }
  }
  // Faithful to mlx-lm's `if bits != 4: raise ValueError` (`utils.py:88`).
  if config.bits != AWQ_BITS {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "transform_awq_weights: AWQ bits (mlx-lm/mlx_lm/utils.py:88-89)",
      "must be 4 (only 4-bit AutoAWQ/GPTQ is supported)",
      config.bits.to_string(),
    )));
  }
  let group_size = config.group_size;
  let group_size_i32 = i32::try_from(group_size).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "transform_awq_weights: group_size",
      "must fit in i32",
      group_size.to_string(),
    ))
  })?;

  // Reject GPTQ `g_idx` upfront — `utils.py:95-100`. mlxrs's port does not
  // implement the non-contiguous-group reorder path; the caller must
  // re-quantize via `mlx_lm.convert` or pick a model without `g_idx`.
  for key in weights.keys() {
    if key.ends_with(".g_idx") {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        key.clone(),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "transform_awq_weights: GPTQ `.g_idx`",
          "must not be present (models with non-contiguous group indices are not supported \
            by mlx-lm's AutoAWQ on-load converter — mlx-lm/mlx_lm/utils.py:95-100 — please use \
            a model without `g_idx` or re-quantize via `mlx_lm.convert`)",
        )),
      )));
    }
  }

  // Collect every prefix of a `.qweight` key (sorted, for deterministic
  // iteration in tests; `resolve_awq_model_dtype` uses a precision rank
  // independent of order — see the dtype-validation step below).
  let mut qweight_prefixes: Vec<String> = weights
    .keys()
    .filter_map(|k| k.strip_suffix(".qweight").map(str::to_string))
    .collect();
  qweight_prefixes.sort();

  // Gate every `.scales` dtype as floating BEFORE resolving
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
    // Collision check with `<prefix>.weight` / `<prefix>.biases`
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
      return Err(Error::KeyCollision(KeyCollisionPayload::new(
        "transform_awq_weights: input contains both `.qweight` and `.weight` (the generated \
          AWQ output would overwrite the stale dense `.weight`); remove the stale dense weight \
          before fusing (precedent: non-AWQ `quantize_weights` refuses analogous orphan/stale \
          collisions via `classify_triple`)",
        weight_key,
      )));
    }
    let biases_key = format!("{prefix}.biases");
    if weights.contains_key(&biases_key) {
      return Err(Error::KeyCollision(KeyCollisionPayload::new(
        "transform_awq_weights: input contains both `.qweight` and `.biases` (the generated \
          AWQ output would overwrite the stale `.biases`); remove the stale biases before \
          fusing (precedent: non-AWQ `quantize_weights` refuses analogous orphan/stale \
          collisions via `classify_triple`)",
        biases_key,
      )));
    }
    let Some(qweight) = weights.get(&qweight_key) else {
      // Should be unreachable (we built the prefix list FROM the keys),
      // but guard defensively.
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "transform_awq_weights: `.qweight` missing after prefix scan (defensive)",
        qweight_key,
      )));
    };
    let Some(scales) = weights.get(&scales_key) else {
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "transform_awq_weights: AWQ `.qweight` missing its `.scales` companion \
          (AutoAWQ writes `.qweight` / `.scales` / `.qzeros` as a triple — refusing to silently \
          drop the layer)",
        scales_key,
      )));
    };
    // Dtype preflight. AutoAWQ's `WQLinear_GEMM`
    // allocates `qweight` / `qzeros` as `torch.int32` (signed); mlx-lm's
    // canonical converter expects `uint32`. Accept BOTH — but reject other
    // dtypes (floats, narrower ints, etc.) here with a clear message, so a
    // hostile/malformed checkpoint cannot slip past to mid-pipeline.
    // `unpack_awq_weights` performs the bit-preserving I32 → U32 view
    // internally; this gate just surfaces the wrong-dtype case as a
    // preflight rather than a per-layer error during the conversion pass.
    let qw_dtype = qweight.dtype()?;
    if !matches!(qw_dtype, Dtype::U32 | Dtype::I32) {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        qweight_key.clone(),
        Error::UnsupportedDtype(UnsupportedDtypePayload::new(
          "transform_awq_weights: qweight (AutoAWQ stores packed nibbles as `uint32` \
            (mlx-lm canonical) or `int32` (AutoAWQ `WQLinear_GEMM` default `torch.int32` \
            allocation))",
          qw_dtype,
          &[Dtype::U32, Dtype::I32],
        )),
      )));
    }
    // Shape validation: `qweight: [in_features, packed_out]` /
    // `scales: [n_groups, out_features]`. The two must agree on
    // out_features (post-pack-factor), and `in_features` must be a
    // multiple of `group_size` (`utils.py:118` `n_groups = in_features //
    // group_size`).
    let q_shape0 = qweight.shape();
    let s_shape0 = scales.shape();
    let q_rank = q_shape0.len();
    let s_rank = s_shape0.len();
    if q_rank != 2 {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        qweight_key.clone(),
        Error::RankMismatch(RankMismatchPayload::new(
          "transform_awq_weights: qweight must be 2-D [in_features, packed_out]",
          q_rank as u32,
          q_shape0,
        )),
      )));
    }
    if s_rank != 2 {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        scales_key.clone(),
        Error::RankMismatch(RankMismatchPayload::new(
          "transform_awq_weights: scales must be 2-D [n_groups, out_features]",
          s_rank as u32,
          s_shape0,
        )),
      )));
    }
    let q_shape = qweight.shape();
    let s_shape = scales.shape();
    let in_features = q_shape[0];
    let packed_out = q_shape[1];
    let out_features = packed_out.checked_mul(AWQ_PACK_FACTOR).ok_or_else(|| {
      Error::LayerKeyed(LayerKeyedPayload::new(
        qweight_key.clone(),
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "transform_awq_weights: out_features = packed_out * AWQ_PACK_FACTOR",
          "usize",
          [
            ("packed_out", packed_out as u64),
            ("multiplier", AWQ_PACK_FACTOR as u64),
          ],
        )),
      ))
    })?;
    if group_size as usize == 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "transform_awq_weights: group_size",
        "must be > 0",
        group_size.to_string(),
      )));
    }
    if in_features % (group_size as usize) != 0 {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        qweight_key.clone(),
        Error::DivisibilityConstraint(DivisibilityConstraintPayload::new(
          "transform_awq_weights: in_features must be a multiple of group_size \
            (utils.py:118: n_groups = in_features // group_size)",
          "in_features",
          in_features as u64,
          "group_size",
          group_size as u64,
        )),
      )));
    }
    let n_groups = in_features / (group_size as usize);
    if s_shape[0] != n_groups || s_shape[1] != out_features {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        prefix.clone(),
        Error::ShapePairMismatch(ShapePairMismatchPayload::new(
          "transform_awq_weights: scales must be [n_groups, out_features] \
            (derived from qweight shape with group_size)",
          vec![n_groups, out_features],
          s_shape,
        )),
      )));
    }
    let qzeros_key = format!("{prefix}.qzeros");
    if let Some(qzeros) = weights.get(&qzeros_key) {
      // Same dtype gate as qweight — accept U32 (mlx-lm canonical) or I32
      // (AutoAWQ `torch.int32`), reject other dtypes here at preflight.
      let qz_dtype = qzeros.dtype()?;
      if !matches!(qz_dtype, Dtype::U32 | Dtype::I32) {
        return Err(Error::LayerKeyed(LayerKeyedPayload::new(
          qzeros_key.clone(),
          Error::UnsupportedDtype(UnsupportedDtypePayload::new(
            "transform_awq_weights: qzeros (accept `uint32` mlx-lm canonical or `int32` \
              AutoAWQ default `torch.int32`)",
            qz_dtype,
            &[Dtype::U32, Dtype::I32],
          )),
        )));
      }
      let z_shape = qzeros.shape();
      if z_shape.len() != 2 || z_shape[0] != n_groups || z_shape[1] != packed_out {
        return Err(Error::LayerKeyed(LayerKeyedPayload::new(
          qzeros_key.clone(),
          Error::ShapePairMismatch(ShapePairMismatchPayload::new(
            "transform_awq_weights: qzeros must be [n_groups, packed_out] \
              (derived from qweight shape with group_size)",
            vec![n_groups, packed_out],
            z_shape,
          )),
        )));
      }
    }
  }

  // Now do the conversion. Move every key out of the input map exactly
  // once — non-AWQ keys flow straight through.
  let mut new_weights: Weights = HashMap::with_capacity(weights.len());
  // Track the AWQ-generated `.scales` / `.biases` keys
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
    let (qw_opt, sc_opt, qz_opt) = awq_components.remove(prefix).ok_or_else(|| {
      Error::LayerKeyed(LayerKeyedPayload::new(
        prefix.clone(),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "transform_awq_weights: AWQ components",
          "must be present (lost mid-pipeline — defensive)",
        )),
      ))
    })?;
    let qweight = qw_opt.ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "transform_awq_weights: `.qweight` disappeared mid-pipeline (defensive)",
        format!("{prefix}.qweight"),
      ))
    })?;
    let scales = sc_opt.ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "transform_awq_weights: `.scales` disappeared mid-pipeline (defensive)",
        format!("{prefix}.scales"),
      ))
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
    // Record the AWQ-generated floating outputs so the
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
  // Floating-dtype unification (`utils.py:163-165`)
  // scoped to AWQ-generated `.scales` / `.biases` ONLY. mlx-lm runs this
  // cast over every floating key in the resulting dict, but doing so in a
  // Rust port (where the input map carries every pass-through tensor —
  // embeddings, LM head, norms, etc.) means a single mixed-half AWQ pair
  // (F16+BF16 → F32 escalation, see `resolve_awq_model_dtype`)
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
    bits: i32::try_from(config.bits).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "transform_awq_weights: bits",
        "must fit in i32",
        config.bits.to_string(),
      ))
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
mod tests;
