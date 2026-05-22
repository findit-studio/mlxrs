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
//! (`utils.py:823-835`), translated from the module tree to the weight map:
//!
//! 1. The key ends in `.weight` — mlx's `Linear` / `Embedding` /
//!    `SwitchLinear` all store the dense matrix as the module's `weight`
//!    parameter, so the weight-map suffix `.weight` is the analogue of
//!    `hasattr(module, "to_quantized")`. The layer **path** is the key with
//!    the `.weight` suffix stripped (mlx-lm passes
//!    `path = "model.layers.0.self_attn.q_proj"` to the predicate; the
//!    weight lives at `"model.layers.0.self_attn.q_proj.weight"`).
//! 2. The array has rank ≥ 2 (mlx-lm `module.weight.shape[-1]` indexes the
//!    last axis; a scalar or 1-D bias is not quantizable).
//! 3. The last axis is divisible by `group_size` (mlx-lm `if
//!    module.weight.shape[-1] % group_size != 0: return False`,
//!    `utils.py:826-827`).
//! 4. The per-layer override (if any) is consulted (mlx-lm
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

use std::collections::HashMap;

use serde::{Deserialize, Deserializer};

use crate::{
  array::Array,
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

    // 1) Parse the global `Quantization` from the same level. The block is
    //    only a *global* quantization if it carries `group_size` + `bits`
    //    (mlx-lm `_quantize` reads them unconditionally; swift's
    //    `Quantization(from: decoder)` requires the keys — and `bits` /
    //    `group_size` are not `Optional` in the swift struct). A bare
    //    `false` for a layer with no global keys is allowed: the per-layer
    //    map is populated below and `quantization` stays `None`.
    let quantization = if map.contains_key("group_size") && map.contains_key("bits") {
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
      Some(
        serde_json::from_value::<Quantization>(Value::Object(globals)).map_err(D::Error::custom)?,
      )
    } else {
      None
    };

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

/// Whether `weights` already carries a quantized triple for `layer_path` —
/// `<layer_path>.scales` is present. mlx-lm `class_predicate`
/// (`utils.py:349-355`) gates on `f"{p}.scales" in weights` as the signal
/// that the checkpoint already pre-quantized this layer; an
/// already-quantized weight is left untouched (no double-quantize).
fn is_already_quantized(weights: &Weights, layer_path: &str) -> bool {
  let mut key = String::with_capacity(layer_path.len() + ".scales".len());
  key.push_str(layer_path);
  key.push_str(".scales");
  weights.contains_key(&key)
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
/// **Failure handling.** Every quantization op is fallible
/// ([`crate::ops::quantized::quantize`] propagates mlx-c's error); a
/// failure mid-walk drops the partially-built result map and returns
/// `Err` — the input `weights` is consumed but no partial output escapes.
pub fn quantize_weights(weights: Weights, cfg: &PerLayerQuantization) -> Result<Weights> {
  // Out-map sized for "at most everything got quantized" (adds up to one
  // `.scales` + one `.biases` per `.weight` quantized, i.e. ≤ 3× the input
  // — a conservative upper bound).
  let mut out: Weights = HashMap::with_capacity(weights.len());

  // Two passes so the predicate sees the COMPLETE input map for the
  // "already quantized" check (`<path>.scales` in weights). Pass 1 chooses
  // which keys to quantize without mutating; pass 2 does the work. mlx-lm's
  // `tree_map_with_path` on `leaf_modules()` is the module-tree analog of
  // this two-pass shape (a sibling `.scales` check needs the full map up
  // front).
  let mut to_quantize: Vec<(String, Quantization)> = Vec::new();
  for (key, arr) in &weights {
    let Some(layer_path) = key.strip_suffix(WEIGHT_SUFFIX) else {
      continue;
    };
    // mlx-lm `utils.py:349-355`: skip when the checkpoint already shipped
    // a `<path>.scales` for this layer (already-quantized).
    if is_already_quantized(&weights, layer_path) {
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
  // verbatim. The iteration order doesn't matter because the output keys
  // are unique (`<path>.weight` / `<path>.scales` / `<path>.biases` are
  // disjoint with anything else in the input — a hostile checkpoint that
  // ships a stray `<path>.biases` for a not-already-quantized layer is the
  // ONLY way to clash, and the `is_already_quantized` gate covers it).
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
      out.insert(format!("{path}.weight"), w_q);
      out.insert(format!("{path}.scales"), scales);
      if let Some(b) = biases {
        out.insert(format!("{path}.biases"), b);
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
/// Non-triple entries (no `.scales` sibling) pass through verbatim — a
/// `.weight` without a matching `.scales` is an already-dense weight, and
/// stray `.scales` / `.biases` without a `.weight` are passed through too
/// (a hostile / corrupt checkpoint shape; mlx-lm leaves them in place too —
/// `dequantize_model` only replaces *modules*, never deletes parameters).
pub fn dequantize_weights(weights: Weights, cfg: &PerLayerQuantization) -> Result<Weights> {
  let mut out: Weights = HashMap::with_capacity(weights.len());

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

#[cfg(test)]
mod tests {
  use super::*;

  fn arr_f32(data: &[f32], shape: &[usize]) -> Array {
    Array::from_slice::<f32>(data, &shape).expect("from_slice")
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
    // Already-quantized layer: `<path>.scales` is present, so its `.weight`
    // is skipped (per mlx-lm `utils.py:349-355`).
    let already_w = arr_f32(&vec![0.0_f32; n_rows * group_size], &[n_rows, group_size]);
    let already_scales = arr_f32(&vec![0.0_f32; n_rows], &[n_rows]);
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
    weights.insert("model.layers.0.q_proj.bias".to_string(), bias);
    weights.insert("model.layers.2.bad.weight".to_string(), odd_last);
    weights.insert("model.norm.gamma".to_string(), other);
    let cfg = PerLayerQuantization::from_global(Quantization::affine(group_size as i32, 4));

    let out = quantize_weights(weights, &cfg).expect("quantize");

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

    // Skipped: already-quantized layer's triple passes through unchanged.
    let pre_q_w = out.get("model.layers.1.v_proj.weight").expect("already-w");
    assert_eq!(pre_q_w.shape(), vec![n_rows, group_size]);
    assert_eq!(pre_q_w.dtype().unwrap(), crate::dtype::Dtype::F32);
    assert!(out.contains_key("model.layers.1.v_proj.scales"));

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

    let quantized = quantize_weights(weights, &cfg).unwrap();
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

    let out = quantize_weights(weights, &cfg).unwrap();
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

    let out = quantize_weights(weights, &cfg).unwrap();
    // Quantized at group_size 32: scales / biases have one group per row
    // (last / group_size = 32 / 32 = 1).
    let scales = out.get("model.embed_tokens.scales").expect(".scales");
    assert_eq!(scales.shape(), vec![n_rows, 1]);
    let w_q = out.get("model.embed_tokens.weight").expect(".weight");
    // bits=4 packs 8 elements per uint32 → last axis is 32 / 8 = 4.
    assert_eq!(w_q.shape(), vec![n_rows, 4]);
  }
}
