//! Mixture-of-Experts (MoE) **Switch** layers: the per-expert routed linear
//! primitives [`SwitchLinear`] / [`QuantizedSwitchLinear`], and the
//! gate-up-down MoE blocks [`SwitchGLU`] / [`SwitchMLP`] composed on top of
//! them.
//!
//! A 1:1 port of mlx-lm's `switch_layers.py` (and mlx-swift's
//! `MLXLMCommon/SwitchLayers.swift`):
//!
//! - [`SwitchLinear`] / [`QuantizedSwitchLinear`] (`switch_layers.py:93` /
//!   `:27`) — each holds a per-expert weight stack of shape `[num_experts,
//!   output_dims, input_dims]` plus an optional per-expert bias of shape
//!   `[num_experts, output_dims]`, and routes each input token through one
//!   (or `k`) experts indexed by a caller-supplied `indices` array. The
//!   forward pass collapses to a single fused mlx-c kernel —
//!   [`ops::linalg_basic::gather_mm`](crate::ops::linalg_basic::gather_mm)
//!   for the dense layer,
//!   [`ops::quantized::gather_qmm`](crate::ops::quantized::gather_qmm) for the
//!   quantized one — instead of `take`+`matmul`.
//! - [`SwitchGLU`] / [`SwitchMLP`] (`switch_layers.py:160` / `:202`) — the
//!   MoE expert blocks. [`SwitchGLU`] is the gated `down(activation(gate(x))
//!   · up(x))` structure (three [`SwitchLinear`]s — `gate_proj` / `up_proj` /
//!   `down_proj`); [`SwitchMLP`] is the plain `fc2(activation(fc1(x)))` two-
//!   projection block. Both `expand_dims` the input to add the `(top-k, M=1)`
//!   matmul axes, and — when routing ≥ 64 index slots — sort tokens by expert
//!   id (`_gather_sort`) so each expert's rows are contiguous for the fused
//!   kernel, then unsort the result (`_scatter_unsort`). Their activations
//!   come from [`crate::lm::nn::activations`].
//!
//! Each block is a struct that holds its sub-layers (and a boxed activation
//! closure) with a `forward(&self, x, indices)` that returns a new lazy
//! [`Array`] — the same config-struct pattern as [`crate::lm::nn::rope::Rope`]
//! / [`crate::lm::nn::norm::RMSNorm`]; eval stays an explicit `&mut` step.
//!
//! # Scope
//!
//! mlx-swift's `FusedGateUpSwitchGLU` (a `SwitchGLU` variant for models
//! shipping a single fused `gate_up_proj` weight — `SwitchLayers.swift`,
//! noted there as "Used by Gemma 4 26B MoE") is a per-model construct with no
//! python-reference analogue, so it is deliberately out of scope here; it
//! belongs with the per-model architecture code that consumes it.
//!
//! mlx-swift does not expose `QuantizedSwitchLinear` construction the same way
//! python does (it derives it from a dense `SwitchLinear` via `toQuantized`);
//! the [`QuantizedSwitchLinear`] port follows the python reference, which is
//! the canonical home of the from-quantized-arrays constructor.

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    ArithmeticOverflowPayload, InvariantViolationPayload, LengthMismatchPayload, OutOfRangePayload,
    RankMismatchPayload, Result, ShapePairMismatchPayload, UnknownEnumValuePayload,
  },
  ops::{arithmetic, indexing, linalg_basic, misc, quantized, shape},
};
use smol_str::format_smolstr;

use super::activations::silu;

/// Per-expert linear layer: holds `num_experts` independent `(output_dims,
/// input_dims)` weight matrices stacked into a single tensor `[E, O, I]`, plus
/// an optional `[E, O]` bias, and routes each input via an `indices` array.
///
/// Mirror of mlx-lm `SwitchLinear` (`mlx-lm/mlx_lm/models/switch_layers.py:93`).
/// The forward pass `apply(x, indices, sorted_indices)` is a single fused
/// `gather_mm` followed by an optional per-expert bias add — identical to the
/// python `__call__`.
///
/// # Shape contract
///
/// - `weight`: `[num_experts, output_dims, input_dims]` (`E, O, I`).
/// - `bias` (optional): `[num_experts, output_dims]` (`E, O`).
/// - `x` (input to `apply`): `[..., 1, input_dims]` — the trailing two dims
///   are the matmul `(M=1, K=input_dims)` pair, and the leading batch dims
///   index per-token. The caller is responsible for the `expand_dims(x,
///   (-2, -3))` reshape that mlx-lm's `SwitchMLP`/`SwitchGLU` apply before
///   calling `SwitchLinear` (see `switch_layers.py:178` / `:218`).
/// - `indices`: integer array of shape `[..., k]` — one flat expert id per
///   `(token, top-k-slot)`. Must broadcast against `x`'s batch dims.
/// - Output: `[..., 1, output_dims]` (same trailing-2 contract as `x`).
///
/// # Construction
///
/// Use [`SwitchLinear::from_parts`] to construct from already-loaded arrays;
/// the constructor verifies the shape contract and surfaces any mismatch as a
/// recoverable error rather than panicking at the first FFI call. The
/// `weight` / `bias` fields are PRIVATE — exposed via the
/// [`weight_ref`](Self::weight_ref) / [`bias`](Self::bias) read-only accessors —
/// specifically so that the constructor's shape validation is the *only*
/// way to populate them. Allowing external `&mut` mutation or struct-literal
/// construction would let a caller install a malformed `bias` (e.g. `[E, 1]`)
/// that silently broadcasts across every output channel inside
/// [`apply`](Self::apply)'s `take_axis + expand_dims + add` path, bypassing
/// the constructor's `[E, O]` rejection. This mirrors mlx-swift's
/// `public let weight` / `public let bias` immutability without giving up
/// the validation.
#[derive(Debug)]
pub struct SwitchLinear {
  /// Stacked per-expert weight tensor of shape `[num_experts, output_dims,
  /// input_dims]`. Stored as the python layout (mlx-lm reads safetensors
  /// directly into this shape); the `swapaxes(-1, -2)` to `[E, I, O]` for
  /// `gather_mm`'s `(M=I, K=O)` contract happens inside [`apply`](Self::apply).
  /// PRIVATE — read via [`Self::weight_ref`] — so the constructor's rank-3
  /// `[E, O, I]` validation can't be bypassed by struct-literal construction
  /// or `&mut` mutation.
  weight: Array,
  /// Optional per-expert bias of shape `[num_experts, output_dims]`. `None`
  /// matches python `SwitchLinear(..., bias=False)`. PRIVATE — read via
  /// [`Self::bias`] — for the same reason as [`Self::weight_ref`]: a malformed
  /// `[E, 1]` bias would silently broadcast across every output channel in
  /// [`apply`](Self::apply) without the constructor's `[E, O]` check.
  bias: Option<Array>,
}

impl SwitchLinear {
  /// Construct from already-loaded weight (and optional bias) arrays. Verifies
  /// the shape contract:
  ///
  /// - `weight` must be 3-D `[E, O, I]`.
  /// - `bias`, if present, must be 2-D `[E, O]` with matching `E` and `O`.
  ///
  /// Mismatches are surfaced as recoverable
  /// [`Error::RankMismatch`](crate::Error::RankMismatch) /
  /// [`Error::ShapePairMismatch`](crate::Error::ShapePairMismatch) rather than
  /// left to fail deep inside the FFI on first `apply`. Does not evaluate the arrays
  /// (lazy; only `shape()` metadata is read).
  pub fn from_parts(weight: Array, bias: Option<Array>) -> Result<Self> {
    let w_shape = weight.shape();
    if w_shape.len() != 3 {
      return Err(crate::Error::RankMismatch(RankMismatchPayload::new(
        "SwitchLinear::from_parts: weight must be 3-D [num_experts, output_dims, input_dims]",
        w_shape.len() as u32,
        w_shape.to_vec(),
      )));
    }
    if let Some(b) = &bias {
      let b_shape = b.shape();
      // Split the bias-shape check into the precise taxonomy: a rank-1 or
      // rank-3 bias must surface as
      // `RankMismatch`, not `ShapePairMismatch` — `ShapePairMismatchPayload`
      // is documented as distinct from `RankMismatchPayload` (rank differs).
      // Only after the ranks match do we compare the full `[E, O]` shape.
      if b_shape.len() != 2 {
        return Err(crate::Error::RankMismatch(RankMismatchPayload::new(
          "SwitchLinear::from_parts: bias must be rank-2 [num_experts, output_dims]",
          b_shape.len() as u32,
          b_shape.to_vec(),
        )));
      }
      if b_shape[0] != w_shape[0] || b_shape[1] != w_shape[1] {
        return Err(crate::Error::ShapePairMismatch(
          ShapePairMismatchPayload::new(
            "SwitchLinear::from_parts: bias must be [num_experts, output_dims]",
            vec![w_shape[0], w_shape[1]],
            b_shape.to_vec(),
          ),
        ));
      }
    }
    Ok(Self { weight, bias })
  }

  /// Read-only accessor for the per-expert weight stack (`[num_experts,
  /// output_dims, input_dims]`).
  ///
  /// The field is private specifically so the constructor's rank-3 shape
  /// validation is the only construction path; see the struct doc for the
  /// invariant rationale. Named `weight_ref` (non-Copy `&Array`
  /// accessor). Lazy — does not evaluate.
  pub fn weight_ref(&self) -> &Array {
    &self.weight
  }

  /// Read-only accessor for the optional per-expert bias (`[num_experts,
  /// output_dims]`, or `None` matching python `bias=False`).
  ///
  /// The field is private specifically so the constructor's `[E, O]` shape
  /// validation is the only construction path — a malformed `[E, 1]` bias
  /// would silently broadcast across every output channel in
  /// [`Self::apply`] otherwise. Lazy — does not evaluate.
  pub fn bias(&self) -> Option<&Array> {
    self.bias.as_ref()
  }

  /// Number of experts (the leading weight dim).
  pub fn num_experts(&self) -> usize {
    self.weight.shape()[0]
  }

  /// Output feature dim (second weight dim — the `O` in `[E, O, I]`).
  pub fn output_dims(&self) -> usize {
    self.weight.shape()[1]
  }

  /// Input feature dim (last weight dim — the `I` in `[E, O, I]`).
  pub fn input_dims(&self) -> usize {
    self.weight.shape()[2]
  }

  /// Forward pass: select per-token expert weights via `indices` and apply.
  /// Mirrors python `SwitchLinear.__call__(x, indices, sorted_indices)`
  /// (`switch_layers.py:120`).
  ///
  /// `x`: `[..., 1, input_dims]` per-token input (caller-prepared — see the
  /// shape contract on the struct doc).
  ///
  /// `indices`: integer array selecting which expert to apply per token
  /// (shape `[..., k]` for top-`k` routing, or `[...]` / `[..., 1]` for
  /// single-expert).
  ///
  /// `sorted_indices`: promise that `indices` is already sorted, enabling the
  /// faster `gather_mm` kernel. mlx-lm's `SwitchMLP`/`SwitchGLU` set this on
  /// the `_gather_sort` rebatched path (`switch_layers.py:181-185`); pass
  /// `false` for the general case.
  ///
  /// Returns `[..., 1, output_dims]`; does **not** evaluate (lazy, like every
  /// `mlxrs` op).
  pub fn apply(&self, x: &Array, indices: &Array, sorted_indices: bool) -> Result<Array> {
    // `self.weight` is `[E, O, I]`; `gather_mm`'s `b` contract is `(..., M, K)`
    // with the last two dims as the matmul pair. We want `x @ weightᵀ`, which
    // for x=(..., 1, I) and weight=(E, O, I) means b=(E, I, O), i.e. the last
    // two dims swapped. python does `self["weight"].swapaxes(-1, -2)`
    // (switch_layers.py:123); we do the same with `ops::shape::swapaxes`.
    let weight_t = shape::swapaxes(&self.weight, -1, -2)?;
    // Fused `gather_mm`: `b=weight_t` is the per-expert weight stack;
    // `rhs_indices=indices` selects which expert's weight matrix applies to
    // each row of `x`'s batch dims. `sorted_indices` is forwarded verbatim.
    let mut out = linalg_basic::gather_mm(x, &weight_t, None, Some(indices), sorted_indices)?;
    // Per-expert bias add: `bias[indices]` is `(..., k, O)`; we then
    // `expand_dims(..., -2)` to `(..., k, 1, O)` so it broadcasts against
    // `out=(..., k, 1, O)`. python: `x = x + mx.expand_dims(self["bias"][indices], -2)`
    // (switch_layers.py:128).
    if let Some(bias) = &self.bias {
      // `bias[indices]` in numpy/mlx fancy-indexing along axis 0 == take_axis.
      let selected = indexing::take_axis(bias, indices, 0)?;
      let broadcastable = shape::expand_dims_axes(&selected, &[-2])?;
      out = out.add(&broadcastable)?;
    }
    Ok(out)
  }
}

/// Quantized counterpart of [`SwitchLinear`]: per-expert affine-quantized
/// weights selected via `indices`, routed through a single fused
/// [`ops::quantized::gather_qmm`](crate::ops::quantized::gather_qmm).
///
/// Mirror of mlx-lm `QuantizedSwitchLinear` (`switch_layers.py:27`). The
/// `(weight, scales, biases)` triple is the output of
/// [`ops::quantized::quantize`](crate::ops::quantized::quantize) applied to a
/// dense `[E, O, I]` weight stack; the optional `bias` is the per-expert
/// addend (`[E, O]`, identical to [`SwitchLinear::bias`]).
///
/// `group_size` / `bits` / `mode` mirror the quantization scheme parameters
/// (mlx defaults: `group_size=64`, `bits=4`, `mode="affine"`).
///
/// # Shape contract
///
/// Same as [`SwitchLinear`] from the caller's POV — `x` is `[..., 1,
/// input_dims]`, `indices` is `[..., k]`, output is `[..., 1, output_dims]`.
/// The packed `weight` shape itself is dictated by the quantization scheme
/// (`gather_qmm` validates it); we surface only the input/output contract.
///
/// All fields are PRIVATE — exposed via the [`weight_ref`](Self::weight_ref) /
/// [`scales_ref`](Self::scales_ref) / [`quant_biases`](Self::quant_biases) /
/// [`bias`](Self::bias) / [`group_size`](Self::group_size) /
/// [`bits`](Self::bits) / [`mode`](Self::mode) read-only accessors — so the
/// constructor's validation is the only construction path. A struct-literal
/// `QuantizedSwitchLinear { bias: Some(bad_bias), .. }` or `&mut` mutation
/// would otherwise let a `[E, 1]` bias silently broadcast through
/// [`apply`](Self::apply)'s `take_axis + expand_dims + add` path, and a
/// post-construction `bits = -1` / `group_size = 0` / `mode = "garbage"`
/// would mis-decode the packed weight inside the FFI. Mirrors mlx-swift's
/// `public let` immutability discipline.
#[derive(Debug)]
pub struct QuantizedSwitchLinear {
  /// Packed quantized weight stack (output of
  /// [`ops::quantized::quantize`](crate::ops::quantized::quantize) on a dense
  /// `[E, O, I]` weight). PRIVATE — read via [`Self::weight_ref`] — so the
  /// constructor's rank-3 validation is the only construction path.
  weight: Array,
  /// Per-group scales (paired with `weight` — same `[E, O, n_groups]` layout
  /// the dense `quantize` op produces). PRIVATE — read via [`Self::scales_ref`] —
  /// for symmetry with `weight`: the `(weight, scales, quant_biases)` triple
  /// is internally consistent only when produced together by
  /// [`ops::quantized::quantize`](crate::ops::quantized::quantize), and
  /// mutating one without the others corrupts dequant inside `gather_qmm`.
  scales: Array,
  /// Per-group biases (the affine-mode addend; `None` for the bias-less float
  /// schemes `mxfp4`/`mxfp8`/`nvfp4`, mirroring `ops::quantized::quantize`'s
  /// `Option<Array>` return). PRIVATE — read via [`Self::quant_biases`] —
  /// for the same triple-consistency reason as [`Self::scales_ref`].
  quant_biases: Option<Array>,
  /// Optional per-expert output bias `[E, O]`. Independent of
  /// `quant_biases`; this is the layer's `Linear.bias`, not the quantization
  /// scheme's per-group `biases`. PRIVATE — read via [`Self::bias`] — so the
  /// constructor's `[E, O]` shape check (which guards against silent
  /// `[E, 1]`-broadcast in [`Self::apply`]) can't be bypassed.
  bias: Option<Array>,
  /// Quantization group size (must match what produced `weight` / `scales`).
  /// PRIVATE — read via [`Self::group_size`] — so the constructor's choice
  /// (which must match what produced `weight` / `scales`) can't be silently
  /// rewritten post-construction (forwarded to `gather_qmm`, which would
  /// mis-decode the packed weight on mismatch).
  group_size: i32,
  /// Quantization bit depth. PRIVATE — read via [`Self::bits`] — same
  /// scheme-consistency reason as [`Self::group_size`].
  bits: i32,
  /// Quantization scheme name (`"affine"` / `"mxfp4"` / …) — forwarded to
  /// `gather_qmm`. PRIVATE — read via [`Self::mode`] — same scheme-consistency
  /// reason as [`Self::group_size`]; a post-construction switch from
  /// `"affine"` to `"mxfp4"` (or vice versa) would re-interpret the packed
  /// `weight` / `scales` / `quant_biases` under the wrong scheme.
  mode: String,
}

impl QuantizedSwitchLinear {
  /// Construct from already-quantized arrays. Mirrors
  /// [`SwitchLinear::from_parts`] for the layer's STRUCTURAL invariants —
  /// per-mode value tables (`bits ∈ {2,3,4,5,6,8}` for affine,
  /// `mxfp4`/`nvfp4` requiring specific `(group_size, bits)` pairs —
  /// `mlx/mlx/ops.cpp:4745-4750,4808-4823`) are intentionally LEFT TO MLX-C
  /// (matches the faithful-port `feedback_match_official_binding_design`
  /// discipline; over-validating here would duplicate `mlx-c`'s
  /// `validate_quantized_input` and drift from upstream as new modes /
  /// param tables land).
  ///
  /// Validated structural invariants:
  ///
  /// - `weight` rank == 3 (packed `[E, O, I_packed]` — the output of
  ///   [`ops::quantized::quantize`](crate::ops::quantized::quantize) on a
  ///   dense `[E, O, I]` stack preserves `E` / `O` in the leading two dims and
  ///   compresses only the last dim, so `weight.shape()[0]` / `[1]` give
  ///   `E` / `O` directly).
  /// - `scales` rank == 3 (matches `weight` rank — mlx `quantize` preserves
  ///   the leading shape across the `(w_q, scales, biases)` triple,
  ///   `mlx/ops.cpp:4789-4798`). Leading dims must match: `scales.shape()[0]
  ///   == E` and `scales.shape()[1] == O`. The last axis is the per-group
  ///   count; `mlx-c` validates it against `group_size`.
  /// - `quant_biases`, if `Some`, must have the same shape as `scales`
  ///   (`affine_quantize` produces a `biases` array with the same `[E, O,
  ///   n_groups]` shape as `scales`, `mlx/ops.cpp:4793-4798`).
  /// - Mode arity (mirrors the `classify_triple`
  ///   `match (q.mode, b_opt)` pattern in `mlxrs/src/lm/quant.rs:613-640`):
  ///   `"affine"` REQUIRES `quant_biases` (3-output `affine_quantize`,
  ///   `mlx/ops.cpp:4793-4798`); `"mxfp4"` / `"mxfp8"` / `"nvfp4"` FORBID
  ///   `quant_biases` (2-output `fp_quantize`,
  ///   `mlx/ops.cpp:4890,4898-4904`). Unknown modes are rejected.
  /// - `bits > 0` and `group_size > 0` (basic non-zero sanity — DOES NOT
  ///   enforce per-mode value tables; mlx-c does).
  /// - `bias` (the per-expert output bias, distinct from `quant_biases`), if
  ///   `Some`, must be 2-D `[E, O]` with matching `E` and `O` (exact-shape
  ///   match, no broadcast — same rejection style as the dense constructor).
  ///   Without this check a malformed `[E, 1]` bias would broadcast silently
  ///   through the `take_axis + expand_dims + add` path in
  ///   [`apply`](Self::apply), adding the wrong scalar across every output
  ///   channel.
  ///
  /// Shape mismatches surface as typed
  /// [`Error::RankMismatch`](crate::Error::RankMismatch) /
  /// [`Error::ShapePairMismatch`](crate::Error::ShapePairMismatch); mode-arity /
  /// unknown-mode / zero-param failures surface as typed
  /// [`Error::InvariantViolation`](crate::Error::InvariantViolation) /
  /// [`Error::UnknownEnumValue`](crate::Error::UnknownEnumValue) /
  /// [`Error::OutOfRange`](crate::Error::OutOfRange). Does not evaluate (lazy;
  /// only `shape()` metadata is read).
  #[allow(clippy::too_many_arguments)]
  pub fn from_parts(
    weight: Array,
    scales: Array,
    quant_biases: Option<Array>,
    bias: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: impl Into<String>,
  ) -> Result<Self> {
    let mode = mode.into();
    let w_shape = weight.shape();
    if w_shape.len() != 3 {
      return Err(crate::Error::RankMismatch(RankMismatchPayload::new(
        "QuantizedSwitchLinear::from_parts: weight must be 3-D [num_experts, output_dims, packed_input_dims]",
        w_shape.len() as u32,
        w_shape.to_vec(),
      )));
    }
    // Packed-quantization dtype signal: mlx packs the affine-quantized
    // `(w_q, scales, biases)` triple's `w_q` into `uint32` words (`mlx
    // `quantize` / `affine_quantize`), and `gather_qmm` rejects any
    // non-`uint32` quantized weight. A rank-3 dense `f32` weight with
    // otherwise-matching scales / quant_biases would pass every shape check
    // and only fail deep inside the FFI on the first `apply`; reject it here
    // instead. Mirrors the `classify_triple` (`mlxrs/src/lm/quant.rs`) check
    // that already requires `.weight` dtype == `U32` for quantized triples.
    let w_dtype = weight.dtype()?;
    if w_dtype != Dtype::U32 {
      return Err(crate::Error::InvariantViolation(
        InvariantViolationPayload::new(
          "QuantizedSwitchLinear::from_parts: weight dtype (gather_qmm rejects non-`uint32` quantized weights)",
          "must be `uint32` (the mlx-quantized-weight dtype)",
        ),
      ));
    }

    let e = w_shape[0];
    let o = w_shape[1];

    // `scales` structural invariants: rank == weight rank (3); leading two
    // dims (E, O) must match weight. The trailing per-group count is
    // validated by mlx-c against `group_size`.
    let s_shape = scales.shape();
    if s_shape.len() != w_shape.len() {
      return Err(crate::Error::RankMismatch(RankMismatchPayload::new(
        "QuantizedSwitchLinear::from_parts: scales rank must match weight rank (mlx `quantize` preserves leading shape across (weight, scales, biases))",
        s_shape.len() as u32,
        s_shape.to_vec(),
      )));
    }
    if s_shape[0] != e || s_shape[1] != o {
      return Err(crate::Error::ShapePairMismatch(
        ShapePairMismatchPayload::new(
          "QuantizedSwitchLinear::from_parts: scales leading dims (E, O) must match weight",
          vec![e, o],
          vec![s_shape[0], s_shape[1]],
        ),
      ));
    }

    // `quant_biases`, when present, shares the per-group `[E, O, n_groups]`
    // layout with `scales` (`affine_quantize` produces both with the same
    // shape, `mlx/ops.cpp:4793-4798`). Split the check
    // so a divergent RANK surfaces as `RankMismatch` rather than being
    // collapsed into `ShapePairMismatch` — the latter is documented as
    // same-rank shape disagreement.
    if let Some(qb) = &quant_biases {
      let qb_shape = qb.shape();
      if qb_shape.len() != s_shape.len() {
        return Err(crate::Error::RankMismatch(RankMismatchPayload::new(
          "QuantizedSwitchLinear::from_parts: quant_biases rank must match scales rank (mlx `affine_quantize` writes identical `[E, O, n_groups]` shape)",
          qb_shape.len() as u32,
          qb_shape.to_vec(),
        )));
      }
      if qb_shape != s_shape {
        return Err(crate::Error::ShapePairMismatch(
          ShapePairMismatchPayload::new(
            "QuantizedSwitchLinear::from_parts: quant_biases shape must match scales (mlx `affine_quantize` writes identical `[E, O, n_groups]` shape)",
            s_shape.to_vec(),
            qb_shape.to_vec(),
          ),
        ));
      }
    }

    // Mode arity: mirrors the `classify_triple` `match (q.mode, b_opt)`
    // pattern at `mlxrs/src/lm/quant.rs:613-640`. `affine` REQUIRES
    // `quant_biases` (3-output `affine_quantize`,
    // `mlx/ops.cpp:4793-4798`); `mxfp4`/`mxfp8`/`nvfp4` FORBID
    // `quant_biases` (2-output `fp_quantize`,
    // `mlx/ops.cpp:4890,4898-4904`). Unknown modes are rejected so a typo
    // doesn't reach mlx-c with an unfamiliar tag.
    match (mode.as_str(), quant_biases.as_ref()) {
      ("affine", None) => {
        return Err(crate::Error::InvariantViolation(
          InvariantViolationPayload::new(
            "QuantizedSwitchLinear::from_parts: `affine` mode quant_biases (mlx `affine_quantize` always writes {w_q, scales, biases})",
            "must be Some for `affine` mode",
          ),
        ));
      }
      ("mxfp4" | "mxfp8" | "nvfp4", Some(_)) => {
        return Err(crate::Error::InvariantViolation(
          InvariantViolationPayload::new(
            "QuantizedSwitchLinear::from_parts: mxfp4 / mxfp8 / nvfp4 mode is scale-only (mlx `fp_quantize` writes {w_q, scales} with no biases); got a stale `quant_biases`",
            "must be None for mxfp4 / mxfp8 / nvfp4 mode",
          ),
        ));
      }
      ("affine", Some(_)) | ("mxfp4" | "mxfp8" | "nvfp4", None) => {
        // Expected layouts — fall through to the remaining checks.
      }
      (other, _) => {
        return Err(crate::Error::UnknownEnumValue(
          UnknownEnumValuePayload::new(
            "QuantizedSwitchLinear::mode",
            other.to_string(),
            &["affine", "mxfp4", "mxfp8", "nvfp4"],
          ),
        ));
      }
    }

    // Basic non-zero sanity on `bits` / `group_size`. Per-mode value tables
    // (`bits ∈ {2,3,4,5,6,8}` for affine — `mlx/ops.cpp:4745-4750`;
    // `mxfp4`/`nvfp4` requiring specific `(group_size, bits)` pairs —
    // `mlx/ops.cpp:4808-4823`) are DEFERRED to mlx-c per the faithful-port
    // discipline; checking them here would duplicate
    // `validate_quantized_input` and drift from upstream.
    if bits <= 0 {
      return Err(crate::Error::OutOfRange(OutOfRangePayload::new(
        "QuantizedSwitchLinear::from_parts: bits (per-mode value tables validated by mlx-c)",
        "must be > 0",
        format_smolstr!("{bits}"),
      )));
    }
    if group_size <= 0 {
      return Err(crate::Error::OutOfRange(OutOfRangePayload::new(
        "QuantizedSwitchLinear::from_parts: group_size (per-mode value tables validated by mlx-c)",
        "must be > 0",
        format_smolstr!("{group_size}"),
      )));
    }

    if let Some(b) = &bias {
      let b_shape = b.shape();
      // Split the bias-shape check into the precise taxonomy: a rank-1 or
      // rank-3 bias surfaces as
      // `RankMismatch`, not `ShapePairMismatch`. Only after the ranks
      // match do we compare the full `[E, O]` shape.
      if b_shape.len() != 2 {
        return Err(crate::Error::RankMismatch(RankMismatchPayload::new(
          "QuantizedSwitchLinear::from_parts: bias must be rank-2 [num_experts, output_dims]",
          b_shape.len() as u32,
          b_shape.to_vec(),
        )));
      }
      if b_shape[0] != e || b_shape[1] != o {
        return Err(crate::Error::ShapePairMismatch(
          ShapePairMismatchPayload::new(
            "QuantizedSwitchLinear::from_parts: bias must be [num_experts, output_dims]",
            vec![e, o],
            b_shape.to_vec(),
          ),
        ));
      }
    }
    Ok(Self {
      weight,
      scales,
      quant_biases,
      bias,
      group_size,
      bits,
      mode,
    })
  }

  /// Read-only accessor for the packed quantized weight stack.
  ///
  /// The field is private specifically so the constructor's rank-3 shape
  /// validation is the only construction path; see the struct doc for the
  /// invariant rationale. Named `weight_ref` (non-Copy `&Array`
  /// accessor). Lazy — does not evaluate.
  pub fn weight_ref(&self) -> &Array {
    &self.weight
  }

  /// Read-only accessor for the per-group quantization scales.
  ///
  /// The field is private to preserve the `(weight, scales, quant_biases)`
  /// triple's internal consistency — they're only well-formed when produced
  /// together by [`ops::quantized::quantize`](crate::ops::quantized::quantize).
  /// Named `scales_ref` (non-Copy `&Array` accessor). Lazy — does not
  /// evaluate.
  pub fn scales_ref(&self) -> &Array {
    &self.scales
  }

  /// Read-only accessor for the optional per-group quantization biases
  /// (`None` for the bias-less float schemes `mxfp4`/`mxfp8`/`nvfp4`).
  ///
  /// The field is private for the same triple-consistency reason as
  /// [`Self::scales_ref`]. Lazy — does not evaluate.
  pub fn quant_biases(&self) -> Option<&Array> {
    self.quant_biases.as_ref()
  }

  /// Read-only accessor for the optional per-expert output bias (`[E, O]`,
  /// or `None`).
  ///
  /// The field is private specifically so the constructor's `[E, O]` shape
  /// validation — which guards against silent `[E, 1]`-broadcast in
  /// [`Self::apply`] — is the only construction path. Lazy — does not
  /// evaluate.
  pub fn bias(&self) -> Option<&Array> {
    self.bias.as_ref()
  }

  /// Read-only accessor for the quantization group size (must match what
  /// produced [`Self::weight_ref`] / [`Self::scales_ref`]).
  ///
  /// The field is private to preserve the scheme-consistency invariant
  /// (`group_size` / `bits` / `mode` must collectively match what produced
  /// the packed arrays — `gather_qmm` mis-decodes them on mismatch).
  pub fn group_size(&self) -> i32 {
    self.group_size
  }

  /// Read-only accessor for the quantization bit depth.
  ///
  /// Private for the same scheme-consistency reason as [`Self::group_size`].
  pub fn bits(&self) -> i32 {
    self.bits
  }

  /// Read-only accessor for the quantization scheme name (`"affine"` /
  /// `"mxfp4"` / …).
  ///
  /// Private for the same scheme-consistency reason as [`Self::group_size`]:
  /// a post-construction switch from `"affine"` to `"mxfp4"` (or vice versa)
  /// would re-interpret the packed `weight` / `scales` / `quant_biases`
  /// under the wrong scheme.
  pub fn mode(&self) -> &str {
    &self.mode
  }

  /// Forward pass — quantized counterpart of [`SwitchLinear::apply`]. Mirrors
  /// python `QuantizedSwitchLinear.__call__(x, indices, sorted_indices)`
  /// (`switch_layers.py:75`): a single fused `gather_qmm` with `transpose=true`
  /// (matching the `[E, O, I]` weight layout's `output @ x` orientation),
  /// followed by the optional per-expert bias add.
  pub fn apply(&self, x: &Array, indices: &Array, sorted_indices: bool) -> Result<Array> {
    // `transpose=true` matches python (`switch_layers.py:82`): the packed
    // weight is laid out for the `output_dims x input_dims` orientation, so
    // gather_qmm transposes implicitly on the kernel side rather than us
    // calling `swapaxes` (cheaper for the quantized path).
    let mut out = quantized::gather_qmm(
      x,
      &self.weight,
      &self.scales,
      self.quant_biases.as_ref(),
      None, // lhs_indices: not used for SwitchLinear (per-token rhs only)
      Some(indices),
      true, // transpose
      self.group_size,
      self.bits,
      &self.mode,
      sorted_indices,
    )?;
    // Same per-expert bias path as the dense layer.
    if let Some(bias) = &self.bias {
      let selected = indexing::take_axis(bias, indices, 0)?;
      let broadcastable = shape::expand_dims_axes(&selected, &[-2])?;
      out = out.add(&broadcastable)?;
    }
    Ok(out)
  }
}

// ───────── MoE block rebatching helpers ─────────

/// Type of the per-block activation closure: a pure `&Array -> Result<Array>`
/// element-wise function (e.g. [`activations::silu`](super::activations::silu)
/// / [`activations::gelu_approx`](super::activations::gelu_approx)).
///
/// Mirrors mlx-swift's `activation: (MLXArray) -> MLXArray` `SwitchGLU` /
/// `SwitchMLP` field — a single-argument squashing function the block applies
/// to the gate (`SwitchGLU`) or to the `fc1` output (`SwitchMLP`). The python
/// reference instead passes a `SwiGLU` *module* whose `__call__(x, gate)` is
/// two-argument; the math is identical (`SwiGLU(x, gate) == silu(gate) * x`),
/// and the one-argument closure is the more Rust-natural shape — `SwitchGLU`
/// recovers the python form by multiplying `activation(gate)` by `up`.
///
/// Boxed (rather than a generic `F: Fn(..)` type parameter) so the block
/// structs stay non-generic — matching how [`crate::lm::generate`] boxes its
/// `Sampler` / `LogitsProcessor` runtime-configurable callables.
pub type Activation = Box<dyn Fn(&Array) -> Result<Array>>;

/// Validate the [`SwitchGLU`] / [`SwitchMLP`] `forward` shape contract on
/// `indices` *before* the `do_sort` / [`gather_sort`] path, so no `indices`
/// shape can be silently misrouted.
///
/// The python reference (`switch_layers.py`) always constructs `indices` with
/// an **explicit trailing top-`k` axis** — every MoE model feeds
/// `mx.argpartition(gates, kth=…, axis=-1)[..., -k:]` (or `[..., :k]`), which
/// keeps a length-`k` axis even for top-1 routing. `_gather_sort` then reads
/// `*_, M = indices.shape` and maps a sorted flat position back to a token
/// row via `order // M`. That arithmetic is only correct when `M` is the
/// genuine top-`k` count — i.e. when `indices` is `[..batch.., k]` with
/// leading dims matching `x`'s batch dims.
///
/// If a caller instead passes a top-1 route shaped like the batch with **no**
/// trailing `k` axis — `[N]` for a flat `x = [N, D]`, or `[B, S]` for `x =
/// [B, S, D]` — the last dim (`N` or `S`) is mis-read as `M`. For `[N]` every
/// `order // M` collapses to token row 0, so all routed rows reuse the first
/// token, yet `_scatter_unsort` + `squeeze` still return a plausibly-shaped
/// `[N, D]` output: silent Mixture-of-Experts corruption rather than a shape
/// error.
///
/// This port follows the reference faithfully — the reference's contract *is*
/// an explicit `k` axis — so an ambiguous `[..batch..]` `indices` (whose shape
/// equals `x`'s batch dims, with no extra trailing axis) is **rejected** here
/// with a recoverable [`Error::RankMismatch`](crate::Error::RankMismatch).
/// A caller doing top-1 routing must pass an explicit `[..batch.., 1]` (the
/// same shape `argpartition(...)[..., -1:]` produces); that singleton-`k`
/// shape sorts correctly (`M == 1`, `order // 1 == order`) and is accepted.
///
/// `x` here is the pre-`expand_dims` input — `[..batch.., input_dims]`, so its
/// batch dims are `x.shape()[..x.ndim()-1]`. The check: `indices` must have
/// exactly one more axis than those batch dims, and `indices.shape()[..n-1]`
/// must equal them. (`indices.ndim() == x.ndim()` with matching leading dims
/// is the ambiguous `[..batch..]` case — the trailing dim is a batch dim, not
/// a `k` axis — and is the one rejected.)
fn check_routing_indices(x: &Array, indices: &Array) -> Result<()> {
  let x_shape = x.shape();
  // `x` is `[..batch.., input_dims]`; the matmul `K` (input_dims) is the last
  // axis, so the batch dims are everything before it. `forward` is never
  // called with a rank-0 `x` (it `expand_dims`es `x` and routes per token),
  // but guard so the slice below cannot underflow.
  if x_shape.is_empty() {
    return Err(crate::Error::RankMismatch(RankMismatchPayload::new(
      "SwitchGLU/SwitchMLP::forward: x must have at least one axis ([..batch.., input_dims])",
      0,
      x_shape.to_vec(),
    )));
  }
  let x_batch = &x_shape[..x_shape.len() - 1];
  let idx_shape = indices.shape();
  // `indices` must be `[..batch.., k]`: exactly one trailing axis (the
  // explicit top-`k`, ≥ 1) beyond `x`'s batch dims, and leading dims equal to
  // them.
  //
  // Taxonomy: split the prior single `ShapePairMismatch`
  // into the precise violation class so a `[N]`-vs-`[N]` ambiguous-rank case
  // can't surface as "expected [N], got [N]":
  //   1. `idx_shape.len() != x_batch.len() + 1` ⇒ `RankMismatch`
  //      (missing/extra top-k axis — `[N]` for `x=[N,D]` falls here).
  //   2. Same rank but exactly one leading dim differs ⇒ `LengthMismatch`.
  //   3. Same rank with ≥ 2 leading dims differing ⇒ `ShapePairMismatch`
  //      (true multi-dim shape disagreement on the batch prefix).
  let expected_rank = x_batch.len() + 1;
  if idx_shape.len() != expected_rank {
    return Err(crate::Error::RankMismatch(RankMismatchPayload::new(
      "SwitchGLU/SwitchMLP::forward: indices must be [..batch.., k] — missing or extra trailing top-k axis (pass [..batch.., 1] for top-1 routing)",
      idx_shape.len() as u32,
      idx_shape.to_vec(),
    )));
  }
  // Same rank: now compare leading batch dims.
  let idx_lead = &idx_shape[..x_batch.len()];
  if idx_lead != x_batch {
    // Count differing dims to choose LengthMismatch (exactly one) vs
    // ShapePairMismatch (≥ 2). At this point ranks agree, so an exact
    // single-dim disparity has a well-defined expected/actual length pair.
    let mut diff_idx: Option<usize> = None;
    let mut diff_count = 0usize;
    for (i, (e, a)) in x_batch.iter().zip(idx_lead.iter()).enumerate() {
      if e != a {
        diff_count += 1;
        if diff_count == 1 {
          diff_idx = Some(i);
        }
      }
    }
    debug_assert!(diff_count >= 1, "idx_lead != x_batch ⇒ at least one diff");
    if diff_count == 1 {
      // SAFETY: `diff_count == 1` ⇒ we set `diff_idx` exactly once above.
      let i = diff_idx.expect("diff_count == 1 ⇒ diff_idx is Some");
      return Err(crate::Error::LengthMismatch(LengthMismatchPayload::new(
        "SwitchGLU/SwitchMLP::forward: indices leading-dim length must match x's corresponding batch dim",
        x_batch[i],
        idx_lead[i],
      )));
    }
    return Err(crate::Error::ShapePairMismatch(
      ShapePairMismatchPayload::new(
        "SwitchGLU/SwitchMLP::forward: indices leading dims must match x's batch dims",
        x_batch,
        idx_lead,
      ),
    ));
  }
  Ok(())
}

/// Sort the routed tokens by expert id so each expert's rows are contiguous,
/// enabling [`SwitchLinear`]'s faster `sorted_indices` `gather_mm` path.
///
/// 1:1 port of mlx-lm's `_gather_sort` (`switch_layers.py:12`) and mlx-swift's
/// `gatherSort` (`SwitchLayers.swift`):
///
/// ```text
/// M = indices.shape[-1]                  # top-k count
/// indices = indices.flatten()            # → 1-D, length B*M
/// order = argsort(indices)               # permutation sorting by expert id
/// inv_order = argsort(order)             # its inverse (undoes the sort)
/// return x.flatten(0, -3)[order // M], indices[order], inv_order
/// ```
///
/// `x` is the post-`expand_dims(x, (-2, -3))` input — rank ≥ 3, trailing dims
/// `(1, 1, D)`. `flatten(0, -3)` collapses every leading axis up to and
/// including `-3` into one, yielding `[B', 1, D]`; gathering that along axis 0
/// with `order // M` (integer-divide each sorted `(token, k)` slot back to its
/// token row) replicates each token's row once per top-k slot, in expert
/// order. Returns `(x_sorted, indices_sorted, inv_order)`; the caller threads
/// `inv_order` into [`scatter_unsort`] to restore the original order.
fn gather_sort(x: &Array, indices: &Array) -> Result<(Array, Array, Array)> {
  // `M = indices.shape[-1]` — the top-k count. `gather_sort` is only ever
  // reached for a non-empty `indices` (the `do_sort` gate requires
  // `size >= 64`), so the shape is non-empty.
  let m = *indices
    .shape()
    .last()
    .expect("gather_sort: indices must have at least one axis");
  // `indices.flatten()` — collapse to 1-D. `flatten(_, 0, -1)` over every
  // axis is mlx's argument-less `.flatten()`.
  let indices_flat = shape::flatten(indices, 0, -1)?;
  // `order = argsort(indices)` — the permutation that sorts the flattened
  // expert ids; `inv_order = argsort(order)` is its inverse (the permutation
  // that undoes the sort), exactly as the reference computes it.
  let order = misc::argsort(&indices_flat)?;
  let inv_order = misc::argsort(&order)?;
  // `x.flatten(0, -3)` — collapse all leading axes up to `-3` into one,
  // leaving `[B', 1, D]` (the trailing `(1, D)` are the matmul `(M=1, K=D)`
  // pair `expand_dims` added).
  let x_flat = shape::flatten(x, 0, -3)?;
  // `order // M` maps each sorted `(token, top-k-slot)` slot back to its
  // token row index; `m_arr` is the shape-`[1]` broadcast divisor.
  let m_u32 = u32::try_from(m).map_err(|_| {
    crate::Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "gather_sort: top-k count exceeds u32::MAX",
      "u32",
      [("top_k", m as u64)],
    ))
  })?;
  let m_arr = Array::from_slice::<u32>(&[m_u32], &(1usize,))?;
  let token_rows = arithmetic::floor_divide(&order, &m_arr)?;
  // `x.flatten(0, -3)[order // M]` — fancy-index along axis 0: each sorted
  // slot pulls its token's row. `indices[order]` reorders the expert ids the
  // same way so they line up with `x_sorted` row-for-row.
  let x_sorted = indexing::take_axis(&x_flat, &token_rows, 0)?;
  let indices_sorted = indexing::take_axis(&indices_flat, &order, 0)?;
  Ok((x_sorted, indices_sorted, inv_order))
}

/// Undo [`gather_sort`]: restore the original token order and (optionally)
/// reshape axis 0 back to the routing `indices` shape.
///
/// 1:1 port of mlx-lm's `_scatter_unsort` (`switch_layers.py:20`) and
/// mlx-swift's `scatterUnsort` (`SwitchLayers.swift`):
///
/// ```text
/// x = x[inv_order]                       # undo the expert-order sort
/// if shape is not None:
///     x = unflatten(x, 0, shape)         # axis-0 → the routing shape
/// ```
///
/// `x[inv_order]` is a fancy-index along axis 0 by the inverse permutation.
/// `unflatten(x, 0, shape)` reshapes the single leading axis (of size
/// `prod(shape)`) into `shape`, keeping the trailing dims — `mlx.unflatten`
/// of one axis into many is exactly a [`reshape`](crate::ops::shape::reshape)
/// of axis 0, so the port uses `reshape` rather than introducing a new op
/// wrapper.
fn scatter_unsort(x: &Array, inv_order: &Array, shape: &[usize]) -> Result<Array> {
  // `x[inv_order]` — fancy-index along axis 0 by the inverse permutation,
  // putting each row back at its pre-sort position.
  let unsorted = indexing::take_axis(x, inv_order, 0)?;
  // `unflatten(x, 0, shape)` — expand the leading axis (size `prod(shape)`)
  // into `shape`, retaining the trailing dims. mlx's `unflatten` of one axis
  // into several == `reshape` with `shape ++ trailing_dims`.
  let trailing = &unsorted.shape()[1..];
  let mut target: Vec<usize> = Vec::with_capacity(shape.len() + trailing.len());
  target.extend_from_slice(shape);
  target.extend_from_slice(trailing);
  shape::reshape(&unsorted, &target.as_slice())
}

// ───────── SwitchGLU ─────────

/// Gated MoE expert block: `down_proj(activation(gate_proj(x)) · up_proj(x))`,
/// routed per-token through `num_experts` experts.
///
/// 1:1 port of mlx-lm's `SwitchGLU` (`switch_layers.py:160`) and mlx-swift's
/// `SwitchGLU` (`MLXLMCommon/SwitchLayers.swift`). Three [`SwitchLinear`]
/// sub-layers — `gate_proj` / `up_proj` (`input_dims → hidden_dims`) and
/// `down_proj` (`hidden_dims → input_dims`) — plus an [`Activation`] applied
/// to the gate branch.
///
/// The python reference's default activation is the two-argument `SwiGLU`
/// module (`silu(gate) * x`); mlx-swift's is the one-argument `silu`. This
/// port follows the swift shape — a one-argument [`Activation`] closure — and
/// [`SwitchGLU::default_activation`] supplies [`silu`]
/// as the default. The forward pass multiplies `activation(x_gate)` by
/// `x_up`, which for `activation == silu` is exactly `swiglu(x_gate, x_up)`,
/// matching the python `SwiGLU` math.
///
/// # Shape contract
///
/// `forward(x, indices)` takes `x` of shape `[..batch.., input_dims]` and
/// `indices` of shape `[..batch.., k]` — the per-token top-`k` expert ids,
/// with leading dims matching `x`'s batch dims and an **explicit trailing
/// top-`k` axis** (use `[..batch.., 1]` for top-1 routing). Internally the
/// input is `expand_dims`'d to `[..batch.., 1, 1, input_dims]` so the trailing
/// dims are the matmul `(M=1, K=input_dims)` pair and the `k` axis
/// materializes by broadcasting against `indices`. The result is
/// `[..batch.., k, input_dims]`. An ambiguous `indices` shaped like the batch
/// with no `k` axis (`[N]` / `[B, S]`) is rejected with
/// [`Error::RankMismatch`](crate::Error::RankMismatch) — see the
/// `forward` method docs.
pub struct SwitchGLU {
  /// `input_dims → hidden_dims` gate projection (`SwitchLinear`). Its output
  /// is squashed by [`Self::activation`] before the elementwise gate.
  gate_proj: SwitchLinear,
  /// `input_dims → hidden_dims` up projection (`SwitchLinear`). Multiplied
  /// (un-activated) by `activation(gate_proj(x))`.
  up_proj: SwitchLinear,
  /// `hidden_dims → input_dims` down projection (`SwitchLinear`), applied to
  /// the gated hidden state.
  down_proj: SwitchLinear,
  /// Element-wise activation applied to the gate branch. Defaults to
  /// [`silu`] (see [`Self::default_activation`]),
  /// matching mlx-swift's `SwitchGLU` default and — composed with the `· up`
  /// multiply — the python `SwiGLU` default.
  activation: Activation,
}

impl SwitchGLU {
  /// The default [`SwitchGLU`] activation: [`silu`].
  ///
  /// Matches mlx-swift's `activation: @escaping (MLXArray) -> MLXArray =
  /// MLXNN.silu` default; composed with the block's `· up_proj(x)` multiply
  /// it reproduces the python reference's `SwiGLU` (`silu(gate) * x`) default.
  /// Exposed so callers can pass it explicitly, or wrap it.
  pub fn default_activation() -> Activation {
    Box::new(silu)
  }

  /// Construct a [`SwitchGLU`] from its three already-built [`SwitchLinear`]
  /// projections and an [`Activation`].
  ///
  /// The python / swift constructors allocate the three `SwitchLinear`s
  /// internally from `(input_dims, hidden_dims, num_experts, bias)`; here the
  /// caller supplies them directly so a loaded checkpoint's weights flow in
  /// without an intermediate random-init + assignment. Pass
  /// [`Self::default_activation`] for the reference default.
  ///
  /// Verifies the inter-projection shape contract: `gate_proj` and `up_proj`
  /// must share `[input_dims, hidden_dims]`, and `down_proj` must be the
  /// `[hidden_dims, input_dims]` inverse — a mismatch (e.g. a `down_proj`
  /// whose `input_dims` is not the shared `hidden_dims`) surfaces as a
  /// recoverable [`Error::ShapePairMismatch`](crate::Error::ShapePairMismatch)
  /// rather than failing deep inside the FFI on the first `forward`.
  pub fn new(
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
    activation: Activation,
  ) -> Result<Self> {
    check_glu_shapes(&gate_proj, &up_proj, &down_proj)?;
    Ok(Self {
      gate_proj,
      up_proj,
      down_proj,
      activation,
    })
  }

  /// Read-only accessor for the gate projection.
  pub fn gate_proj(&self) -> &SwitchLinear {
    &self.gate_proj
  }

  /// Read-only accessor for the up projection.
  pub fn up_proj(&self) -> &SwitchLinear {
    &self.up_proj
  }

  /// Read-only accessor for the down projection.
  pub fn down_proj(&self) -> &SwitchLinear {
    &self.down_proj
  }

  /// Forward pass — port of python `SwitchGLU.__call__` (`switch_layers.py:176`)
  /// and swift `SwitchGLU.callAsFunction`.
  ///
  /// `x`: `[..batch.., input_dims]`. `indices`: `[..batch.., k]` integer
  /// expert ids — leading dims must match `x`'s batch dims, with an **explicit
  /// trailing top-`k` axis** (pass `[..batch.., 1]` for top-1 routing).
  /// Returns `[..batch.., k, input_dims]`.
  ///
  /// An ambiguous `indices` shaped like the batch with **no** explicit `k`
  /// axis — `[N]` for a flat `x = [N, D]`, or `[B, S]` for `x = [B, S, D]` —
  /// is rejected with [`Error::RankMismatch`](crate::Error::RankMismatch).
  /// The python reference always constructs `indices` via
  /// `argpartition(...)[..., -k:]`, which keeps an explicit length-`k` axis
  /// even for top-1; the `_gather_sort` rebatch reads `indices.shape[-1]` as
  /// the top-`k` count, so a batch-shaped `indices` would have its last
  /// *batch* dim silently mis-read as top-`k` and route every token through
  /// the first token's row — silent Mixture-of-Experts corruption. Requiring
  /// the explicit `k` axis (faithful to the reference) makes that
  /// unrepresentable.
  ///
  /// Steps (verbatim from the reference):
  /// 1. `x = expand_dims(x, (-2, -3))` — add the `(top-k, M=1)` axes.
  /// 2. `do_sort = indices.size >= 64` — for many routed slots, sort tokens by
  ///    expert id (`gather_sort`) so each expert's rows are contiguous.
  /// 3. `x_up = up_proj(x)`, `x_gate = gate_proj(x)`, then
  ///    `x = down_proj(activation(x_gate) · x_up)` — all three `SwitchLinear`s
  ///    run with `sorted_indices = do_sort`.
  /// 4. If sorted, `scatter_unsort` restores the original token order.
  /// 5. `x.squeeze(-2)` drops the `M=1` axis.
  ///
  /// The python reference additionally `stop_gradient`s the indices when
  /// `self.training`; `mlxrs` is an inference port with no training mode (and
  /// integer routing indices carry no gradient), so that branch has no
  /// analogue. Returns a new lazy [`Array`] (no implicit eval).
  pub fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
    // Validate the `indices` shape contract *before* `expand_dims` / the
    // `do_sort` path: `indices` must be `[..batch.., k]` with an explicit
    // trailing top-k axis (see `check_routing_indices`). Without this an
    // ambiguous top-1 `[..batch..]` shape (`[N]` / `[B, S]`) would be silently
    // mis-routed by `gather_sort`'s `order // M` (every row would reuse the
    // first token) — silent MoE corruption, not a shape error.
    check_routing_indices(x, indices)?;

    // `x = mx.expand_dims(x, (-2, -3))` — insert the top-k and `M=1` axes,
    // taking `[..batch.., D]` to `[..batch.., 1, 1, D]`.
    let mut x = shape::expand_dims_axes(x, &[-2, -3])?;

    // `do_sort = indices.size >= 64`: with many routed slots, sorting tokens
    // by expert id makes each expert's rows contiguous for the fused kernel.
    let do_sort = indices.size() >= 64;
    // `idx` is the (possibly sorted) expert-id array fed to the projections;
    // `inv_order` is `Some` only on the sorted path, to undo the reorder.
    let mut idx = indices.try_clone()?;
    let mut inv_order: Option<Array> = None;
    if do_sort {
      let (x_sorted, idx_sorted, inv) = gather_sort(&x, indices)?;
      x = x_sorted;
      idx = idx_sorted;
      inv_order = Some(inv);
    }

    // `x_up = self.up_proj(x, idx)`, `x_gate = self.gate_proj(x, idx)` — the
    // two `input_dims → hidden_dims` projections of the routed input.
    let x_up = self.up_proj.apply(&x, &idx, do_sort)?;
    let x_gate = self.gate_proj.apply(&x, &idx, do_sort)?;
    // `self.down_proj(self.activation(x_up, x_gate), idx)`: the python
    // `SwiGLU` activation is `silu(gate) * x`; with the one-argument closure
    // that is `activation(x_gate) · x_up` — applied, then projected back down.
    let gated = (self.activation)(&x_gate)?.multiply(&x_up)?;
    x = self.down_proj.apply(&gated, &idx, do_sort)?;

    // `if do_sort: x = _scatter_unsort(x, inv_order, indices.shape)` — undo
    // the expert-order sort and restore the routing-`indices` leading shape.
    if let Some(inv) = &inv_order {
      x = scatter_unsort(&x, inv, &indices.shape())?;
    }

    // `return x.squeeze(-2)` — drop the `M=1` matmul axis.
    shape::squeeze_axes(&x, &[-2])
  }
}

impl std::fmt::Debug for SwitchGLU {
  /// Hand-written (the boxed [`Activation`] closure is not [`Debug`]); reports
  /// the three projections and elides the activation as `<fn>`.
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SwitchGLU")
      .field("gate_proj", &self.gate_proj)
      .field("up_proj", &self.up_proj)
      .field("down_proj", &self.down_proj)
      .field("activation", &"<fn>")
      .finish()
  }
}

/// Validate [`SwitchGLU`]'s inter-projection shape contract: `gate_proj` and
/// `up_proj` share `[input_dims → hidden_dims]`, and `down_proj` is the
/// `[hidden_dims → input_dims]` inverse. Surfaces a mismatch as a recoverable
/// [`Error::ShapePairMismatch`](crate::Error::ShapePairMismatch).
fn check_glu_shapes(
  gate_proj: &SwitchLinear,
  up_proj: &SwitchLinear,
  down_proj: &SwitchLinear,
) -> Result<()> {
  let (gi, gh, ge) = (
    gate_proj.input_dims(),
    gate_proj.output_dims(),
    gate_proj.num_experts(),
  );
  let (ui, uh, ue) = (
    up_proj.input_dims(),
    up_proj.output_dims(),
    up_proj.num_experts(),
  );
  let (di, dh, de) = (
    down_proj.input_dims(),
    down_proj.output_dims(),
    down_proj.num_experts(),
  );
  // `gate_proj` and `up_proj` are both `input_dims → hidden_dims`.
  if gi != ui || gh != uh {
    return Err(crate::Error::ShapePairMismatch(
      ShapePairMismatchPayload::new(
        "SwitchGLU: gate_proj and up_proj must share [input_dims, hidden_dims]",
        vec![gi, gh],
        vec![ui, uh],
      ),
    ));
  }
  // `down_proj` is `hidden_dims → input_dims` — the inverse of the shared
  // gate/up projection shape.
  if di != gh || dh != gi {
    return Err(crate::Error::ShapePairMismatch(
      ShapePairMismatchPayload::new(
        "SwitchGLU: down_proj must be the [hidden_dims, input_dims] inverse of gate_proj/up_proj",
        vec![gh, gi],
        vec![di, dh],
      ),
    ));
  }
  // Every projection routes the same expert population.
  if ge != ue || ge != de {
    return Err(crate::Error::ShapePairMismatch(
      ShapePairMismatchPayload::new(
        "SwitchGLU: all projections must have the same num_experts (gate_proj, up_proj, down_proj)",
        vec![ge, ge, ge],
        vec![ge, ue, de],
      ),
    ));
  }
  Ok(())
}

// ───────── SwitchMLP ─────────

/// Plain (un-gated) MoE expert block: `fc2(activation(fc1(x)))`, routed
/// per-token through `num_experts` experts.
///
/// 1:1 port of mlx-lm's `SwitchMLP` (`switch_layers.py:202`) and mlx-swift's
/// `SwitchMLP`. Two [`SwitchLinear`] sub-layers — `fc1` (`input_dims →
/// hidden_dims`) and `fc2` (`hidden_dims → input_dims`) — with an
/// [`Activation`] between them. Unlike [`SwitchGLU`] there is no gate branch:
/// the activation is applied to `fc1(x)` directly.
///
/// The python reference's default activation is `nn.GELU(approx="precise")` —
/// the `tanh` approximation, i.e.
/// [`gelu_approx`](super::activations::gelu_approx);
/// [`SwitchMLP::default_activation`] supplies exactly that.
///
/// # Shape contract
///
/// Same as [`SwitchGLU`]: `forward(x, indices)` takes `x` of shape
/// `[..batch.., input_dims]` and `indices` of `[..batch.., k]` (an **explicit
/// trailing top-`k` axis** required — `[..batch.., 1]` for top-1; an ambiguous
/// `[..batch..]` shape is rejected with
/// [`Error::RankMismatch`](crate::Error::RankMismatch), see the `forward`
/// method docs), and returns `[..batch.., k, input_dims]`.
pub struct SwitchMLP {
  /// `input_dims → hidden_dims` first projection (`SwitchLinear`); its output
  /// is squashed by [`Self::activation`].
  fc1: SwitchLinear,
  /// `hidden_dims → input_dims` second projection (`SwitchLinear`), applied
  /// to the activated hidden state.
  fc2: SwitchLinear,
  /// Element-wise activation applied between `fc1` and `fc2`. Defaults to
  /// [`gelu_approx`](super::activations::gelu_approx) (see
  /// [`Self::default_activation`]), matching the python reference's
  /// `nn.GELU(approx="precise")`.
  activation: Activation,
}

impl SwitchMLP {
  /// The default [`SwitchMLP`] activation:
  /// [`gelu_approx`](super::activations::gelu_approx).
  ///
  /// Matches the python reference's `activation=nn.GELU(approx="precise")` —
  /// `approx="precise"` selects the `tanh` GELU approximation, which is
  /// `gelu_approx`. Exposed so callers can pass it explicitly, or wrap it.
  pub fn default_activation() -> Activation {
    Box::new(super::activations::gelu_approx)
  }

  /// Construct a [`SwitchMLP`] from its two already-built [`SwitchLinear`]
  /// projections and an [`Activation`].
  ///
  /// The python / swift constructors allocate the two `SwitchLinear`s
  /// internally; here the caller supplies them directly so a loaded
  /// checkpoint's weights flow in without a random-init + assignment. Pass
  /// [`Self::default_activation`] for the reference default.
  ///
  /// Verifies the inter-projection shape contract: `fc2` must be the
  /// `[hidden_dims, input_dims]` inverse of `fc1`'s `[input_dims,
  /// hidden_dims]`, and both must route the same expert population — a
  /// mismatch surfaces as a recoverable
  /// [`Error::ShapePairMismatch`](crate::Error::ShapePairMismatch) /
  /// [`Error::LengthMismatch`](crate::Error::LengthMismatch).
  pub fn new(fc1: SwitchLinear, fc2: SwitchLinear, activation: Activation) -> Result<Self> {
    // `fc2` is `hidden_dims → input_dims` — the inverse of `fc1`'s
    // `input_dims → hidden_dims`.
    if fc2.input_dims() != fc1.output_dims() || fc2.output_dims() != fc1.input_dims() {
      return Err(crate::Error::ShapePairMismatch(
        ShapePairMismatchPayload::new(
          "SwitchMLP: fc2 must be the [hidden_dims, input_dims] inverse of fc1 [input_dims, hidden_dims]",
          vec![fc1.output_dims(), fc1.input_dims()],
          vec![fc2.input_dims(), fc2.output_dims()],
        ),
      ));
    }
    if fc1.num_experts() != fc2.num_experts() {
      return Err(crate::Error::LengthMismatch(LengthMismatchPayload::new(
        "SwitchMLP: fc1 and fc2 num_experts",
        fc1.num_experts(),
        fc2.num_experts(),
      )));
    }
    Ok(Self {
      fc1,
      fc2,
      activation,
    })
  }

  /// Read-only accessor for the first projection.
  pub fn fc1(&self) -> &SwitchLinear {
    &self.fc1
  }

  /// Read-only accessor for the second projection.
  pub fn fc2(&self) -> &SwitchLinear {
    &self.fc2
  }

  /// Forward pass — port of python `SwitchMLP.__call__` (`switch_layers.py:217`)
  /// and swift `SwitchMLP.callAsFunction`.
  ///
  /// `x`: `[..batch.., input_dims]`. `indices`: `[..batch.., k]` integer
  /// expert ids — leading dims must match `x`'s batch dims, with an **explicit
  /// trailing top-`k` axis** (pass `[..batch.., 1]` for top-1 routing).
  /// Returns `[..batch.., k, input_dims]`. An ambiguous `[..batch..]`
  /// `indices` with no `k` axis is rejected with
  /// [`Error::RankMismatch`](crate::Error::RankMismatch) — identical
  /// contract to [`SwitchGLU::forward`], whose docs explain why.
  ///
  /// Identical rebatching skeleton to [`SwitchGLU::forward`] — `expand_dims`,
  /// the `indices.size >= 64` `gather_sort` / `scatter_unsort` pair, the
  /// trailing `squeeze(-2)` — but the body is the un-gated
  /// `fc2(activation(fc1(x)))` rather than a gate·up product. The
  /// training-only `stop_gradient` has no analogue (inference port). Returns
  /// a new lazy [`Array`] (no implicit eval).
  pub fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
    // Validate the `indices` shape contract *before* `expand_dims` / the
    // `do_sort` path — identical to [`SwitchGLU::forward`]; see
    // `check_routing_indices`. Rejects an ambiguous top-1 `[..batch..]`
    // `indices` that `gather_sort` would otherwise silently mis-route.
    check_routing_indices(x, indices)?;

    // `x = mx.expand_dims(x, (-2, -3))` — add the top-k and `M=1` axes.
    let mut x = shape::expand_dims_axes(x, &[-2, -3])?;

    // `do_sort = indices.size >= 64` — sort tokens by expert id when many
    // slots are routed (see [`gather_sort`]).
    let do_sort = indices.size() >= 64;
    let mut idx = indices.try_clone()?;
    let mut inv_order: Option<Array> = None;
    if do_sort {
      let (x_sorted, idx_sorted, inv) = gather_sort(&x, indices)?;
      x = x_sorted;
      idx = idx_sorted;
      inv_order = Some(inv);
    }

    // `x = self.fc1(x, idx)`; `x = self.activation(x)`; `x = self.fc2(x, idx)`
    // — the plain un-gated two-projection body.
    x = self.fc1.apply(&x, &idx, do_sort)?;
    x = (self.activation)(&x)?;
    x = self.fc2.apply(&x, &idx, do_sort)?;

    // `if do_sort: x = _scatter_unsort(x, inv_order, indices.shape)`.
    if let Some(inv) = &inv_order {
      x = scatter_unsort(&x, inv, &indices.shape())?;
    }

    // `return x.squeeze(-2)`.
    shape::squeeze_axes(&x, &[-2])
  }
}

impl std::fmt::Debug for SwitchMLP {
  /// Hand-written (the boxed [`Activation`] closure is not [`Debug`]); reports
  /// the two projections and elides the activation as `<fn>`.
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SwitchMLP")
      .field("fc1", &self.fc1)
      .field("fc2", &self.fc2)
      .field("activation", &"<fn>")
      .finish()
  }
}

#[cfg(test)]
mod tests;
