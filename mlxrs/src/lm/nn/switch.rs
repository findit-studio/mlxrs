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
      // Split the bias-shape check into the precise taxonomy (Codex
      // 2026-05-27 R2): a rank-1 or rank-3 bias must surface as
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
            vec![w_shape[0], w_shape[1]].as_slice(),
            b_shape.to_vec().as_slice(),
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
  /// invariant rationale. Named `weight_ref` per §3 (non-Copy `&Array`
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
          vec![e, o].as_slice(),
          vec![s_shape[0], s_shape[1]].as_slice(),
        ),
      ));
    }

    // `quant_biases`, when present, shares the per-group `[E, O, n_groups]`
    // layout with `scales` (`affine_quantize` produces both with the same
    // shape, `mlx/ops.cpp:4793-4798`). Split the check (Codex 2026-05-27 R2)
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
            s_shape.to_vec().as_slice(),
            qb_shape.to_vec().as_slice(),
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
      // Split the bias-shape check into the precise taxonomy (Codex
      // 2026-05-27 R2): a rank-1 or rank-3 bias surfaces as
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
            vec![e, o].as_slice(),
            b_shape.to_vec().as_slice(),
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
  /// invariant rationale. Named `weight_ref` per §3 (non-Copy `&Array`
  /// accessor). Lazy — does not evaluate.
  pub fn weight_ref(&self) -> &Array {
    &self.weight
  }

  /// Read-only accessor for the per-group quantization scales.
  ///
  /// The field is private to preserve the `(weight, scales, quant_biases)`
  /// triple's internal consistency — they're only well-formed when produced
  /// together by [`ops::quantized::quantize`](crate::ops::quantized::quantize).
  /// Named `scales_ref` per §3 (non-Copy `&Array` accessor). Lazy — does not
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
  // Taxonomy (Codex 2026-05-27): split the prior single `ShapePairMismatch`
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
        vec![gi, gh].as_slice(),
        vec![ui, uh].as_slice(),
      ),
    ));
  }
  // `down_proj` is `hidden_dims → input_dims` — the inverse of the shared
  // gate/up projection shape.
  if di != gh || dh != gi {
    return Err(crate::Error::ShapePairMismatch(
      ShapePairMismatchPayload::new(
        "SwitchGLU: down_proj must be the [hidden_dims, input_dims] inverse of gate_proj/up_proj",
        vec![gh, gi].as_slice(),
        vec![di, dh].as_slice(),
      ),
    ));
  }
  // Every projection routes the same expert population.
  if ge != ue || ge != de {
    return Err(crate::Error::ShapePairMismatch(
      ShapePairMismatchPayload::new(
        "SwitchGLU: all projections must have the same num_experts (gate_proj, up_proj, down_proj)",
        vec![ge, ge, ge].as_slice(),
        vec![ge, ue, de].as_slice(),
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
          vec![fc1.output_dims(), fc1.input_dims()].as_slice(),
          vec![fc2.input_dims(), fc2.output_dims()].as_slice(),
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
mod tests {
  use super::*;

  /// Hand-traced golden: one weight matrix per expert (E=2), I=4, O=3.
  /// Token 0 (expert 0) selects expert-0 weights; token 1 (expert 1) selects
  /// expert-1 weights. Each weight matrix is `[O, I]`, so per-row dot product
  /// is `sum(W[o] * x)` for o in 0..O.
  ///
  /// Per-expert weights (in python `[E, O, I]` layout):
  /// ```text
  /// expert 0: [[1, 0, 0, 0],   expert 1: [[0, 1, 0, 0],
  ///            [0, 1, 0, 0],              [0, 0, 1, 0],
  ///            [0, 0, 1, 0]]              [0, 0, 0, 1]]
  /// ```
  /// Inputs (after `expand_dims(x, (-2, -3))`-style reshape — here we go
  /// straight to `[N, 1, I]`):
  /// ```text
  /// token 0: [1, 2, 3, 4]   → expert 0 → [1, 2, 3]   (project to first 3 features)
  /// token 1: [5, 6, 7, 8]   → expert 1 → [6, 7, 8]   (project to last 3 features)
  /// ```
  fn hand_traced_weight() -> Array {
    Array::from_slice::<f32>(
      &[
        // expert 0: I=4, O=3 → [O, I] = 3x4
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
        // expert 1: 3x4
        0.0, 1.0, 0.0, 0.0, // row 0
        0.0, 0.0, 1.0, 0.0, // row 1
        0.0, 0.0, 0.0, 1.0, // row 2
      ],
      &(2, 3, 4),
    )
    .unwrap()
  }

  fn hand_traced_input() -> Array {
    // [N=2, 1, I=4]
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &(2, 1, 4)).unwrap()
  }

  #[test]
  fn switch_linear_shape_no_bias() {
    let weight = hand_traced_weight();
    let layer = SwitchLinear::from_parts(weight, None).unwrap();
    assert_eq!(layer.num_experts(), 2);
    assert_eq!(layer.output_dims(), 3);
    assert_eq!(layer.input_dims(), 4);

    let x = hand_traced_input();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2usize,)).unwrap();
    let out = layer.apply(&x, &indices, false).unwrap();
    assert_eq!(out.shape(), vec![2, 1, 3]);
    assert_eq!(out.dtype().unwrap(), Dtype::F32);
  }

  #[test]
  fn switch_linear_hand_traced_no_bias() {
    let layer = SwitchLinear::from_parts(hand_traced_weight(), None).unwrap();
    let x = hand_traced_input();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap();
    let mut out = layer.apply(&x, &indices, false).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // Token 0 via expert 0: [1, 2, 3] (projects to features 0..3 of [1,2,3,4]).
    // Token 1 via expert 1: [6, 7, 8] (projects to features 1..4 of [5,6,7,8]).
    assert_eq!(got, vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0]);
  }

  #[test]
  fn switch_linear_hand_traced_with_bias() {
    // bias[E=2, O=3]; expert-0 adds [10, 20, 30], expert-1 adds [40, 50, 60].
    let bias = Array::from_slice::<f32>(
      &[
        10.0, 20.0, 30.0, // expert 0
        40.0, 50.0, 60.0, // expert 1
      ],
      &(2, 3),
    )
    .unwrap();
    let layer = SwitchLinear::from_parts(hand_traced_weight(), Some(bias)).unwrap();
    let x = hand_traced_input();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap();
    let mut out = layer.apply(&x, &indices, false).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // Token 0: [1+10, 2+20, 3+30] = [11, 22, 33].
    // Token 1: [6+40, 7+50, 8+60] = [46, 57, 68].
    assert_eq!(got, vec![11.0, 22.0, 33.0, 46.0, 57.0, 68.0]);
  }

  #[test]
  fn switch_linear_all_routed_to_one_expert_matches_plain_matmul() {
    // Edge: every token routed to expert 0 → output is equivalent to a plain
    // batched matmul `x @ weight[0]ᵀ`.
    let weight = hand_traced_weight();
    let layer = SwitchLinear::from_parts(weight, None).unwrap();
    let x = hand_traced_input();
    let indices = Array::from_slice::<u32>(&[0, 0], &(2,)).unwrap();
    let mut out = layer.apply(&x, &indices, false).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // Both tokens via expert 0: [1, 2, 3] (token 0) and [5, 6, 7] (token 1).
    assert_eq!(got, vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0]);
  }

  #[test]
  fn switch_linear_sorted_indices_matches_unsorted() {
    // `sorted_indices=true` is a performance hint — the result must match the
    // `false` path bit-for-bit when the indices truly are sorted.
    let layer = SwitchLinear::from_parts(hand_traced_weight(), None).unwrap();
    let x = hand_traced_input();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap(); // already sorted
    let mut via_sorted = layer.apply(&x, &indices, true).unwrap();
    let mut via_unsorted = layer.apply(&x, &indices, false).unwrap();
    assert_eq!(
      via_sorted.to_vec::<f32>().unwrap(),
      via_unsorted.to_vec::<f32>().unwrap()
    );
  }

  #[test]
  fn switch_linear_from_parts_rejects_2d_weight() {
    let bad = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
    let err = SwitchLinear::from_parts(bad, None).unwrap_err();
    assert!(matches!(err, crate::Error::RankMismatch(_)));
  }

  #[test]
  fn switch_linear_from_parts_rejects_mismatched_bias() {
    let weight = hand_traced_weight(); // [2, 3, 4]
    // Bad bias: [3, 3] (wrong E).
    let bad_bias =
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &(3, 3)).unwrap();
    let err = SwitchLinear::from_parts(weight, Some(bad_bias)).unwrap_err();
    assert!(matches!(err, crate::Error::ShapePairMismatch(_)));
  }

  /// Bias-rank-mismatch split (Codex 2026-05-27 R2): a rank-1 (or rank-3)
  /// bias must surface as `RankMismatch`, not as `ShapePairMismatch`, so
  /// typed-error consumers can distinguish the rank-vs-shape categories.
  /// Pre-split, every malformed-rank bias was collapsed into
  /// `ShapePairMismatch` by the single combined check.
  #[test]
  fn switch_linear_from_parts_rejects_rank_mismatch_bias() {
    let weight = hand_traced_weight(); // [2, 3, 4]
    // Rank-1 bias `[2]` (a plausible per-expert flat scalar) — must now
    // be `RankMismatch` with `actual == 1`.
    let bad_bias_rank1 = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap();
    let err =
      SwitchLinear::from_parts(weight.try_clone().unwrap(), Some(bad_bias_rank1)).unwrap_err();
    match err {
      crate::Error::RankMismatch(payload) => {
        assert_eq!(payload.actual(), 1, "rank-1 bias ⇒ actual rank 1");
        assert_eq!(payload.actual_shape(), &[2usize]);
      }
      other => panic!("expected RankMismatch on rank-1 bias, got {other:?}"),
    }
    // Rank-3 bias `[2, 3, 1]` — must also be `RankMismatch`.
    let bad_bias_rank3 =
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2usize, 3usize, 1usize)).unwrap();
    let err = SwitchLinear::from_parts(weight, Some(bad_bias_rank3)).unwrap_err();
    match err {
      crate::Error::RankMismatch(payload) => {
        assert_eq!(payload.actual(), 3, "rank-3 bias ⇒ actual rank 3");
        assert_eq!(payload.actual_shape(), &[2usize, 3, 1]);
      }
      other => panic!("expected RankMismatch on rank-3 bias, got {other:?}"),
    }
  }

  #[test]
  fn switch_linear_top_k_routing_shape() {
    // top-k=2 routing: indices is [N, k]; output is [N, k, O].
    let layer = SwitchLinear::from_parts(hand_traced_weight(), None).unwrap();
    // x must broadcast against the [N, k] indices on the leading batch dims —
    // mlx-lm's SwitchMLP feeds x as [..., 1, 1, I] (expand_dims (-2, -3)) so
    // the k slot broadcasts. Here we go straight to [N=2, k=2, 1, I=4] which
    // is the shape after the expand: token 0 will go through experts (0, 1),
    // token 1 through (1, 0).
    let x = Array::from_slice::<f32>(
      &[
        1.0, 2.0, 3.0, 4.0, // token 0 expert slot 0
        1.0, 2.0, 3.0, 4.0, // token 0 expert slot 1
        5.0, 6.0, 7.0, 8.0, // token 1 expert slot 0
        5.0, 6.0, 7.0, 8.0, // token 1 expert slot 1
      ],
      &(2, 2, 1, 4),
    )
    .unwrap();
    let indices = Array::from_slice::<u32>(&[0, 1, 1, 0], &(2, 2)).unwrap();
    let mut out = layer.apply(&x, &indices, false).unwrap();
    assert_eq!(out.shape(), vec![2, 2, 1, 3]);
    let got = out.to_vec::<f32>().unwrap();
    // token 0 slot 0 (expert 0): [1, 2, 3]
    // token 0 slot 1 (expert 1): [2, 3, 4]
    // token 1 slot 0 (expert 1): [6, 7, 8]
    // token 1 slot 1 (expert 0): [5, 6, 7]
    assert_eq!(
      got,
      vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0]
    );
  }

  // -------- QuantizedSwitchLinear --------

  /// A larger weight stack so the quantizer's `group_size=64` actually has at
  /// least one full group along the last axis (`I=64` here).
  const QUANT_INPUT_DIMS: usize = 64;

  fn quant_dense_weight() -> Array {
    let e: usize = 2;
    let o: usize = 4;
    let i = QUANT_INPUT_DIMS;
    let mut data = Vec::with_capacity(e * o * i);
    // Smooth ramp — friendly to 4-bit affine quant (per `ops_quantized.rs`).
    for ei in 0..e {
      for oi in 0..o {
        for ii in 0..i {
          data.push(((ei * 100 + oi * 10 + ii) as f32) * 0.001);
        }
      }
    }
    Array::from_slice::<f32>(&data, &(e, o, i)).unwrap()
  }

  fn quant_input() -> Array {
    let n: usize = 2;
    let i = QUANT_INPUT_DIMS;
    let mut data = Vec::with_capacity(n * i);
    for ni in 0..n {
      for ii in 0..i {
        data.push(((ni * 50 + ii) as f32) * 0.01);
      }
    }
    Array::from_slice::<f32>(&data, &(n, 1usize, i)).unwrap()
  }

  #[test]
  fn quantized_switch_linear_parity_within_quant_error() {
    let dense_w = quant_dense_weight();
    let dense_layer = SwitchLinear::from_parts(dense_w.try_clone().unwrap(), None).unwrap();

    // Quantize the dense weight using the affine scheme (default).
    let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
    assert!(
      q_biases.is_some(),
      "affine scheme produces per-group biases"
    );
    let q_layer =
      QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, None, 64, 4, "affine").unwrap();

    let x = quant_input();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap();
    let mut dense_out = dense_layer.apply(&x, &indices, false).unwrap();
    let mut quant_out = q_layer.apply(&x, &indices, false).unwrap();
    assert_eq!(dense_out.shape(), quant_out.shape());

    let dense = dense_out.to_vec::<f32>().unwrap();
    let quant = quant_out.to_vec::<f32>().unwrap();
    // 4-bit affine quant on a smooth ramp: per-element drift must stay within
    // a generous band relative to the dense magnitude (matches the tolerance
    // used in `tests/ops_quantized.rs::quantize_then_dequantize_round_trips_*`).
    let max_abs = dense.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (d, q) in dense.iter().zip(quant.iter()) {
      assert!(
        (d - q).abs() <= 0.1 * max_abs + 1e-3,
        "quantized SwitchLinear drift too large: dense={d} quant={q}"
      );
    }
  }

  #[test]
  fn quantized_switch_linear_from_parts_rejects_mismatched_bias() {
    // Quantize a `[E=2, O=4, I=64]` dense stack so the packed `weight` has
    // `shape[0]=E=2` and `shape[1]=O=4`. A `[E, 1]` bias would silently
    // broadcast across every output channel in `apply` (`take_axis` →
    // `expand_dims(-2)` → `add`) without this rejection.
    let dense_w = quant_dense_weight();
    let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
    // Bad bias: rank-2 but trailing dim is 1, not O=4.
    let bad_bias = Array::from_slice::<f32>(&[1.0, 2.0], &(2, 1)).unwrap();
    let err =
      QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, Some(bad_bias), 64, 4, "affine")
        .unwrap_err();
    assert!(matches!(err, crate::Error::ShapePairMismatch(_)));
  }

  /// Bias-rank-mismatch split (Codex 2026-05-27 R2): a rank-1 bias on the
  /// QUANTIZED layer must surface as `RankMismatch`, not as
  /// `ShapePairMismatch` — same taxonomy as the dense [`SwitchLinear`]
  /// sibling.
  #[test]
  fn quantized_switch_linear_from_parts_rejects_rank_mismatch_bias() {
    let dense_w = quant_dense_weight();
    let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
    // Rank-1 bias `[2]` — must now be `RankMismatch` with `actual == 1`.
    let bad_bias_rank1 = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap();
    let err = QuantizedSwitchLinear::from_parts(
      w_q,
      scales,
      q_biases,
      Some(bad_bias_rank1),
      64,
      4,
      "affine",
    )
    .unwrap_err();
    match err {
      crate::Error::RankMismatch(payload) => {
        assert_eq!(payload.actual(), 1, "rank-1 bias ⇒ actual rank 1");
        assert_eq!(payload.actual_shape(), &[2usize]);
      }
      other => panic!("expected RankMismatch on rank-1 bias, got {other:?}"),
    }
  }

  /// `quant_biases` rank must match `scales` rank — split out from the
  /// shape-pair check (Codex 2026-05-27 R2): a divergent rank now surfaces
  /// as `RankMismatch`, not `ShapePairMismatch`. Pre-split, `qb_shape !=
  /// s_shape` collapsed both rank and shape divergences into the same
  /// variant.
  #[test]
  fn quantized_switch_linear_from_parts_rejects_quant_biases_rank_mismatch() {
    // Valid affine triple — `scales` is rank-3 `[E, O, n_groups]`. Supply a
    // rank-2 `quant_biases` and observe `RankMismatch` with `actual == 2`.
    let dense_w = quant_dense_weight(); // [2, 4, 64]
    let (w_q, scales, _q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
    // Bad rank-2 quant_biases `[2, 4]` — wrong rank entirely.
    let bad_qb =
      Array::from_slice::<f32>(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &(2usize, 4usize))
        .unwrap();
    let err = QuantizedSwitchLinear::from_parts(w_q, scales, Some(bad_qb), None, 64, 4, "affine")
      .unwrap_err();
    match err {
      crate::Error::RankMismatch(payload) => {
        assert_eq!(payload.actual(), 2, "rank-2 quant_biases ⇒ actual rank 2");
        assert_eq!(payload.actual_shape(), &[2usize, 4]);
      }
      other => panic!("expected RankMismatch on rank-2 quant_biases, got {other:?}"),
    }
  }

  #[test]
  fn quantized_switch_linear_with_bias_parity_within_quant_error() {
    // Valid `[E=2, O=4]` bias on both the dense and quantized layers; the
    // quantized output (with bias) must stay within the same quant-error band
    // as the bias-less parity test above.
    let dense_w = quant_dense_weight();
    // Distinct per-expert per-channel bias so any wrong-broadcast would visibly
    // diverge from the dense reference.
    let bias = Array::from_slice::<f32>(
      &[
        10.0, 20.0, 30.0, 40.0, // expert 0
        50.0, 60.0, 70.0, 80.0, // expert 1
      ],
      &(2, 4),
    )
    .unwrap();
    let dense_layer = SwitchLinear::from_parts(
      dense_w.try_clone().unwrap(),
      Some(bias.try_clone().unwrap()),
    )
    .unwrap();

    let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
    let q_layer =
      QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, Some(bias), 64, 4, "affine")
        .unwrap();

    let x = quant_input();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap();
    let mut dense_out = dense_layer.apply(&x, &indices, false).unwrap();
    let mut quant_out = q_layer.apply(&x, &indices, false).unwrap();
    assert_eq!(dense_out.shape(), quant_out.shape());

    let dense = dense_out.to_vec::<f32>().unwrap();
    let quant = quant_out.to_vec::<f32>().unwrap();
    let max_abs = dense.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (d, q) in dense.iter().zip(quant.iter()) {
      assert!(
        (d - q).abs() <= 0.1 * max_abs + 1e-3,
        "quantized SwitchLinear (with bias) drift too large: dense={d} quant={q}"
      );
    }
  }

  // ─── QuantizedSwitchLinear::from_parts structural-invariant tests (Codex R4) ───
  //
  // Mirrors the `classify_triple` `match (q.mode, b_opt)` mode-arity pattern
  // (`mlxrs/src/lm/quant.rs:613-640`): validates STRUCTURAL invariants on the
  // packed `(weight, scales, quant_biases)` triple. Per-mode value tables
  // (`bits ∈ {2,3,4,5,6,8}` for affine; `mxfp4` / `nvfp4` require specific
  // `(group_size, bits)` pairs — `mlx/ops.cpp:4745-4750,4808-4823`) are
  // DEFERRED to mlx-c per `feedback_match_official_binding_design` —
  // duplicating them in mlxrs would drift from upstream.
  //
  // Quantization fixtures: a smooth `(2, 4, 64)` ramp under `affine /
  // group_size=64 / bits=4` (matches existing parity tests); a `(2, 4, 64)`
  // ramp under `mxfp4 / group_size=32 / bits=4` (the only `(gs, b)` mlx-c
  // accepts for `mxfp4`, `mlx/ops.cpp:4808-4823`).

  /// `mxfp4` fixture: the only `(group_size, bits)` pair mlx-c accepts for
  /// `mxfp4` is `(32, 4)` (`quantization_params_from_mode` in
  /// `mlx/ops.cpp:4808-4823`). Reuses the same dense `[E=2, O=4, I=64]`
  /// stack as `quant_dense_weight`; the resulting packed `scales` is
  /// `[2, 4, 64/32 = 2]`, and `quant_biases == None` (bias-less fp scheme).
  fn quant_mxfp4_triple() -> (Array, Array, Option<Array>) {
    let dense_w = quant_dense_weight();
    quantized::quantize(&dense_w, 32, 4, "mxfp4", None).unwrap()
  }

  #[test]
  fn quantized_switch_linear_from_parts_rejects_mismatched_scales_leading_dims() {
    // `weight [E=2, O=4, I_packed]` paired with `scales [E=3, O=4, ..]` —
    // the leading `E` mismatches. mlx `quantize` always preserves the
    // leading shape across (weight, scales, biases) (`mlx/ops.cpp:4789-4798`),
    // so this combination is structurally impossible from a real
    // `quantize` call.
    let dense_w = quant_dense_weight(); // [2, 4, 64]
    let (w_q, _scales, _q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
    // Build a mismatched scales with E=3 (instead of E=2). Use a fresh
    // `[3, 4, 1]` rank-3 array; the trailing-axis value is irrelevant
    // because the leading-dim check fires first.
    let bad_scales = Array::from_slice::<f32>(
      &[
        1.0, 1.0, 1.0, 1.0, // E=0
        1.0, 1.0, 1.0, 1.0, // E=1
        1.0, 1.0, 1.0, 1.0, // E=2 (extra)
      ],
      &(3usize, 4usize, 1usize),
    )
    .unwrap();
    let err =
      QuantizedSwitchLinear::from_parts(w_q, bad_scales, None, None, 64, 4, "mxfp4").unwrap_err();
    assert!(
      matches!(err, crate::Error::ShapePairMismatch(_)),
      "expected ShapePairMismatch on scales leading dims, got {err:?}"
    );
  }

  #[test]
  fn quantized_switch_linear_from_parts_rejects_non_u32_weight() {
    // A rank-3 dense `f32` weight `[2, 4, 8]` with otherwise-matching affine
    // `scales` / `quant_biases` passes every shape / mode-arity check but is
    // NOT a packed quantized weight — mlx packs `affine_quantize`'s `w_q`
    // into `uint32` words and `gather_qmm` rejects any non-`uint32` quantized
    // weight. Without the dtype guard `from_parts` returns `Ok` and the
    // failure surfaces deep inside the FFI on the first `apply`; the guard
    // moves it to construction. Mirrors `classify_triple`'s `.weight` ==
    // `U32` requirement for quantized triples.
    let dense_data = vec![0.5f32; 2 * 4 * 8];
    let dense_weight = Array::from_slice::<f32>(&dense_data, &(2usize, 4usize, 8usize)).unwrap();
    // `scales` / `quant_biases` shaped to match the leading `[E=2, O=4, ..]`
    // dims so the dtype check — placed right after the weight-rank check —
    // is what fires, not a downstream shape mismatch.
    let scales_data = vec![1.0f32; 2 * 4];
    let scales = Array::from_slice::<f32>(&scales_data, &(2usize, 4usize, 1usize)).unwrap();
    let qb_data = vec![0.0f32; 2 * 4];
    let quant_biases = Array::from_slice::<f32>(&qb_data, &(2usize, 4usize, 1usize)).unwrap();
    let err = QuantizedSwitchLinear::from_parts(
      dense_weight,
      scales,
      Some(quant_biases),
      None,
      64,
      4,
      "affine",
    )
    .unwrap_err();
    match &err {
      crate::Error::InvariantViolation(payload) => {
        assert!(
          payload.context().contains("weight dtype") || payload.requirement().contains("uint32"),
          "InvariantViolation context/requirement should name the dtype invariant, got context={:?} requirement={:?}",
          payload.context(),
          payload.requirement()
        );
      }
      other => panic!("expected InvariantViolation naming the dtype invariant, got {other:?}"),
    }
  }

  #[test]
  fn quantized_switch_linear_from_parts_rejects_quant_biases_shape_mismatch() {
    // Valid affine triple but `quant_biases` has a shape distinct from
    // `scales` — `affine_quantize` writes them with identical
    // `[E, O, n_groups]` shape (`mlx/ops.cpp:4793-4798`), so a divergent
    // shape is structurally invalid. Use `group_size=32` here so `scales`
    // resolves to `[E=2, O=4, n_groups=2]` and the `[2, 4, 1]` bad
    // `quant_biases` truly mismatches (with the default `group_size=64`,
    // `scales` is `[2, 4, 1]` and a `[2, 4, 1]` bad would coincidentally
    // match, masking the check).
    let dense_w = quant_dense_weight(); // [2, 4, 64]
    let (w_q, scales, _q_biases) = quantized::quantize(&dense_w, 32, 4, "affine", None).unwrap();
    // scales is `[2, 4, 64/32 = 2]`; bad quant_biases is `[2, 4, 1]` —
    // trailing dim mismatches.
    let bad_qb = Array::from_slice::<f32>(
      &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
      &(2usize, 4usize, 1usize),
    )
    .unwrap();
    let err = QuantizedSwitchLinear::from_parts(w_q, scales, Some(bad_qb), None, 32, 4, "affine")
      .unwrap_err();
    assert!(
      matches!(err, crate::Error::ShapePairMismatch(_)),
      "expected ShapePairMismatch on quant_biases shape, got {err:?}"
    );
  }

  #[test]
  fn quantized_switch_linear_from_parts_affine_requires_quant_biases() {
    // `affine` mode is the 3-output `affine_quantize` arity
    // (`mlx/ops.cpp:4793-4798`); a `None` `quant_biases` next to it is a
    // structurally incomplete triple, rejected at construction.
    let dense_w = quant_dense_weight();
    let (w_q, scales, _q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
    let err =
      QuantizedSwitchLinear::from_parts(w_q, scales, None, None, 64, 4, "affine").unwrap_err();
    assert!(
      matches!(err, crate::Error::InvariantViolation(_)),
      "expected InvariantViolation on affine-missing-quant_biases, got {err:?}"
    );
  }

  #[test]
  fn quantized_switch_linear_from_parts_mxfp4_forbids_quant_biases() {
    // `mxfp4` is scale-only (`fp_quantize` 2-output arity,
    // `mlx/ops.cpp:4890,4898-4904`); a stale `quant_biases` next to it
    // would be retained from an unrelated `affine` triple and is rejected
    // at construction.
    let (w_q, scales, _none_qb) = quant_mxfp4_triple();
    // Fabricate a stale `quant_biases` shaped to match `scales` so the
    // mode-arity check fires before the shape-match check.
    let s_shape = scales.shape();
    let n_groups = s_shape[2];
    let stale_qb_data = vec![0.0f32; 2 * 4 * n_groups];
    let stale_qb = Array::from_slice::<f32>(&stale_qb_data, &(2usize, 4usize, n_groups)).unwrap();
    let err = QuantizedSwitchLinear::from_parts(w_q, scales, Some(stale_qb), None, 32, 4, "mxfp4")
      .unwrap_err();
    assert!(
      matches!(err, crate::Error::InvariantViolation(_)),
      "expected InvariantViolation on mxfp4-with-stale-quant_biases, got {err:?}"
    );
  }

  #[test]
  fn quantized_switch_linear_from_parts_unknown_mode() {
    // Unknown mode tag — neither `affine` nor any of the fp schemes — is
    // rejected so a typo doesn't reach mlx-c with an unfamiliar mode
    // string (where it would surface as a less-specific backend error).
    let dense_w = quant_dense_weight();
    let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
    let err =
      QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, None, 64, 4, "unknown").unwrap_err();
    assert!(
      matches!(err, crate::Error::UnknownEnumValue(_)),
      "expected UnknownEnumValue on unknown mode, got {err:?}"
    );
  }

  #[test]
  fn quantized_switch_linear_from_parts_zero_bits_or_group_size() {
    // Basic non-zero sanity on `bits` / `group_size` (per-mode value tables
    // remain deferred to mlx-c — we just catch the trivial 0 here so the
    // FFI doesn't divide-by-zero downstream).
    let dense_w = quant_dense_weight();
    let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();

    let err_bits = QuantizedSwitchLinear::from_parts(
      w_q.try_clone().unwrap(),
      scales.try_clone().unwrap(),
      q_biases.as_ref().map(|q| q.try_clone().unwrap()),
      None,
      64,
      0,
      "affine",
    )
    .unwrap_err();
    assert!(
      matches!(err_bits, crate::Error::OutOfRange(_)),
      "expected OutOfRange on bits=0, got {err_bits:?}"
    );

    let err_gs =
      QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, None, 0, 4, "affine").unwrap_err();
    assert!(
      matches!(err_gs, crate::Error::OutOfRange(_)),
      "expected OutOfRange on group_size=0, got {err_gs:?}"
    );
  }

  /// Regression: a valid `mxfp4` triple (scales-only, `quant_biases ==
  /// None`) constructs cleanly. Closes the new structural-invariant block
  /// over the bias-less fp branch; the existing affine parity tests already
  /// cover the `(affine, Some)` branch.
  #[test]
  fn quantized_switch_linear_from_parts_mxfp4_scales_only_ok() {
    let (w_q, scales, none_qb) = quant_mxfp4_triple();
    assert!(none_qb.is_none(), "mxfp4 quantize must yield None biases");
    let layer = QuantizedSwitchLinear::from_parts(w_q, scales, None, None, 32, 4, "mxfp4").unwrap();
    assert_eq!(layer.weight_ref().shape()[0], 2); // E=2
    assert_eq!(layer.weight_ref().shape()[1], 4); // O=4
    assert!(layer.quant_biases().is_none());
    assert_eq!(layer.mode(), "mxfp4");
  }

  // ─── SwitchLinear / QuantizedSwitchLinear field-visibility regressions (Codex R3) ───

  /// `SwitchLinear`'s `weight` / `bias` are PRIVATE fields with read-only
  /// public accessors. This test exercises the accessors and — by virtue of
  /// compiling without reaching for the fields — confirms the read path
  /// goes through them. Direct field access from outside `super::` would
  /// fail to compile (the fields' visibility is module-private). External
  /// code previously could write `layer.bias = Some(bad_bias)` (any shape)
  /// and then `layer.apply(_)` would silently broadcast a malformed
  /// `[E, 1]` bias across every output channel; with the fields private,
  /// that mutation path is statically impossible — `from_parts` is the
  /// only construction path, and its `[E, O]` check is the only path that
  /// matters.
  #[test]
  fn switch_linear_fields_are_read_only_via_accessors() {
    let layer = SwitchLinear::from_parts(hand_traced_weight(), None).unwrap();
    assert_eq!(layer.weight_ref().shape(), vec![2, 3, 4]);
    assert!(layer.bias().is_none());

    let bias = Array::from_slice::<f32>(
      &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], // [E=2, O=3]
      &(2, 3),
    )
    .unwrap();
    let layer_with_bias = SwitchLinear::from_parts(hand_traced_weight(), Some(bias)).unwrap();
    assert_eq!(layer_with_bias.weight_ref().shape(), vec![2, 3, 4]);
    assert_eq!(layer_with_bias.bias().unwrap().shape(), vec![2, 3]);
    // (compile-fail) external `layer.weight = ...` and `layer.bias = ...`
    // are both private-field errors; trying them here from inside `super::`
    // would compile (same module), so we don't try — the visibility
    // guarantee is what the regression turns on, not a runtime check.
  }

  /// `QuantizedSwitchLinear`'s `weight` / `scales` / `quant_biases` /
  /// `bias` / `group_size` / `bits` / `mode` are PRIVATE fields with
  /// read-only public accessors. Same rationale as
  /// [`switch_linear_fields_are_read_only_via_accessors`]: external
  /// struct-literal construction `QuantizedSwitchLinear { bias:
  /// Some(bad_bias), .. }` or `&mut` mutation would otherwise bypass
  /// `from_parts`'s `[E, O]` bias-shape check, and post-construction
  /// `bits = -1` / `group_size = 0` / `mode = "garbage"` would mis-decode
  /// the packed weight inside the FFI.
  #[test]
  fn quantized_switch_linear_fields_are_read_only_via_accessors() {
    let dense_w = quant_dense_weight();
    let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
    let bias = Array::from_slice::<f32>(
      &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], // [E=2, O=4]
      &(2, 4),
    )
    .unwrap();
    let q_layer =
      QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, Some(bias), 64, 4, "affine")
        .unwrap();
    // All accessors return the constructor-validated values; the field
    // privacy is what guarantees no other write path exists.
    assert_eq!(q_layer.weight_ref().shape()[0], 2); // E=2
    assert_eq!(q_layer.weight_ref().shape()[1], 4); // O=4
    assert_eq!(q_layer.scales_ref().shape()[0], 2); // E
    assert!(q_layer.quant_biases().is_some()); // affine ⇒ Some
    assert_eq!(q_layer.bias().unwrap().shape(), vec![2, 4]);
    assert_eq!(q_layer.group_size(), 64);
    assert_eq!(q_layer.bits(), 4);
    assert_eq!(q_layer.mode(), "affine");
    // (compile-fail) external `q_layer.bits = -1`, `q_layer.mode =
    // "garbage".into()`, etc. are all private-field errors; trying them
    // here from inside `super::` would compile (same module), so we don't
    // try — the visibility guarantee is what the regression turns on, not
    // a runtime check.
  }

  // ─── SwitchGLU / SwitchMLP block tests ───
  //
  // Hand-traced over a tiny known expert set (E=2, I=H=2). The projections
  // are built from explicit per-expert weight stacks so the forward math is
  // exactly reproducible by hand; `silu`/identity activations keep the
  // reference value closed-form.

  /// Logistic sigmoid — the reference scalar formula.
  fn sigmoid_ref(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
  }

  /// `silu(v) = v · σ(v)` — the reference scalar formula.
  fn silu_ref(v: f32) -> f32 {
    v * sigmoid_ref(v)
  }

  /// Per-element near-equality (f32 op-graph vs f64-ish reference).
  fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(
      got.len(),
      want.len(),
      "length mismatch: {got:?} vs {want:?}"
    );
    for (g, w) in got.iter().zip(want.iter()) {
      assert!(
        (g - w).abs() <= 1e-5 + 1e-5 * w.abs(),
        "block output mismatch: got {g}, want {w} (full got {got:?}, want {want:?})"
      );
    }
  }

  /// A `[E=2, O=2, I=2]` weight stack: expert 0 is the 2×2 identity, expert 1
  /// is the 2×2 swap `[[0,1],[1,0]]`. Routing token 0 → expert 0 leaves its
  /// features in place; routing → expert 1 swaps them — so a forward result
  /// reveals which expert each token was routed through.
  fn identity_then_swap_weight() -> Array {
    Array::from_slice::<f32>(
      &[
        // expert 0: identity
        1.0, 0.0, //
        0.0, 1.0, //
        // expert 1: swap
        0.0, 1.0, //
        1.0, 0.0, //
      ],
      &(2, 2, 2),
    )
    .unwrap()
  }

  /// A `[E=2, O=2, I=2]` all-identity weight stack — both experts are the 2×2
  /// identity, so the projection is a no-op `y = x` regardless of routing.
  fn all_identity_weight() -> Array {
    Array::from_slice::<f32>(
      &[
        1.0, 0.0, 0.0, 1.0, // expert 0: identity
        1.0, 0.0, 0.0, 1.0, // expert 1: identity
      ],
      &(2, 2, 2),
    )
    .unwrap()
  }

  #[test]
  fn switch_glu_hand_traced_two_experts() {
    // gate_proj routes through identity (expert 0) / swap (expert 1);
    // up_proj and down_proj are pure identity. With the `silu` activation the
    // block computes `down(silu(gate(x)) · up(x)) = silu(gate_e(x)) · x`.
    let gate_proj = SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap();
    let up_proj = SwitchLinear::from_parts(all_identity_weight(), None).unwrap();
    let down_proj = SwitchLinear::from_parts(all_identity_weight(), None).unwrap();
    let glu = SwitchGLU::new(
      gate_proj,
      up_proj,
      down_proj,
      SwitchGLU::default_activation(), // silu
    )
    .unwrap();

    // Two tokens [1, 2] and [3, 4]; token 0 → expert 0, token 1 → expert 1.
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2, 1)).unwrap();
    let mut out = glu.forward(&x, &indices).unwrap();
    // forward(x) returns [N=2, k=1, I=2].
    assert_eq!(out.shape(), vec![2, 1, 2]);
    let got = out.to_vec::<f32>().unwrap();
    // Token 0 via expert 0 (identity gate): silu([1,2]) · [1,2]
    //   = [silu(1)·1, silu(2)·2].
    // Token 1 via expert 1 (swap gate): silu(swap([3,4])) · [3,4]
    //   = silu([4,3]) · [3,4] = [silu(4)·3, silu(3)·4].
    let want = vec![
      silu_ref(1.0) * 1.0,
      silu_ref(2.0) * 2.0,
      silu_ref(4.0) * 3.0,
      silu_ref(3.0) * 4.0,
    ];
    assert_close(&got, &want);
  }

  #[test]
  fn switch_glu_routing_selects_the_indexed_expert() {
    // Same block, but route BOTH tokens through expert 1 (swap). Every token
    // must show the swapped-gate math — proving `indices` actually selects
    // the expert rather than e.g. always using expert 0.
    let glu = SwitchGLU::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchGLU::default_activation(),
    )
    .unwrap();
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 5.0, 6.0], &(2, 2)).unwrap();
    let indices = Array::from_slice::<u32>(&[1, 1], &(2, 1)).unwrap();
    let mut out = glu.forward(&x, &indices).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // Both via expert 1 (swap gate): silu(swap(x)) · x.
    //   token 0: silu([2,1]) · [1,2] = [silu(2)·1, silu(1)·2].
    //   token 1: silu([6,5]) · [5,6] = [silu(6)·5, silu(5)·6].
    let want = vec![
      silu_ref(2.0) * 1.0,
      silu_ref(1.0) * 2.0,
      silu_ref(6.0) * 5.0,
      silu_ref(5.0) * 6.0,
    ];
    assert_close(&got, &want);
  }

  #[test]
  fn switch_glu_sorted_path_matches_hand_trace() {
    // `do_sort` triggers at `indices.size() >= 64`. Route 64 tokens through
    // alternating experts (0, 1, 0, 1, …): the block sorts them by expert id
    // internally and must `scatter_unsort` the result back so each token's
    // output lands at its original position. A wrong unsort would scramble
    // the per-token values and fail the assertion below.
    let glu = SwitchGLU::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchGLU::default_activation(),
    )
    .unwrap();
    let n = 64usize;
    // Token t has features [t, t + 1]; expert id alternates 0 / 1.
    let mut x_data = Vec::with_capacity(n * 2);
    let mut idx_data = Vec::with_capacity(n);
    for t in 0..n {
      x_data.push(t as f32);
      x_data.push(t as f32 + 1.0);
      idx_data.push((t % 2) as u32);
    }
    let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
    let indices = Array::from_slice::<u32>(&idx_data, &(n, 1usize)).unwrap();
    assert!(indices.size() >= 64, "test must exercise the sorted path");
    let mut out = glu.forward(&x, &indices).unwrap();
    assert_eq!(out.shape(), vec![n, 1, 2]);
    let got = out.to_vec::<f32>().unwrap();
    // Reference: per token, silu(gate_e(x)) · x — expert 0 keeps features,
    // expert 1 swaps them.
    let mut want = Vec::with_capacity(n * 2);
    for t in 0..n {
      let (x0, x1) = (t as f32, t as f32 + 1.0);
      if t % 2 == 0 {
        // expert 0 (identity gate)
        want.push(silu_ref(x0) * x0);
        want.push(silu_ref(x1) * x1);
      } else {
        // expert 1 (swap gate): gate sees [x1, x0]
        want.push(silu_ref(x1) * x0);
        want.push(silu_ref(x0) * x1);
      }
    }
    assert_close(&got, &want);
  }

  #[test]
  fn switch_glu_new_rejects_mismatched_projection_shapes() {
    // down_proj must be the [hidden→input] inverse of gate/up [input→hidden].
    // Here gate/up are [2→2] but down is [2→3] (wrong output_dims) — rejected.
    let bad_down_weight =
      Array::from_slice::<f32>(&[0.0f32; 2 * 3 * 2], &(2usize, 3usize, 2usize)).unwrap();
    let err = SwitchGLU::new(
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(bad_down_weight, None).unwrap(),
      SwitchGLU::default_activation(),
    )
    .unwrap_err();
    assert!(
      matches!(err, crate::Error::ShapePairMismatch(_)),
      "expected ShapePairMismatch on mismatched down_proj, got {err:?}"
    );
  }

  #[test]
  fn switch_glu_new_rejects_mismatched_num_experts() {
    // gate/up have E=2; a down_proj with E=3 is rejected.
    let down_e3 =
      Array::from_slice::<f32>(&[1.0f32; 3 * 2 * 2], &(3usize, 2usize, 2usize)).unwrap();
    let err = SwitchGLU::new(
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(down_e3, None).unwrap(),
      SwitchGLU::default_activation(),
    )
    .unwrap_err();
    assert!(
      matches!(err, crate::Error::ShapePairMismatch(_)),
      "expected ShapePairMismatch on mismatched num_experts, got {err:?}"
    );
  }

  #[test]
  fn switch_mlp_hand_traced_two_experts() {
    // fc1 routes through identity (expert 0) / swap (expert 1); fc2 is
    // identity. With a `square` activation the block computes
    // `fc2(square(fc1(x))) = (fc1_e(x))²`.
    let fc1 = SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap();
    let fc2 = SwitchLinear::from_parts(all_identity_weight(), None).unwrap();
    // Explicit closed-form activation so the trace is exact integer arithmetic
    // (the block's wiring is what's under test here; the reference activation
    // formulas are covered in `activations::tests`).
    let square: Activation = Box::new(|a: &Array| a.multiply(a));
    let mlp = SwitchMLP::new(fc1, fc2, square).unwrap();

    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2, 1)).unwrap();
    let mut out = mlp.forward(&x, &indices).unwrap();
    assert_eq!(out.shape(), vec![2, 1, 2]);
    let got = out.to_vec::<f32>().unwrap();
    // Token 0 via expert 0 (identity): square([1,2]) = [1, 4].
    // Token 1 via expert 1 (swap): square(swap([3,4])) = square([4,3]) = [16, 9].
    assert_eq!(got, vec![1.0, 4.0, 16.0, 9.0]);
  }

  #[test]
  fn switch_mlp_default_activation_is_gelu_approx() {
    // `SwitchMLP::default_activation()` must be `gelu_approx` (the python
    // `nn.GELU(approx="precise")` default). With identity fc1/fc2 the block
    // collapses to the activation itself, so the output must equal
    // `activations::gelu_approx(x)`.
    let mlp = SwitchMLP::new(
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchMLP::default_activation(),
    )
    .unwrap();
    let x = Array::from_slice::<f32>(&[-1.0, 0.5, 1.0, 2.0], &(2, 2)).unwrap();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2, 1)).unwrap();
    let mut out = mlp.forward(&x, &indices).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // Reference: gelu_approx applied element-wise (fc1/fc2 are identity).
    let mut reference = super::super::activations::gelu_approx(&x).unwrap();
    let want = reference.to_vec::<f32>().unwrap();
    assert_close(&got, &want);
  }

  #[test]
  fn switch_mlp_forward_preserves_f16_dtype() {
    // `SwitchMLP::default_activation()` is `gelu_approx`, whose scalar
    // constants are dtype-matched (see `activations::scalar_like`). With F16
    // weights and an F16 input the whole block stays F16 — a stray F32
    // activation constant would promote the output to F32. Weights are cast
    // from f32 so no `half`-crate scalars are needed.
    let w16 = all_identity_weight().astype(Dtype::F16).unwrap();
    let mlp = SwitchMLP::new(
      SwitchLinear::from_parts(w16.try_clone().unwrap(), None).unwrap(),
      SwitchLinear::from_parts(w16, None).unwrap(),
      SwitchMLP::default_activation(),
    )
    .unwrap();
    let x = Array::from_slice::<f32>(&[-1.0, 0.5, 1.0, 2.0], &(2, 2))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let indices = Array::from_slice::<u32>(&[0, 1], &(2, 1)).unwrap();
    let out = mlp.forward(&x, &indices).unwrap();
    assert_eq!(
      out.dtype().unwrap(),
      Dtype::F16,
      "SwitchMLP default forward must preserve the F16 input dtype"
    );
  }

  #[test]
  fn switch_mlp_sorted_path_matches_hand_trace() {
    // Same `indices.size() >= 64` sorted-path exercise as the SwitchGLU
    // sibling test, for the un-gated `fc2(square(fc1(x)))` body.
    let square: Activation = Box::new(|a: &Array| a.multiply(a));
    let mlp = SwitchMLP::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      square,
    )
    .unwrap();
    let n = 64usize;
    let mut x_data = Vec::with_capacity(n * 2);
    let mut idx_data = Vec::with_capacity(n);
    for t in 0..n {
      x_data.push(t as f32);
      x_data.push(t as f32 + 1.0);
      idx_data.push((t % 2) as u32);
    }
    let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
    let indices = Array::from_slice::<u32>(&idx_data, &(n, 1usize)).unwrap();
    assert!(indices.size() >= 64, "test must exercise the sorted path");
    let mut out = mlp.forward(&x, &indices).unwrap();
    assert_eq!(out.shape(), vec![n, 1, 2]);
    let got = out.to_vec::<f32>().unwrap();
    // Reference: per token, square(fc1_e(x)) — expert 0 identity, expert 1 swap.
    let mut want = Vec::with_capacity(n * 2);
    for t in 0..n {
      let (x0, x1) = (t as f32, t as f32 + 1.0);
      if t % 2 == 0 {
        want.push(x0 * x0);
        want.push(x1 * x1);
      } else {
        // expert 1 swaps before squaring
        want.push(x1 * x1);
        want.push(x0 * x0);
      }
    }
    assert_close(&got, &want);
  }

  #[test]
  fn switch_mlp_new_rejects_mismatched_projection_shapes() {
    // fc2 must be the [hidden→input] inverse of fc1 [input→hidden]. fc1 is
    // [2→2]; an fc2 of [2→3] (wrong output_dims) is rejected.
    let bad_fc2 =
      Array::from_slice::<f32>(&[0.0f32; 2 * 3 * 2], &(2usize, 3usize, 2usize)).unwrap();
    let err = SwitchMLP::new(
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(bad_fc2, None).unwrap(),
      SwitchMLP::default_activation(),
    )
    .unwrap_err();
    assert!(
      matches!(err, crate::Error::ShapePairMismatch(_)),
      "expected ShapePairMismatch on mismatched fc2, got {err:?}"
    );
  }

  #[test]
  fn gather_sort_then_scatter_unsort_round_trips() {
    // `gather_sort` reorders rows by expert id; `scatter_unsort` (with the
    // returned `inv_order`) must restore the original order exactly. Round-
    // tripping the *index* array through the pair must yield the input.
    // indices: [N=3, k=2] with deliberately-unsorted expert ids.
    let indices = Array::from_slice::<u32>(&[2, 0, 1, 1, 0, 2], &(3, 2)).unwrap();
    // gather_sort's `x` arg is the post-expand_dims input — rank ≥ 3 with
    // trailing (1, 1, D). Build a [3, 2, 1, 1, D=1] x whose value encodes the
    // flattened (token, k) slot so a mis-sort is visible.
    let x = Array::from_slice::<f32>(
      &(0..6).map(|i| i as f32).collect::<Vec<_>>(),
      &(3usize, 2usize, 1usize, 1usize),
    )
    .unwrap();
    let x_expanded = shape::expand_dims_axes(&x, &[-1]).unwrap(); // [3,2,1,1,1]
    let (_x_sorted, mut idx_sorted, inv_order) = gather_sort(&x_expanded, &indices).unwrap();
    // The sorted expert ids must be non-decreasing.
    let sorted_ids = idx_sorted.to_vec::<u32>().unwrap();
    let mut expected_sorted = vec![2u32, 0, 1, 1, 0, 2];
    expected_sorted.sort_unstable();
    assert_eq!(sorted_ids, expected_sorted);
    // scatter_unsort of the sorted ids (reshaped to indices.shape) restores
    // the original [3, 2] index array.
    let idx_as_rows = shape::expand_dims_axes(&idx_sorted, &[-1]).unwrap(); // [6,1]
    let mut restored = scatter_unsort(&idx_as_rows, &inv_order, &[3, 2]).unwrap();
    assert_eq!(restored.shape(), vec![3, 2, 1]);
    let restored_flat = restored.to_vec::<u32>().unwrap();
    assert_eq!(restored_flat, vec![2, 0, 1, 1, 0, 2]);
  }

  // ─── `indices` shape-contract regression (silent-MoE-corruption guard) ───
  //
  // `gather_sort` (the `indices.size() >= 64` sorted path) reads `M =
  // indices.shape[-1]` as the top-k count and maps a sorted flat slot back to
  // a token row via `order // M`. A top-1 `indices` shaped like the batch with
  // NO explicit trailing k axis (`[N]` for x=`[N, D]`, `[B, S]` for
  // x=`[B, S, D]`) would have its last *batch* dim mis-read as `M` — for `[N]`
  // every `order // N` collapses to row 0, so all routed rows silently reuse
  // token 0, yet unsort + squeeze still return a plausible `[N, D]` output.
  // `check_routing_indices` rejects those ambiguous shapes (the reference
  // always carries an explicit k axis); a top-1 caller must pass `[N, 1]`,
  // which sorts correctly (`M == 1`, `order // 1 == order`).

  #[test]
  fn switch_glu_sorted_path_rejects_ambiguous_flat_indices() {
    // 64 routed tokens, `indices` shaped `[N]` (no trailing k axis) — the
    // sorted path is entered (`size >= 64`) but the shape is ambiguous: `N`
    // would be mis-read as the top-k count `M`. Must be a recoverable
    // `RankMismatch`, not silent corruption.
    let glu = SwitchGLU::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchGLU::default_activation(),
    )
    .unwrap();
    let n = 64usize;
    let mut x_data = Vec::with_capacity(n * 2);
    let mut idx_data = Vec::with_capacity(n);
    for t in 0..n {
      x_data.push(t as f32);
      x_data.push(t as f32 + 1.0);
      idx_data.push((t % 2) as u32);
    }
    let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
    // `[N]` — rank-1, same length as x's batch dim, NO explicit k axis.
    let indices = Array::from_slice::<u32>(&idx_data, &(n,)).unwrap();
    assert!(indices.size() >= 64, "test must exercise the sorted path");
    let err = glu.forward(&x, &indices).unwrap_err();
    // x=[N, D] ⇒ x_batch=[N], expected_rank=2; indices=[N] is rank-1 (missing
    // the trailing k axis) ⇒ now categorised as RankMismatch (Codex 2026-05-27)
    // rather than a misleading "expected [N], got [N]" ShapePairMismatch.
    match err {
      crate::Error::RankMismatch(payload) => {
        assert_eq!(payload.actual(), 1, "rank-1 indices ⇒ actual rank 1");
        assert_eq!(payload.actual_shape(), &[64usize]);
      }
      other => panic!("expected RankMismatch on ambiguous [N] indices, got {other:?}"),
    }
  }

  #[test]
  fn switch_glu_sorted_path_rejects_ambiguous_batch_indices() {
    // 64 routed tokens via a 2-D batch x=`[B=8, S=8, D=2]`, `indices` shaped
    // `[B, S]` (no trailing k axis). `S` would be mis-read as `M`; reject.
    let glu = SwitchGLU::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchGLU::default_activation(),
    )
    .unwrap();
    let (b, s) = (8usize, 8usize);
    let mut x_data = Vec::with_capacity(b * s * 2);
    let mut idx_data = Vec::with_capacity(b * s);
    for t in 0..(b * s) {
      x_data.push(t as f32);
      x_data.push(t as f32 + 1.0);
      idx_data.push((t % 2) as u32);
    }
    let x = Array::from_slice::<f32>(&x_data, &(b, s, 2usize)).unwrap();
    // `[B, S]` — rank matches x's batch dims exactly, NO explicit k axis.
    let indices = Array::from_slice::<u32>(&idx_data, &(b, s)).unwrap();
    assert!(indices.size() >= 64, "test must exercise the sorted path");
    let err = glu.forward(&x, &indices).unwrap_err();
    // x=[B, S, D] ⇒ x_batch=[B, S], expected_rank=3; indices=[B, S] is rank-2
    // (missing the trailing k axis) ⇒ RankMismatch.
    match err {
      crate::Error::RankMismatch(payload) => {
        assert_eq!(payload.actual(), 2, "rank-2 indices ⇒ actual rank 2");
        assert_eq!(payload.actual_shape(), &[8usize, 8]);
      }
      other => panic!("expected RankMismatch on ambiguous [B, S] indices, got {other:?}"),
    }
  }

  #[test]
  fn switch_glu_sorted_path_top1_explicit_k_routes_each_token_to_its_expert() {
    // The accepted top-1 form: `indices` shaped `[N, 1]` (explicit k=1). On the
    // sorted path (`size >= 64`) every token must route to ITS OWN selected
    // expert. Tokens 0..32 → expert 0 (identity gate), tokens 32..64 → expert 1
    // (swap gate); every token has a DISTINCT feature pair, so a `[N]`-style
    // mis-route (all rows reuse token 0) would make every output equal token
    // 0's value and fail the per-row assertion below.
    let glu = SwitchGLU::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchGLU::default_activation(),
    )
    .unwrap();
    let n = 64usize;
    let mut x_data = Vec::with_capacity(n * 2);
    let mut idx_data = Vec::with_capacity(n);
    for t in 0..n {
      // Distinct, non-zero per-token features so reusing token 0 is detectable.
      x_data.push(t as f32 + 1.0);
      x_data.push(t as f32 + 2.0);
      // First half → expert 0, second half → expert 1.
      idx_data.push(if t < n / 2 { 0u32 } else { 1u32 });
    }
    let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
    let indices = Array::from_slice::<u32>(&idx_data, &(n, 1usize)).unwrap();
    assert!(indices.size() >= 64, "test must exercise the sorted path");
    let mut out = glu.forward(&x, &indices).unwrap();
    assert_eq!(out.shape(), vec![n, 1, 2]);
    let got = out.to_vec::<f32>().unwrap();
    // Reference: silu(gate_e(x)) · x — expert 0 keeps features, expert 1 swaps.
    let mut want = Vec::with_capacity(n * 2);
    for t in 0..n {
      let (x0, x1) = (t as f32 + 1.0, t as f32 + 2.0);
      if t < n / 2 {
        // expert 0: identity gate
        want.push(silu_ref(x0) * x0);
        want.push(silu_ref(x1) * x1);
      } else {
        // expert 1: swap gate sees [x1, x0]
        want.push(silu_ref(x1) * x0);
        want.push(silu_ref(x0) * x1);
      }
    }
    assert_close(&got, &want);
  }

  #[test]
  fn switch_glu_sorted_path_explicit_2d_batch_k_routes_each_token() {
    // The explicit-`[..batch.., k]` contract with a 2-D batch: x=`[B=8, S=8,
    // D=2]`, `indices`=`[B, S, k=1]` (one extra trailing axis beyond x's
    // batch dims). Accepted, sorted path, each token routed to its own expert.
    let glu = SwitchGLU::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      SwitchGLU::default_activation(),
    )
    .unwrap();
    let (b, s) = (8usize, 8usize);
    let mut x_data = Vec::with_capacity(b * s * 2);
    let mut idx_data = Vec::with_capacity(b * s);
    for t in 0..(b * s) {
      x_data.push(t as f32 + 1.0);
      x_data.push(t as f32 + 2.0);
      idx_data.push((t % 2) as u32);
    }
    let x = Array::from_slice::<f32>(&x_data, &(b, s, 2usize)).unwrap();
    let indices = Array::from_slice::<u32>(&idx_data, &(b, s, 1usize)).unwrap();
    assert!(indices.size() >= 64, "test must exercise the sorted path");
    let mut out = glu.forward(&x, &indices).unwrap();
    // forward returns `[..batch.., k, input_dims]` == `[B, S, 1, 2]`.
    assert_eq!(out.shape(), vec![b, s, 1, 2]);
    let got = out.to_vec::<f32>().unwrap();
    let mut want = Vec::with_capacity(b * s * 2);
    for t in 0..(b * s) {
      let (x0, x1) = (t as f32 + 1.0, t as f32 + 2.0);
      if t % 2 == 0 {
        want.push(silu_ref(x0) * x0);
        want.push(silu_ref(x1) * x1);
      } else {
        want.push(silu_ref(x1) * x0);
        want.push(silu_ref(x0) * x1);
      }
    }
    assert_close(&got, &want);
  }

  #[test]
  fn switch_mlp_sorted_path_rejects_ambiguous_flat_indices() {
    // `SwitchMLP` sibling of `switch_glu_sorted_path_rejects_ambiguous_flat_indices`:
    // a `[N]` top-1 `indices` on the sorted path is rejected, not mis-routed.
    let square: Activation = Box::new(|a: &Array| a.multiply(a));
    let mlp = SwitchMLP::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      square,
    )
    .unwrap();
    let n = 64usize;
    let mut x_data = Vec::with_capacity(n * 2);
    let mut idx_data = Vec::with_capacity(n);
    for t in 0..n {
      x_data.push(t as f32);
      x_data.push(t as f32 + 1.0);
      idx_data.push((t % 2) as u32);
    }
    let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
    let indices = Array::from_slice::<u32>(&idx_data, &(n,)).unwrap();
    assert!(indices.size() >= 64, "test must exercise the sorted path");
    let err = mlp.forward(&x, &indices).unwrap_err();
    // Codex 2026-05-27: missing-k-axis case ⇒ RankMismatch (was a misleading
    // ShapePairMismatch with expected==actual).
    match err {
      crate::Error::RankMismatch(payload) => {
        assert_eq!(payload.actual(), 1, "rank-1 indices ⇒ actual rank 1");
        assert_eq!(payload.actual_shape(), &[64usize]);
      }
      other => panic!("expected RankMismatch on ambiguous [N] indices, got {other:?}"),
    }
  }

  #[test]
  fn switch_mlp_sorted_path_rejects_ambiguous_batch_indices() {
    // `SwitchMLP` sibling: a `[B, S]` top-1 `indices` (no k axis) on a 2-D
    // batch x=`[B, S, D]` is rejected on the sorted path.
    let square: Activation = Box::new(|a: &Array| a.multiply(a));
    let mlp = SwitchMLP::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      square,
    )
    .unwrap();
    let (b, s) = (8usize, 8usize);
    let mut x_data = Vec::with_capacity(b * s * 2);
    let mut idx_data = Vec::with_capacity(b * s);
    for t in 0..(b * s) {
      x_data.push(t as f32);
      x_data.push(t as f32 + 1.0);
      idx_data.push((t % 2) as u32);
    }
    let x = Array::from_slice::<f32>(&x_data, &(b, s, 2usize)).unwrap();
    let indices = Array::from_slice::<u32>(&idx_data, &(b, s)).unwrap();
    assert!(indices.size() >= 64, "test must exercise the sorted path");
    let err = mlp.forward(&x, &indices).unwrap_err();
    // Codex 2026-05-27: missing-k-axis case ⇒ RankMismatch.
    match err {
      crate::Error::RankMismatch(payload) => {
        assert_eq!(payload.actual(), 2, "rank-2 indices ⇒ actual rank 2");
        assert_eq!(payload.actual_shape(), &[8usize, 8]);
      }
      other => panic!("expected RankMismatch on ambiguous [B, S] indices, got {other:?}"),
    }
  }

  #[test]
  fn switch_mlp_sorted_path_top1_explicit_k_routes_each_token_to_its_expert() {
    // `SwitchMLP` sibling of the SwitchGLU `[N, 1]` regression: the accepted
    // explicit-k=1 top-1 form, sorted path, every token routed to ITS OWN
    // expert with distinct per-token features (a `[N]`-style mis-route reusing
    // token 0 would fail the per-row assertion).
    let square: Activation = Box::new(|a: &Array| a.multiply(a));
    let mlp = SwitchMLP::new(
      SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
      SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
      square,
    )
    .unwrap();
    let n = 64usize;
    let mut x_data = Vec::with_capacity(n * 2);
    let mut idx_data = Vec::with_capacity(n);
    for t in 0..n {
      x_data.push(t as f32 + 1.0);
      x_data.push(t as f32 + 2.0);
      idx_data.push(if t < n / 2 { 0u32 } else { 1u32 });
    }
    let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
    let indices = Array::from_slice::<u32>(&idx_data, &(n, 1usize)).unwrap();
    assert!(indices.size() >= 64, "test must exercise the sorted path");
    let mut out = mlp.forward(&x, &indices).unwrap();
    assert_eq!(out.shape(), vec![n, 1, 2]);
    let got = out.to_vec::<f32>().unwrap();
    // Reference: square(fc1_e(x)) — expert 0 identity, expert 1 swap.
    let mut want = Vec::with_capacity(n * 2);
    for t in 0..n {
      let (x0, x1) = (t as f32 + 1.0, t as f32 + 2.0);
      if t < n / 2 {
        want.push(x0 * x0);
        want.push(x1 * x1);
      } else {
        want.push(x1 * x1);
        want.push(x0 * x0);
      }
    }
    assert_close(&got, &want);
  }
}
