//! Mixture-of-Experts (MoE) **Switch** primitives: [`SwitchLinear`] and
//! [`QuantizedSwitchLinear`].
//!
//! A 1:1 port of mlx-lm's `SwitchLinear` / `QuantizedSwitchLinear`
//! (`mlx-lm/mlx_lm/models/switch_layers.py:93` and `:27`). Each holds a
//! per-expert weight stack of shape `[num_experts, output_dims, input_dims]`
//! plus an optional per-expert bias of shape `[num_experts, output_dims]`, and
//! routes each input token through one (or `k`) experts indexed by a
//! caller-supplied `indices` array. The forward pass collapses to a single
//! fused mlx-c kernel — [`ops::linalg_basic::gather_mm`](crate::ops::linalg_basic::gather_mm)
//! for the dense layer, [`ops::quantized::gather_qmm`](crate::ops::quantized::gather_qmm)
//! for the quantized one — instead of `take`+`matmul`.
//!
//! # Scope
//!
//! Only the two `SwitchLinear` shapes are ported here. The higher-level
//! `SwitchMLP` / `SwitchGLU` MoE blocks (same file, `:202` / `:160`) compose
//! `SwitchLinear` with an activation (`gelu` / `swiglu`) and the
//! `_gather_sort` / `_scatter_unsort` rebatching helpers. Porting them
//! requires the activation surface (`gelu` / `silu` / `swiglu`) under
//! [`crate::lm::nn`], which doesn't exist yet — they are deliberate
//! follow-ups, not in this PR's scope (mirrors the rope-only scope of #N1).
//!
//! mlx-swift does not expose a `SwitchLinear` layer at all (the `MLXNN`
//! surface has no MoE module — only the underlying `gatherMM` /
//! `gatherQuantizedMM` ops in `Source/MLX/Ops.swift`). This port follows the
//! python reference, which is the canonical home of the layer.

use crate::{
  array::Array,
  dtype::Dtype,
  error::Result,
  ops::{indexing, linalg_basic, quantized, shape},
};

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
/// [`weight`](Self::weight) / [`bias`](Self::bias) read-only accessors —
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
  /// PRIVATE — read via [`Self::weight`] — so the constructor's rank-3
  /// `[E, O, I]` validation can't be bypassed by struct-literal construction
  /// or `&mut` mutation.
  weight: Array,
  /// Optional per-expert bias of shape `[num_experts, output_dims]`. `None`
  /// matches python `SwitchLinear(..., bias=False)`. PRIVATE — read via
  /// [`Self::bias`] — for the same reason as [`Self::weight`]: a malformed
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
  /// [`Error::ShapeMismatch`](crate::Error::ShapeMismatch) rather than left to
  /// fail deep inside the FFI on first `apply`. Does not evaluate the arrays
  /// (lazy; only `shape()` metadata is read).
  pub fn from_parts(weight: Array, bias: Option<Array>) -> Result<Self> {
    let w_shape = weight.shape();
    if w_shape.len() != 3 {
      return Err(crate::Error::ShapeMismatch {
        message: format!(
          "SwitchLinear::from_parts: weight must be 3-D [num_experts, output_dims, \
           input_dims], got {w_shape:?}"
        ),
      });
    }
    if let Some(b) = &bias {
      let b_shape = b.shape();
      if b_shape.len() != 2 || b_shape[0] != w_shape[0] || b_shape[1] != w_shape[1] {
        return Err(crate::Error::ShapeMismatch {
          message: format!(
            "SwitchLinear::from_parts: bias must be [num_experts={}, output_dims={}], \
             got {b_shape:?}",
            w_shape[0], w_shape[1]
          ),
        });
      }
    }
    Ok(Self { weight, bias })
  }

  /// Read-only accessor for the per-expert weight stack (`[num_experts,
  /// output_dims, input_dims]`).
  ///
  /// The field is private specifically so the constructor's rank-3 shape
  /// validation is the only construction path; see the struct doc for the
  /// invariant rationale. Lazy — does not evaluate.
  pub fn weight(&self) -> &Array {
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
/// All fields are PRIVATE — exposed via the [`weight`](Self::weight) /
/// [`scales`](Self::scales) / [`quant_biases`](Self::quant_biases) /
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
  /// `[E, O, I]` weight). PRIVATE — read via [`Self::weight`] — so the
  /// constructor's rank-3 validation is the only construction path.
  weight: Array,
  /// Per-group scales (paired with `weight` — same `[E, O, n_groups]` layout
  /// the dense `quantize` op produces). PRIVATE — read via [`Self::scales`] —
  /// for symmetry with `weight`: the `(weight, scales, quant_biases)` triple
  /// is internally consistent only when produced together by
  /// [`ops::quantized::quantize`](crate::ops::quantized::quantize), and
  /// mutating one without the others corrupts dequant inside `gather_qmm`.
  scales: Array,
  /// Per-group biases (the affine-mode addend; `None` for the bias-less float
  /// schemes `mxfp4`/`mxfp8`/`nvfp4`, mirroring `ops::quantized::quantize`'s
  /// `Option<Array>` return). PRIVATE — read via [`Self::quant_biases`] —
  /// for the same triple-consistency reason as [`Self::scales`].
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
  /// Shape mismatches surface as
  /// [`Error::ShapeMismatch`](crate::Error::ShapeMismatch); mode-arity /
  /// unknown-mode / zero-param failures surface as
  /// [`Error::Backend`](crate::Error::Backend). Does not evaluate (lazy;
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
      return Err(crate::Error::ShapeMismatch {
        message: format!(
          "QuantizedSwitchLinear::from_parts: weight must be 3-D [num_experts, output_dims, \
           packed_input_dims], got {w_shape:?}"
        ),
      });
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
      return Err(crate::Error::Backend {
        message: format!(
          "QuantizedSwitchLinear::from_parts: weight must be packed `uint32` (the \
           mlx-quantized-weight dtype — `gather_qmm` rejects non-`uint32` quantized \
           weights), got {w_dtype:?}"
        ),
      });
    }

    let e = w_shape[0];
    let o = w_shape[1];

    // `scales` structural invariants: rank == weight rank (3); leading two
    // dims (E, O) must match weight. The trailing per-group count is
    // validated by mlx-c against `group_size`.
    let s_shape = scales.shape();
    if s_shape.len() != w_shape.len() {
      return Err(crate::Error::ShapeMismatch {
        message: format!(
          "QuantizedSwitchLinear::from_parts: scales rank ({}) does not match weight rank ({}) \
           — mlx `quantize` preserves the leading shape across the (weight, scales, biases) \
           triple (`mlx/ops.cpp:4789-4798`); got scales {s_shape:?}, weight {w_shape:?}",
          s_shape.len(),
          w_shape.len()
        ),
      });
    }
    if s_shape[0] != e || s_shape[1] != o {
      return Err(crate::Error::ShapeMismatch {
        message: format!(
          "QuantizedSwitchLinear::from_parts: scales leading dims must match weight \
           [num_experts={e}, output_dims={o}, ..], got {s_shape:?}"
        ),
      });
    }

    // `quant_biases`, when present, shares the per-group `[E, O, n_groups]`
    // layout with `scales` (`affine_quantize` produces both with the same
    // shape, `mlx/ops.cpp:4793-4798`).
    if let Some(qb) = &quant_biases {
      let qb_shape = qb.shape();
      if qb_shape != s_shape {
        return Err(crate::Error::ShapeMismatch {
          message: format!(
            "QuantizedSwitchLinear::from_parts: quant_biases shape must match scales \
             (mlx `affine_quantize` writes them with identical `[E, O, n_groups]` shape, \
             `mlx/ops.cpp:4793-4798`); got quant_biases {qb_shape:?}, scales {s_shape:?}"
          ),
        });
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
        return Err(crate::Error::Backend {
          message: "QuantizedSwitchLinear::from_parts: `affine` mode requires `quant_biases` \
                    (mlx `affine_quantize` always writes {w_q, scales, biases}, \
                    `mlx/ops.cpp:4793-4798`); got None"
            .to_string(),
        });
      }
      ("mxfp4" | "mxfp8" | "nvfp4", Some(_)) => {
        return Err(crate::Error::Backend {
          message: format!(
            "QuantizedSwitchLinear::from_parts: `{mode}` mode is scale-only \
             (mlx `fp_quantize` writes {{w_q, scales}} with no biases, \
             `mlx/ops.cpp:4890,4898-4904`); got a stale `quant_biases`"
          ),
        });
      }
      ("affine", Some(_)) | ("mxfp4" | "mxfp8" | "nvfp4", None) => {
        // Expected layouts — fall through to the remaining checks.
      }
      (other, _) => {
        return Err(crate::Error::Backend {
          message: format!(
            "QuantizedSwitchLinear::from_parts: unknown mode {other:?}; allowed: \
             \"affine\", \"mxfp4\", \"mxfp8\", \"nvfp4\""
          ),
        });
      }
    }

    // Basic non-zero sanity on `bits` / `group_size`. Per-mode value tables
    // (`bits ∈ {2,3,4,5,6,8}` for affine — `mlx/ops.cpp:4745-4750`;
    // `mxfp4`/`nvfp4` requiring specific `(group_size, bits)` pairs —
    // `mlx/ops.cpp:4808-4823`) are DEFERRED to mlx-c per the faithful-port
    // discipline; checking them here would duplicate
    // `validate_quantized_input` and drift from upstream.
    if bits <= 0 {
      return Err(crate::Error::Backend {
        message: format!(
          "QuantizedSwitchLinear::from_parts: bits must be > 0, got {bits} (per-mode value \
           tables are validated by mlx-c)"
        ),
      });
    }
    if group_size <= 0 {
      return Err(crate::Error::Backend {
        message: format!(
          "QuantizedSwitchLinear::from_parts: group_size must be > 0, got {group_size} \
           (per-mode value tables are validated by mlx-c)"
        ),
      });
    }

    if let Some(b) = &bias {
      let b_shape = b.shape();
      if b_shape.len() != 2 || b_shape[0] != e || b_shape[1] != o {
        return Err(crate::Error::ShapeMismatch {
          message: format!(
            "QuantizedSwitchLinear::from_parts: bias must be [num_experts={e}, \
             output_dims={o}], got {b_shape:?}"
          ),
        });
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
  /// invariant rationale. Lazy — does not evaluate.
  pub fn weight(&self) -> &Array {
    &self.weight
  }

  /// Read-only accessor for the per-group quantization scales.
  ///
  /// The field is private to preserve the `(weight, scales, quant_biases)`
  /// triple's internal consistency — they're only well-formed when produced
  /// together by [`ops::quantized::quantize`](crate::ops::quantized::quantize).
  /// Lazy — does not evaluate.
  pub fn scales(&self) -> &Array {
    &self.scales
  }

  /// Read-only accessor for the optional per-group quantization biases
  /// (`None` for the bias-less float schemes `mxfp4`/`mxfp8`/`nvfp4`).
  ///
  /// The field is private for the same triple-consistency reason as
  /// [`Self::scales`]. Lazy — does not evaluate.
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
  /// produced [`Self::weight`] / [`Self::scales`]).
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
    assert!(matches!(err, crate::Error::ShapeMismatch { .. }));
  }

  #[test]
  fn switch_linear_from_parts_rejects_mismatched_bias() {
    let weight = hand_traced_weight(); // [2, 3, 4]
    // Bad bias: [3, 3] (wrong E).
    let bad_bias =
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &(3, 3)).unwrap();
    let err = SwitchLinear::from_parts(weight, Some(bad_bias)).unwrap_err();
    assert!(matches!(err, crate::Error::ShapeMismatch { .. }));
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
    assert!(matches!(err, crate::Error::ShapeMismatch { .. }));
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
      matches!(err, crate::Error::ShapeMismatch { .. }),
      "expected ShapeMismatch on scales leading dims, got {err:?}"
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
      crate::Error::Backend { message } => {
        assert!(
          message.contains("F32"),
          "Backend error should name the offending dtype, got: {message}"
        );
      }
      other => panic!("expected Backend naming the dtype, got {other:?}"),
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
      matches!(err, crate::Error::ShapeMismatch { .. }),
      "expected ShapeMismatch on quant_biases shape, got {err:?}"
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
      matches!(err, crate::Error::Backend { .. }),
      "expected Backend on affine-missing-quant_biases, got {err:?}"
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
      matches!(err, crate::Error::Backend { .. }),
      "expected Backend on mxfp4-with-stale-quant_biases, got {err:?}"
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
      matches!(err, crate::Error::Backend { .. }),
      "expected Backend on unknown mode, got {err:?}"
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
      matches!(err_bits, crate::Error::Backend { .. }),
      "expected Backend on bits=0, got {err_bits:?}"
    );

    let err_gs =
      QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, None, 0, 4, "affine").unwrap_err();
    assert!(
      matches!(err_gs, crate::Error::Backend { .. }),
      "expected Backend on group_size=0, got {err_gs:?}"
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
    assert_eq!(layer.weight().shape()[0], 2); // E=2
    assert_eq!(layer.weight().shape()[1], 4); // O=4
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
    assert_eq!(layer.weight().shape(), vec![2, 3, 4]);
    assert!(layer.bias().is_none());

    let bias = Array::from_slice::<f32>(
      &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], // [E=2, O=3]
      &(2, 3),
    )
    .unwrap();
    let layer_with_bias = SwitchLinear::from_parts(hand_traced_weight(), Some(bias)).unwrap();
    assert_eq!(layer_with_bias.weight().shape(), vec![2, 3, 4]);
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
    assert_eq!(q_layer.weight().shape()[0], 2); // E=2
    assert_eq!(q_layer.weight().shape()[1], 4); // O=4
    assert_eq!(q_layer.scales().shape()[0], 2); // E
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
}
