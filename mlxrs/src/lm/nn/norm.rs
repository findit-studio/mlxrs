//! Normalization primitives: RMSNorm, LayerNorm, GroupNorm.
//!
//! Ports of the python `mlx.nn` and swift `MLXNN` `Normalization.swift`
//! layers, scoped to what the `lm` (and `vlm`/`audio`) inference stack
//! composes. BatchNorm / InstanceNorm are deliberately deferred — RMSNorm
//! + LayerNorm + GroupNorm cover ~all transformer LM/VLM use.
//!
//! Three configs mirror the references' constructor + call pattern:
//!
//! - [`RMSNorm`] (`weight`, `eps`) — wraps the fused mlx-c
//!   [`mlx_fast_rms_norm`](mlxrs_sys::mlx_fast_rms_norm) primitive. Math:
//!   `x / sqrt(mean(x*x, axis=-1, keepdims=True) + eps) * weight`. Matches
//!   swift `RMSNorm.callAsFunction` / python `RMSNorm.__call__` which both
//!   delegate to `MLXFast.rmsNorm` / `mx.fast.rms_norm`.
//! - [`LayerNorm`] (optional `weight`, optional `bias`, `eps`) — wraps
//!   [`mlx_fast_layer_norm`](mlxrs_sys::mlx_fast_layer_norm). Math:
//!   `(x - mean(x, -1, keepdims)) / sqrt(var(x, -1, keepdims) + eps) *
//!   weight + bias`. Matches the references' `LayerNorm.__call__` (both
//!   delegate to the fused fast kernel). `affine = False` ⇒ `weight =
//!   bias = None`; `bias = False` (with `affine = True`) ⇒ `bias = None`
//!   while `weight = Some(ones)` — the [`LayerNorm::new`] caller decides.
//! - [`GroupNorm`] (`num_groups`, `dims`, `eps`, `affine`,
//!   `pytorch_compatible`) — no fused mlx-c kernel; reproduces the swift
//!   `groupNorm` / `pytorchGroupNorm` paths via [`crate::ops`]:
//!   reshape into per-group tiles, normalize, reshape back, then the
//!   affine `weight * x + bias` (when `affine`). The `pytorch_compatible`
//!   path defers the per-group `(mean, var)` step to the fused
//!   `mlx_fast_layer_norm` (with `weight = bias = None`), exactly as the
//!   python reference does.
//!
//! All three follow the [`crate::lm::nn::rope::Rope`] pattern: a struct
//! that holds the fixed parameters, a `forward(&self, x)` that returns a
//! new lazy [`Array`] (no implicit eval — eval is an explicit `&mut`
//! step on the result).

use crate::{
  array::Array,
  error::{Result, check},
  ops,
  stream::default_stream,
};

// ───────── shared null-handle helper ─────────

/// Produce the NULL-ctx `mlx_array` that mlx-c accepts in any `/* may be
/// null */` slot. Wrapped in the RAII [`Array`] newtype so it is freed on
/// drop, just like a real handle.
///
/// `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
/// per the mlx-c convention; passing it where the C API allows `nullptr`
/// is the documented way to request the no-affine / no-bias path of the
/// fused norm kernels.
#[inline]
fn null_array() -> Array {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL
  // ctx) per the mlx-c convention; wrapped in the RAII newtype so it is
  // freed on drop. The NULL ctx is what mlx-c reads as "absent optional".
  Array(unsafe { mlxrs_sys::mlx_array_new() })
}

/// Forward to [`mlx_fast_layer_norm`](mlxrs_sys::mlx_fast_layer_norm) with
/// `weight` and `bias` mapped through [`null_array`] when `None`. Shared
/// by [`LayerNorm`] and [`GroupNorm`]'s `pytorch_compatible` arm (which
/// reproduces python's `mx.fast.layer_norm(x, weight=None, bias=None,
/// eps=eps)` step).
fn fast_layer_norm(
  x: &Array,
  weight: Option<&Array>,
  bias: Option<&Array>,
  eps: f32,
) -> Result<Array> {
  let null_w = null_array();
  let null_b = null_array();
  let w = weight.unwrap_or(&null_w);
  let b = bias.unwrap_or(&null_b);
  // SAFETY: `mlx_array_new()` yields a fresh empty out handle (NULL ctx);
  // wrapped in the RAII newtype FIRST so an early return / panic frees it.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles, live for
  // the call and not retained past it; `w`/`b` may be the NULL-ctx handle
  // that mlx-c documents as the absent-optional affine arg
  // (`/* may be null */`); the out-param was freshly allocated above and
  // is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_fast_layer_norm(&mut out.0, x.0, w.0, b.0, eps, default_stream())
  })?;
  Ok(out)
}

// ───────── RMSNorm ─────────

/// Root Mean Square normalization config — port of `mlx.nn.RMSNorm`
/// (python `python/mlx/nn/layers/normalization.py`) and swift `RMSNorm`
/// (`MLXNN/Normalization.swift`).
///
/// Computes
/// `out = x / sqrt(mean(x * x, axis=-1, keepdims=True) + eps) * weight`,
/// fused as a single [`mlx_fast_rms_norm`](mlxrs_sys::mlx_fast_rms_norm)
/// kernel (the same primitive both reference impls' `__call__` /
/// `callAsFunction` delegate to). The mean is computed in f32+ for
/// stability per the python docstring.
///
/// `weight` is a per-feature scale (the references initialize it to
/// `ones((dims,))`); RMSNorm has no `bias`. Eps is a small additive
/// constant under the rsqrt for numerical stability (the references
/// default to `1e-5`).
///
/// `forward` returns a new lazy [`Array`] the same shape/dtype as `x`;
/// it does **not** evaluate (eval is an explicit `&mut` step on the
/// result).
#[derive(Debug)]
pub struct RMSNorm {
  /// Per-feature scale of shape `(dims,)` — required (matches the swift
  /// `weight: MLXArray` / python `self.weight` non-optional field).
  pub weight: Array,
  /// Variance floor under the rsqrt (references default `1e-5`).
  pub eps: f32,
}

impl RMSNorm {
  /// Construct an RMSNorm with an explicit `weight` (`(dims,)`) and
  /// `eps`. The reference's `RMSNorm(dims, eps)` constructor allocates a
  /// `ones((dims,))` weight internally — here the caller supplies it
  /// explicitly so a loaded checkpoint can pass the saved tensor in
  /// directly without an intermediate allocation + assignment.
  pub fn new(weight: Array, eps: f32) -> Self {
    Self { weight, eps }
  }

  /// Apply RMSNorm to `x` — forwards to
  /// [`mlx_fast_rms_norm`](mlxrs_sys::mlx_fast_rms_norm), the same fused
  /// kernel mlx-swift's `RMSNorm` / mlx-python's `RMSNorm` delegate to.
  /// Returns a new lazy [`Array`] (no implicit eval).
  pub fn forward(&self, x: &Array) -> Result<Array> {
    // SAFETY: `mlx_array_new()` yields a fresh empty out handle (NULL ctx);
    // wrapped in the RAII newtype FIRST so an early return / panic frees it.
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles, live for
    // the call and not retained past it; `self.weight` is the required
    // per-feature scale (RMSNorm always has a weight; the kernel's "may be
    // null" applies to LayerNorm's bias, not here); the out-param was
    // freshly allocated above and is written by this call; the backend rc
    // is surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_fast_rms_norm(&mut out.0, x.0, self.weight.0, self.eps, default_stream())
    })?;
    Ok(out)
  }
}

// ───────── LayerNorm ─────────

/// Layer Normalization config — port of `mlx.nn.LayerNorm` (python
/// `python/mlx/nn/layers/normalization.py`) and swift `LayerNorm`
/// (`MLXNN/Normalization.swift`).
///
/// Computes
/// `out = (x - mean(x, -1, keepdims)) / sqrt(var(x, -1, keepdims) + eps)
/// * weight + bias`, fused into the single
/// [`mlx_fast_layer_norm`](mlxrs_sys::mlx_fast_layer_norm) kernel both
/// references' `__call__` / `callAsFunction` delegate to.
///
/// `weight` and `bias` are both optional — the swift constructor's
/// `affine` (no weight, no bias) and `bias` (weight but no bias) toggles
/// collapse here to "caller passes `None` for either or both". `None` is
/// forwarded to the kernel as a NULL handle, exactly as mlx-c documents
/// for the `/* may be null */` slot.
///
/// `forward` returns a new lazy [`Array`] the same shape/dtype as `x`;
/// it does **not** evaluate.
#[derive(Debug)]
pub struct LayerNorm {
  /// Optional per-feature affine scale of shape `(dims,)`.
  pub weight: Option<Array>,
  /// Optional per-feature affine shift of shape `(dims,)`. `weight =
  /// None, bias = Some(_)` is rare (mirrors python `affine=False`
  /// dropping both); the kernel still accepts it.
  pub bias: Option<Array>,
  /// Variance floor inside the sqrt (references default `1e-5`).
  pub eps: f32,
}

impl LayerNorm {
  /// Construct a LayerNorm from optional affine parameters and `eps`.
  /// Maps the references' `(affine, bias)` toggles to the caller's
  /// choice: `affine = false` ⇒ both `None`; `affine = true, bias =
  /// false` ⇒ `weight = Some(ones), bias = None`; full affine ⇒ both
  /// `Some`.
  pub fn new(weight: Option<Array>, bias: Option<Array>, eps: f32) -> Self {
    Self { weight, bias, eps }
  }

  /// Apply LayerNorm to `x` — forwards to
  /// [`mlx_fast_layer_norm`](mlxrs_sys::mlx_fast_layer_norm), the same
  /// fused kernel mlx-swift's `LayerNorm` / mlx-python's `LayerNorm`
  /// delegate to. Returns a new lazy [`Array`] (no implicit eval).
  pub fn forward(&self, x: &Array) -> Result<Array> {
    fast_layer_norm(x, self.weight.as_ref(), self.bias.as_ref(), self.eps)
  }
}

// ───────── GroupNorm ─────────

/// Group Normalization config — port of `mlx.nn.GroupNorm` (python
/// `python/mlx/nn/layers/normalization.py`) and swift `GroupNorm`
/// (`MLXNN/Normalization.swift`).
///
/// Splits the feature dimension into [`num_groups`](GroupNorm::num_groups)
/// chunks and normalizes within each group, then applies the optional
/// affine `weight * x + bias`. The feature dim is the last axis;
/// dimensions between the first (batch) and last (features) are treated
/// as spatial.
///
/// Two grouping orders, matching the references:
///
/// - **default** (`pytorch_compatible = false`): reshape `[B, ...rest,
///   dims]` → `[B, -1, num_groups]` (flatten spatial + group axis),
///   reduce over axis 1, reshape back. Each group's `(mean, var)` is
///   computed across the spatial + per-group features.
/// - **pytorch_compatible** (`= true`): reshape `[B, ...rest, dims]` →
///   `[B, -1, num_groups, group_size]`, transpose `(0, 2, 1, 3)` →
///   `[B, num_groups, -1, group_size]`, reshape to `[B, num_groups, -1]`,
///   defer the per-group normalize to
///   [`mlx_fast_layer_norm`](mlxrs_sys::mlx_fast_layer_norm) with
///   `weight = bias = None`, then undo the transpose/reshape. Matches
///   pytorch's grouping order — used when porting checkpoints trained
///   under pytorch GN.
///
/// `forward` returns a new lazy [`Array`] the same shape/dtype as `x`;
/// it does **not** evaluate. No fused mlx-c kernel exists for GroupNorm
/// itself (only LayerNorm/RMSNorm are fused); both paths are pure
/// [`crate::ops`] compositions, faithfully mirroring the python/swift
/// references.
#[derive(Debug)]
pub struct GroupNorm {
  /// Number of feature groups (must divide `dims` evenly; the references
  /// rely on the same `dims / num_groups` integer division and would
  /// produce a malformed reshape otherwise).
  pub num_groups: i32,
  /// Optional per-feature affine scale of shape `(dims,)`. `None` ⇒
  /// `affine=False` in the references.
  pub weight: Option<Array>,
  /// Optional per-feature affine shift of shape `(dims,)`.
  pub bias: Option<Array>,
  /// Variance floor inside the sqrt (references default `1e-5`).
  pub eps: f32,
  /// `true` selects the pytorch-grouping path (see struct docs).
  pub pytorch_compatible: bool,
}

impl GroupNorm {
  /// Construct a GroupNorm matching the swift `GroupNorm(groupCount:,
  /// dimensions:, eps:, affine:, pytorchCompatible:)` init signature.
  ///
  /// `affine = true` materializes the references' `weight = ones((dims,))`
  /// and `bias = zeros((dims,))` (a small fallible alloc), so this is
  /// `Result<Self>`. `affine = false` skips the alloc and leaves both
  /// fields `None`. Pass `dims` even when `affine = false` — it is the
  /// (informational) feature width and is currently used only to allocate
  /// the affine params (matching the references, which also store it on
  /// the module unconditionally).
  pub fn new(
    num_groups: i32,
    dims: i32,
    eps: f32,
    affine: bool,
    pytorch_compatible: bool,
  ) -> Result<Self> {
    let (weight, bias) = if affine {
      let d = usize::try_from(dims).map_err(|_| crate::error::Error::ShapeMismatch {
        message: format!("GroupNorm: dims = {dims} must be non-negative"),
      })?;
      (
        Some(Array::ones::<f32>(&(d,))?),
        Some(Array::zeros::<f32>(&(d,))?),
      )
    } else {
      (None, None)
    };
    Ok(Self {
      num_groups,
      weight,
      bias,
      eps,
      pytorch_compatible,
    })
  }

  /// Apply GroupNorm to `x`. Dispatches to the default or pytorch path
  /// based on [`pytorch_compatible`](Self::pytorch_compatible), then
  /// applies the optional affine `weight * x + bias`. Returns a new lazy
  /// [`Array`] (no implicit eval).
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let normalized = if self.pytorch_compatible {
      self.pytorch_group_norm(x)?
    } else {
      self.group_norm(x)?
    };
    match (&self.weight, &self.bias) {
      (Some(w), Some(b)) => {
        let scaled = ops::arithmetic::multiply(w, &normalized)?;
        ops::arithmetic::add(&scaled, b)
      }
      _ => Ok(normalized),
    }
  }

  /// Validate the input shape against GroupNorm's invariants: rank ≥ 2
  /// (so there is a feature axis distinct from the batch axis),
  /// `num_groups > 0`, and the last (feature) axis evenly divisible by
  /// `num_groups`. Returns the feature dim as `i32` for downstream
  /// `group_size = dims / num_groups` arithmetic.
  ///
  /// Both [`Self::group_norm`] and [`Self::pytorch_group_norm`] call this
  /// up-front. The references don't run an explicit guard (they rely on
  /// the user / framework wiring), but in the safe layer skipping it
  /// produces *silent* activation corruption (e.g. rank-1 `[C]` with
  /// `num_groups=1` would pass as `[C, 1, 1]` and normalize singleton
  /// groups to zero; `[1, 3]` with `num_groups=2` would pass the
  /// element-count divisibility but the 3-wide feature axis isn't
  /// splittable) — surface the misuse as `Err(ShapeMismatch)` instead.
  fn validate_input_shape(&self, orig_shape: &[usize]) -> Result<i32> {
    if orig_shape.len() < 2 {
      return Err(crate::error::Error::ShapeMismatch {
        message: format!(
          "GroupNorm input must have rank >= 2 (at least [batch, dims]), got rank {} shape {orig_shape:?}",
          orig_shape.len()
        ),
      });
    }
    if self.num_groups <= 0 {
      return Err(crate::error::Error::ShapeMismatch {
        message: format!(
          "GroupNorm: num_groups ({}) must be positive",
          self.num_groups
        ),
      });
    }
    let dims = *orig_shape
      .last()
      .expect("rank-≥-2 guarded above ⇒ last() is Some");
    let dims_i32 = i32::try_from(dims).map_err(|_| crate::error::Error::ShapeMismatch {
      message: format!("GroupNorm: feature dim {dims} exceeds i32::MAX"),
    })?;
    if dims_i32 % self.num_groups != 0 {
      return Err(crate::error::Error::ShapeMismatch {
        message: format!(
          "GroupNorm: feature dim ({dims_i32}) must be evenly divisible by num_groups ({})",
          self.num_groups
        ),
      });
    }
    Ok(dims_i32)
  }

  /// Default (`pytorch_compatible = false`) per-group normalize. Mirrors
  /// swift `GroupNorm.groupNorm` / python `GroupNorm._group_norm`:
  /// reshape `[B, ...rest, dims]` → `[B, -1, num_groups]`, mean/var over
  /// axis 1 with `keepdims`, normalize, reshape back to the original
  /// shape.
  ///
  /// The reference's `-1` for the middle axis is `total_elements / (B *
  /// num_groups)`; we compute it numerically so the call passes
  /// [`ops::shape::reshape`]'s non-negative `validate_dims` check.
  fn group_norm(&self, x: &Array) -> Result<Array> {
    let orig_shape = x.shape();
    // Same invariants the pytorch_compatible path checks inline: rank ≥ 2
    // (need a feature dim, not just a batch dim), num_groups > 0, and the
    // last (feature) axis must be splittable evenly by num_groups.
    // Without these the `total / (batch * num_groups)` inference would
    // silently corrupt activations (e.g. rank-1 `[C]` with num_groups=1
    // would pass as `[C, 1, 1]` and normalize singleton groups to zero;
    // `[1, 3]` with num_groups=2 would pass because element count
    // happens to be divisible but the 3-wide feature axis isn't).
    self.validate_input_shape(&orig_shape)?;
    let batch = batch_dim(&orig_shape)?;
    let inferred = inferred_dim(&orig_shape, &[batch, self.num_groups])?;
    let three_d: &[i32] = &[batch, inferred, self.num_groups];
    let reshaped = ops::shape::reshape(x, &three_d)?;

    // Reduce axis 1 (the spatial+per-group axis) with `keepdims=true`,
    // matching `mx.mean(x, axis=1, keepdims=True)` / `mx.var(...)`. The
    // safe layer exposes the plural `*_axes` form; a single-axis call
    // is just `&[1]`.
    let means = ops::reduction::mean_axes(&reshaped, &[1], true)?;
    let var = ops::reduction::var_axes(&reshaped, &[1], true, 0)?;

    // `(x - mean) * rsqrt(var + eps)` — eps added as a Python "weak"
    // scalar so the dtype follows `var` (no f32 promotion of an f16
    // input). Mirrors the reference `var + self.eps` exactly.
    let eps_like = scalar_like(self.eps, &var)?;
    let denom = ops::arithmetic::rsqrt(&ops::arithmetic::add(&var, &eps_like)?)?;
    let centered = ops::arithmetic::subtract(&reshaped, &means)?;
    let normalized = ops::arithmetic::multiply(&centered, &denom)?;

    // Reshape back to the original `[B, ...rest, dims]`.
    let orig_i32 = shape_to_i32(&orig_shape)?;
    let orig_slice: &[i32] = &orig_i32;
    ops::shape::reshape(&normalized, &orig_slice)
  }

  /// `pytorch_compatible = true` per-group normalize. Mirrors swift
  /// `GroupNorm.pytorchGroupNorm` / python
  /// `GroupNorm._pytorch_compatible_group_norm`: reshape into
  /// `[B, -1, num_groups, group_size]`, transpose `(0, 2, 1, 3)`,
  /// flatten to `[B, num_groups, -1]`, defer the per-group `(mean, var)`
  /// to the fused `mlx_fast_layer_norm` (no affine), then undo the
  /// transpose / reshape to the original shape.
  fn pytorch_group_norm(&self, x: &Array) -> Result<Array> {
    let orig_shape = x.shape();
    // Shared rank/num_groups/feature-axis invariants — same bug class as
    // the default path (silent corruption when violated).
    let dims_i32 = self.validate_input_shape(&orig_shape)?;
    let batch = batch_dim(&orig_shape)?;
    let group_size = dims_i32 / self.num_groups;

    // `[B, mid, num_groups, group_size]` where `mid = total / (B *
    // num_groups * group_size)` — the explicit form of the reference's
    // `-1` inferred dim, so `ops::shape::reshape`'s `validate_dims`
    // non-negative check passes.
    let mid = inferred_dim(&orig_shape, &[batch, self.num_groups, group_size])?;
    let four_d: &[i32] = &[batch, mid, self.num_groups, group_size];
    let x = ops::shape::reshape(x, &four_d)?;
    // `transpose(0, 2, 1, 3)`.
    let x = ops::shape::transpose_axes(&x, &[0, 2, 1, 3])?;
    // `reshape(batch, num_groups, mid * group_size)` — the explicit form
    // of the reference's `reshape(batch, num_groups, -1)`. The product
    // is exact (it is the same `total / (B * num_groups)` the reference
    // factors per group across the spatial + per-group features).
    let collapsed =
      mid
        .checked_mul(group_size)
        .ok_or_else(|| crate::error::Error::ShapeMismatch {
          message: "GroupNorm: mid * group_size overflowed i32".into(),
        })?;
    let three_d: &[i32] = &[batch, self.num_groups, collapsed];
    let x = ops::shape::reshape(&x, &three_d)?;

    // Fused per-group normalize: `mx.fast.layer_norm(x, weight=None,
    // bias=None, eps=self.eps)`. Matches python's `_pytorch_compatible_
    // group_norm` exactly.
    let x = fast_layer_norm(&x, None, None, self.eps)?;

    // Undo: `reshape(batch, num_groups, mid, group_size)` then
    // `transpose(0, 2, 1, 3)` then `reshape([batch, *rest, dims])`.
    let four_d_back: &[i32] = &[batch, self.num_groups, mid, group_size];
    let x = ops::shape::reshape(&x, &four_d_back)?;
    let x = ops::shape::transpose_axes(&x, &[0, 2, 1, 3])?;
    let orig_i32 = shape_to_i32(&orig_shape)?;
    let orig_slice: &[i32] = &orig_i32;
    ops::shape::reshape(&x, &orig_slice)
  }
}

// ───────── small shared helpers ─────────

/// Convert `[batch, ...rest, dims]` (a `Vec<usize>` from
/// [`Array::shape`]) to a `Vec<i32>` for re-feeding through
/// [`ops::shape::reshape`]'s `&[i32]` form. Errors on a `usize` past
/// `i32::MAX` (the same check `IntoShape` does for `&[usize]`).
fn shape_to_i32(shape: &[usize]) -> Result<Vec<i32>> {
  shape
    .iter()
    .map(|&d| {
      i32::try_from(d).map_err(|_| crate::error::Error::ShapeMismatch {
        message: format!("dim {d} exceeds i32::MAX"),
      })
    })
    .collect()
}

/// Pull the leading batch dim as `i32`, erroring on rank 0 or on a dim
/// past `i32::MAX`.
fn batch_dim(shape: &[usize]) -> Result<i32> {
  let b = *shape
    .first()
    .ok_or_else(|| crate::error::Error::ShapeMismatch {
      message: "GroupNorm input must have at least one dim (the batch axis)".into(),
    })?;
  i32::try_from(b).map_err(|_| crate::error::Error::ShapeMismatch {
    message: format!("GroupNorm: batch dim {b} exceeds i32::MAX"),
  })
}

/// Compute the `-1`-replacement dim for a reshape: the residual element
/// count after dividing `total = product(shape)` by `product(known_dims)`.
/// The references use mlx's `-1` sentinel; the safe layer keeps the shape
/// numeric so the resulting reshape is trivially auditable (and passes
/// [`crate::shape::validate_dims`]'s non-negative check).
fn inferred_dim(shape: &[usize], known_dims: &[i32]) -> Result<i32> {
  // Checked product (same `try_fold` + `checked_mul` pattern as
  // `Array::from_slice` in `array/construction.rs`): unchecked
  // `.product()` would silently wrap on a large / broadcast shape,
  // yielding the wrong inferred dim → either a downstream reshape
  // boundary failure or, worse, a passing reshape with a corrupted layout.
  let total: usize = shape
    .iter()
    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
    .ok_or_else(|| crate::error::Error::ShapeMismatch {
      message: format!("GroupNorm: shape product overflows usize for shape {shape:?}"),
    })?;
  let mut divisor: usize = 1;
  for &d in known_dims {
    let du = usize::try_from(d).map_err(|_| crate::error::Error::ShapeMismatch {
      message: format!("GroupNorm: known reshape dim {d} must be non-negative"),
    })?;
    divisor = divisor
      .checked_mul(du)
      .ok_or_else(|| crate::error::Error::ShapeMismatch {
        message: "GroupNorm: reshape divisor overflowed usize".into(),
      })?;
  }
  if divisor == 0 || !total.is_multiple_of(divisor) {
    return Err(crate::error::Error::ShapeMismatch {
      message: format!(
        "GroupNorm: cannot reshape {total} elements into a layout requiring {divisor} per inferred slot"
      ),
    });
  }
  i32::try_from(total / divisor).map_err(|_| crate::error::Error::ShapeMismatch {
    message: format!(
      "GroupNorm: inferred dim {} exceeds i32::MAX",
      total / divisor
    ),
  })
}

/// Build a 1-element f32 scalar Array of `value`, cast to `like.dtype()`.
/// Twin of [`crate::lm::sample`]'s private `scalar_like` — kept local so
/// this module is self-contained and `lm::sample` is not exposed.
fn scalar_like(value: f32, like: &Array) -> Result<Array> {
  // `Array::full` runs the fallible `mlx_array_new_float32` ctor BEFORE
  // its `mlx_full` call (whose `default_stream()` arg installs the error
  // handler), so without the eager install here that first ctor could
  // reach mlx-c with no handler installed → its default `printf +
  // exit(-1)` instead of a recoverable `Err`. Same defense-in-depth as
  // `lm::sample::scalar_like`.
  crate::error::ensure_handler_installed();
  ops::misc::astype(&Array::full::<f32>(&(1,), value)?, like.dtype()?)
}

// ───────── tests ─────────

#[cfg(test)]
mod tests {
  use super::*;

  /// `1e-4` matches the existing `lm::nn::rope` golden tolerance for the
  /// f32-vs-f64 fused-kernel rounding gap, with extra slack because
  /// LayerNorm/GroupNorm fold a sqrt/rsqrt + a division on top of the
  /// mean/var reduce.
  const TOL: f32 = 1e-4;

  fn vclose(got: &[f32], want: &[f32]) -> bool {
    if got.len() != want.len() {
      return false;
    }
    got
      .iter()
      .zip(want)
      .all(|(g, w)| (g - w).abs() <= TOL && g.is_finite() && w.is_finite())
  }

  // ─── RMSNorm ───

  #[test]
  fn rms_norm_hand_traced() {
    // RMSNorm of [1, 2, 3] with weight=[1, 1, 1], eps=1e-6:
    //   rms = sqrt(mean(x*x) + eps) = sqrt((1+4+9)/3 + eps) ≈ sqrt(14/3)
    //   out = x / rms * w = [1, 2, 3] / sqrt(14/3)
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let w = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(3,)).unwrap();
    let rn = RMSNorm::new(w, 1e-6);
    let mut y = rn.forward(&x).unwrap();
    let rms = (14.0_f32 / 3.0).sqrt();
    assert!(vclose(
      &y.to_vec::<f32>().unwrap(),
      &[1.0 / rms, 2.0 / rms, 3.0 / rms]
    ));
  }

  #[test]
  fn rms_norm_zero_input_is_finite() {
    // Zero input + eps in the rsqrt ⇒ output is finite (not NaN/Inf).
    let x = Array::from_slice::<f32>(&[0.0, 0.0, 0.0, 0.0], &(1, 4)).unwrap();
    let w = Array::ones::<f32>(&(4,)).unwrap();
    let rn = RMSNorm::new(w, 1e-5);
    let mut y = rn.forward(&x).unwrap();
    let v = y.to_vec::<f32>().unwrap();
    assert!(
      v.iter().all(|x| x.is_finite()),
      "expected finite, got {v:?}"
    );
  }

  #[test]
  fn rms_norm_preserves_rank3_shape() {
    // Rank-3 [2, 3, 4] in → same shape out.
    let x =
      Array::from_slice::<f32>(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &(2, 3, 4)).unwrap();
    let w = Array::ones::<f32>(&(4,)).unwrap();
    let rn = RMSNorm::new(w, 1e-5);
    let y = rn.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 3, 4]);
  }

  #[test]
  fn rms_norm_matches_manual_fallback() {
    // The fused `mlx_fast_rms_norm` kernel must match the manual
    // `x / sqrt(mean(x*x, -1, keepdims) + eps) * weight` composition.
    let x = Array::from_slice::<f32>(&[0.5, -1.5, 2.0, 3.0, 4.0, 5.0], &(1, 2, 3)).unwrap();
    let w = Array::from_slice::<f32>(&[0.5, 1.0, 1.5], &(3,)).unwrap();
    let eps = 1e-5_f32;

    let mut via_kernel = RMSNorm::new(w.try_clone().unwrap(), eps)
      .forward(&x)
      .unwrap();

    // Manual fallback path: `x / sqrt(mean(x*x, -1, keepdims) + eps) * weight`.
    let xx = ops::arithmetic::square(&x).unwrap();
    let m = ops::reduction::mean_axes(&xx, &[-1], true).unwrap();
    let eps_arr = scalar_like(eps, &m).unwrap();
    let denom = ops::arithmetic::rsqrt(&ops::arithmetic::add(&m, &eps_arr).unwrap()).unwrap();
    let scaled = ops::arithmetic::multiply(&x, &denom).unwrap();
    let mut via_manual = ops::arithmetic::multiply(&scaled, &w).unwrap();

    assert!(vclose(
      &via_kernel.to_vec::<f32>().unwrap(),
      &via_manual.to_vec::<f32>().unwrap()
    ));
  }

  // ─── LayerNorm ───

  #[test]
  fn layer_norm_hand_traced() {
    // LayerNorm of [1, 2, 3, 4] with no affine, eps=1e-5:
    //   mean = 2.5, var = ((1.5)^2 + (0.5)^2 + (0.5)^2 + (1.5)^2)/4 = 1.25
    //   out = (x - 2.5) / sqrt(1.25 + 1e-5)
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
    let ln = LayerNorm::new(None, None, 1e-5);
    let mut y = ln.forward(&x).unwrap();
    let denom = (1.25_f32 + 1e-5).sqrt();
    let want = [
      (1.0 - 2.5) / denom,
      (2.0 - 2.5) / denom,
      (3.0 - 2.5) / denom,
      (4.0 - 2.5) / denom,
    ];
    assert!(vclose(&y.to_vec::<f32>().unwrap(), &want));
  }

  #[test]
  fn layer_norm_zero_input_is_finite() {
    // Zero input ⇒ mean=0, var=0; eps prevents the div-by-zero. Output
    // is the all-zero array (numerator is 0 too), which is finite.
    let x = Array::from_slice::<f32>(&[0.0; 6], &(1, 6)).unwrap();
    let ln = LayerNorm::new(None, None, 1e-5);
    let mut y = ln.forward(&x).unwrap();
    let v = y.to_vec::<f32>().unwrap();
    assert!(
      v.iter().all(|x| x.is_finite()),
      "expected finite, got {v:?}"
    );
  }

  #[test]
  fn layer_norm_preserves_rank3_shape() {
    let x =
      Array::from_slice::<f32>(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &(2, 3, 4)).unwrap();
    let ln = LayerNorm::new(None, None, 1e-5);
    let y = ln.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 3, 4]);
  }

  #[test]
  fn layer_norm_affine_applies_weight_and_bias() {
    // LayerNorm with full affine: weight=[2,2,2,2], bias=[1,1,1,1]
    // should produce 2*unaffine + 1 element-wise.
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
    let w = Array::full::<f32>(&(4,), 2.0).unwrap();
    let b = Array::ones::<f32>(&(4,)).unwrap();
    let plain = LayerNorm::new(None, None, 1e-5);
    let affine = LayerNorm::new(Some(w), Some(b), 1e-5);
    let mut p = plain.forward(&x).unwrap();
    let mut a = affine.forward(&x).unwrap();
    let pv = p.to_vec::<f32>().unwrap();
    let av = a.to_vec::<f32>().unwrap();
    let want: Vec<f32> = pv.iter().map(|v| 2.0 * v + 1.0).collect();
    assert!(vclose(&av, &want));
  }

  #[test]
  fn layer_norm_matches_manual_fallback() {
    // The fused `mlx_fast_layer_norm` (no affine) must match
    // `(x - mean) / sqrt(var + eps)` over the last axis.
    let x = Array::from_slice::<f32>(&[0.5, -1.5, 2.0, 3.0, 4.0, 5.0], &(1, 2, 3)).unwrap();
    let eps = 1e-5_f32;
    let mut via_kernel = LayerNorm::new(None, None, eps).forward(&x).unwrap();

    let m = ops::reduction::mean_axes(&x, &[-1], true).unwrap();
    let v = ops::reduction::var_axes(&x, &[-1], true, 0).unwrap();
    let eps_arr = scalar_like(eps, &v).unwrap();
    let denom = ops::arithmetic::rsqrt(&ops::arithmetic::add(&v, &eps_arr).unwrap()).unwrap();
    let centered = ops::arithmetic::subtract(&x, &m).unwrap();
    let mut via_manual = ops::arithmetic::multiply(&centered, &denom).unwrap();

    assert!(vclose(
      &via_kernel.to_vec::<f32>().unwrap(),
      &via_manual.to_vec::<f32>().unwrap()
    ));
  }

  // ─── GroupNorm ───

  #[test]
  fn group_norm_hand_traced_one_group_matches_layer_norm() {
    // GroupNorm with num_groups=1 is equivalent to per-token LayerNorm
    // across the (spatial + dims) features. For a [1, dims] input, that
    // is exactly LayerNorm — the hand-traced reference.
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
    let gn = GroupNorm::new(1, 4, 1e-5, false, false).unwrap();
    let mut y = gn.forward(&x).unwrap();
    // Reference: rank-2 [1, 4] with 1 group ⇒ mean=2.5, var=1.25.
    let denom = (1.25_f32 + 1e-5).sqrt();
    let want = [
      (1.0 - 2.5) / denom,
      (2.0 - 2.5) / denom,
      (3.0 - 2.5) / denom,
      (4.0 - 2.5) / denom,
    ];
    assert!(vclose(&y.to_vec::<f32>().unwrap(), &want));
  }

  #[test]
  fn group_norm_zero_input_is_finite() {
    // Zero rank-3 input ⇒ output is finite (eps in the rsqrt).
    let x = Array::from_slice::<f32>(&[0.0; 12], &(1, 3, 4)).unwrap();
    let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    let mut y = gn.forward(&x).unwrap();
    let v = y.to_vec::<f32>().unwrap();
    assert!(
      v.iter().all(|x| x.is_finite()),
      "expected finite, got {v:?}"
    );
  }

  #[test]
  fn group_norm_preserves_rank4_shape() {
    // Rank-4 [B=2, H=3, W=3, C=4] in → same shape out.
    let n = 2 * 3 * 3 * 4;
    let x = Array::from_slice::<f32>(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), &(2, 3, 3, 4))
      .unwrap();
    let gn = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 3, 3, 4]);
  }

  #[test]
  fn group_norm_pytorch_compat_preserves_shape() {
    // The pytorch_compatible path follows the same input/output shape
    // contract, and must produce a finite result.
    let n = 2 * 3 * 3 * 4;
    let x = Array::from_slice::<f32>(
      &(0..n).map(|i| i as f32 + 1.0).collect::<Vec<_>>(),
      &(2, 3, 3, 4),
    )
    .unwrap();
    let gn = GroupNorm::new(2, 4, 1e-5, false, true).unwrap();
    let mut y = gn.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 3, 3, 4]);
    let v = y.to_vec::<f32>().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
  }

  #[test]
  fn group_norm_affine_applies_weight_and_bias() {
    // affine=true with weight=2*ones, bias=ones should yield 2*unaffine + 1.
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
    let plain = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    let mut affine = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
    // Replace the default ones/zeros with weight=2, bias=1.
    affine.weight = Some(Array::full::<f32>(&(4,), 2.0).unwrap());
    affine.bias = Some(Array::ones::<f32>(&(4,)).unwrap());
    let mut p = plain.forward(&x).unwrap();
    let mut a = affine.forward(&x).unwrap();
    let pv = p.to_vec::<f32>().unwrap();
    let av = a.to_vec::<f32>().unwrap();
    let want: Vec<f32> = pv.iter().map(|v| 2.0 * v + 1.0).collect();
    assert!(vclose(&av, &want));
  }

  #[test]
  fn group_norm_default_constructor_no_affine() {
    // affine=false ⇒ weight/bias are None (no allocation).
    let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    assert!(gn.weight.is_none());
    assert!(gn.bias.is_none());
    assert!(!gn.pytorch_compatible);
    assert_eq!(gn.num_groups, 2);
  }

  #[test]
  fn group_norm_default_constructor_affine_allocates() {
    let gn = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
    assert!(gn.weight.is_some());
    assert!(gn.bias.is_some());
    let mut w = gn.weight.unwrap();
    let mut b = gn.bias.unwrap();
    assert_eq!(w.shape(), vec![4]);
    assert_eq!(b.shape(), vec![4]);
    assert_eq!(w.to_vec::<f32>().unwrap(), vec![1.0; 4]);
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![0.0; 4]);
  }

  // ─── GroupNorm shape-invariant regressions (Codex review) ───

  /// Rank-1 `[C]` with `num_groups=1` used to silently corrupt
  /// activations (passed as `[C, 1, 1]` and normalized singleton groups
  /// to zero); now an explicit `Err(ShapeMismatch)`.
  #[test]
  fn group_norm_rank1_input_errors() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(4,)).unwrap();
    let gn = GroupNorm::new(1, 4, 1e-5, false, false).unwrap();
    let err = gn.forward(&x).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(message.contains("rank"), "unexpected message: {message}");
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }

  /// Rank-2 `[1, 3]` with `num_groups=2` used to silently pass (element
  /// count is divisible — 6/2 = 3 — but the 3-wide feature axis isn't
  /// splittable). Now an explicit `Err(ShapeMismatch)`.
  #[test]
  fn group_norm_feature_dim_not_divisible_errors() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let gn = GroupNorm::new(2, 3, 1e-5, false, false).unwrap();
    let err = gn.forward(&x).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("divisible"),
          "unexpected message: {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }

  /// Same invariants must hold on the `pytorch_compatible` path: rank-1
  /// input rejected.
  #[test]
  fn group_norm_pytorch_compat_rank1_input_errors() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(4,)).unwrap();
    let gn = GroupNorm::new(1, 4, 1e-5, false, true).unwrap();
    let err = gn.forward(&x).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(message.contains("rank"), "unexpected message: {message}");
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }

  /// Same invariants on the `pytorch_compatible` path: non-divisible
  /// feature dim rejected.
  #[test]
  fn group_norm_pytorch_compat_feature_dim_not_divisible_errors() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let gn = GroupNorm::new(2, 3, 1e-5, false, true).unwrap();
    let err = gn.forward(&x).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("divisible"),
          "unexpected message: {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }

  /// Regression: the valid rank-2 case (`[1, 4]` with `num_groups=2`)
  /// must continue to work after the new validation guards.
  #[test]
  fn group_norm_valid_rank2_still_works() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
    let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    let mut y = gn.forward(&x).unwrap();
    let v = y.to_vec::<f32>().unwrap();
    assert_eq!(v.len(), 4);
    assert!(v.iter().all(|x| x.is_finite()));
  }

  // ─── inferred_dim overflow regression (Codex review) ───

  /// `inferred_dim` used to compute `total = shape.iter().product()`
  /// unchecked, so a shape whose `usize` product wraps would yield the
  /// wrong inferred dim (and either a reshape boundary failure or a
  /// passing reshape on a corrupted layout). Now an `Err(ShapeMismatch)`
  /// before we ever reach the divisibility check.
  ///
  /// `usize::MAX` on its own already wraps on the `* 2` step.
  #[test]
  fn inferred_dim_overflow_errors() {
    let shape: [usize; 2] = [usize::MAX, 2];
    let err = inferred_dim(&shape, &[1, 1]).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("overflow"),
          "unexpected message: {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }
}
