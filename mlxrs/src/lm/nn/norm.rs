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
  error::{
    ArithmeticOverflowPayload, DivisibilityConstraintPayload, InvariantViolationPayload,
    LengthMismatchPayload, OutOfRangePayload, RankMismatchPayload, Result, check,
  },
  ops,
  stream::default_stream,
};
use smol_str::format_smolstr;

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
  /// `weight: MLXArray` / python `self.weight` non-optional field). Private
  /// so the constructor is the only installation path; access via
  /// [`Self::weight_ref`].
  weight: Array,
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

  /// Read-only reference to the per-feature scale (`(dims,)` shape).
  ///
  /// Named `weight_ref` (non-Copy `Array` returns `&Array`, not
  /// `Array`; `_ref` suffix signals the borrow). Lazy — does not evaluate.
  #[inline(always)]
  pub fn weight_ref(&self) -> &Array {
    &self.weight
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
  /// Optional per-feature affine scale of shape `(dims,)`. Private so the
  /// constructor is the only installation path; access via
  /// [`Self::weight_ref`].
  weight: Option<Array>,
  /// Optional per-feature affine shift of shape `(dims,)`. `weight =
  /// None, bias = Some(_)` is rare (mirrors python `affine=False`
  /// dropping both); the kernel still accepts it. Private so the
  /// constructor is the only installation path; access via
  /// [`Self::bias_ref`].
  bias: Option<Array>,
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

  /// Read-only reference to the optional affine scale (`(dims,)` shape, or
  /// `None` when `affine=False`).
  ///
  /// Named `weight_ref` (non-Copy `Array` returns `&Array`, not
  /// `Array`; `_ref` suffix signals the borrow). Lazy — does not evaluate.
  #[inline(always)]
  pub fn weight_ref(&self) -> Option<&Array> {
    self.weight.as_ref()
  }

  /// Read-only reference to the optional affine shift (`(dims,)` shape, or
  /// `None` when `bias=False`).
  ///
  /// Named `bias_ref` (non-Copy `Array` returns `&Array`, not
  /// `Array`; `_ref` suffix signals the borrow). Lazy — does not evaluate.
  #[inline(always)]
  pub fn bias_ref(&self) -> Option<&Array> {
    self.bias.as_ref()
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
/// `num_groups` and `dims` are private fields exposed via the
/// [`num_groups`](Self::num_groups) / [`dims`](Self::dims) read-only
/// accessors. They carry the strict modulo invariant
/// `dims % num_groups == 0` that `Self::new` enforces; allowing external
/// `&mut` mutation would let a caller drop `num_groups` to 0 (panicking
/// in the private `validate_input_shape`'s `dims_i32 % self.num_groups`
/// step) or break the divisibility (silent activation corruption). This
/// mirrors swift's `public let groupCount` / `public let dimensions`
/// immutability without giving up the constructor's validation.
/// `eps`/`pytorch_compatible` stay `pub` — they don't carry an invariant.
///
/// `affine` is a single private `Option<(weight, bias)>` exposed via the
/// [`affine`](Self::affine) read-only accessor. mlx's `nn.GroupNorm` has
/// one `affine: bool` toggle — `weight` and `bias` are created *together*
/// inside `if affine:` (both or neither), and `_extra_repr` reports
/// `affine` as the single `'weight' in self` predicate. Two independent
/// `Option<Array>` fields would make the invalid `(Some, None)` /
/// `(None, Some)` partial states representable: a caller could construct
/// the layer then set only one, and `forward` would *silently* drop the
/// lone parameter (wrong activations, no error). The single-`Option`
/// tuple makes a partial affine a compile-time impossibility, for the
/// same both-or-none invariant reason `num_groups`/`dims` are private.
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
  /// produce a malformed reshape otherwise). PRIVATE so it can't be
  /// post-construction mutated to 0 / negative / a value that breaks the
  /// modulo invariant `Self::new` enforces; read via
  /// [`Self::num_groups`].
  num_groups: i32,
  /// Configured feature width (last axis of the expected input). Stored
  /// so [`forward`](Self::forward) can reject a checkpoint/config mismatch
  /// up-front instead of silently normalizing whatever last-axis width
  /// happens to be divisible by `num_groups`. The references store it on
  /// the module unconditionally; we mirror that here. PRIVATE for the
  /// same modulo-invariant reason as [`Self::num_groups`]; read via
  /// [`Self::dims`].
  dims: i32,
  /// Affine parameters: `Some((weight, bias))` for an affine GroupNorm,
  /// `None` otherwise. Both tensors are rank-1 shape `(dims,)`. A SINGLE
  /// `Option` of the *pair* (not two independent `Option`s) so the
  /// invalid `(Some, None)` / `(None, Some)` partial state is
  /// unrepresentable — mlx's `nn.GroupNorm` creates `weight` and `bias`
  /// together inside `if affine:`. PRIVATE so the both-or-none invariant
  /// can't be broken post-construction; read via [`Self::affine`],
  /// installed by [`Self::new`] (default `(ones, zeros)`) or
  /// [`Self::with_affine`] (a checkpoint's learned tensors).
  affine: Option<(Array, Array)>,
  /// Variance floor inside the sqrt (references default `1e-5`).
  pub eps: f32,
  /// `true` selects the pytorch-grouping path (see struct docs).
  pub pytorch_compatible: bool,
}

/// Validate a [`GroupNorm`] `(num_groups, dims)` config: both must be
/// positive and `dims` must be evenly divisible by `num_groups`.
///
/// Shared by [`GroupNorm::new`] and [`GroupNorm::with_affine`] so the
/// rule has one home. Crucially, [`GroupNorm::new`] calls this BEFORE
/// allocating the default `(ones, zeros)` affine tensors — a malformed
/// config is a cheap int-check failure, never one paid for after two
/// MLX array allocations. The checks are pure integer comparisons, so
/// re-running it (when `new` delegates to `with_affine`) is harmless.
fn validate_group_params(num_groups: i32, dims: i32) -> Result<()> {
  if num_groups <= 0 {
    return Err(crate::error::Error::OutOfRange(OutOfRangePayload::new(
      "GroupNorm: num_groups",
      "must be positive (> 0)",
      format_smolstr!("{num_groups}"),
    )));
  }
  if dims <= 0 {
    return Err(crate::error::Error::OutOfRange(OutOfRangePayload::new(
      "GroupNorm: dims",
      "must be positive (> 0)",
      format_smolstr!("{dims}"),
    )));
  }
  if dims % num_groups != 0 {
    return Err(crate::error::Error::DivisibilityConstraint(
      DivisibilityConstraintPayload::new(
        "GroupNorm",
        "dims",
        dims as u64,
        "num_groups",
        num_groups as u64,
      ),
    ));
  }
  Ok(())
}

impl GroupNorm {
  /// Construct a GroupNorm matching the swift `GroupNorm(groupCount:,
  /// dimensions:, eps:, affine:, pytorchCompatible:)` init signature.
  ///
  /// Validates `(num_groups, dims)` up-front: both must be positive and
  /// `dims` must be evenly divisible by `num_groups`. This applies to
  /// BOTH `affine = true` and `affine = false` — without it, an
  /// `affine = false` GroupNorm with an invalid `dims` would silently
  /// pass at construction and then [`Self::forward`] would derive the
  /// feature width from the input's last axis, normalizing whatever
  /// happens to be divisible by `num_groups` (config/checkpoint mismatch
  /// silently corrupting activations).
  ///
  /// `affine = true` materializes the references' `weight = ones((dims,))`
  /// and `bias = zeros((dims,))` (a small fallible alloc) into the single
  /// `Some((weight, bias))` field, so this is `Result<Self>`.
  /// `affine = false` skips the alloc and leaves the field `None`. `dims`
  /// is stored on the struct unconditionally (matching the references)
  /// and is checked against the input's last axis inside
  /// [`Self::forward`].
  ///
  /// This is the `affine: bool` convenience constructor — it can only
  /// install the references' *default* `(ones, zeros)` affine. To install
  /// a checkpoint's LEARNED `weight`/`bias`, use [`Self::with_affine`]
  /// (this delegates to it: `affine = true` builds `ones`/`zeros` then
  /// forwards `Some((ones, zeros))`, `affine = false` forwards `None`).
  pub fn new(
    num_groups: i32,
    dims: i32,
    eps: f32,
    affine: bool,
    pytorch_compatible: bool,
  ) -> Result<Self> {
    // `affine = true` materializes the references' default `(ones,
    // zeros)`; `affine = false` ⇒ no affine. The stored-state construction
    // and the affine-shape validation live in `with_affine` (the single
    // source of truth) — `new` is the bool-to-`Option` shim.
    //
    // Run the FULL `(num_groups, dims)` validation up-front, BEFORE the
    // `ones`/`zeros` alloc: a malformed config (`num_groups <= 0`, or a
    // non-divisible `dims`) must fail as a cheap int check, never after
    // materializing two MLX arrays (allocation pressure / possible OOM
    // for a config that is already known-bad). `with_affine` re-runs the
    // same helper — running it twice is harmless (just int comparisons),
    // and keeping one helper means one source of truth for the rule.
    validate_group_params(num_groups, dims)?;
    let affine = if affine {
      // `validate_group_params` proved `dims > 0` ⇒ `usize::try_from`
      // cannot fail; this alloc is only reached on a valid config.
      let d = usize::try_from(dims).expect("dims > 0 guarded by validate_group_params");
      Some((Array::ones::<f32>(&(d,))?, Array::zeros::<f32>(&(d,))?))
    } else {
      None
    };
    Self::with_affine(num_groups, dims, eps, affine, pytorch_compatible)
  }

  /// Construct a GroupNorm with explicit affine tensors — the fallible
  /// constructor for installing a checkpoint's LEARNED `weight`/`bias`.
  ///
  /// [`Self::new`]'s `affine: bool` toggle can only produce the
  /// references' *default* `(ones, zeros)` affine, so a model whose
  /// GroupNorm carries non-default learned affine params would run with
  /// an identity affine (wrong activations). This constructor takes the
  /// real tensors instead.
  ///
  /// Runs the same `(num_groups, dims)` validation as [`Self::new`]
  /// (both positive, `dims % num_groups == 0`). When `affine` is
  /// `Some((weight, bias))`, BOTH tensors must be exactly rank-1 shape
  /// `[dims]` — a wrong rank yields `Error::RankMismatch`, and a rank-1
  /// length other than `dims` yields `Error::LengthMismatch`, each
  /// naming the offending tensor (`weight` vs `bias`). `affine = None`
  /// ⇒ no affine. The
  /// `Option<(Array, Array)>` is stored directly, preserving the
  /// both-or-none invariant ([`Self::new`] delegates here so the rule has
  /// a single home).
  pub fn with_affine(
    num_groups: i32,
    dims: i32,
    eps: f32,
    affine: Option<(Array, Array)>,
    pytorch_compatible: bool,
  ) -> Result<Self> {
    // Full `(num_groups, dims)` validation first — this is a public entry
    // point in its own right, and the shared helper is the single source
    // of truth for the positive/divisible rule.
    validate_group_params(num_groups, dims)?;
    // When affine tensors are supplied, BOTH must be exactly rank-1
    // `[dims]` — a checkpoint with a `[1, dims]` (rank-2) or `[dims + 1]`
    // (wrong length) tensor would otherwise broadcast/fail unpredictably
    // inside `forward`'s `multiply`/`add`. Validate here so the misuse
    // surfaces at construction with a precise message.
    if let Some((weight, bias)) = &affine {
      // `dims > 0` guaranteed above ⇒ the cast is lossless.
      let dims_usize = dims as usize;
      // Split the affine-shape check into the precise taxonomy:
      //   * `RankMismatch` when the tensor is not rank-1 (e.g. `[1, dims]`).
      //   * `LengthMismatch` when it IS rank-1 but the single dim differs
      //     from `dims` (e.g. `[dims + 1]`).
      //   `ShapePairMismatch` is reserved for same-rank multi-dimension
      //   shape disagreements — which can never arise here (`want` is
      //   rank-1), so we never use it on this branch.
      let w_shape = weight.shape();
      if w_shape.len() != 1 {
        return Err(crate::error::Error::RankMismatch(RankMismatchPayload::new(
          "GroupNorm: affine weight must be rank-1 [dims]",
          w_shape.len() as u32,
          w_shape.to_vec(),
        )));
      }
      if w_shape[0] != dims_usize {
        return Err(crate::error::Error::LengthMismatch(
          LengthMismatchPayload::new(
            "GroupNorm: affine weight length must equal dims",
            dims_usize,
            w_shape[0],
          ),
        ));
      }
      let b_shape = bias.shape();
      if b_shape.len() != 1 {
        return Err(crate::error::Error::RankMismatch(RankMismatchPayload::new(
          "GroupNorm: affine bias must be rank-1 [dims]",
          b_shape.len() as u32,
          b_shape.to_vec(),
        )));
      }
      if b_shape[0] != dims_usize {
        return Err(crate::error::Error::LengthMismatch(
          LengthMismatchPayload::new(
            "GroupNorm: affine bias length must equal dims",
            dims_usize,
            b_shape[0],
          ),
        ));
      }
    }
    Ok(Self {
      num_groups,
      dims,
      affine,
      eps,
      pytorch_compatible,
    })
  }

  /// Configured number of feature groups (`> 0`, divides
  /// [`Self::dims`] evenly — guaranteed by [`Self::new`]).
  ///
  /// Read-only accessor for the private `num_groups` field; the field is
  /// private specifically so it can't be post-construction mutated to a
  /// value that would break the private `validate_input_shape`'s
  /// `dims_i32 % self.num_groups` step (e.g. setting it to `0` would
  /// panic on the modulo).
  pub fn num_groups(&self) -> i32 {
    self.num_groups
  }

  /// Configured feature width (`> 0`, evenly divisible by
  /// [`Self::num_groups`] — guaranteed by [`Self::new`]).
  ///
  /// Read-only accessor for the private `dims` field; the field is
  /// private to preserve the constructor's modulo invariant against
  /// post-construction `&mut` mutation. See [`Self::num_groups`] for the
  /// same rationale.
  pub fn dims(&self) -> i32 {
    self.dims
  }

  /// Affine parameters as `Some((&weight, &bias))` when this GroupNorm
  /// was built with `affine = true`, `None` otherwise.
  ///
  /// Read-only accessor for the private `affine` field. The field is
  /// private — and a single `Option` of the *pair* rather than two
  /// independent `Option`s — so the invalid partial-affine state
  /// (`weight` set but `bias` unset, or vice versa) can be neither
  /// constructed nor mutated into. `weight` and `bias` are always both
  /// present or both absent, matching mlx's `if affine:` block.
  pub fn affine(&self) -> Option<(&Array, &Array)> {
    self.affine.as_ref().map(|(w, b)| (w, b))
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
    // `Some((w, b))` ⇒ scale + shift; `None` ⇒ pure normalization. The
    // single `Option<(Array, Array)>` field has no partial arm — a lone
    // weight or bias is unrepresentable, so the previous silent-drop bug
    // (`(Some, None)` falling through to `_ => Ok(normalized)`) cannot
    // occur.
    match &self.affine {
      Some((w, b)) => {
        let scaled = ops::arithmetic::multiply(w, &normalized)?;
        ops::arithmetic::add(&scaled, b)
      }
      None => Ok(normalized),
    }
  }

  /// Validate the input shape against GroupNorm's invariants: rank ≥ 2
  /// (so there is a feature axis distinct from the batch axis), and the
  /// last (feature) axis matches the configured [`Self::dims`]. Returns
  /// the feature dim as `i32` for downstream `group_size = dims /
  /// num_groups` arithmetic.
  ///
  /// Constructor invariants the dims-equality check piggybacks on:
  /// `Self::new` already rejected `num_groups <= 0`, `dims <= 0`, and
  /// `dims % num_groups != 0`, so once the last axis equals
  /// `self.dims` the divisibility/positivity follow transitively. The
  /// rank check stays explicit (the input shape is per-call, not a
  /// constructor invariant), and the divisibility check is kept as
  /// belt-and-suspenders (the dims-equality assertion makes it
  /// unreachable, but a future refactor that reorders the checks
  /// wouldn't silently regress the bug class).
  ///
  /// Both [`Self::group_norm`] and [`Self::pytorch_group_norm`] call this
  /// up-front. The references don't run an explicit guard (they rely on
  /// the user / framework wiring), but in the safe layer skipping it
  /// produces *silent* activation corruption (e.g. rank-1 `[C]` with
  /// `num_groups=1` would pass as `[C, 1, 1]` and normalize singleton
  /// groups to zero; a checkpoint configured for `dims=4` fed an `[1, 8]`
  /// input would pass the divisibility check but normalize the wrong
  /// channel count) — surface the misuse as `Err(`[`Error::RankMismatch`]`)` /
  /// `Err(`[`Error::LengthMismatch`]`)` / `Err(`[`Error::DivisibilityConstraint`]`)` instead.
  fn validate_input_shape(&self, orig_shape: &[usize]) -> Result<i32> {
    if orig_shape.len() < 2 {
      return Err(crate::error::Error::RankMismatch(RankMismatchPayload::new(
        "GroupNorm input must have rank >= 2 (at least [batch, dims])",
        orig_shape.len() as u32,
        orig_shape.to_vec(),
      )));
    }
    let dims = *orig_shape
      .last()
      .expect("rank-≥-2 guarded above ⇒ last() is Some");
    let dims_i32 = i32::try_from(dims).map_err(|_| {
      crate::error::Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "GroupNorm: feature dim exceeds i32::MAX",
        "i32",
        [("dim", dims as u64)],
      ))
    })?;
    if dims_i32 != self.dims {
      // Both `dims_i32` and `self.dims` are positive i32 (constructor validation).
      return Err(crate::error::Error::LengthMismatch(
        LengthMismatchPayload::new(
          "GroupNorm: input last-axis must match configured dims",
          self.dims as usize,
          dims_i32 as usize,
        ),
      ));
    }
    // Constructor already enforces `dims % num_groups == 0`, so once
    // `dims_i32 == self.dims` this is unreachable. Kept as
    // belt-and-suspenders against a future refactor that reorders the
    // invariant checks.
    if dims_i32 % self.num_groups != 0 {
      // Constructor enforces num_groups > 0, dims > 0; both > 0 here.
      return Err(crate::error::Error::DivisibilityConstraint(
        DivisibilityConstraintPayload::new(
          "GroupNorm",
          "feature_dim",
          dims_i32 as u64,
          "num_groups",
          self.num_groups as u64,
        ),
      ));
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
    let collapsed = mid.checked_mul(group_size).ok_or_else(|| {
      // mid, group_size are positive i32 here (validated above).
      crate::error::Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "GroupNorm: mid * group_size",
        "i32",
        [("mid", mid as u64), ("group_size", group_size as u64)],
      ))
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
      i32::try_from(d).map_err(|_| {
        crate::error::Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "shape_to_i32: dim exceeds i32::MAX",
          "i32",
          [("dim", d as u64)],
        ))
      })
    })
    .collect()
}

/// Pull the leading batch dim as `i32`, erroring on rank 0 or on a dim
/// past `i32::MAX`.
fn batch_dim(shape: &[usize]) -> Result<i32> {
  let b = *shape.first().ok_or_else(|| {
    crate::error::Error::RankMismatch(RankMismatchPayload::new(
      "GroupNorm input must have rank >= 1 (the batch axis)",
      0,
      shape.to_vec(),
    ))
  })?;
  i32::try_from(b).map_err(|_| {
    crate::error::Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "GroupNorm: batch dim exceeds i32::MAX",
      "i32",
      [("batch_dim", b as u64)],
    ))
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
  //
  // Build the payload at the failure site so the overflowing `acc`, the
  // current `dim`, and its index are preserved — a plain
  // `try_fold` returning `Option<usize>` would drop every operand
  // by the time we reached `ok_or_else`.
  let total: usize = shape
    .iter()
    .enumerate()
    .try_fold(1usize, |acc, (idx, &dim)| {
      acc.checked_mul(dim).ok_or_else(|| {
        crate::error::Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "GroupNorm: shape product overflows usize",
          "usize",
          [
            ("acc", acc as u64),
            ("dim", dim as u64),
            ("dim_index", idx as u64),
          ],
        ))
      })
    })?;
  let mut divisor: usize = 1;
  for &d in known_dims {
    let du = usize::try_from(d).map_err(|_| {
      crate::error::Error::OutOfRange(OutOfRangePayload::new(
        "GroupNorm: known reshape dim",
        "must be non-negative",
        format_smolstr!("{d}"),
      ))
    })?;
    divisor = divisor.checked_mul(du).ok_or_else(|| {
      crate::error::Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "GroupNorm: reshape divisor product",
        "usize",
        [("divisor", divisor as u64), ("factor", du as u64)],
      ))
    })?;
  }
  if divisor == 0 {
    return Err(crate::error::Error::InvariantViolation(
      InvariantViolationPayload::new(
        "GroupNorm: inferred_dim reshape divisor",
        "must be non-zero (one of the known_dims was 0)",
      ),
    ));
  }
  if !total.is_multiple_of(divisor) {
    return Err(crate::error::Error::DivisibilityConstraint(
      DivisibilityConstraintPayload::new(
        "GroupNorm: cannot reshape elements into a layout",
        "total_elements",
        total as u64,
        "divisor_per_slot",
        divisor as u64,
      ),
    ));
  }
  let inferred = total / divisor;
  i32::try_from(inferred).map_err(|_| {
    crate::error::Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "GroupNorm: inferred dim exceeds i32::MAX",
      "i32",
      [("inferred_dim", inferred as u64)],
    ))
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
mod tests;
