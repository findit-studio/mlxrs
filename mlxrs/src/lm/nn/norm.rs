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
    return Err(crate::error::Error::ShapeMismatch {
      message: format!("GroupNorm: num_groups ({num_groups}) must be positive"),
    });
  }
  if dims <= 0 {
    return Err(crate::error::Error::ShapeMismatch {
      message: format!("GroupNorm: dims ({dims}) must be positive"),
    });
  }
  if dims % num_groups != 0 {
    return Err(crate::error::Error::ShapeMismatch {
      message: format!(
        "GroupNorm: dims ({dims}) must be evenly divisible by num_groups ({num_groups})"
      ),
    });
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
  /// `[dims]` — a mismatch is rejected with `Error::ShapeMismatch` naming
  /// the expected vs actual shape. `affine = None` ⇒ no affine. The
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
      let want = vec![dims as usize];
      let w_shape = weight.shape();
      if w_shape != want {
        return Err(crate::error::Error::ShapeMismatch {
          message: format!(
            "GroupNorm: affine weight must be shape {want:?} (rank-1, length dims={dims}), got {w_shape:?}"
          ),
        });
      }
      let b_shape = bias.shape();
      if b_shape != want {
        return Err(crate::error::Error::ShapeMismatch {
          message: format!(
            "GroupNorm: affine bias must be shape {want:?} (rank-1, length dims={dims}), got {b_shape:?}"
          ),
        });
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
  /// channel count) — surface the misuse as `Err(ShapeMismatch)` instead.
  fn validate_input_shape(&self, orig_shape: &[usize]) -> Result<i32> {
    if orig_shape.len() < 2 {
      return Err(crate::error::Error::ShapeMismatch {
        message: format!(
          "GroupNorm input must have rank >= 2 (at least [batch, dims]), got rank {} shape {orig_shape:?}",
          orig_shape.len()
        ),
      });
    }
    let dims = *orig_shape
      .last()
      .expect("rank-≥-2 guarded above ⇒ last() is Some");
    let dims_i32 = i32::try_from(dims).map_err(|_| crate::error::Error::ShapeMismatch {
      message: format!("GroupNorm: feature dim {dims} exceeds i32::MAX"),
    })?;
    if dims_i32 != self.dims {
      return Err(crate::error::Error::ShapeMismatch {
        message: format!(
          "GroupNorm: input last-axis ({dims_i32}) must match configured dims ({})",
          self.dims
        ),
      });
    }
    // Constructor already enforces `dims % num_groups == 0`, so once
    // `dims_i32 == self.dims` this is unreachable. Kept as
    // belt-and-suspenders against a future refactor that reorders the
    // invariant checks.
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
  fn group_norm_affine_true_applies_scale_and_shift() {
    // Regression: affine=true output == normalized * weight + bias, where
    // `normalized` is the affine=false (pure-normalization) result and
    // `(weight, bias)` is the pair the `affine()` accessor exposes. The
    // constructor materializes the references' `(ones, zeros)`, so this
    // also pins that the default affine is the identity on `normalized`.
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
    let plain = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    let affine = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
    let mut normalized = plain.forward(&x).unwrap();
    let mut a = affine.forward(&x).unwrap();
    let normalized_v = normalized.to_vec::<f32>().unwrap();
    let av = a.to_vec::<f32>().unwrap();
    // affine=true output == normalized * weight + bias.
    let (w, b) = affine.affine().expect("affine=true ⇒ Some");
    let scaled = ops::arithmetic::multiply(w, &normalized).unwrap();
    let mut want = ops::arithmetic::add(&scaled, b).unwrap();
    assert!(vclose(&av, &want.to_vec::<f32>().unwrap()));
    // default (ones, zeros) ⇒ the affine is the identity on `normalized`.
    assert!(vclose(&av, &normalized_v));
  }

  #[test]
  fn group_norm_affine_false_is_pure_normalization() {
    // affine=false ⇒ `affine()` is None ⇒ `forward` takes the `None` arm
    // and returns the pure normalized result (no scale, no shift). Pinned
    // against `group_norm_affine_true_applies_scale_and_shift`, which
    // asserts the affine=false output is exactly the `normalized` term.
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
    let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    assert!(gn.affine().is_none());
    let mut y = gn.forward(&x).unwrap();
    let v = y.to_vec::<f32>().unwrap();
    assert_eq!(v.len(), 4);
    assert!(v.iter().all(|x| x.is_finite()));
  }

  /// affine is a single both-or-none `Option<(weight, bias)>`:
  /// `affine=true` ⇒ `affine()` is `Some`, `affine=false` ⇒ `None`. A
  /// partial affine (lone weight or lone bias) is a compile-time
  /// impossibility — the field is private and holds the pair, not two
  /// independent `Option`s, so `(Some, None)` / `(None, Some)` cannot be
  /// constructed or mutated into (the old code had a silent
  /// `_ => Ok(normalized)` drop for exactly those states).
  #[test]
  fn group_norm_affine_is_both_or_none_by_construction() {
    let with_affine = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
    assert!(with_affine.affine().is_some());
    let no_affine = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    assert!(no_affine.affine().is_none());
    // (compile-fail) `with_affine.affine = Some((w, ...))` with only a
    // weight is impossible: the field is private AND its type is
    // `Option<(Array, Array)>`, so a lone parameter has no representation.
  }

  #[test]
  fn group_norm_default_constructor_no_affine() {
    // affine=false ⇒ the affine field is None (no allocation).
    let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    assert!(gn.affine().is_none());
    assert!(!gn.pytorch_compatible);
    assert_eq!(gn.num_groups(), 2);
  }

  #[test]
  fn group_norm_default_constructor_affine_allocates() {
    let gn = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
    let (w, b) = gn.affine().expect("affine=true ⇒ Some");
    assert_eq!(w.shape(), vec![4]);
    assert_eq!(b.shape(), vec![4]);
    let mut w = w.try_clone().unwrap();
    let mut b = b.try_clone().unwrap();
    assert_eq!(w.to_vec::<f32>().unwrap(), vec![1.0; 4]);
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![0.0; 4]);
  }

  // ─── GroupNorm::with_affine checkpoint-tensor regressions (Codex review) ───

  /// `with_affine` installs a checkpoint's LEARNED (non-identity)
  /// `(weight, bias)` — the gap `new`'s `affine: bool` couldn't fill
  /// (it can only build the default `(ones, zeros)`). `affine()` must
  /// return those exact tensors back.
  #[test]
  fn group_norm_with_affine_accepts_checkpoint_tensors() {
    let w = Array::from_slice::<f32>(&[2.0, 2.0, 2.0, 2.0], &(4,)).unwrap();
    let b = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0], &(4,)).unwrap();
    let gn = GroupNorm::with_affine(2, 4, 1e-5, Some((w, b)), false).unwrap();
    let (gw, gb) = gn.affine().expect("with_affine(Some(_)) ⇒ Some");
    let mut gw = gw.try_clone().unwrap();
    let mut gb = gb.try_clone().unwrap();
    assert_eq!(gw.to_vec::<f32>().unwrap(), vec![2.0; 4]);
    assert_eq!(gb.to_vec::<f32>().unwrap(), vec![1.0; 4]);
  }

  /// Coverage gap the finding flagged: the prior affine test only used
  /// `(ones, zeros)` (an identity affine), so a broken scale/shift would
  /// not be caught. Construct via `with_affine` with NON-identity
  /// `weight`/`bias` and assert `forward` output is exactly
  /// `normalized * weight + bias` (and NOT the identity `normalized`).
  #[test]
  fn group_norm_with_affine_non_identity_forward_applies_scale_shift() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
    let w = Array::from_slice::<f32>(&[2.0, 2.0, 2.0, 2.0], &(4,)).unwrap();
    let b = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0], &(4,)).unwrap();
    // `normalized` = the pure (affine=None) normalization of `x`.
    let plain = GroupNorm::with_affine(2, 4, 1e-5, None, false).unwrap();
    let mut normalized = plain.forward(&x).unwrap();
    let normalized_v = normalized.to_vec::<f32>().unwrap();
    // `forward` with the non-identity affine.
    let affine = GroupNorm::with_affine(
      2,
      4,
      1e-5,
      Some((w.try_clone().unwrap(), b.try_clone().unwrap())),
      false,
    )
    .unwrap();
    let mut got = affine.forward(&x).unwrap();
    // Expected: `normalized * weight + bias`, computed independently.
    let scaled = ops::arithmetic::multiply(&w, &normalized).unwrap();
    let mut want = ops::arithmetic::add(&scaled, &b).unwrap();
    assert!(vclose(
      &got.to_vec::<f32>().unwrap(),
      &want.to_vec::<f32>().unwrap()
    ));
    // Sanity: the non-identity affine actually moved the result off
    // `normalized` (weight=2/bias=1 cannot be the identity here).
    assert!(
      !vclose(&got.to_vec::<f32>().unwrap(), &normalized_v),
      "non-identity affine must change the output"
    );
  }

  /// `with_affine` rejects a `weight` that is not exactly rank-1
  /// `[dims]` — both a wrong length (`[dims + 1]`) and a wrong rank
  /// (`[1, dims]`) must produce `Err(ShapeMismatch)`.
  #[test]
  fn group_norm_with_affine_rejects_wrong_shape_weight() {
    let bias = Array::zeros::<f32>(&(4,)).unwrap();
    // Wrong length: `[dims + 1]`.
    let long_w = Array::ones::<f32>(&(5,)).unwrap();
    let err = GroupNorm::with_affine(2, 4, 1e-5, Some((long_w, bias.try_clone().unwrap())), false)
      .unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("weight") && message.contains("[5]"),
          "unexpected message: {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
    // Wrong rank: `[1, dims]`.
    let rank2_w = Array::ones::<f32>(&(1, 4)).unwrap();
    let err = GroupNorm::with_affine(2, 4, 1e-5, Some((rank2_w, bias)), false).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("weight") && message.contains("[1, 4]"),
          "unexpected message: {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }

  /// `with_affine` rejects a `bias` that is not exactly rank-1 `[dims]`
  /// — same check as for `weight`, both wrong-length and wrong-rank.
  #[test]
  fn group_norm_with_affine_rejects_wrong_shape_bias() {
    let weight = Array::ones::<f32>(&(4,)).unwrap();
    // Wrong length: `[dims + 1]`.
    let long_b = Array::zeros::<f32>(&(5,)).unwrap();
    let err = GroupNorm::with_affine(
      2,
      4,
      1e-5,
      Some((weight.try_clone().unwrap(), long_b)),
      false,
    )
    .unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("bias") && message.contains("[5]"),
          "unexpected message: {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
    // Wrong rank: `[1, dims]`.
    let rank2_b = Array::zeros::<f32>(&(1, 4)).unwrap();
    let err = GroupNorm::with_affine(2, 4, 1e-5, Some((weight, rank2_b)), false).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("bias") && message.contains("[1, 4]"),
          "unexpected message: {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }

  /// `with_affine(.., None, ..)` ⇒ `affine()` is `None` and `forward`
  /// returns the pure normalized result (no scale, no shift).
  #[test]
  fn group_norm_with_affine_none_is_pure_normalization() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
    let gn = GroupNorm::with_affine(2, 4, 1e-5, None, false).unwrap();
    assert!(gn.affine().is_none());
    // `forward` must equal the default-path `group_norm` normalization.
    let mut got = gn.forward(&x).unwrap();
    let mut want = gn.group_norm(&x).unwrap();
    assert!(vclose(
      &got.to_vec::<f32>().unwrap(),
      &want.to_vec::<f32>().unwrap()
    ));
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
  /// splittable). The new constructor catches `dims % num_groups != 0`
  /// before construction; constructing with valid `dims=4` and then
  /// forwarding `[1, 3]` (whose last-axis 3 != configured 4) exercises
  /// the dims-equality enforcement in the forward.
  #[test]
  fn group_norm_feature_dim_mismatch_errors() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    let err = gn.forward(&x).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("last-axis") && message.contains("4") && message.contains("3"),
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

  /// Same dims-equality invariant on the `pytorch_compatible` path:
  /// mismatch between configured `dims=4` and input last-axis 3 is
  /// rejected.
  #[test]
  fn group_norm_pytorch_compat_feature_dim_mismatch_errors() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
    let gn = GroupNorm::new(2, 4, 1e-5, false, true).unwrap();
    let err = gn.forward(&x).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("last-axis") && message.contains("4") && message.contains("3"),
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

  // ─── GroupNorm constructor validation regressions (Codex R2) ───

  /// Constructor must reject negative `dims` on BOTH affine paths.
  /// Previously only the `affine=true` branch ran `usize::try_from`; the
  /// `affine=false` branch silently accepted any `dims` (including
  /// nonsense) and the forward derived the feature width from
  /// `x.shape().last()`.
  #[test]
  fn group_norm_constructor_rejects_negative_dims() {
    let err = GroupNorm::new(2, -1, 1e-5, false, false).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("dims") && message.contains("positive"),
          "unexpected message: {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }

  /// Constructor must reject `dims` not divisible by `num_groups` on
  /// BOTH affine paths. Previously the divisibility was only checked at
  /// forward-time against `x.shape().last()`, so an `affine=false`
  /// GroupNorm could be constructed with `dims=3, num_groups=2` and
  /// later normalize a `[1, 4]` input (whose last axis happens to
  /// divide 2) — silent config/checkpoint mismatch.
  #[test]
  fn group_norm_constructor_rejects_non_divisible_dims() {
    let err = GroupNorm::new(2, 3, 1e-5, false, false).unwrap_err();
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

  /// `dims == 0` is rejected (`positive` ⇒ `> 0`, not `>= 0`). A
  /// zero-dim GroupNorm has no feature axis to normalize and the
  /// downstream `dims / num_groups` would yield `group_size = 0`.
  #[test]
  fn group_norm_constructor_rejects_zero_dims() {
    let err = GroupNorm::new(2, 0, 1e-5, false, false).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("dims") && message.contains("positive"),
          "unexpected message: {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }

  /// Regression: a valid `affine=false` GroupNorm still constructs Ok
  /// after the new constructor checks. (Belt for the suspenders — the
  /// existing `group_norm_default_constructor_no_affine` test already
  /// covers this; keep an explicit one named for the constructor-spec
  /// item.)
  #[test]
  fn group_norm_constructor_accepts_valid_non_affine() {
    let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    assert_eq!(gn.dims(), 4);
    assert_eq!(gn.num_groups(), 2);
    assert!(gn.affine().is_none());
  }

  /// Forward rejects a config/checkpoint dim mismatch: construct with
  /// `dims=4` and call forward on `[1, 8]`. The 8-wide input is
  /// divisible by `num_groups=2` (would have silently normalized
  /// previously), but doesn't match the configured `dims=4` ⇒
  /// `Err(ShapeMismatch)` naming both expected (4) and actual (8).
  #[test]
  fn group_norm_forward_rejects_dim_mismatch() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &(1, 8)).unwrap();
    let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
    let err = gn.forward(&x).unwrap_err();
    match err {
      crate::error::Error::ShapeMismatch { message } => {
        assert!(
          message.contains("last-axis")
            && message.contains("dims")
            && message.contains("4")
            && message.contains("8"),
          "expected message to name configured dims (4) and actual (8): {message}"
        );
      }
      other => panic!("expected ShapeMismatch, got {other:?}"),
    }
  }

  /// `GroupNorm::new(.., affine = true, ..)` must reject a malformed
  /// `(num_groups, dims)` config on the cheap integer validation
  /// (`validate_group_params`) BEFORE materializing the default
  /// `(ones, zeros)` affine tensors. Previously `new` checked only
  /// `dims > 0`, built the two MLX arrays, and only THEN ran the full
  /// `num_groups`/divisibility validation inside `with_affine` — so a
  /// known-bad config paid for two allocations before erroring. Both a
  /// non-positive `num_groups` and a non-divisible `dims` must `Err`.
  ///
  /// The no-allocation property is structural: `validate_group_params`
  /// is called before the `Array::ones`/`Array::zeros` lines in `new`,
  /// so an `Err` here is returned without ever reaching them.
  #[test]
  fn group_norm_new_affine_true_invalid_config_rejects_before_alloc() {
    // `num_groups = 0` (non-positive) with `affine = true`.
    let err = GroupNorm::new(0, 4, 1e-5, true, false).unwrap_err();
    assert!(
      matches!(err, crate::error::Error::ShapeMismatch { .. }),
      "expected ShapeMismatch for num_groups=0, got {err:?}"
    );
    // `dims = 8` not divisible by `num_groups = 3`, with `affine = true`.
    let err = GroupNorm::new(3, 8, 1e-5, true, false).unwrap_err();
    assert!(
      matches!(err, crate::error::Error::ShapeMismatch { .. }),
      "expected ShapeMismatch for non-divisible dims, got {err:?}"
    );
  }

  // ─── GroupNorm field-visibility regressions (Codex R3) ───

  /// `num_groups` and `dims` are PRIVATE fields with read-only public
  /// accessors. This test demonstrates the accessors return the
  /// constructor-validated values and — by virtue of compiling without
  /// reaching for the field — confirms the read path goes through the
  /// accessor. Direct field access from outside `super::` would fail to
  /// compile (the field's visibility is module-private). External code
  /// previously could write `gn.num_groups = 0` and then `gn.forward(_)`
  /// would PANIC inside `validate_input_shape`'s `dims_i32 % 0`; with
  /// the field private, that mutation path is statically impossible.
  #[test]
  fn group_norm_num_groups_dims_are_read_only_via_accessors() {
    let gn = GroupNorm::new(4, 16, 1e-5, false, false).unwrap();
    assert_eq!(gn.num_groups(), 4);
    assert_eq!(gn.dims(), 16);
    // (compile-fail) external `gn.num_groups = 0` and `gn.dims = 0` are
    // both private-field errors; trying them here from inside `super::`
    // would compile (same module), so we don't try — the visibility
    // guarantee is what the regression turns on, not a runtime check.
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
