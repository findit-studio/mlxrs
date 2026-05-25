//! Pooling strategies that collapse `(batch, seq_len, hidden)` token
//! embeddings into `(batch, hidden)` sentence embeddings.
//!
//! Ported from `mlx-embeddings/models/pooling.py` (`mean_pooling`,
//! `cls_pooling`, `max_pooling`, `lasttoken_pooling`, `pool_by_config`),
//! `base.py::normalize_embeddings`, and `MLXEmbedders` `Pooling.swift`
//! (`Strategy`, `callAsFunction`).

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Error, Result, try_with_capacity},
  ops::{
    arithmetic::{divide, maximum, multiply, subtract},
    comparison::equal,
    indexing::{take_along_axis, take_axis},
    logical::select,
    misc::{argmax, astype},
    reduction::{max_axes, sum_axes},
    shape::{broadcast_to, expand_dims_axes, squeeze_axes},
  },
};

use super::{
  fast::{layer_norm, rms_norm},
  normalize::{DEFAULT_NORMALIZE_EPS, l2_normalize_eps},
  scalar_like,
};

/// swift `MLXFast.layerNorm(pooled, eps: 1e-5)` default (`Pooling.swift`).
pub const LAYER_NORM_EPS: f32 = 1e-5;

/// RMSNorm eps default (no swift `Pooling` reference — RMSNorm is the
/// mlx-c-surfaced post-pool variant applied to the pooled vector; mlx's
/// `nn.RMSNorm` default is `1e-5`).
pub const RMS_NORM_EPS: f32 = 1e-5;

/// Validate the `(token_embeddings, attention_mask)` rank/shape contract
/// shared by every mask-aware pooling helper *before* any `shape[i]`
/// indexing, so a wrong-rank caller gets a recoverable
/// [`Error::ShapeMismatch`] instead of a panic on a safe public API.
///
/// Requires `token_embeddings` rank-3 `(batch, seq_len, hidden)` and
/// `attention_mask` rank-2 `(batch, seq_len)` with agreeing `batch` and
/// `seq_len`. No behavior change for valid inputs (python/swift assume
/// these ranks; this only converts the would-be panic into an `Err`).
fn validate_token_embeddings_and_mask(
  token_embeddings: &Array,
  attention_mask: &Array,
) -> Result<()> {
  let emb_shape = token_embeddings.shape();
  let mask_shape = attention_mask.shape();
  if emb_shape.len() != 3 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "token_embeddings must be rank-3 (batch, seq_len, hidden), got rank {} shape {:?}",
        emb_shape.len(),
        emb_shape
      ),
    });
  }
  if mask_shape.len() != 2 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "attention_mask must be rank-2 (batch, seq_len), got rank {} shape {:?}",
        mask_shape.len(),
        mask_shape
      ),
    });
  }
  if emb_shape[0] != mask_shape[0] || emb_shape[1] != mask_shape[1] {
    return Err(Error::ShapeMismatch {
      message: format!(
        "token_embeddings (batch, seq_len) = ({}, {}) must match attention_mask ({}, {})",
        emb_shape[0], emb_shape[1], mask_shape[0], mask_shape[1]
      ),
    });
  }
  Ok(())
}

/// Validate `token_embeddings` is rank-3 `(batch, seq_len, hidden)` for
/// the mask-free [`first_token_pooling`] entry point — same panic→`Err`
/// guarantee as [`validate_token_embeddings_and_mask`].
fn validate_token_embeddings_rank3(token_embeddings: &Array) -> Result<()> {
  let emb_shape = token_embeddings.shape();
  if emb_shape.len() != 3 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "token_embeddings must be rank-3 (batch, seq_len, hidden), got rank {} shape {:?}",
        emb_shape.len(),
        emb_shape
      ),
    });
  }
  Ok(())
}

/// Mask-aware mean pooling.
///
/// `sum(token_embeddings * mask) / max(sum(mask), 1e-9)` over the sequence
/// axis, where `mask` is the attention mask broadcast to the embedding shape.
/// Padding positions (mask `0`) contribute nothing, and the `1e-9` floor
/// guards the all-padding row against division by zero. Mirrors
/// `mlx-embeddings` `mean_pooling`.
///
/// - `token_embeddings`: `(batch, seq_len, hidden)`
/// - `attention_mask`: `(batch, seq_len)`
/// - returns: `(batch, hidden)`
pub fn mean_pooling(token_embeddings: &Array, attention_mask: &Array) -> Result<Array> {
  validate_token_embeddings_and_mask(token_embeddings, attention_mask)?;
  let shape = token_embeddings.shape();
  let mask = expand_dims_axes(attention_mask, &[-1])?;
  let mask = broadcast_to(&mask, &shape.as_slice())?;
  let mask = astype(&mask, Dtype::F32)?;

  let weighted = multiply(token_embeddings, &mask)?;
  let sum_embeddings = sum_axes(&weighted, &[1], false)?;
  let sum_mask = sum_axes(&mask, &[1], false)?;
  let floor = Array::full::<f32>(&(1,), 1e-9)?;
  let sum_mask = maximum(&sum_mask, &floor)?;
  divide(&sum_embeddings, &sum_mask)
}

/// CLS pooling: select the first real (non-padding) token per sequence.
///
/// The first valid position is `argmax(attention_mask, axis=1)` (the first
/// `1`), gathered via `take_along_axis`. Mirrors `mlx-embeddings`
/// `cls_pooling` and is the strategy the dispatcher's
/// [`PoolingStrategy::Cls`] (and the ST-config CLS key) resolve to —
/// padding-robust, unlike the strict-token-0 [`first_token_pooling`].
///
/// - `token_embeddings`: `(batch, seq_len, hidden)`
/// - `attention_mask`: `(batch, seq_len)`
/// - returns: `(batch, hidden)`
pub fn cls_pooling(token_embeddings: &Array, attention_mask: &Array) -> Result<Array> {
  validate_token_embeddings_and_mask(token_embeddings, attention_mask)?;
  let shape = token_embeddings.shape();
  let batch = shape[0];
  let hidden = shape[2];

  let first_indices = argmax(attention_mask, Some(1), false)?;
  let gather_idx = expand_dims_axes(&first_indices, &[1, 2])?;
  let gather_idx = broadcast_to(&gather_idx, &(batch, 1, hidden))?;
  let gathered = take_along_axis(token_embeddings, &gather_idx, 1)?;
  squeeze_axes(&gathered, &[1])
}

/// Mask-aware max pooling.
///
/// `max(where(mask == 0, -inf, token_embeddings), axis=1)` — padding
/// positions are forced to `-inf` so they never win the per-dimension
/// maximum over the sequence axis. Mirrors `mlx-embeddings` `max_pooling`
/// and swift `MLXEmbedders` `Pooling` `.max`.
///
/// - `token_embeddings`: `(batch, seq_len, hidden)`
/// - `attention_mask`: `(batch, seq_len)`
/// - returns: `(batch, hidden)`
pub fn max_pooling(token_embeddings: &Array, attention_mask: &Array) -> Result<Array> {
  validate_token_embeddings_and_mask(token_embeddings, attention_mask)?;
  let shape = token_embeddings.shape();
  let emb_dtype = token_embeddings.dtype()?;
  let mask = expand_dims_axes(attention_mask, &[-1])?;
  let mask = broadcast_to(&mask, &shape.as_slice())?;
  // python `mask = ....astype(token_embeddings.dtype)` — NOT f32. The
  // `mask == 0` comparand and the `-float("inf")` in
  // `mx.where(mask == 0, -inf, token_embeddings)` are Python scalars →
  // MLX *weak* scalars adopting the embedding dtype, so a f16/bf16 input
  // is preserved (no f32 promotion). For f32 these are no-op casts →
  // bit-identical to the prior forced-f32 path.
  let mask = astype(&mask, emb_dtype)?;
  let zero = scalar_like(0.0, token_embeddings)?;
  let is_pad = equal(&mask, &zero)?;
  let neg_inf = scalar_like(f32::NEG_INFINITY, token_embeddings)?;
  let masked = select(&is_pad, &neg_inf, token_embeddings)?;
  max_axes(&masked, &[1], false)
}

/// Last-token pooling: select the last real (non-padding) token per
/// sequence — **mask-aware for left- *and* right-padding** (and mixed).
///
/// Mirrors python `mlx-embeddings` `lasttoken_pooling` *exactly*:
///
/// ```text
/// flipped       = attention_mask[:, ::-1]                       # reverse axis 1
/// flip_indices  = argmax(flipped, axis=1)                       # first 1 in the
///                                                               #   reversed row =
///                                                               #   last 1 in the
///                                                               #   original row
/// has_any_real  = max(flipped, axis=1)
/// flip_indices  = where(has_any_real == 0, seq_len - 1, flip_indices)
/// last_indices  = seq_len - flip_indices - 1
/// gather (token_embeddings * mask) at last_indices along axis 1
/// ```
///
/// `argmax` returns the *first* maximal index, so on the reversed mask it
/// is the first `1` from the right of the original row — i.e. the index of
/// the **last non-pad token**, correct under any padding side. The
/// all-padding row falls back to `seq_len - 1`, and the trailing
/// `* attention_mask` (python parity) zeroes that fully-masked row.
///
/// The reversal is done by a materialized index gather (`take_axis` with a
/// `[seq_len-1, …, 0]` index), **not** a strided `slice` view (the prior
/// strided-view fix), so the result is numerically identical to python's
/// `attention_mask[:, ::-1]`.
///
/// Right-padded inputs select the same index as the legacy
/// `sum(mask)-1` formula (regression-safe); only left/mixed-padded rows
/// change — to the corrected python-parity value.
///
/// - `token_embeddings`: `(batch, seq_len, hidden)`
/// - `attention_mask`: `(batch, seq_len)`
/// - returns: `(batch, hidden)`
pub fn last_token_pooling(token_embeddings: &Array, attention_mask: &Array) -> Result<Array> {
  validate_token_embeddings_and_mask(token_embeddings, attention_mask)?;
  let shape = token_embeddings.shape();
  let batch = shape[0];
  let seq_len = shape[1];
  let hidden = shape[2];

  // python `flipped = attention_mask[:, ::-1]`. Avoid a strided `slice`
  // view (prior strided-view fix): materialize the reversal via a
  // `take_axis` with the descending index `[seq_len-1, …, 1, 0]` along
  // the sequence axis — numerically identical, contiguous result.
  let mask_i32 = astype(attention_mask, Dtype::I32)?;
  let mut rev_idx: Vec<i32> = try_with_capacity(seq_len)?;
  rev_idx.extend((0..seq_len as i32).rev());
  let rev_idx = Array::from_slice(&rev_idx, &(seq_len,))?;
  let flipped = take_axis(&mask_i32, &rev_idx, 1)?;

  // python `flip_indices = argmax(flipped, axis=1)` (first max index =
  // first `1` from the right of the original row = last non-pad token).
  let flip_indices = astype(&argmax(&flipped, Some(1), false)?, Dtype::I32)?;

  // python all-pad fallback: `where(max(flipped, axis=1) == 0,
  // seq_len-1, flip_indices)`.
  let has_any_real = max_axes(&flipped, &[1], false)?;
  let zero = Array::full::<i32>(&(1,), 0.0)?;
  let is_all_pad = equal(&has_any_real, &zero)?;
  let seq_len_m1 = Array::full::<i32>(&(1,), (seq_len as i32 - 1) as f32)?;
  let flip_indices = select(&is_all_pad, &seq_len_m1, &flip_indices)?;

  // python `last_indices = seq_len - flip_indices - 1`.
  let seq_len_arr = Array::full::<i32>(&(1,), seq_len as f32)?;
  let one = Array::full::<i32>(&(1,), 1.0)?;
  let last_indices = subtract(&subtract(&seq_len_arr, &flip_indices)?, &one)?;

  // python gathers `token_embeddings * mask` (so the all-pad fallback
  // row, pointing at a pad position, contributes zeros).
  let mask = expand_dims_axes(attention_mask, &[-1])?;
  let mask = broadcast_to(&mask, &shape.as_slice())?;
  let mask = astype(&mask, token_embeddings.dtype()?)?;
  let masked = multiply(token_embeddings, &mask)?;

  let gather_idx = expand_dims_axes(&last_indices, &[1, 2])?;
  let gather_idx = broadcast_to(&gather_idx, &(batch, 1, hidden))?;
  let gathered = take_along_axis(&masked, &gather_idx, 1)?;
  squeeze_axes(&gathered, &[1])
}

/// First-token pooling: strictly token-0, ignoring the mask.
///
/// `token_embeddings[:, 0, :]`. This is the swift `MLXEmbedders`
/// `Pooling` `.first` path (`hiddenStates[0..., 0, 0...]`), distinct from
/// [`cls_pooling`] which `argmax`-finds the first *real* token (python
/// `cls_pooling`, robust to left-padding). Used by the dispatcher's
/// [`PoolingStrategy::First`] only — [`PoolingStrategy::Cls`] now routes
/// to the mask-aware [`cls_pooling`] (python / ST `pooling_mode_cls_token`
/// semantics; see [`pool`] / [`PoolingStrategy::Cls`] docs).
///
/// - `token_embeddings`: `(batch, seq_len, hidden)`
/// - returns: `(batch, hidden)`
pub fn first_token_pooling(token_embeddings: &Array) -> Result<Array> {
  validate_token_embeddings_rank3(token_embeddings)?;
  // `take(axis=1, indices=[0])` → contiguous `(batch, 1, hidden)` gather
  // (a `slice` view would be strided / non-materializable), then squeeze.
  let zero = Array::from_slice(&[0_i32], &(1,))?;
  let gathered = take_axis(token_embeddings, &zero, 1)?;
  squeeze_axes(&gathered, &[1])
}

/// Pooling strategy selector for [`pool`].
///
/// Mirrors swift `MLXEmbedders` `Pooling.Strategy` and python
/// `mlx-embeddings` `pool_by_config` modes (`cls`/`mean`/`max`/
/// `lasttoken`), unified into one enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
pub enum PoolingStrategy {
  /// Mask-aware mean ([`mean_pooling`]); swift `.mean`, python `"mean"`.
  Mean,
  /// CLS / classification token — **mask-aware** ([`cls_pooling`]):
  /// the first *real* (non-padding) token via `argmax(attention_mask)`,
  /// correct under left- *and* right-padding. This mirrors python
  /// `mlx-embeddings` `pool_by_config` mode `"cls"` → `cls_pooling`, and
  /// the `sentence-transformers` `pooling_mode_cls_token` /
  /// `pooling_mode: "cls"` resolution (the primary reference). It is the
  /// strategy the ST-config parser maps the CLS key(s) to. Distinct from
  /// [`First`](Self::First) (strict token-0); swift `Pooling` `.cls`
  /// without an in-scope model `pooledOutput` degrades to token-0, but
  /// the python/ST mask-aware behavior is the correct, padding-robust
  /// one and is what this strategy implements.
  Cls,
  /// First token, strictly position 0 ([`first_token_pooling`]),
  /// **ignoring the mask**; swift `MLXEmbedders` `Pooling` `.first`
  /// (`hiddenStates[0..., 0, 0...]`) and the token-0 fallback. Distinct
  /// from [`Cls`](Self::Cls) (mask-aware first *real* token).
  First,
  /// Last *real* (non-padding) token ([`last_token_pooling`]); swift
  /// `.last`, python `"lasttoken"`. **Mask-aware for left- *and*
  /// right-padding** (and mixed) — python `lasttoken_pooling` parity
  /// (reversed-`argmax`), not the right-pad-only `sum(mask)-1`.
  Last,
  /// Mask-aware max ([`max_pooling`]); swift `.max`, python `"max"`.
  Max,
  /// Passthrough — return the hidden states unchanged. Swift `.none`
  /// (with no `pooledOutput` in scope). The result keeps the input
  /// `(batch, seq_len, hidden)` rank.
  None,
}

impl PoolingStrategy {
  /// The lowercase canonical string name for this strategy, matching the
  /// python `pool_by_config` mode strings and swift `Pooling.Strategy`
  /// display names: `"mean"` / `"cls"` / `"first"` / `"last"` / `"max"` /
  /// `"none"`.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Mean => "mean",
      Self::Cls => "cls",
      Self::First => "first",
      Self::Last => "last",
      Self::Max => "max",
      Self::None => "none",
    }
  }

  /// Parse a `sentence-transformers` / python `pool_by_config` mode
  /// string. Accepts `"cls"`, `"mean"`, `"max"`, `"lasttoken"` (python
  /// `_SUPPORTED_POOL_MODES`), plus `"first"` and `"none"` (swift
  /// strategy names). `"cls"` maps to the mask-aware
  /// [`Cls`](Self::Cls) (python `cls_pooling` / ST
  /// `pooling_mode_cls_token` semantics), **not** the strict-token-0
  /// [`First`](Self::First). Unknown / known-unsupported modes
  /// (`"weightedmean"`, `"mean_sqrt_len_tokens"`) are rejected, matching
  /// python `pool_by_config`'s `NotImplementedError`/`ValueError`.
  pub fn from_mode(mode: &str) -> Result<Self> {
    match mode {
      "cls" => Ok(Self::Cls),
      "mean" => Ok(Self::Mean),
      "max" => Ok(Self::Max),
      "lasttoken" | "last" => Ok(Self::Last),
      "first" => Ok(Self::First),
      "none" => Ok(Self::None),
      "weightedmean" | "mean_sqrt_len_tokens" => Err(Error::Backend {
        message: format!(
          "pooling mode {mode:?} is not supported (supported: cls, lasttoken, max, mean)"
        ),
      }),
      other => Err(Error::Backend {
        message: format!("unknown pooling mode {other:?} (supported: cls, lasttoken, max, mean)"),
      }),
    }
  }
}

/// Unified pooling dispatcher — mirrors python `pool_by_config` +
/// swift `Pooling.callAsFunction`.
///
/// Pipeline (matching swift `Pooling.callAsFunction`'s order):
/// 1. pool by `strategy` (mask-aware where relevant);
/// 2. if `apply_layer_norm`, fused `LayerNorm` (eps [`LAYER_NORM_EPS`] =
///    `1e-5`, swift `MLXFast.layerNorm`); else if `apply_rms_norm`,
///    fused `RMSNorm` (eps [`RMS_NORM_EPS`]) — at most one applies, with
///    LayerNorm taking precedence if both are set;
/// 3. if `dimension` is `Some(d)`, matryoshka-truncate the last axis to
///    `d` (swift `pooled[0..., 0 ..< dimension]`);
/// 4. if `normalize`, L2-normalize (python `normalize_embeddings` eps
///    `1e-9`).
///
/// [`PoolingStrategy::Cls`] dispatches to the **mask-aware**
/// [`cls_pooling`] (python `pool_by_config` `"cls"` / ST
/// `pooling_mode_cls_token`: first *real* token, padding-robust), while
/// [`PoolingStrategy::First`] dispatches to [`first_token_pooling`]
/// (strict token-0, swift `.first`) — these are two distinct strategies,
/// only the latter ignores the mask.
///
/// [`PoolingStrategy::None`] returns the hidden states unchanged and
/// *skips pooling* but still honors layer/rms-norm, `dimension`
/// (last-axis truncation), and `normalize` on the `(batch, seq, hidden)`
/// tensor.
///
/// - `token_embeddings`: `(batch, seq_len, hidden)`
/// - `attention_mask`: `(batch, seq_len)` (unused by `First`/`None`)
/// - `strategy`: which pooling to apply
/// - `normalize`: L2-normalize the result (swift `normalize:`)
/// - `dimension`: optional matryoshka last-dim truncation (swift
///   `Pooling.dimension`)
/// - `apply_layer_norm`: pre-truncation fused LayerNorm (swift
///   `applyLayerNorm:`)
/// - `apply_rms_norm`: pre-truncation fused RMSNorm (mlx-c-surfaced
///   variant; ignored if `apply_layer_norm` is also `true`)
pub fn pool(
  token_embeddings: &Array,
  attention_mask: &Array,
  strategy: PoolingStrategy,
  normalize: bool,
  dimension: Option<usize>,
  apply_layer_norm: bool,
  apply_rms_norm: bool,
) -> Result<Array> {
  let pooled = match strategy {
    PoolingStrategy::Mean => mean_pooling(token_embeddings, attention_mask)?,
    PoolingStrategy::Max => max_pooling(token_embeddings, attention_mask)?,
    PoolingStrategy::Last => last_token_pooling(token_embeddings, attention_mask)?,
    PoolingStrategy::Cls => cls_pooling(token_embeddings, attention_mask)?,
    PoolingStrategy::First => first_token_pooling(token_embeddings)?,
    PoolingStrategy::None => token_embeddings.try_clone()?,
  };

  pool_post(
    pooled,
    normalize,
    dimension,
    apply_layer_norm,
    apply_rms_norm,
  )
}

/// Apply the post-pooling tail of [`pool`] to an **already-pooled** vector:
/// fused LayerNorm/RMSNorm → matryoshka `dimension` truncation → L2-normalize,
/// in swift `Pooling.callAsFunction`'s order (steps 2–4 of [`pool`]).
///
/// This is the shared tail [`pool`] runs after the strategy reduction. It is
/// factored out so callers that already hold a pooled vector — notably
/// [`encode`](super::encode::encode) when a model emits a trained
/// [`pooled_output`](super::model::EmbeddingModelOutput::pooled_output) for
/// the [`Cls`](PoolingStrategy::Cls) / [`None`](PoolingStrategy::None) paths
/// (swift `inputs.pooledOutput ?? …`) — apply the *same* normalize / dimension
/// / layer-norm steps without re-deriving a pooled vector from hidden states.
///
/// `pooled` is consumed: when no transform applies it is returned unchanged
/// (no copy), matching the by-value `pool` tail.
///
/// - `pooled`: an already-pooled `(batch, hidden)` (or any-rank) vector
/// - `normalize`: L2-normalize the result (swift `normalize:`)
/// - `dimension`: optional matryoshka last-dim truncation (swift
///   `Pooling.dimension`)
/// - `apply_layer_norm`: pre-truncation fused LayerNorm (swift
///   `applyLayerNorm:`)
/// - `apply_rms_norm`: pre-truncation fused RMSNorm (mlx-c-surfaced variant;
///   ignored if `apply_layer_norm` is also `true`)
pub fn pool_post(
  mut pooled: Array,
  normalize: bool,
  dimension: Option<usize>,
  apply_layer_norm: bool,
  apply_rms_norm: bool,
) -> Result<Array> {
  if apply_layer_norm {
    pooled = layer_norm(&pooled, None, None, LAYER_NORM_EPS)?;
  } else if apply_rms_norm {
    pooled = rms_norm(&pooled, None, RMS_NORM_EPS)?;
  }

  if let Some(d) = dimension {
    pooled = truncate_last_dim(&pooled, d)?;
  }

  if normalize {
    pooled = l2_normalize_eps(&pooled, DEFAULT_NORMALIZE_EPS)?;
  }

  Ok(pooled)
}

/// Matryoshka truncation: slice the last axis to its first `dimension`
/// entries (swift `Pooling`: `pooled[0..., 0 ..< dimension]`).
///
/// A no-op (returns a clone) if `dimension >= current last-dim size`.
/// Works for any rank ≥ 1 (so it also covers the
/// [`PoolingStrategy::None`] `(batch, seq, hidden)` passthrough).
pub fn truncate_last_dim(x: &Array, dimension: usize) -> Result<Array> {
  let shape = x.shape();
  let ndim = shape.len();
  if ndim == 0 {
    return x.try_clone();
  }
  let last = shape[ndim - 1];
  if dimension >= last {
    return x.try_clone();
  }
  // `take_along_axis` on the last axis with a broadcast `0..dimension`
  // index → contiguous gather (same pattern as `cls_pooling`; a `slice`
  // or last-axis `take_axis` view would be strided / fail `to_vec` and
  // downstream materialization).
  let mut idx: Vec<i32> = try_with_capacity(dimension)?;
  idx.extend(0..dimension as i32);
  let mut idx_shape = vec![1_usize; ndim];
  idx_shape[ndim - 1] = dimension;
  let indices = Array::from_slice(&idx, &idx_shape.as_slice())?;
  let mut bshape = shape;
  bshape[ndim - 1] = dimension;
  let indices = broadcast_to(&indices, &bshape.as_slice())?;
  take_along_axis(x, &indices, (ndim - 1) as i32)
}
