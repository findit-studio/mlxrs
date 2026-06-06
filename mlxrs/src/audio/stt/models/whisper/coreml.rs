//! The CoreML / Neural-Engine Whisper inference backend — Apple Silicon only.
//!
//! Drives the compiled WhisperKit `.mlmodelc` models (`AudioEncoder` +
//! `TextDecoder`, with the mel computed by mlxrs's own front-end) on the Neural
//! Engine through pure `objc2-core-ml` (no Objective-C shim), implementing the
//! [`WhisperInference`](super::inference::WhisperInference) surface so the
//! existing decode pipeline drives it unchanged. Auto-selected from the
//! checkpoint shape (a directory carrying
//! `AudioEncoder.mlmodelc` + `TextDecoder.mlmodelc`) by
//! [`WhisperModel`](super::model::WhisperModel); the MLX path is the fallback.
//!
//! ## Model I/O contracts (WhisperKit, Float16 `MLMultiArray`)
//!
//! - `AudioEncoder`: in `melspectrogram_features` `[1, n_mels, 1, 3000]` → out
//!   `encoder_output_embeds` `[1, n_audio_state, 1, n_audio_ctx]`.
//! - `TextDecoder` (one step, EXPLICIT-I/O KV cache): in `input_ids` `[1]`
//!   (Int32), `cache_length` `[1]` (Int32), `key_cache` / `value_cache`
//!   `[1, kv_width, 1, max_ctx]`, `kv_cache_update_mask` `[1, max_ctx]`,
//!   `encoder_output_embeds` `[1, n_audio_state, 1, n_audio_ctx]`,
//!   `decoder_key_padding_mask` `[1, max_ctx]`; out `logits` `[1, 1, n_vocab]`,
//!   `key_cache_updates` / `value_cache_updates` `[1, kv_width, 1, 1]`,
//!   `alignment_heads_weights` `[1, n_audio_ctx]`.
//!
//! The caller carries `key_cache` / `value_cache` in and stitches the
//! `*_cache_updates` back each step (writing the new column at `cache_length`
//! and marking it in `kv_cache_update_mask`) — see [`CoreMlKvCache`].
//!
//! ## `Array` at the boundary
//!
//! Every [`WhisperInference`](super::inference::WhisperInference) method speaks
//! in MLX [`Array`]. The CoreML graph runs off-device on host Float16 tensors,
//! so each method converts at its own boundary: encoder states and logits cross
//! as host `f16` ↔ [`Array`] (built through the [`half::f16`]
//! [`Element`](crate::Dtype) impl, so no manual codec is needed on the `Array`
//! side), while the explicit KV cache stays host-side in [`CoreMlKvCache`].
//! Logits are returned as an `f32` [`Array`] (the trait's "cast to f32"
//! contract), built directly at `f32` from the decoded host values.
//!
//! ## Word timestamps
//!
//! WhisperKit's decoder emits a single pre-stacked `alignment_heads_weights`
//! `[1, n_audio_ctx]` per step (the alignment-head selection + averaging are
//! baked into the model), which does not match the MLX per-layer-per-head
//! `cross_qk` `(1, n_text_head, T, n_audio_ctx)` shape the word-timestamp DTW
//! ([`super::timing`]) consumes. Adapting that format is a documented follow-up;
//! the two cross-`qk` trait methods therefore return a typed
//! [`Error::InvariantViolation`] on this backend (word timestamps fall back to
//! the MLX path), while the default
//! transcription path — which uses only `encode` + `decode_tokens` /
//! `decode_token_lazy` — runs fully on the Neural Engine.

// `MLMultiArray::dataPointer` is marked deprecated upstream in favour of the
// closure-based `getMutableBytesWithHandler:`, but direct contiguous access is
// the simplest correct path here and is what the bindings still expose. The
// WhisperKit tensors are all first-major contiguous, so a single contiguous
// span is exactly the backing store.
#![allow(deprecated)]

use std::path::Path;

use half::f16;
use objc2::{
  AllocAnyThread,
  rc::Retained,
  runtime::{AnyObject, ProtocolObject},
};
use objc2_core_ml::{
  MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
  MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSNumber, NSString, NSURL};

use crate::{
  Array, Error, Result,
  error::{CoreMlPayload, InvariantViolationPayload},
};

use super::{
  config::{AlignmentHeads, ModelDimensions},
  model::WhisperDecodeCache,
};

/// The maximum decoder context the WhisperKit explicit-cache `TextDecoder`
/// carries (the `key_cache` / `value_cache` last dimension, and the mask
/// width). All WhisperKit Whisper exports use `224`.
const MAX_DECODER_CTX: usize = 224;

// ───────────────────────────── f16 host codec ─────────────────────────────

/// Validate then decode an output [`MLMultiArray`]'s Float16 backing store into
/// a host `u16`-bits `Vec` of exactly `expected_count` scalars, in row-major
/// order of its logical shape.
///
/// A `TextDecoder` / `AudioEncoder` output whose `dataType` or extent differs
/// from the model's declared contract would otherwise be misread (wrong dtype
/// reinterpreted as `u16`) or read out of bounds. This accessor rejects every
/// such mismatch as a typed [`Error::CoreMl`] BEFORE touching the pointer, and
/// allocates the result fallibly (`try_reserve`, never an infallible
/// `with_capacity` on an attacker-influenced `count`):
///
/// - the [`MLMultiArrayDataType`] is `Float16`;
/// - the element [`count`](MLMultiArray::count) equals `expected_count`.
///
/// The store's [`strides`](MLMultiArray::strides) are HONORED, not assumed
/// compact: the Neural Engine returns alignment-PADDED outputs (e.g. a trailing
/// extent of 1500 carried with a 1504-element row stride), so a flat
/// `[0, count)` walk would read padding as data. A compact (row-major) store
/// takes the fast flat path; a padded/strided one is gathered element-by-element
/// honoring its strides. Both yield the logical elements in row-major order.
fn read_f16_checked(
  context: &'static str,
  a: &MLMultiArray,
  expected_count: usize,
) -> Result<Vec<u16>> {
  // SAFETY: `dataType` is a plain readonly property.
  let dtype = unsafe { a.dataType() };
  if dtype != MLMultiArrayDataType::Float16 {
    return Err(Error::CoreMl(CoreMlPayload::new(
      context,
      smol_str::format_smolstr!(
        "expected a Float16 MLMultiArray output, got dataType {}",
        dtype.0
      ),
    )));
  }
  let n = ml_count(a);
  if n != expected_count {
    return Err(Error::CoreMl(CoreMlPayload::new(
      context,
      smol_str::format_smolstr!("expected {expected_count} elements, got {n}"),
    )));
  }
  let shape = shape_of(a);
  let strides = strides_of(a);
  let mut out = Vec::new();
  crate::model_validation::reserve_or_error(&mut out, context, n)?;
  if is_row_major_contiguous(&shape, &strides) {
    // SAFETY: validated above — `dataPointer` is the contiguous first-major
    // backing store of exactly `n == expected_count` Float16 scalars (dtype +
    // row-major strides + count all checked), so reading `[0, n)` as `u16` is in
    // bounds and type-correct. The pointer outlives the loop (`a` is borrowed).
    unsafe {
      let p = a.dataPointer().as_ptr().cast::<u16>();
      for i in 0..n {
        out.push(p.add(i).read());
      }
    }
  } else {
    // Padded / strided store (ANE alignment): gather the `n` logical elements
    // honoring `strides`. `strided_extent` is the physical span the descriptor
    // addresses (largest in-shape offset + 1), validated against overflow and the
    // `from_raw_parts` size limit before it lengths the unsafe slice below.
    let extent = strided_extent(context, &shape, &strides)?;
    // SAFETY: `dataPointer` is the Float16 backing store; the MLMultiArray
    // contract guarantees it backs the descriptor's full strided extent, and
    // `strided_extent` bounded `extent` to the `from_raw_parts` size limit, so a
    // `[0, extent)` `u16` slice is in bounds and type-correct (dtype checked).
    // The slice borrows through `a`, which outlives this block.
    let backing =
      unsafe { std::slice::from_raw_parts(a.dataPointer().as_ptr().cast::<u16>(), extent) };
    strided_gather_u16(backing, &shape, &strides, n, &mut out);
  }
  // A correct read yields exactly `n` elements. A short gather means a malformed
  // strided descriptor (an offset fell outside the computed extent) — surface a
  // typed error instead of returning a silently-truncated vector.
  if out.len() != n {
    return Err(Error::CoreMl(CoreMlPayload::new(
      context,
      smol_str::format_smolstr!(
        "MLMultiArray read produced {} of {} expected elements (malformed strided descriptor)",
        out.len(),
        n
      ),
    )));
  }
  Ok(out)
}

/// As [`read_f16_checked`], decoding straight to host `f32` (the logits /
/// encoder-state path, which needs the value not the raw bits).
fn read_f16_checked_f32(
  context: &'static str,
  a: &MLMultiArray,
  expected_count: usize,
) -> Result<Vec<f32>> {
  let bits = read_f16_checked(context, a, expected_count)?;
  Ok(bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
}

/// Number of scalar elements in an [`MLMultiArray`] (clamped to `>= 0`).
fn ml_count(a: &MLMultiArray) -> usize {
  // SAFETY: `count` is a plain readonly property.
  unsafe { a.count() }.max(0) as usize
}

/// The shape of an [`MLMultiArray`] as a `usize` vector.
fn shape_of(a: &MLMultiArray) -> Vec<usize> {
  // SAFETY: `shape` is a readonly `NSArray<NSNumber>` property.
  let arr = unsafe { a.shape() };
  let mut dims = Vec::with_capacity(arr.count());
  for i in 0..arr.count() {
    dims.push(arr.objectAtIndex(i).integerValue().max(0) as usize);
  }
  dims
}

/// The element strides of an [`MLMultiArray`] as a `usize` vector (clamped to
/// `>= 0`).
fn strides_of(a: &MLMultiArray) -> Vec<usize> {
  // SAFETY: `strides` is a readonly `NSArray<NSNumber>` property.
  let arr = unsafe { a.strides() };
  let mut s = Vec::with_capacity(arr.count());
  for i in 0..arr.count() {
    s.push(arr.objectAtIndex(i).integerValue().max(0) as usize);
  }
  s
}

/// Whether `strides` are the compact first-major (row-major) strides of `shape`
/// — i.e. `strides[k] == product(shape[k+1..])`, with the last stride `1`. A
/// compact array's element `i` then lives at flat offset `i`, so a contiguous
/// `dataPointer` walk is valid.
fn is_row_major_contiguous(shape: &[usize], strides: &[usize]) -> bool {
  if shape.len() != strides.len() {
    return false;
  }
  let mut expected = 1usize;
  for k in (0..shape.len()).rev() {
    if strides[k] != expected {
      return false;
    }
    // `shape` came from the same array (every extent fits the element count,
    // itself an `NSInteger`), so this product cannot overflow `usize`.
    expected = expected.saturating_mul(shape[k]);
  }
  true
}

/// The physical element span a strided `MLMultiArray` descriptor addresses — one
/// past the largest in-shape offset, `1 + sum_k (shape[k] - 1) * strides[k]` — or
/// a typed error if the descriptor is malformed (shape / strides rank mismatch)
/// or the span would overflow or exceed the `from_raw_parts` size limit. A zero
/// extent (some axis is empty) is `Ok(0)`; a scalar / empty shape is `Ok(1)`.
///
/// The result bounds an `unsafe` `slice::from_raw_parts` in [`read_f16_checked`],
/// so it MUST be exact and in range: the offset arithmetic uses CHECKED ops (a
/// hostile or version-skewed stride descriptor cannot wrap it), and the final
/// extent is rejected unless its byte size `extent * size_of::<u16>()` fits in
/// `isize::MAX` (the safety precondition of `from_raw_parts`).
fn strided_extent(context: &'static str, shape: &[usize], strides: &[usize]) -> Result<usize> {
  if shape.len() != strides.len() {
    return Err(Error::CoreMl(CoreMlPayload::new(
      context,
      smol_str::format_smolstr!(
        "MLMultiArray shape rank {} does not match strides rank {}",
        shape.len(),
        strides.len()
      ),
    )));
  }
  if shape.contains(&0) {
    return Ok(0);
  }
  // Every extent is >= 1 here (the zero-axis case returned above), so `d - 1`
  // cannot underflow; the multiply and the running sum are checked.
  let mut max_off = 0usize;
  for (&d, &s) in shape.iter().zip(strides) {
    let term = (d - 1)
      .checked_mul(s)
      .ok_or_else(|| strided_extent_overflow(context))?;
    max_off = max_off
      .checked_add(term)
      .ok_or_else(|| strided_extent_overflow(context))?;
  }
  max_off
    .checked_add(1)
    .filter(|&extent| extent <= (isize::MAX as usize) / std::mem::size_of::<u16>())
    .ok_or_else(|| strided_extent_overflow(context))
}

/// Typed error for a [`strided_extent`] whose offset arithmetic overflows or
/// whose span exceeds the addressable `from_raw_parts` size limit.
fn strided_extent_overflow(context: &'static str) -> Error {
  Error::CoreMl(CoreMlPayload::new(
    context,
    smol_str::format_smolstr!("MLMultiArray strided extent overflows the addressable range"),
  ))
}

/// Gather the `n` logical elements of row-major `shape` from a (possibly
/// padded/strided) flat `backing` store into `out` (already reserved for `n`),
/// honoring `strides`. The element at multi-index `idx` is `backing[sum_k
/// idx[k] * strides[k]]`; indices advance in row-major order (last axis
/// fastest). Every computed offset is `< strided_extent(shape, strides) ==
/// backing.len()`, so the indexing is always in bounds. A compact store is a
/// straight prefix copy.
fn strided_gather_u16(
  backing: &[u16],
  shape: &[usize],
  strides: &[usize],
  n: usize,
  out: &mut Vec<u16>,
) {
  if is_row_major_contiguous(shape, strides) {
    out.extend_from_slice(&backing[..n.min(backing.len())]);
    return;
  }
  let d = shape.len();
  let mut idx = vec![0usize; d];
  for _ in 0..n {
    let off = idx
      .iter()
      .zip(strides)
      .map(|(&i, &s)| i.saturating_mul(s))
      .fold(0usize, usize::saturating_add);
    // `off <= max in-shape offset < backing.len()` by construction; `get`
    // keeps a malformed descriptor from panicking (it stops the gather instead).
    match backing.get(off) {
      Some(&v) => out.push(v),
      None => return,
    }
    // Row-major odometer: advance the last axis fastest, carrying up.
    for k in (0..d).rev() {
      idx[k] += 1;
      if idx[k] < shape[k] {
        break;
      }
      idx[k] = 0;
    }
  }
}

// ──────────────────────── MLMultiArray construction ────────────────────────

/// Build an `NSArray<NSNumber>` shape descriptor from a `usize` slice.
fn shape_array(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
  let nums: Vec<Retained<NSNumber>> = dims
    .iter()
    .map(|&d| NSNumber::new_isize(d as isize))
    .collect();
  NSArray::from_retained_slice(&nums)
}

/// Allocate an uninitialized [`MLMultiArray`] of the given shape + dtype.
fn new_multi_array(
  context: &'static str,
  dims: &[usize],
  dtype: MLMultiArrayDataType,
) -> Result<Retained<MLMultiArray>> {
  let shape = shape_array(dims);
  // SAFETY: `shape` is a valid `NSArray<NSNumber>`; `dtype` is a valid enum
  // constant. The returned array's contents are uninitialized — every caller
  // fills the whole store before handing it to a prediction. The `error:`
  // out-param surfaces as the `Err` arm.
  unsafe { MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &shape, dtype) }
    .map_err(|e| coreml_err(context, &e))
}

/// A Float16 [`MLMultiArray`] of `dims`, filled from a host `f32` slice (encoded
/// to Float16). `values.len()` must equal the product of `dims`.
fn f16_array_from_f32(
  context: &'static str,
  dims: &[usize],
  values: &[f32],
) -> Result<Retained<MLMultiArray>> {
  let a = new_multi_array(context, dims, MLMultiArrayDataType::Float16)?;
  let n = ml_count(&a);
  debug_assert_eq!(n, values.len(), "{context}: element count mismatch");
  // SAFETY: `dataPointer` is the contiguous Float16 backing store of `n`
  // scalars we just allocated; we write within `[0, min(n, len))`. `f16::to_bits`
  // is a pure bit reinterpretation.
  unsafe {
    let p = a.dataPointer().as_ptr().cast::<u16>();
    for (i, &v) in values.iter().take(n).enumerate() {
      p.add(i).write(f16::from_f32(v).to_bits());
    }
  }
  Ok(a)
}

/// A Float16 [`MLMultiArray`] of `dims`, filled from a host `f16`-bits slice
/// (the raw cache store — no re-encode). `bits.len()` must equal the product of
/// `dims`.
fn f16_array_from_bits(
  context: &'static str,
  dims: &[usize],
  bits: &[u16],
) -> Result<Retained<MLMultiArray>> {
  let a = new_multi_array(context, dims, MLMultiArrayDataType::Float16)?;
  let n = ml_count(&a);
  debug_assert_eq!(n, bits.len(), "{context}: element count mismatch");
  // SAFETY: contiguous Float16 store of `n` scalars; write within `[0, n)`.
  unsafe {
    let p = a.dataPointer().as_ptr().cast::<u16>();
    for (i, &b) in bits.iter().take(n).enumerate() {
      p.add(i).write(b);
    }
  }
  Ok(a)
}

/// An Int32 [`MLMultiArray`] of `dims`, filled from a host `i32` slice.
fn i32_array(
  context: &'static str,
  dims: &[usize],
  values: &[i32],
) -> Result<Retained<MLMultiArray>> {
  let a = new_multi_array(context, dims, MLMultiArrayDataType::Int32)?;
  let n = ml_count(&a);
  // SAFETY: contiguous Int32 store of `n` scalars; write within `[0, min(n, len))`.
  unsafe {
    let p = a.dataPointer().as_ptr().cast::<i32>();
    for (i, &v) in values.iter().take(n).enumerate() {
      p.add(i).write(v);
    }
  }
  Ok(a)
}

// ───────────────────────────── model loading ───────────────────────────────

/// A `file://` `NSURL` for a `.mlmodelc` directory.
fn dir_url(path: &Path) -> Retained<NSURL> {
  let s = NSString::from_str(&path.to_string_lossy());
  NSURL::fileURLWithPath_isDirectory(&s, true)
}

/// Load a `.mlmodelc` on the `CPUAndNeuralEngine` compute units (the ANE-eligible
/// configuration).
fn load_model(context: &'static str, path: &Path) -> Result<Retained<MLModel>> {
  let url = dir_url(path);
  // SAFETY: `MLModelConfiguration::new` returns a fresh config; `setComputeUnits`
  // takes a valid enum constant. Both are plain property mutations.
  let config = unsafe {
    let c = MLModelConfiguration::new();
    c.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
    c
  };
  // SAFETY: `url` points at an existing `.mlmodelc` directory; `config` is a
  // freshly constructed configuration. The `error:` out-param surfaces as `Err`.
  unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
    .map_err(|e| coreml_err(context, &e))
}

/// Wrap `(name, MLMultiArray)` input pairs into an `MLDictionaryFeatureProvider`.
fn feature_provider(
  context: &'static str,
  pairs: &[(&str, &MLMultiArray)],
) -> Result<Retained<ProtocolObject<dyn MLFeatureProvider>>> {
  let keys: Vec<Retained<NSString>> = pairs.iter().map(|(k, _)| NSString::from_str(k)).collect();
  // SAFETY: `featureValueWithMultiArray:` wraps a valid `MLMultiArray` into a
  // feature value; the array outlives the call.
  let vals: Vec<Retained<MLFeatureValue>> = pairs
    .iter()
    .map(|(_, a)| unsafe { MLFeatureValue::featureValueWithMultiArray(a) })
    .collect();

  let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
  // `MLFeatureValue` is an `NSObject` subclass, so each value coerces to
  // `&AnyObject` via its `AsRef` chain.
  let val_refs: Vec<&AnyObject> = vals.iter().map(|v| v.as_ref()).collect();
  let dict = objc2_foundation::NSDictionary::from_slices(&key_refs, &val_refs);

  // SAFETY: every value in `dict` is an `MLFeatureValue` (a valid feature
  // value), satisfying `initWithDictionary:`'s contract.
  unsafe {
    MLDictionaryFeatureProvider::initWithDictionary_error(
      MLDictionaryFeatureProvider::alloc(),
      &dict,
    )
  }
  .map(ProtocolObject::from_retained)
  .map_err(|e| coreml_err(context, &e))
}

/// Run a prediction and pull a single named Float16 output out of the result.
fn predict_output(
  context: &'static str,
  model: &MLModel,
  inputs: &ProtocolObject<dyn MLFeatureProvider>,
  output: &str,
) -> Result<Retained<MLMultiArray>> {
  // SAFETY: `inputs` satisfies the model's declared input schema (built by the
  // caller from that schema); the `error:` out-param surfaces as `Err`.
  let out =
    unsafe { model.predictionFromFeatures_error(inputs) }.map_err(|e| coreml_err(context, &e))?;
  let key = NSString::from_str(output);
  // SAFETY: `featureValueForName:` returns nil (→ `None`) for an unknown name.
  let fv =
    unsafe { out.featureValueForName(&key) }.ok_or_else(|| coreml_missing(context, output))?;
  // SAFETY: reading the multi-array payload of the feature value; returns nil
  // (→ `None`) if it is not a multi-array.
  unsafe { fv.multiArrayValue() }.ok_or_else(|| coreml_missing(context, output))
}

// ───────────────────────────── error helpers ───────────────────────────────

/// A typed CoreML error from an `NSError` out-param.
fn coreml_err(context: &'static str, err: &objc2_foundation::NSError) -> Error {
  let detail = err.localizedDescription().to_string();
  Error::CoreMl(CoreMlPayload::new(context, detail))
}

/// A typed CoreML error for a missing / wrong-typed output feature.
fn coreml_missing(context: &'static str, output: &str) -> Error {
  Error::CoreMl(CoreMlPayload::new(
    context,
    smol_str::format_smolstr!("output '{output}' missing or not a MultiArray"),
  ))
}

/// The typed "not supported on the CoreML backend" error for the word-timestamp
/// cross-`qk` paths (a documented follow-up — see the module docs).
fn cross_qk_unsupported(context: &'static str) -> Error {
  Error::InvariantViolation(InvariantViolationPayload::new(
    context,
    "word-timestamp cross_qk is not supported on the CoreML/ANE backend yet \
     (WhisperKit emits a pre-stacked alignment_heads_weights, not per-layer \
     per-head cross attention); transcribe without word_timestamps, or use the \
     MLX backend for word timings",
  ))
}

// ────────────────────────────── KV cache ───────────────────────────────────

/// The explicit host-side decoder KV cache the WhisperKit `TextDecoder` threads
/// — the [`CoreMlWhisper`] [`Cache`](super::inference::WhisperInference::Cache).
///
/// Carries the `key_cache` / `value_cache` `[1, kv_width, 1, MAX_DECODER_CTX]`
/// tensors (as raw Float16 bits, the decoder's native dtype — no f32 round-trip)
/// and the running `cache_length` (the number of columns already written). Each
/// decode step writes the model's `*_cache_updates` `[1, kv_width, 1, 1]` column
/// at index `cache_length` and bumps the length, mirroring WhisperKit's
/// `kv_cache_update_mask` + `cache_length` bookkeeping.
pub struct CoreMlKvCache {
  /// `kv_width = n_text_layer * 2 * (n_audio_state / n_text_head) * n_text_head`
  /// — the decoder's stacked key/value channel width (e.g. `1536` for tiny).
  kv_width: usize,
  /// `[kv_width * MAX_DECODER_CTX]` row-major over `(kv_width, MAX_DECODER_CTX)`
  /// (batch + the singleton penultimate dim are 1), raw Float16 bits.
  key_cache: Vec<u16>,
  /// As [`Self::key_cache`], for the value cache.
  value_cache: Vec<u16>,
  /// Number of columns already written (`0 <= cache_length <= MAX_DECODER_CTX`).
  cache_length: usize,
}

impl CoreMlKvCache {
  /// A fresh, all-zero cache for a decoder of stacked width `kv_width`.
  fn new(kv_width: usize) -> Result<Self> {
    let plane = kv_width
      .checked_mul(MAX_DECODER_CTX)
      .ok_or_else(|| dim_overflow("CoreMlKvCache: kv_width * max_ctx"))?;
    let mut key_cache = Vec::new();
    crate::model_validation::reserve_or_error(&mut key_cache, "CoreMlKvCache: key_cache", plane)?;
    key_cache.resize(plane, 0u16);
    let mut value_cache = Vec::new();
    crate::model_validation::reserve_or_error(
      &mut value_cache,
      "CoreMlKvCache: value_cache",
      plane,
    )?;
    value_cache.resize(plane, 0u16);
    Ok(Self {
      kv_width,
      key_cache,
      value_cache,
      cache_length: 0,
    })
  }

  /// Stitch one step's `key_cache_updates` / `value_cache_updates`
  /// `[1, kv_width, 1, 1]` columns into the cache at column `cache_length`, then
  /// advance the length by `n_new` (the number of tokens decoded this step).
  ///
  /// WhisperKit's decoder writes the new key/value column for the freshly
  /// decoded token; the caller places it at the next free slot
  /// (`cache_length`). The update column is a single `(kv_width,)` vector per
  /// step (the decoder advances one token per call), so `n_new == 1` on the warm
  /// path; the explicit-cache decoder cannot ingest a multi-token prefill in one
  /// call, so a `> 1` request is rejected upstream before this is reached.
  fn stitch(&mut self, key_updates: &[u16], value_updates: &[u16], n_new: usize) -> Result<()> {
    if self.cache_length + n_new > MAX_DECODER_CTX {
      return Err(Error::CapExceeded(crate::error::CapExceededPayload::new(
        "CoreMlKvCache: decoder context",
        "max_decoder_ctx",
        MAX_DECODER_CTX as u64,
        (self.cache_length + n_new) as u64,
      )));
    }
    if key_updates.len() != self.kv_width || value_updates.len() != self.kv_width {
      return Err(Error::LengthMismatch(
        crate::error::LengthMismatchPayload::new(
          "CoreMlKvCache: cache update width",
          self.kv_width,
          key_updates.len().min(value_updates.len()),
        ),
      ));
    }
    // Write the new column at `cache_length` for every channel row. The store is
    // row-major over `(kv_width, MAX_DECODER_CTX)`, so channel `c`'s column `col`
    // is at `c * MAX_DECODER_CTX + col`.
    let col = self.cache_length;
    for c in 0..self.kv_width {
      let dst = c * MAX_DECODER_CTX + col;
      self.key_cache[dst] = key_updates[c];
      self.value_cache[dst] = value_updates[c];
    }
    self.cache_length += n_new;
    Ok(())
  }
}

// ───────────────────────────── the backend ─────────────────────────────────

/// The CoreML / Neural-Engine Whisper backend — see the module docs.
///
/// Owns the loaded `AudioEncoder` + `TextDecoder` `MLModel` handles and the
/// checkpoint metadata ([`ModelDimensions`] + word-timing [`AlignmentHeads`]),
/// read from the bundle's `config.json` / `generation_config.json`. Built by
/// [`Self::load`]; auto-selected by [`WhisperModel`](super::model::WhisperModel).
pub struct CoreMlWhisper {
  audio_encoder: Retained<MLModel>,
  text_decoder: Retained<MLModel>,
  dims: ModelDimensions,
  alignment_heads: AlignmentHeads,
  /// The decoder's stacked key/value channel width — read from the decoder's
  /// `key_cache` input description so the cache matches the artifact exactly,
  /// not a derived guess.
  kv_width: usize,
  /// The decoder's maximum KV-cache context (the `key_cache` last dimension /
  /// mask width) — read from the same input description so the decode loop's
  /// context cap matches the artifact exactly. Falls back to [`MAX_DECODER_CTX`]
  /// when the description is unreadable. The cache store itself is sized at the
  /// compile-time [`MAX_DECODER_CTX`]; a future export with a different cap is
  /// surfaced through [`WhisperInference::max_decoder_context`] so the loop stops
  /// at the smaller of the two.
  max_decoder_ctx: usize,
}

impl CoreMlWhisper {
  /// Whether `dir` carries a CoreML Whisper bundle — the
  /// `AudioEncoder.mlmodelc` + `TextDecoder.mlmodelc` pair [`Self::load`]
  /// needs. The mel front-end is computed in mlxrs, so `MelSpectrogram.mlmodelc`
  /// is not required (its presence is harmless).
  pub fn is_present(dir: &Path) -> bool {
    dir.join("AudioEncoder.mlmodelc").is_dir() && dir.join("TextDecoder.mlmodelc").is_dir()
  }

  /// Load the CoreML Whisper backend from a model directory: the
  /// `AudioEncoder` + `TextDecoder` `.mlmodelc` bundles (on the
  /// `CPUAndNeuralEngine` units) and the `config.json` dims +
  /// `generation_config.json` alignment heads.
  ///
  /// # Errors
  /// - [`Error::CoreMl`] if a `.mlmodelc` fails to load;
  /// - [`Error::FileIo`] / [`Error::Parse`] if `config.json` is unreadable /
  ///   malformed;
  /// - propagates [`ModelDimensions::from_dict`] /
  ///   [`AlignmentHeads`] construction errors.
  pub fn load(dir: &Path) -> Result<Self> {
    let audio_encoder = load_model(
      "CoreMlWhisper::load: AudioEncoder.mlmodelc",
      &dir.join("AudioEncoder.mlmodelc"),
    )?;
    let text_decoder = load_model(
      "CoreMlWhisper::load: TextDecoder.mlmodelc",
      &dir.join("TextDecoder.mlmodelc"),
    )?;

    let config = read_json(dir, "config.json")?;
    let dims = ModelDimensions::from_dict(&config)?;

    // The alignment heads default to the last half of the decoder layers; a
    // `generation_config.json` with an `alignment_heads` key overrides it (the
    // same precedence as `WhisperModel::load`).
    let alignment_heads = match read_json_opt(dir, "generation_config.json")? {
      Some(gc) => AlignmentHeads::from_generation_config(&gc, &dims)?
        .unwrap_or_else(|| AlignmentHeads::default_for(&dims)),
      None => AlignmentHeads::default_for(&dims),
    };

    // The key/value caches are SEPARATE `[1, kv_width, 1, max_ctx]` tensors, so
    // `kv_width = n_text_layer * n_audio_state` (one stacked key plane per layer;
    // no ×2). Prefer the width declared by the artifact's `key_cache` input so it
    // always matches the model exactly; fall back to the derived width only if
    // the description is unreadable.
    let (kv_width, key_cache_ctx) = decoder_key_cache_shape(&text_decoder)
      .unwrap_or((dims.n_text_layer() * dims.n_audio_state(), MAX_DECODER_CTX));
    // The cache store is sized at the compile-time `MAX_DECODER_CTX`, so the true
    // usable context is the smaller of that and the artifact's declared cap.
    let max_decoder_ctx = key_cache_ctx.min(MAX_DECODER_CTX);

    Ok(Self {
      audio_encoder,
      text_decoder,
      dims,
      alignment_heads,
      kv_width,
      max_decoder_ctx,
    })
  }

  /// The word-timing alignment heads (consumed by the MLX word-timestamp DTW
  /// when that path is driven; the CoreML cross-`qk` is a follow-up).
  #[inline]
  pub fn alignment_heads(&self) -> &AlignmentHeads {
    &self.alignment_heads
  }

  /// Run the `AudioEncoder` on a host `f32` mel laid out `[1, n_mels, 1, 3000]`
  /// and return the encoder output transposed to the pipeline's
  /// `(1, n_audio_ctx, n_audio_state)` layout as a host `f32` `Vec`.
  fn run_audio_encoder(&self, mel_chw: &[f32]) -> Result<Vec<f32>> {
    let n_mels = self.dims.n_mels();
    let mel_in = f16_array_from_f32(
      "CoreMlWhisper::encode: mel input",
      &[1, n_mels, 1, N_AUDIO_FRAMES],
      mel_chw,
    )?;
    let provider = feature_provider(
      "CoreMlWhisper::encode: feature provider",
      &[("melspectrogram_features", &mel_in)],
    )?;
    let out = predict_output(
      "CoreMlWhisper::encode: AudioEncoder predict",
      &self.audio_encoder,
      &provider,
      "encoder_output_embeds",
    )?;
    // `[1, n_audio_state, 1, n_audio_ctx]` → host f32, then transpose the
    // channel/context axes to `(n_audio_ctx, n_audio_state)`.
    let shape = shape_of(&out);
    let (state, ctx) = (self.dims.n_audio_state(), self.dims.n_audio_ctx());
    if shape != [1, state, 1, ctx] {
      return Err(Error::CoreMl(CoreMlPayload::new(
        "CoreMlWhisper::encode: encoder output shape",
        smol_str::format_smolstr!("expected [1, {state}, 1, {ctx}], got {shape:?}"),
      )));
    }
    // Validate dtype / contiguity / count before the raw read (the shape check
    // above fixes the expected extent `state * ctx`). Row-major over (state, ctx).
    let chw = read_f16_checked_f32(
      "CoreMlWhisper::encode: encoder output read",
      &out,
      state * ctx,
    )?;
    let mut hwc = vec![0.0f32; ctx * state];
    for s in 0..state {
      for t in 0..ctx {
        hwc[t * state + s] = chw[s * ctx + t];
      }
    }
    Ok(hwc)
  }
}

/// The fixed `AudioEncoder` input frame count (30 s of mel at 100 fps).
const N_AUDIO_FRAMES: usize = 3000;

impl super::inference::WhisperInference for CoreMlWhisper {
  type Cache = CoreMlKvCache;

  fn encode(&self, mel: &Array) -> Result<Array> {
    // The pipeline hands a `(num_frames, n_mels)` mel (padded to 3000 frames);
    // the AudioEncoder wants `[1, n_mels, 1, 3000]`. Read the mel to host f32 and
    // transpose `(frames, mels)` → channel-major `(mels, frames)`.
    let n_mels = self.dims.n_mels();
    let shape = mel.shape();
    if shape.len() != 2 || shape[1] != n_mels {
      return Err(Error::RankMismatch(crate::error::RankMismatchPayload::new(
        "CoreMlWhisper::encode: mel must be (num_frames, n_mels)",
        shape.len() as u32,
        shape,
      )));
    }
    let frames = shape[0].min(N_AUDIO_FRAMES);
    let mut mel32 = mel.try_clone()?;
    let host = mel32.to_vec::<f32>()?; // row-major over (frames, mels)
    let mut chw = vec![0.0f32; n_mels * N_AUDIO_FRAMES];
    for t in 0..frames {
      for m in 0..n_mels {
        // dst[m, t] (channel-major, padded width 3000) ← src[t, m].
        chw[m * N_AUDIO_FRAMES + t] = host[t * n_mels + m];
      }
    }
    let enc = self.run_audio_encoder(&chw)?;
    let ctx = i32::try_from(self.dims.n_audio_ctx()).map_err(|_| dim_overflow("n_audio_ctx"))?;
    let state =
      i32::try_from(self.dims.n_audio_state()).map_err(|_| dim_overflow("n_audio_state"))?;
    Array::from_slice::<f32>(&enc, &[1, ctx, state])
  }

  fn decode_tokens(
    &self,
    tokens: &[u32],
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)> {
    self.validate_token_ids("CoreMlWhisper::decode_tokens", tokens)?;
    if tokens.is_empty() {
      return Err(Error::EmptyInput(crate::error::EmptyInputPayload::new(
        "CoreMlWhisper::decode_tokens: tokens",
      )));
    }
    // Fail-fast guard BEFORE any cache allocation, encoder-state materialization,
    // or `T * n_vocab` logits reservation: bound the request against the decoder
    // cache ceiling and require the exact encoder-state shape, so a direct
    // `WhisperBackend` / `WhisperInference` caller (bypassing `DecodingTask`'s
    // clamps) cannot drive those allocations with an over-cap or wrong-shaped
    // request before the per-token `decode_one` would trip the cap.
    self.precheck_decode(cache, tokens.len(), encoder_states)?;
    let t = tokens.len();
    // The WhisperKit decoder advances exactly one token per prediction. A fresh
    // (no-cache) call therefore replays the whole prefix one token at a time,
    // carrying the cache forward; a warm call advances only the new tail. Each
    // step's per-position logits are collected into a `(1, T, n_vocab)` Array —
    // the trait's full-sequence shape — so the pipeline can read both the last
    // position (`last_position_row`) and the `sot_index` position
    // (`no_speech_prob`), exactly as the MLX parallel forward yields.
    let mut cache = match cache {
      Some(c) => clone_cache(c)?,
      None => CoreMlKvCache::new(self.kv_width)?,
    };
    let enc_chw = self.encoder_states_to_chw(encoder_states)?;
    let n_vocab = self.dims.n_vocab();
    let plane = t
      .checked_mul(n_vocab)
      .ok_or_else(|| dim_overflow("CoreMlWhisper::decode_tokens: T * n_vocab"))?;
    let mut all = Vec::new();
    crate::model_validation::reserve_or_error(
      &mut all,
      "CoreMlWhisper::decode_tokens: logits buffer",
      plane,
    )?;
    for &tok in tokens {
      let row = self.decode_one(tok, &enc_chw, &mut cache)?; // (n_vocab,) host f32
      all.extend_from_slice(&row);
    }
    let ti = i32::try_from(t).map_err(|_| dim_overflow("T"))?;
    let vi = i32::try_from(n_vocab).map_err(|_| dim_overflow("n_vocab"))?;
    let logits = Array::from_slice::<f32>(&all, &[1, ti, vi])?;
    Ok((logits, cache))
  }

  fn decode_tokens_batched(
    &self,
    tokens: &[u32],
    n_group: usize,
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)> {
    if n_group == 1 {
      return self.decode_tokens(tokens, encoder_states, cache);
    }
    // Best-of-N decodes `n_group` parallel candidate rows in one forward; the
    // WhisperKit explicit-cache decoder takes a single `input_ids[1]` and cannot
    // batch the candidate dimension. Best-of on the CoreML backend is a
    // documented follow-up.
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "CoreMlWhisper::decode_tokens_batched",
      "best-of-N (n_group > 1) batched decode is not supported on the CoreML/ANE \
       backend (the WhisperKit decoder takes a single token per step); use \
       temperature 0 / best_of = None, or the MLX backend",
    )))
  }

  fn decode_token_lazy(
    &self,
    token: &Array,
    encoder_states: &Array,
    cache: Option<&Self::Cache>,
  ) -> Result<(Array, Self::Cache)> {
    // The MLX backend keeps the next token on-device to avoid a per-step
    // GPU→host readback; the CoreML decode crosses the host boundary every step
    // regardless (the prediction runs off-device), so there is nothing to keep
    // lazy — materialize the token to host ids and forward through `decode_tokens`.
    // Guard BEFORE the host materialization (`to_vec`) so an over-cap or
    // wrong-shaped request is rejected without first copying the token tensor; the
    // element count is read from the shape (no allocation).
    let n_tokens = token
      .shape()
      .iter()
      .copied()
      .fold(1usize, usize::saturating_mul);
    self.precheck_decode(cache, n_tokens, encoder_states)?;
    let mut t = token.try_clone()?;
    let ids = t.to_vec::<u32>()?;
    self.decode_tokens(&ids, encoder_states, cache)
  }

  fn decode_step_with_cross_qk(
    &self,
    _cache: &mut WhisperDecodeCache,
    _enc: &Array,
    _tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)> {
    Err(cross_qk_unsupported(
      "CoreMlWhisper::decode_step_with_cross_qk",
    ))
  }

  fn forward_with_cross_qk(
    &self,
    _mel: &Array,
    _tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)> {
    Err(cross_qk_unsupported("CoreMlWhisper::forward_with_cross_qk"))
  }

  fn broadcast_encoder_states(&self, enc: &Array, n_group: usize) -> Result<Array> {
    if n_group == 1 {
      return enc.try_clone();
    }
    // Only reached on the best-of path, which `decode_tokens_batched` rejects on
    // this backend; surface the same typed reason rather than silently building
    // states the decoder cannot consume.
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "CoreMlWhisper::broadcast_encoder_states",
      "best-of-N (n_group > 1) is not supported on the CoreML/ANE backend",
    )))
  }

  #[inline]
  fn dims(&self) -> &ModelDimensions {
    &self.dims
  }

  #[inline]
  fn max_decoder_context(&self) -> usize {
    // The WhisperKit `TextDecoder` explicit cache caps at its `key_cache` extent
    // (read at load, clamped to the compile-time store size), well below the
    // model's `n_text_ctx` — so the decode loop must stop here, not at 448.
    self.max_decoder_ctx
  }

  #[inline]
  fn alignment_heads(&self) -> &AlignmentHeads {
    &self.alignment_heads
  }

  fn validate_token_ids(&self, context: &'static str, tokens: &[u32]) -> Result<()> {
    let n_vocab = self.dims.n_vocab();
    if let Some(&id) = tokens.iter().find(|&&id| id as usize >= n_vocab) {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        context,
        "token id must be < n_vocab (the decoder token-embedding rows)",
        smol_str::format_smolstr!("id={id}, n_vocab={n_vocab}"),
      )));
    }
    Ok(())
  }
}

impl CoreMlWhisper {
  /// Convert the pipeline's `(1, n_audio_ctx, n_audio_state)` encoder-states
  /// [`Array`] back to the decoder's channel-major `[1, n_audio_state, 1,
  /// n_audio_ctx]` host f32 layout.
  fn encoder_states_to_chw(&self, encoder_states: &Array) -> Result<Vec<f32>> {
    let (ctx, state) = (self.dims.n_audio_ctx(), self.dims.n_audio_state());
    // The exact `(1, n_audio_ctx, n_audio_state)` shape is pre-validated by
    // `precheck_decode` at every decode entry point (before this conversion runs),
    // so the channel-major transpose below can assume it.
    let mut e = encoder_states.try_clone()?;
    let host = e.to_vec::<f32>()?; // row-major over (1, ctx, state)
    let mut chw = vec![0.0f32; state * ctx];
    for t in 0..ctx {
      for s in 0..state {
        chw[s * ctx + t] = host[t * state + s];
      }
    }
    Ok(chw)
  }

  /// Fail-fast guard for every decode entry point, run BEFORE any cache
  /// allocation, token materialization, or logits reservation: bound the request
  /// against the decoder-cache ceiling (`cache_length + n_tokens`) and require the
  /// exact `(1, n_audio_ctx, n_audio_state)` encoder-state shape. Centralizes what
  /// would otherwise be scattered, post-allocation checks across `decode_tokens`
  /// and `decode_token_lazy`, so neither the cache, the `T * n_vocab` logits
  /// buffer, nor the token host copy is materialized for an over-cap or
  /// wrong-shaped request.
  fn precheck_decode(
    &self,
    cache: Option<&CoreMlKvCache>,
    n_tokens: usize,
    encoder_states: &Array,
  ) -> Result<()> {
    let cache_len = cache.map_or(0, |c| c.cache_length);
    let projected = cache_len
      .checked_add(n_tokens)
      .ok_or_else(|| dim_overflow("CoreMlWhisper::precheck_decode: cache_length + T"))?;
    if projected > self.max_decoder_ctx {
      return Err(Error::CapExceeded(crate::error::CapExceededPayload::new(
        "CoreMlWhisper: decoder context",
        "max_decoder_ctx",
        self.max_decoder_ctx as u64,
        projected as u64,
      )));
    }
    let (ctx, state) = (self.dims.n_audio_ctx(), self.dims.n_audio_state());
    let shape = encoder_states.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != ctx || shape[2] != state {
      return Err(Error::ShapePairMismatch(
        crate::error::ShapePairMismatchPayload::new(
          "CoreMlWhisper: encoder states must be (1, n_audio_ctx, n_audio_state)",
          vec![1usize, ctx, state],
          shape.to_vec(),
        ),
      ));
    }
    Ok(())
  }

  /// Run one `TextDecoder` step for `tok` against the channel-major encoder
  /// states `enc_chw`, stitching the cache updates into `cache` and returning
  /// the next-token logits as a host `f32` `(n_vocab,)` row (the caller stacks
  /// the per-step rows into the trait's `(1, T, n_vocab)` Array).
  fn decode_one(&self, tok: u32, enc_chw: &[f32], cache: &mut CoreMlKvCache) -> Result<Vec<f32>> {
    // Preflight the cache cap BEFORE running the prediction: this step would
    // write the new key/value column at index `cache_length`, so it needs a free
    // slot (`cache_length < MAX_DECODER_CTX`). Reject a full cache here so an
    // over-cap decode fails with the typed `CapExceeded` without spending a
    // `TextDecoder` FFI prediction (and before `stitch` would reject the write).
    if cache.cache_length + 1 > MAX_DECODER_CTX {
      return Err(Error::CapExceeded(crate::error::CapExceededPayload::new(
        "CoreMlWhisper::decode: decoder context",
        "max_decoder_ctx",
        MAX_DECODER_CTX as u64,
        (cache.cache_length + 1) as u64,
      )));
    }
    let (state, ctx) = (self.dims.n_audio_state(), self.dims.n_audio_ctx());
    let kv_width = self.kv_width;

    let input_ids = i32_array("CoreMlWhisper::decode: input_ids", &[1], &[tok as i32])?;
    let cache_length = i32_array(
      "CoreMlWhisper::decode: cache_length",
      &[1],
      &[cache.cache_length as i32],
    )?;
    let key_cache = f16_array_from_bits(
      "CoreMlWhisper::decode: key_cache",
      &[1, kv_width, 1, MAX_DECODER_CTX],
      &cache.key_cache,
    )?;
    let value_cache = f16_array_from_bits(
      "CoreMlWhisper::decode: value_cache",
      &[1, kv_width, 1, MAX_DECODER_CTX],
      &cache.value_cache,
    )?;
    // `kv_cache_update_mask`: 1.0 at the slot being written this step
    // (`cache_length`), 0 elsewhere.
    let mut mask = vec![0.0f32; MAX_DECODER_CTX];
    if cache.cache_length < MAX_DECODER_CTX {
      mask[cache.cache_length] = 1.0;
    }
    let kv_update_mask = f16_array_from_f32(
      "CoreMlWhisper::decode: kv_cache_update_mask",
      &[1, MAX_DECODER_CTX],
      &mask,
    )?;
    // `decoder_key_padding_mask`: 0 over the valid (already-written + current)
    // prefix, -inf (a large negative) over the not-yet-written tail, so the
    // decoder self-attention ignores the empty slots.
    let mut kpm = vec![0.0f32; MAX_DECODER_CTX];
    for (i, slot) in kpm.iter_mut().enumerate() {
      if i > cache.cache_length {
        *slot = f32::NEG_INFINITY;
      }
    }
    let kpm_arr = f16_array_from_f32(
      "CoreMlWhisper::decode: decoder_key_padding_mask",
      &[1, MAX_DECODER_CTX],
      &kpm,
    )?;
    let enc_in = f16_array_from_f32(
      "CoreMlWhisper::decode: encoder_output_embeds",
      &[1, state, 1, ctx],
      enc_chw,
    )?;

    let provider = feature_provider(
      "CoreMlWhisper::decode: feature provider",
      &[
        ("input_ids", &input_ids),
        ("cache_length", &cache_length),
        ("key_cache", &key_cache),
        ("value_cache", &value_cache),
        ("kv_cache_update_mask", &kv_update_mask),
        ("encoder_output_embeds", &enc_in),
        ("decoder_key_padding_mask", &kpm_arr),
      ],
    )?;
    // SAFETY: `provider` matches the decoder's declared input schema; the
    // `error:` out-param surfaces as `Err`.
    let out = unsafe { self.text_decoder.predictionFromFeatures_error(&provider) }
      .map_err(|e| coreml_err("CoreMlWhisper::decode: TextDecoder predict", &e))?;

    // logits `[1, 1, n_vocab]` → f32 Array.
    let logits = predict_output_from("CoreMlWhisper::decode: logits", &out, "logits")?;
    let logits_shape = shape_of(&logits);
    let n_vocab = self.dims.n_vocab();
    if logits_shape != [1, 1, n_vocab] {
      return Err(Error::CoreMl(CoreMlPayload::new(
        "CoreMlWhisper::decode: logits shape",
        smol_str::format_smolstr!("expected [1, 1, {n_vocab}], got {logits_shape:?}"),
      )));
    }
    let logits_f32 = read_f16_checked_f32("CoreMlWhisper::decode: logits read", &logits, n_vocab)?;

    // Stitch the cache updates `[1, kv_width, 1, 1]` at `cache_length`. The
    // checked read validates each update's dtype / contiguity / `kv_width`
    // extent before the raw store walk (`stitch` re-checks the width too).
    let key_updates = predict_output_from(
      "CoreMlWhisper::decode: key_cache_updates",
      &out,
      "key_cache_updates",
    )?;
    let value_updates = predict_output_from(
      "CoreMlWhisper::decode: value_cache_updates",
      &out,
      "value_cache_updates",
    )?;
    let key_bits = read_f16_checked(
      "CoreMlWhisper::decode: key_cache_updates read",
      &key_updates,
      kv_width,
    )?;
    let value_bits = read_f16_checked(
      "CoreMlWhisper::decode: value_cache_updates read",
      &value_updates,
      kv_width,
    )?;
    cache.stitch(&key_bits, &value_bits, 1)?;

    Ok(logits_f32)
  }
}

/// Pull a named Float16 output from an already-run prediction result.
fn predict_output_from(
  context: &'static str,
  out: &ProtocolObject<dyn MLFeatureProvider>,
  output: &str,
) -> Result<Retained<MLMultiArray>> {
  let key = NSString::from_str(output);
  // SAFETY: `featureValueForName:` returns nil (→ `None`) for an unknown name.
  let fv =
    unsafe { out.featureValueForName(&key) }.ok_or_else(|| coreml_missing(context, output))?;
  // SAFETY: reading the multi-array payload; nil (→ `None`) if not a multi-array.
  unsafe { fv.multiArrayValue() }.ok_or_else(|| coreml_missing(context, output))
}

/// Deep-copy a [`CoreMlKvCache`] (the trait threads the cache by value; a warm
/// step starts from a clone of the caller's cache so the input is never mutated
/// in place).
fn clone_cache(c: &CoreMlKvCache) -> Result<CoreMlKvCache> {
  let mut key_cache = Vec::new();
  crate::model_validation::reserve_or_error(
    &mut key_cache,
    "CoreMlKvCache: clone key_cache",
    c.key_cache.len(),
  )?;
  key_cache.extend_from_slice(&c.key_cache);
  let mut value_cache = Vec::new();
  crate::model_validation::reserve_or_error(
    &mut value_cache,
    "CoreMlKvCache: clone value_cache",
    c.value_cache.len(),
  )?;
  value_cache.extend_from_slice(&c.value_cache);
  Ok(CoreMlKvCache {
    kv_width: c.kv_width,
    key_cache,
    value_cache,
    cache_length: c.cache_length,
  })
}

/// The decoder's stacked key/value channel width and max KV-cache context, read
/// from the `key_cache` input feature description's shape (`[1, kv_width, 1,
/// max_ctx]`): returns `(kv_width, max_ctx)`. `None` if the description is
/// unavailable or the shape is not the expected 4-D form (the caller falls back
/// to a derived width and the compile-time [`MAX_DECODER_CTX`]).
fn decoder_key_cache_shape(model: &MLModel) -> Option<(usize, usize)> {
  // SAFETY: `modelDescription` is a readonly property returning the model's
  // feature-description bundle.
  let desc = unsafe { model.modelDescription() };
  // SAFETY: `inputDescriptionsByName` is a readonly `NSDictionary` property of
  // the model description.
  let inputs = unsafe { desc.inputDescriptionsByName() };
  let key = NSString::from_str("key_cache");
  // `objectForKey:` returns nil (→ `None`) for an unknown key.
  let feat = inputs.objectForKey(&key)?;
  // SAFETY: `multiArrayConstraint` is the readonly array constraint of the
  // feature; nil (→ `None`) if the feature is not a multi-array.
  let constraint = unsafe { feat.multiArrayConstraint() }?;
  // SAFETY: `shape` is the readonly `NSArray<NSNumber>` declared shape.
  let shape = unsafe { constraint.shape() };
  // `[1, kv_width, 1, max_ctx]` — need both the channel (index 1) and the
  // context (last) extents; a shorter shape is not the expected layout.
  if shape.count() < 4 {
    return None;
  }
  let w = shape.objectAtIndex(1).integerValue();
  let ctx = shape.objectAtIndex(shape.count() - 1).integerValue();
  if w > 0 && ctx > 0 {
    Some((w as usize, ctx as usize))
  } else {
    None
  }
}

// ───────────────────────────── small helpers ───────────────────────────────

/// Read + parse a REQUIRED JSON config file from the model directory, through
/// the shared bounded reader [`crate::io::read_bounded_config_file`] (the same
/// 16 MiB cap + non-regular-reject discipline the rest of the loader applies to
/// `config.json`), so auto-attaching a CoreML-capable checkpoint cannot force an
/// unbounded allocation before backend selection.
fn read_json(dir: &Path, name: &str) -> Result<serde_json::Value> {
  read_json_opt(dir, name)?.ok_or_else(|| {
    Error::FileIo(crate::error::FileIoPayload::new(
      "CoreMlWhisper::load: read config",
      crate::error::FileOp::Open,
      dir.join(name),
      std::io::Error::from(std::io::ErrorKind::NotFound),
    ))
  })
}

/// Read + parse an OPTIONAL JSON config file (`Ok(None)` if absent), through the
/// same shared bounded reader as [`read_json`].
fn read_json_opt(dir: &Path, name: &str) -> Result<Option<serde_json::Value>> {
  let path = dir.join(name);
  let Some(body) = crate::io::read_bounded_config_file(&path, "CoreMlWhisper::load: read config")?
  else {
    return Ok(None);
  };
  serde_json::from_str(&body).map(Some).map_err(|e| {
    Error::Parse(crate::error::ParsePayload::new(
      "CoreMlWhisper::load: config json",
      "json",
      e,
    ))
  })
}

/// The shared `i32` dimension-overflow error.
fn dim_overflow(what: &'static str) -> Error {
  Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::new(what, "i32"))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::audio::stt::models::whisper::inference::WhisperInference;

  /// The local WhisperKit tiny `.mlmodelc` bundle (gitignored), at
  /// `<crate>/../models/whisperkit/openai_whisper-tiny`.
  fn tiny_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("..")
      .join("models")
      .join("whisperkit")
      .join("openai_whisper-tiny")
  }

  /// The CoreML decode path rejects encoder states that are not EXACTLY
  /// `(1, n_audio_ctx, n_audio_state)`: a same-length but transposed tensor is a
  /// typed [`Error::ShapePairMismatch`], not silently-corrupted logits.
  #[test]
  #[ignore = "needs the local WhisperKit tiny .mlmodelc bundle; runs on the ANE"]
  fn coreml_decode_rejects_wrong_shaped_encoder_states() {
    let dir = tiny_dir();
    if !CoreMlWhisper::is_present(&dir) {
      eprintln!("SKIP coreml_decode_rejects_wrong_shaped_encoder_states: {dir:?} has no bundle");
      return;
    }
    let backend = CoreMlWhisper::load(&dir).expect("load CoreML whisper-tiny");
    let (ctx, state) = (backend.dims().n_audio_ctx(), backend.dims().n_audio_state());
    // Transposed `(1, n_audio_state, n_audio_ctx)`: same element count, wrong shape.
    let wrong = Array::full::<f32>(&[1, state as i32, ctx as i32], 0.0).expect("wrong enc");
    // `decode_tokens`' Ok type holds a `CoreMlKvCache` (not `Debug`), so match the
    // `Result` directly rather than `expect_err` (which would require Ok: Debug).
    let result = backend.decode_tokens(&[50258u32], &wrong, None);
    assert!(
      matches!(result, Err(Error::ShapePairMismatch(_))),
      "a transposed encoder-state shape must be rejected with ShapePairMismatch"
    );
  }

  /// The CoreML decode primitive rejects an over-cap request UP FRONT: calling
  /// `decode_tokens` directly with more than the decoder-cache ceiling of valid
  /// tokens is a typed [`Error::CapExceeded`] BEFORE any encoder-state
  /// materialization or `T * n_vocab` logits allocation — not a per-token trip
  /// deep inside the decode loop.
  #[test]
  #[ignore = "needs the local WhisperKit tiny .mlmodelc bundle; runs on the ANE"]
  fn coreml_decode_rejects_over_cap_prefix_up_front() {
    let dir = tiny_dir();
    if !CoreMlWhisper::is_present(&dir) {
      eprintln!("SKIP coreml_decode_rejects_over_cap_prefix_up_front: {dir:?} has no bundle");
      return;
    }
    let backend = CoreMlWhisper::load(&dir).expect("load CoreML whisper-tiny");
    let (ctx, state) = (backend.dims().n_audio_ctx(), backend.dims().n_audio_state());
    // A correctly-shaped encoder-states tensor, so the CAP guard (not the shape
    // guard) is what rejects; `MAX_DECODER_CTX + 1` valid tokens overruns the cache.
    let enc = Array::full::<f32>(&[1, ctx as i32, state as i32], 0.0).expect("enc");
    let over = vec![50258u32; MAX_DECODER_CTX + 1];
    let result = backend.decode_tokens(&over, &enc, None);
    assert!(
      matches!(result, Err(Error::CapExceeded(_))),
      "an over-cap prefix must be rejected up front with CapExceeded"
    );
  }

  /// The lazy single-token decode entry point also guards UP FRONT: an over-cap
  /// token tensor is rejected by `precheck_decode` before the host `to_vec`
  /// materialization (the element count is read from the shape, no copy).
  #[test]
  #[ignore = "needs the local WhisperKit tiny .mlmodelc bundle; runs on the ANE"]
  fn coreml_decode_token_lazy_rejects_over_cap_up_front() {
    let dir = tiny_dir();
    if !CoreMlWhisper::is_present(&dir) {
      eprintln!("SKIP coreml_decode_token_lazy_rejects_over_cap_up_front: {dir:?} has no bundle");
      return;
    }
    let backend = CoreMlWhisper::load(&dir).expect("load CoreML whisper-tiny");
    let (ctx, state) = (backend.dims().n_audio_ctx(), backend.dims().n_audio_state());
    let enc = Array::full::<f32>(&[1, ctx as i32, state as i32], 0.0).expect("enc");
    // A token tensor carrying MAX_DECODER_CTX + 1 valid ids overruns the cache.
    let over = vec![50258u32; MAX_DECODER_CTX + 1];
    let token = Array::from_slice::<u32>(&over, &[1, (MAX_DECODER_CTX + 1) as i32]).expect("token");
    let result = WhisperInference::decode_token_lazy(&backend, &token, &enc, None);
    assert!(
      matches!(result, Err(Error::CapExceeded(_))),
      "an over-cap lazy token must be rejected up front with CapExceeded"
    );
  }

  /// End-to-end ANE path: load the CoreML backend, run the `AudioEncoder` on a
  /// (silent) mel, then one `TextDecoder` step from `<|startoftranscript|>`, and
  /// assert finite logits of shape `[1, 1, n_vocab]` with a sensible argmax. This
  /// proves encode + decode + the explicit-KV-cache stitch run on the Neural
  /// Engine through the [`WhisperInference`] surface.
  #[test]
  #[ignore = "needs the local WhisperKit tiny .mlmodelc bundle; runs on the ANE"]
  fn coreml_encode_then_decode_step_is_finite() {
    let dir = tiny_dir();
    if !CoreMlWhisper::is_present(&dir) {
      eprintln!("SKIP coreml_encode_then_decode_step_is_finite: {dir:?} has no .mlmodelc bundle");
      return;
    }
    let backend = CoreMlWhisper::load(&dir).expect("load CoreML whisper-tiny");
    let (n_mels, n_ctx, n_state, n_vocab) = (
      backend.dims().n_mels(),
      backend.dims().n_audio_ctx(),
      backend.dims().n_audio_state(),
      backend.dims().n_vocab(),
    );
    assert_eq!((n_mels, n_ctx, n_state), (80, 1500, 384));

    // A silent mel `(3000, 80)` → finite encoder states `(1, 1500, 384)`.
    let mel = Array::full::<f32>(&[3000i32, n_mels as i32], 0.0).expect("mel");
    let mut enc = WhisperInference::encode(&backend, &mel).expect("encode on ANE");
    assert_eq!(enc.shape(), &[1, n_ctx, n_state]);
    let enc_host = enc.to_vec::<f32>().expect("encoder states host");
    assert!(
      enc_host.iter().all(|v| v.is_finite()),
      "encoder states must be finite"
    );

    // One decode step from <|startoftranscript|> (50258) with a fresh cache.
    let sot = [50258u32];
    let (mut logits, cache) = backend
      .decode_tokens(&sot, &enc, None)
      .expect("decode step on ANE");
    assert_eq!(logits.shape(), &[1, 1, n_vocab]);
    let row = logits.to_vec::<f32>().expect("logits host");
    assert_eq!(row.len(), n_vocab);
    assert!(row.iter().all(|v| v.is_finite()), "logits must be finite");
    let argmax = row
      .iter()
      .copied()
      .enumerate()
      .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
      .map(|(i, _)| i)
      .expect("argmax");
    assert!(argmax < n_vocab, "argmax token in range");

    // The cache advanced by exactly one column (the decoded token).
    assert_eq!(cache.cache_length, 1, "one token written to the KV cache");

    // A second warm step from the cache also yields finite logits — proves the
    // stitched cache is consumable.
    let (mut logits2, cache2) = backend
      .decode_tokens(&[argmax as u32], &enc, Some(&cache))
      .expect("warm decode step on ANE");
    let row2 = logits2.to_vec::<f32>().expect("warm logits host");
    assert!(
      row2.iter().all(|v| v.is_finite()),
      "warm-step logits must be finite"
    );
    assert_eq!(cache2.cache_length, 2, "two tokens written to the KV cache");

    // A multi-token prefill (the transcribe path's first forward) returns the
    // full `(1, T, n_vocab)` logits — every position finite — so the pipeline's
    // `last_position_row` (reads `[0, T-1]`) and `no_speech_prob` (reads
    // `[0, sot_index]`) both find a real row, not a single collapsed step.
    let prefix = [50258u32, 50259, 50359, 50363];
    let (mut pre_logits, pre_cache) = backend
      .decode_tokens(&prefix, &enc, None)
      .expect("prefill on ANE");
    assert_eq!(pre_logits.shape(), &[1, prefix.len(), n_vocab]);
    assert_eq!(pre_cache.cache_length, prefix.len());
    let pre_host = pre_logits.to_vec::<f32>().expect("prefill logits host");
    assert!(
      pre_host.iter().all(|v| v.is_finite()),
      "all prefill positions must be finite"
    );

    // The decoder-cache ceiling the shared decode loop caps against is read from
    // the artifact (the `key_cache` last dim), clamped to the store size — the
    // WhisperKit tiny export is 224, well below `n_text_ctx`.
    assert_eq!(
      WhisperInference::max_decoder_context(&backend),
      MAX_DECODER_CTX,
      "CoreML decoder-context cap reads the 224-slot key_cache extent"
    );

    eprintln!(
      "CoreML/ANE encode+decode OK: argmax(step1)={argmax}, warm cache_length={}, \
       prefill T={}, max_decoder_ctx={}",
      cache2.cache_length,
      prefix.len(),
      WhisperInference::max_decoder_context(&backend),
    );
  }

  // ─────────────────── checked MLMultiArray output reads ───────────────────

  /// `read_f16_checked` rejects an output array whose `dataType` is not Float16
  /// (here Float32) with a typed [`Error::CoreMl`], never reinterpreting the f32
  /// store as `u16` bits.
  #[test]
  fn read_f16_checked_rejects_wrong_dtype() {
    let a = new_multi_array("test: f32 array", &[1, 4], MLMultiArrayDataType::Float32)
      .expect("alloc f32 array");
    let err = read_f16_checked("test: read", &a, 4).expect_err("wrong dtype must error");
    assert!(
      matches!(err, Error::CoreMl(_)),
      "wrong dtype maps to Error::CoreMl, got {err:?}"
    );
  }

  /// `read_f16_checked` rejects a count mismatch (the store is Float16 but its
  /// element count differs from `expected_count`) with a typed [`Error::CoreMl`]
  /// BEFORE allocating / reading.
  #[test]
  fn read_f16_checked_rejects_wrong_count() {
    let a = f16_array_from_f32("test: f16 array", &[1, 4], &[0.0, 1.0, 2.0, 3.0])
      .expect("alloc f16 array");
    let err = read_f16_checked("test: read", &a, 8).expect_err("wrong count must error");
    assert!(
      matches!(err, Error::CoreMl(_)),
      "count mismatch maps to Error::CoreMl, got {err:?}"
    );
    // `read_f16_checked_f32` shares the same validation gate.
    let err32 = read_f16_checked_f32("test: read f32", &a, 8).expect_err("wrong count must error");
    assert!(matches!(err32, Error::CoreMl(_)), "got {err32:?}");
  }

  /// `read_f16_checked` reads a well-formed contiguous Float16 array of the
  /// expected count back as its exact bits, and `read_f16_checked_f32` as its
  /// values — the positive path the output reads rely on.
  #[test]
  fn read_f16_checked_reads_valid_float16() {
    let vals = [0.0f32, 0.5, -1.0, 2.0];
    let a = f16_array_from_f32("test: f16 array", &[1, 4], &vals).expect("alloc f16 array");
    let bits = read_f16_checked("test: read", &a, 4).expect("valid read");
    assert_eq!(bits.len(), 4);
    let expected: Vec<u16> = vals.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    assert_eq!(bits, expected, "bits round-trip the written values");
    let f = read_f16_checked_f32("test: read f32", &a, 4).expect("valid f32 read");
    assert_eq!(
      f,
      vals.to_vec(),
      "f32 read round-trips (exact at these values)"
    );
  }

  /// The contiguity predicate accepts row-major strides and rejects any
  /// non-compact / non-first-major layout (the gate the raw `dataPointer` walk
  /// depends on for in-bounds reads).
  #[test]
  fn is_row_major_contiguous_predicate() {
    // `[1, 1536, 1, 224]` → row-major strides `[1536*224, 224, 224, 1]`.
    assert!(is_row_major_contiguous(
      &[1, 1536, 1, 224],
      &[1536 * 224, 224, 224, 1]
    ));
    assert!(is_row_major_contiguous(&[4], &[1]));
    assert!(is_row_major_contiguous(&[], &[]));
    // A padded / non-compact inner stride is rejected.
    assert!(!is_row_major_contiguous(&[2, 3], &[4, 1]));
    // A column-major (last stride != 1) layout is rejected.
    assert!(!is_row_major_contiguous(&[2, 3], &[1, 2]));
    // Mismatched rank is rejected.
    assert!(!is_row_major_contiguous(&[2, 3], &[3]));
  }

  /// `strided_extent` is the physical span a (possibly padded) descriptor
  /// addresses: compact → element count; padded rows → count + padding; a zero
  /// extent → 0; a scalar / empty shape → 1.
  #[test]
  fn strided_extent_compact_and_padded() {
    assert_eq!(strided_extent("t", &[2, 3], &[3, 1]).unwrap(), 6); // compact 2x3
    assert_eq!(strided_extent("t", &[2, 3], &[4, 1]).unwrap(), 7); // rows padded to stride 4
    assert_eq!(strided_extent("t", &[0, 3], &[3, 1]).unwrap(), 0); // an empty axis
    assert_eq!(strided_extent("t", &[], &[]).unwrap(), 1); // scalar / empty shape
  }

  /// `strided_extent` is fallible at its `from_raw_parts` boundary: a shape /
  /// strides RANK MISMATCH, an OVERFLOWING offset span, and a span past the
  /// `isize::MAX` byte-size limit are all typed errors — never a wrapped (and
  /// unsound) slice length.
  #[test]
  fn strided_extent_rejects_rank_mismatch_and_overflow() {
    assert!(strided_extent("t", &[2, 3], &[3]).is_err(), "rank mismatch");
    assert!(
      strided_extent("t", &[2, 2], &[usize::MAX, 1]).is_err(),
      "offset arithmetic overflow"
    );
    assert!(
      strided_extent("t", &[isize::MAX as usize, 1], &[1, 1]).is_err(),
      "span past the from_raw_parts size limit"
    );
  }

  /// A compact store gathers as a straight row-major prefix copy.
  #[test]
  fn strided_gather_compact_is_prefix_copy() {
    let backing = [10u16, 11, 12, 13, 14, 15];
    let mut out = Vec::new();
    strided_gather_u16(&backing, &[2, 3], &[3, 1], 6, &mut out);
    assert_eq!(out, vec![10, 11, 12, 13, 14, 15]);
  }

  /// A row-padded store (stride 4 for 3-wide rows) skips the padding element,
  /// yielding only the logical elements in row-major order.
  #[test]
  fn strided_gather_padded_rows_skips_padding() {
    // row0 = 10,11,12, pad=99; row1 = 20,21,22.
    let backing = [10u16, 11, 12, 99, 20, 21, 22];
    assert_eq!(
      strided_extent("t", &[2, 3], &[4, 1]).unwrap(),
      backing.len()
    );
    let mut out = Vec::new();
    strided_gather_u16(&backing, &[2, 3], &[4, 1], 6, &mut out);
    assert_eq!(out, vec![10, 11, 12, 20, 21, 22], "padding 99 is skipped");
  }

  /// The ANE `AudioEncoder` layout: `[1, C, 1, T]` with the trailing extent `T`
  /// padded to a larger row stride. Logical `(0, c, 0, t)` lives at
  /// `c * row_stride + t`, and the gather must return it row-major over `(C, T)`
  /// as `out[c * T + t]` — exactly the order `run_audio_encoder` then transposes.
  #[test]
  fn strided_gather_matches_padded_encoder_layout() {
    let (c, t, row_stride) = (2usize, 3usize, 4usize);
    let shape = [1, c, 1, t];
    let strides = [c * row_stride, row_stride, row_stride, 1];
    let extent = strided_extent("t", &shape, &strides).unwrap();
    let mut backing = vec![0u16; extent];
    for cc in 0..c {
      for tt in 0..t {
        backing[cc * row_stride + tt] = (cc * 10 + tt) as u16;
      }
    }
    let mut out = Vec::new();
    strided_gather_u16(&backing, &shape, &strides, c * t, &mut out);
    let expect: Vec<u16> = (0..c)
      .flat_map(|cc| (0..t).map(move |tt| (cc * 10 + tt) as u16))
      .collect();
    assert_eq!(out, expect);
  }

  // ─────────────────── decoder-cache ceiling (Finding 1) ───────────────────

  /// The explicit KV cache stops cleanly at [`MAX_DECODER_CTX`]: stitching
  /// `MAX_DECODER_CTX` single-token columns succeeds, and the next stitch — the
  /// "never-EOT" step that would overrun the cache — returns a typed
  /// [`Error::CapExceeded`] (the same contract `decode_one`'s preflight enforces
  /// before any `TextDecoder` prediction), never a panic or an out-of-bounds
  /// write. The shared decode loop's `sample_len` cap keeps a real transcription
  /// from reaching this, but the cache itself is the hard backstop.
  #[test]
  fn coreml_kv_cache_caps_at_max_decoder_ctx() {
    let kv_width = 8usize;
    let mut cache = CoreMlKvCache::new(kv_width).expect("fresh cache");
    let key = vec![1u16; kv_width];
    let value = vec![2u16; kv_width];
    for i in 0..MAX_DECODER_CTX {
      cache
        .stitch(&key, &value, 1)
        .unwrap_or_else(|e| panic!("stitch at column {i} must succeed: {e:?}"));
    }
    assert_eq!(
      cache.cache_length, MAX_DECODER_CTX,
      "cache filled to the cap"
    );
    // The (cap + 1)-th write — what a never-EOT decode would attempt — is
    // rejected with the typed cap error, not a panic.
    let err = cache
      .stitch(&key, &value, 1)
      .expect_err("an over-cap stitch must be rejected");
    match err {
      Error::CapExceeded(p) => {
        assert_eq!(p.cap(), MAX_DECODER_CTX as u64);
        assert_eq!(p.observed(), (MAX_DECODER_CTX + 1) as u64);
      }
      other => panic!("expected Error::CapExceeded, got {other:?}"),
    }
    // The rejected write left the length unchanged (no partial mutation).
    assert_eq!(
      cache.cache_length, MAX_DECODER_CTX,
      "length unchanged after rejection"
    );
  }
}
