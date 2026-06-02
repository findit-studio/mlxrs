//! Shared config / weight validation helpers for the model ports.
//!
//! Every model port (`audio::stt` Wav2Vec2, `lm` LFM2, Whisper, and future
//! SigLIP2 / embeddings models) reads an HF `config.json` into a typed config
//! and drains a flat `name → tensor` checkpoint map into its module graph.
//! Before any tensor is allocated or read, the config must be validated and
//! the weight map keyed so a corrupt / hostile / wrong-architecture input
//! **fails fast with a typed [`crate::Error`]** instead of loading silently
//! wrong, panicking on an unchecked cast, or driving an oversized allocation
//! to an out-of-memory abort.
//!
//! The same handful of checks recur per model — pinning an architecture field
//! to its reference value, bounding a layer / head count before it sizes an
//! allocation, overflow-checking a config-derived dimension, fallibly
//! reserving a config-sized buffer, rejecting a duplicate weight key, and
//! gating an optional weight on a config flag. This module factors them into
//! one reusable, allocation-disciplined toolkit so each model composes the
//! helpers rather than re-deriving (and re-reviewing) them.
//!
//! ## Sections
//!
//! 1. **Field pinning** — [`pin_i32`] / [`pin_usize`] / [`pin_bool`] /
//!    [`pin_str`] / [`pin_f64`] / [`pin_i32_slice`]: assert a config field
//!    equals its reference value.
//! 2. **Bounds** — [`require_positive`], [`require_in_range`],
//!    [`require_cardinality`], [`require_divisible`], [`require_even`].
//! 3. **Checked arithmetic** — [`checked_mul`] / [`checked_add`] for
//!    config-derived dimensions.
//! 4. **Fallible allocation** — [`reserve_or_error`] for a config-sized
//!    `Vec` / `HashMap`.
//! 5. **Key collision** — [`insert_unique`] for sanitize / weight-key maps.
//! 6. **Config-gated optional weight** — [`require_if_present`] /
//!    [`take_if`].
//! 7. **Resource extents** — [`Extent`] / [`elem_count`] / [`alloc_filled`]:
//!    lift the bounds / arithmetic / alloc helpers into types so an unbounded
//!    product or an infallible model-buffer allocation is *unwritable* rather
//!    than caught per call site.
//!
//! Every helper returns `Result<()>` or `Result<T>` with a typed
//! [`crate::error`] variant that names the offending field. None panics; the
//! oversized-input paths ([`require_cardinality`], [`checked_mul`],
//! [`reserve_or_error`]) are the bounded-memory guards that turn a hostile
//! config into a recoverable error.
//!
//! Always compiled (no feature gate) so every model feature can rely on it.

use std::{
  collections::{HashMap, TryReserveError},
  hash::Hash,
};

use smol_str::format_smolstr;

use crate::error::{
  AllocFailurePayload, ArithmeticOverflowPayload, CapExceededPayload,
  DivisibilityConstraintPayload, Error, InvariantViolationPayload, KeyCollisionPayload,
  LengthMismatchPayload, MissingKeyPayload, NonFiniteScalarPayload, OutOfRangePayload, Result,
  UnknownEnumValuePayload,
};

// ════════════════════════════ 1. field pinning ════════════════════════════
//
// Pin an architecture-defining config field to the single reference value the
// port implements. A deviating field is not a different *value* of the same
// model — it is a different (unsupported) architecture, so the port must
// reject it before building the wrong graph. Mirrors Wav2Vec2's `check_eq_i32`
// / `check_eq_conv_array` (every base-960h field pinned) and LFM2's exact-value
// config gates.

/// Assert an `i32` config field equals its reference value.
///
/// On mismatch returns [`Error::OutOfRange`] naming `field`, the violated
/// requirement, and the offending value (with the expected value). Pinning a
/// *count* field (e.g. `num_hidden_layers`) with this also **bounds** it: an
/// oversized count can never reach the per-layer allocation loop.
///
/// ```
/// use mlxrs::model_validation::pin_i32;
/// assert!(pin_i32("hidden_size", 768, 768).is_ok());
/// assert!(pin_i32("hidden_size", 512, 768).is_err());
/// ```
pub fn pin_i32(field: &'static str, actual: i32, expected: i32) -> Result<()> {
  if actual != expected {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      field,
      "must equal the reference architecture value",
      format_smolstr!("{actual} (expected {expected})"),
    )));
  }
  Ok(())
}

/// Assert a `usize` config field equals its reference value.
///
/// On mismatch returns [`Error::OutOfRange`] naming `field` and the offending
/// value. The `usize` analogue of [`pin_i32`] for fields already widened to
/// the host pointer width.
///
/// ```
/// use mlxrs::model_validation::pin_usize;
/// assert!(pin_usize("num_layers", 12, 12).is_ok());
/// assert!(pin_usize("num_layers", 13, 12).is_err());
/// ```
pub fn pin_usize(field: &'static str, actual: usize, expected: usize) -> Result<()> {
  if actual != expected {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      field,
      "must equal the reference architecture value",
      format_smolstr!("{actual} (expected {expected})"),
    )));
  }
  Ok(())
}

/// Assert a `bool` config field equals its reference value.
///
/// On mismatch returns [`Error::InvariantViolation`] naming `field` and the
/// required boolean. Mirrors Wav2Vec2's `do_stable_layer_norm` / `conv_bias`
/// gates: a checkpoint whose boolean flag contradicts the wired graph would
/// otherwise load and run silently wrong (e.g. an unconsumed, silently-dropped
/// bias tensor).
///
/// ```
/// use mlxrs::model_validation::pin_bool;
/// assert!(pin_bool("conv_bias", false, false).is_ok());
/// assert!(pin_bool("conv_bias", true, false).is_err());
/// ```
pub fn pin_bool(field: &'static str, actual: bool, expected: bool) -> Result<()> {
  if actual != expected {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      field,
      if expected {
        "must be true"
      } else {
        "must be false"
      },
    )));
  }
  Ok(())
}

/// Assert an enum-style `&str` config field is one of its `allowed`
/// reference values.
///
/// On mismatch returns [`Error::UnknownEnumValue`] naming `field`, the
/// offending value, and the full `allowed` set (so the error can suggest the
/// valid arms). Pass a single-element slice (`&["wav2vec2"]`) to pin one exact
/// value, or several to accept any of a set of aliases. Mirrors Wav2Vec2's
/// `model_type` / `feat_extract_norm` pins (each lists its supported arm(s) as
/// the suggestion).
///
/// ```
/// use mlxrs::model_validation::pin_str;
/// assert!(pin_str("model_type", "wav2vec2", &["wav2vec2"]).is_ok());
/// assert!(pin_str("feat_extract_norm", "group", &["group", "layer"]).is_ok());
/// assert!(pin_str("model_type", "hubert", &["wav2vec2"]).is_err());
/// ```
pub fn pin_str(field: &'static str, actual: &str, allowed: &'static [&'static str]) -> Result<()> {
  if !allowed.contains(&actual) {
    return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
      field, actual, allowed,
    )));
  }
  Ok(())
}

/// Assert an `f64` config field is finite and equals its reference value.
///
/// First rejects a non-finite (NaN / Inf) value with [`Error::NonFiniteScalar`]
/// (a config `eps` / multiplier is a free-floating float), then rejects a
/// finite value that differs from `expected` with [`Error::OutOfRange`].
/// Equality is `==` on `f64` — pass the EXACT reference constant (e.g.
/// `1e-5`), not a rounded approximation. For a tolerance-based compare a caller
/// should band-check via [`require_in_range`] instead.
///
/// ```
/// use mlxrs::model_validation::pin_f64;
/// assert!(pin_f64("layer_norm_eps", 1e-5, 1e-5).is_ok());
/// assert!(pin_f64("layer_norm_eps", 1e-6, 1e-5).is_err());
/// assert!(pin_f64("layer_norm_eps", f64::NAN, 1e-5).is_err());
/// ```
pub fn pin_f64(field: &'static str, actual: f64, expected: f64) -> Result<()> {
  if !actual.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      field, actual,
    )));
  }
  if actual != expected {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      field,
      "must equal the reference architecture value",
      format_smolstr!("{actual} (expected {expected})"),
    )));
  }
  Ok(())
}

/// Assert an `&[i32]` config field equals its reference slice (length +
/// per-element).
///
/// A wrong length is [`Error::LengthMismatch`]; the first deviating element is
/// [`Error::OutOfRange`] naming the index, the offending value, and the
/// expected value. Mirrors Wav2Vec2's `check_eq_conv_array` for the
/// `conv_dim` / `conv_stride` / `conv_kernel` stacks.
///
/// ```
/// use mlxrs::model_validation::pin_i32_slice;
/// assert!(pin_i32_slice("conv_stride", &[5, 2, 2], &[5, 2, 2]).is_ok());
/// assert!(pin_i32_slice("conv_stride", &[5, 2], &[5, 2, 2]).is_err());
/// assert!(pin_i32_slice("conv_stride", &[5, 3, 2], &[5, 2, 2]).is_err());
/// ```
pub fn pin_i32_slice(field: &'static str, actual: &[i32], expected: &[i32]) -> Result<()> {
  if actual.len() != expected.len() {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      field,
      expected.len(),
      actual.len(),
    )));
  }
  for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
    if a != e {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        field,
        "must equal the reference architecture array",
        format_smolstr!("element {i} = {a} (expected {e})"),
      )));
    }
  }
  Ok(())
}

// ════════════════════════════════ 2. bounds ════════════════════════════════

/// Require an `i32` config field to be strictly positive (`> 0`).
///
/// On violation returns [`Error::OutOfRange`] naming `field` and the offending
/// value. Mirrors LFM2's `check_positive`. Use for any width / count a later
/// step divides by or uses to size work — a non-positive value would otherwise
/// divide-by-zero or wrap a length computation.
///
/// ```
/// use mlxrs::model_validation::require_positive;
/// assert!(require_positive("hidden_size", 768).is_ok());
/// assert!(require_positive("hidden_size", 0).is_err());
/// assert!(require_positive("hidden_size", -1).is_err());
/// ```
pub fn require_positive(field: &'static str, value: i32) -> Result<()> {
  if value <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      field,
      "must be a positive integer (> 0)",
      format_smolstr!("{value}"),
    )));
  }
  Ok(())
}

/// Require an `i32` config field to lie in the inclusive range
/// `[min, max]`.
///
/// On violation returns [`Error::OutOfRange`] naming `field`, the bounds, and
/// the offending value. `min` must not exceed `max` (an empty range is a
/// programming error and itself rejects every value). Use for a field with a
/// genuine valid band (not a single pinned value).
///
/// ```
/// use mlxrs::model_validation::require_in_range;
/// assert!(require_in_range("groups", 16, 1, 64).is_ok());
/// assert!(require_in_range("groups", 0, 1, 64).is_err());
/// assert!(require_in_range("groups", 65, 1, 64).is_err());
/// ```
pub fn require_in_range(field: &'static str, value: i32, min: i32, max: i32) -> Result<()> {
  if value < min || value > max {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      field,
      "must be in the inclusive range [min, max]",
      format_smolstr!("{value} (allowed [{min}, {max}])"),
    )));
  }
  Ok(())
}

/// Require a cardinality (layer / head / shard count) to be positive and
/// within a field-specific cap.
///
/// A cardinality sizes an eager allocation (the decoder-layer `Vec`, the
/// per-layer cache, a shard list), so an oversized or hostile value would
/// over-allocate toward an out-of-memory abort. This bounds it **before** the
/// allocation: a non-positive `count` is [`Error::OutOfRange`]; a `count`
/// exceeding `max_cap` is [`Error::CapExceeded`] carrying the cap name, the cap,
/// and the observed value. `field` doubles as the cap name in the payload.
///
/// `count` is taken as `i64` so a raw config `i32` (possibly negative) and a
/// `usize` length both widen losslessly; the bound is checked in `i64` so a
/// near-`i64::MAX` input cannot wrap.
///
/// ```
/// use mlxrs::model_validation::require_cardinality;
/// assert!(require_cardinality("num_hidden_layers", 12, 4096).is_ok());
/// assert!(require_cardinality("num_hidden_layers", 0, 4096).is_err());
/// assert!(require_cardinality("num_hidden_layers", 1 << 30, 4096).is_err());
/// ```
pub fn require_cardinality(field: &'static str, count: i64, max_cap: u64) -> Result<()> {
  if count <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      field,
      "must be a positive count (> 0)",
      format_smolstr!("{count}"),
    )));
  }
  // `count > 0` here, so the `as u64` cast is lossless and non-wrapping.
  let observed = count as u64;
  if observed > max_cap {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      field, field, max_cap, observed,
    )));
  }
  Ok(())
}

/// Require `a` to be exactly divisible by `b` (`a % b == 0`), guarding the
/// divisor against zero.
///
/// Returns [`Error::DivisibilityConstraint`] naming both operands and their
/// values when `a` is not a multiple of `b`. A **zero or negative divisor** is
/// rejected first as [`Error::OutOfRange`] (a `% 0` would panic), and a
/// **negative dividend** is likewise rejected as [`Error::OutOfRange`] (the
/// dividend is a config dimension, non-negative by construction, and would
/// otherwise be misreported as a huge `u64` in the constraint payload), so this
/// is safe on an unvalidated config pair. Mirrors LFM2's `head_dim` /
/// GQA-grouping divisibility gates.
///
/// ```
/// use mlxrs::model_validation::require_divisible;
/// assert!(require_divisible("hidden_size", 768, "num_heads", 12).is_ok());
/// assert!(require_divisible("hidden_size", 768, "num_heads", 5).is_err());
/// assert!(require_divisible("hidden_size", 768, "num_heads", 0).is_err());
/// assert!(require_divisible("hidden_size", -768, "num_heads", 12).is_err());
/// ```
pub fn require_divisible(a_name: &'static str, a: i32, b_name: &'static str, b: i32) -> Result<()> {
  if b <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      b_name,
      "divisor must be a positive integer (> 0)",
      format_smolstr!("{b}"),
    )));
  }
  require_non_negative_operand(a_name, a)?;
  if a % b != 0 {
    return Err(Error::DivisibilityConstraint(
      DivisibilityConstraintPayload::new(
        a_name,
        a_name,
        i64::from(a) as u64,
        b_name,
        i64::from(b) as u64,
      ),
    ));
  }
  Ok(())
}

/// Require an `i32` config field to be even.
///
/// Returns [`Error::OutOfRange`] naming `field` and the offending odd value.
/// Mirrors LFM2's RoPE `head_dim` even-ness gate: an odd dimension loads but
/// only fails deep inside the attention forward pass (where feature `k` pairs
/// with `k + dim/2`), so pinning it even at config time fails fast.
///
/// Works for negative inputs too (`-4` is even, `-3` is odd) via Rust's
/// sign-preserving `%`.
///
/// ```
/// use mlxrs::model_validation::require_even;
/// assert!(require_even("head_dim", 64).is_ok());
/// assert!(require_even("head_dim", 65).is_err());
/// ```
pub fn require_even(field: &'static str, value: i32) -> Result<()> {
  if value % 2 != 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      field,
      "must be even",
      format_smolstr!("{value}"),
    )));
  }
  Ok(())
}

// ════════════════════════════ 3. checked arithmetic ════════════════════════
//
// Config-derived dimension arithmetic (e.g. `num_heads * head_dim`,
// `vocab_size + added`) operates on free-floating config integers and must not
// silently wrap. These wrap `i32::checked_*` into a typed
// [`Error::ArithmeticOverflow`] carrying the named operands. Mirrors LFM2's
// `adjusted_ff_dim` checked-step arithmetic and Wav2Vec2's
// `num_heads * head_dim` guard.

/// Reject a negative operand before it reaches a `u64` error payload.
///
/// The overflow / divisibility payloads carry their operands as `u64`, so a
/// negative `i32` would be reported as its huge two's-complement value
/// (`i32::MIN` → `4294967296`-ish). These helpers operate on config
/// dimensions / counts, which are non-negative by construction, so a negative
/// operand is itself out of range and is rejected as [`Error::OutOfRange`]
/// (carrying the true signed value) before any payload is built.
fn require_non_negative_operand(name: &'static str, value: i32) -> Result<()> {
  if value < 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      name,
      "must be a non-negative dimension (>= 0)",
      format_smolstr!("{value}"),
    )));
  }
  Ok(())
}

/// Multiply two config-derived `i32` dimensions, returning
/// [`Error::ArithmeticOverflow`] on overflow.
///
/// `context` labels the expression (e.g. `"num_heads * head_dim"`); `a_name` /
/// `b_name` label the operands so the overflow payload carries the runtime
/// values. The non-overflow result is the exact `i32` product.
///
/// Operands are config dimensions and must be **non-negative**: a negative
/// operand is rejected as [`Error::OutOfRange`] (naming it and its true value)
/// before any overflow check, so the overflow payload's `u64` operands always
/// hold the real values.
///
/// ```
/// use mlxrs::model_validation::checked_mul;
/// assert_eq!(checked_mul("embed", "heads", 12, "head_dim", 64).unwrap(), 768);
/// assert!(checked_mul("embed", "heads", i32::MAX, "head_dim", 2).is_err());
/// assert!(checked_mul("embed", "heads", -1, "head_dim", 64).is_err());
/// ```
pub fn checked_mul(
  context: &'static str,
  a_name: &'static str,
  a: i32,
  b_name: &'static str,
  b: i32,
) -> Result<i32> {
  require_non_negative_operand(a_name, a)?;
  require_non_negative_operand(b_name, b)?;
  a.checked_mul(b).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      context,
      "i32",
      [(a_name, i64::from(a) as u64), (b_name, i64::from(b) as u64)],
    ))
  })
}

/// Add two config-derived `i32` dimensions, returning
/// [`Error::ArithmeticOverflow`] on overflow.
///
/// The additive companion to [`checked_mul`] — e.g. for `vocab_size +
/// num_added_tokens` or an accumulated sequence offset. Both operands are
/// non-negative dimensions; a negative one is rejected as
/// [`Error::OutOfRange`] before the overflow check (same rationale as
/// [`checked_mul`]).
///
/// ```
/// use mlxrs::model_validation::checked_add;
/// assert_eq!(checked_add("vocab", "base", 32000, "added", 100).unwrap(), 32100);
/// assert!(checked_add("vocab", "base", i32::MAX, "added", 1).is_err());
/// assert!(checked_add("vocab", "base", i32::MIN, "added", 0).is_err());
/// ```
pub fn checked_add(
  context: &'static str,
  a_name: &'static str,
  a: i32,
  b_name: &'static str,
  b: i32,
) -> Result<i32> {
  require_non_negative_operand(a_name, a)?;
  require_non_negative_operand(b_name, b)?;
  a.checked_add(b).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      context,
      "i32",
      [(a_name, i64::from(a) as u64), (b_name, i64::from(b) as u64)],
    ))
  })
}

// ════════════════════════════ 4. fallible allocation ═══════════════════════

/// Reserve exactly `count` additional slots in a `Vec` / `HashMap` (anything
/// implementing [`TryReserve`]), turning an allocator failure into a typed
/// [`Error::AllocFailure`] instead of the abort `Vec::with_capacity` /
/// `reserve` would raise.
///
/// Use after [`require_cardinality`] has bounded `count`: the cap rejects an
/// *adversarially* huge request, while this still recovers gracefully if a
/// *within-cap* but heavyweight reservation exceeds available memory. The
/// payload records `item` (what is being reserved) and `count`. Mirrors LFM2's
/// `layers.try_reserve_exact(...) → AllocFailure` on the decoder-layer `Vec`.
///
/// ```
/// use mlxrs::model_validation::reserve_or_error;
/// let mut v: Vec<u32> = Vec::new();
/// assert!(reserve_or_error(&mut v, "layers", 12).is_ok());
/// assert!(v.capacity() >= 12);
/// ```
pub fn reserve_or_error<R: TryReserve>(
  collection: &mut R,
  item: &'static str,
  count: usize,
) -> Result<()> {
  collection.try_reserve_exact_(count).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "model_validation::reserve_or_error",
      item,
      count as u64,
      e,
    ))
  })
}

/// Collections that support a fallible exact reservation — the abstraction
/// behind [`reserve_or_error`]. Implemented for [`Vec`] and [`HashMap`] (the
/// two config-sized host buffers the loaders build). The trailing-underscore
/// method name avoids shadowing the inherent `try_reserve_exact`.
pub trait TryReserve {
  /// Reserve capacity for exactly `additional` more elements, forwarding the
  /// collection's own `try_reserve_exact`.
  fn try_reserve_exact_(&mut self, additional: usize) -> std::result::Result<(), TryReserveError>;
}

impl<T> TryReserve for Vec<T> {
  #[inline]
  fn try_reserve_exact_(&mut self, additional: usize) -> std::result::Result<(), TryReserveError> {
    self.try_reserve_exact(additional)
  }
}

impl<K, V, S: std::hash::BuildHasher> TryReserve for HashMap<K, V, S>
where
  K: Eq + Hash,
{
  #[inline]
  fn try_reserve_exact_(&mut self, additional: usize) -> std::result::Result<(), TryReserveError> {
    // `HashMap` exposes only `try_reserve` (no `_exact`); it reserves at least
    // `additional`, which satisfies the "room for `additional` more" contract.
    self.try_reserve(additional)
  }
}

// ════════════════════════════ 5. key collision ═════════════════════════════

/// Insert `(key, value)` into `map`, rejecting a **duplicate** key with
/// [`Error::KeyCollision`] instead of silently overwriting.
///
/// A sanitize / weight-rewrite pass that maps two source keys onto the same
/// destination key would otherwise let an arbitrary (per-run nondeterministic,
/// since the source is a `HashMap`) survivor win and silently corrupt the
/// checkpoint. This surfaces the collision as a typed error naming the key.
/// Mirrors Wav2Vec2's `sanitize` need (its `out.insert` assumed unique
/// rewritten keys) and the Vocab "two tokens, one id" rejection.
///
/// On collision the existing entry is left untouched and `value` is dropped.
///
/// Allocation-disciplined like the rest of the toolkit: the duplicate is
/// detected with a non-allocating membership check (so a collision never grows
/// the map), and the vacant path **fallibly reserves** one slot via
/// `HashMap::try_reserve` — mapped to [`Error::AllocFailure`] — *before* the
/// insert, so a sanitize pass with many unique rewritten keys recovers
/// gracefully on allocator failure instead of aborting on `HashMap` growth.
///
/// ```
/// use std::collections::HashMap;
/// use mlxrs::model_validation::insert_unique;
/// let mut m: HashMap<String, u32> = HashMap::new();
/// assert!(insert_unique(&mut m, "encoder.weight".to_string(), 1, "sanitize").is_ok());
/// assert!(insert_unique(&mut m, "encoder.weight".to_string(), 2, "sanitize").is_err());
/// assert_eq!(m["encoder.weight"], 1);
/// ```
pub fn insert_unique<V>(
  map: &mut HashMap<String, V>,
  key: String,
  value: V,
  context: &'static str,
) -> Result<()> {
  // Detect the duplicate WITHOUT allocating: a collision must not grow the map.
  if map.contains_key(&key) {
    return Err(Error::KeyCollision(KeyCollisionPayload::new(context, key)));
  }
  // Vacant path: reserve the one slot fallibly so a map-growth allocator
  // failure surfaces as a typed `AllocFailure` instead of aborting. After a
  // successful `try_reserve(1)` the new-key `insert` cannot reallocate.
  map.try_reserve(1).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "model_validation::insert_unique",
      context,
      1,
      e,
    ))
  })?;
  map.insert(key, value);
  Ok(())
}

// ════════════════════════ 6. config-gated optional weight ══════════════════

/// Validate the presence of a config-flag-gated optional weight without
/// consuming it.
///
/// Mirrors LFM2's `conv_bias`-gated bias contract, expressed as a pure check
/// over whether the key is `present`:
/// - `flag == true` ⇒ the weight is **required**; `present == false` is
///   [`Error::MissingKey`] (a silent run-without-it would be wrong).
/// - `flag == false` ⇒ the weight must be **absent**; `present == true` is
///   [`Error::KeyCollision`] (the checkpoint contradicts the config; a stray
///   tensor would be silently applied — or silently dropped).
/// - the two agreeing cases (`true`/present, `false`/absent) are `Ok`.
///
/// `key` names the weight in the error payload. Use [`take_if`] instead when
/// you also want to drain the tensor out of the map in the same step.
///
/// ```
/// use mlxrs::model_validation::require_if_present;
/// assert!(require_if_present("conv_bias", true, "conv.bias", true).is_ok());
/// assert!(require_if_present("conv_bias", true, "conv.bias", false).is_err());
/// assert!(require_if_present("conv_bias", false, "conv.bias", true).is_err());
/// assert!(require_if_present("conv_bias", false, "conv.bias", false).is_ok());
/// ```
pub fn require_if_present(
  flag_name: &'static str,
  flag: bool,
  key: &str,
  present: bool,
) -> Result<()> {
  match (flag, present) {
    (true, true) | (false, false) => Ok(()),
    (true, false) => Err(Error::MissingKey(MissingKeyPayload::new(flag_name, key))),
    (false, true) => Err(Error::KeyCollision(KeyCollisionPayload::new(
      flag_name, key,
    ))),
  }
}

/// Drain a config-flag-gated optional weight out of `map`, enforcing the same
/// present/absent contract as [`require_if_present`].
///
/// Removes and returns `Some(value)` when `flag == true` and the key is
/// present; returns `Ok(None)` when `flag == false` and the key is absent.
/// Returns [`Error::MissingKey`] for a required-but-absent weight and
/// [`Error::KeyCollision`] for a forbidden-but-present one. The direct
/// reusable form of LFM2's `take_conv_bias`.
///
/// ```
/// use std::collections::HashMap;
/// use mlxrs::model_validation::take_if;
/// let mut m: HashMap<String, u32> = HashMap::from([("conv.bias".to_string(), 7)]);
/// assert_eq!(take_if(&mut m, "conv_bias", true, "conv.bias").unwrap(), Some(7));
/// assert!(m.is_empty());
/// // forbidden but present:
/// let mut m2: HashMap<String, u32> = HashMap::from([("conv.bias".to_string(), 7)]);
/// assert!(take_if(&mut m2, "conv_bias", false, "conv.bias").is_err());
/// ```
pub fn take_if<V>(
  map: &mut HashMap<String, V>,
  flag_name: &'static str,
  flag: bool,
  key: &str,
) -> Result<Option<V>> {
  match (flag, map.remove(key)) {
    (true, Some(v)) => Ok(Some(v)),
    (true, None) => Err(Error::MissingKey(MissingKeyPayload::new(flag_name, key))),
    (false, None) => Ok(None),
    (false, Some(_)) => Err(Error::KeyCollision(KeyCollisionPayload::new(
      flag_name, key,
    ))),
  }
}

// ════════════════════════════ 7. resource extents ══════════════════════════
//
// The bounds / arithmetic / alloc helpers above are correct but applied
// MANUALLY at each call site, so every new buffer / product / entry point is a
// fresh chance to forget one. These types lift them into the type system: a
// config stores `Extent`s (a dimension cannot exceed its cap because the only
// constructor enforces it), a buffer is sized only through `elem_count` (a
// checked product against a total-element cap — the quantity per-axis caps
// miss), and a model-sized buffer is allocated only through `alloc_filled`
// (always fallible). The unsafe path — an unbounded product, an infallible
// `vec![_; n]`, a runtime dimension exceeding the loaded config — becomes
// unwritable rather than caught.

/// A validated dimension: a size capped **at construction**, so an over-cap
/// dimension cannot exist in a built config.
///
/// [`Extent::new`] is the only constructor and it enforces the cap, so a stored
/// `Extent` is itself proof that its value is within bounds — a config that
/// holds `Extent`s (rather than bare `usize`) cannot represent an oversized
/// dimension, and the per-axis bound no longer depends on remembering to call a
/// guard at each use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Extent(usize);

impl Extent {
  /// Construct an `Extent`, enforcing `value <= cap`. Returns
  /// [`Error::CapExceeded`] (naming `field`) otherwise.
  ///
  /// The cap is a **magnitude** bound only; pair with [`require_positive`] /
  /// [`require_cardinality`] where a dimension must additionally be non-zero.
  ///
  /// ```
  /// use mlxrs::model_validation::Extent;
  /// assert_eq!(Extent::new("hidden", 768, 1 << 20).unwrap().get(), 768);
  /// assert!(Extent::new("hidden", 1 << 21, 1 << 20).is_err());
  /// ```
  pub fn new(field: &'static str, value: usize, cap: usize) -> Result<Self> {
    if value > cap {
      return Err(Error::CapExceeded(CapExceededPayload::new(
        field,
        field,
        cap as u64,
        value as u64,
      )));
    }
    Ok(Self(value))
  }

  /// The validated dimension.
  #[inline(always)]
  pub const fn get(self) -> usize {
    self.0
  }
}

/// The only buffer-sizing path: the checked product of `dims` against a total
/// element cap `total_cap`. Returns a count proven both overflow-free and
/// `<= total_cap`.
///
/// Each [`Extent`] is already individually capped; this bounds the **product** —
/// the quantity that actually sizes the buffer and that per-axis caps do not
/// constrain (e.g. a `4096 x 4096` grid, each axis within a per-axis cap, still
/// sizes a 16M-element buffer). Returns [`Error::ArithmeticOverflow`] if the
/// running product overflows `usize`, or [`Error::CapExceeded`] (naming `field`)
/// if it exceeds `total_cap`. An empty `dims` is the empty product `1`.
///
/// ```
/// use mlxrs::model_validation::{Extent, elem_count};
/// let dims = [Extent::new("h", 4, 16).unwrap(), Extent::new("w", 4, 16).unwrap()];
/// assert_eq!(elem_count("buf", &dims, 64).unwrap(), 16);
/// assert!(elem_count("buf", &dims, 8).is_err()); // product 16 > cap 8
/// ```
pub fn elem_count(field: &'static str, dims: &[Extent], total_cap: usize) -> Result<usize> {
  let mut product: usize = 1;
  for d in dims {
    product = product.checked_mul(d.get()).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        field,
        "usize",
        [("running_product", product as u64), ("dim", d.get() as u64)],
      ))
    })?;
  }
  if product > total_cap {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      field,
      field,
      total_cap as u64,
      product as u64,
    )));
  }
  Ok(product)
}

/// The only model-buffer allocator: fallibly allocate a `Vec<T>` of `n` copies
/// of `value`, turning an allocator failure into a typed [`Error::AllocFailure`]
/// (via [`reserve_or_error`]) instead of the abort a bare `vec![value; n]` /
/// `Vec::with_capacity(n)` would raise.
///
/// `n` is expected to come from [`elem_count`] (a bounded count); a bare
/// `vec![_; n]` for a model-sized buffer is the banned idiom this replaces. The
/// reservation is exact, so the fill cannot reallocate.
///
/// `T: Copy` **plus a `resize_with(n, || value)` fill** is deliberate: the fill
/// must run **no user code** that could itself panic or heap-allocate outside
/// [`reserve_or_error`]. `Copy` alone is not enough — a `Copy` type may still
/// carry a hostile manual `Clone`, and a `Clone`-based `Vec::resize` would call
/// it — so each element is produced by **copying** `value` through the closure,
/// never invoking `Clone`. The model buffers this sizes (`u8` / `f32` / `i32`
/// host scratch) are all `Copy`.
///
/// ```
/// use mlxrs::model_validation::alloc_filled;
/// assert_eq!(alloc_filled("buf", 0u8, 4).unwrap(), vec![0u8; 4]);
/// assert!(alloc_filled::<u8>("buf", 0, 0).unwrap().is_empty());
/// ```
pub fn alloc_filled<T: Copy>(field: &'static str, value: T, n: usize) -> Result<Vec<T>> {
  let mut buf: Vec<T> = Vec::new();
  reserve_or_error(&mut buf, field, n)?;
  // `resize_with` produces each element from the closure (a `Copy` of `value`),
  // never calling `Clone::clone` — so even a `Copy` type with a hostile manual
  // `Clone` impl cannot run arbitrary code here. The reservation above is the
  // only allocation; the fill stays within the reserved (exact) capacity.
  buf.resize_with(n, || value);
  Ok(buf)
}

#[cfg(test)]
mod tests;
