//! Unit tests for the architecture-agnostic LM generation surface that do
//! NOT need the full generation loop: `GenConfig` validation / builders /
//! encapsulated accessors, the `FinishReason` taxonomy, the `Debug` impls
//! on `LogitsProcessor` / `Sampler` / `SamplerChain`, the `last_position`
//! shape guards, and the `generate_step` deferred-`Err` channel (empty
//! prompt + invalid `cfg`). Oracles are closed-form, computed by hand from
//! the inputs — no call into the function under test produces the expected.
//!
//! The loop-driven tests (real decode steps, stream/generate, batch) live in
//! the sibling [`super::batch_tests`] / [`super::stop_sequence_tests`]
//! modules and in this file's `loop_paths` section below (reusing the
//! in-crate `crate::lm::model::MockModel` so a single forward + sampler path
//! is exercised against real mlx ops).

use super::*;
use crate::lm::cache::{CacheConfig, KvCache, make_prompt_cache};

// ════════════════════════════════════════════════════════════════════════
//   GenConfig::validate — every typed-error branch (closed-form, no mlx)
// ════════════════════════════════════════════════════════════════════════

/// The all-defaults config validates (`temp == 0` argmax path, every other
/// knob off). Anchors the negative tests below.
#[test]
fn validate_default_config_ok() {
  assert!(GenConfig::default().validate().is_ok());
  // A representative all-features-on but in-range config also validates.
  let cfg = GenConfig {
    temp: 0.7,
    top_p: 0.9,
    min_p: 0.05,
    min_tokens_to_keep: 3,
    top_k: 40,
    xtc_probability: 0.5,
    xtc_threshold: 0.1,
    repetition_penalty: Some(1.1),
    presence_penalty: Some(-0.5), // negative allowed (additive bonus)
    frequency_penalty: Some(0.3),
    logit_bias: vec![(7, 2.5), (9, -1.0)],
    ..Default::default()
  };
  assert!(
    cfg.validate().is_ok(),
    "in-range all-knobs config validates"
  );
}

/// `temp` NaN / Inf ⇒ `NonFiniteScalar`; `temp < 0` ⇒ `OutOfRange`. The
/// payload `context` pins WHICH bound fired (so a later-field bug can't
/// masquerade as a temp failure).
#[test]
fn validate_temp_non_finite_and_negative() {
  for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
    let err = GenConfig::default().with_temp(bad).validate().unwrap_err();
    assert!(
      matches!(err, Error::NonFiniteScalar(ref p) if p.context().contains("temp")),
      "temp={bad} ⇒ NonFiniteScalar(temp), got {err:?}"
    );
  }
  let err = GenConfig::default().with_temp(-0.5).validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context().contains("temp")),
    "temp=-0.5 ⇒ OutOfRange(temp), got {err:?}"
  );
}

/// `top_p`: NaN ⇒ NonFiniteScalar; outside `[0, 1]` ⇒ OutOfRange. A finite
/// `temp > 0` is set so validation reaches the `top_p` checks (which come
/// after temp).
#[test]
fn validate_top_p_bounds() {
  let base = || GenConfig::default().with_temp(0.8);
  let err = {
    let mut c = base();
    c.top_p = f32::NAN;
    c
  }
  .validate()
  .unwrap_err();
  assert!(
    matches!(err, Error::NonFiniteScalar(ref p) if p.context().contains("top_p")),
    "got {err:?}"
  );
  for bad in [-0.1f32, 1.5] {
    let mut c = base();
    c.top_p = bad;
    let err = c.validate().unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(ref p) if p.context().contains("top_p")),
      "top_p={bad} ⇒ OutOfRange(top_p), got {err:?}"
    );
  }
}

/// `min_p`: NaN ⇒ NonFiniteScalar; outside `[0, 1]` ⇒ OutOfRange.
#[test]
fn validate_min_p_bounds() {
  let mut c = GenConfig::default().with_temp(0.8);
  c.min_p = f32::INFINITY;
  let err = c.validate().unwrap_err();
  assert!(
    matches!(err, Error::NonFiniteScalar(ref p) if p.context().contains("min_p")),
    "got {err:?}"
  );
  let mut c = GenConfig::default().with_temp(0.8);
  c.min_p = 2.0;
  let err = c.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context().contains("min_p")),
    "got {err:?}"
  );
}

/// `min_tokens_to_keep < 1` ⇒ OutOfRange.
#[test]
fn validate_min_tokens_to_keep_must_be_positive() {
  let mut c = GenConfig::default().with_temp(0.8);
  c.min_tokens_to_keep = 0;
  let err = c.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context().contains("min_tokens_to_keep")),
    "got {err:?}"
  );
  // negative is also rejected.
  let mut c = GenConfig::default().with_temp(0.8);
  c.min_tokens_to_keep = -3;
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

/// `top_k < 0` ⇒ OutOfRange; `top_k == 0` (off) is accepted.
#[test]
fn validate_top_k_non_negative() {
  let mut c = GenConfig::default().with_temp(0.8);
  c.top_k = -1;
  let err = c.validate().unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context().contains("top_k")),
    "got {err:?}"
  );
  let mut ok = GenConfig::default().with_temp(0.8);
  ok.top_k = 0;
  assert!(ok.validate().is_ok(), "top_k == 0 is 'off', accepted");
}

/// `xtc_probability`: NaN ⇒ NonFiniteScalar; outside `[0, 1]` ⇒ OutOfRange.
#[test]
fn validate_xtc_probability_bounds() {
  let mut c = GenConfig::default().with_temp(0.8);
  c.xtc_probability = f32::NAN;
  assert!(
    matches!(c.validate(), Err(Error::NonFiniteScalar(ref p)) if p.context().contains("xtc_probability"))
  );
  let mut c = GenConfig::default().with_temp(0.8);
  c.xtc_probability = 1.5;
  assert!(
    matches!(c.validate(), Err(Error::OutOfRange(ref p)) if p.context().contains("xtc_probability"))
  );
}

/// `xtc_threshold`: NaN ⇒ NonFiniteScalar; outside `[0, 0.5]` ⇒ OutOfRange
/// (the upper bound is 0.5, not 1.0 — pin that exactly).
#[test]
fn validate_xtc_threshold_bounds() {
  let mut c = GenConfig::default().with_temp(0.8);
  c.xtc_threshold = f32::NEG_INFINITY;
  assert!(
    matches!(c.validate(), Err(Error::NonFiniteScalar(ref p)) if p.context().contains("xtc_threshold"))
  );
  // 0.6 > 0.5 upper bound.
  let mut c = GenConfig::default().with_temp(0.8);
  c.xtc_threshold = 0.6;
  assert!(
    matches!(c.validate(), Err(Error::OutOfRange(ref p)) if p.context().contains("xtc_threshold"))
  );
  // exactly 0.5 is the inclusive upper bound ⇒ ok.
  let mut ok = GenConfig::default().with_temp(0.8);
  ok.xtc_threshold = 0.5;
  assert!(ok.validate().is_ok(), "xtc_threshold == 0.5 is in-range");
}

/// `repetition_penalty = Some(non-finite)` ⇒ NonFiniteScalar;
/// `Some(negative)` ⇒ OutOfRange; `Some(0.0)` / `None` accepted (off).
#[test]
fn validate_repetition_penalty_bounds() {
  let c = GenConfig {
    repetition_penalty: Some(f32::NAN),
    ..Default::default()
  };
  assert!(
    matches!(c.validate(), Err(Error::NonFiniteScalar(ref p)) if p.context().contains("repetition_penalty"))
  );
  let c = GenConfig {
    repetition_penalty: Some(-0.2),
    ..Default::default()
  };
  assert!(
    matches!(c.validate(), Err(Error::OutOfRange(ref p)) if p.context().contains("repetition_penalty"))
  );
  let ok = GenConfig {
    repetition_penalty: Some(0.0),
    ..Default::default()
  };
  assert!(
    ok.validate().is_ok(),
    "Some(0.0) repetition penalty is 'off'"
  );
}

/// `presence_penalty` / `frequency_penalty` are finite-only — negatives are
/// allowed (additive bonus), only NaN / ±Inf are rejected (NonFiniteScalar).
#[test]
fn validate_presence_and_frequency_penalty_finite_only() {
  let c = GenConfig {
    presence_penalty: Some(f32::INFINITY),
    ..Default::default()
  };
  assert!(
    matches!(c.validate(), Err(Error::NonFiniteScalar(ref p)) if p.context().contains("presence_penalty")),
    "presence_penalty Inf ⇒ NonFiniteScalar"
  );
  let c = GenConfig {
    frequency_penalty: Some(f32::NAN),
    ..Default::default()
  };
  assert!(
    matches!(c.validate(), Err(Error::NonFiniteScalar(ref p)) if p.context().contains("frequency_penalty")),
    "frequency_penalty NaN ⇒ NonFiniteScalar"
  );
  // Negative values are accepted for both (additive bonus, not a penalty).
  let ok = GenConfig {
    presence_penalty: Some(-2.0),
    frequency_penalty: Some(-1.5),
    ..Default::default()
  };
  assert!(ok.validate().is_ok(), "negative presence/frequency allowed");
}

/// `logit_bias` with a non-finite VALUE ⇒ NonFiniteScalar (the id is not
/// bounded here). A finite-valued bias list validates.
#[test]
fn validate_logit_bias_value_finite() {
  let c = GenConfig::default().with_logit_bias(vec![(3i32, 1.0f32), (4, f32::NAN)]);
  assert!(
    matches!(c.validate(), Err(Error::NonFiniteScalar(ref p)) if p.context().contains("logit_bias")),
    "a NaN bias value ⇒ NonFiniteScalar(logit_bias value)"
  );
  let ok = GenConfig::default().with_logit_bias(vec![(3i32, -100.0f32), (4, 100.0)]);
  assert!(ok.validate().is_ok(), "finite (even large) bias values ok");
}

// ════════════════════════════════════════════════════════════════════════
//   GenConfig builders + encapsulated accessors (pure, no mlx)
// ════════════════════════════════════════════════════════════════════════

/// `GenConfig::new()` equals `Default::default()` field-for-field on the
/// observable surface.
#[test]
fn gen_config_new_equals_default() {
  let n = GenConfig::new();
  let d = GenConfig::default();
  assert_eq!(n.max_tokens, d.max_tokens);
  assert_eq!(n.prefill_step_size, d.prefill_step_size);
  assert_eq!(n.temp, d.temp);
  assert_eq!(n.collect_logprobs, d.collect_logprobs);
  assert_eq!(n.eos_slice(), d.eos_slice());
  assert_eq!(n.stop_strings_slice(), d.stop_strings_slice());
  assert_eq!(n.logit_bias_slice(), d.logit_bias_slice());
  assert_eq!(n.xtc_special_tokens_slice(), d.xtc_special_tokens_slice());
  // Defaults are exactly mlx-lm's.
  assert_eq!(n.max_tokens, 256);
  assert_eq!(n.prefill_step_size, 2048);
  assert_eq!(n.temp, 0.0);
  assert_eq!(n.min_tokens_to_keep, 1);
  assert_eq!(n.repetition_context_size, DEFAULT_REPETITION_CONTEXT_SIZE);
  assert_eq!(DEFAULT_REPETITION_CONTEXT_SIZE, 20);
  assert!(!n.collect_logprobs);
  assert!(n.eos_slice().is_empty());
}

/// The `with_*` consuming builders set exactly the field they name and the
/// encapsulated `*_slice` accessors read it back.
#[test]
fn gen_config_with_builders_and_slice_accessors() {
  let cfg = GenConfig::new()
    .with_max_tokens(7)
    .with_prefill_step_size(13)
    .with_temp(0.5)
    .with_xtc_special_tokens(vec![1i32, 2, 3])
    .with_logit_bias(vec![(5i32, 1.5f32)])
    .with_eos(vec![2u32, 9])
    .with_stop_strings(vec!["END".to_string(), "STOP".to_string()]);
  assert_eq!(cfg.max_tokens, 7);
  assert_eq!(cfg.prefill_step_size, 13);
  assert_eq!(cfg.temp, 0.5);
  assert_eq!(cfg.xtc_special_tokens_slice(), &[1, 2, 3]);
  assert_eq!(cfg.logit_bias_slice(), &[(5, 1.5)]);
  assert_eq!(cfg.eos_slice(), &[2, 9]);
  assert_eq!(
    cfg.stop_strings_slice(),
    &["END".to_string(), "STOP".to_string()]
  );
}

/// The `set_*` in-place setters mutate `&mut self` and chain (returning
/// `&mut Self`); the final state reflects the last-applied value.
#[test]
fn gen_config_set_inplace_setters_chain() {
  let mut cfg = GenConfig::new();
  cfg
    .set_xtc_special_tokens(vec![4i32, 5])
    .set_logit_bias(vec![(8i32, -2.0f32), (9, 3.0)])
    .set_eos(vec![2u32])
    .set_stop_strings(vec!["<|end|>".to_string()]);
  assert_eq!(cfg.xtc_special_tokens_slice(), &[4, 5]);
  assert_eq!(cfg.logit_bias_slice(), &[(8, -2.0), (9, 3.0)]);
  assert_eq!(cfg.eos_slice(), &[2]);
  assert_eq!(cfg.stop_strings_slice(), &["<|end|>".to_string()]);
  // A second set_* overrides (proves the setter assigns, not appends).
  cfg.set_eos(vec![1u32, 2, 3]);
  assert_eq!(cfg.eos_slice(), &[1, 2, 3]);
}

// ════════════════════════════════════════════════════════════════════════
//   FinishReason taxonomy (pure, no mlx)
// ════════════════════════════════════════════════════════════════════════

/// `as_str()` collapses both `Eos` and `Stop(_)` to the canonical OpenAI
/// tag `"stop"`, and `Length` to `"length"`; `Display` matches `as_str`.
#[test]
fn finish_reason_as_str_and_display() {
  assert_eq!(FinishReason::Eos.as_str(), "stop");
  assert_eq!(FinishReason::Length.as_str(), "length");
  assert_eq!(FinishReason::Stop("xyz".to_string()).as_str(), "stop");
  // Display uses as_str (the `#[display("{}", self.as_str())]` attr).
  assert_eq!(format!("{}", FinishReason::Eos), "stop");
  assert_eq!(format!("{}", FinishReason::Length), "length");
  assert_eq!(format!("{}", FinishReason::Stop("abc".to_string())), "stop");
}

/// `stop_sequence()` returns the matched string ONLY for `Stop(_)`; `None`
/// for `Eos` / `Length`.
#[test]
fn finish_reason_stop_sequence_payload() {
  assert_eq!(
    FinishReason::Stop("</done>".to_string()).stop_sequence(),
    Some("</done>")
  );
  assert_eq!(FinishReason::Eos.stop_sequence(), None);
  assert_eq!(FinishReason::Length.stop_sequence(), None);
}

/// The `derive_more::IsVariant` predicates + `PartialEq` are consistent with
/// the constructed variant.
#[test]
fn finish_reason_is_variant_predicates() {
  assert!(FinishReason::Eos.is_eos());
  assert!(!FinishReason::Eos.is_length());
  assert!(!FinishReason::Eos.is_stop());
  assert!(FinishReason::Length.is_length());
  assert!(FinishReason::Stop("s".to_string()).is_stop());
  assert!(!FinishReason::Stop("s".to_string()).is_eos());
  // Eq distinguishes payloads.
  assert_eq!(
    FinishReason::Stop("a".into()),
    FinishReason::Stop("a".into())
  );
  assert_ne!(
    FinishReason::Stop("a".into()),
    FinishReason::Stop("b".into())
  );
  assert_ne!(FinishReason::Eos, FinishReason::Length);
}

// ════════════════════════════════════════════════════════════════════════
//   Debug impls — LogitsProcessor / Sampler / SamplerChain
// ════════════════════════════════════════════════════════════════════════

/// `Debug for LogitsProcessor` renders each variant's struct name + the
/// fields the hand-written impl exposes. The penalty payloads are pure; the
/// `LogitBias` variant carries an `Array` built once via `from_slice`.
#[test]
fn logits_processor_debug_all_variants() {
  let rep = LogitsProcessor::RepetitionPenalty(RepetitionPenaltyPayload::new(1.3, 17));
  let s = format!("{rep:?}");
  assert!(s.contains("RepetitionPenalty"), "got {s}");
  assert!(s.contains("1.3") && s.contains("17"), "fields shown: {s}");

  let pres = LogitsProcessor::PresencePenalty(PresencePenaltyPayload::new(0.4, 11));
  let s = format!("{pres:?}");
  assert!(s.contains("PresencePenalty") && s.contains("0.4") && s.contains("11"));

  let freq = LogitsProcessor::FrequencyPenalty(FrequencyPenaltyPayload::new(0.25, 5));
  let s = format!("{freq:?}");
  assert!(s.contains("FrequencyPenalty") && s.contains("0.25") && s.contains('5'));

  // LogitBias: the Debug shows the index count `n`, not the array contents.
  let values = Array::from_slice::<f32>(&[1.0f32, 2.0, 3.0], &(3usize,)).unwrap();
  let bias = LogitsProcessor::LogitBias(LogitBiasPayload::new(vec![10, 20, 30], values));
  let s = format!("{bias:?}");
  assert!(s.contains("LogitBias"), "got {s}");
  assert!(s.contains('3'), "n == 3 indices shown: {s}");
  // The payload accessors round-trip.
  if let LogitsProcessor::LogitBias(p) = &bias {
    assert_eq!(p.indices_slice(), &[10, 20, 30]);
    assert_eq!(p.values_ref().shape(), vec![3]);
  } else {
    panic!("constructed LogitBias variant");
  }

  // Custom variant Debug is the bare tuple name.
  let custom = LogitsProcessor::Custom(Box::new(|_t: &[u32], a: &Array| a.try_clone()));
  assert!(format!("{custom:?}").contains("Custom"));

  // IsVariant predicates line up with each constructor.
  assert!(rep.is_repetition_penalty());
  assert!(pres.is_presence_penalty());
  assert!(freq.is_frequency_penalty());
  assert!(bias.is_logit_bias());
  assert!(custom.is_custom());
}

/// `Debug for Sampler`: `Argmax` is a bare string; `Custom` is a tuple name.
/// (The `Chain` arm's Debug is covered separately so its `make_sampler`
/// allocation is isolated.)
#[test]
fn sampler_debug_argmax_and_custom() {
  assert_eq!(format!("{:?}", Sampler::Argmax), "Argmax");
  let custom = Sampler::custom(|a: &Array| a.try_clone());
  assert!(format!("{custom:?}").contains("Custom"));
}

/// `Debug for Sampler::Chain` + `Debug for SamplerChain`: a seeded stochastic
/// `make_sampler` builds a `Chain`, whose Debug nests the `SamplerChain`
/// struct fields. Needs the mlx PRNG-key allocation, so it is grouped with
/// the loop-path tests' mlx usage.
#[test]
fn sampler_chain_debug_renders_struct_fields() {
  // temp != 0 with at least one stage on ⇒ a Chain (not Argmax).
  let sampler = make_sampler(
    0.8, // temp
    0.0, // top_p (off)
    0.0, // min_p (off)
    1,   // min_tokens_to_keep
    40,  // top_k (on)
    0.0, // xtc_probability (off)
    0.0, // xtc_threshold
    &[], // xtc_special_tokens
    Some(1234),
  )
  .expect("make_sampler builds a Chain");
  assert!(matches!(sampler, Sampler::Chain(_)), "stochastic ⇒ Chain");
  let s = format!("{sampler:?}");
  assert!(s.contains("Chain"), "outer Sampler::Chain Debug: {s}");
  assert!(s.contains("SamplerChain"), "nested chain Debug: {s}");
  assert!(s.contains("temp"), "chain fields shown: {s}");
  assert!(s.contains("top_p") && s.contains("min_p") && s.contains("top_k"));
}

/// `make_sampler` with `temp == 0` short-circuits to the pure `Argmax`
/// variant regardless of the other knobs (mlx-lm `make_sampler` line 46).
#[test]
fn make_sampler_temp_zero_is_argmax() {
  let sampler = make_sampler(0.0, 0.9, 0.1, 1, 40, 0.5, 0.1, &[7], Some(1)).unwrap();
  assert!(
    matches!(sampler, Sampler::Argmax),
    "temp == 0 ⇒ Argmax short-circuit"
  );
}

// ════════════════════════════════════════════════════════════════════════
//   make_logits_processors — gated chain construction (pure, no mlx eval)
// ════════════════════════════════════════════════════════════════════════

/// Empty bias + all-`None`/`Some(0.0)` penalties ⇒ NO processors (every
/// stage gated off). `Some(0.0)` is "off" because mlx-lm only includes a
/// penalty when `penalty != 0`.
#[test]
fn make_logits_processors_all_off_is_empty() {
  let procs = make_logits_processors(&[], None, 20, Some(0.0), 20, None, 20).unwrap();
  assert!(
    procs.is_empty(),
    "no bias + zero/none penalties ⇒ empty chain"
  );
}

/// Every stage on ⇒ processors built in mlx-lm order: LogitBias, then
/// repetition, presence, frequency. The variant order + types are the
/// closed-form oracle (no array eval needed — construction only).
#[test]
fn make_logits_processors_full_chain_order() {
  let procs = make_logits_processors(
    &[(3, 1.0), (4, -1.0)],
    Some(1.1),
    8,
    Some(0.5),
    9,
    Some(0.2),
    10,
  )
  .unwrap();
  assert_eq!(procs.len(), 4, "bias + 3 penalties");
  assert!(procs[0].is_logit_bias());
  assert!(procs[1].is_repetition_penalty());
  assert!(procs[2].is_presence_penalty());
  assert!(procs[3].is_frequency_penalty());
  // Each penalty captured its OWN independent context window.
  if let LogitsProcessor::RepetitionPenalty(p) = &procs[1] {
    assert_eq!(p.context_size(), 8);
    assert_eq!(p.penalty(), 1.1);
  }
  if let LogitsProcessor::PresencePenalty(p) = &procs[2] {
    assert_eq!(p.context_size(), 9);
    assert_eq!(p.penalty(), 0.5);
  }
  if let LogitsProcessor::FrequencyPenalty(p) = &procs[3] {
    assert_eq!(p.context_size(), 10);
    assert_eq!(p.penalty(), 0.2);
  }
}

// ════════════════════════════════════════════════════════════════════════
//   last_position — degenerate-shape guards (small Array, no decode loop)
// ════════════════════════════════════════════════════════════════════════

/// `last_position` rejects a non-rank-3 logits tensor with `RankMismatch`
/// (the faithful equivalent of mlx-lm's `logits[:, -1, :]` only being
/// defined for `[B, S, V]`). The matcher pins the observed rank.
#[test]
fn last_position_rejects_non_rank3() {
  // rank-2 [B, V]
  let two = Array::from_slice::<f32>(&[1.0f32, 2.0, 3.0, 4.0], &(2usize, 2usize)).unwrap();
  let err = last_position(&two).unwrap_err();
  assert!(
    matches!(err, Error::RankMismatch(ref p) if p.actual() == 2),
    "rank-2 ⇒ RankMismatch(actual=2), got {err:?}"
  );
  // rank-1 [V]
  let one = Array::from_slice::<f32>(&[1.0f32, 2.0], &(2usize,)).unwrap();
  let err = last_position(&one).unwrap_err();
  assert!(
    matches!(err, Error::RankMismatch(ref p) if p.actual() == 1),
    "rank-1 ⇒ RankMismatch(actual=1), got {err:?}"
  );
}

/// `last_position` rejects a zero-length sequence axis (`S == 0`) or empty
/// vocab axis (`V == 0`) with `OutOfRange` BEFORE the `s - 1` index, so a
/// zero `S` can never underflow.
#[test]
fn last_position_rejects_zero_s_or_v() {
  // S == 0: shape [1, 0, 3] — no data, valid empty array.
  let empty_s = Array::from_slice::<f32>(&[], &(1usize, 0usize, 3usize)).unwrap();
  let err = last_position(&empty_s).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context().contains("S and V")),
    "S == 0 ⇒ OutOfRange, got {err:?}"
  );
  // V == 0: shape [1, 2, 0].
  let empty_v = Array::from_slice::<f32>(&[], &(1usize, 2usize, 0usize)).unwrap();
  let err = last_position(&empty_v).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "V == 0 ⇒ OutOfRange, got {err:?}"
  );
}

/// `last_position` on a well-formed `[B, S, V]` returns the final-position
/// `[B, V]` slice. Closed-form oracle: a `[1, 2, 3]` tensor whose last row
/// (`S=1`) is `[7, 8, 9]` ⇒ `[1, 3]` == `[7, 8, 9]`.
#[test]
fn last_position_extracts_final_row() {
  // row-major [B=1, S=2, V=3]: first position [1,2,3], last [7,8,9].
  let data = [1.0f32, 2.0, 3.0, 7.0, 8.0, 9.0];
  let logits = Array::from_slice::<f32>(&data, &(1usize, 2usize, 3usize)).unwrap();
  let mut last = last_position(&logits).unwrap();
  assert_eq!(last.shape(), vec![1, 3], "[B, V] after dropping the S axis");
  assert_eq!(last.to_vec::<f32>().unwrap(), vec![7.0, 8.0, 9.0]);
}

// ════════════════════════════════════════════════════════════════════════
//   generate_step deferred-Err channel (empty prompt + invalid cfg)
// ════════════════════════════════════════════════════════════════════════

/// An empty prompt is a deferred `Err(EmptyInput)` yielded on the FIRST
/// `next()` (before any model call), after which the iterator fuses.
#[test]
fn generate_step_empty_prompt_is_deferred_err_then_fuses() {
  let model = crate::lm::model::MockModel::new(8);
  let cache: Vec<Box<dyn KvCache>> = Vec::new();
  let mut it = generate_step(&model, &[], cache, GenConfig::default());
  let err = it.next().expect("yields one item").unwrap_err();
  assert!(
    matches!(err, Error::EmptyInput(ref p) if p.context().contains("prompt")),
    "empty prompt ⇒ EmptyInput(prompt), got {err:?}"
  );
  assert!(it.next().is_none(), "fuses after the deferred Err");
}

/// An invalid `cfg` (negative temp) surfaces as the iterator's first `Err`
/// through the same deferred channel, even with a valid prompt — proving
/// `cfg.validate()` runs inside the build closure before any forward.
#[test]
fn generate_step_invalid_cfg_is_deferred_err() {
  let model = crate::lm::model::MockModel::new(8);
  let cache = make_prompt_cache(&CacheConfig {
    num_hidden_layers: 1,
    sliding_window: None,
  });
  let cfg = GenConfig::default().with_temp(-1.0);
  let mut it = generate_step(&model, &[1u32, 2], cache, cfg);
  let err = it.next().expect("yields one item").unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context().contains("temp")),
    "invalid temp ⇒ deferred OutOfRange(temp), got {err:?}"
  );
  assert!(it.next().is_none(), "fuses after the deferred Err");
}

// ════════════════════════════════════════════════════════════════════════
//   Single-seq decode-loop normalization paths (real mlx ops)
// ════════════════════════════════════════════════════════════════════════
//
// These drive `Generator::step`'s 3-way `(needs_normalization, temp>0)`
// branch through `generate_step` with the in-crate `MockModel` (greedy +
// stochastic). The greedy/full-normalization arms are deterministic; the
// stochastic opt-out arm asserts only the RNG-independent structural
// contract (step count + finish-by-length).

fn cache1() -> Vec<Box<dyn KvCache>> {
  make_prompt_cache(&CacheConfig {
    num_hidden_layers: 1,
    sliding_window: None,
  })
}

/// `(needs_normalization=true, _)` arm via `collect_logprobs=true` on a
/// greedy run: the full `logits - logsumexp` runs and the yielded logprobs
/// are `Some([V])`. Closed-form oracle: `MockModel::new(4)` has logits
/// `[0,1,2,3]` ⇒ argmax token 3, and `sum(exp(logprobs)) == 1`.
#[test]
fn step_full_normalization_collect_logprobs_greedy() {
  let model = crate::lm::model::MockModel::new(4); // argmax == 3
  let cfg = {
    let mut c = GenConfig::default().with_max_tokens(1);
    c.collect_logprobs = true;
    c
  };
  let step = generate_step(&model, &[1u32], cache1(), cfg)
    .next()
    .unwrap()
    .unwrap();
  assert_eq!(step.token, 3, "greedy argmax of [0,1,2,3]");
  assert_eq!(step.step_index, 0, "first step is index 0");
  assert!(
    step.finish_reason.is_none(),
    "no eos configured, mid-run step"
  );
  let mut lp = step.logprobs.expect("collect_logprobs=true ⇒ Some");
  assert_eq!(lp.shape(), vec![4], "logprobs squeezed to [V]");
  let v = lp.to_vec::<f32>().unwrap();
  let s: f32 = v.iter().map(|x| x.exp()).sum();
  assert!((s - 1.0).abs() < 1e-4, "exp(logprobs) sums to 1, got {s}");
  // log-softmax is monotonic with logits ⇒ argmax preserved at index 3.
  assert!(v[3] > v[2] && v[2] > v[1] && v[1] > v[0]);
}

/// `(needs_normalization=false, temp>0=true)` arm: a seeded stochastic run
/// (no top_p, no collect_logprobs) takes the cheap `max + subtract`
/// max-shift path. Oracle is RNG-independent: the loop produces exactly
/// `max_tokens` tokens (no eos in this run) and every token is in-vocab.
#[test]
fn step_stochastic_opt_out_max_shift_path_runs() {
  // Peaked logits so the categorical draw is well-defined; the assertion
  // does NOT depend on which token is drawn.
  let model = crate::lm::model::MockModel::new(6);
  let cfg = {
    let mut c = GenConfig::default().with_max_tokens(4).with_temp(0.9);
    c.seed = Some(777);
    c.collect_logprobs = false; // opt-out path
    c
  };
  let steps: Vec<GenStep> = generate_step(&model, &[1u32, 2], cache1(), cfg)
    .map(|r| r.unwrap())
    .collect();
  assert_eq!(steps.len(), 4, "stochastic run yields exactly max_tokens");
  for (i, s) in steps.iter().enumerate() {
    assert_eq!(s.step_index, i, "step_index is the 0-based position");
    assert!((s.token as usize) < 6, "sampled token is in-vocab");
    assert!(s.logprobs.is_none(), "collect_logprobs=false ⇒ None");
    assert!(s.finish_reason.is_none(), "no eos ⇒ no terminal reason");
  }
}

/// `(needs_normalization=false, temp>0=false)` arm: pure-greedy raw-logit
/// path (the true zero-cost opt-out). Deterministic argmax, no logprobs.
#[test]
fn step_pure_greedy_raw_logit_path() {
  let model = crate::lm::model::MockModel::new(5); // argmax == 4
  let cfg = GenConfig::default().with_max_tokens(3); // temp 0, no logprobs
  let toks: Vec<u32> = generate_step(&model, &[1u32], cache1(), cfg)
    .map(|r| r.unwrap().token)
    .collect();
  assert_eq!(
    toks,
    vec![4, 4, 4],
    "greedy argmax repeated, no normalization"
  );
}

/// The EOS-token step carries `Some(FinishReason::Eos)` and the iterator
/// fuses after yielding it (mlx-lm yields the eos token then stops). Driven
/// at the `generate_step` layer (token-only) with an explicit eos set.
#[test]
fn step_eos_token_carries_eos_reason_and_fuses() {
  // argmax == 4; configure eos = {4} so the first decode token is eos.
  let model = crate::lm::model::MockModel::new(5);
  let cfg = GenConfig::default()
    .with_max_tokens(10)
    .with_eos(vec![4u32]);
  let mut it = generate_step(&model, &[1u32], cache1(), cfg);
  let step = it.next().unwrap().unwrap();
  assert_eq!(step.token, 4);
  assert_eq!(
    step.finish_reason,
    Some(FinishReason::Eos),
    "eos step tagged"
  );
  assert!(it.next().is_none(), "iterator fuses after the eos token");
}
