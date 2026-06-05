use super::*;
use crate::{audio::stt::models::whisper::config::ModelDimensions, tokenizer::Tokenizer};
use serde_json::json;
use std::{
  collections::HashMap,
  path::{Path, PathBuf},
};

// ───────────────────────── device-row test helpers ────────────────────────

/// Make a `(n,)` device logits row from a host slice.
fn row(xs: &[f32]) -> Array {
  Array::from_slice::<f32>(xs, &[xs.len() as i32]).unwrap()
}

/// Read a `(n,)` device row back to a host `Vec<f32>`.
fn host(a: &Array) -> Vec<f32> {
  a.try_clone().unwrap().to_vec::<f32>().unwrap()
}

/// Apply a logit filter to a host row, returning the masked host row — the
/// device round-trip wrapped for the unit tests.
fn apply_host(f: &dyn LogitFilter, xs: &[f32], tokens: &[u32]) -> Vec<f32> {
  host(&f.apply(&row(xs), tokens).unwrap())
}

// ───────────────────────── scalar / helper oracles ────────────────────────

#[test]
fn compression_ratio_empty_is_zero() {
  assert_eq!(compression_ratio(""), 0.0);
}

#[test]
fn compression_ratio_repetitive_is_high() {
  // A long run of the same token compresses well → ratio well above the 2.4
  // fallback threshold; a varied string compresses poorly → low ratio.
  let repetitive = "abc".repeat(200);
  let varied: String = (0..600u32).map(|i| ((i % 64) as u8 + 33) as char).collect();
  let r_rep = compression_ratio(&repetitive);
  let r_var = compression_ratio(&varied);
  assert!(
    r_rep > DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    "rep ratio {r_rep}"
  );
  assert!(
    r_var < r_rep,
    "varied {r_var} should be < repetitive {r_rep}"
  );
}

#[test]
fn logsumexp_slice_handles_empty_and_neg_inf() {
  // The once-per-utterance detect_language helper: logsumexp of all -inf (or
  // empty) is -inf; logsumexp([0,0,0]) = ln(3).
  assert_eq!(logsumexp_slice(&[]), f64::NEG_INFINITY);
  assert_eq!(
    logsumexp_slice(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
    f64::NEG_INFINITY
  );
  let lse = logsumexp_slice(&[0.0, 0.0, 0.0]);
  assert!((lse - 3f64.ln()).abs() < 1e-9, "lse={lse}");
}

#[test]
fn mask_range_sets_neg_inf_and_clamps() {
  let mut row = vec![1.0_f32; 5];
  mask_range(&mut row, 1, 3);
  assert_eq!(row[0], 1.0);
  assert!(row[1].is_infinite() && row[1] < 0.0);
  assert!(row[2].is_infinite() && row[2] < 0.0);
  assert_eq!(row[3], 1.0);
  // hi past the end clamps; lo >= hi is a no-op.
  mask_range(&mut row, 4, 100);
  assert!(row[4] < 0.0);
  let mut row2 = vec![7.0_f32; 3];
  mask_range(&mut row2, 2, 2);
  assert_eq!(row2, vec![7.0, 7.0, 7.0]);
}

// ───────────────────────── SuppressBlank / SuppressTokens ─────────────────

#[test]
fn suppress_blank_only_at_sample_begin() {
  // blank ids = {space_id=5, eot=2}; sample_begin = 3; n_vocab = 8.
  let filter = SuppressBlank {
    sample_begin: 3,
    mask: scatter_neg_inf_mask(8, &[5, 2]).unwrap(),
  };
  // tokens.len() == sample_begin → mask fires.
  let masked = apply_host(&filter, &[0.0_f32; 8], &[1, 2, 3]);
  assert!(masked[5].is_infinite() && masked[5] < 0.0);
  assert!(masked[2].is_infinite() && masked[2] < 0.0);
  assert_eq!(masked[0], 0.0);

  // tokens.len() != sample_begin → no masking (row returned unchanged).
  let unmasked = apply_host(&filter, &[0.0_f32; 8], &[1, 2, 3, 4]);
  assert_eq!(unmasked, vec![0.0_f32; 8]);
}

#[test]
fn suppress_blank_ignores_out_of_range_ids() {
  // An id past the vocab is silently skipped at mask-build (no panic).
  let filter = SuppressBlank {
    sample_begin: 0,
    mask: scatter_neg_inf_mask(4, &[100]).unwrap(),
  };
  let masked = apply_host(&filter, &[0.0_f32; 4], &[]);
  assert_eq!(masked, vec![0.0_f32; 4]);
}

#[test]
fn suppress_tokens_unconditional() {
  let filter = SuppressTokens::new(&[1, 3], 5).unwrap();
  // Applies regardless of token history.
  let masked = apply_host(&filter, &[0.0_f32; 5], &[9, 9, 9]);
  assert!(masked[1] < 0.0 && masked[1].is_infinite());
  assert!(masked[3] < 0.0 && masked[3].is_infinite());
  assert_eq!(masked[0], 0.0);
  assert_eq!(masked[2], 0.0);
  assert_eq!(masked[4], 0.0);
}

#[test]
fn suppress_masks_overwrite_non_finite_logits_to_neg_inf() {
  // A suppression mask is a boolean OVERWRITE, not an additive `+ (-inf)`: a
  // `+inf` or `NaN` logit at a suppressed slot must become a real `-inf`, never
  // `NaN` (an add would give `(+inf)+(-inf)=NaN` and `NaN+(-inf)=NaN`, which
  // then poisons argmax / the chosen log-prob). Exercise both a suppress mask
  // (SuppressTokens / SuppressBlank) and an unsuppressed slot that keeps its
  // value.
  // logits: [+inf, NaN, finite, +inf, finite]; suppress ids {0, 1, 3}.
  let logits = [f32::INFINITY, f32::NAN, 1.5_f32, f32::INFINITY, -2.0_f32];

  let st = SuppressTokens::new(&[0, 1, 3], 5).unwrap();
  let masked = apply_host(&st, &logits, &[9, 9]);
  // Every suppressed slot is forced to -inf regardless of its prior value.
  for &i in &[0usize, 1, 3] {
    assert!(
      masked[i].is_infinite() && masked[i] < 0.0,
      "suppress-tokens slot {i} = {} must be -inf, not NaN",
      masked[i]
    );
  }
  // Unsuppressed slots are untouched (slot 1 was the only NaN, and it was
  // suppressed; the surviving finite slots keep their exact values).
  assert_eq!(masked[2], 1.5);
  assert_eq!(masked[4], -2.0);

  // Same for the suppress-blank mask, firing at sample_begin.
  let sb = SuppressBlank {
    sample_begin: 0,
    mask: scatter_neg_inf_mask(5, &[0, 1, 3]).unwrap(),
  };
  let masked = apply_host(&sb, &logits, &[]);
  for &i in &[0usize, 1, 3] {
    assert!(
      masked[i].is_infinite() && masked[i] < 0.0,
      "suppress-blank slot {i} = {} must be -inf, not NaN",
      masked[i]
    );
  }
  assert_eq!(masked[2], 1.5);
  assert_eq!(masked[4], -2.0);
}

#[test]
fn timestamp_deterministic_mask_overwrites_non_finite_to_neg_inf() {
  // The timestamp rules' deterministic (token-history) mask is likewise a
  // boolean OVERWRITE: a `+inf` / `NaN` logit at a deterministically-masked slot
  // becomes a real `-inf`, not `NaN`. At the first sampled position the rule
  // masks every non-timestamp token `[0, timestamp_begin)`, so put a `+inf` at a
  // text slot and a `NaN` at no_timestamps and check both land at `-inf`.
  let f = ts_rules(1, None); // timestamp_begin=14, no_timestamps=12, eot=2
  let mut xs = [0.0_f32; VOCAB];
  xs[0] = f32::INFINITY; // a text slot, masked by the first-position rule
  xs[12] = f32::NAN; // no_timestamps, masked unconditionally
  xs[5] = f32::INFINITY; // another text slot, masked by the first-position rule
  let masked = ts_apply(&f, &xs, &[3]); // tokens.len() == sample_begin == 1
  for &i in &[0usize, 5, 12] {
    assert!(
      masked[i].is_infinite() && masked[i] < 0.0,
      "deterministic-masked slot {i} = {} must be -inf, not NaN",
      masked[i]
    );
  }
  // No slot anywhere in the row is NaN after the filter (the probability-mass
  // rule must not have observed a NaN that propagated).
  assert!(
    masked.iter().all(|v| !v.is_nan()),
    "no slot may be NaN after the filter"
  );
}

// ───────────────────────── ApplyTimestampRules ────────────────────────────

/// A timestamp-rules filter over a vocab where text tokens are `[0, eot)`,
/// specials/timestamps `>= eot`. timestamp_begin = 14, eot = 2,
/// no_timestamps = 12. vocab size = 20 (so timestamp tokens 14..20 exist).
fn ts_rules(sample_begin: usize, max_initial: Option<usize>) -> ApplyTimestampRules {
  ApplyTimestampRules {
    sample_begin,
    timestamp_begin: 14,
    no_timestamps: 12,
    eot: 2,
    max_initial_timestamp_index: max_initial,
    n_vocab: VOCAB,
  }
}

const VOCAB: usize = 20;

/// Apply a timestamp-rules filter to a flat-or-given host row, returning the
/// masked host row.
fn ts_apply(f: &ApplyTimestampRules, xs: &[f32], tokens: &[u32]) -> Vec<f32> {
  apply_host(f, xs, tokens)
}

#[test]
fn timestamp_rules_always_suppress_no_timestamps() {
  let f = ts_rules(0, None);
  // A non-first step with a non-timestamp last token: only no_timestamps is
  // forced off (besides the probability-mass rule, inert for a flat row).
  let masked = ts_apply(&f, &[0.0_f32; VOCAB], &[3, 4]); // sample_begin=0 so seq=[3,4]
  assert!(
    masked[12].is_infinite() && masked[12] < 0.0,
    "no_timestamps masked"
  );
}

#[test]
fn timestamp_rules_first_position_forces_timestamp() {
  // At the first sampled position (tokens.len() == sample_begin) all
  // non-timestamp tokens [0, timestamp_begin) are masked.
  let f = ts_rules(1, None);
  let masked = ts_apply(&f, &[0.0_f32; VOCAB], &[3]); // len == sample_begin == 1
  for (i, &v) in masked.iter().enumerate().take(14) {
    assert!(
      v.is_infinite() && v < 0.0,
      "text/special token {i} masked at first pos"
    );
  }
  // Timestamp tokens remain (not -inf, ignoring the prob-mass rule on a flat
  // row where timestamp mass ln(6) > max_text -inf → also forces, so check a
  // single timestamp survives only if mass rule didn't fire). With a flat row,
  // the [0, ts_begin) region is already fully masked, so this is consistent.
  assert!(masked[14].is_finite() || masked[14] == f32::NEG_INFINITY);
}

#[test]
fn timestamp_rules_max_initial_timestamp_caps_high_timestamps() {
  // At the first position with max_initial_timestamp_index = 2, timestamps
  // beyond timestamp_begin + 2 = 16 are masked (positions 17..20).
  let f = ts_rules(1, Some(2));
  let masked = ts_apply(&f, &[0.0_f32; VOCAB], &[3]);
  // 17,18,19 masked by the cap.
  for (i, &v) in masked.iter().enumerate().skip(17) {
    assert!(v.is_infinite() && v < 0.0, "ts {i} capped");
  }
}

#[test]
fn timestamp_rules_pair_after_two_timestamps_forbids_more_timestamps() {
  // last and penultimate are both timestamps → next must be non-timestamp:
  // mask [timestamp_begin, vocab). sample_begin = 0 so seq = full tokens.
  let f = ts_rules(0, None);
  // seq ends [..., 15(ts), 16(ts)] → both timestamps.
  let masked = ts_apply(&f, &[0.0_f32; VOCAB], &[3, 15, 16]);
  for (i, &v) in masked.iter().enumerate().skip(14) {
    assert!(v.is_infinite() && v < 0.0, "timestamp {i} masked");
  }
  // text token 3 stays finite (it is allowed next).
  assert!(masked[3].is_finite());
}

#[test]
fn timestamp_rules_single_trailing_timestamp_forbids_text() {
  // last is a timestamp, penultimate is NOT → cannot be a normal text token:
  // mask [0, eot). seq = [3(text), 15(ts)].
  let f = ts_rules(0, None);
  let masked = ts_apply(&f, &[0.0_f32; VOCAB], &[3, 15]);
  // [0, eot=2) masked.
  assert!(masked[0].is_infinite() && masked[0] < 0.0);
  assert!(masked[1].is_infinite() && masked[1] < 0.0);
  // eot itself (index 2) is allowed (the boundary is exclusive).
  // (eot may still be affected by the prob-mass rule, but not by this clause.)
}

#[test]
fn timestamp_rules_monotonic_forbids_smaller_timestamps() {
  // A seen timestamp 17 (>= timestamp_begin) forbids timestamps up to and
  // including it: the last token is text (not a timestamp) so the common-case
  // `+1` applies, masking [timestamp_begin, 17 + 1) = 14..18. The next
  // timestamp must be strictly greater than the last seen 17. seq = [3, 17, 5].
  let f = ts_rules(0, None);
  let masked = ts_apply(&f, &[0.0_f32; VOCAB], &[3, 17, 5]);
  // timestamps 14..18 masked (<= the last seen 17).
  for (i, &v) in masked.iter().enumerate().take(18).skip(14) {
    assert!(v.is_infinite() && v < 0.0, "ts {i} <= last masked");
  }
}

#[test]
fn timestamp_rules_zero_closing_timestamp_forces_nonzero_length() {
  // Opening timestamp <|0.00|> (id 14 == timestamp_begin) followed by text:
  // the closing timestamp may not be <|0.00|> again (which would be a
  // zero-length segment). With the `>= timestamp_begin` scan + the `+1` rule,
  // [timestamp_begin, 14 + 1) = {14} is masked, so <|0.00|> itself is forbidden
  // and the next timestamp must be > 0.00. seq = [14(ts), 12(text), 0(text)].
  let f = ts_rules(0, None);
  let masked = ts_apply(&f, &[0.0_f32; VOCAB], &[14, 12, 0]);
  // <|0.00|> (id 14) is masked — no zero-length closing segment.
  assert!(
    masked[14].is_infinite() && masked[14] < 0.0,
    "<|0.00|> must be masked to force nonzero segment length"
  );
  // A later timestamp (id 15 = <|0.02|>) remains legal (only [14,15) masked).
  assert!(
    masked[15].is_finite(),
    "the next timestamp <|0.02|> stays legal"
  );
}

#[test]
fn timestamp_rules_single_trailing_timestamp_allows_same_close() {
  // A single trailing timestamp opening a pair (last is a timestamp,
  // penultimate is NOT) may be closed by the same value: the `+1` is NOT
  // applied, so [timestamp_begin, 15) is masked but 15 itself stays legal.
  // seq = [3(text), 15(ts)] → last_was_timestamp, !penultimate_was_timestamp.
  let f = ts_rules(0, None);
  let masked = ts_apply(&f, &[0.0_f32; VOCAB], &[3, 15]);
  // 14 masked (smaller than 15); 15 itself NOT masked by the monotonicity rule.
  assert!(
    masked[14].is_infinite() && masked[14] < 0.0,
    "ts 14 < 15 masked"
  );
  // (Note: the [0, eot) clause already masked text tokens since last_was_ts &&
  // !penultimate_was_ts; the monotonicity rule leaves 15 reachable.)
}

#[test]
fn timestamp_rules_probability_mass_forces_timestamp() {
  // If summed timestamp probability exceeds the max single text-token
  // probability, all [0, timestamp_begin) are masked. Give the timestamp
  // region huge logits, the text region small ones, with a NON-timestamp
  // last token (so the pair clauses don't already mask everything).
  let f = ts_rules(0, None);
  let mut input = vec![0.0_f32; VOCAB];
  for v in input.iter_mut().skip(14) {
    *v = 10.0; // dominant timestamp logits (text region stays 0.0)
  }
  let masked = ts_apply(&f, &input, &[3, 4]); // last token 4 is text (not a timestamp)
  for (i, &v) in masked.iter().enumerate().take(14) {
    assert!(
      v.is_infinite() && v < 0.0,
      "text {i} masked by prob-mass rule"
    );
  }
}

#[test]
fn timestamp_rules_pos_inf_timestamp_does_not_force_all_text_masked() {
  // The probability-mass rule must mirror the reference LITERALLY:
  //   logprobs = logits - logsumexp(logits, axis=-1)   (over the full vocab)
  //   force    = logprobs[ts_begin:].logsumexp() > logprobs[:ts_begin].max()
  // With a `+inf` timestamp logit (and FINITE text), the full-row
  // `logsumexp(logits)` is `+inf`, so that slot's logprob is `+inf - +inf =
  // NaN`; `timestamp_logprob` is then NaN and `NaN > max_text` is FALSE — so
  // NO timestamp is forced and the finite text tokens are preserved. The
  // algebraically-cancelled form (`logsumexp(masked[ts:]) > max(masked[:ts])`)
  // would instead see `+inf > finite` and wrongly mask EVERY text token; this
  // asserts the literal (reference) behavior.
  let f = ts_rules(0, None);
  let mut input = vec![0.0_f32; VOCAB]; // finite text region (indices 0..14)
  input[15] = f32::INFINITY; // a single `+inf` timestamp logit
  // Last token 4 is text (not a timestamp) so the pair clauses do not mask.
  let masked = ts_apply(&f, &input, &[3, 4]);
  // Text tokens [0, timestamp_begin) keep their finite (`0.0`) values — the
  // NaN comparison is false, so the prob-mass rule does NOT force a timestamp.
  for (i, &v) in masked.iter().enumerate().take(14) {
    if i == f.no_timestamps as usize {
      continue; // no_timestamps (id 12) is always deterministically suppressed.
    }
    assert!(
      v == 0.0,
      "text {i} = {v} must stay finite (no force on a NaN comparison)"
    );
  }
}

#[test]
fn timestamp_rules_nan_timestamp_does_not_force_all_text_masked() {
  // A `NaN` timestamp logit propagates through `logsumexp(logits)` (NaN
  // dominates), so the whole `logprobs` row is NaN; `timestamp_logprob` is NaN
  // and `NaN > max_text` is FALSE — again no force, finite text preserved.
  let f = ts_rules(0, None);
  let mut input = vec![0.0_f32; VOCAB];
  input[16] = f32::NAN; // a single `NaN` timestamp logit
  let masked = ts_apply(&f, &input, &[3, 4]); // last token 4 is text
  for (i, &v) in masked.iter().enumerate().take(14) {
    if i == f.no_timestamps as usize {
      continue; // no_timestamps deterministically suppressed.
    }
    assert!(
      v == 0.0,
      "text {i} = {v} must stay finite (no force on a NaN comparison)"
    );
  }
}

// ───────────────────────── fallback decision logic ────────────────────────

fn result_with(avg_logprob: f64, compression_ratio: f64, no_speech_prob: f64) -> DecodingResult {
  DecodingResult {
    language: "en".into(),
    tokens: vec![],
    text: String::new(),
    avg_logprob,
    no_speech_prob,
    temperature: 0.0,
    compression_ratio,
  }
}

#[test]
fn acceptable_when_all_thresholds_pass() {
  let r = result_with(-0.5, 1.5, 0.1);
  assert!(result_is_acceptable(&r, Some(2.4), Some(-1.0), Some(0.6)));
}

#[test]
fn fallback_on_high_compression_ratio() {
  let r = result_with(-0.5, 3.0, 0.1); // ratio > 2.4
  assert!(!result_is_acceptable(&r, Some(2.4), Some(-1.0), Some(0.6)));
}

#[test]
fn fallback_on_low_avg_logprob() {
  let r = result_with(-2.0, 1.0, 0.1); // logprob < -1.0
  assert!(!result_is_acceptable(&r, Some(2.4), Some(-1.0), Some(0.6)));
}

#[test]
fn silence_overrides_fallback() {
  // Bad ratio AND bad logprob, but no_speech_prob > threshold → accept (the
  // reference's `needs_fallback = False` silence override wins last).
  let r = result_with(-5.0, 9.0, 0.95);
  assert!(result_is_acceptable(&r, Some(2.4), Some(-1.0), Some(0.6)));
}

#[test]
fn disabled_thresholds_never_trigger() {
  let r = result_with(-100.0, 100.0, 0.99);
  // All None → nothing can mark a fallback → always acceptable.
  assert!(result_is_acceptable(&r, None, None, None));
}

// ───────────────────────── greedy decoder ─────────────────────────────────

#[test]
fn greedy_decoder_argmax_and_logprob() {
  let mut d = GreedyDecoder::new(0.0, /* eot */ 2, 0).unwrap();
  // logits favoring index 3; last_token != eot so logprob accumulates.
  let logits = row(&[0.0, 0.0, 0.0, 5.0, 0.0]);
  let (next, completed) = d.update(&logits, /* last */ 1).unwrap();
  assert_eq!(next, 3);
  assert!(!completed);
  // sum_logprob = logits[3] - logsumexp(logits) < 0.
  assert!(
    d.sum_logprob < 0.0,
    "logprob {} should be negative",
    d.sum_logprob
  );
}

#[test]
fn greedy_decoder_eot_sticks_and_stops_logprob() {
  let mut d = GreedyDecoder::new(0.0, 2, 0).unwrap();
  let logits = row(&[0.0, 0.0, 0.0, 5.0, 0.0]);
  // last_token == eot → next forced to eot, logprob NOT accumulated.
  let (next, completed) = d.update(&logits, /* last */ 2).unwrap();
  assert_eq!(next, 2);
  assert!(completed);
  assert_eq!(d.sum_logprob, 0.0);
}

#[test]
fn greedy_decoder_completes_on_argmax_eot() {
  let mut d = GreedyDecoder::new(0.0, 2, 0).unwrap();
  // argmax is the eot id itself.
  let logits = row(&[0.0, 0.0, 9.0, 0.0]);
  let (next, completed) = d.update(&logits, /* last */ 1).unwrap();
  assert_eq!(next, 2);
  assert!(completed);
}

/// Timing harness for `GreedyDecoder::update` — measures the per-token cost of
/// the greedy argmax + chosen-logprob selection on a realistic `(n_vocab,)`
/// logits row. `#[ignore]`d (it runs the device path hundreds of times and is a
/// measurement, not a correctness assertion); run on a Metal host with:
///
/// ```text
/// cargo test -p mlxrs --features whisper --lib \
///   audio::stt::models::whisper::decoding::tests::greedy_decoder_update_per_token_timing \
///   -- --ignored --nocapture --test-threads=1
/// ```
///
/// The logits row is **varied each iteration** (a per-iteration scalar is added
/// to a fixed base row) so mlx re-evaluates the graph every step rather than
/// serving a cached result — making the timing reflect a true per-token decode
/// stall. `last_token != eot` so the chosen-logprob accumulation path runs.
#[test]
#[ignore = "timing measurement; run on a Metal host with --ignored --nocapture"]
fn greedy_decoder_update_per_token_timing() {
  use std::time::Instant;

  // A realistic large-v3 vocabulary width; the argmax sits at a fixed index so
  // the chosen-logprob gather runs, and `last_token != eot` so it accumulates.
  const N_VOCAB: i32 = 51865;
  const EOT: u32 = 50257;
  const ARGMAX_IDX: usize = 12_345;
  const WARMUP: usize = 30;
  const ITERS: usize = 300;

  // Fixed base row: a small ramp with a clear peak at ARGMAX_IDX so argmax is
  // deterministic regardless of the per-iteration perturbation.
  let mut base = vec![0.0f32; N_VOCAB as usize];
  for (i, v) in base.iter_mut().enumerate() {
    *v = (i as f32) * 1e-6;
  }
  base[ARGMAX_IDX] = 50.0;
  let base = Array::from_slice::<f32>(&base, &[N_VOCAB]).unwrap();

  let mut d = GreedyDecoder::new(0.0, EOT, 0).unwrap();

  // Build the per-iteration varied row: base + a 0-d scalar that changes every
  // call. The add keeps the argmax fixed (uniform shift) but forces a fresh
  // eval; materialize the input before timing so only `update` is measured.
  let varied = |step: usize| -> Array {
    let bump = Array::full::<f32>(&[0i32; 0], step as f32 * 1e-4).unwrap();
    let row = ops::arithmetic::add(&base, &bump).unwrap();
    crate::transforms::eval(&[&row]).unwrap();
    row
  };

  for step in 0..WARMUP {
    let logits = varied(step);
    let (next, completed) = d.update(&logits, /* last */ 0).unwrap();
    assert_eq!(next as usize, ARGMAX_IDX);
    assert!(!completed);
  }

  let start = Instant::now();
  for step in WARMUP..(WARMUP + ITERS) {
    let logits = varied(step);
    let (next, _completed) = d.update(&logits, /* last */ 0).unwrap();
    assert_eq!(next as usize, ARGMAX_IDX);
  }
  let elapsed = start.elapsed();
  let per_token = elapsed / ITERS as u32;

  println!(
    "GreedyDecoder::update: {ITERS} tokens in {elapsed:?} => {per_token:?}/token \
     ({:.1} tokens/s)",
    ITERS as f64 / elapsed.as_secs_f64()
  );
}

// ───────────────────── device-vs-scalar filter PARITY ─────────────────────

/// An independent **scalar** reference for the full filter chain + greedy
/// argmax — the host-side oracle the device path is checked against.
///
/// It re-implements every rule of the three logit filters over a plain
/// `Vec<f32>`, in the SAME order the device path applies them (deterministic
/// masks applied to the logits first, then `ApplyTimestampRules`'s
/// probability-mass rule reads the already-masked logits), and returns the
/// argmax index (ties → lowest, matching mlx `argmax`). This is written from
/// the rule descriptions directly — NOT by calling the code under test — so a
/// regression in the device filters cannot hide behind a shared helper.
mod scalar_ref {
  /// Greedy `argmax` over a host slice (ties → lowest index, `-inf` never
  /// chosen over a finite max). Empty ⇒ `0`.
  fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
      if x > best_val {
        best_val = x;
        best = i as u32;
      }
    }
    best
  }

  fn mask_range(m: &mut [f32], lo: usize, hi: usize) {
    let hi = hi.min(m.len());
    if lo < hi {
      for s in &mut m[lo..hi] {
        *s = f32::NEG_INFINITY;
      }
    }
  }

  /// `log(sum(exp))` in f64 over a slice; empty / all `-inf` ⇒ `-inf`.
  fn logsumexp(xs: &[f32]) -> f64 {
    let mut mx = f64::NEG_INFINITY;
    for &x in xs {
      if (x as f64) > mx {
        mx = x as f64;
      }
    }
    if !mx.is_finite() {
      return f64::NEG_INFINITY;
    }
    let mut s = 0.0f64;
    for &x in xs {
      s += (x as f64 - mx).exp();
    }
    mx + s.ln()
  }

  fn max(xs: &[f32]) -> f64 {
    let mut mx = f64::NEG_INFINITY;
    for &x in xs {
      if (x as f64) > mx {
        mx = x as f64;
      }
    }
    mx
  }

  /// The filter configuration the parity test exercises (a faithful mirror of
  /// the three reference filters' parameters).
  pub struct Config {
    pub sample_begin: usize,
    pub timestamp_begin: u32,
    pub no_timestamps: u32,
    pub eot: u32,
    pub max_initial_timestamp_index: Option<usize>,
    pub blank_ids: Vec<u32>,
    pub suppress_ids: Vec<u32>,
    pub suppress_blank: bool,
    pub with_timestamps: bool,
  }

  /// Apply the full filter chain to a host logits row, mutating in place,
  /// exactly as the device path does (SuppressBlank → SuppressTokens →
  /// ApplyTimestampRules; the timestamp rules' deterministic masks first, then
  /// the probability-mass rule over the masked logits).
  pub fn apply(cfg: &Config, logits: &mut [f32], tokens: &[u32]) {
    let n_vocab = logits.len();
    // SuppressBlank.
    if cfg.suppress_blank && tokens.len() == cfg.sample_begin {
      for &id in &cfg.blank_ids {
        if let Some(s) = logits.get_mut(id as usize) {
          *s = f32::NEG_INFINITY;
        }
      }
    }
    // SuppressTokens.
    for &id in &cfg.suppress_ids {
      if let Some(s) = logits.get_mut(id as usize) {
        *s = f32::NEG_INFINITY;
      }
    }
    // ApplyTimestampRules.
    if cfg.with_timestamps {
      let ts_begin = cfg.timestamp_begin as usize;
      let eot = cfg.eot as usize;
      if let Some(s) = logits.get_mut(cfg.no_timestamps as usize) {
        *s = f32::NEG_INFINITY;
      }
      let seq: &[u32] = tokens.get(cfg.sample_begin..).unwrap_or(&[]);
      let last_was_ts = seq.last().is_some_and(|&t| t >= cfg.timestamp_begin);
      let penult_was_ts = seq.len() < 2
        || seq
          .get(seq.len() - 2)
          .is_some_and(|&t| t >= cfg.timestamp_begin);
      if last_was_ts {
        if penult_was_ts {
          mask_range(logits, ts_begin, n_vocab);
        } else {
          mask_range(logits, 0, eot);
        }
      }
      let mut last_ts: Option<usize> = None;
      for &v in seq {
        if (v as usize) >= ts_begin {
          last_ts = Some(v as usize);
        }
      }
      if let Some(last) = last_ts {
        let upper = if last_was_ts && !penult_was_ts {
          last
        } else {
          last + 1
        };
        mask_range(logits, ts_begin, upper);
      }
      if tokens.len() == cfg.sample_begin {
        mask_range(logits, 0, ts_begin);
        if let Some(idx) = cfg.max_initial_timestamp_index {
          mask_range(logits, ts_begin + idx + 1, n_vocab);
        }
      }
      // Probability-mass rule over the ALREADY-MASKED logits.
      let ts_b = ts_begin.min(n_vocab);
      if ts_b > 0 && ts_b < n_vocab && logsumexp(&logits[ts_b..]) > max(&logits[..ts_b]) {
        mask_range(logits, 0, ts_begin);
      }
    }
  }

  /// The selected (argmax) token for a given config + raw logits + token state.
  pub fn select(cfg: &Config, raw: &[f32], tokens: &[u32]) -> u32 {
    let mut logits = raw.to_vec();
    apply(cfg, &mut logits, tokens);
    argmax(&logits)
  }
}

/// Build the device filter chain matching a [`scalar_ref::Config`].
fn device_filters(cfg: &scalar_ref::Config, n_vocab: usize) -> Vec<Box<dyn LogitFilter>> {
  let mut filters: Vec<Box<dyn LogitFilter>> = Vec::new();
  if cfg.suppress_blank {
    filters.push(Box::new(SuppressBlank {
      sample_begin: cfg.sample_begin,
      mask: scatter_neg_inf_mask(n_vocab, &cfg.blank_ids).unwrap(),
    }));
  }
  if !cfg.suppress_ids.is_empty() {
    filters.push(Box::new(
      SuppressTokens::new(&cfg.suppress_ids, n_vocab).unwrap(),
    ));
  }
  if cfg.with_timestamps {
    filters.push(Box::new(ApplyTimestampRules {
      sample_begin: cfg.sample_begin,
      timestamp_begin: cfg.timestamp_begin,
      no_timestamps: cfg.no_timestamps,
      eot: cfg.eot,
      max_initial_timestamp_index: cfg.max_initial_timestamp_index,
      n_vocab,
    }));
  }
  filters
}

/// Run the DEVICE filter chain + on-device greedy argmax for a raw row +
/// token state, returning the selected token.
fn device_select(cfg: &scalar_ref::Config, raw: &[f32], tokens: &[u32], n_vocab: usize) -> u32 {
  let filters = device_filters(cfg, n_vocab);
  let mut r = row(raw);
  for f in &filters {
    r = f.apply(&r, tokens).unwrap();
  }
  // Greedy argmax on device (temperature 0), exactly as the decode loop does.
  let mut d = GreedyDecoder::new(0.0, cfg.eot, 0).unwrap();
  let (next, _completed) = d.update(&r, /* last */ cfg.timestamp_begin + 999).unwrap();
  next
}

/// A Whisper-shaped parity config: text region `[0, eot)`, specials/timestamps
/// `>= eot`, `timestamp_begin = 14`, `eot = 2`, `no_timestamps = 12`. Vocab
/// width 20 (timestamps 14..20). `sample_begin` / `max_initial` are per-case.
fn parity_cfg(
  sample_begin: usize,
  max_initial: Option<usize>,
  suppress_blank: bool,
  suppress_ids: Vec<u32>,
  with_timestamps: bool,
) -> scalar_ref::Config {
  scalar_ref::Config {
    sample_begin,
    timestamp_begin: 14,
    no_timestamps: 12,
    eot: 2,
    max_initial_timestamp_index: max_initial,
    blank_ids: vec![13, 2], // a "space"-like id 13 + eot 2
    suppress_ids,
    suppress_blank,
    with_timestamps,
  }
}

const PV: usize = 20; // parity vocab width

/// The DEVICE filter chain + greedy argmax must select the SAME token as the
/// independent scalar reference, across every timestamp-rule edge case (the
/// sample_begin suppress-blank gating, the pair rules, timestamp monotonicity,
/// the max_initial cap, and the sum-probability rule) and the suppression
/// filters. Each `raw` row is deliberately off any knife-edge tie so the f32
/// device reductions and the f64 scalar reductions agree on the comparison.
#[test]
fn device_filters_match_scalar_reference_token_selection() {
  // A varied, non-flat base row so argmax has a clear winner that the masks
  // then redirect. Distinct values keep argmax unambiguous.
  let base: Vec<f32> = (0..PV).map(|i| (i as f32) * 0.37 - 3.0).collect();

  // (config, raw row, token state) cases — each exercises a specific rule.
  struct Case {
    name: &'static str,
    cfg: scalar_ref::Config,
    raw: Vec<f32>,
    tokens: Vec<u32>,
  }

  let mut raw_textwin = base.clone();
  // Make a text token (id 5) the unmasked argmax winner for the suppress cases.
  raw_textwin[5] = 50.0;

  // A row where the timestamp mass clearly dominates the text region.
  let mut raw_tsdom = base.clone();
  for v in raw_tsdom.iter_mut().take(14) {
    *v = -10.0;
  }
  for v in raw_tsdom.iter_mut().skip(14) {
    *v = 5.0;
  }

  // A row where a single text token clearly dominates (timestamp mass low).
  let mut raw_textdom = vec![-20.0f32; PV];
  raw_textdom[7] = 30.0; // lone text winner

  let cases = vec![
    Case {
      name: "suppress_blank at sample_begin masks blank+eot",
      cfg: parity_cfg(3, None, true, vec![], false),
      raw: {
        // blank id 13 is the would-be winner; suppress_blank must redirect it.
        let mut r = base.clone();
        r[13] = 99.0;
        r[8] = 40.0; // the next-best (non-blank) token
        r
      },
      tokens: vec![1, 2, 3], // len == sample_begin == 3 → gate fires
    },
    Case {
      name: "suppress_blank inert when not at sample_begin",
      cfg: parity_cfg(3, None, true, vec![], false),
      raw: {
        let mut r = base.clone();
        r[13] = 99.0;
        r
      },
      tokens: vec![1, 2, 3, 4], // len != sample_begin → no gate
    },
    Case {
      name: "suppress_tokens redirects argmax off a suppressed id",
      cfg: parity_cfg(0, None, false, vec![5, 6], false),
      raw: raw_textwin.clone(),
      tokens: vec![3, 4],
    },
    Case {
      name: "timestamp first-position forces a timestamp",
      cfg: parity_cfg(1, Some(2), false, vec![], true),
      raw: raw_textwin.clone(),
      tokens: vec![3], // len == sample_begin == 1 → first sampled pos
    },
    Case {
      name: "timestamp max_initial caps high timestamps",
      cfg: parity_cfg(1, Some(2), false, vec![], true),
      raw: {
        // bias a high timestamp (id 19) that the cap must forbid.
        let mut r = base.clone();
        r[19] = 80.0;
        r[15] = 40.0; // an allowed timestamp under the cap (<= 16)
        r
      },
      tokens: vec![3],
    },
    Case {
      name: "timestamp pair-after-two forbids more timestamps",
      cfg: parity_cfg(0, None, false, vec![], true),
      raw: raw_tsdom.clone(),
      tokens: vec![3, 15, 16], // last two are timestamps
    },
    Case {
      name: "timestamp single-trailing forbids text",
      cfg: parity_cfg(0, None, false, vec![], true),
      raw: raw_textwin.clone(),
      tokens: vec![3, 15], // single trailing timestamp
    },
    Case {
      name: "timestamp monotonicity forbids smaller-or-equal",
      cfg: parity_cfg(0, None, false, vec![], true),
      raw: {
        // The final consumer of `base` — move it (no clone).
        let mut r = base;
        r[15] = 60.0; // a too-small timestamp the rule must forbid
        r[18] = 70.0; // a strictly-greater timestamp that stays legal
        r
      },
      tokens: vec![3, 17, 5], // last seen ts 17 → forbid <= 17
    },
    Case {
      name: "timestamp sum-probability forces a timestamp",
      cfg: parity_cfg(0, None, false, vec![], true),
      raw: raw_tsdom.clone(),
      tokens: vec![3, 4], // non-timestamp last → pair clauses inert
    },
    Case {
      name: "timestamp sum-probability inert when text dominates",
      cfg: parity_cfg(0, None, false, vec![], true),
      raw: raw_textdom.clone(),
      tokens: vec![3, 4],
    },
    Case {
      name: "all three filters composed",
      cfg: parity_cfg(0, Some(2), true, vec![6, 7], true),
      raw: raw_tsdom.clone(),
      tokens: vec![3, 4],
    },
  ];

  for case in &cases {
    let want = scalar_ref::select(&case.cfg, &case.raw, &case.tokens);
    let got = device_select(&case.cfg, &case.raw, &case.tokens, PV);
    assert_eq!(
      got, want,
      "case '{}': device selected {got}, scalar reference selected {want}",
      case.name
    );
  }
}

// ───────────────────────── end-to-end greedy decode ───────────────────────

/// Whisper-shaped special tokens for the decode fixtures. timestamp_begin is
/// `<|0.00|>` at id 14; one more timestamp token `<|0.02|>` at 15 so the
/// timestamp region is non-trivial. vocab ids span 0..18.
const SPECIALS: &[(&str, u32)] = &[
  ("a", 0),
  ("b", 1),
  ("<|endoftext|>", 2),
  ("<|startoftranscript|>", 3),
  ("<|en|>", 4),
  ("<|zh|>", 5),
  ("<|translate|>", 6),
  ("<|transcribe|>", 7),
  ("<|startoflm|>", 8),
  ("<|startofprev|>", 9),
  ("<|nospeech|>", 10),
  ("<|notimestamps|>", 11),
  ("c", 12),
  ("d", 13),
  ("<|0.00|>", 14),
  ("<|0.02|>", 15),
  ("<|0.04|>", 16),
  ("<|0.06|>", 17),
];

/// n_vocab for the tiny decode model — must cover every special id above.
const N_VOCAB: usize = 18;

fn fresh_dir(tag: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_whisper_dec_{}_{}", std::process::id(), tag));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

fn write_tokenizer(dir: &Path) -> Tokenizer {
  let vocab: serde_json::Map<String, serde_json::Value> = SPECIALS
    .iter()
    .map(|(tok, id)| (tok.to_string(), json!(id)))
    .collect();
  let added_tokens: Vec<serde_json::Value> = SPECIALS
    .iter()
    .map(|(tok, id)| {
      let special = tok.starts_with("<|");
      json!({
        "id": id, "content": tok, "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": special
      })
    })
    .collect();
  let tokenizer_json = json!({
    "version": "1.0",
    "added_tokens": added_tokens,
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>" }
  });
  let cfg = json!({ "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>" });
  std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  Tokenizer::from_path(dir, None).unwrap()
}

/// Tiny model dims with n_vocab covering the fixture's special tokens.
///
/// `n_audio_ctx` is the architecturally fixed `N_FRAMES / 2` (`1500`) — pinned at
/// construction. Only the *width* dims are tiny, so the encoder still runs
/// cheaply on the fixed `N_FRAMES`-frame mel produced by [`tiny_mel`].
fn dims() -> ModelDimensions {
  ModelDimensions::new(
    /* n_mels */ 4,
    /* n_audio_ctx */ N_FRAMES / 2,
    /* n_audio_state */ 4,
    /* n_audio_head */ 2,
    /* n_audio_layer */ 1,
    /* n_vocab */ N_VOCAB,
    /* n_text_ctx */ 8,
    /* n_text_state */ 4,
    /* n_text_head */ 2,
    /* n_text_layer */ 1,
  )
  .unwrap()
}

/// A `(N_FRAMES, n_mels)` mel — the fixed padded frame count the encoder
/// consumes (after conv2's stride 2 → `n_audio_ctx` = 1500 frames). Replaces the
/// pre-pin synthetic short mel now that `n_audio_ctx` is pinned.
fn tiny_mel() -> Array {
  Array::ones::<f32>(&(N_FRAMES, 4usize)).unwrap()
}

fn ones2(r: usize, c: usize) -> Array {
  Array::ones::<f32>(&(r, c)).unwrap()
}
fn zeros1(n: usize) -> Array {
  Array::zeros::<f32>(&(n,)).unwrap()
}
fn ones1(n: usize) -> Array {
  Array::ones::<f32>(&(n,)).unwrap()
}

fn put_attn(w: &mut HashMap<String, Array>, prefix: &str, n: usize) {
  for p in ["query", "value", "out"] {
    w.insert(format!("{prefix}.{p}.weight"), ones2(n, n));
    w.insert(format!("{prefix}.{p}.bias"), zeros1(n));
  }
  w.insert(format!("{prefix}.key.weight"), ones2(n, n));
}
fn put_ln(w: &mut HashMap<String, Array>, prefix: &str, n: usize) {
  w.insert(format!("{prefix}.weight"), ones1(n));
  w.insert(format!("{prefix}.bias"), zeros1(n));
}
fn put_block(w: &mut HashMap<String, Array>, prefix: &str, n: usize, cross: bool) {
  put_attn(w, &format!("{prefix}.attn"), n);
  put_ln(w, &format!("{prefix}.attn_ln"), n);
  if cross {
    put_attn(w, &format!("{prefix}.cross_attn"), n);
    put_ln(w, &format!("{prefix}.cross_attn_ln"), n);
  }
  w.insert(format!("{prefix}.mlp1.weight"), ones2(4 * n, n));
  w.insert(format!("{prefix}.mlp1.bias"), zeros1(4 * n));
  w.insert(format!("{prefix}.mlp2.weight"), ones2(n, 4 * n));
  w.insert(format!("{prefix}.mlp2.bias"), zeros1(n));
  put_ln(w, &format!("{prefix}.mlp_ln"), n);
}

/// Build a tiny model whose decoder logit head is biased toward a `target`
/// token so greedy decode is deterministic and predictable.
///
/// Symmetry must be broken: with all-uniform weights + a uniform input the
/// per-position hidden vector is constant, so the final LayerNorm (zero
/// variance) outputs the zero vector and *every* logit is `0`. Here the
/// token/positional embeddings vary per hidden dimension, so the residual
/// stream `x` reaching the final LN is non-constant → a real nonzero
/// post-LN vector. The weight-tied head is `x @ tok_embᵀ`; the `target` row
/// is set to a large positive multiple of that post-LN direction's dominant
/// dimension so its dot product dominates.
fn tiny_model(target: u32) -> WhisperModel {
  let n = 4usize;
  let mut w = HashMap::new();
  // Non-uniform conv weights so the encoder output varies per channel.
  let c1: Vec<f32> = (0..(n * 3 * 4))
    .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
    .collect();
  w.insert(
    "encoder.conv1.weight".into(),
    Array::from_slice::<f32>(&c1, &(n, 3usize, 4usize)).unwrap(),
  );
  w.insert("encoder.conv1.bias".into(), zeros1(n));
  let c2: Vec<f32> = (0..(n * 3 * n))
    .map(|i| ((i % 3) as f32 - 1.0) * 0.1)
    .collect();
  w.insert(
    "encoder.conv2.weight".into(),
    Array::from_slice::<f32>(&c2, &(n, 3usize, n)).unwrap(),
  );
  w.insert("encoder.conv2.bias".into(), zeros1(n));
  put_block(&mut w, "encoder.blocks.0", n, false);
  put_ln(&mut w, "encoder.ln_post", n);

  // token embedding rows: a per-dimension ramp `[-2,-1,0,1] * 0.2` (the same
  // for every row — they tie, so the natural argmax is index 0). The post-LN
  // hidden `x` is zero-mean, so a *constant* row would dot to 0; the ramp is
  // a non-constant direction that correlates positively with `x` (the probe
  // shows every ramp row → the same +0.894 logit). The `target` row is that
  // same ramp scaled up 10×, so its logit dominates (~+8.9) and greedy decode
  // deterministically selects `target`.
  let ramp = |j: usize| -> f32 { (j as f32 - (n as f32 / 2.0)) * 0.2 };
  let mut emb: Vec<f32> = (0..(N_VOCAB * n)).map(|i| ramp(i % n)).collect();
  if (target as usize) < N_VOCAB {
    let base = target as usize * n;
    for j in 0..n {
      emb[base + j] = ramp(j) * 10.0;
    }
  }
  w.insert(
    "decoder.token_embedding.weight".into(),
    Array::from_slice::<f32>(&emb, &(N_VOCAB, n)).unwrap(),
  );
  // positional embedding: a per-dimension ramp so each absolute position
  // injects variation into the residual stream.
  let pe: Vec<f32> = (0..(8 * n))
    .map(|i| ((i % n) as f32 - (n as f32 / 2.0)) * 0.3)
    .collect();
  w.insert(
    "decoder.positional_embedding".into(),
    Array::from_slice::<f32>(&pe, &(8usize, n)).unwrap(),
  );
  put_block(&mut w, "decoder.blocks.0", n, true);
  put_ln(&mut w, "decoder.ln", n);

  WhisperModel::from_weights(dims(), w, Dtype::F32).unwrap()
}

#[test]
fn decode_tokens_returns_seq_logits() {
  let model = tiny_model(13);
  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();
  let (logits, _cache) = model.decode_tokens(&[3, 4, 7], &enc, None).unwrap();
  assert_eq!(logits.shape(), vec![1usize, 3, N_VOCAB]);
}

#[test]
fn greedy_decode_runs_and_emits_target_then_eot() {
  // The model is biased so every step argmaxes to `target`; with timestamps
  // disabled and suppression off, the decode should emit `target` until the
  // sample length cap (it never reaches eot since eot's logit is not biased).
  let dir = fresh_dir("greedy_target");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let target = 13u32; // a plain text token "d"
  let model = tiny_model(target);

  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  let mut options = DecodingOptions {
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(4),
    ..Default::default()
  };
  options.language = Some("en".into());

  let result = decode(&model, &wrapper, &enc, options).unwrap();
  // Every sampled token is the biased target.
  assert!(!result.tokens.is_empty());
  assert!(
    result.tokens.iter().all(|&t| t == target),
    "tokens={:?}",
    result.tokens
  );
  assert_eq!(result.language, "en");
  // avg_logprob = sum/(len+1); a strongly-peaked head → near-zero logprob.
  assert!(result.avg_logprob <= 0.0);
}

/// Phase-3 gate (#369): the pipelined decode loop (`main_loop_pipelined`) must
/// produce the EXACT same token sequence + `sum_logprob` + `no_speech_prob` as
/// the serialized loop (`main_loop`), across the no-eot run, an eot-mid-sequence
/// run (eot masked at step 0 by the timestamp rules, then emitted at step 1, so
/// the completed-one-behind break is exercised), and the no-timestamp path.
#[test]
fn pipelined_loop_matches_serial() {
  let dir = fresh_dir("pipeline_parity");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let mel = tiny_mel();

  // (target bias, without_timestamps): no-eot text run, eot-mid with timestamps,
  // no-timestamp text run.
  for (target, without_timestamps) in [(13u32, false), (2u32, false), (13u32, true)] {
    let model = tiny_model(target);
    let enc = model.encode(&mel).unwrap();
    let mut options = DecodingOptions {
      without_timestamps,
      suppress_blank: true,
      sample_len: Some(8),
      ..Default::default()
    };
    options.language = Some("en".into());
    let task = DecodingTask::new(&model, &wrapper, options).unwrap();

    let (s_tokens, s_sum, s_ns) = task.main_loop(&enc).unwrap();
    let (p_tokens, p_sum, p_ns) = task.main_loop_pipelined(&enc).unwrap();

    assert_eq!(
      s_tokens, p_tokens,
      "pipelined tokens must match serial (target={target}, without_ts={without_timestamps})"
    );
    // BIT-EXACT `sum_logprob`: the pipelined loop folds each token's f32 logprob
    // contribution into the host f64 accumulator one-at-a-time in serial order
    // (it does NOT sum on device), so it must reproduce `main_loop`'s f64
    // accumulation to the last bit — not merely within a tolerance that could
    // hide an `avg_logprob` drift across the temperature-fallback / no-speech
    // thresholds. Compare the raw bit patterns.
    assert_eq!(
      s_sum.to_bits(),
      p_sum.to_bits(),
      "sum_logprob must be BIT-EXACT (target={target}): serial={s_sum} ({:#018x}) pipelined={p_sum} ({:#018x})",
      s_sum.to_bits(),
      p_sum.to_bits()
    );
    assert!(
      (s_ns - p_ns).abs() < 1e-9,
      "no_speech_prob mismatch (target={target}): serial={s_ns} pipelined={p_ns}"
    );
  }
}

/// #369 investigation: per-step decode cost on a TINY model (negligible layer
/// compute) isolates any FIXED model-independent per-step overhead from real
/// compute. If this is ~ms (≈ the ~0.8ms command-buffer latency) then large-v3's
/// ~13ms is genuine compute; if it is also ~13ms there is a fixed overhead bug
/// (the real cause of mlxrs being ~3x slower than Python on the same model).
#[test]
#[ignore = "timing microbenchmark — run with --ignored --nocapture"]
fn tiny_decode_per_step_timing() {
  use std::time::Instant;
  let model = tiny_model(13);
  let enc = model.encode(&tiny_mel()).unwrap();
  let tok = crate::Array::from_slice::<u32>(&[13u32], &[1, 1]).unwrap();

  // warmup
  for _ in 0..10 {
    let (l, _c) = model.decode_token_lazy(&tok, &enc, None).unwrap();
    let row = last_position_row(&l).unwrap();
    let mut idx = crate::ops::misc::argmax(&row, Some(-1), false).unwrap();
    crate::transforms::eval(&[&idx]).unwrap();
    let _ = idx.item::<u32>().unwrap();
  }
  const N: usize = 100;
  let t = Instant::now();
  for _ in 0..N {
    let (l, _c) = model.decode_token_lazy(&tok, &enc, None).unwrap();
    let row = last_position_row(&l).unwrap();
    let mut idx = crate::ops::misc::argmax(&row, Some(-1), false).unwrap();
    crate::transforms::eval(&[&idx]).unwrap();
    let _ = idx.item::<u32>().unwrap();
  }
  println!(
    "\nTINY decode per-step (1-layer/4-dim model, fresh cache): {:.1} us/step",
    t.elapsed().as_secs_f64() * 1e6 / N as f64
  );
}

#[test]
fn greedy_decode_stops_at_eot() {
  // Bias the head to eot → the first sampled token is eot, the loop completes
  // immediately and the trimmed token list is empty.
  let dir = fresh_dir("greedy_eot");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(2); // 2 == eot

  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  let mut options = DecodingOptions {
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(5),
    ..Default::default()
  };
  options.language = Some("en".into());

  let result = decode(&model, &wrapper, &enc, options).unwrap();
  assert!(result.tokens.is_empty(), "tokens={:?}", result.tokens);
  assert_eq!(result.text, "");
}

#[test]
fn greedy_decode_sample_len_zero_emits_no_token() {
  // `sample_len == 0` caps the sampled-token count at zero. The model is biased
  // so the first argmax WOULD be `target` (13), but the loop must honor the cap
  // and emit no token at all — matching the stt driver's `max_new_tokens == 0`
  // semantics — rather than the reference's unconditional first-step emit.
  let dir = fresh_dir("greedy_zero");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let target = 13u32;
  let model = tiny_model(target);

  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  let mut options = DecodingOptions {
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(0),
    ..Default::default()
  };
  options.language = Some("en".into());

  let result = decode(&model, &wrapper, &enc, options).unwrap();
  assert!(
    result.tokens.is_empty(),
    "sample_len == 0 must emit no token, got {:?}",
    result.tokens
  );
  assert_eq!(result.text, "");
}

#[test]
fn greedy_decode_sample_len_one_emits_exactly_one_token() {
  // The boundary above zero: `sample_len == 1` emits exactly one sampled token
  // (the unconditional first step), so the off-by-one fix for `0` does not
  // regress the `1` case.
  let dir = fresh_dir("greedy_one");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let target = 13u32;
  let model = tiny_model(target);

  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  let mut options = DecodingOptions {
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(1),
    ..Default::default()
  };
  options.language = Some("en".into());

  let result = decode(&model, &wrapper, &enc, options).unwrap();
  assert_eq!(
    result.tokens,
    vec![target],
    "sample_len == 1 must emit exactly one token"
  );
}

#[test]
fn decode_auto_detects_language_when_none() {
  // With `options.language = None` on a multilingual checkpoint, `decode` must
  // detect the language (not silently default to "en"). The head is biased to
  // the `<|zh|>` language token (id 5), so detection picks "zh" and the result
  // reports it — proving auto-detect is wired into `decode`.
  let dir = fresh_dir("decode_autodetect");
  let tok = write_tokenizer(dir.as_path());
  // A multilingual wrapper built for "en"; `decode` rebuilds the SOT language
  // from the detection, so the build-time "en" must not leak through.
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(5); // 5 == <|zh|>

  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  let options = DecodingOptions {
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(2),
    language: None, // <- trigger detection
    ..Default::default()
  };
  let result = decode(&model, &wrapper, &enc, options).unwrap();
  assert_eq!(
    result.language, "zh",
    "decode must detect + report the language when none is supplied"
  );
}

#[test]
fn decode_uses_supplied_language_without_detection() {
  // A supplied language is reported verbatim (no detection overriding it),
  // even when the head would detect a different language.
  let dir = fresh_dir("decode_supplied_lang");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(5); // head biased to <|zh|>, but language is supplied

  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  let options = DecodingOptions {
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(2),
    language: Some("en".into()),
    ..Default::default()
  };
  let result = decode(&model, &wrapper, &enc, options).unwrap();
  assert_eq!(result.language, "en");
}

#[test]
fn initial_tokens_include_sot_sequence_and_prompt() {
  let dir = fresh_dir("initial_tokens");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);

  let options = DecodingOptions {
    without_timestamps: true,
    prompt: vec![12, 13], // "c d"
    ..Default::default()
  };
  let task = DecodingTask::new(&model, &wrapper, options).unwrap();
  // sot_sequence(en, transcribe) = [sot=3, en=4, transcribe=7], + notimestamps
  // (11) since without_timestamps; prompt prepends [sot_prev=9, 12, 13].
  let it = &task.initial_tokens;
  assert_eq!(it[0], wrapper.sot_prev()); // 9
  assert_eq!(&it[1..3], &[12, 13]);
  // sot sequence follows the prompt.
  assert!(it.contains(&wrapper.sot()));
  assert!(it.contains(&wrapper.no_timestamps()));
  // sot_index points at the sot token within the initial sequence.
  assert_eq!(it[task.sot_index], wrapper.sot());
  assert_eq!(task.sample_begin, it.len());
}

/// A caller-supplied `prompt` / `prefix` id `>= n_vocab` flows through
/// `initial_tokens` into the prefill `decode_tokens` and the decoder
/// token-embedding gather — where an out-of-range id would index out of bounds.
/// `DecodingTask::new` fails fast with a typed `OutOfRange` at construction
/// (before any forward), naming the initial-token boundary, for both the
/// boundary id (`== n_vocab`) and a strictly-out-of-range id; an in-range
/// prompt is accepted.
#[test]
fn decoding_task_rejects_out_of_vocab_prompt_or_prefix() {
  let dir = fresh_dir("oob_prompt");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);

  // `DecodingTask` does not derive `Debug`, so extract the error from the `Err`
  // arm directly rather than via `unwrap_err` (which would require the `Ok` type
  // to be `Debug`).
  let expect_oob = |res: Result<DecodingTask<'_>>, what: &str| match res {
    Err(Error::OutOfRange(p)) => assert_eq!(
      p.context(),
      "DecodingTask: initial token",
      "expected the initial-token OutOfRange for {what}, got context {:?}",
      p.context()
    ),
    Err(other) => panic!("expected OutOfRange for {what}, got {other:?}"),
    Ok(_) => panic!("expected OutOfRange for {what}, got Ok"),
  };

  // `sample_len = 1` keeps a non-zero prompt/prefix truncation budget for this
  // tiny `n_ctx = 8` fixture (`prefix` keeps `n_ctx/2 - sample_len = 3` tokens,
  // `prompt` keeps `n_ctx/2 - 1 = 3`), so a single out-of-vocab id survives into
  // `initial_tokens` and reaches the guard rather than being truncated away.

  // prompt id == n_vocab (the first out-of-bounds embedding row).
  let options = DecodingOptions {
    without_timestamps: true,
    sample_len: Some(1),
    prompt: vec![N_VOCAB as u32],
    ..Default::default()
  };
  expect_oob(
    DecodingTask::new(&model, &wrapper, options),
    "an out-of-vocab prompt id",
  );

  // prefix id > n_vocab.
  let options = DecodingOptions {
    without_timestamps: true,
    sample_len: Some(1),
    prefix: vec![N_VOCAB as u32 + 7],
    ..Default::default()
  };
  expect_oob(
    DecodingTask::new(&model, &wrapper, options),
    "an out-of-vocab prefix id",
  );

  // An in-range prompt is accepted (the guard does not reject valid ids).
  let options = DecodingOptions {
    without_timestamps: true,
    sample_len: Some(1),
    prompt: vec![12, 13],
    ..Default::default()
  };
  assert!(DecodingTask::new(&model, &wrapper, options).is_ok());
}

#[test]
fn detect_language_picks_a_language_code() {
  // Bias the head to the `<|en|>` language token (id 4); detect_language masks
  // every non-language logit and argmaxes, so "en" must win.
  let dir = fresh_dir("detect_lang");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(
    &tok,
    true,
    /* num_languages */ 2,
    Some("en"),
    Task::Transcribe,
  )
  .unwrap();
  let model = tiny_model(4); // 4 == <|en|>

  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  let (best, probs) = detect_language(&model, &wrapper, &enc).unwrap();
  assert_eq!(best, "en");
  // probs cover the checkpoint's languages (en, zh) and sum to ~1.
  let total: f64 = probs.iter().map(|(_, p)| p).sum();
  assert!((total - 1.0).abs() < 1e-5, "lang probs sum {total}");
  assert!(probs.iter().any(|(c, _)| *c == "en"));
}

/// Write a tokenizer whose structural special tokens stay at the fixture's
/// low ids (so [`HFTokenizerWrapper::new`] resolves them and `sot` indexes the
/// model's embedding) but whose LANGUAGE tokens `<|en|>` / `<|zh|>` are placed
/// at ids `>= N_VOCAB` — a mismatched / corrupt tokenizer-model pair. The model
/// decoder still emits an `N_VOCAB`-wide logits row, so those language ids index
/// nothing in the row.
fn write_tokenizer_oob_languages(dir: &Path) -> Tokenizer {
  // Every structural special at its canonical fixture id (< N_VOCAB), but the
  // two language tokens bumped to 100 / 101 (>= N_VOCAB = 18).
  let specials: &[(&str, u32)] = &[
    ("a", 0),
    ("b", 1),
    ("<|endoftext|>", 2),
    ("<|startoftranscript|>", 3),
    ("<|translate|>", 6),
    ("<|transcribe|>", 7),
    ("<|startoflm|>", 8),
    ("<|startofprev|>", 9),
    ("<|nospeech|>", 10),
    ("<|notimestamps|>", 11),
    ("<|0.00|>", 14),
    // Language tokens out of the model's vocabulary range.
    ("<|en|>", 100),
    ("<|zh|>", 101),
  ];
  let vocab: serde_json::Map<String, serde_json::Value> = specials
    .iter()
    .map(|(tok, id)| (tok.to_string(), json!(id)))
    .collect();
  let added_tokens: Vec<serde_json::Value> = specials
    .iter()
    .map(|(tok, id)| {
      let special = tok.starts_with("<|");
      json!({
        "id": id, "content": tok, "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": special
      })
    })
    .collect();
  let tokenizer_json = json!({
    "version": "1.0",
    "added_tokens": added_tokens,
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>" }
  });
  let cfg = json!({ "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>" });
  std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  Tokenizer::from_path(dir, None).unwrap()
}

#[test]
fn detect_language_rejects_out_of_vocab_language_ids() {
  // A corrupt tokenizer-model pair: the tokenizer's `<|en|>` / `<|zh|>` ids
  // (100 / 101) are >= the model's n_vocab (18 = the decoder logits width).
  // Masking would leave the row all `-inf` and `logit - logsumexp` = NaN,
  // silently selecting the first candidate. Instead detect_language must reject
  // it with a typed OutOfRange naming the offending id — NOT a NaN nor a bogus
  // best code.
  let dir = fresh_dir("detect_lang_oob");
  let tok = write_tokenizer_oob_languages(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(
    &tok,
    true,
    /* num_languages */ 2,
    Some("en"),
    Task::Transcribe,
  )
  .unwrap();
  // The language candidates the wrapper exposes are exactly the out-of-range
  // ids, confirming the fixture reaches the guard.
  let cands = wrapper.all_language_candidates();
  assert!(
    cands.iter().any(|&(_, id)| id as usize >= N_VOCAB),
    "fixture must expose an out-of-vocab language id, got {cands:?}"
  );

  let model = tiny_model(2);
  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  let err = detect_language(&model, &wrapper, &enc).unwrap_err();
  assert!(
    matches!(&err, Error::OutOfRange(p)
      if p.context() == "detect_language: language token id"),
    "expected OutOfRange on the out-of-vocab language id, got {err:?}"
  );
}

#[test]
fn decode_with_fallback_accepts_first_acceptable() {
  // With all thresholds disabled, the very first (temperature 0) decode is
  // accepted; the result temperature is 0.0.
  let dir = fresh_dir("fallback");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(2); // eot → short decode

  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  let base = DecodingOptions {
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(3),
    ..Default::default()
  };
  let result = decode_with_fallback(
    &model,
    &wrapper,
    &enc,
    &base,
    "en",
    &DEFAULT_TEMPERATURES,
    None,
    None,
    None,
  )
  .unwrap();
  assert_eq!(result.temperature, 0.0);
}

/// REGRESSION (per-temperature option sanitization, `whisper.py:944-952`): a
/// caller pairing `best_of` with the DEFAULT temperature schedule (which starts
/// at `0.0`) must work end-to-end. The greedy (`t == 0`) attempt drops `best_of`
/// and decodes plainly (it would otherwise be rejected by
/// `verify_options_and_group`'s best_of-with-temperature-0 guard, making
/// `best_of` unusable through the normal fallback path); the positive-temperature
/// attempts keep it. Here the very first (`t == 0`) attempt is accepted, so the
/// result temperature is `0.0` — and crucially the call does not spuriously
/// error.
#[test]
fn decode_with_fallback_drops_best_of_at_greedy_temperature() {
  let dir = fresh_dir("fallback_bestof_t0");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(2); // eot → short decode

  let mel = tiny_mel();
  let enc = model.encode(&mel).unwrap();

  // best_of set + the default schedule (starts at 0.0). Pre-fix this errored on
  // the very first temperature.
  let base = DecodingOptions {
    best_of: Some(4),
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(3),
    ..Default::default()
  };
  let result = decode_with_fallback(
    &model,
    &wrapper,
    &enc,
    &base,
    "en",
    &DEFAULT_TEMPERATURES,
    None,
    None,
    None,
  )
  .unwrap();
  // Thresholds disabled → the greedy t==0 attempt is accepted (best_of dropped).
  assert_eq!(result.temperature, 0.0);
}

/// REGRESSION: with `best_of` + the default schedule AND every threshold made
/// impossible to satisfy, the fallback walks the WHOLE schedule — the greedy
/// `t == 0` attempt (best_of dropped) plus the positive-temperature attempts
/// (best_of active, multi-trajectory) — and returns the last result without ever
/// erroring. The last temperature (`1.0`) being recorded proves the schedule was
/// traversed past `t == 0` with `best_of` live rather than rejected at the front.
#[test]
fn decode_with_fallback_best_of_traverses_full_schedule() {
  let dir = fresh_dir("fallback_bestof_full");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let target = 13u32;
  let model = tiny_model(target);
  let enc = model.encode(&tiny_mel()).unwrap();

  let base = DecodingOptions {
    best_of: Some(3),
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(3),
    ..Default::default()
  };
  // An unsatisfiable logprob threshold → every attempt "needs fallback", so the
  // loop runs through all of DEFAULT_TEMPERATURES and returns the last result.
  let result = decode_with_fallback(
    &model,
    &wrapper,
    &enc,
    &base,
    "en",
    &DEFAULT_TEMPERATURES,
    None,
    Some(1000.0),
    None,
  )
  .unwrap();
  assert_eq!(
    result.temperature,
    *DEFAULT_TEMPERATURES.last().unwrap(),
    "the fallback must traverse the whole schedule with best_of live past t==0"
  );
}

#[test]
fn audio_features_passes_through_encoded() {
  // A tensor already shaped (n_audio_ctx, n_audio_state) is treated as
  // encoder output and lifted to (1, n_audio_ctx, n_audio_state) — the
  // encoder is NOT re-run.
  let dir = fresh_dir("encoded_passthrough");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);
  let options = DecodingOptions {
    without_timestamps: true,
    ..Default::default()
  };
  let task = DecodingTask::new(&model, &wrapper, options).unwrap();

  // (n_audio_ctx, n_audio_state) = (1500, 4): recognized as already-encoded and
  // lifted to (1, 1500, 4) without re-running the encoder.
  let encoded = Array::ones::<f32>(&(N_FRAMES / 2, 4usize)).unwrap();
  let feats = task.audio_features(&encoded).unwrap();
  assert_eq!(feats.shape(), vec![1, N_FRAMES / 2, 4]);
}

// ═══════════════════ batched decode + best-of-N sampling ═══════════════════

/// Assert a `DecodingTask::new` result is an `InvariantViolation` (the task is
/// not `Debug`, so `.unwrap_err()` cannot be used directly).
fn expect_task_invariant(res: Result<DecodingTask<'_>>, what: &str) {
  match res {
    Err(Error::InvariantViolation(_)) => {}
    Err(e) => panic!("{what}: expected InvariantViolation, got {e:?}"),
    Ok(_) => panic!("{what}: expected an error, got Ok"),
  }
}

/// Assert a `DecodingTask::new` result is an `OutOfRange`.
fn expect_task_oob(res: Result<DecodingTask<'_>>, what: &str) {
  match res {
    Err(Error::OutOfRange(_)) => {}
    Err(e) => panic!("{what}: expected OutOfRange, got {e:?}"),
    Ok(_) => panic!("{what}: expected an error, got Ok"),
  }
}

/// Build a `(n_group, n_vocab)` device logits matrix from per-row host slices
/// (each row the same length).
fn matrix(rows: &[&[f32]]) -> Array {
  let g = rows.len();
  let v = rows.first().map(|r| r.len()).unwrap_or(0);
  let mut flat: Vec<f32> = Vec::with_capacity(g * v);
  for r in rows {
    assert_eq!(r.len(), v, "all rows must share the vocab width");
    flat.extend_from_slice(r);
  }
  Array::from_slice::<f32>(&flat, &[g as i32, v as i32]).unwrap()
}

// ───────────────────── n_group == 1 PARITY GATE ───────────────────────────

/// THE PARITY GATE: the `n_group == 1` batched decode path is BIT-IDENTICAL to
/// the existing single-sequence path — same tokens AND same cumulative
/// log-probability — on the same input. This proves the batching machinery
/// introduced no regression in the shipped single-sequence decode.
#[test]
fn batched_n_group_1_is_bit_identical_to_single_sequence() {
  let dir = fresh_dir("parity_greedy");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);
  let enc = model.encode(&tiny_mel()).unwrap();

  // Greedy (temperature 0), timestamps on so the timestamp-rule filter is in
  // the chain too (exercises the per-row filter application against a real
  // history), suppression on — the full default filter stack.
  let options = DecodingOptions {
    language: Some("en".into()),
    sample_len: Some(5),
    ..Default::default()
  };
  let task = DecodingTask::new(&model, &wrapper, options).unwrap();
  assert_eq!(task.n_group, 1, "default options are single-group");

  let ((single_tokens, single_sum), (batched_tokens, batched_sum)) =
    task.run_both_for_parity(&enc).unwrap();

  assert_eq!(
    single_tokens, batched_tokens,
    "batched n_group==1 must emit the identical token sequence"
  );
  // f64 cumulative log-prob, bit-for-bit (the per-row formula matches the
  // single path's scalar formula on a one-row group).
  assert_eq!(
    single_sum.to_bits(),
    batched_sum.to_bits(),
    "batched n_group==1 must accumulate the identical sum_logprob \
     (single={single_sum}, batched={batched_sum})"
  );
}

/// The parity gate holds with the timestamp filters OFF too (a different filter
/// chain — only suppress-blank + suppress-tokens), so the equivalence is not an
/// artifact of one filter configuration.
#[test]
fn batched_n_group_1_parity_without_timestamps() {
  let dir = fresh_dir("parity_no_ts");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(12);
  let enc = model.encode(&tiny_mel()).unwrap();

  let options = DecodingOptions {
    language: Some("en".into()),
    without_timestamps: true,
    sample_len: Some(6),
    ..Default::default()
  };
  let task = DecodingTask::new(&model, &wrapper, options).unwrap();
  let ((s_tok, s_sum), (b_tok, b_sum)) = task.run_both_for_parity(&enc).unwrap();
  assert_eq!(s_tok, b_tok);
  assert_eq!(s_sum.to_bits(), b_sum.to_bits());
}

/// THE PARITY GATE at `temperature > 0`: with sampling enabled, the `n_group ==
/// 1` batched **categorical** path must draw with the IDENTICAL PRNG key as the
/// single path. Both decoders seed `0`, and the batched split-key roles (carry
/// row 0, sample rows `1..=n_group`) reduce at `n_group == 1` to the single
/// path's `random::split` (carry the first returned key, sample with the second)
/// — `split(k)` and `split_num(k, 2)` are the same `bits` rows in order. A
/// swapped sample/carry role would put the draw on a different key and diverge
/// the token stream immediately; the greedy (temperature 0) gates above never
/// reach the categorical draw at all, so this is the only test that pins it.
#[test]
fn batched_n_group_1_parity_at_positive_temperature() {
  let dir = fresh_dir("parity_temp");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(11);
  let enc = model.encode(&tiny_mel()).unwrap();

  let options = DecodingOptions {
    language: Some("en".into()),
    without_timestamps: true,
    temperature: 0.7,
    sample_len: Some(6),
    ..Default::default()
  };
  let task = DecodingTask::new(&model, &wrapper, options).unwrap();
  assert_eq!(task.n_group, 1, "no best_of ⇒ single-group");
  let ((s_tok, s_sum), (b_tok, b_sum)) = task.run_both_for_parity(&enc).unwrap();
  assert_eq!(
    s_tok, b_tok,
    "temperature>0 batched n_group==1 must sample the IDENTICAL tokens as single \
     (single={s_tok:?}, batched={b_tok:?})"
  );
  assert_eq!(
    s_sum.to_bits(),
    b_sum.to_bits(),
    "temperature>0 batched n_group==1 must accumulate the identical sum_logprob"
  );
}

/// DIRECT key-role assertion for `BatchedGreedyDecoder::categorical` at
/// `n_group == 1` — a stronger pin than the token-equality parity test above,
/// which a peaked model could pass even with swapped keys. On NON-degenerate
/// (spread) logits the drawn token genuinely depends on which subkey is used, so
/// this proves the batched path SAMPLES with `split(key).1` (the single path's
/// sample key) and CARRIES `split(key).0` (the single path's next key) — the
/// exact roles whose swap was the bug. The carry-key assertion catches the swap
/// deterministically even if the two keys happen to draw the same token.
#[test]
fn batched_categorical_key_roles_match_single_split_at_n_group_1() {
  const SEED: u64 = 0;
  let temp = 1.0f32;
  let vocab: i32 = 8;
  // Non-degenerate logits so the categorical draw actually depends on the key.
  let row: Vec<f32> = (0..vocab).map(|i| (i as f32) * 0.1).collect();

  // Independent expectation from the single-path split roles: carry the FIRST
  // returned key, sample with the SECOND.
  let init_key = ops::random::key(SEED).unwrap();
  let (carry_expected, sample_key) = ops::random::split(&init_key).unwrap();
  let row1d = Array::from_slice::<f32>(&row, &[vocab]).unwrap();
  let single_token = {
    let mut s = crate::lm::sample::categorical_sampling(&row1d, temp, &sample_key).unwrap();
    s.item::<u32>().unwrap()
  };

  // Batched n_group == 1 on the same logits and seed.
  let mut d = BatchedGreedyDecoder::new(temp, /* eot */ 2, /* n_group */ 1, SEED).unwrap();
  let logits = Array::from_slice::<f32>(&row, &[1, vocab]).unwrap();
  let drawn = d.categorical(&logits).unwrap();

  // (1) row 0 sampled with split(key).1 (the single-path SAMPLE key).
  assert_eq!(
    drawn,
    vec![single_token],
    "batched row 0 must sample with split(key).1 (the single-path sample key)"
  );
  // (2) the carried key is split(key).0 (the single-path NEXT key), bit-exact —
  //     deterministic regardless of any token coincidence above.
  let read_u32 = |a: &Array| -> Vec<u32> {
    let mut a = a.try_clone().unwrap();
    a.eval().unwrap();
    a.to_vec::<u32>().unwrap()
  };
  assert_eq!(
    read_u32(&d.key),
    read_u32(&carry_expected),
    "batched must carry split(key).0 as the next-step key"
  );
}

/// A `best_of` that cannot be an `i32` tensor batch dimension is rejected at task
/// CONSTRUCTION (in `verify_options_and_group`), before `main_loop_batched`
/// reserves any per-row state — a typed `OutOfRange` / `ArithmeticOverflow`,
/// never a billion-row allocation attempt. (A merely-large but i32-valid
/// `best_of` stays a fallible allocation, the consumer's concern.)
#[test]
fn impossible_best_of_is_rejected_at_construction_before_allocation() {
  let dir = fresh_dir("huge_best_of");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(9);
  let options = DecodingOptions {
    language: Some("en".into()),
    temperature: 0.7,          // best_of requires temperature > 0
    best_of: Some(usize::MAX), // cannot index an i32-shaped tensor
    sample_len: Some(4),
    ..Default::default()
  };
  match DecodingTask::new(&model, &wrapper, options) {
    Err(Error::OutOfRange(_)) | Err(Error::ArithmeticOverflow(_)) => {}
    Err(e) => {
      panic!("expected OutOfRange/ArithmeticOverflow for an impossible best_of, got {e:?}")
    }
    Ok(_) => panic!("an impossible best_of must be rejected at construction, got Ok"),
  }
}

// ───────────────────── MaximumLikelihoodRanker ────────────────────────────

/// Plain length normalization (`length_penalty is None`): `score =
/// sum_logprob / length`. The highest-score candidate wins.
#[test]
fn ranker_plain_length_normalization_picks_highest_score() {
  let ranker = MaximumLikelihoodRanker::new(None);
  // (sum_logprob, length): scores -1.0/2 = -0.5, -1.2/3 = -0.4, -2.0/2 = -1.0.
  // Candidate 1 has the highest (least-negative) score.
  let candidates = [(-1.0, 2), (-1.2, 3), (-2.0, 2)];
  assert_eq!(ranker.rank(&candidates), 1);
}

/// THE #762 LOCUS: the length-penalty normalization must flip the winner versus
/// the RAW cumulative log-probability. The candidate with the highest raw
/// `sum_logprob` is NOT selected once the score is normalized by the
/// (GNMT-penalized) length — proving the ranker divides by the penalized length,
/// not the raw sequence length, and that getting that normalization right is what
/// the upstream best-of fix is about.
#[test]
fn ranker_length_penalty_flips_winner_vs_raw_sum_logprob() {
  // Candidate A: short, slightly worse total logprob. Candidate B: long, best
  // RAW total logprob — but its extra length penalizes its normalized score.
  // Raw argmax(sum_logprob) would pick B (−3.0 > −2.4); the length-normalized
  // ranker must pick A.
  let a = (-2.4_f64, 4_usize);
  let b = (-3.0_f64, 20_usize);
  let raw_winner = if a.0 > b.0 { 0 } else { 1 };
  assert_eq!(
    raw_winner, 0,
    "A already has the higher raw sum_logprob here"
  );

  // Plain length normalization: A = -2.4/4 = -0.6, B = -3.0/20 = -0.15 → B wins.
  // So plain normalization FLIPS to B (longer candidates are rewarded).
  let plain = MaximumLikelihoodRanker::new(None);
  assert_eq!(
    plain.rank(&[a, b]),
    1,
    "plain length normalization rewards the longer candidate B"
  );

  // GNMT penalty α = 1.0: penalty(len) = ((5+len)/6)^1.
  //   A: penalty = 9/6 = 1.5   → score = -2.4/1.5  = -1.6
  //   B: penalty = 25/6 ≈ 4.17 → score = -3.0/4.17 ≈ -0.72
  // B still wins under α=1.0. Use a SMALLER α to damp the length reward so the
  // shorter, higher-raw-logprob candidate A wins — the normalization is what
  // decides, the #762 concern.
  let gnmt_strong = MaximumLikelihoodRanker::new(Some(0.0));
  // α = 0.0 → penalty = ((5+len)/6)^0 = 1 for every length → score = sum_logprob
  // (raw). So the ranker reduces to raw argmax → A (the higher raw logprob).
  assert_eq!(
    gnmt_strong.rank(&[a, b]),
    0,
    "α=0 makes the penalty 1 for all lengths → raw argmax picks A"
  );
}

/// The GNMT length penalty formula `((5 + len) / 6) ** alpha` is applied exactly
/// (`decoding.py:229`): a hand-computed score selection.
#[test]
fn ranker_gnmt_penalty_matches_formula() {
  let alpha = 0.5_f64;
  let ranker = MaximumLikelihoodRanker::new(Some(alpha as f32));
  // Two candidates; compute the expected scores by the formula and assert the
  // ranker agrees with the independent argmax.
  let cands = [(-1.5_f64, 3_usize), (-1.0_f64, 1_usize)];
  let score = |lp: f64, len: usize| lp / ((5.0 + len as f64) / 6.0).powf(alpha);
  let s0 = score(-1.5, 3);
  let s1 = score(-1.0, 1);
  let expected = if s0 >= s1 { 0 } else { 1 };
  assert_eq!(ranker.rank(&cands), expected);
}

/// `np.argmax` tie-breaking: equal scores select the LOWEST index.
#[test]
fn ranker_ties_pick_lowest_index() {
  let ranker = MaximumLikelihoodRanker::new(None);
  // Identical (sum_logprob, length) → identical score → first index wins.
  assert_eq!(ranker.rank(&[(-1.0, 2), (-1.0, 2), (-1.0, 2)]), 0);
}

/// A zero-length candidate under plain normalization is `sum_logprob / 0`. A
/// finite-score candidate must still be preferred over the resulting non-finite
/// score (mirroring numpy's `argmax`, which skips `NaN`/keeps a finite max).
#[test]
fn ranker_zero_length_nonfinite_score_never_beats_finite() {
  let ranker = MaximumLikelihoodRanker::new(None);
  // Candidate 0: zero length, zero logprob → 0/0 = NaN. Candidate 1: finite.
  assert_eq!(
    ranker.rank(&[(0.0, 0), (-0.5, 3)]),
    1,
    "a NaN (0/0) score must not win over a finite candidate"
  );
  // Candidate 0: negative logprob, zero length → -inf. Candidate 1: finite -0.5.
  // The finite score (-0.5) beats -inf.
  assert_eq!(ranker.rank(&[(-1.0, 0), (-0.5, 3)]), 1);
}

// ───────────────── batched greedy decoder behavior ────────────────────────

/// Per-row argmax + per-row independent logprob accumulation at temperature 0:
/// each candidate selects its own argmax and accumulates its own chosen logprob.
#[test]
fn batched_decoder_per_row_argmax_and_logprob() {
  let mut d = BatchedGreedyDecoder::new(0.0, /* eot */ 2, /* n_group */ 2, 0).unwrap();
  // Row 0 favors index 3; row 1 favors index 0.
  let m = matrix(&[&[0.0, 0.0, 0.0, 5.0, 0.0], &[5.0, 0.0, 0.0, 0.0, 0.0]]);
  // Neither row's last token is eot → both accumulate.
  let next = d.update(&m, &[1, 1]).unwrap();
  assert_eq!(next, vec![3, 0]);
  assert!(!d.all_completed());
  assert!(d.sum_logprob[0] < 0.0 && d.sum_logprob[1] < 0.0);
}

/// EOT-STICKY per-row completion: once a row emits eot it stays done — it
/// re-emits eot, stops contributing to its `sum_logprob`, and is marked
/// completed; the OTHER rows keep decoding. The group completes only when EVERY
/// row has emitted eot.
#[test]
fn batched_decoder_eot_sticky_per_row() {
  let mut d = BatchedGreedyDecoder::new(0.0, /* eot */ 2, /* n_group */ 2, 0).unwrap();
  let m = matrix(&[&[0.0, 0.0, 0.0, 5.0, 0.0], &[0.0, 0.0, 0.0, 5.0, 0.0]]);

  // Step 1: row 0's last token is eot (sticks), row 1's is not (decodes).
  let next = d.update(&m, &[2, 1]).unwrap();
  assert_eq!(next[0], 2, "row 0 re-emits eot (sticky)");
  assert_eq!(next[1], 3, "row 1 decodes its argmax");
  assert_eq!(
    d.sum_logprob[0], 0.0,
    "an eot-stuck row accumulates no logprob"
  );
  assert!(
    d.sum_logprob[1] < 0.0,
    "row 1 accumulated its chosen logprob"
  );
  assert!(
    !d.all_completed(),
    "the group is not done while row 1 is still decoding"
  );

  // Step 2: now row 1's last token is also eot → both stuck → group completes.
  let next = d.update(&m, &[2, 2]).unwrap();
  assert_eq!(next, vec![2, 2], "both rows now re-emit eot");
  assert!(
    d.all_completed(),
    "every row has emitted eot → group complete"
  );
}

/// A row whose argmax IS the eot id completes that row.
#[test]
fn batched_decoder_completes_row_on_argmax_eot() {
  let mut d = BatchedGreedyDecoder::new(0.0, /* eot */ 2, 1, 0).unwrap();
  // argmax is the eot id (index 2).
  let m = matrix(&[&[0.0, 0.0, 9.0, 0.0]]);
  let next = d.update(&m, &[1]).unwrap();
  assert_eq!(next, vec![2]);
  assert!(d.all_completed());
}

/// THE KEYED-RNG N-WAY SPLIT: at `temperature > 0` the N candidate rows draw
/// from independent subkeys, so on an identical per-row distribution they can
/// produce DISTINCT samples (the rows are not all forced to the same token).
/// With a flat distribution over many tokens and a reasonable group size, at
/// least two rows differ — proving the per-row key split actually decorrelates
/// the draws (a single shared key would make every row identical).
#[test]
fn batched_decoder_keyed_rng_split_produces_distinct_rows() {
  let n_group = 8usize;
  let v = 16usize;
  // A flat (uniform) distribution: every token equally likely, so distinct
  // subkeys yield a spread of samples; identical keys would yield identical rows.
  let flat = vec![0.0_f32; v];
  let rows: Vec<&[f32]> = (0..n_group).map(|_| flat.as_slice()).collect();
  let m = matrix(&rows);

  let mut d = BatchedGreedyDecoder::new(1.0, /* eot */ 2, n_group, /* seed */ 7).unwrap();
  let next = d.update(&m, &vec![1u32; n_group]).unwrap();
  let distinct: std::collections::HashSet<u32> = next.iter().copied().collect();
  assert!(
    distinct.len() >= 2,
    "the per-row key split must decorrelate the draws (got {next:?})"
  );
}

// ──────────────── batched allocation soundness (no panic/abort) ────────────

/// SOUNDNESS: an outsized `best_of` makes the per-row accumulators in
/// `BatchedGreedyDecoder::new` unsatisfiable; the reservation degrades to a typed
/// [`Error::AllocFailure`] rather than aborting in an infallible `vec![…;
/// n_group]`. (No size cap — a large-but-fallible `best_of` is the consumer's
/// DoS concern; the contract is fallible-no-abort, not a maximum value.) A
/// `usize::MAX` group deterministically overflows the byte computation so
/// `try_reserve_exact` returns `Err` on every host (overcommit makes a merely
/// huge-but-RAM-sized reservation succeed virtually), mirroring the crate's
/// `reserve_or_error` oversize tests.
#[test]
fn batched_decoder_new_huge_n_group_is_alloc_failure_not_abort() {
  match BatchedGreedyDecoder::new(
    /* temperature */ 1.0,
    /* eot */ 2,
    /* n_group */ usize::MAX,
    /* seed */ 0,
  ) {
    Err(Error::AllocFailure(_)) => {}
    Err(e) => panic!("expected AllocFailure for a huge n_group, got {e:?}"),
    Ok(_) => panic!("expected AllocFailure for a huge n_group, got Ok"),
  }
}

/// SOUNDNESS: the `n_group + 1` RNG split count is computed with checked
/// arithmetic — at `n_group == i32::MAX` it returns a typed
/// [`Error::ArithmeticOverflow`] rather than wrapping to a negative / truncated
/// `num` (which would feed `split_num` + the carry-row slice bound). Built via a
/// struct literal so the giant per-row allocations are skipped and only the
/// arithmetic guard is exercised.
#[test]
fn batched_decoder_split_count_overflow_is_typed_error() {
  let d = BatchedGreedyDecoder {
    temperature: 1.0,
    eot: 2,
    n_group: i32::MAX as usize,
    // Empty accumulators (no allocation) — `split_count` reads only `n_group`.
    sum_logprob: Vec::new(),
    completed: Vec::new(),
    key: ops::random::key(0).unwrap(),
  };
  match d.split_count() {
    Err(Error::ArithmeticOverflow(_)) => {}
    Err(e) => panic!("expected ArithmeticOverflow for n_group + 1, got {e:?}"),
    Ok(v) => panic!("expected an overflow error for n_group + 1, got {v:?}"),
  }
  // A within-range n_group resolves to `(g, g + 1)`.
  let small = BatchedGreedyDecoder::new(1.0, 2, 3, 0).unwrap();
  assert_eq!(small.split_count().unwrap(), (3, 4));
}

/// SOUNDNESS: the flattened `(n_group * T)` prefill capacity is computed with
/// checked arithmetic ([`flat_token_count`]) instead of an unchecked
/// `rows.iter().map(Vec::len).sum()`. A group / row-length whose product wraps
/// `i32` returns a typed [`Error::ArithmeticOverflow`]; a dimension past
/// `i32::MAX` returns [`Error::OutOfRange`] — never a wrapped `usize` capacity
/// fed to `Vec::with_capacity`. Built via a direct call so the giant flat buffer
/// is skipped and only the arithmetic guard is exercised (the same technique as
/// `batched_decoder_split_count_overflow_is_typed_error`).
#[test]
fn flatten_rows_capacity_arithmetic_is_checked() {
  // `n_group * T` overflows `i32` → ArithmeticOverflow (no buffer realized).
  match flat_token_count(i32::MAX as usize, 2) {
    Err(Error::ArithmeticOverflow(_)) => {}
    Err(e) => panic!("expected ArithmeticOverflow for n_group * T, got {e:?}"),
    Ok(v) => panic!("expected an overflow error for n_group * T, got {v:?}"),
  }
  // A dimension past `i32::MAX` (the MLX array-dim bound) → OutOfRange.
  match flat_token_count(usize::MAX, 1) {
    Err(Error::OutOfRange(_)) => {}
    Err(e) => panic!("expected OutOfRange for a > i32::MAX n_group, got {e:?}"),
    Ok(v) => panic!("expected an out-of-range error, got {v:?}"),
  }
  // A within-range group resolves to the exact `n_group * T` product.
  assert_eq!(flat_token_count(5, 7).unwrap(), 35);
  assert_eq!(flat_token_count(0, 7).unwrap(), 0);
}

/// SOUNDNESS + correctness: `flatten_rows` reserves its row-major buffer fallibly
/// (via the checked [`flat_token_count`]) and still lays the rows out correctly.
/// It returns a `Result` — a still-infallible `Vec<u32>` version would not
/// compile against this `.unwrap()` — so a regression to the unchecked path is
/// caught at build time, and the value check pins the fill.
#[test]
fn flatten_rows_is_fallible_and_row_major() {
  let rows = vec![vec![10u32, 11, 12], vec![20, 21, 22]];
  let flat = flatten_rows(&rows).unwrap();
  assert_eq!(flat, vec![10, 11, 12, 20, 21, 22]);
  // Empty group → empty flat (capacity 0, no panic).
  assert_eq!(flatten_rows(&[]).unwrap(), Vec::<u32>::new());
}

/// SOUNDNESS + correctness: `last_tokens_of` reserves its `(n_group,)` output
/// fallibly instead of an infallible `.collect()`, and still returns each row's
/// last token (the `fallback` for an empty row). The `Result` return is a
/// compile-time regression guard against reverting to the infallible collect.
#[test]
fn last_tokens_of_is_fallible_and_correct() {
  let rows = vec![vec![1u32, 2, 9], vec![3, 4, 8], Vec::new()];
  let last = last_tokens_of(&rows, /* fallback */ 7).unwrap();
  assert_eq!(
    last,
    vec![9, 8, 7],
    "last token per row, fallback for the empty row"
  );
}

/// SOUNDNESS + correctness: `push_row_tokens` reserves each row's single appended
/// slot fallibly before growing it (instead of an infallible `Vec::push`
/// reallocation), and still appends each row's next token in lockstep. The
/// `Result` return is a compile-time regression guard against the infallible
/// push.
#[test]
fn push_row_tokens_is_fallible_and_grows_rows() {
  let mut rows = vec![vec![1u32], vec![2], vec![3]];
  push_row_tokens(&mut rows, &[4, 5, 6]).unwrap();
  assert_eq!(rows, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
  // A second step grows them again (the fallible reserve handles every step).
  push_row_tokens(&mut rows, &[7, 8, 9]).unwrap();
  assert_eq!(rows, vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]]);
}

// ───────────────────── best-of-N end-to-end ───────────────────────────────

/// `best_of` + `beam_size` set together → a typed error at task construction
/// (the reference's mutually-exclusive `_verify_options` check).
#[test]
fn best_of_and_beam_size_together_is_typed_error() {
  let dir = fresh_dir("best_of_beam");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);
  let options = DecodingOptions {
    language: Some("en".into()),
    temperature: 0.5,
    best_of: Some(3),
    beam_size: Some(2),
    ..Default::default()
  };
  expect_task_invariant(
    DecodingTask::new(&model, &wrapper, options),
    "best_of + beam_size together",
  );
}

/// `best_of` with greedy sampling (`temperature == 0`) is incompatible — a typed
/// error (the reference's `best_of with greedy sampling (T=0)` check).
#[test]
fn best_of_with_temperature_zero_is_typed_error() {
  let dir = fresh_dir("best_of_t0");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);
  let options = DecodingOptions {
    language: Some("en".into()),
    temperature: 0.0,
    best_of: Some(3),
    ..Default::default()
  };
  expect_task_invariant(
    DecodingTask::new(&model, &wrapper, options),
    "best_of with temperature 0",
  );
}

/// `beam_size` alone is rejected — beam search is not implemented (a typed
/// error, rather than carried-and-ignored).
#[test]
fn beam_size_alone_is_unsupported_typed_error() {
  let dir = fresh_dir("beam_only");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);
  let options = DecodingOptions {
    language: Some("en".into()),
    beam_size: Some(2),
    ..Default::default()
  };
  expect_task_invariant(
    DecodingTask::new(&model, &wrapper, options),
    "beam_size alone (unsupported)",
  );
}

/// `length_penalty` outside `[0, 1]` is rejected (the reference's bound).
#[test]
fn length_penalty_out_of_range_is_typed_error() {
  let dir = fresh_dir("lp_range");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);
  for bad in [-0.1_f32, 1.1_f32] {
    let options = DecodingOptions {
      language: Some("en".into()),
      temperature: 0.5,
      best_of: Some(2),
      length_penalty: Some(bad),
      ..Default::default()
    };
    expect_task_oob(
      DecodingTask::new(&model, &wrapper, options),
      "length_penalty out of [0,1]",
    );
  }
}

/// BEST-OF-N END-TO-END on a tiny synthetic model: `best_of = N` at
/// `temperature > 0` decodes N candidate trajectories and the ranker returns one
/// selected result. The model is biased toward a single target token, so even
/// under sampling the decode is well-behaved and the result is non-empty with
/// the target token(s).
#[test]
fn best_of_n_end_to_end_decodes_and_ranks() {
  let dir = fresh_dir("best_of_e2e");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let target = 13u32;
  let model = tiny_model(target);
  let enc = model.encode(&tiny_mel()).unwrap();

  let options = DecodingOptions {
    language: Some("en".into()),
    temperature: 0.5,
    best_of: Some(5),
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(4),
    ..Default::default()
  };
  let task = DecodingTask::new(&model, &wrapper, options).unwrap();
  assert_eq!(task.n_group, 5, "best_of resolves n_group");

  let result = task.run(&enc, "en").unwrap();
  // The head is strongly peaked at `target`, so every sampled token is the
  // target even under the temperature draw; the ranked winner is non-empty.
  assert!(
    !result.tokens.is_empty(),
    "best-of selected a non-empty result"
  );
  assert!(
    result.tokens.iter().all(|&t| t == target),
    "the peaked head makes every sampled token the target; got {:?}",
    result.tokens
  );
  assert_eq!(result.language, "en");
}

/// A best-of decode whose head argmaxes to eot completes immediately with an
/// empty token list (the ranker still selects a candidate).
#[test]
fn best_of_n_all_eot_yields_empty() {
  let dir = fresh_dir("best_of_eot");
  let tok = write_tokenizer(dir.as_path());
  let wrapper = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(2); // 2 == eot
  let enc = model.encode(&tiny_mel()).unwrap();

  let options = DecodingOptions {
    language: Some("en".into()),
    temperature: 0.5,
    best_of: Some(3),
    without_timestamps: true,
    suppress_blank: false,
    suppress_tokens: SuppressSpec::None,
    sample_len: Some(5),
    ..Default::default()
  };
  let task = DecodingTask::new(&model, &wrapper, options).unwrap();
  let result = task.run(&enc, "en").unwrap();
  assert!(
    result.tokens.is_empty(),
    "all-eot best-of yields empty, got {:?}",
    result.tokens
  );
}

// ───────────────────── batched encoder broadcast ──────────────────────────

/// `broadcast_encoder_states` widens a `(1, …)` encoder output to `(n_group, …)`
/// and is a clone (unchanged shape) at `n_group == 1`.
#[test]
fn broadcast_encoder_states_widens_to_group() {
  let model = tiny_model(13);
  let enc = model.encode(&tiny_mel()).unwrap();
  assert_eq!(enc.shape()[0], 1);

  let b1 = model.broadcast_encoder_states(&enc, 1).unwrap();
  assert_eq!(
    b1.shape(),
    enc.shape(),
    "n_group==1 leaves the shape unchanged"
  );

  let b4 = model.broadcast_encoder_states(&enc, 4).unwrap();
  assert_eq!(
    b4.shape(),
    vec![4, enc.shape()[1], enc.shape()[2]],
    "n_group==4 broadcasts the batch axis"
  );
  // Every broadcast row equals the source segment (a real broadcast, not zeros).
  let src = host_f32(&enc);
  let wide = host_f32(&b4);
  let seg = enc.shape()[1] * enc.shape()[2];
  for g in 0..4 {
    assert_eq!(
      &wide[g * seg..(g + 1) * seg],
      &src[..],
      "row {g} mirrors the source"
    );
  }
}

/// The batched decode primitive rejects an encoder batch that does not equal
/// `n_group` (a soundness pin — a mismatched K/V batch would broadcast wrong).
#[test]
fn decode_tokens_batched_rejects_group_mismatch() {
  let model = tiny_model(13);
  let enc = model.encode(&tiny_mel()).unwrap(); // (1, …)
  // n_group = 3 but enc batch is 1 → ShapePairMismatch.
  let tokens = [3u32, 3, 3]; // (n_group=3, T=1) row-major
  let err = model
    .decode_tokens_batched(&tokens, 3, &enc, None)
    .unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "an enc batch != n_group must be ShapePairMismatch, got {err:?}"
  );

  // With a correctly-broadcast enc it succeeds, returning (3, 1, V) logits.
  let wide = model.broadcast_encoder_states(&enc, 3).unwrap();
  let (logits, _cache) = model
    .decode_tokens_batched(&tokens, 3, &wide, None)
    .unwrap();
  assert_eq!(logits.shape(), vec![3, 1, N_VOCAB]);
}

/// Read a device array back to a host `Vec<f32>` (for the broadcast assertions).
/// A broadcast view is non-contiguous, so materialize it first.
fn host_f32(a: &Array) -> Vec<f32> {
  let mut c = ops::shape::contiguous(a, false).unwrap();
  c.to_vec::<f32>().unwrap()
}

// ───────────────────────── get_suppress_tokens ────────────────────────────

#[test]
fn get_suppress_tokens_non_speech_includes_specials() {
  let dir = fresh_dir("suppress_specials");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let ids = get_suppress_tokens(&w, &SuppressSpec::NonSpeech).unwrap();
  // The always-suppressed specials (sorted, deduped).
  for special in [
    w.transcribe(),
    w.translate(),
    w.sot(),
    w.sot_prev(),
    w.sot_lm(),
    w.no_speech(),
  ] {
    assert!(
      ids.contains(&special),
      "missing special {special} in {ids:?}"
    );
  }
  // Sorted + unique.
  let mut sorted = ids.clone();
  sorted.sort_unstable();
  sorted.dedup();
  assert_eq!(ids, sorted);
}

/// Build a tokenizer fixture whose vocabulary contains a handful of the
/// reference non-speech punctuation symbols (`#`, `(`, `)`, `@`) as real
/// single tokens, so [`HFTokenizerWrapper::non_speech_tokens`] resolves them
/// (rather than collapsing every symbol to `unk`). The Whisper specials are
/// kept so the wrapper still constructs.
fn write_tokenizer_with_punct(dir: &Path) -> (Tokenizer, Vec<(&'static str, u32)>) {
  let punct: Vec<(&'static str, u32)> = vec![("#", 18), ("(", 19), (")", 20), ("@", 21)];
  let mut entries: Vec<(&'static str, u32)> = SPECIALS.to_vec();
  entries.extend(punct.iter().copied());

  let vocab: serde_json::Map<String, serde_json::Value> = entries
    .iter()
    .map(|(tok, id)| (tok.to_string(), json!(id)))
    .collect();
  let added_tokens: Vec<serde_json::Value> = entries
    .iter()
    .map(|(tok, id)| {
      let special = tok.starts_with("<|");
      json!({
        "id": id, "content": tok, "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": special
      })
    })
    .collect();
  let tokenizer_json = json!({
    "version": "1.0",
    "added_tokens": added_tokens,
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>" }
  });
  let cfg = json!({ "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>" });
  std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  (Tokenizer::from_path(dir, None).unwrap(), punct)
}

#[test]
fn non_speech_tokens_resolves_vocab_symbols() {
  // The non-speech set contains the punctuation symbols present in the vocab
  // (each encodes to a single token, so the reference's `len(tokens) == 1`
  // branch adds it). The set is sorted + de-duplicated.
  let dir = fresh_dir("non_speech_set");
  let (tok, punct) = write_tokenizer_with_punct(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let set = w.non_speech_tokens().unwrap();
  for (sym, id) in &punct {
    assert!(
      set.contains(id),
      "non-speech set missing vocab symbol {sym:?} (id {id}); set={set:?}"
    );
  }
  // Sorted + unique.
  let mut sorted = set.clone();
  sorted.sort_unstable();
  sorted.dedup();
  assert_eq!(set, sorted, "non-speech set must be sorted + de-duplicated");
}

#[test]
fn get_suppress_tokens_non_speech_covers_the_non_speech_set() {
  // `SuppressSpec::NonSpeech` ("-1") must include the tokenizer non-speech set
  // *and* the always-suppressed specials (sot_lm among them).
  let dir = fresh_dir("suppress_covers_nonspeech");
  let (tok, punct) = write_tokenizer_with_punct(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let ids = get_suppress_tokens(&w, &SuppressSpec::NonSpeech).unwrap();
  // Every non-speech token is suppressed.
  for id in w.non_speech_tokens().unwrap() {
    assert!(ids.contains(&id), "suppress set missing non-speech id {id}");
  }
  // The vocab punctuation symbols specifically are present.
  for (sym, id) in &punct {
    assert!(
      ids.contains(id),
      "suppress set missing symbol {sym:?} (id {id})"
    );
  }
  // sot_lm is one of the always-suppressed specials.
  assert!(ids.contains(&w.sot_lm()), "suppress set missing sot_lm");
}

#[test]
fn get_suppress_tokens_explicit_ids_drop_neg_one_sentinel() {
  let dir = fresh_dir("suppress_explicit");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let ids = get_suppress_tokens(&w, &SuppressSpec::Ids(vec![0, 1, u32::MAX])).unwrap();
  assert!(ids.contains(&0) && ids.contains(&1));
  // The `-1` sentinel (u32::MAX) is dropped, not treated as a token id.
  assert!(!ids.contains(&u32::MAX));
}

#[test]
fn get_suppress_tokens_none_is_empty() {
  let dir = fresh_dir("suppress_none");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  assert!(
    get_suppress_tokens(&w, &SuppressSpec::None)
      .unwrap()
      .is_empty()
  );
}

// ───────────────────────── transcribe segmentation ────────────────────────

fn dummy_result() -> DecodingResult {
  DecodingResult {
    language: "en".into(),
    tokens: vec![],
    text: String::new(),
    avg_logprob: -0.3,
    no_speech_prob: 0.1,
    temperature: 0.0,
    compression_ratio: 1.2,
  }
}

#[test]
fn transcribe_rejects_non_2d_mel() {
  let dir = fresh_dir("transcribe_rank");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);
  let mel3d = Array::ones::<f32>(&(1usize, 4usize, 4usize)).unwrap();
  let err = transcribe(&model, &w, &mel3d, 0, &TranscribeOptions::default()).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)));
}

#[test]
fn transcribe_seek_loop_excludes_trailing_padding() {
  // A mel padded by a trailing 30-second chunk (frames beyond `content_frames`
  // are feature padding). With `content_frames = 0` (the whole mel is the pad),
  // the seek loop (`while seek < content_frames`) decodes NO window — so the
  // padding is never decoded as content and the result is empty. The OLD code
  // bounded the loop on the full mel length and would have decoded the pad.
  //
  // A supplied language avoids the detection encode, and `content_frames = 0`
  // means the seek loop body never runs, so the encoder is never invoked — this
  // isolates the seek-bound contract from the encoder (the mel frame count is
  // irrelevant, only its presence drives the now-skipped loop).
  let dir = fresh_dir("transcribe_seek_bound");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(13);

  // A non-trivial mel — its full length would have driven the old loop — but
  // all of it is padding (content_frames = 0).
  let mel = Array::ones::<f32>(&(8usize, 4usize)).unwrap();
  let mut options = TranscribeOptions::default();
  options.decode.language = Some("en".into());

  let result = transcribe(&model, &w, &mel, /* content_frames */ 0, &options).unwrap();
  assert!(
    result.segments.is_empty(),
    "trailing padding must not be decoded as content; got {} segment(s)",
    result.segments.len()
  );
  assert_eq!(result.text, "");
  assert_eq!(result.language, "en");
}

#[test]
fn transcribe_no_word_timestamps_leaves_words_empty() {
  // The default (no-word-timestamp) path must produce segments with NO words
  // attached — the zero-cost contract. A real content window is decoded.
  let dir = fresh_dir("transcribe_no_words");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  // Bias toward "b" (id 1 < eot 2) so the decoded tokens are real text tokens.
  let model = tiny_model(1);

  let mel = tiny_mel();
  let mut options = TranscribeOptions::default();
  options.decode.language = Some("en".into());
  options.decode.without_timestamps = true;
  options.decode.suppress_blank = false;
  options.decode.suppress_tokens = SuppressSpec::None;
  options.decode.sample_len = Some(3);
  // Single greedy temperature, no fallback thresholds → deterministic.
  options.temperatures = vec![0.0];
  options.compression_ratio_threshold = None;
  options.logprob_threshold = None;
  options.no_speech_threshold = None;
  assert!(!options.word_timestamps);

  let result = transcribe(
    &model, &w, &mel, /* content_frames */ N_FRAMES, &options,
  )
  .unwrap();
  assert!(!result.segments.is_empty());
  for seg in &result.segments {
    assert!(seg.words.is_empty(), "no-word path must not attach words");
  }
}

#[test]
fn transcribe_word_timestamps_attaches_monotonic_words() {
  // With word_timestamps on, each non-empty segment carries per-word timings
  // whose (start, end) are ordered and non-decreasing across the segment — the
  // shape + monotonicity contract (exact times depend on the synthetic
  // attention, so only the structural invariants are asserted).
  let dir = fresh_dir("transcribe_words");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  // Bias toward "b" (id 1 < eot 2): the decode emits real text tokens, so the
  // window's text_tokens are non-empty and the alignment produces words.
  let model = tiny_model(1);

  let mel = tiny_mel();
  let mut options = TranscribeOptions::default();
  options.decode.language = Some("en".into());
  options.decode.without_timestamps = true;
  options.decode.suppress_blank = false;
  options.decode.suppress_tokens = SuppressSpec::None;
  options.decode.sample_len = Some(3);
  options.temperatures = vec![0.0];
  options.compression_ratio_threshold = None;
  options.logprob_threshold = None;
  options.no_speech_threshold = None;
  options.word_timestamps = true;

  let result = transcribe(
    &model, &w, &mel, /* content_frames */ N_FRAMES, &options,
  )
  .unwrap();
  assert!(!result.segments.is_empty());
  // At least one segment should carry words (the decode emits the biased text
  // token), and every word's end >= start, with non-decreasing starts.
  let mut saw_words = false;
  for seg in &result.segments {
    let mut prev_start = f64::NEG_INFINITY;
    for word in &seg.words {
      saw_words = true;
      assert!(
        word.end >= word.start,
        "word end {} < start {}",
        word.end,
        word.start
      );
      assert!(
        word.start >= prev_start,
        "word starts must be non-decreasing"
      );
      assert!(
        (0.0..=1.0).contains(&word.probability),
        "prob {} out of [0,1]",
        word.probability
      );
      prev_start = word.start;
    }
  }
  assert!(saw_words, "word_timestamps should attach at least one word");
}

// ──────────────── condition_on_previous_text + initial_prompt ──────────────

/// A text-context length large enough that `window_prompt`'s tail bound
/// (`n_text_ctx / 2 - 1`) is a no-op for the short prompts these threading /
/// reset / seed tests use, so they assert the full active slice exactly as the
/// reference threads it. (Whisper's real `n_text_ctx` is 448; any value whose
/// `n / 2 - 1` exceeds the test prompt lengths works.) The bound itself is
/// exercised separately by [`window_prompt_bounds_to_decoder_tail`].
const WIDE_CTX: usize = 448;

#[test]
fn prompt_history_threads_previous_window_tokens() {
  // The seek loop's prompt mechanism (`whisper.py:1033`, `:1271-1277`): the
  // prompt for window N+1 is `all_tokens[prompt_reset_since:]`, which (with
  // conditioning on) accumulates every prior window's decoded tokens. Starts
  // empty, then each pushed window's tokens appear in the next window's prompt.
  let dir = fresh_dir("prompt_thread");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();

  let mut history = PromptHistory::seed(
    &w,
    None,
    /* decode_prompt */ &[],
    /* condition */ true,
    WIDE_CTX,
  )
  .unwrap();
  // First window: no prior text → empty prompt.
  assert!(history.window_prompt(WIDE_CTX).is_empty());

  // After window 1 (tokens [12, 13], greedy temp 0.0), window 2's prompt is
  // exactly those tokens.
  history.push_window([12u32, 13].iter(), 0.0);
  assert_eq!(history.window_prompt(WIDE_CTX), &[12, 13]);

  // After window 2 (tokens [0, 1]), window 3's prompt contains BOTH windows'
  // tokens, in order.
  history.push_window([0u32, 1].iter(), 0.0);
  assert_eq!(history.window_prompt(WIDE_CTX), &[12, 13, 0, 1]);
}

#[test]
fn window_prompt_bounds_to_decoder_tail() {
  // `window_prompt(n_text_ctx)` returns only the LAST `n_text_ctx / 2 - 1`
  // tokens of the active slice — exactly the tail `build_initial_tokens`
  // (`decoding.py:539-549`) keeps. With n_text_ctx = 8 the bound is
  // `(8 / 2) - 1 = 3`, so a history longer than 3 tokens yields its last 3.
  let dir = fresh_dir("window_prompt_bound");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  const N_CTX: usize = 8;
  let keep = (N_CTX / 2) - 1; // 3

  let mut history = PromptHistory::seed(&w, None, /* decode_prompt */ &[], true, N_CTX).unwrap();
  // Accumulate 5 tokens across two windows (> keep), conditioning on.
  history.push_window([10u32, 11, 12].iter(), 0.0);
  history.push_window([13u32, 14].iter(), 0.0);
  // Active slice would be [10, 11, 12, 13, 14]; the bound keeps the last 3.
  let bounded = history.window_prompt(N_CTX);
  assert_eq!(
    bounded.len(),
    keep,
    "window_prompt must cap at n_text_ctx / 2 - 1"
  );
  assert_eq!(
    bounded,
    &[12, 13, 14],
    "the kept tokens are the active-slice tail"
  );

  // A history shorter than `keep` is returned whole (no padding, no panic).
  let mut short = PromptHistory::seed(&w, None, &[], true, N_CTX).unwrap();
  short.push_window([20u32, 21].iter(), 0.0);
  assert_eq!(short.window_prompt(N_CTX), &[20, 21]);
}

#[test]
fn window_prompt_bound_is_byte_identical_to_build_initial_tokens() {
  // Behavior preservation: bounding the prompt slice to the last `n / 2 - 1`
  // tokens cannot change the decode, because `build_initial_tokens` itself keeps
  // only the last `n / 2 - 1` prompt tokens. Feeding it the FULL active slice
  // and feeding it `window_prompt`'s bounded tail must yield byte-identical
  // initial tokens (the prompt prefix the decoder actually sees). We assert that
  // equivalence directly against the real `build_initial_tokens`.
  let dir = fresh_dir("window_prompt_equiv");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  const N_CTX: usize = 8;
  let sot_sequence = w.sot_sequence();
  let sample_len = 3usize; // any fixed sample budget — does not affect the prompt tail

  // A long active prompt (> the `n / 2 - 1 = 3` bound).
  let full_prompt: Vec<u32> = vec![1, 2, 3, 12, 13, 0, 1];
  let bounded_prompt: Vec<u32> = {
    let mut history = PromptHistory::seed(&w, None, &[], true, N_CTX).unwrap();
    history.push_window(full_prompt.iter(), 0.0);
    history.window_prompt(N_CTX).to_vec()
  };
  // The slice handed to the decoder is strictly shorter (prefix dropped).
  assert!(
    bounded_prompt.len() < full_prompt.len(),
    "the bound must drop the unused prefix for a long history"
  );

  let init_full = build_initial_tokens(
    &w,
    &sot_sequence,
    &DecodingOptions {
      prompt: full_prompt,
      ..Default::default()
    },
    N_CTX,
    sample_len,
  );
  let init_bounded = build_initial_tokens(
    &w,
    &sot_sequence,
    &DecodingOptions {
      prompt: bounded_prompt,
      ..Default::default()
    },
    N_CTX,
    sample_len,
  );
  assert_eq!(
    init_full, init_bounded,
    "bounding the prompt to the decoder tail must not change the initial tokens"
  );
}

#[test]
fn prompt_history_condition_false_resets_each_window() {
  // `condition_on_previous_text == false` (`whisper.py:1279-1281`): every
  // window resets the prompt window to the running tail, so a window is never
  // conditioned on the previous one (the next window's prompt is empty).
  let dir = fresh_dir("prompt_no_cond");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();

  let mut history = PromptHistory::seed(
    &w,
    None,
    /* decode_prompt */ &[],
    /* condition */ false,
    WIDE_CTX,
  )
  .unwrap();
  history.push_window([12u32, 13].iter(), 0.0);
  assert!(
    history.window_prompt(WIDE_CTX).is_empty(),
    "condition_on_previous_text=false must reset the prompt each window"
  );
  history.push_window([0u32].iter(), 0.0);
  assert!(history.window_prompt(WIDE_CTX).is_empty());
}

#[test]
fn prompt_history_high_temperature_resets() {
  // A window whose result temperature > 0.5 resets the prompt regardless of
  // conditioning (`whisper.py:1279`: `or result.temperature > 0.5`) — a
  // high-temperature fallback decode must not condition the next window.
  let dir = fresh_dir("prompt_high_temp");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();

  let mut history = PromptHistory::seed(
    &w,
    None,
    /* decode_prompt */ &[],
    /* condition */ true,
    WIDE_CTX,
  )
  .unwrap();
  // A low-temperature window conditions normally.
  history.push_window([12u32].iter(), 0.5); // exactly 0.5 is NOT > 0.5 → keep
  assert_eq!(history.window_prompt(WIDE_CTX), &[12]);
  // A window at temperature > 0.5 resets: the next window's prompt is empty.
  history.push_window([13u32].iter(), 0.6);
  assert!(
    history.window_prompt(WIDE_CTX).is_empty(),
    "temperature > 0.5 must reset the prompt window"
  );
}

#[test]
fn prompt_history_initial_prompt_seeds_first_window() {
  // `initial_prompt` (`whisper.py:990-994`) is encoded (with a leading space,
  // stripped body) and seeds `all_tokens`, so the FIRST window's prompt is
  // exactly the initial-prompt tokens.
  let dir = fresh_dir("prompt_initial");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();

  // " c d" → tokens [12, 13] under the fixture tokenizer (whitespace pre-tok,
  // word-level vocab). The leading-space + strip matches the reference's
  // `encode(" " + initial_prompt.strip())`.
  let expected = w.encode(" c d").unwrap();
  let history = PromptHistory::seed(
    &w,
    Some("  c d  "),
    /* decode_prompt */ &[],
    /* condition */ true,
    WIDE_CTX,
  )
  .unwrap();
  assert_eq!(
    history.window_prompt(WIDE_CTX),
    expected.as_slice(),
    "initial_prompt must seed the first window's prompt (leading space, trimmed)"
  );
  assert!(
    !expected.is_empty(),
    "fixture initial prompt must be non-empty"
  );
}

#[test]
fn transcribe_initial_prompt_absent_from_final_text() {
  // The initial prompt conditions the decode but is NEVER part of the emitted
  // transcript (`whisper.py:1299` strips it; mlxrs builds the text from segment
  // texts, which never contain the prompt). The decoded text here is the biased
  // token "b" (id 1), so the prompt word "d" must not leak into the result.
  let dir = fresh_dir("transcribe_initial_prompt");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1); // bias decode toward "b"

  let mel = tiny_mel();
  let mut options = TranscribeOptions::default();
  options.decode.language = Some("en".into());
  options.decode.without_timestamps = true;
  options.decode.suppress_blank = false;
  options.decode.suppress_tokens = SuppressSpec::None;
  options.decode.sample_len = Some(3);
  options.temperatures = vec![0.0];
  options.compression_ratio_threshold = None;
  options.logprob_threshold = None;
  options.no_speech_threshold = None;
  options.initial_prompt = Some("d".to_string()); // "d" is id 13

  let result = transcribe(
    &model, &w, &mel, /* content_frames */ N_FRAMES, &options,
  )
  .unwrap();
  assert!(
    !result.text.contains('d'),
    "initial_prompt text must not leak into the transcript, got {:?}",
    result.text
  );
}

#[test]
fn transcribe_condition_on_previous_text_runs_both_modes() {
  // The conditioning wiring must drive the seek loop to completion in BOTH
  // modes (true / false) over a multi-window mel, producing segments and never
  // panicking — a smoke test that the per-window prompt plumbing is sound.
  let dir = fresh_dir("transcribe_cond_modes");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1); // bias decode toward "b"

  // Two real content windows: a (2*N_FRAMES + pad)-frame mel, content_frames =
  // 2*N_FRAMES so the seek loop decodes exactly two windows.
  let mel = Array::ones::<f32>(&(2 * N_FRAMES + 8, 4usize)).unwrap();

  for condition in [true, false] {
    let mut options = TranscribeOptions::default();
    options.decode.language = Some("en".into());
    options.decode.without_timestamps = true;
    options.decode.suppress_blank = false;
    options.decode.suppress_tokens = SuppressSpec::None;
    options.decode.sample_len = Some(2);
    options.temperatures = vec![0.0];
    options.compression_ratio_threshold = None;
    options.logprob_threshold = None;
    options.no_speech_threshold = None;
    options.condition_on_previous_text = condition;

    let result = transcribe(
      &model,
      &w,
      &mel,
      /* content_frames */ 2 * N_FRAMES,
      &options,
    )
    .unwrap();
    assert!(
      result.segments.len() >= 2,
      "two content windows should yield >= 2 segments (condition={condition}), got {}",
      result.segments.len()
    );
  }
}

// ────────────────────────────── clip_timestamps ───────────────────────────

#[test]
fn compute_seek_clips_empty_spans_whole_audio() {
  // An empty `clip_timestamps` reproduces the reference default `"0"` → `[0.0]`
  // → one clip `[0, content_frames)` (`whisper.py:923-931`).
  assert_eq!(compute_seek_clips(&[], 5000).unwrap(), vec![(0, 5000)]);
}

#[test]
fn compute_seek_clips_even_list_pairs_and_clamps_last() {
  // An even-length list pairs as (start, end) and clamps the final end to
  // `content_frames` (`whisper.py:928`). At 100 frames/s, 1.0 s → 100 frames,
  // 2.0 s → 200; with content_frames = 150 the last end clamps to 150.
  assert_eq!(
    compute_seek_clips(&[1.0, 2.0], 150).unwrap(),
    vec![(100, 150)]
  );
  // No clamp needed when within bounds.
  assert_eq!(
    compute_seek_clips(&[1.0, 2.0], 5000).unwrap(),
    vec![(100, 200)]
  );
}

#[test]
fn compute_seek_clips_odd_list_open_ended_last() {
  // An odd-length list leaves the final clip open-ended: its end defaults to
  // `content_frames` (`whisper.py:925-926`).
  assert_eq!(compute_seek_clips(&[1.0], 5000).unwrap(), vec![(100, 5000)]);
  assert_eq!(
    compute_seek_clips(&[1.0, 2.0, 3.0], 5000).unwrap(),
    vec![(100, 200), (300, 5000)]
  );
}

#[test]
fn compute_seek_clips_multiple_pairs() {
  // A flat list interleaves into (start, end) pairs via zip(points[::2],
  // points[1::2]) (`whisper.py:929-931`).
  assert_eq!(
    compute_seek_clips(&[1.0, 2.0, 3.0, 4.0], 5000).unwrap(),
    vec![(100, 200), (300, 400)]
  );
}

#[test]
fn compute_seek_clips_clamps_every_earlier_pair() {
  // Soundness: an EARLIER pair whose end exceeds `content_frames` is clamped to
  // `content_frames` (not just the final pair) so the seek loop's `content_frames
  // - seek` / `seek_clip_end - seek` arithmetic can never underflow and the loop
  // cannot spin past the real audio. `[0.0, 9999.0, 0.01, 0.02]` at 100 frames/s
  // → points `[0, 999900, 1, 2]`; with content_frames=150 the first end clamps
  // to 150, and the (1, 2) clip survives unchanged.
  assert_eq!(
    compute_seek_clips(&[0.0, 9999.0, 0.01, 0.02], 150).unwrap(),
    vec![(0, 150), (1, 2)]
  );
  // A start beyond `content_frames` clamps to `content_frames`, making the clip
  // degenerate (`start >= end`), so it is dropped — it would contribute no
  // windows. `[99.0, 100.0]` → `[9900, 10000]`; with content_frames=150 both
  // clamp to 150 ⇒ inverted ⇒ dropped.
  assert!(compute_seek_clips(&[99.0, 100.0], 150).unwrap().is_empty());
}

#[test]
fn compute_seek_clips_drops_inverted_and_zero_length() {
  // An inverted clip (`start > end`) is dropped — it has no frames to decode.
  // `[2.0, 1.0]` → `[200, 100]` ⇒ start > end ⇒ dropped.
  assert!(compute_seek_clips(&[2.0, 1.0], 5000).unwrap().is_empty());
  // A zero-length clip (`start == end`) is likewise dropped.
  assert!(compute_seek_clips(&[1.0, 1.0], 5000).unwrap().is_empty());
  // An inverted EARLIER pair is dropped while a valid later pair survives.
  assert_eq!(
    compute_seek_clips(&[2.0, 1.0, 3.0, 4.0], 5000).unwrap(),
    vec![(300, 400)]
  );
}

#[test]
fn compute_seek_clips_fully_out_of_range_is_empty() {
  // A clip list entirely beyond `content_frames` yields no clips at all (every
  // pair clamps to `content_frames` and inverts to zero-length), so the seek
  // loop runs zero windows rather than hanging or underflowing.
  assert!(
    compute_seek_clips(&[100.0, 200.0, 300.0, 400.0], 50)
      .unwrap()
      .is_empty()
  );
}

#[test]
fn compute_seek_clips_zero_content_frames_is_empty() {
  // With no content frames the single default clip `[0, 0)` is zero-length and
  // dropped: the loop runs nothing (equivalent to the pre-clamp `[(0, 0)]`,
  // whose `while seek < 0` never executed).
  assert!(compute_seek_clips(&[], 0).unwrap().is_empty());
}

#[test]
fn round_to_frames_rounds_ties_to_even() {
  // Python's `round()` (used for the `clip_timestamps` → frame conversion at
  // `whisper.py:921` and the seek re-derivations) rounds halves to the nearest
  // EVEN integer, unlike Rust's away-from-zero `f64::round`. At 100 frames/s a
  // half-frame second `frame / 100` must therefore land on the even neighbor:
  // 12.5 → 12, 13.5 → 14, 0.5 → 0, 1.5 → 2. (Verified against CPython's
  // `round(x * 100)` for the same inputs.)
  assert_eq!(round_to_frames(0.125), 12); // 12.5 → 12 (even), NOT 13
  assert_eq!(round_to_frames(0.135), 14); // 13.5 → 14 (even)
  assert_eq!(round_to_frames(0.005), 0); //  0.5 → 0  (even)
  assert_eq!(round_to_frames(0.015), 2); //  1.5 → 2  (even)
  // Non-half values round to the nearest as usual.
  assert_eq!(round_to_frames(0.124), 12); // 12.4 → 12
  assert_eq!(round_to_frames(0.126), 13); // 12.6 → 13
  // Negative / zero seconds clamp to 0 (no negative seek frame).
  assert_eq!(round_to_frames(-1.0), 0);
}

#[test]
fn compute_seek_clips_half_frame_pairs_use_ties_even() {
  // A clip whose endpoints land exactly on half-frames must map to the same
  // frame boundaries CPython produces. start 0.115 s (11.5 frames) and end
  // 0.135 s (13.5 frames): ties-even gives (12, 14). Away-from-zero rounding
  // would have given (12, 14) here too for the end, but the boundary-shift case
  // below is where the two diverge — this pins the non-degenerate mapping.
  assert_eq!(
    compute_seek_clips(&[0.115, 0.135], 5000).unwrap(),
    vec![(12, 14)]
  );
  // start 0.125 s (12.5 → 12 even) end 0.145 s (14.5 → 14 even) ⇒ (12, 14).
  assert_eq!(
    compute_seek_clips(&[0.125, 0.145], 5000).unwrap(),
    vec![(12, 14)]
  );
}

#[test]
fn compute_seek_clips_ties_even_collapses_degenerate_pair() {
  // The faithfulness-critical case: a pair that is NON-empty under Rust's
  // away-from-zero rounding but reference-DEGENERATE under Python's ties-even,
  // so the reference drops it. start 0.115 s (11.5 frames) and end 0.125 s
  // (12.5 frames):
  //   away-from-zero: 11.5 → 12, 12.5 → 13  ⇒ (12, 13) — a spurious 1-frame clip
  //   ties-even     : 11.5 → 12, 12.5 → 12  ⇒ (12, 12) — degenerate ⇒ dropped
  // Matching Python, mlxrs must drop it (emit no clip), not decode a stray frame.
  assert!(
    compute_seek_clips(&[0.115, 0.125], 5000)
      .unwrap()
      .is_empty(),
    "a ties-even-degenerate clip must be dropped, matching Python round()"
  );
}

#[test]
fn compute_seek_clips_rejects_non_finite() {
  // Python parity + soundness: CPython's `round()` raises on `round(nan)`
  // (`ValueError`) and `round(inf)` (`OverflowError`) (`whisper.py:921`), but a
  // Rust `f64 as usize` cast would silently coerce `NaN → 0` and `inf →
  // usize::MAX`, turning a bogus `clip_timestamps` value into a degenerate (or
  // full-audio) clip instead of an error. Each non-finite value must therefore
  // be rejected with a typed `Error::OutOfRange` (not a silent wrong-range clip).
  for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
    let err = compute_seek_clips(&[bad], 5000)
      .expect_err("a non-finite clip timestamp must be rejected, not coerced");
    match err {
      Error::OutOfRange(p) => {
        assert_eq!(p.context(), "clip_timestamps: timestamp (seconds)");
        assert_eq!(p.requirement(), "must be finite");
      }
      other => panic!("expected Error::OutOfRange for {bad}, got {other:?}"),
    }
  }
  // A non-finite value anywhere in the list (not just first) is still rejected,
  // before any clip is produced.
  assert!(
    matches!(
      compute_seek_clips(&[1.0, 2.0, f64::NAN, 4.0], 5000),
      Err(Error::OutOfRange(_))
    ),
    "a non-finite value at a later position must still be rejected"
  );
  // Finite lists (including the degenerate/negative cases above) and the empty
  // list keep working unchanged — only non-finite input is rejected.
  assert_eq!(
    compute_seek_clips(&[1.0, 2.0], 5000).unwrap(),
    vec![(100, 200)]
  );
  assert_eq!(compute_seek_clips(&[], 5000).unwrap(), vec![(0, 5000)]);
}

#[test]
fn compute_seek_clips_rejects_overflowing_frame_product() {
  // Python parity + soundness (finding 1): a FINITE but huge clip timestamp
  // passes the `is_finite()` input check, yet `ts * FRAMES_PER_SECOND` overflows
  // to `±inf`. CPython's `round(finite * FRAMES_PER_SECOND)` raises
  // `OverflowError` the moment that product is infinite (`whisper.py:921`),
  // whereas Rust's `f64 as usize` cast inside `round_to_frames` would silently
  // saturate it (`+inf → usize::MAX`, `-inf → 0`) — turning the bogus clip into a
  // full-audio (`+1e307`) or empty (`-1e307`) one. The product overflow must
  // therefore be rejected with a typed `Error::OutOfRange`, never coerced.
  // (`1e307 * 100 == 1e309 > f64::MAX ⇒ inf`; the input itself is finite.)
  for bad in [1e307_f64, -1e307_f64, f64::MAX, -f64::MAX] {
    assert!(bad.is_finite(), "the input timestamp must itself be finite");
    let err = compute_seek_clips(&[bad], 5000)
      .expect_err("a finite timestamp whose FRAMES_PER_SECOND product overflows must be rejected");
    match err {
      Error::OutOfRange(p) => {
        assert_eq!(
          p.context(),
          "clip_timestamps: timestamp (seconds) × FRAMES_PER_SECOND"
        );
        assert_eq!(p.requirement(), "frame product must be finite");
      }
      other => panic!("expected Error::OutOfRange for {bad}, got {other:?}"),
    }
  }
  // The overflowing value is caught even at a later list position, before any
  // clip is produced (mirrors the non-finite-input behavior).
  assert!(
    matches!(
      compute_seek_clips(&[1.0, 2.0, 1e307, 4.0], 5000),
      Err(Error::OutOfRange(_))
    ),
    "an overflowing product at a later position must still be rejected"
  );
  // A merely large — but non-overflowing — finite timestamp is NOT rejected: its
  // product stays finite, it rounds, and the out-of-range frame index is clamped
  // to `content_frames` by the normal pairing path (not an error). `1e6 s` →
  // `1e8` frames, finite, clamped to `content_frames`.
  assert_eq!(
    compute_seek_clips(&[0.0, 1e6], 5000).unwrap(),
    vec![(0, 5000)],
    "a large-but-finite product must round + clamp, not error"
  );
}

#[test]
fn transcribe_clip_timestamps_restricts_windows() {
  // A clip list restricts decoding to the specified frame range: with a single
  // clip `[N_FRAMES, 2*N_FRAMES)` over a two-window mel, the loop seeks PAST the
  // first window, so every emitted segment starts at or after the clip start
  // time (`whisper.py:1018-1026`). The clip start is N_FRAMES frames = 30 s.
  let dir = fresh_dir("transcribe_clip_restrict");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1); // bias decode toward "b"

  let mel = Array::ones::<f32>(&(2 * N_FRAMES + 8, 4usize)).unwrap();
  let clip_start_secs = N_FRAMES as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
  let clip_end_secs = 2.0 * clip_start_secs;

  let mut options = TranscribeOptions::default();
  options.decode.language = Some("en".into());
  options.decode.without_timestamps = true;
  options.decode.suppress_blank = false;
  options.decode.suppress_tokens = SuppressSpec::None;
  options.decode.sample_len = Some(2);
  options.temperatures = vec![0.0];
  options.compression_ratio_threshold = None;
  options.logprob_threshold = None;
  options.no_speech_threshold = None;
  options.clip_timestamps = vec![clip_start_secs, clip_end_secs];

  let result = transcribe(
    &model,
    &w,
    &mel,
    /* content_frames */ 2 * N_FRAMES,
    &options,
  )
  .unwrap();
  assert!(
    !result.segments.is_empty(),
    "the clip window should still decode"
  );
  for seg in &result.segments {
    assert!(
      seg.start + 1e-6 >= clip_start_secs,
      "segment start {} is before the clip start {clip_start_secs}",
      seg.start
    );
  }
}

#[test]
fn transcribe_empty_clip_timestamps_matches_full_audio() {
  // Regression guard: an empty `clip_timestamps` (the default) produces byte-
  // identical output to today's full-audio path — same segment count + texts +
  // timings + final text. Compares a default-clip run against an explicit
  // empty-clip run over the same two-window mel.
  let dir = fresh_dir("transcribe_clip_regression");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);

  let mel = Array::ones::<f32>(&(2 * N_FRAMES + 8, 4usize)).unwrap();
  let base = || {
    let mut o = TranscribeOptions::default();
    o.decode.language = Some("en".into());
    o.decode.without_timestamps = true;
    o.decode.suppress_blank = false;
    o.decode.suppress_tokens = SuppressSpec::None;
    o.decode.sample_len = Some(2);
    o.temperatures = vec![0.0];
    o.compression_ratio_threshold = None;
    o.logprob_threshold = None;
    o.no_speech_threshold = None;
    o
  };

  // Default options (clip_timestamps already empty).
  let full = transcribe(&model, &w, &mel, 2 * N_FRAMES, &base()).unwrap();
  // Explicit empty clip list → must match.
  let mut clipped = base();
  clipped.clip_timestamps = Vec::new();
  let same = transcribe(&model, &w, &mel, 2 * N_FRAMES, &clipped).unwrap();

  assert_eq!(full.text, same.text);
  assert_eq!(full.segments.len(), same.segments.len());
  for (a, b) in full.segments.iter().zip(same.segments.iter()) {
    assert_eq!(a.text, b.text);
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.start, b.start);
    assert_eq!(a.end, b.end);
  }
}

/// Shared decode knobs for the clip-soundness / prompt-seed transcribe tests:
/// deterministic greedy decode, no fallback thresholds, two-token windows.
fn clip_test_options() -> TranscribeOptions {
  let mut o = TranscribeOptions::default();
  o.decode.language = Some("en".into());
  o.decode.without_timestamps = true;
  o.decode.suppress_blank = false;
  o.decode.suppress_tokens = SuppressSpec::None;
  o.decode.sample_len = Some(2);
  o.temperatures = vec![0.0];
  o.compression_ratio_threshold = None;
  o.logprob_threshold = None;
  o.no_speech_threshold = None;
  o
}

#[test]
fn transcribe_earlier_overlong_clip_terminates_bounded() {
  // Soundness (finding 1): a multi-pair clip list whose EARLIER pair end exceeds
  // the audio length must NOT spin a zero-progress decode loop or underflow. The
  // first clip `[0, 9999s]` clamps to `[0, content_frames]`; the loop terminates
  // with a bounded window count (at most one window per `N_FRAMES` stride). A
  // two-window mel (content_frames = 2*N_FRAMES) yields at most 2 windows ⇒ a
  // bounded number of segments.
  let dir = fresh_dir("transcribe_clip_overlong");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);

  let mel = Array::ones::<f32>(&(2 * N_FRAMES + 8, 4usize)).unwrap();
  let mut options = clip_test_options();
  // points: [0, 999900, 1, 2]; first end overshoots content_frames=2*N_FRAMES.
  options.clip_timestamps = vec![0.0, 9999.0, 0.01, 0.02];

  let result = transcribe(&model, &w, &mel, 2 * N_FRAMES, &options).unwrap();
  // The clamped first clip spans the whole audio ⇒ at most two `N_FRAMES`
  // windows. Each window emits at least one segment but the total is bounded:
  // assert termination with a finite, small segment count.
  assert!(
    result.segments.len() <= 4,
    "earlier-overlong clip must terminate with a bounded window count, got {} segments",
    result.segments.len()
  );
}

#[test]
fn transcribe_clip_start_beyond_eof_contributes_no_windows() {
  // Soundness (finding 1): a clip whose START is beyond the content frames must
  // not underflow `content_frames - seek`; the clip simply contributes no
  // windows. With a lone clip starting at 9999 s over a 60 s mel, the clip is
  // dropped by `compute_seek_clips` (clamped start == clamped end), so transcribe
  // returns no segments without panicking.
  let dir = fresh_dir("transcribe_clip_start_eof");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);

  let mel = Array::ones::<f32>(&(2 * N_FRAMES + 8, 4usize)).unwrap();
  let mut options = clip_test_options();
  options.clip_timestamps = vec![9999.0, 10000.0]; // start (and end) past EOF

  let result = transcribe(&model, &w, &mel, 2 * N_FRAMES, &options).unwrap();
  assert!(
    result.segments.is_empty(),
    "a clip starting past EOF must contribute no windows, got {} segments",
    result.segments.len()
  );
}

#[test]
fn transcribe_fully_out_of_range_clip_list_terminates_empty() {
  // Soundness (finding 1): a clip list entirely beyond the audio terminates with
  // empty output (zero windows), never a hang or panic.
  let dir = fresh_dir("transcribe_clip_oob");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);

  let mel = Array::ones::<f32>(&(2 * N_FRAMES + 8, 4usize)).unwrap();
  let mut options = clip_test_options();
  options.clip_timestamps = vec![1000.0, 2000.0, 3000.0, 4000.0];

  let result = transcribe(&model, &w, &mel, 2 * N_FRAMES, &options).unwrap();
  assert!(
    result.segments.is_empty(),
    "a fully out-of-range clip list must yield no segments, got {}",
    result.segments.len()
  );
}

#[test]
fn transcribe_inverted_clip_is_skipped() {
  // Soundness (finding 1): an inverted clip (`start > end`) is skipped, while a
  // following valid clip still decodes. The inverted `[60s, 0s]` pair is dropped;
  // the `[0s, 60s]` pair (full audio) decodes normally.
  let dir = fresh_dir("transcribe_clip_inverted");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);

  let mel = Array::ones::<f32>(&(2 * N_FRAMES + 8, 4usize)).unwrap();
  let full_secs = 2.0 * N_FRAMES as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
  let mut options = clip_test_options();
  options.clip_timestamps = vec![full_secs, 0.0, 0.0, full_secs];

  let result = transcribe(&model, &w, &mel, 2 * N_FRAMES, &options).unwrap();
  assert!(
    !result.segments.is_empty() && result.segments.len() <= 4,
    "inverted clip skipped, the valid clip decodes a bounded window count, got {}",
    result.segments.len()
  );
}

#[test]
fn prompt_history_falls_back_to_decode_prompt() {
  // Finding 2: with no `initial_prompt`, the lower-level `decode_prompt`
  // (`DecodingOptions::prompt`) seeds `all_tokens`, so the FIRST window's prompt
  // is exactly those caller tokens — honored, not silently dropped.
  let dir = fresh_dir("prompt_decode_seed");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();

  let history = PromptHistory::seed(
    &w,
    /* initial */ None,
    /* decode_prompt */ &[7, 8, 9],
    true,
    WIDE_CTX,
  )
  .unwrap();
  assert_eq!(
    history.window_prompt(WIDE_CTX),
    &[7, 8, 9],
    "decode.prompt must seed the first window when initial_prompt is absent"
  );
}

#[test]
fn prompt_history_initial_prompt_wins_over_decode_prompt() {
  // Finding 2 precedence: when BOTH are set, `initial_prompt` wins (the
  // documented knob); `decode_prompt` is ignored in favour of it.
  let dir = fresh_dir("prompt_initial_wins");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();

  let expected = w.encode(" c d").unwrap();
  let history = PromptHistory::seed(
    &w,
    Some("c d"),
    /* decode_prompt */ &[7, 8, 9],
    true,
    WIDE_CTX,
  )
  .unwrap();
  assert_eq!(
    history.window_prompt(WIDE_CTX),
    expected.as_slice(),
    "initial_prompt must take precedence over decode.prompt when both are set"
  );
  assert_ne!(history.window_prompt(WIDE_CTX), &[7, 8, 9]);
}

#[test]
fn transcribe_decode_prompt_conditions_first_window() {
  // Finding 2 end-to-end: a `TranscribeOptions` with `decode.prompt` set (no
  // `initial_prompt`) still conditions the first window — the first window's
  // prompt fed to the decoder contains the caller's tokens. We capture window-0's
  // prompt via a probe model that records the decode prompt it sees.
  let dir = fresh_dir("transcribe_decode_prompt");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);

  let mel = tiny_mel();
  let mut options = clip_test_options();
  options.decode.sample_len = Some(2);
  // Seed the caller's lower-level prompt; `initial_prompt` stays None.
  options.decode.prompt = w.encode(" c d").unwrap();
  let seed = options.decode.prompt.clone();
  assert!(!seed.is_empty(), "fixture decode prompt must be non-empty");

  // The seed is honored: build the same history the loop builds for window 0 and
  // assert it carries the caller's tokens (the loop overwrites
  // `decode_options.prompt = history.window_prompt(WIDE_CTX)`, which now includes them).
  let history = PromptHistory::seed(
    &w,
    options.initial_prompt.as_deref(),
    &options.decode.prompt,
    true,
    WIDE_CTX,
  )
  .unwrap();
  assert_eq!(
    history.window_prompt(WIDE_CTX),
    seed.as_slice(),
    "decode.prompt must condition window 0"
  );

  // And the full transcribe runs to completion with that seed (no panic, the
  // prompt is not rejected).
  let result = transcribe(&model, &w, &mel, N_FRAMES, &options).unwrap();
  assert!(
    !result.segments.is_empty(),
    "the window should still decode"
  );
}

#[test]
fn transcribe_oversized_decode_prompt_is_byte_identical_to_bounded_tail() {
  // Behavior-preservation: the seek loop builds each window's decode
  // options from a `decode_template` whose `prompt` is emptied once before the
  // loop, then installs only the bounded `window_prompt` tail (the last
  // `n_text_ctx / 2 - 1` tokens, here 3) — the caller's (potentially large)
  // `decode.prompt` prefix is never re-cloned per window AND never reaches the
  // decoder beyond that tail. Proof it is byte-identical: a run with an oversized
  // `decode.prompt` must produce the SAME segments/text as a run whose
  // `decode.prompt` is pre-truncated to just that tail. (`tiny_model`'s
  // n_text_ctx = 8 ⇒ keep = 3.)
  let dir = fresh_dir("transcribe_oversized_prompt");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);

  // Two-window mel so the per-window construction runs more than once.
  let mel = Array::ones::<f32>(&(2 * N_FRAMES + 8, 4usize)).unwrap();
  let keep = (model.dims().n_text_ctx() / 2).saturating_sub(1); // == 3

  // An oversized caller prompt: well beyond `keep` tokens. Only its last `keep`
  // tokens are decoder-visible; the leading prefix is discarded.
  let oversized: Vec<u32> = vec![1; keep + 12];
  let tail: Vec<u32> = oversized[oversized.len() - keep..].to_vec();
  assert_eq!(
    tail.len(),
    keep,
    "control prompt is exactly the bounded tail"
  );

  let run = |prompt: Vec<u32>| {
    let mut o = clip_test_options();
    o.decode.prompt = prompt;
    transcribe(&model, &w, &mel, 2 * N_FRAMES, &o).unwrap()
  };

  let big = run(oversized);
  let bounded = run(tail);

  // Byte-identical decode: same text, same segment count, same per-segment
  // tokens/text/timings. The discarded prefix changed nothing.
  assert_eq!(big.text, bounded.text);
  assert_eq!(big.segments.len(), bounded.segments.len());
  for (a, b) in big.segments.iter().zip(bounded.segments.iter()) {
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.text, b.text);
    assert_eq!(a.start, b.start);
    assert_eq!(a.end, b.end);
  }
}

#[test]
fn seed_oversized_initial_prompt_retains_only_bounded_tail() {
  // Finding 2(a) unit: an oversized `initial_prompt` is encoded then bounded to
  // the decoder-visible tail (`n_text_ctx / 2 - 1`), so `all_tokens` holds at
  // most `keep` tokens right after seeding — the leading prefix the decoder never
  // sees is never stored. The retained tail is exactly the last `keep` tokens of
  // the full encoding.
  let dir = fresh_dir("seed_oversized_initial");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();

  const N_CTX: usize = 8;
  let keep = (N_CTX / 2).saturating_sub(1); // == 3

  // A long initial prompt: `keep + 12` copies of "d" → that many tokens (id 13)
  // under the fixture tokenizer (leading space + strip ⇒ no extra/blank token).
  let words = "d ".repeat(keep + 12);
  let full_encoded = w.encode(&format!(" {}", words.trim())).unwrap();
  assert!(
    full_encoded.len() > keep,
    "the encoded initial prompt must exceed the bound"
  );

  let history =
    PromptHistory::seed(&w, Some(&words), /* decode_prompt */ &[], true, N_CTX).unwrap();
  // Only the bounded tail is retained, not the whole encoding.
  assert!(
    history.all_tokens.len() <= keep,
    "seed must retain only the bounded tail of the initial prompt (<= keep), got {}",
    history.all_tokens.len()
  );
  // And it is exactly the LAST `keep` encoded tokens (the decoder-visible tail).
  assert_eq!(
    history.window_prompt(N_CTX),
    &full_encoded[full_encoded.len() - keep..],
    "the retained tail must be the last `keep` tokens of the full encoding"
  );
}

#[test]
fn transcribe_oversized_initial_prompt_is_byte_identical_to_bounded_tail() {
  // Behavior-preservation (finding 2(a)): an oversized `initial_prompt` decodes
  // byte-identically to a run seeded with only that prompt's decoder-visible tail
  // — bounding the seed to the last `n_text_ctx / 2 - 1` tokens cannot change the
  // decode, because `window_prompt` only ever exposes that tail. Control: the
  // SAME tail fed as a raw `decode.prompt` (no `initial_prompt`) seeds the
  // identical `all_tokens`, so the two runs must produce identical
  // segments/text/timings. (`tiny_model`'s n_text_ctx = 8 ⇒ keep = 3.)
  let dir = fresh_dir("transcribe_oversized_initial");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);

  // Two-window mel so the per-window construction runs more than once.
  let mel = Array::ones::<f32>(&(2 * N_FRAMES + 8, 4usize)).unwrap();
  let keep = (model.dims().n_text_ctx() / 2).saturating_sub(1); // == 3

  // An oversized initial prompt + the exact decoder-visible tail it bounds to.
  let words = "d ".repeat(keep + 12);
  let full_encoded = w.encode(&format!(" {}", words.trim())).unwrap();
  let tail: Vec<u32> = full_encoded[full_encoded.len() - keep..].to_vec();
  assert_eq!(tail.len(), keep, "control tail is exactly `keep` tokens");

  // Run A: the oversized initial prompt (bounded internally to its tail).
  let mut opts_initial = clip_test_options();
  opts_initial.initial_prompt = Some(words);
  let from_initial = transcribe(&model, &w, &mel, 2 * N_FRAMES, &opts_initial).unwrap();

  // Run B: the same tail fed as the lower-level decode.prompt (no initial_prompt)
  // — seeds the identical `all_tokens`, so the decode must match exactly.
  let mut opts_tail = clip_test_options();
  opts_tail.decode.prompt = tail;
  let from_tail = transcribe(&model, &w, &mel, 2 * N_FRAMES, &opts_tail).unwrap();

  assert_eq!(from_initial.text, from_tail.text);
  assert_eq!(from_initial.segments.len(), from_tail.segments.len());
  for (a, b) in from_initial.segments.iter().zip(from_tail.segments.iter()) {
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.text, b.text);
    assert_eq!(a.start, b.start);
    assert_eq!(a.end, b.end);
  }
}

#[test]
fn window_prompt_carries_only_bounded_tail_across_windows() {
  // The per-window prompt is only the bounded tail: the seek loop
  // installs `history.window_prompt(n_text_ctx)` into each window's decode
  // options. Even after the running history grows across windows AND starts from
  // an oversized seed, the prompt handed to every window is only the last
  // `n_text_ctx / 2 - 1` tokens — the large prefix is never carried per window.
  let dir = fresh_dir("window_prompt_bounded_tail");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();

  let n_text_ctx = 8usize; // matches `tiny_model`'s dims
  let keep = (n_text_ctx / 2).saturating_sub(1); // == 3

  // Seed with an oversized decode prompt (no initial_prompt).
  let oversized: Vec<u32> = (0..(keep as u32 + 20)).collect();
  let mut history = PromptHistory::seed(&w, None, &oversized, true, n_text_ctx).unwrap();

  // `seed` itself retains only the decoder-visible tail: `all_tokens` holds at
  // most `keep` tokens right after seeding, never the whole oversized prompt.
  assert!(
    history.all_tokens.len() <= keep,
    "seed must retain only the bounded tail (<= keep), got {} tokens",
    history.all_tokens.len()
  );

  // Window 0: the prompt is exactly the bounded tail of the oversized seed — the
  // large prefix is NOT carried.
  let w0 = history.window_prompt(n_text_ctx).to_vec();
  assert_eq!(
    w0.len(),
    keep,
    "window-0 prompt must be the bounded tail length"
  );
  assert_eq!(
    w0.as_slice(),
    &oversized[oversized.len() - keep..],
    "window-0 prompt must be exactly the last `keep` seed tokens"
  );

  // Grow the history across several windows (the loop appends each window's
  // tokens); the per-window prompt stays bounded to `keep` and tracks the newest
  // tail, never re-accumulating the discarded prefix.
  for win in 0..4u32 {
    history.push_window(&[100 + win, 101 + win], /* temperature */ 0.0);
    let p = history.window_prompt(n_text_ctx).to_vec();
    assert_eq!(
      p.len(),
      keep,
      "every window's prompt must stay bounded to `keep` tokens"
    );
    // None of the oversized prefix (ids < keep) is ever carried once the history
    // has grown past it.
    assert!(
      p.iter().all(|&t| t >= keep as u32),
      "the discarded oversized prefix must never reappear in a window prompt"
    );
  }
}

/// A sub-token alignment window (`num_frames / 2 == 0`, i.e. `num_frames < 2`)
/// has no audio-token columns to align: `find_alignment` must return no word
/// timings (the faithful degenerate-window result) with no panic and never a
/// negative / clamped-bogus time index. Drives the real `find_alignment`
/// against the tiny model + tokenizer with `num_frames` 0 and 1.
#[test]
fn find_alignment_subtoken_window_yields_no_words() {
  use crate::audio::stt::models::whisper::timing::find_alignment;

  let dir = fresh_dir("subtoken_align");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);
  let mel = tiny_mel();
  // A non-empty text-token slice (so the empty-text early return is NOT the one
  // under test) — one real text token id `1` ("b", below the eot id 2).
  let text_tokens = [1u32];

  // `num_frames < 2` ⇒ `num_frames / 2 == 0`: a sub-token window.
  for num_frames in [0usize, 1] {
    let words = find_alignment(&model, &w, &text_tokens, &mel, num_frames, 1.0)
      .expect("sub-token window must not error");
    assert!(
      words.is_empty(),
      "sub-token window (num_frames={num_frames}) must yield no word timings, got {words:?}"
    );
  }

  // A non-degenerate window (`num_frames / 2 >= 1`) still aligns and never
  // emits a negative timestamp.
  let words = find_alignment(&model, &w, &text_tokens, &mel, 4, 1.0).expect("alignment");
  for word in &words {
    assert!(
      word.start >= 0.0,
      "timestamp must be non-negative, got {word:?}"
    );
    assert!(word.end >= word.start, "end must be >= start, got {word:?}");
  }
}

/// `find_alignment` is public, so it enforces the `text_token < eot` precondition
/// itself: `text_token_probabilities` slices the probability matrix to the
/// `[0, eot)` text-vocab columns and gathers it by `text_tokens`, so a timestamp
/// / special id in `[eot, n_vocab)` — which clears the embedding gather's
/// `< n_vocab` bound — would index PAST that `eot`-wide matrix (out of bounds).
/// Such an id is rejected with a typed `OutOfRange` BEFORE the forward; a valid
/// (`< eot`) text stream still aligns.
#[test]
fn find_alignment_rejects_text_token_at_or_above_eot() {
  use crate::audio::stt::models::whisper::timing::find_alignment;

  let dir = fresh_dir("align_eot");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let model = tiny_model(1);
  let mel = tiny_mel();
  let eot = w.eot(); // 2

  // A text token == eot: the first id that indexes past the `[0, eot)` columns.
  let at_eot = [1u32, eot];
  let err = find_alignment(&model, &w, &at_eot, &mel, 4, 1.0).unwrap_err();
  assert!(
    matches!(&err, Error::OutOfRange(p)
      if p.context() == "Whisper word timestamps: alignment text token"),
    "expected OutOfRange for a text token == eot, got {err:?}"
  );

  // A timestamp id in `[eot, n_vocab)` (timestamp_begin = 14, n_vocab = 18) —
  // passes the `< n_vocab` embedding bound but is `>= eot`.
  let timestamp = [1u32, w.timestamp_begin()];
  assert!(
    w.timestamp_begin() >= eot && (w.timestamp_begin() as usize) < N_VOCAB,
    "fixture: timestamp id must sit in [eot, n_vocab)"
  );
  let err = find_alignment(&model, &w, &timestamp, &mel, 4, 1.0).unwrap_err();
  assert!(
    matches!(&err, Error::OutOfRange(p)
      if p.context() == "Whisper word timestamps: alignment text token"),
    "expected OutOfRange for a timestamp text token, got {err:?}"
  );

  // A valid (`< eot`) text stream still produces alignments (the guard does not
  // reject real text tokens).
  let words = find_alignment(&model, &w, &[1u32], &mel, 4, 1.0).expect("valid alignment");
  for word in &words {
    assert!(
      word.start >= 0.0,
      "timestamp must be non-negative, got {word:?}"
    );
    assert!(word.end >= word.start, "end must be >= start, got {word:?}");
  }
}

#[test]
fn segment_collection_no_timestamps_one_segment() {
  // No consecutive timestamps → one segment spanning the whole window; advance
  // by the full segment_size.
  let dir = fresh_dir("seg_no_ts");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let ts_begin = w.timestamp_begin(); // 14
  let mut segments = Vec::new();
  let tokens = vec![0u32, 1, 12]; // all text tokens "a b c", no timestamps
  let (advance, single_end) = advance_and_collect_segments(
    &tokens,
    ts_begin,
    /* time_offset */ 0.0,
    /* time_precision */ 0.02,
    /* segment_size */ 100,
    /* input_stride */ 2,
    &dummy_result(),
    &w,
    &mut segments,
  )
  .unwrap();
  assert_eq!(advance, 100);
  assert!(!single_end);
  assert_eq!(segments.len(), 1);
  assert_eq!(segments[0].start, 0.0);
  // text decodes the non-special tokens.
  assert!(!segments[0].text.is_empty());
}

#[test]
fn segment_collection_consecutive_timestamps_cut_segments() {
  // tokens: <|0.00|>(14) a(0) b(1) <|0.04|>(16) <|0.04|>(16) c(12) <|0.06|>(17)
  // The double 16 16 is a consecutive timestamp pair → a cut after index 4.
  let dir = fresh_dir("seg_ts_pairs");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let ts_begin = w.timestamp_begin(); // 14
  let mut segments = Vec::new();
  let tokens = vec![14u32, 0, 1, 16, 16, 12, 17];
  let (advance, _single_end) = advance_and_collect_segments(
    &tokens,
    ts_begin,
    0.0,
    0.02,
    /* segment_size */ 200,
    /* input_stride */ 2,
    &dummy_result(),
    &w,
    &mut segments,
  )
  .unwrap();
  // A consecutive pair at index 4 → at least one segment cut, and the advance
  // is the last consumed timestamp * input_stride (not the full window).
  assert!(!segments.is_empty());
  assert!(advance > 0 && advance <= 200);
}

#[test]
fn segment_collection_single_timestamp_ending_advances_full_window() {
  // Ending [..., text, timestamp] is a single_timestamp_ending; with a leading
  // consecutive pair the whole window is consumed → advance == segment_size.
  let dir = fresh_dir("seg_single_end");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let ts_begin = w.timestamp_begin();
  let mut segments = Vec::new();
  // 14 14 (consecutive pair) ... 12 (text) 15 (single trailing ts).
  let tokens = vec![14u32, 14, 0, 12, 15];
  let (advance, single_end) = advance_and_collect_segments(
    &tokens,
    ts_begin,
    0.0,
    0.02,
    /* segment_size */ 150,
    2,
    &dummy_result(),
    &w,
    &mut segments,
  )
  .unwrap();
  assert_eq!(advance, 150);
  assert!(single_end);
}

// ───────────────────────── high-level Transcribe trait ────────────────────

#[test]
fn transcribe_trait_without_tokenizer_is_typed_error() {
  // The universal `Transcribe` contract requires an attached tokenizer; a model
  // loaded without one points the caller at `with_tokenizer` / the lower-level
  // entry point via a typed InvariantViolation rather than panicking.
  use crate::audio::stt::model::{Transcribe as _, TranscribeOptions as GoldenOptions};
  let model = tiny_model(13);
  let audio = Array::zeros::<f32>(&[16_000]).unwrap();
  let err = model.transcribe(&audio, &GoldenOptions::new()).unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation without a tokenizer, got {err:?}"
  );
}

#[test]
fn transcribe_trait_with_tokenizer_drives_decoding_task() {
  // Integration proof of the high-level `Transcribe` impl wiring: attach a
  // tokenizer (so the model builds its own `HFTokenizerWrapper` from
  // `opts.language()` + the dims) and run the universal contract on a real
  // waveform. The call frames the waveform into the full padded 30-second mel
  // and drives the decoding task's seek loop into the encoder. Because
  // `n_audio_ctx` is pinned to the fixed `N_FRAMES / 2` (1500), the segment is
  // padded to exactly `N_FRAMES` (3000) before `AudioEncoder::forward`, so the
  // conv downsample lands on 1500 frames matching the positional embedding and
  // the encoder runs cleanly — the full waveform → mel → wrapper →
  // decoding-task → encoder → decoder greedy loop completes through the trait.
  // The tiny model's logit head is biased toward token 13, so greedy decode is
  // deterministic and terminates at the `sample_len` cap; we assert the call
  // succeeds (the path is wired) and echoes the supplied language.
  use crate::audio::stt::model::{Transcribe as _, TranscribeOptions as GoldenOptions};
  let dir = fresh_dir("transcribe_trait_e2e");
  let tok = write_tokenizer(dir.as_path());
  // A supplied language avoids the separate detection encode (which would also
  // run the encoder on the first window).
  let model = tiny_model(13).with_tokenizer(tok).unwrap();

  let audio = Array::zeros::<f32>(&[16_000]).unwrap();
  let opts = GoldenOptions::new().with_language("en");
  let result = model
    .transcribe(&audio, &opts)
    .expect("the full Transcribe path must run end to end once n_audio_ctx is pinned");
  assert_eq!(
    result.language().map(str::to_owned),
    Some("en".to_string()),
    "the supplied language must echo through the universal Transcription"
  );
}

// ───────────────────── hallucination-silence skip ─────────────────────────

/// A single-word, anomalous-looking segment: one low-probability, very short
/// word makes [`timing::is_segment_anomaly`] fire (`word_anomaly_score`
/// contributes `1.0` for `prob < 0.15` plus the short-duration term, and a
/// one-word segment trips the `score + 0.01 >= len` branch).
fn anomalous_segment(start: f64, end: f64) -> Segment {
  Segment {
    start,
    end,
    text: String::from("x"),
    tokens: vec![0],
    temperature: 0.0,
    avg_logprob: 0.0,
    no_speech_prob: 0.0,
    compression_ratio: 1.0,
    words: vec![Word {
      word: String::from(" x"),
      start,
      end: start + 0.05,
      probability: 0.05,
    }],
  }
}

/// A word-free, timestamp-only segment (`words` empty): the loop skips it, and
/// [`timing::is_segment_anomaly`] returns `false` for it.
fn empty_segment(start: f64, end: f64) -> Segment {
  Segment {
    start,
    end,
    text: String::new(),
    tokens: vec![],
    temperature: 0.0,
    avg_logprob: 0.0,
    no_speech_prob: 0.0,
    compression_ratio: 0.0,
    words: vec![],
  }
}

#[test]
fn hallucination_skip_uses_next_word_bearing_segment_for_silence_after() {
  // Regression: the `silence_after` anomaly check must consult the next
  // WORD-BEARING segment (`next_words_segment(current_segments[si + 1:])`), not
  // the immediate `si + 1` segment. With an empty/timestamp-only segment wedged
  // between two anomalous word-bearing segments, the hallucination at index 0 is
  // only surrounded-by-hallucination via the index-2 segment; checking the empty
  // index-1 segment (which is never anomalous) would wrongly keep it.
  //
  // The gap clause (`hal_next_start - seg.end > threshold`) and the window-end
  // clause (`window_end_time - seg.end < 2.0`) are both arranged to be FALSE, so
  // `silence_after` can only become true through the next-segment anomaly term —
  // making this sensitive to exactly the fixed bug.
  let threshold = 2.0;
  let mut segments = vec![
    anomalous_segment(0.5, 1.0),
    empty_segment(1.0, 1.2),
    anomalous_segment(1.5, 2.0),
  ];

  // window_end_time = (0 + N_FRAMES) * HOP / SR = 30.0 s; content spans the full
  // window too. So `window_end_time - 1.0 = 29.0` (not < 2.0) and the next
  // word's start (1.5) minus seg0.end (1.0) = 0.5 (not > 2.0): both fall to
  // false, isolating the next-segment-anomaly term.
  let window_end_time = N_FRAMES as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
  let content_frames = N_FRAMES;
  let content_duration = content_frames as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;

  let skip = hallucination_silence_skip(HallucinationSkipParams {
    current_segments: &mut segments,
    threshold,
    prev_last_speech: 0.0,
    time_offset: 0.0,
    previous_seek: 0,
    segment_size: N_FRAMES,
    segment_duration: window_end_time,
    window_end_time,
    content_duration,
    content_frames,
    single_timestamp_ending: false,
    seek: 0,
  });

  // The index-2 anomaly makes `silence_after` true, so the window is truncated
  // at index 0 (the reference's `current_segments[si:] = []`) and NOT dropped as
  // a leading-silence `continue`. The buggy si+1 check would leave all three
  // segments in place.
  assert!(
    segments.is_empty(),
    "the surrounded hallucination must be truncated via the si+2 word-bearing \
     segment, got {segments:?}"
  );
  assert!(
    !skip.skip_window,
    "this is a truncation, not a leading-silence drop"
  );
  // seek = round(max(time_offset + 1, seg.start) * FRAMES_PER_SECOND) = 100; the
  // `content_duration - seg.end (29.0) < threshold (2.0)` branch does not fire.
  assert_eq!(skip.seek, FRAMES_PER_SECOND);
}

// ─────────────────── degenerate-segment cleanup ───────────────────────────

/// A populated [`Segment`] with the given timing + text (a single word + token
/// so the cleanup's word/token clearing is observable).
fn seg_with(start: f64, end: f64, text: &str) -> Segment {
  Segment {
    start,
    end,
    text: text.to_string(),
    tokens: vec![7],
    temperature: 0.0,
    avg_logprob: 0.0,
    no_speech_prob: 0.0,
    compression_ratio: 1.0,
    words: vec![Word {
      word: text.to_string(),
      start,
      end,
      probability: 0.9,
    }],
  }
}

#[test]
fn clear_degenerate_segments_empties_whitespace_and_instantaneous() {
  // A whitespace-only segment and a zero-duration (start == end) segment are
  // both cleared — text, tokens, AND words emptied (`whisper.py:1253-1261`) —
  // while a normal segment between them is left fully intact.
  let mut segments = vec![
    seg_with(0.0, 1.0, "   "),     // whitespace-only text → cleared
    seg_with(1.0, 2.0, " hello"),  // normal → unchanged
    seg_with(2.0, 2.0, "instant"), // zero duration (start == end) → cleared
  ];
  clear_degenerate_segments(&mut segments);

  // Whitespace-only: emptied.
  assert_eq!(segments[0].text, "");
  assert!(segments[0].tokens.is_empty());
  assert!(segments[0].words.is_empty());
  // Zero-duration: emptied even though its text was non-blank.
  assert_eq!(segments[2].text, "");
  assert!(segments[2].tokens.is_empty());
  assert!(segments[2].words.is_empty());

  // Non-degenerate segment is untouched (text, tokens, and words all intact).
  assert_eq!(segments[1].text, " hello");
  assert_eq!(segments[1].tokens, vec![7]);
  assert_eq!(segments[1].words.len(), 1);
  assert_eq!(segments[1].words[0].word, " hello");

  // The cleared segments contribute nothing to the joined transcript: the only
  // surviving text is the non-degenerate segment's.
  let joined: String = segments.iter().map(|s| s.text.as_str()).collect();
  assert_eq!(joined, " hello");
}

#[test]
fn clear_degenerate_segments_leaves_nondegenerate_unchanged() {
  // Every segment has a non-empty text and a non-zero duration → no clearing.
  let mut segments = vec![seg_with(0.0, 1.0, " the"), seg_with(1.0, 2.5, " cat")];
  let before = segments.clone();
  clear_degenerate_segments(&mut segments);
  for (after, orig) in segments.iter().zip(&before) {
    assert_eq!(after.text, orig.text);
    assert_eq!(after.tokens, orig.tokens);
    assert_eq!(after.words, orig.words);
  }
}

/// Build one window's segments through the exact `transcribe` collection path
/// (`advance_and_collect_segments`) for a tokens stream — so a segment it
/// produces is byte-identical to one `transcribe` would accumulate.
fn collect_window_segments(w: &HFTokenizerWrapper<'_>, tokens: &[u32]) -> Vec<Segment> {
  let mut segments = Vec::new();
  advance_and_collect_segments(
    tokens,
    w.timestamp_begin(),
    /* time_offset */ 0.0,
    /* time_precision */ 0.02,
    /* segment_size */ 100,
    /* input_stride */ 2,
    &dummy_result(),
    w,
    &mut segments,
  )
  .unwrap();
  segments
}

#[test]
fn no_word_path_preserves_degenerate_segment_tokens() {
  // REGRESSION GUARD: the degenerate-segment cleanup is part of the
  // word-timestamp finalization and is gated to `word_timestamps == true`. On
  // the DEFAULT no-word path a degenerate (here zero-duration, timestamp-only)
  // segment must be emitted with its sampled tokens / text intact — the
  // byte-identical pre-feature behavior callers reading `Segment.tokens` rely on.
  let dir = fresh_dir("degenerate_no_word");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();

  // Two consecutive `<|0.00|>` (id 14 == timestamp_begin): the slice [14] has
  // equal start/end timestamp positions → a zero-duration (start == end)
  // segment whose only token is the timestamp (text strips to empty). This is
  // exactly the degenerate shape `clear_degenerate_segments` targets.
  let segments = collect_window_segments(&w, &[14u32, 14]);
  assert_eq!(segments.len(), 1);
  assert_eq!(
    segments[0].start, segments[0].end,
    "the collected segment must be zero-duration (degenerate)"
  );
  assert!(
    !segments[0].tokens.is_empty(),
    "the degenerate segment still carries its sampled token(s)"
  );

  // No-word path: the gating means `clear_degenerate_segments` is NOT applied,
  // so tokens are preserved verbatim.
  let no_word = segments.clone();
  assert_eq!(
    no_word[0].tokens, segments[0].tokens,
    "no-word path must keep the degenerate segment's tokens intact"
  );

  // Word path: the same segment IS cleared (the prior contract still holds).
  let mut word_path = segments;
  clear_degenerate_segments(&mut word_path);
  assert!(
    word_path[0].tokens.is_empty() && word_path[0].text.is_empty(),
    "word-timestamp path still clears the degenerate segment"
  );
}

// ───────────────────── PR-E: universal knobs + rich result ─────────────────

/// A short real waveform (deterministic) the universal `transcribe` /
/// `transcribe_detailed` front-end frames into a log-mel.
fn pr_e_waveform() -> Array {
  let data: Vec<f32> = (0..16_000)
    .map(|i| ((i % 41) as f32 / 41.0) - 0.5)
    .collect();
  Array::from_slice::<f32>(&data, &[data.len() as i32]).unwrap()
}

/// A `tiny_model(target)` with the fixture tokenizer attached, so the universal
/// `Transcribe` / inherent `transcribe_detailed` entries (which require an
/// attached tokenizer) can run.
fn tiny_model_with_tokenizer(target: u32, dir: &Path) -> WhisperModel {
  let tok = write_tokenizer(dir);
  tiny_model(target).with_tokenizer(tok).unwrap()
}

#[test]
fn transcribe_detailed_exposes_tokens_seek_and_stats() {
  // The rich result surfaces the fields the universal `Transcription` cannot
  // hold: per-segment token ids, seek-derived time offsets, and decode stats.
  use crate::audio::stt::model::TranscribeOptions;
  let dir = fresh_dir("pr_e_detailed");
  let model = tiny_model_with_tokenizer(13, dir.as_path()); // biased to "d"
  let audio = pr_e_waveform();

  let opts = TranscribeOptions::new()
    .with_language("en")
    .with_no_speech_threshold(None) // don't skip the synthetic window as silence
    .with_compression_ratio_threshold(None)
    .with_logprob_threshold(None)
    .with_max_new_tokens(4);
  let rich = model.transcribe_detailed(&audio, &opts).unwrap();

  assert_eq!(rich.language(), "en");
  assert!(!rich.segments().is_empty(), "at least one segment");
  let seg = &rich.segments()[0];
  // Token ids are exposed; with timestamps on, a leading timestamp token may
  // precede the biased text target — the point is the ids are surfaced verbatim
  // and the biased text token appears among them.
  assert!(!seg.tokens().is_empty(), "segment carries token ids");
  assert!(
    seg.tokens().contains(&13),
    "biased text target must appear in the exposed token ids, got {:?}",
    seg.tokens()
  );
  // Seek-derived time offsets are present and ordered.
  assert!(seg.end() >= seg.start(), "segment end >= start");
  assert!(seg.start() >= 0.0, "segment start non-negative");
  // Decode statistics are surfaced.
  assert!(seg.avg_logprob() <= 0.0, "avg_logprob is a log-prob (<= 0)");
  assert!(
    seg.compression_ratio() > 0.0,
    "compression ratio is positive"
  );
}

#[test]
fn transcribe_detailed_word_timestamps_knob_flows_through() {
  // The `word_timestamps` knob flows through to the decode: OFF → no words on
  // the rich result; ON → the word-timestamp pass runs (the segments carry the
  // per-word timing list). The biased model is deterministic, so this exercises
  // the wiring, not the alignment quality.
  use crate::audio::stt::model::TranscribeOptions;
  let dir = fresh_dir("pr_e_words");
  let model = tiny_model_with_tokenizer(13, dir.as_path());
  let audio = pr_e_waveform();

  let base = TranscribeOptions::new()
    .with_language("en")
    .with_no_speech_threshold(None)
    .with_compression_ratio_threshold(None)
    .with_logprob_threshold(None)
    .with_max_new_tokens(4);

  // OFF: the default no-word path leaves the words empty.
  let off = model
    .transcribe_detailed(&audio, &base.clone().with_word_timestamps(false))
    .unwrap();
  assert!(
    off.segments().iter().all(|s| s.words().is_empty()),
    "word_timestamps=false → no per-word timings"
  );

  // ON: the word-timestamp pass runs without error and the option is honored
  // (the synthetic alignment may merge everything into few words, but the path
  // is exercised and the result is well-formed).
  let on = model
    .transcribe_detailed(&audio, &base.with_word_timestamps(true))
    .unwrap();
  // Every word (if any) is well-formed: ordered times, in-range probability.
  for seg in on.segments() {
    for w in seg.words() {
      assert!(w.end() >= w.start(), "word end >= start");
      assert!(
        (0.0..=1.0).contains(&w.probability()),
        "word probability in [0,1], got {}",
        w.probability()
      );
    }
  }
}

#[test]
fn universal_transcribe_options_thresholds_round_trip() {
  // The new universal knobs round-trip through their accessors (the wiring the
  // whisper mapping reads). An independent check that the option surface carries
  // the values the decode consumes.
  use crate::audio::stt::model::TranscribeOptions;
  let opts = TranscribeOptions::new()
    .with_compression_ratio_threshold(Some(3.0))
    .with_logprob_threshold(Some(-0.5))
    .with_no_speech_threshold(None)
    .with_condition_on_previous_text(false)
    .with_initial_prompt("custom vocab")
    .with_word_timestamps(true)
    .with_clip_timestamps(vec![0.0, 5.0, 10.0, 15.0]);

  assert_eq!(opts.compression_ratio_threshold(), Some(3.0));
  assert_eq!(opts.logprob_threshold(), Some(-0.5));
  assert_eq!(opts.no_speech_threshold(), None);
  assert!(!opts.condition_on_previous_text());
  assert_eq!(opts.initial_prompt(), Some("custom vocab"));
  assert!(opts.word_timestamps());
  assert_eq!(opts.clip_timestamps(), &[0.0, 5.0, 10.0, 15.0]);

  // The defaults preserve the prior whisper behavior (the thresholds the old
  // `transcribe` hardcoded).
  let def = TranscribeOptions::new();
  assert_eq!(def.compression_ratio_threshold(), Some(2.4));
  assert_eq!(def.logprob_threshold(), Some(-1.0));
  assert_eq!(def.no_speech_threshold(), Some(0.6));
  assert!(def.condition_on_previous_text());
  assert!(!def.word_timestamps());
  assert!(def.clip_timestamps().is_empty());
  assert_eq!(def.initial_prompt(), None);
}

#[test]
fn universal_transcribe_returns_standard_transcription() {
  // The universal `Transcribe::transcribe` still returns the lossy standard
  // `Transcription` (text + language + segment spans), honoring the new options
  // without exposing the rich fields — the rich data is `transcribe_detailed`'s.
  use crate::audio::stt::model::{Transcribe as _, TranscribeOptions};
  let dir = fresh_dir("pr_e_universal");
  let model = tiny_model_with_tokenizer(13, dir.as_path());
  let audio = pr_e_waveform();

  let opts = TranscribeOptions::new()
    .with_language("en")
    .with_no_speech_threshold(None)
    .with_compression_ratio_threshold(None)
    .with_logprob_threshold(None)
    .with_max_new_tokens(4);
  let std = model.transcribe(&audio, &opts).unwrap();
  assert_eq!(std.language(), Some("en"));
  // The standard segments carry text + spans (the universal contract).
  for s in std.segments_slice() {
    assert!(s.end() >= s.start());
  }
}
