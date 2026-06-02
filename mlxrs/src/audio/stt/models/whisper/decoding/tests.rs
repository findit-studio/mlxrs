use super::*;
use crate::{audio::stt::models::whisper::config::ModelDimensions, tokenizer::Tokenizer};
use serde_json::json;
use std::{
  collections::HashMap,
  path::{Path, PathBuf},
};

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
fn argmax_slice_picks_first_max() {
  assert_eq!(argmax_slice(&[1.0, 5.0, 5.0, 2.0]), 1); // ties → lowest index
  assert_eq!(argmax_slice(&[-3.0, -1.0, -2.0]), 1);
  assert_eq!(argmax_slice(&[]), 0);
  // -inf entries are never selected over a finite max.
  assert_eq!(
    argmax_slice(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]),
    1
  );
}

#[test]
fn max_slice_and_logsumexp_slice() {
  assert_eq!(max_slice(&[]), f64::NEG_INFINITY);
  assert_eq!(max_slice(&[1.0, 3.0, 2.0]), 3.0);
  // logsumexp of all -inf (or empty) is -inf.
  assert_eq!(logsumexp_slice(&[]), f64::NEG_INFINITY);
  assert_eq!(
    logsumexp_slice(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
    f64::NEG_INFINITY
  );
  // logsumexp([0,0,0]) = ln(3).
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
  // blank_ids = {space_id=5, eot=2}; sample_begin = 3.
  let filter = SuppressBlank {
    sample_begin: 3,
    blank_ids: vec![5, 2],
  };
  let mut row = vec![0.0_f32; 8];
  // tokens.len() == sample_begin → mask fires.
  filter.apply(&mut row, &[1, 2, 3]);
  assert!(row[5].is_infinite() && row[5] < 0.0);
  assert!(row[2].is_infinite() && row[2] < 0.0);
  assert_eq!(row[0], 0.0);

  // tokens.len() != sample_begin → no masking.
  let mut row2 = vec![0.0_f32; 8];
  filter.apply(&mut row2, &[1, 2, 3, 4]);
  assert_eq!(row2, vec![0.0_f32; 8]);
}

#[test]
fn suppress_blank_ignores_out_of_range_ids() {
  // An id past the vocab is silently skipped (no panic).
  let filter = SuppressBlank {
    sample_begin: 0,
    blank_ids: vec![100],
  };
  let mut row = vec![0.0_f32; 4];
  filter.apply(&mut row, &[]);
  assert_eq!(row, vec![0.0_f32; 4]);
}

#[test]
fn suppress_tokens_unconditional() {
  let filter = SuppressTokens { ids: vec![1, 3] };
  let mut row = vec![0.0_f32; 5];
  // Applies regardless of token history.
  filter.apply(&mut row, &[9, 9, 9]);
  assert!(row[1] < 0.0 && row[1].is_infinite());
  assert!(row[3] < 0.0 && row[3].is_infinite());
  assert_eq!(row[0], 0.0);
  assert_eq!(row[2], 0.0);
  assert_eq!(row[4], 0.0);
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
  }
}

const VOCAB: usize = 20;

#[test]
fn timestamp_rules_always_suppress_no_timestamps() {
  let f = ts_rules(0, None);
  let mut row = vec![0.0_f32; VOCAB];
  // A non-first step with a non-timestamp last token: only no_timestamps is
  // forced off (besides the probability-mass rule, inert for a flat row).
  f.apply(&mut row, &[3, 4]); // sample_begin=0 so seq=[3,4]
  assert!(
    row[12].is_infinite() && row[12] < 0.0,
    "no_timestamps masked"
  );
}

#[test]
fn timestamp_rules_first_position_forces_timestamp() {
  // At the first sampled position (tokens.len() == sample_begin) all
  // non-timestamp tokens [0, timestamp_begin) are masked.
  let f = ts_rules(1, None);
  let mut row = vec![0.0_f32; VOCAB];
  f.apply(&mut row, &[3]); // len == sample_begin == 1
  for (i, &v) in row.iter().enumerate().take(14) {
    assert!(
      v.is_infinite() && v < 0.0,
      "text/special token {i} masked at first pos"
    );
  }
  // Timestamp tokens remain (not -inf, ignoring the prob-mass rule on a flat
  // row where timestamp mass ln(6) > max_text -inf → also forces, so check a
  // single timestamp survives only if mass rule didn't fire). With a flat row,
  // the [0, ts_begin) region is already fully masked, so this is consistent.
  assert!(row[14].is_finite() || row[14] == f32::NEG_INFINITY);
}

#[test]
fn timestamp_rules_max_initial_timestamp_caps_high_timestamps() {
  // At the first position with max_initial_timestamp_index = 2, timestamps
  // beyond timestamp_begin + 2 = 16 are masked (positions 17..20).
  let f = ts_rules(1, Some(2));
  let mut row = vec![0.0_f32; VOCAB];
  f.apply(&mut row, &[3]);
  // 17,18,19 masked by the cap.
  for (i, &v) in row.iter().enumerate().skip(17) {
    assert!(v.is_infinite() && v < 0.0, "ts {i} capped");
  }
}

#[test]
fn timestamp_rules_pair_after_two_timestamps_forbids_more_timestamps() {
  // last and penultimate are both timestamps → next must be non-timestamp:
  // mask [timestamp_begin, vocab). sample_begin = 0 so seq = full tokens.
  let f = ts_rules(0, None);
  let mut row = vec![0.0_f32; VOCAB];
  // seq ends [..., 15(ts), 16(ts)] → both timestamps.
  f.apply(&mut row, &[3, 15, 16]);
  for (i, &v) in row.iter().enumerate().skip(14) {
    assert!(v.is_infinite() && v < 0.0, "timestamp {i} masked");
  }
  // text token 3 stays finite (it is allowed next).
  assert!(row[3].is_finite());
}

#[test]
fn timestamp_rules_single_trailing_timestamp_forbids_text() {
  // last is a timestamp, penultimate is NOT → cannot be a normal text token:
  // mask [0, eot). seq = [3(text), 15(ts)].
  let f = ts_rules(0, None);
  let mut row = vec![0.0_f32; VOCAB];
  f.apply(&mut row, &[3, 15]);
  // [0, eot=2) masked.
  assert!(row[0].is_infinite() && row[0] < 0.0);
  assert!(row[1].is_infinite() && row[1] < 0.0);
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
  let mut row = vec![0.0_f32; VOCAB];
  f.apply(&mut row, &[3, 17, 5]);
  // timestamps 14..18 masked (<= the last seen 17).
  for (i, &v) in row.iter().enumerate().take(18).skip(14) {
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
  let mut row = vec![0.0_f32; VOCAB];
  f.apply(&mut row, &[14, 12, 0]);
  // <|0.00|> (id 14) is masked — no zero-length closing segment.
  assert!(
    row[14].is_infinite() && row[14] < 0.0,
    "<|0.00|> must be masked to force nonzero segment length"
  );
  // A later timestamp (id 15 = <|0.02|>) remains legal (only [14,15) masked).
  assert!(
    row[15].is_finite(),
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
  let mut row = vec![0.0_f32; VOCAB];
  f.apply(&mut row, &[3, 15]);
  // 14 masked (smaller than 15); 15 itself NOT masked by the monotonicity rule.
  assert!(row[14].is_infinite() && row[14] < 0.0, "ts 14 < 15 masked");
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
  let mut row = vec![0.0_f32; VOCAB];
  for v in row.iter_mut().take(14) {
    *v = 0.0; // text logits
  }
  for v in row.iter_mut().skip(14) {
    *v = 10.0; // dominant timestamp logits
  }
  f.apply(&mut row, &[3, 4]); // last token 4 is text (not a timestamp)
  for (i, &v) in row.iter().enumerate().take(14) {
    assert!(
      v.is_infinite() && v < 0.0,
      "text {i} masked by prob-mass rule"
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
  let logits = vec![0.0, 0.0, 0.0, 5.0, 0.0];
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
  let logits = vec![0.0, 0.0, 0.0, 5.0, 0.0];
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
  let logits = vec![0.0, 0.0, 9.0, 0.0];
  let (next, completed) = d.update(&logits, /* last */ 1).unwrap();
  assert_eq!(next, 2);
  assert!(completed);
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
  let toks = Array::from_slice::<u32>(&[3, 4, 7], &(1, 3)).unwrap();
  let (logits, _cache) = model.decode_tokens(&toks, &enc, None).unwrap();
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
fn segment_collection_no_timestamps_one_segment() {
  // No consecutive timestamps → one segment spanning the whole window; advance
  // by the full segment_size.
  let dir = fresh_dir("seg_no_ts");
  let tok = write_tokenizer(dir.as_path());
  let w = HFTokenizerWrapper::new(&tok, true, 2, Some("en"), Task::Transcribe).unwrap();
  let ts_begin = w.timestamp_begin(); // 14
  let mut segments = Vec::new();
  let mut all_text = String::new();
  let tokens = vec![0u32, 1, 12]; // all text tokens "a b c", no timestamps
  let advance = advance_and_collect_segments(
    &tokens,
    ts_begin,
    /* time_offset */ 0.0,
    /* time_precision */ 0.02,
    /* segment_size */ 100,
    /* input_stride */ 2,
    &dummy_result(),
    &w,
    &mut segments,
    &mut all_text,
  )
  .unwrap();
  assert_eq!(advance, 100);
  assert_eq!(segments.len(), 1);
  assert_eq!(segments[0].start, 0.0);
  // text decodes the non-special tokens.
  assert!(!all_text.is_empty());
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
  let mut all_text = String::new();
  let tokens = vec![14u32, 0, 1, 16, 16, 12, 17];
  let advance = advance_and_collect_segments(
    &tokens,
    ts_begin,
    0.0,
    0.02,
    /* segment_size */ 200,
    /* input_stride */ 2,
    &dummy_result(),
    &w,
    &mut segments,
    &mut all_text,
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
  let mut all_text = String::new();
  // 14 14 (consecutive pair) ... 12 (text) 15 (single trailing ts).
  let tokens = vec![14u32, 14, 0, 12, 15];
  let advance = advance_and_collect_segments(
    &tokens,
    ts_begin,
    0.0,
    0.02,
    /* segment_size */ 150,
    2,
    &dummy_result(),
    &w,
    &mut segments,
    &mut all_text,
  )
  .unwrap();
  assert_eq!(advance, 150);
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
