//! L5 — multi-token / string stop sequences driven through the real
//! [`stream_generate`] / [`generate`] entry points with a deterministic,
//! scriptable single-seq model and the committed `WordLevel` fixture
//! tokenizer (tokens 3-8 = `hello world the quick brown fox`). Stop
//! strings are derived from `tok.decode(...)` so the tests never hardcode
//! the tokenizer's spacing. The pure matcher logic (overlap, char
//! boundaries, first-match-wins) is unit-tested in [`crate::lm::stop`];
//! these assert the generate-loop wiring: `finish_reason="stop"`, the
//! trim, and the eos-only fallback when `stop_strings` is empty.

use super::*;
use crate::lm::cache::{CacheConfig, KvCache, make_prompt_cache};

/// Resolve the fixture tokenizer directory (`mlxrs/tests/fixtures`),
/// reachable from the in-crate `#[cfg(test)]` build via `CARGO_MANIFEST_DIR`.
fn fixture_tokenizer() -> crate::tokenizer::Tokenizer {
  let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("fixtures");
  crate::tokenizer::Tokenizer::from_path(&dir, None).expect("load fixture tokenizer")
}

/// A scriptable single-seq model: at decode step `t` (0-based, first decode
/// after prefill) it emits `script[t]` as the argmax. The script cursor is
/// `cache.offset() - prompt_len` (offset reaches `prompt_len` at the first
/// decode forward, then +1/step), so prefill chunking never shifts it —
/// identical wiring to the L1 `MockBatchModel`, single row.
struct ScriptModel {
  vocab: usize,
  prompt_len: usize,
  script: Vec<u32>,
}

impl Model for ScriptModel {
  fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    let shape = tokens.shape();
    let (batch, seq) = match shape.as_slice() {
      [b, s] => (*b, *s),
      [s] => (1usize, *s),
      other => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "ScriptModel::forward: tokens must be rank-1 [S] or rank-2 [B, S]",
          other.len() as u32,
          other.to_vec(),
        )));
      }
    };
    // Advance every cache so `offset()` increments like a real layer.
    for layer in cache.iter_mut() {
      let elems = batch * seq;
      let k = Array::from_slice::<f32>(&vec![1.0_f32; elems], &(batch, 1usize, seq, 1usize))?;
      let v = Array::from_slice::<f32>(&vec![2.0_f32; elems], &(batch, 1usize, seq, 1usize))?;
      layer.update(&k, &v)?;
    }
    let cache_offset = cache.first().map(|c| c.offset()).unwrap_or(0);
    let script_idx = cache_offset.checked_sub(self.prompt_len);
    let pred = script_idx
      .and_then(|i| self.script.get(i).copied())
      .unwrap_or(0);

    let mut data = vec![0.0_f32; batch * seq * self.vocab];
    if (pred as usize) < self.vocab {
      for pos in 0..batch * seq {
        data[pos * self.vocab + pred as usize] = 10.0;
      }
    }
    Array::from_slice::<f32>(&data, &(batch, seq, self.vocab))
  }
}

/// Run `generate` (collect every streamed segment) over a scripted decode.
/// `prompt` seeds the cache offset; `script` is the per-step argmax id
/// sequence; `stop_strings` configures L5. Returns the collected text plus
/// the per-response `finish_reason`s (in order) for assertions.
fn run(
  prompt: &[u32],
  script: Vec<u32>,
  max_tokens: usize,
  stop_strings: Vec<String>,
) -> (String, Vec<Option<FinishReason>>) {
  let tok = fixture_tokenizer();
  let vocab = 16usize;
  let model = ScriptModel {
    vocab,
    prompt_len: prompt.len(),
    script,
  };
  let cache = make_prompt_cache(&CacheConfig {
    num_hidden_layers: 1,
    sliding_window: None,
  });
  let cfg = GenConfig {
    max_tokens,
    stop_strings,
    ..Default::default()
  };
  let mut text = String::new();
  let mut reasons = Vec::new();
  for resp in stream_generate(&model, &tok, prompt, cache, cfg) {
    let r = resp.expect("stream step");
    text.push_str(&r.text);
    reasons.push(r.finish_reason);
  }
  (text, reasons)
}

/// The fixture's decode of a token-id slice (the exact text the
/// detokenizer reconstructs), used to build spacing-agnostic stop strings.
fn decode(ids: &[u32]) -> String {
  fixture_tokenizer().decode(ids, false).expect("decode")
}

#[test]
fn empty_stop_strings_is_eos_only_unchanged() {
  // Script: hello world </s>(eos=2) ... . With no stop strings, generation
  // ends on the eos token (finish_reason="stop"); the eos token is not
  // detokenized. Output is exactly the decode of the pre-eos tokens.
  let prompt = [1u32, 3]; // <s> hello
  let script = vec![4u32, 5, 2, 6, 7]; // world the </s> ...
  let (text, reasons) = run(&prompt, script, 32, Vec::new());
  assert_eq!(text, decode(&[4, 5]));
  assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Eos));
  // Only one "stop" (the eos), no premature stop.
  assert_eq!(reasons.iter().filter(|r| r.is_some()).count(), 1);
}

#[test]
fn single_token_stop_string_stops_and_trims() {
  // Stop on the single token `world` (id 4). Script produces hello world
  // the ...; generation must stop AT world and trim it, leaving `hello`.
  let prompt = [1u32, 3];
  let script = vec![3u32, 4, 5, 6, 7]; // hello world the quick brown
  let stop = decode(&[4]); // " world" (or "world") — whatever the fixture renders
  let (text, reasons) = run(&prompt, script, 32, vec![stop.clone()]);
  let full = decode(&[3, 4, 5]); // hello world the
  let cut = full.find(&stop).expect("stop substring present in decode");
  assert_eq!(text, full[..cut].to_string());
  // Typed FinishReason::Stop(matched) — as_str() collapses to "stop"
  // canonically, payload carries the matched sequence.
  assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(stop)));
}

#[test]
fn multi_token_stop_spanning_boundary_stops_and_trims() {
  // Stop string spans TWO tokens: decode([5,6]) = "the quick" (+ leading
  // space per the fixture). The match completes only when BOTH tokens have
  // been produced — a token boundary in the middle of the stop string.
  let prompt = [1u32, 3];
  let script = vec![3u32, 5, 6, 7, 8]; // hello the quick brown fox
  let stop = decode(&[5, 6]); // multi-token stop
  let (text, reasons) = run(&prompt, script, 32, vec![stop.clone()]);
  let full = decode(&[3, 5, 6, 7]); // up to brown
  let cut = full.find(&stop).expect("multi-token stop present");
  assert_eq!(text, full[..cut].to_string());
  // Crucially: it did NOT stop after the first token of the stop sequence —
  // the leading `hello` token survived (text is the non-empty pre-stop
  // prefix), and the full stop string is absent from the output.
  assert!(!text.is_empty());
  assert!(!text.contains(&stop));
  assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(stop)));
}

#[test]
fn partial_match_then_diverge_does_not_stop() {
  // Stop string = decode([5,6]) ("the quick"). Script produces the FIRST
  // token of it (`the`) then DIVERGES to `fox` — the partial match must
  // NOT fire; generation runs to max_tokens.
  let prompt = [1u32, 3];
  let script = vec![3u32, 5, 8, 4, 7]; // hello the fox world brown (no "the quick")
  let stop = decode(&[5, 6]);
  let (text, reasons) = run(&prompt, script, 5, vec![stop.clone()]);
  // No stop completed ⇒ ends on length, full text retained.
  assert!(!text.contains(&stop), "stop string must not appear");
  assert_eq!(text, decode(&[3, 5, 8, 4, 7]));
  assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Length));
  assert!(
    reasons
      .iter()
      .all(|r| r.as_ref() != Some(&FinishReason::Eos)),
    "no premature stop on the partial match"
  );
}

#[test]
fn stop_completes_mid_token_trims_at_char_boundary() {
  // The "mid-token" case the token-level trie cannot handle: the stop
  // string is a CHARACTER PREFIX of a token's decoded text. Token `quick`
  // (id 6) decodes to text containing "qui"; stopping on "qui" must trim
  // mid-token at the exact character boundary, dropping "qui" and the rest.
  let prompt = [1u32, 3];
  let script = vec![3u32, 6, 7, 8, 4]; // hello quick brown fox world
  let quick = decode(&[6]); // e.g. " quick"
  // Build a stop that is a strict character prefix of the `quick` token
  // text, ending mid-token (drop the last char so it cannot be a whole
  // token). Skip a leading space if present so we cut inside the word.
  let trimmed = quick.trim_start();
  assert!(trimmed.len() >= 3, "need a multi-char token to cut");
  let stop = trimmed[..trimmed.len() - 1].to_string(); // e.g. "quic"
  let (text, reasons) = run(&prompt, script, 32, vec![stop.clone()]);
  let full = decode(&[3, 6, 7]); // hello quick brown
  let cut = full.find(&stop).expect("mid-token stop prefix present");
  assert_eq!(text, full[..cut].to_string());
  // The stop string itself must be gone, and the cut is at the char
  // boundary where the stop began (mid the `quick` token's text).
  assert!(!text.contains(&stop));
  // Typed FinishReason::Stop(matched) carries the stop sequence;
  // FinishReason::as_str() still collapses to canonical "stop".
  assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(stop)));
}

#[test]
fn multiple_stop_strings_first_completion_wins() {
  // Two stops: one completes earlier in the stream than the other. The
  // earlier completion wins and trims there.
  let prompt = [1u32, 3];
  let script = vec![3u32, 4, 5, 6, 7]; // hello world the quick brown
  let early = decode(&[4]); // "world" — completes at step 2
  let late = decode(&[6]); // "quick" — would complete at step 4
  let (text, reasons) = run(&prompt, script, 32, vec![late.clone(), early.clone()]);
  let full = decode(&[3, 4]); // hello world
  let cut = full.find(&early).expect("early stop present");
  assert_eq!(text, full[..cut].to_string());
  assert!(!text.contains(&early));
  assert!(!text.contains(&late));
  // The early stop is the one that matched, so the typed payload carries it.
  assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(early)));
}

#[test]
fn finish_reason_is_stop_on_stop_string_match() {
  // Focused assertion: a stop-string match yields exactly one terminal
  // response with finish_reason == Some(Stop(matched)) and nothing after.
  let prompt = [1u32, 3];
  let script = vec![3u32, 4, 5, 6, 7];
  let stop = decode(&[5]); // "the"
  let (_text, reasons) = run(&prompt, script, 32, vec![stop.clone()]);
  assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(stop)));
  // Exactly one terminal reason, and it's the final element.
  assert_eq!(reasons.iter().filter(|r| r.is_some()).count(), 1);
}

// ── finalized-tail re-check on the active-matcher terminal paths ──────────
//
// Some detokenizers withhold tail text from `text()` until `finalize()`
// (the real BPE detok holds a single bare-space token for one step). The
// mid-stream matcher runs on `text()` BEFORE finalization, so a stop string
// completed only by that withheld tail is invisible until the terminal
// branch finalizes. These tests drive the exact unit the EOS / max_tokens
// active-matcher branches now call — [`finalize_active_tail`] — through a
// mock that reproduces the withhold-until-finalize behavior, asserting the
// tail completes the stop (trim + "stop"), including the max_tokens case
// where a finalized-tail stop must win over "length".

/// A mock [`StreamingDetokenizer`](crate::tokenizer::StreamingDetokenizer)
/// that withholds the most-recently-added "tail" token's text from `text()`
/// until the next `add_token` / `finalize` flushes it — exactly the BPE
/// detok's single-bare-space hold-back, but deterministic and tokenizer-free.
#[derive(Default)]
struct WithholdDetokenizer {
  /// Committed (visible) text — what `text()` returns.
  text: String,
  /// The withheld tail not yet visible in `text()` (flushed on the next
  /// `push` / `finalize`).
  pending: String,
  tokens: Vec<u32>,
  offset: usize,
}

impl WithholdDetokenizer {
  /// Add a token whose decoded text is `s`. When `withhold` is true the text
  /// is held back from `text()` until the next push/finalize (BPE bare-space
  /// semantics); otherwise it (and any pending tail) commits immediately.
  fn push(&mut self, s: &str, withhold: bool) {
    // A previously-withheld tail becomes visible as soon as another token
    // arrives (the BPE detok flushes `unflushed` on the next step).
    self.text.push_str(&self.pending);
    self.pending.clear();
    if withhold {
      self.pending.push_str(s);
    } else {
      self.text.push_str(s);
    }
    self.tokens.push(self.tokens.len() as u32);
  }
}

impl crate::tokenizer::StreamingDetokenizer for WithholdDetokenizer {
  fn reset(&mut self) {
    self.text.clear();
    self.pending.clear();
    self.tokens.clear();
    self.offset = 0;
  }
  fn add_token(&mut self, _token: u32) {}
  fn finalize(&mut self) {
    // Flush the withheld tail into the visible text (BPE `finalize`).
    self.text.push_str(&self.pending);
    self.pending.clear();
  }
  fn text(&self) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Borrowed(&self.text)
  }
  fn tokens(&self) -> &[u32] {
    &self.tokens
  }
  fn offset(&self) -> usize {
    self.offset
  }
  fn set_offset(&mut self, offset: usize) {
    self.offset = offset;
  }
}

/// Sanity: before `finalize`, the withheld tail is invisible in `text()`;
/// `finalize` makes it visible — the precondition that makes the bug bite.
#[test]
fn mock_withholds_tail_until_finalize() {
  use crate::tokenizer::StreamingDetokenizer;
  let mut d = WithholdDetokenizer::default();
  d.push("hello", false);
  d.push(" ", true); // bare space withheld
  assert_eq!(d.text().as_ref(), "hello"); // the space is NOT yet visible
  d.finalize();
  assert_eq!(d.text().as_ref(), "hello "); // now flushed
}

#[test]
fn finalized_tail_completes_stop_on_eos_trims_and_reports_stop() {
  // EOS terminal path: a withheld bare-space token completes the stop " ".
  // The mid-stream matcher (run on pre-finalize text "hello") never saw the
  // space; only the finalized text "hello " contains the stop. The eos
  // branch passes default_reason="stop"; finalize_active_tail must trim the
  // space and report "stop".
  let stop = crate::lm::stop::StopMatcher::new(vec![" ".to_string()]);
  let mut d = WithholdDetokenizer::default();
  d.push("hello", false);
  d.push(" ", true); // the eos-preceding token; held back until finalize
  // The visible "hello" was already streamed mid-loop (emitted_len tracks it).
  let mut emitted_len = "hello".len();
  crate::tokenizer::StreamingDetokenizer::finalize(&mut d);
  let (text, reason) = finalize_active_tail(&d, &stop, &mut emitted_len, FinishReason::Eos);
  // The stop " " starts at byte 5; trimmed_len=5 == emitted_len ⇒ nothing
  // new emitted (the space is trimmed away, not returned). The finalized
  // tail completed the stop, so the typed reason is Stop(matched), not Eos.
  assert_eq!(text, "");
  assert_eq!(reason, FinishReason::Stop(" ".to_string()));
  assert!(!text.contains(' '), "the bare space must not be emitted");
}

#[test]
fn finalized_tail_completes_stop_on_max_tokens_wins_over_length() {
  // max_tokens terminal path: the final allowed token is a withheld bare
  // space that completes the stop " ". finalize_active_tail is called with
  // default_reason="length", but a stop completed by the finalized tail must
  // OVERRIDE to "stop" (and trim), not report "length".
  let stop = crate::lm::stop::StopMatcher::new(vec![" ".to_string()]);
  let mut d = WithholdDetokenizer::default();
  d.push("hi", false);
  d.push(" ", true); // final allowed token, withheld until finalize
  let mut emitted_len = "hi".len();
  crate::tokenizer::StreamingDetokenizer::finalize(&mut d);
  let (text, reason) = finalize_active_tail(&d, &stop, &mut emitted_len, FinishReason::Length);
  assert_eq!(
    reason,
    FinishReason::Stop(" ".to_string()),
    "finalized-tail stop must win over length and carry the matched payload"
  );
  assert_eq!(text, ""); // the space is trimmed, not emitted
  assert!(!text.contains(' '));
}

#[test]
fn finalized_tail_no_stop_emits_tail_with_default_reason() {
  // Control: when the finalized tail does NOT complete a stop, the tail is
  // emitted and default_reason is preserved (length on max_tokens, stop on
  // eos). Guards against the re-check spuriously trimming/relabeling.
  let stop = crate::lm::stop::StopMatcher::new(vec!["ZZZ".to_string()]); // never matches
  let mut d = WithholdDetokenizer::default();
  d.push("hi", false);
  d.push(" ", true); // withheld tail, no stop completion
  let mut emitted_len = "hi".len();
  crate::tokenizer::StreamingDetokenizer::finalize(&mut d);
  // max_tokens semantics → Length, and the withheld space is emitted.
  let (text, reason) = finalize_active_tail(&d, &stop, &mut emitted_len, FinishReason::Length);
  assert_eq!(text, " ");
  assert_eq!(reason, FinishReason::Length);
  // eos semantics → Eos, same emitted tail.
  let mut d2 = WithholdDetokenizer::default();
  d2.push("hi", false);
  d2.push(" ", true);
  let mut emitted_len2 = "hi".len();
  crate::tokenizer::StreamingDetokenizer::finalize(&mut d2);
  let (text2, reason2) = finalize_active_tail(&d2, &stop, &mut emitted_len2, FinishReason::Eos);
  assert_eq!(text2, " ");
  assert_eq!(reason2, FinishReason::Eos);
}

#[test]
fn finalized_tail_completes_multichar_stop_spanning_into_tail() {
  // The withheld tail supplies the final char of a multi-char stop that
  // straddles the commit/withhold boundary: visible "ab", withheld "c",
  // stop "abc". Pre-finalize text "ab" has no match; finalized "abc" does.
  // finalize_active_tail trims at the match start (byte 0), but emitted_len
  // is already 2 (the "ab" was streamed), so nothing new is emitted and the
  // reason is "stop".
  let stop = crate::lm::stop::StopMatcher::new(vec!["abc".to_string()]);
  let mut d = WithholdDetokenizer::default();
  d.push("ab", false);
  d.push("c", true);
  let mut emitted_len = "ab".len();
  crate::tokenizer::StreamingDetokenizer::finalize(&mut d);
  let (text, reason) = finalize_active_tail(&d, &stop, &mut emitted_len, FinishReason::Length);
  assert_eq!(reason, FinishReason::Stop("abc".to_string()));
  assert_eq!(text, "");
  // emitted_len clamped to match start (0) max emitted (2) = 2.
  assert_eq!(emitted_len, 2);
}

// ════════════════════════════════════════════════════════════════════════
//   EOS through the stream_generate active-matcher path + generate() entry
// ════════════════════════════════════════════════════════════════════════

/// EOS terminal path WITH an active (but non-matching) stop matcher: the eos
/// token reaches `stream_generate`'s `if eos.contains(&token)` branch, which
/// — because `matcher.is_active()` — calls `finalize_active_tail` with
/// `FinishReason::Eos`. The never-matching stop string means the tail is
/// emitted as-is and the reason stays `Eos`. Exercises the active-matcher
/// EOS branch the existing Stop/Length tests skip.
#[test]
fn eos_with_active_non_matching_matcher_reports_eos() {
  // hello world </s>(eos) ...; stop "ZZZ" never matches.
  let prompt = [1u32, 3];
  let script = vec![4u32, 5, 2, 6, 7]; // world the </s> ...
  let (text, reasons) = run(&prompt, script, 32, vec!["ZZZ".to_string()]);
  // eos(2) is not detokenized ⇒ output is decode of the pre-eos tokens.
  assert_eq!(text, decode(&[4, 5]));
  assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Eos));
  // Exactly one terminal reason (the eos), no premature stop.
  assert_eq!(reasons.iter().filter(|r| r.is_some()).count(), 1);
}

/// `generate` collects every `stream_generate` segment into one `String`
/// and returns the final response's stats. Closed-form oracle: an
/// eos-terminated script `[4, 5, 2]` yields text == decode([4, 5]) (eos not
/// detokenized) and `generation_tokens == 3` (mlx-lm `n + 1`: two emitted
/// tokens + the eos-bearing final response).
#[test]
fn generate_collects_text_and_reports_stats_on_eos() {
  let tok = fixture_tokenizer();
  let model = ScriptModel {
    vocab: 16,
    prompt_len: 2,
    script: vec![4u32, 5, 2, 6, 7], // world the </s>
  };
  let cache = make_prompt_cache(&CacheConfig {
    num_hidden_layers: 1,
    sliding_window: None,
  });
  let cfg = GenConfig {
    max_tokens: 32,
    ..Default::default()
  };
  let (text, stats) = generate(&model, &tok, &[1u32, 3], cache, cfg).expect("generate ok");
  assert_eq!(text, decode(&[4, 5]), "eos token contributes no text");
  assert_eq!(stats.prompt_tokens, 2, "prompt was 2 tokens");
  assert_eq!(
    stats.generation_tokens, 3,
    "two emitted tokens + the eos-bearing final response (n + 1)"
  );
  // tps are non-negative (wall-clock derived; exact value not asserted).
  assert!(stats.prompt_tps >= 0.0 && stats.generation_tps >= 0.0);
}

/// `generate` with `max_tokens == 0`: `stream_generate` yields nothing, so
/// the `None`-final-response branch returns the empty string + a zero-counts
/// `GenerationStats` that still carries the original `prompt_tokens`.
#[test]
fn generate_zero_max_tokens_empty_text_zero_stats() {
  let tok = fixture_tokenizer();
  let model = ScriptModel {
    vocab: 16,
    prompt_len: 3,
    script: vec![4u32, 5, 6],
  };
  let cache = make_prompt_cache(&CacheConfig {
    num_hidden_layers: 1,
    sliding_window: None,
  });
  let cfg = GenConfig {
    max_tokens: 0,
    ..Default::default()
  };
  let (text, stats) = generate(&model, &tok, &[1u32, 3, 4], cache, cfg).expect("generate ok");
  assert_eq!(text, "", "no tokens produced ⇒ empty output");
  assert_eq!(stats.generation_tokens, 0);
  assert_eq!(
    stats.prompt_tokens, 3,
    "prompt_tokens preserved on the empty run"
  );
  assert_eq!(stats.prompt_tps, 0.0);
  assert_eq!(stats.generation_tps, 0.0);
}

/// `generate` propagates an underlying step error as `Err` (short-circuits
/// the collection). A model whose `forward` always fails drives the
/// `stream_generate` Iterator-`Err` contract through `generate`.
#[test]
fn generate_propagates_step_error() {
  struct FailModel;
  impl Model for FailModel {
    fn forward(&self, _t: &Array, _c: &mut [Box<dyn KvCache>]) -> Result<Array> {
      Err(Error::InvariantViolation(
        crate::error::InvariantViolationPayload::new("FailModel::forward", "mock forward failure"),
      ))
    }
  }
  let tok = fixture_tokenizer();
  let cache = make_prompt_cache(&CacheConfig {
    num_hidden_layers: 1,
    sliding_window: None,
  });
  let cfg = GenConfig {
    max_tokens: 4,
    ..Default::default()
  };
  let res = generate(&FailModel, &tok, &[1u32, 3], cache, cfg);
  assert!(
    res.is_err(),
    "a forward failure surfaces as Err from generate"
  );
}
