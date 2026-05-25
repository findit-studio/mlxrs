//! Multi-token / string stop-sequence detection for the generation loop
//! (L5), ported from `mlx_lm.generate`'s stop-sequence handling.
//!
//! # Reference behavior
//!
//! mlx-lm supports arbitrary **stop strings** (multi-token sequences), not
//! just the single-token eos set. Internally its
//! [`SequenceStateMachine`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
//! runs an Aho-Corasick trie over the **token** stream: each stop string is
//! tokenized once (`tokenizer.encode(w, add_special_tokens=False)`) and the
//! trie matches token-by-token. When a token completes a stop sequence the
//! `finish_reason` becomes `"stop"`, and the server's
//! `_process_control_tokens` (`server.py`) **zeroes the text of every token
//! in the matched sequence** — a deque buffers the last
//! `max(len(seq))` token-responses so the whole matched sequence (not just
//! its final token) is trimmed from the returned text. mlx-swift-lm has no
//! multi-token stop support (single-token `extraEOSTokens` only), so mlx-lm
//! is the sole multi-token reference.
//!
//! # Approach chosen here: decode-and-string-match
//!
//! This module matches stop strings against the **decoded text** rather than
//! the token stream. The two are character-equivalent for the common case,
//! and string-matching additionally handles the case a token-level trie
//! cannot: a stop string completing *mid-token* (a single produced token
//! whose decoded text contains the stop string plus trailing characters) —
//! string-matching trims at the exact character boundary, which is the
//! behavior L5 specifies. It is also the simplest faithful port: the trim
//! semantics ("remove the whole matched stop sequence from the output") fall
//! out directly from truncating the decoded text at the match start.
//!
//! # Streaming hold-back (the overlap problem)
//!
//! A stop string may complete across a token boundary, and a token may carry
//! only a *prefix* of a stop string. To stream incrementally without ever
//! emitting text that later turns out to be (part of) a stop sequence, the
//! matcher holds back the longest suffix of the running text that is a proper
//! prefix of some stop string. That suffix is at most
//! `max_stop_len - 1` bytes, mirroring mlx-lm's deque `buffer_size`
//! (`max(len(s) for s in sequences)`): only text that can no longer become
//! part of any stop is released. When a stop completes, the held-back +
//! newly matched bytes are dropped together, so the whole stop sequence is
//! trimmed exactly once.

/// Payload for [`StopDecision::Continue`] — carries the safe-to-emit byte
/// length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuePayload {
  safe_len: usize,
}

impl ContinuePayload {
  /// Construct a [`ContinuePayload`].
  pub fn new(safe_len: usize) -> Self {
    Self { safe_len }
  }

  /// Byte length of the cumulative text that is safe to emit.
  #[inline(always)]
  pub fn safe_len(&self) -> usize {
    self.safe_len
  }
}

/// Payload for [`StopDecision::Stop`] — carries the trimmed-text length and
/// which stop string matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopPayload {
  trimmed_len: usize,
  stop: String,
}

impl StopPayload {
  /// Construct a [`StopPayload`].
  pub fn new(trimmed_len: usize, stop: impl Into<String>) -> Self {
    Self {
      trimmed_len,
      stop: stop.into(),
    }
  }

  /// Byte length of the cumulative text up to (excluding) the matched stop.
  #[inline(always)]
  pub fn trimmed_len(&self) -> usize {
    self.trimmed_len
  }

  /// Which configured stop string matched.
  #[inline(always)]
  pub fn stop(&self) -> &str {
    &self.stop
  }
}

/// Outcome of feeding the running decoded text to a [`StopMatcher`].
#[derive(
  Debug, Clone, PartialEq, Eq, derive_more::IsVariant, derive_more::Unwrap, derive_more::TryUnwrap,
)]
#[unwrap(ref, ref_mut)]
#[try_unwrap(ref, ref_mut)]
#[non_exhaustive]
pub enum StopDecision {
  /// No stop string has completed yet. `safe_len` is the byte length of the
  /// prefix of the cumulative decoded text that is now safe to emit (no stop
  /// string can complete using only the held-back suffix). It only ever
  /// grows across calls; the caller emits the newly-safe slice.
  Continue(ContinuePayload),
  /// A stop string completed. `trimmed_len` is the byte length of the
  /// cumulative decoded text with the matched stop sequence (and anything
  /// after it) removed — i.e. the text truncated at the match start. The
  /// matched stop string is reported in `stop`.
  Stop(StopPayload),
}

/// A decode-and-string-match stop-sequence matcher.
///
/// Construct with [`StopMatcher::new`] (empty stop list ⇒ a no-op matcher
/// that always returns [`StopDecision::Continue`] with the whole text safe),
/// then call [`StopMatcher::step`] with the cumulative decoded generation
/// text after each token. The matcher is purely text-driven and holds no
/// model/tokenizer state, so it is unit-testable in isolation.
#[derive(Debug, Clone)]
pub struct StopMatcher {
  /// Configured non-empty stop strings, in caller order (used for
  /// first-match tie-breaking).
  stops: Vec<String>,
  /// `max(stop.len())`; `0` when there are no stops.
  max_len: usize,
}

impl StopMatcher {
  /// Build a matcher from the configured stop strings. Empty strings are
  /// dropped (they can never meaningfully "complete"); if nothing remains
  /// the matcher is inert ([`StopMatcher::is_active`] is `false`).
  pub fn new<I, S>(stop_strings: I) -> Self
  where
    I: IntoIterator<Item = S>,
    S: Into<String>,
  {
    let stops: Vec<String> = stop_strings
      .into_iter()
      .map(Into::into)
      .filter(|s| !s.is_empty())
      .collect();
    let max_len = stops.iter().map(String::len).max().unwrap_or(0);
    Self { stops, max_len }
  }

  /// `true` iff at least one non-empty stop string is configured. When
  /// `false`, [`StopMatcher::step`] always returns
  /// [`StopDecision::Continue`] with the entire text safe (eos-only
  /// behavior, identical to no stop support at all).
  pub fn is_active(&self) -> bool {
    !self.stops.is_empty()
  }

  /// Feed the cumulative decoded generation text (everything generated so
  /// far, in order) and decide whether a stop string has completed.
  ///
  /// Returns [`StopDecision::Stop`] at the **earliest** completion across all
  /// configured stops (earliest match start wins; among stops sharing that
  /// start the first in construction order is reported), with the text
  /// trimmed at the match start. Otherwise returns [`StopDecision::Continue`]
  /// with the safe-to-emit prefix length, holding back the longest trailing
  /// partial stop-prefix so a later token can still complete it.
  pub fn step(&self, full_text: &str) -> StopDecision {
    if self.stops.is_empty() {
      return StopDecision::Continue(ContinuePayload::new(full_text.len()));
    }

    // Earliest completed match: minimize the match START (so the shortest
    // returned text wins), tie-broken by construction order via `find`'s
    // first-occurrence and a stable scan over `self.stops`.
    let mut best: Option<(usize, &str)> = None;
    for stop in &self.stops {
      if let Some(start) = full_text.find(stop.as_str()) {
        match best {
          Some((b, _)) if start >= b => {}
          _ => best = Some((start, stop.as_str())),
        }
      }
    }
    if let Some((start, stop)) = best {
      return StopDecision::Stop(StopPayload::new(start, stop));
    }

    // No completion. Hold back the longest suffix of `full_text` that is a
    // proper prefix of some stop string (it could still grow into a match),
    // capped at `max_len - 1`. Everything before it is safe to emit.
    let held = self.held_back_suffix(full_text);
    StopDecision::Continue(ContinuePayload::new(full_text.len() - held))
  }

  /// Byte length of the longest suffix of `text` that is a proper prefix of
  /// some configured stop string (the in-progress partial match to withhold).
  /// Capped at `max_len - 1` (a length-`max_len` suffix equal to a whole stop
  /// would have already matched in [`StopMatcher::step`]). Always lands on a
  /// `char` boundary so the caller can slice safely.
  fn held_back_suffix(&self, text: &str) -> usize {
    let cap = self.max_len.saturating_sub(1).min(text.len());
    // Try the longest candidate suffix first; the first (longest) suffix that
    // is a proper prefix of some stop is the amount to hold back.
    let mut len = cap;
    while len > 0 {
      let start = text.len() - len;
      if text.is_char_boundary(start) {
        let suffix = &text[start..];
        if self
          .stops
          .iter()
          .any(|s| s.len() > suffix.len() && s.as_bytes().starts_with(suffix.as_bytes()))
        {
          return len;
        }
      }
      len -= 1;
    }
    0
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn matcher(stops: &[&str]) -> StopMatcher {
    StopMatcher::new(stops.iter().copied())
  }

  #[test]
  fn inert_when_empty() {
    let m = matcher(&[]);
    assert!(!m.is_active());
    // Whole text is always safe; never stops.
    assert_eq!(
      m.step("anything at all"),
      StopDecision::Continue(ContinuePayload::new("anything at all".len()))
    );
  }

  #[test]
  fn empty_strings_are_dropped() {
    let m = matcher(&["", ""]);
    assert!(!m.is_active());
    assert_eq!(m.step("x"), StopDecision::Continue(ContinuePayload::new(1)));
  }

  #[test]
  fn simple_match_trims_at_start() {
    let m = matcher(&["STOP"]);
    // "abcSTOPdef" -> trim at the 'S' (byte 3).
    assert_eq!(
      m.step("abcSTOPdef"),
      StopDecision::Stop(StopPayload::new(3, "STOP"))
    );
  }

  #[test]
  fn no_match_holds_back_partial_prefix() {
    let m = matcher(&["STOP"]);
    // "abcST" ends with "ST", a proper prefix of "STOP": hold back 2 bytes.
    assert_eq!(
      m.step("abcST"),
      StopDecision::Continue(ContinuePayload::new(3))
    );
    // "abc" ends with no stop-prefix: all safe.
    assert_eq!(
      m.step("abc"),
      StopDecision::Continue(ContinuePayload::new(3))
    );
  }

  #[test]
  fn partial_then_diverge_releases_held_text() {
    let m = matcher(&["STOP"]);
    // Step 1: "...ST" holds back "ST".
    assert_eq!(
      m.step("xxST"),
      StopDecision::Continue(ContinuePayload::new(2))
    );
    // Step 2: next token made it "STX" — "ST" did not become "STOP", and the
    // new tail "X"/"TX"/"STX" is not a stop prefix, so everything is safe.
    assert_eq!(
      m.step("xxSTX"),
      StopDecision::Continue(ContinuePayload::new(5))
    );
  }

  #[test]
  fn first_match_wins_earliest_start() {
    // "foo" starts at 0, "bar" at 3: "foo" is the earliest completion.
    let m = matcher(&["bar", "foo"]);
    assert_eq!(
      m.step("foobar"),
      StopDecision::Stop(StopPayload::new(0, "foo"))
    );
  }

  #[test]
  fn first_match_wins_tie_broken_by_order() {
    // Both start at byte 0; construction order picks the first listed.
    let m = matcher(&["ab", "abc"]);
    assert_eq!(m.step("abc"), StopDecision::Stop(StopPayload::new(0, "ab")));
  }

  #[test]
  fn multibyte_held_back_suffix_is_char_safe() {
    // Stop "é!" (é is 2 bytes). Text ends with the bare "é": held back as a
    // partial prefix without splitting the codepoint.
    let m = matcher(&["é!"]);
    let d = m.step("abé");
    // "abé" = 4 bytes; held back is the 2-byte "é".
    assert_eq!(d, StopDecision::Continue(ContinuePayload::new(2)));
  }

  #[test]
  fn multibyte_match_trims_at_char_boundary() {
    let m = matcher(&["é!"]);
    assert_eq!(
      m.step("abé!cd"),
      StopDecision::Stop(StopPayload::new(2, "é!"))
    );
  }
}
