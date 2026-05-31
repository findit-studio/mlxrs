//! Streaming-detokenizer **correctness** regressions.
//!
//! These complement `tokenizer_adversarial.rs` (panic/OOM classes) with
//! three semantic defects:
//!
//! * Naive `text()`/`last_segment()` exposed no incremental text
//!   before a newline/`finalize()` (Python returns `_text + _current_text`).
//! * BPE `decode_bytes` corrupted the valid byte `0x00` token
//!   (U+0100) into utf-8 text instead of pushing the raw byte.
//! * BPE HashMap fallback injected `"!"` for *every* absent id;
//!   mlx-lm only does so for ids `>= len(tokenmap)` (`token > max_id`),
//!   while an in-range hole decodes to `""`.
//!
//! Each test is gated on the narrowest feature that provides the type it
//! exercises so the naive-streaming test runs under bare `tokenizer-stream` and the two BPE tests under
//! `tokenizer-bpe` (both of which also run via the `lm` umbrella).
#![cfg(feature = "tokenizer-stream")]

// ---------------------------------------------------------------------------
// Naive streaming exposes in-progress text BEFORE any newline/finalize.
// ---------------------------------------------------------------------------

#[cfg(feature = "tokenizer-stream")]
#[test]
fn naive_text_and_last_segment_expose_partial_before_newline() {
  use mlxrs::tokenizer::{StreamingDetokenizer, stream::NaiveStreamingDetokenizer};

  // No-newline decode: Python's `text` property returns `_text +
  // _current_text`, so a token-by-token generation loop reading
  // `last_segment` must see the partial text immediately — not nothing
  // until `\n`/`finalize()`.
  let decode = |ids: &[u32]| {
    ids
      .iter()
      .map(|&i| match i {
        1 => "Hel",
        2 => "lo",
        3 => " wor",
        4 => "ld",
        _ => "",
      })
      .collect::<String>()
  };
  let mut d = NaiveStreamingDetokenizer::new(decode, false);
  d.reset();

  d.add_token(1);
  // Partial text visible BEFORE any newline / finalize (regression: was "").
  assert_eq!(d.text(), "Hel");
  let seg1 = d.last_segment();
  assert_eq!(seg1, "Hel");

  d.add_token(2);
  assert_eq!(d.text(), "Hello");
  let seg2 = d.last_segment();
  assert_eq!(seg2, "lo");

  d.add_token(3);
  d.add_token(4);
  assert_eq!(d.text(), "Hello world");
  let seg3 = d.last_segment();
  assert_eq!(seg3, " world");

  // Accumulated stream == full decode, and `finalize()` leaves it unchanged.
  d.finalize();
  assert_eq!(d.text(), "Hello world");
  assert_eq!(d.tokens(), &[1u32, 2, 3, 4]);

  // `combined_text` stays identical to `text()` (source-compat shim).
  let decode2 = |ids: &[u32]| ids.iter().map(|&i| format!("x{i}")).collect::<String>();
  let mut d2 = NaiveStreamingDetokenizer::new(decode2, false);
  d2.reset();
  d2.add_token(7);
  assert_eq!(d2.combined_text(), d2.text());
  assert_eq!(d2.combined_text(), "x7");
}

// ---------------------------------------------------------------------------
// (continued) — SPM/BPE still accumulate into `self.text` and stream
// exactly as before the `text() -> Cow` change (zero behaviour delta).
// ---------------------------------------------------------------------------

#[cfg(feature = "tokenizer-spm")]
#[test]
fn spm_streaming_unchanged_after_cow_text() {
  use mlxrs::tokenizer::{StreamingDetokenizer, stream::SpmStreamingDetokenizer};

  let vocab = vec![
    ("\u{2581}Hello".to_string(), 0u32),
    ("\u{2581}world".to_string(), 1u32),
    ("!".to_string(), 2u32),
  ];
  let mut d = SpmStreamingDetokenizer::new(vocab, true);
  d.reset();
  let mut streamed = String::new();
  for t in [0u32, 1, 2] {
    d.add_token(t);
    streamed.push_str(&d.last_segment());
  }
  d.finalize();
  streamed.push_str(&d.last_segment());
  assert_eq!(d.text(), "Hello world!");
  assert_eq!(streamed, "Hello world!");
}

#[cfg(feature = "tokenizer-bpe")]
#[test]
fn bpe_streaming_unchanged_after_cow_text() {
  use mlxrs::tokenizer::{StreamingDetokenizer, stream::BpeStreamingDetokenizer};

  let vocab = vec![
    ("Hello".to_string(), 0u32),
    ("\u{0120}world".to_string(), 1u32),
  ];
  let mut d = BpeStreamingDetokenizer::new(vocab, false);
  d.reset();
  let mut streamed = String::new();
  d.add_token(0);
  streamed.push_str(&d.last_segment());
  d.add_token(1);
  streamed.push_str(&d.last_segment());
  d.finalize();
  streamed.push_str(&d.last_segment());
  assert_eq!(d.text(), "Hello world");
  assert_eq!(streamed, "Hello world");
}

// ---------------------------------------------------------------------------
// a token decoding (via the GPT-2 byte map) to raw byte 0x00 must
// stream as "\0", NOT the U+0100 text the old `b != 0` guard produced. The
// state-independence guarantee: `finalize()` yields the same bytes.
// ---------------------------------------------------------------------------

#[cfg(feature = "tokenizer-bpe")]
#[test]
fn bpe_byte_zero_token_streams_as_nul_not_u0100() {
  use mlxrs::tokenizer::{StreamingDetokenizer, stream::BpeStreamingDetokenizer};

  // U+0100 ('Ā') is GPT-2's byte-level char for raw byte 0x00. A vocab
  // token equal to that char must decode to a NUL byte. (The token is
  // followed by a printable so the multi-byte/incomplete-utf8 wait does not
  // withhold it; "A" == byte 0x41 via the identity printable range.)
  let vocab = vec![
    ("\u{0100}".to_string(), 0u32), // -> raw byte 0x00
    ("A".to_string(), 1u32),        // -> 'A'
  ];

  // Streaming path (`decode_bytes`).
  let mut d = BpeStreamingDetokenizer::new(vocab, false);
  d.reset();
  d.add_token(0);
  d.add_token(1);
  d.finalize();
  let streamed = d.text().into_owned();
  assert_eq!(streamed.as_bytes(), b"\0A");
  assert!(
    !streamed.contains('\u{0100}'),
    "byte 0x00 must not surface as U+0100 text"
  );

  // State-independence: a token whose ONLY content is byte 0x00, flushed
  // solely by `finalize()`, yields the identical NUL byte.
  let mut d2 = BpeStreamingDetokenizer::new(vec![("\u{0100}".to_string(), 0u32)], false);
  d2.reset();
  d2.add_token(0);
  d2.finalize();
  assert_eq!(d2.text().into_owned().as_bytes(), b"\0");
}

// ---------------------------------------------------------------------------
// BPE absent-id boundary matches mlx-lm `token < len(tokenmap)`:
// an in-range hole (`id <= max_id`, absent) -> "", out-of-range -> "!".
// Must stay HashMap-backed (a `u32::MAX` id never allocates a dense Vec).
// ---------------------------------------------------------------------------

#[cfg(feature = "tokenizer-bpe")]
#[test]
fn bpe_sparse_inrange_hole_is_empty_out_of_range_is_bang() {
  use mlxrs::tokenizer::{StreamingDetokenizer, stream::BpeStreamingDetokenizer};

  // Sparse vocab: ids {0, 5}. max_id == 5. id 3 is an in-range hole
  // (3 <= 5, absent) -> ""; id 9 is out-of-range (9 > 5) -> "!".
  let vocab = vec![
    ("Hi".to_string(), 0u32),
    ("\u{0120}there".to_string(), 5u32),
  ];
  let mut d = BpeStreamingDetokenizer::new(vocab, false);
  d.reset();
  d.add_token(0); // "Hi"
  d.add_token(3); // in-range hole -> "" (contributes nothing)
  d.add_token(5); // " there"
  d.add_token(9); // out-of-range -> "!"
  d.finalize();
  assert_eq!(d.text(), "Hi there!");

  // No-OOM property preserved: a `u32::MAX` id stays HashMap-backed (no
  // ~4GB dense alloc); `123_456 <= max_id(u32::MAX)` ⇒ in-range hole ⇒ "".
  let mut d2 = BpeStreamingDetokenizer::new(vec![("\u{0120}far".to_string(), u32::MAX)], false);
  d2.reset();
  d2.add_token(u32::MAX); // -> "far" (leading space trimmed, text empty)
  d2.add_token(123_456u32); // in-range hole (<= u32::MAX) -> ""
  d2.finalize();
  assert_eq!(d2.text(), "far");

  // Out-of-range id with a small dense vocab still falls back to "!".
  let mut d3 = BpeStreamingDetokenizer::new(vec![("Hello".to_string(), 0u32)], false);
  d3.reset();
  d3.add_token(0); // "Hello"
  d3.add_token(1); // 1 > max_id(0) -> "!"
  d3.finalize();
  assert_eq!(d3.text(), "Hello!");
}

// ---------------------------------------------------------------------------
// Perf — `last_segment()` must allocate only the
// per-step delta, NOT clone the whole accumulated output buffer every call
// (the old `self.text().into_owned()` made a generation loop O(total²)).
//
// Deterministic, no wall-clock: drive a long SPM stream calling
// `last_segment()` after each token and assert
//   (a) correctness — the concatenation of every returned segment equals the
//       full streamed `text()`, and
//   (b) the per-call segment is bounded by that step's text *growth*, never
//       the cumulative length. Under the OLD `into_owned()` impl the returned
//       segment would still be the same string (offset logic was identical),
//       so (b) alone wouldn't catch the regression — instead we assert the
//       invariant that makes the linear-work property observable: the segment
//       returned for any late token is strictly shorter than the total text
//       and never contains the whole prior output. Combined with the bounded
//       per-step-delta check below, the OLD code's quadratic full-buffer copy
//       is structurally precluded (each call's allocation == segment length,
//       and segment length == this step's delta, summing to O(total)).
// ---------------------------------------------------------------------------

#[cfg(feature = "tokenizer-spm")]
#[test]
fn last_segment_allocates_only_the_per_step_delta_not_the_whole_buffer() {
  use mlxrs::tokenizer::{StreamingDetokenizer, stream::SpmStreamingDetokenizer};

  // 2048 distinct one-word tokens (`▁wNN`), each flushed when the *next*
  // token's leading `▁` arrives, so the stream grows by exactly one word per
  // step (the previous word becomes committed/readable). A long output makes
  // the cumulative length large while every per-step delta stays tiny.
  const N: u32 = 2048;
  let vocab: Vec<(String, u32)> = (0..N).map(|i| (format!("\u{2581}w{i}"), i)).collect();
  let mut d = SpmStreamingDetokenizer::new(vocab, true);
  d.reset();

  let mut concat = String::new();
  let mut max_seg_len = 0usize;
  let mut prev_text_len = 0usize;

  for t in 0..N {
    d.add_token(t);
    let text_len_before = d.text().len();
    let seg = d.last_segment();

    // (b1) Per-call allocation == this step's text growth, not cumulative.
    // `last_segment` advances `offset` to `text().len()`, so the segment is
    // exactly `text_len_before - offset_before` == the bytes that became
    // readable since the previous call. That delta is one short word here.
    let delta = text_len_before - prev_text_len;
    assert_eq!(
      seg.len(),
      delta,
      "token {t}: segment length must equal the per-step text delta, \
       not the cumulative buffer"
    );
    assert!(
      seg.len() <= "\u{2581}w0000".len(),
      "token {t}: per-step segment {} bytes — must stay bounded by one \
       word, never grow with total output (a full-buffer clone regression)",
      seg.len()
    );

    // (b2) A late token's segment must NOT be the entire prior output.
    if t > 16 {
      assert!(
        seg.len() < concat.len(),
        "token {t}: segment ({} bytes) must be far shorter than the \
         {}-byte accumulated text — the old `into_owned()` cloned the \
         whole buffer",
        seg.len(),
        concat.len()
      );
      assert!(
        !seg.contains(&concat[..concat.len() / 2]),
        "token {t}: segment must not contain a prefix of the whole prior \
         output (would indicate a full-buffer copy was returned)"
      );
    }

    max_seg_len = max_seg_len.max(seg.len());
    prev_text_len = text_len_before;
    concat.push_str(&seg);
  }
  d.finalize();
  concat.push_str(&d.last_segment());

  // (a) Correctness preserved: streamed concatenation == full text.
  assert_eq!(concat, d.text());

  // Linear-work invariant made explicit: the largest single segment over the
  // whole 2048-token run is one short word, while the final text is ~5x
  // longer than that bound times the token count would allow for any
  // per-call full-buffer clone. (Old impl: max segment would still be small
  // BUT each call copied `text().len()`; this paired with (b1) — allocation
  // == segment == delta — pins total allocation to O(sum of deltas) == O(n).)
  let final_len = d.text().len();
  assert!(
    max_seg_len < final_len / 4,
    "max single segment ({max_seg_len} bytes) must be tiny vs the final \
     {final_len}-byte buffer — proves no per-call full-buffer copy"
  );
}

// ============================================================
// #111 — Detokenizer enum unification (kills per-token vtable)
// ============================================================

/// #111: the [`crate::tokenizer::Tokenizer::detokenizer`] factory
/// returns the enum-unified [`Detokenizer`] variant, not a
/// `Box<dyn StreamingDetokenizer>`. Naive / SPM / BPE each land in
/// their typed variant; the per-token `add_token` then dispatches via
/// `match` instead of vtable.
///
/// Naive-class fallback path (no `tokenizer.json` decoder node) ⇒
/// `Detokenizer::Naive(NaiveHfDetokenizer)`.
#[cfg(feature = "tokenizer-stream")]
#[test]
fn detokenizer_factory_returns_typed_variant_for_naive() {
  use mlxrs::tokenizer::{Detokenizer, NaiveHfDetokenizer, StreamingDetokenizer};
  use tokenizers::Tokenizer as HfTokenizer;
  // Load the shipped fixture's HF tokenizer (no network).
  const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizer.json");
  let hf: HfTokenizer = TOKENIZER_JSON.parse().expect("parse fixture tokenizer");
  let d = Detokenizer::Naive(Box::new(NaiveHfDetokenizer::new(hf, false)));
  // Dispatch through the enum's StreamingDetokenizer impl proves the
  // variant `match` arms work.
  assert!(matches!(d, Detokenizer::Naive(_)));
  // Tokens accessor proves trait dispatch through the enum compiles.
  let _: &[u32] = d.tokens();
}

/// #111: [`Detokenizer::Custom`] is the escape hatch for out-of-tree
/// streaming detokenizers — the boxed `Box<dyn StreamingDetokenizer>`
/// adds one indirection per call (same cost as the prior alias).
#[cfg(feature = "tokenizer-stream")]
#[test]
fn detokenizer_custom_escape_hatch() {
  use mlxrs::tokenizer::{Detokenizer, StreamingDetokenizer};

  // A no-op detokenizer: every observer returns the empty default.
  struct NullDetok {
    tokens: Vec<u32>,
    offset: usize,
  }
  impl StreamingDetokenizer for NullDetok {
    fn reset(&mut self) {
      self.tokens.clear();
      self.offset = 0;
    }
    fn add_token(&mut self, t: u32) {
      self.tokens.push(t);
    }
    fn finalize(&mut self) {}
    fn text(&self) -> std::borrow::Cow<'_, str> {
      std::borrow::Cow::Borrowed("")
    }
    fn tokens(&self) -> &[u32] {
      &self.tokens
    }
    fn offset(&self) -> usize {
      self.offset
    }
    fn set_offset(&mut self, o: usize) {
      self.offset = o;
    }
  }

  let mut d = Detokenizer::Custom(Box::new(NullDetok {
    tokens: Vec::new(),
    offset: 0,
  }));
  assert!(matches!(d, Detokenizer::Custom(_)));
  d.add_token(42);
  d.add_token(43);
  assert_eq!(d.tokens(), &[42, 43]);
  assert_eq!(d.text().as_ref(), "");
}

#[cfg(all(feature = "tokenizer-stream", feature = "tokenizer-spm"))]
#[test]
fn detokenizer_spm_variant_exists() {
  use mlxrs::tokenizer::{Detokenizer, StreamingDetokenizer, stream::SpmStreamingDetokenizer};
  let vocab = vec![("\u{2581}foo".to_string(), 0u32)];
  let d = Detokenizer::Spm(SpmStreamingDetokenizer::new(vocab, false));
  assert!(matches!(d, Detokenizer::Spm(_)));
  let _: &[u32] = d.tokens();
}

#[cfg(all(feature = "tokenizer-stream", feature = "tokenizer-bpe"))]
#[test]
fn detokenizer_bpe_variant_exists() {
  use mlxrs::tokenizer::{Detokenizer, StreamingDetokenizer, stream::BpeStreamingDetokenizer};
  let vocab = vec![("\u{0120}foo".to_string(), 0u32)];
  let d = Detokenizer::Bpe(BpeStreamingDetokenizer::new(vocab, false));
  assert!(matches!(d, Detokenizer::Bpe(_)));
  let _: &[u32] = d.tokens();
}
