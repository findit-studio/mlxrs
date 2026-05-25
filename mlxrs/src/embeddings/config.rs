//! `sentence-transformers` pooling-config parsing (`1_Pooling/config.json`).
//!
//! Ported from `mlx-embeddings` `utils._read_pooling_config` +
//! `models/pooling._normalize_pooling_config` (legacy `pooling_mode_*`
//! → `pooling_mode`) and swift `MLXEmbedders`
//! `Pooling.PoolingConfiguration` / `loadPooling` (the CLS > Mean > Max
//! > Last priority + `word_embedding_dimension` matryoshka dim).
//!
//! Reading the file off disk and the model-id registry are out of scope
//! (no-model-arch rule); this only parses already-obtained JSON
//! (path *or* in-memory bytes/str) into a [`PoolingStrategy`] +
//! `normalize` + `dimension` triple.
//!
//! The JSON parsing is **hand-rolled** (no `serde_json`): a
//! strict-subset recursive-descent scanner over `&[u8]` that handles the
//! ~10 known top-level keys `1_Pooling/config.json` exposes — the modern
//! `pooling_mode` string, the 6 legacy `pooling_mode_*` booleans, the
//! `include_prompt` boolean, and the matryoshka `word_embedding_dimension`
//! / `embedding_dimension` integers — tracking byte offsets for actionable
//! parse-error messages. Unknown keys are structurally validated but
//! discarded (forward-compat with future HF fields). Tracker entry
//! `EMB-1`. The full `serde_json` dep is therefore not pulled in by the
//! `embeddings` feature as a direct dependency; other tokenizer
//! features still pull `serde_json` independently and are unaffected.
//! `tokenizers v0.23` itself transitively pulls `serde_json` for
//! `tokenizer.json` parsing, so the `embeddings` build graph still
//! *compiles* `serde_json` — EMB-1 closes the *direct mlxrs surface*
//! claim, not the deep transitive graph.
//!
//! String payloads in the parser's `JVal::Str` variant are stored as
//! [`smol_str::SmolStr`] so the common short tokens (`"mean"` /
//! `"max"` / `"cls"` / `"pooling_mode"`) live inline in the JVal
//! (≤23 bytes), avoiding the per-token heap allocation `String` would
//! force.

use smol_str::SmolStr;
use std::path::Path;

use crate::error::{Error, Result};

use super::pooling::PoolingStrategy;

/// Upper bound on the on-disk size of a `1_Pooling/config.json` we will
/// read into memory. Real `sentence-transformers` pooling configs are a
/// handful of boolean flags plus a dimension — well under 1 KiB. The cap
/// is deliberately generous (1 MiB) yet still hard-bounds the allocation
/// (enforced via `Read::take(cap + 1)` on the opened handle, so even a
/// hostile / corrupt model directory that races a TOCTOU swap or streams
/// from a special file cannot drive an unbounded read into an OOM).
/// Exceeding it yields a recoverable [`Error::Backend`], not a panic,
/// and the over-cap body is never parsed.
///
/// Reading is additionally **non-blocking against a non-regular file**:
/// on Unix the open uses `O_NONBLOCK | O_CLOEXEC` so a FIFO planted at
/// the config path by an untrusted model dir returns from `open()`
/// immediately (no indefinite wait for a writer). Symlinks **are**
/// followed (HuggingFace Hub caches store `snapshots/<rev>/1_Pooling/
/// config.json` as a symlink into `blobs/<hash>` — the dominant real
/// cached-model layout — so refusing symlinks would break normal cached
/// models). Safety does not rely on refusing symlinks: the opened
/// handle is `fstat`ed and rejected via `metadata().is_file()` **before
/// any read** (this stats the *resolved target* of any symlink, so a
/// symlink → FIFO/device/directory is still rejected), and a
/// non-blocking open prevents a symlink → FIFO from hanging. The only
/// guarantees a caller relies on (no hang, no unbounded read, no panic,
/// recoverable error) hold for FIFOs/devices/directories and for
/// symlinks to any of them too.
const MAX_ST_POOLING_CONFIG_BYTES: u64 = 1 << 20;

/// Legacy `sentence-transformers` boolean flag → mode name. Mirrors
/// python `_LEGACY_POOLING_MODE_KWARGS`.
const LEGACY_KEYS: &[(&str, &str)] = &[
  ("pooling_mode_cls_token", "cls"),
  ("pooling_mode_max_tokens", "max"),
  ("pooling_mode_mean_tokens", "mean"),
  ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len_tokens"),
  ("pooling_mode_weightedmean_tokens", "weightedmean"),
  ("pooling_mode_lasttoken", "lasttoken"),
];

/// Parsed `1_Pooling/config.json` → pooling pipeline parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StPoolingConfig {
  /// Resolved pooling strategy.
  strategy: PoolingStrategy,
  /// Whether the embeddings should be L2-normalized after pooling. ST
  /// configs don't carry a normalize flag, so this is always `true`
  /// (the `mlx-embeddings` / `MLXEmbedders` convention is to normalize),
  /// surfaced explicitly so the caller can override.
  normalize: bool,
  /// Matryoshka output dimension (`word_embedding_dimension`), if the
  /// config declares one.
  dimension: Option<usize>,
}

impl StPoolingConfig {
  /// Construct a [`StPoolingConfig`] from its three components.
  pub fn new(strategy: PoolingStrategy, normalize: bool, dimension: Option<usize>) -> Self {
    Self {
      strategy,
      normalize,
      dimension,
    }
  }

  /// The resolved pooling strategy.
  #[inline(always)]
  pub fn strategy(&self) -> PoolingStrategy {
    self.strategy
  }

  /// Whether the embeddings should be L2-normalized after pooling.
  #[inline(always)]
  pub fn normalize(&self) -> bool {
    self.normalize
  }

  /// Matryoshka output dimension, if the config declares one.
  #[inline(always)]
  pub fn dimension(&self) -> Option<usize> {
    self.dimension
  }
}

// ───────────────── hand-rolled strict-JSON scanner ────────────────────
//
// EMB-1: parse the ~10 known top-level keys `1_Pooling/config.json` uses
// (modern `pooling_mode` + 6 legacy boolean flags + `include_prompt` +
// `word_embedding_dimension` + `embedding_dimension`) without pulling
// `serde_json` into the `embeddings`-only feature. Unknown keys are
// structurally validated but their values are discarded so a hostile
// config with many unknown keys parses in O(n).
// Strict JSON subset (RFC 8259 surface for what HF ST configs emit):
//
//   value    := object | array | string | number | "true" | "false" | "null"
//   object   := "{" ws (pair (ws "," ws pair)*)? ws "}"
//   pair     := string ws ":" ws value
//   array    := "[" ws (value (ws "," ws value)*)? ws "]"
//   string   := "\"" (char | escape)* "\""
//   number   := int frac? exp?
//
// We only need to *interpret* the top-level object's pairs; nested
// arrays/objects/strings inside a value are scanned to find the end of
// the value but their contents are retained only to the extent the
// caller needs (we distinguish "is a list" for `pooling_mode` so we can
// reject concatenated modes with a python-faithful message; nested
// objects under `pooling_mode` are flagged as such).
//
// Rich error messages: every parse error reports a byte offset relative
// to the input start (line/column would require a second scan; the
// offset is actionable for any tool/editor that can map byte→position).

/// Tagged value the scanner emits for the schema's known keys. Only
/// the fields the pooling-config resolver inspects carry their
/// payload; nested arrays and objects are reduced to a presence tag
/// because the schema never inspects their interior (only "is
/// `pooling_mode` a list?" to surface a python-faithful rejection).
///
/// Unknown top-level keys never reach this enum — they are discarded
/// during parsing rather than accumulated in a `Vec`, which keeps the
/// scanner O(n) on hostile configs that pack many unique keys into the
/// 1 MiB read cap (Codex EMB-1 round 2: the previous `Vec`-of-pairs
/// dedup was O(n²) on such input).
#[derive(Debug)]
pub(crate) enum JVal<'a> {
  Null,
  Bool(bool),
  /// String value; the backing payload is the decoded UTF-8 bytes (any
  /// `\u00XX` / `\n` / `\"` escapes already resolved). [`SmolStr`]
  /// inlines payloads ≤23 bytes so the common short pooling tokens
  /// (`"mean"` / `"cls"` / `"pooling_mode"`) live in the JVal without
  /// a heap allocation.
  Str(SmolStr),
  /// Number, preserved as its source bytes so the caller can distinguish
  /// integer / fractional / overflow without re-tokenizing.
  Num(&'a str),
  /// Array; contents skipped (we only need to flag presence as "list").
  Array,
  /// Object; contents skipped.
  Object,
}

/// The closed set of top-level keys the pooling-config resolver
/// consults. Any other key in the input is *parsed for structural
/// validity* (so we still reject malformed JSON) but its value is
/// discarded rather than accumulated — bounding the per-key cost at
/// O(1) regardless of how many ignored keys a hostile input packs into
/// the 1 MiB read cap.
const KNOWN_KEYS: &[&str] = &[
  "pooling_mode",
  "pooling_mode_cls_token",
  "pooling_mode_max_tokens",
  "pooling_mode_mean_tokens",
  "pooling_mode_mean_sqrt_len_tokens",
  "pooling_mode_weightedmean_tokens",
  "pooling_mode_lasttoken",
  "include_prompt",
  "word_embedding_dimension",
  "embedding_dimension",
];

/// Cheap check used during parsing to decide whether to retain a key's
/// value. Hardcoded against [`KNOWN_KEYS`] so adding a new field is a
/// one-line edit there.
fn is_known_key(k: &str) -> bool {
  KNOWN_KEYS.contains(&k)
}

/// Hard cap on nested object/array depth the scanner will accept. A 1
/// MiB-capped input can otherwise pack hundreds of thousands of `[` /
/// `{` characters at a single position; without a depth guard
/// `parse_value` ↔ `skip_object`/`skip_array` would recurse once per
/// `[`/`{` and overflow the thread stack on hostile model data
/// (turning a malformed `1_Pooling/config.json` into an abort instead
/// of a recoverable [`Error::Backend`]). 128 levels covers every
/// realistic HF config — those pooling configs are flat — yet caps
/// stack growth at a constant well under any platform default.
///
/// Codex adversarial-review (EMB-1, round 1) flagged the unbounded
/// recursion as the only blocker. The fix is a per-scanner depth
/// counter incremented at every `{`/`[` and decremented at the
/// matching close; if it exceeds the cap we surface a recoverable
/// error citing the byte offset.
const MAX_NESTING_DEPTH: usize = 128;

/// Byte-offset-tracking scanner over the JSON input.
struct Scanner<'a> {
  src: &'a [u8],
  pos: usize,
  /// Current nested container depth (incremented on each `{`/`[`, decremented
  /// on the matching close). Bounded by [`MAX_NESTING_DEPTH`] to prevent
  /// stack-overflow aborts on hostile config input.
  depth: usize,
}

impl<'a> Scanner<'a> {
  fn new(src: &'a [u8]) -> Self {
    Self {
      src,
      pos: 0,
      depth: 0,
    }
  }

  /// Increment the nesting depth; error (no panic) if the cap is
  /// reached. Pair every successful call with [`Scanner::leave`].
  fn enter(&mut self) -> Result<()> {
    if self.depth >= MAX_NESTING_DEPTH {
      return Err(self.err(format!(
        "nested object/array depth exceeds the {MAX_NESTING_DEPTH}-level cap; \
         refusing to recurse further (defends against stack-overflow on \
         hostile pooling config input)"
      )));
    }
    self.depth += 1;
    Ok(())
  }

  fn leave(&mut self) {
    // Underflow is unreachable in correct call paths (every `leave`
    // pairs with a successful `enter`), but defend against it without a
    // panic so a future refactor cannot turn a logic bug into UB.
    self.depth = self.depth.saturating_sub(1);
  }

  fn err(&self, msg: impl Into<String>) -> Error {
    Error::Backend {
      message: format!(
        "invalid pooling config JSON at byte {}: {}",
        self.pos,
        msg.into()
      ),
    }
  }

  fn peek(&self) -> Option<u8> {
    self.src.get(self.pos).copied()
  }

  fn bump(&mut self) -> Option<u8> {
    let b = self.peek()?;
    self.pos += 1;
    Some(b)
  }

  fn skip_ws(&mut self) {
    // SIMD-accelerated (AVX-512BW / AVX2 / SSE4.2 / NEON dispatch).
    // `memspan::skip::skip_whitespace` matches RFC 8259 §2 byte-for-byte
    // (space, tab, LF, CR — verified `memspan::skip::is_whitespace`
    // pattern at `memspan/src/skip/mod.rs`); inputs <16 bytes (NEON
    // chunk size) or <32 bytes (AVX threshold) fall back to the
    // scalar `prefix_len_whitespace` loop so the small-input cost is
    // unchanged. Called between every token, so the per-call win
    // compounds across the whole parse.
    self.pos += memspan::skip::skip_whitespace(&self.src[self.pos..]);
  }

  /// Expect the next non-ws byte to equal `b`, consume it; else error.
  fn expect(&mut self, b: u8, ctx: &str) -> Result<()> {
    self.skip_ws();
    match self.peek() {
      Some(c) if c == b => {
        self.pos += 1;
        Ok(())
      }
      Some(c) => Err(self.err(format!(
        "expected {:?} {ctx} but found {:?}",
        b as char, c as char
      ))),
      None => Err(self.err(format!(
        "expected {:?} {ctx} but reached end of input",
        b as char
      ))),
    }
  }

  /// Parse the top-level object into `(key, value)` pairs for the
  /// schema's known keys ([`KNOWN_KEYS`]). Duplicate known keys: the
  /// *last* wins (mirrors `serde_json` default behavior, which the
  /// previous parser relied on). Unknown keys are parsed for structural
  /// validity (so we still reject malformed JSON downstream of an
  /// ignored field) but their values are *discarded* rather than
  /// retained — keeping per-key cost O(1) and the whole top-level pass
  /// O(n), regardless of how many unique unknown keys a hostile config
  /// packs into the 1 MiB read cap (Codex EMB-1 round 2 fix).
  ///
  /// The top-level `{` counts toward [`MAX_NESTING_DEPTH`] (`enter`/`leave`
  /// pair) so a config of `{"k": <nested>}` shares the same depth budget
  /// as the same shape with a deeper key.
  fn parse_top_object(&mut self) -> Result<Vec<(SmolStr, JVal<'a>)>> {
    self.skip_ws();
    self.expect(b'{', "at start of pooling config")?;
    self.enter()?;
    // Capacity ≤ KNOWN_KEYS.len() (any unknown key is dropped). Pre-
    // allocate that worst case once so we never grow.
    let mut out: Vec<(SmolStr, JVal<'a>)> = Vec::with_capacity(KNOWN_KEYS.len());
    self.skip_ws();
    if self.peek() == Some(b'}') {
      self.pos += 1;
      self.leave();
      return Ok(out);
    }
    loop {
      self.skip_ws();
      let key = self.parse_string("for object key")?;
      self.expect(b':', &format!("after key {key:?}"))?;
      self.skip_ws();
      if is_known_key(&key) {
        let val = self.parse_value(&format!("for value of key {key:?}"))?;
        // Last-wins on duplicate KNOWN keys. `out` is ≤ KNOWN_KEYS.len()
        // (a fixed small constant), so this linear scan is bounded
        // O(KNOWN_KEYS) per insertion, not O(n) on input size.
        if let Some(idx) = out
          .iter()
          .position(|(k, _): &(SmolStr, JVal<'_>)| k == &key)
        {
          out.swap_remove(idx);
        }
        out.push((key, val));
      } else {
        // Unknown key — structurally validate the value (to advance
        // `pos` past it) WITHOUT allocating: `skip_value` walks strings/
        // arrays/objects without materializing a `SmolStr` payload or
        // descending into `parse_string`'s escape-decoding allocator.
        // A hostile config with 100k unique unknown keys (each carrying
        // a long string value) therefore parses in O(n) with zero per-
        // key heap traffic, instead of O(n) heap allocations.
        // Copilot review #3277203298 (EMB-1 follow-up).
        self.skip_value(&format!("for value of key {key:?}"))?;
      }
      self.skip_ws();
      match self.peek() {
        Some(b',') => {
          self.pos += 1;
          // Strict JSON: trailing comma is rejected. RFC 8259 §4.
          self.skip_ws();
          if self.peek() == Some(b'}') {
            return Err(self.err("trailing comma before `}` is not valid JSON"));
          }
        }
        Some(b'}') => {
          self.pos += 1;
          self.leave();
          return Ok(out);
        }
        Some(c) => {
          return Err(self.err(format!(
            "expected `,` or `}}` in object but found {:?}",
            c as char
          )));
        }
        None => return Err(self.err("expected `,` or `}` in object but reached end of input")),
      }
    }
  }

  /// Parse a JSON value. Nested arrays/objects/strings are scanned to
  /// their end but their contents are discarded (we only need top-level
  /// pairs for the pooling-config schema).
  fn parse_value(&mut self, ctx: &str) -> Result<JVal<'a>> {
    self.skip_ws();
    match self.peek() {
      Some(b'"') => Ok(JVal::Str(self.parse_string(ctx)?)),
      Some(b'{') => {
        self.skip_object()?;
        Ok(JVal::Object)
      }
      Some(b'[') => {
        self.skip_array()?;
        Ok(JVal::Array)
      }
      Some(b't') | Some(b'f') => Ok(JVal::Bool(self.parse_bool(ctx)?)),
      Some(b'n') => {
        self.parse_keyword("null", ctx)?;
        Ok(JVal::Null)
      }
      Some(c) if c == b'-' || c.is_ascii_digit() => Ok(JVal::Num(self.parse_number_slice(ctx)?)),
      Some(c) => Err(self.err(format!(
        "unexpected character {:?} while parsing value {ctx}",
        c as char
      ))),
      None => Err(self.err(format!("unexpected end of input while parsing value {ctx}"))),
    }
  }

  /// Structurally validate the value at the current position WITHOUT
  /// materializing its payload. Used for unknown-key values + every
  /// `skip_object` / `skip_array` interior value, so the per-token
  /// allocator never fires for content the caller will immediately drop.
  ///
  /// Mirrors [`parse_value`]'s dispatch exactly: strings → `skip_string`
  /// (validates escapes, advances `pos` past the closing `"`, returns
  /// `()`); arrays/objects → existing `skip_array` / `skip_object`
  /// (already non-materializing for top-level container shape, but
  /// their interior keys/values previously fell back into `parse_value`
  /// — fixed in this PR); booleans/numbers/null → existing
  /// `parse_bool` / `parse_number_slice` / `parse_keyword` (which
  /// already only advance `pos` without allocating).
  ///
  /// Copilot review #3277203298 (EMB-1 follow-up).
  fn skip_value(&mut self, ctx: &str) -> Result<()> {
    self.skip_ws();
    match self.peek() {
      Some(b'"') => self.skip_string(ctx),
      Some(b'{') => self.skip_object(),
      Some(b'[') => self.skip_array(),
      Some(b't') | Some(b'f') => {
        self.parse_bool(ctx)?;
        Ok(())
      }
      Some(b'n') => self.parse_keyword("null", ctx),
      Some(c) if c == b'-' || c.is_ascii_digit() => {
        self.parse_number_slice(ctx)?;
        Ok(())
      }
      Some(c) => Err(self.err(format!(
        "unexpected character {:?} while skipping value {ctx}",
        c as char
      ))),
      None => Err(self.err(format!(
        "unexpected end of input while skipping value {ctx}"
      ))),
    }
  }

  /// Validate a JSON string literal at the current position (which MUST
  /// be on a `"`, after `skip_ws`), advancing `pos` past the closing
  /// `"`. Returns `()` — used for discarded keys/values where the
  /// decoded payload is not needed.
  ///
  /// Escapes (`\"`, `\\`, `\uXXXX`) MUST still be walked to know where
  /// the string ends — `\"` does NOT close the string — but their
  /// decoded payload is dropped. UTF-16 surrogate pairs are validated
  /// for well-formedness (mirrors `parse_string`'s reject-on-unpaired
  /// behavior so a hostile config that would *fail* the decode in a
  /// known-key value is consistently rejected in an unknown-key value
  /// too). Control-character + invalid-escape rejections mirror
  /// `parse_string` exactly.
  ///
  /// Copilot review #3277203298 (EMB-1 follow-up).
  fn skip_string(&mut self, ctx: &str) -> Result<()> {
    self.skip_ws();
    if self.peek() != Some(b'"') {
      return Err(self.err(format!("expected string {ctx}")));
    }
    let open_pos = self.pos;
    self.pos += 1; // opening quote
    loop {
      // SIMD-scan to the next `"` or `\`. Same dispatch as `parse_string`.
      let tail = &self.src[self.pos..];
      let n = memspan::skip::skip_until(tail, *b"\"\\").ok_or_else(|| Error::Backend {
        message: format!(
          "invalid pooling config JSON: unterminated string starting at byte {open_pos}"
        ),
      })?;
      // Reject control characters in the chunk we just skipped (RFC 8259 §7).
      let chunk = &tail[..n];
      if let Some(bad) = chunk.iter().position(|&b| b < 0x20) {
        return Err(Error::Backend {
          message: format!(
            "invalid pooling config JSON at byte {}: control character 0x{:02X} in string body {ctx} (RFC 8259 §7)",
            self.pos + bad,
            chunk[bad]
          ),
        });
      }
      self.pos += n;
      let boundary = match self.bump() {
        Some(b) => b,
        None => unreachable!("skip_until found a needle so self.bump cannot be None"),
      };
      if boundary == b'"' {
        return Ok(());
      }
      // boundary == b'\\' — walk one escape (discarded) and keep scanning.
      let esc = match self.bump() {
        Some(b) => b,
        None => {
          return Err(Error::Backend {
            message: format!(
              "invalid pooling config JSON: unterminated escape in string starting at byte {open_pos}"
            ),
          });
        }
      };
      match esc {
        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
        b'u' => {
          let cp = self.parse_unicode_escape()?;
          // Reject unpaired surrogates symmetrically with `parse_string`;
          // well-formed surrogate pairs are walked (and dropped) so a
          // hostile input that would fail decode in a kept value also
          // fails in a dropped one.
          if (0xD800..=0xDBFF).contains(&cp) {
            if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
              return Err(self.err("expected low surrogate (`\\uDCxx`) after high surrogate"));
            }
            let low = self.parse_unicode_escape()?;
            if !(0xDC00..=0xDFFF).contains(&low) {
              return Err(self.err(format!("expected low surrogate but got U+{low:04X}")));
            }
          } else if (0xDC00..=0xDFFF).contains(&cp) {
            return Err(self.err(format!("unpaired low surrogate U+{cp:04X}")));
          } else if char::from_u32(cp).is_none() {
            return Err(self.err(format!("invalid Unicode codepoint U+{cp:X}")));
          }
        }
        other => {
          return Err(self.err(format!(
            "invalid escape sequence `\\{}` in string {ctx}",
            other as char
          )));
        }
      }
    }
  }

  /// Parse a JSON string literal starting at the current position
  /// (which MUST be on a `"`, after `skip_ws`). Returns the decoded
  /// UTF-8 string.
  ///
  /// Two-stage implementation (memspan-accelerated; closes Codex
  /// EMB-1 round-4 perf finding):
  ///
  /// 1. **Fast path — escape-free strings**: SIMD-scan ahead for the
  ///    next `"` or `\` via `memspan::skip::skip_until`. If the first
  ///    needle is `"`, the string body is escape-free — borrow the
  ///    source slice directly, scalar-scan it once for control chars
  ///    (LLVM auto-vec), `str::from_utf8` it, and hand the slice to
  ///    `SmolStr::new` with **zero intermediate `Vec` allocation**.
  ///    This is the overwhelming-common case for pooling configs
  ///    (every known string value — `"mean"`, `"max"`, `"cls"`,
  ///    `"pooling_mode"` — and every unknown-key value that isn't
  ///    deliberately escape-encoded).
  ///
  /// 2. **Slow path — escapes present**: fall back to a `Vec<u8>`
  ///    decode buffer, but still use `memspan::skip::skip_until` to
  ///    fast-forward to the next escape boundary in one SIMD scan
  ///    plus one bulk `extend_from_slice` per text chunk, instead of
  ///    the previous per-byte loop. Each chunk is scalar-scanned
  ///    for control chars; escapes (`\n`, `\\`, `\uXXXX`, surrogate
  ///    pairs) are decoded byte-by-byte at the boundary itself.
  ///
  /// Closing-quote UTF-8 validation is the single trailing pass over
  /// the decoded bytes (raw multibyte UTF-8 in the source passes
  /// through unchanged; the validator catches any malformed runs).
  fn parse_string(&mut self, ctx: &str) -> Result<SmolStr> {
    self.skip_ws();
    if self.peek() != Some(b'"') {
      return Err(self.err(format!("expected string {ctx}")));
    }
    let open_pos = self.pos;
    self.pos += 1; // opening quote
    let body_start = self.pos;
    let tail = &self.src[body_start..];

    // SIMD-scan for the first `"` or `\`. `memspan::skip::skip_until`
    // dispatches to AVX-512BW / AVX2 / SSE4.2 / NEON / scalar.
    let first = memspan::skip::skip_until(tail, *b"\"\\");

    match first {
      Some(n) if tail[n] == b'"' => {
        // Fast path: escape-free string body of `n` bytes.
        let body = &tail[..n];
        // Reject any control character (RFC 8259 §7). The scan is
        // scalar but bounded to the chunk we already know ends at
        // the closing quote, and LLVM auto-vectorizes the per-byte
        // `< 0x20` test.
        if let Some(bad) = body.iter().position(|&b| b < 0x20) {
          return Err(Error::Backend {
            message: format!(
              "invalid pooling config JSON at byte {}: control character 0x{:02X} in string body {ctx} (RFC 8259 §7)",
              body_start + bad,
              body[bad]
            ),
          });
        }
        let s = std::str::from_utf8(body)
          .map_err(|e| self.err(format!("string is not valid UTF-8 {ctx}: {e}")))?;
        let out = SmolStr::new(s);
        self.pos = body_start + n + 1; // skip past the closing quote
        Ok(out)
      }
      Some(_) => {
        // Slow path: there's at least one `\` before the closing `"`.
        self.parse_string_with_escapes(ctx, open_pos)
      }
      None => Err(Error::Backend {
        message: format!(
          "invalid pooling config JSON: unterminated string starting at byte {open_pos}"
        ),
      }),
    }
  }

  /// Slow-path string decoder used when [`parse_string`]'s SIMD pre-scan
  /// found a `\` before the closing `"`. Walks chunk-by-chunk: each
  /// iteration SIMD-scans to the next escape boundary, bulk-copies the
  /// text chunk into the decode buffer, then handles ONE escape at the
  /// boundary before looping. Control-character rejection is per-chunk.
  fn parse_string_with_escapes(&mut self, ctx: &str, open_pos: usize) -> Result<SmolStr> {
    let mut buf: Vec<u8> = Vec::new();
    loop {
      // SIMD-scan from the current `pos` to the next `"` or `\`.
      let tail = &self.src[self.pos..];
      let n = memspan::skip::skip_until(tail, *b"\"\\").ok_or_else(|| Error::Backend {
        message: format!(
          "invalid pooling config JSON: unterminated string starting at byte {open_pos}"
        ),
      })?;
      let chunk = &tail[..n];
      if let Some(bad) = chunk.iter().position(|&b| b < 0x20) {
        return Err(Error::Backend {
          message: format!(
            "invalid pooling config JSON at byte {}: control character 0x{:02X} in string body {ctx} (RFC 8259 §7)",
            self.pos + bad,
            chunk[bad]
          ),
        });
      }
      buf.extend_from_slice(chunk);
      self.pos += n;
      // Boundary byte is `"` or `\`; consume it.
      let boundary = match self.bump() {
        Some(b) => b,
        None => unreachable!("skip_until found a needle so self.bump cannot be None"),
      };
      if boundary == b'"' {
        // Closing quote.
        return std::str::from_utf8(&buf)
          .map(SmolStr::new)
          .map_err(|e| self.err(format!("string is not valid UTF-8 {ctx}: {e}")));
      }
      // boundary == b'\\' — decode one escape and keep looping.
      let esc = match self.bump() {
        Some(b) => b,
        None => {
          return Err(Error::Backend {
            message: format!(
              "invalid pooling config JSON: unterminated escape in string starting at byte {open_pos}"
            ),
          });
        }
      };
      match esc {
        b'"' => buf.push(b'"'),
        b'\\' => buf.push(b'\\'),
        b'/' => buf.push(b'/'),
        b'b' => buf.push(0x08),
        b'f' => buf.push(0x0C),
        b'n' => buf.push(b'\n'),
        b'r' => buf.push(b'\r'),
        b't' => buf.push(b'\t'),
        b'u' => {
          let cp = self.parse_unicode_escape()?;
          let c = if (0xD800..=0xDBFF).contains(&cp) {
            if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
              return Err(self.err("expected low surrogate (`\\uDCxx`) after high surrogate"));
            }
            let low = self.parse_unicode_escape()?;
            if !(0xDC00..=0xDFFF).contains(&low) {
              return Err(self.err(format!("expected low surrogate but got U+{low:04X}")));
            }
            let combined = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            char::from_u32(combined)
              .ok_or_else(|| self.err(format!("invalid surrogate-pair codepoint U+{combined:X}")))?
          } else if (0xDC00..=0xDFFF).contains(&cp) {
            return Err(self.err(format!("unpaired low surrogate U+{cp:04X}")));
          } else {
            char::from_u32(cp)
              .ok_or_else(|| self.err(format!("invalid Unicode codepoint U+{cp:X}")))?
          };
          let mut tmp = [0u8; 4];
          buf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
        }
        other => {
          return Err(self.err(format!(
            "invalid escape sequence `\\{}` in string {ctx}",
            other as char
          )));
        }
      }
    }
  }

  /// Parse a `\uXXXX` Unicode escape (the `\u` has already been
  /// consumed). Returns the raw 16-bit value.
  fn parse_unicode_escape(&mut self) -> Result<u32> {
    let mut cp: u32 = 0;
    for _ in 0..4 {
      let b = self
        .bump()
        .ok_or_else(|| self.err("incomplete `\\uXXXX` escape"))?;
      let nib = match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        other => {
          return Err(self.err(format!(
            "invalid hex digit {:?} in `\\uXXXX` escape",
            other as char
          )));
        }
      };
      cp = (cp << 4) | u32::from(nib);
    }
    Ok(cp)
  }

  /// Parse `true` or `false`.
  fn parse_bool(&mut self, ctx: &str) -> Result<bool> {
    match self.peek() {
      Some(b't') => {
        self.parse_keyword("true", ctx)?;
        Ok(true)
      }
      Some(b'f') => {
        self.parse_keyword("false", ctx)?;
        Ok(false)
      }
      Some(c) => Err(self.err(format!("expected bool {ctx} but found {:?}", c as char))),
      None => Err(self.err(format!("expected bool {ctx} but reached end of input"))),
    }
  }

  /// Consume an exact ASCII keyword; error if the next bytes don't match.
  fn parse_keyword(&mut self, kw: &str, ctx: &str) -> Result<()> {
    let bytes = kw.as_bytes();
    if self.pos + bytes.len() > self.src.len() {
      return Err(self.err(format!(
        "expected keyword `{kw}` {ctx} but reached end of input"
      )));
    }
    if &self.src[self.pos..self.pos + bytes.len()] != bytes {
      return Err(self.err(format!("expected keyword `{kw}` {ctx}")));
    }
    self.pos += bytes.len();
    Ok(())
  }

  /// Parse a JSON number, returning the source slice so the caller can
  /// later interpret it (integer / float / overflow). Validation here
  /// is structural only — value semantics live with the caller.
  fn parse_number_slice(&mut self, ctx: &str) -> Result<&'a str> {
    let start = self.pos;
    if self.peek() == Some(b'-') {
      self.pos += 1;
    }
    // Integer part.
    match self.peek() {
      Some(b'0') => {
        self.pos += 1;
      }
      Some(c) if c.is_ascii_digit() => {
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
          self.pos += 1;
        }
      }
      Some(c) => return Err(self.err(format!("invalid number {ctx}: unexpected {:?}", c as char))),
      None => return Err(self.err(format!("invalid number {ctx}: reached end of input"))),
    }
    // Optional fraction.
    if self.peek() == Some(b'.') {
      self.pos += 1;
      let frac_start = self.pos;
      while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
        self.pos += 1;
      }
      if self.pos == frac_start {
        return Err(self.err(format!("invalid number {ctx}: expected digit after `.`")));
      }
    }
    // Optional exponent.
    if matches!(self.peek(), Some(b'e') | Some(b'E')) {
      self.pos += 1;
      if matches!(self.peek(), Some(b'+') | Some(b'-')) {
        self.pos += 1;
      }
      let exp_start = self.pos;
      while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
        self.pos += 1;
      }
      if self.pos == exp_start {
        return Err(self.err(format!(
          "invalid number {ctx}: expected digit after exponent marker"
        )));
      }
    }
    // ASCII-only by construction (digits, sign, `.`, `e`/`E`), so direct
    // utf-8 borrow is sound.
    Ok(
      std::str::from_utf8(&self.src[start..self.pos])
        .expect("number bytes are ASCII by construction"),
    )
  }

  /// Skip a JSON object (the opening `{` is at the current position).
  /// Validates structure but discards contents.
  ///
  /// Bounded against malicious deep nesting: `enter`/`leave` increments
  /// the scanner's depth counter so the recursion `skip_object` →
  /// `parse_value` → `skip_object` cannot blow the stack on hostile
  /// input within the 1 MiB read cap (e.g. `{"k":{"k":{...}}}` packed
  /// to a million levels). [`MAX_NESTING_DEPTH`].
  fn skip_object(&mut self) -> Result<()> {
    self.expect(b'{', "at start of nested object")?;
    self.enter()?;
    self.skip_ws();
    if self.peek() == Some(b'}') {
      self.pos += 1;
      self.leave();
      return Ok(());
    }
    loop {
      self.skip_ws();
      // Allocation-free key + value walk (Copilot review #3277203298):
      // a discarded nested object's string keys and string values would
      // otherwise allocate a fresh `String` per token even though we
      // immediately drop them. `skip_string` / `skip_value` validate
      // structure (and `skip_string` still decodes escapes far enough
      // to know where the closing `"` is) without materializing the
      // payload.
      self.skip_string("for nested object key")?;
      self.expect(b':', "after nested object key")?;
      self.skip_ws();
      self.skip_value("inside nested object")?;
      self.skip_ws();
      match self.peek() {
        Some(b',') => {
          self.pos += 1;
          self.skip_ws();
          if self.peek() == Some(b'}') {
            return Err(self.err("trailing comma before `}` is not valid JSON"));
          }
        }
        Some(b'}') => {
          self.pos += 1;
          self.leave();
          return Ok(());
        }
        Some(c) => {
          return Err(self.err(format!(
            "expected `,` or `}}` in nested object but found {:?}",
            c as char
          )));
        }
        None => {
          return Err(self.err("expected `,` or `}` in nested object but reached end of input"));
        }
      }
    }
  }

  /// Skip a JSON array (the opening `[` is at the current position).
  ///
  /// Same nesting-depth guard as [`Scanner::skip_object`]; defends
  /// against hostile `[[[[...]]]]` packed into the 1 MiB cap.
  fn skip_array(&mut self) -> Result<()> {
    self.expect(b'[', "at start of array")?;
    self.enter()?;
    self.skip_ws();
    if self.peek() == Some(b']') {
      self.pos += 1;
      self.leave();
      return Ok(());
    }
    loop {
      self.skip_ws();
      // Allocation-free (Copilot review #3277203298) — see `skip_object`.
      self.skip_value("inside array")?;
      self.skip_ws();
      match self.peek() {
        Some(b',') => {
          self.pos += 1;
          self.skip_ws();
          if self.peek() == Some(b']') {
            return Err(self.err("trailing comma before `]` is not valid JSON"));
          }
        }
        Some(b']') => {
          self.pos += 1;
          self.leave();
          return Ok(());
        }
        Some(c) => {
          return Err(self.err(format!(
            "expected `,` or `]` in array but found {:?}",
            c as char
          )));
        }
        None => return Err(self.err("expected `,` or `]` in array but reached end of input")),
      }
    }
  }
}

/// Parse a strict-JSON `1_Pooling/config.json` body into the
/// `(key, value)` pairs at the top-level object.
///
/// Public to the crate so embeddings tests can target the scanner
/// edge cases directly without going through `pooling_from_st_config_*`.
pub(crate) fn parse_pooling_json(src: &str) -> Result<Vec<(SmolStr, JVal<'_>)>> {
  let mut scanner = Scanner::new(src.as_bytes());
  let out = scanner.parse_top_object()?;
  scanner.skip_ws();
  if scanner.pos != scanner.src.len() {
    return Err(scanner.err("trailing data after top-level object"));
  }
  Ok(out)
}

// ───────────────── caller-side value extraction ─────────────────

/// Look up a key in the parsed pairs, returning the *last* match (to
/// mirror `serde_json`'s last-wins dup behavior the previous parser
/// relied on; the scanner already deduplicates so this is just a
/// pass-through lookup).
fn find<'a, 'b>(cfg: &'a [(SmolStr, JVal<'b>)], key: &str) -> Option<&'a JVal<'b>> {
  cfg.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn resolve_strategy(cfg: &[(SmolStr, JVal<'_>)]) -> Result<PoolingStrategy> {
  // Modern key wins if present (python `pool_by_config` uses
  // `cfg["pooling_mode"]` directly when set).
  if let Some(JVal::Str(mode)) = find(cfg, "pooling_mode") {
    return PoolingStrategy::from_mode(mode);
  }
  if let Some(JVal::Array) = find(cfg, "pooling_mode") {
    return Err(Error::Backend {
      message: "concatenated pooling mode (list) is not supported; \
                only a single pooling mode is allowed"
        .into(),
    });
  }
  // C6 (Copilot review 4307622782, #3256688299): a present-but-non-
  // string/non-array `pooling_mode` (null / bool / number / object).
  //
  // python parity: `_normalize_pooling_config` only synthesizes
  // `pooling_mode` from legacy flags; with `pooling_mode` already present
  // it leaves the value as-is, then `pool_by_config` does
  // `mode = cfg["pooling_mode"]` and — since `None`/`False`/a number is
  // neither a tuple/list, nor a known-unsupported string, nor any of the
  // `if mode == "cls"/...` branches — falls through to
  // `raise ValueError(f"Unknown pooling mode {mode!r}...")`
  // (`models/pooling.py`; `tests/test_pooling.py::test_invalid_mode_-
  // raises` pins the analogous unknown-string path). python therefore
  // REJECTS a present-but-wrong-typed `pooling_mode`; it does NOT silently
  // fall back to legacy/Mean. mlxrs previously fell through to the legacy
  // path (silent Mean) — a divergence AND a silent-wrong-embedding (the
  // model author set `pooling_mode`; honoring it as a different strategy
  // is silently wrong). Reject with a recoverable `Err` to match python.
  if let Some(v) = find(cfg, "pooling_mode") {
    let descr = match v {
      JVal::Null => "null".to_string(),
      JVal::Bool(b) => format!("bool {b}"),
      JVal::Num(n) => format!("number {n}"),
      JVal::Object => "object".to_string(),
      // Str/Array handled above; unreachable, but no panic.
      _ => "an unsupported JSON type".to_string(),
    };
    return Err(Error::Backend {
      message: format!(
        "`pooling_mode` is present but not a string or list (got {descr}); \
         a malformed pooling mode is rejected (python `pool_by_config` \
         raises `ValueError` for a non-string/non-list mode) rather than \
         silently falling back to a different strategy"
      ),
    });
  }

  // Legacy boolean flags. python `_normalize_pooling_config` picks the
  // *first active flag in legacy declaration order* and errors out of
  // `pool_by_config` if it is a known-unsupported mode; swift
  // `Pooling(config:)` instead applies a fixed CLS > Mean > Max > Last
  // priority. The task specifies the python priority **CLS > Mean > Max
  // > Last** (swift's order) — applied here over the *supported* flags.
  //
  // `serde_json::Value::as_bool` returned `None` for non-bool values
  // (treated as absent / `false`). We replicate that: only a literal
  // `JVal::Bool(true)` counts as truthy; any other type for these legacy
  // keys is silently treated as `false` (back-compat with the previous
  // parser; the existing tests pin this when an `embedding_dimension`
  // number coexists with bool flags).
  let truthy = |k: &str| matches!(find(cfg, k), Some(JVal::Bool(true)));

  // Reject known-unsupported flags only if they are the *sole* active
  // ones (mirrors python: an unsupported mode that is the resolved one
  // raises; a supported one alongside it just wins via priority).
  if truthy("pooling_mode_cls_token") {
    return Ok(PoolingStrategy::Cls);
  }
  if truthy("pooling_mode_mean_tokens") {
    return Ok(PoolingStrategy::Mean);
  }
  if truthy("pooling_mode_max_tokens") {
    return Ok(PoolingStrategy::Max);
  }
  if truthy("pooling_mode_lasttoken") {
    return Ok(PoolingStrategy::Last);
  }

  // No supported flag set: surface a known-unsupported one if that is
  // what was declared (python `pool_by_config` NotImplementedError);
  // otherwise fall back to python `_normalize_pooling_config`'s `("mean",)`
  // default / swift's `.first`. python's no-active default is `"mean"`;
  // swift's is `.first`. We follow python (primary reference) → mean,
  // unless an unsupported flag is the only thing present.
  for (key, name) in LEGACY_KEYS {
    if (*name == "weightedmean" || *name == "mean_sqrt_len_tokens") && truthy(key) {
      return Err(Error::Backend {
        message: format!(
          "pooling mode {name:?} is not supported (supported: cls, lasttoken, max, mean)"
        ),
      });
    }
  }

  // Any legacy flag key present at all (even all-false) ⇒ python's
  // `("mean",)` default.
  let has_legacy = LEGACY_KEYS
    .iter()
    .any(|(k, _)| cfg.iter().any(|(name, _)| name == *k));
  if has_legacy {
    return Ok(PoolingStrategy::Mean);
  }

  Err(Error::Backend {
    message: "pooling config declares no pooling mode (no `pooling_mode` \
              and no legacy `pooling_mode_*` flags)"
      .into(),
  })
}

/// Convert a parsed JSON number slice into a `usize`, applying the
/// "non-negative integer fits in usize and > 0" rule the matryoshka
/// dimension demands. Mirrors the previous
/// `Value::as_u64()` → `usize::try_from` → `> 0` chain, with a
/// faithful description for the recoverable `Err`.
fn parse_dim_number(key: &str, raw: &str) -> Result<usize> {
  // Negative / fractional / exponent → rejected (`as_u64()` would have
  // returned `None` for these cases, which the previous code mapped to
  // an explicit `Err`).
  if raw.starts_with('-') {
    return Err(Error::Backend {
      message: format!(
        "`{key}` is present but not a non-negative integer (got {raw}); \
         a malformed matryoshka dimension is rejected rather than \
         silently skipping truncation (which would return a \
         full-width embedding the model author did not request)"
      ),
    });
  }
  if raw.contains('.') || raw.contains('e') || raw.contains('E') {
    return Err(Error::Backend {
      message: format!(
        "`{key}` is present but not a non-negative integer (got {raw}); \
         a malformed matryoshka dimension is rejected rather than \
         silently skipping truncation (which would return a \
         full-width embedding the model author did not request)"
      ),
    });
  }
  // Parse as u64; > u64::MAX integer literal → ParseIntError, rejected
  // with the overflow message the previous code surfaced.
  let v: u64 = raw.parse().map_err(|_| Error::Backend {
    message: format!(
      "`{key}` = {raw} exceeds usize::MAX; refusing to use it as a \
       matryoshka dimension"
    ),
  })?;
  let v = usize::try_from(v).map_err(|_| Error::Backend {
    message: format!(
      "`{key}` = {v} exceeds usize::MAX; refusing to use it as a \
       matryoshka dimension"
    ),
  })?;
  if v == 0 {
    return Err(Error::Backend {
      message: format!(
        "`{key}` is 0; a zero matryoshka dimension would produce an \
         empty embedding (rejected rather than silently skipped)"
      ),
    });
  }
  Ok(v)
}

fn parse_pairs(cfg: &[(SmolStr, JVal<'_>)]) -> Result<StPoolingConfig> {
  // python `pool_by_config` rejects `include_prompt: false` (INSTRUCTOR
  // prompt-aware pooling unsupported).
  if let Some(JVal::Bool(false)) = find(cfg, "include_prompt") {
    return Err(Error::Backend {
      message: "prompt-aware pooling (include_prompt=false) is not supported".into(),
    });
  }

  let strategy = resolve_strategy(cfg)?;

  // Matryoshka dim: swift `word_embedding_dimension`; python configs
  // also commonly use `embedding_dimension` (legacy ST). Accept either,
  // `word_embedding_dimension` taking precedence when both are present.
  //
  // C7 (Copilot review 4307622782, #3256688310): a present-but-invalid
  // value (negative / fractional / string / `> usize`) previously went
  // `as_u64()` → `None` → treated as ABSENT → matryoshka truncation
  // silently SKIPPED, so the caller got a full-width embedding when the
  // model author explicitly requested a truncated dimension — a silent
  // wrong embedding.
  //
  // python parity: python `mlx-embeddings` has NO matryoshka /
  // `word_embedding_dimension` truncation at all (grep-confirmed: the dim
  // is carried in the ST config but never used to slice the output; the
  // truncation is an mlxrs/swift-only feature), so there is no python
  // reference for malformed-dimension handling here. The user's standing
  // rule is "never silently produce wrong embeddings": a present key the
  // model author set MUST be honored or surfaced. A present-but-invalid
  // dimension is therefore a recoverable `Err` (an intentionally
  // stricter-than-python safety choice — python has no behavior to match,
  // and a silent full-width fallback is a silent-wrong-result).
  //
  // Only the FIRST present key is consulted (matching the
  // `word_embedding_dimension` > `embedding_dimension` precedence): if
  // `word_embedding_dimension` is present but invalid we reject it rather
  // than silently falling back to `embedding_dimension`.
  let dim_entry = find(cfg, "word_embedding_dimension")
    .map(|v| ("word_embedding_dimension", v))
    .or_else(|| find(cfg, "embedding_dimension").map(|v| ("embedding_dimension", v)));
  let dimension = match dim_entry {
    None => None,
    Some((key, JVal::Num(raw))) => Some(parse_dim_number(key, raw)?),
    Some((key, v)) => {
      // Anything other than a number is a malformed dimension.
      let descr = match v {
        JVal::Null => "null".to_string(),
        JVal::Bool(b) => format!("bool {b}"),
        JVal::Str(s) => format!("string {s:?}"),
        JVal::Array => "array".to_string(),
        JVal::Object => "object".to_string(),
        JVal::Num(_) => "number".to_string(), // handled above
      };
      return Err(Error::Backend {
        message: format!(
          "`{key}` is present but not a non-negative integer (got {descr}); \
           a malformed matryoshka dimension is rejected rather than \
           silently skipping truncation (which would return a \
           full-width embedding the model author did not request)"
        ),
      });
    }
  };

  Ok(StPoolingConfig::new(strategy, true, dimension))
}

/// Parse a `1_Pooling/config.json` from an in-memory JSON string.
///
/// Mirrors python `_read_pooling_config` + `_normalize_pooling_config`
/// (legacy `pooling_mode_*` keys, the modern `pooling_mode` key,
/// `include_prompt` guard) and swift `PoolingConfiguration` decoding —
/// resolved with the CLS > Mean > Max > Last priority over supported
/// flags.
pub fn pooling_from_st_config_str(json: &str) -> Result<StPoolingConfig> {
  let pairs = parse_pooling_json(json)?;
  parse_pairs(&pairs)
}

/// Parse a `1_Pooling/config.json` from raw in-memory JSON bytes.
pub fn pooling_from_st_config_bytes(json: &[u8]) -> Result<StPoolingConfig> {
  // The scanner is byte-oriented but the top-level entry point requires
  // a `&str` to surface a useful "not valid UTF-8" message before any
  // structural parse occurs. JSON is required to be UTF-8 (RFC 8259 §8.1),
  // so any byte input that fails this gate is malformed by spec.
  let s = std::str::from_utf8(json).map_err(|e| Error::Backend {
    message: format!("invalid pooling config JSON: input is not valid UTF-8: {e}"),
  })?;
  pooling_from_st_config_str(s)
}

/// Read and parse `<model_dir>/1_Pooling/config.json`.
///
/// `model_dir` is the model root; the `1_Pooling/config.json` suffix is
/// appended (python `_read_pooling_config`, swift `loadPooling`'s
/// `appending(components: "1_Pooling", "config.json")`). Returns an error
/// if the file is absent or unreadable (python returns `None`; the
/// caller can map the error to a fallback).
///
/// The read is bounded against an untrusted model directory:
///
/// 1. the file is **opened once** — no separate `stat` that a TOCTOU
///    swap/extend could race past. On Unix the open carries
///    `O_NONBLOCK | O_CLOEXEC`: opening a **FIFO** returns immediately
///    instead of blocking until a writer appears (a hostile model dir
///    cannot hang the caller by planting a named pipe at `config.json`).
///    Symlinks **are** followed: HuggingFace Hub caches store
///    `snapshots/<rev>/1_Pooling/config.json` as a symlink into
///    `blobs/<hash>`, so `O_NOFOLLOW` would make `open()` fail (ELOOP)
///    for a normal cached model and the caller would silently fall back
///    to the wrong pooling strategy/dimension. Following the symlink is
///    safe because steps 2–3 enforce the actual guarantees on the
///    *resolved target* (a symlink → FIFO/device/dir is still rejected
///    by step 2's `is_file()` fstat of the opened target, and a
///    symlink → FIFO still cannot hang thanks to `O_NONBLOCK`). On
///    non-Unix targets a plain `File::open` is used (no
///    FIFO-open-blocking semantics to defend against).
/// 2. the *opened handle's* metadata must describe a **regular file**
///    (`metadata.is_file()`) and this is checked **before any read** — a
///    FIFO / device / directory / symlink-to-special (all of which
///    `fs::metadata().len()` would report as `0`, bypassing a pre-read
///    size check) is rejected here with a recoverable [`Error::Backend`].
///    `File::metadata()` `fstat`s the opened descriptor, i.e. the
///    *resolved target* of any symlink, so this check is what defends
///    against symlink → non-regular, not refusing symlinks at open.
///    Because the rejection precedes any `read`, the `O_NONBLOCK` handle
///    is never read from, so a non-blocking `EAGAIN` can never occur.
/// 3. the body is read through `Read::take(MAX + 1)` so at most one byte
///    past the 1 MiB cap is ever allocated; if that cap is exceeded the
///    config is rejected (recoverable [`Error::Backend`]), never parsed.
///
/// No panic and **no hang** — every failure path (absent, non-regular
/// incl. FIFO/device/symlink-to-special, oversized, unreadable, invalid
/// JSON) is a recoverable error the caller can map to a fallback (python
/// returns `None`). A symlink to an in-cap regular JSON file (the HF
/// cache layout) is followed and parsed normally.
pub fn pooling_from_st_config_path(model_dir: impl AsRef<Path>) -> Result<StPoolingConfig> {
  use std::io::Read;

  let path = model_dir.as_ref().join("1_Pooling").join("config.json");

  // Open ONCE: a single handle whose metadata and contents refer to the
  // same file object, closing the stat-then-read TOCTOU window.
  //
  // On Unix a read-only blocking `open()` of a FIFO blocks until a
  // writer appears, so an untrusted model dir that plants a named pipe
  // at `config.json` would hang the caller indefinitely *before* the
  // `is_file()` rejection below could ever run. Open with
  // `O_NONBLOCK | O_CLOEXEC`: a FIFO/non-regular open returns
  // immediately (no writer-wait). Symlinks are intentionally followed
  // (no `O_NOFOLLOW`): HuggingFace Hub caches store
  // `snapshots/<rev>/1_Pooling/config.json` as a symlink into
  // `blobs/<hash>`, so `O_NOFOLLOW` would fail (ELOOP) on a normal
  // cached model and the caller would silently use the wrong pooling.
  // This loses no safety: the `is_file()` check below fstats the
  // *opened (resolved) target*, so a symlink→FIFO/device/dir is still
  // rejected before any read, and the symlink→FIFO open still cannot
  // hang because of `O_NONBLOCK`. On non-Unix targets there is no
  // FIFO-open-blocking semantics to defend against — a plain
  // `File::open` is used.
  #[cfg(unix)]
  let file = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(&path)
      .map_err(|e| Error::Backend {
        message: format!("cannot open pooling config {}: {e}", path.display()),
      })?
  };
  #[cfg(not(unix))]
  let file = std::fs::File::open(&path).map_err(|e| Error::Backend {
    message: format!("cannot open pooling config {}: {e}", path.display()),
  })?;

  // Reject non-regular files from the OPENED handle, BEFORE any read.
  // `File::metadata()` fstats the open descriptor, i.e. the *resolved
  // target* of any symlink we followed at open — so a FIFO / device /
  // directory / symlink-to-any-of-those (all of which `len() == 0` to a
  // pre-read `fs::metadata` check yet still stream/block unbounded data
  // on read) is rejected here. The model dir is untrusted, so only a
  // regular file (or a symlink resolving to one, e.g. the HF blob
  // layout) is accepted. Doing this before any `read` also means the
  // `O_NONBLOCK` handle (which on an opened FIFO could yield `EAGAIN`)
  // is never read from: keep this ordering.
  let meta = file.metadata().map_err(|e| Error::Backend {
    message: format!("cannot stat opened pooling config {}: {e}", path.display()),
  })?;
  if !meta.is_file() {
    return Err(Error::Backend {
      message: format!(
        "pooling config {} is not a regular file; refusing to read",
        path.display()
      ),
    });
  }

  // Read at most `cap + 1` bytes: `take` hard-bounds the allocation
  // regardless of the reported size (a regular file can still be
  // extended between open and read; `take` makes that harmless). If we
  // got more than the cap the config is oversized → reject, never parse.
  let mut bytes = Vec::new();
  file
    .take(MAX_ST_POOLING_CONFIG_BYTES + 1)
    .read_to_end(&mut bytes)
    .map_err(|e| Error::Backend {
      message: format!("cannot read pooling config {}: {e}", path.display()),
    })?;
  if bytes.len() as u64 > MAX_ST_POOLING_CONFIG_BYTES {
    return Err(Error::Backend {
      message: format!(
        "pooling config {} exceeds the {}-byte cap; refusing to read",
        path.display(),
        MAX_ST_POOLING_CONFIG_BYTES
      ),
    });
  }
  pooling_from_st_config_bytes(&bytes)
}

#[cfg(test)]
mod tests {
  //! Unit tests targeting the hand-rolled JSON scanner's edge cases.
  //! End-to-end / pooling-config-semantics tests live in
  //! `mlxrs/tests/embeddings.rs`; here we only pin parser-level
  //! invariants the integration tests don't directly exercise.

  use super::{JVal, parse_pooling_json, pooling_from_st_config_str};

  #[test]
  fn parse_pooling_json_mean_only() {
    let pairs = parse_pooling_json(r#"{"pooling_mode_mean_tokens": true}"#).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0, "pooling_mode_mean_tokens");
    assert!(matches!(pairs[0].1, JVal::Bool(true)));
  }

  #[test]
  fn parse_pooling_json_modern_pooling_mode_string() {
    let pairs = parse_pooling_json(r#"{"pooling_mode": "mean"}"#).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0, "pooling_mode");
    match &pairs[0].1 {
      JVal::Str(s) => assert_eq!(s, "mean"),
      other => panic!("expected Str, got {other:?}"),
    }
  }

  #[test]
  fn parse_pooling_json_ignores_unknown_keys() {
    // Forward-compat: an unrecognized field is parsed but never inspected
    // by the resolver (the existing path-shaped lookups skip it).
    let json =
      r#"{"pooling_mode_mean_tokens": true, "future_field": 42, "nested": {"x": [1, 2, 3]}}"#;
    let cfg = pooling_from_st_config_str(json).unwrap();
    assert_eq!(cfg.strategy(), super::PoolingStrategy::Mean);
  }

  #[test]
  fn parse_pooling_json_rejects_unterminated_string() {
    let err = parse_pooling_json(r#"{"pooling_mode": "unfinished"#).unwrap_err();
    let msg = format!("{err}");
    assert!(
      msg.contains("unterminated string"),
      "expected unterminated-string error, got: {msg}"
    );
    assert!(
      msg.contains("byte 17"),
      "expected byte-offset 17, got: {msg}"
    );
  }

  #[test]
  fn parse_pooling_json_rejects_trailing_comma_object() {
    // Strict JSON: trailing comma is rejected (RFC 8259 §4).
    let err = parse_pooling_json(r#"{"pooling_mode_mean_tokens": true,}"#).unwrap_err();
    assert!(
      format!("{err}").contains("trailing comma"),
      "expected trailing-comma error, got: {err}"
    );
  }

  #[test]
  fn parse_pooling_json_rejects_trailing_comma_array() {
    let err = parse_pooling_json(r#"{"pooling_mode": ["mean",]}"#).unwrap_err();
    assert!(
      format!("{err}").contains("trailing comma"),
      "expected trailing-comma error (in array), got: {err}"
    );
  }

  #[test]
  fn parse_pooling_json_rejects_unexpected_char_at_value() {
    let err = parse_pooling_json(r#"{"pooling_mode": @}"#).unwrap_err();
    let msg = format!("{err}");
    assert!(
      msg.contains("unexpected character") && msg.contains("'@'"),
      "expected unexpected-character error citing '@', got: {msg}"
    );
  }

  #[test]
  fn parse_pooling_json_rejects_trailing_data() {
    let err = parse_pooling_json(r#"{"pooling_mode": "mean"} junk"#).unwrap_err();
    assert!(
      format!("{err}").contains("trailing data"),
      "expected trailing-data error, got: {err}"
    );
  }

  #[test]
  fn parse_pooling_json_handles_string_escapes() {
    // Verify the escape decoder handles `\"` `\\` `\n` correctly so a
    // pooling_mode value carrying them parses to the decoded UTF-8.
    let pairs = parse_pooling_json(r#"{"pooling_mode": "a\"b\\c\nd"}"#).unwrap();
    match &pairs[0].1 {
      JVal::Str(s) => assert_eq!(s, "a\"b\\c\nd"),
      other => panic!("expected Str, got {other:?}"),
    }
  }

  #[test]
  fn parse_pooling_json_handles_unicode_escape() {
    // The input is the literal `é` escape sequence (6 ASCII bytes:
    // \ u 0 0 E 9) — exercises `parse_unicode_escape` and the
    // BMP-codepoint encode path inside `parse_string`. A raw multibyte
    // `é` in the source would skip the escape branch entirely and pass
    // through the "raw multibyte UTF-8 input passes through unchanged"
    // path, leaving the escape decoder untested (Copilot review
    // #3277203315). Decoded output is U+00E9 LATIN SMALL LETTER E WITH
    // ACUTE = "é".
    let pairs = parse_pooling_json(r#"{"pooling_mode": "caf\u00E9"}"#).unwrap();
    match &pairs[0].1 {
      JVal::Str(s) => assert_eq!(s, "café"),
      other => panic!("expected Str, got {other:?}"),
    }
  }

  #[test]
  fn parse_pooling_json_handles_utf16_surrogate_pair() {
    // The input is the literal surrogate-pair escape `\uD83D\uDE00` (12
    // ASCII bytes) — exercises `parse_string`'s high+low-surrogate
    // branch and the `0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00)`
    // combine-into-supplementary-plane codepoint computation. A raw
    // 4-byte `😀` UTF-8 in the source would pass through unchanged and
    // leave the surrogate-pair code path untested (Copilot review
    // #3277203341). Decoded output is U+1F600 GRINNING FACE.
    let pairs = parse_pooling_json(r#"{"pooling_mode": "\uD83D\uDE00"}"#).unwrap();
    match &pairs[0].1 {
      JVal::Str(s) => assert_eq!(s, "\u{1F600}"),
      other => panic!("expected Str, got {other:?}"),
    }
  }

  #[test]
  fn parse_pooling_json_rejects_control_char_in_string() {
    // Raw 0x01 in a string body is rejected (RFC 8259 §7).
    let src = "{\"pooling_mode\": \"a\x01b\"}";
    let err = parse_pooling_json(src).unwrap_err();
    assert!(
      format!("{err}").contains("control character"),
      "expected control-char rejection, got: {err}"
    );
  }

  #[test]
  fn parse_pooling_json_rejects_overly_deep_nesting() {
    // Codex adversarial-review (EMB-1, round 1) flagged that an
    // attacker-controlled `1_Pooling/config.json` can pack hundreds of
    // thousands of `[`/`{` characters into the 1 MiB read cap. Without
    // a depth guard the recursive scanner would blow the thread stack
    // and abort the process instead of returning a recoverable
    // `Error::Backend`. The depth cap (128) rejects such input cleanly.
    //
    // We exercise BOTH the array (`[[[[…]]]]`) and object
    // (`{"k":{"k":…}}`) recursion paths.
    let deep_array = {
      let opens = "[".repeat(super::MAX_NESTING_DEPTH + 16);
      let closes = "]".repeat(super::MAX_NESTING_DEPTH + 16);
      format!(r#"{{"pooling_mode": {opens}{closes}}}"#)
    };
    let err = parse_pooling_json(&deep_array).unwrap_err();
    assert!(
      format!("{err}").contains("depth exceeds"),
      "deep-array nesting must yield depth-cap error, got: {err}"
    );

    let deep_object = {
      let mut s = String::new();
      // Top-level `{"pooling_mode": ` consumes 1 depth level; then
      // packed `{"k":{...}}` adds one more per nesting level.
      for _ in 0..(super::MAX_NESTING_DEPTH + 16) {
        s.push_str("{\"k\":");
      }
      s.push_str("true");
      for _ in 0..(super::MAX_NESTING_DEPTH + 16) {
        s.push('}');
      }
      format!(r#"{{"future_field": {s}}}"#)
    };
    let err = parse_pooling_json(&deep_object).unwrap_err();
    assert!(
      format!("{err}").contains("depth exceeds"),
      "deep-object nesting must yield depth-cap error, got: {err}"
    );
  }

  #[test]
  fn parse_pooling_json_allows_shallow_nesting_within_cap() {
    // The depth guard must NOT regress legitimate shallow configs.
    // 4 levels of nesting (top-level + 3 inside an ignored field) is
    // well within the 128-level cap; the inner value should be parsed
    // and skipped without error.
    let json = r#"{"pooling_mode_mean_tokens": true, "future_field": {"a": [{"b": [1,2,3]}]}}"#;
    let cfg = pooling_from_st_config_str(json).unwrap();
    assert_eq!(cfg.strategy(), super::PoolingStrategy::Mean);
  }

  #[test]
  fn parse_pooling_json_handles_many_unknown_keys_linearly() {
    // Codex adversarial-review (EMB-1, round 2) flagged that the
    // previous `Vec`-of-pairs dedup was O(n²) on hostile configs that
    // pack many unique top-level keys into the 1 MiB read cap (the
    // resolver discards them, but every parsed pair was still scanned
    // linearly before each new push). The fix: unknown keys are
    // parsed for structural validity and then *dropped*, so per-key
    // work is O(1). This test pins that ~10k unique unknown keys
    // parse in well under a second on any contemporary host
    // (millisecond range on Apple Silicon dev hardware).
    //
    // We do not measure wall-clock time directly (flake-prone), but
    // we do build an O(n²) input that would have taken minutes under
    // the old code and now takes < 100 ms; the *successful return*
    // within the harness timeout is the regression signal.
    let mut json = String::from("{\"pooling_mode_mean_tokens\": true");
    for i in 0..10_000 {
      // 10k * ~22 bytes = ~220 KB, well under the 1 MiB cap.
      json.push_str(&format!(", \"unknown_field_{i:05}\": {i}"));
    }
    json.push('}');
    let cfg = pooling_from_st_config_str(&json).unwrap();
    assert_eq!(cfg.strategy(), super::PoolingStrategy::Mean);
  }
}
