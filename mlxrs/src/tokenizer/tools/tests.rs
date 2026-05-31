use super::*;

/// Inline (no-tag) parser used to exercise the brace-counting code path of
/// [`ToolCallProcessor`]: its `name()` is absent from the marker table, so
/// both `tool_call_start` / `tool_call_end` resolve to `""`, just like the
/// Swift `Llama3ToolCallParser` (`startTag == nil`). It parses a plain
/// `{"name": ..., "arguments": ...}` JSON object.
struct InlineJson;

impl ToolParser for InlineJson {
  fn parse(&self, text: &str, _tools: Option<&Value>) -> Result<Vec<ToolCall>, Error> {
    let v: Value =
      serde_json::from_str(text.trim()).map_err(|e| err(format!("inline_json: {e}")))?;
    let name = v
      .get("name")
      .and_then(Value::as_str)
      .ok_or_else(|| err("inline_json: missing name"))?;
    let args = v.get("arguments").cloned().unwrap_or(Value::Null);
    Ok(obj(name, args))
  }
  fn name(&self) -> &'static str {
    "inline_json_test_parser"
  }
  /// Inline-format test parser: `tool_call_start` is empty so the streaming
  /// processor routes via `process_inline_chunk` and never invokes this
  /// method. Lock-step with `parse`: balance the first JSON object, return
  /// the call with end_pos = one past the `}`.
  fn try_parse_one_call(
    &self,
    buffer: &str,
    tools: Option<&Value>,
  ) -> Result<Option<(Vec<ToolCall>, usize)>, Error> {
    let Some((_, obj_end)) = balanced_json_object_prefix(buffer) else {
      return Ok(None);
    };
    let inner = buffer[..obj_end].trim();
    match self.parse(inner, tools) {
      Ok(calls) if !calls.is_empty() => Ok(Some((calls, obj_end))),
      _ => Ok(Some((Vec::new(), obj_end))),
    }
  }
}

// --- tagged formats (json_tools: <tool_call>{json}</tool_call>) ----------

#[test]
fn streaming_tagged_json_single_chunk() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let out = p.process_chunk(r#"<tool_call>{"name": "get_time", "arguments": {}}</tool_call>"#);
  // Whole call consumed in one chunk: no display text leaks.
  assert_eq!(out, None);
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "get_time");
  assert_eq!(*p.tool_calls[0].arguments(), serde_json::json!({}));
}

#[test]
fn streaming_tagged_json_split_across_chunks() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // Payload split in the middle of the JSON body.
  assert_eq!(
    p.process_chunk(r#"<tool_call>{"name": "get_weather", "#),
    None
  );
  assert_eq!(p.tool_calls.len(), 0); // not complete yet
  assert_eq!(
    p.process_chunk(r#""arguments": {"city": "Tokyo"}}</tool_call>"#),
    None
  );
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "get_weather");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"city": "Tokyo"})
  );
}

#[test]
fn streaming_tagged_json_split_mid_token() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // The `<tool_call>` start tag itself is split mid-token across two feeds.
  assert_eq!(p.process_chunk("<tool_"), None); // partial tag — buffered
  assert_eq!(p.tool_calls.len(), 0);
  assert_eq!(
    p.process_chunk(r#"call>{"name": "ping", "arguments": {}}</tool_call>"#),
    None
  );
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "ping");
}

#[test]
fn streaming_leading_text_then_tool_call() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // Leading prose in the same chunk as the start tag is display text and is
  // emitted as soon as the start tag is confirmed
  // — it must not be silently dropped because the call completes later.
  let out = p.process_chunk(r#"Let me check. <tool_call>{"name": "ls", "arguments": {}}"#);
  assert_eq!(out.as_deref(), Some("Let me check. "));
  assert_eq!(p.process_chunk("</tool_call>"), None);
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "ls");
}

#[test]
fn streaming_trailing_text_after_end_tag() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let out = p.process_chunk(r#"<tool_call>{"name": "ls", "arguments": {}}</tool_call> all done"#);
  // Text after the end tag is emitted as ordinary display text.
  assert_eq!(out.as_deref(), Some(" all done"));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "ls");
}

#[test]
fn streaming_multiple_tool_calls() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // Two back-to-back tool calls; the trailing token after the first end tag
  // holds the second start char, so processing recurses into it.
  let out = p.process_chunk(
      r#"<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>"#,
    );
  assert_eq!(out, None);
  assert_eq!(p.tool_calls.len(), 2);
  assert_eq!(p.tool_calls[0].name(), "a");
  assert_eq!(p.tool_calls[1].name(), "b");
}

// --- no tool call --------------------------------------------------------

#[test]
fn streaming_passthrough_no_tool_call() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // Plain generation with no tool call: every chunk passes straight through
  // and nothing is extracted.
  assert_eq!(
    p.process_chunk("The capital of France ").as_deref(),
    Some("The capital of France ")
  );
  assert_eq!(p.process_chunk("is Paris.").as_deref(), Some("is Paris."));
  p.process_eos();
  assert!(p.tool_calls.is_empty());
}

#[test]
fn streaming_false_start_flushed() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // `<thinking>` shares the `<` start char but is not the `<tool_call>` tag:
  // the partial match fails and the buffered text is flushed back out.
  let out = p.process_chunk("<thinking>hmm</thinking>");
  assert_eq!(out.as_deref(), Some("<thinking>hmm</thinking>"));
  assert!(p.tool_calls.is_empty());
}

#[test]
fn streaming_false_start_split_then_flushed() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // `<t` is a genuine prefix of `<tool_call>`, so it stays buffered as
  // ambiguous; only once `<thinking>` diverges from the tag is it flushed.
  assert_eq!(p.process_chunk("<t"), None); // still a valid tag prefix
  let out = p.process_chunk("hinking>");
  assert_eq!(out.as_deref(), Some("<thinking>"));
  assert!(p.tool_calls.is_empty());
}

// --- inline (no-tag) format: brace counting ------------------------------

#[test]
fn streaming_inline_single_chunk() {
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  let out = p.process_chunk(r#"{"name": "now", "arguments": {}}"#);
  assert_eq!(out, None);
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "now");
}

#[test]
fn streaming_inline_split_across_chunks() {
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  // Unbalanced braces buffer; the call emits once balanced + parseable.
  assert_eq!(p.process_chunk(r#"{"name": "now", "#), None);
  assert_eq!(p.tool_calls.len(), 0);
  assert_eq!(p.process_chunk(r#""arguments": {"tz": "UTC"}}"#), None);
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "now");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"tz": "UTC"})
  );
}

#[test]
fn streaming_inline_leading_text() {
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  // Text before the first `{` is returned for display.
  let out = p.process_chunk(r#"sure {"name": "now", "arguments": {}}"#);
  assert_eq!(out.as_deref(), Some("sure "));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "now");
}

#[test]
fn streaming_inline_balanced_non_tool_call_flushed() {
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  // Balanced JSON that is not a tool call (no `name`): flushed back as text.
  let out = p.process_chunk(r#"{"unrelated": 1}"#);
  assert_eq!(out.as_deref(), Some(r#"{"unrelated": 1}"#));
  assert!(p.tool_calls.is_empty());
}

#[test]
fn streaming_inline_no_brace_passthrough() {
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  assert_eq!(
    p.process_chunk("just plain text").as_deref(),
    Some("just plain text")
  );
  assert!(p.tool_calls.is_empty());
}

// --- end-of-sequence (mistral: start tag, no end tag) --------------------

#[test]
fn streaming_mistral_eos() {
  let mut p = ToolCallProcessor::new(Box::new(Mistral), None);
  // Mistral has no end tag in the text stream — the call stays buffered ...
  assert_eq!(
    p.process_chunk(r#"[TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo"}"#),
    None
  );
  assert_eq!(p.tool_calls.len(), 0);
  // ... until process_eos parses the buffered tail.
  p.process_eos();
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "get_weather");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"city": "Tokyo"})
  );
}

#[test]
fn streaming_eos_noop_when_normal() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  p.process_chunk("plain text");
  // EOS while in Normal state with an empty buffer extracts nothing.
  p.process_eos();
  assert!(p.tool_calls.is_empty());
}

// --- malformed / partial input: no panic ---------------------------------

#[test]
fn streaming_malformed_partial_no_panic() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // A start tag with garbage payload that never closes: buffered, no panic,
  // no spurious tool call, and process_eos drops the unparseable tail.
  assert_eq!(p.process_chunk("<tool_call>{not valid json"), None);
  assert!(p.tool_calls.is_empty());
  p.process_eos();
  assert!(p.tool_calls.is_empty());
}

#[test]
fn streaming_malformed_unicode_chunks_no_panic() {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // Non-ASCII text around a partial tag must not slice a UTF-8 boundary.
  let _ = p.process_chunk("héllo <");
  let _ = p.process_chunk("tøøl");
  let _ = p.process_chunk("</tool_call>");
  p.process_eos();
  // No assertion on contents — the contract here is "does not panic".
}

#[test]
fn streaming_malformed_inline_garbage_no_panic() {
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  // Unbalanced inline braces with garbage: buffered without panic.
  assert_eq!(p.process_chunk("{{{ broken"), None);
  assert!(p.tool_calls.is_empty());
  p.process_eos();
  assert!(p.tool_calls.is_empty());
}

// --- adversarial regression coverage -------------------------------------

#[test]
fn streaming_inline_object_then_suffix_one_chunk() {
  // An inline JSON object immediately followed by display
  // text in the SAME chunk. Extraction must not depend on the chunk ending
  // exactly at the closing brace: the object is parsed as the tool call and
  // the suffix is returned as display text.
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  let out = p.process_chunk(r#"{"name":"now","arguments":{}} done"#);
  assert_eq!(out.as_deref(), Some(" done"));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "now");
  // And the same bytes split right after the brace behave identically.
  let mut p2 = ToolCallProcessor::new(Box::new(InlineJson), None);
  assert_eq!(p2.process_chunk(r#"{"name":"now","arguments":{}}"#), None);
  assert_eq!(p2.process_chunk(" done").as_deref(), Some(" done"));
  assert_eq!(p2.tool_calls.len(), 1);
  assert_eq!(p2.tool_calls[0].name(), "now");
}

#[test]
fn streaming_inline_suffix_is_a_second_tool_call() {
  // When the suffix after a balanced object is
  // itself another object, it is extracted as a subsequent tool call rather
  // than leaked as text — all in one chunk.
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  let out = p.process_chunk(r#"{"name":"a","arguments":{}}{"name":"b","arguments":{}}"#);
  assert_eq!(out, None);
  assert_eq!(p.tool_calls.len(), 2);
  assert_eq!(p.tool_calls[0].name(), "a");
  assert_eq!(p.tool_calls[1].name(), "b");
}

#[test]
fn streaming_inline_braces_inside_string_value() {
  // Braces inside a JSON string value must not be counted
  // by the balance scan. `{"unrelated":"}"}` is ONE balanced object, not a
  // truncated `{"unrelated":"}` plus stray `"}`. It has no `name`, so the
  // flush-on-balanced-but-unparseable path returns it verbatim as text.
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  let out = p.process_chunk(r#"{"unrelated":"}"}"#);
  assert_eq!(out.as_deref(), Some(r#"{"unrelated":"}"}"#));
  assert!(p.tool_calls.is_empty());
  // The same braces-in-string inside a real tool call still parse.
  let mut p2 = ToolCallProcessor::new(Box::new(InlineJson), None);
  let out2 = p2.process_chunk(r#"{"name":"echo","arguments":{"s":"a}b{c"}}"#);
  assert_eq!(out2, None);
  assert_eq!(p2.tool_calls.len(), 1);
  assert_eq!(p2.tool_calls[0].name(), "echo");
  assert_eq!(
    *p2.tool_calls[0].arguments(),
    serde_json::json!({"s": "a}b{c"})
  );
}

#[test]
fn streaming_inline_unbalanced_stream_is_bounded() {
  // An inline JSON object whose braces never balance must
  // not let the buffer grow without bound. Once past the cap the processor
  // recovers (drops the runaway tool content) instead of OOM-ing/panicking.
  let mut p = ToolCallProcessor::new(Box::new(InlineJson), None);
  // Open an object and never close it; feed well past the cap in chunks.
  assert_eq!(p.process_chunk(r#"{"name":"now","arguments":{"x":""#), None);
  let big = "a".repeat(64 * 1024);
  // Peak after a post-append cap check: the cap plus the last chunk.
  let bound = MAX_TOOL_CALL_BUFFER_BYTES + big.len();
  let total_fed: usize = 8 * big.len();
  for _ in 0..8 {
    let _ = p.process_chunk(&big);
    // Bounded at every step: it tracks the cap, never the total fed.
    assert!(p.tool_call_buffer.len() <= bound);
  }
  // Far more bytes were fed than the buffer ever held — growth is O(cap),
  // not O(total output).
  assert!(total_fed > bound);
  // The runaway tool content was dropped, never parsed into a tool call.
  assert!(p.tool_calls.is_empty());
  assert_eq!(p.tool_call_buffer.len(), 0);
  // The processor recovers to a working state: a fresh valid call parses.
  let out = p.process_chunk(r#"{"name":"ok","arguments":{}}"#);
  assert_eq!(out, None);
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "ok");
}

#[test]
fn streaming_tagged_missing_end_tag_is_bounded() {
  // A tagged tool call whose end tag never arrives must
  // also be bounded. `<tool_call>` is confirmed, then content streams
  // forever with no `</tool_call>`; at the cap the malformed content is
  // dropped and the buffer reset.
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  assert_eq!(p.process_chunk(r#"<tool_call>{"name":"now""#), None);
  let big = "b".repeat(64 * 1024);
  let bound = MAX_TOOL_CALL_BUFFER_BYTES + big.len();
  for _ in 0..8 {
    let _ = p.process_chunk(&big);
    assert!(p.tool_call_buffer.len() <= bound);
  }
  assert_eq!(p.tool_call_buffer.len(), 0);
  assert!(p.tool_calls.is_empty());
}

#[test]
fn streaming_mistral_empty_end_tag_is_bounded() {
  // Mistral's end tag is empty (closed at EOS only). A
  // runaway generation must still be bounded rather than buffering forever.
  let mut p = ToolCallProcessor::new(Box::new(Mistral), None);
  assert_eq!(
    p.process_chunk(r#"[TOOL_CALLS]get_weather[ARGS]{"city":""#),
    None
  );
  let big = "c".repeat(64 * 1024);
  let bound = MAX_TOOL_CALL_BUFFER_BYTES + big.len();
  for _ in 0..8 {
    let _ = p.process_chunk(&big);
    assert!(p.tool_call_buffer.len() <= bound);
  }
  assert_eq!(p.tool_call_buffer.len(), 0);
  // EOS after recovery is a clean no-op (nothing buffered, nothing parsed).
  p.process_eos();
  assert!(p.tool_calls.is_empty());
}

#[test]
fn streaming_many_back_to_back_tagged_calls_no_stack_overflow() {
  // A single chunk packed with thousands of back-to-back
  // tagged tool calls. The trailing-text-after-end-tag handling must be an
  // iterative loop, not recursive self-calls — recursion would overflow the
  // stack here. All calls must be extracted, in order.
  const N: usize = 4000;
  let mut chunk = String::with_capacity(N * 56);
  for i in 0..N {
    chunk.push_str(&format!(
      r#"<tool_call>{{"name":"f","arguments":{{"i":{i}}}}}</tool_call>"#
    ));
  }
  // The whole batch stays under the buffer cap, so the iterative
  // extraction path (not the cap-rejection path) is exercised.
  assert!(chunk.len() <= MAX_TOOL_CALL_BUFFER_BYTES);
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let out = p.process_chunk(&chunk);
  assert_eq!(out, None);
  assert_eq!(p.tool_calls.len(), N);
  // Ordering preserved: argument `i` increases monotonically with index.
  for (idx, call) in p.tool_calls.iter().enumerate() {
    assert_eq!(call.name(), "f");
    assert_eq!(*call.arguments(), serde_json::json!({ "i": idx }));
  }
}

// --- adversarial: display-text + in-string-delimiter coverage ------------

/// Feed `full` to a fresh `json_tools` processor in the given `chunks`,
/// returning the concatenated display text and the extracted tool calls.
/// A `None` from `process_chunk` contributes nothing to the display string.
fn run_tagged_stream(chunks: &[&str]) -> (String, Vec<ToolCall>) {
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let mut display = String::new();
  for c in chunks {
    if let Some(d) = p.process_chunk(c) {
      display.push_str(&d);
    }
  }
  p.process_eos();
  (display, p.tool_calls)
}

#[test]
fn streaming_tagged_leading_text_is_boundary_equivalent() {
  // Display text before a *real* start tag must be
  // emitted, and the stream's output must be identical regardless of where
  // chunk boundaries fall. Whole-chunk vs split-immediately-before-the-tag
  // must yield the same display text and the same tool calls.
  let whole = r#"Let me check. <tool_call>{"name":"ls","arguments":{}}</tool_call>"#;
  let (d_whole, calls_whole) = run_tagged_stream(&[whole]);
  let (d_split, calls_split) = run_tagged_stream(&[
    "Let me check. ",
    r#"<tool_call>{"name":"ls","arguments":{}}</tool_call>"#,
  ]);
  // The leading prose is never dropped...
  assert_eq!(d_whole, "Let me check. ");
  // ...and is identical no matter the boundary.
  assert_eq!(d_whole, d_split);
  assert_eq!(calls_whole.len(), 1);
  assert_eq!(calls_split.len(), 1);
  assert_eq!(calls_whole[0].name, "ls");
  assert_eq!(calls_split[0].name, "ls");
  // A boundary in the *middle* of the leading text is equivalent too.
  let (d_mid, calls_mid) = run_tagged_stream(&[
    "Let me ",
    r#"check. <tool_call>{"name":"ls","arguments":{}}</tool_call>"#,
  ]);
  assert_eq!(d_mid, "Let me check. ");
  assert_eq!(calls_mid.len(), 1);
}

#[test]
fn streaming_tagged_display_text_between_two_calls() {
  // Display text *between* two
  // back-to-back tagged calls must also be emitted, both when the whole
  // stream arrives in one chunk and when it is split. The trailing-suffix
  // loop re-enters `Normal` on the second start tag, so the inter-call text
  // is leading text for the second call and must surface.
  let whole = concat!(
    r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
    " and then ",
    r#"<tool_call>{"name":"b","arguments":{}}</tool_call>"#,
  );
  let (d_whole, calls_whole) = run_tagged_stream(&[whole]);
  assert_eq!(d_whole, " and then ");
  assert_eq!(calls_whole.len(), 2);
  assert_eq!(calls_whole[0].name, "a");
  assert_eq!(calls_whole[1].name, "b");
  // Same stream, split at every gap — identical output.
  let (d_split, calls_split) = run_tagged_stream(&[
    r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
    " and then ",
    r#"<tool_call>{"name":"b","arguments":{}}</tool_call>"#,
  ]);
  assert_eq!(d_split, " and then ");
  assert_eq!(calls_split.len(), 2);
  assert_eq!(calls_split[1].name, "b");
}

#[test]
fn streaming_tagged_end_delimiter_inside_json_string_value() {
  // A `json_tools` tagged call whose argument
  // *string value* contains the literal end delimiter `</tool_call>`. A
  // naive `contains` / first-match split would cut the JSON object at the
  // delimiter inside the string, fail to parse, and discard the call (the
  // tail leaking as display text). The JSON-string-aware end-tag scan must
  // find the *real* close after the balanced object, extracting the call
  // intact with the delimiter preserved inside the string.
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let out =
    p.process_chunk(r#"<tool_call>{"name":"echo","arguments":{"s":"</tool_call>"}}</tool_call>"#);
  assert_eq!(out, None, "no suffix may leak as display text");
  assert_eq!(p.tool_calls.len(), 1, "the call must not be discarded");
  assert_eq!(p.tool_calls[0].name(), "echo");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"s": "</tool_call>"}),
    "the delimiter inside the string argument is preserved verbatim"
  );
}

#[test]
fn streaming_tagged_end_delimiter_in_string_split_across_chunks() {
  // The same payload, but the chunk boundary lands
  // *inside* the string value that contains the delimiter. The premature
  // `</tool_call>` inside the still-open object must not end collection;
  // only the real end tag after the balanced object completes the call.
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  assert_eq!(
    p.process_chunk(r#"<tool_call>{"name":"echo","arguments":{"s":"<"#),
    None
  );
  assert_eq!(p.tool_calls.len(), 0);
  assert_eq!(p.process_chunk(r#"/tool_call>"}}</tool_call>"#), None);
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "echo");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"s": "</tool_call>"})
  );
}

#[test]
fn streaming_tagged_end_delimiter_in_string_then_trailing_text() {
  // A delimiter-bearing call followed by genuine
  // display text. The real end tag is located after the balanced object, so
  // the trailing text — and only the trailing text — is emitted.
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let out = p.process_chunk(
    r#"<tool_call>{"name":"echo","arguments":{"s":"</tool_call>"}}</tool_call> done"#,
  );
  assert_eq!(out.as_deref(), Some(" done"));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "echo");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"s": "</tool_call>"})
  );
}

#[test]
fn per_parser_try_parse_one_call_routing() {
  // Structural unification: each parser owns ONE method
  // (`try_parse_one_call`) that performs extraction AND end-detection in
  // lock-step. This unit exercises that the per-parser implementations
  // each return the correct `end_pos` for an adversarial in-payload
  // end-tag literal — a future regression in any one parser's override
  // trips here rather than only in an end-to-end streaming test.

  // -- json_tools: balanced object then plain-substring after `}` -------
  {
    // Adversarial payload: string value contains the wrapper end_tag
    // literal. The balanced-object scan must skip the in-string match.
    // Use a payload with `name` so parse() actually emits a call.
    let buf = r#"<tool_call>{"name":"echo","arguments":{"s":"</tool_call>"}}</tool_call>"#;
    let (calls, end_pos) = JsonTools
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len(), "end_pos lands at buffer end");
    assert_eq!(calls.len(), 1, "one call extracted intact");
    assert_eq!(calls[0].name(), "echo");
    assert_eq!(
      *calls[0].arguments(),
      serde_json::json!({"s": "</tool_call>"}),
      "in-string end-tag literal preserved verbatim"
    );
    // Object still open — `Ok(None)` keep collecting.
    assert!(matches!(
      JsonTools.try_parse_one_call(r#"<tool_call>{"s":"</tool_call>"#, None),
      Ok(None)
    ));
  }

  // -- glm47: classify-then-scan (object OR array OR XML) ---------------
  {
    let buf = r#"<tool_call>{"s":"</tool_call>"}</tool_call>"#;
    let (_, end_pos) = Glm47
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
    let buf = r#"<tool_call>[{"s":"</tool_call>"}]</tool_call>"#;
    let (_, end_pos) = Glm47
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
    let xml = "<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>";
    let (_, end_pos) = Glm47
      .try_parse_one_call(xml, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, xml.len());
    assert!(matches!(
      Glm47.try_parse_one_call(r#"<tool_call>[{"s":"</tool_call>"#, None),
      Ok(None)
    ));
  }

  // -- longcat: object fast-path then XML-aware otherwise ---------------
  {
    let buf = r#"<longcat_tool_call>{"s":"</longcat_tool_call>"}</longcat_tool_call>"#;
    let (_, end_pos) = Longcat
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
    let xml = "<longcat_tool_call>name<longcat_arg_key>k</longcat_arg_key><longcat_arg_value>v</longcat_arg_value></longcat_tool_call>";
    let (_, end_pos) = Longcat
      .try_parse_one_call(xml, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, xml.len());
  }

  // -- pythonic: quote/bracket aware ----------------------------------
  {
    let buf = "<|tool_call_start|>[echo(s='<|tool_call_end|>')]<|tool_call_end|>";
    let (_, end_pos) = Pythonic
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
    let buf = r#"<|tool_call_start|>[echo(s="<|tool_call_end|>")]<|tool_call_end|>"#;
    let (_, end_pos) = Pythonic
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
    assert!(matches!(
      Pythonic.try_parse_one_call("<|tool_call_start|>[echo(s='[", None),
      Ok(None)
    ));
  }

  // -- qwen3_coder: forward-scan `</function>` then first end tag ------
  {
    let buf =
      "<tool_call><function=echo><parameter=s></tool_call></parameter></function></tool_call>";
    let (_, end_pos) = Qwen3Coder
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
  }

  // -- minimax_m2: walk `<invoke …></invoke>` blocks --------------------
  {
    let buf = concat!(
      "<minimax:tool_call>",
      r#"<invoke name="f"><parameter name="p">v</parameter></invoke>"#,
      "</minimax:tool_call>",
    );
    let (_, end_pos) = MinimaxM2
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
    let buf = concat!(
      "<minimax:tool_call>",
      r#"<invoke name="f"><parameter name="p"></minimax:tool_call></parameter></invoke>"#,
      "</minimax:tool_call>",
    );
    let (_, end_pos) = MinimaxM2
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
  }

  // -- kimi_k2: balanced JSON args per call, then section end -----------
  {
    let buf = concat!(
      "<|tool_calls_section_begin|>",
      "<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>",
      r#"{"s":"<|tool_calls_section_end|>"}"#,
      "<|tool_call_end|>",
      "<|tool_calls_section_end|>",
    );
    let (_, end_pos) = KimiK2
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
  }

  // -- function_gemma: find `}` outside <escape>...<escape> -------------
  {
    let buf =
      "<start_function_call>call:f{k:<escape><end_function_call><escape>}<end_function_call>";
    let (_, end_pos) = FunctionGemma
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
  }

  // -- gemma4: walk `call:name{...}` blocks (brace + <|"|> aware) -------
  {
    let buf = r#"<|tool_call>call:f{k: <|"|><tool_call|><|"|>}<tool_call|>"#;
    let (_, end_pos) = Gemma4
      .try_parse_one_call(buf, None)
      .expect("Ok")
      .expect("Some");
    assert_eq!(end_pos, buf.len());
  }
}

#[test]
fn payload_starts_with_json_value_classification() {
  // The dynamic JSON-vs-substring switch hinges on this classifier; cover
  // every variant (`Object`, `Array`, `None`) plus the boundary cases that
  // matter in practice (whitespace skipping, partial bytes, multibyte).
  let cases: &[(&str, JsonPayloadStart)] = &[
    // None: empty / all-whitespace / non-JSON leading byte.
    ("", JsonPayloadStart::None),
    ("   ", JsonPayloadStart::None),
    ("\t\n\r ", JsonPayloadStart::None),
    ("<", JsonPayloadStart::None),
    ("<invoke>", JsonPayloadStart::None),
    ("name ", JsonPayloadStart::None),
    ("123", JsonPayloadStart::None),
    (r#""str""#, JsonPayloadStart::None),
    ("null", JsonPayloadStart::None),
    // Object: `{` after RFC-8259 whitespace.
    ("{}", JsonPayloadStart::Object),
    ("{\"k\":1}", JsonPayloadStart::Object),
    ("  {\"k\":1}", JsonPayloadStart::Object),
    ("\n\t{}", JsonPayloadStart::Object),
    // Array: `[` after RFC-8259 whitespace.
    ("[]", JsonPayloadStart::Array),
    ("[1,2,3]", JsonPayloadStart::Array),
    ("  [{\"a\":1}]", JsonPayloadStart::Array),
    ("\n\t[]", JsonPayloadStart::Array),
    // Multibyte content after whitespace doesn't crash the classifier and
    // is correctly classified `None` (first non-ws byte is a UTF-8 lead).
    ("  é", JsonPayloadStart::None),
  ];
  for (input, expected) in cases {
    assert_eq!(
      classify_json_payload_start(input),
      *expected,
      "classify_json_payload_start({input:?})"
    );
  }
}

// --- helper unit coverage ------------------------------------------------

#[test]
fn balanced_json_object_prefix_basics() {
  // No brace at all -> no object.
  assert_eq!(balanced_json_object_prefix(""), None);
  assert_eq!(balanced_json_object_prefix("plain text"), None);
  // Smallest object, and a nested one.
  assert_eq!(balanced_json_object_prefix("{}"), Some((0, 2)));
  assert_eq!(
    balanced_json_object_prefix(r#"{"a": {"b": 1}}"#),
    Some((0, 15))
  );
  // Still open -> keep buffering.
  assert_eq!(balanced_json_object_prefix("{"), None);
  assert_eq!(balanced_json_object_prefix(r#"{"a": {"b":"#), None);
}

#[test]
fn balanced_json_object_prefix_is_string_aware() {
  // Braces *inside* a JSON string value must not count.
  // A naive byte counter sees `{ } {` -> unbalanced; the JSON-aware scan
  // sees one balanced object.
  assert_eq!(
    balanced_json_object_prefix(r#"{"unrelated":"}"}"#),
    Some((0, 17))
  );
  // An escaped quote inside the string keeps string state correct, so the
  // brace after it is still inside the string and not counted.
  assert_eq!(
    balanced_json_object_prefix(r#"{"k":"a\"}b"}"#),
    Some((0, 13))
  );
  // A brace-only string body: every brace is inside the string.
  assert_eq!(
    balanced_json_object_prefix(r#"{"x":"{{{{"}"#),
    Some((0, 12))
  );
}

#[test]
fn balanced_json_array_prefix_basic() {
  // Arrays must scan with the same string/escape/depth discipline
  // as objects.

  // No bracket at all -> no array.
  assert_eq!(balanced_json_array_prefix(""), None);
  assert_eq!(balanced_json_array_prefix("plain text"), None);
  assert_eq!(balanced_json_array_prefix("{not_array:1}"), None);

  // Smallest array, scalars, nested arrays, and an array of objects.
  assert_eq!(balanced_json_array_prefix("[]"), Some((0, 2)));
  assert_eq!(balanced_json_array_prefix("[1,2,3]"), Some((0, 7)));
  assert_eq!(balanced_json_array_prefix(r#"[{"a":1}]"#), Some((0, 9)));
  assert_eq!(balanced_json_array_prefix("[[1],[2]]"), Some((0, 9)));

  // String-aware: a `]` inside a `"..."` element value is not counted.
  assert_eq!(
    balanced_json_array_prefix(r#"["unrelated]"]"#),
    Some((0, 14))
  );
  // Escaped quote inside a string keeps string state correct.
  assert_eq!(balanced_json_array_prefix(r#"["a\"]b"]"#), Some((0, 9)));
  // Bracket-only string body: every bracket is inside the string.
  assert_eq!(balanced_json_array_prefix(r#"["]]]]"]"#), Some((0, 8)));

  // Unbalanced / still open -> keep buffering (None).
  assert_eq!(balanced_json_array_prefix("["), None);
  assert_eq!(balanced_json_array_prefix("[1,2"), None);
  assert_eq!(balanced_json_array_prefix(r#"[{"a":["#), None);

  // Leading text before the first `[` is excluded from the array span.
  let s = "hi [1,2] bye";
  let (st, en) = balanced_json_array_prefix(s).expect("balanced array");
  assert_eq!(&s[..st], "hi ");
  assert_eq!(&s[st..en], "[1,2]");

  // Trailing text after the array is excluded too — the suffix is
  // re-examined by the caller, mirroring the object scanner.
  let t = "[1,2,3] done";
  let (st2, en2) = balanced_json_array_prefix(t).expect("balanced array");
  assert_eq!(&t[st2..en2], "[1,2,3]");
  assert_eq!(&t[en2..], " done");
}

#[test]
fn balanced_json_object_prefix_finds_prefix_and_suffix() {
  // A complete object followed by trailing text returns
  // the object span only, so the suffix can be handled separately.
  let s = r#"{"name":"now","arguments":{}} done"#;
  let (start, end) = balanced_json_object_prefix(s).expect("balanced object");
  assert_eq!(start, 0);
  assert_eq!(&s[start..end], r#"{"name":"now","arguments":{}}"#);
  assert_eq!(&s[end..], " done");
  // Leading text before the first `{` is excluded from the object span.
  let s2 = r#"hi {"a":1} bye"#;
  let (start2, end2) = balanced_json_object_prefix(s2).expect("balanced object");
  assert_eq!(&s2[..start2], "hi ");
  assert_eq!(&s2[start2..end2], r#"{"a":1}"#);
}

#[test]
fn partial_match_basics() {
  assert!(partial_match("", "<tool_call>"));
  assert!(partial_match("<tool", "<tool_call>"));
  assert!(partial_match("<tool_call>", "<tool_call>"));
  // A longer buffer matches only if it starts with the full tag.
  assert!(partial_match("<tool_call>extra", "<tool_call>"));
  assert!(!partial_match("<thinking>", "<tool_call>"));
}

#[test]
fn strip_markers_tagged_and_inline() {
  // Tagged: both delimiters removed, inner trimmed.
  let inner = strip_markers(&JsonTools, "<tool_call>  {\"x\": 1}  </tool_call>");
  assert_eq!(inner, r#"{"x": 1}"#);
  // Inline parser (empty markers): only trimmed.
  let inner = strip_markers(&InlineJson, "  {\"x\": 1}  ");
  assert_eq!(inner, r#"{"x": 1}"#);
}

// --- structural fix: pending_display + dynamic JSON end detection
// ----------------------------------------------------------------------

/// Feed an arbitrary `[parser_factory]`-flavoured tagged stream as `chunks`
/// and return the concatenated display text + extracted tool calls (running
/// `process_eos` to flush trailing state). Generic over the parser so the
/// same harness exercises `json_tools`, `glm47`, `longcat`.
fn run_with_parser(parser: Box<dyn ToolParser>, chunks: &[&str]) -> (String, Vec<ToolCall>) {
  let mut p = ToolCallProcessor::new(parser, None);
  let mut display = String::new();
  for c in chunks {
    if let Some(d) = p.process_chunk(c) {
      display.push_str(&d);
    }
  }
  p.process_eos();
  (display, p.tool_calls)
}

#[test]
fn streaming_leading_text_split_inside_start_tag_persists() {
  // pending_display: a chunk boundary that lands
  // *inside* the start tag must still emit the leading prose. With the
  // prior per-chunk `leading_token` local, "Let me <" parked the leading
  // text in the next chunk's `tool_call_buffer` only — the second chunk
  // entered `PotentialToolCall` with no leading_token, and the prose was
  // silently dropped at confirmation. With `pending_display` persisting on
  // the processor, the split is byte-equivalent to the one-chunk version.
  let (d_split, c_split) = run_with_parser(
    Box::new(JsonTools),
    &[
      "Let me <",
      r#"tool_call>{"name":"ls","arguments":{}}</tool_call>"#,
    ],
  );
  let (d_whole, c_whole) = run_with_parser(
    Box::new(JsonTools),
    &[r#"Let me <tool_call>{"name":"ls","arguments":{}}</tool_call>"#],
  );
  assert_eq!(d_split, "Let me ");
  assert_eq!(d_split, d_whole, "split-inside-start-tag must equal whole");
  assert_eq!(c_split.len(), 1);
  assert_eq!(c_whole.len(), 1);
  assert_eq!(c_split[0].name, "ls");
  assert_eq!(c_whole[0].name, "ls");
}

#[test]
fn streaming_leading_text_every_byte_boundary_inside_start_tag() {
  // The structural fix is "no split inside the start tag drops leading
  // text" for *every* byte boundary — exercise them all. The start tag
  // `<tool_call>` is 11 bytes; the leading text is "Let me " (7 bytes);
  // for k = 1..=(7 + 11) split the stream after `k` bytes of
  // "Let me <tool_call>" and verify identical output.
  let prefix = "Let me <tool_call>";
  let tail = r#"{"name":"ls","arguments":{}}</tool_call>"#;
  let combined: String = format!("{prefix}{tail}");
  let (d_baseline, c_baseline) = run_with_parser(Box::new(JsonTools), &[&combined]);
  assert_eq!(d_baseline, "Let me ");
  assert_eq!(c_baseline.len(), 1);
  for k in 1..prefix.len() {
    let head = &combined[..k];
    let rest = &combined[k..];
    let (d, c) = run_with_parser(Box::new(JsonTools), &[head, rest]);
    assert_eq!(
      d, d_baseline,
      "byte split at k={k} ({head:?}|{rest:?}) lost leading text"
    );
    assert_eq!(c.len(), 1, "byte split at k={k} lost the call");
    assert_eq!(c[0].name, "ls");
  }
}

#[test]
fn streaming_pending_display_flushed_on_false_start() {
  // `pending_display` carrying leading text across chunks must
  // also flush back to display when the start tag turns out to be a false
  // start (strict-prefix divergence). Without this the leading prose would
  // stick in `pending_display` and either leak into a later call or just
  // be silently lost.
  let (d, c) = run_with_parser(
    Box::new(JsonTools),
    &["Let me <", "thinking>oops</thinking> and continue"],
  );
  assert_eq!(c.len(), 0, "no tool call from a false start");
  assert_eq!(
    d, "Let me <thinking>oops</thinking> and continue",
    "leading prose + false-start prefix + remainder all surface"
  );
}

#[test]
fn streaming_back_to_back_with_trailing_partial_next_start() {
  // Display text between two tagged calls where the second chunk's tail is
  // a partial *next* start tag prefix. This exercises the trailing-suffix
  // loop *plus* pending_display persistence across the loop iteration: the
  // " and then <" is leading text for the second tool-call attempt and
  // must survive into the chunk that completes the tag.
  let (d, c) = run_with_parser(
    Box::new(JsonTools),
    &[
      r#"<tool_call>{"name":"a","arguments":{}}</tool_call> and then <"#,
      r#"tool_call>{"name":"b","arguments":{}}</tool_call>"#,
    ],
  );
  assert_eq!(d, " and then ");
  assert_eq!(c.len(), 2);
  assert_eq!(c[0].name, "a");
  assert_eq!(c[1].name, "b");
}

#[test]
fn streaming_glm47_json_fallback_end_tag_in_string_extracted() {
  // Dynamic JSON-end detection: glm47's parse() falls back to
  // `glm_parse_json` for payloads that are a plain `{...}` JSON object —
  // not just `json_tools`. The static per-parser flag missed this and a
  // glm47 JSON-fallback payload whose argument string contains the literal
  // `</tool_call>` would be cut at the in-string delimiter, fail to parse,
  // and discard the call. The dynamic per-payload scan accepts the end
  // tag only after the balanced object closes, so the call extracts
  // intact with the delimiter preserved.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &[r#"<tool_call>{"name":"echo","arguments":{"s":"</tool_call>"}}</tool_call>"#],
  );
  assert_eq!(d, "", "no suffix leaks");
  assert_eq!(c.len(), 1, "glm47 JSON-fallback call must extract intact");
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"s": "</tool_call>"}),
    "in-string delimiter preserved verbatim"
  );
}

#[test]
fn streaming_glm47_json_fallback_end_tag_in_string_split_across_chunks() {
  // Same as above, with the chunk boundary inside the in-string delimiter
  // — the premature `</tool_call>` inside a still-open object must not end
  // collection.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &[
      r#"<tool_call>{"name":"echo","arguments":{"s":"<"#,
      r#"/tool_call>"}}</tool_call>"#,
    ],
  );
  assert_eq!(d, "");
  assert_eq!(c.len(), 1);
  assert_eq!(c[0].name, "echo");
  assert_eq!(c[0].arguments, serde_json::json!({"s": "</tool_call>"}));
}

#[test]
fn streaming_glm47_json_array_end_tag_in_string_extracted() {
  // glm47's `glm_parse_json` also accepts `Value::Array`
  // (matches `[{...}, ...]` and takes the first object). A payload of the
  // form `[{"name":"echo","arguments":{"s":"</tool_call>"}}]` must not be
  // routed to the plain-substring path (a scan that only matches a leading
  // `{` would truncate at the in-string `</tool_call>` and either drop the
  // call or leak the rest as display text). The array-shape classifier +
  // balanced array scanner accepts the end tag only after the top-level `]`
  // closes, so the call extracts intact.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &[r#"<tool_call>[{"name":"echo","arguments":{"s":"</tool_call>"}}]</tool_call>"#],
  );
  assert_eq!(d, "", "no suffix leaks (end tag matched after the array)");
  assert_eq!(c.len(), 1, "glm47 JSON-array call must extract intact");
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"s": "</tool_call>"}),
    "in-string delimiter preserved verbatim"
  );
}

#[test]
fn streaming_glm47_json_array_end_tag_in_string_split_across_chunks() {
  // Same as above but with the chunk boundary inside the in-string end tag
  // — the premature `</tool_call>` inside a still-open top-level array
  // must not end collection.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &[
      r#"<tool_call>[{"name":"echo","arguments":{"s":"<"#,
      r#"/tool_call>"}}]</tool_call>"#,
    ],
  );
  assert_eq!(d, "");
  assert_eq!(c.len(), 1);
  assert_eq!(c[0].name, "echo");
  assert_eq!(c[0].arguments, serde_json::json!({"s": "</tool_call>"}));
}

// `longcat` and `json_tools` do
// NOT accept top-level JSON arrays — `longcat.parse` only takes the `{`
// fast-path, and `json_tools.parse` requires a top-level object with
// `name`/`arguments` keys (an array fails `v.get("name")`). The array-shape
// classifier + balanced array scanner still defends those parsers' buffers
// (an in-string `</tool_call>` inside a JSON-array payload does not cut
// the buffer mid-array), even though the parsers themselves reject the
// resulting array shape; that's preferable to truncating the buffer at the
// wrong byte. No dedicated parse-result
// streaming tests are added for those parsers because their `parse()`
// legitimately rejects an array, so the call surface there is unchanged.

#[test]
fn streaming_longcat_json_fastpath_end_tag_in_string_extracted() {
  // Longcat's parse() has a `{...}` fast-path that accepts JSON payloads
  // — same defect class as glm47. The dynamic per-payload scan must apply
  // to longcat too, since the structural fix is "anything that looks like
  // JSON gets the JSON-aware scan", with no per-parser opt-in.
  let (d, c) = run_with_parser(
    Box::new(Longcat),
    &[
      r#"<longcat_tool_call>{"name":"echo","arguments":{"s":"</longcat_tool_call>"}}</longcat_tool_call>"#,
    ],
  );
  assert_eq!(d, "");
  assert_eq!(c.len(), 1, "longcat JSON-fastpath call must extract intact");
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"s": "</longcat_tool_call>"}),
  );
}

#[test]
fn streaming_longcat_json_fastpath_end_tag_in_string_split_across_chunks() {
  let (d, c) = run_with_parser(
    Box::new(Longcat),
    &[
      r#"<longcat_tool_call>{"name":"echo","arguments":{"s":"<"#,
      r#"/longcat_tool_call>"}}</longcat_tool_call>"#,
    ],
  );
  assert_eq!(d, "");
  assert_eq!(c.len(), 1);
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"s": "</longcat_tool_call>"}),
  );
}

#[test]
fn streaming_pending_display_counted_against_cap() {
  // Cap: `MAX_TOOL_CALL_BUFFER_BYTES` must bound the COMBINED size
  // `tool_call_buffer.len() + pending_display.len()`. Without that, an
  // adversary could pile arbitrarily large leading text into
  // `pending_display` before any start char, and the per-buffer cap on
  // `tool_call_buffer` alone would never trigger. Feed long leading text
  // followed by a start char that never gets to a confirmed tag; the
  // combined-size cap must trip and flush.
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // No start char until the very end: every chunk lands entirely in
  // `pending_display` once we cross into PotentialToolCall — but we never
  // do until a `<` arrives. Prime with one `<` so subsequent chunks accrue
  // into `tool_call_buffer`; combined with a long pre-seed in
  // `pending_display`, the cap-check must still bound growth.
  let _ = p.process_chunk("Let me say a lot first <");
  let big = "x".repeat(64 * 1024);
  let bound = MAX_TOOL_CALL_BUFFER_BYTES + big.len();
  for _ in 0..8 {
    let _ = p.process_chunk(&big);
    // Combined bound: neither buffer alone, nor their sum, can pass `bound`.
    assert!(p.tool_call_buffer.len() + p.pending_display.len() <= bound);
  }
  // After recovery the processor is usable again for a fresh stream.
  assert_eq!(p.tool_call_buffer.len(), 0);
  assert_eq!(p.pending_display.len(), 0);
  let out = p.process_chunk(r#"<tool_call>{"name":"ok","arguments":{}}</tool_call>"#);
  // The recovered "Let me say a lot first <" was already returned during
  // the cap-trip; this fresh call has no leading prose.
  assert_eq!(out, None);
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "ok");
}

#[test]
fn streaming_pending_display_cleared_on_eos() {
  // `pending_display` accumulated before a never-arrived start
  // confirmation must be cleared on `process_eos` so it cannot leak into
  // a subsequent generation reusing the same processor.
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let _ = p.process_chunk("Just thinking <");
  // EOS arrives mid-PotentialToolCall — both buffers must be empty after.
  p.process_eos();
  assert!(
    p.pending_display.is_empty(),
    "pending_display leaked past EOS"
  );
  assert!(p.tool_call_buffer.is_empty());
  // A fresh stream is unaffected.
  let out = p.process_chunk("hello");
  assert_eq!(out.as_deref(), Some("hello"));
}

// --- structural fix: parser-capability-aware tagged dispatch ------------
// -----------------------------------------------------------------------

#[test]
fn streaming_pythonic_single_quoted_string_with_unmatched_bracket_preserves_trailing_display() {
  // A prior fix routed every tagged payload whose
  // first non-whitespace byte was `[` through `balanced_json_array_prefix`,
  // which only understands JSON-quoted `"..."` strings. Pythonic legitimately
  // emits `[name(args)]` payloads with SINGLE-quoted string values; the
  // unmatched `[` inside `'[abc'` increments the array scanner's depth,
  // never balances, and the real `<|tool_call_end|>` is never accepted.
  // The processor then collects until EOS / the buffer cap, and
  // `strip_markers` discards the trailing ` after` text.
  //
  // Pythonic is routed through `TaggedPayloadShape::Plain`, which
  // NEVER consults the JSON scanners. End-tag detection is a plain
  // substring search that handles this case correctly.
  let payload = "<|tool_call_start|>[echo(s='[abc')]<|tool_call_end|> after";

  // (i) Single-chunk variant.
  let (d_one, c_one) = run_with_parser(Box::new(Pythonic), &[payload]);
  assert_eq!(
    d_one, " after",
    "single-chunk: trailing display must survive byte-for-byte"
  );
  assert_eq!(c_one.len(), 1, "single-chunk: tool call must extract");
  assert_eq!(c_one[0].name, "echo");
  assert_eq!(c_one[0].arguments, serde_json::json!({"s": "[abc"}));

  // (ii) Split-across-chunks variant — exercise a boundary *inside* the
  // single-quoted string (where a naive array scanner would see the
  // unmatched `[` and never close).
  let (d_split, c_split) = run_with_parser(
    Box::new(Pythonic),
    &[
      "<|tool_call_start|>[echo(s='[",
      "abc')]<|tool_call_end|> after",
    ],
  );
  assert_eq!(
    d_split, " after",
    "split-chunk: trailing display must survive across the split"
  );
  assert_eq!(c_split.len(), 1, "split-chunk: tool call must extract");
  assert_eq!(c_split[0].name, "echo");
  assert_eq!(c_split[0].arguments, serde_json::json!({"s": "[abc"}));

  // (iii) No buffer growth past the end marker — after both variants run,
  // a fresh processor never carries data beyond the call. We run a third
  // standalone processor and observe its internal state directly.
  let mut p = ToolCallProcessor::new(Box::new(Pythonic), None);
  let _ = p.process_chunk(payload);
  p.process_eos();
  assert!(
    p.tool_call_buffer.is_empty(),
    "no buffer growth past the end marker (tool_call_buffer)"
  );
  assert!(
    p.pending_display.is_empty(),
    "no buffer growth past the end marker (pending_display)"
  );
}

#[test]
fn streaming_json_tools_leading_bracket_buffer_extraction_is_contract_correct() {
  // Invariant: `json_tools`'
  // `parse()` requires a top-level `{name, arguments}` object — a top-level
  // array fails `v.get("name")`. `try_parse_one_call` uses the balanced
  // JSON object scanner which finds the FIRST `{` in the payload. For
  // `[{...}]` that first `{` is the inner object; the method returns
  // `Some((Vec::new(), end_pos))` (zero calls but valid section advance)
  // so the call site / processor still progresses past the section. The
  // contract: the call surface returns sensible output (no hang) and
  // trailing display text reaches the caller.
  let buf = r#"<tool_call>[{"name":"echo","arguments":{}}]</tool_call> trailing"#;
  let (calls, end_pos) = JsonTools
    .try_parse_one_call(buf, None)
    .expect("Ok")
    .expect("Some — section is closeable");
  assert!(buf[..end_pos].ends_with("</tool_call>"));
  assert_eq!(
    calls.len(),
    0,
    "json_tools rejects a top-level array shape (no `name` field)"
  );
  // End-to-end streaming: the parser rejects the array shape, but the
  // trailing display text still reaches the caller (no buffer hang).
  let (d, c) = run_with_parser(Box::new(JsonTools), &[buf]);
  assert_eq!(c.len(), 0, "json_tools rejects a top-level array (no name)");
  assert_eq!(
    d, " trailing",
    "trailing display must survive even though parse() rejected the call"
  );
}

#[test]
fn parser_try_parse_one_call_audit_assignments() {
  // Audit lock: each parser's `try_parse_one_call` extraction is
  // fixed here so a future regression in any implementation (or a silent
  // removal of an override) trips a unit test rather than only an
  // integration symptom. Mirrors the per-parser audit:
  //
  // | parser          | strategy                                                                  |
  // |-----------------|----------------------------------------------------------------------------|
  // | json_tools      | balanced JSON object then plain-substring after `}`                       |
  // | glm47           | classify payload: `{` object / `[` array / else `<arg_key>` race          |
  // | longcat         | `{` object fast-path; else value-aware XML scan                           |
  // | pythonic        | quote/bracket aware `)]` then plain-substring                             |
  // | mistral         | EOS-closed; try_parse_one_call mirrors parse via `[ARGS]{json}`           |
  // | qwen3_coder     | forward-scan `</function>` outside `<parameter=…>…</parameter>`         |
  // | minimax_m2      | walk `<invoke …>…</invoke>` blocks; opener-vs-end race                    |
  // | kimi_k2         | walk per-call `<|tool_call_begin|>...{json}<|tool_call_end|>`; opener-vs-end race |
  // | function_gemma  | `call:name{...}` with `<escape>` aware `}` then plain-substring           |
  // | gemma4          | `call:name{...}` with `<|"|>` + balanced braces; opener-vs-end race       |
  //
  // The test fixture: each parser is given a *minimal*, *in-protocol* buffer
  // whose payload data carries the end_tag LITERAL where the parser's own
  // syntax says it is INSIDE the payload, plus a TRAILING legitimate end
  // tag. `try_parse_one_call` must extract the call AND return an `end_pos`
  // exactly at the trailing end-tag's close.

  struct Row {
    label: &'static str,
    parser: Box<dyn ToolParser>,
    buffer: &'static str,
    // Expected end_pos: the byte position one past the trailing end-tag.
    // Set to `buffer.len()` for buffers whose adversarial payload ends at
    // the section close (the common in-table case).
    expect_end_pos_eq_len: bool,
  }
  let rows = [
    Row {
      label: "json_tools",
      parser: Box::new(JsonTools),
      buffer: r#"<tool_call>{"s":"</tool_call>"}</tool_call>"#,
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "glm47 (object)",
      parser: Box::new(Glm47),
      buffer: r#"<tool_call>{"s":"</tool_call>"}</tool_call>"#,
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "glm47 (array)",
      parser: Box::new(Glm47),
      buffer: r#"<tool_call>[{"s":"</tool_call>"}]</tool_call>"#,
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "longcat",
      parser: Box::new(Longcat),
      buffer: r#"<longcat_tool_call>{"s":"</longcat_tool_call>"}</longcat_tool_call>"#,
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "pythonic (single-quoted)",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>[echo(s='<|tool_call_end|>')]<|tool_call_end|>",
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "pythonic (double-quoted)",
      parser: Box::new(Pythonic),
      buffer: r#"<|tool_call_start|>[echo(s="<|tool_call_end|>")]<|tool_call_end|>"#,
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "qwen3_coder",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call><function=echo><parameter=s></tool_call></parameter></function></tool_call>",
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "minimax_m2",
      parser: Box::new(MinimaxM2),
      buffer: "<minimax:tool_call><invoke name=\"f\"><parameter name=\"p\"></minimax:tool_call></parameter></invoke></minimax:tool_call>",
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "kimi_k2",
      parser: Box::new(KimiK2),
      buffer: "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{\"s\":\"<|tool_calls_section_end|>\"}<|tool_call_end|><|tool_calls_section_end|>",
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "function_gemma",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>call:f{k:<escape><end_function_call><escape>}<end_function_call>",
      expect_end_pos_eq_len: true,
    },
    Row {
      label: "gemma4",
      parser: Box::new(Gemma4),
      buffer: r#"<|tool_call>call:f{k: <|"|><tool_call|><|"|>}<tool_call|>"#,
      expect_end_pos_eq_len: true,
    },
  ];
  for row in &rows {
    let result = row
      .parser
      .try_parse_one_call(row.buffer, None)
      .unwrap_or_else(|e| panic!("{}: try_parse_one_call errored: {e}", row.label));
    let (_calls, end_pos) =
      result.unwrap_or_else(|| panic!("{}: section not detected as complete", row.label));
    if row.expect_end_pos_eq_len {
      assert_eq!(
        end_pos,
        row.buffer.len(),
        "{}: end_pos must land at buffer end (one past the trailing close)",
        row.label
      );
    }
    assert!(
      end_pos > 0,
      "{}: end_pos must advance past at least one byte",
      row.label
    );
  }
  // Mistral has no end tag — the streaming processor short-circuits the
  // empty-end-tag case so `try_parse_one_call` is exercised only via
  // `parse`'s default loop and a dedicated mistral test below.
  assert!(Mistral.tool_call_end().is_empty());
}

// --- structural regression coverage --------------------------------------
// A Pythonic payload whose single-quoted string
// argument contains the `<|tool_call_end|>` literal must not drop the
// call. The per-parser scanner closes that defect — exercise it from
// end-to-end streaming, with single-chunk + split-chunk variants.

#[test]
fn streaming_pythonic_string_argument_contains_literal_end_marker_preserves_payload() {
  // Single-quoted argument string carries the end marker literal.
  let payload = "<|tool_call_start|>[echo(s='<|tool_call_end|>')]<|tool_call_end|> after";

  // (i) Single-chunk.
  let (d_one, c_one) = run_with_parser(Box::new(Pythonic), &[payload]);
  assert_eq!(
    d_one, " after",
    "trailing display must survive byte-for-byte"
  );
  assert_eq!(c_one.len(), 1, "tool call must extract");
  assert_eq!(c_one[0].name, "echo");
  assert_eq!(
    c_one[0].arguments,
    serde_json::json!({"s": "<|tool_call_end|>"}),
    "in-string end marker preserved verbatim"
  );

  // (ii) Split inside the in-string `<|tool_call_end|>`.
  let (d_split, c_split) = run_with_parser(
    Box::new(Pythonic),
    &[
      "<|tool_call_start|>[echo(s='<|tool_call_",
      "end|>')]<|tool_call_end|> after",
    ],
  );
  assert_eq!(d_split, " after");
  assert_eq!(c_split.len(), 1);
  assert_eq!(c_split[0].name, "echo");
  assert_eq!(
    c_split[0].arguments,
    serde_json::json!({"s": "<|tool_call_end|>"})
  );
}

#[test]
fn streaming_pythonic_double_quoted_string_with_literal_end_marker_preserves_payload() {
  // Double-quoted argument string carries the end marker literal.
  let payload = r#"<|tool_call_start|>[echo(s="<|tool_call_end|>")]<|tool_call_end|> after"#;

  let (d_one, c_one) = run_with_parser(Box::new(Pythonic), &[payload]);
  assert_eq!(d_one, " after");
  assert_eq!(c_one.len(), 1);
  assert_eq!(c_one[0].name, "echo");
  assert_eq!(
    c_one[0].arguments,
    serde_json::json!({"s": "<|tool_call_end|>"})
  );

  // Split inside the in-string `<|tool_call_end|>`.
  let (d_split, c_split) = run_with_parser(
    Box::new(Pythonic),
    &[
      r#"<|tool_call_start|>[echo(s="<|tool_call_"#,
      r#"end|>")]<|tool_call_end|> after"#,
    ],
  );
  assert_eq!(d_split, " after");
  assert_eq!(c_split.len(), 1);
  assert_eq!(c_split[0].name, "echo");
  assert_eq!(
    c_split[0].arguments,
    serde_json::json!({"s": "<|tool_call_end|>"})
  );
}

#[test]
fn streaming_json_tools_string_value_contains_end_marker_preserves_payload() {
  // In-string
  // `</tool_call>` inside a json_tools payload still does not cut the
  // object mid-payload. (The single-chunk test is above; this one
  // explicitly re-verifies under the per-parser scanner.)
  let payload = r#"<tool_call>{"name":"echo","arguments":{"s":"</tool_call>"}}</tool_call>"#;
  let (d, c) = run_with_parser(Box::new(JsonTools), &[payload]);
  assert_eq!(d, "");
  assert_eq!(c.len(), 1);
  assert_eq!(c[0].name, "echo");
  assert_eq!(c[0].arguments, serde_json::json!({"s": "</tool_call>"}));
}

// --- per-parser override regression coverage (one streaming test per
// override that the audit table requires) --------------------------------

#[test]
fn streaming_qwen3_coder_parameter_value_contains_end_marker_extracted() {
  // qwen3_coder's parameter VALUE legitimately contains `</tool_call>`
  // text. The override skips to `</function>` first, so the in-VALUE
  // literal does not cut the buffer mid-payload.
  let payload =
    "<tool_call><function=echo><parameter=s></tool_call></parameter></function></tool_call> after";
  let mut p = ToolCallProcessor::new(Box::new(Qwen3Coder), None);
  let out = p.process_chunk(payload);
  assert_eq!(out.as_deref(), Some(" after"));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "echo");
}

#[test]
fn streaming_minimax_m2_parameter_value_contains_end_marker_extracted() {
  // minimax_m2's parameter VALUE contains `</minimax:tool_call>` text. The
  // override scans the `<invoke …></invoke>` block before searching for
  // the section end.
  let payload = "<minimax:tool_call><invoke name=\"f\"><parameter name=\"p\"></minimax:tool_call></parameter></invoke></minimax:tool_call> after";
  let mut p = ToolCallProcessor::new(Box::new(MinimaxM2), None);
  let out = p.process_chunk(payload);
  assert_eq!(out.as_deref(), Some(" after"));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "f");
}

#[test]
fn streaming_kimi_k2_argument_string_contains_section_end_marker_extracted() {
  // kimi_k2's per-call `{json}` argument contains the section end marker
  // literal inside a string value. The override uses the balanced JSON
  // object scanner before consuming `<|tool_call_end|>`, then plain
  // substring for the section end.
  let payload = concat!(
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>functions.echo:0<|tool_call_argument_begin|>",
    r#"{"s":"<|tool_calls_section_end|>"}"#,
    "<|tool_call_end|>",
    "<|tool_calls_section_end|>",
    " after",
  );
  let mut p = ToolCallProcessor::new(Box::new(KimiK2), None);
  let out = p.process_chunk(payload);
  assert_eq!(out.as_deref(), Some(" after"));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "echo");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"s": "<|tool_calls_section_end|>"}),
  );
}

#[test]
fn streaming_function_gemma_escape_string_contains_end_marker_extracted() {
  // function_gemma's `<escape>...</escape>` string region contains the end
  // marker literal. The override scans for `}` outside of `<escape>` so
  // the in-string literal does not truncate the call.
  let payload =
    "<start_function_call>call:f{k:<escape><end_function_call><escape>}<end_function_call> after";
  let mut p = ToolCallProcessor::new(Box::new(FunctionGemma), None);
  let out = p.process_chunk(payload);
  assert_eq!(out.as_deref(), Some(" after"));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "f");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"k": "<end_function_call>"})
  );
}

#[test]
fn streaming_gemma4_string_contains_end_marker_extracted() {
  // gemma4's `<|"|>...<|"|>` string region contains the end marker literal
  // (`<tool_call|>`). The override uses balanced braces ignoring the
  // string region so the in-string literal does not truncate the call.
  let payload = r#"<|tool_call>call:f{k: <|"|><tool_call|><|"|>}<tool_call|> after"#;
  let mut p = ToolCallProcessor::new(Box::new(Gemma4), None);
  let out = p.process_chunk(payload);
  assert_eq!(out.as_deref(), Some(" after"));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "f");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"k": "<tool_call|>"})
  );
}

// --- multi-block opener-vs-end-tag race regression coverage --------------
// Multi-block scanners (`xml_invoke_then_end_tag`,
// `kimi_section_then_end_tag`, `gemma4_calls_then_end_tag`) must not
// search for the NEXT block opener before the section `end_tag`. If
// trailing display text after the section close happens to contain the
// inner-block opener literal (e.g. ` text <invoke name="x">` after a closed
// `</minimax:tool_call>`) such a scanner would mis-classify that literal as
// another in-section block, then never find a closing `</invoke>` and
// return `None`, so the completed call would never be emitted. The scanner
// races end_tag against opener at each cursor and returns whichever comes
// first.

#[test]
fn streaming_minimax_m2_trailing_display_with_inner_opener_does_not_hide_end_tag() {
  // Trailing display text after the section close contains the literal
  // inner-opener `<invoke name=`. The scanner must return the end-tag
  // position (BEFORE the trailing display) so the call is emitted and the
  // trailing bytes reach display via the parser's normal flush mechanism.
  let payload = concat!(
    "<minimax:tool_call>",
    r#"<invoke name="f"><parameter name="p">v</parameter></invoke>"#,
    "</minimax:tool_call>",
    r#" some text <invoke name="x">"#,
  );
  let (d, c) = run_with_parser(Box::new(MinimaxM2), &[payload]);
  assert_eq!(c.len(), 1, "completed tool call must be emitted");
  assert_eq!(c[0].name, "f");
  assert_eq!(
    d, r#" some text <invoke name="x">"#,
    "trailing display (with the inner-opener literal) reaches the caller \
       byte-for-byte; the in-display opener does not re-open collection"
  );
}

#[test]
fn streaming_minimax_m2_trailing_display_with_inner_opener_split_across_chunks() {
  // Same payload as above, but split inside the trailing display's literal
  // `<invoke name=` opener: the split must not change behaviour.
  let (d, c) = run_with_parser(
    Box::new(MinimaxM2),
    &[
      concat!(
        "<minimax:tool_call>",
        r#"<invoke name="f"><parameter name="p">v</parameter></invoke>"#,
        "</minimax:tool_call>",
        " some text <invoke ",
      ),
      r#"name="x">"#,
    ],
  );
  assert_eq!(c.len(), 1);
  assert_eq!(c[0].name, "f");
  assert_eq!(d, r#" some text <invoke name="x">"#);
}

#[test]
fn streaming_kimi_k2_trailing_display_with_inner_opener_does_not_hide_end_tag() {
  // Trailing display after the kimi_k2 section close contains the
  // literal per-call opener `<|tool_call_begin|>` (e.g. quoted in a model
  // self-narration). The end-tag must be returned at the section close.
  let payload = concat!(
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>",
    r#"{"k":"v"}"#,
    "<|tool_call_end|>",
    "<|tool_calls_section_end|>",
    " some text <|tool_call_begin|>functions.x:1",
  );
  let (d, c) = run_with_parser(Box::new(KimiK2), &[payload]);
  assert_eq!(c.len(), 1, "completed tool call must be emitted");
  assert_eq!(c[0].name, "f");
  assert_eq!(c[0].arguments, serde_json::json!({"k": "v"}));
  assert_eq!(
    d, " some text <|tool_call_begin|>functions.x:1",
    "trailing display (with the inner-opener literal) reaches the caller"
  );
}

#[test]
fn streaming_kimi_k2_trailing_display_with_inner_opener_split_across_chunks() {
  let (d, c) = run_with_parser(
    Box::new(KimiK2),
    &[
      concat!(
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>",
        r#"{"k":"v"}"#,
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
        " some text <|tool_call_",
      ),
      "begin|>functions.x:1",
    ],
  );
  assert_eq!(c.len(), 1);
  assert_eq!(c[0].name, "f");
  assert_eq!(d, " some text <|tool_call_begin|>functions.x:1");
}

#[test]
fn streaming_gemma4_trailing_display_with_inner_opener_does_not_hide_end_tag() {
  // Trailing display after gemma4's section close contains the literal
  // `call:name{` opener. The end-tag must be returned at the section close
  // so the gemma4 call is emitted; trailing bytes reach display.
  let payload = concat!(
    "<|tool_call>",
    r#"call:f{"k":"v"}"#,
    "<tool_call|>",
    " some text call:x{abc",
  );
  let (d, c) = run_with_parser(Box::new(Gemma4), &[payload]);
  assert_eq!(c.len(), 1, "completed tool call must be emitted");
  assert_eq!(c[0].name, "f");
  assert_eq!(c[0].arguments, serde_json::json!({"k": "v"}));
  assert_eq!(
    d, " some text call:x{abc",
    "trailing display (with the inner-opener literal) reaches the caller"
  );
}

#[test]
fn streaming_gemma4_trailing_display_with_inner_opener_split_across_chunks() {
  let (d, c) = run_with_parser(
    Box::new(Gemma4),
    &[
      concat!(
        "<|tool_call>",
        r#"call:f{"k":"v"}"#,
        "<tool_call|>",
        " some text call",
      ),
      ":x{abc",
    ],
  );
  assert_eq!(c.len(), 1);
  assert_eq!(c[0].name, "f");
  assert_eq!(d, " some text call:x{abc");
}

// glm47 + longcat non-JSON XML fallbacks must not scan for the
// wrapper end-tag with a plain substring search; an in-`<arg_value>` (resp.
// `<longcat_arg_value>`) end-tag literal is VALID value text, and a plain
// search would truncate the call there. They use
// `xml_value_aware_end_tag_scan`, which skips value regions while
// searching.

#[test]
fn streaming_glm47_xml_arg_value_contains_wrapper_end_literal_not_truncated() {
  // glm47 XML payload whose `<arg_value>` body contains `</tool_call>`
  // literal. The wrapper end tag must match ONLY at the position AFTER the
  // value's `</arg_value>` close — not at the in-value literal.
  let payload = concat!(
    "<tool_call>",
    "echo<arg_key>s</arg_key><arg_value>blah</tool_call> more blah</arg_value>",
    "</tool_call>",
    " after",
  );
  let (d, c) = run_with_parser(Box::new(Glm47), &[payload]);
  assert_eq!(d, " after", "trailing display reaches caller");
  assert_eq!(c.len(), 1, "tool call must extract intact");
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"s": "blah</tool_call> more blah"}),
    "in-value wrapper-end literal preserved verbatim inside the arg value"
  );
}

#[test]
fn streaming_glm47_xml_arg_value_contains_wrapper_end_literal_split_across_chunks() {
  // Same as above with the chunk boundary INSIDE the in-value
  // `</tool_call>` literal — neither the scanner nor the parser may
  // mis-detect the wrapper close across the split.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &[
      concat!(
        "<tool_call>",
        "echo<arg_key>s</arg_key><arg_value>blah</tool_",
      ),
      "call> more blah</arg_value></tool_call> after",
    ],
  );
  assert_eq!(d, " after");
  assert_eq!(c.len(), 1);
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"s": "blah</tool_call> more blah"}),
  );
}

#[test]
fn streaming_longcat_xml_arg_value_contains_wrapper_end_literal_not_truncated() {
  // Longcat XML payload whose `<longcat_arg_value>` body contains the
  // wrapper end literal `</longcat_tool_call>`. The wrapper end tag must
  // match ONLY at the position AFTER the value's `</longcat_arg_value>`
  // close.
  let payload = concat!(
    "<longcat_tool_call>",
    "echo<longcat_arg_key>s</longcat_arg_key>",
    "<longcat_arg_value>blah</longcat_tool_call> more blah</longcat_arg_value>",
    "</longcat_tool_call>",
    " after",
  );
  let (d, c) = run_with_parser(Box::new(Longcat), &[payload]);
  assert_eq!(d, " after", "trailing display reaches caller");
  assert_eq!(c.len(), 1, "tool call must extract intact");
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"s": "blah</longcat_tool_call> more blah"}),
    "in-value wrapper-end literal preserved verbatim"
  );
}

#[test]
fn streaming_longcat_xml_arg_value_contains_wrapper_end_literal_split_across_chunks() {
  let (d, c) = run_with_parser(
    Box::new(Longcat),
    &[
      concat!(
        "<longcat_tool_call>",
        "echo<longcat_arg_key>s</longcat_arg_key>",
        "<longcat_arg_value>blah</longcat_",
      ),
      "tool_call> more blah</longcat_arg_value></longcat_tool_call> after",
    ],
  );
  assert_eq!(d, " after");
  assert_eq!(c.len(), 1);
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"s": "blah</longcat_tool_call> more blah"}),
  );
}

// glm47's non-JSON branch in `find_end_tag_in_buffer` must route
// shape 1 (XML-style with `<arg_key>`/`<arg_value>` pairs) and shape 3
// (`glm_parse_plain` fallback — opaque text that may carry a raw
// `<arg_value>` literal) DIFFERENTLY. The previous unconditional value-aware
// scan blocked on a plain-fallback payload containing an unmatched
// `<arg_value>` literal (no `</arg_value>` ever arrives) and dropped the
// call at buffer cap. The discriminator is `<arg_key>` presence, mirroring
// `parse()`'s XML branch gate.

#[test]
fn streaming_glm47_plain_fallback_with_unmatched_arg_value_literal_does_not_block() {
  // Plain-fallback payload (no `<arg_key>`) whose opaque arg text contains a
  // raw `<arg_value>` literal with no matching `</arg_value>`. The end-tag
  // scanner must NOT lock on the missing `</arg_value>` — it must accept
  // `</tool_call>` directly via plain substring search and surface a plain
  // tool call.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &["<tool_call>echo <arg_value></tool_call> after"],
  );
  assert_eq!(d, " after", "trailing display reaches caller");
  assert_eq!(
    c.len(),
    1,
    "plain-fallback call must extract (not be dropped)"
  );
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"raw": "<arg_value>"}),
    "raw `<arg_value>` literal preserved verbatim as plain arg text"
  );
}

#[test]
fn streaming_glm47_plain_fallback_with_unmatched_arg_value_literal_split_across_chunks() {
  // Same as the single-chunk variant but with the chunk boundary inside
  // the wrapper end tag. The plain-substring routing must still locate the
  // `</tool_call>` once both halves arrive.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &["<tool_call>echo <arg_value></tool_", "call> after"],
  );
  assert_eq!(d, " after");
  assert_eq!(c.len(), 1, "plain-fallback call must extract across chunks");
  assert_eq!(c[0].name, "echo");
  assert_eq!(c[0].arguments, serde_json::json!({"raw": "<arg_value>"}));
}

#[test]
fn streaming_glm47_xml_arg_key_arrives_in_later_chunk() {
  // Streaming corner case: the chunk boundary lands inside the `<arg_key>`
  // OPEN tag itself (`<arg_ke|y>`), so chunk 1's buffer contains NO
  // `<arg_key>` literal. The non-JSON arm must NOT latch into "plain" mode
  // on the basis of chunk 1 alone — each call re-evaluates the full buffer,
  // so once chunk 2 reveals `<arg_key>`, routing flips to the value-aware
  // XML scanner and the call extracts intact.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &[
      "<tool_call>echo<arg_ke",
      "y>s</arg_key><arg_value>v</arg_value></tool_call> after",
    ],
  );
  assert_eq!(d, " after");
  assert_eq!(
    c.len(),
    1,
    "XML-style call must extract once `<arg_key>` arrives"
  );
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"s": "v"}),
    "XML-aware routing recovers the key/value pair"
  );
}

// A simple `<arg_key>`-presence rule that scanned the ENTIRE buffer
// (including bytes AFTER the real `</tool_call>` end) would misbehave: a
// valid plain payload like `<tool_call>echo <arg_value></tool_call> after
// <arg_key>` has no `<arg_key>` in the actual payload, but the trailing
// display contains `<arg_key>` → discriminator would flip to XML-aware →
// scanner waits for `</arg_value>` that never comes → buffer grows to cap →
// call dropped. The discriminator is therefore bounded to the prefix up to
// a candidate end_tag:
// race the first end_tag against the first `<arg_key>` and only when the
// key arrives STRICTLY BEFORE the end_tag is this an XML-style payload.

#[test]
fn streaming_glm47_plain_fallback_with_trailing_arg_key_in_display_does_not_block() {
  // Trailing display contains `<arg_key>` AFTER the real
  // `</tool_call>` close. With an unbounded scan, this would flip the
  // discriminator into XML-aware mode and block waiting for `</arg_value>`.
  // With the prefix-bounded race, `</tool_call>` is found at offset 0 of
  // the search (within the payload), so the plain branch wins.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &["<tool_call>echo <arg_value></tool_call> after <arg_key>"],
  );
  assert_eq!(d, " after <arg_key>", "trailing display reaches caller");
  assert_eq!(
    c.len(),
    1,
    "plain-fallback call must extract (not be dropped)"
  );
  assert_eq!(c[0].name, "echo");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"raw": "<arg_value>"}),
    "raw `<arg_value>` literal preserved verbatim as plain arg text"
  );
}

#[test]
fn streaming_glm47_plain_fallback_with_trailing_arg_key_in_display_split_across_chunks() {
  // Split-chunk variant: chunk boundary lands inside the trailing
  // ` after <arg_key>` literal. Once both halves arrive the plain branch
  // still wins because `</tool_call>` precedes any `<arg_key>` in the
  // combined buffer.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &[
      "<tool_call>echo <arg_value></tool_call> after <arg",
      "_key>",
    ],
  );
  assert_eq!(d, " after <arg_key>");
  assert_eq!(c.len(), 1, "plain-fallback call must extract across chunks");
  assert_eq!(c[0].name, "echo");
  assert_eq!(c[0].arguments, serde_json::json!({"raw": "<arg_value>"}));
}

#[test]
fn streaming_glm47_xml_style_with_trailing_arg_key_in_display_does_not_misroute() {
  // Mirror case: payload IS XML-style (`<arg_key>` precedes the end_tag)
  // and the trailing display ALSO contains a stray `<arg_key>` literal.
  // The XML-aware scan must terminate at the FIRST `</tool_call>` that
  // follows the value close (i.e. the one strictly after
  // `</arg_value>`), and the trailing `<arg_key>` must reach display.
  let (d, c) = run_with_parser(
    Box::new(Glm47),
    &[concat!(
      "<tool_call><arg_key>k</arg_key><arg_value>v</arg_value>",
      "</tool_call> bonus <arg_key>"
    )],
  );
  assert_eq!(
    d, " bonus <arg_key>",
    "trailing display (with stray `<arg_key>`) reaches caller"
  );
  assert_eq!(c.len(), 1, "XML-style call must extract intact");
  assert_eq!(c[0].name, "", "no name prefix before the first `<arg_key>`");
  assert_eq!(
    c[0].arguments,
    serde_json::json!({"k": "v"}),
    "key/value extracted via the XML-aware scan"
  );
}

// --- structural-unification regression coverage -------------------------
// The defining defect class: qwen3_coder's `parse()` uses
// `text.rfind("</function>")` because parameter VALUES legitimately carry
// `</function>` literals. A scanner using `find()` (FIRST match) would
// diverge from `parse`'s rfind, so the in-value `</function>` (and the
// in-value wrapper end `</tool_call>`) would cut the section at the wrong
// byte. Extraction + end-detection are UNIFIED in `try_parse_one_call`,
// which uses the SAME rfind chain as `parse`, so both literals are safely
// inside the section.

#[test]
fn streaming_qwen3_coder_parameter_value_with_function_close_and_tool_call_close_literals_extracts_intact()
 {
  // Adversarial payload: parameter VALUE contains BOTH `</function>` and
  // `</tool_call>` literals. The rfind chain (last `</tool_call>`, then
  // last `</function>` before it) must skip the in-value literals.
  let payload = concat!(
    "<tool_call><function=f><parameter=p>v containing ",
    "</function> and </tool_call>",
    "</parameter></function></tool_call>",
  );
  // (i) Single-chunk.
  let (d, c) = run_with_parser(Box::new(Qwen3Coder), &[payload]);
  assert_eq!(d, "", "no trailing display leak");
  assert_eq!(c.len(), 1, "one tool call extracted");
  assert_eq!(c[0].name, "f");
  let p_value = c[0]
    .arguments
    .as_object()
    .and_then(|m| m.get("p"))
    .and_then(Value::as_str)
    .expect("string parameter `p`");
  assert!(
    p_value.contains("</function>"),
    "`</function>` literal preserved verbatim inside the parameter value (got: {p_value:?})"
  );
  assert!(
    p_value.contains("</tool_call>"),
    "`</tool_call>` literal preserved verbatim inside the parameter value (got: {p_value:?})"
  );

  // (ii) Split-across-chunks: boundary inside the in-value `</tool_call>`.
  let (d2, c2) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      concat!(
        "<tool_call><function=f><parameter=p>v containing ",
        "</function> and </tool_",
      ),
      "call></parameter></function></tool_call>",
    ],
  );
  assert_eq!(d2, "");
  assert_eq!(c2.len(), 1);
  assert_eq!(c2[0].name, "f");
}

// --- lock-step audit: try_parse_one_call_matches_parse ------------------
// For every parser, asserting that `try_parse_one_call(buffer)` returns
// the SAME call set that running `parse(strip_markers(buffer))` would,
// for a battery of representative payloads. This is the structural
// safety net: if a future maintenance change drifts a parser's
// `try_parse_one_call` away from its `parse`, this trips immediately.

fn assert_try_parse_one_call_matches_parse(parser: &dyn ToolParser, label: &str, buffer: &str) {
  let try_result = parser
    .try_parse_one_call(buffer, None)
    .unwrap_or_else(|e| panic!("{label}: try_parse_one_call errored: {e}"));
  let (try_calls, end_pos) = try_result
    .unwrap_or_else(|| panic!("{label}: try_parse_one_call returned None (incomplete buffer)"));
  // Run parse() over the EXACT same section bytes the processor would
  // delegate (start/end markers stripped, trimmed) — the contract is that
  // both methods agree call-by-call on the same section.
  let inner = strip_section_markers(
    &buffer[..end_pos],
    parser.tool_call_start(),
    parser.tool_call_end(),
  );
  let parse_calls = parser.parse(inner, None).unwrap_or_default();
  assert_eq!(
    try_calls.len(),
    parse_calls.len(),
    "{label}: try_parse_one_call vs parse call-count mismatch"
  );
  for (i, (a, b)) in try_calls.iter().zip(parse_calls.iter()).enumerate() {
    assert_eq!(a.name, b.name, "{label}[{i}]: name mismatch");
    assert_eq!(a.arguments, b.arguments, "{label}[{i}]: arguments mismatch");
    assert_eq!(a.id, b.id, "{label}[{i}]: id mismatch");
  }
}

#[test]
fn try_parse_one_call_matches_parse_json_tools() {
  let cases = [
    r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
    r#"<tool_call>{"name":"echo","arguments":{"s":"</tool_call>"}}</tool_call>"#,
    // With trailing display: end_pos must land at the section close,
    // and the parsed call from that slice must match.
    r#"<tool_call>{"name":"a","arguments":{}}</tool_call> trailing"#,
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&JsonTools, "json_tools", c);
  }
}

#[test]
fn try_parse_one_call_matches_parse_pythonic() {
  let cases = [
    "<|tool_call_start|>[ping()]<|tool_call_end|>",
    "<|tool_call_start|>[echo(s='hello')]<|tool_call_end|>",
    "<|tool_call_start|>[echo(s='<|tool_call_end|>')]<|tool_call_end|>",
    r#"<|tool_call_start|>[echo(s="<|tool_call_end|>")]<|tool_call_end|>"#,
    "<|tool_call_start|>[ping()]<|tool_call_end|> after",
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&Pythonic, "pythonic", c);
  }
}

#[test]
fn try_parse_one_call_matches_parse_mistral() {
  let cases = [
    r#"[TOOL_CALLS]get_weather[ARGS]{"city":"Tokyo"}"#,
    r#"[TOOL_CALLS]ping[ARGS]{}"#,
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&Mistral, "mistral", c);
  }
}

#[test]
fn try_parse_one_call_matches_parse_qwen3_coder() {
  let cases = [
    "<tool_call><function=ping></function></tool_call>",
    "<tool_call><function=echo><parameter=s>hello</parameter></function></tool_call>",
    // In-value `</function>` AND `</tool_call>` literals.
    concat!(
      "<tool_call><function=f><parameter=p>v containing ",
      "</function> and </tool_call>",
      "</parameter></function></tool_call>",
    ),
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&Qwen3Coder, "qwen3_coder", c);
  }
}

#[test]
fn try_parse_one_call_matches_parse_glm47() {
  let cases = [
    // XML-style.
    "<tool_call>echo<arg_key>s</arg_key><arg_value>v</arg_value></tool_call>",
    // JSON-object fallback.
    r#"<tool_call>{"name":"echo","arguments":{"s":"hi"}}</tool_call>"#,
    // JSON-array fallback.
    r#"<tool_call>[{"name":"echo","arguments":{"s":"hi"}}]</tool_call>"#,
    // Plain fallback (no `<arg_key>`, no JSON leading byte).
    "<tool_call>plain command</tool_call>",
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&Glm47, "glm47", c);
  }
}

#[test]
fn try_parse_one_call_matches_parse_longcat() {
  let cases = [
    // XML-style.
    "<longcat_tool_call>echo<longcat_arg_key>s</longcat_arg_key><longcat_arg_value>v</longcat_arg_value></longcat_tool_call>",
    // JSON fast-path.
    r#"<longcat_tool_call>{"name":"echo","arguments":{"s":"hi"}}</longcat_tool_call>"#,
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&Longcat, "longcat", c);
  }
}

#[test]
fn try_parse_one_call_matches_parse_minimax_m2() {
  let cases = [
    concat!(
      "<minimax:tool_call>",
      r#"<invoke name="f"><parameter name="p">v</parameter></invoke>"#,
      "</minimax:tool_call>",
    ),
    concat!(
      "<minimax:tool_call>",
      r#"<invoke name="a"></invoke><invoke name="b"></invoke>"#,
      "</minimax:tool_call>",
    ),
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&MinimaxM2, "minimax_m2", c);
  }
}

#[test]
fn try_parse_one_call_matches_parse_kimi_k2() {
  let cases = [
    concat!(
      "<|tool_calls_section_begin|>",
      "<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>",
      r#"{"k":"v"}"#,
      "<|tool_call_end|>",
      "<|tool_calls_section_end|>",
    ),
    // Two per-call blocks in one section.
    concat!(
      "<|tool_calls_section_begin|>",
      "<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>",
      r#"{}"#,
      "<|tool_call_end|>",
      "<|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>",
      r#"{}"#,
      "<|tool_call_end|>",
      "<|tool_calls_section_end|>",
    ),
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&KimiK2, "kimi_k2", c);
  }
}

#[test]
fn try_parse_one_call_matches_parse_function_gemma() {
  let cases = [
    "<start_function_call>call:f{k:1}<end_function_call>",
    "<start_function_call>call:f{k:<escape>hello<escape>}<end_function_call>",
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&FunctionGemma, "function_gemma", c);
  }
}

#[test]
fn try_parse_one_call_matches_parse_gemma4() {
  let cases = [
    r#"<|tool_call>call:f{k: 1}<tool_call|>"#,
    r#"<|tool_call>call:f{k: <|"|>hello<|"|>}<tool_call|>"#,
    // Two calls in one section.
    r#"<|tool_call>call:a{k: 1}call:b{k: 2}<tool_call|>"#,
  ];
  for c in cases {
    assert_try_parse_one_call_matches_parse(&Gemma4, "gemma4", c);
  }
}

// --- prefix-bounded end-tag regression coverage -------------------------
// A qwen3_coder `try_parse_one_call` that rfound the wrapper end_tag over
// the whole accumulated buffer would misbehave: for batch `parse()` (input
// is the section payload) that is fine, but for the STREAMING buffer it
// picks the LATER `</tool_call>` whenever trailing display text or a
// back-to-back section is present, swallowing past the real close. The
// current code uses a
// forward-scan that is prefix-bounded to the first section: find the first
// `</function>` outside any `<parameter=…>…</parameter>` region (parameter
// VALUES can carry `</function>` literals), then the FIRST
// `</tool_call>` after that real `</function>`.

#[test]
fn streaming_qwen3_coder_trailing_display_with_tool_call_close_literal_does_not_consume_past_real_close()
 {
  // Trailing display after the real `</tool_call>` itself contains a
  // literal `</tool_call>` token. A whole-buffer rfind would pick the
  // LATER `</tool_call>` and corrupt both the section span and the
  // trailing-token displayed downstream. The forward-scan must pick the
  // FIRST `</tool_call>` after the real `</function>`.
  let payload = concat!(
    "<tool_call><function=f></function></tool_call>",
    " some text containing </tool_call>",
  );
  // (i) Single-chunk.
  let (d, c) = run_with_parser(Box::new(Qwen3Coder), &[payload]);
  assert_eq!(
    c.len(),
    1,
    "exactly one tool call extracted (got {})",
    c.len()
  );
  assert_eq!(c[0].name, "f");
  assert_eq!(
    d, " some text containing </tool_call>",
    "trailing display reaches output byte-for-byte"
  );

  // (ii) Split-across-chunks: boundary inside the trailing display.
  let (d2, c2) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      "<tool_call><function=f></function></tool_call> some text ",
      "containing </tool_call>",
    ],
  );
  assert_eq!(c2.len(), 1);
  assert_eq!(c2[0].name, "f");
  assert_eq!(d2, " some text containing </tool_call>");
}

#[test]
fn streaming_qwen3_coder_back_to_back_calls_extracted_separately() {
  // Back-to-back sections: a whole-buffer rfind would pick the SECOND
  // `</tool_call>` and collapse both calls into ONE (the second
  // `<function=` is past the first `</function>`, so `parse()`'s own rfind
  // chain on the combined slice would still see only the LAST function
  // close). The forward-scan must stop at the first section's real close
  // so the processor loop peels off two separate calls.
  let payload = concat!(
    "<tool_call><function=f></function></tool_call>",
    "<tool_call><function=g></function></tool_call>",
  );
  // (i) Single-chunk.
  let (d, c) = run_with_parser(Box::new(Qwen3Coder), &[payload]);
  assert_eq!(d, "", "no display leak between back-to-back calls");
  assert_eq!(c.len(), 2, "exactly two tool calls extracted");
  assert_eq!(c[0].name, "f");
  assert_eq!(c[1].name, "g");

  // (ii) Split-across-chunks: boundary in the middle of the first
  // `</tool_call>`, before the second `<tool_call>` opens.
  let (d2, c2) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      "<tool_call><function=f></function></tool_",
      "call><tool_call><function=g></function></tool_call>",
    ],
  );
  assert_eq!(d2, "");
  assert_eq!(c2.len(), 2);
  assert_eq!(c2[0].name, "f");
  assert_eq!(c2[1].name, "g");
}

// --- per-parser audit-lock: try_parse_one_call is prefix-bounded
// (back-to-back sections must extract only the FIRST, not collapse) ------

#[test]
fn try_parse_one_call_back_to_back_per_parser_audit() {
  // For every tagged parser (mistral has no end_tag and is EOS-closed, so
  // it's excluded as in the audit table), feed a buffer containing
  // TWO complete sections back-to-back. The parser's try_parse_one_call
  // must return ONLY the first section's calls AND an end_pos that lands
  // exactly at the byte one past the FIRST section's last byte — the
  // streaming processor relies on this to peel off subsequent sections.

  struct Row {
    label: &'static str,
    parser: Box<dyn ToolParser>,
    buffer: &'static str,
    // Byte position one past the FIRST section's close tag.
    expect_end_pos: usize,
    expect_first_name: &'static str,
  }
  let rows = [
    Row {
      label: "json_tools",
      parser: Box::new(JsonTools),
      buffer: concat!(
        r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
        r#"<tool_call>{"name":"b","arguments":{}}</tool_call>"#,
      ),
      expect_end_pos: r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#.len(),
      expect_first_name: "a",
    },
    Row {
      label: "glm47 (object)",
      parser: Box::new(Glm47),
      buffer: concat!(
        r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
        r#"<tool_call>{"name":"b","arguments":{}}</tool_call>"#,
      ),
      expect_end_pos: r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#.len(),
      expect_first_name: "a",
    },
    Row {
      label: "longcat (object)",
      parser: Box::new(Longcat),
      buffer: concat!(
        r#"<longcat_tool_call>{"name":"a","arguments":{}}</longcat_tool_call>"#,
        r#"<longcat_tool_call>{"name":"b","arguments":{}}</longcat_tool_call>"#,
      ),
      expect_end_pos: r#"<longcat_tool_call>{"name":"a","arguments":{}}</longcat_tool_call>"#.len(),
      expect_first_name: "a",
    },
    Row {
      label: "pythonic",
      parser: Box::new(Pythonic),
      buffer: concat!(
        "<|tool_call_start|>[a()]<|tool_call_end|>",
        "<|tool_call_start|>[b()]<|tool_call_end|>",
      ),
      expect_end_pos: "<|tool_call_start|>[a()]<|tool_call_end|>".len(),
      expect_first_name: "a",
    },
    Row {
      label: "qwen3_coder",
      parser: Box::new(Qwen3Coder),
      buffer: concat!(
        "<tool_call><function=a></function></tool_call>",
        "<tool_call><function=b></function></tool_call>",
      ),
      expect_end_pos: "<tool_call><function=a></function></tool_call>".len(),
      expect_first_name: "a",
    },
    Row {
      label: "minimax_m2",
      parser: Box::new(MinimaxM2),
      buffer: concat!(
        r#"<minimax:tool_call><invoke name="a"></invoke></minimax:tool_call>"#,
        r#"<minimax:tool_call><invoke name="b"></invoke></minimax:tool_call>"#,
      ),
      expect_end_pos: r#"<minimax:tool_call><invoke name="a"></invoke></minimax:tool_call>"#.len(),
      expect_first_name: "a",
    },
    Row {
      label: "kimi_k2",
      parser: Box::new(KimiK2),
      buffer: concat!(
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
        "<|tool_calls_section_end|>",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>{}<|tool_call_end|>",
        "<|tool_calls_section_end|>",
      ),
      expect_end_pos: concat!(
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
        "<|tool_calls_section_end|>",
      )
      .len(),
      expect_first_name: "a",
    },
    Row {
      label: "function_gemma",
      parser: Box::new(FunctionGemma),
      buffer: concat!(
        "<start_function_call>call:a{}<end_function_call>",
        "<start_function_call>call:b{}<end_function_call>",
      ),
      expect_end_pos: "<start_function_call>call:a{}<end_function_call>".len(),
      expect_first_name: "a",
    },
    Row {
      label: "gemma4",
      parser: Box::new(Gemma4),
      buffer: concat!(
        r#"<|tool_call>call:a{}<tool_call|>"#,
        r#"<|tool_call>call:b{}<tool_call|>"#,
      ),
      expect_end_pos: r#"<|tool_call>call:a{}<tool_call|>"#.len(),
      expect_first_name: "a",
    },
  ];
  for row in &rows {
    let result = row
      .parser
      .try_parse_one_call(row.buffer, None)
      .unwrap_or_else(|e| panic!("{}: try_parse_one_call errored: {e}", row.label));
    let (calls, end_pos) =
      result.unwrap_or_else(|| panic!("{}: first section not detected complete", row.label));
    assert_eq!(
      end_pos, row.expect_end_pos,
      "{}: end_pos must land one past the FIRST section's close, not the second's",
      row.label
    );
    assert!(
      !calls.is_empty(),
      "{}: at least one call from the first section",
      row.label
    );
    assert_eq!(
      calls[0].name(),
      row.expect_first_name,
      "{}: first section's first call name",
      row.label
    );
  }
  // Mistral is the empty-end-tag exception (mirror the audit table).
  assert!(Mistral.tool_call_end().is_empty());
}

// --- unconditional-reset regression coverage ----------------------------
// An Err arm of process_tagged_chunk that called `cap_recover_into` (a
// no-op below MAX_TOOL_CALL_BUFFER_BYTES) would leave the malformed buffer
// in CollectingToolCall on an Err result, suppressing every subsequent
// output token until cap or EOS. `reset_on_malformed` instead
// drains UNCONDITIONALLY so the next chunk starts fresh in Normal.

/// Tagged-format test parser that returns `Err` from `try_parse_one_call`
/// once a complete `<tc>...</tc>` section is present in the buffer. Used
/// to exercise the Err arm of `process_tagged_chunk` deterministically.
struct AlwaysErrParser;

impl ToolParser for AlwaysErrParser {
  fn parse(&self, _text: &str, _tools: Option<&Value>) -> Result<Vec<ToolCall>, Error> {
    Err(err("always_err: malformed"))
  }
  fn name(&self) -> &'static str {
    "always_err_test_parser"
  }
  fn tool_call_start(&self) -> &'static str {
    "<tc>"
  }
  fn tool_call_end(&self) -> &'static str {
    "</tc>"
  }
  fn try_parse_one_call(
    &self,
    buffer: &str,
    _tools: Option<&Value>,
  ) -> Result<Option<(Vec<ToolCall>, usize)>, Error> {
    // Complete section detected (start + end both present): return Err so
    // the processor exercises its Err arm. Otherwise None (incomplete).
    if buffer.contains("<tc>") && buffer.contains("</tc>") {
      Err(err("always_err: rejected"))
    } else {
      Ok(None)
    }
  }
}

#[test]
fn processor_err_from_try_parse_one_call_clears_buffer_immediately() {
  // Feed a complete malformed-section chunk, then a plain chunk. An Err arm
  // that held the malformed bytes in tool_call_buffer (cap not reached)
  // would force the plain chunk through `process_tagged_chunk`'s
  // collecting branch where its `<` would re-arm a tool-call detection.
  // The unconditional reset drains — buffer drained, state Normal, next plain
  // chunk passes through untouched.
  let mut p = ToolCallProcessor::new(Box::new(AlwaysErrParser), None);
  // Chunk 1: a complete `<tc>...</tc>` section that the parser rejects.
  // Per `recover_at_cap` semantics for CollectingToolCall, the tool buffer
  // is dropped (it's not valid display text) and any pre-confirmation
  // pending_display is surfaced. Here there's no pre-confirmation prose,
  // so this chunk produces no display.
  let out1 = p.process_chunk("<tc>malformed</tc>");
  assert_eq!(out1, None, "no display leak from the Err recovery itself");
  assert!(
    p.tool_call_buffer.is_empty(),
    "tool_call_buffer drained immediately after Err (got {} bytes)",
    p.tool_call_buffer.len()
  );
  assert!(
    p.pending_display.is_empty(),
    "pending_display cleared after Err",
  );
  assert_eq!(
    p.state,
    State::Normal,
    "state reset to Normal after Err — next chunk starts fresh",
  );
  assert_eq!(p.tool_calls.len(), 0, "no tool calls extracted");

  // Chunk 2: plain text with no start char. Must pass through untouched.
  let out2 = p.process_chunk("hello world");
  assert_eq!(
    out2.as_deref(),
    Some("hello world"),
    "subsequent plain chunk passes through immediately (not suppressed until cap)",
  );
}

#[test]
fn processor_err_does_not_suppress_output_until_cap() {
  // Companion of the previous test: feed a SHORT malformed section then a
  // SHORT trailing chunk; without the unconditional reset, the
  // malformed bytes would sit under the cap (no flush), and the next
  // chunk's bytes would be appended to tool_call_buffer (or otherwise
  // mishandled) until the cap eventually fires. The reset returns the trailing
  // chunk verbatim on the same call.
  let mut p = ToolCallProcessor::new(Box::new(AlwaysErrParser), None);
  p.process_chunk("<tc>x</tc>");
  // Sanity: buffer is well under MAX_TOOL_CALL_BUFFER_BYTES.
  assert!(
    "<tc>x</tc>".len() < MAX_TOOL_CALL_BUFFER_BYTES,
    "test premise: malformed section is below the cap",
  );
  // Confirm the buffer is empty BEFORE the next chunk (the unconditional
  // reset already fired; if cap_recover_into had been called the
  // buffer would still hold `<tc>x</tc>` because it's below the cap).
  assert!(p.tool_call_buffer.is_empty());
  assert_eq!(p.state, State::Normal);
  // Plain text passes through normally.
  let out = p.process_chunk("plain");
  assert_eq!(out.as_deref(), Some("plain"));
}

// --- suffix-preservation regression coverage ----------------------------
// Routing a confirmed-but-rejected section through the Err arm of
// `process_tagged_chunk` calls `reset_on_malformed`, which drops the ENTIRE
// `tool_call_buffer`. When the buffer also holds suffix bytes AFTER a
// malformed section's end-tag from the SAME chunk those bytes are
// permanently lost. The trait contract therefore requires:
// confirmed-but-rejected sections MUST return `Ok(Some((Vec::new(),
// end_pos)))` so the processor can preserve the suffix as display. The
// Err arm is reserved for truly indeterminate failures where no end_pos
// is known (the tests above still cover that case verbatim).

/// Tagged-format test parser that exemplifies the tightened contract:
/// once a complete `<tc>...</tc>` section is present it returns
/// `Ok(Some((Vec::new(), end_pos)))` (empty calls + the end_pos one past
/// the `</tc>` close) so the processor preserves the same-chunk suffix.
struct RejectedSectionParser;

impl ToolParser for RejectedSectionParser {
  fn parse(&self, _text: &str, _tools: Option<&Value>) -> Result<Vec<ToolCall>, Error> {
    // `parse` rejects every payload — matching the streaming behaviour.
    Err(err("rejected_section_test_parser: rejected"))
  }
  fn name(&self) -> &'static str {
    "rejected_section_test_parser"
  }
  fn tool_call_start(&self) -> &'static str {
    "<tc>"
  }
  fn tool_call_end(&self) -> &'static str {
    "</tc>"
  }
  fn try_parse_one_call(
    &self,
    buffer: &str,
    _tools: Option<&Value>,
  ) -> Result<Option<(Vec<ToolCall>, usize)>, Error> {
    // Detect a complete `<tc>...</tc>` section and return zero calls plus
    // the byte position one past the section close. This is the
    // contract for confirmed-but-rejected sections: identifying the end
    // boundary lets the processor preserve any same-chunk suffix.
    let start = "<tc>";
    let end = "</tc>";
    let Some(s) = buffer.find(start) else {
      return Ok(None);
    };
    let after_start = s + start.len();
    let Some(e_rel) = buffer[after_start..].find(end) else {
      return Ok(None);
    };
    let end_pos = after_start + e_rel + end.len();
    Ok(Some((Vec::new(), end_pos)))
  }
}

#[test]
fn processor_rejected_section_preserves_same_chunk_suffix() {
  // The malformed section closes mid-chunk and the trailing
  // bytes (`visible`) arrive in the SAME process_chunk call. An Err
  // contract would lose the trailing bytes to `reset_on_malformed`'s
  // buffer drop. Under the Ok-empty contract
  // the processor truncates to `[end_pos..]` and surfaces the suffix as
  // display text, byte-for-byte.
  let (display, calls) = run_with_parser(Box::new(RejectedSectionParser), &["<tc>bad</tc>visible"]);
  assert_eq!(calls.len(), 0, "rejected section emits no tool calls");
  assert_eq!(
    display, "visible",
    "trailing suffix from the SAME chunk must survive the rejected section"
  );
}

#[test]
fn processor_rejected_section_preserves_same_chunk_suffix_split_chunk() {
  // The same rejected section + trailing suffix split
  // across chunk boundaries (start tag in chunk 1; close + suffix in
  // chunk 2). The suffix still reaches display because the section is
  // closed in chunk 2 and its [end_pos..] is processed there.
  let (display, calls) = run_with_parser(
    Box::new(RejectedSectionParser),
    &["<tc>bad", "</tc>visible"],
  );
  assert_eq!(calls.len(), 0, "rejected section emits no tool calls");
  assert_eq!(
    display, "visible",
    "trailing suffix split across chunks must still reach display"
  );
}

#[test]
fn processor_rejected_section_returns_to_normal_state() {
  // After the processor consumes a rejected section + suffix it must
  // return to `State::Normal` with empty buffers — a follow-up chunk of
  // plain text must pass through verbatim (the contract exercised for
  // Err is also exercised here for the Ok-empty path).
  let mut p = ToolCallProcessor::new(Box::new(RejectedSectionParser), None);
  let out1 = p.process_chunk("<tc>bad</tc>visible");
  assert_eq!(out1.as_deref(), Some("visible"));
  assert!(p.tool_call_buffer.is_empty(), "buffer drained");
  assert!(p.pending_display.is_empty(), "pending_display drained");
  assert_eq!(p.state, State::Normal, "state reset");
  let out2 = p.process_chunk("hello world");
  assert_eq!(out2.as_deref(), Some("hello world"));
}

#[test]
fn processor_rejected_section_back_to_back_with_suffix() {
  // Two rejected sections back-to-back in the same chunk, with a
  // suffix after the second close. The processor's trailing-token re-feed
  // loop must consume both sections AND surface the suffix.
  let (display, calls) = run_with_parser(
    Box::new(RejectedSectionParser),
    &["<tc>a</tc><tc>b</tc>tail"],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(
    display, "tail",
    "back-to-back rejected sections + trailing suffix"
  );
}

#[test]
fn processor_rejected_section_preserves_leading_display() {
  // Pre-tag prose is parked in `pending_display` and
  // must surface in stream order BEFORE the rejected section's suffix is
  // emitted. The trailing-token logic already runs through the
  // shared Ok(Some) arm, so emptiness of `calls` cannot suppress the
  // leading display flush.
  let (display, calls) = run_with_parser(
    Box::new(RejectedSectionParser),
    &["before <tc>bad</tc>after"],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(
    display, "before after",
    "leading prose (`before `) + trailing suffix (`after`) survive in stream order"
  );
}

#[test]
fn processor_err_for_truly_indeterminate_buffer_still_resets() {
  // Explicit preservation of the Err arm for the case it was
  // designed for: a parser that legitimately cannot identify the section
  // end (no `end_pos` available). Returning Err is still the correct
  // contract for that — the processor drops the buffer and resets.
  // Mirrors `processor_err_from_try_parse_one_call_clears_buffer_immediately`
  // above but locked here so a regression that re-routes Err through the
  // Ok-empty path (and silently drops the documented Err contract)
  // fails an explicit test.
  let mut p = ToolCallProcessor::new(Box::new(AlwaysErrParser), None);
  let out1 = p.process_chunk("<tc>indeterminate</tc>");
  assert_eq!(out1, None, "Err recovery drops the whole buffer");
  assert!(p.tool_call_buffer.is_empty());
  assert!(p.pending_display.is_empty());
  assert_eq!(p.state, State::Normal);
  // Subsequent plain chunk passes through normally.
  let out2 = p.process_chunk("next");
  assert_eq!(out2.as_deref(), Some("next"));
}

// --- per-parser audit lock ----------------------------------------------
// Every production parser's `try_parse_one_call` already converts a
// confirmed-but-rejected section to `Ok(Some((Vec::new(), end_pos)))` (see
// each parser's `match self.parse(...)` block: the catch-all `_` arm
// returns the empty-calls + end_pos pair). The contract tightening
// does not change that production code path — it documents and locks the
// contract. These tests verify each parser-internal Err-or-empty path
// surfaces a same-chunk suffix through the streaming processor, so an
// accidental future swap to `?`-propagated `Err` trips an explicit test.
//
// Each row constructs a buffer whose section is structurally a tagged-call
// shape but whose body is rejected by the parser, with a trailing display
// suffix in the SAME chunk. Assertion: zero calls extracted AND the
// trailing suffix reaches display verbatim.

#[test]
fn try_parse_one_call_rejected_section_with_same_chunk_suffix_per_parser_audit() {
  // Each tuple: (label, parser, buffer-with-trailing-suffix, expected
  // display). The parser is freshly boxed per row so the trait-object
  // (`!Copy`) can be moved into `run_with_parser`. Vec (not array) so
  // heterogeneous-but-same-type boxes coexist.
  let rows: Vec<(
    &'static str,
    Box<dyn ToolParser>,
    &'static str,
    &'static str,
  )> = vec![
    // json_tools: a top-level array fails `v.get("name")` so `parse`
    // rejects, the streaming impl returns Ok-empty + end_pos (cited:
    // tools.rs json_tools `try_parse_one_call`, `_` arm).
    (
      "json_tools (array body — no `name`)",
      Box::new(JsonTools),
      r#"<tool_call>[{"x":1}]</tool_call>tail"#,
      "tail",
    ),
    // gemma4: an args body that fails `gemma4_args_to_json` → JSON parse —
    // `parse` returns Err, but `balanced_brace_end` closes the body so
    // `try_parse_one_call` has an end_pos. The catch-all `_` arm returns
    // Ok-empty + end_pos (cited: tools.rs gemma4 `try_parse_one_call`,
    // `_` arm).
    (
      "gemma4 (unparseable args body)",
      Box::new(Gemma4),
      r#"<|tool_call>call:f{!bad!}<tool_call|>tail"#,
      "tail",
    ),
  ];
  for (label, parser, buffer, expect_display) in rows {
    let (display, calls) = run_with_parser(parser, &[buffer]);
    assert_eq!(
      calls.len(),
      0,
      "{}: parser rejected the body so zero calls",
      label,
    );
    assert_eq!(
      display, expect_display,
      "{}: same-chunk suffix must reach display",
      label,
    );
  }
}

// --- early-return end-tag regression coverage ---------------------------
// The per-parser audit assumed every parser routed confirmed-but-rejected
// sections through the `_` arm of the final `match self.parse(...)` block.
// That claim missed the EARLY-RETURN paths: a malformed body (e.g. `bad` for
// json_tools) makes `balanced_json_object_prefix` fail BEFORE reaching the
// final match, returning `Ok(None)` even when the wrapper end-tag is
// already in the buffer. The processor treats `Ok(None)` as "incomplete"
// and keeps the whole buffer, so the same-chunk suffix after the malformed
// section is suppressed until cap/EOS (`<tool_call>bad</tool_call>visible`
// never surfaces `visible`). The tightened per-parser early-return
// contract: when the end-tag IS locatable in the buffer, return
// `Ok(Some((Vec::new(), end_pos)))` so the processor preserves the suffix.

#[test]
fn streaming_json_tools_malformed_body_in_closed_section_preserves_same_chunk_suffix() {
  // The motivating case. The body `bad` is unparseable as JSON; a naive
  // implementation where `balanced_json_object_prefix` fails and
  // returns `Ok(None)` even though `</tool_call>` is already in the
  // buffer would leave the suffix `visible` unsurfaced.
  let (display, calls) =
    run_with_parser(Box::new(JsonTools), &["<tool_call>bad</tool_call>visible"]);
  assert_eq!(calls.len(), 0, "malformed body emits no calls");
  assert_eq!(
    display, "visible",
    "trailing suffix from the SAME chunk must survive the malformed-but-closed section"
  );
}

#[test]
fn streaming_json_tools_malformed_body_in_closed_section_preserves_suffix_split_chunk() {
  // Companion across a chunk boundary: start-tag and start of body in
  // chunk 1, end-tag and trailing suffix in chunk 2.
  let (display, calls) = run_with_parser(
    Box::new(JsonTools),
    &["<tool_call>bad", "</tool_call>visible"],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(
    display, "visible",
    "split-chunk: end-tag + suffix in chunk 2 still surface `visible`"
  );
}

#[test]
fn streaming_json_tools_malformed_body_returns_state_to_normal() {
  // After consuming the malformed-but-closed section + suffix the
  // processor must be back in `State::Normal` with empty buffers so a
  // subsequent plain chunk passes through untouched.
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let out1 = p.process_chunk("<tool_call>bad</tool_call>visible");
  assert_eq!(out1.as_deref(), Some("visible"));
  assert!(p.tool_call_buffer.is_empty(), "buffer drained");
  assert!(p.pending_display.is_empty(), "pending_display drained");
  assert_eq!(p.state, State::Normal, "state reset to Normal");
  let out2 = p.process_chunk("hello world");
  assert_eq!(out2.as_deref(), Some("hello world"));
}

#[test]
fn streaming_json_tools_object_body_unbalanced_with_outside_end_tag_closes() {
  // Unit case for the JSON-string-aware quote helper: body opens with
  // `{` but never closes the object. The `</tool_call>` is OUTSIDE any
  // JSON string so the section is closed-but-malformed, the suffix
  // surfaces. This is the case the simple plain-substring scan would have
  // got right, but a naive "stay Ok(None) when body opens with `{`"
  // workaround would have suppressed.
  let (display, calls) = run_with_parser(Box::new(JsonTools), &["<tool_call>{</tool_call>visible"]);
  assert_eq!(calls.len(), 0);
  assert_eq!(display, "visible");
}

#[test]
fn streaming_json_tools_in_string_end_tag_with_incomplete_object_stays_buffered() {
  // Contract guard: `<tool_call>{"s":"</tool_call>` is INCOMPLETE
  // (JSON string open, in-string `</tool_call>` literal). The bound logic
  // MUST NOT falsely close this section — a follow-up chunk must complete
  // the call legitimately. (Locked at the unit level by
  // `per_parser_try_parse_one_call_routing` for json_tools / glm47 /
  // pythonic; this is the end-to-end streaming counterpart.)
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let out1 = p.process_chunk(r#"<tool_call>{"name":"echo","arguments":{"s":"</tool_call>"#);
  assert_eq!(
    out1, None,
    "in-string `</tool_call>` MUST NOT close section"
  );
  assert_eq!(p.tool_calls.len(), 0);
  let out2 = p.process_chunk(r#""}}</tool_call> done"#);
  assert_eq!(out2.as_deref(), Some(" done"));
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "echo");
  assert_eq!(
    *p.tool_calls[0].arguments(),
    serde_json::json!({"s": "</tool_call>"}),
    "in-string `</tool_call>` literal preserved verbatim"
  );
}

// --- per-parser audit-locking -------------------------------------------
// For EVERY parser, construct a payload of shape `<start>BAD<end>visible`
// where BAD is unparseable for the parser's body grammar AND would
// trigger an early `Ok(None)` in a naive implementation. Assert the
// trailing `visible` reaches display byte-for-byte.

#[test]
fn try_parse_one_call_malformed_body_in_closed_section_per_parser_audit() {
  // Each row: (label, parser, buffer with `<start>BAD<end>visible`,
  // expected display).
  //
  // Mistral is excluded — its `tool_call_end` is empty and the streaming
  // processor short-circuits via the `end_tag.is_empty()` branch in
  // `process_tagged_chunk` (no `try_parse_one_call` invocation), so the
  // contract does not bite there.
  let rows: Vec<(&'static str, Box<dyn ToolParser>, String, &'static str)> = vec![
    // json_tools: body `bad` makes `balanced_json_object_prefix` fail (no
    // `{` opener at all → `JsonPayloadStart::None` branch).
    (
      "json_tools (no-{ malformed body)",
      Box::new(JsonTools),
      "<tool_call>bad</tool_call>visible".to_owned(),
      "visible",
    ),
    // json_tools: body `{` makes `balanced_json_object_prefix` fail (opens
    // but never closes); the JSON-aware helper finds end-tag OUTSIDE the
    // (still-open) object structure since no string is in flight.
    (
      "json_tools ({-open malformed body)",
      Box::new(JsonTools),
      "<tool_call>{</tool_call>visible".to_owned(),
      "visible",
    ),
    // pythonic: body `bad` (no `[` opener) makes `pythonic_call_close`
    // return None.
    (
      "pythonic (no-[ malformed body)",
      Box::new(Pythonic),
      "<|tool_call_start|>bad<|tool_call_end|>visible".to_owned(),
      "visible",
    ),
    // pythonic: body `[bad` (call opens but never closes) — quote-aware
    // helper finds end-tag outside any quote.
    (
      "pythonic ([-open malformed body)",
      Box::new(Pythonic),
      "<|tool_call_start|>[bad<|tool_call_end|>visible".to_owned(),
      "visible",
    ),
    // qwen3_coder: body `bad` (no `<function=`) hits the top-level
    // `<function=` not-found early return.
    (
      "qwen3_coder (no-<function= malformed body)",
      Box::new(Qwen3Coder),
      "<tool_call>bad</tool_call>visible".to_owned(),
      "visible",
    ),
    // qwen3_coder: `<function=f` opened but no `</function>` close (and
    // no `<parameter=` in flight) — value-aware helper finds end-tag
    // outside every `<parameter=…></parameter>` region.
    (
      "qwen3_coder (<function= without close)",
      Box::new(Qwen3Coder),
      "<tool_call><function=f</tool_call>visible".to_owned(),
      "visible",
    ),
    // glm47: body `{` (object opens but never closes) — JSON-aware helper
    // finds end-tag OUTSIDE any JSON string.
    (
      "glm47 ({-open malformed body)",
      Box::new(Glm47),
      "<tool_call>{</tool_call>visible".to_owned(),
      "visible",
    ),
    // glm47: body `[` (array opens but never closes) — JSON-aware helper
    // finds end-tag OUTSIDE any JSON string.
    (
      "glm47 ([-open malformed body)",
      Box::new(Glm47),
      "<tool_call>[</tool_call>visible".to_owned(),
      "visible",
    ),
    // longcat: body `{` (object opens but never closes) — JSON-aware
    // helper finds end-tag OUTSIDE any JSON string.
    (
      "longcat ({-open malformed body)",
      Box::new(Longcat),
      "<longcat_tool_call>{</longcat_tool_call>visible".to_owned(),
      "visible",
    ),
    // kimi_k2: `<|tool_call_begin|>` opener found but no
    // `<|tool_call_argument_begin|>` after — plain helper finds the
    // section end-tag.
    (
      "kimi_k2 (call_begin without argument_begin)",
      Box::new(KimiK2),
      concat!(
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.f:0BAD",
        "<|tool_calls_section_end|>visible",
      )
      .to_owned(),
      "visible",
    ),
    // kimi_k2: args region opens with `{` but never balances — JSON-aware
    // helper finds the section end-tag outside any JSON string.
    (
      "kimi_k2 (args { without close)",
      Box::new(KimiK2),
      concat!(
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{",
        "<|tool_calls_section_end|>visible",
      )
      .to_owned(),
      "visible",
    ),
    // minimax_m2: `<invoke name="f"` opened but no `</invoke>` close,
    // and no `<parameter name=` in flight — xml-value-aware helper finds
    // section end-tag outside every `<parameter name=…></parameter>`
    // region.
    (
      "minimax_m2 (<invoke without close)",
      Box::new(MinimaxM2),
      r#"<minimax:tool_call><invoke name="f"</minimax:tool_call>visible"#.to_owned(),
      "visible",
    ),
    // function_gemma: body has no `call:` marker — plain helper finds the
    // wrapper end-tag.
    (
      "function_gemma (no call: marker)",
      Box::new(FunctionGemma),
      "<start_function_call>bad<end_function_call>visible".to_owned(),
      "visible",
    ),
    // function_gemma: `call:f` found but no `{` body opener — plain
    // helper finds the wrapper end-tag.
    (
      "function_gemma (call:NAME without {)",
      Box::new(FunctionGemma),
      "<start_function_call>call:f<end_function_call>visible".to_owned(),
      "visible",
    ),
    // function_gemma: `call:f{garbage` opens body but never closes; no
    // `<escape>` region in flight — value-aware helper finds wrapper
    // end-tag outside every `<escape>...</escape>` region.
    (
      "function_gemma (call:f{ without close)",
      Box::new(FunctionGemma),
      "<start_function_call>call:f{garbage<end_function_call>visible".to_owned(),
      "visible",
    ),
    // gemma4: body has no `call:` (terminates via `(Some(e), None)` arm).
    (
      "gemma4 (no call: marker)",
      Box::new(Gemma4),
      "<|tool_call>bad<tool_call|>visible".to_owned(),
      "visible",
    ),
    // gemma4: `call:f{garbage` opens body but never closes; no `<|"|>`
    // region in flight — value-aware helper finds wrapper end-tag outside
    // every `<|"|>...<|"|>` region.
    (
      "gemma4 (call:f{ without close)",
      Box::new(Gemma4),
      "<|tool_call>call:f{garbage<tool_call|>visible".to_owned(),
      "visible",
    ),
  ];
  for (label, parser, buffer, expect_display) in rows {
    let (display, calls) = run_with_parser(parser, &[buffer.as_str()]);
    assert_eq!(
      calls.len(),
      0,
      "{}: malformed body must produce zero calls",
      label,
    );
    assert_eq!(
      display, expect_display,
      "{}: same-chunk suffix must reach display verbatim",
      label,
    );
  }
}

// --- contract-guard regression tests ------------------------------------
// Each row of the audit above is the positive case: a body that triggers
// an early return path, AND the end-tag IS in the buffer outside any
// legitimate in-value structure → closed-but-malformed. These tests lock
// the negative cases: bodies whose end-tag candidate is INSIDE an open
// in-value region (or no end-tag at all) MUST keep returning `Ok(None)`
// so the next chunk can complete the legitimate call.

#[test]
fn try_parse_one_call_in_value_end_tag_stays_buffered_per_parser_audit() {
  // For each parser: a payload where the end-tag literal is inside an
  // open in-VALUE region. The processor must NOT extract or close —
  // returns None (no display yet), then we feed a closing chunk and
  // expect the full call to materialize.
  //
  // We exercise this via single-chunk `try_parse_one_call` so the
  // Ok(None) discipline is locked at the unit level (the processor's
  // end-to-end behaviour is already exercised by split-chunk tests).
  let rows: Vec<(&'static str, Box<dyn ToolParser>, &'static str)> = vec![
    // json_tools: JSON string open, in-string end-tag literal.
    (
      "json_tools (in-JSON-string end-tag)",
      Box::new(JsonTools),
      r#"<tool_call>{"s":"</tool_call>"#,
    ),
    // pythonic: single-quoted Python string open, in-string end-tag.
    (
      "pythonic (in-single-quote end-tag)",
      Box::new(Pythonic),
      "<|tool_call_start|>[echo(s='<|tool_call_end|>",
    ),
    // glm47 Object: in-JSON-string end-tag.
    (
      "glm47 (in-JSON-string end-tag, object body)",
      Box::new(Glm47),
      r#"<tool_call>{"s":"</tool_call>"#,
    ),
    // glm47 Array: in-JSON-string end-tag.
    (
      "glm47 (in-JSON-string end-tag, array body)",
      Box::new(Glm47),
      r#"<tool_call>[{"s":"</tool_call>"#,
    ),
    // longcat: in-JSON-string end-tag.
    (
      "longcat (in-JSON-string end-tag)",
      Box::new(Longcat),
      r#"<longcat_tool_call>{"s":"</longcat_tool_call>"#,
    ),
    // kimi_k2: args JSON string open, in-string section-end literal.
    (
      "kimi_k2 (in-args-JSON-string section-end)",
      Box::new(KimiK2),
      concat!(
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>",
        r#"{"s":"<|tool_calls_section_end|>"#,
      ),
    ),
    // qwen3_coder: `<parameter=p>` open with in-value end-tag literal.
    (
      "qwen3_coder (in-parameter-value end-tag)",
      Box::new(Qwen3Coder),
      "<tool_call><function=f><parameter=p></tool_call>",
    ),
    // minimax_m2: `<parameter name="p">` open with in-value end-tag.
    (
      "minimax_m2 (in-parameter-value end-tag)",
      Box::new(MinimaxM2),
      r#"<minimax:tool_call><invoke name="f"><parameter name="p"></minimax:tool_call>"#,
    ),
    // function_gemma: `<escape>` open with in-escape end-tag.
    (
      "function_gemma (in-escape end-tag)",
      Box::new(FunctionGemma),
      "<start_function_call>call:f{k:<escape><end_function_call>",
    ),
    // gemma4: `<|"|>` open with in-STR end-tag.
    (
      "gemma4 (in-STR end-tag)",
      Box::new(Gemma4),
      r#"<|tool_call>call:f{k: <|"|><tool_call|>"#,
    ),
  ];
  for (label, parser, buffer) in rows {
    let result = parser
      .try_parse_one_call(buffer, None)
      .unwrap_or_else(|e| panic!("{}: try_parse_one_call errored: {e}", label));
    assert!(
      result.is_none(),
      "{}: in-value end-tag literal MUST NOT close the section (got {:?})",
      label,
      result,
    );
  }
}

// --- STRUCTURAL: parser-opener-search bias by suffix bytes --------------
//
// Without the bound-first step, every single-section parser would perform
// its opener-search (`payload.find("<function=")`,
// `balanced_json_object_prefix(payload)`, `payload.find("[")`,
// `payload.find("call:")`) over the WHOLE payload —
// including bytes AFTER the wrapper end-tag. A buffer shaped
// `<wrapper>BAD</wrapper>SUFFIX-WITH-PARSER-SYNTAX` made the body scan
// lock onto the SUFFIX's parser-syntax (a JSON `{...}`, pythonic `[...]`,
// qwen `<function=...>`, function_gemma `call:f{...}`), then the
// end-tag-after-it search failed, the call returned `Ok(None)`, and the
// same-chunk suffix was silently dropped until cap/EOS.
//
// The STRUCTURAL fix: each single-section parser's `try_parse_one_call`
// now runs a per-parser `bound_section` step FIRST, returning the body
// bytes BEFORE the wrapper close. The opener-search then operates ONLY
// on the bounded prefix — suffix bytes can NEVER bias the body scan.
//
// These tests construct the attack shape per parser and assert that:
//   * zero calls are extracted (the BAD body is rejected),
//   * the SUFFIX (which contains the parser-syntax bait) survives intact
//     as display text, not silently dropped or partially parsed.

#[test]
fn streaming_json_tools_suffix_object_after_malformed_section_preserved() {
  // Without the bound-first step, balanced_json_object_prefix(payload) would lock onto the
  // suffix `{"name":"x","arguments":{}}` after the closed-malformed
  // section, no end-tag found after it → Ok(None) → suffix dropped.
  // Now bound_section finds the wrapper close BEFORE the body scan,
  // body scan sees only `bad`, returns Ok-empty with end_pos, suffix
  // surfaces verbatim.
  let (display, calls) = run_with_parser(
    Box::new(JsonTools),
    &[r#"<tool_call>bad</tool_call>{"name":"x","arguments":{}} tail"#],
  );
  assert_eq!(
    calls.len(),
    0,
    "malformed body must not produce a call, and the suffix object must not be confused for one in the same section"
  );
  assert_eq!(
    display, r#"{"name":"x","arguments":{}} tail"#,
    "FULL suffix (object literal + tail text) survives the suffix-bait attack"
  );
}

#[test]
fn streaming_json_tools_suffix_object_after_malformed_section_preserved_split_chunk() {
  let (display, calls) = run_with_parser(
    Box::new(JsonTools),
    &[
      "<tool_call>bad",
      r#"</tool_call>{"name":"x","arguments":{}} tail"#,
    ],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(display, r#"{"name":"x","arguments":{}} tail"#);
}

#[test]
fn streaming_pythonic_suffix_call_after_malformed_section_preserved() {
  // Without the bound-first step, pythonic_call_close(payload) would lock
  // onto the suffix `[echo(x=1)]` after the closed-malformed section, no
  // end-tag after it → Ok(None) → suffix dropped.
  let (display, calls) = run_with_parser(
    Box::new(Pythonic),
    &["<|tool_call_start|>bad<|tool_call_end|>[echo(x=1)] tail"],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(
    display, "[echo(x=1)] tail",
    "FULL suffix (pythonic call literal + tail) survives the suffix-bait attack"
  );
}

#[test]
fn streaming_pythonic_suffix_call_after_malformed_section_preserved_split_chunk() {
  let (display, calls) = run_with_parser(
    Box::new(Pythonic),
    &[
      "<|tool_call_start|>bad",
      "<|tool_call_end|>[echo(x=1)] tail",
    ],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(display, "[echo(x=1)] tail");
}

#[test]
fn streaming_qwen3_coder_suffix_function_after_malformed_section_preserved() {
  // Without the bound-first step, payload.find("<function=") would lock
  // onto the suffix `<function=f>...</function>` after the closed-malformed
  // section, forward-scan finds `</function>`, end-tag-after-it search fails
  // → Ok(None) → suffix dropped.
  let (display, calls) = run_with_parser(
    Box::new(Qwen3Coder),
    &["<tool_call>bad</tool_call><function=f><parameter=p>v</parameter></function> tail"],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(
    display, "<function=f><parameter=p>v</parameter></function> tail",
    "FULL suffix (qwen function literal + tail) survives the suffix-bait attack"
  );
}

#[test]
fn streaming_qwen3_coder_suffix_function_after_malformed_section_preserved_split_chunk() {
  let (display, calls) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      "<tool_call>bad",
      "</tool_call><function=f><parameter=p>v</parameter></function> tail",
    ],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(
    display,
    "<function=f><parameter=p>v</parameter></function> tail"
  );
}

#[test]
fn streaming_function_gemma_suffix_call_after_malformed_section_preserved() {
  // Without the bound-first step, payload.find("call:") would lock onto the
  // suffix `call:f{k:v}` after the closed-malformed section, body-scan finds
  // `}`, end-tag-after-it search fails → Ok(None) → suffix dropped.
  let (display, calls) = run_with_parser(
    Box::new(FunctionGemma),
    &["<start_function_call>bad<end_function_call>call:f{k:v} tail"],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(
    display, "call:f{k:v} tail",
    "FULL suffix (function_gemma call literal + tail) survives the suffix-bait attack"
  );
}

#[test]
fn streaming_function_gemma_suffix_call_after_malformed_section_preserved_split_chunk() {
  let (display, calls) = run_with_parser(
    Box::new(FunctionGemma),
    &[
      "<start_function_call>bad",
      "<end_function_call>call:f{k:v} tail",
    ],
  );
  assert_eq!(calls.len(), 0);
  assert_eq!(display, "call:f{k:v} tail");
}

#[test]
fn streaming_glm47_suffix_object_after_malformed_section_preserved() {
  // glm47's `parse()` is permissive: a plain-text body `bad` is
  // accepted as a tool-call name (`glm_parse_plain` returns
  // `ToolCall::new_nameless_id("bad", {})` rather than rejecting), so this
  // single permissive call is emitted at the FIRST wrapper
  // close. The invariant for glm47 is *suffix preservation*: the
  // body scan must not advance into the suffix object. So
  // exactly ONE call (the body's plain-text name) is emitted AND
  // the SUFFIX surfaces verbatim — the suffix-syntax bait is not
  // mis-parsed as part of the malformed section.
  let (display, calls) = run_with_parser(
    Box::new(Glm47),
    &[r#"<tool_call>bad</tool_call>{"name":"y"} tail"#],
  );
  assert_eq!(
    calls.len(),
    1,
    "glm47 is permissive: plain-text body `bad` becomes ToolCall(`bad`); the invariant is suffix preservation, not call rejection"
  );
  assert_eq!(calls[0].name(), "bad", "plain-text body parsed as name");
  assert_eq!(
    display, r#"{"name":"y"} tail"#,
    "FULL suffix (object literal + tail) survives the suffix-bait attack — body scan must not lock onto the suffix object"
  );
}

#[test]
fn streaming_longcat_suffix_object_after_malformed_section_preserved() {
  // Longcat's `parse()` is permissive on `{`-leading bodies but
  // *strict* on the XML/plain-text fallback: a body that is neither
  // a JSON object nor contains `<longcat_arg_key>` returns `Err`
  // (`"longcat: no function name"`), which `try_parse_one_call`
  // surfaces via the `_` match arm as `Ok(Some((Vec::new(), end_pos)))`
  // — zero calls, but the bounded section is still confirmed.
  // The invariant is suffix preservation.
  let (display, calls) = run_with_parser(
    Box::new(Longcat),
    &[r#"<longcat_tool_call>bad</longcat_tool_call>{"name":"y"} tail"#],
  );
  assert_eq!(
    calls.len(),
    0,
    "longcat rejects body `bad` (no `<longcat_arg_key>`, not JSON) → zero calls"
  );
  assert_eq!(
    display, r#"{"name":"y"} tail"#,
    "FULL suffix (object literal + tail) survives the suffix-bait attack — body scan must not lock onto the suffix object"
  );
}

/// **Table audit:** one row per single-section parser, asserting
/// the invariant: a `<wrapper>BAD</wrapper>SUFFIX-WITH-PARSER-SYNTAX`
/// shape preserves the SUFFIX verbatim as display text. The body scan
/// MUST NOT lock onto the suffix-syntax bait (the defect class this
/// guards against drops the entire suffix while waiting for a
/// never-arriving end-tag).
///
/// Call-count expectations are parser-dependent: parsers whose
/// `parse()` rejects malformed bodies (`json_tools`, `pythonic`,
/// `qwen3_coder`, `function_gemma`) emit zero calls; the permissive
/// `glm47` / `longcat` accept a plain-text body as a tool-call name.
/// Both behaviours are baseline — the bound-section step changes neither.
///
/// Multi-block parsers (`kimi_k2`, `minimax_m2`, `gemma4`) are
/// structurally exempt — their per-section opener-vs-end race is
/// already prefix-bounded. Mistral has empty end_tag
/// and is short-circuited by the streaming processor before
/// `try_parse_one_call` is reached.
#[test]
fn try_parse_one_call_suffix_starting_with_parser_syntax_per_parser_audit() {
  struct AuditRow {
    label: &'static str,
    parser: Box<dyn ToolParser>,
    buffer: &'static str,
    expect_display: &'static str,
    expect_calls: usize,
  }
  let rows: Vec<AuditRow> = vec![
    // json_tools: SUFFIX is a JSON object that an unbounded
    // `balanced_json_object_prefix(payload)` would lock onto.
    // `parse()` rejects the body `bad` → zero calls.
    AuditRow {
      label: "json_tools (suffix = JSON object)",
      parser: Box::new(JsonTools),
      buffer: r#"<tool_call>bad</tool_call>{"name":"x","arguments":{}} tail"#,
      expect_display: r#"{"name":"x","arguments":{}} tail"#,
      expect_calls: 0,
    },
    // pythonic: SUFFIX is a `[name(args)]` literal that an unbounded
    // `pythonic_call_close(payload)` would lock onto. `parse()`
    // rejects body `bad` (no `[name(` shape) → zero calls.
    AuditRow {
      label: "pythonic (suffix = [call(args)])",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>bad<|tool_call_end|>[echo(x=1)] tail",
      expect_display: "[echo(x=1)] tail",
      expect_calls: 0,
    },
    // qwen3_coder: SUFFIX is a complete `<function=...>` block that an
    // unbounded `payload.find("<function=")` would lock onto. `parse()`
    // rejects body `bad` (no `<function=` shape) → zero calls.
    AuditRow {
      label: "qwen3_coder (suffix = <function=...>)",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call>bad</tool_call><function=f><parameter=p>v</parameter></function> tail",
      expect_display: "<function=f><parameter=p>v</parameter></function> tail",
      expect_calls: 0,
    },
    // glm47 (None-arm): SUFFIX is a JSON object. `glm_parse_plain`
    // is permissive: body `bad` becomes `ToolCall::new_nameless_id("bad", {})`
    // (one call). The invariant is suffix preservation.
    AuditRow {
      label: "glm47 (suffix = JSON object after non-JSON body)",
      parser: Box::new(Glm47),
      buffer: r#"<tool_call>bad</tool_call>{"name":"y"} tail"#,
      expect_display: r#"{"name":"y"} tail"#,
      expect_calls: 1,
    },
    // longcat: SUFFIX is a JSON object. Unlike glm47, longcat is
    // STRICT on the XML fallback — a body without `<longcat_arg_key>`
    // and not `{`-leading returns `Err` (zero calls).
    AuditRow {
      label: "longcat (suffix = JSON object after non-JSON body)",
      parser: Box::new(Longcat),
      buffer: r#"<longcat_tool_call>bad</longcat_tool_call>{"name":"y"} tail"#,
      expect_display: r#"{"name":"y"} tail"#,
      expect_calls: 0,
    },
    // function_gemma: SUFFIX is a `call:NAME{...}` literal. `parse()`
    // rejects body `bad` (no `call:` shape) → zero calls.
    AuditRow {
      label: "function_gemma (suffix = call:f{k:v})",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>bad<end_function_call>call:f{k:v} tail",
      expect_display: "call:f{k:v} tail",
      expect_calls: 0,
    },
  ];
  for row in rows {
    let (display, calls) = run_with_parser(row.parser, &[row.buffer]);
    assert_eq!(
      calls.len(),
      row.expect_calls,
      "{}: call count must match parser's per-body acceptance baseline (suffix preservation changes neither)",
      row.label,
    );
    assert_eq!(
      display, row.expect_display,
      "{}: FULL suffix bytes must reach display verbatim (not silently dropped, not partially parsed)",
      row.label,
    );
  }
}

/// Unit-level audit at the `try_parse_one_call` boundary: for
/// every single-section parser, the bound-section step must surface a
/// confirmed-bounded section (`Ok(Some((calls, end_pos)))`) where
/// `end_pos` lands at the FIRST wrapper close — even when the SUFFIX
/// after the wrapper close carries parser-syntax bait. This locks the
/// invariant at the unit level so an end-to-end-only regression in
/// `ToolCallProcessor` plumbing cannot mask a parser drifting back to
/// a whole-payload opener search.
///
/// `expect_calls_empty` mirrors each parser's per-body acceptance:
/// strict parsers (`json_tools`, `pythonic`, `qwen3_coder`,
/// `function_gemma`) reject the malformed body → zero calls; the
/// permissive `glm47` / `longcat` accept a plain-text body as a
/// tool-call name. Both are baseline; the bound-section step changes neither.
#[test]
fn try_parse_one_call_suffix_bait_end_pos_lands_at_wrapper_close_per_parser_audit() {
  struct Row {
    label: &'static str,
    parser: Box<dyn ToolParser>,
    buffer: &'static str,
    // Byte position one past the FIRST `</wrapper>` close — the body
    // scan MUST NOT advance past this even when suffix-syntax bait
    // is present.
    expect_end_pos: usize,
    expect_calls_empty: bool,
  }
  let rows = [
    Row {
      label: "json_tools",
      parser: Box::new(JsonTools),
      buffer: r#"<tool_call>bad</tool_call>{"name":"x","arguments":{}} tail"#,
      expect_end_pos: "<tool_call>bad</tool_call>".len(),
      expect_calls_empty: true,
    },
    Row {
      label: "pythonic",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>bad<|tool_call_end|>[echo(x=1)] tail",
      expect_end_pos: "<|tool_call_start|>bad<|tool_call_end|>".len(),
      expect_calls_empty: true,
    },
    Row {
      label: "qwen3_coder",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call>bad</tool_call><function=f><parameter=p>v</parameter></function> tail",
      expect_end_pos: "<tool_call>bad</tool_call>".len(),
      expect_calls_empty: true,
    },
    Row {
      label: "glm47",
      parser: Box::new(Glm47),
      buffer: r#"<tool_call>bad</tool_call>{"name":"y"} tail"#,
      expect_end_pos: "<tool_call>bad</tool_call>".len(),
      // Permissive: plain-text body `bad` → ToolCall::new_nameless_id("bad", {}).
      expect_calls_empty: false,
    },
    Row {
      label: "longcat",
      parser: Box::new(Longcat),
      buffer: r#"<longcat_tool_call>bad</longcat_tool_call>{"name":"y"} tail"#,
      expect_end_pos: "<longcat_tool_call>bad</longcat_tool_call>".len(),
      // Strict on XML fallback: body without `<longcat_arg_key>` and
      // not `{`-leading returns `Err` → zero calls (the `_` arm).
      expect_calls_empty: true,
    },
    Row {
      label: "function_gemma",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>bad<end_function_call>call:f{k:v} tail",
      expect_end_pos: "<start_function_call>bad<end_function_call>".len(),
      expect_calls_empty: true,
    },
  ];
  for row in &rows {
    let result = row
      .parser
      .try_parse_one_call(row.buffer, None)
      .unwrap_or_else(|e| panic!("{}: try_parse_one_call errored: {e}", row.label));
    let (calls, end_pos) = result.unwrap_or_else(|| {
        panic!(
          "{}: confirmed-bounded section expected (the wrapper end-tag is in the buffer), got Ok(None) — regression: opener-search likely locked onto suffix-bait",
          row.label,
        )
      });
    assert_eq!(
      end_pos, row.expect_end_pos,
      "{}: end_pos must land at the FIRST wrapper close — body scan must not advance past the bound prefix",
      row.label,
    );
    assert_eq!(
      calls.is_empty(),
      row.expect_calls_empty,
      "{}: per-parser call-acceptance baseline for malformed body inside bounded prefix (got {:?})",
      row.label,
      calls,
    );
  }
}

// --- STRUCTURAL: orphan value markers hide the real wrapper close -------
//
// Without the context gate, each `bound_section`'s quote/value-aware scanner
// would run over the RAW payload BEFORE the parser opener context is proven.
// Orphan markers
// in MALFORMED bodies (a stray `"` in non-JSON garbage / a `<parameter=`
// before any `<function=` / an `<escape>` before any `call:name{`) fooled
// the syntax-aware scanners into either waiting forever for a missing
// close or skipping past the real wrapper end-tag — silently dropping the
// same-chunk suffix until cap/EOS.
//
// The STRUCTURAL fix: each `bound_section` now RACES the parser opener
// against the first end_tag candidate BEFORE running the syntax-aware
// scanner. When the opener is missing in `payload[..first_end_rel]` (no
// parser context proven), the bounded close falls back to the plain
// end_tag position — orphan scanner-bait markers can never hide the real
// wrapper close.
//
// These tests construct the orphan-marker prefix shape per parser and
// assert the safe end-to-end outcome: ZERO (or permissive) calls + FULL
// suffix bytes reach display verbatim.

#[test]
fn streaming_json_tools_orphan_quote_in_malformed_body_does_not_hide_close() {
  // In payload `<tool_call>bad"</tool_call>{"name":"x"}` the
  // orphan `"` BEFORE the wrapper close could fool the JSON-string-quote-
  // aware scanner into entering string state at `"`, walking through
  // `</tool_call>{`, finding the matching `"` deep in the suffix
  // (`"name"`), continuing… no `</tool_call>` ever appears OUTSIDE
  // strings in the body, scanner returns None, `bound_section` returns
  // None, caller returns `Ok(None)` → suffix dropped silently until
  // cap/EOS.
  //
  // Race(`{`, end_tag) → no `{` in `payload[..first_end_rel]`
  // (only `bad"`) → PlainEnd → end_pos lands at the FIRST wrapper
  // close; bounded body `bad"` is unbalanced JSON → empty calls;
  // suffix `{"name":"x"}` reaches display.
  let (display, calls) = run_with_parser(
    Box::new(JsonTools),
    &[r#"<tool_call>bad"</tool_call>{"name":"x"}"#],
  );
  assert_eq!(
    calls.len(),
    0,
    "orphan `\"` BEFORE wrapper close must not hide the real end-tag",
  );
  assert_eq!(
    display, r#"{"name":"x"}"#,
    "FULL suffix bytes reach display — body scan must not lock onto the orphan `\"`",
  );
}

#[test]
fn streaming_pythonic_orphan_quote_in_malformed_body_does_not_hide_close() {
  // In payload `<|tool_call_start|>bad'<|tool_call_end|>[echo(x=1)] tail`
  // the orphan `'` BEFORE the wrapper close could fool the Python-quote-
  // aware scanner (which tracks both `'` and `"`) into entering string
  // state at `'`, walking forward looking for matching `'`… never
  // finding one outside strings in the body → bound returns None →
  // Ok(None) → suffix dropped.
  //
  // Race(`[`, end_tag) → no `[` in `payload[..first_end_rel]`
  // (only `bad'`) → PlainEnd → end_pos lands at FIRST wrapper close;
  // bounded body `bad'` has no `[name(` → empty calls; suffix
  // `[echo(x=1)] tail` reaches display.
  let (display, calls) = run_with_parser(
    Box::new(Pythonic),
    &["<|tool_call_start|>bad'<|tool_call_end|>[echo(x=1)] tail"],
  );
  assert_eq!(
    calls.len(),
    0,
    "orphan `'` BEFORE wrapper close must not hide the real end-tag",
  );
  assert_eq!(
    display, "[echo(x=1)] tail",
    "FULL suffix bytes reach display — body scan must not lock onto the orphan `'`",
  );
}

#[test]
fn streaming_qwen3_coder_orphan_parameter_in_malformed_body_does_not_hide_close() {
  // In payload
  // `<tool_call>bad<parameter=p></tool_call><function=f><parameter=p>v</parameter></function> tail`
  // the orphan `<parameter=` BEFORE any `<function=` could fool the
  // parameter-value-aware scanner into entering a parameter region at
  // the orphan, looking for matching `</parameter>` that never lands
  // inside the body (the `</parameter>` is in the SUFFIX past the
  // wrapper close) → scanner returns None → bound returns None →
  // Ok(None) → suffix dropped silently until cap/EOS.
  //
  // Race(`<function=`, end_tag) → no `<function=` in
  // `payload[..first_end_rel]` (only `bad<parameter=p>`) → PlainEnd
  // → end_pos lands at FIRST wrapper close; bounded body
  // `bad<parameter=p>` has no `<function=` → empty calls; suffix
  // `<function=f>...` reaches display.
  let (display, calls) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      "<tool_call>bad<parameter=p></tool_call><function=f><parameter=p>v</parameter></function> tail",
    ],
  );
  assert_eq!(
    calls.len(),
    0,
    "orphan `<parameter=` BEFORE wrapper close must not hide the real end-tag",
  );
  assert_eq!(
    display, "<function=f><parameter=p>v</parameter></function> tail",
    "FULL suffix bytes reach display — body scan must not lock onto the orphan `<parameter=`",
  );
}

#[test]
fn streaming_glm47_orphan_quote_in_malformed_body_does_not_hide_close() {
  // glm47 Object arm: body starts with `{` so classify=Object. The
  // race for the Object arm is trivially `{` at byte 0 < any
  // end_tag → OpenerProven → JSON-string-quote-aware scan runs.
  //
  // This test exercises the Object arm's gated path for a CLEAN
  // body: the quote-aware scan correctly finds the wrapper close and
  // the same-chunk suffix is preserved. (The body-balancer step then
  // rejects the bounded JSON as invalid because the body is
  // intentionally malformed; the call surface is glm47's permissive
  // plain-text fallback OR zero — both are baseline. The
  // invariant verified here is *suffix preservation* — the close
  // must land at the FIRST wrapper close, never advanced past it.)
  let (display, calls) = run_with_parser(
    Box::new(Glm47),
    &[r#"<tool_call>{garbage}</tool_call>{"name":"y"} tail"#],
  );
  // glm47 is permissive: a body like `{garbage}` is treated as a
  // plain-text name by glm_parse_plain (no `<arg_key>`, not valid
  // JSON), so one call surfaces with name=`{garbage}`. The
  // assertion is suffix preservation.
  assert_eq!(calls.len(), 1, "glm47 permissive parse on `{{garbage}}`");
  assert_eq!(calls[0].name(), "{garbage}");
  assert_eq!(
    display, r#"{"name":"y"} tail"#,
    "FULL suffix bytes reach display — Object arm race must close at the FIRST wrapper end-tag",
  );
}

#[test]
fn streaming_glm47_orphan_bracket_in_malformed_body_does_not_hide_close() {
  // glm47 Array arm: body starts with `[` so classify=Array. The
  // race for the Array arm is trivially `[` at byte 0 < any end_tag
  // → OpenerProven → JSON-string-quote-aware scan runs. Cleanly-
  // closed `[garbage]` body in the Array arm: the scan finds the
  // wrapper close, suffix preserved.
  let (display, calls) = run_with_parser(
    Box::new(Glm47),
    &[r#"<tool_call>[garbage]</tool_call>{"name":"y"} tail"#],
  );
  // glm47 permissive: `[garbage]` body, glm_parse_plain treats as name.
  assert_eq!(calls.len(), 1, "glm47 permissive parse on `[garbage]`");
  assert_eq!(calls[0].name(), "[garbage]");
  assert_eq!(
    display, r#"{"name":"y"} tail"#,
    "FULL suffix bytes reach display — Array arm race must close at the FIRST wrapper end-tag",
  );
}

#[test]
fn streaming_glm47_orphan_arg_key_in_malformed_body_does_not_hide_close() {
  // glm47 None arm with the arg-key race preserved: body has an `<arg_key>`
  // opener BEFORE the wrapper end-tag, so `first_key < first_end`
  // routes to the xml_value_aware scanner. An UNTERMINATED `<arg_key>`
  // (no matching `</arg_key>`) stays a benign plain text segment for
  // the scanner — the scanner only skips `<arg_value>...</arg_value>`
  // regions. So the wrapper close is found and the suffix is
  // preserved.
  //
  // Invariant for the None arm: the existing arg-key race plumbing
  // continues to surface the FIRST wrapper close, not silently drop
  // the same-chunk suffix.
  let (display, calls) = run_with_parser(
    Box::new(Glm47),
    &[r#"<tool_call>bad<arg_key></tool_call>{"name":"y"} tail"#],
  );
  // glm47 permissive: body parsed via glm_parse_plain → name=`bad`.
  assert_eq!(calls.len(), 1, "glm47 permissive parse extracts one call");
  assert_eq!(calls[0].name(), "bad");
  assert_eq!(
    display, r#"{"name":"y"} tail"#,
    "FULL suffix bytes reach display — the None-arm arg-key race stays correct against the orphan-quote case",
  );
}

#[test]
fn streaming_longcat_orphan_quote_in_malformed_body_does_not_hide_close() {
  // Longcat Object arm: body starts with `{` so the `{`-leading
  // fast-path runs. Race: `{` at byte 0 < end_tag → OpenerProven
  // → JSON-string-quote-aware scan runs. Cleanly-closed `{garbage}`
  // body: the scan finds the wrapper close, suffix preserved.
  let (display, calls) = run_with_parser(
    Box::new(Longcat),
    &[r#"<longcat_tool_call>{garbage}</longcat_tool_call>{"name":"y"} tail"#],
  );
  // Longcat's `{`-leading fast-path requires valid JSON; `{garbage}`
  // is invalid JSON → falls through the `serde_json::from_str` check
  // → drops into the `<longcat_arg_key>` path which errors (no
  // function name) → try_parse_one_call's `_` arm returns
  // `Ok(Some((Vec::new(), end_pos)))`. Zero calls.
  assert_eq!(
    calls.len(),
    0,
    "longcat strict on malformed `{{garbage}}` body"
  );
  assert_eq!(
    display, r#"{"name":"y"} tail"#,
    "FULL suffix bytes reach display — Object arm race must close at the FIRST wrapper end-tag",
  );
}

#[test]
fn streaming_function_gemma_orphan_escape_in_malformed_body_does_not_hide_close() {
  // In payload
  // `<start_function_call>bad<escape><end_function_call>call:f{k:v} tail`
  // the orphan `<escape>` BEFORE any `call:` could fool the escape-
  // region-aware scanner into entering a value region at the orphan,
  // looking for matching `<escape>` close that never lands inside the
  // body (the body only has ONE `<escape>`) → scanner returns None →
  // bound returns None → Ok(None) → suffix dropped silently.
  //
  // Race(`call:`, end_tag) → no `call:` in
  // `payload[..first_end_rel]` (only `bad<escape>`) → PlainEnd →
  // end_pos lands at FIRST wrapper close; bounded body `bad<escape>`
  // has no `call:` → empty calls; suffix `call:f{k:v} tail` reaches
  // display.
  let (display, calls) = run_with_parser(
    Box::new(FunctionGemma),
    &["<start_function_call>bad<escape><end_function_call>call:f{k:v} tail"],
  );
  assert_eq!(
    calls.len(),
    0,
    "orphan `<escape>` BEFORE wrapper close must not hide the real end-tag",
  );
  assert_eq!(
    display, "call:f{k:v} tail",
    "FULL suffix bytes reach display — body scan must not lock onto the orphan `<escape>`",
  );
}

/// Table audit: one row per patched parser, asserting that
/// orphan value-markers BEFORE the parser opener context do NOT hide
/// the real wrapper close. The body scan must close at the FIRST
/// wrapper end-tag and the same-chunk suffix must reach display
/// verbatim.
///
/// Per-arm coverage:
/// * `json_tools` — orphan `"` (string-open bait).
/// * `pythonic` — orphan `'` (Python string-open bait).
/// * `qwen3_coder` — orphan `<parameter=` (value-region bait).
/// * `glm47` None arm — orphan `<arg_key>` text (arg-key race
///   confirmed: `<arg_key>` opener seen but unterminated body stays
///   benign under the xml_value_aware scanner).
/// * `longcat` Object arm — malformed `{`-leading body.
/// * `function_gemma` — orphan `<escape>` (value-region bait).
#[test]
fn try_parse_one_call_orphan_value_markers_per_parser_audit() {
  struct Row {
    label: &'static str,
    parser: Box<dyn ToolParser>,
    buffer: &'static str,
    // Byte position one past the FIRST wrapper end-tag — the body
    // scan MUST NOT advance past this even when an orphan value
    // marker appears in `payload[..first_end_rel]`.
    expect_end_pos: usize,
  }
  let rows = [
    Row {
      label: "json_tools (orphan `\"`)",
      parser: Box::new(JsonTools),
      buffer: r#"<tool_call>bad"</tool_call>{"name":"x"}"#,
      expect_end_pos: r#"<tool_call>bad"</tool_call>"#.len(),
    },
    Row {
      label: "pythonic (orphan `'`)",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>bad'<|tool_call_end|>[echo(x=1)] tail",
      expect_end_pos: "<|tool_call_start|>bad'<|tool_call_end|>".len(),
    },
    Row {
      label: "qwen3_coder (orphan `<parameter=`)",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call>bad<parameter=p></tool_call><function=f><parameter=p>v</parameter></function> tail",
      expect_end_pos: "<tool_call>bad<parameter=p></tool_call>".len(),
    },
    Row {
      label: "glm47 None arm (orphan `<arg_key>`)",
      parser: Box::new(Glm47),
      buffer: r#"<tool_call>bad<arg_key></tool_call>{"name":"y"} tail"#,
      expect_end_pos: r#"<tool_call>bad<arg_key></tool_call>"#.len(),
    },
    Row {
      label: "longcat Object arm (malformed `{`-leading)",
      parser: Box::new(Longcat),
      buffer: r#"<longcat_tool_call>{garbage}</longcat_tool_call>{"name":"y"} tail"#,
      expect_end_pos: r#"<longcat_tool_call>{garbage}</longcat_tool_call>"#.len(),
    },
    Row {
      label: "function_gemma (orphan `<escape>`)",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>bad<escape><end_function_call>call:f{k:v} tail",
      expect_end_pos: "<start_function_call>bad<escape><end_function_call>".len(),
    },
  ];
  for row in &rows {
    let result = row
      .parser
      .try_parse_one_call(row.buffer, None)
      .unwrap_or_else(|e| panic!("{}: try_parse_one_call errored: {e}", row.label));
    let (_, end_pos) = result.unwrap_or_else(|| {
        panic!(
          "{}: confirmed-bounded section expected (the wrapper end-tag is in the buffer), got Ok(None) — regression: orphan value marker hid the real wrapper close",
          row.label,
        )
      });
    assert_eq!(
      end_pos, row.expect_end_pos,
      "{}: end_pos must land at the FIRST wrapper close — orphan value marker must not bias the body scan",
      row.label,
    );
  }
}

// --- STRUCTURAL: stray opener literals fool a generic race --------------
//
// A generic `race_opener_vs_end_tag` used a GENERIC `payload.find(opener_lit)`
// check. A stray opener literal in MALFORMED body bytes still satisfied
// "opener before end_tag" → OpenerProven → syntax-aware scanner ran → an
// orphan scanner-bait marker in the body could still hide the wrapper
// close — defeated by injecting BOTH a bait marker AND a bare opener
// literal in the same malformed body.
//
// Examples that broke it:
//   * json_tools: `<tool_call>bad{"</tool_call>{"name":"x"}` — the `{`
//     in `bad{"` satisfied `payload.find("{")`, OpenerProven →
//     quote-aware scan saw the orphan `"` → wait-forever → suffix lost.
//   * pythonic: `<|tool_call_start|>bad['<|tool_call_end|>[name(x=1)]
//     tail` — the `[` in `bad[` satisfied `payload.find("[")`,
//     OpenerProven → Python-quote-aware scan saw orphan `'` → wait
//     forever → suffix lost.
//   * function_gemma: `<start_function_call>bad call:<escape>
//     <end_function_call>call:f{k:v} tail` — the `call:` in `bad call:`
//     satisfied `payload.find("call:")`, OpenerProven → escape-
//     region-aware scan saw orphan `<escape>` → wait forever → suffix
//     lost.
//
// The STRUCTURAL fix: replace the generic literal race with per-parser
// CONTEXT PREDICATES that demand the parser's structural opening shape
// (`{` as first non-whitespace; `[name(`; `<function=name>`;
// `call:name{`). A stray opener literal in body garbage does NOT match
// the structural shape, so the context predicate returns false, the gate
// returns the plain end_tag position, and the suffix is preserved.

#[test]
fn streaming_json_tools_stray_open_brace_in_malformed_body_does_not_hide_close() {
  // Motivating case for json_tools. Body `bad{"` contains a `{` so
  // a generic `payload.find("{")` was satisfied (OpenerProven). The quote-
  // aware scanner then entered string state at the orphan `"` after the
  // `{`, walked through `</tool_call>{`, found the `"` of `"name"` in
  // the suffix — no `</tool_call>` outside strings in the body → scanner
  // returns None → bound returns None → Ok(None) → suffix dropped.
  //
  // The predicate `json_object_context_proven` requires the FIRST non-
  // whitespace byte to be `{`. Body `bad{"` starts with `b` not `{` →
  // predicate false → PlainEnd → end_pos lands at FIRST wrapper close;
  // bounded body `bad{"` is unbalanced JSON → empty calls; suffix
  // `{"name":"x"}` reaches display.
  let (display, calls) = run_with_parser(
    Box::new(JsonTools),
    &[r#"<tool_call>bad{"</tool_call>{"name":"x"}"#],
  );
  assert_eq!(
    calls.len(),
    0,
    "stray `{{` in malformed body must not unlock JSON-quote-aware scan",
  );
  assert_eq!(
    display, r#"{"name":"x"}"#,
    "FULL suffix bytes reach display — context predicate requires `{{` as LEADING shape, not any-position match",
  );
}

#[test]
fn streaming_pythonic_stray_open_bracket_in_malformed_body_does_not_hide_close() {
  // Motivating case for pythonic. Body `bad[` contains a `[` so a generic
  // `payload.find("[")` was satisfied (OpenerProven). The Python-quote-
  // aware scanner then entered single-quote state at the orphan `'`,
  // walked forward looking for matching `'` that never lands inside the
  // body → scanner returns None → bound returns None → Ok(None) →
  // suffix dropped.
  //
  // The predicate `pythonic_call_context_proven` requires `[name(`
  // shape. Body `bad[` has `[` but no name+`(` after → predicate false
  // → PlainEnd → end_pos lands at FIRST wrapper close; bounded body
  // `bad['` has no `[name(` → empty calls; suffix `[name(x=1)] tail`
  // reaches display.
  let (display, calls) = run_with_parser(
    Box::new(Pythonic),
    &["<|tool_call_start|>bad['<|tool_call_end|>[name(x=1)] tail"],
  );
  assert_eq!(
    calls.len(),
    0,
    "stray `[` (without `[name(` shape) in malformed body must not unlock Python-quote-aware scan",
  );
  assert_eq!(
    display, "[name(x=1)] tail",
    "FULL suffix bytes reach display — context predicate requires `[name(` SHAPE, not just any `[`",
  );
}

#[test]
fn streaming_qwen3_coder_stray_function_open_in_malformed_body_does_not_hide_close() {
  // Motivating case for qwen3_coder. Body `bad<function= ` contains
  // the literal `<function=` so a generic `payload.find("<function=")` was
  // satisfied (OpenerProven). The parameter-value-aware scanner then
  // looked for end_tag outside every `<parameter=...></parameter>`
  // region; the only `</parameter>` is in the SUFFIX past the wrapper
  // close so the scan walks past the real `</tool_call>` looking for a
  // `</parameter>` that anchors a region close — fooled → bound returns
  // None → Ok(None) → suffix dropped.
  //
  // The predicate `qwen_function_context_proven` requires the FULL
  // `<function=NAME>` tag shape. Body `bad<function= ` has `<function=`
  // but no name+`>` after → predicate false → PlainEnd → end_pos lands
  // at FIRST wrapper close; bounded body has no valid `<function=NAME>`
  // → empty calls; suffix `<function=f><parameter=p>v</parameter>
  // </function> tail` reaches display.
  let (display, calls) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      "<tool_call>bad<function= <parameter=p></tool_call><function=f><parameter=p>v</parameter></function> tail",
    ],
  );
  assert_eq!(
    calls.len(),
    0,
    "stray `<function=` (without `NAME>` close) in malformed body must not unlock parameter-value-aware scan",
  );
  assert_eq!(
    display, "<function=f><parameter=p>v</parameter></function> tail",
    "FULL suffix bytes reach display — context predicate requires `<function=NAME>` SHAPE, not just `<function=` literal",
  );
}

#[test]
fn streaming_function_gemma_stray_call_in_malformed_body_does_not_hide_close() {
  // Motivating case for function_gemma. Body `bad call:<escape>`
  // contains the literal `call:` so a generic `payload.find("call:")` was
  // satisfied (OpenerProven). The escape-region-aware scanner then
  // entered a value region at the orphan `<escape>`, looking for the
  // matching `<escape>` close that never lands inside the body
  // (suffix `call:f{k:v}` contains no second `<escape>`) → scanner
  // returns None → bound returns None → Ok(None) → suffix dropped.
  //
  // The predicate `function_gemma_call_context_proven` requires the
  // FULL `call:NAME{` shape. Body `bad call:` has `call:` but no
  // name+`{` after → predicate false → PlainEnd → end_pos lands at
  // FIRST wrapper close; bounded body has no valid `call:NAME{` →
  // empty calls; suffix `call:f{k:v} tail` reaches display.
  let (display, calls) = run_with_parser(
    Box::new(FunctionGemma),
    &["<start_function_call>bad call:<escape><end_function_call>call:f{k:v} tail"],
  );
  assert_eq!(
    calls.len(),
    0,
    "stray `call:` (without `NAME{{` shape) in malformed body must not unlock escape-region-aware scan",
  );
  assert_eq!(
    display, "call:f{k:v} tail",
    "FULL suffix bytes reach display — context predicate requires `call:NAME{{` SHAPE, not just `call:` literal",
  );
}

/// Table audit: per-parser stray-opener variants where the
/// malformed body contains BOTH (a) an opener-LITERAL that a generic
/// race accepted and (b) an orphan scanner-bait marker. The context
/// predicate must reject the stray literal because it does not match the
/// parser's STRUCTURAL opening shape — end_pos must land at the FIRST
/// wrapper close, the same-chunk suffix must reach display verbatim.
///
/// Per-arm coverage:
/// * `json_tools` — stray `{` (not leading) + orphan `"`.
/// * `pythonic` — stray `[` (without `name(`) + orphan `'`.
/// * `qwen3_coder` — stray `<function=` (without `NAME>`) + orphan
///   `<parameter=`.
/// * `function_gemma` — stray `call:` (without `NAME{`) + orphan
///   `<escape>`.
/// * `glm47 Object` — body shape `bad{garbage}` is REJECTED by the
///   predicate (first non-ws byte is `b` not `{`); the Object arm's
///   classify dispatch never enters here for non-`{`-leading bodies, so
///   we exercise the predicate via the None arm with `<arg_key>`
///   absence (baseline glm47 None-arm test stays green: an absent
///   `<arg_key>` triggers the plain-end fallback). The Object/Array
///   arms' predicate is consistent with `classify_json_payload_start`
///   already determining the leading shape, so the predicate trivially
///   passes when the arm is selected — no stray-opener attack is
///   possible past the classifier.
/// * `longcat Object` — same as glm47 Object: classify already
///   determined the leading shape.
#[test]
fn try_parse_one_call_stray_opener_per_parser_audit() {
  struct Row {
    label: &'static str,
    parser: Box<dyn ToolParser>,
    buffer: &'static str,
    // Byte position one past the FIRST wrapper end-tag — the body scan
    // MUST close here even when stray opener literals + orphan bait
    // appear in the malformed body.
    expect_end_pos: usize,
  }
  let rows = [
    Row {
      label: "json_tools (stray `{` + orphan `\"`)",
      parser: Box::new(JsonTools),
      buffer: r#"<tool_call>bad{"</tool_call>{"name":"x"}"#,
      expect_end_pos: r#"<tool_call>bad{"</tool_call>"#.len(),
    },
    Row {
      label: "pythonic (stray `[` + orphan `'`)",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>bad['<|tool_call_end|>[name(x=1)] tail",
      expect_end_pos: "<|tool_call_start|>bad['<|tool_call_end|>".len(),
    },
    Row {
      label: "qwen3_coder (stray `<function=` + orphan `<parameter=`)",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call>bad<function= <parameter=p></tool_call><function=f><parameter=p>v</parameter></function> tail",
      expect_end_pos: "<tool_call>bad<function= <parameter=p></tool_call>".len(),
    },
    Row {
      label: "function_gemma (stray `call:` + orphan `<escape>`)",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>bad call:<escape><end_function_call>call:f{k:v} tail",
      expect_end_pos: "<start_function_call>bad call:<escape><end_function_call>".len(),
    },
    // glm47 None arm: the predicate is the `<arg_key>` literal. A body
    // without `<arg_key>` and without `<arg_value>` orphan still falls
    // back cleanly to the plain end_tag. Locked here as a baseline guard:
    // a stray `<arg_value>` (orphan-value bait) before any end_tag
    // and without `<arg_key>` triggers PlainEnd cleanly.
    Row {
      label: "glm47 None arm (stray `<arg_value>` without `<arg_key>`)",
      parser: Box::new(Glm47),
      buffer: r#"<tool_call>bad<arg_value></tool_call>{"name":"y"} tail"#,
      expect_end_pos: r#"<tool_call>bad<arg_value></tool_call>"#.len(),
    },
  ];
  for row in &rows {
    let result = row
      .parser
      .try_parse_one_call(row.buffer, None)
      .unwrap_or_else(|e| panic!("{}: try_parse_one_call errored: {e}", row.label));
    let (_, end_pos) = result.unwrap_or_else(|| {
        panic!(
          "{}: confirmed-bounded section expected (the wrapper end-tag is in the buffer), got Ok(None) — regression: stray opener literal unlocked syntax-aware scan and orphan marker hid the wrapper close",
          row.label,
        )
      });
    assert_eq!(
      end_pos, row.expect_end_pos,
      "{}: end_pos must land at the FIRST wrapper close — stray opener literal must not satisfy the structural context predicate",
      row.label,
    );
  }
}

// ----------------------------------------------------------------------
// Predicate ↔ parser-body recognizer drift.
// ----------------------------------------------------------------------
// The per-parser context predicates gate the syntax-aware
// body scanners behind PROOF of the parser's grammar. Two
// drift cases existed where the predicate grammar diverged from what
// `find_*_call` / `try_parse_one_call` actually recognise:
//
// pythonic:
//   * Predicate rejected digit-leading names (`is_alphabetic` check).
//     Parser's `find_pythonic_call` accepts ANY non-empty ASCII
//     alphanumeric/underscore run (`\w+`) before `(`.
//     False-negative: `[1tool(s='<|tool_call_end|>')]<|tool_call_end|>`
//     → predicate rejects, plain-end used, in-string end-marker treated
//     as wrapper close, real call dropped.
//   * Predicate allowed whitespace between `[` and name AND between
//     name and `(`. Parser does NOT allow whitespace there.
//     False-positive: `bad[name (<|tool_call_end|>...` → predicate
//     accepts, recreates the stray-opener failure mode.
//
// function_gemma:
//   * Predicate skipped whitespace between name and `{`. Parser's
//     `try_parse_one_call` and `gemma_call` require IMMEDIATE `{` after
//     the name. False-positive: `bad call:f {<escape>...` satisfies
//     predicate but isn't a valid opener; escape-aware scanner treats
//     orphan `<escape>` as value region → returns None → hides wrapper
//     close.
//
// The structural fix: extract the per-parser call-start recognizer
// into a shared helper (`pythonic_call_start_at` /
// `find_first_pythonic_call_start`, `function_gemma_call_start_at` /
// `find_first_function_gemma_call_start`) used by BOTH the predicate
// and the parser body. The predicate becomes
// `find_first_*_call_start(prefix).is_some()` — it is impossible for
// the predicate to accept a payload the parser would reject (or
// vice-versa) because they share the same recognizer code.

#[test]
fn streaming_pythonic_digit_leading_name_with_in_string_end_marker_does_not_drop_call() {
  // False-negative case. The body
  // `[1tool(s='<|tool_call_end|>')]` is a legitimate pythonic call:
  // Python's `\w+` accepts digit-leading names and the args use a
  // single-quoted string carrying the wrapper end-marker literal
  // verbatim. The PARSER (`find_pythonic_call`) walks every `[` and
  // accepts ANY non-empty alnum/underscore name run — `1tool` is fine.
  //
  // A `pythonic_call_context_proven` running a SEPARATE name
  // check that required `is_ascii_alphabetic() || _` for the first
  // byte would REJECT `[1tool(`. With the predicate false the
  // gate returns plain-end → end_pos lands at the FIRST `<|tool_call_end|>`
  // literal (the one INSIDE the single-quoted string) → bounded body
  // `[1tool(s='` has no closing `)]` → empty calls → the suffix
  // `')]<|tool_call_end|> tail` reaches display verbatim — the real
  // call silently dropped.
  //
  // The shared `pythonic_call_start_at` is used by both predicate and parser
  // body: both accept `[1tool(`. Predicate true → quote-aware scan
  // skips the single-quoted string → end_pos at the FIRST end-tag
  // OUTSIDE every string (the SECOND `<|tool_call_end|>` literal) →
  // bounded body is the full `[1tool(s='<|tool_call_end|>')]` → call
  // extracted with `s = "<|tool_call_end|>"` intact, suffix ` tail`
  // reaches display.
  let (display, calls) = run_with_parser(
    Box::new(Pythonic),
    &["<|tool_call_start|>[1tool(s='<|tool_call_end|>')]<|tool_call_end|> tail"],
  );
  assert_eq!(
    calls.len(),
    1,
    "digit-leading pythonic name MUST be accepted by the shared recognizer",
  );
  assert_eq!(calls[0].name(), "1tool");
  assert_eq!(
    *calls[0].arguments(),
    serde_json::json!({ "s": "<|tool_call_end|>" }),
    "in-single-quoted-string `<|tool_call_end|>` literal MUST survive the quote-aware scan when the recognizer accepts the digit-leading name",
  );
  assert_eq!(
    display, " tail",
    "FULL suffix (just the ` tail` past the SECOND wrapper end-tag) reaches display — the in-string end-marker MUST NOT be treated as the wrapper close",
  );
}

#[test]
fn streaming_pythonic_stray_open_bracket_with_whitespace_in_malformed_body_does_not_hide_close() {
  // False-positive case. The body `bad[name (`
  // contains `[name` followed by a SPACE and then `(`. The parser's
  // `find_pythonic_call` requires IMMEDIATE `(` after the name run
  // (no whitespace — `bytes[j] == b'('` is the check), so the parser
  // rejects this as a call start.
  //
  // A `pythonic_call_context_proven` that ALLOWED whitespace
  // between the name and `(` (skipping ` \t\n\r` before the `(`
  // check) would ACCEPT `bad[name (` → context proven
  // → quote-aware scanner runs but finds no end-tag outside strings
  // (there are no strings here, and no `<|tool_call_end|>` literal in
  // the body) → bound returns None → Ok(None) → buffer keeps
  // collecting → cap-recovery drops the suffix at the cap or EOS.
  //
  // The shared `pythonic_call_start_at` is used by both predicate and parser
  // body: both REJECT `bad[name (` (whitespace between name and `(`
  // is not part of the grammar). Predicate false → plain-end →
  // end_pos lands at the FIRST `<|tool_call_end|>` → empty calls,
  // suffix `[real(x=1)] tail` reaches display (parser-less display
  // path because the suffix has no `<|tool_call_start|>` wrapper).
  let (display, calls) = run_with_parser(
    Box::new(Pythonic),
    &["<|tool_call_start|>bad[name (<|tool_call_end|>[real(x=1)] tail"],
  );
  assert_eq!(
    calls.len(),
    0,
    "stray `[name (` (whitespace before `(`) MUST NOT context-prove pythonic — predicate must match the parser's no-whitespace recognizer",
  );
  assert_eq!(
    display, "[real(x=1)] tail",
    "FULL suffix bytes reach display — the wrapper close MUST be hit at the FIRST end-tag when the predicate correctly rejects the whitespace-bearing opener",
  );
}

#[test]
fn streaming_function_gemma_stray_call_with_whitespace_in_malformed_body_does_not_hide_close() {
  // The body `bad call:f {<escape>` has `call:f`
  // followed by a SPACE and then `{`. The parser's
  // `try_parse_one_call` / `gemma_call` require IMMEDIATE `{` after
  // the name run (no whitespace — `bytes[j] != b'{'` bails without
  // skipping whitespace), so the parser rejects this as a call start.
  //
  // A `function_gemma_call_context_proven` that ALLOWED
  // whitespace between the name and `{` (skipping ` \t\n\r` before
  // the `{` check) would ACCEPT `bad call:f {` →
  // context proven → escape-aware scanner enters an escape region at
  // the orphan `<escape>` looking for a matching `<escape>` close
  // that doesn't exist in the body (the suffix's `call:f{k:v}` has
  // no second `<escape>`) → bound returns None → Ok(None) → buffer
  // keeps collecting → cap-recovery drops the suffix at the cap or
  // EOS.
  //
  // The shared `function_gemma_call_start_at` is used by both predicate and
  // parser body: both REJECT `bad call:f {` (whitespace between
  // name and `{` is not part of the grammar). Predicate false →
  // plain-end → end_pos lands at the FIRST `<end_function_call>`
  // → empty calls, suffix `call:f{k:v} tail` reaches display (no
  // `<start_function_call>` wrapper in the suffix → display path).
  let (display, calls) = run_with_parser(
    Box::new(FunctionGemma),
    &["<start_function_call>bad call:f {<escape><end_function_call>call:f{k:v} tail"],
  );
  assert_eq!(
    calls.len(),
    0,
    "stray `call:f {{` (whitespace before `{{`) MUST NOT context-prove function_gemma — predicate must match the parser's no-whitespace recognizer",
  );
  assert_eq!(
    display, "call:f{k:v} tail",
    "FULL suffix bytes reach display — the wrapper close MUST be hit at the FIRST end-tag when the predicate correctly rejects the whitespace-bearing opener",
  );
}

// ===== qwen3_coder predicate / recognizer drift =========================
//
// The pythonic + function_gemma drift was fixed by sharing each parser's
// call-start recognizer between the predicate and the parser body. The
// audit row for qwen3_coder only covered `<function=foo>` (an
// `[A-Za-z0-9_-]+` name) and `<function= ` (orphan), so it MISSED the
// dotted/spaced-name case: the parser body's `find('>')` accepts any
// bytes before `>` (foo.bar, foo bar, foo:1, ...), but a stricter predicate
// restricted NAME to `[A-Za-z0-9_-]+` — a parser-accepted body would
// fail the predicate, the gate would plain-close on an in-parameter
// `</tool_call>` literal, and the call would silently disappear.
//
// The structural fix: extract `qwen_function_open_at` /
// `find_first_qwen_function_open` as the shared recognizer used by the
// predicate, the parser body's gate in `try_parse_one_call`, AND
// `Qwen3Coder::parse`'s opener search. The recognizer accepts ANY
// non-empty `<function=NAME>` opener whose NAME contains neither `>`
// nor `<` — matching the parser body's accepted grammar exactly.

#[test]
fn streaming_qwen3_coder_dotted_name_with_in_parameter_end_marker_does_not_drop_call() {
  // False-negative case. The body
  // `<function=foo.bar><parameter=p>v</parameter></function>` is a
  // legitimate qwen3_coder call: `Qwen3Coder::parse` finds the first
  // `>` to terminate the name (`foo.bar` is accepted because the dot is
  // neither `>` nor `<`), and the parameter value is opaque text.
  //
  // A `qwen_function_context_proven` that restricted NAME to
  // `[A-Za-z0-9_-]+` would let the dot REJECT the open-tag. With the
  // predicate false the gate returns the plain end_tag position → in
  // this baseline shape the FIRST `</tool_call>` is the real wrapper
  // close → call extracts. So a bare dotted-name body would not
  // fail visibly. But add a `</tool_call>` literal INSIDE the
  // parameter value (parameter values can carry the wrapper end-tag
  // verbatim), and the plain-end gate locks onto THAT
  // in-parameter end-tag literal → bounded body is truncated to the
  // bytes before `</tool_call>` → `</function>` close is not in
  // the bounded prefix → empty calls → the rest of the body PLUS the
  // real wrapper close PLUS the suffix all reach display verbatim,
  // silently dropping the call.
  //
  // The shared `qwen_function_open_at` is used by both predicate and parser
  // body: both accept `<function=foo.bar>`. Predicate true → parameter-
  // value-aware end-tag scan SKIPS the `<parameter=p>...</parameter>`
  // region whole → end_pos lands at the FIRST `</tool_call>` OUTSIDE
  // every parameter region (the real wrapper close) → call extracted
  // with `p` containing `</tool_call>` intact, suffix ` tail` reaches
  // display.
  let (display, calls) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      "<tool_call><function=foo.bar><parameter=p>contains </tool_call> bytes</parameter></function></tool_call> tail",
    ],
  );
  assert_eq!(
    calls.len(),
    1,
    "dotted-name qwen3_coder body MUST be accepted by the shared recognizer",
  );
  assert_eq!(calls[0].name(), "foo.bar");
  assert_eq!(
    *calls[0].arguments(),
    serde_json::json!({ "p": "contains </tool_call> bytes" }),
    "in-parameter `</tool_call>` literal MUST survive the parameter-value-aware scan when the recognizer accepts the dotted name",
  );
  assert_eq!(
    display, " tail",
    "FULL suffix (just the ` tail` past the REAL wrapper end-tag) reaches display — the in-parameter end-marker MUST NOT be treated as the wrapper close",
  );
}

#[test]
fn streaming_qwen3_coder_spaced_name_with_in_parameter_end_marker_does_not_drop_call() {
  // False-negative case with whitespace in name. The body
  // `<function=foo bar><parameter=p>v</parameter></function>` is also a
  // legitimate qwen3_coder call — the parser body's
  // `body.find('>')` accepts ANY bytes before `>`, so a space-in-name
  // (`foo bar`) is parser-accepted. A stricter `[A-Za-z0-9_-]+` predicate
  // rejected the space → same false-negative as the dotted case when
  // paired with an in-parameter end-marker.
  //
  // The shared recognizer `qwen_function_open_at` accepts spaces
  // (and anything else that's neither `>` nor `<`).
  let (display, calls) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      "<tool_call><function=foo bar><parameter=p>has </tool_call> in value</parameter></function></tool_call> tail",
    ],
  );
  assert_eq!(
    calls.len(),
    1,
    "space-bearing qwen3_coder name MUST be accepted by the shared recognizer",
  );
  assert_eq!(calls[0].name(), "foo bar");
  assert_eq!(
    *calls[0].arguments(),
    serde_json::json!({ "p": "has </tool_call> in value" }),
    "in-parameter `</tool_call>` literal MUST survive the parameter-value-aware scan when the recognizer accepts the spaced name",
  );
  assert_eq!(
    display, " tail",
    "FULL suffix bytes reach display — the in-parameter end-marker MUST NOT be treated as the wrapper close when the recognizer accepts the spaced name",
  );
}

// ===========================================================
// Terminal-on-first-marker for qwen3_coder
// -----------------------------------------------------------
// The shared recognizer `find_first_qwen_function_open` (used by
// predicate and parser body) must not scan EVERY byte position for a valid
// `<function=NAME>` open. A malformed outer opener
// (`<function=a<function=real>...`) correctly fails
// `qwen_function_open_at` at the outer marker; a scan that
// CONTINUED past it would find the nested `<function=real>` as
// a valid opener — so the parser body would extract `"real"` as a
// tool call from structurally-malformed bytes, defeating the
// section-level structural rejection.
//
// `find_first_qwen_function_open` is therefore TERMINAL on the
// first `<function=` literal: that literal IS the section's
// structural anchor; if it is malformed, the section as a
// whole is malformed (return None) — we don't pretend a later
// nested opener is a new valid section. The two tests below
// pin the two malformed-anchor shapes (name contains `<`,
// empty name) and assert that the nested-but-valid marker is
// NOT extracted as a call.

#[test]
fn streaming_qwen3_coder_malformed_outer_opener_with_nested_valid_does_not_extract_nested_as_call()
{
  // Motivating case (name contains `<`). The body
  // `<function=a<function=real><parameter=p>v</parameter></function>`
  // has an outer `<function=a<...>` opener whose name `a` is
  // followed by `<` — `qwen_function_open_at` correctly rejects
  // the outer marker (NAME must contain neither `>` nor `<`).
  //
  // A `find_first_qwen_function_open` that continued scanning
  // past the rejected outer marker would hit the nested
  // `<function=real>` at byte 12 and accept it as a valid
  // opener. The predicate would then prove context true, the
  // parameter-value-aware scan would skip `<parameter=p>...</parameter>`,
  // the wrapper close would land at the real `</tool_call>`, the
  // parser body's separate `(0..bytes.len()).find_map(...)`
  // scan would ALSO find the nested opener — and emit `"real"`
  // as a tool call from structurally-invalid bytes.
  //
  // The recognizer is terminal-on-first-marker: first `<function=` at byte
  // 0 is malformed → recognizer returns None → predicate false
  // → PlainEnd → bounded body is everything before
  // `</tool_call>`; the bounded prefix's first marker is also
  // malformed → empty calls. Same-chunk suffix ` tail` reaches
  // display.
  let (display, calls) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      "<tool_call><function=a<function=real><parameter=p>v</parameter></function></tool_call> tail",
    ],
  );
  assert_eq!(
    calls.len(),
    0,
    "malformed outer `<function=a<...>` opener MUST NOT be bypassed by scanning past to a nested `<function=real>` opener (the first `<function=` literal IS the section's structural anchor — if it is malformed the section as a whole is malformed)",
  );
  assert_eq!(
    display, " tail",
    "FULL same-chunk suffix bytes reach display — terminal-on-first-marker MUST reject the section without emitting a nested-marker call",
  );
}

#[test]
fn streaming_qwen3_coder_empty_name_opener_with_nested_valid_does_not_extract_nested() {
  // Motivating case (empty name). The body
  // `<function=><function=real><parameter=p>v</parameter></function>`
  // has an outer `<function=>` opener whose name is empty —
  // `qwen_function_open_at` correctly rejects (NAME must be
  // non-empty).
  //
  // A scan that continued past the rejected outer marker would
  // hit `<function=real>` at byte 11, accept it → predicate
  // true → parser body extracts `"real"` as a tool call.
  //
  // The recognizer is terminal-on-first-marker: first `<function=` at byte
  // 0 is malformed (empty name) → recognizer returns None →
  // predicate false → PlainEnd → empty calls. Same-chunk
  // suffix ` tail` reaches display.
  let (display, calls) = run_with_parser(
    Box::new(Qwen3Coder),
    &[
      "<tool_call><function=><function=real><parameter=p>v</parameter></function></tool_call> tail",
    ],
  );
  assert_eq!(
    calls.len(),
    0,
    "malformed outer `<function=>` (empty name) opener MUST NOT be bypassed by scanning past to a nested `<function=real>` opener",
  );
  assert_eq!(
    display, " tail",
    "FULL same-chunk suffix bytes reach display — terminal-on-first-marker MUST reject the section without emitting a nested-marker call",
  );
}

/// Audit-locking test: for each parser whose `bound_section`
/// uses a STRUCTURAL context predicate (not just the dispatcher's
/// own leading-byte classifier), the predicate's acceptance grammar
/// MUST match the parser's `try_parse_one_call` body recognizer
/// EXACTLY. The test exercises a curated set of "should-prove"
/// payloads (grammar-edge accepts) and "should-NOT-prove" payloads
/// (grammar-edge rejects), then for each:
///   * "should-prove": the parser's `try_parse_one_call` MUST extract
///     a real call when the body is a complete tagged section (the
///     predicate's acceptance corresponds to a parseable opener).
///   * "should-NOT-prove": the parser's `try_parse_one_call` MUST
///     surface ZERO calls when the body contains only the
///     predicate-rejected opener-shape garbage (the predicate's
///     rejection corresponds to no parseable call).
///
/// Drift surfaces immediately: if the predicate accepts a payload the
/// parser rejects (or vice-versa), the corresponding assertion fires.
/// New parsers that add a structural predicate MUST extend this table
/// rather than chasing the same drift class round-after-round.
#[test]
fn try_parse_one_call_context_predicate_matches_recognizer_per_parser() {
  struct Row {
    label: &'static str,
    parser: Box<dyn ToolParser>,
    // A buffer wrapping a complete tagged section whose body is the
    // shape under test. The grammar edge IS the contents of the body
    // (digit-leading name, whitespace, etc.).
    buffer: &'static str,
    // When true, the predicate MUST accept the body shape AND the
    // parser MUST extract at least one call from the buffer. When
    // false, the predicate MUST reject the body shape AND the parser
    // MUST surface zero calls from the buffer.
    should_extract: bool,
  }
  let rows: Vec<Row> = vec![
    // --- pythonic grammar edges --------------------------------------
    // Accept: digit-leading name (`\w+` allows leading digits).
    Row {
      label: "pythonic accept: digit-leading name `[1tool(x=1)]`",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>[1tool(x=1)]<|tool_call_end|>",
      should_extract: true,
    },
    // Accept: underscore-leading name.
    Row {
      label: "pythonic accept: underscore-leading name `[_tool(x=1)]`",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>[_tool(x=1)]<|tool_call_end|>",
      should_extract: true,
    },
    // Reject: whitespace between `[` and name.
    Row {
      label: "pythonic reject: whitespace before name `[ tool(x=1)]`",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>[ tool(x=1)]<|tool_call_end|>",
      should_extract: false,
    },
    // Reject: whitespace between name and `(`.
    Row {
      label: "pythonic reject: whitespace before `(` `[tool (x=1)]`",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>[tool (x=1)]<|tool_call_end|>",
      should_extract: false,
    },
    // Reject: empty name `[(`.
    Row {
      label: "pythonic reject: empty name `[(x=1)]`",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>[(x=1)]<|tool_call_end|>",
      should_extract: false,
    },
    // Reject: stray `[` only.
    Row {
      label: "pythonic reject: stray `[` only `bad[`",
      parser: Box::new(Pythonic),
      buffer: "<|tool_call_start|>bad[<|tool_call_end|>",
      should_extract: false,
    },
    // --- function_gemma grammar edges --------------------------------
    // Accept: ASCII-alpha name with immediate `{`.
    Row {
      label: "function_gemma accept: `call:foo{k:v}`",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>call:foo{k:v}<end_function_call>",
      should_extract: true,
    },
    // Accept: digit-leading name (alnum run allowed; matches `gemma_call`).
    Row {
      label: "function_gemma accept: digit-leading `call:1foo{k:v}`",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>call:1foo{k:v}<end_function_call>",
      should_extract: true,
    },
    // Accept: hyphen in name.
    Row {
      label: "function_gemma accept: hyphen `call:foo-bar{k:v}`",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>call:foo-bar{k:v}<end_function_call>",
      should_extract: true,
    },
    // Reject: whitespace between name and `{`.
    Row {
      label: "function_gemma reject: whitespace before `{` `call:foo {k:v}`",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>call:foo {k:v}<end_function_call>",
      should_extract: false,
    },
    // Reject: empty name.
    Row {
      label: "function_gemma reject: empty name `call:{k:v}`",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>call:{k:v}<end_function_call>",
      should_extract: false,
    },
    // Reject: stray `call:` only.
    Row {
      label: "function_gemma reject: stray `call:` only `bad call:`",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>bad call:<end_function_call>",
      should_extract: false,
    },
    // --- json_tools (predicate = leading-`{` shape) ------------------
    // Accept: leading `{`.
    Row {
      label: "json_tools accept: leading `{`",
      parser: Box::new(JsonTools),
      buffer: r#"<tool_call>{"name":"x","arguments":{}}</tool_call>"#,
      should_extract: true,
    },
    // Reject: stray `{` after garbage.
    Row {
      label: "json_tools reject: stray `{` after garbage `bad{`",
      parser: Box::new(JsonTools),
      buffer: r#"<tool_call>bad{"name":"x"}</tool_call>"#,
      should_extract: false,
    },
    // --- qwen3_coder (predicate = `<function=NAME>` shape) -----------
    // Accept: complete `<function=foo>` open-tag.
    Row {
      label: "qwen3_coder accept: `<function=foo></function>`",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call><function=foo></function></tool_call>",
      should_extract: true,
    },
    // Accept: dotted name `<function=foo.bar>` — parser body's
    // `body.find('>')` accepts ANY bytes before `>`, so a dot is fine.
    // A stricter `[A-Za-z0-9_-]+` predicate REJECTED dots; the shared
    // recognizer accepts.
    Row {
      label: "qwen3_coder accept: dotted name `<function=foo.bar></function>`",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call><function=foo.bar></function></tool_call>",
      should_extract: true,
    },
    // Accept: spaced name `<function=foo bar>` — same logic; the
    // parser body accepts whitespace inside the name. The stricter predicate
    // rejected; the shared recognizer accepts.
    Row {
      label: "qwen3_coder accept: spaced name `<function=foo bar></function>`",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call><function=foo bar></function></tool_call>",
      should_extract: true,
    },
    // Accept: special-char name `<function=ns:method/v2>` — colons,
    // slashes, digits are all parser-accepted (none of them is `>` or
    // `<`). The stricter predicate rejected; the shared recognizer accepts.
    Row {
      label: "qwen3_coder accept: special-char name `<function=ns:method/v2></function>`",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call><function=ns:method/v2></function></tool_call>",
      should_extract: true,
    },
    // Reject: `<function=` without name+`>` close.
    Row {
      label: "qwen3_coder reject: `<function=` without `NAME>` close",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call>bad<function= </tool_call>",
      should_extract: false,
    },
    // Reject: empty name `<function=>` — the shared recognizer
    // requires a non-empty name run. Matches the parser body's tightened
    // recognizer-based opener search (which also rejects the empty
    // name).
    Row {
      label: "qwen3_coder reject: empty name `<function=></function>`",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call><function=></function></tool_call>",
      should_extract: false,
    },
    // Reject: name contains `<` — would break the surrounding XML
    // framing because `<` opens a new tag. Recognizer requires NAME to
    // contain neither `>` nor `<`.
    Row {
      label: "qwen3_coder reject: name with `<` `<function=a<b></function>`",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call><function=a<b></function></tool_call>",
      should_extract: false,
    },
    // Reject: malformed outer opener `<function=a<...>` with a
    // nested valid opener `<function=real>`. The first `<function=`
    // literal IS the section's structural anchor; if its NAME contains
    // `<` it is malformed and the section as a whole is malformed —
    // the parser MUST NOT scan past the rejected outer marker and
    // emit `"real"` as a tool call from the nested marker.
    // `find_first_qwen_function_open` is terminal on the first
    // `<function=` literal.
    Row {
      label: "qwen3_coder reject: malformed outer + nested valid `<function=a<function=real></parameter></function>`",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call><function=a<function=real><parameter=p>v</parameter></function></tool_call>",
      should_extract: false,
    },
    // Reject: malformed outer opener `<function=>` (empty name)
    // with a nested valid opener `<function=real>`. Same structural
    // logic: empty-name first marker is malformed → section malformed
    // → terminal-on-first-marker rejects → no call from the nested
    // `<function=real>`.
    Row {
      label: "qwen3_coder reject: empty-name outer + nested valid `<function=><function=real></parameter></function>`",
      parser: Box::new(Qwen3Coder),
      buffer: "<tool_call><function=><function=real><parameter=p>v</parameter></function></tool_call>",
      should_extract: false,
    },
    // --- function_gemma baseline regressions -------------------------
    // Reject (orphan escape preserved): `call:` without `NAME{`.
    Row {
      label: "function_gemma reject: stray `call:` + orphan `<escape>`",
      parser: Box::new(FunctionGemma),
      buffer: "<start_function_call>bad call:<escape><end_function_call>",
      should_extract: false,
    },
  ];

  for row in &rows {
    let result = row
      .parser
      .try_parse_one_call(row.buffer, None)
      .unwrap_or_else(|e| panic!("{}: try_parse_one_call errored: {e}", row.label));
    let (calls, _end_pos) = result.unwrap_or_else(|| {
        panic!(
          "{}: confirmed-bounded section expected (the wrapper end-tag is in the buffer), got Ok(None) — predicate/recognizer drift hid the wrapper close",
          row.label,
        )
      });
    if row.should_extract {
      assert!(
        !calls.is_empty(),
        "{}: the predicate must ACCEPT this body shape (the parser's recognizer accepts it); got zero calls — predicate is STRICTER than the parser body (false-negative drift)",
        row.label,
      );
    } else {
      assert!(
        calls.is_empty(),
        "{}: the predicate must REJECT this body shape (the parser's recognizer rejects it); got {} call(s) — predicate is LOOSER than the parser body (false-positive drift)",
        row.label,
        calls.len(),
      );
    }
  }
}

// ========================================================================
// Batch-parse + pure-helper coverage.
//
// The streaming tests above exercise `try_parse_one_call` and
// `ToolCallProcessor` thoroughly. The tests below close the remaining gaps
// in the PURE-LOGIC surface: each parser's `parse()` batch path, the
// literal/value coercion helpers (`literal_eval`, `deserialize`,
// `convert_param_value`, `schema_types`, `convert_with_types`), the
// extraction helpers (`find_pythonic_call`, `parse_kw_args`, `find_all`,
// `find_kv_pairs`, `gemma_call`, `gemma4_calls`, `gemma4_args_to_json`,
// `glm_parse_json`, `glm_parse_plain`, `extract_name`, `balanced_brace_end`,
// `utf8_char_width`, `pythonic_call_close`, `xml_value_aware_end_tag_scan`),
// the selection functions (`parser_by_name`, `infer_tool_parser`) and the
// default `parse` loop (`default_parse_via_try_parse_one_call`). Oracles are
// closed-form: every expected value is hand-written from the documented
// grammar, never produced by calling the function under test.
// ========================================================================

// --- literal_eval: every coercion branch --------------------------------

#[test]
fn literal_eval_json_first() {
  // Bare JSON values are parsed by serde first (line ~545).
  assert_eq!(literal_eval("42"), serde_json::json!(42));
  assert_eq!(literal_eval("3.5"), serde_json::json!(3.5));
  assert_eq!(literal_eval(r#"{"a":1}"#), serde_json::json!({"a": 1}));
  assert_eq!(literal_eval("[1,2]"), serde_json::json!([1, 2]));
  // Surrounding whitespace is trimmed before parsing.
  assert_eq!(literal_eval("  7  "), serde_json::json!(7));
}

#[test]
fn literal_eval_python_bool_none() {
  // Python literals that are not JSON: True / False / None.
  assert_eq!(literal_eval("True"), Value::Bool(true));
  assert_eq!(literal_eval("False"), Value::Bool(false));
  assert_eq!(literal_eval("None"), Value::Null);
  // Lowercase JSON spellings also map (covered by the explicit match arm).
  assert_eq!(literal_eval("true"), Value::Bool(true));
  assert_eq!(literal_eval("null"), Value::Null);
}

#[test]
fn literal_eval_single_and_double_quoted_strings() {
  // Single-quoted: serde rejects, the quote-strip arm (line ~554) unwraps.
  assert_eq!(literal_eval("'hello'"), Value::String("hello".to_owned()));
  // A bare double-quoted string IS valid JSON, so it round-trips identically.
  assert_eq!(literal_eval(r#""hi""#), Value::String("hi".to_owned()));
  // Empty single-quoted string (len == 2 boundary).
  assert_eq!(literal_eval("''"), Value::String(String::new()));
}

#[test]
fn literal_eval_bare_int_and_float_non_json() {
  // `+5` is NOT valid JSON (no leading `+`) but parses as i64 (line ~559).
  assert_eq!(literal_eval("+5"), serde_json::json!(5));
  // `+1.5` fails JSON and i64, parses as f64 (line ~562-565).
  assert_eq!(literal_eval("+1.5"), serde_json::json!(1.5));
  // A leading-zero-padded int `007` is rejected by serde JSON but parses i64.
  assert_eq!(literal_eval("007"), serde_json::json!(7));
}

#[test]
fn literal_eval_python_container_quote_swap() {
  // Python list with single quotes: serde fails, the bracket/quote-swap arm
  // (line ~568-581) replaces `'`->`"` and parses.
  assert_eq!(literal_eval("['a', 'b']"), serde_json::json!(["a", "b"]));
  // Python tuple `(...)` is rewritten to a JSON array `[...]`.
  assert_eq!(literal_eval("(1, 2)"), serde_json::json!([1, 2]));
  // Python dict with single-quoted keys + `True`/`None` tokens swapped.
  assert_eq!(
    literal_eval("{'ok': True, 'x': None}"),
    serde_json::json!({"ok": true, "x": null})
  );
}

#[test]
fn literal_eval_unparseable_falls_back_to_string() {
  // Fails JSON, bool/none, quote-strip, int, float, and the bracket swap
  // (no enclosing bracket): the final fallback returns the trimmed string
  // (line ~583).
  assert_eq!(literal_eval("foo bar"), Value::String("foo bar".to_owned()));
  // A bracketed value whose swap still fails JSON also lands on the string
  // fallback (e.g. an unbalanced bracket).
  assert_eq!(literal_eval("[oops"), Value::String("[oops".to_owned()));
}

#[test]
fn deserialize_json_then_literal() {
  // `deserialize` is JSON-first, then `literal_eval`.
  assert_eq!(deserialize(r#"{"a":1}"#), serde_json::json!({"a": 1}));
  // Single-quoted Python string: JSON fails, literal_eval unwraps.
  assert_eq!(deserialize("'x'"), Value::String("x".to_owned()));
  // Unparseable: literal_eval's string fallback.
  assert_eq!(deserialize("plain"), Value::String("plain".to_owned()));
}

// --- convert_param_value: type-directed coercion (qwen3_coder) ----------

/// Build a minimal `properties` map for one parameter with the given JSON
/// `type` string, wrapped as a `tools` array so `tool_properties` resolves it.
fn tools_with_param_type(func: &str, param: &str, ty: &str) -> Value {
  serde_json::json!([{
    "function": {
      "name": func,
      "parameters": { "properties": { param: { "type": ty } } }
    }
  }])
}

#[test]
fn convert_param_value_no_props_is_string() {
  // No props at all → raw string (line ~1370-1372).
  assert_eq!(
    convert_param_value("5", "n", None),
    Value::String("5".to_owned())
  );
}

#[test]
fn convert_param_value_null_literal() {
  // Case-insensitive `null` → Value::Null regardless of schema (line ~1367).
  assert_eq!(convert_param_value("NULL", "x", None), Value::Null);
  assert_eq!(convert_param_value("null", "x", None), Value::Null);
}

#[test]
fn convert_param_value_missing_schema_for_name() {
  // props present but the param name is absent → raw string (line ~1373-1375).
  let props = serde_json::json!({ "other": { "type": "string" } });
  let map = props.as_object().unwrap();
  assert_eq!(
    convert_param_value("v", "missing", Some(map)),
    Value::String("v".to_owned())
  );
}

#[test]
fn convert_param_value_typed_branches() {
  // Helper: pull the single param's schema map out of a tools array.
  fn props_for(tools: &Value, func: &str) -> serde_json::Map<String, Value> {
    tools
      .as_array()
      .unwrap()
      .iter()
      .find(|t| t["function"]["name"] == func)
      .unwrap()["function"]["parameters"]["properties"]
      .as_object()
      .unwrap()
      .clone()
  }

  // string-family type → raw string (line ~1385).
  let t = tools_with_param_type("f", "p", "string");
  let p = props_for(&t, "f");
  assert_eq!(
    convert_param_value("hello", "p", Some(&p)),
    Value::String("hello".to_owned())
  );

  // int-family → parse i64 (line ~1387-1394).
  let t = tools_with_param_type("f", "p", "int");
  let p = props_for(&t, "f");
  assert_eq!(
    convert_param_value("12", "p", Some(&p)),
    serde_json::json!(12)
  );
  // int-family parse failure → string fallback.
  assert_eq!(
    convert_param_value("nope", "p", Some(&p)),
    Value::String("nope".to_owned())
  );

  // number/float → ALWAYS a JSON float, never promoted to int (line ~1404-1409).
  let t = tools_with_param_type("f", "p", "number");
  let p = props_for(&t, "f");
  assert_eq!(
    convert_param_value("2.5", "p", Some(&p)),
    serde_json::json!(2.5)
  );
  // Non-finite has no JSON representation → string fallback (line ~1408).
  assert_eq!(
    convert_param_value("NaN", "p", Some(&p)),
    Value::String("NaN".to_owned())
  );

  // boolean → case-insensitive true/anything-else (line ~1410-1411).
  let t = tools_with_param_type("f", "p", "boolean");
  let p = props_for(&t, "f");
  assert_eq!(
    convert_param_value("TRUE", "p", Some(&p)),
    Value::Bool(true)
  );
  assert_eq!(convert_param_value("no", "p", Some(&p)), Value::Bool(false));

  // object/array/dict/list → JSON parse, else literal_eval (line ~1412-1414).
  let t = tools_with_param_type("f", "p", "object");
  let p = props_for(&t, "f");
  assert_eq!(
    convert_param_value(r#"{"k":1}"#, "p", Some(&p)),
    serde_json::json!({"k": 1})
  );
  // Unparseable object body falls back to literal_eval (→ string here).
  assert_eq!(
    convert_param_value("plain", "p", Some(&p)),
    Value::String("plain".to_owned())
  );

  // unknown type → literal_eval (line ~1416).
  let t = tools_with_param_type("f", "p", "weird");
  let p = props_for(&t, "f");
  assert_eq!(
    convert_param_value("True", "p", Some(&p)),
    Value::Bool(true)
  );
}

// --- schema_types + convert_with_types (minimax_m2) ---------------------

#[test]
fn schema_types_variants() {
  // Single string type.
  assert_eq!(
    schema_types(&serde_json::json!({"type": "integer"})),
    vec!["integer".to_owned()]
  );
  // Array of types (sorted via BTreeSet).
  assert_eq!(
    schema_types(&serde_json::json!({"type": ["string", "null"]})),
    vec!["null".to_owned(), "string".to_owned()]
  );
  // enum values map to their JSON-shape type names.
  assert_eq!(
    schema_types(&serde_json::json!({"enum": [1, "a", true, null]})),
    vec![
      "boolean".to_owned(),
      "integer".to_owned(),
      "null".to_owned(),
      "string".to_owned()
    ]
  );
  // anyOf/oneOf/allOf recursion.
  assert_eq!(
    schema_types(&serde_json::json!({
      "anyOf": [{"type": "integer"}, {"type": "boolean"}]
    })),
    vec!["boolean".to_owned(), "integer".to_owned()]
  );
  // Empty / non-object schema → default ["string"].
  assert_eq!(
    schema_types(&serde_json::json!({})),
    vec!["string".to_owned()]
  );
  assert_eq!(
    schema_types(&serde_json::json!(123)),
    vec!["string".to_owned()]
  );
}

#[test]
fn convert_with_types_branches() {
  // null literal short-circuits regardless of types (line ~2100).
  assert_eq!(
    convert_with_types("null", &["integer".to_owned()]),
    Value::Null
  );
  // type-set contains null OR value is none/nil (line ~2105).
  assert_eq!(
    convert_with_types("nil", &["string".to_owned()]),
    Value::Null
  );
  assert_eq!(convert_with_types("x", &["null".to_owned()]), Value::Null);
  // integer parse.
  assert_eq!(
    convert_with_types("7", &["integer".to_owned()]),
    serde_json::json!(7)
  );
  // number → JSON float, never int promotion.
  assert_eq!(
    convert_with_types("2.0", &["number".to_owned()]),
    serde_json::json!(2.0)
  );
  // boolean true/false synonyms.
  assert_eq!(
    convert_with_types("yes", &["boolean".to_owned()]),
    Value::Bool(true)
  );
  assert_eq!(
    convert_with_types("off", &["boolean".to_owned()]),
    Value::Bool(false)
  );
  // object/array → JSON parse.
  assert_eq!(
    convert_with_types("[1,2]", &["array".to_owned()]),
    serde_json::json!([1, 2])
  );
  // The union is walked in a FIXED priority order
  // (integer, int, number, float, boolean, bool, object, array, string, ...),
  // so `integer` is tried before `string`: a parseable int returns a number
  // even when `string` is also in the union.
  assert_eq!(
    convert_with_types("42", &["string".to_owned(), "integer".to_owned()]),
    serde_json::json!(42),
    "integer is visited before string in the priority order"
  );
  // A non-numeric value with the same union falls through `integer` (parse
  // fails, no early return) and lands on the `string` branch.
  assert_eq!(
    convert_with_types("xyz", &["string".to_owned(), "integer".to_owned()]),
    Value::String("xyz".to_owned()),
    "integer parse fails → falls through to the string branch"
  );
  // No recognised type at all → final serde-or-string fallback (line ~2154).
  assert_eq!(
    convert_with_types("bare", &["unknown_type".to_owned()]),
    Value::String("bare".to_owned())
  );
  assert_eq!(
    convert_with_types("99", &["unknown_type".to_owned()]),
    serde_json::json!(99),
    "no recognised type → final serde fallback parses the JSON number"
  );
}

#[test]
fn extract_name_quote_stripping() {
  assert_eq!(extract_name(r#""foo""#), "foo");
  assert_eq!(extract_name("'bar'"), "bar");
  assert_eq!(extract_name("  baz  "), "baz", "trimmed, no quotes");
  // A single stray quote is not a matched pair → returned verbatim (trimmed).
  assert_eq!(extract_name(r#""x"#), r#""x"#);
}

// --- find_pythonic_call + parse_kw_args ---------------------------------

#[test]
fn find_pythonic_call_extracts_first() {
  // First `[name(...)]`; args region is everything up to `)]`.
  assert_eq!(
    find_pythonic_call("[echo(a=1, b=2)]"),
    Some(("echo".to_owned(), "a=1, b=2".to_owned()))
  );
  // Leading text before the call is skipped.
  assert_eq!(
    find_pythonic_call("noise [f()] tail"),
    Some(("f".to_owned(), String::new()))
  );
  // No call form at all.
  assert_eq!(find_pythonic_call("plain text"), None);
  // `[` without an immediate name+`(` is not a call start.
  assert_eq!(find_pythonic_call("[ not_a_call ]"), None);
}

#[test]
fn parse_kw_args_quoted_unquoted_and_unterminated() {
  // Mixed quoted + unquoted values.
  assert_eq!(
    parse_kw_args(r#"a="x", b=2, c=hello"#).unwrap(),
    vec![
      ("a".to_owned(), "x".to_owned()),
      ("b".to_owned(), "2".to_owned()),
      ("c".to_owned(), "hello".to_owned()),
    ]
  );
  // Empty input → no pairs.
  assert!(parse_kw_args("").unwrap().is_empty());
  // No `=` at all → loop breaks immediately, no pairs (line ~925-926).
  assert!(parse_kw_args("garbage").unwrap().is_empty());
  // Unterminated quote keeps the remainder as the value (line ~945-946).
  assert_eq!(
    parse_kw_args(r#"a="unterminated"#).unwrap(),
    vec![("a".to_owned(), "unterminated".to_owned())]
  );
  // Whitespace before a non-ASCII value must not panic on a UTF-8 boundary.
  assert_eq!(
    parse_kw_args(r#"city=  "é""#).unwrap(),
    vec![("city".to_owned(), "é".to_owned())]
  );
}

// --- find_all + find_kv_pairs -------------------------------------------

#[test]
fn find_all_basic_and_unclosed() {
  assert_eq!(
    find_all("<a>1</a><a>2</a>", "<a>", "</a>"),
    vec!["1".to_owned(), "2".to_owned()]
  );
  // An open with no matching close terminates the scan (line ~1431).
  assert_eq!(
    find_all("<a>1</a><a>dangling", "<a>", "</a>"),
    vec!["1".to_owned()]
  );
  // No open at all → empty.
  assert!(find_all("nothing here", "<a>", "</a>").is_empty());
}

#[test]
fn find_kv_pairs_basic_and_truncated() {
  assert_eq!(
    find_kv_pairs(
      "<k>a</k><v>1</v><k>b</k><v>2</v>",
      "<k>",
      "</k>",
      "<v>",
      "</v>"
    ),
    vec![
      ("a".to_owned(), "1".to_owned()),
      ("b".to_owned(), "2".to_owned())
    ]
  );
  // A key with no value tag stops the scan (the value-open `find` fails).
  assert_eq!(
    find_kv_pairs("<k>a</k>no value", "<k>", "</k>", "<v>", "</v>"),
    Vec::<(String, String)>::new()
  );
}

// --- glm_parse_json + glm_parse_plain (glm47 fallbacks) -----------------

#[test]
fn glm_parse_json_shapes() {
  // {name, arguments}
  assert_eq!(
    glm_parse_json(r#"{"name":"f","arguments":{"x":1}}"#, None),
    Some(ToolCall::new_nameless_id("f", serde_json::json!({"x": 1})))
  );
  // {function, arguments} alias for the name key.
  assert_eq!(
    glm_parse_json(r#"{"function":"g","arguments":{}}"#, None),
    Some(ToolCall::new_nameless_id("g", serde_json::json!({})))
  );
  // {tool: {name, arguments}} nested shape.
  assert_eq!(
    glm_parse_json(r#"{"tool":{"name":"h","arguments":{"a":2}}}"#, None),
    Some(ToolCall::new_nameless_id("h", serde_json::json!({"a": 2})))
  );
  // Top-level array → first object is taken.
  assert_eq!(
    glm_parse_json(r#"[{"name":"arr","arguments":{}}]"#, None),
    Some(ToolCall::new_nameless_id("arr", serde_json::json!({})))
  );
  // No name → None.
  assert_eq!(glm_parse_json(r#"{"arguments":{}}"#, None), None);
  // A `tool` object with no `arguments` key → arguments defaults to the empty
  // object (the `None =>` arm). NB the top-level `{"name",...}` shape requires
  // BOTH `name` and `arguments`, so `{"name":"k"}` alone parses to None; the
  // `tool`-wrapped form is what reaches the default-empty-arguments arm.
  assert_eq!(
    glm_parse_json(r#"{"tool":{"name":"k"}}"#, None),
    Some(ToolCall::new_nameless_id("k", serde_json::json!({})))
  );
  // arguments as a JSON-encoded string → deserialized (line ~1642-1643).
  assert_eq!(
    glm_parse_json(r#"{"name":"k","arguments":"{\"z\":3}"}"#, None),
    Some(ToolCall::new_nameless_id("k", serde_json::json!({"z": 3})))
  );
  // arguments present but not an object (and not a string) → None (line ~1652).
  assert_eq!(glm_parse_json(r#"{"name":"k","arguments":5}"#, None), None);
  // Not JSON at all → None.
  assert_eq!(glm_parse_json("not json", None), None);
}

#[test]
fn glm_parse_plain_shapes() {
  // Empty / whitespace → None (line ~1658).
  assert_eq!(glm_parse_plain("   ", None), None);
  // First line is the name, remainder is a JSON object body (line ~1661-1670).
  assert_eq!(
    glm_parse_plain("myfunc\n{\"a\":1}", None),
    Some(ToolCall::new_nameless_id(
      "myfunc",
      serde_json::json!({"a": 1})
    ))
  );
  // Single token, no args → empty-object call (line ~1679-1681).
  assert_eq!(
    glm_parse_plain("solo", None),
    Some(ToolCall::new_nameless_id("solo", serde_json::json!({})))
  );
  // `name <json-object>` after a space split → object args (line ~1682-1685).
  assert_eq!(
    glm_parse_plain(r#"f {"k":2}"#, None),
    Some(ToolCall::new_nameless_id("f", serde_json::json!({"k": 2})))
  );
  // `name key=value key=value` → parsed pairs (line ~1687-1709).
  assert_eq!(
    glm_parse_plain("f a=1 b=2", None),
    Some(ToolCall::new_nameless_id(
      "f",
      serde_json::json!({"a": 1, "b": 2})
    ))
  );
  // A malformed token (no `=`) makes the kv path bail, so the whole rest is
  // surfaced as `{"raw": ...}` (line ~1701-1714).
  assert_eq!(
    glm_parse_plain("f a=1 bad", None),
    Some(ToolCall::new_nameless_id(
      "f",
      serde_json::json!({"raw": "a=1 bad"})
    ))
  );
}

// --- gemma_call + gemma4 helpers ----------------------------------------

#[test]
fn gemma_call_first_call_and_rejects() {
  assert_eq!(
    gemma_call("call:foo{a:1}", false),
    Some(("foo".to_owned(), "a:1".to_owned()))
  );
  // `call:` without an immediate `{` after the name is not a call start.
  assert_eq!(gemma_call("call:foo bar", false), None);
  // No `call:` marker at all.
  assert_eq!(gemma_call("nothing", false), None);
}

#[test]
fn gemma4_calls_walks_blocks() {
  // Two back-to-back calls; braces balanced.
  assert_eq!(
    gemma4_calls("call:a{x:1}call:b{y:2}"),
    vec![
      ("a".to_owned(), "{x:1}".to_owned()),
      ("b".to_owned(), "{y:2}".to_owned())
    ]
  );
  // A `call:` with no `{` body is skipped, the next valid one is found.
  assert_eq!(
    gemma4_calls("call:skip then call:c{z:3}"),
    vec![("c".to_owned(), "{z:3}".to_owned())]
  );
  // No call at all.
  assert!(gemma4_calls("plain").is_empty());
}

#[test]
fn gemma4_args_to_json_quotes_keys_and_restores_strings() {
  // Bare keys get quoted; `<|"|>STR<|"|>` becomes a JSON string. A `}` inside
  // the string region does not close the object. The bare-key rewrite
  // requires the key to IMMEDIATELY follow `{` or `,` (no whitespace), so a
  // second key sits directly after the comma.
  let json = gemma4_args_to_json(r#"{k: <|"|>v}x<|"|>,n: 2}"#);
  let parsed: Value = serde_json::from_str(&json).expect("valid JSON output");
  assert_eq!(parsed, serde_json::json!({"k": "v}x", "n": 2}));
  // A single-key body with a `<|"|>` string carrying a `}` literal also
  // round-trips (the string region is opaque to brace matching).
  let json2 = gemma4_args_to_json(r#"{s: <|"|>a}b<|"|>}"#);
  let parsed2: Value = serde_json::from_str(&json2).expect("valid JSON output");
  assert_eq!(parsed2, serde_json::json!({"s": "a}b"}));
}

#[test]
fn balanced_brace_end_and_utf8_width() {
  // Smallest object.
  assert_eq!(balanced_brace_end("{}"), Some(1));
  // Nested + a `}` inside a `<|"|>` literal must not close early.
  assert_eq!(balanced_brace_end(r#"{a:{b:1}}"#), Some(8));
  // `{s:<|"|>}<|"|>}` — the `<|"|>}<|"|>` region (open at byte 3, close
  // ending at byte 13) is skipped whole, so the `}` at byte 8 does NOT close
  // the object; the real closing `}` is at byte 14.
  assert_eq!(
    balanced_brace_end(r#"{s:<|"|>}<|"|>}"#),
    Some(14),
    "the `}}` inside the <|\"|> literal is skipped; the real close is last"
  );
  // Not starting with `{` → None.
  assert_eq!(balanced_brace_end("x{}"), None);
  // Never closes → None.
  assert_eq!(balanced_brace_end("{a:1"), None);

  // utf8_char_width: ASCII=1, 2/3/4-byte leads, stray continuation byte → 1.
  assert_eq!(utf8_char_width(b'a'), 1);
  assert_eq!(utf8_char_width(0xC3), 2);
  assert_eq!(utf8_char_width(0xE2), 3);
  assert_eq!(utf8_char_width(0xF0), 4);
  assert_eq!(
    utf8_char_width(0x80),
    1,
    "stray continuation byte advances 1"
  );
}

// --- pythonic_call_close: the no-`)` re-scan + depth-underflow branches --

#[test]
fn pythonic_call_close_variants() {
  // Simple close.
  let s = "[f(a=1)]";
  assert_eq!(pythonic_call_close(s), Some(s.len()));
  // No `[` → None.
  assert_eq!(pythonic_call_close("plain"), None);
  // A `)]` literal inside a single-quoted string does NOT close the call.
  let q = "[echo(s=')]')]";
  assert_eq!(pythonic_call_close(q), Some(q.len()));
  // First bracket pair closes WITHOUT a preceding `)`, so the scanner skips
  // ahead to the next `[name(...)]` (line ~3568-3589).
  let rescan = "[bad][g(x=1)]";
  assert_eq!(pythonic_call_close(rescan), Some(rescan.len()));
  // A non-call `[...]` with no following `[` → the re-scan runs off the end
  // and returns None (line ~3581-3583).
  assert_eq!(pythonic_call_close("[just text]"), None);
}

// --- qwen_function_open_at: boundary / malformed-name rejections ---------

#[test]
fn qwen_function_open_at_boundaries() {
  // Too short for the `<function=` needle (line ~1305-1307).
  assert_eq!(qwen_function_open_at("<fun", 0), None);
  // Wrong literal at the offset (line ~1308-1310).
  assert_eq!(qwen_function_open_at("xxfunction=f>", 0), None);
  // Valid open: returns (name_start, after_close_bracket).
  // `<function=` is 10 bytes, name `foo` 3 bytes, `>` at 13, after = 14.
  assert_eq!(qwen_function_open_at("<function=foo>", 0), Some((10, 14)));
  // Empty name `<function=>` → None.
  assert_eq!(qwen_function_open_at("<function=>", 0), None);
  // Name runs to end of buffer with no `>` → None.
  assert_eq!(qwen_function_open_at("<function=foo", 0), None);
}

// --- xml_value_aware_end_tag_scan: value-skip + not-found branches -------

#[test]
fn xml_value_aware_end_tag_scan_branches() {
  // End-tag before any value region → found immediately.
  assert_eq!(
    xml_value_aware_end_tag_scan("END rest", "<v>", "</v>", "END"),
    Some(0)
  );
  // A value region opens before the end-tag candidate; its body (which holds
  // an in-value `END` literal) is skipped, and the real `END` after the
  // value close is returned.
  let s = "<v>inner END</v>END";
  let expected = s.find("</v>").unwrap() + "</v>".len();
  assert_eq!(
    xml_value_aware_end_tag_scan(s, "<v>", "</v>", "END"),
    Some(expected)
  );
  // A value opens but never closes → None (in-value body still streaming).
  assert_eq!(
    xml_value_aware_end_tag_scan("<v>unterminated END", "<v>", "</v>", "END"),
    None
  );
  // Neither value nor end-tag present → None.
  assert_eq!(
    xml_value_aware_end_tag_scan("plain bytes", "<v>", "</v>", "END"),
    None
  );
}

// --- balanced_json_{array,object}_prefix: extra-close + leading-skip -----
// NOTE: the `depth < 0` arms (object line ~3413-3416, array line ~3472-3475)
// are defensive and structurally UNREACHABLE through normal scanning — the
// scan returns at the FIRST `depth == 0`, so a trailing extra `]`/`}` is
// never processed and depth can never reach -1. These tests therefore cover
// the first-balanced-close return and the leading-byte-skip, not the
// dead underflow arms.

#[test]
fn balanced_json_array_prefix_extra_close_and_leading_skip() {
  // The first balanced `[...]` returns immediately; the trailing extra `]`
  // is left as suffix and never drives depth negative.
  assert_eq!(balanced_json_array_prefix("[]]"), Some((0, 2)));
  // Leading `]` before any `[` is ignored (scan starts at first `[`).
  assert_eq!(balanced_json_array_prefix("][1]"), Some((1, 4)));
}

#[test]
fn balanced_json_object_prefix_extra_close_is_suffix() {
  // Scan starts at `{`, closes at the first balanced `}`. A bare extra `}`
  // after the complete object is suffix (never processed).
  assert_eq!(balanced_json_object_prefix("{}}"), Some((0, 2)));
}

// --- batch parse() per parser: closed-form oracles ----------------------

#[test]
fn parse_json_tools_batch() {
  let out = JsonTools
    .parse(r#"{"name":"get_time","arguments":{"tz":"UTC"}}"#, None)
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "get_time",
      serde_json::json!({"tz": "UTC"})
    )]
  );
  // Missing name → typed tokenizer error (not a panic).
  let e = JsonTools.parse(r#"{"arguments":{}}"#, None).unwrap_err();
  assert!(
    matches!(e, Error::Tokenizer(_)),
    "missing-name is a Tokenizer error"
  );
  // Invalid JSON → typed tokenizer error.
  assert!(matches!(
    JsonTools.parse("not json", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
  // arguments omitted → Null arguments (line ~693).
  assert_eq!(
    JsonTools.parse(r#"{"name":"f"}"#, None).unwrap(),
    vec![ToolCall::new_nameless_id("f", Value::Null)]
  );
}

#[test]
fn parse_pythonic_batch() {
  let out = Pythonic
    .parse(r#"[echo(a="x", b=2, c=True)]"#, None)
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "echo",
      serde_json::json!({"a": "x", "b": 2, "c": true})
    )]
  );
  // No call form → "No function provided." tokenizer error.
  assert!(matches!(
    Pythonic.parse("nope", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
}

#[test]
fn parse_mistral_batch_and_errors() {
  let out = Mistral
    .parse(r#"get_weather[ARGS]{"city":"Tokyo"}"#, None)
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "get_weather",
      serde_json::json!({"city": "Tokyo"})
    )]
  );
  // No `[ARGS]` (line ~982-983).
  assert!(matches!(
    Mistral.parse("nothing", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
  // `[ARGS]` but no `{` after it (line ~987-988).
  assert!(matches!(
    Mistral.parse("f[ARGS] no-json", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
  // `[ARGS]{...}` but invalid JSON (line ~990).
  assert!(matches!(
    Mistral.parse("f[ARGS]{bad}", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
}

#[test]
fn parse_qwen3_coder_batch_with_types() {
  // Parameter values coerced via the provided schema (int + string).
  let tools = serde_json::json!([{
    "function": {
      "name": "f",
      "parameters": { "properties": {
        "n": { "type": "integer" },
        "s": { "type": "string" }
      } }
    }
  }]);
  let out = Qwen3Coder
    .parse(
      "<function=f><parameter=n>42</parameter><parameter=s>hi</parameter></function>",
      Some(&tools),
    )
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "f",
      serde_json::json!({"n": 42, "s": "hi"})
    )]
  );
  // A leading/trailing newline inside a parameter value is stripped
  // (line ~1139-1144).
  let out2 = Qwen3Coder
    .parse(
      "<function=g><parameter=p>\nval\n</parameter></function>",
      None,
    )
    .unwrap();
  assert_eq!(
    out2,
    vec![ToolCall::new_nameless_id(
      "g",
      serde_json::json!({"p": "val"})
    )]
  );
  // No `<function=` → "No function provided." (line ~1117-1118).
  assert!(matches!(
    Qwen3Coder.parse("plain", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
  // `<function=f>` with no `</function>` close → error (line ~1122-1124).
  assert!(matches!(
    Qwen3Coder
      .parse("<function=f><parameter=p>v</parameter>", None)
      .unwrap_err(),
    Error::Tokenizer(_)
  ));
}

#[test]
fn parse_glm47_batch_all_three_shapes() {
  // (1) XML-style with string-typed arg coerced to a string.
  let tools = serde_json::json!([{
    "function": {
      "name": "echo",
      "parameters": { "properties": { "s": { "type": "string" } } }
    }
  }]);
  let out = Glm47
    .parse(
      "echo<arg_key>s</arg_key><arg_value>123</arg_value>",
      Some(&tools),
    )
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "echo",
      serde_json::json!({"s": "123"})
    )],
    "string-typed arg stays a string even though `123` looks numeric"
  );
  // (2) JSON-object fallback.
  let out = Glm47
    .parse(r#"{"name":"f","arguments":{"x":1}}"#, None)
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id("f", serde_json::json!({"x": 1}))]
  );
  // (3) plain fallback → name + raw rest.
  let out = Glm47.parse("toolname key=val", None).unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "toolname",
      serde_json::json!({"key": "val"})
    )]
  );
  // (4) unknown shape (empty) → the catch-all "unknown" call (line ~1538-1541).
  let out = Glm47.parse("", None).unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "unknown",
      serde_json::json!({"raw": ""})
    )]
  );
}

#[test]
fn parse_glm47_xml_non_string_arg_is_deserialized() {
  // No schema → numeric-looking arg values get `deserialize`d (→ number).
  let out = Glm47
    .parse("f<arg_key>n</arg_key><arg_value>7</arg_value>", None)
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id("f", serde_json::json!({"n": 7}))]
  );
}

#[test]
fn parse_kimi_k2_batch_single_and_multi() {
  // Single block (no find_all matches → parse_single fallback, line ~1775-1776).
  let single = "functions.weather:0<|tool_call_argument_begin|>{\"city\":\"NYC\"}";
  let out = KimiK2.parse(single, None).unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new(
      "weather",
      serde_json::json!({"city": "NYC"}),
      Some("functions.weather:0".to_owned())
    )]
  );
  // Multi-block via the section wrappers.
  let multi = concat!(
    "<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
    "<|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>{}<|tool_call_end|>",
  );
  let out = KimiK2.parse(multi, None).unwrap();
  assert_eq!(out.len(), 2);
  assert_eq!(out[0].name(), "a");
  assert_eq!(out[0].id(), Some("functions.a:0"));
  assert_eq!(out[1].name(), "b");
  assert_eq!(out[1].id(), Some("functions.b:1"));
  // Name without the `functions.` prefix is kept as-is.
  let bare = "weather:3<|tool_call_argument_begin|>{}";
  let out = KimiK2.parse(bare, None).unwrap();
  assert_eq!(out[0].name(), "weather");
  assert_eq!(out[0].id(), Some("weather:3"));
}

#[test]
fn parse_kimi_k2_single_errors() {
  // No `<|tool_call_argument_begin|>` → error (line ~1752-1754).
  assert!(matches!(
    KimiK2.parse("no-arg-begin", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
  // No `:` in the head → error (line ~1757-1759).
  assert!(matches!(
    KimiK2
      .parse("functions.noindex<|tool_call_argument_begin|>{}", None)
      .unwrap_err(),
    Error::Tokenizer(_)
  ));
  // Non-numeric index after `:` → error (line ~1760-1762).
  assert!(matches!(
    KimiK2
      .parse("functions.f:notnum<|tool_call_argument_begin|>{}", None)
      .unwrap_err(),
    Error::Tokenizer(_)
  ));
}

#[test]
fn parse_longcat_batch() {
  // JSON fast-path: name/arguments surfaced.
  let out = Longcat
    .parse(r#"{"name":"f","arguments":{"x":1}}"#, None)
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id("f", serde_json::json!({"x": 1}))]
  );
  // JSON fast-path with NO `arguments` key → the whole object is the args
  // and name defaults to "" (line ~1953-1962).
  let out = Longcat.parse(r#"{"other":2}"#, None).unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "",
      serde_json::json!({"other": 2})
    )]
  );
  // XML-style.
  let out = Longcat
    .parse(
      "f<longcat_arg_key>k</longcat_arg_key><longcat_arg_value>1</longcat_arg_value>",
      None,
    )
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id("f", serde_json::json!({"k": 1}))]
  );
  // Neither JSON nor `<longcat_arg_key>` → error (line ~1964-1966).
  assert!(matches!(
    Longcat.parse("plain text", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
}

#[test]
fn parse_minimax_m2_batch() {
  // Two invokes, with a typed parameter (integer coercion via schema).
  let tools = serde_json::json!([{
    "function": {
      "name": "a",
      "parameters": { "properties": { "p": { "type": "integer" } } }
    }
  }]);
  let text = concat!(
    r#"<invoke name="a"><parameter name="p">5</parameter></invoke>"#,
    r#"<invoke name="b"></invoke>"#,
  );
  let out = MinimaxM2.parse(text, Some(&tools)).unwrap();
  assert_eq!(out.len(), 2);
  assert_eq!(out[0].name(), "a");
  assert_eq!(*out[0].arguments(), serde_json::json!({"p": 5}));
  assert_eq!(out[1].name(), "b");
  assert_eq!(*out[1].arguments(), serde_json::json!({}));
  // No invoke at all → error (line ~2160-2161).
  assert!(matches!(
    MinimaxM2.parse("no invokes", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
}

#[test]
fn parse_function_gemma_batch() {
  // Escaped string value + a plain JSON value.
  let out = FunctionGemma
    .parse("call:f{k:<escape>hi there<escape>,n:2}", None)
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "f",
      serde_json::json!({"k": "hi there", "n": 2})
    )]
  );
  // A non-JSON unquoted value is kept as a string (line ~2341-2342).
  let out = FunctionGemma.parse("call:g{k:bareword}", None).unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "g",
      serde_json::json!({"k": "bareword"})
    )]
  );
  // No `call:NAME{` → "No function provided." (line ~2303-2304).
  assert!(matches!(
    FunctionGemma.parse("plain", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
}

#[test]
fn parse_gemma4_batch_and_errors() {
  // Bare keys + a `<|"|>` string + a number. The second key follows the comma
  // directly (the bare-key rewrite requires no whitespace before the key).
  let out = Gemma4
    .parse(r#"call:f{k: <|"|>hello<|"|>,n: 3}"#, None)
    .unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "f",
      serde_json::json!({"k": "hello", "n": 3})
    )]
  );
  // No call → "No function provided." (line ~2513-2514).
  assert!(matches!(
    Gemma4.parse("plain", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
  // A call whose args body is not valid JSON after rewriting → error
  // (line ~2519-2520).
  assert!(matches!(
    Gemma4.parse("call:f{!bad!}", None).unwrap_err(),
    Error::Tokenizer(_)
  ));
}

// --- default parse() loop (default_parse_via_try_parse_one_call) --------

/// A test parser that does NOT override the trait `parse`, so calling
/// `parse` exercises [`default_parse_via_try_parse_one_call`]. Each
/// `try_parse_one_call` peels one `(name)` token: `(a)(b)` → two calls.
/// `(0width)` exercises the defensive zero-width break.
struct DefaultParseProbe {
  /// When true, `try_parse_one_call` reports `end_pos == 0` for the first
  /// token to exercise the zero-width-advance defensive break.
  zero_width: bool,
}

impl ToolParser for DefaultParseProbe {
  // `parse` intentionally NOT overridden → default loop is used.
  fn name(&self) -> &'static str {
    "default_parse_probe"
  }
  fn tool_call_start(&self) -> &'static str {
    "("
  }
  fn tool_call_end(&self) -> &'static str {
    ")"
  }
  fn try_parse_one_call(
    &self,
    buffer: &str,
    _tools: Option<&Value>,
  ) -> Result<Option<(Vec<ToolCall>, usize)>, Error> {
    let Some(open) = buffer.find('(') else {
      return Ok(None);
    };
    let Some(close_rel) = buffer[open + 1..].find(')') else {
      return Ok(None);
    };
    let name = buffer[open + 1..open + 1 + close_rel].to_owned();
    let end_pos = if self.zero_width {
      0
    } else {
      open + 1 + close_rel + 1
    };
    Ok(Some((
      vec![ToolCall::new_nameless_id(name, serde_json::json!({}))],
      end_pos,
    )))
  }
}

#[test]
fn default_parse_loop_extracts_back_to_back_via_trait_default() {
  // The DEFAULT trait `parse` (which loops on `try_parse_one_call`) peels
  // two back-to-back `(name)` tokens.
  let probe = DefaultParseProbe { zero_width: false };
  let out = probe.parse("(a)(b)", None).unwrap();
  assert_eq!(out.len(), 2);
  assert_eq!(out[0].name(), "a");
  assert_eq!(out[1].name(), "b");
  // No token at all → the loop's `None` break yields an empty result
  // (line ~258).
  assert!(probe.parse("no tokens", None).unwrap().is_empty());
}

#[test]
fn default_parse_loop_zero_width_advance_breaks() {
  // A `try_parse_one_call` that returns `end_pos == 0` would loop forever;
  // the default loop defensively breaks (line ~249-254). The `break` runs
  // BEFORE `out.extend(calls)`, so the zero-width batch is discarded and the
  // result is empty — the contract is "no progress, stop", not "keep this
  // batch".
  let probe = DefaultParseProbe { zero_width: true };
  let out = probe.parse("(a)(b)", None).unwrap();
  assert!(
    out.is_empty(),
    "zero-width advance breaks before extending, so no calls are collected"
  );
}

// --- selection: parser_by_name + infer_tool_parser ----------------------

#[test]
fn parser_by_name_known_and_unknown() {
  for name in [
    "json_tools",
    "pythonic",
    "mistral",
    "qwen3_coder",
    "glm47",
    "kimi_k2",
    "longcat",
    "minimax_m2",
    "function_gemma",
    "gemma4",
  ] {
    let p = parser_by_name(name).unwrap_or_else(|| panic!("known parser {name}"));
    assert_eq!(p.name(), name, "boxed parser reports its own name");
  }
  assert!(parser_by_name("does_not_exist").is_none());
}

#[test]
fn infer_tool_parser_select_rules() {
  // None template → None.
  assert_eq!(infer_tool_parser(None), None);
  // minimax_m2: `all = ["<minimax:tool_call>"]`.
  assert_eq!(
    infer_tool_parser(Some("uses <minimax:tool_call> here")),
    Some("minimax_m2")
  );
  // gemma4: needs BOTH `<|tool_call>` and `<tool_call|>`.
  assert_eq!(
    infer_tool_parser(Some("a <|tool_call> b <tool_call|> c")),
    Some("gemma4")
  );
  // function_gemma: `<start_function_call>`.
  assert_eq!(
    infer_tool_parser(Some("<start_function_call>")),
    Some("function_gemma")
  );
  // longcat.
  assert_eq!(
    infer_tool_parser(Some("<longcat_tool_call>")),
    Some("longcat")
  );
  // glm47: `<arg_key>`.
  assert_eq!(
    infer_tool_parser(Some("has <arg_key> token")),
    Some("glm47")
  );
  // pythonic: `<|tool_list_start|>`.
  assert_eq!(
    infer_tool_parser(Some("<|tool_list_start|>")),
    Some("pythonic")
  );
  // qwen3_coder: any_of with the `<tool_call>\n<function=` literal.
  assert_eq!(
    infer_tool_parser(Some("<tool_call>\n<function=")),
    Some("qwen3_coder")
  );
  // kimi_k2.
  assert_eq!(
    infer_tool_parser(Some("<|tool_calls_section_begin|>")),
    Some("kimi_k2")
  );
  // mistral.
  assert_eq!(infer_tool_parser(Some("[TOOL_CALLS]")), Some("mistral"));
  // json_tools: needs BOTH `<tool_call>` and `tool_call.name`.
  assert_eq!(
    infer_tool_parser(Some("<tool_call> ... tool_call.name")),
    Some("json_tools")
  );
  // A template matching no rule → None.
  assert_eq!(infer_tool_parser(Some("no markers at all")), None);
}

#[test]
fn infer_tool_parser_first_rule_wins() {
  // minimax_m2 precedes gemma4 in the table; a template carrying BOTH
  // minimax_m2's marker and gemma4's pair must select minimax_m2 (the
  // declaration-order if/elif semantics).
  let ct = "<minimax:tool_call> <|tool_call> <tool_call|>";
  assert_eq!(infer_tool_parser(Some(ct)), Some("minimax_m2"));
}

// --- marker lookup + ToolCall accessors ---------------------------------

#[test]
fn parser_marker_lookup() {
  // tool_call_start / tool_call_end resolve from the generated table.
  assert_eq!(JsonTools.tool_call_start(), "<tool_call>");
  assert_eq!(JsonTools.tool_call_end(), "</tool_call>");
  assert_eq!(Pythonic.tool_call_start(), "<|tool_call_start|>");
  assert_eq!(Pythonic.tool_call_end(), "<|tool_call_end|>");
  // Mistral has an empty end marker.
  assert_eq!(Mistral.tool_call_start(), "[TOOL_CALLS]");
  assert_eq!(Mistral.tool_call_end(), "");
}

#[test]
fn tool_call_public_accessors() {
  let tc = ToolCall::new(
    "fname",
    serde_json::json!({"a": 1}),
    Some("id-7".to_owned()),
  );
  assert_eq!(tc.name(), "fname");
  assert_eq!(*tc.arguments(), serde_json::json!({"a": 1}));
  assert_eq!(tc.id(), Some("id-7"));
  // new_nameless_id leaves id as None.
  let tc2 = ToolCall::new_nameless_id("g", Value::Null);
  assert_eq!(tc2.id(), None);
}

// --- processor: process_eos pending_display clear + recover_at_cap paths -

#[test]
fn process_eos_clears_pending_display_in_potential_state() {
  // Enter PotentialToolCall with leading prose parked in `pending_display`
  // (and only the ambiguous `<` prefix in `tool_call_buffer`). EOS runs
  // `parse_eos` over the unparseable `<` (yielding nothing) and clears
  // `pending_display` so the leading prose cannot leak into a later
  // generation on a reused processor (line ~2992-2995). State returns to
  // Normal with empty buffers.
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let _ = p.process_chunk("leading <");
  assert_eq!(p.state, State::PotentialToolCall);
  assert!(!p.pending_display.is_empty());
  p.process_eos();
  assert!(
    p.pending_display.is_empty(),
    "pending_display cleared at EOS"
  );
  assert!(p.tool_call_buffer.is_empty(), "buffer cleared at EOS");
  assert_eq!(p.state, State::Normal);
  assert!(p.tool_calls.is_empty());
}

#[test]
fn recover_at_cap_collecting_drops_runaway_tool_buffer() {
  // A processor in CollectingToolCall on a never-terminating body: the
  // pre-confirmation prose is flushed at confirmation (so the FIRST chunk
  // returns it), and the runaway tool buffer is dropped at the cap
  // (recover_at_cap, CollectingToolCall arm, line ~3026-3038) — never parsed
  // into a call.
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  // Leading prose + a confirmed start tag, then a runaway body. The prose is
  // returned here at start-tag confirmation.
  let first = p.process_chunk(r#"prose <tool_call>{"name":"x""#);
  assert_eq!(
    first.as_deref(),
    Some("prose "),
    "pre-confirmation prose flushed at start-tag confirmation"
  );
  assert_eq!(p.state, State::CollectingToolCall);
  assert!(
    p.pending_display.is_empty(),
    "prose moved out of pending_display"
  );
  let big = "z".repeat(64 * 1024);
  let bound = MAX_TOOL_CALL_BUFFER_BYTES + big.len();
  for _ in 0..8 {
    let _ = p.process_chunk(&big);
    // Bounded at every step (tracks the cap, never the total fed).
    assert!(p.tool_call_buffer.len() <= bound);
  }
  assert_eq!(
    p.tool_call_buffer.len(),
    0,
    "runaway tool buffer dropped at cap"
  );
  assert!(
    p.tool_calls.is_empty(),
    "runaway content never became a call"
  );
}

#[test]
fn process_eos_noop_when_already_normal() {
  // EOS while in `Normal` (e.g. after a clean passthrough) returns early
  // without touching anything (line ~2975-2977).
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  assert_eq!(p.process_chunk("plain").as_deref(), Some("plain"));
  assert_eq!(p.state, State::Normal);
  p.process_eos();
  assert_eq!(p.state, State::Normal);
  assert!(p.tool_calls.is_empty());
  assert!(p.tool_call_buffer.is_empty());
  assert!(p.pending_display.is_empty());
}

// ========================================================================
// Residual-branch coverage.
//
// The blocks above cover the streaming hot path, the batch `parse()` of
// every parser, and the literal/value-coercion helpers. The tests below
// close the remaining scattered branches: each parser's
// `try_parse_one_call` "start tag not yet in the buffer" early return
// (`locate_tagged_payload` → `Ok(None)`), the multi-block walkers'
// no-more-opener arms, a handful of helper edge branches, and the
// closed-but-malformed end-position helpers exercised directly so the
// empty-tag and not-found arms are pinned at the unit level. Every
// expected value is a hand-written closed-form oracle.
// ========================================================================

// --- try_parse_one_call: start-tag-absent → Ok(None) (per parser) -------

#[test]
fn try_parse_one_call_start_tag_absent_returns_none_per_parser() {
  // For every parser, `try_parse_one_call` on a buffer that does NOT yet
  // contain the parser's `tool_call_start` must return `Ok(None)`
  // (`locate_tagged_payload` returns `None` → the early `Ok(None)` arm).
  // The buffer is plain prose with no start-tag substring for any parser.
  let no_tag = "just some streamed prose with no start tag";
  let parsers: Vec<(&'static str, Box<dyn ToolParser>)> = vec![
    ("json_tools", Box::new(JsonTools)),
    ("pythonic", Box::new(Pythonic)),
    ("qwen3_coder", Box::new(Qwen3Coder)),
    ("glm47", Box::new(Glm47)),
    ("longcat", Box::new(Longcat)),
    ("kimi_k2", Box::new(KimiK2)),
    ("minimax_m2", Box::new(MinimaxM2)),
    ("function_gemma", Box::new(FunctionGemma)),
    ("gemma4", Box::new(Gemma4)),
  ];
  for (label, parser) in parsers {
    let r = parser
      .try_parse_one_call(no_tag, None)
      .unwrap_or_else(|e| panic!("{label}: errored: {e}"));
    assert!(
      r.is_none(),
      "{label}: start tag absent must yield Ok(None) (got {r:?})",
    );
  }
}

#[test]
fn try_parse_one_call_mistral_start_tag_absent_returns_none() {
  // Mistral's empty `tool_call_end` means `locate_tagged_payload` still
  // gates on the non-empty start tag `[TOOL_CALLS]`. With the start tag
  // absent the early `Ok(None)` arm fires (line ~1012).
  let r = Mistral
    .try_parse_one_call("no tool call markers here", None)
    .expect("Ok");
  assert!(r.is_none(), "mistral start tag absent → Ok(None)");
}

// --- multi-block walkers: opener-without-end then no-more-opener arms ----

#[test]
fn try_parse_one_call_kimi_k2_open_block_without_section_end_stays_none() {
  // A COMPLETE inner block (`<|tool_call_begin|>…{}<|tool_call_end|>`) but
  // NO section end yet. First loop iteration takes the `(None, Some(o))`
  // arm (no section end, opener present), advances the cursor past the
  // balanced `{}` args and the inner `<|tool_call_end|>`; the second
  // iteration finds neither a section end nor a further opener and takes
  // the `(None, None) => Ok(None)` arm (lines ~1829-1830).
  let buf = concat!(
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
  );
  let r = KimiK2.try_parse_one_call(buf, None).expect("Ok");
  assert!(
    r.is_none(),
    "kimi_k2 complete inner block without section end keeps collecting (Ok(None))",
  );
}

#[test]
fn try_parse_one_call_kimi_k2_balanced_args_without_inner_end_stays_none() {
  // Opener + `<|tool_call_argument_begin|>` + a balanced `{}` args object,
  // but NO inner `<|tool_call_end|>` and NO section end. The Object arm
  // balances the args, then the `args_region[obj_end..].find(call_end)`
  // fails → `r13_plain()` (line ~1856); with the section end-tag absent,
  // `closed_but_malformed_end_pos` returns None → `Ok(None)`.
  let buf = concat!(
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}",
  );
  let r = KimiK2.try_parse_one_call(buf, None).expect("Ok");
  assert!(
    r.is_none(),
    "kimi_k2 balanced args but no inner end + no section end → Ok(None)",
  );
}

#[test]
fn try_parse_one_call_minimax_m2_open_invoke_without_section_end_stays_none() {
  // `<invoke name="f">…</invoke>` complete but NO section end. First
  // iteration matches `(None, Some(o))` (no section end, opener present),
  // advances past the matching `</invoke>`; the next iteration finds
  // neither end nor opener → `(None, None) => Ok(None)` (lines ~2225-2226).
  let buf = concat!(
    "<minimax:tool_call>",
    r#"<invoke name="f"><parameter name="p">v</parameter></invoke>"#,
  );
  let r = MinimaxM2.try_parse_one_call(buf, None).expect("Ok");
  assert!(
    r.is_none(),
    "minimax_m2 complete invoke without section end → Ok(None)",
  );
}

#[test]
fn try_parse_one_call_gemma4_call_without_section_end_stays_none() {
  // `call:f{…}` complete but NO section end. The walker matches
  // `(None, Some(c))` (no section end, `call:` present), advances past the
  // brace-balanced body; the next iteration finds neither end nor `call:`
  // → `(None, None) => Ok(None)` (lines ~2554-2556).
  let buf = r#"<|tool_call>call:f{"k":"v"}"#;
  let r = Gemma4.try_parse_one_call(buf, None).expect("Ok");
  assert!(
    r.is_none(),
    "gemma4 complete call without section end → Ok(None)",
  );
}

#[test]
fn try_parse_one_call_gemma4_marker_without_open_brace_skips_then_none() {
  // `call:` markers whose name is NOT immediately followed by `{` must be
  // SKIPPED (the `j >= bytes.len() || bytes[j] != b'{'` branch advances
  // `cursor = after_marker; continue;`, lines ~2564-2566). With no section
  // end and no valid `call:name{` opener the walker eventually hits
  // `(None, None) => Ok(None)`.
  let buf = "<|tool_call>call:f no brace here";
  let r = Gemma4.try_parse_one_call(buf, None).expect("Ok");
  assert!(
    r.is_none(),
    "gemma4 `call:` without `{{` is skipped, no section end → Ok(None)",
  );
}

// --- closed_but_malformed_end_pos* helpers: empty-tag + not-found arms ---

#[test]
fn closed_but_malformed_end_pos_empty_and_absent() {
  // Empty end_tag → None (the mistral short-circuit guard, line ~318).
  assert_eq!(closed_but_malformed_end_pos("anything", 0, ""), None);
  // End_tag genuinely absent in `buffer[payload_at..]` → None.
  assert_eq!(
    closed_but_malformed_end_pos("no marker here", 0, "</tool_call>"),
    None
  );
  // Present after payload_at: returns one past the end tag.
  let s = "head</tc>tail";
  assert_eq!(
    closed_but_malformed_end_pos(s, 0, "</tc>"),
    Some(s.find("</tc>").unwrap() + "</tc>".len())
  );
  // payload_at offset is honoured: a match BEFORE payload_at is ignored,
  // only the one at/after it counts.
  let t = "</tc>AAA</tc>";
  let second = t.rfind("</tc>").unwrap();
  assert_eq!(
    closed_but_malformed_end_pos(t, 5, "</tc>"),
    Some(second + "</tc>".len()),
    "the match at offset 0 is before payload_at=5 and is skipped"
  );
}

#[test]
fn closed_but_malformed_end_pos_value_aware_empty_and_absent() {
  // Any empty arg → None (the early guard, line ~339).
  assert_eq!(
    closed_but_malformed_end_pos_value_aware("x", 0, "", "<v>", "</v>"),
    None
  );
  assert_eq!(
    closed_but_malformed_end_pos_value_aware("x", 0, "END", "", "</v>"),
    None
  );
  assert_eq!(
    closed_but_malformed_end_pos_value_aware("x", 0, "END", "<v>", ""),
    None
  );
  // End-tag OUTSIDE every value region → its position one past the tag.
  let s = "<v>inner END</v> END tail";
  let want = s.rfind("END").unwrap() + "END".len();
  assert_eq!(
    closed_but_malformed_end_pos_value_aware(s, 0, "END", "<v>", "</v>"),
    Some(want),
    "the in-value END is skipped; the outside-value END closes",
  );
  // An UNTERMINATED value region swallows the rest of the buffer; any
  // in-region end-tag is in-value text → None (line ~362, the final
  // `None`, via the `?` on the missing value close).
  assert_eq!(
    closed_but_malformed_end_pos_value_aware("<v>unterminated END", 0, "END", "<v>", "</v>"),
    None
  );
  // Neither value region nor end-tag present → the loop runs off the end
  // → None (line ~362).
  assert_eq!(
    closed_but_malformed_end_pos_value_aware("plain bytes only", 0, "END", "<v>", "</v>"),
    None
  );
}

#[test]
fn closed_but_malformed_end_pos_quote_aware_empty_escape_and_absent() {
  // Empty end_tag → None (line ~391).
  assert_eq!(
    closed_but_malformed_end_pos_quote_aware("x", 0, "", b"\""),
    None
  );
  // End-tag OUTSIDE any quoted region → one past the tag.
  let s = r#"{"s":"END"}END"#;
  let want = s.rfind("END").unwrap() + "END".len();
  assert_eq!(
    closed_but_malformed_end_pos_quote_aware(s, 0, "END", b"\""),
    Some(want),
    "the in-string END is skipped; the outside-string END closes",
  );
  // An ESCAPED quote inside the string keeps string state (the `\\"`
  // escape branch, lines ~403-405): the `"` after `\"` does NOT close the
  // string, so the END inside it stays in-string and the real close is the
  // END after the string terminator.
  let esc = r#""a\"END\"b"END"#;
  let want_esc = esc.rfind("END").unwrap() + "END".len();
  assert_eq!(
    closed_but_malformed_end_pos_quote_aware(esc, 0, "END", b"\""),
    Some(want_esc),
    "escaped quote does not end the string; in-string END is skipped",
  );
  // Still-open quoted region with the only end-tag candidate inside it
  // → None (genuinely mid-streaming).
  assert_eq!(
    closed_but_malformed_end_pos_quote_aware(r#"{"s":"END"#, 0, "END", b"\""),
    None
  );
}

#[test]
fn bound_context_or_plain_end_empty_tag_and_arms() {
  // Empty end_tag → None (line ~461).
  assert_eq!(bound_context_or_plain_end("anything", "", |_| true), None);
  // End_tag absent → None (the `payload.find(end_tag)?`).
  assert_eq!(
    bound_context_or_plain_end("no end here", "</tc>", |_| true),
    None
  );
  // Context proven (predicate true) → Some(None): caller runs its
  // syntax-aware scan over the full payload.
  assert_eq!(
    bound_context_or_plain_end("body</tc>rest", "</tc>", |_| true),
    Some(None)
  );
  // Context NOT proven (predicate false) → Some(Some(end_pos)) where
  // end_pos is one past the FIRST end_tag.
  let s = "body</tc>rest";
  let first_end = s.find("</tc>").unwrap() + "</tc>".len();
  assert_eq!(
    bound_context_or_plain_end(s, "</tc>", |_| false),
    Some(Some(first_end))
  );
}

#[test]
fn xml_value_aware_end_tag_scan_value_open_without_end_tag_then_unterminated() {
  // A value region opens and there is NO end-tag candidate anywhere
  // (`(Some(v), None)` arm, lines ~3645-3651): the region is skipped, and
  // since no end-tag follows the scan returns None.
  assert_eq!(
    xml_value_aware_end_tag_scan("<v>body</v> nothing", "<v>", "</v>", "END"),
    None,
    "value region with no end-tag candidate → None",
  );
  // A value opens before the end-tag but never closes (`(Some(v), None)`
  // arm with a missing `value_close` → the `?` returns None).
  assert_eq!(
    xml_value_aware_end_tag_scan("<v>unterminated forever", "<v>", "</v>", "END"),
    None
  );
}

// --- locate_tagged_payload: empty start tag + present/absent ------------

#[test]
fn locate_tagged_payload_empty_present_absent() {
  // Empty start tag → None (line ~275).
  assert_eq!(locate_tagged_payload("anything", ""), None);
  // Present: returns (one-past-start-tag, payload slice).
  let s = "pre<tool_call>body";
  let (at, payload) = locate_tagged_payload(s, "<tool_call>").expect("found");
  assert_eq!(at, "pre<tool_call>".len());
  assert_eq!(payload, "body");
  // Absent → None.
  assert_eq!(locate_tagged_payload("no tag", "<tool_call>"), None);
}

// --- pythonic_call_close: escaped quote inside the string ---------------

#[test]
fn pythonic_call_close_escaped_quote_in_string() {
  // A backslash-escaped quote inside the single-quoted string keeps string
  // state (the `b == b'\\' => escaped = true` branch, lines ~3549-3551):
  // the escaped `'` does NOT end the string, so the `)]` inside it does not
  // close the call. The real close is the trailing `)]`.
  let s = r#"[echo(s='a\'b)]c')]"#;
  assert_eq!(
    pythonic_call_close(s),
    Some(s.len()),
    "escaped quote keeps the string open; the in-string `)]` is skipped",
  );
  // Double-quoted analogue.
  let d = r#"[echo(s="x\")]y")]"#;
  assert_eq!(pythonic_call_close(d), Some(d.len()));
}

// --- glm_parse_json: name-as-object nested shape ------------------------

#[test]
fn glm_parse_json_name_is_object_pulls_nested_arguments() {
  // The `tool` wrapper supplies a `name` that is itself an OBJECT and NO
  // top-level `arguments`. The `if let Some(Value::Object(nm)) = &name`
  // block (lines ~1636-1640) then pulls `arguments` from the nested object
  // (the `arguments.is_none()` branch, line ~1638) and replaces `name`
  // with the nested `name` (line ~1640).
  assert_eq!(
    glm_parse_json(
      r#"{"tool":{"name":{"name":"inner","arguments":{"c":3}}}}"#,
      None
    ),
    Some(ToolCall::new_nameless_id(
      "inner",
      serde_json::json!({"c": 3})
    ))
  );
  // Top-level `{name: <object>, arguments: <object>}`: `name` is an object,
  // `arguments` already present (so line ~1638 is skipped), `name` replaced
  // by the nested `name` (line ~1640).
  assert_eq!(
    glm_parse_json(r#"{"name":{"name":"deep"},"arguments":{"a":1}}"#, None),
    Some(ToolCall::new_nameless_id(
      "deep",
      serde_json::json!({"a": 1})
    ))
  );
}

// --- glm_parse_plain: key=value with a string-typed schema arg ----------

#[test]
fn glm_parse_plain_key_value_with_string_typed_arg() {
  // `name key=value` where `key` is declared `string` in the schema → the
  // value is kept as a raw string instead of being `deserialize`d (the
  // `string_args.contains(&key)` branch, line ~1696). A numeric-looking
  // value `1` therefore stays the string "1".
  let tools = serde_json::json!([{
    "function": {
      "name": "f",
      "parameters": { "properties": { "s": { "type": "string" } } }
    }
  }]);
  assert_eq!(
    glm_parse_plain("f s=1", Some(&tools)),
    Some(ToolCall::new_nameless_id(
      "f",
      serde_json::json!({"s": "1"})
    )),
    "string-typed key=value stays a string",
  );
}

// --- find_kv_pairs: value-open present but value-close missing ----------

#[test]
fn find_kv_pairs_value_open_without_close_breaks() {
  // A key/value pair where the value-OPEN tag is found but its CLOSE tag
  // is missing terminates the scan with whatever pairs were already
  // collected (the `text[vabs..].find(vc)` → `None` → `break`, line
  // ~1733). The first complete pair is returned, the truncated one is
  // dropped.
  assert_eq!(
    find_kv_pairs(
      "<k>a</k><v>1</v><k>b</k><v>dangling",
      "<k>",
      "</k>",
      "<v>",
      "</v>"
    ),
    vec![("a".to_owned(), "1".to_owned())],
    "the second value never closes → scan stops after the first pair",
  );
}

// --- schema_types: enum float / array / object JSON-shape names ---------

#[test]
fn schema_types_enum_float_array_object_shapes() {
  // The enum-value → type-name map (lines ~2069-2076) needs a FLOAT
  // (`Value::Number(_)` non-integer → "number", line ~2073), an ARRAY
  // (→ "array", line ~2075) and an OBJECT (→ "object", line ~2076) to
  // cover the arms the existing integer/string/bool/null test misses.
  assert_eq!(
    schema_types(&serde_json::json!({"enum": [1.5, [1, 2], {"k": 1}]})),
    vec!["array".to_owned(), "number".to_owned(), "object".to_owned()],
    "float→number, list→array, dict→object (BTreeSet-sorted)",
  );
}

// --- normalize_arguments: string-typed coercion of a non-string value ---

#[test]
fn normalize_arguments_string_typed_value_is_stringified() {
  // `normalize_arguments` (used by glm47's JSON/plain fallbacks) coerces a
  // value whose key is declared `string` in the schema: a non-string value
  // is rendered via `to_string()` (the `other => Value::String(...)`
  // branch, lines ~1606-1610). A NON-string-typed key whose value IS a
  // string is `deserialize`d; a non-string-typed non-string value passes
  // through unchanged.
  let tools = serde_json::json!([{
    "function": {
      "name": "f",
      "parameters": { "properties": { "s": { "type": "string" } } }
    }
  }]);
  let mut args = serde_json::Map::new();
  // string-typed key, numeric value → stringified to "5".
  args.insert("s".to_owned(), serde_json::json!(5));
  // non-string-typed key, string value → deserialized to a number.
  args.insert("n".to_owned(), serde_json::json!("7"));
  // non-string-typed key, non-string value → passes through unchanged.
  args.insert("b".to_owned(), serde_json::json!(true));
  let out = normalize_arguments("f", &args, Some(&tools));
  assert_eq!(out.get("s"), Some(&Value::String("5".to_owned())));
  assert_eq!(out.get("n"), Some(&serde_json::json!(7)));
  assert_eq!(out.get("b"), Some(&serde_json::json!(true)));
}

// --- minimax_m2 parse: a parameter with no `>` is skipped ---------------

#[test]
fn parse_minimax_m2_parameter_without_gt_is_skipped() {
  // Inside `<invoke …>`, a `<parameter name=…` fragment with NO closing
  // `>` before the `</parameter>` is skipped (`pm.find('>')` → None →
  // `continue`, lines ~2170-2173). The well-formed parameter is still
  // captured; the malformed one contributes nothing.
  let text = concat!(
    r#"<invoke name="f">"#,
    // malformed parameter: no `>` between the name and the close tag.
    r#"<parameter name="bad"</parameter>"#,
    r#"<parameter name="ok">v</parameter>"#,
    r#"</invoke>"#,
  );
  let out = MinimaxM2.parse(text, None).unwrap();
  assert_eq!(out.len(), 1);
  assert_eq!(out[0].name(), "f");
  assert_eq!(
    *out[0].arguments(),
    serde_json::json!({"ok": "v"}),
    "the `>`-less parameter is skipped; only the well-formed one is kept",
  );
}

// --- qwen3_coder try_parse_one_call: bounded body, valid opener, no
//     `</function>` close (forward-scan exhausts) ------------------------

#[test]
fn try_parse_one_call_qwen3_coder_valid_opener_without_function_close_is_bounded_empty() {
  // The body has a VALID `<function=f>` opener and a complete
  // `<parameter=p>v</parameter>` region, but NO `</function>` before the
  // wrapper `</tool_call>` close. `bound_section` (via the value-aware
  // scan that skips the parameter region) finds the wrapper close, then
  // the forward-scan over the bounded prefix consumes the parameter
  // region and exhausts with no `</function>` candidate — the
  // `(None, _) => break None` arm (line ~1244) → `fn_close_found.is_none()`
  // → `Ok(Some((Vec::new(), end_pos)))` (line ~1269). The processor then
  // surfaces the same-chunk suffix.
  let buf = "<tool_call><function=f><parameter=p>v</parameter></tool_call>visible";
  let (calls, end_pos) = Qwen3Coder
    .try_parse_one_call(buf, None)
    .expect("Ok")
    .expect("Some — wrapper close is in the buffer");
  assert_eq!(calls.len(), 0, "no `</function>` close → zero calls");
  assert!(
    buf[..end_pos].ends_with("</tool_call>"),
    "end_pos lands one past the wrapper close",
  );
  // End-to-end: the malformed-but-bounded section drops, the suffix
  // `visible` reaches display.
  let (display, c) = run_with_parser(Box::new(Qwen3Coder), &[buf]);
  assert_eq!(c.len(), 0);
  assert_eq!(display, "visible", "same-chunk suffix survives");
}

// --- qwen3_coder parse: a parameter with no `>` is skipped --------------

#[test]
fn parse_qwen3_coder_parameter_without_gt_is_skipped() {
  // qwen3_coder's `parse` mirrors the minimax skip: a `<parameter=` capture
  // with no `>` inside it is skipped (`cap.find('>')` → None → `continue`,
  // line ~1135). The well-formed parameter is still parsed.
  let text = concat!(
    "<function=f>",
    // The first capture between `<parameter=` and `</parameter>` is just
    // `nogt` with no `>` inside it → skipped.
    "<parameter=nogt</parameter>",
    "<parameter=p>v</parameter>",
    "</function>",
  );
  let out = Qwen3Coder.parse(text, None).unwrap();
  assert_eq!(out.len(), 1);
  assert_eq!(out[0].name(), "f");
  assert_eq!(
    *out[0].arguments(),
    serde_json::json!({"p": "v"}),
    "the `>`-less parameter capture is skipped",
  );
}

// --- function_gemma parse: trailing comma / boundary value arms ---------

#[test]
fn parse_function_gemma_value_to_end_and_no_colon_break() {
  // An unquoted value that runs to the END of the args (no trailing `,`)
  // exercises the `split < args_str.len()` false branch → `args_str` set
  // to empty (lines ~2336-2340). Two keys: the first has a trailing comma,
  // the second runs to end-of-body.
  let out = FunctionGemma.parse("call:f{a:1,b:2}", None).unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id(
      "f",
      serde_json::json!({"a": 1, "b": 2})
    )]
  );
  // A non-empty body fragment with no `:` terminates the arg loop (the
  // `args_str.find(':')` → `None => break`, line ~2310): `call:g{nocolon}`
  // enters the loop with `nocolon`, finds no `:`, and breaks with no args.
  let out = FunctionGemma.parse("call:g{nocolon}", None).unwrap();
  assert_eq!(
    out,
    vec![ToolCall::new_nameless_id("g", serde_json::json!({}))],
    "non-empty body with no `:` → break → empty arguments",
  );
}

// --- recover_at_cap: PotentialToolCall flushes pending + tool buffer -----

#[test]
fn recover_at_cap_potential_state_flushes_pending_and_prefix() {
  // In `PotentialToolCall` at the cap, BOTH `pending_display` (leading
  // prose) AND `tool_call_buffer` (the ambiguous tag-prefix) are flushed
  // back as display text in stream order (recover_at_cap's
  // `!drop_tool_buffer` arm, lines ~3031-3037, via the strict-prefix
  // `cap_recover_into` at ~3267).
  //
  // Reaching this arm requires `pending_display` ALONE to exceed the cap
  // while `tool_call_buffer` is still a valid strict prefix of the start
  // tag (which can only be the few bytes of the tag). `pending_display`
  // only grows in `Normal` when a chunk has leading text before its first
  // start char — so the leading prose and the start char must arrive in
  // ONE chunk: a huge prose followed by a single `<` (a strict prefix of
  // `<tool_call>`).
  let mut p = ToolCallProcessor::new(Box::new(JsonTools), None);
  let huge_prose = "p".repeat(MAX_TOOL_CALL_BUFFER_BYTES + 1024);
  // One chunk: `huge_prose` (→ pending_display) + `<` (→ tool_call_buffer,
  // a strict prefix). The combined size already exceeds the cap, so the
  // strict-prefix branch's `cap_recover_into` trips recovery on this chunk.
  let out = p.process_chunk(&format!("{huge_prose}<"));
  // recover_at_cap (PotentialToolCall arm) flushes pending_display THEN the
  // `<` prefix bytes as display, in stream order, and resets to Normal.
  let displayed = out.expect("cap recovery flushes the buffered bytes as display");
  assert!(
    displayed.starts_with(&huge_prose),
    "leading prose is flushed first (stream order)",
  );
  assert!(
    displayed.ends_with('<'),
    "the ambiguous `<` prefix is flushed after the prose (not dropped)",
  );
  assert_eq!(displayed.len(), huge_prose.len() + 1);
  assert_eq!(p.state, State::Normal, "reset to Normal after cap recovery");
  assert_eq!(p.tool_call_buffer.len(), 0);
  assert_eq!(p.pending_display.len(), 0);
  // The processor is reusable: a fresh valid call still parses.
  let out2 = p.process_chunk(r#"<tool_call>{"name":"ok","arguments":{}}</tool_call>"#);
  assert_eq!(out2, None);
  assert_eq!(p.tool_calls.len(), 1);
  assert_eq!(p.tool_calls[0].name(), "ok");
}
