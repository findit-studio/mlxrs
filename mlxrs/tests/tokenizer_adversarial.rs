//! Adversarial / untrusted-input regression tests for the three Codex
//! findings (streaming-detok sparse-vocab OOM, deepseek_v32 thinking-mode
//! OOB panic, pythonic Unicode-slice panic) plus the byte-index parser
//! audit fixes. Each test is gated on the specific capability feature it
//! exercises so it also runs under `--features tokenizer-bpe` /
//! `tokenizer-spm` / `tokenizer-deepseek-v32` / `tokenizer-tools`, not just
//! the `lm` umbrella. Model output / `tokenizer.json` is untrusted: every
//! case must return `Err` or succeed, never panic or OOM.
#![cfg(any(
  feature = "tokenizer-spm",
  feature = "tokenizer-bpe",
  feature = "tokenizer-deepseek-v32",
  feature = "tokenizer-tools"
))]

// ---------------------------------------------------------------------------
// Finding 1 — streaming detokenizer dense-Vec OOM/overflow.
// A huge sparse token id (u32::MAX) alongside small ids must construct
// without a max-id-sized allocation and still detokenize correctly.
// ---------------------------------------------------------------------------

#[cfg(feature = "tokenizer-spm")]
#[test]
fn spm_huge_sparse_vocab_id_no_oom_and_detok_correct() {
  use mlxrs::tokenizer::{StreamingDetokenizer, stream::SpmStreamingDetokenizer};

  // u32::MAX as a dense Vec index would be a ~4GB+ allocation / overflow.
  let vocab = vec![
    ("\u{2581}Hello".to_string(), 0u32),
    ("\u{2581}world".to_string(), 1u32),
    ("!".to_string(), 2u32),
    ("\u{2581}sparse".to_string(), u32::MAX),
    ("\u{2581}mid".to_string(), 9_999_999u32),
  ];
  let mut d = SpmStreamingDetokenizer::new(vocab, true);
  d.reset();
  // Dense small ids detok identically to the non-sparse case.
  for t in [0u32, 1, 2] {
    d.add_token(t);
  }
  d.finalize();
  assert_eq!(d.text(), "Hello world!");

  // The huge / mid sparse ids resolve via the HashMap (no panic).
  let mut d2 = SpmStreamingDetokenizer::new(
    vec![
      ("\u{2581}far".to_string(), u32::MAX),
      ("\u{2581}away".to_string(), 4_000_000_000u32),
    ],
    true,
  );
  d2.reset();
  d2.add_token(u32::MAX);
  d2.add_token(4_000_000_000u32);
  d2.finalize();
  assert_eq!(d2.text(), "far away");

  // SPM no-space (trim_space=false) variant stays correct too.
  let mut d3 = SpmStreamingDetokenizer::new(vec![("\u{2581}x".to_string(), u32::MAX)], false);
  d3.reset();
  d3.add_token(u32::MAX);
  d3.finalize();
  assert_eq!(d3.text(), " x");
}

#[cfg(feature = "tokenizer-spm")]
#[test]
fn spm_byte_token_with_non_ascii_does_not_panic() {
  use mlxrs::tokenizer::{StreamingDetokenizer, stream::SpmStreamingDetokenizer};

  // `<0x€` — a `<0x`-prefixed token whose 4th+ bytes are a multi-byte char.
  // The old `value[3..5]` slice would panic on the non-char-boundary; now
  // it falls back to the raw bytes. Valid `<0x41>` still decodes to `A`.
  let vocab = vec![
    ("<0x\u{20AC}".to_string(), 0u32),
    ("<0x41>".to_string(), 1u32),
  ];
  let mut d = SpmStreamingDetokenizer::new(vocab, false);
  d.reset();
  d.add_token(1);
  d.finalize();
  assert_eq!(d.text(), "A");
}

#[cfg(feature = "tokenizer-bpe")]
#[test]
fn bpe_huge_sparse_vocab_id_no_oom_and_detok_correct() {
  use mlxrs::tokenizer::{StreamingDetokenizer, stream::BpeStreamingDetokenizer};

  let vocab = vec![
    ("Hello".to_string(), 0u32),
    ("\u{0120}world".to_string(), 1u32),
    ("\u{0120}sparse".to_string(), u32::MAX),
    ("\u{0120}mid".to_string(), 3_000_000_000u32),
  ];
  let mut d = BpeStreamingDetokenizer::new(vocab, false);
  d.reset();
  d.add_token(0);
  d.add_token(1);
  d.finalize();
  assert_eq!(d.text(), "Hello world");

  // Sparse huge ids resolve via the HashMap. F3: the previous assertion
  // codified the WRONG `"!"` semantics — `tokenmap.get(&token).unwrap_or("!")`
  // returned `"!"` for *every* absent id. mlx-lm df1d3f3 is
  // `tokenmap[token] if token < len(tokenmap) else "!"` over a dense
  // `[None] * len(vocab)` list, i.e. an in-range hole (`token <= max_id`,
  // here `max_id == u32::MAX`) is `None` → decodes to `""`, and only an
  // out-of-range id (`token > max_id`) falls back to `"!"`. Since the vocab
  // here contains `u32::MAX`, `123_456 <= max_id` ⇒ in-range hole ⇒ `""`
  // (NOT `"!"`). Updated to assert the correct mlx-lm boundary semantics.
  let mut d2 = BpeStreamingDetokenizer::new(vec![("\u{0120}far".to_string(), u32::MAX)], false);
  d2.reset();
  d2.add_token(u32::MAX);
  d2.add_token(123_456u32); // in-range hole (<= max_id == u32::MAX) → ""
  d2.finalize();
  // Leading space is trimmed on the first token (text empty); the in-range
  // hole contributes nothing. Still HashMap-backed — `u32::MAX` never
  // allocates a dense ~4GB id-indexed `Vec`.
  assert_eq!(d2.text(), "far");
}

// ---------------------------------------------------------------------------
// Finding 2 — deepseek_v32 thinking-mode developer-message OOB panic.
// `[developer, user]` (and `[developer, user, assistant]`) in thinking mode
// must not panic; output must match upstream rendering.
// ---------------------------------------------------------------------------

#[cfg(feature = "tokenizer-deepseek-v32")]
#[test]
fn deepseek_v32_thinking_developer_user_no_panic() {
  use mlxrs::tokenizer::chat::{ChatTemplateOverride, DeepseekV32};
  use serde_json::json;

  let bos = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>";
  let think_start = "<think>";
  let think_end = "</think>";

  // [developer, user], thinking enabled. developer is user-equivalent:
  // retained by drop_thinking_messages, so `full.len() == messages.len()`
  // and no OOB. The last user-equivalent message is the trailing `user`.
  let messages = json!([
    {"role": "developer", "content": "be terse"},
    {"role": "user", "content": "hi"},
  ]);
  let out = DeepseekV32
    .apply(messages.as_array().unwrap(), None, true, false, true)
    .expect("must not panic / error on [developer, user]");
  // developer (not last user idx) → ends with think_end; user (last) →
  // ends with think_start (then add_generation_prompt=true keeps it).
  let expected = format!(
    "{bos}<\u{ff5c}User\u{ff5c}>\n\n# The user's message is: be terse<\u{ff5c}Assistant\u{ff5c}>{think_end}<\u{ff5c}User\u{ff5c}>hi<\u{ff5c}Assistant\u{ff5c}>{think_start}"
  );
  assert_eq!(out, expected);

  // [developer, user, assistant] in thinking mode: also no panic.
  let messages2 = json!([
    {"role": "developer", "content": "ctx"},
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "a", "reasoning_content": "r"},
  ]);
  let out2 = DeepseekV32
    .apply(messages2.as_array().unwrap(), None, true, false, true)
    .expect("must not panic / error on [developer, user, assistant]");
  assert!(out2.starts_with(bos));
  assert!(out2.contains("# The user's message is: ctx"));
  assert!(out2.contains("<\u{ff5c}User\u{ff5c}>q<\u{ff5c}Assistant\u{ff5c}>"));
}

// ---------------------------------------------------------------------------
// Finding 3 — pythonic tool parser Unicode-after-space slice panic, plus the
// audited function_gemma `<escape>` offset site. Malformed / Unicode model
// output must return Err, never panic.
// ---------------------------------------------------------------------------

#[cfg(feature = "tokenizer-tools")]
#[test]
fn pythonic_unicode_value_after_space_no_panic() {
  use mlxrs::tokenizer::tools::{Pythonic, ToolParser};
  use serde_json::json;

  // `city=  "é"` — non-ASCII after the `=` and whitespace. The old
  // consumed-offset arithmetic sliced mid-UTF-8 and panicked.
  let calls = Pythonic
    .parse(
      "<|tool_call_start|>[f(city=  \"\u{e9}\", n= 2)]<|tool_call_end|>",
      None,
    )
    .expect("unicode-after-space must parse, not panic");
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "f");
  assert_eq!(calls[0].arguments["city"], json!("\u{e9}"));
  assert_eq!(calls[0].arguments["n"], json!(2));

  // Multi-byte unquoted value after spaces, and a trailing emoji value.
  let calls2 = Pythonic
    .parse("[g(a=  \u{1f600}\u{1f680}, b=\"\u{4e2d}\u{6587}\")]", None)
    .expect("multibyte unquoted/quoted values must not panic");
  assert_eq!(calls2[0].name, "g");
  assert_eq!(calls2[0].arguments["b"], json!("\u{4e2d}\u{6587}"));

  // ASCII path unchanged (zero behavior delta for valid inputs).
  let calls3 = Pythonic
    .parse(
      "<|tool_call_start|>[get_weather(city=\"Paris\", days=3)]<|tool_call_end|>",
      None,
    )
    .unwrap();
  assert_eq!(calls3[0].name, "get_weather");
  assert_eq!(calls3[0].arguments["city"], json!("Paris"));
  assert_eq!(calls3[0].arguments["days"], json!(3));
}

#[cfg(feature = "tokenizer-tools")]
#[test]
fn function_gemma_escape_unicode_after_value_no_panic() {
  use mlxrs::tokenizer::tools::{FunctionGemma, ToolParser};

  // Audited site: `<escape>v<escape>` followed by a non-ASCII char where
  // the `+ len(escape) + 1` byte landed. Must Err/parse, never panic.
  let r = FunctionGemma.parse(
    "<start_function_call>call:f{k:<escape>v<escape>\u{e9}}<end_function_call>",
    None,
  );
  // Either a clean parse or a tokenizer Error — never a panic.
  let _ = r;

  // The well-formed (comma-after-escape) path is unaffected.
  let ok = FunctionGemma
    .parse("call:greet{name:<escape>Bob<escape>,count:3}", None)
    .expect("valid function_gemma must still parse");
  assert_eq!(ok[0].name, "greet");
  assert_eq!(ok[0].arguments["name"], serde_json::json!("Bob"));
  assert_eq!(ok[0].arguments["count"], serde_json::json!(3));
}

#[cfg(feature = "tokenizer-tools")]
#[test]
fn gemma4_balanced_brace_non_ascii_inside_no_panic() {
  use mlxrs::tokenizer::tools::{Gemma4, ToolParser};

  // Audited site: balanced_brace_end advanced a single byte over a
  // multi-byte char, then `s[idx..]` sliced mid-codepoint and panicked.
  // Non-ASCII string content inside the braces must not crash. (gemma4
  // bare keys must immediately follow `{`/`,` — no spaces — per the
  // upstream `(?<=[{,])(\w+):` regex.)
  let calls = Gemma4
    .parse(
      "call:f{city:<|\"|>\u{e9}\u{1f600}<|\"|>,note:<|\"|>\u{4e2d}\u{6587}<|\"|>}",
      None,
    )
    .expect("non-ASCII gemma4 string values must parse, not panic");
  assert_eq!(calls[0].name, "f");
  assert_eq!(
    calls[0].arguments["city"],
    serde_json::json!("\u{e9}\u{1f600}")
  );
  assert_eq!(
    calls[0].arguments["note"],
    serde_json::json!("\u{4e2d}\u{6587}")
  );

  // ASCII gemma4 still parses correctly (no behavior delta).
  let ok = Gemma4
    .parse("call:f{name:<|\"|>Bob<|\"|>,n:2}", None)
    .expect("valid gemma4 must still parse");
  assert_eq!(ok[0].name, "f");
  assert_eq!(ok[0].arguments["name"], serde_json::json!("Bob"));
  assert_eq!(ok[0].arguments["n"], serde_json::json!(2));
}
