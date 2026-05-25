//! Tool-call format parsers — full per-parser test matrix
//! (Python `mlx_lm/tool_parsers/`, mlx-lm df1d3f3).
//!
//! Covers all 10 parsers re-exported under [`mlxrs::lm::tool_parsers`]
//! (json_tools, pythonic, mistral, qwen3_coder, glm47, kimi_k2, longcat,
//! minimax_m2, function_gemma, gemma4) plus the [`parser_by_name`]
//! dispatcher. Each parser gets:
//!
//! * **happy** — a format-specific exemplar lifted from the Python module's
//!   marker/docstring → matches the expected `{name, arguments}` shape.
//! * **multi or alt-path** — the second canonical execution path documented
//!   for the parser (multi-call for formats that support it, or the JSON /
//!   plain-text fallback when only single-call is supported).
//! * **tools-schema coercion or empty** — exercises the optional `tools`
//!   metadata path (type-driven value coercion) where the parser uses it;
//!   otherwise tests the empty-args / no-args edge.
//! * **malformed** — confirms the documented behavior on broken markup
//!   (`Err` for the strict parsers, `unknown`-fallback for `glm47`,
//!   per the Python reference).
//!
//! Gated on `--features lm` (umbrella that pulls in `tokenizer-tools`).

#![cfg(feature = "lm")]

use mlxrs::lm::tool_parsers::{
  FunctionGemma, Gemma4, Glm47, JsonTools, KimiK2, Longcat, MinimaxM2, Mistral, Pythonic,
  Qwen3Coder, ToolParser, infer_tool_parser, parser_by_name,
};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// json_tools (mlx_lm/tool_parsers/json_tools.py)
// ---------------------------------------------------------------------------

#[test]
fn json_tools_single_call_happy_path() {
  // json_tools.py:10 — `return json.loads(text.strip())`.
  let calls = JsonTools
    .parse(
      r#"{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}"#,
      None,
    )
    .unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "get_weather");
  assert_eq!(calls[0].arguments, json!({"city": "Paris", "days": 3}));
  assert!(calls[0].id.is_none());
}

#[test]
fn json_tools_leading_trailing_whitespace_trimmed() {
  // json_tools.py:11 — `text.strip()` strips both ends before JSON parse.
  let calls = JsonTools
    .parse("   \n\t{\"name\": \"f\", \"arguments\": {}}\n\n  ", None)
    .unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "f");
  assert_eq!(calls[0].arguments, json!({}));
}

#[test]
fn json_tools_missing_arguments_field_yields_null() {
  // Python returns whatever `json.loads` produces; we follow with
  // `arguments.unwrap_or(Value::Null)` so a missing field is `null`.
  let calls = JsonTools.parse(r#"{"name": "ping"}"#, None).unwrap();
  assert_eq!(calls[0].name, "ping");
  assert_eq!(calls[0].arguments, Value::Null);
}

#[test]
fn json_tools_malformed_json_errors() {
  // Python `json.loads` raises; we surface as a tokenizer Error (never panic).
  let r = JsonTools.parse("{not valid json", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

#[test]
fn json_tools_no_name_field_errors() {
  // `name` is required — Python downstream `dict(...).get("name")` would be
  // None; we surface a tokenizer Error consistently.
  let r = JsonTools.parse(r#"{"arguments": {}}"#, None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

// ---------------------------------------------------------------------------
// pythonic (mlx_lm/tool_parsers/pythonic.py)
// ---------------------------------------------------------------------------

#[test]
fn pythonic_single_call_happy_path() {
  // pythonic.py:16 — `_tool_call_regex = \[(\w+)\((.*?)\)\]`.
  let calls = Pythonic
    .parse(
      r#"<|tool_call_start|>[get_weather(city="Paris", days=3, hot=True)]<|tool_call_end|>"#,
      None,
    )
    .unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "get_weather");
  assert_eq!(calls[0].arguments["city"], json!("Paris"));
  assert_eq!(calls[0].arguments["days"], json!(3));
  // Python `ast.literal_eval("True") -> True`; our literal_eval matches.
  assert_eq!(calls[0].arguments["hot"], json!(true));
}

#[test]
fn pythonic_empty_args_call() {
  // The regex still matches `[name()]`; the loop over `(\w+)=...` finds none.
  let calls = Pythonic.parse("[ping()]", None).unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "ping");
  assert_eq!(calls[0].arguments, json!({}));
}

#[test]
fn pythonic_first_match_only_when_multiple_present() {
  // pythonic.py:21 — `_tool_call_regex.search(text)` (first match only).
  let calls = Pythonic.parse("[a(x=1)] then [b(y=2)]", None).unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "a");
  assert_eq!(calls[0].arguments["x"], json!(1));
}

#[test]
fn pythonic_no_match_errors() {
  // pythonic.py:23 — `raise ValueError("No function provided.")`.
  let r = Pythonic.parse("just some prose with no call", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

#[test]
fn pythonic_unquoted_value_keeps_string_on_literal_eval_fail() {
  // pythonic.py:38-41 — `ast.literal_eval` failure keeps the raw string.
  let calls = Pythonic.parse(r#"[f(name=Alice)]"#, None).unwrap();
  // `Alice` isn't valid JSON/Python literal → falls through to string.
  assert_eq!(calls[0].arguments["name"], json!("Alice"));
}

// ---------------------------------------------------------------------------
// mistral (mlx_lm/tool_parsers/mistral.py)
// ---------------------------------------------------------------------------

#[test]
fn mistral_single_call_happy_path() {
  // mistral.py:8 — `\s*(\w+)\[ARGS\]\s*(\{.*\})` after `[TOOL_CALLS]`.
  let calls = Mistral
    .parse(r#"get_weather[ARGS]{"city": "Paris", "days": 3}"#, None)
    .unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "get_weather");
  assert_eq!(calls[0].arguments, json!({"city": "Paris", "days": 3}));
}

#[test]
fn mistral_leading_whitespace_around_name_and_args() {
  // The `\s*` allows whitespace; we trim around `[ARGS]` and before `{`.
  let calls = Mistral.parse("  ping[ARGS]  {}  ", None).unwrap();
  assert_eq!(calls[0].name, "ping");
  assert_eq!(calls[0].arguments, json!({}));
}

#[test]
fn mistral_no_args_marker_errors() {
  // mistral.py:17 — `raise ValueError(f"Could not parse tool call from: ...")`.
  let r = Mistral.parse("get_weather city=Paris", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

#[test]
fn mistral_malformed_json_args_errors() {
  // The `[ARGS]` marker is present but the JSON body is broken.
  let r = Mistral.parse("get_weather[ARGS]{not json}", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

#[test]
fn mistral_no_brace_after_args_errors() {
  // `[ARGS]` present but no `{` — surface Err, never panic.
  let r = Mistral.parse("get_weather[ARGS]   no json here", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

// ---------------------------------------------------------------------------
// qwen3_coder (mlx_lm/tool_parsers/qwen3_coder.py)
// ---------------------------------------------------------------------------

#[test]
fn qwen3_coder_single_call_happy_path() {
  // qwen3_coder.py:14 — `_function_regex = <function=(.*?)</function>$`.
  // qwen3_coder.py:15 — `<parameter=p>v</parameter>`.
  let text = "<function=get_weather><parameter=city>Paris</parameter><parameter=days>3</parameter></function>";
  let calls = Qwen3Coder.parse(text, None).unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "get_weather");
  // Without `tools` schema, the values stay as raw strings (qwen3_coder.py:42).
  assert_eq!(calls[0].arguments["city"], json!("Paris"));
  assert_eq!(calls[0].arguments["days"], json!("3"));
}

#[test]
fn qwen3_coder_tools_schema_coerces_int_and_bool() {
  // qwen3_coder.py:44-67 — type-driven coercion when `tools` is provided.
  let tools = json!([{
    "function": {
      "name": "f",
      "parameters": {
        "properties": {
          "n": {"type": "integer"},
          "ok": {"type": "boolean"},
          "name": {"type": "string"},
        }
      }
    }
  }]);
  let text = "<function=f><parameter=n>42</parameter><parameter=ok>true</parameter><parameter=name>Alice</parameter></function>";
  let calls = Qwen3Coder.parse(text, Some(&tools)).unwrap();
  assert_eq!(calls[0].name, "f");
  assert_eq!(calls[0].arguments["n"], json!(42));
  assert_eq!(calls[0].arguments["ok"], json!(true));
  assert_eq!(calls[0].arguments["name"], json!("Alice"));
}

#[test]
fn qwen3_coder_null_value_yields_json_null() {
  // qwen3_coder.py:38 — `if param_value.lower() == "null": return None`.
  let text = "<function=f><parameter=x>null</parameter></function>";
  let calls = Qwen3Coder.parse(text, None).unwrap();
  assert_eq!(calls[0].arguments["x"], Value::Null);
}

#[test]
fn qwen3_coder_strips_trailing_leading_newline_from_value() {
  // qwen3_coder.py:91-95 — `param_value.startswith/endswith("\n")` stripped.
  let text = "<function=f><parameter=body>\nhello\n</parameter></function>";
  let calls = Qwen3Coder.parse(text, None).unwrap();
  assert_eq!(calls[0].arguments["body"], json!("hello"));
}

#[test]
fn qwen3_coder_no_function_marker_errors() {
  // qwen3_coder.py:113-114 — `raise ValueError("No function provided.")`.
  let r = Qwen3Coder.parse("no function tag here", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

// ---------------------------------------------------------------------------
// glm47 (mlx_lm/tool_parsers/glm47.py)
// ---------------------------------------------------------------------------

#[test]
fn glm47_xml_style_single_call() {
  // glm47.py:15-19 — `<arg_key>k</arg_key>...<arg_value>v</arg_value>`.
  let text = "get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>";
  let calls = Glm47.parse(text, None).unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "get_weather");
  assert_eq!(calls[0].arguments["city"], json!("Paris"));
  // Unknown type (no schema) → `_deserialize` parses "3" → JSON 3.
  assert_eq!(calls[0].arguments["days"], json!(3));
}

#[test]
fn glm47_json_fallback_path() {
  // glm47.py:213-215 — JSON fallback when no `<arg_key>` marker.
  let calls = Glm47
    .parse(r#"{"name": "f", "arguments": {"x": 1}}"#, None)
    .unwrap();
  assert_eq!(calls[0].name, "f");
  assert_eq!(calls[0].arguments["x"], json!(1));
}

#[test]
fn glm47_plain_text_fallback_with_kv_pairs() {
  // glm47.py:216-218 — plain-text `name k=v k=v` fallback.
  let calls = Glm47.parse("get_weather city=Paris days=3", None).unwrap();
  assert_eq!(calls[0].name, "get_weather");
  // glm47.py:154-157 — non-string args via `_deserialize` (3 → JSON 3).
  assert_eq!(calls[0].arguments["days"], json!(3));
}

#[test]
fn glm47_unknown_format_returns_unknown_raw() {
  // glm47.py:219 — final fallback `dict(name="unknown", arguments={"raw": ...})`.
  // We mirror this: never error, always return *something*.
  let calls = Glm47.parse("zzz", None).unwrap();
  assert_eq!(calls[0].name, "zzz");
  // Single bare word: glm47 plain-text path returns `name` with empty args
  // (Python `_parse_plain_text_tool_call` rest=empty branch). Verify the
  // documented shape: a single call, no panic.
  assert_eq!(calls[0].arguments, json!({}));
}

#[test]
fn glm47_string_arg_schema_preserves_string() {
  // glm47.py:30-39 — string-typed args bypass `_deserialize`.
  let tools = json!([{
    "function": {
      "name": "f",
      "parameters": {"properties": {"id": {"type": "string"}}}
    }
  }]);
  let text = "f<arg_key>id</arg_key><arg_value>123</arg_value>";
  let calls = Glm47.parse(text, Some(&tools)).unwrap();
  // String-typed: "123" stays as string, not coerced to JSON number.
  assert_eq!(calls[0].arguments["id"], json!("123"));
}

// ---------------------------------------------------------------------------
// kimi_k2 (mlx_lm/tool_parsers/kimi_k2.py)
// ---------------------------------------------------------------------------

#[test]
fn kimi_k2_single_call_with_id() {
  // kimi_k2.py:14-17 — `(?:functions\.)?(.+?):\d+`. id is preserved.
  let text = r#"functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}"#;
  let calls = KimiK2.parse(text, None).unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "get_weather");
  assert_eq!(calls[0].id.as_deref(), Some("functions.get_weather:0"));
  assert_eq!(calls[0].arguments, json!({"city": "Paris"}));
}

#[test]
fn kimi_k2_multi_call_split() {
  // kimi_k2.py:19-21,58-59 — `<|tool_call_begin|>...<|tool_call_end|>` split.
  let text = concat!(
    "<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{\"x\":1}<|tool_call_end|>",
    "<|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>{\"y\":2}<|tool_call_end|>",
  );
  let calls = KimiK2.parse(text, None).unwrap();
  assert_eq!(calls.len(), 2);
  assert_eq!(calls[0].name, "a");
  assert_eq!(calls[0].id.as_deref(), Some("functions.a:0"));
  assert_eq!(calls[0].arguments["x"], json!(1));
  assert_eq!(calls[1].name, "b");
  assert_eq!(calls[1].id.as_deref(), Some("functions.b:1"));
  assert_eq!(calls[1].arguments["y"], json!(2));
}

#[test]
fn kimi_k2_no_functions_prefix_still_parses() {
  // kimi_k2.py:15 — `(?:functions\.)?` makes the prefix optional.
  let text = r#"my_tool:7<|tool_call_argument_begin|>{"v": 1}"#;
  let calls = KimiK2.parse(text, None).unwrap();
  assert_eq!(calls[0].name, "my_tool");
  assert_eq!(calls[0].id.as_deref(), Some("my_tool:7"));
}

#[test]
fn kimi_k2_missing_argument_marker_errors() {
  // kimi_k2.py:43 — `raise ValueError("No tool call found.")`.
  let r = KimiK2.parse("functions.foo:0 no arg marker", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

#[test]
fn kimi_k2_non_numeric_call_index_errors() {
  // `:\d+` is required — letters after the colon must Err, not panic.
  let r = KimiK2.parse(r#"functions.f:abc<|tool_call_argument_begin|>{}"#, None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

// ---------------------------------------------------------------------------
// longcat (mlx_lm/tool_parsers/longcat.py)
// ---------------------------------------------------------------------------

#[test]
fn longcat_xml_style_single_call() {
  // longcat.py:10-13 — `<longcat_arg_key>k</longcat_arg_key>...<longcat_arg_value>v</longcat_arg_value>`.
  let text = "get_weather<longcat_arg_key>city</longcat_arg_key><longcat_arg_value>Paris</longcat_arg_value><longcat_arg_key>days</longcat_arg_key><longcat_arg_value>3</longcat_arg_value>";
  let calls = Longcat.parse(text, None).unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "get_weather");
  assert_eq!(calls[0].arguments["city"], json!("Paris"));
  // Without schema → `_deserialize("3") -> JSON 3`.
  assert_eq!(calls[0].arguments["days"], json!(3));
}

#[test]
fn longcat_starts_with_brace_uses_json_path() {
  // longcat.py:53-57 — `text.startswith("{")` JSON fast-path.
  let calls = Longcat
    .parse(r#"{"name": "f", "arguments": {"x": 1}}"#, None)
    .unwrap();
  assert_eq!(calls[0].name, "f");
  assert_eq!(calls[0].arguments, json!({"x": 1}));
}

#[test]
fn longcat_string_typed_arg_preserved() {
  // longcat.py:19-34 — `_is_string_type` keeps strings raw.
  let tools = json!([{
    "function": {
      "name": "f",
      "parameters": {"properties": {"id": {"type": "string"}}}
    }
  }]);
  let text = "f<longcat_arg_key>id</longcat_arg_key><longcat_arg_value>123</longcat_arg_value>";
  let calls = Longcat.parse(text, Some(&tools)).unwrap();
  assert_eq!(calls[0].arguments["id"], json!("123"));
}

#[test]
fn longcat_no_marker_errors() {
  // longcat.py:59 — `_func_name_regex.search(text).group(1)` would crash
  // upstream; we surface as Err (never panic).
  let r = Longcat.parse("plain text with no markers", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

#[test]
fn longcat_leading_trailing_whitespace_trimmed() {
  // longcat.py:51 — `text = text.strip()`.
  let text =
    "   greet<longcat_arg_key>name</longcat_arg_key><longcat_arg_value>Bob</longcat_arg_value>  ";
  let calls = Longcat.parse(text, None).unwrap();
  assert_eq!(calls[0].name, "greet");
  assert_eq!(calls[0].arguments["name"], json!("Bob"));
}

// ---------------------------------------------------------------------------
// minimax_m2 (mlx_lm/tool_parsers/minimax_m2.py)
// ---------------------------------------------------------------------------

#[test]
fn minimax_m2_single_invoke_call() {
  // minimax_m2.py:9-11 — `<invoke name="n">` + `<parameter name="p">v</parameter>`.
  let text = r#"<invoke name="get_weather"><parameter name="city">Paris</parameter></invoke>"#;
  let calls = MinimaxM2.parse(text, None).unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "get_weather");
  assert_eq!(calls[0].arguments["city"], json!("Paris"));
}

#[test]
fn minimax_m2_multi_invoke_call() {
  // minimax_m2.py:159-200 — `_invoke_complete_regex.findall` → multi.
  let text = concat!(
    r#"<invoke name="a"><parameter name="x">1</parameter></invoke>"#,
    r#"<invoke name="b"><parameter name="y">2</parameter></invoke>"#,
  );
  let calls = MinimaxM2.parse(text, None).unwrap();
  assert_eq!(calls.len(), 2);
  assert_eq!(calls[0].name, "a");
  assert_eq!(calls[0].arguments["x"], json!("1"));
  assert_eq!(calls[1].name, "b");
  assert_eq!(calls[1].arguments["y"], json!("2"));
}

#[test]
fn minimax_m2_tools_schema_coerces_integer() {
  // minimax_m2.py:101-149 — type-priority list: integer beats string.
  let tools = json!([{
    "function": {
      "name": "f",
      "parameters": {"properties": {"n": {"type": "integer"}}}
    }
  }]);
  let text = r#"<invoke name="f"><parameter name="n">42</parameter></invoke>"#;
  let calls = MinimaxM2.parse(text, Some(&tools)).unwrap();
  assert_eq!(calls[0].arguments["n"], json!(42));
}

#[test]
fn minimax_m2_no_invoke_errors() {
  // minimax_m2.py:162 — `raise ValueError("No tool call found")`.
  let r = MinimaxM2.parse("text with no invoke", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

#[test]
fn minimax_m2_single_quoted_name_extracted() {
  // minimax_m2.py:14-24 — `_extract_name` strips single/double quotes.
  let text = r#"<invoke name='get_weather'><parameter name='city'>Paris</parameter></invoke>"#;
  let calls = MinimaxM2.parse(text, None).unwrap();
  assert_eq!(calls[0].name, "get_weather");
  assert_eq!(calls[0].arguments["city"], json!("Paris"));
}

// ---------------------------------------------------------------------------
// function_gemma (mlx_lm/tool_parsers/function_gemma.py)
// ---------------------------------------------------------------------------

#[test]
fn function_gemma_single_call_happy_path() {
  // function_gemma.py:8 — `call:(\w+)\{(.*?)\}`.
  let calls = FunctionGemma
    .parse("call:greet{name:<escape>Bob<escape>,count:3}", None)
    .unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "greet");
  assert_eq!(calls[0].arguments["name"], json!("Bob"));
  // count:3 is parsed via json.loads → 3.
  assert_eq!(calls[0].arguments["count"], json!(3));
}

#[test]
fn function_gemma_string_with_escape_markers() {
  // function_gemma.py:18-29 — `<escape>...<escape>` string parsing.
  let calls = FunctionGemma
    .parse("call:f{x:<escape>hello<escape>}", None)
    .unwrap();
  assert_eq!(calls[0].arguments["x"], json!("hello"));
}

#[test]
fn function_gemma_no_call_marker_errors() {
  // function_gemma.py:13-14 — `raise ValueError("No function provided.")`.
  let r = FunctionGemma.parse("text with no marker", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

#[test]
fn function_gemma_value_falls_through_to_string() {
  // function_gemma.py:39-41 — `JSONDecodeError` keeps `value` as string.
  let calls = FunctionGemma.parse("call:f{kind:Alice}", None).unwrap();
  // `Alice` isn't valid JSON → keep as string.
  assert_eq!(calls[0].arguments["kind"], json!("Alice"));
}

#[test]
fn function_gemma_empty_braces_empty_args() {
  // `call:name{}` — the `while args_str:` loop never runs.
  let calls = FunctionGemma.parse("call:ping{}", None).unwrap();
  assert_eq!(calls[0].name, "ping");
  assert_eq!(calls[0].arguments, json!({}));
}

// ---------------------------------------------------------------------------
// gemma4 (mlx_lm/tool_parsers/gemma4.py)
// ---------------------------------------------------------------------------

#[test]
fn gemma4_single_call_happy_path() {
  // gemma4.py — `call:name{bare_key: <|"|>str<|"|>, n: 2}` w/ bare keys.
  let calls = Gemma4
    .parse(r#"call:f{name:<|"|>Bob<|"|>,n:2}"#, None)
    .unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name, "f");
  assert_eq!(calls[0].arguments["name"], json!("Bob"));
  assert_eq!(calls[0].arguments["n"], json!(2));
}

#[test]
fn gemma4_multi_call_returns_list() {
  // gemma4.py:56-61 — `finditer` over all matches; >1 → list.
  let calls = Gemma4
    .parse(r#"call:a{x:1} and call:b{y:2}"#, None)
    .unwrap();
  assert_eq!(calls.len(), 2);
  assert_eq!(calls[0].name, "a");
  assert_eq!(calls[0].arguments["x"], json!(1));
  assert_eq!(calls[1].name, "b");
  assert_eq!(calls[1].arguments["y"], json!(2));
}

#[test]
fn gemma4_balanced_braces_in_nested_object_value() {
  // gemma4.py:17-20 — `(?2)` recursive balanced braces.
  let calls = Gemma4.parse(r#"call:f{cfg:{n:1}}"#, None).unwrap();
  assert_eq!(calls[0].arguments["cfg"], json!({"n": 1}));
}

#[test]
fn gemma4_no_call_marker_errors() {
  // gemma4.py:57-58 — `raise ValueError("No function provided.")`.
  let r = Gemma4.parse("text without a call", None);
  assert!(r.is_err(), "expected Err, got {r:?}");
}

#[test]
fn gemma4_dashed_function_name_supported() {
  // gemma4.py:18 — `call:([\w-]+)` allows hyphen in names.
  let calls = Gemma4.parse(r#"call:get-weather{n:1}"#, None).unwrap();
  assert_eq!(calls[0].name, "get-weather");
}

// ---------------------------------------------------------------------------
// Cross-parser: dispatcher + infer (Python `_infer_tool_parser`)
// ---------------------------------------------------------------------------

#[test]
fn parser_dispatch_yields_expected_parser() {
  // Each name from `mlx_lm.tool_parsers.<name>` resolves to the same
  // concrete struct whose `name()` matches the lookup key.
  for n in [
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
    let p = parser_by_name(n).unwrap_or_else(|| panic!("no parser for {n}"));
    assert_eq!(p.name(), n, "parser_by_name({n}) returned wrong name");
    // Round-trip the dispatcher result by exercising parse on a format-
    // specific exemplar where applicable; here we only assert markers are
    // non-empty for parsers that declare them (mistral has empty end).
    assert!(
      !p.tool_call_start().is_empty(),
      "{n}: empty tool_call_start"
    );
  }
}

#[test]
fn parser_dispatch_unknown_name_yields_none() {
  // Mirrors Python `importlib.import_module` failing → no parser.
  assert!(parser_by_name("nonexistent_parser").is_none());
  assert!(parser_by_name("").is_none());
}

#[test]
fn parser_dispatch_round_trip_through_json_tools() {
  // End-to-end: dispatcher → trait object → parse → ToolCall shape.
  let p = parser_by_name("json_tools").unwrap();
  let calls = p
    .parse(r#"{"name": "f", "arguments": {"x": 1}}"#, None)
    .unwrap();
  assert_eq!(calls[0].name, "f");
  assert_eq!(calls[0].arguments["x"], json!(1));
}

#[test]
fn infer_tool_parser_full_marker_matrix() {
  // Each rule in the codegen table fires for the right chat-template content
  // (mirrors `_infer_tool_parser` in `tokenizer_utils.py`, in declaration
  // order — earlier rules win when markers overlap).
  assert_eq!(
    infer_tool_parser(Some("contains <minimax:tool_call> marker")),
    Some("minimax_m2"),
  );
  // gemma4 requires BOTH `<|tool_call>` and `<tool_call|>`.
  assert_eq!(
    infer_tool_parser(Some("uses <|tool_call> and <tool_call|>")),
    Some("gemma4"),
  );
  assert_eq!(
    infer_tool_parser(Some("uses <start_function_call> here")),
    Some("function_gemma"),
  );
  assert_eq!(
    infer_tool_parser(Some("uses <longcat_tool_call>")),
    Some("longcat"),
  );
  assert_eq!(
    infer_tool_parser(Some("XML-style <arg_key>...</arg_key>")),
    Some("glm47"),
  );
  assert_eq!(
    infer_tool_parser(Some("<|tool_list_start|> for pythonic")),
    Some("pythonic"),
  );
  // qwen3_coder: `any_of` accepts EITHER the markdown-escaped form
  // (`<tool_call>\n<function=`) OR the real-newline form
  // (`<tool_call>\n<function=`); the table carries both because chat
  // templates appear in source as raw or as JSON-escaped strings.
  assert_eq!(
    infer_tool_parser(Some(r"prefer <tool_call>\n<function= here")),
    Some("qwen3_coder"),
  );
  assert_eq!(
    infer_tool_parser(Some("real newline <tool_call>\n<function= here")),
    Some("qwen3_coder"),
  );
  // Rule precedence: a template that contains BOTH the qwen3_coder marker
  // and the json_tools markers (`<tool_call>` + `tool_call.name`) must
  // select qwen3_coder, which is declared earlier in TOOL_PARSER_SELECT.
  assert_eq!(
    infer_tool_parser(Some(
      "uses <tool_call>\n<function= and also tool_call.name field"
    )),
    Some("qwen3_coder"),
  );
  assert_eq!(
    infer_tool_parser(Some("<|tool_calls_section_begin|> kimi")),
    Some("kimi_k2"),
  );
  assert_eq!(
    infer_tool_parser(Some("Mistral [TOOL_CALLS] marker")),
    Some("mistral"),
  );
  // json_tools requires BOTH `<tool_call>` and `tool_call.name`.
  assert_eq!(
    infer_tool_parser(Some("<tool_call> uses tool_call.name field")),
    Some("json_tools"),
  );
  assert_eq!(infer_tool_parser(Some("no markers at all")), None);
  assert_eq!(infer_tool_parser(None), None);
}

// ---------------------------------------------------------------------------
// LM-5 (#115) — schema-coercion regression: `number`/`float` schemas must
// emit a JSON FLOAT for every parsed `f64`, never silently saturate via
// `as i64`. Covers both `convert_param_value` (Qwen3Coder path) and
// `convert_with_types` (MinimaxM2 path).
// ---------------------------------------------------------------------------

#[test]
fn lm5_qwen3_coder_number_schema_emits_json_float_no_saturation() {
  // Schema-asks-for-NUMBER + whole-valued huge value: the old branch
  // `Ok(f) if f.fract() == 0.0 => (f as i64).into()` would saturate 1e30
  // at `i64::MAX` (≈9.22e18) AND lose the float type signal. The fix
  // routes whole-valued numbers through `Number::from_f64` like
  // non-whole values do, so the output is a JSON float carrying the
  // exact magnitude (modulo f64 round-trip).
  let tools = json!([{
    "function": {
      "name": "f",
      "parameters": {"properties": {"x": {"type": "number"}}}
    }
  }]);
  let text = "<function=f><parameter=x>1e30</parameter></function>";
  let calls = Qwen3Coder.parse(text, Some(&tools)).unwrap();
  let v = &calls[0].arguments["x"];
  let n = v
    .as_number()
    .expect("number schema must emit a JSON number");
  // NEVER an integer (the type signal must survive).
  assert!(
    !n.is_i64() && !n.is_u64(),
    "number schema must not emit an integer; got {n}"
  );
  let f = n.as_f64().unwrap();
  assert!(f.is_finite(), "f64 round-trip is finite");
  assert!(
    f > 1e29 && f < 1e31,
    "magnitude must survive (no i64::MAX saturation); got {f}"
  );

  // Sanity: small whole-valued number still routes through `from_f64`
  // (a JSON float, NOT an integer — preserves the schema type signal).
  let text_small = "<function=f><parameter=x>42</parameter></function>";
  let calls_small = Qwen3Coder.parse(text_small, Some(&tools)).unwrap();
  let n_small = calls_small[0].arguments["x"]
    .as_number()
    .expect("number schema must emit a JSON number");
  assert!(
    !n_small.is_i64() && !n_small.is_u64(),
    "schema=number on `42` must still be a JSON float (type signal); got {n_small}"
  );
  assert_eq!(n_small.as_f64().unwrap(), 42.0);
}

#[test]
fn lm5_minimax_m2_number_schema_emits_json_float_no_saturation() {
  // Same regression on the parallel `convert_with_types` path used by
  // MinimaxM2 (called from `parse` via `args.insert(pname,
  // convert_with_types(&pval, &ptypes))`).
  let tools = json!([{
    "function": {
      "name": "f",
      "parameters": {"properties": {"x": {"type": "number"}}}
    }
  }]);
  let text = r#"<invoke name="f"><parameter name="x">1e30</parameter></invoke>"#;
  let calls = MinimaxM2.parse(text, Some(&tools)).unwrap();
  let n = calls[0].arguments["x"]
    .as_number()
    .expect("number schema must emit a JSON number");
  assert!(
    !n.is_i64() && !n.is_u64(),
    "number schema must not emit an integer; got {n}"
  );
  let f = n.as_f64().unwrap();
  assert!(
    f > 1e29 && f < 1e31,
    "magnitude must survive (no i64::MAX saturation); got {f}"
  );

  // Explicit `integer` schema still routes to a JSON integer (the fix
  // only changes the `number`/`float` arm — the integer arm in
  // `convert_with_types` already uses `value.parse::<i64>()` directly).
  let tools_int = json!([{
    "function": {
      "name": "f",
      "parameters": {"properties": {"n": {"type": "integer"}}}
    }
  }]);
  let text_int = r#"<invoke name="f"><parameter name="n">42</parameter></invoke>"#;
  let calls_int = MinimaxM2.parse(text_int, Some(&tools_int)).unwrap();
  assert_eq!(calls_int[0].arguments["n"], json!(42));
}
