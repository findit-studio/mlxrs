//! Regression: the `tojson` jinja filter must match transformers'
//! `_compile_jinja_template` override —
//! `json.dumps(x, ensure_ascii=False, indent=indent)`.
//!
//! Codex round-4 F3: the previous one-arg `serde_json::to_string` closure
//! (a) rejected the `indent` kwarg HF tool-schema templates use
//! (`{{ tools | tojson(indent=4) }}`) and (b) emitted serde_json's compact
//! `,`/`:` separators instead of Python's `, `/`: `, and would HTML-escape
//! via minijinja's built-in. Assert both the compact and indented forms are
//! byte-identical to Python `json.dumps(..., ensure_ascii=False)`, with
//! non-ASCII preserved (no `\uXXXX`, no `&`/`<` escaping).
#![cfg(feature = "tokenizer-chat")]

use mlxrs::tokenizer::chat::render_jinja;
use serde_json::{Value, json};

fn render(template: &str, tools: &Value, extra: &Value) -> String {
  render_jinja(
    template,
    &json!([]), // messages (unused by these templates)
    Some(tools),
    false, // add_generation_prompt
    None,  // bos_token
    None,  // eos_token
    false, // enable_thinking
    extra,
  )
  .expect("render_jinja")
}

#[test]
fn tojson_indent_kwarg_matches_python_json_dumps() {
  // HF tool-schema templates: `{{ tools | tojson(indent=4) }}`. Single-key
  // object so the assertion is independent of serde_json's map key order.
  let tools = json!([{ "name": "café" }]);
  let out = render("{{ tools | tojson(indent=4) }}", &tools, &json!({}));

  // Python: json.dumps([{"name": "café"}], ensure_ascii=False, indent=4)
  let expected = "[\n    {\n        \"name\": \"café\"\n    }\n]";
  assert_eq!(
    out, expected,
    "tojson(indent=4) must match Python json.dumps"
  );
  // ensure_ascii=False parity: the non-ASCII char survives verbatim.
  assert!(out.contains("café"), "non-ASCII must be preserved");
  assert!(
    !out.contains("\\u"),
    "must not ASCII-escape (ensure_ascii=False)"
  );
}

#[test]
fn tojson_plain_matches_python_compact_separators() {
  // `{{ x | tojson }}` — Python's default compact form uses `, ` / `: `
  // separators (separators=None, indent=None), NOT serde_json's `,`/`:`.
  let extra = json!({
    "obj": { "name": "café" },
    "arr": [1, 2, 3],
  });
  let out = render("{{ obj | tojson }}|{{ arr | tojson }}", &json!([]), &extra);

  // Python: json.dumps({"name": "café"}, ensure_ascii=False) -> {"name": "café"}
  //         json.dumps([1, 2, 3], ensure_ascii=False)        -> [1, 2, 3]
  assert_eq!(
    out, "{\"name\": \"café\"}|[1, 2, 3]",
    "compact tojson must use Python `, `/`: ` separators, non-ASCII preserved"
  );
}

#[test]
fn tojson_no_html_escaping() {
  // transformers overrides Jinja's built-in *because* it HTML-escapes
  // (`<`/`>`/`&`/`'` -> `<` …). Ours must NOT.
  let extra = json!({ "s": "a<b>&'\"c" });
  let out = render("{{ s | tojson }}", &json!([]), &extra);
  // Python: json.dumps("a<b>&'\"c", ensure_ascii=False) -> "a<b>&'\"c"
  assert_eq!(out, "\"a<b>&'\\\"c\"");
  assert!(!out.contains("\\u00"), "no HTML \\u-escaping");
}

#[test]
fn tojson_positional_indent_arg() {
  // minijinja built-in also accepts a positional indent; keep parity so
  // `{{ x | tojson(2) }}` works alongside the kwarg form.
  let extra = json!({ "v": [1] });
  let out = render("{{ v | tojson(2) }}", &json!([]), &extra);
  assert_eq!(out, "[\n  1\n]");
}

#[test]
fn tojson_bool_indent_matches_python_bool_as_int() {
  // Codex round-10: Python `bool` is a subclass of `int`, so HF's
  // `json.dumps(x, ensure_ascii=False, indent=<bool>)` treats `False` ≡ 0
  // and `True` ≡ 1 (both NOT None → indented path). Expected bytes captured
  // from CPython: json.dumps(v, ensure_ascii=False, indent=False|True).
  let extra = json!({ "v": [1], "o": { "b": 1, "a": 2 } });

  // indent=false ≡ Python indent=0 → newline-separated, empty indent.
  assert_eq!(
    render("{{ v | tojson(indent=false) }}", &json!([]), &extra),
    "[\n1\n]",
    "tojson(indent=false) must equal json.dumps(indent=False)=indent=0"
  );
  // indent=true ≡ Python indent=1 → 1-space indent.
  assert_eq!(
    render("{{ v | tojson(indent=true) }}", &json!([]), &extra),
    "[\n 1\n]",
    "tojson(indent=true) must equal json.dumps(indent=True)=indent=1"
  );
  // Insertion order preserved under the bool-indent path too (preserve_order).
  assert_eq!(
    render("{{ o | tojson(indent=false) }}", &json!([]), &extra),
    "{\n\"b\": 1,\n\"a\": 2\n}"
  );
  assert_eq!(
    render("{{ o | tojson(indent=true) }}", &json!([]), &extra),
    "{\n \"b\": 1,\n \"a\": 2\n}"
  );
}

#[test]
fn tojson_string_indent_matches_python_json_dumps() {
  // Codex round-11: Python `json.dumps` accepts a STRING `indent`, used
  // verbatim as the per-level indent. Transformers forwards it directly.
  // Expected bytes captured from CPython
  // json.dumps(v, ensure_ascii=False, indent="<s>").
  let extra = json!({ "v": [1], "o": { "b": 1, "a": 2 } });

  assert_eq!(
    render(r#"{{ v | tojson(indent="--") }}"#, &json!([]), &extra),
    "[\n--1\n]",
    r#"tojson(indent="--") must equal json.dumps(indent="--")"#
  );
  // Empty string ≡ Python indent="" → newline, no indent (same as indent=0).
  assert_eq!(
    render(r#"{{ v | tojson(indent="") }}"#, &json!([]), &extra),
    "[\n1\n]"
  );
  // Tab indent, incl. an insertion-ordered object.
  assert_eq!(
    render(r#"{{ v | tojson(indent="\t") }}"#, &json!([]), &extra),
    "[\n\t1\n]"
  );
  assert_eq!(
    render(r#"{{ o | tojson(indent="\t") }}"#, &json!([]), &extra),
    "{\n\t\"b\": 1,\n\t\"a\": 2\n}"
  );
}

#[test]
fn tojson_enormous_integer_indent_errors_not_aborts() {
  // Codex round-13: a model-controlled (untrusted) chat_template calling
  // `tojson(indent=<huge int>)` must NOT trigger an unbounded allocation /
  // process abort. `render_jinja` must return a recoverable Err instead.
  let tools = json!([1]);
  for tmpl in [
    "{{ tools | tojson(indent=9223372036854775807) }}", // i64::MAX
    "{{ tools | tojson(indent=999999999999) }}",
  ] {
    let r = render_jinja(
      tmpl,
      &json!([]),
      Some(&tools),
      false,
      None,
      None,
      false,
      &json!({}),
    );
    assert!(
      r.is_err(),
      "huge integer indent must return a recoverable error, not abort: {tmpl}"
    );
  }
  // A legitimate indent still works (cap is far above real usage).
  assert_eq!(
    render("{{ tools | tojson(indent=8) }}", &json!([1]), &json!({})),
    "[\n        1\n]"
  );
}
