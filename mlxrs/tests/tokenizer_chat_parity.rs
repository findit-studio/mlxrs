//! HF/Transformers chat-template byte-fidelity regressions (Codex round-5).
//!
//! Transformers' `_compile_jinja_template` builds the jinja environment as
//! `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
//! extensions=[AssistantTracker, jinja2.ext.loopcontrols])` and registers a
//! `tojson` that is `json.dumps(x, ensure_ascii=False, indent=indent)`.
//! `json.dumps` preserves dict insertion order. These two facts gate
//! prompt-byte parity:
//!
//!  * F1 — `trim_blocks`/`lstrip_blocks` must be enabled so multi-line
//!    templates (real HF templates put `{% for %}`/`{% if %}` on their own
//!    indented lines) render without spurious blank lines / indentation.
//!  * F2 — JSON objects flowing through the chat context / `tojson` must keep
//!    *insertion* order, not be lexically sorted (serde_json's default
//!    `BTreeMap`), to match Python `json.dumps`.
//!
//! Expected bytes below were produced by running the templates through the
//! real Transformers jinja2 reference
//! (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
//! extensions=[jinja2.ext.loopcontrols])` with the HF `tojson`), not derived
//! by hand.
#![cfg(feature = "tokenizer-chat")]

use mlxrs::tokenizer::chat::render_jinja;
use serde_json::{Value, json};

fn render(template: &str, messages: &Value, tools: Option<&Value>, extra: &Value) -> String {
  render_jinja(
    template, messages, tools, false, // add_generation_prompt
    None,  // bos_token
    None,  // eos_token
    false, // enable_thinking
    extra,
  )
  .expect("render_jinja")
}

// ---------------------------------------------------------------------------
// F1 — trim_blocks + lstrip_blocks (Transformers jinja env parity)
// ---------------------------------------------------------------------------

#[test]
fn multiline_template_block_tags_trimmed_and_lstripped() {
  // Representative of real HF templates: `{% for %}` / `{% if %}` / `{% endif
  // %}` / `{% endfor %}` each on their own line, the `if`/`endif` indented.
  let template = "\
<|start|>
{% for m in messages %}
  {% if m[\"role\"] == \"user\" %}
USER: {{ m[\"content\"] }}
  {% endif %}
{% endfor %}
<|end|>";
  let messages = json!([
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "yo"},
    {"role": "user", "content": "bye"},
  ]);
  let out = render(template, &messages, None, &json!({}));

  // Reference: Transformers jinja2 `ImmutableSandboxedEnvironment(
  //   trim_blocks=True, lstrip_blocks=True,
  //   extensions=[jinja2.ext.loopcontrols])`.from_string(template).render(...)
  //   -> '<|start|>\nUSER: hi\nUSER: bye\n<|end|>'
  //
  // Derivation: `lstrip_blocks` strips the 2-space indent before the
  // `{% if %}` / `{% endif %}` tags; `trim_blocks` drops the single newline
  // right after every block tag. Net: every block-tag line emits nothing and
  // adds no blank line; only the literal lines (`<|start|>`, `USER: …` for
  // the two `user` messages, `<|end|>`) reach the output. Without the two
  // flags the result would carry spurious blank lines + leading indentation.
  assert_eq!(
    out, "<|start|>\nUSER: hi\nUSER: bye\n<|end|>",
    "multi-line template must render with HF trim_blocks+lstrip_blocks whitespace"
  );
}

// ---------------------------------------------------------------------------
// F2 — JSON object insertion order preserved (Python json.dumps parity)
// ---------------------------------------------------------------------------

#[test]
fn tojson_compact_preserves_insertion_order() {
  // `{"b":1,"a":2}` must NOT be lexically sorted to `{"a": 2, "b": 1}`.
  // serde_json `preserve_order` + minijinja `preserve_order` keep the order
  // end-to-end (JSON text -> serde_json::Value -> minijinja Value -> tojson).
  let extra = json!({ "o": { "b": 1, "a": 2 } });
  let out = render("{{ o | tojson }}", &json!([]), None, &extra);
  // Python: json.dumps({"b":1,"a":2}, ensure_ascii=False) -> '{"b": 1, "a": 2}'
  assert_eq!(
    out, "{\"b\": 1, \"a\": 2}",
    "compact tojson must keep insertion order with Python `, `/`: ` separators"
  );
}

#[test]
fn tojson_nested_tool_schema_preserves_insertion_order_compact_and_indented() {
  // HF tool-schema shape; every nested object's key order must be insertion
  // order, not lexical (`type` before `function`, `name`/`description`/
  // `parameters`, `type`/`object`/`properties`, `city`/`units`).
  let tool = json!({
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather",
      "parameters": {
        "type": "object",
        "properties": {
          "city": { "type": "string" },
          "units": { "type": "string" }
        }
      }
    }
  });
  let extra = json!({ "o": tool });

  // Reference: json.dumps(tool, ensure_ascii=False) (insertion order, `, `/`: `).
  let compact = render("{{ o | tojson }}", &json!([]), None, &extra);
  assert_eq!(
    compact,
    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \
     \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \
     \"properties\": {\"city\": {\"type\": \"string\"}, \"units\": {\"type\": \
     \"string\"}}}}}",
    "nested tool-schema compact tojson must preserve insertion order"
  );

  // Reference: json.dumps(tool, ensure_ascii=False, indent=4).
  let indented = render("{{ o | tojson(indent=4) }}", &json!([]), None, &extra);
  let expected = concat!(
    "{\n",
    "    \"type\": \"function\",\n",
    "    \"function\": {\n",
    "        \"name\": \"get_weather\",\n",
    "        \"description\": \"Get weather\",\n",
    "        \"parameters\": {\n",
    "            \"type\": \"object\",\n",
    "            \"properties\": {\n",
    "                \"city\": {\n",
    "                    \"type\": \"string\"\n",
    "                },\n",
    "                \"units\": {\n",
    "                    \"type\": \"string\"\n",
    "                }\n",
    "            }\n",
    "        }\n",
    "    }\n",
    "}"
  );
  assert_eq!(
    indented, expected,
    "nested tool-schema tojson(indent=4) must preserve insertion order"
  );
}

// ---------------------------------------------------------------------------
// Codex round-14 — `documents` is always defined (Transformers passes
// `documents=documents`, default None, to every render call).
// ---------------------------------------------------------------------------

#[test]
fn documents_is_always_defined_matching_transformers() {
  // No documents: Transformers still passes `documents=None`, so the var is
  // *defined* (and falsy). jinja2+Transformers renders "D|NONE".
  assert_eq!(
    render(
      "{% if documents is defined %}D{% else %}M{% endif %}|\
       {% if documents %}HAS{% else %}NONE{% endif %}",
      &json!([]),
      None,
      &json!({}),
    ),
    "D|NONE",
    "documents must be defined-but-None when caller passes none (Transformers parity)"
  );

  // A documents list (callers supply it via the context): defined + usable.
  assert_eq!(
    render(
      "{% if documents %}{{ documents[0].title }}{% endif %}",
      &json!([]),
      None,
      &json!({ "documents": [ { "title": "doc-A" } ] }),
    ),
    "doc-A",
    "a supplied documents list must be defined and iterable"
  );
}
