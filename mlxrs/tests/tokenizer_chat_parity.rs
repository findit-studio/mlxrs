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
    false, // continue_final_message
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

// ---------------------------------------------------------------------------
// `continue_final_message` — HF Transformers' post-render trim. The rendered
// prompt must end exactly at the final message's content, with the trailing
// end-of-turn / EOS the template appends *after* it stripped. The reference
// is `render_jinja_template` in
// `transformers/utils/chat_template_utils.py`: append the
// `"CONTINUE_FINAL_MESSAGE_TAG "` sentinel to the final message's content,
// render, then `rendered.rindex(tag.strip())` + truncate.
// ---------------------------------------------------------------------------

/// `render_jinja` with `continue_final_message=true`.
fn render_continue(template: &str, messages: &Value) -> String {
  render_jinja(
    template,
    messages,
    None,  // tools
    false, // add_generation_prompt (mutually exclusive with continue)
    true,  // continue_final_message
    None,  // bos_token
    None,  // eos_token
    false, // enable_thinking
    &json!({}),
  )
  .expect("render_jinja")
}

#[test]
fn continue_final_message_strips_trailing_end_of_turn_token() {
  // A template that appends an explicit `<|im_end|>` end-of-turn token after
  // every message's content — the common Qwen/ChatML shape.
  let template = "{% for m in messages %}\
                   <|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n\
                   {% endfor %}";
  let messages = json!([
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "hello"},
  ]);

  // continue_final_message=false: the final user turn keeps its `<|im_end|>\n`
  // (and would tokenize one-or-more extra terminator tokens into the cache).
  assert_eq!(
    render(template, &messages, None, &json!({})),
    "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n",
    "without continue_final_message the final turn's <|im_end|> is present"
  );

  // continue_final_message=true: HF appends the sentinel to "hello", renders
  // `...userhelloCONTINUE_FINAL_MESSAGE_TAG <|im_end|>\n`, then `rindex`es the
  // stripped sentinel and truncates — the prompt ends exactly at "hello", with
  // the trailing `<|im_end|>\n` GONE.
  assert_eq!(
    render_continue(template, &messages),
    "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nhello",
    "continue_final_message must strip the final turn's <|im_end|> (HF parity)"
  );
}

#[test]
fn continue_final_message_difference_is_exactly_the_terminator() {
  // The two renders differ by *exactly* the trailing end-of-turn run — the
  // cache-offset divergence the Codex finding flagged. `</s>` here.
  let template = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}";
  let messages = json!([{"role": "user", "content": "the quick brown fox"}]);

  let without = render(template, &messages, None, &json!({}));
  let with = render_continue(template, &messages);
  assert_eq!(without, "<|user|>the quick brown fox</s>");
  assert_eq!(with, "<|user|>the quick brown fox");
  // The continued render is the non-continued render minus the terminator.
  assert_eq!(
    with,
    without.strip_suffix("</s>").unwrap(),
    "continue_final_message drops exactly the trailing terminator"
  );
}

#[test]
fn continue_final_message_rstrip_branch_when_template_eats_trailing_space() {
  // HF's `if/else`: it checks whether the *full* sentinel (with its trailing
  // space) survived verbatim at `tag_loc`; if not — e.g. the template applied
  // `| trim` to the final content, eating the sentinel's trailing space — HF
  // falls back to `rendered[:tag_loc].rstrip()`. Here the final message's
  // content is `"answer  "` (trailing spaces); the template renders
  // `{{ m['content'] | trim }}`, so the mutated
  // `"answer  CONTINUE_FINAL_MESSAGE_TAG "` is trimmed to
  // `"answer  CONTINUE_FINAL_MESSAGE_TAG"` — the sentinel's trailing space is
  // gone, the full-tag check fails, and `rstrip()` removes the content's own
  // trailing spaces too. The prompt ends exactly at the trimmed content.
  let template = "{% for m in messages %}{{ m['content'] | trim }}{% endfor %}";
  let messages = json!([{"role": "user", "content": "answer  "}]);
  let out = render_continue(template, &messages);
  assert_eq!(
    out, "answer",
    "rstrip branch trims to the final content when the template eats the sentinel's space"
  );
}

#[test]
fn continue_final_message_no_terminator_template_is_a_plain_prefix() {
  // A template that appends nothing after the final content: continue and
  // non-continue render identically (continue only ever *removes* a suffix).
  let template = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}";
  let messages = json!([
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "yo"},
  ]);
  assert_eq!(
    render_continue(template, &messages),
    render(template, &messages, None, &json!({})),
    "with no trailing terminator, continue_final_message is a no-op"
  );
}

#[test]
fn continue_final_message_rejects_empty_conversation() {
  // HF raises ValueError when there is no final message to continue.
  let r = render_jinja(
    "x",
    &json!([]),
    None,
    false, // add_generation_prompt
    true,  // continue_final_message
    None,
    None,
    false,
    &json!({}),
  );
  assert!(
    r.is_err(),
    "continue_final_message over an empty conversation must error"
  );
}

/// `render_jinja` with `continue_final_message=true`, returning the `Result`
/// (the error cases below assert it is `Err`).
fn render_continue_result(template: &str, messages: &Value) -> Result<String, mlxrs::Error> {
  render_jinja(
    template,
    messages,
    None,
    false, // add_generation_prompt
    true,  // continue_final_message
    None,  // bos_token
    None,  // eos_token
    false, // enable_thinking
    &json!({}),
  )
}

#[test]
fn continue_final_message_template_emits_literal_sentinel_but_drops_content_errors() {
  // ADVERSARIAL (Codex finding): a template that emits a LITERAL sentinel
  // independent of the user's content, and never renders the content itself.
  // HF's guard (`final_message.strip() not in rendered_chat`) rejects this —
  // a trim that keyed only off the sentinel would silently return the literal
  // prefix for the WRONG prompt (caching/saving a cache the user never asked
  // for). Must be an `Err`.
  let template =
    "fixed-prefix CONTINUE_FINAL_MESSAGE_TAG {% for m in messages %}{{ m['role'] }}{% endfor %}";
  let messages = json!([{"role": "user", "content": "the secret password"}]);
  let r = render_continue_result(template, &messages);
  assert!(
    r.is_err(),
    "a template that emits a literal sentinel but never renders the final \
     content must error, not silently return the literal prefix"
  );
}

#[test]
fn continue_final_message_template_drops_sentinel_errors() {
  // ADVERSARIAL: a template that renders the content but DROPS the appended
  // sentinel entirely (e.g. it only emits a fixed string, ignoring the
  // mutated content). HF's guard (`continue_final_message_tag.strip() not in
  // rendered_chat`) → `ValueError`. Must be an `Err` (no sentinel ⇒ nowhere
  // to truncate).
  let template = "{% for m in messages %}<|{{ m['role'] }}|>{% endfor %}only-roles-no-content";
  let messages = json!([{"role": "user", "content": "hello"}]);
  let r = render_continue_result(template, &messages);
  assert!(
    r.is_err(),
    "a template that drops the continue sentinel entirely must error"
  );
}

#[test]
fn continue_final_message_normal_case_still_works() {
  // The normal case (content + sentinel both rendered) still trims correctly
  // after the validation was added — a guard against over-rejecting.
  let template = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}";
  let messages = json!([{"role": "user", "content": "the quick brown fox"}]);
  assert_eq!(
    render_continue(template, &messages),
    "<|user|>the quick brown fox",
    "the normal continue_final_message case still trims at the content"
  );
}

#[test]
fn continue_final_message_empty_final_content_is_valid() {
  // EMPTY-CONTENT case: an empty final `content` is valid — `"".strip()` is
  // `""` and `rendered.contains("")` is always true (Python `"" in s`), so the
  // guard passes and the trim yields the prefix up to the sentinel. Here the
  // template emits the role then the (empty) content then `</s>`; continue
  // truncates at the sentinel, leaving everything before the empty content.
  let template = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}";
  let messages = json!([{"role": "user", "content": ""}]);
  let out = render_continue(template, &messages);
  assert_eq!(
    out, "<|user|>",
    "empty final content + sentinel present is valid and yields the prefix"
  );
  // Sanity: the non-continued render keeps the trailing `</s>` the continue
  // path strips.
  assert_eq!(
    render(template, &messages, None, &json!({})),
    "<|user|></s>"
  );
}
