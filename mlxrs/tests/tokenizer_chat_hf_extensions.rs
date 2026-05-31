//! HF/Transformers chat-template extension parity.
//!
//! Transformers' `_cached_compile_jinja_template` builds the jinja env as
//! `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
//! extensions=[AssistantTracker, jinja2.ext.loopcontrols])` and registers
//! `strftime_now(fmt) = datetime.now().strftime(fmt)`. Two facts gate
//! prompt-byte parity for a whole class of real HF templates:
//!
//!  * `AssistantTracker` provides a custom `{% generation %}` /
//!    `{% endgeneration %}` block tag. Its only job is the *training-only*
//!    `return_assistant_tokens_mask`; for inference the block is
//!    semantically transparent (jinja2 renders the body, the tags emit
//!    nothing). minijinja has no extension API, so `render_jinja` rewrites
//!    those tags into a no-output `{% if true %}...{% endif %}` block before
//!    loading. The expected bytes below were produced by running the
//!    *identical* template strings through the real Transformers reference
//!    (`transformers.utils.chat_template_utils._compile_jinja_template`,
//!    transformers 5.8.1 / jinja2 3.1.6 — `AssistantTracker` active,
//!    rendering the body and emitting nothing for the tags).
//!
//!  * `strftime_now` must format *local naive now* per Python
//!    `strftime`, not return `""` (real Llama-3.x system prompts embed
//!    `{{ strftime_now('%Y-%m-%d') }}`). Implemented via `jiff`. The
//!    formatting is factored through `strftime_at(fixed_civil_dt, fmt)` so
//!    tests assert against a FIXED instant; the expected strings are exactly
//!    what CPython `datetime(2024, 3, 7, 13, 5, 9).strftime(fmt)` produces
//!    (verified with Python 3.12).
#![cfg(feature = "tokenizer-chat")]

use mlxrs::tokenizer::chat::{render_jinja, strftime_at};
use serde_json::{Value, json};

fn render_gp(template: &str, messages: &Value, add_gen: bool, bos: Option<&str>) -> String {
  render_jinja(
    template,
    messages,
    None, // tools
    add_gen,
    false, // continue_final_message
    bos,
    None,  // eos_token
    false, // enable_thinking
    &json!({}),
  )
  .expect("render_jinja")
}

// ---------------------------------------------------------------------------
// `{% generation %}` / `{% endgeneration %}` rewrite (AssistantTracker)
// ---------------------------------------------------------------------------

/// Realistic HF shape: the assistant turn is wrapped in a generation block
/// whose `{% generation %}` / `{% endgeneration %}` tags sit on their own
/// INDENTED lines. With trim_blocks+lstrip_blocks the tag lines must vanish
/// entirely (no blank line, no leading indentation), the body must render,
/// and `add_generation_prompt` must still flow.
#[test]
fn generation_block_own_indented_lines_matches_transformers() {
  // Byte-identical to the template fed to the real Transformers reference.
  let template = "\
{{ bos_token }}
{% for m in messages %}
{% if m[\"role\"] == \"user\" %}
<|user|>
{{ m[\"content\"] }}
{% elif m[\"role\"] == \"system\" %}
<|system|>
{{ m[\"content\"] }}
{% else %}
<|assistant|>
  {% generation %}
{{ m[\"content\"] }}
  {% endgeneration %}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant|>
{% endif %}";
  let messages = json!([
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello there"},
    {"role": "user", "content": "bye"},
  ]);

  // Reference (transformers 5.8.1 / jinja2 3.1.6, AssistantTracker active):
  //   _compile_jinja_template(t).render(messages=..., bos_token="<s>",
  //                                     add_generation_prompt=False)
  let out = render_gp(template, &messages, false, Some("<s>"));
  assert_eq!(
    out, "<s>\n<|system|>\nbe brief\n<|user|>\nhi\n<|assistant|>\nhello there\n<|user|>\nbye\n",
    "generation block (own indented lines), add_generation_prompt=False"
  );

  // Reference: same, add_generation_prompt=True.
  let out_gp = render_gp(template, &messages, true, Some("<s>"));
  assert_eq!(
    out_gp,
    "<s>\n<|system|>\nbe brief\n<|user|>\nhi\n<|assistant|>\nhello there\n<|user|>\nbye\n<|assistant|>\n",
    "generation block (own indented lines), add_generation_prompt=True"
  );
}

/// Inline variant: `{% generation %}` / `{% endgeneration %}` mid-line. The
/// tags must emit nothing and the body must render in place.
#[test]
fn generation_block_inline_matches_transformers() {
  let template = "BEFORE-{% generation %}{{ messages[-1][\"content\"] }}-{% endgeneration %}-AFTER";
  let messages = json!([
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello there"},
    {"role": "user", "content": "bye"},
  ]);
  let out = render_gp(template, &messages, false, Some("<s>"));
  // Reference: _compile_jinja_template(t).render(...) -> "BEFORE-bye--AFTER"
  assert_eq!(out, "BEFORE-bye--AFTER", "inline generation block");
}

/// `{%- ... -%}` whitespace-control dash variant: leading + trailing dashes
/// must still be honoured after the rewrite (the rewrite preserves the dash
/// markers verbatim, so minijinja's lexer strips exactly as jinja2 does).
#[test]
fn generation_block_dash_whitespace_control_matches_transformers() {
  let template =
    "A\n   {%- generation -%}\n   {{ messages[0][\"content\"] }}\n   {%- endgeneration -%}\nB";
  let messages = json!([
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello there"},
    {"role": "user", "content": "bye"},
  ]);
  let out = render_gp(template, &messages, false, Some("<s>"));
  // Reference: _compile_jinja_template(t).render(...) -> "Abe briefB"
  assert_eq!(
    out, "Abe briefB",
    "dash whitespace-control generation block (left+right -)"
  );
}

/// No-space (`{%generation%}` / `{%-generation-%}`) and arbitrary internal
/// whitespace must all be recognized (jinja2's lexer tolerates them; the
/// rewrite normalizes spacing but preserves dashes). All four forms wrap the
/// same body inline, so each must produce the bare body.
#[test]
fn generation_block_spacing_variants_recognized() {
  let messages = json!([{"role": "user", "content": "Z"}]);
  for tmpl in [
    "x{%generation%}{{ messages[0][\"content\"] }}{%endgeneration%}y",
    "x{%-generation-%}{{ messages[0][\"content\"] }}{%-endgeneration-%}y",
    "x{%   generation   %}{{ messages[0][\"content\"] }}{%   endgeneration   %}y",
    "x{% generation %}{{ messages[0][\"content\"] }}{% endgeneration %}y",
  ] {
    let out = render_gp(tmpl, &messages, false, None);
    // `{%-...-%}` strips the surrounding `x`/`y`-adjacent nothing here (no
    // whitespace between them and the tags), so every form yields "xZy".
    assert_eq!(out, "xZy", "spacing variant must render body: {tmpl:?}");
  }
}

// ---------------------------------------------------------------------------
// `{% generation %}` rewrite must be Jinja
// context-aware: a `{% generation %}` *inside* a `{% raw %}` block is LITERAL
// text in jinja2/Transformers (AssistantTracker never sees it) and must NOT
// be rewritten; a non-target tag whose content merely contains the substring
// `generation` or a quoted `%}` must not be corrupted. Every expected string
// below was produced by rendering the IDENTICAL template through the real
// reference `transformers.utils.chat_template_utils._compile_jinja_template`
// (transformers 5.8.1 / jinja2 3.1.6, `AssistantTracker` active).
// ---------------------------------------------------------------------------

/// `{% generation %}` inside `{% raw %}…{% endraw %}` is literal text:
/// Transformers emits the tag verbatim (it never reaches AssistantTracker),
/// so the rewrite must leave it alone. Reference:
///   _compile_jinja_template(t).render()
///     -> "PRE-{% generation %}x{% endgeneration %}-POST"
#[test]
fn generation_inside_raw_block_is_literal_matches_transformers() {
  let template = "PRE-{% raw %}{% generation %}x{% endgeneration %}{% endraw %}-POST";
  let out = render_gp(template, &json!([]), false, None);
  assert_eq!(
    out, "PRE-{% generation %}x{% endgeneration %}-POST",
    "generation tags inside a raw block must be preserved literally"
  );
}

/// Whitespace-control raw variant `{%- raw -%} … {%- endraw -%}`: the dashes
/// strip surrounding whitespace exactly as jinja2 does, and the inner
/// `{% generation %}` is still literal. Reference:
///   _compile_jinja_template(
///     "A\n   {%- raw -%}\n{% generation %}body{% endgeneration %}\n   {%- endraw -%}\nB"
///   ).render() -> "A{% generation %}body{% endgeneration %}B"
#[test]
fn generation_inside_ws_control_raw_block_matches_transformers() {
  let template = "A\n   {%- raw -%}\n{% generation %}body{% endgeneration %}\n   {%- endraw -%}\nB";
  let out = render_gp(template, &json!([]), false, None);
  assert_eq!(
    out, "A{% generation %}body{% endgeneration %}B",
    "ws-control raw open/close whitespace control + literal generation"
  );
}

/// Real-shaped template that uses `{% generation %}` *normally* (must be
/// rewritten, body renders) AND has a separate `{% raw %}` block elsewhere
/// containing literal `{% generation %}` / `{{ … }}` text (must be preserved
/// verbatim). Both behaviours must hold together.
///
/// The template is laid out on a single line (real HF templates are JSON
/// strings, so this is representative). A *multi-line* nested-block →
/// newline → `{% raw %}` layout instead exercises a pre-existing,
/// rewrite-independent minijinja-vs-jinja2 core `trim_blocks` divergence
/// (minijinja drops one `\n` jinja2 keeps — the same documented lexer-class
/// difference as the CRLF case in `strip_generation_tags`' docs; verified by
/// feeding the byte-identical post-rewrite template to *both* engines). That
/// is out of scope to fix and is intentionally not what this contract test
/// asserts; this single-line shape renders byte-identically in minijinja and
/// jinja2 while still proving the raw-context contract. Reference
/// (transformers 5.8.1 / jinja2 3.1.6, AssistantTracker active):
///   render(messages=[{user,hi},{assistant,"hello there"}],
///          bos_token="<s>", add_generation_prompt=False|True)
#[test]
fn real_template_mixes_generation_and_raw_block_matches_transformers() {
  let template = "{{ bos_token }} {% for m in messages %}{% if m[\"role\"] == \"user\" %}<|u|>{{ m[\"content\"] }}{% else %}<|a|>{% generation %}{{ m[\"content\"] }}{% endgeneration %}{% endif %}{% endfor %} {% raw %}LIT {% generation %} {{ x }} {% endgeneration %} END{% endraw %} DONE{% if add_generation_prompt %} <|a|>{% endif %}";
  let messages = json!([
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello there"},
  ]);
  let out = render_gp(template, &messages, false, Some("<s>"));
  assert_eq!(
    out, "<s> <|u|>hi<|a|>hello there LIT {% generation %} {{ x }} {% endgeneration %} END DONE",
    "mixed real generation + literal raw block, add_generation_prompt=False"
  );
  let out_gp = render_gp(template, &messages, true, Some("<s>"));
  assert_eq!(
    out_gp,
    "<s> <|u|>hi<|a|>hello there LIT {% generation %} {{ x }} {% endgeneration %} END DONE <|a|>",
    "mixed real generation + literal raw block, add_generation_prompt=True"
  );
}

/// A non-target `{% set %}` tag whose string literal contains both the
/// substring `generation` AND a `%}` brace must pass through opaque (not be
/// split at the in-string `%}`, not be rewritten). Reference:
///   _compile_jinja_template(
///     "{% set generation_x = \"has %} brace and generation word\" %}[{{ generation_x }}]"
///   ).render() -> "[has %} brace and generation word]"
#[test]
fn non_target_tag_with_substring_and_quoted_brace_not_corrupted() {
  let template =
    "{% set generation_x = \"has %} brace and generation word\" %}[{{ generation_x }}]";
  let out = render_gp(template, &json!([]), false, None);
  assert_eq!(
    out, "[has %} brace and generation word]",
    "non-target tag with generation substring + quoted brace must not be corrupted"
  );
}

/// A non-target tag binding a string `"generation"` next to a *real*
/// `{% generation %}` tag: the string is opaque, the real tag is rewritten.
/// Reference:
///   _compile_jinja_template(
///     "{% set x = \"generation\" %}{{ x }}|{% generation %}IN{% endgeneration %}|done"
///   ).render() -> "generation|IN|done"
#[test]
fn quoted_generation_string_then_real_generation_tag_matches_transformers() {
  let template = "{% set x = \"generation\" %}{{ x }}|{% generation %}IN{% endgeneration %}|done";
  let out = render_gp(template, &json!([]), false, None);
  assert_eq!(
    out, "generation|IN|done",
    "quoted `generation` string is opaque; adjacent real generation tag still rewritten"
  );
}

// ---------------------------------------------------------------------------
// the rewrite is now a COMPLETE Jinja delimiter
// pre-lexer: a `{% generation %}`-looking sequence inside a `{{ … }}`
// expression string literal or a `{# … #}` comment is NOT a statement tag in
// jinja2/Transformers and must be copied through byte-for-byte verbatim
// (the earlier rewrite only guarded `{% raw %}` + non-target `{% … %}` tags
// and would corrupt these). Every expected string below was produced by
// rendering the
// IDENTICAL template through the real reference
// `transformers.utils.chat_template_utils._compile_jinja_template`
// (venv `transformers==5.8.1 jinja2==3.1.6`, `AssistantTracker` active).
// ---------------------------------------------------------------------------

/// `{{ "{% generation %}" }}` / `{{ '{% endgeneration %}' }}`: the tag text
/// is a *string literal inside an expression* — valid Jinja, and Transformers
/// emits the LITERAL text. The pre-lexer must copy the whole expression
/// opaquely and never rewrite it. Reference:
///   _compile_jinja_template('{{ "{% generation %}" }}').render()
///     -> "{% generation %}"
///   _compile_jinja_template("{{ '{% endgeneration %}' }}").render()
///     -> "{% endgeneration %}"
#[test]
fn generation_tag_text_inside_expression_is_literal_matches_transformers() {
  let out = render_gp(r#"{{ "{% generation %}" }}"#, &json!([]), false, None);
  assert_eq!(
    out, "{% generation %}",
    "literal generation tag inside an expression string must be preserved"
  );
  let out2 = render_gp(r#"{{ '{% endgeneration %}' }}"#, &json!([]), false, None);
  assert_eq!(
    out2, "{% endgeneration %}",
    "literal endgeneration tag inside an expression string must be preserved"
  );
}

/// `{# {% generation %} #}` — a `{% generation %}` inside a Jinja comment is
/// not a statement tag; the whole comment is opaque (and non-nesting). And a
/// comment that contains `{% raw %}`/`{% endraw %}` placed ADJACENT to a real
/// `{% generation %}`…`{% endgeneration %}` block must stay opaque while the
/// real block is still correctly rewritten. References:
///   _compile_jinja_template('A{# {% generation %} #}B').render() -> "AB"
///   _compile_jinja_template(
///     'X{# c with {% raw %} {% endraw %} inside #}'
///     '{% generation %}{{ messages[0]["content"] }}{% endgeneration %}Y'
///   ).render(messages=[{user,hi}]) -> "XhiY"
#[test]
fn generation_and_raw_inside_comment_are_opaque_matches_transformers() {
  let out = render_gp("A{# {% generation %} #}B", &json!([]), false, None);
  assert_eq!(out, "AB", "generation tag inside a comment must be opaque");

  let template = "X{# c with {% raw %} {% endraw %} inside #}{% generation %}{{ messages[0][\"content\"] }}{% endgeneration %}Y";
  let messages = json!([{"role": "user", "content": "hi"}]);
  let out2 = render_gp(template, &messages, false, None);
  assert_eq!(
    out2, "XhiY",
    "comment containing raw/generation is opaque; adjacent real generation block still rewritten"
  );
}

/// `{{ "{# not a comment #}" }}`: the `{# … #}` text is inside an expression
/// string literal, so it is NOT a comment — Transformers emits it literally.
/// And a single template mixing an expression-with-literal-tag, a
/// nested-looking comment (Jinja comments do NOT nest — first `#}` closes), a
/// real generation block, a raw block, and a comment-text expression must all
/// render correctly together. References:
///   _compile_jinja_template('{{ "{# not a comment #}" }}').render()
///     -> "{# not a comment #}"
///   _compile_jinja_template(
///     'S{{ "{% generation %}" }}|'
///     '{# {# nested? #} still comment? #}|'
///     '{% generation %}{{ messages[0]["content"] }}{% endgeneration %}|'
///     '{% raw %}{% generation %}LIT{{ x }}{% endraw %}|'
///     '{{ "{# c #}" }}E'
///   ).render(messages=[{user,hi}])
///     -> "S{% generation %}| still comment? #}|hi|{% generation %}LIT{{ x }}|{# c #}E"
#[test]
fn comment_text_inside_expression_and_mixed_constructs_matches_transformers() {
  let out = render_gp(r#"{{ "{# not a comment #}" }}"#, &json!([]), false, None);
  assert_eq!(
    out, "{# not a comment #}",
    "comment-delimited text inside an expression string literal is literal, not a comment"
  );

  let template = concat!(
    r#"S{{ "{% generation %}" }}|"#,
    "{# {# nested? #} still comment? #}|",
    r#"{% generation %}{{ messages[0]["content"] }}{% endgeneration %}|"#,
    "{% raw %}{% generation %}LIT{{ x }}{% endraw %}|",
    r#"{{ "{# c #}" }}E"#,
  );
  let messages = json!([{"role": "user", "content": "hi"}]);
  let out2 = render_gp(template, &messages, false, None);
  assert_eq!(
    out2, "S{% generation %}| still comment? #}|hi|{% generation %}LIT{{ x }}|{# c #}E",
    "expression+nested-looking-comment+real-generation+raw+comment-text all correct together"
  );
}

// ---------------------------------------------------------------------------
// Jinja `+` whitespace-control parity. jinja2 3.1.6's
// generic `block_begin` regex accepts `{%(\-|\+|)…(\-|\+|)%}`: `-` strips,
// `+` *explicitly keeps* (overriding trim_blocks/lstrip_blocks), none =
// default. Transformers parses `generation` with the standard jinja2 parser,
// so `{%+ generation %}` / `{% generation +%}` / `{%- … +%}` mixes are all
// valid; the rewrite must recognize them and preserve each side's exact
// `-`/`+`/none marker so minijinja 2.19 (which honours `-`/`+` identically)
// reproduces jinja2's whitespace bytes. Every expected string below was
// produced by rendering the IDENTICAL template through the real reference
// `transformers.utils.chat_template_utils._compile_jinja_template`
// (venv `transformers==5.8.1 jinja2==3.1.6`, `AssistantTracker` active).
// ---------------------------------------------------------------------------

/// `{%+ generation %}` and `{% generation +%}` (and `-`/`+` mixes) are valid
/// jinja2 block tags; the old "strip `-` only" classifier mis-saw them as an
/// unknown `generation` tag. References (transformers 5.8.1 / jinja2 3.1.6,
/// AssistantTracker active):
///   _compile_jinja_template('A{%+ generation %}X{% endgeneration %}B')   -> "AXB"
///   _compile_jinja_template('A{% generation +%}X{% endgeneration %}B')   -> "AXB"
///   _compile_jinja_template('A{%+ generation -%}X{%- endgeneration +%}B')-> "AXB"
///   _compile_jinja_template('A{%- generation +%}X{%+ endgeneration -%}B')-> "AXB"
#[test]
fn generation_plus_ws_control_recognized_matches_transformers() {
  for (tmpl, expect) in [
    ("A{%+ generation %}X{% endgeneration %}B", "AXB"),
    ("A{% generation +%}X{% endgeneration %}B", "AXB"),
    ("A{%+ generation -%}X{%- endgeneration +%}B", "AXB"),
    ("A{%- generation +%}X{%+ endgeneration -%}B", "AXB"),
  ] {
    let out = render_gp(tmpl, &json!([]), false, None);
    assert_eq!(out, expect, "`+`/`-` ws-control generation tag: {tmpl:?}");
  }
}

/// `+` must *keep* whitespace (overriding trim_blocks/lstrip_blocks) exactly
/// where `-` would strip it — own-line/indented shape. References:
///   _compile_jinja_template(
///     "A\n   {%+ generation %}\n   X\n   {% endgeneration +%}\n   B"
///   ).render() -> "A\n      X\n\n   B"
///   _compile_jinja_template(
///     "A\n   {%- generation %}\n   X\n   {% endgeneration -%}\n   B"
///   ).render() -> "A   X\nB"
#[test]
fn generation_plus_keeps_ws_minus_strips_matches_transformers() {
  let keep = "A\n   {%+ generation %}\n   X\n   {% endgeneration +%}\n   B";
  assert_eq!(
    render_gp(keep, &json!([]), false, None),
    "A\n      X\n\n   B",
    "`+` keeps whitespace, overriding trim_blocks/lstrip_blocks"
  );
  let strip = "A\n   {%- generation %}\n   X\n   {% endgeneration -%}\n   B";
  assert_eq!(
    render_gp(strip, &json!([]), false, None),
    "A   X\nB",
    "`-` strips whitespace (comparison baseline)"
  );
  // Body still renders normally with `+` controls.
  let with_msg = "A{%+ generation %}{{ messages[0][\"content\"] }}{% endgeneration +%}B";
  assert_eq!(
    render_gp(
      with_msg,
      &json!([{"role": "user", "content": "hi"}]),
      false,
      None
    ),
    "AhiB",
    "`+`-controlled generation block still renders its body"
  );
}

/// `{%+ raw %}…{% endraw +%}` — jinja2's *dedicated* `raw_begin` regex
/// accepts `-`/`+`/none on the open and `-`/`+`/none on the endraw, so this
/// is a valid raw block; its interior `{% generation %}` is LITERAL and the
/// whole region (with its exact `+`/`-` markers) is copied verbatim.
/// References:
///   _compile_jinja_template(
///     'P{%+ raw %}{% generation %}Z{% endgeneration %}{% endraw +%}Q'
///   ).render() -> "P{% generation %}Z{% endgeneration %}Q"
///   _compile_jinja_template(
///     'P{%- raw -%} {% generation %} {% endraw +%}Q'
///   ).render() -> "P{% generation %} Q"
#[test]
fn raw_plus_ws_control_block_is_literal_matches_transformers() {
  let t1 = "P{%+ raw %}{% generation %}Z{% endgeneration %}{% endraw +%}Q";
  assert_eq!(
    render_gp(t1, &json!([]), false, None),
    "P{% generation %}Z{% endgeneration %}Q",
    "plus-control raw block: interior generation stays literal, markers preserved"
  );
  let t2 = "P{%- raw -%} {% generation %} {% endraw +%}Q";
  assert_eq!(
    render_gp(t2, &json!([]), false, None),
    "P{% generation %} Q",
    "minus/plus ws-control raw block, literal interior"
  );
}

// ---------------------------------------------------------------------------
// `{% raw %}` is a pure raw-text region: ALL interior
// `{%`/`{{`/`{#`/quotes are literal raw body, not tags. The old `find_endraw`
// parsed every interior `{% … %}` as a tag and skipped past non-endraw ones,
// so an interior unterminated-looking `{% foo` swallowed the real
// `{% endraw %}` and left the post-raw real generation block unrewritten
// (minijinja then rejected). The fixed scanner does a literal endraw
// delimiter text search. References: real transformers 5.8.1 / jinja2 3.1.6,
// AssistantTracker active.
// ---------------------------------------------------------------------------

/// `{% raw %}A {% foo B {% endraw %}{% generation %}X{% endgeneration %}`:
/// the interior `{% foo B ` is literal raw body (jinja2's raw lexer never
/// parses it), so the FIRST `{% endraw %}` closes the block and the post-raw
/// real `{% generation %}` is correctly rewritten. References:
///   _compile_jinja_template(
///     '{% raw %}A {% foo B {% endraw %}{% generation %}X{% endgeneration %}'
///   ).render() -> "A {% foo B X"
#[test]
fn raw_interior_unterminated_looking_tag_is_literal_matches_transformers() {
  let template = "{% raw %}A {% foo B {% endraw %}{% generation %}X{% endgeneration %}";
  assert_eq!(
    render_gp(template, &json!([]), false, None),
    "A {% foo B X",
    "interior foo-looking tag is literal raw body; first endraw closes; post-raw generation rewritten"
  );
}

/// Interior `{{`, `{#`, an unbalanced quote, an unterminated-looking `{%`
/// before the real `{% endraw %}`, then a real generation block after — all
/// must render byte-identically to Transformers. Also covers a
/// whitespace-controlled `{%- endraw -%}` close delimiter found by the
/// raw-mode text scan. References:
///   _compile_jinja_template(
///     '{% raw %}A {{ x }} {# c #} {% if y B "q {% endraw %}'
///     '{% generation %}{{ messages[0]["content"] }}{% endgeneration %}DONE'
///   ).render(messages=[{user,hi}]) -> 'A {{ x }} {# c #} {% if y B "q hiDONE'
///   _compile_jinja_template(
///     '{% raw %}lit {% if z {%- endraw -%}'
///     '{% generation %}{{ messages[0]["content"] }}{% endgeneration %}END'
///   ).render(messages=[{user,hi}]) -> "lit {% if zhiEND"
#[test]
fn raw_interior_mixed_constructs_and_ws_control_endraw_matches_transformers() {
  let messages = json!([{"role": "user", "content": "hi"}]);

  let t1 = concat!(
    r#"{% raw %}A {{ x }} {# c #} {% if y B "q {% endraw %}"#,
    r#"{% generation %}{{ messages[0]["content"] }}{% endgeneration %}DONE"#,
  );
  assert_eq!(
    render_gp(t1, &messages, false, None),
    r#"A {{ x }} {# c #} {% if y B "q hiDONE"#,
    "interior expr/comment/quote/unterminated-tag are literal raw body; post-raw generation rewritten"
  );

  let t2 = concat!(
    "{% raw %}lit {% if z {%- endraw -%}",
    r#"{% generation %}{{ messages[0]["content"] }}{% endgeneration %}END"#,
  );
  assert_eq!(
    render_gp(t2, &messages, false, None),
    "lit {% if zhiEND",
    "ws-controlled minus-endraw-minus close found by raw-mode text scan; post-raw generation rewritten"
  );
}

// ---------------------------------------------------------------------------
// strftime_now via jiff (injectable fixed-clock seam: strftime_at)
// ---------------------------------------------------------------------------

/// `strftime_at(fixed, fmt)` must equal exactly what CPython
/// `datetime(2024, 3, 7, 13, 5, 9).strftime(fmt)` produces. Reference values
/// captured with Python 3.12:
///   %Y-%m-%d            -> "2024-03-07"
///   %B %d, %Y           -> "March 07, 2024"
///   %Y-%m-%d %H:%M:%S   -> "2024-03-07 13:05:09"
///   %A %a %b %j %p %I %y %-d %e -> see asserts below
#[test]
fn strftime_at_matches_python_datetime_strftime() {
  // Fixed naive civil datetime (Thursday 2024-03-07 13:05:09), matching
  // Python's `datetime(2024, 3, 7, 13, 5, 9)` (time-zone-naive, like
  // `datetime.now()` that Transformers' `strftime_now` formats).
  let dt = jiff::civil::date(2024, 3, 7).at(13, 5, 9, 0);

  let s = |f: &str| strftime_at(dt, f).expect("supported directive");
  assert_eq!(s("%Y-%m-%d"), "2024-03-07");
  assert_eq!(s("%B %d, %Y"), "March 07, 2024");
  assert_eq!(s("%Y-%m-%d %H:%M:%S"), "2024-03-07 13:05:09");
  assert_eq!(s("%A"), "Thursday");
  assert_eq!(s("%a"), "Thu");
  assert_eq!(s("%b"), "Mar");
  assert_eq!(s("%j"), "067");
  assert_eq!(s("%p"), "PM");
  assert_eq!(s("%I"), "01");
  assert_eq!(s("%y"), "24");
  assert_eq!(s("%-d"), "7");
  assert_eq!(s("%e"), " 7");
  // Literal `%%` (Python: "100% 2024").
  assert_eq!(s("100%% %Y"), "100% 2024");
}

/// An untrusted `chat_template` must not panic the process
/// via `strftime_now`. `%z`/`%Z` on a naive datetime → CPython `""`;
/// `%%z` is the literal text `%z` (not stripped); an unknown directive is a
/// *recoverable* error, never a panic/abort.
#[test]
fn strftime_naive_offset_directives_and_unknown_are_safe() {
  let dt = jiff::civil::date(2024, 3, 7).at(13, 5, 9, 0);
  // CPython: datetime(2024,3,7,13,5,9).strftime("%z")=="", ("%Z")=="".
  assert_eq!(strftime_at(dt, "%z").unwrap(), "");
  assert_eq!(strftime_at(dt, "%Z").unwrap(), "");
  assert_eq!(strftime_at(dt, "[%z]").unwrap(), "[]");
  assert_eq!(strftime_at(dt, "%Y%z-%m").unwrap(), "2024-03");
  // `%%z` = literal `%` + `z` (CPython: "%z"), NOT the stripped directive.
  assert_eq!(strftime_at(dt, "%%z").unwrap(), "%z");
  // Unknown directive → recoverable Err, not a panic.
  assert!(strftime_at(dt, "%Q").is_err());

  // Through the jinja path: `{{ strftime_now('%z') }}` renders "" (no panic).
  assert_eq!(
    render_gp("[{{ strftime_now('%z') }}]", &json!([]), false, None),
    "[]"
  );
  // Unknown directive surfaces as a recoverable render error (no abort).
  let r = render_jinja(
    "{{ strftime_now('%Q') }}",
    &json!([]),
    None,
    false, // add_generation_prompt
    false, // continue_final_message
    None,
    None,
    false,
    &json!({}),
  );
  assert!(
    r.is_err(),
    "unknown strftime directive must be a recoverable error"
  );
}

/// A chat template embedding `{{ strftime_now('%Y-%m-%d') }}` (Llama-3.x
/// system-prompt shape) must render a non-empty `YYYY-MM-DD` date, not "".
#[test]
fn strftime_now_in_chat_template_renders_real_date() {
  let template = "Today Date: {{ strftime_now('%Y-%m-%d') }}";
  let out = render_gp(template, &json!([]), false, None);
  let date = out.strip_prefix("Today Date: ").expect("prefix");
  assert_ne!(date, "", "strftime_now must not return empty string");
  // Shape check: `YYYY-MM-DD`, all-numeric components.
  let parts: Vec<&str> = date.split('-').collect();
  assert_eq!(parts.len(), 3, "expected YYYY-MM-DD, got {date:?}");
  assert_eq!(parts[0].len(), 4, "4-digit year, got {date:?}");
  assert_eq!(parts[1].len(), 2, "2-digit month, got {date:?}");
  assert_eq!(parts[2].len(), 2, "2-digit day, got {date:?}");
  assert!(
    date.chars().all(|c| c.is_ascii_digit() || c == '-'),
    "date must be numeric-only, got {date:?}"
  );
}
