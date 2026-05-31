//! Chat-template rendering.
//!
//! Renders a model's `chat_template` (from `tokenizer_config.json`) via
//! `minijinja`, registering the jinja extensions/globals `transformers`
//! exposes so real Llama/Qwen/Mistral templates render byte-identically:
//! `raise_exception`, `tojson`, `strftime_now`, plus `bos_token` /
//! `eos_token` / `add_generation_prompt` context.
//!
//! Also ports the Python `mlx_lm/chat_templates/` override registry and the
//! one concrete override shipped there (`deepseek_v32`). Mirrors the Python
//! `TokenizerWrapper.apply_chat_template` dispatch: a registered override
//! wins over the model's jinja `chat_template`.

use minijinja::{Environment, Error as JErr, ErrorKind, Value as JValue, value::Kwargs};
use serde_json::Value;

use crate::Error;

/// `serde_json::ser::Formatter` reproducing Python `json.dumps`'s *compact*
/// separators (`separators=None, indent=None` → item `", "`, key `": "`),
/// which differ from `serde_json`'s default (`","` / `":"`). HF/transformers
/// `_compile_jinja_template` uses `json.dumps(x, ensure_ascii=False)`, so the
/// `{{ x | tojson }}` (no-indent) path must emit `, ` / `: ` to be
/// byte-identical to the Python reference. The indented path instead uses
/// `serde_json::ser::PrettyFormatter`, whose `,` / `: ` separators already
/// match Python's `json.dumps(indent=N)`.
struct PyCompactFormatter;

impl serde_json::ser::Formatter for PyCompactFormatter {
  fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
  where
    W: ?Sized + std::io::Write,
  {
    if first {
      Ok(())
    } else {
      writer.write_all(b", ")
    }
  }

  fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
  where
    W: ?Sized + std::io::Write,
  {
    if first {
      Ok(())
    } else {
      writer.write_all(b", ")
    }
  }

  fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
  where
    W: ?Sized + std::io::Write,
  {
    writer.write_all(b": ")
  }
}

/// Serialize `v` exactly like HF/transformers' `tojson` filter:
/// `json.dumps(v, ensure_ascii=False, indent=indent)`.
///
/// * `ensure_ascii=False`: non-ASCII is left verbatim and (unlike minijinja's
///   built-in `tojson`) `<`/`>`/`&`/`'` are NOT HTML-escaped — `serde_json`
///   already satisfies this.
/// * `indent=None` (the default / absent): compact with Python's `", "` /
///   `": "` separators via [`PyCompactFormatter`].
/// * `indent=Some(pad)`: pretty-print using `pad` as the literal per-level
///   indent string via `serde_json::ser::PrettyFormatter` (separators `,` /
///   `: `, matching Python's indented `json.dumps`). `pad` is the already-
///   normalized indent bytes — Python `json.dumps` accepts `indent` as
///   `int` (→ that many spaces; 0/negative → empty), `bool` (`True`≡1,
///   `False`≡0), or `str` (used verbatim, e.g. `"\t"`/`"--"`/`""`);
///   [`coerce_indent`] normalizes all of those to the byte string here, so
///   an empty `pad` yields Python's newline-per-element/no-indent form.
///
/// # Known limitation — float spelling
///
/// `serde_json`'s number formatter is not byte-identical to CPython's
/// `repr(float)`/`json.dumps` for some floats: notably the exponent form
/// (`1e-6` vs CPython `1e-06`, which zero-pads the exponent to ≥2 digits)
/// and the exponent thresholds. Strings, integers, booleans, `null`, and
/// nested objects/arrays — i.e. everything real HF chat templates serialize
/// through `tojson` (tool JSON-Schemas and message content) — are
/// byte-exact. Real model `chat_template`s do not emit exponent-threshold
/// float literals, so this does not affect produced prompts in practice.
/// Reproducing CPython's shortest-round-trip float repr exactly is an
/// unbounded compatibility surface deliberately left out of scope (same
/// discipline as not hardening minijinja's core lexer); documented and
/// tracked rather than chased.
fn py_json_dumps<S: serde::Serialize>(
  v: &S,
  indent: Option<&[u8]>,
) -> Result<String, serde_json::Error> {
  match indent {
    None => {
      let mut buf = Vec::new();
      let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyCompactFormatter);
      v.serialize(&mut ser)?;
      // PyCompactFormatter only writes ASCII separators; the rest is
      // serde_json's UTF-8 string escaping, so this is always valid UTF-8.
      Ok(String::from_utf8(buf).expect("serde_json emits UTF-8"))
    }
    Some(pad) => {
      let mut buf = Vec::new();
      let fmt = serde_json::ser::PrettyFormatter::with_indent(pad);
      let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
      v.serialize(&mut ser)?;
      Ok(String::from_utf8(buf).expect("serde_json emits UTF-8"))
    }
  }
}

/// Interpret the `indent` argument the way HF/transformers' `tojson` does —
/// it is forwarded straight to Python `json.dumps(..., indent=indent)`, which
/// accepts the full domain: `None` (absent → compact, the default), `int`
/// (that many spaces; `0`/negative → empty per-level indent → newline-per-
/// element), `bool` (a subclass of `int`: `True`≡`1`, `False`≡`0`, both NOT
/// `None`), or `str` (used verbatim as the per-level indent string, e.g.
/// `"\t"`, `"--"`, `""`). All non-`None` forms are normalized here to the
/// literal indent byte string passed to [`py_json_dumps`]. HF tool-schema
/// templates normally pass an explicit int such as `indent=4`. Verified
/// byte-for-byte against CPython `json.dumps(..., ensure_ascii=False)`.
fn coerce_indent(val: &JValue) -> Result<Option<Vec<u8>>, JErr> {
  if val.is_none() || val.is_undefined() {
    return Ok(None);
  }
  if let Ok(b) = bool::try_from(val.clone()) {
    // Python `json.dumps(indent=<bool>)`: `True` ≡ int 1, `False` ≡ int 0
    // (NOT None) — both take the indented path. Verified vs CPython.
    return Ok(Some(if b { b" ".to_vec() } else { Vec::new() }));
  }
  if let Some(s) = val.as_str() {
    // Python `json.dumps(indent="<s>")`: `s` is the literal per-level indent
    // string (`""` → newline/no-indent, like `indent=0`). Verified vs CPython.
    return Ok(Some(s.as_bytes().to_vec()));
  }
  let n = i64::try_from(val.clone()).map_err(|_| {
    JErr::new(
      ErrorKind::InvalidOperation,
      "tojson: `indent` must be an integer, boolean, or string",
    )
  })?;
  // `int` → that many spaces; `0`/negative → empty (newline, no indent).
  //
  // A `chat_template` from `tokenizer_config.json` is UNTRUSTED (downloaded
  // models). An enormous integer indent (e.g. `i64::MAX`) would build a
  // multi-exabyte `Vec` → allocation failure → process abort, letting a
  // hostile/corrupt config kill the host just by rendering a prompt.
  // CPython fails *recoverably* (MemoryError); return a recoverable `JErr`
  // instead. No legitimate template uses an indent beyond a handful — the
  // cap is far above any real value and bounds the allocation to ≤1 KiB.
  const MAX_INDENT: i64 = 1024;
  if n > MAX_INDENT {
    return Err(JErr::new(
      ErrorKind::InvalidOperation,
      "tojson: `indent` too large (max 1024)",
    ));
  }
  Ok(Some(vec![b' '; n.max(0) as usize]))
}

/// A registered Rust chat-template override (Python
/// `mlx_lm.chat_templates.<type>.apply_chat_template`). Selected by the
/// `chat_template_type` key in `tokenizer_config.json`.
///
/// The override registry is gated on `tokenizer-deepseek-v32`; jinja
/// `chat_template` rendering (`tokenizer-chat`) is always available.
#[cfg(feature = "tokenizer-deepseek-v32")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-deepseek-v32")))]
pub trait ChatTemplateOverride: Send + Sync {
  /// Render messages to a prompt string.
  fn apply(
    &self,
    messages: &[Value],
    tools: Option<&Value>,
    add_generation_prompt: bool,
    continue_final_message: bool,
    enable_thinking: bool,
  ) -> Result<String, Error>;
}

/// Look up a chat-template override by its `chat_template_type` name. Only
/// `deepseek_v32` is shipped by `mlx-lm`; unknown names return `None`.
/// (`tokenizer-deepseek-v32` only.)
#[cfg(feature = "tokenizer-deepseek-v32")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-deepseek-v32")))]
pub fn override_by_name(name: &str) -> Option<Box<dyn ChatTemplateOverride>> {
  match name {
    "deepseek_v32" => Some(Box::new(DeepseekV32)),
    _ => None,
  }
}

/// Rewrite Transformers' `{% generation %}` / `{% endgeneration %}` block
/// tags into a semantically transparent, no-output minijinja block so HF
/// templates that use them can render.
///
/// ## Correctness invariant (the exact contract)
///
/// `strip_generation_tags` rewrites a byte span if and only if Jinja2 (with
/// `trim_blocks`+`lstrip_blocks`, AssistantTracker active) would parse that
/// exact span as a top-level `{% generation %}` / `{% endgeneration %}`
/// *statement tag*. Every other byte — text, `{{ expression }}`,
/// `{# comment #}`, `{% raw %}…{% endraw %}` bodies, string-literal contents
/// inside any tag, and all non-target `{% … %}` tags — is copied through
/// byte-for-byte verbatim.
///
/// ## Why this is needed
///
/// Transformers' `_compile_jinja_template` installs an `AssistantTracker`
/// jinja *extension* that registers a custom `generation` tag (parsed as a
/// `jinja2.nodes.CallBlock`). Its sole purpose is to compute
/// `return_assistant_tokens_mask` — a *training / data-prep* token-offset
/// feature. For an inference prompt-building port that mask is out of scope
/// (same discipline as not porting per-model architectures), and the block is
/// **semantically transparent**: jinja2 renders the block *body* normally and
/// the `{% generation %}` / `{% endgeneration %}` tags themselves emit
/// nothing. minijinja has no custom-tag/extension API, so without this pass
/// `add_template` would reject every HF template that uses these tags.
///
/// ## Why substitute `{% if true %}` rather than delete + re-implement
/// whitespace
///
/// A `CallBlock` is, for whitespace purposes, just a regular jinja2 block
/// tag: with `trim_blocks=true` + `lstrip_blocks=true` (which Transformers
/// always sets and which we mirror) jinja2's *lexer* strips the leading
/// indentation before `{%` and the single newline after `%}` purely from the
/// tag delimiters, independent of *which* tag it is. minijinja's lexer
/// applies byte-identical trim/lstrip rules to a regular no-output block. So
/// rather than hand-roll a whitespace replicator, we keep the `{%` / `%}`
/// delimiters and any whitespace-control dashes *exactly* in place and only
/// swap the keyword:
///
/// * `{% generation %}`    -> `{% if true %}`   (body renders unconditionally)
/// * `{% endgeneration %}` -> `{% endif %}`
///
/// `{% if true %}...{% endif %}` renders its body unconditionally and emits no
/// tag output — exactly the `CallBlock` semantics — and minijinja then does
/// all the trim_blocks/lstrip_blocks work itself, matching jinja2. This was
/// verified against the real Transformers reference across every
/// whitespace-control variant (own-line/indented, inline, `{%- ... -%}`,
/// left-only `{%- ... %}`, right-only `{% ... -%}`, no-space `{%generation%}`
/// / `{%-generation-%}`, and arbitrary internal whitespace): minijinja's
/// output is byte-identical to jinja2+`AssistantTracker` (which renders the
/// body and emits nothing for the tags) in all of them.
///
/// Two known divergences exist, *both* pre-existing minijinja-vs-jinja2 core
/// lexer differences independent of this rewrite (feeding the byte-identical
/// post-rewrite template to both engines reproduces each), so hardening
/// minijinja's core lexer is out of scope — documented, not worked around:
///
/// 1. *Raw CRLF immediately after the tag*: jinja2's `trim_blocks`
///    normalizes a `\r\n` after `%}` to drop the whole `\r\n`, whereas
///    minijinja drops only the `\n` and keeps the `\r`. Affects *every*
///    block tag, not specific to `generation`.
/// 2. A specific *multi-line* shape — a nested block close
///    (`{% endif %}\n{% endfor %}`) immediately followed by `\n{% raw %}` on
///    its own line — makes minijinja's `trim_blocks` drop one `\n` that
///    jinja2 keeps (one extra trimmed newline before the literal raw body).
///    The much commoner single-line / text-then-`{% raw %}` /
///    block-then-`{% raw %}` layouts render byte-identically in both
///    engines.
///
/// HF chat templates are JSON strings with `\n` line endings and almost
/// universally single-logical-line, so neither divergence arises for real
/// templates in practice; both are documented for completeness.
///
/// ## Grammar
///
/// jinja2 3.1.6's generic `block_begin` lexer regex is
/// `\{%(\-|\+|)\s*…\s*(\-|\+|)%\}`: after `{%` (and before `%}`) it accepts
/// **one optional whitespace-control marker that is `-` *or* `+` or none** —
/// not only `-`. `-` strips adjacent whitespace; `+` *explicitly keeps* it,
/// overriding the environment's `trim_blocks`/`lstrip_blocks` (both of which
/// Transformers always sets and which this port mirrors); none = the
/// environment default. Transformers parses `generation` with the standard
/// jinja2 parser, so `{%+ generation %}`, `{% generation +%}`, `{%- … +%}`
/// mixes, etc. are all valid. We match the same: the inner text of a
/// `{% ... %}` block, after stripping one optional leading and one optional
/// trailing `-`/`+`/none marker and surrounding ASCII whitespace (incl.
/// tabs/newlines), must be *exactly* `generation` or `endgeneration`.
/// Internal spacing is irrelevant to jinja2's output (it is trimmed), so we
/// normalize the rewritten tag to a single canonical spacing while preserving
/// the `-`/`+`/none markers **verbatim** (see [`WsCtrl`]) — those markers are
/// the only part of a block tag that influences whitespace, and minijinja
/// 2.19 honours `-`/`+` with byte-identical semantics, so preserving them
/// keeps output byte-identical. No other `{% ... %}` tag is touched.
///
/// ## The complete Jinja delimiter pre-lexer
///
/// The rewrite is a *pre-parse* text pass, so to satisfy the invariant above
/// it must reproduce Jinja2's *delimiter-level lexer* well enough that it can
/// only ever touch real top-level `{% generation %}` / `{% endgeneration %}`
/// statement tags. Rather than special-case the few constructs that happen to
/// embed the target keyword, this is a single forward pass that recognizes
/// **all three** Jinja delimiter constructs and copies the non-target ones
/// through *opaquely as whole units* before it ever inspects a keyword:
///
/// Jinja's left delimiter is disambiguated purely by the **second byte** after
/// a `{`: `{{` → expression, `{%` → statement, `{#` → comment. A lone `{`
/// (EOF, or `{x` for any other `x`) is literal text. We match that exactly.
///
/// * **`{# … #}` comment** — Jinja comments are *opaque and non-nesting*:
///   jinja2's lexer scans for the first literal `#}` with **no** string-,
///   `{{`-, `{%`-, `{# `-nesting awareness whatsoever (verified: `{# a {# b
///   #} c #}` closes at the first `#}`; `{# "unclosed quote {% generation %}
///   #}` closes at its `#}`; `{# has %} and }} #}` is fully consumed).
///   minijinja 2.19's comment lexer is byte-identical here (proven by the
///   regression tests rendering identical templates through both engines and
///   asserting equality with the captured jinja2 reference). We therefore copy
///   verbatim through the first `#}`. An unterminated `{#` is copied to EOF so
///   minijinja emits the same "unexpected end of comment" parse error jinja2
///   does — we never silently "fix" it.
/// * **`{{ … }}` expression** — copied verbatim through the matching `}}`.
///   jinja2's expression lexer *is* string-literal-aware (single/double quoted
///   with a `\`-escape), so a `}}` inside a string literal does **not** close
///   the expression (verified: `{{ "}}" }}` renders `}}`; `{{ "{% generation
///   %}" }}` renders the literal text — Transformers emits it verbatim and it
///   must NOT be rewritten). We track the same quote/`\`-escape state (shared
///   with the statement-tag scanner) so the literal `{% generation %}`,
///   `{# … #}`, etc. inside an expression string are never mistaken for tags.
///   An unterminated `{{` is copied to EOF (same parse error in both engines).
/// * **`{% … %}` statement tag** — its real `%}` is found with the same
///   string-literal-aware scan (a `%}` inside a tag string does not close it;
///   jinja2 permits that). The tag is then classified by its FULL trimmed
///   delimiter-to-delimiter inner content, after stripping one optional
///   leading and one optional trailing `-`/`+`/none whitespace-control
///   marker and ASCII whitespace ([`classify_tag`]):
///   * exactly `raw` → enter a **verbatim raw region**. jinja2 uses a
///     *dedicated* `raw_begin` regex stricter than the generic block regex:
///     it accepts `-`/`+`/none on the open but the tail is **only** `-%}` or
///     plain `%}` (a `+%}` tail makes jinja2 reject the template with
///     "Encountered unknown tag 'raw'."), so a `{% … raw +%}` is *not* a
///     raw-open and stays opaque. Once inside raw, ALL interior bytes are
///     opaque raw text — [`find_endraw`] is a pure text scan that does NOT
///     parse interior `{%`/`{{`/`{#`/strings and stops at the FIRST endraw
///     delimiter (jinja2 raw does not nest — the first `endraw` closes it; a
///     later `{% endraw %}` is then ordinary post-raw text). The whole raw
///     region (open delimiter + body + close delimiter, with their exact
///     `-`/`+`/none markers) is copied byte-for-byte verbatim (never
///     rewritten). Unterminated `{% raw %}` (no endraw delimiter) → copied to
///     EOF (same "Missing end of raw directive" / "unexpected end of raw
///     block" error in both engines).
///   * exactly `generation` / `endgeneration` → the rewrite (below).
///   * anything else → the whole tag copied through verbatim, so a quoted
///     `%}` or an embedded `generation` substring can never be corrupted.
///   * an unterminated `{%` is copied to EOF (same parse error in both).
/// * **Plain text between constructs** — copied verbatim byte-for-byte.
///
/// Because the comment / expression / non-target-statement / raw-body bytes
/// are all emitted *before* any keyword lookup, the only spans this pass can
/// ever rewrite are exactly top-level `{% generation %}` / `{% endgeneration
/// %}` statement tags — precisely the stated invariant. The pass is a single
/// linear scan with no regex/backtracking.
///
/// ### The `generation` → `if true` substitution
///
/// `{% generation %}` → `{% if true %}` and `{% endgeneration %}` →
/// `{% endif %}`, preserving the `{%` / `%}` delimiters and the exact
/// leading/trailing whitespace-control markers — `-` (strip), `+` (keep), or
/// none — *byte-identically* (see the
/// whitespace discussion and [`WsCtrl`]). The `-`/`+`/none marker on each
/// side is the only part of a block tag that influences whitespace under
/// `trim_blocks`+`lstrip_blocks`; minijinja 2.19 implements `-`/`+` with
/// byte-identical semantics to jinja2 3.1.6, so preserving each side's marker
/// verbatim keeps minijinja's output byte-identical to jinja2's, while the
/// canonical single-space interior is irrelevant (jinja2 trims all internal
/// tag whitespace anyway).
///
/// ### Residual limitation (precise, and why it cannot diverge)
///
/// The expression/statement string-literal tracker keys off the quote chars
/// `'` / `"` and a `\`-escape only (jinja2's tag/expression lexer is itself
/// regex-based and treats `\` as a literal char — Python string escapes are
/// not applied at the lexer level). For the sole decision this pass makes —
/// "is this *entire* trimmed statement-tag body exactly
/// `generation`/`endgeneration`?" — that is exact: a target tag contains no
/// quote at all, and any tag containing a quote is by definition not the bare
/// keyword and is passed through untouched regardless of how its quotes nest.
/// The only theoretical residual is a *non-target* construct whose string
/// literal holds an unbalanced lone quote spilling into another construct;
/// such a template is already a jinja2 *syntax error* (both engines reject it
/// before any rendering), so it cannot produce divergent prompt bytes for any
/// input both engines accept. No correctness gap remains for valid templates.
fn strip_generation_tags(template: &str) -> std::borrow::Cow<'_, str> {
  // Fast path: skip allocation/scan entirely when the keyword is absent.
  // (`raw` blocks / comments / expressions without `generation` need no
  // special handling — minijinja parses them natively, identically to
  // jinja2.)
  if !template.contains("generation") {
    return std::borrow::Cow::Borrowed(template);
  }
  let bytes = template.as_bytes();
  let mut out = String::with_capacity(template.len());
  let mut i = 0;
  let mut rewrote = false;
  while i < bytes.len() {
    if bytes[i] == b'{' && i + 1 < bytes.len() {
      // Jinja disambiguates the left delimiter purely by the 2nd byte.
      match bytes[i + 1] {
        b'#' => {
          // Comment: opaque, non-nesting, NOT string-aware. Copy verbatim
          // through the first literal `#}`. Unterminated → copy to EOF so
          // minijinja reports the same parse error jinja2 does.
          let end = find_comment_close(template, i + 2);
          out.push_str(&template[i..end]);
          i = end;
          continue;
        }
        b'{' => {
          // Expression: string-literal-aware. Copy verbatim through the
          // matching `}}`. Unterminated → copy to EOF (same error in both).
          let end = find_expr_close(template, i + 2);
          out.push_str(&template[i..end]);
          i = end;
          continue;
        }
        b'%' => {
          if let Some(close) = find_tag_close(template, i + 2) {
            // `close` is the byte index of the `%` in this tag's `%}`.
            let inner = &template[i + 2..close];
            let (kw, lead, trail) = classify_tag(inner);
            if kw == TagKw::Raw {
              // The whole raw region — its `{% raw %}` open delimiter, body,
              // and `{% endraw %}` close delimiter (with their exact
              // `-`/`+`/none ws-control markers) — is copied byte-for-byte
              // verbatim. `find_endraw` is a pure raw-mode text scan from
              // just past this open delimiter: it does NOT parse interior
              // `{%`/`{{`/`{#`/strings (jinja2 raw ignores all of them) and
              // stops at the FIRST literal endraw delimiter (raw does not
              // nest). No endraw delimiter → copy to EOF (same
              // "Missing end of raw directive" error in both engines).
              let raw_end = find_endraw(template, close + 2);
              out.push_str(&template[i..raw_end]);
              i = raw_end;
              continue;
            }
            let repl = match kw {
              TagKw::Generation => Some("if true"),
              TagKw::EndGeneration => Some("endif"),
              _ => None,
            };
            if let Some(repl) = repl {
              rewrote = true;
              out.push_str("{%");
              // Preserve the leading whitespace-control marker verbatim
              // (`-` strip / `+` keep / none default): it (and only it)
              // governs lstrip/whitespace stripping on the left.
              out.push_str(lead.open_str());
              out.push_str(repl);
              // Likewise the trailing marker governs the right side /
              // trim_blocks.
              out.push_str(trail.close_str());
              i = close + 2;
              continue;
            }
            // Non-target tag: pass the WHOLE tag through opaquely so a
            // quoted `%}` or an embedded `generation` substring cannot be
            // corrupted, then continue scanning after its real `%}`.
            out.push_str(&template[i..close + 2]);
            i = close + 2;
            continue;
          }
          // No closing `%}` at all — copy the rest verbatim; minijinja
          // reports the same unterminated-tag parse error jinja2 does.
          out.push_str(&template[i..]);
          break;
        }
        // `{` followed by any other byte is literal text — fall through to
        // the verbatim copy below (copy just the `{`, then continue).
        _ => {}
      }
    }
    // Default: copy this char through unchanged. (Indexing by byte is safe
    // because `{`/`%`/`#` are ASCII; multi-byte UTF-8 continuation bytes are
    // never these and are copied verbatim here.)
    let ch_len = utf8_char_len(bytes[i]);
    out.push_str(&template[i..i + ch_len]);
    i += ch_len;
  }
  if rewrote {
    std::borrow::Cow::Owned(out)
  } else {
    // Keyword present only as substring (e.g. inside text / a `{{ }}`
    // expression / a `{# #}` comment / a `{% raw %}` block / a non-target
    // tag); avoid the needless allocation.
    std::borrow::Cow::Borrowed(template)
  }
}

/// Scan from byte `start` (just past `{#`) to the byte index just past the
/// first literal `#}`, i.e. the end of a Jinja comment. Jinja2/minijinja
/// comments are *opaque and non-nesting*: the lexer scans for the first `#}`
/// with no string-, `{{`-, `{%`-, or nested-`{#`-awareness (verified against
/// transformers 5.8.1 / jinja2 3.1.6 and minijinja 2.19 via the regression
/// suite). If no `#}` is found the comment is unterminated; return the
/// template length so the remainder is copied verbatim and minijinja reports
/// the same parse error jinja2 does.
fn find_comment_close(template: &str, start: usize) -> usize {
  let b = template.as_bytes();
  let mut i = start;
  while i + 1 < b.len() {
    if b[i] == b'#' && b[i + 1] == b'}' {
      return i + 2;
    }
    i += 1;
  }
  b.len()
}

/// Scan from byte `start` (just past `{{`) to the byte index just past the
/// matching `}}`, i.e. the end of a Jinja expression. jinja2's expression
/// lexer *is* string-literal-aware (single/double quoted with a `\`-escape),
/// so a `}}` inside a string literal does not close the expression (verified:
/// `{{ "}}" }}` → `}}`). If no closing `}}` is found the expression is
/// unterminated; return the template length so the remainder is copied
/// verbatim and minijinja reports the same parse error jinja2 does.
fn find_expr_close(template: &str, start: usize) -> usize {
  let b = template.as_bytes();
  let mut i = start;
  let mut quote: Option<u8> = None;
  while i < b.len() {
    match quote {
      Some(q) => {
        if b[i] == b'\\' && i + 1 < b.len() {
          // jinja2's lexer keeps the backslash literal; we only need it not
          // to let an escaped quote toggle string state.
          i += 2;
          continue;
        }
        if b[i] == q {
          quote = None;
        }
      }
      None => {
        if b[i] == b'\'' || b[i] == b'"' {
          quote = Some(b[i]);
        } else if b[i] == b'}' && i + 1 < b.len() && b[i + 1] == b'}' {
          return i + 2;
        }
      }
    }
    i += 1;
  }
  b.len()
}

/// One side's Jinja whitespace-control marker. Jinja2 3.1.6 (and minijinja
/// 2.19) recognize **two** explicit markers, not just `-`:
///
/// * `-` (`Strip`) — trim adjacent whitespace on that side.
/// * `+` (`Keep`) — explicitly DO NOT trim, *overriding* the environment's
///   `trim_blocks` / `lstrip_blocks` (both of which Transformers always sets
///   and which this port mirrors).
/// * absent (`None`) — environment default (`trim_blocks`/`lstrip_blocks`).
///
/// Modelling `+` explicitly (rather than the old "dash present?" boolean) is
/// load-bearing for the invariant: the AssistantTracker `generation` tag is a
/// standard jinja2 block tag, so `{%+ generation %}` / `{% generation +%}` are
/// valid and Transformers' standard jinja2 parser accepts them. Treating them
/// as opaque (the old code's behaviour) would feed minijinja an unknown
/// `generation` tag (template rejected) or fail to enter a `{%+ raw %}` block
/// (interior `generation`-looking text wrongly rewritten). Preserving the
/// exact `-`/`+`/none marker verbatim on the rewritten `if true`/`endif` makes
/// minijinja 2.19 (which honours `-`/`+` with byte-identical semantics)
/// reproduce jinja2's whitespace bytes exactly.
#[derive(PartialEq, Eq, Clone, Copy)]
enum WsCtrl {
  /// No marker — environment default (`trim_blocks`/`lstrip_blocks`).
  None,
  /// `-` — strip adjacent whitespace.
  Strip,
  /// `+` — explicitly keep adjacent whitespace (overrides trim/lstrip).
  Keep,
}

impl WsCtrl {
  /// Re-emit this marker exactly as it appeared, in the rewritten tag's
  /// leading position (just after `{%`): `{%-` / `{%+` / `{%`.
  fn open_str(self) -> &'static str {
    match self {
      WsCtrl::None => " ",
      WsCtrl::Strip => "- ",
      WsCtrl::Keep => "+ ",
    }
  }

  /// Re-emit this marker in the rewritten tag's trailing position (just
  /// before `%}`): `-%}` / `+%}` / `%}`.
  fn close_str(self) -> &'static str {
    match self {
      WsCtrl::None => " %}",
      WsCtrl::Strip => " -%}",
      WsCtrl::Keep => " +%}",
    }
  }
}

/// The classified identity of a `{% … %}` tag for the generation rewrite.
#[derive(PartialEq, Eq, Clone, Copy)]
enum TagKw {
  /// Exactly `generation` (after ws-control/whitespace strip) — rewrite to
  /// `if true`.
  Generation,
  /// Exactly `endgeneration` — rewrite to `endif`.
  EndGeneration,
  /// Exactly `raw` — opens a verbatim block ending at the first `endraw`.
  Raw,
  /// Anything else — opaque, passed through untouched.
  Other,
}

/// Split one optional leading whitespace-control marker (`-` or `+`) off the
/// front of a tag body, then ASCII whitespace, returning the marker and the
/// remaining text. Mirrors jinja2's `block_begin` regex `\{%(\-|\+|)\s*`,
/// generalizing the leading-`-` handling to `-`/`+`/none with
/// byte-identical structure (trim, then strip one marker, then trim) so every
/// already-covered case is preserved verbatim — only the `+` arm is new.
fn split_lead_ws_ctrl(inner: &str) -> (WsCtrl, &str) {
  let t = inner.trim_start();
  if let Some(rest) = t.strip_prefix('-') {
    (WsCtrl::Strip, rest.trim_start())
  } else if let Some(rest) = t.strip_prefix('+') {
    (WsCtrl::Keep, rest.trim_start())
  } else {
    (WsCtrl::None, t)
  }
}

/// Split one optional trailing whitespace-control marker (`-` or `+`) off the
/// end of a tag body, after trimming trailing ASCII whitespace, returning the
/// remaining text and the marker. Mirrors jinja2's tail `\s*(\-|\+|)%\}`,
/// generalizing the trailing-`-` handling to `-`/`+`/none with
/// byte-identical structure (trim, then strip one marker) so every
/// already-covered case is preserved verbatim — only the `+` arm is new.
fn split_trail_ws_ctrl(inner: &str) -> (&str, WsCtrl) {
  let t = inner.trim_end();
  if let Some(rest) = t.strip_suffix('-') {
    (rest.trim_end(), WsCtrl::Strip)
  } else if let Some(rest) = t.strip_suffix('+') {
    (rest.trim_end(), WsCtrl::Keep)
  } else {
    (t, WsCtrl::None)
  }
}

/// Classify a `{% … %}` tag (delimiter-to-delimiter inner text, exclusive of
/// `{%`/`%}`) by its trimmed keyword, *after* stripping one optional leading
/// and one optional trailing whitespace-control marker (`-` **or** `+`) and
/// surrounding ASCII whitespace, returning the keyword and both markers.
///
/// jinja2's generic `block_begin` regex accepts `-`, `+`, or none on *both*
/// the open (`{%[-+]?`) and the tail (`[-+]?%}`) for normal statement tags
/// such as `generation` / `endgeneration`. Returns [`TagKw::Other`] for any
/// non-target keyword (passed through opaque).
///
/// `raw` is special-cased here to match jinja2's *dedicated* `raw_begin`
/// regex, which is **stricter** than the generic block regex on the tail:
/// `\{%(\-|\+|)\s*raw\s*(?:\-%\}\s*|%\})` — it accepts `-` or `+` or none on
/// the open but the tail is **only** `-%}` or plain `%}`; a `+%}` tail makes
/// jinja2 reject the template with "Encountered unknown tag 'raw'." So a
/// `{% … raw +%}` is *not* a raw-open in jinja2 and is classified
/// [`TagKw::Other`] (opaque) here, never entering verbatim raw mode.
fn classify_tag(inner: &str) -> (TagKw, WsCtrl, WsCtrl) {
  let (lead, body) = split_lead_ws_ctrl(inner);
  let (kw, trail) = split_trail_ws_ctrl(body);
  let tag = match kw.trim() {
    "generation" => TagKw::Generation,
    "endgeneration" => TagKw::EndGeneration,
    // jinja2's `raw_begin` regex rejects a `+%}` tail (treats it as an
    // unknown `raw` tag), so only a `-` or no trailing marker opens raw.
    "raw" if trail != WsCtrl::Keep => TagKw::Raw,
    _ => TagKw::Other,
  };
  (tag, lead, trail)
}

/// Find the closing `%}` of the `{% … %}` tag whose body starts at byte
/// `start`, returning the byte index of the `%` in `%}`. Jinja string
/// literals (`'…'` / `"…"`, with `\`-escape) are tracked so a `%}` *inside*
/// a string does not falsely close the tag (jinja2 permits `%}` in tag
/// strings). Returns `None` if the tag is unterminated.
fn find_tag_close(template: &str, start: usize) -> Option<usize> {
  let b = template.as_bytes();
  let mut i = start;
  let mut quote: Option<u8> = None;
  while i < b.len() {
    match quote {
      Some(q) => {
        if b[i] == b'\\' && i + 1 < b.len() {
          // jinja2's tag lexer keeps the backslash literal; we only need to
          // not let an escaped quote toggle string state.
          i += 2;
          continue;
        }
        if b[i] == q {
          quote = None;
        }
      }
      None => {
        if b[i] == b'\'' || b[i] == b'"' {
          quote = Some(b[i]);
        } else if b[i] == b'%' && i + 1 < b.len() && b[i + 1] == b'}' {
          return Some(i);
        }
      }
    }
    i += 1;
  }
  None
}

/// Match the literal `endraw` *delimiter* starting at byte `i` (which must be
/// the `{` of a candidate `{%`). Returns the byte index just past the closing
/// `%}` if the bytes there are exactly an endraw delimiter, else `None`.
///
/// This mirrors jinja2 3.1.6's `raw_begin`-state regex for the end of a raw
/// block, byte-for-byte:
///
/// ```text
/// (?:\{%(\-|\+|))\s*endraw\s*(?:\+%\}|\-%\}\s*|%\}\n?)
/// ```
///
/// i.e. `{%`, then **one** optional `-` or `+`, then ASCII whitespace, then
/// the literal `endraw`, then ASCII whitespace, then a close that is `+%}`
/// **or** `-%}` **or** `%}`. (The regex's trailing `\s*` after `-%}` and
/// `\n?` after `%}` are jinja2's *post*-match `trim_blocks` whitespace
/// handling, applied to the bytes *after* the delimiter — not part of the
/// delimiter itself; they are reproduced identically by minijinja 2.19's own
/// `handle_raw_tag` on the verbatim-copied close, so this returns the index
/// just past `%}` and lets minijinja apply trim_blocks to the same bytes.)
///
/// Crucially this is a pure *text* match: it does NOT track string literals
/// or skip interior `{%`/`{{`/`{#`. Inside a jinja2 `{% raw %}` block ALL
/// interior syntax is ignored — the lexer scans raw text for the first
/// endraw delimiter — so an interior `{% foo`, `{{`, `{#`, or lone quote is
/// literal raw body, not a tag to parse past.
fn match_endraw_delim(template: &str, i: usize) -> Option<usize> {
  let b = template.as_bytes();
  // `{%`
  if i + 1 >= b.len() || b[i] != b'{' || b[i + 1] != b'%' {
    return None;
  }
  let mut p = i + 2;
  // optional ONE leading ws-control marker: `-` or `+`
  if p < b.len() && (b[p] == b'-' || b[p] == b'+') {
    p += 1;
  }
  // `\s*` (ASCII whitespace)
  while p < b.len() && b[p].is_ascii_whitespace() {
    p += 1;
  }
  // literal `endraw`
  if !template[p..].starts_with("endraw") {
    return None;
  }
  p += "endraw".len();
  // `\s*` (ASCII whitespace)
  while p < b.len() && b[p].is_ascii_whitespace() {
    p += 1;
  }
  // close: `+%}` | `-%}` | `%}`
  if p + 1 < b.len() && (b[p] == b'-' || b[p] == b'+') && b[p + 1] == b'%' {
    if p + 2 < b.len() && b[p + 2] == b'}' {
      return Some(p + 3);
    }
    return None;
  }
  if p + 1 < b.len() && b[p] == b'%' && b[p + 1] == b'}' {
    return Some(p + 2);
  }
  None
}

/// Raw-mode text scan from byte `start` (just past the `{% raw %}` *open*
/// delimiter's `%}`) to the byte index just past the FIRST `{% endraw %}`
/// *close* delimiter, i.e. the end of the verbatim raw region.
///
/// Per jinja2 3.1.6 (`raw_begin` lexer state), once inside `{% raw %}` every
/// interior byte is opaque raw text: the lexer performs a pure text scan for
/// the next endraw delimiter and does **not** parse interior `{%`/`{{`/`{#`
/// or string literals. So this never calls the tag parser — it tests
/// [`match_endraw_delim`] at each `{%` and treats everything else (including
/// an unterminated-looking `{% foo`, an interior `{{`/`{#`, or an unbalanced
/// quote) as literal raw body. Raw blocks do **not** nest: the first endraw
/// delimiter closes the block (a later `{% endraw %}` is then ordinary
/// post-raw text). If no endraw delimiter exists the raw block is
/// unterminated; return the template length so the remainder is copied
/// verbatim and minijinja 2.19 reports the same "unexpected end of raw block"
/// / jinja2 "Missing end of raw directive" parse error.
fn find_endraw(template: &str, start: usize) -> usize {
  let b = template.as_bytes();
  let mut i = start;
  while i < b.len() {
    if b[i] == b'{' && i + 1 < b.len() && b[i + 1] == b'%' {
      if let Some(end) = match_endraw_delim(template, i) {
        return end;
      }
      // Not an endraw delimiter — this `{%` is literal raw text (jinja2's
      // raw lexer does not parse interior tags). Advance one byte and keep
      // scanning the raw body for the real endraw delimiter.
      i += 1;
      continue;
    }
    i += utf8_char_len(b[i]);
  }
  b.len()
}

/// Byte length of the UTF-8 char starting at `b` (1 for ASCII / invalid lead).
fn utf8_char_len(b: u8) -> usize {
  if b < 0x80 {
    1
  } else if b >> 5 == 0b110 {
    2
  } else if b >> 4 == 0b1110 {
    3
  } else if b >> 3 == 0b11110 {
    4
  } else {
    1
  }
}

/// HF's sentinel for the `continue_final_message` post-render trim
/// (`transformers/utils/chat_template_utils.py`,
/// `continue_final_message_tag = "CONTINUE_FINAL_MESSAGE_TAG "`). Appended to
/// the final message's `content` before rendering, then located + truncated
/// out of the rendered string. The trailing space is significant — see
/// [`continue_final_message_trim`] for how the full-tag-with-space check
/// selects HF's plain-truncate vs. `rstrip` branch.
const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

/// Faithful port of HF Transformers' `continue_final_message` handling
/// (`render_jinja_template` in `transformers/utils/chat_template_utils.py`).
///
/// HF's mechanism is a *string-level* post-render trim, not a template flag:
///
/// 1. **Pre-render** — deep-copy the conversation and append the sentinel
///    [`CONTINUE_FINAL_MESSAGE_TAG`] to the final message's `content` (HF:
///    `chat[-1][continue_final_message] = ... + continue_final_message_tag`,
///    with `continue_final_message` defaulting to the `"content"` field).
/// 2. **Render** — the template renders the augmented conversation normally.
/// 3. **Post-render trim** — locate the sentinel and cut everything from it
///    onward, so the rendered prompt ends *exactly* at the final message's
///    content (no trailing end-of-turn / EOS / generation-prompt tokens the
///    template appended after it):
///    ```text
///    tag_loc = rendered.rindex(TAG.trim_end())
///    if rendered[tag_loc .. tag_loc + TAG.len()] == TAG { rendered[..tag_loc] }
///    else                                                { rendered[..tag_loc].trim_end() }
///    ```
///    The `if` branch matches HF byte-for-byte: when the template emits the
///    final-message content verbatim the trailing space of `TAG` survives, so
///    the cut is a plain truncation; when the template applied a transform
///    (e.g. `| trim`) that ate the trailing space, HF `.rstrip()`s the result.
///
/// This function returns the pre-render conversation with the sentinel
/// appended **plus the final message's original `content`** (an `Err` for an
/// empty conversation / a final message lacking a string `content` — HF
/// raises `ValueError` for the same cases); [`continue_final_message_trim`]
/// is the post-render step, which the original content is threaded into so it
/// can validate the content actually rendered (HF's `final_message.strip()
/// not in rendered_chat` guard). They are split so [`render_jinja`] can append
/// the sentinel before building the jinja context and trim after `render`.
fn continue_final_message_mutate(messages: &Value) -> Result<(Value, String), Error> {
  let arr = messages
    .as_array()
    .ok_or_else(|| Error::tokenizer("messages must be a list"))?;
  let last = arr.last().ok_or_else(|| {
    Error::tokenizer(
      "continue_final_message is set but the conversation has no final message to continue",
    )
  })?;
  // HF defaults the continued field to "content" and requires it to exist on
  // the final message (`if (final_message := chat[-1].get(...)) is None:
  // raise ValueError`). mlxrs only supports the default `content` field, and
  // it must be a string (HF's list/tuple content-block form is the
  // multimodal-VLM shape, out of scope for this text-prompt path).
  let content = last.get("content").and_then(Value::as_str).ok_or_else(|| {
    Error::tokenizer(
      "continue_final_message is set but the final message has no string \"content\" to continue",
    )
  })?;
  // Capture the ORIGINAL content (before the sentinel append) so the trim can
  // validate it actually rendered — HF keeps `final_message` bound to the
  // pre-append content for its post-render `final_message.strip() not in
  // rendered_chat` check.
  let original_content = content.to_string();
  let mut mutated = arr.clone();
  let new_content = format!("{content}{CONTINUE_FINAL_MESSAGE_TAG}");
  // `last` is the final element; reborrow it mutably in the clone.
  if let Some(obj) = mutated.last_mut().and_then(Value::as_object_mut) {
    obj.insert("content".to_string(), Value::String(new_content));
  }
  Ok((Value::Array(mutated), original_content))
}

/// Post-render trim for `continue_final_message` — see
/// [`continue_final_message_mutate`].
///
/// Faithful to HF's post-render guard + cut
/// (`transformers/utils/chat_template_utils.py`):
///
/// ```python
/// if (final_message.strip() not in rendered_chat) or (
///     continue_final_message_tag.strip() not in rendered_chat
/// ):
///     raise ValueError("continue_final_message is set but the final message
///         does not appear in the chat after rendering. ...")
/// tag_loc = rendered_chat.rindex(continue_final_message_tag.strip())
/// ```
///
/// `original_content` is the final message's content *before* the sentinel was
/// appended (HF's `final_message`, still bound to the pre-append value at the
/// guard). Both the original content (`.strip()`) AND the sentinel must be
/// present in `rendered` before truncating; if either is missing the template
/// dropped the user's content (or the sentinel), so this returns an `Err`
/// rather than silently producing a prefix for the WRONG prompt — the
/// security-relevant case a template emitting a literal sentinel independent
/// of the user content would otherwise slip through. The empty-content case is
/// implicitly valid: `"".strip()` is `""`, and `rendered.contains("")` is
/// always `true` (mirroring Python's `"" in s`), so an empty original content
/// with the sentinel present yields the prefix up to the sentinel.
fn continue_final_message_trim(rendered: &str, original_content: &str) -> Result<String, Error> {
  // HF guard: BOTH the original final-message content (stripped) AND the
  // sentinel (stripped) must appear in the rendered output, else `ValueError`.
  // `str::trim` matches Python `str.strip` (leading/trailing Unicode
  // whitespace). `contains("")` is `true`, so empty content is always valid.
  let needle = CONTINUE_FINAL_MESSAGE_TAG.trim_end();
  if !rendered.contains(original_content.trim()) || !rendered.contains(needle) {
    return Err(Error::tokenizer(
      "continue_final_message is set but the final message does not appear in the \
       rendered chat (the template dropped the final message's content or the \
       continue sentinel) — refusing to continue a prompt the template did not render",
    ));
  }
  // HF: `rendered_chat.rindex(continue_final_message_tag.strip())`. The guard
  // above already proved the sentinel is present, so this `rfind` succeeds;
  // it stays an `Err` (not an `unwrap`) for defensiveness — never a panic.
  let tag_loc = rendered.rfind(needle).ok_or_else(|| {
    Error::tokenizer(
      "continue_final_message: the rendered template does not contain the continue \
       sentinel",
    )
  })?;
  // HF: if the full tag (with its trailing space) survived verbatim, plain
  // truncate; otherwise the template transformed the trailing whitespace, so
  // `rstrip` the truncation. `get` guards the slice when the tail is shorter
  // than the full tag (sentinel at the very end after a transform).
  let full_tag_present = rendered.get(tag_loc..tag_loc + CONTINUE_FINAL_MESSAGE_TAG.len())
    == Some(CONTINUE_FINAL_MESSAGE_TAG);
  let head = &rendered[..tag_loc];
  Ok(if full_tag_present {
    head.to_string()
  } else {
    head.trim_end().to_string()
  })
}

/// Render a jinja `chat_template` with the HF-compatible environment.
///
/// `messages` / `tools` are JSON values (list of message objects / list of
/// tool objects). `extra` adds arbitrary template variables (the Python
/// `additional_context` / template kwargs). `bos_token` / `eos_token` come
/// from the tokenizer config.
///
/// `continue_final_message` ports HF Transformers' flag of the same name
/// (`render_jinja_template` in `transformers/utils/chat_template_utils.py`):
/// when set, the rendered prompt is trimmed so it ends exactly at the final
/// message's content — the model *continues* that message rather than
/// starting a new turn. HF's mechanism is a string-level post-render trim,
/// faithfully reproduced here: the `"CONTINUE_FINAL_MESSAGE_TAG "` sentinel
/// is appended to the final message's `content` before rendering, then the
/// rendered string is cut at the sentinel (`rendered.rindex(tag.strip())`),
/// dropping the trailing end-of-turn / EOS tokens the template appended after
/// the content. It is mutually exclusive with `add_generation_prompt` (HF
/// rejects both at once); callers must not set both — the dispatching
/// [`super::Tokenizer`] methods reject the combination up front.
#[allow(clippy::too_many_arguments)]
pub fn render_jinja(
  template: &str,
  messages: &Value,
  tools: Option<&Value>,
  add_generation_prompt: bool,
  continue_final_message: bool,
  bos_token: Option<&str>,
  eos_token: Option<&str>,
  enable_thinking: bool,
  extra: &Value,
) -> Result<String, Error> {
  let mut env = Environment::new();
  // pycompat: resolve Python str/list/dict methods (`.strip()`, `.split()`,
  // `.items()`, …) that real transformers chat templates rely on.
  env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
  install_hf_extensions(&mut env);

  // Match transformers' chat-template jinja environment byte-for-byte:
  // `_compile_jinja_template` builds
  // `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, …)`.
  // These strip the per-line whitespace real HF templates assume:
  //  * `trim_blocks`  — drop the first newline right after a block tag, so
  //    `{% for %}` / `{% if %}` on their own line emit no trailing blank line.
  //  * `lstrip_blocks` — strip leading spaces/tabs before a block tag, so an
  //    indented `{% … %}` line produces no spurious indentation.
  // Without these, multi-line templates render with extra blank lines /
  // leading indentation vs. the Python reference. Both must be set BEFORE
  // `add_template`: minijinja snapshots the whitespace config at template
  // load time (later changes only affect future loads). transformers does
  // NOT pass `keep_trailing_newline`, so Jinja2's default (`False`) applies,
  // which matches minijinja's default — left unchanged. autoescape /
  // undefined / pycompat are unchanged (transformers' env does not diverge
  // there in a way that affects rendered prompt bytes).
  env.set_trim_blocks(true);
  env.set_lstrip_blocks(true);

  // Transformers' jinja env also installs an `AssistantTracker` extension
  // providing `{% generation %}` / `{% endgeneration %}` (a training-only
  // assistant-token-mask feature). minijinja has no extension API, so rewrite
  // those tags into a semantically transparent no-output block *before*
  // loading — see `strip_generation_tags`. Done here (after trim/lstrip are
  // set, before `add_template`) so minijinja's whitespace lexing matches
  // jinja2's for the rewritten tags.
  let template = strip_generation_tags(template);

  // `loop_controls` covers `{% break %}` / `{% continue %}` in templates.
  env
    .add_template("chat", &template)
    .map_err(|e| Error::tokenizer(format!("chat template parse: {e}")))?;
  let tmpl = env
    .get_template("chat")
    .map_err(|e| Error::tokenizer(format!("chat template: {e}")))?;

  // HF's `continue_final_message`: append the sentinel to the final message's
  // content *before* building the jinja context, so the template renders the
  // augmented conversation; the rendered string is trimmed at the sentinel
  // after `render` below. An owned `Cow`-style local keeps the borrowed
  // `messages` untouched for the no-continuation path (no clone). The original
  // (pre-append) content is captured here too, threaded into the post-render
  // trim so it can validate the content actually rendered.
  let continued_messages;
  let mut original_final_content = String::new();
  let messages: &Value = if continue_final_message {
    let (mutated, original) = continue_final_message_mutate(messages)?;
    continued_messages = mutated;
    original_final_content = original;
    &continued_messages
  } else {
    messages
  };

  let mut ctx = serde_json::Map::new();
  ctx.insert("messages".into(), messages.clone());
  ctx.insert(
    "add_generation_prompt".into(),
    Value::Bool(add_generation_prompt),
  );
  ctx.insert("enable_thinking".into(), Value::Bool(enable_thinking));
  if let Some(t) = tools {
    ctx.insert("tools".into(), t.clone());
  } else {
    ctx.insert("tools".into(), Value::Null);
  }
  // Transformers' `apply_chat_template` ALWAYS passes `documents=documents`
  // (default `None`) to render, so RAG/defensive templates branching on
  // `documents is defined` / `{% if documents %}` see it as *defined*.
  // Default it to null here (before the `extra` merge below, so a caller
  // can supply an actual document list via `extra`), matching Transformers.
  ctx.insert("documents".into(), Value::Null);
  if let Some(b) = bos_token {
    ctx.insert("bos_token".into(), Value::String(b.to_owned()));
  }
  if let Some(e) = eos_token {
    ctx.insert("eos_token".into(), Value::String(e.to_owned()));
  }
  if let Some(obj) = extra.as_object() {
    for (k, v) in obj {
      ctx.insert(k.clone(), v.clone());
    }
  }

  let rendered = tmpl
    .render(JValue::from_serialize(Value::Object(ctx)))
    .map_err(|e| Error::tokenizer(format!("chat template render: {e}")))?;

  // HF's `continue_final_message` post-render trim: validate the original
  // content + sentinel actually rendered, then cut the rendered string at the
  // sentinel so the prompt ends exactly at the final message's content.
  if continue_final_message {
    continue_final_message_trim(&rendered, &original_final_content)
  } else {
    Ok(rendered)
  }
}

/// Register the globals/filters that `transformers`' jinja sandbox exposes so
/// real model templates resolve identically.
fn install_hf_extensions(env: &mut Environment<'_>) {
  // raise_exception(msg) — transformers' template error hook.
  env.add_function("raise_exception", |msg: String| -> Result<JValue, JErr> {
    Err(JErr::new(ErrorKind::InvalidOperation, msg))
  });
  // tojson — match transformers' `_compile_jinja_template` override:
  // `json.dumps(x, ensure_ascii=False, indent=indent)`. transformers
  // explicitly overrides Jinja's built-in because the built-in HTML-escapes
  // (`<` -> `<`, `>`/`&`/`'` likewise); minijinja's built-in does the
  // same, so we register our own. HF tool-schema templates routinely use
  // `| tojson(indent=4)`.
  //
  // Accepts `indent` positionally (`| tojson(4)`) or as a kwarg
  // (`| tojson(indent=4)`), mirroring minijinja's built-in signature so
  // both `{{ x | tojson }}` and `{{ x | tojson(indent=4) }}` resolve.
  env.add_filter(
    "tojson",
    |v: JValue, indent: Option<JValue>, kwargs: Kwargs| -> Result<JValue, JErr> {
      let indent_arg = match indent {
        Some(i) => Some(i),
        None => kwargs.get::<Option<JValue>>("indent")?,
      };
      kwargs.assert_all_used()?;
      let indent = match indent_arg {
        Some(ref i) => coerce_indent(i)?,
        None => None,
      };
      // Serialize the minijinja `Value` directly (as its built-in `tojson`
      // does) so map key order / value representation is preserved exactly.
      let out = py_json_dumps(&v, indent.as_deref())
        .map_err(|e| JErr::new(ErrorKind::InvalidOperation, e.to_string()))?;
      // `from_safe_string`: the output is valid JSON/HTML-inert text; do not
      // let minijinja's autoescape mangle quotes/brackets (parity with
      // transformers, where the filter result is used verbatim).
      Ok(JValue::from_safe_string(out))
    },
  );
  // strftime_now(fmt) — transformers' date helper, defined in
  // `_cached_compile_jinja_template` as exactly:
  //   def strftime_now(format): return datetime.now().strftime(format)
  // i.e. *naive local* current time formatted by Python `strftime`. Real HF
  // templates (Llama-3.x etc.) embed e.g.
  //   {{ strftime_now('%Y-%m-%d') }}
  // in the system prompt; emitting "" silently diverges prompt bytes, so we
  // implement it for real via `jiff` (local now + `strftime`).
  env.add_function("strftime_now", |fmt: String| -> Result<JValue, JErr> {
    // Recoverable: an untrusted chat_template format that jiff can't render
    // (e.g. an unknown directive) returns a tokenizer error, never a panic
    // unwinding through minijinja.
    strftime_now(&fmt)
      .map(JValue::from)
      .map_err(|e| JErr::new(ErrorKind::InvalidOperation, format!("strftime_now: {e}")))
  });
}

/// Format `dt` (a fixed, time-zone-naive civil datetime) per a Python
/// `strftime` format string. This is the injectable-clock seam: production
/// passes "local now" (see `strftime_now`); tests pass a fixed
/// `civil::DateTime` so expected bytes are deterministic.
///
/// `jiff`'s `strftime` conversion specifiers match Python's `datetime.strftime`
/// for every specifier real HF chat templates use (`%Y %m %d %H %M %S %B %b
/// %A %a %j %p %I %y %e`, the no-pad `%-d`, literal `%%`, ...). Verified
/// against CPython for the documented regression values.
///
/// `pub` so the regression suite can assert against a *fixed* civil datetime
/// (deterministic expected bytes) while production uses real local now via
/// `strftime_now` — a plain internal seam, no env-var clock hack.
/// Python `datetime.strftime` on a *naive* datetime renders the offset/zone
/// directives `%z` and `%Z` as empty strings (no tz info). A `jiff` civil
/// `DateTime` has no offset/zone, so its formatter *errors* on those — and
/// `Display::to_string()` on that error panics. Strip exactly `%z`/`%Z` to
/// match CPython before formatting. `%%` is a literal percent and preserved
/// (so `%%z` → the text `%z`, NOT stripped); other directives pass through
/// to jiff. A model-controlled `chat_template` is untrusted, so this must
/// never panic.
fn strip_naive_unsupported(format: &str) -> String {
  let mut out = String::with_capacity(format.len());
  let mut chars = format.chars();
  while let Some(c) = chars.next() {
    if c == '%' {
      match chars.next() {
        Some('%') => out.push_str("%%"),
        Some('z') | Some('Z') => {} // CPython naive datetime → "" for these
        Some(d) => {
          out.push('%');
          out.push(d);
        }
        None => out.push('%'),
      }
    } else {
      out.push(c);
    }
  }
  out
}

/// Format `dt` per a Python `strftime` format string, *fallibly* — a
/// model-supplied (untrusted) `chat_template` must never panic the process.
/// Uses jiff's fallible `fmt::strtime::format` (NOT the panic-on-error
/// `Display::to_string()`); `%z`/`%Z` are pre-stripped to match CPython's
/// empty-string output for a naive datetime.
pub fn strftime_at(dt: jiff::civil::DateTime, format: &str) -> Result<String, jiff::Error> {
  jiff::fmt::strtime::format(strip_naive_unsupported(format), dt)
}

/// Transformers' `strftime_now(format)` = `datetime.now().strftime(format)`:
/// the *system-local, time-zone-naive* current datetime formatted per Python
/// `strftime`. `jiff::Zoned::now()` reads the system time zone (matching
/// Python's `datetime.now()` local clock); we take its civil (naive) datetime
/// so formatting carries no offset/zone artifacts, exactly like Python's
/// naive `datetime`.
fn strftime_now(format: &str) -> Result<String, jiff::Error> {
  strftime_at(jiff::Zoned::now().datetime(), format)
}

// ---------------------------------------------------------------------------
// deepseek_v32 override — port of mlx_lm/chat_templates/deepseek_v32.py.
// Gated on the `tokenizer-deepseek-v32` feature.
// ---------------------------------------------------------------------------

#[cfg(feature = "tokenizer-deepseek-v32")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-deepseek-v32")))]
pub use deepseek_v32::DeepseekV32;

#[cfg(feature = "tokenizer-deepseek-v32")]
mod deepseek_v32 {
  use serde_json::Value;

  use super::ChatTemplateOverride;
  use crate::Error;

  const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
  const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
  const THINK_START: &str = "<think>";
  const THINK_END: &str = "</think>";
  const DSML: &str = "｜DSML｜";

  /// The deepseek_v32 tools-system template text. Externalized to
  /// `mlxrs/data/tokenizer/chat_template_overrides/deepseek_v32.jinja` (single
  /// source of truth) and embedded byte-for-byte by `cargo xtask-codegen`
  /// into the committed `crate::tokenizer::generated` module; the
  /// `str::replace` substitution pipeline (`{dsml}` / `{think_start}` /
  /// `{think_end}` / `{tool_schemas}`) stays Rust below — zero behavior change.
  ///
  /// Provenance: mlx-lm df1d3f3 `mlx_lm/chat_templates/deepseek_v32.py`
  /// `TOOLS_SYSTEM_TEMPLATE` (placeholder names rewritten to the Rust port's).
  const TOOLS_SYSTEM_TEMPLATE: &str = crate::tokenizer::generated::DEEPSEEK_V32_TEMPLATE;

  /// `deepseek_v32` chat-template override (Python
  /// `mlx_lm/chat_templates/deepseek_v32.py`).
  pub struct DeepseekV32;

  fn to_json(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "null".into())
  }

  fn render_tools(tools: &Value) -> String {
    let schemas = tools
      .as_array()
      .map(|a| a.iter().map(to_json).collect::<Vec<_>>().join("\n"))
      .unwrap_or_default();
    TOOLS_SYSTEM_TEMPLATE
      .replace("{dsml}", DSML)
      .replace("{think_start}", THINK_START)
      .replace("{think_end}", THINK_END)
      .replace("{tool_schemas}", &schemas)
  }

  fn tools_from_openai(tools: &Value) -> Value {
    match tools.as_array() {
      Some(arr) => Value::Array(
        arr
          .iter()
          .map(|t| t.get("function").cloned().unwrap_or_else(|| t.clone()))
          .collect(),
      ),
      None => Value::Array(vec![]),
    }
  }

  fn find_last_user_index(messages: &[Value]) -> i64 {
    for idx in (0..messages.len()).rev() {
      if let Some(r) = messages[idx].get("role").and_then(Value::as_str)
        && (r == "user" || r == "developer")
      {
        return idx as i64;
      }
    }
    -1
  }

  fn encode_arguments_to_dsml(tc: &Value) -> Result<String, Error> {
    let args_raw = tc.get("arguments");
    let arguments: Value = match args_raw {
      Some(Value::String(s)) => {
        serde_json::from_str(s).map_err(|e| Error::tokenizer(format!("deepseek_v32 args: {e}")))?
      }
      Some(other) => other.clone(),
      None => Value::Object(Default::default()),
    };
    let obj = arguments
      .as_object()
      .ok_or_else(|| Error::tokenizer("deepseek_v32: arguments not object"))?;
    let mut parts = Vec::new();
    for (k, v) in obj {
      let is_str = v.is_string();
      let value = if let Value::String(s) = v {
        s.clone()
      } else {
        to_json(v)
      };
      parts.push(format!(
        "<{DSML}parameter name=\"{k}\" string=\"{}\">{value}</{DSML}parameter>",
        if is_str { "true" } else { "false" }
      ));
    }
    Ok(parts.join("\n"))
  }

  fn render_message(
    index: usize,
    messages: &[Value],
    thinking_mode: &str,
    tools: Option<&Value>,
  ) -> Result<String, Error> {
    let mut prompt = String::new();
    // Bounds-safe: a `drop_thinking`/role-set mismatch must surface as a
    // tokenizer `Error`, never an OOB index panic on untrusted input.
    let msg = messages.get(index).ok_or_else(|| {
      Error::tokenizer(format!(
        "deepseek_v32: message index {index} out of range (len {})",
        messages.len()
      ))
    })?;
    let last_user_idx = find_last_user_index(messages);
    let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
    let content = msg.get("content").and_then(Value::as_str).unwrap_or("");
    let msg_tools = tools.cloned().or_else(|| msg.get("tools").cloned());
    let response_format = msg.get("response_format");
    let mut tool_calls = msg.get("tool_calls").cloned();
    if let Some(Value::Array(tcs)) = &tool_calls {
      tool_calls = Some(Value::Array(
        tcs
          .iter()
          .map(|tc| {
            let f = tc.get("function");
            serde_json::json!({
              "name": f.and_then(|x| x.get("name")).cloned().unwrap_or(Value::Null),
              "arguments": f.and_then(|x| x.get("arguments")).cloned().unwrap_or(Value::Null),
            })
          })
          .collect(),
      ));
    }
    let reasoning_content = msg
      .get("reasoning_content")
      .and_then(Value::as_str)
      .unwrap_or("");

    match role {
      "system" => {
        prompt.push_str(content);
        if let Some(t) = &msg_tools {
          prompt.push_str("\n\n");
          prompt.push_str(&render_tools(&tools_from_openai(t)));
        }
        if let Some(rf) = response_format {
          prompt.push_str(&format!(
          "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}",
          to_json(rf)
        ));
        }
      }
      "developer" => {
        let mut cd = String::new();
        if let Some(t) = &msg_tools {
          cd.push_str("\n\n");
          cd.push_str(&render_tools(&tools_from_openai(t)));
        }
        if let Some(rf) = response_format {
          cd.push_str(&format!(
          "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}",
          to_json(rf)
        ));
        }
        cd.push_str(&format!("\n\n# The user's message is: {content}"));
        prompt.push_str(&format!("<｜User｜>{cd}<｜Assistant｜>"));
        if index as i64 == last_user_idx && thinking_mode == "thinking" {
          prompt.push_str(THINK_START);
        } else {
          prompt.push_str(THINK_END);
        }
      }
      "user" => {
        prompt.push_str(&format!("<｜User｜>{content}<｜Assistant｜>"));
        if index as i64 == last_user_idx && thinking_mode == "thinking" {
          prompt.push_str(THINK_START);
        } else {
          prompt.push_str(THINK_END);
        }
      }
      "tool" => {
        let mut prev = index as i64 - 1;
        while prev >= 0
          && messages[prev as usize].get("role").and_then(Value::as_str) == Some("tool")
        {
          prev -= 1;
        }
        let assistant = &messages[prev.max(0) as usize];
        let order = index as i64 - prev;
        let assistant_tcs = assistant
          .get("tool_calls")
          .and_then(Value::as_array)
          .map(|a| a.len())
          .unwrap_or(0);
        if order == 1 {
          prompt.push_str("\n\n<function_results>");
        }
        prompt.push_str(&format!("\n<result>{content}</result>"));
        if order as usize == assistant_tcs {
          prompt.push_str("\n</function_results>");
          if index as i64 >= last_user_idx && thinking_mode == "thinking" {
            prompt.push_str(&format!("\n\n{THINK_START}"));
          } else {
            prompt.push_str(&format!("\n\n{THINK_END}"));
          }
        }
      }
      "assistant" => {
        let mut thinking_part = String::new();
        let mut tool_calls_content = String::new();
        if let Some(Value::Array(tcs)) = &tool_calls {
          let mut rendered = Vec::new();
          for tc in tcs {
            let name = tc.get("name").and_then(Value::as_str).unwrap_or("");
            rendered.push(format!(
              "<{DSML}invoke name=\"{name}\">\n{}\n</{DSML}invoke>",
              encode_arguments_to_dsml(tc)?
            ));
          }
          tool_calls_content.push_str(&format!(
            "\n\n<{DSML}function_calls>\n{}\n</{DSML}function_calls>",
            rendered.join("\n")
          ));
        }
        if thinking_mode == "thinking" && index as i64 > last_user_idx {
          thinking_part = format!("{reasoning_content}{THINK_END}");
        }
        prompt.push_str(&format!(
          "{thinking_part}{content}{tool_calls_content}{EOS_TOKEN}"
        ));
      }
      other => {
        return Err(Error::tokenizer(format!(
          "deepseek_v32: unknown role: {other}"
        )));
      }
    }
    Ok(prompt)
  }

  fn drop_thinking_messages(messages: &[Value]) -> Vec<Value> {
    let last_user_idx = find_last_user_index(messages);
    let mut out = Vec::new();
    for (idx, msg) in messages.iter().enumerate() {
      let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
      // `developer` is a user-equivalent role everywhere in this template
      // (`find_last_user_index` / `render_message` treat it exactly like
      // `user`), so it is retained here alongside `user`. Python upstream
      // omits `developer` from this list, which can drop a leading developer
      // message and leave `full` shorter than `messages` (an upstream
      // AssertionError path); retaining it keeps lengths consistent and the
      // rendering identical for the valid `[developer, ...]` role sets.
      if ["user", "developer", "system", "tool"].contains(&role) || idx as i64 >= last_user_idx {
        out.push(msg.clone());
      } else if role == "assistant" {
        let mut m = msg.clone();
        if let Some(o) = m.as_object_mut() {
          o.remove("reasoning_content");
        }
        out.push(m);
      }
    }
    out
  }

  impl ChatTemplateOverride for DeepseekV32 {
    fn apply(
      &self,
      messages: &[Value],
      tools: Option<&Value>,
      add_generation_prompt: bool,
      continue_final_message: bool,
      enable_thinking: bool,
    ) -> Result<String, Error> {
      let thinking_mode = if enable_thinking { "thinking" } else { "chat" };
      let mut full = messages.to_vec();
      if thinking_mode == "thinking" {
        full = drop_thinking_messages(&full);
      }
      let mut out = String::from(BOS_TOKEN);
      // Python iterates `range(len(messages))` and `render_message`'s
      // `assert 0 <= index < len(messages)` turns a `drop_thinking` length
      // mismatch into an error. Mirror that exactly: same index range, but
      // `render_message`'s bounds-checked `.get()` returns a tokenizer
      // `Error` instead of panicking (the `developer`-retention fix in
      // `drop_thinking_messages` keeps `full.len() == messages.len()` for
      // all valid role sets, so rendering is byte-identical to upstream).
      for idx in 0..messages.len() {
        out.push_str(&render_message(idx, &full, thinking_mode, tools)?);
      }
      if continue_final_message && add_generation_prompt {
        return Err(Error::tokenizer(
          "Only one of continue_final_message or add_generation_prompt can be True",
        ));
      }
      let last_role = messages
        .last()
        .and_then(|m| m.get("role"))
        .and_then(Value::as_str);
      if !add_generation_prompt
        && last_role == Some("user")
        && let Some(stripped) = out.strip_suffix("<｜Assistant｜><think>")
      {
        out = stripped.to_owned();
      }
      if continue_final_message
        && last_role == Some("assistant")
        && let Some(stripped) = out.strip_suffix(EOS_TOKEN)
      {
        out = stripped.to_owned();
      }
      Ok(out)
    }
  }
}
