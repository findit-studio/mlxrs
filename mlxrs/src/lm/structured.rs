//! Grammar-constrained decoding — port of
//! [`mlx_vlm/structured.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/structured.py)
//! (V6 / issue #180). At each decode step the processor masks the model's
//! logits to `-inf` for any token id that cannot be the next byte-grammar-
//! valid continuation, leaving only allowed tokens samplable. Backed by the
//! upstream Rust [`llguidance`] crate (the same engine the Python reference
//! uses via `llguidance` + `llguidance.hf` + `llguidance.mlx`).
//!
//! **Surface** (mirroring `mlx_vlm/structured.py:7-121`):
//!
//! - [`GrammarSpec`](crate::lm::structured::GrammarSpec) — the grammar
//!   source. `JsonSchema` (Python `grammar_from("json_schema", schema)`
//!   line 120), `Regex`, and `Lark` variants cover the three formats
//!   `llguidance::api::TopLevelGrammar` accepts as a top-level entry
//!   point.
//! - [`LLGuidanceLogitsProcessor`](crate::lm::structured::LLGuidanceLogitsProcessor)
//!   — the port of the Python class lines 7-91. Stateful (advances a
//!   [`llguidance::Matcher`] one token per step); exposes
//!   [`apply`](crate::lm::structured::LLGuidanceLogitsProcessor::apply)
//!   for direct use and
//!   [`into_logits_processor`](crate::lm::structured::LLGuidanceLogitsProcessor::into_logits_processor)
//!   to plug into [`crate::lm::generate::make_logits_processors`]' output
//!   list.
//! - [`build_json_schema_logits_processor`](crate::lm::structured::build_json_schema_logits_processor)
//!   — the port of the module function lines 105-121
//!   (`build_json_schema_logits_processor`). Single-call helper for the
//!   common "give me a JSON-schema-constrained processor" path.
//!
//! **Per-step contract.** [`crate::lm::generate::LogitsProcessor`] is now a
//! public `enum` (P1 #109) with built-in variants for common cases
//! (`LogitBias`, `RepetitionPenalty`, `PresencePenalty`, `FrequencyPenalty`).
//! Custom or stateful logits processors plug in through the
//! `LogitsProcessor::Custom(Box::new(...))` escape hatch, which preserves
//! the previous `Box<dyn Fn(&[u32], &Array) -> Result<Array>>` semantics at
//! the cost of one vtable dispatch per token. This module's
//! [`LLGuidanceLogitsProcessor::into_logits_processor`](crate::lm::structured::LLGuidanceLogitsProcessor::into_logits_processor)
//! wraps the matcher state in exactly that `Custom` variant. On the **first** call the matcher
//! is freshly built and the input-history's last token is NOT consumed
//! (mirroring the Python class's `is_first_token` flag,
//! `structured.py:18, 70-75`): the prompt has already produced the current
//! logits, so the matcher's initial state already governs which token may be
//! sampled. On every subsequent call the last id from `tokens` is fed
//! through [`llguidance::Matcher::consume_token`] BEFORE computing the next
//! mask. The returned `Array` has the same shape + dtype as the input
//! `logits`; tokens not in the matcher's allowed set are replaced with
//! `-inf` (in the logits' dtype) via [`crate::ops::logical::select`], using
//! the same `Array`/`select` masking idiom as
//! [`crate::lm::sample::apply_min_p`].
//!
//! **Shape support.** `logits` may be `[V]` (single-row) or `[1, V]` (the
//! `make_logits_processors` per-step shape — `generate.rs:86`). Larger
//! batch shapes return an error: the Python class supports arbitrary
//! `(batch, vocab)` (it tracks one matcher per batch row), but the
//! `mlxrs::lm::generate` loop only ever feeds `[1, V]`, so we keep the
//! single-matcher port and reject other shapes up front. The token-history
//! input may likewise be `[u32]` of any length — only its last element
//! matters (and only after the first step).
//!
//! **Cargo feature gate.** The whole module is gated on the `llguidance`
//! cargo feature so the `lm` umbrella alone doesn't pull in the grammar-
//! engine compile cost; callers opt in with `cargo … --features
//! "lm llguidance"`.
//!
//! **Tokenizer adapter.** [`llguidance::ParserFactory`] needs a
//! `toktrie::TokEnv` (a byte-level view of the vocab) to compile a
//! grammar against. The `toktrie_hf_tokenizers` crate builds one from a
//! HuggingFace `tokenizers::Tokenizer`; we bridge mlxrs's
//! [`crate::tokenizer::Tokenizer`] across via `serde_json`-roundtripped
//! JSON bytes (mlxrs ships `tokenizers = "0.23"` while
//! `toktrie_hf_tokenizers` pins `tokenizers = "0.21"` — the JSON wire
//! format is stable across these versions, so this avoids dragging in two
//! `tokenizers` versions transitively while still using mlxrs's own
//! tokenizer instance as the source of truth).

use std::cell::RefCell;

use llguidance::{Matcher, ParserFactory, api::TopLevelGrammar, toktrie::TokEnv};
use serde_json::Value;
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use smol_str::format_smolstr;

use crate::{
  array::Array,
  error::{
    Error, LengthMismatchPayload, OutOfRangePayload, ParsePayload, RankMismatchPayload, Result,
  },
  lm::generate::LogitsProcessor,
  ops,
  tokenizer::Tokenizer,
};

/// Specification of the grammar to constrain decoding to.
///
/// Mirrors the three top-level entry points
/// [`llguidance::api::TopLevelGrammar`] exposes via
/// `from_json_schema` / `from_regex` / `from_lark` (and the Python
/// reference's `llguidance.grammar_from("json_schema" | "regex" | "lark",
/// ...)` calls — `structured.py:120`, `mlx_vlm/server.py` callers). Any
/// extra `llguidance` constraint surfaces (GBNF, choice lists, the
/// pre-built `"llguidance"` envelope) live one level below `Lark`/the
/// grammar-list APIs and can be added later without an API break.
#[derive(Debug, Clone)]
pub enum GrammarSpec {
  /// A JSON schema (parsed `serde_json::Value`). Compiled via
  /// [`TopLevelGrammar::from_json_schema`] — the Python reference's
  /// `grammar_from("json_schema", _serialize_schema(schema))`,
  /// `structured.py:120`.
  JsonSchema(Value),
  /// A Rust-`regex`-syntax regular expression. Compiled via
  /// [`TopLevelGrammar::from_regex`].
  Regex(String),
  /// A Lark-grammar source string. Compiled via
  /// [`TopLevelGrammar::from_lark`]. See the upstream
  /// <https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md>
  /// for the supported Lark subset.
  Lark(String),
}

impl GrammarSpec {
  /// Compile the spec into a [`TopLevelGrammar`].
  fn into_top_level(self) -> TopLevelGrammar {
    match self {
      GrammarSpec::JsonSchema(value) => TopLevelGrammar::from_json_schema(value),
      GrammarSpec::Regex(rx) => TopLevelGrammar::from_regex(&rx),
      GrammarSpec::Lark(src) => TopLevelGrammar::from_lark(src),
    }
  }
}

/// Build a [`TokEnv`] (byte-level vocab view for [`llguidance`]) from an
/// [`mlxrs` `Tokenizer`](Tokenizer), optionally padding the vocab to
/// `model_vocab_size` placeholder special tokens.
///
/// Mirrors the Python `llguidance.hf.from_tokenizer(tokenizer)` call
/// (`structured.py:117`). The mlxrs `tokenizers = "0.23"` and
/// `toktrie_hf_tokenizers`'s pinned `tokenizers = "0.21"` are two
/// different crate versions in the dep tree: passing the live `Tokenizer`
/// across would force a second `tokenizers` major to be compiled. The
/// HuggingFace tokenizer.json wire format is stable across both versions,
/// so we round-trip through `serde_json::to_vec` + `ByteTokenizer::from_json_bytes`
/// (which calls `Tokenizer::from_bytes` on the v0.21 side). Result: one
/// `tokenizers` version in the dep graph, no behavioural change.
///
/// **Padded vocabularies.** Many transformer LMs round the LM-head's
/// output dim up (e.g. 32064 for Llama with a 32000-token tokenizer) so
/// the logits' last axis is LARGER than `tokenizer.get_vocab_size(true)`.
/// `ByteTokenizer::into_tok_env(Some(n))` pads the toktrie with
/// placeholder special tokens up to `n` (see `toktrie_hf_tokenizers`'s
/// `ByteTokenizerEnv::new`), so the resulting mask has the model's vocab
/// width and the placeholder ids fall in the "no real byte sequence
/// maps to this id" bucket — the grammar engine never allows them, so
/// they're masked to `-inf` for free. `None` falls back to the
/// tokenizer's own vocab size (the previous behaviour, fine for models
/// whose LM head matches the tokenizer exactly).
fn tok_env_from_tokenizer(
  tokenizer: &Tokenizer,
  model_vocab_size: Option<usize>,
) -> Result<TokEnv> {
  let json = serde_json::to_vec(tokenizer.hf()).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "llguidance: serialize HF tokenizer",
      "HF tokenizer JSON",
      Box::new(e) as Box<dyn std::error::Error + Send + Sync>,
    ))
  })?;
  let bt = ByteTokenizer::from_json_bytes(&json).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "llguidance: build ByteTokenizer",
      "HF tokenizer JSON",
      std::io::Error::other(e.to_string()),
    ))
  })?;

  // Sync mlxrs's configured EOS ids into the resulting [`TokEnv`]'s
  // `tok_trie().eos_token_set()` so terminal-grammar EOS-only masks
  // unmask the model's ACTUAL stop ids. Upstream
  // `ByteTokenizer::from_tokenizer`
  // (`toktrie_hf_tokenizers/src/lib.rs:186-205`) only auto-detects a
  // small hardcoded set of EOS strings (`</s>`, `<|endoftext|>`,
  // `<|end_of_text|>`, DeepSeek's `<｜end▁of▁sentence｜>`, `<eos>`) and
  // silently defaults `tok_eos` to id `0` for everything else (note: it
  // classifies `<|im_end|>`/`<|eot_id|>` as `tok_end_of_turn`, NOT
  // `tok_eos`). Without this sync:
  //   - a caller-supplied `eos_token_ids` override is ignored by
  //     llguidance,
  //   - a `tokenizer_config.json` `eos_token` string outside the
  //     hardcoded list (e.g. `<|im_end|>`) is silently dropped,
  //   - and `compute_mask_or_eos` returns an EOS-only mask gated by the
  //     WRONG eos id (id `0`).
  //
  // **Padded-vocab support (V6 R4).** We register the configured EOS ids
  // AFTER widening the toktrie via `ByteTokenizerEnv::new(bt,
  // model_vocab_size)`. The widened `TokTrie::vocab_size()` then equals
  // `model_vocab_size.unwrap_or(bt_vocab)`, and
  // [`TokTrie::with_eos_tokens`]
  // (`toktrie/src/toktree.rs:300-313`) asserts every id against THAT
  // widened vocab — so a padded-range EOS id (e.g. `120` for a model
  // with `bt_vocab=99` + `model_vocab_size=Some(128)`) is now legitimate
  // and fully registered in `tok_trie.eos_token_set()`.
  //
  // The earlier R3 design called `bt.set_eos_tokens` BEFORE
  // `into_tok_env`, against the still-unpadded
  // `bt.tokrx_info().vocab_size`. That meant padded-range ids could
  // only be silently filtered out (otherwise upstream's `assert!`
  // panicked), and a config supplying ONLY a padded-range EOS would
  // leave the trie's EOS set defaulted to upstream's auto-detected id
  // (often `0`) — `compute_mask_or_eos` would then unmask the WRONG
  // token in a terminal-grammar state. Switching to post-widening
  // registration via `ByteTokenizerEnv::new` + `tok_trie.with_eos_tokens`
  // closes that silent-failure case (the recoverable Err still fires on
  // ids above the WIDENED bound).
  //
  // **Out-of-range validation.** The mlxrs
  // [`Tokenizer::eos_token_ids()`] is populated from caller-supplied
  // and/or `tokenizer_config.json`-derived ids without any vocab-size
  // check (existing tests install ids as high as `4242`), so we MUST
  // validate before crossing the FFI boundary into
  // `TokTrie::with_eos_tokens`'s `assert!`. The effective bound is the
  // widened `env.tok_trie.vocab_size()` — same value the upstream
  // assert checks against — surfaced as a recoverable
  // [`Error::OutOfRange`] with the offending id + bound.
  //
  // The mlxrs `Tokenizer::eos_token_ids()` returns a `BTreeSet<u32>` —
  // iterating in sorted-numeric order is deterministic; for mask
  // correctness only the SET membership matters
  // (`eos_token_set()` collects every registered id), so the chosen
  // slot-0 primary doesn't change which ids are unmasked. Skip the call
  // when the set is empty (no eos configured at all) —
  // `with_eos_tokens` panics on an empty slice, and in that case
  // upstream's hardcoded detection is the only signal we have anyway.
  let configured_eos: Vec<u32> = tokenizer.eos_token_ids_iter().collect();

  let mut env = ByteTokenizerEnv::new(bt, model_vocab_size).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "llguidance: build ByteTokenizerEnv",
      "tokenizer environment",
      std::io::Error::other(e.to_string()),
    ))
  })?;

  if !configured_eos.is_empty() {
    let widened_vocab = env.tok_trie.vocab_size();
    for &eos in &configured_eos {
      if (eos as usize) >= widened_vocab {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "llguidance: configured EOS token id",
          "must be < tok_trie vocab bound",
          format_smolstr!("{eos} (vocab_bound={widened_vocab})"),
        )));
      }
    }
    // Register against the WIDENED vocab — padded-range ids pass, and
    // `tok_trie.eos_token_set()` now reflects the caller-supplied set
    // exactly (no silent drop).
    env.tok_trie = env.tok_trie.with_eos_tokens(&configured_eos);
  }

  Ok(env.to_env())
}

/// MLX logits processor backed by [`llguidance`].
///
/// Port of `mlx_vlm/structured.py`'s `LLGuidanceLogitsProcessor` class
/// (lines 7-91). Holds the constraint state machine (a
/// [`llguidance::Matcher`]) plus an `is_first_token` flag mirroring the
/// reference (`structured.py:18, 70-75`). One processor per generation —
/// not safe to share across concurrent generations because the matcher is
/// stateful.
///
/// # Mutability
///
/// [`crate::lm::generate::LogitsProcessor`] is now a public `enum`
/// (P1 #109); stateful custom processors plug in via
/// `LogitsProcessor::Custom(Box::new(...))`, whose closure type is
/// `Box<dyn Fn(&[u32], &Array) -> Result<Array>>` (not `FnMut`). Processors
/// that own mutable state — exactly this one — therefore hold it behind a
/// [`RefCell`]. The borrow is taken inside [`apply`](Self::apply) and
/// released before the call returns; calling the same processor
/// re-entrantly (e.g. composing it with another processor that re-invokes
/// it) would panic, but the single-call-per-step
/// `make_logits_processors` chain never does that.
pub struct LLGuidanceLogitsProcessor {
  matcher: RefCell<Matcher>,
  is_first_token: RefCell<bool>,
}

impl LLGuidanceLogitsProcessor {
  /// Construct a new processor from a [`GrammarSpec`] + tokenizer
  /// (optionally pinned to the model's LM-head vocab width).
  ///
  /// Internally: builds a [`TokEnv`] from `tokenizer` (one `~1.5s` walk
  /// of the vocab — the Python reference caches this; see
  /// [`build_json_schema_logits_processor`] for the schema-side
  /// equivalent), compiles the grammar through
  /// [`ParserFactory::new_simple`] + [`ParserFactory::create_parser`],
  /// and wraps the resulting [`llguidance::TokenParser`] in a
  /// [`Matcher`]. Any grammar-compile error from `llguidance` surfaces
  /// as an [`Error::Backend`].
  ///
  /// # `model_vocab_size`
  ///
  /// `Some(n)` pins the resulting mask width to `n` (the logits' last-
  /// axis size), padding the underlying toktrie with placeholder special
  /// tokens beyond the tokenizer's own `get_vocab_size(true)`. Use this
  /// when the LM-head's output dim is wider than the tokenizer's vocab
  /// (a common case — Llama-style models round the LM head's output dim
  /// up for hardware alignment, leaving 64+ "padding" ids that have no
  /// real bytes). Without it, `apply` would surface a
  /// [`Error::RankMismatch`] / [`Error::LengthMismatch`] on the first call. `None` keeps the
  /// previous behaviour (mask width = tokenizer vocab size), fine for
  /// models whose LM head matches the tokenizer exactly.
  pub fn new(
    grammar: GrammarSpec,
    tokenizer: &Tokenizer,
    model_vocab_size: Option<usize>,
  ) -> Result<Self> {
    let tok_env = tok_env_from_tokenizer(tokenizer, model_vocab_size)?;
    let mut factory = ParserFactory::new_simple(&tok_env).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "llguidance: ParserFactory",
        "llguidance grammar factory",
        std::io::Error::other(e.to_string()),
      ))
    })?;
    // Match the Python reference's quiet default; the `mlx_vlm` Python
    // call sites don't set `log_level`, so llguidance's level-1
    // "warnings to stderr" default would print mid-decode.
    factory.set_stderr_log_level(0);
    let top = grammar.into_top_level();
    let parser = factory.create_parser(top);
    let matcher = Matcher::new(parser);
    // `Matcher::new` swallows a parser-construction `Err` into a
    // sentinel error-state matcher; surface that as an `Err` here so
    // the caller hears about a bad grammar at construction time rather
    // than via every per-step `apply` call.
    if let Some(err) = matcher.get_error() {
      return Err(Error::Parse(ParsePayload::new(
        "llguidance: grammar compile",
        "llguidance grammar",
        std::io::Error::other(err),
      )));
    }
    Ok(Self {
      matcher: RefCell::new(matcher),
      is_first_token: RefCell::new(true),
    })
  }

  /// Apply the constraint to one step's logits.
  ///
  /// Mirrors `LLGuidanceLogitsProcessor.__call__` (`structured.py:78-91`):
  ///
  /// 1. On the first call, the matcher's initial state (post-grammar-
  ///    compile) is what governs the very next token; we do NOT consume
  ///    any history token because the prompt's last token came from the
  ///    user, not from the grammar.
  /// 2. On every subsequent call, the last id in `tokens` is what the
  ///    model just emitted, so consume it through
  ///    [`Matcher::consume_token`] before recomputing the mask.
  /// 3. Compute the allowed-token bit-vector with
  ///    [`Matcher::compute_mask_or_eos`] (which forces an EOS-only mask
  ///    when the grammar has reached a terminal/stopped state, instead
  ///    of erroring as `compute_mask` would — the documented "terminal
  ///    grammar → next token must be EOS" path), iterate it to build a
  ///    `[V]` boolean "disallowed" array, broadcast across the logits'
  ///    batch axis, and mask via `select(disallowed, -inf, logits)`.
  ///
  /// `logits` may be `[V]` or `[1, V]`; the returned `Array` keeps the
  /// input shape + dtype (the `-inf` scalar is cast `astype(logits.dtype)`
  /// — same idiom as [`crate::lm::sample::apply_min_p`] /
  /// `apply_top_k`).
  pub fn apply(&self, tokens: &[u32], logits: &Array) -> Result<Array> {
    let shape = logits.shape();
    let vocab = match shape.as_slice() {
      [v] => *v,
      [1, v] => *v,
      other => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "LLGuidanceLogitsProcessor: expected logits shape `[V]` or `[1, V]`",
          other.len() as u32,
          other.to_vec(),
        )));
      }
    };

    {
      let mut first = self.is_first_token.borrow_mut();
      if *first {
        // First step: do NOT consume any history token (the prompt's
        // last token is upstream of the grammar). Mirrors
        // `structured.py:70-75` `is_first_token` branch.
        *first = false;
      } else if let Some(&last) = tokens.last() {
        // Subsequent steps: feed the previously-sampled token into the
        // matcher before recomputing the mask. `consume_token` returns
        // `Err` on an invalid token (which would mean the sampler
        // picked a disallowed token — i.e. the constraint pipeline is
        // broken upstream), so surface it.
        self.matcher.borrow_mut().consume_token(last).map_err(|e| {
          Error::Parse(ParsePayload::new(
            "llguidance: consume_token",
            "previously-sampled token",
            std::io::Error::other(format!("token={last}: {e}")),
          ))
        })?;
      }
    }

    // Compute the allowed-bit vector for the next token.
    //
    // [`Matcher::compute_mask`] errors out when the grammar has finished
    // (`StopReason != NotStopped` — e.g. a `Regex("a")` grammar after the
    // single `a` token has been consumed). For a terminal grammar the
    // documented next step is to force EOS, not to abort generation, so
    // we call [`Matcher::compute_mask_or_eos`] which auto-returns an
    // EOS-only [`toktrie::SimpleVob`] when the parser is stopped (it
    // delegates to `compute_mask` otherwise). The downstream
    // `is_allowed`-based loop then naturally masks every token but EOS
    // to `-inf`, the documented "terminal grammar → force EOS" path.
    let mask = self
      .matcher
      .borrow_mut()
      .compute_mask_or_eos()
      .map_err(|e| {
        Error::Parse(ParsePayload::new(
          "llguidance: compute_mask_or_eos",
          "llguidance allowed-mask",
          std::io::Error::other(e.to_string()),
        ))
      })?;

    // Validate sizes match: `mask.len()` is `tokrx_info.vocab_size`
    // (padded up to 32-bit granularity); the logits' last axis is the
    // model's vocab. They MUST match; otherwise the bit→logit mapping is
    // garbage. The byte-tokenizer adapter sets `vocab_size` from
    // `tokenizers::Tokenizer::get_vocab_size(true)`, so this catches the
    // "wrong tokenizer for this model" footgun up front.
    if mask.len() < vocab {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "LLGuidanceLogitsProcessor: matcher mask vs logits vocab",
        vocab,
        mask.len(),
      )));
    }

    // Build the `[V]` boolean "disallowed" array. `SimpleVob::is_allowed`
    // returns `true` for tokens the grammar accepts; we invert (and
    // truncate to the logits' vocab width — masks can be slightly wider
    // due to bitmask alignment padding).
    let mut disallowed: Vec<bool> = Vec::with_capacity(vocab);
    for tok in 0..vocab {
      disallowed.push(!mask.is_allowed(tok as u32));
    }

    // Reshape `[V]` → match the logits' rank so `select`'s broadcast
    // semantics align the mask with the last (vocab) axis. For `[1, V]`
    // logits we want a `[1, V]` mask; for `[V]` logits we keep `[V]`.
    let mask_shape: Vec<i32> = match shape.as_slice() {
      [v] => vec![*v as i32],
      [b, v] => vec![*b as i32, *v as i32],
      _ => unreachable!("shape validated above"),
    };
    // The boolean dense mask has shape `[V]`; broadcast to `[1, V]` via
    // reshape (the mask is already vocab-length so this is free).
    let bool_mask_flat = Array::from_slice::<bool>(&disallowed, &(vocab,))?;
    let bool_mask = if mask_shape.len() == 1 {
      bool_mask_flat
    } else {
      let dims: &[i32] = &mask_shape;
      ops::shape::reshape(&bool_mask_flat, &dims)?
    };

    // `-inf` scalar in the logits' dtype, exactly the
    // `apply_top_k`/`apply_min_p` idiom (`sample.rs:110, 161`).
    let neg_inf_f32 = Array::full::<f32>(&(1,), f32::NEG_INFINITY)?;
    let neg_inf = ops::misc::astype(&neg_inf_f32, logits.dtype()?)?;

    // `out = where(disallowed, -inf, logits)` — same shape + dtype as input.
    ops::logical::select(&bool_mask, &neg_inf, logits)
  }

  /// Reset the matcher to its initial state. Mirrors `structured.py:23-26`
  /// `reset()`. After this call the next `apply` is treated as the first
  /// step again.
  pub fn reset(&self) -> Result<()> {
    self.matcher.borrow_mut().reset().map_err(|e| {
      Error::Parse(ParsePayload::new(
        "llguidance: reset",
        "llguidance matcher state",
        std::io::Error::other(e.to_string()),
      ))
    })?;
    *self.is_first_token.borrow_mut() = true;
    Ok(())
  }

  /// Wrap into a [`LogitsProcessor`] so the processor plugs into
  /// [`crate::lm::generate::make_logits_processors`]' output list.
  ///
  /// Returns the [`LogitsProcessor::Custom`] variant (the
  /// out-of-tree-processor escape hatch — see the type's `# Breaking
  /// change` note for the enum-unification rationale). The boxed
  /// closure captures `self` by move; one processor instance per
  /// generation (the matcher is stateful — see the type-level note).
  pub fn into_logits_processor(self) -> LogitsProcessor {
    LogitsProcessor::Custom(Box::new(move |tokens: &[u32], logits: &Array| {
      self.apply(tokens, logits)
    }))
  }
}

/// One-shot helper: build a [`LLGuidanceLogitsProcessor`] from a JSON
/// schema + tokenizer (+ optional model vocab-size override).
///
/// Port of `mlx_vlm/structured.py:105-121`
/// (`build_json_schema_logits_processor`). The Python reference caches the
/// per-tokenizer LL tokenizer; mlxrs's caller (the upcoming structured-
/// response wiring) owns the tokenizer lifecycle and can construct
/// processors at the request boundary — so this thin helper is the
/// natural shim. Equivalent to
/// `LLGuidanceLogitsProcessor::new(GrammarSpec::JsonSchema(schema),
/// tokenizer, model_vocab_size)`. See
/// [`LLGuidanceLogitsProcessor::new`]'s `model_vocab_size` doc for when
/// to pass `Some(n)` (padded LM heads).
pub fn build_json_schema_logits_processor(
  schema: Value,
  tokenizer: &Tokenizer,
  model_vocab_size: Option<usize>,
) -> Result<LLGuidanceLogitsProcessor> {
  LLGuidanceLogitsProcessor::new(GrammarSpec::JsonSchema(schema), tokenizer, model_vocab_size)
}
