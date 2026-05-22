//! Stateful multi-turn chat session — a port of mlx-swift-lm's
//! [`ChatSession`](https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/MLXLMCommon/ChatSession.swift)
//! (`MLXLMCommon/ChatSession.swift`). mlx-lm's Python side has only the
//! `chat.py` REPL **application** (argparse + stdin loop), not a reusable
//! session type, so the Swift `ChatSession` is the authoritative reference
//! for the *library* surface.
//!
//! A [`ChatSession`] owns the four pieces a conversation needs and nothing
//! else — the model, the tokenizer, the per-layer KV cache, and the running
//! message history — and exposes a single [`ChatSession::respond`] turn-taking
//! API in two shapes:
//!
//! - **non-streaming** — [`ChatSession::respond`] returns the assembled
//!   `Result<String>` for the turn;
//! - **streaming** — [`ChatSession::stream_respond`] returns an
//!   `Iterator<Item = Result<GenerationResponse>>` (the same per-token
//!   [`GenerationResponse`] [`crate::lm::generate::stream_generate`] yields),
//!   and the *consumed* turn — prompt + every produced token — is appended to
//!   the held history when the iterator is dropped.
//!
//! Each turn renders the full history through
//! [`Tokenizer::apply_chat_template`], generates over the **held** KV cache,
//! and appends both the user prompt and the model's reply to the history — so
//! turn N+1 reuses turn N's cache (only the new tokens are prefilled, never
//! the whole conversation).
//!
//! ## Incremental prefill (the cross-turn cache-reuse mechanism)
//!
//! mlx-swift-lm's `ChatSession` *claims* "only the new tokens are prefilled"
//! but its `LLMModel.prepare` feeds the **full** rendered `input.text` into
//! the model every turn, and `KVCacheSimple.update` unconditionally appends
//! `keys.dim(2)` to its offset — so the Swift reference actually re-appends
//! the entire prior conversation's KV each turn (offset grows by the *whole*
//! render, the prefix is duplicated). [`ChatSession`] does **not** port that
//! defect: it ports the *documented* contract via real incremental prefill.
//!
//! Every realised cache carries the exact *known* token sequence in its KV
//! state (a `Vec<u32>`), plus a count of leading **opaque** tokens — a
//! builder-restored prefix whose ids the session was never given
//! (`opaque_len + known.len()` equals the cache `offset()`). The opaque
//! prefix is the one part of the KV state the session cannot name; it is
//! **never** re-rendered into a turn's prompt (the session's history holds
//! only turns it itself ran), so a render begins with the *known* region
//! only. On a new turn the session:
//!
//! 1. renders the full prompt → `prompt_ids`;
//! 2. checks whether `prompt_ids` begins with the cache's *known* token
//!    sequence (the opaque prefix sits implicitly in front of the cache, not
//!    in front of `prompt_ids`);
//! 3. if so — the render **extends** the cache — feeds [`generate_step`] only
//!    the suffix beyond the known ids; the cache continues from its current
//!    `offset()`, so turn N+1's prefill cost is the *new* tokens only. A
//!    fresh builder-restored cache (empty `known`) extends on every render,
//!    so the entire new prompt is fed as the suffix continuing the opaque
//!    prefix;
//! 4. if not — the render **diverges** (an `instructions` change, a
//!    non-prefix-stable template) — discards the stale cache and **rebuilds
//!    from scratch** for that turn, feeding the full `prompt_ids` — slower,
//!    but never wrong.
//!
//! The generated tokens are folded into the cached *known* sequence alongside
//! the prompt, so turn N+2 sees turn N+1's prompt *and* reply as the cached
//! known prefix. (The speculative-decoding path cannot reuse its cache — its
//! KV caches are consumed by the speculative generator and not handed back —
//! so it always rebuilds and re-prefills; that is a documented divergence.)
//!
//! ## Concurrency divergence from the Swift reference (deliberate)
//!
//! mlx-swift-lm's `ChatSession` is a `final class` that wraps its KV cache in a
//! `SerialAccessContainer` lock and runs each turn on a `Task`, so several
//! *distinct* sessions can generate on background threads in parallel; its own
//! doc-comment still states a single `ChatSession` "is not thread-safe" and
//! "should be used from a single task/thread at a time".
//!
//! mlxrs's [`Array`](crate::array::Array) is `!Send`/`!Sync` (single-thread,
//! matching MLX's compute-stream model). [`ChatSession`] therefore ports the
//! **logic** — state ownership, the `respond` turn, history accumulation, KV
//! cache reuse across turns — as a plain owning struct driven by `&mut self`,
//! and drops the actor/`Task`/lock machinery. This is **not** a behavioural
//! divergence: the Swift type's documented single-session contract is already
//! "one thread at a time", which `&mut self` enforces at compile time. The
//! only capability not ported is running *multiple* sessions on background
//! threads — out of reach for a `!Send` `Array` and out of scope here.
//!
//! ## What it reuses (no reimplementation)
//!
//! - **Generation** — every turn drives [`crate::lm::generate::generate_step`]
//!   (the architecture-agnostic [`Generator`] loop) for the standard path and
//!   [`crate::lm::speculative::speculative_stream_generate`] for the
//!   speculative-decoding path. The session never re-implements the decode
//!   loop; it only adds the streaming-detokenizer + history glue around it
//!   (the same eos-terminated glue [`crate::lm::generate::stream_generate`]
//!   applies — `ChatSession` has no string `stop_words`, so the eos-only path
//!   is faithful and complete).
//! - **Template** — the prompt is rendered by
//!   [`Tokenizer::apply_chat_template`]; the session never renders jinja.
//! - **Cache** — built by [`make_prompt_cache`] and carried across turns via
//!   [`Generator::into_cache`]; the session never reaches into a concrete
//!   cache type.

use std::rc::Rc;

use serde_json::{Value, json};

use crate::{
  error::{Error, Result},
  lm::{
    cache::{CacheConfig, KvCache, make_prompt_cache, save_prompt_cache},
    generate::{GenConfig, GenerationResponse, Generator, generate_step},
    model::Model,
    speculative::{DraftConfig, SpeculativeStream, speculative_stream_generate},
  },
  tokenizer::{Tokenizer, wrapper::BoxedDetokenizer},
};

/// The role of a [`ChatMessage`] in the conversation — mlx-swift-lm's
/// `Chat.Message.Role`.
///
/// Rendered to the lowercase string the chat template expects (`"system"` /
/// `"user"` / `"assistant"` / `"tool"`); [`ChatSession`] tags the prompt with
/// [`Role::User`] and the model's reply with [`Role::Assistant`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
  /// System instructions (the optional leading message).
  System,
  /// A user turn.
  User,
  /// An assistant (model) turn.
  Assistant,
  /// A tool-result turn.
  Tool,
}

impl Role {
  /// The lowercase template key for this role (`messages[i]["role"]`).
  fn as_str(self) -> &'static str {
    match self {
      Role::System => "system",
      Role::User => "user",
      Role::Assistant => "assistant",
      Role::Tool => "tool",
    }
  }
}

/// One message in a [`ChatSession`]'s history — mlx-swift-lm's
/// `Chat.Message`, restricted to the text fields the text-only port needs
/// (the Swift type's `images` / `videos` are VLM-only and out of scope).
///
/// `{ role, content }` is exactly the object the chat template iterates over
/// (`{% for m in messages %}{{ m['role'] }}{{ m['content'] }}{% endfor %}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
  /// The speaker (`messages[i]["role"]`).
  pub role: Role,
  /// The message text (`messages[i]["content"]`).
  pub content: String,
}

impl ChatMessage {
  /// A `system` message (the optional leading instructions).
  pub fn system(content: impl Into<String>) -> Self {
    Self {
      role: Role::System,
      content: content.into(),
    }
  }

  /// A `user` message.
  pub fn user(content: impl Into<String>) -> Self {
    Self {
      role: Role::User,
      content: content.into(),
    }
  }

  /// An `assistant` (model) message.
  pub fn assistant(content: impl Into<String>) -> Self {
    Self {
      role: Role::Assistant,
      content: content.into(),
    }
  }

  /// A `tool`-result message.
  pub fn tool(content: impl Into<String>) -> Self {
    Self {
      role: Role::Tool,
      content: content.into(),
    }
  }
}

/// Configuration for speculative decoding inside a [`ChatSession`] — a port of
/// mlx-swift-lm's `SpeculativeDecodingConfig`.
///
/// When a session is built with [`ChatSessionBuilder::speculative`], every
/// turn runs through [`crate::lm::speculative::speculative_stream_generate`]:
/// the small `draft_model` proposes candidate tokens that the main model
/// verifies in a single forward pass (~2-3× speedup, no quality change). Both
/// models **must share the same tokenizer vocabulary**.
///
/// The draft model's KV cache is allocated once (from the same
/// [`CacheConfig`]) and reused across turns, exactly like the main cache.
///
/// The draft model is held as an [`Rc`] (not a `Box`): a [`ChatSession`] keeps
/// its draft model across turns, but
/// [`crate::lm::speculative::speculative_stream_generate`] *consumes* the
/// [`DraftConfig`] it is handed each turn — so the session clones the cheap
/// [`Rc`] handle per turn rather than surrendering ownership. [`Rc`] (not
/// `Arc`) matches mlxrs's single-thread, `!Send` [`Array`](crate::array::Array)
/// model.
pub struct SpeculativeDecodingConfig {
  /// The lightweight model that proposes candidate tokens (mlx-swift-lm
  /// `draftModel`; mlx-lm `draft_model`).
  pub draft_model: Rc<dyn Model>,
  /// Tokens proposed by the draft model per verification cycle
  /// (mlx-swift-lm `numDraftTokens`, default `5`).
  pub num_draft_tokens: usize,
  /// The draft model's cache shape — one [`KvCache`] per draft-model decoder
  /// layer. Kept alongside `draft_model` so the session can build the draft
  /// cache without a second `CacheConfig` argument.
  pub draft_cache_config: CacheConfig,
}

impl SpeculativeDecodingConfig {
  /// The Swift `numDraftTokens` default.
  pub const DEFAULT_NUM_DRAFT_TOKENS: usize = 5;

  /// Build a config with the default `num_draft_tokens` (`5`).
  pub fn new(draft_model: Rc<dyn Model>, draft_cache_config: CacheConfig) -> Self {
    Self {
      draft_model,
      num_draft_tokens: Self::DEFAULT_NUM_DRAFT_TOKENS,
      draft_cache_config,
    }
  }
}

/// One per-layer KV cache — `make_prompt_cache`'s output, one boxed
/// [`KvCache`] per decoder layer.
type KvCaches = Vec<Box<dyn KvCache>>;

/// The exact token sequence already represented in a realised cache's KV
/// state — the bookkeeping that drives [`ChatSession`]'s incremental prefill
/// (see the [module docs](self)).
///
/// `known` is the suffix of the cached tokens whose ids the session itself
/// fed (`opaque_len + known.len()` always equals the cache `offset()`).
/// `opaque_len` is the count of *leading* cached tokens whose ids are
/// unknown — `0` for a session-built cache (every token is `known`), nonzero
/// only for a cache restored via [`ChatSessionBuilder::cache`] (a pre-built
/// prefix whose token ids the builder was never given).
#[derive(Clone)]
struct CachedTokens {
  /// Count of leading KV tokens whose ids are unknown (a restored opaque
  /// prefix); `0` for a session-built cache.
  opaque_len: usize,
  /// The token ids the session fed into the cache after the opaque prefix —
  /// `opaque_len + known.len() == cache.offset()`.
  known: Vec<u32>,
}

impl CachedTokens {
  /// A freshly-built, empty cache: no opaque prefix, nothing fed yet.
  fn empty() -> Self {
    Self {
      opaque_len: 0,
      known: Vec::new(),
    }
  }

  /// A builder-restored cache of `offset` tokens whose ids are all unknown.
  fn opaque(offset: usize) -> Self {
    Self {
      opaque_len: offset,
      known: Vec::new(),
    }
  }
}

/// The session's KV-cache slot — a port of the Swift `ChatSession.Cache`
/// enum (`empty` / `kvcache` / `history`).
///
/// mlx-swift-lm defers cache allocation: a fresh session is `empty`, a
/// re-hydrated one is `history`, and either transitions to `kvcache` on the
/// first generation. The same three states are ported here so a session's
/// cache is `None` until the first turn (matching the Swift `currentCache()`
/// observable: `nil` before generation, `nil` again after `clear()`).
enum CacheSlot {
  /// No cache, no replayed history — a fresh session.
  Empty,
  /// The realised per-layer KV cache (built on the first turn). For the
  /// speculative path the draft cache is carried alongside.
  Realised {
    /// The main model's per-layer KV cache, advanced across turns.
    cache: KvCaches,
    /// The draft model's per-layer KV cache, present iff the session was
    /// built with a [`SpeculativeDecodingConfig`].
    draft_cache: Option<KvCaches>,
    /// The exact token sequence in `cache`'s KV state — drives incremental
    /// prefill on the next turn (see [`CachedTokens`] / the module docs).
    cached: CachedTokens,
  },
  /// A re-hydrated message history awaiting its first generation — the
  /// Swift `.history` case (used by [`ChatSessionBuilder::history`]).
  History(Vec<ChatMessage>),
  /// A speculative session that has run at least one turn. The
  /// [`speculative_stream_generate`] iterator *consumes* its caches and does
  /// not return them, so there is no advanced cache to carry — the next turn
  /// rebuilds from the [`CacheConfig`] (a documented divergence — see
  /// [`SpeculativeTurn`]). This state is deliberately **not** [`Realised`]:
  /// presenting a freshly-allocated offset-0 cache as the "current" cache
  /// would let [`ChatSession::save_cache`] persist a cache that does not
  /// encode the conversation, so a speculative session reports
  /// [`ChatSession::has_cache`] `false` and `save_cache` returns
  /// [`ChatSessionError::SpeculativeCacheUnsupported`].
  SpeculativeSpent,
}

/// Errors thrown by [`ChatSession`] — a port of mlx-swift-lm's
/// `ChatSessionError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatSessionError {
  /// [`ChatSession::save_cache`] was called before any generation occurred —
  /// the Swift `ChatSessionError.noCacheAvailable`.
  NoCacheAvailable,
  /// [`ChatSession::save_cache`] was called on a speculative-decoding
  /// session. [`speculative_stream_generate`] consumes its KV caches and
  /// does not return them, so a speculative session holds no advanced cache
  /// to persist — saving the freshly-rebuilt offset-0 cache would write a
  /// cache that does not encode the conversation. No Swift analogue (the
  /// Swift `ChatSession` never made the speculative cache observable).
  SpeculativeCacheUnsupported,
  /// [`ChatSessionBuilder::build`] was called with both
  /// [`ChatSessionBuilder::cache`] (a restored opaque KV prefix) and
  /// [`ChatSessionBuilder::speculative`] set. This combination is
  /// **unsupported** because [`speculative_stream_generate`] consumes its KV
  /// caches and does not return them: the first speculative turn would use
  /// the restored prefix, but the second turn would silently rebuild from a
  /// fresh offset-0 cache (the opaque prefix's ids are unknown, so the
  /// session cannot re-render and re-prefill it). Pick one of:
  /// drop [`ChatSessionBuilder::cache`] (run speculative decoding without a
  /// restored prefix) or drop [`ChatSessionBuilder::speculative`] (use the
  /// standard path, which reuses its KV cache across turns and preserves the
  /// restored prefix).
  SpeculativeCacheRestoreUnsupported,
}

impl std::fmt::Display for ChatSessionError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      ChatSessionError::NoCacheAvailable => f.write_str(
        "no KV cache is available: call respond() / stream_respond() before save_cache()",
      ),
      ChatSessionError::SpeculativeCacheUnsupported => f.write_str(
        "speculative-decoding sessions do not support cache save: the speculative \
         generator consumes its KV caches and rebuilds them each turn, so there is \
         no advanced cache that encodes the conversation to persist",
      ),
      ChatSessionError::SpeculativeCacheRestoreUnsupported => f.write_str(
        "ChatSessionBuilder::cache() combined with ChatSessionBuilder::speculative() \
         is unsupported: speculative_stream_generate consumes its KV caches and does \
         not return them, so the restored opaque prefix would be used on the first \
         turn and silently lost on every subsequent turn (the opaque prefix's token \
         ids are unknown, so the session cannot re-prefill it). Drop .cache(..) to \
         run speculative decoding without a restored prefix, or drop .speculative(..) \
         to use the standard path (which reuses its KV cache across turns and \
         preserves the restored prefix)",
      ),
    }
  }
}

impl std::error::Error for ChatSessionError {}

impl From<ChatSessionError> for Error {
  fn from(e: ChatSessionError) -> Self {
    Error::Backend {
      message: e.to_string(),
    }
  }
}

/// Builder for a [`ChatSession`] — the Rust-idiomatic collapse of
/// mlx-swift-lm's eight overlapping `ChatSession.init(...)` overloads.
///
/// The Swift type ships four pairs of initializers (the cartesian product of
/// `ModelContainer`/`ModelContext` × plain / `history:` / `cache:`). mlxrs has
/// one model handle ([`Box<dyn Model>`](Model)), so the variation that
/// remains is the initial state — fresh, re-hydrated from a history, or
/// restored from a pre-built KV cache. Those three plus the optional
/// `instructions` / `generate_params` / speculative knobs are expressed as
/// builder methods, then [`ChatSessionBuilder::build`] produces the session.
///
/// ```ignore
/// let session = ChatSession::builder(model, tokenizer, cache_config)
///     .instructions("You are a helpful assistant.")
///     .build()?;
/// ```
pub struct ChatSessionBuilder {
  model: Box<dyn Model>,
  tokenizer: Tokenizer,
  cache_config: CacheConfig,
  instructions: Option<String>,
  generate_params: GenConfig,
  speculative: Option<SpeculativeDecodingConfig>,
  initial: CacheSlot,
}

impl ChatSessionBuilder {
  /// Optional system instructions prepended (as a [`Role::System`] message)
  /// to every rendered prompt — mlx-swift-lm's `instructions`.
  pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
    self.instructions = Some(instructions.into());
    self
  }

  /// Parameters controlling generation (sampler, `max_tokens`, penalties, …)
  /// — mlx-swift-lm's `generateParameters`. Defaults to [`GenConfig::default`]
  /// (greedy, `max_tokens = 256`).
  pub fn generate_params(mut self, params: GenConfig) -> Self {
    self.generate_params = params;
    self
  }

  /// Enable speculative decoding — mlx-swift-lm's `speculativeDecoding`.
  pub fn speculative(mut self, config: SpeculativeDecodingConfig) -> Self {
    self.speculative = Some(config);
    self
  }

  /// Restore an existing message history — mlx-swift-lm's `history:`
  /// initializer ("prompt re-hydration" for persistent chat apps). The full
  /// message array (including any leading system message) is replayed on the
  /// first turn; the cache stays unrealised (`currentCache()` is `None`)
  /// until then, exactly like the Swift `.history` cache state.
  ///
  /// Mutually exclusive with [`ChatSessionBuilder::cache`]: the last one set
  /// wins.
  pub fn history(mut self, history: Vec<ChatMessage>) -> Self {
    self.initial = CacheSlot::History(history);
    self
  }

  /// Restore a pre-built KV cache — mlx-swift-lm's `cache:` initializer
  /// (prefix caching: prefill a long shared context once, persist it via
  /// [`ChatSession::save_cache`], restore it across sessions).
  ///
  /// > If the cache already encodes a system prompt, do **not** also set
  /// > [`ChatSessionBuilder::instructions`] — they would be re-tokenized on
  /// > every turn without matching KV state, producing incoherent output
  /// > (the same caveat the Swift `cache:` initializer documents).
  ///
  /// Mutually exclusive with [`ChatSessionBuilder::history`]: the last one
  /// set wins.
  pub fn cache(mut self, cache: Vec<Box<dyn KvCache>>) -> Self {
    // The restored cache encodes some prefix (a system prompt + document),
    // but the builder is not given its token ids — so the whole prefill is
    // *opaque*: `opaque_len` is the cache's current `offset()` and `known`
    // is empty. The first turn feeds its rendered prompt as the suffix that
    // continues this opaque prefix (the documented prefix-caching use); the
    // session can verify+extend the `known` portion it appends, and falls
    // back to a rebuild if a later render does not extend it.
    let offset = cache.first().map(|c| c.offset()).unwrap_or(0);
    self.initial = CacheSlot::Realised {
      cache,
      draft_cache: None,
      cached: CachedTokens::opaque(offset),
    };
    self
  }

  /// Finish building the [`ChatSession`].
  ///
  /// Returns [`ChatSessionError::SpeculativeCacheRestoreUnsupported`] if both
  /// [`ChatSessionBuilder::cache`] (a restored opaque KV prefix) and
  /// [`ChatSessionBuilder::speculative`] were set: see that variant's doc for
  /// why the combination is rejected and what to drop instead. Every other
  /// builder shape (fresh, `.history(..)`-seeded, `.cache(..)` alone,
  /// `.speculative(..)` alone) is supported and returns `Ok`.
  pub fn build(self) -> Result<ChatSession> {
    // Reject `.cache(restored).speculative(..)` — Finding 1. A restored cache
    // is stored as `CacheSlot::Realised` with an OPAQUE prefix whose token
    // ids the builder was never given. The first speculative turn would
    // consume that cache (correctly using the restored prefix) and the
    // `commit()` write-back would set the slot to `CacheSlot::SpeculativeSpent`,
    // because `speculative_stream_generate` does not hand its caches back.
    // The next turn's `take_cache()` treats `SpeculativeSpent` like `Empty`
    // and allocates a fresh offset-0 cache with `CachedTokens::empty()` —
    // silently losing the restored prefix (it cannot be re-rendered, since
    // the session was never given its token ids). Rather than re-prefilling
    // the opaque region every turn (expensive, changes the speculative
    // driver's contract) we reject the combination at build time with an
    // actionable message pointing at the two alternatives.
    if self.speculative.is_some() && matches!(self.initial, CacheSlot::Realised { .. }) {
      return Err(ChatSessionError::SpeculativeCacheRestoreUnsupported.into());
    }
    Ok(ChatSession {
      model: self.model,
      tokenizer: self.tokenizer,
      cache_config: self.cache_config,
      instructions: self.instructions,
      generate_params: self.generate_params,
      speculative: self.speculative,
      cache: self.initial,
      history: Vec::new(),
    })
  }
}

/// A stateful, multi-turn chat session — see the [module docs](self).
///
/// Owns the model, tokenizer, KV cache and conversation history; each
/// [`ChatSession::respond`] / [`ChatSession::stream_respond`] call renders the
/// running history, generates over the **held** cache, and appends the new
/// turn — so the cache is reused turn-to-turn. `&mut self` on the turn-taking
/// methods enforces the Swift type's documented "one turn at a time" contract
/// at compile time (see the module docs' concurrency note).
pub struct ChatSession {
  /// The language model (immutable after load; `Model::forward` is `&self`).
  model: Box<dyn Model>,
  /// The tokenizer — owns the chat template and the streaming detokenizer.
  tokenizer: Tokenizer,
  /// The model's cache shape, used to (re)build the KV cache on the first
  /// turn — the faithful stand-in for Swift's `model.newCache(parameters:)`
  /// (the [`Model`] trait does not expose a layer count).
  cache_config: CacheConfig,
  /// Optional system instructions, prepended to every rendered prompt — the
  /// Swift `instructions` (public there; a public accessor pair here).
  instructions: Option<String>,
  /// Generation parameters — the Swift `generateParameters`.
  generate_params: GenConfig,
  /// Speculative-decoding config, `None` when disabled — the Swift
  /// `speculativeDecoding`.
  speculative: Option<SpeculativeDecodingConfig>,
  /// The KV-cache slot (`Empty` / `Realised` / `History`) — the Swift
  /// `ChatSession.Cache` enum.
  cache: CacheSlot,
  /// The accumulated conversation, **excluding** the system message (that is
  /// re-derived from `instructions` each turn, matching the Swift loop which
  /// re-prepends `.system(instructions)` every call).
  history: Vec<ChatMessage>,
}

impl ChatSession {
  /// Start building a session for `model` / `tokenizer` whose KV cache has
  /// the shape `cache_config` describes (one [`KvCache`] per decoder layer).
  ///
  /// See [`ChatSessionBuilder`] for the optional `instructions` /
  /// `generate_params` / `speculative` / `history` / `cache` knobs; call
  /// [`ChatSessionBuilder::build`] to finish.
  pub fn builder(
    model: Box<dyn Model>,
    tokenizer: Tokenizer,
    cache_config: CacheConfig,
  ) -> ChatSessionBuilder {
    ChatSessionBuilder {
      model,
      tokenizer,
      cache_config,
      instructions: None,
      generate_params: GenConfig::default(),
      speculative: None,
      initial: CacheSlot::Empty,
    }
  }

  /// The current system instructions, if any — the Swift public
  /// `instructions` getter.
  pub fn instructions(&self) -> Option<&str> {
    self.instructions.as_deref()
  }

  /// Replace the system instructions — the Swift public `instructions`
  /// setter. Takes effect on the **next** turn (the prompt is re-rendered
  /// each call); does not retroactively change the KV cache.
  pub fn set_instructions(&mut self, instructions: Option<String>) {
    self.instructions = instructions;
  }

  /// The generation parameters — the Swift public `generateParameters`
  /// getter.
  pub fn generate_params(&self) -> &GenConfig {
    &self.generate_params
  }

  /// Mutable access to the generation parameters — the Swift public
  /// `generateParameters` setter (e.g. raise `max_tokens` between turns).
  pub fn generate_params_mut(&mut self) -> &mut GenConfig {
    &mut self.generate_params
  }

  /// The accumulated conversation history (every turn appended so far),
  /// **excluding** the system message — see [`ChatSession::history`]'s field
  /// doc. Empty for a fresh session before the first turn.
  pub fn history(&self) -> &[ChatMessage] {
    &self.history
  }

  /// Whether the KV cache has been realised (a turn has run since
  /// construction / the last [`ChatSession::clear`]).
  ///
  /// This is the observable behind the Swift test-support `withCache` /
  /// `currentCache()`: `false` for a fresh or `history`-seeded session before
  /// the first turn, `true` after a turn, `false` again after
  /// [`ChatSession::clear`]. `true` immediately for a session built via
  /// [`ChatSessionBuilder::cache`].
  ///
  /// Always `false` for a **speculative** session, even after a turn:
  /// [`speculative_stream_generate`] consumes its caches and does not return
  /// an advanced cache to carry (the next turn rebuilds), so there is no
  /// realised cache to observe or save.
  pub fn has_cache(&self) -> bool {
    matches!(self.cache, CacheSlot::Realised { .. })
  }

  /// Borrow the realised per-layer KV cache, or `None` if it has not been
  /// realised yet — the Swift test-support `withCache(_:)`.
  pub fn current_cache(&self) -> Option<&[Box<dyn KvCache>]> {
    match &self.cache {
      CacheSlot::Realised { cache, .. } => Some(cache),
      _ => None,
    }
  }

  /// Clear the session: drop the KV cache and the conversation history,
  /// **preserving** the system instructions and generation parameters — the
  /// Swift `clear()`.
  ///
  /// After `clear` the session is exactly as freshly built (minus a
  /// re-hydrated history / cache): [`ChatSession::has_cache`] is `false` and
  /// the next turn starts from a fresh cache.
  pub fn clear(&mut self) {
    self.cache = CacheSlot::Empty;
    self.history.clear();
  }

  /// Save the current KV cache to `path` (a `safetensors` file) via
  /// [`save_prompt_cache`] — the Swift `saveCache(to:)`.
  ///
  /// Restore it later with [`ChatSessionBuilder::cache`] +
  /// [`crate::lm::cache::load_prompt_cache`].
  ///
  /// Returns [`ChatSessionError::NoCacheAvailable`] (faithful to the Swift
  /// `ChatSessionError.noCacheAvailable`) if no generation has occurred — the
  /// cache is not realised, so there is nothing to persist.
  /// Returns [`ChatSessionError::SpeculativeCacheUnsupported`] for a
  /// speculative session: its caches are consumed and rebuilt each turn, so
  /// no cache encoding the conversation exists to persist.
  pub fn save_cache(&self, path: &std::path::Path) -> Result<()> {
    match &self.cache {
      CacheSlot::Realised { cache, .. } => {
        save_prompt_cache(path, cache, &std::collections::HashMap::new())
      }
      CacheSlot::SpeculativeSpent => Err(ChatSessionError::SpeculativeCacheUnsupported.into()),
      _ => Err(ChatSessionError::NoCacheAvailable.into()),
    }
  }

  /// Render the prompt for the next turn and resolve which messages still
  /// need to be appended to the persisted `history`.
  ///
  /// Mirrors the Swift turn setup: prepend `instructions` (if any), replay a
  /// re-hydrated `.history` exactly once, append the new user message, then
  /// render through [`Tokenizer::apply_chat_template`] with
  /// `add_generation_prompt = true` (the trailing assistant marker that cues
  /// the model to reply).
  ///
  /// Returns `(prompt_ids, replayed_history)`: `prompt_ids` is the encoded
  /// prompt for [`generate_step`], and `replayed_history` is the messages a
  /// `.history`-seeded first turn must fold into the persisted `history`
  /// (empty on every later turn).
  fn build_turn_prompt(&self, prompt: &str, role: Role) -> Result<(Vec<u32>, Vec<ChatMessage>)> {
    // The Swift loop rebuilds `messages` from scratch each turn: optional
    // system message, then the conversation, then the new turn.
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(instructions) = &self.instructions {
      messages.push(ChatMessage::system(instructions.clone()));
    }

    // A `.history`-seeded session replays its restored messages on the first
    // turn only; once realised they live in `self.history` like any other
    // turn. `replayed` is what the caller must fold into `self.history`.
    let replayed: Vec<ChatMessage> = match &self.cache {
      CacheSlot::History(h) => h.clone(),
      _ => Vec::new(),
    };
    messages.extend(replayed.iter().cloned());
    messages.extend(self.history.iter().cloned());
    messages.push(ChatMessage {
      role,
      content: prompt.to_string(),
    });

    let json_messages = Value::Array(
      messages
        .iter()
        .map(|m| json!({ "role": m.role.as_str(), "content": m.content }))
        .collect(),
    );

    // `add_generation_prompt = true` appends the assistant turn marker (the
    // Swift `UserInput` / processor default); `continue_final_message =
    // false` — the two are mutually exclusive (see `apply_chat_template`).
    let prompt_ids = self
      .tokenizer
      .apply_chat_template_ids(&json_messages, None, true, false, None)
      .map_err(|e| Error::Backend {
        message: format!("ChatSession: apply_chat_template failed: {e}"),
      })?;
    Ok((prompt_ids, replayed))
  }

  /// Take the per-layer KV cache(s) out of the slot, building a fresh cache
  /// on the first turn — the Swift `switch cache { case .empty / .kvcache /
  /// .history }` cache-preparation block.
  ///
  /// Returns `(main_cache, draft_cache, cached_tokens)`: `draft_cache` is
  /// `Some` iff the session uses speculative decoding (built once, then
  /// carried), and `cached_tokens` records what the returned `main_cache`'s
  /// KV already encodes (drives incremental prefill). The slot is left
  /// [`CacheSlot::Empty`] — [`ChatResponseStream`]'s `Drop` write-back puts
  /// the advanced cache(s) back when the turn completes.
  fn take_cache(&mut self) -> (KvCaches, Option<KvCaches>, CachedTokens) {
    let slot = std::mem::replace(&mut self.cache, CacheSlot::Empty);
    match slot {
      // `.kvcache` — reuse the carried cache(s) and their token bookkeeping.
      CacheSlot::Realised {
        cache,
        draft_cache,
        cached,
      } => {
        // A speculative session whose draft cache was not yet built (e.g.
        // restored via `cache:`) gets it allocated now, once.
        let draft = match (&self.speculative, draft_cache) {
          (Some(spec), None) => Some(make_prompt_cache(&spec.draft_cache_config)),
          (_, existing) => existing,
        };
        (cache, draft, cached)
      }
      // `.empty` / `.history` / speculative-spent — allocate a fresh cache
      // (the `.history` messages were already folded into the rendered
      // prompt by `build_turn_prompt`, so only the cache shape matters).
      CacheSlot::Empty | CacheSlot::History(_) | CacheSlot::SpeculativeSpent => {
        let cache = make_prompt_cache(&self.cache_config);
        let draft = self
          .speculative
          .as_ref()
          .map(|spec| make_prompt_cache(&spec.draft_cache_config));
        (cache, draft, CachedTokens::empty())
      }
    }
  }

  /// Decide the incremental-prefill plan for a standard turn.
  ///
  /// Given the freshly-rendered `prompt_ids`, the cache taken from the slot
  /// and the tokens that cache already encodes, returns the cache to feed
  /// [`generate_step`], the token *slice range* of `prompt_ids` to feed it,
  /// and the [`CachedTokens::opaque_len`] the resulting cache carries.
  ///
  /// ## The opaque prefix is *not* part of `prompt_ids`
  ///
  /// The crucial bookkeeping fact (Finding 1): a cache's `opaque_len`
  /// leading tokens are tokens the session *never knew* — a builder-restored
  /// prefix (a system prompt + document prefilled outside the session). The
  /// session's [`ChatSession::history`] never contains them, so a turn's
  /// render **never re-renders the opaque region**. By contrast `cached.known`
  /// *is* re-rendered: those ids came from prompts / replies the session does
  /// hold in its history. So a prefix-extending render `prompt_ids` begins
  /// with `cached.known` (the opaque region is *implicitly* in front of the
  /// cache, not in front of `prompt_ids`).
  ///
  /// - **Reuse** (the common path): if `prompt_ids` begins with exactly the
  ///   known cached ids `cached.known` and is strictly longer (a non-empty
  ///   new suffix), the cache is kept and only the suffix
  ///   `prompt_ids[known.len() ..]` is fed. The cache continues from its
  ///   current `offset()` (`opaque_len + known.len()`); turn N+1 prefills the
  ///   *new* tokens only. The `opaque_len` is carried forward unchanged — a
  ///   restored opaque prefix stays opaque, and the full new render is folded
  ///   into `known` by [`ChatResponseStream::commit`]. This subsumes the
  ///   builder-restored-cache case: a fresh (`known` empty) restored cache
  ///   reuses on *every* render (`prompt_ids` trivially "begins with" the
  ///   empty `known`), so the entire newly-rendered prompt is fed as the
  ///   suffix that continues the opaque prefix — exactly the documented
  ///   prefix-caching contract.
  /// - **Rebuild** (divergence / degenerate): if the render does *not* begin
  ///   with `cached.known` (an `instructions` change, a non-prefix-stable
  ///   template) — or if it equals `cached.known` exactly, leaving no suffix
  ///   for [`generate_step`]'s non-empty-prompt contract — the stale cache is
  ///   dropped, a fresh cache is built, and the *full* `prompt_ids` is fed
  ///   (`opaque_len` resets to `0`). Slower (re-prefills the conversation)
  ///   but always correct.
  ///
  /// Returns `(cache, prefill_start, opaque_len)`: feed
  /// `prompt_ids[prefill_start..]` to `generate_step` over `cache`; the
  /// resulting realised cache will encode an opaque prefix of `opaque_len`.
  fn decide_prefill(
    &self,
    prompt_ids: &[u32],
    cache: KvCaches,
    cached: &CachedTokens,
  ) -> (KvCaches, usize, usize) {
    // The render must begin with exactly the *known* cached ids (the opaque
    // prefix is implicitly in front of the CACHE, never re-rendered into
    // `prompt_ids` — see the method doc) and be strictly longer than them,
    // so `prompt_ids[known.len()..]` is a non-empty suffix for
    // `generate_step`. `prompt_ids.len() > known.len()` makes that slice
    // in-bounds and non-empty.
    let extends = prompt_ids.len() > cached.known.len() && prompt_ids.starts_with(&cached.known);
    if extends {
      // Reuse: keep the cache, feed only the suffix beyond the known ids;
      // the opaque prefix is carried forward unchanged. For a fresh
      // builder-restored cache (`known` empty) this feeds the WHOLE
      // `prompt_ids` as the suffix continuing the opaque prefix.
      (cache, cached.known.len(), cached.opaque_len)
    } else {
      // Divergence or degenerate render: rebuild from scratch and feed the
      // whole prompt. `cache` (the stale cache) is dropped here.
      drop(cache);
      (make_prompt_cache(&self.cache_config), 0, 0)
    }
  }

  /// Produce a complete response to `prompt` — the non-streaming Swift
  /// `respond(to:)`.
  ///
  /// Equivalent to draining [`ChatSession::stream_respond`] and concatenating
  /// every text segment (exactly the Swift `respond` = "for await chunk in
  /// streamResponse { output += chunk }"). The user prompt and the model's
  /// reply are appended to the held [`ChatSession::history`], and the KV
  /// cache is advanced for the next turn.
  ///
  /// `prompt` is tagged [`Role::User`]; use [`ChatSession::respond_as`] for
  /// another role. Any generation error propagates as `Err` (the turn is
  /// still recorded up to the failure — see [`ChatSession::stream_respond`]).
  pub fn respond(&mut self, prompt: &str) -> Result<String> {
    self.respond_as(prompt, Role::User)
  }

  /// Produce a complete response to `prompt` sent under `role` — the Swift
  /// `respond(to:role:)`.
  pub fn respond_as(&mut self, prompt: &str, role: Role) -> Result<String> {
    let mut output = String::new();
    let mut last_err: Option<Error> = None;
    {
      let mut stream = self.stream_respond_as(prompt, role)?;
      for response in &mut stream {
        match response {
          Ok(r) => output.push_str(&r.text),
          Err(e) => {
            last_err = Some(e);
            break;
          }
        }
      }
      // `stream` is dropped here: its `Drop` impl writes the advanced cache
      // and the (prompt + reply) turn back into the session.
    }
    match last_err {
      Some(e) => Err(e),
      None => Ok(output),
    }
  }

  /// Produce a **streaming** response to `prompt` — the Swift
  /// `streamResponse(to:)` (and its `streamDetails` cousin: this yields the
  /// full per-token [`GenerationResponse`], not just the text chunk).
  ///
  /// Returns a [`ChatResponseStream`], an `Iterator<Item =
  /// Result<GenerationResponse>>` — the same item plain
  /// [`crate::lm::generate::stream_generate`] yields. The held KV cache is
  /// passed in, advanced by every step, and — together with the new turn
  /// (`prompt` + the assembled reply) — written back into the session when
  /// the stream is **dropped** (whether fully consumed, partially consumed,
  /// or abandoned), so the next turn reuses it. Dropping early therefore
  /// still records a (possibly partial) assistant turn, matching the Swift
  /// `streamResponse` interrupt semantics (`testChatSessionAsyncInterrupt`).
  ///
  /// `prompt` is tagged [`Role::User`]; use [`ChatSession::stream_respond_as`]
  /// for another role.
  pub fn stream_respond(&mut self, prompt: &str) -> Result<ChatResponseStream<'_>> {
    self.stream_respond_as(prompt, Role::User)
  }

  /// Produce a streaming response to `prompt` sent under `role` — the Swift
  /// `streamResponse(to:role:)`.
  pub fn stream_respond_as(&mut self, prompt: &str, role: Role) -> Result<ChatResponseStream<'_>> {
    // 1. Render the prompt (history + new turn) through the chat template.
    let (prompt_ids, replayed) = self.build_turn_prompt(prompt, role)?;

    // 2. Take the KV cache out of the slot (fresh on the first turn) along
    //    with the token sequence it already encodes.
    let (cache, draft_cache, cached) = self.take_cache();

    // 3. Fold a re-hydrated `.history` into the persisted history (once),
    //    then append the new user turn. The assistant reply is appended by
    //    `ChatResponseStream::drop` once generation finishes.
    self.history.extend(replayed);
    self.history.push(ChatMessage {
      role,
      content: prompt.to_string(),
    });

    // 4. Prepare the per-turn `GenConfig`. `stream_generate` overrides
    //    `cfg.eos` with the tokenizer's set; mirror that so `finish_reason`
    //    matches the plain entry points exactly.
    let detok = self.tokenizer.detokenizer();
    let eos: Vec<u32> = self.tokenizer.eos_token_ids().iter().copied().collect();
    let mut cfg = self.generate_params.clone();
    cfg.eos = eos.clone();
    let max_tokens = cfg.max_tokens;

    // 5. Decide the incremental-prefill plan for the standard path: reuse
    //    the cache + feed only the new suffix when the render extends the
    //    cached prefix, else rebuild + feed the whole prompt (see
    //    `decide_prefill`). The speculative path cannot reuse its caches
    //    (`speculative_stream_generate` consumes them), so it always feeds
    //    the full `prompt_ids` and rebuilds — handled in step 7.
    let (std_cache, prefill_start, opaque_len) = if self.speculative.is_none() {
      self.decide_prefill(&prompt_ids, cache, &cached)
    } else {
      // `cache` is moved into the speculative turn below; no plan needed.
      (cache, 0, 0)
    };

    // 6. Split the `&mut self` borrow into disjoint field references: the
    //    generation driver borrows the model + tokenizer *immutably* for the
    //    stream's lifetime, while the stream needs *mutable* access to the
    //    cache slot + history for the `Drop` write-back. A whole-`self`
    //    `&mut` cannot coexist with the driver's `&self.model`, so the
    //    fields are borrowed separately (all disjoint — sound).
    let ChatSession {
      model,
      tokenizer,
      speculative,
      cache: cache_slot,
      history,
      ..
    } = self;
    let model: &dyn Model = &**model;

    // 7. Build the per-token generation driver — speculative or standard.
    //    The streaming-detokenizer + finish-reason glue is identical to
    //    `stream_generate`'s eos path (`ChatSession` has no `stop_words`).
    let driver = match speculative.as_ref() {
      Some(spec) => {
        // The draft cache is always `Some` here: `take_cache` builds it for
        // any speculative session. Defensive fallback keeps this total.
        let draft_cache =
          draft_cache.unwrap_or_else(|| make_prompt_cache(&spec.draft_cache_config));
        Driver::Speculative(Box::new(SpeculativeTurn::new(
          model,
          tokenizer,
          // The speculative path re-prefills the whole conversation each
          // turn (its caches are consumed, not reused).
          &prompt_ids,
          std_cache,
          draft_cache,
          DraftConfig {
            // `DraftConfig` owns its `draft_model`; the session keeps its
            // own copy by cloning the cheap `Rc` handle (the weights are
            // shared, not duplicated) — see `SpeculativeDecodingConfig`.
            draft_model: Box::new(RcModel(Rc::clone(&spec.draft_model))),
            n_draft_tokens: spec.num_draft_tokens,
          },
          cfg,
        )))
      }
      None => Driver::Standard(Box::new(StandardTurn {
        // Feed `generate_step` ONLY the suffix beyond the cached prefix
        // (incremental prefill): the cache it is handed continues from its
        // current `offset()`, so turn N+1 never re-prefills the prior
        // conversation. `decide_prefill` guarantees `prefill_start <
        // prompt_ids.len()`, so this slice is always non-empty.
        generator: generate_step(model, &prompt_ids[prefill_start..], std_cache, cfg),
        // The standard path carries the draft cache (always `None` here)
        // through so the write-back round-trips the slot shape.
        draft_cache,
      })),
    };

    Ok(ChatResponseStream {
      cache_slot,
      history,
      driver: Some(driver),
      detok,
      eos,
      max_tokens,
      prompt_tokens: prompt_ids.len(),
      produced: 0,
      reply: String::new(),
      // The full rendered prompt + the opaque-prefix length: `commit()`
      // folds the generated tokens onto `prompt_ids` to form the next
      // turn's cached-token sequence (incremental-prefill bookkeeping).
      prompt_ids,
      opaque_len,
      generated: Vec::new(),
      finished: false,
      detok_finalized: false,
      committed: false,
    })
  }
}

/// A `Model` trait object that forwards to an [`Rc`]-shared model — lets
/// [`DraftConfig`] (which *owns* its `draft_model: Box<dyn Model>`) be fed the
/// draft model a [`ChatSession`] keeps across turns, without duplicating the
/// weights.
///
/// [`speculative_stream_generate`] consumes the [`DraftConfig`] it is handed
/// each turn, so the session cannot move its own `speculative.draft_model`
/// out. Instead it clones the cheap [`Rc`] handle and boxes an `RcModel` —
/// `'static` (the inner `dyn Model` is `'static`), so it satisfies
/// `DraftConfig`'s `Box<dyn Model>`. The forward is a single virtual call
/// through the `Rc` — no weight copy.
struct RcModel(Rc<dyn Model>);

impl Model for RcModel {
  fn forward(
    &self,
    tokens: &crate::array::Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<crate::array::Array> {
    self.0.forward(tokens, cache)
  }

  fn forward_embeddings(
    &self,
    embeddings: &crate::array::Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<crate::array::Array> {
    self.0.forward_embeddings(embeddings, cache)
  }
}

/// The per-turn generation driver — the standard [`Generator`] loop or the
/// speculative-decoding loop. One driver covers both paths inside
/// [`ChatResponseStream`]; both variants are boxed so the enum stays small.
enum Driver<'a> {
  /// Standard decoding: the architecture-agnostic [`Generator`] iterator
  /// ([`generate_step`]) — see [`StandardTurn`].
  Standard(Box<StandardTurn<'a>>),
  /// Speculative decoding: the [`speculative_stream_generate`] iterator plus
  /// the caches it will own (recovered when the turn finishes) — see
  /// [`SpeculativeTurn`].
  Speculative(Box<SpeculativeTurn<'a>>),
}

/// Holds a standard-decoding turn: the [`generate_step`] [`Generator`]
/// iterator (which owns and advances the main KV cache in place) plus the
/// draft cache slot.
///
/// `draft_cache` is always `None` here — it is carried only so the cache slot
/// round-trips its `(main, draft)` shape unchanged. The advanced main cache is
/// reclaimed after the turn via [`Generator::into_cache`].
struct StandardTurn<'a> {
  /// The token-level generation iterator; owns + advances the main cache.
  generator: Generator<'a, dyn Model>,
  /// The draft cache slot (always `None` on the standard path).
  draft_cache: Option<KvCaches>,
}

/// Holds a speculative-decoding turn: the [`SpeculativeStream`] iterator
/// threaded through the supplied caches.
///
/// [`speculative_stream_generate`] consumes both caches and — unlike the
/// standard [`Generator`] — does not hand them back, so a speculative turn
/// has **no advanced cache to carry**: after the turn the session's cache
/// slot becomes [`CacheSlot::SpeculativeSpent`] and the next turn rebuilds
/// its caches from the [`CacheConfig`] (a documented divergence — speculative
/// sessions re-prefill the conversation each turn, and cannot save a cache).
/// The standard path reuses the cache via [`Generator::into_cache`] and
/// supports incremental prefill + cache save.
///
/// The stream is held as the concrete [`SpeculativeStream`] (not a boxed
/// `dyn Iterator`) so an interrupted turn can reach
/// [`SpeculativeStream::finalize_tail`] in [`ChatResponseStream::commit`]
/// (Finding 2): the speculative driver's streaming detokenizer is finalized
/// only on eos / `max_tokens`, so without that call a stream dropped
/// mid-generation would lose any tail a BPE/SPM detokenizer withheld.
struct SpeculativeTurn<'a> {
  /// The streaming speculative iterator, yielding one response per token.
  iter: SpeculativeStream<'a>,
}

impl<'a> SpeculativeTurn<'a> {
  /// Build the speculative turn — constructs the
  /// [`speculative_stream_generate`] iterator over the supplied caches.
  #[allow(clippy::too_many_arguments)]
  fn new(
    target: &'a dyn Model,
    tokenizer: &'a Tokenizer,
    prompt: &[u32],
    cache: Vec<Box<dyn KvCache>>,
    draft_cache: Vec<Box<dyn KvCache>>,
    draft_cfg: DraftConfig,
    cfg: GenConfig,
  ) -> Self {
    let iter = speculative_stream_generate(
      target,
      tokenizer,
      prompt,
      cache,
      draft_cache,
      draft_cfg,
      cfg,
    );
    Self { iter }
  }
}

/// A streaming chat response — the iterator returned by
/// [`ChatSession::stream_respond`].
///
/// Yields one [`GenerationResponse`] per generated token (the same item
/// [`crate::lm::generate::stream_generate`] yields: text segment, token id,
/// counts, `finish_reason`). On [`Drop`] — exhausted or abandoned early — the
/// advanced KV cache and the assistant reply assembled so far are written
/// back into the originating [`ChatSession`], so the next turn reuses the
/// cache and sees the completed history.
///
/// The lifetime `'s` borrows the session for the duration of the stream — the
/// `&mut self` turn-taking contract: a session has at most one live stream.
/// Internally it holds *disjoint* borrows of the session's fields (the
/// generation driver borrows the model and tokenizer immutably; the cache
/// slot and history are borrowed mutably for the `Drop` write-back), which is
/// why [`ChatSession::stream_respond_as`] splits the `&mut self`.
pub struct ChatResponseStream<'s> {
  /// Mutable borrow of the session's KV-cache slot — the `Drop` write-back
  /// stores the advanced cache here.
  cache_slot: &'s mut CacheSlot,
  /// Mutable borrow of the session's conversation history — the `Drop`
  /// write-back appends the assistant reply here.
  history: &'s mut Vec<ChatMessage>,
  /// The per-turn generation driver. `Option` so [`Drop`] can `take` it and
  /// reclaim the [`Generator`]'s cache by value.
  driver: Option<Driver<'s>>,
  /// The streaming detokenizer — maps token ids to readable text segments.
  detok: BoxedDetokenizer,
  /// The tokenizer's eos id set: generation ends on the first eos token.
  eos: Vec<u32>,
  /// `max_tokens` from the session's [`GenConfig`] — the "length" stop.
  max_tokens: usize,
  /// The encoded prompt length for this turn — surfaced on every yielded
  /// [`GenerationResponse::prompt_tokens`] (the standard `Generator` path
  /// only; the speculative path adopts the iterator's own response, which
  /// already carries it).
  prompt_tokens: usize,
  /// Tokens yielded so far (mlx-lm's `n`).
  produced: usize,
  /// The assistant reply assembled so far — appended to the session history
  /// on `Drop`.
  reply: String,
  /// The full rendered prompt for this turn (all `prompt_tokens` ids).
  /// `commit()` forms the next turn's *known* cached-token region by
  /// concatenating `prompt_ids` with [`generated`](Self::generated) and
  /// truncating to `offset() - opaque_len` — the incremental-prefill
  /// bookkeeping (see the [module docs](self)). The opaque prefix is *not*
  /// part of `prompt_ids` (the session never re-renders it).
  prompt_ids: Vec<u32>,
  /// The count of leading KV tokens whose ids are unknown (a builder-restored
  /// opaque prefix); `0` for a session-built cache. Carried unchanged into
  /// the next turn's [`CachedTokens::opaque_len`] on a cache-reuse turn, or
  /// reset to `0` when `decide_prefill` rebuilds.
  opaque_len: usize,
  /// Every token id the model sampled this turn, in order — folded after
  /// `prompt_ids` to form the next turn's cached-token sequence. The standard
  /// path appends each sampled token here as it streams; the speculative path
  /// leaves this empty (its cache is rebuilt, not reused).
  generated: Vec<u32>,
  /// `true` once the stream has ended (eos, `max_tokens`, or an `Err`) — the
  /// iterator fuses afterwards, never re-entering the model.
  finished: bool,
  /// `true` once the streaming detokenizer has been **finalized** for this
  /// turn (`detok.finalize()` called and any withheld tail appended to
  /// `reply`). Split from [`finished`](Self::finished) for Finding 2:
  /// `finished` is also set on the `Err` branch of [`Iterator::next`], but an
  /// `Err`-terminated turn has NOT had its detokenizer finalized — its
  /// withheld tail (e.g. the BPE detok's bare-space token) is still in the
  /// detok's `unflushed` buffer. [`commit`](Self::commit) finalizes the detok
  /// (standard or speculative path) whenever `!detok_finalized`, so an
  /// error-terminated stream — like the early-drop and natural-termination
  /// paths — records token-complete history. Set to `true` by the natural-
  /// finish branches (eos / `max_tokens` in `next`, which already finalize
  /// inline), and again by `commit` when it finalizes on the error / early-
  /// drop path; idempotent (`finalize_tail()` returns "" on repeats).
  detok_finalized: bool,
  /// `true` once the turn's cache + reply have been written back to the
  /// session, so a `Drop` after a manual drain does not commit twice.
  committed: bool,
}

impl ChatResponseStream<'_> {
  /// Write the advanced cache and the assembled reply back into the session
  /// — the body of [`Drop`], factored out so it runs at most once
  /// (idempotent via the `committed` flag).
  ///
  /// **Standard path** — the [`Generator`]'s cache is reclaimed by value
  /// ([`Generator::into_cache`]): the *advanced* cache, holding the KV for
  /// the prefilled prompt **and** the generated tokens. It is stored as
  /// [`CacheSlot::Realised`] together with the exact token sequence it now
  /// encodes ([`CachedTokens`]): the carried-forward opaque prefix
  /// (`opaque_len` leading tokens — a builder-restored prefix, never part of
  /// `prompt_ids`) plus a *known* region of `prompt_ids` followed by the
  /// sampled tokens, the known region truncated to `offset() - opaque_len`
  /// so the bookkeeping is self-correcting (the final sampled token is
  /// sampled but never fed back into the cache, so the cache is one token
  /// shorter than `opaque + prompt + generated`; the truncation drops
  /// exactly that token). Turn N+1's [`ChatSession::decide_prefill`] uses
  /// this to feed only the new suffix.
  ///
  /// If the stream was abandoned before the standard generator finished
  /// (eos / `max_tokens`), the streaming detokenizer is **finalized** here
  /// first (Finding 3): BPE/SPM detokenizers withhold bytes until
  /// `finalize()`, so without this the recorded assistant text could be
  /// missing a tail that the cache's generated tokens *do* encode — the next
  /// turn would then render a history that no longer matches the cache.
  ///
  /// **Speculative path** — `speculative_stream_generate` consumed the caches
  /// and does not return them, so there is no advanced cache to carry. The
  /// slot is set to [`CacheSlot::SpeculativeSpent`] (not `Realised`): a
  /// freshly-rebuilt offset-0 cache does not encode the conversation, and
  /// presenting it as the current cache would let [`ChatSession::save_cache`]
  /// persist a cache that cannot restore the session (Finding 2). The next
  /// speculative turn rebuilds its caches from the [`CacheConfig`].
  ///
  /// If a speculative stream was abandoned before the driver reached
  /// eos / `max_tokens`, its inner streaming detokenizer is **finalized**
  /// here first via [`SpeculativeStream::finalize_tail`] (Finding 2): that
  /// detok is finalized only on eos / `max_tokens`, so without this the
  /// recorded assistant text could miss the tail of the last produced
  /// token — and the next speculative turn would rebuild from a truncated
  /// history.
  fn commit(&mut self) {
    if self.committed {
      return;
    }
    self.committed = true;

    match self.driver.take() {
      Some(Driver::Standard(turn)) => {
        // Finding 2 (round 3): finalize the detokenizer on ANY non-natural
        // termination — early drop AND yielded `Err`. The natural eos /
        // max_tokens paths already finalized inline (and set
        // `detok_finalized`), so they skip here; the early-drop AND
        // error-terminated paths land here with `detok_finalized = false`
        // and flush the withheld tail. Without this, an `Err`-terminated
        // BPE/SPM stream would commit a TRUNCATED assistant message — the
        // detok's `unflushed` tail (e.g. a bare-space token) is dropped
        // while its tokens are in the cache, desyncing recorded history
        // from KV state. Earlier rounds keyed on `!self.finished`, but
        // `next()` sets `finished = true` on Err too, so the Err path was
        // silently skipping this flush (Finding 2 in this round).
        if !self.detok_finalized {
          self.detok.finalize();
          self.detok_finalized = true;
          let tail = self.detok.last_segment();
          self.reply.push_str(&tail);
        }

        let cache = turn.generator.into_cache();
        // Finding 1: record exactly what the advanced cache encodes. The
        // cache holds `opaque_len` leading opaque tokens (a builder-restored
        // prefix the session never knew — NOT part of `prompt_ids`) followed
        // by a *known* region. That known region is the freshly-rendered
        // prompt followed by the sampled tokens: `prompt_ids ++ generated`.
        // Its length in the cache is `offset() - opaque_len` — clamp the
        // logical `prompt_ids ++ generated` to exactly that, which drops the
        // final sampled token (sampled but never fed back into the cache, so
        // the cache is one token short of `prompt + generated`).
        let offset = cache.first().map(|c| c.offset()).unwrap_or(0);
        let mut logical: Vec<u32> =
          Vec::with_capacity(self.prompt_ids.len() + self.generated.len());
        logical.extend_from_slice(&self.prompt_ids);
        logical.extend_from_slice(&self.generated);
        // The invariant `opaque_len + known.len() == offset()` must hold so
        // the next turn's `decide_prefill` feeds the correct suffix. The
        // known region the cache encodes has length `offset - opaque_len`;
        // the expected case is `known_len <= logical.len()` (truncate to it,
        // dropping the un-fed final token). If the cache somehow advanced
        // past everything we can name (it should not for the standard
        // `Generator` — or if `opaque_len` somehow exceeds `offset`), fall
        // back to treating the whole cache as an opaque prefix — always
        // sound: the next turn either extends it (against an empty `known`)
        // or rebuilds.
        let opaque_len = self.opaque_len.min(offset);
        let known_len = offset - opaque_len;
        let cached = if known_len <= logical.len() {
          logical.truncate(known_len);
          CachedTokens {
            opaque_len,
            known: logical,
          }
        } else {
          CachedTokens::opaque(offset)
        };
        *self.cache_slot = CacheSlot::Realised {
          cache,
          draft_cache: turn.draft_cache,
          cached,
        };
      }
      Some(Driver::Speculative(mut turn)) => {
        // Finding 2 (detok tail, round 3): the speculative driver's
        // streaming detokenizer is finalized only on eos / `max_tokens`. On
        // ANY non-natural termination — early drop OR yielded `Err` — that
        // detok still holds a withheld tail (e.g. a BPE detok's trailing
        // bare-space token) and must be flushed here so the recorded `reply`
        // is token-complete and the next speculative turn rebuilds from an
        // exact history. Earlier rounds gated on `!self.finished`, but
        // `next()` sets `finished = true` on Err too, so the error-
        // terminated path was silently skipping this flush (Finding 2).
        // `finalize_tail()` is idempotent — natural completion already set
        // `detok_finalized` so we never re-finalize there.
        if !self.detok_finalized {
          let tail = turn.iter.finalize_tail();
          self.detok_finalized = true;
          self.reply.push_str(&tail);
        }
        // Finding 2 (cache): the speculative iterator consumed the caches
        // and does not return them. Do NOT store a freshly-rebuilt offset-0
        // cache as `Realised` — it would not encode the conversation, yet
        // `save_cache` would happily persist it. Mark the slot spent:
        // `has_cache()` is `false` and `save_cache()` returns a
        // speculative-specific error. The reply was still generated
        // correctly over the consumed caches, so the history below is
        // right; only cache *reuse / save* is lost.
        *self.cache_slot = CacheSlot::SpeculativeSpent;
      }
      None => return,
    }

    // Append the assistant reply to the history. mlx-swift-lm appends the
    // turn regardless of how much was streamed (an interrupted stream still
    // records the partial reply); a turn that produced no text still closes
    // the conversation with an empty assistant message so the next prompt
    // renders a well-formed alternation.
    self
      .history
      .push(ChatMessage::assistant(std::mem::take(&mut self.reply)));
  }
}

impl Iterator for ChatResponseStream<'_> {
  type Item = Result<GenerationResponse>;

  fn next(&mut self) -> Option<Self::Item> {
    // Fused: once finished, every further poll is `None` — never a panic,
    // never a re-entry into the model (the `stream_generate` contract).
    if self.finished {
      return None;
    }

    // Pull the next token from whichever driver backs this turn. Both paths
    // converge on `(token, finish_reason_seed)`: the standard `Generator`
    // yields a `GenStep` (the loop applies its own eos / max_tokens stop on
    // top), the speculative iterator yields a fully-formed
    // `GenerationResponse` we adopt directly (it already carries the
    // detokenized text + `finish_reason`).
    match self.driver.as_mut() {
      // ---- speculative path: adopt the iterator's response verbatim ----
      Some(Driver::Speculative(turn)) => match turn.iter.next() {
        Some(Ok(spec)) => {
          let response = spec.response;
          self.produced = response.generation_tokens;
          if response.text.is_empty() {
            // nothing to fold into the reply
          } else {
            self.reply.push_str(&response.text);
          }
          if response.finish_reason.is_some() {
            // Natural termination (eos / max_tokens) — the speculative
            // driver already finalized its inner detokenizer in this step's
            // response, so `commit()` MUST NOT re-finalize (would double-
            // append). Distinct from the `Err` arm below: that one fuses
            // the stream without finalization, so `commit` still flushes.
            self.finished = true;
            self.detok_finalized = true;
          }
          Some(Ok(response))
        }
        Some(Err(e)) => {
          // Finding 2: `finished = true` here fuses the iterator, but the
          // speculative driver's detok has NOT been finalized (the inner
          // driver finalizes only on eos / max_tokens). Leave `detok_finalized
          // = false` so `commit()` flushes the withheld tail.
          self.finished = true;
          Some(Err(e))
        }
        None => {
          // The inner iterator returned None — by contract this only follows
          // a natural-termination response (eos / max_tokens), so the detok
          // is already finalized. Mirror that invariant here.
          self.finished = true;
          self.detok_finalized = true;
          None
        }
      },

      // ---- standard path: `Generator` + streaming detokenizer ----
      Some(Driver::Standard(turn)) => {
        let step = match turn.generator.next() {
          Some(Ok(step)) => step,
          Some(Err(e)) => {
            // Finding 2: `finished = true` fuses the stream, but the detok
            // has NOT been finalized — leave `detok_finalized = false` so
            // `commit()` flushes the BPE/SPM withheld tail before recording
            // the assistant message. Without this, an error-terminated
            // standard stream commits a TRUNCATED reply (history shorter
            // than the token sequence advanced into the cache).
            self.finished = true;
            return Some(Err(e));
          }
          None => {
            // `Generator::next == None` is unexpected on the standard path:
            // we yield "length" inline at `produced >= max_tokens`, so the
            // generator should not run out underneath us. Treat as an
            // unexpected stop — leave `detok_finalized = false` so `commit`
            // flushes any withheld tail.
            self.finished = true;
            return None;
          }
        };
        let token = step.token;

        // Incremental-prefill bookkeeping (Finding 1): record every sampled
        // token in order. `commit()` folds these after the rendered prompt
        // to form the next turn's cached-token sequence, clamped to the
        // cache's real `offset()` (the final sampled token is sampled but
        // not fed back into the cache, so the clamp drops it).
        self.generated.push(token);

        // mlx-lm / `stream_generate`: `if token in eos: break` BEFORE
        // `add_token` — the eos token is not detokenized; the final
        // response carries `finish_reason = "stop"` and an (empty /
        // finalized-tail) text segment.
        if self.eos.contains(&token) {
          self.finished = true;
          self.detok.finalize();
          // Detok finalized inline on natural eos — `commit()` MUST NOT
          // re-finalize (would double-append the tail to the reply).
          self.detok_finalized = true;
          let text = self.detok.last_segment();
          self.reply.push_str(&text);
          return Some(Ok(GenerationResponse {
            text,
            token,
            logprobs: step.logprobs,
            prompt_tokens: self.prompt_tokens,
            prompt_tps: 0.0,
            generation_tokens: self.produced + 1,
            generation_tps: 0.0,
            peak_memory_bytes: crate::memory::peak_memory().ok(),
            finish_reason: Some("stop".to_string()),
          }));
        }

        self.detok.add_token(token);
        self.produced += 1;
        let text = self.detok.last_segment();
        self.reply.push_str(&text);

        // mlx-lm / `stream_generate`: `if (n + 1) == max_tokens: break` —
        // a final `finish_reason = "length"` response with the finalized
        // tail. `produced` already counts this token.
        if self.produced >= self.max_tokens {
          self.finished = true;
          self.detok.finalize();
          // Detok finalized inline on natural max_tokens — `commit()` MUST
          // NOT re-finalize (would double-append the tail).
          self.detok_finalized = true;
          // `finalize()` may release a withheld tail (e.g. the BPE detok's
          // bare-space token); append it to both the yielded text and the
          // accumulated reply.
          let tail = self.detok.last_segment();
          self.reply.push_str(&tail);
          return Some(Ok(GenerationResponse {
            text: format!("{text}{tail}"),
            token,
            logprobs: step.logprobs,
            prompt_tokens: self.prompt_tokens,
            prompt_tps: 0.0,
            generation_tokens: self.produced,
            generation_tps: 0.0,
            peak_memory_bytes: crate::memory::peak_memory().ok(),
            finish_reason: Some("length".to_string()),
          }));
        }

        Some(Ok(GenerationResponse {
          text,
          token,
          logprobs: step.logprobs,
          prompt_tokens: self.prompt_tokens,
          prompt_tps: 0.0,
          generation_tokens: self.produced,
          generation_tps: 0.0,
          peak_memory_bytes: crate::memory::peak_memory().ok(),
          finish_reason: None,
        }))
      }

      None => {
        self.finished = true;
        None
      }
    }
  }
}

impl Drop for ChatResponseStream<'_> {
  /// Write the advanced cache + the assembled reply back into the session.
  ///
  /// Runs whether the stream was fully drained, partially consumed, or
  /// abandoned — so an interrupted turn still records its (partial) reply
  /// and the cache it advanced, matching the Swift `streamResponse` interrupt
  /// semantics. Idempotent (the `committed` flag), so an explicit drain
  /// followed by the implicit drop commits exactly once.
  fn drop(&mut self) {
    self.commit();
  }
}

#[cfg(test)]
mod tests {
  //! In-isolation, hand-traced [`ChatSession`] tests built on the crate's
  //! deterministic [`crate::lm::model::MockModel`] fixture and the committed
  //! `WordLevel` chat-template fixture tokenizer (`mlxrs/tests/fixtures`).
  //!
  //! `MockModel`'s argmax is its last vocab index, and `forward` advances
  //! every cache layer by the token-window length — so a turn is fully
  //! predictable (the reply is a run of the last-index token) and the cache
  //! `offset()` is an exact, observable witness of cross-turn reuse.

  use super::*;
  use crate::{lm::model::MockModel, tokenizer::StreamingDetokenizer};

  /// Load the committed fixture tokenizer (`<s> <unk> </s> hello world the
  /// quick brown fox <think> </think>`, with a `<|role|>content` chat
  /// template). Reachable from the in-crate `#[cfg(test)]` build.
  fn fixture_tokenizer() -> Tokenizer {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("tests")
      .join("fixtures");
    Tokenizer::from_path(&dir, None).expect("load fixture tokenizer")
  }

  /// A small, non-sliding cache config (matches `MockModel`'s flat KV state).
  fn cache_config() -> CacheConfig {
    CacheConfig {
      num_hidden_layers: 2,
      sliding_window: None,
    }
  }

  /// Build a fresh session: `MockModel` (vocab 11, the fixture vocab size)
  /// and the fixture tokenizer, with a small `max_tokens` for a quick
  /// deterministic run.
  fn session(max_tokens: usize) -> ChatSession {
    let cfg = GenConfig {
      max_tokens,
      ..Default::default()
    };
    ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .generate_params(cfg)
    .build()
    .expect("build")
  }

  #[test]
  fn fresh_session_has_no_cache_until_first_turn() {
    // The Swift `currentCache()` observable: `nil` before generation.
    let mut s = session(4);
    assert!(!s.has_cache(), "fresh session: cache unrealised");
    assert!(s.current_cache().is_none());
    assert!(s.history().is_empty());

    let reply = s.respond("hello").expect("respond");
    assert!(!reply.is_empty(), "MockModel produces a non-empty reply");
    assert!(s.has_cache(), "cache realised after the first turn");
  }

  #[test]
  fn multi_turn_reuses_cache_and_accumulates_history() {
    // The core contract: turn 2 reuses turn 1's cache (the cache offset
    // grows monotonically — turn 2 only prefills the *new* tokens, never
    // re-prefills turn 1), and both turns land in the history.
    let mut s = session(3);

    let _ = s.respond("hello world").expect("turn 1");
    // history: [user "hello world", assistant <reply>]
    assert_eq!(s.history().len(), 2);
    assert_eq!(s.history()[0].role, Role::User);
    assert_eq!(s.history()[1].role, Role::Assistant);

    let offset_after_turn_1 = s
      .current_cache()
      .expect("cache realised")
      .first()
      .expect(">=1 layer")
      .offset();
    assert!(offset_after_turn_1 > 0, "turn 1 advanced the cache");

    let _ = s.respond("the quick fox").expect("turn 2");
    // history: 4 messages now (two full turns).
    assert_eq!(s.history().len(), 4);
    assert_eq!(s.history()[2].role, Role::User);
    assert_eq!(s.history()[2].content, "the quick fox");
    assert_eq!(s.history()[3].role, Role::Assistant);

    let offset_after_turn_2 = s
      .current_cache()
      .expect("cache realised")
      .first()
      .expect("layer")
      .offset();
    // Monotonic growth witnesses reuse: turn 2 extended turn 1's cache
    // rather than starting from a fresh (offset-0) cache.
    assert!(
      offset_after_turn_2 > offset_after_turn_1,
      "turn 2 reused + extended the cache (offset {offset_after_turn_1} -> {offset_after_turn_2})"
    );
  }

  #[test]
  fn turn_two_prefills_only_the_new_suffix_not_the_whole_history() {
    // Finding 1 — the core incremental-prefill contract. Turn 2's cache
    // offset must grow by exactly the *new* rendered suffix (the tokens not
    // already in the cache) plus the tokens generated this turn — NOT by the
    // whole turn-2 render (which re-includes turn 1). The bug being guarded:
    // feeding the full render onto the already-advanced cache, re-appending
    // the entire prior conversation's KV.
    let max_tokens = 3;
    let mut s = session(max_tokens);

    // Turn 1.
    let _ = s.respond("hello world").expect("turn 1");
    let off1 = s.current_cache().expect("cache realised")[0].offset();
    assert!(off1 > 0, "turn 1 advanced the cache");

    // Render turn 2's full prompt the same way `stream_respond_as` will —
    // the running history now holds turn 1, so this render re-includes it.
    let (prompt2, _) = s
      .build_turn_prompt("the quick fox", Role::User)
      .expect("render turn 2");
    let full_render2 = prompt2.len();

    // The suffix actually fed to `generate_step` is `prompt2[off1..]`: the
    // tokens beyond what the cache already encodes.
    assert!(
      full_render2 > off1,
      "turn-2 render ({full_render2}) extends past the cached prefix ({off1})"
    );
    let suffix_len = full_render2 - off1;

    // Turn 2.
    let _ = s.respond("the quick fox").expect("turn 2");
    let off2 = s.current_cache().expect("cache realised")[0].offset();

    // `generate_step` over a P-token prompt advances the cache by `P - 1`
    // (prefill) + `1` (first decode step) + `max_tokens - 1` (later decode
    // steps) = `P - 1 + max_tokens`. With incremental prefill `P` is the
    // SUFFIX length, so the growth is exactly this:
    let expected_growth = suffix_len - 1 + max_tokens;
    assert_eq!(
      off2 - off1,
      expected_growth,
      "turn 2 grew the cache by the new suffix ({suffix_len}) + generated \
       ({max_tokens}) only, not the whole render"
    );

    // The decisive anti-bug assertion: the growth is strictly LESS than a
    // full re-prefill of the turn-2 render would cost. If the session were
    // re-feeding the whole conversation, `off2 - off1` would be
    // `full_render2 - 1 + max_tokens` instead.
    assert!(
      off2 - off1 < full_render2,
      "turn 2 did NOT re-prefill the whole conversation (grew {}, full \
       render is {full_render2})",
      off2 - off1
    );
  }

  #[test]
  fn instructions_change_forces_a_cache_rebuild_not_wrong_output() {
    // Finding 1 — the divergence fallback. Changing `instructions` between
    // turns makes turn 2's render NOT a prefix-extension of turn 1's cached
    // tokens (a new leading system message shifts everything). The session
    // must detect this and rebuild the cache from scratch — feeding the full
    // turn-2 render — rather than feed a bogus "suffix" onto a stale cache.
    let max_tokens = 3;
    let mut s = session(max_tokens);

    let _ = s.respond("hello").expect("turn 1");
    let off1 = s.current_cache().expect("realised")[0].offset();

    // Change the system instructions — turn 2's render gains a leading
    // `<|system|>...` block, so it no longer extends turn 1's cached prefix.
    s.set_instructions(Some("hello world the quick".to_string()));
    let (prompt2, _) = s
      .build_turn_prompt("world", Role::User)
      .expect("render turn 2");
    let full_render2 = prompt2.len();

    let _ = s.respond("world").expect("turn 2");
    let off2 = s.current_cache().expect("realised")[0].offset();

    // A rebuild re-prefills the WHOLE turn-2 render: growth from offset 0 is
    // `full_render2 - 1 + max_tokens`. (Incremental prefill is impossible
    // here — the divergence is real — so the larger cost is correct.)
    assert_eq!(
      off2,
      full_render2 - 1 + max_tokens,
      "instructions change rebuilt the cache from scratch (offset reset, \
       full render re-prefilled)"
    );
    // The rebuilt cache offset bears no relation to turn 1's — it did not
    // continue the stale cache.
    assert!(
      off2 != off1 + (full_render2 - 1 + max_tokens),
      "the stale cache was discarded, not extended"
    );
  }

  #[test]
  fn every_cache_layer_advances_in_lockstep() {
    // `make_prompt_cache` builds one cache per layer; a turn must advance
    // all of them equally (the model drives every layer each `forward`).
    let mut s = session(3);
    let _ = s.respond("hello").expect("turn");
    let cache = s.current_cache().expect("realised");
    assert_eq!(cache.len(), 2, "one cache per decoder layer");
    let off0 = cache[0].offset();
    assert!(off0 > 0);
    assert!(
      cache.iter().all(|c| c.offset() == off0),
      "all layers advance in lockstep"
    );
  }

  #[test]
  fn streaming_and_non_streaming_respond_are_consistent() {
    // `respond` is documented as "drain `stream_respond`, concatenate the
    // text" — two sessions given the identical turn must agree.
    let mut a = session(5);
    let non_streaming = a.respond("hello world").expect("non-streaming");

    let mut b = session(5);
    let mut streamed = String::new();
    {
      let stream = b.stream_respond("hello world").expect("stream");
      for resp in stream {
        streamed.push_str(&resp.expect("stream step").text);
      }
    }
    assert_eq!(
      non_streaming, streamed,
      "streaming and non-streaming respond produce the same text"
    );
    // Both sessions recorded the same history shape.
    assert_eq!(a.history().len(), b.history().len());
    assert_eq!(a.history()[1].content, b.history()[1].content);
  }

  #[test]
  fn streaming_reply_matches_recorded_history() {
    // The text yielded by the stream must equal the assistant message the
    // `Drop` write-back appends to the history.
    let mut s = session(4);
    let mut streamed = String::new();
    {
      let stream = s.stream_respond("hello").expect("stream");
      for resp in stream {
        streamed.push_str(&resp.expect("step").text);
      }
    }
    assert_eq!(s.history().len(), 2);
    assert_eq!(
      s.history()[1].content,
      streamed,
      "the recorded assistant turn equals the streamed text"
    );
  }

  #[test]
  fn finish_reason_is_length_when_max_tokens_reached() {
    // `MockModel`'s argmax (last vocab index, 10 = `</think>`) is never an
    // eos id (eos = `</s>` = 2), so generation always runs to `max_tokens`
    // — the final response must report `finish_reason = "length"`.
    let mut s = session(3);
    let mut reasons = Vec::new();
    {
      let stream = s.stream_respond("hello").expect("stream");
      for resp in stream {
        reasons.push(resp.expect("step").finish_reason);
      }
    }
    assert_eq!(reasons.last().unwrap().as_deref(), Some("length"));
    // Exactly one terminal reason; the rest are `None`.
    assert_eq!(reasons.iter().filter(|r| r.is_some()).count(), 1);
    assert_eq!(reasons.len(), 3, "max_tokens responses produced");
  }

  #[test]
  fn clear_drops_cache_and_history_keeps_instructions() {
    // The Swift `clear()`: cache + history reset, instructions preserved.
    let mut s = ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .instructions("be terse")
    .generate_params(GenConfig {
      max_tokens: 3,
      ..Default::default()
    })
    .build()
    .expect("build");

    let _ = s.respond("hello").expect("turn");
    assert!(s.has_cache());
    assert!(!s.history().is_empty());

    s.clear();
    assert!(!s.has_cache(), "clear() drops the cache");
    assert!(s.history().is_empty(), "clear() drops the history");
    assert_eq!(
      s.instructions(),
      Some("be terse"),
      "clear() preserves instructions"
    );

    // A turn after clear starts from a fresh cache (offset resets).
    let _ = s.respond("world").expect("post-clear turn");
    assert!(s.has_cache());
    assert_eq!(s.history().len(), 2, "history restarts after clear");
  }

  #[test]
  fn early_drop_of_stream_still_records_partial_turn() {
    // The Swift interrupt semantics (`testChatSessionAsyncInterrupt`): an
    // abandoned stream still commits its (partial) reply + advanced cache.
    let mut s = session(10);
    let mut streamed = String::new();
    {
      let mut stream = s.stream_respond("hello").expect("stream");
      // consume exactly one token, then drop the stream
      let first = stream.next().expect("first token").expect("ok");
      streamed.push_str(&first.text);
      assert!(first.finish_reason.is_none() || first.finish_reason.is_some());
    }
    // Drop committed: cache realised + a (partial) assistant turn recorded.
    assert!(s.has_cache(), "interrupted turn still realised the cache");
    assert_eq!(s.history().len(), 2, "interrupted turn still recorded");
    assert_eq!(s.history()[1].role, Role::Assistant);

    // Finding 3: `commit()` finalizes the detokenizer on an interrupted
    // standard stream, so the recorded reply is token-complete — it includes
    // every streamed segment (and would include any tail a BPE/SPM detok
    // withholds until `finalize()`). The recorded text must therefore start
    // with everything the stream yielded.
    assert!(
      s.history()[1].content.starts_with(&streamed),
      "recorded reply ({:?}) includes the streamed text ({streamed:?})",
      s.history()[1].content
    );

    // The session is still usable for a follow-up turn (cache reused).
    let off_before = s.current_cache().unwrap()[0].offset();
    let _ = s.respond("world").expect("follow-up turn");
    let off_after = s.current_cache().unwrap()[0].offset();
    assert!(off_after > off_before, "follow-up reused the cache");
  }

  #[test]
  fn early_drop_then_followup_does_incremental_prefill() {
    // Finding 1 + Finding 3 together: after an early-dropped standard
    // stream, the next turn must still do *incremental* prefill — feeding
    // only the new suffix. That only works if the partial reply recorded by
    // `commit()` (after `finalize()`) round-trips to exactly the tokens the
    // interrupted turn left in the cache; a token-incomplete reply would
    // make turn 2's render diverge and force a (slower, but still correct)
    // full rebuild. Asserting incremental prefill here transitively proves
    // the committed history matches the committed cache state.
    let max_tokens = 8;
    let mut s = session(max_tokens);
    {
      let mut stream = s.stream_respond("hello world").expect("stream");
      // Consume a few tokens, then abandon the stream mid-generation.
      let _ = stream.next().expect("token 1").expect("ok");
      let _ = stream.next().expect("token 2").expect("ok");
    }
    let off1 = s.current_cache().expect("realised")[0].offset();
    assert!(off1 > 0, "interrupted turn advanced the cache");

    // Render turn 2 the way `stream_respond_as` will.
    let (prompt2, _) = s
      .build_turn_prompt("the quick fox", Role::User)
      .expect("render turn 2");
    let full_render2 = prompt2.len();
    assert!(
      full_render2 > off1,
      "turn-2 render extends past the interrupted cache"
    );
    let suffix_len = full_render2 - off1;

    let _ = s.respond("the quick fox").expect("turn 2");
    let off2 = s.current_cache().expect("realised")[0].offset();

    // Incremental prefill: growth == new suffix + generated, NOT the whole
    // render. If the committed partial reply did not match the cache, turn 2
    // would diverge and rebuild (growth `full_render2 - 1 + max_tokens`).
    assert_eq!(
      off2 - off1,
      suffix_len - 1 + max_tokens,
      "follow-up after an early drop still prefilled only the new suffix"
    );
    assert!(
      off2 - off1 < full_render2,
      "follow-up did not re-prefill the whole conversation"
    );
  }

  #[test]
  fn history_seeded_session_replays_then_realises_cache() {
    // The Swift `history:` initializer: cache `nil` until the first turn,
    // then the restored messages are folded into the live history.
    let seeded = vec![ChatMessage::user("hello"), ChatMessage::assistant("world")];
    let mut s = ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .history(seeded)
    .generate_params(GenConfig {
      max_tokens: 3,
      ..Default::default()
    })
    .build()
    .expect("build");

    // `.history` state behaves like `.empty`: no cache before generation.
    assert!(!s.has_cache(), "history-seeded: cache unrealised pre-turn");
    assert!(s.history().is_empty(), "live history empty pre-turn");

    let _ = s.respond("the fox").expect("first turn");
    assert!(s.has_cache(), "cache realised after the first turn");
    // live history: 2 replayed + (user "the fox" + assistant reply) = 4.
    assert_eq!(s.history().len(), 4, "replayed history folded in");
    assert_eq!(s.history()[0].content, "hello");
    assert_eq!(s.history()[1].content, "world");
    assert_eq!(s.history()[2].content, "the fox");
    assert_eq!(s.history()[3].role, Role::Assistant);
  }

  #[test]
  fn save_cache_errors_before_any_generation() {
    // The Swift `ChatSessionError.noCacheAvailable`.
    let s = session(3);
    let path = std::env::temp_dir().join("mlxrs-l11-chat-session-nocache.safetensors");
    let err = s.save_cache(&path).expect_err("no cache yet");
    // Surfaced as a Backend error carrying the ChatSessionError message.
    assert!(
      format!("{err}").contains("no KV cache"),
      "noCacheAvailable surfaced: {err}"
    );
  }

  #[test]
  fn instructions_are_rendered_into_the_prompt() {
    // A session with instructions must prepend a system message: the
    // rendered prompt differs from an instruction-free session's, and the
    // cache offset (== prompt length on turn 1) is therefore larger.
    let with = {
      let mut s = ChatSession::builder(
        Box::new(MockModel::new(11)),
        fixture_tokenizer(),
        cache_config(),
      )
      .instructions("hello world the quick brown fox")
      .generate_params(GenConfig {
        max_tokens: 1,
        ..Default::default()
      })
      .build()
      .expect("build");
      let _ = s.respond("hello").expect("turn");
      s.current_cache().unwrap()[0].offset()
    };
    let without = {
      let mut s = session(1);
      let _ = s.respond("hello").expect("turn");
      s.current_cache().unwrap()[0].offset()
    };
    assert!(
      with > without,
      "the system instructions lengthened the prompt ({without} -> {with})"
    );
  }

  #[test]
  fn set_instructions_and_generate_params_accessors() {
    // The Swift public `instructions` / `generateParameters` getters+setters.
    let mut s = session(3);
    assert!(s.instructions().is_none());
    s.set_instructions(Some("be brief".to_string()));
    assert_eq!(s.instructions(), Some("be brief"));
    s.set_instructions(None);
    assert!(s.instructions().is_none());

    assert_eq!(s.generate_params().max_tokens, 3);
    s.generate_params_mut().max_tokens = 7;
    assert_eq!(s.generate_params().max_tokens, 7);
  }

  #[test]
  fn speculative_session_runs_multi_turn_and_accumulates_history() {
    // The optional speculative-decoding path (the Swift
    // `SpeculativeDecodingConfig`). A `MockModel` self-draft (the same
    // deterministic model as target and draft) accepts every proposed
    // token, so the turn completes; the session must accumulate the history
    // and stay usable for a second turn.
    let mut s = ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .speculative(SpeculativeDecodingConfig::new(
      Rc::new(MockModel::new(11)),
      cache_config(),
    ))
    .generate_params(GenConfig {
      max_tokens: 4,
      ..Default::default()
    })
    .build()
    .expect("build");

    assert!(
      !s.has_cache(),
      "speculative session: cache unrealised pre-turn"
    );
    let reply1 = s.respond("hello").expect("speculative turn 1");
    assert!(!reply1.is_empty(), "speculative decoding produced a reply");
    assert_eq!(s.history().len(), 2);

    // A second turn still works (the speculative path rebuilds its cache
    // each turn — a documented divergence — but the history is correct).
    let reply2 = s.respond("world").expect("speculative turn 2");
    assert!(!reply2.is_empty());
    assert_eq!(s.history().len(), 4);
    assert_eq!(s.history()[2].content, "world");
    assert_eq!(s.history()[3].role, Role::Assistant);
  }

  #[test]
  fn speculative_session_does_not_expose_a_saveable_cache() {
    // Finding 2: `speculative_stream_generate` consumes its KV caches and
    // does not return them. A speculative turn must NOT present a freshly
    // rebuilt (offset-0) cache as the current/saveable cache — that cache
    // does not encode the conversation, so `save_cache` would persist a
    // cache that cannot restore the session. `has_cache()` stays `false`
    // even after a turn, and `save_cache` returns a speculative-specific
    // error (NOT `noCacheAvailable` — the session HAS run).
    let mut s = ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .speculative(SpeculativeDecodingConfig::new(
      Rc::new(MockModel::new(11)),
      cache_config(),
    ))
    .generate_params(GenConfig {
      max_tokens: 3,
      ..Default::default()
    })
    .build()
    .expect("build");

    let _ = s.respond("hello").expect("speculative turn");
    // History accumulated, the turn ran — but no realised cache.
    assert_eq!(s.history().len(), 2, "the turn was still recorded");
    assert!(
      !s.has_cache(),
      "a speculative session never exposes a realised cache (Finding 2)"
    );
    assert!(
      s.current_cache().is_none(),
      "no current cache for a speculative session"
    );

    // save_cache fails with the speculative-specific error, distinct from
    // the pre-generation `noCacheAvailable`.
    let path = std::env::temp_dir().join("mlxrs-l11-chat-session-spec-nocache.safetensors");
    let err = s
      .save_cache(&path)
      .expect_err("speculative cache not saveable");
    let msg = format!("{err}");
    assert!(
      msg.contains("speculative"),
      "speculative-specific save error surfaced: {msg}"
    );
    assert!(
      !msg.contains("call respond"),
      "not the pre-generation noCacheAvailable error: {msg}"
    );
    assert!(
      !path.exists(),
      "no cache file was written for the speculative session"
    );
  }

  /// Build a per-layer KV cache pre-advanced to exactly `n_tokens` — the
  /// stand-in for a `ChatSessionBuilder::cache`-restored prefix whose token
  /// ids the builder was never given (`MockModel::forward` advances every
  /// layer by the token-window length). `n_tokens` must be `> 0`.
  fn prefilled_opaque_cache(n_tokens: usize) -> Vec<Box<dyn KvCache>> {
    assert!(n_tokens > 0, "an opaque cache must have a non-empty prefix");
    let model = MockModel::new(11);
    let mut cache = make_prompt_cache(&cache_config());
    let window: Vec<i32> = (0..n_tokens as i32).map(|i| i % 11).collect();
    let arr = crate::array::Array::from_slice::<i32>(&window, &(1usize, n_tokens))
      .expect("opaque-prefill token window");
    let _ = model.forward(&arr, &mut cache).expect("opaque prefill");
    assert_eq!(cache[0].offset(), n_tokens, "opaque cache pre-advanced");
    cache
  }

  /// Build a `ChatSession` restored from a `prefilled_opaque_cache`.
  fn cache_restored_session(opaque_len: usize, max_tokens: usize) -> ChatSession {
    ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .generate_params(GenConfig {
      max_tokens,
      ..Default::default()
    })
    .cache(prefilled_opaque_cache(opaque_len))
    .build()
    .expect("build")
  }

  #[test]
  fn cache_restore_short_prompt_reuses_opaque_prefix_not_rebuilds() {
    // Finding 1 — the documented `ChatSessionBuilder::cache` prefix-caching
    // path, the case the round-1 code corrupted. The restored cache holds an
    // OPAQUE prefix of `opaque_len` tokens whose ids the session never knew;
    // a turn's render NEVER re-renders that prefix. A *short* first prompt
    // (rendered shorter than `opaque_len`) must still REUSE the restored
    // cache — feeding the WHOLE new render as the suffix that continues the
    // opaque prefix — NOT rebuild from an empty cache (which would silently
    // discard the restored context).
    let max_tokens = 3;
    // `opaque_len` chosen strictly between the short and long renders below,
    // so the short render is shorter than the opaque prefix (the exact case
    // the buggy `prompt_ids.len() > opaque_len` guard rebuilt-from-empty).
    let opaque_len = 10;

    // Measure the short render the way `stream_respond_as` will (a fresh,
    // history-free session renders identically — `.cache()` adds no history).
    let p_short = session(max_tokens)
      .build_turn_prompt("hello", Role::User)
      .expect("render short prompt")
      .0
      .len();
    assert!(
      p_short < opaque_len,
      "the short render ({p_short}) must be shorter than the opaque prefix \
       ({opaque_len}) to exercise the rebuilt-from-empty bug"
    );

    let mut s = cache_restored_session(opaque_len, max_tokens);
    // A `cache:`-restored session reports a realised cache immediately.
    assert!(s.has_cache(), "cache-restored session: cache realised");

    let _ = s.respond("hello").expect("first turn over restored cache");
    let off = s.current_cache().expect("cache realised")[0].offset();

    // The restored cache continued from `opaque_len`: the WHOLE short render
    // was fed as the suffix, then `max_tokens` decode steps ran. Offset =
    // opaque + (P - 1 prefill) + max_tokens. The `-1` is the final sampled
    // token (sampled, never fed back).
    assert_eq!(
      off,
      opaque_len + p_short - 1 + max_tokens,
      "the full new render ({p_short}) was fed onto the opaque prefix \
       ({opaque_len}); offset = opaque + P - 1 + generated"
    );
    // Decisive anti-bug assertion: a rebuild-from-empty would give
    // `p_short - 1 + max_tokens` with NO `opaque_len` term — strictly less
    // than `opaque_len`. The offset exceeding `opaque_len` witnesses that
    // the restored prefix was kept, not discarded.
    assert!(
      off > opaque_len + max_tokens,
      "the restored opaque prefix was REUSED, not rebuilt-from-empty \
       (offset {off} retains the opaque {opaque_len} tokens)"
    );

    // A second turn must do incremental prefill against the now-`known`
    // region (turn-1 render + reply), proving `commit()` recorded `known`
    // correctly relative to `offset - opaque_len`.
    let (prompt2, _) = s
      .build_turn_prompt("world", Role::User)
      .expect("render turn 2");
    let full_render2 = prompt2.len();
    let _ = s.respond("world").expect("second turn");
    let off2 = s.current_cache().expect("realised")[0].offset();
    // Reuse (not rebuild): grew by only the new suffix, strictly less than a
    // full re-prefill of the turn-2 render.
    assert!(
      off2 > off && off2 - off < full_render2,
      "turn 2 reused the cache (grew {} < full render {full_render2})",
      off2 - off
    );
  }

  #[test]
  fn cache_restore_long_prompt_feeds_full_new_prompt_no_dropped_tokens() {
    // Finding 1 — the other half of the corrupted `cache:` path. When the
    // first render is LONGER than the opaque prefix, the round-1 code fed
    // `prompt_ids[opaque_len..]` — dropping the first `opaque_len` tokens of
    // the ACTUAL new prompt. The fix feeds the ENTIRE new render as the
    // suffix continuing the opaque prefix.
    let max_tokens = 4;
    let opaque_len = 10;

    let p_long = session(max_tokens)
      .build_turn_prompt("hello world the quick brown fox", Role::User)
      .expect("render long prompt")
      .0
      .len();
    assert!(
      p_long > opaque_len,
      "the long render ({p_long}) must exceed the opaque prefix ({opaque_len}) \
       to exercise the dropped-first-tokens bug"
    );

    let mut s = cache_restored_session(opaque_len, max_tokens);
    let _ = s
      .respond("hello world the quick brown fox")
      .expect("first turn over restored cache");
    let off = s.current_cache().expect("cache realised")[0].offset();

    // The FULL `p_long`-token render was fed onto the opaque prefix — NOT
    // `prompt_ids[opaque_len..]`. Offset = opaque + (p_long - 1) + max_tokens.
    assert_eq!(
      off,
      opaque_len + p_long - 1 + max_tokens,
      "the FULL new render ({p_long} tokens) was fed; no first-{opaque_len} \
       tokens dropped"
    );
    // The dropped-tokens bug would have fed only `p_long - opaque_len` tokens
    // ⇒ offset `opaque_len + (p_long - opaque_len) - 1 + max_tokens` =
    // `p_long - 1 + max_tokens`. The real offset is larger by exactly
    // `opaque_len` — the witness that no leading tokens were dropped.
    assert_eq!(
      off - (p_long - 1 + max_tokens),
      opaque_len,
      "offset retains the full opaque prefix; the dropped-tokens bug would \
       lose exactly {opaque_len} tokens"
    );
  }

  /// Build a temp-dir tokenizer whose `decoder` is `ByteLevel` (so
  /// `Tokenizer::detokenizer()` yields a real `BpeStreamingDetokenizer`) and
  /// whose vocab includes the token `"â"` — a single GPT-2 byte-encoded char
  /// for raw byte `0xE2`, a UTF-8 *lead* byte. A run of `"â"` tokens decodes
  /// to an incomplete UTF-8 sequence (`"\u{fffd}…"`), which the BPE
  /// detokenizer **withholds in its `unflushed` buffer** — released only by
  /// `finalize()`. Returns `(tokenizer, â_id, vocab_size)`.
  fn bpe_withholding_tokenizer() -> (Tokenizer, u32, usize) {
    // The committed WordLevel fixture vocab (ids 0..=10) plus `"â"` at 11.
    let a_id = 11u32;
    let tokenizer_json = json!({
      "version": "1.0",
      "truncation": Value::Null,
      "padding": Value::Null,
      "added_tokens": [
        { "id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
          "rstrip": false, "normalized": false, "special": true },
        { "id": 1, "content": "<s>", "single_word": false, "lstrip": false,
          "rstrip": false, "normalized": false, "special": true },
        { "id": 2, "content": "</s>", "single_word": false, "lstrip": false,
          "rstrip": false, "normalized": false, "special": true }
      ],
      "normalizer": Value::Null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": Value::Null,
      // The ByteLevel decoder is what `infer_detokenizer_class` keys on to
      // pick the BPE streaming detokenizer.
      "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
                   "use_regex": true },
      "model": {
        "type": "WordLevel",
        "vocab": {
          "<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4, "the": 5,
          "quick": 6, "brown": 7, "fox": 8, "<think>": 9, "</think>": 10,
          "â": a_id
        },
        "unk_token": "<unk>"
      }
    });
    let config_json = json!({
      "bos_token": "<s>",
      "eos_token": "</s>",
      "unk_token": "<unk>",
      "clean_up_tokenization_spaces": false,
      "chat_template":
        "{{ bos_token }}{% for m in messages %}{{ '<|' + m['role'] + '|>' }}\
         {{ m['content'] }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
    });

    let dir = std::env::temp_dir().join(format!("mlxrs-l11-bpe-withhold-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp tokenizer dir");
    std::fs::write(
      dir.join("tokenizer.json"),
      serde_json::to_string(&tokenizer_json).expect("serialize tokenizer.json"),
    )
    .expect("write tokenizer.json");
    std::fs::write(
      dir.join("tokenizer_config.json"),
      serde_json::to_string(&config_json).expect("serialize tokenizer_config.json"),
    )
    .expect("write tokenizer_config.json");

    let tok = Tokenizer::from_path(&dir, None).expect("load BPE-decoder tokenizer");
    // The vocab is 0..=11 (12 ids).
    (tok, a_id, 12)
  }

  #[test]
  fn speculative_interrupted_stream_flushes_detokenizer_tail() {
    // Finding 2 — an interrupted speculative turn must record token-complete
    // text. The speculative driver's streaming detokenizer is finalized only
    // on eos / `max_tokens`; a `ChatResponseStream` dropped mid-stream must
    // flush that detokenizer's withheld tail in `commit()`, or the recorded
    // assistant message loses the tail of the last produced token and the
    // next speculative turn rebuilds from a truncated history.
    let (tok, a_id, vocab) = bpe_withholding_tokenizer();
    // A `MockModel` whose argmax is `â` (id 11): every produced token decodes
    // to byte 0xE2, so the BPE detok withholds the whole reply in `unflushed`
    // until `finalize()` (`last_segment()` sees an empty `text` meanwhile).
    let mut canned = vec![0.0_f32; vocab];
    canned[a_id as usize] = 10.0;
    let target = MockModel {
      canned: canned.clone(),
      n_kv_heads: 1,
      head_dim: 2,
    };
    let draft = MockModel {
      canned,
      n_kv_heads: 1,
      head_dim: 2,
    };

    let mut s = ChatSession::builder(Box::new(target), tok, cache_config())
      .speculative(SpeculativeDecodingConfig::new(
        Rc::new(draft),
        cache_config(),
      ))
      .generate_params(GenConfig {
        max_tokens: 12,
        ..Default::default()
      })
      .build()
      .expect("build");

    // Drain only a few tokens, then drop the stream mid-generation.
    let mut produced_tokens: Vec<u32> = Vec::new();
    let mut streamed = String::new();
    {
      let mut stream = s.stream_respond("hello").expect("speculative stream");
      for _ in 0..3 {
        let r = stream.next().expect("token").expect("ok");
        produced_tokens.push(r.token);
        streamed.push_str(&r.text);
      }
      // `stream` dropped here mid-generation (finish_reason never seen).
    }
    assert_eq!(produced_tokens.len(), 3, "drained 3 tokens before drop");
    assert!(
      produced_tokens.iter().all(|&t| t == a_id),
      "the mock samples the withheld `â` token every step"
    );

    // The committed assistant turn.
    assert_eq!(s.history().len(), 2, "interrupted turn still recorded");
    let recorded = &s.history()[1].content;

    // Token-complete oracle: feed the SAME produced tokens into an
    // independent BPE detokenizer and `finalize()` — the committed history
    // text must equal that full detokenization. Without the Finding-2 fix
    // the recorded text would be MISSING the withheld tail (the BPE detok
    // buffers every `â` in `unflushed` until `finalize()`), so `recorded`
    // would be the empty mid-stream `text` instead.
    let reference = {
      let mut d = crate::tokenizer::BpeStreamingDetokenizer::new(
        // `(token_string, id)` vocab pairs — only `â` is produced, but a
        // faithful detok is built over the full vocab.
        vec![("â".to_string(), a_id)],
        false,
      );
      for &t in &produced_tokens {
        d.add_token(t);
      }
      d.finalize();
      d.last_segment()
    };
    assert_eq!(
      *recorded, reference,
      "the interrupted speculative turn recorded token-complete text \
       (detokenizer tail flushed): recorded {recorded:?} == finalized {reference:?}"
    );
    // The withheld tail is non-empty here, so the fix is load-bearing: the
    // recorded text is strictly longer than what the stream yielded
    // mid-flight (every mid-stream `last_segment()` returned "").
    assert!(
      !reference.is_empty() && recorded.len() > streamed.len(),
      "the BPE detok genuinely withheld a tail that `commit()` flushed \
       (streamed {streamed:?}, recorded {recorded:?})"
    );
  }

  /// A model that delegates every call to `inner` for the first `ok_calls`
  /// `forward` invocations and then returns a [`Error::Backend`] on every
  /// later one. Used to drive a [`ChatSession`] stream into the `Err` branch
  /// of [`Iterator::next`] partway through generation so the round-3 Finding 2
  /// (detokenizer not finalized on an error-terminated stream) is exercised.
  ///
  /// `forward` is `&self`, so the call counter sits behind a [`Cell`]
  /// ([`MockModel`] is single-thread). The optional `forward_embeddings` hook
  /// just delegates — the LM path only uses `forward`.
  struct ErringAfterModel {
    inner: MockModel,
    ok_calls: std::cell::Cell<usize>,
  }

  impl ErringAfterModel {
    fn new(inner: MockModel, ok_calls: usize) -> Self {
      Self {
        inner,
        ok_calls: std::cell::Cell::new(ok_calls),
      }
    }
  }

  impl Model for ErringAfterModel {
    fn forward(
      &self,
      tokens: &crate::array::Array,
      cache: &mut [Box<dyn KvCache>],
    ) -> Result<crate::array::Array> {
      let remaining = self.ok_calls.get();
      if remaining == 0 {
        return Err(Error::Backend {
          message: "ErringAfterModel: budget exhausted (test fixture)".into(),
        });
      }
      self.ok_calls.set(remaining - 1);
      self.inner.forward(tokens, cache)
    }

    fn forward_embeddings(
      &self,
      embeddings: &crate::array::Array,
      cache: &mut [Box<dyn KvCache>],
    ) -> Result<crate::array::Array> {
      self.inner.forward_embeddings(embeddings, cache)
    }
  }

  #[test]
  fn build_rejects_cache_plus_speculative_combination() {
    // Finding 1 (round 3) — the `.cache(restored).speculative(..)` builder
    // combination MUST be rejected at `build()` time, with no session
    // constructed. The opaque cache cannot be re-rendered from history (its
    // token ids are unknown to the session), and `speculative_stream_generate`
    // consumes its KV caches and does not return them — so the first
    // speculative turn would use the restored prefix and the next turn would
    // silently fall back to a fresh offset-0 cache, losing the prefix without
    // warning. We reject up-front with an actionable message pointing at
    // either workaround (drop one of `.cache(..)` / `.speculative(..)`).
    let restored = prefilled_opaque_cache(8);
    let res = ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .cache(restored)
    .speculative(SpeculativeDecodingConfig::new(
      Rc::new(MockModel::new(11)),
      cache_config(),
    ))
    .build();

    let err = match res {
      Ok(_) => panic!("cache + speculative must be rejected at build()"),
      Err(e) => e,
    };
    let msg = format!("{err}");
    // Decisive: the error names the unsupported combination AND points at
    // BOTH alternatives, so the caller can act on it without reading source.
    assert!(
      msg.contains("speculative") && msg.contains("cache"),
      "error names the rejected combination: {msg}"
    );
    assert!(
      msg.contains(".cache") && msg.contains(".speculative"),
      "error points at both workarounds: {msg}"
    );

    // Reverse builder order: `.speculative(..)` first then `.cache(..)` — the
    // rejection is order-independent (both fields are set at `build()` time).
    let restored2 = prefilled_opaque_cache(8);
    let res2 = ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .speculative(SpeculativeDecodingConfig::new(
      Rc::new(MockModel::new(11)),
      cache_config(),
    ))
    .cache(restored2)
    .build();
    assert!(
      res2.is_err(),
      "rejection is independent of builder-method order"
    );

    // Sanity-check the same combination minus `.cache(..)` builds fine — the
    // rejection is precisely scoped to the unsupported pair, not a regression
    // of the lone `.speculative(..)` path.
    let ok = ChatSession::builder(
      Box::new(MockModel::new(11)),
      fixture_tokenizer(),
      cache_config(),
    )
    .speculative(SpeculativeDecodingConfig::new(
      Rc::new(MockModel::new(11)),
      cache_config(),
    ))
    .build();
    assert!(
      ok.is_ok(),
      ".speculative(..) alone still builds (only the cache+speculative combo is unsupported)"
    );
  }

  #[test]
  fn standard_error_terminated_stream_flushes_detokenizer_tail() {
    // Finding 2 (round 3) — the round-3 defect on the STANDARD path. An
    // error-terminated stream (the `Generator` yields `Err`) used to skip
    // `commit()`'s finalize because the gate was `!self.finished`, but
    // `next()` sets `finished = true` on BOTH natural termination AND `Err`.
    // The fix splits natural completion (which already finalized) from
    // error/early-drop termination (which has NOT) via `detok_finalized`, so
    // an `Err`-terminated BPE/SPM stream now flushes the withheld tail and
    // records token-complete history.
    let (tok, a_id, vocab) = bpe_withholding_tokenizer();
    // Argmax samples `â` (id 11) every step — every produced token decodes to
    // byte 0xE2, so the BPE detok withholds the whole reply in `unflushed`
    // until `finalize()` (mid-stream `last_segment()` returns "").
    let mut canned = vec![0.0_f32; vocab];
    canned[a_id as usize] = 10.0;
    let inner = MockModel {
      canned,
      n_kv_heads: 1,
      head_dim: 2,
    };
    // Allow `ok_steps` successful `forward` calls (1 prefill + decode steps),
    // then return Err on every later one — so the stream's standard generator
    // yields a few `Ok(GenStep)`s and then an `Err`, fusing with
    // `finished = true` but `detok_finalized = false`. Pre-fix, `commit()`
    // would skip the flush and the assistant message would be empty (every
    // mid-stream segment was "").
    let ok_steps: usize = 4;
    let model = ErringAfterModel::new(inner, ok_steps);

    let mut s = ChatSession::builder(Box::new(model), tok, cache_config())
      .generate_params(GenConfig {
        // `max_tokens` set high so the error fires before the natural-finish
        // branch can set `detok_finalized = true`.
        max_tokens: 32,
        ..Default::default()
      })
      .build()
      .expect("build");

    // Drain the stream — collect every token actually produced (the run ends
    // when the wrapped model returns `Err`).
    let mut produced_tokens: Vec<u32> = Vec::new();
    let mut streamed = String::new();
    let mut saw_err = false;
    {
      let stream = s.stream_respond("hello").expect("stream");
      for resp in stream {
        match resp {
          Ok(r) => {
            produced_tokens.push(r.token);
            streamed.push_str(&r.text);
          }
          Err(_) => {
            saw_err = true;
            break;
          }
        }
        // `break` on Err — the explicit drain ends, the stream is then
        // dropped (its `commit()` must flush the tail).
      }
    }
    assert!(saw_err, "the stream MUST yield an Err mid-generation");
    assert!(
      !produced_tokens.is_empty(),
      "at least one token must have streamed before the Err"
    );
    assert!(
      produced_tokens.iter().all(|&t| t == a_id),
      "every sampled token is the withheld `â`"
    );

    // The committed assistant turn.
    assert_eq!(s.history().len(), 2, "error-terminated turn still recorded");
    let recorded = &s.history()[1].content;

    // Token-complete oracle: feed the SAME produced tokens into an
    // independent BPE detokenizer and `finalize()` — the committed history
    // text must equal that full detokenization. Pre-fix (gating on
    // `!self.finished`), `commit()` would have skipped the flush on the Err
    // path: `recorded` would equal `streamed`, which is empty (the BPE detok
    // buffered every `â` in `unflushed`).
    let reference = {
      let mut d =
        crate::tokenizer::BpeStreamingDetokenizer::new(vec![("â".to_string(), a_id)], false);
      for &t in &produced_tokens {
        d.add_token(t);
      }
      d.finalize();
      d.last_segment()
    };
    assert_eq!(
      *recorded, reference,
      "the error-terminated standard turn recorded token-complete text \
       (detok tail flushed in commit): recorded {recorded:?} == finalized {reference:?}"
    );
    // Load-bearing: the BPE detok genuinely withheld a tail — pre-fix the
    // recorded text would have been empty (`streamed.is_empty()`).
    assert!(
      !reference.is_empty() && recorded.len() > streamed.len(),
      "the BPE detok genuinely withheld a tail commit() flushed \
       (streamed {streamed:?}, recorded {recorded:?})"
    );
  }

  #[test]
  fn speculative_error_terminated_stream_flushes_detokenizer_tail() {
    // Finding 2 (round 3) — the same round-3 defect on the SPECULATIVE
    // path. An `Err` from the speculative iterator sets `finished = true`
    // without finalizing the inner driver's detok; pre-fix `commit()` keyed
    // on `!self.finished` and so skipped the flush. The fix: `commit()` now
    // gates on `detok_finalized`, which the Err arm intentionally leaves
    // `false`, so `finalize_tail()` runs and flushes the withheld tail.
    let (tok, a_id, vocab) = bpe_withholding_tokenizer();
    let mut canned = vec![0.0_f32; vocab];
    canned[a_id as usize] = 10.0;

    // Wrap BOTH target and draft in `ErringAfterModel`: the speculative
    // driver calls both each cycle, so erring either eventually surfaces as
    // an `Err` step from the iterator. The driver's call shape is
    // `1 + n_draft_tokens` draft forwards per cycle and `1 + 1` target
    // forwards per cycle (1 prefill + 1 verify), so a `draft.ok_calls` budget
    // sized to allow ~1 full cycle (1 prefill + 5 propose) plus a partial
    // second-cycle draft propose lets the iterator emit up to a full burst of
    // tokens (~6) and then `Err` in the next cycle's draft phase. The target
    // budget is set high so the error is sourced from the draft (easier to
    // reason about: target stays consistent for stream bookkeeping).
    let target_inner = MockModel {
      canned: canned.clone(),
      n_kv_heads: 1,
      head_dim: 2,
    };
    let draft_inner = MockModel {
      canned,
      n_kv_heads: 1,
      head_dim: 2,
    };
    // target survives the whole run — the Err comes from draft.
    let target = ErringAfterModel::new(target_inner, 64);
    // draft: 1 prefill + 5 propose = 6 (one full cycle), then a partial 2nd
    // cycle: 1 propose succeeds, next propose Errs → ≥1 burst of tokens
    // already streamed when the Err arrives.
    let draft = ErringAfterModel::new(draft_inner, 7);

    let mut s = ChatSession::builder(Box::new(target), tok, cache_config())
      .speculative(SpeculativeDecodingConfig::new(
        Rc::new(draft),
        cache_config(),
      ))
      .generate_params(GenConfig {
        max_tokens: 32,
        ..Default::default()
      })
      .build()
      .expect("build");

    // Drain the stream until an `Err` is yielded; then drop.
    let mut produced_tokens: Vec<u32> = Vec::new();
    let mut streamed = String::new();
    let mut saw_err = false;
    {
      let stream = s.stream_respond("hello").expect("speculative stream");
      for resp in stream {
        match resp {
          Ok(r) => {
            produced_tokens.push(r.token);
            streamed.push_str(&r.text);
          }
          Err(_) => {
            saw_err = true;
            break;
          }
        }
      }
    }
    assert!(
      saw_err,
      "the speculative stream MUST yield an Err mid-generation"
    );
    assert!(
      !produced_tokens.is_empty(),
      "at least one token must have streamed before the speculative Err"
    );
    assert!(
      produced_tokens.iter().all(|&t| t == a_id),
      "every sampled token is the withheld `â`"
    );

    // Committed assistant turn must equal the independent finalized detok of
    // the tokens actually produced. Pre-fix `recorded` would equal the empty
    // `streamed` — every mid-stream `last_segment()` returned "" because the
    // BPE detok buffered every `â`.
    assert_eq!(s.history().len(), 2, "error-terminated turn still recorded");
    let recorded = &s.history()[1].content;

    let reference = {
      let mut d =
        crate::tokenizer::BpeStreamingDetokenizer::new(vec![("â".to_string(), a_id)], false);
      for &t in &produced_tokens {
        d.add_token(t);
      }
      d.finalize();
      d.last_segment()
    };
    assert_eq!(
      *recorded, reference,
      "the error-terminated speculative turn recorded token-complete text: \
       recorded {recorded:?} == finalized {reference:?}"
    );
    assert!(
      !reference.is_empty() && recorded.len() > streamed.len(),
      "speculative BPE detok genuinely withheld a tail commit() flushed \
       (streamed {streamed:?}, recorded {recorded:?})"
    );
  }
}
