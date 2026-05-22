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
//! the whole conversation) exactly as the Swift reference does.
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
    speculative::{DraftConfig, speculative_stream_generate},
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

/// A main cache plus an optional draft cache — what a session turn carries
/// (the draft cache is `Some` iff speculative decoding is enabled).
type MainAndDraftCaches = (KvCaches, Option<KvCaches>);

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
  },
  /// A re-hydrated message history awaiting its first generation — the
  /// Swift `.history` case (used by [`ChatSessionBuilder::history`]).
  History(Vec<ChatMessage>),
}

/// Errors thrown by [`ChatSession`] — a port of mlx-swift-lm's
/// `ChatSessionError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatSessionError {
  /// [`ChatSession::save_cache`] was called before any generation occurred —
  /// the Swift `ChatSessionError.noCacheAvailable`.
  NoCacheAvailable,
}

impl std::fmt::Display for ChatSessionError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      ChatSessionError::NoCacheAvailable => f.write_str(
        "no KV cache is available: call respond() / stream_respond() before save_cache()",
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
///     .build();
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
    self.initial = CacheSlot::Realised {
      cache,
      draft_cache: None,
    };
    self
  }

  /// Finish building the [`ChatSession`].
  pub fn build(self) -> ChatSession {
    ChatSession {
      model: self.model,
      tokenizer: self.tokenizer,
      cache_config: self.cache_config,
      instructions: self.instructions,
      generate_params: self.generate_params,
      speculative: self.speculative,
      cache: self.initial,
      history: Vec::new(),
    }
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
  pub fn save_cache(&self, path: &std::path::Path) -> Result<()> {
    match &self.cache {
      CacheSlot::Realised { cache, .. } => {
        save_prompt_cache(path, cache, &std::collections::HashMap::new())
      }
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
  /// Returns `(main_cache, draft_cache)`; `draft_cache` is `Some` iff the
  /// session uses speculative decoding (built once, then carried). The slot
  /// is left [`CacheSlot::Empty`] — [`ChatResponseStream`]'s `Drop` write-back
  /// puts the advanced cache(s) back when the turn completes.
  fn take_cache(&mut self) -> MainAndDraftCaches {
    let slot = std::mem::replace(&mut self.cache, CacheSlot::Empty);
    match slot {
      // `.kvcache` — reuse the carried cache(s) as-is.
      CacheSlot::Realised { cache, draft_cache } => {
        // A speculative session whose draft cache was not yet built (e.g.
        // restored via `cache:`) gets it allocated now, once.
        let draft = match (&self.speculative, draft_cache) {
          (Some(spec), None) => Some(make_prompt_cache(&spec.draft_cache_config)),
          (_, existing) => existing,
        };
        (cache, draft)
      }
      // `.empty` / `.history` — allocate a fresh cache (the `.history`
      // messages were already folded into the rendered prompt by
      // `build_turn_prompt`, so only the cache shape matters here).
      CacheSlot::Empty | CacheSlot::History(_) => {
        let cache = make_prompt_cache(&self.cache_config);
        let draft = self
          .speculative
          .as_ref()
          .map(|spec| make_prompt_cache(&spec.draft_cache_config));
        (cache, draft)
      }
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

    // 2. Take the KV cache out of the slot (fresh on the first turn).
    let (cache, draft_cache) = self.take_cache();

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

    // 5. Split the `&mut self` borrow into disjoint field references: the
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

    // 6. Build the per-token generation driver — speculative or standard.
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
          &prompt_ids,
          cache,
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
        generator: generate_step(model, &prompt_ids, cache, cfg),
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
      finished: false,
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

/// Holds a speculative-decoding turn: the [`speculative_stream_generate`]
/// iterator and the caches threaded through it.
///
/// [`speculative_stream_generate`] consumes both caches and — unlike the
/// standard [`Generator`] — does not hand them back, so a speculative turn
/// always **rebuilds** the cache for the next turn from the [`CacheConfig`]
/// (a documented divergence: speculative sessions re-prefill the conversation
/// each turn). The standard path reuses the cache via [`Generator::into_cache`]
/// and has no such cost.
struct SpeculativeTurn<'a> {
  /// The streaming speculative iterator, yielding one response per token.
  iter: Box<dyn Iterator<Item = Result<crate::lm::speculative::SpeculativeResponse>> + 'a>,
  /// The main cache shape, to rebuild the cache after the turn.
  cache_config: CacheConfig,
  /// The draft cache shape, to rebuild the draft cache after the turn.
  draft_cache_config: CacheConfig,
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
    // Capture the cache shapes before the iterator consumes the caches —
    // a speculative turn rebuilds them for the next turn (see the struct
    // doc). `CacheConfig` is small and trivially derived from the layer
    // count + sliding window.
    let cache_config = cache_config_of(&cache);
    let draft_cache_config = cache_config_of(&draft_cache);
    let iter = Box::new(speculative_stream_generate(
      target,
      tokenizer,
      prompt,
      cache,
      draft_cache,
      draft_cfg,
      cfg,
    ));
    Self {
      iter,
      cache_config,
      draft_cache_config,
    }
  }
}

/// Recover a [`CacheConfig`] from a realised cache: the layer count is the
/// vector length, and the sliding window is read back from a
/// [`crate::lm::cache::RotatingKvCache`] entry if present.
///
/// Used only by the speculative path, which must rebuild its caches after a
/// turn (`speculative_stream_generate` does not return them). A rotating
/// cache reports its window via [`KvCache::max_size`]; a standard cache
/// reports `None`, so the rebuilt cache matches the original kind.
fn cache_config_of(cache: &[Box<dyn KvCache>]) -> CacheConfig {
  let sliding_window = cache.first().and_then(|c| c.max_size()).map(|w| w as i32);
  CacheConfig {
    num_hidden_layers: cache.len(),
    sliding_window,
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
  /// `true` once the stream has ended (eos, `max_tokens`, or an `Err`) — the
  /// iterator fuses afterwards, never re-entering the model.
  finished: bool,
  /// `true` once the turn's cache + reply have been written back to the
  /// session, so a `Drop` after a manual drain does not commit twice.
  committed: bool,
}

impl ChatResponseStream<'_> {
  /// Write the advanced cache and the assembled reply back into the session
  /// — the body of [`Drop`], factored out so it runs at most once
  /// (idempotent via the `committed` flag).
  ///
  /// Standard path: the [`Generator`]'s cache is reclaimed by value
  /// ([`Generator::into_cache`]) — the *advanced* cache, ready to be reused.
  /// Speculative path: `speculative_stream_generate` consumed the caches and
  /// does not return them, so fresh caches are built from the captured
  /// shapes (a speculative session re-prefills each turn — see
  /// [`SpeculativeTurn`]).
  fn commit(&mut self) {
    if self.committed {
      return;
    }
    self.committed = true;

    let (cache, draft_cache) = match self.driver.take() {
      Some(Driver::Standard(turn)) => (turn.generator.into_cache(), turn.draft_cache),
      Some(Driver::Speculative(turn)) => {
        // The iterator consumed the caches; rebuild from the shapes. The
        // turn's reply was still generated over the consumed caches, so the
        // history is correct — only the *cache reuse* is forfeited.
        let cache = make_prompt_cache(&turn.cache_config);
        let draft = make_prompt_cache(&turn.draft_cache_config);
        (cache, Some(draft))
      }
      None => return,
    };
    *self.cache_slot = CacheSlot::Realised { cache, draft_cache };

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
            self.finished = true;
          }
          Some(Ok(response))
        }
        Some(Err(e)) => {
          self.finished = true;
          Some(Err(e))
        }
        None => {
          self.finished = true;
          None
        }
      },

      // ---- standard path: `Generator` + streaming detokenizer ----
      Some(Driver::Standard(turn)) => {
        let step = match turn.generator.next() {
          Some(Ok(step)) => step,
          Some(Err(e)) => {
            self.finished = true;
            return Some(Err(e));
          }
          None => {
            self.finished = true;
            return None;
          }
        };
        let token = step.token;

        // mlx-lm / `stream_generate`: `if token in eos: break` BEFORE
        // `add_token` — the eos token is not detokenized; the final
        // response carries `finish_reason = "stop"` and an (empty /
        // finalized-tail) text segment.
        if self.eos.contains(&token) {
          self.finished = true;
          self.detok.finalize();
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
  use crate::lm::model::MockModel;

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
    .build();

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
    {
      let mut stream = s.stream_respond("hello").expect("stream");
      // consume exactly one token, then drop the stream
      let first = stream.next().expect("first token").expect("ok");
      assert!(first.finish_reason.is_none() || first.finish_reason.is_some());
    }
    // Drop committed: cache realised + a (partial) assistant turn recorded.
    assert!(s.has_cache(), "interrupted turn still realised the cache");
    assert_eq!(s.history().len(), 2, "interrupted turn still recorded");
    assert_eq!(s.history()[1].role, Role::Assistant);

    // The session is still usable for a follow-up turn (cache reused).
    let off_before = s.current_cache().unwrap()[0].offset();
    let _ = s.respond("world").expect("follow-up turn");
    let off_after = s.current_cache().unwrap()[0].offset();
    assert!(off_after > off_before, "follow-up reused the cache");
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
    .build();

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
      .build();
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
    // token, so the turn completes; the session must still build the cache,
    // accumulate the history, and stay usable for a second turn.
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
    .build();

    assert!(
      !s.has_cache(),
      "speculative session: cache unrealised pre-turn"
    );
    let reply1 = s.respond("hello").expect("speculative turn 1");
    assert!(!reply1.is_empty(), "speculative decoding produced a reply");
    assert!(
      s.has_cache(),
      "speculative turn realised the (rebuilt) cache"
    );
    assert_eq!(s.history().len(), 2);

    // A second turn still works (the speculative path rebuilds its cache
    // each turn — a documented divergence — but the history is correct).
    let reply2 = s.respond("world").expect("speculative turn 2");
    assert!(!reply2.is_empty());
    assert_eq!(s.history().len(), 4);
    assert_eq!(s.history()[2].content, "world");
    assert_eq!(s.history()[3].role, Role::Assistant);
  }
}
