//! Local jsonl-backed fine-tuning datasets — the data side of mlx-lm
//! `mlx_lm/tuner/datasets.py` (lines `1..=219`, `309..=332`), cross-referenced
//! against mlx-swift-lm `MLXLLM/Lora+Data.swift`'s jsonl loader.
//!
//! # Surface
//!
//! Each dataset type holds a `Vec<serde_json::Value>` (one JSON object per
//! parsed jsonl line) plus a borrowed [`Tokenizer`] and the per-type config
//! scalars, mirroring the Python `__init__` shapes:
//!
//! - [`TextDataset`] (Python `tuner/datasets.py:11..=36`) — each line has a
//!   `"text"` field (or a user-overridden `text_key`); [`Dataset::process`]
//!   returns `(tokenizer.encode(text) + [eos], 0)` — the full sequence is the
//!   loss target.
//! - [`ChatDataset`] (Python `tuner/datasets.py:39..=83`) — each line has a
//!   `"messages"` array (HF chat format) plus optional `"tools"`;
//!   [`Dataset::process`] runs `tokenizer.apply_chat_template(messages, tools)`
//!   and (when `mask_prompt`) returns the **prefix-length** as the loss-mask
//!   `offset` so the trainer can ignore everything before the final assistant
//!   message.
//! - [`CompletionsDataset`] (Python `tuner/datasets.py:86..=133`) — each line
//!   has a `"prompt"` + `"completion"` pair (or user-overridden keys);
//!   [`Dataset::process`] renders the two as a two-message chat
//!   (`user`+`assistant`) so the rendering goes through the tokenizer's chat
//!   template, and (when `mask_prompt`) returns the prompt-prefix length as
//!   the `offset`.
//! - [`ConcatenatedDataset`] (Python `tuner/datasets.py:136..=155`) — wraps a
//!   `Vec<Box<dyn Dataset>>` and indexes ACROSS the inner datasets, routing
//!   `__getitem__`/`process` to whichever inner dataset owns the index. This
//!   is **NOT** sequence packing (the Python type does not pack to a fixed
//!   length; the spec's "packed batches" phrasing is a misnomer); it is a
//!   plain concat-by-index, exactly as the Python class.
//! - [`CacheDataset`] (Python `tuner/datasets.py:158..=172`) — memoizes the
//!   per-index `process()` result the FIRST time an index is touched. Python
//!   keeps the cache **in-memory per instance** (`self._proc_data = [None] *
//!   len(data)`), NOT in a sidecar `.cache` file; this port mirrors that
//!   exactly. A "source mtime change" therefore invalidates the cache via the
//!   natural mechanism: the next [`load_dataset`] call constructs a fresh
//!   [`CacheDataset`] whose `_proc_data` starts empty (see
//!   [the cache-invalidation test](#cache-dataset-invalidates-on-source-mtime-change)).
//!
//! And the file-path entry point:
//!
//! - [`load_dataset`] (Python `tuner/datasets.py:205..=219`,
//!   `309..=332`) — reads a local `.jsonl` file, auto-detects the dataset
//!   type from the first record's shape (Python's `create_dataset`), and
//!   wraps it in a [`CacheDataset`] (the typical training-time wrapper, as
//!   `tuner/trainer.py` does).
//!
//! # Loss-mask convention — `offset`
//!
//! Both Python and this port carry the mask as a SINGLE `usize` offset (not a
//! per-token `Vec<bool>`): tokens at positions `[0, offset)` are the prompt
//! prefix and excluded from the training loss; tokens at `[offset, len)` are
//! the completion and contribute to the loss. `offset == 0` means "no
//! masking" (the entire sequence is the loss target). `offset == tokens.len()`
//! would mask the entire sequence (zero loss) and is degenerate — never
//! produced by the canonical paths.
//!
//! The spec's `(token_ids, loss_mask: Vec<bool>)` per-token-bool framing is a
//! misnomer; the Python reference uses `(tokens, offset)` everywhere and the
//! training loop builds the bool mask from the offset. Mirroring the Python
//! shape keeps the data flat, avoids `Vec<bool>` allocations per example, and
//! preserves bit-for-bit parity with the upstream trainer's expectations.
//!
//! # Scope boundary
//!
//! - HuggingFace Hub datasets (`load_hf_dataset`, `load_custom_hf_dataset`)
//!   are **excluded** per the project's local-only policy — see
//!   [`load_dataset`]'s `hf://`-path rejection. Mirrors the same fence
//!   already applied in [`crate::lm::lora`] and [`crate::lm::factory`].
//! - The training-loop side (`tuner/trainer.py`) is blocked on autograd
//!   (the A4 milestone); this module ships the data side only.
//! - Per-model arch hooks are out of scope — see the project memory rule on
//!   no per-model arch porting.
//!
//! # Conventions
//!
//! - [`Result`]-fallible everywhere; recoverable IO / JSON / shape failures
//!   map to [`Error::Backend`] / [`Error::Tokenizer`] / [`Error::ShapeMismatch`]
//!   with clear messages.
//! - The datasets themselves are `Send` (they hold only owned
//!   [`serde_json::Value`]s and immutable borrows of the [`Tokenizer`]) — no
//!   `Array` handle is touched on this side of the M3 split.
//!
//! [`Error::Backend`]: crate::Error::Backend
//! [`Error::Tokenizer`]: crate::Error::Tokenizer
//! [`Error::ShapeMismatch`]: crate::Error::ShapeMismatch

use std::{
  cell::RefCell,
  io::{BufRead, BufReader},
  path::Path,
};

use serde_json::Value;

use crate::{
  error::{Error, Result},
  tokenizer::Tokenizer,
};

// ───────────────────────────── defaults ─────────────────────────────

/// Default jsonl field name for [`TextDataset`] — Python
/// `tuner/datasets.py:20` (`text_key: str = "text"`). Also the default
/// `text_feature` in `create_dataset` (Python `tuner/datasets.py:182`).
pub const DEFAULT_TEXT_KEY: &str = "text";

/// Default jsonl field name for [`ChatDataset`] — Python
/// `tuner/datasets.py:49` (`chat_key: str = "messages"`). Also the default
/// `chat_feature` in `create_dataset` (Python `tuner/datasets.py:184`).
pub const DEFAULT_CHAT_KEY: &str = "messages";

/// Default jsonl field name for [`CompletionsDataset`]'s prompt — Python
/// `tuner/datasets.py:181` (`prompt_feature: str = "prompt"`).
pub const DEFAULT_PROMPT_KEY: &str = "prompt";

/// Default jsonl field name for [`CompletionsDataset`]'s completion —
/// Python `tuner/datasets.py:183` (`completion_feature: str = "completion"`).
pub const DEFAULT_COMPLETION_KEY: &str = "completion";

/// Upper bound on the bytes [`load_dataset`] will read off a single jsonl
/// file. A training set CAN legitimately be many MiB; this is a defense
/// against an untrusted path that maps a multi-GiB blob, similar in spirit
/// to (but generous beyond) [`crate::lm::lora::MAX_ADAPTER_SAFETENSORS_BYTES`].
/// At 2 GiB we accommodate even very large jsonl shards while still bounding
/// an obviously hostile mount.
pub const MAX_DATASET_FILE_BYTES: u64 = 2 << 30;

// ───────────────────────────── trait ─────────────────────────────

/// A processed dataset example: `(token_ids, mask_offset)`.
///
/// Tokens at positions `[0, mask_offset)` are the prompt prefix and excluded
/// from the loss; tokens at `[mask_offset, len)` are the completion. See the
/// [module-level note on the offset convention](self#loss-mask-convention--offset).
pub type Example = (Vec<u32>, usize);

/// A pre-tokenization dataset of `(token_ids, mask_offset)` examples.
///
/// Mirrors the duck-typed Python `Dataset`-shaped object the trainer reads:
/// `len()`, `__getitem__(idx)` (here [`Dataset::get`] returning the raw
/// per-line JSON), and `process(record)` (here [`Dataset::process`] taking
/// the index directly — Python calls `data[idx]` then `data.process(...)`,
/// which this port collapses into a single index-keyed entry to keep the
/// trait `dyn`-safe and to keep all token-id ownership inside the dataset).
///
/// `process(idx)` returns `(tokens, offset)` — see the [module-level note
/// on the offset convention](self#loss-mask-convention--offset).
pub trait Dataset {
  /// Number of examples (Python `__len__`,
  /// `tuner/datasets.py:35,82,132,154,171`).
  fn len(&self) -> usize;

  /// Is the dataset empty? (`len() == 0`)
  fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// The raw per-line JSON at index `idx` (Python `__getitem__`,
  /// `tuner/datasets.py:32,79,129,141,166`). Used by
  /// [`ConcatenatedDataset`]'s routing and by [`CacheDataset`] as the
  /// argument to a wrapped inner [`Dataset::process`].
  fn get(&self, idx: usize) -> Result<&Value>;

  /// Tokenize-and-mask the per-index example, returning
  /// `(tokens, mask_offset)`.
  ///
  /// Mirrors Python `tuner/datasets.py` per-type `process(d)`. Errors are
  /// [`Error::Backend`] (missing/wrong-typed jsonl field) or
  /// [`Error::Tokenizer`] (chat-template / encode failure).
  fn process(&self, idx: usize) -> Result<Example>;
}

// ───────────────────────────── TextDataset ─────────────────────────────

/// Light-weight wrapper for a jsonl-backed plain-text dataset — Python
/// `mlx_lm/tuner/datasets.py:11..=36` (`class TextDataset`).
///
/// Each parsed jsonl line is expected to be an object with a string under
/// the configured `text_key` (default [`DEFAULT_TEXT_KEY`]). The
/// [`Dataset::process`] tokenizes the string and appends the tokenizer's
/// primary EOS id when missing (Python `tuner/datasets.py:27..=30`), then
/// returns `(tokens, 0)` — no prompt masking (the entire sequence is the
/// loss target).
pub struct TextDataset<'a> {
  data: Vec<Value>,
  tokenizer: &'a Tokenizer,
  text_key: String,
}

impl std::fmt::Debug for TextDataset<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("TextDataset")
      .field("len", &self.data.len())
      .field("text_key", &self.text_key)
      .finish()
  }
}

impl<'a> TextDataset<'a> {
  /// Construct a [`TextDataset`] from already-parsed jsonl records, mirroring
  /// Python `TextDataset.__init__` (`tuner/datasets.py:16..=24`).
  pub fn new(data: Vec<Value>, tokenizer: &'a Tokenizer, text_key: impl Into<String>) -> Self {
    Self {
      data,
      tokenizer,
      text_key: text_key.into(),
    }
  }
}

impl Dataset for TextDataset<'_> {
  fn len(&self) -> usize {
    self.data.len()
  }

  fn get(&self, idx: usize) -> Result<&Value> {
    self.data.get(idx).ok_or_else(|| Error::Backend {
      message: format!(
        "TextDataset: index {idx} out of range (len={})",
        self.data.len()
      ),
    })
  }

  /// Python `TextDataset.process` (`tuner/datasets.py:26..=30`):
  /// `d = tokenizer.encode(d[text_key]); if d[-1] != eos: d.append(eos);
  /// return (d, 0)`.
  fn process(&self, idx: usize) -> Result<Example> {
    let record = self.get(idx)?;
    let text = field_as_str(record, &self.text_key, "TextDataset")?;
    // Python passes the bare string to `tokenizer.encode`, which adds
    // special tokens per the tokenizer's defaults. Mirror with
    // `add_special_tokens = true`.
    let mut tokens = self.tokenizer.encode(text, true)?;
    // `tuner/datasets.py:28..=29`: append the primary EOS if the encoded
    // sequence does not already end with it. A tokenizer with NO primary
    // EOS leaves the sequence unchanged (matches Python: `eos_token_id`
    // being `None` falls through the `if d[-1] != None` comparison —
    // both branches of the python `!= None` against an int are `True`,
    // so the append would happen; but a `None` eos cannot be appended
    // either, so the Python path raises. Here we keep the sequence
    // verbatim and treat a missing eos as a clean no-op — adding `None`
    // is not representable, and the trainer's loss is well-defined on
    // an eos-less sequence).
    if let Some(eos) = self.tokenizer.eos_token_id()
      && tokens.last() != Some(&eos)
    {
      tokens.push(eos);
    }
    Ok((tokens, 0))
  }
}

// ───────────────────────────── ChatDataset ─────────────────────────────

/// jsonl-backed HF-chat-format dataset — Python
/// `mlx_lm/tuner/datasets.py:39..=83` (`class ChatDataset`).
///
/// Each parsed jsonl line is expected to be an object with a `"messages"`
/// array under the configured `chat_key` (default [`DEFAULT_CHAT_KEY`]) and
/// optional `"tools"` field. The [`Dataset::process`] runs
/// [`Tokenizer::apply_chat_template_ids`] on the messages (Python
/// `tuner/datasets.py:60..=64`), and (when `mask_prompt`) renders the
/// `messages[:-1]` prefix with `add_generation_prompt` set to whether the
/// final message is from the `assistant` role, returning the prefix length
/// as the loss-mask offset (Python `tuner/datasets.py:65..=75`).
pub struct ChatDataset<'a> {
  data: Vec<Value>,
  tokenizer: &'a Tokenizer,
  chat_key: String,
  mask_prompt: bool,
}

impl std::fmt::Debug for ChatDataset<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("ChatDataset")
      .field("len", &self.data.len())
      .field("chat_key", &self.chat_key)
      .field("mask_prompt", &self.mask_prompt)
      .finish()
  }
}

impl<'a> ChatDataset<'a> {
  /// Construct a [`ChatDataset`] from already-parsed jsonl records, mirroring
  /// Python `ChatDataset.__init__` (`tuner/datasets.py:45..=55`).
  pub fn new(
    data: Vec<Value>,
    tokenizer: &'a Tokenizer,
    chat_key: impl Into<String>,
    mask_prompt: bool,
  ) -> Self {
    Self {
      data,
      tokenizer,
      chat_key: chat_key.into(),
      mask_prompt,
    }
  }
}

impl Dataset for ChatDataset<'_> {
  fn len(&self) -> usize {
    self.data.len()
  }

  fn get(&self, idx: usize) -> Result<&Value> {
    self.data.get(idx).ok_or_else(|| Error::Backend {
      message: format!(
        "ChatDataset: index {idx} out of range (len={})",
        self.data.len()
      ),
    })
  }

  /// Python `ChatDataset.process` (`tuner/datasets.py:57..=77`).
  fn process(&self, idx: usize) -> Result<Example> {
    let record = self.get(idx)?;
    let messages = record.get(&self.chat_key).ok_or_else(|| Error::Backend {
      message: format!(
        "ChatDataset: jsonl record missing '{}' field",
        self.chat_key
      ),
    })?;
    if !messages.is_array() {
      return Err(Error::ShapeMismatch {
        message: format!(
          "ChatDataset: '{}' field must be a JSON array, got {}",
          self.chat_key,
          json_kind(messages),
        ),
      });
    }
    let tools = record.get("tools");
    let tokens = self
      .tokenizer
      .apply_chat_template_ids(messages, tools, false, false, None)?;

    if !self.mask_prompt {
      return Ok((tokens, 0));
    }

    // Python `messages[:-1]` + `add_generation_prompt = messages[-1]["role"]
    // == "assistant"`. The prefix encode determines the offset (only the
    // length is needed; we discard the prefix ids).
    let arr = messages
      .as_array()
      .expect("messages.is_array() was checked above");
    let last_role = arr
      .last()
      .and_then(|m| m.get("role"))
      .and_then(Value::as_str);
    let add_generation_prompt = last_role == Some("assistant");
    let prefix = Value::Array(arr[..arr.len().saturating_sub(1)].to_vec());
    let prefix_tokens =
      self
        .tokenizer
        .apply_chat_template_ids(&prefix, tools, add_generation_prompt, false, None)?;
    Ok((tokens, prefix_tokens.len()))
  }
}

// ───────────────────────────── CompletionsDataset ─────────────────────────────

/// jsonl-backed prompt/completion dataset — Python
/// `mlx_lm/tuner/datasets.py:86..=133` (`class CompletionsDataset`).
///
/// Each parsed jsonl line is expected to be an object with a string under
/// `prompt_key` and a string under `completion_key` (defaults
/// [`DEFAULT_PROMPT_KEY`] / [`DEFAULT_COMPLETION_KEY`]). The two are
/// wrapped in a synthetic two-message chat (`{role: user, content: prompt}`,
/// `{role: assistant, content: completion}`) and rendered through the
/// tokenizer's chat template (Python `tuner/datasets.py:108..=115`).
///
/// When `mask_prompt`, the prefix `messages[:-1]` (i.e. just the user
/// prompt) is rendered with `add_generation_prompt = true` and its length
/// returned as the loss-mask offset (Python `tuner/datasets.py:116..=125`).
pub struct CompletionsDataset<'a> {
  data: Vec<Value>,
  tokenizer: &'a Tokenizer,
  prompt_key: String,
  completion_key: String,
  mask_prompt: bool,
}

impl std::fmt::Debug for CompletionsDataset<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("CompletionsDataset")
      .field("len", &self.data.len())
      .field("prompt_key", &self.prompt_key)
      .field("completion_key", &self.completion_key)
      .field("mask_prompt", &self.mask_prompt)
      .finish()
  }
}

impl<'a> CompletionsDataset<'a> {
  /// Construct a [`CompletionsDataset`] from already-parsed jsonl records,
  /// mirroring Python `CompletionsDataset.__init__`
  /// (`tuner/datasets.py:93..=105`).
  pub fn new(
    data: Vec<Value>,
    tokenizer: &'a Tokenizer,
    prompt_key: impl Into<String>,
    completion_key: impl Into<String>,
    mask_prompt: bool,
  ) -> Self {
    Self {
      data,
      tokenizer,
      prompt_key: prompt_key.into(),
      completion_key: completion_key.into(),
      mask_prompt,
    }
  }
}

impl Dataset for CompletionsDataset<'_> {
  fn len(&self) -> usize {
    self.data.len()
  }

  fn get(&self, idx: usize) -> Result<&Value> {
    self.data.get(idx).ok_or_else(|| Error::Backend {
      message: format!(
        "CompletionsDataset: index {idx} out of range (len={})",
        self.data.len()
      ),
    })
  }

  /// Python `CompletionsDataset.process` (`tuner/datasets.py:107..=127`).
  fn process(&self, idx: usize) -> Result<Example> {
    let record = self.get(idx)?;
    let prompt = field_as_str(record, &self.prompt_key, "CompletionsDataset")?;
    let completion = field_as_str(record, &self.completion_key, "CompletionsDataset")?;
    let tools = record.get("tools");

    let messages = serde_json::json!([
      { "role": "user", "content": prompt },
      { "role": "assistant", "content": completion },
    ]);
    let tokens = self
      .tokenizer
      .apply_chat_template_ids(&messages, tools, false, false, None)?;

    if !self.mask_prompt {
      return Ok((tokens, 0));
    }

    // Python `messages[:-1]` rendered with `add_generation_prompt = True`
    // (the user-only prefix that conditions the assistant turn).
    let prefix = serde_json::json!([
      { "role": "user", "content": prompt },
    ]);
    let prefix_tokens = self
      .tokenizer
      .apply_chat_template_ids(&prefix, tools, true, false, None)?;
    Ok((tokens, prefix_tokens.len()))
  }
}

// ───────────────────────────── ConcatenatedDataset ─────────────────────────────

/// Concat-by-index wrapper across multiple inner datasets — Python
/// `mlx_lm/tuner/datasets.py:136..=155` (`class ConcatenatedDataset`).
///
/// This is **NOT** sequence packing: it routes index access across the
/// sequence of inner datasets in declaration order. An index `idx` is
/// resolved by subtracting each inner `len()` in turn until the routed
/// inner is found (exactly Python's `for data_idx, data in enumerate(...);
/// j = idx - len(data); if j < 0: break; idx = j`).
///
/// `get(idx)` dispatches through to the inner dataset's `get`; `process(idx)`
/// dispatches through to the inner dataset's `process`. This matches the
/// Python class shape EXCEPT for one Python-specific tag-write: Python's
/// `__getitem__` mutates the returned `dict` with a `"_dataset"` key so
/// `process(d)` can route back to the original sub-dataset. This port
/// routes by **index** instead (which avoids the mutating side-effect and
/// keeps `&Value` purely immutable), preserving the observable behavior.
pub struct ConcatenatedDataset<'a> {
  data: Vec<Box<dyn Dataset + 'a>>,
  len: usize,
}

impl std::fmt::Debug for ConcatenatedDataset<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("ConcatenatedDataset")
      .field("inner_count", &self.data.len())
      .field("len", &self.len)
      .finish()
  }
}

impl<'a> ConcatenatedDataset<'a> {
  /// Construct a [`ConcatenatedDataset`] from a sequence of inner datasets,
  /// mirroring Python `ConcatenatedDataset.__init__`
  /// (`tuner/datasets.py:137..=139`: `_len = sum(len(d) for d in _data)`).
  pub fn new(data: Vec<Box<dyn Dataset + 'a>>) -> Self {
    let len = data.iter().map(|d| d.len()).sum();
    Self { data, len }
  }

  /// Resolve a global `idx` into `(inner_dataset_index, local_idx)` —
  /// the Python `for data_idx, data in enumerate(...); j = idx - len(data);
  /// if j < 0: break; idx = j` traversal.
  fn resolve(&self, idx: usize) -> Result<(usize, usize)> {
    let mut remaining = idx;
    for (data_idx, inner) in self.data.iter().enumerate() {
      if remaining < inner.len() {
        return Ok((data_idx, remaining));
      }
      remaining -= inner.len();
    }
    Err(Error::Backend {
      message: format!(
        "ConcatenatedDataset: index {idx} out of range (len={})",
        self.len,
      ),
    })
  }
}

impl Dataset for ConcatenatedDataset<'_> {
  fn len(&self) -> usize {
    self.len
  }

  fn get(&self, idx: usize) -> Result<&Value> {
    let (di, li) = self.resolve(idx)?;
    self.data[di].get(li)
  }

  fn process(&self, idx: usize) -> Result<Example> {
    let (di, li) = self.resolve(idx)?;
    self.data[di].process(li)
  }
}

// ───────────────────────────── CacheDataset ─────────────────────────────

/// In-memory `process()` memoizer — Python
/// `mlx_lm/tuner/datasets.py:158..=172` (`class CacheDataset`).
///
/// Wraps an inner [`Dataset`] and lazily caches the per-index
/// `(tokens, offset)` pair the first time it is requested
/// (`tuner/datasets.py:167..=169`: `if self._proc_data[idx] is None:
/// self._proc_data[idx] = self._data.process(self._data[idx])`).
///
/// Python's cache is in-memory only (`self._proc_data = [None] *
/// len(data)`); there is **no** sidecar `.cache` file. A source-jsonl mtime
/// change invalidates the cache via the natural mechanism: the next
/// [`load_dataset`] call constructs a fresh [`CacheDataset`] whose
/// `_proc_data` starts empty.
///
/// Interior mutability ([`RefCell`]) keeps [`Dataset::process`] taking
/// `&self`, matching the trait and the Python `__getitem__` shape.
pub struct CacheDataset<'a> {
  data: Box<dyn Dataset + 'a>,
  proc_data: RefCell<Vec<Option<Example>>>,
}

impl std::fmt::Debug for CacheDataset<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let cached = self
      .proc_data
      .try_borrow()
      .map(|c| c.iter().filter(|e| e.is_some()).count())
      .ok();
    f.debug_struct("CacheDataset")
      .field("len", &self.data.len())
      .field("cached_count", &cached)
      .finish()
  }
}

impl<'a> CacheDataset<'a> {
  /// Wrap an inner dataset and pre-size the cache to `inner.len()`.
  pub fn new(data: Box<dyn Dataset + 'a>) -> Self {
    let n = data.len();
    Self {
      data,
      proc_data: RefCell::new(vec![None; n]),
    }
  }

  /// Cached token-sequence length at `idx` — Python `itemlen`
  /// (`tuner/datasets.py:163..=164`: `len(self._data[idx])`).
  /// Returns the cached `tokens.len()` if the entry is already processed,
  /// else freshly processes and caches it.
  pub fn item_len(&self, idx: usize) -> Result<usize> {
    let cached = self.process(idx)?;
    Ok(cached.0.len())
  }
}

impl Dataset for CacheDataset<'_> {
  fn len(&self) -> usize {
    self.data.len()
  }

  fn get(&self, idx: usize) -> Result<&Value> {
    self.data.get(idx)
  }

  /// Python `CacheDataset.__getitem__` — lazy populate then return.
  ///
  /// Returns the cached pair as a fresh clone each call. The clone is
  /// cheap (a `Vec<u32>`) and keeps the trait `dyn`-safe by not exposing
  /// a `Ref` into the [`RefCell`] (whose borrow lifetime would leak into
  /// the trait method's return type via a generic associated type, which
  /// is not `dyn`-compatible).
  fn process(&self, idx: usize) -> Result<Example> {
    {
      let cache = self.proc_data.borrow();
      if let Some(Some(pair)) = cache.get(idx) {
        return Ok(pair.clone());
      }
    }
    // Compute outside any borrow.
    let computed = self.data.process(idx)?;
    let mut cache = self.proc_data.borrow_mut();
    if idx >= cache.len() {
      return Err(Error::Backend {
        message: format!(
          "CacheDataset: index {idx} out of range (len={})",
          cache.len()
        ),
      });
    }
    cache[idx] = Some(computed.clone());
    Ok(computed)
  }
}

// ───────────────────────────── factory ─────────────────────────────

/// What dataset shape to construct — the explicit form of Python
/// `create_dataset`'s sample-driven dispatch
/// (`tuner/datasets.py:175..=202`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetType {
  /// Tokenize a single `text` field verbatim — see [`TextDataset`].
  Text,
  /// Apply the chat template to a `messages` array — see [`ChatDataset`].
  Chat,
  /// Render a `prompt`+`completion` pair as a two-turn chat — see
  /// [`CompletionsDataset`].
  Completions,
  /// Auto-detect from the FIRST jsonl record's fields, mirroring Python
  /// `create_dataset`'s sample-driven dispatch:
  /// `prompt` + `completion` ⇒ [`DatasetType::Completions`]; else
  /// `messages` ⇒ [`DatasetType::Chat`]; else `text` ⇒ [`DatasetType::Text`];
  /// else error.
  Auto,
}

/// Per-call dataset config — the typed analogue of Python's
/// `types.SimpleNamespace`-style `config` argument to `create_dataset`
/// (`tuner/datasets.py:175..=202`).
///
/// The defaults mirror Python's `getattr(config, ..., default)` defaults
/// exactly:
/// - `mask_prompt = false` (Python `tuner/datasets.py:180`)
/// - `prompt_feature = "prompt"` (Python `tuner/datasets.py:181`)
/// - `text_feature = "text"` (Python `tuner/datasets.py:182`)
/// - `completion_feature = "completion"` (Python `tuner/datasets.py:183`)
/// - `chat_feature = "messages"` (Python `tuner/datasets.py:184`)
#[derive(Debug, Clone)]
pub struct DatasetConfig {
  /// Whether to set the prompt-mask offset (the `(tokens, offset)`'s second
  /// element). When `false`, every dataset returns `(tokens, 0)`.
  pub mask_prompt: bool,
  /// jsonl field name for [`TextDataset`].
  pub text_feature: String,
  /// jsonl field name for [`ChatDataset`].
  pub chat_feature: String,
  /// jsonl field name for [`CompletionsDataset`]'s prompt.
  pub prompt_feature: String,
  /// jsonl field name for [`CompletionsDataset`]'s completion.
  pub completion_feature: String,
}

impl Default for DatasetConfig {
  fn default() -> Self {
    Self {
      mask_prompt: false,
      text_feature: DEFAULT_TEXT_KEY.to_owned(),
      chat_feature: DEFAULT_CHAT_KEY.to_owned(),
      prompt_feature: DEFAULT_PROMPT_KEY.to_owned(),
      completion_feature: DEFAULT_COMPLETION_KEY.to_owned(),
    }
  }
}

/// Build the right dataset type from already-parsed jsonl records — Python
/// `create_dataset` (`tuner/datasets.py:175..=202`).
///
/// `data` is the parsed jsonl (one [`Value`] per line). The first record's
/// shape drives auto-detection: `prompt_feature` + `completion_feature` ⇒
/// [`CompletionsDataset`]; `chat_feature` ⇒ [`ChatDataset`]; `text_feature`
/// ⇒ [`TextDataset`]; else an [`Error::Backend`] with the same
/// "Unsupported data format" message as Python `tuner/datasets.py:199..=202`.
///
/// `mask_prompt` on a [`DatasetType::Text`] is an error
/// ([`Error::Backend`]), mirroring Python `tuner/datasets.py:195..=196`:
/// `raise ValueError("Prompt masking not supported for text dataset.")`.
pub fn create_dataset<'a>(
  data: Vec<Value>,
  tokenizer: &'a Tokenizer,
  config: &DatasetConfig,
  dataset_type: DatasetType,
) -> Result<Box<dyn Dataset + 'a>> {
  let resolved = match dataset_type {
    DatasetType::Auto => auto_detect(&data, config)?,
    other => other,
  };
  match resolved {
    DatasetType::Text => {
      if config.mask_prompt {
        return Err(Error::Backend {
          message: "Prompt masking not supported for text dataset.".to_owned(),
        });
      }
      Ok(Box::new(TextDataset::new(
        data,
        tokenizer,
        config.text_feature.clone(),
      )))
    }
    DatasetType::Chat => Ok(Box::new(ChatDataset::new(
      data,
      tokenizer,
      config.chat_feature.clone(),
      config.mask_prompt,
    ))),
    DatasetType::Completions => Ok(Box::new(CompletionsDataset::new(
      data,
      tokenizer,
      config.prompt_feature.clone(),
      config.completion_feature.clone(),
      config.mask_prompt,
    ))),
    DatasetType::Auto => unreachable!("auto_detect returned Auto"),
  }
}

/// Python `tuner/datasets.py:185..=202` — `sample = data[0]` field
/// detection.
fn auto_detect(data: &[Value], config: &DatasetConfig) -> Result<DatasetType> {
  let sample = data.first().ok_or_else(|| Error::Backend {
    message:
      "cannot auto-detect dataset type from an empty jsonl (pass an explicit DatasetType instead)"
        .to_owned(),
  })?;
  let has = |k: &str| sample.get(k).is_some();
  if has(&config.prompt_feature) && has(&config.completion_feature) {
    Ok(DatasetType::Completions)
  } else if has(&config.chat_feature) {
    Ok(DatasetType::Chat)
  } else if has(&config.text_feature) {
    Ok(DatasetType::Text)
  } else {
    // Match Python `tuner/datasets.py:199..=202` verbatim.
    Err(Error::Backend {
      message: "Unsupported data format, check the supported formats here:\n\
                https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#Data."
        .to_owned(),
    })
  }
}

// ───────────────────────────── load_dataset ─────────────────────────────

/// Load a single local jsonl file as a [`CacheDataset`]-wrapped
/// [`TextDataset`] / [`ChatDataset`] / [`CompletionsDataset`].
///
/// The Python entry point (`mlx_lm/tuner/datasets.py:309..=332`) dispatches
/// over a `(train, valid, test)` triple of jsonl files in a directory; this
/// port exposes the per-file primitive so callers can build that triple
/// themselves (and skip `valid` / `test` cleanly when absent — Python
/// returns an empty list, here the caller chooses not to call). The wrap
/// in a [`CacheDataset`] mirrors what Python `tuner/trainer.py` does at
/// the consume site.
///
/// `dataset_type` chooses the dataset shape explicitly; pass
/// [`DatasetType::Auto`] to mirror Python's `create_dataset` sample-driven
/// dispatch.
///
/// # Errors
///
/// - HuggingFace Hub paths (`hf://...`-prefixed) are rejected with a clear
///   [`Error::Backend`] message — they are out of scope per the project's
///   local-only policy (see [the module docs](self#scope-boundary)).
/// - An oversized file (above [`MAX_DATASET_FILE_BYTES`]) is rejected
///   before any read; a hostile mount cannot push an unbounded blob into
///   memory.
/// - A malformed jsonl line surfaces as [`Error::Backend`] with the line
///   number; an empty file returns a [`CacheDataset`] wrapping an
///   empty inner dataset (consistent with Python `load_subset` returning
///   `[]` for an absent `valid.jsonl` / `test.jsonl`).
pub fn load_dataset<'a>(
  path: &Path,
  tokenizer: &'a Tokenizer,
  dataset_type: DatasetType,
  config: &DatasetConfig,
) -> Result<CacheDataset<'a>> {
  // Reject HF Hub URIs up front. Python `tuner/datasets.py:309..=318`
  // routes non-existent local paths to `load_hf_dataset`; this port
  // **excludes** the HF Hub side entirely.
  if let Some(s) = path.to_str()
    && (s.starts_with("hf://") || s.starts_with("hf:"))
  {
    return Err(Error::Backend {
      message: format!(
        "HF Hub datasets are out of scope for the local-only mlxrs build \
         (path: {s}); pass a local jsonl file path instead"
      ),
    });
  }

  if !path.exists() {
    return Err(Error::Backend {
      message: format!("jsonl path does not exist: {}", path.display()),
    });
  }

  let meta = std::fs::metadata(path).map_err(|e| Error::Backend {
    message: format!("stat jsonl {}: {e}", path.display()),
  })?;
  if meta.len() > MAX_DATASET_FILE_BYTES {
    return Err(Error::Backend {
      message: format!(
        "jsonl {} is {} bytes — refusing to read above the {}-byte cap; \
         pass a smaller file or shard the data",
        path.display(),
        meta.len(),
        MAX_DATASET_FILE_BYTES,
      ),
    });
  }

  let file = std::fs::File::open(path).map_err(|e| Error::Backend {
    message: format!("open jsonl {}: {e}", path.display()),
  })?;
  let mut reader = BufReader::new(file);
  let mut data: Vec<Value> = Vec::new();
  let mut buf = String::new();
  let mut lineno: usize = 0;
  loop {
    buf.clear();
    let n = reader.read_line(&mut buf).map_err(|e| Error::Backend {
      message: format!("read jsonl {} line {}: {e}", path.display(), lineno + 1),
    })?;
    if n == 0 {
      break;
    }
    lineno += 1;
    // Skip blank lines defensively — jsonl files commonly end with a
    // trailing newline that BufRead would surface as an extra empty
    // line; the Python loader's `[json.loads(l) for l in fid]` would
    // error on the empty string, so a stricter port would too, but a
    // trailing newline is universal enough that swallowing it cleanly
    // is the friendlier behavior. A NON-trailing blank line is also
    // skipped on the same grounds.
    let trimmed = buf.trim();
    if trimmed.is_empty() {
      continue;
    }
    let v: Value = serde_json::from_str(trimmed).map_err(|e| Error::Backend {
      message: format!("parse jsonl {} line {}: {e}", path.display(), lineno),
    })?;
    data.push(v);
  }

  let inner = if data.is_empty() {
    // An empty file yields an empty dataset of the explicit type (or
    // [`DatasetType::Text`] when [`DatasetType::Auto`] is requested
    // and there is no sample to dispatch on — the friendlier mirror of
    // Python's `load_subset` returning `[]`).
    match dataset_type {
      DatasetType::Auto | DatasetType::Text => Box::new(TextDataset::new(
        Vec::new(),
        tokenizer,
        config.text_feature.clone(),
      )) as Box<dyn Dataset + 'a>,
      DatasetType::Chat => Box::new(ChatDataset::new(
        Vec::new(),
        tokenizer,
        config.chat_feature.clone(),
        config.mask_prompt,
      )) as Box<dyn Dataset + 'a>,
      DatasetType::Completions => Box::new(CompletionsDataset::new(
        Vec::new(),
        tokenizer,
        config.prompt_feature.clone(),
        config.completion_feature.clone(),
        config.mask_prompt,
      )) as Box<dyn Dataset + 'a>,
    }
  } else {
    create_dataset(data, tokenizer, config, dataset_type)?
  };
  Ok(CacheDataset::new(inner))
}

// ───────────────────────────── helpers ─────────────────────────────

/// Extract a string field from a jsonl record, returning a clean error
/// when missing or wrong-typed.
fn field_as_str<'a>(record: &'a Value, key: &str, type_name: &str) -> Result<&'a str> {
  let v = record.get(key).ok_or_else(|| Error::Backend {
    message: format!("{type_name}: jsonl record missing '{key}' field"),
  })?;
  v.as_str().ok_or_else(|| Error::ShapeMismatch {
    message: format!(
      "{type_name}: '{key}' field must be a string, got {}",
      json_kind(v),
    ),
  })
}

/// Brief tag for a JSON value's kind, for error messages.
fn json_kind(v: &Value) -> &'static str {
  match v {
    Value::Null => "null",
    Value::Bool(_) => "bool",
    Value::Number(_) => "number",
    Value::String(_) => "string",
    Value::Array(_) => "array",
    Value::Object(_) => "object",
  }
}

#[cfg(test)]
mod tests {
  use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
  };

  use serde_json::json;

  use super::*;

  // ───────────────────────── fixtures ─────────────────────────

  fn fresh_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
      "mlxrs-lm-tuner-datasets-{tag}-{}-{n}",
      std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// A minimal `WordLevel` tokenizer with a chat template that lets us
  /// hand-trace what the dataset's process() should emit. Vocabulary is
  /// designed so each word/marker is a single token id we can assert on.
  fn write_tokenizer(dir: &Path) -> Tokenizer {
    // Vocab:
    //   0=<unk>  1=<s>  2=</s>  3=hello  4=world  5=user  6=assistant
    //   7=:    8=tools
    let tokenizer_json = json!({
      "version": "1.0",
      "added_tokens": [
        {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 1, "content": "<s>",   "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 2, "content": "</s>",  "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 7, "content": ":",     "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 8, "content": "tools", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
      ],
      "normalizer": null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {
          "<unk>": 0, "<s>": 1, "</s>": 2,
          "hello": 3, "world": 4,
          "user": 5, "assistant": 6,
          ":": 7, "tools": 8
        },
        "unk_token": "<unk>"
      }
    });
    let cfg = json!({
      "bos_token": "<s>",
      "eos_token": "</s>",
      "unk_token": "<unk>",
      // A trivial chat template: emits the role token, ':', then content
      // tokens. add_generation_prompt appends 'assistant :' so the
      // prefix-length offset for a one-message user prefix is
      // {user, :, <content>, assistant, :}.
      "chat_template":
        "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} \
         {% endfor %}{% if add_generation_prompt %}assistant : {% endif %}"
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
    Tokenizer::from_path(dir, None).unwrap()
  }

  /// Build a temp-dir tokenizer and yield it; the directory is leaked
  /// intentionally (process-lifetime fixture).
  fn tokenizer_fixture(tag: &str) -> Tokenizer {
    let dir = fresh_dir(tag);
    write_tokenizer(&dir)
  }

  fn write_jsonl(path: &Path, lines: &[Value]) {
    let mut s = String::new();
    for v in lines {
      s.push_str(&v.to_string());
      s.push('\n');
    }
    std::fs::write(path, s).unwrap();
  }

  // ───────────────────────── TextDataset ─────────────────────────

  #[test]
  fn text_dataset_happy_path_appends_eos_when_missing() {
    let tok = tokenizer_fixture("text_happy");
    let data = vec![
      json!({ "text": "hello world" }),
      json!({ "text": "world hello" }),
      json!({ "text": "hello" }),
    ];
    let ds = TextDataset::new(data, &tok, DEFAULT_TEXT_KEY);
    assert_eq!(ds.len(), 3);
    let (toks0, off0) = ds.process(0).unwrap();
    let (toks1, off1) = ds.process(1).unwrap();
    let (toks2, off2) = ds.process(2).unwrap();
    // hello world </s>
    assert_eq!(toks0, vec![3, 4, 2]);
    // world hello </s>
    assert_eq!(toks1, vec![4, 3, 2]);
    // hello </s>
    assert_eq!(toks2, vec![3, 2]);
    // No prompt masking on text.
    assert_eq!(off0, 0);
    assert_eq!(off1, 0);
    assert_eq!(off2, 0);
  }

  #[test]
  fn text_dataset_does_not_double_append_eos() {
    let tok = tokenizer_fixture("text_no_dup_eos");
    // The bare-string `"hello </s>"` Whitespace-tokenizes to `[3, 2]`
    // (the `</s>` is a registered special-token literal). encode then
    // sees the trailing 2 and the dataset must NOT push another one.
    let data = vec![json!({ "text": "hello </s>" })];
    let ds = TextDataset::new(data, &tok, DEFAULT_TEXT_KEY);
    let (toks, _) = ds.process(0).unwrap();
    assert_eq!(toks, vec![3, 2]);
  }

  #[test]
  fn text_dataset_missing_field_errors() {
    let tok = tokenizer_fixture("text_missing_field");
    let data = vec![json!({ "not_text": "hello" })];
    let ds = TextDataset::new(data, &tok, DEFAULT_TEXT_KEY);
    let err = ds.process(0).unwrap_err();
    let msg = err.to_string();
    assert!(
      msg.contains("missing 'text' field"),
      "expected missing-field error, got: {msg}"
    );
  }

  #[test]
  fn text_dataset_wrong_type_errors() {
    let tok = tokenizer_fixture("text_wrong_type");
    let data = vec![json!({ "text": 42 })];
    let ds = TextDataset::new(data, &tok, DEFAULT_TEXT_KEY);
    let err = ds.process(0).unwrap_err();
    assert!(
      matches!(err, Error::ShapeMismatch { .. }),
      "expected ShapeMismatch, got: {err:?}"
    );
  }

  // ───────────────────────── ChatDataset ─────────────────────────

  #[test]
  fn chat_dataset_happy_path_no_mask() {
    let tok = tokenizer_fixture("chat_happy_no_mask");
    let data = vec![json!({
      "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
      ]
    })];
    let ds = ChatDataset::new(data, &tok, DEFAULT_CHAT_KEY, false);
    let (toks, off) = ds.process(0).unwrap();
    // Template renders: "user : hello assistant : world "
    // → [5, 7, 3, 6, 7, 4]
    assert_eq!(toks, vec![5, 7, 3, 6, 7, 4]);
    assert_eq!(off, 0);
  }

  #[test]
  fn chat_dataset_mask_prompt_returns_prefix_offset() {
    let tok = tokenizer_fixture("chat_mask");
    let data = vec![json!({
      "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
      ]
    })];
    let ds = ChatDataset::new(data, &tok, DEFAULT_CHAT_KEY, true);
    let (toks, off) = ds.process(0).unwrap();
    // Full:   [5, 7, 3, 6, 7, 4]   (user : hello assistant : world)
    // Prefix (messages[:-1]=user, last_role==assistant so
    // add_generation_prompt=true): "user : hello assistant : "
    // → [5, 7, 3, 6, 7]
    assert_eq!(toks, vec![5, 7, 3, 6, 7, 4]);
    assert_eq!(off, 5);
  }

  #[test]
  fn chat_dataset_missing_messages_errors() {
    let tok = tokenizer_fixture("chat_missing");
    let data = vec![json!({ "no_messages_field": [] })];
    let ds = ChatDataset::new(data, &tok, DEFAULT_CHAT_KEY, false);
    let err = ds.process(0).unwrap_err();
    assert!(
      err.to_string().contains("missing 'messages' field"),
      "got: {err}"
    );
  }

  #[test]
  fn chat_dataset_messages_not_array_errors() {
    let tok = tokenizer_fixture("chat_not_array");
    let data = vec![json!({ "messages": "not an array" })];
    let ds = ChatDataset::new(data, &tok, DEFAULT_CHAT_KEY, false);
    let err = ds.process(0).unwrap_err();
    assert!(
      matches!(err, Error::ShapeMismatch { .. }),
      "expected ShapeMismatch, got: {err:?}"
    );
  }

  // ───────────────────────── CompletionsDataset ─────────────────────────

  #[test]
  fn completions_dataset_happy_path_no_mask() {
    let tok = tokenizer_fixture("comp_happy_no_mask");
    let data = vec![json!({ "prompt": "hello", "completion": "world" })];
    let ds = CompletionsDataset::new(
      data,
      &tok,
      DEFAULT_PROMPT_KEY,
      DEFAULT_COMPLETION_KEY,
      false,
    );
    let (toks, off) = ds.process(0).unwrap();
    // Same template rendering: "user : hello assistant : world "
    // → [5, 7, 3, 6, 7, 4]
    assert_eq!(toks, vec![5, 7, 3, 6, 7, 4]);
    assert_eq!(off, 0);
  }

  #[test]
  fn completions_dataset_mask_prompt_returns_prefix_offset() {
    let tok = tokenizer_fixture("comp_mask");
    let data = vec![json!({ "prompt": "hello", "completion": "world" })];
    let ds = CompletionsDataset::new(data, &tok, DEFAULT_PROMPT_KEY, DEFAULT_COMPLETION_KEY, true);
    let (toks, off) = ds.process(0).unwrap();
    // Full:    [5, 7, 3, 6, 7, 4]
    // Prefix (user-only + add_generation_prompt=true):
    //   "user : hello assistant : " → [5, 7, 3, 6, 7]
    assert_eq!(toks, vec![5, 7, 3, 6, 7, 4]);
    assert_eq!(off, 5);
  }

  #[test]
  fn completions_dataset_missing_prompt_errors() {
    let tok = tokenizer_fixture("comp_missing_prompt");
    let data = vec![json!({ "completion": "world" })];
    let ds = CompletionsDataset::new(
      data,
      &tok,
      DEFAULT_PROMPT_KEY,
      DEFAULT_COMPLETION_KEY,
      false,
    );
    let err = ds.process(0).unwrap_err();
    assert!(
      err.to_string().contains("missing 'prompt' field"),
      "got: {err}"
    );
  }

  // ───────────────────────── ConcatenatedDataset ─────────────────────────

  #[test]
  fn concatenated_dataset_indexes_across_inner_in_order() {
    let tok = tokenizer_fixture("concat_indexes");
    let a = TextDataset::new(vec![json!({ "text": "hello" })], &tok, DEFAULT_TEXT_KEY);
    let b = TextDataset::new(
      vec![json!({ "text": "world" }), json!({ "text": "hello world" })],
      &tok,
      DEFAULT_TEXT_KEY,
    );
    let cat = ConcatenatedDataset::new(vec![Box::new(a), Box::new(b)]);
    assert_eq!(cat.len(), 3);
    // idx 0 → a[0]: hello </s>
    assert_eq!(cat.process(0).unwrap().0, vec![3, 2]);
    // idx 1 → b[0]: world </s>
    assert_eq!(cat.process(1).unwrap().0, vec![4, 2]);
    // idx 2 → b[1]: hello world </s>
    assert_eq!(cat.process(2).unwrap().0, vec![3, 4, 2]);
  }

  #[test]
  fn concatenated_dataset_out_of_range_errors() {
    let tok = tokenizer_fixture("concat_oor");
    let a = TextDataset::new(vec![json!({ "text": "hello" })], &tok, DEFAULT_TEXT_KEY);
    let cat = ConcatenatedDataset::new(vec![Box::new(a)]);
    assert!(cat.process(7).is_err());
  }

  #[test]
  fn concatenated_dataset_empty_inputs_yield_empty_dataset() {
    let tok = tokenizer_fixture("concat_empty");
    let a = TextDataset::new(vec![], &tok, DEFAULT_TEXT_KEY);
    let b = TextDataset::new(vec![], &tok, DEFAULT_TEXT_KEY);
    let cat = ConcatenatedDataset::new(vec![Box::new(a), Box::new(b)]);
    assert_eq!(cat.len(), 0);
    assert!(cat.is_empty());
  }

  // ───────────────────────── CacheDataset ─────────────────────────

  #[test]
  fn cache_dataset_returns_consistent_result_on_repeat() {
    let tok = tokenizer_fixture("cache_repeat");
    let inner = TextDataset::new(
      vec![json!({ "text": "hello" }), json!({ "text": "world" })],
      &tok,
      DEFAULT_TEXT_KEY,
    );
    let cache = CacheDataset::new(Box::new(inner));
    assert_eq!(cache.len(), 2);
    let first = cache.process(0).unwrap();
    let second = cache.process(0).unwrap();
    assert_eq!(first, second);
    assert_eq!(cache.item_len(1).unwrap(), 2); // "world </s>" → 2 ids
  }

  #[test]
  fn cache_dataset_out_of_range_errors() {
    let tok = tokenizer_fixture("cache_oor");
    let inner = TextDataset::new(vec![json!({ "text": "hello" })], &tok, DEFAULT_TEXT_KEY);
    let cache = CacheDataset::new(Box::new(inner));
    assert!(cache.process(99).is_err());
  }

  // ───────────────────────── load_dataset entry points ─────────────────────────

  #[test]
  fn load_dataset_text() {
    let tok = tokenizer_fixture("load_text");
    let dir = fresh_dir("load_text_data");
    let p = dir.join("train.jsonl");
    write_jsonl(
      &p,
      &[json!({ "text": "hello" }), json!({ "text": "world" })],
    );
    let ds = load_dataset(&p, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
    assert_eq!(ds.len(), 2);
    assert_eq!(ds.process(0).unwrap().0, vec![3, 2]);
    assert_eq!(ds.process(1).unwrap().0, vec![4, 2]);
  }

  #[test]
  fn load_dataset_chat() {
    let tok = tokenizer_fixture("load_chat");
    let dir = fresh_dir("load_chat_data");
    let p = dir.join("train.jsonl");
    write_jsonl(
      &p,
      &[json!({
        "messages": [
          {"role": "user", "content": "hello"},
          {"role": "assistant", "content": "world"},
        ]
      })],
    );
    let ds = load_dataset(&p, &tok, DatasetType::Chat, &DatasetConfig::default()).unwrap();
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.process(0).unwrap().0, vec![5, 7, 3, 6, 7, 4]);
  }

  #[test]
  fn load_dataset_completions() {
    let tok = tokenizer_fixture("load_comp");
    let dir = fresh_dir("load_comp_data");
    let p = dir.join("train.jsonl");
    write_jsonl(&p, &[json!({ "prompt": "hello", "completion": "world" })]);
    let ds = load_dataset(
      &p,
      &tok,
      DatasetType::Completions,
      &DatasetConfig::default(),
    )
    .unwrap();
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.process(0).unwrap().0, vec![5, 7, 3, 6, 7, 4]);
  }

  #[test]
  fn load_dataset_concatenated() {
    let tok = tokenizer_fixture("load_concat");
    let dir = fresh_dir("load_concat_data");
    let p1 = dir.join("train.jsonl");
    let p2 = dir.join("valid.jsonl");
    write_jsonl(&p1, &[json!({ "text": "hello" })]);
    write_jsonl(&p2, &[json!({ "text": "world" })]);
    let a = load_dataset(&p1, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
    let b = load_dataset(&p2, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
    let cat = ConcatenatedDataset::new(vec![Box::new(a), Box::new(b)]);
    assert_eq!(cat.len(), 2);
    assert_eq!(cat.process(0).unwrap().0, vec![3, 2]);
    assert_eq!(cat.process(1).unwrap().0, vec![4, 2]);
  }

  #[test]
  fn load_dataset_cache() {
    let tok = tokenizer_fixture("load_cache");
    let dir = fresh_dir("load_cache_data");
    let p = dir.join("train.jsonl");
    write_jsonl(&p, &[json!({ "text": "hello" })]);
    let ds = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap();
    // `load_dataset` always wraps in a CacheDataset; the public type makes
    // that visible (vs returning a `Box<dyn Dataset>`), and a repeat call
    // returns the same `(tokens, offset)` pair.
    let first = ds.process(0).unwrap();
    let second = ds.process(0).unwrap();
    assert_eq!(first, second);
  }

  #[test]
  fn load_dataset_auto_detects_completions_first() {
    let tok = tokenizer_fixture("load_auto_comp");
    let dir = fresh_dir("load_auto_comp_data");
    let p = dir.join("train.jsonl");
    // Both completions (prompt+completion) AND text keys present →
    // Python's create_dataset picks completions first.
    write_jsonl(
      &p,
      &[json!({ "prompt": "hello", "completion": "world", "text": "ignored" })],
    );
    let ds = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap();
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.process(0).unwrap().0, vec![5, 7, 3, 6, 7, 4]);
  }

  #[test]
  fn load_dataset_auto_unsupported_format_errors() {
    let tok = tokenizer_fixture("load_auto_bad");
    let dir = fresh_dir("load_auto_bad_data");
    let p = dir.join("train.jsonl");
    write_jsonl(&p, &[json!({ "irrelevant": "junk" })]);
    let err = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap_err();
    assert!(
      err.to_string().contains("Unsupported data format"),
      "got: {err}"
    );
  }

  #[test]
  fn load_dataset_rejects_hf_hub_path() {
    let tok = tokenizer_fixture("load_hf");
    let p = PathBuf::from("hf://datasets/mlx-community/some-dataset");
    let err = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap_err();
    assert!(
      err.to_string().contains("HF Hub datasets are out of scope"),
      "got: {err}"
    );
  }

  #[test]
  fn load_dataset_text_with_mask_prompt_errors() {
    let tok = tokenizer_fixture("load_text_mask_err");
    let dir = fresh_dir("load_text_mask_err_data");
    let p = dir.join("train.jsonl");
    write_jsonl(&p, &[json!({ "text": "hello" })]);
    let cfg = DatasetConfig {
      mask_prompt: true,
      ..DatasetConfig::default()
    };
    let err = load_dataset(&p, &tok, DatasetType::Text, &cfg).unwrap_err();
    assert!(
      err.to_string().contains("not supported for text dataset"),
      "got: {err}"
    );
  }

  #[test]
  fn load_dataset_empty_file_returns_empty_iterator() {
    let tok = tokenizer_fixture("load_empty");
    let dir = fresh_dir("load_empty_data");
    let p = dir.join("train.jsonl");
    std::fs::write(&p, "").unwrap();
    let ds = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap();
    assert_eq!(ds.len(), 0);
    assert!(ds.is_empty());
  }

  #[test]
  fn load_dataset_malformed_line_errors_with_line_number() {
    let tok = tokenizer_fixture("load_malformed");
    let dir = fresh_dir("load_malformed_data");
    let p = dir.join("train.jsonl");
    std::fs::write(
      &p,
      "{\"text\": \"hello\"}\n{this is not json}\n{\"text\": \"world\"}\n",
    )
    .unwrap();
    let err = load_dataset(&p, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap_err();
    let s = err.to_string();
    assert!(
      s.contains("line 2"),
      "expected line number in error, got: {s}"
    );
  }

  #[test]
  fn load_dataset_nonexistent_path_errors() {
    let tok = tokenizer_fixture("load_nopath");
    let p = std::env::temp_dir().join(format!(
      "mlxrs-a6-does-not-exist-{}.jsonl",
      std::process::id()
    ));
    let err = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap_err();
    assert!(err.to_string().contains("does not exist"), "got: {err}");
  }

  #[test]
  fn cache_dataset_invalidates_on_source_mtime_change() {
    // The Python `CacheDataset` is in-memory per-instance — there is no
    // sidecar `.cache` file. A source mtime change invalidates the cache
    // via the natural mechanism: the next `load_dataset` call constructs
    // a FRESH `CacheDataset` whose `_proc_data` is empty, so the new
    // file contents are observed.
    let tok = tokenizer_fixture("cache_mtime");
    let dir = fresh_dir("cache_mtime_data");
    let p = dir.join("train.jsonl");

    // First version of the file.
    write_jsonl(&p, &[json!({ "text": "hello" })]);
    let first = {
      let ds = load_dataset(&p, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
      ds.process(0).unwrap()
    };
    assert_eq!(first.0, vec![3, 2]); // hello </s>

    // Mutate the file (simulating an mtime change with new content).
    // Sleep one milli to make the mtime change observable on every fs.
    std::thread::sleep(std::time::Duration::from_millis(10));
    write_jsonl(&p, &[json!({ "text": "world" })]);

    // Second load constructs a fresh CacheDataset → reads the new content.
    let second = {
      let ds = load_dataset(&p, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
      ds.process(0).unwrap()
    };
    assert_eq!(second.0, vec![4, 2]); // world </s>
    assert_ne!(first.0, second.0);
  }
}
