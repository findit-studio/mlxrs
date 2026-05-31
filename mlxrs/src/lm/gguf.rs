//! GGUF export pipeline — port of
//! [`mlx_lm/gguf.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/gguf.py).
//!
//! Translates an mlx-lm checkpoint (HF-style `config.json` + safetensors
//! weights + HF `tokenizer.json`) to a single `.gguf` file via
//! [`crate::io::save_gguf`]. The pipeline mirrors `convert_to_gguf` in
//! `mlx_lm/gguf.py` line-for-line; helpers ([`translate_weight_names`],
//! [`permute_weights`], [`prepare_metadata`], [`HfVocab`]) are direct
//! ports of the corresponding Python free functions and class so the
//! emitted GGUF byte-stream is the same family `mlx_lm` produces.
//!
//! ```text
//!     convert_to_gguf(model_path, gguf_path)
//!         │
//!         ├── load(model_path)                    [crate::lm::load]
//!         │     (Config, Weights, Tokenizer)
//!         │
//!         ├── reject config.quantization          (matches mlx_lm/gguf.py:271)
//!         │
//!         ├── permute_weights for self_attn.{q,k}_proj.weight
//!         │       (mlx_lm/gguf.py:133-141; head-interleave)
//!         │
//!         ├── translate_weight_names              (mlx_lm/gguf.py:103-130)
//!         │       HF naming → GGUF "blk.N.attn_*" / "ffn_*" / "token_embd" / …
//!         │
//!         ├── HfVocab + prepare_metadata          (mlx_lm/gguf.py:24-258)
//!         │       general.* / llama.* / tokenizer.ggml.* keys
//!         │
//!         ├── normalize weight dtypes             (mlx_lm/gguf.py:303-310)
//!         │       bf16 → f16; "norm" → f32; else pass through
//!         │
//!         └── crate::io::save_gguf(weights, metadata)
//! ```
//!
//! Cited references throughout point to file:line in
//! `/Users/user/Develop/findit-studio/mlx-lm/mlx_lm/gguf.py` so reviewers
//! can diff line-by-line.
//!
//! **Scope boundaries:**
//! - Per-architecture model implementations are NOT ported; the GGUF
//!   key prefixes are emitted as `llama.*` because the mlx-lm reference
//!   hard-codes that family (`mlx_lm/gguf.py:146-228` — every key is
//!   `general.*` / `llama.*` / `tokenizer.ggml.*`). A model whose
//!   `model_type` is outside the LM-side supported set is rejected by
//!   [`convert_to_gguf`] rather than silently mislabeled.
//! - Quantized → GGUF conversion is NOT ported — the reference explicitly
//!   raises `NotImplementedError` on a quantized checkpoint
//!   (`mlx_lm/gguf.py:271-274`); [`convert_to_gguf`] returns an
//!   [`Error::Backend`] in the same case.
//! - HF Hub download is NOT ported (project policy: local-only) — the
//!   reference `convert_to_gguf` is a pure local-file driver anyway.

use std::{
  collections::{BTreeMap, BTreeSet, HashMap, HashSet},
  path::PathBuf,
};

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    ArithmeticOverflowPayload, Error, InvariantViolationPayload, MissingKeyPayload,
    OutOfRangePayload, ParsePayload, Result, UnknownEnumValuePayload,
  },
  io::GgufMetadata,
  lm::load::{Config, Weights},
};

/// GGUF token-type tag — port of
/// [`TokenType`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/gguf.py#L10-L16)
/// (`mlx_lm/gguf.py:10-16`).
///
/// Mirrors the llama.cpp `convert.py` token-type enum the reference inherits
/// from. The integer values are the on-disk GGUF encoding and MUST NOT be
/// renumbered — the values are part of the `.gguf` file format and a third-
/// party loader (`llama.cpp`) reads them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TokenType {
  /// Plain vocabulary token.
  Normal = 1,
  /// Unknown token (`<unk>`).
  Unknown = 2,
  /// Control / special token (e.g. `<s>`, `</s>`).
  Control = 3,
  /// User-defined added token (added after the base vocab).
  UserDefined = 4,
  /// Unused slot.
  Unused = 5,
  /// Byte fallback token (`<0xAB>`).
  Byte = 6,
}

/// GGUF file-type tag — port of
/// [`GGMLFileType`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/gguf.py#L19-L20)
/// (`mlx_lm/gguf.py:19-20`).
///
/// The reference only emits one variant (`GGML_TYPE_F16 = 1`); we keep the
/// enum + repr so the on-disk metadata value matches the Python output bit-
/// for-bit (`mlx_lm/gguf.py:219-226`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlFileType {
  /// Half-precision float weights (the reference's only output type).
  F16 = 1,
}

/// Vocabulary packer — port of
/// [`HfVocab`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/gguf.py#L24-L100)
/// (`mlx_lm/gguf.py:24-100`).
///
/// Walks the loaded HF tokenizer and emits `(text, score, toktype)` triples
/// in the order GGUF expects: the base vocab (sorted by id, skipping ids
/// owned by added tokens — `mlx_lm/gguf.py:55-65`), then the added tokens
/// in the original insertion order (`mlx_lm/gguf.py:38-44`, `77-85`).
///
/// `score` is hard-coded to `-1000.0` — the reference returns a constant for
/// every base / added id (`mlx_lm/gguf.py:73-75`), since GGUF carries the
/// score channel but HF tokenizers do not actually expose per-token scores
/// (mlx-lm inherits the constant from llama.cpp's `convert.py`).
///
/// `toktype` classification (`mlx_lm/gguf.py:67-72`):
/// - text matches `<0x[0-9A-Fa-f]{2}>` → [`TokenType::Byte`]
/// - id is in `all_special_ids` → [`TokenType::Control`]
/// - else → [`TokenType::Normal`] (base path) / [`TokenType::UserDefined`]
///   (added path; `mlx_lm/gguf.py:80-84`)
pub struct HfVocab {
  /// Names of added tokens whose id is >= the base vocab size, paired
  /// with their ids in id-order — mirrors `self.added_tokens_list`
  /// (`mlx_lm/gguf.py:38-44`). The id is carried alongside the text so
  /// the emission walk in [`Self::all_tokens`] can classify each added
  /// token by id against [`Self::special_ids`] — looking up by text via
  /// `specials` would miss config-declared specials whose
  /// `added_tokens_decoder.special` flag is `false`.
  added_tokens_list: Vec<(u32, String)>,
  /// Ids of added tokens (`mlx_lm/gguf.py:44`) used to skip them in the
  /// base-vocab walk so an id is never emitted twice.
  added_tokens_ids: BTreeSet<u32>,
  /// `{special_token_text -> id}` — port of
  /// `self.specials` (`mlx_lm/gguf.py:45-48`). Carried for parity with
  /// the reference (a future LoRA-adapter / reverse-vocab consumer may
  /// need it) but no longer used to classify added tokens in
  /// [`Self::all_tokens`] — see [`Self::added_tokens_list`].
  #[allow(dead_code)]
  specials: HashMap<String, u32>,
  /// Set of all special-token ids — port of
  /// `set(self.tokenizer.all_special_ids)` (`mlx_lm/gguf.py:49`).
  ///
  /// HF's `PreTrainedTokenizerBase.all_special_ids` is the UNION of
  /// `added_tokens_decoder` entries flagged `special=true` AND the ids
  /// resolved from `tokenizer_config.json`'s `bos_token`/`eos_token`/
  /// `unk_token`/`pad_token`/`additional_special_tokens` strings.
  /// Crucially the latter may point at BASE-VOCAB ids (a BOS/EOS that
  /// is part of the base vocab and so does not live in
  /// `added_tokens_decoder`), so we MUST union both sources — building
  /// `special_ids` from `added_tokens_decoder` alone would misclassify
  /// such tokens as `Normal` instead of `Control` and emit a
  /// `tokenizer.ggml.token_type` array inconsistent with the metadata's
  /// own `bos_token_id`/`eos_token_id`/`unknown_token_id` scalar fields.
  /// The `HashSet` is used inside `get_token_type` for O(1) membership.
  special_ids: HashSet<u32>,
  /// `tokenizer.vocab_size` — base vocab (`mlx_lm/gguf.py:50`).
  vocab_size_base: u32,
  /// `vocab_size_base + len(added_tokens_list)` — the GGUF token count
  /// (`mlx_lm/gguf.py:51`). Asserted equal to the emitted token list
  /// length in [`prepare_metadata`] (matches `mlx_lm/gguf.py:240`).
  vocab_size: u32,
  /// Reverse lookup `{id -> text}` for the base vocab walk
  /// (`mlx_lm/gguf.py:56-58`); kept as a `Vec<Option<String>>` indexed by
  /// id so the per-id lookup is O(1).
  reverse_base_vocab: Vec<Option<String>>,
  /// `tokenizer.bos_token_id` — copied through to GGUF metadata
  /// (`mlx_lm/gguf.py:244-247`).
  bos_token_id: Option<u32>,
  /// `tokenizer.eos_token_id` — copied through to GGUF metadata
  /// (`mlx_lm/gguf.py:248-251`).
  eos_token_id: Option<u32>,
  /// `tokenizer.unk_token_id` — copied through to GGUF metadata
  /// (`mlx_lm/gguf.py:252-255`).
  unk_token_id: Option<u32>,
}

impl HfVocab {
  /// Build a [`HfVocab`] from a loaded tokenizer.
  ///
  /// Mirrors `HfVocab.__init__` (`mlx_lm/gguf.py:24-53`). The Python
  /// reference re-loads the tokenizer from `fname_tokenizer` via
  /// `AutoTokenizer.from_pretrained` — here the loaded
  /// [`crate::tokenizer::Tokenizer`] is passed in directly (the
  /// [`convert_to_gguf`] driver already has it from `crate::lm::load::load`),
  /// avoiding a re-load.
  pub fn from_tokenizer(tokenizer: &crate::tokenizer::Tokenizer) -> Result<Self> {
    let hf = tokenizer.hf();
    // `tokenizer.vocab_size` — HF tokenizer's BASE vocab size, NOT
    // including added tokens (`mlx_lm/gguf.py:50`).
    let vocab_size_base_usize = hf.get_vocab_size(false);
    let vocab_size_base = u32::try_from(vocab_size_base_usize).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "HfVocab: tokenizer base vocab size",
        "must fit in u32",
        vocab_size_base_usize.to_string(),
      ))
    })?;

    // `tokenizer.get_added_vocab()` returns ALL added tokens by name->id
    // (`mlx_lm/gguf.py:39-44`). We filter to those whose id is `>= vocab_size`
    // (per the Python `if tokidx >= self.tokenizer.vocab_size: ...`) — these
    // are the ones whose ids fall OUTSIDE the base vocab range and must be
    // appended after the base walk. The reference sorts by id (Python
    // `sorted(..., key=lambda x: x[1])`); we do the same so the on-disk order
    // is stable and matches the reference.
    let added_vocab = hf.get_added_vocabulary().get_vocab();
    let mut added: Vec<(u32, String)> = added_vocab
      .iter()
      .filter(|&(_, &id)| id >= vocab_size_base)
      .map(|(name, &id)| (id, name.clone()))
      .collect();
    added.sort_by_key(|(id, _)| *id);

    let mut added_tokens_list: Vec<(u32, String)> = Vec::with_capacity(added.len());
    let mut added_tokens_ids = BTreeSet::new();
    for (id, name) in &added {
      added_tokens_list.push((*id, name.clone()));
      added_tokens_ids.insert(*id);
    }

    // `self.specials = {tok: vocab[tok] for tok in tokenizer.all_special_tokens}`
    // (`mlx_lm/gguf.py:45-48`) AND
    // `self.special_ids = set(self.tokenizer.all_special_ids)`
    // (`mlx_lm/gguf.py:49`).
    //
    // HF's `PreTrainedTokenizerBase.all_special_ids` is built from the
    // UNION of:
    //   (a) `added_tokens_decoder` entries flagged `special=true`, and
    //   (b) BOS/EOS/UNK/PAD/`additional_special_tokens` declared in
    //       `tokenizer_config.json` — these may resolve to BASE-VOCAB ids,
    //       NOT just added-vocab ids (e.g. a tokenizer whose `<s>` lives
    //       at base id 1, not as an `added_tokens_decoder` entry).
    //
    // Building `special_ids` from (a) alone would misclassify any BOS/EOS
    // that happens to live in the base vocab as `Normal` instead of
    // `Control`, emitting a `tokenizer.ggml.token_type` array inconsistent
    // with the metadata's `bos/eos/unknown_token_id` scalar fields.
    //
    // `specials` (text→id) and `special_ids` (id-set) are populated from
    // both sources so the added-token-walk classifier
    // (`mlx_lm/gguf.py:78-84`) and the base-vocab-walk classifier
    // (`mlx_lm/gguf.py:63-65`) both see the full set.
    let mut specials: HashMap<String, u32> = HashMap::new();
    let mut special_ids: HashSet<u32> = HashSet::new();
    // (a) added_tokens_decoder entries with `special == true`.
    for (id, tok) in hf.get_added_tokens_decoder() {
      if tok.special {
        specials.insert(tok.content.clone(), id);
        special_ids.insert(id);
      }
    }
    // (b) tokenizer_config.json BOS/EOS/UNK/PAD + additional_special_tokens.
    //     `Tokenizer`'s `*_token_id` accessors return the resolved vocab id
    //     for the singular fields (or `None` if absent / unresolvable);
    //     `additional_special_token_ids` returns the resolved vec for the
    //     plural field. Each one is unioned into `special_ids`, which is a
    //     `HashSet` (so re-adding an id already from (a) is a no-op).
    for id in [
      tokenizer.bos_token_id(),
      tokenizer.eos_token_id(),
      tokenizer.unk_token_id(),
      tokenizer.pad_token_id(),
    ]
    .into_iter()
    .flatten()
    {
      special_ids.insert(id);
    }
    for id in tokenizer.additional_special_token_ids() {
      special_ids.insert(id);
    }

    // Reverse-vocab `{id -> text}` for the base vocab range
    // (`mlx_lm/gguf.py:56-58`). The base path skips added-token ids, so
    // every position 0..vocab_size_base that the loop visits MUST have an
    // entry — a hole would mean the tokenizer has no token at that id,
    // which mlx-lm would crash on (KeyError) and we surface as
    // [`Error::Backend`]. Using `Vec<Option<String>>` (rather than a
    // `HashMap<u32, String>`) makes the per-id lookup O(1) and the hole
    // check trivial.
    let full_vocab = hf.get_vocab(true);
    let mut reverse_base_vocab: Vec<Option<String>> = vec![None; vocab_size_base_usize];
    for (text, id) in &full_vocab {
      if (*id as usize) < vocab_size_base_usize {
        reverse_base_vocab[*id as usize] = Some(text.clone());
      }
    }

    let added_u32 = u32::try_from(added_tokens_list.len()).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "HfVocab: added token count",
        "must fit in u32",
        added_tokens_list.len().to_string(),
      ))
    })?;
    let vocab_size = vocab_size_base.checked_add(added_u32).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "vocab_size_base + added",
        "u32",
        [
          ("vocab_size_base", u64::from(vocab_size_base)),
          ("added", u64::from(added_u32)),
        ],
      ))
    })?;

    Ok(HfVocab {
      added_tokens_list,
      added_tokens_ids,
      specials,
      special_ids,
      vocab_size_base,
      vocab_size,
      reverse_base_vocab,
      bos_token_id: tokenizer.bos_token_id(),
      eos_token_id: tokenizer.eos_token_id(),
      unk_token_id: tokenizer.unk_token_id(),
    })
  }

  /// Total vocab size emitted (base + added).
  ///
  /// Mirrors `self.vocab_size` (`mlx_lm/gguf.py:51`).
  pub fn vocab_size(&self) -> u32 {
    self.vocab_size
  }

  /// Base vocab size (without added tokens).
  ///
  /// Mirrors `self.vocab_size_base` (`mlx_lm/gguf.py:50`).
  pub fn vocab_size_base(&self) -> u32 {
    self.vocab_size_base
  }

  /// `bos_token_id` (if any). Mirrors `tokenizer.bos_token_id`
  /// (`mlx_lm/gguf.py:244-247`).
  pub fn bos_token_id(&self) -> Option<u32> {
    self.bos_token_id
  }

  /// `eos_token_id` (if any). Mirrors `tokenizer.eos_token_id`
  /// (`mlx_lm/gguf.py:248-251`).
  pub fn eos_token_id(&self) -> Option<u32> {
    self.eos_token_id
  }

  /// `unk_token_id` (if any). Mirrors `tokenizer.unk_token_id`
  /// (`mlx_lm/gguf.py:252-255`).
  pub fn unk_token_id(&self) -> Option<u32> {
    self.unk_token_id
  }

  /// Classify a base-vocab id — port of `get_token_type`
  /// (`mlx_lm/gguf.py:67-72`).
  fn get_token_type(&self, token_id: u32, token_text: &str) -> TokenType {
    if is_byte_token(token_text) {
      TokenType::Byte
    } else if self.special_ids.contains(&token_id) {
      TokenType::Control
    } else {
      TokenType::Normal
    }
  }

  /// Constant score for every token — port of `get_token_score`
  /// (`mlx_lm/gguf.py:73-75`).
  fn get_token_score(&self, _token_id: u32) -> f32 {
    -1000.0
  }

  /// Iterate the full token list (base, then added), each yielding
  /// `(text, score, toktype)` — port of `all_tokens`
  /// (`mlx_lm/gguf.py:90-92`).
  ///
  /// The caller (typically [`prepare_metadata`]) drains this into
  /// three parallel GGUF metadata fields:
  /// `tokenizer.ggml.tokens` / `.scores` / `.token_type`.
  ///
  /// **Errors** if any base-vocab id 0..[`vocab_size_base`](Self::vocab_size_base)
  /// is missing from the tokenizer's reverse vocab (skipping ids owned by
  /// added tokens, per `mlx_lm/gguf.py:59-62`) — a missing slot would
  /// make `len(tokens) != vocab.vocab_size`, which the reference
  /// asserts on (`mlx_lm/gguf.py:240`) and we surface up-front.
  pub fn all_tokens(&self) -> Result<Vec<(String, f32, TokenType)>> {
    let mut out = Vec::with_capacity(self.vocab_size as usize);

    // hf_tokens — base path. Skip ids that are owned by added tokens
    // (`mlx_lm/gguf.py:59-62`) so we never emit an id twice.
    for id in 0..self.vocab_size_base {
      if self.added_tokens_ids.contains(&id) {
        continue;
      }
      let text = self.reverse_base_vocab[id as usize]
        .as_deref()
        .ok_or_else(|| {
          Error::MissingKey(MissingKeyPayload::new(
            "HfVocab: base vocab token",
            id.to_string(),
          ))
        })?;
      let score = self.get_token_score(id);
      let toktype = self.get_token_type(id, text);
      out.push((text.to_owned(), score, toktype));
    }

    // added_tokens — appended path (`mlx_lm/gguf.py:77-85`). An added
    // token whose id is in `special_ids` classifies as `Control`;
    // everything else is `UserDefined`.
    //
    // Looking the added-token text up in `self.specials` (the
    // `{text → id}` map populated only from `added_tokens_decoder`
    // entries flagged `special=true`) would miss config-declared
    // specials whose `added_tokens_decoder.special` flag is `false` —
    // e.g. a token listed in `tokenizer_config.json`'s
    // `additional_special_tokens` but with `special=false` in the
    // decoder. Such ids ARE unioned into `special_ids` by the
    // constructor (sources (a) AND (b)), but a text-based lookup would
    // classify them as `UserDefined`, emitting a
    // `tokenizer.ggml.token_type` array inconsistent with the
    // constructor's union semantics. Classifying via
    // `special_ids.contains(&id)` directly (matching the base-vocab
    // walk's path) closes the gap.
    for (id, text) in &self.added_tokens_list {
      let (toktype, score) = if self.special_ids.contains(id) {
        // The Python reference passes `""` for the text here
        // (`mlx_lm/gguf.py:80-84`) — no byte-regex match possible — so
        // `get_token_type` resolves to `Control`. We mirror that.
        (self.get_token_type(*id, ""), self.get_token_score(*id))
      } else {
        (TokenType::UserDefined, -1000.0)
      };
      out.push((text.clone(), score, toktype));
    }

    Ok(out)
  }

  /// Whether the vocabulary carries a newline token in either of the
  /// two forms HF tokenizers ship — port of `has_newline_token`
  /// (`mlx_lm/gguf.py:87-88`). Exposed for parity; not used by
  /// [`prepare_metadata`] (the reference uses it only in the LoRA
  /// adapter path, out of scope here).
  pub fn has_newline_token(&self, tokenizer: &crate::tokenizer::Tokenizer) -> bool {
    let vocab = tokenizer.hf().get_vocab(true);
    vocab.contains_key("<0x0A>") || vocab.contains_key("\n")
  }
}

/// Whether `text` looks like a `<0xAB>` byte token — port of the regex
/// `re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token_text)`
/// (`mlx_lm/gguf.py:70`). Six characters, fixed bracket positions,
/// uppercase / lowercase hex either way.
fn is_byte_token(text: &str) -> bool {
  text.len() == 6
    && text.starts_with("<0x")
    && text.ends_with('>')
    && text.as_bytes()[3].is_ascii_hexdigit()
    && text.as_bytes()[4].is_ascii_hexdigit()
}

/// HF → GGUF weight-name remap — port of `translate_weight_names`
/// (`mlx_lm/gguf.py:103-130`).
///
/// Applies a fixed sequence of literal `str.replace` and `re.sub` rules
/// matching the reference. The substitutions are intentionally ordered
/// (e.g. `model.layers.` is folded to `blk.` first so the per-layer
/// suffixes below match `blk.N.*` rather than `model.layers.N.*`).
///
/// The rules (in order) — exactly mirroring `mlx_lm/gguf.py`:
///
/// - `model.layers.` → `blk.`
/// - `block_sparse_moe.gate` → `ffn_gate_inp`
/// - `block_sparse_moe.experts.N.w1.weight` → `ffn_gate.N.weight`
/// - `block_sparse_moe.experts.N.w2.weight` → `ffn_down.N.weight`
/// - `block_sparse_moe.experts.N.w3.weight` → `ffn_up.N.weight`
/// - `mlp.gate_proj` → `ffn_gate`
/// - `mlp.down_proj` → `ffn_down`
/// - `mlp.up_proj` → `ffn_up`
/// - `self_attn.q_proj` → `attn_q`
/// - `self_attn.k_proj` → `attn_k`
/// - `self_attn.v_proj` → `attn_v`
/// - `self_attn.o_proj` → `attn_output`
/// - `input_layernorm` → `attn_norm`
/// - `post_attention_layernorm` → `ffn_norm`
/// - `model.embed_tokens` → `token_embd`
/// - `model.norm` → `output_norm`
/// - `lm_head` → `output`
pub fn translate_weight_names(name: &str) -> String {
  // 1. Per-layer prefix: `model.layers.N.…` → `blk.N.…`
  let mut s = name.replace("model.layers.", "blk.");

  // 2. Mixtral router gate (`mlx_lm/gguf.py:105-106`).
  s = s.replace("block_sparse_moe.gate", "ffn_gate_inp");

  // 3-5. Mixtral expert FFN — `re.sub` over `wK.weight` → `ffn_*.K.weight`.
  // We inline the substitution with a simple parser to avoid pulling in
  // the `regex` crate for three trivial captures (the regex crate is gated
  // on the `lm` feature anyway, but per-call regex construction is
  // expensive). The three rules are identical modulo `wK` ↔ `ffn_*`.
  s = remap_moe_expert(&s, "w1", "ffn_gate");
  s = remap_moe_expert(&s, "w2", "ffn_down");
  s = remap_moe_expert(&s, "w3", "ffn_up");

  // 6-17. Per-component MLP / attention / norm / embed / lm_head
  // (`mlx_lm/gguf.py:118-129`).
  s = s.replace("mlp.gate_proj", "ffn_gate");
  s = s.replace("mlp.down_proj", "ffn_down");
  s = s.replace("mlp.up_proj", "ffn_up");
  s = s.replace("self_attn.q_proj", "attn_q");
  s = s.replace("self_attn.k_proj", "attn_k");
  s = s.replace("self_attn.v_proj", "attn_v");
  s = s.replace("self_attn.o_proj", "attn_output");
  s = s.replace("input_layernorm", "attn_norm");
  s = s.replace("post_attention_layernorm", "ffn_norm");
  s = s.replace("model.embed_tokens", "token_embd");
  s = s.replace("model.norm", "output_norm");
  s = s.replace("lm_head", "output");
  s
}

/// Substitute `block_sparse_moe.experts.N.{wK}.weight` →
/// `{replacement}.N.weight` everywhere in `s`. `N` is one or more ASCII
/// digits — exactly the Python `\d+` capture
/// (`mlx_lm/gguf.py:108-116`).
fn remap_moe_expert(s: &str, wk: &str, replacement: &str) -> String {
  let prefix = "block_sparse_moe.experts.";
  let middle = format!(".{wk}.weight");
  let mut out = String::with_capacity(s.len());
  let mut rest = s;
  while let Some(pos) = rest.find(prefix) {
    out.push_str(&rest[..pos]);
    let tail = &rest[pos + prefix.len()..];
    // Consume digits.
    let digit_end = tail
      .as_bytes()
      .iter()
      .position(|b| !b.is_ascii_digit())
      .unwrap_or(tail.len());
    if digit_end == 0 || !tail[digit_end..].starts_with(&middle) {
      // Not a match — emit the prefix verbatim and advance past it.
      out.push_str(prefix);
      rest = tail;
      continue;
    }
    let digits = &tail[..digit_end];
    out.push_str(&format!("{replacement}.{digits}.weight"));
    rest = &tail[digit_end + middle.len()..];
  }
  out.push_str(rest);
  out
}

/// Apply the GGUF head-interleave permutation to a Q or K attention
/// weight — port of `permute_weights` (`mlx_lm/gguf.py:133-141`).
///
/// GGUF stores attention Q / K weights with the per-head halves
/// interleaved differently from HF — the reference re-orders by
/// reshaping `[D, ...]` → `[n_head_eff, 2, D / n_head_eff / 2, ...]`,
/// swapping the middle two axes, and reshaping back to `[D, ...]`.
///
/// `n_head_kv` (if `Some` and different from `n_head`) overrides
/// `n_head` — the reference's `if n_head_kv is not None and n_head !=
/// n_head_kv: n_head = n_head_kv` (`mlx_lm/gguf.py:134-135`). [`convert_to_gguf`]
/// passes:
/// - Q weights: `n_head_kv = num_attention_heads` (i.e. same) — `n_head` stays
/// - K weights: `n_head_kv = num_key_value_heads` (GQA) — `n_head` overridden
///
/// Errors if `weights.shape[0]` is not divisible by `2 * effective n_head`
/// (the reference would silently produce a wrongly-shaped reshape; we
/// surface the mismatch up-front as a recoverable
/// [`Error::Backend`]).
pub fn permute_weights(weights: &Array, n_head: i32, n_head_kv: Option<i32>) -> Result<Array> {
  let effective = match n_head_kv {
    Some(kv) if kv != n_head => kv,
    _ => n_head,
  };
  if effective <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "permute_weights: n_head",
      "must be positive",
      format!("{effective}"),
    )));
  }
  let original_shape = weights.shape();
  let original_shape_i32: Vec<i32> = original_shape
    .iter()
    .map(|&d| {
      i32::try_from(d).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "permute_weights: shape dim",
          "must fit in i32",
          d.to_string(),
        ))
      })
    })
    .collect::<Result<_>>()?;
  if original_shape.is_empty() {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "permute_weights: weights rank",
      "must be >= 1 (requires at least 1-D weights)",
    )));
  }
  let d0 = original_shape_i32[0];
  let twice = 2_i32.checked_mul(effective).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "permute_weights: 2 * n_head",
      "i32",
      [("two", 2u64), ("n_head", effective as u64)],
    ))
  })?;
  if d0 % twice != 0 {
    return Err(Error::DivisibilityConstraint(
      crate::error::DivisibilityConstraintPayload::new(
        "permute_weights: leading dim must be divisible by 2 * n_head",
        "leading_dim",
        d0 as u64,
        "2*n_head",
        twice as u64,
      ),
    ));
  }
  let split = d0 / twice;

  // reshape to [n_head_eff, 2, d0 / n_head_eff / 2, *rest]
  let mut reshape_dims: Vec<i32> = Vec::with_capacity(3 + original_shape_i32.len() - 1);
  reshape_dims.push(effective);
  reshape_dims.push(2);
  reshape_dims.push(split);
  reshape_dims.extend_from_slice(&original_shape_i32[1..]);

  let reshaped = weights.reshape(&&reshape_dims[..])?;
  // swapaxes(1, 2)
  let swapped = reshaped.swapaxes(1, 2)?;
  // back to the original shape
  swapped.reshape(&&original_shape_i32[..])
}

/// Build the GGUF metadata map from a checkpoint's [`Config`] and the
/// packed [`HfVocab`] — port of `prepare_metadata`
/// (`mlx_lm/gguf.py:144-258`).
///
/// Mirrors the reference's two-stage construction:
///
/// 1. The `general.*` + `llama.*` keys (`mlx_lm/gguf.py:144-208`,
///    optional-value-aware: only present when the source field is
///    populated; `mlx_lm/gguf.py:257` filters `v is None` away).
/// 2. Rope-scaling override (`mlx_lm/gguf.py:210-217`), file-type +
///    quantization-version + alignment (`mlx_lm/gguf.py:219-229`),
///    architecture / name strings (`mlx_lm/gguf.py:227-228`).
/// 3. `tokenizer.ggml.*` vocab block (`mlx_lm/gguf.py:231-255`),
///    asserts `len(tokens) == vocab.vocab_size`.
///
/// The reference hard-codes the `llama.*` prefix
/// (`mlx_lm/gguf.py:146-208`); [`convert_to_gguf`] rejects model types
/// outside the LM-side supported set before calling this so a non-Llama
/// tag is never silently emitted on a non-Llama checkpoint.
///
/// `name_override` (mlx-lm computes `config.get("_name_or_path",
/// "llama").split("/")[-1]`; `mlx_lm/gguf.py:227`) is supplied by the
/// caller so [`prepare_metadata`] does not need to know about the
/// untyped `_name_or_path` JSON key (it lives outside [`Config`]'s
/// typed subset).
pub fn prepare_metadata(
  config: &Config,
  raw_config: &serde_json::Value,
  vocab: &HfVocab,
) -> Result<HashMap<String, GgufMetadata>> {
  let mut metadata: HashMap<String, GgufMetadata> = HashMap::new();

  // Helper: untyped `raw_config` field lookup, accepting both an integer
  // and a float JSON value. Returns `Some(value)` only on success.
  let get_u32 = |key: &str| -> Option<u32> {
    raw_config
      .get(key)
      .and_then(|v| v.as_u64())
      .and_then(|n| u32::try_from(n).ok())
  };
  let get_f32 = |key: &str| -> Option<f32> {
    raw_config
      .get(key)
      .and_then(|v| v.as_f64())
      .map(|f| f as f32)
  };

  // `general.name` — initial placeholder mirroring the reference
  // (`mlx_lm/gguf.py:146`); overridden below from `_name_or_path`
  // (`mlx_lm/gguf.py:227`).
  metadata.insert(
    "general.name".to_string(),
    GgufMetadata::String("llama".to_string()),
  );

  // `llama.context_length` ← max_position_embeddings
  if let Some(v) = get_u32("max_position_embeddings") {
    metadata.insert(
      "llama.context_length".to_string(),
      GgufMetadata::Array(scalar_u32(v)?),
    );
  }
  // `llama.embedding_length` ← hidden_size
  if let Some(v) = get_u32("hidden_size") {
    metadata.insert(
      "llama.embedding_length".to_string(),
      GgufMetadata::Array(scalar_u32(v)?),
    );
  }
  // `llama.block_count` ← num_hidden_layers
  if let Some(v) = get_u32("num_hidden_layers") {
    metadata.insert(
      "llama.block_count".to_string(),
      GgufMetadata::Array(scalar_u32(v)?),
    );
  }
  // `llama.feed_forward_length` ← intermediate_size
  if let Some(v) = get_u32("intermediate_size") {
    metadata.insert(
      "llama.feed_forward_length".to_string(),
      GgufMetadata::Array(scalar_u32(v)?),
    );
  }
  // `llama.rope.dimension_count` ← hidden_size / num_attention_heads
  if let (Some(hidden), Some(heads)) = (get_u32("hidden_size"), get_u32("num_attention_heads"))
    && heads > 0
  {
    metadata.insert(
      "llama.rope.dimension_count".to_string(),
      GgufMetadata::Array(scalar_u32(hidden / heads)?),
    );
  }
  // `llama.attention.head_count` ← num_attention_heads
  if let Some(v) = get_u32("num_attention_heads") {
    metadata.insert(
      "llama.attention.head_count".to_string(),
      GgufMetadata::Array(scalar_u32(v)?),
    );
    // `llama.attention.head_count_kv` ← num_key_value_heads || num_attention_heads
    let kv = get_u32("num_key_value_heads").unwrap_or(v);
    metadata.insert(
      "llama.attention.head_count_kv".to_string(),
      GgufMetadata::Array(scalar_u32(kv)?),
    );
  }
  // `llama.expert_count` ← num_local_experts (Mixtral)
  if let Some(v) = get_u32("num_local_experts") {
    metadata.insert(
      "llama.expert_count".to_string(),
      GgufMetadata::Array(scalar_u32(v)?),
    );
  }
  // `llama.expert_used_count` ← num_experts_per_tok (Mixtral)
  if let Some(v) = get_u32("num_experts_per_tok") {
    metadata.insert(
      "llama.expert_used_count".to_string(),
      GgufMetadata::Array(scalar_u32(v)?),
    );
  }
  // `llama.attention.layer_norm_rms_epsilon` ← rms_norm_eps (default 1e-5)
  if let Some(v) = get_f32("rms_norm_eps") {
    metadata.insert(
      "llama.attention.layer_norm_rms_epsilon".to_string(),
      GgufMetadata::Array(scalar_f32(v)?),
    );
  }
  // `llama.rope.freq_base` ← rope_theta (default 10000)
  if let Some(v) = get_f32("rope_theta") {
    metadata.insert(
      "llama.rope.freq_base".to_string(),
      GgufMetadata::Array(scalar_f32(v)?),
    );
  }

  // Rope-scaling override block (`mlx_lm/gguf.py:210-217`). The
  // reference checks `rope_scaling.get("type")` truthy AND only writes
  // metadata when the type is "linear" (other types silently skip).
  if let Some(rope_scaling) = raw_config.get("rope_scaling").and_then(|v| v.as_object())
    && let Some(typ) = rope_scaling.get("type").and_then(|v| v.as_str())
    && typ == "linear"
  {
    metadata.insert(
      "llama.rope.scaling.type".to_string(),
      GgufMetadata::String("linear".to_string()),
    );
    if let Some(factor) = rope_scaling.get("factor").and_then(|v| v.as_f64()) {
      metadata.insert(
        "llama.rope.scaling.factor".to_string(),
        GgufMetadata::Array(scalar_f32(factor as f32)?),
      );
    }
  }

  // `general.file_type` (`mlx_lm/gguf.py:219-222`) — always the F16 tag.
  metadata.insert(
    "general.file_type".to_string(),
    GgufMetadata::Array(scalar_u32(GgmlFileType::F16 as u32)?),
  );
  // `general.quantization_version` (`mlx_lm/gguf.py:223-226`) — same value.
  metadata.insert(
    "general.quantization_version".to_string(),
    GgufMetadata::Array(scalar_u32(GgmlFileType::F16 as u32)?),
  );
  // `general.name` overwrite (`mlx_lm/gguf.py:227`) — the original
  // initial placeholder above is replaced with the `_name_or_path`
  // basename or `"llama"`.
  let name_or_path = raw_config
    .get("_name_or_path")
    .and_then(|v| v.as_str())
    .unwrap_or("llama");
  let base_name = name_or_path
    .rsplit('/')
    .next()
    .unwrap_or("llama")
    .to_owned();
  metadata.insert("general.name".to_string(), GgufMetadata::String(base_name));
  // `general.architecture` (`mlx_lm/gguf.py:228`).
  metadata.insert(
    "general.architecture".to_string(),
    GgufMetadata::String("llama".to_string()),
  );
  // `general.alignment` (`mlx_lm/gguf.py:229`).
  metadata.insert(
    "general.alignment".to_string(),
    GgufMetadata::Array(scalar_u32(32)?),
  );

  // Tokenizer vocab block (`mlx_lm/gguf.py:231-255`).
  metadata.insert(
    "tokenizer.ggml.model".to_string(),
    GgufMetadata::String("llama".to_string()),
  );

  let triples = vocab.all_tokens()?;
  // assert len(tokens) == vocab.vocab_size (`mlx_lm/gguf.py:240`).
  if triples.len() as u32 != vocab.vocab_size {
    return Err(Error::LengthMismatch(
      crate::error::LengthMismatchPayload::new(
        "prepare_metadata: emitted tokens vs vocab.vocab_size",
        vocab.vocab_size as usize,
        triples.len(),
      ),
    ));
  }
  let mut tokens = Vec::with_capacity(triples.len());
  let mut scores = Vec::with_capacity(triples.len());
  let mut toktypes = Vec::with_capacity(triples.len());
  for (text, score, toktype) in triples {
    tokens.push(text);
    scores.push(score);
    toktypes.push(toktype as u32);
  }
  metadata.insert(
    "tokenizer.ggml.tokens".to_string(),
    GgufMetadata::StringList(tokens),
  );
  metadata.insert(
    "tokenizer.ggml.scores".to_string(),
    GgufMetadata::Array(Array::from_slice::<f32>(&scores, &(scores.len(),))?),
  );
  metadata.insert(
    "tokenizer.ggml.token_type".to_string(),
    GgufMetadata::Array(Array::from_slice::<u32>(&toktypes, &(toktypes.len(),))?),
  );
  if let Some(id) = vocab.bos_token_id() {
    metadata.insert(
      "tokenizer.ggml.bos_token_id".to_string(),
      GgufMetadata::Array(scalar_u32(id)?),
    );
  }
  if let Some(id) = vocab.eos_token_id() {
    metadata.insert(
      "tokenizer.ggml.eos_token_id".to_string(),
      GgufMetadata::Array(scalar_u32(id)?),
    );
  }
  if let Some(id) = vocab.unk_token_id() {
    metadata.insert(
      "tokenizer.ggml.unknown_token_id".to_string(),
      GgufMetadata::Array(scalar_u32(id)?),
    );
  }

  // The reference's `metadata = {k: v for k, v in metadata.items() if v is
  // not None}` (`mlx_lm/gguf.py:257`) is implicit here — we only insert
  // values that exist, so no post-filter is needed.

  // Suppress unused warning: `_ = config` if the future may need other typed-config-only keys.
  let _ = config;
  Ok(metadata)
}

/// Build a 0-D `u32` scalar mlx [`Array`] — the dtype `prepare_metadata`
/// emits for every integer-typed metadata value
/// (`mlx_lm/gguf.py:147-187`: `mx.array(int, dtype=mx.uint32)`).
///
/// 1-D `[value]` is used (mlx-c's gguf path handles 0/1-D scalars
/// equivalently; see `mlx/io/gguf.cpp:354-360`), which matches the
/// reference's behavior bit-for-bit when read back.
fn scalar_u32(value: u32) -> Result<Array> {
  Array::from_slice::<u32>(&[value], &(1_usize,))
}

/// Build a 0-D `f32` scalar mlx [`Array`] — the dtype `prepare_metadata`
/// emits for float metadata (`mlx_lm/gguf.py:199, 204, 217`:
/// `mx.array(float, dtype=mx.float32)`).
fn scalar_f32(value: f32) -> Result<Array> {
  Array::from_slice::<f32>(&[value], &(1_usize,))
}

/// The set of `model_type` values the LM-side GGUF export pipeline can
/// faithfully tag.
///
/// The reference's `prepare_metadata` hard-codes the `llama.*` key
/// prefix (`mlx_lm/gguf.py:146-208`) — it is faithful only for models
/// that share Llama's config keys + attention shape. We mirror the
/// upstream supported set: Llama and its close relatives, plus Mistral
/// / Mixtral (the only checkpoints `mlx_lm.gguf.convert_to_gguf` is
/// known to round-trip through llama.cpp loaders today, since the
/// reference also defaults the architecture tag to `"llama"` for them).
///
/// A `model_type` outside this set causes [`convert_to_gguf`] to return
/// a clear [`Error::Backend`] rather than silently produce a corrupt
/// GGUF (fail-fast on the LM-side surface; per-arch hooks are out of
/// scope).
const SUPPORTED_MODEL_TYPES: &[&str] = &["llama", "mistral", "mixtral"];

/// Arguments to [`convert_to_gguf`] — the structured form of
/// `convert_to_gguf`'s positional parameters
/// (`mlx_lm/gguf.py:261-266`).
///
/// Carries the on-disk paths and nothing else: the reference also takes
/// `weights` and `config` positionally, but we re-load them inside
/// [`convert_to_gguf`] (via [`crate::lm::load::load`]) so the driver is
/// self-contained — the user only points at the source/destination
/// paths.
#[derive(Debug, Clone)]
pub struct ConvertToGgufArgs {
  /// Source model directory (HF-style `config.json` + safetensors
  /// shards + `tokenizer.json`). The same shape the loaders
  /// consume.
  pub model_path: PathBuf,
  /// Destination `.gguf` file. The reference appends `.gguf` if missing
  /// (`mlx/io/gguf.cpp:299-301`); the underlying [`crate::io::save_gguf`]
  /// inherits that behavior, so a path without the `.gguf` suffix still
  /// produces a valid file but written to `{path}.gguf`.
  pub gguf_path: PathBuf,
}

/// Top-level GGUF export driver — port of `convert_to_gguf`
/// (`mlx_lm/gguf.py:261-314`).
///
/// Pipeline (the Python reference receives `weights` + `config` already
/// loaded by its caller, so it has no choice about ordering;
/// `convert_to_gguf` here owns the load, so we run every fail-fast
/// validation BEFORE the multi-GB weight read to avoid OOM'ing on a
/// checkpoint we are about to reject):
///
/// 1. Load the (small) `config.json` via [`crate::lm::load::load_config`].
///    Yields `(Config, raw_json)` — the typed subset plus the raw JSON
///    body for fields outside [`Config`]'s typed subset (e.g.
///    `intermediate_size`, `_name_or_path`, `rope_scaling.*`,
///    `quantization_config`).
/// 2. Fail-fast validation (BEFORE the weight load):
///    - **2a.** Reject a non-Llama-family `model_type` (a `model_type`
///      outside the LM-side supported set `{"llama", "mistral",
///      "mixtral"}` is rejected up front so a non-Llama tag is never
///      silently emitted — see the module-doc scope boundary on
///      per-model-arch porting).
///    - **2b.** Reject a quantized checkpoint — the reference raises
///      `NotImplementedError("Conversion of quantized models is not yet
///      supported.")` (`mlx_lm/gguf.py:271-274`); we surface the same
///      as [`Error::Backend`]. Both `config["quantization"]` (already
///      typed on [`Config`]) and `config["quantization_config"]` (some
///      HF checkpoints + post-quantize artifacts use the longer key)
///      trip this gate.
///    - **2c.** Build the tokenizer via [`crate::lm::load::load_tokenizer`]
///      (`mlx_lm/gguf.py:297-298` — the reference only `Path.exists()`-
///      checks `tokenizer.json` because its caller has the tokenizer
///      already; our driver owns the load, so we tighten the gate to
///      a full parse so a directory / unreadable / malformed
///      `tokenizer.json` cannot OOM us on the weight read). The
///      resolved [`crate::tokenizer::Tokenizer`] is threaded to the
///      [`HfVocab`] builder below — no re-load.
/// 3. Load the multi-GB weights ONLY after every (2) validation has
///    passed (the tokenizer is already in hand from 2c).
/// 4. Permute Q / K attention weights via [`permute_weights`]
///    (`mlx_lm/gguf.py:277-292`).
/// 5. Remap HF weight names to GGUF via [`translate_weight_names`]
///    (`mlx_lm/gguf.py:295`).
/// 6. Build vocab + metadata via [`HfVocab`] + [`prepare_metadata`]
///    (`mlx_lm/gguf.py:300-301`).
/// 7. Normalize weight dtypes (`mlx_lm/gguf.py:303-310`):
///    bf16 weights cast through f32 → f16; any weight whose name carries
///    "norm" cast to f32; everything else passed through unchanged.
/// 8. Write via [`crate::io::save_gguf`] (`mlx_lm/gguf.py:313`).
///
/// Returns `Ok(())` on success; any failure (load / unsupported
/// quantization / unsupported arch / shape mismatch in
/// [`permute_weights`] / IO) is an [`Error::Backend`] whose message
/// names the offending input.
pub fn convert_to_gguf(args: &ConvertToGgufArgs) -> Result<()> {
  // 1. Load the (small) `config.json` FIRST — the bounded read is fast
  //    and a few kilobytes at most. `load_config` already applies the
  //    `generation_config.json` eos override; we additionally need the
  //    raw `config.json` body for the untyped fields `prepare_metadata`
  //    consumes (`intermediate_size`, `max_position_embeddings`,
  //    `_name_or_path`, `rope_scaling.*`).
  let (config, raw_json) = crate::lm::load::load_config(&args.model_path)?;

  // 2. Fail-fast validation block — runs BEFORE the multi-GB weight
  //    load + tokenizer build so an unsupported / quantized checkpoint
  //    cannot OOM us on the weight read before the rejection path
  //    fires. The reference Python (`mlx_lm/gguf.py:261-274`) receives
  //    `weights` + `config` already loaded by its caller so the order
  //    there is forced; our `convert_to_gguf` owns the load, so we
  //    reject up front. Tests planting a sentinel weight file that
  //    would error on read (`convert_to_gguf_rejects_unsupported_arch`
  //    + `convert_to_gguf_rejects_quantized`) prove these paths return
  //    the validation `Err(Backend)` WITHOUT touching the weights.

  //   2a. Architecture gate:
  //       reject any `model_type` outside the LM-side supported set so a
  //       non-Llama tag is never silently emitted on a non-Llama
  //       checkpoint. The reference's `prepare_metadata` hard-codes the
  //       `llama.*` key prefix — see `SUPPORTED_MODEL_TYPES`.
  if !SUPPORTED_MODEL_TYPES.contains(&config.model_type()) {
    return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
      "convert_to_gguf: model_type (LM-side GGUF exporter supported set)",
      config.model_type().to_string(),
      SUPPORTED_MODEL_TYPES,
    )));
  }

  //   2b. Quantized → reject (`mlx_lm/gguf.py:270-274`). [`Config`] only
  //       carries the strongly-typed `quantization` block, but a few HF
  //       checkpoints (and the mlx-lm convert pipeline post-quantize
  //       artifacts) ship the same payload under the `quantization_config`
  //       key — we reject either so an unsupported quantized checkpoint
  //       cannot slip through. The GGUF LM export targets dense F16/F32;
  //       dequantize first via `lm::convert` if needed.
  let raw_config: serde_json::Value = serde_json::from_str(&raw_json).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "convert_to_gguf: cannot re-parse config.json",
      "JSON",
      e,
    ))
  })?;
  if config.quantization.is_some() || raw_config.get("quantization_config").is_some() {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "convert_to_gguf: checkpoint quantization",
      "must be None (the GGUF LM export targets dense F16/F32 GGUF; dequantize first \
        via lm::convert)",
    )));
  }

  //   2c. Tokenizer build — fail-fast. Merely checking that
  //       `tokenizer.json` exists via `Path::exists()` would accept a
  //       directory at that path, a zero-byte file, malformed JSON, or a
  //       structurally-invalid tokenizer — all of which would force the
  //       multi-GB weight read before the tokenizer error surfaced.
  //       Calling `load_tokenizer` here parses the tokenizer up front
  //       (the file is at most a few MB and the parse is cheap relative
  //       to the weight load). The resolved `Tokenizer` is then threaded
  //       down to the `HfVocab` builder below so we never re-load.
  //
  //       The reference Python (`mlx_lm/gguf.py:297-298`) uses a
  //       `Path.exists()` check because its caller has already loaded
  //       the tokenizer for it — our `convert_to_gguf` owns the load,
  //       so we tighten the gate to match the reference's effective
  //       contract (a bad tokenizer cannot OOM us on the weight read).
  let tokenizer = crate::lm::load::load_tokenizer(&args.model_path, &config)?;

  //   2d. Attention head counts. `num_attention_heads` /
  //       `num_key_value_heads` are required typed `i32` fields on
  //       [`Config`] — `load_config` (step 1) already rejects a config
  //       whose head counts are missing or non-integer, so they are read
  //       straight off the typed config here (no second `raw_config`
  //       parse, and resolved BEFORE the multi-GB weight load). They feed
  //       the Q/K permute in step 4.
  let num_attention_heads = config.num_attention_heads;
  let num_key_value_heads = config.num_key_value_heads;

  // 3. NOW load the multi-GB weights — only after every fail-fast
  //    validation in (2) has passed (including the tokenizer parse in
  //    2c), so an unsupported / malformed checkpoint never pays the
  //    weight-load cost. The tokenizer resolved above is re-used for
  //    the `HfVocab` builder; no second load.
  let weights = crate::lm::load::load_weights(&args.model_path)?;

  // 4. Permute Q / K attention weights. The reference uses
  //    `n_head_kv = num_attention_heads` for Q and `num_key_value_heads`
  //    for K (`mlx_lm/gguf.py:278-289`).
  //
  //    We iterate the weight map and rebuild it (rather than mutate in
  //    place) so the permuted arrays own fresh storage and the original
  //    map can be dropped — mlx Array is `!Send` and `!Clone`, so the
  //    only no-copy bridge is `try_clone`; we go through `permute_weights`
  //    which builds a new array regardless.
  let mut permuted: Weights = HashMap::with_capacity(weights.len());
  for (key, val) in weights {
    if key.contains("self_attn.q_proj.weight") {
      permuted.insert(
        key,
        permute_weights(&val, num_attention_heads, Some(num_attention_heads))?,
      );
    } else if key.contains("self_attn.k_proj.weight") {
      permuted.insert(
        key,
        permute_weights(&val, num_attention_heads, Some(num_key_value_heads))?,
      );
    } else {
      permuted.insert(key, val);
    }
  }

  // 5. Rename weights for GGUF. We build a `BTreeMap` so the on-disk
  //    write order is deterministic (mlx-c's gguf writer iterates the
  //    `unordered_map` in implementation-defined order; for byte-for-byte
  //    parity with mlx-lm's output we don't try to match insertion order
  //    but we do want a stable order for our own round-trip tests).
  let renamed: BTreeMap<String, Array> = permuted
    .into_iter()
    .map(|(k, v)| (translate_weight_names(&k), v))
    .collect();

  // 6. Vocab + metadata.
  let vocab = HfVocab::from_tokenizer(&tokenizer)?;
  let metadata = prepare_metadata(&config, &raw_config, &vocab)?;

  // 7. Normalize dtypes (`mlx_lm/gguf.py:303-310`):
  //    - bf16 → cast through f32 to f16
  //    - any name containing "norm" → cast to f32
  //    - else: pass through
  let mut normalized: HashMap<String, Array> = HashMap::with_capacity(renamed.len());
  for (key, val) in renamed {
    let dt = val.dtype()?;
    let out = if dt == Dtype::BF16 {
      // `v.astype(mx.float32).astype(mx.float16)` — the reference goes
      // through f32 explicitly. mlx's astype does the cast in one step
      // either way, but we keep the two-stage form so a future mlx-c
      // change to bf16 → f16 direct casts (e.g. precision loss
      // semantics) does not silently diverge from the reference.
      let f32_arr = val.astype(Dtype::F32)?;
      f32_arr.astype(Dtype::F16)?
    } else if key.contains("norm") {
      val.astype(Dtype::F32)?
    } else {
      val
    };
    normalized.insert(key, out);
  }

  // 8. Write the GGUF file.
  crate::io::save_gguf(&args.gguf_path, &normalized, &metadata)
}

// ─────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod tests;
