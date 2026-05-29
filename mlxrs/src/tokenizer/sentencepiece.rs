//! SentencePiece Unigram / BPE tokenizer with protobuf reader + Viterbi
//! lattice + byte-fallback decoding.
//!
//! Faithful port of
//! [`mlx-audio-swift/Sources/MLXAudioCore/SentencePieceTokenizer.swift`][swift-ref]:
//! a self-contained SentencePiece reader that consumes either a raw
//! `*.model` protobuf file (the upstream SentencePiece serialization
//! format) or the JSON-flavored `tokenizer.json` "model" subtree HF emits
//! for SPM-style tokenizers. Pieces are scored by the model; the Unigram
//! path runs a token lattice (Viterbi) over UTF-8 character positions
//! and picks the best path, while the BPE path greedily merges adjacent
//! symbols by score. Both paths fall back to per-byte `<0xHH>` pieces
//! for any input character missing from the trained vocabulary.
//!
//! Lives at this top-level path (not `audio/...`) because the SPM
//! tokenizer is reusable beyond STT — the same protobuf format underpins
//! most LLM tokenizers shipped as `tokenizer.model` (Llama, T5, Gemma,
//! etc.). It is gated under the `audio` feature for now only because
//! `crate::audio::stt::streaming` is the first consumer; promote to a
//! standalone feature when a non-audio caller needs it.
//!
//! The protobuf reader is **hand-rolled** (~80 LOC, ~4 fields, wire types
//! 0/1/2/5 only) so the dep graph stays minimal — adding `prost` /
//! `prost-build` / a vendored `.proto` file would pull in a non-trivial
//! transitive dep tree for parsing only the `pieces` and
//! `trainer_spec.model_type` subset of the full SentencePiece schema.
//! See the [hand-rolled vs. `prost` decision][decision] in this module's
//! commit message.
//!
//! [swift-ref]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioCore/SentencePieceTokenizer.swift
//! [decision]: https://github.com/uqio/mlxrs/pulls?q=AUDIO-A14
//!
//! # Example
//!
//! ```ignore
//! use mlxrs::tokenizer::sentencepiece::SentencePieceTokenizer;
//! use std::path::Path;
//!
//! let tok = SentencePieceTokenizer::from_model_file(Path::new("tokenizer.model"))?;
//! let ids = tok.encode_with_byte_fallback("hello world");
//! let text = tok.decode(&ids);
//! # Ok::<_, mlxrs::Error>(())
//! ```

#[cfg(feature = "tokenizer-config")]
use serde_json::Value as JsonValue;

use std::{collections::HashMap, fs, path::Path};

use smol_str::format_smolstr;

use crate::error::{
  ArithmeticOverflowPayload, Error, FileIoPayload, FileOp, MissingFieldPayload, ParsePayload,
  Result, UnknownEnumValuePayload,
};

/// SentencePiece piece-type enum, matching the upstream
/// `ModelProto.SentencePiece.Type` ordinals.
///
/// `#[non_exhaustive]` because the upstream SentencePiece protobuf schema
/// can gain new ordinals in future releases. The `Unknown(i32)` variant
/// captures any ordinal not recognized at compile time, preserving round-trip
/// identity: `from_raw(x.as_raw()) == x` for every value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum SentencePiecePieceType {
  /// Trained vocabulary piece.
  Normal,
  /// Unknown / OOV catch-all.
  Unknown,
  /// Reserved control token (BOS/EOS/PAD).
  Control,
  /// User-defined token (atomic — never split during BPE merges).
  UserDefined,
  /// Unused vocabulary entry (skipped during decode).
  Unused,
  /// Byte-fallback piece (e.g. `<0xFF>`).
  Byte,
  /// Unrecognized ordinal from a future or extended SentencePiece schema.
  UnknownOrdinal(i32),
}

impl SentencePiecePieceType {
  /// Lowercase string identifier for this piece type.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Normal => "normal",
      Self::Unknown => "unknown",
      Self::Control => "control",
      Self::UserDefined => "user_defined",
      Self::Unused => "unused",
      Self::Byte => "byte",
      Self::UnknownOrdinal(_) => "unknown",
    }
  }

  /// Raw ordinal as stored in the protobuf.
  pub fn as_raw(self) -> i32 {
    match self {
      Self::Normal => 1,
      Self::Unknown => 2,
      Self::Control => 3,
      Self::UserDefined => 4,
      Self::Unused => 5,
      Self::Byte => 6,
      Self::UnknownOrdinal(n) => n,
    }
  }

  fn from_raw(raw: u64) -> Self {
    match raw {
      1 => SentencePiecePieceType::Normal,
      2 => SentencePiecePieceType::Unknown,
      3 => SentencePiecePieceType::Control,
      4 => SentencePiecePieceType::UserDefined,
      5 => SentencePiecePieceType::Unused,
      6 => SentencePiecePieceType::Byte,
      n => SentencePiecePieceType::UnknownOrdinal(n as i32),
    }
  }
}

/// SentencePiece training algorithm — Unigram (default) or BPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
pub enum SentencePieceModelType {
  /// Unigram language model — Viterbi-decoded.
  Unigram,
  /// Byte-pair encoding — greedy-merge-decoded.
  Bpe,
}

impl SentencePieceModelType {
  /// Lowercase string identifier for this model type.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Unigram => "unigram",
      Self::Bpe => "bpe",
    }
  }

  fn from_raw(raw: u64) -> Option<Self> {
    match raw {
      1 => Some(SentencePieceModelType::Unigram),
      2 => Some(SentencePieceModelType::Bpe),
      _ => None,
    }
  }
}

/// A single SentencePiece vocabulary entry — `(token text, log-score,
/// piece type)`.
#[derive(Debug, Clone)]
pub struct SentencePieceToken {
  /// The piece string (UTF-8). May contain the U+2581 metaspace marker
  /// `▁` for word-initial pieces and `<0xHH>` byte-fallback entries.
  token: String,
  /// Per-piece log-probability score from the trained model. Higher is
  /// more likely; the Unigram Viterbi maximizes the sum of these.
  score: f32,
  /// Piece category — controls byte-fallback / decode-skip / atomic-BPE
  /// behavior.
  piece_type: SentencePiecePieceType,
}

impl SentencePieceToken {
  /// Build a piece with the given token, score, and piece type.
  pub fn new(token: impl Into<String>, score: f32, piece_type: SentencePiecePieceType) -> Self {
    Self {
      token: token.into(),
      score,
      piece_type,
    }
  }

  /// The piece string (UTF-8).
  #[inline(always)]
  pub fn token(&self) -> &str {
    &self.token
  }

  /// Per-piece log-probability score.
  #[inline(always)]
  pub fn score(&self) -> f32 {
    self.score
  }

  /// Piece category.
  #[inline(always)]
  pub fn piece_type(&self) -> SentencePiecePieceType {
    self.piece_type
  }
}

/// Minimal hand-rolled protobuf reader for the
/// `ModelProto` subset SentencePiece serializes. Handles wire types `0`
/// (varint), `1` (fixed64), `2` (length-delimited), and `5` (fixed32);
/// any other wire type errors with [`Error::UnknownEnumValue`].
struct SentencePieceProtobufReader<'a> {
  data: &'a [u8],
  index: usize,
}

impl<'a> SentencePieceProtobufReader<'a> {
  fn new(data: &'a [u8]) -> Self {
    Self { data, index: 0 }
  }

  fn is_at_end(&self) -> bool {
    self.index >= self.data.len()
  }

  fn read_varint(&mut self) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    while self.index < self.data.len() && shift < 64 {
      let byte = self.data[self.index];
      self.index += 1;
      value |= u64::from(byte & 0x7f) << shift;
      if byte & 0x80 == 0 {
        return Ok(value);
      }
      shift += 7;
    }
    Err(Error::Backend(
      "SentencePiece protobuf: malformed varint".into(),
    ))
  }

  fn read_length_delimited(&mut self) -> Result<&'a [u8]> {
    let length = self.read_varint()? as usize;
    let end = self
      .index
      .checked_add(length)
      .ok_or(Error::ArithmeticOverflow(ArithmeticOverflowPayload::new(
        "SentencePiece protobuf: length-delimited field",
        "usize",
      )))?;
    if end > self.data.len() {
      return Err(Error::Backend(
        "SentencePiece protobuf: truncated length-delimited field".into(),
      ));
    }
    let slice = &self.data[self.index..end];
    self.index = end;
    Ok(slice)
  }

  fn read_fixed32(&mut self) -> Result<u32> {
    let end = self.index.checked_add(4).ok_or(Error::ArithmeticOverflow(
      ArithmeticOverflowPayload::new("SentencePiece protobuf: fixed32 offset", "usize"),
    ))?;
    if end > self.data.len() {
      return Err(Error::Backend(
        "SentencePiece protobuf: truncated fixed32 field".into(),
      ));
    }
    let slice = &self.data[self.index..end];
    self.index = end;
    let mut value: u32 = 0;
    for (i, &b) in slice.iter().enumerate() {
      value |= u32::from(b) << (i * 8);
    }
    Ok(value)
  }

  fn skip_field(&mut self, wire_type: u64) -> Result<()> {
    match wire_type {
      0 => {
        let _ = self.read_varint()?;
      }
      1 => {
        let end = self.index.checked_add(8).ok_or(Error::ArithmeticOverflow(
          ArithmeticOverflowPayload::new("SentencePiece protobuf: fixed64 offset", "usize"),
        ))?;
        if end > self.data.len() {
          return Err(Error::Backend(
            "SentencePiece protobuf: truncated fixed64 field".into(),
          ));
        }
        self.index = end;
      }
      2 => {
        let _ = self.read_length_delimited()?;
      }
      5 => {
        let _ = self.read_fixed32()?;
      }
      other => {
        return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
          "SentencePiece protobuf: wire type",
          format_smolstr!("{other}"),
          &[
            "0 (varint)",
            "1 (fixed64)",
            "2 (length-delimited)",
            "5 (fixed32)",
          ],
        )));
      }
    }
    Ok(())
  }
}

/// Parsed `ModelProto` payload — the pieces + resolved unknown id + model
/// type. Returned by [`parse_pieces`].
struct ParsedModel {
  pieces: Vec<SentencePieceToken>,
  unknown_token_id: usize,
  model_type: SentencePieceModelType,
}

fn parse_pieces(data: &[u8]) -> Result<ParsedModel> {
  let mut reader = SentencePieceProtobufReader::new(data);
  let mut pieces: Vec<SentencePieceToken> = Vec::new();
  let mut unknown_token_id: Option<usize> = None;
  let mut model_type: SentencePieceModelType = SentencePieceModelType::Unigram;

  while !reader.is_at_end() {
    let key = reader.read_varint()?;
    let field_number = key >> 3;
    let wire_type = key & 0x7;

    if field_number == 1 && wire_type == 2 {
      let piece_data = reader.read_length_delimited()?;
      if let Some(piece) = parse_piece(piece_data)? {
        if piece.piece_type() == SentencePiecePieceType::Unknown && unknown_token_id.is_none() {
          unknown_token_id = Some(pieces.len());
        }
        pieces.push(piece);
      }
    } else if field_number == 2 && wire_type == 2 {
      let trainer_spec_data = reader.read_length_delimited()?;
      if let Some(t) = parse_trainer_spec_model_type(trainer_spec_data)? {
        model_type = t;
      }
    } else {
      reader.skip_field(wire_type)?;
    }
  }

  if pieces.is_empty() {
    return Err(Error::Backend(
      "SentencePiece model: did not contain any vocabulary pieces".into(),
    ));
  }

  let resolved_unknown_id = unknown_token_id
    .or_else(|| pieces.iter().position(|p| p.token() == "<unk>"))
    .unwrap_or(0);

  Ok(ParsedModel {
    pieces,
    unknown_token_id: resolved_unknown_id,
    model_type,
  })
}

fn parse_piece(data: &[u8]) -> Result<Option<SentencePieceToken>> {
  let mut reader = SentencePieceProtobufReader::new(data);
  let mut token: Option<String> = None;
  let mut score: f32 = 0.0;
  let mut r#type: SentencePiecePieceType = SentencePiecePieceType::Normal;

  while !reader.is_at_end() {
    let key = reader.read_varint()?;
    let field_number = key >> 3;
    let wire_type = key & 0x7;
    match (field_number, wire_type) {
      (1, 2) => {
        let token_data = reader.read_length_delimited()?;
        token = Some(String::from_utf8_lossy(token_data).into_owned());
      }
      (2, 5) => {
        score = f32::from_bits(reader.read_fixed32()?);
      }
      (3, 0) => {
        r#type = SentencePiecePieceType::from_raw(reader.read_varint()?);
      }
      _ => reader.skip_field(wire_type)?,
    }
  }
  Ok(token.map(|token| SentencePieceToken::new(token, score, r#type)))
}

fn parse_trainer_spec_model_type(data: &[u8]) -> Result<Option<SentencePieceModelType>> {
  let mut reader = SentencePieceProtobufReader::new(data);
  while !reader.is_at_end() {
    let key = reader.read_varint()?;
    let field_number = key >> 3;
    let wire_type = key & 0x7;
    if field_number == 3 && wire_type == 0 {
      return Ok(SentencePieceModelType::from_raw(reader.read_varint()?));
    }
    reader.skip_field(wire_type)?;
  }
  Ok(None)
}

// ----------------------------------------------------------------------
// Token lattice (Unigram Viterbi)
// ----------------------------------------------------------------------

/// One candidate node in the Viterbi lattice — covers a contiguous range
/// of the input sentence with a single token id + its model score.
///
/// Mirrors the Swift `TokenLatticeNode` reference type — `prev`/
/// `backtrace_score` are mutated during the Viterbi pass and read back
/// during path reconstruction.
#[derive(Debug, Clone)]
struct TokenLatticeNode {
  token_id: usize,
  /// Index into the lattice's per-character vector (NOT bytes).
  char_start: usize,
  /// Length in characters (NOT bytes).
  char_len: usize,
  score: f32,
  /// Index into the lattice's `nodes` arena pointing at the
  /// best-previous node, or `None` for BOS / no path. Replaces the
  /// Swift `prev: TokenLatticeNode?` reference link — arena-indexed so
  /// the Viterbi state cycle stays acyclic in Rust borrow terms.
  prev: Option<usize>,
  backtrace_score: f32,
}

/// Viterbi lattice over a per-character indexed sentence.
///
/// Faithful Rust port of the Swift `TokenLattice` value-type +
/// `TokenLatticeNode` reference-type pair: nodes live in a flat `nodes`
/// arena, and `begin_nodes` / `end_nodes` hold arena indices per
/// character offset. The pair has the same `insert(start, len, score,
/// token_id)` and `viterbi()` API surface as the reference.
struct TokenLattice {
  /// Character-decomposed sentence — indices into this slice are the
  /// `char_start` / `char_len` axis used throughout the lattice.
  chars: Vec<char>,
  /// BOS/EOS token ids — kept on the struct for parity with the Swift
  /// reference's `bosTokenId` / `eosTokenId` field carriage even though
  /// they're consumed only by `new` (which stamps them onto the
  /// sentinel BOS/EOS nodes).
  #[allow(dead_code)]
  bos_token_id: usize,
  #[allow(dead_code)]
  eos_token_id: usize,

  nodes: Vec<TokenLatticeNode>,
  /// `begin_nodes[i]` holds the arena indices of every node STARTING at
  /// character offset `i`. Always has `chars.len() + 1` slots so the EOS
  /// node at the trailing boundary has a home.
  begin_nodes: Vec<Vec<usize>>,
  /// Symmetric end-side table — `end_nodes[i]` holds the indices of
  /// every node ENDING at offset `i`.
  end_nodes: Vec<Vec<usize>>,
}

impl TokenLattice {
  fn new(sentence: &str, bos_token_id: usize, eos_token_id: usize) -> Self {
    let chars: Vec<char> = sentence.chars().collect();
    let n = chars.len();

    let bos = TokenLatticeNode {
      token_id: bos_token_id,
      char_start: 0,
      char_len: 0,
      score: 0.0,
      prev: None,
      backtrace_score: 0.0,
    };
    let eos = TokenLatticeNode {
      token_id: eos_token_id,
      char_start: n,
      char_len: 0,
      score: 0.0,
      prev: None,
      backtrace_score: 0.0,
    };

    let mut nodes = Vec::with_capacity(n + 2);
    nodes.push(bos);
    nodes.push(eos);

    let mut begin_nodes = vec![Vec::<usize>::new(); n + 1];
    let mut end_nodes = vec![Vec::<usize>::new(); n + 1];
    end_nodes[0].push(0); // BOS at arena index 0
    begin_nodes[n].push(1); // EOS at arena index 1

    Self {
      chars,
      bos_token_id,
      eos_token_id,
      nodes,
      begin_nodes,
      end_nodes,
    }
  }

  fn char_count(&self) -> usize {
    self.chars.len()
  }

  fn insert(&mut self, char_start: usize, char_len: usize, score: f32, token_id: usize) {
    let idx = self.nodes.len();
    self.nodes.push(TokenLatticeNode {
      token_id,
      char_start,
      char_len,
      score,
      prev: None,
      backtrace_score: 0.0,
    });
    self.begin_nodes[char_start].push(idx);
    self.end_nodes[char_start + char_len].push(idx);
  }

  /// Run the Viterbi pass and return the best-scoring path (BOS / EOS
  /// stripped). Returns an empty vec if any character offset has no
  /// begin-node (a degenerate lattice — mirrors the Swift early-return).
  fn viterbi(&mut self) -> Vec<TokenLatticeNode> {
    let count = self.char_count();
    for offset in 0..=count {
      if self.begin_nodes[offset].is_empty() {
        return Vec::new();
      }

      // Snapshot lists so the &mut self body below can mutate self.nodes
      // without aliasing the lattice index vectors.
      let rnode_indices = self.begin_nodes[offset].clone();
      let lnode_indices = self.end_nodes[offset].clone();

      for &rnode_idx in &rnode_indices {
        let rnode_score = self.nodes[rnode_idx].score;

        self.nodes[rnode_idx].prev = None;

        let mut best_score: f32 = 0.0;
        let mut best_lnode_idx: Option<usize> = None;
        for &lnode_idx in &lnode_indices {
          let lnode_backtrace = self.nodes[lnode_idx].backtrace_score;
          let candidate = lnode_backtrace + rnode_score;
          if best_lnode_idx.is_none() || candidate > best_score {
            best_lnode_idx = Some(lnode_idx);
            best_score = candidate;
          }
        }

        if best_lnode_idx.is_some() {
          self.nodes[rnode_idx].prev = best_lnode_idx;
          self.nodes[rnode_idx].backtrace_score = best_score;
        }
      }
    }

    // EOS sits at begin_nodes[count][0] by construction.
    let root_idx = self.begin_nodes[count][0];
    let mut prev = match self.nodes[root_idx].prev {
      Some(i) => i,
      None => return Vec::new(),
    };

    let mut result: Vec<TokenLatticeNode> = Vec::new();
    loop {
      let node = self.nodes[prev].clone();
      let next = node.prev;
      result.push(node);
      match next {
        Some(i) => prev = i,
        None => break,
      }
    }
    result.reverse();
    result
  }

  /// Extract the UTF-8 substring covered by a lattice node — uses the
  /// character offsets stored on the node, not byte offsets.
  fn piece(&self, node: &TokenLatticeNode) -> String {
    let end = node.char_start + node.char_len;
    self.chars[node.char_start..end].iter().collect()
  }
}

// ----------------------------------------------------------------------
// Trie for common-prefix lookup (the vocabulary index)
// ----------------------------------------------------------------------

#[derive(Debug, Default)]
struct TrieNode {
  children: HashMap<char, TrieNode>,
  is_end: bool,
}

#[derive(Debug, Default)]
struct Trie {
  root: TrieNode,
}

impl Trie {
  fn append_all<I, S>(&mut self, tokens: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
    for token in tokens {
      self.insert(token.as_ref());
    }
  }

  fn insert(&mut self, token: &str) {
    let mut node = &mut self.root;
    for ch in token.chars() {
      node = node.children.entry(ch).or_default();
    }
    node.is_end = true;
  }

  /// All vocabulary tokens that prefix the supplied character slice.
  fn common_prefix_search(&self, chars: &[char]) -> Vec<String> {
    let mut results: Vec<String> = Vec::new();
    let mut node = &self.root;
    let mut current = String::new();
    for &ch in chars {
      match node.children.get(&ch) {
        Some(next) => {
          current.push(ch);
          node = next;
          if node.is_end {
            results.push(current.clone());
          }
        }
        None => break,
      }
    }
    results
  }
}

// ----------------------------------------------------------------------
// Tokenizer
// ----------------------------------------------------------------------

/// Self-contained SentencePiece tokenizer — Unigram + BPE, both with
/// byte-fallback.
#[derive(Debug)]
pub struct SentencePieceTokenizer {
  vocab: Vec<SentencePieceToken>,
  unknown_token_id: usize,
  unknown_token_score: f32,
  model_type: SentencePieceModelType,
  tokens_to_ids: HashMap<String, usize>,
  trie: Trie,
  /// Byte-fallback piece map: `byte → token id`, populated from
  /// `<0xHH>` pieces in the vocabulary. Cached on construction (Swift
  /// `lazy var byteMap`).
  byte_map: [Option<usize>; 256],
  /// Atomic BPE pieces — `user_defined`-typed entries, sorted by
  /// descending character length so longest matches win during
  /// [`initial_bpe_symbols`](Self::initial_bpe_symbols). Cached on
  /// construction (Swift `lazy var bpeAtomicPieces`).
  bpe_atomic_pieces: Vec<String>,
}

impl SentencePieceTokenizer {
  fn new(
    vocab: Vec<SentencePieceToken>,
    unknown_token_id: usize,
    model_type: SentencePieceModelType,
  ) -> Self {
    let min_score = vocab
      .iter()
      .map(|t| t.score())
      .fold(f32::INFINITY, f32::min);
    let unknown_token_score = min_score - 10.0;

    let mut tokens_to_ids: HashMap<String, usize> = HashMap::with_capacity(vocab.len());
    for (i, tok) in vocab.iter().enumerate() {
      tokens_to_ids.insert(tok.token().to_owned(), i);
    }

    let mut trie = Trie::default();
    trie.append_all(vocab.iter().map(|t| t.token()));

    let mut byte_map: [Option<usize>; 256] = [None; 256];
    for (i, tok) in vocab.iter().enumerate() {
      let s = tok.token();
      if let Some(byte) = parse_byte_fallback_piece(s) {
        byte_map[byte as usize] = Some(i);
      }
    }

    let mut bpe_atomic_pieces: Vec<String> = vocab
      .iter()
      .filter(|t| t.piece_type() == SentencePiecePieceType::UserDefined)
      .map(|t| t.token().to_owned())
      .collect();
    bpe_atomic_pieces.sort_by_key(|piece| std::cmp::Reverse(piece.chars().count()));

    Self {
      vocab,
      unknown_token_id,
      unknown_token_score,
      model_type,
      tokens_to_ids,
      trie,
      byte_map,
      bpe_atomic_pieces,
    }
  }

  /// Build from raw `.model` protobuf bytes.
  ///
  /// # Errors
  /// [`Error::Backend`] for any malformed-protobuf, truncated-field, or
  /// empty-vocabulary input; [`Error::UnknownEnumValue`] for an
  /// unsupported protobuf wire type.
  /// [`Error::ArithmeticOverflow`] when a length-delimited / fixed32 /
  /// fixed64 field's `checked_add` reader-index advance overflows
  /// (oversized length varint after the reader index has advanced).
  pub fn from_model_bytes(data: &[u8]) -> Result<Self> {
    let parsed = parse_pieces(data)?;
    Ok(Self::new(
      parsed.pieces,
      parsed.unknown_token_id,
      parsed.model_type,
    ))
  }

  /// Load a `.model` protobuf from disk + parse it.
  ///
  /// # Errors
  /// [`Error::FileIo`] carrying the underlying [`std::io::Error`] when the
  /// file fails to read, or any of the protobuf-parse [`Error::Backend`] /
  /// [`Error::ArithmeticOverflow`] errors from [`Self::from_model_bytes`].
  pub fn from_model_file(path: &Path) -> Result<Self> {
    let bytes = fs::read(path).map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "SentencePieceTokenizer: failed to read model file",
        FileOp::Read,
        path.to_path_buf(),
        e,
      ))
    })?;
    Self::from_model_bytes(&bytes)
  }

  /// Build from a parsed `tokenizer.json` JSON value (HF Tokenizers
  /// SPM-style format — Llama / T5 / Gemma).
  ///
  /// Looks for `tokenizer_json["model"]["vocab"]` (a `[[piece, score],
  /// ...]` array) and `tokenizer_json["model"]["unk_id"]` (an integer).
  /// Falls back to Unigram unless `model.type` is `"BPE"`.
  ///
  /// ## PieceType inference
  ///
  /// HF `tokenizer.json` does not store SentencePiece's `PieceType` enum
  /// (Normal/Unknown/Control/UserDefined/Byte/Unused) directly; this
  /// loader reconstructs it from three sources so consumers that depend
  /// on the type (byte-fallback encode, control-token decode-skip — see
  /// `SentencePiecePieceType` doc) get behavior parity with the protobuf
  /// `.model` path:
  ///
  /// 1. `model.unk_id` → that piece's type is marked
  ///    [`SentencePiecePieceType::Unknown`].
  /// 2. Pieces whose `content` matches the byte-fallback convention
  ///    `<0xNN>` (where `NN` is two hex digits) are marked
  ///    [`SentencePiecePieceType::Byte`]. These exist when the HF model
  ///    was trained with `byte_fallback=true`; decoders that need to
  ///    surface raw bytes for unencodable UTF-8 sequences rely on this.
  /// 3. Tokens listed in the sibling `tokenizer_json["added_tokens"]`
  ///    array (HF's special / added-token surface) are marked
  ///    [`SentencePiecePieceType::Control`] when `special: true` and
  ///    [`SentencePiecePieceType::UserDefined`] when `special: false`.
  ///    Matching is by exact `content` string against the vocab piece.
  ///
  /// All other pieces stay [`SentencePiecePieceType::Normal`] (the
  /// majority case). The precedence ordering is `Unknown > Byte > added
  /// (Control/UserDefined) > Normal`: e.g. if a model declared
  /// `unk_id = K` and `<0xK_hex>` happens to be the same vocab index,
  /// the Unknown marking wins (matches the protobuf semantics).
  ///
  /// # Errors
  /// [`Error::MissingField`] when `model`, `model.unk_id`, or
  /// `model.vocab` is absent. [`Error::Backend`] for malformed
  /// `model.vocab` entries (non-array entry, wrong arity, non-string
  /// token, or non-numeric score).
  ///
  /// Available when the `tokenizer-config` feature is enabled (which
  /// any `audio` build pulls in transitively via `lm`).
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn from_tokenizer_json(tokenizer_json: &JsonValue) -> Result<Self> {
    let model =
      tokenizer_json
        .get("model")
        .ok_or(Error::MissingField(MissingFieldPayload::new(
          "SentencePieceTokenizer",
          "model",
        )))?;
    let unk_id = model
      .get("unk_id")
      .and_then(|v| v.as_u64())
      .ok_or(Error::MissingField(MissingFieldPayload::new(
        "SentencePieceTokenizer",
        "model.unk_id",
      )))? as usize;
    let vocab_list = model
      .get("vocab")
      .and_then(|v| v.as_array())
      .ok_or(Error::MissingField(MissingFieldPayload::new(
        "SentencePieceTokenizer",
        "model.vocab",
      )))?;

    // Pass 1: assemble all pieces with their tokens + scores.
    // Initial piece_type derivation per the doc precedence:
    //   - byte-fallback `<0xNN>` → Byte
    //   - everything else → Normal (will be promoted to Control/UserDefined
    //     by the added_tokens pass below if matched, or to Unknown by the
    //     unk_id pass last so Unknown wins).
    let mut pieces: Vec<SentencePieceToken> = Vec::with_capacity(vocab_list.len());
    for entry in vocab_list {
      let arr = entry.as_array().ok_or_else(|| {
        Error::Backend("SentencePieceTokenizer: `model.vocab` entry not an array".into())
      })?;
      if arr.len() != 2 {
        return Err(Error::Backend(
          "SentencePieceTokenizer: `model.vocab` entry must be a [token, score] pair".into(),
        ));
      }
      let token = arr[0].as_str().ok_or_else(|| {
        Error::Backend("SentencePieceTokenizer: `model.vocab` entry[0] not a string".into())
      })?;
      let score = arr[1].as_f64().ok_or_else(|| {
        Error::Backend("SentencePieceTokenizer: `model.vocab` entry[1] not a number".into())
      })? as f32;
      let initial_type = if is_byte_fallback_piece(token) {
        SentencePiecePieceType::Byte
      } else {
        SentencePiecePieceType::Normal
      };
      pieces.push(SentencePieceToken::new(
        token.to_string(),
        score,
        initial_type,
      ));
    }

    // Pass 2: promote pieces named in `added_tokens` to Control/UserDefined
    // (does NOT overwrite Byte; an explicit byte-fallback token in
    // added_tokens stays Byte since that's the more specific semantic).
    if let Some(added) = tokenizer_json
      .get("added_tokens")
      .and_then(|v| v.as_array())
    {
      for at in added {
        let Some(content) = at.get("content").and_then(|v| v.as_str()) else {
          continue;
        };
        let special = at.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
        let target_type = if special {
          SentencePiecePieceType::Control
        } else {
          SentencePiecePieceType::UserDefined
        };
        for p in &mut pieces {
          if p.token() == content && p.piece_type() != SentencePiecePieceType::Byte {
            *p = SentencePieceToken::new(p.token().to_string(), p.score(), target_type);
          }
        }
      }
    }

    // Pass 3: the unk_id piece is Unknown — this wins over any prior
    // Normal / Control / UserDefined / Byte marking (matches the protobuf
    // path's precedence; `<unk>` is never anything but Unknown).
    if let Some(unk_piece) = pieces.get_mut(unk_id) {
      *unk_piece = SentencePieceToken::new(
        unk_piece.token().to_string(),
        unk_piece.score(),
        SentencePiecePieceType::Unknown,
      );
    }

    let model_type = match model.get("type").and_then(|v| v.as_str()) {
      Some(t) if t.eq_ignore_ascii_case("BPE") => SentencePieceModelType::Bpe,
      _ => SentencePieceModelType::Unigram,
    };

    Ok(Self::new(pieces, unk_id, model_type))
  }

  /// Build from raw `tokenizer.json` bytes.
  ///
  /// # Errors
  /// [`Error::Parse`] for any JSON-parse failure, plus the same
  /// [`Error::MissingField`] / [`Error::Backend`] errors propagated
  /// from [`Self::from_tokenizer_json`].
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn from_tokenizer_json_bytes(data: &[u8]) -> Result<Self> {
    let json: JsonValue = serde_json::from_slice(data).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "SentencePieceTokenizer::from_tokenizer_json_bytes",
        "tokenizer.json",
        e,
      ))
    })?;
    Self::from_tokenizer_json(&json)
  }

  /// Vocabulary size.
  pub fn vocab_size(&self) -> usize {
    self.vocab.len()
  }

  /// The configured unknown-token id (`<unk>`).
  pub fn unknown_token_id(&self) -> usize {
    self.unknown_token_id
  }

  /// The configured model type (Unigram or BPE).
  pub fn model_type(&self) -> SentencePieceModelType {
    self.model_type
  }

  /// Read-only access to a single piece by id.
  pub fn piece(&self, id: usize) -> Option<&SentencePieceToken> {
    self.vocab.get(id)
  }

  /// Encode `text` to a token-id sequence, mapping any out-of-vocabulary
  /// characters to per-byte `<0xHH>` pieces (or the unknown-token id
  /// when even the byte piece is absent).
  ///
  /// Routes to Viterbi (Unigram) or greedy merges (BPE) based on the
  /// model type stored at construction. Always applies the SentencePiece
  /// metaspace preprocessing first (`' '` → `'▁'`, prefix `'▁'`).
  pub fn encode_with_byte_fallback(&self, text: &str) -> Vec<usize> {
    if self.model_type == SentencePieceModelType::Bpe {
      return self.encode_bpe_with_byte_fallback(text);
    }
    self.encode_unigram_with_byte_fallback(text)
  }

  fn encode_unigram_with_byte_fallback(&self, text: &str) -> Vec<usize> {
    let pre = apply_metaspace(text);
    let mut lattice = TokenLattice::new(&pre, self.unknown_token_id, self.unknown_token_id);
    let chars: Vec<char> = pre.chars().collect();

    let mut begin_pos = 0;
    while begin_pos < chars.len() {
      let mblen = 1;
      let mut has_single_node = false;

      for token in self.trie.common_prefix_search(&chars[begin_pos..]) {
        let Some(&token_id) = self.tokens_to_ids.get(&token) else {
          continue;
        };
        let token_char_count = token.chars().count();
        let token_score = self.vocab[token_id].score();
        lattice.insert(begin_pos, token_char_count, token_score, token_id);
        if !has_single_node && token_char_count == mblen {
          has_single_node = true;
        }
      }

      if !has_single_node {
        lattice.insert(
          begin_pos,
          mblen,
          self.unknown_token_score,
          self.unknown_token_id,
        );
      }
      begin_pos += mblen;
    }

    let path = lattice.viterbi();
    let mut ids: Vec<usize> = Vec::with_capacity(path.len());
    for node in &path {
      if node.token_id == self.unknown_token_id {
        let piece = lattice.piece(node);
        for &b in piece.as_bytes() {
          ids.push(self.byte_map[b as usize].unwrap_or(self.unknown_token_id));
        }
      } else {
        ids.push(node.token_id);
      }
    }
    ids
  }

  fn encode_bpe_with_byte_fallback(&self, text: &str) -> Vec<usize> {
    let pre = apply_metaspace(text);
    let mut symbols = self.initial_bpe_symbols(&pre);

    while symbols.len() > 1 {
      let mut best_index: Option<usize> = None;
      let mut best_piece = String::new();
      let mut best_score = f32::NEG_INFINITY;

      for index in 0..symbols.len() - 1 {
        let mut candidate = String::with_capacity(symbols[index].len() + symbols[index + 1].len());
        candidate.push_str(&symbols[index]);
        candidate.push_str(&symbols[index + 1]);
        let Some(&token_id) = self.tokens_to_ids.get(&candidate) else {
          continue;
        };
        let tok = &self.vocab[token_id];
        if !matches!(
          tok.piece_type(),
          SentencePiecePieceType::Normal | SentencePiecePieceType::UserDefined
        ) {
          continue;
        }
        if best_index.is_none() || tok.score() > best_score {
          best_index = Some(index);
          best_piece = candidate;
          best_score = tok.score();
        }
      }

      let Some(index) = best_index else { break };
      symbols.splice(index..=index + 1, std::iter::once(best_piece));
    }

    let mut ids: Vec<usize> = Vec::new();
    for symbol in &symbols {
      if let Some(&token_id) = self.tokens_to_ids.get(symbol) {
        ids.push(token_id);
      } else {
        for &b in symbol.as_bytes() {
          ids.push(self.byte_map[b as usize].unwrap_or(self.unknown_token_id));
        }
      }
    }
    ids
  }

  fn initial_bpe_symbols(&self, text: &str) -> Vec<String> {
    let mut symbols: Vec<String> = Vec::new();
    let mut tail = text;

    while !tail.is_empty() {
      if let Some(atomic) = self
        .bpe_atomic_pieces
        .iter()
        .find(|piece| tail.starts_with(piece.as_str()))
      {
        symbols.push(atomic.clone());
        tail = &tail[atomic.len()..];
      } else {
        let mut iter = tail.char_indices();
        let _ = iter.next();
        let next_byte = iter.next().map(|(i, _)| i).unwrap_or(tail.len());
        symbols.push(tail[..next_byte].to_string());
        tail = &tail[next_byte..];
      }
    }

    symbols
  }

  /// Decode a token-id slice back to text, reassembling any
  /// `<0xHH>` byte-fallback pieces and stripping the metaspace marker.
  ///
  /// Skips Control / Unused pieces, mirroring the Swift reference. The
  /// final whitespace trim leaves the caller a "clean" UTF-8 string.
  pub fn decode(&self, ids: &[usize]) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    let mut pieces: Vec<String> = Vec::new();
    for &id in ids {
      let Some(token) = self.vocab.get(id) else {
        continue;
      };
      if matches!(
        token.piece_type(),
        SentencePiecePieceType::Control | SentencePiecePieceType::Unused
      ) {
        continue;
      }
      let tok = token.token();
      if let Some(byte) = parse_byte_fallback_piece(tok) {
        bytes.push(byte);
        continue;
      }
      if !bytes.is_empty() {
        if let Ok(s) = std::str::from_utf8(&bytes) {
          pieces.push(s.to_string());
        }
        bytes.clear();
      }
      pieces.push(tok.to_owned());
    }
    if !bytes.is_empty()
      && let Ok(s) = std::str::from_utf8(&bytes)
    {
      pieces.push(s.to_string());
    }
    let joined: String = pieces.concat();
    let restored = joined.replace('\u{2581}', " ");
    restored.trim().to_string()
  }
}

/// Parse a `<0xHH>` byte-fallback piece (`<0x` + 2 hex digits + `>`)
/// into its byte value. Returns `None` for any other format.
fn parse_byte_fallback_piece(piece: &str) -> Option<u8> {
  // Cheap pre-check (ASCII only, fixed length); strip and parse the hex.
  let bytes = piece.as_bytes();
  if bytes.len() != 6 || !bytes.starts_with(b"<0x") || bytes[5] != b'>' {
    return None;
  }
  let hex = &piece[3..5];
  u8::from_str_radix(hex, 16).ok()
}

/// Boolean wrapper around [`parse_byte_fallback_piece`] for use in
/// [`SentencePieceTokenizer::from_tokenizer_json`]'s piece-type inference.
#[cfg(feature = "tokenizer-config")]
fn is_byte_fallback_piece(piece: &str) -> bool {
  parse_byte_fallback_piece(piece).is_some()
}

/// SentencePiece metaspace preprocessing — `' '` → U+2581 + prefix the
/// whole string with U+2581. Mirrors `applyMetaspace` in the Swift ref.
fn apply_metaspace(text: &str) -> String {
  let replaced = text.replace(' ', "\u{2581}");
  let mut out = String::with_capacity(replaced.len() + 3);
  out.push('\u{2581}');
  out.push_str(&replaced);
  out
}

#[cfg(test)]
// Wire-tag arithmetic uses `| wire_type` even for wire type `0` (varint), so
// `| 0` is intentional documentation of the protobuf wire format, not dead
// arithmetic. Pre-allocate-then-push patterns also reflect the wire layout
// step-by-step and would be obscured by `vec![...]`.
#[allow(clippy::identity_op, clippy::vec_init_then_push)]
mod tests {
  use super::*;

  // ----------------------------------------------------------------
  // Protobuf builders for the test fixtures.
  //
  // Wire format (one tag byte per field):
  //   tag = (field_number << 3) | wire_type
  //   wire 0 = varint, wire 2 = length-delimited, wire 5 = fixed32
  // ----------------------------------------------------------------

  fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value > 0x7f {
      out.push((value & 0x7f) as u8 | 0x80);
      value >>= 7;
    }
    out.push(value as u8);
  }

  fn build_piece(token: &str, score: f32, piece_type: u8) -> Vec<u8> {
    let mut piece = Vec::new();
    // field 1, wire 2 (token string)
    piece.push((1 << 3) | 2);
    write_varint(&mut piece, token.len() as u64);
    piece.extend_from_slice(token.as_bytes());
    // field 2, wire 5 (score f32 little-endian)
    piece.push((2 << 3) | 5);
    piece.extend_from_slice(&score.to_bits().to_le_bytes());
    // field 3, wire 0 (type)
    piece.push((3 << 3) | 0);
    write_varint(&mut piece, u64::from(piece_type));
    piece
  }

  fn build_model_with_pieces(pieces: &[(&str, f32, u8)], model_type: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for (token, score, piece_type) in pieces {
      let piece_bytes = build_piece(token, *score, *piece_type);
      // field 1, wire 2 (SentencePiece pieces)
      out.push((1 << 3) | 2);
      write_varint(&mut out, piece_bytes.len() as u64);
      out.extend_from_slice(&piece_bytes);
    }
    // trainer_spec — field 2, wire 2 — containing field 3, wire 0 = model_type
    let mut trainer = Vec::new();
    trainer.push((3 << 3) | 0);
    write_varint(&mut trainer, model_type);
    out.push((2 << 3) | 2);
    write_varint(&mut out, trainer.len() as u64);
    out.extend_from_slice(&trainer);
    out
  }

  // ----------------------------------------------------------------
  // Tests
  // ----------------------------------------------------------------

  #[test]
  fn parse_minimal_unigram_protobuf_yields_vocab_and_model_type() {
    let data = build_model_with_pieces(
      &[
        ("<unk>", 0.0, SentencePiecePieceType::Unknown.as_raw() as u8),
        (
          "\u{2581}hello",
          -1.0,
          SentencePiecePieceType::Normal.as_raw() as u8,
        ),
        (
          "\u{2581}world",
          -2.0,
          SentencePiecePieceType::Normal.as_raw() as u8,
        ),
      ],
      1, // unigram
    );
    let tok = SentencePieceTokenizer::from_model_bytes(&data).unwrap();
    assert_eq!(tok.vocab_size(), 3);
    assert_eq!(tok.unknown_token_id(), 0);
    assert_eq!(tok.model_type(), SentencePieceModelType::Unigram);
    assert_eq!(tok.piece(1).map(|p| p.token()), Some("\u{2581}hello"));
  }

  #[test]
  fn malformed_protobuf_errors_with_actionable_message() {
    // Truncated length-delimited piece — first byte declares a sub-message
    // whose declared length exceeds the remaining buffer.
    let mut bad = Vec::new();
    bad.push((1 << 3) | 2);
    bad.push(50); // declared length = 50, but no body follows
    let err = SentencePieceTokenizer::from_model_bytes(&bad).unwrap_err();
    let Error::Backend(message) = err else {
      panic!("expected Error::Backend, got {err:?}");
    };
    assert!(message.contains("SentencePiece"), "message: {message}");
    assert!(message.contains("truncated"), "message: {message}");
  }

  #[test]
  fn empty_vocab_protobuf_is_rejected() {
    // Only a trainer_spec field — no pieces. Must error.
    let mut data = Vec::new();
    let mut trainer = Vec::new();
    trainer.push((3 << 3) | 0);
    write_varint(&mut trainer, 1);
    data.push((2 << 3) | 2);
    write_varint(&mut data, trainer.len() as u64);
    data.extend_from_slice(&trainer);
    let err = SentencePieceTokenizer::from_model_bytes(&data).unwrap_err();
    let Error::Backend(message) = err else {
      panic!("expected Error::Backend, got {err:?}");
    };
    assert!(message.contains("vocabulary"));
  }

  /// Build a small fixture matching the toy vocab used by the
  /// `encode/decode` tests. The vocab is
  ///  0: `<unk>`         (Unknown)
  ///  1: `▁hello`        (Normal, score -1.0)
  ///  2: `▁world`        (Normal, score -1.0)
  ///  3: `▁`             (Normal, score -3.0)
  ///  4: `<0x21>`        (Byte — `!` byte 0x21)
  ///  5: `<0x3F>`        (Byte — `?` byte 0x3F)
  fn toy_tokenizer() -> SentencePieceTokenizer {
    let data = build_model_with_pieces(
      &[
        ("<unk>", 0.0, SentencePiecePieceType::Unknown.as_raw() as u8),
        (
          "\u{2581}hello",
          -1.0,
          SentencePiecePieceType::Normal.as_raw() as u8,
        ),
        (
          "\u{2581}world",
          -1.0,
          SentencePiecePieceType::Normal.as_raw() as u8,
        ),
        (
          "\u{2581}",
          -3.0,
          SentencePiecePieceType::Normal.as_raw() as u8,
        ),
        ("<0x21>", -5.0, SentencePiecePieceType::Byte.as_raw() as u8),
        ("<0x3F>", -5.0, SentencePiecePieceType::Byte.as_raw() as u8),
      ],
      1,
    );
    SentencePieceTokenizer::from_model_bytes(&data).unwrap()
  }

  #[test]
  fn encode_unigram_known_input_yields_expected_piece_sequence() {
    let tok = toy_tokenizer();
    // Expect: "hello world" → [▁hello, ▁world]
    let ids = tok.encode_with_byte_fallback("hello world");
    assert_eq!(ids, vec![1, 2], "ids={:?}", ids);
  }

  #[test]
  fn encode_unigram_byte_fallback_for_out_of_vocab_chars() {
    let tok = toy_tokenizer();
    // "?" is not in vocab as a piece — only as a byte-fallback piece.
    let ids = tok.encode_with_byte_fallback("hello?");
    // ids should contain the byte-fallback id (5) for `?` somewhere.
    assert!(
      ids.contains(&5),
      "byte-fallback for `?` (id=5) missing in ids={ids:?}"
    );
  }

  #[test]
  fn encode_then_decode_is_lossless_round_trip_on_known_input() {
    let tok = toy_tokenizer();
    let original = "hello world";
    let ids = tok.encode_with_byte_fallback(original);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, original, "round-trip mismatch: ids={ids:?}");
  }

  #[test]
  fn decode_skips_control_and_unused_pieces() {
    let data = build_model_with_pieces(
      &[
        ("<unk>", 0.0, SentencePiecePieceType::Unknown.as_raw() as u8),
        ("<s>", 0.0, SentencePiecePieceType::Control.as_raw() as u8),
        ("<pad>", 0.0, SentencePiecePieceType::Unused.as_raw() as u8),
        (
          "\u{2581}hi",
          -1.0,
          SentencePiecePieceType::Normal.as_raw() as u8,
        ),
      ],
      1,
    );
    let tok = SentencePieceTokenizer::from_model_bytes(&data).unwrap();
    let decoded = tok.decode(&[1, 2, 3]); // control, unused, ▁hi
    assert_eq!(decoded, "hi");
  }

  #[test]
  fn decode_reassembles_byte_fallback_pieces_into_valid_utf8() {
    // Build a vocab covering the 3 bytes of "é" (U+00E9 = 0xC3 0xA9).
    // Whether the encode path triggers byte-fallback depends on the
    // Viterbi tie-break; we exercise decode directly to keep the test
    // deterministic and assert byte-fallback REASSEMBLY (the
    // round-trip-critical half of byte-fallback).
    let data = build_model_with_pieces(
      &[
        ("<unk>", 0.0, SentencePiecePieceType::Unknown.as_raw() as u8),
        ("<0xC3>", -5.0, SentencePiecePieceType::Byte.as_raw() as u8),
        ("<0xA9>", -5.0, SentencePiecePieceType::Byte.as_raw() as u8),
        (
          "\u{2581}",
          -1.0,
          SentencePiecePieceType::Normal.as_raw() as u8,
        ),
      ],
      1,
    );
    let tok = SentencePieceTokenizer::from_model_bytes(&data).unwrap();
    let decoded = tok.decode(&[3, 1, 2]); // ▁, 0xC3, 0xA9
    assert_eq!(decoded, "é");
  }

  #[test]
  fn from_model_file_propagates_io_error_for_missing_path() {
    let err =
      SentencePieceTokenizer::from_model_file(Path::new("/nonexistent/path.model")).unwrap_err();
    match err {
      Error::FileIo(p) => {
        assert_eq!(p.op(), FileOp::Read);
        assert_eq!(p.path(), Path::new("/nonexistent/path.model"));
        assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
        assert!(p.context().contains("failed to read"));
      }
      other => panic!("expected Error::FileIo, got {other:?}"),
    }
  }

  #[cfg(feature = "tokenizer-config")]
  #[test]
  fn from_tokenizer_json_parses_unigram_vocab() {
    let json: serde_json::Value = serde_json::json!({
      "model": {
        "type": "Unigram",
        "unk_id": 0,
        "vocab": [
          ["<unk>", 0.0],
          ["\u{2581}hello", -1.0],
          ["\u{2581}world", -1.0],
        ],
      }
    });
    let tok = SentencePieceTokenizer::from_tokenizer_json(&json).unwrap();
    assert_eq!(tok.vocab_size(), 3);
    assert_eq!(tok.unknown_token_id(), 0);
    let ids = tok.encode_with_byte_fallback("hello world");
    assert_eq!(ids, vec![1, 2]);
  }

  #[cfg(feature = "tokenizer-config")]
  #[test]
  fn from_tokenizer_json_bytes_rejects_invalid_json() {
    let err = SentencePieceTokenizer::from_tokenizer_json_bytes(b"not json").unwrap_err();
    let Error::Parse(p) = err else {
      panic!("expected Error::Parse, got {err:?}");
    };
    assert_eq!(p.input_kind(), "tokenizer.json");
    assert_eq!(
      p.context(),
      "SentencePieceTokenizer::from_tokenizer_json_bytes"
    );
  }

  #[cfg(feature = "tokenizer-config")]
  #[test]
  fn from_tokenizer_json_errors_on_missing_model_field() {
    let json: serde_json::Value = serde_json::json!({"other": 1});
    let err = SentencePieceTokenizer::from_tokenizer_json(&json).unwrap_err();
    match err {
      Error::MissingField(p) => {
        assert_eq!(p.type_name(), "SentencePieceTokenizer");
        assert_eq!(p.field(), "model");
      }
      other => panic!("expected Error::MissingField, got {other:?}"),
    }
  }

  /// PieceType inference from HF tokenizer.json (#258 MODERATE): the
  /// 4 sources (unk_id → Unknown, `<0xNN>` → Byte, `added_tokens.special` →
  /// Control, `added_tokens` non-special → UserDefined, all others →
  /// Normal) must materialize correctly. This covers all 5 SentencePiece
  /// PieceType variants the protobuf path preserves.
  #[cfg(feature = "tokenizer-config")]
  #[test]
  fn from_tokenizer_json_infers_piece_types_from_unk_byte_and_added_tokens() {
    let json: serde_json::Value = serde_json::json!({
      "model": {
        "type": "Unigram",
        "unk_id": 0,
        "vocab": [
          ["<unk>", 0.0],         // id 0 — promoted to Unknown via unk_id
          ["\u{2581}hello", -1.0],// id 1 — Normal
          ["<0x41>", -2.0],       // id 2 — byte-fallback → Byte ('A')
          ["<s>", -3.0],          // id 3 — added_tokens special → Control
          ["<custom>", -4.0],     // id 4 — added_tokens non-special → UserDefined
          ["\u{2581}world", -5.0],// id 5 — Normal
        ],
      },
      "added_tokens": [
        { "id": 3, "content": "<s>",      "special": true },
        { "id": 4, "content": "<custom>", "special": false },
      ],
    });
    let tok = SentencePieceTokenizer::from_tokenizer_json(&json).unwrap();
    assert_eq!(tok.vocab_size(), 6);
    assert_eq!(tok.unknown_token_id(), 0);
    assert_eq!(
      tok.piece(0).unwrap().piece_type(),
      SentencePiecePieceType::Unknown
    );
    assert_eq!(
      tok.piece(1).unwrap().piece_type(),
      SentencePiecePieceType::Normal
    );
    assert_eq!(
      tok.piece(2).unwrap().piece_type(),
      SentencePiecePieceType::Byte
    );
    assert_eq!(
      tok.piece(3).unwrap().piece_type(),
      SentencePiecePieceType::Control
    );
    assert_eq!(
      tok.piece(4).unwrap().piece_type(),
      SentencePiecePieceType::UserDefined
    );
    assert_eq!(
      tok.piece(5).unwrap().piece_type(),
      SentencePiecePieceType::Normal
    );
  }

  /// Precedence: when a byte-fallback token also appears in added_tokens,
  /// `Byte` wins (it's the more specific decode contract). When the same
  /// vocab id is both `unk_id` and looks like `<0xNN>`, Unknown wins
  /// (matches the protobuf semantics: `<unk>` is never anything but Unknown).
  #[cfg(feature = "tokenizer-config")]
  #[test]
  fn from_tokenizer_json_piece_type_precedence() {
    let json: serde_json::Value = serde_json::json!({
      "model": {
        "type": "Unigram",
        "unk_id": 1,
        "vocab": [
          ["<0xFF>", 0.0],        // id 0 — Byte; added_tokens entry must NOT promote.
          ["<0x00>", 0.0],        // id 1 — would be Byte, but unk_id wins → Unknown.
          ["\u{2581}x", 0.0],
        ],
      },
      "added_tokens": [
        { "id": 0, "content": "<0xFF>", "special": true },
      ],
    });
    let tok = SentencePieceTokenizer::from_tokenizer_json(&json).unwrap();
    assert_eq!(
      tok.piece(0).unwrap().piece_type(),
      SentencePiecePieceType::Byte
    );
    assert_eq!(
      tok.piece(1).unwrap().piece_type(),
      SentencePiecePieceType::Unknown
    );
  }
}
