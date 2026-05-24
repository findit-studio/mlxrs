//! Integration tests for `tokenizer::sentencepiece::SentencePieceTokenizer`
//! (protobuf reader + Viterbi + byte-fallback). Complements the in-crate
//! unit tests by exercising the public API surface a downstream consumer
//! sees through the integration boundary.
//!
//! Wire-tag arithmetic uses `| wire_type` even for wire type `0`
//! (varint), so `| 0` is intentional documentation of the protobuf wire
//! format, not dead arithmetic.
#![cfg(feature = "audio")]
#![allow(clippy::identity_op, clippy::vec_init_then_push)]

use mlxrs::tokenizer::sentencepiece::{
  SentencePieceModelType, SentencePiecePieceType, SentencePieceTokenizer,
};

// --- Tiny protobuf builders (mirror the unit-test helpers) -------------

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
  while value > 0x7f {
    out.push((value & 0x7f) as u8 | 0x80);
    value >>= 7;
  }
  out.push(value as u8);
}

fn build_piece(token: &str, score: f32, piece_type: u8) -> Vec<u8> {
  let mut piece = Vec::new();
  piece.push((1 << 3) | 2);
  write_varint(&mut piece, token.len() as u64);
  piece.extend_from_slice(token.as_bytes());
  piece.push((2 << 3) | 5);
  piece.extend_from_slice(&score.to_bits().to_le_bytes());
  piece.push((3 << 3) | 0);
  write_varint(&mut piece, u64::from(piece_type));
  piece
}

fn build_model_with_pieces(pieces: &[(&str, f32, u8)], model_type: u64) -> Vec<u8> {
  let mut out = Vec::new();
  for (token, score, piece_type) in pieces {
    let body = build_piece(token, *score, *piece_type);
    out.push((1 << 3) | 2);
    write_varint(&mut out, body.len() as u64);
    out.extend_from_slice(&body);
  }
  // trainer_spec (field 2, wire 2) containing the model_type (field 3, wire 0)
  let mut trainer = Vec::new();
  trainer.push((3 << 3) | 0);
  write_varint(&mut trainer, model_type);
  out.push((2 << 3) | 2);
  write_varint(&mut out, trainer.len() as u64);
  out.extend_from_slice(&trainer);
  out
}

// --- Tests --------------------------------------------------------------

#[test]
fn from_model_bytes_parses_unigram_vocab() {
  let data = build_model_with_pieces(
    &[
      ("<unk>", 0.0, SentencePiecePieceType::Unknown as u8),
      ("\u{2581}hello", -1.0, SentencePiecePieceType::Normal as u8),
      ("\u{2581}world", -1.0, SentencePiecePieceType::Normal as u8),
      ("\u{2581}", -3.0, SentencePiecePieceType::Normal as u8),
    ],
    1, // unigram
  );
  let tok = SentencePieceTokenizer::from_model_bytes(&data).unwrap();
  assert_eq!(tok.vocab_size(), 4);
  assert_eq!(tok.model_type(), SentencePieceModelType::Unigram);
  assert_eq!(tok.unknown_token_id(), 0);
}

#[test]
fn unigram_round_trip_recovers_original() {
  let data = build_model_with_pieces(
    &[
      ("<unk>", 0.0, SentencePiecePieceType::Unknown as u8),
      ("\u{2581}hello", -1.0, SentencePiecePieceType::Normal as u8),
      ("\u{2581}world", -1.0, SentencePiecePieceType::Normal as u8),
      ("\u{2581}", -3.0, SentencePiecePieceType::Normal as u8),
    ],
    1,
  );
  let tok = SentencePieceTokenizer::from_model_bytes(&data).unwrap();
  let ids = tok.encode_with_byte_fallback("hello world");
  assert_eq!(ids, vec![1, 2]);
  let decoded = tok.decode(&ids);
  assert_eq!(decoded, "hello world");
}

#[test]
fn unigram_byte_fallback_for_oov_character() {
  let data = build_model_with_pieces(
    &[
      ("<unk>", 0.0, SentencePiecePieceType::Unknown as u8),
      ("\u{2581}hello", -1.0, SentencePiecePieceType::Normal as u8),
      ("\u{2581}", -3.0, SentencePiecePieceType::Normal as u8),
      // Byte-fallback for `!` (0x21)
      ("<0x21>", -5.0, SentencePiecePieceType::Byte as u8),
    ],
    1,
  );
  let tok = SentencePieceTokenizer::from_model_bytes(&data).unwrap();
  let ids = tok.encode_with_byte_fallback("hello!");
  // `!` is OOV as a piece — must appear via byte-fallback id = 3.
  assert!(
    ids.contains(&3),
    "expected byte-fallback piece id 3 in {ids:?}"
  );
  // Decode round-trips the byte-fallback piece back into the literal `!`.
  let decoded = tok.decode(&ids);
  assert!(decoded.ends_with('!'), "decoded='{decoded}'");
}

#[test]
fn unrecognized_byte_falls_back_to_unknown_id_not_panic() {
  // No byte-fallback pieces in the vocab, so an OOV char like `?`
  // must round-trip through `unknown_token_id` instead of panicking.
  let data = build_model_with_pieces(
    &[
      ("<unk>", 0.0, SentencePiecePieceType::Unknown as u8),
      ("\u{2581}hi", -1.0, SentencePiecePieceType::Normal as u8),
      ("\u{2581}", -3.0, SentencePiecePieceType::Normal as u8),
    ],
    1,
  );
  let tok = SentencePieceTokenizer::from_model_bytes(&data).unwrap();
  let ids = tok.encode_with_byte_fallback("hi?");
  // `?` → `<unk>` (id 0) since neither `?` nor `<0x3F>` is in the vocab.
  assert!(ids.contains(&0), "expected unknown id 0 in {ids:?}");
}

#[test]
fn bpe_greedy_merge_picks_highest_scoring_pair() {
  // BPE model: "ab" merges to "abc" via higher score than "bc".
  let data = build_model_with_pieces(
    &[
      ("<unk>", 0.0, SentencePiecePieceType::Unknown as u8),
      ("\u{2581}", -3.0, SentencePiecePieceType::Normal as u8),
      ("\u{2581}a", -2.0, SentencePiecePieceType::Normal as u8),
      ("\u{2581}ab", -1.0, SentencePiecePieceType::Normal as u8),
      ("b", -2.0, SentencePiecePieceType::Normal as u8),
      ("c", -2.0, SentencePiecePieceType::Normal as u8),
      ("\u{2581}abc", -0.5, SentencePiecePieceType::Normal as u8),
    ],
    2, // bpe
  );
  let tok = SentencePieceTokenizer::from_model_bytes(&data).unwrap();
  assert_eq!(tok.model_type(), SentencePieceModelType::Bpe);
  let ids = tok.encode_with_byte_fallback("abc");
  // The greedy BPE loop should converge to the single "▁abc" piece (id 6)
  // because it has the highest score among the candidate merges.
  assert_eq!(ids, vec![6], "ids={ids:?}");
  let decoded = tok.decode(&ids);
  assert_eq!(decoded, "abc");
}

#[test]
fn malformed_protobuf_surfaces_actionable_error_message() {
  // Length-delimited field declaring 50 bytes of payload but providing none.
  let bad = vec![(1 << 3) | 2, 50];
  let err = SentencePieceTokenizer::from_model_bytes(&bad).unwrap_err();
  let s = err.to_string();
  assert!(s.contains("SentencePiece"), "{s}");
  assert!(s.contains("truncated"), "{s}");
}
