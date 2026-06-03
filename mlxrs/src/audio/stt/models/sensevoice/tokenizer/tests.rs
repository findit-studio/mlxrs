//! Detokenizer oracles for SenseVoice-Small: the SentencePiece decode of a
//! known id sequence, the `tokens.json` piece-list join (with the `▁` metaspace
//! strip + trim), and the degenerate id-join fallback.
//!
//! The SentencePiece fixture is built from a hand-written `ModelProto` protobuf
//! (the same minimal wire format the shared `SentencePieceTokenizer` parses), so
//! the expected decode is derived from the fixture vocabulary independently of
//! the tokenizer implementation. The `tokens.json` + id-join expectations are
//! closed-form string arithmetic.

use super::*;
use crate::tokenizer::sentencepiece::{SentencePiecePieceType, SentencePieceTokenizer};

// ───────────────────────── SentencePiece protobuf fixture ─────────────────────────
//
// Minimal `ModelProto` wire format (one tag byte per field):
//   tag = (field_number << 3) | wire_type
//   wire 0 = varint, wire 2 = length-delimited, wire 5 = fixed32
// field 1 = repeated SentencePiece pieces; piece sub-message fields:
//   1 (wire 2) = token string, 2 (wire 5) = score f32 LE, 3 (wire 0) = type.
// field 2 = trainer_spec; field 3 (wire 0) = model_type (1 = unigram).

#[allow(clippy::identity_op, clippy::vec_init_then_push)]
fn write_varint(out: &mut Vec<u8>, mut value: u64) {
  while value > 0x7f {
    out.push((value & 0x7f) as u8 | 0x80);
    value >>= 7;
  }
  out.push(value as u8);
}

#[allow(clippy::identity_op, clippy::vec_init_then_push)]
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

#[allow(clippy::identity_op, clippy::vec_init_then_push)]
fn build_model(pieces: &[(&str, f32, u8)]) -> Vec<u8> {
  let mut out = Vec::new();
  for (token, score, piece_type) in pieces {
    let body = build_piece(token, *score, *piece_type);
    out.push((1 << 3) | 2);
    write_varint(&mut out, body.len() as u64);
    out.extend_from_slice(&body);
  }
  // trainer_spec { model_type = 1 (unigram) }
  let mut trainer = Vec::new();
  trainer.push((3 << 3) | 0);
  write_varint(&mut trainer, 1);
  out.push((2 << 3) | 2);
  write_varint(&mut out, trainer.len() as u64);
  out.extend_from_slice(&trainer);
  out
}

/// A toy SPM whose vocabulary is:
///  0: `<unk>`   (Unknown)
///  1: `▁hello`  (Normal)
///  2: `▁world`  (Normal)
fn toy_spm() -> SentencePieceTokenizer {
  let normal = SentencePiecePieceType::Normal.as_raw() as u8;
  let unknown = SentencePiecePieceType::Unknown.as_raw() as u8;
  let data = build_model(&[
    ("<unk>", 0.0, unknown),
    ("\u{2581}hello", -1.0, normal),
    ("\u{2581}world", -2.0, normal),
  ]);
  SentencePieceTokenizer::from_model_bytes(&data).unwrap()
}

// ───────────────────────── SentencePiece decode ─────────────────────────

#[test]
fn sentencepiece_decode_of_known_id_sequence() {
  // The shared SentencePieceTokenizer is the source of truth for the decode;
  // assert the SenseVoice wrapper routes [1, 2] (`▁hello`, `▁world`) through it
  // and gets the metaspace-stripped, trimmed "hello world".
  let spm = toy_spm();
  let expected = spm.decode(&[1usize, 2usize]);
  assert_eq!(expected, "hello world");

  let tok = SenseVoiceTokenizer::from_sentencepiece(toy_spm());
  let got = tok.decode(&[1u32, 2u32]);
  assert_eq!(got, expected);
  assert_eq!(got, "hello world");
}

#[test]
fn sentencepiece_variant_is_not_id_join() {
  let tok = SenseVoiceTokenizer::from_sentencepiece(toy_spm());
  assert!(!tok.is_id_join());
}

// ───────────────────────── tokens.json piece-list decode ─────────────────────────

#[test]
fn token_list_join_strips_metaspace_and_trims() {
  // The reference `tokens.json` path: index each id, join, replace `▁` -> " ",
  // strip (`sensevoice.py:443-447`). Pieces with leading `▁` become
  // space-separated words; the final trim drops the leading space.
  let tokens = vec![
    "<unk>".to_string(),
    "\u{2581}hello".to_string(),
    "\u{2581}world".to_string(),
    "!".to_string(),
  ];
  let tok = SenseVoiceTokenizer::from_token_list(tokens);
  // ids [1, 2] -> "▁hello▁world" -> " hello world" -> trim -> "hello world".
  assert_eq!(tok.decode(&[1, 2]), "hello world");
  // ids [1, 2, 3] -> "▁hello▁world!" -> " hello world!" -> "hello world!".
  assert_eq!(tok.decode(&[1, 2, 3]), "hello world!");
}

#[test]
fn token_list_skips_out_of_range_ids() {
  // The reference guards `if 0 <= t < len(self._token_list)` (`sensevoice.py:444`);
  // an out-of-range id contributes nothing.
  let tokens = vec!["\u{2581}a".to_string(), "\u{2581}b".to_string()];
  let tok = SenseVoiceTokenizer::from_token_list(tokens);
  // id 5 is out of range and skipped; [0, 5, 1] -> "▁a▁b" -> "a b".
  assert_eq!(tok.decode(&[0, 5, 1]), "a b");
}

#[test]
fn token_list_empty_ids_is_empty_string() {
  let tokens = vec!["\u{2581}a".to_string()];
  let tok = SenseVoiceTokenizer::from_token_list(tokens);
  assert_eq!(tok.decode(&[]), "");
}

// ───────────────────────── id-join degenerate fallback ─────────────────────────

#[test]
fn id_join_renders_space_separated_decimals() {
  // The no-asset fallback (`sensevoice.py:448`):
  // `" ".join(str(t) for t in token_ids)`.
  let tok = SenseVoiceTokenizer::id_join();
  assert!(tok.is_id_join());
  assert_eq!(tok.decode(&[3, 14, 159]), "3 14 159");
  assert_eq!(tok.decode(&[7]), "7");
  assert_eq!(tok.decode(&[]), "");
}

// ───────────────────────── fallible try_decode parity ─────────────────────────

#[test]
fn try_decode_token_list_matches_infallible_decode() {
  // The fallible rich-path `try_decode` is byte-identical to the infallible
  // `decode` on the `tokens.json` variant — same join + metaspace-strip + trim,
  // only the output buffer is reserved fallibly (and the `▁`->" " substitution
  // is written inline instead of via a separate `replace` allocation).
  let tokens = vec![
    "<unk>".to_string(),
    "\u{2581}hello".to_string(),
    "\u{2581}world".to_string(),
    "!".to_string(),
  ];
  let tok = SenseVoiceTokenizer::from_token_list(tokens);
  for ids in [
    &[1u32, 2][..],
    &[1, 2, 3][..],
    &[0, 5, 1][..], // out-of-range id 5 skipped
    &[][..],
    &[3][..], // a piece with no metaspace + no surrounding space
  ] {
    assert_eq!(
      tok.try_decode(ids).unwrap(),
      tok.decode(ids),
      "try_decode must equal decode for ids {ids:?}"
    );
  }
  // Pin the headline values explicitly.
  assert_eq!(tok.try_decode(&[1, 2]).unwrap(), "hello world");
  assert_eq!(tok.try_decode(&[1, 2, 3]).unwrap(), "hello world!");
  assert_eq!(tok.try_decode(&[]).unwrap(), "");
}

#[test]
fn try_decode_token_list_trims_internal_metaspace_runs() {
  // Multiple / leading / trailing metaspaces all normalize to single spaces and
  // the result trims, identically to `decode` (= `replace(▁, " ").trim()`).
  let tokens = vec![
    "\u{2581}\u{2581}a".to_string(), // double leading metaspace
    "\u{2581}b\u{2581}".to_string(), // leading + trailing metaspace
  ];
  let tok = SenseVoiceTokenizer::from_token_list(tokens);
  // "▁▁a" + "▁b▁" -> "  a b " -> trim -> "a b" (leading run trimmed, the inner
  // metaspace becomes the word separator, trailing metaspace trimmed).
  assert_eq!(tok.try_decode(&[0, 1]).unwrap(), tok.decode(&[0, 1]));
  assert_eq!(tok.try_decode(&[0, 1]).unwrap(), "a b");
}

#[test]
fn try_decode_id_join_matches_infallible_decode() {
  let tok = SenseVoiceTokenizer::id_join();
  for ids in [
    &[3u32, 14, 159][..],
    &[7][..],
    &[][..],
    &[0, 0, 4294967295][..],
  ] {
    assert_eq!(tok.try_decode(ids).unwrap(), tok.decode(ids));
  }
  assert_eq!(tok.try_decode(&[3, 14, 159]).unwrap(), "3 14 159");
  assert_eq!(tok.try_decode(&[u32::MAX]).unwrap(), "4294967295");
}

#[test]
fn try_decode_token_list_checked_capacity_sum_succeeds_on_normal_input() {
  // The capacity reservation sums the in-range piece byte-lengths with
  // `checked_add` (so a pathological id/piece set cannot wrap the `usize` sum to
  // an undersized capacity and let the subsequent `push` grow unbounded). A true
  // `usize` overflow is not constructible here (the toy pieces cannot sum past
  // `usize::MAX`), so this pins that the checked accumulation path — exercised
  // across several multibyte pieces plus a skipped out-of-range id — returns the
  // correct decode (no spurious `ArithmeticOverflow`).
  let tokens = vec![
    "\u{2581}alpha".to_string(), // 3-byte metaspace + 5 ASCII = 8 bytes
    "\u{2581}beta".to_string(),  // 7 bytes
    "\u{2581}gamma".to_string(), // 8 bytes
  ];
  let tok = SenseVoiceTokenizer::from_token_list(tokens);
  // ids [0, 1, 9, 2]: id 9 is out of range and skipped by the checked sum.
  let got = tok.try_decode(&[0, 1, 9, 2]).unwrap();
  assert_eq!(got, tok.decode(&[0, 1, 9, 2]));
  assert_eq!(got, "alpha beta gamma");
}

#[test]
fn try_decode_sentencepiece_matches_infallible_decode() {
  // The SentencePiece variant routes both paths through the shared decode; only
  // the SenseVoice-owned `usize` id buffer differs (fallible vs `collect`).
  let tok = SenseVoiceTokenizer::from_sentencepiece(toy_spm());
  for ids in [&[1u32, 2][..], &[2, 1][..], &[][..]] {
    assert_eq!(tok.try_decode(ids).unwrap(), tok.decode(ids));
  }
  assert_eq!(tok.try_decode(&[1, 2]).unwrap(), "hello world");
}

// ───────────────────────── bounded SPM .model read ─────────────────────────

/// A per-process tmpdir for the on-disk `.model` fixtures.
fn spm_temp_dir(name: &str) -> std::path::PathBuf {
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_sensevoice_spm_{}_{}",
    std::process::id(),
    name
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

#[test]
fn from_spm_file_reads_and_parses_a_small_model() {
  // A valid `.model` on disk round-trips through the bounded reader +
  // `from_spm_bytes`: the SentencePiece variant decodes the same as the
  // in-memory fixture.
  let dir = spm_temp_dir("spm_ok");
  let normal = SentencePiecePieceType::Normal.as_raw() as u8;
  let unknown = SentencePiecePieceType::Unknown.as_raw() as u8;
  let data = build_model(&[
    ("<unk>", 0.0, unknown),
    ("\u{2581}hello", -1.0, normal),
    ("\u{2581}world", -2.0, normal),
  ]);
  let path = dir.join(SPM_MODEL_FILE);
  std::fs::write(&path, &data).unwrap();

  let tok = SenseVoiceTokenizer::from_spm_file(&path)
    .expect("bounded read + parse")
    .expect("present file yields Some");
  assert!(!tok.is_id_join());
  assert_eq!(tok.decode(&[1u32, 2u32]), "hello world");
}

#[test]
fn from_spm_file_absent_is_none() {
  // An absent `.model` yields `Ok(None)` (the caller's "fall through to
  // tokens.json" signal — and the TOCTOU-closed bounded read handles presence
  // itself).
  let dir = spm_temp_dir("spm_absent");
  let path = dir.join(SPM_MODEL_FILE);
  assert!(SenseVoiceTokenizer::from_spm_file(&path).unwrap().is_none());
}

#[test]
fn from_spm_file_rejects_oversized_model() {
  // A `.model` over the generous MAX_SPM_MODEL_BYTES cap is rejected by the
  // bounded read (a soundness guard against a hostile huge `.model`) rather than
  // read into memory unbounded — a typed CapExceeded, not an OOM. The body need
  // not be valid SPM: the size check fires before any parse.
  let dir = spm_temp_dir("spm_oversized");
  let path = dir.join(SPM_MODEL_FILE);
  let oversized = vec![0u8; MAX_SPM_MODEL_BYTES as usize + 1];
  std::fs::write(&path, &oversized).unwrap();

  let err = SenseVoiceTokenizer::from_spm_file(&path);
  assert!(
    matches!(err, Err(crate::error::Error::CapExceeded(_))),
    "oversized .model must be a typed CapExceeded, got {err:?}"
  );
  let _ = std::fs::remove_dir_all(&dir);
}
