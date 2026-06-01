//! Hand-traced `encode` tests over a deterministic text-embedder mock: a real
//! tokenizer encodes a 2-text batch, the padding / mask logic is asserted
//! explicitly via `tokenize_and_pad`, and the thin `encode` pipeline is shown to
//! forward the tokenized + padded id batch to the model's `embed_text` verbatim.
//!
//! Pooling / normalization moved into the model (`pool` / `pool_embed`, covered
//! by `embeddings::pooling`'s tests), so `encode` no longer pools — these tests
//! cover only the tokenize → pad → `embed_text` orchestration it still owns.

use super::*;
use crate::{
  dtype::Dtype,
  embeddings::{Embed, Embedding, TextInput},
};

const TOL: f32 = 1e-5;

/// A deterministic text-embedder mock: its "embedding" is the model input's
/// `(batch, seq_len)` token-id matrix cast to `f32`, returned verbatim. Because
/// the output is exactly the padded id batch, a test can build the expected
/// embedding by hand from the known tokenization (an independent oracle) and
/// assert `encode` forwards that batch to `embed_text` unchanged. It implements
/// [`Embed<TextInput>`] (so it is a [`TextEmbedder`] via the blanket projection)
/// and is passed to `encode` as `&dyn TextEmbedder`.
struct IdentityTextEmbedder;

impl<'a> Embed<TextInput<'a>> for IdentityTextEmbedder {
  type Output = Embedding;

  fn embed(&self, input: TextInput<'a>) -> Result<Embedding> {
    // The padded `(batch, seq_len)` `I32` ids cast to `f32`, verbatim — no
    // pooling. `encode`'s returned array is therefore exactly this matrix.
    Ok(Embedding::new(input.token_ids().astype(Dtype::F32)?))
  }
}

fn close(a: f32, b: f32) -> bool {
  (a - b).abs() <= TOL
}

fn vclose(a: &[f32], b: &[f32]) -> bool {
  a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

/// A whitespace word-level tokenizer with no special tokens: each distinct
/// word maps to a stable id (`a`→0 … `e`→4). Built in-memory via the public
/// `tokenizers` API, serialized to a temp `tokenizer.json`, and loaded
/// through [`Tokenizer::from_path`] — the same feature-combo-agnostic load
/// path the integration tests use (no dependence on the cfg-gated
/// `from_loaded` signature). Two texts of different word counts exercise the
/// pad / mask path.
fn word_tokenizer() -> Tokenizer {
  use tokenizers::{
    Tokenizer as HfTokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
  };

  // `WordLevelBuilder::vocab` takes the crate's `AHashMap<String, u32>`;
  // collect into it via the arg's inferred type (no extra dep named).
  let vocab = ["a", "b", "c", "d", "e"]
    .iter()
    .enumerate()
    .map(|(i, w)| ((*w).to_string(), i as u32))
    .collect();
  let wl = WordLevel::builder()
    .vocab(vocab)
    .unk_token("a".to_string())
    .build()
    .unwrap();
  let mut hf = HfTokenizer::new(wl);
  hf.with_pre_tokenizer(Some(Whitespace {}));

  // Serialize to a per-process temp dir (write-once), then load via the
  // public `from_path`. The content is deterministic; a `OnceLock` removes
  // the parallel-test write race while every test reads the same file.
  static FIXTURE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
  let dir = FIXTURE.get_or_init(|| {
    let dir = std::env::temp_dir().join(format!("mlxrs-emb-encode-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    hf.save(dir.join("tokenizer.json"), false).unwrap();
    dir
  });
  Tokenizer::from_path(dir, None).unwrap()
}

/// Same vocab / pre-tokenizer as [`word_tokenizer`], but with HF **padding
/// enabled** (`PaddingStrategy::Fixed(4)`, right side, `pad_id = 4`). Every
/// encoding is padded to length 4, so short inputs gain trailing pad cells
/// whose id is `4` — deliberately a *real* vocab id (`"e"`) so that if the
/// pad cells were ever treated as real tokens (mask `1`) they would corrupt
/// mask-aware pooling. The pad-stripping `encode_with` path must drop them.
/// Serialized to its own temp dir so it never collides with the unpadded
/// fixture above.
fn padded_word_tokenizer() -> Tokenizer {
  use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer as HfTokenizer,
    models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
  };

  let vocab = ["a", "b", "c", "d", "e"]
    .iter()
    .enumerate()
    .map(|(i, w)| ((*w).to_string(), i as u32))
    .collect();
  let wl = WordLevel::builder()
    .vocab(vocab)
    .unk_token("a".to_string())
    .build()
    .unwrap();
  let mut hf = HfTokenizer::new(wl);
  hf.with_pre_tokenizer(Some(Whitespace {}));
  hf.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::Fixed(4),
    direction: PaddingDirection::Right,
    pad_to_multiple_of: None,
    pad_id: 4,
    pad_type_id: 0,
    pad_token: "e".to_string(),
  }));

  static FIXTURE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
  let dir = FIXTURE.get_or_init(|| {
    let dir = std::env::temp_dir().join(format!("mlxrs-emb-encode-pad-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    hf.save(dir.join("tokenizer.json"), false).unwrap();
    dir
  });
  Tokenizer::from_path(dir, None).unwrap()
}

#[test]
fn tokenize_and_pad_builds_right_padded_ids_and_mask() {
  let tok = word_tokenizer();
  // "a b c" -> [0,1,2] ; "d e" -> [3,4]. Batch max len = 3.
  let (mut ids, mut mask, seq_len) =
    tokenize_and_pad(&tok, &["a b c", "d e"], false, None, 7).unwrap();
  assert_eq!(seq_len, 3);
  assert_eq!(ids.shape(), vec![2, 3]);
  assert_eq!(mask.shape(), vec![2, 3]);
  // Row 1 is right-padded with pad_token_id = 7 and mask 0 in the tail.
  // `input_ids` is `I32` (MLX's index dtype), so read it as `i32`.
  assert_eq!(ids.to_vec::<i32>().unwrap(), vec![0, 1, 2, 3, 4, 7]);
  assert_eq!(
    mask.to_vec::<f32>().unwrap(),
    vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
  );
}

#[test]
fn tokenize_and_pad_truncates_to_max_length() {
  let tok = word_tokenizer();
  // max_length = 2 trims "a b c" to [0,1]; "d e" is already 2. seq_len = 2.
  let (mut ids, mut mask, seq_len) =
    tokenize_and_pad(&tok, &["a b c", "d e"], false, Some(2), 0).unwrap();
  assert_eq!(seq_len, 2);
  assert_eq!(ids.to_vec::<i32>().unwrap(), vec![0, 1, 3, 4]);
  assert_eq!(mask.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
}

// ---- encode forwards the tokenized + padded batch to the model verbatim ----

#[test]
fn encode_forwards_padded_id_batch_to_embed_text() {
  // `encode` is thin: tokenize → right-pad → `embed_text`. With the identity
  // mock (embedding == the padded id matrix as f32), `encode`'s output must be
  // exactly the `(batch, seq_len)` padded ids the tokenizer produced. The
  // oracle is built by hand from the known tokenization, NOT from
  // `tokenize_and_pad`: "a b c" -> [0,1,2], "d e" -> [3,4] + one pad cell
  // (pad_token_id = 7) → seq_len = 3, padded rows [[0,1,2],[3,4,7]].
  let tok = word_tokenizer();
  let model = IdentityTextEmbedder;
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_pad_token_id(7);
  let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
  assert_eq!(emb.shape(), vec![2, 3]);
  // Exactly the padded id batch (the pad cell id = 7), cast to f32, verbatim.
  assert!(vclose(
    &emb.to_vec::<f32>().unwrap(),
    &[0.0, 1.0, 2.0, 3.0, 4.0, 7.0]
  ));
}

#[test]
fn encode_truncates_then_forwards_to_embed_text() {
  // The `max_length` truncation happens in the tokenize step, so the model
  // sees the truncated + padded batch. max_length = 2 trims "a b c" to [0,1];
  // "d e" stays [3,4]. No padding is needed (both rows length 2), so the
  // oracle is the rank-2 `(2, 2)` matrix [[0,1],[3,4]] cast to f32.
  let tok = word_tokenizer();
  let model = IdentityTextEmbedder;
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_max_length(Some(2));
  let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
  assert_eq!(emb.shape(), vec![2, 2]);
  assert!(vclose(&emb.to_vec::<f32>().unwrap(), &[0.0, 1.0, 3.0, 4.0]));
}

#[test]
fn encode_empty_batch_forwards_zero_row_embedding() {
  // An empty `texts` slice produces a `(0, 0)` id batch; the identity mock
  // returns it verbatim, so `encode` yields a zero-row `(0, 0)` embedding (no
  // panic, no implicit row). seq_len = 0 because there are no rows.
  let tok = word_tokenizer();
  let model = IdentityTextEmbedder;
  let cfg = EncodeConfig::new().with_add_special_tokens(false);
  let mut emb = encode(&model, &tok, &[], &cfg).unwrap();
  assert_eq!(emb.shape(), vec![0, 0]);
  assert!(emb.to_vec::<f32>().unwrap().is_empty());
}

// ---- Regression: tokenizer-applied padding must be masked ----

/// A tokenizer with HF padding enabled (`Fixed(4)`, `pad_id = 4`) must NOT
/// leak its pad cells into the attention mask. With a single text the batch
/// max equals the real length, so `tokenize_and_pad` adds no manual padding;
/// every `0` in the mask therefore proves an HF pad cell was stripped.
#[test]
fn tokenize_and_pad_strips_tokenizer_applied_padding() {
  let tok = padded_word_tokenizer();
  // "a b c" -> real ids [0,1,2]; HF then pads to length 4 with id 4. The
  // pad-stripping encode path must yield ids [0,1,2] + mask [1,1,1] (single
  // text => seq_len = 3, no manual padding), NOT a length-4 row whose pad
  // cell (id 4 = "e") is marked as a real token.
  let (mut ids, mut mask, seq_len) = tokenize_and_pad(&tok, &["a b c"], false, None, 0).unwrap();
  assert_eq!(seq_len, 3, "pad cells must be stripped, not counted");
  assert_eq!(ids.shape(), vec![1, 3]);
  assert_eq!(ids.to_vec::<i32>().unwrap(), vec![0, 1, 2]);
  assert_eq!(mask.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0]);
}

/// Cross-check: a 2-text batch through the padding-enabled tokenizer yields
/// the *same* ids + mask as the unpadded tokenizer — the only `0` mask cells
/// come from manual batch padding (row 1's single tail cell), never from the
/// tokenizer's own `Fixed(4)` padding.
#[test]
fn tokenize_and_pad_padded_tokenizer_matches_unpadded() {
  let unpadded = word_tokenizer();
  let padded = padded_word_tokenizer();
  let (mut u_ids, mut u_mask, u_seq) =
    tokenize_and_pad(&unpadded, &["a b c", "d e"], false, None, 7).unwrap();
  let (mut p_ids, mut p_mask, p_seq) =
    tokenize_and_pad(&padded, &["a b c", "d e"], false, None, 7).unwrap();
  assert_eq!(u_seq, p_seq);
  assert_eq!(
    u_ids.to_vec::<i32>().unwrap(),
    p_ids.to_vec::<i32>().unwrap()
  );
  assert_eq!(
    u_mask.to_vec::<f32>().unwrap(),
    p_mask.to_vec::<f32>().unwrap()
  );
  // Sanity: the shared mask is exactly the manual-pad layout (row 1 tail 0).
  assert_eq!(
    p_mask.to_vec::<f32>().unwrap(),
    vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
  );
}

// ---- EncodeConfig builders + accessors (round-trip) ----

/// Every `with_*` builder must persist its value and every accessor must
/// return it, across the three surviving tokenization knobs
/// (`add_special_tokens` / `max_length` / `pad_token_id`), and the documented
/// defaults are pinned. (Pooling / normalization config moved into the model,
/// so `EncodeConfig` no longer carries `strategy` / `normalize` / `dimension`
/// / `apply_*_norm`.)
#[test]
fn encode_config_builders_and_accessors_round_trip() {
  // Documented defaults (mirrors `EncodeConfig::default`).
  let d = EncodeConfig::new();
  assert!(d.add_special_tokens());
  assert_eq!(d.max_length(), Some(512));
  assert_eq!(d.pad_token_id(), 0);

  // Override every field to a non-default value and read it back.
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_max_length(None)
    .with_pad_token_id(7);
  assert!(!cfg.add_special_tokens());
  assert_eq!(cfg.max_length(), None);
  assert_eq!(cfg.pad_token_id(), 7);

  // `with_max_length(Some(n))` carries the inner value (the `None` arm is
  // covered above).
  let some = EncodeConfig::new().with_max_length(Some(8));
  assert_eq!(some.max_length(), Some(8));
}

// ---- tokenize_and_pad: id-overflow guards ----

/// A whitespace word-level tokenizer whose single vocab entry maps to a token
/// id `> i32::MAX` (`0x8000_0000` = 2_147_483_648). Encoding that word yields
/// an id that does NOT fit in `i32` (MLX's index dtype), so
/// `tokenize_and_pad`'s CHECKED `u32 -> i32` conversion of a *real* token id
/// must reject it with [`Error::OutOfRange`] rather than wrapping to a
/// negative index. Built + loaded the same `from_path` way as
/// [`word_tokenizer`], in its own temp dir.
fn huge_id_tokenizer() -> Tokenizer {
  use tokenizers::{
    Tokenizer as HfTokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
  };

  // One word -> an id beyond i32::MAX. `unk_token` must be a key present in
  // the vocab, so it shares the same out-of-range id (every produced id is
  // therefore out of i32 range).
  let big_id: u32 = 0x8000_0000; // 2_147_483_648 > i32::MAX (2_147_483_647)
  let vocab = [("big".to_string(), big_id)].into_iter().collect();
  let wl = WordLevel::builder()
    .vocab(vocab)
    .unk_token("big".to_string())
    .build()
    .unwrap();
  let mut hf = HfTokenizer::new(wl);
  hf.with_pre_tokenizer(Some(Whitespace {}));

  static FIXTURE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
  let dir = FIXTURE.get_or_init(|| {
    let dir =
      std::env::temp_dir().join(format!("mlxrs-emb-encode-huge-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    hf.save(dir.join("tokenizer.json"), false).unwrap();
    dir
  });
  Tokenizer::from_path(dir, None).unwrap()
}

/// A real token id `> i32::MAX` must be rejected with [`Error::OutOfRange`]
/// (the CHECKED per-id `u32 -> i32` conversion), not silently wrapped to a
/// negative index. `pad_token_id` is left at a valid `0` so the failure is
/// unambiguously the token-id path, not the pad-id pre-check.
#[test]
fn tokenize_and_pad_rejects_token_id_above_i32_max() {
  let tok = huge_id_tokenizer();
  let err = tokenize_and_pad(&tok, &["big"], false, None, 0).unwrap_err();
  match err {
    Error::OutOfRange(p) => {
      assert_eq!(p.context(), "encode: token id");
      assert_eq!(p.value(), "2147483648");
    }
    other => panic!("expected OutOfRange(token id), got {other:?}"),
  }
}

/// The same overflow surfaces through the public [`encode`] entry (which calls
/// `tokenize_and_pad` first): a vocab id `> i32::MAX` is a recoverable
/// `OutOfRange`, never a panic or wrapped index.
#[test]
fn encode_rejects_token_id_above_i32_max() {
  let tok = huge_id_tokenizer();
  let model = IdentityTextEmbedder;
  let cfg = EncodeConfig::new().with_add_special_tokens(false);
  let err = encode(&model, &tok, &["big"], &cfg).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context() == "encode: token id"),
    "expected OutOfRange(token id), got {err:?}"
  );
}

/// A `pad_token_id` `> i32::MAX` must be rejected up front with
/// [`Error::OutOfRange`] (the single pad-id range-check), independent of
/// whether the batch actually needs any padding. A single in-range word keeps
/// the per-token id path clean, so the only out-of-range value is the pad id.
#[test]
fn tokenize_and_pad_rejects_pad_token_id_above_i32_max() {
  let tok = word_tokenizer();
  let bad_pad: u32 = 0x8000_0000; // > i32::MAX
  let err = tokenize_and_pad(&tok, &["a b"], false, None, bad_pad).unwrap_err();
  match err {
    Error::OutOfRange(p) => {
      assert_eq!(p.context(), "encode: pad_token_id");
      assert_eq!(p.value(), "2147483648");
    }
    other => panic!("expected OutOfRange(pad_token_id), got {other:?}"),
  }
}

/// The `pad_token_id` overflow also surfaces through the public [`encode`]
/// entry via its [`EncodeConfig::with_pad_token_id`] knob.
#[test]
fn encode_rejects_pad_token_id_above_i32_max() {
  let tok = word_tokenizer();
  let model = IdentityTextEmbedder;
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_pad_token_id(0x8000_0000);
  let err = encode(&model, &tok, &["a b"], &cfg).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context() == "encode: pad_token_id"),
    "expected OutOfRange(pad_token_id), got {err:?}"
  );
}
