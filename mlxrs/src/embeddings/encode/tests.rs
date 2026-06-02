//! Hand-traced `encode` tests over a deterministic text-embedder mock: a real
//! tokenizer encodes a 2-text batch, the padding / mask logic is asserted
//! explicitly via `tokenize_and_pad` (both `Padding` schemes), and the thin
//! `encode` pipeline is shown to read the mock's `TextEncoding` and forward the
//! tokenized + padded id batch to the model's `embed_text` verbatim.
//!
//! Pooling / normalization are the model's concern (`pool` / `pool_embed`,
//! covered by `embeddings::pooling`'s tests), so `encode` no longer pools —
//! these tests cover only the tokenize → pad → `embed_text` orchestration it
//! still owns, including the model-declared `Padding::DynamicRightPad` (the
//! sentence-encoder default) and `Padding::FixedLength` (a fixed-position pooler
//! like SigLIP) schemes.

use super::*;
use crate::{dtype::Dtype, embeddings::Embedding};

const TOL: f32 = 1e-5;

/// A deterministic text-embedder mock parameterized by the [`TextEncoding`] it
/// declares: its "embedding" is the model input's `(batch, seq_len)` token-id
/// matrix cast to `f32`, returned verbatim. Because the output is exactly the
/// padded id batch, a test can build the expected embedding by hand from the
/// known tokenization (an independent oracle) and assert `encode` tokenizes +
/// pads per the declared encoding and forwards that batch to `embed_text`
/// unchanged. It implements [`TextEmbedder`] directly and is passed to `encode`
/// as `&dyn TextEmbedder`.
struct IdentityTextEmbedder {
  encoding: TextEncoding,
}

impl IdentityTextEmbedder {
  fn new(encoding: TextEncoding) -> Self {
    Self { encoding }
  }
}

impl TextEmbedder for IdentityTextEmbedder {
  fn text_encoding(&self) -> TextEncoding {
    self.encoding
  }

  fn embed_text(&self, input_ids: &Array, _attention_mask: &Array) -> Result<Embedding> {
    // The padded `(batch, seq_len)` `I32` ids cast to `f32`, verbatim — no
    // pooling. `encode`'s returned array is therefore exactly this matrix.
    Ok(Embedding::new(input_ids.astype(Dtype::F32)?))
  }
}

/// A [`TextEncoding`] with `DynamicRightPad` (the sentence-encoder default):
/// no special tokens, the given `max_length`, right-pad with `pad_token_id`.
fn dynamic(max_length: Option<usize>, pad_token_id: u32) -> TextEncoding {
  TextEncoding::new(false, max_length, Padding::DynamicRightPad { pad_token_id })
}

/// A [`TextEncoding`] with `FixedLength` (a fixed-position pooler): no special
/// tokens, no separate cap, pad/truncate every row to `length` with
/// `pad_token_id` (all-`1` mask). `eos_token_id` is `None` (plain
/// head-truncation); [`fixed_eos`] exercises the EOS-preserving variant.
fn fixed(length: usize, pad_token_id: u32) -> TextEncoding {
  TextEncoding::new(
    false,
    None,
    Padding::FixedLength {
      length,
      pad_token_id,
      eos_token_id: None,
    },
  )
}

/// A [`TextEncoding`] with `FixedLength` whose overlength truncation **preserves
/// the EOS** at the final position (`eos_token_id = Some(eos)`), the sticky-EOS
/// pooler contract (SigLIP).
fn fixed_eos(length: usize, pad_token_id: u32, eos: u32) -> TextEncoding {
  TextEncoding::new(
    false,
    None,
    Padding::FixedLength {
      length,
      pad_token_id,
      eos_token_id: Some(eos),
    },
  )
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

// ---- DynamicRightPad: right-pad to batch max + mask 0 over pad cells ----

#[test]
fn tokenize_and_pad_builds_right_padded_ids_and_mask() {
  let tok = word_tokenizer();
  // "a b c" -> [0,1,2] ; "d e" -> [3,4]. Batch max len = 3.
  let (mut ids, mut mask, seq_len) =
    tokenize_and_pad(&tok, &["a b c", "d e"], &dynamic(None, 7)).unwrap();
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
    tokenize_and_pad(&tok, &["a b c", "d e"], &dynamic(Some(2), 0)).unwrap();
  assert_eq!(seq_len, 2);
  assert_eq!(ids.to_vec::<i32>().unwrap(), vec![0, 1, 3, 4]);
  assert_eq!(mask.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
}

// ---- FixedLength: pad/truncate every row to a fixed length + all-1 mask ----

#[test]
fn tokenize_and_pad_fixed_length_pads_and_truncates_with_all_one_mask() {
  let tok = word_tokenizer();
  // Fixed length 4 with pad id 7. "a b c" -> [0,1,2] right-pads to [0,1,2,7];
  // "d e" -> [3,4] right-pads to [3,4,7,7]. The mask is ALL ones (a
  // fixed-position pooler reads every position, including pad cells).
  let (mut ids, mut mask, seq_len) =
    tokenize_and_pad(&tok, &["a b c", "d e"], &fixed(4, 7)).unwrap();
  assert_eq!(seq_len, 4);
  assert_eq!(ids.shape(), vec![2, 4]);
  assert_eq!(mask.shape(), vec![2, 4]);
  assert_eq!(ids.to_vec::<i32>().unwrap(), vec![0, 1, 2, 7, 3, 4, 7, 7]);
  assert_eq!(
    mask.to_vec::<f32>().unwrap(),
    vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "FixedLength uses an all-1 mask (no pad cells are masked out)"
  );
}

#[test]
fn tokenize_and_pad_fixed_length_truncates_longer_rows() {
  let tok = word_tokenizer();
  // Fixed length 2 truncates "a b c" -> [0,1] (head kept) and leaves "d e" as
  // [3,4]. seq_len = 2, all-1 mask, no padding needed.
  let (mut ids, mut mask, seq_len) =
    tokenize_and_pad(&tok, &["a b c", "d e"], &fixed(2, 7)).unwrap();
  assert_eq!(seq_len, 2);
  assert_eq!(ids.to_vec::<i32>().unwrap(), vec![0, 1, 3, 4]);
  assert_eq!(mask.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
}

// ---- FixedLength: EOS-preserving overlength truncation (sticky-EOS) ----

/// A sticky-EOS `FixedLength` truncation must keep the EOS at the **final**
/// position: an overlength row keeps its head to `length - 1` and the EOS id is
/// forced into the last slot, mirroring the HF SigLIP processor's
/// truncate-then-append-EOS (`post_process` head-truncates content to
/// `max_length - n_added_tokens` and the template then appends the EOS). A naive
/// `ids[..length]` head-keep would leave a content token (`c`) in the pooled
/// last slot — wrong for a sticky-EOS tower. Oracle: the documented HF semantics
/// computed by hand from the known tokenization, NOT the code under test.
#[test]
fn tokenize_and_pad_fixed_eos_keeps_eos_at_last_position_on_truncation() {
  let tok = word_tokenizer();
  // "a b c d" -> [0,1,2,3] (length 4) at fixed length 3 with EOS id 9.
  // HF truncate-then-append: keep first length-1 = 2 content ids [0,1], then
  // EOS at position 2 -> [0,1,9]. The dropped tail is [2,3] (incl. what would
  // have been the trailing content); the EOS, not `c`/`d`, is the last slot.
  let (mut ids, mut mask, seq_len) =
    tokenize_and_pad(&tok, &["a b c d"], &fixed_eos(3, 7, 9)).unwrap();
  assert_eq!(seq_len, 3);
  assert_eq!(ids.shape(), vec![1, 3]);
  let id_vec = ids.to_vec::<i32>().unwrap();
  assert_eq!(
    id_vec,
    vec![0, 1, 9],
    "overlength row: head kept to length-1, EOS forced at the last position"
  );
  assert_eq!(*id_vec.last().unwrap(), 9, "last position must be the EOS");
  // All-1 mask regardless of EOS preservation.
  assert_eq!(mask.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0]);
}

/// A batch where one row overflows and one fits: the overflowing row gets the
/// forced trailing EOS; the fitting row is right-padded normally (its own
/// trailing token is untouched — EOS preservation only fires on a genuine
/// truncation).
#[test]
fn tokenize_and_pad_fixed_eos_mixed_overlength_and_fitting_rows() {
  let tok = word_tokenizer();
  // Fixed length 3, pad id 7, EOS id 9.
  // "a b c d e" -> [0,1,2,3,4] (len 5 > 3): truncate -> [0,1] + EOS -> [0,1,9].
  // "a b"       -> [0,1]       (len 2 < 3): right-pad   -> [0,1,7].
  let (mut ids, _mask, seq_len) =
    tokenize_and_pad(&tok, &["a b c d e", "a b"], &fixed_eos(3, 7, 9)).unwrap();
  assert_eq!(seq_len, 3);
  assert_eq!(
    ids.to_vec::<i32>().unwrap(),
    vec![0, 1, 9, 0, 1, 7],
    "overflow row ends in EOS; fitting row is plain pad-filled (no forced EOS)"
  );
}

/// EOS preservation is a no-op for a row that already fits: a within-length
/// batch is **byte-identical** whether `eos_token_id` is set or not (the EOS is
/// only forced on a genuine truncation). This is what keeps a real SigLIP batch
/// of short prompts unchanged — the e2e parity floor is preserved.
#[test]
fn tokenize_and_pad_fixed_eos_byte_identical_to_plain_when_within_length() {
  let tok = word_tokenizer();
  // Both rows fit within length 4: "a b c" -> [0,1,2,7], "d e" -> [3,4,7,7].
  let (mut plain, _m1, _s1) = tokenize_and_pad(&tok, &["a b c", "d e"], &fixed(4, 7)).unwrap();
  let (mut with_eos, _m2, _s2) =
    tokenize_and_pad(&tok, &["a b c", "d e"], &fixed_eos(4, 7, 9)).unwrap();
  assert_eq!(
    plain.to_vec::<i32>().unwrap(),
    with_eos.to_vec::<i32>().unwrap(),
    "within-length rows are unaffected by eos_token_id (no forced EOS)"
  );
  // And neither contains the EOS id 9 (it was never forced).
  assert!(!with_eos.to_vec::<i32>().unwrap().contains(&9));
}

/// A row of EXACTLY `length` ids is **not** truncated, so no EOS is forced — the
/// boundary case `ids.len() == length`. The all-`length` row is kept verbatim
/// (the EOS-preservation predicate is strictly `ids.len() > length`).
#[test]
fn tokenize_and_pad_fixed_eos_exact_length_row_not_truncated() {
  let tok = word_tokenizer();
  // "a b c d" -> [0,1,2,3] is exactly length 4: kept verbatim, no forced EOS.
  let (mut ids, _mask, seq_len) =
    tokenize_and_pad(&tok, &["a b c d"], &fixed_eos(4, 7, 9)).unwrap();
  assert_eq!(seq_len, 4);
  assert_eq!(
    ids.to_vec::<i32>().unwrap(),
    vec![0, 1, 2, 3],
    "ids.len() == length is not a truncation; the row is kept verbatim"
  );
}

/// The `FixedLength` attention mask is built fallibly via
/// [`crate::model_validation::alloc_filled`]: every cell is `1.0` and the buffer
/// length is exactly `batch * length`. A bare `vec![1.0; total]` would abort on
/// allocation pressure; the fallible helper returns a typed
/// [`crate::Error::AllocFailure`] instead. This pins the mask is the all-`1`
/// fallibly-allocated buffer for both the fitting and overlength paths.
#[test]
fn tokenize_and_pad_fixed_eos_mask_is_all_ones_via_fallible_alloc() {
  let tok = word_tokenizer();
  // Mixed batch (one overlength, one fitting) at length 3 -> mask is 2*3 = 6
  // ones, independent of truncation / EOS forcing.
  let (_ids, mut mask, seq_len) =
    tokenize_and_pad(&tok, &["a b c d e", "a b"], &fixed_eos(3, 7, 9)).unwrap();
  assert_eq!(seq_len, 3);
  assert_eq!(mask.shape(), vec![2, 3]);
  assert_eq!(
    mask.to_vec::<f32>().unwrap(),
    vec![1.0; 6],
    "fixed-length mask is all-1, length batch*length, via the fallible alloc_filled helper"
  );
}

/// The forced EOS id is range-checked like every other written id: an
/// `eos_token_id > i32::MAX` reaching a truncated row is a recoverable
/// [`Error::OutOfRange`] (`encode: eos_token_id`), never a silent wrap to a
/// negative index.
#[test]
fn tokenize_and_pad_fixed_eos_rejects_eos_id_above_i32_max() {
  let tok = word_tokenizer();
  let bad_eos: u32 = 0x8000_0000; // > i32::MAX
  // "a b c d" (len 4) at length 2 forces the (bad) EOS on truncation.
  let err = tokenize_and_pad(&tok, &["a b c d"], &fixed_eos(2, 7, bad_eos)).unwrap_err();
  match err {
    Error::OutOfRange(p) => {
      assert_eq!(p.context(), "encode: eos_token_id");
      assert_eq!(p.value(), "2147483648");
    }
    other => panic!("expected OutOfRange(eos_token_id), got {other:?}"),
  }
}

/// A fixed-length sticky-EOS model truncates the INTERMEDIATE encoded row to its
/// own derived cap (`length + 1`, the SigLIP scheme) so a prompt that exceeds the
/// fixed length head-truncates to the cap before `build_fixed_length` forces the
/// EOS into the final slot. This drives the two real private stages in order —
/// `encode_rows` (tokenize with the `length + 1` cap) then `build_fixed_length`:
/// the intermediate is exactly `length + 1` ids and the final row ends in the
/// forced EOS. Oracle: a 20-word prompt against length 4 — the intermediate is
/// exactly 5 ids, and the final row ends in EOS.
#[test]
fn fixed_eos_caps_intermediate_encoded_row_and_preserves_eos() {
  let tok = word_tokenizer();
  let length = 4usize;
  let eos = 9u32;
  // A prompt longer than the fixed length: 20 whitespace words -> 20 ids, which
  // the derived cap (5) head-truncates.
  let long: String = std::iter::repeat_n("a", 20).collect::<Vec<_>>().join(" ");

  // Stage 1: the derived tokenizer cap (`length + 1`) head-truncates the row. The
  // cap is what `effective_token_cap` derives for an EOS-preserving fixed length.
  let cap = length + 1;
  let rows = encode_rows(&tok, &[long.as_str()], false, Some(cap)).unwrap();
  assert_eq!(rows.len(), 1);
  let (intermediate_ids, intermediate_mask) = &rows[0];
  assert_eq!(
    intermediate_ids.len(),
    cap,
    "the intermediate encoded row is head-truncated to length + 1"
  );
  assert_eq!(
    intermediate_mask.len(),
    cap,
    "the synthesized mask matches the truncated ids"
  );

  // Stage 2: the fixed-length builder forces the EOS into the final slot. Because
  // the intermediate (5 ids) still exceeds `length` (4), the head is kept to
  // `length - 1` and the EOS occupies the last position.
  let (mut ids, _mask, seq_len) = build_fixed_length(&rows, length, 7, Some(eos)).unwrap();
  assert_eq!(seq_len, length);
  let id_vec = ids.to_vec::<i32>().unwrap();
  assert_eq!(id_vec.len(), length);
  assert_eq!(
    *id_vec.last().unwrap(),
    eos as i32,
    "the EOS is preserved in the final slot after the capped intermediate"
  );
}

// ---- effective_token_cap: central derivation from the padding scheme ----

/// `effective_token_cap` derives the tokenizer cap CENTRALLY from the padding
/// scheme, so a `FixedLength` encoding is bounded even when `max_length = None`:
/// `length + 1` for a sticky-EOS fixed length, `length` for a plain one, and
/// `None` (no intrinsic cap) for `DynamicRightPad`.
#[test]
fn effective_token_cap_derives_from_fixed_length_padding() {
  // Sticky-EOS fixed length with NO explicit max_length: cap is length + 1.
  assert_eq!(
    effective_token_cap(&fixed_eos(64, 1, 1)),
    Some(65),
    "sticky-EOS FixedLength derives length + 1 even with max_length = None"
  );
  // Plain fixed length (no EOS preservation): cap is exactly length.
  assert_eq!(
    effective_token_cap(&fixed(64, 1)),
    Some(64),
    "plain FixedLength derives length"
  );
  // DynamicRightPad has no intrinsic cap; it is exactly the explicit max_length.
  assert_eq!(
    effective_token_cap(&dynamic(None, 0)),
    None,
    "DynamicRightPad with no max_length stays uncapped"
  );
  assert_eq!(
    effective_token_cap(&dynamic(Some(128), 0)),
    Some(128),
    "DynamicRightPad cap is its explicit max_length"
  );
}

/// When BOTH an intrinsic (padding-scheme) cap and an explicit `max_length` are
/// present, `effective_token_cap` takes the TIGHTER (minimum) bound for a PLAIN
/// fixed length — an explicit cap can only shrink the bound, never loosen the
/// padding scheme's truncation. For a sticky-EOS fixed length the cap is instead
/// floored at `length + 1` (see
/// [`effective_token_cap_sticky_eos_floors_explicit_cap_at_length_plus_one`]).
#[test]
fn effective_token_cap_takes_tighter_of_intrinsic_and_explicit() {
  // PLAIN fixed length (no EOS): length 64 vs explicit 10 -> the tighter 10.
  let tight_plain = TextEncoding::new(
    true,
    Some(10),
    Padding::FixedLength {
      length: 64,
      pad_token_id: 1,
      eos_token_id: None,
    },
  );
  assert_eq!(
    effective_token_cap(&tight_plain),
    Some(10),
    "a tighter explicit max_length wins over a plain fixed length's intrinsic cap"
  );
  // length + 1 = 65 vs explicit 1000 -> the tighter 65 (the padding scheme), for
  // a sticky-EOS fixed length (the explicit cap is looser than the floor).
  let loose_explicit = TextEncoding::new(
    true,
    Some(1000),
    Padding::FixedLength {
      length: 64,
      pad_token_id: 1,
      eos_token_id: Some(1),
    },
  );
  assert_eq!(
    effective_token_cap(&loose_explicit),
    Some(65),
    "the intrinsic length + 1 wins over a looser explicit max_length"
  );
}

/// A sticky-EOS fixed length floors its effective cap at `length + 1`: an
/// explicit `max_length` that is `<= length` (the corruption case) must NOT pull
/// the cap below `length + 1`, because the `+ 1` slot is what lets the tokenizer
/// keep one id past the fixed length so `build_fixed_length` forces the trailing
/// EOS into the pooled final slot. Were the cap allowed to drop to `length`, an
/// overlength prompt would be truncated to exactly `length` content ids and the
/// EOS forcing would be skipped (`ids.len() == length`, not `> length`).
#[test]
fn effective_token_cap_sticky_eos_floors_explicit_cap_at_length_plus_one() {
  // The exact corruption case: max_length == Some(length). The naive min would
  // yield `length` (64); the floor holds it at `length + 1` (65).
  let cap_equals_length = TextEncoding::new(
    true,
    Some(64),
    Padding::FixedLength {
      length: 64,
      pad_token_id: 1,
      eos_token_id: Some(1),
    },
  );
  assert_eq!(
    effective_token_cap(&cap_equals_length),
    Some(65),
    "max_length == length must not suppress the sticky-EOS + 1 slot"
  );
  // An even tighter explicit cap (below length) is likewise floored at length + 1
  // for a sticky-EOS fixed length: the fixed length is the binding bound and the
  // EOS preservation needs the full window.
  let cap_below_length = TextEncoding::new(
    true,
    Some(8),
    Padding::FixedLength {
      length: 64,
      pad_token_id: 1,
      eos_token_id: Some(1),
    },
  );
  assert_eq!(
    effective_token_cap(&cap_below_length),
    Some(65),
    "an explicit cap below length is floored at length + 1 for a sticky-EOS scheme"
  );
  // A PLAIN fixed length has no such floor — a tighter explicit cap still wins
  // (no EOS slot to preserve).
  let plain_cap_below_length = TextEncoding::new(
    true,
    Some(8),
    Padding::FixedLength {
      length: 64,
      pad_token_id: 1,
      eos_token_id: None,
    },
  );
  assert_eq!(
    effective_token_cap(&plain_cap_below_length),
    Some(8),
    "a plain fixed length is not floored — the tighter explicit cap wins"
  );
}

/// The central derivation is what truncates a `FixedLength` model that leaves
/// `max_length = None`: `tokenize_and_pad` over such an encoding still
/// head-truncates the tokenizer to the derived cap, so a prompt longer than the
/// fixed length is truncated correctly. Drives the full `tokenize_and_pad` path
/// (which derives the cap internally) and checks the resulting fixed-length row,
/// with EOS forced on the genuine overlength.
#[test]
fn tokenize_and_pad_fixed_length_truncates_with_none_max_length() {
  let tok = word_tokenizer();
  // FixedLength { length: 3, eos } with max_length = None. A 20-word prompt
  // exceeds length 3, so the derived cap (4) head-truncates, then the EOS is
  // forced at the last slot.
  let enc = fixed_eos(3, 7, 9);
  assert_eq!(enc.max_length, None, "the encoding leaves max_length unset");
  let (mut ids, _mask, seq_len) = tokenize_and_pad(&tok, &[&"a ".repeat(20)], &enc).unwrap();
  assert_eq!(seq_len, 3);
  let id_vec = ids.to_vec::<i32>().unwrap();
  assert_eq!(id_vec.len(), 3);
  assert_eq!(
    *id_vec.last().unwrap(),
    9,
    "the derived cap truncated the tokenizer and the EOS is forced on truncation"
  );
}

/// Regression: a sticky-EOS `FixedLength` whose model ALSO sets an explicit
/// `max_length == Some(length)` must STILL preserve the trailing EOS on an
/// overlength prompt — the final pooled slot is the EOS, never a content token.
///
/// Without the sticky-EOS floor, `effective_token_cap` would take
/// `min(length + 1, length) == length`, the tokenizer would head-truncate the
/// overlength prompt to exactly `length` content ids, and `build_fixed_length`
/// would see `ids.len() == length` (not `> length`) and skip the EOS forcing —
/// leaving a content token in the pooled final slot and silently corrupting the
/// embedding. The floor holds the cap at `length + 1`, so the row is `length + 1`
/// ids and the EOS is forced into the last position. Oracle: the documented
/// sticky-EOS contract computed by hand from the known tokenization.
#[test]
fn tokenize_and_pad_fixed_eos_explicit_cap_equal_length_still_preserves_eos() {
  let tok = word_tokenizer();
  let length = 3usize;
  let eos = 9u32;
  // A sticky-EOS FixedLength that redundantly pins max_length == length (the
  // corruption case the floor fixes).
  let enc = TextEncoding::new(
    false,
    Some(length),
    Padding::FixedLength {
      length,
      pad_token_id: 7,
      eos_token_id: Some(eos),
    },
  );
  assert_eq!(
    enc.max_length,
    Some(length),
    "the encoding pins max_length == length (the corruption case)"
  );
  // A 20-word prompt exceeds length 3, so it is a genuine overlength truncation.
  let long: String = std::iter::repeat_n("a", 20).collect::<Vec<_>>().join(" ");
  let (mut ids, _mask, seq_len) = tokenize_and_pad(&tok, &[long.as_str()], &enc).unwrap();
  assert_eq!(seq_len, length);
  let id_vec = ids.to_vec::<i32>().unwrap();
  assert_eq!(id_vec.len(), length);
  assert_eq!(
    *id_vec.last().unwrap(),
    eos as i32,
    "an explicit max_length == length must NOT suppress the sticky-EOS final slot"
  );
}

/// The truncation cap is OUTPUT-IDENTICAL for a prompt the cap keeps in full: a
/// real (small) prompt encodes to the exact same ids / mask whether or not a cap
/// is supplied, because head-truncation only drops tokens past the cap. This is
/// the parity floor — a fitting prompt's observable encoding is unchanged.
#[test]
fn encode_rows_cap_is_output_identical_for_fitting_prompt() {
  let tok = word_tokenizer();
  // "a b c" (3 tokens) under any cap >= 3 fits in full, so the row is the same as
  // the uncapped encoding of the same text.
  let capped = encode_rows(&tok, &["a b c"], false, Some(8)).unwrap();
  let reference = encode_rows(&tok, &["a b c"], false, None).unwrap();
  assert_eq!(
    capped[0].0, reference[0].0,
    "a fitting prompt's ids are identical with or without a truncation cap"
  );
  assert_eq!(capped[0].0, vec![0u32, 1, 2]);
}

/// A prompt containing a long (70-byte) token among the first `cap` tokens
/// encodes IDENTICALLY through `encode_rows` and through a direct tokenize: the
/// caller's input is tokenized UNMODIFIED (never sliced), so any token is encoded
/// correctly regardless of its byte length. Oracle: the tokenizer output
/// head-capped to the same `cap`, computed independently of the code under test.
#[test]
fn encode_rows_long_token_matches_direct_tokenize() {
  // A WordLevel tokenizer whose first vocab word is 70 bytes long, plus two short
  // words.
  use tokenizers::{
    Tokenizer as HfTokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
  };
  let long_word = "z".repeat(70); // one whitespace-delimited token, 70 bytes
  let vocab = [
    (long_word.clone(), 0u32),
    ("a".to_string(), 1),
    ("b".to_string(), 2),
  ]
  .into_iter()
  .collect();
  let wl = WordLevel::builder()
    .vocab(vocab)
    .unk_token("a".to_string())
    .build()
    .unwrap();
  let mut hf = HfTokenizer::new(wl);
  hf.with_pre_tokenizer(Some(Whitespace {}));
  static FIXTURE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
  let dir = FIXTURE.get_or_init(|| {
    let dir = std::env::temp_dir().join(format!("mlxrs-emb-encode-longtok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    hf.save(dir.join("tokenizer.json"), false).unwrap();
    dir
  });
  let tok = Tokenizer::from_path(dir, None).unwrap();

  // Prompt = [long_word, "a", "b"] -> ids [0, 1, 2]; head-capped to cap = 2.
  let prompt = format!("{long_word} a b");
  let cap = 2usize;

  // Independent oracle: the direct tokenization, head-capped to the same cap.
  let opts = EncodeOptions::new()
    .with_add_special(false)
    .with_truncate_to(Some(cap))
    .with_return_attention_mask(true);
  let oracle = tok.encode_with(&prompt, &opts).unwrap();

  let rows = encode_rows(&tok, &[prompt.as_str()], false, Some(cap)).unwrap();
  assert_eq!(
    rows[0].0,
    oracle.ids().to_vec(),
    "encode_rows encodes the 70-byte token identically to a direct tokenize — \
     the input is never sliced"
  );
  assert_eq!(
    rows[0].0,
    vec![0u32, 1],
    "the 70-byte token (id 0) is the kept first id, intact"
  );
}

// ---- encode forwards the tokenized + padded batch to the model verbatim ----

#[test]
fn encode_forwards_dynamic_padded_id_batch_to_embed_text() {
  // `encode` is thin: read the model's `TextEncoding`, tokenize → pad →
  // `embed_text`. With the identity mock (embedding == the padded id matrix as
  // f32) and a `DynamicRightPad` encoding, `encode`'s output must be exactly
  // the `(batch, seq_len)` padded ids. The oracle is built by hand from the
  // known tokenization, NOT from `tokenize_and_pad`: "a b c" -> [0,1,2],
  // "d e" -> [3,4] + one pad cell (pad_token_id = 7) → seq_len = 3, padded
  // rows [[0,1,2],[3,4,7]].
  let tok = word_tokenizer();
  let model = IdentityTextEmbedder::new(dynamic(None, 7));
  let mut emb = encode(&model, &tok, &["a b c", "d e"]).unwrap();
  assert_eq!(emb.shape(), vec![2, 3]);
  // Exactly the padded id batch (the pad cell id = 7), cast to f32, verbatim.
  assert!(vclose(
    &emb.to_vec::<f32>().unwrap(),
    &[0.0, 1.0, 2.0, 3.0, 4.0, 7.0]
  ));
}

#[test]
fn encode_forwards_fixed_length_padded_id_batch_to_embed_text() {
  // A `FixedLength` encoding routes the same texts to a fixed `(batch, 4)`
  // batch with an all-1 mask: "a b c" -> [0,1,2,7], "d e" -> [3,4,7,7]. The
  // identity mock returns the ids verbatim, proving `encode` reads the model's
  // declared fixed-length scheme (this is the SigLIP routing path).
  let tok = word_tokenizer();
  let model = IdentityTextEmbedder::new(fixed(4, 7));
  let mut emb = encode(&model, &tok, &["a b c", "d e"]).unwrap();
  assert_eq!(emb.shape(), vec![2, 4]);
  assert!(vclose(
    &emb.to_vec::<f32>().unwrap(),
    &[0.0, 1.0, 2.0, 7.0, 3.0, 4.0, 7.0, 7.0]
  ));
}

#[test]
fn encode_truncates_then_forwards_to_embed_text() {
  // The `max_length` truncation happens in the tokenize step, so the model
  // sees the truncated + padded batch. max_length = 2 trims "a b c" to [0,1];
  // "d e" stays [3,4]. No padding is needed (both rows length 2), so the
  // oracle is the rank-2 `(2, 2)` matrix [[0,1],[3,4]] cast to f32.
  let tok = word_tokenizer();
  let model = IdentityTextEmbedder::new(dynamic(Some(2), 0));
  let mut emb = encode(&model, &tok, &["a b c", "d e"]).unwrap();
  assert_eq!(emb.shape(), vec![2, 2]);
  assert!(vclose(&emb.to_vec::<f32>().unwrap(), &[0.0, 1.0, 3.0, 4.0]));
}

#[test]
fn encode_empty_batch_forwards_zero_row_embedding() {
  // An empty `texts` slice produces a `(0, 0)` id batch; the identity mock
  // returns it verbatim, so `encode` yields a zero-row `(0, 0)` embedding (no
  // panic, no implicit row). seq_len = 0 because there are no rows.
  let tok = word_tokenizer();
  let model = IdentityTextEmbedder::new(dynamic(None, 0));
  let mut emb = encode(&model, &tok, &[]).unwrap();
  assert_eq!(emb.shape(), vec![0, 0]);
  assert!(emb.to_vec::<f32>().unwrap().is_empty());
}

// ---- Regression: tokenizer-applied padding must be masked ----

/// A tokenizer with HF padding enabled (`Fixed(4)`, `pad_id = 4`) must NOT
/// leak its pad cells into the attention mask under `DynamicRightPad`. With a
/// single text the batch max equals the real length, so `tokenize_and_pad`
/// adds no manual padding; every `0` in the mask therefore proves an HF pad
/// cell was stripped.
#[test]
fn tokenize_and_pad_strips_tokenizer_applied_padding() {
  let tok = padded_word_tokenizer();
  // "a b c" -> real ids [0,1,2]; HF then pads to length 4 with id 4. The
  // pad-stripping encode path must yield ids [0,1,2] + mask [1,1,1] (single
  // text => seq_len = 3, no manual padding), NOT a length-4 row whose pad
  // cell (id 4 = "e") is marked as a real token.
  let (mut ids, mut mask, seq_len) = tokenize_and_pad(&tok, &["a b c"], &dynamic(None, 0)).unwrap();
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
    tokenize_and_pad(&unpadded, &["a b c", "d e"], &dynamic(None, 7)).unwrap();
  let (mut p_ids, mut p_mask, p_seq) =
    tokenize_and_pad(&padded, &["a b c", "d e"], &dynamic(None, 7)).unwrap();
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

// ---- TextEncoding builders + accessors (round-trip) ----

/// [`TextEncoding::new`] persists its three fields and the public fields read
/// back, across both [`Padding`] variants.
#[test]
fn text_encoding_round_trips_both_padding_schemes() {
  let dyn_enc = TextEncoding::new(
    true,
    Some(256),
    Padding::DynamicRightPad { pad_token_id: 0 },
  );
  assert!(dyn_enc.add_special_tokens);
  assert_eq!(dyn_enc.max_length, Some(256));
  assert_eq!(
    dyn_enc.padding,
    Padding::DynamicRightPad { pad_token_id: 0 }
  );

  let fixed_enc = TextEncoding::new(
    false,
    None,
    Padding::FixedLength {
      length: 64,
      pad_token_id: 1,
      eos_token_id: Some(1),
    },
  );
  assert!(!fixed_enc.add_special_tokens);
  assert_eq!(fixed_enc.max_length, None);
  assert_eq!(
    fixed_enc.padding,
    Padding::FixedLength {
      length: 64,
      pad_token_id: 1,
      eos_token_id: Some(1),
    }
  );
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
  let err = tokenize_and_pad(&tok, &["big"], &dynamic(None, 0)).unwrap_err();
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
  let model = IdentityTextEmbedder::new(dynamic(None, 0));
  let err = encode(&model, &tok, &["big"]).unwrap_err();
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
  let err = tokenize_and_pad(&tok, &["a b"], &dynamic(None, bad_pad)).unwrap_err();
  match err {
    Error::OutOfRange(p) => {
      assert_eq!(p.context(), "encode: pad_token_id");
      assert_eq!(p.value(), "2147483648");
    }
    other => panic!("expected OutOfRange(pad_token_id), got {other:?}"),
  }
}

/// The `FixedLength` scheme range-checks its pad id too (a fixed-length batch
/// always writes the pad id into the `(batch, length)` tensor). A
/// `pad_token_id > i32::MAX` is a recoverable `OutOfRange`.
#[test]
fn tokenize_and_pad_fixed_length_rejects_pad_token_id_above_i32_max() {
  let tok = word_tokenizer();
  let err = tokenize_and_pad(&tok, &["a b"], &fixed(4, 0x8000_0000)).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context() == "encode: pad_token_id"),
    "expected OutOfRange(pad_token_id), got {err:?}"
  );
}

/// The `pad_token_id` overflow also surfaces through the public [`encode`]
/// entry via the model's declared [`TextEncoding`].
#[test]
fn encode_rejects_pad_token_id_above_i32_max() {
  let tok = word_tokenizer();
  let model = IdentityTextEmbedder::new(dynamic(None, 0x8000_0000));
  let err = encode(&model, &tok, &["a b"]).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(ref p) if p.context() == "encode: pad_token_id"),
    "expected OutOfRange(pad_token_id), got {err:?}"
  );
}

// ---- Cap derivation passes any explicit cap through (no input bound) ----

/// `effective_token_cap` passes a `DynamicRightPad` `max_length` through
/// UNCHANGED, however large — it is the tokenizer's head-truncation cap, not a
/// resource bound. A huge explicit cap is therefore just an effectively-uncapped
/// tokenize: a small prompt that fits well within it encodes normally (no
/// rejection). The library imposes no input-size limit; bounding an oversized
/// prompt is the consuming application's responsibility.
#[test]
fn dynamic_right_pad_large_cap_passes_through_and_tokenizes_fitting_prompt() {
  let tok = word_tokenizer();
  // A pathologically large explicit cap is returned unchanged by the derivation.
  let enc = dynamic(Some(usize::MAX), 0);
  assert_eq!(
    effective_token_cap(&enc),
    Some(usize::MAX),
    "DynamicRightPad's effective cap is its explicit max_length, unclamped"
  );
  // A short prompt that fits well within that cap encodes normally — no rejection,
  // since the cap is just a (huge) head-truncation bound, not an input guard.
  let (mut ids, _mask, seq_len) = tokenize_and_pad(&tok, &["a b c"], &enc).unwrap();
  assert_eq!(seq_len, 3);
  assert_eq!(ids.to_vec::<i32>().unwrap(), vec![0, 1, 2]);
  // The same holds through the public `encode` entry.
  let model = IdentityTextEmbedder::new(dynamic(Some(usize::MAX), 0));
  let mut emb = encode(&model, &tok, &["a b c"]).unwrap();
  assert_eq!(emb.shape(), vec![1, 3]);
  assert!(vclose(&emb.to_vec::<f32>().unwrap(), &[0.0, 1.0, 2.0]));
}
