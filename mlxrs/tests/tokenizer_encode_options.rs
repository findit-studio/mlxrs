//! `EncodeOptions` + `Tokenizer::encode_with` tests.
//! The richer-surface builder is gated on the bare `tokenizer`
//! feature, but the tests below assert `add_eos`-appended ids against the
//! tokenizer's `eos_token_id` (`tokenizer-config`-gated accessor), so the
//! whole file gates on `tokenizer-config`. Mirrors the `tokenizer.rs`
//! fixture pattern (`OnceLock` write-once temp dir of the committed
//! `tokenizer.json` + `tokenizer_config.json`).
#![cfg(feature = "tokenizer-config")]

use std::io::Write;

use mlxrs::tokenizer::{EncodeOptions, Encoded, Tokenizer};

const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizer.json");
const TOKENIZER_CONFIG_JSON: &str = include_str!("fixtures/tokenizer_config.json");

/// Write-once fixture dir (same pattern as `tokenizer_core.rs` /
/// `tokenizer.rs`): `cargo test` runs the tests in parallel, but a constant
/// payload + `OnceLock` removes the rewrite race without serializing.
fn fixture_dir() -> std::path::PathBuf {
  static FIXTURE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
  FIXTURE
    .get_or_init(|| {
      let dir = std::env::temp_dir().join(format!(
        "mlxrs-tok-encode-opts-fixture-{}",
        std::process::id()
      ));
      std::fs::create_dir_all(&dir).unwrap();
      let mut f = std::fs::File::create(dir.join("tokenizer.json")).unwrap();
      f.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
      let mut c = std::fs::File::create(dir.join("tokenizer_config.json")).unwrap();
      c.write_all(TOKENIZER_CONFIG_JSON.as_bytes()).unwrap();
      dir
    })
    .clone()
}

#[test]
fn encode_with_defaults_matches_encode_true() {
  // `EncodeOptions::default()` has `add_special = true`, matching the
  // existing `encode(text, true)` call path. Back-compat assertion: the
  // additive `encode_with` does not alter the result the old API returns.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let text = "hello world the quick brown fox";

  let legacy = tok.encode(text, true).unwrap();
  let out = tok.encode_with(text, &EncodeOptions::default()).unwrap();

  assert_eq!(out.ids(), legacy);
  assert!(out.attention_mask().is_empty()); // default leaves the mask off
}

#[test]
fn encode_with_add_special_false_matches_legacy_false() {
  // The `add_special` flag must propagate to HF `tokenizer.encode`'s
  // `add_special_tokens`. The fixture's post-processor is null, so both
  // branches happen to produce the same ids — but the parity test still
  // proves the flag is wired through (not silently ignored).
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let text = "hello world";

  let legacy = tok.encode(text, false).unwrap();
  let out = tok
    .encode_with(text, &EncodeOptions::new().with_add_special(false))
    .unwrap();

  assert_eq!(out.ids(), legacy);
}

#[test]
fn encode_with_add_eos_uses_primary_not_smallest_id() {
  // When the eos set has multiple ids, the
  // appended EOS must be the PRIMARY (first user-supplied), NOT the
  // numerically smallest. Construct a tokenizer with a caller-supplied
  // multi-id stop list `[2, 0]` — id 2 is the real "</s>" (primary), id
  // 0 is "<unk>". If `add_eos` used `BTreeSet::iter().next()` it would
  // append 0 (the smallest), corrupting the prompt. The fix records the
  // FIRST slice element separately and uses it for emission.
  let tok = Tokenizer::from_path(fixture_dir(), Some(&[2, 0])).unwrap();
  // The full stop set has both ids (order-independent).
  assert!(tok.contains_eos_id(0));
  assert!(tok.contains_eos_id(2));

  let out = tok
    .encode_with(
      "hello world",
      &EncodeOptions::new()
        .with_add_special(false)
        .with_add_eos(true),
    )
    .unwrap();

  // The PRIMARY (first-supplied) eos id is 2 — that's what must be
  // appended, NOT the numerically smaller 0.
  assert_eq!(out.ids().last().copied(), Some(2u32));
  assert!(
    !out.ids()[..out.ids().len() - 1].contains(&2),
    "primary eos should appear only at the tail; got {:?}",
    out.ids()
  );
}

#[test]
fn encode_with_add_eos_appends_eos_id() {
  // `add_eos: true` must append the first id from `eos_token_ids`. The
  // fixture's `tokenizer_config.json` has `eos_token = "</s>"`, which the
  // fixture's `tokenizer.json` maps to id 2 — so the result is
  // `[ids..., 2]`. Asserts the EOS id is the LAST element, not buried.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let eos = tok.eos_token_id().expect("fixture has an eos token");
  let text = "hello world";

  let base = tok
    .encode_with(text, &EncodeOptions::new().with_add_special(false))
    .unwrap();
  let with_eos = tok
    .encode_with(
      text,
      &EncodeOptions::new()
        .with_add_special(false)
        .with_add_eos(true),
    )
    .unwrap();

  assert_eq!(with_eos.ids().last().copied(), Some(eos));
  assert_eq!(with_eos.ids().len(), base.ids().len() + 1);
  assert_eq!(&with_eos.ids()[..base.ids().len()], base.ids());
}

#[test]
fn encode_with_add_eos_errors_when_no_eos_configured() {
  // With no `tokenizer_config.json` eos AND no caller-supplied
  // `eos_token_ids`, the wrapper's eos set is empty (asserted by the empty
  // `eos_token_ids()`). `add_eos: true` must refuse rather than silently
  // no-op, exposing the config gap to the caller.
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-tok-encode-opts-noeos-{}",
    std::process::id()
  ));
  std::fs::create_dir_all(&dir).unwrap();
  let mut f = std::fs::File::create(dir.join("tokenizer.json")).unwrap();
  f.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
  // Deliberately no `tokenizer_config.json` written: `from_path` falls back
  // to an empty Value and `eos_token_ids` stays empty (no user-supplied set).
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert!(tok.eos_token_ids_iter().next().is_none());

  let err = tok
    .encode_with("hi", &EncodeOptions::new().with_add_eos(true))
    .unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("eos"),
    "expected eos-related error, got: {msg}"
  );
}

#[test]
fn encode_with_truncate_to_caps_length() {
  // `truncate_to: Some(n)` keeps the first `n` ids (HF
  // `TruncationDirection::Right` — drop the tail). The fixture's
  // "hello world the quick brown fox" encodes to 6 ids; cap to 3.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let text = "hello world the quick brown fox";

  let full = tok.encode(text, false).unwrap();
  assert!(
    full.len() > 3,
    "fixture must encode to >3 ids for this test"
  );

  let out = tok
    .encode_with(
      text,
      &EncodeOptions::new()
        .with_add_special(false)
        .with_truncate_to(Some(3)),
    )
    .unwrap();

  assert_eq!(out.ids().len(), 3);
  assert_eq!(out.ids(), &full[..3]);
}

#[test]
fn encode_with_truncate_to_above_length_is_noop() {
  // `truncate_to: Some(n)` with `n >= encoding_len` must not mutate ids —
  // mirrors HF's `Encoding::truncate` (`if max_len >= encoding_len { return }`).
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let text = "hello world";

  let full = tok.encode(text, false).unwrap();
  let out = tok
    .encode_with(
      text,
      &EncodeOptions::new()
        .with_add_special(false)
        .with_truncate_to(Some(full.len() + 100)),
    )
    .unwrap();

  assert_eq!(out.ids(), full);
}

#[test]
fn encode_with_return_attention_mask_matches_ids_len() {
  // `return_attention_mask: true` must yield a mask whose length equals
  // `ids.len()`. For a non-padded encoding the mask is all 1s.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let text = "hello world the quick brown fox";

  let out = tok
    .encode_with(text, &EncodeOptions::new().with_return_attention_mask(true))
    .unwrap();

  let mask = out.attention_mask();
  assert_eq!(mask.len(), out.ids().len());
  assert!(!mask.is_empty(), "mask requested");
  assert!(mask.iter().all(|&m| m == 1), "non-padded mask is all 1s");
}

#[test]
fn encode_with_truncate_zero_yields_empty_ids_and_mask() {
  // `truncate_to(Some(0))` yields an empty result. Because mlxrs does its
  // own bounded slicing rather than calling HF `Encoding::truncate`, the
  // empty case is just a zero-len slice with no special branch. Both `ids`
  // and `attention_mask` must end up empty in lock-step.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();

  let out = tok
    .encode_with(
      "hello world",
      &EncodeOptions::new()
        .with_truncate_to(Some(0))
        .with_return_attention_mask(true),
    )
    .unwrap();

  assert!(out.ids().is_empty());
  assert!(
    out.attention_mask().is_empty(),
    "mask requested + empty in lock-step with ids"
  );
}

#[test]
fn encode_with_truncate_zero_dominates_add_eos() {
  // The `n == 0` edge case of the "EOS guaranteed last" contract: an
  // empty cap dominates `add_eos`, so NO eos is appended (the output
  // must be empty). Pins the doc'd exception to the EOS+truncation
  // guarantee (the guarantee qualifies as `n >= 1`).
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();

  let out = tok
    .encode_with(
      "hello world",
      &EncodeOptions::new()
        .with_add_eos(true)
        .with_truncate_to(Some(0))
        .with_return_attention_mask(true),
    )
    .expect("eos is configured in the fixture, so add_eos does not error");

  assert!(
    out.ids().is_empty(),
    "n=0 cap dominates add_eos → empty ids"
  );
  assert!(
    out.attention_mask().is_empty(),
    "mask empty in lock-step with ids (mask requested)"
  );
}

#[cfg(feature = "tokenizer-stream")]
#[test]
fn encode_with_padded_tokenizer_strips_pads_and_eos_lands_after_real() {
  // When the HF tokenizer has padding enabled, naively
  // `merge_with`-ing an EOS encoding places `[tokens, pad..., eos]` with the
  // pad cells preserved. `encode_with` instead inserts EOS at the unpadded
  // boundary AND drops trailing pads — the returned ids/mask describe only
  // the real attended sequence.
  //
  // Test gates additionally on `tokenizer-stream` because `from_loaded`
  // takes the `DetokenizerClass` only when that feature is on.
  use tokenizers::{
    Tokenizer as HfTokenizer,
    utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy},
  };

  let mut hf = HfTokenizer::from_file(fixture_dir().join("tokenizer.json")).unwrap();
  // Pad every encoding out to 16 tokens with id=0 (`<unk>` in the fixture).
  hf.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::Fixed(16),
    direction: PaddingDirection::Right,
    pad_to_multiple_of: None,
    pad_id: 0,
    pad_type_id: 0,
    pad_token: "<unk>".into(),
  }));

  let cfg_bytes = std::fs::read(fixture_dir().join("tokenizer_config.json")).unwrap();
  let cfg: serde_json::Value = serde_json::from_slice(&cfg_bytes).unwrap();
  let tok =
    Tokenizer::from_loaded(hf, cfg, mlxrs::tokenizer::DetokenizerClass::Naive, None).unwrap();
  let eos = tok.eos_token_id().expect("fixture has eos");

  // Baseline: "hello world" without EOS, just to know the real length.
  let bare = tok
    .encode_with(
      "hello world",
      &EncodeOptions::new()
        .with_add_special(false)
        .with_return_attention_mask(true),
    )
    .unwrap();
  // Padded HF encoding would be 16 tokens; encode_with strips pads.
  assert_eq!(bare.ids().len(), 2);
  let bare_mask = bare.attention_mask();
  assert!(!bare_mask.is_empty(), "mask requested");
  assert!(bare_mask.iter().all(|&m| m == 1));

  // With `add_eos`: EOS sits at the unpadded boundary, no pads remain in
  // the result, and the mask stays all-1s (every cell is real / attended).
  let with_eos = tok
    .encode_with(
      "hello world",
      &EncodeOptions::new()
        .with_add_special(false)
        .with_add_eos(true)
        .with_return_attention_mask(true),
    )
    .unwrap();
  assert_eq!(with_eos.ids().len(), bare.ids().len() + 1);
  assert_eq!(with_eos.ids().last().copied(), Some(eos));
  // Padding id is 0; assert none of the result is the pad id.
  assert!(
    !with_eos.ids().contains(&0),
    "result must not contain pad id 0; got {:?}",
    with_eos.ids()
  );
  let mask = with_eos.attention_mask();
  assert!(!mask.is_empty(), "mask requested");
  assert_eq!(mask.len(), with_eos.ids().len());
  assert!(mask.iter().all(|&m| m == 1));
}

#[cfg(feature = "tokenizer-stream")]
#[test]
fn encode_with_left_padded_tokenizer_drops_leading_pads() {
  // When the HF tokenizer has LEFT padding
  // enabled (`[0, 0, real, real]`), `encode_with` must drop the leading
  // pad cells just as it does the trailing ones — every `mask == 0` cell
  // is dropped regardless of position, and the returned mask is all-1s.
  use tokenizers::{
    Tokenizer as HfTokenizer,
    utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy},
  };

  let mut hf = HfTokenizer::from_file(fixture_dir().join("tokenizer.json")).unwrap();
  hf.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::Fixed(8),
    direction: PaddingDirection::Left,
    pad_to_multiple_of: None,
    pad_id: 0,
    pad_type_id: 0,
    pad_token: "<unk>".into(),
  }));

  let cfg_bytes = std::fs::read(fixture_dir().join("tokenizer_config.json")).unwrap();
  let cfg: serde_json::Value = serde_json::from_slice(&cfg_bytes).unwrap();
  let tok =
    Tokenizer::from_loaded(hf, cfg, mlxrs::tokenizer::DetokenizerClass::Naive, None).unwrap();

  // Unpadded encoding of "hello world" is `[3, 4]` (2 tokens); left-padded
  // to 8 it becomes `[0, 0, 0, 0, 0, 0, 3, 4]`. `encode_with` must yield
  // only the 2 real ids in their HF order.
  let bare = tok
    .encode_with(
      "hello world",
      &EncodeOptions::new()
        .with_add_special(false)
        .with_return_attention_mask(true),
    )
    .unwrap();
  assert_eq!(bare.ids(), &[3u32, 4]);
  let bare_mask = bare.attention_mask();
  assert!(!bare_mask.is_empty(), "mask requested");
  assert!(bare_mask.iter().all(|&m| m == 1));

  // With `add_eos`: result is `[3, 4, eos]`, no leading pads.
  let eos = tok.eos_token_id().expect("fixture has eos");
  let with_eos = tok
    .encode_with(
      "hello world",
      &EncodeOptions::new()
        .with_add_special(false)
        .with_add_eos(true),
    )
    .unwrap();
  assert_eq!(with_eos.ids(), &[3u32, 4, eos]);
}

#[cfg(feature = "tokenizer-stream")]
#[test]
fn legacy_encode_preserves_hf_padding_layout() {
  // The legacy `encode` API must NOT strip
  // HF-applied padding cells — callers that pass a padded HfTokenizer
  // through `from_loaded` and read raw padded ids from `encode` rely on
  // the exact HF Encoding layout. Pad-stripping is opt-in via
  // `encode_with` only.
  use tokenizers::{
    Tokenizer as HfTokenizer,
    utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy},
  };

  let mut hf = HfTokenizer::from_file(fixture_dir().join("tokenizer.json")).unwrap();
  hf.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::Fixed(8),
    direction: PaddingDirection::Right,
    pad_to_multiple_of: None,
    pad_id: 0,
    pad_type_id: 0,
    pad_token: "<unk>".into(),
  }));

  let cfg_bytes = std::fs::read(fixture_dir().join("tokenizer_config.json")).unwrap();
  let cfg: serde_json::Value = serde_json::from_slice(&cfg_bytes).unwrap();
  let tok =
    Tokenizer::from_loaded(hf, cfg, mlxrs::tokenizer::DetokenizerClass::Naive, None).unwrap();

  // Legacy `encode` returns 8 ids (2 real + 6 pad), exactly as HF emits.
  let ids = tok.encode("hello world", false).unwrap();
  assert_eq!(ids.len(), 8);
  assert_eq!(&ids[..2], &[3u32, 4][..]);
  assert!(
    ids[2..].iter().all(|&id| id == 0),
    "trailing pads must be id=0; got {ids:?}"
  );
}

#[test]
fn encode_with_add_eos_errors_without_calling_hf_encode() {
  // The `add_eos` precondition is validated
  // BEFORE the underlying `hf.encode` call. We can't observe "did hf.encode
  // run" directly, but we can use a large valid input that would be
  // comparatively expensive to tokenize. Empty eos set + add_eos=true must
  // error fast regardless of input size/content.
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-tok-encode-opts-noeos-fastfail-{}",
    std::process::id()
  ));
  std::fs::create_dir_all(&dir).unwrap();
  let mut f = std::fs::File::create(dir.join("tokenizer.json")).unwrap();
  f.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert!(tok.eos_token_ids_iter().next().is_none());

  // Use a large input that would be expensive to tokenize: this validates
  // the precondition path errors regardless of input size.
  let big = "hello ".repeat(1024);
  let err = tok
    .encode_with(&big, &EncodeOptions::new().with_add_eos(true))
    .unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("eos"), "{msg}");
}

#[test]
fn encode_with_truncate_far_below_input_is_bounded_alloc() {
  // `Encoding::truncate(n)` preserves the discarded
  // tail in `Encoding::overflowing` (HF 0.23 behavior), so a 10k-token
  // input truncated to 8 would allocate the full 10k Encoding. `encode_with`
  // uses bounded slicing instead. Smoke check: a long input + tiny
  // `truncate_to` yields exactly `n` ids, no panic / OOM on a sized input
  // well past the truncation cap.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  // The fixture only has 6 in-vocab words, but `<unk>` (id 0) maps every
  // unknown word, so we can synthesize a long sequence cheaply.
  let mut buf = String::with_capacity(20_000);
  for i in 0..10_000 {
    if i > 0 {
      buf.push(' ');
    }
    buf.push_str("hello");
  }

  let out = tok
    .encode_with(
      &buf,
      &EncodeOptions::new()
        .with_add_special(false)
        .with_truncate_to(Some(8)),
    )
    .unwrap();
  assert_eq!(out.ids().len(), 8);
  // All 8 ids should be the "hello" id (== 3 in the fixture).
  assert!(out.ids().iter().all(|&id| id == 3));
}

#[test]
fn encode_with_add_eos_then_truncate_caps_including_eos() {
  // When `add_eos` is combined with `truncate_to(N)`, the EOS is
  // guaranteed-present in the result (it takes priority over the last
  // attended-id slot). For `truncate_to(2)` over a 6-id `base`, the head
  // is sliced to `final_len - 1 = 1` id then EOS appended → `[base[0],
  // eos]`. This matches the typical LM-training expectation that
  // "truncate-to-N with EOS" still ends in EOS.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let text = "hello world the quick brown fox";

  let base = tok
    .encode_with(text, &EncodeOptions::new().with_add_special(false))
    .unwrap();
  assert!(base.ids().len() >= 2);
  let eos = tok.eos_token_id().expect("fixture has eos");

  let out = tok
    .encode_with(
      text,
      &EncodeOptions::new()
        .with_add_special(false)
        .with_add_eos(true)
        .with_truncate_to(Some(2)),
    )
    .unwrap();

  assert_eq!(out.ids().len(), 2);
  assert_eq!(out.ids(), &[base.ids()[0], eos]);
}

#[test]
fn encoded_and_options_are_debug_clone() {
  // Compile-time trait-bound assertion: both types are `Debug + Clone`. The
  // bound on the generic helper drives a `where T: Debug + Clone` check, so
  // any future Debug/Clone regression fails the build (no runtime `.clone()`
  // needed — that would trip `clippy::redundant_clone`).
  fn assert_debug_clone<T: std::fmt::Debug + Clone>() {}
  assert_debug_clone::<EncodeOptions>();
  assert_debug_clone::<Encoded>();

  // Runtime smoke that the derived `Debug` is informative (field names
  // present), and that a cloned value retains the same Debug projection —
  // exercises the Clone impl without `let _ = x.clone()` (which clippy
  // flags as no-op).
  let opts = EncodeOptions::new()
    .with_add_special(false)
    .with_add_eos(true)
    .with_truncate_to(Some(128))
    .with_return_attention_mask(true);
  // Bind the clone to a real variable + mutate it so the value is observed
  // (clippy's `redundant_clone` triggers when the cloned value is dropped
  // without use). Field-equality on the chained-builder field-change proves
  // the Clone+chain composition is independent of the source.
  let opts_cloned = opts.clone().with_add_eos(false);
  assert!(opts.add_eos() && !opts_cloned.add_eos());
  let s = format!("{opts:?}");
  // Field names visible in the derived Debug — assert a couple as a smoke
  // signal that the format is not empty / not just "EncodeOptions".
  assert!(s.contains("add_eos"));
  assert!(s.contains("truncate_to"));

  let encoded = Encoded::new(vec![1, 2, 3], vec![1, 1, 1]);
  // Same Clone-is-independent assertion for `Encoded`.
  let mut encoded_cloned_ids = encoded.ids().to_vec();
  encoded_cloned_ids.push(4);
  assert_eq!(encoded.ids().len(), 3);
  assert_eq!(encoded_cloned_ids.len(), 4);
  let es = format!("{encoded:?}");
  assert!(es.contains("ids"));
}

// ---------------------------------------------------------------------------
// #112 — `encode_batch_with`: batch analogue of `encode_with` with
// the same `EncodeOptions` semantics applied per item, plus the same
// fast-fail-on-missing-eos contract.
// ---------------------------------------------------------------------------

#[test]
fn encode_batch_with_matches_encode_with_per_item() {
  // Parity: `encode_batch_with(texts, opts)` produces, item-for-item,
  // the same `Encoded` (ids + mask) as a hand-rolled `for t in texts {
  // tok.encode_with(t, opts) }` loop. Asserts on the exact opts that
  // exercise every post-processing branch (add_eos + truncate_to +
  // return_attention_mask + add_special) so any per-item drift surfaces.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let texts = vec![
    "hello world".to_owned(),
    "hello world the quick brown fox".to_owned(),
    "hello".to_owned(),
  ];
  let opts = EncodeOptions::new()
    .with_add_special(false)
    .with_add_eos(true)
    .with_truncate_to(Some(3))
    .with_return_attention_mask(true);

  let batched = tok.encode_batch_with(texts.clone(), &opts).unwrap();
  assert_eq!(batched.len(), texts.len());
  for (i, text) in texts.iter().enumerate() {
    let single = tok.encode_with(text, &opts).unwrap();
    assert_eq!(
      batched[i].ids(),
      single.ids(),
      "item {i} ids must match encode_with"
    );
    assert_eq!(
      batched[i].attention_mask(),
      single.attention_mask(),
      "item {i} mask must match encode_with"
    );
    // truncate_to(3) + add_eos => last id is EOS in EVERY non-empty item.
    let eos = tok.eos_token_id().expect("fixture has eos");
    assert_eq!(batched[i].ids().last().copied(), Some(eos));
    assert!(batched[i].ids().len() <= 3, "truncate cap honored");
  }
}

#[test]
fn encode_batch_with_add_eos_errors_without_calling_hf_encode_batch() {
  // Same fast-fail contract as `encode_with`: missing primary EOS +
  // `add_eos = true` rejects BEFORE the (potentially large) batch
  // tokenizer call runs, exposing the configuration gap up front.
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-tok-encode-opts-batch-noeos-{}",
    std::process::id()
  ));
  std::fs::create_dir_all(&dir).unwrap();
  let mut f = std::fs::File::create(dir.join("tokenizer.json")).unwrap();
  f.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert!(tok.eos_token_ids_iter().next().is_none());

  // Many large inputs that would be expensive to tokenize: the fast-fail
  // path errors regardless of batch size / per-item size.
  let big = "hello ".repeat(1024);
  let texts: Vec<String> = (0..32).map(|_| big.clone()).collect();
  let err = tok
    .encode_batch_with(texts, &EncodeOptions::new().with_add_eos(true))
    .unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("eos"),
    "expected eos-related error, got: {msg}"
  );
}

#[test]
fn encode_batch_with_empty_input_is_empty_output() {
  // Edge case: empty input vec yields empty output vec. Should NOT
  // error on `add_eos = true` if no items would have been encoded — but
  // because the precondition runs up front (before consulting input
  // length), the empty-input + add_eos + missing-eos case is still an
  // error. Test only the no-eos branch for the empty-vec round-trip.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let out = tok
    .encode_batch_with(Vec::new(), &EncodeOptions::new())
    .unwrap();
  assert!(out.is_empty());
}
