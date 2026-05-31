//! Hand-traced `encode` tests over a [`MockEmbeddingModel`]: a real
//! tokenizer encodes a 2-text batch, the padding / mask logic is asserted
//! explicitly, and mean / cls pooling + L2-normalization are checked against
//! values computed by hand from the canned hidden states.

use super::*;
use crate::embeddings::model::{EmbeddingModel, EmbeddingModelOutput, MockEmbeddingModel};

const TOL: f32 = 1e-5;

/// A model that returns hidden states like [`MockEmbeddingModel`] but emits a
/// **caller-supplied raw `pooled_output` array** verbatim — so a test can
/// inject a deliberately malformed pooler (wrong rank or wrong batch) that
/// [`MockEmbeddingModel::with_pooled`] (which always tiles to a well-formed
/// `(batch, hidden)`) cannot produce. Used to exercise the encode-side
/// `pooled_output` shape guard.
struct RawPooledModel {
  inner: MockEmbeddingModel,
  /// The exact `(hidden,)`-flat data and shape to return as `pooled_output`.
  pooled_data: Vec<f32>,
  pooled_shape: Vec<usize>,
}

impl EmbeddingModel for RawPooledModel {
  fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<EmbeddingModelOutput> {
    let out = self.inner.forward(input_ids, attention_mask)?;
    let pooled = Array::from_slice::<f32>(&self.pooled_data, &self.pooled_shape.as_slice())?;
    // Use `into_parts()` to move the inner `last_hidden_state` Array out
    // (avoids the `try_clone()?` allocation that the borrowed-accessor
    // path would otherwise require). Drops the inner `pooled_output`
    // since this helper overrides it with `pooled`.
    let (last_hidden_state, _) = out.into_parts();
    Ok(EmbeddingModelOutput::new(last_hidden_state, Some(pooled)))
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

#[test]
fn encode_mean_pool_normalized_two_text_batch() {
  let tok = word_tokenizer();
  // Canned per-position hidden rows (hidden = 2):
  //   pos0 = [1, 0], pos1 = [0, 1], pos2 = [1, 1]
  let model = MockEmbeddingModel::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);

  // "a b c" -> 3 real tokens (pos0,1,2); "d e" -> 2 real (pos0,1) + 1 pad.
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Mean)
    .with_normalize(true);
  let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
  assert_eq!(emb.shape(), vec![2, 2]);
  let v = emb.to_vec::<f32>().unwrap();

  // Row 0 mean over pos0,1,2 = ([1,0]+[0,1]+[1,1])/3 = [2/3, 2/3];
  // L2-normalized = [1/√2, 1/√2].
  let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
  assert!(vclose(&v[0..2], &[inv_sqrt2, inv_sqrt2]));

  // Row 1 mean over pos0,1 (pad excluded by mask) = ([1,0]+[0,1])/2 = [0.5,0.5];
  // L2-normalized = [1/√2, 1/√2].
  assert!(vclose(&v[2..4], &[inv_sqrt2, inv_sqrt2]));
}

#[test]
fn encode_mean_pool_unnormalized_excludes_padding() {
  let tok = word_tokenizer();
  let model = MockEmbeddingModel::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Mean)
    .with_normalize(false);
  let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
  let v = emb.to_vec::<f32>().unwrap();
  // Row 0: [2/3, 2/3] ; Row 1: [0.5, 0.5] (pad position excluded).
  assert!(vclose(&v[0..2], &[2.0 / 3.0, 2.0 / 3.0]));
  assert!(vclose(&v[2..4], &[0.5, 0.5]));
}

#[test]
fn encode_cls_pool_selects_first_real_token() {
  let tok = word_tokenizer();
  // pos0 distinctive so CLS (first real token) is identifiable.
  let model = MockEmbeddingModel::new(vec![vec![9.0, 3.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Cls)
    .with_normalize(false);
  let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
  assert_eq!(emb.shape(), vec![2, 2]);
  let v = emb.to_vec::<f32>().unwrap();
  // Both rows are right-padded, so the first real token is pos0 = [9, 3].
  assert!(vclose(&v[0..2], &[9.0, 3.0]));
  assert!(vclose(&v[2..4], &[9.0, 3.0]));
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

/// End-to-end: mean-pooled embeddings must be identical whether the
/// tokenizer has padding enabled or disabled. The pad id is a *real* vocab
/// id (`4` = "e", hidden row `[1,1]`), so a buggy mask-`1` pad cell would
/// pull that row into the mean and diverge — this asserts it does not.
#[test]
fn encode_mean_pool_invariant_to_tokenizer_padding() {
  let canned = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
  let model_a = MockEmbeddingModel::new(canned.clone());
  let model_b = MockEmbeddingModel::new(canned);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Mean)
    .with_normalize(false);

  let mut emb_unpadded = encode(&model_a, &word_tokenizer(), &["a b c", "d e"], &cfg).unwrap();
  let mut emb_padded = encode(&model_b, &padded_word_tokenizer(), &["a b c", "d e"], &cfg).unwrap();
  assert_eq!(emb_unpadded.shape(), emb_padded.shape());
  let vu = emb_unpadded.to_vec::<f32>().unwrap();
  let vp = emb_padded.to_vec::<f32>().unwrap();
  assert!(vclose(&vu, &vp), "padded={vp:?} unpadded={vu:?}");
  // Hand-checked unpadded values: row0 = [2/3,2/3], row1 = [0.5,0.5].
  assert!(vclose(&vp[0..2], &[2.0 / 3.0, 2.0 / 3.0]));
  assert!(vclose(&vp[2..4], &[0.5, 0.5]));
}

// ---- Regression: Cls/None honor the model's pooled_output ----

/// `Cls` strategy with a model-provided `pooled_output` must return that
/// trained pooler vector (swift `inputs.pooledOutput ?? …`), NOT the
/// hidden-states CLS token. The pooled rows are made distinct from every
/// canned hidden row so the source is unambiguous.
#[test]
fn encode_cls_uses_model_pooled_output_when_present() {
  let tok = word_tokenizer();
  // Hidden CLS (first real token) would be [9, 3]; the pooler emits [7, 5]
  // for item 0 and [6, 4] for item 1 — neither equals any hidden row.
  let model = MockEmbeddingModel::new(vec![vec![9.0, 3.0], vec![0.0, 1.0], vec![1.0, 1.0]])
    .with_pooled(vec![vec![7.0, 5.0], vec![6.0, 4.0]]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Cls)
    .with_normalize(false);
  let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
  assert_eq!(emb.shape(), vec![2, 2]);
  let v = emb.to_vec::<f32>().unwrap();
  assert!(
    vclose(&v[0..2], &[7.0, 5.0]),
    "expected pooled row 0, got {:?}",
    &v[0..2]
  );
  assert!(
    vclose(&v[2..4], &[6.0, 4.0]),
    "expected pooled row 1, got {:?}",
    &v[2..4]
  );
}

/// The `pooled_output` path still honors the normalize / dimension tail
/// (`pool_post`): L2-normalizing item 0's pooler row `[3, 4]` → `[0.6, 0.8]`.
#[test]
fn encode_cls_pooled_output_applies_normalize_and_dimension() {
  let tok = word_tokenizer();
  let model =
    MockEmbeddingModel::new(vec![vec![9.0, 3.0], vec![0.0, 1.0]]).with_pooled(vec![vec![3.0, 4.0]]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Cls)
    .with_normalize(true);
  let mut emb = encode(&model, &tok, &["a b", "a b"], &cfg).unwrap();
  let v = emb.to_vec::<f32>().unwrap();
  // Both batch items reuse the single pooler row [3,4]; ‖[3,4]‖ = 5.
  assert!(vclose(&v[0..2], &[0.6, 0.8]));
  assert!(vclose(&v[2..4], &[0.6, 0.8]));
}

/// When the model emits NO `pooled_output`, `Cls` falls back to the
/// hidden-states (mask-aware) CLS path unchanged — guards the `??` fallback.
#[test]
fn encode_cls_falls_back_to_hidden_states_without_pooled_output() {
  let tok = word_tokenizer();
  let model = MockEmbeddingModel::new(vec![vec![9.0, 3.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
  assert!(model.pooled.is_none());
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Cls)
    .with_normalize(false);
  let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
  let v = emb.to_vec::<f32>().unwrap();
  // Hidden-states CLS = first real token = pos0 = [9, 3] for both rows.
  assert!(vclose(&v[0..2], &[9.0, 3.0]));
  assert!(vclose(&v[2..4], &[9.0, 3.0]));
}

// ---- Regression: validate pooled_output shape before bypassing pooling ----

/// Build a [`RawPooledModel`] over the standard 3-position canned hidden
/// states, emitting `pooled_data` reshaped to `pooled_shape` verbatim.
fn raw_pooled_model(pooled_data: Vec<f32>, pooled_shape: Vec<usize>) -> RawPooledModel {
  RawPooledModel {
    inner: MockEmbeddingModel::new(vec![vec![9.0, 3.0], vec![0.0, 1.0], vec![1.0, 1.0]]),
    pooled_data,
    pooled_shape,
  }
}

/// A rank-1 (squeezed `[hidden]`) `pooled_output` must be rejected with
/// [`Error::RankMismatch`] for `Cls` — not normalized and returned as if it
/// covered the batch. Without the guard a custom / version-skewed model that
/// squeezed a batch-1 pooler would silently yield a wrong-shape embedding.
#[test]
fn encode_cls_rejects_wrong_rank_pooled_output() {
  let tok = word_tokenizer();
  // Squeezed rank-1 [hidden] pooler instead of (batch, hidden).
  let model = raw_pooled_model(vec![7.0, 5.0], vec![2]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Cls)
    .with_normalize(false);
  let err = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap_err();
  assert!(
    matches!(err, Error::RankMismatch(ref p) if p.actual() == 1),
    "expected RankMismatch(actual=1), got {err:?}"
  );
}

/// Same wrong-rank guard for the `None` strategy's `pooled_output` bypass.
#[test]
fn encode_none_rejects_wrong_rank_pooled_output() {
  let tok = word_tokenizer();
  let model = raw_pooled_model(vec![7.0, 5.0], vec![2]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::None)
    .with_normalize(false);
  let err = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap_err();
  assert!(
    matches!(err, Error::RankMismatch(ref p) if p.actual() == 1),
    "expected RankMismatch(actual=1), got {err:?}"
  );
}

/// A stale `[1, hidden]` `pooled_output` for a 2-text batch must be rejected
/// with [`Error::LengthMismatch`] for `Cls` — the batch dim (1) does not
/// cover the request (2), so normalizing / returning it would silently drop a
/// text.
#[test]
fn encode_cls_rejects_wrong_batch_pooled_output() {
  let tok = word_tokenizer();
  // (1, hidden) pooler for a 2-text batch.
  let model = raw_pooled_model(vec![7.0, 5.0], vec![1, 2]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Cls)
    .with_normalize(false);
  let err = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap_err();
  assert!(
    matches!(err, Error::LengthMismatch(_)),
    "expected LengthMismatch, got {err:?}"
  );
}

/// Same wrong-batch guard for the `None` strategy's `pooled_output` bypass.
#[test]
fn encode_none_rejects_wrong_batch_pooled_output() {
  let tok = word_tokenizer();
  let model = raw_pooled_model(vec![7.0, 5.0], vec![1, 2]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::None)
    .with_normalize(false);
  let err = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap_err();
  assert!(
    matches!(err, Error::LengthMismatch(_)),
    "expected LengthMismatch, got {err:?}"
  );
}

/// A correctly-ranked, correctly-batched `pooled_output` whose hidden width
/// differs from `last_hidden_state`'s hidden dim must be rejected with
/// [`Error::ShapePairMismatch`] for `Cls` — otherwise it would be normalized
/// / truncated and returned as embeddings of an unexpected dimension. The
/// canned hidden states are `(.., .., 2)`, so a `(2, 3)` pooler is
/// wrong-width while still passing the rank-2 and batch checks.
#[test]
fn encode_cls_rejects_wrong_hidden_width_pooled_output() {
  let tok = word_tokenizer();
  // (batch=2, hidden=3) pooler, but the model's hidden dim is 2.
  let model = raw_pooled_model(vec![7.0, 5.0, 1.0, 6.0, 4.0, 2.0], vec![2, 3]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Cls)
    .with_normalize(false);
  let err = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch, got {err:?}"
  );
}

/// Same wrong-hidden-width guard for the `None` strategy's `pooled_output`
/// bypass.
#[test]
fn encode_none_rejects_wrong_hidden_width_pooled_output() {
  let tok = word_tokenizer();
  let model = raw_pooled_model(vec![7.0, 5.0, 1.0, 6.0, 4.0, 2.0], vec![2, 3]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::None)
    .with_normalize(false);
  let err = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap_err();
  assert!(
    matches!(err, Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch, got {err:?}"
  );
}

/// A correctly-shaped `(batch, hidden)` raw `pooled_output` still passes the
/// guard and is returned (sanity: the guard rejects only malformed shapes,
/// not the valid path that the `with_pooled` tests already cover).
#[test]
fn encode_cls_accepts_correct_shape_raw_pooled_output() {
  let tok = word_tokenizer();
  let model = raw_pooled_model(vec![7.0, 5.0, 6.0, 4.0], vec![2, 2]);
  let cfg = EncodeConfig::new()
    .with_add_special_tokens(false)
    .with_strategy(PoolingStrategy::Cls)
    .with_normalize(false);
  let mut emb = encode(&model, &tok, &["a b c", "d e"], &cfg).unwrap();
  assert_eq!(emb.shape(), vec![2, 2]);
  let v = emb.to_vec::<f32>().unwrap();
  assert!(vclose(&v[0..2], &[7.0, 5.0]));
  assert!(vclose(&v[2..4], &[6.0, 4.0]));
}
