//! M4 — VLM `Model` trait + `vlm_generate` multimodal Iterator tests.
//!
//! Reference basis:
//! - python `mlx-vlm/mlx_vlm/generate.py::generate_step` (lines ~700–963 —
//!   `get_input_embeddings(input_ids, pixel_values, …)` →
//!   `_step(input_ids, inputs_embeds=…)` → `while True: _step(y[None])`),
//! - python `mlx-vlm/mlx_vlm/models/base.py` (`VisionLanguageModel`
//!   protocol; per-model `merge_input_ids_with_image_features` shape
//!   contract at `pixtral.py:104-153`),
//! - swift `mlx-swift-lm/Libraries/MLXVLM/VLMModel.swift` (the
//!   `VLMModel: LanguageModel, LoRAModel` marker).
//!
//! Deterministic, no real model. A local [`MockVlmModel`] returns canned
//! `[1, S, V]` logits and canned `[N, D]` vision embeds, and tracks every
//! `encode_image` / `embed_tokens` / `merge_embeddings` /
//! `forward_embeddings` / `forward` call so the per-step pipeline order,
//! span-splice positions, multi-image concat, zero-image passthrough, and
//! `image_processor_config` override are all observable without any
//! network or filesystem I/O beyond a single deterministic-PNG round-trip
//! per multimodal test.
#![cfg(feature = "vlm")]

use std::{
  cell::RefCell,
  fs,
  path::{Path, PathBuf},
  process,
};

use mlxrs::{
  Array, Error,
  error::RankMismatchPayload,
  lm::{
    cache::{CacheConfig, KvCache, make_prompt_cache},
    generate::GenConfig,
    model::Model as LmModel,
  },
  vlm::{
    generate::{VlmGenConfig, vlm_generate},
    image::{ColorOrder, ImageProcessorConfig, ResizeFilter},
    model::Model as VlmModel,
    prompt::MarkerPolicy,
  },
};

// ─────────────────────────── test infrastructure ─────────────────────────

/// A list of half-open `(start, end)` image spans (the shape the
/// `vlm::Model` trait passes around). Aliased for readability so the
/// capture types below aren't deeply-nested raw tuples.
type ImageSpans = Vec<(usize, usize)>;

/// `(text_embeds shape, image_embeds shape, image_spans)` — captured by
/// the mock's `merge_embeddings` override for per-call splice-contract
/// assertions.
type MergeCallSnapshot = (Vec<usize>, Vec<usize>, ImageSpans);

/// Process- and case-scoped temp dir for test image fixtures.
fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_vlm_generate_{}_{}", process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

/// Write a deterministic synthetic 8×8 RGB PNG and return its path. The
/// pixel content is irrelevant to the tests — [`MockVlmModel::encode_image`]
/// ignores the input and returns canned embeddings — but the PNG must be
/// a real file `vlm::image::load_image` can decode.
fn write_test_image(dir: &Path, name: &str) -> PathBuf {
  let path = dir.join(name);
  let mut buf = ::image::RgbImage::new(8, 8);
  for y in 0..8 {
    for x in 0..8 {
      buf.put_pixel(x, y, ::image::Rgb([(x * 32) as u8, (y * 32) as u8, 128]));
    }
  }
  ::image::DynamicImage::ImageRgb8(buf)
    .save_with_format(&path, ::image::ImageFormat::Png)
    .unwrap();
  path
}

/// A mock vision-language model that:
/// - returns canned `[1, S, V]` logits whose argmax is the LAST vocab
///   index (deterministic greedy decode),
/// - returns canned `[N_per_image, D]` vision embeddings (default
///   `D = 4`) — each image emits the same canned slab, distinguishable
///   only by call count,
/// - returns canned `[1, T, D]` text embeddings (zeros) so the
///   `merge_embeddings` default has unambiguous "text" positions,
/// - records every trait-method invocation in an internal log so a test
///   can assert the per-step call ordering, span widths, and shape
///   contracts.
struct MockVlmModel {
  vocab: usize,
  hidden_dim: usize,
  num_tokens_per_image: usize,
  /// Optional non-default `ImageProcessorConfig` returned by
  /// `image_processor_config`. `None` ⇒ the trait default.
  processor_cfg_override: Option<ImageProcessorConfig>,
  /// One entry per `encode_image` call: the input image's shape (the
  /// preprocessed array, which is `[H, W, 3]` channel-last).
  encode_calls: RefCell<Vec<Vec<usize>>>,
  /// One entry per `embed_tokens` call: the input token-window shape.
  embed_calls: RefCell<Vec<Vec<usize>>>,
  /// One entry per `merge_embeddings` call: `(text_shape, image_shape,
  /// spans)` — a complete snapshot of the splice contract.
  merge_calls: RefCell<Vec<MergeCallSnapshot>>,
  /// One entry per `forward_embeddings` call: the input embed shape
  /// (`[1, T, D]`).
  forward_emb_calls: RefCell<Vec<Vec<usize>>>,
  /// One entry per `forward` call: the input token-window shape
  /// (`[B, S]`).
  forward_calls: RefCell<Vec<Vec<usize>>>,
  /// When `true`, [`Self::forward`] returns an `Err` — drives the
  /// "per-step error is yielded once then the iterator fuses" path
  /// without needing a separate model type.
  fail_forward: bool,
  /// When `true`, [`Self::encode_image`] returns an `Err` so a test can
  /// assert encode-time failures surface as the `Err` of the
  /// `vlm_generate` `Result` (synchronous; before any step runs).
  fail_encode: bool,
}

impl MockVlmModel {
  fn new(vocab: usize, hidden_dim: usize, num_tokens_per_image: usize) -> Self {
    Self {
      vocab,
      hidden_dim,
      num_tokens_per_image,
      processor_cfg_override: None,
      encode_calls: RefCell::new(Vec::new()),
      embed_calls: RefCell::new(Vec::new()),
      merge_calls: RefCell::new(Vec::new()),
      forward_emb_calls: RefCell::new(Vec::new()),
      forward_calls: RefCell::new(Vec::new()),
      fail_forward: false,
      fail_encode: false,
    }
  }

  fn with_processor_cfg(mut self, cfg: ImageProcessorConfig) -> Self {
    self.processor_cfg_override = Some(cfg);
    self
  }
}

impl LmModel for MockVlmModel {
  fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    if self.fail_forward {
      return Err(Error::Backend("mock forward failure".into()));
    }
    let shape = tokens.shape();
    let (batch, seq) = match shape.as_slice() {
      [b, s] => (*b, *s),
      [s] => (1, *s),
      _ => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "MockVlmModel::forward expects [B, S] tokens (rank 1 or 2)",
          shape.len() as u32,
          shape.clone(),
        )));
      }
    };
    self.forward_calls.borrow_mut().push(shape);
    // Advance every cache layer by `seq` so cache wiring is observable.
    for layer in cache.iter_mut() {
      let elems = batch * seq * 2;
      let k = Array::from_slice::<f32>(&vec![1.0_f32; elems], &(batch, 1_usize, seq, 2_usize))?;
      let v = Array::from_slice::<f32>(&vec![2.0_f32; elems], &(batch, 1_usize, seq, 2_usize))?;
      layer.update(&k, &v)?;
    }
    // Ramp logits → argmax is vocab-1 (deterministic).
    let mut data = Vec::with_capacity(batch * seq * self.vocab);
    let canned: Vec<f32> = (0..self.vocab).map(|i| i as f32).collect();
    for _ in 0..batch * seq {
      data.extend_from_slice(&canned);
    }
    Array::from_slice::<f32>(&data, &(batch, seq, self.vocab))
  }

  fn forward_embeddings(
    &self,
    embeddings: &Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> mlxrs::Result<Array> {
    let shape = embeddings.shape();
    // Expect [1, T, D]; advance cache by T.
    let (batch, seq) = match shape.as_slice() {
      [b, s, _d] => (*b, *s),
      _ => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "MockVlmModel::forward_embeddings expects [B, T, D] (rank 3)",
          shape.len() as u32,
          shape.clone(),
        )));
      }
    };
    self.forward_emb_calls.borrow_mut().push(shape);
    for layer in cache.iter_mut() {
      let elems = batch * seq * 2;
      let k = Array::from_slice::<f32>(&vec![1.0_f32; elems], &(batch, 1_usize, seq, 2_usize))?;
      let v = Array::from_slice::<f32>(&vec![2.0_f32; elems], &(batch, 1_usize, seq, 2_usize))?;
      layer.update(&k, &v)?;
    }
    let mut data = Vec::with_capacity(batch * seq * self.vocab);
    let canned: Vec<f32> = (0..self.vocab).map(|i| i as f32).collect();
    for _ in 0..batch * seq {
      data.extend_from_slice(&canned);
    }
    Array::from_slice::<f32>(&data, &(batch, seq, self.vocab))
  }
}

impl VlmModel for MockVlmModel {
  fn embed_tokens(&self, tokens: &Array) -> mlxrs::Result<Array> {
    let shape = tokens.shape();
    let (b, t) = match shape.as_slice() {
      [b, t] => (*b, *t),
      _ => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "MockVlmModel::embed_tokens expects [B, T] (rank 2)",
          shape.len() as u32,
          shape.clone(),
        )));
      }
    };
    self.embed_calls.borrow_mut().push(shape);
    // Canned text embeds: zeros [B, T, D] — the merge splices image
    // embeds INTO these positions, so any non-image-span position in the
    // merged output should still be zero (observable by `to_vec()`).
    let data = vec![0.0_f32; b * t * self.hidden_dim];
    Array::from_slice::<f32>(&data, &(b, t, self.hidden_dim))
  }

  fn encode_image(&self, image: &Array) -> mlxrs::Result<Array> {
    if self.fail_encode {
      return Err(Error::Backend("mock encode_image failure".into()));
    }
    self.encode_calls.borrow_mut().push(image.shape());
    // Per-image-token slabs are deterministic & distinguishable: every
    // image emits the same canned [N, D] slab where each row [c, c, c,
    // c] (D=4) carries the row's 1-indexed token offset (so
    // image_embeds[0] = [1, 1, 1, 1], [1] = [2, 2, 2, 2], …). When the
    // splice runs, the merged output will have these canned values at
    // image-span positions and zeros elsewhere — a direct check on the
    // splice contract.
    let mut data = Vec::with_capacity(self.num_tokens_per_image * self.hidden_dim);
    for i in 0..self.num_tokens_per_image {
      for _ in 0..self.hidden_dim {
        data.push((i + 1) as f32);
      }
    }
    Array::from_slice::<f32>(&data, &(self.num_tokens_per_image, self.hidden_dim))
  }

  fn merge_embeddings(
    &self,
    text_embeds: &Array,
    image_embeds: &Array,
    image_spans: &[(usize, usize)],
  ) -> mlxrs::Result<Array> {
    self.merge_calls.borrow_mut().push((
      text_embeds.shape(),
      image_embeds.shape(),
      image_spans.to_vec(),
    ));
    // Delegate to the trait default — that's the splice we want to test.
    // To call the default impl from inside an override, we'd normally
    // structure the trait differently; here we manually invoke a copy
    // of the default's logic by re-routing through a free function.
    // For ergonomics, just call into the trait default via fully-
    // qualified syntax via a separate non-override helper.
    default_merge(text_embeds, image_embeds, image_spans)
  }

  fn image_processor_config(&self) -> ImageProcessorConfig {
    self.processor_cfg_override.unwrap_or_default()
  }
}

/// Free-function copy of [`crate::vlm::model::Model::merge_embeddings`]
/// default body, used by [`MockVlmModel::merge_embeddings`] so the
/// instrumented override still exercises the default-splice contract
/// (vs reimplementing a divergent splice, which would silently miss
/// regressions in the trait default).
fn default_merge(
  text_embeds: &Array,
  image_embeds: &Array,
  image_spans: &[(usize, usize)],
) -> mlxrs::Result<Array> {
  // The cleanest way to reuse the default body is via a tiny shim type
  // whose `merge_embeddings` is NOT overridden, then call that.
  struct Shim;
  impl LmModel for Shim {
    fn forward(&self, _tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      unreachable!("Shim::forward never called")
    }
  }
  impl VlmModel for Shim {
    fn embed_tokens(&self, _tokens: &Array) -> mlxrs::Result<Array> {
      unreachable!("Shim::embed_tokens never called")
    }
    fn encode_image(&self, _image: &Array) -> mlxrs::Result<Array> {
      unreachable!("Shim::encode_image never called")
    }
  }
  Shim.merge_embeddings(text_embeds, image_embeds, image_spans)
}

/// Build a fresh KV cache for the mock model's "1 layer" config.
fn mock_cache() -> Vec<Box<dyn KvCache>> {
  make_prompt_cache(&CacheConfig {
    num_hidden_layers: 2,
    sliding_window: None,
  })
}

/// Shared `VlmGenConfig` for the multimodal tests:
/// - max_tokens: small, so the iterator finishes promptly,
/// - greedy decode (temp = 0),
/// - image_token_id = 99, marker = 99 (single-id model),
/// - num_tokens_per_image = 3,
/// - MarkerPolicy::Required.
fn vlm_cfg(max_tokens: usize, num_tokens_per_image: usize) -> VlmGenConfig {
  VlmGenConfig::new(
    GenConfig::default().with_max_tokens(max_tokens),
    99,
    num_tokens_per_image,
    MarkerPolicy::Required,
  )
}

// ─────────────────────────── basic pipeline ──────────────────────────────

#[test]
fn vlm_generate_pipeline_smoke() {
  // One image, 3 image tokens per image. Prompt: [1, 2, 99, 3, 4]
  // (marker=99 in position 2). After splice: same shape (run-length 1
  // -> error). We need run-length to match image_count=1. So prompt
  // must have exactly 1 marker. After splice: [1, 2, 99, 99, 99, 3, 4]
  // (T=7), image_spans = [(2, 5)].
  let model = MockVlmModel::new(/*vocab=*/ 5, /*D=*/ 4, /*N_per_img=*/ 3);
  let dir = temp_dir("smoke");
  let img = write_test_image(&dir, "img.png");

  let prompt = [1_u32, 2, 99, 3, 4]; // marker run of len 1 → 1 image
  let it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    vlm_cfg(/*max=*/ 3, /*N_per_img=*/ 3),
  )
  .expect("vlm_generate construction succeeds");

  let mut tokens = Vec::new();
  for step in it {
    tokens.push(step.expect("step ok").token);
  }
  // Greedy decode on ramp logits ⇒ argmax = vocab-1 = 4, repeating
  // max_tokens times.
  assert_eq!(tokens, vec![4_u32, 4, 4]);

  // Pipeline invariants — each method called the expected number of
  // times in the expected shape.
  assert_eq!(model.encode_calls.borrow().len(), 1);
  assert_eq!(model.embed_calls.borrow().len(), 1);
  assert_eq!(model.merge_calls.borrow().len(), 1);
  // forward_embeddings called ONCE (prefill).
  assert_eq!(model.forward_emb_calls.borrow().len(), 1);
  // forward called (max_tokens - 1) times (each decode step except the
  // prefill-derived first token).
  assert_eq!(model.forward_calls.borrow().len(), 2);

  // Embed-input shape was [1, T, D] = [1, 7, 4].
  assert_eq!(model.forward_emb_calls.borrow()[0], vec![1_usize, 7, 4]);
  // Decode-input shape was [1, 1] each step.
  for s in model.forward_calls.borrow().iter() {
    assert_eq!(s, &vec![1_usize, 1]);
  }

  // Merge contract — text_embeds [1, 7, 4], image_embeds [3, 4],
  // spans [(2, 5)].
  let (text_shape, image_shape, spans) = model.merge_calls.borrow()[0].clone();
  assert_eq!(text_shape, vec![1, 7, 4]);
  assert_eq!(image_shape, vec![3, 4]);
  assert_eq!(spans, vec![(2_usize, 5_usize)]);
}

#[test]
fn vlm_generate_zero_images_passthrough() {
  // No images ⇒ pure lm::generate. Neither encode_image nor merge
  // should be called; forward_embeddings should NEVER fire (the zero-
  // image branch dispatches straight to lm::generate which never
  // touches forward_embeddings).
  let model = MockVlmModel::new(5, 4, 3);
  let prompt = [1_u32, 2, 3];
  let it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[],
    mock_cache(),
    vlm_cfg(2, 3),
  )
  .expect("vlm_generate constructs in zero-image mode");

  let mut tokens = Vec::new();
  for step in it {
    tokens.push(step.expect("step ok").token);
  }
  assert_eq!(tokens, vec![4_u32, 4]);
  assert!(model.encode_calls.borrow().is_empty());
  assert!(model.embed_calls.borrow().is_empty());
  assert!(model.merge_calls.borrow().is_empty());
  assert!(model.forward_emb_calls.borrow().is_empty());
  // forward called for each token (lm::generate's prefill chunks
  // prompt[..T-1] then first-step + decode loop — but at most
  // max_tokens forward calls for a tiny prompt).
  assert!(!model.forward_calls.borrow().is_empty());
}

/// L3: the zero-image VLM branch delegates to
/// `lm::generate_step`, which honors `cfg.lm.collect_logprobs` — and that
/// field's `Default` is `false`. The multimodal decode loop in
/// `vlm::generate` ALWAYS emits `Some(logprobs)` (see the comment at the
/// post-sampler squeeze: "VLM has not adopted the `collect_logprobs`
/// opt-in yet, so we always emit `Some`"), so the cross-crate VLM
/// contract is "every `GenStep.logprobs` is `Some`". Without an explicit
/// override the zero-image branch silently flipped to `None`, breaking
/// the contract — fixed by forcing `collect_logprobs = true` on the
/// branch-local `cfg.lm` clone before delegating. This test pins that:
/// a `vlm_generate(..., images=[], cfg with default cfg.lm)` MUST yield
/// `Some(logprobs)` on every step.
#[test]
fn vlm_generate_zero_image_preserves_logprobs() {
  let model = MockVlmModel::new(/*vocab=*/ 5, /*D=*/ 4, /*N_per_img=*/ 3);
  let prompt = [1_u32, 2, 3];
  // `vlm_cfg` builds a `VlmGenConfig` with `cfg.lm = GenConfig {
  // max_tokens, ..Default::default() }` — and `Default` leaves
  // `collect_logprobs == false`, which is exactly the regression
  // surface: a default-cfg zero-image VLM run would otherwise return
  // `None` logprobs and break the documented "VLM always Some" contract.
  let it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[],
    mock_cache(),
    vlm_cfg(2, 3),
  )
  .expect("vlm_generate constructs in zero-image mode");

  let mut saw_step = false;
  for step in it {
    let s = step.expect("step ok");
    assert!(
      s.logprobs.is_some(),
      "zero-image VLM must preserve Some(logprobs) even though \
       cfg.lm.collect_logprobs defaults to false (VLM contract drift \
       across the two branches)"
    );
    let lp = s.logprobs.as_ref().unwrap();
    // The mock returns `[1, S, vocab=5]`; per-step squeeze yields `[V]`.
    assert_eq!(
      lp.shape(),
      vec![5_usize],
      "zero-image VLM logprobs shape is the post-squeeze [V] vector"
    );
    saw_step = true;
  }
  assert!(saw_step, "vlm_generate must yield at least one step here");
}

#[test]
fn vlm_generate_marker_required_missing_errors() {
  // No marker in prompt + MarkerPolicy::Required ⇒ the assembler
  // surfaces MissingField as the `Err` of the `Result` (synchronous).
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("marker_missing");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 2, 3]; // no marker (99)
  let res = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    vlm_cfg(/*max=*/ 1, /*N_per_img=*/ 3),
  );
  match res {
    Ok(_) => panic!("expected Err on missing marker under Required policy"),
    Err(e) => {
      assert!(
        matches!(e, Error::MissingField(_)),
        "expected MissingField under MarkerPolicy::Required, got: {e:?}"
      );
    }
  }
}

// ─────────────────────── splice-position correctness ─────────────────────

#[test]
fn vlm_generate_image_tokens_spliced_correctly() {
  // After splice the merged embeds at image-span positions must equal
  // the canned encoder output (1, 2, 3 across all D), and the
  // non-image positions must be the canned text embed (zeros).
  //
  // Drive this by intercepting `merge_embeddings` — the mock's override
  // calls back into the default's free-function copy AND returns it,
  // so we can assert on the actual splice output via to_vec.

  // Build a custom model that captures the merge_embeddings return
  // value (the spliced [1, T, D] array) so the test can inspect it.
  struct CapturingModel {
    inner: MockVlmModel,
    captured: RefCell<Option<Vec<f32>>>,
    captured_shape: RefCell<Option<Vec<usize>>>,
  }
  impl LmModel for CapturingModel {
    fn forward(&self, t: &Array, c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      self.inner.forward(t, c)
    }
    fn forward_embeddings(&self, e: &Array, c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      self.inner.forward_embeddings(e, c)
    }
  }
  impl VlmModel for CapturingModel {
    fn embed_tokens(&self, t: &Array) -> mlxrs::Result<Array> {
      self.inner.embed_tokens(t)
    }
    fn encode_image(&self, i: &Array) -> mlxrs::Result<Array> {
      self.inner.encode_image(i)
    }
    fn merge_embeddings(
      &self,
      text: &Array,
      image: &Array,
      spans: &[(usize, usize)],
    ) -> mlxrs::Result<Array> {
      let mut merged = self.inner.merge_embeddings(text, image, spans)?;
      *self.captured_shape.borrow_mut() = Some(merged.shape());
      *self.captured.borrow_mut() = Some(merged.to_vec::<f32>()?);
      Ok(merged)
    }
  }

  let model = CapturingModel {
    inner: MockVlmModel::new(5, 4, 3),
    captured: RefCell::new(None),
    captured_shape: RefCell::new(None),
  };
  let dir = temp_dir("splice_correct");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 2, 99, 3, 4];
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    vlm_cfg(1, 3),
  )
  .expect("vlm_generate construction");
  it.next().expect("one step").expect("step ok");

  let shape = model.captured_shape.borrow().clone().expect("merge ran");
  let data = model.captured.borrow().clone().expect("merge ran");
  assert_eq!(shape, vec![1_usize, 7, 4]);
  // Layout: positions [0, 1] are text (zeros), [2..5] are image, [5, 6]
  // are text (zeros). D=4.
  let row = |pos: usize| -> Vec<f32> { data[pos * 4..pos * 4 + 4].to_vec() };
  assert_eq!(row(0), vec![0.0, 0.0, 0.0, 0.0]);
  assert_eq!(row(1), vec![0.0, 0.0, 0.0, 0.0]);
  // Image rows are [1,1,1,1], [2,2,2,2], [3,3,3,3] per the mock's
  // canned encoder.
  assert_eq!(row(2), vec![1.0, 1.0, 1.0, 1.0]);
  assert_eq!(row(3), vec![2.0, 2.0, 2.0, 2.0]);
  assert_eq!(row(4), vec![3.0, 3.0, 3.0, 3.0]);
  assert_eq!(row(5), vec![0.0, 0.0, 0.0, 0.0]);
  assert_eq!(row(6), vec![0.0, 0.0, 0.0, 0.0]);
}

// ─────────────────────── multi-image, concat ─────────────────────────────

#[test]
fn vlm_generate_two_images_concat_and_two_spans() {
  // Two images, 3 tokens per image, prompt marker run of len 2 → splice
  // produces 6 placeholders. spans: [(s, s+3), (s+3, s+6)]. encode
  // called twice; merge sees [6, D] image embeds.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("two_images");
  let img1 = write_test_image(&dir, "img1.png");
  let img2 = write_test_image(&dir, "img2.png");
  let prompt = [1_u32, 2, 99, 99, 3]; // marker run of len 2
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img1, img2],
    mock_cache(),
    vlm_cfg(/*max=*/ 1, /*N_per_img=*/ 3),
  )
  .expect("two-image vlm_generate constructs");
  let step = it.next().expect("step yielded").expect("step ok");
  assert_eq!(step.token, 4);

  assert_eq!(model.encode_calls.borrow().len(), 2);
  // After splice: 2 text + 6 image + 1 text = 9 positions.
  assert_eq!(model.forward_emb_calls.borrow()[0], vec![1_usize, 9, 4]);
  // Merge sees image_embeds [6, D] (3 per image × 2 images, axis-0
  // concatenated).
  let (_, image_shape, spans) = model.merge_calls.borrow()[0].clone();
  assert_eq!(image_shape, vec![6_usize, 4]);
  // Spans are per-image: (2, 5) and (5, 8) — assemble's preservation.
  assert_eq!(spans, vec![(2_usize, 5_usize), (5_usize, 8_usize)]);
}

// ─────────────────────── per-model config override ───────────────────────

#[test]
fn vlm_generate_uses_image_processor_config_override() {
  // Drive a model that overrides `image_processor_config` to a custom
  // (size, filter, color_order, mean, std) and assert the preprocess
  // step honors it (observable through the encode_image call shape:
  // the input is [H_override, W_override, 3]).
  let custom = ImageProcessorConfig::new()
    .with_size((16, 32)) // non-default (height, width)
    .with_mean([0.5, 0.5, 0.5])
    .with_std([0.5, 0.5, 0.5])
    .with_rescale_factor(1.0 / 255.0)
    .with_do_resize(true)
    .with_do_rescale(true)
    .with_do_normalize(true)
    .with_resample(ResizeFilter::Bilinear) // non-default
    .with_color_order(ColorOrder::Rgb);
  let model = MockVlmModel::new(5, 4, 3).with_processor_cfg(custom);
  let dir = temp_dir("cfg_override");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 99, 2];
  let _ = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    vlm_cfg(1, 3),
  )
  .expect("constructs")
  .next()
  .expect("step yields")
  .expect("step ok");
  // The preprocess step resized to (16, 32) so encode_image saw a
  // [16, 32, 3] array.
  assert_eq!(model.encode_calls.borrow()[0], vec![16_usize, 32, 3]);
}

// ─────────────────────── error propagation ───────────────────────────────

#[test]
fn vlm_generate_encode_failure_propagates_synchronously() {
  // fail_encode=true ⇒ the first encode_image call returns Err. That
  // surfaces as the `Err` of the `Result` returned by vlm_generate
  // (BEFORE the iterator runs — preprocess + encode are synchronous
  // in the construction path).
  let mut model = MockVlmModel::new(5, 4, 3);
  model.fail_encode = true;
  let dir = temp_dir("encode_fail");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 99, 2];
  let res = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    vlm_cfg(1, 3),
  );
  match res {
    Ok(_) => panic!("expected Err on encode failure"),
    Err(e) => {
      assert!(format!("{e}").contains("mock encode_image failure"));
    }
  }
}

#[test]
fn vlm_generate_forward_failure_yields_err_then_fuses() {
  // fail_forward=true ⇒ the first decode `forward` call returns Err.
  // The prefill_step (via forward_embeddings) runs successfully and
  // yields the first token; the SECOND iterator next() (decode) hits
  // the failing forward and yields Err. Subsequent next() return None
  // (the iterator fuses).
  let mut model = MockVlmModel::new(5, 4, 3);
  model.fail_forward = true;
  let dir = temp_dir("forward_fail");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 99, 2];
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    vlm_cfg(5, 3),
  )
  .expect("vlm_generate construction succeeds");
  // First step (prefill): uses forward_embeddings, succeeds.
  let s1 = it.next().expect("one step").expect("prefill step ok");
  assert_eq!(s1.token, 4);
  // Second step (decode): hits failing forward, yields Err.
  let s2 = it.next().expect("second step yielded");
  assert!(s2.is_err());
  // Third call: iterator fused.
  assert!(it.next().is_none());
}

// ─────────────────────── max_tokens / eos ────────────────────────────────

#[test]
fn vlm_generate_respects_max_tokens() {
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("max_tokens");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 99, 2];
  // max_tokens = 7: expect 7 tokens.
  let it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    vlm_cfg(7, 3),
  )
  .expect("vlm_generate ok");
  let mut n_tokens = 0_usize;
  for step in it {
    step.expect("step ok");
    n_tokens += 1;
  }
  assert_eq!(n_tokens, 7);
}

#[test]
fn vlm_generate_eos_stops_iteration() {
  // EOS at token 4 (the argmax of the ramp logits) ⇒ the first
  // sampled token IS eos, so the iterator yields one step then ends.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("eos");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 99, 2];
  let mut cfg = vlm_cfg(10, 3);
  cfg.lm_mut().set_eos(vec![4]); // argmax of ramp logits with vocab=5
  let it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    cfg,
  )
  .expect("vlm_generate ok");
  let tokens: Vec<u32> = it.map(|r| r.expect("step ok").token).collect();
  // EOS token IS yielded (per `lm::generate` convention), then the
  // iterator fuses.
  assert_eq!(tokens, vec![4_u32]);
}

// ─────────────────────── default merge_embeddings ────────────────────────

#[test]
fn default_merge_embeddings_rejects_wrong_image_shape() {
  // Direct unit test on `default_merge` (which calls the trait default)
  // — image_embeds [N, D] is the only accepted rank-2 shape; rank-3 is
  // rejected.
  let text = Array::from_slice::<f32>(&[0.0_f32; 5 * 4], &(1_usize, 5, 4)).unwrap();
  // [1, 3, 4] — wrong rank (3 instead of 2).
  let bad = Array::from_slice::<f32>(&[0.0_f32; 12], &(1_usize, 3, 4)).unwrap();
  let res = default_merge(&text, &bad, &[(1_usize, 4_usize)]);
  match res {
    Ok(_) => panic!("expected Err on rank-3 image_embeds"),
    Err(e) => {
      let msg = format!("{e}");
      assert!(msg.contains("rank-2"), "unexpected: {msg}");
    }
  }
}

#[test]
fn default_merge_embeddings_rejects_dim_mismatch() {
  // text_embeds D = 4, image_embeds D = 8 ⇒ LengthMismatch on hidden-dim D.
  let text = Array::from_slice::<f32>(&[0.0_f32; 5 * 4], &(1_usize, 5, 4)).unwrap();
  let img = Array::from_slice::<f32>(&[0.0_f32; 3 * 8], &(3_usize, 8_usize)).unwrap();
  let res = default_merge(&text, &img, &[(1_usize, 4_usize)]);
  match res.unwrap_err() {
    Error::LengthMismatch(p) => {
      assert!(
        p.context().contains("hidden-dim D"),
        "expected context to name hidden-dim D, got: {}",
        p.context()
      );
      assert_eq!(p.expected(), 4);
      assert_eq!(p.actual(), 8);
    }
    e => panic!("expected LengthMismatch, got: {e:?}"),
  }
}

#[test]
fn default_merge_embeddings_rejects_width_sum_mismatch() {
  // image_embeds has 3 rows, spans sum to 2 ⇒ LengthMismatch (expected =
  // caller-supplied placeholder span sum, actual = image_embeds row count).
  let text = Array::from_slice::<f32>(&[0.0_f32; 5 * 4], &(1_usize, 5, 4)).unwrap();
  let img = Array::from_slice::<f32>(&[0.0_f32; 3 * 4], &(3_usize, 4_usize)).unwrap();
  let res = default_merge(&text, &img, &[(1_usize, 3_usize)]); // width = 2, N = 3
  match res.unwrap_err() {
    Error::LengthMismatch(p) => {
      assert_eq!(
        p.expected(),
        2,
        "expected = sum of caller-supplied span widths"
      );
      assert_eq!(p.actual(), 3, "actual = image_embeds row count");
    }
    e => panic!("expected LengthMismatch, got: {e:?}"),
  }
}

#[test]
fn default_merge_embeddings_rejects_empty_spans() {
  // Empty spans + non-zero image embeds: the contract says use
  // forward(tokens) for the text-only path. Reject loudly.
  let text = Array::from_slice::<f32>(&[0.0_f32; 5 * 4], &(1_usize, 5, 4)).unwrap();
  let img = Array::from_slice::<f32>(&[0.0_f32; 3 * 4], &(3_usize, 4_usize)).unwrap();
  let res = default_merge(&text, &img, &[]);
  assert!(
    matches!(res.unwrap_err(), Error::EmptyInput(_)),
    "expected EmptyInput for empty image_spans"
  );
}

#[test]
fn default_merge_embeddings_splice_output_correct() {
  // Direct splice unit test independent of the generate pipeline.
  // text [1, 5, 2] = zeros; image [3, 2] = rows [10,10], [20,20],
  // [30,30]. spans = [(1, 4)]. Expect merged [1, 5, 2] with rows
  // pos 0 = [0,0], pos 1 = [10,10], pos 2 = [20,20], pos 3 = [30,30],
  // pos 4 = [0,0].
  let text = Array::from_slice::<f32>(&[0.0_f32; 10], &(1_usize, 5, 2)).unwrap();
  let img = Array::from_slice::<f32>(
    &[10.0_f32, 10.0, 20.0, 20.0, 30.0, 30.0],
    &(3_usize, 2_usize),
  )
  .unwrap();
  let mut merged = default_merge(&text, &img, &[(1_usize, 4_usize)]).unwrap();
  assert_eq!(merged.shape(), vec![1_usize, 5, 2]);
  let v = merged.to_vec::<f32>().unwrap();
  assert_eq!(
    v,
    vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 0.0, 0.0]
  );
}

// ─────────────────────── distinct marker vs placeholder ──────────────────

#[test]
fn vlm_generate_distinct_marker_and_placeholder_ids() {
  // image_marker_id (e.g. <|image|>) differs from image_token_id
  // (<|image_pad|>). The prompt contains the marker id; after splice
  // the spans hold the placeholder id. Verifies the marker-id pass-
  // through into `assemble_multimodal_prompt`.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("distinct_marker");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 50, 2]; // marker = 50, placeholder = 99
  let cfg = VlmGenConfig::new(
    GenConfig::default().with_max_tokens(1),
    99,
    3,
    MarkerPolicy::Required,
  )
  .with_image_marker_id(Some(50));
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    cfg,
  )
  .expect("vlm_generate ok");
  let s = it.next().expect("step").expect("ok");
  assert_eq!(s.token, 4);
  // After splice, T = 5 (1 + 3 + 1).
  assert_eq!(model.forward_emb_calls.borrow()[0], vec![1_usize, 5, 4]);
  let (_, _, spans) = model.merge_calls.borrow()[0].clone();
  assert_eq!(spans, vec![(1_usize, 4_usize)]);
}

// ─────────────────────── cache wiring observability ──────────────────────

#[test]
fn vlm_generate_advances_cache_across_prefill_and_decode() {
  // Prefill advances cache by T (here T = 7); each decode step
  // advances by 1. After 3 produced tokens (1 prefill + 2 decode),
  // every cache layer's offset is T + 2.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("cache");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 2, 99, 3, 4];
  let cache = mock_cache();
  let n_layers = cache.len();

  // We can't easily inspect the cache after the iterator runs (the
  // iterator owns it), so instead we observe the model's forward call
  // record: prefill = 1 call with embed shape [1, 7, D]; decode = 2
  // calls with token shape [1, 1].
  let it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    cache,
    vlm_cfg(3, 3),
  )
  .expect("ok");
  let n: usize = it.map(|r| r.is_ok() as usize).sum();
  assert_eq!(n, 3);
  // Per-layer cache offsets observed: 1 prefill (advances by 7) + 2
  // decode (each advances by 1) ⇒ total 9.
  // (The mock's forward call recorder reflects this indirectly: 1
  // forward_emb call and 2 forward calls.)
  assert_eq!(model.forward_emb_calls.borrow().len(), 1);
  assert_eq!(model.forward_calls.borrow().len(), 2);
  // Sanity: cache had `n_layers` layers (matches CacheConfig).
  assert_eq!(n_layers, 2);
}

// ─────────────────────── trait default config ─────────────────────────────

#[test]
fn vlm_model_image_processor_config_default_is_imagenet() {
  // Trait default returns the ImageNet baseline. A model that does
  // NOT override the default returns the same as
  // ImageProcessorConfig::default().
  struct DefaultModel;
  impl LmModel for DefaultModel {
    fn forward(&self, _t: &Array, _c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      unreachable!()
    }
  }
  impl VlmModel for DefaultModel {
    fn embed_tokens(&self, _t: &Array) -> mlxrs::Result<Array> {
      unreachable!()
    }
    fn encode_image(&self, _i: &Array) -> mlxrs::Result<Array> {
      unreachable!()
    }
  }
  let m = DefaultModel;
  let cfg = m.image_processor_config();
  let want = ImageProcessorConfig::default();
  assert_eq!(cfg, want);
}

// ─────────────── Per-image encoder shape contract ──────

#[test]
fn vlm_generate_rejects_per_image_shape_mismatch() {
  // A model whose encoder returns N != cfg.num_tokens_per_image rows
  // must be rejected synchronously — without this check, two images
  // whose row counts sum to (2 * num_tokens_per_image) would pass the
  // merge-layer total-width contract but silently mis-align image
  // features into wrong prompt spans.
  struct VariableEncoder {
    inner: MockVlmModel,
    /// Per-image row counts to return from `encode_image` (one entry
    /// per call). The test passes [2, 4] with `num_tokens_per_image=3`:
    /// the total IS 6 = 2*3 so a total-only check would accept it, but
    /// the per-image check catches it.
    counts: RefCell<Vec<usize>>,
    /// Index into `counts` (`encode_image` consumes left-to-right).
    next: RefCell<usize>,
  }
  impl LmModel for VariableEncoder {
    fn forward(&self, t: &Array, c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      self.inner.forward(t, c)
    }
    fn forward_embeddings(&self, e: &Array, c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      self.inner.forward_embeddings(e, c)
    }
  }
  impl VlmModel for VariableEncoder {
    fn embed_tokens(&self, t: &Array) -> mlxrs::Result<Array> {
      self.inner.embed_tokens(t)
    }
    fn encode_image(&self, _image: &Array) -> mlxrs::Result<Array> {
      let mut idx = self.next.borrow_mut();
      let counts = self.counts.borrow();
      let n = counts[*idx];
      *idx += 1;
      let d = 4;
      let mut data = Vec::with_capacity(n * d);
      for i in 0..n {
        for _ in 0..d {
          data.push((i + 1) as f32);
        }
      }
      Array::from_slice::<f32>(&data, &(n, d))
    }
  }

  let model = VariableEncoder {
    inner: MockVlmModel::new(5, 4, 3),
    counts: RefCell::new(vec![2_usize, 4_usize]),
    next: RefCell::new(0),
  };
  let dir = temp_dir("variable_encoder");
  let img1 = write_test_image(&dir, "img1.png");
  let img2 = write_test_image(&dir, "img2.png");
  let prompt = [1_u32, 99, 99, 2]; // marker run of len 2 → 2 images
  let res = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img1, img2],
    mock_cache(),
    vlm_cfg(/*max=*/ 1, /*N_per_img=*/ 3),
  );
  match res {
    Ok(_) => panic!("expected Err on variable per-image encoder output"),
    Err(e) => match &e {
      Error::LengthMismatch(p) => {
        assert_eq!(p.expected(), 3, "expected = cfg.num_tokens_per_image");
        assert_eq!(p.actual(), 2, "actual = encoded rows");
      }
      _ => panic!("expected LengthMismatch, got: {e:?}"),
    },
  }
}

// ─────────────── Prompt history → first-token processors ──

#[test]
fn vlm_generate_first_token_sees_prompt_history_in_logit_bias() {
  // Configure a logit_bias that gives token id 0 a massive positive
  // boost ONLY when applied; without processors active the argmax of
  // the ramp logits is `vocab - 1 = 4`. With the logit_bias applied to
  // the prefill `_step`, the argmax becomes 0.
  //
  // The previous implementation passed an
  // empty `step_inputs` to the prefill `sample_from_logits`, so the
  // `if !processors.is_empty() && !step_inputs.is_empty()` guard
  // skipped the processors entirely on the first token. Test that the
  // assembled prompt tokens now ARE the step_inputs (so processors
  // run on the prefill `_step` whenever the prompt is non-empty,
  // which is always when we reach prefill).
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("first_token_bias");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 99, 2]; // T=1+3+1=5 after splice
  let cfg = VlmGenConfig::new(
    GenConfig::default()
      .with_max_tokens(1)
      .with_logit_bias(vec![(0, 1000.0)]),
    99,
    3,
    MarkerPolicy::Required,
  );
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    cfg,
  )
  .expect("vlm_generate ok");
  let step = it.next().expect("step").expect("ok");
  // If processors had been skipped (old behavior), the argmax would
  // be 4 (vocab-1). With the fix, the logit_bias applies on the
  // prefill `_step` and shifts the argmax to 0.
  assert_eq!(
    step.token, 0,
    "first VLM token must be subject to logit_bias (regression)"
  );
}

// ── Validation precedes the vision pipeline ──────────────

#[test]
fn vlm_generate_marker_missing_errors_before_any_image_work() {
  // A malformed prompt under MarkerPolicy::Required (no marker present)
  // must surface MissingField SYNCHRONOUSLY without loading,
  // preprocessing, or encoding any images. Validation ordering must come
  // BEFORE the expensive vision pipeline.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("marker_missing_no_encode");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 2, 3]; // no marker (99)
  let res = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    vlm_cfg(/*max=*/ 1, /*N_per_img=*/ 3),
  );
  assert!(res.is_err());
  // The key assertion: NO encode_image was called.
  assert_eq!(
    model.encode_calls.borrow().len(),
    0,
    "encode_image must NOT run when prompt validation fails (regression)"
  );
  // Same for embed_tokens — the text-side pipeline starts only after
  // prompt validation has succeeded.
  assert_eq!(model.embed_calls.borrow().len(), 0);
}

#[test]
fn vlm_generate_marker_count_mismatch_errors_before_any_image_work() {
  // A marker run length that doesn't match image_count must also
  // surface before vision work — `insert_image_tokens` validates it
  // and we've reordered the call to happen first.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("marker_count_mismatch_no_encode");
  let img1 = write_test_image(&dir, "img1.png");
  let img2 = write_test_image(&dir, "img2.png");
  // Marker run of length 1 but 2 images supplied → insert_image_tokens
  // returns Err.
  let prompt = [1_u32, 99, 2];
  let res = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img1, img2],
    mock_cache(),
    vlm_cfg(1, 3),
  );
  assert!(res.is_err());
  assert_eq!(
    model.encode_calls.borrow().len(),
    0,
    "encode_image must NOT run when prompt validation fails"
  );
}

// ── Chunked prefill respects cfg.lm.prefill_step_size ──────

#[test]
fn vlm_generate_span_aware_chunking_never_splits_image_span() {
  // **Trait redesign**:
  // chunked multimodal prefill is now offset-aware AND span-aware. It
  // chunks by `prefill_step_size` but (1) never splits an image span
  // across a chunk boundary — when the natural boundary lands inside a
  // span, the chunk extends to the span end — and (2) passes
  // chunk-local spans + `cache_offset` to `forward_embeddings_multimodal`
  // so a mask builder sizes `[chunk × (past + chunk)]` correctly.
  //
  // prompt [1, 2, 99, 3, 4] with num_tokens_per_image=3 splices to
  // assembled = [1, 2, IMG, IMG, IMG, 3, 4] (T=7), span (2, 5).
  // With prefill_step_size=2 the span-aware chunk boundaries are:
  //   - [0,2): natural end=2 (span starts at 2, not split) → embed 2
  //   - [2,5): natural end=4 would split span (2<4<5) → extend to 5;
  //            holds the FULL image span → embed 3 (NOT 2)
  //   - [5,7): natural end=7 → embed 2
  // So per-chunk `forward_embeddings_multimodal` embed widths are
  // [2, 3, 2] — the middle chunk is the full image span, proving it was
  // never split. (A naive chunker would cut [2,2,2,1], splitting the
  // span at position 4.) Incremental embed/merge: each chunk embeds only
  // its own tokens, so the recorded widths are the per-chunk embed sizes.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("span_aware_chunking");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 2, 99, 3, 4]; // T = 7 after splice, span (2,5)
  let cfg = VlmGenConfig::new(
    GenConfig::default()
      .with_max_tokens(1)
      .with_prefill_step_size(2),
    99,
    3,
    MarkerPolicy::Required,
  );
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    cfg,
  )
  .expect("vlm_generate ok");
  let step = it.next().expect("step").expect("ok");
  assert_eq!(step.token, 4);

  // 3 chunks of embed widths [2, 3, 2] — the middle chunk holds the full
  // 3-token image span (never split). Peak memory bounded by chunk +
  // image features; mask coordinates correct via cache_offset.
  let calls = model.forward_emb_calls.borrow().clone();
  assert_eq!(
    calls.len(),
    3,
    "expected 3 span-aware chunks [2,3,2], got {} chunks",
    calls.len()
  );
  assert_eq!(calls[0], vec![1_usize, 2, 4]);
  assert_eq!(
    calls[1],
    vec![1_usize, 3, 4],
    "middle chunk = full image span (never split)"
  );
  assert_eq!(calls[2], vec![1_usize, 2, 4]);
}

#[test]
fn vlm_generate_threads_cache_offset_and_chunk_local_spans() {
  // Each chunk's `forward_embeddings_multimodal` must receive
  // the ABSOLUTE cache offset (initial + cursor) and CHUNK-LOCAL spans.
  // With prompt [1,2,99,3,4] (T=7, span (2,5)) at prefill_step_size=2 the
  // span-aware chunks are [0,2) [2,5) [5,7), so the captured
  // (cache_offset, chunk_local_spans) per chunk are:
  //   chunk 0: offset=0, spans=[]
  //   chunk 1: offset=2, spans=[(0,3)]   (the image span, shifted local)
  //   chunk 2: offset=5, spans=[]
  // (The cache starts empty so initial_offset=0; the offsets equal the
  // cursors.)
  //
  // One captured prefill chunk — named (vs a bare nested
  // `(usize, Vec<(usize, usize)>)` tuple) for readability.
  #[derive(Debug, PartialEq)]
  struct ChunkCapture {
    /// The absolute cache offset the chunk was forwarded at.
    cache_offset: usize,
    /// The chunk-local image spans the chunk received.
    spans: ImageSpans,
  }
  struct OffsetCapturingModel {
    inner: MockVlmModel,
    captured: RefCell<Vec<ChunkCapture>>,
  }
  impl LmModel for OffsetCapturingModel {
    fn forward(&self, t: &Array, c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      self.inner.forward(t, c)
    }
    fn forward_embeddings(&self, e: &Array, c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      self.inner.forward_embeddings(e, c)
    }
  }
  impl VlmModel for OffsetCapturingModel {
    fn embed_tokens(&self, t: &Array) -> mlxrs::Result<Array> {
      self.inner.embed_tokens(t)
    }
    fn encode_image(&self, i: &Array) -> mlxrs::Result<Array> {
      self.inner.encode_image(i)
    }
    fn forward_embeddings_multimodal(
      &self,
      embeddings: &Array,
      image_spans: &[(usize, usize)],
      cache_offset: usize,
      cache: &mut [Box<dyn KvCache>],
    ) -> mlxrs::Result<Array> {
      self.captured.borrow_mut().push(ChunkCapture {
        cache_offset,
        spans: image_spans.to_vec(),
      });
      LmModel::forward_embeddings(self, embeddings, cache)
    }
  }

  let model = OffsetCapturingModel {
    inner: MockVlmModel::new(5, 4, 3),
    captured: RefCell::new(Vec::new()),
  };
  let dir = temp_dir("offset_capture");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 2, 99, 3, 4];
  let cfg = VlmGenConfig::new(
    GenConfig::default()
      .with_max_tokens(1)
      .with_prefill_step_size(2),
    99,
    3,
    MarkerPolicy::Required,
  );
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    cfg,
  )
  .expect("vlm_generate ok");
  it.next().expect("step").expect("ok");

  let cap = model.captured.borrow();
  assert_eq!(cap.len(), 3, "expected 3 chunks");
  assert_eq!(
    cap[0],
    ChunkCapture {
      cache_offset: 0,
      spans: vec![]
    },
    "chunk 0: offset 0, no spans"
  );
  assert_eq!(
    cap[1],
    ChunkCapture {
      cache_offset: 2,
      spans: vec![(0, 3)]
    },
    "chunk 1: offset 2, chunk-local span (0,3) = the full image"
  );
  assert_eq!(
    cap[2],
    ChunkCapture {
      cache_offset: 5,
      spans: vec![]
    },
    "chunk 2: offset 5, no spans"
  );
}

// (Pure-text-through-vlm_generate chunking path: covered indirectly by
// `lm::generate`'s own chunked-prefill tests — the VLM path's chunking
// loop is feature-identical to lm::generate's when image_spans is empty.
// A direct vlm_generate(&[]) pure-text test is awkward because the
// marker insertion / splice logic requires at least one image for the
// per-image-token expansion to be coherent; constructing a contrived
// "0-images-but-marker-prepended" prompt only exercises the splice
// short-circuit and adds little signal over the image-prompt test
// above + lm::generate's existing chunked-prefill coverage.)

#[test]
fn vlm_generate_single_chunk_when_prefill_step_size_ge_t() {
  // When `prefill_step_size >= T`, the prefill runs ONE chunk over
  // the full merged sequence (no slice allocation; pass merged by
  // reference). Pin the fast path so it doesn't regress.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("single_chunk");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 2, 99, 3, 4]; // T = 7 after splice
  let cfg = VlmGenConfig::new(
    GenConfig::default()
      .with_max_tokens(1)
      .with_prefill_step_size(100),
    99,
    3,
    MarkerPolicy::Required,
  );
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    cfg,
  )
  .expect("vlm_generate ok");
  it.next().expect("step").expect("ok");
  let calls = model.forward_emb_calls.borrow().clone();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0], vec![1_usize, 7, 4]);
}

#[test]
fn vlm_generate_max_tokens_zero_does_no_image_work() {
  // **Bundle #62**: `max_tokens == 0` must
  // yield an empty iterator and do ZERO model/vision work — mirroring
  // the LM-side contract where `lm::generate`'s iterator checks
  // `produced >= max_tokens` BEFORE prefill. The VLM multimodal path
  // does its vision pipeline (load / preprocess / encode_image /
  // embed_tokens / merge) eagerly at construction, so without the
  // top-of-function short-circuit a zero-output request would still
  // trigger image I/O + vision compute. Assert NO encode / embed /
  // merge / forward call was recorded.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("max_tokens_zero");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 2, 99, 3, 4];
  let cfg = VlmGenConfig::new(
    GenConfig::default().with_max_tokens(0),
    99,
    3,
    MarkerPolicy::Required,
  );
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    cfg,
  )
  .expect("vlm_generate ok");
  assert!(
    it.next().is_none(),
    "max_tokens=0 must yield an empty iterator"
  );

  // No vision or model work happened at construction.
  assert_eq!(
    model.encode_calls.borrow().len(),
    0,
    "no encode_image calls"
  );
  assert_eq!(model.embed_calls.borrow().len(), 0, "no embed_tokens calls");
  assert_eq!(
    model.merge_calls.borrow().len(),
    0,
    "no merge_embeddings calls"
  );
  assert_eq!(
    model.forward_emb_calls.borrow().len(),
    0,
    "no forward_embeddings calls"
  );
  assert_eq!(model.forward_calls.borrow().len(), 0, "no forward calls");
}

// ─── Mask handoff via per-request spans, not &self ──

#[test]
fn vlm_generate_threads_per_request_spans_no_cross_iterator_pollution() {
  // The mask-requiring path: each iterator's
  // `prefill_step` calls `forward_embeddings_multimodal(embeds, spans,
  // cache)` with THIS iterator's spans — never via &self state. Two
  // iterators with DIFFERENT spans constructed against the same model
  // and polled out of order must each see their own spans.
  //
  // Mock model records every spans tuple it observes on
  // `forward_embeddings_multimodal`. Constructing two iterators (with
  // distinct prompts → distinct spans) BEFORE polling either, then
  // polling iterator B first, then iterator A, must show:
  //   - capture[0] = B's spans (B polled first)
  //   - capture[1] = A's spans (A polled second)
  // If spans were stored on &self, the second iterator's construction
  // would clobber the first iterator's spans and capture[1] would
  // equal capture[0] (B's spans).
  struct SpanCapturingModel {
    inner: MockVlmModel,
    captured_spans: RefCell<Vec<ImageSpans>>,
  }
  impl LmModel for SpanCapturingModel {
    fn forward(&self, t: &Array, c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      self.inner.forward(t, c)
    }
    fn forward_embeddings(&self, e: &Array, c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
      self.inner.forward_embeddings(e, c)
    }
  }
  impl VlmModel for SpanCapturingModel {
    fn embed_tokens(&self, t: &Array) -> mlxrs::Result<Array> {
      self.inner.embed_tokens(t)
    }
    fn encode_image(&self, i: &Array) -> mlxrs::Result<Array> {
      self.inner.encode_image(i)
    }
    fn forward_embeddings_multimodal(
      &self,
      embeddings: &Array,
      image_spans: &[(usize, usize)],
      _cache_offset: usize,
      cache: &mut [Box<dyn KvCache>],
    ) -> mlxrs::Result<Array> {
      // Capture the spans this call sees. With prefill_step_size >= T
      // (the default) each prompt prefills as a single chunk at
      // cache_offset 0, so the chunk-local spans equal the absolute
      // spans — A sees [(0,3)], B sees [(2,5)].
      self.captured_spans.borrow_mut().push(image_spans.to_vec());
      // Default dispatch (the trait default body, since this override
      // calls the LM's `forward_embeddings`).
      LmModel::forward_embeddings(self, embeddings, cache)
    }
  }

  let model = SpanCapturingModel {
    inner: MockVlmModel::new(5, 4, 3),
    captured_spans: RefCell::new(Vec::new()),
  };
  let dir = temp_dir("cross_iter");
  let img_a = write_test_image(&dir, "imga.png");
  let img_b = write_test_image(&dir, "imgb.png");
  // Prompt A: marker at position 0 → spans [(0, 3)] (T = 3).
  let prompt_a = [99_u32, 1, 2];
  // Prompt B: marker at position 2 → spans [(2, 5)] (T = 5).
  let prompt_b = [1_u32, 2, 99, 3, 4];

  let mut it_a = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt_a,
    &[img_a],
    mock_cache(),
    vlm_cfg(/*max=*/ 1, /*N_per_img=*/ 3),
  )
  .expect("iter A constructs");
  let mut it_b = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt_b,
    &[img_b],
    mock_cache(),
    vlm_cfg(/*max=*/ 1, /*N_per_img=*/ 3),
  )
  .expect("iter B constructs");

  // Poll B first.
  it_b.next().expect("step B").expect("ok");
  // Then poll A.
  it_a.next().expect("step A").expect("ok");

  let cap = model.captured_spans.borrow();
  assert_eq!(cap.len(), 2, "expected two prefill captures");
  // B was polled first → cap[0] = B's spans = [(2, 5)].
  assert_eq!(
    cap[0],
    vec![(2_usize, 5_usize)],
    "iter B (polled first) should see its OWN spans"
  );
  // A was polled second → cap[1] = A's spans = [(0, 3)].
  // If spans were stored on &self, cap[1] would equal cap[0] (B's
  // spans clobbered A's at B's construction).
  assert_eq!(
    cap[1],
    vec![(0_usize, 3_usize)],
    "iter A (polled second) must see its OWN spans, not B's"
  );
}

#[test]
fn vlm_generate_forward_embeddings_multimodal_default_dispatches_to_lm() {
  // The trait default for `forward_embeddings_multimodal` dispatches to
  // the LM's `forward_embeddings` and ignores `image_spans` — pin this
  // so a future refactor can't silently break the default contract.
  // Using the unmodified MockVlmModel (no override): the prefill must
  // hit `forward_embeddings` (the LM seam) just like before this PR.
  let model = MockVlmModel::new(5, 4, 3);
  let dir = temp_dir("default_dispatch");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 99, 2];
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    vlm_cfg(1, 3),
  )
  .expect("vlm_generate ok");
  it.next().expect("step").expect("ok");
  // The default dispatch goes through the LM's `forward_embeddings`
  // (observed via `forward_emb_calls`); the mock never overrides
  // `forward_embeddings_multimodal`, so this counter MUST advance.
  assert_eq!(model.forward_emb_calls.borrow().len(), 1);
}

// ────────────────────── Attention mask elided ──────────

#[test]
fn vlm_generate_does_not_build_unused_attention_mask() {
  // The previous implementation called `assemble_multimodal_prompt`
  // which builds an O(T*T) attention mask — but the mask was never
  // threaded to `forward_embeddings`. The fix is
  // to call `insert_image_tokens` directly (no mask construction in
  // the hot path).
  //
  // We can't directly observe "mask was not built" from the public
  // API, but we CAN observe that the iterator completes successfully
  // for a moderately large T where `T*T` would be a large mask (and
  // the operation would be visible in any allocation/timing trace).
  // The semantic correctness is: the iterator yields tokens normally
  // with the same per-step behavior as before — which all the other
  // tests already cover. This test pins the NEGATIVE behavior: a
  // single image with `num_tokens_per_image = 100` (so T = 102 and
  // T*T = 10404 mask elements would be wasted under the old path)
  // still produces correct generation through the trait surface.
  let model = MockVlmModel::new(5, 4, /*N_per_img=*/ 100);
  let dir = temp_dir("no_unused_mask");
  let img = write_test_image(&dir, "img.png");
  let prompt = [1_u32, 99, 2]; // T = 1 + 100 + 1 = 102 after splice
  let cfg = VlmGenConfig::new(
    GenConfig::default().with_max_tokens(1),
    99,
    100,
    MarkerPolicy::Required,
  );
  let mut it = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    cfg,
  )
  .expect("vlm_generate ok");
  let step = it.next().expect("step").expect("ok");
  assert_eq!(step.token, 4);
  // Sanity: prefill embed shape was [1, 102, 4].
  assert_eq!(model.forward_emb_calls.borrow()[0], vec![1_usize, 102, 4]);
}

// ────────────────────── #136: eager cfg.lm.validate ─────────────

/// `vlm_generate` MUST call `cfg.lm.validate()` at the TOP of the function
/// — BEFORE the `max_tokens == 0` short-circuit, BEFORE the zero-image /
/// multimodal split, and (most importantly) BEFORE any image
/// load / preprocess / encode_image call. An invalid `cfg.lm` (NaN
/// `logit_bias` here) must surface as the synchronous `Err` returned by
/// `vlm_generate` itself — no Iterator construction, no vision pipeline,
/// no model trait calls. The mock's call-count assertions prove the
/// vision pipeline never ran; the nonexistent image path proves
/// `load_image` was never reached (a `load_image` call on a missing path
/// would surface a different "no such file" error).
#[test]
fn vlm_generate_rejects_invalid_lm_config_before_image_load() {
  let model = MockVlmModel::new(/*vocab=*/ 5, /*D=*/ 4, /*N_per_img=*/ 3);
  // Deliberately nonexistent path: if validation regressed and image
  // load ran, this would surface as a "no such file" Err rather than
  // the validation Err we expect — proving the validate gate ran first.
  let bogus_img = PathBuf::from("/nonexistent/vlm_validate_test/image_that_does_not_exist.png");
  let prompt = [1_u32, 2, 99, 3, 4]; // valid prompt (has marker)
  let cfg = VlmGenConfig::new(
    GenConfig::default()
      .with_max_tokens(3)
      .with_logit_bias(vec![(0, 1.0), (1, f32::NAN)]),
    99,
    3,
    MarkerPolicy::Required,
  );
  let res = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[bogus_img],
    mock_cache(),
    cfg,
  );
  let err = match res {
    Ok(_) => panic!("vlm_generate must return Err for an invalid cfg.lm (NaN logit_bias)"),
    Err(e) => e,
  };
  let msg = format!("{err}");
  assert!(
    msg.contains("logit_bias"),
    "vlm_generate surfaced an error that does not reference logit_bias — \
     validate fail-fast may have regressed (got: {msg})"
  );
  assert!(
    !msg.contains("nonexistent")
      && !msg.contains("No such file")
      && !msg.contains("image_that_does_not_exist"),
    "vlm_generate reached image load before validate — fail-fast regression (got: {msg})"
  );

  // Mock-trait call counters: every VLM-side method MUST be unobserved.
  assert!(
    model.encode_calls.borrow().is_empty(),
    "encode_image was called {} time(s); validate gate did not fail-fast",
    model.encode_calls.borrow().len(),
  );
  assert!(
    model.embed_calls.borrow().is_empty(),
    "embed_tokens was called; validate gate did not fail-fast",
  );
  assert!(
    model.merge_calls.borrow().is_empty(),
    "merge_embeddings was called; validate gate did not fail-fast",
  );
  assert!(
    model.forward_emb_calls.borrow().is_empty(),
    "forward_embeddings was called; validate gate did not fail-fast",
  );
  assert!(
    model.forward_calls.borrow().is_empty(),
    "forward was called; validate gate did not fail-fast",
  );
}

/// Companion: invalid `cfg.lm` MUST be rejected even with `images=[]`
/// — the zero-image branch in `vlm_generate` delegates to
/// `lm::generate::generate_step` (which validates internally), so this
/// branch was already covered before #136. The eager validate at the
/// top of `vlm_generate` collapses BOTH branches to the same surface:
/// the `vlm_generate` `Result` is synchronously `Err` on a bad cfg,
/// without entering the Iterator construction at all.
#[test]
fn vlm_generate_rejects_invalid_lm_config_no_images() {
  let model = MockVlmModel::new(/*vocab=*/ 5, /*D=*/ 4, /*N_per_img=*/ 3);
  let prompt = [1_u32, 2, 3];
  let cfg = VlmGenConfig::new(
    GenConfig::default().with_max_tokens(2).with_temp(-1.0),
    99,
    3,
    MarkerPolicy::Required,
  );
  let res = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[],
    mock_cache(),
    cfg,
  );
  match res {
    Ok(_) => panic!("vlm_generate must return Err for an invalid cfg.lm (negative temp)"),
    Err(e) => {
      let msg = format!("{e}");
      assert!(
        msg.contains("temp"),
        "expected validation error referencing `temp`, got: {msg}"
      );
    }
  }
  // No model trait method should have been called either.
  assert!(model.forward_calls.borrow().is_empty());
  assert!(model.forward_emb_calls.borrow().is_empty());
  assert!(model.encode_calls.borrow().is_empty());
  assert!(model.embed_calls.borrow().is_empty());
  assert!(model.merge_calls.borrow().is_empty());
}

/// Companion: invalid `cfg.lm` MUST be rejected EVEN with
/// `cfg.lm.max_tokens == 0` (the zero-budget short-circuit). Previously
/// the `max_tokens == 0` branch returned an empty Iterator unconditionally,
/// meaning a bad cfg silently produced an empty result instead of erroring.
/// With the eager validate at the top of `vlm_generate`, the surface is
/// uniformly "bad cfg = Err" regardless of `max_tokens`.
#[test]
fn vlm_generate_rejects_invalid_lm_config_under_zero_max_tokens() {
  let model = MockVlmModel::new(/*vocab=*/ 5, /*D=*/ 4, /*N_per_img=*/ 3);
  let prompt = [1_u32, 2, 99, 3, 4];
  let cfg = VlmGenConfig::new(
    GenConfig::default()
      .with_max_tokens(0)
      .with_logit_bias(vec![(0, f32::INFINITY)]),
    99,
    3,
    MarkerPolicy::Required,
  );
  let dir = temp_dir("validate_zero_max");
  let img = write_test_image(&dir, "img.png");
  let res = vlm_generate(
    &model,
    &model.image_processor_config(),
    &prompt,
    &[img],
    mock_cache(),
    cfg,
  );
  match res {
    Ok(_) => panic!(
      "vlm_generate(max_tokens=0) must STILL return Err for an invalid cfg.lm, \
       not silently return an empty iterator"
    ),
    Err(e) => {
      let msg = format!("{e}");
      assert!(
        msg.contains("logit_bias"),
        "expected validation error referencing logit_bias, got: {msg}"
      );
    }
  }
}
