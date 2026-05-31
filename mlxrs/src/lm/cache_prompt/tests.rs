//! In-crate prefill-boundary unit tests for the driver, sharing the
//! `model::MockModel` fixture (`#[cfg(test)] pub(crate)`, visible here).
//! The cross-tool round-trip (fill → save → load → continue) lives in the
//! integration test `tests/lm_cache_prompt_driver.rs`, which exercises the
//! public `save_prompt_cache`/`load_prompt_cache`.

use std::{cell::Cell, rc::Rc};

use super::*;
use crate::lm::{
  cache::{MaskMode, RotatingKvCache, StandardKvCache},
  model::MockModel,
};

/// A [`CacheConfig`] for `layers` full-attention (non-sliding-window)
/// decoder layers — what the driver allocates a `StandardKvCache` per.
fn config(layers: usize) -> CacheConfig {
  CacheConfig {
    num_hidden_layers: layers,
    sliding_window: None,
  }
}

/// A freshly built per-layer cache for `layers` full-attention layers —
/// used by the `prefill_full` boundary tests, which drive the cache
/// directly (the public `cache_prompt_ids` allocates its own internally).
fn cache(layers: usize) -> Vec<Box<dyn KvCache>> {
  make_prompt_cache(&config(layers))
}

/// A [`KvCache`] that delegates to a [`StandardKvCache`] but counts every
/// [`materialize`](KvCache::materialize) call into a shared counter — used
/// to observe the per-chunk [`materialize_caches`] barrier firing during
/// prefill (the barrier calls `materialize()` once per layer per chunk).
struct CountingCache {
  inner: StandardKvCache,
  materialize_calls: Rc<Cell<usize>>,
}

impl KvCache for CountingCache {
  fn offset(&self) -> usize {
    self.inner.offset()
  }
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    self.inner.update(keys, values)
  }
  fn state(&self) -> Result<Vec<Array>> {
    self.inner.state()
  }
  fn materialize(&mut self) -> Result<()> {
    self.materialize_calls.set(self.materialize_calls.get() + 1);
    self.inner.materialize()
  }
  fn set_state(&mut self, state: Vec<Array>) -> Result<()> {
    self.inner.set_state(state)
  }
  fn make_mask(&self, n: usize, w: Option<usize>, ret: bool) -> Result<MaskMode> {
    self.inner.make_mask(n, w, ret)
  }
  fn nbytes(&self) -> usize {
    self.inner.nbytes()
  }
  fn is_empty(&self) -> bool {
    self.inner.is_empty()
  }
  fn copy(&self) -> Result<Box<dyn KvCache>> {
    self.inner.copy()
  }
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
  // `reference_class_name` is REQUIRED (no default) — forward
  // to the wrapped cache so persistence/dispatch sees the inner's name.
  fn reference_class_name(&self) -> &'static str {
    self.inner.reference_class_name()
  }
}

/// `prefill_full` advances every layer's cache to exactly the prompt length
/// (offset `P`), regardless of the chunk size — the whole prompt lands in
/// the cache (the `generate_step(max_tokens=0)` fill contract).
#[test]
fn prefill_full_fills_cache_to_prompt_len() {
  let model = MockModel::new(5);
  let prompt = [1u32, 2, 3, 4, 5, 6, 7];
  // A small chunk so the leading P-1 loop runs multiple chunks + the tail.
  let mut c = cache(2);
  prefill_full(&model, &prompt, &mut c, 3).unwrap();
  assert!(
    c.iter().all(|x| x.offset() == prompt.len()),
    "every layer cache must be at offset P after a full prefill"
  );
}

/// The chunk boundaries are byte-identical to mlx-lm's
/// `generate_step(max_tokens=0)`: the leading `P-1` tokens in
/// `prefill_step_size` chunks, then a final 1-token forward. A
/// seq-len-recording model pins the exact `forward` window sequence.
#[test]
fn prefill_full_chunk_boundaries_match_reference() {
  use std::cell::RefCell;
  struct Recorder {
    bias: Vec<f32>,
    seq_lens: RefCell<Vec<usize>>,
  }
  impl Model for Recorder {
    fn forward(&self, tokens: &Array, _c: &mut [Box<dyn KvCache>]) -> Result<Array> {
      let s = match tokens.shape().as_slice() {
        [_, s] => *s,
        [s] => *s,
        other => {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "Recorder::forward: tokens must be rank-1 [S] or rank-2 [B, S]",
            other.len() as u32,
            other.to_vec(),
          )));
        }
      };
      self.seq_lens.borrow_mut().push(s);
      let vocab = self.bias.len();
      let mut data = Vec::with_capacity(s * vocab);
      for _ in 0..s {
        data.extend_from_slice(&self.bias);
      }
      Array::from_slice::<f32>(&data, &(1usize, s, vocab))
    }
  }
  let model = Recorder {
    bias: vec![0.0, 1.0, 2.0],
    seq_lens: RefCell::new(Vec::new()),
  };
  // P = 7, step = 3 ⇒ leading P-1 = 6 tokens as chunks [3, 3], then the
  // final token as [1]. Exactly mlx-lm's prefill loop + first `_step`.
  let prompt = [1u32, 2, 1, 2, 1, 2, 1];
  let mut c: Vec<Box<dyn KvCache>> = Vec::new();
  prefill_full(&model, &prompt, &mut c, 3).unwrap();
  assert_eq!(model.seq_lens.into_inner(), vec![3, 3, 1]);
}

/// A single-token prompt: the leading-`P-1` loop never runs (0 tokens), and
/// only the final 1-token forward fires — cache ends at offset 1. Uses the
/// cache-advancing [`MockModel`] (`forward` updates each layer) so the
/// offset is observable; the count side is pinned by the dedicated
/// chunk-boundary test above.
#[test]
fn prefill_full_single_token_prompt() {
  let model = MockModel::new(4);
  let mut c = cache(1);
  prefill_full(&model, &[42u32], &mut c, 8).unwrap();
  assert!(
    c.iter().all(|x| x.offset() == 1),
    "a 1-token prompt fills the cache to offset 1 via the single tail forward"
  );
}

/// The per-chunk cache materialization barrier (mlx-lm `generate.py:442`)
/// fires once per layer per leading chunk on a multi-chunk prompt (`P >
/// step`): the lazy graph is materialized between chunks, not accumulated to
/// the save. A [`CountingCache`] counts `materialize()` calls during
/// prefill; with `P = 7`, `step = 2` the leading `P-1 = 6` tokens are 3
/// chunks `[2,2,2]`, so the barrier runs **3 times** (> 1 chunk) — proving
/// the prefill is memory-bounded (the graph never spans the whole prompt).
#[test]
fn prefill_full_materializes_caches_per_chunk() {
  let model = MockModel::new(5);
  let counter = Rc::new(Cell::new(0usize));
  let mut c: Vec<Box<dyn KvCache>> = vec![Box::new(CountingCache {
    inner: StandardKvCache::new(),
    materialize_calls: Rc::clone(&counter),
  })];
  // P = 7, step = 2 ⇒ leading 6 tokens as [2,2,2] ⇒ 3 barrier calls.
  let prompt = [1u32, 2, 3, 4, 5, 6, 7];
  prefill_full(&model, &prompt, &mut c, 2).unwrap();
  assert_eq!(
    counter.get(),
    3,
    "the per-chunk materialize barrier must fire once per leading chunk (3 chunks for P=7, step=2)"
  );
  assert!(
    counter.get() > 1,
    "a multi-chunk prefill runs the barrier more than once"
  );
  // The cache still ends at offset P (the tail forward ran after the loop).
  assert!(c.iter().all(|x| x.offset() == prompt.len()));
}

/// A single-chunk prompt (`P - 1 <= step`) runs the barrier exactly once
/// (the lone leading chunk); a `P == 1` prompt (no leading chunk) never
/// enters the loop, so the barrier does not fire — both still leave the
/// cache at offset `P` via the tail forward.
#[test]
fn prefill_full_barrier_count_matches_chunking() {
  // P = 4, step = 8 ⇒ leading 3 tokens in ONE chunk ⇒ 1 barrier call.
  let model = MockModel::new(5);
  let one = Rc::new(Cell::new(0usize));
  let mut c1: Vec<Box<dyn KvCache>> = vec![Box::new(CountingCache {
    inner: StandardKvCache::new(),
    materialize_calls: Rc::clone(&one),
  })];
  prefill_full(&model, &[1u32, 2, 3, 4], &mut c1, 8).unwrap();
  assert_eq!(one.get(), 1, "a single leading chunk runs the barrier once");

  // P = 1 ⇒ no leading chunk ⇒ 0 barrier calls (only the tail forward).
  let zero = Rc::new(Cell::new(0usize));
  let mut c0: Vec<Box<dyn KvCache>> = vec![Box::new(CountingCache {
    inner: StandardKvCache::new(),
    materialize_calls: Rc::clone(&zero),
  })];
  prefill_full(&model, &[42u32], &mut c0, 8).unwrap();
  assert_eq!(
    zero.get(),
    0,
    "a 1-token prompt has no leading chunk, so no barrier"
  );
  assert!(c0.iter().all(|x| x.offset() == 1));
}

/// A `prefill_step_size` of `0` is clamped to `1` (still makes progress),
/// mirroring `generate_step`'s `prefill_step_size.max(1)`.
#[test]
fn prefill_full_zero_step_is_clamped() {
  let model = MockModel::new(4);
  let mut c = cache(1);
  prefill_full(&model, &[1u32, 2, 3], &mut c, 0).unwrap();
  assert!(c.iter().all(|x| x.offset() == 3));
}

/// The per-chunk barrier routes through [`KvCache::materialize`] — and on a
/// [`RotatingKvCache`] whose ring buffer has *over-allocated* (`offset <
/// buffer_len`, the regime an `S == 1` `prefill_step_size == 1` update grows
/// the ring into) it must materialize the genuine stored ring buffers, not
/// the offset-length `state()` serialization slices.
/// This wraps a `RotatingKvCache` and records, on each `materialize()` call,
/// whether the full stored ring (`nbytes()`) exceeded the offset-length
/// serialized `state()` — i.e. the over-allocated regime — proving the
/// barrier ran on exactly the slice-view-diverging state the hook targets.
#[test]
fn prefill_full_materializes_rotating_live_ring_buffers() {
  /// Byte size of a cache's serialized `state()` arrays (`size * 4` for the
  /// f32 K/V here) — the *logical* (offset-length) size.
  fn state_nbytes(c: &dyn KvCache) -> usize {
    c.state().unwrap().iter().map(|a| a.size() * 4).sum()
  }

  struct ObservingRotatingCache {
    inner: RotatingKvCache,
    materialize_calls: Rc<Cell<usize>>,
    saw_overallocated_buffer: Rc<Cell<bool>>,
  }

  impl KvCache for ObservingRotatingCache {
    fn offset(&self) -> usize {
      self.inner.offset()
    }
    fn max_size(&self) -> Option<usize> {
      self.inner.max_size()
    }
    fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
      self.inner.update(keys, values)
    }
    fn state(&self) -> Result<Vec<Array>> {
      self.inner.state()
    }
    fn set_state(&mut self, state: Vec<Array>) -> Result<()> {
      self.inner.set_state(state)
    }
    fn materialize(&mut self) -> Result<()> {
      self.materialize_calls.set(self.materialize_calls.get() + 1);
      // Full stored ring buffer (`nbytes`) vs the offset-length serialized
      // state: `>` ⇔ the ring over-allocated ⇔ `state()` is returning slice
      // views, the regime the barrier must materialize the live buffers for.
      if self.inner.nbytes() > state_nbytes(&self.inner) {
        self.saw_overallocated_buffer.set(true);
      }
      self.inner.materialize()
    }
    fn make_mask(&self, n: usize, w: Option<usize>, ret: bool) -> Result<MaskMode> {
      self.inner.make_mask(n, w, ret)
    }
    fn nbytes(&self) -> usize {
      self.inner.nbytes()
    }
    fn is_empty(&self) -> bool {
      self.inner.is_empty()
    }
    fn copy(&self) -> Result<Box<dyn KvCache>> {
      self.inner.copy()
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
      self
    }
    // `reference_class_name` is REQUIRED (no default) — forward
    // to the wrapped rotating cache so persistence/dispatch sees the
    // inner's name.
    fn reference_class_name(&self) -> &'static str {
      self.inner.reference_class_name()
    }
  }

  let model = MockModel::new(8);
  // P = 9 tokens; step = 1 ⇒ leading P-1 = 8 single-token chunks ⇒ 8
  // barrier calls; each S==1 update grows the ring (window 4 << buffer
  // step 256), so the buffer over-allocates.
  let prompt: Vec<u32> = (0..9u32).map(|i| i % 7).collect();
  let materialize_calls = Rc::new(Cell::new(0usize));
  let saw_over = Rc::new(Cell::new(false));
  let mut c: Vec<Box<dyn KvCache>> = vec![Box::new(ObservingRotatingCache {
    inner: RotatingKvCache::new(4, 2),
    materialize_calls: Rc::clone(&materialize_calls),
    saw_overallocated_buffer: Rc::clone(&saw_over),
  })];

  prefill_full(&model, &prompt, &mut c, 1).unwrap();

  assert!(c.iter().all(|x| x.offset() == prompt.len()));
  assert_eq!(
    materialize_calls.get(),
    prompt.len() - 1,
    "the per-chunk materialize barrier must fire once per leading single-token chunk"
  );
  assert!(
    saw_over.get(),
    "the rotating ring must over-allocate during step==1 prefill, so the barrier \
       is exercised on the slice-view-diverging regime the fix targets"
  );
}

/// `cache_prompt_ids` over an empty prompt is a recoverable `Err` and
/// writes no file (faithful to mlx-lm's empty-prompt `ValueError`).
#[test]
fn cache_prompt_ids_empty_prompt_errors_without_writing() {
  let model = MockModel::new(4);
  let dir = std::env::temp_dir().join(format!("mlxrs_cache_prompt_empty_{}", std::process::id()));
  let _ = std::fs::create_dir_all(&dir);
  let out = dir.join("empty.safetensors");
  let _ = std::fs::remove_file(&out);
  let r = cache_prompt_ids(
    &model,
    &[],
    &config(1),
    &out,
    "mock",
    "{}",
    8,
    &HashMap::new(),
  );
  assert!(r.is_err(), "empty prompt must error");
  assert!(
    !out.exists(),
    "no cache file written on the empty-prompt error"
  );
}

/// `cache_prompt_ids` allocates its KV cache **internally** (via
/// `make_prompt_cache`, exactly cache_prompt.py:111) — there is no
/// caller-provided cache parameter. Each call therefore starts from a
/// fresh cache, so the saved cache represents *exactly* the requested
/// prompt: two back-to-back runs over different prompts each persist a
/// cache at that run's own prompt length (a cross-request leak — the old
/// caller-cache hazard — is structurally impossible here, since there is
/// no cache object to reuse). The `tokens_processed` count and the saved
/// cache offset match the prompt length on *every* run.
#[test]
fn cache_prompt_ids_allocates_a_fresh_cache_per_call() {
  let model = MockModel::new(8);
  let dir = std::env::temp_dir().join(format!("mlxrs_cache_prompt_fresh_{}", std::process::id()));
  let _ = std::fs::create_dir_all(&dir);

  // Run 1: a 4-token prompt.
  let out1 = dir.join("run1.safetensors");
  let _ = std::fs::remove_file(&out1);
  let info1 = cache_prompt_ids(
    &model,
    &[1u32, 2, 3, 4],
    &config(2),
    &out1,
    "mock",
    "{}",
    8,
    &HashMap::new(),
  )
  .expect("run 1 must prefill + save successfully");
  assert_eq!(info1.tokens_processed, 4);
  assert!(out1.exists(), "run 1 must write the cache file");
  let (loaded1, _m1) = crate::lm::cache::load_prompt_cache(&out1).unwrap();
  assert!(
    loaded1.iter().all(|c| c.offset() == 4),
    "run 1's saved cache is exactly its 4-token prompt"
  );

  // Run 2: a *shorter* 2-token prompt. Because the cache is freshly
  // allocated inside `cache_prompt_ids`, run 2 cannot inherit run 1's
  // state — its saved cache is exactly 2 tokens, not 4 + 2.
  let out2 = dir.join("run2.safetensors");
  let _ = std::fs::remove_file(&out2);
  let info2 = cache_prompt_ids(
    &model,
    &[5u32, 6],
    &config(2),
    &out2,
    "mock",
    "{}",
    8,
    &HashMap::new(),
  )
  .expect("run 2 must prefill + save successfully");
  assert_eq!(
    info2.tokens_processed, 2,
    "run 2 processes only its own 2-token prompt"
  );
  let (loaded2, _m2) = crate::lm::cache::load_prompt_cache(&out2).unwrap();
  assert!(
    loaded2.iter().all(|c| c.offset() == 2),
    "run 2's saved cache represents exactly its 2-token prompt — no leaked \
       prior-request context (the internally-allocated cache is fresh)"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// A sliding-window `cache_config` (non-`None` `sliding_window`) makes
/// `cache_prompt_ids` allocate a [`crate::lm::cache::RotatingKvCache`] per
/// layer — the model-appropriate cache `make_prompt_cache` builds — and the
/// prefill + save still round-trips at offset `P`.
#[test]
fn cache_prompt_ids_sliding_window_config_uses_rotating_cache() {
  let model = MockModel::new(8);
  let dir = std::env::temp_dir().join(format!("mlxrs_cache_prompt_sliding_{}", std::process::id()));
  let _ = std::fs::create_dir_all(&dir);
  let out = dir.join("sliding.safetensors");
  let _ = std::fs::remove_file(&out);

  let cfg = CacheConfig {
    num_hidden_layers: 2,
    sliding_window: Some(8),
  };
  let info = cache_prompt_ids(
    &model,
    &[1u32, 2, 3, 4, 5],
    &cfg,
    &out,
    "mock",
    "{}",
    2,
    &HashMap::new(),
  )
  .expect("a sliding-window config must prefill + save successfully");
  assert_eq!(info.tokens_processed, 5);

  let (loaded, _m) = crate::lm::cache::load_prompt_cache(&out).unwrap();
  assert!(loaded.iter().all(|c| c.offset() == 5));
  assert!(
    loaded
      .iter()
      .all(|c| c.reference_class_name() == "RotatingKVCache"),
    "a sliding-window config allocates RotatingKVCache layers"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// [`effective_safetensors_path`] mirrors mlx core's extension-append: a
/// path already ending in `.safetensors` is unchanged; any other path gets
/// `.safetensors` appended (so the atomic rename target == the path mlx
/// would write for a direct save).
#[test]
fn effective_safetensors_path_matches_mlx_extension_rule() {
  assert_eq!(
    effective_safetensors_path(Path::new("/tmp/cache.safetensors")),
    PathBuf::from("/tmp/cache.safetensors"),
  );
  assert_eq!(
    effective_safetensors_path(Path::new("/tmp/cache")),
    PathBuf::from("/tmp/cache.safetensors"),
  );
  assert_eq!(
    effective_safetensors_path(Path::new("/tmp/cache.bin")),
    PathBuf::from("/tmp/cache.bin.safetensors"),
  );
}

/// Atomic-save crash safety: when the save FAILS, a
/// previously valid cache at the destination is left **intact** and no
/// partial tempfile remains. We first write a good cache, then point a
/// second save into a directory made read-only so mlx's write into the
/// tempfile fails — the original `out_path` must still load, byte-identical.
///
/// Read-only-dir failure injection is POSIX (`unix`); on other targets the
/// no-partial-file path is covered by
/// [`save_prompt_cache_atomic_failed_save_to_fresh_path_leaves_nothing`].
#[cfg(unix)]
#[test]
fn save_prompt_cache_atomic_failed_save_keeps_original_intact() {
  use std::os::unix::fs::PermissionsExt;

  use crate::lm::cache::load_prompt_cache;

  let model = MockModel::new(4);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_cache_prompt_atomic_intact_{}",
    std::process::id()
  ));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  let out = dir.join("cache.safetensors");

  // 1. Write a good cache.
  cache_prompt_ids(
    &model,
    &[1u32, 2, 3, 4],
    &config(2),
    &out,
    "good",
    "{}",
    2,
    &HashMap::new(),
  )
  .unwrap();
  assert!(out.exists(), "the first save must produce a cache");
  let (orig_loaded, orig_meta) = load_prompt_cache(&out).unwrap();
  let orig_offsets: Vec<usize> = orig_loaded.iter().map(|x| x.offset()).collect();

  // 2. Make the directory read-only so the next save's tempfile create /
  //    mlx write fails. (Root could bypass this; CI/dev users are not root.)
  let mut perms = fs::metadata(&dir).unwrap().permissions();
  let orig_mode = perms.mode();
  perms.set_mode(0o500); // r-x------ : no write ⇒ create/write fails
  fs::set_permissions(&dir, perms).unwrap();

  let r = cache_prompt_ids(
    &model,
    &[5u32, 6, 7, 8],
    &config(2),
    &out,
    "SHOULD-NOT-WIN",
    "{}",
    2,
    &HashMap::new(),
  );

  // Restore write perms BEFORE asserting (so cleanup + reads work even if an
  // assert fails).
  let mut restore = fs::metadata(&dir).unwrap().permissions();
  restore.set_mode(orig_mode);
  fs::set_permissions(&dir, restore).unwrap();

  assert!(r.is_err(), "a save into a read-only dir must fail");

  // 3. The original cache is untouched: same metadata + offsets, and no
  //    leftover `.tmp.safetensors` partial file in the directory.
  assert!(out.exists(), "the failed save must not delete the original");
  let (after_loaded, after_meta) = load_prompt_cache(&out).unwrap();
  assert_eq!(
    after_meta.get(META_MODEL).map(String::as_str),
    Some("good"),
    "the original cache's metadata must survive the failed save (not 'SHOULD-NOT-WIN')"
  );
  assert_eq!(after_meta, orig_meta, "original metadata must be unchanged");
  let after_offsets: Vec<usize> = after_loaded.iter().map(|x| x.offset()).collect();
  assert_eq!(
    after_offsets, orig_offsets,
    "the original cache contents must be unchanged"
  );
  let leftover_tmp = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).any(|e| {
    e.file_name()
      .to_string_lossy()
      .ends_with(".tmp.safetensors")
  });
  assert!(
    !leftover_tmp,
    "no partial tempfile may remain after a failed save"
  );

  let _ = fs::remove_dir_all(&dir);
}

/// Atomic-save crash safety, fresh-path variant: a FAILED
/// save to a destination that did not previously exist leaves it **absent**
/// (no partial file). Injects failure via a read-only parent directory.
#[cfg(unix)]
#[test]
fn save_prompt_cache_atomic_failed_save_to_fresh_path_leaves_nothing() {
  use std::os::unix::fs::PermissionsExt;

  let model = MockModel::new(4);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_cache_prompt_atomic_fresh_{}",
    std::process::id()
  ));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  let out = dir.join("never.safetensors");

  let mut perms = fs::metadata(&dir).unwrap().permissions();
  let orig_mode = perms.mode();
  perms.set_mode(0o500);
  fs::set_permissions(&dir, perms).unwrap();

  let r = cache_prompt_ids(
    &model,
    &[1u32, 2, 3],
    &config(1),
    &out,
    "m",
    "{}",
    8,
    &HashMap::new(),
  );

  let mut restore = fs::metadata(&dir).unwrap().permissions();
  restore.set_mode(orig_mode);
  fs::set_permissions(&dir, restore).unwrap();

  assert!(r.is_err(), "save into a read-only dir must fail");
  assert!(
    !out.exists(),
    "a failed save to a fresh path must leave no file at the destination"
  );
  let any_file = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).count();
  assert_eq!(
    any_file, 0,
    "no partial / tempfile may remain in the directory"
  );

  let _ = fs::remove_dir_all(&dir);
}

/// Atomic-save cleanup AFTER the tempfile is written: when the final
/// `rename` fails (here the destination path is an existing **directory**,
/// which `fs::rename(file -> dir)` rejects), the tempfile that mlx already
/// wrote must be removed — no `.tmp.safetensors` leftover, and the
/// destination directory is untouched. This exercises the write-succeeds /
/// rename-fails branch the read-only-dir tests (which fail at tempfile
/// *create*) do not reach.
#[test]
fn save_prompt_cache_atomic_rename_failure_cleans_up_tempfile() {
  let model = MockModel::new(4);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_cache_prompt_atomic_rename_{}",
    std::process::id()
  ));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  // The destination is itself a directory ⇒ the final rename (file -> dir)
  // fails, but the tempfile create + mlx write succeed first.
  let out = dir.join("dest.safetensors");
  fs::create_dir_all(&out).unwrap();

  let r = cache_prompt_ids(
    &model,
    &[1u32, 2, 3],
    &config(1),
    &out,
    "m",
    "{}",
    8,
    &HashMap::new(),
  );
  assert!(r.is_err(), "rename onto an existing directory must fail");
  // The dest directory still exists (untouched) and is still a directory.
  assert!(out.is_dir(), "the destination directory must be untouched");
  // No leftover tempfile: the post-write rename failure cleaned it up.
  let leftover_tmp = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).any(|e| {
    e.file_name()
      .to_string_lossy()
      .ends_with(".tmp.safetensors")
  });
  assert!(
    !leftover_tmp,
    "the tempfile mlx wrote must be removed when the rename fails"
  );

  let _ = fs::remove_dir_all(&dir);
}
