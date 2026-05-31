//! ColVision base processor seam — ports
//! [`mlx_embeddings/colvision_processor.py`](https://github.com/Blaizzy/mlx-embeddings/blob/main/mlx_embeddings/colvision_processor.py)
//! `BaseColVisionProcessor` (lines 9-110), which itself is a port of
//! illuin-tech / colpali-engine's `BaseVisualRetrieverProcessor`
//! (`colpali_engine/utils/processing_utils.py`).
//!
//! ## What this module ships
//!
//! Mirroring the python reference's structure faithfully:
//!
//! - [`BaseColVisionProcessor`] — a **trait** mirroring the python
//!   abstract base class (python lines 9-41). It declares the three
//!   abstract methods every concrete processor (ColIdefics3 / ColQwen2_5 /
//!   …) implements:
//!     * [`process_images`](BaseColVisionProcessor::process_images) —
//!       python `process_images` (lines 18-23): turn a batch of images
//!       into the model-specific multimodal `BatchFeature` / `BatchEncoding`
//!       (image arrays + token ids + masks). Per the
//!       no-model-arch rule this trait declares the *shape* of the
//!       contract only; the dictionary-of-tensors return type is
//!       represented as a [`ProcessorBatch`] map keyed by the python
//!       field name. Concrete implementations live with each model arch.
//!     * [`process_queries`](BaseColVisionProcessor::process_queries) —
//!       python `process_queries` (lines 25-32): tokenize a batch of
//!       string queries (with the model-specific prefix / suffix /
//!       augmentation tokens) into a [`ProcessorBatch`].
//!     * [`score`](BaseColVisionProcessor::score) — python `score`
//!       (lines 34-41): the scoring entry every concrete subclass
//!       overrides (colidefics3.py:325 delegates to `score_multi_vector`
//!       for late-interaction / MaxSim).
//!
//! - [`score_single_vector`] — module-level free function mirroring
//!   python `@staticmethod BaseColVisionProcessor.score_single_vector`
//!   (lines 43-63). Dot-product score between single-vector queries `qs`
//!   and passages `ps`. Cast to `f32`.
//!
//! - [`score_multi_vector`] — module-level free function mirroring
//!   python `@staticmethod BaseColVisionProcessor.score_multi_vector`
//!   (lines 65-110). Late-interaction / MaxSim (ColBERT-like) score
//!   between multi-vector queries and passages, batched on `batch_size`.
//!   Cast to `f32`.
//!
//! Python `@staticmethod` ports as a module-level free function: the
//! method is callable without an instance in the reference (it is invoked
//! both as `BaseColVisionProcessor.score_single_vector(...)` and
//! `self.score_multi_vector(...)`), and Rust trait dispatch adds nothing.
//! This follows Rust-idiomatic ergonomics over verbatim mirroring of
//! OO sugar that has no Rust analog.
//!
//! ## What this module deliberately does NOT ship (no per-model
//! arch porting)
//!
//! Concrete model-specific processors (the python
//! `colidefics3.Processor` / `colqwen2_5.Processor` subclasses, which
//! also inherit from the HF `transformers` `*Processor` mixin and own
//! the image preprocessor + tokenizer state) are **out of scope**.
//! Those are model architectures and ship per-usecase. This module ships
//! the cross-architecture seam those subclasses register into — exactly
//! the python `colvision_processor.py` boundary.
//!
//! ## [`ProcessorBatch`] — the cross-architecture return type
//!
//! The python `process_images` / `process_queries` return either
//! `BatchFeature` or `BatchEncoding` — both are HF dict-of-tensors
//! containers (the model-specific `Processor` runs the HF processor +
//! calls `mx.array` on every numpy value, e.g. `colidefics3.py` lines
//! 287-296 and 315-322). The cross-architecture shape is a string-keyed
//! map of [`Array`] values; [`ProcessorBatch`] is a [`std::collections::HashMap`]
//! `String → Array` (the field names are model-specific:
//! `input_ids`, `attention_mask`, `pixel_values`, …). No new dep.
//!
//! ## Errors
//!
//! Recoverable failures (empty `qs`/`ps`, ragged shapes the python
//! reference rejects, shape mismatches MLX would have silently
//! broadcasted, allocation pressure on the inner batch-loop) return
//! [`Result`] with an [`Error`] message naming the cause. Python
//! `ValueError("No queries provided")` / `"No passages provided"`
//! (lines 51-54 and 75-78) map to [`Error::EmptyInput`] with the
//! python message text preserved for parity.

use std::collections::HashMap;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    EmptyInputPayload, Error, InvariantViolationPayload, LengthMismatchPayload, OutOfRangePayload,
    RankMismatchPayload, Result,
  },
  ops::{
    linalg_basic::matmul,
    logical::select,
    misc::astype,
    reduction::{max_axes, sum_axes},
    shape::{broadcast_to, concatenate, expand_dims_axes, stack, transpose_axes},
  },
};

/// Which side of a `(queries, passages)` pair the colvision scoring code
/// is inspecting. Routes the per-side context labels through a closed
/// enum so the typed-error payloads can carry static `&'static str`
/// contexts (no `format!` for the side tag).
#[derive(Clone, Copy)]
enum ColVisionSide {
  Queries,
  Passages,
}

impl ColVisionSide {
  const fn single_vector_zero_token_context(self) -> &'static str {
    match self {
      Self::Queries => "score_single_vector: queries[i]",
      Self::Passages => "score_single_vector: passages[i]",
    }
  }
  const fn multi_vector_zero_token_context(self) -> &'static str {
    match self {
      Self::Queries => "score_multi_vector: queries[i]",
      Self::Passages => "score_multi_vector: passages[i]",
    }
  }
}

/// A `String → Array` dictionary mirroring the python `BatchFeature` /
/// `BatchEncoding` return shape that `process_images` and
/// `process_queries` produce.
///
/// Concrete fields are model-specific (e.g. `input_ids`,
/// `attention_mask`, `pixel_values`, `pixel_attention_mask`) — the
/// cross-architecture seam only fixes the **type** of the bundle, not
/// the schema, exactly like the python `BatchFeature` / `BatchEncoding`
/// dict.
pub type ProcessorBatch = HashMap<String, Array>;

/// The cross-architecture ColVision processor seam, mirroring
/// [`mlx_embeddings/colvision_processor.py`](https://github.com/Blaizzy/mlx-embeddings/blob/main/mlx_embeddings/colvision_processor.py)
/// `BaseColVisionProcessor` (lines 9-41).
///
/// Concrete model-specific processors (`ColIdefics3` /
/// `ColQwen2_5` / …) own the image preprocessor + tokenizer + chat
/// template and implement this trait. The trait itself only fixes the
/// three abstract method shapes; no state lives on the seam. Per the
/// no-model-arch rule the per-model
/// implementations are out of scope for this module.
///
/// Each implementation is expected to be `!Send`-tolerant: [`Array`] is
/// `!Send`, so a `BaseColVisionProcessor` impl that holds [`Array`]
/// fields will inherit that (intentional).
pub trait BaseColVisionProcessor {
  /// Process a batch of images into the model-specific multimodal
  /// inputs. Mirrors python `process_images` (lines 18-23).
  ///
  /// The python signature accepts `List[Image.Image]` (PIL images) and
  /// returns a `BatchFeature` / `BatchEncoding` (dict of tensors).
  /// Rust:
  /// - **Images**: the byte-level image representation is delegated to
  ///   the concrete impl (a PIL-image port is the model arch's
  ///   responsibility — typically [`crate::vlm`] / a model-specific
  ///   image preprocessor). The cross-architecture seam only fixes the
  ///   **shape** of the contract: a borrowed slice of `Vec<u8>` image
  ///   blobs. A concrete impl decodes/normalizes them per its model's
  ///   preprocessor.
  /// - **Returns**: a [`ProcessorBatch`] (the Rust analog of the python
  ///   dict).
  fn process_images(&self, images: &[Vec<u8>]) -> Result<ProcessorBatch>;

  /// Process a batch of string queries into the tokenized query inputs.
  /// Mirrors python `process_queries` (lines 25-32).
  ///
  /// `max_length` and `suffix` mirror the python keyword arguments
  /// exactly (`max_length: int = 50`, `suffix: Optional[str] = None`).
  /// Concrete implementations override the prefix / suffix / padding /
  /// `max_length` truncation per the colpali-engine convention.
  fn process_queries(
    &self,
    queries: &[&str],
    max_length: usize,
    suffix: Option<&str>,
  ) -> Result<ProcessorBatch>;

  /// Score a batch of query embeddings against a batch of passage
  /// embeddings. Mirrors python `score` (lines 34-41).
  ///
  /// Concrete implementations dispatch to [`score_single_vector`] or
  /// [`score_multi_vector`] depending on the embedding shape
  /// (single-vector vs. multi-vector / MaxSim). The python `score`
  /// takes `**kwargs` passed through to the helper; the Rust seam
  /// keeps only the explicit `batch_size` knob used by
  /// `score_multi_vector` (python default `128`). A concrete impl
  /// ignoring the knob (single-vector arms) is free to do so.
  fn score(&self, qs: &[Array], ps: &[Array], batch_size: usize) -> Result<Array>;
}

/// Dot-product score between single-vector queries `qs` and passages
/// `ps`. Mirrors python `BaseColVisionProcessor.score_single_vector`
/// (lines 43-63) — `mx.einsum("bd,cd->bc", qs_stacked, ps_stacked)` is
/// equivalent to `qs_stacked @ ps_stacked.T`, computed here as
/// [`matmul`] after [`transpose_axes`] (no `einsum` FFI binding is
/// wrapped in mlxrs and a 2-D matmul + transpose is the canonical
/// equivalent — mlx itself rewrites the same einsum to a matmul
/// internally).
///
/// Each input is a single-vector embedding `(d,)`; they are
/// [`stack`]ed into a `(b, d)` query batch and a `(c, d)` passage
/// batch. The result is the `(b, c)` similarity matrix, cast to
/// [`Dtype::F32`] (python `scores.astype(mx.float32)`).
///
/// ## Errors
/// - `qs.is_empty()` → [`Error::EmptyInput`] with the python message
///   `"No queries provided"` (line 52).
/// - `ps.is_empty()` → [`Error::EmptyInput`] with the python message
///   `"No passages provided"` (line 54).
/// - Any input with `shape[0] == 0` (zero-element vector / zero-token
///   embedding) → [`Error::OutOfRange`] whose message contains
///   `"zero tokens"`. A zero-element single vector would dot-product
///   with every passage to `0.0` regardless of content, silently
///   collapsing the ranking signal; the equivalent precondition that
///   the internal `pad_to_max` helper enforces for the multi-vector
///   path is enforced here directly (this function does not go through
///   `pad_to_max`).
/// - Underlying [`stack`] / [`matmul`] errors propagate (e.g. inputs
///   with mismatched `d`).
pub fn score_single_vector(qs: &[Array], ps: &[Array]) -> Result<Array> {
  // python lines 51-54: `if len(qs) == 0: raise ValueError("No queries
  // provided"); if len(ps) == 0: raise ValueError("No passages
  // provided")`. Preserve the exact message text for parity.
  if qs.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "score_single_vector: No queries provided",
    )));
  }
  if ps.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "score_single_vector: No passages provided",
    )));
  }
  // Reject zero-element vectors. This is the single-vector analog of
  // the `pad_to_max` zero-token guard used by [`score_multi_vector`]:
  // a `(0,)` input would dot-product to `0.0` against every passage
  // regardless of content, silently collapsing the ranking signal.
  // [`stack`] of `(0,)` slices would also produce a `(B, 0)` matrix
  // and the subsequent [`matmul`] would be a no-op multiplication of
  // empty contraction axes — not what the caller intends.
  for (label, slice) in [(ColVisionSide::Queries, qs), (ColVisionSide::Passages, ps)] {
    for (i, a) in slice.iter().enumerate() {
      let sh = a.shape();
      // `shape[0] == 0` flags both `(0,)` rank-1 vectors and any
      // higher-rank input whose leading axis is empty.
      if sh.first().copied() == Some(0) {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          label.single_vector_zero_token_context(),
          "shape[0] must be > 0 (non-empty embedding vectors are required; \
           a zero-token vector would dot-product to 0.0 against every passage)",
          smol_str::format_smolstr!("index {i}, shape[0] = 0"),
        )));
      }
    }
  }
  // python line 56-57: `mx.stack(qs)`, `mx.stack(ps)`. `mlxrs::stack`
  // takes `&[&Array]`; collect borrows without cloning the arrays.
  let qs_refs: Vec<&Array> = qs.iter().collect();
  let ps_refs: Vec<&Array> = ps.iter().collect();
  let qs_stacked = stack(&qs_refs)?;
  let ps_stacked = stack(&ps_refs)?;
  // python line 59: `mx.einsum("bd,cd->bc", qs_stacked, ps_stacked)`.
  // For 2-D inputs this is exactly `qs @ ps.T`. Use `matmul` + a
  // last-two-axis transpose. (No `mlx_einsum` wrapper exists in mlxrs;
  // mlx's einsum lowers a bd,cd→bc to a matmul anyway.)
  let ps_t = transpose_axes(&ps_stacked, &[1, 0])?;
  let scores = matmul(&qs_stacked, &ps_t)?;
  // python lines 60-62: `assert scores.shape[0] == len(qs)`. mlx's
  // matmul of `(b,d) @ (d,c)` always produces `(b,c)` so this is
  // structurally guaranteed for valid inputs; the python assert is a
  // defensive sanity check we mirror as a non-panicking
  // [`Error::LengthMismatch`] (zero overhead for the success path).
  let s = scores.shape();
  if s.first().copied() != Some(qs.len()) {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "score_single_vector: scores.shape[0] must equal qs.len()",
      qs.len(),
      s.first().copied().unwrap_or(0),
    )));
  }
  // python line 63: `.astype(mx.float32)`. A no-op cast for f32 inputs.
  astype(&scores, Dtype::F32)
}

/// MaxSim / late-interaction (ColBERT-like) score between multi-vector
/// queries `qs` and passages `ps`. Mirrors python
/// `BaseColVisionProcessor.score_multi_vector` (lines 65-110).
///
/// Each query is an `(n_i, d)` multi-vector embedding, each passage an
/// `(s_j, d)`. Within each `(batch_size, batch_size)` tile the function
/// zero-pads ragged token-axis lengths to the tile max (the python
/// `pad_to_max` inner helper, lines 80-91), then for each query-passage tile computes the
/// `(n, s)` similarity matrix, takes the max over the passage tokens
/// (`axis=3`), and sums over the query tokens (`axis=2`) — the MaxSim
/// reduction. Tiles are [`concatenate`]d along the passage axis (within
/// a query tile) and then along the query axis. Result is `(B, C)`
/// where `B = qs.len()` and `C = ps.len()`, cast to [`Dtype::F32`]
/// (python `scores.astype(mx.float32)`).
///
/// `batch_size` mirrors the python keyword (`batch_size: int = 128`).
///
/// `einsum("bnd,csd->bcns", qs_batch, ps_batch)` is implemented via
/// rank-4 [`matmul`] after expanding `qs_batch (b,n,d)` to `(b,1,n,d)`,
/// expanding `ps_batch.transpose(0,2,1) (c,d,s)` to `(1,c,d,s)`, and
/// matmuling — mlx's `matmul` batches the leading dims, so the result
/// is exactly `(b,c,n,s)` (the einsum semantic). No `mlx_einsum` FFI
/// wrapper is needed.
///
/// ## Deliberate divergence from the python reference: padded-passage masking
///
/// The python reference at
/// [`mlx_embeddings/colvision_processor.py`](https://github.com/Blaizzy/mlx-embeddings/blob/main/mlx_embeddings/colvision_processor.py)
/// lines 80-102 zero-pads ragged passages with **zero vectors** and
/// then includes those padded columns in `mx.max(sim, axis=3)`. For
/// signed embeddings (e.g. anything with negative similarity components
/// — common with non-normalized or non-ReLU'd encoders) this is a
/// **correctness bug**: a real similarity of `-1.0` between query and
/// a passage token loses to the padded `0.0` dot product, so MaxSim
/// reports `0` instead of `-1` whenever the passage was tile-padded.
/// Worst-case, a passage's score depends on its `batch_size` tile
/// neighbours: passage `p0 = [[-1, 0]]` returns `-1.0` alone but `0.0`
/// when tiled with a length-2 passage. This violates the contract
/// (MaxSim should be batch-size-agnostic) and corrupts ranking.
///
/// The mlxrs port fixes this by masking the padded positions to
/// `f32::NEG_INFINITY` (cast via [`astype`] to the input dtype to
/// preserve f16/bf16) **before** the [`max_axes`] reduction. Padded
/// positions can never win the max, so the per-tile result equals the
/// untiled result for every passage, restoring batch-size invariance.
///
/// Dtype choice: `f32::NEG_INFINITY` (via the existing `scalar_like`
/// pattern shared with [`crate::embeddings::pooling::max_pooling`])
/// rather than `T::MIN` finite — mlx's [`max_axes`] handles `-inf`
/// cleanly (no NaN propagation: the input similarities are finite, so
/// `max(finite, -inf) = finite` always). The all-padded edge case
/// cannot arise here because the internal `pad_to_max` helper only ever
/// pads *up to* the tile max length, so every column at index
/// `< max_len` exists for at least one passage in the tile. (A `-inf`
/// result would only escape
/// the max if a real similarity happened to be `-inf`, which finite
/// f32/f16/bf16 dot products of finite embeddings cannot produce.)
///
/// ### Precondition (enforced): no zero-token queries or passages
///
/// The "padded positions can never win the max" invariant above
/// depends on every input having **at least one** real token (i.e.
/// `shape[0] >= 1`). A zero-token passage `(0, d)` would record `0`
/// in the internal `pad_to_max` helper's `original_lengths`; the
/// per-passage mask row would then be all-`false`, [`select`] would
/// replace every position with `-inf`, and `max(axis=3)` on an
/// all-`-inf` row returns `-inf`, which `sum(axis=2)` would propagate
/// as a non-finite ranking score. To preserve the invariant, the
/// internal `pad_to_max` helper explicitly rejects any array with
/// `shape[0] == 0` with an [`Error::OutOfRange`] whose message
/// contains `"zero tokens"`. Both the query and passage paths of
/// `score_multi_vector` inherit this precondition through their
/// `pad_to_max` calls. Callers must filter out empty-tokenization
/// inputs before invoking the scorer.
///
/// An upstream issue should be filed against
/// <https://github.com/Blaizzy/mlx-embeddings> referencing this PR.
///
/// ## Errors
/// - `qs.is_empty()` → [`Error::EmptyInput`] with the python message
///   `"No queries provided"` (line 76).
/// - `ps.is_empty()` → [`Error::EmptyInput`] with the python message
///   `"No passages provided"` (line 78).
/// - `batch_size == 0` → [`Error::InvariantViolation`] (the python
///   `range(0, len(qs), batch_size)` would `ValueError` on
///   `batch_size == 0`; surface the equivalent recoverable error
///   instead of looping forever).
/// - Any query or passage with `shape[0] == 0` → [`Error::OutOfRange`]
///   whose message identifies the offending input by path tag and
///   *global* index (e.g. `"score_multi_vector: passages[3] has zero
///   tokens (shape[0] == 0); ..."`) and contains the substring
///   `"zero tokens"`. The check runs up front, before any tiling, so
///   the index is the caller's index into `qs` / `ps` — not a
///   tile-local index. See the "Precondition (enforced)" subsection of
///   the divergence note above. (The internal `pad_to_max` helper
///   also rejects `shape[0] == 0` as defense-in-depth, but its message
///   is never observed by `score_multi_vector` callers because the
///   pre-validation fires first.)
/// - Per-tile shape errors (mismatched `d`, non-rank-2 inputs) propagate
///   from [`stack`] / [`matmul`].
pub fn score_multi_vector(qs: &[Array], ps: &[Array], batch_size: usize) -> Result<Array> {
  // python lines 75-78: same "No queries"/"No passages" guards.
  if qs.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "score_multi_vector: No queries provided",
    )));
  }
  if ps.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "score_multi_vector: No passages provided",
    )));
  }
  // Python `range(0, len(qs), batch_size)` would raise `ValueError:
  // range() arg 3 must not be zero` on `batch_size == 0`. Surface the
  // recoverable equivalent rather than enter an infinite loop.
  if batch_size == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "score_multi_vector: batch_size",
      "must be > 0 (a zero batch_size would yield an infinite tiling loop)",
    )));
  }
  // Pre-validate zero-token inputs up front so the error identifies the
  // offending array by *global* index AND path tag (queries vs passages).
  // The inner [`pad_to_max`] helper also rejects `shape[0] == 0` as
  // defense-in-depth, but its message uses the *tile-local* index and
  // has no path tag — e.g. with `batch_size = 128`, passage `ps[129]`
  // would otherwise be reported as `array 1`. The early guard fires
  // before the tile loop, so `pad_to_max` never sees zero-token input
  // from this caller. Mirrors the single-vector path's pre-validation
  // (lines 219-233).
  for (label, slice) in [(ColVisionSide::Queries, qs), (ColVisionSide::Passages, ps)] {
    for (i, a) in slice.iter().enumerate() {
      let sh = a.shape();
      // `shape[0] == 0` flags both `(0,)` rank-1 vectors and any
      // higher-rank input whose leading axis is empty.
      if sh.first().copied() == Some(0) {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          label.multi_vector_zero_token_context(),
          "shape[0] must be > 0 (non-empty token sequences are required for the \
           masked MaxSim contract)",
          smol_str::format_smolstr!("index {i}, shape[0] = 0"),
        )));
      }
    }
  }
  // python lines 93-105: outer-loop over query tiles, inner-loop over
  // passage tiles, MaxSim per tile, concat along passage axis (axis=1)
  // within a query tile, then concat along query axis (axis=0) across
  // all query tiles.
  let mut scores_list: Vec<Array> = Vec::with_capacity(qs.len().div_ceil(batch_size));
  let mut i = 0usize;
  while i < qs.len() {
    let j_end_q = i.saturating_add(batch_size).min(qs.len());
    // Query padding is benign: a zero query token has 0 dot product with
    // every passage token (real or padded), so its post-max(axis=3) is
    // 0 and contributes 0 to the sum(axis=2) — the per-passage score is
    // unaffected. We therefore discard the query-side lengths.
    let (qs_batch, _q_lens) = pad_to_max(&qs[i..j_end_q])?;
    let mut scores_batch_parts: Vec<Array> = Vec::with_capacity(ps.len().div_ceil(batch_size));
    let mut j = 0usize;
    while j < ps.len() {
      let j_end_p = j.saturating_add(batch_size).min(ps.len());
      // Passage padding is NOT benign — see the divergence note on the
      // `score_multi_vector` doc: a zero passage column dot-producted
      // with a real query token yields 0, and `max(real_negative, 0) = 0`
      // → padded positions win the max for signed embeddings. Retain
      // the original passage lengths so we can mask them out below.
      let (ps_batch, p_lens) = pad_to_max(&ps[j..j_end_p])?;
      // python line 100: `mx.einsum("bnd,csd->bcns", qs_batch,
      // ps_batch)`. Expand qs_batch (b,n,d) → (b,1,n,d) and
      // ps_batch (c,s,d) → transpose to (c,d,s) → (1,c,d,s); a rank-4
      // matmul broadcasts the leading dims and contracts the last two
      // dims, producing exactly (b,c,n,s).
      let qs_b = expand_dims_axes(&qs_batch, &[1])?; // (b,1,n,d)
      let ps_t = transpose_axes(&ps_batch, &[0, 2, 1])?; // (c,d,s)
      let ps_b = expand_dims_axes(&ps_t, &[0])?; // (1,c,d,s)
      let sim = matmul(&qs_b, &ps_b)?; // (b,c,n,s)
      // DIVERGENCE FROM PYTHON REFERENCE: mask padded passage columns to
      // -inf so they cannot win `max(axis=3)`. See the module-level
      // divergence note. The mask is shape `(c, s)` (one row per
      // passage in the tile, one column per token position). It is
      // explicitly expanded to `(1, c, 1, s)` and broadcast to
      // `(b, c, n, s)` before [`select`] so the masking is independent
      // of every `(b, n)` query position.
      let s_shape = sim.shape(); // (b, c, n, s)
      let (b, c, n, s_max) = (s_shape[0], s_shape[1], s_shape[2], s_shape[3]);
      // Build a flat (c * s_max) bool mask: true = real, false = padded.
      // p_lens[k] is the *original* length of passage k in this tile.
      let mut flat_mask: Vec<bool> = Vec::with_capacity(c * s_max);
      for &len in &p_lens {
        for t in 0..s_max {
          flat_mask.push(t < len);
        }
      }
      let mask_cs = Array::from_slice::<bool>(&flat_mask, &(c, s_max))?; // (c, s)
      // Reshape to (1, c, 1, s_max), broadcast to (b, c, n, s_max).
      let mask_4d = expand_dims_axes(&mask_cs, &[0, 2])?; // (1, c, 1, s_max)
      let mask_full = broadcast_to(&mask_4d, &(b, c, n, s_max))?;
      // -inf scalar in `sim`'s dtype so the mask preserves f16/bf16 (no
      // silent f32 promotion). Mirrors the `scalar_like` pattern used in
      // `embeddings::pooling::max_pooling`.
      let neg_inf_scalar = Array::full::<f32>(&(1,), f32::NEG_INFINITY)?;
      let neg_inf = astype(&neg_inf_scalar, sim.dtype()?)?;
      let neg_inf_bcast = broadcast_to(&neg_inf, &(b, c, n, s_max))?;
      let sim_masked = select(&mask_full, &sim, &neg_inf_bcast)?;
      // python line 101: `mx.max(sim, axis=3)` → (b,c,n). Padded
      // positions are now -inf so they can't win the max.
      let maxsim = max_axes(&sim_masked, &[3], false)?;
      // python line 102: `mx.sum(maxsim, axis=2)` → (b,c).
      let summed = sum_axes(&maxsim, &[2], false)?;
      scores_batch_parts.push(summed);
      j = j_end_p;
    }
    // python line 104: `mx.concatenate(scores_batch, axis=1)`.
    let scores_batch_refs: Vec<&Array> = scores_batch_parts.iter().collect();
    let scores_batch = concatenate(&scores_batch_refs, 1)?;
    scores_list.push(scores_batch);
    i = j_end_q;
  }
  // python line 106: `mx.concatenate(scores_list, axis=0)`.
  let scores_list_refs: Vec<&Array> = scores_list.iter().collect();
  let scores = concatenate(&scores_list_refs, 0)?;
  // python lines 107-109: `assert scores.shape[0] == len(qs)`. Mirror
  // as a recoverable [`Error::LengthMismatch`] (structurally guaranteed
  // for valid inputs; defensive).
  let s = scores.shape();
  if s.first().copied() != Some(qs.len()) {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "score_multi_vector: scores.shape[0] must equal qs.len()",
      qs.len(),
      s.first().copied().unwrap_or(0),
    )));
  }
  // python line 110: `.astype(mx.float32)`. No-op cast for f32 inputs.
  astype(&scores, Dtype::F32)
}

/// Zero-pad a slice of `(n_i, d)` multi-vector arrays to the slice's
/// max `n` and [`stack`] them into a `(len, max_n, d)` batch. Mirrors
/// the python inner helper `pad_to_max` (lines 80-91):
///
/// ```python
/// def pad_to_max(arrays):
///     max_len = max(a.shape[0] for a in arrays)
///     emb_dim = arrays[0].shape[1]
///     padded = []
///     for a in arrays:
///         pad_width = max_len - a.shape[0]
///         if pad_width > 0:
///             pad = mx.zeros((pad_width, emb_dim), dtype=a.dtype)
///             padded.append(mx.concatenate([a, pad], axis=0))
///         else:
///             padded.append(a)
///     return mx.stack(padded)
/// ```
///
/// Padding is done with [`Array::zeros`] of the *input dtype*
/// ([`Dtype`] preserved from the input, mirroring python's
/// `dtype=a.dtype`) so a half-precision query batch stays half — no
/// silent f32 promotion (the dtype-fidelity discipline this module shares with the rest of
/// `embeddings`).
///
/// Empty `arrays` is rejected by the callers
/// ([`score_multi_vector`]'s outer guards), so this helper documents
/// `!arrays.is_empty()` as a precondition; calling with an empty slice
/// returns [`Error::EmptyInput`] (defensive — would otherwise panic
/// on `arrays[0].shape()`).
///
/// Any **individual** array with `shape[0] == 0` is also rejected with
/// an [`Error::OutOfRange`] whose message contains `"zero tokens"`.
/// A zero-token sequence would record `0` in `original_lengths`, and
/// the [`score_multi_vector`] masking loop would then build an
/// all-`false` row for that passage — every position in `sim` for that
/// passage would be replaced with `-inf` by [`select`], `max(axis=3)`
/// would return `-inf`, and `sum(axis=2)` would propagate a non-finite
/// ranking score. Enforcing the precondition here (rather than in the
/// caller) means both the query and passage paths of
/// [`score_multi_vector`] inherit the guard for free. See the
/// "Precondition (enforced)" subsection of the divergence note on
/// [`score_multi_vector`].
///
/// ## Return shape (diverges from python)
///
/// Returns `(padded, original_lengths)`:
/// - `padded` — the stacked `(len, max_n, d)` batch (the python return).
/// - `original_lengths` — a `Vec<usize>` of length `arrays.len()` where
///   `original_lengths[i] == arrays[i].shape()[0]` (before padding).
///   Required by [`score_multi_vector`] to mask the zero-padded
///   passage columns to `-inf` before the MaxSim `max(axis=3)`. See
///   the divergence note on [`score_multi_vector`].
pub(crate) fn pad_to_max(arrays: &[Array]) -> Result<(Array, Vec<usize>)> {
  if arrays.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "pad_to_max: arrays slice",
    )));
  }
  // python line 81: `max_len = max(a.shape[0] for a in arrays)`. Also
  // validate rank-2 (the python ref assumes it; surface a recoverable
  // error if not).
  let first_shape = arrays[0].shape();
  if first_shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "pad_to_max: arrays must be rank-2 (n, d)",
      first_shape.len() as u32,
      first_shape,
    )));
  }
  let emb_dim = first_shape[1]; // python line 82
  let mut max_len: usize = 0;
  let mut original_lengths: Vec<usize> = Vec::with_capacity(arrays.len());
  for (i, a) in arrays.iter().enumerate() {
    let sh = a.shape();
    if sh.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "pad_to_max: arrays must be rank-2 (n, d)",
        sh.len() as u32,
        sh,
      )));
    }
    if sh[1] != emb_dim {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "pad_to_max: all arrays must share emb_dim",
        emb_dim,
        sh[1],
      )));
    }
    // Reject zero-token sequences. A `(0, d)` array would record `0` in
    // `original_lengths`, which the [`score_multi_vector`] masking loop
    // would turn into an all-false row for that passage → every position
    // becomes `-inf` after [`select`] → `max(axis=3)` returns `-inf` →
    // `sum(axis=2)` propagates a non-finite ranking score. The
    // mask-padded-positions-to-`-inf` invariant only holds when every
    // input has at least one real token; enforce that precondition here
    // rather than emit non-finite scores downstream. Mirrors the
    // python reference's implicit assumption (the upstream
    // `pad_to_max` also breaks on zero-token inputs: `max_len = 0`
    // gives a `(len, 0, d)` stack and `max(axis=3)` of an empty axis
    // is undefined). See the divergence note on [`score_multi_vector`].
    if sh[0] == 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "pad_to_max: arrays[i]",
        "shape[0] must be > 0 (non-empty token sequences are required for the \
         masked MaxSim contract)",
        smol_str::format_smolstr!("index {i}, shape[0] = 0"),
      )));
    }
    if sh[0] > max_len {
      max_len = sh[0];
    }
    original_lengths.push(sh[0]);
  }
  // python lines 83-90: zero-pad each array to (max_len, emb_dim) in
  // its own dtype; arrays already at max_len pass through.
  let mut padded: Vec<Array> = Vec::with_capacity(arrays.len());
  for a in arrays {
    let n = a.shape()[0];
    let pad_width = max_len - n;
    if pad_width > 0 {
      // python line 87: `pad = mx.zeros((pad_width, emb_dim),
      // dtype=a.dtype)`. Mirror dtype via [`astype`] of a f32 zero
      // tensor — a f32→f32 cast is a no-op, and a f16/bf16 input
      // batch stays in that dtype (no silent promotion).
      let dtype = a.dtype()?;
      let pad = match dtype {
        Dtype::F32 => Array::zeros::<f32>(&(pad_width, emb_dim))?,
        _ => {
          let z = Array::zeros::<f32>(&(pad_width, emb_dim))?;
          astype(&z, dtype)?
        }
      };
      // python line 88: `mx.concatenate([a, pad], axis=0)`.
      let cat = concatenate(&[a, &pad], 0)?;
      padded.push(cat);
    } else {
      // python line 90: `padded.append(a)` (no allocation; cheap
      // try_clone preserves the lazy graph reference).
      padded.push(a.try_clone()?);
    }
  }
  // python line 91: `mx.stack(padded)`.
  let refs: Vec<&Array> = padded.iter().collect();
  let stacked = stack(&refs)?;
  Ok((stacked, original_lengths))
}

#[cfg(test)]
mod tests;
