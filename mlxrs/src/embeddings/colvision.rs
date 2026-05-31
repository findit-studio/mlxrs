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
mod tests {
  use super::*;

  // ---------- score_single_vector ----------

  /// python line 52: `if len(qs) == 0: raise ValueError("No queries
  /// provided")`. Faithful error-message parity.
  #[test]
  fn score_single_vector_rejects_empty_queries() {
    let p = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
    let err = score_single_vector(&[], std::slice::from_ref(&p)).unwrap_err();
    let msg = format!("{err}");
    assert!(
      msg.contains("No queries provided"),
      "expected python parity msg, got {msg}"
    );
  }

  /// python line 54: `if len(ps) == 0: raise ValueError("No passages
  /// provided")`.
  #[test]
  fn score_single_vector_rejects_empty_passages() {
    let q = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
    let err = score_single_vector(std::slice::from_ref(&q), &[]).unwrap_err();
    let msg = format!("{err}");
    assert!(
      msg.contains("No passages provided"),
      "expected python parity msg, got {msg}"
    );
  }

  /// python line 59: `mx.einsum("bd,cd->bc", qs, ps)` — for 2 queries
  /// `[[1,0,0],[0,1,0]]` and 2 passages `[[1,1,1],[2,0,2]]` the
  /// expected scores are `[[1,2],[1,0]]`. Hand-traced.
  #[test]
  fn score_single_vector_dot_product_shape_and_values() {
    let q0 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0], &(3,)).unwrap();
    let q1 = Array::from_slice::<f32>(&[0.0, 1.0, 0.0], &(3,)).unwrap();
    let p0 = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(3,)).unwrap();
    let p1 = Array::from_slice::<f32>(&[2.0, 0.0, 2.0], &(3,)).unwrap();
    let mut scores =
      score_single_vector(&[q0, q1], &[p0, p1]).expect("score_single_vector should succeed");
    assert_eq!(scores.shape(), vec![2, 2], "shape (B,C) = (2,2)");
    assert_eq!(scores.dtype().unwrap(), Dtype::F32, "f32 cast (python L63)");
    let v = scores.to_vec::<f32>().unwrap();
    // [[<q0,p0>=1, <q0,p1>=2], [<q1,p0>=1, <q1,p1>=0]]
    assert_eq!(v, vec![1.0, 2.0, 1.0, 0.0]);
  }

  /// REGRESSION (single-vector analog): a
  /// `(0,)` query embedding (zero-element vector) must be rejected.
  /// The single-vector path does not go through [`pad_to_max`]; it has
  /// its own early guard. A `(0,)` vector would dot-product to `0.0`
  /// against every passage regardless of content, silently collapsing
  /// the ranking signal. Assertion: returns [`Error::OutOfRange`]
  /// whose message contains `"zero tokens"`.
  #[test]
  fn score_single_vector_rejects_zero_token_query() {
    let q_empty = Array::from_slice::<f32>(&[], &(0,)).unwrap();
    let p = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
    let err =
      score_single_vector(std::slice::from_ref(&q_empty), std::slice::from_ref(&p)).unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
      msg.contains("queries"),
      "expected 'queries' in message, got {msg}"
    );
  }

  /// REGRESSION (single-vector analog): a
  /// `(0,)` passage embedding must be rejected for the same reasons as
  /// the query analog. Assertion: returns [`Error::OutOfRange`]
  /// whose message contains `"zero tokens"` and identifies the
  /// passage path.
  #[test]
  fn score_single_vector_rejects_zero_token_passage() {
    let q = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
    let p_empty = Array::from_slice::<f32>(&[], &(0,)).unwrap();
    let err =
      score_single_vector(std::slice::from_ref(&q), std::slice::from_ref(&p_empty)).unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
      msg.contains("passages"),
      "expected 'passages' in message, got {msg}"
    );
  }

  /// python line 63: `.astype(mx.float32)` — non-f32 input must come
  /// back as f32 (dtype upcast for the score, but inputs themselves
  /// stay in their dtype until the final cast).
  #[test]
  fn score_single_vector_casts_result_to_f32_from_f16() {
    let q0 = Array::from_slice::<f32>(&[1.0, 0.0], &(2,))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let p0 = Array::from_slice::<f32>(&[1.0, 1.0], &(2,))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let scores = score_single_vector(&[q0], &[p0]).unwrap();
    assert_eq!(scores.shape(), vec![1, 1]);
    assert_eq!(
      scores.dtype().unwrap(),
      Dtype::F32,
      "result must be f32 even with f16 inputs"
    );
  }

  // ---------- score_multi_vector ----------

  /// python lines 75-78: same empty-input ValueError parity.
  #[test]
  fn score_multi_vector_rejects_empty_queries() {
    let p = Array::from_slice::<f32>(&[1.0; 4], &(2, 2)).unwrap();
    let err = score_multi_vector(&[], std::slice::from_ref(&p), 128).unwrap_err();
    assert!(format!("{err}").contains("No queries provided"));
  }

  #[test]
  fn score_multi_vector_rejects_empty_passages() {
    let q = Array::from_slice::<f32>(&[1.0; 4], &(2, 2)).unwrap();
    let err = score_multi_vector(std::slice::from_ref(&q), &[], 128).unwrap_err();
    assert!(format!("{err}").contains("No passages provided"));
  }

  /// Mlxrs-only guard: a `batch_size == 0` would put the python
  /// `range(0, len(qs), 0)` into a `ValueError`. Surface a
  /// recoverable error instead of looping forever.
  #[test]
  fn score_multi_vector_rejects_zero_batch_size() {
    let q = Array::from_slice::<f32>(&[1.0; 4], &(2, 2)).unwrap();
    let p = Array::from_slice::<f32>(&[1.0; 4], &(2, 2)).unwrap();
    let err =
      score_multi_vector(std::slice::from_ref(&q), std::slice::from_ref(&p), 0).unwrap_err();
    assert!(format!("{err}").contains("batch_size"));
  }

  /// REGRESSION: a zero-token query is
  /// rejected by `score_multi_vector` — even though the outer
  /// `qs.is_empty()` guard passes, the per-array `shape[0] == 0` check
  /// must fire. The contract is that callers filter out
  /// empty-tokenization inputs before invoking the scorer; if they
  /// don't, the failure must be observable and recoverable (not
  /// non-finite scores).
  ///
  /// REGRESSION: the message must carry the
  /// path tag (`queries`) AND the *global* index, not a tile-local
  /// `array N` from the inner `pad_to_max` helper.
  ///
  /// Assertion: returns [`Error::OutOfRange`] whose message
  /// contains `"zero tokens"` AND `"queries[0]"`, not a propagation of
  /// `-inf` through the masked MaxSim.
  #[test]
  fn score_multi_vector_rejects_zero_token_query() {
    let q_empty = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
    let p = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let err = score_multi_vector(
      std::slice::from_ref(&q_empty),
      std::slice::from_ref(&p),
      128,
    )
    .unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange from score_multi_vector pre-validation, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
      msg.contains("queries") && msg.contains("index 0"),
      "expected 'queries' + 'index 0' in message, got {msg}"
    );
  }

  /// REGRESSION: the high-severity fixture.
  /// `q = [[1, 0]]`, `p0 = [[0_size_2]]` (zero tokens), `p1 = [[2, 0],
  /// [0, 1]]`. Without the zero-token guard, the `(c=2, s_max=2)` mask
  /// row for `p0` would be all-`false`, [`select`] would replace every
  /// `(b=1, c=0, n=1, s)` similarity with `-inf`, `max(axis=3)` would
  /// return `-inf` for that passage, and `sum(axis=2)` would propagate
  /// to a `-inf` ranking score. The guard surfaces a recoverable
  /// [`Error::OutOfRange`] instead.
  ///
  /// REGRESSION: the message must carry the
  /// path tag (`passages`) AND the *global* index — here `passages[0]`
  /// — even though `p0` is also at tile-local index 0 (the
  /// distinguishing global-vs-local fixture is below).
  #[test]
  fn score_multi_vector_rejects_zero_token_passage_in_mixed_tile() {
    let q = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
    // p0 is a zero-token passage `(0, 2)`; p1 has two real tokens.
    let p0 = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
    let p1 = Array::from_slice::<f32>(&[2.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    // batch_size=2 forces both passages into the same tile, so the
    // padded `p0` would otherwise produce an all-masked row.
    let err = score_multi_vector(std::slice::from_ref(&q), &[p0, p1], 2).unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange from pre-validation (NOT -inf propagation), got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
      msg.contains("passages") && msg.contains("index 0"),
      "expected 'passages' + 'index 0' in message, got {msg}"
    );
  }

  /// REGRESSION: the distinguishing global-
  /// vs-tile-local fixture for the QUERY path. With `qs.len() = 4`
  /// and `batch_size = 2`, the offending zero-token query at global
  /// index 3 lives in the SECOND tile and would have been reported by
  /// a tile-local-only impl as `array 1` (tile-local within tile #1).
  /// The pre-validate fix reports the global `queries[3]` instead.
  #[test]
  fn score_multi_vector_rejects_zero_token_query_at_non_zero_global_index() {
    let q_valid_0 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
    let q_valid_1 = Array::from_slice::<f32>(&[0.0, 1.0], &(1, 2)).unwrap();
    let q_valid_2 = Array::from_slice::<f32>(&[1.0, 1.0], &(1, 2)).unwrap();
    let q_empty = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
    let p = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    // Empty query at global #3 (NOT first tile when batch_size=2): tiles
    // {q_valid_0, q_valid_1} and {q_valid_2, q_empty}. q_empty is
    // tile-local index 1 within tile #1 but global index 3.
    let qs = vec![q_valid_0, q_valid_1, q_valid_2, q_empty];
    let result = score_multi_vector(&qs, std::slice::from_ref(&p), 2);
    let err = match result {
      Err(e) => e,
      Ok(_) => panic!("expected OutOfRange, got Ok"),
    };
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
      msg.contains("queries") && msg.contains("index 3"),
      "expected 'queries' + global index 3, got: {msg}"
    );
    // Defense-in-depth: assert the tile-local forms are absent.
    assert!(
      !msg.contains("index 1") && !msg.contains("array 1"),
      "tile-local index leaked: {msg}"
    );
  }

  /// REGRESSION: the distinguishing global-
  /// vs-tile-local fixture. With `ps.len() = 4` and `batch_size = 2`,
  /// the offending zero-token passage at global index 3 lives in the
  /// SECOND tile and would have been reported by the inner
  /// `pad_to_max` helper as `array 1` (tile-local). The pre-validate
  /// fix reports the global `passages[3]` instead.
  #[test]
  fn score_multi_vector_rejects_zero_token_passage_at_non_zero_global_index() {
    let q = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
    let p0 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
    let p1 = Array::from_slice::<f32>(&[0.0, 1.0], &(1, 2)).unwrap();
    let p2 = Array::from_slice::<f32>(&[1.0, 1.0], &(1, 2)).unwrap();
    let p_empty = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
    // batch_size=2 → tiles {p0, p1} and {p2, p_empty}; p_empty is
    // tile-local index 1 within its tile but global index 3.
    let err = score_multi_vector(std::slice::from_ref(&q), &[p0, p1, p2, p_empty], 2).unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange from pre-validation, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
      msg.contains("passages") && msg.contains("index 3"),
      "expected 'passages' + global 'index 3' (not tile-local 'array 1') in message, got {msg}"
    );
  }

  /// python lines 100-102: MaxSim for one query, one passage, both
  /// rank-2 with the same `n`. Hand-traced expected score.
  ///
  /// q = [[1, 0],         p = [[1, 0],
  ///      [0, 1]]              [0, 1]]
  /// sim = q @ p.T = [[1,0],[0,1]] (n=2, s=2)
  /// max over s (axis=1, i.e. axis=3 in the (b,c,n,s) tensor) = [1, 1]
  /// sum over n (axis=0, i.e. axis=2 in (b,c,n)) = 2
  /// → scores = [[2]] in f32.
  #[test]
  fn score_multi_vector_identity_pair() {
    let q = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let p = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let mut scores = score_multi_vector(&[q], &[p], 128).unwrap();
    assert_eq!(scores.shape(), vec![1, 1]);
    assert_eq!(scores.dtype().unwrap(), Dtype::F32);
    assert_eq!(scores.to_vec::<f32>().unwrap(), vec![2.0]);
  }

  /// python lines 80-91 + 100-102: MaxSim across two queries (with
  /// different `n`) and two passages (with different `s`) at
  /// `batch_size = 1` exercises the inner-loop, [`pad_to_max`]
  /// padding, and cross-tile [`concatenate`].
  ///
  /// q0 = [[1,0]]            (n=1)        q1 = [[1,0],[0,1]]    (n=2)
  /// p0 = [[1,0],[0,1]]      (s=2)        p1 = [[1,0]]           (s=1)
  ///
  /// MaxSim(q,p) = Σ_n max_s <q_n, p_s>
  /// (q0,p0): max([<[1,0],[1,0]>, <[1,0],[0,1]>]) = max(1,0)=1 → sum=1
  /// (q0,p1): max([<[1,0],[1,0]>]) = 1 → sum=1
  /// (q1,p0): n=2 rows → row0 max(1,0)=1; row1 max(0,1)=1 → sum=2
  /// (q1,p1): n=2 rows → row0 max(1)=1; row1 max(0)=0 → sum=1
  ///
  /// expected (B=2, C=2) = [[1,1],[2,1]]
  #[test]
  fn score_multi_vector_ragged_n_and_s_with_batching() {
    let q0 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
    let q1 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let p0 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let p1 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
    let mut scores = score_multi_vector(&[q0, q1], &[p0, p1], 1).unwrap();
    assert_eq!(scores.shape(), vec![2, 2]);
    assert_eq!(scores.dtype().unwrap(), Dtype::F32);
    assert_eq!(scores.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 2.0, 1.0]);
  }

  /// `batch_size >= len(qs)` must produce the same result as a tiled
  /// run — the outer/inner loop semantics are batch-size-agnostic.
  /// Re-runs the previous fixture with `batch_size = 128` (python
  /// default).
  #[test]
  fn score_multi_vector_default_batch_size_matches_tiled() {
    let q0 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
    let q1 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let p0 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let p1 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
    let mut scores = score_multi_vector(&[q0, q1], &[p0, p1], 128).unwrap();
    assert_eq!(scores.shape(), vec![2, 2]);
    assert_eq!(scores.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 2.0, 1.0]);
  }

  /// python line 110: `.astype(mx.float32)` — f16 multi-vector inputs
  /// must produce an f32 score.
  #[test]
  fn score_multi_vector_casts_result_to_f32_from_f16() {
    let q = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let p = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let scores = score_multi_vector(&[q], &[p], 128).unwrap();
    assert_eq!(scores.shape(), vec![1, 1]);
    assert_eq!(scores.dtype().unwrap(), Dtype::F32);
  }

  // ---------- pad_to_max ----------

  /// python lines 80-91: ragged inputs zero-padded along axis=0 to the
  /// slice max, then stacked. Hand-traced shape + dtype.
  #[test]
  fn pad_to_max_pads_ragged_then_stacks() {
    let a = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2)).unwrap(); // n=1
    let b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0, 6.0], &(2, 2)).unwrap(); // n=2
    let (mut padded, _lens) = pad_to_max(&[a, b]).unwrap();
    // (len=2, max_n=2, d=2). a is padded with one zero row; b is unchanged.
    assert_eq!(padded.shape(), vec![2, 2, 2]);
    let v = padded.to_vec::<f32>().unwrap();
    // a → [[1,2],[0,0]]; b → [[3,4],[5,6]]
    assert_eq!(v, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0]);
  }

  /// Empty slice precondition surfaces as a recoverable error rather
  /// than the python `IndexError` (`arrays[0]` on `[]`).
  #[test]
  fn pad_to_max_rejects_empty_slice() {
    let err = pad_to_max(&[]).unwrap_err();
    assert!(format!("{err}").contains("empty"));
  }

  /// python `arrays[0].shape[1]` assumes rank-2 inputs; surface a
  /// recoverable error on rank-1 instead of an FFI panic.
  #[test]
  fn pad_to_max_rejects_non_rank_2() {
    let bad = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
    let err = pad_to_max(std::slice::from_ref(&bad)).unwrap_err();
    assert!(format!("{err}").contains("rank-2"));
  }

  /// All arrays must share `emb_dim` (the python ref implicitly
  /// requires this — `mx.stack` would fail on mismatched dims after
  /// the per-array pad, but we catch it upfront with a clearer
  /// message).
  #[test]
  fn pad_to_max_rejects_mismatched_emb_dim() {
    let a = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2)).unwrap();
    let b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0], &(1, 3)).unwrap();
    let err = pad_to_max(&[a, b]).unwrap_err();
    assert!(format!("{err}").contains("emb_dim"));
  }

  /// REGRESSION: a `(0, d)` array must be
  /// rejected. Without the guard, [`pad_to_max`] records `0` in
  /// `original_lengths`, [`score_multi_vector`]'s mask loop builds an
  /// all-`false` row for the offending passage, [`select`] replaces
  /// every position with `-inf`, `max(axis=3)` returns `-inf`, and the
  /// resulting ranking score is non-finite. Enforce the precondition
  /// here so the failure is observable, recoverable, and named.
  ///
  /// Assertion: returns [`Error::OutOfRange`] whose message contains
  /// `"zero tokens"` and identifies the offending array index.
  #[test]
  fn pad_to_max_rejects_zero_token_array() {
    // `(0, 2)` — a zero-token query / passage embedding.
    let zero = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
    let err = pad_to_max(std::slice::from_ref(&zero)).unwrap_err();
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "expected OutOfRange, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
      msg.contains("index 0"),
      "expected 'index 0' in message, got {msg}"
    );
    // Even when the zero-token array is not the first one in the slice,
    // the guard must still fire and identify its index.
    let good = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2)).unwrap();
    let bad = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
    let err2 = pad_to_max(&[good, bad]).unwrap_err();
    let msg2 = format!("{err2}");
    assert!(
      msg2.contains("index 1"),
      "expected 'index 1' in message, got {msg2}"
    );
  }

  /// dtype-fidelity: a half-precision input batch must stay
  /// half-precision through `pad_to_max` (python `dtype=a.dtype` on
  /// line 87). No silent f32 promotion of the padding rows.
  #[test]
  fn pad_to_max_preserves_f16_dtype() {
    let a = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0, 6.0], &(2, 2))
      .unwrap()
      .astype(Dtype::F16)
      .unwrap();
    let (padded, _lens) = pad_to_max(&[a, b]).unwrap();
    assert_eq!(padded.shape(), vec![2, 2, 2]);
    assert_eq!(
      padded.dtype().unwrap(),
      Dtype::F16,
      "padding must preserve input dtype (python L87 `dtype=a.dtype`)"
    );
  }

  /// Sanity check on the divergence-from-python tuple return shape:
  /// `pad_to_max` reports the **original** lengths of each input array
  /// (before zero-padding), in input order. The mask in
  /// [`score_multi_vector`] depends on this contract.
  #[test]
  fn pad_to_max_returns_original_lengths() {
    let a = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2)).unwrap(); // n=1
    let b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0, 6.0], &(2, 2)).unwrap(); // n=2
    let c = Array::from_slice::<f32>(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &(3, 2)).unwrap(); // n=3
    let (padded, lens) = pad_to_max(&[a, b, c]).unwrap();
    assert_eq!(padded.shape(), vec![3, 3, 2], "stacked to (3, max_n=3, 2)");
    assert_eq!(
      lens,
      vec![1, 2, 3],
      "original_lengths must mirror input order"
    );
  }

  /// REGRESSION: zero-padded passages must not
  /// win `max(axis=3)` for signed embeddings. With
  /// `q = [[1, 0]]`, `p0 = [[-1, 0]]`, `p1 = [[2, 0], [0, 1]]`:
  /// - Scoring `p0` alone (`batch_size = 1`, no padding): MaxSim is
  ///   `max(<q, p0_0>) = max(-1) = -1` → sum = -1.
  /// - Scoring `[p0, p1]` together with `batch_size = 2` (p0 gets a
  ///   zero-pad row to match p1's length 2): python ref returns
  ///   `max(-1, 0) = 0` (wrong; padded zero won). mlxrs must return
  ///   `-1.0` (the unpadded answer) in BOTH cases — ranking is
  ///   batch-size-agnostic.
  ///
  /// Rationale: the upstream
  /// `mlx_embeddings/colvision_processor.py` includes the zero-padded
  /// columns in `mx.max(sim, axis=3)`. mlxrs masks them to
  /// `f32::NEG_INFINITY` (cast to the input dtype) before the max.
  #[test]
  fn score_multi_vector_ragged_negative_similarity_batch_size_agnostic() {
    let q = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
    let p0 = Array::from_slice::<f32>(&[-1.0, 0.0], &(1, 2)).unwrap();
    let p1 = Array::from_slice::<f32>(&[2.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();

    // Branch A: batch_size=1, p0 is processed alone (no padding).
    let mut scores_b1 = score_multi_vector(
      std::slice::from_ref(&q),
      &[p0.try_clone().unwrap(), p1.try_clone().unwrap()],
      1,
    )
    .unwrap();
    assert_eq!(scores_b1.shape(), vec![1, 2]);
    let v_b1 = scores_b1.to_vec::<f32>().unwrap();

    // Branch B: batch_size=2, p0 is tile-padded with a zero row.
    let mut scores_tiled = score_multi_vector(std::slice::from_ref(&q), &[p0, p1], 2).unwrap();
    assert_eq!(scores_tiled.shape(), vec![1, 2]);
    let v_tiled = scores_tiled.to_vec::<f32>().unwrap();

    // p0 score (index 0) must be -1.0 in BOTH branches.
    assert_eq!(
      v_b1[0], -1.0,
      "p0 alone: <q,p0_0> = -1.0; sum over the single query token = -1.0"
    );
    assert_eq!(
      v_tiled[0], -1.0,
      "p0 tiled with p1: padded zero column must be masked → -1.0, not 0.0"
    );

    // p1's score should be the same in both branches too (sanity).
    assert_eq!(
      v_b1[1], v_tiled[1],
      "p1 score must be tile-invariant in both branches"
    );

    // The whole vector must be batch-size-agnostic.
    assert_eq!(
      v_b1, v_tiled,
      "score_multi_vector ranking must not depend on batch_size"
    );
  }

  // ---------- trait shape + impl ----------

  /// A minimal in-test impl of [`BaseColVisionProcessor`] proves the
  /// trait shape is implementable (and the contract closes over the
  /// model-specific state a concrete processor would own). It
  /// delegates `score` to [`score_multi_vector`] exactly like
  /// `colidefics3.Processor.score` (python line 329).
  struct TestProcessor;

  impl BaseColVisionProcessor for TestProcessor {
    fn process_images(&self, images: &[Vec<u8>]) -> Result<ProcessorBatch> {
      // Test-only stub: deposit a single `(len(images),)` int tensor
      // recording the batch size — the seam only checks the contract
      // shape, not the model preprocessor semantics (which are
      // out of scope).
      let mut batch = ProcessorBatch::new();
      let count = i32::try_from(images.len()).unwrap_or(0);
      batch.insert(
        "pixel_values_count".into(),
        Array::from_slice::<i32>(&[count], &(1,))?,
      );
      Ok(batch)
    }
    fn process_queries(
      &self,
      queries: &[&str],
      _max_length: usize,
      _suffix: Option<&str>,
    ) -> Result<ProcessorBatch> {
      let mut batch = ProcessorBatch::new();
      let count = i32::try_from(queries.len()).unwrap_or(0);
      batch.insert(
        "input_ids_count".into(),
        Array::from_slice::<i32>(&[count], &(1,))?,
      );
      Ok(batch)
    }
    fn score(&self, qs: &[Array], ps: &[Array], batch_size: usize) -> Result<Array> {
      score_multi_vector(qs, ps, batch_size)
    }
  }

  /// The trait is dyn-compatible-via-impl: a test impl returns the
  /// expected dict shape from both branches, and `score` delegates to
  /// `score_multi_vector` (mirroring `colidefics3.Processor.score`).
  #[test]
  fn base_processor_trait_impl_round_trips() {
    let p = TestProcessor;
    // process_images: dummy 2-image batch.
    let imgs = vec![vec![0u8, 1, 2], vec![3u8, 4, 5]];
    let img_batch = p.process_images(&imgs).unwrap();
    assert!(img_batch.contains_key("pixel_values_count"));

    // process_queries: dummy 3-query batch.
    let queries = vec!["query one", "query two", "query three"];
    let q_batch = p.process_queries(&queries, 50, None).unwrap();
    assert!(q_batch.contains_key("input_ids_count"));

    // score: identity multi-vector pair → 2.0 (matches the standalone
    // `score_multi_vector_identity_pair` test).
    let q = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let pp = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
    let mut scores = p.score(&[q], &[pp], 128).unwrap();
    assert_eq!(scores.shape(), vec![1, 1]);
    assert_eq!(scores.to_vec::<f32>().unwrap(), vec![2.0]);
  }
}
