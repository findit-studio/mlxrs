//! Multimodal prompt-assembly core primitives — faithful 1:1 port of the
//! model-agnostic helpers in `mlx-vlm/mlx_vlm/prompt_utils.py` and the
//! reusable splice/mask patterns scattered across `mlx-vlm/mlx_vlm/utils.py`
//! (`prepare_inputs` text-chunk splice, lines ~1370–1392) and
//! `mlx-vlm/mlx_vlm/models/falcon_ocr/language.py::create_falcon_ocr_mask`
//! (lines ~120–149).
//!
//! Per-model chat templates (`MessageFormatter`/`MessageFormat` in
//! `prompt_utils.py`) and the per-model `merge_input_ids_with_image_features`
//! embedding-space splice are deliberately **out of scope** (the latter
//! operates on embeddings inside each model's forward pass; here we only
//! touch tokens + an attention mask). See the `project_no_per_model_arch_porting`
//! convention.
//!
//! ## API at a glance
//!
//! - [`locate_image_tokens`] — `&[u32] → Vec<(start, end)>`: find contiguous
//!   runs of an image placeholder token. Rust-idiomatic half-open ranges.
//!   Adjacent multi-image placeholders collapse into one run (faithful to
//!   the post-tokenization view; per-image separation lives in the
//!   assembly helper that knows the per-image slot width).
//! - [`insert_image_tokens`] — splice `image_count` runs of
//!   `[image_token_id; num_tokens_per_image]` replacing the FIRST contiguous
//!   run of `image_marker_id`. Mirrors the Python `_format_with_token`
//!   `prefix = token * num_images` pattern (one contiguous run of markers,
//!   one splice). Marker-required vs prepend-fallback behavior is
//!   caller-selected via [`MarkerPolicy`], matching the python per-formatter
//!   selection (`IMAGE_TOKEN` vs `PROMPT_WITH_IMAGE_TOKEN` families).
//!   Fallible — guards against overflow, marker-run length mismatch, and
//!   missing-marker under [`MarkerPolicy::Required`].
//! - [`build_multimodal_mask`] — `[1, 1, T, T]` bool attend-mask: causal
//!   everywhere except image-token positions WITHIN a single image span,
//!   which are bidirectional. Matches `create_falcon_ocr_mask`'s rank-4
//!   `(1, 1, S, S)` return contract.
//! - [`assemble_multimodal_prompt`] — end-to-end: splice + per-image span
//!   computation (preserves causal-across-images boundary) + mask.
//!
//! ## Mask semantics
//!
//! Returned mask is **bool**, shape `[1, 1, T, T]` (matches the upstream
//! `create_falcon_ocr_mask` rank-4 contract: `(batch=1, heads=1, S, S)`
//! broadcastable over batch and head axes by mlx's attention kernels), with
//! `true = attend` via `causal | same_image`. We deliberately keep `bool`
//! rather than the upstream's additive-float form because callers commonly
//! AND/OR multiple sub-masks before the final `where(attend, 0, -inf)` cast;
//! per-model adapters perform that cast at the attention layer.

use crate::{
  array::Array,
  error::{Error, Result, try_to_vec, try_with_capacity},
};

/// Inclusive-start / exclusive-end half-open token-index ranges marking
/// contiguous runs of an image placeholder token. One entry per image span
/// (a single image typically occupies one run of `num_tokens_per_image`
/// placeholders). Returned by [`locate_image_tokens`].
pub type ImageTokenSpans = Vec<(usize, usize)>;

/// Selects how [`insert_image_tokens`] and [`assemble_multimodal_prompt`]
/// behave when `image_count > 0` but no `image_marker_id` is present in
/// `text_tokens`.
///
/// The python reference selects this at the formatter level (per
/// `MessageFormat` in `prompt_utils.py`), not per-call: marker-required
/// formatters (e.g. `IMAGE_TOKEN`, `_format_with_token`) tokenize a
/// `<image>` marker into the prompt; the `PROMPT_WITH_IMAGE_TOKEN`
/// formatter (paligemma) instead prepends `"<image>" * num_images` directly
/// with no expected marker. Modeling that as an explicit `MarkerPolicy`
/// here forces the caller to declare intent at the call site, so a missing
/// marker under a marker-required template surfaces as a hard
/// `Error::ShapeMismatch` instead of silently corrupting prompt order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerPolicy {
  /// The input MUST contain an `image_marker_id` (a contiguous run of
  /// exactly `image_count` markers); missing markers under
  /// `image_count > 0` are an error. Mirrors the python
  /// `_format_with_token` / `IMAGE_TOKEN`-family formatters.
  Required,
  /// If no `image_marker_id` is found in the input, prepend the
  /// `image_count * num_tokens_per_image` placeholder run to the start of
  /// the prompt. Mirrors the python `PROMPT_WITH_IMAGE_TOKEN` /
  /// `PROMPT_WITH_START_IMAGE_TOKEN` formatters (e.g. paligemma).
  PrependIfAbsent,
}

/// Locate contiguous runs of `image_token_id` in `tokens`.
///
/// Returns one half-open `(start, end)` span per contiguous run.
/// `end - start` is the run length. The empty input yields an empty `Vec`.
///
/// Mirrors the per-batch `image_positions = input_ids == image_token_id`
/// bool-mask pattern that recurs throughout per-model `merge_input_ids_*`
/// implementations (e.g.
/// `mlx-vlm/mlx_vlm/models/qwen2_vl/qwen2_vl.py:94`), but returns the
/// compact run-encoding directly so callers can index spans without an
/// O(T) cumsum.
///
/// # Examples
///
/// ```
/// use mlxrs::vlm::prompt::locate_image_tokens;
///
/// // Two images, 3 placeholders each, surrounded by text.
/// let tokens = vec![10_u32, 99, 99, 99, 20, 30, 99, 99, 99, 40];
/// let spans = locate_image_tokens(&tokens, 99);
/// assert_eq!(spans, vec![(1, 4), (6, 9)]);
/// ```
pub fn locate_image_tokens(tokens: &[u32], image_token_id: u32) -> ImageTokenSpans {
  let mut spans = ImageTokenSpans::new();
  let mut i = 0;
  while i < tokens.len() {
    if tokens[i] == image_token_id {
      let start = i;
      // Walk to the end of the contiguous run.
      while i < tokens.len() && tokens[i] == image_token_id {
        i += 1;
      }
      spans.push((start, i));
    } else {
      i += 1;
    }
  }
  spans
}

/// Splice `image_count` runs of `num_tokens_per_image` `image_token_id`
/// placeholders into `text_tokens` at the FIRST contiguous run of
/// `image_marker_id` (the whole run is REPLACED, not preserved). The
/// `policy` argument selects the markerless behavior — see [`MarkerPolicy`].
///
/// Behavior matrix:
/// - `image_count == 0` → returns a copy of `text_tokens` unchanged (any
///   markers, if any, are left in place; faithful to the Python no-image
///   pass-through path).
/// - marker run present → the entire FIRST contiguous run of markers is
///   replaced by `image_count * num_tokens_per_image` copies of
///   `image_token_id`. This matches the `MessageFormatter::_format_with_token`
///   `prefix = token * num_images` pattern in `prompt_utils.py:350-371`,
///   which emits N adjacent markers as a single prefix that tokenizes to
///   one contiguous run of marker tokens. The contiguous run length MUST
///   equal `image_count` (else `Error::ShapeMismatch`).
/// - marker absent + `MarkerPolicy::PrependIfAbsent` → the placeholder run
///   is PREPENDED to `text_tokens` (mirrors the
///   `MessageFormat::PROMPT_WITH_IMAGE_TOKEN` `"<image>" * num_images + prompt`
///   path in `prompt_utils.py:265-267`).
/// - marker absent + `MarkerPolicy::Required` → `Error::ShapeMismatch`.
///   Fails closed against chat-template / tokenizer-version drift that
///   would silently rewrite prompt order under a marker-required
///   formatter.
///
/// Mirrors the splice pattern in
/// `mlx-vlm/mlx_vlm/utils.py::prepare_inputs` (lines ~1370–1387:
/// `ids = chunks[0] + [image_token_index] + chunks[1]`), generalized so each
/// image expands to a run of `num_tokens_per_image` placeholders rather than
/// a single token (the Qwen-VL/LLaVA-Next pattern, where the vision tower
/// emits `num_tokens_per_image` features per image).
///
/// # Errors
///
/// - `Error::ShapeMismatch` if `image_count * num_tokens_per_image` overflows
///   `usize`, or if the resulting buffer capacity (text length + placeholder
///   delta) overflows.
/// - `Error::ShapeMismatch` if `text_tokens` contains an additional
///   non-contiguous `image_marker_id` AFTER the first run (the python
///   reference's `prepare_inputs` splice has no support for multiple
///   non-adjacent marker positions).
/// - `Error::ShapeMismatch` if the contiguous-marker-run length differs
///   from `image_count` (chat-template producer should emit exactly
///   `image_count` adjacent markers).
/// - `Error::ShapeMismatch` if `policy == MarkerPolicy::Required` and no
///   `image_marker_id` is found while `image_count > 0`.
///
/// # Examples
///
/// ```
/// use mlxrs::vlm::prompt::{insert_image_tokens, MarkerPolicy};
///
/// // Marker present: 1 image, 3 tokens per image → marker replaced by `[99,99,99]`.
/// let text = vec![1_u32, 2, 7, 3, 4]; // marker=7
/// let out = insert_image_tokens(&text, 1, 7, 99, 3, MarkerPolicy::Required).unwrap();
/// assert_eq!(out, vec![1, 2, 99, 99, 99, 3, 4]);
///
/// // Multi-image: a contiguous run of N markers (the `"<image>" * N` chat-
/// // template pattern) is consumed as one unit; replaced with
/// // `image_count * num_tokens_per_image` placeholders.
/// let text = vec![1_u32, 7, 7, 2];
/// let out = insert_image_tokens(&text, 2, 7, 99, 3, MarkerPolicy::Required).unwrap();
/// assert_eq!(out, vec![1, 99, 99, 99, 99, 99, 99, 2]);
///
/// // PrependIfAbsent: no marker in text → prepend
/// // (PROMPT_WITH_IMAGE_TOKEN-family formatters).
/// let text = vec![1_u32, 2, 3];
/// let out = insert_image_tokens(&text, 1, 7, 99, 3, MarkerPolicy::PrependIfAbsent).unwrap();
/// assert_eq!(out, vec![99, 99, 99, 1, 2, 3]);
///
/// // Required: no marker + image_count>0 → error (fails closed against
/// // chat-template drift).
/// let text = vec![1_u32, 2, 3];
/// let err = insert_image_tokens(&text, 1, 7, 99, 3, MarkerPolicy::Required).unwrap_err();
/// assert!(format!("{err}").contains("no image_marker_id"));
///
/// // Zero images: passthrough (marker left in place).
/// let text = vec![1_u32, 7, 2];
/// let out = insert_image_tokens(&text, 0, 7, 99, 3, MarkerPolicy::Required).unwrap();
/// assert_eq!(out, vec![1, 7, 2]);
/// ```
pub fn insert_image_tokens(
  text_tokens: &[u32],
  image_count: usize,
  image_marker_id: u32,
  image_token_id: u32,
  num_tokens_per_image: usize,
  policy: MarkerPolicy,
) -> Result<Vec<u32>> {
  if image_count == 0 {
    return try_to_vec(text_tokens);
  }

  // Reject `image_count > 0 && num_tokens_per_image == 0` — a degenerate
  // model/config state where each image expands to zero placeholders.
  // Silently accepting it would emit a text-only prompt and drop the
  // caller's images on the floor (downstream generation proceeds with the
  // images invisible to attention). Fail closed.
  if num_tokens_per_image == 0 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "insert_image_tokens: num_tokens_per_image=0 with image_count={image_count} > 0 \
         would silently drop {image_count} image(s) — config/model state is degenerate"
      ),
    });
  }

  // Checked placeholder total — guards against caller-supplied counts that
  // overflow `usize` (`saturating_mul` would silently cap at `usize::MAX`
  // and then drive an unbounded `Vec::with_capacity`, which aborts on OOM).
  let placeholder_total = image_count
    .checked_mul(num_tokens_per_image)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!(
        "insert_image_tokens: image_count * num_tokens_per_image overflows usize \
         (image_count={image_count}, num_tokens_per_image={num_tokens_per_image})"
      ),
    })?;

  // First-marker-run splice: consume the entire FIRST CONTIGUOUS RUN of
  // markers (mirrors the `"<image>" * N` chat-template prefix pattern,
  // which tokenizes to one contiguous run of `image_marker_id` tokens).
  // Replace with `placeholder_total` copies of `image_token_id`. Any
  // additional non-contiguous markers in the tail trigger a hard error
  // because the python reference's `prepare_inputs` only supports a single
  // marker position (its `prompt.split("<image>")` is hardcoded to 2
  // chunks) and silently leaving them would corrupt vision-feature
  // alignment — especially when `image_marker_id == image_token_id`, where
  // residual markers would silently inflate the placeholder count.
  if let Some(run_start) = text_tokens.iter().position(|&t| t == image_marker_id) {
    let run_end = text_tokens[run_start..]
      .iter()
      .position(|&t| t != image_marker_id)
      .map_or(text_tokens.len(), |off| run_start + off);
    let run_len = run_end - run_start;

    // Reject extra markers after the consumed run.
    if text_tokens[run_end..].contains(&image_marker_id) {
      return Err(Error::ShapeMismatch {
        message: format!(
          "insert_image_tokens: input contains a non-contiguous additional \
           image_marker_id ({image_marker_id}) after the first run \
           [{run_start}..{run_end}); the splice supports at most one \
           contiguous marker run"
        ),
      });
    }

    // Reject contiguous-run length mismatch with `image_count`. Faithful to
    // `_format_with_token` in `prompt_utils.py:350-371`: the producer emits
    // EXACTLY `num_images` adjacent markers (`prefix = token * num_images`).
    // A run length other than `image_count` indicates a producer/caller bug
    // (chat-template version skew, miscounted images, etc.) — silently
    // accepting it would either delete extra markers (under-count) or
    // duplicate image features (over-count), corrupting vision-feature
    // alignment without surfacing the upstream defect.
    if run_len != image_count {
      return Err(Error::ShapeMismatch {
        message: format!(
          "insert_image_tokens: contiguous marker run length {run_len} does not match \
           image_count {image_count} (the chat-template producer should emit exactly \
           `marker * image_count` adjacent markers; mismatch suggests caller/template \
           skew)"
        ),
      });
    }

    // Capacity = text.len() + placeholder_total - run_len (replacing the
    // entire run), with checked arithmetic to surface overflow as a
    // recoverable error rather than panic-on-grow / OOM-abort.
    let cap = text_tokens
      .len()
      .checked_add(placeholder_total)
      .and_then(|n| n.checked_sub(run_len))
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "insert_image_tokens: text_len + placeholder_total - run_len overflows usize \
           (text_len={}, placeholder_total={placeholder_total}, run_len={run_len})",
          text_tokens.len()
        ),
      })?;
    // Recoverable reservation (Codex VLM-8 R3F1): a huge non-overflowing
    // `cap` (large `num_tokens_per_image` × `image_count`) would abort
    // the process via `Vec::with_capacity`; `try_reserve_exact` surfaces
    // it as `Error::OutOfMemory`. `vlm_generate` calls this before any
    // other recoverable boundary.
    let mut out: Vec<u32> = try_with_capacity(cap)?;
    out.extend_from_slice(&text_tokens[..run_start]);
    out.extend(std::iter::repeat_n(image_token_id, placeholder_total));
    out.extend_from_slice(&text_tokens[run_end..]);
    Ok(out)
  } else {
    // No marker: behavior is caller-selected via MarkerPolicy.
    if policy == MarkerPolicy::Required {
      return Err(Error::ShapeMismatch {
        message: format!(
          "insert_image_tokens: no image_marker_id ({image_marker_id}) found in \
           text_tokens but image_count={image_count} > 0 under MarkerPolicy::Required \
           (chat-template / tokenizer drift detected; pass MarkerPolicy::PrependIfAbsent \
           if the model uses the PROMPT_WITH_IMAGE_TOKEN-family formatter)"
        ),
      });
    }
    // PrependIfAbsent → PROMPT_WITH_IMAGE_TOKEN path.
    let cap = text_tokens
      .len()
      .checked_add(placeholder_total)
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "insert_image_tokens: text_len + placeholder_total overflows usize \
           (text_len={}, placeholder_total={placeholder_total})",
          text_tokens.len()
        ),
      })?;
    // Recoverable reservation (Codex VLM-8 R3F1) — see the marker-present
    // branch above.
    let mut out: Vec<u32> = try_with_capacity(cap)?;
    out.extend(std::iter::repeat_n(image_token_id, placeholder_total));
    out.extend_from_slice(text_tokens);
    Ok(out)
  }
}

/// Build a `[1, 1, T, T]` bool attention mask for a multimodal prompt of
/// length `seq_len` containing `image_spans` (typically the output of
/// [`locate_image_tokens`] or the per-image splits emitted by
/// [`assemble_multimodal_prompt`]).
///
/// Semantics (`true = attend`, faithful to
/// `mlx-vlm/mlx_vlm/models/falcon_ocr/language.py::create_falcon_ocr_mask`
/// at lines ~120–149):
/// - text→text: causal (lower-triangular, `q >= k` attends).
/// - text→image, image→text: causal.
/// - image→image WITHIN the same span: bidirectional (attend regardless of
///   order — same-image patches see each other).
/// - image→image ACROSS different spans: causal (no leak from a later image
///   back to an earlier query — generation order is preserved).
///
/// Spans must be:
/// - half-open `(start, end)` with `start < end`,
/// - non-overlapping,
/// - bounded by `seq_len` (i.e. `end <= seq_len`).
///
/// Violations return `Error::ShapeMismatch` with a descriptive message;
/// no panic.
///
/// Shape `[1, 1, T, T]` matches the upstream `create_falcon_ocr_mask` return
/// rank `(batch=1, heads=1, S, S)` and is broadcastable over batch and head
/// axes by mlx's attention kernels. The per-model attention layer can
/// convert to an additive float mask (`where(attend, 0, -inf)`) at the
/// attention call site if needed.
///
/// # Errors
///
/// - `Error::ShapeMismatch` if any span is empty, out of bounds, or overlaps
///   another span; or if `seq_len.checked_mul(seq_len)` overflows `usize`;
///   or if `seq_len` exceeds `i32::MAX` (mlx dimensions are signed 32-bit).
///
/// # Examples
///
/// ```
/// use mlxrs::vlm::prompt::build_multimodal_mask;
///
/// // 4 text tokens only → pure causal lower-triangular.
/// let mut mask = build_multimodal_mask(4, &[]).unwrap();
/// assert_eq!(mask.shape(), vec![1, 1, 4, 4]);
/// let v = mask.to_vec::<bool>().unwrap();
/// // Row q, col k: attend iff k <= q.
/// for q in 0..4 {
///   for k in 0..4 {
///     assert_eq!(v[q * 4 + k], k <= q);
///   }
/// }
/// ```
pub fn build_multimodal_mask(seq_len: usize, image_spans: &[(usize, usize)]) -> Result<Array> {
  // The non-chunked / single-forward case is the `past_len == 0`
  // specialization of the offset-aware builder. Delegate so the
  // validation + block-id logic lives in one place.
  build_multimodal_mask_with_past(seq_len, 0, image_spans)
}

/// Offset-aware variant of [`build_multimodal_mask`] for **chunked prefill**:
/// build a `[1, 1, seq_len, past_len + seq_len]` bool attention mask for a
/// chunk of `seq_len` query positions whose keys are the `past_len` already
/// cached tokens PLUS this chunk's own `seq_len` tokens.
///
/// `image_spans` are **chunk-local** — half-open `(start, end)` ranges in
/// `[0, seq_len)` identifying image runs WITHIN this chunk (the caller shifts
/// absolute spans by the chunk's start offset). Span-aware chunking
/// guarantees no image span is split across a chunk boundary, so every
/// query's bidirectional-within-image partners are in the same chunk — the
/// past keys never participate in a span's bidirectional block.
///
/// Mask semantics for query `q` (chunk-local `0..seq_len`, absolute
/// `past_len + q`) and key `k` (`0..past_len + seq_len`):
/// - **`k < past_len` (a cached past key)**: always attend. Past tokens are
///   strictly before this chunk (`k < past_len <= past_len + q`), so the
///   causal rule admits them unconditionally; no past key shares a
///   bidirectional image block with a current-chunk query (spans aren't
///   split).
/// - **`k >= past_len` (a current-chunk key)**: chunk-local `k' = k -
///   past_len`; attend iff `k' <= q` (causal) OR `q` and `k'` are in the
///   same chunk-local image span (bidirectional-within-image).
///
/// With `past_len == 0` this is byte-identical to the original
/// `[1, 1, seq_len, seq_len]` mask.
///
/// # Errors
///
/// Same contract as [`build_multimodal_mask`] (empty/out-of-bounds/
/// overlapping spans; `seq_len`/`past_len` exceeding `i32::MAX`; total
/// buffer overflow), evaluated against the chunk-local `seq_len`.
pub fn build_multimodal_mask_with_past(
  seq_len: usize,
  past_len: usize,
  image_spans: &[(usize, usize)],
) -> Result<Array> {
  let total_keys = past_len
    .checked_add(seq_len)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!(
        "build_multimodal_mask_with_past: past_len ({past_len}) + seq_len ({seq_len}) overflows usize"
      ),
    })?;

  // Empty chunk: faithful zero-query [1, 1, 0, past_len] array. Non-empty
  // spans on a zero-length chunk is an inconsistent state — fail closed.
  if seq_len == 0 {
    if !image_spans.is_empty() {
      return Err(Error::ShapeMismatch {
        message: format!(
          "build_multimodal_mask_with_past: seq_len=0 but image_spans has {} entry(s); \
           empty chunk cannot contain any image span",
          image_spans.len()
        ),
      });
    }
    return Array::from_slice::<bool>(&[], &(1_usize, 1_usize, 0_usize, total_keys));
  }

  // mlx dimensions are signed 32-bit; reject oversized dims BEFORE any
  // host-side allocation.
  if total_keys > i32::MAX as usize {
    return Err(Error::ShapeMismatch {
      message: format!(
        "build_multimodal_mask_with_past: past_len + seq_len ({total_keys}) exceeds i32::MAX (mlx dimension limit)"
      ),
    });
  }

  // Validate chunk-local spans (start<end, end<=seq_len, ordered/non-overlapping).
  let mut sorted: Vec<(usize, usize)> = try_to_vec(image_spans)?;
  sorted.sort_unstable_by_key(|&(s, _)| s);
  let mut prev_end = 0usize;
  for &(s, e) in &sorted {
    if s >= e {
      return Err(Error::ShapeMismatch {
        message: format!(
          "build_multimodal_mask_with_past: image span ({s}, {e}) is empty (start>=end)"
        ),
      });
    }
    if e > seq_len {
      return Err(Error::ShapeMismatch {
        message: format!(
          "build_multimodal_mask_with_past: chunk-local image span ({s}, {e}) end exceeds seq_len {seq_len}"
        ),
      });
    }
    if s < prev_end {
      return Err(Error::ShapeMismatch {
        message: format!(
          "build_multimodal_mask_with_past: image span ({s}, {e}) overlaps the previous span ending at {prev_end}"
        ),
      });
    }
    prev_end = e;
  }

  // Total buffer size with overflow guard: seq_len rows × total_keys cols.
  let total = seq_len
    .checked_mul(total_keys)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!(
        "build_multimodal_mask_with_past: seq_len*({past_len}+{seq_len}) overflows usize"
      ),
    })?;

  // block_id[i] = 1-indexed chunk-local image span index for chunk position
  // i; 0 = not in any image. Length seq_len (chunk-local). Allocated
  // fallibly for the same reason as `buf` below: these are the TWO
  // sequence-scaled buffers (`block_id` is O(seq_len); `buf` is
  // O(seq_len · total_keys) — the dominant allocation, up to MBs), so a
  // large valid chunk would otherwise abort in `vec![0u32; seq_len]`
  // before the recoverable `buf.try_reserve_exact` could report OOM
  // (Codex VLM-8 R4F1). The small auxiliaries here (`sorted`, a clone of
  // `image_spans` — O(num_images), a handful of `(usize,usize)` pairs)
  // follow the crate's standard infallible-`Vec` idiom: they cannot
  // realistically OOM (model image counts are small constants), and a
  // blanket try_reserve on every Vec would diverge from the rest of
  // mlxrs + the python/swift references without a real threat-model gain
  // (see VLM-9 in docs/rust-golden-standard-followups.md for the
  // coordinated allocation-policy deferral).
  let mut block_id: Vec<u32> = try_with_capacity(seq_len)?;
  block_id.resize(seq_len, 0);
  for (idx, &(s, e)) in sorted.iter().enumerate() {
    let block = (idx + 1) as u32;
    for slot in block_id.iter_mut().take(e).skip(s) {
      *slot = block;
    }
  }

  // Fill row-major: row=q (chunk-local), col=k (absolute over past+chunk).
  // Past keys (k < past_len) are unconditionally attended; current-chunk
  // keys use chunk-local causal + same-image-span.
  //
  // Recoverable reservation (Codex VLM-8 R1F2): on late chunks
  // `total = seq_len * (past_len + seq_len)` grows with the cached
  // context, so a long prompt's mask can be large. `try_reserve_exact`
  // surfaces an allocator failure as a recoverable `Error::OutOfMemory`
  // instead of the `Vec::with_capacity` abort. (The mask is dense by
  // contract here; a symbolic causal-base + sparse image-overlay
  // representation is the documented future optimization in VLM-8 — it
  // does not change this function's observable output.)
  let mut buf: Vec<bool> = Vec::new();
  buf
    .try_reserve_exact(total)
    .map_err(|_| Error::OutOfMemory)?;
  for (q, &q_blk) in block_id.iter().enumerate() {
    for k in 0..total_keys {
      let attend = if k < past_len {
        true
      } else {
        let k_local = k - past_len;
        let causal = k_local <= q;
        let same_image_span = q_blk != 0 && q_blk == block_id[k_local];
        causal || same_image_span
      };
      buf.push(attend);
    }
  }

  Array::from_slice::<bool>(&buf, &(1_usize, 1_usize, seq_len, total_keys))
}

/// End-to-end multimodal prompt assembly result.
///
/// Bundles the spliced token sequence, the located image-token spans, and
/// the multimodal attention mask — the exact triple a downstream
/// `VisionLanguageModel` forward pass consumes.
#[derive(Debug)]
pub struct MultimodalPrompt {
  /// Tokens with image placeholders spliced in (output of
  /// [`insert_image_tokens`]).
  pub tokens: Vec<u32>,
  /// Half-open `(start, end)` image-token runs in [`Self::tokens`] (output
  /// of [`locate_image_tokens`]).
  pub image_spans: ImageTokenSpans,
  /// `[1, 1, T, T]` bool attention mask (output of
  /// [`build_multimodal_mask`]), `true = attend`.
  pub attention_mask: Array,
}

/// Compose the three primitives end-to-end.
///
/// Steps (matches the mlx-vlm `prepare_inputs` + per-model-attn-mask flow):
/// 1. Splice via [`insert_image_tokens`].
/// 2. Compute per-image spans `[(base + i*n, base + (i+1)*n)]` directly from
///    the splice's base offset (NOT via [`locate_image_tokens`], which would
///    collapse adjacent images into a single contiguous run and erase the
///    per-image causal boundary).
/// 3. Build mask via [`build_multimodal_mask`].
///
/// `text_tokens` is the pre-tokenized prompt (typically from
/// `Tokenizer::apply_chat_template`, available behind the
/// `tokenizer-chat` feature in [`crate::tokenizer::chat`]).
///
/// `policy` selects the markerless behavior — see [`MarkerPolicy`]:
/// `Required` for marker-required formatters (most models), `PrependIfAbsent`
/// for the `PROMPT_WITH_IMAGE_TOKEN`-family (e.g. paligemma).
///
/// Per-image span preservation matters when `image_count >= 2`: each image
/// gets its own bidirectional-within-image block, and images are causal
/// across each other (no leak from a later image back to an earlier image's
/// query positions). This mirrors the per-block delimiting that
/// `create_falcon_ocr_mask` achieves via explicit `soi/eoi` sentinel tokens
/// in the falcon_ocr language model.
///
/// # Errors
///
/// Propagates from [`insert_image_tokens`] (overflow on placeholder
/// arithmetic, marker-run length mismatch, missing-marker under
/// `MarkerPolicy::Required`) and [`build_multimodal_mask`] (T*T overflow).
/// Also fails fast if the final assembled length exceeds `i32::MAX` (mlx
/// dimension limit), BEFORE the splice allocates a host buffer.
///
/// # Examples
///
/// ```
/// use mlxrs::vlm::prompt::{assemble_multimodal_prompt, MarkerPolicy};
///
/// // Two images, 3 tokens per image, marker=7, image_token=99.
/// // Chat-template producer emits 2 adjacent markers (`marker * num_images`);
/// // per-image span separation: (2,5) and (5,8), NOT one collapsed (2,8).
/// let text = vec![1_u32, 2, 7, 7, 3];
/// let p = assemble_multimodal_prompt(&text, 2, 7, 99, 3, MarkerPolicy::Required).unwrap();
/// assert_eq!(p.tokens, vec![1, 2, 99, 99, 99, 99, 99, 99, 3]);
/// assert_eq!(p.image_spans, vec![(2, 5), (5, 8)]);
/// assert_eq!(p.attention_mask.shape(), vec![1, 1, 9, 9]);
/// ```
pub fn assemble_multimodal_prompt(
  text_tokens: &[u32],
  image_count: usize,
  image_marker_id: u32,
  image_token_id: u32,
  num_tokens_per_image: usize,
  policy: MarkerPolicy,
) -> Result<MultimodalPrompt> {
  // Reject the degenerate zero-width image config before doing any work
  // (matches `insert_image_tokens`'s guard; surfacing this from the
  // assembler avoids the silent text-only-prompt fail-open path).
  if image_count > 0 && num_tokens_per_image == 0 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "assemble_multimodal_prompt: num_tokens_per_image=0 with image_count={image_count} > 0 \
         would silently drop {image_count} image(s) — config/model state is degenerate"
      ),
    });
  }

  // Determine the splice's base offset and the first contiguous marker
  // run's length BEFORE building the buffer so we can:
  //   (a) emit per-image spans without re-scanning the spliced tokens, and
  //   (b) precompute final_len precisely (text.len + placeholder_total -
  //       run_len), used for the early i32::MAX rejection below.
  // Mirrors `insert_image_tokens`'s marker-vs-prepend branch.
  let (base, marker_run_len) = if image_count == 0 {
    (0_usize, 0_usize)
  } else if let Some(run_start) = text_tokens.iter().position(|&t| t == image_marker_id) {
    let run_end = text_tokens[run_start..]
      .iter()
      .position(|&t| t != image_marker_id)
      .map_or(text_tokens.len(), |off| run_start + off);
    (run_start, run_end - run_start)
  } else {
    (0_usize, 0_usize)
  };

  // EARLY i32::MAX check on the final assembled length: mlx dimensions are
  // signed 32-bit and any T > i32::MAX would be rejected by
  // `build_multimodal_mask` anyway — but checking before
  // `insert_image_tokens` avoids spending a multi-GB host allocation on a
  // request that is guaranteed to fail downstream. Final length follows
  // `insert_image_tokens`'s two branches: text.len + placeholder_total -
  // marker_run_len (marker-run replaced) or text.len + placeholder_total
  // (prepend).
  let placeholder_total = image_count
    .checked_mul(num_tokens_per_image)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!(
        "assemble_multimodal_prompt: image_count * num_tokens_per_image overflows usize \
         (image_count={image_count}, num_tokens_per_image={num_tokens_per_image})"
      ),
    })?;
  let final_len = if placeholder_total == 0 {
    text_tokens.len()
  } else {
    text_tokens
      .len()
      .checked_add(placeholder_total)
      .and_then(|n| n.checked_sub(marker_run_len))
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "assemble_multimodal_prompt: text_len + placeholder_total - marker_run_len overflows usize \
           (text_len={}, placeholder_total={placeholder_total}, marker_run_len={marker_run_len})",
          text_tokens.len()
        ),
      })?
  };
  if final_len > i32::MAX as usize {
    return Err(Error::ShapeMismatch {
      message: format!(
        "assemble_multimodal_prompt: final assembled length {final_len} exceeds i32::MAX \
         (mlx dimension limit); reject before allocating splice buffer"
      ),
    });
  }

  let tokens = insert_image_tokens(
    text_tokens,
    image_count,
    image_marker_id,
    image_token_id,
    num_tokens_per_image,
    policy,
  )?;

  // Per-image spans `[(base + i*n, base + (i+1)*n)]`. Empty when
  // image_count == 0 or num_tokens_per_image == 0 (degenerate cases: no
  // placeholders were emitted). Uses checked arithmetic to surface
  // pathological inputs as recoverable errors rather than panic.
  let mut image_spans = ImageTokenSpans::new();
  if image_count > 0 && num_tokens_per_image > 0 {
    image_spans = try_with_capacity(image_count)?;
    for i in 0..image_count {
      let start = base
        .checked_add(
          i.checked_mul(num_tokens_per_image)
            .ok_or_else(|| Error::ShapeMismatch {
              message: format!(
                "assemble_multimodal_prompt: i * num_tokens_per_image overflows usize \
               (i={i}, num_tokens_per_image={num_tokens_per_image})"
              ),
            })?,
        )
        .ok_or_else(|| Error::ShapeMismatch {
          message: format!(
            "assemble_multimodal_prompt: base + i*num_tokens_per_image overflows usize \
             (base={base}, i={i}, num_tokens_per_image={num_tokens_per_image})"
          ),
        })?;
      let end = start
        .checked_add(num_tokens_per_image)
        .ok_or_else(|| Error::ShapeMismatch {
          message: format!(
            "assemble_multimodal_prompt: start + num_tokens_per_image overflows usize \
             (start={start}, num_tokens_per_image={num_tokens_per_image})"
          ),
        })?;
      image_spans.push((start, end));
    }
  }

  let attention_mask = build_multimodal_mask(tokens.len(), &image_spans)?;
  Ok(MultimodalPrompt {
    tokens,
    image_spans,
    attention_mask,
  })
}
