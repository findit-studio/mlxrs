//! Multimodal prompt-assembly core primitives — faithful 1:1 port of the
//! model-agnostic helpers in `mlx-vlm/mlx_vlm/prompt_utils.py` and the
//! reusable splice/mask patterns scattered across `mlx-vlm/mlx_vlm/utils.py`
//! (`prepare_inputs` text-chunk splice, lines ~1370–1392) and
//! `mlx-vlm/mlx_vlm/models/falcon_ocr/language.py::create_falcon_ocr_mask`
//! (lines ~120–149).
//!
//! ## Chat-format builder (`MessageFormat` + `MessageFormatter`)
//!
//! The per-model chat-format selection layer (`MessageFormat` enum,
//! `MODEL_CONFIG` per-family map, `SINGLE_IMAGE_ONLY_MODELS` set,
//! `MessageBuilder`, `MessageFormatter`, `get_message_json`) is now ported
//! 1:1 from `mlx-vlm/mlx_vlm/prompt_utils.py` (the 15-variant `MessageFormat`
//! enum at lines 6–23 + the ~60-entry `MODEL_CONFIG` dict at lines 27–89 +
//! the formatter dispatch at lines 192–441). This is declarative
//! configuration data + a small dispatch — NOT per-model architecture.
//!
//! What stays out of scope: the per-model `merge_input_ids_with_image_features`
//! embedding-space splice (operates on embeddings inside each model's forward
//! pass) and per-model architecture impls — see the
//! `project_no_per_model_arch_porting` convention.
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
//! - [`MessageFormat`] — 15-variant enum mirroring the Python
//!   `MessageFormat(Enum)` at lines 6–23 (one-to-one).
//! - [`MODEL_CONFIG`] — `&[(model_type, MessageFormat)]` per-family map
//!   mirroring the Python `MODEL_CONFIG` dict at lines 27–89.
//! - [`SINGLE_IMAGE_ONLY_MODELS`] — model-type allow-list mirroring the
//!   Python set at lines 92–100.
//! - [`MessageBuilder`] — content-item constructors mirroring the Python
//!   `MessageBuilder` class at lines 151–189 (text / content / image /
//!   image_url / audio / video).
//! - [`MessageFormatter`] — per-model dispatch mirroring the Python class
//!   at lines 192–441.
//! - [`get_message_json`] — top-level helper mirroring the Python free
//!   function at lines 444–480.
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
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, EmptyInputPayload, Error,
    InvariantViolationPayload, LengthMismatchPayload, MissingFieldPayload, OutOfRangePayload,
    Result, try_to_vec, try_with_capacity,
  },
};

/// Defensive upper bound on the number of items (image / audio / video
/// entries) that [`MessageFormatter::format_message`] and
/// [`get_message_json`] will allocate for in a single call. Caller-
/// supplied `num_images` / `num_audios` / `video.len()` above this cap
/// return `Error::Backend` with an actionable cap message; below it the
/// allocation goes through fallible `try_reserve_exact` (so even within
/// the cap a hostile input fails recoverably rather than aborting).
///
/// 1024 was chosen as a multi-image / multi-modal upper bound that is
/// well beyond any realistic VLM chat-turn (the largest documented
/// HF VLM image batch we've seen is ~64) but small enough that even a
/// pathological caller (e.g. an attacker-controlled token count) cannot
/// drive an OOM by walking up to the cap on every dispatch site.
pub const MAX_MESSAGE_FORMAT_ITEMS: usize = 1024;

/// Validate a caller-controlled item count for a `format_*` helper
/// against [`MAX_MESSAGE_FORMAT_ITEMS`].
///
/// Returns `Err(Error::Backend)` (cap-exceeded — caller-supplied count
/// is too large to allocate) when `count > MAX_MESSAGE_FORMAT_ITEMS`,
/// else `Ok(())`. `label` is interpolated into the error message
/// (e.g. `"num_images"`, `"num_audios"`, `"video.len()"`) for caller
/// triage.
fn check_format_count(count: usize, label: &'static str, _model_name: &str) -> Result<()> {
  if count > MAX_MESSAGE_FORMAT_ITEMS {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      label,
      "MAX_MESSAGE_FORMAT_ITEMS",
      MAX_MESSAGE_FORMAT_ITEMS as u64,
      count as u64,
    )));
  }
  Ok(())
}

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
/// `Error::MissingField` instead of silently corrupting prompt order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
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

impl MarkerPolicy {
  /// Lowercase string tag.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Required => "required",
      Self::PrependIfAbsent => "prepend_if_absent",
    }
  }
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
///   equal `image_count` (else `Error::LengthMismatch`).
/// - marker absent + `MarkerPolicy::PrependIfAbsent` → the placeholder run
///   is PREPENDED to `text_tokens` (mirrors the
///   `MessageFormat::PROMPT_WITH_IMAGE_TOKEN` `"<image>" * num_images + prompt`
///   path in `prompt_utils.py:265-267`).
/// - marker absent + `MarkerPolicy::Required` → `Error::MissingField`.
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
/// - `Error::ArithmeticOverflow` if `image_count * num_tokens_per_image`
///   overflows `usize`, or if the resulting buffer capacity (text length +
///   placeholder delta) overflows.
/// - `Error::InvariantViolation` if `text_tokens` contains an additional
///   non-contiguous `image_marker_id` AFTER the first run (the python
///   reference's `prepare_inputs` splice has no support for multiple
///   non-adjacent marker positions).
/// - `Error::LengthMismatch` if the contiguous-marker-run length differs
///   from `image_count` (chat-template producer should emit exactly
///   `image_count` adjacent markers).
/// - `Error::MissingField` if `policy == MarkerPolicy::Required` and no
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
/// // chat-template drift). Surfaced as `Error::MissingField` whose `field`
/// // names the absent marker.
/// let text = vec![1_u32, 2, 3];
/// let err = insert_image_tokens(&text, 1, 7, 99, 3, MarkerPolicy::Required).unwrap_err();
/// assert!(matches!(err, mlxrs::Error::MissingField(_)));
/// assert!(format!("{err}").contains("image_marker_id"));
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
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "insert_image_tokens: num_tokens_per_image (with image_count > 0)",
      "must be > 0 — otherwise images would silently drop, config/model state is degenerate",
    )));
  }

  // Checked placeholder total — guards against caller-supplied counts that
  // overflow `usize` (`saturating_mul` would silently cap at `usize::MAX`
  // and then drive an unbounded `Vec::with_capacity`, which aborts on OOM).
  let placeholder_total = image_count
    .checked_mul(num_tokens_per_image)
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "insert_image_tokens: placeholder_total (image_count * num_tokens_per_image)",
        "usize",
        [
          ("image_count", image_count as u64),
          ("num_tokens_per_image", num_tokens_per_image as u64),
        ],
      ))
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
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "insert_image_tokens: image_marker_id occurrences (after the first contiguous run)",
        "must be 0 — the splice supports at most one contiguous marker run \
         (mirrors python prompt_utils' `prompt.split(\"<image>\")` 2-chunk contract)",
      )));
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
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "insert_image_tokens: contiguous marker run length vs image_count (the chat-template \
           producer should emit exactly `marker * image_count` adjacent markers; mismatch \
           suggests caller/template skew)",
        image_count,
        run_len,
      )));
    }

    // Capacity = text.len() + placeholder_total - run_len (replacing the
    // entire run), with checked arithmetic to surface overflow as a
    // recoverable error rather than panic-on-grow / OOM-abort.
    let cap = text_tokens
      .len()
      .checked_add(placeholder_total)
      .and_then(|n| n.checked_sub(run_len))
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "insert_image_tokens: cap (text_len + placeholder_total - run_len)",
          "usize",
          [
            ("text_len", text_tokens.len() as u64),
            ("placeholder_total", placeholder_total as u64),
            ("run_len", run_len as u64),
          ],
        ))
      })?;
    // Recoverable reservation: a huge non-overflowing
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
      return Err(Error::MissingField(MissingFieldPayload::new(
        "insert_image_tokens (MarkerPolicy::Required, image_count > 0; chat-template / tokenizer \
           drift detected — pass MarkerPolicy::PrependIfAbsent if the model uses the \
           PROMPT_WITH_IMAGE_TOKEN-family formatter)",
        "image_marker_id token in text_tokens",
      )));
    }
    // PrependIfAbsent → PROMPT_WITH_IMAGE_TOKEN path.
    let cap = text_tokens
      .len()
      .checked_add(placeholder_total)
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "insert_image_tokens: cap (text_len + placeholder_total)",
          "usize",
          [
            ("text_len", text_tokens.len() as u64),
            ("placeholder_total", placeholder_total as u64),
          ],
        ))
      })?;
    // Recoverable reservation — see the marker-present
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
/// Violations return a typed error with a descriptive message;
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
/// - `Error::InvariantViolation` if any span is empty or overlaps another;
///   `Error::OutOfRange` if any span end exceeds `seq_len` or if `seq_len`
///   exceeds `i32::MAX` (mlx dimensions are signed 32-bit);
///   `Error::ArithmeticOverflow` if `seq_len * seq_len` overflows `usize`.
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
  let total_keys = past_len.checked_add(seq_len).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "build_multimodal_mask_with_past: total_keys (past_len + seq_len)",
      "usize",
      [("past_len", past_len as u64), ("seq_len", seq_len as u64)],
    ))
  })?;

  // Empty chunk: faithful zero-query [1, 1, 0, past_len] array. Non-empty
  // spans on a zero-length chunk is an inconsistent state — fail closed.
  if seq_len == 0 {
    if !image_spans.is_empty() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "build_multimodal_mask_with_past: image_spans (with seq_len=0)",
        "must be empty — an empty chunk cannot contain any image span",
      )));
    }
    return Array::from_slice::<bool>(&[], &(1_usize, 1_usize, 0_usize, total_keys));
  }

  // mlx dimensions are signed 32-bit; reject oversized dims BEFORE any
  // host-side allocation.
  if total_keys > i32::MAX as usize {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "build_multimodal_mask_with_past: total_keys (past_len + seq_len)",
      "must be <= i32::MAX (mlx dimension limit)",
      format!("{total_keys}"),
    )));
  }

  // Validate chunk-local spans (start<end, end<=seq_len, ordered/non-overlapping).
  let mut sorted: Vec<(usize, usize)> = try_to_vec(image_spans)?;
  sorted.sort_unstable_by_key(|&(s, _)| s);
  let mut prev_end = 0usize;
  for &(s, e) in &sorted {
    if s >= e {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "build_multimodal_mask_with_past: image span (start, end)",
        "start must be strictly less than end (empty spans not allowed)",
      )));
    }
    if e > seq_len {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "build_multimodal_mask_with_past: chunk-local image span end vs seq_len",
        "must be <= seq_len",
        format!("end={e}, seq_len={seq_len}"),
      )));
    }
    if s < prev_end {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "build_multimodal_mask_with_past: image span order (s vs prev_end)",
        "spans must be monotone non-overlapping",
      )));
    }
    prev_end = e;
  }

  // Total buffer size with overflow guard: seq_len rows × total_keys cols.
  let total = seq_len.checked_mul(total_keys).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "build_multimodal_mask_with_past: total (seq_len * total_keys)",
      "usize",
      [
        ("seq_len", seq_len as u64),
        ("total_keys", total_keys as u64),
      ],
    ))
  })?;

  // block_id[i] = 1-indexed chunk-local image span index for chunk position
  // i; 0 = not in any image. Length seq_len (chunk-local). Allocated
  // fallibly for the same reason as `buf` below: these are the TWO
  // sequence-scaled buffers (`block_id` is O(seq_len); `buf` is
  // O(seq_len · total_keys) — the dominant allocation, up to MBs), so a
  // large valid chunk would otherwise abort in `vec![0u32; seq_len]`
  // before the recoverable `buf.try_reserve_exact` could report OOM.
  // The small auxiliaries here (`sorted`, a clone of
  // `image_spans` — O(num_images), a handful of `(usize,usize)` pairs)
  // follow the crate's standard infallible-`Vec` idiom: they cannot
  // realistically OOM (model image counts are small constants), and a
  // blanket try_reserve on every Vec would diverge from the rest of
  // mlxrs + the python/swift references without a real threat-model gain
  // (see the entry in docs/rust-golden-standard-followups.md for the
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
  // Recoverable reservation: on late chunks
  // `total = seq_len * (past_len + seq_len)` grows with the cached
  // context, so a long prompt's mask can be large. `try_reserve_exact`
  // surfaces an allocator failure as a recoverable `Error::OutOfMemory`
  // instead of the `Vec::with_capacity` abort. (The mask is dense by
  // contract here; a symbolic causal-base + sparse image-overlay
  // representation is the documented future optimization — it
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
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "assemble_multimodal_prompt: num_tokens_per_image (with image_count > 0)",
      "must be > 0 — otherwise images would silently drop, config/model state is degenerate",
    )));
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
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "assemble_multimodal_prompt: placeholder_total (image_count * num_tokens_per_image)",
        "usize",
        [
          ("image_count", image_count as u64),
          ("num_tokens_per_image", num_tokens_per_image as u64),
        ],
      ))
    })?;
  let final_len = if placeholder_total == 0 {
    text_tokens.len()
  } else {
    text_tokens
      .len()
      .checked_add(placeholder_total)
      .and_then(|n| n.checked_sub(marker_run_len))
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "assemble_multimodal_prompt: final_len (text_len + placeholder_total - marker_run_len)",
          "usize",
          [
            ("text_len", text_tokens.len() as u64),
            ("placeholder_total", placeholder_total as u64),
            ("marker_run_len", marker_run_len as u64),
          ],
        ))
      })?
  };
  if final_len > i32::MAX as usize {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "assemble_multimodal_prompt: final assembled length",
      "must be <= i32::MAX (mlx dimension limit; reject before allocating splice buffer)",
      format!("{final_len}"),
    )));
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
      let i_times_n = i.checked_mul(num_tokens_per_image).ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "assemble_multimodal_prompt: i * num_tokens_per_image",
          "usize",
          [
            ("i", i as u64),
            ("num_tokens_per_image", num_tokens_per_image as u64),
          ],
        ))
      })?;
      let start = base.checked_add(i_times_n).ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "assemble_multimodal_prompt: start (base + i * num_tokens_per_image)",
          "usize",
          [("base", base as u64), ("i_times_n", i_times_n as u64)],
        ))
      })?;
      let end = start.checked_add(num_tokens_per_image).ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "assemble_multimodal_prompt: end (start + num_tokens_per_image)",
          "usize",
          [
            ("start", start as u64),
            ("num_tokens_per_image", num_tokens_per_image as u64),
          ],
        ))
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

// ==========================================================================
// Chat-format builder (`MessageFormat` + `MessageFormatter`)
// ==========================================================================
//
// Faithful 1:1 port of the model-agnostic chat-format selection layer in
// `mlx-vlm/mlx_vlm/prompt_utils.py`. Per the
// `project_no_per_model_arch_porting` convention this is declarative
// configuration data (the per-model-family `MessageFormat` selection) +
// a small format dispatcher — NOT model-architecture impls.
//
// The Python ref is intentionally a thin builder: it produces a
// `{"role": ..., "content": ...}` dict (or sometimes a plain `str`) for a
// single chat turn; the per-model image/audio token-injection convention
// is encoded in the `MessageFormat` selected by [`MODEL_CONFIG`].

/// 15-variant `MessageFormat` enum — one-to-one with the Python
/// `MessageFormat(Enum)` at `mlx-vlm/mlx_vlm/prompt_utils.py:6–23`.
///
/// Each variant selects how a chat message's `content` is built for a
/// model-family. The selection per model_type is encoded in
/// [`MODEL_CONFIG`]. The 15 variants split into four families:
///
/// 1. **List-with-image** (image is a separate content item, image-tag
///    position is configurable): [`MessageFormat::ListWithImage`],
///    [`MessageFormat::ListWithImageFirst`],
///    [`MessageFormat::ListWithImageUrlFirst`],
///    [`MessageFormat::ListWithImageType`],
///    [`MessageFormat::ListWithImageTypeText`],
///    [`MessageFormat::ListWithImageTypeTextImageLast`].
/// 2. **Image-token in text** (image is a sentinel token embedded in the
///    text content): [`MessageFormat::ImageToken`],
///    [`MessageFormat::ImageTokenPipe`],
///    [`MessageFormat::StartImageToken`],
///    [`MessageFormat::ImageTokenNewline`],
///    [`MessageFormat::NumberedImageTokens`].
/// 3. **Prompt-only** (just the prompt string, possibly with image-token
///    prefix): [`MessageFormat::PromptOnly`],
///    [`MessageFormat::PromptWithImageToken`],
///    [`MessageFormat::PromptWithStartImageToken`].
/// 4. **Video-with-text** (one entry per video plus a trailing text):
///    [`MessageFormat::VideoWithText`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum MessageFormat {
  /// `list_with_image` — `content = [text, image*]` (text first, then
  /// image entries). Used by idefics2, aya_vision, qwen2_vl, llava, …
  ListWithImage,
  /// `list_with_image_first` — `content = [image*, text]` (images first).
  /// Used by idefics3, qwen2_5_vl, qwen3_vl, mistral3, smolvlm, …
  ListWithImageFirst,
  /// `list_with_image_url_first` — same as `list_with_image_first` but
  /// the image entries use `{"type": "image_url"}` (ERNIE-family).
  ListWithImageUrlFirst,
  /// `list_with_image_type` — `content = [image*, content_text]` with
  /// `content_message` (`{type:"text", text, content}`) for text and
  /// `image_message` for image; also appends audio entries after text in
  /// the user role. Default for internvl_chat, nemotron-h.
  ListWithImageType,
  /// `list_with_image_type_text` — variant of [`Self::ListWithImageType`]
  /// using `text_message` instead of `content_message`. Used by gemma3n,
  /// gemma4, pixtral.
  ListWithImageTypeText,
  /// `list_with_image_type_text_image_last` — variant of
  /// [`Self::ListWithImageTypeText`] but with images AFTER the text.
  /// Declared in the enum at line 14, but no model in the Python ref's
  /// `MODEL_CONFIG` selects it; ported for parity.
  ListWithImageTypeTextImageLast,
  /// `image_token` — content is a string `f"{<image>*N}{prompt}"`
  /// (image-token prefix). Used by minicpmo, multi_modality.
  ImageToken,
  /// `image_token_pipe` — content is a string with `<|image|>` token
  /// prefix. Used by jvlm / jina_vlm.
  ImageTokenPipe,
  /// `start_image_token` — content is a string `f"{prompt}{<start_of_image>*N}"`
  /// (image token APPENDED, not prepended). Used by gemma3.
  StartImageToken,
  /// `image_token_newline` — `<image>\n` token prefix (per-image).
  /// Used by llava-qwen2, bunny-llama, deepseek_vl_v2.
  ImageTokenNewline,
  /// `numbered_image_tokens` — `<|image_1|><|image_2|>...` prefix
  /// (followed by `<|audio_N|>` numbered audio tokens). Used by
  /// phi3_v, phi4mm.
  NumberedImageTokens,
  /// `prompt_only` — content is the bare prompt string, no tokens
  /// injected. Used by florence2, molmo, moondream3, falcon_ocr.
  PromptOnly,
  /// `prompt_with_image_token` — content is `f"<image>*N + prompt"`
  /// (a flat string, not a dict). Used by paligemma.
  PromptWithImageToken,
  /// `prompt_with_start_image_token` — content is
  /// `f"prompt + <start_of_image>*N"`. Declared in the enum at line 22,
  /// but no model in the Python ref's `MODEL_CONFIG` selects it;
  /// ported for parity.
  PromptWithStartImageToken,
  /// `video_with_text` — `content = [{type:"video", video, max_pixels,
  /// fps} * N, text]`. Used by qwen2_vl / qwen2_5_vl / qwen3_vl /
  /// qwen3_omni_moe / gemma4 when `video=` is passed.
  VideoWithText,
}

impl MessageFormat {
  /// Lowercase snake_case string tag matching the Python
  /// `MessageFormat(Enum)` value strings (e.g. `Qwen2Vl` → `"qwen2_vl"`).
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::ListWithImage => "list_with_image",
      Self::ListWithImageFirst => "list_with_image_first",
      Self::ListWithImageUrlFirst => "list_with_image_url_first",
      Self::ListWithImageType => "list_with_image_type",
      Self::ListWithImageTypeText => "list_with_image_type_text",
      Self::ListWithImageTypeTextImageLast => "list_with_image_type_text_image_last",
      Self::ImageToken => "image_token",
      Self::ImageTokenPipe => "image_token_pipe",
      Self::StartImageToken => "start_image_token",
      Self::ImageTokenNewline => "image_token_newline",
      Self::NumberedImageTokens => "numbered_image_tokens",
      Self::PromptOnly => "prompt_only",
      Self::PromptWithImageToken => "prompt_with_image_token",
      Self::PromptWithStartImageToken => "prompt_with_start_image_token",
      Self::VideoWithText => "video_with_text",
    }
  }
}

/// All 15 [`MessageFormat`] variants in declaration order — used by the
/// `message_format_15_variants_table` test to assert the enum matches the
/// Python `MessageFormat(Enum)` declaration faithfully.
///
/// (The dispatcher prompt referred to "18 variants" as an audit
/// estimate; the Python ref has EXACTLY 15 enum variants — verified by
/// `grep '= "' prompt_utils.py | head -20`. The Rust port matches that
/// count one-to-one. The other "shapes" alluded to by the audit are the
/// 6 [`MessageBuilder`] static constructors below.)
pub const MESSAGE_FORMAT_VARIANTS: &[MessageFormat] = &[
  MessageFormat::ListWithImage,
  MessageFormat::ListWithImageFirst,
  MessageFormat::ListWithImageUrlFirst,
  MessageFormat::ListWithImageType,
  MessageFormat::ListWithImageTypeText,
  MessageFormat::ListWithImageTypeTextImageLast,
  MessageFormat::ImageToken,
  MessageFormat::ImageTokenPipe,
  MessageFormat::StartImageToken,
  MessageFormat::ImageTokenNewline,
  MessageFormat::NumberedImageTokens,
  MessageFormat::PromptOnly,
  MessageFormat::PromptWithImageToken,
  MessageFormat::PromptWithStartImageToken,
  MessageFormat::VideoWithText,
];

/// Per-model-family [`MessageFormat`] selection — faithful 1:1 port of
/// the Python `MODEL_CONFIG` dict at
/// `mlx-vlm/mlx_vlm/prompt_utils.py:27–89`.
///
/// Stored as a sorted (lexicographic by `model_type`) slice so that
/// [`MessageFormatter::for_model`] can binary-search without allocating a
/// `HashMap` (the table is ~60 small entries — a binary search is faster
/// than a hash lookup at this size and avoids the `HashMap` pulled
/// dependency surface).
///
/// Keys are the lowercased `model_type` strings the python reference
/// also lowercases at line 196 (`self.model_name = model_name.lower()`).
/// Match the Python entries verbatim — including the deprecated aliases
/// (`jvlm`, `lfm2_vl`, `llava-qwen2`, `llava_qwen2`, `bunny-llama`,
/// `deepseekocr_2`) so a caller migrating from the python ref sees the
/// same `model_type`s accepted.
pub const MODEL_CONFIG: &[(&str, MessageFormat)] = &[
  // ─── kept in lexicographic order so binary_search_by works ────────
  ("aya_vision", MessageFormat::ListWithImage),
  ("bunny-llama", MessageFormat::ImageTokenNewline),
  ("cohere2_vision", MessageFormat::ListWithImage),
  ("deepseek_vl_v2", MessageFormat::ImageTokenNewline),
  ("deepseekocr", MessageFormat::ImageTokenNewline),
  ("deepseekocr_2", MessageFormat::ImageTokenNewline),
  ("dots_ocr", MessageFormat::ListWithImageFirst),
  ("ernie4_5_moe_vl", MessageFormat::ListWithImageUrlFirst),
  ("falcon_ocr", MessageFormat::PromptOnly),
  ("florence2", MessageFormat::PromptOnly),
  ("gemma3", MessageFormat::StartImageToken),
  ("gemma3n", MessageFormat::ListWithImageTypeText),
  ("gemma4", MessageFormat::ListWithImageTypeText),
  ("glm4v", MessageFormat::ListWithImageFirst),
  ("glm4v_moe", MessageFormat::ListWithImageFirst),
  ("glm_ocr", MessageFormat::ListWithImageFirst),
  ("granite4_vision", MessageFormat::ListWithImage),
  ("granite_vision", MessageFormat::ListWithImage),
  ("hunyuan_vl", MessageFormat::ListWithImageFirst),
  ("idefics2", MessageFormat::ListWithImage),
  ("idefics3", MessageFormat::ListWithImageFirst),
  ("internvl_chat", MessageFormat::ListWithImageType),
  ("jina_vlm", MessageFormat::ImageTokenPipe),
  ("jvlm", MessageFormat::ImageTokenPipe),
  ("kimi_k25", MessageFormat::ListWithImage),
  ("kimi_vl", MessageFormat::ListWithImage),
  ("lfm2-vl", MessageFormat::ListWithImageFirst),
  ("lfm2_vl", MessageFormat::ListWithImageFirst),
  ("llama4", MessageFormat::ListWithImage),
  ("llava", MessageFormat::ListWithImage),
  ("llava-qwen2", MessageFormat::ImageTokenNewline),
  ("llava_next", MessageFormat::ListWithImage),
  ("llava_qwen2", MessageFormat::ImageTokenNewline),
  ("minicpmo", MessageFormat::ImageToken),
  ("mistral3", MessageFormat::ListWithImageFirst),
  ("mllama", MessageFormat::ListWithImage),
  ("molmo", MessageFormat::PromptOnly),
  ("molmo2", MessageFormat::ListWithImageFirst),
  ("molmo_point", MessageFormat::ListWithImageFirst),
  ("moondream3", MessageFormat::PromptOnly),
  ("multi_modality", MessageFormat::ImageToken),
  ("nemotron_h_nano_omni", MessageFormat::ListWithImageType),
  (
    "nemotronh_nano_omni_reasoning_v3",
    MessageFormat::ListWithImageType,
  ),
  ("paddleocr_vl", MessageFormat::ListWithImageFirst),
  ("paligemma", MessageFormat::PromptWithImageToken),
  ("phi3_v", MessageFormat::NumberedImageTokens),
  ("phi4-siglip", MessageFormat::ImageTokenNewline),
  ("phi4mm", MessageFormat::NumberedImageTokens),
  ("pixtral", MessageFormat::ListWithImageTypeText),
  ("qwen2_5_vl", MessageFormat::ListWithImageFirst),
  ("qwen2_vl", MessageFormat::ListWithImage),
  ("qwen3_5", MessageFormat::ListWithImageFirst),
  ("qwen3_5_moe", MessageFormat::ListWithImageFirst),
  ("qwen3_omni_moe", MessageFormat::ListWithImageFirst),
  ("qwen3_vl", MessageFormat::ListWithImageFirst),
  ("qwen3_vl_moe", MessageFormat::ListWithImageFirst),
  ("smolvlm", MessageFormat::ListWithImageFirst),
  ("youtu_vl", MessageFormat::ListWithImageFirst),
];

/// Models that do NOT support multi-image chat — faithful 1:1 port of
/// the Python `SINGLE_IMAGE_ONLY_MODELS` set at
/// `mlx-vlm/mlx_vlm/prompt_utils.py:92–100`. Sorted for binary search.
pub const SINGLE_IMAGE_ONLY_MODELS: &[&str] = &[
  "bunny-llama",
  "falcon_ocr",
  "llava-qwen2",
  "llava_next",
  "mllama",
  "multi_modality",
  "paligemma",
];

/// Models that emit the video format on `video=` — faithful 1:1 port
/// of the literal list at `mlx-vlm/mlx_vlm/prompt_utils.py:221–230`.
/// Sorted for binary search.
const VIDEO_FORMAT_MODELS: &[&str] = &[
  "gemma4",
  "qwen2_5_vl",
  "qwen2_vl",
  "qwen3_5",
  "qwen3_5_moe",
  "qwen3_omni_moe",
  "qwen3_vl",
  "qwen3_vl_moe",
];

/// A single content item in a [`Message::content`] list — the typed
/// equivalent of the Python dicts produced by [`MessageBuilder`].
///
/// `mlx-vlm`'s `MessageBuilder` returns plain `dict`s (e.g.
/// `{"type": "image"}` at line 167, or
/// `{"type": "text", "text": ..., "content": ...}` at line 157). The Rust
/// port models them as a strongly-typed enum so a downstream chat-template
/// renderer can match on the variant rather than string-comparing
/// `item.get("type")` (mirrors the `MessageBuilder` static constructors
/// at `prompt_utils.py:151–189` one-to-one).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContentItem {
  /// `{type:"text", text, content}` — produced by
  /// [`MessageBuilder::text_message`]. The Python ref carries the same
  /// string in two fields (`text` AND `content`) at line 157 — a faithful
  /// port keeps both so downstream consumers that read either field see
  /// the same value.
  Text {
    /// The text payload (mirrors `dict["text"]` AND `dict["content"]`
    /// in the python ref).
    text: String,
  },
  /// `{type:"text", text, content}` — produced by
  /// [`MessageBuilder::content_message`] (line 161); semantically
  /// identical to [`Self::Text`] but the python ref distinguishes them
  /// because `_format_list_with_image_type` line 323 selects between
  /// `content_message` and `text_message` based on `message_type=
  /// "content" | "text"`. We preserve the distinction so the
  /// dispatcher's selection round-trips faithfully.
  ContentText {
    /// The text payload (mirrors `dict["text"]` AND `dict["content"]`
    /// in the python ref).
    text: String,
  },
  /// `{type:"image"}` — produced by [`MessageBuilder::image_message`]
  /// (line 167). The image *data* is carried separately (see
  /// `prepare_inputs`); this entry only marks the per-image position
  /// in the message content.
  Image,
  /// `{type:"image_url"}` — produced by
  /// [`MessageBuilder::image_url_message`] (line 172). Same as
  /// [`Self::Image`] but with a different `type` tag (ERNIE-family).
  ImageUrl,
  /// `{type:"audio"}` — produced by
  /// [`MessageBuilder::audio_message`] (line 177).
  Audio,
  /// `{type:"video", video, max_pixels, fps}` — produced by
  /// [`MessageBuilder::video_message`] (line 184). Carries the source
  /// path plus the sampling parameters (pixels-cap + frames-per-second)
  /// the chat template's video preprocessor reads at render time.
  Video {
    /// Source path (or URL) for the video. Mirrors `dict["video"]` in
    /// the python ref.
    video: String,
    /// Maximum pixel count per frame. Mirrors `dict["max_pixels"]` in
    /// the python ref (default `224 * 224`).
    max_pixels: u32,
    /// Frame sampling rate. Mirrors `dict["fps"]` in the python ref
    /// (default `1`).
    fps: u32,
  },
}

/// Static content-item constructors — faithful 1:1 port of the Python
/// `MessageBuilder` class at `mlx-vlm/mlx_vlm/prompt_utils.py:151–189`.
///
/// Each method mirrors one of the 6 `@staticmethod`s on the Python class
/// and returns a typed [`ContentItem`] instead of a `dict`.
#[derive(Debug, Clone, Copy)]
pub struct MessageBuilder;

impl MessageBuilder {
  /// `text_message(text)` — `mlx-vlm/mlx_vlm/prompt_utils.py:154–157`.
  ///
  /// The python ref returns `{"type":"text", "text":text, "content":text}`
  /// (the same string under two keys). The typed [`ContentItem::Text`]
  /// variant stores it once and the chat-template renderer can write
  /// either key.
  pub fn text_message(text: impl Into<String>) -> ContentItem {
    ContentItem::Text { text: text.into() }
  }

  /// `content_message(content)` — `mlx-vlm/mlx_vlm/prompt_utils.py:159–162`.
  ///
  /// Returns the [`ContentItem::ContentText`] discriminant; semantically
  /// identical to [`Self::text_message`] but the python `_format_list_with_image_type`
  /// at line 323 selects between them based on `message_type`.
  pub fn content_message(content: impl Into<String>) -> ContentItem {
    ContentItem::ContentText {
      text: content.into(),
    }
  }

  /// `image_message()` — `mlx-vlm/mlx_vlm/prompt_utils.py:164–167`.
  pub fn image_message() -> ContentItem {
    ContentItem::Image
  }

  /// `image_url_message()` — `mlx-vlm/mlx_vlm/prompt_utils.py:169–172`.
  pub fn image_url_message() -> ContentItem {
    ContentItem::ImageUrl
  }

  /// `audio_message()` — `mlx-vlm/mlx_vlm/prompt_utils.py:174–177`.
  pub fn audio_message() -> ContentItem {
    ContentItem::Audio
  }

  /// `video_message(video_path, max_pixels=224*224, fps=1)` —
  /// `mlx-vlm/mlx_vlm/prompt_utils.py:179–189`.
  pub fn video_message(video_path: impl Into<String>, max_pixels: u32, fps: u32) -> ContentItem {
    ContentItem::Video {
      video: video_path.into(),
      max_pixels,
      fps,
    }
  }
}

/// One chat turn — the typed equivalent of the
/// `{"role": ..., "content": ...}` dict the Python formatters return.
///
/// `content` is either a list of [`ContentItem`]s (multimodal turn, mirrors
/// the python `list` branch) or a flat string (token-prefix or
/// prompt-only turn, mirrors the python `str` branch). The python ref
/// returns one or the other depending on the [`MessageFormat`] selected;
/// the [`MessageContent`] enum surfaces that distinction in the type
/// system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
  /// `messages[i]["role"]` — typically `"user"` / `"assistant"` /
  /// `"system"` / `"tool"`. Stored as a string (not [`crate::lm::session::Role`])
  /// because the python ref's `format_message(role=...)` accepts any
  /// string — the chat-template renderer ultimately interprets it.
  pub role: String,
  /// `messages[i]["content"]` — either a list (multimodal turn) or a
  /// flat string (token-prefix or prompt-only turn).
  pub content: MessageContent,
}

/// `messages[i]["content"]` — either a list of items or a flat string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageContent {
  /// `content` is a `[ContentItem*]` list — the multimodal turn.
  Items(Vec<ContentItem>),
  /// `content` is a flat string — the prompt-only / token-prefix turn.
  Text(String),
}

/// Output of [`MessageFormatter::format_message`] — either a [`Message`]
/// (the standard `{role, content}` dict branch) or a bare `String` (the
/// `PROMPT_ONLY`-family branch where the python ref returns the prompt
/// itself, not a dict).
///
/// The python ref's `format_message` returns
/// `Union[str, Dict[str, Any]]` (line 210); this enum models the same
/// distinction in the type system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormattedMessage {
  /// The python `dict` branch — the formatter produced a `{role, content}`
  /// message.
  Message(Message),
  /// The python `str` branch — the formatter produced a bare prompt
  /// string (`PROMPT_ONLY`, `PROMPT_WITH_IMAGE_TOKEN`, or
  /// `PROMPT_WITH_START_IMAGE_TOKEN`).
  String(String),
}

/// Per-call options for [`MessageFormatter::format_message`] and
/// [`get_message_json`] — faithful 1:1 port of the keyword arguments at
/// `mlx-vlm/mlx_vlm/prompt_utils.py:201–209` (formatter) and `:444–480`
/// (`get_message_json` free function).
///
/// **Two different default sets exist in the python reference**:
/// `MessageFormatter::format_message` defaults `num_images=1,
/// num_audios=1` (lines 207–208), while the `get_message_json` free
/// function defaults `num_images=0, num_audios=0` (lines 450–451). Use
/// [`Self::formatter_default`] for the formatter-internal defaults and
/// [`Self::get_message_default`] for the public-API defaults; the
/// blanket `Default::default()` matches `formatter_default()` (the
/// in-class defaults at python line 207–208) — callers of
/// [`get_message_json`] should pass `None` (which substitutes
/// `get_message_default()`) or build a `FormatOpts` explicitly.
#[derive(Debug, Clone)]
pub struct FormatOpts {
  /// `role` — defaults to `"user"`.
  pub role: String,
  /// `skip_image_token` — defaults to `false`. If true, no image entries
  /// are added (used by `apply_chat_template` for non-target turns).
  pub skip_image_token: bool,
  /// `skip_audio_token` — defaults to `false`.
  pub skip_audio_token: bool,
  /// `num_images` — formatter-internal default `1` (python
  /// `prompt_utils.py:207`); `get_message_json` public-API default `0`
  /// (python `prompt_utils.py:450`). Use [`Self::formatter_default`] or
  /// [`Self::get_message_default`] explicitly.
  pub num_images: usize,
  /// `num_audios` — formatter-internal default `1` (python
  /// `prompt_utils.py:208`); `get_message_json` public-API default `0`
  /// (python `prompt_utils.py:451`). Use [`Self::formatter_default`] or
  /// [`Self::get_message_default`] explicitly.
  pub num_audios: usize,
  /// `video` — paths for the [`MessageFormat::VideoWithText`] /
  /// `_format_video_message` branch. Empty when there is no video.
  /// Stored as a `Vec<String>` to mirror python's "scalar or list"
  /// accepted at line 424.
  pub video: Vec<String>,
  /// `max_pixels` — per-frame pixel cap for the video branch (line 428,
  /// default `224 * 224`).
  pub max_pixels: u32,
  /// `fps` — sampling fps for the video branch (line 429, default `1`).
  /// One value per entry in [`Self::video`], or a single value applied
  /// to every video; an empty `Vec` means "use the python default
  /// (`fps=1`) for every entry".
  pub fps: Vec<u32>,
}

impl FormatOpts {
  /// Formatter-internal defaults (`num_images=1, num_audios=1`) —
  /// matches `MessageFormatter::format_message` at
  /// `mlx-vlm/mlx_vlm/prompt_utils.py:201–209` (specifically lines
  /// 207–208). Use when calling [`MessageFormatter::format_message`]
  /// directly and you want the python in-class kwarg defaults.
  pub fn formatter_default() -> Self {
    Self {
      role: "user".to_string(),
      skip_image_token: false,
      skip_audio_token: false,
      num_images: 1,
      num_audios: 1,
      video: Vec::new(),
      max_pixels: 224 * 224,
      fps: Vec::new(),
    }
  }

  /// Public-API defaults (`num_images=0, num_audios=0`) — matches the
  /// `get_message_json` free function at
  /// `mlx-vlm/mlx_vlm/prompt_utils.py:444–480` (specifically lines
  /// 450–451). Use when calling [`get_message_json`] and you want the
  /// python public-API kwarg defaults; pass `None` to
  /// [`get_message_json`] for this to be applied automatically.
  ///
  /// Why the split: the python free function's signature has
  /// `num_images=0, num_audios=0` because the public-API caller is
  /// often text-only and shouldn't get a spurious image/audio entry
  /// injected by default. The formatter's own kwargs default to 1
  /// because once you're inside the formatter you usually want exactly
  /// one image/audio for the common per-turn case.
  pub fn get_message_default() -> Self {
    Self {
      role: "user".to_string(),
      skip_image_token: false,
      skip_audio_token: false,
      num_images: 0,
      num_audios: 0,
      video: Vec::new(),
      max_pixels: 224 * 224,
      fps: Vec::new(),
    }
  }
}

/// `Default::default()` mirrors the **formatter-internal** defaults
/// (python `prompt_utils.py:201–209`, lines 207–208 → `num_images=1,
/// num_audios=1`). Callers of [`get_message_json`] who want the
/// public-API defaults (python lines 450–451 → `num_images=0,
/// num_audios=0`) must use [`FormatOpts::get_message_default`] or pass
/// `None` to [`get_message_json`].
impl Default for FormatOpts {
  fn default() -> Self {
    Self::formatter_default()
  }
}

/// Per-model chat-format dispatcher — faithful 1:1 port of the Python
/// `MessageFormatter` class at `mlx-vlm/mlx_vlm/prompt_utils.py:192–441`.
///
/// Constructed with a `model_type` (lowercased, like the python ref at
/// line 196), looks up the matching [`MessageFormat`] in [`MODEL_CONFIG`],
/// then dispatches [`Self::format_message`] to the right
/// `_format_*` body.
#[derive(Debug, Clone)]
pub struct MessageFormatter {
  /// The lowercased model type passed to [`Self::for_model`].
  pub model_name: String,
  /// The [`MessageFormat`] selected from [`MODEL_CONFIG`].
  pub format_type: MessageFormat,
}

impl MessageFormatter {
  /// Construct a formatter for `model_type` by looking it up in
  /// [`MODEL_CONFIG`]. Mirrors the python `__init__` at lines 195–199.
  ///
  /// Lowercases the input first to match the python `model_name.lower()`
  /// at line 196.
  ///
  /// # Errors
  ///
  /// `Error::MissingKey` if `model_type` is not in [`MODEL_CONFIG`]
  /// (matches the python `raise ValueError(f"Unsupported model: ...")`).
  pub fn for_model(model_type: &str) -> Result<Self> {
    let lower = model_type.to_lowercase();
    // MODEL_CONFIG is sorted lexicographically; binary search ~60 entries
    // beats a linear scan and avoids pulling a `HashMap` dependency.
    let idx = MODEL_CONFIG
      .binary_search_by(|(k, _)| (*k).cmp(lower.as_str()))
      .map_err(|_| {
        Error::MissingKey(crate::error::MissingKeyPayload::new(
          "MessageFormatter::for_model: model_type not in MODEL_CONFIG",
          model_type.to_owned(),
        ))
      })?;
    Ok(Self {
      model_name: lower,
      format_type: MODEL_CONFIG[idx].1,
    })
  }

  /// Dispatch the prompt to the right `_format_*` body. Mirrors the
  /// python `format_message` at lines 201–282.
  ///
  /// # Errors
  ///
  /// - `Error::OutOfRange` if `opts.num_images > 1` and the model is
  ///   in [`SINGLE_IMAGE_ONLY_MODELS`] (mirrors python lines 214–218).
  /// - `Error::EmptyInput` if the [`MessageFormat::VideoWithText`]
  ///   branch is selected but `opts.video.is_empty()` (the python branch
  ///   at line 424 unconditionally dereferences `kwargs["video"]` — port
  ///   surfaces the missing-video case as a hard error instead of an
  ///   `IndexError`).
  /// - `Error::LengthMismatch` if [`FormatOpts::fps`] length differs from
  ///   [`FormatOpts::video`] length (mirrors python lines 431–434).
  /// - `Error::Backend` if `opts.num_images`, `opts.num_audios`, or
  ///   `opts.video.len()` exceeds [`MAX_MESSAGE_FORMAT_ITEMS`] (caller-
  ///   controlled-count allocation cap — see
  ///   [`MAX_MESSAGE_FORMAT_ITEMS`]).
  /// - `Error::OutOfMemory` if a host-side `Vec` / `String` reservation
  ///   fails (the request-scaled allocations use `try_reserve_exact`).
  pub fn format_message(&self, prompt: &str, opts: &FormatOpts) -> Result<FormattedMessage> {
    // Single-image guard — python lines 214–218.
    if opts.num_images > 1
      && SINGLE_IMAGE_ONLY_MODELS
        .binary_search(&self.model_name.as_str())
        .is_ok()
    {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "MessageFormatter::format_message: opts.num_images (this model is in \
           SINGLE_IMAGE_ONLY_MODELS — please use only 1 image)",
        "must be <= 1",
        format!("{}", opts.num_images),
      )));
    }

    // Video special-case — python lines 221–231. The python check is
    // `model_name in [...] and kwargs.get("video")` (truthy iff non-
    // empty list), so a model whose normal MessageFormat would be
    // ListWithImageFirst (e.g. qwen2_5_vl) routes to _format_video_message
    // when video= is passed.
    if !opts.video.is_empty()
      && VIDEO_FORMAT_MODELS
        .binary_search(&self.model_name.as_str())
        .is_ok()
    {
      return self.format_video_message(prompt, opts);
    }

    // Main dispatch — python lines 234–271.
    match self.format_type {
      MessageFormat::ListWithImage => self.format_list_with_image(prompt, opts, false, false),
      MessageFormat::ListWithImageFirst => self.format_list_with_image(prompt, opts, true, false),
      MessageFormat::ListWithImageUrlFirst => self.format_list_with_image(prompt, opts, true, true),
      MessageFormat::ListWithImageType => {
        self.format_list_with_image_type(prompt, opts, ContentMessageKind::Content, true)
      }
      MessageFormat::ListWithImageTypeText => {
        self.format_list_with_image_type(prompt, opts, ContentMessageKind::Text, true)
      }
      MessageFormat::ListWithImageTypeTextImageLast => {
        self.format_list_with_image_type(prompt, opts, ContentMessageKind::Text, false)
      }
      MessageFormat::ImageToken => self.format_with_token(prompt, opts, "<image>", true),
      MessageFormat::ImageTokenPipe => self.format_with_token(prompt, opts, "<|image|>", true),
      MessageFormat::StartImageToken => {
        self.format_with_token(prompt, opts, "<start_of_image>", false)
      }
      MessageFormat::ImageTokenNewline => self.format_with_token(prompt, opts, "<image>\n", true),
      MessageFormat::NumberedImageTokens => self.format_numbered_tokens(prompt, opts),
      MessageFormat::PromptOnly => Ok(FormattedMessage::String(prompt.to_string())),
      MessageFormat::PromptWithImageToken => self.format_prompt_with_image_token(prompt, opts),
      MessageFormat::PromptWithStartImageToken => {
        self.format_prompt_with_start_image_token(prompt, opts)
      }
      MessageFormat::VideoWithText => self.format_video_message(prompt, opts),
    }
  }

  /// `PROMPT_WITH_IMAGE_TOKEN` — python lines 265–267: `"<image>" *
  /// num_images + prompt`.
  ///
  /// Faithful 1:1 port: the python lambda at `prompt_utils.py:265-269`
  /// emits `"<image>" * num_images + prompt` **unconditionally** — it
  /// does NOT gate on `role` or `skip_image_token`. The lambda only
  /// closes over `num_images` and `prompt`, ignoring the `role` /
  /// `skip_image_token` / `skip_audio_token` positional args entirely.
  /// Paligemma maps to this format and relies on the `<image>` prefix
  /// being emitted for ALL roles (including `assistant`) and when
  /// `skip_image_token=true`.
  ///
  /// Allocation hardening (caller-controlled `num_images`):
  /// [`MAX_MESSAGE_FORMAT_ITEMS`] cap via `check_format_count` BEFORE
  /// allocation, `checked_mul` / `checked_add` overflow guards on the
  /// reserve size, and `try_reserve_exact` for the fallible host-side
  /// reservation.
  fn format_prompt_with_image_token(
    &self,
    prompt: &str,
    opts: &FormatOpts,
  ) -> Result<FormattedMessage> {
    // Unconditional effective_n — see python `prompt_utils.py:265-267`.
    // The cap (`check_format_count`) still bounds the allocation budget
    // regardless of `role` / `skip_image_token`, so a pathological count
    // cannot drive an OOM through this path.
    check_format_count(opts.num_images, "num_images", &self.model_name)?;
    let effective_n = opts.num_images;
    let cap = effective_n
      .checked_mul(7) // "<image>".len()
      .and_then(|n| n.checked_add(prompt.len()))
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "MessageFormatter::format_prompt_with_image_token: cap (7 * num_images + prompt.len())",
          "usize",
          [
            ("num_images", effective_n as u64),
            ("prompt_len", prompt.len() as u64),
          ],
        ))
      })?;
    let mut s = String::new();
    s.try_reserve_exact(cap).map_err(|_| Error::OutOfMemory)?;
    for _ in 0..effective_n {
      s.push_str("<image>");
    }
    s.push_str(prompt);
    Ok(FormattedMessage::String(s))
  }

  /// `PROMPT_WITH_START_IMAGE_TOKEN` — python lines 268–269: `prompt +
  /// "<start_of_image>" * num_images`.
  ///
  /// Faithful 1:1 port: same unconditional behavior as
  /// [`Self::format_prompt_with_image_token`] — the python lambda at
  /// `prompt_utils.py:265-269` emits the token-suffix `unconditionally`
  /// (no `role` / `skip_image_token` gating). No model in `MODEL_CONFIG`
  /// currently maps to this format, but the helper is kept faithful for
  /// completeness and forward-compatibility.
  ///
  /// Allocation hardening: same pattern as
  /// [`Self::format_prompt_with_image_token`] (cap + overflow guards +
  /// fallible reserve).
  fn format_prompt_with_start_image_token(
    &self,
    prompt: &str,
    opts: &FormatOpts,
  ) -> Result<FormattedMessage> {
    // Unconditional effective_n — see python `prompt_utils.py:268-269`.
    check_format_count(opts.num_images, "num_images", &self.model_name)?;
    let effective_n = opts.num_images;
    let cap = effective_n
      .checked_mul(16) // "<start_of_image>".len()
      .and_then(|n| n.checked_add(prompt.len()))
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "MessageFormatter::format_prompt_with_start_image_token: cap (16 * num_images + \
             prompt.len())",
          "usize",
          [
            ("num_images", effective_n as u64),
            ("prompt_len", prompt.len() as u64),
          ],
        ))
      })?;
    let mut s = String::new();
    s.try_reserve_exact(cap).map_err(|_| Error::OutOfMemory)?;
    s.push_str(prompt);
    for _ in 0..effective_n {
      s.push_str("<start_of_image>");
    }
    Ok(FormattedMessage::String(s))
  }

  /// `_format_list_with_image` — python lines 284–308.
  ///
  /// Builds `content = [text]` plus, if `role=="user"` and not skipped,
  /// `num_images` image entries (`ContentItem::Image` or
  /// `ContentItem::ImageUrl`), placed first or last per `image_first`.
  ///
  /// Allocation hardening: the role / skip-token gate runs BEFORE the
  /// count-scaled `try_reserve_exact`, so a `skip_image_token=true` or
  /// `role="assistant"` call with a pathological `num_images` allocates
  /// for only the single text item. `num_images` is capped at
  /// [`MAX_MESSAGE_FORMAT_ITEMS`] (`Error::Backend`); within the cap the
  /// reserve is fallible (`Error::OutOfMemory`).
  fn format_list_with_image(
    &self,
    prompt: &str,
    opts: &FormatOpts,
    image_first: bool,
    use_image_url: bool,
  ) -> Result<FormattedMessage> {
    // Role / skip gate BEFORE the count-scaled allocation.
    let effective_n = if opts.role == "user" && !opts.skip_image_token {
      check_format_count(opts.num_images, "num_images", &self.model_name)?;
      opts.num_images
    } else {
      0
    };
    let cap = 1usize.checked_add(effective_n).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "MessageFormatter::format_list_with_image: cap (1 + num_images)",
        "usize",
        [("num_images", effective_n as u64)],
      ))
    })?;
    let mut content: Vec<ContentItem> = try_with_capacity(cap)?;
    let text_first = !image_first || effective_n == 0;
    if text_first {
      content.push(MessageBuilder::text_message(prompt));
    }

    if effective_n > 0 {
      let image_builder = if use_image_url {
        MessageBuilder::image_url_message
      } else {
        MessageBuilder::image_message
      };
      for _ in 0..effective_n {
        content.push(image_builder());
      }
    }
    if !text_first {
      content.push(MessageBuilder::text_message(prompt));
    }

    Ok(FormattedMessage::Message(Message {
      role: opts.role.clone(),
      content: MessageContent::Items(content),
    }))
  }

  /// `_format_list_with_image_type` — python lines 310–348.
  ///
  /// Builds `content = [msg_func(prompt)]` (where `msg_func` is
  /// `content_message` for the default `Content` and `text_message`
  /// for `Text`); then, if `role=="user"`, prepends/appends image
  /// entries and appends audio entries. If `role=="assistant"`,
  /// collapses content to a flat string (line 343–346).
  ///
  /// Allocation hardening: image / audio role + skip-token gates run
  /// BEFORE the count-scaled `try_reserve_exact`, so an assistant /
  /// skip-true call with pathological `num_images` / `num_audios`
  /// allocates only the single text item. Both counts are capped at
  /// [`MAX_MESSAGE_FORMAT_ITEMS`] each; within the cap the reserve is
  /// fallible.
  fn format_list_with_image_type(
    &self,
    prompt: &str,
    opts: &FormatOpts,
    msg_kind: ContentMessageKind,
    image_first: bool,
  ) -> Result<FormattedMessage> {
    // Assistant role → collapse-to-string fast path BEFORE any
    // multi-item allocation. The python ref at lines 343–346 returns
    // `{role: "assistant", content: str}` regardless of image / audio
    // counts; honoring that here means a pathological num_images +
    // role="assistant" call allocates only the text string.
    if opts.role == "assistant" {
      let s = match msg_kind {
        ContentMessageKind::Content | ContentMessageKind::Text => prompt.to_string(),
      };
      return Ok(FormattedMessage::Message(Message {
        role: opts.role.clone(),
        content: MessageContent::Text(s),
      }));
    }

    // Role / skip gates BEFORE allocation.
    let n_img = if opts.role == "user" && !opts.skip_image_token {
      check_format_count(opts.num_images, "num_images", &self.model_name)?;
      opts.num_images
    } else {
      0
    };
    let n_aud = if opts.role == "user" && !opts.skip_audio_token {
      check_format_count(opts.num_audios, "num_audios", &self.model_name)?;
      opts.num_audios
    } else {
      0
    };

    // Total cells: 1 text + n_img images + n_aud audios.
    let cap = 1usize
      .checked_add(n_img)
      .and_then(|n| n.checked_add(n_aud))
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "MessageFormatter::format_list_with_image_type: cap (1 + num_images + num_audios)",
          "usize",
          [("num_images", n_img as u64), ("num_audios", n_aud as u64)],
        ))
      })?;

    let msg = match msg_kind {
      ContentMessageKind::Content => MessageBuilder::content_message(prompt),
      ContentMessageKind::Text => MessageBuilder::text_message(prompt),
    };
    let mut content: Vec<ContentItem> = try_with_capacity(cap)?;

    // Build content with the image_first ordering, applied directly
    // (no temp-Vec + concat dance).
    let text_first = !image_first || n_img == 0;
    if text_first {
      content.push(msg);
      for _ in 0..n_img {
        content.push(MessageBuilder::image_message());
      }
    } else {
      for _ in 0..n_img {
        content.push(MessageBuilder::image_message());
      }
      content.push(msg);
    }
    for _ in 0..n_aud {
      content.push(MessageBuilder::audio_message());
    }

    Ok(FormattedMessage::Message(Message {
      role: opts.role.clone(),
      content: MessageContent::Items(content),
    }))
  }

  /// `_format_with_token` — python lines 350–373.
  ///
  /// Builds `content = f"{token*N}{prompt}"` (or `{prompt}{token*N}` if
  /// `image_first=false`), then prepends `<|audio_1|><|audio_2|>...` for
  /// audio.
  ///
  /// Allocation hardening: image / audio role + skip-token gates apply
  /// BEFORE the count-scaled `try_reserve_exact`; counts are capped at
  /// [`MAX_MESSAGE_FORMAT_ITEMS`] each.
  fn format_with_token(
    &self,
    prompt: &str,
    opts: &FormatOpts,
    token: &str,
    image_first: bool,
  ) -> Result<FormattedMessage> {
    // Image / audio role + skip-token gates BEFORE the count-scaled
    // allocation. n=0 means we never expand the loop, so a pathological
    // count cannot reach the reserve.
    let n_img = if opts.role == "user" && !opts.skip_image_token {
      check_format_count(opts.num_images, "num_images", &self.model_name)?;
      opts.num_images
    } else {
      0
    };
    let n_aud = if opts.role == "user" && !opts.skip_audio_token {
      check_format_count(opts.num_audios, "num_audios", &self.model_name)?;
      opts.num_audios
    } else {
      0
    };

    // The audio prefix is `<|audio_{i+1}|>` per audio — width grows
    // with the digit count of i+1, conservatively bounded by
    // `12 + ceil(log10(n_aud+1))` (12 = "<|audio_|>"). For the cap of
    // 1024 audios, the digits are at most 4, so 16-byte/audio is a
    // tight upper bound; we use 32-byte/audio for safety (still well
    // under the cap).
    const AUDIO_BYTES_PER_AUDIO: usize = 32;
    let token_bytes = token.len().checked_mul(n_img).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "MessageFormatter::format_with_token: token_bytes (token.len() * num_images)",
        "usize",
        [
          ("token_len", token.len() as u64),
          ("num_images", n_img as u64),
        ],
      ))
    })?;
    let audio_bytes = AUDIO_BYTES_PER_AUDIO.checked_mul(n_aud).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "MessageFormatter::format_with_token: audio_bytes (AUDIO_BYTES_PER_AUDIO * num_audios)",
        "usize",
        [
          ("AUDIO_BYTES_PER_AUDIO", AUDIO_BYTES_PER_AUDIO as u64),
          ("num_audios", n_aud as u64),
        ],
      ))
    })?;
    let cap = token_bytes
      .checked_add(audio_bytes)
      .and_then(|n| n.checked_add(prompt.len()))
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "MessageFormatter::format_with_token: cap (token_bytes + audio_bytes + prompt.len())",
          "usize",
          [
            ("token_bytes", token_bytes as u64),
            ("audio_bytes", audio_bytes as u64),
            ("prompt_len", prompt.len() as u64),
          ],
        ))
      })?;

    let mut content = String::new();
    content
      .try_reserve_exact(cap)
      .map_err(|_| Error::OutOfMemory)?;

    // Audio prefix first (matches python's `f"{audio_prefix}{content}"`
    // wrap), then image-prefix-or-suffix relative to the prompt.
    for i in 0..n_aud {
      content.push_str(&format!("<|audio_{}|>", i + 1));
    }
    if image_first {
      for _ in 0..n_img {
        content.push_str(token);
      }
      content.push_str(prompt);
    } else {
      content.push_str(prompt);
      for _ in 0..n_img {
        content.push_str(token);
      }
    }

    Ok(FormattedMessage::Message(Message {
      role: opts.role.clone(),
      content: MessageContent::Text(content),
    }))
  }

  /// `_format_numbered_tokens` — python lines 375–405.
  ///
  /// Builds `content = "<|image_1|><|image_2|>...<|audio_1|>...prompt"`
  /// (images first, then audio, matching the Phi-4 convention).
  ///
  /// Allocation hardening: role + skip-token gates BEFORE the
  /// count-scaled `try_reserve_exact`; both image / audio counts capped
  /// at [`MAX_MESSAGE_FORMAT_ITEMS`].
  fn format_numbered_tokens(&self, prompt: &str, opts: &FormatOpts) -> Result<FormattedMessage> {
    let n_img = if opts.role == "user" && !opts.skip_image_token {
      check_format_count(opts.num_images, "num_images", &self.model_name)?;
      opts.num_images
    } else {
      0
    };
    let n_aud = if opts.role == "user" && !opts.skip_audio_token {
      check_format_count(opts.num_audios, "num_audios", &self.model_name)?;
      opts.num_audios
    } else {
      0
    };

    // Token widths: `<|image_N|>` = 11 bytes for N <= 9; for N up to
    // MAX_MESSAGE_FORMAT_ITEMS (1024) the digits are at most 4, so the
    // worst-case width is `<|image_1024|>` = 14 bytes. We use a
    // conservative 16-byte upper bound per token (and per audio token,
    // `<|audio_N|>` = 12 bytes, same width concern).
    const BYTES_PER_TOKEN: usize = 16;
    let img_bytes = BYTES_PER_TOKEN.checked_mul(n_img).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "MessageFormatter::format_numbered_tokens: img_bytes (BYTES_PER_TOKEN * num_images)",
        "usize",
        [
          ("BYTES_PER_TOKEN", BYTES_PER_TOKEN as u64),
          ("num_images", n_img as u64),
        ],
      ))
    })?;
    let aud_bytes = BYTES_PER_TOKEN.checked_mul(n_aud).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "MessageFormatter::format_numbered_tokens: aud_bytes (BYTES_PER_TOKEN * num_audios)",
        "usize",
        [
          ("BYTES_PER_TOKEN", BYTES_PER_TOKEN as u64),
          ("num_audios", n_aud as u64),
        ],
      ))
    })?;
    let cap = img_bytes
      .checked_add(aud_bytes)
      .and_then(|n| n.checked_add(prompt.len()))
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "MessageFormatter::format_numbered_tokens: cap (img_bytes + aud_bytes + prompt.len())",
          "usize",
          [
            ("img_bytes", img_bytes as u64),
            ("aud_bytes", aud_bytes as u64),
            ("prompt_len", prompt.len() as u64),
          ],
        ))
      })?;

    let mut content = String::new();
    content
      .try_reserve_exact(cap)
      .map_err(|_| Error::OutOfMemory)?;
    for i in 0..n_img {
      content.push_str(&format!("<|image_{}|>", i + 1));
    }
    for i in 0..n_aud {
      content.push_str(&format!("<|audio_{}|>", i + 1));
    }
    content.push_str(prompt);

    Ok(FormattedMessage::Message(Message {
      role: opts.role.clone(),
      content: MessageContent::Text(content),
    }))
  }

  /// `_format_video_message` — python lines 407–441.
  ///
  /// Emits one [`ContentItem::Video`] entry per video path, followed by
  /// a text item.
  ///
  /// Allocation hardening: `opts.video.len()` is capped at
  /// [`MAX_MESSAGE_FORMAT_ITEMS`] BEFORE the `try_reserve_exact`; the
  /// fps_list and content `Vec`s both use fallible reservation.
  fn format_video_message(&self, prompt: &str, opts: &FormatOpts) -> Result<FormattedMessage> {
    if opts.video.is_empty() {
      return Err(Error::EmptyInput(EmptyInputPayload::new(
        "MessageFormatter::format_video_message: opts.video (the python branch unconditionally \
           dereferences kwargs['video'])",
      )));
    }

    // Cap the video count BEFORE the count-scaled reserves.
    check_format_count(opts.video.len(), "video.len()", &self.model_name)?;

    let n_vid = opts.video.len();

    // fps_list (python lines 430–434): scalar applied to all, or
    // per-video. Empty → use the python default (1) for every entry.
    // Allocation is bounded by the cap-checked n_vid above.
    let fps_list: Vec<u32> = if opts.fps.is_empty() {
      let mut v: Vec<u32> = try_with_capacity(n_vid)?;
      v.resize(n_vid, 1u32);
      v
    } else if opts.fps.len() == 1 {
      let mut v: Vec<u32> = try_with_capacity(n_vid)?;
      v.resize(n_vid, opts.fps[0]);
      v
    } else if opts.fps.len() == n_vid {
      let mut v: Vec<u32> = try_with_capacity(n_vid)?;
      v.extend_from_slice(&opts.fps);
      v
    } else {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "MessageFormatter::format_video_message: opts.fps vs opts.video length (fps must be empty, \
           a scalar, or match video.len() exactly)",
        n_vid,
        opts.fps.len(),
      )));
    };

    let cap = n_vid.checked_add(1).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "MessageFormatter::format_video_message: cap (video.len() + 1)",
        "usize",
        [("video_len", n_vid as u64)],
      ))
    })?;
    let mut content: Vec<ContentItem> = try_with_capacity(cap)?;
    for (v, f) in opts.video.iter().zip(fps_list.iter()) {
      content.push(MessageBuilder::video_message(
        v.clone(),
        opts.max_pixels,
        *f,
      ));
    }
    content.push(MessageBuilder::text_message(prompt));

    Ok(FormattedMessage::Message(Message {
      role: opts.role.clone(),
      content: MessageContent::Items(content),
    }))
  }
}

/// `message_type` selector for `_format_list_with_image_type` — mirrors
/// python line 318 (`message_type: str = "content"`) restricted to the
/// two values the dispatcher actually passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentMessageKind {
  /// `message_type="content"` — `content_message` (line 324–325).
  Content,
  /// `message_type="text"` — `text_message` (line 326–327).
  Text,
}

/// Top-level helper — faithful 1:1 port of the Python `get_message_json`
/// free function at `mlx-vlm/mlx_vlm/prompt_utils.py:444–480`.
///
/// Returns a [`FormattedMessage`] (the `Union[str, Dict[str, Any]]`
/// branch of the python signature).
///
/// ## Defaults
///
/// `opts = None` substitutes [`FormatOpts::get_message_default`], which
/// matches the python free-function defaults at
/// `prompt_utils.py:444–480` (specifically lines 450–451 →
/// `num_images=0, num_audios=0`). This is intentionally different from
/// [`FormatOpts::default`] / [`FormatOpts::formatter_default`] (which
/// matches the in-class defaults at lines 207–208 → `num_images=1,
/// num_audios=1`); the public-API caller is often text-only and
/// shouldn't get a spurious image/audio entry injected by default.
///
/// Pass `Some(&FormatOpts { num_images: N, .. })` to opt in to the
/// image branch, mirroring a python caller passing `num_images=N`.
///
/// # Errors
///
/// Propagates from [`MessageFormatter::for_model`] (unsupported model)
/// and [`MessageFormatter::format_message`] (multi-image / video
/// validation / allocation cap).
pub fn get_message_json(
  model_name: &str,
  prompt: &str,
  opts: Option<&FormatOpts>,
) -> Result<FormattedMessage> {
  let formatter = MessageFormatter::for_model(model_name)?;
  let defaults;
  let resolved = match opts {
    Some(o) => o,
    None => {
      defaults = FormatOpts::get_message_default();
      &defaults
    }
  };
  formatter.format_message(prompt, resolved)
}

#[cfg(test)]
mod tests {
  use super::*;

  // Test-local helpers — build a `FormatOpts` from the formatter defaults
  // (num_images=1, num_audios=1) with selected overrides, so each test
  // declares only the fields it cares about.

  fn opts_with(role: &str, num_images: usize, num_audios: usize) -> FormatOpts {
    FormatOpts {
      role: role.to_string(),
      num_images,
      num_audios,
      ..FormatOpts::formatter_default()
    }
  }

  // Pull the `Vec<ContentItem>` out of a `FormattedMessage::Message` whose
  // content is `Items`; panic with a clear message otherwise.
  fn items(fm: &FormattedMessage) -> Vec<ContentItem> {
    match fm {
      FormattedMessage::Message(Message {
        content: MessageContent::Items(v),
        ..
      }) => v.clone(),
      other => panic!("expected Message(Items), got {other:?}"),
    }
  }

  // Pull the flat `String` out of a `FormattedMessage::Message` whose content
  // is `Text`.
  fn text_content(fm: &FormattedMessage) -> String {
    match fm {
      FormattedMessage::Message(Message {
        content: MessageContent::Text(s),
        ..
      }) => s.clone(),
      other => panic!("expected Message(Text), got {other:?}"),
    }
  }

  // ── MarkerPolicy::as_str + Display + IsVariant (lines 149–152) ──────────

  #[test]
  fn marker_policy_as_str_and_display() {
    assert_eq!(MarkerPolicy::Required.as_str(), "required");
    assert_eq!(MarkerPolicy::PrependIfAbsent.as_str(), "prepend_if_absent");
    // derive_more::Display delegates to as_str.
    assert_eq!(format!("{}", MarkerPolicy::Required), "required");
    assert_eq!(
      format!("{}", MarkerPolicy::PrependIfAbsent),
      "prepend_if_absent"
    );
    // derive_more::IsVariant.
    assert!(MarkerPolicy::Required.is_required());
    assert!(MarkerPolicy::PrependIfAbsent.is_prepend_if_absent());
    assert!(!MarkerPolicy::Required.is_prepend_if_absent());
  }

  // ── locate_image_tokens edge cases ──────────────────────────────────────

  #[test]
  fn locate_image_tokens_empty_and_runs() {
    // Empty input → empty Vec.
    assert_eq!(locate_image_tokens(&[], 99), ImageTokenSpans::new());
    // No matches.
    assert_eq!(locate_image_tokens(&[1, 2, 3], 99), ImageTokenSpans::new());
    // Run at the very end (exercises the `i == len` loop exit inside the run).
    assert_eq!(locate_image_tokens(&[1, 99, 99], 99), vec![(1, 3)]);
    // Run at the very start.
    assert_eq!(locate_image_tokens(&[99, 1], 99), vec![(0, 1)]);
  }

  // ── insert_image_tokens: marker-present cap overflow (lines 372–378) ────
  //
  // `placeholder_total = image_count * num_tokens_per_image` computes
  // (does NOT overflow at the earlier guard), the contiguous marker run
  // length equals image_count, but `text.len() + placeholder_total`
  // overflows usize. The `.checked_add` returns None BEFORE any allocation,
  // so the error surfaces with no heap pressure.

  #[test]
  fn insert_image_tokens_marker_cap_overflow() {
    // image_count=2 markers `[7,7]`; num_tokens_per_image = usize::MAX/2 so
    // placeholder_total = usize::MAX-1 (no overflow at the product guard),
    // then 2 + (usize::MAX-1) overflows.
    let text = [7_u32, 7];
    let err =
      insert_image_tokens(&text, 2, 7, 99, usize::MAX / 2, MarkerPolicy::Required).unwrap_err();
    assert!(
      matches!(err, Error::ArithmeticOverflow(_)),
      "expected ArithmeticOverflow, got {err:?}"
    );
    if let Error::ArithmeticOverflow(p) = &err {
      assert_eq!(
        p.context(),
        "insert_image_tokens: cap (text_len + placeholder_total - run_len)"
      );
    }
  }

  // ── insert_image_tokens: prepend cap overflow (lines 407–412) ───────────
  //
  // No marker present + PrependIfAbsent: `cap = text.len() + placeholder_total`
  // overflows. Same usize::MAX/2 trick.

  #[test]
  fn insert_image_tokens_prepend_cap_overflow() {
    let text = [1_u32, 2]; // no marker token (7) present
    let err = insert_image_tokens(
      &text,
      2,
      7,
      99,
      usize::MAX / 2,
      MarkerPolicy::PrependIfAbsent,
    )
    .unwrap_err();
    assert!(
      matches!(err, Error::ArithmeticOverflow(_)),
      "expected ArithmeticOverflow, got {err:?}"
    );
    if let Error::ArithmeticOverflow(p) = &err {
      assert_eq!(
        p.context(),
        "insert_image_tokens: cap (text_len + placeholder_total)"
      );
    }
  }

  // Supporting coverage of the other insert_image_tokens error branches
  // (degenerate zero-width, product overflow, extra-marker, run-length
  // mismatch, missing-marker) so the function is exercised end-to-end.

  #[test]
  fn insert_image_tokens_degenerate_and_overflow_and_invariant() {
    // num_tokens_per_image == 0 with image_count > 0 → InvariantViolation.
    let err = insert_image_tokens(&[1, 7, 2], 1, 7, 99, 0, MarkerPolicy::Required).unwrap_err();
    assert!(matches!(err, Error::InvariantViolation(_)));

    // image_count * num_tokens_per_image overflows the product guard.
    let err =
      insert_image_tokens(&[7, 7], 2, 7, 99, usize::MAX, MarkerPolicy::Required).unwrap_err();
    assert!(matches!(err, Error::ArithmeticOverflow(_)));
    if let Error::ArithmeticOverflow(p) = &err {
      assert_eq!(
        p.context(),
        "insert_image_tokens: placeholder_total (image_count * num_tokens_per_image)"
      );
    }

    // Extra non-contiguous marker after the first run → InvariantViolation.
    let err = insert_image_tokens(&[7, 1, 7], 1, 7, 99, 3, MarkerPolicy::Required).unwrap_err();
    assert!(matches!(err, Error::InvariantViolation(_)));

    // Run length (2) != image_count (1) → LengthMismatch.
    let err = insert_image_tokens(&[1, 7, 7, 2], 1, 7, 99, 3, MarkerPolicy::Required).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch(_)));
    if let Error::LengthMismatch(p) = &err {
      assert_eq!(p.expected(), 1);
      assert_eq!(p.actual(), 2);
    }

    // Missing marker under Required → MissingField.
    let err = insert_image_tokens(&[1, 2], 1, 7, 99, 3, MarkerPolicy::Required).unwrap_err();
    assert!(matches!(err, Error::MissingField(_)));
    if let Error::MissingField(p) = &err {
      assert_eq!(p.field(), "image_marker_id token in text_tokens");
    }
  }

  // ── build_multimodal_mask_with_past: total_keys overflow (lines 521–524) ─

  #[test]
  fn mask_with_past_total_keys_overflow() {
    let err = build_multimodal_mask_with_past(1, usize::MAX, &[]).unwrap_err();
    assert!(matches!(err, Error::ArithmeticOverflow(_)));
    if let Error::ArithmeticOverflow(p) = &err {
      assert_eq!(
        p.context(),
        "build_multimodal_mask_with_past: total_keys (past_len + seq_len)"
      );
    }
  }

  // ── build_multimodal_mask_with_past: total_keys > i32::MAX (lines 543–546)
  //
  // past_len + seq_len exceeds i32::MAX but does NOT overflow usize, and
  // seq_len != 0 so the empty-chunk branch is skipped.

  #[test]
  fn mask_with_past_total_keys_exceeds_i32_max() {
    let err = build_multimodal_mask_with_past(1, i32::MAX as usize, &[]).unwrap_err();
    assert!(matches!(err, Error::OutOfRange(_)));
    if let Error::OutOfRange(p) = &err {
      assert_eq!(
        p.context(),
        "build_multimodal_mask_with_past: total_keys (past_len + seq_len)"
      );
    }
  }

  // Empty-chunk branch (seq_len == 0): valid zero-query mask, plus the
  // non-empty-spans-on-zero-chunk InvariantViolation.

  #[test]
  fn mask_with_past_empty_chunk() {
    // seq_len=0, no spans → [1,1,0,past_len] array.
    let mask = build_multimodal_mask_with_past(0, 3, &[]).unwrap();
    assert_eq!(mask.shape(), vec![1, 1, 0, 3]);
    // seq_len=0 but a span supplied → InvariantViolation.
    let err = build_multimodal_mask_with_past(0, 3, &[(0, 1)]).unwrap_err();
    assert!(matches!(err, Error::InvariantViolation(_)));
  }

  // Span validation branches: empty span, end>seq_len, overlap.

  #[test]
  fn mask_span_validation_errors() {
    // start >= end (empty span).
    let err = build_multimodal_mask(4, &[(2, 2)]).unwrap_err();
    assert!(matches!(err, Error::InvariantViolation(_)));
    // end > seq_len.
    let err = build_multimodal_mask(4, &[(2, 5)]).unwrap_err();
    assert!(matches!(err, Error::OutOfRange(_)));
    // overlapping spans (sorted, second starts before first ends).
    let err = build_multimodal_mask(8, &[(1, 4), (3, 6)]).unwrap_err();
    assert!(matches!(err, Error::InvariantViolation(_)));
  }

  // ── build_multimodal_mask: full semantic oracle with an image span ──────
  //
  // seq_len=4, one image span (1,3). Independent closed-form expectation:
  // attend iff a past-style causal rule OR same-image-span bidirectionality.
  // Positions 1,2 are in image block 1; 0,3 are text.

  #[test]
  fn mask_image_span_bidirectional_oracle() {
    let mut mask = build_multimodal_mask(4, &[(1, 3)]).unwrap();
    assert_eq!(mask.shape(), vec![1, 1, 4, 4]);
    let v = mask.to_vec::<bool>().unwrap();
    // block_id = [0, 1, 1, 0].
    let block = [0u32, 1, 1, 0];
    for q in 0..4usize {
      for k in 0..4usize {
        let causal = k <= q;
        let same_img = block[q] != 0 && block[q] == block[k];
        let expected = causal || same_img;
        assert_eq!(v[q * 4 + k], expected, "mask[{q}][{k}] expected {expected}");
      }
    }
    // Spot-check the only non-causal cell: q=1 (image), k=2 (image, k>q) →
    // bidirectional within the same span → attend. Row q=1, col k=2.
    assert!(v[6]);
  }

  // Offset-aware (chunked-prefill) oracle with past keys all-attend.

  #[test]
  fn mask_with_past_attends_all_past_keys() {
    // seq_len=2, past_len=2, one chunk-local span (0,2): both current
    // positions are image block 1 → bidirectional with each other; both
    // past keys always attend.
    let mut mask = build_multimodal_mask_with_past(2, 2, &[(0, 2)]).unwrap();
    assert_eq!(mask.shape(), vec![1, 1, 2, 4]);
    let v = mask.to_vec::<bool>().unwrap();
    let block = [1u32, 1]; // chunk-local block ids
    for q in 0..2usize {
      for k in 0..4usize {
        let expected = if k < 2 {
          true
        } else {
          let kl = k - 2;
          (kl <= q) || (block[q] != 0 && block[q] == block[kl])
        };
        assert_eq!(v[q * 4 + k], expected, "mask[{q}][{k}]");
      }
    }
  }

  // ── assemble_multimodal_prompt: degenerate zero-width (lines 723–725) ───

  #[test]
  fn assemble_degenerate_zero_width() {
    let err =
      assemble_multimodal_prompt(&[1, 7, 2], 1, 7, 99, 0, MarkerPolicy::Required).unwrap_err();
    assert!(matches!(err, Error::InvariantViolation(_)));
    if let Error::InvariantViolation(p) = &err {
      assert_eq!(
        p.context(),
        "assemble_multimodal_prompt: num_tokens_per_image (with image_count > 0)"
      );
    }
  }

  // ── assemble_multimodal_prompt: placeholder_total overflow (758–763) ────

  #[test]
  fn assemble_placeholder_total_overflow() {
    let err = assemble_multimodal_prompt(&[7, 7], 2, 7, 99, usize::MAX, MarkerPolicy::Required)
      .unwrap_err();
    assert!(matches!(err, Error::ArithmeticOverflow(_)));
    if let Error::ArithmeticOverflow(p) = &err {
      assert_eq!(
        p.context(),
        "assemble_multimodal_prompt: placeholder_total (image_count * num_tokens_per_image)"
      );
    }
  }

  // ── assemble_multimodal_prompt: final_len overflow (lines 775–781) ──────
  //
  // No marker present (marker_run_len=0), placeholder_total computes
  // (= usize::MAX-1), then text.len() + placeholder_total overflows.

  #[test]
  fn assemble_final_len_overflow() {
    let err = assemble_multimodal_prompt(
      &[1, 2],
      2,
      7,
      99,
      usize::MAX / 2,
      MarkerPolicy::PrependIfAbsent,
    )
    .unwrap_err();
    assert!(matches!(err, Error::ArithmeticOverflow(_)));
    if let Error::ArithmeticOverflow(p) = &err {
      assert_eq!(
        p.context(),
        "assemble_multimodal_prompt: final_len (text_len + placeholder_total - marker_run_len)"
      );
    }
  }

  // ── assemble_multimodal_prompt: final_len > i32::MAX (early reject) ──────

  #[test]
  fn assemble_final_len_exceeds_i32_max() {
    // 1 image, num_tokens_per_image just over i32::MAX, no marker, prepend.
    let n = i32::MAX as usize + 1;
    let err =
      assemble_multimodal_prompt(&[1], 1, 7, 99, n, MarkerPolicy::PrependIfAbsent).unwrap_err();
    assert!(matches!(err, Error::OutOfRange(_)));
    if let Error::OutOfRange(p) = &err {
      assert_eq!(
        p.context(),
        "assemble_multimodal_prompt: final assembled length"
      );
    }
  }

  // ── assemble_multimodal_prompt: multi-image happy path (lines 807–840) ──
  //
  // Two images, 3 tokens each, marker run of 2. Per-image spans must be
  // (2,5) and (5,8), NOT one collapsed (2,8). Closed-form expected tokens.

  #[test]
  fn assemble_multi_image_spans() {
    let text = [1_u32, 2, 7, 7, 3];
    let p = assemble_multimodal_prompt(&text, 2, 7, 99, 3, MarkerPolicy::Required).unwrap();
    assert_eq!(p.tokens, vec![1, 2, 99, 99, 99, 99, 99, 99, 3]);
    assert_eq!(p.image_spans, vec![(2, 5), (5, 8)]);
    assert_eq!(p.attention_mask.shape(), vec![1, 1, 9, 9]);
  }

  // Zero-image assemble: passthrough tokens, empty spans, pure-causal mask.

  #[test]
  fn assemble_zero_images() {
    let text = [1_u32, 2, 3];
    let p = assemble_multimodal_prompt(&text, 0, 7, 99, 3, MarkerPolicy::Required).unwrap();
    assert_eq!(p.tokens, vec![1, 2, 3]);
    assert!(p.image_spans.is_empty());
    assert_eq!(p.attention_mask.shape(), vec![1, 1, 3, 3]);
  }

  // ── MessageFormat::as_str — all 15 variants (lines 954–970) ─────────────

  #[test]
  fn message_format_as_str_all_variants() {
    let pairs: &[(MessageFormat, &str)] = &[
      (MessageFormat::ListWithImage, "list_with_image"),
      (MessageFormat::ListWithImageFirst, "list_with_image_first"),
      (
        MessageFormat::ListWithImageUrlFirst,
        "list_with_image_url_first",
      ),
      (MessageFormat::ListWithImageType, "list_with_image_type"),
      (
        MessageFormat::ListWithImageTypeText,
        "list_with_image_type_text",
      ),
      (
        MessageFormat::ListWithImageTypeTextImageLast,
        "list_with_image_type_text_image_last",
      ),
      (MessageFormat::ImageToken, "image_token"),
      (MessageFormat::ImageTokenPipe, "image_token_pipe"),
      (MessageFormat::StartImageToken, "start_image_token"),
      (MessageFormat::ImageTokenNewline, "image_token_newline"),
      (MessageFormat::NumberedImageTokens, "numbered_image_tokens"),
      (MessageFormat::PromptOnly, "prompt_only"),
      (
        MessageFormat::PromptWithImageToken,
        "prompt_with_image_token",
      ),
      (
        MessageFormat::PromptWithStartImageToken,
        "prompt_with_start_image_token",
      ),
      (MessageFormat::VideoWithText, "video_with_text"),
    ];
    for (fmt, s) in pairs {
      assert_eq!(fmt.as_str(), *s);
      // derive_more::Display delegates to as_str.
      assert_eq!(format!("{fmt}"), *s);
    }
    // The constant table must enumerate exactly these 15 in order.
    assert_eq!(MESSAGE_FORMAT_VARIANTS.len(), 15);
    for (i, (fmt, _)) in pairs.iter().enumerate() {
      assert_eq!(MESSAGE_FORMAT_VARIANTS[i], *fmt);
    }
  }

  // ── MessageBuilder constructors ─────────────────────────────────────────

  #[test]
  fn message_builder_constructors() {
    assert_eq!(
      MessageBuilder::text_message("hi"),
      ContentItem::Text {
        text: "hi".to_string()
      }
    );
    assert_eq!(
      MessageBuilder::content_message("hi"),
      ContentItem::ContentText {
        text: "hi".to_string()
      }
    );
    assert_eq!(MessageBuilder::image_message(), ContentItem::Image);
    assert_eq!(MessageBuilder::image_url_message(), ContentItem::ImageUrl);
    assert_eq!(MessageBuilder::audio_message(), ContentItem::Audio);
    assert_eq!(
      MessageBuilder::video_message("v.mp4", 100, 2),
      ContentItem::Video {
        video: "v.mp4".to_string(),
        max_pixels: 100,
        fps: 2,
      }
    );
  }

  // ── FormatOpts default variants ─────────────────────────────────────────

  #[test]
  fn format_opts_defaults() {
    let f = FormatOpts::formatter_default();
    assert_eq!(f.num_images, 1);
    assert_eq!(f.num_audios, 1);
    assert_eq!(f.role, "user");
    assert_eq!(f.max_pixels, 224 * 224);
    // Default::default() == formatter_default().
    let d = FormatOpts::default();
    assert_eq!(d.num_images, 1);
    assert_eq!(d.num_audios, 1);
    // get_message_default → 0/0.
    let g = FormatOpts::get_message_default();
    assert_eq!(g.num_images, 0);
    assert_eq!(g.num_audios, 0);
    assert_eq!(g.role, "user");
  }

  // ── MessageFormatter::for_model: lookup + lowercasing + missing-key ─────

  #[test]
  fn for_model_lookup_and_errors() {
    let f = MessageFormatter::for_model("Qwen2_VL").unwrap();
    // Lowercased to match the python `model_name.lower()`.
    assert_eq!(f.model_name, "qwen2_vl");
    assert_eq!(f.format_type, MessageFormat::ListWithImage);

    // Unknown model → MissingKey.
    let err = MessageFormatter::for_model("nonexistent_model").unwrap_err();
    assert!(matches!(err, Error::MissingKey(_)));
    if let Error::MissingKey(p) = &err {
      assert_eq!(p.key(), "nonexistent_model");
    }
  }

  // ── format_message dispatch: single-image guard (SINGLE_IMAGE_ONLY) ─────

  #[test]
  fn dispatch_single_image_guard() {
    // paligemma is in SINGLE_IMAGE_ONLY_MODELS; num_images=2 → OutOfRange.
    let f = MessageFormatter::for_model("paligemma").unwrap();
    let opts = opts_with("user", 2, 0);
    let err = f.format_message("hi", &opts).unwrap_err();
    assert!(matches!(err, Error::OutOfRange(_)));
  }

  // ── format_message dispatch arms ────────────────────────────────────────
  //
  // Each arm is selected by the model's MODEL_CONFIG entry. We assert the
  // exact formatted content as an independent closed-form oracle.

  // ListWithImage (text-first): qwen2_vl. (line 1480)
  #[test]
  fn dispatch_list_with_image() {
    let f = MessageFormatter::for_model("qwen2_vl").unwrap();
    let it = items(&f.format_message("hi", &opts_with("user", 1, 0)).unwrap());
    assert_eq!(
      it,
      vec![
        ContentItem::Text {
          text: "hi".to_string()
        },
        ContentItem::Image
      ]
    );
  }

  // ListWithImageFirst (image-first): qwen2_5_vl. (line 1481)
  #[test]
  fn dispatch_list_with_image_first() {
    let f = MessageFormatter::for_model("qwen2_5_vl").unwrap();
    let it = items(&f.format_message("hi", &opts_with("user", 2, 0)).unwrap());
    assert_eq!(
      it,
      vec![
        ContentItem::Image,
        ContentItem::Image,
        ContentItem::Text {
          text: "hi".to_string()
        },
      ]
    );
  }

  // ListWithImageUrlFirst (image_url, image-first): ernie4_5_moe_vl. (line 1482)
  #[test]
  fn dispatch_list_with_image_url_first() {
    let f = MessageFormatter::for_model("ernie4_5_moe_vl").unwrap();
    let it = items(&f.format_message("hi", &opts_with("user", 1, 0)).unwrap());
    assert_eq!(
      it,
      vec![
        ContentItem::ImageUrl,
        ContentItem::Text {
          text: "hi".to_string()
        },
      ]
    );
  }

  // ListWithImageType (Content kind, image-first): internvl_chat. (lines 1483–1485)
  #[test]
  fn dispatch_list_with_image_type_content() {
    let f = MessageFormatter::for_model("internvl_chat").unwrap();
    // 1 image, 1 audio, user role.
    let it = items(&f.format_message("hi", &opts_with("user", 1, 1)).unwrap());
    // image-first: [image, content_text, audio].
    assert_eq!(
      it,
      vec![
        ContentItem::Image,
        ContentItem::ContentText {
          text: "hi".to_string()
        },
        ContentItem::Audio,
      ]
    );
  }

  // ListWithImageTypeText (Text kind, image-first): pixtral. (lines 1486–1488)
  #[test]
  fn dispatch_list_with_image_type_text() {
    let f = MessageFormatter::for_model("pixtral").unwrap();
    let it = items(&f.format_message("hi", &opts_with("user", 1, 0)).unwrap());
    assert_eq!(
      it,
      vec![
        ContentItem::Image,
        ContentItem::Text {
          text: "hi".to_string()
        },
      ]
    );
  }

  // ImageToken ("<image>", image_first): minicpmo. (line 1492)
  #[test]
  fn dispatch_image_token() {
    let f = MessageFormatter::for_model("minicpmo").unwrap();
    let s = text_content(&f.format_message("hi", &opts_with("user", 2, 0)).unwrap());
    assert_eq!(s, "<image><image>hi");
  }

  // ImageTokenPipe ("<|image|>"): jina_vlm. (line 1493)
  #[test]
  fn dispatch_image_token_pipe() {
    let f = MessageFormatter::for_model("jina_vlm").unwrap();
    let s = text_content(&f.format_message("hi", &opts_with("user", 1, 0)).unwrap());
    assert_eq!(s, "<|image|>hi");
  }

  // StartImageToken ("<start_of_image>", image LAST): gemma3. (lines 1494–1496)
  #[test]
  fn dispatch_start_image_token() {
    let f = MessageFormatter::for_model("gemma3").unwrap();
    let s = text_content(&f.format_message("hi", &opts_with("user", 1, 0)).unwrap());
    assert_eq!(s, "hi<start_of_image>");
  }

  // ImageTokenNewline ("<image>\n"): deepseek_vl_v2. (line 1497)
  #[test]
  fn dispatch_image_token_newline() {
    let f = MessageFormatter::for_model("deepseek_vl_v2").unwrap();
    let s = text_content(&f.format_message("hi", &opts_with("user", 1, 0)).unwrap());
    assert_eq!(s, "<image>\nhi");
  }

  // NumberedImageTokens: phi3_v. (line 1498)
  #[test]
  fn dispatch_numbered_image_tokens() {
    let f = MessageFormatter::for_model("phi3_v").unwrap();
    let s = text_content(&f.format_message("hi", &opts_with("user", 2, 1)).unwrap());
    assert_eq!(s, "<|image_1|><|image_2|><|audio_1|>hi");
  }

  // PromptOnly: florence2 → bare String. (line 1499)
  #[test]
  fn dispatch_prompt_only() {
    let f = MessageFormatter::for_model("florence2").unwrap();
    let out = f.format_message("hi", &opts_with("user", 1, 0)).unwrap();
    assert_eq!(out, FormattedMessage::String("hi".to_string()));
  }

  // PromptWithImageToken: paligemma → "<image>" * N + prompt. (line 1500,
  // body lines 1525–1556)
  #[test]
  fn dispatch_prompt_with_image_token() {
    let f = MessageFormatter::for_model("paligemma").unwrap();
    // Single image (paligemma is single-image-only; >1 errors).
    let out = f.format_message("hi", &opts_with("user", 1, 0)).unwrap();
    assert_eq!(out, FormattedMessage::String("<image>hi".to_string()));
    // Unconditional: assistant role still emits the prefix.
    let out = f
      .format_message("hi", &opts_with("assistant", 1, 0))
      .unwrap();
    assert_eq!(out, FormattedMessage::String("<image>hi".to_string()));
    // num_images=0 → bare prompt.
    let out = f.format_message("hi", &opts_with("user", 0, 0)).unwrap();
    assert_eq!(out, FormattedMessage::String("hi".to_string()));
  }

  // ── VideoWithText dispatch + format_video_message branches (line 1504) ──
  //
  // qwen2_vl with a non-empty `video` routes to format_video_message via the
  // special-case at lines 1470–1476. We also drive the fps branches and the
  // empty-video error directly through models in VIDEO_FORMAT_MODELS.

  #[test]
  fn dispatch_video_with_text_default_fps() {
    let f = MessageFormatter::for_model("qwen2_vl").unwrap();
    let opts = FormatOpts {
      role: "user".to_string(),
      video: vec!["a.mp4".to_string(), "b.mp4".to_string()],
      fps: Vec::new(), // empty → default fps=1 for each (lines 1960–1963)
      max_pixels: 256,
      ..FormatOpts::formatter_default()
    };
    let it = items(&f.format_message("cap", &opts).unwrap());
    assert_eq!(
      it,
      vec![
        ContentItem::Video {
          video: "a.mp4".to_string(),
          max_pixels: 256,
          fps: 1,
        },
        ContentItem::Video {
          video: "b.mp4".to_string(),
          max_pixels: 256,
          fps: 1,
        },
        ContentItem::Text {
          text: "cap".to_string()
        },
      ]
    );
  }

  #[test]
  fn video_message_scalar_fps_broadcast() {
    // fps.len()==1 → scalar applied to all (lines 1964–1967).
    let f = MessageFormatter::for_model("qwen2_5_vl").unwrap();
    let opts = FormatOpts {
      role: "user".to_string(),
      video: vec!["a.mp4".to_string(), "b.mp4".to_string()],
      fps: vec![5],
      max_pixels: 224 * 224,
      ..FormatOpts::formatter_default()
    };
    let it = items(&f.format_message("cap", &opts).unwrap());
    assert_eq!(
      it,
      vec![
        ContentItem::Video {
          video: "a.mp4".to_string(),
          max_pixels: 224 * 224,
          fps: 5,
        },
        ContentItem::Video {
          video: "b.mp4".to_string(),
          max_pixels: 224 * 224,
          fps: 5,
        },
        ContentItem::Text {
          text: "cap".to_string()
        },
      ]
    );
  }

  #[test]
  fn video_message_per_video_fps() {
    // fps.len()==video.len() → per-video (lines 1968–1971).
    let f = MessageFormatter::for_model("qwen3_vl").unwrap();
    let opts = FormatOpts {
      role: "user".to_string(),
      video: vec!["a.mp4".to_string(), "b.mp4".to_string()],
      fps: vec![2, 3],
      ..FormatOpts::formatter_default()
    };
    let it = items(&f.format_message("cap", &opts).unwrap());
    match (&it[0], &it[1]) {
      (ContentItem::Video { fps: f0, .. }, ContentItem::Video { fps: f1, .. }) => {
        assert_eq!(*f0, 2);
        assert_eq!(*f1, 3);
      }
      _ => panic!("expected two Video items"),
    }
  }

  #[test]
  fn video_message_fps_length_mismatch() {
    // fps.len() neither 1 nor video.len() → LengthMismatch (lines 1972–1979).
    let f = MessageFormatter::for_model("qwen2_vl").unwrap();
    let opts = FormatOpts {
      role: "user".to_string(),
      video: vec![
        "a.mp4".to_string(),
        "b.mp4".to_string(),
        "c.mp4".to_string(),
      ],
      fps: vec![1, 2],
      ..FormatOpts::formatter_default()
    };
    let err = f.format_message("cap", &opts).unwrap_err();
    assert!(matches!(err, Error::LengthMismatch(_)));
    if let Error::LengthMismatch(p) = &err {
      assert_eq!(p.expected(), 3);
      assert_eq!(p.actual(), 2);
    }
  }

  // (The empty-video EmptyInput guard at lines 1946–1948 is covered through
  // the synthetic `VideoWithText` base-format arm in
  // `video_with_text_base_format_arm` below, the only public surface that
  // can reach `format_video_message` with an empty video list — the video
  // special-case route short-circuits on an empty list.)

  // ── format_list_with_image: assistant / skip gates (else-0 branch) ──────

  #[test]
  fn list_with_image_assistant_role_no_images() {
    // role != "user" → effective_n = 0 (the else branch at ~1626): just text.
    let f = MessageFormatter::for_model("qwen2_vl").unwrap();
    let it = items(
      &f.format_message("hi", &opts_with("assistant", 3, 0))
        .unwrap(),
    );
    assert_eq!(
      it,
      vec![ContentItem::Text {
        text: "hi".to_string()
      }]
    );
  }

  #[test]
  fn list_with_image_skip_image_token() {
    // skip_image_token=true → effective_n = 0.
    let f = MessageFormatter::for_model("qwen2_vl").unwrap();
    let opts = FormatOpts {
      role: "user".to_string(),
      num_images: 3,
      skip_image_token: true,
      ..FormatOpts::formatter_default()
    };
    let it = items(&f.format_message("hi", &opts).unwrap());
    assert_eq!(
      it,
      vec![ContentItem::Text {
        text: "hi".to_string()
      }]
    );
  }

  // ── format_list_with_image_type: assistant collapse + else-0 gates ──────
  //
  // assistant role → collapse-to-string fast path (lines 1687–1695).

  #[test]
  fn list_with_image_type_assistant_collapse() {
    let f = MessageFormatter::for_model("internvl_chat").unwrap();
    let out = f
      .format_message("hi", &opts_with("assistant", 2, 2))
      .unwrap();
    assert_eq!(out, text_content_msg("assistant", "hi"));
  }

  // System role (neither user nor assistant): image/audio gates → 0
  // (lines 1702, 1708 else branches). Content built with just the text msg.
  #[test]
  fn list_with_image_type_system_role_no_media() {
    let f = MessageFormatter::for_model("internvl_chat").unwrap();
    let it = items(&f.format_message("sys", &opts_with("system", 5, 5)).unwrap());
    // Not user → n_img=0, n_aud=0 → [content_text] only.
    assert_eq!(
      it,
      vec![ContentItem::ContentText {
        text: "sys".to_string()
      }]
    );
  }

  // ListWithImageType with image LAST + audio (text_first path, lines
  // 1731–1745). ListWithImageTypeTextImageLast is declared but no model
  // maps to it; we still cover the text-first (image-last) ordering + audio
  // loop through ListWithImageType by using num_images>0 with image_first
  // true is image-first; to hit the text-first-then-images branch we use a
  // 0-image + audio case so `text_first` is true and the audio loop runs.
  #[test]
  fn list_with_image_type_text_first_with_audio() {
    let f = MessageFormatter::for_model("internvl_chat").unwrap();
    // image_first is true for ListWithImageType, but n_img==0 forces
    // text_first=true (line 1731), so [content_text, audio, audio].
    let it = items(&f.format_message("hi", &opts_with("user", 0, 2)).unwrap());
    assert_eq!(
      it,
      vec![
        ContentItem::ContentText {
          text: "hi".to_string()
        },
        ContentItem::Audio,
        ContentItem::Audio,
      ]
    );
  }

  // ── format_with_token: else-0 audio gate + audio prefix loop ────────────
  //
  // ImageToken format ("<image>", image_first). With audio, the audio prefix
  // `<|audio_i|>` is emitted FIRST (line 1835 loop), then image tokens, then
  // prompt.

  #[test]
  fn with_token_audio_prefix_then_images() {
    let f = MessageFormatter::for_model("minicpmo").unwrap();
    // minicpmo is single-image-only; use 1 image + 2 audios.
    let s = text_content(&f.format_message("hi", &opts_with("user", 1, 2)).unwrap());
    assert_eq!(s, "<|audio_1|><|audio_2|><image>hi");
  }

  #[test]
  fn with_token_assistant_no_media() {
    // assistant role → n_img=0, n_aud=0 (else branches incl. line 1782):
    // bare prompt, no tokens.
    let f = MessageFormatter::for_model("jina_vlm").unwrap();
    let s = text_content(
      &f.format_message("hi", &opts_with("assistant", 3, 3))
        .unwrap(),
    );
    assert_eq!(s, "hi");
  }

  #[test]
  fn with_token_skip_audio_only() {
    // skip_audio_token=true → n_aud=0 (else branch at line 1782), images
    // still emitted.
    let f = MessageFormatter::for_model("minicpmo").unwrap();
    let opts = FormatOpts {
      role: "user".to_string(),
      num_images: 1,
      num_audios: 3,
      skip_audio_token: true,
      ..FormatOpts::formatter_default()
    };
    let s = text_content(&f.format_message("hi", &opts).unwrap());
    assert_eq!(s, "<image>hi");
  }

  // ── format_numbered_tokens: else-0 gates (lines 1868, 1874) ─────────────

  #[test]
  fn numbered_tokens_assistant_no_media() {
    let f = MessageFormatter::for_model("phi3_v").unwrap();
    let s = text_content(
      &f.format_message("hi", &opts_with("assistant", 3, 3))
        .unwrap(),
    );
    assert_eq!(s, "hi");
  }

  #[test]
  fn numbered_tokens_skip_image_keeps_audio() {
    // skip_image_token → n_img=0 (line 1868 else), audio still numbered.
    let f = MessageFormatter::for_model("phi4mm").unwrap();
    let opts = FormatOpts {
      role: "user".to_string(),
      num_images: 3,
      num_audios: 2,
      skip_image_token: true,
      ..FormatOpts::formatter_default()
    };
    let s = text_content(&f.format_message("hi", &opts).unwrap());
    assert_eq!(s, "<|audio_1|><|audio_2|>hi");
  }

  // ── check_format_count cap (CapExceeded) via format_message ─────────────
  //
  // num_images above MAX_MESSAGE_FORMAT_ITEMS → CapExceeded. Use a model
  // NOT in SINGLE_IMAGE_ONLY (qwen2_vl) so the single-image guard does not
  // pre-empt the cap check.

  #[test]
  fn format_count_cap_exceeded() {
    let f = MessageFormatter::for_model("qwen2_vl").unwrap();
    let opts = opts_with("user", MAX_MESSAGE_FORMAT_ITEMS + 1, 0);
    let err = f.format_message("hi", &opts).unwrap_err();
    assert!(matches!(err, Error::CapExceeded(_)));
    if let Error::CapExceeded(p) = &err {
      assert_eq!(p.cap(), MAX_MESSAGE_FORMAT_ITEMS as u64);
      assert_eq!(p.observed(), (MAX_MESSAGE_FORMAT_ITEMS + 1) as u64);
      assert_eq!(p.cap_name(), "MAX_MESSAGE_FORMAT_ITEMS");
    }
  }

  // At exactly the cap it must NOT error (boundary). Use a small prompt so
  // the 1025-element content Vec is a cheap allocation.
  #[test]
  fn format_count_at_cap_ok() {
    assert!(check_format_count(MAX_MESSAGE_FORMAT_ITEMS, "num_images", "m").is_ok());
    assert!(check_format_count(MAX_MESSAGE_FORMAT_ITEMS + 1, "num_images", "m").is_err());
  }

  // ── format_prompt_with_start_image_token (lines 1571–1599) ──────────────
  //
  // No model in MODEL_CONFIG maps to PromptWithStartImageToken, so it cannot
  // be reached through `format_message` dispatch. The helper is private;
  // we exercise its observable behavior through a hand-built formatter whose
  // format_type we set to PromptWithStartImageToken (struct fields are pub).

  #[test]
  fn prompt_with_start_image_token_direct() {
    let f = MessageFormatter {
      model_name: "synthetic".to_string(),
      format_type: MessageFormat::PromptWithStartImageToken,
    };
    let out = f.format_message("hi", &opts_with("user", 2, 0)).unwrap();
    assert_eq!(
      out,
      FormattedMessage::String("hi<start_of_image><start_of_image>".to_string())
    );
    // Unconditional even for assistant.
    let out = f
      .format_message("hi", &opts_with("assistant", 1, 0))
      .unwrap();
    assert_eq!(
      out,
      FormattedMessage::String("hi<start_of_image>".to_string())
    );
  }

  // ── ListWithImageTypeTextImageLast (lines 1489–1491) ────────────────────
  //
  // Also unreachable via MODEL_CONFIG; drive through a synthetic formatter.
  // image_first=false → text first, then images, then audio.

  #[test]
  fn list_with_image_type_text_image_last_direct() {
    let f = MessageFormatter {
      model_name: "synthetic".to_string(),
      format_type: MessageFormat::ListWithImageTypeTextImageLast,
    };
    let it = items(&f.format_message("hi", &opts_with("user", 2, 1)).unwrap());
    // text_message (Text kind), image LAST, then audio.
    assert_eq!(
      it,
      vec![
        ContentItem::Text {
          text: "hi".to_string()
        },
        ContentItem::Image,
        ContentItem::Image,
        ContentItem::Audio,
      ]
    );
  }

  // ── VideoWithText as the base format_type (line 1504) ───────────────────
  //
  // Reach the explicit `MessageFormat::VideoWithText => format_video_message`
  // dispatch arm (line 1504), distinct from the video special-case route
  // (lines 1470–1476). Built synthetically since no model maps to it. Also
  // covers the EmptyInput guard (lines 1946–1948) directly via this arm.

  #[test]
  fn video_with_text_base_format_arm() {
    let f = MessageFormatter {
      model_name: "synthetic".to_string(),
      format_type: MessageFormat::VideoWithText,
    };
    // Non-empty video → one Video + trailing text.
    let opts = FormatOpts {
      role: "user".to_string(),
      video: vec!["v.mp4".to_string()],
      fps: vec![4],
      max_pixels: 64,
      ..FormatOpts::formatter_default()
    };
    let it = items(&f.format_message("cap", &opts).unwrap());
    assert_eq!(
      it,
      vec![
        ContentItem::Video {
          video: "v.mp4".to_string(),
          max_pixels: 64,
          fps: 4,
        },
        ContentItem::Text {
          text: "cap".to_string()
        },
      ]
    );

    // Empty video through the VideoWithText arm → EmptyInput (lines 1946–1948).
    let empty = FormatOpts {
      role: "user".to_string(),
      video: Vec::new(),
      ..FormatOpts::formatter_default()
    };
    let err = f.format_message("cap", &empty).unwrap_err();
    assert!(matches!(err, Error::EmptyInput(_)));
    if let Error::EmptyInput(p) = &err {
      assert!(p.context().contains("format_video_message"));
    }
  }

  // ── get_message_json: None opts → public-API defaults (0 images) ────────

  #[test]
  fn get_message_json_default_text_only() {
    // qwen2_vl + None → num_images=0 → content == [text] only.
    let out = get_message_json("qwen2_vl", "hi", None).unwrap();
    let it = items(&out);
    assert_eq!(
      it,
      vec![ContentItem::Text {
        text: "hi".to_string()
      }]
    );
  }

  #[test]
  fn get_message_json_explicit_opts_and_unknown_model() {
    // Explicit opts with 1 image → image appears.
    let opts = opts_with("user", 1, 0);
    let out = get_message_json("qwen2_vl", "hi", Some(&opts)).unwrap();
    assert_eq!(
      items(&out),
      vec![
        ContentItem::Text {
          text: "hi".to_string()
        },
        ContentItem::Image,
      ]
    );
    // Unknown model propagates MissingKey.
    let err = get_message_json("does_not_exist", "hi", None).unwrap_err();
    assert!(matches!(err, Error::MissingKey(_)));
  }

  // Helper: an assistant collapse-to-string expectation.
  fn text_content_msg(role: &str, text: &str) -> FormattedMessage {
    FormattedMessage::Message(Message {
      role: role.to_string(),
      content: MessageContent::Text(text.to_string()),
    })
  }

  // ── const-table sort invariants ─────────────────────────────────────────
  //
  // `for_model` (binary_search_by over MODEL_CONFIG) and `format_message`
  // (binary_search over SINGLE_IMAGE_ONLY_MODELS and VIDEO_FORMAT_MODELS) all
  // assume their tables are sorted lexicographically by key. An unsorted
  // insertion in a future edit would silently break those lookups (a present
  // model would `Err(MissingKey)` / a single-image model would skip its
  // guard) WITHOUT a compile error. Pin the invariant with a closed-form
  // monotonicity check (independent of the production binary-search code).

  #[test]
  fn model_config_keys_strictly_sorted_and_unique() {
    for pair in MODEL_CONFIG.windows(2) {
      let (a, _) = pair[0];
      let (b, _) = pair[1];
      assert!(
        a < b,
        "MODEL_CONFIG must be strictly ascending for binary_search_by: {a:?} !< {b:?}"
      );
    }
  }

  #[test]
  fn single_image_only_models_strictly_sorted_and_unique() {
    for pair in SINGLE_IMAGE_ONLY_MODELS.windows(2) {
      assert!(
        pair[0] < pair[1],
        "SINGLE_IMAGE_ONLY_MODELS must be strictly ascending for binary_search: \
         {:?} !< {:?}",
        pair[0],
        pair[1]
      );
    }
  }

  #[test]
  fn video_format_models_strictly_sorted_and_unique() {
    for pair in VIDEO_FORMAT_MODELS.windows(2) {
      assert!(
        pair[0] < pair[1],
        "VIDEO_FORMAT_MODELS must be strictly ascending for binary_search: {:?} !< {:?}",
        pair[0],
        pair[1]
      );
    }
  }

  // Every MODEL_CONFIG key must resolve through `for_model` to the table's
  // declared format (round-trips the binary_search lookup against the
  // independently-iterated table; also confirms lowercased keys are
  // already-lowercase so `to_lowercase()` is a no-op for the canonical
  // entries).
  #[test]
  fn model_config_every_key_resolves() {
    for &(key, fmt) in MODEL_CONFIG {
      let f = MessageFormatter::for_model(key).unwrap();
      assert_eq!(f.format_type, fmt, "for_model({key:?}) format mismatch");
      assert_eq!(
        f.model_name, key,
        "for_model({key:?}) should lowercase to itself"
      );
    }
  }

  // Every SINGLE_IMAGE_ONLY_MODELS entry that is ALSO in MODEL_CONFIG must
  // reject num_images>1 (the guard at format_message lines 1452–1463). This
  // exercises the guard against the actual allow-list rather than a single
  // hand-picked model.
  #[test]
  fn single_image_only_models_reject_multi_image() {
    for &model in SINGLE_IMAGE_ONLY_MODELS {
      // Only those present in MODEL_CONFIG are constructible via for_model.
      if let Ok(f) = MessageFormatter::for_model(model) {
        let err = f
          .format_message("hi", &opts_with("user", 2, 0))
          .unwrap_err();
        assert!(
          matches!(err, Error::OutOfRange(_)),
          "model {model:?} should reject num_images=2, got {err:?}"
        );
      }
    }
  }

  // ── MESSAGE_FORMAT_VARIANTS ↔ as_str round-trip (no two tags collide) ───
  //
  // Independent of `message_format_as_str_all_variants` (which checks fixed
  // expected strings): here we assert the 15 tags are pairwise-distinct, so
  // the snake_case mapping is injective (a duplicated tag would make two
  // formats indistinguishable to a string-driven chat-template renderer).
  #[test]
  fn message_format_tags_are_pairwise_distinct() {
    let tags: Vec<&'static str> = MESSAGE_FORMAT_VARIANTS.iter().map(|f| f.as_str()).collect();
    for i in 0..tags.len() {
      for j in (i + 1)..tags.len() {
        assert_ne!(
          tags[i], tags[j],
          "duplicate MessageFormat tag {:?}",
          tags[i]
        );
      }
    }
  }

  // ── locate_image_tokens: adjacent-run collapse contract (closed-form) ────
  //
  // The module doc states adjacent multi-image placeholders collapse into a
  // single contiguous run; a single non-marker token between two marker runs
  // splits them. Hand-written oracle, byte-exact spans.
  #[test]
  fn locate_image_tokens_collapse_and_split() {
    // Two runs separated by one text token (split into two spans).
    assert_eq!(
      locate_image_tokens(&[5, 9, 9, 5, 9, 5], 9),
      vec![(1, 3), (4, 5)]
    );
    // A contiguous block of 4 markers is ONE run (collapse), even though it
    // could represent several images post-tokenization.
    assert_eq!(locate_image_tokens(&[9, 9, 9, 9], 9), vec![(0, 4)]);
  }

  // ── build_multimodal_mask: span end == seq_len is in-bounds (boundary) ──
  //
  // The end-vs-seq_len check is strict (`e > seq_len` errors), so a span
  // whose end equals seq_len is valid. Closed-form oracle over the resulting
  // mask (block_id = [0, 1, 1] for span (1,3) in seq_len=3).
  #[test]
  fn mask_span_end_equals_seq_len_is_valid() {
    let mut mask = build_multimodal_mask(3, &[(1, 3)]).unwrap();
    assert_eq!(mask.shape(), vec![1, 1, 3, 3]);
    let v = mask.to_vec::<bool>().unwrap();
    let block = [0u32, 1, 1];
    for q in 0..3usize {
      for k in 0..3usize {
        let causal = k <= q;
        let same_img = block[q] != 0 && block[q] == block[k];
        assert_eq!(v[q * 3 + k], causal || same_img, "mask[{q}][{k}]");
      }
    }
  }

  // ── format_list_with_image: image-first model in a NON-user role ────────
  //
  // Distinct from `list_with_image_assistant_role_no_images` (which uses an
  // image_first=FALSE model, qwen2_vl). Here an image_first=TRUE model
  // (qwen2_5_vl) in the assistant role takes the `effective_n == 0` →
  // text_first short-circuit (line 1636 `!image_first || effective_n == 0`),
  // emitting only the text item with no trailing image push.
  #[test]
  fn list_with_image_first_assistant_role_text_only() {
    let f = MessageFormatter::for_model("qwen2_5_vl").unwrap();
    let it = items(
      &f.format_message("hi", &opts_with("assistant", 4, 0))
        .unwrap(),
    );
    assert_eq!(
      it,
      vec![ContentItem::Text {
        text: "hi".to_string()
      }]
    );
  }

  // ── get_message_json: String branch (PromptOnly) via the top-level helper ─
  //
  // The existing get_message_json tests only reach the Items branch. Route a
  // PromptOnly model (florence2) through get_message_json with explicit opts
  // to cover the FormattedMessage::String return flowing back through the
  // free function.
  #[test]
  fn get_message_json_prompt_only_string_branch() {
    let opts = opts_with("user", 1, 0);
    let out = get_message_json("florence2", "describe", Some(&opts)).unwrap();
    assert_eq!(out, FormattedMessage::String("describe".to_string()));
  }
}
