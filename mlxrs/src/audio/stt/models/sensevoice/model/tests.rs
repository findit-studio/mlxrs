//! Oracles for the SenseVoice-Small CTC head, query-prefix, decode, rich-info,
//! and golden-trait wiring.
//!
//! Every expected value is computed independently of the code under test: the
//! greedy collapse by a hand-written run-length-dedup + drop-blank over a known
//! argmax sequence, the rich-info mapping by the literal `lid/emo/event` id
//! tables, the query-prefix layout by an `embed` table whose row `i` is the
//! constant `i` (so a gathered row is identifiable), the log-softmax by the
//! closed-form `x - logsumexp(x)`, and the speech-only slice by frame counting —
//! never by invoking the implementation a second time. A tiny synthetic model
//! (1 `encoders0` block, 0 `encoders`, 0 `tp`) exercises the full forward + both
//! trait routes at trivial size.

use std::collections::HashMap;

use super::*;
use crate::{
  array::Array,
  audio::stt::model::{Transcribe, TranscribeExt, TranscribeOptions},
  error::Error,
  lm::quant::{PerLayerQuantization, Quantization},
};

/// The global affine quantization config the head quantized-path oracle threads
/// in (the common SenseVoice scheme). `build_head` resolves `ctc_lo` / `embed`
/// per prefix from it via `quantization_for`.
fn global_quant(group_size: i32, bits: i32) -> PerLayerQuantization {
  PerLayerQuantization::from_global(Quantization::affine(group_size, bits))
}

// ───────────────────────────── tiny synthetic model ─────────────────────────────

/// Tiny dims: input width `D`, hidden width `H`, 1 head, FFN width `F`, vocab
/// `V`. `H` and `D` are kept small but `H % 1 == 0` (1 head) and `D >= 2` (the
/// sinusoidal PE divisor). `V` is large enough to host a couple of distinct CTC
/// ids in the decode tests.
///
/// `D` is also the front-end's mel count (`n_mels = D`, with `lfr_m = 1` so the
/// LFR step does not stack): a real fbank then produces `(T', D)` features that
/// match the `D`-wide `embed` table and the first encoder block's `in_size`.
const D: i32 = 8; // input_size == n_mels (lfr_m = 1, no stacking)
const H: i32 = 4; // output_size (hidden)
const F: i32 = 6; // linear_units
const V: i32 = 32; // vocab_size
const K: i32 = 3; // FSMN kernel size

/// A `(rows, cols)` array filled by `f(r, c)` — a deterministic test weight.
fn filled(rows: i32, cols: i32, f: impl Fn(i32, i32) -> f32) -> Array {
  let mut data = Vec::with_capacity((rows * cols) as usize);
  for r in 0..rows {
    for c in 0..cols {
      data.push(f(r, c));
    }
  }
  Array::from_slice::<f32>(&data, &[rows, cols]).unwrap()
}

/// A 1-D `(n,)` array filled by `f(i)`.
fn filled1(n: i32, f: impl Fn(i32) -> f32) -> Array {
  let data: Vec<f32> = (0..n).map(f).collect();
  Array::from_slice::<f32>(&data, &[n]).unwrap()
}

/// A config with the tiny dims: 1 `encoders0` block (num_blocks = 1), 0
/// `encoders`, 0 `tp_blocks`, 1 head, kernel `K`. The front-end uses `n_mels =
/// D` and `lfr_m = lfr_n = 1` so the real fbank -> LFR yields `D`-wide features
/// (no stacking), matching the encoder + `embed` width.
fn tiny_config() -> Config {
  let json = format!(
    r#"{{
      "model_type": "sensevoice",
      "vocab_size": {V},
      "input_size": {D},
      "encoder_conf": {{
        "output_size": {H},
        "attention_heads": 1,
        "linear_units": {F},
        "num_blocks": 1,
        "tp_blocks": 0,
        "kernel_size": {K}
      }},
      "frontend_conf": {{ "n_mels": {D}, "lfr_m": 1, "lfr_n": 1 }}
    }}"#
  );
  let config: Config = serde_json::from_str(&json).unwrap();
  config.validate().unwrap();
  config
}

/// The full dense weight map for the tiny encoder + head. `embed` row `i` is the
/// constant `i` across all `D` columns, so a gathered query row is identifiable
/// by its (constant) value.
fn tiny_weights() -> HashMap<String, Array> {
  let mut w: HashMap<String, Array> = HashMap::new();

  // encoders0.0.self_attn: linear_q_k_v (3H, D), linear_out (H, H),
  // fsmn_block.weight (H, K, 1) [post-sanitize MLX layout]. Small distinct
  // values keep the forward finite.
  w.insert(
    "encoder.encoders0.0.self_attn.linear_q_k_v.weight".to_string(),
    filled(3 * H, D, |r, c| 0.01 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.self_attn.linear_out.weight".to_string(),
    filled(H, H, |r, c| if r == c { 0.5 } else { 0.0 }),
  );
  // fsmn weight (H, K, 1): all zeros -> the FSMN branch is the pure `+ inputs`
  // residual (so the forward stays simple + finite).
  w.insert(
    "encoder.encoders0.0.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (H * K) as usize], &[H, K, 1]).unwrap(),
  );
  // feed_forward: w_1 (F, H), w_2 (H, F).
  w.insert(
    "encoder.encoders0.0.feed_forward.w_1.weight".to_string(),
    filled(F, H, |r, c| 0.02 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.feed_forward.w_2.weight".to_string(),
    filled(H, F, |r, c| 0.02 * ((r + c) as f32)),
  );
  // norm1 (D,), norm2 (H,): unit weight, zero bias (identity LayerNorm scale).
  w.insert(
    "encoder.encoders0.0.norm1.weight".to_string(),
    filled1(D, |_| 1.0),
  );
  w.insert(
    "encoder.encoders0.0.norm1.bias".to_string(),
    filled1(D, |_| 0.0),
  );
  w.insert(
    "encoder.encoders0.0.norm2.weight".to_string(),
    filled1(H, |_| 1.0),
  );
  w.insert(
    "encoder.encoders0.0.norm2.bias".to_string(),
    filled1(H, |_| 0.0),
  );
  // after_norm (H,), tp_norm (H,).
  w.insert("encoder.after_norm.weight".to_string(), filled1(H, |_| 1.0));
  w.insert("encoder.after_norm.bias".to_string(), filled1(H, |_| 0.0));
  w.insert("encoder.tp_norm.weight".to_string(), filled1(H, |_| 1.0));
  w.insert("encoder.tp_norm.bias".to_string(), filled1(H, |_| 0.0));

  // ctc_lo (V, H) + bias (V,).
  w.insert(
    "ctc_lo.weight".to_string(),
    filled(V, H, |r, c| 0.03 * ((r + c) as f32)),
  );
  w.insert("ctc_lo.bias".to_string(), filled1(V, |_| 0.0));

  // embed (16, D): row i = constant i.
  w.insert("embed.weight".to_string(), filled(16, D, |r, _| r as f32));

  w
}

/// Build a tiny dense `SenseVoiceModel` with the given detokenizer. The encoder
/// is built through the real [`Encoder::from_weights`], the head through
/// [`build_head`], and the model assembled with no CMVN statistics.
fn tiny_model(tokenizer: SenseVoiceTokenizer) -> SenseVoiceModel {
  let config = tiny_config();
  let mut w = tiny_weights();
  let encoder =
    Encoder::from_weights(&mut w, config.input_size(), config.encoder_conf(), None).unwrap();
  let (ctc_lo, embed) = build_head(
    &mut w,
    config.input_size(),
    encoder.output_size(),
    config.vocab_size(),
    None,
  )
  .unwrap();
  SenseVoiceModel::new(config, encoder, ctc_lo, embed, tokenizer, None, None)
}

// ───────────────────────────── log_softmax ─────────────────────────────

#[test]
fn log_softmax_matches_closed_form() {
  // log_softmax(x)[i] = x[i] - logsumexp(x) over the last axis. Independent
  // closed-form reference over a small (2, 3) grid.
  let rows = [[1.0f32, 2.0, 3.0], [0.5, -1.0, 2.0]];
  let flat: Vec<f32> = rows.iter().flatten().copied().collect();
  let x = Array::from_slice::<f32>(&flat, &[2, 3]).unwrap();
  let mut out = log_softmax_last_axis(&x).unwrap();
  assert_eq!(out.shape(), vec![2, 3]);
  let got = out.to_vec::<f32>().unwrap();

  for (r, row) in rows.iter().enumerate() {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = row.iter().map(|v| (v - max).exp()).sum();
    let lse = max + sum_exp.ln();
    for (c, &v) in row.iter().enumerate() {
      let want = v - lse;
      let g = got[r * 3 + c];
      assert!((g - want).abs() < 1e-5, "[{r},{c}] got {g} want {want}");
    }
  }
}

#[test]
fn log_softmax_rows_are_normalized() {
  // exp(log_softmax) sums to 1 along the last axis.
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
  let lp = log_softmax_last_axis(&x).unwrap().to_vec::<f32>().unwrap();
  let sum: f32 = lp.iter().map(|v| v.exp()).sum();
  assert!((sum - 1.0).abs() < 1e-5, "sum of softmax = {sum}");
}

// ───────────────────────────── greedy CTC collapse ─────────────────────────────

/// Build a `(t, vocab)` log-prob grid whose per-frame argmax is exactly
/// `peaks[t]` (a one-hot-ish row: the peak id gets a large value, the rest 0).
fn grid_with_argmax(peaks: &[u32], vocab: i32) -> Array {
  let t = peaks.len() as i32;
  let mut data = vec![0.0f32; (t * vocab) as usize];
  for (ti, &p) in peaks.iter().enumerate() {
    data[ti * vocab as usize + p as usize] = 10.0;
  }
  Array::from_slice::<f32>(&data, &[t, vocab]).unwrap()
}

/// Independent run-length-dedup + drop-blank reference (`sensevoice.py:454-461`).
fn collapse_ref(argmax: &[u32], blank: u32) -> Vec<u32> {
  let mut deduped = Vec::new();
  let mut prev: Option<u32> = None;
  for &t in argmax {
    if prev != Some(t) {
      deduped.push(t);
      prev = Some(t);
    }
  }
  deduped.into_iter().filter(|&t| t != blank).collect()
}

#[test]
fn greedy_collapse_dedups_then_drops_blank() {
  // [3,3,0,3,5,5] -> dedup [3,0,3,5] -> drop 0 -> [3,5,3] (wait: dedup keeps
  // the second 3 since a blank separates the runs). Verify against the ref.
  let peaks = [3u32, 3, 0, 3, 5, 5];
  let grid = grid_with_argmax(&peaks, V);
  let got = SenseVoiceModel::greedy_collapse(&grid).unwrap();
  let want = collapse_ref(&peaks, BLANK_ID);
  assert_eq!(got, want);
  assert_eq!(got, vec![3, 3, 5]); // [3,3,0,3,5,5]->[3,0,3,5]->[3,3,5]
}

#[test]
fn greedy_collapse_leading_blanks_dropped() {
  // [0,0,7] -> dedup [0,7] -> drop 0 -> [7].
  let peaks = [0u32, 0, 7];
  let grid = grid_with_argmax(&peaks, V);
  let got = SenseVoiceModel::greedy_collapse(&grid).unwrap();
  assert_eq!(got, collapse_ref(&peaks, BLANK_ID));
  assert_eq!(got, vec![7]);
}

#[test]
fn greedy_collapse_all_blank_is_empty() {
  let peaks = [0u32, 0, 0, 0];
  let grid = grid_with_argmax(&peaks, V);
  let got = SenseVoiceModel::greedy_collapse(&grid).unwrap();
  assert!(got.is_empty());
}

#[test]
fn greedy_collapse_adjacent_distinct_all_kept() {
  // No repeats, no blanks -> every id survives.
  let peaks = [1u32, 2, 3, 4, 5];
  let grid = grid_with_argmax(&peaks, V);
  let got = SenseVoiceModel::greedy_collapse(&grid).unwrap();
  assert_eq!(got, vec![1, 2, 3, 4, 5]);
}

#[test]
fn greedy_collapse_fallible_path_matches_oracle() {
  // The collapsed-id buffer is reserved FALLIBLY (`reserve_or_error`, bounded by
  // the `T'` frame count). A normal-size synthetic argmax sequence must collapse
  // correctly THROUGH that fallible path and equal the independent run-length
  // dedup + drop-blank oracle (`collapse_ref`) — the allocation became fallible
  // without changing the collapse output.
  let peaks = [
    4u32, 4, 0, 9, 9, 9, 0, 0, 7, 7, 3, 3, 3, 0, 11, 11, 5, 0, 0, 8,
  ];
  let grid = grid_with_argmax(&peaks, V);
  let got = SenseVoiceModel::greedy_collapse(&grid).expect("fallible collapse reserves Ok");
  // Independent oracle: dedup consecutive ids, then drop the blank id.
  let want = collapse_ref(&peaks, BLANK_ID);
  assert_eq!(got, want);
  // Pin the known expected sequence: dedup -> [4,0,9,0,7,3,0,11,5,0,8] -> drop 0
  // -> [4,9,7,3,11,5,8].
  assert_eq!(got, vec![4, 9, 7, 3, 11, 5, 8]);
  // The reserved capacity is bounded by `T'` (the collapse never grows the Vec).
  assert!(got.len() <= peaks.len());
}

// ───────────────────────────── rich-info argmax ─────────────────────────────

/// Build a `(frames, vocab)` grid where frame `f` peaks at `peak_ids[f]`. Vocab
/// is sized to host the rich-info ids (which run up to ~25009).
fn rich_grid(peak_ids: &[u32]) -> Array {
  let vocab: i32 = 25_055;
  let frames = peak_ids.len() as i32;
  let mut data = vec![0.0f32; (frames * vocab) as usize];
  for (f, &p) in peak_ids.iter().enumerate() {
    data[f * vocab as usize + p as usize] = 10.0;
  }
  Array::from_slice::<f32>(&data, &[frames, vocab]).unwrap()
}

#[test]
fn rich_info_maps_argmax_to_labels() {
  // Frame 0 peaks at 24885 -> "en", frame 1 at 25004 -> "neutral", frame 2 at
  // 24993 -> "Speech" (`sensevoice.py:469-500`). Frames 3+ are present but
  // unread; add a 4th frame so the grid has the 4 query frames.
  let tok = tiny_model(SenseVoiceTokenizer::id_join());
  let grid = rich_grid(&[24885, 25004, 24993, 0]);
  let rich = tok.rich_info(&grid).unwrap();
  assert_eq!(rich.language(), "en");
  assert_eq!(rich.emotion(), "neutral");
  assert_eq!(rich.event(), "Speech");
}

#[test]
fn rich_info_unknown_ids_fall_back() {
  // Unrecognized lid -> "unknown"; unrecognized emo / event -> "token_<id>"
  // (`sensevoice.py:477/491/500`).
  let tok = tiny_model(SenseVoiceTokenizer::id_join());
  let grid = rich_grid(&[999, 12345, 6789, 0]);
  let rich = tok.rich_info(&grid).unwrap();
  assert_eq!(rich.language(), "unknown");
  assert_eq!(rich.emotion(), "token_12345");
  assert_eq!(rich.event(), "token_6789");
}

#[test]
fn rich_info_reads_only_frames_0_1_2() {
  // Frame 3 (the textnorm slot) must NOT influence the rich info (plan §9 Q8):
  // peak frame 3 at a known emotion id; the emotion must still come from frame 1.
  let tok = tiny_model(SenseVoiceTokenizer::id_join());
  let grid = rich_grid(&[24884, 25001, 24995, 25004]);
  let rich = tok.rich_info(&grid).unwrap();
  assert_eq!(rich.language(), "zh"); // frame 0 = 24884
  assert_eq!(rich.emotion(), "happy"); // frame 1 = 25001, NOT frame 3's 25004
  assert_eq!(rich.event(), "BGM"); // frame 2 = 24995
}

#[test]
fn build_query_layout_and_rich_info_agree_on_emotion_event() {
  // Integrated swap-detector coupling `build_query` (which ABSOLUTE embed row
  // lands at which prefix index) with `rich_info` (which prefix-frame index
  // decodes as emotion vs event). Both sides are pinned to the reference's
  // hardcoded layout — NOT to the port's named constants — so a divergence on
  // EITHER side is caught:
  //
  //  * the reference gathers `event_emo_query = embed([[1, 2]])`
  //    (`sensevoice.py:410`): embed row 1 lands at prefix index 1, row 2 at
  //    index 2 (a constant-row `embed` makes the absolute row id the gathered
  //    value). Swapping the gather order to `[2, 1]` is detected here.
  //  * the reference decodes frame 1 -> emotion (`sensevoice.py:479`) and frame
  //    2 -> event (`sensevoice.py:493`). Swapping `rich_info`'s frame->label is
  //    detected by the grid asserts below.
  //
  // Together they pin: embed row 1 == the emotion head, embed row 2 == the event
  // head — agreeing with the reference. A real forward on the (former) swapped
  // mapping would decode the emotion query as event and vice-versa.
  let model = tiny_model(SenseVoiceTokenizer::id_join());

  // (1) build_query layout: the `embed` table has row i = constant value i, so a
  // gathered query row's ABSOLUTE row id IS its column-0 value. The reference
  // `embed([[1, 2]])` pins prefix index 1 -> embed row 1, index 2 -> embed row 2.
  let (_textnorm, mut input_query) = model.build_query(1, "auto", false).unwrap();
  assert_eq!(input_query.shape(), vec![1, 3, D as usize]);
  let iq = input_query.to_vec::<f32>().unwrap();
  let gathered_row = |frame: usize| iq[frame * D as usize] as i32;
  assert_eq!(
    gathered_row(1),
    1,
    "prefix index 1 must gather embed row 1 (reference embed([[1, 2]]))"
  );
  assert_eq!(
    gathered_row(2),
    2,
    "prefix index 2 must gather embed row 2 (reference embed([[1, 2]]))"
  );

  // (2) rich_info extraction, pinned to the reference's ABSOLUTE frame indices:
  // a known emotion id at frame 1, a known event id at frame 2, a known language
  // id at frame 0 (frame 3, the textnorm slot, is unread). `rich_info` must read
  // frame 1 -> emotion and frame 2 -> event.
  const LANG_ID: u32 = 24885; // "en"
  const EMO_ID: u32 = 25002; // "sad"   (the FRAME-1 emotion head)
  const EVENT_ID: u32 = 24997; // "Laughter" (the FRAME-2 event head)
  let grid = rich_grid(&[LANG_ID, EMO_ID, EVENT_ID, 0]);
  let rich = model.rich_info(&grid).unwrap();
  assert_eq!(rich.language(), "en");
  assert_eq!(
    rich.emotion(),
    "sad",
    "frame 1 (the emotion query head, embed row 1) must decode as emotion"
  );
  assert_eq!(
    rich.event(),
    "Laughter",
    "frame 2 (the event query head, embed row 2) must decode as event"
  );

  // (3) the full real path runs build_query -> forward -> rich_info end-to-end
  // without panicking, yielding well-formed tags off the prepended query frames.
  let t = 4i32;
  let feats = filled(t, D, |r, c| 0.05 * ((r + c) as f32));
  let feats = crate::ops::shape::reshape(&feats, &[1, t, D]).unwrap();
  let log_probs = model.forward(&feats, "auto", false).unwrap();
  let utt = crate::ops::shape::squeeze_axes(&log_probs, &[0]).unwrap();
  let rich_fwd = model.rich_info(&utt).unwrap();
  assert!(!rich_fwd.emotion().is_empty());
  assert!(!rich_fwd.event().is_empty());
}

#[test]
fn rich_info_rejects_too_few_frames() {
  let tok = tiny_model(SenseVoiceTokenizer::id_join());
  let grid = rich_grid(&[24884, 25001, 24995]); // only 3 frames < QUERY_FRAMES
  assert!(matches!(tok.rich_info(&grid), Err(Error::OutOfRange(_))));
}

// ───────────────────────────── speech-only slice ─────────────────────────────

#[test]
fn speech_frames_drops_the_four_query_rows() {
  // A (4 + 3, vocab) grid -> the speech slice is the trailing 3 frames
  // (`log_probs[4:]`, `sensevoice.py:533`). Tag each frame's argmax distinctly
  // so the surviving rows are identifiable.
  let grid = grid_with_argmax(&[1, 2, 3, 4, 5, 6, 7], V);
  let speech = SenseVoiceModel::speech_frames(&grid).unwrap();
  assert_eq!(speech.shape(), vec![3, V as usize]);
  // The speech argmax must be [5, 6, 7] (frames 4, 5, 6).
  let mut arg = crate::ops::misc::argmax(&speech, Some(1), false).unwrap();
  assert_eq!(arg.to_vec::<u32>().unwrap(), vec![5, 6, 7]);
}

#[test]
fn speech_frames_exactly_four_is_empty() {
  // A grid with exactly the 4 query frames -> an empty (0, vocab) speech slice.
  let grid = grid_with_argmax(&[1, 2, 3, 4], V);
  let speech = SenseVoiceModel::speech_frames(&grid).unwrap();
  assert_eq!(speech.shape(), vec![0, V as usize]);
}

#[test]
fn speech_frames_rejects_too_few() {
  let grid = grid_with_argmax(&[1, 2, 3], V);
  assert!(matches!(
    SenseVoiceModel::speech_frames(&grid),
    Err(Error::OutOfRange(_))
  ));
}

// ───────────────────────────── query-prefix layout ─────────────────────────────

#[test]
fn build_query_injects_lid_emotion_event_textnorm_rows() {
  // The `embed` table has row i = constant i; a gathered query row is therefore
  // identifiable by its constant value. Assert the prefix layout
  // [language(lid), emotion(1), event(2)] for `input_query` and textnorm(14/15)
  // for `textnorm_query` (`sensevoice.py:403-424`): the `event_emo_query =
  // embed([[1, 2]])` pair is decoded frame 1 -> emotion (`sensevoice.py:479`),
  // frame 2 -> event (`sensevoice.py:493`), so embed row 1 is the emotion head.
  let model = tiny_model(SenseVoiceTokenizer::id_join());

  // language = "en" -> lid row 4; use_itn = true -> textnorm row 14.
  let (mut textnorm_query, mut input_query) = model.build_query(1, "en", true).unwrap();

  // input_query is (1, 3, D): rows [lid=4, emotion=row 1, event=row 2].
  assert_eq!(input_query.shape(), vec![1, 3, D as usize]);
  let iq = input_query.to_vec::<f32>().unwrap();
  // Row r is constant value `row_id`; read column 0 of each of the 3 rows.
  let row_val = |r: usize| iq[r * D as usize];
  assert_eq!(row_val(0), 4.0, "frame 0 = language (lid=en=4)");
  assert_eq!(row_val(1), 1.0, "frame 1 = emotion (embed row 1)");
  assert_eq!(row_val(2), 2.0, "frame 2 = event (embed row 2)");

  // textnorm_query is (1, 1, D): row 14 (withitn).
  assert_eq!(textnorm_query.shape(), vec![1, 1, D as usize]);
  let tq = textnorm_query.to_vec::<f32>().unwrap();
  assert_eq!(tq[0], 14.0, "textnorm = withitn (row 14)");
}

#[test]
fn build_query_auto_language_and_woitn() {
  // language = "auto" -> lid row 0; use_itn = false -> textnorm row 15 (woitn).
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let (mut textnorm_query, mut input_query) = model.build_query(1, "auto", false).unwrap();
  let iq = input_query.to_vec::<f32>().unwrap();
  assert_eq!(iq[0], 0.0, "frame 0 = language (lid=auto=0)");
  let tq = textnorm_query.to_vec::<f32>().unwrap();
  assert_eq!(tq[0], 15.0, "textnorm = woitn (row 15)");
}

#[test]
fn build_query_unknown_language_falls_back_to_auto() {
  // An unrecognized language code resolves to lid 0 (`sensevoice.py:403`).
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let (_textnorm, mut input_query) = model.build_query(1, "klingon", false).unwrap();
  let iq = input_query.to_vec::<f32>().unwrap();
  assert_eq!(iq[0], 0.0, "unknown language -> lid auto (0)");
}

#[test]
fn build_query_broadcasts_over_batch() {
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let (textnorm_query, mut input_query) = model.build_query(3, "zh", false).unwrap();
  assert_eq!(input_query.shape(), vec![3, 3, D as usize]);
  assert_eq!(textnorm_query.shape(), vec![3, 1, D as usize]);
  // Every batch element carries the same lid row (3 = zh) in frame 0.
  let iq = input_query.to_vec::<f32>().unwrap();
  for b in 0..3 {
    let frame0 = iq[b * 3 * D as usize];
    assert_eq!(frame0, 3.0, "batch {b} frame 0 = lid zh (3)");
  }
}

// ───────────────────────────── forward shape + CTC head ─────────────────────────────

#[test]
fn forward_prepends_four_query_frames_and_log_softmaxes() {
  // feats (1, T, D) -> log_probs (1, T+4, V), each frame a valid log-softmax
  // (exp sums to ~1).
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let t = 5i32;
  let feats = filled(t, D, |r, c| 0.1 * ((r + c) as f32));
  let feats = crate::ops::shape::reshape(&feats, &[1, t, D]).unwrap();
  let mut log_probs = model.forward(&feats, "auto", false).unwrap();
  assert_eq!(log_probs.shape(), vec![1, (t + 4) as usize, V as usize]);

  // Spot-check the first frame is a normalized log-prob distribution.
  let lp = log_probs.to_vec::<f32>().unwrap();
  let frame0: f32 = (0..V as usize).map(|v| lp[v].exp()).sum();
  assert!(
    (frame0 - 1.0).abs() < 1e-4,
    "frame 0 softmax sum = {frame0}"
  );
}

#[test]
fn forward_rejects_non_rank3_feats() {
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let bad = Array::from_slice::<f32>(&[0.0; D as usize], &[D]).unwrap();
  assert!(matches!(
    model.forward(&bad, "auto", false),
    Err(Error::RankMismatch(_))
  ));
}

// ───────────────────────────── CtcModel::logits speech-only ─────────────────────────────

/// Build a 1-D mono waveform of `n` samples (a simple ramp), enough to produce a
/// few LFR frames through the real front-end.
fn ramp_waveform(n: i32) -> Array {
  let data: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();
  Array::from_slice::<f32>(&data, &[n]).unwrap()
}

#[test]
fn ctc_logits_returns_speech_only_rank2() {
  // CtcModel::logits runs the front-end + encoder + ctc_lo + log_softmax, then
  // slices off the 4 query frames -> rank-2 (T', V) with T' >= 0.
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let wav = ramp_waveform(4000);
  let logits = CtcModel::logits(&model, &wav).unwrap();
  assert_eq!(logits.shape().len(), 2, "speech-only logits are rank-2");
  assert_eq!(logits.shape()[1], V as usize, "vocab axis = V");
}

#[test]
fn ctc_blank_id_is_zero() {
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  assert_eq!(CtcModel::blank_id(&model), 0);
  assert_eq!(BLANK_ID, 0);
}

#[test]
fn ctc_decode_ids_routes_through_tokenizer() {
  // decode_ids over a token-list tokenizer renders the pieces.
  let tokens = vec![
    "<blank>".to_string(),
    "\u{2581}hi".to_string(),
    "\u{2581}there".to_string(),
  ];
  let model = tiny_model(SenseVoiceTokenizer::from_token_list(tokens));
  assert_eq!(CtcModel::decode_ids(&model, &[1, 2]), "hi there");
}

// ───────────────────────────── Transcribe end-to-end ─────────────────────────────

#[test]
fn transcribe_produces_text_and_language_segment() {
  // The universal Transcribe runs ONE forward, fills the LID-head language at
  // the top level, and carries the text in a single full-utterance segment.
  let tokens = vec!["<blank>".to_string(), "\u{2581}a".to_string()];
  let model = tiny_model(SenseVoiceTokenizer::from_token_list(tokens));
  let wav = ramp_waveform(4000);
  let out = model
    .transcribe(&wav, &TranscribeOptions::default())
    .unwrap();
  // Exactly one segment spanning the utterance.
  assert_eq!(out.segments_slice().len(), 1);
  assert_eq!(out.segments_slice()[0].text(), out.text());
  // The top-level language is either a detected label or None (unknown
  // sentinel); for the tiny random model it is one of the lid labels or None.
  let lang_ok = match out.language() {
    None => true,
    Some(l) => ["zh", "en", "yue", "ja", "ko", "nospeech"].contains(&l),
  };
  assert!(lang_ok, "language = {:?}", out.language());
}

#[test]
fn transcribe_audio_convenience_matches_default() {
  // The `TranscribeExt::transcribe_audio` convenience == transcribe with default
  // options.
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let wav = ramp_waveform(3500);
  let a = model.transcribe_audio(&wav).unwrap();
  let b = model
    .transcribe(&wav, &TranscribeOptions::default())
    .unwrap();
  assert_eq!(a.text(), b.text());
}

#[test]
fn transcribe_rich_carries_all_three_tags() {
  // The inherent rich transcription exposes language / emotion / event (the
  // model-local result the universal Segment cannot carry).
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let wav = ramp_waveform(4000);
  let rich = model.transcribe_rich(&wav, "auto", false).unwrap();
  // The tags are well-formed strings (exact values depend on the random head).
  assert!(!rich.rich().language().is_empty());
  assert!(!rich.rich().emotion().is_empty());
  assert!(!rich.rich().event().is_empty());
  // The text matches the round-trip through the CtcModel collapse path.
  let collapsed =
    SenseVoiceModel::greedy_collapse(&CtcModel::logits(&model, &wav).unwrap()).unwrap();
  assert_eq!(rich.text(), model.tokenizer_ref().decode(&collapsed));
}

#[test]
fn transcribe_language_option_sets_lid_query() {
  // An explicit language option conditions the LID query row; the call must
  // succeed and produce a transcription (the conditioning path is exercised).
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let wav = ramp_waveform(3000);
  let opts = TranscribeOptions::default().with_language("zh");
  let out = model.transcribe(&wav, &opts).unwrap();
  assert_eq!(out.segments_slice().len(), 1);
}

// ───────────────────────────── quantized ctc_lo path ─────────────────────────────

/// mlx supports only group sizes 32 / 64 / 128, so the quantized fixtures use a
/// larger config whose quantized-tensor input widths are multiples of `QGROUP`:
/// the `ctc_lo` input (`QH = output_size`) and the `embed` input (`QD =
/// input_size`) are both `>= QGROUP` and divisible by it.
const QGROUP: i32 = 32;
const QH: i32 = 32; // quant config output_size (ctc_lo input width)
const QD: i32 = 64; // quant config input_size (embed input width)
const QF: i32 = 64; // quant config linear_units
const QV: i32 = 40; // quant config vocab

/// A quant-friendly config: `n_mels = QD`, `lfr_m = 1` (so the fbank LFR width
/// is `QD`), `output_size = QH`, 4 heads (QH % 4 == 0), 1 block, 0 tp.
fn quant_config() -> Config {
  let json = format!(
    r#"{{
      "model_type": "sensevoice",
      "vocab_size": {QV},
      "input_size": {QD},
      "encoder_conf": {{
        "output_size": {QH},
        "attention_heads": 4,
        "linear_units": {QF},
        "num_blocks": 1,
        "tp_blocks": 0,
        "kernel_size": {K}
      }},
      "frontend_conf": {{ "n_mels": {QD}, "lfr_m": 1, "lfr_n": 1 }}
    }}"#
  );
  let config: Config = serde_json::from_str(&json).unwrap();
  config.validate().unwrap();
  config
}

/// The dense weight map for the quant config (mirrors [`tiny_weights`] at the
/// larger dims).
fn quant_weights() -> HashMap<String, Array> {
  let mut w: HashMap<String, Array> = HashMap::new();
  w.insert(
    "encoder.encoders0.0.self_attn.linear_q_k_v.weight".to_string(),
    filled(3 * QH, QD, |r, c| 0.001 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.self_attn.linear_out.weight".to_string(),
    filled(QH, QH, |r, c| if r == c { 0.5 } else { 0.0 }),
  );
  w.insert(
    "encoder.encoders0.0.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (QH * K) as usize], &[QH, K, 1]).unwrap(),
  );
  w.insert(
    "encoder.encoders0.0.feed_forward.w_1.weight".to_string(),
    filled(QF, QH, |r, c| 0.001 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.feed_forward.w_2.weight".to_string(),
    filled(QH, QF, |r, c| 0.001 * ((r + c) as f32)),
  );
  w.insert(
    "encoder.encoders0.0.norm1.weight".to_string(),
    filled1(QD, |_| 1.0),
  );
  w.insert(
    "encoder.encoders0.0.norm1.bias".to_string(),
    filled1(QD, |_| 0.0),
  );
  w.insert(
    "encoder.encoders0.0.norm2.weight".to_string(),
    filled1(QH, |_| 1.0),
  );
  w.insert(
    "encoder.encoders0.0.norm2.bias".to_string(),
    filled1(QH, |_| 0.0),
  );
  w.insert(
    "encoder.after_norm.weight".to_string(),
    filled1(QH, |_| 1.0),
  );
  w.insert("encoder.after_norm.bias".to_string(), filled1(QH, |_| 0.0));
  w.insert("encoder.tp_norm.weight".to_string(), filled1(QH, |_| 1.0));
  w.insert("encoder.tp_norm.bias".to_string(), filled1(QH, |_| 0.0));
  w.insert(
    "ctc_lo.weight".to_string(),
    filled(QV, QH, |r, c| 0.001 * ((r + c) as f32)),
  );
  w.insert("ctc_lo.bias".to_string(), filled1(QV, |_| 0.0));
  w.insert("embed.weight".to_string(), filled(16, QD, |r, _| r as f32));
  w
}

/// Replace the dense `<prefix>.weight` in `w` with the real
/// `ops::quantized::quantize` 8-bit affine triple (packed `<prefix>.weight` +
/// `<prefix>.scales` + `<prefix>.biases`), mirroring an mlx-community quantized
/// checkpoint. `group_size` must divide the weight's input (last) axis and be
/// one of mlx's supported sizes (32 / 64 / 128).
fn quantize_in_place(w: &mut HashMap<String, Array>, prefix: &str, group_size: i32) {
  let dense = w.remove(&format!("{prefix}.weight")).unwrap();
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, group_size, 8, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

#[test]
fn quantized_ctc_lo_loads_and_forwards() {
  // Quantize ctc_lo (input QH) + embed (input QD) at group_size = QGROUP (32),
  // and assert the head builds as Quantized and the forward runs end-to-end.
  let config = quant_config();
  let mut w = quant_weights();
  let encoder =
    Encoder::from_weights(&mut w, config.input_size(), config.encoder_conf(), None).unwrap();
  quantize_in_place(&mut w, "ctc_lo", QGROUP);
  quantize_in_place(&mut w, "embed", QGROUP);

  let (ctc_lo, embed) = build_head(
    &mut w,
    config.input_size(),
    encoder.output_size(),
    config.vocab_size(),
    Some(&global_quant(QGROUP, 8)),
  )
  .unwrap();
  assert!(ctc_lo.is_quantized(), "ctc_lo built quantized");
  assert!(embed.is_quantized(), "embed built quantized");

  let model = SenseVoiceModel::new(
    config,
    encoder,
    ctc_lo,
    embed,
    SenseVoiceTokenizer::id_join(),
    None,
    None,
  );

  // The quantized forward runs end-to-end and produces speech-only logits.
  let wav = ramp_waveform(4000);
  let logits = CtcModel::logits(&model, &wav).unwrap();
  assert_eq!(logits.shape().len(), 2);
  assert_eq!(logits.shape()[1], QV as usize);
}

#[test]
fn build_head_resolves_quant_per_prefix_parameter_override() {
  // Per-prefix resolution at the head: `ctc_lo` quantizes via the global default
  // (group_size = 32) while `embed` is OVERRIDDEN to group_size = 64. A single
  // collapsed global tuple (group_size = 32) would mis-decode the group_size = 64
  // packed `embed` (its scales width is `QD/64 = 1`, not `QD/32 = 2`) and fail
  // the triple validator — so the build succeeding proves each head resolved its
  // OWN scheme via `quantization_for`.
  const EMBED_GROUP: i32 = 64; // QD (64) is divisible by 64
  let config = quant_config();
  let mut w = quant_weights();
  let encoder =
    Encoder::from_weights(&mut w, config.input_size(), config.encoder_conf(), None).unwrap();
  quantize_in_place(&mut w, "ctc_lo", QGROUP); // group_size = 32 (global default)
  quantize_in_place(&mut w, "embed", EMBED_GROUP); // group_size = 64 (override)

  let mut per_layer = HashMap::new();
  per_layer.insert(
    "embed".to_string(),
    crate::lm::quant::QuantizationOption::Quantize(Quantization::affine(EMBED_GROUP, 8)),
  );
  let quant = PerLayerQuantization::new(Some(Quantization::affine(QGROUP, 8)), per_layer);

  let (ctc_lo, embed) = build_head(
    &mut w,
    config.input_size(),
    encoder.output_size(),
    config.vocab_size(),
    Some(&quant),
  )
  .expect("each head must resolve its own per-prefix scheme (ctc_lo=32, embed=64)");
  assert!(
    ctc_lo.is_quantized(),
    "ctc_lo built quantized (global default)"
  );
  assert!(
    embed.is_quantized(),
    "embed built quantized (per-layer override)"
  );

  // Cross-check: resolving the GLOBAL tuple (group_size = 32) for the
  // group_size = 64 packed `embed` is a load-time error — the two schemes are
  // genuinely distinguishable, so the override is load-bearing.
  let mut w_wrong = quant_weights();
  let _enc = Encoder::from_weights(
    &mut w_wrong,
    config.input_size(),
    config.encoder_conf(),
    None,
  )
  .unwrap();
  quantize_in_place(&mut w_wrong, "ctc_lo", QGROUP);
  quantize_in_place(&mut w_wrong, "embed", EMBED_GROUP);
  assert!(
    build_head(
      &mut w_wrong,
      config.input_size(),
      encoder.output_size(),
      config.vocab_size(),
      Some(&global_quant(QGROUP, 8)), // group_size = 32 for BOTH — wrong for embed
    )
    .is_err(),
    "the global group_size must mis-decode the group_size=64 packed embed"
  );
}

#[test]
fn build_head_resolves_quant_per_layer_only_no_global_default() {
  // A per-layer-only config (no GLOBAL default) loads its explicitly-listed
  // layers: `ctc_lo` + `embed` each have an explicit override, `quantization`
  // (the global default) is `None`. The OLD code collapsed `quantization` to a
  // single global tuple and would have rejected this (no global → `None` →
  // `.scales` present → InvariantViolation). Per-prefix resolution loads it.
  let config = quant_config();
  let mut w = quant_weights();
  let encoder =
    Encoder::from_weights(&mut w, config.input_size(), config.encoder_conf(), None).unwrap();
  quantize_in_place(&mut w, "ctc_lo", QGROUP);
  quantize_in_place(&mut w, "embed", QGROUP);

  let mut per_layer = HashMap::new();
  per_layer.insert(
    "ctc_lo".to_string(),
    crate::lm::quant::QuantizationOption::Quantize(Quantization::affine(QGROUP, 8)),
  );
  per_layer.insert(
    "embed".to_string(),
    crate::lm::quant::QuantizationOption::Quantize(Quantization::affine(QGROUP, 8)),
  );
  // No global default — only the explicitly-listed layers are quantized.
  let quant = PerLayerQuantization::new(None, per_layer);

  let (ctc_lo, embed) = build_head(
    &mut w,
    config.input_size(),
    encoder.output_size(),
    config.vocab_size(),
    Some(&quant),
  )
  .expect("a per-layer-only config (no global default) must load its listed layers");
  assert!(
    ctc_lo.is_quantized(),
    "ctc_lo built quantized (explicit override)"
  );
  assert!(
    embed.is_quantized(),
    "embed built quantized (explicit override)"
  );
}

#[test]
fn quantized_head_scales_present_but_no_quant_errors() {
  // A `.scales` sibling with `quant == None` is a checkpoint/config
  // inconsistency -> typed InvariantViolation (the MaybeQuantized* contract).
  let config = quant_config();
  let mut w = quant_weights();
  let _encoder =
    Encoder::from_weights(&mut w, config.input_size(), config.encoder_conf(), None).unwrap();
  quantize_in_place(&mut w, "ctc_lo", QGROUP);
  // Pass quant = None despite the present `.scales` sibling (the
  // InvariantViolation fires inside the per-layer builder, before the shape-pin).
  assert!(matches!(
    build_head(&mut w, config.input_size(), QH, config.vocab_size(), None),
    Err(Error::InvariantViolation(_))
  ));
}

// ───────────────────────────── head shape-pin at load ─────────────────────────────

#[test]
fn build_head_rejects_wrong_embed_row_count() {
  // The `embed` table must be the fixed 16-row query+token table
  // (`sensevoice.py:348`). A 15-row table (wrong shard) is a typed
  // ShapePairMismatch at load, not a deferred out-of-bounds query gather.
  let config = tiny_config();
  let mut w = tiny_weights();
  // Replace the 16-row embed with a 15-row one (ctc_lo stays correct so the
  // embed check is the one that fires).
  w.insert("embed.weight".to_string(), filled(15, D, |r, _| r as f32));
  assert!(matches!(
    build_head(
      &mut w,
      config.input_size(),
      config.encoder_conf().output_size(),
      config.vocab_size(),
      None,
    ),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn build_head_rejects_wrong_embed_width() {
  // A 16-row embed but with the wrong feature width (input_size + 1) is also a
  // ShapePairMismatch — the gathered query rows would be the wrong width.
  let config = tiny_config();
  let mut w = tiny_weights();
  w.insert(
    "embed.weight".to_string(),
    filled(16, D + 1, |r, _| r as f32),
  );
  assert!(matches!(
    build_head(
      &mut w,
      config.input_size(),
      config.encoder_conf().output_size(),
      config.vocab_size(),
      None,
    ),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn build_head_rejects_wrong_ctc_lo_projection() {
  // `ctc_lo` must project `output_size -> vocab_size` (mlx weight
  // `(vocab_size, output_size)`, `sensevoice.py:347`). A weight whose input
  // width is not `output_size` is a typed ShapePairMismatch at load.
  let config = tiny_config();
  let mut w = tiny_weights();
  // ctc_lo with input width H + 1 (wrong) instead of H.
  w.insert(
    "ctc_lo.weight".to_string(),
    filled(V, H + 1, |r, c| 0.03 * ((r + c) as f32)),
  );
  assert!(matches!(
    build_head(
      &mut w,
      config.input_size(),
      config.encoder_conf().output_size(),
      config.vocab_size(),
      None,
    ),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn build_head_rejects_wrong_ctc_lo_vocab() {
  // A `ctc_lo` whose output (row) count is not `vocab_size` is likewise a typed
  // ShapePairMismatch (it would emit logits over an unusable vocab).
  let config = tiny_config();
  let mut w = tiny_weights();
  w.insert(
    "ctc_lo.weight".to_string(),
    filled(V + 1, H, |r, c| 0.03 * ((r + c) as f32)),
  );
  w.insert("ctc_lo.bias".to_string(), filled1(V + 1, |_| 0.0));
  assert!(matches!(
    build_head(
      &mut w,
      config.input_size(),
      config.encoder_conf().output_size(),
      config.vocab_size(),
      None,
    ),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn build_head_rejects_wrong_ctc_lo_bias_length() {
  // The dense arm of `MaybeQuantizedLinear` does not validate the optional
  // `ctc_lo.bias`; `build_head` pins it to `(vocab_size,)`. A wrong-length bias
  // (correct weight) would broadcast a single wrong offset across every logit —
  // a typed LengthMismatch at load.
  let config = tiny_config();
  let mut w = tiny_weights();
  // ctc_lo weight stays the correct (V, H); shrink ONLY the bias to (V - 1,).
  w.insert("ctc_lo.bias".to_string(), filled1(V - 1, |_| 0.0));
  assert!(matches!(
    build_head(
      &mut w,
      config.input_size(),
      config.encoder_conf().output_size(),
      config.vocab_size(),
      None,
    ),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn build_head_rejects_scalar_ctc_lo_bias() {
  // A stray `(1,)` `ctc_lo` bias (correct weight) — the SILENT-wrong-output case
  // a single broadcast offset across all `vocab_size` logits — is pinned to a
  // typed LengthMismatch at load (expected `(vocab_size,)`).
  let config = tiny_config();
  let mut w = tiny_weights();
  w.insert("ctc_lo.bias".to_string(), filled1(1, |_| 0.0));
  assert!(matches!(
    build_head(
      &mut w,
      config.input_size(),
      config.encoder_conf().output_size(),
      config.vocab_size(),
      None,
    ),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn build_head_rejects_rank2_ctc_lo_bias() {
  // A `(1, vocab)` `ctc_lo` bias (rank-2, correct weight) is a typed
  // RankMismatch — the mlx.nn.Linear bias must be the rank-1 `(vocab_size,)`.
  let config = tiny_config();
  let mut w = tiny_weights();
  w.insert(
    "ctc_lo.bias".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; V as usize], &[1, V]).unwrap(),
  );
  assert!(matches!(
    build_head(
      &mut w,
      config.input_size(),
      config.encoder_conf().output_size(),
      config.vocab_size(),
      None,
    ),
    Err(Error::RankMismatch(_))
  ));
}

#[test]
fn build_head_accepts_correct_shapes() {
  // The control: the correct `(16, input_size)` embed + `(vocab, output_size)`
  // ctc_lo build cleanly (the pins do not reject the real shapes), including the
  // correct `(vocab,)` ctc_lo dense bias that `tiny_weights` writes.
  let config = tiny_config();
  let mut w = tiny_weights();
  let (ctc_lo, embed) = build_head(
    &mut w,
    config.input_size(),
    config.encoder_conf().output_size(),
    config.vocab_size(),
    None,
  )
  .expect("correct head shapes build");
  assert_eq!(embed.logical_shape().unwrap(), (16, D));
  assert_eq!(ctc_lo.logical_shape().unwrap(), (V, H));
}

// ───────────────────────────── CMVN integration ─────────────────────────────

#[test]
fn extract_features_applies_cmvn_when_present() {
  // With CMVN stats the LFR features are shifted+scaled `(feats + means) * istd`;
  // verify the model path applies them (vs the no-CMVN model) on the same audio.
  let config = tiny_config();
  let mut w = tiny_weights();
  let encoder =
    Encoder::from_weights(&mut w, config.input_size(), config.encoder_conf(), None).unwrap();
  let (ctc_lo, embed) = build_head(
    &mut w,
    config.input_size(),
    encoder.output_size(),
    config.vocab_size(),
    None,
  )
  .unwrap();

  // CMVN means = 1.0, istd = 2.0 across the D LFR dims.
  let means = filled1(D, |_| 1.0);
  let istd = filled1(D, |_| 2.0);
  let model = SenseVoiceModel::new(
    config,
    encoder,
    ctc_lo,
    embed,
    SenseVoiceTokenizer::id_join(),
    Some(means),
    Some(istd),
  );

  let wav = ramp_waveform(3000);
  let mut feats = model.extract_features(&wav).unwrap();
  // Shape is (T', D); the CMVN op preserves shape.
  assert_eq!(feats.shape()[1], D as usize);

  // Compare to the same model without CMVN: the values must differ (the shift +
  // scale changed them).
  let model_no_cmvn = tiny_model(SenseVoiceTokenizer::id_join());
  let mut feats_plain = model_no_cmvn.extract_features(&wav).unwrap();
  let a = feats.to_vec::<f32>().unwrap();
  let b = feats_plain.to_vec::<f32>().unwrap();
  assert_eq!(a.len(), b.len());
  // `(x + 1) * 2` != `x` for the non-degenerate features.
  assert!(
    a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6),
    "CMVN must change the features"
  );
}

// ──────────── frame_argmax batching A/B (rich-info optimization) ────────────

/// The per-frame argmax form of the rich-info heads, kept verbatim as the A/B
/// reference: slice the single row `[frame, frame+1)` -> reshape `(vocab,)` ->
/// `argmax` (no axis) -> `item::<u32>()` — one GPU→CPU sync per frame, three
/// calls for the three query rows (`sensevoice.py:468/479/493`). Shares no code
/// with the batched `query_argmax_ids` (one slice + one `argmax(axis=1)` + one
/// host copy); the A/B asserts the two give identical ids.
fn frame_argmax_reference(log_probs: &Array, frame: i32) -> u32 {
  let vocab = log_probs.shape()[1] as i32;
  let row = ops::indexing::slice(log_probs, &[frame, 0], &[frame + 1, vocab], &[1, 1]).unwrap();
  let row = ops::shape::reshape(&row, &[vocab]).unwrap();
  let mut arg = ops::misc::argmax(&row, None, false).unwrap();
  arg.item::<u32>().unwrap()
}

/// CORRECTNESS A/B for the batched rich-info argmax: `query_argmax_ids` must
/// return EXACTLY the three ids the per-frame reference computes, for real label
/// ids, unknown ids, and — crucially — a TIE row (two equal maxima), since
/// argmax tie-breaking (first index, U32) must be identical batched vs per-row.
#[test]
fn query_argmax_batched_matches_per_frame_reference() {
  // Each row: a distinct argmax. Row 4+ are present but ignored by the heads.
  // Cases mix real ids, an unknown id, and a deliberate tie in a query row.
  let cases: &[Vec<u32>] = &[
    vec![24884, 25001, 24993, 0],     // zh / happy / Speech.
    vec![24885, 25004, 24995, 17],    // en / neutral / BGM + a text row.
    vec![24888, 25009, 24997, 24999], // yue / unk / Laughter.
    vec![999, 12345, 0, 5],           // all-unknown query ids.
  ];
  for peaks in cases {
    let grid = rich_grid(peaks);
    let batched = SenseVoiceModel::query_argmax_ids(&grid).unwrap();
    let want = [
      frame_argmax_reference(&grid, 0),
      frame_argmax_reference(&grid, 1),
      frame_argmax_reference(&grid, 2),
    ];
    assert_eq!(batched, want, "batched ids differ for peaks {peaks:?}");
  }
}

/// A grid with an explicit TIE in query row 1 (two columns share the max value):
/// both the batched and per-frame argmax must pick the SAME (lowest) index, so
/// the resulting label is identical. Pins tie-breaking parity directly.
#[test]
fn query_argmax_tie_breaks_identically() {
  let vocab: i32 = 25_055;
  let frames = 4i32;
  let mut data = vec![0.0f32; (frames * vocab) as usize];
  // Row 0: clean peak at 24884.
  data[24884] = 10.0;
  // Row 1: TIE — equal maxima at 25001 and 25006 (argmax must pick 25001).
  data[vocab as usize + 25001] = 7.5;
  data[vocab as usize + 25006] = 7.5;
  // Row 2: clean peak at 24993.
  data[2 * vocab as usize + 24993] = 10.0;
  let grid = Array::from_slice::<f32>(&data, &[frames, vocab]).unwrap();

  let batched = SenseVoiceModel::query_argmax_ids(&grid).unwrap();
  let want = [
    frame_argmax_reference(&grid, 0),
    frame_argmax_reference(&grid, 1),
    frame_argmax_reference(&grid, 2),
  ];
  assert_eq!(batched, want);
  assert_eq!(batched[1], 25001, "tie must resolve to the lowest index");
}

/// End-to-end `rich_info` parity: the labels the (now batched) `rich_info`
/// produces must equal the labels assembled from the three per-frame reference
/// argmaxes — confirming the batching change preserves the full rich-info
/// output, not just raw ids.
#[test]
fn rich_info_labels_unchanged_by_batching() {
  let model = tiny_model(SenseVoiceTokenizer::id_join());
  let grid = rich_grid(&[24885, 25004, 24993, 0]);
  let rich = model.rich_info(&grid).unwrap();
  // Reference labels via the per-frame argmax + the same label maps.
  let want_lang = lid_label(frame_argmax_reference(&grid, 0));
  let want_emo = emotion_label(frame_argmax_reference(&grid, 1));
  let want_event = event_label(frame_argmax_reference(&grid, 2));
  assert_eq!(rich.language(), want_lang);
  assert_eq!(rich.emotion(), want_emo);
  assert_eq!(rich.event(), want_event);
}

/// PERF A/B for the batched rich-info argmax: `query_argmax_ids` (one device
/// sync) vs three per-frame `frame_argmax_reference` calls (three syncs) at the
/// real vocab width, over many calls. Reports best-of-N min for both; batched
/// must be no slower (fewer GPU→CPU round-trips).
///
/// `#[ignore]`d: timing is machine/thermal-dependent. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore = "timing micro-bench — run with --ignored --nocapture"]
fn bench_rich_argmax_batched_vs_per_frame() {
  use std::time::Instant;
  // The real rich-info grid is the encoder output's first rows at the full
  // CTC vocab. Build a representative (8, 25055) grid.
  let grid = rich_grid(&[24884, 25001, 24993, 0, 1, 2, 3, 4]);
  crate::transforms::eval(&[&grid]).unwrap();

  let per_frame = |g: &Array| {
    [
      frame_argmax_reference(g, 0),
      frame_argmax_reference(g, 1),
      frame_argmax_reference(g, 2),
    ]
  };
  for _ in 0..5 {
    let _ = per_frame(&grid);
    let _ = SenseVoiceModel::query_argmax_ids(&grid).unwrap();
  }
  let bench = |label: &str, f: &dyn Fn() -> [u32; 3]| {
    let mut times = Vec::with_capacity(50);
    for _ in 0..50 {
      let t0 = Instant::now();
      let ids = f();
      std::hint::black_box(ids);
      times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
      "  {label:<16} min={:.4}ms median={:.4}ms",
      times[0],
      times[times.len() / 2]
    );
    times[0]
  };
  println!("\nrich-info argmax (3 query rows, vocab=25055):");
  let per_min = bench("per-frame (3x)", &|| per_frame(&grid));
  let batched_min = bench("batched (1x)", &|| {
    SenseVoiceModel::query_argmax_ids(&grid).unwrap()
  });
  println!(
    "  speedup (per-frame/batched) = {:.2}x",
    per_min / batched_min
  );
}
