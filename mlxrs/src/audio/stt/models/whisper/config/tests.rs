use super::*;
use crate::error::Error;

/// The native MLX `config.json` for whisper-tiny (multilingual): the ten
/// `ModelDimensions` fields verbatim.
fn mlx_tiny() -> Value {
  serde_json::json!({
    "model_type": "whisper",
    "n_mels": 80,
    "n_audio_ctx": 1500,
    "n_audio_state": 384,
    "n_audio_head": 6,
    "n_audio_layer": 4,
    "n_vocab": 51865,
    "n_text_ctx": 448,
    "n_text_state": 384,
    "n_text_head": 6,
    "n_text_layer": 4
  })
}

/// A HuggingFace `config.json` for whisper-tiny: the HF field names. `d_model`
/// backs both the audio and text state width.
fn hf_tiny() -> Value {
  serde_json::json!({
    "model_type": "whisper",
    "num_mel_bins": 80,
    "max_source_positions": 1500,
    "d_model": 384,
    "encoder_attention_heads": 6,
    "encoder_layers": 4,
    "vocab_size": 51865,
    "max_target_positions": 448,
    "decoder_attention_heads": 6,
    "decoder_layers": 4
  })
}

#[test]
fn from_dict_mlx_format() {
  let dims = ModelDimensions::from_dict(&mlx_tiny()).unwrap();
  assert_eq!(dims.n_mels(), 80);
  assert_eq!(dims.n_audio_ctx(), 1500);
  assert_eq!(dims.n_audio_state(), 384);
  assert_eq!(dims.n_audio_head(), 6);
  assert_eq!(dims.n_audio_layer(), 4);
  assert_eq!(dims.n_vocab(), 51865);
  assert_eq!(dims.n_text_ctx(), 448);
  assert_eq!(dims.n_text_state(), 384);
  assert_eq!(dims.n_text_head(), 6);
  assert_eq!(dims.n_text_layer(), 4);
}

#[test]
fn from_dict_hf_format_maps_to_same_dims() {
  // The HF layout must resolve to exactly the same ModelDimensions as the
  // native MLX layout for the same model.
  let mlx = ModelDimensions::from_dict(&mlx_tiny()).unwrap();
  let hf = ModelDimensions::from_dict(&hf_tiny()).unwrap();
  assert_eq!(mlx, hf);
}

#[test]
fn from_dict_hf_d_model_backs_both_states() {
  // `d_model` is the only state-width source in the HF config; it must back
  // BOTH n_audio_state and n_text_state.
  let hf = ModelDimensions::from_dict(&hf_tiny()).unwrap();
  assert_eq!(hf.n_audio_state(), 384);
  assert_eq!(hf.n_text_state(), 384);
}

#[test]
fn from_dict_hf_uses_defaults_for_absent_keys() {
  // A bare HF config (only `encoder_layers` present to select the HF branch)
  // falls back to the reference defaults for every other field — large-v3
  // shape (d_model 1280, 128 mels, vocab 51866, 32 layers, 20 heads).
  let cfg = serde_json::json!({ "encoder_layers": 32 });
  let dims = ModelDimensions::from_dict(&cfg).unwrap();
  assert_eq!(dims.n_audio_layer(), 32);
  assert_eq!(dims.n_audio_state(), 1280);
  assert_eq!(dims.n_text_state(), 1280);
  assert_eq!(dims.n_mels(), 128);
  assert_eq!(dims.n_vocab(), 51866);
  assert_eq!(dims.n_text_layer(), 32);
  assert_eq!(dims.n_audio_head(), 20);
  assert_eq!(dims.n_text_head(), 20);
  assert_eq!(dims.n_audio_ctx(), 1500);
  assert_eq!(dims.n_text_ctx(), 448);
}

#[test]
fn from_dict_mlx_missing_field_errors() {
  // Drop a required MLX field — the reference's `cls(**filtered)` TypeError,
  // here a typed MissingField.
  let mut obj = mlx_tiny();
  obj.as_object_mut().unwrap().remove("n_text_layer");
  let err = ModelDimensions::from_dict(&obj).unwrap_err();
  assert!(
    matches!(err, Error::MissingField(p) if p.field() == "n_text_layer"),
    "expected MissingField(n_text_layer), got {err:?}"
  );
}

#[test]
fn from_dict_rejects_non_object_root() {
  let cfg = serde_json::json!([1, 2, 3]);
  assert!(matches!(
    ModelDimensions::from_dict(&cfg),
    Err(Error::Parse(_))
  ));
}

#[test]
fn from_dict_rejects_non_integer_field() {
  let mut obj = mlx_tiny();
  obj.as_object_mut().unwrap()["n_mels"] = serde_json::json!("eighty");
  assert!(matches!(
    ModelDimensions::from_dict(&obj),
    Err(Error::Parse(_))
  ));
}

#[test]
fn from_dict_rejects_negative_field() {
  let mut obj = mlx_tiny();
  obj.as_object_mut().unwrap()["n_mels"] = serde_json::json!(-80);
  assert!(matches!(
    ModelDimensions::from_dict(&obj),
    Err(Error::Parse(_))
  ));
}

#[test]
fn is_multilingual_threshold() {
  // 51865 is multilingual; 51864 (the *.en models) is not.
  let mut obj = mlx_tiny();
  obj.as_object_mut().unwrap()["n_vocab"] = serde_json::json!(51865);
  assert!(ModelDimensions::from_dict(&obj).unwrap().is_multilingual());

  obj.as_object_mut().unwrap()["n_vocab"] = serde_json::json!(51864);
  assert!(!ModelDimensions::from_dict(&obj).unwrap().is_multilingual());
}

#[test]
fn num_languages_matches_reference_formula() {
  // num_languages = n_vocab - 51765 - is_multilingual.
  // tiny multilingual: 51865 - 51765 - 1 = 99.
  let dims = ModelDimensions::from_dict(&mlx_tiny()).unwrap();
  assert_eq!(dims.num_languages(), 99);
}

#[test]
fn validate_rejects_zero_field() {
  // A zero dimension is a non-positive cardinality → OutOfRange naming the
  // field (the shared `require_cardinality` puts the field name in `context`).
  let err = ModelDimensions::new(80, 1500, 384, 6, 4, 51865, 448, 384, 6, 0).unwrap_err();
  assert!(
    matches!(&err, Error::OutOfRange(p) if p.context() == "n_text_layer"),
    "expected OutOfRange on zero n_text_layer, got {err:?}"
  );
}

#[test]
fn validate_rejects_oversized_field() {
  // A dimension above MAX_DIM is the bounded-memory cap guard → CapExceeded
  // naming the field, the cap, and the observed value.
  let huge = (1usize << 22) + 1;
  let err = ModelDimensions::new(80, 1500, 384, 6, 4, huge, 448, 384, 6, 4).unwrap_err();
  assert!(
    matches!(&err, Error::CapExceeded(p) if p.cap_name() == "n_vocab" && p.observed() == huge as u64),
    "expected CapExceeded on oversized n_vocab, got {err:?}"
  );
}

#[test]
fn validate_rejects_non_divisible_audio_state() {
  // n_audio_state 384 not divisible by n_audio_head 5.
  let err = ModelDimensions::new(80, 1500, 384, 5, 4, 51865, 448, 384, 6, 4).unwrap_err();
  assert!(
    matches!(&err, Error::DivisibilityConstraint(p) if p.name_dividend() == "n_audio_state"),
    "expected DivisibilityConstraint on n_audio_state, got {err:?}"
  );
}

#[test]
fn validate_rejects_non_divisible_text_state() {
  let err = ModelDimensions::new(80, 1500, 384, 6, 4, 51865, 448, 384, 5, 4).unwrap_err();
  assert!(
    matches!(&err, Error::DivisibilityConstraint(p) if p.name_dividend() == "n_text_state"),
    "expected DivisibilityConstraint on n_text_state, got {err:?}"
  );
}

#[test]
fn validate_rejects_odd_audio_state_for_sinusoid() {
  // The encoder positional embedding is `sinusoids(n_audio_ctx, n_audio_state)`,
  // which requires `n_audio_state` EVEN (the `concat([sin, cos])` halves the
  // width). An odd `n_audio_state` passes the per-field / divisibility guards
  // (here n_audio_head 3 divides n_audio_state 3) but would fail only after
  // `AudioEncoder::new` consumes weights; `validate` now rejects it up front as
  // `OutOfRange` naming `n_audio_state`. n_text_state matches (3) so the
  // equal-width pin is not what trips.
  let err = ModelDimensions::new(80, 1500, 3, 3, 4, 51865, 448, 3, 3, 4).unwrap_err();
  assert!(
    matches!(&err, Error::OutOfRange(p) if p.context() == "n_audio_state"),
    "expected OutOfRange on odd n_audio_state, got {err:?}"
  );
}

#[test]
fn validate_rejects_width_one_audio_state_for_sinusoid() {
  // `n_audio_state = 1` is odd → rejected by the even-ness half of the sinusoid
  // precondition (n_audio_head 1 divides it, so divisibility passes first).
  let err = ModelDimensions::new(80, 1500, 1, 1, 4, 51865, 448, 1, 1, 4).unwrap_err();
  assert!(
    matches!(&err, Error::OutOfRange(p) if p.context() == "n_audio_state"),
    "expected OutOfRange on width-1 n_audio_state, got {err:?}"
  );
}

#[test]
fn validate_rejects_width_two_audio_state_for_sinusoid() {
  // `n_audio_state = 2` is EVEN but degenerate for the sinusoid: `half =
  // n_audio_state/2 = 1`, so `inv_timescales`'s divisor `half - 1 == 0` produces
  // a `+inf` increment and a `0 * inf` NaN positional row. The non-degenerate
  // lower bound (`n_audio_state >= 4`, i.e. `half >= 2`) rejects it as
  // `OutOfRange` naming `n_audio_state` — passing the divisibility (n_audio_head
  // 2 divides 2) and even-ness checks but failing the range check. n_text_state
  // matches (2) so the equal-width pin is not what trips.
  let err = ModelDimensions::new(80, 1500, 2, 2, 4, 51865, 448, 2, 2, 4).unwrap_err();
  assert!(
    matches!(&err, Error::OutOfRange(p) if p.context() == "n_audio_state"),
    "expected OutOfRange on degenerate width-2 n_audio_state, got {err:?}"
  );
}

#[test]
fn validate_accepts_smallest_non_degenerate_audio_state() {
  // The boundary `n_audio_state = 4` (even, `half = 2 >= 2`) is the smallest
  // width the sinusoid construction handles, and is accepted (with a matching
  // n_text_state and a divisible head count). This pins the bound is not
  // off-by-one against the real `sinusoids` precondition.
  assert!(ModelDimensions::new(80, 1500, 4, 2, 4, 51865, 448, 4, 2, 4).is_ok());
}

#[test]
fn validate_rejects_unequal_audio_text_state_widths() {
  // The encoder and decoder hidden widths must be EQUAL — the decoder's
  // cross-attention consumes the encoder states `(1, n_audio_ctx, n_audio_state)`
  // through square `n_text_state` projections, and the crate carries no
  // unequal-width bridge. Here n_audio_state=384 (head 6) and n_text_state=512
  // (head 8) are each individually valid (divisible, small, every product cap
  // cleared), but unequal — rejected with a typed `OutOfRange` naming the field.
  let err = ModelDimensions::new(80, 1500, 384, 6, 4, 51865, 448, 512, 8, 4).unwrap_err();
  assert!(
    matches!(&err, Error::OutOfRange(p) if p.context() == "n_text_state"),
    "expected OutOfRange on n_text_state (unequal state widths), got {err:?}"
  );
  // The equal-width sibling (both 384) builds.
  assert!(ModelDimensions::new(80, 1500, 384, 6, 4, 51865, 448, 384, 6, 4).is_ok());
}

#[test]
fn validate_rejects_oversized_audio_pos_emb_product() {
  // `n_audio_ctx` is pinned to the fixed 1500, so the encoder positional
  // embedding product is driven over DENSE_2D_ELEM_CAP (1 << 26 = 67_108_864) by
  // a large `n_audio_state` instead: 1500 * 50_000 = 75_000_000. Each field is
  // individually <= MAX_DIM (1 << 22 = 4_194_304); the shared `elem_count`
  // rejects the product as CapExceeded naming the extent. The positional
  // embedding is the FIRST 2-D product checked, so it trips before the conv1
  // activation (N_FRAMES * n_audio_state) reaches the same cap.
  let n_audio_state = 50_000usize; // divisible by n_audio_head 8
  assert!(n_audio_state < (1 << 22));
  let err = ModelDimensions::new(80, 1500, n_audio_state, 8, 4, 51865, 448, 384, 6, 4).unwrap_err();
  let expected = (1500 * n_audio_state) as u64;
  assert!(
    matches!(&err, Error::CapExceeded(p)
      if p.cap_name() == "n_audio_ctx * n_audio_state (encoder positional embedding)"
        && p.observed() == expected),
    "expected CapExceeded on the audio positional-embedding product, got {err:?}"
  );
}

#[test]
fn validate_rejects_oversized_causal_mask_product() {
  // n_text_ctx 10_000 is individually valid (< MAX_DIM) but the causal mask is
  // n_text_ctx^2 = 100_000_000 > DENSE_2D_ELEM_CAP. The audio + text
  // positional-embedding products here stay within cap (1500*384 and
  // 10_000*384), so the mask product is the one that trips.
  let n_text_ctx = 10_000usize;
  assert!(n_text_ctx < (1 << 22));
  // n_text_state 384 keeps n_text_ctx * n_text_state = 3_840_000 under cap, so
  // the quadratic mask (n_text_ctx^2) is the first product to exceed it.
  let err = ModelDimensions::new(80, 1500, 384, 6, 4, 51865, n_text_ctx, 384, 6, 4).unwrap_err();
  let expected = (n_text_ctx * n_text_ctx) as u64;
  assert!(
    matches!(&err, Error::CapExceeded(p)
      if p.cap_name() == "n_text_ctx * n_text_ctx (decoder causal mask)"
        && p.observed() == expected),
    "expected CapExceeded on the causal-mask product, got {err:?}"
  );
}

#[test]
fn validate_rejects_oversized_mel_filter_product() {
  // n_mels 100_000 is individually valid (< MAX_DIM) but the mel filterbank is
  // n_mels * n_freqs (n_freqs = 400/2 + 1 = 201) = 20_100_000 >
  // MEL_FILTER_ELEM_CAP (1 << 20 = 1_048_576).
  let n_mels = 100_000usize;
  assert!(n_mels < (1 << 22));
  let err = ModelDimensions::new(n_mels, 1500, 384, 6, 4, 51865, 448, 384, 6, 4).unwrap_err();
  let expected = (n_mels * 201) as u64;
  assert!(
    matches!(&err, Error::CapExceeded(p)
      if p.cap_name() == "n_mels * n_freqs (mel filterbank)" && p.observed() == expected),
    "expected CapExceeded on the mel-filterbank product, got {err:?}"
  );
}

#[test]
fn validate_accepts_largest_real_checkpoint_products() {
  // large-v3 shape: EVERY derived extent — the 2-D products AND every 3-D
  // forward-path tensor (attention scores, MLP hidden, vocab projection, KV
  // cache) — clears its cap with room to spare, so the caps reject no released
  // checkpoint. The concrete large-v3 extents vs their caps:
  //   enc self-attn  20*1500*1500 =  45_000_000 < ATTN_SCORE_ELEM_CAP (1<<29 = 536_870_912)
  //   cross-attn     20* 448*1500 =  13_440_000 < ATTN_SCORE_ELEM_CAP
  //   enc MLP       1500*   4*1280 =   7_680_000 < MLP_HIDDEN_ELEM_CAP  (1<<27 = 134_217_728)
  //   vocab table  51866*    1280  =  66_388_480 < VOCAB_PROJ_ELEM_CAP  (1<<28 = 268_435_456)
  //   logits         448*   51866  =  23_235_968 < VOCAB_PROJ_ELEM_CAP
  //   self KV        32* 448*1280  =  18_350_080 < KV_CACHE_ELEM_CAP    (1<<29 = 536_870_912)
  //   cross KV       32*1500*1280  =  61_440_000 < KV_CACHE_ELEM_CAP
  assert!(ModelDimensions::new(128, 1500, 1280, 20, 32, 51866, 448, 1280, 20, 32).is_ok());
}

/// Assert a [`ModelDimensions::new`] call fails with [`Error::CapExceeded`]
/// naming `cap` and carrying observed extent `observed`.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn assert_cap_exceeded(
  n_mels: usize,
  n_audio_ctx: usize,
  n_audio_state: usize,
  n_audio_head: usize,
  n_audio_layer: usize,
  n_vocab: usize,
  n_text_ctx: usize,
  n_text_state: usize,
  n_text_head: usize,
  n_text_layer: usize,
  cap: &str,
  observed: u64,
) {
  let err = ModelDimensions::new(
    n_mels,
    n_audio_ctx,
    n_audio_state,
    n_audio_head,
    n_audio_layer,
    n_vocab,
    n_text_ctx,
    n_text_state,
    n_text_head,
    n_text_layer,
  )
  .unwrap_err();
  assert!(
    matches!(&err, Error::CapExceeded(p) if p.cap_name() == cap && p.observed() == observed),
    "expected CapExceeded({cap}, observed={observed}), got {err:?}"
  );
}

#[test]
fn validate_rejects_oversized_encoder_self_attention_scores() {
  // `n_audio_ctx` is pinned to 1500, so the encoder self-attention score tensor
  // n_audio_head * n_audio_ctx^2 is driven over ATTN_SCORE_ELEM_CAP (1<<29 =
  // 536_870_912) by a large `n_audio_head`: 256 * 1500 * 1500 = 576_000_000.
  // n_audio_state = 256 (divisible by the 256 heads) keeps the 2-D products
  // (pos-emb 1500*256, conv N_FRAMES*256) under DENSE_2D_ELEM_CAP, so the score
  // tensor is the first extent to trip.
  assert_cap_exceeded(
    80,
    1500,
    256,
    256,
    4,
    51865,
    448,
    384,
    6,
    4,
    "n_audio_head * n_audio_ctx * n_audio_ctx (encoder self-attention scores)",
    256 * 1500 * 1500,
  );
}

#[test]
fn validate_rejects_oversized_cross_attention_scores() {
  // `n_audio_ctx` is pinned to 1500. The encoder and decoder self-attention
  // scores stay within cap, but the cross-attention score
  // n_text_head * n_text_ctx * n_audio_ctx = 400 * 1000 * 1500 = 600_000_000
  // exceeds ATTN_SCORE_ELEM_CAP (1<<29 = 536_870_912). n_audio_head/state = 4
  // (even and >= 4, clearing the sinusoid precondition) keep enc self-attn
  // (4*1500^2 = 9M) under cap, and n_text_ctx = 1000 keeps dec self-attn
  // (400*1000^2 = 400M) under cap; since 1500 > n_text_ctx, the cross product
  // (which swaps the second n_text_ctx for n_audio_ctx = 1500) exceeds the dec
  // self-attn and is the first to trip.
  assert_cap_exceeded(
    80,
    1500,
    4,
    4,
    4,
    51865,
    1000,
    400,
    400,
    4,
    "n_text_head * n_text_ctx * n_audio_ctx (cross-attention scores)",
    400 * 1000 * 1500,
  );
}

#[test]
fn validate_rejects_oversized_decoder_mlp_hidden() {
  // Attention scores within cap, but the decoder MLP hidden activation
  // n_text_ctx * 4 * n_text_state = 2048 * 4 * 20000 = 163_840_000 exceeds
  // MLP_HIDDEN_ELEM_CAP (1<<27 = 134_217_728). The decoder pos-emb
  // (2048*20000 = 40.96M < 1<<26) and causal mask (2048^2 < 1<<26) stay within
  // the 2-D cap; the encoder MLP (1500*4*384 = 2.3M) stays under its cap, so the
  // decoder MLP is the first to trip.
  assert_cap_exceeded(
    80,
    1500,
    384,
    6,
    4,
    51865,
    2048,
    20000,
    20,
    4,
    "n_text_ctx * 4 * n_text_state (decoder MLP hidden)",
    2048 * 4 * 20000,
  );
}

#[test]
fn validate_rejects_oversized_vocab_projection_table() {
  // Every attention / MLP extent within cap, but the token-embedding /
  // tied-logit table n_vocab * n_text_state = 300_000 * 1024 = 307_200_000
  // exceeds VOCAB_PROJ_ELEM_CAP (1<<28 = 268_435_456). n_vocab = 300_000 is
  // individually valid (< MAX_DIM); the decoder logits (448*300_000 = 134.4M)
  // stay under cap, so the table is the first vocab extent to trip.
  assert_cap_exceeded(
    80,
    1500,
    384,
    6,
    4,
    300_000,
    448,
    1024,
    8,
    4,
    "n_vocab * n_text_state (token-embedding / tied-logit table)",
    300_000 * 1024,
  );
}

#[test]
fn validate_rejects_oversized_self_attention_kv_cache() {
  // Every attention / MLP / vocab extent within cap, but the cumulative decoder
  // self-attention KV cache n_text_layer * n_text_ctx * n_text_state =
  // 64 * 448 * 20000 = 573_440_000 exceeds KV_CACHE_ELEM_CAP (1<<29 =
  // 536_870_912). n_vocab = 10_000 keeps the vocab table (10_000*20000 = 200M)
  // under cap, and the decoder MLP (448*4*20000 = 35.8M) stays under cap, so the
  // KV cache is the first cumulative extent to trip.
  assert_cap_exceeded(
    80,
    1500,
    384,
    6,
    4,
    10_000,
    448,
    20000,
    20,
    64,
    "n_text_layer * n_text_ctx * n_text_state (decoder self-attention KV cache)",
    64 * 448 * 20000,
  );
}

#[test]
fn validate_attention_score_beyond_i32_is_capexceeded() {
  // A config whose attention-score product exceeds i32 but fits in usize is
  // rejected as CapExceeded (not a wrap, not an overflow): the shared
  // `elem_count` accumulates the product in usize, so the only ceiling reached
  // is the element cap. `n_audio_ctx` is pinned to 1500, so the >i32 product is
  // driven by a large `n_audio_head` instead: n_audio_head = 2000 (< MAX_DIM =
  // 1<<22) → encoder self-attention 3-way product 2000 * 1500 * 1500 =
  // 4_500_000_000 > i32::MAX (2_147_483_647) and > ATTN_SCORE_ELEM_CAP (1<<29 =
  // 536_870_912), but far below usize::MAX. n_audio_state = 2000 (divisible by
  // the 2000 heads) keeps the 2-D products (pos-emb 1500*2000, conv
  // N_FRAMES*2000) under DENSE_2D_ELEM_CAP, so the 3-way score product is the
  // first extent the element cap rejects.
  let err = ModelDimensions::new(80, 1500, 2000, 2000, 4, 51865, 448, 384, 6, 4).unwrap_err();
  let expected = 2000u64 * 1500 * 1500;
  assert!(
    matches!(&err, Error::CapExceeded(p)
      if p.cap_name() == "n_audio_head * n_audio_ctx * n_audio_ctx (encoder self-attention scores)"
        && p.observed() == expected),
    "expected CapExceeded on the >i32 attention-score product, got {err:?}"
  );
}

#[test]
fn elem_count_overflow_is_arithmetic_overflow_for_whisper_axes() {
  // The shared `elem_count` still surfaces a genuine usize-overflowing product
  // as ArithmeticOverflow rather than wrapping. Whisper caps every axis at
  // MAX_DIM (1<<22), so a 3-way product of near-cap axes reaches ~2^66 and
  // overflows usize (2^64) — the path `elem_count` reports as
  // ArithmeticOverflow. (In `validate` this never fires before a CapExceeded,
  // because any axis large enough to overflow a 3-way product first trips a
  // 2-way element cap; this asserts the overflow contract on the same axis cap
  // regime directly.)
  use crate::model_validation::{Extent, elem_count};
  let max_axis = Extent::new("axis", 1 << 22, 1 << 22).unwrap();
  let err = elem_count(
    "three near-MAX_DIM axes (usize-overflowing product)",
    &[max_axis, max_axis, max_axis],
    usize::MAX,
  )
  .unwrap_err();
  assert!(
    matches!(&err, Error::ArithmeticOverflow(_)),
    "expected ArithmeticOverflow on the usize-overflowing product, got {err:?}"
  );
}

#[test]
fn from_dict_propagates_validation() {
  // A config with a zero field is rejected at from_dict (validate is eager).
  let mut obj = mlx_tiny();
  obj.as_object_mut().unwrap()["n_audio_head"] = serde_json::json!(0);
  assert!(matches!(
    ModelDimensions::from_dict(&obj),
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn build_pins_n_audio_ctx_to_fixed_value() {
  // `n_audio_ctx` is architecturally fixed at N_FRAMES / CONV_DOWNSAMPLE = 1500
  // (Whisper pads every segment to N_FRAMES = 3000 before the encoder, and
  // conv2's stride 2 halves that). A config with any other value — even one
  // individually valid and well under MAX_DIM — is an unsupported architecture
  // and is rejected at construction with OutOfRange naming the field, BEFORE the
  // conv1 activation cap (computed from the real N_FRAMES extent) could bound a
  // tensor the encoder never builds.
  for bad in [1usize, 1499, 1501, 3000, 750] {
    assert!(bad < (1 << 22));
    let err = ModelDimensions::new(80, bad, 384, 6, 4, 51865, 448, 384, 6, 4).unwrap_err();
    assert!(
      matches!(&err, Error::OutOfRange(p) if p.context() == "n_audio_ctx"),
      "expected OutOfRange pinning n_audio_ctx for value {bad}, got {err:?}"
    );
  }
  // The fixed value itself is accepted.
  assert!(ModelDimensions::new(80, 1500, 384, 6, 4, 51865, 448, 384, 6, 4).is_ok());
}

#[test]
fn build_pin_uses_n_frames_over_conv_downsample() {
  // The pinned `n_audio_ctx` equals the real downsampled frame count, so the
  // accepted value is exactly N_FRAMES / 2. This guards the pin constant against
  // drifting from the audio front-end's N_FRAMES.
  use super::super::audio::N_FRAMES;
  let ok = N_FRAMES / 2;
  assert!(ModelDimensions::new(80, ok, 384, 6, 4, 51865, 448, 384, 6, 4).is_ok());
  let err = ModelDimensions::new(80, ok + 1, 384, 6, 4, 51865, 448, 384, 6, 4).unwrap_err();
  assert!(matches!(&err, Error::OutOfRange(p) if p.context() == "n_audio_ctx"));
}

#[test]
fn validate_conv1_activation_cap_uses_n_frames_extent() {
  // The encoder conv1 runs on the FIXED N_FRAMES (3000) padded frames, so its
  // activation cap is N_FRAMES * n_audio_state — TWICE the encoder positional
  // embedding (n_audio_ctx * n_audio_state = 1500 * n_audio_state). A
  // `n_audio_state` is chosen so the positional embedding stays UNDER
  // DENSE_2D_ELEM_CAP (1<<26 = 67_108_864) while the conv activation exceeds it:
  //   pos-emb   1500 * 30_000 = 45_000_000  < cap
  //   conv1     3000 * 30_000 = 90_000_000  > cap   ← trips
  // This proves the cap bounds the real (doubled) runtime extent, not the
  // smaller config-derived positional-embedding product. n_audio_state 30_000 is
  // divisible by n_audio_head 10; the conv is the FIRST extent to exceed its cap
  // (the encoder MLP hidden, 1500*4*30_000 = 180M, comes later in `validate`).
  let n_audio_state = 30_000usize;
  assert!(n_audio_state < (1 << 22));
  assert!(1500 * n_audio_state < (1 << 26)); // pos-emb under cap
  assert!(N_FRAMES * n_audio_state > (1 << 26)); // conv activation over cap
  let err =
    ModelDimensions::new(80, 1500, n_audio_state, 10, 4, 51865, 448, 384, 6, 4).unwrap_err();
  let expected = (N_FRAMES * n_audio_state) as u64;
  assert!(
    matches!(&err, Error::CapExceeded(p)
      if p.cap_name() == "N_FRAMES * n_audio_state (encoder conv1 pre-downsample activation)"
        && p.observed() == expected),
    "expected CapExceeded on the conv1 N_FRAMES activation, got {err:?}"
  );
}

#[test]
fn build_rejects_over_cap_audio_layer_count() {
  // The encoder layer count sizes the eager per-layer block `Vec`; it is bounded
  // by MAX_LAYERS (1<<12 = 4096) at construction so a millions-of-layers config
  // cannot over-reserve toward an out-of-memory abort. A count above the cap is
  // CapExceeded naming `n_audio_layer` (the value is still < MAX_DIM, so only the
  // tighter layer cap rejects it).
  let over = (1usize << 12) + 1;
  assert!(over < (1 << 22));
  let err = ModelDimensions::new(80, 1500, 384, 6, over, 51865, 448, 384, 6, 4).unwrap_err();
  assert!(
    matches!(&err, Error::CapExceeded(p)
      if p.cap_name() == "n_audio_layer" && p.observed() == over as u64),
    "expected CapExceeded on n_audio_layer, got {err:?}"
  );
}

#[test]
fn build_rejects_over_cap_text_layer_count() {
  // Same MAX_LAYERS guard for the decoder layer count (the decoder block `Vec`
  // and the per-step KV-cache `Vec`).
  let over = (1usize << 12) + 1;
  assert!(over < (1 << 22));
  let err = ModelDimensions::new(80, 1500, 384, 6, 4, 51865, 448, 384, 6, over).unwrap_err();
  assert!(
    matches!(&err, Error::CapExceeded(p)
      if p.cap_name() == "n_text_layer" && p.observed() == over as u64),
    "expected CapExceeded on n_text_layer, got {err:?}"
  );
}

#[test]
fn build_accepts_layer_count_at_cap() {
  // The cap is inclusive: exactly MAX_LAYERS layers is accepted (the boundary is
  // not off-by-one). The other product extents stay within their caps at this
  // tiny width, so only the layer-count guard is exercised.
  let at_cap = 1usize << 12;
  assert!(ModelDimensions::new(80, 1500, 8, 2, at_cap, 51865, 448, 8, 2, at_cap).is_ok());
}
