//! Config parse + validate oracles for SenseVoice-Small.
//!
//! Every expected value is the reference dataclass default (`config.py`) or a
//! hand-specified literal — never read back from the code under test.

use super::*;
use crate::error::Error;

// ───────────────────────── defaults ─────────────────────────

#[test]
fn defaults_match_reference_dataclass() {
  // `ModelConfig` defaults (`config.py:54-56`) + the nested dataclass defaults
  // (`config.py:7-17`, `:34-40`).
  let c = Config::default();
  assert_eq!(c.model_type(), "sensevoice");
  assert_eq!(c.vocab_size(), 25055);
  assert_eq!(c.input_size(), 560);

  let enc = c.encoder_conf();
  assert_eq!(enc.output_size(), 512);
  assert_eq!(enc.attention_heads(), 4);
  assert_eq!(enc.linear_units(), 2048);
  assert_eq!(enc.num_blocks(), 50);
  assert_eq!(enc.tp_blocks(), 20);
  assert_eq!(enc.kernel_size(), 11);
  assert_eq!(enc.sanm_shift(), 0);
  assert!(enc.normalize_before());

  let fc = c.frontend_conf();
  assert_eq!(fc.fs(), 16000);
  assert_eq!(fc.window(), "hamming");
  assert_eq!(fc.n_mels(), 80);
  assert_eq!(fc.frame_length(), 25);
  assert_eq!(fc.frame_shift(), 10);
  assert_eq!(fc.lfr_m(), 7);
  assert_eq!(fc.lfr_n(), 6);

  assert!(c.cmvn_means().is_none());
  assert!(c.cmvn_istd().is_none());

  // The crate-exported model-type constant matches the accessor.
  assert_eq!(MODEL_TYPE, "sensevoice");
}

#[test]
fn default_config_validates() {
  assert!(Config::default().validate().is_ok());
}

// ───────────────────────── from JSON ─────────────────────────

#[test]
fn parses_nested_config_json() {
  // A minimal `config.json` with overrides on a couple of fields per nested
  // object; the rest fall through to the dataclass defaults.
  let json = r#"{
    "model_type": "sensevoice",
    "vocab_size": 25055,
    "input_size": 560,
    "encoder_conf": { "output_size": 512, "attention_heads": 4, "num_blocks": 50, "tp_blocks": 20 },
    "frontend_conf": { "lfr_m": 7, "lfr_n": 6 }
  }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert_eq!(c.input_size(), 560);
  assert_eq!(c.encoder_conf().output_size(), 512);
  assert_eq!(c.encoder_conf().linear_units(), 2048); // default fell through
  assert_eq!(c.frontend_conf().lfr_m(), 7);
  assert_eq!(c.frontend_conf().window(), "hamming"); // default fell through
  c.validate().expect("validate");
}

#[test]
fn empty_object_falls_through_to_defaults() {
  // An empty object resolves every field to its dataclass default (the
  // `#[serde(default)]` on every field + the nested `Default`s).
  let c: Config = serde_json::from_str("{}").expect("parse");
  assert_eq!(c.model_type(), "sensevoice");
  assert_eq!(c.vocab_size(), 25055);
  assert_eq!(c.encoder_conf().num_blocks(), 50);
  assert_eq!(c.frontend_conf().lfr_n(), 6);
}

// ───────────────────────── the sanm_shfit typo alias ─────────────────────────

#[test]
fn sanm_shfit_typo_alias_fills_sanm_shift() {
  // The upstream `config.json` misspells the key `sanm_shfit`; serde's `alias`
  // maps it to `sanm_shift` (`config.py:26-28`).
  let json = r#"{ "encoder_conf": { "sanm_shfit": 3 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert_eq!(c.encoder_conf().sanm_shift(), 3);
}

#[test]
fn canonical_sanm_shift_takes_precedence_over_typo() {
  // When BOTH keys are present the canonical `sanm_shift` wins (matching the
  // reference `if "sanm_shfit" in params and "sanm_shift" not in params`).
  let json = r#"{ "encoder_conf": { "sanm_shift": 5, "sanm_shfit": 99 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert_eq!(c.encoder_conf().sanm_shift(), 5);
}

#[test]
fn neither_sanm_key_uses_default_zero() {
  let json = r#"{ "encoder_conf": { "output_size": 512 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert_eq!(c.encoder_conf().sanm_shift(), 0);
}

// ───────────────────────── in-config CMVN ─────────────────────────

#[test]
fn parses_in_config_cmvn_pair() {
  // `input_size = 4` with a matching 4-wide means/istd pair validates. The
  // frontend is set to `lfr_m * n_mels = 1 * 4 = 4` so the LFR-width invariant
  // also holds (isolating the CMVN-length logic under test).
  let json = r#"{
    "input_size": 4,
    "frontend_conf": { "n_mels": 4, "lfr_m": 1 },
    "cmvn_means": [0.1, 0.2, 0.3, 0.4],
    "cmvn_istd":  [1.0, 1.0, 1.0, 1.0]
  }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert_eq!(c.cmvn_means(), Some([0.1f32, 0.2, 0.3, 0.4].as_slice()));
  assert_eq!(c.cmvn_istd(), Some([1.0f32, 1.0, 1.0, 1.0].as_slice()));
  c.validate().expect("validate");
}

#[test]
fn cmvn_length_mismatch_is_rejected() {
  // A 3-wide means against `input_size = 4` is a typed LengthMismatch. The
  // frontend matches `input_size` (`1 * 4`) so the LFR-width check passes and
  // the CMVN length is the only failing invariant.
  let json = r#"{ "input_size": 4, "frontend_conf": { "n_mels": 4, "lfr_m": 1 },
                  "cmvn_means": [0.1, 0.2, 0.3], "cmvn_istd": [1.0, 1.0, 1.0, 1.0] }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::LengthMismatch(_))));
}

#[test]
fn cmvn_means_without_istd_is_rejected() {
  // `n_mels = 4` (the fbank-derived `n_mels > 3` floor) with `lfr_m = 1` keeps
  // `input_size = lfr_m * n_mels = 4`; the half-present CMVN pair is the failing
  // invariant (the `(Some, None)` arm rejects regardless of vector length).
  let json = r#"{ "input_size": 4, "frontend_conf": { "n_mels": 4, "lfr_m": 1 },
                  "cmvn_means": [0.1, 0.2] }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::LengthMismatch(_))));
}

#[test]
fn cmvn_istd_without_means_is_rejected() {
  let json = r#"{ "input_size": 4, "frontend_conf": { "n_mels": 4, "lfr_m": 1 },
                  "cmvn_istd": [1.0, 1.0] }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::LengthMismatch(_))));
}

// ───────────────────────── validation ─────────────────────────

#[test]
fn rejects_output_size_not_divisible_by_heads() {
  // 510 % 4 != 0 — the SANM head reshape would be inexact.
  let mut json =
    serde_json::json!({ "encoder_conf": { "output_size": 510, "attention_heads": 4 } });
  let c: Config = serde_json::from_value(json.take()).expect("parse");
  assert!(matches!(
    c.validate(),
    Err(Error::DivisibilityConstraint(_))
  ));
}

#[test]
fn divisible_output_size_validates() {
  let json = r#"{ "encoder_conf": { "output_size": 512, "attention_heads": 8 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  c.validate().expect("512 % 8 == 0");
}

#[test]
fn rejects_non_positive_vocab_size() {
  let json = r#"{ "vocab_size": 0 }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn rejects_non_positive_input_size() {
  let json = r#"{ "input_size": 0 }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn rejects_zero_fs() {
  let json = r#"{ "frontend_conf": { "fs": 0 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn rejects_negative_sanm_shift() {
  let json = r#"{ "encoder_conf": { "sanm_shift": -1 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn tp_blocks_zero_is_allowed() {
  // A checkpoint with no second stage (`tp_blocks = 0`) is valid.
  let json = r#"{ "encoder_conf": { "tp_blocks": 0 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  c.validate().expect("tp_blocks = 0 is valid");
}

#[test]
fn rejects_over_cap_num_blocks() {
  // `num_blocks` sizes eager per-block `Vec`s, so a value past
  // `MAX_CONFIG_CARDINALITY` (4096) is a typed CapExceeded — it would otherwise
  // request a multi-gigabyte allocation before the first missing-key error.
  let json = r#"{ "encoder_conf": { "num_blocks": 100000 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::CapExceeded(_))));
}

#[test]
fn rejects_over_cap_tp_blocks() {
  // `tp_blocks` likewise sizes an eager `Vec`; an over-cap positive count is a
  // typed CapExceeded (a normal SenseVoice-Small uses 20).
  let json = r#"{ "encoder_conf": { "tp_blocks": 5000 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::CapExceeded(_))));
}

#[test]
fn cardinality_cap_boundary_validates() {
  // Exactly at the cap (4096) is accepted; only strictly-over is rejected. Pair
  // it with a divisible `output_size` so only the cardinality is under test.
  let json = r#"{ "encoder_conf": { "num_blocks": 4096, "tp_blocks": 4096,
                  "output_size": 512, "attention_heads": 4 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  c.validate().expect("counts at the cap are valid");
}

#[test]
fn rejects_non_positive_num_blocks() {
  // `require_cardinality` rejects a non-positive count as OutOfRange (the
  // `encoders0` first block needs `num_blocks >= 1`).
  let json = r#"{ "encoder_conf": { "num_blocks": 0 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn rejects_non_positive_kernel_size() {
  let json = r#"{ "encoder_conf": { "kernel_size": 0 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

// ───────────────────────── LFR width invariant ─────────────────────────

#[test]
fn default_lfr_width_matches_input_size() {
  // The released SenseVoice-Small relation `lfr_m * n_mels == input_size`
  // (`7 * 80 == 560`, `sensevoice.py:62` / `:244-254`).
  let c = Config::default();
  assert_eq!(
    c.frontend_conf().lfr_m() * c.frontend_conf().n_mels(),
    c.input_size()
  );
  c.validate().expect("default LFR width matches input_size");
}

#[test]
fn rejects_lfr_width_mismatching_input_size() {
  // `lfr_m * n_mels = 2 * 80 = 160 != input_size 560` -> a typed LengthMismatch
  // (the front-end would emit 160-wide frames the 560-input encoder can't feed).
  let json = r#"{ "input_size": 560, "frontend_conf": { "lfr_m": 2, "n_mels": 80 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::LengthMismatch(_))));
}

#[test]
fn accepts_non_default_but_consistent_lfr_width() {
  // A non-default front-end whose `lfr_m * n_mels` still equals `input_size`
  // validates (e.g. `5 * 80 = 400`).
  let json = r#"{ "input_size": 400,
                  "frontend_conf": { "lfr_m": 5, "n_mels": 80 },
                  "encoder_conf": { "output_size": 512, "attention_heads": 4 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  c.validate().expect("5 * 80 == 400");
}

#[test]
fn rejects_over_cap_lfr_m() {
  // `lfr_m` sizes the LFR stack width / tile; a value past MAX_CONFIG_CARDINALITY
  // (4096) is a typed CapExceeded (a normal SenseVoice-Small uses 7).
  let json = r#"{ "frontend_conf": { "lfr_m": 100000 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::CapExceeded(_))));
}

#[test]
fn rejects_over_cap_lfr_n() {
  let json = r#"{ "frontend_conf": { "lfr_n": 100000 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::CapExceeded(_))));
}

// ───────────────────────── normalize_before ─────────────────────────

#[test]
fn rejects_normalize_before_false() {
  // SenseVoice-Small is pre-norm only (`config.py:17`); the encoder wires the
  // pre-norm order unconditionally (`sensevoice.py:220-237`). A `false` value is
  // a typed unsupported-configuration InvariantViolation, not a silent mis-run.
  let json = r#"{ "encoder_conf": { "normalize_before": false } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::InvariantViolation(_))));
}

#[test]
fn accepts_normalize_before_true() {
  let json = r#"{ "encoder_conf": { "normalize_before": true } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  c.validate()
    .expect("pre-norm is the supported configuration");
}

// ───────────────────────── FSMN pad split ─────────────────────────

#[test]
fn rejects_sanm_shift_overflowing_right_pad() {
  // `left_padding = (kernel_size - 1) / 2 + sanm_shift`,
  // `right_padding = kernel_size - 1 - left_padding` (`sensevoice.py:164-168`).
  // With `kernel_size = 11` the right slack is `10 - 5 = 5`; a `sanm_shift = 6`
  // drives `right_padding = -1`, which `mx.pad` rejects only at forward — pin it
  // at load as OutOfRange.
  let json = r#"{ "encoder_conf": { "kernel_size": 11, "sanm_shift": 6 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn accepts_sanm_shift_within_right_pad() {
  // `kernel_size = 11`, `sanm_shift = 5` -> `right_padding = 0` (the boundary),
  // which is valid.
  let json = r#"{ "encoder_conf": { "kernel_size": 11, "sanm_shift": 5 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  c.validate().expect("right_padding == 0 is valid");
}

// ───────────────────────── fbank-derived invariants (hoisted to validate) ─────────────────────────

#[test]
fn rejects_unknown_window() {
  // `compute_fbank_kaldi` accepts only hamming / hanning / povey / rectangular
  // (`dsp.py:918-929`); an unknown `window` would fail only at the first fbank.
  // Hoisted to validate as a typed OutOfRange. Keep the LFR width consistent so
  // the window is the only failing invariant.
  let json =
    r#"{ "input_size": 80, "frontend_conf": { "window": "blackman", "n_mels": 80, "lfr_m": 1 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn accepts_each_known_window() {
  // Every window the fbank accepts validates (with a consistent LFR width).
  for window in ["hamming", "hanning", "povey", "rectangular"] {
    let json = format!(
      r#"{{ "input_size": 80, "frontend_conf": {{ "window": "{window}", "n_mels": 80, "lfr_m": 1 }} }}"#
    );
    let c: Config = serde_json::from_str(&json).expect("parse");
    c.validate()
      .unwrap_or_else(|e| panic!("window {window} must validate: {e:?}"));
  }
}

#[test]
fn rejects_n_mels_at_or_below_three() {
  // `get_mel_banks_kaldi` asserts `num_bins > 3` (`dsp.py:822`); `n_mels <= 3`
  // fails only at the first fbank otherwise. Hoisted to validate. `n_mels = 3`
  // with `lfr_m = 1` keeps `input_size = 3` so the mel floor is the only failing
  // invariant.
  let json = r#"{ "input_size": 3, "frontend_conf": { "n_mels": 3, "lfr_m": 1 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn accepts_n_mels_just_above_three() {
  // `n_mels = 4` is the smallest valid mel count.
  let json = r#"{ "input_size": 4, "frontend_conf": { "n_mels": 4, "lfr_m": 1 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  c.validate().expect("n_mels = 4 is the floor and validates");
}

#[test]
fn rejects_derived_win_len_below_two() {
  // `win_len = fs * frame_length / 1000`. `fs = 100`, `frame_length = 10` ->
  // `100 * 10 / 1000 = 1 < 2`, which `compute_fbank_kaldi` rejects (the window
  // denominator is `win_len - 1`). Hoisted to validate as OutOfRange. The mel /
  // LFR / divisibility invariants are kept valid so the derived window is the
  // only failing one.
  let json = r#"{ "input_size": 80,
                  "frontend_conf": { "fs": 100, "frame_length": 10, "frame_shift": 50, "n_mels": 80, "lfr_m": 1 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn rejects_derived_win_inc_zero() {
  // `win_inc = fs * frame_shift / 1000`. `fs = 100`, `frame_shift = 1` ->
  // `100 * 1 / 1000 = 0`, which `compute_fbank_kaldi` rejects (the framing hop
  // cannot be zero). Hoisted to validate as InvariantViolation. `frame_length`
  // is large enough that `win_len >= 2` so `win_inc` is the failing invariant.
  let json = r#"{ "input_size": 80,
                  "frontend_conf": { "fs": 100, "frame_length": 25, "frame_shift": 1, "n_mels": 80, "lfr_m": 1 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::InvariantViolation(_))));
}

#[test]
fn rejects_oversized_derived_win_len() {
  // A pathological `fs` / `frame_length` whose derived `win_len` exceeds the
  // fbank's `MAX_DECODED_SAMPLES` window budget is a typed CapExceeded at load
  // (`compute_fbank_kaldi` enforces the same upper bound). `fs = u32::MAX`,
  // `frame_length = i32::MAX` -> `win_len ≈ 9.2e15 >> 64 Mi`.
  let json = format!(
    r#"{{ "input_size": 80,
          "frontend_conf": {{ "fs": {fs}, "frame_length": {fl}, "frame_shift": 1000, "n_mels": 80, "lfr_m": 1 }} }}"#,
    fs = u32::MAX,
    fl = i32::MAX
  );
  let c: Config = serde_json::from_str(&json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::CapExceeded(_))));
}

#[test]
fn rejects_fs_whose_nyquist_is_at_or_below_low_freq() {
  // SenseVoice's `compute_fbank` always passes the fixed `low_freq = 20 Hz`, and
  // `get_mel_banks_kaldi` rejects `low_freq >= nyquist` (`nyquist = fs / 2`,
  // `dsp.py:826-831`). `fs = 40` -> `nyquist = 20.0 == low_freq`, which fails the
  // strict `low_freq < nyquist` only at the first transcribe otherwise — hoisted
  // to validate as OutOfRange. All other fbank invariants are kept valid
  // (`frame_length = 1000` -> `win_len = 40`; `frame_shift = 100` -> `win_inc =
  // 4`; `n_mels = 80`) so the Nyquist relation is the ONLY failing invariant.
  let json = r#"{ "input_size": 80,
                  "frontend_conf": { "fs": 40, "frame_length": 1000, "frame_shift": 100, "n_mels": 80, "lfr_m": 1 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::OutOfRange(_))));
}

#[test]
fn accepts_fs_whose_nyquist_just_exceeds_low_freq() {
  // The boundary the other side: `fs = 42` -> `nyquist = 21.0 > 20.0`, so the
  // fixed-low-freq Nyquist invariant holds and (with the other fbank invariants
  // kept valid) the config validates. Pins that the check is the strict `>`
  // boundary, not an off-by-one over-rejection of a barely-valid `fs`.
  let json = r#"{ "input_size": 80,
                  "frontend_conf": { "fs": 42, "frame_length": 1000, "frame_shift": 100, "n_mels": 80, "lfr_m": 1 } }"#;
  let c: Config = serde_json::from_str(json).expect("parse");
  c.validate()
    .expect("fs = 42 (nyquist 21 > low_freq 20) validates");
}

#[test]
fn default_frontend_derived_window_sizes_are_valid() {
  // The released SenseVoice-Small defaults (`fs = 16000`, `frame_length = 25`,
  // `frame_shift = 10`) derive `win_len = 400`, `win_inc = 160` — both within
  // the fbank's accepted range, so the default front-end validates.
  let c = Config::default();
  c.validate()
    .expect("default derived window sizes are valid");
}

#[test]
fn left_padding_addition_overflow_is_typed_not_panic() {
  // `left_padding = (kernel_size - 1) / 2 + sanm_shift` is computed through
  // checked arithmetic; an adversarial `kernel_size` + `sanm_shift` pair whose
  // sum overflows `i32` is a typed ArithmeticOverflow, never a debug panic.
  let json = format!(
    r#"{{ "encoder_conf": {{ "kernel_size": {max}, "sanm_shift": {max} }} }}"#,
    max = i32::MAX
  );
  let c: Config = serde_json::from_str(&json).expect("parse");
  assert!(matches!(c.validate(), Err(Error::ArithmeticOverflow(_))));
}
