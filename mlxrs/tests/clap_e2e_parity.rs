//! Gated end-to-end parity test for the CLAP-HTSAT-unfused dual-tower
//! audio+text embeddings model (`laion/clap-htsat-unfused`).
//!
//! This is the Phase-5 parity gate: it loads the real checkpoint into the
//! crate's [`ClapModel`], runs the SAME `sample.wav` + the SAME zero-shot
//! labels that `textclap` ran through its ONNX graph, and compares each output
//! (the audio embedding, the per-label text embeddings, and the `classify`
//! ranking) to `textclap`'s committed ONNX goldens.
//!
//! ## Checkpoint state — why this is dormant today
//!
//! `laion/clap-htsat-unfused` is downloaded at the gitignored workspace path
//! `models/clap-htsat-unfused/`, but the HF repo ships **`pytorch_model.bin`
//! only** — mlxrs loads `model.safetensors` (the factory's `load_safetensors`
//! shard loader; this test merges every `*.safetensors` in the directory). The
//! `.bin → model.safetensors` conversion is **pending** (the local environment
//! lacks `torch`); convert with, e.g.:
//!
//! ```text
//! python -c "import torch, safetensors.torch as st; \
//!   sd = torch.load('models/clap-htsat-unfused/pytorch_model.bin', \
//!                   map_location='cpu', weights_only=True); \
//!   st.save_file({k: v.contiguous() for k, v in sd.items()}, \
//!                'models/clap-htsat-unfused/model.safetensors')"
//! ```
//!
//! Until that lands the test is **`#[ignore]`d** (CI skips it; opt in with
//! `cargo test --features clap -- --ignored clap_e2e`, or `tarpaulin
//! --include-ignored`) AND **checkpoint-absent-guarded** — if neither
//! `model.safetensors` nor `MLXRS_CLAP_MODEL_DIR` resolves, it prints a skip
//! line and returns `Ok` rather than panicking. So an accidental `--ignored`
//! run without the converted weights is a no-op, not a failure.
//!
//! ## Tolerance — why cosine / rank-agreement, NOT max-abs
//!
//! `textclap`'s goldens come from the **int8-quantized** ONNX graphs
//! (`audio_model_quantized.onnx` / `text_model_quantized.onnx`), while mlxrs
//! loads the **fp32** safetensors. The end-to-end tolerance therefore has to
//! absorb three independent drift sources, none of which a max-abs bound can
//! accommodate:
//!
//! 1. **int8-vs-fp quantization drift** — `textclap`'s own design budgets
//!    ~1e-3..5e-3 for the int8 ONNX vs fp reference.
//! 2. **f64-vs-f32 STFT mel drift** — `textclap` computes the mel STFT in f64
//!    to match HF (~1e-4 vs an f32 STFT); mlxrs's `stft` is f32/MLX.
//! 3. **`reshape_mel2img` align-corners bicubic boundary approximation** — the
//!    HTSAT mel→image fold's bicubic interpolation differs at the grid
//!    boundary from the reference (noted in the audio tower), plus Swin
//!    attention summation-order differences.
//!
//! So both goldens and the mlxrs outputs are **L2-normalized** unit vectors
//! (`textclap`'s `regen_golden.py` saves `raw / ‖raw‖`; mlxrs `embed_audio` /
//! `embed_text` L2-normalize as their final step), and the gate is a
//! **`1 − cos ≤ COSINE_DISTANCE_TOL`** bound on the final embeddings plus a
//! **rank-agreement** check on `classify` (the top-k labels and their order
//! match, scores within a loose band) — never bit-exact scores. If a tighter
//! parity is ever needed, regenerate an fp32-ONNX `textclap` golden to remove
//! drift source (1).
//!
//! ## Tokenizer note (R7)
//!
//! `textclap`'s text goldens were tokenized with its bundled Xenova
//! `tokenizer.json`; mlxrs loads the `laion/clap-htsat-unfused` checkpoint's
//! own `tokenizer.json` (the two differ subtly per the port plan). The cosine
//! tolerance absorbs any resulting near-boundary token-id differences on these
//! short labels.
//!
//! ## `.npy` reader
//!
//! mlxrs's `io::load_npy` yields a lazy [`Array`]; this test instead ships a
//! minimal little-endian C-order `float32` reader ([`read_npy_f32`]) returning
//! `(data, shape)` so the comparison is pure-host (the goldens are 1-D
//! `(512,)` and 2-D `(5, 512)`). A non-conforming header is a test failure, not
//! a silent mis-read.

#![cfg(feature = "clap")]

use std::path::{Path, PathBuf};

use mlxrs::{
  array::Array,
  embeddings::{EmbeddingModel, clap::ClapModel},
};

/// The end-to-end `1 − cos` budget on the final L2-normalized embeddings. Set
/// to the int8-ONNX-vs-fp budget from the port plan (§9): int8 quantization
/// drift (~1e-3..5e-3) + f64-vs-f32 STFT mel drift (~1e-4) + the
/// `reshape_mel2img` align-corners bicubic boundary approximation + Swin
/// summation-order differences. Cosine-based, never max-abs.
const COSINE_DISTANCE_TOL: f32 = 1e-2;

/// The zero-shot labels `textclap`'s `regen_golden.py` used to produce
/// `golden_text_embs.npy`, **in row order** (row `i` of the golden is
/// `LABELS[i]`). Sourced verbatim from `regen_golden.py` `LABELS` and the
/// fixtures `README.md` (`sample.wav` is a dog bark, so `"a dog barking"`
/// is the expected top-1).
const LABELS: [&str; 5] = ["a dog barking", "rain", "music", "silence", "door creaking"];

/// The audio fixture's native sample rate (`sample.wav` is 48 kHz mono, the
/// CLAP front-end's required rate — no resample needed; the model's
/// `embed_audio` repeat-pads / head-truncates to its fixed 10 s window).
const AUDIO_SAMPLE_RATE: u32 = 48_000;

/// CLAP's shared embedding dimension (`projection_dim`); both goldens are
/// `512`-wide.
const PROJECTION_DIM: usize = 512;

/// Resolve the model directory: `MLXRS_CLAP_MODEL_DIR` if set, else the
/// gitignored in-repo `models/clap-htsat-unfused/`. Returns `None` (→ skip)
/// when the directory or its `model.safetensors` is absent (the pending-
/// conversion state documented at the top of this file).
fn model_dir() -> Option<PathBuf> {
  let dir = if let Ok(dir) = std::env::var("MLXRS_CLAP_MODEL_DIR") {
    PathBuf::from(dir)
  } else {
    // The crate root is `mlxrs/`; the gitignored weights live at the workspace
    // root `models/clap-htsat-unfused/`.
    let candidates = [
      PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/clap-htsat-unfused"),
      PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/clap-htsat-unfused"),
    ];
    candidates.into_iter().find(|p| p.is_dir())?
  };
  // The checkpoint must carry at least one `*.safetensors` shard. The HF repo
  // ships `pytorch_model.bin` only; until it is converted to
  // `model.safetensors`, the directory exists but has no safetensors, so this
  // returns `None` (→ skip) rather than asserting an empty weight map later.
  let has_safetensors = std::fs::read_dir(&dir).ok().is_some_and(|entries| {
    entries
      .flatten()
      .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
  });
  has_safetensors.then_some(dir)
}

/// Resolve the `textclap` fixtures directory (`sample.wav` + the `golden_*.npy`
/// oracles): `MLXRS_CLAP_FIXTURES_DIR` if set, else the sibling `textclap`
/// crate's `tests/fixtures/`. Returns `None` if absent.
fn fixtures_dir() -> Option<PathBuf> {
  if let Ok(dir) = std::env::var("MLXRS_CLAP_FIXTURES_DIR") {
    let p = PathBuf::from(dir);
    return p.is_dir().then_some(p);
  }
  let candidates = [
    PathBuf::from("/Users/al/Developer/findit-studio/textclap/tests/fixtures"),
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../textclap/tests/fixtures"),
  ];
  candidates.into_iter().find(|p| p.is_dir())
}

/// Load + build the [`ClapModel`] from a local directory (`config.json` + the
/// merged `*.safetensors` shards), running the CLAP `sanitize` — the same
/// `config-parse → load-safetensors → sanitize → from_weights` path the load
/// factory drives.
fn load_model(dir: &Path) -> ClapModel {
  let config_json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config =
    mlxrs::embeddings::clap::ClapConfig::from_json(&config_json).expect("parse config.json");

  // Merge every `*.safetensors` shard in the directory (single-file
  // `model.safetensors` or a `model-0000N-of-...` shard set).
  let mut raw = std::collections::HashMap::new();
  for entry in std::fs::read_dir(dir).expect("read model dir") {
    let path = entry.expect("dir entry").path();
    if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
      let part = mlxrs::io::load_safetensors(&path).expect("load safetensors shard");
      raw.extend(part);
    }
  }
  assert!(!raw.is_empty(), "no safetensors found in {}", dir.display());
  let weights = mlxrs::embeddings::clap::sanitize(raw).expect("sanitize weights");
  ClapModel::from_weights(config, weights).expect("build ClapModel")
}

/// Cosine similarity between two equal-length f32 vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
  assert_eq!(a.len(), b.len(), "cosine: length mismatch");
  let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
  let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
  let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
  dot / (na * nb).max(1e-12)
}

/// Evaluate an `Array` to a host `Vec<f32>`.
fn to_vec(mut a: Array) -> Vec<f32> {
  a.eval().expect("eval");
  a.to_vec::<f32>().expect("to_vec f32")
}

/// Minimal little-endian C-order `float32` `.npy` v1.0/v2.0 reader, returning
/// `(data, shape)`. Scoped to the fixture format (`<f4`, `fortran_order:
/// False`); anything else is a test failure.
fn read_npy_f32(path: &Path) -> (Vec<f32>, Vec<usize>) {
  let bytes = std::fs::read(path).expect("read npy");
  assert!(
    bytes.len() > 10 && &bytes[0..6] == b"\x93NUMPY",
    "not a .npy: {}",
    path.display()
  );
  let major = bytes[6];
  // Header length is u16 (v1) or u32 (v2+) little-endian, after the 8-byte
  // magic+version.
  let (header_len, header_start) = if major == 1 {
    (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
  } else {
    (
      u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
      12usize,
    )
  };
  let header =
    std::str::from_utf8(&bytes[header_start..header_start + header_len]).expect("npy header utf8");
  assert!(
    header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\""),
    "npy must be little-endian float32: {header}"
  );
  assert!(
    header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false"),
    "npy must be C-order: {header}"
  );
  let shape = parse_npy_shape(header);
  let n: usize = shape.iter().product();
  let data_start = header_start + header_len;
  let data_bytes = &bytes[data_start..];
  assert!(
    data_bytes.len() >= n * 4,
    "npy data truncated: have {} need {}",
    data_bytes.len(),
    n * 4
  );
  let mut data = Vec::with_capacity(n);
  for i in 0..n {
    let off = i * 4;
    data.push(f32::from_le_bytes([
      data_bytes[off],
      data_bytes[off + 1],
      data_bytes[off + 2],
      data_bytes[off + 3],
    ]));
  }
  (data, shape)
}

/// Parse the `'shape': (..)` tuple out of a `.npy` header dict string.
fn parse_npy_shape(header: &str) -> Vec<usize> {
  let key = header.find("'shape'").or_else(|| header.find("\"shape\""));
  let start = key.expect("npy header has shape");
  let open = header[start..].find('(').expect("shape tuple open") + start + 1;
  let close = header[open..].find(')').expect("shape tuple close") + open;
  header[open..close]
    .split(',')
    .filter_map(|s| {
      let t = s.trim();
      (!t.is_empty()).then(|| t.parse::<usize>().expect("npy shape dim"))
    })
    .collect()
}

/// Load `sample.wav` as 48 kHz mono `f32` samples via the crate's bounded audio
/// loader. The fixture is already 48 kHz mono, so no resample is needed (a
/// surprising rate is a fixture regression, asserted here rather than silently
/// resampled — the goldens were generated at exactly 48 kHz).
fn load_samples(fixtures: &Path) -> Vec<f32> {
  let (samples, sample_rate) =
    mlxrs::audio::io::load_audio(&fixtures.join("sample.wav")).expect("load sample.wav");
  assert_eq!(
    sample_rate, AUDIO_SAMPLE_RATE,
    "sample.wav is expected to be {AUDIO_SAMPLE_RATE} Hz (the CLAP front-end rate the goldens used)"
  );
  samples
}

/// Build the `(num_labels, seq_len)` token-id batch + matching `{0,1}`
/// attention mask `ClapModel::classify` consumes, reproducing the text tower's
/// `DynamicRightPad` scheme (the model owns tokenization via its `TextEmbedder`
/// seam, but not the label→ids step — exactly what the generic `encode`
/// pipeline drives). Each label is tokenized **with** RoBERTa's `<s>` / `</s>`
/// special tokens, then every row is right-padded to the batch-max length with
/// the pad id (`1`), the mask carrying `1` over real tokens and `0` over pad
/// cells. (The pooled embedding is mask-invariant to the per-row length / pad
/// id, so this matches the per-label `encode` path used for the embedding
/// parity below.)
fn tokenize_labels_batch(
  tokenizer: &mlxrs::tokenizer::Tokenizer,
  labels: &[&str],
) -> (Array, Array) {
  // RoBERTa pad id (`config.json` `pad_token_id = 1`, mirrored by the text
  // tower's `DynamicRightPad { pad_token_id: 1 }`).
  const PAD_TOKEN_ID: i32 = 1;

  let rows: Vec<Vec<u32>> = labels
    .iter()
    .map(|label| tokenizer.encode(label, true).expect("encode label"))
    .collect();
  let seq_len = rows.iter().map(Vec::len).max().unwrap_or(0);

  let mut ids: Vec<i32> = Vec::with_capacity(rows.len() * seq_len);
  // The attention mask is an f32 `{0, 1}` array — `build_additive_mask`
  // compares it against an f32 `0` (matching the encode pipeline + the text
  // tower unit tests).
  let mut mask: Vec<f32> = Vec::with_capacity(rows.len() * seq_len);
  for row in &rows {
    for j in 0..seq_len {
      if j < row.len() {
        ids.push(row[j] as i32);
        mask.push(1.0);
      } else {
        ids.push(PAD_TOKEN_ID);
        mask.push(0.0);
      }
    }
  }
  let shape = (rows.len(), seq_len);
  (
    Array::from_slice::<i32>(&ids, &shape).expect("label ids array"),
    Array::from_slice::<f32>(&mask, &shape).expect("label mask array"),
  )
}

#[test]
#[ignore = "requires the converted laion/clap-htsat-unfused model.safetensors in models/clap-htsat-unfused/ (HF ships pytorch_model.bin only; see the module docs for the conversion)"]
fn clap_e2e_audio_parity() {
  let Some(dir) = model_dir() else {
    eprintln!(
      "skipping clap_e2e_audio_parity: models/clap-htsat-unfused/model.safetensors absent \
       (HF ships pytorch_model.bin only — convert it, or set MLXRS_CLAP_MODEL_DIR; see module docs)"
    );
    return;
  };
  let Some(fix) = fixtures_dir() else {
    eprintln!("skipping clap_e2e_audio_parity: textclap fixtures dir absent");
    return;
  };
  let model = load_model(&dir);
  let samples = load_samples(&fix);

  // mlxrs audio embedding: `extract_mel → HTSAT tower → audio_projection →
  // L2-normalize`, the `(1, 512)` unit embedding.
  let embed = to_vec(
    model
      .embed_audio(&samples)
      .expect("embed_audio")
      .into_array(),
  );
  assert_eq!(embed.len(), PROJECTION_DIM, "audio embedding dim");

  // Golden: `textclap`'s int8 audio-ONNX output, L2-normalized — `(512,)`.
  let (reference, ref_shape) = read_npy_f32(&fix.join("golden_audio_emb.npy"));
  assert_eq!(
    ref_shape,
    vec![PROJECTION_DIM],
    "golden_audio_emb.npy must be (512,)"
  );

  let dist = 1.0 - cosine(&embed, &reference);
  assert!(
    dist <= COSINE_DISTANCE_TOL,
    "audio embedding: 1 - cos = {dist} > tol {COSINE_DISTANCE_TOL}"
  );
  eprintln!("clap_e2e_audio_parity: 1 - cos = {dist} <= {COSINE_DISTANCE_TOL}");
}

#[test]
#[ignore = "requires the converted laion/clap-htsat-unfused model.safetensors in models/clap-htsat-unfused/ (HF ships pytorch_model.bin only; see the module docs for the conversion)"]
fn clap_e2e_text_parity() {
  let Some(dir) = model_dir() else {
    eprintln!(
      "skipping clap_e2e_text_parity: models/clap-htsat-unfused/model.safetensors absent \
       (HF ships pytorch_model.bin only — convert it, or set MLXRS_CLAP_MODEL_DIR; see module docs)"
    );
    return;
  };
  let Some(fix) = fixtures_dir() else {
    eprintln!("skipping clap_e2e_text_parity: textclap fixtures dir absent");
    return;
  };
  let model = load_model(&dir);

  // The RoBERTa BPE tokenizer ships with the checkpoint (`tokenizer.json`).
  let tokenizer = mlxrs::tokenizer::Tokenizer::from_path(&dir, None).expect("load tokenizer");

  // Golden: `textclap`'s per-label int8 text-ONNX outputs, L2-normalized,
  // stacked `(5, 512)` in `LABELS` row order.
  let (reference, ref_shape) = read_npy_f32(&fix.join("golden_text_embs.npy"));
  assert_eq!(
    ref_shape,
    vec![LABELS.len(), PROJECTION_DIM],
    "golden_text_embs.npy must be (num_labels, 512)"
  );

  // Drive the model's text tower through the REAL generic `encode` pipeline (it
  // reads the model's `TextEncoding` — RoBERTa special tokens + DynamicRightPad
  // — tokenizes, masks, and runs `embed_text`), one label at a time to mirror
  // `regen_golden.py` (which embeds each label independently). Each output is
  // the `(1, 512)` L2-normalized text embedding.
  let text_embedder = model
    .as_text_embedder()
    .expect("clap exposes a text embedder");
  for (i, label) in LABELS.iter().enumerate() {
    let embed =
      to_vec(mlxrs::embeddings::encode(text_embedder, &tokenizer, &[label]).expect("encode label"));
    assert_eq!(
      embed.len(),
      PROJECTION_DIM,
      "text embedding dim ({label:?})"
    );
    let ref_row = &reference[i * PROJECTION_DIM..(i + 1) * PROJECTION_DIM];
    let dist = 1.0 - cosine(&embed, ref_row);
    assert!(
      dist <= COSINE_DISTANCE_TOL,
      "text label {i} ({label:?}): 1 - cos = {dist} > tol {COSINE_DISTANCE_TOL}"
    );
  }
  eprintln!(
    "clap_e2e_text_parity: {} labels all within 1 - cos <= {COSINE_DISTANCE_TOL}",
    LABELS.len()
  );
}

#[test]
#[ignore = "requires the converted laion/clap-htsat-unfused model.safetensors in models/clap-htsat-unfused/ (HF ships pytorch_model.bin only; see the module docs for the conversion)"]
fn clap_e2e_classify_rank_agreement() {
  let Some(dir) = model_dir() else {
    eprintln!(
      "skipping clap_e2e_classify_rank_agreement: models/clap-htsat-unfused/model.safetensors \
       absent (HF ships pytorch_model.bin only — convert it, or set MLXRS_CLAP_MODEL_DIR; see \
       module docs)"
    );
    return;
  };
  let Some(fix) = fixtures_dir() else {
    eprintln!("skipping clap_e2e_classify_rank_agreement: textclap fixtures dir absent");
    return;
  };
  let model = load_model(&dir);
  let tokenizer = mlxrs::tokenizer::Tokenizer::from_path(&dir, None).expect("load tokenizer");
  let samples = load_samples(&fix);

  // Reference ranking: cosine(golden audio embed, each golden text embed),
  // sorted descending — the ranking `textclap`'s ONNX `classify` produces from
  // the same goldens. (Both are unit vectors, so cosine == dot.)
  let (audio_ref, _) = read_npy_f32(&fix.join("golden_audio_emb.npy"));
  let (text_ref, _) = read_npy_f32(&fix.join("golden_text_embs.npy"));
  let mut ref_scores: Vec<(usize, f32)> = (0..LABELS.len())
    .map(|i| {
      let row = &text_ref[i * PROJECTION_DIM..(i + 1) * PROJECTION_DIM];
      (i, cosine(&audio_ref, row))
    })
    .collect();
  ref_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
  let ref_order: Vec<usize> = ref_scores.iter().map(|&(i, _)| i).collect();

  // mlxrs ranking: `classify(samples, label_ids, label_masks, k)` — embed the
  // audio once, embed each label, cosine top-k (descending, stable tie-break).
  let (label_ids, label_masks) = tokenize_labels_batch(&tokenizer, &LABELS);
  let ranked = model
    .classify(&samples, &label_ids, &label_masks, LABELS.len())
    .expect("classify");
  assert_eq!(
    ranked.len(),
    LABELS.len(),
    "classify returns all {} labels for k = len",
    LABELS.len()
  );
  let mlx_order: Vec<usize> = ranked.iter().map(|&(i, _)| i).collect();

  // Rank-agreement (the §9 loose gate): the TOP LABEL must match the reference
  // top label (the dog-bark fixture → `"a dog barking"`, index 0). This is the
  // zero-shot-classification primitive — getting the winner right is the
  // contract, while the exact tail order can wobble under the int8-vs-fp +
  // mel-drift budget.
  assert_eq!(
    mlx_order.first(),
    ref_order.first(),
    "classify top-1 disagrees: mlxrs {mlx_order:?} vs reference {ref_order:?}"
  );
  // The dog-bark audio must rank `"a dog barking"` (index 0) first.
  assert_eq!(
    mlx_order.first(),
    Some(&0usize),
    "expected the dog-bark fixture to rank \"a dog barking\" (index 0) top, got {mlx_order:?}"
  );

  // Per-label score band: each mlxrs cosine score must be close to the
  // reference cosine for the SAME label (a loose band, not bit-exact — the
  // int8-vs-fp + mel-drift budget). Build a label→reference-score lookup.
  let mut ref_by_label = [0f32; LABELS.len()];
  for &(i, s) in &ref_scores {
    ref_by_label[i] = s;
  }
  for &(i, score) in &ranked {
    let gap = (score - ref_by_label[i]).abs();
    assert!(
      gap <= SCORE_BAND,
      "classify score for label {i} ({:?}): mlxrs {score} vs reference {} (gap {gap} > band \
       {SCORE_BAND})",
      LABELS[i],
      ref_by_label[i]
    );
  }
  eprintln!(
    "clap_e2e_classify_rank_agreement: top-1 = label {} ({:?}); full order {mlx_order:?} \
     (reference {ref_order:?})",
    mlx_order[0], LABELS[mlx_order[0]]
  );
}

/// Loose per-label cosine-score band for the `classify` parity. The scores are
/// cosines of unit embeddings (in `[-1, 1]`); the int8-ONNX-vs-fp +
/// mel/STFT-drift budget can shift an individual label's score by more than the
/// `1e-2` embedding `1 − cos` bound (a small embedding rotation moves a cosine
/// against a fixed audio vector more than it moves the self-cosine), so this
/// band is deliberately wider than `COSINE_DISTANCE_TOL`. The rank check (top-1
/// agreement) is the primary gate; this band guards against a grossly wrong
/// score while tolerating the documented drift.
const SCORE_BAND: f32 = 5e-2;
