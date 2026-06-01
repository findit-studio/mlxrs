//! Gated end-to-end oracle-parity test for the SigLIP2 NaFlex dual-tower
//! embeddings model.
//!
//! This is the test that ultimately validates the per-image **bilinear +
//! antialias position-embedding resize** + the NaFlex resize fidelity
//! end-to-end: it
//! loads the real `google/siglip2-base-patch16-naflex` checkpoint, runs the
//! crate's preprocessing → `encode_image` / `encode_text`, and compares each
//! output to the matching PyTorch-reference `.npy` by cosine similarity
//! against the upstream worst-case floor.
//!
//! ## Why `#[ignore]` + skip-when-absent
//!
//! The checkpoint weights live in the **gitignored** `/models/siglip2-naflex/`
//! directory (never committed — model weights are out-of-tree per the project
//! convention), so this test cannot run in a clean checkout / CI without the
//! weights present. It is therefore:
//!
//! - `#[ignore]` — excluded from the default `cargo test` run; opt in with
//!   `cargo test --features siglip2-naflex -- --ignored siglip2_oracle`;
//! - **skip-clean** — if `models/siglip2-naflex/` (or the override env var
//!   `MLXRS_SIGLIP2_MODEL_DIR`) is absent, the test prints a skip line and
//!   returns `Ok` rather than failing, so a developer without the weights is
//!   not blocked.
//!
//! ## Fixtures
//!
//! The reference inputs + outputs are the user's published `siglip2-naflex`
//! crate fixtures (`tests/fixtures/`): 12 lossless keyframe PNGs spanning the
//! NaFlex aspect-ratio space with a 1-D `float32` `<name>.npy` per image, plus
//! `text_prompts.json` (12 multilingual prompts) + a `[N, 768]`
//! `text_embeddings.npy`. The fixtures dir defaults to the sibling crate path
//! and can be overridden via `MLXRS_SIGLIP2_FIXTURES_DIR`.
//!
//! ## `.npy` reader
//!
//! mlxrs has no `.npy` dependency, so this test ships a **minimal** little-
//! endian `float32` v1.0 `.npy` parser ([`read_npy_f32`]) — enough to read the
//! reference arrays (1-D `(768,)` and 2-D `(N, 768)`). It is deliberately
//! scoped to the fixture format (C-order, `<f4`); a non-conforming header is a
//! test failure, not a silent mis-read.

#![cfg(feature = "siglip2-naflex")]

use std::path::{Path, PathBuf};

use mlxrs::{
  array::Array,
  embeddings::siglip2_naflex::{
    Siglip2NaflexModel, config::Siglip2NaflexConfig, processing::preprocess, sanitize,
  },
};

/// The upstream worst-case cosine floor from the SigLIP2 NaFlex release
/// validation (the same figure the `siglip2-naflex` crate's parity gate uses).
const COSINE_FLOOR: f32 = 0.99917;

/// The NaFlex per-image patch budget for `base-patch16-naflex`.
const MAX_NUM_PATCHES: u32 = 256;
const PATCH_SIZE: u32 = 16;
const CHANNELS: u32 = 3;
/// Fixed text sequence length the SigLIP2 processor pads to.
const TEXT_SEQ_LEN: usize = 64;

/// Resolve the model directory: `MLXRS_SIGLIP2_MODEL_DIR` if set, else the
/// gitignored in-repo `models/siglip2-naflex/`. Returns `None` (→ skip) if absent.
fn model_dir() -> Option<PathBuf> {
  if let Ok(dir) = std::env::var("MLXRS_SIGLIP2_MODEL_DIR") {
    let p = PathBuf::from(dir);
    return p.is_dir().then_some(p);
  }
  // The crate root is `mlxrs/`; the gitignored weights live at the workspace
  // root `models/siglip2-naflex/`.
  let candidates = [
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/siglip2-naflex"),
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/siglip2-naflex"),
  ];
  candidates.into_iter().find(|p| p.is_dir())
}

/// Resolve the fixtures directory: `MLXRS_SIGLIP2_FIXTURES_DIR` if set, else
/// the sibling `siglip2-naflex` crate's `tests/fixtures/`. Returns `None` if
/// absent.
fn fixtures_dir() -> Option<PathBuf> {
  if let Ok(dir) = std::env::var("MLXRS_SIGLIP2_FIXTURES_DIR") {
    let p = PathBuf::from(dir);
    return p.is_dir().then_some(p);
  }
  let candidates = [
    PathBuf::from("/Users/al/Developer/findit-studio/siglip2-naflex/tests/fixtures"),
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../siglip2-naflex/tests/fixtures"),
  ];
  candidates.into_iter().find(|p| p.is_dir())
}

/// Load + build the model from a local directory (config.json + the merged
/// safetensors shards), running the SigLIP2 `sanitize`.
fn load_model(dir: &Path) -> Siglip2NaflexModel {
  let config_json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config = Siglip2NaflexConfig::from_json(&config_json).expect("parse config.json");

  // Merge every `*.safetensors` shard in the directory (single-file or
  // sharded). A real checkpoint is one of `model.safetensors` or a
  // `model-0000N-of-...` shard set.
  let mut raw = std::collections::HashMap::new();
  for entry in std::fs::read_dir(dir).expect("read model dir") {
    let path = entry.expect("dir entry").path();
    if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
      let part = mlxrs::io::load_safetensors(&path).expect("load safetensors shard");
      raw.extend(part);
    }
  }
  assert!(!raw.is_empty(), "no safetensors found in {}", dir.display());
  let weights = sanitize(raw).expect("sanitize weights");
  Siglip2NaflexModel::from_weights(config, weights).expect("build model")
}

/// Decode a PNG to interleaved RGB bytes via the crate's bounded loader.
fn decode_rgb(path: &Path) -> (Vec<u8>, u32, u32) {
  let img = mlxrs::vlm::image::load_image(path).expect("decode fixture png");
  let rgb = img.to_rgb8();
  let (w, h) = (rgb.width(), rgb.height());
  (rgb.into_raw(), w, h)
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
  // Parse the shape tuple `'shape': (a, b, ...)`.
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

#[test]
#[ignore = "requires the gitignored google/siglip2-base-patch16-naflex weights in models/siglip2-naflex/"]
fn siglip2_oracle_image_parity() {
  let Some(dir) = model_dir() else {
    eprintln!(
      "skipping siglip2_oracle_image_parity: models/siglip2-naflex/ absent \
       (set MLXRS_SIGLIP2_MODEL_DIR to run)"
    );
    return;
  };
  let Some(fix) = fixtures_dir() else {
    eprintln!("skipping siglip2_oracle_image_parity: fixtures dir absent");
    return;
  };
  let model = load_model(&dir);

  let images_dir = fix.join("images");
  let embeds_dir = fix.join("embeddings");
  let mut checked = 0usize;
  for entry in std::fs::read_dir(&images_dir).expect("read images dir") {
    let img_path = entry.expect("image entry").path();
    if img_path.extension().and_then(|e| e.to_str()) != Some("png") {
      continue;
    }
    let stem = img_path.file_stem().and_then(|s| s.to_str()).expect("stem");
    let ref_path = embeds_dir.join(format!("{stem}.npy"));
    if !ref_path.is_file() {
      continue;
    }
    let (rgb, w, h) = decode_rgb(&img_path);
    let inputs = preprocess(&rgb, w, h, PATCH_SIZE, CHANNELS, MAX_NUM_PATCHES).expect("preprocess");
    let embed = to_vec(model.encode_image(&inputs).expect("encode_image"));
    let (reference, ref_shape) = read_npy_f32(&ref_path);
    assert_eq!(
      embed.len(),
      reference.len(),
      "embedding dim mismatch for {stem}: got {} ref {:?}",
      embed.len(),
      ref_shape
    );
    let cos = cosine(&embed, &reference);
    assert!(
      cos >= COSINE_FLOOR,
      "image {stem}: cosine {cos} < floor {COSINE_FLOOR}"
    );
    checked += 1;
  }
  assert!(checked > 0, "no image fixtures checked");
  eprintln!("siglip2_oracle_image_parity: {checked} images >= {COSINE_FLOOR}");
}

#[test]
#[ignore = "requires the gitignored google/siglip2-base-patch16-naflex weights in models/siglip2-naflex/"]
fn siglip2_oracle_text_parity() {
  let Some(dir) = model_dir() else {
    eprintln!(
      "skipping siglip2_oracle_text_parity: models/siglip2-naflex/ absent \
       (set MLXRS_SIGLIP2_MODEL_DIR to run)"
    );
    return;
  };
  let Some(fix) = fixtures_dir() else {
    eprintln!("skipping siglip2_oracle_text_parity: fixtures dir absent");
    return;
  };
  let model = load_model(&dir);

  // The SigLIP2 tokenizer ships with the checkpoint (tokenizer.json).
  let tokenizer = mlxrs::tokenizer::Tokenizer::from_path(&dir, None).expect("load tokenizer");

  let prompts_json =
    std::fs::read_to_string(fix.join("text_prompts.json")).expect("read text_prompts.json");
  let prompts: Vec<String> = parse_json_string_array(&prompts_json);
  assert!(!prompts.is_empty(), "no text prompts in fixture");

  let (reference, ref_shape) = read_npy_f32(&fix.join("text_embeddings.npy"));
  assert_eq!(
    ref_shape.len(),
    2,
    "text_embeddings.npy must be 2-D (N, dim)"
  );
  let (n_ref, dim) = (ref_shape[0], ref_shape[1]);
  assert_eq!(n_ref, prompts.len(), "prompt count vs reference rows");

  for (i, prompt) in prompts.iter().enumerate() {
    // Tokenize with special tokens, then pad/truncate to the fixed seq len
    // with the canonical SigLIP pad id (id 1 for the SigLIP sentencepiece —
    // EOS-sticky pooling reads the last position regardless).
    let mut ids = tokenizer.encode(prompt, true).expect("encode prompt");
    pad_or_truncate(&mut ids, TEXT_SEQ_LEN, siglip_pad_id(&tokenizer));
    let ids_i32: Vec<i32> = ids.iter().map(|&u| u as i32).collect();
    let id_arr = Array::from_slice::<i32>(&ids_i32, &(1usize, TEXT_SEQ_LEN)).expect("ids array");
    let embed = to_vec(model.encode_text(&id_arr).expect("encode_text"));
    assert_eq!(embed.len(), dim, "text embedding dim mismatch");
    let ref_row = &reference[i * dim..(i + 1) * dim];
    let cos = cosine(&embed, ref_row);
    assert!(
      cos >= COSINE_FLOOR,
      "text prompt {i} ({prompt:?}): cosine {cos} < floor {COSINE_FLOOR}"
    );
  }
  eprintln!(
    "siglip2_oracle_text_parity: {} prompts >= {COSINE_FLOOR}",
    prompts.len()
  );
}

/// The SigLIP sentencepiece pad token id (the model pads to a fixed length
/// with this id). Falls back to `1` if the tokenizer does not expose a pad id.
fn siglip_pad_id(_tokenizer: &mlxrs::tokenizer::Tokenizer) -> u32 {
  // SigLIP's processor pads with the EOS/pad token (id 1 in the canonical
  // SigLIP sentencepiece). The sticky-EOS pooling reads the LAST position, so
  // padding with the pad id keeps the last real token in place only when the
  // sequence is already at length; for shorter prompts SigLIP's processor
  // right-pads and the last position is the pad token — which is what the
  // reference fixtures encode, so mirror it exactly.
  1
}

/// Right-pad `ids` to `len` with `pad`, or truncate to `len`.
fn pad_or_truncate(ids: &mut Vec<u32>, len: usize, pad: u32) {
  if ids.len() > len {
    ids.truncate(len);
  } else {
    ids.resize(len, pad);
  }
}

/// Minimal parser for a JSON array of strings (the `text_prompts.json`
/// fixture). A simple two-state scanner: outside a string, only `"` (string
/// open) and `]` (array end) are significant; inside, `\` escapes the next
/// char and `"` closes. Handles unicode content; not a general JSON parser
/// (scoped to the flat string-array fixture).
fn parse_json_string_array(json: &str) -> Vec<String> {
  let mut out = Vec::new();
  let mut chars = json.chars();
  let mut in_string = false;
  let mut escaped = false;
  let mut current = String::new();
  for c in chars.by_ref() {
    if in_string {
      if escaped {
        current.push(c);
        escaped = false;
      } else if c == '\\' {
        escaped = true;
      } else if c == '"' {
        out.push(std::mem::take(&mut current));
        in_string = false;
      } else {
        current.push(c);
      }
    } else if c == '"' {
      in_string = true;
    } else if c == ']' {
      break;
    }
  }
  out
}
