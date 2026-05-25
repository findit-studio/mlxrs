//! M3 piece — happy-path round-trip tests for safetensors + GGUF IO.

use std::{collections::HashMap, fs, path::PathBuf, process};

#[cfg(feature = "gguf")]
use mlxrs::io::GgufMetadata;
use mlxrs::{Array, io};

fn temp_path(name: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!("mlxrs_io_test_{}_{}", process::id(), name));
  p
}

fn sample_arrays() -> HashMap<String, Array> {
  let mut m = HashMap::new();
  m.insert(
    "weight".to_string(),
    Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap(),
  );
  m.insert(
    "bias".to_string(),
    Array::from_slice(&[0.5_f32, 1.5], &(2,)).unwrap(),
  );
  m
}

#[test]
fn safetensors_round_trip() {
  let path = temp_path("rt.safetensors");
  let arrays = sample_arrays();

  io::save_safetensors(&path, &arrays).unwrap();
  let mut loaded = io::load_safetensors(&path).unwrap();

  assert_eq!(loaded.len(), 2);
  let mut w = loaded.remove("weight").unwrap();
  assert_eq!(w.shape(), vec![2, 3]);
  assert_eq!(
    w.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
  let mut b = loaded.remove("bias").unwrap();
  assert_eq!(b.shape(), vec![2]);
  assert_eq!(b.to_vec::<f32>().unwrap(), vec![0.5, 1.5]);

  let _ = fs::remove_file(&path);
}

#[test]
fn safetensors_metadata_round_trip() {
  let path = temp_path("meta.safetensors");
  let arrays = sample_arrays();
  let mut meta = HashMap::new();
  meta.insert("format".to_string(), "mlxrs".to_string());
  meta.insert("version".to_string(), "1".to_string());

  io::save_safetensors_with_metadata(&path, &arrays, &meta).unwrap();
  let (loaded, loaded_meta) = io::load_safetensors_with_metadata(&path).unwrap();

  assert_eq!(loaded.len(), 2);
  assert_eq!(loaded_meta.get("format").map(String::as_str), Some("mlxrs"));
  assert_eq!(loaded_meta.get("version").map(String::as_str), Some("1"));

  let _ = fs::remove_file(&path);
}

#[cfg(feature = "gguf")]
#[test]
fn gguf_round_trip() {
  let path = temp_path("rt.gguf");
  let mut weights = HashMap::new();
  weights.insert(
    "blk.0.weight".to_string(),
    Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap(),
  );
  let mut meta = HashMap::new();
  meta.insert(
    "general.name".to_string(),
    GgufMetadata::String("mlxrs-test".to_string()),
  );
  meta.insert(
    "tokenizer.tokens".to_string(),
    GgufMetadata::StringList(vec!["a".to_string(), "b".to_string()]),
  );

  io::save_gguf(&path, &weights, &meta).unwrap();
  let (mut loaded, loaded_meta) = io::load_gguf(&path).unwrap();

  let mut w = loaded.remove("blk.0.weight").unwrap();
  assert_eq!(w.shape(), vec![2, 2]);
  assert_eq!(w.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

  // GGUF *metadata* round-trip is NOT achievable via mlx-c today: `load_gguf`
  // enumerates keys via `mlx_io_gguf_get_keys`, which mlx-c implements over
  // the weights/arrays map (`GGUFLoad.first`) ONLY — the metadata map
  // (`.second`) is not key-enumerable (vendored `mlx/c/io_types.cpp`; see the
  // `load_gguf` doc comment in `io.rs`). So metadata-only keys are unreachable
  // on load. These negative assertions lock that documented mlx-c upstream-API
  // limitation as a regression guard; the real metadata-enumeration capability
  // is the separately-tracked deferred gguf-metadata follow-up (which would
  // also need to extend the vendored mlx-c surface).
  assert!(
    !loaded_meta.contains_key("general.name"),
    "metadata-only key unexpectedly enumerable — did mlx-c gain metadata-key \
     enumeration? Upgrade this test to assert real round-trip and close the \
     gguf-metadata follow-up."
  );
  assert!(!loaded_meta.contains_key("tokenizer.tokens"));

  let _ = fs::remove_file(&path);
}
