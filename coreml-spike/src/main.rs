//! CoreML -> ANE-from-Rust feasibility spike.
//!
//! Proves that pure Rust (via the `objc2-core-ml` bindings) can:
//!   1. load a compiled `.mlmodelc` CoreML model,
//!   2. build `MLMultiArray` inputs and run `prediction(from:)`,
//!   3. request the Neural Engine via `MLModelConfiguration.computeUnits`,
//!   4. drive a stateful model through `MLState` (the KV-cache path),
//! all without an Objective-C shim.
//!
//! Models used: the WhisperKit `openai_whisper-tiny` bundle
//! (`MelSpectrogram` / `AudioEncoder` / `TextDecoder` `.mlmodelc`), located at
//! `../models/whisperkit/openai_whisper-tiny/` relative to this crate
//! (gitignored — present for local runs only).
//!
//! Off Apple platforms the crate compiles to a stub `main` that explains the
//! spike is macOS-only, so the workspace stays buildable everywhere.

#[cfg(not(target_vendor = "apple"))]
fn main() {
  eprintln!("coreml-spike: CoreML is Apple-only; nothing to do on this target.");
}

#[cfg(target_vendor = "apple")]
fn main() -> std::process::ExitCode {
  match apple::run() {
    Ok(()) => std::process::ExitCode::SUCCESS,
    Err(e) => {
      eprintln!("coreml-spike FAILED: {e}");
      std::process::ExitCode::FAILURE
    }
  }
}

#[cfg(target_vendor = "apple")]
// `MLMultiArray::dataPointer` is marked deprecated upstream in favour of the
// closure-based `getMutableBytesWithHandler:`, but direct contiguous access is
// the simplest correct path for a spike and is what the bindings still expose.
#[allow(deprecated)]
mod apple {
  use std::path::{Path, PathBuf};

  use objc2::rc::Retained;
  use objc2::runtime::{AnyObject, ProtocolObject};
  use objc2::AllocAnyThread;
  use objc2_core_ml::{
    MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType, MLComputeUnits,
  };
  use objc2_foundation::{NSArray, NSNumber, NSString, NSURL};

  /// Directory holding the WhisperKit tiny `.mlmodelc` bundles, relative to the
  /// crate root (`CARGO_MANIFEST_DIR` = `.../coreml-spike`).
  fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("..")
      .join("models")
      .join("whisperkit")
      .join("openai_whisper-tiny")
  }

  /// Decode one IEEE-754 binary16 (half) value, stored as raw `u16` bits, into
  /// `f32`. Avoids pulling in the `half` crate so the spike stays self-contained.
  fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };
    match exp {
      0 => {
        // subnormal / zero
        sign_f * (frac as f32) * 2f32.powi(-24)
      }
      0x1f => {
        if frac == 0 {
          sign_f * f32::INFINITY
        } else {
          f32::NAN
        }
      }
      _ => {
        let m = 1.0f32 + (frac as f32) / 1024.0;
        sign_f * m * 2f32.powi(exp as i32 - 15)
      }
    }
  }

  /// Build an `NSArray<NSNumber>` shape descriptor from a `usize` slice.
  fn shape_array(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
    let nums: Vec<Retained<NSNumber>> =
      dims.iter().map(|&d| NSNumber::new_isize(d as isize)).collect();
    NSArray::from_retained_slice(&nums)
  }

  /// Allocate an uninitialized `MLMultiArray` of the given shape + dtype.
  fn new_multi_array(
    dims: &[usize],
    dtype: MLMultiArrayDataType,
  ) -> Result<Retained<MLMultiArray>, String> {
    let shape = shape_array(dims);
    // SAFETY: `shape` is a valid NSArray<NSNumber>; `dtype` is a valid enum
    // constant. The returned array's contents are uninitialized (we fill them
    // before use). The `error:` out-param is surfaced as the `Err` arm.
    unsafe {
      MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &shape, dtype)
    }
    .map_err(|e| format!("MLMultiArray init failed: {e:?}"))
  }

  /// Number of scalar elements in an `MLMultiArray`.
  fn ml_count(a: &MLMultiArray) -> usize {
    // SAFETY: `count` is a plain readonly property.
    unsafe { a.count() }.max(0) as usize
  }

  /// Fill an entire Float16 `MLMultiArray` with one constant (given as f32),
  /// writing the raw half bits directly into the backing store.
  fn fill_f16(a: &MLMultiArray, value: f32) {
    let n = ml_count(a);
    let bits = f32_to_f16_bits(value);
    // SAFETY: `dataPointer` is the contiguous first-major backing store of the
    // array; it holds exactly `count` Float16 (u16) scalars, which we just
    // allocated as Float16. We write within `[0, n)`.
    unsafe {
      let p = a.dataPointer().as_ptr().cast::<u16>();
      for i in 0..n {
        p.add(i).write(bits);
      }
    }
  }

  /// Fill an entire Int32 `MLMultiArray` with one constant.
  fn fill_i32(a: &MLMultiArray, value: i32) {
    let n = ml_count(a);
    // SAFETY: as `fill_f16`, but the store holds `count` Int32 scalars.
    unsafe {
      let p = a.dataPointer().as_ptr().cast::<i32>();
      for i in 0..n {
        p.add(i).write(value);
      }
    }
  }

  /// Encode an `f32` to IEEE-754 binary16 bits (round-toward-zero is fine for a
  /// dummy constant input).
  fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let mant = x & 0x7f_ffff;
    if exp <= 0 {
      // flush subnormals/underflow to signed zero (adequate for constants)
      return sign;
    }
    if exp >= 0x1f {
      return sign | 0x7c00; // inf
    }
    let mant10 = (mant >> 13) as u16;
    sign | ((exp as u16) << 10) | mant10
  }

  /// Read the first `k` scalars of a Float16 `MLMultiArray` as `f32`.
  fn read_f16_head(a: &MLMultiArray, k: usize) -> Vec<f32> {
    let n = ml_count(a).min(k);
    let mut out = Vec::with_capacity(n);
    // SAFETY: reading within the array's `count` Float16 scalars.
    unsafe {
      let p = a.dataPointer().as_ptr().cast::<u16>();
      for i in 0..n {
        out.push(f16_bits_to_f32(p.add(i).read()));
      }
    }
    out
  }

  /// Are all of the first `k` scalars finite?
  fn all_finite_head(a: &MLMultiArray, k: usize) -> bool {
    read_f16_head(a, k).into_iter().all(f32::is_finite)
  }

  /// Load a `.mlmodelc` with the given compute units.
  fn load_model(
    path: &Path,
    units: MLComputeUnits,
  ) -> Result<Retained<MLModel>, String> {
    let url = file_url(path);
    let config = unsafe {
      let c = MLModelConfiguration::new();
      c.setComputeUnits(units);
      c
    };
    // SAFETY: `url` points at an existing `.mlmodelc` directory; `config` is a
    // freshly constructed configuration. Errors surface via the `Err` arm.
    unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
      .map_err(|e| format!("load {} failed: {e:?}", path.display()))
  }

  /// A `file://` `NSURL` for a `.mlmodelc` directory.
  fn file_url(path: &Path) -> Retained<NSURL> {
    let s = NSString::from_str(&path.to_string_lossy());
    NSURL::fileURLWithPath_isDirectory(&s, true)
  }

  /// Wrap (name -> MLMultiArray) pairs into an `MLDictionaryFeatureProvider`.
  fn feature_provider(
    pairs: &[(&str, &MLMultiArray)],
  ) -> Result<Retained<ProtocolObject<dyn MLFeatureProvider>>, String> {
    let keys: Vec<Retained<NSString>> =
      pairs.iter().map(|(k, _)| NSString::from_str(k)).collect();
    let vals: Vec<Retained<MLFeatureValue>> = pairs
      .iter()
      .map(|(_, a)| unsafe { MLFeatureValue::featureValueWithMultiArray(a) })
      .collect();

    // Build NSDictionary<NSString, AnyObject>. MLFeatureValue is an NSObject
    // subclass, so each value coerces to `&AnyObject` via its AsRef chain.
    let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    let val_refs: Vec<&AnyObject> = vals.iter().map(|v| v.as_ref()).collect();
    let dict = objc2_foundation::NSDictionary::from_slices(&key_refs, &val_refs);

    // SAFETY: every value in `dict` is an `MLFeatureValue` (a valid feature
    // value), satisfying `initWithDictionary:`'s contract.
    unsafe {
      MLDictionaryFeatureProvider::initWithDictionary_error(
        MLDictionaryFeatureProvider::alloc(),
        &dict,
      )
    }
    .map(|p| ProtocolObject::from_retained(p))
    .map_err(|e| format!("feature provider init failed: {e:?}"))
  }

  /// Pull a named `MLMultiArray` output out of a prediction result.
  fn output_array(
    out: &ProtocolObject<dyn MLFeatureProvider>,
    name: &str,
  ) -> Result<Retained<MLMultiArray>, String> {
    let key = NSString::from_str(name);
    // SAFETY: `featureValueForName:` returns nil for an unknown name (-> None).
    let fv = unsafe { out.featureValueForName(&key) }
      .ok_or_else(|| format!("output '{name}' missing"))?;
    unsafe { fv.multiArrayValue() }
      .ok_or_else(|| format!("output '{name}' is not a MultiArray"))
  }

  /// Human-readable shape of an `MLMultiArray`.
  fn shape_of(a: &MLMultiArray) -> Vec<usize> {
    // SAFETY: `shape` is a readonly NSArray<NSNumber> property.
    let arr = unsafe { a.shape() };
    let mut dims = Vec::new();
    for i in 0..arr.count() {
      let n = arr.objectAtIndex(i);
      dims.push(n.integerValue().max(0) as usize);
    }
    dims
  }

  pub fn run() -> Result<(), String> {
    let dir = models_dir();
    if !dir.exists() {
      return Err(format!(
        "models dir not found: {} (download the WhisperKit tiny bundle)",
        dir.display()
      ));
    }
    println!("== CoreML -> ANE Rust spike ==");
    println!("models: {}", dir.display());

    // ---- Step 2/3: load + predict MelSpectrogram on the ANE-eligible config.
    // Simplest single-array input: audio Float16 [480000] -> mel [1,80,1,3000].
    let mel_path = dir.join("MelSpectrogram.mlmodelc");
    println!("\n[1] MelSpectrogram: load (CPUAndNeuralEngine) + predict");
    let mel_model = load_model(&mel_path, MLComputeUnits::CPUAndNeuralEngine)?;
    println!("    loaded OK (computeUnits = CPUAndNeuralEngine)");

    let audio = new_multi_array(&[480000], MLMultiArrayDataType::Float16)?;
    fill_f16(&audio, 0.0); // silent input -> finite mel
    let mel_in = feature_provider(&[("audio", &audio)])?;
    // SAFETY: input matches the model's declared input schema.
    let mel_out = unsafe { mel_model.predictionFromFeatures_error(&mel_in) }
      .map_err(|e| format!("MelSpectrogram predict failed: {e:?}"))?;
    let mel = output_array(&mel_out, "melspectrogram_features")?;
    println!(
      "    predict OK -> melspectrogram_features shape {:?}, head {:?}, finite={}",
      shape_of(&mel),
      read_f16_head(&mel, 5),
      all_finite_head(&mel, 4096)
    );

    // ---- AudioEncoder: feed the real mel output straight through.
    let enc_path = dir.join("AudioEncoder.mlmodelc");
    println!("\n[2] AudioEncoder: load (CPUAndNeuralEngine) + predict");
    let enc_model = load_model(&enc_path, MLComputeUnits::CPUAndNeuralEngine)?;
    println!("    loaded OK");
    let enc_in = feature_provider(&[("melspectrogram_features", &mel)])?;
    let enc_out = unsafe { enc_model.predictionFromFeatures_error(&enc_in) }
      .map_err(|e| format!("AudioEncoder predict failed: {e:?}"))?;
    let enc = output_array(&enc_out, "encoder_output_embeds")?;
    let enc_shape = shape_of(&enc);
    println!(
      "    predict OK -> encoder_output_embeds shape {:?}, head {:?}, finite={}",
      enc_shape,
      read_f16_head(&enc, 5),
      all_finite_head(&enc, 4096)
    );

    // ---- Step 4: TextDecoder. This WhisperKit decoder uses EXPLICIT-I/O KV
    // cache tensors (key_cache/value_cache in, *_cache_updates out), NOT the
    // iOS18 MLState mechanism. We drive one decode step through it, then also
    // probe MLState creation to confirm that binding is usable from Rust.
    let dec_path = dir.join("TextDecoder.mlmodelc");
    println!("\n[3] TextDecoder: load + one explicit-KV-cache decode step");
    let dec_model = load_model(&dec_path, MLComputeUnits::CPUAndNeuralEngine)?;
    println!("    loaded OK");

    let input_ids = new_multi_array(&[1], MLMultiArrayDataType::Int32)?;
    fill_i32(&input_ids, 50258); // <|startoftranscript|>
    let cache_length = new_multi_array(&[1], MLMultiArrayDataType::Int32)?;
    fill_i32(&cache_length, 0);
    let key_cache = new_multi_array(&[1, 1536, 1, 224], MLMultiArrayDataType::Float16)?;
    fill_f16(&key_cache, 0.0);
    let value_cache = new_multi_array(&[1, 1536, 1, 224], MLMultiArrayDataType::Float16)?;
    fill_f16(&value_cache, 0.0);
    let kv_update_mask = new_multi_array(&[1, 224], MLMultiArrayDataType::Float16)?;
    fill_f16(&kv_update_mask, 0.0);
    // position 0 is the slot being written this step
    // SAFETY: writing within the [1,224] mask store.
    unsafe {
      kv_update_mask.dataPointer().as_ptr().cast::<u16>().write(f32_to_f16_bits(1.0));
    }
    let kpm = new_multi_array(&[1, 224], MLMultiArrayDataType::Float16)?;
    fill_f16(&kpm, 0.0);

    let dec_in = feature_provider(&[
      ("input_ids", &input_ids),
      ("cache_length", &cache_length),
      ("key_cache", &key_cache),
      ("value_cache", &value_cache),
      ("kv_cache_update_mask", &kv_update_mask),
      ("encoder_output_embeds", &enc),
      ("decoder_key_padding_mask", &kpm),
    ])?;
    let dec_out = unsafe { dec_model.predictionFromFeatures_error(&dec_in) }
      .map_err(|e| format!("TextDecoder predict failed: {e:?}"))?;
    let logits = output_array(&dec_out, "logits")?;
    let logits_shape = shape_of(&logits);
    let head = read_f16_head(&logits, 8);
    let argmax = argmax_f16(&logits);
    println!(
      "    predict OK -> logits shape {:?}, head {:?}, argmax token = {}, finite={}",
      logits_shape,
      head,
      argmax,
      all_finite_head(&logits, 51865)
    );

    // ---- MLState probe: can we create + use an MLState from Rust at all?
    probe_mlstate(&dec_model);

    // ---- Step 5: ANE confirmation (best-effort, no sudo).
    enumerate_compute_devices();
    bench_encoder_ane(&enc_model, &mel);
    bench_cpu_vs_ane(&enc_path, &mel);

    println!("\n== spike complete: load + predict + KV-step all RAN ==");
    Ok(())
  }

  /// Best-effort ANE confirmation #3: a CPU-only vs Neural-Engine differential.
  /// We load the SAME encoder twice — once `computeUnits = .cpuOnly`, once
  /// `.cpuAndNeuralEngine` — and time each. A clear speedup on the ANE config is
  /// direct evidence the Neural Engine is actually executing the graph (not just
  /// being listed as available). This needs no sudo/entitlement.
  fn bench_cpu_vs_ane(enc_path: &Path, mel: &MLMultiArray) {
    use std::time::Instant;
    const ITERS: u32 = 200;
    println!("\n[5c] CPU-only vs ANE differential (same AudioEncoder, {ITERS} iters)");

    let time_under = |units: MLComputeUnits, label: &str| -> Option<f64> {
      let model = match load_model(enc_path, units) {
        Ok(m) => m,
        Err(e) => {
          println!("    ({label} load failed: {e})");
          return None;
        }
      };
      let provider = feature_provider(&[("melspectrogram_features", mel)]).ok()?;
      // warmup
      // SAFETY: input matches the model schema.
      let _ = unsafe { model.predictionFromFeatures_error(&provider) };
      let t0 = Instant::now();
      for _ in 0..ITERS {
        // SAFETY: input matches the model schema.
        if unsafe { model.predictionFromFeatures_error(&provider) }.is_err() {
          return None;
        }
      }
      let per = t0.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);
      println!("    {label:<22} {per:.3} ms/encode");
      Some(per)
    };

    let cpu = time_under(MLComputeUnits::CPUOnly, "CPUOnly");
    let ane = time_under(MLComputeUnits::CPUAndNeuralEngine, "CPUAndNeuralEngine");
    if let (Some(cpu), Some(ane)) = (cpu, ane) {
      let speedup = cpu / ane;
      println!(
        "    => ANE speedup vs CPU-only: {speedup:.2}x  ({})",
        if speedup >= 1.15 {
          "Neural Engine is doing work"
        } else {
          "inconclusive — CoreML may have kept this graph on CPU/GPU"
        }
      );
    }
  }

  /// Best-effort ANE confirmation #1: enumerate the compute devices CoreML can
  /// schedule onto, and report whether an `MLNeuralEngineComputeDevice` exists
  /// (+ its core count). Synchronous, no entitlements/sudo required.
  fn enumerate_compute_devices() {
    println!("\n[5a] Compute-device enumeration (MLAllComputeDevices)");
    // SAFETY: `MLAllComputeDevices` is a plain C function returning a retained,
    // non-null NSArray of compute-device protocol objects.
    let devices = unsafe { objc2_core_ml::MLAllComputeDevices() };
    let mut saw_ane = false;
    for i in 0..devices.count() {
      let dev = devices.objectAtIndex(i);
      let any: &AnyObject = dev.as_ref();
      // Identify each device by class membership.
      if let Some(ane) = any.downcast_ref::<objc2_core_ml::MLNeuralEngineComputeDevice>() {
        // SAFETY: `totalCoreCount` is a readonly property on the ANE device.
        let cores = unsafe { ane.totalCoreCount() };
        println!("    device[{i}] = NeuralEngine (ANE), totalCoreCount = {cores}");
        saw_ane = true;
      } else if any
        .downcast_ref::<objc2_core_ml::MLGPUComputeDevice>()
        .is_some()
      {
        println!("    device[{i}] = GPU");
      } else if any
        .downcast_ref::<objc2_core_ml::MLCPUComputeDevice>()
        .is_some()
      {
        println!("    device[{i}] = CPU");
      } else {
        println!("    device[{i}] = <other compute device>");
      }
    }
    println!(
      "    => Neural Engine available as a CoreML scheduling target: {}",
      if saw_ane { "YES" } else { "no" }
    );
  }

  /// Best-effort ANE confirmation #2: time many encoder predictions. A human can
  /// run `sudo powermetrics --samplers ane_power` (or Instruments' Neural Engine
  /// track) alongside this to observe ANE residency; the throughput itself is
  /// also far above what CPU-only Float16 conv attention would yield on tiny.
  fn bench_encoder_ane(enc_model: &MLModel, mel: &MLMultiArray) {
    use std::time::Instant;
    const ITERS: u32 = 300;
    println!("\n[5b] AudioEncoder timing loop ({ITERS} iters, watch ANE externally)");
    let provider = match feature_provider(&[("melspectrogram_features", mel)]) {
      Ok(p) => p,
      Err(e) => {
        println!("    (skipped: {e})");
        return;
      }
    };
    // Warm up (first run triggers ANE program specialization/caching).
    // SAFETY: input matches the model schema.
    if let Err(e) = unsafe { enc_model.predictionFromFeatures_error(&provider) } {
      println!("    (warmup failed: {e:?})");
      return;
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
      // SAFETY: input matches the model schema; result is dropped each iter.
      if unsafe { enc_model.predictionFromFeatures_error(&provider) }.is_err() {
        println!("    (a prediction failed mid-loop)");
        return;
      }
    }
    let dt = t0.elapsed();
    let per = dt.as_secs_f64() * 1e3 / f64::from(ITERS);
    println!(
      "    {ITERS} encoder predictions in {:.2}s => {per:.3} ms/encode ({:.1} enc/s)",
      dt.as_secs_f64(),
      f64::from(ITERS) / dt.as_secs_f64()
    );
  }

  /// argmax over a Float16 logits array (small enough for tiny whisper vocab).
  fn argmax_f16(a: &MLMultiArray) -> usize {
    let n = ml_count(a);
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    // SAFETY: scanning within the array's `count` Float16 scalars.
    unsafe {
      let p = a.dataPointer().as_ptr().cast::<u16>();
      for i in 0..n {
        let v = f16_bits_to_f32(p.add(i).read());
        if v > best_v {
          best_v = v;
          best = i;
        }
      }
    }
    best
  }

  /// Probe whether `MLState` (the iOS18 stateful-model KV mechanism) is
  /// creatable + threadable from Rust. WhisperKit's tiny decoder is NOT
  /// declared stateful, so `newState()` returns an empty state and a stateful
  /// prediction is equivalent to a stateless one — but exercising the call
  /// path proves the binding works for genuinely-stateful models.
  fn probe_mlstate(model: &MLModel) {
    println!("\n[4] MLState probe (stateful-decoder KV path)");
    // SAFETY: `newState` is always callable; returns an empty state for a
    // stateless model.
    let state = unsafe { model.newState() };
    println!("    model.newState() -> MLState created OK: {state:p}");

    // Re-run a decode step via predictionFromFeatures_usingState_error to prove
    // the stateful prediction entrypoint is reachable from Rust.
    let input_ids = match new_multi_array(&[1], MLMultiArrayDataType::Int32) {
      Ok(a) => {
        fill_i32(&a, 50258);
        a
      }
      Err(e) => {
        println!("    (skip stateful predict: {e})");
        return;
      }
    };
    let cache_length = new_multi_array(&[1], MLMultiArrayDataType::Int32).unwrap();
    fill_i32(&cache_length, 0);
    let key_cache =
      new_multi_array(&[1, 1536, 1, 224], MLMultiArrayDataType::Float16).unwrap();
    fill_f16(&key_cache, 0.0);
    let value_cache =
      new_multi_array(&[1, 1536, 1, 224], MLMultiArrayDataType::Float16).unwrap();
    fill_f16(&value_cache, 0.0);
    let kv_update_mask =
      new_multi_array(&[1, 224], MLMultiArrayDataType::Float16).unwrap();
    fill_f16(&kv_update_mask, 0.0);
    let enc =
      new_multi_array(&[1, 384, 1, 1500], MLMultiArrayDataType::Float16).unwrap();
    fill_f16(&enc, 0.0);
    let kpm = new_multi_array(&[1, 224], MLMultiArrayDataType::Float16).unwrap();
    fill_f16(&kpm, 0.0);

    let provider = match feature_provider(&[
      ("input_ids", &input_ids),
      ("cache_length", &cache_length),
      ("key_cache", &key_cache),
      ("value_cache", &value_cache),
      ("kv_cache_update_mask", &kv_update_mask),
      ("encoder_output_embeds", &enc),
      ("decoder_key_padding_mask", &kpm),
    ]) {
      Ok(p) => p,
      Err(e) => {
        println!("    (skip stateful predict: {e})");
        return;
      }
    };
    // SAFETY: stateful-prediction entrypoint; `state` came from this model's
    // `newState()`, inputs match the model schema. Each call is serial.
    match unsafe {
      model.predictionFromFeatures_usingState_error(&provider, &state)
    } {
      Ok(out) => match output_array(&out, "logits") {
        Ok(logits) => println!(
          "    prediction(from:using:) OK -> logits shape {:?} (stateful path reachable)",
          shape_of(&logits)
        ),
        Err(e) => println!("    stateful predict ran but output read failed: {e}"),
      },
      Err(e) => println!("    stateful predict failed: {e}"),
    }
  }
}
