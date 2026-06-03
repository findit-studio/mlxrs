//! The SenseVoice-Small file-loading factory — `from_weights` /
//! `from_weights_quantized`, the safetensors shard walk + merge, the
//! quantization resolution, the `am.mvn` / tokenizer asset loading (the
//! `post_load_hook` equivalent), and the `load(path) -> Box<dyn Transcribe>`
//! entry the STT factory dispatches to.
//!
//! Faithful port of the reference loading path in [`sensevoice.py`][sv]:
//! `sanitize` (`:554-565`, already in [`super::frontend`]) and `post_load_hook`
//! (`:567-598`) — the `am.mvn` CMVN parse + the in-config CMVN fallback + the
//! two-tier SentencePiece / `tokens.json` tokenizer resolution. The disk
//! pipeline mirrors mlx-audio's shared loader (`get_model_path` ->
//! `load_config` -> `apply_quantization` -> the per-architecture
//! `sanitize` + `load_weights` + `post_load_hook`), reusing the same shared
//! helpers the whisper / wav2vec2 loaders do.
//!
//! ## Quantization resolution (the qwen3 discriminator)
//!
//! Every quantize-aware layer ([`crate::nn::MaybeQuantizedLinear`] /
//! [`crate::nn::MaybeQuantizedEmbedding`]) selects quantized-vs-dense by the per-layer
//! `<prefix>.scales` sibling.
//! [`SenseVoiceModel::from_weights_quantized`](crate::audio::stt::models::sensevoice::SenseVoiceModel::from_weights_quantized)
//! threads the parsed [`PerLayerQuantization`] to the builders ONLY when
//! [`has_relevant_scales`] finds a
//! `.scales` for some layer the model actually loads — a one-pass pre-scan over
//! the weight keys, so a **dense** checkpoint loads through the unchanged dense
//! path regardless of any stale / partial `quantization` block the config may
//! still carry. The `(group_size, bits, mode)` for each consumed prefix is then
//! resolved PER PREFIX inside the per-layer builder via
//! [`PerLayerQuantization::quantization_for`] (the qwen3 idiom), so a per-layer
//! parameter override builds that layer with its own scheme, a per-layer `Skip`
//! builds it dense, and a per-layer-only config (no global default) loads its
//! listed layers. A `.scales`-bearing layer that resolves to no scheme for THAT
//! layer (a `Skip` / no override + no global default) is a typed
//! [`Error::InvariantViolation`] inside the per-layer builder (the weights say
//! quantized, the config resolved dense for this layer).
//!
//! [sv]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sensevoice/sensevoice.py

use std::{
  collections::HashMap,
  path::{Path, PathBuf},
};

use smol_str::format_smolstr;

use crate::{
  array::Array,
  audio::stt::model::Transcribe,
  error::{
    Error, FileIoPayload, FileOp, LayerKeyedPayload, LengthMismatchPayload, MalformedDataPayload,
    MissingKeyPayload, Result, UnknownEnumValuePayload,
  },
  lm::quant::PerLayerQuantization,
};

use super::{
  config::{Config, MODEL_TYPE},
  encoder::Encoder,
  frontend::{parse_am_mvn, sanitize},
  model::{SenseVoiceModel, build_head},
  tokenizer::{SPM_MODEL_FILE, SenseVoiceTokenizer, TOKENS_JSON_FILE},
};

/// The Kaldi MVN statistics filename the reference loads in `post_load_hook`
/// (`sensevoice.py:571`: `mvn_path = model_path / "am.mvn"`).
const AM_MVN_FILE: &str = "am.mvn";

// ───────────────────────────── from_weights ─────────────────────────────

impl SenseVoiceModel {
  /// Build a [`SenseVoiceModel`] from a parsed [`Config`], a **sanitized**
  /// weight map, the [`SenseVoiceTokenizer`], and the optional global CMVN
  /// statistics — the dense entry point, the quantization-aware
  /// [`SenseVoiceModel::from_weights_quantized`] with `quantization = None`.
  ///
  /// `cmvn` is the `(means, istd)` pair (the `am.mvn` `<AddShift>` / `<Rescale>`
  /// vectors or the in-config fallback), or `None` when the checkpoint ships no
  /// CMVN statistics — the reference loads the pair together
  /// (`sensevoice.py:573-579`), so it is passed as a single `Option`.
  ///
  /// # Errors
  /// The [`SenseVoiceModel::from_weights_quantized`] errors.
  pub fn from_weights(
    config: Config,
    weights: HashMap<String, Array>,
    tokenizer: SenseVoiceTokenizer,
    cmvn: Option<(Array, Array)>,
  ) -> Result<Self> {
    Self::from_weights_quantized(config, weights, tokenizer, cmvn, None)
  }

  /// Build a [`SenseVoiceModel`] from a parsed [`Config`], a **sanitized**
  /// weight map, the [`SenseVoiceTokenizer`], the optional global CMVN
  /// statistics, and the optional parsed quantization config — the
  /// quantization-aware analogue of [`SenseVoiceModel::from_weights`].
  ///
  /// The config is validated first ([`Config::validate`]), so a malformed
  /// `config.json` is rejected before any tensor is built. The [`Encoder`]
  /// tower ([`Encoder::from_weights`]) and the CTC head + query table
  /// ([`build_head`]) are then composed from the (drained) weight map.
  ///
  /// **Quantization** (the qwen3 `.scales` discriminator): `quantization` is
  /// threaded to the builders ONLY when [`has_relevant_scales`] finds a
  /// `<prefix>.scales` sibling for some layer the model loads (a one-pass
  /// pre-scan). A dense checkpoint (no `.scales`) loads through the unchanged
  /// dense path regardless of any stale `quantization` block. Every
  /// quantize-aware layer then picks quantized vs dense per-layer by its own
  /// `.scales` sibling ([`crate::nn::MaybeQuantizedLinear::from_weights`]) and
  /// resolves its `(group_size, bits, mode)` PER PREFIX via
  /// [`PerLayerQuantization::quantization_for`] (the qwen3 idiom); a `.scales`
  /// that resolves to no scheme for THAT layer is a typed
  /// [`Error::InvariantViolation`].
  ///
  /// `quantization` is the parsed
  /// [`crate::lm::quant::PerLayerQuantization`] (the `config.json`
  /// `quantization` block parsed by the shared audio resolver
  /// [`crate::audio::load::apply_quantization`]). SenseVoice typically quantizes
  /// its whole transformer with a single global scheme (mlx-community emits a
  /// global `{ group_size, bits, [mode] }` block with no per-layer override),
  /// but a per-layer parameter override is honored at its own prefix, a per-layer
  /// `Skip` builds that one layer dense, and a per-layer-only config (no global
  /// default) loads its explicitly-listed layers; a `<prefix>.scales` that
  /// resolves to no scheme for its layer (a `Skip` / no override + no global
  /// default) is rejected.
  ///
  /// # Errors
  /// - the [`Config::validate`] errors (non-positive / non-divisible dims, an
  ///   over-cap block count, a malformed in-config CMVN length);
  /// - the [`Encoder::from_weights`] / [`build_head`] errors (a missing weight,
  ///   a `.scales` that resolves to no scheme for its layer, an `embed` /
  ///   `ctc_lo` whose shape does not match the SenseVoice invariants);
  /// - [`Error::InvariantViolation`] if a `.scales` is present but
  ///   `quantization` resolved no scheme for that layer.
  pub fn from_weights_quantized(
    config: Config,
    mut weights: HashMap<String, Array>,
    tokenizer: SenseVoiceTokenizer,
    cmvn: Option<(Array, Array)>,
    quantization: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Single config-validation gate BEFORE any tensor is built.
    config.validate()?;

    // Thread the parsed quantization config to the builders ONLY when the
    // checkpoint actually carries a `.scales` sibling for some layer the model
    // loads (the `.scales`-presence discriminator, hoisted to the whole map). A
    // DENSE checkpoint loads through the unchanged dense path regardless of a
    // stale `quantization` block. The non-quant `validate` above always runs.
    //
    // The `(group_size, bits, mode)` is then resolved PER CONSUMED PREFIX inside
    // each builder via [`PerLayerQuantization::quantization_for`] (the qwen3
    // idiom), so a per-layer parameter override builds that layer with its OWN
    // scheme, a per-layer `Skip` builds it dense, and a per-layer-only config (no
    // global default) loads its listed layers — none of which a single collapsed
    // global tuple could express.
    let quant = if has_relevant_scales(&config, &weights) {
      quantization
    } else {
      None
    };

    let encoder = Encoder::from_weights(
      &mut weights,
      config.input_size(),
      config.encoder_conf(),
      quant,
    )?;
    let (ctc_lo, embed) = build_head(
      &mut weights,
      config.input_size(),
      encoder.output_size(),
      config.vocab_size(),
      quant,
    )?;

    let (cmvn_means, cmvn_istd) = match cmvn {
      Some((means, istd)) => (Some(means), Some(istd)),
      None => (None, None),
    };

    Ok(Self::new(
      config, encoder, ctc_lo, embed, tokenizer, cmvn_means, cmvn_istd,
    ))
  }
}

/// Probe whether the (sanitized) weight map carries a `<prefix>.scales` sibling
/// for any layer the model actually loads — the load-time half of the
/// `.scales`-presence discriminator the per-layer
/// [`crate::nn::MaybeQuantizedLinear::from_weights`] /
/// [`crate::nn::MaybeQuantizedEmbedding::from_weights`] use, hoisted to the
/// whole map (the qwen3 pre-scan pattern).
///
/// Probes the exact `<prefix>.scales` keys the matching builders would build,
/// with the same `<prefix>` format: the `ctc_lo` head, the 16-row `embed`
/// table, and every SANM block's `linear_q_k_v` / `linear_out` /
/// `feed_forward.w_1` / `feed_forward.w_2` across the three sub-stacks
/// (`encoders0`, `encoders.{0..num_blocks-2}`, `tp_encoders.{0..tp_blocks-1}`)
/// for the ACTUAL loaded block counts. A foreign key, an out-of-range block
/// index, or a never-quantized conv / norm `.scales` is correctly ignored —
/// exactly the keys no builder ever consults. The depthwise FSMN conv and the
/// LayerNorms are never quantized (mlx quantizes `nn.Linear` / `nn.Embedding`
/// only), so their prefixes are not probed. Reads only the map's keys (cheap
/// string lookups); no [`Array`] is touched.
///
/// `config` is the validated [`Config`]; `num_blocks` / `tp_blocks` are
/// non-negative (pinned by [`Config::validate`]).
pub fn has_relevant_scales(config: &Config, weights: &HashMap<String, Array>) -> bool {
  let has_scales = |prefix: &str| weights.contains_key(&format!("{prefix}.scales"));

  // The CTC head + the prompt-embedding table.
  if has_scales("ctc_lo") || has_scales("embed") {
    return true;
  }

  let enc = config.encoder_conf();
  // Every SANM block carries four quantize-aware linears.
  let block_has_scales = |prefix: &str| {
    has_scales(&format!("{prefix}.self_attn.linear_q_k_v"))
      || has_scales(&format!("{prefix}.self_attn.linear_out"))
      || has_scales(&format!("{prefix}.feed_forward.w_1"))
      || has_scales(&format!("{prefix}.feed_forward.w_2"))
  };

  // `encoders0` holds exactly the one width-changing first block.
  if block_has_scales("encoder.encoders0.0") {
    return true;
  }
  // `encoders`: `num_blocks - 1` constant-width blocks.
  let encoders = (enc.num_blocks() - 1).max(0);
  if (0..encoders).any(|i| block_has_scales(&format!("encoder.encoders.{i}"))) {
    return true;
  }
  // `tp_encoders`: `tp_blocks` blocks.
  let tp = enc.tp_blocks().max(0);
  (0..tp).any(|i| block_has_scales(&format!("encoder.tp_encoders.{i}")))
}

// ───────────────────────────── asset loading ─────────────────────────────

/// Load the global CMVN statistics for the model directory, mirroring the CMVN
/// half of `post_load_hook` (`sensevoice.py:571-579`).
///
/// Precedence (faithful to the reference):
/// 1. an `am.mvn` file in `dir` -> [`parse_am_mvn`] the `<AddShift>` means and
///    the `<Rescale>` inverse-stddev (`sensevoice.py:572-575`);
/// 2. otherwise, the in-`config.json` `cmvn_means` / `cmvn_istd` fallback
///    (`sensevoice.py:577-579`, `config.py:59-61`);
/// 3. otherwise `None` — the model runs without CMVN
///    ([`SenseVoiceModel::extract_features`] then skips it, the reference's
///    `if self._cmvn_means is not None` guard `sensevoice.py:392`).
///
/// The `am.mvn` body is read through the shared bounded
/// [`crate::lm::load::read_bounded_config_file`] (the 1 MiB config-read
/// convention, same class as `config.json` / `tokens.json`) so a hostile model
/// directory cannot OOM the loader by planting a huge stats file. Whichever
/// source supplies the pair, both vectors must be exactly
/// [`Config::input_size`]-wide — the CMVN stats are applied element-wise to the
/// `(T', input_size)` LFR features (`(feats + means) * istd`,
/// `sensevoice.py:80`), so a length-1 (or otherwise wrong-width) vector would
/// broadcast silently with the wrong statistics rather than fail.
///
/// The vectors are widened into 1-D `(D,)` [`Array`]s broadcast across the LFR
/// features by [`crate::audio::stt::models::sensevoice::frontend::apply_cmvn`].
///
/// # Errors
/// - [`Error::FileIo`] / a size-cap error from the bounded `am.mvn` read;
/// - [`Error::MalformedData`] if a present `am.mvn` is missing its `<AddShift>`
///   / `<Rescale>` block or carries a non-float token (via [`parse_am_mvn`]);
/// - [`Error::LengthMismatch`] if either CMVN vector's length is not
///   `config.input_size()`;
/// - propagates the [`Array::from_slice`] construction error.
fn load_cmvn(dir: &Path, config: &Config) -> Result<Option<(Array, Array)>> {
  let mvn_path = dir.join(AM_MVN_FILE);
  // Bounded, TOCTOU-closed read (the `tokens.json` / `config.json` convention):
  // `Ok(Some(text))` present, `Ok(None)` absent (fall through to the in-config
  // fallback), `Err` on oversized / non-regular / IO failure.
  if let Some(text) = crate::lm::load::read_bounded_config_file(&mvn_path, "sensevoice am.mvn")? {
    let (means, istd) = parse_am_mvn(&text)?;
    return Ok(Some(floats_to_arrays(&means, &istd, config.input_size())?));
  }

  // In-config fallback (`sensevoice.py:577-579`). `Config::validate` (run by
  // `from_weights_quantized` before this is consumed) already pins the pair to
  // be present together and `input_size`-wide; the length check below is a
  // self-contained re-assertion so this loader never builds a mis-sized CMVN.
  match (config.cmvn_means(), config.cmvn_istd()) {
    (Some(means), Some(istd)) => Ok(Some(floats_to_arrays(means, istd, config.input_size())?)),
    _ => Ok(None),
  }
}

/// Build the `(means, istd)` 1-D [`Array`] pair from the parsed float vectors,
/// after pinning both lengths to `input_size`.
///
/// # Errors
/// - [`Error::LengthMismatch`] if `means.len()` or `istd.len()` is not
///   `input_size`;
/// - [`Error::MalformedData`] if a length does not fit in `i32`;
/// - propagates the [`Array::from_slice`] construction error.
fn floats_to_arrays(means: &[f32], istd: &[f32], input_size: i32) -> Result<(Array, Array)> {
  // The CMVN stats broadcast element-wise over the `(T', input_size)` LFR
  // features, so each vector must be exactly `input_size` wide (`sensevoice.py:80`).
  let expected = input_size.max(0) as usize;
  if means.len() != expected {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "sensevoice load: CMVN means length vs input_size",
      expected,
      means.len(),
    )));
  }
  if istd.len() != expected {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "sensevoice load: CMVN istd length vs input_size",
      expected,
      istd.len(),
    )));
  }

  let m_len = i32::try_from(means.len()).map_err(|_| {
    Error::MalformedData(MalformedDataPayload::new(
      "sensevoice load: CMVN means length",
      "must fit in i32",
    ))
  })?;
  let i_len = i32::try_from(istd.len()).map_err(|_| {
    Error::MalformedData(MalformedDataPayload::new(
      "sensevoice load: CMVN istd length",
      "must fit in i32",
    ))
  })?;
  let means = Array::from_slice::<f32>(means, &[m_len])?;
  let istd = Array::from_slice::<f32>(istd, &[i_len])?;
  Ok((means, istd))
}

/// Load the detokenizer for the model directory, mirroring the tokenizer half
/// of `post_load_hook` (`sensevoice.py:581-597`).
///
/// Precedence (faithful to the reference):
/// 1. a `chn_jpn_yue_eng_ko_spectok.bpe.model` SentencePiece model in `dir` ->
///    [`SenseVoiceTokenizer::from_spm_file`] (`sensevoice.py:584-590`);
/// 2. otherwise, a `tokens.json` piece list -> [`SenseVoiceTokenizer::from_token_list`]
///    (`sensevoice.py:594-596`);
/// 3. otherwise the degenerate id-join detokenizer
///    ([`SenseVoiceTokenizer::id_join`], `sensevoice.py:448`).
///
/// Both assets are read through the shared bounded readers so a hostile
/// directory cannot OOM the loader: the SentencePiece `.model` through
/// [`crate::lm::load::read_bounded_bytes_file`] with the generous
/// [`crate::audio::stt::models::sensevoice::tokenizer::MAX_SPM_MODEL_BYTES`]
/// cap (a binary protobuf), and the `tokens.json` body through
/// [`crate::lm::load::read_bounded_config_file`] (the 1 MiB config-read
/// convention), then parsed as a `Vec<String>` of pieces.
///
/// # Errors
/// - [`Error::FileIo`] / size cap from the bounded `.model` / `tokens.json`
///   read;
/// - [`Error::Parse`] if a present `tokens.json` is not a JSON string list;
/// - propagates [`SenseVoiceTokenizer::from_spm_file`]'s read / parse errors.
fn load_tokenizer(dir: &Path) -> Result<SenseVoiceTokenizer> {
  // The bounded `.model` read is open-once + TOCTOU-closed, so it handles the
  // presence check itself (`Ok(None)` ⇒ absent ⇒ fall through to `tokens.json`).
  let spm_path = dir.join(SPM_MODEL_FILE);
  if let Some(spm) = SenseVoiceTokenizer::from_spm_file(&spm_path)? {
    return Ok(spm);
  }

  let tokens_path = dir.join(TOKENS_JSON_FILE);
  match crate::lm::load::read_bounded_config_file(&tokens_path, "sensevoice tokens.json")? {
    Some(body) => {
      let tokens: Vec<String> = serde_json::from_str(&body).map_err(|e| {
        Error::Parse(crate::error::ParsePayload::new(
          "sensevoice load: tokens.json",
          "JSON string list",
          e,
        ))
      })?;
      Ok(SenseVoiceTokenizer::from_token_list(tokens))
    }
    None => Ok(SenseVoiceTokenizer::id_join()),
  }
}

// ───────────────────────────── shard walk ─────────────────────────────

/// Read and merge every `*.safetensors` shard under `dir` into one weight map —
/// the same shard walk the whisper loader uses (`load_all_safetensors`,
/// `whisper/model.rs`), mirrored exactly (no hand-rolled glob): sort the
/// `.safetensors` entries by name and merge with
/// [`crate::model_validation::insert_unique`] so a cross-shard duplicate key
/// fails closed rather than letting the later-sorted shard silently overwrite
/// an earlier tensor.
///
/// The swift reference shard-loads every `*.safetensors` sorted by name and
/// merges (`SenseVoiceModel.swift:543-552`); the merged map is then
/// [`sanitize`]d (the `ctc.ctc_lo.` strip + the FSMN conv transpose) by the
/// caller.
///
/// # Errors
/// - [`Error::FileIo`] if `dir` cannot be read (an entry fails mid-walk);
/// - [`Error::MissingKey`] if `dir` holds no `*.safetensors`;
/// - [`Error::LayerKeyed`] (the offending shard file name) wrapping an
///   [`Error::KeyCollision`] (the duplicated tensor key) if two shards define
///   the same key;
/// - propagates [`crate::io::load_safetensors`] read errors.
fn load_all_safetensors(dir: &Path) -> Result<HashMap<String, Array>> {
  let entries = std::fs::read_dir(dir).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "sensevoice load: read model directory",
      FileOp::Read,
      dir.to_path_buf(),
      e,
    ))
  })?;
  let mut files: Vec<PathBuf> = entries
    .map(|entry| {
      entry.map(|e| e.path()).map_err(|e| {
        Error::FileIo(FileIoPayload::new(
          "sensevoice load: read model directory entry",
          FileOp::Read,
          dir.to_path_buf(),
          e,
        ))
      })
    })
    .collect::<Result<Vec<_>>>()?;
  files.retain(|p| p.extension().is_some_and(|ext| ext == "safetensors"));
  files.sort();
  if files.is_empty() {
    return Err(Error::MissingKey(MissingKeyPayload::new(
      "sensevoice load: no *.safetensors in model directory",
      format_smolstr!("{}", dir.display()),
    )));
  }

  let mut all = HashMap::new();
  for f in &files {
    let shard = crate::io::load_safetensors(f)?;
    for (key, value) in shard {
      crate::model_validation::insert_unique(
        &mut all,
        key,
        value,
        "sensevoice load: duplicate tensor key across shards",
      )
      .map_err(|e| match e {
        Error::KeyCollision(_) => {
          Error::LayerKeyed(LayerKeyedPayload::new(f.to_string_lossy().into_owned(), e))
        }
        other => other,
      })?;
    }
  }
  Ok(all)
}

// ───────────────────────────── load entry + factory ─────────────────────────────

impl SenseVoiceModel {
  /// Load a [`SenseVoiceModel`] from a local on-disk model directory — the
  /// convenience entry point mirroring mlx-audio's `stt.load` for this
  /// architecture, returning the concrete model (with its inherent rich-info
  /// API [`SenseVoiceModel::transcribe_rich`]).
  ///
  /// Pipeline (mirroring the shared loader + the reference `post_load_hook`):
  /// 1. [`crate::audio::load::get_model_path`] — resolve `path` to a local
  ///    directory (a Hub id is rejected per the no-network policy);
  /// 2. [`crate::audio::load::load_config`] — read + bound `config.json`,
  ///    parse the [`Config`];
  /// 3. `load_all_safetensors` — walk + merge the `*.safetensors` shards;
  /// 4. [`sanitize`] — the `ctc.ctc_lo.` strip + the FSMN conv transpose;
  /// 5. [`crate::audio::load::apply_quantization`] — parse the optional
  ///    `quantization` block, but ONLY when [`has_relevant_scales`] proves the
  ///    sanitized weights carry a `.scales` for some layer the model loads (the
  ///    qwen3 pre-scan gate), so a dense checkpoint with a stale / malformed
  ///    `quantization` block loads dense rather than failing at the parse;
  /// 6. `load_cmvn` — the `am.mvn` CMVN parse (or the in-config fallback);
  /// 7. `load_tokenizer` — the SentencePiece / `tokens.json` resolution;
  /// 8. [`SenseVoiceModel::from_weights_quantized`] — build the model.
  ///
  /// # Errors
  /// The errors of every pipeline step above (a missing directory / config,
  /// a malformed config, a missing or duplicated weight, a malformed `am.mvn`
  /// / `tokens.json`, a `.scales` with no resolvable scheme). A malformed
  /// `quantization` block is surfaced ONLY for an actually-quantized checkpoint
  /// (one carrying relevant `.scales`); a dense checkpoint ignores it.
  pub fn load(path: &str) -> Result<Self> {
    let dir = crate::audio::load::get_model_path(path)?;
    let config_json = crate::audio::load::load_config(&dir)?;
    let config: Config = serde_json::from_str(&config_json).map_err(|e| {
      Error::Parse(crate::error::ParsePayload::new(
        "sensevoice load: config.json",
        "JSON",
        e,
      ))
    })?;
    // Reject a malformed config before reading the (large) weight shards.
    config.validate()?;

    // Walk + merge the safetensors shards (the whisper / swift shard walk),
    // then sanitize (the `ctc.ctc_lo.` strip + the FSMN conv transpose).
    let raw = load_all_safetensors(&dir)?;
    let weights = sanitize(raw)?;

    // Parse the optional quantization block through the shared audio resolver
    // (top-level `quantization` or the HF `quantization_config`) ONLY when the
    // sanitized checkpoint actually carries a `.scales` for some layer the model
    // loads — the qwen3 `.scales`-presence pre-scan, applied at the parse so a
    // DENSE checkpoint with a stale / partial / malformed `quantization` block
    // loads dense instead of failing in `apply_quantization`. When no relevant
    // scale is present the block is irrelevant (nothing to interpret), so it is
    // never parsed.
    let quantization = if has_relevant_scales(&config, &weights) {
      crate::audio::load::apply_quantization(&config_json)?
    } else {
      None
    };

    // The `post_load_hook` assets (`sensevoice.py:567-598`).
    let cmvn = load_cmvn(&dir, &config)?;
    let tokenizer = load_tokenizer(&dir)?;

    Self::from_weights_quantized(config, weights, tokenizer, cmvn, quantization.as_ref())
  }
}

/// Load a SenseVoice-Small CTC model from a local on-disk directory, erasing the
/// model behind the universal [`Transcribe`] contract — the STT-factory entry
/// the registry dispatches `model_type == "sensevoice"` to.
///
/// Reads `config.json`, rejects a `model_type` other than [`MODEL_TYPE`]
/// (`"sensevoice"`, the reference `config.py:54` tag /
/// `MODEL_REMAPPING["sensevoice"]` key) with a typed
/// [`Error::UnknownEnumValue`], then builds the model via
/// [`SenseVoiceModel::load`] and boxes it. For the concrete (non-erased)
/// [`SenseVoiceModel`] — with its inherent [`SenseVoiceModel::transcribe_rich`]
/// rich-info API — call [`SenseVoiceModel::load`] directly.
///
/// `path` is the local on-disk path (a `hf://…` / `org/name` repo id is
/// rejected by [`crate::audio::load::get_model_path`] with a clear no-network
/// message).
///
/// # Errors
/// - the [`SenseVoiceModel::load`] errors;
/// - [`Error::UnknownEnumValue`] if `config.json`'s `model_type` is not
///   `"sensevoice"`.
#[cfg(feature = "sensevoice")]
#[cfg_attr(docsrs, doc(cfg(feature = "sensevoice")))]
pub fn load(path: &str) -> Result<Box<dyn Transcribe>> {
  let dir = crate::audio::load::get_model_path(path)?;
  let config_json = crate::audio::load::load_config(&dir)?;
  let config: Config = serde_json::from_str(&config_json).map_err(|e| {
    Error::Parse(crate::error::ParsePayload::new(
      "sensevoice load: config.json",
      "JSON",
      e,
    ))
  })?;
  let model_type = config.model_type();
  if model_type != MODEL_TYPE {
    return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
      "sensevoice load: model_type",
      model_type.to_string(),
      SUPPORTED_MODEL_TYPES,
    )));
  }
  Ok(Box::new(SenseVoiceModel::load(path)?))
}

/// The `model_type` value this loader accepts (the single FunAudioLLM
/// SenseVoice id, `config.py:54` / `MODEL_REMAPPING["sensevoice"]`).
const SUPPORTED_MODEL_TYPES: &[&str] = &[MODEL_TYPE];

#[cfg(test)]
mod tests;
