//! Whisper tokenizer wrapper — [`HFTokenizerWrapper`] (`whisper.py:36-235`)
//! plus the [`LANGUAGES`] table + the [`to_language_code`] name→code lookup
//! (`tokenizer.py`).
//!
//! Whisper uses a GPT-2 byte-level BPE vocabulary extended with special
//! tokens (`<|startoftranscript|>`, `<|transcribe|>`, the per-language
//! `<|en|>` …, and the `<|0.00|>` … timestamp tokens). The reference wraps a
//! HuggingFace `WhisperTokenizer` and resolves every special token id by
//! **string lookup** (`convert_tokens_to_ids("<|…|>")`) rather than hard-
//! coding ids, because the multilingual / English-only vocabularies place
//! them at different offsets. This port mirrors that exactly on top of the
//! crate's [`crate::tokenizer::Tokenizer`].

use std::fmt;

use crate::{Error, Result, error::MissingKeyPayload, tokenizer::Tokenizer};

/// The Whisper language codes → English names, in the reference's order
/// (`tokenizer.py:3-104`) — 100 entries (large-v3 added Cantonese as the
/// 100th; the `num_languages = 99` default is the large-v2 slice count). The
/// order is load-bearing: `num_languages` slices the first N entries to
/// enumerate the language tokens for a given checkpoint.
pub static LANGUAGES: &[(&str, &str)] = &[
  ("en", "english"),
  ("zh", "chinese"),
  ("de", "german"),
  ("es", "spanish"),
  ("ru", "russian"),
  ("ko", "korean"),
  ("fr", "french"),
  ("ja", "japanese"),
  ("pt", "portuguese"),
  ("tr", "turkish"),
  ("pl", "polish"),
  ("ca", "catalan"),
  ("nl", "dutch"),
  ("ar", "arabic"),
  ("sv", "swedish"),
  ("it", "italian"),
  ("id", "indonesian"),
  ("hi", "hindi"),
  ("fi", "finnish"),
  ("vi", "vietnamese"),
  ("he", "hebrew"),
  ("uk", "ukrainian"),
  ("el", "greek"),
  ("ms", "malay"),
  ("cs", "czech"),
  ("ro", "romanian"),
  ("da", "danish"),
  ("hu", "hungarian"),
  ("ta", "tamil"),
  ("no", "norwegian"),
  ("th", "thai"),
  ("ur", "urdu"),
  ("hr", "croatian"),
  ("bg", "bulgarian"),
  ("lt", "lithuanian"),
  ("la", "latin"),
  ("mi", "maori"),
  ("ml", "malayalam"),
  ("cy", "welsh"),
  ("sk", "slovak"),
  ("te", "telugu"),
  ("fa", "persian"),
  ("lv", "latvian"),
  ("bn", "bengali"),
  ("sr", "serbian"),
  ("az", "azerbaijani"),
  ("sl", "slovenian"),
  ("kn", "kannada"),
  ("et", "estonian"),
  ("mk", "macedonian"),
  ("br", "breton"),
  ("eu", "basque"),
  ("is", "icelandic"),
  ("hy", "armenian"),
  ("ne", "nepali"),
  ("mn", "mongolian"),
  ("bs", "bosnian"),
  ("kk", "kazakh"),
  ("sq", "albanian"),
  ("sw", "swahili"),
  ("gl", "galician"),
  ("mr", "marathi"),
  ("pa", "punjabi"),
  ("si", "sinhala"),
  ("km", "khmer"),
  ("sn", "shona"),
  ("yo", "yoruba"),
  ("so", "somali"),
  ("af", "afrikaans"),
  ("oc", "occitan"),
  ("ka", "georgian"),
  ("be", "belarusian"),
  ("tg", "tajik"),
  ("sd", "sindhi"),
  ("gu", "gujarati"),
  ("am", "amharic"),
  ("yi", "yiddish"),
  ("lo", "lao"),
  ("uz", "uzbek"),
  ("fo", "faroese"),
  ("ht", "haitian creole"),
  ("ps", "pashto"),
  ("tk", "turkmen"),
  ("nn", "nynorsk"),
  ("mt", "maltese"),
  ("sa", "sanskrit"),
  ("lb", "luxembourgish"),
  ("my", "myanmar"),
  ("bo", "tibetan"),
  ("tl", "tagalog"),
  ("mg", "malagasy"),
  ("as", "assamese"),
  ("tt", "tatar"),
  ("haw", "hawaiian"),
  ("ln", "lingala"),
  ("ha", "hausa"),
  ("ba", "bashkir"),
  ("jw", "javanese"),
  ("su", "sundanese"),
  ("yue", "cantonese"),
];

/// Extra language-name aliases beyond the inverse of [`LANGUAGES`]
/// (`tokenizer.py:109-120`). [`to_language_code`] checks the inverse of
/// `LANGUAGES` first, then these.
static LANGUAGE_ALIASES: &[(&str, &str)] = &[
  ("burmese", "my"),
  ("valencian", "ca"),
  ("flemish", "nl"),
  ("haitian", "ht"),
  ("letzeburgesch", "lb"),
  ("pushto", "ps"),
  ("panjabi", "pa"),
  ("moldavian", "ro"),
  ("moldovan", "ro"),
  ("sinhalese", "si"),
  ("castilian", "es"),
  ("mandarin", "zh"),
];

/// Normalize a language name or code to its two/three-letter code — the
/// reference's `TO_LANGUAGE_CODE` dict lookup (`tokenizer.py:107-121`).
///
/// Returns:
/// - the input unchanged if it is already a known code (a key of
///   [`LANGUAGES`]);
/// - the mapped code if the input is a full English language name (the
///   inverse of [`LANGUAGES`]) or one of the extra `LANGUAGE_ALIASES`;
/// - `None` if the input is neither (the reference leaves an unknown value
///   as-is; the caller surfaces the resulting unknown-token error).
pub fn to_language_code(name_or_code: &str) -> Option<&'static str> {
  // Already a code?
  if let Some((code, _)) = LANGUAGES.iter().find(|(c, _)| *c == name_or_code) {
    return Some(code);
  }
  // A full English name (inverse of LANGUAGES)?
  if let Some((code, _)) = LANGUAGES.iter().find(|(_, n)| *n == name_or_code) {
    return Some(code);
  }
  // An alias?
  LANGUAGE_ALIASES
    .iter()
    .find(|(alias, _)| *alias == name_or_code)
    .map(|(_, code)| *code)
}

/// A thin Whisper-decoding view over the crate's [`Tokenizer`] —
/// `HFTokenizerWrapper` (`whisper.py:36-235`).
///
/// Borrows an existing [`Tokenizer`] and resolves the Whisper special-token
/// ids eagerly at construction (by `convert_token_to_id` string lookup), so
/// the decoding loop reads them as plain accessors. The `multilingual` flag
/// and normalized `language` / `task` selection drive [`Self::sot_sequence`].
///
/// [`Debug`] is implemented manually (the borrowed
/// [`Tokenizer`] is not `Debug`): only the resolved scalar state is printed.
pub struct HFTokenizerWrapper<'a> {
  tokenizer: &'a Tokenizer,
  multilingual: bool,
  num_languages: usize,
  /// Normalized language code (e.g. `"en"`).
  language: String,
  /// `"transcribe"` or `"translate"`.
  task: String,
  // Resolved special-token ids (string lookup at construction).
  sot: u32,
  eot: u32,
  transcribe: u32,
  translate: u32,
  sot_prev: u32,
  sot_lm: u32,
  no_timestamps: u32,
  no_speech: u32,
  timestamp_begin: u32,
}

impl fmt::Debug for HFTokenizerWrapper<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("HFTokenizerWrapper")
      .field("multilingual", &self.multilingual)
      .field("num_languages", &self.num_languages)
      .field("language", &self.language)
      .field("task", &self.task)
      .field("sot", &self.sot)
      .field("eot", &self.eot)
      .field("transcribe", &self.transcribe)
      .field("translate", &self.translate)
      .field("sot_prev", &self.sot_prev)
      .field("sot_lm", &self.sot_lm)
      .field("no_timestamps", &self.no_timestamps)
      .field("no_speech", &self.no_speech)
      .field("timestamp_begin", &self.timestamp_begin)
      .finish()
  }
}

/// The decoding task — `"transcribe"` (default) or `"translate"`
/// (`whisper.py:48`, `:143-147`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
  /// Transcribe speech in the source language.
  #[default]
  Transcribe,
  /// Translate speech into English.
  Translate,
}

impl Task {
  /// The reference task string (`"transcribe"` / `"translate"`).
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Transcribe => "transcribe",
      Self::Translate => "translate",
    }
  }
}

impl<'a> HFTokenizerWrapper<'a> {
  /// The end-of-transcript / end-of-text token string
  /// (`tokenizer.eos_token`).
  const EOT: &'static str = "<|endoftext|>";
  /// The start-of-transcript token string.
  const SOT: &'static str = "<|startoftranscript|>";
  /// The transcribe-task token string.
  const TRANSCRIBE: &'static str = "<|transcribe|>";
  /// The translate-task token string.
  const TRANSLATE: &'static str = "<|translate|>";
  /// The start-of-previous(-context) token string.
  const SOT_PREV: &'static str = "<|startofprev|>";
  /// The start-of-language-model token string.
  const SOT_LM: &'static str = "<|startoflm|>";
  /// The suppress-timestamps token string.
  const NO_TIMESTAMPS: &'static str = "<|notimestamps|>";
  /// The no-speech token string.
  const NO_SPEECH: &'static str = "<|nospeech|>";
  /// The first timestamp token string (`0.00` seconds).
  const TIMESTAMP_BEGIN: &'static str = "<|0.00|>";

  /// Wrap `tokenizer` for Whisper decoding (`whisper.py:42-58`).
  ///
  /// `language` accepts either a code (`"en"`) or an English name
  /// (`"english"`); it is normalized via [`to_language_code`] (falling back
  /// to `"en"` when `None` or unrecognized). `num_languages` is the
  /// checkpoint's language-token count (`ModelDimensions::num_languages`).
  ///
  /// Every special token is resolved by string lookup; a tokenizer missing
  /// one of them (not a Whisper vocabulary) is a typed
  /// [`Error::MissingKey`] rather than a later silent `unk`.
  ///
  /// # Errors
  /// [`Error::MissingKey`] if any required Whisper special token is absent
  /// from the tokenizer vocabulary.
  pub fn new(
    tokenizer: &'a Tokenizer,
    multilingual: bool,
    num_languages: usize,
    language: Option<&str>,
    task: Task,
  ) -> Result<Self> {
    let language = language
      .and_then(to_language_code)
      .unwrap_or("en")
      .to_string();

    let resolve = |tok: &'static str| -> Result<u32> {
      tokenizer.convert_token_to_id(tok).ok_or_else(|| {
        Error::MissingKey(MissingKeyPayload::new(
          "HFTokenizerWrapper: Whisper special token not in vocabulary",
          tok,
        ))
      })
    };

    // `eot` is the tokenizer's own eos id (`whisper.py:80-83`), falling back
    // to the `<|endoftext|>` string lookup when the tokenizer-config did not
    // record one.
    let eot = match tokenizer.eos_token_id() {
      Some(id) => id,
      None => resolve(Self::EOT)?,
    };

    Ok(Self {
      tokenizer,
      multilingual,
      num_languages,
      language,
      task: task.as_str().to_string(),
      sot: resolve(Self::SOT)?,
      eot,
      transcribe: resolve(Self::TRANSCRIBE)?,
      translate: resolve(Self::TRANSLATE)?,
      sot_prev: resolve(Self::SOT_PREV)?,
      sot_lm: resolve(Self::SOT_LM)?,
      no_timestamps: resolve(Self::NO_TIMESTAMPS)?,
      no_speech: resolve(Self::NO_SPEECH)?,
      timestamp_begin: resolve(Self::TIMESTAMP_BEGIN)?,
    })
  }

  /// Rebuild this wrapper for a different `language`, reusing the borrowed
  /// tokenizer + the `multilingual` / `num_languages` / `task` selection —
  /// the reference's `get_tokenizer(language=…, task=…)` re-resolution after
  /// language detection (`whisper.py:907-913`).
  ///
  /// `language` is normalized via [`to_language_code`] (falling back to `"en"`
  /// when unrecognized); the [`Self::sot_sequence`]'s language token reflects
  /// the new code. The special-token ids are unchanged (same vocabulary), so
  /// this only recomputes the normalized [`Self::language`].
  pub fn with_language(&self, language: &str) -> Self {
    Self {
      tokenizer: self.tokenizer,
      multilingual: self.multilingual,
      num_languages: self.num_languages,
      language: to_language_code(language).unwrap_or("en").to_string(),
      task: self.task.clone(),
      sot: self.sot,
      eot: self.eot,
      transcribe: self.transcribe,
      translate: self.translate,
      sot_prev: self.sot_prev,
      sot_lm: self.sot_lm,
      no_timestamps: self.no_timestamps,
      no_speech: self.no_speech,
      timestamp_begin: self.timestamp_begin,
    }
  }

  /// Encode text to token ids, **without** special tokens
  /// (`whisper.py:60-62`: `encode(text, add_special_tokens=False)`).
  ///
  /// # Errors
  /// Propagates the underlying tokenizer encode error.
  pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
    self.tokenizer.encode(text, false)
  }

  /// Decode token ids to text, dropping any timestamp tokens
  /// (`>= timestamp_begin`) first — the reference's `decode`
  /// (`whisper.py:64-72`).
  ///
  /// # Errors
  /// Propagates the underlying tokenizer decode error.
  pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
    let filtered: Vec<u32> = tokens
      .iter()
      .copied()
      .filter(|&t| t < self.timestamp_begin)
      .collect();
    self.tokenizer.decode(&filtered, skip_special_tokens)
  }

  /// Decode token ids to text **including** timestamp tokens
  /// (`whisper.py:74-78`).
  ///
  /// # Errors
  /// Propagates the underlying tokenizer decode error.
  pub fn decode_with_timestamps(&self, tokens: &[u32]) -> Result<String> {
    self.tokenizer.decode(tokens, false)
  }

  /// The start-of-transcript sequence (`whisper.py:131-135`):
  /// `(sot, language_token, task_token)` for a multilingual model, else
  /// `(sot,)`.
  pub fn sot_sequence(&self) -> Vec<u32> {
    if self.multilingual {
      vec![self.sot, self.language_token(), self.task_token()]
    } else {
      vec![self.sot]
    }
  }

  /// The start-of-transcript sequence with the `no_timestamps` token
  /// appended (`whisper.py:137-140`).
  pub fn sot_sequence_including_notimestamps(&self) -> Vec<u32> {
    let mut seq = self.sot_sequence();
    seq.push(self.no_timestamps);
    seq
  }

  /// The language token id for the configured language
  /// (`whisper.py:125-128`: `<|{language}|>`). Returns the tokenizer's
  /// `unk` id (or `0` if none) when the configured language has no token —
  /// matching the reference's `convert_tokens_to_ids` fallback.
  pub fn language_token(&self) -> u32 {
    let tok = format!("<|{}|>", self.language);
    self
      .tokenizer
      .convert_token_to_id(&tok)
      .or_else(|| self.tokenizer.unk_token_id())
      .unwrap_or(0)
  }

  /// The task token id — [`Self::translate`] when the task is `"translate"`,
  /// else [`Self::transcribe`] (`whisper.py:142-147`).
  pub fn task_token(&self) -> u32 {
    if self.task == "translate" {
      self.translate
    } else {
      self.transcribe
    }
  }

  /// All language token ids for this checkpoint (the first
  /// [`Self::num_languages`] entries of [`LANGUAGES`]), skipping any that
  /// resolve to the tokenizer's `unk` id (`whisper.py:149-157`).
  pub fn all_language_tokens(&self) -> Vec<u32> {
    let unk = self.tokenizer.unk_token_id();
    LANGUAGES
      .iter()
      .take(self.num_languages)
      .filter_map(|(code, _)| {
        let id = self.tokenizer.convert_token_to_id(&format!("<|{code}|>"))?;
        if Some(id) == unk { None } else { Some(id) }
      })
      .collect()
  }

  /// The language codes for this checkpoint — the first
  /// [`Self::num_languages`] codes of [`LANGUAGES`] (`whisper.py:159-162`).
  pub fn all_language_codes(&self) -> Vec<&'static str> {
    LANGUAGES
      .iter()
      .take(self.num_languages)
      .map(|(code, _)| *code)
      .collect()
  }

  /// The `(code, token_id)` pairs for this checkpoint's language tokens — the
  /// first [`Self::num_languages`] entries of [`LANGUAGES`], skipping any whose
  /// `<|code|>` token is absent or resolves to the tokenizer's `unk` id.
  ///
  /// This is the aligned counterpart to [`Self::all_language_tokens`] +
  /// [`Self::all_language_codes`]: those two build their lists in *separate*
  /// passes (the token list filters out missing / `unk` languages while the
  /// code list does not), so positionally zipping them drifts — a missing
  /// earlier language shifts every later code onto the wrong token. Building the
  /// code and its id together in one pass makes that misalignment unrepresentable,
  /// so language-id consumers ([`super::decoding::detect_language`]) pair over
  /// this instead of zipping the two separate vectors.
  pub fn all_language_candidates(&self) -> Vec<(&'static str, u32)> {
    let unk = self.tokenizer.unk_token_id();
    LANGUAGES
      .iter()
      .take(self.num_languages)
      .filter_map(|(code, _)| {
        let id = self.tokenizer.convert_token_to_id(&format!("<|{code}|>"))?;
        if Some(id) == unk {
          None
        } else {
          Some((*code, id))
        }
      })
      .collect()
  }

  /// The non-speech token set to suppress to avoid non-speech annotations —
  /// `non_speech_tokens` (`whisper.py:165-182`).
  ///
  /// Builds the set by encoding a fixed list of punctuation / symbol strings
  /// (and their space-prefixed forms): a single-token encoding is added, and
  /// every "miscellaneous" musical glyph is added regardless of token count.
  /// `" -"` and `" '"` always contribute their first token. The set is sorted
  /// and de-duplicated. An encoding that yields no tokens contributes nothing.
  ///
  /// # Errors
  /// Propagates the underlying tokenizer encode error.
  pub fn non_speech_tokens(&self) -> Result<Vec<u32>> {
    use std::collections::BTreeSet;

    // The punctuation / symbol symbols (`whisper.py:169`).
    const SYMBOL_CHARS: &str = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』";
    // The multi-character symbol groups (`whisper.py:170-172`).
    const SYMBOL_GROUPS: &[&str] = &[
      "<<",
      ">>",
      "<<<",
      ">>>",
      "--",
      "---",
      "-(",
      "-[",
      "('",
      "(\"",
      "((",
      "))",
      "(((",
      ")))",
      "[[",
      "]]",
      "{{",
      "}}",
      "♪♪",
      "♪♪♪",
    ];
    // The "miscellaneous" musical glyphs, always added whatever the token count
    // (`whisper.py:174`).
    const MISCELLANEOUS: &str = "♩♪♫♬♭♮♯";

    let mut result: BTreeSet<u32> = BTreeSet::new();

    // `{self.encode(" -")[0], self.encode(" '")[0]}` (`whisper.py:176`).
    if let Some(&first) = self.encode(" -")?.first() {
      result.insert(first);
    }
    if let Some(&first) = self.encode(" '")?.first() {
      result.insert(first);
    }

    // `for symbol in symbols + list(miscellaneous)` (`whisper.py:177`).
    let symbols = SYMBOL_CHARS
      .chars()
      .map(|c| c.to_string())
      .chain(SYMBOL_GROUPS.iter().map(|s| (*s).to_string()))
      .chain(MISCELLANEOUS.chars().map(|c| c.to_string()));
    let misc_set: BTreeSet<char> = MISCELLANEOUS.chars().collect();
    for symbol in symbols {
      let is_misc =
        symbol.chars().count() == 1 && symbol.chars().next().is_some_and(|c| misc_set.contains(&c));
      // `[self.encode(symbol), self.encode(" " + symbol)]` (`whisper.py:178`).
      for text in [symbol.clone(), format!(" {symbol}")] {
        let tokens = self.encode(&text)?;
        // `if len(tokens) == 1 or symbol in miscellaneous: result.add(tokens[0])`
        // (`whisper.py:179-180`).
        if let Some(&first) = tokens.first()
          && (tokens.len() == 1 || is_misc)
        {
          result.insert(first);
        }
      }
    }

    Ok(result.into_iter().collect())
  }

  /// The normalized language code (e.g. `"en"`).
  #[inline(always)]
  pub fn language(&self) -> &str {
    &self.language
  }

  /// Whether the wrapped checkpoint is multilingual.
  #[inline(always)]
  pub fn is_multilingual(&self) -> bool {
    self.multilingual
  }

  /// The checkpoint's language-token count.
  #[inline(always)]
  pub fn num_languages(&self) -> usize {
    self.num_languages
  }

  /// `<|startoftranscript|>`.
  #[inline(always)]
  pub fn sot(&self) -> u32 {
    self.sot
  }

  /// End of transcript / end of text (the tokenizer eos id).
  #[inline(always)]
  pub fn eot(&self) -> u32 {
    self.eot
  }

  /// `<|transcribe|>`.
  #[inline(always)]
  pub fn transcribe(&self) -> u32 {
    self.transcribe
  }

  /// `<|translate|>`.
  #[inline(always)]
  pub fn translate(&self) -> u32 {
    self.translate
  }

  /// `<|startofprev|>`.
  #[inline(always)]
  pub fn sot_prev(&self) -> u32 {
    self.sot_prev
  }

  /// `<|startoflm|>` (`whisper.py:95-98`).
  #[inline(always)]
  pub fn sot_lm(&self) -> u32 {
    self.sot_lm
  }

  /// `<|notimestamps|>`.
  #[inline(always)]
  pub fn no_timestamps(&self) -> u32 {
    self.no_timestamps
  }

  /// `<|nospeech|>`.
  #[inline(always)]
  pub fn no_speech(&self) -> u32 {
    self.no_speech
  }

  /// `<|0.00|>` — the first timestamp token id. Tokens `>= timestamp_begin`
  /// are timestamp tokens.
  #[inline(always)]
  pub fn timestamp_begin(&self) -> u32 {
    self.timestamp_begin
  }
}

#[cfg(test)]
mod tests;
