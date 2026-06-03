# CLAP mel front-end oracle fixtures

These fixtures pin the CLAP mel / spectrogram front-end
(`src/embeddings/clap/mel.rs`) numerically, **before** any checkpoint. They are
copied from the Findit-AI `textclap` crate
(`textclap/tests/fixtures/`), which generates them from the real HF
`laion/clap-htsat-unfused` model via `transformers.ClapFeatureExtractor` +
`librosa` (`textclap/tests/fixtures/regen_golden.py`).

- `filterbank_row_{0,10,32}.npy` — `<f4`, shape `(513,)`. Rows of the librosa
  Slaney-scale, Slaney-normalized mel filterbank
  (`librosa.filters.mel(sr=48000, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
  htk=False, norm="slaney")`). Row 10 lands near the 1 kHz Slaney inflection (a
  discriminator vs an HTK build). The oracle asserts the mlxrs
  `mel_filter_bank_scaled(Slaney, slaney_norm=true)` rows match these.
- `golden_mel.npy` — `<f4`, shape `(1001, 64)`. The full log-mel spectrogram the
  HF `ClapFeatureExtractor` produces for `sample.wav` (time-major
  `(T=1001, mel=64)`). The oracle asserts the mlxrs mel front-end reproduces it.
- `sample_audio.npy` — `<f4`, shape `(240000,)`. The decoded 48 kHz mono f32
  waveform of `textclap/tests/fixtures/sample.wav` (a 5.0 s dog-bark clip,
  `Xenova/transformers.js-docs/dog_barking.wav` resampled to 48 kHz s16le mono),
  decoded the exact `textclap` way (`i16 / 32768.0`). Committed as `.npy` so the
  oracle compares against `golden_mel.npy` on byte-identical input, independent
  of any WAV-decoder differences.

To regenerate `golden_mel.npy` / the filterbank rows, see
`textclap/tests/fixtures/regen_golden.py`. `sample_audio.npy` is the i16-PCM
`data` chunk of `sample.wav` scaled by `1/32768` into little-endian f32.
