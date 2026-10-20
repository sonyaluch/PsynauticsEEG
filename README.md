# PsynauticsEEG

EEG analysis for the Psynautics at-home study.

- Preprocessing pipeline for resting-state EEG collected with Muse S (Gen 2) headbands
- At-home study testing participants at baseline (Days 1-5), treatment (Day 6), and post-treatment (Days 7-11).
- 4 channels (TP9, AF7, AF8, TP10).

Three derived markers are extracted per recording in the analysis,
**Cognition** / "Performance" (AF8 broadband power, 1-44Hz),
**Emotion** / "Mood" (AF7 alpha-band power, 7.5-13Hz), and
**Awareness** (mean normalized permutation entropy, across all 4 channels).

## What it does

The Muse sends its EEG over Bluetooth in 12-sample packets, and the export stamps every row with the packet's
*arrival* time -- so rows inside a packet are ~0.05 ms apart and arrival times jitter by tens of ms. These stamps are
**not** the samples' true times. The samples themselves are a continuous 256 Hz stream.

The loader (`src/muse_pipeline/io.py`):

1. Treats rows, in file order, as consecutive 256 Hz samples (it deliberately does not sort or resample on the timestamps).
2. Uses arrival times only to size a hole where a gap exceeds 200 ms (samples were lost). Holes and sensor NaNs are filled by
   linear interpolation so the filters see a continuous signal, and runs of 0.2 s or more are annotated `BAD_dropout`.

(An earlier version resampled on the arrival times, which distorted the spectrum -- low frequencies inflated ~2x, 30-44 Hz cut ~4x,
the 60 Hz mains line nearly erased. `tests/test_loader_timebase.py` now guards against this with a synthetic signal of known frequencies.)

The preprocessor (`src/muse_pipeline/preprocess.py`) then:

1. Band-pass filters 1-45 Hz and notches the Bluetooth packet-rate interference (21.33 Hz and its 42.67 Hz harmonic).
   The 60 Hz mains line is already outside the passband.
2. Flags 1 s windows per channel as `BAD_artifact` using a robust MAD-based z-score computed over the whole recording.
3. Flags 1 s windows as `BAD_EMG` where 20-40 Hz power is a large outlier for that channel in that recording (robust z > 5 on log power).

Only 4 channels means that ICA and spherical-spline interpolation are unreliable (too few spatial degrees of freedom)
Data is never discarded, but instead annotated as `BAD_*` or `raw.info["bads"]` flags.
These tags are excluded in marker extraction.

Any channel that is uniformly bad for the *whole* recording (e.g. persistently poor electrode contact) is auto-detected and added to `raw.info["bads"]`: its filtered signal's MAD exceeds 30 µV (about 4x the typical channel).

Marker extraction (`src/muse_pipeline/markers.py`) then epochs the cleaned signal into fixed 4s windows (excluding any epoch overlapping a `BAD_*` annotation), estimates the spectrum with Welch's method (1 s segments, 50% overlap, explicit because MNE's default for epochs is multitaper, which is not scaled to µV²/Hz), and computes, per recording:

- **Cognition** ("Performance"): mean broadband (delta through gamma, 1-44Hz) power spectral density at AF8 (µV²/Hz, not integrated band power). Corrected from an earlier beta-band-only (13-30Hz) definition per an update from the study.
- **Emotion** ("Mood"): mean alpha-band (7.5-13Hz) power spectral density at AF7 (µV²/Hz). Corrected from an earlier AF7-vs-AF8 alpha asymmetry formula per an update from the study -- now single-channel, so only needs AF7 clean (not both AF7 and AF8).
- **Awareness**: mean normalized permutation entropy, across all 4 channels. Corrected from an earlier formula that also averaged in normalized Lempel-Ziv complexity (LZC), which is no longer computed.

A marker whose required channel(s) are in `raw.info["bads"]` is reported as unavailable.

## Layout

```
src/muse_pipeline/
  config.py      shared constants -- channel mapping, thresholds, unit conversion, marker params
  io.py          CSV loading, resampling, dropout detection
  preprocess.py  filtering, artifact-window flagging, global bad-channel detection
  markers.py     Cognition / Emotion / Awareness extraction
  manifest.py    cross-references the Psynautics export manifest + participant key
  qc.py          PSD / annotated-trace QC plots
scripts/
  run_preprocess.py     single-file preprocessing CLI
  batch_preprocess.py   batch preprocessing CLI across a directory tree, manifest-aware
  extract_markers.py    batch marker extraction over preprocessed .fif files
  make_qc_figures.py    cohort-level QC summary figures from batch_qc_summary.csv
```

`data/`, `output/`, and `tools/` (gcloud CLI + credentials) are gitignored --
raw and derived EEG data and any credentials never belong in this repo. See
[Data access](#data-access) below.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Usage

Single file:

```bash
./.venv/bin/python scripts/run_preprocess.py <path/to/*_eeg.csv> --outdir output
```

Saves a cleaned `.fif`, before/after PSD plot, annotated-trace plot, and a
JSON QC report.

**Whenever new readings land** (new participants, or existing participants'
later sessions), the whole pipeline can be re-run in one command:

```bash
./scripts/run_full_pipeline.sh              # pulls new/changed files from GCS first
./scripts/run_full_pipeline.sh --skip-sync  # skip the GCS pull, use data/raw/ as-is
```

Always rebuilds from the full current cohort in `data/raw/`.

If you want more control:

Batch, across a directory of raw exports, cross-referenced against the export manifest for participant/module metadata:

```bash
./.venv/bin/python scripts/batch_preprocess.py \
  --input-dir data/raw \
  --manifest /path/to/eeg_summary.csv \
  --participant-key /path/to/participant_key_CONFIDENTIAL.csv \
  --outdir output/batch_full \
  --eeg-only
```

`--eeg-only` restricts to the 11 resting-state EEG modules (baseline/treatment/post) and skips sleep, questionnaire, and clinic-info modules that also appear in the manifest. A single bad file logs an error and is skipped-- it doesn't abort the run. The `clean_raw/` output directory is cleared atthe start of each run so a file excluded on a later run can't leave a stale`.fif` behind. Output: `clean_raw/*.fif` per recording plus one aggregate `batch_qc_summary.csv` (duration, dropout %, artifact %, bad channels, pass/fail) across the whole batch.

`batch_preprocess.py` also hardcodes one small manual override discovered during cohort QC, documented inline at the top of the script: `EXCLUDED_SESSION_HASHES`. These are whole files excluded outright (ie. one recording ran 20+ hours with 99% dropout, indicating a device left on and unattended).

Then extract markers from the preprocessed files:

```bash
./.venv/bin/python scripts/extract_markers.py \
  --input-dir output/batch_full/clean_raw \
  --outdir output/batch_full
```

Output: one `marker_summary.csv` row per recording (Cognition/Emotion/Awareness, or blank where a required channel was unavailable).

## Data access

Raw exports live in a private GCS bucket (`psynautics-files`), pulled with:

```bash
source gcloud_env.sh   # points gcloud at a project-local, non-sudo config
gcloud storage cp -r gs://psynautics-files/2026_Feb_files/eeg/* data/raw/
```

The export manifest (`eeg_summary.csv`) and `participant_key_CONFIDENTIAL.csv`
(the email -> pseudonymous subject-code mapping) are kept outside github entirely, so never commit them.

## Known assumptions and limits

- Raw CSV values are in microvolts (baseline ~800 with hard rails at 0 and ~1650, matching the Muse's documented 0-1682.8 µV range).
- `ch1`-`ch4` map to `TP9, AF7, AF8, TP10` in that order (the standard Muse electrode order).
- `ch5`/`ch6` are unused aux columns, optional, and ignored when empty; a file where they contain real data doesn't match the expected
  4-channel EEG-only export and is excluded.
- All three marker definitions are CONFIRMED per an update from the study: Performance/Cognition = AF8 broadband 1-44 Hz,
  Mood/Emotion = AF7 alpha 7.5-13 Hz, Awareness = permutation entropy over all four channels. The Lempel-Ziv value is an extra
  companion measure, not part of the Awareness definition. Values are mean PSD (µV²/Hz), not integrated band power.
- The 5 MAD thresholds and the 30 µV bad-channel cutoff are judgement calls: the corrected data show no natural gap to place them in.
  See the comments in `config.py` for the evidence behind each.
- Performance and Mood come from frontal dry electrodes and may include facial/jaw muscle activity, which 4 channels cannot fully separate
  from brain activity.
- The app's own 0-10 scores (Conor's `lambda_function.py` and "Data Transform EEG score.xlsx", Oct 2026) are NOT reproduced here. The app
  works from Muse's pre-computed band-power columns, which our raw exports don't contain: Mood = mean `Alpha_AF7`, Performance = mean of the
  five AF8 band values, Awareness = permutation entropy of the 20 band-power time series; scores are Mood (x-1.7)*-10, Performance (x-1)*-10
  (higher score = lower power) and Awareness x100 in the script but x10 in the spreadsheet. These differ from our confirmed definitions, and
  the spreadsheet is not Gerard's data, so nothing could be validated. Reports therefore use our own units.
- The recordings' protocol-step labels come from the app and are used as-is; some participants have repeated, missing or out-of-order steps.

## Tests

```bash
./.venv/bin/python tests/test_loader_timebase.py   # known sinusoids through a Muse-like packetised stream
./.venv/bin/python tests/test_filter_markers.py    # filter gain, PSD units, epoch rejection, entropy
```
