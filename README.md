# PsynauticsEEG

EEG analysis for the Psynautics at-home study.

Preprocessing pipeline for resting-state EEG collected with Muse S (Gen 2)
headbands as part of the Psynautics protocol: an 11-day citizen-neuroscience
study with baseline (Days 1-5), treatment (Day 6), and post-treatment (Days
7-11) recordings, 4 channels (TP9, AF7, AF8, TP10).

Three derived markers are extracted per recording -- **Cognition** (AF8 beta-
band power), **Emotion** (AF7/AF8 alpha asymmetry), and **Awareness** (mean
of normalized Lempel-Ziv complexity and permutation entropy, across all 4
channels). This repo covers the full path from raw CSV export to those three
scalars: loading, cleaning, QC-flagging, and marker extraction.

## What it does

Muse streams over BLE in irregular packet bursts, not a fixed-rate stream, so
naively trusting the sample index (rather than the true per-sample
timestamps) silently misrepresents timing after any dropout. The loader
(`src/muse_pipeline/io.py`):

1. Reads true per-sample timestamps from the CSV export.
2. Resamples onto a uniform grid at the nominal 256 Hz via linear
   interpolation (MNE requires evenly-sampled data).
3. Flags any inter-sample gap over 200ms as a real dropout (vs. normal
   ~46ms BLE packet cadence) and annotates it `BAD_dropout`, so interpolated
   samples inside the gap are excluded from any annotation-aware downstream
   step rather than treated as real signal.

The preprocessor (`src/muse_pipeline/preprocess.py`) then:

1. Band-pass filters (1-40 Hz) and notch-filters (60 Hz mains).
2. Flags short artifact windows per-channel using a robust MAD-based
   z-score computed once over the whole recording (not per-window, which
   would let a sustained artifact hide by normalizing against itself).

With only 4 channels, ICA and spherical-spline interpolation are unreliable
(too few spatial degrees of freedom), so bad data is never discarded --
everything is left as `BAD_*` annotations or `raw.info["bads"]` channel
flags, and it's up to downstream analysis to decide what to exclude.

Additionally, a channel that's uniformly bad for the *whole* recording (e.g.
persistently poor electrode contact) is auto-detected and added to
`raw.info["bads"]`: its overall MAD is compared against its sibling
channels', since the per-window check above only catches anomalies relative
to a channel's own baseline and is blind to a channel that's bad throughout.

Marker extraction (`src/muse_pipeline/markers.py`) then epochs the cleaned
signal into fixed 4s windows (dropping any epoch overlapping a `BAD_*`
annotation) and computes, per recording:

- **Cognition**: mean beta-band (13-30Hz) power spectral density at AF8 (µV²/Hz, not integrated band power).
- **Emotion**: `ln(AF8 alpha power) - ln(AF7 alpha power)` (frontal alpha
  asymmetry).
- **Awareness**: mean of normalized Lempel-Ziv complexity and normalized
  permutation entropy, across all 4 channels.

A marker whose required channel(s) are in `raw.info["bads"]` is reported as
unavailable rather than computed on incomplete data.

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

Always rebuilds from the full current cohort in `data/raw/`, not just the
new files, so results stay consistent. The steps it runs -- individually,
if you want more control:

Batch, across a directory of raw exports, cross-referenced against the
export manifest for participant/module metadata:

```bash
./.venv/bin/python scripts/batch_preprocess.py \
  --input-dir data/raw \
  --manifest /path/to/eeg_summary.csv \
  --participant-key /path/to/participant_key_CONFIDENTIAL.csv \
  --outdir output/batch_full \
  --eeg-only
```

`--eeg-only` restricts to the 11 resting-state EEG modules (baseline/
treatment/post) and skips sleep, questionnaire, and clinic-info modules that
also appear in the manifest. A single bad file logs an error and is skipped
-- it doesn't abort the run. The `clean_raw/` output directory is cleared at
the start of each run so a file excluded on a later run can't leave a stale
`.fif` behind. Output: `clean_raw/*.fif` per recording plus one aggregate
`batch_qc_summary.csv` (duration, dropout %, artifact %, bad channels,
pass/fail) across the whole batch.

`batch_preprocess.py` also hardcodes one small manual override discovered
during cohort QC, documented inline at the top of the script:
`EXCLUDED_SESSION_HASHES` -- whole files excluded outright (e.g. a
"5-minute baseline" recording that actually ran 20+ hours with 99% dropout --
a device left on unattended, not real resting-state data). Per-channel bad
flags are no longer a manual list; they're auto-detected (see above).

Then extract markers from the preprocessed files:

```bash
./.venv/bin/python scripts/extract_markers.py \
  --input-dir output/batch_full/clean_raw \
  --outdir output/batch_full
```

Output: one `marker_summary.csv` row per recording (Cognition/Emotion/
Awareness, or blank where a required channel was unavailable).

## Data access

Raw exports live in a private GCS bucket (`psynautics-files`), pulled with:

```bash
source gcloud_env.sh   # points gcloud at a project-local, non-sudo config
gcloud storage cp -r gs://psynautics-files/2026_Feb_files/eeg/* data/raw/
```

The export manifest (`eeg_summary.csv`) and `participant_key_CONFIDENTIAL.csv`
(the email -> pseudonymous subject-code mapping) are kept outside this repo
entirely -- never commit them.

## Known assumptions

Flagged inline in `config.py` as `ASSUMPTION`, since they're inferred from
the data rather than confirmed against Muse/Psynautics documentation:

- Raw CSV channel values are already in microvolts (not raw ADC counts) --
  baseline/std magnitudes are consistent with uV-scale EEG.
- `ch1`-`ch4` map to `TP9, AF7, AF8, TP10` in that order (the standard Muse
  electrode order used by muse-lsl, BlueMuse, Mind Monitor).
- `ch5`/`ch6` are unused aux columns, optional, and ignored when empty; a
  file where they contain real data doesn't match the expected 4-channel
  EEG-only export and is excluded rather than guessed at.
- A channel is "globally bad" for a whole recording if its overall MAD is
  more than 5x its sibling channels' -- chosen empirically and confirmed by
  eye against annotated-trace plots across the threshold range, not against
  a physiological calibration.
- The marker formulas themselves (Cognition = AF8 beta PSD, Emotion =
  frontal alpha asymmetry, Awareness = LZC + permutation entropy) are
  standard EEG-literature choices, not confirmed Psynautics protocol specs
  -- see the `ASSUMPTION` comments in `config.py` for the reasoning and
  citations behind each.
- Cognition (beta-band PSD at AF8, a frontal dry-electrode site) is likely
  a mix of cortical beta and frontalis-muscle EMG artifact -- confirmed not
  to be a resampling/interpolation bug (see the CAVEAT comment above
  `COGNITION_CHANNEL` in `config.py`), but not currently distinguished from
  genuine brain signal by any check in this pipeline.
