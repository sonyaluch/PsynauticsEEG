"""Shared constants for the EEG preprocessing

Flagged assumptions  marked ASSUMPTION
"""

# --- Channel mapping -------------------------------------------------------
# ASSUMPTION: Channel order ch1->ch4 is mapped to the standard Muse electrode order used by
# essentially all Muse tooling (muse-lsl, BlueMuse, Mind Monitor exports):
# TP9, AF7, AF8, TP10.
RAW_TO_MUSE_CHANNEL = {
    "ch1": "TP9",
    "ch2": "AF7",
    "ch3": "AF8",
    "ch4": "TP10",
}
MUSE_CHANNEL_ORDER = ["TP9", "AF7", "AF8", "TP10"]
# ch5/ch6 are unused aux columns -- optional (some export variants omit them
# entirely), and ignored when present and empty. A file where they're present
# WITH data is excluded rather than processed (see io.py) since that means
# the export isn't the plain 4-channel EEG format this pipeline expects.
EMPTY_RAW_COLUMNS = ["ch5", "ch6"]

# Channels used by the Psynautics markers (for reference downstream):
#   Cognition: AF8 frontal spectral power
#   Emotion:   AF7 vs AF8 alpha asymmetry
#   Awareness: all 4 channels (LZC / permutation entropy)

# --- Sampling ---------------------------------------------------------------
# Muse S Gen 2 native EEG sampling rate (hardware spec). The raw CSV
# timestamps are irregular (bursty BLE packets + dropouts), so we resample
# onto a uniform grid at this nominal rate rather than trusting instantaneous
# inter-sample deltas.
NOMINAL_SFREQ = 256.0

# ASSUMPTION: raw CSV channel values are already in microvolts (see io.py for
# the reasoning). MNE's internal EEG convention is volts, so this converts
# uV -> V at load time. Set to 1e-6 if confirmed uV; change if a calibration
# formula for raw ADC counts is confirmed instead.
RAW_UNITS_TO_VOLTS = 1e-6

# Any gap between consecutive raw samples longer than this is treated as a
# real dropout (BLE disconnect / buffer stall), not sampling jitter. The
# region is linearly interpolated for filter continuity but annotated
# BAD_dropout so downstream epoching/feature extraction excludes it.
#
# EMPIRICAL BASIS (from the first sample recording): the Muse delivers data
# in BLE packet bursts, not a true continuous 256 Hz stream -- within a
# packet, samples are timestamped with near-zero deltas (<1ms), while the
# gap *between* packets clusters tightly around ~46ms (median), with p99.9
# at ~76ms and nothing observed between ~190ms and several seconds. That gap
# is the natural separator between "normal packet cadence" and "real
# dropout": in the sample file, only 3 gaps exceeded 200ms (12.6s, 4.8s,
# 4.8s = 6.9% of the recording), which is a plausible dropout rate for
# unattended home BLE streaming; a naive threshold near the nominal sample
# period (e.g. a few ms) instead flags nearly the entire recording as bad.
# Revisit this value if a different Muse export/firmware shows a different
# packet cadence.
MAX_JITTER_S = 0.2

# --- Filtering ---------------------------------------------------------------
BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 40.0
# US recruitment per the protocol (mains hum at 60 Hz). Only applied if it
# falls below the bandpass upper edge / Nyquist, otherwise it's a no-op
# handled defensively in preprocess.py.
NOTCH_FREQS_HZ = [60.0]

# ASSUMPTION: a channel is "globally bad" for the WHOLE recording (not just
# a transient window) if its overall MAD is far higher than its sibling
# channels'. EMPIRICAL BASIS: known-clean recordings show ~2x spread across
# per-channel MAD; recordings with a visually-confirmed bad electrode (poor
# contact for the whole session) show 12-140x. 5x sits clear of both and is
# how these get caught -- the per-window artifact check below normalizes
# each channel against its OWN whole-recording baseline, so it's blind to a
# channel that's uniformly noisy throughout (no clean baseline to compare
# against within that channel).
GLOBAL_BAD_CHANNEL_MAD_RATIO = 5.0

# ASSUMPTION: a window is flagged BAD_EMG if the ratio of high-frequency
# (20-40Hz) to low-frequency (1-10Hz) mean power exceeds this threshold.
# Cortical EEG has a roughly 1/f spectrum (low >> high); EMG has a much
# flatter, broadband spectrum, so a channel/window with disproportionate
# high-frequency power relative to low is a signature of muscle artifact
# bleeding in, not brain signal. EMPIRICAL BASIS (2026-08-30): sampled
# ~174k channel-windows across 60 recordings -- the ratio distribution has
# no clean gap (10th pct 0.017, median 0.059, 90th pct 0.163, 99th pct
# 0.323), consistent with the bad-channel-MAD-ratio threshold (also no
# clean gap; confirmed by eye instead). What DOES separate clean from
# suspect recordings is the *rate* of high-ratio windows: a known-clean
# reference recording has ~0.3% of windows above 0.3, vs 1.8-3.0% in
# recordings with a visually-confirmed spectral bump (see the CAVEAT above
# COGNITION_CHANNEL). 0.3 sits at roughly the 99th percentile overall while
# concentrating on the already-suspect files, rather than flagging a large
# fraction of the whole cohort the way a lower threshold would.
EMG_LOW_BAND_HZ = (1.0, 10.0)
EMG_HIGH_BAND_HZ = (20.0, 40.0)
EMG_RATIO_THRESH = 0.3

# --- Artifact rejection ------------------------------------------------------
# Robust (MAD-based) z-score threshold for flagging short windows as
# artifact, applied per channel on top of dropout annotations. Chosen to be
# conservative given raw units are not confirmed to be calibrated uV.
ARTIFACT_WINDOW_S = 1.0
ARTIFACT_MAD_ZSCORE_THRESH = 5.0

# --- Marker extraction -------------------------------------------------------
# ASSUMPTION: none of these formulas/parameters are confirmed against a
# Psynautics protocol spec -- they're standard choices from the EEG
# literature, chosen so the markers are well-defined and reproducible while
# a real spec is tracked down. Revisit each if a documented spec turns up.

# Fixed-length, non-overlapping epochs for windowed spectral/entropy
# estimation. 4s is a common resting-state EEG epoch length -- long enough
# for stable band-power and entropy estimates, short enough that a 5-10min
# baseline recording still yields ~75-150 epochs. Epochs overlapping any
# BAD_* annotation are dropped entirely (via MNE's reject_by_annotation).
MARKER_EPOCH_S = 4.0

FREQ_BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# CAVEAT: beta-band power at frontal/dry-electrode sites (AF8 here) is a
# well-documented target for frontalis-muscle EMG contamination, which is
# NOT distinguished from cortical beta by this pipeline -- the per-window
# and global-bad-channel checks catch amplitude outliers, not EMG's
# broadband spectral signature. Investigated 2026-08-30: a spectral bump in
# the ~15-20Hz range appears in several "clean" (non-excluded) recordings.
# Ruled out as a resampling/interpolation artifact (synthetic white noise
# resampled through the same recording's real irregular timestamps shows no
# comparable bump), and its frequency is inconsistent across recordings
# (not a fixed hardware/electronic artifact) -- both point toward genuine
# EMG rather than a pipeline bug. Treat Cognition as a mix of cortical beta
# and muscle artifact until a dedicated EMG check is added.
#
# ASSUMPTION: beta-band (13-30Hz) mean PSD (uV^2/Hz, not integrated band
# power) at AF8 as the "Cognition" scalar --
# beta is the standard EEG correlate of active cognitive engagement /
# concentration in consumer-EEG literature (cf. Muse's own attention
# scoring, Pope et al. 1995 engagement index components). Frontal theta is
# the other common candidate (cognitive-load paradigms); swap COGNITION_BAND
# if the protocol turns out to mean that instead.
COGNITION_CHANNEL = "AF8"
COGNITION_BAND = "beta"

# ASSUMPTION: frontal alpha asymmetry (Davidson's approach-withdrawal
# paradigm), computed as ln(right alpha power) - ln(left alpha power) =
# ln(AF8) - ln(AF7). Alpha is an "idling" rhythm (less alpha = more cortical
# activity), so a positive value means relatively greater LEFT frontal
# activation, associated with approach-oriented / positive affect in this
# paradigm; negative = withdrawal / negative affect.
EMOTION_LEFT_CHANNEL = "AF7"
EMOTION_RIGHT_CHANNEL = "AF8"
EMOTION_BAND = "alpha"

# ASSUMPTION: "Awareness" = mean of normalized Lempel-Ziv complexity (LZC)
# and normalized permutation entropy (PE), averaged across all 4 channels
# and all epochs. Both are standard complexity/entropy measures used as EEG
# consciousness/awareness proxies in the anesthesia-depth and meditation
# literature. LZC needs a binary sequence -- each epoch is binarized via a
# per-epoch median split (x >= median(x)), the standard approach for
# continuous-valued signals per antropy's own documentation.
PERM_ENTROPY_ORDER = 3
PERM_ENTROPY_DELAY = 1

# --- Filename metadata parsing ----------------------------------------------
# e.g. 2026_Feb_files_eeg_<email>_<date>_<time>_<utc_offset>_<hash>_eeg.csv
FILENAME_PATTERN = (
    r".*_eeg_(?P<email>[^_]+@[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<time>[\d-]+)_(?P<utc_offset>[\d-]+)_(?P<session_hash>[a-f0-9]+)_eeg\.csv$"
)
