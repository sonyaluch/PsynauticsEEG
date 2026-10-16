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
# 45Hz, not 40Hz: Cognition (below) is defined as broadband power up to
# 44Hz, so the passband needs to clear 44Hz with a little margin -- a
# filter's transition band isn't a brick wall, content right at the nominal
# edge is partially attenuated. 45Hz still sits comfortably below the 60Hz
# mains-hum frequency, so the notch-filter no-op logic below is unaffected.
BANDPASS_HIGH_HZ = 45.0
# US recruitment per the protocol (mains hum at 60 Hz). Only applied if it
# falls below the bandpass upper edge / Nyquist, otherwise it's a no-op
# handled defensively in preprocess.py.
NOTCH_FREQS_HZ = [60.0]

# The Muse sends 12-sample packets at 256 Hz, i.e. 21.33 packets/s. Some
# channels in some recordings carry a narrow interference spike at exactly
# that packet rate and its harmonic (found in the RAW samples, up to ~14x the
# neighbouring noise floor; a 17 Hz control frequency shows none), and both
# fall inside the 1-44 Hz Cognition window. Removed with narrow notches.
PACKET_RATE_NOTCH_HZ = [NOMINAL_SFREQ / 12, 2 * NOMINAL_SFREQ / 12]
PACKET_RATE_NOTCH_WIDTH_HZ = 1.0

# A channel is "globally bad" for the WHOLE recording (poor electrode contact
# throughout) if the MAD of its filtered signal exceeds this many microvolts.
# Absolute rather than relative-to-sibling-channels because (a) the per-window
# artifact check normalises each channel against its own baseline and so is
# blind to a channel that is uniformly noisy, and (b) a sibling-relative rule
# both misses cases where several channels are bad at once and mis-flags a
# normal channel sitting next to an unusually quiet one (both seen in review).
#
# EMPIRICAL BASIS (corrected signal, 952 channel-recordings): the median
# channel has MAD ~7 uV (IQR 4.7-12); there is no natural gap -- the tail runs
# continuously from ~11 to ~580 uV -- so the cutoff is a judgement call. 30 uV
# (~4x the median; SD-equivalent ~44 uV) flags ~12% of channels and agrees with
# the earlier sibling-ratio rule on ~100 of ~118 channels. Valid because the
# export is in real microvolts (baseline ~800, hard rails at 0 and ~1650,
# matching the Muse's documented 0-1682.8 uV range).
GLOBAL_BAD_CHANNEL_MAD_UV = 30.0

# A 1 s window is flagged BAD_EMG when its high-frequency (20-40Hz) power is
# far above that channel's OWN typical level in the same recording: robust
# z-score of log10 HF power (median / MAD over the whole recording) above the
# threshold below. Muscle (EMG) activity shows up as bursts of broadband HF
# power; flagging relative to the recording's own baseline avoids needing any
# absolute spectral shape.
#
# HISTORY: the first version flagged windows where the HF/LF power ratio
# exceeded 0.3, calibrated on spectra that were distorted by a loader bug
# (arrival-time resampling, see io.py) which suppressed HF power ~4x. On the
# correctly-timed signal the Muse spectrum is nearly flat above 10Hz (dry-
# electrode noise floor), so the typical ratio is ~0.3 and that rule flagged
# nearly everything -- an absolute-ratio rule cannot work here.
#
# EMPIRICAL BASIS (corrected signal): at 5 MAD the median channel-recording has
# ~1% of windows flagged, the 90th percentile ~9%; 5 MAD of log10 HF power is
# ~15x the recording's typical HF power.
EMG_HIGH_BAND_HZ = (20.0, 40.0)
EMG_HF_ZSCORE_THRESH = 5.0

# --- Artifact rejection ------------------------------------------------------
# Robust (MAD-based) z-score threshold for flagging short windows as
# artifact, applied per channel on top of dropout annotations. Chosen to be
# conservative given raw units are not confirmed to be calibrated uV.
ARTIFACT_WINDOW_S = 1.0
ARTIFACT_MAD_ZSCORE_THRESH = 5.0

# --- Marker extraction -------------------------------------------------------
# All three marker definitions (Cognition/Performance, Emotion/Mood,
# Awareness) are now CONFIRMED per an update from the study -- see each
# marker's own comment below for the specific definition and correction
# history. Epoching parameters (below) remain a pipeline design choice, not
# part of the confirmed spec.

# Fixed-length, non-overlapping epochs for windowed spectral/entropy
# estimation. 4s is a common resting-state EEG epoch length -- long enough
# for stable band-power and entropy estimates, short enough that a 5-10min
# baseline recording still yields ~75-150 epochs. Epochs overlapping any
# BAD_* annotation are dropped entirely (via MNE's reject_by_annotation).
MARKER_EPOCH_S = 4.0

# PSD estimation: Welch, 1 s segments (1 Hz resolution at 256 Hz) with 50%
# overlap, averaged within each 4 s epoch. Stated explicitly because MNE's
# Epochs.compute_psd() DEFAULTS TO MULTITAPER, whose default normalisation is
# not divided by the sampling rate -- it returned values 256x larger than true
# spectral density (uV^2/Hz), found by a white-noise unit test.
PSD_N_FFT = 256

FREQ_BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 44.0),
    "broadband": (1.0, 44.0),  # delta through gamma
    "mood_alpha": (7.5, 13.0),  # Emotion/Mood's alpha band -- confirmed distinct from the general "alpha" above
}

# CAVEAT: broadband power at a frontal dry-electrode site (AF8) can include
# facial/jaw muscle activity (EMG) and other noise that is not cortical. The
# pipeline removes dropouts, amplitude artifacts, whole-recording bad
# channels, high-frequency bursts (BAD_EMG) and BLE packet-rate interference,
# but cannot separate residual muscle activity from brain activity with 4
# channels. NOTE: an earlier version of this note described a "15-20Hz
# spectral bump" as muscle artifact. That bump was produced by a loader bug
# (samples were resampled on packet-ARRIVAL times, see io.py) and does not
# exist in the correctly-timed signal -- that conclusion is withdrawn.
#
# CONFIRMED (per an update from the study): Cognition is AF8 broadband
# power (delta through gamma, 1-44Hz), not beta-band only -- corrected from
# the earlier ASSUMPTION-tagged beta-band-only definition. Mean PSD
# (uV^2/Hz), not integrated band power -- see _band_power_uv2() in
# markers.py.
COGNITION_CHANNEL = "AF8"
COGNITION_BAND = "broadband"

# CONFIRMED (per an update from the study): Emotion/"Mood" is single-channel
# AF7 alpha-band (7.5-13Hz) mean PSD (uV^2/Hz) -- corrected from the earlier
# ASSUMPTION-tagged AF7-vs-AF8 asymmetry formula (ln(AF8) - ln(AF7)), which
# is no longer used.
EMOTION_CHANNEL = "AF7"
EMOTION_BAND = "mood_alpha"

# CONFIRMED (per an update from the study): Awareness is normalized
# permutation entropy (PE) alone, averaged across all 4 channels and all
# epochs -- corrected from the earlier ASSUMPTION-tagged formula, which
# averaged PE with normalized Lempel-Ziv complexity (LZC). LZC is no longer
# computed.
PERM_ENTROPY_ORDER = 3
PERM_ENTROPY_DELAY = 1

# --- Filename metadata parsing ----------------------------------------------
# e.g. 2026_Feb_files_eeg_<email>_<date>_<time>_<utc_offset>_<hash>_eeg.csv
FILENAME_PATTERN = (
    r".*_eeg_(?P<email>[^_]+@[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<time>[\d-]+)_(?P<utc_offset>[\d-]+)_(?P<session_hash>[a-f0-9]+)_eeg\.csv$"
)
