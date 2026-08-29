"""
Filtering and artifact annotation for loaded Muse Raw objects.

With only 4 channels, ICA-based artifact removal and spherical-spline bad
channel interpolation are unreliable (too few spatial degrees of freedom),
so this pipeline deliberately avoids both. Instead it:
  - band-pass + notch filters the continuous signal
  - flags short bad windows per-channel using a robust (MAD-based) z-score,
    which doesn't require assuming a calibrated physical unit
  - leaves rejection to annotation (BAD_*), so downstream epoching can use
    `reject_by_annotation=True` rather than discarding whole channels
"""
from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np

from . import config


@dataclass
class PreprocessReport:
    pct_annotated_bad: float
    n_artifact_windows: int
    n_dropout_windows: int
    per_channel_flagged_windows: dict[str, int]


def _flag_artifact_windows(raw: mne.io.RawArray) -> tuple[list[tuple[float, float, str]], dict[str, int]]:
    """Per-channel MAD z-score over fixed windows; flag a window if ANY channel exceeds threshold.

    The median/MAD reference is computed once per channel over the WHOLE
    recording, not per-window. A per-window reference would self-normalize
    against a sustained artifact that dominates its own window (e.g. a
    multi-second amplitude burst), hiding exactly the kind of artifact this
    check is meant to catch.

    Channels already marked bad (raw.info["bads"]) are skipped entirely --
    a known-bad channel (e.g. a saturated electrode) shouldn't drag every
    window into BAD_artifact when the other channels are clean.
    """
    sfreq = raw.info["sfreq"]
    win_samples = int(round(config.ARTIFACT_WINDOW_S * sfreq))
    data = raw.get_data()  # (n_channels, n_samples)
    n_channels, n_samples = data.shape
    n_windows = n_samples // win_samples

    good_chs = [ch for ch in raw.ch_names if ch not in raw.info["bads"]]
    per_channel_flags = {ch: 0 for ch in good_chs}
    annotations: list[tuple[float, float, str]] = []

    channel_medians = np.median(data, axis=1)
    channel_mads = np.median(np.abs(data - channel_medians[:, None]), axis=1)

    for w in range(n_windows):
        start = w * win_samples
        stop = start + win_samples
        window_bad = False
        for ci, ch in enumerate(raw.ch_names):
            if ch not in good_chs:
                continue
            seg = data[ci, start:stop]
            mad = channel_mads[ci]
            if mad == 0:
                continue
            z = (seg - channel_medians[ci]) / (1.4826 * mad)
            if np.max(np.abs(z)) > config.ARTIFACT_MAD_ZSCORE_THRESH:
                per_channel_flags[ch] += 1
                window_bad = True
        if window_bad:
            onset = start / sfreq
            duration = win_samples / sfreq
            annotations.append((onset, duration, "BAD_artifact"))

    return annotations, per_channel_flags


def preprocess_raw(raw: mne.io.RawArray) -> tuple[mne.io.RawArray, PreprocessReport]:
    """Band-pass + notch filter, then annotate artifact windows on top of any
    dropout annotations already present from loading.

    Returns a new Raw (input is not modified) plus a QC report.
    """
    raw = raw.copy()
    existing_annotations = raw.annotations
    n_dropout = int(np.sum(existing_annotations.description == "BAD_dropout"))

    nyquist = raw.info["sfreq"] / 2.0
    notch = [f for f in config.NOTCH_FREQS_HZ if f < min(nyquist, config.BANDPASS_HIGH_HZ)]
    if notch:
        raw.notch_filter(notch, verbose=False)

    raw.filter(
        l_freq=config.BANDPASS_LOW_HZ,
        h_freq=config.BANDPASS_HIGH_HZ,
        verbose=False,
    )

    artifact_annots, per_channel_flags = _flag_artifact_windows(raw)
    if artifact_annots:
        onsets, durations, descriptions = zip(*artifact_annots)
        new_annots = mne.Annotations(onset=list(onsets), duration=list(durations), description=list(descriptions))
        raw.set_annotations(existing_annotations + new_annots)

    total_bad_s = sum(
        d for d in raw.annotations.duration
    )
    # Overlapping annotations could double-count; good enough for a QC estimate,
    # exact accounting would require merging intervals.
    pct_bad = 100.0 * total_bad_s / raw.times[-1] if raw.times[-1] > 0 else 0.0

    report = PreprocessReport(
        pct_annotated_bad=pct_bad,
        n_artifact_windows=len(artifact_annots),
        n_dropout_windows=n_dropout,
        per_channel_flagged_windows=per_channel_flags,
    )
    return raw, report
