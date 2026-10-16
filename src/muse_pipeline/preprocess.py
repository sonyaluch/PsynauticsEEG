"""
Filtering and artifact annotation for loaded Muse Raw objects.

With only 4 channels, ICA-based artifact removal and spherical-spline bad
channel interpolation are unreliable (too few spatial degrees of freedom),
so this pipeline deliberately avoids both. Instead it:
  - band-pass + notch filters the continuous signal
  - flags short bad windows per-channel using a robust (MAD-based) z-score,
    which doesn't require assuming a calibrated physical unit
  - flags short windows contaminated by muscle (EMG) artifact via spectral
    shape (high/low frequency power ratio), which the amplitude-based check
    above can miss -- EMG doesn't always produce an outlier amplitude spike
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
    n_emg_windows: int
    per_channel_flagged_windows: dict[str, int]
    per_channel_emg_flagged_windows: dict[str, int]
    globally_bad_channels: list[str]


def _detect_globally_bad_channels(raw: mne.io.RawArray) -> list[str]:
    """Flag a channel bad for the WHOLE recording if the MAD of its filtered
    signal exceeds GLOBAL_BAD_CHANNEL_MAD_UV microvolts (see config.py for the
    empirical basis). This catches persistently poor electrode contact, which
    the per-window artifact check below cannot: that check normalises each
    channel against its own whole-recording baseline, so a channel that is
    uniformly noisy throughout has no clean baseline within itself to look
    anomalous against.
    """
    data = raw.get_data() * 1e6  # volts -> uV
    medians = np.median(data, axis=1)
    mads = np.median(np.abs(data - medians[:, None]), axis=1)
    return [ch for ch, m in zip(raw.ch_names, mads) if m > config.GLOBAL_BAD_CHANNEL_MAD_UV]


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


def _flag_emg_windows(raw: mne.io.RawArray) -> tuple[list[tuple[float, float, str]], dict[str, int]]:
    """Flag a window BAD_EMG when its 20-40Hz power is a large outlier for that
    channel within this recording (robust z-score of log10 HF power > EMG_HF_ZSCORE_THRESH).

    Muscle activity appears as bursts of broadband high-frequency power. The
    reference is the channel's own median/MAD over the whole recording, so no
    absolute spectral shape is assumed (the Muse spectrum is nearly flat above
    10Hz, which defeats a fixed HF/LF ratio -- see config.py). Complements the
    amplitude-based check above, which can miss HF-only bursts.

    Channels already in raw.info["bads"] are skipped, same as the artifact check.
    """
    sfreq = raw.info["sfreq"]
    win_samples = int(round(config.ARTIFACT_WINDOW_S * sfreq))
    data = raw.get_data()
    n_windows = data.shape[1] // win_samples

    freqs = np.fft.rfftfreq(win_samples, d=1.0 / sfreq)
    hi = (freqs >= config.EMG_HIGH_BAND_HZ[0]) & (freqs <= config.EMG_HIGH_BAND_HZ[1])
    window_fn = np.hanning(win_samples)

    good_chs = [ch for ch in raw.ch_names if ch not in raw.info["bads"]]
    per_channel_flags = {ch: 0 for ch in good_chs}
    flagged = np.zeros(n_windows, dtype=bool)
    if n_windows:
        segs = data[:, : n_windows * win_samples].reshape(len(raw.ch_names), n_windows, win_samples) * window_fn
        hf_power = (np.abs(np.fft.rfft(segs, axis=-1)) ** 2)[..., hi].mean(axis=-1)
        for ci, ch in enumerate(raw.ch_names):
            if ch not in good_chs:
                continue
            log_p = np.log10(np.maximum(hf_power[ci], 1e-30))
            med = np.median(log_p)
            mad = 1.4826 * np.median(np.abs(log_p - med))
            if mad == 0:
                continue
            bad = (log_p - med) / mad > config.EMG_HF_ZSCORE_THRESH
            per_channel_flags[ch] = int(bad.sum())
            flagged |= bad

    annotations = [(w * win_samples / sfreq, win_samples / sfreq, "BAD_EMG") for w in np.where(flagged)[0]]
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
    # BLE packet-rate interference (21.33 Hz + harmonic), see config.py
    raw.notch_filter(config.PACKET_RATE_NOTCH_HZ, notch_widths=config.PACKET_RATE_NOTCH_WIDTH_HZ, verbose=False)

    # Defense-in-depth: a NaN anywhere in the signal at this point would
    # silently poison every downstream computation that touches it (band
    # power, entropy, ...) without ever showing up as a BAD_* annotation.
    # io.py already strips NaN source samples before interpolation for
    # exactly this reason -- this is a loud backstop in case some other
    # export variant introduces NaN by a path that check doesn't cover.
    if np.isnan(raw.get_data()).any():
        raise ValueError("NaN present in signal before filtering -- check the loader for an unhandled data gap.")

    raw.filter(
        l_freq=config.BANDPASS_LOW_HZ,
        h_freq=config.BANDPASS_HIGH_HZ,
        verbose=False,
    )

    globally_bad = _detect_globally_bad_channels(raw)
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(globally_bad))

    artifact_annots, per_channel_flags = _flag_artifact_windows(raw)
    emg_annots, per_channel_emg_flags = _flag_emg_windows(raw)
    all_new_annots = artifact_annots + emg_annots
    if all_new_annots:
        onsets, durations, descriptions = zip(*all_new_annots)
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
        n_emg_windows=len(emg_annots),
        per_channel_flagged_windows=per_channel_flags,
        per_channel_emg_flagged_windows=per_channel_emg_flags,
        globally_bad_channels=globally_bad,
    )
    return raw, report
