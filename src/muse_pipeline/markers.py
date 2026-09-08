"""
Marker extraction from preprocessed Muse Raw objects: Cognition/Performance
(AF8 broadband power, 1-44Hz), Emotion/Mood (AF7 alpha-band power,
7.5-13Hz), Awareness (permutation entropy across all 4 channels).

All three definitions are CONFIRMED per an update from the study -- see
config.py for band definitions and correction history (each was previously
an ASSUMPTION-tagged literature-standard guess; Cognition's guess turned
out correct, Emotion's and Awareness's did not).

`awareness_lzc` (normalized Lempel-Ziv complexity, same 4-channel gating as
Awareness) is reported alongside Awareness as an additional companion
metric, at the user's explicit request -- it is NOT part of the confirmed
Awareness spec (which is permutation entropy alone) and is not averaged
into the `awareness` value.

Computed per fixed-length epoch (any epoch overlapping a BAD_* annotation is
dropped via MNE's reject_by_annotation), then averaged across epochs into
one scalar per recording. A marker whose required channel(s) are in
raw.info["bads"] is reported as unavailable (None) rather than silently
computed on incomplete data -- Cognition needs AF8, Emotion needs AF7,
Awareness and awareness_lzc need all 4 channels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import antropy as ant
import mne
import numpy as np

from . import config


@dataclass
class MarkerReport:
    cognition: float | None  # AF8 broadband (1-44Hz) mean PSD, uV^2/Hz, mean across epochs
    emotion: float | None  # AF7 mood-alpha (7.5-13Hz) mean PSD, uV^2/Hz, mean across epochs
    awareness: float | None  # mean normalized permutation entropy across all 4 channels + epochs
    awareness_lzc: float | None  # mean normalized Lempel-Ziv complexity across all 4 channels + epochs -- additional, not part of the confirmed Awareness spec
    n_epochs_total: int
    n_epochs_used: int  # after dropping epochs overlapping BAD_* annotations
    bad_channels: list[str] = field(default_factory=list)


def _make_epochs(raw: mne.io.RawArray, reject_by_annotation: bool) -> mne.Epochs:
    return mne.make_fixed_length_epochs(
        raw,
        duration=config.MARKER_EPOCH_S,
        overlap=0.0,
        reject_by_annotation=reject_by_annotation,
        preload=True,
        verbose=False,
    )


def _band_power_uv2(epochs: mne.Epochs, ch_name: str, band: str) -> float:
    """Mean PSD (uV^2/Hz) in `band` at `ch_name`, averaged across epochs (mean across freq bins within the band, NOT integrated/summed -- this is spectral density, not total band power)."""
    fmin, fmax = config.FREQ_BANDS_HZ[band]
    spectrum = epochs.compute_psd(picks=[ch_name], fmin=fmin, fmax=fmax, verbose=False)
    psd = spectrum.get_data()  # (n_epochs, 1, n_freqs), V^2/Hz
    band_power_per_epoch = psd.mean(axis=-1)[:, 0] * 1e12  # V^2/Hz -> uV^2/Hz
    return float(np.mean(band_power_per_epoch))


def _awareness_channel_score(epochs: mne.Epochs, ch_name: str) -> float:
    """Mean normalized permutation entropy across epochs, for one channel."""
    data = epochs.get_data(picks=[ch_name])[:, 0, :]  # (n_epochs, n_times)
    scores = [
        ant.perm_entropy(epoch, order=config.PERM_ENTROPY_ORDER, delay=config.PERM_ENTROPY_DELAY, normalize=True)
        for epoch in data
    ]
    return float(np.mean(scores))


def _lzc_normalized(x: np.ndarray) -> float:
    """Normalized Lempel-Ziv complexity of one epoch, via a per-epoch median-split binarization."""
    binary = (x >= np.median(x)).astype(int)
    return float(ant.lziv_complexity(binary, normalize=True))


def _awareness_lzc_channel_score(epochs: mne.Epochs, ch_name: str) -> float:
    """Mean normalized Lempel-Ziv complexity across epochs, for one channel."""
    data = epochs.get_data(picks=[ch_name])[:, 0, :]  # (n_epochs, n_times)
    return float(np.mean([_lzc_normalized(epoch) for epoch in data]))


def extract_markers(raw: mne.io.RawArray) -> MarkerReport:
    bad_channels = list(raw.info["bads"])

    n_total = len(_make_epochs(raw, reject_by_annotation=False))
    epochs = _make_epochs(raw, reject_by_annotation=True)
    n_used = len(epochs)

    cognition = None
    if n_used > 0 and config.COGNITION_CHANNEL not in bad_channels:
        cognition = _band_power_uv2(epochs, config.COGNITION_CHANNEL, config.COGNITION_BAND)

    emotion = None
    if n_used > 0 and config.EMOTION_CHANNEL not in bad_channels:
        emotion = _band_power_uv2(epochs, config.EMOTION_CHANNEL, config.EMOTION_BAND)

    awareness = None
    awareness_lzc = None
    if n_used > 0 and not bad_channels:
        per_channel = [_awareness_channel_score(epochs, ch) for ch in config.MUSE_CHANNEL_ORDER]
        awareness = float(np.mean(per_channel))
        per_channel_lzc = [_awareness_lzc_channel_score(epochs, ch) for ch in config.MUSE_CHANNEL_ORDER]
        awareness_lzc = float(np.mean(per_channel_lzc))

    return MarkerReport(
        cognition=cognition,
        emotion=emotion,
        awareness=awareness,
        awareness_lzc=awareness_lzc,
        n_epochs_total=n_total,
        n_epochs_used=n_used,
        bad_channels=bad_channels,
    )
