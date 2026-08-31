"""
Marker extraction from preprocessed Muse Raw objects: Cognition (AF8 band
power), Emotion (AF7/AF8 alpha asymmetry), Awareness (Lempel-Ziv complexity +
permutation entropy across all 4 channels).

All three formulas are ASSUMPTIONs based on established EEG literature, not
confirmed Psynautics protocol specs -- see config.py for band definitions,
formulas, and citations.

Computed per fixed-length epoch (any epoch overlapping a BAD_* annotation is
dropped via MNE's reject_by_annotation), then averaged across epochs into
one scalar per recording. A marker whose required channel(s) are in
raw.info["bads"] is reported as unavailable (None) rather than silently
computed on incomplete data -- Emotion needs both AF7 and AF8, Awareness
needs all 4 channels, Cognition needs only AF8.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import antropy as ant
import mne
import numpy as np

from . import config


@dataclass
class MarkerReport:
    cognition: float | None  # AF8 beta-band mean PSD, uV^2/Hz, mean across epochs
    emotion: float | None  # ln(AF8 alpha power) - ln(AF7 alpha power), mean across epochs
    awareness: float | None  # mean(normalized LZC, normalized PE) across all 4 channels + epochs
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


def _lzc_normalized(x: np.ndarray) -> float:
    binary = (x >= np.median(x)).astype(int)
    return float(ant.lziv_complexity(binary, normalize=True))


def _awareness_channel_score(epochs: mne.Epochs, ch_name: str) -> float:
    """Mean of (normalized LZC, normalized permutation entropy), across epochs, for one channel."""
    data = epochs.get_data(picks=[ch_name])[:, 0, :]  # (n_epochs, n_times)
    scores = []
    for epoch in data:
        lzc = _lzc_normalized(epoch)
        pe = ant.perm_entropy(
            epoch, order=config.PERM_ENTROPY_ORDER, delay=config.PERM_ENTROPY_DELAY, normalize=True
        )
        scores.append((lzc + pe) / 2.0)
    return float(np.mean(scores))


def extract_markers(raw: mne.io.RawArray) -> MarkerReport:
    bad_channels = list(raw.info["bads"])

    n_total = len(_make_epochs(raw, reject_by_annotation=False))
    epochs = _make_epochs(raw, reject_by_annotation=True)
    n_used = len(epochs)

    cognition = None
    if n_used > 0 and config.COGNITION_CHANNEL not in bad_channels:
        cognition = _band_power_uv2(epochs, config.COGNITION_CHANNEL, config.COGNITION_BAND)

    emotion = None
    if (
        n_used > 0
        and config.EMOTION_LEFT_CHANNEL not in bad_channels
        and config.EMOTION_RIGHT_CHANNEL not in bad_channels
    ):
        left = _band_power_uv2(epochs, config.EMOTION_LEFT_CHANNEL, config.EMOTION_BAND)
        right = _band_power_uv2(epochs, config.EMOTION_RIGHT_CHANNEL, config.EMOTION_BAND)
        emotion = float(np.log(right) - np.log(left))

    awareness = None
    if n_used > 0 and not bad_channels:
        per_channel = [_awareness_channel_score(epochs, ch) for ch in config.MUSE_CHANNEL_ORDER]
        awareness = float(np.mean(per_channel))

    return MarkerReport(
        cognition=cognition,
        emotion=emotion,
        awareness=awareness,
        n_epochs_total=n_total,
        n_epochs_used=n_used,
        bad_channels=bad_channels,
    )
