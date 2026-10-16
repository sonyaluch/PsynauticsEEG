"""
Load raw Muse EEG CSV exports into an MNE Raw object.

The core problem this module solves: the Muse streams over BLE in bursts,
so raw sample timestamps are NOT evenly spaced -- they cluster into packets
with sub-millisecond deltas separated by irregular gaps, occasionally
including multi-second dropouts. Naively treating the data as if it were
sampled at a fixed rate (e.g. by ignoring timestamps and just using
row-index / assumed fs) silently misrepresents the timing of every sample
after the first dropout. This loader instead:

  1. Reads the true per-sample timestamps.
  2. Resamples onto a uniform grid at the nominal hardware rate via linear
     interpolation (needed because MNE requires evenly-sampled data).
  3. Flags any gap larger than a small jitter tolerance as a real dropout
     and annotates it BAD_dropout, so the interpolated (fabricated) samples
     inside that gap are excluded from any annotation-aware downstream step
     (epoching, PSD, etc.) rather than silently treated as real signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from . import config


@dataclass
class LoadReport:
    """Diagnostics about data quality, independent of what preprocessing does later."""

    path: Path
    n_raw_samples: int
    n_nan_samples: int  # sensor NaN readings (all channels); interpolated, not deleted
    duration_s: float
    n_dropouts: int
    total_dropout_s: float
    pct_dropout: float
    max_gap_s: float
    metadata: dict


def parse_filename_metadata(path: Path) -> dict:
    """Best-effort extraction of participant/session info from the export filename.

    Returns an empty dict (with a note) if the filename doesn't match the
    expected pattern -- this is metadata, not required for signal processing.
    """
    m = re.match(config.FILENAME_PATTERN, path.name)
    if not m:
        return {"filename_parsed": False}
    d = m.groupdict()
    d["filename_parsed"] = True
    return d


def _sample_positions(ts: np.ndarray) -> tuple[np.ndarray, int]:
    """Grid position (in samples at NOMINAL_SFREQ) of every row, honouring genuine gaps.

    The Muse sends 12-sample packets and stamps every row with the packet's
    ARRIVAL time, so rows inside a packet are only ~0.05 ms apart and arrival
    times jitter by tens of ms -- they are NOT the samples' true times. The
    samples themselves are a continuous 256 Hz stream (verified: gap-free runs
    contain 256.1 rows/s, and the 60 Hz mains line lands at exactly 60.0 Hz
    only when rows are treated as consecutive samples). So rows keep their
    order and sit on consecutive grid positions; arrival times are used only to
    size a hole where an arrival gap exceeds MAX_JITTER_S (samples were lost).
    """
    gaps = np.maximum(np.diff(ts), 0.0)
    big = np.where(gaps > config.MAX_JITTER_S)[0]
    missing = np.maximum(np.round(gaps[big] * config.NOMINAL_SFREQ).astype(int) - 1, 1)
    add = np.zeros(len(ts), dtype=np.int64)
    add[big + 1] = missing
    pos = np.arange(len(ts)) + np.cumsum(add)
    return pos, int(pos[-1]) + 1


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of True in a boolean array."""
    if not mask.any():
        return []
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return [(int(a_), int(b_ - a_)) for a_, b_ in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]


def load_muse_csv(path: str | Path) -> tuple[mne.io.RawArray, LoadReport]:
    """Load one Muse EEG CSV export into an MNE Raw object.

    Parameters
    ----------
    path : path to a raw `..._eeg.csv` export with columns ts, ch1..ch6.

    Returns
    -------
    raw : mne.io.RawArray
        4-channel EEG (TP9, AF7, AF8, TP10), resampled to a uniform grid at
        config.NOMINAL_SFREQ, with BAD_dropout annotations marking any
        interpolated-over gap. raw.info["subject_info"] and raw.info["description"]
        carry filename-parsed metadata where available.
    report : LoadReport
        Data-quality summary for logging / QC.
    """
    path = Path(path)
    df = pd.read_csv(path)

    required_cols = ["ts"] + list(config.RAW_TO_MUSE_CHANNEL)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns {missing}. "
            f"Found columns: {list(df.columns)}. "
            "The channel mapping in config.py may need updating for this export format."
        )

    # ch5/ch6 are unused aux columns and some export variants omit them
    # entirely -- that's fine, they're not required. But if they're present
    # AND contain data, this file isn't the plain 4-channel EEG export this
    # pipeline expects (likely a PPG/optics-enabled export or other device
    # variant); exclude it rather than silently dropping unknown channels.
    present_aux = [c for c in config.EMPTY_RAW_COLUMNS if c in df.columns]
    non_empty_aux = [c for c in present_aux if df[c].notna().any()]
    if non_empty_aux:
        raise ValueError(
            f"{path.name}: {non_empty_aux} contain data (not the usual empty aux "
            "columns) -- this export doesn't match the expected 4-channel EEG-only "
            "format and is excluded rather than processed."
        )

    # Row order IS sample order (the export writes packets as they arrive).
    # Deliberately NOT sorted by timestamp: back-to-back packets can carry
    # stamps within a fraction of a millisecond of each other, and sorting
    # would interleave their samples.
    ts = df["ts"].to_numpy(dtype=np.float64)
    if len(ts) < 2:
        raise ValueError(f"{path.name}: not enough samples to process ({len(ts)}).")

    pos, n_total = _sample_positions(ts)
    fs = config.NOMINAL_SFREQ

    # Lay each channel's real samples onto the grid. Holes (lost samples) and
    # sensor NaNs are NaN here; they are filled by linear interpolation so the
    # filters see a continuous signal, and runs of >= ~MAX_JITTER_S are
    # annotated BAD_dropout so nothing downstream treats them as real signal.
    # (Previously NaN rows were deleted, which silently shortened the timeline.
    # Left as NaN in the filter input they would be smeared over seconds.)
    data = np.full((len(config.MUSE_CHANNEL_ORDER), n_total), np.nan)
    n_nan_samples = 0
    for i, raw_col in enumerate(config.RAW_TO_MUSE_CHANNEL):
        v = df[raw_col].to_numpy(dtype=np.float64)
        n_nan_samples += int(np.isnan(v).sum())
        data[i, pos] = v
    missing = np.isnan(data).any(axis=0)
    grid = np.arange(n_total)
    for i in range(data.shape[0]):
        ok = ~np.isnan(data[i])
        if not ok.any():
            raise ValueError(f"{path.name}: channel {config.MUSE_CHANNEL_ORDER[i]} has no valid samples.")
        data[i] = np.interp(grid, grid[ok], data[i, ok])

    min_run = max(int(config.MAX_JITTER_S * fs) - 1, 1)
    dropouts = [(st / fs, ln / fs) for st, ln in _runs(missing) if ln >= min_run]
    duration_s = n_total / fs

    # ASSUMPTION (checked): exported values are in microvolts -- baseline ~800
    # with hard rails at 0 and ~1650, matching the Muse's documented 0-1682.8
    # uV range. MNE's internal convention for EEG is volts, so convert here.
    data *= config.RAW_UNITS_TO_VOLTS

    info = mne.create_info(
        ch_names=config.MUSE_CHANNEL_ORDER,
        sfreq=config.NOMINAL_SFREQ,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    try:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn")
    except ValueError:
        pass

    if dropouts:
        raw.set_annotations(
            mne.Annotations(
                onset=[t for t, _d in dropouts],
                duration=[d for _t, d in dropouts],
                description=["BAD_dropout"] * len(dropouts),
            )
        )

    metadata = parse_filename_metadata(path)
    raw.info["description"] = str(metadata)

    total_dropout_s = sum(d for _, d in dropouts)
    report = LoadReport(
        path=path,
        n_raw_samples=len(ts),
        n_nan_samples=n_nan_samples,
        duration_s=duration_s,
        n_dropouts=len(dropouts),
        total_dropout_s=total_dropout_s,
        pct_dropout=100.0 * total_dropout_s / duration_s if duration_s > 0 else 0.0,
        max_gap_s=max((d for _, d in dropouts), default=0.0),
        metadata=metadata,
    )
    return raw, report
