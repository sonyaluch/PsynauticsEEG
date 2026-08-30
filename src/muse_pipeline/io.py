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
    n_nan_dropped: int
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


def _find_dropouts(ts: np.ndarray) -> list[tuple[float, float]]:
    """Return (start_time, duration) for every gap exceeding the jitter tolerance."""
    dt = np.diff(ts)
    gap_idx = np.where(dt > config.MAX_JITTER_S)[0]
    return [(float(ts[i]), float(dt[i])) for i in gap_idx]


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

    # Some exports contain a handful of NaN readings on individual channels --
    # a sensor read glitch, not a BLE gap (the packet has a valid timestamp
    # but a missing/corrupt channel value). Left in place, a NaN poisons
    # np.interp's output around it, and FIR filtering later spreads that into
    # a much wider NaN block (its convolution window) well beyond the
    # original bad sample -- observed as a 32s NaN gap in one recording from
    # just 12 raw NaN samples, completely unflagged by any annotation.
    # Dropping the affected rows folds this into the existing dropout path:
    # if enough consecutive rows are affected, the resulting timestamp gap is
    # picked up by _find_dropouts and annotated BAD_dropout like any other.
    n_before_nan_drop = len(df)
    df = df.dropna(subset=list(config.RAW_TO_MUSE_CHANNEL))
    n_nan_dropped = n_before_nan_drop - len(df)

    df = df.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
    ts = df["ts"].to_numpy(dtype=np.float64)
    if len(ts) < 2:
        raise ValueError(f"{path.name}: not enough samples to process ({len(ts)}).")

    duration_s = float(ts[-1] - ts[0])
    n_target = int(round(duration_s * config.NOMINAL_SFREQ)) + 1
    target_ts = ts[0] + np.arange(n_target) / config.NOMINAL_SFREQ

    dropouts = _find_dropouts(ts)

    data = np.empty((len(config.MUSE_CHANNEL_ORDER), n_target), dtype=np.float64)
    for i, raw_col in enumerate(config.RAW_TO_MUSE_CHANNEL):
        data[i] = np.interp(target_ts, ts, df[raw_col].to_numpy(dtype=np.float64))
    # ASSUMPTION: exported values are already in microvolts (baseline ~800,
    # per-channel std ~7-24 in the sample file -- consistent with uV-scale
    # EEG fluctuations, not raw 12-bit ADC counts). MNE's internal convention
    # for EEG data is volts, so convert uV -> V here. If this export format
    # turns out to be raw ADC counts instead, replace this with the correct
    # counts-to-uV calibration formula before this line.
    data *= config.RAW_UNITS_TO_VOLTS

    info = mne.create_info(
        ch_names=config.MUSE_CHANNEL_ORDER,
        sfreq=config.NOMINAL_SFREQ,
        ch_types="eeg",
    )
    # Muse "raw" units aren't a confirmed calibrated physical unit; MNE
    # internally assumes volts for EEG. We keep values as-exported (do not
    # rescale) and note this explicitly rather than silently asserting a
    # unit that hasn't been verified.
    raw = mne.io.RawArray(data, info, verbose=False)
    try:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn")
    except ValueError:
        pass

    if dropouts:
        onsets = [t - ts[0] for t, _dur in dropouts]
        durations = [dur for _t, dur in dropouts]
        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=durations,
                description=["BAD_dropout"] * len(dropouts),
            )
        )

    metadata = parse_filename_metadata(path)
    raw.info["description"] = str(metadata)

    total_dropout_s = sum(d for _, d in dropouts)
    report = LoadReport(
        path=path,
        n_raw_samples=len(ts),
        n_nan_dropped=n_nan_dropped,
        duration_s=duration_s,
        n_dropouts=len(dropouts),
        total_dropout_s=total_dropout_s,
        pct_dropout=100.0 * total_dropout_s / duration_s if duration_s > 0 else 0.0,
        max_gap_s=max((d for _, d in dropouts), default=0.0),
        metadata=metadata,
    )
    return raw, report
