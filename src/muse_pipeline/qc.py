"""QC plots for a preprocessing run: before/after PSD, raw trace with annotations."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from . import config


def plot_psd_before_after(raw_before: mne.io.RawArray, raw_after: mne.io.RawArray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    raw_before.compute_psd(fmax=60, verbose=False).plot(axes=axes[0], show=False, amplitude=False)
    axes[0].set_title("Before filtering")
    raw_after.compute_psd(fmax=60, verbose=False).plot(axes=axes[1], show=False, amplitude=False)
    axes[1].set_title(f"After filtering ({config.BANDPASS_LOW_HZ}-{config.BANDPASS_HIGH_HZ} Hz + notch)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_trace_with_annotations(raw: mne.io.RawArray, out_path: Path, max_seconds: float = 60.0) -> None:
    """Plot the first `max_seconds` of each channel, shading BAD_* annotated regions."""
    sfreq = raw.info["sfreq"]
    n_samples = min(int(max_seconds * sfreq), raw.n_times)
    data, times = raw.get_data(start=0, stop=n_samples, return_times=True)

    fig, axes = plt.subplots(len(raw.ch_names), 1, figsize=(12, 7), sharex=True)
    for i, (ch, ax) in enumerate(zip(raw.ch_names, axes)):
        ax.plot(times, data[i] * 1e6, linewidth=0.5, color="black")
        ax.set_ylabel(ch, rotation=0, ha="right", va="center")
        ax.set_xlim(times[0], times[-1])

    for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if onset > times[-1]:
            continue
        color = "red" if desc == "BAD_dropout" else "orange"
        for ax in axes:
            ax.axvspan(onset, min(onset + duration, times[-1]), color=color, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"First {max_seconds:.0f}s -- red=dropout, orange=artifact")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
