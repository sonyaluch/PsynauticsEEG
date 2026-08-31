#!/usr/bin/env python
"""
Sanity-check the pipeline's spectral estimation against a well-known,
within-recording EEG phenomenon: a resting-state alpha (8-13Hz) peak, rather
than relying on cross-study absolute-power comparison (which is known to be
unreliable across different devices/references/impedances -- see the
Cognition literature-comparison discussion).

For each channel, fits a 1/f background (linear in log-power vs log-freq,
excluding an 8-13Hz guard band) using MNE's Welch PSD computed the same way
markers.py does (epoched, BAD_*-annotation-aware). A "peak" is flagged if
the observed alpha-band power exceeds the fitted background by a given
ratio at that frequency.

Restricted to "clean" recordings (no globally-bad channel, <20%
annotated-bad) so a contaminated channel's broadband noise can't mask or
fake a peak.

Usage:
    .venv/bin/python scripts/check_alpha_peak.py \
        --qc-summary output/batch_full/batch_qc_summary.csv \
        --clean-raw-dir output/batch_full/clean_raw \
        --outdir output/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from muse_pipeline import config  # noqa: E402
from muse_pipeline.markers import _make_epochs  # noqa: E402

ALPHA_LO, ALPHA_HI = config.FREQ_BANDS_HZ["alpha"]
GUARD_LO, GUARD_HI = 6.0, 15.0  # excluded from the background fit
FIT_FMIN, FIT_FMAX = 2.0, 40.0
PEAK_RATIO_THRESHOLDS = [1.25, 1.5, 2.0]


def channel_psd(epochs: mne.Epochs, ch_name: str) -> tuple[np.ndarray, np.ndarray]:
    spectrum = epochs.compute_psd(picks=[ch_name], fmin=FIT_FMIN, fmax=FIT_FMAX, verbose=False)
    psd = spectrum.get_data()[:, 0, :].mean(axis=0) * 1e12  # mean across epochs, V^2/Hz -> uV^2/Hz
    freqs = spectrum.freqs
    return freqs, psd


def fit_background(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    """Linear fit of log10(psd) ~ log10(freq), excluding the alpha guard band."""
    mask = (freqs < GUARD_LO) | (freqs > GUARD_HI)
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    slope, intercept = np.polyfit(log_f, log_p, 1)
    return slope, intercept


def alpha_peak_ratio(freqs: np.ndarray, psd: np.ndarray) -> float:
    slope, intercept = fit_background(freqs, psd)
    alpha_mask = (freqs >= ALPHA_LO) & (freqs <= ALPHA_HI)
    observed_max = psd[alpha_mask].max()
    peak_freq = freqs[alpha_mask][np.argmax(psd[alpha_mask])]
    predicted_bg = 10 ** (slope * np.log10(peak_freq) + intercept)
    return observed_max / predicted_bg


def fig_example(freqs, psd, out_path: Path, label: str) -> None:
    slope, intercept = fit_background(freqs, psd)
    bg = 10 ** (slope * np.log10(freqs) + intercept)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(freqs, psd, color="black", linewidth=1, label="Observed PSD")
    ax.loglog(freqs, bg, color="gray", linestyle="--", linewidth=1, label="Fitted 1/f background")
    ax.axvspan(ALPHA_LO, ALPHA_HI, color="orange", alpha=0.2, label="Alpha band (8-13Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD ($\mu V^2/Hz$)")
    ax.set_title(label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-summary", type=Path, required=True)
    parser.add_argument("--clean-raw-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    qc = pd.read_csv(args.qc_summary)
    clean = qc[(qc["bad_channels"].isna()) & (qc["pct_annotated_bad"] < 20) & (qc["error"].isna())]
    print(f"{len(clean)} clean recordings ({clean['subject_code'].nunique()} subjects)")

    rows = []
    example_saved = 0
    for _, row in clean.iterrows():
        fif_path = args.clean_raw_dir / row["output_file"]
        if not fif_path.exists():
            continue
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        epochs = _make_epochs(raw, reject_by_annotation=True)
        if len(epochs) == 0:
            continue
        for ch in config.MUSE_CHANNEL_ORDER:
            freqs, psd = channel_psd(epochs, ch)
            ratio = alpha_peak_ratio(freqs, psd)
            rows.append({"subject_code": row["subject_code"], "module_id": row["module_id"], "channel": ch, "ratio": ratio})
            if example_saved < 4 and ch == "AF8" and ratio > 1.5:
                fig_example(
                    freqs, psd, args.outdir / f"alpha_peak_example_{example_saved+1}.png",
                    f"{row['subject_code']} {row['module_id']}, AF8 (peak ratio {ratio:.2f}x)",
                )
                example_saved += 1

    results = pd.DataFrame(rows)
    results.to_csv(args.outdir / "alpha_peak_check.csv", index=False)

    print(f"\n{len(results)} channel-recordings checked ({results['subject_code'].nunique()} subjects)")
    print("\nAlpha peak detection rate at each threshold (any channel in the recording):")
    per_recording_max = results.groupby(["subject_code", "module_id"])["ratio"].max()
    for thresh in PEAK_RATIO_THRESHOLDS:
        pct = 100.0 * (per_recording_max > thresh).mean()
        print(f"  ratio > {thresh}x: {pct:.0f}% of recordings ({int((per_recording_max > thresh).sum())}/{len(per_recording_max)})")

    print("\nPer-channel detection rate (ratio > 1.5x):")
    for ch in config.MUSE_CHANNEL_ORDER:
        sub = results[results["channel"] == ch]
        pct = 100.0 * (sub["ratio"] > 1.5).mean()
        print(f"  {ch}: {pct:.0f}% ({int((sub['ratio']>1.5).sum())}/{len(sub)})")

    print(f"\nSaved {example_saved} example figures + alpha_peak_check.csv to {args.outdir}")


if __name__ == "__main__":
    main()
