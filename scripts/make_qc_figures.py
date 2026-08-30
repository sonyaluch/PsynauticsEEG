#!/usr/bin/env python
"""
Generate cohort-level QC summary figures from batch_qc_summary.csv, for
sharing preprocessing progress (e.g. in a progress meeting) without digging
through the raw CSV.

Usage:
    .venv/bin/python scripts/make_qc_figures.py \
        --summary output/batch_full/batch_qc_summary.csv \
        --outdir output/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Files skipped outright before reaching the CSV (see EXCLUDED_SESSION_HASHES
# in batch_preprocess.py) -- kept here too so the cohort-outcome figure can
# show them, since the summary CSV has no row for them at all.
MANUALLY_EXCLUDED = [
    ("Sub_36", "BASELINE_EEG_2", "20.5hr duration, 99% dropout"),
    ("Sub_51", "BASELINE_EEG_5", "3.0hr duration"),
    ("Sub_16", "BASELINE_EEG_1", "40% dropout, repeated BLE disconnects"),
]

MODULE_ORDER = [
    "PSYNAUTICS_BASELINE_EEG_1",
    "PSYNAUTICS_BASELINE_EEG_2",
    "PSYNAUTICS_BASELINE_EEG_3",
    "PSYNAUTICS_BASELINE_EEG_4",
    "PSYNAUTICS_BASELINE_EEG_5",
    "PSYNAUTICS_BASELINE_EEG_TDAY",
    "PSYNAUTICS_EEG_TREATMENT",
    "PSYNAUTICS_POST_EEG_1",
    "PSYNAUTICS_POST_EEG_2",
    "PSYNAUTICS_POST_EEG_3",
    "PSYNAUTICS_POST_EEG_4",
]
MODULE_LABELS = {
    "PSYNAUTICS_BASELINE_EEG_1": "Baseline 1",
    "PSYNAUTICS_BASELINE_EEG_2": "Baseline 2",
    "PSYNAUTICS_BASELINE_EEG_3": "Baseline 3",
    "PSYNAUTICS_BASELINE_EEG_4": "Baseline 4",
    "PSYNAUTICS_BASELINE_EEG_5": "Baseline 5",
    "PSYNAUTICS_BASELINE_EEG_TDAY": "Baseline\n(Tx day)",
    "PSYNAUTICS_EEG_TREATMENT": "Treatment",
    "PSYNAUTICS_POST_EEG_1": "Post 1",
    "PSYNAUTICS_POST_EEG_2": "Post 2",
    "PSYNAUTICS_POST_EEG_3": "Post 3",
    "PSYNAUTICS_POST_EEG_4": "Post 4",
}


def fig_quality_distributions(ok: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(ok["pct_dropout"], bins=30, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("% of recording flagged BAD_dropout")
    axes[0].set_ylabel("Number of recordings")
    axes[0].set_title(f"Dropout %  (n={len(ok)})")

    axes[1].hist(ok["pct_annotated_bad"], bins=30, color="#C44E52", edgecolor="white")
    axes[1].set_xlabel("% of recording flagged BAD (dropout + artifact)")
    axes[1].set_title(f"Total annotated-bad %  (n={len(ok)})")

    fig.suptitle("Data quality across the cohort")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_cohort_outcomes(df: pd.DataFrame, out_path: Path) -> None:
    n_ok = int(df["error"].isna().sum())
    n_failed = int(df["error"].notna().sum())
    n_partial = int(df["bad_channels"].notna().sum())
    n_excluded = len(MANUALLY_EXCLUDED)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Clean", "Partial\n(1 bad channel)", "Excluded\n(schema mismatch)", "Excluded\n(implausible duration)"]
    counts = [n_ok - n_partial, n_partial, n_failed, n_excluded]
    colors = ["#55A868", "#DD8452", "#C44E52", "#C44E52"]
    bars = ax.bar(labels, counts, color=colors)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(c), ha="center")
    ax.set_ylabel("Number of recordings")
    ax.set_title(f"Cohort outcome  (n={n_ok + n_failed + n_excluded} EEG-resting-state recordings)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_duration_by_module(ok: pd.DataFrame, out_path: Path) -> None:
    present = [m for m in MODULE_ORDER if m in ok["module_id"].unique()]
    data = [ok.loc[ok["module_id"] == m, "duration_s"] / 60.0 for m in present]
    labels = [MODULE_LABELS[m] for m in present]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("Recording duration (minutes)")
    ax.set_title("Recording duration by protocol module")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)
    ok = df[df["error"].isna()].copy()

    fig_quality_distributions(ok, args.outdir / "quality_distributions.png")
    fig_cohort_outcomes(df, args.outdir / "cohort_outcomes.png")
    fig_duration_by_module(ok, args.outdir / "duration_by_module.png")

    print(f"Saved 3 figures to {args.outdir}")


if __name__ == "__main__":
    main()
