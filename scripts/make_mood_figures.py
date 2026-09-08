#!/usr/bin/env python
"""
Mood (Emotion) marker figures from marker_summary.csv, for sharing analysis
progress (e.g. in a meeting).

Mood (AF7 alpha-band, 7.5-13Hz, PSD) spans several orders of magnitude
across the cohort, so every figure here uses a log10 scale -- plotting it
linearly would make the bulk of recordings indistinguishable near zero.

Usage:
    .venv/bin/python scripts/make_mood_figures.py \
        --summary output/batch_full/marker_summary.csv \
        --outdir output/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

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
# One point per module per subject -- collapses duplicate re-recordings
# (a participant occasionally has 2 files for the same module) by taking
# the mean in log-space, so the trajectory/box figures show one value per
# subject per protocol step rather than double-weighting a few subjects.


def fig_distribution(df: pd.DataFrame, out_path: Path) -> None:
    log_vals = np.log10(df["emotion"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(log_vals, bins=30, color="#C44E52", edgecolor="white")
    ax.set_xlabel(r"Mood (AF7 alpha-band PSD, 7.5-13Hz, $\mu V^2/Hz$, log$_{10}$ scale)")
    ax.set_ylabel("Number of recordings")
    ax.set_title(f"Mood marker distribution across the cohort  (n={len(df)})")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _pos: f"{10**t:,.0f}"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_by_module(per_subject: pd.DataFrame, out_path: Path) -> None:
    present = [m for m in MODULE_ORDER if m in per_subject["module_id"].unique()]
    data = [per_subject.loc[per_subject["module_id"] == m, "emotion"] for m in present]
    labels = [MODULE_LABELS[m] for m in present]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel(r"Mood (AF7 alpha-band PSD, $\mu V^2/Hz$)")
    ax.set_title("Mood by protocol module  (one point per subject per module)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_trajectories(per_subject: pd.DataFrame, out_path: Path) -> None:
    present = [m for m in MODULE_ORDER if m in per_subject["module_id"].unique()]
    x_pos = {m: i for i, m in enumerate(present)}
    labels = [MODULE_LABELS[m] for m in present]

    fig, ax = plt.subplots(figsize=(11, 5))
    for subject, g in per_subject.groupby("subject_code"):
        g = g.sort_values("module_id", key=lambda s: s.map(x_pos))
        xs = g["module_id"].map(x_pos)
        ax.plot(xs, g["emotion"], marker="o", markersize=3, linewidth=0.8, alpha=0.4, color="#C44E52")

    ax.set_yscale("log")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"Mood (AF7 alpha-band PSD, $\mu V^2/Hz$)")
    ax.set_title("Per-participant Mood trajectory across the protocol")
    ax.axvline(x_pos.get("PSYNAUTICS_EEG_TREATMENT", 5) - 0.5, color="gray", linestyle="--", linewidth=1)
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
    df = df[df["emotion"].notna()].copy()

    fig_distribution(df, args.outdir / "mood_distribution.png")

    # collapse to one row per (subject, module): mean in log-space
    df["log_emotion"] = np.log10(df["emotion"])
    per_subject = (
        df.groupby(["subject_code", "module_id"], as_index=False)["log_emotion"].mean()
    )
    per_subject["emotion"] = 10 ** per_subject["log_emotion"]

    fig_by_module(per_subject, args.outdir / "mood_by_module.png")
    fig_trajectories(per_subject, args.outdir / "mood_trajectories.png")

    print(f"Saved 3 figures to {args.outdir}")


if __name__ == "__main__":
    main()
