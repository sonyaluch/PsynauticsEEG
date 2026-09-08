#!/usr/bin/env python
"""
Awareness marker figures from marker_summary.csv, for sharing analysis
progress (e.g. in a meeting).

Unlike Cognition/Mood (raw power, heavy-tailed across orders of magnitude),
Awareness is normalized permutation entropy averaged across all 4 channels
-- bounded [0, 1], and empirically tightly clustered (~0.70-0.83 across the
cohort). All figures here use a LINEAR scale, not log10.

Usage:
    .venv/bin/python scripts/make_awareness_figures.py \
        --summary output/batch_full/marker_summary.csv \
        --outdir output/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
# the plain mean (linear space, unlike Cognition/Mood's log-space mean),
# so the trajectory/box figures show one value per subject per protocol
# step rather than double-weighting a few subjects.


def fig_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["awareness"], bins=30, color="#55A868", edgecolor="white")
    ax.set_xlabel("Awareness (mean normalized permutation entropy, all 4 channels)")
    ax.set_ylabel("Number of recordings")
    ax.set_title(f"Awareness marker distribution across the cohort  (n={len(df)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_by_module(per_subject: pd.DataFrame, out_path: Path) -> None:
    present = [m for m in MODULE_ORDER if m in per_subject["module_id"].unique()]
    data = [per_subject.loc[per_subject["module_id"] == m, "awareness"] for m in present]
    labels = [MODULE_LABELS[m] for m in present]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("Awareness (mean normalized permutation entropy)")
    ax.set_title("Awareness by protocol module  (one point per subject per module)")
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
        ax.plot(xs, g["awareness"], marker="o", markersize=3, linewidth=0.8, alpha=0.4, color="#55A868")

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Awareness (mean normalized permutation entropy)")
    ax.set_title("Per-participant Awareness trajectory across the protocol")
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
    df = df[df["awareness"].notna()].copy()

    fig_distribution(df, args.outdir / "awareness_distribution.png")

    # collapse to one row per (subject, module): plain mean (linear space)
    per_subject = df.groupby(["subject_code", "module_id"], as_index=False)["awareness"].mean()

    fig_by_module(per_subject, args.outdir / "awareness_by_module.png")
    fig_trajectories(per_subject, args.outdir / "awareness_trajectories.png")

    print(f"Saved 3 figures to {args.outdir}")


if __name__ == "__main__":
    main()
