#!/usr/bin/env python
"""
Paired baseline-vs-treatment statistical test for the Awareness marker and
its awareness_lzc companion metric.

Compares, within each subject, their mean baseline value (across all
available PSYNAUTICS_BASELINE_EEG_* sessions, including the Tx-day baseline)
against their single Treatment-day value. Paired because it's the same
subjects measured twice, not independent samples.

Both awareness (permutation entropy) and awareness_lzc (Lempel-Ziv
complexity) are bounded [0, 1] and not heavily skewed, so testing/plotting
here works directly on raw values -- no log10 transform.

Usage:
    .venv/bin/python scripts/stats_baseline_vs_treatment_awareness.py \
        --summary output/batch_full/marker_summary.csv \
        --outdir output/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

BASELINE_MODULES = {
    "PSYNAUTICS_BASELINE_EEG_1",
    "PSYNAUTICS_BASELINE_EEG_2",
    "PSYNAUTICS_BASELINE_EEG_3",
    "PSYNAUTICS_BASELINE_EEG_4",
    "PSYNAUTICS_BASELINE_EEG_5",
    "PSYNAUTICS_BASELINE_EEG_TDAY",
}
TREATMENT_MODULE = "PSYNAUTICS_EEG_TREATMENT"

# (csv column, output-file prefix, axis/title label, plot color)
COMPONENTS = [
    ("awareness", "awareness", "Awareness (mean normalized permutation entropy)", "#55A868"),
    ("awareness_lzc", "awareness_lzc", "Awareness -- Lempel-Ziv complexity (mean normalized LZC)", "#8172B2"),
]


def build_pairs(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df[df[col].notna()].copy()

    baseline = df[df["module_id"].isin(BASELINE_MODULES)].groupby("subject_code")[col].mean().rename("baseline")
    treatment = df[df["module_id"] == TREATMENT_MODULE].groupby("subject_code")[col].mean().rename("treatment")

    pairs = pd.concat([baseline, treatment], axis=1).dropna()
    pairs["diff"] = pairs["treatment"] - pairs["baseline"]
    return pairs.reset_index()


def fig_paired(pairs: pd.DataFrame, label: str, color: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5.5))
    for _, row in pairs.iterrows():
        ax.plot([0, 1], [row["baseline"], row["treatment"]], color=color, alpha=0.4, marker="o", markersize=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline\n(subject mean)", "Treatment"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(label)
    ax.set_title(f"Baseline vs. Treatment, paired by subject  (n={len(pairs)})")
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

    for col, prefix, label, color in COMPONENTS:
        pairs = build_pairs(df, col)

        print(f"=== {col} ===")
        print(f"n subjects with both a baseline and a treatment value: {len(pairs)}")
        print("Median baseline: %.4f   Median treatment: %.4f" % (
            pairs["baseline"].median(), pairs["treatment"].median()
        ))
        print("Median difference (treatment - baseline): %.4f" % pairs["diff"].median())

        wilcoxon = stats.wilcoxon(pairs["treatment"], pairs["baseline"])
        print(f"Wilcoxon signed-rank test (primary): statistic={wilcoxon.statistic:.2f}, p={wilcoxon.pvalue:.4f}")

        ttest = stats.ttest_rel(pairs["treatment"], pairs["baseline"])
        print(f"Paired t-test (secondary): t={ttest.statistic:.2f}, p={ttest.pvalue:.4f}")

        fig_paired(pairs, label, color, args.outdir / f"{prefix}_baseline_vs_treatment.png")
        pairs.to_csv(args.outdir / f"{prefix}_baseline_vs_treatment_pairs.csv", index=False)
        print(f"Saved figure + pairs to {args.outdir}\n")


if __name__ == "__main__":
    main()
