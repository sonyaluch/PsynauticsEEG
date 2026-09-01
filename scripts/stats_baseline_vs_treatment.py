#!/usr/bin/env python
"""
Paired baseline-vs-treatment statistical test for the Cognition marker.

Compares, within each subject, their mean baseline Cognition against their single Treatment-day Cognition value.

All testing and plotting works in log10.

Usage:
    .venv/bin/python scripts/stats_baseline_vs_treatment.py \
        --summary output/batch_full/marker_summary.csv \
        --outdir output/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["cognition"].notna()].copy()
    df["log_cognition"] = np.log10(df["cognition"])

    baseline = (
        df[df["module_id"].isin(BASELINE_MODULES)]
        .groupby("subject_code")["log_cognition"]
        .mean()
        .rename("baseline_log")
    )
    # a subject can have >1 treatment file (rare re-recording); collapse the
    # same way as baseline, by mean in log-space
    treatment = (
        df[df["module_id"] == TREATMENT_MODULE]
        .groupby("subject_code")["log_cognition"]
        .mean()
        .rename("treatment_log")
    )

    pairs = pd.concat([baseline, treatment], axis=1).dropna()
    pairs["baseline"] = 10 ** pairs["baseline_log"]
    pairs["treatment"] = 10 ** pairs["treatment_log"]
    pairs["diff_log"] = pairs["treatment_log"] - pairs["baseline_log"]
    return pairs.reset_index()


def fig_paired(pairs: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5.5))
    for _, row in pairs.iterrows():
        ax.plot([0, 1], [row["baseline"], row["treatment"]], color="#4C72B0", alpha=0.4, marker="o", markersize=4)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline\n(subject mean)", "Treatment"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(r"Cognition (AF8 beta-band PSD, $\mu V^2/Hz$)")
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
    pairs = build_pairs(df)

    print(f"n subjects with both a baseline and a treatment Cognition value: {len(pairs)}")
    print()
    print("Median baseline: %.1f uV^2/Hz   Median treatment: %.1f uV^2/Hz" % (
        pairs["baseline"].median(), pairs["treatment"].median()
    ))
    print("Median log10 difference (treatment - baseline): %.3f  (x%.2f fold)" % (
        pairs["diff_log"].median(), 10 ** pairs["diff_log"].median()
    ))
    print()

    wilcoxon = stats.wilcoxon(pairs["treatment_log"], pairs["baseline_log"])
    print(f"Wilcoxon signed-rank test (non-parametric, primary): statistic={wilcoxon.statistic:.2f}, p={wilcoxon.pvalue:.4f}")

    ttest = stats.ttest_rel(pairs["treatment_log"], pairs["baseline_log"])
    print(f"Paired t-test on log10 values (parametric, secondary): t={ttest.statistic:.2f}, p={ttest.pvalue:.4f}")

    fig_paired(pairs, args.outdir / "cognition_baseline_vs_treatment.png")
    pairs.to_csv(args.outdir / "cognition_baseline_vs_treatment_pairs.csv", index=False)
    print(f"\nSaved figure + per-subject pairs to {args.outdir}")


if __name__ == "__main__":
    main()
