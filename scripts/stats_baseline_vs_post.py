#!/usr/bin/env python
"""
Paired baseline-vs-POST-treatment test for every marker (Performance/
Cognition, Mood/Emotion, Awareness, plus the LZC companion).

Within each subject: mean baseline value (PSYNAUTICS_BASELINE_EEG_1-5 plus
the Tx-day baseline) vs. mean post-treatment value (PSYNAUTICS_POST_EEG_1-4,
Days 7-11). The Day 6 treatment-session recording itself is NOT used here --
see stats_baseline_vs_treatment*.py for baseline vs. that session.

Paired because it's the same subjects measured twice. Cognition and Mood are
tested in log10 space (raw power is heavy-tailed across orders of magnitude);
Awareness and LZC are bounded [0, 1] and tested on raw values.

Usage:
    .venv/bin/python scripts/stats_baseline_vs_post.py \
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

BASELINE_MODULES = {f"PSYNAUTICS_BASELINE_EEG_{i}" for i in ["1", "2", "3", "4", "5", "TDAY"]}
POST_MODULES = {f"PSYNAUTICS_POST_EEG_{i}" for i in ["1", "2", "3", "4"]}

# csv column, file prefix, display name, log10?, color
MARKERS = [
    ("cognition", "cognition", "Performance\n(AF8 broadband 1-44Hz)", True, "#4C72B0"),
    ("emotion", "mood", "Mood\n(AF7 alpha 7.5-13Hz)", True, "#C44E52"),
    ("awareness", "awareness", "Awareness\n(permutation entropy)", False, "#55A868"),
    ("awareness_lzc", "awareness_lzc", "Awareness companion\n(Lempel-Ziv complexity)", False, "#8172B2"),
]


def build_pairs(df: pd.DataFrame, col: str, log: bool) -> pd.DataFrame:
    d = df[df[col].notna()].copy()
    d["v"] = np.log10(d[col]) if log else d[col]
    base = d[d["module_id"].isin(BASELINE_MODULES)].groupby("subject_code")["v"].mean().rename("baseline_v")
    post = d[d["module_id"].isin(POST_MODULES)].groupby("subject_code")["v"].mean().rename("post_v")
    p = pd.concat([base, post], axis=1).dropna()
    p["diff"] = p["post_v"] - p["baseline_v"]
    if log:
        p["baseline"], p["post"] = 10 ** p["baseline_v"], 10 ** p["post_v"]
    else:
        p["baseline"], p["post"] = p["baseline_v"], p["post_v"]
    return p.reset_index()


def draw_pairs(ax, p: pd.DataFrame, name: str, log: bool, color: str) -> dict:
    for _, r in p.iterrows():
        ax.plot([0, 1], [r["baseline"], r["post"]], color=color, alpha=0.45, marker="o", markersize=4)
    if log:
        ax.set_yscale("log")
    w = stats.wilcoxon(p["post_v"], p["baseline_v"])
    t = stats.ttest_rel(p["post_v"], p["baseline_v"])
    eff = f"median x{10 ** p['diff'].median():.2f}" if log else f"median diff {p['diff'].median():+.3f}"
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline\n(subject mean)", "Post-treatment\n(subject mean, Days 7-11)"], fontsize=8)
    ax.set_xlim(-0.3, 1.3)
    ax.set_title(f"{name}\nn={len(p)}, {eff}, Wilcoxon p={w.pvalue:.2f}", fontsize=9)
    return {"n": len(p), "effect": eff, "wilcoxon_p": w.pvalue, "ttest_p": t.pvalue, "dz": p["diff"].mean() / p["diff"].std()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2))
    for ax, (col, prefix, name, log, color) in zip(axes, MARKERS):
        p = build_pairs(df, col, log)
        res = draw_pairs(ax, p, name, log, color)
        print(f"=== {col} ===  n={res['n']}  {res['effect']}  Wilcoxon p={res['wilcoxon_p']:.4f}  paired t p={res['ttest_p']:.4f}  dz={res['dz']:.2f}")
        p.to_csv(args.outdir / f"{prefix}_baseline_vs_post_pairs.csv", index=False)

        f1, a1 = plt.subplots(figsize=(5.5, 5.8))
        draw_pairs(a1, p, name, log, color)
        a1.set_ylabel(name.replace("\n", " "))
        f1.tight_layout()
        f1.savefig(args.outdir / f"{prefix}_baseline_vs_post.png", dpi=150)
        plt.close(f1)

    fig.suptitle("Baseline vs. post-treatment (paired within subject)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.outdir / "summary_baseline_vs_post.png", dpi=150)
    plt.close(fig)
    print(f"Saved figures + pairs to {args.outdir}")


if __name__ == "__main__":
    main()
