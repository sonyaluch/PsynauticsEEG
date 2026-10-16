#!/usr/bin/env python
"""
Study-level summary figures for the three markers (Performance/Cognition,
Mood/Emotion, Awareness + its LZC companion): what was found, how messy the
data is, and whether the study can plausibly detect an effect.

Writes three figures to --outdir:
  summary_data_messiness.png   recording funnel, bad channels, EMG / artifact rates
  summary_findings.png         paired baseline-vs-treatment for each marker
  summary_plausibility.png     inter-marker correlation, artifact-tracking,
                               within-subject stability (ICC), statistical power

Needs the outputs of extract_markers.py, batch_preprocess.py and the
stats_baseline_vs_treatment*.py scripts (the *_pairs.csv files).

Usage:
    .venv/bin/python scripts/make_summary_figures.py \
        --marker-summary output/batch_full/marker_summary.csv \
        --qc-summary output/batch_full/batch_qc_summary.csv \
        --pairs-dir output/figures --raw-dir data/raw --outdir output/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_preprocess import EXCLUDED_SESSION_HASHES  # noqa: E402

BASELINE_MODULES = {f"PSYNAUTICS_BASELINE_EEG_{i}" for i in ["1", "2", "3", "4", "5", "TDAY"]}
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# key, display name, csv column, pairs-file prefix, log-scale?, color
MARKERS = [
    ("cognition", "Performance\n(AF8 broadband 1-44Hz)", "cognition", "cognition", True, "#4C72B0"),
    ("mood", "Mood\n(AF7 alpha 7.5-13Hz)", "emotion", "mood", True, "#C44E52"),
    ("awareness", "Awareness\n(permutation entropy)", "awareness", "awareness", False, "#55A868"),
    ("lzc", "Awareness companion\n(Lempel-Ziv complexity)", "awareness_lzc", "awareness_lzc", False, "#8172B2"),
]


def _analysis_col(df: pd.DataFrame, col: str, log: bool) -> pd.Series:
    return np.log10(df[col]) if log else df[col]


def icc1(df: pd.DataFrame, col: str) -> float:
    """One-way random-effects ICC(1): share of variance that is between-subject."""
    groups = [g[col].to_numpy() for _, g in df.dropna(subset=[col]).groupby("subject_code") if len(g) >= 2]
    n = len(groups)
    allv = np.concatenate(groups)
    gm = allv.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (n - 1), ssw / (len(allv) - n)
    k0 = (len(allv) - sum(len(g) ** 2 for g in groups) / len(allv)) / (n - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def fig_messiness(qc: pd.DataFrame, marker_df: pd.DataFrame, n_raw: int, out: Path) -> None:
    ok = qc[qc["error"].isna()]
    n_candidates = len(qc) + len(EXCLUDED_SESSION_HASHES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    labels = ["Raw files\ndownloaded", "Resting-state\nEEG recordings", "Preprocessed\nOK", "Mood\navailable", "Performance\navailable", "Awareness\navailable"]
    vals = [
        n_raw, n_candidates, len(ok),
        int(marker_df["emotion"].notna().sum()), int(marker_df["cognition"].notna().sum()), int(marker_df["awareness"].notna().sum()),
    ]
    colors = ["#999999", "#999999", "#777777", "#C44E52", "#4C72B0", "#55A868"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha="center", fontsize=9)
    ax.set_ylabel("Recordings")
    ax.set_title("Where recordings are lost")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[0, 1]
    counts = [int(ok["bad_channels"].fillna("").str.contains(c).sum()) for c in CHANNELS]
    bars = ax.bar(CHANNELS, counts, color=["#DD8452" if c in ("TP9", "TP10") else "#8C8C8C" for c in CHANNELS])
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} ({100 * v / len(ok):.0f}%)", ha="center", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_ylabel(f"Recordings with channel flagged bad (of {len(ok)})")
    ax.set_title("Channels flagged bad for a whole recording")

    ax = axes[1, 0]
    emg_rate = (ok["n_emg_windows"] / ok["duration_s"] * 100).clip(upper=100)
    ax.hist(emg_rate, bins=40, color="#8172B2", edgecolor="white")
    ax.axvline(emg_rate.median(), color="k", linestyle="--", linewidth=1)
    ax.text(emg_rate.median() + 1, ax.get_ylim()[1] * 0.9, f"median {emg_rate.median():.1f}%", fontsize=9)
    ax.set_xlabel("% of recording flagged as high-frequency (muscle-like) bursts")
    ax.set_ylabel("Recordings")
    ax.set_title("High-frequency burst share per recording")

    ax = axes[1, 1]
    ax.hist(ok["pct_annotated_bad"].clip(upper=110), bins=40, color="#DD8452", edgecolor="white")
    ax.axvline(ok["pct_annotated_bad"].median(), color="k", linestyle="--", linewidth=1)
    ax.text(ok["pct_annotated_bad"].median() + 1, ax.get_ylim()[1] * 0.9, f"median {ok['pct_annotated_bad'].median():.0f}%", fontsize=9)
    ax.set_xlabel("% of recording annotated bad (overlaps double-count)")
    ax.set_ylabel("Recordings")
    ax.set_title("Share of each recording annotated bad")

    fig.suptitle("Data messiness", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_findings(pairs_dir: Path, out: Path) -> list[dict]:
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    summary = []
    for ax, (key, name, col, prefix, log, color) in zip(axes, MARKERS):
        p = pd.read_csv(pairs_dir / f"{prefix}_baseline_vs_treatment_pairs.csv")
        for _, r in p.iterrows():
            ax.plot([0, 1], [r["baseline"], r["treatment"]], color=color, alpha=0.45, marker="o", markersize=4)
        if log:
            ax.set_yscale("log")
            diff = p["diff_log"]
            eff = f"median x{10 ** diff.median():.2f}"
        else:
            diff = p["diff"]
            eff = f"median diff {diff.median():+.3f}"
        # test in the same space the per-marker stats scripts use (log10 for the power markers)
        a, b = (p["treatment_log"], p["baseline_log"]) if log else (p["treatment"], p["baseline"])
        w = stats.wilcoxon(a, b)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Treatment"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"{name}\nn={len(p)}, {eff}, Wilcoxon p={w.pvalue:.2f}", fontsize=9)
        summary.append({"key": key, "n": len(p), "dz": diff.mean() / diff.std(), "p": w.pvalue})
    any_sig = any(x["p"] < 0.05 for x in summary)
    fig.suptitle("Baseline vs. treatment day (paired within subject)" + ("" if any_sig else ": no marker shows a significant change"), fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return summary


def fig_plausibility(d: pd.DataFrame, qc: pd.DataFrame, summary: list[dict], out: Path) -> None:
    d = d.merge(qc[["session_hash", "n_emg_windows", "duration_s"]], on="session_hash", how="left")
    d["emg_rate"] = d["n_emg_windows"] / d["duration_s"] * 100
    cols = {}
    for key, name, col, _, log, _c in MARKERS:
        cols[key] = _analysis_col(d, col, log)
        d["a_" + key] = cols[key]
    short = {"cognition": "Performance", "mood": "Mood", "awareness": "Awareness\n(PE)", "lzc": "LZC"}
    keys = [m[0] for m in MARKERS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    corr = d[["a_" + k for k in keys]].corr(method="spearman").to_numpy()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(4))
    ax.set_xticklabels([short[k] for k in keys], fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels([short[k] for k in keys], fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Correlation between markers (Spearman rho)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    rs, ps = [], []
    for k in keys:
        s = d.dropna(subset=["a_" + k, "emg_rate"])
        r, p = stats.spearmanr(s["a_" + k], s["emg_rate"])
        rs.append(r)
        ps.append(p)
    bars = ax.bar([short[k] for k in keys], rs, color=[m[5] for m in MARKERS])
    for b, r, p in zip(bars, rs, ps):
        ax.text(b.get_x() + b.get_width() / 2, (r if r >= 0 else 0) + 0.01, f"{r:.2f}{'*' if p < 0.05 else ''}", ha="center", fontsize=10)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("Spearman rho vs. % of recording flagged")
    ax.set_title("Do markers track the share of high-frequency (muscle-like) bursts?\n(* p<0.05)", fontsize=10)

    ax = axes[1, 0]
    base = d[d["module_id"].isin(BASELINE_MODULES)]
    iccs = [icc1(base, "a_" + k) for k in keys]
    bars = ax.bar([short[k] for k in keys], iccs, color=[m[5] for m in MARKERS])
    for b, v in zip(bars, iccs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ICC across baseline sessions (0=noise, 1=trait)")
    ax.set_title("Within-subject stability across baseline sessions", fontsize=10)

    ax = axes[1, 1]
    obs = [abs(s["dz"]) for s in summary]
    det = [(stats.t.ppf(0.975, s["n"] - 1) + stats.t.ppf(0.8, s["n"] - 1)) / np.sqrt(s["n"]) for s in summary]
    x = np.arange(4)
    ax.bar(x - 0.18, obs, 0.36, color=[m[5] for m in MARKERS], label="Observed effect")
    ax.bar(x + 0.18, det, 0.36, color="#CCCCCC", edgecolor="k", label="Smallest effect detectable (80% power)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{short[k]}\nn={s['n']}" for k, s in zip(keys, summary)], fontsize=8)
    ax.set_ylabel("Standardised effect (paired Cohen's dz)")
    ax.set_title("Statistical power: observed effect vs. smallest detectable", fontsize=10)
    ax.legend(fontsize=8)

    print("plausibility numbers:")
    print("  inter-marker Spearman:", {f"{keys[i]}-{keys[j]}": round(float(corr[i, j]), 2) for i in range(4) for j in range(i + 1, 4)})
    print("  rho vs HF-burst share:", {k: (round(r, 2), round(p_, 3)) for k, r, p_ in zip(keys, rs, ps)})
    print("  ICC (baseline sessions):", {k: round(v, 2) for k, v in zip(keys, iccs)})
    print("  observed |dz| / detectable dz:", [(k, round(o, 2), round(dd, 2), s_["n"]) for k, o, dd, s_ in zip(keys, obs, det, summary)])
    fig.suptitle("Study plausibility checks", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marker-summary", type=Path, required=True)
    ap.add_argument("--qc-summary", type=Path, required=True)
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.marker_summary)
    qc = pd.read_csv(args.qc_summary)
    n_raw = sum(1 for p in args.raw_dir.rglob("*_eeg.csv") if "summary" not in p.name.lower())

    fig_messiness(qc, d, n_raw, args.outdir / "summary_data_messiness.png")
    summary = fig_findings(args.pairs_dir, args.outdir / "summary_findings.png")
    fig_plausibility(d, qc, summary, args.outdir / "summary_plausibility.png")
    print(f"Saved 3 summary figures to {args.outdir}")


if __name__ == "__main__":
    main()
