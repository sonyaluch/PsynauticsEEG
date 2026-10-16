#!/usr/bin/env python
"""
Individual participant report as a single self-contained HTML page, laid out
like the Psynautics/Muse session reports: banner, session-info bar, then
"Longitudinal metrics", "Pre/post comparison" and "Treatment metrics" sections
with explanatory text on the left and a chart on the right.

Scores are PERCENTILE RANKS against every other participant's recordings
(50 = typical) so the three markers can share one axis. This is a context
scale, not a quality judgement -- the app's own 0-18 transform is not used
because it is not available to this pipeline.

Usage:
    .venv/bin/python scripts/make_participant_report_html.py --subject Sub_53 \
        --label "Gerard Walsh" --outdir output/participant_reports
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import io
import sys
from pathlib import Path

import antropy as ant
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from make_participant_report import (  # noqa: E402
    BASELINE, MARKERS, MODULE_ORDER, POST, TREATMENT, fmt, load_all,
)
from muse_pipeline import config  # noqa: E402
from muse_pipeline.markers import _make_epochs  # noqa: E402

MAIN = MARKERS[:3]  # Performance, Mood, Awareness (LZC is a companion, shown in the table only)
STEP_LABELS = ["day 1", "day 2", "day 3", "day 4", "day 5", "day 6\n(pre)", "day 6\ntreatment", "post 1", "post 2", "post 3", "post 4"]
SHORT = {"cognition": "Performance", "emotion": "Mood", "awareness": "Awareness"}


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def add_pct(me: pd.DataFrame, others: pd.DataFrame) -> pd.DataFrame:
    me = me.copy()
    for col, *_ in MAIN:
        pool = others[col].dropna().to_numpy()
        me["pct_" + col] = [percentileofscore(pool, v) if pd.notna(v) else np.nan for v in me[col]]
    return me


def fig_longitudinal(me: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for col, name, _log, color, _u in MAIN:
        d = me.dropna(subset=["pct_" + col])
        if d.empty:
            continue
        ax.scatter(d["step"], d["pct_" + col], color=color, alpha=0.3, s=18)
        g = d.groupby("step")["pct_" + col].mean()
        ax.plot(g.index, g.values, "-o", color=color, label=SHORT[col], markersize=5)
    ax.axhline(50, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 10.5)
    ax.set_xticks(range(11))
    ax.set_xticklabels(STEP_LABELS, fontsize=7)
    ax.set_ylabel("Percentile vs. other participants (50 = typical)", fontsize=8)
    first = me.sort_values("datetime").groupby("step")["datetime"].first()
    top = ax.secondary_xaxis("top")
    top.set_xticks(list(first.index))
    top.set_xticklabels([f"{t:%m-%d %H:%M}" for t in first], rotation=90, fontsize=6)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    return b64(fig)


def fig_prepost(me: pd.DataFrame) -> tuple[str, dict]:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    info = {}
    w = 0.36
    for i, (col, name, _log, color, _u) in enumerate(MAIN):
        c = "pct_" + col
        for j, (grp, mods, alpha) in enumerate([("Pre", BASELINE, 1.0), ("Post", POST, 0.45)]):
            v = me[me["module_id"].isin(mods)][c].dropna()
            info[(col, grp)] = (v.mean() if len(v) else None, len(v))
            if len(v):
                ax.bar(i + (j - 0.5) * w, v.mean(), w, color=color, alpha=alpha, label=f"{SHORT[col]} {grp}")
                if len(v) > 1:
                    ax.errorbar(i + (j - 0.5) * w, v.mean(), yerr=v.std(), color="k", capsize=3, linewidth=1)
                ax.text(i + (j - 0.5) * w, v.mean() + 2, f"n={len(v)}", ha="center", fontsize=7)
            else:
                ax.text(i + (j - 0.5) * w, 3, "n/a", ha="center", fontsize=7, color="#888")
    ax.set_xticks(range(3))
    ax.set_xticklabels([SHORT[m[0]] for m in MAIN])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Mean percentile (bars: +/- SD across sessions)", fontsize=8)
    ax.axhline(50, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    return b64(fig), info


def epoch_series(fif: Path) -> dict:
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    ep = _make_epochs(raw, reject_by_annotation=True)
    bads = list(raw.info["bads"])
    t = ep.events[:, 0] / raw.info["sfreq"] / 60.0
    out = {"minutes": raw.times[-1] / 60.0, "t": t, "bads": bads, "n_ep": len(ep)}
    if len(ep) == 0:
        return out

    def band(ch, band_name):
        lo, hi = config.FREQ_BANDS_HZ[band_name]
        s = ep.compute_psd(picks=[ch], fmin=lo, fmax=hi, verbose=False).get_data()
        return s.mean(axis=-1)[:, 0] * 1e12

    out["cognition"] = None if config.COGNITION_CHANNEL in bads else band(config.COGNITION_CHANNEL, config.COGNITION_BAND)
    out["emotion"] = None if config.EMOTION_CHANNEL in bads else band(config.EMOTION_CHANNEL, config.EMOTION_BAND)
    if bads:
        out["awareness"] = None
    else:
        per = []
        for ch in config.MUSE_CHANNEL_ORDER:
            x = ep.get_data(picks=[ch])[:, 0, :]
            per.append([ant.perm_entropy(e, order=config.PERM_ENTROPY_ORDER, delay=config.PERM_ENTROPY_DELAY, normalize=True) for e in x])
        out["awareness"] = np.mean(per, axis=0)
    return out


def fig_treatment(series: dict) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, (col, name, log, color, unit) in zip(axes, MAIN):
        v = series.get(col)
        if v is None:
            ax.text(0.5, 0.5, f"{SHORT[col]}: not available (a required channel was flagged bad)", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#888")
            ax.set_yticks([])
        else:
            ax.plot(series["t"], v, "-o", color=color, markersize=3, linewidth=0.8)
            if log:
                ax.set_yscale("log")
            ax.set_ylabel(f"{SHORT[col]}\n({unit or '0-1'})", fontsize=7)
    axes[-1].set_xlabel("Minutes into the treatment-day recording (each point = one clean 4 s window)", fontsize=8)
    axes[-1].set_xlim(0, max(series["minutes"], 1))
    fig.tight_layout()
    return b64(fig)


CSS = """
body{margin:0;background:#f4f4f4;font-family:-apple-system,Helvetica,Arial,sans-serif;color:#222}
.page{max-width:1000px;margin:20px auto;background:#fff}
.top{padding:22px 36px;color:#888;font-size:13px}
.banner{background:#3b3045;color:#fff;text-align:center;padding:30px 20px}
.banner h1{margin:0 0 8px;font-size:26px;letter-spacing:.5px}
.banner div{font-size:14px}
.info{padding:22px 36px;font-size:14px}
.sec{background:#eeeeee;padding:14px 36px;font-size:22px;margin-top:10px}
.row{display:grid;grid-template-columns:30% 70%;gap:20px;padding:18px 36px}
.row h3{grid-column:1/3;margin:6px 0 0;font-weight:400;font-size:17px;letter-spacing:.3px;text-transform:uppercase}
.txt{font-size:12.5px;color:#777;line-height:1.5}
.txt b{color:#555}
img{max-width:100%}
table{border-collapse:collapse;width:100%;font-size:12px}
th{background:#e6ebf5;text-align:left}
th,td{border:1px solid #ddd;padding:5px 7px}
tr:nth-child(even) td{background:#fafafa}
.note{background:#fff7e6;border-left:4px solid #e0a030;padding:10px 14px;margin:12px 36px;font-size:12.5px}
"""


def pct_txt(info, col):
    pre, post = info[(col, "Pre")], info[(col, "Post")]
    if pre[0] is None or post[0] is None:
        return f"<b>{SHORT[col]}:</b> cannot compare (usable sessions: {pre[1]} pre, {post[1]} post)."
    return f"<b>{SHORT[col]}:</b> pre {pre[0]:.0f}th percentile (n={pre[1]}), post {post[0]:.0f}th (n={post[1]})."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--marker-summary", type=Path, default=Path("output/batch_full/marker_summary.csv"))
    ap.add_argument("--qc-summary", type=Path, default=Path("output/batch_full/batch_qc_summary.csv"))
    ap.add_argument("--clean-raw-dir", type=Path, default=Path("output/batch_full/clean_raw"))
    ap.add_argument("--clinicinfo-dir", type=Path, default=Path("data/clinicinfo"))
    ap.add_argument("--participant-key", type=Path, default=Path("/Users/sonyaluchanskaya/Desktop/Psynautics/participant_key_CONFIDENTIAL.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("output/participant_reports"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    label = a.label or a.subject

    d = load_all(a.marker_summary, a.qc_summary)
    me, others = d[d["subject_code"] == a.subject].sort_values("datetime"), d[d["subject_code"] != a.subject]
    if me.empty:
        raise SystemExit(f"No recordings found for {a.subject}")
    me = add_pct(me, others)

    long_png = fig_longitudinal(me)
    pp_png, info = fig_prepost(me)

    tx = me[me["module_id"] == TREATMENT]
    tx_png, tx_note, tx_date = None, "No treatment-day recording found.", ""
    if len(tx):
        r = tx.sort_values("duration_s").iloc[-1]
        series = epoch_series(a.clean_raw_dir / r["file"])
        tx_date = f"{r['datetime']:%Y-%m-%d %H:%M}"
        tx_png = fig_treatment(series)
        tx_note = (f"Treatment-day recording of {series['minutes']:.1f} min with {series['n_ep']} clean 4-second windows "
                   f"({100 * series['n_ep'] * 4 / 60 / max(series['minutes'], 1e-9):.0f}% of it usable).")
        if len(tx) > 1:
            tx_note += f" {len(tx)} treatment-labelled recordings exist; the longest is shown."

    key = pd.read_csv(a.participant_key, header=None, names=["s", "e"])
    emails = key.loc[key["s"] == a.subject, "e"].tolist()
    clin = []
    for f in sorted(a.clinicinfo_dir.glob("*.csv")):
        em, date = f.stem.rsplit("__", 1)
        if em in emails:
            x = pd.read_csv(f)
            row = dict(zip(x["program_question_id"], x["latest_answer"].fillna("").astype(str)))
            clin.append((date, {k: " ".join(row.get(k, "").split()) for k in ["Dose", "RouteOfAdmin", "ClinicDetails", "AdditionalInfo"]}))

    # tables
    rec_rows = ""
    for _, r in me.iterrows():
        rec_rows += ("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in [
            STEP_LABELS[int(r["step"])].replace("\n", " "), f"{r['datetime']:%Y-%m-%d %H:%M}", f"{r['duration_s'] / 60:.1f}",
            f"{r['n_epochs_used']:.0f}", f"{r['pct_annotated_bad']:.0f}%", r["bad_channels"] if isinstance(r["bad_channels"], str) else "-",
            fmt(r["cognition"], True), fmt(r["emotion"], True), fmt(r["awareness"], False), fmt(r["awareness_lzc"], False)]) + "</tr>")
    bc = me["bad_channels"].fillna("")
    ch_txt = ", ".join(f"{c} {int(bc.str.contains(c).sum())}/{len(me)}" for c in config.MUSE_CHANNEL_ORDER)
    missing = [STEP_LABELS[i].replace("\n", " ") for i in range(11) if i not in set(me["step"].astype(int))]
    dup = me["module_id"].value_counts()
    irregular = []
    if len(dup[dup > 1]):
        irregular.append("repeated steps: " + ", ".join(STEP_LABELS[MODULE_ORDER.index(m)].replace("\n", " ") for m in dup[dup > 1].index))
    if (np.diff(me["step"].to_numpy()) < 0).any():
        irregular.append("dates not in protocol order")
    if missing:
        irregular.append("no recording for " + ", ".join(missing))
    clin_html = "".join(
        f"<p>Form dated {html.escape(dte)}: dose “{html.escape(c['Dose'] or 'not given')}”, route “{html.escape(c['RouteOfAdmin'] or 'not given')}”, "
        f"clinic/referral “{html.escape(c['ClinicDetails'] or 'not given')}”, additional “{html.escape(c['AdditionalInfo'] or 'none')}”.</p>"
        for dte, c in clin) or "<p>No clinic-info form found.</p>"

    first, last = me["datetime"].min(), me["datetime"].max()
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Study report {html.escape(label)}</title><style>{CSS}</style></head><body><div class="page">
<div class="top"><b style="color:#3b3045;font-size:20px">PSYNAUTICS</b></div>
<div class="banner"><h1>INDIVIDUAL STUDY REPORT</h1><div>{html.escape(label)} &nbsp;|&nbsp; RECORDINGS: {first:%Y-%m-%d} to {last:%Y-%m-%d}</div></div>
<div class="info"><b>RECORDINGS:</b> {len(me)} &nbsp;|&nbsp; <b>PROTOCOL STEPS COVERED:</b> {me['step'].nunique()} of 11 &nbsp;|&nbsp; <b>TREATMENT DAY:</b> {tx_date or 'n/a'} &nbsp;|&nbsp; <b>DEVICE:</b> Muse S (4-channel EEG) &nbsp;|&nbsp; <b>GENERATED:</b> {dt.date.today():%Y-%m-%d}</div>
<div class="note"><b>Read this first.</b> This is one person's data. Scores are percentile ranks against all other participants' recordings (50 = typical); they are not a measure of how good or bad anything is, and no statistical tests are possible with n=1. Data quality for this participant is limited — see the Data quality section.</div>
<div class="sec">Brain Activity</div>
<div class="row"><h3>Longitudinal metrics</h3>
<div class="txt"><b>Key EEG findings:</b> Performance (AF8 broadband), Mood (AF7 alpha) and Awareness (permutation entropy, all 4 channels) across the protocol, as percentiles of the rest of the cohort. Faint dots are individual recordings; lines join each step's average. Dates above the chart are each step's first recording. Gaps mean the marker could not be computed (a required electrode was flagged bad).</div>
<div><img src="data:image/png;base64,{long_png}"></div></div>
<div class="row"><h3>Pre / post comparison</h3>
<div class="txt">Pre = baseline sessions (days 1–5 and the day-6 pre-treatment recording); post = the four post-treatment sessions. <br><br>{'<br>'.join(pct_txt(info, m[0]) for m in MAIN)}</div>
<div><img src="data:image/png;base64,{pp_png}"></div></div>
<div class="row"><h3>Treatment metrics</h3>
<div class="txt">Each marker over the treatment-day recording, per clean 4-second window, in raw units (Performance and Mood in µV²/Hz on a log axis; Awareness 0–1). Windows with dropout, artifact or muscle contamination are removed, so the line has gaps.<br><br>{html.escape(tx_note)}</div>
<div>{f'<img src="data:image/png;base64,{tx_png}">' if tx_png else '<p>No treatment-day recording.</p>'}</div></div>
<div class="sec">Recordings</div>
<div class="row" style="grid-template-columns:1fr"><table><tr><th>Step</th><th>Date</th><th>Min</th><th>Usable windows</th><th>Time lost</th><th>Bad channels</th><th>Performance</th><th>Mood</th><th>Awareness</th><th>Lempel-Ziv</th></tr>{rec_rows}</table>
<div class="txt">Performance and Mood in µV²/Hz; Awareness and Lempel-Ziv are 0–1; n/a = not computable because a required channel was flagged bad. Lempel-Ziv is an extra companion measure, not part of the Awareness definition.</div></div>
<div class="sec">Data quality</div>
<div class="row" style="grid-template-columns:1fr"><div class="txt">
<p>Channels flagged bad (recordings affected): {ch_txt}.</p>
<p>Median share of each recording lost to dropout, artifact or muscle activity: {me['pct_annotated_bad'].median():.0f}% (rest of cohort: {others['pct_annotated_bad'].median():.0f}%).</p>
<p>Usable recordings — Performance: {int(me['cognition'].notna().sum())}, Mood: {int(me['emotion'].notna().sum())}, Awareness: {int(me['awareness'].notna().sum())} of {len(me)}. Awareness needs all four electrodes clean at once.</p>
{('<p>Protocol irregularities: ' + '; '.join(irregular) + '.</p>') if irregular else ''}</div></div>
<div class="sec">Self-reported treatment details</div>
<div class="row" style="grid-template-columns:1fr"><div class="txt">{clin_html}<p><i>Free text, self-reported and unverified; not used in any calculation.</i></p></div></div>
<div class="sec">How to read this</div>
<div class="row" style="grid-template-columns:1fr"><div class="txt"><ul>
<li>Performance and Mood come from frontal dry electrodes and are partly contaminated by facial/jaw muscle activity that the pipeline only partly removes; a change may reflect muscle tension rather than brain activity.</li>
<li>Awareness (permutation entropy) is very stable within a person and moves little, so small changes are hard to interpret.</li>
<li>When ear electrodes (TP9/TP10) are often bad, comparisons with the cohort are against cleaner data than this person's.</li>
<li>Questionnaire-based sections (personality, emotionality) are not included in this version.</li></ul></div></div>
</div></body></html>"""
    out = a.outdir / f"{a.subject}_study_report.html"
    out.write_text(doc, encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
