#!/usr/bin/env python
"""
Individual participant report as a single self-contained HTML page, laid out
like the Psynautics/Muse session reports: banner, session-info bar, then
"Longitudinal metrics", "Pre/post comparison" and "Treatment metrics" sections
with explanatory text on the left and a chart on the right.

Each marker is shown in its own units and compared ONLY with this
participant's own baseline sessions -- nothing is ranked or compared against
other participants. (The app's own 0-18 score transform is not used because
it is not available to this pipeline.)

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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from make_participant_report import (  # noqa: E402
    BASELINE, MARKERS, MODULE_ORDER, POST, TREATMENT, fmt, load_all,
)
from muse_pipeline import config  # noqa: E402
from muse_pipeline.markers import _make_epochs, epoch_psd_uv2  # noqa: E402

# Charts are drawn at about the width they are displayed (6-7 in), so these sizes are close to what the reader sees.
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10})

MAIN = MARKERS[:3]  # Performance, Mood, Awareness (LZC is a companion, shown in the table only)
STEP_LABELS = ["day 1", "day 2", "day 3", "day 4", "day 5", "day 6\n(pre)", "day 6\ntreatment", "post 1", "post 2", "post 3", "post 4"]
AXIS_LABELS = ["day 1", "day 2", "day 3", "day 4", "day 5", "day 6 pre", "day 6 treatment", "post 1", "post 2", "post 3", "post 4"]
SHORT = {"cognition": "Performance", "emotion": "Mood", "awareness": "Awareness"}


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def log_axis(ax) -> None:
    """Log y-axis with plain numbers (0.5, 1, 2) instead of 5 x 10^-1."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def geo(vals, log: bool) -> float:
    """Mean of a marker's values; geometric (log-space) for the power markers, which span orders of magnitude."""
    v = np.asarray(vals, dtype=float)
    return float(10 ** np.log10(v).mean()) if log else float(v.mean())


def fig_longitudinal(me: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.8), sharex=True)
    tx_i = MODULE_ORDER.index(TREATMENT)
    for ax, (col, name, log, color, unit) in zip(axes, MAIN):
        d = me.dropna(subset=[col])
        ax.axvspan(tx_i - 0.5, tx_i + 0.5, color="#EEEEEE", zorder=0)
        if d.empty:
            ax.text(0.5, 0.5, f"{SHORT[col]}: not available in any recording (a required channel was flagged bad)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#888", wrap=True)
            ax.set_yticks([])
            continue
        base = d[d["module_id"].isin(BASELINE)][col]
        if len(base):
            ax.axhspan(base.min(), base.max(), color=color, alpha=0.12, label="Own baseline range")
            ax.axhline(geo(base, log), color=color, linestyle=":", linewidth=1, label="Own baseline mean")
        ax.scatter(d["step"], d[col], color=color, alpha=0.4, s=20, label="Recordings")
        g = d.groupby("step")[col].apply(lambda v: geo(v, log))
        ax.plot(g.index, g.values, "-o", color=color, markersize=5, label="Step average")
        if log:
            log_axis(ax)
        else:
            ax.set_ylim(0, 1)  # bounded 0-1 measure: show the full scale so tiny differences don't look dramatic
        ax.set_ylabel(f"{SHORT[col]}\n({unit or '0-1'})")
    axes[-1].set_xlim(-0.5, 10.5)
    axes[-1].set_xticks(range(11))
    axes[-1].set_xticklabels(AXIS_LABELS, rotation=45, ha="right", rotation_mode="anchor")
    first = me.sort_values("datetime").groupby("step")["datetime"].first()
    top = axes[0].secondary_xaxis("top")
    top.set_xticks(list(first.index))
    top.set_xticklabels([f"{t:%m-%d %H:%M}" for t in first], rotation=90, fontsize=9)
    handles, labels = next(((h, l) for h, l in (a_.get_legend_handles_labels() for a_ in axes) if h), ([], []))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    return b64(fig)


def fig_prepost(me: pd.DataFrame) -> tuple[str, dict]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.6))
    info = {}
    rng = np.random.default_rng(0)
    for ax, (col, name, log, color, unit) in zip(axes, MAIN):
        for j, (grp, mods, alpha) in enumerate([("Baseline", BASELINE, 1.0), ("Post", POST, 0.55)]):
            v = me[me["module_id"].isin(mods)][col].dropna()
            info[(col, grp)] = (geo(v, log) if len(v) else None, len(v))
            if len(v):
                mean = geo(v, log)
                ax.hlines(mean, j - 0.3, j + 0.3, color=color, linewidth=4, alpha=alpha)
                if len(v) > 1:  # +/-1 SD across sessions (multiplicative for the log-scale markers)
                    if log:
                        sd = np.log10(v).std(ddof=1)
                        lo, hi = mean / 10 ** sd, mean * 10 ** sd
                    else:
                        lo, hi = mean - v.std(ddof=1), mean + v.std(ddof=1)
                    ax.errorbar(j, mean, yerr=[[mean - lo], [hi - mean]], color="k", capsize=5, linewidth=1.3, zorder=6)
                ax.scatter(j + rng.uniform(-0.12, 0.12, len(v)), v, color="k", s=14, zorder=5)
                ax.text(j, 1.02, f"n={len(v)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=10)
            else:
                ax.text(j, 0.5, "n/a", transform=ax.get_xaxis_transform(), ha="center", fontsize=10, color="#888")
        if log:
            log_axis(ax)
        else:
            ax.set_ylim(0, 1)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Post"])
        ax.set_title(f"{SHORT[col]}\n({unit or '0-1'})", pad=18)
    fig.tight_layout()
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
        psd, _ = epoch_psd_uv2(ep, ch, lo, hi)
        return psd.mean(axis=-1)

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
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.6), sharex=True)
    for ax, (col, name, log, color, unit) in zip(axes, MAIN):
        v = series.get(col)
        if v is None:
            ax.text(0.5, 0.5, f"{SHORT[col]}: not available (a required channel was flagged bad)", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#888", wrap=True)
            ax.set_yticks([])
        else:
            ax.plot(series["t"], v, "-o", color=color, markersize=3, linewidth=0.8)
            if log:
                log_axis(ax)
            ax.set_ylabel(f"{SHORT[col]}\n({unit or '0-1'})")
    axes[-1].set_xlabel("Minutes into the treatment-day recording\n(each point = one clean 4 s window)")
    axes[-1].set_xlim(0, max(series["minutes"], 1))
    fig.tight_layout()
    return b64(fig)


CSS = """
body{margin:0;background:#f4f4f4;font-family:-apple-system,Helvetica,Arial,sans-serif;color:#222}
.page{max-width:1100px;margin:20px auto;background:#fff}
.top{padding:22px 36px;color:#888;font-size:15px}
.banner{background:#3b3045;color:#fff;text-align:center;padding:30px 20px}
.banner h1{margin:0 0 8px;font-size:30px;letter-spacing:.5px}
.banner div{font-size:16px}
.info{padding:22px 36px;font-size:16px}
.sec{background:#eeeeee;padding:14px 36px;font-size:25px;margin-top:10px}
.row{display:grid;grid-template-columns:32% 68%;gap:20px;padding:18px 36px}
.row h3{grid-column:1/3;margin:6px 0 0;font-weight:400;font-size:20px;letter-spacing:.3px;text-transform:uppercase}
.txt{font-size:16px;color:#555;line-height:1.55}
.txt b{color:#222}
img{max-width:100%}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th{background:#e6ebf5;text-align:left}
th,td{border:1px solid #ddd;padding:5px 7px}
tr:nth-child(even) td{background:#fafafa}
.note{background:#fff7e6;border-left:4px solid #e0a030;padding:10px 14px;margin:12px 36px;font-size:16px;line-height:1.5}
"""


def cmp_txt(info, col, log):
    b, p = info[(col, "Baseline")], info[(col, "Post")]
    if b[0] is None or p[0] is None:
        return f"<b>{SHORT[col]}:</b> cannot compare (usable sessions: {b[1]} baseline, {p[1]} post)."
    chg = f"x{p[0] / b[0]:.2f}" if log else f"{p[0] - b[0]:+.3f}"
    return (f"<b>{SHORT[col]}:</b> baseline {fmt(b[0], log)} (n={b[1]}), post {fmt(p[0], log)} (n={p[1]}), change {chg}.")


def collect(a):
    a.outdir.mkdir(parents=True, exist_ok=True)
    label = a.label or a.subject

    d = load_all(a.marker_summary, a.qc_summary)
    me = d[d["subject_code"] == a.subject].sort_values("datetime")
    if me.empty:
        raise SystemExit(f"No recordings found for {a.subject}")

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

    return locals()


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
    c = collect(a)
    label, me, long_png, pp_png, info = (c[k] for k in ['label', 'me', 'long_png', 'pp_png', 'info'])
    tx_png, tx_note, tx_date, clin = (c[k] for k in ['tx_png', 'tx_note', 'tx_date', 'clin'])
    rec_rows, ch_txt, irregular, clin_html = (c[k] for k in ['rec_rows', 'ch_txt', 'irregular', 'clin_html'])
    first, last = me["datetime"].min(), me["datetime"].max()
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Study report {html.escape(label)}</title><style>{CSS}</style></head><body><div class="page">
<div class="top"><b style="color:#3b3045;font-size:22px">PSYNAUTICS</b></div>
<div class="banner"><h1>INDIVIDUAL STUDY REPORT</h1><div>{html.escape(label)} &nbsp;|&nbsp; RECORDINGS: {first:%Y-%m-%d} to {last:%Y-%m-%d}</div></div>
<div class="info"><b>RECORDINGS:</b> {len(me)} &nbsp;|&nbsp; <b>PROTOCOL STEPS COVERED:</b> {me['step'].nunique()} of 11 &nbsp;|&nbsp; <b>TREATMENT DAY:</b> {tx_date or 'n/a'} &nbsp;|&nbsp; <b>DEVICE:</b> Muse S (4-channel EEG) &nbsp;|&nbsp; <b>GENERATED:</b> {dt.date.today():%Y-%m-%d}</div>
<div class="note"><b>Read this first.</b> This is one person's data. Each marker is shown in its own units and compared only with this participant's own baseline sessions. Nothing here is ranked or compared against other people, none of it says what is good or bad, and no statistical tests are possible with n=1. Data quality for this participant is limited — see the Data quality section.</div>
<div class="sec">Brain Activity</div>
<div class="row"><h3>Longitudinal metrics</h3>
<div class="txt"><b>Key EEG findings:</b> Performance (AF8 broadband), Mood (AF7 alpha) and Awareness (permutation entropy, all 4 channels) across the protocol, each in its own units (Performance and Mood on a log axis). Dots are individual recordings and the line joins each step's average. The shaded band is this participant's own baseline range and the dotted line their baseline mean; the grey column marks the treatment-day session. Dates above the chart are each step's first recording. Gaps mean the marker could not be computed (a required electrode was flagged bad).</div>
<div><img src="data:image/png;base64,{long_png}"></div></div>
<div class="row"><h3>Pre / post comparison</h3>
<div class="txt">Pre = baseline sessions (days 1–5 and the day-6 pre-treatment recording); post = the four post-treatment sessions. Black dots are individual sessions; the thick line is the average (geometric for Performance and Mood); error bars show ±1 standard deviation across sessions (multiplicative on the log axis), and are omitted when there is only one session. <br><br>{'<br>'.join(cmp_txt(info, m[0], m[2]) for m in MAIN)}</div>
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
<p>Median share of each recording lost to dropout, artifact or muscle activity: {me['pct_annotated_bad'].median():.0f}%.</p>
<p>Usable recordings — Performance: {int(me['cognition'].notna().sum())}, Mood: {int(me['emotion'].notna().sum())}, Awareness: {int(me['awareness'].notna().sum())} of {len(me)}. Awareness needs all four electrodes clean at once.</p>
{('<p>Protocol irregularities: ' + '; '.join(irregular) + '.</p>') if irregular else ''}</div></div>
<div class="sec">Self-reported treatment details</div>
<div class="row" style="grid-template-columns:1fr"><div class="txt">{clin_html}<p><i>Free text, self-reported and unverified; not used in any calculation.</i></p></div></div>
<div class="sec">How to read this</div>
<div class="row" style="grid-template-columns:1fr"><div class="txt"><ul>
<li>Performance and Mood come from frontal dry electrodes and are partly contaminated by facial/jaw muscle activity that the pipeline only partly removes; a change may reflect muscle tension rather than brain activity.</li>
<li>Awareness (permutation entropy) is very stable within a person and moves little, so small changes are hard to interpret.</li>
<li>Questionnaire-based sections (personality, emotionality) are not included in this version.</li></ul></div></div>
</div></body></html>"""
    out = a.outdir / f"{a.subject}_study_report.html"
    out.write_text(doc, encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
