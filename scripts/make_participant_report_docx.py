#!/usr/bin/env python
"""
Individual participant report as a Word document, in the same session-report
layout as make_participant_report_html.py (banner, session-info bar, grey
section bars, text-left / chart-right rows). Shares all data and charts with
the HTML version, so the two never disagree.

Each marker is shown in its own units and compared ONLY with this
participant's own baseline sessions -- nothing is ranked against others.

Usage:
    .venv/bin/python scripts/make_participant_report_docx.py --subject Sub_53 \
        --label "Gerard Walsh" --outdir output/participant_reports
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_participant_report import fmt  # noqa: E402
from make_participant_report_html import MAIN, SHORT, STEP_LABELS, collect  # noqa: E402

PURPLE, GREY = "3B3045", RGBColor(0x77, 0x77, 0x77)
FULL = 7.3  # usable page width in inches


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear")
    s.set(qn("w:fill"), fill)
    tcPr.append(s)


def fix_table(t, widths_in):
    """Fixed layout + explicit grid/cell widths, so Word and Quick Look honour the column sizes."""
    t.autofit = False
    tblPr = t._tbl.tblPr
    w = tblPr.find(qn("w:tblW"))
    if w is None:
        w = OxmlElement("w:tblW")
        tblPr.append(w)
    w.set(qn("w:type"), "dxa")
    w.set(qn("w:w"), str(int(sum(widths_in) * 1440)))
    for col, wi in zip(t.columns, widths_in):
        col.width = Inches(wi)
    for row in t.rows:
        for cell, wi in zip(row.cells, widths_in):
            cell.width = Inches(wi)


def shaded_par(doc, fill, parts, align=None, before=0, after=0):
    """Full-width shaded band: a one-cell table at 100% width (cell shading renders in Word and Quick Look)."""
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    w = tblPr.find(qn("w:tblW"))
    w.set(qn("w:type"), "pct")
    w.set(qn("w:w"), "5000")
    cell = t.rows[0].cells[0]
    cell.width = Inches(FULL)
    shade(cell, fill)
    return cell_paras(cell, parts, align, before, after)


def cell_paras(cell, parts, align, before, after):
    """Write runs into the cell, starting a new paragraph for every ('br', ...) part."""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(before), Pt(after)
    if align is not None:
        p.alignment = align
    for text, kw in parts:
        if kw.get("newpar"):
            p = cell.add_paragraph()
            p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(0), Pt(after)
            if align is not None:
                p.alignment = align
            kw = {k: v for k, v in kw.items() if k != "newpar"}
        run(p, text, **kw)
    return cell


def run(p, text, bold=False, italic=False, size=None, color=None):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color is not None:
        r.font.color.rgb = color
    return r


def spacer(doc, pts=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    run(p, " ", size=pts)


def section_bar(doc, title):
    shaded_par(doc, "EEEEEE", [(title, {"size": 15})], before=5, after=5)
    spacer(doc)


def chart_row(doc, heading, text_parts, png_b64, caption=None):
    """Heading, then text on the left (2.1in) and chart on the right (5.0in)."""
    h = doc.add_paragraph()
    h.paragraph_format.space_after = Pt(2)
    run(h, heading.upper(), size=11)
    t = doc.add_table(rows=1, cols=2)
    fix_table(t, [2.1, 5.2])
    left, right = t.rows[0].cells
    p = left.paragraphs[0]
    for kind, txt in text_parts:
        if kind == "br":
            p = left.add_paragraph()
            continue
        run(p, txt, bold=(kind == "b"), size=8.5, color=GREY)
    rp = right.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if png_b64:
        rp.add_run().add_picture(io.BytesIO(base64.b64decode(png_b64)), width=Inches(5.0))
    else:
        run(rp, "No treatment-day recording.", italic=True, size=9)
    spacer(doc, 6)


def data_table(doc, headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, hd in enumerate(headers):
        c = t.rows[0].cells[i]
        c.width = Inches(widths[i])
        shade(c, "E6EBF5")
        c.paragraphs[0].text = ""
        run(c.paragraphs[0], hd, bold=True, size=7)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].width = Inches(widths[i])
            cells[i].paragraphs[0].text = ""
            run(cells[i].paragraphs[0], str(v), size=7)
            if ri % 2:
                shade(cells[i], "FAFAFA")
    fix_table(t, widths)


def bullet(doc, text, size=9):
    p = doc.add_paragraph(style="List Bullet")
    run(p, text, size=size)


def cmp_parts(info):
    parts = []
    for col, _n, log, _c, _u in MAIN:
        b_, p_ = info[(col, "Baseline")], info[(col, "Post")]
        parts += [("br", ""), ("b", f"{SHORT[col]}: ")]
        if b_[0] is None or p_[0] is None:
            parts.append(("t", f"cannot compare (usable sessions: {b_[1]} baseline, {p_[1]} post)."))
        else:
            chg = f"x{p_[0] / b_[0]:.2f}" if log else f"{p_[0] - b_[0]:+.3f}"
            parts.append(("t", f"baseline {fmt(b_[0], log)} (n={b_[1]}), post {fmt(p_[0], log)} (n={p_[1]}), change {chg}."))
    return parts


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
    label, me, info = c["label"], c["me"], c["info"]

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.6)
    sec.top_margin = sec.bottom_margin = Inches(0.6)

    p = doc.add_paragraph()
    run(p, "PSYNAUTICS", bold=True, size=15, color=RGBColor(0x3B, 0x30, 0x45))

    first, last = me["datetime"].min(), me["datetime"].max()
    white = RGBColor(255, 255, 255)
    shaded_par(doc, PURPLE, [
        ("INDIVIDUAL STUDY REPORT", {"bold": True, "size": 22, "color": white}),
        (f"{label}  |  RECORDINGS: {first:%Y-%m-%d} to {last:%Y-%m-%d}", {"size": 10, "color": white, "newpar": True}),
    ], align=WD_ALIGN_PARAGRAPH.CENTER, before=14, after=14)
    spacer(doc, 6)

    p = doc.add_paragraph()
    for lab, val_ in [("RECORDINGS: ", str(len(me))), ("PROTOCOL STEPS COVERED: ", f"{me['step'].nunique()} of 11"),
                      ("TREATMENT DAY: ", c["tx_date"] or "n/a"), ("DEVICE: ", "Muse S (4-channel EEG)"),
                      ("GENERATED: ", f"{dt.date.today():%Y-%m-%d}")]:
        run(p, lab, bold=True, size=9)
        run(p, val_ + "   |   " if lab != "GENERATED: " else val_, size=9)

    shaded_par(doc, "FFF7E6", [
        ("Read this first. ", {"bold": True, "size": 9}),
        ("This is one person's data. Each marker is shown in its own units and compared only with this "
         "participant's own baseline sessions. Nothing here is ranked or compared against other people, none of it says what is "
         "good or bad, and no statistical tests are possible with n=1. Data quality for this participant is limited; see the "
         "Data quality section.", {"size": 9})],
        before=5, after=5)
    spacer(doc, 6)

    section_bar(doc, "Brain Activity")
    chart_row(doc, "Longitudinal metrics", [
        ("b", "Key EEG findings: "),
        ("t", "Performance (AF8 broadband), Mood (AF7 alpha) and Awareness (permutation entropy, all 4 channels) across the "
              "protocol, each in its own units (Performance and Mood on a log axis). Dots are individual recordings and the line joins "
              "each step's average. The shaded band is this participant's own baseline range and the dotted line their baseline mean; "
              "the grey column marks the treatment-day session. Dates above the chart are each step's first recording. Gaps mean the "
              "marker could not be computed (a required electrode was flagged bad)."),
    ], c["long_png"])
    chart_row(doc, "Pre / post comparison", [
        ("t", "Pre = baseline sessions (days 1-5 and the day-6 pre-treatment recording); post = the four post-treatment sessions. Black dots are individual sessions; the thick line is the average "
              "(geometric for Performance and Mood); error bars show +/-1 standard deviation across sessions (multiplicative on the "
              "log axis), and are omitted when there is only one session.")
    ] + cmp_parts(info), c["pp_png"])
    chart_row(doc, "Treatment metrics", [
        ("t", "Each marker over the treatment-day recording, per clean 4-second window, in raw units (Performance and Mood in "
              "uV^2/Hz on a log axis; Awareness 0-1). Windows with dropout, artifact or muscle contamination are removed, so the "
              "line has gaps."), ("br", ""), ("t", c["tx_note"]),
    ], c["tx_png"])

    section_bar(doc, "Recordings")
    rows = []
    for _, r in me.iterrows():
        rows.append([STEP_LABELS[int(r["step"])].replace("\n", " "), f"{r['datetime']:%Y-%m-%d %H:%M}", f"{r['duration_s'] / 60:.1f}",
                     f"{r['n_epochs_used']:.0f}", f"{r['pct_annotated_bad']:.0f}%",
                     r["bad_channels"] if isinstance(r["bad_channels"], str) else "-",
                     fmt(r["cognition"], True), fmt(r["emotion"], True), fmt(r["awareness"], False), fmt(r["awareness_lzc"], False)])
    data_table(doc, ["Step", "Date", "Min", "Usable windows", "Time lost", "Bad channels", "Performance", "Mood", "Awareness", "Lempel-Ziv"],
               rows, [0.8, 1.0, 0.4, 0.6, 0.55, 0.9, 0.85, 0.85, 0.7, 0.65])
    p = doc.add_paragraph()
    run(p, "Performance and Mood in uV^2/Hz; Awareness and Lempel-Ziv are 0-1; n/a = not computable because a required channel was "
           "flagged bad. Lempel-Ziv is an extra companion measure, not part of the Awareness definition.", italic=True, size=8, color=GREY)

    section_bar(doc, "Data quality")
    bullet(doc, f"Channels flagged bad (recordings affected): {c['ch_txt']}.")
    bullet(doc, f"Median share of each recording lost to dropout, artifact or muscle activity: {me['pct_annotated_bad'].median():.0f}%.")
    bullet(doc, f"Usable recordings: Performance {int(me['cognition'].notna().sum())}, Mood {int(me['emotion'].notna().sum())}, "
                f"Awareness {int(me['awareness'].notna().sum())} of {len(me)}. Awareness needs all four electrodes clean at once.")
    if c["irregular"]:
        bullet(doc, "Protocol irregularities: " + "; ".join(c["irregular"]) + ".")

    section_bar(doc, "Self-reported treatment details")
    if c["clin"]:
        for dte, f in c["clin"]:
            bullet(doc, f"Form dated {dte}: dose \"{f['Dose'] or 'not given'}\", route \"{f['RouteOfAdmin'] or 'not given'}\", "
                        f"clinic/referral \"{f['ClinicDetails'] or 'not given'}\", additional \"{f['AdditionalInfo'] or 'none'}\".")
        p = doc.add_paragraph()
        run(p, "Free text, self-reported and unverified; not used in any calculation.", italic=True, size=8, color=GREY)
    else:
        bullet(doc, "No clinic-info form found.")

    section_bar(doc, "How to read this")
    for t in [
        "Performance and Mood come from frontal dry electrodes and are partly contaminated by facial/jaw muscle activity that the "
        "pipeline only partly removes; a change may reflect muscle tension rather than brain activity.",
        "Awareness (permutation entropy) is very stable within a person and moves little, so small changes are hard to interpret.",
        "Questionnaire-based sections (personality, emotionality) are not included in this version.",
    ]:
        bullet(doc, t)

    out = a.outdir / f"{a.subject}_session_report.docx"
    doc.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
