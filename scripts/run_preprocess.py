#!/usr/bin/env python
"""
Preprocess a single raw Muse EEG CSV export.

Usage:
    .venv/bin/python scripts/run_preprocess.py <path/to/..._eeg.csv> [--outdir output]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from muse_pipeline import qc  # noqa: E402
from muse_pipeline.io import load_muse_csv  # noqa: E402
from muse_pipeline.preprocess import preprocess_raw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent.parent / "output")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.csv_path.stem

    print(f"Loading {args.csv_path.name} ...")
    raw, load_report = load_muse_csv(args.csv_path)
    print(
        f"  {load_report.n_raw_samples} raw samples, {load_report.duration_s:.1f}s duration\n"
        f"  {load_report.n_dropouts} dropout(s) totaling {load_report.total_dropout_s:.2f}s "
        f"({load_report.pct_dropout:.2f}% of recording), max gap {load_report.max_gap_s:.2f}s\n"
        f"  filename metadata: {load_report.metadata}"
    )

    raw_before = raw.copy()
    print("Filtering + flagging artifacts ...")
    raw_clean, prep_report = preprocess_raw(raw)
    print(
        f"  {prep_report.n_artifact_windows} artifact window(s) flagged "
        f"(on top of {prep_report.n_dropout_windows} dropout region(s))\n"
        f"  ~{prep_report.pct_annotated_bad:.2f}% of recording annotated bad (upper bound, overlaps not merged)\n"
        f"  per-channel flagged windows: {prep_report.per_channel_flagged_windows}"
    )

    fif_path = args.outdir / f"{stem}_clean_raw.fif"
    raw_clean.save(fif_path, overwrite=True, verbose=False)
    print(f"Saved cleaned Raw -> {fif_path}")

    psd_path = args.outdir / f"{stem}_psd_before_after.png"
    qc.plot_psd_before_after(raw_before, raw_clean, psd_path)
    trace_path = args.outdir / f"{stem}_trace_annotated.png"
    qc.plot_trace_with_annotations(raw_clean, trace_path)
    print(f"Saved QC plots -> {psd_path.name}, {trace_path.name}")

    report_path = args.outdir / f"{stem}_report.json"
    report_path.write_text(
        json.dumps(
            {
                "load": {
                    "n_raw_samples": load_report.n_raw_samples,
                    "duration_s": load_report.duration_s,
                    "n_dropouts": load_report.n_dropouts,
                    "total_dropout_s": load_report.total_dropout_s,
                    "pct_dropout": load_report.pct_dropout,
                    "max_gap_s": load_report.max_gap_s,
                    "metadata": load_report.metadata,
                },
                "preprocess": {
                    "pct_annotated_bad": prep_report.pct_annotated_bad,
                    "n_artifact_windows": prep_report.n_artifact_windows,
                    "n_dropout_windows": prep_report.n_dropout_windows,
                    "per_channel_flagged_windows": prep_report.per_channel_flagged_windows,
                },
            },
            indent=2,
        )
    )
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
