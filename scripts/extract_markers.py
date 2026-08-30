#!/usr/bin/env python
"""
Batch-extract Cognition/Emotion/Awareness markers from preprocessed Muse
`_clean_raw.fif` files (produced by batch_preprocess.py), writing one row
per recording to a summary CSV.

Usage:
    .venv/bin/python scripts/extract_markers.py \
        --input-dir output/batch_full/clean_raw \
        --outdir output/batch_full
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import mne
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from muse_pipeline.markers import extract_markers  # noqa: E402

# Matches the naming batch_preprocess.py writes: {subject_code}_{module_id}_{session_hash}_clean_raw.fif
FIF_FILENAME_RE = re.compile(
    r"^(?P<subject_code>.+?)_(?P<module_id>PSYNAUTICS_[A-Z_0-9]+|UNKNOWN)_(?P<session_hash>[0-9a-f]+)_clean_raw\.fif$"
)


def parse_fif_filename(name: str) -> dict | None:
    m = FIF_FILENAME_RE.match(name)
    return m.groupdict() if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of *_clean_raw.fif files.")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*_clean_raw.fif"))
    print(f"Found {len(files)} preprocessed file(s) under {args.input_dir}")

    rows = []
    for i, path in enumerate(files, 1):
        meta = parse_fif_filename(path.name)
        if meta is None:
            print(f"[{i}/{len(files)}] {path.name}  SKIPPED: filename doesn't match expected pattern")
            continue

        print(f"[{i}/{len(files)}] {path.name}  (subject={meta['subject_code']}, module={meta['module_id']})")
        result = {**meta, "file": path.name, "error": None}
        try:
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
            report = extract_markers(raw)
            result.update(
                cognition=report.cognition,
                emotion=report.emotion,
                awareness=report.awareness,
                n_epochs_total=report.n_epochs_total,
                n_epochs_used=report.n_epochs_used,
                bad_channels=",".join(report.bad_channels) if report.bad_channels else None,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad file must not kill the batch
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {result['error']}")
            traceback.print_exc(limit=1)

        rows.append(result)

    summary = pd.DataFrame(rows)
    summary_path = args.outdir / "marker_summary.csv"
    summary.to_csv(summary_path, index=False)

    n_ok = int(summary["error"].isna().sum()) if len(summary) else 0
    n_fail = len(summary) - n_ok
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed. Marker summary -> {summary_path}")
    if n_ok:
        ok = summary.loc[summary["error"].isna()]
        for marker in ["cognition", "emotion", "awareness"]:
            n_avail = int(ok[marker].notna().sum())
            print(f"  {marker}: available for {n_avail}/{n_ok} recordings")


if __name__ == "__main__":
    main()
