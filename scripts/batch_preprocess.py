#!/usr/bin/env python
"""
Batch-preprocess all raw Muse EEG CSV exports found under a directory tree,
cross-referencing the Psynautics export manifest for participant/module
metadata, and producing one aggregate QC summary CSV across the whole
cohort. A single bad file logs an error and is skipped -- it does not abort
the run.

Usage:
    .venv/bin/python scripts/batch_preprocess.py \
        --input-dir /path/to/raw/eeg/files \
        --manifest "/Users/sonyaluchanskaya/Desktop/Psynautics/2026_Feb_files_eeg_eeg_summary.csv" \
        --participant-key "/Users/sonyaluchanskaya/Desktop/Psynautics/participant_key_CONFIDENTIAL.csv" \
        --eeg-only
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from muse_pipeline.io import load_muse_csv  # noqa: E402
from muse_pipeline.manifest import extract_session_hash, load_manifest, load_participant_key  # noqa: E402
from muse_pipeline.preprocess import preprocess_raw  # noqa: E402

# EEG resting-state modules relevant to the Cognition/Emotion/Awareness
# markers -- excludes sleep, questionnaire-only, and clinic-info modules
# that also appear in the manifest.
EEG_RESTING_MODULES = {
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
}


# Recordings excluded from the cohort: implausible durations for their module
# (a "baseline" recording is supposed to run ~5-10 min) that dominate the
# aggregate pct_dropout/pct_annotated_bad stats and are almost certainly a
# device left running unattended rather than real resting-state EEG.
EXCLUDED_SESSION_HASHES = {
    "a9d0e537da2043ab941fc34e30ae7638": "Sub_36 BASELINE_EEG_2, 20.5hr duration, 99.0% dropout",
    "48cdf8d5dbee479083fbca6517fe24c2": "Sub_51 BASELINE_EEG_5, 3.0hr duration",
    "4f310879923e49c6a6345562049ecb40": "Sub_16 BASELINE_EEG_1, 40% dropout across 8 gaps (repeated BLE disconnects)",
}



def find_raw_csvs(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*_eeg.csv") if "summary" not in p.name.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory to search recursively for *_eeg.csv files.")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to the eeg_summary.csv export manifest.")
    parser.add_argument("--participant-key", type=Path, default=None, help="Path to participant_key_CONFIDENTIAL.csv.")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent.parent / "output" / "batch")
    parser.add_argument(
        "--eeg-only",
        action="store_true",
        help="Skip files whose manifest module_id is not a resting-state EEG module (sleep/questionnaire/clinic-info rows).",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    fif_dir = args.outdir / "clean_raw"
    fif_dir.mkdir(exist_ok=True)
    # Clear stale outputs from prior runs -- otherwise a recording that used
    # to succeed and later gets added to EXCLUDED_SESSION_HASHES (or starts
    # failing for some other reason) leaves its old .fif sitting here
    # indefinitely, silently picked up by anything that reads this directory.
    for stale in fif_dir.glob("*_clean_raw.fif"):
        stale.unlink()

    manifest = load_manifest(args.manifest) if args.manifest else None
    participant_key = load_participant_key(args.participant_key) if args.participant_key else {}

    files = find_raw_csvs(args.input_dir)
    print(f"Found {len(files)} raw EEG CSV file(s) under {args.input_dir}")
    if manifest is None:
        print("WARNING: no --manifest given, output will be labeled by filename only (no subject/module metadata).")

    rows = []
    for i, path in enumerate(files, 1):
        session_hash = extract_session_hash(path.name)
        if session_hash in EXCLUDED_SESSION_HASHES:
            print(f"[{i}/{len(files)}] {path.name}  EXCLUDED: {EXCLUDED_SESSION_HASHES[session_hash]}")
            continue
        manifest_row = (
            manifest.loc[session_hash].to_dict()
            if (manifest is not None and session_hash in manifest.index)
            else None
        )
        module_id = manifest_row["module_id"] if manifest_row else None

        if args.eeg_only and manifest_row is not None and module_id not in EEG_RESTING_MODULES:
            continue

        email = manifest_row["email"] if manifest_row else None
        subject_code = participant_key.get(email, "unmatched" if manifest_row is None else email)

        print(f"[{i}/{len(files)}] {path.name}  (subject={subject_code}, module={module_id})")

        result = {
            "file": path.name,
            "session_hash": session_hash,
            "subject_code": subject_code,
            "module_id": module_id,
            "datetime": manifest_row["datetime"] if manifest_row else None,
            "manifest_matched": manifest_row is not None,
            "error": None,
        }
        try:
            raw, load_report = load_muse_csv(path)
            raw_clean, prep_report = preprocess_raw(raw)
            bad_channels = prep_report.globally_bad_channels

            out_name = f"{subject_code}_{module_id or 'UNKNOWN'}_{session_hash or path.stem}_clean_raw.fif"
            raw_clean.save(fif_dir / out_name, overwrite=True, verbose=False)

            result.update(
                duration_s=load_report.duration_s,
                n_nan_dropped=load_report.n_nan_dropped,
                n_dropouts=load_report.n_dropouts,
                pct_dropout=load_report.pct_dropout,
                max_gap_s=load_report.max_gap_s,
                bad_channels=",".join(bad_channels) if bad_channels else None,
                n_artifact_windows=prep_report.n_artifact_windows,
                pct_annotated_bad=prep_report.pct_annotated_bad,
                output_file=out_name,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad file must not kill the batch
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {result['error']}")
            traceback.print_exc(limit=1)

        rows.append(result)

    summary = pd.DataFrame(rows)
    summary_path = args.outdir / "batch_qc_summary.csv"
    summary.to_csv(summary_path, index=False)

    n_ok = int(summary["error"].isna().sum()) if len(summary) else 0
    n_fail = len(summary) - n_ok
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed. QC summary -> {summary_path}")
    if n_ok:
        ok = summary.loc[summary["error"].isna()]
        print(ok[["pct_dropout", "pct_annotated_bad", "duration_s"]].describe().to_string())


if __name__ == "__main__":
    main()
