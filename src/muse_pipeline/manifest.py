"""
Cross-reference raw EEG filenames against the Psynautics export manifest
(`eeg_summary.csv`) and the confidential participant key, so batch runs can
attach reliable participant/module/date metadata instead of guessing from
filename conventions that vary between export tools.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# The 32-char hex session hash is embedded consistently at the end of every
# raw filename across the naming variants seen so far (both the flattened
# "2026_Feb_files_eeg_<email>_..._<hash>_eeg.csv" downloads and the
# "<email>/<date>_..._<hash>_eeg.csv" paths referenced in the manifest's
# file_path column), so matching on it is more robust than parsing the full
# filename structure.
SESSION_HASH_RE = re.compile(r"([0-9a-f]{32})_eeg\.csv$", re.IGNORECASE)

REQUIRED_MANIFEST_COLUMNS = {"user_id", "session_id", "email", "datetime", "module_id", "file_path"}


def extract_session_hash(filename: str) -> str | None:
    m = SESSION_HASH_RE.search(filename)
    return m.group(1) if m else None


def load_manifest(manifest_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_csv}: missing expected columns {missing}")
    return df.set_index("session_id", drop=False)


def load_participant_key(key_csv: str | Path) -> dict[str, str]:
    """Map email -> pseudonymous subject code (e.g. 'Sub_01').

    The key file has no header row: column 0 is the subject code, column 1
    the email. Used only to label derived outputs pseudonymously -- never
    write the raw key file's contents elsewhere.
    """
    df = pd.read_csv(key_csv, header=None, names=["subject_code", "email"])
    return dict(zip(df["email"], df["subject_code"]))
