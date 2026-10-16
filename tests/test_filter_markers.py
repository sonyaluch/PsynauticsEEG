#!/usr/bin/env python
"""
Ground-truth tests for everything after loading: the bandpass, PSD unit
scaling, epoch rejection by annotation, and the entropy measures.

Run:  .venv/bin/python tests/test_filter_markers.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import antropy as ant
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from muse_pipeline import config  # noqa: E402
from muse_pipeline.markers import _band_power_uv2, _make_epochs, extract_markers  # noqa: E402
from muse_pipeline.preprocess import preprocess_raw  # noqa: E402

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
FS = 256
rng = np.random.default_rng(0)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


def make_raw(sig_uv: np.ndarray) -> mne.io.RawArray:
    data = np.vstack([sig_uv] * 4) * 1e-6
    return mne.io.RawArray(data, mne.create_info(config.MUSE_CHANNEL_ORDER, FS, "eeg"), verbose=False)


# 1. Bandpass gain at known frequencies (one sine per test, 120 s, no noise)
t = np.arange(FS * 120) / FS
expect = {19: (0.97, 1.05), 21.333: (0.0, 0.1), 24: (0.97, 1.05), 42.667: (0.0, 0.1), 0.3: (0.0, 0.3), 1: (0.9, 1.05), 3: (0.95, 1.05), 10: (0.95, 1.05), 30: (0.95, 1.05), 44: (0.9, 1.05), 60: (0.0, 0.05)}
for f, (lo, hi) in expect.items():
    raw = make_raw(100 * np.sin(2 * np.pi * f * t))
    out, _ = preprocess_raw(raw)
    seg = out.get_data()[0, 10 * FS:-10 * FS] * 1e6  # trim filter edges
    gain = (np.abs(np.fft.rfft(seg))[np.argmin(abs(np.fft.rfftfreq(seg.size, 1 / FS) - f))] * 2 / seg.size) / 100
    check(f"bandpass gain at {f} Hz in [{lo}, {hi}]", lo <= gain <= hi, f"gain {gain:.3f}")

# 2. PSD unit scaling: white noise of sigma 10 uV => one-sided PSD = 2*sigma^2/fs uV^2/Hz
sig = rng.normal(0, 10, FS * 120)
ep = _make_epochs(make_raw(sig), reject_by_annotation=True)
exp_psd = 2 * 10 ** 2 / FS
for band in ["broadband", "mood_alpha"]:
    got = _band_power_uv2(ep, "AF8", band)
    check(f"PSD scale ({band}) matches 2*sigma^2/fs", abs(got / exp_psd - 1) < 0.08, f"{got:.3f} vs {exp_psd:.3f} uV^2/Hz")

# 3. Epoch rejection: 60 s, one BAD annotation 10-12 s => only the 8-12 s epoch overlaps
raw = make_raw(rng.normal(0, 10, FS * 60))
raw.set_annotations(mne.Annotations([10.0], [2.0], ["BAD_artifact"]))
n_all, n_kept = len(_make_epochs(raw, False)), len(_make_epochs(raw, True))
check("annotated epoch is dropped (15 -> 14)", (n_all, n_kept) == (15, 14), f"{n_all} -> {n_kept}")
for name in ["BAD_dropout", "BAD_EMG"]:
    raw.set_annotations(mne.Annotations([10.0], [2.0], [name]))
    check(f"{name} is honoured by epoching", len(_make_epochs(raw, True)) == 14)

# 4. Entropy measures behave: noise ~ maximal, a sine is far more regular
noise, sine = rng.normal(0, 1, 1024), np.sin(2 * np.pi * 10 * np.arange(1024) / FS)
pe_n = ant.perm_entropy(noise, order=config.PERM_ENTROPY_ORDER, delay=config.PERM_ENTROPY_DELAY, normalize=True)
pe_s = ant.perm_entropy(sine, order=config.PERM_ENTROPY_ORDER, delay=config.PERM_ENTROPY_DELAY, normalize=True)
check("permutation entropy: noise ~1, sine much lower", pe_n > 0.97 and pe_s < 0.6, f"noise {pe_n:.3f}, sine {pe_s:.3f}")

# 5. end-to-end: marker of a pure 10 Hz sine + noise puts power in the mood-alpha band, not the Performance band dominance reversal
raw = make_raw(40 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 3, t.size))
out, _ = preprocess_raw(raw)
m = extract_markers(out)
check("markers computed for a clean synthetic recording", None not in (m.cognition, m.emotion, m.awareness), f"cog {m.cognition:.1f}, mood {m.emotion:.1f}, aw {m.awareness:.3f}")
check("alpha-band PSD (10 Hz sine) exceeds broadband mean PSD", m.emotion > m.cognition, f"{m.emotion:.1f} > {m.cognition:.1f}")

print("\nALL PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
