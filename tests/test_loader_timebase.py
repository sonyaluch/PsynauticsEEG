#!/usr/bin/env python
"""
Ground-truth test for the loader's timebase. Builds a synthetic Muse-like CSV
(known 10 Hz + 60 Hz sinusoids, 12-sample packets stamped with jittery ARRIVAL
times, a 3 s packet loss, a short and a long run of sensor NaNs) and checks the
loader recovers the true frequencies/amplitudes, keeps samples intact, and
annotates only genuine dropouts.

Run:  .venv/bin/python tests/test_loader_timebase.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from muse_pipeline.io import load_muse_csv  # noqa: E402

FS, DUR = 256, 60
rng = np.random.default_rng(1)
t = np.arange(FS * DUR) / FS
true = {
    f"ch{i + 1}": 800 + 50 * np.sin(2 * np.pi * 10 * t + i) + 10 * np.sin(2 * np.pi * 60 * t) + rng.normal(0, 5, t.size)
    for i in range(4)
}

# 12-sample packets, arrival-stamped with jitter (occasional 100 ms delays), monotonic
n_pk = t.size // 12
send = (np.arange(n_pk) * 12 + 11) / FS + rng.uniform(0, 0.03, n_pk) + (rng.random(n_pk) < 0.1) * 0.1
send = np.maximum.accumulate(send + np.arange(n_pk) * 1e-4)
lost = (np.arange(n_pk) * 12 >= 20 * FS) & (np.arange(n_pk) * 12 < 23 * FS)  # 3 s of packets never arrive
rows = []
for k in np.where(~lost)[0]:
    for j in range(12):
        idx = k * 12 + j
        rows.append([1.7e9 + send[k] + j * 5e-5] + [true[c][idx] for c in true])
df = pd.DataFrame(rows, columns=["ts", "ch1", "ch2", "ch3", "ch4"])
df["ch5"] = np.nan
df["ch6"] = np.nan
n_short, n_long = 30, 120
short_start = df.index[df.ts > 1.7e9 + 10][0]
long_start = df.index[df.ts > 1.7e9 + 40][0]
df.loc[short_start:short_start + n_short - 1, "ch2"] = np.nan
df.loc[long_start:long_start + n_long - 1, "ch3"] = np.nan

with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "synthetic_eeg.csv"
    df.to_csv(path, index=False)
    raw, rep = load_muse_csv(path)

data = raw.get_data() * 1e6  # back to uV
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


check("duration within 0.3 s of 60 s", abs(raw.times[-1] - DUR) < 0.3, f"{raw.times[-1]:.2f}s")
check("first 200 samples preserved exactly", np.allclose(data[0, :200], true["ch1"][:200], atol=1e-6))

# spectrum of a clean stretch (2-18 s, before the loss and NaNs)
seg = data[0, 2 * FS:18 * FS]
amp = np.abs(np.fft.rfft(seg - seg.mean())) * 2 / seg.size
fr = np.fft.rfftfreq(seg.size, 1 / FS)
a10, a60 = amp[np.argmin(abs(fr - 10))], amp[np.argmin(abs(fr - 60))]
check("10 Hz amplitude ~50 uV", abs(a10 - 50) < 2, f"{a10:.1f}")
check("60 Hz amplitude ~10 uV", abs(a60 - 10) < 1, f"{a60:.1f}")

drops = [(o, d_) for o, d_, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description) if desc == "BAD_dropout"]
check("exactly two dropout annotations (3 s loss + 120-row NaN run)", len(drops) == 2, f"{len(drops)} found: {[(round(o, 2), round(d, 2)) for o, d in drops]}")
if len(drops) == 2:
    check("loss annotated near 20 s, ~3 s long", abs(drops[0][0] - 20) < 0.3 and abs(drops[0][1] - 3) < 0.3)
    check("long NaN run annotated near 40 s, ~0.47 s", abs(drops[1][0] - 40) < 0.3 and abs(drops[1][1] - n_long / FS) < 0.1)
check("short 30-row NaN run interpolated, not annotated", not any(5 < o < 15 for o, _ in drops))
check("no NaN left in signal", not np.isnan(data).any())
check("sensor NaN count reported", rep.n_nan_samples == n_short + n_long, str(rep.n_nan_samples))

print("\nALL PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
