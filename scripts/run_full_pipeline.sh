#!/usr/bin/env bash
# Run the full pipeline end-to-end (pull new raw data, preprocess, extract markers, regenerate all figures/stats)
# Safe to re-run with new readings. Rebuilds full cohort in data/raw/
#
# Usage:
#   ./scripts/run_full_pipeline.sh              # sync from GCS, then run everything
#   ./scripts/run_full_pipeline.sh --skip-sync   # skip the GCS pull, use data/raw/ as-is
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST="/Users/sonyaluchanskaya/Desktop/Psynautics/2026_Feb_files_eeg_eeg_summary.csv"
PARTICIPANT_KEY="/Users/sonyaluchanskaya/Desktop/Psynautics/participant_key_CONFIDENTIAL.csv"

if [[ "${1:-}" != "--skip-sync" ]]; then
    echo "=== 1/5: syncing new/changed files from GCS ==="
    source gcloud_env.sh
    gcloud storage rsync -r gs://psynautics-files/2026_Feb_files/eeg/ data/raw/
else
    echo "=== 1/5: skipped (--skip-sync) ==="
fi

echo "=== 2/5: preprocessing ==="
./.venv/bin/python scripts/batch_preprocess.py \
    --input-dir data/raw \
    --manifest "$MANIFEST" \
    --participant-key "$PARTICIPANT_KEY" \
    --outdir output/batch_full \
    --eeg-only

echo "=== 3/5: marker extraction ==="
./.venv/bin/python scripts/extract_markers.py \
    --input-dir output/batch_full/clean_raw \
    --outdir output/batch_full

echo "=== 4/5: figures (cohort QC + each marker) ==="
FIG="--summary output/batch_full/marker_summary.csv --outdir output/figures"
./.venv/bin/python scripts/make_qc_figures.py --summary output/batch_full/batch_qc_summary.csv --outdir output/figures
./.venv/bin/python scripts/make_cognition_figures.py $FIG
./.venv/bin/python scripts/make_mood_figures.py $FIG
./.venv/bin/python scripts/make_awareness_figures.py $FIG

echo "=== 5/5: statistics (baseline vs. treatment day, baseline vs. post-treatment) + study summary ==="
./.venv/bin/python scripts/stats_baseline_vs_treatment.py $FIG
./.venv/bin/python scripts/stats_baseline_vs_treatment_mood.py $FIG
./.venv/bin/python scripts/stats_baseline_vs_treatment_awareness.py $FIG
./.venv/bin/python scripts/stats_baseline_vs_post.py $FIG
./.venv/bin/python scripts/make_summary_figures.py --marker-summary output/batch_full/marker_summary.csv \
    --qc-summary output/batch_full/batch_qc_summary.csv --pairs-dir output/figures --raw-dir data/raw --outdir output/figures

echo
echo "Done. Results in output/batch_full/ and output/figures/."
