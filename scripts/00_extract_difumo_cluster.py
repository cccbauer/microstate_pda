#!/usr/bin/env python3
"""
00_extract_difumo_cluster.py
Extract DiFuMo-64 parcel timeseries from fMRIPrep BOLD output.
Uses nilearn NiftiLabelsMasker with DiFuMo-64 atlas.
Output TSV columns: volume, time, ROI_01..ROI_64
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────
FMRIPREP_ROOT = Path("/projects/swglab/data/DMNELF/derivatives/fmriprep_24.1.1")
OUT_DIR       = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/neurobolt/difumo_timeseries")
CACHE_DIR     = Path("/projects/swglab/software/nilearn_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set nilearn data dir to shared cache
os.environ["NILEARN_DATA"] = str(CACHE_DIR)

SUBJECTS = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf010']

TASK_RUNS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

TR = 1.2  # seconds

# ── Load DiFuMo atlas ─────────────────────────────────
print("=" * 55)
print("Loading DiFuMo-64 atlas")
print("=" * 55)

from nilearn import datasets
difumo = datasets.fetch_atlas_difumo(
    dimension=64,
    resolution_mm=2,
    data_dir=str(CACHE_DIR),
    legacy_format=False
)
atlas_img  = difumo.maps
atlas_labels = list(difumo.labels_img.dataobj) if hasattr(difumo, "labels_img") else []

print("Atlas loaded: " + str(atlas_img))

# ── Build masker ──────────────────────────────────────
from nilearn.maskers import NiftiMapsMasker
masker = NiftiMapsMasker(
    maps_img=atlas_img,
    standardize=False,
    memory=str(CACHE_DIR / "nilearn_cache"),
    verbose=0
)
masker.fit()
print("Masker fitted")

# ── Main loop ─────────────────────────────────────────
print()
print("=" * 55)
print("Extracting DiFuMo timeseries")
print("=" * 55)

n_done    = 0
n_skipped = 0
n_missing = 0

for subject in SUBJECTS:
    for task, runs in TASK_RUNS.items():
        for run in runs:
            # Input BOLD file
            bold_name = (subject + "_ses-dmnelf_task-" + task
                         + "_run-" + run
                         + "_space-MNI152NLin6Asym_res-2"
                         + "_desc-preproc_bold.nii.gz")
            bold_path = (FMRIPREP_ROOT / subject / "ses-dmnelf"
                         / "func" / bold_name)

            # Output TSV file
            tsv_name = (subject + "_ses-dmnelf_task-" + task
                        + "_run-" + run
                        + "_desc-difumo64_timeseries.tsv")
            tsv_path = OUT_DIR / tsv_name

            if not bold_path.exists():
                print("  MISSING: " + bold_name)
                n_missing += 1
                continue

            if tsv_path.exists():
                print("  EXISTS (skip): " + tsv_name)
                n_skipped += 1
                continue

            print("  Extracting: " + subject
                  + "  task-" + task + "  run-" + run)

            # Extract timeseries
            time_series = masker.transform(str(bold_path))
            n_vols = time_series.shape[0]

            # Build TSV with volume, time, ROI_01..ROI_64
            df = pd.DataFrame()
            df["volume"] = np.arange(n_vols)
            df["time"]   = np.arange(n_vols) * TR
            for i in range(time_series.shape[1]):
                col = "ROI_" + str(i + 1).zfill(2)
                df[col] = time_series[:, i]

            df.to_csv(str(tsv_path), sep="\t", index=False)

            print("    shape: " + str(time_series.shape)
                  + "  vols=" + str(n_vols))
            print("    saved: " + tsv_name)
            n_done += 1

print()
print("=" * 55)
print("Summary")
print("=" * 55)
print("  Extracted: " + str(n_done))
print("  Skipped:   " + str(n_skipped) + " (already existed)")
print("  Missing:   " + str(n_missing) + " (BOLD not found)")
print()
print("DONE")