#!/usr/bin/env python3
"""
00d_extract_personal_pda_cluster.py
Extract PDA_direct from personalized DMN/CEN masks applied directly
to fMRIPrep BOLD. Matches MURFI real-time computation logic.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path
import warnings

FMRIPREP_ROOT = Path("/projects/swglab/data/DMNELF/derivatives/fmriprep_24.1.1")
MASKS_ROOT    = Path("/projects/swglab/data/DMNELF/derivatives/network_masks")
CLUSTER_BASE  = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
OUT_DIR       = CLUSTER_BASE / "targets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS     = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003']
MISSING_RUNS = {'sub-dmnelf1002': [('rest', '02')], 'sub-dmnelf1003': [('feedback', '04')]}
BASELINE_VOLS = 25

TASK_RUNS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

def baseline_zscore(x, n=25):
    mu  = x[:n].mean()
    sig = x[:n].std()
    if sig < 1e-10:
        sig = 1e-10
    return (x - mu) / sig

print("=" * 55)
print("Extracting PDA_direct from personal masks")
print("=" * 55)

from nilearn.maskers import NiftiMasker

n_done    = 0
n_missing = 0

for subject in SUBJECTS:
    # Load personal masks
    mask_dir = MASKS_ROOT / subject
    dmn_mask = mask_dir / (subject
               + "_space-MNI152NLin6Asym_res-2_dmn_mask.nii.gz")
    cen_mask = mask_dir / (subject
               + "_space-MNI152NLin6Asym_res-2_cen_mask.nii.gz")

    if not dmn_mask.exists() or not cen_mask.exists():
        print("  MISSING MASKS: " + subject)
        n_missing += 1
        continue

    # Build maskers
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dmn_masker = NiftiMasker(
            mask_img=str(dmn_mask),
            standardize=False,
            verbose=0
        )
        cen_masker = NiftiMasker(
            mask_img=str(cen_mask),
            standardize=False,
            verbose=0
        )
        dmn_masker.fit()
        cen_masker.fit()

    print()
    print(subject)

    for task, runs in TASK_RUNS.items():
        for run in runs:

            if subject in MISSING_RUNS:
                if (task, run) in MISSING_RUNS[subject]:
                    continue

            bold_name = (subject + "_ses-dmnelf"
                         + "_task-" + task
                         + "_run-"  + run
                         + "_space-MNI152NLin6Asym_res-2"
                         + "_desc-preproc_bold.nii.gz")
            bold_path = (FMRIPREP_ROOT / subject
                         / "ses-dmnelf" / "func" / bold_name)

            out_path = OUT_DIR / (subject + "_task-" + task
                                  + "_run-" + run
                                  + "_pda_direct.npy")

            if not bold_path.exists():
                print("  MISSING BOLD: " + bold_name)
                n_missing += 1
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dmn_ts = dmn_masker.transform(str(bold_path))
                    cen_ts = cen_masker.transform(str(bold_path))
            except Exception as e:
                print("  ERROR: " + str(e))
                n_missing += 1
                continue

            # Mean across voxels within each mask
            dmn_mean = dmn_ts.mean(axis=1)
            cen_mean = cen_ts.mean(axis=1)

            # Baseline z-score (MURFI convention)
            dmn_z = baseline_zscore(dmn_mean)
            cen_z = baseline_zscore(cen_mean)

            # PDA = CEN - DMN
            pda = (cen_z - dmn_z).astype(np.float32)

            np.save(str(out_path), pda)

            print("  " + task + " run-" + run
                  + "  n=" + str(len(pda))
                  + "  pda_std=" + str(round(pda.std(), 3))
                  + "  dmn_vox=" + str(dmn_ts.shape[1])
                  + "  cen_vox=" + str(cen_ts.shape[1]))
            n_done += 1

print()
print("=" * 55)
print("DONE")
print("  Computed: " + str(n_done))
print("  Missing:  " + str(n_missing))
print("=" * 55)