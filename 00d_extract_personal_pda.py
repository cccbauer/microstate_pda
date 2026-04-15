# 00d_extract_personal_pda.py
# Run locally: python 00d_extract_personal_pda.py
# Deploys cluster script, submits SLURM job.
#
# What it does on the cluster:
#   For each subject x task x run:
#   1. Load personalized DMN and CEN binary masks (from 00b)
#   2. Apply NiftiMasker to fMRIPrep BOLD — mean signal within each mask
#   3. Baseline z-score (first 25 vols, matching MURFI)
#   4. Compute PDA_direct = CEN_z - DMN_z
#   5. Save as (n_vols,) float32
#
# Output:
#   targets/{subject}_task-{task}_run-{run}_pda_direct.npy

import py_compile
import time
from pathlib import Path
from utils import run_ssh, scp_to
from config import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS_EEG_FMRI_ALL, LOCAL_BASE,
    MISSING_RUNS
)

FMRIPREP_ROOT = "/projects/swglab/data/DMNELF/derivatives/fmriprep_24.1.1"
MASKS_ROOT    = "/projects/swglab/data/DMNELF/derivatives/network_masks"

lines = [
    '#!/usr/bin/env python3',
    '"""',
    '00d_extract_personal_pda_cluster.py',
    'Extract PDA_direct from personalized DMN/CEN masks applied directly',
    'to fMRIPrep BOLD. Matches MURFI real-time computation logic.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'from pathlib import Path',
    'import warnings',
    '',
    'FMRIPREP_ROOT = Path("' + FMRIPREP_ROOT + '")',
    'MASKS_ROOT    = Path("' + MASKS_ROOT + '")',
    'CLUSTER_BASE  = Path("' + CLUSTER_BASE + '")',
    'OUT_DIR       = CLUSTER_BASE / "targets"',
    'OUT_DIR.mkdir(parents=True, exist_ok=True)',
    '',
    'SUBJECTS     = ' + str(SUBJECTS_EEG_FMRI_ALL),
    'MISSING_RUNS = ' + str(MISSING_RUNS),
    'BASELINE_VOLS = 25',
    '',
    'TASK_RUNS = {',
    '    "rest":      ["01", "02"],',
    '    "shortrest": ["01"],',
    '    "feedback":  ["01", "02", "03", "04"],',
    '}',
    '',
    'def baseline_zscore(x, n=25):',
    '    mu  = x[:n].mean()',
    '    sig = x[:n].std()',
    '    if sig < 1e-10:',
    '        sig = 1e-10',
    '    return (x - mu) / sig',
    '',
    'print("=" * 55)',
    'print("Extracting PDA_direct from personal masks")',
    'print("=" * 55)',
    '',
    'from nilearn.maskers import NiftiMasker',
    '',
    'n_done    = 0',
    'n_missing = 0',
    '',
    'for subject in SUBJECTS:',
    '    # Load personal masks',
    '    mask_dir = MASKS_ROOT / subject',
    '    dmn_mask = mask_dir / (subject',
    '               + "_space-MNI152NLin6Asym_res-2_dmn_mask.nii.gz")',
    '    cen_mask = mask_dir / (subject',
    '               + "_space-MNI152NLin6Asym_res-2_cen_mask.nii.gz")',
    '',
    '    if not dmn_mask.exists() or not cen_mask.exists():',
    '        print("  MISSING MASKS: " + subject)',
    '        n_missing += 1',
    '        continue',
    '',
    '    # Build maskers',
    '    with warnings.catch_warnings():',
    '        warnings.simplefilter("ignore")',
    '        dmn_masker = NiftiMasker(',
    '            mask_img=str(dmn_mask),',
    '            standardize=False,',
    '            verbose=0',
    '        )',
    '        cen_masker = NiftiMasker(',
    '            mask_img=str(cen_mask),',
    '            standardize=False,',
    '            verbose=0',
    '        )',
    '        dmn_masker.fit()',
    '        cen_masker.fit()',
    '',
    '    print()',
    '    print(subject)',
    '',
    '    for task, runs in TASK_RUNS.items():',
    '        for run in runs:',
    '',
    '            if subject in MISSING_RUNS:',
    '                if (task, run) in MISSING_RUNS[subject]:',
    '                    continue',
    '',
    '            bold_name = (subject + "_ses-dmnelf"',
    '                         + "_task-" + task',
    '                         + "_run-"  + run',
    '                         + "_space-MNI152NLin6Asym_res-2"',
    '                         + "_desc-preproc_bold.nii.gz")',
    '            bold_path = (FMRIPREP_ROOT / subject',
    '                         / "ses-dmnelf" / "func" / bold_name)',
    '',
    '            out_path = OUT_DIR / (subject + "_task-" + task',
    '                                  + "_run-" + run',
    '                                  + "_pda_direct.npy")',
    '',
    '            if not bold_path.exists():',
    '                print("  MISSING BOLD: " + bold_name)',
    '                n_missing += 1',
    '                continue',
    '',
    '            try:',
    '                with warnings.catch_warnings():',
    '                    warnings.simplefilter("ignore")',
    '                    dmn_ts = dmn_masker.transform(str(bold_path))',
    '                    cen_ts = cen_masker.transform(str(bold_path))',
    '            except Exception as e:',
    '                print("  ERROR: " + str(e))',
    '                n_missing += 1',
    '                continue',
    '',
    '            # Mean across voxels within each mask',
    '            dmn_mean = dmn_ts.mean(axis=1)',
    '            cen_mean = cen_ts.mean(axis=1)',
    '',
    '            # Baseline z-score (MURFI convention)',
    '            dmn_z = baseline_zscore(dmn_mean)',
    '            cen_z = baseline_zscore(cen_mean)',
    '',
    '            # PDA = CEN - DMN',
    '            pda = (cen_z - dmn_z).astype(np.float32)',
    '',
    '            np.save(str(out_path), pda)',
    '',
    '            print("  " + task + " run-" + run',
    '                  + "  n=" + str(len(pda))',
    '                  + "  pda_std=" + str(round(pda.std(), 3))',
    '                  + "  dmn_vox=" + str(dmn_ts.shape[1])',
    '                  + "  cen_vox=" + str(cen_ts.shape[1]))',
    '            n_done += 1',
    '',
    'print()',
    'print("=" * 55)',
    'print("DONE")',
    'print("  Computed: " + str(n_done))',
    'print("  Missing:  " + str(n_missing))',
    'print("=" * 55)',
]

# ── Save cluster script ─────────────────────────────────────
script_name = "00d_extract_personal_pda_cluster.py"
script_path = LOCAL_BASE / "scripts" / script_name
script_path.parent.mkdir(parents=True, exist_ok=True)

with open(script_path, "w") as f:
    f.write("\n".join(lines))

# ── Syntax check ───────────────────────────────────────────
print("Checking syntax...")
try:
    py_compile.compile(str(script_path), doraise=True)
    print("Syntax OK: " + script_name)
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR: " + str(e))
    raise

# ── SLURM script ───────────────────────────────────────────
job_name = "extract_pda_direct"
sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=06:00:00",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --mem=64G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

sbatch_name = "00d_extract_personal_pda.sh"
sbatch_path = LOCAL_BASE / "scripts" / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

# ── Deploy ─────────────────────────────────────────────────
print("\nDeploying...")
scp_to(script_path,
       CLUSTER_BASE + "/scripts/" + script_name,
       verbose=False)
scp_to(sbatch_path,
       CLUSTER_BASE + "/scripts/" + sbatch_name,
       verbose=False)
print("Deployed: " + script_name)

# ── Submit ─────────────────────────────────────────────────
print("\nSubmitting SLURM job...")
result = run_ssh("sbatch " + CLUSTER_BASE + "/scripts/" + sbatch_name)
job_id = ""
for line in result.stdout.strip().split("\n"):
    if "Submitted" in line:
        job_id = line.strip().split()[-1]
        print("Job ID: " + job_id)

# ── Monitor ────────────────────────────────────────────────
if job_id:
    print("\nMonitoring job " + job_id + "  (Ctrl+C to stop)")
    print("-" * 55)
    try:
        while True:
            r = run_ssh(
                "squeue -j " + job_id
                + " --format=%.8i_%.8T_%.10M 2>/dev/null",
                verbose=False
            )
            status = r.stdout.strip()
            if status and "JOBID" not in status.split("\n")[-1]:
                print(status)
            else:
                print("Job finished — checking log...")
                log = run_ssh(
                    "tail -30 " + CLUSTER_BASE
                    + "/logs/" + job_name + "_" + job_id
                    + ".out 2>/dev/null",
                    verbose=False
                )
                print(log.stdout)
                break
            time.sleep(20)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        print("  tail -f " + CLUSTER_BASE
              + "/logs/" + job_name + "_" + job_id + ".out")