# 01_fit_microstates.py
# Run locally: python 01_fit_microstates.py
# Deploys cluster script, submits SLURM job, monitors.
#
# What it does on the cluster:
#   - Loads all rest EEG runs across all subjects
#   - Extracts GFP peaks, rejects outliers > 3 SD
#   - Runs polarity-invariant k-means (k=7, Custo 2017)
#   - Saves templates.npy (7, n_ch) and gev.npy
#   - QC: reports GEV per map

import py_compile
import time
from pathlib import Path
from utils import run_ssh, scp_to, make_cluster_dirs
from config import (
    CLUSTER_BASE, CLUSTER_SSH, SLURM_ACCOUNT, PYTHON,
    SUBJECTS, EEG_ROOT, N_MICROSTATES, N_KMEANS_RESTARTS,
    KMEANS_MAX_ITER, GFP_OUTLIER_SD, SFREQ, LOCAL_BASE
)

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '01_fit_microstates_cluster.py',
    'Fit 7 microstate templates on pooled rest EEG.',
    'Method: polarity-invariant k-means on GFP peaks (Custo 2017).',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'from pathlib import Path',
    'import mne',
    'from scipy.signal import argrelmax',
    '',
    '# ── Paths and constants ───────────────────────────────',
    'EEG_ROOT      = Path("' + EEG_ROOT + '")',
    'OUT_DIR       = Path("' + CLUSTER_BASE + '")',
    'SUBJECTS      = ' + str(SUBJECTS),
    'SFREQ         = ' + str(SFREQ),
    'N_MICROSTATES = ' + str(N_MICROSTATES),
    'N_RESTARTS    = ' + str(N_KMEANS_RESTARTS),
    'MAX_ITER      = ' + str(KMEANS_MAX_ITER),
    'OUTLIER_SD    = ' + str(GFP_OUTLIER_SD),
    '',
    '# ── Helpers ───────────────────────────────────────────',
    'def load_eeg(fif_path):',
    '    raw = mne.io.read_raw_fif(str(fif_path), preload=True,',
    '                               verbose=False)',
    '    drop = [ch for ch in raw.ch_names',
    '            if any(x in ch.upper() for x in',
    '                   ("ECG","EKG","EMG","EOG","STIM","STATUS"))]',
    '    if drop:',
    '        raw.drop_channels(drop)',
    '    ref = float(np.abs(raw.get_data().mean(axis=0)).mean())',
    '    if ref > 1e-7:',
    '        raw.set_eeg_reference("average", projection=False,',
    '                               verbose=False)',
    '    if raw.info["sfreq"] != SFREQ:',
    '        raw.resample(SFREQ, verbose=False)',
    '    return (raw.get_data() * 1e6).astype(np.float32)',
    '',
    'def compute_gfp(eeg):',
    '    return eeg.std(axis=0).astype(np.float32)',
    '',
    'def get_gfp_peaks(gfp):',
    '    peaks = argrelmax(gfp, order=1)[0]',
    '    mu    = gfp[peaks].mean()',
    '    sig   = gfp[peaks].std()',
    '    peaks = peaks[gfp[peaks] < mu + OUTLIER_SD * sig]',
    '    return peaks',
    '',
    'def normalize_map(m):',
    '    n = np.linalg.norm(m)',
    '    return m / n if n > 1e-10 else m',
    '',
    'def polarity_corr(a, b):',
    '    a = a - a.mean()',
    '    b = b - b.mean()',
    '    d = np.linalg.norm(a) * np.linalg.norm(b)',
    '    return float(abs(np.dot(a, b) / d)) if d > 1e-10 else 0.0',
    '',
    '# ── Load all rest GFP peaks ───────────────────────────',
    'print("=" * 55)',
    'print("1. Loading rest EEG GFP peaks")',
    'print("=" * 55)',
    '',
    'all_peaks = []',
    'n_loaded  = 0',
    '',
    'for subject in SUBJECTS:',
    '    for run in ["01", "02"]:',
    '        fname = (subject + "_ses-dmnelf_task-rest"',
    '                 + "_run-" + run + "_desc-preproc_eeg.fif")',
    '        fif = (EEG_ROOT / subject / "ses-dmnelf"',
    '               / "eeg" / fname)',
    '        if not fif.exists():',
    '            print("  MISSING: " + str(fif))',
    '            continue',
    '        print("  Loading: " + subject + "  run-" + run)',
    '        eeg  = load_eeg(fif)',
    '        gfp  = compute_gfp(eeg)',
    '        pksi = get_gfp_peaks(gfp)',
    '        maps = eeg[:, pksi].T',
    '        # Normalize each map',
    '        maps = np.array([normalize_map(m) for m in maps])',
    '        all_peaks.append(maps)',
    '        n_loaded += 1',
    '        print("    shape=" + str(eeg.shape)',
    '              + "  peaks=" + str(len(pksi)))',
    '',
    'all_peaks = np.concatenate(all_peaks, axis=0)',
    'print()',
    'print("Runs loaded:  " + str(n_loaded))',
    'print("Total peaks:  " + str(len(all_peaks)))',
    'print("Map shape:    " + str(all_peaks.shape))',
    '',
    '# ── Polarity-invariant k-means ────────────────────────',
    'print()',
    'print("=" * 55)',
    'print("2. Fitting k-means  k=" + str(N_MICROSTATES))',
    'print("=" * 55)',
    '',
    'best_gev      = -1.0',
    'best_templates = None',
    '',
    'for restart in range(N_RESTARTS):',
    '    # Random init from data',
    '    idx  = np.random.choice(len(all_peaks), N_MICROSTATES,',
    '                             replace=False)',
    '    maps = all_peaks[idx].copy()',
    '',
    '    prev_labels = None',
    '    for iteration in range(MAX_ITER):',
    '        # Assignment step — polarity invariant',
    '        corrs  = np.abs(all_peaks @ maps.T)',
    '        labels = corrs.argmax(axis=1)',
    '',
    '        # Update step',
    '        new_maps = np.zeros_like(maps)',
    '        for k in range(N_MICROSTATES):',
    '            members = all_peaks[labels == k]',
    '            if len(members) == 0:',
    '                new_maps[k] = all_peaks[',
    '                    np.random.randint(len(all_peaks))]',
    '                continue',
    '            # Flip polarity to align signs before averaging',
    '            ref = members[0]',
    '            for i in range(len(members)):',
    '                if np.dot(members[i], ref) < 0:',
    '                    members[i] = -members[i]',
    '            new_maps[k] = normalize_map(members.mean(axis=0))',
    '',
    '        # Check convergence',
    '        if prev_labels is not None:',
    '            if np.all(labels == prev_labels):',
    '                break',
    '        prev_labels = labels.copy()',
    '        maps = new_maps',
    '',
    '    # Compute GEV',
    '    gfp_sq = (all_peaks ** 2).sum(axis=1)',
    '    gev    = 0.0',
    '    for k in range(N_MICROSTATES):',
    '        members_gfp = gfp_sq[labels == k]',
    '        corr_k      = np.abs(all_peaks[labels == k] @ maps[k])',
    '        if len(members_gfp) > 0:',
    '            gev += float((corr_k ** 2 * members_gfp).sum()',
    '                         / gfp_sq.sum())',
    '',
    '    print("  restart " + str(restart + 1).zfill(2)',
    '          + "  iter=" + str(iteration)',
    '          + "  GEV=" + "{:.4f}".format(gev))',
    '',
    '    if gev > best_gev:',
    '        best_gev       = gev',
    '        best_templates = maps.copy()',
    '',
    '# ── Save ──────────────────────────────────────────────',
    'print()',
    'print("=" * 55)',
    'print("3. Saving templates")',
    'print("=" * 55)',
    '',
    'out_templates = OUT_DIR / "microstates" / "templates.npy"',
    'out_gev       = OUT_DIR / "microstates" / "gev.npy"',
    'out_labels    = OUT_DIR / "microstates" / "final_labels.npy"',
    '(OUT_DIR / "microstates").mkdir(parents=True, exist_ok=True)',
    '',
    'np.save(str(out_templates), best_templates)',
    'np.save(str(out_gev),       np.array([best_gev]))',
    '',
    'print("Templates: " + str(best_templates.shape))',
    'print("Best GEV:  " + "{:.4f}".format(best_gev))',
    'print("Saved:     " + str(out_templates))',
    '',
    '# ── QC per-map stats ──────────────────────────────────',
    'print()',
    'print("=" * 55)',
    'print("4. Per-map stats")',
    'print("=" * 55)',
    '',
    'corrs  = np.abs(all_peaks @ best_templates.T)',
    'labels = corrs.argmax(axis=1)',
    'gfp_sq = (all_peaks ** 2).sum(axis=1)',
    '',
    'for k in range(N_MICROSTATES):',
    '    members = (labels == k).sum()',
    '    cov     = 100.0 * members / len(labels)',
    '    gev_k   = float((np.abs(all_peaks[labels == k]',
    '                     @ best_templates[k]) ** 2',
    '                     * gfp_sq[labels == k]).sum()',
    '                    / gfp_sq.sum())',
    '    print("  MS" + chr(65 + k)',
    '          + "  coverage=" + "{:.1f}".format(cov) + "%"',
    '          + "  GEV="      + "{:.4f}".format(gev_k))',
    '',
    'print()',
    'print("DONE")',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "01_fit_microstates_cluster.py"
script_path = LOCAL_BASE / "scripts" / script_name
script_path.parent.mkdir(parents=True, exist_ok=True)

with open(script_path, "w") as f:
    f.write("\n".join(lines))

# ── 3. Syntax check ────────────────────────────────────────
print("Checking syntax...")
try:
    py_compile.compile(str(script_path), doraise=True)
    print("Syntax OK: " + script_name)
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR: " + str(e))
    raise

# ── 4. Make cluster dirs + deploy ──────────────────────────
print("\nCreating cluster directories...")
make_cluster_dirs()

print("Deploying script...")
remote_script = CLUSTER_BASE + "/scripts/" + script_name
scp_to(script_path, remote_script)

# ── 5. Submit SLURM job ────────────────────────────────────
print("\nSubmitting SLURM job...")

sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=fit_microstates",
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/fit_microstates_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/fit_microstates_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=02:00:00",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --mem=32G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

sbatch_path = LOCAL_BASE / "scripts" / "01_fit_microstates.sh"
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

remote_sbatch = CLUSTER_BASE + "/scripts/01_fit_microstates.sh"
scp_to(sbatch_path, remote_sbatch, verbose=False)

result = run_ssh("sbatch " + remote_sbatch)
job_id = ""
for line in result.stdout.strip().split("\n"):
    if "Submitted" in line:
        job_id = line.strip().split()[-1]
        print("Job ID: " + job_id)

# ── 6. Monitor ─────────────────────────────────────────────
if job_id:
    print("\nMonitoring job " + job_id + "  (Ctrl+C to stop watching)")
    print("-" * 55)
    try:
        while True:
            r = run_ssh("squeue -j " + job_id + " --format='%.8i %.8T %.10M' 2>/dev/null",
                        verbose=False)
            status = r.stdout.strip()
            if status and "JOBID" not in status.split("\n")[-1]:
                print(status)
            else:
                print("Job finished or not found — checking log...")
                log = run_ssh(
                    "tail -30 " + CLUSTER_BASE
                    + "/logs/fit_microstates_" + job_id + ".out 2>/dev/null",
                    verbose=False
                )
                print(log.stdout)
                break
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nStopped watching. Check manually:")
        print("  squeue -j " + job_id)
        print("  tail -f " + CLUSTER_BASE + "/logs/fit_microstates_" + job_id + ".out")