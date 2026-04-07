#!/usr/bin/env python3
"""
01_fit_microstates_cluster.py
Fit 7 microstate templates on pooled rest EEG.
Method: polarity-invariant k-means on GFP peaks (Custo 2017).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path
import mne
from scipy.signal import argrelmax

# ── Paths and constants ───────────────────────────────
EEG_ROOT      = Path("/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed")
OUT_DIR       = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
SUBJECTS      = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf010']
SFREQ         = 200
N_MICROSTATES = 7
N_RESTARTS    = 20
MAX_ITER      = 1000
OUTLIER_SD    = 3.0

# ── Helpers ───────────────────────────────────────────
def load_eeg(fif_path):
    raw = mne.io.read_raw_fif(str(fif_path), preload=True,
                               verbose=False)
    drop = [ch for ch in raw.ch_names
            if any(x in ch.upper() for x in
                   ("ECG","EKG","EMG","EOG","STIM","STATUS"))]
    if drop:
        raw.drop_channels(drop)
    ref = float(np.abs(raw.get_data().mean(axis=0)).mean())
    if ref > 1e-7:
        raw.set_eeg_reference("average", projection=False,
                               verbose=False)
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    return (raw.get_data() * 1e6).astype(np.float32)

def compute_gfp(eeg):
    return eeg.std(axis=0).astype(np.float32)

def get_gfp_peaks(gfp):
    peaks = argrelmax(gfp, order=1)[0]
    mu    = gfp[peaks].mean()
    sig   = gfp[peaks].std()
    peaks = peaks[gfp[peaks] < mu + OUTLIER_SD * sig]
    return peaks

def normalize_map(m):
    n = np.linalg.norm(m)
    return m / n if n > 1e-10 else m

def polarity_corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.dot(a, b) / d)) if d > 1e-10 else 0.0

# ── Load all rest GFP peaks ───────────────────────────
print("=" * 55)
print("1. Loading rest EEG GFP peaks")
print("=" * 55)

all_peaks = []
n_loaded  = 0

for subject in SUBJECTS:
    for run in ["01", "02"]:
        fname = (subject + "_ses-dmnelf_task-rest"
                 + "_run-" + run + "_desc-preproc_eeg.fif")
        fif = (EEG_ROOT / subject / "ses-dmnelf"
               / "eeg" / fname)
        if not fif.exists():
            print("  MISSING: " + str(fif))
            continue
        print("  Loading: " + subject + "  run-" + run)
        eeg  = load_eeg(fif)
        gfp  = compute_gfp(eeg)
        pksi = get_gfp_peaks(gfp)
        maps = eeg[:, pksi].T
        # Normalize each map
        maps = np.array([normalize_map(m) for m in maps])
        all_peaks.append(maps)
        n_loaded += 1
        print("    shape=" + str(eeg.shape)
              + "  peaks=" + str(len(pksi)))

all_peaks = np.concatenate(all_peaks, axis=0)
print()
print("Runs loaded:  " + str(n_loaded))
print("Total peaks:  " + str(len(all_peaks)))
print("Map shape:    " + str(all_peaks.shape))

# ── Polarity-invariant k-means ────────────────────────
print()
print("=" * 55)
print("2. Fitting k-means  k=" + str(N_MICROSTATES))
print("=" * 55)

best_gev      = -1.0
best_templates = None

for restart in range(N_RESTARTS):
    # Random init from data
    idx  = np.random.choice(len(all_peaks), N_MICROSTATES,
                             replace=False)
    maps = all_peaks[idx].copy()

    prev_labels = None
    for iteration in range(MAX_ITER):
        # Assignment step — polarity invariant
        corrs  = np.abs(all_peaks @ maps.T)
        labels = corrs.argmax(axis=1)

        # Update step
        new_maps = np.zeros_like(maps)
        for k in range(N_MICROSTATES):
            members = all_peaks[labels == k]
            if len(members) == 0:
                new_maps[k] = all_peaks[
                    np.random.randint(len(all_peaks))]
                continue
            # Flip polarity to align signs before averaging
            ref = members[0]
            for i in range(len(members)):
                if np.dot(members[i], ref) < 0:
                    members[i] = -members[i]
            new_maps[k] = normalize_map(members.mean(axis=0))

        # Check convergence
        if prev_labels is not None:
            if np.all(labels == prev_labels):
                break
        prev_labels = labels.copy()
        maps = new_maps

    # Compute GEV
    gfp_sq = (all_peaks ** 2).sum(axis=1)
    gev    = 0.0
    for k in range(N_MICROSTATES):
        members_gfp = gfp_sq[labels == k]
        corr_k      = np.abs(all_peaks[labels == k] @ maps[k])
        if len(members_gfp) > 0:
            gev += float((corr_k ** 2 * members_gfp).sum()
                         / gfp_sq.sum())

    print("  restart " + str(restart + 1).zfill(2)
          + "  iter=" + str(iteration)
          + "  GEV=" + "{:.4f}".format(gev))

    if gev > best_gev:
        best_gev       = gev
        best_templates = maps.copy()

# ── Save ──────────────────────────────────────────────
print()
print("=" * 55)
print("3. Saving templates")
print("=" * 55)

out_templates = OUT_DIR / "microstates" / "templates.npy"
out_gev       = OUT_DIR / "microstates" / "gev.npy"
out_labels    = OUT_DIR / "microstates" / "final_labels.npy"
(OUT_DIR / "microstates").mkdir(parents=True, exist_ok=True)

np.save(str(out_templates), best_templates)
np.save(str(out_gev),       np.array([best_gev]))

print("Templates: " + str(best_templates.shape))
print("Best GEV:  " + "{:.4f}".format(best_gev))
print("Saved:     " + str(out_templates))

# ── QC per-map stats ──────────────────────────────────
print()
print("=" * 55)
print("4. Per-map stats")
print("=" * 55)

corrs  = np.abs(all_peaks @ best_templates.T)
labels = corrs.argmax(axis=1)
gfp_sq = (all_peaks ** 2).sum(axis=1)

for k in range(N_MICROSTATES):
    members = (labels == k).sum()
    cov     = 100.0 * members / len(labels)
    gev_k   = float((np.abs(all_peaks[labels == k]
                     @ best_templates[k]) ** 2
                     * gfp_sq[labels == k]).sum()
                    / gfp_sq.sum())
    print("  MS" + chr(65 + k)
          + "  coverage=" + "{:.1f}".format(cov) + "%"
          + "  GEV="      + "{:.4f}".format(gev_k))

print()
print("DONE")