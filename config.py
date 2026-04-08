# config.py  — DMNELF microstate PDA pipeline
from pathlib import Path

# ── SSH / Cluster ──────────────────────────────────────────
CLUSTER_USER   = "cccbauer"
CLUSTER_HOST   = "explorer.northeastern.edu"
CLUSTER_SSH    = CLUSTER_USER + "@" + CLUSTER_HOST
SLURM_ACCOUNT  = "suewhit"
PYTHON         = "$HOME/my_anaconda/bin/python"

# ── Cluster paths ──────────────────────────────────────────
CLUSTER_BASE   = "/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3"
EEG_ROOT       = "/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed"
DIFUMO_ROOT    = "/projects/swglab/data/DMNELF/analysis/MNE/jupyter/neurobolt/difumo_timeseries"
CONFOUND_ROOT  = "/projects/swglab/data/DMNELF/derivatives/fmriprep_24.1.1_wo_FS"

# ── Local paths (machine-agnostic) ─────────────────────────
import os
_user = os.environ.get("USER", "")
if _user == "anitya":
    _dropbox = Path("/Users/anitya/MIT Dropbox/Clemen Bauer/00_2022/ResearchScientist"
                    "/01_grants/01_R21-2021/Analysis/MNE")
elif _user == "cccbauer":
    _dropbox = Path("/Users/cccbauer/Dropbox (MIT)/00_2022/ResearchScientist"
                    "/01_grants/01_R21-2021/Analysis/MNE")
else:
    _dropbox = Path.home() / "microstate_pda_v3"

LOCAL_BASE  = _dropbox / "microstate_pda_v3"
SCRIPTS_DIR = LOCAL_BASE / "scripts"
FIGURES_DIR = LOCAL_BASE / "figures"
LOGS_DIR    = LOCAL_BASE / "logs"
MODELS_DIR  = LOCAL_BASE / "models"
RESULTS_DIR = LOCAL_BASE / "results"
SCRIPTS_DIR    = LOCAL_BASE / "scripts"
FIGURES_DIR    = LOCAL_BASE / "figures"
LOGS_DIR       = LOCAL_BASE / "logs"
MODELS_DIR     = LOCAL_BASE / "models"
RESULTS_DIR    = LOCAL_BASE / "results"

# ── Subjects ───────────────────────────────────────────────
SUBJECTS = [
    "sub-dmnelf001",
    "sub-dmnelf004",
    "sub-dmnelf005",
    "sub-dmnelf006",
    "sub-dmnelf007",
    "sub-dmnelf008",
    "sub-dmnelf010",
]

# ── Tasks / runs ───────────────────────────────────────────
RUN_MAP = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

# ── EEG ────────────────────────────────────────────────────
SFREQ          = 200        # Hz after resampling
N_CHANNELS     = 31
TR_SAMPLES     = 240        # EEG samples per TR at 200 Hz
ALIGNMENT_TOL  = 480        # max tolerated misalignment (~2 TR)

# ── Microstate — Custo 2017 ────────────────────────────────
N_MICROSTATES       = 7     # NOT 4 — validated against Tarailis 2023
N_KMEANS_RESTARTS   = 20
KMEANS_MAX_ITER     = 1000
GFP_OUTLIER_SD      = 3.0   # reject GFP peaks above this for fitting
POLARITY_INVARIANT  = True  # standard for microstate analysis

# ── fMRI / DiFuMo-64 ───────────────────────────────────────
N_PARCELS      = 64
# 0-based indices into ROI_ columns of DiFuMo TSV
# Source: labels_64_dictionary.csv (Dadi et al. 2020), verified manually
DMN_IDX = [3, 6, 22, 29, 35, 38, 58, 60]
CEN_IDX = [4, 31, 47, 48, 50, 51]

# ── MURFI-aligned PDA ──────────────────────────────────────
BASELINE_VOLS  = 25         # first 25 vols = 30s baseline per run
PSA_WINDOW     = 25         # rolling window for PSA multiplier

# ── HRF convolution ────────────────────────────────────────
TR             = 1.2        # seconds
HRF_PEAK_S     = 5.0        # canonical HRF peak delay (seconds)

# ── Decoder ────────────────────────────────────────────────
# Features: 7 T-hat (TESS) + GFP + GMD = 9 per TR
N_FEATURES     = 9
DECODER_MODEL  = "elasticnet"
ALPHAS         = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CV_STRATEGY    = "leave_one_run_out"
TRAIN_TASK     = "feedback"
TEST_TASKS     = ["shortrest", "feedback"]
Z_SCORE_FEATS  = True
Z_SCORE_TARGET = True