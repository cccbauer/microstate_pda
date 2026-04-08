markdown# Methods — Microstate PDA Pipeline
## EEG Microstate Prediction of Real-Time fMRI Neurofeedback Signal

**Study:** DMNELF (simultaneous EEG-fMRI neurofeedback)  
**Author:** Clemens C.C. Bauer, MD PhD  
**Lab:** EPIC Brain Lab, Northeastern University / Gabrieli Lab, MIT  

---

## Overview

We develop an offline EEG→fMRI decoder that predicts the real-time
neurofeedback signal (Positive Diametric Activity, PDA) from EEG
microstates. The pipeline replicates and validates the TESS method
(Custo et al. 2014/2017) on simultaneous EEG-fMRI data collected
during mindfulness-based neurofeedback (Bloom et al. 2023).

The ultimate goal is to replace the real-time fMRI signal with an
EEG-derived proxy, enabling neurofeedback without an MRI scanner.

---

## Dataset

**Study:** DMNELF  
**Design:** Simultaneous EEG-fMRI, Siemens Prisma 3T  
**Subjects:** 15 (analysis proceeds with complete subjects)  
**Sessions:** Single session (ses-dmnelf)  
**Tasks:**
- `rest` — 2 runs × ~350 volumes (TR=1.2s) — used to fit microstate maps
- `shortrest` — 1 run × ~100 volumes — held-out near-transfer test
- `feedback` — 4 runs × ~125 volumes — decoder training + far-transfer test

**EEG:** 32-channel BrainProducts MRI-compatible, preprocessed to
31 channels, average reference, 200 Hz, BCG artifact corrected.

**fMRI:** TR=1.2s, 2mm isotropic, preprocessed with fMRIPrep 24.1.1,
MNI152NLin6Asym space, res-2.

---

## Step 0 — DiFuMo-64 Timeseries Extraction

**Script:** `00_extract_difumo.py`

### What
Extract BOLD signal from 64 predefined brain parcels from the
fMRIPrep preprocessed images.

### How
We use nilearn's `NiftiMapsMasker` with the DiFuMo-64 probabilistic
atlas. Each parcel's timeseries is the weighted average of all voxels
within that parcel's probability map, yielding a (n_volumes, 64)
matrix per run. Output TSV columns: `volume`, `time`, `ROI_01`..`ROI_64`.

### Why DiFuMo-64
DiFuMo (Dictionary of Functional Modes) parcels are data-driven,
overlapping, and anatomically meaningful. At 64 components they
provide good coverage of DMN and CEN with minimal spatial leakage
between networks — critical for computing a clean PDA signal.

### Inputs
fmriprep_24.1.1/{subject}/ses-dmnelf/func/
{subject}_ses-dmnelf_task-{task}_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz

### Outputs
difumo_timeseries/
{subject}_ses-dmnelf_task-{task}_run-{run}_desc-difumo64_timeseries.tsv

### Reference
Dadi et al. (2020). Fine-grain atlases of functional modes for fMRI
analysis. *NeuroImage*, 221, 117126.

---

## Step 1 — Microstate Map Fitting

**Script:** `01_fit_microstates.py`

### What
Learn 7 prototype scalp topographies (microstate maps) that summarize
the dominant spatial configurations of the resting-state EEG signal.

### How
1. Load all rest EEG runs across all subjects (pooled)
2. Extract samples at **GFP peaks only** — moments when the scalp
   field is most clearly defined (standard approach, Lehmann et al. 1987)
3. Reject outlier peaks (GFP > 3 SD) to avoid artifact contamination
4. Run **polarity-invariant k-means** with k=7 and 20 random restarts
5. Select the solution with highest Global Explained Variance (GEV)

Polarity invariance: EEG topographies flip polarity ~every 50ms due
to the oscillatory nature of the signal. The same underlying neural
generator produces both polarities. Standard microstate analysis
ignores polarity during template assignment (Michel & Koenig 2018).

### Why k=7 and not k=4
The canonical 4-map solution (Koenig et al. 2002) merges microstate C
(DMN, self-referential) and microstate E (salience network) into one
cluster because their scalp topographies are spatially similar
(r > 0.7). Custo et al. (2017) showed that k=7 separates these
cleanly. Tarailis et al. (2023) systematically reviewed 50 studies
and confirmed that k=4 causes systematic misattribution of DMN vs
salience network dynamics — the C/E confusion is the most important
methodological finding in the modern microstate literature.

### Microstate functional assignments (Custo 2017, Tarailis 2023)
| Map | Network | Sources | Expected PDA role |
|-----|---------|---------|-------------------|
| MS A | Auditory/arousal | Temporal cortex | Minimal |
| MS B | Visual/self | Occipital, precuneus | Minimal |
| MS C | DMN (self-referential) | PCC, precuneus, angular | **Negative weight** |
| MS D | FPN/CEN (executive) | Right IPS, dlPFC | **Positive weight** |
| MS E | Salience | dACC, insula, mPFC | Secondary |
| MS F | Anterior DMN | mPFC, anterior DMN | Secondary |
| MS G | Somatosensory | Right parietal, cerebellum | Minimal |

### Outputs
microstates/
templates.npy   — (7, 31) float32, L2-normalized prototype maps
gev.npy         — (1,) best GEV across restarts

### References
- Custo et al. (2017). Electroencephalographic resting-state networks:
  source localization of microstates. *Brain Connectivity*, 7(9), 671–682.
- Tarailis et al. (2023). The functional aspects of resting EEG
  microstates: a systematic review. *Brain Topography*, 37, 181–217.
- Michel & Koenig (2018). EEG microstates as a tool for studying the
  temporal dynamics of whole-brain neuronal networks. *NeuroImage*, 180.
- Koenig et al. (2002). Millisecond by millisecond, year by year.
  *NeuroImage*, 16(1), 41–48.

---

## Step 2 — TESS Features

**Script:** `02_tess_features.py`

### What
For every EEG sample in every run, compute how strongly each of the
7 microstate topographies is expressed — yielding continuous time
courses that can be linked to fMRI BOLD.

### How — TESS Stage 1 (Spatial GLM Projection)
At each EEG sample t, the instantaneous scalp topography v(t) is
projected onto each template map via matrix multiplication:
T-hat(t) = Templates × v(t)        shape: (7, n_samples)

This gives 7 continuous T-hat coefficients per sample — one per
microstate. Unlike binary winner-takes-all labeling, these preserve
signal amplitude and within-TR temporal dynamics.

### How — HRF Convolution + Downsampling
Each T-hat timecourse is convolved with a canonical double-gamma HRF
(Glover 1999) and downsampled to the fMRI TR (1.2s) by averaging
within each TR window. This accounts for the ~6s neurovascular delay
between electrical brain state and BOLD signal.

Two additional features per TR:
- **GFP** (Global Field Power) = spatial std across channels,
  reflects overall signal strength
- **GMD** (Global Map Dissimilarity) = rate of topographic change,
  reflects microstate transitions

### Why TESS over binary labels
Binary winner-takes-all aggregation discards amplitude and temporal
dynamics within each TR. At 200 Hz EEG and TR=1.2s, each volume
contains 240 EEG samples — a binary label loses all this information.
TESS continuous T-hat coefficients preserve it.

Our previous pipeline with binary features achieved r~0.025 with PDA.
The improvement of TESS over binary labeling has been demonstrated by
Custo et al. (2017) who showed significantly higher BOLD prediction
accuracy using continuous spatial projections.

### Feature matrix
(n_vols, 9) float32
Columns: T-hat_A, T-hat_B, T-hat_C, T-hat_D,
T-hat_E, T-hat_F, T-hat_G, GFP, GMD

### Outputs
features/
{subject}_task-{task}_run-{run}_features.npy   — (n_vols, 9)

### References
- Custo et al. (2014). Generalized TESS. *Brain Topography*, 27(1).
- Britz et al. (2010). BOLD correlates of EEG topography reveal rapid
  resting-state network dynamics. *NeuroImage*, 52(4), 1160–1170.
- Glover (1999). Deconvolution of impulse response in event-related
  BOLD fMRI. *NeuroImage*, 9(4), 416–429.

---

## Step 3 — PDA Target Computation

**Script:** `03_compute_pda.py`

### What
Compute the neurofeedback target signal (PDA) from the DiFuMo-64
timeseries, aligned to what subjects actually received as feedback.

### Definition
PDA(t) = mean(CEN_z(t)) - mean(DMN_z(t))
Positive PDA = CEN dominant = ball moves UP on screen
Negative PDA = DMN dominant = ball moves DOWN on screen

### MURFI Alignment
The MURFI real-time system (Hinds et al. 2011) z-scores each voxel's
BOLD residual relative to the first 30s baseline (25 volumes at
TR=1.2s) at the start of each run:

```python
def baseline_zscore(x, n=25):
    mu  = x[:n].mean()
    sig = x[:n].std()
    return (x - mu) / sig
```

We apply this same normalization to each DiFuMo parcel timeseries
before computing PDA. This ensures our offline decoder target is
conceptually identical to what subjects received as neurofeedback.

### Network parcels (DiFuMo-64, 0-based indices)
Verified against labels_64_dictionary.csv (Dadi et al. 2020):

**DMN:** `[3, 6, 22, 29, 35, 38, 58, 60]`
- ROI_04 PCC, ROI_07 STS/angular gyrus, ROI_23 ACC,
  ROI_30 mPFC, ROI_36 angular gyrus sup, ROI_39 dmPFC,
  ROI_59 angular gyrus inf, ROI_61 MTG

**CEN:** `[4, 31, 47, 48, 50, 51]`
- ROI_05 parieto-occipital sulcus, ROI_32 IPS,
  ROI_48 IPS RH, ROI_49 IFsulcus, ROI_51 MFG/dlPFC, ROI_52 IFG

### Note on DMN-CEN correlation
Group mean DMN-CEN r = +0.636 in this dataset. This positive
correlation reflects shared global signal in the absence of GSR
(Murphy et al. 2009). Per the GSR debate (Fox et al. 2005), we do
not apply GSR as it mathematically forces anticorrelation. The TESS
T-hat time courses capture moment-to-moment topographic dynamics
that can predict PDA variance independently of mean network correlation.

### Outputs
targets/
{subject}_task-{task}_run-{run}_pda.npy   — (n_vols,) float32

### References
- Hinds et al. (2011). Computing moment-to-moment BOLD activation
  for real-time neurofeedback. *NeuroImage*, 54(1), 361–368.
- Bloom et al. (2023). Mindfulness-based real-time fMRI neurofeedback:
  a randomized controlled trial. *BMC Psychiatry*, 23, 757.
- Murphy et al. (2009). The impact of global signal regression.
  *Journal of Neuroscience*, 29(38), 13513–13531.
- Dadi et al. (2020). DiFuMo atlas. *NeuroImage*, 221, 117126.

---

## Step 4 — Decoder Training

**Script:** `04_train_decoder.py`

### What
Train a regularized linear model to predict single-TR PDA from the
9 EEG microstate features.

### How
- **Model:** ElasticNet (Zou & Hastie 2005) — combines L1 sparsity
  and L2 stability, appropriate for correlated features
- **Cross-validation:** Leave-one-run-out (LORO) on feedback runs
  (4 runs × ~125 TR per subject = ~500 training samples)
- **Feature scaling:** z-score features and target before fitting
- **Hyperparameters:** α and L1 ratio selected via inner CV
- **One decoder per subject** (subject-specific model)

### Theoretical predictions
Based on Custo et al. (2017) source localization and Tarailis et al.
(2023) functional synthesis:

| Feature | Predicted weight | Rationale |
|---------|-----------------|-----------|
| T-hat_C | **Negative** | DMN sources (PCC/precuneus) → higher C = DMN dominant = lower PDA |
| T-hat_D | **Positive** | FPN sources (IPS/dlPFC) → higher D = CEN dominant = higher PDA |
| T-hat_E | Secondary negative | Salience/mPFC, partial DMN overlap |
| T-hat_F | Secondary negative | Anterior DMN |
| T-hat_A,B,G | Near zero | Sensory/somatosensory, no DMN/CEN involvement |
| GFP | Unknown | Overall signal amplitude |
| GMD | Unknown | Topographic change rate |

Testing these predictions validates the functional interpretability
of the learned decoder.

### Outputs
models/
{subject}_decoder.pkl       — fitted ElasticNet model
{subject}_weights.npy       — (9,) feature weights
{subject}_cv_scores.npy     — LORO CV r per fold

### References
- Zou & Hastie (2005). Regularization and variable selection via the
  elastic net. *JRSS-B*, 67(2), 301–320.
- Meir-Hasson et al. (2014). One-class FMRI-inspired EEG model for
  self-regulation training. *PLOS ONE*, 9(6).
- Meir-Hasson et al. (2016). An EEG finger-print of fMRI deep regional
  activation. *NeuroImage*, 131, 120–128.

---

## Step 5 — Evaluation

**Script:** `05_evaluate_decoder.py`

### What
Test decoder performance on held-out data using two test sets.

### Test sets
**Near transfer — shortrest:**
Same session, same subject, no active neurofeedback. Tests whether
the EEG-PDA relationship generalizes from active feedback to passive
rest. Positive result here would indicate microstate dynamics
reflect DMN/CEN balance independently of task engagement.

**Far transfer — feedback (LORO CV):**
Left-out feedback runs from training CV. Tests within-task
generalization. Primary performance metric.

### Metrics
- **Pearson r** — predicted vs actual PDA (primary)
- **R²** — variance explained
- Per-run breakdown
- Group-level mean and standard deviation

### Expected performance
Meir-Hasson et al. (2016) achieved r~0.3 with the EFP framework
(Stockwell transforms + ridge regression) on amygdala BOLD.
Our preliminary results with binary microstate features on 7 subjects
showed r~0.2. We target r > 0.25 with TESS features on 15 subjects.

### Outputs
results/
evaluation.csv              — per-subject per-task r and R²
group_summary.csv           — group mean ± SD
qc/
scatter_predicted_vs_actual.png
decoder_weights.png

### References
- Meir-Hasson et al. (2016). An EEG finger-print of fMRI deep regional
  activation. *NeuroImage*, 131, 120–128.
- Zhang et al. (2023). Reducing DMN connectivity with mindfulness-based
  fMRI neurofeedback. *Molecular Psychiatry*, 28, 1–9.

---

## Full Reference List

1. Bloom et al. (2023). Mindfulness-based real-time fMRI neurofeedback:
   a randomized controlled trial to optimize dosing for depressed
   adolescents. *BMC Psychiatry*, 23, 757.

2. Britz et al. (2010). BOLD correlates of EEG topography reveal rapid
   resting-state network dynamics. *NeuroImage*, 52(4), 1160–1170.

3. Custo et al. (2014). Generalized TESS for EEG source localization.
   *Brain Topography*, 27(1), 95–105.

4. Custo et al. (2017). Electroencephalographic resting-state networks:
   source localization of microstates. *Brain Connectivity*, 7(9), 671–682.

5. Dadi et al. (2020). Fine-grain atlases of functional modes for fMRI
   analysis. *NeuroImage*, 221, 117126.

6. Fox et al. (2005). The human brain is intrinsically organized into
   dynamic, anticorrelated functional networks. *PNAS*, 102(27).

7. Glover (1999). Deconvolution of impulse response in event-related
   BOLD fMRI. *NeuroImage*, 9(4), 416–429.

8. Hinds et al. (2011). Computing moment-to-moment BOLD activation for
   real-time neurofeedback. *NeuroImage*, 54(1), 361–368.

9. Koenig et al. (2002). Millisecond by millisecond, year by year:
   normative EEG microstates and developmental stages. *NeuroImage*,
   16(1), 41–48.

10. Meir-Hasson et al. (2014). One-class FMRI-inspired EEG model for
    self-regulation training. *PLOS ONE*, 9(6), e99246.

11. Meir-Hasson et al. (2016). An EEG finger-print of fMRI deep regional
    activation. *NeuroImage*, 131, 120–128.

12. Michel & Koenig (2018). EEG microstates as a tool for studying the
    temporal dynamics of whole-brain neuronal networks: a review.
    *NeuroImage*, 180, 577–593.

13. Murphy et al. (2009). The impact of global signal regression on
    resting state correlations. *Journal of Neuroscience*, 29(38).

14. Tarailis et al. (2023). The functional aspects of resting EEG
    microstates: a systematic review. *Brain Topography*, 37, 181–217.

15. Zhang et al. (2023). Reducing default mode network connectivity
    with mindfulness-based fMRI neurofeedback: a pilot study.
    *Molecular Psychiatry*, 28, 1–9.

16. Zou & Hastie (2005). Regularization and variable selection via the
    elastic net. *JRSS-B*, 67(2), 301–320.