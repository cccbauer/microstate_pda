#!/usr/bin/env python3
"""
00b_extract_personal_masks_cluster.py
Adapted from pineuro mask_extraction.py.
Extract personalized DMN/CEN masks via CanICA + Yeo-17 correlation.
fMRIPrep BOLD already in MNI space — no ANTs registration needed.

CEN spatial constraint: skip components with midline posterior
centroid (|x|<15mm AND y<-45mm) to exclude PCC/precuneus,
which are DMN hubs that leak into ContA Yeo label.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import json
from pathlib import Path
import os
import warnings
import nibabel as nib

# ── Paths ─────────────────────────────────────────────
FMRIPREP_ROOT = Path("/projects/swglab/data/DMNELF/derivatives/fmriprep_24.1.1")
MASKS_ROOT    = Path("/projects/swglab/data/DMNELF/derivatives/network_masks")
DIFUMO_CACHE  = Path("/projects/swglab/software/nilearn_data")
MASKS_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["NILEARN_DATA"] = str(DIFUMO_CACHE)

SUBJECTS = ['sub-dmnelf001', 'sub-dmnelf002', 'sub-dmnelf003', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003', 'sub-dmnelf999']

# Parameters
N_COMPONENTS  = 35   # matching real-time pipeline (Bloom 2023)
N_VOXELS_MASK = 2000 # top N voxels per mask

# Yeo-17 labels (verified spatially):
# DMN: DefaultA(14)=PCC, DefaultB(15)=medTL, DefaultC(16)=mPFC
# CEN: ContA(11)=lat frontoparietal, ContB(12)=frontoparietal
YEO_DMN_LABELS = [14, 15, 16]
YEO_CEN_LABELS = [11, 12]

# CEN spatial constraint: exclude midline posterior components
# PCC/precuneus signature: |x| < 15mm AND y < -45mm
CEN_EXCL_X = 15.0
CEN_EXCL_Y = -45.0

# ── Load atlases ──────────────────────────────────────
print("=" * 55)
print("Loading atlases")
print("=" * 55)
from nilearn import datasets
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    yeo = datasets.fetch_atlas_yeo_2011()
yeo_path = yeo.thick_17 if hasattr(yeo, "thick_17") else yeo["thick_17"]
yeo_img  = nib.load(yeo_path)
yeo_data = np.squeeze(yeo_img.get_fdata()).astype(np.int32)
print("Yeo-17 shape: " + str(yeo_data.shape))

difumo = datasets.fetch_atlas_difumo(
    dimension=64, resolution_mm=2,
    data_dir=str(DIFUMO_CACHE)
)
difumo_maps = difumo.maps if hasattr(difumo, "maps") else difumo["maps"]
difumo_img  = nib.load(difumo_maps)
print("DiFuMo-64 shape: " + str(difumo_img.shape))

# ── Helpers ───────────────────────────────────────────
def resample_to(source_img, target_img, interpolation="nearest"):
    from nilearn.image import resample_to_img
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return resample_to_img(
            source_img, target_img,
            interpolation=interpolation
        )

def spatial_corr_masked(X_masked, ref_mask_flat):
    """
    Vectorized spatial correlation of all ICA components
    against a binary reference map, inside brain mask only.
    Adapted from pineuro spatial_correlation().
    X_masked:       (V, n_comp)
    ref_mask_flat:  (V,)
    Returns: (n_comp,)
    """
    X_dm   = X_masked - X_masked.mean(axis=0, keepdims=True)
    x_norm = np.sqrt((X_dm**2).sum(axis=0))
    y      = ref_mask_flat.astype(np.float64)
    y_dm   = y - y.mean()
    y_norm = np.sqrt((y_dm**2).sum())
    if y_norm < 1e-10:
        return np.zeros(X_masked.shape[1])
    num   = X_dm.T @ y_dm
    denom = x_norm * y_norm
    denom[denom < 1e-10] = 1e-10
    return num / denom

def zscore_combine(c1, c2):
    """
    Z-score each component then average.
    Adapted from pineuro combine_ica_components().
    Ensures equal contribution regardless of value range.
    """
    def zs(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else x - x.mean()
    return (zs(c1) + zs(c2)) / 2.0

def top_n_mask(component, n_voxels, affine):
    """Binary mask of top N positive voxels."""
    flat = component.ravel()
    pos  = np.where(flat > 0)[0]
    if len(pos) <= n_voxels:
        binary = (flat > 0).astype(np.uint8).reshape(component.shape)
    else:
        top    = np.argpartition(flat[pos], -n_voxels)[-n_voxels:]
        result = np.zeros_like(flat, dtype=np.uint8)
        result[pos[top]] = 1
        binary = result.reshape(component.shape)
    return nib.Nifti1Image(binary, affine)

def get_centroid_mni(ic, components, aff, n_top=500):
    """MNI centroid of the top N voxels of component ic."""
    comp   = components[..., ic]
    thresh = np.sort(np.abs(comp).ravel())[-n_top]
    xi, yi, zi = np.where(np.abs(comp) >= thresh)
    cx = float(xi.mean() * aff[0, 0] + aff[0, 3])
    cy = float(yi.mean() * aff[1, 1] + aff[1, 3])
    cz = float(zi.mean() * aff[2, 2] + aff[2, 3])
    return cx, cy, cz

def is_midline_posterior(ic, components, aff):
    """
    Returns True if component centroid is midline posterior.
    Flags PCC/precuneus (DMN hubs leaking into ContA Yeo label).
    Criterion: |x| < 15mm AND y < -45mm
    """
    cx, cy, cz = get_centroid_mni(ic, components, aff)
    return abs(cx) < CEN_EXCL_X and cy < CEN_EXCL_Y

def check_laterality(mask_img, label):
    data = mask_img.get_fdata()
    aff  = mask_img.affine
    xi, yi, zi = np.where(data > 0)
    mx = xi * aff[0, 0] + aff[0, 3]
    print("  " + label + ":"
          + "  L=" + str(int((mx < 0).sum()))
          + "  R=" + str(int((mx > 0).sum())))

# ── Main loop ─────────────────────────────────────────
print()
print("=" * 55)
print("Extracting personalized masks")
print("=" * 55)

for subject in SUBJECTS:
    print()
    print("--- " + subject + " ---")

    sub_dir = MASKS_ROOT / subject
    sub_dir.mkdir(parents=True, exist_ok=True)

    prefix      = subject + "_space-MNI152NLin6Asym_res-2"
    out_dmn     = sub_dir / (prefix + "_dmn_mask.nii.gz")
    out_cen     = sub_dir / (prefix + "_cen_mask.nii.gz")
    out_weights = sub_dir / (prefix + "_parcel_weights.json")

    if out_dmn.exists() and out_cen.exists() and out_weights.exists():
        print("  EXISTS (skip): " + subject)
        continue

    # ── Load BOLD and brain mask ───────────────────────
    bold_name = (subject
                 + "_ses-dmnelf_task-rest_run-01"
                 + "_space-MNI152NLin6Asym_res-2"
                 + "_desc-preproc_bold.nii.gz")
    bold_path = (FMRIPREP_ROOT / subject / "ses-dmnelf"
                 / "func" / bold_name)

    mask_name = (subject
                 + "_ses-dmnelf_task-rest_run-01"
                 + "_space-MNI152NLin6Asym_res-2"
                 + "_desc-brain_mask.nii.gz")
    mask_path = (FMRIPREP_ROOT / subject / "ses-dmnelf"
                 / "func" / mask_name)

    if not bold_path.exists():
        print("  MISSING BOLD: " + bold_name)
        continue

    if not mask_path.exists():
        anat_mask = (FMRIPREP_ROOT / subject / "ses-dmnelf"
                     / "anat" / (subject
                     + "_ses-dmnelf_space-MNI152NLin6Asym_res-2"
                     + "_desc-brain_mask.nii.gz"))
        if anat_mask.exists():
            mask_path = anat_mask
            print("  Using anat brain mask")
        else:
            mask_path = None
            print("  WARNING: no brain mask found")

    print("  BOLD: " + bold_name)

    # ── CanICA with brain mask ─────────────────────────
    from nilearn.decomposition import CanICA
    canica_kwargs = dict(
        n_components=N_COMPONENTS,
        memory=str(DIFUMO_CACHE / "nilearn_cache"),
        memory_level=2,
        random_state=42,
        standardize=True,
        n_jobs=-1,
        verbose=0
    )
    if mask_path is not None:
        canica_kwargs["mask"] = str(mask_path)
    canica = CanICA(**canica_kwargs)
    print("  Running CanICA (" + str(N_COMPONENTS) + " components)...")
    canica.fit([str(bold_path)])
    components_img = canica.components_img_
    components     = components_img.get_fdata()  # (x,y,z,n_comp)
    aff            = components_img.affine
    print("  Components shape: " + str(components.shape))

    # ── Resample Yeo to BOLD space ────────────────────
    yeo_dmn = np.zeros_like(yeo_data, dtype=np.float32)
    yeo_cen = np.zeros_like(yeo_data, dtype=np.float32)
    for l in YEO_DMN_LABELS:
        yeo_dmn[yeo_data == l] = 1.0
    for l in YEO_CEN_LABELS:
        yeo_cen[yeo_data == l] = 1.0

    yeo_dmn_r = np.squeeze(resample_to(
        nib.Nifti1Image(yeo_dmn, yeo_img.affine),
        components_img
    ).get_fdata())
    yeo_cen_r = np.squeeze(resample_to(
        nib.Nifti1Image(yeo_cen, yeo_img.affine),
        components_img
    ).get_fdata())

    print("  Yeo DMN voxels in BOLD space: "
          + str(int((yeo_dmn_r > 0).sum())))
    print("  Yeo CEN voxels in BOLD space: "
          + str(int((yeo_cen_r > 0).sum())))

    # ── Brain mask for correlation ─────────────────────
    brain_mask = (np.abs(components).sum(axis=-1) > 0)
    V = brain_mask.sum()
    print("  Brain mask voxels: " + str(int(V)))

    X_masked       = components[brain_mask]    # (V, n_comp)
    yeo_dmn_masked = yeo_dmn_r[brain_mask]     # (V,)
    yeo_cen_masked = yeo_cen_r[brain_mask]     # (V,)

    # ── Vectorized spatial correlations ───────────────
    corr_dmn = spatial_corr_masked(X_masked, yeo_dmn_masked)
    corr_cen = spatial_corr_masked(X_masked, yeo_cen_masked)

    dmn_sorted = np.abs(corr_dmn).argsort()[::-1]
    cen_sorted = np.abs(corr_cen).argsort()[::-1]

    print("  Top 5 DMN correlations:")
    for i in dmn_sorted[:5]:
        cx, cy, cz = get_centroid_mni(int(i), components, aff)
        print("    IC" + str(i)
              + "  r=" + "{:+.3f}".format(corr_dmn[i])
              + "  ctr=(" + str(round(cx,0))
              + "," + str(round(cy,0))
              + "," + str(round(cz,0)) + ")")

    print("  Top 5 CEN correlations:")
    for i in cen_sorted[:5]:
        cx, cy, cz = get_centroid_mni(int(i), components, aff)
        midline = is_midline_posterior(int(i), components, aff)
        flag = " [SKIP:PCC]" if midline else ""
        print("    IC" + str(i)
              + "  r=" + "{:+.3f}".format(corr_cen[i])
              + "  ctr=(" + str(round(cx,0))
              + "," + str(round(cy,0))
              + "," + str(round(cz,0)) + ")"
              + flag)

    # ── Select DMN: top 2 by absolute correlation ─────
    dmn_comp  = int(dmn_sorted[0])
    dmn_comp2 = int(dmn_sorted[1])
    used      = {dmn_comp, dmn_comp2}

    # ── Select CEN: top 2 excluding midline posterior ─
    cen_comp  = None
    cen_comp2 = None
    skipped   = []
    for ic in cen_sorted:
        ic = int(ic)
        if ic in used:
            continue
        if is_midline_posterior(ic, components, aff):
            cx, cy, cz = get_centroid_mni(ic, components, aff)
            skipped.append("IC" + str(ic)
                           + " (x=" + str(round(cx,1))
                           + " y=" + str(round(cy,1)) + ")")
            continue
        if cen_comp is None:
            cen_comp = ic
            used.add(ic)
        elif cen_comp2 is None:
            cen_comp2 = ic
            used.add(ic)
            break

    if skipped:
        print("  Skipped PCC components: " + ", ".join(skipped))

    # Fallback if constraint filtered too aggressively
    if cen_comp is None or cen_comp2 is None:
        print("  WARNING: fallback — relaxing spatial constraint")
        for ic in cen_sorted:
            ic = int(ic)
            if ic not in used:
                if cen_comp is None:
                    cen_comp = ic
                    used.add(ic)
                elif cen_comp2 is None:
                    cen_comp2 = ic
                    used.add(ic)
                    break

    print("  Selected DMN IC1: IC" + str(dmn_comp)
          + "  r=" + "{:+.3f}".format(corr_dmn[dmn_comp]))
    print("  Selected DMN IC2: IC" + str(dmn_comp2)
          + "  r=" + "{:+.3f}".format(corr_dmn[dmn_comp2]))
    print("  Selected CEN IC1: IC" + str(cen_comp)
          + "  r=" + "{:+.3f}".format(corr_cen[cen_comp]))
    print("  Selected CEN IC2: IC" + str(cen_comp2)
          + "  r=" + "{:+.3f}".format(corr_cen[cen_comp2]))

    # ── Sign-correct then z-score combine ─────────────
    dmn_c1 = components[..., dmn_comp].copy()
    dmn_c2 = components[..., dmn_comp2].copy()
    cen_c1 = components[..., cen_comp].copy()
    cen_c2 = components[..., cen_comp2].copy()

    if corr_dmn[dmn_comp] < 0:
        dmn_c1 = -dmn_c1
        print("  Flipped DMN IC1")
    if corr_dmn[dmn_comp2] < 0:
        dmn_c2 = -dmn_c2
        print("  Flipped DMN IC2")
    if corr_cen[cen_comp] < 0:
        cen_c1 = -cen_c1
        print("  Flipped CEN IC1")
    if corr_cen[cen_comp2] < 0:
        cen_c2 = -cen_c2
        print("  Flipped CEN IC2")

    dmn_combined = zscore_combine(dmn_c1, dmn_c2)
    cen_combined = zscore_combine(cen_c1, cen_c2)

    # ── Threshold to top N voxels ─────────────────────
    dmn_mask = top_n_mask(dmn_combined, N_VOXELS_MASK, aff)
    cen_mask = top_n_mask(cen_combined, N_VOXELS_MASK, aff)

    check_laterality(dmn_mask, "DMN")
    check_laterality(cen_mask, "CEN")

    # ── DiFuMo parcel overlap weights ─────────────────
    difumo_r = resample_to(
        difumo_img, components_img,
        interpolation="linear"
    ).get_fdata()  # (x,y,z,64)

    dmn_flat    = dmn_mask.get_fdata().ravel()
    cen_flat    = cen_mask.get_fdata().ravel()
    dmn_weights = np.zeros(64)
    cen_weights = np.zeros(64)
    for p in range(64):
        parcel         = difumo_r[..., p].ravel()
        dmn_weights[p] = float((parcel * dmn_flat).sum())
        cen_weights[p] = float((parcel * cen_flat).sum())

    if dmn_weights.sum() > 0:
        dmn_weights = dmn_weights / dmn_weights.sum()
    if cen_weights.sum() > 0:
        cen_weights = cen_weights / cen_weights.sum()

    dmn_top8 = np.argsort(dmn_weights)[::-1][:8].tolist()
    cen_top6 = np.argsort(cen_weights)[::-1][:6].tolist()

    print("  Top DMN parcels (personal): " + str(dmn_top8))
    print("  Group DMN parcels:          [3,6,22,29,35,38,58,60]")
    print("  Top CEN parcels (personal): " + str(cen_top6))
    print("  Group CEN parcels:          [4,31,47,48,50,51]")

    # ── Save masks ────────────────────────────────────
    nib.save(dmn_mask, str(out_dmn))
    nib.save(cen_mask, str(out_cen))
    print("  Saved: " + str(out_dmn.name))
    print("  Saved: " + str(out_cen.name))

    # ── Save weights JSON ─────────────────────────────
    weights_dict = {
        "subject":             subject,
        "dmn_weights":         dmn_weights.tolist(),
        "cen_weights":         cen_weights.tolist(),
        "dmn_top8":            dmn_top8,
        "cen_top6":            cen_top6,
        "dmn_ica_component1":  dmn_comp,
        "dmn_ica_component2":  dmn_comp2,
        "cen_ica_component1":  cen_comp,
        "cen_ica_component2":  cen_comp2,
        "dmn_yeo_corr1":       float(corr_dmn[dmn_comp]),
        "dmn_yeo_corr2":       float(corr_dmn[dmn_comp2]),
        "cen_yeo_corr1":       float(corr_cen[cen_comp]),
        "cen_yeo_corr2":       float(corr_cen[cen_comp2]),
        "group_dmn_idx":       [3,6,22,29,35,38,58,60],
        "group_cen_idx":       [4,31,47,48,50,51],
        "n_voxels_mask":       N_VOXELS_MASK,
        "n_ica_components":    N_COMPONENTS,
        "yeo_dmn_labels":      YEO_DMN_LABELS,
        "yeo_cen_labels":      YEO_CEN_LABELS,
        "cen_excl_x":          CEN_EXCL_X,
        "cen_excl_y":          CEN_EXCL_Y,
        "cen_skipped_pcc":     skipped,
    }
    with open(str(out_weights), "w") as f:
        json.dump(weights_dict, f, indent=2)
    print("  Saved: " + str(out_weights.name))

print()
print("=" * 55)
print("DONE")
print("=" * 55)