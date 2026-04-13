#!/usr/bin/env python3
"""
eeg_preproc.py
Automated EEG preprocessing pipeline for DMNELF simultaneous EEG-fMRI data.

Usage:
    python eeg_preproc.py --subject sub-dmnelf009 --task rest --run 01
    python eeg_preproc.py --all                          # process all missing
    python eeg_preproc.py --subject sub-dmnelf009        # all runs for one subject

Steps:
    1. Load minimally preprocessed EDF (BVA gradient correction, 1kHz)
    2. Detect ECG channel, compute R-peaks (NeuroKit2)
    3. Auto-detect bad channels (variance + HF noise z-score)
    4. Annotate noisy edges (scanner ramp artifact)
    5. Bandpass filter (1-40 Hz)
    6. BCG correction (average artifact subtraction)
    7. ICA (29 components, ICLabel + cardiac correlation)
    8. Interpolate bad channels
    9. Average reference
    10. Save FIF + QC images
"""

import argparse
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
EEG_RAW_ROOT = Path("/projects/swglab/data/DMNELF/rawdata_eeg")
EEG_OUT_ROOT = Path("/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed")

SUBJECTS_EEG = [
    "sub-dmnelf001", "sub-dmnelf004", "sub-dmnelf005", "sub-dmnelf006",
    "sub-dmnelf007", "sub-dmnelf008", "sub-dmnelf009", "sub-dmnelf010",
    "sub-dmnelf011", "sub-dmnelf1001", "sub-dmnelf1002", "sub-dmnelf1003",
]

TASKS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

MISSING_RUNS = {
    "sub-dmnelf1002": [("rest", "02")],
    "sub-dmnelf1003": [("feedback", "04")],
}

# ── Preprocessing parameters ───────────────────────────────────────────────
SFREQ_TARGET    = 250.0   # downsample to 250 Hz
HIGHPASS        = 1.0     # Hz
LOWPASS         = 40.0    # Hz
N_ICA           = 29      # ICA components
BCG_TMIN        = -0.2    # s before R-peak
BCG_TMAX        = 0.6     # s after R-peak
N_VOXELS_MASK   = 2000
BAD_CH_Z        = 3.0     # z-score threshold for bad channels
BAD_HF_Z        = 2.5     # z-score for HF noise
EDGE_SEARCH_SEC = 5.0     # seconds to search for noisy edges
EDGE_THRESH     = 3.0     # x stable RMS


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_run(subject, task, run, overwrite=False):
    import mne
    mne.set_log_level('WARNING')

    session  = "ses-dmnelf"
    eeg_dir  = EEG_RAW_ROOT / subject / session / "eeg"
    out_dir  = EEG_OUT_ROOT / subject / session / "eeg"
    qc_dir   = out_dir / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    edf_file   = eeg_dir / (subject + "_" + session
                            + "_task-" + task + "_run-" + run
                            + "_desc-bvaAC1kHz_eeg.edf")
    output_fif = out_dir / (subject + "_" + session
                            + "_task-" + task + "_run-" + run
                            + "_desc-preproc_eeg.fif")

    tag = subject + " task-" + task + " run-" + run

    print("\n" + "=" * 60)
    print(tag)
    print("=" * 60)

    if output_fif.exists() and not overwrite:
        print("  EXISTS (skip): " + output_fif.name)
        return True

    if not edf_file.exists():
        print("  MISSING EDF: " + edf_file.name)
        return False

    # ── 1. Load ────────────────────────────────────────────
    print("  Loading EDF...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)

    # Set ECG channel type
    ecg_ch_name = None
    for ch in raw.ch_names:
        if any(x in ch.upper() for x in ['ECG', 'EKG', 'HEART', 'CARDIO']):
            raw.set_channel_types({ch: 'ecg'})
            ecg_ch_name = ch
            break

    print("  sfreq=" + str(raw.info['sfreq']) + "Hz"
          + "  channels=" + str(len(raw.ch_names))
          + "  duration=" + str(round(raw.times[-1], 1)) + "s"
          + ("  ECG=" + ecg_ch_name if ecg_ch_name else "  no ECG"))

    # ── 2. R-peak detection ────────────────────────────────
    cardiac_events = None
    cardiac_freq   = 1.2  # default ~72 BPM
    avg_hr         = 72.0

    if ecg_ch_name is not None:
        try:
            import neurokit2 as nk
            ecg_data    = raw.get_data(picks=ecg_ch_name)[0]
            ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=raw.info['sfreq'])
            _, rpeaks   = nk.ecg_peaks(ecg_cleaned, sampling_rate=raw.info['sfreq'])
            r_idx       = rpeaks['ECG_R_Peaks']
            if len(r_idx) > 1:
                avg_hr       = len(r_idx) / (raw.times[-1] / 60)
                cardiac_freq = avg_hr / 60
                cardiac_events = r_idx
                print("  R-peaks=" + str(len(r_idx))
                      + "  HR=" + str(round(avg_hr, 1)) + "BPM")
        except Exception as e:
            print("  R-peak detection failed: " + str(e))

    # ── 3. Bad channel detection ───────────────────────────
    eeg_data     = raw.get_data(picks='eeg')
    eeg_ch_names = [ch for ch in raw.ch_names if ch != ecg_ch_name]

    ch_std   = eeg_data.std(axis=1)
    z_scores = (ch_std - ch_std.mean()) / (ch_std.std() + 1e-10)

    raw_hf   = raw.copy().filter(40, 50, picks='eeg', verbose=False)
    hf_data  = raw_hf.get_data(picks='eeg')
    hf_power = hf_data.var(axis=1)
    hf_z     = (hf_power - hf_power.mean()) / (hf_power.std() + 1e-10)
    del raw_hf

    auto_bads = list(set(
        [eeg_ch_names[i] for i, z in enumerate(z_scores) if abs(z) > BAD_CH_Z] +
        [eeg_ch_names[i] for i, z in enumerate(hf_z)     if z > BAD_HF_Z]
    ))
    if auto_bads:
        raw.info['bads'] = auto_bads
        print("  Bad channels: " + str(auto_bads))
    else:
        print("  Bad channels: none")

    # ── 4. Annotate noisy edges ────────────────────────────
    rms        = np.sqrt((eeg_data**2).mean(axis=0))
    sfreq      = raw.info['sfreq']
    n_search   = int(EDGE_SEARCH_SEC * sfreq)
    mid_start  = int(len(rms) * 0.1)
    mid_end    = int(len(rms) * 0.9)
    stable_rms = np.median(rms[mid_start:mid_end])
    threshold  = stable_rms * EDGE_THRESH

    onset_end = 0.0
    for i in range(n_search):
        if rms[i] > threshold:
            onset_end = (i + 1) / sfreq
        else:
            break

    offset_start = raw.times[-1]
    for i in range(len(rms)-1, len(rms)-n_search-1, -1):
        if rms[i] > threshold:
            offset_start = i / sfreq
        else:
            break

    new_annots = []
    if onset_end > 0.1:
        new_annots.append((0.0, onset_end, 'BAD_edge_start'))
    if offset_start < raw.times[-1] - 0.1:
        new_annots.append((offset_start, raw.times[-1] - offset_start, 'BAD_edge_end'))
    if new_annots:
        from mne import Annotations
        annots = Annotations(
            onset=[a[0] for a in new_annots],
            duration=[a[1] for a in new_annots],
            description=[a[2] for a in new_annots],
            orig_time=raw.info.get('meas_date')
        )
        raw.set_annotations(raw.annotations + annots)
        print("  Edge annotations: " + str(len(new_annots)))

    # ── 5. QC image — raw ─────────────────────────────────
    _save_qc_raw(raw, eeg_ch_names, cardiac_events, auto_bads,
                 qc_dir, subject, session, task, run)

    # ── 6. Filter ─────────────────────────────────────────
    print("  Filtering " + str(HIGHPASS) + "-" + str(LOWPASS) + "Hz...")
    raw_filtered = raw.copy().filter(
        HIGHPASS, LOWPASS, picks='eeg',
        method='fir', verbose=False
    )

    # ── 7. BCG correction ─────────────────────────────────
    if cardiac_events is not None:
        print("  BCG correction...")
        events = np.column_stack([
            cardiac_events,
            np.zeros(len(cardiac_events), dtype=int),
            np.ones(len(cardiac_events),  dtype=int)
        ])
        try:
            epochs_bcg = mne.Epochs(
                raw_filtered, events, event_id=1,
                tmin=BCG_TMIN, tmax=BCG_TMAX,
                baseline=None, preload=True,
                picks='eeg', reject_by_annotation=True,
                verbose=False
            )
            if len(epochs_bcg) > 5:
                bcg_template = epochs_bcg.average()
                # Subtract template from raw
                template_data = bcg_template.data  # (n_ch, n_times)
                n_tmpl = template_data.shape[1]
                raw_data = raw_filtered.get_data(picks='eeg')
                for idx, r_samp in enumerate(cardiac_events):
                    onset = r_samp + int(BCG_TMIN * sfreq)
                    end   = onset + n_tmpl
                    if onset >= 0 and end <= raw_data.shape[1]:
                        raw_data[:, onset:end] -= template_data
                raw_filtered._data[mne.pick_types(raw_filtered.info, eeg=True)] = raw_data
                print("  BCG correction applied (" + str(len(epochs_bcg)) + " epochs)")
            else:
                print("  BCG skipped (too few epochs: " + str(len(epochs_bcg)) + ")")
        except Exception as e:
            print("  BCG failed: " + str(e))
    else:
        print("  BCG skipped (no cardiac events)")

    # ── 8. Downsample ─────────────────────────────────────
    if raw_filtered.info['sfreq'] > SFREQ_TARGET:
        print("  Downsampling to " + str(SFREQ_TARGET) + "Hz...")
        raw_filtered.resample(SFREQ_TARGET, verbose=False)

    # ── 9. ICA ────────────────────────────────────────────
    from mne.preprocessing import ICA
    n_eeg        = len(mne.pick_types(raw_filtered.info, eeg=True))
    n_components = min(N_ICA, n_eeg - 1)
    print("  ICA (" + str(n_components) + " components)...")

    ica = ICA(
        n_components=n_components,
        method='fastica',
        random_state=42,
        max_iter=500,
        verbose=False
    )
    ica.fit(raw_filtered, picks='eeg',
            reject_by_annotation=True, verbose=False)

    artifact_components = []

    # ICLabel
    try:
        from mne_icalabel import label_components
        ic_labels = label_components(raw_filtered, ica, method='iclabel')
        labels    = ic_labels['labels']
        artifact_components = [i for i, l in enumerate(labels) if l != 'brain']
        print("  ICLabel artifacts: " + str(artifact_components))
    except Exception as e:
        print("  ICLabel failed: " + str(e))
        labels = None

    # Cardiac correlation
    if ecg_ch_name is not None and cardiac_events is not None:
        from scipy import signal as scipy_signal
        sources    = ica.get_sources(raw_filtered).get_data()
        ecg_signal = raw_filtered.get_data(picks=ecg_ch_name)[0]
        sfreq_f    = raw_filtered.info['sfreq']
        cardiac_components = []
        for idx in range(n_components):
            freqs_w, psd = scipy_signal.welch(
                sources[idx], fs=sfreq_f, nperseg=1024, noverlap=512
            )
            c_idx  = np.argmin(np.abs(freqs_w - cardiac_freq))
            c2_idx = np.argmin(np.abs(freqs_w - 2*cardiac_freq))
            mean_p = np.mean(psd[(freqs_w > 0.5) & (freqs_w < 10)])
            if (psd[c_idx] > 3*mean_p or psd[c2_idx] > 3*mean_p):
                min_len = min(len(sources[idx]), len(ecg_signal))
                corr = abs(np.corrcoef(
                    sources[idx][:min_len], ecg_signal[:min_len]
                )[0, 1])
                if corr > 0.3:
                    cardiac_components.append(idx)
        artifact_components = sorted(set(artifact_components + cardiac_components))
        print("  Cardiac ICA: " + str(cardiac_components))

    if artifact_components:
        ica.exclude = artifact_components
        raw_clean   = raw_filtered.copy()
        ica.apply(raw_clean, verbose=False)
        print("  ICA excluded: " + str(artifact_components))
    else:
        raw_clean = raw_filtered
        print("  ICA: no components excluded")

    # ── 10. Interpolate + reference ───────────────────────
    if raw_clean.info['bads']:
        raw_clean.interpolate_bads(reset_bads=True, verbose=False)
        print("  Bad channels interpolated")

    raw_clean.set_eeg_reference('average', projection=False, verbose=False)
    print("  Average reference applied")

    # ── 11. QC image — final ──────────────────────────────
    _save_qc_final(raw_filtered, raw_clean,
                   qc_dir, subject, session, task, run, cardiac_freq)

    # ── 12. Save FIF ──────────────────────────────────────
    raw_clean.save(str(output_fif), overwrite=True, verbose=False)
    fsize = os.path.getsize(str(output_fif)) / 1e6
    print("  Saved: " + output_fif.name + "  (" + str(round(fsize, 1)) + "MB)")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# QC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _save_qc_raw(raw, eeg_ch_names, cardiac_events,
                 auto_bads, qc_dir, subject, session, task, run):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Panel 1: channel std
    eeg_data = raw.get_data(picks='eeg')
    ch_std   = eeg_data.std(axis=1)
    colors   = ['red' if c in auto_bads else 'steelblue' for c in eeg_ch_names]
    axes[0].bar(range(len(ch_std)), ch_std * 1e6, color=colors)
    axes[0].set_title(subject + " task-" + task + " run-" + run + " — Channel std")
    axes[0].set_xticks(range(len(eeg_ch_names)))
    axes[0].set_xticklabels(eeg_ch_names, rotation=90, fontsize=7)
    axes[0].set_ylabel("Std (µV)")

    # Panel 2: first 20s
    sfreq     = raw.info['sfreq']
    n_plot    = int(20 * sfreq)
    data_plot = eeg_data[:, :n_plot]
    t_plot    = raw.times[:n_plot]
    offset    = 0
    for i, ch in enumerate(eeg_ch_names):
        color = 'red' if ch in auto_bads else 'black'
        axes[1].plot(t_plot, data_plot[i] * 1e6 + offset,
                     color=color, linewidth=0.3, alpha=0.7)
        offset += 150
    for ann in raw.annotations:
        if 'BAD' in ann['description']:
            axes[1].axvspan(ann['onset'],
                            min(ann['onset'] + ann['duration'], 20),
                            alpha=0.2, color='red')
    axes[1].set_title("First 20s (red=bad, shading=BAD annotation)")
    axes[1].set_ylabel("Channels (offset µV)")
    axes[1].set_xlabel("Time (s)")

    # Panel 3: PSD
    psd   = raw.compute_psd(fmax=50, picks='eeg', verbose=False)
    psds  = psd.get_data()
    freqs = psd.freqs
    axes[2].semilogy(freqs, psds.mean(axis=0))
    axes[2].fill_between(freqs, psds.min(axis=0), psds.max(axis=0), alpha=0.2)
    axes[2].set_title("PSD before BCG correction")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Power (V²/Hz)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = qc_dir / (subject + "_ses-dmnelf_task-" + task
                      + "_run-" + run + "_qc_raw.png")
    fig.savefig(str(fname), dpi=100)
    plt.close()


def _save_qc_final(raw_before, raw_after,
                   qc_dir, subject, session, task, run, cardiac_freq):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Panel 1: PSD comparison
    psd_b = raw_before.compute_psd(fmax=50, picks='eeg', verbose=False)
    psd_a = raw_after.compute_psd(fmax=50, picks='eeg', verbose=False)
    axes[0].semilogy(psd_b.freqs, psd_b.get_data().mean(axis=0),
                     'b--', linewidth=1.5, alpha=0.7, label='Before')
    axes[0].semilogy(psd_a.freqs, psd_a.get_data().mean(axis=0),
                     'g-',  linewidth=1.5, label='After')
    axes[0].axvspan(cardiac_freq - 0.2, cardiac_freq + 0.2,
                    alpha=0.15, color='red', label='cardiac')
    axes[0].set_title(subject + " task-" + task + " run-" + run + " — Final PSD")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (V²/Hz)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: time traces before
    sfreq  = raw_before.info['sfreq']
    t_mask = (raw_before.times >= 30) & (raw_before.times <= 50)
    d_b    = raw_before.get_data(picks='eeg')[:, t_mask]
    t_plot = raw_before.times[t_mask]
    offset = 0
    for i in range(min(15, d_b.shape[0])):
        axes[1].plot(t_plot, d_b[i] * 1e6 + offset,
                     linewidth=0.4, color='steelblue', alpha=0.7)
        offset += 150
    axes[1].set_title("Before (30-50s, first 15 channels)")
    axes[1].set_ylabel("Channels (offset µV)")

    # Panel 3: time traces after
    t_mask2 = (raw_after.times >= 30) & (raw_after.times <= 50)
    d_a     = raw_after.get_data(picks='eeg')[:, t_mask2]
    t_plot2 = raw_after.times[t_mask2]
    offset  = 0
    for i in range(min(15, d_a.shape[0])):
        axes[2].plot(t_plot2, d_a[i] * 1e6 + offset,
                     linewidth=0.4, color='green', alpha=0.7)
        offset += 150
    axes[2].set_title("After (30-50s, first 15 channels)")
    axes[2].set_ylabel("Channels (offset µV)")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    fname = qc_dir / (subject + "_ses-dmnelf_task-" + task
                      + "_run-" + run + "_qc_final.png")
    fig.savefig(str(fname), dpi=100)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DMNELF EEG preprocessing")
    parser.add_argument("--subject",   type=str, default=None)
    parser.add_argument("--task",      type=str, default=None)
    parser.add_argument("--run",       type=str, default=None)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all or (args.subject and not args.task):
        subjects = [args.subject] if args.subject else SUBJECTS_EEG
        n_ok = 0
        n_fail = 0
        for subject in subjects:
            for task, runs in TASKS.items():
                for run in runs:
                    if subject in MISSING_RUNS:
                        if (task, run) in MISSING_RUNS[subject]:
                            continue
                    ok = preprocess_run(subject, task, run, args.overwrite)
                    if ok:
                        n_ok += 1
                    else:
                        n_fail += 1
        print("\n" + "=" * 60)
        print("DONE  OK=" + str(n_ok) + "  FAILED=" + str(n_fail))
        print("=" * 60)

    elif args.subject and args.task and args.run:
        preprocess_run(args.subject, args.task, args.run, args.overwrite)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()