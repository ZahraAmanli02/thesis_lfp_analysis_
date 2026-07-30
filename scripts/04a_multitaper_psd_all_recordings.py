# ============================================================
# 04A_MULTITAPER_PSD_ALL_RECORDINGS.PY  
# Purpose:
# Compute multitaper PSD for the differential recordings
# (output of step 03a filtered) — recording-level, no recovery.
#
# Reads .npy differential files (float32).
#
# Preprocessing before PSD:
#   - NONE on the differential signal:
#       * no 50 Hz notch (line noise cancels in the channel subtraction)
#       * no per-segment mean removal (signal already high-pass > 1 Hz)
#     The notch/detrend switches exist only for optional diagnostics.
#
# Input:
#   outputs/03a_differential_summary_filtered_<CABLE>.csv
#
# Output:
#   outputs/04a_multitaper_psd_filtered_<CABLE>/
#     - one PSD CSV per recording
#     - one PSD PNG per recording
#   outputs/04a_multitaper_psd_summary_filtered_<CABLE>.csv
#     - master table; carries metadata (days_on_diet, body_weight, ...)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal.windows import dpss
from scipy.signal import iirnotch, filtfilt


# ============================================================
# 1. SETTINGS
# ============================================================

FS = 1250

NFFT = 2048
WIN_LENGTH = NFFT
N_OVERLAP = WIN_LENGTH // 2
WIN_STEP = WIN_LENGTH - N_OVERLAP

NW = 3
N_TAPERS = 2 * NW - 1

FREQ_MIN = 1
FREQ_MAX = 140

# --- Preprocessing options ---
#     DIFFERENTIAL (current-flow-density) signals:
#   - No 50 Hz notch: line noise is common to both channels and is
#     already cancelled by the subtraction; a notch would only distort
#     the spectrum (the line is never exactly 50 Hz).
#   - No per-segment mean removal / detrending: the signal is already
#     high-pass filtered above 1 Hz at acquisition, so there is no
#     baseline drift to remove.
# Both are OFF. Kept as switches only for optional diagnostics
# (e.g. to verify there is really no 50 Hz line in the CFD).
APPLY_NOTCH = False      # do NOT notch differential signals
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
APPLY_DETREND = False    # do NOT remove per-segment mean (no drift to remove)

CABLE = "Cable1"

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Input: filtered differential summary from 03a
SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    f"03a_differential_summary_filtered_{CABLE}.csv"
)

# Output folders
PSD_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    f"04a_multitaper_psd_filtered_{CABLE}"
)
PSD_CSV_DIR = os.path.join(PSD_OUTPUT_DIR, "csv")
PSD_FIG_DIR = os.path.join(PSD_OUTPUT_DIR, "figures")

os.makedirs(PSD_OUTPUT_DIR, exist_ok=True)
os.makedirs(PSD_CSV_DIR, exist_ok=True)
os.makedirs(PSD_FIG_DIR, exist_ok=True)


# ============================================================
# 2. PREPROCESSING — NOTCH FILTER
# Removes 50 Hz line noise and its 100 Hz harmonic.
# Important so gamma bands (30-100 Hz) are not contaminated.
# ============================================================

def apply_notch_filters(signal, fs, base_freq=50.0, q=30.0):
    signal = signal.astype(np.float64)

    b, a = iirnotch(w0=base_freq, Q=q, fs=fs)
    signal = filtfilt(b, a, signal)

    harmonic = 2 * base_freq
    if harmonic < fs / 2:
        b, a = iirnotch(w0=harmonic, Q=q, fs=fs)
        signal = filtfilt(b, a, signal)

    return signal


# ============================================================
# 3. MULTITAPER PSD — chunk -> taper -> FFT -> average
# Each chunk is detrended so DC drift does not inflate delta.
# ============================================================

def multitaper_psd(signal, detrend=True):
    signal = signal.astype(np.float64)
    n_samples = len(signal)
    n_chunks = int(np.floor((n_samples - WIN_LENGTH) / WIN_STEP))

    if n_chunks <= 0:
        raise ValueError("Signal too short for the chosen window length.")

    tapers = dpss(WIN_LENGTH, NW, Kmax=N_TAPERS, sym=False)

    f = np.fft.rfftfreq(NFFT, d=1 / FS)
    freq_mask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
    f_selected = f[freq_mask]

    y = []

    for i in range(n_chunks):
        start = i * WIN_STEP
        end = start + WIN_LENGTH
        segment = signal[start:end]

        if detrend:
            segment = segment - np.mean(segment)

        tapered = tapers * segment[np.newaxis, :]
        fft_out = np.fft.rfft(tapered, n=NFFT, axis=1)
        power = np.mean(np.abs(fft_out) ** 2, axis=0)

        y.append(power[freq_mask])

    y = np.array(y)
    mean_power = np.mean(y, axis=0)

    return y, f_selected, mean_power


# ============================================================
# 4. LOAD FILTERED DIFFERENTIAL SUMMARY
# ============================================================

if not os.path.exists(SUMMARY_PATH):
    raise FileNotFoundError(
        f"Filtered differential summary not found:\n{SUMMARY_PATH}\n"
        f"Run step 03a (filtered) first."
    )

summary_df = pd.read_csv(SUMMARY_PATH)

print(f"\nLoaded filtered differential summary: {SUMMARY_PATH}")
print(f"Recordings: {len(summary_df)}")
print(summary_df.head())


# ============================================================
# 5. PICK PATH COLUMN
# ============================================================

if "diff_npy_path" in summary_df.columns:
    path_column = "diff_npy_path"
elif "diff_eeg_path" in summary_df.columns:
    path_column = "diff_eeg_path"
else:
    raise ValueError("Summary missing diff_npy_path / diff_eeg_path.")

print(f"\nUsing path column: {path_column}")


# ============================================================
# 6. PROCESS ALL RECORDINGS
# ============================================================

psd_summary_rows = []

print("\nPreprocessing:")
print(f"  Notch filter: {APPLY_NOTCH} (at {NOTCH_FREQ} Hz + harmonic)")
print(f"  Per-segment detrend: {APPLY_DETREND}\n")

for idx, row in summary_df.iterrows():

    mouse_id = int(row["mouse"])
    group = row["group"]
    cable = row["cable"]

    diff_file = row["differential_file"]
    diff_path = row[path_column]

    base_name = diff_file.replace(".eeg", "").replace(".npy", "")

    # Carry forward the metadata fields from 03a
    metadata = {
        "recording_date":     row.get("recording_date", ""),
        "diet_phase":         row.get("diet_phase", ""),
        "days_on_diet":       row.get("days_on_diet", np.nan),
        "days_in_recovery":   row.get("days_in_recovery", np.nan),
        "body_weight":        row.get("body_weight", np.nan),
        "body_weight_source": row.get("body_weight_source", ""),
        "estrous_phase":      row.get("estrous_phase", ""),
        "duration_min":       row.get("duration_min", np.nan),
        "differential":       row.get("differential", ""),
    }

    print(f"Processing: {diff_file}")

    # ----- Build a row template that always has metadata -----
    row_template = {
        "mouse": mouse_id,
        "group": group,
        "cable": cable,
        "differential_file": diff_file,
        **metadata,
        "status": "OK",
        "error": "",
        "n_time_windows": np.nan,
        "n_frequencies": np.nan,
        "psd_csv_path": "",
        "psd_figure_path": "",
        "notch_applied": APPLY_NOTCH,
        "detrend_applied": APPLY_DETREND,
    }

    # ----- Existence check -----
    if not os.path.exists(diff_path):
        print(f"File not found, skipping.")
        row_template["status"] = "FAILED"
        row_template["error"] = "file not found"
        psd_summary_rows.append(row_template)
        continue

    # ----- Load -----
    try:
        signal = np.load(diff_path)
    except Exception as e:
        print(f"Failed to load: {e}")
        row_template["status"] = "FAILED"
        row_template["error"] = f"load error: {e}"
        psd_summary_rows.append(row_template)
        continue

    # ----- Notch -----
    if APPLY_NOTCH:
        try:
            signal = apply_notch_filters(
                signal, fs=FS, base_freq=NOTCH_FREQ, q=NOTCH_Q
            )
        except Exception as e:
            print(f"Notch failed: {e}")
            row_template["status"] = "FAILED"
            row_template["error"] = f"notch error: {e}"
            psd_summary_rows.append(row_template)
            continue

    # ----- PSD -----
    try:
        y, f, mean_power = multitaper_psd(signal, detrend=APPLY_DETREND)
    except Exception as e:
        print(f"PSD failed: {e}")
        row_template["status"] = "FAILED"
        row_template["error"] = str(e)
        psd_summary_rows.append(row_template)
        continue

    # ----- Save PSD CSV -----
    psd_df = pd.DataFrame({
        "frequency_Hz": f,
        "mean_power": mean_power,
    })
    psd_csv_path = os.path.join(
        PSD_CSV_DIR, f"{base_name}_multitaper_psd.csv"
    )
    psd_df.to_csv(psd_csv_path, index=False)

    # ----- Save PSD figure -----
    fig_path = os.path.join(
        PSD_FIG_DIR, f"{base_name}_multitaper_psd.png"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(f, mean_power, linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")

    notch_label = "ON" if APPLY_NOTCH else "OFF"
    plt.title(
        f"Multitaper PSD | Mouse {mouse_id} | {group}\n"
        f"days_on_diet={metadata['days_on_diet']}, "
        f"weight={metadata['body_weight']:.1f}g | notch={notch_label}"
        if not pd.isna(metadata["body_weight"]) else
        f"Multitaper PSD | Mouse {mouse_id} | {group} | notch={notch_label}"
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # ----- Fill in success info -----
    row_template["n_time_windows"] = y.shape[0]
    row_template["n_frequencies"] = y.shape[1]
    row_template["psd_csv_path"] = psd_csv_path
    row_template["psd_figure_path"] = fig_path

    psd_summary_rows.append(row_template)

    print(f"  ✓ done ({y.shape[0]} windows)")


# ============================================================
# 7. SAVE PSD SUMMARY
# ============================================================

psd_summary_df = pd.DataFrame(psd_summary_rows)

psd_summary_path = os.path.join(
    OUTPUT_DIR,
    f"04a_multitaper_psd_summary_filtered_{CABLE}.csv"
)
psd_summary_df.to_csv(psd_summary_path, index=False)

print(f"\nSaved PSD summary:\n{psd_summary_path}")


# ============================================================
# 8. PRINT SUMMARY
# ============================================================

n_ok = (psd_summary_df["status"] == "OK").sum()
n_failed = (psd_summary_df["status"] == "FAILED").sum()

print(f"\nSuccessful PSDs: {n_ok}")
print(f"Failed PSDs: {n_failed}")

if n_failed > 0:
    print("\nFailures:")
    failed = psd_summary_df[psd_summary_df["status"] == "FAILED"]
    for _, r in failed.iterrows():
        print(f"  - {r['differential_file']}: {r['error']}")
else:
    print("All recordings processed successfully.")

if n_ok > 0:
    ok_df = psd_summary_df[psd_summary_df["status"] == "OK"]
    print("\nRecordings per group:")
    print(ok_df.groupby("group").size())
    print("\nRecordings per mouse:")
    print(ok_df.groupby(["group", "mouse"]).size())


print("\nSTEP 04A (filtered) finished successfully.")