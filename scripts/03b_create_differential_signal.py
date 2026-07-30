# ============================================================
# 03B_DIFFERENTIAL_CHECK_PLOTS.PY  (FILTERED VERSION)
# Purpose:
# Create Spike2-like visual plots of differential signals from
# the FILTERED inventory (output of 03a filtered).
# Reads .npy analysis files (float32).
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt


# ============================================================
# 1. SETTINGS
# ============================================================

FS = 1250
CABLE = "Cable1"

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    f"03a_differential_summary_filtered_{CABLE}.csv"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    f"03b_differential_check_plots_filtered_{CABLE}"
)
os.makedirs(PLOT_DIR, exist_ok=True)

PLOT_SECONDS = 30
START_SECONDS = 0
LOWCUT = 1
HIGHCUT = 30
Y_LIMIT = 1.2


# ============================================================
# 2. FILTER FUNCTION
# ============================================================

def bandpass_filter(signal, fs, lowcut=1, highcut=30, order=4):
    nyquist = fs / 2
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)


# ============================================================
# 3. LOAD SUMMARY
# ============================================================

if not os.path.exists(SUMMARY_PATH):
    raise FileNotFoundError(
        f"Summary not found:\n{SUMMARY_PATH}\n"
        f"Run step 03a first."
    )

summary_df = pd.read_csv(SUMMARY_PATH)

print(f"\nLoaded summary: {SUMMARY_PATH}")
print(f"Recordings: {len(summary_df)}")


# Pick path column
if "diff_npy_path" in summary_df.columns:
    path_column = "diff_npy_path"
elif "diff_eeg_path" in summary_df.columns:
    path_column = "diff_eeg_path"
else:
    raise ValueError("Summary missing diff_npy_path / diff_eeg_path.")


# ============================================================
# 4. SELECT ONE EXAMPLE RECORDING PER MOUSE
# ============================================================

example_df = (
    summary_df.sort_values(["group", "mouse", "original_file"])
    .groupby(["group", "mouse"])
    .first()
    .reset_index()
)

print("\nExample recordings (one per mouse):")
print(example_df[["mouse", "group", "differential", "original_file"]])


# ============================================================
# 5. TIME AXIS
# ============================================================

start_sample = int(START_SECONDS * FS)
end_sample = int((START_SECONDS + PLOT_SECONDS) * FS)
time = np.arange(start_sample, end_sample) / FS


# ============================================================
# 6. CREATE PLOTS
# ============================================================

for _, row in example_df.iterrows():

    mouse = row["mouse"]
    group = row["group"]
    differential = row["differential"]
    filename = row["original_file"]
    diff_path = row[path_column]

    if not os.path.exists(diff_path):
        print(f"\nFile not found, skipping: {diff_path}")
        continue

    signal = np.load(diff_path)
    if len(signal) < end_sample:
        print(f"\nRecording too short, skipping: {filename}")
        continue

    signal = signal.astype(np.float32)
    filtered = bandpass_filter(signal, FS, LOWCUT, HIGHCUT, order=4)

    segment = filtered[start_sample:end_sample]
    max_abs = np.max(np.abs(segment))
    segment_plot = segment / max_abs if max_abs > 0 else segment

    plt.figure(figsize=(14, 4))
    plt.plot(time, segment_plot, linewidth=0.6)
    plt.ylim(-Y_LIMIT, Y_LIMIT)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized amplitude")
    plt.title(
        f"Spike2-like Differential View ({LOWCUT}-{HIGHCUT} Hz)\n"
        f"Mouse {mouse} | {group} | {differential}\n{filename}"
    )
    plt.tight_layout()

    output_name = (
        f"Mouse{mouse}_{group}_{differential}_"
        f"spike2_like_{LOWCUT}_{HIGHCUT}Hz.png"
    )
    output_name = output_name.replace("-", "_").replace(" ", "_").replace("/", "_")
    output_path = os.path.join(PLOT_DIR, output_name)

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


print("\nSTEP 3B (filtered) finished successfully.")