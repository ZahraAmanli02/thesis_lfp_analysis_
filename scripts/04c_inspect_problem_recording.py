# ============================================================
# 04C_INSPECT_PROBLEM_RECORDING.PY  (FILTERED VERSION)
# Purpose:
# Visually inspect the raw differential signal of one specific
# recording. This is the manual confirmation step of the quality
# screening: 04b proposes bad candidates from quality metrics,
# and here you look at each candidate's raw trace to decide whether
# it is a true artifact (e.g. saturation at +/-65535, flat line,
# non-LFP "boom-boom"). Confirmed-bad recordings are then listed in
# the CONFIRMED_BAD block of 04b, which the outlier test (04e) and
# feature extraction exclude.
#
# Tip: open
#   04b_bad_recording_check_filtered_<CABLE>/
#       04b_bad_recording_candidates_<CABLE>.csv
# to see which recordings 04b flagged, then inspect each here.
#
# Reads .npy differential file from step 03a (filtered).
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

CHECK_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    f"04c_problem_recording_check_filtered_{CABLE}"
)
os.makedirs(CHECK_OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. WHICH RECORDING TO INSPECT
# ------------------------------------------------------------
# Option A: exact differential file name
# Option B: mouse number + recording number
# ============================================================

PROBLEM_FILE = None         # e.g. "Mouse23_Cable1_HF_2026-03-26_13-09-11_DIFF_Ch2_minus_Ch3.npy"

TARGET_MOUSE = 9           # change to inspect a different mouse
RECORDING_NUMBER = 1        # 1 = first recording of that mouse


# ============================================================
# 3. LOAD SUMMARY
# ============================================================

if not os.path.exists(SUMMARY_PATH):
    raise FileNotFoundError(
        f"Differential summary not found:\n{SUMMARY_PATH}\n"
        f"Run step 03a (filtered) first."
    )

summary_df = pd.read_csv(SUMMARY_PATH)

if "diff_npy_path" in summary_df.columns:
    path_column = "diff_npy_path"
elif "diff_eeg_path" in summary_df.columns:
    path_column = "diff_eeg_path"
else:
    raise ValueError("Summary missing diff_npy_path / diff_eeg_path.")


# ============================================================
# 4. SELECT RECORDING
# ============================================================

if PROBLEM_FILE is not None:
    matches = summary_df[summary_df["differential_file"] == PROBLEM_FILE]
    if len(matches) == 0:
        raise ValueError(f"PROBLEM_FILE not in summary: {PROBLEM_FILE}")
    selected = matches.iloc[0]
    print(f"\nSelected by FILE: {PROBLEM_FILE}")
else:
    mouse_rows = summary_df[
        summary_df["mouse"] == TARGET_MOUSE
    ].reset_index(drop=True)
    if len(mouse_rows) == 0:
        raise ValueError(f"No differentials for Mouse {TARGET_MOUSE}.")
    if RECORDING_NUMBER < 1 or RECORDING_NUMBER > len(mouse_rows):
        raise ValueError(
            f"Mouse {TARGET_MOUSE} has {len(mouse_rows)} recordings, "
            f"but RECORDING_NUMBER = {RECORDING_NUMBER}."
        )
    selected = mouse_rows.iloc[RECORDING_NUMBER - 1]
    print(
        f"\nSelected by MOUSE: Mouse {TARGET_MOUSE}, "
        f"recording {RECORDING_NUMBER} of {len(mouse_rows)}"
    )

problem_file = selected["differential_file"]
problem_path = selected[path_column]

print(f"File: {problem_file}")
print(f"Path: {problem_path}")
print(f"Mouse: {selected['mouse']} | Group: {selected['group']}")
print(f"days_on_diet = {selected.get('days_on_diet', '')}")
print(f"body_weight  = {selected.get('body_weight', '')}")


# ============================================================
# 5. LOAD SIGNAL (.npy float32)
# ============================================================

if not os.path.exists(problem_path):
    raise FileNotFoundError(f"File not found:\n{problem_path}")

signal = np.load(problem_path)

duration_sec = len(signal) / FS
duration_min = duration_sec / 60

print(f"\nDuration: {duration_sec:.1f} s ({duration_min:.2f} min)")
print(
    f"Signal: min={signal.min():.2f}  max={signal.max():.2f}  "
    f"mean={signal.mean():.2f}  std={signal.std():.2f}"
)


# ============================================================
# 6. PLOT WINDOW HELPER
# ============================================================

def plot_window(start_sec, duration_sec_window, title_suffix, output_name):
    start_idx = int(start_sec * FS)
    end_idx = int((start_sec + duration_sec_window) * FS)
    if end_idx > len(signal):
        end_idx = len(signal)

    window = signal[start_idx:end_idx]
    time = np.arange(len(window)) / FS + start_sec

    plt.figure(figsize=(14, 4))
    plt.plot(time, window, linewidth=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"{problem_file}\n{title_suffix}")
    plt.tight_layout()

    fig_path = os.path.join(CHECK_OUTPUT_DIR, output_name)
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"\nSaved: {fig_path}")


# ============================================================
# 7. PLOT 3 WINDOWS (start, middle, end)
# ============================================================

WINDOW_SEC = 30

plot_window(
    start_sec=0,
    duration_sec_window=WINDOW_SEC,
    title_suffix="First 30 seconds",
    output_name=f"04c_Mouse{selected['mouse']}_problem_first_30s.png"
)

middle_start = max(0, duration_sec / 2 - WINDOW_SEC / 2)
plot_window(
    start_sec=middle_start,
    duration_sec_window=WINDOW_SEC,
    title_suffix="Middle 30 seconds",
    output_name=f"04c_Mouse{selected['mouse']}_problem_middle_30s.png"
)

last_start = max(0, duration_sec - WINDOW_SEC)
plot_window(
    start_sec=last_start,
    duration_sec_window=WINDOW_SEC,
    title_suffix="Last 30 seconds",
    output_name=f"04c_Mouse{selected['mouse']}_problem_last_30s.png"
)


# ============================================================
# 8. FULL RECORDING
# ============================================================

time_full = np.arange(len(signal)) / FS

plt.figure(figsize=(16, 4))
plt.plot(time_full, signal, linewidth=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Full recording\n{problem_file}")
plt.tight_layout()

full_fig_path = os.path.join(
    CHECK_OUTPUT_DIR,
    f"04c_Mouse{selected['mouse']}_problem_full_recording.png"
)
plt.savefig(full_fig_path, dpi=300)
plt.show()
print(f"\nSaved: {full_fig_path}")


print("\nSTEP 04C (filtered) finished successfully.")