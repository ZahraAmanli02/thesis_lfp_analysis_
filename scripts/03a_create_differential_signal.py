# ============================================================
# 03A_CREATE_DIFFERENTIAL_SIGNAL.PY  (FILTERED VERSION)
# Purpose:
# Create differential LFP signals from the FILTERED inventory
# (output of step 01c)
#
# Two files per recording, saved in SEPARATE folders:
#   1) .npy  -> float32, TRUE values (no clipping)
#              Used by the whole analysis pipeline.
#   2) .eeg  -> int16, SCALED. For Spike2 viewing only.
#
# The output summary CSV also carries the per-recording metadata
# (days_on_diet, body_weight, ...) so later steps can use it.
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

FS = 1250
N_CHANNELS = 4
CABLE = "Cable1"

INT16_SAFE_LIMIT = 32000

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Raw data folders (used as fallback if the inventory path is outdated)
RAW_DATA_BASE = "/Users/amanlizahra/Desktop"
RAW_DATA_FOLDERS = {
    "CTRL": os.path.join(RAW_DATA_BASE, f"{CABLE}_CTRL"),
    "HF":   os.path.join(RAW_DATA_BASE, f"{CABLE}_HF"),
}

# INPUT: filtered inventory from step 01c
INVENTORY_PATH = os.path.join(
    OUTPUT_DIR,
    f"01c_recordings_filtered_{CABLE}.csv"
)

# OUTPUT folders (new "_filtered" suffix so we don't overwrite old results)
NPY_DIR = os.path.join(
    OUTPUT_DIR,
    f"03a_differential_npy_filtered_{CABLE}"
)
EEG_DIR = os.path.join(
    OUTPUT_DIR,
    f"03a_differential_eeg_for_spike2_filtered_{CABLE}"
)

os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(EEG_DIR, exist_ok=True)


# ============================================================
# 2. BEST CHANNEL PAIRS
# ============================================================

BEST_CHANNEL_PAIRS = {
    # HF mice
    1:  (2, 3),
    4:  (2, 3),
    9:  (3, 4),
    10: (3, 4),  
    14: (2, 3),
    17: (2, 3),
    21: (2, 3),
    22: (1, 4),
    23: (2, 3),  
    25: (2, 3),

    # CTRL mice
    2:  (3, 4),
    5:  (2, 3),
    7:  (3, 4),
    8:  (3, 4),
    13: (2, 3),
    18: (3, 4),
    19: (3, 4),
    24: (2, 3),
    26: (3, 4)
}


# ============================================================
# 3. FUNCTIONS
# ============================================================

def load_eeg_file(file_path, n_channels=4):
    raw = np.fromfile(file_path, dtype=np.int16)
    if len(raw) % n_channels != 0:
        print("  Warning: file length not divisible by n_channels.")
        raw = raw[:len(raw) - (len(raw) % n_channels)]
    return raw.reshape(-1, n_channels)


def channel_to_column(ch):
    return ch - 1


def find_file_robustly(filename, file_path, group):
    """Find the file even if the inventory path is outdated."""
    if os.path.exists(file_path):
        return file_path

    folder = RAW_DATA_FOLDERS.get(group)
    if folder is None or not os.path.isdir(folder):
        return None

    candidate = os.path.join(folder, filename)
    if os.path.exists(candidate):
        return candidate

    def normalize(name):
        return name.replace("-", "_").lower()

    target_norm = normalize(filename)
    for f in os.listdir(folder):
        if f.endswith(".eeg") and normalize(f) == target_norm:
            return os.path.join(folder, f)
    return None


def create_output_base_name(filename, mouse_id, group, cable, ch1, ch2):
    """Clean base name (no extension) for differential files."""
    base_name = filename.replace(".eeg", "")
    pattern = rf"^Mouse{mouse_id}[_\-]{cable}[_\-]"
    new_prefix = f"Mouse{mouse_id}_{cable}_{group}_"

    if re.match(pattern, base_name):
        base_name = re.sub(pattern, new_prefix, base_name, count=1)
    else:
        base_name = f"Mouse{mouse_id}_{cable}_{group}_{base_name}"

    return f"{base_name}_DIFF_Ch{ch1}_minus_Ch{ch2}"


def scale_to_int16(differential):
    max_abs = np.max(np.abs(differential))
    if max_abs > INT16_SAFE_LIMIT:
        scale = INT16_SAFE_LIMIT / max_abs
    else:
        scale = 1.0
    scaled = np.round(differential * scale).astype(np.int16)
    return scaled, scale


# ============================================================
# 4. LOAD FILTERED INVENTORY
# ============================================================

if not os.path.exists(INVENTORY_PATH):
    raise FileNotFoundError(
        f"Filtered inventory not found:\n{INVENTORY_PATH}\n"
        f"Run step 01c first."
    )

recordings_df = pd.read_csv(INVENTORY_PATH)

print(f"\nLoaded FILTERED inventory: {INVENTORY_PATH}")
print(f"Total recordings (after filtering): {len(recordings_df)}")
print(recordings_df.head())


# ============================================================
# 5. CREATE DIFFERENTIAL FILES
# ============================================================

summary_rows = []
skipped_rows = []

for idx, row in recordings_df.iterrows():

    mouse_id = int(row["mouse"])
    group = row["group"]
    filename = row["filename"]
    file_path = row["file_path"]

    if mouse_id not in BEST_CHANNEL_PAIRS:
        print(f"Skipping Mouse {mouse_id}: no channel pair defined.")
        continue

    resolved_path = find_file_robustly(filename, file_path, group)
    if resolved_path is None:
        print(f"Skipping (file not found): {filename}")
        skipped_rows.append({"filename": filename, "reason": "file not found"})
        continue

    ch1, ch2 = BEST_CHANNEL_PAIRS[mouse_id]
    col1 = channel_to_column(ch1)
    col2 = channel_to_column(ch2)

    try:
        data = load_eeg_file(resolved_path, N_CHANNELS)
    except Exception as e:
        print(f"Skipping (error reading): {filename} -> {e}")
        skipped_rows.append({"filename": filename, "reason": str(e)})
        continue

    differential = (
        data[:, col1].astype(np.float64) - data[:, col2].astype(np.float64)
    )
    differential_float32 = differential.astype(np.float32)

    base_name = create_output_base_name(
        filename=filename, mouse_id=mouse_id, group=group,
        cable=CABLE, ch1=ch1, ch2=ch2
    )

    # Save .npy (float32) for analysis
    npy_filename = f"{base_name}.npy"
    npy_path = os.path.join(NPY_DIR, npy_filename)
    np.save(npy_path, differential_float32)

    # Save .eeg (scaled int16) for Spike2
    scaled_int16, scale_factor = scale_to_int16(differential)
    eeg_filename = f"{base_name}.eeg"
    eeg_path = os.path.join(EEG_DIR, eeg_filename)
    scaled_int16.tofile(eeg_path)

    was_scaled = scale_factor != 1.0
    duration_sec = len(differential_float32) / FS

    # Build the summary row -> carry the metadata from 01c forward
    summary_rows.append({
        # core IDs
        "mouse": mouse_id,
        "group": group,
        "cable": CABLE,
        "original_file": filename,
        "differential_base_name": base_name,

        # files
        "differential_file": npy_filename,
        "diff_npy_path": npy_path,
        "spike2_eeg_file": eeg_filename,
        "spike2_eeg_path": eeg_path,
        "spike2_scale_factor": scale_factor,
        "spike2_was_scaled": was_scaled,

        # channels
        "channel_1": ch1,
        "channel_2": ch2,
        "differential": f"Ch{ch1}-Ch{ch2}",

        # duration
        "duration_sec": duration_sec,
        "duration_min": duration_sec / 60,

        # paths
        "original_path": resolved_path,
        "diff_eeg_path": npy_path,   # backward-compat

        # ----- metadata from 01c (recording level) -----
        "recording_date": row.get("recording_date", ""),
        "diet_phase":     row.get("diet_phase", ""),
        "days_on_diet":   row.get("days_on_diet", np.nan),
        "days_in_recovery": row.get("days_in_recovery", np.nan),
        "body_weight":    row.get("body_weight", np.nan),
        "body_weight_source": row.get("body_weight_source", ""),
        "is_recovery":    row.get("is_recovery", False),
        "estrous_phase":  row.get("estrous_phase", ""),
    })

    note = f" (scaled x{scale_factor:.4f})" if was_scaled else ""
    print(f"Saved | Mouse {mouse_id} | {group} | {base_name}{note}")


# ============================================================
# 6. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(
    OUTPUT_DIR,
    f"03a_differential_summary_filtered_{CABLE}.csv"
)
summary_df.to_csv(summary_path, index=False)

print(f"\nSaved differential summary:\n{summary_path}")


# ============================================================
# 7. PRINT BASIC SUMMARY
# ============================================================

if len(summary_df) > 0:
    print("\nDifferentials per group:")
    print(summary_df.groupby("group").size())

    print("\nDifferentials per mouse:")
    print(summary_df.groupby(["group", "mouse"]).size())

    n_scaled = int(summary_df["spike2_was_scaled"].sum())
    print(f"\nRecordings whose Spike2 .eeg was scaled: {n_scaled}")

if len(skipped_rows) > 0:
    print(f"\n⚠️  {len(skipped_rows)} recordings were skipped:")
    for s in skipped_rows:
        print(f"   - {s['filename']}: {s['reason']}")
else:
    print("\n No recordings were skipped.")


# ============================================================
# 8. PLOT ONE EXAMPLE DIFFERENTIAL
# ============================================================

if len(summary_df) > 0:
    example = summary_df.iloc[0]

    diff_signal = np.load(example["diff_npy_path"])

    plot_seconds = 30
    n_samples = int(plot_seconds * FS)
    if len(diff_signal) < n_samples:
        n_samples = len(diff_signal)

    time = np.arange(n_samples) / FS

    plt.figure(figsize=(14, 4))
    plt.plot(time, diff_signal[:n_samples], linewidth=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(
        f"Example Differential LFP (from .npy)\n"
        f"Mouse {example['mouse']} | {example['group']} | {example['differential']}"
    )
    plt.tight_layout()

    fig_path = os.path.join(
        OUTPUT_DIR,
        f"03a_example_differential_filtered_{CABLE}.png"
    )
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"\nSaved example figure:\n{fig_path}")


print("\nSTEP 3A (filtered) finished successfully.")