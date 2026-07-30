# ============================================================
# 03C_VALIDATE_DIFFERENTIAL_SIGNAL.PY  (FILTERED VERSION)
# Purpose:
# Visually validate one recording from the FILTERED inventory:
# show channel A, channel B, and (A - B) on the same y-axis.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

FS = 1250
N_CHANNELS = 4
CABLE = "Cable1"

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Use the FILTERED inventory from 01c
INVENTORY_PATH = os.path.join(
    OUTPUT_DIR,
    f"01c_recordings_filtered_{CABLE}.csv"
)

VALIDATE_DIR = os.path.join(
    OUTPUT_DIR,
    f"03c_validate_differential_filtered_{CABLE}"
)
os.makedirs(VALIDATE_DIR, exist_ok=True)


# ============================================================
# 2. SELECT RECORDING TO VALIDATE
# Same selection style as before — by filename, mouse number, or index.
# ============================================================

TARGET_FILENAME = None        # e.g. "2026-03-31"
TARGET_MOUSE = 1
RECORDING_NUMBER = 1
TARGET_INDEX = 0

CHANNEL_1 = 2
CHANNEL_2 = 3

START_SEC = 0
DURATION_SEC = None           # None = full recording


# ============================================================
# 3. SELECT
# ============================================================

if not os.path.exists(INVENTORY_PATH):
    raise FileNotFoundError(
        f"Filtered inventory not found:\n{INVENTORY_PATH}\n"
        f"Run step 01c first."
    )

recordings_df = pd.read_csv(INVENTORY_PATH)
print("Total recordings in filtered inventory:", len(recordings_df))


def select_recording(df):
    if TARGET_FILENAME is not None:
        matches = df[df["filename"].str.contains(TARGET_FILENAME, na=False)]
        if len(matches) == 0:
            raise ValueError(f"No recording matches '{TARGET_FILENAME}'.")
        print(f"\nSelected by FILENAME: '{TARGET_FILENAME}'")
        return matches.iloc[0]

    if TARGET_MOUSE is not None:
        mouse_rows = df[df["mouse"] == TARGET_MOUSE].reset_index(drop=True)
        if len(mouse_rows) == 0:
            raise ValueError(f"No recordings for Mouse {TARGET_MOUSE}.")
        if RECORDING_NUMBER < 1 or RECORDING_NUMBER > len(mouse_rows):
            raise ValueError(
                f"Mouse {TARGET_MOUSE} has {len(mouse_rows)} recordings, "
                f"but RECORDING_NUMBER = {RECORDING_NUMBER}."
            )
        print(
            f"\nSelected by MOUSE: Mouse {TARGET_MOUSE}, "
            f"recording {RECORDING_NUMBER} of {len(mouse_rows)}"
        )
        return mouse_rows.iloc[RECORDING_NUMBER - 1]

    if TARGET_INDEX >= len(df):
        raise IndexError(
            f"TARGET_INDEX = {TARGET_INDEX} but only {len(df)} recordings."
        )
    print(f"\nSelected by INDEX: {TARGET_INDEX}")
    return df.iloc[TARGET_INDEX]


selected = select_recording(recordings_df)
file_path = selected["file_path"]
filename = selected["filename"]

print(f"Selected file: {filename}")
print(f"Mouse: {selected['mouse']} | Group: {selected['group']}")


# ============================================================
# 4. LOAD RAW
# ============================================================

raw = np.fromfile(file_path, dtype=np.int16)
raw = raw[:len(raw) - (len(raw) % N_CHANNELS)]
data = raw.reshape(-1, N_CHANNELS)

print(f"Data shape: {data.shape}")
print(f"Duration min: {data.shape[0] / FS / 60:.2f}")


# ============================================================
# 5. EXTRACT CHANNELS
# ============================================================

ch1 = data[:, CHANNEL_1 - 1]
ch2 = data[:, CHANNEL_2 - 1]
diff = ch1.astype(np.float32) - ch2.astype(np.float32)


# ============================================================
# 6. TIME WINDOW
# ============================================================

start_idx = int(START_SEC * FS)
if DURATION_SEC is None:
    end_idx = len(ch1)
else:
    end_idx = int((START_SEC + DURATION_SEC) * FS)
    if end_idx > len(ch1):
        end_idx = len(ch1)

time = np.arange(end_idx - start_idx) / FS + START_SEC

ch1_w = ch1[start_idx:end_idx]
ch2_w = ch2[start_idx:end_idx]
diff_w = diff[start_idx:end_idx]

print(f"Plotting from {START_SEC} s to {end_idx / FS:.1f} s")


# ============================================================
# 7. SHARED Y-AXIS
# ============================================================

y_min = min(ch1_w.min(), ch2_w.min(), diff_w.min())
y_max = max(ch1_w.max(), ch2_w.max(), diff_w.max())


# ============================================================
# 8. FIGURE
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

axes[0].plot(time, ch1_w, linewidth=0.5)
axes[0].set_title(f"Channel {CHANNEL_1}")
axes[0].set_ylim(y_min, y_max)

axes[1].plot(time, ch2_w, linewidth=0.5)
axes[1].set_title(f"Channel {CHANNEL_2}")
axes[1].set_ylim(y_min, y_max)

axes[2].plot(time, diff_w, linewidth=0.5)
axes[2].set_title(f"Differential (Ch{CHANNEL_1} - Ch{CHANNEL_2})")
axes[2].set_ylim(y_min, y_max)
axes[2].set_xlabel("Time (s)")

fig.suptitle(filename, fontsize=11)
plt.tight_layout()


# ============================================================
# 9. SAVE
# ============================================================

safe_name = filename.replace(".eeg", "").replace(" ", "_")
output_path = os.path.join(
    VALIDATE_DIR,
    f"03c_Ch{CHANNEL_1}_minus_Ch{CHANNEL_2}_{safe_name}.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved figure:\n{output_path}")

print("\nSTEP 3C (filtered) finished successfully.")