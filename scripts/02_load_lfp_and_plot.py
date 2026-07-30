import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

FS = 1250
N_CHANNELS = 4
PLOT_SECONDS = 30

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
INVENTORY_PATH = os.path.join(BASE_DIR, "outputs", "01_recordings_inventory_Cable1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# WHICH RECORDING TO PLOT
# ------------------------------------------------------------
# You can choose a recording in 3 ways. The script checks them
# in this order and uses the FIRST one that is set (not None).
#
# 1) By file name (most exact):
#    Write any part of the file name. For example a date.
#    Example: TARGET_FILENAME = "2026-03-31"
#    Example: TARGET_FILENAME = "Mouse23_Cable1-2026-04-10"
#
# 2) By mouse number:
#    TARGET_MOUSE = 23  -> picks recordings of mouse 23.
#    RECORDING_NUMBER = 1 -> the 1st recording of that mouse
#                            (2 = second, 3 = third, ...).
#
# 3) By table index (the old way, as a fallback):
#    TARGET_INDEX = 0 -> the first row in the inventory table.
#
# To use a method, set it; to ignore it, leave it as None.
# ============================================================

# --- Method 1: by file name (set to None to ignore) ---
TARGET_FILENAME = None        # e.g. "2026-03-31" or "Mouse23"

# --- Method 2: by mouse number (set TARGET_MOUSE to None to ignore) ---
TARGET_MOUSE = 5              # e.g. 23
RECORDING_NUMBER = 1          # 1 = first recording of that mouse

# --- Method 3: by table index (used only if the two above are None) ---
TARGET_INDEX = 0


# ============================================================
# LOAD INVENTORY
# ============================================================

recordings_df = pd.read_csv(INVENTORY_PATH)

print("Loaded inventory:")
print(recordings_df.head())

print("\nTotal recordings in inventory:")
print(len(recordings_df))


# ============================================================
# SELECT THE RECORDING
# ============================================================

def select_recording(df):
    """
    Pick one recording based on the settings above.
    Returns a single row (pandas Series).
    """

    # ---- Method 1: by file name ----
    if TARGET_FILENAME is not None:
        matches = df[df["filename"].str.contains(TARGET_FILENAME, na=False)]

        if len(matches) == 0:
            raise ValueError(
                f"No recording found containing '{TARGET_FILENAME}' in its name."
            )

        if len(matches) > 1:
            print(f"\nNote: {len(matches)} recordings match '{TARGET_FILENAME}'.")
            print("Using the first match. Matching files:")
            for name in matches["filename"].tolist():
                print("   ", name)

        print(f"\nSelected by FILE NAME containing: '{TARGET_FILENAME}'")
        return matches.iloc[0]

    # ---- Method 2: by mouse number ----
    if TARGET_MOUSE is not None:
        mouse_rows = df[df["mouse"] == TARGET_MOUSE].reset_index(drop=True)

        if len(mouse_rows) == 0:
            raise ValueError(f"No recordings found for Mouse {TARGET_MOUSE}.")

        if RECORDING_NUMBER < 1 or RECORDING_NUMBER > len(mouse_rows):
            raise ValueError(
                f"Mouse {TARGET_MOUSE} has {len(mouse_rows)} recordings, "
                f"but RECORDING_NUMBER = {RECORDING_NUMBER} "
                f"(valid range 1..{len(mouse_rows)})."
            )

        print(
            f"\nSelected by MOUSE: Mouse {TARGET_MOUSE}, "
            f"recording number {RECORDING_NUMBER} of {len(mouse_rows)}"
        )
        # RECORDING_NUMBER is 1-based, so subtract 1
        return mouse_rows.iloc[RECORDING_NUMBER - 1]

    # ---- Method 3: by table index ----
    if TARGET_INDEX >= len(df):
        raise IndexError(
            f"TARGET_INDEX = {TARGET_INDEX} but there are only "
            f"{len(df)} recordings (valid range 0..{len(df) - 1})."
        )

    print(f"\nSelected by TABLE INDEX: {TARGET_INDEX}")
    return df.iloc[TARGET_INDEX]


selected = select_recording(recordings_df)

file_path = selected["file_path"]
filename = selected["filename"]

print("\nSelected file:")
print(filename)
print("Mouse:", selected["mouse"], "| Group:", selected["group"])


# ============================================================
# LOAD LFP FILE
# Raw recordings are stored as int16 -> read as int16.
# ============================================================

raw = np.fromfile(file_path, dtype=np.int16)

if len(raw) % N_CHANNELS != 0:
    raw = raw[:len(raw) - (len(raw) % N_CHANNELS)]

data = raw.reshape(-1, N_CHANNELS)

duration_sec = data.shape[0] / FS

print("\nData shape:", data.shape)
print("Duration seconds:", duration_sec)
print("Duration minutes:", duration_sec / 60)


# ============================================================
# CREATE TIME AXIS
# ============================================================

n_samples_plot = int(PLOT_SECONDS * FS)

# If the recording is shorter than PLOT_SECONDS, plot what we have.
if n_samples_plot > data.shape[0]:
    n_samples_plot = data.shape[0]

plot_data = data[:n_samples_plot, :]
time = np.arange(n_samples_plot) / FS


# ============================================================
# PLOT IN SPIKE2 ORDER
# Top: Ch4, Ch3, Ch2, Bottom: Ch1
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)

channel_order = [
    (3, "Ch4"),
    (2, "Ch3"),
    (1, "Ch2"),
    (0, "Ch1"),
]

for ax, (col_idx, ch_name) in zip(axes, channel_order):
    ax.plot(time, plot_data[:, col_idx], linewidth=0.6)
    ax.set_ylabel(ch_name)

axes[-1].set_xlabel("Time (s)")

fig.suptitle(f"Raw LFP channels - Spike2 order\n{filename}", fontsize=12)
plt.tight_layout()

# Build a clean output name from the selected file
safe_name = filename.replace(".eeg", "").replace(" ", "_")
output_path = os.path.join(
    OUTPUT_DIR,
    f"02_raw_channels_spike2_order_{safe_name}.png"
)

plt.savefig(output_path, dpi=300)
plt.show()

print("\nSaved figure to:")
print(output_path)

print("\nSTEP 2 finished successfully.")