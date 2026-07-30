# ============================================================
# 01_LOAD_RECORDINGS.PY
# Purpose:
# Find all .eeg recordings for selected mice and create a dataset table.
# ============================================================

import os
import glob
import re
import pandas as pd


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

CABLE = "Cable1"   # Later change to "Cable3"

BASE_DIR = "/Users/amanlizahra/Desktop"

DATA_FOLDERS = {
    "CTRL": os.path.join(BASE_DIR, f"{CABLE}_CTRL"),
    "HF": os.path.join(BASE_DIR, f"{CABLE}_HF"),
}


# ============================================================
# 2. SELECTED MICE FOR PILOT ANALYSIS
# ============================================================

HF_MICE = [1, 4, 9, 10, 14, 17, 21, 22, 23, 25]   # Later change to full dataset
CTRL_MICE = [2, 5, 7, 8, 13, 18, 19, 24, 26]       # Later change to full dataset

SELECTED_MICE = {
    "CTRL": CTRL_MICE,
    "HF": HF_MICE,
}


# ============================================================
# 3. FUNCTION TO EXTRACT MOUSE ID FROM FILE NAME
# ============================================================

def extract_mouse_id(filename):
    """
    Example:
    Mouse21_Cable1-2026-04-10_14-08-00.eeg
    returns 21
    """
    match = re.search(r"Mouse(\d+)", filename)

    if match:
        return int(match.group(1))
    else:
        return None


# ============================================================
# 4. FIND RECORDINGS
# ============================================================

recording_rows = []

for group, folder_path in DATA_FOLDERS.items():

    eeg_files = glob.glob(os.path.join(folder_path, "*.eeg"))

    for file_path in eeg_files:

        filename = os.path.basename(file_path)
        mouse_id = extract_mouse_id(filename)

        if mouse_id in SELECTED_MICE[group]:

            recording_rows.append({
                "mouse": mouse_id,
                "group": group,
                "cable": CABLE,
                "filename": filename,
                "file_path": file_path
            })


recordings_df = pd.DataFrame(recording_rows)

if len(recordings_df) > 0:
    recordings_df = recordings_df.sort_values(
        by=["group", "mouse", "filename"]
    ).reset_index(drop=True)


# ============================================================
# 5. PRINT RESULTS
# ============================================================

print("\n========================================")
print("DATASET INVENTORY")
print("========================================")

print("\nCable:")
print(CABLE)

print("\nSelected CTRL mice:")
print(CTRL_MICE)

print("\nSelected HF mice:")
print(HF_MICE)

print("\nFound recordings:")
print(recordings_df)

print("\nNumber of recordings per group:")
print(recordings_df.groupby("group").size())

print("\nNumber of recordings per mouse:")
print(recordings_df.groupby(["group", "mouse"]).size())


# ============================================================
# 6. CHECK MISSING MICE
# ============================================================

print("\nMissing mice check:")

for group, mice_list in SELECTED_MICE.items():

    found_mice = recordings_df.loc[
        recordings_df["group"] == group, "mouse"
    ].unique()

    missing_mice = sorted(list(set(mice_list) - set(found_mice)))

    print(f"{group} missing mice:", missing_mice)


# ============================================================
# 7. SAVE OUTPUT TABLE
# ============================================================

output_path = os.path.join(
    BASE_DIR,
    "thesis_lfp_analysis",
    "outputs",
    f"01_recordings_inventory_{CABLE}.csv"
)

# Make sure the output folder exists before saving
# (prevents FileNotFoundError on the first run).
os.makedirs(os.path.dirname(output_path), exist_ok=True)

recordings_df.to_csv(output_path, index=False)

print("\nSaved inventory table to:")
print(output_path)

print("\nSTEP 1 finished successfully.")