# ============================================================
# 04B_FIND_BAD_RECORDINGS.PY  (FILTERED VERSION)
# Purpose:
# Automatically flag bad / suspicious differential recordings
# based on signal-quality metrics.
#
# Reads .npy differential files (float32) from 03a (filtered).
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

FS = 1250
CABLE = "Cable1"

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DIFF_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    f"03a_differential_summary_filtered_{CABLE}.csv"
)

BAD_CHECK_DIR = os.path.join(
    OUTPUT_DIR,
    f"04b_bad_recording_check_filtered_{CABLE}"
)
os.makedirs(BAD_CHECK_DIR, exist_ok=True)

QUALITY_OUTPUT_PATH = os.path.join(
    BAD_CHECK_DIR,
    f"04b_recording_quality_metrics_{CABLE}.csv"
)
BAD_OUTPUT_PATH = os.path.join(
    BAD_CHECK_DIR,
    f"04b_bad_recording_candidates_{CABLE}.csv"
)


# ============================================================
# 2. LOAD DIFFERENTIAL SUMMARY
# ============================================================

if not os.path.exists(DIFF_SUMMARY_PATH):
    raise FileNotFoundError(
        f"Differential summary not found:\n{DIFF_SUMMARY_PATH}"
    )

summary_df = pd.read_csv(DIFF_SUMMARY_PATH)

print(f"\nLoaded differential summary:")
print(f"Total recordings: {len(summary_df)}")


# Pick path column
if "diff_npy_path" in summary_df.columns:
    path_column = "diff_npy_path"
elif "diff_eeg_path" in summary_df.columns:
    path_column = "diff_eeg_path"
else:
    raise ValueError("Summary missing diff_npy_path / diff_eeg_path.")


# ============================================================
# 3. COMPUTE QUALITY METRICS
# ============================================================

quality_rows = []

for _, row in summary_df.iterrows():
    mouse = int(row["mouse"])
    group = row["group"]
    cable = row["cable"]
    diff_file = row["differential_file"]
    diff_path = row[path_column]

    if not os.path.exists(diff_path):
        print(f"File not found, skipping: {diff_path}")
        continue

    try:
        signal = np.load(diff_path).astype(np.float64)
    except Exception as e:
        print(f"Failed to load {diff_file}: {e}")
        continue

    if len(signal) == 0:
        print(f"Empty file, skipping: {diff_file}")
        continue

    sig_min = float(signal.min())
    sig_max = float(signal.max())
    sig_mean = float(signal.mean())
    sig_std = float(signal.std())
    sig_rms = float(np.sqrt(np.mean(signal ** 2)))
    sig_ptp = float(sig_max - sig_min)

    duration_sec = len(signal) / FS
    duration_min = duration_sec / 60

    quality_rows.append({
        "mouse": mouse,
        "group": group,
        "cable": cable,
        "differential_file": diff_file,
        "days_on_diet": row.get("days_on_diet", ""),
        "body_weight": row.get("body_weight", ""),
        "duration_min": duration_min,
        "min": sig_min,
        "max": sig_max,
        "mean": sig_mean,
        "std": sig_std,
        "rms": sig_rms,
        "peak_to_peak": sig_ptp,
        "diff_path": diff_path,
    })


quality_df = pd.DataFrame(quality_rows)

if len(quality_df) == 0:
    raise ValueError("No quality metrics were created.")


# ============================================================
# 4. ROBUST THRESHOLDS (median + multiplier * MAD)
# ============================================================

def robust_threshold(values, multiplier=6):
    values = np.asarray(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return median
    return median + multiplier * mad


std_threshold = robust_threshold(quality_df["std"], multiplier=6)
rms_threshold = robust_threshold(quality_df["rms"], multiplier=6)
ptp_threshold = robust_threshold(quality_df["peak_to_peak"], multiplier=6)

print("\nThresholds (median + 6 * MAD):")
print(f"  STD threshold:          {std_threshold:.2f}")
print(f"  RMS threshold:          {rms_threshold:.2f}")
print(f"  Peak-to-peak threshold: {ptp_threshold:.2f}")


# ============================================================
# 5. FLAG BAD CANDIDATES
# ============================================================

quality_df["bad_std"] = quality_df["std"] > std_threshold
quality_df["bad_rms"] = quality_df["rms"] > rms_threshold
quality_df["bad_peak_to_peak"] = quality_df["peak_to_peak"] > ptp_threshold

quality_df["bad_candidate"] = (
    quality_df["bad_std"]
    | quality_df["bad_rms"]
    | quality_df["bad_peak_to_peak"]
)

bad_df = quality_df[quality_df["bad_candidate"] == True].copy()
bad_df = bad_df.sort_values(
    by=["std", "peak_to_peak"], ascending=False
).reset_index(drop=True)


# ============================================================
# 6. SAVE
# ============================================================

quality_df.to_csv(QUALITY_OUTPUT_PATH, index=False)
bad_df.to_csv(BAD_OUTPUT_PATH, index=False)


# ============================================================
# 7. PRINT
# ============================================================

print(f"\nSaved quality metrics to: {QUALITY_OUTPUT_PATH}")
print(f"Saved bad candidates to:  {BAD_OUTPUT_PATH}")

print(f"\nTotal recordings checked: {len(quality_df)}")
print(f"Number of bad candidates: {len(bad_df)}")

if len(bad_df) > 0:
    print("\nBad candidates per group:")
    print(bad_df.groupby("group").size())

    print("\nBad candidates per mouse:")
    print(bad_df.groupby(["group", "mouse"]).size())

    print("\nTop bad candidates:")
    print(bad_df[[
        "mouse", "group", "differential_file",
        "days_on_diet", "body_weight",
        "std", "rms", "peak_to_peak"
    ]].head(15).to_string(index=False))
else:
    print("\nNo bad candidates found.")


print("\nSTEP 04B (filtered) finished successfully.")


# ============================================================
# 8. CONFIRMED BAD RECORDINGS  (manual, visually verified)
# ------------------------------------------------------------
# Section 3-7 above only PROPOSE candidates from quality metrics.
# Arecording is excluded only after looking at the
# raw signal (use 04c to inspect each candidate). List here the
# differential_file names you have visually confirmed as bad
# (true artifacts / not analysable LFP). THIS file is what the
# downstream steps (04e outlier test and feature extraction)
# actually exclude, so the automatic thresholds never drop a
# recording on their own.
# ============================================================

CONFIRMED_BAD = [
    # "Mouse23_Cable1_HF_2026-03-26_13-09-11_DIFF_Ch2_minus_Ch3.npy",
    # add visually-confirmed bad recordings here, one per line
]

CONFIRMED_BAD_PATH = os.path.join(
    BAD_CHECK_DIR,
    f"04b_confirmed_bad_recordings_{CABLE}.csv"
)

confirmed_df = quality_df[
    quality_df["differential_file"].isin(CONFIRMED_BAD)
].copy()

# Keep a stable, minimal schema even when the list is empty,
# so downstream code can always read this file.
if len(confirmed_df) == 0:
    confirmed_df = pd.DataFrame(columns=[
        "mouse", "group", "differential_file",
        "std", "rms", "peak_to_peak"
    ])

confirmed_df.to_csv(CONFIRMED_BAD_PATH, index=False)

print("\n" + "=" * 60)
print("CONFIRMED BAD RECORDINGS (manual list)")
print("=" * 60)
print(f"  Listed: {len(CONFIRMED_BAD)}")
print(f"  Matched in this cable: {len(confirmed_df)}")
print(f"  Saved to: {CONFIRMED_BAD_PATH}")
if len(CONFIRMED_BAD) > 0 and len(confirmed_df) < len(CONFIRMED_BAD):
    missing = set(CONFIRMED_BAD) - set(confirmed_df["differential_file"])
    print(f"  WARNING: {len(missing)} listed name(s) not found in summary:")
    for m in missing:
        print(f"    - {m}")