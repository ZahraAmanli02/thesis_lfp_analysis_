# ============================================================
# 04B2_BATCH_INSPECT_CANDIDATES.PY  
# Purpose:
# Batch visual + objective check of ALL bad candidates found by
# 04b, so you do not have to open each recording one by one.
#
# For every candidate in
#   04b_bad_recording_check_filtered_<CABLE>/
#       04b_bad_recording_candidates_<CABLE>.csv
# this script:
#   1) loads the raw differential signal (.npy from 03a),
#   2) saves ONE full-recording trace with the saturation limits
#      drawn on top (quick "is it clipped?" view),
#   3) computes an objective saturation fraction:
#        fraction of samples at (or extremely close to) the int16
#        rail, i.e. |x| >= SAT_LIMIT * SAT_TOLERANCE.
#
# The point is to make the manual decision easy and reproducible:
# you look at the panel + the saturation_fraction number, then
# list the truly-bad recordings in the CONFIRMED_BAD block of 04b.
# This script NEVER excludes anything on its own.
#
# Input:
#   04b_bad_recording_check_filtered_<CABLE>/
#       04b_bad_recording_candidates_<CABLE>.csv
#
# Output:
#   04b2_candidate_inspection_<CABLE>/
#     - one PNG per candidate (full trace + saturation lines)
#     - 04b2_candidate_saturation_table_<CABLE>.csv
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

# Input: the candidates table written by 04b
CANDIDATES_PATH = os.path.join(
    OUTPUT_DIR,
    f"04b_bad_recording_check_filtered_{CABLE}",
    f"04b_bad_recording_candidates_{CABLE}.csv"
)

# Output folder for this batch inspection
OUT_DIR = os.path.join(
    OUTPUT_DIR,
    f"04b2_candidate_inspection_{CABLE}"
)
os.makedirs(OUT_DIR, exist_ok=True)

SAT_TABLE_PATH = os.path.join(
    OUT_DIR,
    f"04b2_candidate_saturation_table_{CABLE}.csv"
)

# int16 saturation rail. The differential files are int16-derived,
# so a clipped sample sits at +/-65535 (here treated symmetrically).
SAT_LIMIT = 65535.0
# A sample counts as "at the rail" if its magnitude is within this
# fraction of the rail (covers 65048, 64983, etc. seen in the data).
SAT_TOLERANCE = 0.98


# ============================================================
# 2. LOAD CANDIDATES
# ============================================================

if not os.path.exists(CANDIDATES_PATH):
    raise FileNotFoundError(
        f"Candidates file not found:\n{CANDIDATES_PATH}\n"
        f"Run 04b first."
    )

cand = pd.read_csv(CANDIDATES_PATH)

print(f"\nLoaded candidates: {CANDIDATES_PATH}")
print(f"  {len(cand)} candidate recording(s)")

if len(cand) == 0:
    print("No candidates to inspect. Nothing to do.")
    raise SystemExit(0)

# Pick the path column written by 04b
if "diff_path" in cand.columns:
    path_column = "diff_path"
elif "diff_npy_path" in cand.columns:
    path_column = "diff_npy_path"
else:
    raise ValueError("Candidates table has no diff_path / diff_npy_path column.")


# ============================================================
# 3. INSPECT EACH CANDIDATE
# ============================================================

sat_limit_low = SAT_LIMIT * SAT_TOLERANCE

rows = []

for i, row in cand.reset_index(drop=True).iterrows():

    diff_file = row["differential_file"]
    diff_path = row[path_column]
    mouse = row.get("mouse", "")
    group = row.get("group", "")

    print(f"\n[{i + 1}/{len(cand)}] {diff_file}")

    if not os.path.exists(diff_path):
        print(f"  File not found, skipping: {diff_path}")
        rows.append({
            "mouse": mouse, "group": group,
            "differential_file": diff_file,
            "n_samples": np.nan,
            "saturation_fraction": np.nan,
            "max_abs": np.nan,
            "status": "file_not_found",
        })
        continue

    try:
        signal = np.load(diff_path).astype(np.float64)
    except Exception as e:
        print(f"  Load error: {e}")
        rows.append({
            "mouse": mouse, "group": group,
            "differential_file": diff_file,
            "n_samples": np.nan,
            "saturation_fraction": np.nan,
            "max_abs": np.nan,
            "status": f"load_error: {e}",
        })
        continue

    n = len(signal)
    max_abs = float(np.max(np.abs(signal)))

    # Objective saturation fraction: how much of the recording sits
    # at (or extremely close to) the int16 rail.
    n_sat = int(np.sum(np.abs(signal) >= sat_limit_low))
    sat_fraction = n_sat / n if n > 0 else np.nan

    print(f"  n={n}  max|x|={max_abs:.0f}  "
          f"saturated={100 * sat_fraction:.2f}%")

    rows.append({
        "mouse": mouse, "group": group,
        "differential_file": diff_file,
        "n_samples": n,
        "saturation_fraction": round(sat_fraction, 5),
        "max_abs": max_abs,
        "status": "ok",
    })

    # ----- One full-recording figure with saturation lines -----
    t = np.arange(n) / FS

    plt.figure(figsize=(16, 4))
    plt.plot(t, signal, linewidth=0.4)
    plt.axhline(SAT_LIMIT, color="red", linestyle="--",
                linewidth=0.8, alpha=0.7, label="int16 rail (+/-65535)")
    plt.axhline(-SAT_LIMIT, color="red", linestyle="--",
                linewidth=0.8, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(
        f"{diff_file}\n"
        f"Mouse {mouse} ({group}) | "
        f"saturated {100 * sat_fraction:.2f}% of samples"
    )
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    safe_name = diff_file.replace(".npy", "").replace(".eeg", "")
    fig_path = os.path.join(OUT_DIR, f"04b2_{safe_name}_full.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()


# ============================================================
# 4. SAVE SATURATION TABLE
# ============================================================

sat_df = pd.DataFrame(rows)
sat_df = sat_df.sort_values(
    by="saturation_fraction", ascending=False, na_position="last"
).reset_index(drop=True)

sat_df.to_csv(SAT_TABLE_PATH, index=False)

 
# ============================================================
# 5. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE SATURATION SUMMARY")
print("=" * 70)
print(f"\nSaved per-candidate traces to: {OUT_DIR}")
print(f"Saved saturation table to:     {SAT_TABLE_PATH}\n")

show_cols = ["mouse", "group", "differential_file",
             "saturation_fraction", "max_abs", "status"]
show_cols = [c for c in show_cols if c in sat_df.columns]
print(sat_df[show_cols].to_string(index=False))

print(
    "\nNext step:\n"
    "  Look at the PNGs + the saturation_fraction column, then copy the\n"
    "  truly-bad differential_file names into the CONFIRMED_BAD list in\n"
    "  04b_find_bad_recordings.py and re-run 04b. This script decides\n"
    "  nothing on its own."
)

print("\nSTEP 04B2 (batch candidate inspection) finished successfully.")