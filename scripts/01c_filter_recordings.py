# ============================================================
# 01C_FILTER_RECORDINGS.PY
# Purpose:
# Filter the enriched inventory from step 01b down to the recordings
# used for downstream analysis, and log exactly what was removed and
# why (full traceability — 01b stays the complete record, 01c is the
# analysis selection).
#
# Filters applied (in order; each can be toggled):
#   1) ANALYSIS WINDOW (date range). Keep only recordings whose date is
#      within [ANALYSIS_START_DATE, ANALYSIS_END_DATE] inclusive.
#      Agreed with professor: 2026-02-19 .. 2026-04-02.
#        - before 19 Feb: pre-baseline (body-weight grounds)
#        - after  2 Apr : recovery period with no swab/cycle data
#      ON by default.
#
#   2) RECOVERY period (diet_phase == "recovery"). Whether to also drop
#      the recovery recordings that fall INSIDE the window (30 Mar..2 Apr).
#      OFF by default — all data in the window is used.
#
#   3) MISSING estrous phase. Drop recordings with no A/B/C/D phase.
#      ON by default. Inside the window swab coverage is complete, so
#      this normally removes nothing; it is a safety net.
#
# Input:
#   outputs/01b_recordings_inventory_with_metadata_<CABLE>.csv
#
# Output:
#   outputs/01c_recordings_filtered_<CABLE>.csv
#   outputs/01c_recordings_excluded_<CABLE>.csv  (log of removed rows)
# ============================================================

import os
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

# Resolve the project root from THIS script's location, so the paths
# work no matter what the project folder is called. scripts/ -> root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

INPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"01b_recordings_inventory_with_metadata_{CABLE}.csv"
)

OUTPUT_KEPT_PATH = os.path.join(
    OUTPUT_DIR,
    f"01c_recordings_filtered_{CABLE}.csv"
)

OUTPUT_EXCLUDED_PATH = os.path.join(
    OUTPUT_DIR,
    f"01c_recordings_excluded_{CABLE}.csv"
)


# ============================================================
# 2. FILTER OPTIONS
# ============================================================

# ---- Filter 1: analysis window (date range) ----
# Agreed with professor: keep 2026-02-19 .. 2026-04-02 (inclusive).
FILTER_WINDOW = True
ANALYSIS_START_DATE = pd.Timestamp("2026-02-19")
ANALYSIS_END_DATE   = pd.Timestamp("2026-04-02")

# ---- Filter 2: recovery period ----
# Drop recordings with diet_phase == "recovery" (after 2026-03-30) that
# still fall inside the window. Default: False — all data in the window
# is used (recovery recordings in the window have cycle + weight data).
FILTER_RECOVERY = False

# ---- Filter 3: missing estrous phase ----
# Drop recordings without a usable A/B/C/D phase. Default: True.
FILTER_NO_ESTROUS = True


# ============================================================
# 3. LOAD INPUT
# ============================================================

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_PATH}\n"
        f"Run step 01b first."
    )

df = pd.read_csv(INPUT_PATH)

print(f"\nLoaded inventory: {INPUT_PATH}")
print(f"Total recordings: {len(df)}")

# Parse recording_date as datetime for the window filter and logs
if "recording_date" in df.columns:
    df["recording_date"] = pd.to_datetime(df["recording_date"], errors="coerce")


# ============================================================
# 4. APPLY FILTERS (track which rule excluded each row)
# ------------------------------------------------------------
# Each recording is checked against the active filters in order. The
# FIRST rule that excludes a recording is recorded as its reason, so the
# excluded log shows one clear reason per removed recording.
# ============================================================

df["excluded"] = False
df["exclusion_reason"] = ""

n_start = len(df)

# ---- Filter 1: analysis window ----
if FILTER_WINDOW:
    out_of_window = (
        df["recording_date"].isna()
        | (df["recording_date"] < ANALYSIS_START_DATE)
        | (df["recording_date"] > ANALYSIS_END_DATE)
    )
    mask = (df["excluded"] == False) & out_of_window
    n_removed = int(mask.sum())
    df.loc[mask, "excluded"] = True
    df.loc[mask, "exclusion_reason"] = "outside_analysis_window"
    print(f"\nFilter 1: Analysis window "
          f"{ANALYSIS_START_DATE.date()}..{ANALYSIS_END_DATE.date()} "
          f"-> removed {n_removed} recordings")
else:
    print("\nFilter 1: Analysis window -> SKIPPED")

# ---- Filter 2: recovery period ----
if FILTER_RECOVERY:
    mask = (df["excluded"] == False) & (df["diet_phase"] == "recovery")
    n_removed = int(mask.sum())
    df.loc[mask, "excluded"] = True
    df.loc[mask, "exclusion_reason"] = "recovery_period"
    print(f"Filter 2: Recovery period -> removed {n_removed} recordings")
else:
    print("Filter 2: Recovery period -> SKIPPED (kept)")

# ---- Filter 3: missing estrous phase ----
if FILTER_NO_ESTROUS:
    estrous_is_missing = (
        df["estrous_phase"].isna()
        | (df["estrous_phase"].astype(str).str.strip() == "")
    )
    mask = (df["excluded"] == False) & estrous_is_missing
    n_removed = int(mask.sum())
    df.loc[mask, "excluded"] = True
    df.loc[mask, "exclusion_reason"] = "no_estrous_phase"
    print(f"Filter 3: Missing estrous phase -> removed {n_removed} recordings")
else:
    print("Filter 3: Missing estrous phase -> SKIPPED (kept)")


# ============================================================
# 5. SPLIT INTO KEPT vs EXCLUDED
# ============================================================

kept_df = df[df["excluded"] == False].copy()
excluded_df = df[df["excluded"] == True].copy()

# Tidy up the kept table: drop the bookkeeping columns
kept_df = kept_df.drop(columns=["excluded", "exclusion_reason"])


# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

kept_df.to_csv(OUTPUT_KEPT_PATH, index=False)
excluded_df.to_csv(OUTPUT_EXCLUDED_PATH, index=False)

print(f"\nSaved KEPT recordings to:\n{OUTPUT_KEPT_PATH}")
print(f"Saved EXCLUDED recordings to:\n{OUTPUT_EXCLUDED_PATH}")


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

n_kept = len(kept_df)
n_excluded = len(excluded_df)

print(f"\nStarting recordings:  {n_start}")
print(f"Kept for analysis:    {n_kept}")
print(f"Excluded:             {n_excluded}")

if n_excluded > 0:
    print("\nExclusion reasons:")
    print(excluded_df["exclusion_reason"].value_counts())

print("\nKept recordings per group:")
print(kept_df.groupby("group").size())

if "diet_phase" in kept_df.columns:
    print("\nKept recordings per diet phase:")
    print(kept_df.groupby("diet_phase").size())

if "estrous_phase" in kept_df.columns:
    print("\nKept recordings per estrous phase:")
    print(kept_df["estrous_phase"].value_counts(dropna=False))

if n_kept > 0 and "days_on_diet" in kept_df.columns:
    diet_only = kept_df.loc[kept_df["diet_phase"] == "diet", "days_on_diet"]
    if len(diet_only) > 0:
        print("\nDays-on-diet range (diet phase only, kept):")
        print(f"  min = {diet_only.min():.0f} days")
        print(f"  max = {diet_only.max():.0f} days")

if n_kept > 0 and "body_weight" in kept_df.columns:
    valid_w = kept_df["body_weight"].dropna()
    if len(valid_w) > 0:
        print("\nBody-weight range (kept):")
        print(f"  min  = {valid_w.min():.1f} g")
        print(f"  max  = {valid_w.max():.1f} g")
        print(f"  mean = {valid_w.mean():.1f} g")
        print("\nBody-weight source distribution (kept):")
        print(kept_df["body_weight_source"].value_counts())


print("\nSTEP 01C finished successfully.")