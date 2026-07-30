# ============================================================
# 01B_ADD_METADATA.PY
# Purpose:
# Add per-recording metadata to the inventory from step 01:
#   - recording_date     : extracted from the filename
#   - diet_phase         : "baseline" / "diet" / "recovery"
#   - days_on_diet       : recording_date - DIET_START_DATE, ONLY during
#                          the diet phase (NaN in baseline and recovery)
#   - days_in_recovery   : recording_date - RECOVERY_START_DATE, ONLY during
#                          the recovery phase (NaN otherwise)
#   - body_weight        : from lab's Excel; pre-diet recordings get the
#                          baseline (first) measurement, diet-phase gaps are
#                          linearly interpolated, recordings after the last
#                          measurement are left as NaN (no extrapolation)
#   - body_weight_source : "measured" / "interpolated" / "baseline" /
#                          "extrapolated" / "no_data"
#   - estrous_phase      : merged from lab's swab Excel (A/B/C/D), using
#                          the final-state rule for transitions (see below)
#
# This step does NOT drop any recordings. It only ENRICHES every
# recording from step 01 with metadata. All filtering (analysis-window
# date range, missing-estrous, etc.) happens later in step 01c, so this
# table stays the complete, traceable record of all recordings.
#
# Input:
#   outputs/01_recordings_inventory_<CABLE>.csv     (from step 01)
#   data/Overview_Weighing_CD1.xlsx                 (from lab)
#   data/Swab_Results_CD1.xlsx                      (from lab; estrous cycle)
#
# Output:
#   outputs/01b_recordings_inventory_with_metadata_<CABLE>.csv
# ============================================================

import os
import re
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

# Key dates from lab's notes in the Excel file
DIET_START_DATE     = pd.Timestamp("2026-02-23")   # HF / CTRL food started
RECOVERY_START_DATE = pd.Timestamp("2026-03-30")   # back to normal food

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

INVENTORY_PATH = os.path.join(
    OUTPUT_DIR,
    f"01_recordings_inventory_{CABLE}.csv"
)

WEIGHT_XLSX_PATH = os.path.join(
    DATA_DIR,
    "Overview_Weighing_CD1.xlsx"
)

SWAB_XLSX_PATH = os.path.join(
    DATA_DIR,
    "Swab_Results_CD1.xlsx"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"01b_recordings_inventory_with_metadata_{CABLE}.csv"
)


# ============================================================
# 2. HELPERS
# ============================================================

def extract_date_from_filename(filename):
    """
    Find a yyyy-mm-dd date inside a filename like:
      Mouse1_Cable1-2026-03-31_10-10-35.eeg
    Returns a pd.Timestamp or NaT.
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match is None:
        return pd.NaT
    try:
        return pd.Timestamp(match.group(1))
    except Exception:
        return pd.NaT


def parse_weight_value(value):
    """
    lab's weight cells are like '36.2 g' (string) or 36.2 (number).
    Convert to a float (grams). Returns NaN if the cell is empty.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    # string like "36.2 g" or "36,2 g"
    s = str(value).strip().lower().replace("g", "").replace(",", ".").strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalize_cycle_phase(value):
    """
    Map a raw swab annotation to a single estrous phase letter (A/B/C/D)
    or pd.NA.

    Rule for transitions = FINAL STATE. The swab is taken in the
    afternoon AFTER the recording, so it captures the state the mouse
    has transitioned INTO; for the recording we use that final state.

    Examples (all real values in Swab_Results_CD1.xlsx):
        "A", "B", "C", "D"            -> same letter
        "A -> B", "B -> C", ...       -> final letter (B, C, ...)
        "B (->C)"                     -> C
        "late A", "late B"            -> A, B
        "C (blood!!)"                 -> C
        empty / NaN                   -> NA
        free-text notes (e.g. "very weird; ...", "Did not continue ...")
                                      -> NA  (not a usable phase)
    """
    if pd.isna(value):
        return pd.NA
    s = str(value).strip()
    if s == "":
        return pd.NA

    # "X -> Y" / "X ->Y": take the final state
    m = re.match(r"^[ABCD]\s*->\s*([ABCD])", s)
    if m:
        return m.group(1)

    # "B (->C)": take the state inside the parentheses
    m = re.match(r"^[ABCD]\s*\(\s*->\s*([ABCD])\s*\)", s)
    if m:
        return m.group(1)

    # "late A" / "late B": take the letter
    m = re.match(r"^late\s+([ABCD])", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # "C (blood!!)" etc: clean leading letter followed by a parenthesis
    m = re.match(r"^([ABCD])\s*\(", s)
    if m:
        return m.group(1)

    # plain single letter
    if s in ["A", "B", "C", "D"]:
        return s

    # anything else (free-text notes) -> not a usable phase
    return pd.NA


def load_weight_table(xlsx_path):
    """
    Read lab's Excel (Tabelle1) and return a tidy DataFrame:
        mouse | date | body_weight

    Layout in the Excel:
      - Row 1: "Weighing Overview CD1 mice"  (title)
      - Row 2: empty
      - Row 3: "ID" + dates (2/23/26, 3/2/26, ...)
      - Row 4+: mouse number + weights for each date
    """

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Weight Excel file not found:\n{xlsx_path}\n"
            f"Please put the file in the data/ folder."
        )

    raw = pd.read_excel(xlsx_path, sheet_name="Tabelle1", header=None)

    # ----- Find the header row that contains "ID" in column 0 -----
    header_row_idx = None
    date_columns = {}

    for i in range(min(15, len(raw))):
        row = raw.iloc[i]

        # Get the first non-NaN cell on this row
        first_val = None
        for v in row.values:
            if pd.notna(v):
                first_val = v
                break

        if first_val is None:
            continue

        if not (isinstance(first_val, str) and first_val.strip().lower() == "id"):
            continue

        # Found "ID" row. Collect the dates that follow.
        candidate_dates = {}
        for col_idx, value in enumerate(row.values):
            if pd.isna(value):
                continue
            try:
                ts = pd.Timestamp(value)
                candidate_dates[col_idx] = pd.Timestamp(ts.date())
            except (ValueError, TypeError):
                continue

        if len(candidate_dates) >= 3:
            header_row_idx = i
            date_columns = candidate_dates
            break

    if header_row_idx is None:
        raise ValueError(
            "Could not find the date header row in the Excel file.\n"
            "Expected a row with 'ID' in column A followed by date columns."
        )

    print(f"  Found header row at Excel row {header_row_idx + 1}")
    print(f"  Found {len(date_columns)} date columns")
    print(f"  Date range: {min(date_columns.values()).date()} -> {max(date_columns.values()).date()}")

    # ----- Walk the rows below and collect mouse_id + weights -----
    rows = []
    for r in range(header_row_idx + 1, len(raw)):
        row = raw.iloc[r]
        raw_id = row.iloc[0]

        if pd.isna(raw_id):
            continue

        mouse_id = None
        if isinstance(raw_id, (int, float)) and not pd.isna(raw_id):
            mouse_id = int(raw_id)
        elif isinstance(raw_id, str):
            s = raw_id.strip().lstrip("#")
            if s.isdigit():
                mouse_id = int(s)

        if mouse_id is None:
            continue

        for col_idx, date in date_columns.items():
            weight = parse_weight_value(row.iloc[col_idx])
            if not np.isnan(weight):
                rows.append({
                    "mouse": mouse_id,
                    "date": date,
                    "body_weight": weight
                })

    weight_df = pd.DataFrame(rows)
    weight_df = weight_df.sort_values(["mouse", "date"]).reset_index(drop=True)

    return weight_df


def load_cycle_table(xlsx_path):
    """
    Read lab's swab Excel and return a tidy DataFrame:
        mouse | date | estrous_phase   (normalized A/B/C/D or NA)

    Layout in Swab_Results_CD1.xlsx (Sheet1):
      - Row index 8 (0-based): "ID" in column 0, then one date per column
        (2026-02-12 ... 2026-04-09) across columns 1..57.
      - Rows 9..27: mouse id in column 0 ("#01" ... "#26"), then the raw
        swab annotation for each date.

    Phases are normalized with normalize_cycle_phase() (final-state rule).
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Swab Excel file not found:\n{xlsx_path}\n"
            f"Please put the file in the data/ folder."
        )

    raw = pd.read_excel(xlsx_path, sheet_name="Sheet1", header=None)

    # ----- Find the header row whose column 0 == "ID" and that is
    #       followed by many date columns (the data grid header) -----
    header_row_idx = None
    date_columns = {}

    for i in range(min(20, len(raw))):
        first = raw.iloc[i, 0]
        if not (isinstance(first, str) and first.strip().lower() == "id"):
            continue
        candidate_dates = {}
        for col_idx in range(1, raw.shape[1]):
            value = raw.iloc[i, col_idx]
            if pd.isna(value):
                continue
            try:
                ts = pd.Timestamp(value)
                candidate_dates[col_idx] = pd.Timestamp(ts.date())
            except (ValueError, TypeError):
                continue
        if len(candidate_dates) >= 10:
            header_row_idx = i
            date_columns = candidate_dates
            break

    if header_row_idx is None:
        raise ValueError(
            "Could not find the swab date-header row (column 0 == 'ID' "
            "followed by date columns)."
        )

    print(f"  Found swab header at Excel row {header_row_idx + 1}")
    print(f"  Found {len(date_columns)} date columns")
    print(f"  Date range: {min(date_columns.values()).date()} -> {max(date_columns.values()).date()}")

    # ----- Walk the mouse rows and build long format -----
    rows = []
    n_raw = 0
    for r in range(header_row_idx + 1, len(raw)):
        raw_id = raw.iloc[r, 0]
        if pd.isna(raw_id):
            continue

        mouse_id = None
        if isinstance(raw_id, (int, float)) and not pd.isna(raw_id):
            mouse_id = int(raw_id)
        elif isinstance(raw_id, str):
            s = raw_id.strip().lstrip("#")
            if s.isdigit():
                mouse_id = int(s)
        if mouse_id is None:
            continue

        for col_idx, date in date_columns.items():
            raw_val = raw.iloc[r, col_idx]
            phase = normalize_cycle_phase(raw_val)
            if pd.notna(raw_val):
                n_raw += 1
            rows.append({
                "mouse": mouse_id,
                "date": date,
                "estrous_phase": phase,
            })

    cycle_df = pd.DataFrame(rows)
    # Keep only rows that carry a usable phase (A/B/C/D)
    cycle_df = cycle_df[cycle_df["estrous_phase"].notna()].reset_index(drop=True)

    n_usable = len(cycle_df)
    print(f"  Raw non-empty swab cells: {n_raw}")
    print(f"  Usable phases after normalization (A/B/C/D): {n_usable}")
    print(f"  Phase counts: "
          f"{cycle_df['estrous_phase'].value_counts().to_dict()}")

    return cycle_df


def assign_weight(mouse_id, recording_date, weight_df, post_tolerance_days=3):
    """
    Assign a body weight (grams) to one recording, and report how it
    was obtained. Returns a tuple: (weight, source).

    Rules (agreed with professor):
      - No measurements for the mouse, or no recording date -> (NaN, "no_data")
      - Recording BEFORE the first measurement (i.e. before the diet,
        the baseline period): assign the first measurement, which is the
        pre-diet baseline weight (~Feb 23). lab confirmed the baseline
        recordings were done within ~1-2 weeks of this measurement and
        the grown, weight-stable mice did not change meaningfully in that
        window.                                        -> (baseline, "baseline")
      - Recording BETWEEN two measurements: linear interpolation.
                                                       -> (value, "interpolated")
      - Recording exactly on a measurement day: that measured value.
                                                       -> (value, "measured")
      - Recording shortly AFTER the last measurement (within
        post_tolerance_days, e.g. a 17 Apr recording vs a 16 Apr
        measurement): carry the last value forward, since one or two days
        cannot change the weight meaningfully.         -> (last, "carried")
      - Recording far AFTER the last measurement (beyond the tolerance):
        true extrapolation with no later anchor, so no value is invented.
                                                       -> (NaN, "extrapolated")
    """

    sub = weight_df[weight_df["mouse"] == mouse_id].copy()
    if len(sub) == 0 or pd.isna(recording_date):
        return np.nan, "no_data"

    sub = sub.sort_values("date").reset_index(drop=True)

    base = sub["date"].iloc[0]
    x = (sub["date"] - base).dt.days.astype(float).values
    y = sub["body_weight"].values
    x_query = (recording_date - base).days

    # Before / at the first measurement -> pre-diet baseline weight.
    if x_query < x[0]:
        return float(y[0]), "baseline"
    if x_query == x[0]:
        return float(y[0]), "measured"

    # After the last measurement.
    if x_query > x[-1]:
        if x_query - x[-1] <= post_tolerance_days:
            # Only a day or two later -> carry the last value forward.
            return float(y[-1]), "carried"
        # Too far after the last measurement -> do NOT extrapolate.
        return np.nan, "extrapolated"

    # Exactly on a later measurement day -> that measured value.
    exact = np.where(x == x_query)[0]
    if exact.size > 0:
        return float(y[exact[0]]), "measured"

    # Otherwise between two measurements -> linear interpolation.
    return float(np.interp(x_query, x, y)), "interpolated"


# ============================================================
# 3. LOAD INVENTORY
# ============================================================

if not os.path.exists(INVENTORY_PATH):
    raise FileNotFoundError(
        f"Inventory file not found:\n{INVENTORY_PATH}\n"
        f"Run step 01 first."
    )

inventory = pd.read_csv(INVENTORY_PATH)

print(f"\nLoaded inventory: {INVENTORY_PATH}")
print(f"Total recordings: {len(inventory)}")
print(inventory.head())


# ============================================================
# 4. COMPUTE RECORDING DATE (no filtering here)
# ------------------------------------------------------------
# Every recording is kept. The recording_date is parsed from the
# filename and used below for diet-phase, body-weight and the cycle
# merge. All row filtering (analysis window, etc.) is done in 01c.
# ============================================================

inventory["recording_date"] = inventory["filename"].apply(
    extract_date_from_filename
)

n_no_date = int(inventory["recording_date"].isna().sum())
if n_no_date > 0:
    print(f"\nNote: {n_no_date} recording(s) had no date in filename.")


# ============================================================
# 5. LOAD AND PARSE WEIGHT TABLE
# ============================================================

print(f"\nReading weight Excel: {WEIGHT_XLSX_PATH}")
weight_df = load_weight_table(WEIGHT_XLSX_PATH)
print(f"\nParsed weight measurements: {len(weight_df)} total")
print(f"Mice with weight data: {sorted(weight_df['mouse'].unique().tolist())}")


# ============================================================
# 6. LOAD AND PARSE ESTROUS CYCLE TABLE
# ============================================================

print(f"\nReading swab Excel: {SWAB_XLSX_PATH}")
cycle_df = load_cycle_table(SWAB_XLSX_PATH)
print(f"Mice with cycle data: {sorted(cycle_df['mouse'].unique().tolist())}")


# ============================================================
# 7. ADD METADATA TO EACH RECORDING (diet phase + body weight)
# ============================================================

print("\nAdding diet-phase and body-weight metadata...")

diet_phase = []        # "baseline" / "diet" / "recovery"
days_on_diet = []      # only during the diet phase, else NaN
days_in_recovery = []  # only during the recovery phase, else NaN
body_weights = []
weight_sources = []

for _, row in inventory.iterrows():

    mouse_id = int(row["mouse"])
    rec_date = row["recording_date"]

    if pd.isna(rec_date):
        diet_phase.append("unknown")
        days_on_diet.append(np.nan)
        days_in_recovery.append(np.nan)
        body_weights.append(np.nan)
        weight_sources.append("no_date_in_filename")
        continue

    # ----- Diet phase + per-phase day counters -----
    # Timeline: diet starts 23 Feb (day 0), old food reintroduced 30 Mar.
    #   before 23 Feb            -> baseline   (no diet-day, no recovery-day)
    #   23 Feb ... 30 Mar (incl) -> diet       (days_on_diet = date - 23 Feb)
    #   after 30 Mar             -> recovery   (days_in_recovery = date - 30 Mar)
    if rec_date < DIET_START_DATE:
        diet_phase.append("baseline")
        days_on_diet.append(np.nan)
        days_in_recovery.append(np.nan)
    elif rec_date <= RECOVERY_START_DATE:
        diet_phase.append("diet")
        days_on_diet.append((rec_date - DIET_START_DATE).days)
        days_in_recovery.append(np.nan)
    else:
        diet_phase.append("recovery")
        days_on_diet.append(np.nan)
        days_in_recovery.append((rec_date - RECOVERY_START_DATE).days)

    # ----- Body weight + source -----
    weight, source = assign_weight(mouse_id, rec_date, weight_df)
    if not pd.isna(weight):
        weight = round(weight, 2)
    body_weights.append(weight)
    weight_sources.append(source)

inventory["diet_phase"] = diet_phase
inventory["days_on_diet"] = days_on_diet
inventory["days_in_recovery"] = days_in_recovery
inventory["body_weight"] = body_weights
inventory["body_weight_source"] = weight_sources


# ============================================================
# 8. MERGE ESTROUS PHASE (mouse + date)
# ------------------------------------------------------------
# cycle_df has one (mouse, date) -> phase row. We merge on both keys.
# Within the analysis window swab coverage is complete, so every
# recording should receive a phase; a safety check flags any that
# do not (unexpected).
# ============================================================

print("\nMerging estrous phase on (mouse, date)...")

# Normalize key dtypes for a clean merge
inv_keys = inventory.copy()
inv_keys["mouse"] = inv_keys["mouse"].astype(int)
inv_keys["recording_date"] = pd.to_datetime(inv_keys["recording_date"])

cyc = cycle_df.copy()
cyc["mouse"] = cyc["mouse"].astype(int)
cyc["date"] = pd.to_datetime(cyc["date"])

merged = inv_keys.merge(
    cyc.rename(columns={"date": "recording_date"}),
    on=["mouse", "recording_date"],
    how="left",
    validate="many_to_one",   # a recording maps to at most one swab phase
)

inventory = merged

# Safety / info check: how many recordings have no phase?
# (These are recordings whose date falls outside the swab coverage,
#  e.g. pre-19-Feb or post-2-Apr. They are NOT dropped here — 01c
#  decides that — but it is useful to see the count.)
n_missing_phase = int(inventory["estrous_phase"].isna().sum())
if n_missing_phase > 0:
    print(f"  Note: {n_missing_phase} recording(s) have no estrous phase "
          f"(date outside swab coverage). Kept here; 01c will filter.")
else:
    print("  All recordings received an estrous phase.")


# ============================================================
# 9. SAVE OUTPUT
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
inventory.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved inventory with metadata to:\n{OUTPUT_PATH}")


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nTotal recordings (all kept; filtering happens in 01c): {len(inventory)}")

print("\nRecordings per group:")
print(inventory.groupby("group").size())

print("\nRecordings per diet phase:")
print(inventory.groupby("diet_phase").size())

diet_only = inventory.loc[inventory["diet_phase"] == "diet", "days_on_diet"]
if len(diet_only) > 0:
    print("\nDays on diet (diet phase only):")
    print(f"  min = {diet_only.min():.0f} days")
    print(f"  max = {diet_only.max():.0f} days")

valid_w = inventory["body_weight"].dropna()
if len(valid_w) > 0:
    print("\nBody weight (range, non-NaN):")
    print(f"  min  = {valid_w.min():.1f} g")
    print(f"  max  = {valid_w.max():.1f} g")
    print(f"  mean = {valid_w.mean():.1f} g")

print("\nBody weight sources:")
print(inventory["body_weight_source"].value_counts())

print("\nEstrous phase counts:")
print(inventory["estrous_phase"].value_counts(dropna=False))

print("\nPreview of the new table:")
preview_cols = [
    "mouse", "group", "filename", "recording_date",
    "diet_phase", "days_on_diet", "body_weight",
    "body_weight_source", "estrous_phase",
]
preview_cols = [c for c in preview_cols if c in inventory.columns]
print(inventory[preview_cols].head(10).to_string(index=False))

print("\nSTEP 01B finished successfully.")