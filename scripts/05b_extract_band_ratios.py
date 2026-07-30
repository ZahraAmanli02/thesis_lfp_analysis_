# ============================================================
# 05B_EXTRACT_BAND_RATIOS.PY
# Purpose:
# Turn the per-recording absolute band powers (from 05a) into a
# small set of band-power RATIOS.
#
# A ratio normalises one band by another measured in the SAME
# recording. Because numerator and denominator share the same
# recording-level scaling (electrode impedance, tissue contact,
# day-to-day amplitude), a ratio cancels much of the between-
# recording / between-animal amplitude variability that raw
# absolute power suffers from, and isolates the relative balance
# between two rhythms.
#
# Ratios computed (all from ABSOLUTE band power):
#   theta_delta       = theta / delta        state / arousal
#   low_gamma_theta   = low_gamma / theta     slow-gamma coupling
#   high_gamma_theta  = high_gamma / theta    high-gamma coupling
#   fast_gamma_theta  = fast_gamma / theta    high-frequency component
#   beta_theta        = beta / theta          feeding transitions
#
# All ratios except theta_delta share theta as the denominator,
# i.e. they describe how each faster rhythm sits relative to the
# theta rhythm (theta-nested cross-frequency balance). theta_delta
# is kept as a separate state/arousal context variable.
#
# For each ratio we store BOTH:
#   - the raw ratio        (for plotting / interpretation)
#   - log10(ratio)         (for statistics; ratios are right-skewed,
#                           the log makes them roughly symmetric and
#                           suitable for mixed-effects models)
#
# Input:
#   outputs/05a_band_powers_<CABLE>/
#       05a_band_powers_<CABLE>.csv
#     (already excludes confirmed-bad recordings, so no further
#      filtering is needed here — 05b inherits 05a's clean set.)
#
# Output:
#   outputs/05b_band_ratios_<CABLE>/
#       05b_band_ratios_<CABLE>.csv
#     one row per clean recording: metadata + 5 ratios + 5 log10 ratios
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

# Resolve project root from this script's location (scripts/ -> root),
# so paths work regardless of the project folder name.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR,
    f"05a_band_powers_{CABLE}",
    f"05a_band_powers_{CABLE}.csv"
)

OUT_DIR = os.path.join(
    OUTPUT_DIR,
    f"05b_band_ratios_{CABLE}"
)
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(
    OUT_DIR,
    f"05b_band_ratios_{CABLE}.csv"
)


# Ratio definitions: name -> (numerator_band, denominator_band)
# Each operand is the ABSOLUTE band-power column "<band>_abs" in 05a.
RATIOS = {
    "theta_delta":      ("theta",      "delta"),
    "low_gamma_theta":  ("low_gamma",  "theta"),
    "high_gamma_theta": ("high_gamma", "theta"),
    "fast_gamma_theta": ("fast_gamma", "theta"),
    "beta_theta":       ("beta",       "theta"),
}

# Metadata columns to carry through if present in the 05a table
META_COLS = [
    "mouse", "group", "cable", "differential_file",
    "recording_date", "diet_phase", "days_on_diet",
    "days_in_recovery", "body_weight", "body_weight_source",
    "estrous_phase",
]


# ============================================================
# 2. LOAD 05A BAND POWERS
# ============================================================

if not os.path.exists(BAND_POWERS_PATH):
    raise FileNotFoundError(
        f"Band-power table not found:\n{BAND_POWERS_PATH}\n"
        f"Run step 05a first."
    )

bp = pd.read_csv(BAND_POWERS_PATH)
print(f"\nLoaded 05a band powers: {len(bp)} recordings")

# Sanity: required absolute-power columns must exist
needed_bands = set()
for num, den in RATIOS.values():
    needed_bands.add(num)
    needed_bands.add(den)

missing_cols = [f"{b}_abs" for b in needed_bands
                if f"{b}_abs" not in bp.columns]
if missing_cols:
    raise KeyError(
        f"Missing absolute-power columns in 05a output: {missing_cols}"
    )


# ============================================================
# 3. COMPUTE RATIOS (+ log10)
# ------------------------------------------------------------
# raw ratio   = numerator_abs / denominator_abs
# log10 ratio = log10(raw ratio)  for statistics
#
# Denominators here (delta, theta) are strong, always-present
# bands, so division is safe. We still guard against any
# non-positive / missing value to avoid inf or invalid logs.
# ============================================================

out = pd.DataFrame()

# Carry metadata
for c in META_COLS:
    if c in bp.columns:
        out[c] = bp[c]

n_bad_ratio = 0

for ratio_name, (num_band, den_band) in RATIOS.items():
    num = bp[f"{num_band}_abs"].astype(float)
    den = bp[f"{den_band}_abs"].astype(float)

    # Safe ratio: invalid where denominator <= 0 or either is NaN
    valid = (den > 0) & num.notna() & den.notna()
    ratio = pd.Series(np.nan, index=bp.index, dtype=float)
    ratio[valid] = num[valid] / den[valid]

    # log10 only where ratio is finite and > 0
    log_valid = valid & (ratio > 0)
    log_ratio = pd.Series(np.nan, index=bp.index, dtype=float)
    log_ratio[log_valid] = np.log10(ratio[log_valid])

    out[ratio_name] = ratio
    out[f"log_{ratio_name}"] = log_ratio

    n_bad_ratio += int((~valid).sum())

if n_bad_ratio > 0:
    print(f"Note: {n_bad_ratio} ratio value(s) set to NaN "
          f"(non-positive/missing denominator or numerator).")


# ============================================================
# 4. SAVE
# ============================================================

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved band ratios: {OUT_PATH}")
print(f"  {len(out)} recordings x {len(RATIOS)} ratios "
      f"(+ {len(RATIOS)} log10 ratios, + metadata)")


# ============================================================
# 5. QUICK SUMMARY (sanity check)
# ============================================================

print("\n" + "=" * 60)
print("BAND-RATIO FEATURE SUMMARY")
print("=" * 60)

if len(out) > 0:
    print("\nRecordings per group:")
    print(out.groupby("group").size())

    ratio_cols = list(RATIOS.keys())

    print("\nMedian ratio per group (raw ratios; median is robust "
          "to the right skew):")
    summary = out.groupby("group")[ratio_cols].median()
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nMean log10(ratio) per group (what statistics will use):")
    log_cols = [f"log_{r}" for r in ratio_cols]
    summary_log = out.groupby("group")[log_cols].mean()
    summary_log.columns = [c.replace("log_", "") for c in summary_log.columns]
    print(summary_log.to_string(float_format=lambda x: f"{x:.3f}"))

    # NaN / inf checks
    n_nan = out[ratio_cols].isna().any(axis=1).sum()
    if n_nan > 0:
        print(f"\nWarning: {n_nan} recording(s) have NaN in >=1 ratio.")
    n_inf = np.isinf(out[ratio_cols].to_numpy()).any()
    if n_inf:
        print("Warning: infinite ratio value(s) present — check inputs.")
else:
    print("\nNo ratios were computed — check inputs.")


print("\nSTEP 05B (band ratios) finished successfully.")