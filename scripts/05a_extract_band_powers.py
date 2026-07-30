# ============================================================
# 05A_EXTRACT_BAND_POWERS.PY  (FILTERED VERSION)
# Purpose:
# Turn each recording's multitaper PSD (from 04a) into a small
# set of meaningful band-power features.
#
# For every CLEAN recording (confirmed-bad artifacts from 04b
# are excluded first) we compute, for each frequency band:
#   - absolute power  : mean PSD value inside the band
#   - relative power  : band absolute power / total absolute power
#                       (total = 1-140 Hz), i.e. the fraction of
#                       the spectrum that sits in that band.
#
# Absolute power tells you "how much energy"; relative power tells
# you "how the energy is distributed", which is robust to overall
# amplitude differences between recordings/animals.
#
# Bands (Hz; lower bound inclusive, upper bound exclusive):
#   delta       1-4
#   theta       4-10
#   beta        15-30
#   low_gamma   30-60
#   high_gamma  60-100
#   fast_gamma  100-140
#
# Note: the 10-15 Hz range is intentionally NOT assigned to any
# band (the former "transition" band was dropped). It is still
# included in the 1-140 Hz total used for relative power, so
# relative powers across the listed bands sum to slightly BELOW 1
# (the missing fraction is the 10-15 Hz gap). This is expected and
# correct: relative power means "fraction of the 1-140 Hz spectrum
# in this band", and that gap genuinely carries some power.
#
# Power is the MEAN PSD value inside the band (same convention as
# the total power used in 04e), so all magnitudes are comparable.
#
# Input:
#   outputs/04a_multitaper_psd_summary_filtered_<CABLE>.csv
#   outputs/04b_bad_recording_check_filtered_<CABLE>/
#       04b_confirmed_bad_recordings_<CABLE>.csv   (exclusion list)
#
# Output:
#   outputs/05a_band_powers_<CABLE>/
#       05a_band_powers_<CABLE>.csv
#     one row per clean recording: metadata + 6 absolute + 6 relative
# ============================================================

import os
import numpy as np
import pandas as pd

# numpy renamed trapz -> trapezoid in newer versions; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

# Resolve project root from this script's location (scripts/ -> root),
# so paths work regardless of the project folder name.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PSD_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    f"04a_multitaper_psd_summary_filtered_{CABLE}.csv"
)

CONFIRMED_BAD_PATH = os.path.join(
    OUTPUT_DIR,
    f"04b_bad_recording_check_filtered_{CABLE}",
    f"04b_confirmed_bad_recordings_{CABLE}.csv"
)

OUT_DIR = os.path.join(
    OUTPUT_DIR,
    f"05a_band_powers_{CABLE}"
)
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(
    OUT_DIR,
    f"05a_band_powers_{CABLE}.csv"
)


# Frequency bands (lower inclusive, upper exclusive)
# NOTE: 10-15 Hz is deliberately not assigned to any band; it is
# still part of TOTAL_RANGE so it counts toward total/relative power.
BANDS = {
    "delta":       (1, 4),
    "theta":       (4, 10),
    "beta":        (15, 30),
    "low_gamma":   (30, 60),
    "high_gamma":  (60, 100),
    "fast_gamma":  (100, 140),
}

# Total range for relative power (denominator)
TOTAL_RANGE = (1, 140)


# ============================================================
# 2. LOAD PSD SUMMARY
# ============================================================

if not os.path.exists(PSD_SUMMARY_PATH):
    raise FileNotFoundError(
        f"PSD summary not found:\n{PSD_SUMMARY_PATH}\n"
        f"Run step 04a first."
    )

psd_summary = pd.read_csv(PSD_SUMMARY_PATH)
psd_summary = psd_summary[psd_summary["status"] == "OK"].reset_index(drop=True)

print(f"\nLoaded PSD summary: {len(psd_summary)} successful recordings")


# ============================================================
# 3. EXCLUDE CONFIRMED BAD RECORDINGS (from 04b)
# ------------------------------------------------------------
# Artifact recordings confirmed in 04b must not contribute features.
# Note: this removes individual bad RECORDINGS, not whole mice — a
# mouse with a few bad recordings still keeps its good ones.
# ============================================================

n_before = len(psd_summary)
if os.path.exists(CONFIRMED_BAD_PATH):
    confirmed_bad = pd.read_csv(CONFIRMED_BAD_PATH)
    bad_files = set(
        confirmed_bad.get("differential_file", pd.Series([], dtype=str))
    )
    if bad_files:
        psd_summary = psd_summary[
            ~psd_summary["differential_file"].isin(bad_files)
        ].reset_index(drop=True)
    print(f"Confirmed-bad exclusion (from 04b): "
          f"removed {n_before - len(psd_summary)}, "
          f"{len(psd_summary)} clean recordings remain.")
else:
    print("Confirmed-bad list not found (run 04b first) — "
          f"proceeding on all {len(psd_summary)} recordings.")


# ============================================================
# 4. BAND-POWER HELPERS
# ------------------------------------------------------------
# absolute power = MEAN PSD value in the band (same convention as
#   the total power used in 04e, so magnitudes stay comparable).
# relative power = band AREA / total AREA (trapezoidal integral),
#   i.e. the fraction of spectral power in that band. Relative
#   powers therefore sum to ~1 across bands that tile 1-140 Hz,
#   and read naturally as proportions.
# ============================================================

def band_mean_power(psd_df, lo, hi):
    """Mean PSD value in [lo, hi) Hz. NaN if the band is empty."""
    band = psd_df[
        (psd_df["frequency_Hz"] >= lo) &
        (psd_df["frequency_Hz"] < hi)
    ]
    if len(band) == 0:
        return np.nan
    return float(band["mean_power"].mean())


def band_area_power(psd_df, lo, hi):
    """Area under the PSD in [lo, hi) Hz (trapezoidal integral)."""
    band = psd_df[
        (psd_df["frequency_Hz"] >= lo) &
        (psd_df["frequency_Hz"] < hi)
    ]
    if len(band) < 2:
        return np.nan
    return float(_trapz(band["mean_power"].values,
                        band["frequency_Hz"].values))


# ============================================================
# 5. EXTRACT FEATURES PER RECORDING
# ============================================================

# Metadata columns to carry through if present in the summary
META_COLS = [
    "mouse", "group", "cable", "differential_file",
    "recording_date", "diet_phase", "days_on_diet",
    "days_in_recovery", "body_weight", "body_weight_source",
    "estrous_phase",
]

rows = []
n_missing_psd = 0

for _, row in psd_summary.iterrows():
    psd_path = row["psd_csv_path"]
    if not isinstance(psd_path, str) or not os.path.exists(psd_path):
        n_missing_psd += 1
        continue

    psd_df = pd.read_csv(psd_path)

    # Total area = denominator for relative (area-fraction) power
    total_area = band_area_power(psd_df, TOTAL_RANGE[0], TOTAL_RANGE[1])

    feat = {}
    for c in META_COLS:
        if c in psd_summary.columns:
            feat[c] = row.get(c, np.nan)

    # Absolute (mean) + relative (area fraction) power per band
    for band_name, (lo, hi) in BANDS.items():
        abs_power = band_mean_power(psd_df, lo, hi)
        area_power = band_area_power(psd_df, lo, hi)
        feat[f"{band_name}_abs"] = abs_power
        if total_area and total_area > 0 and area_power is not None \
                and not np.isnan(area_power):
            feat[f"{band_name}_rel"] = area_power / total_area
        else:
            feat[f"{band_name}_rel"] = np.nan

    # Keep mean total power too (comparable to 04e's total_power)
    feat["total_power"] = band_mean_power(psd_df, TOTAL_RANGE[0], TOTAL_RANGE[1])
    rows.append(feat)

features_df = pd.DataFrame(rows)

if n_missing_psd > 0:
    print(f"Note: {n_missing_psd} recording(s) skipped (PSD file missing).")


# ============================================================
# 6. SAVE
# ============================================================

features_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved band-power features: {OUT_PATH}")
print(f"  {len(features_df)} recordings x "
      f"{len(BANDS) * 2} band features (+ metadata, total_power)")


# ============================================================
# 7. QUICK SUMMARY (sanity check)
# ============================================================

print("\n" + "=" * 60)
print("BAND-POWER FEATURE SUMMARY")
print("=" * 60)

if len(features_df) > 0:
    print("\nRecordings per group:")
    print(features_df.groupby("group").size())

    rel_cols = [f"{b}_rel" for b in BANDS]
    print("\nMean relative power per band, by group:")
    summary = features_df.groupby("group")[rel_cols].mean()
    summary.columns = [c.replace("_rel", "") for c in summary.columns]
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    # Relative powers sum to BELOW 1 because the 10-15 Hz range is
    # not assigned to any band but is still part of the 1-140 Hz
    # total. The "missing" fraction = power in the 10-15 Hz gap.
    rel_sum = features_df[rel_cols].sum(axis=1)
    gap_area = []
    for _, r in psd_summary.iterrows():
        p = r["psd_csv_path"]
        if isinstance(p, str) and os.path.exists(p):
            pdf = pd.read_csv(p)
            tot = band_area_power(pdf, TOTAL_RANGE[0], TOTAL_RANGE[1])
            gap = band_area_power(pdf, 10, 15)
            gap_area.append(gap / tot if tot and tot > 0 else np.nan)
    gap_frac = np.nanmean(gap_area) if gap_area else np.nan
    print(f"\nRelative-power sum across bands (expected BELOW 1; the "
          f"10-15 Hz gap is excluded from bands but kept in total):")
    print(f"  mean = {rel_sum.mean():.3f}, "
          f"min = {rel_sum.min():.3f}, max = {rel_sum.max():.3f}")
    print(f"  mean 10-15 Hz gap fraction = {gap_frac:.3f} "
          f"(sum + gap should be ~1)")

    n_nan = features_df[rel_cols].isna().any(axis=1).sum()
    if n_nan > 0:
        print(f"\nWarning: {n_nan} recording(s) have NaN in >=1 band.")
else:
    print("\nNo features were extracted — check inputs.")


print("\nSTEP 05A (band powers) finished successfully.")