# ============================================================
# 10A_PREPARE_FEATURES.PY
#
# Purpose:
#   Prepare the per-recording feature table used by 10b and 10c.
#   This is the FIRST of three scripts following PROFESSOR's supervision
#   meeting on 2026-07-09 (see MEMORY / meeting transcript).
#
#
#   * separate models per estrous cycle phase (A, B, C, D)
#   * separate models per frequency band (6 bands)
#   * separate models per band-to-band ratio (5 ratios)
#   * NO body weight as a feature — weight becomes a target
#   * NO band/total ratios — relative power already IS this
#   * cables stay separate (Cable1 / Cable3)
#
# What this script does:
#   1. Load band-power features from 05a.
#   2. Add log10 of absolute band power (magnitude on log scale).
#   3. Add log10 of the fifteen pairwise band-to-band ratios
#      (all from ABSOLUTE power; higher-frequency band / lower).
#      The five ratios PROFESSOR originally named are marked ***:
#         theta_delta            = theta / delta            ***  arousal
#         beta_delta             = beta / delta
#         low_gamma_delta        = low_gamma / delta
#         high_gamma_delta       = high_gamma / delta
#         fast_gamma_delta       = fast_gamma / delta
#         beta_theta             = beta / theta             ***  feeding
#         low_gamma_theta        = low_gamma / theta        ***  slow-gamma
#         high_gamma_theta       = high_gamma / theta       ***  high-gamma
#         fast_gamma_theta       = fast_gamma / theta       ***  fast gamma
#         low_gamma_beta         = low_gamma / beta
#         high_gamma_beta        = high_gamma / beta
#         fast_gamma_beta        = fast_gamma / beta
#         high_gamma_low_gamma   = high_gamma / low_gamma
#         fast_gamma_low_gamma   = fast_gamma / low_gamma
#         fast_gamma_high_gamma  = fast_gamma / high_gamma
#   4. Keep only recordings that have an estrous phase in {A,B,C,D}
#      (per-phase models cannot use rows with missing phase).
#   5. Sanity-check body_weight is present for every kept row.
#   6. Report per-phase eligibility (n recordings, n mice, HF/CTRL
#      balance) so 10b and 10c can decide which cells to skip.
#   7. Save a clean feature CSV that 10b and 10c will both read.
#
# Feature blocks used downstream (per model):
#     band cell   -> [log_<band>_abs, <band>_rel]       (2 features)
#     ratio cell  -> [log_<ratio>]                       (1 feature)
#   No cross-band mixing.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#
# Output:
#   outputs/10a_features_<CABLE>/
#       10a_features_<CABLE>.csv         (clean feature table)
#       10a_eligibility_<CABLE>.txt      (per-phase counts + notes)
#
# Usage:
#   Set CABLE = "Cable1" or "Cable3" in SETTINGS, run twice.
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"                 # switch to "Cable3" for the parallel run
POSITIVE_GROUP = "HF"            # HF is the positive class in 10b

# Minimum data for a phase to be worth modelling in 10b/10c.
# PROFESSOR did not name a threshold; these are conservative defaults.
MIN_RECORDINGS_PER_PHASE = 12
MIN_MICE_PER_PHASE = 6

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

# Band-to-band ratios, all computed from ABSOLUTE band power.
# Full pairwise set = C(6,2) = 15 ratios, always higher-frequency
# band in the numerator and lower-frequency in the denominator
# (neuroscience convention).
# Original 5 named by PROFESSOR are marked ***.
# Order preserved for the downstream heatmap x-axis.
RATIOS = {
    # /delta baseline (5 ratios: everything vs slow)
    "theta_delta":            ("theta",      "delta"),   # ***  arousal
    "beta_delta":             ("beta",       "delta"),
    "low_gamma_delta":        ("low_gamma",  "delta"),
    "high_gamma_delta":       ("high_gamma", "delta"),
    "fast_gamma_delta":       ("fast_gamma", "delta"),
    # /theta baseline (4 ratios: cross-frequency vs arousal band)
    "beta_theta":             ("beta",       "theta"),   # ***  feeding transitions
    "low_gamma_theta":        ("low_gamma",  "theta"),   # ***  slow-gamma coupling
    "high_gamma_theta":       ("high_gamma", "theta"),   # ***  high-gamma coupling
    "fast_gamma_theta":       ("fast_gamma", "theta"),   # ***  high-frequency component
    # /beta baseline (3 ratios: gamma vs activation band)
    "low_gamma_beta":         ("low_gamma",  "beta"),
    "high_gamma_beta":        ("high_gamma", "beta"),
    "fast_gamma_beta":        ("fast_gamma", "beta"),
    # within-gamma (3 ratios: gamma sub-band shifts)
    "high_gamma_low_gamma":   ("high_gamma",  "low_gamma"),
    "fast_gamma_low_gamma":   ("fast_gamma",  "low_gamma"),
    "fast_gamma_high_gamma":  ("fast_gamma",  "high_gamma"),
}

ESTROUS_PHASES = ["A", "B", "C", "D"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"10a_features_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"10a_features_{CABLE}.csv")
OUT_TXT = os.path.join(OUT_DIR, f"10a_eligibility_{CABLE}.txt")


# ============================================================
# 2. LOAD + SANITY CHECKS
# ============================================================

if not os.path.exists(BAND_POWERS_PATH):
    raise FileNotFoundError(f"Missing 05a output:\n{BAND_POWERS_PATH}")

df = pd.read_csv(BAND_POWERS_PATH)

required_cols = (
    ["mouse", "group", "estrous_phase", "body_weight"]
    + [f"{b}_abs" for b in BANDS]
    + [f"{b}_rel" for b in BANDS]
)
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"05a is missing required columns: {missing}")


# ============================================================
# 3. ENGINEER LOG FEATURES (absolute power + band-to-band ratios)
# ============================================================
# log10 of raw absolute band power (per band) and log10 of each
# band-to-band ratio. Log scale is used because both absolute
# powers and ratios are right-skewed across recordings.

for b in BANDS:
    df[f"log_{b}_abs"] = np.log10(df[f"{b}_abs"])

ratio_nan_counts = {}
for r_name, (num_band, den_band) in RATIOS.items():
    num = df[f"{num_band}_abs"].to_numpy(dtype=float)
    den = df[f"{den_band}_abs"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
        log_ratio = np.log10(ratio)
    log_ratio = np.where(np.isfinite(log_ratio), log_ratio, np.nan)
    df[f"log_{r_name}"] = log_ratio
    ratio_nan_counts[r_name] = int(np.isnan(log_ratio).sum())


# ============================================================
# 4. FILTER — recordings must have a valid estrous phase
# ============================================================

n_before = len(df)
df = df[df["estrous_phase"].isin(ESTROUS_PHASES)].reset_index(drop=True)
n_dropped_no_phase = n_before - len(df)

n_missing_wt = df["body_weight"].isna().sum()
if n_missing_wt:
    raise ValueError(
        f"{n_missing_wt} kept rows have no body_weight — "
        "01b / 05a should have interpolated these."
    )


# ============================================================
# 5. PER-PHASE ELIGIBILITY REPORT
# ============================================================

def phase_stats(sub):
    """Return a small dict describing a per-phase slice."""
    n = len(sub)
    n_mice = sub["mouse"].nunique()
    n_hf = int((sub["group"] == POSITIVE_GROUP).sum())
    n_ctrl = int((sub["group"] != POSITIVE_GROUP).sum())
    ok = (
        n >= MIN_RECORDINGS_PER_PHASE
        and n_mice >= MIN_MICE_PER_PHASE
        and n_hf > 0
        and n_ctrl > 0
    )
    reason = ""
    if not ok:
        if n < MIN_RECORDINGS_PER_PHASE:
            reason = "too_few_recordings"
        elif n_mice < MIN_MICE_PER_PHASE:
            reason = "too_few_mice"
        else:
            reason = "one_group_only"
    return {
        "n_recordings": n,
        "n_mice": n_mice,
        "n_HF": n_hf,
        "n_CTRL": n_ctrl,
        "eligible": ok,
        "reason": reason,
    }


eligibility = {p: phase_stats(df[df["estrous_phase"] == p]) for p in ESTROUS_PHASES}


# ============================================================
# 6. SAVE FEATURES + REPORT
# ============================================================

df.to_csv(OUT_CSV, index=False)

lines = []
lines.append("=" * 74)
lines.append(f"10A FEATURE PREP — {CABLE}")
lines.append("=" * 74)
lines.append("")
lines.append(f"Input  : {BAND_POWERS_PATH}")
lines.append(f"Output : {OUT_CSV}")
lines.append("")
lines.append(f"Rows in 05a               : {n_before}")
lines.append(f"Dropped (no estrous phase): {n_dropped_no_phase}")
lines.append(f"Rows kept for modelling   : {len(df)}")
lines.append(f"HF / CTRL                 : "
             f"{(df['group']=='HF').sum()} / {(df['group']!='HF').sum()}")
lines.append(f"Unique mice               : {df['mouse'].nunique()}")
lines.append("")
lines.append(f"Eligibility rule (per phase):")
lines.append(f"  n >= {MIN_RECORDINGS_PER_PHASE}, "
             f"mice >= {MIN_MICE_PER_PHASE}, both groups present")
lines.append("")
lines.append(f"{'phase':<8}{'n':>5}{'mice':>6}{'HF':>5}{'CTRL':>6}"
             f"{'eligible':>11}  reason")
for p in ESTROUS_PHASES:
    s = eligibility[p]
    lines.append(f"{p:<8}{s['n_recordings']:>5}{s['n_mice']:>6}"
                 f"{s['n_HF']:>5}{s['n_CTRL']:>6}"
                 f"{str(s['eligible']):>11}  {s['reason']}")
lines.append("")
lines.append("Feature columns added (per band):")
for b in BANDS:
    lines.append(f"  log_{b}_abs   (log10 of {b}_abs)")
lines.append("")
lines.append("Feature columns added (band-to-band ratios, from absolute power):")
for r_name, (num_band, den_band) in RATIOS.items():
    nan_note = f" [{ratio_nan_counts[r_name]} NaN]" if ratio_nan_counts[r_name] else ""
    lines.append(f"  log_{r_name:<18s} = log10({num_band}_abs / {den_band}_abs){nan_note}")
lines.append("")
lines.append("Next: run 10b_classify_group.py and 10c_regress_weight.py.")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nSTEP 10A finished successfully.")
