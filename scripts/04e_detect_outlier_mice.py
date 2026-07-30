# ============================================================
# 04E_DETECT_OUTLIER_MICE.PY  (FILTERED VERSION)
# Purpose:
# OBJECTIVELY detect mice whose overall signal magnitude is
# statistically inconsistent with the rest of the cohort.
#
# This step replaces the previous mouse-average PSD script
# Its new job is METHODOLOGICAL: to provide
# a transparent, reproducible criterion for flagging outlier
# mice — instead of "I saw it in a plot, so I excluded it."
#
# Method (robust to outliers themselves):
#   1. For each recording: compute total broadband power from
#      its multitaper PSD (output of step 04a, filtered).
#   2. Aggregate to MOUSE LEVEL: median total power per mouse.
#   3. Robust z-score across mice:
#        modified_z = 0.6745 * (x - median) / MAD
#      (Iglewicz & Hoaglin, 1993)
#   4. Flag any mouse with |modified_z| > 3.5 as an outlier.
#
# We run the same check on three independent magnitude metrics
# so that "Mouse X is an outlier" is not driven by a single
# arbitrary number:
#   - total_power_median  (median over the mouse's recordings)
#   - total_power_max     (max recording for that mouse)
#   - total_power_iqr     (within-mouse variability)
#
# Input:
#   outputs/04a_multitaper_psd_summary_filtered_<CABLE>.csv
#
# Output:
#   outputs/04e_outlier_detection_filtered_<CABLE>/
#     04e_per_recording_total_power_<CABLE>.csv
#     04e_per_mouse_outlier_scores_<CABLE>.csv
#     04e_outlier_mice_flagged_<CABLE>.csv
#     04e_outlier_detection_plot_<CABLE>.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PSD_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    f"04a_multitaper_psd_summary_filtered_{CABLE}.csv"
)

OUT_DIR = os.path.join(
    OUTPUT_DIR,
    f"04e_outlier_detection_filtered_{CABLE}"
)
os.makedirs(OUT_DIR, exist_ok=True)


# Frequency range for total power (matches 05a)
TOTAL_RANGE = (1, 140)

# Threshold for the modified z-score (Iglewicz & Hoaglin, 1993)
MODIFIED_Z_THRESHOLD = 3.5

# Multiplier so that 0.6745 / MAD ~ std of a normal
MAD_CONSTANT = 0.6745


# ============================================================
# 2. LOAD PSD SUMMARY
# ============================================================

if not os.path.exists(PSD_SUMMARY_PATH):
    raise FileNotFoundError(
        f"PSD summary not found:\n{PSD_SUMMARY_PATH}\n"
        f"Run step 04a (filtered) first."
    )

psd_summary = pd.read_csv(PSD_SUMMARY_PATH)
psd_summary = psd_summary[psd_summary["status"] == "OK"].reset_index(drop=True)

print(f"\nLoaded PSD summary:")
print(f"  Successful PSDs: {len(psd_summary)}")
print(psd_summary.groupby("group").size())


# ------------------------------------------------------------
# Remove CONFIRMED BAD recordings BEFORE the statistical test.
# Stage 1 (04b quality screening + 04c/04b2 visual confirmation)
# decides which recordings are true artifacts; this outlier test
# is Stage 2 and must run only on clean signals. 
# never run the outlier test on noise/artifacts.
# The confirmed-bad list is written by 04b.
# ------------------------------------------------------------
CONFIRMED_BAD_PATH = os.path.join(
    OUTPUT_DIR,
    f"04b_bad_recording_check_filtered_{CABLE}",
    f"04b_confirmed_bad_recordings_{CABLE}.csv"
)

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
    print(f"\nConfirmed-bad exclusion (from 04b):")
    print(f"  Excluded {n_before - len(psd_summary)} recording(s); "
          f"{len(psd_summary)} clean recordings remain.")
else:
    print(f"\nConfirmed-bad list not found (run 04b first).")
    print(f"  Proceeding on all {len(psd_summary)} recordings.")


# ============================================================
# 3. COMPUTE TOTAL POWER PER RECORDING
# ============================================================

def compute_total_power(psd_csv_path):
    """Mean PSD power in [TOTAL_RANGE[0], TOTAL_RANGE[1])."""
    df = pd.read_csv(psd_csv_path)
    band = df[
        (df["frequency_Hz"] >= TOTAL_RANGE[0]) &
        (df["frequency_Hz"] < TOTAL_RANGE[1])
    ]
    if len(band) == 0:
        return np.nan
    return float(band["mean_power"].mean())


print("\nComputing per-recording total power...")

per_rec_rows = []
for _, row in psd_summary.iterrows():
    psd_path = row["psd_csv_path"]
    if not os.path.exists(psd_path):
        print(f"PSD missing, skip: {psd_path}")
        continue

    total_power = compute_total_power(psd_path)

    per_rec_rows.append({
        "mouse": int(row["mouse"]),
        "group": row["group"],
        "differential_file": row["differential_file"],
        "days_on_diet": row.get("days_on_diet", np.nan),
        "body_weight": row.get("body_weight", np.nan),
        "total_power": total_power,
    })

per_rec_df = pd.DataFrame(per_rec_rows)

per_rec_path = os.path.join(
    OUT_DIR, f"04e_per_recording_total_power_{CABLE}.csv"
)
per_rec_df.to_csv(per_rec_path, index=False)
print(f"\nSaved: {per_rec_path}")


# ============================================================
# 4. AGGREGATE TO MOUSE LEVEL
# ============================================================

per_mouse = (
    per_rec_df.groupby(["mouse", "group"])
    .agg(
        n_recordings=("total_power", "count"),
        total_power_median=("total_power", "median"),
        total_power_max=("total_power", "max"),
        total_power_q25=("total_power", lambda x: np.percentile(x, 25)),
        total_power_q75=("total_power", lambda x: np.percentile(x, 75)),
    )
    .reset_index()
)

per_mouse["total_power_iqr"] = (
    per_mouse["total_power_q75"] - per_mouse["total_power_q25"]
)

print("\nPer-mouse magnitude summary:")
print(
    per_mouse[[
        "mouse", "group", "n_recordings",
        "total_power_median", "total_power_max", "total_power_iqr"
    ]].to_string(index=False)
)


# ============================================================
# 5. ROBUST OUTLIER TEST PER METRIC
# ============================================================

def modified_z_scores(values):
    """
    Iglewicz & Hoaglin (1993) modified z-score:
        z_i = 0.6745 * (x_i - median) / MAD
    Robust to outliers because median and MAD are not pulled by them.
    """
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        # Fallback: avoid division by zero
        return np.zeros_like(values), median, 0.0
    z = MAD_CONSTANT * (values - median) / mad
    return z, float(median), float(mad)


metrics_to_test = ["total_power_median", "total_power_max", "total_power_iqr"]

# Apply to LOG-transformed values, because power spans many orders of magnitude.
# Outliers in raw power look enormous on linear scale; log makes the comparison fair.
#
# IMPORTANT: we flag only the HIGH side (z > +threshold), not |z|. For all three
# metrics, only an unusually HIGH value is a problem:
#   - median / max : abnormally HIGH power (possible artifact or biological extreme)
#   - iqr          : abnormally HIGH between-recording variability
# A LOW value (e.g. a mouse whose recordings are very consistent -> low IQR, or
# low overall power) is not an outlier, so a two-sided |z| test would wrongly flag
# perfectly clean, stable mice.
for metric in metrics_to_test:
    raw = per_mouse[metric].values
    # All positive (powers); guard anyway
    safe = np.where(raw > 0, raw, np.nan)
    log_vals = np.log10(safe)
    z, med, mad = modified_z_scores(log_vals)

    per_mouse[f"{metric}_log10"] = log_vals
    per_mouse[f"{metric}_modz"] = z
    per_mouse[f"{metric}_is_outlier"] = z > MODIFIED_Z_THRESHOLD

    print(
        f"\n{metric}: log10 median={med:.3f}, MAD={mad:.3f}, "
        f"threshold modz > +{MODIFIED_Z_THRESHOLD} (high side only)"
    )


# A mouse is flagged overall if ANY of the three metrics flag it.
outlier_cols = [f"{m}_is_outlier" for m in metrics_to_test]
per_mouse["is_outlier_any"] = per_mouse[outlier_cols].any(axis=1)
per_mouse["n_metrics_flagging"] = per_mouse[outlier_cols].sum(axis=1)


# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

per_mouse_path = os.path.join(
    OUT_DIR, f"04e_per_mouse_outlier_scores_{CABLE}.csv"
)
per_mouse.to_csv(per_mouse_path, index=False)
print(f"\nSaved per-mouse outlier scores: {per_mouse_path}")

flagged = per_mouse[per_mouse["is_outlier_any"]].copy()
flagged_path = os.path.join(
    OUT_DIR, f"04e_outlier_mice_flagged_{CABLE}.csv"
)
flagged.to_csv(flagged_path, index=False)
print(f"Saved flagged outlier mice:    {flagged_path}")


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER DETECTION RESULT")
print("=" * 70)

print("\nPer-mouse modified z-scores (log10 power):")
display_cols = [
    "mouse", "group", "n_recordings",
    "total_power_median_modz",
    "total_power_max_modz",
    "total_power_iqr_modz",
    "n_metrics_flagging",
    "is_outlier_any",
]
print(per_mouse[display_cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

print(f"\nMice flagged as outliers (modz > +{MODIFIED_Z_THRESHOLD} on >=1 metric):")
if len(flagged) == 0:
    print("  None.")
else:
    for _, r in flagged.iterrows():
        print(
            f"  Mouse {int(r['mouse'])} ({r['group']}): "
            f"flagged on {int(r['n_metrics_flagging'])}/3 metrics"
        )


# ============================================================
# 8. PLOT
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

GROUP_COLORS = {"CTRL": "tab:blue", "HF": "tab:red"}

for ax, metric in zip(axes, metrics_to_test):
    sorted_df = per_mouse.sort_values(by=f"{metric}_modz").reset_index(drop=True)

    colors = [GROUP_COLORS.get(g, "gray") for g in sorted_df["group"]]
    edge = [
        "black" if flag else "none"
        for flag in sorted_df[f"{metric}_is_outlier"]
    ]
    linewidths = [
        2.0 if flag else 0.0
        for flag in sorted_df[f"{metric}_is_outlier"]
    ]

    ax.bar(
        x=range(len(sorted_df)),
        height=sorted_df[f"{metric}_modz"],
        color=colors,
        edgecolor=edge,
        linewidth=linewidths,
    )

    # Threshold lines
    ax.axhline(MODIFIED_Z_THRESHOLD, color="black",
               linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-MODIFIED_Z_THRESHOLD, color="black",
               linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)

    # X-tick labels = mouse IDs
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(
        [f"M{int(m)}" for m in sorted_df["mouse"]],
        rotation=90, fontsize=8
    )

    ax.set_ylabel("Modified z-score (log10 power)")
    ax.set_title(metric.replace("_", " "))
    ax.grid(True, alpha=0.3, axis="y")

# Title first (top), then legend below it, with reserved top margin
fig.suptitle(
    f"Objective outlier detection — per-mouse robust z-scores ({CABLE})",
    fontsize=13, y=1.10
)

# Legend
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()
] + [
    Patch(facecolor="white", edgecolor="black",
          linewidth=2, label=f"Outlier (modz > +{MODIFIED_Z_THRESHOLD})")
]
fig.legend(handles=legend_elems, loc="upper center",
           ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.04))

plt.tight_layout()
# Reserve room at the top (title+legend) and bottom (rotated mouse labels)
plt.subplots_adjust(top=0.88, bottom=0.12)

plot_path = os.path.join(
    OUT_DIR, f"04e_outlier_detection_plot_{CABLE}.png"
)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved plot: {plot_path}")


# ============================================================
# 9. METHOD NOTE for thesis / presentation
# ============================================================

method_note = f"""
============================================================
METHOD NOTE  (paste into thesis / presentation)
============================================================

Objective outlier detection (mouse level)
-----------------------------------------
To avoid subjective exclusion of animals, each mouse was
screened with a transparent, reproducible criterion before
proceeding with downstream analysis.

For each mouse, the broadband total power ({TOTAL_RANGE[0]}–{TOTAL_RANGE[1]} Hz)
of every recording was computed from its multitaper PSD.
Three summary statistics were derived per mouse on the
log10-transformed values: the median, the maximum, and the
inter-quartile range across recordings.

A robust modified z-score
(Iglewicz & Hoaglin, 1993; z = 0.6745 * (x - median) / MAD)
was computed across mice for each of the three statistics.
A mouse was flagged as an outlier if z > +{MODIFIED_Z_THRESHOLD} (high side only)
on at least one of the three metrics. Only abnormally HIGH power or
variability is treated as an outlier; an unusually low or very
consistent mouse is not flagged.

All downstream analyses were then performed in TWO versions:
including all mice, and excluding flagged outlier mice. This
allows the reader to assess whether any reported group effect
depends on the outlier(s).
"""
print(method_note)

note_path = os.path.join(OUT_DIR, f"04e_method_note_{CABLE}.txt")
with open(note_path, "w") as f:
    f.write(method_note)
print(f"Saved method note: {note_path}")


print("\nSTEP 04E (objective outlier detection) finished successfully.")