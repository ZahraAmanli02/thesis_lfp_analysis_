# ============================================================
# 04F_OUTLIER_PLOT_SIMPLE.PY  (FILTERED VERSION)
# Purpose:
# A simple, biologist-friendly outlier figure that makes the
# result of the statistical test (04e) obvious by eye.
#
# Two panels:
#   A) Every recording as one dot, grouped by mouse, on a log-y
#      axis, with the "normal range" (95% of the recordings from
#      non-outlier mice) shaded so any outlier mouse stands out.
#   B) Per-mouse summary: dot = median, vertical line = min-max
#      across that mouse's recordings, same shaded normal range.
#
# Outlier mice (black border) are taken directly from 04e's
# per-mouse table (high-side modified z-score). This script only
# visualises 04e's decision; it does not re-decide anything.
#
# Input  (from 04e's folder):
#   04e_outlier_detection_filtered_<CABLE>/
#     04e_per_recording_total_power_<CABLE>.csv
#     04e_per_mouse_outlier_scores_<CABLE>.csv
#
# Output (its own folder):
#   04f_outlier_plot_simple_<CABLE>/
#     04f_outlier_detection_SIMPLE_<CABLE>.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 04e's output folder — this is where the CSVs to read live
IN_DIR = os.path.join(
    OUTPUT_DIR,
    f"04e_outlier_detection_filtered_{CABLE}"
)

PER_REC_CSV = os.path.join(
    IN_DIR, f"04e_per_recording_total_power_{CABLE}.csv"
)
PER_MOUSE_CSV = os.path.join(
    IN_DIR, f"04e_per_mouse_outlier_scores_{CABLE}.csv"
)

# 04f's own output folder — the simple plot is written here
OUT_DIR = os.path.join(
    OUTPUT_DIR,
    f"04f_outlier_plot_simple_{CABLE}"
)
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_COLORS = {"CTRL": "tab:blue", "HF": "tab:red"}


# ============================================================
# 2. LOAD
# ============================================================

for p in (PER_REC_CSV, PER_MOUSE_CSV):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Required 04e output not found:\n{p}\nRun 04e first."
        )

per_rec = pd.read_csv(PER_REC_CSV)
per_mouse = pd.read_csv(PER_MOUSE_CSV)

print(f"\nLoaded:")
print(f"  Per-recording: {len(per_rec)} rows")
print(f"  Per-mouse:     {len(per_mouse)} rows")


# ============================================================
# 3. ORDER MICE: CTRL first, then HF, by mouse number
# ============================================================

group_order = {"CTRL": 0, "HF": 1}
per_mouse["__order"] = per_mouse.apply(
    lambda r: (group_order.get(r["group"], 99), r["mouse"]), axis=1
)
per_mouse = per_mouse.sort_values("__order").reset_index(drop=True)
mouse_order = per_mouse["mouse"].tolist()
mouse_to_x = {m: i for i, m in enumerate(mouse_order)}


# ============================================================
# 4. WHICH MICE ARE OUTLIERS (from 04e)
# ------------------------------------------------------------
# 04e already flagged outliers on the HIGH side (median/max/iqr).
# We just read its decision. "is_outlier_any" is the union flag.
# ============================================================

if "is_outlier_any" in per_mouse.columns:
    outlier_mice = per_mouse.loc[
        per_mouse["is_outlier_any"] == True, "mouse"
    ].tolist()
else:
    # Fallback: rebuild from the three per-metric flags
    flag_cols = [
        c for c in per_mouse.columns if c.endswith("_is_outlier")
    ]
    outlier_mice = per_mouse.loc[
        per_mouse[flag_cols].any(axis=1), "mouse"
    ].tolist()

print(f"\nOutlier mice (from 04e): {outlier_mice}")


# ============================================================
# 5. NORMAL RANGE (95% of recordings from NON-outlier mice)
# ============================================================

non_outlier_powers = per_rec[
    ~per_rec["mouse"].isin(outlier_mice)
]["total_power"].dropna()
non_outlier_powers = non_outlier_powers[non_outlier_powers > 0]

if len(non_outlier_powers) == 0:
    # Degenerate guard: if every mouse was flagged, use all recordings
    non_outlier_powers = per_rec["total_power"].dropna()
    non_outlier_powers = non_outlier_powers[non_outlier_powers > 0]

normal_low = np.percentile(non_outlier_powers, 2.5)
normal_high = np.percentile(non_outlier_powers, 97.5)
normal_median = np.median(non_outlier_powers)

print(
    f"\nNormal range (2.5-97.5 pct of non-outlier recordings): "
    f"{normal_low:.2e} to {normal_high:.2e}"
)


# ============================================================
# 6. PLOT
# ============================================================

fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6))


# ----- Panel A: each recording as a dot -----
rng = np.random.default_rng(0)  # reproducible jitter
for _, row in per_rec.iterrows():
    power = row["total_power"]
    if pd.isna(power) or power <= 0:
        continue
    if row["mouse"] not in mouse_to_x:
        continue
    x = mouse_to_x[row["mouse"]]
    is_out = row["mouse"] in outlier_mice
    axA.scatter(
        x + rng.uniform(-0.12, 0.12),
        power,
        s=55,
        color=GROUP_COLORS.get(row["group"], "gray"),
        alpha=0.75,
        edgecolor="black" if is_out else "white",
        linewidth=1.5 if is_out else 0.5,
        zorder=3,
    )

axA.axhspan(normal_low, normal_high, color="gray", alpha=0.13, zorder=1,
            label="Normal range (non-outlier mice)")
axA.axhline(normal_median, color="gray", linestyle="--",
            linewidth=0.9, alpha=0.7, zorder=2)

axA.set_yscale("log")
axA.set_xticks(range(len(mouse_order)))
axA.set_xticklabels(
    [f"M{m}\n{per_mouse.loc[per_mouse['mouse'] == m, 'group'].iloc[0]}"
     for m in mouse_order],
    fontsize=10
)
axA.set_ylabel("Total signal power per recording (1-140 Hz)")
axA.set_title("A. Every dot is one recording")
axA.grid(True, axis="y", alpha=0.3)


# ----- Panel B: per-mouse median + min-max -----
for _, row in per_mouse.iterrows():
    if row["mouse"] not in mouse_to_x:
        continue
    x = mouse_to_x[row["mouse"]]
    is_out = row["mouse"] in outlier_mice

    recs = per_rec[per_rec["mouse"] == row["mouse"]]["total_power"]
    recs = recs[(recs.notna()) & (recs > 0)]
    if len(recs) == 0:
        continue
    lo, hi = recs.min(), recs.max()
    median_val = row.get("total_power_median", np.median(recs))

    axB.plot([x, x], [lo, hi],
             color=GROUP_COLORS.get(row["group"], "gray"),
             linewidth=2, alpha=0.5)
    axB.scatter(
        x, median_val,
        s=140,
        color=GROUP_COLORS.get(row["group"], "gray"),
        edgecolor="black" if is_out else "white",
        linewidth=2.0 if is_out else 0.6,
        zorder=4,
    )

axB.axhspan(normal_low, normal_high, color="gray", alpha=0.13, zorder=1)
axB.axhline(normal_median, color="gray", linestyle="--",
            linewidth=0.9, alpha=0.7, zorder=2)

axB.set_yscale("log")
axB.set_xticks(range(len(mouse_order)))
axB.set_xticklabels(
    [f"M{m}\n{per_mouse.loc[per_mouse['mouse'] == m, 'group'].iloc[0]}"
     for m in mouse_order],
    fontsize=10
)
axB.set_ylabel("Total signal power per recording (1-140 Hz)")
axB.set_title("B. Per-mouse summary: dot = median, line = min-max")
axB.grid(True, axis="y", alpha=0.3)

# Annotate outlier mice on Panel B
for m in outlier_mice:
    if m not in mouse_to_x:
        continue
    x = mouse_to_x[m]
    yrow = per_mouse.loc[per_mouse["mouse"] == m, "total_power_median"]
    if len(yrow) == 0:
        continue
    y = yrow.iloc[0]
    axB.annotate(
        f"Mouse {m}\n(outlier)",
        xy=(x, y),
        xytext=(x - 1.2, y * 4),
        fontsize=10, fontweight="bold", ha="right",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )


# ----- Shared legend + title -----
legend_elements = [
    Patch(facecolor=GROUP_COLORS["CTRL"], label="CTRL"),
    Patch(facecolor=GROUP_COLORS["HF"], label="HF"),
    Patch(facecolor="lightgray", label="Normal range (95% of non-outlier recordings)"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=2, markersize=11,
               label="Outlier mouse (black border)"),
]

fig.suptitle(
    f"Why a mouse is flagged as an outlier - {CABLE}",
    fontsize=14, y=1.06
)
fig.legend(handles=legend_elements, loc="upper center",
           ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.00))

plt.tight_layout()
plt.subplots_adjust(top=0.86)

out_path = os.path.join(
    OUT_DIR, f"04f_outlier_detection_SIMPLE_{CABLE}.png"
)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved simple plot: {out_path}")

print("\nSTEP 04F (simple outlier plot) finished successfully.")