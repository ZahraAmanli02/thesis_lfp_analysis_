# ============================================================
# 05AB_FEATURE_OVERVIEW_FIGURE.PY
# Purpose:
# One-page overview figure of the extracted features, for showing
# the pipeline output (e.g. in a supervisor meeting). Confirmed-bad
# recordings are excluded; no mixed-effects / per-mouse clustering is
# applied, so the panels are descriptive at the recording level.
#
# Panel A: relative band-power profile per band (HFD vs CTRL),
#          mean +/- SEM. Shows the overall spectral shape.
# Panel B: the 5 band-power ratios (HFD vs CTRL) on a log10 scale,
#          as boxplots (median + IQR). Shows the ratio features.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05b_band_ratios_<CABLE>/05b_band_ratios_<CABLE>.csv
#
# Output:
#   outputs/05ab_feature_overview_<CABLE>/
#       05ab_feature_overview_<CABLE>.png
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)
BAND_RATIOS_PATH = os.path.join(
    OUTPUT_DIR, f"05b_band_ratios_{CABLE}", f"05b_band_ratios_{CABLE}.csv"
)

OUT_DIR = os.path.join(
    OUTPUT_DIR, f"05ab_feature_overview_{CABLE}"
)
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(OUT_DIR, f"05ab_feature_overview_{CABLE}.png")


# Colours (colour-blind-friendly blue / red)
C_CTRL = "#4C72B0"
C_HF = "#C44E52"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

# Bands shown in Panel A (must match 05a column names)
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
BAND_LABELS = ["Delta\n1-4", "Theta\n4-10", "Beta\n15-30",
               "Low \u03b3\n30-60", "High \u03b3\n60-100", "Fast \u03b3\n100-140"]

# Ratios shown in Panel B (must match 05b column names)
RATIOS = ["theta_delta", "low_gamma_theta", "high_gamma_theta",
          "fast_gamma_theta", "beta_theta"]
RATIO_LABELS = ["theta/delta", "low \u03b3/theta", "high \u03b3/theta",
                "fast \u03b3/theta", "beta/theta"]


# ============================================================
# 2. LOAD
# ============================================================

if not os.path.exists(BAND_POWERS_PATH):
    raise FileNotFoundError(f"Missing 05a output:\n{BAND_POWERS_PATH}")
if not os.path.exists(BAND_RATIOS_PATH):
    raise FileNotFoundError(f"Missing 05b output:\n{BAND_RATIOS_PATH}")

bp = pd.read_csv(BAND_POWERS_PATH)
br = pd.read_csv(BAND_RATIOS_PATH)

rel_cols = [f"{b}_rel" for b in BANDS]
log_cols = [f"log_{r}" for r in RATIOS]


# ============================================================
# 3. FIGURE
# ============================================================

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Panel A: relative band power profile ----
x = np.arange(len(BANDS))
w = 0.38
for i, (grp, col) in enumerate([("CTRL", C_CTRL), ("HF", C_HF)]):
    sub = bp[bp["group"] == grp]
    means = [sub[c].mean() for c in rel_cols]
    sems = [sub[c].std() / np.sqrt(len(sub)) for c in rel_cols]
    axA.bar(x + (i - 0.5) * w, means, w, yerr=sems, capsize=3,
            color=col, alpha=0.85, label=f"{grp} (n={len(sub)})",
            edgecolor="white", linewidth=0.5)

axA.set_xticks(x)
axA.set_xticklabels(BAND_LABELS, fontsize=9)
axA.set_ylabel("Relative power\n(fraction of 1-140 Hz)")
axA.set_xlabel("Frequency band (Hz)", fontsize=10)
axA.set_title("A.  Spectral profile by band", loc="left")
axA.legend(frameon=False, fontsize=10)

# ---- Panel B: band ratios (log10) ----
positions = np.arange(len(RATIOS))
box_w = 0.34
for i, (grp, col) in enumerate([("CTRL", C_CTRL), ("HF", C_HF)]):
    sub = br[br["group"] == grp]
    data = [sub[c].dropna().values for c in log_cols]
    pos = positions + (i - 0.5) * box_w
    bplot = axB.boxplot(data, positions=pos, widths=box_w * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
    for patch in bplot["boxes"]:
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

axB.set_xticks(positions)
axB.set_xticklabels(RATIO_LABELS, fontsize=9, rotation=15)
axB.set_ylabel("log10(ratio)")
axB.set_title("B.  Band-power ratios", loc="left")
axB.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
axB.legend(handles=[Patch(facecolor=C_CTRL, alpha=0.75, label="CTRL"),
                    Patch(facecolor=C_HF, alpha=0.75, label="HF")],
           frameon=False, fontsize=10)

# ---- titles / caption ----
fig.suptitle(f"LH LFP feature extraction \u2014 {CABLE}  (HFD vs CTRL)",
             fontsize=13, fontweight="bold", y=1.02)
n_total = len(bp)
caption = (f"Feature overview ({n_total} clean recordings; confirmed-bad "
           "recordings excluded). No mixed-effects / per-mouse clustering "
           "applied, so group differences are descriptive at the recording "
           "level. Panel A: mean \u00b1 SEM. Panel B: box = IQR, line = "
           "median, outliers hidden.")
fig.text(0.5, -0.07, caption, ha="center", va="top", fontsize=8,
         color="#444444", wrap=True)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved overview figure: {OUT_PATH}")
print(f"  CTRL n={len(bp[bp['group']=='CTRL'])}, "
      f"HF n={len(bp[bp['group']=='HF'])}")