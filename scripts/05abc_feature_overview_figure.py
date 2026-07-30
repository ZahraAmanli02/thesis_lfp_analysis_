# ============================================================
# 05ABC_FEATURE_OVERVIEW_FIGURE.PY
# Purpose:
# One-page, three-panel overview of ALL extracted features, for
# showing the pipeline output (e.g. in a supervisor meeting).
# Confirmed-bad recordings are excluded; no mixed-effects / per-mouse
# clustering is applied, so the panels are descriptive at the
# recording level.
#
# Panel A: relative band-power profile per band (05a), HFD vs CTRL,
#          mean +/- SEM. Shows the overall spectral shape.
# Panel B: the 5 band-power ratios (05b), HFD vs CTRL, log10 scale,
#          boxplots (median + IQR). Shows the ratio features.
# Panel C: oscillation-episode features (05c), HFD vs CTRL. Shows the
#          time-domain features (how often / how long rhythms burst).
#          Episode RATE per band is plotted by default (interpretable,
#          comparable across bands); switch EPISODE_METRIC to use a
#          different feature.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05b_band_ratios_<CABLE>/05b_band_ratios_<CABLE>.csv
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_oscillation_episode_features_<CABLE>.csv
#
# Output:
#   outputs/05abc_feature_overview_<CABLE>/
#       05abc_feature_overview_<CABLE>.png
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

# Which 05c episode feature to show in Panel C.
# Options: "episode_rate", "fraction_of_time",
#          "mean_duration_sec", "mean_amplitude"
EPISODE_METRIC = "episode_rate"
EPISODE_METRIC_LABEL = {
    "episode_rate":      "Episode rate (per min)",
    "fraction_of_time":  "Fraction of time in episodes",
    "mean_duration_sec": "Mean episode duration (s)",
    "mean_amplitude":    "Mean episode amplitude",
}[EPISODE_METRIC]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)
BAND_RATIOS_PATH = os.path.join(
    OUTPUT_DIR, f"05b_band_ratios_{CABLE}", f"05b_band_ratios_{CABLE}.csv"
)
EPISODES_PATH = os.path.join(
    OUTPUT_DIR, f"05c_oscillation_episodes_{CABLE}",
    f"05c_oscillation_episode_features_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"05abc_feature_overview_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, f"05abc_feature_overview_{CABLE}.png")


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

# Panel A bands (05a column names)
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
BAND_LABELS = ["Delta\n1-4", "Theta\n4-10", "Beta\n15-30",
               "Low \u03b3\n30-60", "High \u03b3\n60-100", "Fast \u03b3\n100-140"]

# Panel B ratios (05b column names)
RATIOS = ["theta_delta", "low_gamma_theta", "high_gamma_theta",
          "fast_gamma_theta", "beta_theta"]
RATIO_LABELS = ["theta/delta", "low \u03b3/theta", "high \u03b3/theta",
                "fast \u03b3/theta", "beta/theta"]

# Panel C episode bands (05c column names)
EP_BANDS = ["beta", "low_gamma", "high_gamma"]
EP_LABELS = ["Beta\n15-30", "Low \u03b3\n30-60", "High \u03b3\n60-100"]


# ============================================================
# 2. LOAD
# ============================================================

for p, step in [(BAND_POWERS_PATH, "05a"), (BAND_RATIOS_PATH, "05b"),
                (EPISODES_PATH, "05c")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {step} output:\n{p}")

bp = pd.read_csv(BAND_POWERS_PATH)
br = pd.read_csv(BAND_RATIOS_PATH)
ep = pd.read_csv(EPISODES_PATH)

rel_cols = [f"{b}_rel" for b in BANDS]
log_cols = [f"log_{r}" for r in RATIOS]
ep_cols = [f"{b}_{EPISODE_METRIC}" for b in EP_BANDS]


# ============================================================
# 3. FIGURE (3 panels)
# ============================================================

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17, 5.2))

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
axA.set_title("A.  Spectral profile (05a)", loc="left")
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
axB.set_title("B.  Band-power ratios (05b)", loc="left")
axB.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
axB.legend(handles=[Patch(facecolor=C_CTRL, alpha=0.75, label="CTRL"),
                    Patch(facecolor=C_HF, alpha=0.75, label="HF")],
           frameon=False, fontsize=10)

# ---- Panel C: episode features ----
positions = np.arange(len(EP_BANDS))
for i, (grp, col) in enumerate([("CTRL", C_CTRL), ("HF", C_HF)]):
    sub = ep[ep["group"] == grp]
    data = [sub[c].dropna().values for c in ep_cols]
    pos = positions + (i - 0.5) * box_w
    bplot = axC.boxplot(data, positions=pos, widths=box_w * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
    for patch in bplot["boxes"]:
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
axC.set_xticks(positions)
axC.set_xticklabels(EP_LABELS, fontsize=9)
axC.set_ylabel(EPISODE_METRIC_LABEL)
axC.set_xlabel("Frequency band (Hz)", fontsize=10)
axC.set_title("C.  Oscillation episodes (05c)", loc="left")
axC.legend(handles=[Patch(facecolor=C_CTRL, alpha=0.75, label="CTRL"),
                    Patch(facecolor=C_HF, alpha=0.75, label="HF")],
           frameon=False, fontsize=10)

# ---- titles / caption ----
fig.suptitle(f"LH LFP feature extraction \u2014 {CABLE}  (HFD vs CTRL)",
             fontsize=14, fontweight="bold", y=1.02)
n_total = len(bp)
caption = (f"Feature overview ({n_total} clean recordings; confirmed-bad "
           "recordings excluded). No mixed-effects / per-mouse clustering "
           "applied, so group differences are descriptive at the recording "
           "level. A: mean \u00b1 SEM. B, C: box = IQR, line = median, "
           "outliers hidden. Power and ratios from the multitaper PSD; "
           "episodes from the Hilbert envelope (median + 3\u00b7MAD threshold).")
fig.text(0.5, -0.06, caption, ha="center", va="top", fontsize=8,
         color="#444444", wrap=True)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved 3-panel overview figure: {OUT_PATH}")
print(f"  CTRL n={len(bp[bp['group']=='CTRL'])}, "
      f"HF n={len(bp[bp['group']=='HF'])}")
print(f"  Panel C metric: {EPISODE_METRIC}")