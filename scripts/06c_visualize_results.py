# ============================================================
# 06C_VISUALIZE_RESULTS.PY
# Purpose:
# Visualize the mixed-effects model results from 06a in three
# figures designed for the thesis Results chapter and for sharing
# with the supervisor.
#
# Figure 1 — Forest plot of group[T.HF] effects:
#   All 11 primary outcomes (6 band powers + 5 ratios) shown on a
#   single coefficient axis with 95% CI bars. The zero line marks
#   "no group effect". Outcomes that survive FDR (p_fdr<0.05) are
#   highlighted in colour and starred. Lets the reader see the full
#   pattern in one glance: which outcomes shift, in which direction,
#   and by how much.
#
# Figure 2 — Delta-power time trajectory:
#   The strongest finding (delta) plotted across days_on_diet,
#   separately for HF and CTRL. Per-mouse trajectories shown thin
#   in the background, group means thick on top. Makes the result
#   from the model visible at a glance and shows the time dimension
#   the model controls for.
#
# Figure 3 — Exploratory low-gamma episode features:
#   Two significant uncorrected findings (mean duration, fraction
#   of time) shown as boxplots side by side with the corresponding
#   model effect sizes. Bonus figure for the exploratory results.
#
# Input:
#   outputs/06a_mixed_models_<CABLE>/
#       06a_mixed_model_results_<CABLE>.csv
#   outputs/05a_band_powers_<CABLE>/
#       05a_band_powers_<CABLE>.csv          (for raw data overlays)
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_features_combined_with_episodes_<CABLE>.csv
#
# Output:
#   outputs/06c_visualizations_<CABLE>/
#       06c_fig1_forest_plot_<CABLE>.png
#       06c_fig2_delta_trajectory_<CABLE>.png
#       06c_fig3_low_gamma_episodes_<CABLE>.png
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

# Resolve project root from this script's location (scripts/ -> root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RESULTS_PATH = os.path.join(
    OUTPUT_DIR, f"06a_mixed_models_{CABLE}",
    f"06a_mixed_model_results_{CABLE}.csv"
)
BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}",
    f"05a_band_powers_{CABLE}.csv"
)
COMBINED_PATH = os.path.join(
    OUTPUT_DIR, f"05c_oscillation_episodes_{CABLE}",
    f"05c_features_combined_with_episodes_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"06c_visualizations_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)

FIG1_PATH = os.path.join(OUT_DIR, f"06c_fig1_forest_plot_{CABLE}.png")
FIG2_PATH = os.path.join(OUT_DIR, f"06c_fig2_delta_trajectory_{CABLE}.png")
FIG3_PATH = os.path.join(OUT_DIR, f"06c_fig3_low_gamma_episodes_{CABLE}.png")


# Colours
C_CTRL = "#4C72B0"
C_HF   = "#C44E52"
C_SIG  = "#C44E52"   # red for significant
C_NS   = "#888888"   # grey for non-significant

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


# ============================================================
# 2. LOAD
# ============================================================

if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(f"Missing 06a output:\n{RESULTS_PATH}")
if not os.path.exists(BAND_POWERS_PATH):
    raise FileNotFoundError(f"Missing 05a output:\n{BAND_POWERS_PATH}")
if not os.path.exists(COMBINED_PATH):
    raise FileNotFoundError(f"Missing 05c output:\n{COMBINED_PATH}")

results = pd.read_csv(RESULTS_PATH)
df_05a = pd.read_csv(BAND_POWERS_PATH)
df_05c = pd.read_csv(COMBINED_PATH)

# Add log delta power (for Figure 2)
df_05a["log_delta_abs"] = np.log10(df_05a["delta_abs"])

# Restrict to diet-phase recordings (same subset that the model used)
diet_05a = df_05a[df_05a["days_on_diet"].notna()].copy()
diet_05c = df_05c[df_05c["days_on_diet"].notna()].copy()


# ============================================================
# 3. FIGURE 1 — FOREST PLOT
# ------------------------------------------------------------
# Show every primary outcome's group[T.HF] coefficient with 95% CI.
# Sort by raw p-value so the strongest effects sit at the top.
# ============================================================

# Pick primary group[T.HF] rows
primary_grp = results[
    (results["tier"] == "primary") &
    (results["effect"] == "group[T.HF]")
].sort_values("p_value").reset_index(drop=True)

# Nice display labels
LABEL_MAP = {
    "log_delta_abs":         "Delta (1-4 Hz)",
    "log_theta_abs":         "Theta (4-10 Hz)",
    "log_beta_abs":          "Beta (15-30 Hz)",
    "log_low_gamma_abs":     "Low gamma (30-60 Hz)",
    "log_high_gamma_abs":    "High gamma (60-100 Hz)",
    "log_fast_gamma_abs":    "Fast gamma (100-140 Hz)",
    "log_theta_delta":       "theta / delta",
    "log_low_gamma_theta":   "low gamma / theta",
    "log_high_gamma_theta":  "high gamma / theta",
    "log_fast_gamma_theta":  "fast gamma / theta",
    "log_beta_theta":        "beta / theta",
}
primary_grp["label"] = primary_grp["outcome"].map(LABEL_MAP)

fig, ax = plt.subplots(figsize=(9, 7))

# Plot from bottom to top (matplotlib quirk: y-axis goes up)
# We want strongest effect (smallest p) at TOP, so reverse the order.
n = len(primary_grp)
y_positions = np.arange(n)[::-1]

for i, row in primary_grp.iterrows():
    y = y_positions[i]
    is_sig = row["p_fdr"] < 0.05
    colour = C_SIG if is_sig else C_NS
    # Wider band-power family vs ratio family — slight visual distinction
    is_ratio = "/" in row["label"]
    marker = "s" if is_ratio else "o"

    # CI line
    ax.plot([row["ci_low"], row["ci_high"]], [y, y],
            color=colour, linewidth=2, alpha=0.85)
    # Point estimate
    ax.scatter(row["coef"], y, color=colour, s=80,
               marker=marker, edgecolors="white", linewidths=1.2,
               zorder=3)
    # Star + p_fdr text on the right
    label_text = f"  p_fdr = {row['p_fdr']:.3f}"
    if is_sig:
        label_text += "  *"
    ax.text(row["ci_high"] + 0.05, y, label_text,
            va="center", fontsize=9,
            color=colour, fontweight="bold" if is_sig else "normal")

# Zero line
ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

# Axis limits: pad left for the delta CI, and add room on the right so
# the per-row p_fdr text is never clipped. Vertical padding keeps the
# top/bottom rows off the frame edge.
xmin = float(primary_grp["ci_low"].min()) - 0.25
xmax = float(primary_grp["ci_high"].max()) + 0.75
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.8, n - 1 + 0.8)

ax.set_yticks(y_positions)
ax.set_yticklabels(primary_grp["label"], fontsize=10)
ax.set_xlabel("group[T.HF] coefficient (log10 units, 95% CI)",
              fontsize=11)
ax.set_title("Figure 1.  Diet effect on LH LFP features\n"
             "(forest plot, all 11 primary outcomes)",
             loc="left")

# Direction annotation BELOW the panel area (not over the data)
ax.text(0.01, -0.09,
        "Negative coefficient = HF lower than CTRL    "
        "Positive coefficient = HF higher than CTRL",
        transform=ax.transAxes, fontsize=9, va="top",
        color="#444", style="italic")

# Legend — placed in the empty lower-left region (the bottom ratio rows
# sit on the right, so this corner is clear of data and of the p_fdr
# text column on the right).
ax.legend(handles=[
    Patch(facecolor=C_SIG, label="FDR significant (p_fdr < 0.05)"),
    Patch(facecolor=C_NS,  label="Not significant"),
], frameon=True, framealpha=0.9, edgecolor="none", fontsize=9,
   loc="lower left")

# Caption further below
n_total = 89
n_mice = 19
caption = (f"Forest plot of group[T.HF] fixed-effect coefficients from "
           f"11 mixed-effects models (n={n_total} recordings / {n_mice} "
           f"mice; random intercept per mouse). FDR-BH correction "
           f"applied across the 11 primary tests.\n"
           f"Markers: circles = band powers, squares = band ratios. "
           f"Strongest effect: delta band (~6-fold reduction in HF).")
fig.text(0.5, -0.05, caption, ha="center", va="top",
         fontsize=8, color="#444", wrap=True)

plt.tight_layout()
plt.savefig(FIG1_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved Figure 1: {FIG1_PATH}")


# ============================================================
# 4. FIGURE 2 — DELTA POWER TIME TRAJECTORY
# ------------------------------------------------------------
# The strongest finding visualised: log_delta vs days_on_diet,
# per mouse (thin) + group mean (thick).
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

# Per-mouse trajectories (thin lines)
for grp, colour in [("CTRL", C_CTRL), ("HF", C_HF)]:
    sub = diet_05a[diet_05a["group"] == grp]
    for mouse_id in sub["mouse"].unique():
        m_data = sub[sub["mouse"] == mouse_id].sort_values("days_on_diet")
        ax.plot(m_data["days_on_diet"], m_data["log_delta_abs"],
                color=colour, alpha=0.25, linewidth=1)
        ax.scatter(m_data["days_on_diet"], m_data["log_delta_abs"],
                   color=colour, alpha=0.4, s=18,
                   edgecolors="none")

# Group means via binned days_on_diet (smoothing for visual clarity)
for grp, colour in [("CTRL", C_CTRL), ("HF", C_HF)]:
    sub = diet_05a[diet_05a["group"] == grp].copy()
    # Bin into ~5-day windows
    sub["day_bin"] = pd.cut(sub["days_on_diet"],
                            bins=np.arange(5, 41, 5))
    binned = sub.groupby("day_bin", observed=True).agg(
        x=("days_on_diet", "mean"),
        y=("log_delta_abs", "mean"),
        sem=("log_delta_abs",
             lambda v: v.std() / max(np.sqrt(len(v)), 1)),
    ).dropna()
    ax.errorbar(binned["x"], binned["y"], yerr=binned["sem"],
                color=colour, linewidth=2.5, marker="o", markersize=7,
                capsize=4, label=f"{grp} (group mean +/- SEM)",
                zorder=5)

ax.set_xlabel("Days on diet")
ax.set_ylabel("log10(delta absolute power)")
ax.set_title("Figure 2.  Delta-band power across diet exposure\n"
             "(strongest mixed-model finding, p_fdr = 0.015)",
             loc="left")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.grid(True, alpha=0.2, linestyle=":")

# Caption
caption = (f"Per-recording delta power (thin dots/lines, one trace "
           f"per mouse) and binned group means (thick lines, 5-day "
           f"windows, error bars = SEM).\n"
           f"HF mice show a sustained reduction across the diet window. "
           f"Mixed-effects model: group[T.HF] beta = -0.781, "
           f"p_FDR = 0.015.")
fig.text(0.5, -0.03, caption, ha="center", va="top",
         fontsize=8, color="#444", wrap=True)

plt.tight_layout()
plt.savefig(FIG2_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved Figure 2: {FIG2_PATH}")


# ============================================================
# 5. FIGURE 3 — EXPLORATORY: LOW GAMMA EPISODE FEATURES
# ------------------------------------------------------------
# Two significant uncorrected findings:
#   low_gamma_mean_duration_sec  (p=0.003)
#   low_gamma_fraction_of_time   (p=0.025)
# Boxplots + effect sizes from the model.
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

EPISODE_PLOTS = [
    ("low_gamma_mean_duration_sec",
     "Mean duration (s)", "Low gamma burst duration"),
    ("low_gamma_fraction_of_time",
     "Fraction of time", "Low gamma time fraction"),
]

for ax, (col, ylab, title) in zip(axes, EPISODE_PLOTS):
    # Boxplots
    data = [
        diet_05c[diet_05c["group"] == "CTRL"][col].dropna().values,
        diet_05c[diet_05c["group"] == "HF"][col].dropna().values,
    ]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, colour in zip(bp["boxes"], [C_CTRL, C_HF]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    # Jittered raw points
    rng = np.random.default_rng(0)
    for i, (vals, colour) in enumerate(zip(data, [C_CTRL, C_HF])):
        xj = rng.normal(i, 0.05, size=len(vals))
        ax.scatter(xj, vals, color=colour, alpha=0.5, s=18,
                   edgecolors="white", linewidths=0.4, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CTRL", "HF"])
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")

    # Effect-size annotation from the model
    row = results[(results["tier"] == "exploratory") &
                  (results["effect"] == "group[T.HF]") &
                  (results["outcome"] == col)].iloc[0]
    txt = (f"beta = {row['coef']:+.3f}\n"
           f"p = {row['p_value']:.3f}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=10, va="top", color=C_SIG,
            fontweight="bold")

fig.suptitle("Figure 3.  Low-gamma oscillation episodes (exploratory)",
             fontweight="bold", y=1.02, fontsize=12)
caption = ("Boxplots: box = IQR, line = median, dots = recordings "
           "(jittered). Effect sizes from mixed-effects models on the "
           "diet-phase subset (89 recordings / 19 mice). These "
           "exploratory findings are uncorrected for multiple "
           "comparisons.")
fig.text(0.5, -0.02, caption, ha="center", va="top",
         fontsize=8, color="#444", wrap=True)

plt.tight_layout()
plt.savefig(FIG3_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved Figure 3: {FIG3_PATH}")


print("\nSTEP 06C finished successfully.")