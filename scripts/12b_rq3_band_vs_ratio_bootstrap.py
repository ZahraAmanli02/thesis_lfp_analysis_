# ============================================================
# 12B_RQ3_BAND_VS_RATIO_BOOTSTRAP.PY
#
# Purpose:
#   RQ3 follow-up — "which band is more informative?"
#
#   Two comparisons on the RQ1 bootstrap results (SVM-RBF diet
#   classification):
#     1. bands (6 single-band cells) vs ratios (15 cross-band
#        cells) — does the diet signal live in raw band power
#        or in cross-frequency coupling?
#     2. slow bands (delta, theta, beta) vs gamma bands
#        (low_gamma, high_gamma, fast_gamma) — is the signal
#        concentrated in gamma, as the LH / metabolic literature
#        suggests?
#
#   For each comparison we show, per estrous phase, the
#   distribution of mean bootstrap balanced accuracy across
#   the cells in each group, plus overall statistics.
#
# Input:
#   outputs/10b_bootstrap_full/10b_bootstrap_results_long.csv
#
# Outputs:
#   outputs/12b_rq3_band_vs_ratio_bootstrap/
#       12b_band_vs_ratio.png
#       12b_band_vs_ratio_summary.txt
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RESULTS_CSV = os.path.join(
    OUTPUT_DIR, "10b_bootstrap_full", "10b_bootstrap_results_long.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, "12b_rq3_band_vs_ratio_bootstrap")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "12b_band_vs_ratio.png")
OUT_TXT = os.path.join(OUT_DIR, "12b_band_vs_ratio_summary.txt")

MODEL = "svm_rbf"
CHANCE = 0.5

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
SLOW_BANDS = ["delta", "theta", "beta"]
GAMMA_BANDS = ["low_gamma", "high_gamma", "fast_gamma"]
ESTROUS_PHASES = ["A", "B", "C", "D"]

PHASE_NAMES = {"A": "pro-estrus", "B": "estrus",
               "C": "metestrus", "D": "diestrus"}

COLOR_BAND = "#3b7ba9"
COLOR_RATIO = "#4faf4a"
COLOR_SLOW = "#7fb3d5"
COLOR_GAMMA = "#e67e22"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. LOAD RESULTS
# ============================================================

if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Missing results file:\n{RESULTS_CSV}")

df = pd.read_csv(RESULTS_CSV)
svm = df[df["model"] == MODEL].dropna(subset=["boot_mean"]).copy()
svm["group_bands_ratios"] = svm["cell_type"]     # "band" or "ratio"


def band_group(cell):
    if cell in SLOW_BANDS:
        return "slow"
    if cell in GAMMA_BANDS:
        return "gamma"
    return None


svm["band_speed"] = svm["cell"].apply(band_group)   # "slow", "gamma", or None


# ============================================================
# 3. PLOT — 2 panels side by side
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))

phase_x = np.arange(len(ESTROUS_PHASES))
box_offset = 0.18


def draw_paired_dots(ax, groups, color_map, x_positions, jitter=0.06):
    """Draw a small filled circle for every cell's bal_acc, coloured by group."""
    rng = np.random.default_rng(0)
    for phase_idx, phase in enumerate(ESTROUS_PHASES):
        for g_idx, (gname, gcolor) in enumerate(color_map.items()):
            vals = groups.get((phase, gname), [])
            if len(vals) == 0:
                continue
            xc = x_positions[phase_idx] + (g_idx - 0.5) * (2 * box_offset)
            xs = xc + rng.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(xs, vals, s=42, color=gcolor,
                       edgecolor="black", linewidth=0.4,
                       alpha=0.9, zorder=3)


def draw_box_summary(ax, groups, color_map, x_positions):
    """Draw a small horizontal segment at the group mean per phase."""
    for phase_idx, phase in enumerate(ESTROUS_PHASES):
        for g_idx, (gname, gcolor) in enumerate(color_map.items()):
            vals = groups.get((phase, gname), [])
            if len(vals) == 0:
                continue
            xc = x_positions[phase_idx] + (g_idx - 0.5) * (2 * box_offset)
            m = float(np.mean(vals))
            ax.plot([xc - box_offset * 0.75, xc + box_offset * 0.75],
                    [m, m], color=gcolor, lw=3, zorder=4,
                    solid_capstyle="round")


# ---- Panel A: bands vs ratios ----
axA = axes[0]
groups_A = {}
for phase in ESTROUS_PHASES:
    for gname in ("band", "ratio"):
        vals = svm[(svm["phase"] == phase)
                   & (svm["cell_type"] == gname)]["boot_mean"].to_numpy()
        groups_A[(phase, gname)] = vals

axA.axhline(CHANCE, color="#7f7f7f", lw=1.3, ls="--", zorder=1)
for h in (0.4, 0.6, 0.7):
    axA.axhline(h, color="#eeeeee", lw=0.7, zorder=0)

color_map_A = {"band": COLOR_BAND, "ratio": COLOR_RATIO}
draw_paired_dots(axA, groups_A, color_map_A, phase_x)
draw_box_summary(axA, groups_A, color_map_A, phase_x)

axA.set_xticks(phase_x)
axA.set_xticklabels([f"{p}\n{PHASE_NAMES[p]}" for p in ESTROUS_PHASES],
                    fontsize=9)
axA.set_ylabel("Balanced accuracy (bootstrap mean)", fontsize=11)
axA.set_ylim(0.30, 0.80)
axA.set_title("A.  Frequency band  vs  band-to-band ratio",
              fontweight="bold", loc="left", fontsize=11)

# small legend for panel A
axA.scatter([], [], s=42, color=COLOR_BAND, edgecolor="black",
            linewidth=0.4, label=f"band (n=6)")
axA.scatter([], [], s=42, color=COLOR_RATIO, edgecolor="black",
            linewidth=0.4, label=f"ratio (n=15)")
axA.legend(loc="upper right", frameon=False, fontsize=9)

# ---- Panel B: slow vs gamma (bands only) ----
axB = axes[1]
groups_B = {}
for phase in ESTROUS_PHASES:
    for gname in ("slow", "gamma"):
        vals = svm[(svm["phase"] == phase)
                   & (svm["band_speed"] == gname)]["boot_mean"].to_numpy()
        groups_B[(phase, gname)] = vals

axB.axhline(CHANCE, color="#7f7f7f", lw=1.3, ls="--", zorder=1)
for h in (0.4, 0.6, 0.7):
    axB.axhline(h, color="#eeeeee", lw=0.7, zorder=0)

color_map_B = {"slow": COLOR_SLOW, "gamma": COLOR_GAMMA}
draw_paired_dots(axB, groups_B, color_map_B, phase_x)
draw_box_summary(axB, groups_B, color_map_B, phase_x)

axB.set_xticks(phase_x)
axB.set_xticklabels([f"{p}\n{PHASE_NAMES[p]}" for p in ESTROUS_PHASES],
                    fontsize=9)
axB.set_ylabel("Balanced accuracy (bootstrap mean)", fontsize=11)
axB.set_ylim(0.30, 0.80)
axB.set_title("B.  Slow bands (δ, θ, β)  vs  gamma bands "
              "(low, high, fast γ)",
              fontweight="bold", loc="left", fontsize=11)

axB.scatter([], [], s=42, color=COLOR_SLOW, edgecolor="black",
            linewidth=0.4, label="slow (δ, θ, β)")
axB.scatter([], [], s=42, color=COLOR_GAMMA, edgecolor="black",
            linewidth=0.4, label="gamma (low, high, fast γ)")
axB.legend(loc="upper right", frameon=False, fontsize=9)

# title
fig.text(0.02, 0.97,
         "RQ3 — which band is more informative?  (SVM-RBF, diet classification)",
         ha="left", va="top", fontsize=14, fontweight="bold")
fig.text(0.02, 0.935,
         "pooled Cable 1 + Cable 3   ·   bootstrap 1000 iterations   ·   "
         "dots = individual cells,  bar = mean per group",
         ha="left", va="top", fontsize=9.5, color="#666", style="italic")

plt.subplots_adjust(left=0.06, right=0.98, top=0.88,
                    bottom=0.10, wspace=0.20)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure:\n{OUT_PNG}")


# ============================================================
# 4. SUMMARY (TXT)
# ============================================================

def stats_line(vals):
    if len(vals) == 0:
        return "n=0"
    return (f"n={len(vals):2d}  mean={np.mean(vals):.3f}  "
            f"median={np.median(vals):.3f}  "
            f"best={max(vals):.3f}")


lines = []
lines.append("=" * 88)
lines.append("12B RQ3 — BAND vs RATIO   |   SLOW vs GAMMA")
lines.append("=" * 88)
lines.append(f"Source: SVM-RBF bootstrap balanced accuracy per (phase × cell).")
lines.append("")

lines.append("-" * 88)
lines.append("A. FREQUENCY BAND (n=6) vs BAND-TO-BAND RATIO (n=15)")
lines.append("-" * 88)
lines.append(f"{'phase':<26} {'BAND':<45} {'RATIO':<45}")
for phase in ESTROUS_PHASES:
    band_vals = groups_A[(phase, "band")]
    ratio_vals = groups_A[(phase, "ratio")]
    label = f"{phase} — {PHASE_NAMES[phase]}"
    lines.append(f"{label:<26} {stats_line(band_vals):<45} "
                 f"{stats_line(ratio_vals):<45}")
overall_band = svm[svm["cell_type"] == "band"]["boot_mean"].to_numpy()
overall_ratio = svm[svm["cell_type"] == "ratio"]["boot_mean"].to_numpy()
lines.append("")
lines.append(f"{'OVERALL (all phases)':<26} {stats_line(overall_band):<45} "
             f"{stats_line(overall_ratio):<45}")
lines.append("")

lines.append("-" * 88)
lines.append("B. SLOW BANDS (delta, theta, beta) vs GAMMA BANDS "
             "(low, high, fast gamma)")
lines.append("-" * 88)
lines.append(f"{'phase':<26} {'SLOW':<45} {'GAMMA':<45}")
for phase in ESTROUS_PHASES:
    slow_vals = groups_B[(phase, "slow")]
    gamma_vals = groups_B[(phase, "gamma")]
    label = f"{phase} — {PHASE_NAMES[phase]}"
    lines.append(f"{label:<26} {stats_line(slow_vals):<45} "
                 f"{stats_line(gamma_vals):<45}")
overall_slow = svm[svm["band_speed"] == "slow"]["boot_mean"].to_numpy()
overall_gamma = svm[svm["band_speed"] == "gamma"]["boot_mean"].to_numpy()
lines.append("")
lines.append(f"{'OVERALL (all phases)':<26} {stats_line(overall_slow):<45} "
             f"{stats_line(overall_gamma):<45}")
lines.append("")

lines.append("-" * 88)
lines.append("HOW TO READ THIS")
lines.append("-" * 88)
lines.append("  * If the RATIO mean is consistently higher than the BAND mean")
lines.append("    per phase, the diet signal lives in cross-frequency coupling")
lines.append("    rather than in raw band power alone.")
lines.append("  * If the GAMMA mean is higher than the SLOW mean, the signal")
lines.append("    is concentrated in gamma bands — consistent with LH-modulation")
lines.append("    literature and metabolic-hormonal state effects on cortex.")
lines.append("  * Look at Phase B (estrus) in particular — this is where the")
lines.append("    only CI-confirmed cell (fast_gamma / delta) sits, and the")
lines.append("    ratio-vs-band and gamma-vs-slow patterns here matter most.")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved summary:\n{OUT_TXT}")

print("\nSTEP 12B RQ3 (band vs ratio) finished.")
