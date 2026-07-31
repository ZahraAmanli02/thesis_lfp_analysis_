# ============================================================
# 10B3_BOOTSTRAP.PY
#
# Purpose:
#   Simple vertical bar chart of the top-N (phase × cell)
#   combinations, ranked by bootstrap-mean balanced accuracy.
#   Reads the same long-format results file produced by
#   10b_bootstrap_full.py.
#
# Design:
#   * one primary model only (SVM-RBF)
#   * bars sorted descending by mean balanced accuracy
#   * uniform bar colour — no confirmation/CI-based highlighting
#   * bars grow bottom-up; mean value printed on top of each bar
#   * chance reference line at 0.5
#   
#
# Input:
#   outputs/10b_bootstrap_full/10b_bootstrap_results_long.csv
#
# Output:
#   outputs/10b3_bootstrap/10b3_top10_bar.png
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

OUT_DIR = os.path.join(OUTPUT_DIR, "10b3_bootstrap")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "10b3_top10_bar.png")

PRIMARY_MODEL = "svm_rbf"
TOP_N = 10
CHANCE = 0.5
BAR_COLOR_MUTED = "#b0b0b0"      # CI includes / crosses chance
BAR_COLOR_CONFIRMED = "#c0392b"  # 95% CI lies entirely above chance

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

PHASE_BADGE_COLOR = {
    "A": "#e41a1c", "B": "#377eb8",
    "C": "#4daf4a", "D": "#984ea3",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. HELPERS
# ============================================================

def format_cell(cell):
    """Pretty ratio names: 'fast_gamma_delta' -> 'fast_gamma / delta'."""
    if cell in BANDS:
        return cell
    for b in sorted(BANDS, key=len, reverse=True):
        if cell.startswith(b + "_"):
            rest = cell[len(b) + 1:]
            if rest in BANDS:
                return f"{b} / {rest}"
    return cell


# ============================================================
# 3. LOAD & RANK
# ============================================================

if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Missing results file:\n{RESULTS_CSV}")

df = pd.read_csv(RESULTS_CSV)
svm = (df[df["model"] == PRIMARY_MODEL]
       .dropna(subset=["boot_mean"])
       .copy())

# Rank by the 95% CI LOWER bound (boot_ci_lo). This puts the cells
# whose distribution is most solidly above chance at the top, and pushes
# high-mean-but-wide-CI cells down the list — which reflects statistical
# confidence rather than the raw point estimate.
top = (svm.sort_values(["boot_ci_lo", "boot_mean"], ascending=[False, False])
          .head(TOP_N)
          .reset_index(drop=True))

print(f"Top {TOP_N} cells by 95% CI lower bound ({PRIMARY_MODEL}):")
for i, row in top.iterrows():
    print(f"  {i+1:2d}. phase {row['phase']}  {format_cell(row['cell']):<24s}  "
          f"mean={row['boot_mean']:.3f}  "
          f"CI=[{row['boot_ci_lo']:.3f}, {row['boot_ci_hi']:.3f}]")


# ============================================================
# 4. PLOT — vertical bars, descending order
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))
x_pos = np.arange(len(top))

# subtle horizontal gridlines
for h in (0.5, 0.6, 0.7, 0.8):
    ax.axhline(h, color="#eeeeee", lw=0.8, zorder=0)

# chance reference line + label pinned to the right edge outside the bars
ax.axhline(CHANCE, color="#7f7f7f", lw=1.4, ls="--", zorder=1)
ax.text(1.005, CHANCE, "chance = 0.5",
        transform=ax.get_yaxis_transform(),
        ha="left", va="center",
        fontsize=9, color="#7f7f7f", zorder=1)

# bars — highlight the cells whose 95% CI lies entirely above chance
bar_colors = [BAR_COLOR_CONFIRMED if lo > CHANCE else BAR_COLOR_MUTED
              for lo in top["boot_ci_lo"]]
bar_widths = [1.4 if lo > CHANCE else 0.5 for lo in top["boot_ci_lo"]]
bar_alphas = [0.90 if lo > CHANCE else 0.60 for lo in top["boot_ci_lo"]]
for i, (mean, color, ew, al) in enumerate(zip(top["boot_mean"], bar_colors,
                                              bar_widths, bar_alphas)):
    ax.bar(i, mean, width=0.66,
           color=color, edgecolor="black", linewidth=ew,
           alpha=al, zorder=2)

# mean value on top of each bar
for i, val in enumerate(top["boot_mean"]):
    ax.text(i, val + 0.012, f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="black")

# x-axis: cell name labels rotated for readability
ax.set_xticks(x_pos)
cell_labels = [format_cell(c) for c in top["cell"]]
ax.set_xticklabels(cell_labels, rotation=25, ha="right", fontsize=10)

# phase badges placed well below the cell-name labels so they don't overlap
for i, row in top.iterrows():
    ax.text(i, -0.24, row["phase"],
            transform=ax.get_xaxis_transform(),
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.35",
                      facecolor=PHASE_BADGE_COLOR[row["phase"]],
                      edgecolor="none"),
            clip_on=False)

# phase legend at the bottom — explains what the coloured badges mean
# lab's swab convention (source of truth for A/B/C/D labels):
#   A = pro-estrus  (predominantly nucleated epithelial cells)
#   B = estrus       (anucleated cornified cells, often in clusters)
#   C = metestrus   (mix of leukocytes + cornified + nucleated cells)
#   D = diestrus    (predominantly leukocytes)
PHASE_NAMES = {
    "A": "proestrus",
    "B": "estrus",
    "C": "metestrus",
    "D": "diestrus",
}
legend_y = -0.36
legend_start_x = 0.02
legend_gap = 0.24
ax.text(legend_start_x - 0.015, legend_y, "Estrous phase:",
        transform=ax.transAxes,
        ha="right", va="center",
        fontsize=10, fontweight="bold", color="#333")
for i, ph in enumerate(("A", "B", "C", "D")):
    x = legend_start_x + i * legend_gap
    ax.text(x, legend_y, ph,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.30",
                      facecolor=PHASE_BADGE_COLOR[ph],
                      edgecolor="none"),
            clip_on=False)
    ax.text(x + 0.022, legend_y, PHASE_NAMES[ph],
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=10, color="#333",
            clip_on=False)

# y-axis
ax.set_ylim(0.0, 0.9)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_ylabel("Balanced accuracy (bootstrap mean)", fontsize=11, labelpad=8)

ax.set_xlabel("")

# titles
fig.text(0.02, 0.97,
         f"Top {TOP_N} (phase × cell) combinations — SVM-RBF",
         ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.02, 0.935,
         "ranked by 95% CI lower bound (most confident above chance first)"
         "   ·   pooled Cable 1 + Cable 3   ·   1000 iterations   ·   "
         "red bar = CI excludes chance",
         ha="left", va="top", fontsize=10, color="#666", style="italic")

plt.subplots_adjust(left=0.07, right=0.94, top=0.90, bottom=0.36)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved figure:\n{OUT_PNG}")
