# ============================================================
# 11C2_BOOTSTRAP_WEIGHT_CHANGE_TOP10.PY
#
# Purpose:
#   Top-10 vertical bar chart of the HFD-only weight-CHANGE
#   regression bootstrap (reads 11c1_bootstrap_weight_change
#   results). Same clean visual style as 10b3 / 11c2 (absolute).
#
# Input:
#   outputs/11c1_bootstrap_weight_change_full/
#       11c1_bootstrap_weight_change_results_long.csv
#
# Output:
#   outputs/11c2_bootstrap_weight_change_top10/
#       11c2_weight_change_top10_bar.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RESULTS_CSV = os.path.join(
    OUTPUT_DIR, "11c1_bootstrap_weight_change_full",
    "11c1_bootstrap_weight_change_results_long.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, "11c2_bootstrap_weight_change_top10")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "11c2_weight_change_top10_bar.png")

SUBSET_TO_SHOW = "HFD"
TOP_N = 10
CHANCE = 0.0
BAR_COLOR = "#4a7ba6"

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

PHASE_NAMES = {"A": "pro-estrus", "B": "estrus",
               "C": "metestrus", "D": "diestrus"}
PHASE_BADGE_COLOR = {"A": "#e41a1c", "B": "#377eb8",
                     "C": "#4daf4a", "D": "#984ea3"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def format_cell(cell):
    if cell in BANDS:
        return cell
    for b in sorted(BANDS, key=len, reverse=True):
        if cell.startswith(b + "_"):
            rest = cell[len(b) + 1:]
            if rest in BANDS:
                return f"{b} / {rest}"
    return cell


if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Missing results file:\n{RESULTS_CSV}")

df = pd.read_csv(RESULTS_CSV)
sub = (df[df["subset"] == SUBSET_TO_SHOW]
         .dropna(subset=["boot_mean"])
         .copy())
top = (sub.sort_values("boot_mean", ascending=False)
           .head(TOP_N)
           .reset_index(drop=True))

print(f"Top {TOP_N} cells by mean bootstrap R² ({SUBSET_TO_SHOW} · weight_delta):")
for i, row in top.iterrows():
    print(f"  {i+1:2d}. phase {row['phase']}  "
          f"{format_cell(row['cell']):<24s}  "
          f"R²={row['boot_mean']:+.3f}")


fig, ax = plt.subplots(figsize=(14, 7))
x_pos = np.arange(len(top))

for h in (-0.4, -0.2, 0.2, 0.4, 0.6):
    ax.axhline(h, color="#eeeeee", lw=0.8, zorder=0)

ax.axhline(CHANCE, color="#7f7f7f", lw=1.4, ls="--", zorder=1)
ax.text(1.005, CHANCE, "chance = 0",
        transform=ax.get_yaxis_transform(),
        ha="left", va="center",
        fontsize=9, color="#7f7f7f", zorder=1)

ax.bar(x_pos, top["boot_mean"], width=0.66,
       color=BAR_COLOR, edgecolor="black", linewidth=0.5,
       alpha=0.85, zorder=2)

for i, val in enumerate(top["boot_mean"]):
    dy = 0.015 if val >= 0 else -0.015
    va = "bottom" if val >= 0 else "top"
    ax.text(i, val + dy, f"{val:+.2f}",
            ha="center", va=va,
            fontsize=11, fontweight="bold", color="black")

ax.set_xticks(x_pos)
cell_labels = [format_cell(c) for c in top["cell"]]
ax.set_xticklabels(cell_labels, rotation=25, ha="right", fontsize=10)

for i, row in top.iterrows():
    ax.text(i, -0.24, row["phase"],
            transform=ax.get_xaxis_transform(),
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.35",
                      facecolor=PHASE_BADGE_COLOR[row["phase"]],
                      edgecolor="none"),
            clip_on=False)

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

lo_data = min(0.0, float(top["boot_mean"].min()))
hi_data = max(0.0, float(top["boot_mean"].max()))
pad = max(0.05, (hi_data - lo_data) * 0.20)
ax.set_ylim(lo_data - pad, hi_data + pad)
ax.set_ylabel(r"$R^2$   (bootstrap mean)", fontsize=11, labelpad=8)
ax.set_xlabel("")

fig.text(0.02, 0.97,
         f"Top {TOP_N} (phase × cell) — HFD weight-CHANGE regression",
         ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.02, 0.935,
         "target = body_weight - baseline_of_that_mouse   ·   "
         "pooled Cable 1 + Cable 3   ·   cluster bootstrap   ·   "
         "1000 iterations",
         ha="left", va="top", fontsize=10, color="#666", style="italic")

plt.subplots_adjust(left=0.07, right=0.94, top=0.90, bottom=0.36)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved figure:\n{OUT_PNG}")
