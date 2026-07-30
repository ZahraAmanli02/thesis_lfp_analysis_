# ============================================================
# 10B5_BOOTSTRAP_HEATMAP_CLEAN.PY
#
# Purpose:
#   Simplified re-draw of the bootstrap heatmap produced by
#   10b_bootstrap_full.py. Each cell shows ONLY the bootstrap
#   mean balanced accuracy (no CI text). This is the version
#   for PROFESSOR / thesis figures where the CI clutter isn't
#   needed.
#
# Design:
#   * two panels: SVM-RBF (top), Random Forest (bottom)
#   * cell value = boot_mean, rendered in a large, bold font
#   * diverging colormap centred on chance = 0.5
#   * vertical black line separates bands from ratios
#   * y-axis: phase letter + estrous name (lab's convention)
#
# Input:
#   outputs/10b_bootstrap_full/10b_bootstrap_results_long.csv
#
# Output:
#   outputs/10b5_bootstrap_heatmap_clean/
#       10b5_bootstrap_heatmap_clean.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RESULTS_CSV = os.path.join(
    OUTPUT_DIR, "10b_bootstrap_full", "10b_bootstrap_results_long.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, "10b5_bootstrap_heatmap_clean")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "10b5_bootstrap_heatmap_clean.png")

CHANCE = 0.5
V_MIN = 0.30
V_MAX = 0.85

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
RATIOS = [
    "theta_delta", "beta_delta", "low_gamma_delta",
    "high_gamma_delta", "fast_gamma_delta",
    "beta_theta", "low_gamma_theta", "high_gamma_theta", "fast_gamma_theta",
    "low_gamma_beta", "high_gamma_beta", "fast_gamma_beta",
    "high_gamma_low_gamma", "fast_gamma_low_gamma", "fast_gamma_high_gamma",
]
CELLS = BANDS + RATIOS
ESTROUS_PHASES = ["A", "B", "C", "D"]

PHASE_LABEL = {
    "A": "A · pro-estrus",
    "B": "B · estrus",
    "C": "C · metestrus",
    "D": "D · diestrus",
}

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


# ============================================================
# 2. LOAD & RESHAPE
# ============================================================

if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Missing results file:\n{RESULTS_CSV}")

df = pd.read_csv(RESULTS_CSV)


def pivot_mean(model):
    sub = df[df["model"] == model]
    if sub.empty:
        return pd.DataFrame(index=ESTROUS_PHASES, columns=CELLS, dtype=float)
    return (sub.pivot(index="phase", columns="cell", values="boot_mean")
               .reindex(index=ESTROUS_PHASES, columns=CELLS))


# ============================================================
# 3. DRAW
# ============================================================

def draw_heatmap(ax, tab, title):
    data = tab.to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=V_MIN, vcenter=CHANCE, vmax=V_MAX)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([format_cell(c) for c in CELLS],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(ESTROUS_PHASES)))
    ax.set_yticklabels([PHASE_LABEL[p] for p in ESTROUS_PHASES],
                       fontsize=10)
    ax.set_xlabel("Feature cell   (bands  |  band-to-band ratios)",
                  fontsize=10, labelpad=6)
    ax.set_ylabel("Estrous phase", fontsize=10)
    ax.set_title(title, fontweight="bold", loc="left",
                 fontsize=12, pad=6)

    max_dist = max(abs(V_MAX - CHANCE), abs(V_MIN - CHANCE))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center",
                        color="#666666", fontsize=10, fontweight="bold")
                continue
            dist = abs(v - CHANCE) / max_dist
            text_color = "white" if dist > 0.55 else "black"
            outline_color = "black" if text_color == "white" else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=13, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground=outline_color)])

    # divider between bands and ratios
    ax.axvline(len(BANDS) - 0.5, color="black", lw=2)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("bootstrap mean balanced accuracy")
    cbar.ax.axhline(CHANCE, color="grey", lw=1, ls="--")


fig, axes = plt.subplots(2, 1, figsize=(17, 8.5))
draw_heatmap(axes[0], pivot_mean("svm_rbf"),
             "A.  SVM-RBF   —   bootstrap mean balanced accuracy")
draw_heatmap(axes[1], pivot_mean("random_forest"),
             "B.  Random Forest   —   bootstrap mean balanced accuracy")

fig.suptitle(
    "10b bootstrap — diet classification, phase × cell   "
    "(pooled Cable 1 + Cable 3, 1000 iterations, cluster-level)",
    fontsize=13, fontweight="bold", y=0.995,
)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure:\n{OUT_PNG}")
