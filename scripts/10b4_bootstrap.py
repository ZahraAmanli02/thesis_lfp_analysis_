# ============================================================
# 10B4_BOOTSTRAP.PY
#
# Purpose:
#   pooled Cable 1 + Cable 3 bootstrap.
#   Four rows (phases A / B / C / D), one column (SVM-RBF only).
#   Each panel shows the bootstrap-mean balanced accuracy for
#   every feature cell (6 bands + 15 band-to-band ratios), coloured
#   blue for bands and green for ratios
#
# Design:
#   * primary model only (SVM-RBF)
#   * bar height = bootstrap mean of balanced accuracy
#   * dashed vertical line separates bands from ratios
#   * chance reference line at 0.5
#
# Input:
#   outputs/10b_bootstrap_full/10b_bootstrap_results_long.csv
#
# Output:
#   outputs/10b4_bootstrap/10b4_per_phase_svm.png
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

OUT_DIR = os.path.join(OUTPUT_DIR, "10b4_bootstrap")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "10b4_per_phase_svm.png")

PRIMARY_MODEL = "svm_rbf"
CHANCE = 0.5

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

# lab's swab convention:
PHASE_NAMES = {
    "A": "pro-estrus",
    "B": "estrus",
    "C": "metestrus",
    "D": "diestrus",
}
PHASE_BADGE_COLOR = {
    "A": "#e41a1c", "B": "#377eb8",
    "C": "#4daf4a", "D": "#984ea3",
}

COLOR_BAND = "#3b7ba9"
COLOR_RATIO = "#4faf4a"
COLOR_CONFIRMED = "#c0392b"       # 95% CI lies entirely above chance

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
    if cell in BANDS:
        return cell
    for b in sorted(BANDS, key=len, reverse=True):
        if cell.startswith(b + "_"):
            rest = cell[len(b) + 1:]
            if rest in BANDS:
                return f"{b} / {rest}"
    return cell


# ============================================================
# 3. LOAD DATA
# ============================================================

if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Missing results file:\n{RESULTS_CSV}")

df = pd.read_csv(RESULTS_CSV)
svm = df[df["model"] == PRIMARY_MODEL].copy()

# reindex into phase × cell tables of mean and CI-lower-bound
mean_tab = (svm.pivot(index="phase", columns="cell", values="boot_mean")
              .reindex(index=ESTROUS_PHASES, columns=CELLS))
ci_lo_tab = (svm.pivot(index="phase", columns="cell", values="boot_ci_lo")
               .reindex(index=ESTROUS_PHASES, columns=CELLS))

# n_recordings / n_mice per phase (same for every cell, so just take first)
per_phase = (svm.groupby("phase")[["n_recordings", "n_mice"]]
                .first()
                .reindex(ESTROUS_PHASES))


# ============================================================
# 4. PLOT — 4 rows x 1 col
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=False)

x_pos = np.arange(len(CELLS))
bar_colors = [COLOR_BAND if c in BANDS else COLOR_RATIO for c in CELLS]

for row_i, phase in enumerate(ESTROUS_PHASES):
    ax = axes[row_i]
    values = mean_tab.loc[phase].to_numpy(dtype=float)

    # subtle horizontal grid
    for h in (0.4, 0.5, 0.6, 0.7, 0.8):
        ax.axhline(h, color="#eeeeee", lw=0.6, zorder=0)

    # chance line
    ax.axhline(CHANCE, color="#7f7f7f", lw=1.3, ls="--", zorder=1)

    # bars — cells whose 95% CI is entirely above chance get COLOR_CONFIRMED
    # with a thicker outline; everything else keeps its band/ratio colour.
    valid_mask = ~np.isnan(values)
    lo_row = ci_lo_tab.loc[phase].to_numpy(dtype=float)
    for i in np.where(valid_mask)[0]:
        lo = lo_row[i]
        confirmed = (not np.isnan(lo)) and (lo > CHANCE)
        color = COLOR_CONFIRMED if confirmed else bar_colors[i]
        edge_lw = 1.4 if confirmed else 0.4
        alpha = 0.92 if confirmed else 0.85
        ax.bar(i, values[i], width=0.72,
               color=color, edgecolor="black", linewidth=edge_lw,
               alpha=alpha, zorder=2)

    # mean value on top of each valid bar
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        ax.text(i, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=8, color="black")

    # vertical divider between bands and ratios
    ax.axvline(len(BANDS) - 0.5, color="black", lw=1.2,
               ls=":", alpha=0.6, zorder=1)

    # y-axis
    ax.set_ylim(0.3, 0.95)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel("Balanced accuracy", fontsize=10)

    # panel title (phase letter + name + counts)
    n_rec = int(per_phase.loc[phase, "n_recordings"])
    n_mice = int(per_phase.loc[phase, "n_mice"])
    title = (f"Phase {phase} — {PHASE_NAMES[phase]}"
             f"    ·    n = {n_rec} recordings / {n_mice} mice")
    ax.set_title(title, fontweight="bold", loc="left", fontsize=11, pad=6)

    # x-axis: cell names on EVERY panel
    ax.set_xticks(x_pos)
    ax.set_xticklabels([format_cell(c) for c in CELLS],
                       rotation=45, ha="right", fontsize=8)

# x-axis label only on the bottom panel
axes[-1].set_xlabel("Feature cell   (bands  |  band-to-band ratios)",
                    fontsize=10, labelpad=8)

# figure title
fig.suptitle(
    "Per-phase overview — SVM-RBF   ·   pooled Cable 1 + Cable 3   ·   "
    "bootstrap 1000 iterations   ·   red bar = 95% CI excludes chance",
    fontsize=13, fontweight="bold", y=0.995,
)

# ============================================================
# 5. LEGEND (bottom) — band/ratio colour + estrous phase mapping
# ============================================================

# subplots_adjust to leave enough room below the rotated x-axis labels
# for the two legend rows
plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.15, hspace=0.85)

# ---- legend row 1: feature-colour (band vs ratio) ----
row1_y = 0.055
fig.text(0.06, row1_y, "Feature colour:",
         fontsize=10, fontweight="bold", color="#333", va="center")
fig.patches.append(plt.Rectangle((0.155, row1_y - 0.006), 0.017, 0.014,
                                  transform=fig.transFigure,
                                  facecolor=COLOR_BAND, edgecolor="black",
                                  linewidth=0.4))
fig.text(0.178, row1_y, "frequency band",
         fontsize=10, color="#333", va="center")
fig.patches.append(plt.Rectangle((0.30, row1_y - 0.006), 0.017, 0.014,
                                  transform=fig.transFigure,
                                  facecolor=COLOR_RATIO, edgecolor="black",
                                  linewidth=0.4))
fig.text(0.323, row1_y, "band-to-band ratio",
         fontsize=10, color="#333", va="center")
fig.patches.append(plt.Rectangle((0.46, row1_y - 0.006), 0.017, 0.014,
                                  transform=fig.transFigure,
                                  facecolor=COLOR_CONFIRMED, edgecolor="black",
                                  linewidth=1.2))
fig.text(0.483, row1_y, "95% CI excludes chance",
         fontsize=10, color="#333", va="center")

# ---- legend row 2: estrous-phase mapping ----
row2_y = 0.020
fig.text(0.06, row2_y, "Estrous phase:",
         fontsize=10, fontweight="bold", color="#333", va="center")
start_x = 0.155
gap = 0.11
for i, ph in enumerate(("A", "B", "C", "D")):
    x = start_x + i * gap
    fig.text(x, row2_y, ph,
             ha="center", va="center",
             fontsize=10, fontweight="bold", color="white",
             bbox=dict(boxstyle="circle,pad=0.28",
                       facecolor=PHASE_BADGE_COLOR[ph],
                       edgecolor="none"))
    fig.text(x + 0.017, row2_y, PHASE_NAMES[ph],
             ha="left", va="center", fontsize=10, color="#333")

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure:\n{OUT_PNG}")
