# ============================================================
# 11C3_BOOTSTRAP_WEIGHT_PER_PHASE.PY
#
# Purpose:
#   Per-phase overview panel for RQ2. Four rows (phases A–D),
#   two columns (HFD | CTRL), showing bootstrap-mean R² for
#   every feature cell. Same visual language as 10b4:
#   blue = frequency band, green = band-to-band ratio, chance
#   reference at R² = 0.
#
#   The HFD-vs-CTRL side-by-side layout is the negative-control
#   test: HFD should show non-trivial R² in cells where LFP
#   tracks weight change; CTRL should sit around zero across
#   the board (no weight change to predict).
#
# Input:
#   outputs/11c1_bootstrap_weight_full/
#       11c1_bootstrap_weight_results_long.csv
#
# Output:
#   outputs/11c3_bootstrap_weight_per_phase/
#       11c3_per_phase_hfd_vs_ctrl.png
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
    OUTPUT_DIR, "11c1_bootstrap_weight_full",
    "11c1_bootstrap_weight_results_long.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, "11c3_bootstrap_weight_per_phase")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "11c3_per_phase_hfd_vs_ctrl.png")

CHANCE = 0.0

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
SUBSET_COLUMNS = ("HFD", "CTRL")

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
# 3. LOAD & RESHAPE
# ============================================================

if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"Missing results file:\n{RESULTS_CSV}")

df = pd.read_csv(RESULTS_CSV)

# mean R² tables per subset:  phase x cell
mean_tabs = {}
per_phase_counts = {}
for subset in SUBSET_COLUMNS:
    s = df[df["subset"] == subset]
    mean_tabs[subset] = (s.pivot(index="phase", columns="cell",
                                 values="boot_mean")
                          .reindex(index=ESTROUS_PHASES, columns=CELLS))
    per_phase_counts[subset] = (s.groupby("phase")[["n_recordings", "n_mice"]]
                                  .first()
                                  .reindex(ESTROUS_PHASES))

# shared y-axis limits (data-driven, symmetrical around 0 if negatives exist)
all_vals = pd.concat([mean_tabs[s].stack() for s in SUBSET_COLUMNS])
y_lo = min(0.0, float(all_vals.min()))
y_hi = max(0.0, float(all_vals.max()))
y_pad = max(0.08, (y_hi - y_lo) * 0.15)
Y_LO = y_lo - y_pad
Y_HI = y_hi + y_pad


# ============================================================
# 4. PLOT — 4 rows x 2 cols (HFD | CTRL)
# ============================================================

fig, axes = plt.subplots(4, 2, figsize=(22, 14), sharex=False)

x_pos = np.arange(len(CELLS))
bar_colors = [COLOR_BAND if c in BANDS else COLOR_RATIO for c in CELLS]

for row_i, phase in enumerate(ESTROUS_PHASES):
    for col_i, subset in enumerate(SUBSET_COLUMNS):
        ax = axes[row_i, col_i]
        values = mean_tabs[subset].loc[phase].to_numpy(dtype=float)

        for h in np.arange(round(Y_LO, 1), round(Y_HI, 1) + 0.01, 0.1):
            if abs(h - CHANCE) > 1e-9:
                ax.axhline(h, color="#eeeeee", lw=0.6, zorder=0)

        ax.axhline(CHANCE, color="#7f7f7f", lw=1.3, ls="--", zorder=1)

        valid = ~np.isnan(values)
        ax.bar(x_pos[valid], values[valid], width=0.72,
               color=[bar_colors[i] for i in np.where(valid)[0]],
               edgecolor="black", linewidth=0.4,
               alpha=0.85, zorder=2)

        for i, v in enumerate(values):
            if np.isnan(v):
                continue
            dy = 0.008 if v >= 0 else -0.008
            va = "bottom" if v >= 0 else "top"
            ax.text(i, v + dy, f"{v:+.2f}",
                    ha="center", va=va,
                    fontsize=7, color="black")

        ax.axvline(len(BANDS) - 0.5, color="black", lw=1.2,
                   ls=":", alpha=0.6, zorder=1)

        ax.set_ylim(Y_LO, Y_HI)
        ax.set_ylabel(r"$R^2$", fontsize=10)

        n_rec = per_phase_counts[subset].loc[phase, "n_recordings"]
        n_mice = per_phase_counts[subset].loc[phase, "n_mice"]
        if pd.isna(n_rec):
            title = f"{subset}  |  Phase {phase} — {PHASE_NAMES[phase]}   (no data)"
        else:
            title = (f"{subset}  |  Phase {phase} — {PHASE_NAMES[phase]}"
                     f"    ·    n = {int(n_rec)} rec / {int(n_mice)} mice")
        ax.set_title(title, fontweight="bold", loc="left",
                     fontsize=10.5, pad=6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([format_cell(c) for c in CELLS],
                           rotation=45, ha="right", fontsize=7.5)

for col_i in range(2):
    axes[-1, col_i].set_xlabel(
        "Feature cell   (bands  |  band-to-band ratios)",
        fontsize=10, labelpad=8,
    )

fig.suptitle(
    "Per-phase overview — body-weight regression (RQ2)   ·   "
    "HFD vs CTRL   ·   pooled Cable 1 + Cable 3   ·   bootstrap 1000 iterations",
    fontsize=13, fontweight="bold", y=0.995,
)

# ============================================================
# 5. LEGEND (bottom)
# ============================================================

plt.subplots_adjust(left=0.05, right=0.99, top=0.94,
                    bottom=0.13, hspace=0.85, wspace=0.14)

row1_y = 0.055
fig.text(0.05, row1_y, "Feature colour:",
         fontsize=10, fontweight="bold", color="#333", va="center")
fig.patches.append(plt.Rectangle((0.14, row1_y - 0.006), 0.014, 0.012,
                                  transform=fig.transFigure,
                                  facecolor=COLOR_BAND,
                                  edgecolor="black", linewidth=0.4))
fig.text(0.158, row1_y, "frequency band",
         fontsize=10, color="#333", va="center")
fig.patches.append(plt.Rectangle((0.25, row1_y - 0.006), 0.014, 0.012,
                                  transform=fig.transFigure,
                                  facecolor=COLOR_RATIO,
                                  edgecolor="black", linewidth=0.4))
fig.text(0.268, row1_y, "band-to-band ratio",
         fontsize=10, color="#333", va="center")

row2_y = 0.020
fig.text(0.05, row2_y, "Estrous phase:",
         fontsize=10, fontweight="bold", color="#333", va="center")
start_x = 0.14
gap = 0.10
for i, ph in enumerate(("A", "B", "C", "D")):
    x = start_x + i * gap
    fig.text(x, row2_y, ph,
             ha="center", va="center",
             fontsize=10, fontweight="bold", color="white",
             bbox=dict(boxstyle="circle,pad=0.28",
                       facecolor=PHASE_BADGE_COLOR[ph],
                       edgecolor="none"))
    fig.text(x + 0.014, row2_y, PHASE_NAMES[ph],
             ha="left", va="center", fontsize=10, color="#333")

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure:\n{OUT_PNG}")
