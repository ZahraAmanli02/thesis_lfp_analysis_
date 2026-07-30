# ============================================================
# 10B_BOOTSTRAP_FIGURES.PY
#
# Purpose:
#   Build defense-ready figures from the 10b bootstrap results
#   produced by 10b_bootstrap_full.py.
#   Three figures, SVM as the primary model, RF as robustness:
#     1. Forest plot per phase (4 panels)
#     2. Top-N bar chart (best cells overall)
#     3. SVM vs RF agreement scatter
#
# Input:
#   outputs/10b_bootstrap_full/10b_bootstrap_results_long.csv
#
# Output:
#   outputs/10b2_bootstrap_figures/
#       10b2_fig1_forest_per_phase.png
#       10b2_fig2_top10_bar.png
#       10b2_fig3_svm_vs_rf_scatter.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IN_CSV = os.path.join(
    OUTPUT_DIR, "10b_bootstrap_full", "10b_bootstrap_results_long.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, "10b2_bootstrap_figures")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG1 = os.path.join(OUT_DIR, "10b2_fig1_forest_per_phase.png")
OUT_FIG2 = os.path.join(OUT_DIR, "10b2_fig2_top10_bar.png")
OUT_FIG3 = os.path.join(OUT_DIR, "10b2_fig3_svm_vs_rf_scatter.png")

PRIMARY_MODEL = "svm_rbf"
SECONDARY_MODEL = "random_forest"

ESTROUS_PHASES = ["A", "B", "C", "D"]
# Rough labels — replace with confirmed mapping from lab's swab annotations
PHASE_LABEL = {
    "A": "Phase A (pro-estrus)",
    "B": "Phase B (estrus)",
    "C": "Phase C (metestrus)",
    "D": "Phase D (diestrus)",
}

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
RATIOS = [
    "theta_delta", "beta_delta", "low_gamma_delta",
    "high_gamma_delta", "fast_gamma_delta",
    "beta_theta", "low_gamma_theta", "high_gamma_theta", "fast_gamma_theta",
    "low_gamma_beta", "high_gamma_beta", "fast_gamma_beta",
    "high_gamma_low_gamma", "fast_gamma_low_gamma", "fast_gamma_high_gamma",
]

CHANCE = 0.5
COLOR_BAND = "#1f77b4"       # blue
COLOR_RATIO = "#ff7f0e"      # orange
COLOR_ABOVE_CHANCE = "#d62728"   # red, for CI excluding 0.5 above
COLOR_BELOW_CHANCE = "#2ca02c"   # green, for CI excluding 0.5 below (rare)
COLOR_STRADDLE = "#7f7f7f"       # grey, for CI including 0.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. LOAD
# ============================================================

if not os.path.exists(IN_CSV):
    raise FileNotFoundError(
        f"Missing bootstrap results:\n{IN_CSV}\n"
        "Run 10b2_bootstrap_full.py first."
    )

df = pd.read_csv(IN_CSV)
print(f"Loaded {len(df)} rows from {IN_CSV}")
print(f"  models: {sorted(df['model'].unique())}")
print(f"  phases: {sorted(df['phase'].unique())}")


def ci_status(lo, hi, chance=CHANCE):
    """Return 'above', 'below', or 'straddle' relative to chance."""
    if pd.isna(lo) or pd.isna(hi):
        return "na"
    if lo > chance:
        return "above"
    if hi < chance:
        return "below"
    return "straddle"


def color_for_status(status):
    return {
        "above": COLOR_ABOVE_CHANCE,
        "below": COLOR_BELOW_CHANCE,
        "straddle": COLOR_STRADDLE,
        "na": "#cccccc",
    }[status]


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


PHASE_BADGE_COLOR = {
    "A": "#e41a1c", "B": "#377eb8",
    "C": "#4daf4a", "D": "#984ea3",
}


# ============================================================
# 3. FIGURE 1 — FOREST PLOT PER PHASE (SVM only)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()

for ax, phase in zip(axes, ESTROUS_PHASES):
    sub = df[(df["phase"] == phase) & (df["model"] == PRIMARY_MODEL)].copy()
    sub = sub.sort_values("boot_mean", ascending=True).reset_index(drop=True)

    ymax = len(sub)
    for i, row in sub.iterrows():
        mean = row["boot_mean"]
        lo, hi = row["boot_ci_lo"], row["boot_ci_hi"]
        cell = row["cell"]
        cell_type = row.get("cell_type",
                            "band" if cell in BANDS else "ratio")
        status = ci_status(lo, hi)

        # tick colour tells cell type; marker colour tells CI status
        tick_color = COLOR_BAND if cell_type == "band" else COLOR_RATIO
        marker_color = color_for_status(status)

        if pd.isna(mean):
            ax.text(0.5, i, "no data (skipped)", ha="center", va="center",
                    color="#999", fontsize=8, style="italic")
            continue

        ax.hlines(i, lo, hi, color=marker_color, lw=2.2, zorder=2)
        ax.plot(mean, i, marker="o", color=marker_color,
                markersize=7, zorder=3,
                markeredgecolor="black", markeredgewidth=0.6)

        ax.text(-0.02, i, cell, ha="right", va="center",
                fontsize=8, color=tick_color,
                transform=ax.get_yaxis_transform())

    ax.axvline(CHANCE, color="grey", lw=1.2, ls="--", zorder=1)
    n_rec_first = int(sub["n_recordings"].iloc[0]) if len(sub) else 0
    n_mice_first = int(sub["n_mice"].iloc[0]) if len(sub) else 0
    ax.set_title(f"{PHASE_LABEL[phase]}   "
                 f"n_rec={n_rec_first}   n_mice={n_mice_first}",
                 fontweight="bold", loc="left")
    ax.set_yticks([])
    ax.set_ylim(-1, ymax)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Balanced accuracy   (mean + 95% CI)")

# legend outside the panels
legend_handles = [
    plt.Line2D([0], [0], color=COLOR_ABOVE_CHANCE, lw=2.5,
               marker="o", markersize=7, markeredgecolor="black",
               label="CI excludes 0.5 (above chance)"),
    plt.Line2D([0], [0], color=COLOR_STRADDLE, lw=2.5,
               marker="o", markersize=7, markeredgecolor="black",
               label="CI straddles 0.5 (no evidence)"),
    plt.Line2D([0], [0], color=COLOR_BELOW_CHANCE, lw=2.5,
               marker="o", markersize=7, markeredgecolor="black",
               label="CI excludes 0.5 (below chance)"),
    plt.Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COLOR_BAND, markersize=8,
               label="Band-cell label"),
    plt.Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COLOR_RATIO, markersize=8,
               label="Ratio-cell label"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)

fig.suptitle(
    f"10b bootstrap — diet classification by phase and feature cell\n"
    f"SVM-RBF, pooled Cable1 + Cable3, 1000 iterations, cluster-level",
    fontsize=13, fontweight="bold", y=0.995,
)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig(OUT_FIG1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure 1: {OUT_FIG1}")


# ============================================================
# 4. FIGURE 2 — TOP-N CELLS OVERALL (SVM)
# ============================================================

TOP_N = 10

svm_only = df[df["model"] == PRIMARY_MODEL].copy()
svm_only = svm_only.dropna(subset=["boot_mean"])
top = (svm_only.sort_values(["boot_mean", "boot_ci_lo"],
                            ascending=[False, False])
              .head(TOP_N)
              .reset_index(drop=True))

fig, ax = plt.subplots(figsize=(13, 7.5))
y_pos = np.arange(len(top))[::-1]

# alternating background stripes for readability
for i in range(TOP_N):
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color="#f6f6f6", zorder=0)

# subtle vertical gridlines at the "reference" tick values
for v in (0.5, 0.6, 0.7, 0.8, 0.9):
    ax.axvline(v, color="#e5e5e5", lw=0.8, zorder=0)

# chance reference line
ax.axvline(CHANCE, color="#7f7f7f", lw=1.4, ls="--", zorder=1)

for i, row in top.iterrows():
    y = y_pos[i]
    mean = row["boot_mean"]
    lo, hi = row["boot_ci_lo"], row["boot_ci_hi"]
    phase = row["phase"]
    cell = row["cell"]
    status = ci_status(lo, hi)

    # bar: highlight if CI clears chance, otherwise muted grey
    if status == "above":
        bar_color = "#c0392b"
        bar_alpha = 0.85
        edge_lw = 1.4
    else:
        bar_color = "#b0b0b0"
        bar_alpha = 0.55
        edge_lw = 0.6

    ax.barh(y, mean, height=0.62,
            color=bar_color, alpha=bar_alpha,
            edgecolor="black", linewidth=edge_lw, zorder=2)

    # CI whiskers (custom, capped)
    ax.plot([lo, hi], [y, y], color="black", lw=1.4, zorder=3)
    ax.plot([lo, lo], [y - 0.12, y + 0.12], color="black", lw=1.4, zorder=3)
    ax.plot([hi, hi], [y - 0.12, y + 0.12], color="black", lw=1.4, zorder=3)

    # phase badge (coloured circle with the letter) — far left column
    ax.text(-0.36, y, phase,
            transform=ax.get_yaxis_transform(),
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.40",
                      facecolor=PHASE_BADGE_COLOR[phase],
                      edgecolor="none"),
            zorder=5)

    # cell name (pretty-formatted for ratios) — right-aligned column
    ax.text(-0.03, y, format_cell(cell),
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            fontsize=11, color="#222", zorder=5)

    # numeric value at bar end + CI in grey  (with breathing room)
    val_x = min(mean + 0.015, 0.97)
    ax.text(val_x, y, f"{mean:.2f}",
            va="center", ha="left",
            fontsize=11, fontweight="bold", color="black", zorder=5)
    ax.text(val_x + 0.075, y, f"[{lo:.2f}, {hi:.2f}]",
            va="center", ha="left",
            fontsize=9, color="#666", zorder=5)

# chance label placed under the plot, next to the reference line
ax.text(CHANCE, -1.15, "chance = 0.5",
        ha="center", va="top", fontsize=9, color="#7f7f7f")

ax.set_yticks([])
ax.set_xlim(0.0, 1.02)
ax.set_ylim(-1.4, TOP_N - 0.3)
ax.set_xticks([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Balanced accuracy   (bootstrap mean · 95% CI)",
              fontsize=11, labelpad=8)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

# stacked title (main + subtitle) placed above the axes area
fig.text(0.02, 0.97,
         f"Top {TOP_N} (phase × cell) combinations — SVM-RBF",
         ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.02, 0.935,
         "pooled Cable 1 + Cable 3   ·   cluster bootstrap   ·   "
         "1000 iterations   ·   red bar = CI excludes 0.5",
         ha="left", va="top", fontsize=10, color="#666", style="italic")

plt.subplots_adjust(left=0.32, right=0.93, top=0.90, bottom=0.10)
plt.savefig(OUT_FIG2, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure 2: {OUT_FIG2}")


# ============================================================
# 5. FIGURE 3 — SVM vs RF AGREEMENT SCATTER
# ============================================================

svm_df = df[df["model"] == PRIMARY_MODEL][
    ["phase", "cell", "boot_mean", "boot_ci_lo", "boot_ci_hi"]
].rename(columns={"boot_mean": "svm_mean",
                  "boot_ci_lo": "svm_lo",
                  "boot_ci_hi": "svm_hi"})
rf_df = df[df["model"] == SECONDARY_MODEL][
    ["phase", "cell", "boot_mean", "boot_ci_lo", "boot_ci_hi"]
].rename(columns={"boot_mean": "rf_mean",
                  "boot_ci_lo": "rf_lo",
                  "boot_ci_hi": "rf_hi"})
paired = svm_df.merge(rf_df, on=["phase", "cell"], how="inner")
paired = paired.dropna(subset=["svm_mean", "rf_mean"])

phase_colors = {"A": "#e41a1c", "B": "#377eb8",
                "C": "#4daf4a", "D": "#984ea3"}

fig, ax = plt.subplots(figsize=(8, 8))

# diagonal (perfect agreement)
lo_ax, hi_ax = 0.15, 0.90
ax.plot([lo_ax, hi_ax], [lo_ax, hi_ax], color="black", lw=1, ls=":",
        zorder=1, label="perfect agreement")

# chance lines
ax.axvline(CHANCE, color="grey", lw=1, ls="--", zorder=1)
ax.axhline(CHANCE, color="grey", lw=1, ls="--", zorder=1)

for phase in ESTROUS_PHASES:
    sub = paired[paired["phase"] == phase]
    if len(sub) == 0:
        continue
    ax.scatter(sub["svm_mean"], sub["rf_mean"],
               color=phase_colors[phase], s=60, alpha=0.75,
               edgecolor="black", linewidth=0.6, zorder=3,
               label=PHASE_LABEL[phase])

# label the strongest cells (SVM mean top 5)
top5 = paired.sort_values("svm_mean", ascending=False).head(5)
for _, row in top5.iterrows():
    ax.annotate(f"{row['phase']}·{row['cell']}",
                (row["svm_mean"], row["rf_mean"]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=8, color="black",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# quadrant labels
ax.text(0.97, 0.03,
        "both below chance",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#666", style="italic")
ax.text(0.97, 0.97,
        "both above chance\n(robust findings)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#333", style="italic", fontweight="bold")

ax.set_xlim(lo_ax, hi_ax)
ax.set_ylim(lo_ax, hi_ax)
ax.set_aspect("equal")
ax.set_xlabel("SVM-RBF   bootstrap mean bal_acc")
ax.set_ylabel("Random Forest   bootstrap mean bal_acc")
ax.set_title("SVM vs Random Forest — agreement across (phase × cell) cells\n"
             "cells in the upper-right quadrant are supported by both models",
             fontweight="bold", loc="left")
ax.legend(loc="upper left", frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(OUT_FIG3, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure 3: {OUT_FIG3}")

print("\nDONE. Three figures saved in:")
print(f"  {OUT_DIR}")
