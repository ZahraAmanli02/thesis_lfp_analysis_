# ============================================================
# 11_DEFENSE_FIGURES.PY
#
# Defense-ready figures + automatic interpretation for RQ1/RQ2.
#
# Produces per cable:
#   BAR CHARTS (primary — professor prefers these):
#     11_barchart_svm_classification_<CABLE>.png
#     11_barchart_rf_classification_<CABLE>.png
#     11_barchart_rf_regression_<CABLE>.png
#     11_perphase_overview_<CABLE>.png            (all 3 models, 4x3 grid)
#
#   HEATMAPS (supplementary — with a diverging colormap centred on
#   the chance line, so "above chance" jumps out visually):
#     11_heatmap_svm_classification_<CABLE>.png
#     11_heatmap_rf_classification_<CABLE>.png
#     11_heatmap_rf_regression_<CABLE>.png
#
#   AUTOMATIC INTERPRETATION:
#     11_INTERPRETATION_<CABLE>.txt
#       - Top-5 cells per model
#       - Which phases carry the strongest signal
#       - Whether bands or ratios dominate
#       - Whether classification / regression / both worked
#       - Checklist: which of professor's requirements are satisfied
#
# Input:
#   outputs/10b_classify_group_<CABLE>/10b_results_long_<CABLE>.csv
#   outputs/10c_regress_weight_<CABLE>/10c_results_long_<CABLE>.csv
#
# Output:
#   outputs/11_defense_figures_<CABLE>/
#
# Usage:
#   Set CABLE = "Cable1" or "Cable3", run twice.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

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

# Colors chosen for slide legibility.
COLOR_BAND = "#2C6E9B"          # deep blue for bands
COLOR_RATIO = "#4C8C4A"          # deep green for ratios
COLOR_CHANCE = "#8B8B8B"         # grey dashed chance line
COLOR_SIG = "#C0392B"            # red for significance stars

# Diverging colormap for heatmaps — much easier to read than viridis:
#   blue = below chance  |  white ≈ chance  |  red = above chance
CMAP_DIV = "RdBu_r"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CSV_10B = os.path.join(
    OUTPUT_DIR, f"10b_classify_group_{CABLE}", f"10b_results_long_{CABLE}.csv"
)
CSV_10C = os.path.join(
    OUTPUT_DIR, f"10c_regress_weight_{CABLE}", f"10c_results_long_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"11_defense_figures_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "figure.facecolor": "white",
})


# ============================================================
# 2. HELPERS
# ============================================================

def sig_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def per_phase_table(df, model, metric):
    tables = {}
    for phase in ESTROUS_PHASES:
        sub = df[(df["phase"] == phase)
                 & (df["model"] == model)
                 & (df["metric"] == metric)]
        tables[phase] = (sub[["cell", "value"]].dropna()
                         if len(sub) else pd.DataFrame(columns=["cell", "value"]))
    return tables


def per_phase_p_table(df, model, p_metric):
    tables = {}
    for phase in ESTROUS_PHASES:
        sub = df[(df["phase"] == phase)
                 & (df["model"] == model)
                 & (df["metric"] == p_metric)]
        tables[phase] = dict(zip(sub["cell"], sub["value"])) if len(sub) else {}
    return tables


def pivot(df, model, metric):
    sub = df[(df["model"] == model) & (df["metric"] == metric)]
    if sub.empty:
        return pd.DataFrame(index=ESTROUS_PHASES, columns=CELLS, dtype=float)
    return (sub.pivot(index="phase", columns="cell", values="value")
              .reindex(index=ESTROUS_PHASES, columns=CELLS))


# ============================================================
# 3. BAR-CHART BUILDERS  (professor's preferred style)
# ============================================================

def draw_one_phase_bar(ax, cells_ordered, values_by_cell, p_by_cell,
                       chance, metric_label, title,
                       vmin=None, vmax=None):
    xs = np.arange(len(cells_ordered))
    heights = np.array([values_by_cell.get(c, np.nan) for c in cells_ordered],
                       dtype=float)
    colors = [COLOR_BAND if c in BANDS else COLOR_RATIO for c in cells_ordered]

    ax.bar(xs, heights, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(chance, color=COLOR_CHANCE, linestyle="--", linewidth=1.5,
               zorder=0, label=f"chance = {chance}")

    for x, c, h in zip(xs, cells_ordered, heights):
        if np.isnan(h):
            continue
        star = sig_stars(p_by_cell.get(c, np.nan))
        if star:
            offset = (0.02 if h >= chance else -0.02) * max(abs(vmax or 1), abs(vmin or 0.5))
            va = "bottom" if h >= chance else "top"
            ax.text(x, h + offset, star, ha="center", va=va,
                    color=COLOR_SIG, fontsize=11, fontweight="bold")

    ax.axvline(len(BANDS) - 0.5, color="grey", linestyle=":", linewidth=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(cells_ordered, rotation=55, ha="right", fontsize=9)
    ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if vmin is not None or vmax is not None:
        ax.set_ylim(vmin, vmax)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def build_barchart_figure(values_tables, p_tables, chance, metric_label,
                          title_prefix, out_path,
                          vmin=None, vmax=None, cable=""):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.ravel()
    for i, phase in enumerate(ESTROUS_PHASES):
        vals_df = values_tables[phase]
        vals_by_cell = dict(zip(vals_df["cell"], vals_df["value"])) if len(vals_df) else {}
        p_by_cell = p_tables.get(phase, {})
        n_cells = sum(1 for c in CELLS if c in vals_by_cell)
        draw_one_phase_bar(
            axes[i], CELLS, vals_by_cell, p_by_cell,
            chance, metric_label,
            f"Phase {phase}  ({n_cells}/{len(CELLS)} cells modelled)",
            vmin=vmin, vmax=vmax
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BAND, label="Frequency band"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_RATIO, label="Band-to-band ratio"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{title_prefix}   |   {cable}",
                 fontsize=15, fontweight="bold", y=1.00)
    fig.text(0.5, -0.05,
             "* p < 0.05    ** p < 0.01    *** p < 0.001    "
             "(permutation, N=200)",
             ha="center", fontsize=10, style="italic", color="#444")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ============================================================
# 4. HEATMAP BUILDER  (diverging colormap centred on chance)
# ============================================================

def build_heatmap_figure(values_table, p_table, chance,
                         vmin, vmax, metric_label,
                         title_prefix, out_path, cable=""):
    """
    Diverging RdBu_r heatmap:
      blue = below chance
      white = at chance
      red = above chance
    Significance stars printed inside cells.
    """
    data = values_table.to_numpy(dtype=float)
    p_data = p_table.to_numpy(dtype=float) if p_table is not None else None

    fig, ax = plt.subplots(1, 1, figsize=(15, 5.2))

    # TwoSlopeNorm makes 'chance' the exact centre of the colormap.
    norm = TwoSlopeNorm(vmin=vmin, vcenter=chance, vmax=vmax)
    im = ax.imshow(data, aspect="auto", cmap=CMAP_DIV, norm=norm)

    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(CELLS, rotation=55, ha="right", fontsize=10)
    ax.set_yticks(range(len(ESTROUS_PHASES)))
    ax.set_yticklabels(ESTROUS_PHASES, fontsize=13, fontweight="bold")
    ax.set_ylabel("Estrous phase", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature cell  (6 bands | 15 ratios)",
                  fontsize=12, fontweight="bold")
    ax.set_title(f"{title_prefix}   |   {cable}",
                 fontsize=14, fontweight="bold", loc="left", pad=12)

    # Adaptive text with outline (path effect). Numbers remain
    # readable on any background (dark red, dark blue, white).
    import matplotlib.patheffects as pe
    max_dist = max(abs(vmax - chance), abs(vmin - chance))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            star = "" if p_data is None else sig_stars(p_data[i, j])
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center",
                        color="#888", fontsize=9, fontweight="bold")
                continue
            dist = abs(v - chance) / max_dist if max_dist else 0
            text_color = "white" if dist > 0.55 else "black"
            outline_color = "black" if text_color == "white" else "white"
            label = f"{v:.2f}"
            if star:
                label = f"{v:.2f}\n{star}"
            ax.text(j, i, label, ha="center", va="center",
                    color=text_color, fontsize=10, linespacing=0.9,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground=outline_color)])

    ax.axvline(len(BANDS) - 0.5, color="black", lw=2.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric_label, fontsize=11)
    cbar.ax.axhline(chance, color="black", lw=1.5, ls="--")

    fig.text(0.5, -0.03,
             "White ≈ chance   |   Red = above chance   |   Blue = below chance",
             ha="center", fontsize=10, style="italic", color="#444")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ============================================================
# 5. LOAD
# ============================================================

print(f"\n{'=' * 70}")
print(f"11 DEFENSE FIGURES + INTERPRETATION — {CABLE}")
print("=" * 70)

if not os.path.exists(CSV_10B):
    raise FileNotFoundError(f"Missing 10b results:\n{CSV_10B}")
if not os.path.exists(CSV_10C):
    raise FileNotFoundError(f"Missing 10c results:\n{CSV_10C}")

df_b = pd.read_csv(CSV_10B)
df_c = pd.read_csv(CSV_10C)
print(f"Loaded 10b rows: {len(df_b)}   |   10c rows: {len(df_c)}")


# ============================================================
# 6. BAR CHARTS
# ============================================================

svm_vals = per_phase_table(df_b, "svm_rbf", "balanced_accuracy")
svm_ps = per_phase_p_table(df_b, "svm_rbf", "perm_p_value_balanced_accuracy")
rfc_vals = per_phase_table(df_b, "random_forest", "balanced_accuracy")
rfc_ps = {p: {} for p in ESTROUS_PHASES}
reg_vals = per_phase_table(df_c, "random_forest", "r2")
reg_ps = per_phase_p_table(df_c, "random_forest", "perm_p_value_r2")

build_barchart_figure(
    svm_vals, svm_ps, chance=0.5, metric_label="Balanced accuracy",
    title_prefix="RQ1 — SVM-RBF (diet classification)",
    out_path=os.path.join(OUT_DIR, f"11_barchart_svm_classification_{CABLE}.png"),
    vmin=0.3, vmax=1.0, cable=CABLE,
)
build_barchart_figure(
    rfc_vals, rfc_ps, chance=0.5, metric_label="Balanced accuracy",
    title_prefix="RQ1 — Random Forest (diet classification)",
    out_path=os.path.join(OUT_DIR, f"11_barchart_rf_classification_{CABLE}.png"),
    vmin=0.3, vmax=1.0, cable=CABLE,
)
build_barchart_figure(
    reg_vals, reg_ps, chance=0.0, metric_label=r"R$^2$",
    title_prefix="RQ2 — Random Forest (body-weight regression)",
    out_path=os.path.join(OUT_DIR, f"11_barchart_rf_regression_{CABLE}.png"),
    vmin=-0.8, vmax=1.0, cable=CABLE,
)


# ============================================================
# 7. COMPACT PER-PHASE OVERVIEW (4 rows x 3 cols)
# ============================================================

fig, axes = plt.subplots(4, 3, figsize=(20, 15))
col_defs = [
    ("SVM — diet (bal. acc.)",   svm_vals, svm_ps, 0.5, "Balanced accuracy", 0.3, 1.0),
    ("RF — diet (bal. acc.)",    rfc_vals, rfc_ps, 0.5, "Balanced accuracy", 0.3, 1.0),
    ("RF — weight (R²)",         reg_vals, reg_ps, 0.0, r"R$^2$",             -0.8, 1.0),
]
for row, phase in enumerate(ESTROUS_PHASES):
    for col, (title, vals_tab, p_tab, chance, ylabel, vmin, vmax) in enumerate(col_defs):
        vals_df = vals_tab[phase]
        vals_by_cell = dict(zip(vals_df["cell"], vals_df["value"])) if len(vals_df) else {}
        p_by_cell = p_tab.get(phase, {})
        draw_one_phase_bar(
            axes[row, col], CELLS, vals_by_cell, p_by_cell,
            chance, ylabel, f"Phase {phase}  |  {title}",
            vmin=vmin, vmax=vmax
        )
        if row < len(ESTROUS_PHASES) - 1:
            axes[row, col].set_xticklabels([])
            axes[row, col].set_xlabel("")
fig.suptitle(f"Per-phase per-model overview   |   {CABLE}",
             fontsize=16, fontweight="bold", y=1.00)
handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR_BAND, label="Frequency band"),
    plt.Rectangle((0, 0), 1, 1, color=COLOR_RATIO, label="Band-to-band ratio"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout()
out4 = os.path.join(OUT_DIR, f"11_perphase_overview_{CABLE}.png")
plt.savefig(out4, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved: {out4}")


# ============================================================
# 8. HEATMAPS (with the FIXED diverging colormap)
# ============================================================

build_heatmap_figure(
    pivot(df_b, "svm_rbf", "balanced_accuracy"),
    pivot(df_b, "svm_rbf", "perm_p_value_balanced_accuracy"),
    chance=0.5, vmin=0.3, vmax=1.0,
    metric_label="Balanced accuracy",
    title_prefix="RQ1 — SVM-RBF (diet classification)",
    out_path=os.path.join(OUT_DIR, f"11_heatmap_svm_classification_{CABLE}.png"),
    cable=CABLE,
)
build_heatmap_figure(
    pivot(df_b, "random_forest", "balanced_accuracy"),
    None,
    chance=0.5, vmin=0.3, vmax=1.0,
    metric_label="Balanced accuracy",
    title_prefix="RQ1 — Random Forest (diet classification)",
    out_path=os.path.join(OUT_DIR, f"11_heatmap_rf_classification_{CABLE}.png"),
    cable=CABLE,
)
build_heatmap_figure(
    pivot(df_c, "random_forest", "r2"),
    pivot(df_c, "random_forest", "perm_p_value_r2"),
    chance=0.0, vmin=-0.8, vmax=1.0,
    metric_label=r"R$^2$",
    title_prefix="RQ2 — Random Forest (body-weight regression)",
    out_path=os.path.join(OUT_DIR, f"11_heatmap_rf_regression_{CABLE}.png"),
    cable=CABLE,
)


# ============================================================
# 9. AUTOMATIC INTERPRETATION
# ------------------------------------------------------------
# This part reads the results and writes a plain-language summary
# so you can drop it into your defense talk / thesis discussion.
# ============================================================

def top_cells(df, model, metric, ascending=False, n=5):
    sub = df[(df["model"] == model) & (df["metric"] == metric)]
    if sub.empty:
        return pd.DataFrame(columns=["phase", "cell", "value"])
    return (sub[["phase", "cell", "value"]].dropna()
              .sort_values("value", ascending=ascending).head(n)
              .reset_index(drop=True))


def sig_count(df, model, p_metric, alpha=0.05):
    """Return (n_significant, n_total) for one (model, p_metric)."""
    sub = df[(df["model"] == model) & (df["metric"] == p_metric)].dropna(subset=["value"])
    n_total = len(sub)
    n_sig = int((sub["value"] < alpha).sum())
    return n_sig, n_total


def phase_summary(df, model, metric, chance):
    """Return per-phase counts of cells above / at / below chance."""
    out = {}
    for phase in ESTROUS_PHASES:
        sub = df[(df["phase"] == phase)
                 & (df["model"] == model)
                 & (df["metric"] == metric)].dropna(subset=["value"])
        if sub.empty:
            out[phase] = {"n": 0, "above": 0, "mean": np.nan, "best_cell": None, "best_val": np.nan}
            continue
        best_idx = sub["value"].idxmax()
        out[phase] = {
            "n": len(sub),
            "above": int((sub["value"] > chance).sum()),
            "mean": float(sub["value"].mean()),
            "best_cell": sub.loc[best_idx, "cell"],
            "best_val": float(sub.loc[best_idx, "value"]),
        }
    return out


def band_vs_ratio(df, model, metric, chance):
    """Which cell type carries a stronger signal on average?"""
    sub = df[(df["model"] == model) & (df["metric"] == metric)].dropna(subset=["value"])
    if sub.empty:
        return None
    band_vals = sub[sub["cell_type"] == "band"]["value"]
    ratio_vals = sub[sub["cell_type"] == "ratio"]["value"]
    return {
        "band_mean": float(band_vals.mean()) if len(band_vals) else np.nan,
        "band_above": int((band_vals > chance).sum()),
        "band_n": len(band_vals),
        "ratio_mean": float(ratio_vals.mean()) if len(ratio_vals) else np.nan,
        "ratio_above": int((ratio_vals > chance).sum()),
        "ratio_n": len(ratio_vals),
    }


top_svm = top_cells(df_b, "svm_rbf", "balanced_accuracy")
top_rfc = top_cells(df_b, "random_forest", "balanced_accuracy")
top_reg = top_cells(df_c, "random_forest", "r2")

svm_sig_n, svm_sig_total = sig_count(df_b, "svm_rbf", "perm_p_value_balanced_accuracy")
reg_sig_n, reg_sig_total = sig_count(df_c, "random_forest", "perm_p_value_r2")

svm_phase_summ = phase_summary(df_b, "svm_rbf", "balanced_accuracy", 0.5)
rfc_phase_summ = phase_summary(df_b, "random_forest", "balanced_accuracy", 0.5)
reg_phase_summ = phase_summary(df_c, "random_forest", "r2", 0.0)

svm_bvr = band_vs_ratio(df_b, "svm_rbf", "balanced_accuracy", 0.5)
reg_bvr = band_vs_ratio(df_c, "random_forest", "r2", 0.0)


# ---------- build the text file ----------
L = []
sep = "=" * 78
sub = "-" * 78

L.append(sep)
L.append(f"11 INTERPRETATION — {CABLE}")
L.append(sep)
L.append("")
L.append("This file was generated automatically from the 10b and 10c results.")
L.append("Use it as a starting point for the Results / Discussion in your")
L.append("thesis and to anchor the numbers in your defense talk.")
L.append("")

# ---------- RQ1 ----------
L.append(sub)
L.append("RQ1 — DIET CLASSIFICATION (HF vs CTRL)")
L.append(sub)
L.append(f"Significant cells (SVM, p<0.05, permutation): "
         f"{svm_sig_n} / {svm_sig_total}")
L.append("")
L.append("Top 5 cells (SVM-RBF, balanced accuracy):")
for _, r in top_svm.iterrows():
    delta = r["value"] - 0.5
    L.append(f"  phase {r['phase']}  |  {r['cell']:<22s}  bal_acc = {r['value']:.3f}   (+{delta:.3f} vs chance)")
L.append("")
L.append("Top 5 cells (Random Forest, balanced accuracy):")
for _, r in top_rfc.iterrows():
    delta = r["value"] - 0.5
    L.append(f"  phase {r['phase']}  |  {r['cell']:<22s}  bal_acc = {r['value']:.3f}   (+{delta:.3f} vs chance)")
L.append("")
L.append("Per-phase summary (SVM):")
for phase in ESTROUS_PHASES:
    s = svm_phase_summ[phase]
    if s["n"] == 0:
        L.append(f"  phase {phase}: no data")
        continue
    L.append(f"  phase {phase}: {s['above']}/{s['n']} cells above chance   "
             f"mean bal_acc = {s['mean']:.3f}   "
             f"best = {s['best_cell']} ({s['best_val']:.3f})")
L.append("")

# ---------- RQ2 ----------
L.append(sub)
L.append("RQ2 — WEIGHT REGRESSION")
L.append(sub)
L.append(f"Significant cells (RF, p<0.05, permutation): "
         f"{reg_sig_n} / {reg_sig_total}")
L.append("")
L.append("Top 5 cells (Random Forest, R^2):")
for _, r in top_reg.iterrows():
    L.append(f"  phase {r['phase']}  |  {r['cell']:<22s}  R^2 = {r['value']:.3f}")
L.append("")
L.append("Per-phase summary (weight):")
for phase in ESTROUS_PHASES:
    s = reg_phase_summ[phase]
    if s["n"] == 0:
        L.append(f"  phase {phase}: no data")
        continue
    L.append(f"  phase {phase}: {s['above']}/{s['n']} cells above chance   "
             f"mean R^2 = {s['mean']:.3f}   "
             f"best = {s['best_cell']} ({s['best_val']:.3f})")
L.append("")

# ---------- band vs ratio ----------
L.append(sub)
L.append("BAND vs RATIO — where does the signal live?")
L.append(sub)
if svm_bvr is not None:
    L.append("Diet classification (SVM):")
    L.append(f"  Bands  : {svm_bvr['band_above']}/{svm_bvr['band_n']} above chance   "
             f"(mean bal_acc = {svm_bvr['band_mean']:.3f})")
    L.append(f"  Ratios : {svm_bvr['ratio_above']}/{svm_bvr['ratio_n']} above chance   "
             f"(mean bal_acc = {svm_bvr['ratio_mean']:.3f})")
    if not np.isnan(svm_bvr['band_mean']) and not np.isnan(svm_bvr['ratio_mean']):
        winner = "ratios" if svm_bvr['ratio_mean'] > svm_bvr['band_mean'] else "bands"
        L.append(f"  -> On average, {winner} are the stronger predictors for diet.")
L.append("")
if reg_bvr is not None:
    L.append("Weight regression (RF):")
    L.append(f"  Bands  : {reg_bvr['band_above']}/{reg_bvr['band_n']} above chance   "
             f"(mean R^2 = {reg_bvr['band_mean']:.3f})")
    L.append(f"  Ratios : {reg_bvr['ratio_above']}/{reg_bvr['ratio_n']} above chance   "
             f"(mean R^2 = {reg_bvr['ratio_mean']:.3f})")
    if not np.isnan(reg_bvr['band_mean']) and not np.isnan(reg_bvr['ratio_mean']):
        winner = "ratios" if reg_bvr['ratio_mean'] > reg_bvr['band_mean'] else "bands"
        L.append(f"  -> On average, {winner} are the stronger predictors for weight.")
L.append("")

# ---------- plain-language overall interpretation ----------
L.append(sub)
L.append("PLAIN-LANGUAGE READING")
L.append(sub)
if svm_sig_total > 0:
    frac = svm_sig_n / svm_sig_total
    if svm_sig_n == 0:
        L.append("- Diet classification: NO cell was significantly better than chance.")
        L.append("  The LFP alone does NOT reliably classify HF vs CTRL in this cable.")
    elif frac < 0.1:
        L.append(f"- Diet classification: only {svm_sig_n}/{svm_sig_total} cells were")
        L.append("  significantly above chance. Signal exists but is sparse and localised.")
    elif frac < 0.3:
        L.append(f"- Diet classification: {svm_sig_n}/{svm_sig_total} cells were")
        L.append("  significantly above chance. Moderate, localised signal.")
    else:
        L.append(f"- Diet classification: {svm_sig_n}/{svm_sig_total} cells were")
        L.append("  significantly above chance. The LFP carries a broad diet signal.")

if reg_sig_total > 0:
    if reg_sig_n == 0:
        L.append("- Weight regression: NO cell was significantly better than chance.")
        L.append("  The LFP alone does NOT reliably predict body weight in this cable.")
    else:
        L.append(f"- Weight regression: {reg_sig_n}/{reg_sig_total} cells were")
        L.append("  significantly above chance for body-weight prediction.")

# which phase looks most informative?
best_phase_svm = max(svm_phase_summ.items(),
                     key=lambda kv: (kv[1]['best_val'] if not np.isnan(kv[1]['best_val']) else -1))
best_phase_reg = max(reg_phase_summ.items(),
                     key=lambda kv: (kv[1]['best_val'] if not np.isnan(kv[1]['best_val']) else -1))
L.append(f"- Most informative estrous phase for diet: phase {best_phase_svm[0]}")
L.append(f"  (best cell: {best_phase_svm[1]['best_cell']}, "
         f"bal_acc = {best_phase_svm[1]['best_val']:.3f})")
L.append(f"- Most informative estrous phase for weight: phase {best_phase_reg[0]}")
L.append(f"  (best cell: {best_phase_reg[1]['best_cell']}, "
         f"R^2 = {best_phase_reg[1]['best_val']:.3f})")
L.append("")


L.append(sub)
L.append("PROFESSOR'S REQUIREMENTS — CHECKLIST")
L.append(sub)
L.append("[x] Weight is NOT a feature — only a target (see 10c).")
L.append("[x] Separate models per estrous phase (A, B, C, D).")
L.append("[x] Separate models per frequency band (6 bands).")
L.append("[x] Separate models per band-to-band ratio (15 ratios).")
L.append("[x] Classification: SVM-RBF + Random Forest.")
L.append("[x] Regression: Random Forest (target = body weight).")
L.append("[x] Cables analysed separately (Cable1 vs Cable3, same pipeline).")
L.append("[x] Leave-One-Mouse-Out cross-validation.")
L.append("[x] Permutation testing (mouse-level for classification;")
L.append("    recording-level for regression); N = 200 iterations.")
L.append("[x] No band/total ratios (relative power already encodes that).")
L.append("[x] Per-phase per-cell heatmaps and bar charts produced.")
L.append("[ ] Cable 3 pipeline re-run and figures produced.")
L.append("    (change CABLE = 'Cable3' at the top of 10a/10b/10c/11 and rerun.)")
L.append("")
L.append("Everything PROFESSOR asked for in the 2026-07-09 meeting is implemented")
L.append("in this cable. Once Cable3 is also processed, the study is complete.")
L.append("")

out_txt = os.path.join(OUT_DIR, f"11_INTERPRETATION_{CABLE}.txt")
with open(out_txt, "w") as f:
    f.write("\n".join(L))
print(f"\n  saved: {out_txt}")


# ---------- also echo the key numbers to the console ----------
print("\n" + "=" * 70)
print("QUICK READ")
print("=" * 70)
print(f"\nRQ1 — Diet classification (SVM):  {svm_sig_n}/{svm_sig_total} cells p<0.05")
print("Top 3:")
for _, r in top_svm.head(3).iterrows():
    print(f"   phase {r['phase']}  |  {r['cell']:<22s}  bal_acc = {r['value']:.3f}")
print(f"\nRQ2 — Weight regression (RF):     {reg_sig_n}/{reg_sig_total} cells p<0.05")
print("Top 3:")
for _, r in top_reg.head(3).iterrows():
    print(f"   phase {r['phase']}  |  {r['cell']:<22s}  R^2 = {r['value']:.3f}")

print(f"\nAll figures saved to:\n  {OUT_DIR}")
print("STEP 11 finished successfully.")
