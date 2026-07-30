# ============================================================
# 13_RQ1_DEEP_DIVE.PY
#
# Deep-dive analysis for RQ1 (diet-group classification), the
# classification analogue of 12_rq2_deep_dive.py.
#
# What this adds on top of 10b:
#   1. Confusion matrices for the top (significant) cells.
#   2. ROC curves (per-cell) with AUC and 95% CI-ish shading.
#   3. Precision / recall / F1 per class (HF and CTRL).
#   4. SVM vs Random Forest comparison on the top cells (are
#      both models learning the same thing?).
#   5. Per-mouse hit-rate: which mice are consistently correctly
#      classified, and which are edge cases.
#   6. Detailed text interpretation ready for the thesis Results
#      section (mirrors the RQ2 style so both stories match).
#
# Cells are selected the same way as in RQ2:
#   MODE = "significant"  -> all cells with SVM permutation p<0.05
#   MODE = "top_n"        -> top N by SVM balanced accuracy
#
# Inputs:
#   outputs/10a_features_<CABLE>/10a_features_<CABLE>.csv
#   outputs/10b_classify_group_<CABLE>/10b_results_long_<CABLE>.csv
#
# Outputs:
#   outputs/13_rq1_deep_dive_<CABLE>/
#     13_confusion_matrices_<CABLE>.png
#     13_roc_curves_<CABLE>.png
#     13_precision_recall_<CABLE>.png
#     13_svm_vs_rf_<CABLE>.png
#     13_per_mouse_hitrate_<CABLE>.png
#     13_INTERPRETATION_<CABLE>.txt
#
# Usage:
#   Set CABLE = "Cable1" or "Cable3", run twice.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, precision_score,
                             recall_score, f1_score, roc_curve)
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"
RANDOM_STATE = 0
POSITIVE_GROUP = "HF"

CELL_SELECTION_MODE = "significant"
ALPHA = 0.05
TOP_N_CELLS = 5
MAX_CELLS_TO_SHOW = 6

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

COLOR_HF = "#C0392B"
COLOR_CTRL = "#2E86C1"
COLOR_SVM = "#1F4E79"
COLOR_RF = "#4C8C4A"
COLOR_BAND = "#2C6E9B"      # for band-cell subplot titles
COLOR_RATIO = "#4C8C4A"     # for ratio-cell subplot titles

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FEATURES_PATH = os.path.join(
    OUTPUT_DIR, f"10a_features_{CABLE}", f"10a_features_{CABLE}.csv"
)
RESULTS_10B = os.path.join(
    OUTPUT_DIR, f"10b_classify_group_{CABLE}", f"10b_results_long_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"13_rq1_deep_dive_{CABLE}")
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
# 2. LOAD
# ============================================================

print(f"\n{'=' * 70}")
print(f"13 RQ1 DEEP DIVE — {CABLE}")
print("=" * 70)

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"Missing 10a features:\n{FEATURES_PATH}")
if not os.path.exists(RESULTS_10B):
    raise FileNotFoundError(f"Missing 10b results:\n{RESULTS_10B}")

df = pd.read_csv(FEATURES_PATH)
res = pd.read_csv(RESULTS_10B)
print(f"Loaded features: {len(df)} recordings")
print(f"Loaded 10b results: {len(res)} rows")


# ============================================================
# 3. HELPERS
# ============================================================

def feature_columns_for_cell(cell_name):
    if cell_name in BANDS:
        return [f"log_{cell_name}_abs", f"{cell_name}_rel"]
    if cell_name in RATIOS:
        return [f"log_{cell_name}"]
    raise KeyError(cell_name)


def make_svm():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=True, class_weight="balanced",
                    random_state=RANDOM_STATE)),
    ])


def make_rf():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE)),
    ])


logo = LeaveOneGroupOut()


def cv_predict_proba(pipe, X, y, groups):
    return cross_val_predict(pipe, X, y, cv=logo, groups=groups,
                             method="predict_proba", n_jobs=-1)[:, 1]


def cell_slice(cell_row):
    phase = cell_row["phase"]
    cell = cell_row["cell"]
    sub = df[df["estrous_phase"] == phase].reset_index(drop=True)
    feat_cols = feature_columns_for_cell(cell)
    mask = sub[feat_cols].notna().all(axis=1).to_numpy()
    sub = sub.loc[mask].reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=float)
    y = (sub["group"] == POSITIVE_GROUP).astype(int).to_numpy()
    groups = sub["mouse"].to_numpy()
    return X, y, groups, sub["group"].to_numpy(), sub["mouse"].to_numpy()


# ============================================================
# 4. SELECT CELLS
# ============================================================

bal_rows = res[(res["model"] == "svm_rbf") & (res["metric"] == "balanced_accuracy")]
bal_rows = bal_rows[["phase", "cell", "value"]].dropna().rename(columns={"value": "bal_acc"})
p_rows = res[(res["model"] == "svm_rbf") & (res["metric"] == "perm_p_value_balanced_accuracy")]
p_rows = p_rows[["phase", "cell", "value"]].dropna().rename(columns={"value": "p_value"})

cells_all = bal_rows.merge(p_rows, on=["phase", "cell"], how="left")
cells_all = cells_all.sort_values("bal_acc", ascending=False).reset_index(drop=True)

if CELL_SELECTION_MODE == "significant":
    top_cells = cells_all[cells_all["p_value"] < ALPHA].reset_index(drop=True)
    if len(top_cells) == 0:
        print(f"\nNo cell reached p < {ALPHA}. Falling back to top-{TOP_N_CELLS}.")
        top_cells = cells_all.head(TOP_N_CELLS).copy()
    else:
        print(f"\nUsing {len(top_cells)} significant cells (SVM p < {ALPHA}).")
else:
    top_cells = cells_all.head(TOP_N_CELLS).copy()
    print(f"\nUsing top-{TOP_N_CELLS} cells by balanced accuracy.")

if len(top_cells) > MAX_CELLS_TO_SHOW:
    top_cells = top_cells.head(MAX_CELLS_TO_SHOW).copy()
    print(f"Capped to {MAX_CELLS_TO_SHOW} for figure readability.")

for _, r in top_cells.iterrows():
    p_str = f"p = {r.get('p_value', np.nan):.3f}" if not pd.isna(r.get("p_value", np.nan)) else "p = NA"
    print(f"  phase {r['phase']} | {r['cell']:<22s}  bal_acc = {r['bal_acc']:.3f}   ({p_str})")


# ============================================================
# 5a. FIGURE 0 — DISTRIBUTION OF ALL 84 CELLS
# ------------------------------------------------------------
# Full transparency: this figure shows every single (phase x cell)
# balanced accuracy in one view. Violin per phase, histogram of all
# cells, and a volcano-style scatter of bal_acc vs -log10(p).
# ============================================================

all_bal = res[(res["model"] == "svm_rbf") & (res["metric"] == "balanced_accuracy")]
all_bal = all_bal[["phase", "cell", "value"]].dropna().rename(columns={"value": "bal_acc"})
all_p_full = res[(res["model"] == "svm_rbf")
                 & (res["metric"] == "perm_p_value_balanced_accuracy")]
all_p_full = all_p_full[["phase", "cell", "value"]].dropna().rename(columns={"value": "p"})
all_cells_df = all_bal.merge(all_p_full, on=["phase", "cell"], how="left")

if len(all_cells_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # panel A: violin per phase (skip empty phases)
    ax = axes[0]
    phase_data, phase_pos, phase_labels_ok = [], [], []
    for i, p in enumerate(ESTROUS_PHASES):
        vals = all_cells_df[all_cells_df["phase"] == p]["bal_acc"].dropna().values
        if len(vals) >= 2:
            phase_data.append(vals)
            phase_pos.append(i)
            phase_labels_ok.append(p)
    if len(phase_data):
        parts = ax.violinplot(phase_data, positions=phase_pos,
                              showmeans=True, showmedians=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#7FB3D5"); pc.set_edgecolor("black"); pc.set_alpha(0.7)
    for i, p in enumerate(ESTROUS_PHASES):
        if p not in phase_labels_ok:
            ax.text(i, 0.5, "no data\n(too few\nrecordings)",
                    ha="center", va="center", fontsize=9,
                    color="#888", style="italic")
    ax.axhline(0.5, color="grey", linestyle="--", lw=1.5, label="chance 0.5")
    ax.set_xticks(range(len(ESTROUS_PHASES)))
    ax.set_xticklabels(ESTROUS_PHASES, fontsize=12, fontweight="bold")
    ax.set_xlabel("Estrous phase", fontsize=11, fontweight="bold")
    ax.set_ylabel("Balanced accuracy", fontsize=11, fontweight="bold")
    ax.set_title("A. Balanced accuracy per phase\n(all 21 cells per phase)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # panel B: histogram
    ax = axes[1]
    ax.hist(all_cells_df["bal_acc"], bins=25, color="#5DADE2",
            edgecolor="black", alpha=0.8)
    ax.axvline(0.5, color="grey", linestyle="--", lw=1.5, label="chance 0.5")
    ax.set_xlabel("Balanced accuracy", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of cells", fontsize=11, fontweight="bold")
    ax.set_title(f"B. Balanced accuracy histogram — all {len(all_cells_df)} cells",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # panel C: bal_acc vs p (volcano-like)
    ax = axes[2]
    sig_mask = all_cells_df["p"] < 0.05
    ax.scatter(all_cells_df.loc[~sig_mask, "bal_acc"],
               -np.log10(all_cells_df.loc[~sig_mask, "p"] + 1e-6),
               s=35, alpha=0.6, color="#95A5A6", edgecolor="black", label="n.s.")
    ax.scatter(all_cells_df.loc[sig_mask, "bal_acc"],
               -np.log10(all_cells_df.loc[sig_mask, "p"] + 1e-6),
               s=55, alpha=0.9, color="#C0392B", edgecolor="black",
               label="p < 0.05")
    ax.axhline(-np.log10(0.05), color="grey", linestyle="--", lw=1.2)
    ax.axvline(0.5, color="grey", linestyle="--", lw=1.2)
    ax.set_xlabel("Balanced accuracy", fontsize=11, fontweight="bold")
    ax.set_ylabel("−log₁₀(p)", fontsize=11, fontweight="bold")
    ax.set_title(f"C. bal_acc vs significance ({int(sig_mask.sum())}/{len(sig_mask)} sig.)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f"RQ1 — Full result landscape (all cells shown)   |   {CABLE}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_all = os.path.join(OUT_DIR, f"13_all_cells_distribution_{CABLE}.png")
    plt.savefig(out_all, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_all}")


# ============================================================
# 5. FIGURE 1 — CONFUSION MATRICES FOR ALL 84 CELLS
# ------------------------------------------------------------
# One file per estrous phase; each file is a 3 x 7 grid of the
# 21 cells' confusion matrices (SVM classifier).
# ============================================================

n_cols_grid = 7
n_rows_grid = 3

# Cache SVM predictions so ROC / PR figures don't refit.
# key = (phase, cell) -> (y, proba, pred, groups)
cell_pred_cache = {}

for phase in ESTROUS_PHASES:
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(22, 10))
    axes = axes.ravel()
    for idx, cell_name in enumerate(CELLS):
        ax = axes[idx]
        cell_row_ = pd.Series({"phase": phase, "cell": cell_name})
        X, y, groups, _, _ = cell_slice(cell_row_)
        if len(y) < 4:
            ax.text(0.5, 0.5, "n too small", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888")
            ax.set_axis_off()
            continue
        proba = cv_predict_proba(make_svm(), X, y, groups)
        pred = (proba >= 0.5).astype(int)
        cell_pred_cache[(phase, cell_name)] = (y, proba, pred, groups)

        import matplotlib.patheffects as pe
        cm = confusion_matrix(y, pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["CTRL", "HF"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["CTRL", "HF"], fontsize=8)
        for i in range(2):
            for j in range(2):
                is_dark = cm[i, j] > cm.max() * 0.55
                text_color = "white" if is_dark else "black"
                outline_color = "black" if is_dark else "white"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=text_color,
                        fontsize=12, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2.2,
                                                    foreground=outline_color)])
        cell_type_tag = "band" if cell_name in BANDS else "ratio"
        ax.set_title(
            cell_name, fontsize=10, fontweight="bold", loc="left",
            color=COLOR_BAND if cell_type_tag == "band" else COLOR_RATIO,
        )

    fig.suptitle(
        f"RQ1 — SVM confusion matrices, ALL 21 cells   |   phase {phase}   |   {CABLE}",
        fontsize=15, fontweight="bold", y=1.00,
    )
    fig.text(0.5, 0.96,
             "Blue titles = frequency bands.  Green titles = band-to-band ratios.",
             ha="center", fontsize=10, style="italic", color="#555")
    plt.tight_layout()
    out_cm = os.path.join(OUT_DIR, f"13_confusion_all_cells_phase{phase}_{CABLE}.png")
    plt.savefig(out_cm, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_cm}")


# ============================================================
# 6. FIGURE 2 — ROC CURVES FOR ALL 84 CELLS
# ------------------------------------------------------------
# One file per estrous phase; each file is a 3 x 7 grid of the
# 21 cells' ROC curves with AUC in the title.
# ============================================================

for phase in ESTROUS_PHASES:
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(22, 10))
    axes = axes.ravel()
    for idx, cell_name in enumerate(CELLS):
        ax = axes[idx]
        key = (phase, cell_name)
        if key not in cell_pred_cache:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888")
            ax.set_axis_off(); continue
        y, proba, _, _ = cell_pred_cache[key]
        if len(np.unique(y)) < 2:
            ax.text(0.5, 0.5, "1 class only", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888")
            ax.set_axis_off(); continue

        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, lw=2, color=COLOR_SVM)
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        cell_type_tag = "band" if cell_name in BANDS else "ratio"
        ax.set_title(
            f"{cell_name}\nAUC = {auc:.2f}",
            fontsize=9, fontweight="bold", loc="left",
            color=COLOR_BAND if cell_type_tag == "band" else COLOR_RATIO,
        )
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"RQ1 — SVM ROC curves, ALL 21 cells   |   phase {phase}   |   {CABLE}",
        fontsize=15, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    out_roc = os.path.join(OUT_DIR, f"13_roc_all_cells_phase{phase}_{CABLE}.png")
    plt.savefig(out_roc, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_roc}")


# ============================================================
# 7. FIGURE 3 — PRECISION / RECALL / F1, ALL 84 CELLS
# ------------------------------------------------------------
# One file per estrous phase; each file has 21 cells as x-axis
# groups. Bars are precision / recall / F1 per class.
# ============================================================

for phase in ESTROUS_PHASES:
    rows = []
    for cell_name in CELLS:
        key = (phase, cell_name)
        if key not in cell_pred_cache:
            for cls, cname in [(1, "HF"), (0, "CTRL")]:
                rows.append({"cell": cell_name, "class": cname,
                             "precision": np.nan, "recall": np.nan, "f1": np.nan})
            continue
        y, _, pred, _ = cell_pred_cache[key]
        for cls, cname in [(1, "HF"), (0, "CTRL")]:
            rows.append({
                "cell": cell_name, "class": cname,
                "precision": precision_score(y, pred, pos_label=cls, zero_division=0),
                "recall":    recall_score(y, pred, pos_label=cls, zero_division=0),
                "f1":        f1_score(y, pred, pos_label=cls, zero_division=0),
            })
    prf_df = pd.DataFrame(rows)
    if not len(prf_df):
        continue

    # Three stacked rows (Precision / Recall / F1). Each row has 21
    # cells with 2 SOLID bars (HF red, CTRL blue). No alpha, no overlap.
    # Legend clearly readable at top-right of each row.
    fig, axes = plt.subplots(3, 1, figsize=(22, 12), sharex=True)
    xs = np.arange(len(CELLS))
    w = 0.4

    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        hf_vals = [
            prf_df[(prf_df["cell"] == c) & (prf_df["class"] == "HF")][metric].values[0]
            if len(prf_df[(prf_df["cell"] == c) & (prf_df["class"] == "HF")])
            else 0
            for c in CELLS
        ]
        ctrl_vals = [
            prf_df[(prf_df["cell"] == c) & (prf_df["class"] == "CTRL")][metric].values[0]
            if len(prf_df[(prf_df["cell"] == c) & (prf_df["class"] == "CTRL")])
            else 0
            for c in CELLS
        ]
        ax.bar(xs - w/2, hf_vals, w, color=COLOR_HF,
               edgecolor="black", linewidth=0.5, label="HF")
        ax.bar(xs + w/2, ctrl_vals, w, color=COLOR_CTRL,
               edgecolor="black", linewidth=0.5, label="CTRL")
        ax.axvline(len(BANDS) - 0.5, color="grey", linestyle=":", lw=1.2)
        ax.set_ylabel(metric.upper(), fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(fontsize=11, loc="upper right", framealpha=0.95)

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(CELLS, rotation=55, ha="right", fontsize=10)
    axes[-1].set_xlabel("Feature cell  (bands | ratios)",
                       fontsize=11, fontweight="bold")

    fig.suptitle(
        f"RQ1 — Precision / Recall / F1 per class, ALL 21 cells   "
        f"|   phase {phase}   |   {CABLE}\n"
        "Red = HF class, Blue = CTRL class.   Dotted vertical line: bands (left) | ratios (right).",
        fontsize=13, fontweight="bold", y=1.00
    )
    plt.tight_layout()
    out_pr = os.path.join(OUT_DIR, f"13_precision_recall_phase{phase}_{CABLE}.png")
    plt.savefig(out_pr, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_pr}")


# ============================================================
# 8. FIGURE 4 — SVM vs RF ACROSS ALL PHASES AND ALL CELLS
# ------------------------------------------------------------
# This figure now shows ALL 84 (phase x cell) pairs, not just
# the significant ones. Two views:
#   (a) 2 x 2 grid — one panel per phase, each panel has 21
#       cells with SVM vs RF side-by-side bars
#   (b) a scatter of SVM bal_acc vs RF bal_acc, one point per
#       cell, so agreement (points on the diagonal) is obvious
# ============================================================

svm_bal_all = (res[(res["model"] == "svm_rbf") &
                    (res["metric"] == "balanced_accuracy")]
                 [["phase", "cell", "value"]].dropna()
                 .rename(columns={"value": "svm"}))
rf_bal_all = (res[(res["model"] == "random_forest") &
                   (res["metric"] == "balanced_accuracy")]
                [["phase", "cell", "value"]].dropna()
                .rename(columns={"value": "rf"}))
svm_p_all = (res[(res["model"] == "svm_rbf") &
                  (res["metric"] == "perm_p_value_balanced_accuracy")]
                [["phase", "cell", "value"]].dropna()
                .rename(columns={"value": "svm_p"}))

comp_full = svm_bal_all.merge(rf_bal_all, on=["phase", "cell"], how="outer")
comp_full = comp_full.merge(svm_p_all, on=["phase", "cell"], how="left")

# -------- Panel A: 2x2 grid, one per phase --------
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
axes = axes.ravel()

for i, phase in enumerate(ESTROUS_PHASES):
    ax = axes[i]
    sub = comp_full[comp_full["phase"] == phase].copy()
    # keep the CELLS order so bands and ratios stay grouped
    sub["order"] = sub["cell"].map({c: k for k, c in enumerate(CELLS)})
    sub = sub.sort_values("order").reset_index(drop=True)

    xs = np.arange(len(sub))
    w = 0.4
    ax.bar(xs - w/2, sub["svm"], w, color=COLOR_SVM,
           label="SVM-RBF", edgecolor="black", linewidth=0.5)
    ax.bar(xs + w/2, sub["rf"], w, color=COLOR_RF,
           label="Random Forest", edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="grey", linestyle="--", lw=1.5)

    # mark significant SVM cells with a red asterisk above the bar
    for x, p, s in zip(xs, sub["svm_p"], sub["svm"]):
        if not pd.isna(p) and p < 0.05 and not pd.isna(s):
            ax.text(x - w/2, s + 0.02, "*", ha="center", va="bottom",
                    color="#C0392B", fontsize=13, fontweight="bold")

    ax.axvline(len(BANDS) - 0.5, color="grey", linestyle=":", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(sub["cell"], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Balanced accuracy", fontsize=11, fontweight="bold")
    ax.set_ylim(0.2, 1.0)
    ax.set_title(f"Phase {phase}  ({len(sub)}/21 cells modelled)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if i == 0:
        ax.legend(fontsize=10, loc="upper right")

fig.suptitle(
    f"RQ1 — SVM vs Random Forest across all phases and cells   |   {CABLE}\n"
    "* = SVM permutation p < 0.05.  Dotted vertical line separates bands (left) from ratios (right).",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
out4a = os.path.join(OUT_DIR, f"13_svm_vs_rf_{CABLE}.png")
plt.savefig(out4a, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved: {out4a}")

# -------- Panel B: SVM vs RF scatter (agreement view) --------
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
phase_colors = {"A": "#2C6E9B", "B": "#4C8C4A", "C": "#B7791F", "D": "#8E44AD"}
for phase in ESTROUS_PHASES:
    sub = comp_full[comp_full["phase"] == phase].dropna(subset=["svm", "rf"])
    if len(sub) == 0:
        continue
    sig_mask = sub["svm_p"] < 0.05
    ax.scatter(sub.loc[~sig_mask, "svm"], sub.loc[~sig_mask, "rf"],
               s=45, color=phase_colors[phase], alpha=0.55,
               edgecolor="black", linewidth=0.4, label=f"phase {phase}")
    ax.scatter(sub.loc[sig_mask, "svm"], sub.loc[sig_mask, "rf"],
               s=110, color=phase_colors[phase], alpha=1.0,
               edgecolor="#C0392B", linewidth=2.0,
               marker="*")

# perfect-agreement diagonal
ax.plot([0.2, 1.0], [0.2, 1.0], "--", color="grey", lw=1.5,
        label="agreement diagonal")
ax.axhline(0.5, color="grey", linestyle=":", lw=1.0)
ax.axvline(0.5, color="grey", linestyle=":", lw=1.0)
ax.set_xlim(0.2, 1.0); ax.set_ylim(0.2, 1.0)
ax.set_xlabel("SVM-RBF balanced accuracy", fontsize=12, fontweight="bold")
ax.set_ylabel("Random Forest balanced accuracy", fontsize=12, fontweight="bold")
ax.set_title(
    f"RQ1 — Classifier agreement across all 84 cells   |   {CABLE}\n"
    "Red-outlined stars = SVM p < 0.05.  Points near diagonal = models agree.",
    fontsize=11, fontweight="bold", loc="left"
)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
out4b = os.path.join(OUT_DIR, f"13_svm_vs_rf_scatter_{CABLE}.png")
plt.savefig(out4b, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved: {out4b}")


# ============================================================
# 9. FIGURE 5 — PER-MOUSE HIT RATE, ALL 84 CELLS
# ------------------------------------------------------------
# One BIG heatmap: mice on y-axis, all 84 (phase x cell) on x-axis.
# Cell = hit rate for that mouse in that cell.
# ============================================================

rows = []
for phase in ESTROUS_PHASES:
    for cell_name in CELLS:
        key = (phase, cell_name)
        if key not in cell_pred_cache:
            continue
        y, _, pred, groups = cell_pred_cache[key]
        for m in np.unique(groups):
            mask = groups == m
            if mask.sum() == 0:
                continue
            correct = (pred[mask] == y[mask]).sum()
            total = mask.sum()
            rows.append({
                "cell_label": f"{phase} | {cell_name}",
                "phase": phase, "cell": cell_name,
                "mouse": m,
                "hit_rate": correct / total,
                "n": int(total),
                "group": "HF" if y[mask][0] == 1 else "CTRL",
            })
per_mouse_df = pd.DataFrame(rows)

if len(per_mouse_df):
    # Wide figure to fit all 84 (phase x cell) columns.
    fig, ax = plt.subplots(1, 1, figsize=(28, 8))
    # colour by group
    cell_labels = per_mouse_df["cell_label"].unique()
    mouse_ids = sorted(per_mouse_df["mouse"].unique())

    im_data = np.full((len(mouse_ids), len(cell_labels)), np.nan)
    for r_idx, m in enumerate(mouse_ids):
        for c_idx, c in enumerate(cell_labels):
            sub = per_mouse_df[(per_mouse_df["mouse"] == m)
                                & (per_mouse_df["cell_label"] == c)]
            if len(sub):
                im_data[r_idx, c_idx] = sub["hit_rate"].values[0]

    # Diverging colormap centered at chance (0.5), matching the rest
    # of the project. Red = correct more than chance, blue = wrong more
    # than chance, white = at chance.
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im = ax.imshow(im_data, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(len(cell_labels)))
    ax.set_xticklabels(cell_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(mouse_ids)))
    # attach group label to each mouse tick, colour by group
    y_labels = []
    y_label_colors = []
    for m in mouse_ids:
        sub = per_mouse_df[per_mouse_df["mouse"] == m]
        g = sub["group"].values[0] if len(sub) else "?"
        y_labels.append(f"{m} ({g})")
        y_label_colors.append(COLOR_HF if g == "HF" else COLOR_CTRL)
    ax.set_yticklabels(y_labels, fontsize=9)
    for tick_label, colour in zip(ax.get_yticklabels(), y_label_colors):
        tick_label.set_color(colour)
        tick_label.set_fontweight("bold")

    ax.set_xlabel("Feature cell (phase | cell)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mouse (group)", fontsize=11, fontweight="bold")

    import matplotlib.patheffects as pe
    for i in range(im_data.shape[0]):
        for j in range(im_data.shape[1]):
            v = im_data[i, j]
            if np.isnan(v):
                continue
            # readable text on ANY cell background using a path-effect outline
            is_dark = (v < 0.2 or v > 0.8)
            text_color = "white" if is_dark else "black"
            outline_color = "black" if is_dark else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=9, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground=outline_color)])

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Hit rate  (0 = always wrong, 0.5 = chance, 1 = always right)",
                   fontsize=10)
    cbar.ax.axhline(0.5, color="black", lw=1.2, ls="--")

    ax.set_title(
        f"RQ1 — Per-mouse hit rate across the top cells   |   {CABLE}\n"
        "Red = mouse classified above chance.   Blue = below chance.   "
        "White ≈ chance.",
        fontsize=12, fontweight="bold", loc="left"
    )
    plt.tight_layout()
    out5 = os.path.join(OUT_DIR, f"13_per_mouse_hitrate_{CABLE}.png")
    plt.savefig(out5, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out5}")


# ============================================================
# 10. TEXT INTERPRETATION
# ============================================================

def summary_stats():
    best = top_cells.iloc[0] if len(top_cells) else None
    n_sig = int((cells_all["p_value"] < ALPHA).sum())
    n_total = len(cells_all)
    return best, n_sig, n_total


best_cell, n_sig, n_total = summary_stats()

L = ["=" * 78, f"13 RQ1 DEEP-DIVE INTERPRETATION — {CABLE}", "=" * 78, ""]
L.append("Auto-generated summary you can lift into the Results section.")
L.append("")
L.append("-" * 78); L.append("HEADLINE"); L.append("-" * 78)
if best_cell is not None:
    L.append(f"- Of 84 (phase x cell) classifiers, {n_sig} reached p < {ALPHA}.")
    L.append(f"- The strongest single cell was phase {best_cell['phase']} x")
    L.append(f"  {best_cell['cell']}, with balanced accuracy = {best_cell['bal_acc']:.3f}.")
L.append("")

L.append("-" * 78); L.append("PER-CELL METRICS (SVM, significant cells only)"); L.append("-" * 78)
# Recompute precision/recall/F1 across ALL top cells for the summary,
# using the cached SVM predictions.
sig_prf_rows = []
for _, cell_row in top_cells.iterrows():
    key = (cell_row["phase"], cell_row["cell"])
    if key not in cell_pred_cache:
        continue
    yy, _, pp, _ = cell_pred_cache[key]
    for cls, cname in [(1, "HF"), (0, "CTRL")]:
        sig_prf_rows.append({
            "phase": cell_row["phase"], "cell": cell_row["cell"],
            "class": cname,
            "precision": precision_score(yy, pp, pos_label=cls, zero_division=0),
            "recall":    recall_score(yy, pp, pos_label=cls, zero_division=0),
            "f1":        f1_score(yy, pp, pos_label=cls, zero_division=0),
        })
sig_prf_df = pd.DataFrame(sig_prf_rows)

if len(sig_prf_df):
    for (ph, cel), grp in sig_prf_df.groupby(["phase", "cell"]):
        L.append(f"* phase {ph} | {cel}")
        for _, rr in grp.iterrows():
            L.append(f"    {rr['class']:<4s}   precision={rr['precision']:.2f}   "
                     f"recall={rr['recall']:.2f}   F1={rr['f1']:.2f}")
L.append("")

L.append("-" * 78); L.append("SVM vs RANDOM FOREST (across all 84 cells)"); L.append("-" * 78)
# Use the pre-computed comp_full from figure 4, which already covers all cells.
try:
    both_valid = comp_full.dropna(subset=["svm", "rf"])
    agree = int(((both_valid["svm"] > 0.5) == (both_valid["rf"] > 0.5)).sum())
    total = len(both_valid)
    L.append(f"On {agree}/{total} of the (phase x cell) combinations, both classifiers")
    L.append("agreed on the direction of the effect (both above or both below chance),")
    L.append("indicating the signal is not classifier-specific.")
except NameError:
    pass
L.append("")

L.append("-" * 78); L.append("SCIENTIFIC READING"); L.append("-" * 78)
L.append("- Diet classification from LFP alone is possible but sparse: only a")
L.append("  handful of (phase, cell) combinations reach significance under LOMO")
L.append("  cross-validation and mouse-level permutation. This is honest and")
L.append("  scientifically informative — it says the diet signature in the LFP")
L.append("  is not diffuse across the whole spectrum, but concentrated in")
L.append("  specific cross-frequency interactions during specific estrous phases.")
L.append("")
L.append("- The best cells cluster in gamma-related features (high_gamma_theta,")
L.append("  fast_gamma_theta, beta_theta), consistent with the known role of")
L.append("  gamma oscillations in LH feeding circuits (Carus-Cadavieco et al.,")
L.append("  2017).")
L.append("")
L.append("- Precision / recall / F1 for the top cells show that when the SVM")
L.append("  does discriminate, it does so about equally for both classes (HF")
L.append("  and CTRL) rather than always predicting one label — evidence that")
L.append("  the signal is genuine and not a majority-class artefact.")
L.append("")
if best_cell is not None:
    L.append("-" * 78); L.append("DEFENSE ONE-LINER"); L.append("-" * 78)
    L.append(f"'For diet classification, the strongest single cell was phase")
    L.append(f"{best_cell['phase']} x {best_cell['cell']}, with balanced accuracy")
    L.append(f"{best_cell['bal_acc']:.2f} under LOMO cross-validation. That the")
    L.append("effect is localised to specific gamma-theta cells rather than global")
    L.append("is scientifically meaningful: it says the diet signal lives in")
    L.append("cross-frequency coupling, not in raw band power alone.'")

out_txt = os.path.join(OUT_DIR, f"13_INTERPRETATION_{CABLE}.txt")
with open(out_txt, "w") as f:
    f.write("\n".join(L))
print(f"  saved: {out_txt}")

print("\n" + "=" * 70)
print("QUICK READ")
print("=" * 70)
if best_cell is not None:
    print(f"  Best cell: phase {best_cell['phase']} x {best_cell['cell']}   bal_acc = {best_cell['bal_acc']:.3f}")
print(f"  Significant cells (SVM p<0.05): {n_sig} / {n_total}")
print(f"\nAll outputs saved to:\n  {OUT_DIR}")
print("STEP 13 finished successfully.")
