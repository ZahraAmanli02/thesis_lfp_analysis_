# ============================================================
# 10B_CLASSIFY_GROUP.PY
#
# Purpose:
#   RQ1 revisited (PROFESSOR, 2026-07-09).
#   Can LFP band power alone classify HF vs CTRL, and if so,
#   inside which (estrous phase, feature cell) does it work?
#
# Design (per PROFESSOR):
#   * one classification model per (estrous phase, feature cell)
#     - feature cells = 6 frequency bands  + 15 band-to-band ratios
#     -> 21 cells x 4 phases = 84 models per algorithm per cable
#   * two algorithms per cell: SVM-RBF and Random Forest
#   * feature block per model:
#         band cell   -> [log_<band>_abs, <band>_rel]    (2 features)
#         ratio cell  -> [log_<ratio>]                    (1 feature)
#     Band/total ratios are NOT used (relative power is already that).
#   * body weight is NOT used as a feature (it correlates with group
#     by design, so it would inflate accuracy). Weight goes in 10c.
#   * Leave-One-Mouse-Out cross-validation
#   * permutation null on the primary (SVM) model,
#     shuffled at MOUSE level because group is fixed per mouse
#
# Metrics:
#   balanced accuracy, ROC-AUC, accuracy
#
# Input:
#   outputs/10a_features_<CABLE>/10a_features_<CABLE>.csv
#
# Output:
#   outputs/10b_classify_group_<CABLE>/
#       10b_results_long_<CABLE>.csv
#       10b_summary_<CABLE>.txt
#       10b_heatmap_group_<CABLE>.png    (phase x cell, bal_acc)
#
# Usage:
#   Set CABLE = "Cable1" or "Cable3" in SETTINGS, run twice.
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
                             accuracy_score)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"                 # switch to "Cable3" for the parallel run
RANDOM_STATE = 0
N_PERMUTATIONS = 200             # 200 keeps p-value resolution at 0.005; 1000 was too slow
POSITIVE_GROUP = "HF"

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
# Full pairwise ratio set (15 = C(6,2)), higher-freq / lower-freq.
# Order MUST match 10a's RATIOS dict so the heatmap columns line up.
RATIOS = [
    # /delta baseline
    "theta_delta", "beta_delta", "low_gamma_delta",
    "high_gamma_delta", "fast_gamma_delta",
    # /theta baseline
    "beta_theta", "low_gamma_theta", "high_gamma_theta", "fast_gamma_theta",
    # /beta baseline
    "low_gamma_beta", "high_gamma_beta", "fast_gamma_beta",
    # within-gamma
    "high_gamma_low_gamma", "fast_gamma_low_gamma", "fast_gamma_high_gamma",
]
CELLS = BANDS + RATIOS                    # 6 + 15 = 21 cells per phase

ESTROUS_PHASES = ["A", "B", "C", "D"]
MIN_RECORDINGS_PER_PHASE = 8      # lowered from 12 to include Cable 3 phase C (metestrus is the shortest cycle phase, ~6-8 h → naturally under-sampled; reported as exploratory)
MIN_MICE_PER_PHASE = 6

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FEATURES_PATH = os.path.join(
    OUTPUT_DIR, f"10a_features_{CABLE}", f"10a_features_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"10b_classify_group_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"10b_results_long_{CABLE}.csv")
OUT_TXT = os.path.join(OUT_DIR, f"10b_summary_{CABLE}.txt")
OUT_PNG = os.path.join(OUT_DIR, f"10b_heatmap_group_{CABLE}.png")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. LOAD FEATURES (from 10a)
# ============================================================

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(
        f"Missing 10a output:\n{FEATURES_PATH}\n"
        "Run 10a_prepare_features.py first."
    )

df = pd.read_csv(FEATURES_PATH)
print(f"\n{CABLE}: loaded {len(df)} rows from 10a")


# ============================================================
# 3. FEATURE BLOCK PER CELL
# ------------------------------------------------------------
# band cell   -> two features   (log absolute + relative)
# ratio cell  -> one feature    (log of the band-to-band ratio)
# ============================================================

def feature_columns_for_cell(cell_name):
    if cell_name in BANDS:
        return [f"log_{cell_name}_abs", f"{cell_name}_rel"]
    if cell_name in RATIOS:
        return [f"log_{cell_name}"]
    raise KeyError(f"Unknown cell name: {cell_name}")


# ============================================================
# 4. MODEL BUILDERS
# ============================================================

def make_classifier_pipes():
    """SVM-RBF + Random Forest, both class-balanced. Fresh objects each call."""
    return {
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=True, class_weight="balanced",
                        random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500, class_weight="balanced",
                random_state=RANDOM_STATE)),
        ]),
    }


logo = LeaveOneGroupOut()


def evaluate_classification(pipe, X, y, groups):
    """LOMO out-of-fold predictions + three headline metrics."""
    proba = cross_val_predict(pipe, X, y, cv=logo, groups=groups,
                              method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": accuracy_score(y, pred),
    }


def eligible(sub):
    if len(sub) < MIN_RECORDINGS_PER_PHASE:
        return False, "too_few_recordings"
    if sub["mouse"].nunique() < MIN_MICE_PER_PHASE:
        return False, "too_few_mice"
    if sub["group"].nunique() < 2:
        return False, "one_group_only"
    return True, "ok"


# ============================================================
# 5. RUN — PER PHASE x PER CELL
# ============================================================

rows = []
print("\n" + "=" * 74)
print("PER-PHASE x PER-CELL CLASSIFICATION (HF vs CTRL)")
print(f"  cells = {len(BANDS)} bands + {len(RATIOS)} ratios = {len(CELLS)}")
print("=" * 74)

for phase in ESTROUS_PHASES:
    sub = df[df["estrous_phase"] == phase].reset_index(drop=True)
    n_rec = len(sub)
    n_mice = sub["mouse"].nunique()
    n_hf = int((sub["group"] == POSITIVE_GROUP).sum())
    n_ctrl = int((sub["group"] != POSITIVE_GROUP).sum())
    ok, reason = eligible(sub)

    header = (f"\n[phase {phase}]  n={n_rec}  mice={n_mice}  "
              f"HF={n_hf}  CTRL={n_ctrl}")
    if not ok:
        print(header + f"  --> skipped ({reason})")
        rows.append({"phase": phase, "cell": "*", "cell_type": "*",
                     "model": "*", "metric": "skipped_reason",
                     "value": np.nan, "note": reason,
                     "n_recordings": n_rec, "n_mice": n_mice})
        continue
    print(header)

    groups_arr = sub["mouse"].to_numpy()
    y = (sub["group"] == POSITIVE_GROUP).astype(int).to_numpy()

    # Per-mouse HF label used for the permutation null.
    mouse_series = sub.groupby("mouse")["group"].first()
    mouse_ids = mouse_series.index.to_numpy()
    mouse_labels = (mouse_series == POSITIVE_GROUP).astype(int).to_numpy()

    for cell_name in CELLS:
        cell_type = "band" if cell_name in BANDS else "ratio"
        feat_cols = feature_columns_for_cell(cell_name)

        # Any NaN in this cell's features would break sklearn.
        # Ratios can be NaN if the denominator was <=0 — skip those rows.
        feat_block = sub[feat_cols]
        mask = feat_block.notna().all(axis=1).to_numpy()
        n_masked = int((~mask).sum())
        X = feat_block.to_numpy(dtype=float)[mask]
        y_local = y[mask]
        groups_local = groups_arr[mask]

        # --- fit both algorithms, log all metrics ---
        for model_name, pipe in make_classifier_pipes().items():
            m = evaluate_classification(pipe, X, y_local, groups_local)
            for metric, val in m.items():
                rows.append({"phase": phase, "cell": cell_name,
                             "cell_type": cell_type, "model": model_name,
                             "metric": metric, "value": val,
                             "note": ("" if not n_masked
                                      else f"dropped {n_masked} rows with NaN feature"),
                             "n_recordings": int(mask.sum()), "n_mice": n_mice})

        # --- permutation null on the primary (SVM) model ---
        # group is fixed per mouse -> shuffle labels at mouse level.
        primary = make_classifier_pipes()["svm_rbf"]
        observed_bal = evaluate_classification(
            primary, X, y_local, groups_local)["balanced_accuracy"]

        rng = np.random.default_rng(RANDOM_STATE)
        null_bal = np.empty(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            perm = rng.permutation(mouse_labels)
            mapping = dict(zip(mouse_ids, perm))
            y_perm = np.array([mapping[g] for g in groups_local])
            if len(np.unique(y_perm)) < 2:
                null_bal[i] = 0.5
                continue
            pipe_p = make_classifier_pipes()["svm_rbf"]
            null_bal[i] = evaluate_classification(
                pipe_p, X, y_perm, groups_local)["balanced_accuracy"]
        p_bal = (1 + np.sum(null_bal >= observed_bal)) / (1 + N_PERMUTATIONS)

        rows.append({"phase": phase, "cell": cell_name, "cell_type": cell_type,
                     "model": "svm_rbf",
                     "metric": "perm_null_mean_balanced_accuracy",
                     "value": float(null_bal.mean()), "note": "",
                     "n_recordings": int(mask.sum()), "n_mice": n_mice})
        rows.append({"phase": phase, "cell": cell_name, "cell_type": cell_type,
                     "model": "svm_rbf",
                     "metric": "perm_p_value_balanced_accuracy",
                     "value": p_bal, "note": "",
                     "n_recordings": int(mask.sum()), "n_mice": n_mice})

        # short progress line so long runs don't look frozen
        g_line = [r for r in rows if r["phase"] == phase and r["cell"] == cell_name
                  and r["metric"] == "balanced_accuracy"]
        g_str = " / ".join(f"{r['model']}={r['value']:.2f}" for r in g_line)
        print(f"  [{cell_type:>5}] {cell_name:<18s} bal_acc: {g_str:<45s}"
              f" (perm p={p_bal:.3f})")


results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved long-format results:\n{OUT_CSV}")


# ============================================================
# 6. SUMMARY (TXT)
# ============================================================

def pivot_metric(model_name, metric):
    """phase x cell table for a given (model, metric)."""
    sub = results_df[(results_df["model"] == model_name)
                     & (results_df["metric"] == metric)]
    if sub.empty:
        return pd.DataFrame(index=ESTROUS_PHASES, columns=CELLS, dtype=float)
    return (sub.pivot(index="phase", columns="cell", values="value")
               .reindex(index=ESTROUS_PHASES, columns=CELLS))


lines = []
lines.append("=" * 88)
lines.append(f"10B CLASSIFICATION (HF vs CTRL) — {CABLE}")
lines.append("=" * 88)
lines.append("")
lines.append("Design (per PROFESSOR, 2026-07-09):")
lines.append("  * one classifier per (estrous phase, feature cell)")
lines.append(f"  * cells = {len(BANDS)} bands + {len(RATIOS)} band-to-band ratios "
             f"= {len(CELLS)} cells per phase")
lines.append("  * feature block:")
lines.append("      band cell  -> [log_<band>_abs, <band>_rel]  (2 features)")
lines.append("      ratio cell -> [log_<ratio>]                  (1 feature)")
lines.append("  * body weight NOT used as feature (see 10c for weight target)")
lines.append("  * validation = leave-one-mouse-out")
lines.append("  * permutation p-value on SVM primary, mouse-level shuffle")
lines.append("")

for model_name in ["svm_rbf", "random_forest"]:
    tab = pivot_metric(model_name, "balanced_accuracy")
    lines.append("-" * 88)
    lines.append(f"{model_name}  |  balanced accuracy  (chance = 0.500)")
    lines.append("-" * 88)
    lines.append(tab.round(3).fillna("--").to_string())
    lines.append("")

lines.append("-" * 88)
lines.append("PERMUTATION p-VALUES  (svm_rbf, mouse-level shuffle)")
lines.append("-" * 88)
p_tab = pivot_metric("svm_rbf", "perm_p_value_balanced_accuracy")
lines.append(p_tab.round(4).fillna("--").to_string())
lines.append("")
lines.append("Reading guide:")
lines.append("  - Cell = one (phase, band-or-ratio) model.")
lines.append("  - bal_acc > 0.5 = better than chance.")
lines.append("  - A phase-specific effect appears as one row noticeably")
lines.append("    above the others, with a small permutation p in that cell.")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 7. HEATMAP (phase x cell)
# ------------------------------------------------------------
# Bands and ratios are visually separated by a vertical line so
# it's obvious which columns are which.
# ============================================================

from matplotlib.colors import TwoSlopeNorm

def heatmap(ax, tab, title, vmin, vmax, cmap, cbar_label, ref_line=None):
    data = tab.to_numpy(dtype=float)
    # Diverging colormap centred on chance (ref_line). Red = above chance,
    # white ≈ chance, blue = below chance. Much clearer than viridis.
    if ref_line is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=ref_line, vmax=vmax)
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm)
    else:
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(CELLS, rotation=45, ha="right")
    ax.set_yticks(range(len(ESTROUS_PHASES)))
    ax.set_yticklabels(ESTROUS_PHASES)
    ax.set_xlabel("Feature cell  (bands | band-to-band ratios)")
    ax.set_ylabel("Estrous phase")
    ax.set_title(title, fontweight="bold", loc="left")
    # Text with a contrasting outline (path effect) so numbers stay
    # readable on ANY cell background — dark blue, dark red, or white.
    import matplotlib.patheffects as pe
    if ref_line is not None:
        max_dist = max(abs(vmax - ref_line), abs(vmin - ref_line))
    else:
        max_dist = max(abs(vmax), abs(vmin)) or 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center",
                        color="#666666", fontsize=9, fontweight="bold")
                continue
            if ref_line is not None:
                dist = abs(v - ref_line) / max_dist
            else:
                dist = abs(v) / max_dist
            # bright white text on dark cells, black on near-chance cells
            text_color = "white" if dist > 0.55 else "black"
            outline_color = "black" if text_color == "white" else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground=outline_color)])
    # Separator between the last band and the first ratio.
    ax.axvline(len(BANDS) - 0.5, color="black", lw=2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label)
    if ref_line is not None:
        cbar.ax.axhline(ref_line, color="grey", lw=1, ls="--")


fig, axes = plt.subplots(2, 1, figsize=(12, 8))
heatmap(axes[0],
        pivot_metric("svm_rbf", "balanced_accuracy"),
        "A. SVM-RBF — balanced accuracy",
        vmin=0.4, vmax=0.9, cmap="viridis",
        cbar_label="balanced accuracy", ref_line=0.5)
heatmap(axes[1],
        pivot_metric("random_forest", "balanced_accuracy"),
        "B. Random Forest — balanced accuracy",
        vmin=0.4, vmax=0.9, cmap="viridis",
        cbar_label="balanced accuracy", ref_line=0.5)
fig.suptitle(f"Step 10b — diet classification, phase x cell  ({CABLE})",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved heatmap:\n{OUT_PNG}")

print("\nSTEP 10B finished successfully.")
