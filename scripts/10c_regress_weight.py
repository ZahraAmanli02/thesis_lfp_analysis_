# ============================================================
# 10C_REGRESS_WEIGHT.PY
#
# Purpose:
#   RQ2 revisited (PROFESSOR, 2026-07-09).
#   Can LFP band power alone predict body weight, and if so,
#   inside which (estrous phase, feature cell) does it work?
#
# Design (per PROFESSOR):
#   * one regression model per (estrous phase, feature cell)
#     - feature cells = 6 frequency bands  + 15 band-to-band ratios
#     -> 21 cells x 4 phases = 84 models per cable
#   * algorithm: Random Forest regressor (PROFESSOR named this one)
#   * feature block per model (same as 10b, so cells line up):
#         band cell   -> [log_<band>_abs, <band>_rel]    (2 features)
#         ratio cell  -> [log_<ratio>]                    (1 feature)
#   * body weight is the TARGET here (it was NOT a feature in 10b)
#   * Leave-One-Mouse-Out cross-validation
#   * permutation null shuffled at RECORDING level
#     (body weight varies within a mouse across the diet, so the
#     unit of exchange is the recording, not the mouse)
#
# Metrics:
#   R^2, MAE (in grams)
#
# Input:
#   outputs/10a_features_<CABLE>/10a_features_<CABLE>.csv
#
# Output:
#   outputs/10c_regress_weight_<CABLE>/
#       10c_results_long_<CABLE>.csv
#       10c_summary_<CABLE>.txt
#       10c_heatmap_weight_<CABLE>.png    (phase x cell, R^2)
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"                 # switch to "Cable3" for the parallel run
RANDOM_STATE = 0
N_PERMUTATIONS = 200             # 200 keeps p-value resolution at 0.005; 1000 was too slow
TARGET = "body_weight"

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

OUT_DIR = os.path.join(OUTPUT_DIR, f"10c_regress_weight_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"10c_results_long_{CABLE}.csv")
OUT_TXT = os.path.join(OUT_DIR, f"10c_summary_{CABLE}.txt")
OUT_PNG = os.path.join(OUT_DIR, f"10c_heatmap_weight_{CABLE}.png")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
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

def make_regressor():
    """Fresh Random Forest regressor pipeline.
    n_estimators reduced from 500 to 100 — with 2 features and ~25 samples per
    fold, 100 trees is already saturating. n_jobs=1 here so the outer
    cross_val_predict can parallelize LOMO folds instead (better speedup).
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("reg", RandomForestRegressor(
            n_estimators=100, n_jobs=1, random_state=RANDOM_STATE)),
    ])


logo = LeaveOneGroupOut()


def evaluate_regression(pipe, X, y, groups):
    # n_jobs=-1 parallelizes across LOMO folds (biggest speedup for LOMO
    # since we have ~15-25 folds and modern Macs have 8+ cores).
    yhat = cross_val_predict(pipe, X, y, cv=logo, groups=groups, n_jobs=-1)
    return {
        "r2": r2_score(y, yhat),
        "mae": mean_absolute_error(y, yhat),
    }


def eligible(sub):
    if len(sub) < MIN_RECORDINGS_PER_PHASE:
        return False, "too_few_recordings"
    if sub["mouse"].nunique() < MIN_MICE_PER_PHASE:
        return False, "too_few_mice"
    return True, "ok"


# ============================================================
# 5. RUN — PER PHASE x PER CELL
# ============================================================

rows = []
print("\n" + "=" * 74)
print("PER-PHASE x PER-CELL REGRESSION (body weight)")
print(f"  cells = {len(BANDS)} bands + {len(RATIOS)} ratios = {len(CELLS)}")
print("=" * 74)

for phase in ESTROUS_PHASES:
    sub = df[df["estrous_phase"] == phase].reset_index(drop=True)
    n_rec = len(sub)
    n_mice = sub["mouse"].nunique()
    ok, reason = eligible(sub)

    header = f"\n[phase {phase}]  n={n_rec}  mice={n_mice}"
    if not ok:
        print(header + f"  --> skipped ({reason})")
        rows.append({"phase": phase, "cell": "*", "cell_type": "*",
                     "model": "*", "metric": "skipped_reason",
                     "value": np.nan, "note": reason,
                     "n_recordings": n_rec, "n_mice": n_mice})
        continue
    print(header)

    groups_arr = sub["mouse"].to_numpy()
    y = sub[TARGET].to_numpy(dtype=float)

    for cell_name in CELLS:
        cell_type = "band" if cell_name in BANDS else "ratio"
        feat_cols = feature_columns_for_cell(cell_name)

        # Ratios can be NaN where the denominator was <=0 — mask them out.
        feat_block = sub[feat_cols]
        mask = feat_block.notna().all(axis=1).to_numpy()
        n_masked = int((~mask).sum())
        X = feat_block.to_numpy(dtype=float)[mask]
        y_local = y[mask]
        groups_local = groups_arr[mask]

        pipe = make_regressor()
        m = evaluate_regression(pipe, X, y_local, groups_local)
        for metric, val in m.items():
            rows.append({"phase": phase, "cell": cell_name,
                         "cell_type": cell_type, "model": "random_forest",
                         "metric": metric, "value": val,
                         "note": ("" if not n_masked
                                  else f"dropped {n_masked} rows with NaN feature"),
                         "n_recordings": int(mask.sum()), "n_mice": n_mice})

        # --- permutation null: recording-level shuffle ---
        observed_r2 = m["r2"]
        rng = np.random.default_rng(RANDOM_STATE + 1)
        null_r2 = np.empty(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            y_perm = rng.permutation(y_local)
            pipe_p = make_regressor()
            null_r2[i] = evaluate_regression(
                pipe_p, X, y_perm, groups_local)["r2"]
        p_r2 = (1 + np.sum(null_r2 >= observed_r2)) / (1 + N_PERMUTATIONS)

        rows.append({"phase": phase, "cell": cell_name, "cell_type": cell_type,
                     "model": "random_forest",
                     "metric": "perm_null_mean_r2",
                     "value": float(null_r2.mean()), "note": "",
                     "n_recordings": int(mask.sum()), "n_mice": n_mice})
        rows.append({"phase": phase, "cell": cell_name, "cell_type": cell_type,
                     "model": "random_forest",
                     "metric": "perm_p_value_r2",
                     "value": p_r2, "note": "",
                     "n_recordings": int(mask.sum()), "n_mice": n_mice})

        print(f"  [{cell_type:>5}] {cell_name:<18s} R^2={m['r2']:>6.2f}  "
              f"MAE={m['mae']:>5.2f}g  (perm p={p_r2:.3f})")


results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved long-format results:\n{OUT_CSV}")


# ============================================================
# 6. SUMMARY (TXT)
# ============================================================

def pivot_metric(metric):
    sub = results_df[(results_df["model"] == "random_forest")
                     & (results_df["metric"] == metric)]
    if sub.empty:
        return pd.DataFrame(index=ESTROUS_PHASES, columns=CELLS, dtype=float)
    return (sub.pivot(index="phase", columns="cell", values="value")
               .reindex(index=ESTROUS_PHASES, columns=CELLS))


lines = []
lines.append("=" * 88)
lines.append(f"10C REGRESSION (body weight) — {CABLE}")
lines.append("=" * 88)
lines.append("")
lines.append("Design (per PROFESSOR, 2026-07-09):")
lines.append("  * one Random Forest regressor per (estrous phase, feature cell)")
lines.append(f"  * cells = {len(BANDS)} bands + {len(RATIOS)} band-to-band ratios "
             f"= {len(CELLS)} cells per phase")
lines.append("  * feature block:")
lines.append("      band cell  -> [log_<band>_abs, <band>_rel]  (2 features)")
lines.append("      ratio cell -> [log_<ratio>]                  (1 feature)")
lines.append("  * body weight is the TARGET (never a feature)")
lines.append("  * validation = leave-one-mouse-out")
lines.append("  * permutation p-value with recording-level shuffle")
lines.append("")

lines.append("-" * 88)
lines.append("R^2  (chance = 0.0; negative = worse than mean baseline)")
lines.append("-" * 88)
lines.append(pivot_metric("r2").round(3).fillna("--").to_string())
lines.append("")
lines.append("-" * 88)
lines.append("MAE (grams)")
lines.append("-" * 88)
lines.append(pivot_metric("mae").round(3).fillna("--").to_string())
lines.append("")
lines.append("-" * 88)
lines.append("PERMUTATION p-VALUES  (recording-level shuffle)")
lines.append("-" * 88)
lines.append(pivot_metric("perm_p_value_r2").round(4).fillna("--").to_string())
lines.append("")
lines.append("Reading guide:")
lines.append("  - Cell = one (phase, band-or-ratio) model.")
lines.append("  - R^2 > 0 means the model beats predicting the mean.")
lines.append("  - Compare against 10b: a cell where weight is predictable")
lines.append("    but group is not tells us the LFP tracks physiology, not diet.")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 7. HEATMAP (phase x cell)
# ============================================================

from matplotlib.colors import TwoSlopeNorm

def heatmap(ax, tab, title, vmin, vmax, cmap, cbar_label, ref_line=None):
    data = tab.to_numpy(dtype=float)
    # Diverging RdBu_r centred on chance for defense-ready contrast.
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
    # readable on ANY cell background.
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
            text_color = "white" if dist > 0.55 else "black"
            outline_color = "black" if text_color == "white" else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground=outline_color)])
    ax.axvline(len(BANDS) - 0.5, color="black", lw=2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label)
    if ref_line is not None:
        cbar.ax.axhline(ref_line, color="grey", lw=1, ls="--")


fig, ax = plt.subplots(1, 1, figsize=(11, 4.6))
heatmap(ax,
        pivot_metric("r2"),
        "Random Forest — R^2 (body weight)",
        vmin=-0.5, vmax=0.8, cmap="magma",
        cbar_label="R^2", ref_line=0.0)
fig.suptitle(f"Step 10c — body-weight regression, phase x cell  ({CABLE})",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved heatmap:\n{OUT_PNG}")

print("\nSTEP 10C finished successfully.")
