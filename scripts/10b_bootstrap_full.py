# ============================================================
# 10B_BOOTSTRAP_FULL.PY
#
# Purpose:
#   Full bootstrap re-run of 10b (RQ1) 
#   supervision meeting (2026-07-09 + 2026-07-19 follow-up).
#   Pool Cable1 + Cable3 features, resample mice with
#   replacement, fit SVM-RBF + Random Forest, and report the
#   mean + 95% CI of balanced accuracy across 1000 bootstrap
#   iterations for every (estrous phase × feature cell) pair.
#
# Design (same as 10b_bootstrap_prototype, scaled up):
#   * cluster-level bootstrap — sample mice with replacement,
#     take ALL their recordings for that phase.
#   * Same physical mice recorded with both cables, so
#     mouse_uid = mouse_id. Cable1 + Cable3 rows from one
#     animal always travel together in a bootstrap draw.
#   * OOB = mice not drawn this iteration; skip iteration if
#     OOB has < MIN_OOB_ROWS rows or a single class.
#   * SVM-RBF (C=1, gamma='scale', class_weight='balanced')
#     and Random Forest (200 trees, class_weight='balanced').
#     Same hyper-params as 10b except RF trees 500 → 200 to
#     keep total runtime under ~1h; 200 is plenty for the
#     row counts we have (~40–70 per cell).
#   * N_BOOTSTRAP = 1000, ONE global RNG advancing across cells
#     so bootstrap draws in different cells are independent.
#   * Metric = balanced accuracy on OOB rows.
#   * Eligibility: MIN_RECORDINGS_PER_PHASE = 8, MIN_MICE_PER_PHASE = 6
#     (same as pooled 10b).
#
# Note (PROFESSOR call):
#   PROFESSOR suggested "200 observations, 1000 iterations". We
#   went cluster-level (17 mice, not 200 rows) because the
#   current 10b uses leave-one-mouse-out CV — row-level would
#   put the same animal in train + OOB and inflate accuracy.
#   Prototype summary explains this; will confirm with him.
#
# Inputs:
#   Cable1: <this project>/outputs/10a_features_Cable1/
#           10a_features_Cable1.csv
#   Cable3: /Users/amanlizahra/Desktop/For CABLE 3/thesis_lfp_analysis/
#           outputs/10a_features_Cable3/10a_features_Cable3.csv
#
# Outputs:
#   outputs/10b_bootstrap_full/
#       10b_bootstrap_results_long.csv   per (phase, cell, model)
#       10b_bootstrap_summary.txt        readable tables + CIs
#       10b_bootstrap_heatmap.png        phase × cell, mean bal_acc
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CABLE1_CSV = os.path.join(
    OUTPUT_DIR, "10a_features_Cable1", "10a_features_Cable1.csv"
)
CABLE3_CSV = (
    "/Users/amanlizahra/Desktop/For CABLE 3/thesis_lfp_analysis/"
    "outputs/10a_features_Cable3/10a_features_Cable3.csv"
)

RANDOM_STATE = 0
N_BOOTSTRAP = 1000
POSITIVE_GROUP = "HF"
MIN_OOB_ROWS = 5
MIN_RECORDINGS_PER_PHASE = 8
MIN_MICE_PER_PHASE = 6
RF_TREES = 200                     # reduced from 500; still stable for n<80

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

OUT_DIR = os.path.join(OUTPUT_DIR, "10b_bootstrap_full")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "10b_bootstrap_results_long.csv")
OUT_TXT = os.path.join(OUT_DIR, "10b_bootstrap_summary.txt")
OUT_PNG = os.path.join(OUT_DIR, "10b_bootstrap_heatmap.png")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. LOAD & POOL
# ============================================================

for p in (CABLE1_CSV, CABLE3_CSV):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing 10a feature file:\n{p}")

c1 = pd.read_csv(CABLE1_CSV)
c3 = pd.read_csv(CABLE3_CSV)
pooled = pd.concat([c1, c3], ignore_index=True)
pooled["mouse_uid"] = pooled["mouse"].astype(str)

print(f"Cable1: {len(c1):>4d} rec ({c1['mouse'].nunique()} mice)")
print(f"Cable3: {len(c3):>4d} rec ({c3['mouse'].nunique()} mice)")
print(f"Pooled: {len(pooled):>4d} rec ({pooled['mouse_uid'].nunique()} unique animals)")


# ============================================================
# 3. HELPERS
# ============================================================

def feature_columns_for_cell(cell_name):
    if cell_name in BANDS:
        return [f"log_{cell_name}_abs", f"{cell_name}_rel"]
    if cell_name in RATIOS:
        return [f"log_{cell_name}"]
    raise KeyError(f"Unknown cell name: {cell_name}")


def make_pipes():
    return {
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=False, class_weight="balanced",
                        random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=RF_TREES, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1)),
        ]),
    }


def eligible(sub):
    if len(sub) < MIN_RECORDINGS_PER_PHASE:
        return False, "too_few_recordings"
    if sub["mouse_uid"].nunique() < MIN_MICE_PER_PHASE:
        return False, "too_few_mice"
    if sub["group"].nunique() < 2:
        return False, "one_group_only"
    return True, "ok"


# ============================================================
# 4. MAIN LOOP — bootstrap for every (phase × cell)
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)   # advances across all cells
rows = []
t0 = time.time()

print("\n" + "=" * 78)
print("FULL BOOTSTRAP  —  phase × cell  (cluster-level, mouse_uid)")
print("=" * 78)

for phase in ESTROUS_PHASES:
    sub = pooled[pooled["estrous_phase"] == phase].reset_index(drop=True)
    ok, reason = eligible(sub)
    n_rec = len(sub)
    n_mice_phase = sub["mouse_uid"].nunique()
    n_hf = int((sub["group"] == POSITIVE_GROUP).sum())
    n_ctrl = int((sub["group"] != POSITIVE_GROUP).sum())

    header = (f"\n[phase {phase}]  n={n_rec}  mice={n_mice_phase}  "
              f"HF={n_hf}  CTRL={n_ctrl}")
    if not ok:
        print(header + f"  --> skipped ({reason})")
        for cell_name in CELLS:
            cell_type = "band" if cell_name in BANDS else "ratio"
            for model_name in ("svm_rbf", "random_forest"):
                rows.append({
                    "phase": phase, "cell": cell_name, "cell_type": cell_type,
                    "model": model_name,
                    "boot_mean": np.nan, "boot_median": np.nan,
                    "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
                    "n_valid_iters": 0, "n_skipped_iters": np.nan,
                    "n_recordings": n_rec, "n_mice": n_mice_phase,
                    "note": f"phase_skipped_{reason}",
                })
        continue
    print(header)

    for cell_name in CELLS:
        cell_type = "band" if cell_name in BANDS else "ratio"
        feat_cols = feature_columns_for_cell(cell_name)

        feat_block = sub[feat_cols]
        mask = feat_block.notna().all(axis=1).to_numpy()
        X_full = feat_block.to_numpy(dtype=float)[mask]
        y_full = (sub.loc[mask, "group"] == POSITIVE_GROUP).astype(int).to_numpy()
        uid_full = sub.loc[mask, "mouse_uid"].to_numpy()

        unique_uids = np.unique(uid_full)
        n_mice_cell = len(unique_uids)
        n_rows_cell = int(mask.sum())

        if n_mice_cell < MIN_MICE_PER_PHASE or len(np.unique(y_full)) < 2:
            for model_name in ("svm_rbf", "random_forest"):
                rows.append({
                    "phase": phase, "cell": cell_name, "cell_type": cell_type,
                    "model": model_name,
                    "boot_mean": np.nan, "boot_median": np.nan,
                    "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
                    "n_valid_iters": 0, "n_skipped_iters": np.nan,
                    "n_recordings": n_rows_cell, "n_mice": n_mice_cell,
                    "note": "cell_skipped_too_few_mice_or_one_class",
                })
            print(f"  [{cell_type:>5}] {cell_name:<22s}  skipped "
                  f"(mice={n_mice_cell}, classes={len(np.unique(y_full))})")
            continue

        rows_by_uid = {u: np.where(uid_full == u)[0] for u in unique_uids}

        boot_vals = {"svm_rbf": [], "random_forest": []}
        skipped_iters = 0

        for b in range(N_BOOTSTRAP):
            sampled = rng.choice(unique_uids, size=n_mice_cell, replace=True)
            sampled_set = set(sampled.tolist())
            oob_uids = [u for u in unique_uids if u not in sampled_set]

            train_idx = np.concatenate([rows_by_uid[u] for u in sampled])
            oob_idx = (np.concatenate([rows_by_uid[u] for u in oob_uids])
                       if oob_uids else np.array([], dtype=int))

            if len(oob_idx) < MIN_OOB_ROWS:
                skipped_iters += 1
                continue

            y_train = y_full[train_idx]
            y_oob = y_full[oob_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_oob)) < 2:
                skipped_iters += 1
                continue

            X_train = X_full[train_idx]
            X_oob = X_full[oob_idx]

            for model_name, pipe in make_pipes().items():
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_oob)
                boot_vals[model_name].append(balanced_accuracy_score(y_oob, pred))

        for model_name, vals in boot_vals.items():
            vals = np.asarray(vals, dtype=float)
            if len(vals) == 0:
                mean_v = med_v = lo = hi = np.nan
            else:
                mean_v = float(vals.mean())
                med_v = float(np.median(vals))
                lo, hi = [float(x) for x in np.percentile(vals, [2.5, 97.5])]
            rows.append({
                "phase": phase, "cell": cell_name, "cell_type": cell_type,
                "model": model_name,
                "boot_mean": mean_v, "boot_median": med_v,
                "boot_ci_lo": lo, "boot_ci_hi": hi,
                "n_valid_iters": len(vals),
                "n_skipped_iters": skipped_iters,
                "n_recordings": n_rows_cell, "n_mice": n_mice_cell,
                "note": "",
            })

        svm_mean = boot_vals["svm_rbf"] and np.mean(boot_vals["svm_rbf"])
        rf_mean = boot_vals["random_forest"] and np.mean(boot_vals["random_forest"])
        elapsed = time.time() - t0
        print(f"  [{cell_type:>5}] {cell_name:<22s}  "
              f"svm={svm_mean:.2f}  rf={rf_mean:.2f}  "
              f"skipped={skipped_iters:<3d}  ({elapsed/60:5.1f} min elapsed)")


results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved long-format results:\n{OUT_CSV}")


# ============================================================
# 5. SUMMARY (TXT)
# ============================================================

def pivot_metric(model_name, metric):
    sub = results_df[results_df["model"] == model_name]
    if sub.empty:
        return pd.DataFrame(index=ESTROUS_PHASES, columns=CELLS, dtype=float)
    return (sub.pivot(index="phase", columns="cell", values=metric)
               .reindex(index=ESTROUS_PHASES, columns=CELLS))


lines = []
lines.append("=" * 100)
lines.append("10B BOOTSTRAP FULL  —  pooled Cable1 + Cable3, cluster-level")
lines.append("=" * 100)
lines.append("")
lines.append("Design:")
lines.append(f"  * pool: Cable1 ({len(c1)} rec) + Cable3 ({len(c3)} rec) "
             f"= {len(pooled)} rec, "
             f"{pooled['mouse_uid'].nunique()} unique animals")
lines.append(f"  * cluster bootstrap: sample mice with replacement "
             f"(mouse_uid = mouse_id; same animal in both cables travels together)")
lines.append(f"  * OOB = mice not drawn this iteration")
lines.append(f"  * {N_BOOTSTRAP} iterations per (phase × cell), skip if OOB < "
             f"{MIN_OOB_ROWS} rows or single-class")
lines.append(f"  * SVM-RBF + Random Forest ({RF_TREES} trees), "
             f"class_weight='balanced'")
lines.append(f"  * metric: balanced accuracy on OOB rows")
lines.append(f"  * eligibility: min {MIN_RECORDINGS_PER_PHASE} rec, "
             f"min {MIN_MICE_PER_PHASE} mice per phase")
lines.append("")

for model_name in ("svm_rbf", "random_forest"):
    lines.append("-" * 100)
    lines.append(f"{model_name.upper()}  —  bootstrap MEAN balanced accuracy  "
                 "(chance = 0.500)")
    lines.append("-" * 100)
    tab = pivot_metric(model_name, "boot_mean")
    lines.append(tab.round(3).fillna("--").to_string())
    lines.append("")

    lines.append(f"{model_name.upper()}  —  95% percentile CI  [lo, hi]")
    lines.append("-" * 100)
    lo_tab = pivot_metric(model_name, "boot_ci_lo")
    hi_tab = pivot_metric(model_name, "boot_ci_hi")
    for phase in ESTROUS_PHASES:
        for cell in CELLS:
            lo = lo_tab.loc[phase, cell]
            hi = hi_tab.loc[phase, cell]
            lines.append(f"  phase {phase}  {cell:<22s}  "
                         f"[{lo:.3f}, {hi:.3f}]"
                         if not (pd.isna(lo) or pd.isna(hi))
                         else f"  phase {phase}  {cell:<22s}  --")
    lines.append("")

lines.append("=" * 100)
lines.append("READING GUIDE")
lines.append("=" * 100)
lines.append("  * boot_mean > 0.5 => model, on average, predicts diet better than chance.")
lines.append("  * If the 95% CI EXCLUDES 0.5, that's evidence of above-chance performance.")
lines.append("  * A phase-specific effect appears as one row whose CI clearly")
lines.append("    excludes 0.5 for a given cell while other rows straddle 0.5.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 6. HEATMAP (phase × cell)
# ============================================================

from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe


def heatmap_panel(ax, mean_tab, lo_tab, hi_tab, title,
                  vmin=0.30, vmax=0.85, ref_line=0.5):
    data = mean_tab.to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=ref_line, vmax=vmax)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(CELLS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ESTROUS_PHASES)))
    ax.set_yticklabels(ESTROUS_PHASES)
    ax.set_xlabel("Feature cell  (bands | band-to-band ratios)")
    ax.set_ylabel("Estrous phase")
    ax.set_title(title, fontweight="bold", loc="left")

    max_dist = max(abs(vmax - ref_line), abs(vmin - ref_line))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center",
                        color="#666666", fontsize=8, fontweight="bold")
                continue
            lo = lo_tab.iloc[i, j]
            hi = hi_tab.iloc[i, j]
            dist = abs(v - ref_line) / max_dist
            text_color = "white" if dist > 0.55 else "black"
            outline_color = "black" if text_color == "white" else "white"
            ax.text(j, i, f"{v:.2f}\n[{lo:.2f},{hi:.2f}]",
                    ha="center", va="center",
                    color=text_color, fontsize=7, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.6,
                                                foreground=outline_color)])

    ax.axvline(len(BANDS) - 0.5, color="black", lw=2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("mean balanced accuracy")
    cbar.ax.axhline(ref_line, color="grey", lw=1, ls="--")


fig, axes = plt.subplots(2, 1, figsize=(16, 9))
for i, model_name in enumerate(("svm_rbf", "random_forest")):
    heatmap_panel(
        axes[i],
        pivot_metric(model_name, "boot_mean"),
        pivot_metric(model_name, "boot_ci_lo"),
        pivot_metric(model_name, "boot_ci_hi"),
        f"{'A' if i == 0 else 'B'}. {model_name}  —  bootstrap mean [95% CI]",
    )
fig.suptitle(
    f"10b bootstrap — diet classification, phase × cell "
    f"(pooled Cable1 + Cable3, {N_BOOTSTRAP} iterations, cluster-level)",
    fontsize=12, fontweight="bold", y=1.00,
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved heatmap:\n{OUT_PNG}")

print(f"\nSTEP 10B BOOTSTRAP FULL finished in {(time.time()-t0)/60:.1f} min.")
