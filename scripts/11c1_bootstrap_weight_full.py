# ============================================================
# 11C1_BOOTSTRAP_WEIGHT_FULL.PY
#
# Purpose:
#   RQ2 — pooled Cable 1 + Cable 3 bootstrap for BODY-WEIGHT
#   regression. For every (estrous phase × feature cell) cell,
#   fit a Random Forest regressor to predict body weight and
#   record the R² distribution across 1000 cluster-bootstrap
#   iterations. Separate analyses for HFD-only and CTRL-only
#   mice.
#
# Design:
#   * pool Cable 1 + Cable 3 features (10a outputs)
#   * cluster bootstrap: sample mice with replacement, take ALL
#     their recordings for the phase, evaluate on OOB mice
#   * mouse_uid = mouse_id (same animals across both cables)
#   * 1000 iterations per (subset × phase × cell)
#   * RandomForestRegressor (n_estimators = 100)
#   * metric = R² on OOB rows (chance = 0)
#   * eligibility: MIN_RECORDINGS_PER_PHASE = 8,
#                  MIN_MICE_PER_PHASE = 6, within the SUBSET
#
# Inputs:
#   Cable 1: <this project>/outputs/10a_features_Cable1/
#            10a_features_Cable1.csv
#   Cable 3: /Users/amanlizahra/Desktop/For CABLE 3/
#            thesis_lfp_analysis/outputs/10a_features_Cable3/
#            10a_features_Cable3.csv
#
# Outputs:
#   outputs/11c1_bootstrap_weight_full/
#       11c1_bootstrap_weight_results_long.csv
#       11c1_bootstrap_weight_summary.txt
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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
TARGET = "body_weight"
MIN_OOB_ROWS = 5
MIN_RECORDINGS_PER_PHASE = 8
MIN_MICE_PER_PHASE = 6
RF_TREES = 100                   

SUBSETS = ("HFD", "CTRL")
SUBSET_FILTER = {                 # rows.group value that defines each subset
    "HFD": "HF",
    "CTRL": "CTRL",
}

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

OUT_DIR = os.path.join(OUTPUT_DIR, "11c1_bootstrap_weight_full")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "11c1_bootstrap_weight_results_long.csv")
OUT_TXT = os.path.join(OUT_DIR, "11c1_bootstrap_weight_summary.txt")


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

# body_weight must be present for regression
n_missing_weight = int(pooled[TARGET].isna().sum())
if n_missing_weight:
    print(f"Dropping {n_missing_weight} rows with missing {TARGET}")
    pooled = pooled[pooled[TARGET].notna()].reset_index(drop=True)

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


def make_regressor():
    return Pipeline([
        ("scale", StandardScaler()),
        ("reg", RandomForestRegressor(
            n_estimators=RF_TREES, n_jobs=1,
            random_state=RANDOM_STATE)),
    ])


def eligible(sub):
    if len(sub) < MIN_RECORDINGS_PER_PHASE:
        return False, "too_few_recordings"
    if sub["mouse_uid"].nunique() < MIN_MICE_PER_PHASE:
        return False, "too_few_mice"
    if sub[TARGET].nunique() < 2:
        return False, "no_weight_variance"
    return True, "ok"


# ============================================================
# 4. MAIN LOOP — bootstrap per (subset × phase × cell)
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)   # advances across everything
rows = []
t0 = time.time()

print("\n" + "=" * 78)
print("FULL BOOTSTRAP — body-weight regression (RQ2)")
print("=" * 78)

for subset_name in SUBSETS:
    grp_val = SUBSET_FILTER[subset_name]
    subset_df = pooled[pooled["group"] == grp_val].reset_index(drop=True)
    print(f"\n[subset {subset_name}]  n rec = {len(subset_df)}  "
          f"n mice = {subset_df['mouse_uid'].nunique()}")

    for phase in ESTROUS_PHASES:
        sub = subset_df[subset_df["estrous_phase"] == phase].reset_index(drop=True)
        ok, reason = eligible(sub)
        n_rec = len(sub)
        n_mice_phase = sub["mouse_uid"].nunique()

        header = (f"  [phase {phase}]  n={n_rec}  mice={n_mice_phase}")
        if not ok:
            print(header + f"  --> skipped ({reason})")
            for cell_name in CELLS:
                cell_type = "band" if cell_name in BANDS else "ratio"
                rows.append({
                    "subset": subset_name, "phase": phase,
                    "cell": cell_name, "cell_type": cell_type,
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
            y_full = sub.loc[mask, TARGET].to_numpy(dtype=float)
            uid_full = sub.loc[mask, "mouse_uid"].to_numpy()

            unique_uids = np.unique(uid_full)
            n_mice_cell = len(unique_uids)
            n_rows_cell = int(mask.sum())

            if (n_mice_cell < MIN_MICE_PER_PHASE
                    or len(np.unique(y_full)) < 2):
                rows.append({
                    "subset": subset_name, "phase": phase,
                    "cell": cell_name, "cell_type": cell_type,
                    "boot_mean": np.nan, "boot_median": np.nan,
                    "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
                    "n_valid_iters": 0, "n_skipped_iters": np.nan,
                    "n_recordings": n_rows_cell, "n_mice": n_mice_cell,
                    "note": "cell_skipped_too_few_or_no_variance",
                })
                continue

            rows_by_uid = {u: np.where(uid_full == u)[0]
                           for u in unique_uids}

            boot_vals = []
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
                if np.unique(y_train).size < 2 or np.unique(y_oob).size < 2:
                    skipped_iters += 1
                    continue

                pipe = make_regressor()
                pipe.fit(X_full[train_idx], y_train)
                pred = pipe.predict(X_full[oob_idx])
                boot_vals.append(r2_score(y_oob, pred))

            vals = np.asarray(boot_vals, dtype=float)
            if vals.size == 0:
                mean_v = med_v = lo = hi = np.nan
            else:
                mean_v = float(vals.mean())
                med_v = float(np.median(vals))
                lo, hi = [float(x) for x in np.percentile(vals, [2.5, 97.5])]

            rows.append({
                "subset": subset_name, "phase": phase,
                "cell": cell_name, "cell_type": cell_type,
                "boot_mean": mean_v, "boot_median": med_v,
                "boot_ci_lo": lo, "boot_ci_hi": hi,
                "n_valid_iters": int(vals.size),
                "n_skipped_iters": int(skipped_iters),
                "n_recordings": n_rows_cell, "n_mice": n_mice_cell,
                "note": "",
            })

            elapsed = time.time() - t0
            print(f"    [{cell_type:>5}] {cell_name:<22s}  "
                  f"R2={mean_v:+.2f}  skipped={skipped_iters:<3d}  "
                  f"({elapsed/60:5.1f} min elapsed)")


results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved long-format results:\n{OUT_CSV}")


# ============================================================
# 5. SUMMARY (TXT)
# ============================================================

lines = []
lines.append("=" * 100)
lines.append("11C1 BOOTSTRAP — body-weight regression   ·   pooled Cable 1 + Cable 3")
lines.append("=" * 100)
lines.append("")
lines.append("Design:")
lines.append(f"  * pool: Cable 1 ({len(c1)} rec) + Cable 3 ({len(c3)} rec) "
             f"= {len(pooled)} rec, "
             f"{pooled['mouse_uid'].nunique()} unique animals")
lines.append("  * separate bootstrap for HFD-only and CTRL-only mice")
lines.append("  * cluster bootstrap (sample mice with replacement)")
lines.append(f"  * {N_BOOTSTRAP} iterations per (subset × phase × cell); "
             f"skip if OOB < {MIN_OOB_ROWS} rows")
lines.append(f"  * RandomForestRegressor ({RF_TREES} trees)")
lines.append("  * metric: R² on OOB rows (chance = 0)")
lines.append("")

for subset_name in SUBSETS:
    lines.append("-" * 100)
    lines.append(f"{subset_name}  —  bootstrap MEAN R²  (chance = 0.0)")
    lines.append("-" * 100)
    sub = results_df[results_df["subset"] == subset_name]
    tab = (sub.pivot(index="phase", columns="cell", values="boot_mean")
              .reindex(index=ESTROUS_PHASES, columns=CELLS))
    lines.append(tab.round(3).fillna("--").to_string())
    lines.append("")

    lines.append(f"{subset_name}  —  95% percentile CI  [lo, hi]")
    lines.append("-" * 100)
    lo_tab = (sub.pivot(index="phase", columns="cell", values="boot_ci_lo")
                 .reindex(index=ESTROUS_PHASES, columns=CELLS))
    hi_tab = (sub.pivot(index="phase", columns="cell", values="boot_ci_hi")
                 .reindex(index=ESTROUS_PHASES, columns=CELLS))
    for phase in ESTROUS_PHASES:
        for cell in CELLS:
            lo = lo_tab.loc[phase, cell]
            hi = hi_tab.loc[phase, cell]
            if pd.isna(lo) or pd.isna(hi):
                lines.append(f"  phase {phase}  {cell:<22s}  --")
            else:
                lines.append(f"  phase {phase}  {cell:<22s}  "
                             f"[{lo:+.3f}, {hi:+.3f}]")
    lines.append("")

lines.append("=" * 100)
lines.append("READING GUIDE")
lines.append("=" * 100)
lines.append("  * boot_mean > 0  ->  model predicts body weight better than the mean.")
lines.append("  * 95% CI EXCLUDES 0  ->  evidence the effect is real (not just noise).")
lines.append("  * Compare HFD vs CTRL side by side: strong signal in HFD combined")
lines.append("    with near-zero R² in CTRL is the substantive RQ2 finding — LFP")
lines.append("    tracks diet-induced weight change but not baseline stability.")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved summary:\n{OUT_TXT}")
print(f"\nSTEP 11C1 finished in {(time.time()-t0)/60:.1f} min.")
