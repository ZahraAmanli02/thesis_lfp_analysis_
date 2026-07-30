# ============================================================
# 11C1_BOOTSTRAP_WEIGHT_CHANGE_FULL.PY
#
# Purpose:
#   RQ2 revisited — predict WEIGHT CHANGE (weight_delta), not
#   absolute body weight.
#
#   The first attempt (11c1_bootstrap_weight_full.py) tried to
#   predict absolute body_weight and failed in a predictable
#   way: between-mouse baseline variance dominates the target,
#   so any bootstrap where train and OOB draw different mice
#   produces a huge, systematic offset in the predictions and
#   R² goes strongly negative.
#
#  "HFD and CTRL separately, because
#   CTRL doesn't change weight" — implicitly points to weight
#   CHANGE as the real target. Under CHANGE:
#     * every mouse starts at delta = 0 (its own baseline)
#     * HFD mice accumulate a positive delta as the diet
#       progresses -> LFP may track that trajectory
#     * CTRL mice stay near delta = 0 -> no signal to fit,
#       negative-control R² around 0 is expected
#
# Weight-delta definition:
#   weight_delta = body_weight - baseline_body_weight_of_that_mouse
#
#   Baseline body weight per mouse = mean of the mouse's rows
#   with diet_phase == "baseline". If a mouse has no baseline
#   rows, fall back to the mouse's earliest recording.
#
#   Only diet-phase rows (diet_phase == "diet") enter the
#   bootstrap — baseline rows are delta == 0 by construction
#   and add no signal; recovery rows have their own dynamics.
#
# Everything else matches 11c1_bootstrap_weight_full.py:
#   * pool Cable 1 + Cable 3 at the FEATURE level
#   * mouse_uid = mouse_id (same physical animals across cables)
#   * cluster bootstrap (sample mice with replacement)
#   * 1000 iterations per (subset × phase × cell)
#   * skip iteration if OOB has < MIN_OOB_ROWS rows
#   * RandomForestRegressor (100 trees)
#   * metric = R² on OOB rows (chance = 0)
#   * separate analyses for HFD-only and CTRL-only mice
#
# Inputs:
#   Cable 1: <this project>/outputs/10a_features_Cable1/
#            10a_features_Cable1.csv
#   Cable 3: /Users/amanlizahra/Desktop/For CABLE 3/
#            thesis_lfp_analysis/outputs/10a_features_Cable3/
#            10a_features_Cable3.csv
#
# Outputs:
#   outputs/11c1_bootstrap_weight_change_full/
#       11c1_bootstrap_weight_change_results_long.csv
#       11c1_bootstrap_weight_change_summary.txt
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
TARGET = "weight_delta"           # NEW: weight change vs mouse baseline
MIN_OOB_ROWS = 5
MIN_RECORDINGS_PER_PHASE = 8
MIN_MICE_PER_PHASE = 6
RF_TREES = 100

SUBSETS = ("HFD", "CTRL")
SUBSET_FILTER = {"HFD": "HF", "CTRL": "CTRL"}

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

OUT_DIR = os.path.join(OUTPUT_DIR, "11c1_bootstrap_weight_change_full")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR,
                       "11c1_bootstrap_weight_change_results_long.csv")
OUT_TXT = os.path.join(OUT_DIR,
                       "11c1_bootstrap_weight_change_summary.txt")


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

pooled = pooled[pooled["body_weight"].notna()].reset_index(drop=True)

print(f"Cable1: {len(c1):>4d} rec ({c1['mouse'].nunique()} mice)")
print(f"Cable3: {len(c3):>4d} rec ({c3['mouse'].nunique()} mice)")
print(f"Pooled: {len(pooled):>4d} rec ({pooled['mouse_uid'].nunique()} unique animals)")


# ============================================================
# 3. COMPUTE weight_delta = body_weight - mouse-baseline
# ============================================================

# baseline per mouse (mean of rows tagged diet_phase == "baseline")
baseline_rows = pooled[pooled["diet_phase"] == "baseline"]
baseline_by_mouse = (baseline_rows.groupby("mouse_uid")["body_weight"]
                                  .mean())

# fallback for mice with no baseline row: use the earliest recording
mice_without_baseline = [
    m for m in pooled["mouse_uid"].unique()
    if m not in baseline_by_mouse.index
]
if mice_without_baseline:
    earliest = (pooled.sort_values("recording_date")
                      .groupby("mouse_uid")["body_weight"]
                      .first())
    for m in mice_without_baseline:
        baseline_by_mouse[m] = earliest[m]

pooled["baseline_weight"] = pooled["mouse_uid"].map(baseline_by_mouse)
pooled["weight_delta"] = pooled["body_weight"] - pooled["baseline_weight"]

print("\nBaseline weight per mouse (grams):")
print(baseline_by_mouse.sort_index().round(2).to_string())

# only diet-phase rows carry a meaningful delta signal
diet_rows = pooled[pooled["diet_phase"] == "diet"].reset_index(drop=True)
print(f"\nDiet-phase rows kept for regression: {len(diet_rows)} "
      f"({diet_rows['mouse_uid'].nunique()} mice)")

# quick sanity summary of the delta distribution per subset
for subset_name in SUBSETS:
    d = diet_rows[diet_rows["group"] == SUBSET_FILTER[subset_name]]
    if len(d) == 0:
        continue
    print(f"  {subset_name}: n={len(d)}  "
          f"delta mean={d['weight_delta'].mean():+.2f} g  "
          f"range=[{d['weight_delta'].min():+.2f}, {d['weight_delta'].max():+.2f}]")

pooled = diet_rows   # everything downstream uses only diet-phase rows


# ============================================================
# 4. HELPERS
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
        return False, "no_delta_variance"
    return True, "ok"


# ============================================================
# 5. MAIN LOOP — bootstrap per (subset × phase × cell)
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)
rows = []
t0 = time.time()

print("\n" + "=" * 78)
print("FULL BOOTSTRAP — weight-CHANGE regression (RQ2)")
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
# 6. SUMMARY (TXT)
# ============================================================

lines = []
lines.append("=" * 100)
lines.append("11C1 BOOTSTRAP — WEIGHT-CHANGE regression   ·   pooled Cable 1 + Cable 3")
lines.append("=" * 100)
lines.append("")
lines.append("Target: weight_delta = body_weight - baseline_weight_of_that_mouse")
lines.append(f"  (baseline = mean weight over diet_phase=='baseline' rows per mouse)")
lines.append(f"  (analysis restricted to diet_phase=='diet' rows only)")
lines.append("")
lines.append("Design:")
lines.append(f"  * pool: Cable 1 ({len(c1)} rec) + Cable 3 ({len(c3)} rec)")
lines.append(f"  * diet-phase rows kept: {len(pooled)}   "
             f"mice: {pooled['mouse_uid'].nunique()}")
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
lines.append("  * boot_mean > 0  ->  model predicts weight_delta better than the mean.")
lines.append("  * 95% CI EXCLUDES 0  ->  evidence the effect is real (not just noise).")
lines.append("  * HFD panel = substantive test (there IS weight change to predict).")
lines.append("  * CTRL panel = negative control (weight roughly stable -> expect R² ≈ 0).")
lines.append("  * Strong HFD signal + near-zero CTRL signal in the same cell is the")
lines.append("    hallmark of a genuine diet-driven LFP-weight coupling.")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved summary:\n{OUT_TXT}")
print(f"\nSTEP 11C1 (weight-change) finished in {(time.time()-t0)/60:.1f} min.")
