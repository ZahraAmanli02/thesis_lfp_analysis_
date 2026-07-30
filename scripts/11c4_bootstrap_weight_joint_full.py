# ============================================================
# 11C4_BOOTSTRAP_WEIGHT_JOINT_FULL.PY
#
# Purpose:
#   RQ2 alternative approach A — instead of one regression
#   model per (phase × cell) with only 1–2 features, fit ONE
#   regression model per (phase) that uses ALL 21 features
#   (6 log-band powers, 6 relative powers, 15 log-ratios)
#   jointly.
#
#   Rationale: the per-cell versions (11c1 absolute and
#   11c1 delta) both failed with strongly negative R². A likely
#   reason is that 1–2 features per cell cannot express the
#   cross-frequency structure that carries the weight signal.
#   A joint model has access to the full feature space and
#   should capture richer patterns.
#
# Design:
#   * pool Cable 1 + Cable 3 features (same as 11c1)
#   * separate bootstrap for HFD-only and CTRL-only mice
#   * target: weight_delta (change from mouse baseline) — the
#     delta target avoids between-mouse baseline variance and
#     matches the request from the previous meeting
#   * feature vector per row = ALL 21 cell features joined
#   * mouse-cluster bootstrap, 1000 iterations, OOB evaluation
#   * RandomForestRegressor (200 trees to accommodate the
#     larger feature space)
#   * metric: R² on OOB rows
#
# Inputs:
#   Cable 1: <this project>/outputs/10a_features_Cable1/
#            10a_features_Cable1.csv
#   Cable 3: /Users/amanlizahra/Desktop/For CABLE 3/
#            thesis_lfp_analysis/outputs/10a_features_Cable3/
#            10a_features_Cable3.csv
#
# Outputs:
#   outputs/11c4_bootstrap_weight_joint_full/
#       11c4_bootstrap_weight_joint_results.csv
#       11c4_bootstrap_weight_joint_summary.txt
#       11c4_bootstrap_weight_joint_bar.png
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
TARGET = "weight_delta"
MIN_OOB_ROWS = 5
MIN_RECORDINGS_PER_PHASE = 8
MIN_MICE_PER_PHASE = 6
RF_TREES = 200

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
ESTROUS_PHASES = ["A", "B", "C", "D"]
PHASE_NAMES = {"A": "pro-estrus", "B": "estrus",
               "C": "metestrus", "D": "diestrus"}

# Full joint feature vector — 6 log-band + 6 relative + 15 log-ratio = 27
BAND_ABS_FEATS = [f"log_{b}_abs" for b in BANDS]
BAND_REL_FEATS = [f"{b}_rel" for b in BANDS]
RATIO_FEATS = [f"log_{r}" for r in RATIOS]
JOINT_FEATURES = BAND_ABS_FEATS + BAND_REL_FEATS + RATIO_FEATS

OUT_DIR = os.path.join(OUTPUT_DIR, "11c4_bootstrap_weight_joint_full")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "11c4_bootstrap_weight_joint_results.csv")
OUT_TXT = os.path.join(OUT_DIR, "11c4_bootstrap_weight_joint_summary.txt")
OUT_PNG = os.path.join(OUT_DIR, "11c4_bootstrap_weight_joint_bar.png")


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. LOAD & POOL & compute weight_delta
# ============================================================

for p in (CABLE1_CSV, CABLE3_CSV):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing 10a feature file:\n{p}")

c1 = pd.read_csv(CABLE1_CSV)
c3 = pd.read_csv(CABLE3_CSV)
pooled = pd.concat([c1, c3], ignore_index=True)
pooled["mouse_uid"] = pooled["mouse"].astype(str)
pooled = pooled[pooled["body_weight"].notna()].reset_index(drop=True)

baseline_rows = pooled[pooled["diet_phase"] == "baseline"]
baseline_by_mouse = (baseline_rows.groupby("mouse_uid")["body_weight"]
                                  .mean())
mice_missing = [m for m in pooled["mouse_uid"].unique()
                if m not in baseline_by_mouse.index]
if mice_missing:
    earliest = (pooled.sort_values("recording_date")
                      .groupby("mouse_uid")["body_weight"]
                      .first())
    for m in mice_missing:
        baseline_by_mouse[m] = earliest[m]

pooled["baseline_weight"] = pooled["mouse_uid"].map(baseline_by_mouse)
pooled["weight_delta"] = pooled["body_weight"] - pooled["baseline_weight"]

diet_rows = pooled[pooled["diet_phase"] == "diet"].reset_index(drop=True)
print(f"Cable1: {len(c1):>4d} rec")
print(f"Cable3: {len(c3):>4d} rec")
print(f"Diet-phase rows: {len(diet_rows)} "
      f"({diet_rows['mouse_uid'].nunique()} mice)")


# ============================================================
# 3. HELPERS
# ============================================================

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
# 4. MAIN LOOP — bootstrap per (subset × phase)
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)
rows = []
t0 = time.time()

print("\n" + "=" * 78)
print("FULL BOOTSTRAP  —  JOINT feature model  (RQ2 alternative A)")
print(f"features used per model: {len(JOINT_FEATURES)}")
print("=" * 78)

for subset_name in SUBSETS:
    grp_val = SUBSET_FILTER[subset_name]
    subset_df = diet_rows[diet_rows["group"] == grp_val].reset_index(drop=True)
    print(f"\n[subset {subset_name}]  n rec = {len(subset_df)}  "
          f"n mice = {subset_df['mouse_uid'].nunique()}")

    for phase in ESTROUS_PHASES:
        sub = (subset_df[subset_df["estrous_phase"] == phase]
               .reset_index(drop=True))
        ok, reason = eligible(sub)
        n_rec = len(sub)
        n_mice_phase = sub["mouse_uid"].nunique()

        header = f"  [phase {phase}]  n={n_rec}  mice={n_mice_phase}"
        if not ok:
            print(header + f"  --> skipped ({reason})")
            rows.append({
                "subset": subset_name, "phase": phase,
                "boot_mean": np.nan, "boot_median": np.nan,
                "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
                "n_valid_iters": 0, "n_skipped_iters": np.nan,
                "n_recordings": n_rec, "n_mice": n_mice_phase,
                "note": f"phase_skipped_{reason}",
            })
            continue

        # feature matrix — all 27 features jointly
        feat_block = sub[JOINT_FEATURES]
        mask = feat_block.notna().all(axis=1).to_numpy()
        X_full = feat_block.to_numpy(dtype=float)[mask]
        y_full = sub.loc[mask, TARGET].to_numpy(dtype=float)
        uid_full = sub.loc[mask, "mouse_uid"].to_numpy()
        unique_uids = np.unique(uid_full)
        n_mice_cell = len(unique_uids)
        n_rows_cell = int(mask.sum())

        if n_mice_cell < MIN_MICE_PER_PHASE:
            print(header + "  --> skipped (too few mice after NaN drop)")
            continue

        rows_by_uid = {u: np.where(uid_full == u)[0] for u in unique_uids}

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
            "boot_mean": mean_v, "boot_median": med_v,
            "boot_ci_lo": lo, "boot_ci_hi": hi,
            "n_valid_iters": int(vals.size),
            "n_skipped_iters": int(skipped_iters),
            "n_recordings": n_rows_cell, "n_mice": n_mice_cell,
            "note": "",
        })

        elapsed = time.time() - t0
        print(f"    {phase} — {PHASE_NAMES[phase]}  "
              f"R2 mean = {mean_v:+.3f}  CI = [{lo:+.2f}, {hi:+.2f}]   "
              f"skipped={skipped_iters:<3d}  ({elapsed/60:5.1f} min)")


results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved long-format results:\n{OUT_CSV}")


# ============================================================
# 5. SUMMARY (TXT)
# ============================================================

lines = []
lines.append("=" * 90)
lines.append("11C4 BOOTSTRAP — WEIGHT-CHANGE regression, JOINT feature model")
lines.append("=" * 90)
lines.append(f"Features used per model: {len(JOINT_FEATURES)} "
             f"(6 log-band abs + 6 rel + 15 log-ratio)")
lines.append(f"Target: weight_delta = body_weight - baseline (per mouse)")
lines.append(f"Rows used: diet_phase == 'diet' only")
lines.append("")
lines.append("Design:")
lines.append(f"  * pool: Cable 1 ({len(c1)} rec) + Cable 3 ({len(c3)} rec)")
lines.append(f"  * diet-phase rows kept: {len(diet_rows)}")
lines.append("  * separate bootstrap for HFD-only and CTRL-only mice")
lines.append("  * cluster bootstrap (sample mice with replacement)")
lines.append(f"  * {N_BOOTSTRAP} iterations per (subset × phase)")
lines.append(f"  * RandomForestRegressor ({RF_TREES} trees, all 27 features)")
lines.append("  * metric: R² on OOB rows (chance = 0)")
lines.append("")

for subset in SUBSETS:
    lines.append("-" * 90)
    lines.append(f"{subset}")
    lines.append("-" * 90)
    sub = results_df[results_df["subset"] == subset]
    for phase in ESTROUS_PHASES:
        row = sub[sub["phase"] == phase]
        if row.empty or pd.isna(row.iloc[0]["boot_mean"]):
            lines.append(f"  phase {phase}  --")
            continue
        r = row.iloc[0]
        lines.append(f"  phase {phase} — {PHASE_NAMES[phase]:<12}  "
                     f"n={int(r['n_recordings'])}/{int(r['n_mice'])}mice  "
                     f"R² mean = {r['boot_mean']:+.3f}  "
                     f"CI = [{r['boot_ci_lo']:+.3f}, {r['boot_ci_hi']:+.3f}]")
    lines.append("")

lines.append("=" * 90)
lines.append("READING GUIDE")
lines.append("=" * 90)
lines.append("  * R² > 0 -> joint feature model predicts weight change above baseline.")
lines.append("  * 95% CI excludes 0 -> the effect is statistically distinguishable")
lines.append("    from a mean-only predictor.")
lines.append("  * Compare to per-cell version (11c1 change): if joint model gives")
lines.append("    positive R² where per-cell gave strongly negative, this confirms")
lines.append("    the earlier failure was a feature-poverty issue, not a true absence")
lines.append("    of signal.")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved summary:\n{OUT_TXT}")


# ============================================================
# 6. BAR PLOT (HFD vs CTRL per phase)
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

phase_x = np.arange(len(ESTROUS_PHASES))
bar_w = 0.36

for i, subset in enumerate(SUBSETS):
    sub = results_df[results_df["subset"] == subset].set_index("phase")
    means = np.array([sub.loc[p, "boot_mean"]
                      if p in sub.index else np.nan
                      for p in ESTROUS_PHASES], dtype=float)
    los = np.array([sub.loc[p, "boot_ci_lo"]
                    if p in sub.index else np.nan
                    for p in ESTROUS_PHASES], dtype=float)
    his = np.array([sub.loc[p, "boot_ci_hi"]
                    if p in sub.index else np.nan
                    for p in ESTROUS_PHASES], dtype=float)

    color = "#c0392b" if subset == "HFD" else "#7f7f7f"
    x_offset = -bar_w / 2 if subset == "HFD" else bar_w / 2
    ax.bar(phase_x + x_offset, means, width=bar_w,
           color=color, alpha=0.85, edgecolor="black", linewidth=0.5,
           label=subset)
    # error bars for CI
    for j, (m, lo, hi) in enumerate(zip(means, los, his)):
        if not np.isnan(m):
            ax.plot([phase_x[j] + x_offset, phase_x[j] + x_offset],
                    [lo, hi], color="black", lw=1.2)
    for j, m in enumerate(means):
        if not np.isnan(m):
            va = "bottom" if m >= 0 else "top"
            dy = 0.03 if m >= 0 else -0.03
            ax.text(phase_x[j] + x_offset, m + dy, f"{m:+.2f}",
                    ha="center", va=va, fontsize=9, fontweight="bold")

ax.axhline(0.0, color="grey", lw=1.2, ls="--", zorder=1)
ax.set_xticks(phase_x)
ax.set_xticklabels([f"{p}\n{PHASE_NAMES[p]}" for p in ESTROUS_PHASES],
                   fontsize=10)
ax.set_ylabel(r"$R^2$   (bootstrap mean, ±95% CI)", fontsize=11)
ax.set_xlabel("Estrous phase", fontsize=11, labelpad=6)
ax.legend(loc="upper right", frameon=False, fontsize=10,
          title="Subset", title_fontsize=10)

fig.text(0.02, 0.97,
         "RQ2 alternative A — joint feature model (weight-change regression)",
         ha="left", va="top", fontsize=14, fontweight="bold")
fig.text(0.02, 0.935,
         "pooled Cable 1 + Cable 3   ·   HFD vs CTRL   ·   "
         "27 features per model (all bands + ratios)   ·   "
         f"bootstrap {N_BOOTSTRAP} iterations",
         ha="left", va="top", fontsize=9.5, color="#666", style="italic")

plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure:\n{OUT_PNG}")

print(f"\nSTEP 11C4 finished in {(time.time()-t0)/60:.1f} min.")
