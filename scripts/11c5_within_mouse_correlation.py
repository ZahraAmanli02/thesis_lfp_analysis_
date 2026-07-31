# ============================================================
# 11C5_WITHIN_MOUSE_CORRELATION.PY
#
# Purpose:
#   RQ2 alternative approach — within-mouse correlation.
#
#   Instead of cross-mouse regression, this
#   script asks a per-animal question:
#
#     "For this individual HFD mouse, does the value of a
#      given LFP feature co-vary with the mouse's own weight
#      gain across its diet-phase recordings?"
#
#   The between-mouse variance is removed by construction —
#   each mouse is compared only to itself. If a feature is
#   truly informative for HFD-induced weight change, we should
#   see a consistent same-sign correlation in most mice.
#
# Method:
#   For each HFD mouse with at least MIN_REC_PER_MOUSE
#   diet-phase recordings:
#     For each of the 27 LFP features (6 log-band abs, 6
#     relative, 15 log-ratio):
#       Spearman correlation between the feature value and
#       weight_delta across the mouse's diet-phase recordings.
#
#   Then aggregate across mice per feature:
#     * median within-mouse correlation
#     * sign test: how many mice show positive correlation?
#     * two-sided binomial p-value against 0.5 (chance)
#
# Inputs:
#   Cable 1: <this project>/outputs/10a_features_Cable1/
#            10a_features_Cable1.csv
#   Cable 3: /Users/amanlizahra/Desktop/For CABLE 3/
#            thesis_lfp_analysis/outputs/10a_features_Cable3/
#            10a_features_Cable3.csv
#
# Outputs:
#   outputs/11c5_within_mouse_correlation/
#       11c5_within_mouse_correlations.csv   long-format
#       11c5_within_mouse_summary.txt        median + sign-test
#       11c5_within_mouse_bar.png            median r per feature
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, binomtest

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

TARGET = "weight_delta"
SUBSET_GROUP = "HF"                # HFD only 
MIN_REC_PER_MOUSE = 4              # need enough diet-phase recordings per mouse for a stable within-mouse correlation
ALPHA = 0.05

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
RATIOS = [
    "theta_delta", "beta_delta", "low_gamma_delta",
    "high_gamma_delta", "fast_gamma_delta",
    "beta_theta", "low_gamma_theta", "high_gamma_theta", "fast_gamma_theta",
    "low_gamma_beta", "high_gamma_beta", "fast_gamma_beta",
    "high_gamma_low_gamma", "fast_gamma_low_gamma", "fast_gamma_high_gamma",
]
BAND_ABS_FEATS = [f"log_{b}_abs" for b in BANDS]
BAND_REL_FEATS = [f"{b}_rel" for b in BANDS]
RATIO_FEATS = [f"log_{r}" for r in RATIOS]
FEATURES = BAND_ABS_FEATS + BAND_REL_FEATS + RATIO_FEATS

OUT_DIR = os.path.join(OUTPUT_DIR, "11c5_within_mouse_correlation")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "11c5_within_mouse_correlations.csv")
OUT_TXT = os.path.join(OUT_DIR, "11c5_within_mouse_summary.txt")
OUT_PNG = os.path.join(OUT_DIR, "11c5_within_mouse_bar.png")


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def format_feat(feat):
    if feat.startswith("log_") and feat.endswith("_abs"):
        return feat[4:-4] + " (log abs)"
    if feat.endswith("_rel"):
        return feat[:-4] + " (rel)"
    if feat.startswith("log_"):
        stem = feat[4:]
        for b in sorted(BANDS, key=len, reverse=True):
            if stem.startswith(b + "_"):
                rest = stem[len(b) + 1:]
                if rest in BANDS:
                    return f"{b} / {rest}"
    return feat


# ============================================================
# 2. LOAD, POOL, weight_delta
# ============================================================

for p in (CABLE1_CSV, CABLE3_CSV):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing 10a feature file:\n{p}")

c1 = pd.read_csv(CABLE1_CSV)
c3 = pd.read_csv(CABLE3_CSV)
pooled = pd.concat([c1, c3], ignore_index=True)
pooled["mouse_uid"] = pooled["mouse"].astype(str)
pooled = pooled[pooled["body_weight"].notna()].reset_index(drop=True)

# baseline per mouse (mean of rows with diet_phase == "baseline")
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

# only diet-phase HFD rows enter the analysis
hfd_diet = pooled[(pooled["diet_phase"] == "diet")
                  & (pooled["group"] == SUBSET_GROUP)].reset_index(drop=True)

print(f"Pooled: {len(pooled)} rec, HFD diet-phase rows: {len(hfd_diet)}")
print(f"HFD mice: {hfd_diet['mouse_uid'].nunique()}")


# ============================================================
# 3. WITHIN-MOUSE SPEARMAN CORRELATIONS
# ============================================================

records = []
for uid, grp in hfd_diet.groupby("mouse_uid"):
    n_rec = len(grp)
    if n_rec < MIN_REC_PER_MOUSE:
        continue
    y = grp[TARGET].to_numpy(dtype=float)
    if np.unique(y).size < 2:
        continue
    for feat in FEATURES:
        if feat not in grp.columns:
            continue
        x = grp[feat].to_numpy(dtype=float)
        if np.isnan(x).any() or np.unique(x).size < 2:
            continue
        r, p = spearmanr(x, y)
        records.append({
            "mouse_uid": uid,
            "feature": feat,
            "n_rec": n_rec,
            "spearman_r": float(r),
            "spearman_p": float(p),
        })

corr_df = pd.DataFrame(records)
corr_df.to_csv(OUT_CSV, index=False)

n_mice_used = corr_df["mouse_uid"].nunique()
print(f"\nMice with ≥ {MIN_REC_PER_MOUSE} diet recordings: {n_mice_used}")
print(f"Total within-mouse correlations computed: {len(corr_df)}")


# ============================================================
# 4. AGGREGATE PER FEATURE — median, sign test
# ============================================================

feature_stats = []
for feat in FEATURES:
    sub = corr_df[corr_df["feature"] == feat]
    n = len(sub)
    if n == 0:
        continue
    r_vals = sub["spearman_r"].to_numpy()
    median_r = float(np.median(r_vals))
    mean_r = float(np.mean(r_vals))
    n_positive = int((r_vals > 0).sum())
    n_negative = int((r_vals < 0).sum())
    # two-sided binomial sign test against 0.5
    n_nonzero = n_positive + n_negative
    if n_nonzero == 0:
        sign_p = 1.0
    else:
        successes = max(n_positive, n_negative)
        sign_p = float(binomtest(successes, n_nonzero, p=0.5,
                                 alternative="two-sided").pvalue)
    feature_stats.append({
        "feature": feat,
        "n_mice": n,
        "median_r": median_r,
        "mean_r": mean_r,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "sign_p": sign_p,
    })

stats_df = pd.DataFrame(feature_stats).sort_values("median_r",
                                                   ascending=False)


# ============================================================
# 5. SUMMARY (TXT)
# ============================================================

lines = []
lines.append("=" * 100)
lines.append("11C5 WITHIN-MOUSE CORRELATION — HFD only")
lines.append("=" * 100)
lines.append(f"Question: for individual HFD mice, does an LFP feature "
             "co-vary with the mouse's own weight change?")
lines.append("")
lines.append(f"Pool: {len(hfd_diet)} HFD diet-phase recordings, "
             f"{hfd_diet['mouse_uid'].nunique()} mice.")
lines.append(f"Mice with ≥ {MIN_REC_PER_MOUSE} recordings "
             f"(used in this analysis): {n_mice_used}")
lines.append(f"Correlations computed: {len(corr_df)} "
             f"(mice × features).")
lines.append("")
lines.append(f"For each mouse × feature: Spearman correlation between "
             f"the feature value and weight_delta across that mouse's "
             f"diet-phase recordings.")
lines.append("")
lines.append("Interpretation:")
lines.append("  * median_r > 0 across mice AND sign-test p < 0.05")
lines.append("    -> feature consistently tracks weight change within animal.")
lines.append("  * strong same-sign consistency (e.g. 8/9 mice positive) is")
lines.append("    the substantive finding — cross-mouse regression failed but")
lines.append("    within-mouse coupling can still be real.")
lines.append("")

lines.append("-" * 100)
lines.append(f"{'feature':<28} {'n_mice':>7} {'median_r':>10} "
             f"{'mean_r':>9} {'n+':>4} {'n-':>4} {'sign_p':>9} sig?")
lines.append("-" * 100)
for _, row in stats_df.iterrows():
    star = " *" if row["sign_p"] < ALPHA else ""
    lines.append(f"{format_feat(row['feature']):<28} "
                 f"{int(row['n_mice']):>7d} "
                 f"{row['median_r']:>+10.3f} "
                 f"{row['mean_r']:>+9.3f} "
                 f"{int(row['n_positive']):>4d} "
                 f"{int(row['n_negative']):>4d} "
                 f"{row['sign_p']:>9.4f}{star}")

lines.append("")
n_sig = int((stats_df["sign_p"] < ALPHA).sum())
lines.append(f"Features with significant same-sign consistency "
             f"(sign-test p < {ALPHA}): {n_sig} / {len(stats_df)}")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved summary:\n{OUT_TXT}")


# ============================================================
# 6. BAR PLOT — median within-mouse correlation per feature
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))
stats_sorted = stats_df.sort_values("median_r", ascending=True)

y_pos = np.arange(len(stats_sorted))
colors = []
edge_widths = []
for _, row in stats_sorted.iterrows():
    if row["sign_p"] < ALPHA and row["median_r"] > 0:
        colors.append("#c0392b")
        edge_widths.append(1.4)
    elif row["sign_p"] < ALPHA and row["median_r"] < 0:
        colors.append("#2c6bad")
        edge_widths.append(1.4)
    else:
        colors.append("#b0b0b0")
        edge_widths.append(0.5)

for i, (_, row) in enumerate(stats_sorted.iterrows()):
    ax.barh(i, row["median_r"], height=0.7,
            color=colors[i], edgecolor="black",
            linewidth=edge_widths[i], alpha=0.85, zorder=2)

# individual-mouse dots overlaid
for i, (_, row) in enumerate(stats_sorted.iterrows()):
    r_vals = corr_df[corr_df["feature"] == row["feature"]]["spearman_r"]
    rng = np.random.default_rng(0)
    y_jitter = i + rng.uniform(-0.12, 0.12, size=len(r_vals))
    ax.scatter(r_vals, y_jitter, s=14, color="black",
               alpha=0.5, zorder=3)

ax.axvline(0.0, color="grey", lw=1.2, ls="--", zorder=1)
ax.set_yticks(y_pos)
ax.set_yticklabels([format_feat(f) for f in stats_sorted["feature"]],
                   fontsize=9)
ax.set_xlabel("Within-mouse Spearman r  (feature vs weight_delta)",
              fontsize=11, labelpad=6)
ax.set_xlim(-1.0, 1.0)

fig.text(0.02, 0.97,
         "RQ2 alternative C — within-mouse LFP–weight correlation (HFD)",
         ha="left", va="top", fontsize=14, fontweight="bold")
fig.text(0.02, 0.945,
         f"{n_mice_used} HFD mice with ≥ {MIN_REC_PER_MOUSE} diet recordings   ·   "
         "bars = median across mice, dots = individual mice   ·   "
         f"red / blue = sign-test p < {ALPHA}",
         ha="left", va="top", fontsize=9.5, color="#666", style="italic")

plt.subplots_adjust(left=0.24, right=0.98, top=0.93, bottom=0.07)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure:\n{OUT_PNG}")

print("\nSTEP 11C5 (within-mouse correlation) finished.")
