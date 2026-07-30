# ============================================================
# 08B_BODYWEIGHT_PREDICTION.PY
# Purpose:
#   RQ2: Can hypothalamic LFP activity be used to PREDICT the body
#        weight of a mouse?
#
# This is the regression counterpart to 08a. Where 08a asked a
# yes/no question (which diet), here the target is continuous
# (body weight in grams), so the models are regressors and the
# metrics are R^2 and mean absolute error (MAE, in grams).
#
# Design (kept parallel to 08a for a clean thesis narrative):
#   - Unit of observation is the RECORDING (all 125 clean
#     recordings, Feb 19 - Apr 2). Same set as 08a / 08a2.
#   - Features are LFP-derived ONLY (the 29 features from 08a).
#     Body weight is the TARGET here, so it cannot be a feature;
#     diet group is also excluded (we ask what LFP itself predicts,
#     not what the experimental label implies).
#   - Leave-One-Mouse-Out (LOMO) cross-validation: predict a
#     held-out mouse's body weight from a model trained only on the
#     other mice. No mouse in train + test (same non-independence
#     control as the 06 random intercept / 08a grouping).
#   - Three models: Ridge (linear, the regression analogue of the
#     logistic model), SVR (non-linear kernel), random forest
#     regressor (non-linear ensemble).
#
# Metrics (pooled out-of-fold predictions):
#   - R^2  : fraction of body-weight variance explained
#            (0 = no better than predicting the mean; 1 = perfect;
#             negative = worse than the mean).
#   - MAE  : mean absolute error in grams (interpretable).
#   Reported against a baseline that always predicts the mean
#   weight (its MAE is the mean absolute deviation of body weight).
#
# Permutation test:
#   Body weight varies per RECORDING (it is NOT a fixed per-mouse
#   label, unlike diet group in 08a). The valid null therefore
#   shuffles the target at the recording level, breaking the LFP ->
#   weight link, and re-runs LOMO. Done for the primary model
#   (Ridge). p = (1 + #{null R^2 >= observed R^2}) / (1 + N_perm).
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_features_combined_with_episodes_<CABLE>.csv
#
# Output:
#   outputs/08b_bodyweight_prediction_<CABLE>/
#       08b_bodyweight_prediction_results_<CABLE>.csv   (long format)
#       08b_bodyweight_prediction_summary_<CABLE>.txt   (readable)
#       08b_bodyweight_prediction_<CABLE>.png           (4-panel figure)
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"
RANDOM_STATE = 0
N_PERMUTATIONS = 1000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)
COMBINED_PATH = os.path.join(
    OUTPUT_DIR, f"05c_oscillation_episodes_{CABLE}",
    f"05c_features_combined_with_episodes_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"08b_bodyweight_prediction_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"08b_bodyweight_prediction_results_{CABLE}.csv")
OUT_TXT = os.path.join(OUT_DIR, f"08b_bodyweight_prediction_summary_{CABLE}.txt")
OUT_PNG = os.path.join(OUT_DIR, f"08b_bodyweight_prediction_{CABLE}.png")

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

# --- Feature set: LFP-derived only (identical to 08a) ---
LOG_POWER_FEATS = [f"log_{b}_abs" for b in BANDS]
REL_POWER_FEATS = [f"{b}_rel" for b in BANDS]
LOG_RATIO_FEATS = [
    "log_theta_delta", "log_low_gamma_theta", "log_high_gamma_theta",
    "log_fast_gamma_theta", "log_beta_theta",
]
EPISODE_BANDS = ["beta", "low_gamma", "high_gamma"]
EPISODE_METRICS = ["episode_rate", "mean_duration_sec",
                   "mean_amplitude", "fraction_of_time"]
EPISODE_FEATS = [f"{b}_{m}" for b in EPISODE_BANDS for m in EPISODE_METRICS]
ALL_FEATURES = LOG_POWER_FEATS + REL_POWER_FEATS + LOG_RATIO_FEATS + EPISODE_FEATS

TARGET = "body_weight"

C_CTRL = "#4C72B0"
C_HF = "#C44E52"
C_ACCENT = "#1D9E75"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. LOAD + BUILD FEATURE MATRIX
# ============================================================

for p, step in [(BAND_POWERS_PATH, "05a"), (COMBINED_PATH, "05c")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {step} output:\n{p}")

df_05a = pd.read_csv(BAND_POWERS_PATH)
df_05c = pd.read_csv(COMBINED_PATH)

for b in BANDS:
    df_05a[f"log_{b}_abs"] = np.log10(df_05a[f"{b}_abs"])

merge_cols = ["differential_file"] + LOG_RATIO_FEATS + EPISODE_FEATS
df = df_05a.merge(df_05c[merge_cols], on="differential_file",
                  how="left", validate="one_to_one")

print(f"\nLoaded feature table: {df.shape[0]} recordings x {df.shape[1]} cols")

missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}")
if df[ALL_FEATURES + [TARGET]].isna().any(axis=1).sum() > 0:
    raise ValueError("NaN in features/target — check 05a/05c outputs.")


# ============================================================
# 3. FEATURES, TARGET, GROUPS
# ============================================================

X = df[ALL_FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)
groups = df["mouse"].to_numpy()

n_obs = len(y)
n_mice = len(np.unique(groups))

# Baseline: always predict the mean weight -> MAE = mean abs deviation
baseline_mae = float(np.mean(np.abs(y - y.mean())))

print(f"  {n_obs} recordings / {n_mice} mice")
print(f"  body weight: mean={y.mean():.1f} g, sd={y.std():.1f} g, "
      f"range {y.min():.1f}-{y.max():.1f} g")
print(f"  baseline MAE (predicting the mean) = {baseline_mae:.2f} g")
print(f"  features: {len(ALL_FEATURES)} (LFP only; weight is the target)")


# ============================================================
# 4. MODELS (scaling fit inside each training fold)
# ============================================================

def make_ridge():
    return Pipeline([
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])


def make_svr():
    return Pipeline([
        ("scale", StandardScaler()),
        ("reg", SVR(kernel="rbf", C=1.0, gamma="scale")),
    ])


def make_rf():
    return Pipeline([
        ("scale", StandardScaler()),   # harmless for RF; uniform interface
        ("reg", RandomForestRegressor(n_estimators=500,
                                      random_state=RANDOM_STATE)),
    ])


MODELS = {"ridge": make_ridge, "svr_rbf": make_svr,
          "random_forest": make_rf}
PRIMARY_MODEL = "ridge"


# ============================================================
# 5. EVALUATION HELPERS
# ============================================================

logo = LeaveOneGroupOut()


def pooled_pred(make_model, X, y, groups):
    """Pooled out-of-fold predictions under LOMO."""
    return cross_val_predict(make_model(), X, y, cv=logo, groups=groups)


def score(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


# ============================================================
# 6. RUN EACH MODEL (LOMO)
# ============================================================

print("\n" + "=" * 60)
print("LEAVE-ONE-MOUSE-OUT REGRESSION (body weight)")
print("=" * 60)

rows = []
pred_store = {}

for model_name, make_model in MODELS.items():
    y_pred = pooled_pred(make_model, X, y, groups)
    m = score(y, y_pred)
    pred_store[model_name] = y_pred
    for metric, val in m.items():
        rows.append({"model": model_name, "metric": metric, "value": val})
    print(f"  {model_name:<16s} R2={m['r2']:+.3f}   MAE={m['mae']:.2f} g "
          f"(baseline {baseline_mae:.2f} g)")

results_df = pd.DataFrame(rows)


# ============================================================
# 7. PERMUTATION TEST (recording-level target shuffle, Ridge)
# ------------------------------------------------------------
# Body weight is a per-recording quantity (not a fixed per-mouse
# label), so the valid null shuffles the target across recordings,
# severing the LFP->weight link, and re-runs LOMO. A positive R^2
# that beats this null indicates a genuine LFP->weight signal.
# ============================================================

print("\n" + "=" * 60)
print(f"PERMUTATION TEST (Ridge, LOMO, {N_PERMUTATIONS} shuffles)")
print("=" * 60)

observed_r2 = results_df.query(
    "model == @PRIMARY_MODEL and metric == 'r2'")["value"].iloc[0]

rng = np.random.default_rng(RANDOM_STATE)
null_r2 = np.empty(N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    y_perm = rng.permutation(y)
    y_pred_p = pooled_pred(make_ridge, X, y_perm, groups)
    null_r2[i] = r2_score(y_perm, y_pred_p)
    if (i + 1) % 200 == 0:
        print(f"  ...{i + 1}/{N_PERMUTATIONS} done")

perm_p = (1 + np.sum(null_r2 >= observed_r2)) / (1 + N_PERMUTATIONS)
null_mean = float(null_r2.mean())

print(f"\n  observed R2      = {observed_r2:+.3f}")
print(f"  null-mean R2     = {null_mean:+.3f}")
print(f"  permutation p    = {perm_p:.4f}")

rows.append({"model": PRIMARY_MODEL, "metric": "perm_null_mean_r2",
             "value": null_mean})
rows.append({"model": PRIMARY_MODEL, "metric": "perm_p_value_r2",
             "value": perm_p})
results_df = pd.DataFrame(rows)


# ============================================================
# 8. SAVE — LONG-FORMAT CSV
# ============================================================

results_df = results_df[["model", "metric", "value"]]
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved results CSV:\n{OUT_CSV}")


# ============================================================
# 9. SAVE — HUMAN-READABLE TXT
# ============================================================

def get(model, metric):
    q = results_df.query("model == @model and metric == @metric")
    return q["value"].iloc[0] if len(q) else float("nan")

lines = []
lines.append("=" * 70)
lines.append(f"08B BODY-WEIGHT PREDICTION FROM LFP — {CABLE}")
lines.append("=" * 70)
lines.append("")
lines.append(f"Observations : {n_obs} recordings / {n_mice} mice")
lines.append(f"Target       : body weight (mean {y.mean():.1f} g, "
             f"range {y.min():.1f}-{y.max():.1f} g)")
lines.append(f"Features     : {len(ALL_FEATURES)} LFP-derived (weight is "
             f"the target; diet group excluded)")
lines.append(f"Validation   : leave-one-mouse-out")
lines.append(f"Baseline MAE (predict the mean) : {baseline_mae:.2f} g")
lines.append("")
lines.append("-" * 70)
lines.append(f"{'model':<16s} {'R2':>8s} {'MAE (g)':>10s} "
             f"{'vs baseline':>12s}")
lines.append("-" * 70)
for model in MODELS:
    mae = get(model, "mae")
    lines.append(f"{model:<16s} {get(model, 'r2'):>+8.3f} {mae:>10.2f} "
                 f"{mae - baseline_mae:>+12.2f}")
lines.append("")
lines.append("-" * 70)
lines.append("PERMUTATION TEST (Ridge, LOMO)")
lines.append("-" * 70)
lines.append(f"  observed R2   : {observed_r2:+.3f}")
lines.append(f"  null-mean R2  : {null_mean:+.3f}")
lines.append(f"  p-value ({N_PERMUTATIONS}) : {perm_p:.4f}")
lines.append("")
lines.append("Reading guide:")
lines.append("  - R2 <= 0 means LFP does not predict weight better than just")
lines.append("    guessing the average weight (MAE >= baseline).")
lines.append("  - MAE is in grams: 'on average we are off by this much'.")
lines.append("  - A positive R2 that beats the permutation null would be")
lines.append("    evidence of a genuine LFP -> body-weight relationship.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 10. FIGURE (4 panels)
#   A) predicted vs actual weight (primary model), coloured by group
#   B) R^2 per model
#   C) MAE per model vs baseline
#   D) permutation null for R^2 (primary model)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# --- A: predicted vs actual (primary) ---
ax = axes[0, 0]
y_pred_primary = pred_store[PRIMARY_MODEL]
is_hf = (df["group"] == "HF").to_numpy()
ax.scatter(y[~is_hf], y_pred_primary[~is_hf], s=32, alpha=0.7,
           color=C_CTRL, edgecolors="white", linewidths=0.5, label="CTRL")
ax.scatter(y[is_hf], y_pred_primary[is_hf], s=32, alpha=0.7,
           color=C_HF, edgecolors="white", linewidths=0.5, label="HF")
lims = [min(y.min(), y_pred_primary.min()) - 1,
        max(y.max(), y_pred_primary.max()) + 1]
ax.plot(lims, lims, color="grey", ls="--", lw=1.0, label="perfect")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual body weight (g)")
ax.set_ylabel("Predicted body weight (g)")
ax.set_title(f"A. Predicted vs actual ({PRIMARY_MODEL}, LOMO)\n"
             f"R2={get(PRIMARY_MODEL, 'r2'):+.2f}, "
             f"MAE={get(PRIMARY_MODEL, 'mae'):.2f} g",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")

# --- B: R2 per model ---
ax = axes[0, 1]
model_list = list(MODELS.keys())
r2_vals = [get(m, "r2") for m in model_list]
colors = ["#C44E52", "#8172B3", "#4C72B0"]
ax.bar(range(len(model_list)), r2_vals, color=colors, alpha=0.9)
ax.axhline(0, color="grey", ls="--", lw=0.8, label="R2 = 0 (mean baseline)")
ax.set_xticks(range(len(model_list)))
ax.set_xticklabels([m.replace("_", "\n") for m in model_list])
ax.set_ylabel("R2 (LOMO)")
ax.set_title("B. Variance explained by model", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8)

# --- C: MAE per model vs baseline ---
ax = axes[1, 0]
mae_vals = [get(m, "mae") for m in model_list]
ax.bar(range(len(model_list)), mae_vals, color=colors, alpha=0.9)
ax.axhline(baseline_mae, color="black", ls="--", lw=1.0,
           label=f"baseline (predict mean) = {baseline_mae:.2f} g")
ax.set_xticks(range(len(model_list)))
ax.set_xticklabels([m.replace("_", "\n") for m in model_list])
ax.set_ylabel("MAE (g)")
ax.set_title("C. Mean absolute error by model", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8)

# --- D: permutation null for R^2 ---
ax = axes[1, 1]
ax.hist(null_r2, bins=30, color="#B0B0B0", alpha=0.85,
        label="null (shuffled weight)")
ax.axvline(null_mean, color="grey", ls="--", lw=1.2,
           label=f"null mean ({null_mean:+.2f})")
ax.axvline(observed_r2, color=C_HF, lw=2.0,
           label=f"observed ({observed_r2:+.2f})")
ax.set_xlabel("R2")
ax.set_ylabel("Permutations")
ax.set_title(f"D. Permutation null (p={perm_p:.3f})",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8)

fig.suptitle(f"Step 08b — body-weight prediction from LFP ({CABLE}, "
             f"{n_obs} recordings / {n_mice} mice)",
             fontsize=13, fontweight="bold", y=1.00)
fig.text(0.5, 0.005,
         "LFP-only features, leave-one-mouse-out. R2 <= 0 means LFP does not "
         "beat simply predicting the average weight.",
         ha="center", fontsize=9, color="#444444", style="italic")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved figure:\n{OUT_PNG}")

print("\nSTEP 08B (body-weight prediction) finished successfully.")