# ============================================================
# 08A2_DIET_CLASSIFICATION_WEIGHT_COMPARISON.PY
# Purpose:
# Companion to 08a. 08a answered RQ1 with LFP-only features and
# found near-chance classification. This script completes the RQ1
# answer by asking WHERE the discriminative information actually
# lives, using three feature sets under the SAME valid mouse-
# grouped cross-validation:
#
#   (1) lfp_only        : the 29 LFP features from 08a (no weight)
#   (2) weight_only     : body weight alone (1 feature)
#   (3) lfp_plus_weight : the 29 LFP features + body weight
#
# The logic: HF mice weigh more by construction, so if body weight
# carries the diet signal, weight_only should classify well while
# lfp_only stays at chance, and lfp_plus_weight should look like
# weight_only (LFP adds little). That pattern shows the separable
# information is metabolic (weight), not in the LFP itself —
# consistent with Step 06, where body weight had to be controlled
# because it masks the diet signal.
#
# Three classifiers are run on every feature set (as requested):
#   - logistic_regression : linear, L2-regularised (interpretable)
#   - svm_rbf             : non-linear kernel SVM
#   - random_forest       : non-linear tree ensemble
#
# Validation:
#   - Leave-One-Mouse-Out (LOMO): no mouse in train + test.
#   - Metrics from pooled out-of-fold predictions: balanced
#     accuracy, ROC-AUC, accuracy.
#   - Permutation test (mouse-level label shuffle) on the primary
#     model (logistic regression) for each feature set, giving an
#     empirical chance level and a p-value.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_features_combined_with_episodes_<CABLE>.csv
#
# Output:
#   outputs/08a2_diet_classification_weight_<CABLE>/
#       08a2_diet_classification_weight_results_<CABLE>.csv   (long)
#       08a2_diet_classification_weight_summary_<CABLE>.txt   (readable)
#       08a2_diet_classification_weight_<CABLE>.png           (figure)
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             accuracy_score, roc_curve)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"
RANDOM_STATE = 0
N_PERMUTATIONS = 1000
POSITIVE_GROUP = "HF"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)
COMBINED_PATH = os.path.join(
    OUTPUT_DIR, f"05c_oscillation_episodes_{CABLE}",
    f"05c_features_combined_with_episodes_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"08a2_diet_classification_weight_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(
    OUT_DIR, f"08a2_diet_classification_weight_results_{CABLE}.csv")
OUT_TXT = os.path.join(
    OUT_DIR, f"08a2_diet_classification_weight_summary_{CABLE}.txt")
OUT_PNG = os.path.join(
    OUT_DIR, f"08a2_diet_classification_weight_{CABLE}.png")

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

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

LFP_FEATURES = LOG_POWER_FEATS + REL_POWER_FEATS + LOG_RATIO_FEATS + EPISODE_FEATS
WEIGHT_FEATURE = ["body_weight"]

# The three feature sets compared in this script
FEATURE_SETS = {
    "lfp_only":        LFP_FEATURES,
    "weight_only":     WEIGHT_FEATURE,
    "lfp_plus_weight": LFP_FEATURES + WEIGHT_FEATURE,
}

C_SETS = {"lfp_only": "#4C72B0", "weight_only": "#DD8452",
          "lfp_plus_weight": "#1D9E75"}

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

all_needed = sorted(set(LFP_FEATURES + WEIGHT_FEATURE))
missing = [c for c in all_needed if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}")
if df[all_needed].isna().any(axis=1).sum() > 0:
    raise ValueError("NaN in features — check 05a/05c outputs.")


# ============================================================
# 3. TARGET / GROUPS
# ============================================================

y = (df["group"] == POSITIVE_GROUP).astype(int).to_numpy()
groups = df["mouse"].to_numpy()
n_obs, n_mice = len(y), len(np.unique(groups))
n_hf, n_ctrl = int(y.sum()), int((1 - y).sum())
majority_acc = max(n_hf, n_ctrl) / n_obs

print(f"  {n_obs} recordings / {n_mice} mice (HF {n_hf} / CTRL {n_ctrl})")
print(f"  majority-class accuracy = {majority_acc:.3f}")


# ============================================================
# 4. MODELS (scaling fit inside each training fold)
# ============================================================

def make_logreg():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=5000,
                                   class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])


def make_svm():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                    class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def make_rf():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=500,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)),
    ])


MODELS = {"logistic_regression": make_logreg,
          "svm_rbf": make_svm,
          "random_forest": make_rf}


# ============================================================
# 5. EVALUATION HELPERS
# ============================================================

logo = LeaveOneGroupOut()


def pooled_predictions(make_model, X, y, groups):
    model = make_model()
    proba = cross_val_predict(model, X, y, cv=logo, groups=groups,
                              method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proba, pred


def score(y_true, proba, pred):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
    }


# ============================================================
# 6. RUN EVERY MODEL ON EVERY FEATURE SET (LOMO)
# ============================================================

print("\n" + "=" * 64)
print("LEAVE-ONE-MOUSE-OUT CLASSIFICATION (feature set x model)")
print("=" * 64)

rows = []
proba_store = {}   # (feature_set, model) -> pooled proba (for figure)

for fs_name, cols in FEATURE_SETS.items():
    X = df[cols].to_numpy(dtype=float)
    print(f"\n[{fs_name}]  ({len(cols)} feature"
          f"{'s' if len(cols) > 1 else ''})")
    for model_name, make_model in MODELS.items():
        proba, pred = pooled_predictions(make_model, X, y, groups)
        m = score(y, proba, pred)
        proba_store[(fs_name, model_name)] = proba
        for metric, val in m.items():
            rows.append({"feature_set": fs_name, "model": model_name,
                         "metric": metric, "value": val})
        print(f"  {model_name:<20s} bal_acc={m['balanced_accuracy']:.3f}"
              f"  auc={m['roc_auc']:.3f}  acc={m['accuracy']:.3f}")

results_df = pd.DataFrame(rows)


# ============================================================
# 7. PERMUTATION TEST (mouse-level shuffle, logistic, per set)
# ============================================================

print("\n" + "=" * 64)
print(f"PERMUTATION TEST (logistic regression, LOMO, "
      f"{N_PERMUTATIONS} shuffles per feature set)")
print("=" * 64)

mouse_series = df.groupby("mouse")["group"].first()
mouse_ids = mouse_series.index.to_numpy()
mouse_labels = (mouse_series == POSITIVE_GROUP).astype(int).to_numpy()

perm_summary = {}
for fs_name, cols in FEATURE_SETS.items():
    X = df[cols].to_numpy(dtype=float)
    observed = results_df.query(
        "feature_set == @fs_name and model == 'logistic_regression' "
        "and metric == 'balanced_accuracy'")["value"].iloc[0]

    rng = np.random.default_rng(RANDOM_STATE)
    null = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        permuted = rng.permutation(mouse_labels)
        mapping = dict(zip(mouse_ids, permuted))
        y_perm = np.array([mapping[g] for g in groups])
        _, pred_p = pooled_predictions(make_logreg, X, y_perm, groups)
        null[i] = balanced_accuracy_score(y_perm, pred_p)

    p_val = (1 + np.sum(null >= observed)) / (1 + N_PERMUTATIONS)
    perm_summary[fs_name] = {"observed": observed,
                             "null_mean": float(null.mean()),
                             "p_value": p_val, "null": null}
    print(f"  {fs_name:<16s} observed={observed:.3f}  "
          f"chance={null.mean():.3f}  p={p_val:.4f}")

    rows.append({"feature_set": fs_name, "model": "logistic_regression",
                 "metric": "perm_null_mean_balanced_accuracy",
                 "value": float(null.mean())})
    rows.append({"feature_set": fs_name, "model": "logistic_regression",
                 "metric": "perm_p_value_balanced_accuracy",
                 "value": p_val})

results_df = pd.DataFrame(rows)


# ============================================================
# 8. SAVE — LONG-FORMAT CSV
# ============================================================

results_df = results_df[["feature_set", "model", "metric", "value"]]
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved results CSV:\n{OUT_CSV}")


# ============================================================
# 9. SAVE — HUMAN-READABLE TXT
# ============================================================

def get(fs, model, metric):
    q = results_df.query("feature_set == @fs and model == @model "
                         "and metric == @metric")
    return q["value"].iloc[0] if len(q) else float("nan")

lines = []
lines.append("=" * 72)
lines.append(f"08A2 DIET CLASSIFICATION — WEIGHT CONTROL COMPARISON — {CABLE}")
lines.append("=" * 72)
lines.append("")
lines.append(f"Observations : {n_obs} recordings / {n_mice} mice "
             f"(HF {n_hf} / CTRL {n_ctrl})")
lines.append(f"Validation   : leave-one-mouse-out")
lines.append(f"Chance / majority-class accuracy : 0.500 / {majority_acc:.3f}")
lines.append("")
lines.append("-" * 72)
lines.append(f"{'feature set':<18s} {'model':<22s} {'bal_acc':>8s}"
             f" {'roc_auc':>8s} {'acc':>7s}")
lines.append("-" * 72)
for fs in FEATURE_SETS:
    for model in MODELS:
        lines.append(f"{fs:<18s} {model:<22s} "
                     f"{get(fs, model, 'balanced_accuracy'):>8.3f}"
                     f" {get(fs, model, 'roc_auc'):>8.3f}"
                     f" {get(fs, model, 'accuracy'):>7.3f}")
    lines.append("")
lines.append("-" * 72)
lines.append("PERMUTATION TEST (logistic regression, LOMO)")
lines.append("-" * 72)
lines.append(f"{'feature set':<18s} {'observed':>9s} {'chance':>8s} {'p':>8s}")
for fs in FEATURE_SETS:
    ps = perm_summary[fs]
    lines.append(f"{fs:<18s} {ps['observed']:>9.3f} {ps['null_mean']:>8.3f}"
                 f" {ps['p_value']:>8.4f}")
lines.append("")
lines.append("Reading guide:")
lines.append("  - lfp_only near chance  -> LFP alone does not separate diet.")
lines.append("  - weight_only high      -> body weight carries the signal.")
lines.append("  - lfp_plus_weight ~ weight_only -> LFP adds little on top of")
lines.append("    weight; the separable information is metabolic, not in LFP.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 10. FIGURE (2 panels)
#   A) grouped bars: balanced accuracy, feature set x model
#   B) ROC (logistic), the three feature sets overlaid
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# --- A: grouped bars ---
ax = axes[0]
fs_list = list(FEATURE_SETS.keys())
model_list = list(MODELS.keys())
x = np.arange(len(fs_list))
w = 0.26
model_colors = {"logistic_regression": "#C44E52",
                "svm_rbf": "#8172B3",
                "random_forest": "#4C72B0"}
for j, model in enumerate(model_list):
    vals = [get(fs, model, "balanced_accuracy") for fs in fs_list]
    ax.bar(x + (j - 1) * w, vals, w, color=model_colors[model],
           alpha=0.9, label=model.replace("_", " "))
ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels([fs.replace("_", "\n") for fs in fs_list])
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0, 1)
ax.set_title("A. Balanced accuracy by feature set (LOMO)",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=2)

# --- B: ROC, logistic, three feature sets ---
ax = axes[1]
for fs in fs_list:
    proba = proba_store[(fs, "logistic_regression")]
    fpr, tpr, _ = roc_curve(y, proba)
    auc = get(fs, "logistic_regression", "roc_auc")
    ax.plot(fpr, tpr, color=C_SETS[fs], lw=1.9,
            label=f"{fs.replace('_', ' ')} (AUC={auc:.2f})")
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("B. ROC — logistic regression (LOMO)",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle(f"Step 08a2 — where the diet signal lives ({CABLE}, "
             f"{n_obs} recordings / {n_mice} mice)",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.5, -0.02,
         "LFP alone stays near chance; adding body weight lifts accuracy, so "
         "the separable diet information is metabolic (weight), not in the LFP.",
         ha="center", fontsize=9, color="#444444", style="italic")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved figure:\n{OUT_PNG}")

print("\nSTEP 08A2 (weight-control comparison) finished successfully.")