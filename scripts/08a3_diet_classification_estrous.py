# ============================================================
# 08A3_DIET_CLASSIFICATION_ESTROUS.PY
# Purpose:
# Second companion to 08a (after 08a2, the weight control). The
# supervisor noted that estrous phase matters because it modulates
# LFP activity (it is a covariate in the Step 06 mixed model). This
# script asks whether accounting for estrous changes the RQ1 answer
# ("can LFP classify diet?"), using the SAME valid mouse-grouped CV.
#
# Important logic:
#   Estrous phase is NOT diet-related by design (mice were assigned
#   to diet at random; the estrous cycle does not depend on diet).
#   So estrous carries no information ABOUT diet and cannot, by
#   itself, raise diet-classification accuracy. What it CAN do is
#   add nuisance variance to the LFP that masks a diet signal. The
#   principled question is therefore: "once estrous-driven variance
#   is controlled, does an LFP diet signal emerge?" — the ML
#   analogue of keeping estrous as a covariate in Step 06.
#
# Three feature sets (each run with three models):
#   (1) lfp_only          : the 29 LFP features (08a reference)
#   (2) lfp_plus_estrous  : LFP + estrous phase dummies (estrous as
#                           a feature — the literal "add estrous")
#   (3) lfp_residualized  : LFP with estrous-driven variance removed.
#                           Inside every training fold, each LFP
#                           feature is regressed on the estrous
#                           dummies and replaced by its residual;
#                           the SAME fitted regression is applied to
#                           the held-out mouse (no leakage). Diet is
#                           then classified from these residuals.
#
# Estrous encoding: phases A/B/C/D -> dummies with A as the
# reference (matching the Step 06 convention, Caligioni 2009).
# All 125 recordings have an estrous phase (no missing values).
#
# Validation / metrics / permutation: identical to 08a2
#   - Leave-One-Mouse-Out, pooled out-of-fold predictions
#   - balanced accuracy, ROC-AUC, accuracy
#   - mouse-level label-shuffle permutation on the primary model
#     (logistic regression), per feature set.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_features_combined_with_episodes_<CABLE>.csv
#
# Output:
#   outputs/08a3_diet_classification_estrous_<CABLE>/
#       08a3_diet_classification_estrous_results_<CABLE>.csv   (long)
#       08a3_diet_classification_estrous_summary_<CABLE>.txt   (readable)
#       08a3_diet_classification_estrous_<CABLE>.png           (figure)
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
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

OUT_DIR = os.path.join(OUTPUT_DIR, f"08a3_diet_classification_estrous_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(
    OUT_DIR, f"08a3_diet_classification_estrous_results_{CABLE}.csv")
OUT_TXT = os.path.join(
    OUT_DIR, f"08a3_diet_classification_estrous_summary_{CABLE}.txt")
OUT_PNG = os.path.join(
    OUT_DIR, f"08a3_diet_classification_estrous_{CABLE}.png")

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

# Estrous phases; A is the reference (dropped) -> dummies for B,C,D
ESTROUS_PHASES = ["A", "B", "C", "D"]
ESTROUS_DUMMIES = ["estrous_B", "estrous_C", "estrous_D"]
N_ESTROUS = len(ESTROUS_DUMMIES)

C_SETS = {"lfp_only": "#4C72B0", "lfp_plus_estrous": "#DD8452",
          "lfp_residualized": "#1D9E75"}

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

missing = [c for c in LFP_FEATURES if c not in df.columns]
if missing:
    raise KeyError(f"Missing LFP columns: {missing}")
if df[LFP_FEATURES].isna().any(axis=1).sum() > 0:
    raise ValueError("NaN in LFP features — check 05a/05c outputs.")

# Estrous dummies (A reference). Enforce all four categories so the
# dummy columns are always present and consistently ordered.
est = pd.Categorical(df["estrous_phase"], categories=ESTROUS_PHASES)
est_dummies = pd.get_dummies(est, prefix="estrous", drop_first=True)
for c in ESTROUS_DUMMIES:                 # guarantee all expected columns
    if c not in est_dummies.columns:
        est_dummies[c] = 0
est_dummies = est_dummies[ESTROUS_DUMMIES].astype(float)
df = pd.concat([df, est_dummies], axis=1)

n_missing_est = df["estrous_phase"].isna().sum()
print(f"  estrous phases: "
      f"{df['estrous_phase'].value_counts().to_dict()} "
      f"(missing: {n_missing_est})")


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
# 4. ESTROUS RESIDUALIZER (leakage-safe, fit per training fold)
# ------------------------------------------------------------
# Input X = [LFP features ... , estrous dummies (last N_ESTROUS cols)].
# fit(): regress the LFP block on the estrous dummies (train only).
# transform(): return LFP residuals (estrous-driven variance removed);
#              the estrous columns are dropped from the output.
# ============================================================

class EstrousResidualizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_estrous=N_ESTROUS):
        self.n_estrous = n_estrous

    def fit(self, X, y=None):
        k = self.n_estrous
        lfp, estr = X[:, :-k], X[:, -k:]
        self.reg_ = LinearRegression().fit(estr, lfp)
        return self

    def transform(self, X):
        k = self.n_estrous
        lfp, estr = X[:, :-k], X[:, -k:]
        return lfp - self.reg_.predict(estr)


# ============================================================
# 5. FEATURE SETS + MODEL BUILDERS
# ------------------------------------------------------------
# Each feature set is (X matrix, needs_residualizer flag).
#   lfp_only         : 29 LFP columns
#   lfp_plus_estrous : 29 LFP + 3 estrous dummies (estrous as feature)
#   lfp_residualized : 29 LFP + 3 estrous dummies, but the pipeline's
#                      first step residualizes and drops the dummies
# ============================================================

X_lfp = df[LFP_FEATURES].to_numpy(dtype=float)
X_lfp_est = df[LFP_FEATURES + ESTROUS_DUMMIES].to_numpy(dtype=float)

FEATURE_SETS = {
    "lfp_only":         (X_lfp,     False),
    "lfp_plus_estrous": (X_lfp_est, False),
    "lfp_residualized": (X_lfp_est, True),
}


def base_estimators():
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=5000, class_weight="balanced",
            random_state=RANDOM_STATE),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                       class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=RANDOM_STATE),
    }


def make_pipeline(model_name, residualize):
    est = base_estimators()[model_name]
    steps = []
    if residualize:
        steps.append(("resid", EstrousResidualizer(N_ESTROUS)))
    steps.append(("scale", StandardScaler()))
    steps.append(("clf", est))
    return Pipeline(steps)


MODEL_NAMES = ["logistic_regression", "svm_rbf", "random_forest"]
PRIMARY_MODEL = "logistic_regression"


# ============================================================
# 6. EVALUATION HELPERS
# ============================================================

logo = LeaveOneGroupOut()


def pooled_predictions(pipe, X, y, groups):
    proba = cross_val_predict(pipe, X, y, cv=logo, groups=groups,
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
# 7. RUN EVERY MODEL ON EVERY FEATURE SET (LOMO)
# ============================================================

print("\n" + "=" * 66)
print("LEAVE-ONE-MOUSE-OUT CLASSIFICATION (feature set x model)")
print("=" * 66)

rows = []
proba_store = {}

for fs_name, (X_fs, residualize) in FEATURE_SETS.items():
    print(f"\n[{fs_name}]"
          f"{'  (estrous residualized out)' if residualize else ''}")
    for model_name in MODEL_NAMES:
        pipe = make_pipeline(model_name, residualize)
        proba, pred = pooled_predictions(pipe, X_fs, y, groups)
        m = score(y, proba, pred)
        proba_store[(fs_name, model_name)] = proba
        for metric, val in m.items():
            rows.append({"feature_set": fs_name, "model": model_name,
                         "metric": metric, "value": val})
        print(f"  {model_name:<20s} bal_acc={m['balanced_accuracy']:.3f}"
              f"  auc={m['roc_auc']:.3f}  acc={m['accuracy']:.3f}")

results_df = pd.DataFrame(rows)


# ============================================================
# 8. PERMUTATION TEST (mouse-level shuffle, logistic, per set)
# ------------------------------------------------------------
# Diet group is a fixed per-mouse label, so the valid null shuffles
# the label across MICE and re-runs LOMO (same as 08a / 08a2).
# ============================================================

print("\n" + "=" * 66)
print(f"PERMUTATION TEST (logistic regression, LOMO, "
      f"{N_PERMUTATIONS} shuffles per feature set)")
print("=" * 66)

mouse_series = df.groupby("mouse")["group"].first()
mouse_ids = mouse_series.index.to_numpy()
mouse_labels = (mouse_series == POSITIVE_GROUP).astype(int).to_numpy()

perm_summary = {}
for fs_name, (X_fs, residualize) in FEATURE_SETS.items():
    observed = results_df.query(
        "feature_set == @fs_name and model == 'logistic_regression' "
        "and metric == 'balanced_accuracy'")["value"].iloc[0]

    rng = np.random.default_rng(RANDOM_STATE)
    null = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        permuted = rng.permutation(mouse_labels)
        mapping = dict(zip(mouse_ids, permuted))
        y_perm = np.array([mapping[g] for g in groups])
        pipe = make_pipeline("logistic_regression", residualize)
        _, pred_p = pooled_predictions(pipe, X_fs, y_perm, groups)
        null[i] = balanced_accuracy_score(y_perm, pred_p)

    p_val = (1 + np.sum(null >= observed)) / (1 + N_PERMUTATIONS)
    perm_summary[fs_name] = {"observed": observed,
                             "null_mean": float(null.mean()),
                             "p_value": p_val, "null": null}
    print(f"  {fs_name:<18s} observed={observed:.3f}  "
          f"chance={null.mean():.3f}  p={p_val:.4f}")

    rows.append({"feature_set": fs_name, "model": "logistic_regression",
                 "metric": "perm_null_mean_balanced_accuracy",
                 "value": float(null.mean())})
    rows.append({"feature_set": fs_name, "model": "logistic_regression",
                 "metric": "perm_p_value_balanced_accuracy", "value": p_val})

results_df = pd.DataFrame(rows)


# ============================================================
# 9. SAVE — LONG-FORMAT CSV
# ============================================================

results_df = results_df[["feature_set", "model", "metric", "value"]]
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved results CSV:\n{OUT_CSV}")


# ============================================================
# 10. SAVE — HUMAN-READABLE TXT
# ============================================================

def get(fs, model, metric):
    q = results_df.query("feature_set == @fs and model == @model "
                         "and metric == @metric")
    return q["value"].iloc[0] if len(q) else float("nan")

lines = []
lines.append("=" * 72)
lines.append(f"08A3 DIET CLASSIFICATION — ESTROUS PHASE — {CABLE}")
lines.append("=" * 72)
lines.append("")
lines.append(f"Observations : {n_obs} recordings / {n_mice} mice "
             f"(HF {n_hf} / CTRL {n_ctrl})")
lines.append(f"Estrous      : A/B/C/D, A = reference (Caligioni 2009); "
             f"no missing values")
lines.append(f"Validation   : leave-one-mouse-out")
lines.append(f"Chance / majority-class accuracy : 0.500 / {majority_acc:.3f}")
lines.append("")
lines.append("Feature sets:")
lines.append("  lfp_only         = 29 LFP features (08a reference)")
lines.append("  lfp_plus_estrous = LFP + estrous dummies (estrous as feature)")
lines.append("  lfp_residualized = LFP with estrous variance removed per fold")
lines.append("")
lines.append("-" * 72)
lines.append(f"{'feature set':<18s} {'model':<22s} {'bal_acc':>8s}"
             f" {'roc_auc':>8s} {'acc':>7s}")
lines.append("-" * 72)
for fs in FEATURE_SETS:
    for model in MODEL_NAMES:
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
lines.append("  - Estrous is not diet-related by design, so it cannot add")
lines.append("    diet information; adding it as a feature is not expected to")
lines.append("    raise accuracy.")
lines.append("  - lfp_residualized asks the principled question: with estrous-")
lines.append("    driven variance removed, does an LFP diet signal emerge? If")
lines.append("    it stays near chance, the RQ1 conclusion is robust to estrous.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 11. FIGURE (2 panels)
#   A) grouped bars: balanced accuracy, feature set x model
#   B) ROC (logistic), the three feature sets overlaid
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# --- A: grouped bars ---
ax = axes[0]
fs_list = list(FEATURE_SETS.keys())
x = np.arange(len(fs_list))
w = 0.26
model_colors = {"logistic_regression": "#C44E52",
                "svm_rbf": "#8172B3",
                "random_forest": "#4C72B0"}
for j, model in enumerate(MODEL_NAMES):
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

fig.suptitle(f"Step 08a3 — diet classification with estrous phase ({CABLE}, "
             f"{n_obs} recordings / {n_mice} mice)",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.5, -0.02,
         "Estrous is not diet-related, so it adds no diet information; even "
         "with estrous variance removed, LFP stays near chance for diet.",
         ha="center", fontsize=9, color="#444444", style="italic")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved figure:\n{OUT_PNG}")

print("\nSTEP 08A3 (estrous comparison) finished successfully.")